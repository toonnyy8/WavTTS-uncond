"""
ein notation:
b - batch
n - sequence
nw - raw wave length
d - dimension
"""
# ruff: noqa: F722 F821

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torchdiffeq import odeint

from wavtts.model.backbones.dit import STATE_CLEAN, STATE_NULL
from wavtts.model.modules import MelSpectrogramLoss
from wavtts.model.utils import exists, get_epss_timesteps, lens_to_mask


class CFM(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        sigma=0.0,
        odeint_kwargs: dict = dict(
            method="euler"  # 'midpoint'
        ),
        p_mix: float = 0.5,
        p_concat: float = 0.5,
        mix_lambda_range: tuple[float, float] = (0.3, 0.7),
        concat_point_range: tuple[float, float] = (0.3, 0.7),
        concat_xfade_ms: float = 20.0,
        state_null_prob: float = 0.5,
        waveform_kwargs: dict = dict(),
        prediction: str = "flow",  # "flow" | "x_pred"
        loss_space: str = "flow",  # "flow" | "v" | "x"
        t_sampling: str = "uniform",  # "uniform" | "logistic_normal"
        P_mean: float = 0.0,
        P_std: float = 1.0,
        time_shift: float = 1.0,
        t_eps: float = 1e-4,
        use_aux_mel_loss: bool = False,
        aux_mel_loss_weight: float = 0.0,
        aux_mel_loss_masked: bool = True,
        sample_rate: int = 16000,
        latents_scale: float = 1.0,
        self_flow: bool = False,
        rep_loss_weight: float = 0.8,
        mask_ratio: float = 0.5,
        mask_block: int = 1,
        student_layer_frac: float = 0.3,
        teacher_layer_frac: float = 0.7,
        rep_proj_hidden: int | None = None,
    ):
        super().__init__()

        # waveform geometry
        waveform_kwargs = dict(waveform_kwargs)
        self.wav_frame_len = int(waveform_kwargs.pop("wav_frame_len", 160))
        self.num_channels = self.wav_frame_len

        # no-leaky mixing augmentation / state conditioning
        self.p_mix = p_mix
        self.p_concat = p_concat
        self.mix_lambda_range = tuple(mix_lambda_range)
        self.concat_point_range = tuple(concat_point_range)
        self.concat_xfade_ms = concat_xfade_ms
        self.state_null_prob = state_null_prob

        # transformer
        self.transformer = transformer
        self.dim = transformer.dim

        if hasattr(self.transformer, "set_wav_frame_len"):
            self.transformer.set_wav_frame_len(self.wav_frame_len)

        # conditional flow related
        self.sigma = sigma

        # sampling related
        self.odeint_kwargs = odeint_kwargs

        # enhanced flow model settings
        self.prediction = prediction
        self.loss_space = loss_space
        self.t_sampling = t_sampling
        self.P_mean = P_mean
        self.P_std = P_std
        self.time_shift = time_shift
        self.t_eps = t_eps
        self.latents_scale = latents_scale
        self.target_sample_rate = sample_rate
        # aux mel loss
        self.use_aux_mel_loss = use_aux_mel_loss
        self.aux_mel_loss_masked = aux_mel_loss_masked
        if self.use_aux_mel_loss:
            self.aux_mel_loss = MelSpectrogramLoss(
                sample_rate=sample_rate,
                n_mels=[5, 10, 20, 40, 80, 160, 320],
                window_lengths=[32, 64, 128, 256, 512, 1024, 2048],
                mel_fmin=[0, 0, 0, 0, 0, 0, 0],
                mel_fmax=[None] * 7,
                pow=1.0,
                clamp_eps=1e-5,
                weight=aux_mel_loss_weight,
            )
        else:
            self.aux_mel_loss = None

        # Self-Flow (Chefer, Esser et al. 2026). Dual-Timestep Scheduling noises a random
        # half of the tokens at a second timestep, and the student is asked to predict --
        # from that partially corrupted view -- the features an EMA teacher produces from
        # the uniformly *cleaner* one. The asymmetry is what forces global structure: local
        # denoising alone cannot recover a token whose neighbours are the cleaner ones.
        self.self_flow = self_flow
        self.rep_loss_weight = rep_loss_weight
        self.mask_ratio = mask_ratio
        self.mask_block = int(mask_block)
        self.rep_proj = None
        if self_flow:
            depth = transformer.depth
            self.student_layer = max(1, round(student_layer_frac * depth))
            self.teacher_layer = max(1, round(teacher_layer_frac * depth))
            if self.student_layer >= self.teacher_layer:
                raise ValueError(
                    f"self-flow needs student layer < teacher layer, got "
                    f"{self.student_layer} >= {self.teacher_layer} at depth {depth}"
                )
            # the paper's "lightweight projection head, ~10M parameters" at their dim=1152
            hidden = rep_proj_hidden if rep_proj_hidden is not None else 2 * self.dim
            self.rep_proj = nn.Sequential(
                nn.Linear(self.dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Linear(hidden, self.dim),
            )

    @property
    def device(self):
        return next(self.parameters()).device

    def _sample_time(self, batch: int, dtype, device):
        if self.t_sampling == "uniform":
            t = torch.rand((batch,), dtype=dtype, device=device)
        elif self.t_sampling == "logistic_normal":
            # JiT: t = sigmoid(N(P_mean, P_std))
            z = torch.randn((batch,), device=device, dtype=dtype) * self.P_std + self.P_mean
            t = torch.sigmoid(z)
        else:
            raise ValueError(f"Unknown t_sampling: {self.t_sampling}")

        if self.time_shift != 1.0:
            t = t / (t + self.time_shift * (1 - t))

        return t

    def _x_to_v(self, x_pred, z, t):
        # v_pred = (x_pred - z) / (1 - t)
        denom = (1.0 - t).clamp_min(self.t_eps)
        while denom.ndim < z.ndim:
            denom = denom.unsqueeze(-1)
        return (x_pred - z) / denom

    def _mix_augment(self, x1: float["b nw"], lens: int["b"], mix_flags: bool["b"]):
        """No-leaky mixing augmentation, applied to the flagged samples.

        overlap: equal-power blend with a batch-roll partner (simultaneous speakers)
        concat:  equal-power crossfade into the partner at a random switch point
                 (temporal speaker switch)

        Mixed samples carry no state of their own — the caller has already labelled
        them null, so the speaker-inconsistent direction lives inside the
        unconditional distribution rather than beside it.
        """
        batch, seq_len = x1.shape
        device = x1.device
        if batch < 2 or not mix_flags.any():
            return x1  # a batch of one would roll onto itself

        partner = x1.roll(1, dims=0)
        concat_flags = torch.rand(batch, device=device) < self.p_concat

        # overlap: x = sqrt(1-lam)*x1 + sqrt(lam)*partner
        lo, hi = self.mix_lambda_range
        lam = torch.empty((batch, 1), device=device, dtype=x1.dtype).uniform_(lo, hi)
        overlap = torch.sqrt(1.0 - lam) * x1 + torch.sqrt(lam) * partner

        # concat: switch to partner at s with an equal-power crossfade
        # (a hard cut's click would let the model detect "mixed" from the boundary
        #  artifact instead of speaker identity, breaking the CFG direction)
        xfade_len = max(1, int(self.concat_xfade_ms * self.target_sample_rate / 1000.0))
        plo, phi = self.concat_point_range
        u = torch.empty((batch,), device=device, dtype=x1.dtype).uniform_(plo, phi)
        s = (u * lens.to(x1.dtype)).long().clamp(min=1, max=seq_len - 1)
        idx = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, n]
        prog = ((idx - s.unsqueeze(-1)).to(x1.dtype) / xfade_len).clamp(0.0, 1.0)
        g_in = torch.sin(prog * math.pi / 2)  # partner fades in
        g_out = torch.cos(prog * math.pi / 2)  # original fades out; g_in^2 + g_out^2 = 1
        concat = g_out * x1 + g_in * partner

        mixed = torch.where(concat_flags.unsqueeze(-1), concat, overlap)
        # ponytail: batch-roll partner; padding tails dilute the mixed content slightly,
        # switch to dataset-level pair loading if purity ever matters.
        return torch.where(mix_flags.unsqueeze(-1), mixed, x1)

    def _dpmpp_2m(self, fn, y0, t):
        """DPM-Solver++(2M) multistep, data-prediction form, for the rectified-flow
        interpolant x_t = (1-t)·x0 + t·x1 (alpha_t = t, sigma_t = 1-t,
        lambda_t = log(t / (1-t)), endpoints clamped by t_eps). fn returns v;
        x_pred = x + (1-t)·v reuses the CFG combination unchanged."""
        eps = self.t_eps

        def lam(s: float) -> float:
            s = min(max(s, eps), 1.0 - eps)
            return math.log(s / (1.0 - s))

        x = y0
        states = [x]
        x_pred_prev, h_prev = None, None
        for i in range(len(t) - 1):
            t_cur, t_next = float(t[i]), float(t[i + 1])
            v = fn(t[i], x)
            x_pred = x + (1.0 - t_cur) * v
            if t_next >= 1.0 - eps:  # final step lands on the data prediction
                x = x_pred
            else:
                h = lam(t_next) - lam(t_cur)
                if x_pred_prev is None:
                    d = x_pred  # first step: first-order (DPM-Solver++(1))
                else:
                    r = h_prev / h
                    d = (1.0 + 1.0 / (2.0 * r)) * x_pred - (1.0 / (2.0 * r)) * x_pred_prev
                x = ((1.0 - t_next) / (1.0 - t_cur)) * x - t_next * math.expm1(-h) * d
                x_pred_prev, h_prev = x_pred, h
            states.append(x)
        return torch.stack(states)

    @torch.no_grad()
    def sample(
        self,
        duration: int,  # number of samples at target_sample_rate
        *,
        batch: int = 1,
        steps: int = 32,
        cfg_strength: float = 2.0,
        sway_sampling_coef: float | None = None,
        timestep_mapping: str = "sway_sampling",
        timestep_power: float | None = None,
        shift: float = 1.0,
        use_epss: bool = True,
        seed: int | None = None,
        solver: str = "euler",  # "euler" | "dpmpp"
    ):
        self.eval()
        device = self.device
        dtype = next(self.parameters()).dtype

        if solver not in ("euler", "dpmpp"):
            raise ValueError(f"Unknown solver: {solver}")
        state = torch.full((batch,), STATE_CLEAN, device=device, dtype=torch.long)

        requested = int(duration)
        aligned = int(math.ceil(requested / self.wav_frame_len) * self.wav_frame_len)

        # dedicated generator: sampling with a fixed seed must not perturb the
        # global RNG (e.g. mid-training checkpoint sampling)
        generator = None
        if exists(seed):
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))
        y0 = torch.randn(batch, aligned, device=device, dtype=dtype, generator=generator)

        def fn(t, x):
            def to_v(pred):
                if self.prediction == "flow":
                    return pred
                return self._x_to_v(pred, x, t)

            # state_null_prob == 0 means the null branch was never trained, so its
            # prediction is noise: guidance is off no matter what the caller asked for
            if cfg_strength < 1e-5 or self.state_null_prob <= 0:
                pred = self.transformer(x=x, state=state, time=t)
                return to_v(pred)

            # classifier-free guidance against the null branch, which carries the
            # mixing augmentation: the guidance term is a classifier gradient and
            # self-extinguishes once x is unambiguously clean
            pred_cfg = self.transformer(x=x, state=state, time=t, cfg_infer=True)
            pred, neg_pred = torch.chunk(pred_cfg, 2, dim=0)
            v_pos = to_v(pred)
            v_neg = to_v(neg_pred)
            return v_pos + (v_pos - v_neg) * cfg_strength

        use_epss = use_epss and timestep_mapping == "sway_sampling"
        if use_epss:  # use Empirically Pruned Step Sampling for low NFE
            t = get_epss_timesteps(steps, device=device, dtype=torch.float32)
        else:
            t = torch.linspace(0, 1, steps + 1, device=device, dtype=torch.float32)

        if timestep_mapping == "uniform":
            pass
        elif timestep_mapping == "sway_sampling":
            if sway_sampling_coef is not None:
                t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)
        elif timestep_mapping == "power":
            if timestep_power is None:
                raise ValueError("timestep_power must be provided when timestep_mapping='power'")
            t = t.pow(timestep_power)
        else:
            raise ValueError(f"Unknown timestep_mapping: {timestep_mapping}")

        if shift != 1.0:
            t = t / (t + shift * (1 - t))

        if solver == "dpmpp":
            trajectory = self._dpmpp_2m(fn, y0, t)
        else:
            # WavTTS default inference uses Euler ODE sampling.
            odeint_kwargs = dict(self.odeint_kwargs)
            odeint_kwargs["method"] = "euler"
            trajectory = odeint(fn, y0, t, **odeint_kwargs)

        out = trajectory[-1] / self.latents_scale
        return out[:, :requested], trajectory

    def _dual_timestep(self, batch, n_tok, seq_len, dtype, device):
        """Dual-Timestep Scheduling: a per-token noise level tau, plus the scalar the
        teacher sees. Two timesteps t, s are drawn from the usual distribution and a random
        mask of ratio mask_ratio decides which tokens take s; the rest take t. The teacher
        sees whichever of the two is *cleaner* applied uniformly -- here t=1 is data and
        t=0 is noise, so that is the larger one (the paper's convention is inverted, and
        its tau_min is this tau_clean).

        The mask is drawn over blocks of mask_block tokens rather than per token. The paper
        draws it i.i.d. per token, but its audio tokens are 40 ms Songbloom latents while
        ours are 10 ms raw-waveform frames, so the same i.i.d. draw gives a noise pattern
        that alternates 4x faster in time: a mean run of 20 ms against the paper's 80 ms.
        That matters because the whole mechanism is "your neighbours are cleaner, so local
        denoising cannot recover you" -- and 20 ms spans only 2-4 pitch periods, which
        interpolation handles. Blocking restores the paper's time scale. Note that ratio and
        block size are coupled under an i.i.d. draw (a masked run averages 1/(1-mask_ratio)
        tokens), so raising mask_ratio cannot buy the same thing without also changing how
        much is masked; only blocking separates the two.
        """
        t = self._sample_time(batch, dtype=dtype, device=device)
        s = self._sample_time(batch, dtype=dtype, device=device)

        n_blk = math.ceil(n_tok / self.mask_block)
        m = torch.rand((batch, n_blk), device=device) < self.mask_ratio
        if self.mask_block > 1:
            m = m.repeat_interleave(self.mask_block, dim=-1)[:, :n_tok]
        tau_tok = torch.where(m, s.unsqueeze(-1), t.unsqueeze(-1))  # b n
        tau_clean = torch.maximum(t, s)  # b

        # tokens are wav_frame_len samples wide; spread each token's level over its samples
        tau_wav = tau_tok.repeat_interleave(self.wav_frame_len, dim=-1)[:, :seq_len]
        return tau_tok, tau_wav, tau_clean

    def forward(
        self,
        inp: float["b nw"],  # raw waveform
        *,
        lens: int["b"] | None = None,
        teacher: nn.Module | None = None,  # EMA copy of self.transformer, for self-flow
    ):
        # handle raw waveform
        if inp.ndim != 2:
            raise ValueError(f"WavTTS expects raw waveform input [B, N], got {tuple(inp.shape)}")

        batch, seq_len, dtype, device = *inp.shape[:2], inp.dtype, self.device

        # lens and mask
        if not exists(lens):  # if lens not acquired by trainer from collate_fn
            lens = torch.full((batch,), seq_len, device=device, dtype=torch.long)
        mask = lens_to_mask(lens, length=seq_len)

        # The label comes first, the augmentation follows it: a sample is null with
        # probability state_null_prob, and only null samples may be mixed. So the clean
        # branch is pure single-speaker speech, while
        #     p_null = (1 - p_mix) * p_clean + p_mix * p_mixed
        # and the CFG term (v_clean - v_null) points away from speaker inconsistency
        # AND saturates to zero once x is unambiguously clean -- p_mixed(x) vanishes
        # there faster than any mixing weight, for any p_mix < 1. Giving mixed its own
        # state buys the first property and loses the second: that term pushes harder
        # the further x gets from the mixed manifold.
        null_flags = torch.rand(batch, device=device) < self.state_null_prob
        mix_flags = null_flags & (torch.rand(batch, device=device) < self.p_mix)
        x1 = self._mix_augment(inp, lens, mix_flags)
        state = torch.where(
            null_flags,
            torch.full((batch,), STATE_NULL, device=device, dtype=torch.long),
            torch.full((batch,), STATE_CLEAN, device=device, dtype=torch.long),
        )

        x1 = x1 * self.latents_scale

        # x0 is gaussian noise
        x0 = torch.randn_like(x1)

        # loud, not silent: a caller that forgets the teacher would otherwise train a
        # plain flow model for days under a self-flow config
        if self.self_flow and teacher is None:
            raise ValueError("self_flow=True requires a teacher module in forward(teacher=...)")
        run_self_flow = self.self_flow
        n_tok = math.ceil(seq_len / self.wav_frame_len)

        if run_self_flow:
            # per-token noise levels; t broadcasts over the token's samples already
            time, t, time_clean = self._dual_timestep(batch, n_tok, seq_len, dtype, device)
        else:
            time = self._sample_time(batch, dtype=dtype, device=device)
            t = time.unsqueeze(-1)

        # sample xt (phi_t(x) in the paper)
        φ = (1 - t) * x0 + t * x1
        flow = x1 - x0

        if run_self_flow:
            # both passes must see the same randomized positions, or the features the
            # student is asked to match were computed over a different geometry
            rope = self.transformer.make_rope(batch, n_tok, device)
            raw_pred, h_student = self.transformer(
                x=φ, state=state, time=time, mask=mask, lens=lens, rope=rope, hidden_at=self.student_layer
            )
            with torch.no_grad():
                φ_clean = (1 - time_clean.unsqueeze(-1)) * x0 + time_clean.unsqueeze(-1) * x1
                h_teacher = teacher(
                    x=φ_clean,
                    state=state,
                    time=time_clean,
                    mask=mask,
                    lens=lens,
                    rope=rope,
                    hidden_at=self.teacher_layer,
                    hidden_only=True,
                )
        else:
            raw_pred = self.transformer(x=φ, state=state, time=time, mask=mask, lens=lens)

        # interpret prediction
        if self.prediction == "flow":
            v_pred = raw_pred
            x_pred = φ + (1.0 - t) * v_pred
        elif self.prediction == "x_pred":
            x_pred = raw_pred
            v_pred = self._x_to_v(x_pred, φ, t)
        else:
            raise ValueError(f"Unknown prediction: {self.prediction}")

        # loss space
        if self.loss_space == "flow":
            loss = F.mse_loss(v_pred, flow, reduction="none")
        elif self.loss_space == "v":
            # v-loss (same target flow, but v_pred computed from x_pred) & use clamp_min.
            # Built from `t` rather than `time`: under dual-timestep scheduling `time` is
            # per token while `t` is the same levels at sample resolution.
            denom = (1.0 - t).clamp_min(self.t_eps)
            while denom.ndim < φ.ndim:
                denom = denom.unsqueeze(-1)
            target = (x1 - φ) / denom
            loss = F.mse_loss(v_pred, target, reduction="none")
        elif self.loss_space == "x":
            loss = F.mse_loss(x_pred, x1, reduction="none")
        else:
            raise ValueError(f"Unknown loss_space: {self.loss_space}")

        loss = loss[mask]
        flow_loss = loss.mean()
        total_loss = flow_loss

        rep_loss = torch.tensor(0.0, device=device)
        if run_self_flow:
            # cosine alignment between the student's shallow features (partial, corrupt
            # view) and the teacher's deep ones (cleaner view), over valid tokens only
            token_mask = lens_to_mask((lens + self.wav_frame_len - 1) // self.wav_frame_len, length=h_student.shape[1])
            sim = F.cosine_similarity(self.rep_proj(h_student), h_teacher.detach(), dim=-1)
            rep_loss = -sim[token_mask].mean()
            total_loss = total_loss + self.rep_loss_weight * rep_loss

        aux_mel_loss = torch.tensor(0.0, device=device)
        if self.use_aux_mel_loss and self.aux_mel_loss is not None:
            x1_flat_unscaled = x1 / self.latents_scale
            x1_pred_flat_unscaled = x_pred / self.latents_scale

            aux_mel_kwargs = {}
            if self.aux_mel_loss_masked:
                aux_mel_kwargs.update(
                    frame_mask=mask,
                    frame_lengths=lens,
                )

            aux_mel_loss = self.aux_mel_loss(
                x1_pred_flat_unscaled,
                x1_flat_unscaled,
                **aux_mel_kwargs,
            )
            total_loss = total_loss + aux_mel_loss

        loss_dict = {
            "total_loss": total_loss,
            "flow_loss": flow_loss,
            "aux_mel_loss": aux_mel_loss,
            "rep_loss": rep_loss,
        }

        return total_loss, loss_dict
