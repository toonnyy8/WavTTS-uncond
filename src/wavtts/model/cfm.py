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
        mask_run_len: float | None = None,
        student_layer_frac: float = 0.3,
        teacher_layer_frac: float = 0.7,
        rep_proj_hidden: int | None = None,
    ):
        super().__init__()

        # waveform geometry
        waveform_kwargs = dict(waveform_kwargs)
        self.wav_frame_len = int(waveform_kwargs.pop("wav_frame_len", 160))
        self.num_channels = self.wav_frame_len

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
        # mask_run_len replaces the block draw with a two-state chain: run lengths are
        # geometric, so the mean masked run is a knob of its own instead of being pinned
        # to 1/(1-mask_ratio) by the i.i.d. draw. The unmasked mean follows from the ratio,
        # which keeps mask_ratio the primary knob (it is what the paper sweeps, and what
        # "preserves the per-token marginal" refers to).
        self.mask_run_len = self.unmask_run_len = None
        if mask_run_len is not None:
            if self.mask_block != 1:
                raise ValueError("mask_run_len and mask_block are two ways to set the same "
                                 f"thing; got mask_run_len={mask_run_len}, mask_block={self.mask_block}. "
                                 f"mask_block={self.mask_block} is mask_run_len="
                                 f"{self.mask_block / (1 - mask_ratio):.4g} at this ratio.")
            if not 0.0 < mask_ratio < 1.0:
                raise ValueError(f"mask_run_len needs 0 < mask_ratio < 1, got {mask_ratio}")
            self.mask_run_len = float(mask_run_len)
            self.unmask_run_len = self.mask_run_len * (1 - mask_ratio) / mask_ratio
            # a run is at least one token, so a mean below 1 is unreachable
            if self.mask_run_len < 1.0 or self.unmask_run_len < 1.0:
                raise ValueError(
                    f"mask_run_len={self.mask_run_len:.4g} at mask_ratio={mask_ratio} implies an "
                    f"unmasked run of {self.unmask_run_len:.4g} tokens; both means must be >= 1"
                )
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
            pred = self.transformer(x=x, time=t)
            if self.prediction == "flow":
                return pred
            return self._x_to_v(pred, x, t)

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

    def _mask_tokens(self, batch: int, n_tok: int, device) -> bool["b n"]:
        """Which tokens take the second timestep. Marginally mask_ratio either way; the
        two draws differ in how long a masked stretch runs.

        The paper draws the mask i.i.d. per token, but its audio tokens are 40 ms Songbloom
        latents while ours are 10 ms raw-waveform frames, so the same draw alternates 4x
        faster in time: a masked run of 20 ms against the paper's 80 ms. That matters
        because the whole mechanism is "your neighbours are cleaner, so local denoising
        cannot recover you", and 20 ms spans only 2-4 pitch periods, which interpolation
        handles. Under an i.i.d. draw the run length is not free -- it is 1/(1-mask_ratio)
        -- so the time scale needs a second knob. There are two:

        mask_block   draw per block of k tokens. Mean masked run k/(1-mask_ratio). Cheap,
                     but boundaries land on a fixed k-grid and no run is shorter than k.
        mask_run_len draw the run lengths instead, from a two-state chain. Both runs are
                     geometric -- which is what the i.i.d. draw already produces, just with
                     the mean pinned -- so this is the same family with the scale freed.
                     Continuous, no grid, and mask_run_len=1/(1-mask_ratio) reproduces the
                     i.i.d. draw exactly.
        """
        if self.mask_run_len is None:
            n_blk = math.ceil(n_tok / self.mask_block)
            m = torch.rand((batch, n_blk), device=device) < self.mask_ratio
            if self.mask_block > 1:
                m = m.repeat_interleave(self.mask_block, dim=-1)[:, :n_tok]
            return m

        # Draw alternating run lengths and lay them end to end. Sampling the start state
        # from the stationary law (rather than a coin flip) is what makes the marginal come
        # out at exactly mask_ratio: the geometric is memoryless, so a fresh run at index 0
        # is already in equilibrium and no length-biasing correction is needed.
        start = (torch.rand((batch, 1), device=device) < self.mask_ratio).long()
        # ponytail: runs are >= 1 token, so n runs always cover n tokens -- no coverage math
        state = (torch.arange(n_tok, device=device) + start) % 2  # b n, 1 = masked
        p = torch.where(state.bool(), 1.0 / self.mask_run_len, 1.0 / self.unmask_run_len)
        # torch.rand can return exactly 0, and log(0) -> -inf -> a garbage index below
        u = torch.rand((batch, n_tok), device=device).clamp_min(torch.finfo(torch.float32).tiny)
        run_len = (u.log() / torch.log1p(-p)).floor().long() + 1  # geometric on {1, 2, ...}
        # run boundaries are strictly increasing, so they never collide in the scatter;
        # anything past the end lands in the throwaway column n_tok
        bounds = run_len.cumsum(-1).clamp_(0, n_tok)
        edges = torch.zeros((batch, n_tok + 1), dtype=torch.long, device=device)
        edges.scatter_(1, bounds, 1)
        return ((edges[:, :n_tok].cumsum(-1) + start) % 2).bool()

    def _dual_timestep(self, batch, n_tok, seq_len, dtype, device):
        """Dual-Timestep Scheduling: a per-token noise level tau, plus the scalar the
        teacher sees. Two timesteps t, s are drawn from the usual distribution and a random
        mask of ratio mask_ratio decides which tokens take s; the rest take t. The teacher
        sees whichever of the two is *cleaner* applied uniformly -- here t=1 is data and
        t=0 is noise, so that is the larger one (the paper's convention is inverted, and
        its tau_min is this tau_clean).

        See _mask_tokens for how the mask itself is drawn.
        """
        t = self._sample_time(batch, dtype=dtype, device=device)
        s = self._sample_time(batch, dtype=dtype, device=device)

        m = self._mask_tokens(batch, n_tok, device)
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

        x1 = inp * self.latents_scale

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
                x=φ, time=time, mask=mask, lens=lens, rope=rope, hidden_at=self.student_layer
            )
            with torch.no_grad():
                φ_clean = (1 - time_clean.unsqueeze(-1)) * x0 + time_clean.unsqueeze(-1) * x1
                h_teacher = teacher(
                    x=φ_clean,
                    time=time_clean,
                    mask=mask,
                    lens=lens,
                    rope=rope,
                    hidden_at=self.teacher_layer,
                    hidden_only=True,
                )
        else:
            raw_pred = self.transformer(x=φ, time=time, mask=mask, lens=lens)

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
