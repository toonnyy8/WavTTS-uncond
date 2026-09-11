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
from wavtts.model.ddo import ddo_loss
from wavtts.model.modules import MelSpectrogramLoss
from wavtts.model.rope import randomized_positions
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

        # DDO (off unless attach_ddo_ref is called). The reference model lives in a plain
        # list so nn.Module never registers it -- see attach_ddo_ref for why that matters.
        self._ddo_ref: list[nn.Module] = []
        self.ddo_alpha = 1.0
        self.ddo_beta = 1.0
        self.ddo_delta_normalize = "mean"
        self.ddo_anchor_weight = 1.0
        self.ddo_beta_mel = 0.0

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
        duration: int | None = None,  # number of samples at target_sample_rate, shared by every row
        *,
        lens: int["b"] | list[int] | None = None,  # per-row sample counts instead of one duration
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

        # Mixed lengths in one batch, the way training already runs them: every row is padded
        # to the longest and carries its own length through `lens` and `mask`, so attention
        # never sees a padded key, the conv position embedding zeroes them, and the entropy
        # scaling counts only real ones. Row i's valid region is then bit-for-bit the clip
        # that generating it alone at lens[i] would produce -- there is no crop of a longer
        # utterance, the model plans lens[i] samples of speech and the padding lane is
        # simply never read. Mixed batches pack a frame budget ~2.3x tighter than equal
        # lengths do on this corpus' length distribution (gen_fake_pool.plan_batches).
        if lens is not None:
            if duration is not None:
                raise ValueError("sample() takes duration or lens, not both")
            lens = torch.as_tensor(lens, device=device, dtype=torch.long)
            if lens.ndim != 1 or lens.numel() == 0 or bool((lens <= 0).any()):
                raise ValueError(f"lens must be a non-empty 1-d tensor of positive sample counts, got {lens}")
            batch = int(lens.numel())
            requested = int(lens.max())
        elif duration is not None:
            requested = int(duration)
        else:
            raise ValueError("sample() needs duration or lens")
        state = torch.full((batch,), STATE_CLEAN, device=device, dtype=torch.long)

        aligned = int(math.ceil(requested / self.wav_frame_len) * self.wav_frame_len)
        mask = lens_to_mask(lens, length=aligned) if lens is not None else None

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
                pred = self.transformer(x=x, state=state, time=t, mask=mask, lens=lens)
                # the integrator's state stays in x's dtype: under a bf16 autocast the
                # backbone emits bf16 and the ODE must not accumulate in it
                return to_v(pred).to(x.dtype)

            # classifier-free guidance against the null branch, which carries the
            # mixing augmentation: the guidance term is a classifier gradient and
            # self-extinguishes once x is unambiguously clean
            pred_cfg = self.transformer(x=x, state=state, time=t, mask=mask, lens=lens, cfg_infer=True)
            pred, neg_pred = torch.chunk(pred_cfg, 2, dim=0)
            v_pos = to_v(pred)
            v_neg = to_v(neg_pred)
            return (v_pos + (v_pos - v_neg) * cfg_strength).to(x.dtype)

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
        # with `lens`, row i is valid up to lens[i]; the caller crops the padding lane
        return out[:, :requested], trajectory

    # ---------------------------------------------------------------- DDO

    def attach_ddo_ref(
        self,
        ref: "CFM",
        *,
        alpha: float,
        beta: float,
        delta_normalize: str = "mean",
        anchor_weight: float = 1.0,
        beta_mel: float = 0.0,
    ) -> None:
        """Freeze `ref` as p_ref and switch forward() onto the DDO path.

        beta_mel > 0 opens a second discriminator channel in the multi-scale log-mel
        domain: Δ_mel = -(mel_θ - mel_ref) per row, and the logit becomes
        β·Δ_v + β_mel·Δ_mel (spec 3.10). Needs the aux mel module (use_aux_mel_loss).

        The reference is stored in a **one-element list**, not as an attribute. A bare
        `self.ddo_ref = ref` would go through `nn.Module.__setattr__`, which registers any
        Module it is handed: the optimizer would start updating p_ref, EMA would average
        it, DDP would all-reduce gradients it does not have, `state_dict()` would double
        in size, and every key in it would stop matching what make_pretrained_init.py and
        sample_uncond.py expect. A list is not a Module, so `__setattr__` falls through to
        plain object assignment and the reference stays invisible to all of them.

        The price is that `.to(device)` and `.half()` do not recurse into it -- the trainer
        moves it explicitly once, after accelerator.prepare (spec S3.9).
        """
        if delta_normalize not in ("mean", "sum"):
            raise ValueError(f"Unknown delta_normalize: {delta_normalize}")

        ref.eval()
        ref.requires_grad_(False)
        if hasattr(ref.transformer, "checkpoint_activations"):
            # ref only ever runs under no_grad, so there is no backward pass to recompute
            # activations for; leaving it on buys nothing and costs a second forward.
            ref.transformer.checkpoint_activations = False

        self._ddo_ref = [ref]
        self.ddo_alpha = float(alpha)
        self.ddo_beta = float(beta)
        self.ddo_delta_normalize = delta_normalize
        self.ddo_anchor_weight = float(anchor_weight)
        self.ddo_beta_mel = float(beta_mel)
        if self.ddo_beta_mel > 0 and self.aux_mel_loss is None:
            raise ValueError("beta_mel > 0 needs use_aux_mel_loss=True: the mel channel is built on that module")
        if self.ddo_beta_mel > 0 and self.ddo_beta <= 0:
            raise ValueError("beta_mel > 0 needs beta > 0: the two channels share one logit, scaled by beta")

    @property
    def ddo_ref(self) -> "CFM | None":
        return self._ddo_ref[0] if self._ddo_ref else None

    def _predictions(self, raw_pred, φ, time):
        """(v_pred, x_pred) from whatever the backbone emits, per self.prediction."""
        t = time.unsqueeze(-1)
        if self.prediction == "flow":
            v_pred = raw_pred
            x_pred = φ + (1.0 - t) * v_pred
        elif self.prediction == "x_pred":
            x_pred = raw_pred
            v_pred = self._x_to_v(x_pred, φ, time)
        else:
            raise ValueError(f"Unknown prediction: {self.prediction}")
        return v_pred, x_pred

    def _elementwise_loss(self, raw_pred, φ, x1, time) -> float["b nw"]:
        """The flow loss per waveform sample, in fp32.

        Everything is cast **before** the squaring, not after. Δ is the difference of two
        nearly identical models' losses; early in a DDO round their relative gap is ~1e-3,
        and bf16's 8 mantissa bits do not survive that subtraction -- Δ degenerates into
        quantization noise, with no error anywhere (spec S3.7). Squaring in bf16 and
        casting the result up would keep the noise, so the cast comes first.
        """
        raw_pred, φ, x1, time = raw_pred.float(), φ.float(), x1.float(), time.float()
        v_pred, x_pred = self._predictions(raw_pred, φ, time)

        if self.loss_space in ("flow", "v"):
            # One branch for two spaces, because for this interpolant they share a target:
            # x1 - x0 == (x1 - φ)/(1 - t) identically, so "flow" needs no x0 to reconstruct
            # it. The only difference from the pooled path is the clamped denominator that
            # "flow" does without, and here the clamp is the safer choice: Δ subtracts two
            # losses computed with the same clamp, while an unclamped row with t within
            # t_eps of 1 would blow up and swallow the batch.
            denom = (1.0 - time).clamp_min(self.t_eps).unsqueeze(-1)
            return F.mse_loss(v_pred, (x1 - φ) / denom, reduction="none")
        if self.loss_space == "x":
            return F.mse_loss(x_pred, x1, reduction="none")
        raise ValueError(f"Unknown loss_space: {self.loss_space}")

    def _per_sample_loss(self, raw_pred, φ, x1, time, mask) -> float["b"]:
        """Per-row masked mean of the flow loss, in fp32 -- the ℓ that Δ is built from.

        Deliberately *not* shared with the pooled `loss[mask].mean()` in forward(): a mean
        over pooled elements and a mean of per-row means are different quantities whenever
        rows differ in length, and folding the two together would silently redefine the
        loss every already-trained arm was fitted with.
        """
        loss = self._elementwise_loss(raw_pred, φ, x1, time)
        m = mask.to(loss.dtype)
        return (loss * m).sum(dim=-1) / m.sum(dim=-1).clamp_min(1.0)

    def _ddo_pair_rows(self, is_fake: bool["b"]) -> tuple[torch.Tensor, torch.Tensor]:
        """(fake_idx, partner_idx): every fake row paired with a real row of the same batch.

        Common random numbers are what makes Δ low-variance (spec S3.5): the paper draws
        one (t, ε) and reuses it across the real and the fake batch. Both are shared here
        too. The rows of a batch are padded to one width, so ε has the same shape on both
        sides and a fake row can simply reuse its partner's noise at the same sample
        positions; t is a scalar per row and pairs trivially. Pairing is by position with
        wraparound, since the two sides are rarely the same size. A one-sided batch pairs
        nothing and both sides come back empty.
        """
        real_idx = (~is_fake).nonzero(as_tuple=True)[0]
        fake_idx = is_fake.nonzero(as_tuple=True)[0]
        if real_idx.numel() == 0 or fake_idx.numel() == 0:
            empty = fake_idx.new_zeros((0,))
            return empty, empty
        partner = real_idx[torch.arange(fake_idx.numel(), device=is_fake.device) % real_idx.numel()]
        return fake_idx, partner

    def _ddo_sample_time(self, batch: int, is_fake: bool["b"], *, dtype, device) -> float["b"]:
        """One t per row, with each fake row reusing its partner real row's draw.

        The full batch is drawn first and then partly overwritten, so the number of RNG
        draws per step does not depend on how many fake rows the sampler happened to pick.
        """
        time = self._sample_time(batch, dtype=dtype, device=device)
        fake_idx, partner = self._ddo_pair_rows(is_fake)
        if fake_idx.numel() > 0:
            time = time.index_copy(0, fake_idx, time[partner])
        return time

    def _ddo_sample_noise(self, x1: float["b nw"], is_fake: bool["b"]) -> float["b nw"]:
        """x0 per row, with each fake row reusing its partner real row's noise.

        Same RNG accounting as _ddo_sample_time: a full batch of noise is drawn and the
        fake rows are then overwritten, so the draw count never depends on the batch mix.
        """
        x0 = torch.randn_like(x1)
        fake_idx, partner = self._ddo_pair_rows(is_fake)
        if fake_idx.numel() > 0:
            x0 = x0.index_copy(0, fake_idx, x0[partner])
        return x0

    def _ddo_forward(self, inp: float["b nw"], *, lens: int["b"] | None, is_fake: bool["b"] | None):
        batch, seq_len, dtype, device = *inp.shape[:2], inp.dtype, self.device

        if not exists(lens):
            lens = torch.full((batch,), seq_len, device=device, dtype=torch.long)
        mask = lens_to_mask(lens, length=seq_len)

        if not exists(is_fake):
            is_fake = torch.zeros((batch,), device=device, dtype=torch.bool)
        else:
            is_fake = is_fake.to(device=device, dtype=torch.bool)

        # Same label-then-augment order as pretraining, with one extra conjunct: a fake row
        # is never null and never mixed. Fakes stand in for samples of p_ref itself, and
        # the negative term is only a likelihood ratio if both models score them under the
        # same (clean) condition. Mixing them would also hand the discriminator a feature
        # that says nothing about generation quality (spec S3.4).
        null_flags = (torch.rand(batch, device=device) < self.state_null_prob) & ~is_fake
        mix_flags = null_flags & (torch.rand(batch, device=device) < self.p_mix)
        x1 = self._mix_augment(inp, lens, mix_flags)
        state = torch.where(
            null_flags,
            torch.full((batch,), STATE_NULL, device=device, dtype=torch.long),
            torch.full((batch,), STATE_CLEAN, device=device, dtype=torch.long),
        )

        x1 = x1 * self.latents_scale
        x0 = self._ddo_sample_noise(x1, is_fake)
        time = self._ddo_sample_time(batch, is_fake, dtype=dtype, device=device)
        t = time.unsqueeze(-1)
        φ = (1 - t) * x0 + t * x1

        # Drawn once here and handed to both models. DiT would otherwise draw its own
        # randperm inside each forward, and Δ would become the loss gap between two
        # different position assignments instead of between two models (spec S3.6). The
        # augmentation stays *on* during DDO: turning it off would finetune the model on
        # contiguous positions only and erode the length extrapolation it was pretrained for.
        positions = None
        rpe_gamma = getattr(self.transformer, "rpe_gamma", 1.0)
        if self.training and rpe_gamma > 1.0:
            num_tokens = math.ceil(seq_len / self.wav_frame_len)
            positions = randomized_positions(batch, num_tokens, rpe_gamma, device)

        raw_pred = self.transformer(x=φ, state=state, time=time, mask=mask, lens=lens, positions=positions)
        elem_loss = self._elementwise_loss(raw_pred, φ, x1, time)

        # DDO rows are everything the discriminator sees: the real-positive rows and the
        # fake-negative rows. Null rows are held out of it entirely -- they get the MLE
        # anchor below instead.
        ddo_rows = ~null_flags
        idx = ddo_rows.nonzero(as_tuple=True)[0]
        if idx.numel() > 0:
            with torch.no_grad():
                # The backbone directly, not ref(...): ref.forward would redraw its own
                # labels, mixing and t, and Δ is only a log-ratio if both models score the
                # very same (x_t, t, state) rows. θ's forward stays on self.transformer so
                # that the whole DDO path runs inside DDP's __call__ (spec S3.9); ref needs
                # no such care because it never produces a gradient.
                #
                # Row slices, not a re-masked full batch: the padded width is identical, so
                # slicing rows keeps positions[idx] aligned token-for-token with θ's draw.
                ref_pred = self.ddo_ref.transformer(
                    x=φ[idx],
                    state=state[idx],
                    time=time[idx],
                    mask=mask[idx],
                    lens=lens[idx],
                    positions=None if positions is None else positions[idx],
                )
            m = mask[idx].to(elem_loss.dtype)
            loss_theta = (elem_loss[idx] * m).sum(dim=-1) / m.sum(dim=-1).clamp_min(1.0)
            loss_ref = self._per_sample_loss(ref_pred, φ[idx], x1[idx], time[idx], mask[idx])
            # Δ = -(ℓ_θ - ℓ_ref): a *lower* loss under θ means θ assigns the row more
            # likelihood than ref does, which is the positive direction of the log-ratio.
            delta = -(loss_theta - loss_ref)
            if self.ddo_delta_normalize == "sum":
                # The paper's unnormalized sum, restored by undoing the per-row mean. Only
                # sane at a fixed length: with 0.3-30 s clips a single global beta cannot
                # put βΔ at O(1) for both ends, and length becomes the one feature the
                # discriminator needs (spec S3.2). Kept as an ablation knob.
                delta = delta * m.sum(dim=-1)
            delta_v = delta

            # The mel channel (spec S3.10). The same discriminator reads a second
            # log-ratio proxy, the gap between the two models' multi-scale log-mel
            # reconstruction errors on the same (x_t, t), and the logit is
            # β·Δ_v + β_mel·Δ_mel -- one sigmoid, so the two channels saturate together
            # and the stats below describe what the loss actually sees. Divided by the
            # module's inner weight so β_mel is in raw log-mel L1 units. ref's x_pred
            # comes from the ref forward already made; only θ's side carries gradient.
            delta_mel = None
            if self.ddo_beta_mel > 0:
                _, x_pred_theta = self._predictions(raw_pred[idx].float(), φ[idx].float(), time[idx].float())
                _, x_pred_ref = self._predictions(ref_pred.float(), φ[idx].float(), time[idx].float())
                mel_kwargs = {}
                if self.aux_mel_loss_masked:
                    mel_kwargs.update(frame_mask=mask[idx], frame_lengths=lens[idx])
                x1_rows = x1[idx].float() / self.latents_scale
                mel_theta = self.aux_mel_loss(x_pred_theta / self.latents_scale, x1_rows, reduction="none", **mel_kwargs)
                with torch.no_grad():
                    mel_ref = self.aux_mel_loss(x_pred_ref / self.latents_scale, x1_rows, reduction="none", **mel_kwargs)
                delta_mel = -(mel_theta - mel_ref) / self.aux_mel_loss.weight
                delta = delta_v + (self.ddo_beta_mel / self.ddo_beta) * delta_mel
        else:
            delta = elem_loss.new_zeros((0,))
            delta_v, delta_mel = delta, None

        ddo_term, loss_dict = ddo_loss(delta, is_fake[idx], alpha=self.ddo_alpha, beta=self.ddo_beta)
        total_loss = ddo_term

        nan = elem_loss.new_full((), float("nan"))
        # per-channel views of Δ: delta_v_* always (equal to ddo/delta_* when the mel channel
        # is off), delta_mel_* only when it is on
        fake_rows_idx = is_fake[idx]
        for name, vec in (("delta_v", delta_v), ("delta_mel", delta_mel)):
            if vec is None or vec.numel() == 0:
                loss_dict[f"ddo/{name}_real"] = loss_dict[f"ddo/{name}_fake"] = loss_dict[f"ddo/{name}_std"] = nan
                continue
            r, f = vec[~fake_rows_idx], vec[fake_rows_idx]
            loss_dict[f"ddo/{name}_real"] = r.mean().detach() if r.numel() else nan
            loss_dict[f"ddo/{name}_fake"] = f.mean().detach() if f.numel() else nan
            loss_dict[f"ddo/{name}_std"] = vec.std().detach() if vec.numel() > 1 else nan
        anchor_value, aux_mel_value = nan, nan
        anchor_idx = null_flags.nonzero(as_tuple=True)[0]
        if anchor_idx.numel() > 0:
            # The null branch keeps its pretraining objective verbatim -- pooled over
            # elements, not per row. It is the model of the CFG negative, so it *should*
            # stay mode-covering; sharpening it narrows the subtrahend and points guidance
            # somewhere wrong, and leaving it unsupervised lets it drift with the shared
            # trunk while the clean branch is sharpened (spec S3.4).
            anchor_loss = elem_loss[anchor_idx][mask[anchor_idx]].mean()

            if self.use_aux_mel_loss and self.aux_mel_loss is not None:
                _, x_pred = self._predictions(
                    raw_pred[anchor_idx].float(), φ[anchor_idx].float(), time[anchor_idx].float()
                )
                aux_mel_kwargs = {}
                if self.aux_mel_loss_masked:
                    aux_mel_kwargs.update(frame_mask=mask[anchor_idx], frame_lengths=lens[anchor_idx])
                aux_mel_loss = self.aux_mel_loss(
                    x_pred / self.latents_scale,
                    x1[anchor_idx] / self.latents_scale,
                    **aux_mel_kwargs,
                )
                anchor_loss = anchor_loss + aux_mel_loss
                aux_mel_value = aux_mel_loss.detach()
            else:
                aux_mel_value = elem_loss.new_zeros(())

            # The aux mel term is a perceptual regularizer, not part of the ELBO, so it
            # never enters Δ -- it only lives here, on the anchor rows (spec S3.1).
            anchor_loss = self.ddo_anchor_weight * anchor_loss
            total_loss = total_loss + anchor_loss
            anchor_value = anchor_loss.detach()

        # flow_loss over the real rows is a divergence alarm, not a quality metric: DDO
        # trades likelihood for sample quality, so it is *expected* to rise. Past roughly
        # 2x the pretraining value the round has been pushed too far (spec S4).
        real_rows = ~is_fake
        flow_value = elem_loss[real_rows][mask[real_rows]].mean().detach() if real_rows.any() else nan

        loss_dict["total_loss"] = total_loss
        loss_dict["flow_loss"] = flow_value
        loss_dict["aux_mel_loss"] = aux_mel_value
        loss_dict["anchor_loss"] = anchor_value

        return total_loss, loss_dict

    def forward(
        self,
        inp: float["b nw"],  # raw waveform
        *,
        lens: int["b"] | None = None,
        is_fake: bool["b"] | None = None,  # DDO only: True = row came from p_ref's pool
    ):
        # handle raw waveform
        if inp.ndim != 2:
            raise ValueError(f"WavTTS expects raw waveform input [B, N], got {tuple(inp.shape)}")

        # No reference attached means no DDO, and then nothing below this line differs from
        # pretraining -- is_fake is ignored rather than raising, so a DDO-shaped dataloader
        # can feed an ordinary run unchanged.
        if self.ddo_ref is not None:
            return self._ddo_forward(inp, lens=lens, is_fake=is_fake)

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

        # time step
        time = self._sample_time(batch, dtype=dtype, device=device)

        # sample xt (phi_t(x) in the paper)
        t = time.unsqueeze(-1)
        φ = (1 - t) * x0 + t * x1
        flow = x1 - x0

        raw_pred = self.transformer(x=φ, state=state, time=time, mask=mask, lens=lens)

        # interpret prediction
        if self.prediction == "flow":
            v_pred = raw_pred
            x_pred = φ + (1.0 - t) * v_pred
        elif self.prediction == "x_pred":
            x_pred = raw_pred
            v_pred = self._x_to_v(x_pred, φ, time)
        else:
            raise ValueError(f"Unknown prediction: {self.prediction}")

        # loss space
        if self.loss_space == "flow":
            loss = F.mse_loss(v_pred, flow, reduction="none")
        elif self.loss_space == "v":
            # v-loss (same target flow, but v_pred computed from x_pred) & use clamp_min
            denom = (1.0 - time).clamp_min(self.t_eps)
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
        }

        return total_loss, loss_dict
