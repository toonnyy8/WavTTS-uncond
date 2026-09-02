"""Zero-shot speaker-conditioned generation on an unconditional WavTTS.

Training-free: generation runs on a concatenated [ref | gen] sequence and the ref half
is pinned back onto a reference trajectory after every ODE step. Attention is the only
channel — the generated half reads an always-plausible reference and inherits its
speaker. This is RePaint-style replacement (Lugmayr et al. 2022) on a deterministic
rectified-flow ODE; see docs/superpowers/plans/wavtts_inversion_replacement_spk_cond.md.

Two ways to build the reference trajectory (`ref_traj`):

  "noise"  — x^ref_t = t·x_ref + (1-t)·eps with one fixed eps. Every state is an exact
             sample of the training marginal p(x_t | x_1 = x_ref), which is all the
             replacement needs, and it costs nothing.
  "invert" — the plan's §3.1: run the ODE backwards from x_ref and cache the states.
             Exact under the model's own flow *if* the flow inverts. On this checkpoint
             it does not — round-trip reconstruction sits near 0 dB against the plan's
             25 dB gate at every NFE from 16 to 256 — so what gets pinned is a path the
             model never travels. `invert_t_start` < 1 is what makes it usable anyway:
             entering below the untrained top of the t range stops the inversion from
             integrating an unlearned field through a diverging 1/(1-t), and pulls the
             noise endpoint back from rms 0.68 to 0.98. Still behind "noise" on identity
             and naturalness at four times the cost, so it stays the ablation.

`repl_cfg` adds guidance between the replaced and un-replaced branches: the velocity of
the full [ref | gen] sequence, extrapolated away from the velocity the generated half
produces on its own. That difference is the reference's entire contribution, so pushing
along it amplifies the conditioning without any extra conditioning signal. It is computed
in the data domain and the result is rescaled to the conditional branch's moments before
going back to velocity (`repl_cfg_rescale`), which is what keeps strong guidance from
blowing the waveform's level out.

Not implemented from that plan: speaker-embedding guidance (§3.4, a frozen speaker
encoder's gradient — a different thing from `repl_cfg`) and rolling long-form generation
(§3.5). Both are additive on top of what is here.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from importlib.resources import files

import torch
import torch.nn.functional as F
import torchaudio
from omegaconf import OmegaConf

from wavtts.infer.sample_uncond import load_model
from wavtts.model import CFM
from wavtts.model.utils import peak_normalize


def prepare_ref(
    path: str,
    *,
    sample_rate: int,
    target_rms: float,
    hop: int,
    max_sec: float | None = None,
    device="cpu",
    dtype=torch.float32,
) -> torch.Tensor:
    """Load a reference clip into the domain the model was trained on: mono, 16 kHz,
    DC removed, RMS normalized to `target_rms` (exactly what CustomDataset does), and
    a whole number of hops long so the ref/gen split lands on a token boundary."""
    wav, sr = torchaudio.load(path)
    wav = wav.mean(dim=0, keepdim=True).to(torch.float32)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    if max_sec is not None:
        wav = wav[:, : int(max_sec * sample_rate)]

    wav = wav[:, : (wav.shape[1] // hop) * hop]
    if wav.shape[1] < hop:
        raise ValueError(f"reference {path} is shorter than one frame hop ({hop} samples)")

    if target_rms > 0:
        wav = wav - wav.mean()
        rms = wav.pow(2).mean().sqrt()
        if rms > 1e-5:
            wav = wav * (target_rms / rms)
    return wav.to(device=device, dtype=dtype)


@torch.inference_mode()
def invert(model: CFM, ref: torch.Tensor, t: torch.Tensor, *, cfg_strength: float = 0.0) -> torch.Tensor:
    """Backward Euler along the sampling ODE, data end to noise end.

    `ref` is a waveform already in the model's scaled domain. Returns the full
    trajectory [K+1, b, n], indexed like `t`: traj[K] is the reference itself and
    traj[0] the noise it inverts to. Keeping every state (rather than only the noise
    and interpolating linearly at generation time) is what makes the replacement exact:
    the ref half then rides the model's own curved trajectory, not a straight-line
    guess at it.
    """
    x = ref
    traj = [None] * len(t)
    traj[-1] = x
    for i in range(len(t) - 1, 0, -1):
        # x-prediction turns into velocity through 1/(1-t), which is unbounded at the
        # data end; the model's own t_eps is the floor training ever used, so evaluate
        # no closer than that even when the schedule's last node is t=1
        t_eval = t[i].clamp(max=1.0 - model.t_eps)
        x = x - (t[i] - t[i - 1]) * model.velocity(x, t_eval, cfg_strength=cfg_strength)
        traj[i - 1] = x
    return torch.stack(traj)


@torch.inference_mode()
def integrate(model: CFM, x: torch.Tensor, t: torch.Tensor, *, cfg_strength: float = 0.0) -> torch.Tensor:
    """Plain forward Euler on the same grid — the inverse of `invert`, used to check it."""
    for i in range(len(t) - 1):
        x = x + (t[i + 1] - t[i]) * model.velocity(x, t[i], cfg_strength=cfg_strength)
    return x


def reconstruction_snr(ref: torch.Tensor, recon: torch.Tensor) -> float:
    err = (recon - ref).pow(2).mean()
    return float(10 * torch.log10(ref.pow(2).mean() / err.clamp_min(1e-12)))


def _renoise(x: torch.Tensor, t_b: float, t_a: float, generator=None) -> torch.Tensor:
    """Step back down the schedule, t_b -> t_a < t_b, keeping the marginal right.

    For x_t = (1-t)·x_0 + t·x_1 the data part scales by t_a/t_b and the noise variance
    has to be topped back up to (1-t_a)^2. (RePaint's time-travel, rewritten for the
    linear interpolant rather than a VP diffusion.)
    """
    if t_b <= 0:
        return x
    scale = t_a / t_b
    var = max((1.0 - t_a) ** 2 - (scale * (1.0 - t_b)) ** 2, 0.0)
    eps = torch.empty_like(x).normal_(generator=generator)
    return scale * x + var**0.5 * eps


def _smoothstep(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


@dataclass(frozen=True)
class CCG:
    """Context-consistency guidance — §3.2 of
    docs/superpowers/plans/context_consistency_guidance.md, in this repo's time
    convention (t=0 noise, t=1 data; the document's t is 1 - ours).

    The document's strength is `w` on `v_short + w·Δ`; `w` here is that minus one, so 0
    is the plain long branch (ordinary replacement) and the scale matches --repl_cfg.
    """

    w: float = 0.0  # 0 = no extrapolation, i.e. plain replacement
    window: tuple[float, float] | None = None  # (t_lo, t_hi): full below t_lo, 0 above t_hi
    lp_k: int = 1  # Φ_LP downsample factor; 1 = off
    eta_apg: float = 1.0  # Φ_APG: how much of the component along x_cond survives; 1 = off
    rescale: float = 1.0  # restore the conditional branch's moments; 0 = off
    kappa: float = 1.0  # context noise level: t_ctx = 1 - κ(1-t). 1 = synchronous (design A)

    def strength(self, t: float) -> float:
        """w(t). Identity is a global, low-frequency property settled early, while the
        late steps' Δ is mostly OOD residue and high-frequency misdirection — hence the
        limited interval (Kynkäänniemi et al. 2024)."""
        if self.window is None:
            return self.w
        lo, hi = self.window
        if hi <= lo:
            return self.w if t <= lo else 0.0
        return self.w * _smoothstep((hi - t) / (hi - lo))


def _lowpass(x: torch.Tensor, k: int) -> torch.Tensor:
    """Φ_LP: decimate by k and interpolate back. Long inputs degrade the model's high
    frequencies before its low ones (FreeLong), and identity lives in the low ones."""
    if k <= 1:
        return x
    n = x.shape[-1]
    pad = (-n) % k
    y = F.avg_pool1d(F.pad(x, (0, pad)).unsqueeze(1), k).squeeze(1)
    y = F.interpolate(y.unsqueeze(1), size=n + pad, mode="linear", align_corners=False).squeeze(1)
    return y[:, :n]


def _match_moments(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Rescale x to ref's per-sample mean and standard deviation."""
    xs = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
    return (x - x.mean(dim=-1, keepdim=True)) * (ref.std(dim=-1, keepdim=True) / xs) + ref.mean(dim=-1, keepdim=True)


def _apply_guidance(model: CFM, x: torch.Tensor, t, v_base: torch.Tensor, delta_v: torch.Tensor, ccg: CCG):
    """x_pred = x_base + w·Φ(Δ), all in the data domain.

    `v_base` is the sampler's own w=0 output — what it would do with no guidance at all,
    and the reference the rescale restores. `delta_v` is the direction more context
    points in; which branch is the base and which supplies Δ differs between samplers and
    is decided at the call site.

    The detour through the data prediction matters only for Φ: the extrapolation itself
    is affine, and so is x <-> v at fixed t, so that part is domain-independent. Φ_LP,
    Φ_APG and the rescale are not affine, and a waveform's frequency content and level
    only mean anything in the data domain.
    """
    w = ccg.strength(float(t))
    if abs(w) < 1e-5:
        return v_base

    # clamp_min(t_eps) is the same floor _x_to_v used on the way out, so this round-trips
    # exactly rather than approximately
    denom = (1.0 - t).clamp_min(model.t_eps)
    x_base = x + denom * v_base

    delta = _lowpass(denom * delta_v, ccg.lp_k)
    if ccg.eta_apg != 1.0:
        # Δ measured on this model runs 0.75-0.83 cosine with the prediction itself, so
        # most of the extra context's "contribution" is a gain change rather than a
        # direction. Keeping only the orthogonal part is what APG proposes (Sadat et al.
        # 2025 — projecting onto the prediction, rather than onto the short branch as the
        # plan writes it).
        scale = (delta * x_base).sum(-1, keepdim=True) / x_base.pow(2).sum(-1, keepdim=True).clamp_min(1e-8)
        d_par = scale * x_base
        delta = ccg.eta_apg * d_par + (delta - d_par)

    x_pred = x_base + delta * w
    if ccg.rescale > 0:
        # extrapolation inflates the prediction's scale, and that inflation — not the
        # direction — is what turns into blown-out audio at large w. Put the base
        # branch's first two moments back and keep only the direction. (Lin et al. 2024,
        # "Common Diffusion Noise Schedules and Sample Steps are Flawed", §3.4.)
        x_pred = torch.lerp(x_pred, _match_moments(x_pred, x_base), ccg.rescale)
    return (x_pred - x) / denom


def _guided_velocity(model: CFM, x: torch.Tensor, ctx_len: int, t, *, cfg_strength: float, ccg: CCG):
    """Segment-wise branches: the long branch runs on the full [ctx | cur] sequence, the
    short branch on the current segment alone. Their difference Δ is what the context adds
    through attention — approximately ∇ log p(ctx | cur), the evidence that the two halves
    belong together. The short branch is computed on the segment alone and never on an
    overlap-average, or Δ would already contain the context it is meant to isolate.

    The two branches are different sequence lengths, which is a confound: with
    logn_ref_len=500 a 5 s segment runs at attention temperature 1.000 and a 10 s
    [ctx | cur] at 1.112, and the conv position embedding sees a padded edge in one and
    continuous audio in the other. Measured, that accounts for about 3% of ‖Δ‖ here.
    `generate_ola` avoids it entirely by making both branches the same window length.
    """
    v = model.velocity(x, t, cfg_strength=cfg_strength)
    if abs(ccg.strength(float(t))) < 1e-5:
        return v
    x_cur = x[:, ctx_len:]
    v_free = model.velocity(x_cur, t, cfg_strength=cfg_strength)
    # base is the long branch: w=0 must be plain replacement, which is this sampler's
    # own baseline, and Δ pushes further away from the no-context branch
    v_long = v[:, ctx_len:]
    guided = _apply_guidance(model, x_cur, t, v_long, v_long - v_free, ccg)
    return torch.cat([v[:, :ctx_len], guided], dim=-1)


@torch.inference_mode()
def generate(
    model: CFM,
    traj: torch.Tensor,
    gen_samples: int,
    t: torch.Tensor,
    *,
    batch: int = 1,
    cfg_strength: float = 0.0,
    resample_rounds: int = 1,
    resample_range: tuple[float, float] = (0.15, 0.5),
    ccg: CCG = CCG(),
    seed: int | None = None,
) -> torch.Tensor:
    """Replacement sampling on [ctx | cur]. Returns the generated half, unscaled."""
    device, dtype = traj.device, traj.dtype
    ref_len = traj.shape[-1]
    aligned = int(-(-gen_samples // model.wav_frame_hop) * model.wav_frame_hop)

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
    noise = torch.randn(batch, aligned, device=device, dtype=dtype, generator=generator)

    def pin(x, i):  # ref half back onto the reference trajectory at t[i]
        return torch.cat([traj[i].expand(batch, -1), x[:, ref_len:]], dim=-1)

    x = torch.cat([traj[0].expand(batch, -1), noise], dim=-1)
    lo, hi = resample_range
    for i in range(len(t) - 1):
        # the ref half is nearly pure noise at small t, which is exactly where global
        # structure (identity included) is decided — resampling there gives the two
        # halves more rounds to agree before the choice is locked in
        rounds = resample_rounds if lo <= float(t[i]) <= hi else 1
        for u in range(rounds):
            v = _guided_velocity(model, x, ref_len, t[i], cfg_strength=cfg_strength, ccg=ccg)
            x = pin(x + (t[i + 1] - t[i]) * v, i + 1)
            if u < rounds - 1:
                x = pin(_renoise(x, float(t[i + 1]), float(t[i]), generator), i)

    return x[:, ref_len : ref_len + gen_samples] / model.latents_scale


def _interp_noise(n: torch.Tensor, grid: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Linear interpolation of a noise path onto another time grid. Times above the
    grid's top hold its last value — there is nothing above t_start to interpolate."""
    tq = t.clamp(max=float(grid[-1]))
    hi = torch.searchsorted(grid, tq, right=True).clamp(1, len(grid) - 1)
    lo = hi - 1
    w = ((tq - grid[lo]) / (grid[hi] - grid[lo]).clamp_min(1e-12)).view(-1, *([1] * (n.ndim - 1)))
    return torch.lerp(n[lo], n[hi], w)


def ref_trajectory(
    model: CFM,
    ref: torch.Tensor,
    t: torch.Tensor,
    *,
    mode: str = "noise",
    cfg_strength: float = 0.0,
    seed: int | None = None,
    kappa: float = 1.0,
    invert_t_start: float = 0.95,
    invert_steps: int | None = None,
    invert_renorm: bool = False,
) -> torch.Tensor:
    """The [K+1, b, n] reference states the replacement pins to. `ref` is already scaled.

    Both modes are the same object underneath — a noise path n(t) with the state read off
    as x^ref_t = t·x_ref + (1-t)·n(t). Writing it this way keeps the data part exact by
    construction and leaves only n(t) to argue about:

      "noise"   n(t) = eps, one fixed draw. Every state is then an exact sample of the
                training marginal p(x_t | x_1 = x_ref).
      "invert"  n(t) comes from running the ODE backwards, so it carries the model's own
                curvature — but only its *direction* is trusted; see invert_renorm.
    """
    generator = None
    if seed is not None:
        generator = torch.Generator(device=ref.device)
        generator.manual_seed(int(seed))

    tt = t.to(ref.dtype).view(-1, *([1] * ref.ndim))
    if kappa != 1.0:
        # fractional context (plan §3.1 design C): the context sits at a lower noise
        # level than the segment being generated, so it carries more of itself into
        # attention. κ=1 is design A, the synchronous noise the fixed-length prior was
        # trained on; below 1 trades in-distribution-ness for context strength.
        tt = 1.0 - kappa * (1.0 - tt)
    if mode == "noise":
        # one eps for the whole schedule, not a fresh draw per step: the ref half then
        # walks a continuous path from noise to x_ref instead of jittering under the
        # generated half's attention, and traj[0] is still a plain N(0, I) sample
        eps = torch.empty_like(ref).normal_(generator=generator)
        return tt * ref + (1.0 - tt) * eps
    if mode != "invert":
        raise ValueError(f"Unknown ref_traj mode: {mode}")

    # Start the inversion below t=1. Logit-normal t sampling (P_mean -0.8) leaves the
    # top of the range essentially untrained, and that is also where x-prediction turns
    # into velocity through a vanishing 1/(1-t) — inverting through it is integrating an
    # unlearned field with an exploding gain. x_start is an exact marginal sample, so
    # nothing is given up by entering lower down.
    t_start = min(float(invert_t_start), 1.0)
    x_start = ref
    if t_start < 1.0:
        eps0 = torch.empty_like(ref).normal_(generator=generator)
        x_start = t_start * ref + (1.0 - t_start) * eps0

    k = int(invert_steps) if invert_steps else len(t) - 1
    grid = torch.linspace(0.0, t_start, k + 1, device=ref.device, dtype=torch.float32)
    states = invert(model, x_start, grid, cfg_strength=cfg_strength)

    gg = grid.to(ref.dtype).view(-1, *([1] * ref.ndim))
    n = (states - gg * ref) / (1.0 - gg).clamp_min(1e-6)  # implied noise at each node
    if invert_renorm:
        # the inversion drifts off the training statistics — n_ref lands near rms 0.68
        # where the model has only ever seen 1.0. Rescaling to the moments the marginal
        # demands keeps the inversion's structure (which direction the noise points) and
        # throws away only its miscalibration.
        n = (n - n.mean(dim=-1, keepdim=True)) / n.std(dim=-1, keepdim=True).clamp_min(1e-6)
    return tt * ref + (1.0 - tt) * _interp_noise(n, grid, t.to(ref.dtype))


def speaker_conditioned_sample(
    model: CFM,
    ref_wav: torch.Tensor,
    duration: int,
    t: torch.Tensor,
    *,
    batch: int = 1,
    ref_traj: str = "noise",
    cfg_strength: float = 0.0,
    resample_rounds: int = 1,
    resample_range: tuple[float, float] = (0.15, 0.5),
    ccg: CCG = CCG(),
    seed: int | None = None,
    invert_t_start: float = 0.95,
    invert_steps: int | None = None,
    invert_renorm: bool = False,
) -> tuple[torch.Tensor, float | None]:
    """Reference trajectory + replacement, end to end. `ref_wav` is [1, n] in the
    loader's domain (see prepare_ref). Returns (generated waveform [batch, duration],
    inversion reconstruction SNR or None when the trajectory needs no inversion)."""
    x_ref = ref_wav * model.latents_scale
    # the reference path's eps and the generated half's initial noise are two different
    # draws, and have to stay that way: at equal ref and gen lengths one seed hands both
    # halves the identical noise, which starts the generated half on the reference's own
    # trajectory and flatters every similarity number downstream
    ref_seed = None if seed is None else int(seed) + 1_000_003
    traj = ref_trajectory(
        model,
        x_ref,
        t,
        mode=ref_traj,
        cfg_strength=cfg_strength,
        seed=ref_seed,
        kappa=ccg.kappa,
        invert_t_start=invert_t_start,
        invert_steps=invert_steps,
        invert_renorm=invert_renorm,
    )
    snr = None
    if ref_traj == "invert":
        snr = reconstruction_snr(x_ref, integrate(model, traj[0], t, cfg_strength=cfg_strength))
    wav = generate(
        model,
        traj,
        duration,
        t,
        batch=batch,
        cfg_strength=cfg_strength,
        resample_rounds=resample_rounds,
        resample_range=resample_range,
        ccg=ccg,
        seed=seed,
    )
    return wav, snr


def _window_starts(n: int, win: int, hop: int) -> list[int]:
    """Cover [0, n) with `win`-long windows spaced `hop` apart, the last one snapped to
    the end so no samples are left uncovered."""
    if n <= win:
        return [0]
    starts = list(range(0, n - win + 1, hop))
    if starts[-1] + win < n:
        starts.append(n - win)
    return starts


@torch.inference_mode()
def _ola_velocity(model: CFM, x: torch.Tensor, t, *, win: int, hop: int, cfg_strength: float = 0.0) -> torch.Tensor:
    """Velocity of a long clip as the plain average of the overlapping `win`-long windows
    covering each sample.

    Uniform weights, not a taper: every window's prediction of a sample counts the same.
    The overlap is here so no sample sits next to a window edge in every window it
    belongs to — it is not trying to blend seams away, because with a velocity field
    there are none to blend.
    """
    n = x.shape[-1]
    if n <= win:
        return model.velocity(x, t, cfg_strength=cfg_strength)

    starts = _window_starts(n, win, hop)
    v = model.velocity(torch.cat([x[:, s : s + win] for s in starts], dim=0), t, cfg_strength=cfg_strength)

    acc, count = torch.zeros_like(x), torch.zeros_like(x)
    for i, st in enumerate(starts):
        acc[:, st : st + win] += v[i : i + 1]
        count[:, st : st + win] += 1.0
    return acc / count.clamp_min(1.0)


@torch.inference_mode()
def generate_ola(
    model: CFM,
    total_samples: int,
    t: torch.Tensor,
    *,
    win: int,
    hop: int,
    cfg_strength: float = 0.0,
    resample_rounds: int = 1,
    resample_range: tuple[float, float] = (0.15, 0.5),
    ccg: CCG = CCG(),
    seed: int | None = None,
) -> torch.Tensor:
    """Sample a long clip in one piece, guiding the windowed sampler toward what the
    model does when it can see everything.

    Every ODE step is taken on the whole waveform. There are no segments and no joins:

      short branch  overlapping `win`-long windows, overlap-added — the standard way a
                    fixed-length prior is driven long, and this sampler's w=0 output. Each
                    sample sees only its own neighbourhood.
      long branch   one forward on the entire clip. Every sample attends to every other,
                    so this is where any global consistency has to come from.

    Δ = v_full − v_windowed is what seeing the whole clip adds. Guidance runs *from* the
    windowed branch toward it rather than starting there, because the full-length forward
    is the one that degrades as the clip grows past what the model was trained on: w=0 is
    the windowed sampler untouched, w=1 is the full pass, and in between is a blend whose
    balance can be tuned to how far out of its depth the full pass is.

    `win >= total_samples` collapses the short branch onto the long one and turns this
    into a plain single-pass sampler, which is the baseline worth beating.
    """
    generator = None
    if seed is not None:
        generator = torch.Generator(device=model.device)
        generator.manual_seed(int(seed))
    aligned = int(-(-total_samples // model.wav_frame_hop) * model.wav_frame_hop)
    dtype = next(model.parameters()).dtype
    x = torch.randn(1, aligned, device=model.device, dtype=dtype, generator=generator)

    lo, hi = resample_range
    for i in range(len(t) - 1):
        rounds = resample_rounds if lo <= float(t[i]) <= hi else 1
        for u in range(rounds):
            v = _ola_velocity(model, x, t[i], win=win, hop=hop, cfg_strength=cfg_strength)
            if abs(ccg.strength(float(t[i]))) >= 1e-5:
                v_full = model.velocity(x, t[i], cfg_strength=cfg_strength)
                v = _apply_guidance(model, x, t[i], v, v_full - v, ccg)
            x = x + (t[i + 1] - t[i]) * v
            if u < rounds - 1:
                x = _renoise(x, float(t[i + 1]), float(t[i]), generator)

    return x[:, :total_samples] / model.latents_scale


@torch.inference_mode()
def generate_long(
    model: CFM,
    total_samples: int,
    t: torch.Tensor,
    *,
    seg_samples: int,
    ctx_samples: int,
    init_ref: torch.Tensor | None = None,
    cfg_strength: float = 0.0,
    resample_rounds: int = 1,
    resample_range: tuple[float, float] = (0.15, 0.5),
    ccg: CCG = CCG(),
    seed: int | None = None,
) -> torch.Tensor:
    """Roll segment by segment, each conditioned on the tail of what came before
    (plan §4). Returns [1, total_samples] in the loader's domain.

    `init_ref` anchors the first segment to a real speaker; without it the first segment
    is a plain unconditional sample and everything after inherits whoever it turned out
    to be. `ctx_samples <= 0` disables the rolling entirely. The context is pinned rather
    than regenerated, so segments abut on audio the model actually saw — no overlap or
    crossfade is needed to join them.
    """
    context = init_ref
    out: list[torch.Tensor] = []
    produced = 0
    seg_i = 0
    while produced < total_samples:
        n = min(seg_samples, total_samples - produced)
        step_seed = None if seed is None else int(seed) + 7919 * seg_i
        if context is None:
            wav, _ = model.sample(n, batch=1, steps=len(t) - 1, cfg_strength=cfg_strength, seed=step_seed)
        else:
            wav, _ = speaker_conditioned_sample(
                model,
                context,
                n,
                t,
                cfg_strength=cfg_strength,
                resample_rounds=resample_rounds,
                resample_range=resample_range,
                ccg=ccg,
                seed=step_seed,
            )
        out.append(wav[:, :n])
        produced += n
        seg_i += 1
        # the context is the tail of everything produced so far, so a segment shorter
        # than ctx_samples still hands the next one a full-length window.
        # ctx_samples <= 0 carries nothing: every segment comes from the bare prior,
        # which is the drift baseline the guidance is measured against
        context = torch.cat(out, dim=-1)[:, -ctx_samples:] if ctx_samples > 0 else None
    return torch.cat(out, dim=-1)[:, :total_samples]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="WavTTS zero-shot speaker-conditioned sampling")
    parser.add_argument("--ckpt", required=True, help="checkpoint .pt path")
    parser.add_argument("--config", default=None, help="training yaml (default: packaged WavTTS.yaml)")
    parser.add_argument("--ref", required=True, nargs="+", help="reference wav(s)")
    parser.add_argument("--ref_sec", type=float, default=None, help="truncate each reference to this many seconds")
    parser.add_argument("--duration_sec", type=float, default=5.0)
    parser.add_argument("--num", type=int, default=1, help="samples per reference")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--cfg_strength", type=float, default=0.0)
    parser.add_argument(
        "--ref_traj",
        choices=["noise", "invert"],
        default="noise",
        help="reference path: forward noising (default) or the plan's ODE inversion",
    )
    parser.add_argument(
        "--invert_t_start",
        type=float,
        default=0.95,
        help="--ref_traj invert: enter the ODE here rather than at t=1 (1.0 is the plan's literal §3.1)",
    )
    parser.add_argument("--invert_steps", type=int, default=None, help="--ref_traj invert: NFE (default: --steps)")
    parser.add_argument("--invert_renorm", action="store_true", help="--ref_traj invert: moment-match the noise path")
    parser.add_argument("--resample_rounds", type=int, default=1, help="RePaint time-travel rounds (1 = off)")
    parser.add_argument("--resample_range", type=float, nargs=2, default=(0.15, 0.5), metavar=("LO", "HI"))
    parser.add_argument(
        "--repl_cfg",
        type=float,
        default=0.0,
        help="extrapolate away from the un-referenced branch; 0 = plain replacement, costs a 2nd forward/step",
    )
    parser.add_argument(
        "--repl_cfg_rescale",
        type=float,
        default=1.0,
        help="how much of the conditional branch's scale to restore after extrapolating (0 = none, 1 = all)",
    )
    parser.add_argument(
        "--ccg_window",
        type=float,
        nargs=2,
        default=None,
        metavar=("T_LO", "T_HI"),
        help="guide at full strength below T_LO and taper to 0 at T_HI (default: every step)",
    )
    parser.add_argument("--ccg_lp_k", type=int, default=1, help="low-pass the guidance by this factor; 1 = off")
    parser.add_argument("--ccg_eta_apg", type=float, default=1.0, help="keep this much of Δ along x_cond; 1 = off")
    parser.add_argument("--ccg_kappa", type=float, default=1.0, help="context noise level t_ctx = 1-κ(1-t); 1 = sync")
    parser.add_argument("--long_sec", type=float, default=None, help="generate a clip this long")
    parser.add_argument(
        "--sampler",
        choices=["rolling", "ola"],
        default="ola",
        help="--long_sec: segment-by-segment with a pinned context, or one overlap-added pass",
    )
    parser.add_argument(
        "--win_sec",
        type=float,
        default=0.0,
        help="--sampler ola: window the model sees; 0 (default) means one pass over the whole clip",
    )
    parser.add_argument("--ola_hop_sec", type=float, default=5.0, help="--sampler ola: spacing between windows")
    parser.add_argument("--seg_sec", type=float, default=5.0, help="--long_sec: length generated per segment")
    parser.add_argument("--ctx_sec", type=float, default=5.0, help="--long_sec: context carried between segments")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--with_ref", action="store_true", help="also write ref+gen concatenated, for listening")
    parser.add_argument("--out_dir", default="samples_spk_cond")
    parser.add_argument("--device", default=None, help="cpu | cuda | cuda:N (default: auto)")
    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)

    config_path = args.config or str(files("wavtts").joinpath("configs/WavTTS.yaml"))
    cfg = OmegaConf.load(config_path)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, cfg, device)

    sr = cfg.model.cfm.sample_rate
    t = model.timesteps(args.steps, device=device)
    duration = int(args.duration_sec * sr)
    ccg = CCG(
        w=args.repl_cfg,
        window=tuple(args.ccg_window) if args.ccg_window else None,
        lp_k=args.ccg_lp_k,
        eta_apg=args.ccg_eta_apg,
        rescale=args.repl_cfg_rescale,
        kappa=args.ccg_kappa,
    )
    os.makedirs(args.out_dir, exist_ok=True)

    for ref_path in args.ref:
        ref = prepare_ref(
            ref_path,
            sample_rate=sr,
            target_rms=cfg.model.waveform.target_rms,
            hop=model.wav_frame_hop,
            max_sec=args.ref_sec,
            device=device,
            dtype=next(model.parameters()).dtype,
        )
        if args.long_sec is not None and args.sampler == "ola":
            wav = generate_ola(
                model,
                int(args.long_sec * sr),
                t,
                win=int(args.win_sec * sr) if args.win_sec else int(args.long_sec * sr),
                hop=int(args.ola_hop_sec * sr) if args.win_sec else int(args.long_sec * sr),
                cfg_strength=args.cfg_strength,
                resample_rounds=args.resample_rounds,
                resample_range=tuple(args.resample_range),
                ccg=ccg,
                seed=args.seed,
            )
            wavs, snr = wav, None
        elif args.long_sec is not None:
            wav = generate_long(
                model,
                int(args.long_sec * sr),
                t,
                seg_samples=int(args.seg_sec * sr),
                ctx_samples=int(args.ctx_sec * sr),
                init_ref=ref,
                cfg_strength=args.cfg_strength,
                resample_rounds=args.resample_rounds,
                resample_range=tuple(args.resample_range),
                ccg=ccg,
                seed=args.seed,
            )
            wavs, snr = wav, None
        else:
            wavs, snr = speaker_conditioned_sample(
                model,
                ref,
                duration,
                t,
                batch=args.num,
                ref_traj=args.ref_traj,
                cfg_strength=args.cfg_strength,
                resample_rounds=args.resample_rounds,
                resample_range=tuple(args.resample_range),
                ccg=ccg,
                seed=args.seed,
                invert_t_start=args.invert_t_start,
                invert_steps=args.invert_steps,
                invert_renorm=args.invert_renorm,
            )
        stem = os.path.splitext(os.path.basename(ref_path))[0]
        print(f"{stem}: ref {ref.shape[1] / sr:.2f}s")
        if snr is not None:
            # only says whether the flow inverts. It is near 0 dB for the forward-noising
            # path too, which is the one that conditions best, so a low number here is not
            # by itself a reason to distrust the sample
            print(f"  inversion reconstruction SNR {snr:.1f} dB (plan's gate: > 25 dB)")

        for i in range(args.num):
            wav = wavs[i : i + 1].to(torch.float32).cpu()
            if args.with_ref:
                wav = torch.cat([ref[:, -wav.shape[1] :].to(torch.float32).cpu(), wav], dim=-1)
            out_path = os.path.join(args.out_dir, f"{stem}_gen{i}.wav")
            torchaudio.save(out_path, peak_normalize(wav), sr)
            print(f"  saved {out_path}")


if __name__ == "__main__":
    main()
