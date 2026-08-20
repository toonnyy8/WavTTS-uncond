"""YaRN rotary embedding and randomized positional encoding, for length extrapolation.

Follows "Randomized YaRN Improves Length Generalization for Long-Context Reasoning"
(Mehta, Yin & Durrett, arXiv:2606.23687), adapted to 100 Hz waveform frames:

  1. YaRN (Peng et al., 2024) with a *fixed* scale `s` during training. Inference
     uses `s' >= s`; the paper's Appendix B shows `s' > s` reaches beyond `s * native_ctx`.
  2. Randomized positional encoding (Ruoss et al., 2023): training positions are
     `randperm(L_t)[:n].sort()` instead of `arange(n)`, so a short clip still shows
     the model rotations it would only meet in a long one. Training only.

Two deliberate deviations from the paper:

`L_t` is a multiple of each sample's own length, not a fixed constant. Their sequences
all sit near the training cap, so a fixed `L_t` stretches every sample about equally.
Ours run 0.3-30 s, and a fixed `L_t` would stretch the median 4.5 s clip ~13x — in
waveform modelling relative position *is* physical time, carrying pitch period and
formant transitions, so an uneven stretch attacks exactly the structure the model
depends on.

`L_t` is drawn per sample from `U[n, n*gamma]` instead of stepping through a
curriculum. A curriculum makes the stretch factor a property of the *update*: every
sample in a batch is stretched the same amount, and the model never sees a contiguous
sequence again once the schedule leaves k=1. A random upper bound makes it a property
of the *sample*, so from the first update every batch contains rows at every stretch
from contiguous to gamma. That also removes a discrete regime change from the middle
of training, which is one less thing for a weight average to straddle.

Outputs follow the x_transformers rotary contract — `(freqs, scale)` consumable by
`apply_rotary_pos_emb`, which already broadcasts a per-sample `[b, n, d]` freqs.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.amp import autocast


def _correction_dim(num_rotations: float, dim: int, base: float, native_ctx: int) -> float:
    """RoPE dimension index that completes `num_rotations` turns across `native_ctx`."""
    return (dim * math.log(native_ctx / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def yarn_inv_freq(
    dim: int,
    base: float,
    scale: float,
    native_ctx: int,
    alpha: float = 1.0,
    beta: float = 32.0,
) -> torch.Tensor:
    """NTK-by-parts interpolated inverse frequencies (YaRN, Peng et al. 2024).

    Dimensions turning more than `beta` times across `native_ctx` are left to
    extrapolate, those turning less than `alpha` times are fully interpolated by
    `scale`, and the band between ramps linearly.
    """
    pos_freqs = base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (scale * pos_freqs)

    low = math.floor(_correction_dim(beta, dim, base, native_ctx))
    high = math.ceil(_correction_dim(alpha, dim, base, native_ctx))
    low, high = max(low, 0), min(high, dim - 1)
    if low == high:
        high += 0.001  # ponytail: guard the degenerate ramp, same as the reference impl

    ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low)).clamp(0, 1)
    extrapolation_weight = 1 - ramp
    return inv_freq_interpolation * (1 - extrapolation_weight) + inv_freq_extrapolation * extrapolation_weight


def yarn_attention_factor(scale: float) -> float:
    """YaRN's length-dependent attention temperature, `sqrt(1/t) = 0.1*ln(s) + 1`.

    Applied to both query and key in the reference implementation, so the logits
    end up scaled by its square.
    """
    return 0.1 * math.log(scale) + 1.0 if scale > 1.0 else 1.0


class YaRNRotaryEmbedding(nn.Module):
    """Drop-in for x_transformers' RotaryEmbedding, with YaRN frequencies.

    `forward` takes explicit positions (`[n]` or `[b, n]`) so randomized positional
    encoding needs nothing more than a different position tensor.
    """

    def __init__(
        self,
        dim: int,
        scale: float = 1.0,
        native_ctx: int = 3000,
        base: float = 10000.0,
        alpha: float = 1.0,
        beta: float = 32.0,
    ):
        super().__init__()
        self.scale = scale
        self.native_ctx = native_ctx
        self.register_buffer("inv_freq", yarn_inv_freq(dim, base, scale, native_ctx, alpha, beta), persistent=False)

    @autocast("cuda", enabled=False)
    def forward(self, positions: torch.Tensor):
        if positions.ndim == 1:
            positions = positions.unsqueeze(0)
        freqs = torch.einsum("b i , j -> b i j", positions.type_as(self.inv_freq), self.inv_freq)
        freqs = torch.stack((freqs, freqs), dim=-1).flatten(start_dim=-2)
        return freqs, 1.0

    def forward_from_seq_len(self, seq_len: int):
        return self.forward(torch.arange(seq_len, device=self.inv_freq.device))


def randomized_positions(
    batch: int,
    seq_len: int,
    gamma: float,
    device: torch.device,
) -> torch.Tensor:
    """Sorted samples of `seq_len` unique positions, each row drawn from its own range.

    Row `b` draws an upper bound `L_b ~ U[seq_len, seq_len * gamma]` and then takes
    `seq_len` distinct positions from `[0, L_b)`. Sorting keeps token order intact while
    the absolute indices — and so every relative distance the attention sees — span a
    range the raw sequence never would.

    `L_b == seq_len` yields contiguous positions, so the low end of the draw is exactly
    the un-augmented sequence. Every batch therefore carries the whole range of stretch
    factors at once, from the first update onward.

    `gamma <= 1` disables the augmentation.
    """
    if gamma <= 1.0:
        return torch.arange(seq_len, device=device).expand(batch, seq_len)

    max_len = math.ceil(seq_len * gamma)
    bounds = seq_len + (torch.rand(batch, device=device) * (max_len - seq_len)).long()

    # argsort of uniform noise is a permutation; pushing everything at or past a row's
    # own bound to +inf keeps it out of the first `seq_len` picks, so one kernel draws
    # `batch` independent samples from `batch` different ranges
    noise = torch.rand(batch, max_len, device=device)
    noise.masked_fill_(torch.arange(max_len, device=device)[None, :] >= bounds[:, None], float("inf"))
    positions = noise.argsort(dim=-1)[:, :seq_len]
    return positions.sort(dim=-1).values
