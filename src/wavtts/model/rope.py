"""Randomized positional encoding, for length extrapolation.

Ruoss et al. (2023): training positions are `randperm(L_t)[:n].sort()` instead of
`arange(n)`, so a short clip still shows the model rotations it would only meet in a
long one. Training only — inference uses contiguous positions.

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

The stretch being *random* is what makes it an augmentation rather than a covariate
shift. A deterministic length-dependent stretch — a dynamic YaRN scale, say — gives
each clip length its own angle-per-frame map, which the model learns to depend on and
then meets unseen at any untrained length. Drawn fresh per sample, no such map exists
to depend on, and local physical timing falls to `ConvPositionEmbedding` (receptive
field 61 frames = 610 ms) while the rotary embedding carries coarse relative order.

Why no YaRN here: its every design choice — NTK-by-parts sparing the high-frequency
dims, the `0.1*ln(s) + 1` attention temperature, the short adaptation run — exists to
preserve structure a model already learned under a different spectrum. Training from
scratch there is nothing to preserve, so a fixed interpolated spectrum is just an
oddly-shaped one chosen for no reason, and for this data an actively bad one: over the
longest position the augmentation ever produces (4 * 2979 = 11916), vanilla base=10000
already leaves 5 of 32 dim pairs short of a single turn, and s=4 makes that 10 of 32.
Length generalization comes from the positions trained on, which is this file's job.

Output feeds `x_transformers.RotaryEmbedding.forward`, which takes `[b, n]` positions
directly and returns a `[b, n, d]` freqs that `apply_rotary_pos_emb` broadcasts.
"""

from __future__ import annotations

import math

import torch


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
