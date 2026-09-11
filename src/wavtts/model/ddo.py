"""Direct Discriminative Optimization (DDO) — arXiv:2503.01103.

The loss lives here; the plumbing that feeds it lives in CFM.forward. See
docs/superpowers/specs/2026-09-11-ddo-design.md for the derivation and for the
four places this port deliberately departs from the paper.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def load_ref_state_dict(ckpt_path: str) -> dict[str, torch.Tensor]:
    """The weights to freeze as p_ref, read out of a training checkpoint.

    EMA first: those are the weights inference ships and the weights the fake pool
    was generated from, and "the fakes came from p_ref" is the premise the whole
    likelihood-ratio identity rests on. A checkpoint saved by Trainer carries both,
    so the order here is a choice, not a fallback chain -- the fallbacks are for
    weight-only files (pretrained_*.pt, safetensors exports).
    """
    ckpt = torch.load(ckpt_path, weights_only=True, map_location="cpu", mmap=True)

    if "ema_model_state_dict" in ckpt:
        return {
            k.removeprefix("ema_model."): v
            for k, v in ckpt["ema_model_state_dict"].items()
            if k not in ("initted", "update", "step")
        }
    if "model_state_dict" in ckpt:
        return dict(ckpt["model_state_dict"])
    return dict(ckpt)


def ddo_loss(
    delta: torch.Tensor,  # [b] fp32; the log-ratio proxy, positive = theta likes this row more than ref
    is_fake: torch.Tensor,  # [b] bool; True = drawn from p_ref's sample pool
    *,
    alpha: float,
    beta: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """The DDO objective, Eq. 13 under the Jensen bound of Eq. 16.

    Transcribed from the paper's Appendix D reference implementation (spec appendix A):

        loss = -F.logsigmoid(beta * delta_real) - alpha * F.logsigmoid(-beta * delta_fake)

    `F.logsigmoid` rather than `log(1 - sigmoid(x))`: the latter loses every bit of
    precision once `sigmoid(x)` rounds to 1.0, which is exactly the saturated regime a
    badly-scaled beta lands in, and it turns a finite gradient into a NaN.

    `delta` must already be fp32 -- it is a difference of two nearly identical models'
    losses, and in bf16 that subtraction is catastrophic cancellation (spec S3.7). This
    function does not cast, so a bf16 delta stays bf16 and the tests can see it.
    """
    is_fake = is_fake.to(torch.bool)
    real, fake = delta[~is_fake], delta[is_fake]
    nan = delta.new_full((), float("nan"))

    # An empty side contributes a real zero of delta's own dtype/device: a python 0 would
    # make `loss` a CPU fp32 scalar under bf16 autocast on a cuda batch, and .backward()
    # on it would silently reach no parameter at all.
    loss_real = -F.logsigmoid(beta * real).mean() if real.numel() else delta.new_zeros(())
    loss_fake = -alpha * F.logsigmoid(-beta * fake).mean() if fake.numel() else delta.new_zeros(())

    # Gradient-scale normalisation from the official VAR trainer (spec S3.3): dividing by
    # max(alpha, 1) keeps the gradient magnitude roughly fixed as alpha sweeps, so an
    # alpha grid search does not double as an LR grid search.
    scale = max(float(alpha), 1.0)
    loss = (loss_real + loss_fake) / scale

    # The logged terms are post-scale, so total_loss == ddo/loss_real + ddo/loss_fake
    # (+ anchor_loss) holds in the trainer's curves and the three are directly comparable.
    stats = {
        "ddo/loss_real": (loss_real / scale).detach() if real.numel() else nan,
        "ddo/loss_fake": (loss_fake / scale).detach() if fake.numel() else nan,
        "ddo/delta_real": real.mean().detach() if real.numel() else nan,
        "ddo/delta_fake": fake.mean().detach() if fake.numel() else nan,
        # beta is calibrated off this one: beta ~ 1/delta_std puts beta*delta at O(1),
        # which is where logsigmoid still has gradient (spec S3.3).
        "ddo/delta_std": delta.std().detach() if delta.numel() > 1 else nan,
    }
    stats["ddo/margin"] = stats["ddo/delta_real"] - stats["ddo/delta_fake"]
    # The accuracy of the discriminator d_theta = sigmoid(beta*delta); target 0.6-0.75.
    # Each side contributes half, so a one-sided batch reports nan rather than a number
    # that looks like an accuracy but is only half of one. A delta of exactly 0 is a tie
    # and scores half a hit, not a miss: that is the state at theta == theta_ref, where
    # a strict `> 0` would log acc = 0 for a discriminator that is precisely at chance.
    acc_real = _hit_rate(real, 1.0) if real.numel() else nan
    acc_fake = _hit_rate(fake, -1.0) if fake.numel() else nan
    stats["ddo/acc"] = 0.5 * acc_real + 0.5 * acc_fake

    return loss, stats


def _hit_rate(delta: torch.Tensor, sign: float) -> torch.Tensor:
    """Fraction of rows the discriminator puts on the right side, ties counting a half."""
    signed = sign * delta
    return ((signed > 0).to(delta.dtype) + 0.5 * (signed == 0).to(delta.dtype)).mean().detach()
