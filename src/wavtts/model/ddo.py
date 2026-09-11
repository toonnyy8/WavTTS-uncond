"""Direct Discriminative Optimization (DDO) — arXiv:2503.01103.

The loss lives here; the plumbing that feeds it lives in CFM.forward. See
docs/superpowers/specs/2026-09-11-ddo-design.md for the derivation and for the
four places this port deliberately departs from the paper.
"""

from __future__ import annotations

import torch


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
