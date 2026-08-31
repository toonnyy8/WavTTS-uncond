"""Turn a finished run's checkpoint into a weights-only init for a new run.

Trainer.load_checkpoint() decides between full-state resume and weight-only init by
looking for an "update" key: with it, optimizer/scheduler/step all come back; without
it, it rebuilds the model from ema_model_state_dict and starts at update 0. So the
whole job here is dropping everything but the EMA weights.

    python scripts/make_pretrained_init.py <src_checkpoint.pt> <dst_ckpt_dir>

The output is named pretrained_*.pt, which the checkpoint rotation is already told to
leave alone.
"""

import os
import sys

import torch


def main(src, dst_dir):
    ckpt = torch.load(src, weights_only=True, map_location="cpu", mmap=True)
    ema = dict(ckpt["ema_model_state_dict"])

    # The EMA's own step rides along in its state dict, and at 200k+ the decay sits pinned
    # at beta (0.9999, ~100k updates of averaging with update_every=10). A run starting at
    # update 0 would then log samples from weights that ignore everything it just learned.
    # Zeroing it restarts the warmup ramp so the EMA tracks the new run from the beginning.
    ema["step"] = torch.zeros_like(ema["step"])
    ema["initted"] = torch.ones_like(ema["initted"])  # weights are real, not a blank init

    os.makedirs(dst_dir, exist_ok=True)
    out = os.path.join(dst_dir, "pretrained_" + os.path.basename(src))
    torch.save({"ema_model_state_dict": ema}, out)
    print(f"{out}  ({os.path.getsize(out) / 2**30:.2f} GiB, from update {ckpt.get('update')})")


if __name__ == "__main__":
    main(*sys.argv[1:])
