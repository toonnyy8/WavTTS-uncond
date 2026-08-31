"""Turn a finished run's checkpoint into a weights-only init for a new run.

Trainer.load_checkpoint() decides between full-state resume and weight-only init by
looking for an "update" key: with it, optimizer/scheduler/step all come back; without
it, it rebuilds the model from ema_model_state_dict and starts at update 0. So the
whole job here is dropping everything but the EMA weights.

    python scripts/make_pretrained_init.py <src_checkpoint.pt> <dst_ckpt_dir> [config.yaml]

Give it a config and the weights are also reshaped to fit that config's model, which is
what lets a run with a different wav_frame_len start from an existing one. Without it the
weights are copied as they are.

The output is named pretrained_*.pt, which the checkpoint rotation is already told to
leave alone.
"""

import os
import sys

import torch


def widen(old: torch.Tensor, shape) -> torch.Tensor:
    """Grow one axis of a frame-shaped weight, keeping the old values at the tail.

    Only wav_frame_len moves: it is the input width of input_embed.x_proj and the output
    width of proj_out. Widening it means each token now covers earlier samples too --
    frame_len 160 -> 320 at hop 160 turns token t's input from frame t into frames
    t-1 and t concatenated -- so the samples the old weights were trained on sit at the
    END of the new frame, and the new span is the zeros in front. Same on the way out:
    the tail of the 320-wide prediction is the 160 samples the old head predicted, and
    the head that covers the earlier half starts at zero because nothing in the old model
    ever predicted a frame from the token after it.

    Overlap-add then reconstructs those samples as w[160+j] * old_prediction, since the
    other contribution is the zeroed half and the two window halves sum to 1. The output
    starts out ramped down rather than exact -- unavoidable, one of the two overlapping
    frames has no trained head -- but every one of the 664M backbone parameters carries
    over untouched, which is the part worth keeping.
    """
    if tuple(old.shape) == tuple(shape):
        return old
    axes = [i for i, (a, b) in enumerate(zip(old.shape, shape)) if a != b]
    if len(axes) != 1 or shape[axes[0]] < old.shape[axes[0]]:
        raise ValueError(f"cannot widen {tuple(old.shape)} to {tuple(shape)}")
    ax = axes[0]
    out = old.new_zeros(shape)
    index = [slice(None)] * old.ndim
    index[ax] = slice(shape[ax] - old.shape[ax], None)
    out[tuple(index)] = old
    return out


def target_shapes(config_name):
    import hydra
    from omegaconf import OmegaConf

    from wavtts.model import CFM

    path = config_name
    if not os.path.exists(path):
        path = f"src/wavtts/configs/{config_name}"
    cfg = OmegaConf.load(path)
    model_cls = hydra.utils.get_class(f"wavtts.model.{cfg.model.backbone}")
    model = CFM(
        transformer=model_cls(**cfg.model.arch, wav_frame_len=cfg.model.waveform.wav_frame_len),
        waveform_kwargs=cfg.model.waveform,
        **(cfg.model.cfm or {}),
    )
    return {k: tuple(v.shape) for k, v in model.state_dict().items()}


def main(src, dst_dir, config_name=None):
    ckpt = torch.load(src, weights_only=True, map_location="cpu", mmap=True)
    ema = dict(ckpt["ema_model_state_dict"])

    if config_name is not None:
        shapes = target_shapes(config_name)
        for k in ema:
            if k in ("initted", "step"):
                continue
            shape = shapes[k.removeprefix("ema_model.")]
            if tuple(ema[k].shape) != shape:
                print(f"widen {k}: {tuple(ema[k].shape)} -> {shape}")
            ema[k] = widen(ema[k], shape)

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
