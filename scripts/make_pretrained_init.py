"""Turn a finished run's checkpoint into a weights-only init for a new run.

Trainer.load_checkpoint() decides between full-state resume and weight-only init by
looking for an "update" key: with it, optimizer/scheduler/step all come back; without
it, it rebuilds the model from ema_model_state_dict and starts at update 0. So the
whole job here is dropping everything but the EMA weights.

    python scripts/make_pretrained_init.py <src_checkpoint.pt> <dst_ckpt_dir> [config.yaml]

Give it a config and the EMA is filtered down to the keys that config's model actually
has. That is what lets a run drop a submodule the source had: turning use_aux_mel_loss
off removes CFM.aux_mel_loss, whose STFT windows and mel filterbanks are registered
buffers and so ride along in the source state dict. Trainer.load_checkpoint() loads
strictly, so those 14 leftover tensors abort the load. They are constants, not anything
the source learned, and dropping them costs nothing.

The output is named pretrained_*.pt, which the checkpoint rotation is already told to
leave alone.
"""

import os
import sys

import torch


def target_keys(config_name):
    """The state dict keys of the model the destination config describes."""
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
    return set(model.state_dict())


def main(src, dst_dir, config_name=None):
    ckpt = torch.load(src, weights_only=True, map_location="cpu", mmap=True)
    ema = dict(ckpt["ema_model_state_dict"])

    if config_name is not None:
        keys = target_keys(config_name)
        # "initted"/"step" are the EMA wrapper's own, not the wrapped model's
        extra = [k for k in ema if k not in ("initted", "step") and k.removeprefix("ema_model.") not in keys]
        missing = [k for k in keys if f"ema_model.{k}" not in ema]
        if missing:
            raise SystemExit(
                f"{config_name} needs {len(missing)} tensors the source does not have, "
                f"e.g. {missing[:3]} — these two models are not the same architecture"
            )
        for k in extra:
            del ema[k]
        print(f"dropped {len(extra)} tensors absent from {config_name}: {extra[:2]} ...")

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
