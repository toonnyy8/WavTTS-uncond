# training script.

import os
from importlib.resources import files

import hydra
from omegaconf import OmegaConf

from wavtts.model import CFM, Trainer
from wavtts.model.dataset import load_dataset, load_ddo_dataset
from wavtts.model.ddo import load_ref_state_dict
from wavtts.model.utils import seed_everything


os.chdir(str(files("wavtts").joinpath("../..")))  # change working directory to root of project (local editable)


@hydra.main(version_base="1.3", config_path=str(files("wavtts").joinpath("configs")), config_name=None)
def main(model_cfg):
    seed = int(model_cfg.get("seed", 666))
    seed_everything(seed)  # run-level reproducibility (also sets cudnn deterministic)

    model_cls = hydra.utils.get_class(f"wavtts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch
    cfm_kwargs = getattr(model_cfg.model, "cfm", {}) or {}

    exp_name = model_cfg.ckpts.exp_name
    wandb_resume_id = None

    # set model
    model = CFM(
        transformer=model_cls(**model_arc, wav_frame_len=model_cfg.model.waveform.wav_frame_len),
        waveform_kwargs=model_cfg.model.waveform,
        **cfm_kwargs,
    )

    # DDO finetuning (docs/superpowers/specs/2026-09-11-ddo-design.md). The block is
    # optional and absent for every pretraining config, in which case not one line below
    # this point behaves differently from before.
    ddo_cfg = model_cfg.get("ddo", None)
    if ddo_cfg is not None:
        # p_ref is built from the *same* arch / cfm / waveform config as theta. Any
        # asymmetry between the two -- a different dropout rate, loss space or t schedule --
        # lands in Delta as an offset present on every row, which is a feature the
        # discriminator can separate real from fake on without ever improving a sample.
        ref = CFM(
            transformer=model_cls(**model_arc, wav_frame_len=model_cfg.model.waveform.wav_frame_len),
            waveform_kwargs=model_cfg.model.waveform,
            **cfm_kwargs,
        )
        ref.load_state_dict(load_ref_state_dict(ddo_cfg.ref_ckpt))
        model.attach_ddo_ref(
            ref,
            alpha=ddo_cfg.alpha,
            beta=ddo_cfg.beta,
            delta_normalize=ddo_cfg.get("delta_normalize", "mean"),
            anchor_weight=ddo_cfg.get("anchor_weight", 1.0),
        )
        # Recorded, not plumbed: this study runs no CFG at all, so the fake pool is drawn
        # guidance-free and the only correct value is 0. It lives in the config so the
        # checkpoint's logged run config states which pool it was trained against
        # (spec S3.4, S3.5).
        if float(ddo_cfg.get("fake_cfg_strength", 0.0)) != 0.0:
            print(
                "WavTTS WARNING: ddo.fake_cfg_strength is non-zero. Fakes generated with "
                "guidance are not samples of p_ref, and the likelihood-ratio identity DDO "
                "rests on does not hold for them."
            )

    save_dir = model_cfg.ckpts.save_dir
    if os.path.isabs(save_dir):
        checkpoint_path = save_dir
    else:
        checkpoint_path = str(files("wavtts").joinpath(f"../../{save_dir}"))

    if ddo_cfg is not None:
        # theta has to start at p_ref, and Trainer.load_checkpoint only makes that happen if
        # save_dir already holds the pretrained_*.pt that make_pretrained_init.py writes. An
        # empty save_dir would train a randomly initialised theta against a pretrained
        # reference: Delta is then all initialisation gap, and nothing raises.
        has_init = os.path.isdir(checkpoint_path) and any(
            f.endswith((".pt", ".safetensors")) for f in os.listdir(checkpoint_path)
        )
        if not has_init:
            raise SystemExit(
                f"DDO: {checkpoint_path} holds no checkpoint, so theta would not start at p_ref. Run\n"
                f"  uv run python scripts/make_pretrained_init.py {ddo_cfg.ref_ckpt} {checkpoint_path}\n"
                "first (a new round needs its own model.name, so this directory is fresh)."
            )

    # init trainer
    trainer = Trainer(
        model,
        epochs=model_cfg.optim.epochs,
        learning_rate=model_cfg.optim.learning_rate,
        num_warmup_updates=model_cfg.optim.num_warmup_updates,
        save_per_updates=model_cfg.ckpts.save_per_updates,
        keep_last_n_checkpoints=model_cfg.ckpts.keep_last_n_checkpoints,
        checkpoint_path=checkpoint_path,
        batch_size_per_gpu=model_cfg.datasets.batch_size_per_gpu,
        batch_size_type=model_cfg.datasets.batch_size_type,
        max_samples=model_cfg.datasets.max_samples,
        grad_accumulation_steps=model_cfg.optim.grad_accumulation_steps,
        max_grad_norm=model_cfg.optim.max_grad_norm,
        logger=model_cfg.ckpts.logger,
        wandb_project="WavTTS",
        wandb_run_name=exp_name,
        wandb_resume_id=wandb_resume_id,
        last_per_updates=model_cfg.ckpts.last_per_updates,
        # both optional and absent from every pretraining config: a DDO round stops on a
        # fixed update count, and needs a far shorter EMA than the ema_pytorch defaults
        # (spec S3.8)
        max_updates=model_cfg.optim.get("max_updates", None),
        lr_decay_end_factor=model_cfg.optim.get("lr_decay_end_factor", 1e-8),
        ema_kwargs=OmegaConf.to_container(model_cfg.optim.get("ema_kwargs", {}) or {}, resolve=True),
        log_per_updates=model_cfg.ckpts.get("log_per_updates", 1),
        log_samples=model_cfg.ckpts.log_samples,
        log_samples_seeds=model_cfg.ckpts.log_samples_seeds,
        log_samples_secs=model_cfg.ckpts.log_samples_secs,
        spk_ckpt_path=model_cfg.ckpts.spk_ckpt_path,
        bnb_optimizer=model_cfg.optim.bnb_optimizer,
        model_cfg_dict=OmegaConf.to_container(model_cfg, resolve=True),
    )

    if ddo_cfg is not None:
        # Real corpus + offline fake pool as one dataset; the repeat count that pulls the
        # per-batch real:fake frame ratio to real_fake_ratio is computed in there.
        train_dataset = load_ddo_dataset(
            model_cfg.datasets.name,
            ddo_cfg.fake_dataset,
            waveform_kwargs=model_cfg.model.waveform,
            real_fake_ratio=ddo_cfg.get("real_fake_ratio", 1.0),
        )
    else:
        train_dataset = load_dataset(model_cfg.datasets.name, waveform_kwargs=model_cfg.model.waveform)
    trainer.train(
        train_dataset,
        num_workers=model_cfg.datasets.num_workers,
        resumable_with_seed=seed,  # seed for shuffling dataset
    )


if __name__ == "__main__":
    main()
