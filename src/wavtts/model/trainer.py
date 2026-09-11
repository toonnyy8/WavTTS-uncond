from __future__ import annotations

import gc
import math
import os
import random

import torch
import torchaudio
import wandb
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from ema_pytorch import EMA
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from tqdm import tqdm

from wavtts.model import CFM
from wavtts.model.dataset import DynamicBatchSampler, collate_fn
from wavtts.model.utils import default, exists, peak_normalize


def pair_sample_lengths(seeds: list[int], secs: list[float] | None) -> list[float]:
    """One clip length per seed. A short list pads with its last value, so a config
    that gives a single length still yields a length for every seed."""
    out = list(secs) if secs else [5.0]
    if len(out) < len(seeds):
        out = out + [out[-1]] * (len(seeds) - len(out))
    return out[: len(seeds)]


# trainer


class Trainer:
    @staticmethod
    def _format_params_in_mb(count: int) -> str:
        return f"{count / 1e6:,.3f}M / {count / 1e9:,.6f}B"

    @staticmethod
    def _count_model_params(model: torch.nn.Module) -> tuple[int, int, int]:
        total = 0
        trainable = 0
        frozen = 0
        for param in model.parameters():
            numel = param.numel()
            total += numel
            if param.requires_grad:
                trainable += numel
            else:
                frozen += numel
        return total, trainable, frozen

    def _print_model_param_summary(self, model: torch.nn.Module):
        total, trainable, frozen = self._count_model_params(model)
        trainable_ratio = (trainable / total * 100.0) if total > 0 else 0.0
        frozen_ratio = (frozen / total * 100.0) if total > 0 else 0.0

        table_width = 76
        print("\n" + "=" * table_width)
        print("Model Parameter Summary".center(table_width))
        print("-" * table_width)
        print(f"{'Type':<12}{'Count':>18}{'Scale (M / B)':>30}{'Ratio':>16}")
        print("-" * table_width)
        print(
            f"{'Total':<12}{total:>18,}{self._format_params_in_mb(total):>30}{'100.00%':>16}"
        )
        print(
            f"{'Trainable':<12}{trainable:>18,}{self._format_params_in_mb(trainable):>30}{f'{trainable_ratio:.2f}%':>16}"
        )
        print(
            f"{'Frozen':<12}{frozen:>18,}{self._format_params_in_mb(frozen):>30}{f'{frozen_ratio:.2f}%':>16}"
        )
        print("=" * table_width)

    def __init__(
        self,
        model: CFM,
        epochs,
        learning_rate,
        num_warmup_updates=20000,
        max_updates: int | None = None,  # hard stop in updates; None = run `epochs` passes
        save_per_updates=1000,
        keep_last_n_checkpoints: int = -1,  # -1 to keep all, 0 to not save intermediate, > 0 to keep last N checkpoints
        checkpoint_path=None,
        batch_size_per_gpu=32,
        batch_size_type: str = "sample",
        max_samples=32,
        grad_accumulation_steps=1,
        max_grad_norm=1.0,
        logger: str | None = "wandb",  # "wandb" | "tensorboard" | None
        wandb_project="test_wavtts",
        wandb_run_name="test_run",
        wandb_resume_id: str = None,
        log_per_updates: int = 1,  # scalar logging interval (loss/lr), in updates
        log_samples: bool = False,
        log_samples_seeds: list[int] | None = None,  # fixed seeds: clips comparable across checkpoints
        log_samples_secs: list[float] | None = None,  # one length per seed; the ladder past
        # the training maximum is what makes extrapolation failure visible
        spk_ckpt_path: str | None = None,  # ECAPA ckpt for gen/spk_sim_self; None disables
        last_per_updates=None,
        accelerate_kwargs: dict = dict(),
        ema_kwargs: dict = dict(),
        bnb_optimizer: bool = False,
        model_cfg_dict: dict = dict(),  # training config
    ):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        if logger == "wandb" and not wandb.api.api_key:
            logger = None
        self.log_per_updates = max(1, int(log_per_updates))
        self.log_samples = log_samples
        self.log_samples_seeds = list(log_samples_seeds) if log_samples_seeds is not None else [0, 1, 2, 3]
        self.log_samples_secs = pair_sample_lengths(self.log_samples_seeds, log_samples_secs)
        self.spk_ckpt_path = spk_ckpt_path

        self.accelerator = Accelerator(
            log_with=logger if logger == "wandb" else None,
            kwargs_handlers=[ddp_kwargs],
            gradient_accumulation_steps=grad_accumulation_steps,
            **accelerate_kwargs,
        )

        self.logger = logger
        if self.logger == "wandb":
            if exists(wandb_resume_id):
                init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name, "id": wandb_resume_id}}
            else:
                init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name}}

            if not model_cfg_dict:
                model_cfg_dict = {
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "num_warmup_updates": num_warmup_updates,
                    "batch_size_per_gpu": batch_size_per_gpu,
                    "batch_size_type": batch_size_type,
                    "max_samples": max_samples,
                    "grad_accumulation_steps": grad_accumulation_steps,
                    "max_grad_norm": max_grad_norm,
                }
            model_cfg_dict["gpus"] = self.accelerator.num_processes
            self.accelerator.init_trackers(
                project_name=wandb_project,
                init_kwargs=init_kwargs,
                config=model_cfg_dict,
            )

        elif self.logger == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=f"runs/{wandb_run_name}")

        self.model = model

        if self.is_main:
            self.ema_model = EMA(model, include_online_model=False, **ema_kwargs)
            # EMA deepcopies the model, and a one-element list *is* copied by deepcopy --
            # what keeps p_ref invisible is nn.Module.__setattr__ declining to register it,
            # not deepcopy declining to look. So the copy silently carries a second frozen
            # p_ref (bf16, ~1.33 GiB on the large arch) that nothing will ever read: the EMA
            # weights are only used for sample(), which never goes down the DDO path. Drop
            # it here rather than giving CFM a __deepcopy__ -- that would change what copying
            # a CFM means for every other caller in order to fix one of them (spec S3.8).
            if hasattr(self.ema_model.ema_model, "_ddo_ref"):
                self.ema_model.ema_model._ddo_ref = []
            self.ema_model.to(self.accelerator.device)
            self._print_model_param_summary(self.model)

            print(f"Using logger: {logger}")
            if grad_accumulation_steps > 1:
                print(
                    "Gradient accumulation checkpointing with per_updates now, old logic per_steps used with before f992c4e"
                )

        self.epochs = epochs
        self.num_warmup_updates = num_warmup_updates
        self.max_updates = max_updates
        self.save_per_updates = save_per_updates
        self.keep_last_n_checkpoints = keep_last_n_checkpoints
        self.last_per_updates = default(last_per_updates, save_per_updates)
        self.checkpoint_path = default(checkpoint_path, "ckpts/test_wavtts")

        self.batch_size_per_gpu = batch_size_per_gpu
        self.batch_size_type = batch_size_type
        self.max_samples = max_samples
        self.grad_accumulation_steps = grad_accumulation_steps
        self.max_grad_norm = max_grad_norm

        if bnb_optimizer:
            import bitsandbytes as bnb

            self.optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=learning_rate)
        else:
            self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)

        # p_ref lives in a one-element list on the CFM (spec S3.9), so it is not in the
        # module tree and neither .to() nor accelerator.prepare() reaches it -- the model
        # would move to cuda while the reference stayed on cpu, and the first DDO batch
        # would die on a device mismatch. Moved once here, after prepare has settled which
        # device this process owns and before any batch can arrive.
        ref = self.accelerator.unwrap_model(self.model).ddo_ref
        if ref is not None:
            ref.to(self.accelerator.device)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @staticmethod
    def _scalar_logs(loss_dict: dict) -> dict[str, float]:
        """Every finite scalar in loss_dict, keyed as it should appear in the logs.

        The trainer used to name the two keys pretraining emits. The DDO path emits eleven,
        and which of them carry a number depends on what the batch contained -- a step whose
        sampler handed out only real rows has no ddo/loss_fake, and a clean-arm run has no
        anchor_loss ever. So the dict is walked instead of indexed, and nan (the loss's way
        of saying "no rows of this kind this step") is skipped rather than written: a single
        nan poisons a tensorboard curve's autoscale for the whole run.

        total_loss is logged as "loss" so the curve stays continuous across the
        pretraining -> DDO switch; it is exactly the quantity backward() was called on.
        """
        out: dict[str, float] = {}
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                if value.ndim != 0:
                    continue
                value = value.item()
            elif not isinstance(value, (int, float)):
                continue
            value = float(value)
            if math.isnan(value):
                continue
            out["loss" if key == "total_loss" else key] = value
        return out

    def save_checkpoint(self, update, last=False):
        self.accelerator.wait_for_everyone()
        if self.is_main:
            checkpoint = dict(
                model_state_dict=self.accelerator.unwrap_model(self.model).state_dict(),
                optimizer_state_dict=self.optimizer.state_dict(),
                ema_model_state_dict=self.ema_model.state_dict(),
                scheduler_state_dict=self.scheduler.state_dict(),
                update=update,
                # resume picks up the same noise/t/mixing draws instead of replaying
                # the run's opening seed. weights_only=True load allows tensors and
                # plain tuples, so keep it to these; numpy is unused in this loop.
                # ponytail: rank-0 RNG only — seed_everything gives every rank the
                # same seed anyway, so restoring it everywhere changes nothing.
                rng_state=dict(
                    python=random.getstate(),
                    torch=torch.get_rng_state(),
                    cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                ),
            )
            if not os.path.exists(self.checkpoint_path):
                os.makedirs(self.checkpoint_path)
            if last:
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_last.pt")
                print(f"Saved last checkpoint at update {update}")
            else:
                if self.keep_last_n_checkpoints == 0:
                    return
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_{update}.pt")
                if self.keep_last_n_checkpoints > 0:
                    # Updated logic to exclude pretrained model from rotation
                    checkpoints = [
                        f
                        for f in os.listdir(self.checkpoint_path)
                        if f.startswith("model_")
                        and not f.startswith("pretrained_")  # Exclude pretrained models
                        and f.endswith(".pt")
                        and f != "model_last.pt"
                    ]
                    checkpoints.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
                    while len(checkpoints) > self.keep_last_n_checkpoints:
                        oldest_checkpoint = checkpoints.pop(0)
                        os.remove(os.path.join(self.checkpoint_path, oldest_checkpoint))
                        print(f"Removed old checkpoint: {oldest_checkpoint}")

    def load_checkpoint(self):
        if (
            not exists(self.checkpoint_path)
            or not os.path.exists(self.checkpoint_path)
            or not any(filename.endswith((".pt", ".safetensors")) for filename in os.listdir(self.checkpoint_path))
        ):
            return 0

        self.accelerator.wait_for_everyone()
        if "model_last.pt" in os.listdir(self.checkpoint_path):
            latest_checkpoint = "model_last.pt"
        else:
            # Updated to consider pretrained models for loading but prioritize training checkpoints
            all_checkpoints = [
                f
                for f in os.listdir(self.checkpoint_path)
                if (f.startswith("model_") or f.startswith("pretrained_")) and f.endswith((".pt", ".safetensors"))
            ]

            # First try to find regular training checkpoints
            training_checkpoints = [f for f in all_checkpoints if f.startswith("model_") and f != "model_last.pt"]
            if training_checkpoints:
                latest_checkpoint = sorted(
                    training_checkpoints,
                    key=lambda x: int("".join(filter(str.isdigit, x))),
                )[-1]
            else:
                # If no training checkpoints, use pretrained model
                latest_checkpoint = next(f for f in all_checkpoints if f.startswith("pretrained_"))

        if latest_checkpoint.endswith(".safetensors"):  # always a pretrained checkpoint
            from safetensors.torch import load_file

            checkpoint = load_file(f"{self.checkpoint_path}/{latest_checkpoint}", device="cpu")
            checkpoint = {"ema_model_state_dict": checkpoint}
        elif latest_checkpoint.endswith(".pt"):
            # checkpoint = torch.load(f"{self.checkpoint_path}/{latest_checkpoint}", map_location=self.accelerator.device)  # rather use accelerator.load_state ಥ_ಥ
            checkpoint = torch.load(
                f"{self.checkpoint_path}/{latest_checkpoint}", weights_only=True, map_location="cpu"
            )

        if self.is_main:
            print("=" * 80)
            print("Resume training: found checkpoint and loading state")
            print(f"Checkpoint directory: {self.checkpoint_path}")
            print(f"Checkpoint file: {latest_checkpoint}")
            print("=" * 80)

        if self.is_main:
            self.ema_model.load_state_dict(checkpoint["ema_model_state_dict"])

        if "update" in checkpoint or "step" in checkpoint:
            # patch for backward compatibility, with before f992c4e
            if "step" in checkpoint:
                checkpoint["update"] = checkpoint["step"] // self.grad_accumulation_steps
                if self.grad_accumulation_steps > 1 and self.is_main:
                    print(
                        "WavTTS WARNING: Loading checkpoint saved with per_steps logic (before f992c4e), will convert to per_updates according to grad_accumulation_steps setting, may have unexpected behaviour."
                    )
            self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if self.scheduler:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            update = checkpoint["update"]
            rng_state = checkpoint.get("rng_state")  # absent in checkpoints saved before this was added
            if rng_state is not None:
                random.setstate(tuple(rng_state["python"]))
                torch.set_rng_state(rng_state["torch"])
                if rng_state["cuda"] is not None and torch.cuda.is_available():
                    # One state per device was saved, so a checkpoint from a 4-gpu run has 4 of
                    # them and set_rng_state_all indexes straight past the end of a 2-gpu box --
                    # a hard crash on what is otherwise a perfectly good resume. Restore what
                    # maps onto the devices that exist; any device with no saved state keeps the
                    # one seed_everything already gave it. Only the per-device noise draws ride
                    # on this, and those are already unreproducible across a gpu-count change.
                    saved = list(rng_state["cuda"])
                    for i in range(min(len(saved), torch.cuda.device_count())):
                        torch.cuda.set_rng_state(saved[i], i)
                    if self.is_main and len(saved) != torch.cuda.device_count():
                        print(
                            f"WavTTS: checkpoint carries {len(saved)} cuda rng states, this run has "
                            f"{torch.cuda.device_count()} device(s); restored the overlap"
                        )
            if self.is_main:
                print(
                    f"Resume mode: full-state resume (model + optimizer + scheduler"
                    f"{' + rng' if rng_state is not None else ''}), update={update}"
                )
        else:
            checkpoint["model_state_dict"] = {
                k.replace("ema_model.", ""): v
                for k, v in checkpoint["ema_model_state_dict"].items()
                if k not in ["initted", "update", "step"]
            }
            self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"])
            update = 0
            if self.is_main:
                print("Resume mode: pretrained-weight init only, update=0")

        del checkpoint
        gc.collect()
        return update

    def train(self, train_dataset: Dataset, num_workers=16, resumable_with_seed: int = None):
        if self.log_samples:
            from wavtts.train.metrics import GenMetrics

            target_sample_rate = train_dataset.target_sample_rate
            log_samples_path = f"{self.checkpoint_path}/samples"
            os.makedirs(log_samples_path, exist_ok=True)
            gen_metrics = GenMetrics(sample_rate=target_sample_rate, spk_ckpt_path=self.spk_ckpt_path)

        if exists(resumable_with_seed):
            generator = torch.Generator()
            generator.manual_seed(resumable_with_seed)
        else:
            generator = None

        if self.batch_size_type == "sample":
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True,
                batch_size=self.batch_size_per_gpu,
                shuffle=True,
                generator=generator,
            )
        elif self.batch_size_type == "frame":
            self.accelerator.even_batches = False
            sampler = SequentialSampler(train_dataset)
            batch_sampler = DynamicBatchSampler(
                sampler,
                self.batch_size_per_gpu,
                max_samples=self.max_samples,
                random_seed=resumable_with_seed,  # This enables reproducible shuffling
                drop_residual=False,
            )
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True,
                batch_sampler=batch_sampler,
            )
        else:
            raise ValueError(f"batch_size_type must be either 'sample' or 'frame', but received {self.batch_size_type}")

        #  accelerator.prepare() dispatches batches to devices;
        #  which means the length of dataloader calculated before, should consider the number of devices
        warmup_updates = (
            self.num_warmup_updates * self.accelerator.num_processes
        )  # consider a fixed warmup steps while using accelerate multi-gpu ddp
        # otherwise by default with split_batches=False, warmup steps change with num_processes
        if exists(self.max_updates):
            # A DDO round is a few thousand updates and stops on max_updates, long before
            # `epochs` passes are done -- and its dataloader is *longer* than pretraining's
            # because the fake pool is concatenated in, so deriving the horizon from epochs
            # would put the floor tens of epochs away and leave the LR essentially flat for
            # the whole round. max_updates counts optimizer steps, so it takes the same
            # num_processes factor warmup_updates does, for the reason spelled out below.
            total_updates = self.max_updates * self.accelerator.num_processes
        else:
            # No num_processes factor here, and the reason is easy to get backwards. The
            # wrapped scheduler is stepped num_processes times per optimizer step, so the
            # horizon does have to be in those units -- but len(train_dataloader) is still the
            # GLOBAL batch count at this point: prepare() below is what rebinds the name to
            # the per-process shard. Global batches / accumulation IS internal steps per epoch
            # already. Multiplying again stretches the decay num_processes-fold and the LR
            # never reaches its floor.
            total_updates = math.ceil(len(train_dataloader) / self.grad_accumulation_steps) * self.epochs
        # clamped, because a horizon at or below the warmup makes LinearLR divide by its
        # own total_iters of zero and take the run down on the first step. That is reachable
        # by config now that max_updates exists (a 300-update beta probe against the config's
        # 200 warmup updates is a sane thing to ask for), and a one-step decay is a more
        # useful answer than a traceback.
        decay_updates = max(1, total_updates - warmup_updates)
        warmup_scheduler = LinearLR(self.optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_updates)
        decay_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_updates)
        self.scheduler = SequentialLR(
            self.optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_updates]
        )
        train_dataloader, self.scheduler = self.accelerator.prepare(
            train_dataloader, self.scheduler
        )  # actual multi_gpu updates = single_gpu updates / gpu nums
        start_update = self.load_checkpoint()
        global_update = start_update

        if exists(resumable_with_seed):
            orig_epoch_step = len(train_dataloader)
            start_step = start_update * self.grad_accumulation_steps
            skipped_epoch = int(start_step // orig_epoch_step)
            skipped_batch = start_step % orig_epoch_step
            skipped_dataloader = self.accelerator.skip_first_batches(train_dataloader, num_batches=skipped_batch)
        else:
            skipped_epoch = 0

        for epoch in range(skipped_epoch, self.epochs):
            self.model.train()
            if exists(resumable_with_seed) and epoch == skipped_epoch:
                progress_bar_initial = math.ceil(skipped_batch / self.grad_accumulation_steps)
                current_dataloader = skipped_dataloader
            else:
                progress_bar_initial = 0
                current_dataloader = train_dataloader

            # Set epoch for the batch sampler if it exists
            if hasattr(train_dataloader, "batch_sampler") and hasattr(train_dataloader.batch_sampler, "set_epoch"):
                train_dataloader.batch_sampler.set_epoch(epoch)

            progress_bar = tqdm(
                range(math.ceil(len(train_dataloader) / self.grad_accumulation_steps)),
                desc=f"Epoch {epoch + 1}/{self.epochs}",
                unit="update",
                disable=not self.accelerator.is_local_main_process,
                initial=progress_bar_initial,
            )

            for batch in current_dataloader:
                with self.accelerator.accumulate(self.model):
                    wav = batch["wav"]
                    wav_lengths = batch["wav_lengths"]

                    # is_fake is present only when the dataset is a DDO tagged concat; a
                    # missing key is None, which CFM.forward reads as "every row is real".
                    loss, loss_dict = self.model(wav, lens=wav_lengths, is_fake=batch.get("is_fake"))
                    self.accelerator.backward(loss)

                    if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    if self.is_main:
                        self.ema_model.update()

                    global_update += 1
                    progress_bar.update(1)
                    # Nothing is indexed unconditionally: which terms a step produces depends
                    # on the objective and on what the batch happened to contain (no fake
                    # rows this step means no ddo/loss_fake), so the postfix shows whichever
                    # of these the dict actually carries.
                    scalars = self._scalar_logs(loss_dict)
                    postfix = {"update": str(global_update), "loss": scalars.get("loss", loss.item())}
                    for key in ("flow_loss", "aux_mel_loss", "anchor_loss", "ddo/margin", "ddo/acc"):
                        if key in scalars:
                            postfix[key] = scalars[key]
                    progress_bar.set_postfix(postfix)

                if self.accelerator.is_local_main_process and global_update % self.log_per_updates == 0:
                    logs = {"lr": self.scheduler.get_last_lr()[0]}
                    logs.update(self._scalar_logs(loss_dict))
                    logs.setdefault("loss", loss.item())  # a loss_dict without total_loss
                    self.accelerator.log(logs, step=global_update)

                    if self.logger == "tensorboard":
                        for key, value in logs.items():
                            self.writer.add_scalar(key, value, global_update)

                if global_update % self.last_per_updates == 0 and self.accelerator.sync_gradients:
                    self.save_checkpoint(global_update, last=True)

                if global_update % self.save_per_updates == 0 and self.accelerator.sync_gradients:
                    self.save_checkpoint(global_update)

                # samples and metrics ride along with model_last.pt, so quality is visible
                # at every last-checkpoint interval, not only at the numbered ones
                if (
                    global_update % self.last_per_updates == 0
                    and self.accelerator.sync_gradients
                    and self.log_samples
                    and self.accelerator.is_local_main_process
                ):
                    from wavtts.train.metrics import clipping_rate, mel_figure, rms, silence_ratio

                    # sample from the EMA weights: they are what inference ships, and they
                    # move smoothly enough that a fixed seed's clip stays comparable across
                    # checkpoints instead of jumping with every batch
                    gen_model = self.ema_model.ema_model
                    torch.cuda.empty_cache()  # the long clips want every spare block

                    # one length per seed, held fixed across checkpoints: a length ladder
                    # past the 30 s training maximum is what makes extrapolation failure
                    # visible, and per-length metrics keep a collapsing 60 s clip from
                    # being averaged away by a healthy 5 s one
                    gen_audios = {}
                    with torch.inference_mode():
                        for gen_seed, gen_sec in zip(self.log_samples_seeds, self.log_samples_secs):
                            try:
                                generated, _ = gen_model.sample(
                                    duration=int(gen_sec * target_sample_rate),
                                    steps=32,
                                    cfg_strength=2.0,
                                    sway_sampling_coef=-1.0,
                                    seed=gen_seed,
                                )
                            except torch.cuda.OutOfMemoryError:
                                # the longest clip is the one that can fail, and losing it
                                # must not take the run down with it
                                print(f"\nWARNING: skipped {gen_sec}s sample at update {global_update}, OOM")
                                torch.cuda.empty_cache()
                                continue
                            gen_audios[(gen_seed, gen_sec)] = generated.to(torch.float32).cpu()  # [1, N_gen]
                    torch.cuda.empty_cache()

                    keys = ("utmos", "silence_ratio", "clipping_rate", "rms", "spk_sim_self")
                    scores = {k: [] for k in keys}
                    for (gen_seed, gen_sec), gen_audio in gen_audios.items():
                        tag = f"{gen_sec:g}s_seed{gen_seed}"
                        # the model generates at target_rms, which is well outside +-1.
                        # Everything meant for a human — the wav file, the TB player, the
                        # mel plot's colour scale — gets the scaled copy; the metrics get
                        # the raw output, so gen/rms stays the one place level is read.
                        file_audio = peak_normalize(gen_audio)
                        torchaudio.save(
                            f"{log_samples_path}/update_{global_update}_{tag}.wav",
                            file_audio,
                            target_sample_rate,
                        )
                        wav_1d = gen_audio[0]
                        clip = {
                            "silence_ratio": silence_ratio(wav_1d),
                            "clipping_rate": clipping_rate(wav_1d),
                            "rms": rms(wav_1d),
                            "utmos": gen_metrics.utmos(wav_1d, self.accelerator.device),
                            "spk_sim_self": gen_metrics.spk_sim_self(wav_1d, self.accelerator.device),
                        }
                        clip_log = {f"gen_{gen_sec:g}s/{k}": v for k, v in clip.items() if v is not None}
                        for k, v in clip.items():
                            if v is not None:
                                scores[k].append(v)

                        self.accelerator.log(clip_log, step=global_update)
                        if self.logger == "tensorboard":
                            for k, v in clip_log.items():
                                self.writer.add_scalar(k, v, global_update)
                            self.writer.add_audio(
                                f"gen/audio_{tag}", file_audio, global_update, sample_rate=target_sample_rate
                            )
                            self.writer.add_figure(
                                f"gen/mel_{tag}", mel_figure(file_audio[0], target_sample_rate), global_update
                            )

                    # the mean across lengths stays for continuity with earlier runs; read
                    # the per-length curves above to tell extrapolation from overall quality
                    metric_log = {f"gen/{k}": sum(v) / len(v) for k, v in scores.items() if len(v) > 0}
                    self.accelerator.log(metric_log, step=global_update)
                    if self.logger == "tensorboard":
                        for k, v in metric_log.items():
                            self.writer.add_scalar(k, v, global_update)

                # Checked last, so a boundary update still gets its checkpoint, samples and
                # metrics before the run ends. Breaking both loops rather than relying on
                # the epoch count: a DDO round's max_updates lands in the middle of an epoch
                # by construction, and the dataloader would happily keep handing out batches.
                if exists(self.max_updates) and global_update >= self.max_updates:
                    break

            if exists(self.max_updates) and global_update >= self.max_updates:
                if self.is_main:
                    print(f"Reached max_updates={self.max_updates}, stopping.")
                break

        self.save_checkpoint(global_update, last=True)

        self.accelerator.end_training()

        if self.logger == "tensorboard":
            self.writer.close()
