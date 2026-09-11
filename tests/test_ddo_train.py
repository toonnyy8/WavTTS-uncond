"""CPU-only tests for the DDO training wiring: Trainer, train.py and the round-1 config.

Group A proved the loss is right and Group B proved the data is right; what is left to go
wrong is all plumbing, and the failures are quiet ones. A schedule derived from epochs
leaves the LR flat for a whole round. An EMA at the ema_pytorch defaults averages the
round away and every gen/* curve comes out looking like nothing trained. A reference model
left on the cpu, or copied a second time into the EMA, costs memory or a crash. None of
those raise anything the run would notice, so they are pinned here instead.
"""

import json
import math
import os

import pytest
import torch
import torchaudio


# Every Accelerator in this file must stay off the GPU: the machine this repo is developed
# on has four of them and they are usually busy with a real run. accelerate reads this once,
# when the first PartialState is built, so it is set at import rather than per test.
os.environ["ACCELERATE_USE_CPU"] = "1"

SR = 16000
FRAME = 160

# the corpora live under tmp_path, so they are addressed by full path rather than by the
# data/<name> convention load_dataset() defaults to
PATHY = {"dataset_type": "CustomDatasetPath"}


# --------------------------------------------------------------------------- helpers


def _write_corpus(root, name, durations, *, amplitude=0.3):
    """A prepared corpus in the real data/<name> layout: wavs + raw + duration.json."""
    from datasets import Dataset as Dataset_

    d = root / name
    (d / "wavs").mkdir(parents=True, exist_ok=True)
    paths = []
    for i, dur in enumerate(durations):
        n = round(dur * SR)
        t = torch.arange(n, dtype=torch.float32) / SR
        wav = amplitude * torch.sin(2 * torch.pi * (110 + 7 * i) * t)
        p = d / "wavs" / f"clip_{i:04d}.wav"
        torchaudio.save(str(p), wav.unsqueeze(0), SR)
        paths.append(str(p))

    Dataset_.from_dict(
        {"audio_path": paths, "duration": [float(x) for x in durations], "text": [""] * len(paths)}
    ).save_to_disk(str(d / "raw"))
    with open(d / "duration.json", "w", encoding="utf-8") as f:
        json.dump({"duration": [float(x) for x in durations]}, f)
    return d


def _tiny_cfm(**cfm_overrides):
    """A CFM small enough for a cpu, with the DDO-relevant settings of the real config."""
    from wavtts.model import CFM, DiT

    cfm_kwargs = dict(
        prediction="x_pred",
        loss_space="v",
        t_sampling="logistic_normal",
        P_mean=-0.8,
        P_std=0.8,
        t_eps=0.02,
        state_null_prob=0.0,
        use_aux_mel_loss=False,
        sample_rate=SR,
    )
    cfm_kwargs.update(cfm_overrides)
    return CFM(
        transformer=DiT(
            dim=64,
            depth=2,
            heads=2,
            dim_head=32,
            ff_mult=2,
            dropout=0.0,  # DiT's default is 0.1, and a non-zero one makes Delta noise
            attn_mask_enabled=True,
            rpe_gamma=4.0,
            logn_ref_len=500,
            wav_frame_len=FRAME,
        ),
        waveform_kwargs={"target_sample_rate": SR, "wav_frame_len": FRAME, "target_rms": 1.0},
        **cfm_kwargs,
    )


class _WavDataset(torch.utils.data.Dataset):
    """An in-memory stand-in for CustomDataset. Module level so it survives pickling into
    a dataloader worker -- persistent_workers forbids num_workers=0."""

    target_sample_rate = SR
    wav_frame_len = FRAME

    def __init__(self, n=20, samples=1600):
        self.n = n
        self.samples = samples

    def __len__(self):
        return self.n

    def get_frame_len(self, index):
        return self.samples / self.wav_frame_len

    def __getitem__(self, index):
        g = torch.Generator().manual_seed(index)
        return {"wav": torch.randn(self.samples, generator=g) * 0.1}


class _TaggedWavDataset(_WavDataset):
    """_WavDataset with every row tagged as coming from p_ref's pool.

    Every row rather than every other one: with sample-wise batching the composition of
    any one batch is a draw from the shuffle, and the assertion below has to hold on the
    first batch whatever that draw was.
    """

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item["is_fake"] = True
        return item


def _trainer(model, checkpoint_path, **overrides):
    from wavtts.model.trainer import Trainer

    kwargs = dict(
        epochs=5,
        learning_rate=1e-4,
        num_warmup_updates=1,
        save_per_updates=100,
        last_per_updates=100,
        checkpoint_path=str(checkpoint_path),
        batch_size_per_gpu=2,
        batch_size_type="sample",
        grad_accumulation_steps=1,
        logger=None,
        log_samples=False,
    )
    kwargs.update(overrides)
    return Trainer(model, **kwargs)


# --------------------------------------------------------------------------- Task C1


def test_max_updates_stops_the_run(tmp_path):
    """The round ends on its update budget, not when the epochs run out.

    The dataset below would supply 50 updates over 5 epochs; a DDO round asks for 3 and
    must get exactly 3, with model_last.pt written on the way out.
    """
    torch.manual_seed(0)
    model = _tiny_cfm()
    trainer = _trainer(model, tmp_path, max_updates=3)
    trainer.train(_WavDataset(n=20), num_workers=1, resumable_with_seed=None)

    ckpt = torch.load(str(tmp_path / "model_last.pt"), weights_only=True, map_location="cpu")
    assert ckpt["update"] == 3


def test_max_updates_shortens_the_lr_horizon(tmp_path):
    """The decay horizon follows max_updates, so the LR actually reaches its floor.

    With the epoch-derived horizon the same run would decay over 50 updates and still be
    at ~94% of the peak LR when the round ended -- the round would be a constant-LR run
    wearing a schedule.
    """
    torch.manual_seed(0)
    trainer = _trainer(_tiny_cfm(), tmp_path, max_updates=3)
    trainer.train(_WavDataset(n=20), num_workers=1, resumable_with_seed=None)

    # warmup 1 + decay 2 == the 3 updates asked for, so the last update sits on the floor
    assert trainer.scheduler.get_last_lr()[0] == pytest.approx(1e-4 * 1e-8, rel=1e-3)


def test_lr_decay_end_factor_sets_the_floor(tmp_path):
    """The paper never anneals a round to zero (CIFAR: warmup only; EDM2: inverse-sqrt to
    ~0.4x by the end), so the floor is a knob, and the default stays the 1e-8 pretraining
    always had."""
    torch.manual_seed(0)
    trainer = _trainer(_tiny_cfm(), tmp_path, max_updates=3, lr_decay_end_factor=0.3)
    trainer.train(_WavDataset(n=20), num_workers=1, resumable_with_seed=None)
    assert trainer.scheduler.get_last_lr()[0] == pytest.approx(1e-4 * 0.3, rel=1e-3)


def test_resuming_a_finished_round_is_refused(tmp_path):
    """Round n+1 launched under round n's model.name finds round n's model_last.pt in the
    same save_dir and would full-state-resume it: optimizer state carried across rounds,
    update already at max_updates, one batch, and the file rewritten with the weights it
    started from. The trainer refuses rather than producing a round of nothing."""
    torch.manual_seed(0)
    trainer = _trainer(_tiny_cfm(), tmp_path, max_updates=2)
    trainer.train(_WavDataset(n=20), num_workers=1, resumable_with_seed=None)
    assert (tmp_path / "model_last.pt").exists()

    torch.manual_seed(0)
    again = _trainer(_tiny_cfm(), tmp_path, max_updates=2)
    with pytest.raises(RuntimeError, match="finished run"):
        again.train(_WavDataset(n=20), num_workers=1, resumable_with_seed=None)


def test_loss_dict_logging_skips_nan():
    """nan means "no rows of this kind this step", and must not reach the logger.

    A single nan point makes a tensorboard curve's autoscale useless for the rest of the
    run, and the DDO loss_dict emits them by design -- ddo/loss_fake is nan on any batch
    the sampler happened to fill with real rows only.
    """
    from wavtts.model.trainer import Trainer

    logs = Trainer._scalar_logs(
        {
            "total_loss": torch.tensor(1.5),
            "flow_loss": torch.tensor(2.0),
            "aux_mel_loss": torch.tensor(float("nan")),
            "ddo/acc": torch.tensor(0.625),
            "ddo/loss_fake": float("nan"),
            "ddo/margin": 0.25,
            "not_a_scalar": torch.zeros(3),
            "not_a_number": None,
        }
    )
    # total_loss is renamed to "loss" so the curve is continuous across the
    # pretraining -> DDO switch
    assert logs == {"loss": 1.5, "flow_loss": 2.0, "ddo/acc": 0.625, "ddo/margin": 0.25}
    assert all(not math.isnan(v) for v in logs.values())


def test_ema_does_not_hold_a_second_reference(tmp_path):
    """EMA deepcopies the model, and the one-element list holding p_ref is copied with it.

    What makes p_ref invisible is nn.Module.__setattr__ declining to register a list, not
    deepcopy declining to look at one -- so without the trainer's explicit clear the EMA
    silently carries a second frozen 664M-parameter model that nothing ever reads.
    """
    torch.manual_seed(0)
    model = _tiny_cfm()
    ref = _tiny_cfm()
    model.attach_ddo_ref(ref, alpha=1.0, beta=1.0)

    trainer = _trainer(model, tmp_path)

    assert trainer.ema_model.ema_model.ddo_ref is None
    assert trainer.ema_model.ema_model._ddo_ref == []
    # the online model keeps its reference, and it is the same object, not a copy
    assert trainer.accelerator.unwrap_model(trainer.model).ddo_ref is ref
    # and the EMA weights are still a full copy of theta -- the clear touched only the list
    assert set(trainer.ema_model.ema_model.state_dict()) == set(model.state_dict())


def test_reference_is_moved_onto_the_training_device(tmp_path):
    """p_ref is outside the module tree, so accelerator.prepare() cannot reach it.

    On a gpu box the model moves and the reference does not, and the first DDO batch dies
    on a device mismatch. The trainer moves it by hand once, after prepare.
    """
    torch.manual_seed(0)
    model = _tiny_cfm()
    ref = _tiny_cfm()
    model.attach_ddo_ref(ref, alpha=1.0, beta=1.0)

    trainer = _trainer(model, tmp_path)
    device = trainer.accelerator.device
    assert all(p.device.type == device.type for p in ref.parameters())


def test_is_fake_reaches_the_model(tmp_path, monkeypatch):
    """The tag the collate_fn attaches has to survive the trip into CFM.forward.

    Drop it and the run still trains happily -- on a batch where every fake row is scored
    as a real-positive, which is the DDO objective with its negative term inverted.
    """
    from wavtts.model import CFM

    seen = []
    original = CFM.forward

    def spy(self, inp, *, lens=None, is_fake=None):
        seen.append(None if is_fake is None else is_fake.clone())
        return original(self, inp, lens=lens, is_fake=is_fake)

    monkeypatch.setattr(CFM, "forward", spy)

    torch.manual_seed(0)
    trainer = _trainer(_tiny_cfm(), tmp_path, max_updates=2)
    trainer.train(_TaggedWavDataset(n=8), num_workers=1, resumable_with_seed=None)

    assert seen, "the training loop never called the model"
    assert seen[0] is not None, "the fake tag was dropped between collate_fn and CFM.forward"
    assert seen[0].dtype == torch.bool
    assert seen[0].all()


# --------------------------------------------------------------------------- Task C2


def _ddo_config(tmp_path, real_dir, fake_dir, ref_ckpt):
    """WavTTS_ddo_r1.yaml, shrunk to something a cpu finishes in seconds."""
    from importlib.resources import files as pkg_files

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    config_dir = str(pkg_files("wavtts").joinpath("configs"))
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="WavTTS_ddo_r1")
    OmegaConf.set_struct(cfg, False)

    cfg.model.arch.dim = 64
    cfg.model.arch.depth = 2
    cfg.model.arch.heads = 2
    cfg.model.arch.dim_head = 32
    cfg.model.arch.ff_mult = 2
    cfg.model.arch.checkpoint_activations = False
    cfg.model.arch.audio_proj_dim = 64
    cfg.model.arch.audio_proj_hidden = 64
    cfg.model.cfm.use_aux_mel_loss = False

    cfg.datasets.name = str(real_dir)
    cfg.ddo.fake_dataset = str(fake_dir)
    cfg.ddo.ref_ckpt = str(ref_ckpt)
    cfg.datasets.batch_size_per_gpu = 400  # frames; ~6 of the half-second clips per batch
    cfg.datasets.max_samples = 6
    cfg.datasets.num_workers = 1  # persistent_workers forbids 0

    cfg.optim.epochs = 1
    cfg.optim.num_warmup_updates = 1
    cfg.optim.max_updates = 2
    cfg.optim.grad_accumulation_steps = 1

    cfg.ckpts.logger = None  # a tensorboard writer would litter runs/ in the repo
    cfg.ckpts.log_samples = False
    cfg.ckpts.save_dir = str(tmp_path / "ckpts")
    return cfg


def _ema_checkpoint(path, model):
    """A checkpoint in the shape Trainer.save_checkpoint writes, EMA weights and all.

    load_ref_state_dict prefers the EMA half and strips ema_pytorch's bookkeeping tensors,
    so the fixture hands it exactly that rather than a bare state dict.
    """
    ema = {f"ema_model.{k}": v for k, v in model.state_dict().items()}
    ema["initted"] = torch.tensor(True)
    ema["step"] = torch.tensor(7)
    torch.save({"ema_model_state_dict": ema}, str(path))
    return str(path)


@pytest.fixture(scope="module")
def ddo_run(tmp_path_factory):
    """One end-to-end round of two updates, driven through train.py's own main().

    Module-scoped because the three tests below all read different things out of the same
    run and it is the slowest thing in this file.
    """
    import hydra
    from omegaconf import OmegaConf

    import wavtts.train.train as train_mod
    from wavtts.model import CFM
    from wavtts.model.trainer import Trainer

    tmp_path = tmp_path_factory.mktemp("ddo_run")
    # The real durations are deliberately off the frame grid and the fake ones deliberately
    # on it -- that is the real corpus' geometry in miniature, and what TaggedConcatDataset's
    # sub-frame dither exists to interleave. Equal-length corpora would sort into one solid
    # block of real followed by one of fake and every batch here would be one-sided.
    real_dir = _write_corpus(tmp_path, "real", [0.5 + 0.0137 * i for i in range(24)])
    fake_dir = _write_corpus(tmp_path, "fake", [0.51, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.53])

    torch.manual_seed(0)
    cfg = _ddo_config(tmp_path, real_dir, fake_dir, tmp_path / "ref.pt")

    # the reference weights, built from the same (shrunk) config so the keys line up
    model_cls = hydra.utils.get_class(f"wavtts.model.{cfg.model.backbone}")
    ref_template = CFM(
        transformer=model_cls(
            **OmegaConf.to_container(cfg.model.arch, resolve=True),
            wav_frame_len=cfg.model.waveform.wav_frame_len,
        ),
        waveform_kwargs=OmegaConf.to_container(cfg.model.waveform, resolve=True),
        **OmegaConf.to_container(cfg.model.cfm, resolve=True),
    )
    _ema_checkpoint(tmp_path / "ref.pt", ref_template)
    # what make_pretrained_init.py writes into a fresh save_dir: the round starts at theta ==
    # p_ref, which train.py now checks for before it will run a DDO config at all
    (tmp_path / "ckpts").mkdir()
    _ema_checkpoint(tmp_path / "ckpts" / "pretrained_ref.pt", ref_template)

    captured = {}
    with pytest.MonkeyPatch.context() as mp:
        # the corpora live under tmp_path rather than data/<name>, which is the one thing
        # train.py cannot express; everything else about its wiring is under test
        real_loader = train_mod.load_ddo_dataset

        def by_path(real, fake, waveform_kwargs, real_fake_ratio=1.0):
            return real_loader(real, fake, waveform_kwargs, real_fake_ratio=real_fake_ratio, **PATHY)

        mp.setattr(train_mod, "load_ddo_dataset", by_path)

        attach = CFM.attach_ddo_ref

        def spy(self, ref, **kwargs):
            captured["ref"] = ref
            captured["ref_before"] = {k: v.detach().clone() for k, v in ref.state_dict().items()}
            captured["attach_kwargs"] = dict(kwargs)
            return attach(self, ref, **kwargs)

        mp.setattr(CFM, "attach_ddo_ref", spy)

        # what the trainer would have written to tensorboard, step by step
        scalar_logs = Trainer._scalar_logs
        captured["logs"] = []

        def log_spy(loss_dict):
            out = scalar_logs(loss_dict)
            captured["logs"].append(out)
            return out

        mp.setattr(Trainer, "_scalar_logs", staticmethod(log_spy))

        train_mod.main.__wrapped__(cfg)

    captured["cfg"] = cfg
    captured["ckpt_dir"] = tmp_path / "ckpts"
    return captured


def test_ddo_config_instantiates_and_trains_one_step(ddo_run):
    """WavTTS_ddo_r1.yaml goes through train.py end to end and produces a checkpoint."""
    ckpt_path = ddo_run["ckpt_dir"] / "model_last.pt"
    assert ckpt_path.exists()
    ckpt = torch.load(str(ckpt_path), weights_only=True, map_location="cpu")
    assert ckpt["update"] == 2

    # the ddo block reached attach_ddo_ref with the config's values, not the defaults
    assert ddo_run["attach_kwargs"] == {
        "alpha": 1.0,
        "beta": 100.0,  # the round-1 calibration, recorded in the config's own comments
        "delta_normalize": "mean",
        "anchor_weight": 1.0,
        "real_mel_weight": 0.0,
    }

    # the DDO objective is what actually ran, and its statistics reached the logger under
    # their own names rather than being dropped by a hardcoded two-key logging block
    logs = ddo_run["logs"]
    assert all(math.isfinite(lg["loss"]) for lg in logs)
    assert any("ddo/loss_real" in lg for lg in logs)
    assert any("ddo/loss_fake" in lg for lg in logs)
    assert any("ddo/margin" in lg and "ddo/acc" in lg for lg in logs)
    # this arm has no null rows, so the anchor term never fires and never gets logged
    assert all("anchor_loss" not in lg for lg in logs)


def test_ddo_refuses_to_start_without_an_init(ddo_run, tmp_path):
    """An empty save_dir means theta would start from a random init against a pretrained
    p_ref, and the run would go ahead on a Delta that is all initialisation gap."""
    import copy

    import wavtts.train.train as train_mod

    cfg = copy.deepcopy(ddo_run["cfg"])
    cfg.ckpts.save_dir = str(tmp_path / "never_initialised")
    with pytest.raises(SystemExit, match="holds no checkpoint"):
        train_mod.main.__wrapped__(cfg)


def test_ddo_config_keeps_the_no_cfg_and_no_dropout_invariants():
    """Two config values that break DDO silently if they drift.

    state_null_prob > 0 would reintroduce guidance geometry this study does not use, and
    a non-zero dropout would have theta (train mode) and p_ref (eval mode) score the same
    row under different masks -- Delta stops being a difference between models.
    """
    from importlib.resources import files as pkg_files

    from omegaconf import OmegaConf

    cfg = OmegaConf.load(str(pkg_files("wavtts").joinpath("configs/WavTTS_ddo_r1.yaml")))
    assert cfg.model.cfm.state_null_prob == 0.0
    assert cfg.model.arch.dropout == 0.0
    assert cfg.ddo.fake_cfg_strength == 0.0
    assert cfg.ddo.delta_normalize == "mean"
    # the EMA that the spec's S3.8 insists on. make_pretrained_init.py zeroes the EMA step,
    # so ema_pytorch's decay ramp sets the window, not beta: under the default ramp beta is
    # never reached inside a 9k-update round, and power 1.0 is what gets it there by 1000
    assert cfg.optim.ema_kwargs.beta == 0.999
    assert cfg.optim.ema_kwargs.update_every == 1
    assert cfg.optim.ema_kwargs.update_after_step == 0
    assert cfg.optim.ema_kwargs.power == 1.0
    assert cfg.optim.max_updates == 9000
    # the paper never anneals a round to zero; 1e-8 would freeze the last third of it
    assert cfg.optim.lr_decay_end_factor == 0.3


def test_checkpoint_has_no_ref_weights(ddo_run):
    """p_ref must not ride along in the checkpoint.

    If it did, the file would be twice the size and every key in it would stop matching
    what make_pretrained_init.py and sample_uncond.py expect -- which is the whole reason
    the reference is parked in a list instead of assigned as an attribute.
    """
    import hydra
    from omegaconf import OmegaConf

    from wavtts.model import CFM

    cfg = ddo_run["cfg"]
    model_cls = hydra.utils.get_class(f"wavtts.model.{cfg.model.backbone}")
    plain = CFM(
        transformer=model_cls(
            **OmegaConf.to_container(cfg.model.arch, resolve=True),
            wav_frame_len=cfg.model.waveform.wav_frame_len,
        ),
        waveform_kwargs=OmegaConf.to_container(cfg.model.waveform, resolve=True),
        **OmegaConf.to_container(cfg.model.cfm, resolve=True),
    )

    ckpt = torch.load(str(ddo_run["ckpt_dir"] / "model_last.pt"), weights_only=True, map_location="cpu")
    saved = ckpt["model_state_dict"]
    assert set(saved) == set(plain.state_dict())
    assert not any("ddo" in k or "ref" in k for k in saved)

    saved_numel = sum(v.numel() for v in saved.values())
    plain_numel = sum(v.numel() for v in plain.state_dict().values())
    assert saved_numel == plain_numel  # not doubled

    # the EMA half is the same single model, not a model plus a stowaway reference
    ema = {k for k in ckpt["ema_model_state_dict"] if k not in ("initted", "step")}
    assert {k.removeprefix("ema_model.") for k in ema} == set(plain.state_dict())


def test_ref_weights_are_unchanged_after_a_step(ddo_run):
    """p_ref is frozen: no optimizer sees it, no gradient reaches it, EMA does not move it.

    The whole method rests on the reference staying put -- a drifting p_ref turns the
    log-ratio into a comparison with a moving target and Theorem 3.1 stops applying.
    """
    ref = ddo_run["ref"]
    before = ddo_run["ref_before"]
    after = ref.state_dict()

    assert set(before) == set(after)
    for key, value in before.items():
        assert torch.equal(value, after[key]), f"p_ref moved at {key}"

    assert not any(p.requires_grad for p in ref.parameters())
    assert all(p.grad is None for p in ref.parameters())
    assert not ref.training  # attach_ddo_ref puts it in eval mode and leaves it there
