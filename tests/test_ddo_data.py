"""CPU-only tests for the DDO data path: tagged concat dataset + offline fake pool.

The theme running through all of them is spec 3.5: DDO makes the model its own
discriminator, so anything that separates the real corpus from the fake pool other than
speech quality is a shortcut the objective will happily take. These tests pin the four
shortcuts shut -- length distribution, loudness, frame grid, file precision -- by
checking that both halves go through one loading path and that the pool is written in a
format that path accepts.
"""

import json
import os
import re

import pytest
import soundfile as sf
import torch
import torchaudio


SR = 16000
FRAME = 160

# the tests build corpora under tmp_path, so they address them by full path rather than
# by the data/<name> convention load_dataset() defaults to
PATHY = {"dataset_type": "CustomDatasetPath"}

SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")


def _gen_module():
    """scripts/ is not a package, so load gen_fake_pool.py by path."""
    import importlib.util
    import sys

    if "gen_fake_pool" in sys.modules:
        return sys.modules["gen_fake_pool"]
    spec = importlib.util.spec_from_file_location("gen_fake_pool", os.path.join(SCRIPTS_DIR, "gen_fake_pool.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules["gen_fake_pool"] = module
    spec.loader.exec_module(module)
    return module


def _write_corpus(root, name, durations, *, amplitude=0.3):
    """A prepared corpus in the real data/<name> layout: wavs + raw + duration.json."""
    from datasets import Dataset as Dataset_

    d = root / name
    (d / "wavs").mkdir(parents=True, exist_ok=True)
    paths = []
    for i, dur in enumerate(durations):
        n = round(dur * SR)
        t = torch.arange(n, dtype=torch.float32) / SR
        wav = amplitude * torch.sin(2 * torch.pi * (110 + 10 * i) * t)
        p = d / "wavs" / f"clip_{i:04d}.wav"
        torchaudio.save(str(p), wav.unsqueeze(0), SR)
        paths.append(str(p))

    Dataset_.from_dict(
        {"audio_path": paths, "duration": [float(x) for x in durations], "text": [""] * len(paths)}
    ).save_to_disk(str(d / "raw"))
    with open(d / "duration.json", "w", encoding="utf-8") as f:
        json.dump({"duration": [float(x) for x in durations]}, f)
    return d


def _waveform_kwargs():
    return {"target_sample_rate": SR, "wav_frame_len": FRAME, "target_rms": 1.0, "rand_frame_offset": True}


# --------------------------------------------------------------------------- Task B1


def test_tagged_concat_routes_indices_and_frame_lens(tmp_path):
    from wavtts.model.dataset import TaggedConcatDataset, load_dataset

    real_dir = _write_corpus(tmp_path, "real", [1.0, 2.0])
    fake_dir = _write_corpus(tmp_path, "fake", [0.5, 1.5, 2.5])

    kw = _waveform_kwargs()
    real = load_dataset(str(real_dir), dataset_type="CustomDatasetPath", waveform_kwargs=kw)
    fake = load_dataset(str(fake_dir), dataset_type="CustomDatasetPath", waveform_kwargs={**kw, "is_fake": True})

    ds = TaggedConcatDataset(real, fake, fake_repeat=3)
    assert len(ds) == 2 + 3 * 3
    assert ds.target_sample_rate == SR

    # the real block comes first, untagged, with the real frame lengths
    assert [ds.get_frame_len(i) for i in range(2)] == [1.0 * SR / FRAME, 2.0 * SR / FRAME]
    assert all(ds[i]["is_fake"] is False for i in range(2))

    # and the fake block wraps modulo len(fake), so copy k of clip j is clip j.
    # Fake frame lengths carry a deterministic sub-frame dither (see _grid_dither), so they
    # are the true length plus something in [0, 1) frames -- never less, never a whole frame.
    for rep in range(3):
        for j in range(3):
            idx = 2 + rep * 3 + j
            true_len = [0.5, 1.5, 2.5][j] * SR / FRAME
            dither = ds.get_frame_len(idx) - true_len
            assert 0.0 <= dither < 1.0
            assert ds.get_frame_len(idx) == ds.get_frame_len(idx)  # deterministic
            assert ds[idx]["is_fake"] is True

    # DynamicBatchSampler only ever needs these two, which is why nothing else changes
    from torch.utils.data import SequentialSampler

    from wavtts.model.dataset import DynamicBatchSampler

    sampler = DynamicBatchSampler(SequentialSampler(ds), frames_threshold=400, max_samples=0)
    batched = [i for batch in sampler.batches for i in batch]
    assert sorted(batched) == list(range(len(ds)))


def test_fake_repeat_hits_the_requested_ratio(tmp_path, capsys):
    from wavtts.model.dataset import load_ddo_dataset

    # 10 h-equivalent of real against 1 h-equivalent of fake, in seconds
    _write_corpus(tmp_path, "real", [2.0] * 10)
    _write_corpus(tmp_path, "fake", [1.0] * 2)

    kw = _waveform_kwargs()
    ds = load_ddo_dataset(str(tmp_path / "real"), str(tmp_path / "fake"), kw, real_fake_ratio=1.0, **PATHY)
    # 20 s real vs 2 s fake -> repeat 10x to reach parity
    assert ds.fake_repeat == 10
    real_frames = sum(ds.real.get_frame_len(i) for i in range(len(ds.real)))
    fake_frames = sum(ds.fake.get_frame_len(i) for i in range(len(ds.fake))) * ds.fake_repeat
    assert real_frames == pytest.approx(fake_frames)
    assert ds[len(ds) - 1]["is_fake"] is True

    # asking for twice as much real as fake halves the repeat count
    ds2 = load_ddo_dataset(str(tmp_path / "real"), str(tmp_path / "fake"), kw, real_fake_ratio=2.0, **PATHY)
    assert ds2.fake_repeat == 5

    assert "real:fake frame ratio" in capsys.readouterr().out


def test_collate_defaults_is_fake_to_false():
    from wavtts.model.dataset import collate_fn

    # a batch that predates DDO carries no flag at all; item["is_fake"] would raise here
    out = collate_fn([{"wav": torch.randn(320)}, {"wav": torch.randn(160)}])
    assert set(out.keys()) == {"wav", "wav_lengths"}
    assert out["wav"].shape == (2, 320)

    # and a mixed batch (one tagged item, one legacy dict) still comes out all-real-safe
    out = collate_fn([{"wav": torch.randn(320), "is_fake": False}, {"wav": torch.randn(160)}])
    assert out["is_fake"].dtype == torch.bool
    assert out["is_fake"].tolist() == [False, False]


def test_collate_marks_fake_rows():
    from wavtts.model.dataset import collate_fn

    batch = [
        {"wav": torch.randn(320), "is_fake": False},
        {"wav": torch.randn(160), "is_fake": True},
        {"wav": torch.randn(480), "is_fake": True},
    ]
    out = collate_fn(batch)
    assert out["is_fake"].dtype == torch.bool
    assert out["is_fake"].tolist() == [False, True, True]
    assert out["is_fake"].shape == (3,)


def test_frame_batches_mix_real_and_fake_rows():
    """Batches must contain both kinds of row, not just the dataset as a whole.

    DynamicBatchSampler sorts by frame length and packs neighbours with a stable sort, so
    exactly-equal keys keep index order: real rows first, fake rows after. Real lengths are
    dense floats (arbitrary recorded sample counts over wav_frame_len) while generated
    lengths are whole frames, so without TaggedConcatDataset._grid_dither every fake row
    piles onto an integer key, the pile is contiguous and longer than one frame budget, and
    the sampler emits runs of all-fake batches. The global real:fake ratio stays perfect
    while the per-batch ratio -- the one the DDO loss and the shared time draws actually
    see -- is 0 or 1 nearly every step.
    """
    from torch.utils.data import SequentialSampler

    from wavtts.model.dataset import CustomDataset, DynamicBatchSampler, TaggedConcatDataset

    torch.manual_seed(0)
    # only get_frame_len is exercised here, so the rows need no audio behind them
    real_samples = torch.randint(30 * FRAME, 60 * FRAME, (600,))
    real_durs = (real_samples.double() / SR).tolist()
    fake_durs = (torch.randint(30, 60, (60,)).double() * FRAME / SR).tolist()  # whole frames

    real = CustomDataset([{"duration": d} for d in real_durs], durations=real_durs, wav_frame_len=FRAME)
    fake = CustomDataset([{"duration": d} for d in fake_durs], durations=fake_durs, wav_frame_len=FRAME, is_fake=True)
    ds = TaggedConcatDataset(real, fake, fake_repeat=10)

    sampler = DynamicBatchSampler(SequentialSampler(ds), frames_threshold=480, max_samples=0)
    n_real = len(real)
    fractions = [sum(i >= n_real for i in b) / len(b) for b in sampler.batches]
    one_sided = sum(f in (0.0, 1.0) for f in fractions) / len(fractions)
    assert one_sided < 0.25, f"{one_sided:.0%} of batches are one-sided; the DDO loss needs both halves"
    assert sum(fractions) / len(fractions) == pytest.approx(0.5, abs=0.1)

    # the dither only ever over-reports, and by less than one frame, so the sampler's budget
    # stays conservative and no batch grows past frames_threshold
    for i in range(n_real, len(ds)):
        slack = ds.get_frame_len(i) - fake.get_frame_len((i - n_real) % len(fake))
        assert 0.0 <= slack < 1.0


def test_fake_pool_loudness_and_grid_are_erased_by_the_shared_loader(tmp_path):
    """The reason the pool is written as a corpus instead of handed to the model directly.

    A pool at a wildly different level than the corpus would let the discriminator win on
    loudness alone; the shared CustomDataset puts both on target_rms. Likewise the model
    generates exactly on the frame grid, and rand_frame_offset knocks it off.
    """
    from wavtts.model.dataset import load_ddo_dataset

    _write_corpus(tmp_path, "real", [1.0] * 4, amplitude=0.3)
    _write_corpus(tmp_path, "fake", [1.0] * 4, amplitude=0.003)  # 100x quieter on disk

    ds = load_ddo_dataset(str(tmp_path / "real"), str(tmp_path / "fake"), _waveform_kwargs(), **PATHY)
    rms = [ds[i]["wav"].pow(2).mean().sqrt().item() for i in range(len(ds))]
    assert all(r == pytest.approx(1.0, rel=1e-3) for r in rms)

    torch.manual_seed(0)
    lens = {ds[len(ds) - 1]["wav"].shape[0] for _ in range(8)}
    assert len(lens) > 1, "rand_frame_offset must jitter fake clips too, or the grid gives them away"


# --------------------------------------------------------------------------- Task B2


def _tiny_config(tmp_path):
    """WavTTS_clean.yaml with the architecture shrunk to something a CPU can run."""
    from importlib.resources import files as pkg_files

    from omegaconf import OmegaConf

    cfg = OmegaConf.load(str(pkg_files("wavtts").joinpath("configs/WavTTS_clean.yaml")))
    cfg.model.arch.dim = 64
    cfg.model.arch.depth = 2
    cfg.model.arch.heads = 2
    cfg.model.arch.dim_head = 32
    cfg.model.arch.ff_mult = 2
    cfg.model.arch.checkpoint_activations = False
    cfg.model.arch.audio_proj_dim = 64
    cfg.model.arch.audio_proj_hidden = 64
    cfg.model.cfm.use_aux_mel_loss = False
    path = tmp_path / "tiny.yaml"
    OmegaConf.save(cfg, str(path))
    return cfg, str(path)


def _tiny_checkpoint(tmp_path, cfg):
    """A randomly initialised model saved as a weights-only checkpoint."""
    from hydra.utils import get_class
    from omegaconf import OmegaConf
    from torch import nn

    from wavtts.model import CFM

    torch.manual_seed(0)
    model_cls = get_class(f"wavtts.model.{cfg.model.backbone}")
    model = CFM(
        transformer=model_cls(
            **OmegaConf.to_container(cfg.model.arch, resolve=True),
            wav_frame_len=cfg.model.waveform.wav_frame_len,
        ),
        waveform_kwargs=OmegaConf.to_container(cfg.model.waveform, resolve=True),
        **OmegaConf.to_container(cfg.model.cfm, resolve=True),
    )
    # zero-initialised AdaLN/proj_out would make every generated clip a pure function of
    # the noise; perturb so the pool looks like output rather than input
    for p in model.parameters():
        nn.init.normal_(p, std=0.02)
    path = tmp_path / "ref.pt"
    torch.save({"model_state_dict": model.state_dict()}, str(path))
    return str(path)


def _ref_durations_file(tmp_path, durations):
    path = tmp_path / "ref_duration.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"duration": [float(d) for d in durations]}, f)
    return str(path)


def _gen_argv(tmp_path, ckpt, config, ref_json, out, **overrides):
    args = {
        "--hours": "0.002",
        "--steps": "2",
        "--cfg_strength": "0.0",
        "--solver": "euler",
        "--max_batch_frames": "3200",
        "--seed": "1234",
        "--device": "cpu",
    }
    args.update({k: str(v) for k, v in overrides.items()})
    argv = ["--ckpt", ckpt, "--config", config, "--ref_durations", ref_json, "--out", str(out)]
    for k, v in args.items():
        argv += [k, v]
    return argv


def test_gen_fake_pool_end_to_end(tmp_path):
    from wavtts.model.dataset import load_dataset, load_ddo_dataset

    gen = _gen_module()
    cfg, config_path = _tiny_config(tmp_path)
    ckpt = _tiny_checkpoint(tmp_path, cfg)
    # a spread of real durations, so the pool has more than one length to group
    ref_json = _ref_durations_file(tmp_path, [0.35, 0.4, 0.5, 0.6, 0.8])
    out = tmp_path / "LibriTTS_fake_r1"

    gen.main(_gen_argv(tmp_path, ckpt, config_path, ref_json, out))

    # the real corpus layout, exactly
    assert (out / "raw").is_dir()
    assert (out / "duration.json").is_file()

    with open(out / "duration.json", "r", encoding="utf-8") as f:
        durations = json.load(f)["duration"]
    wavs = sorted((out / "wavs").glob("fake_*.wav"))
    assert len(wavs) == len(durations) > 0
    assert sum(durations) >= 0.002 * 3600

    for path, dur in zip(wavs, durations):
        info = sf.info(str(path))
        assert info.samplerate == SR
        assert info.frames == round(dur * SR), "duration.json must match the file on disk"
        assert info.frames % FRAME == 0, "clips must land on the frame grid the model generates on"

    # the acceptance condition: the existing loader reads it back with no special casing
    ds = load_dataset(str(out), dataset_type="CustomDatasetPath", waveform_kwargs=_waveform_kwargs())
    assert len(ds) == len(durations)
    assert torch.isfinite(ds[0]["wav"]).all()

    # and it pairs with a real corpus through load_ddo_dataset
    _write_corpus(tmp_path, "real", [0.5] * 20)
    pooled = load_ddo_dataset(str(tmp_path / "real"), str(out), _waveform_kwargs(), **PATHY)
    assert pooled[len(pooled) - 1]["is_fake"] is True


def test_gen_fake_pool_resumes_without_changing_the_pool(tmp_path, capsys):
    gen = _gen_module()
    cfg, config_path = _tiny_config(tmp_path)
    ckpt = _tiny_checkpoint(tmp_path, cfg)
    ref_json = _ref_durations_file(tmp_path, [0.35, 0.5, 0.8])
    out = tmp_path / "pool"

    gen.main(_gen_argv(tmp_path, ckpt, config_path, ref_json, out))
    first = {p.name: sf.read(str(p), dtype="float32")[0] for p in sorted((out / "wavs").glob("*.wav"))}

    # a complete pool costs nothing to "resume"
    capsys.readouterr()
    gen.main(_gen_argv(tmp_path, ckpt, config_path, ref_json, out))
    assert "wrote 0 new clips" in capsys.readouterr().out

    # drop a few clips and re-run: the missing ones come back byte-identical, because the
    # clip list and the clip -> batch assignment are planned from the seed, not from what
    # happens to be on disk. Resume granularity is the batch, not the clip -- a partially
    # written batch is redone whole so its shared seed still yields the same waveforms --
    # so this rewrites more than the two files, but strictly fewer than all of them.
    for name in list(first)[:2]:
        os.remove(out / "wavs" / name)
    capsys.readouterr()
    gen.main(_gen_argv(tmp_path, ckpt, config_path, ref_json, out))
    n_written = int(re.search(r"wrote (\d+) new clips", capsys.readouterr().out).group(1))
    assert 2 <= n_written < len(first)

    second = {p.name: sf.read(str(p), dtype="float32")[0] for p in sorted((out / "wavs").glob("*.wav"))}
    assert set(first) == set(second)
    for name, wav in first.items():
        assert (wav == second[name]).all(), name


def test_fake_wavs_are_float32_and_unclipped(tmp_path, monkeypatch):
    """The pool is written in 32-bit float and keeps peaks outside +-1.

    The model generates at target_rms with speech's crest factor, so its output leaves
    +-1 on essentially every clip. 16-bit PCM would clip those peaks, and "clipped" is a
    feature no recorded clip has -- a free win for the discriminator that teaches the
    model nothing (spec 3.5, shortcut 4). peak_normalize() is the same mistake: it would
    make peak level constant on the fake half and variable on the real half.

    This also guards a live trap: torchaudio.save(encoding="PCM_F", bits_per_sample=32)
    *silently ignores* both arguments under torchaudio's TorchCodec backend and writes
    clipping 16-bit PCM, which is why the writer uses soundfile.
    """
    from wavtts.model import CFM

    gen = _gen_module()
    cfg, config_path = _tiny_config(tmp_path)
    ckpt = _tiny_checkpoint(tmp_path, cfg)
    ref_json = _ref_durations_file(tmp_path, [0.5])
    out = tmp_path / "loud_pool"

    original = CFM.sample

    def loud_sample(self, *a, **kw):
        wav, traj = original(self, *a, **kw)
        return wav / wav.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) * 3.0, traj

    monkeypatch.setattr(CFM, "sample", loud_sample)
    gen.main(_gen_argv(tmp_path, ckpt, config_path, ref_json, out))

    paths = sorted((out / "wavs").glob("*.wav"))
    assert paths
    for path in paths:
        assert sf.info(str(path)).subtype == "FLOAT"
        data, _ = sf.read(str(path), dtype="float32")
        assert abs(data).max() == pytest.approx(3.0, rel=1e-5)
        # and the loader the trainer actually uses sees it unclipped too
        loaded, _ = torchaudio.load(str(path))
        assert loaded.abs().max().item() == pytest.approx(3.0, rel=1e-5)


def test_sampled_durations_track_the_reference_distribution():
    sample_clip_lengths = _gen_module().sample_clip_lengths

    # bimodal, deliberately lopsided so the median is well defined (a 50/50 bimodal has no
    # stable median and "the medians match" would be a coin flip, not a test)
    ref = [0.5] * 600 + [5.0] * 400
    total = 2000 * 2.3  # E[duration] = 0.6*0.5 + 0.4*5.0 = 2.3 s

    lengths = sample_clip_lengths(
        ref, total, frame_samples=FRAME, sample_rate=SR, generator=torch.Generator().manual_seed(0)
    )
    durations = torch.tensor([n / SR for n in lengths], dtype=torch.float64)

    assert 1900 <= durations.numel() <= 2100
    # both peaks present, at roughly their reference weights -- a fixed length or a uniform
    # draw would fail here, and either would hand the discriminator a free separator
    short = (durations < 1.0).double().mean().item()
    assert short == pytest.approx(0.6, abs=0.05)
    assert (durations > 4.0).double().mean().item() == pytest.approx(0.4, abs=0.05)
    assert set(durations.unique().tolist()) == {0.5, 5.0}
    # and the central statistics line up with the reference's
    ref_t = torch.tensor(ref, dtype=torch.float64)
    assert durations.median().item() == pytest.approx(ref_t.median().item())
    assert durations.mean().item() == pytest.approx(ref_t.mean().item(), rel=0.07)
