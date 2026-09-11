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

import numpy as np
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

    # the real block comes first, untagged. Every row's frame length carries a deterministic
    # sub-frame dither (see _grid_dither): the true length plus something in [0, 1) frames --
    # never less, never a whole frame -- so the sampler's budget stays conservative
    for i, true_len in enumerate([1.0 * SR / FRAME, 2.0 * SR / FRAME]):
        assert 0.0 <= ds.get_frame_len(i) - true_len < 1.0
        assert ds[i]["is_fake"] is False

    # and the fake block wraps modulo len(fake), so copy k of clip j is clip j
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
    exactly-equal keys keep index order: real rows first, fake rows after. Generated
    lengths are whole frames by construction, and so are the recorded ones here:
    LibriTTS_460's duration.json is rounded to 0.01 s, which at 100 frames/s is exactly
    one frame. Both halves pile onto the same integer keys, each pile is longer than one
    frame budget, and without TaggedConcatDataset._grid_dither on *every* row the sampler
    emits runs of all-real then all-fake batches (96.8% one-sided on the real corpus). The
    global real:fake ratio stays perfect while the per-batch ratio -- the one the DDO loss
    and the shared (t, eps) draws actually see -- is 0 or 1 nearly every step.
    """
    from torch.utils.data import SequentialSampler

    from wavtts.model.dataset import CustomDataset, DynamicBatchSampler, TaggedConcatDataset

    torch.manual_seed(0)
    # only get_frame_len is exercised here, so the rows need no audio behind them. Real
    # durations are rounded to 0.01 s exactly as the corpus' prepare script writes them
    real_durs = [round(x, 2) for x in (torch.randint(30 * FRAME, 60 * FRAME, (600,)).double() / SR).tolist()]
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
    for i in range(len(ds)):
        src, idx = ds._route(i)
        slack = ds.get_frame_len(i) - src.get_frame_len(idx)
        assert 0.0 <= slack < 1.0


def test_paired_sampler_puts_both_kinds_in_every_batch():
    """The sampler the DDO trainer actually uses: every batch of two or more rows has a
    real and a fake row, the budget holds, nothing is dropped or repeated, and the shuffle
    is the parent's (seeded, per epoch)."""
    from torch.utils.data import SequentialSampler

    from wavtts.model.dataset import CustomDataset, DynamicBatchSampler, PairedDynamicBatchSampler, TaggedConcatDataset

    torch.manual_seed(1)
    # a wide length spread including clips too long to pair inside the budget
    real_durs = [round(x, 2) for x in (torch.randint(20 * FRAME, 300 * FRAME, (400,)).double() / SR).tolist()]
    fake_durs = (torch.randint(20, 300, (50,)).double() * FRAME / SR).tolist()
    real = CustomDataset([{"duration": d} for d in real_durs], durations=real_durs, wav_frame_len=FRAME)
    fake = CustomDataset([{"duration": d} for d in fake_durs], durations=fake_durs, wav_frame_len=FRAME, is_fake=True)
    ds = TaggedConcatDataset(real, fake, fake_repeat=8)
    n_real = len(real)
    budget = 1200  # four of the longest clips: everything can pair, so one-sided means the tail

    paired = PairedDynamicBatchSampler(ds, budget, max_samples=0, random_seed=0)
    assert sorted(i for b in paired.batches for i in b) == list(range(len(ds)))
    # a side is exhausted once its longest row has been placed; batches built after that
    # point can only hold the other kind, and are the one legitimate one-sided case
    last_real = max(range(n_real), key=ds.get_frame_len)
    last_fake = max(range(n_real, len(ds)), key=ds.get_frame_len)
    exhausted_at = min(next(k for k, b in enumerate(paired.batches) if i in b) for i in (last_real, last_fake))
    for k, b in enumerate(paired.batches):
        frames = sum(ds.get_frame_len(i) for i in b)
        assert len(b) == 1 or frames <= budget
        kinds = {i >= n_real for i in b}
        if len(b) >= 2 and k <= exhausted_at:
            assert kinds == {False, True}, f"one-sided batch of {len(b)} rows at {k}"
        # a batch pairs rows of similar length: sorted lists consumed in lockstep
        lens = [ds.get_frame_len(i) for i in b]
        assert max(lens) - min(lens) < 0.5 * max(lens) + 1

    plain = DynamicBatchSampler(SequentialSampler(ds), budget, max_samples=0, random_seed=0)
    one_sided = lambda s: sum(len({i >= n_real for i in b}) == 1 for b in s.batches) / len(s.batches)  # noqa: E731
    assert one_sided(paired) < one_sided(plain)
    assert one_sided(paired) < 0.05  # what is left is the tail of the side that ran out last

    # shuffling is seeded and per epoch, like the parent
    first = list(iter(paired))
    paired.set_epoch(1)
    second = list(iter(paired))
    assert first != second and sorted(map(tuple, first)) == sorted(map(tuple, second))
    paired.set_epoch(0)
    assert list(iter(paired)) == first
    assert len(paired) == len(paired.batches) and paired.drop_last is True


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


def test_gen_fake_pool_shards_compose_the_same_pool(tmp_path, capsys):
    """n shards of one plan write disjoint slices of one identical pool, and the metadata
    only appears once every clip exists -- a raw/ listing clips that are still being
    generated on another GPU would take the training run down mid-epoch."""
    gen = _gen_module()
    cfg, config_path = _tiny_config(tmp_path)
    ckpt = _tiny_checkpoint(tmp_path, cfg)
    ref_json = _ref_durations_file(tmp_path, [0.35, 0.5, 0.8])

    whole = tmp_path / "whole"
    gen.main(_gen_argv(tmp_path, ckpt, config_path, ref_json, whole, **{"--max_batch_frames": 400}))
    expected = {p.name: sf.read(str(p), dtype="float32")[0] for p in sorted((whole / "wavs").glob("*.wav"))}
    assert len(expected) > 2

    sharded = tmp_path / "sharded"
    capsys.readouterr()
    # a budget of 400 frames holds ~5 of these half-second clips, so the pool spans several
    # batches and both shards have work; the default would pack all of it into one batch
    gen.main(_gen_argv(tmp_path, ckpt, config_path, ref_json, sharded, **{"--shard": "1/2", "--max_batch_frames": 400}))
    assert "not written" in capsys.readouterr().out
    assert not (sharded / "raw").exists() and not (sharded / "duration.json").exists()
    first_half = {p.name for p in (sharded / "wavs").glob("*.wav")}
    assert 0 < len(first_half) < len(expected)

    gen.main(_gen_argv(tmp_path, ckpt, config_path, ref_json, sharded, **{"--shard": "0/2", "--max_batch_frames": 400}))
    assert (sharded / "raw").exists() and (sharded / "duration.json").exists()
    got = {p.name: sf.read(str(p), dtype="float32")[0] for p in sorted((sharded / "wavs").glob("*.wav"))}
    assert got.keys() == expected.keys()
    for name in expected:
        assert np.array_equal(got[name], expected[name]), name
    assert json.load(open(sharded / "duration.json")) == json.load(open(whole / "duration.json"))
    assert not list(sharded.glob("*.tmp*")), "staging files left behind"

    with pytest.raises(SystemExit):
        gen.parse_shard("2/2")


def test_gen_fake_pool_resumes_without_changing_the_pool(tmp_path, capsys):
    gen = _gen_module()
    cfg, config_path = _tiny_config(tmp_path)
    ckpt = _tiny_checkpoint(tmp_path, cfg)
    ref_json = _ref_durations_file(tmp_path, [0.35, 0.5, 0.8])
    out = tmp_path / "pool"

    gen.main(_gen_argv(tmp_path, ckpt, config_path, ref_json, out, **{"--max_batch_frames": 400}))
    first = {p.name: sf.read(str(p), dtype="float32")[0] for p in sorted((out / "wavs").glob("*.wav"))}

    # a complete pool costs nothing to "resume"
    capsys.readouterr()
    gen.main(_gen_argv(tmp_path, ckpt, config_path, ref_json, out, **{"--max_batch_frames": 400}))
    assert "wrote 0 new clips" in capsys.readouterr().out

    # drop a few clips and re-run: the missing ones come back byte-identical, because the
    # clip list and the clip -> batch assignment are planned from the seed, not from what
    # happens to be on disk. Resume granularity is the batch, not the clip -- a partially
    # written batch is redone whole so its shared seed still yields the same waveforms --
    # so this rewrites more than the two files, but strictly fewer than all of them.
    for name in list(first)[:2]:
        os.remove(out / "wavs" / name)
    capsys.readouterr()
    gen.main(_gen_argv(tmp_path, ckpt, config_path, ref_json, out, **{"--max_batch_frames": 400}))
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


def test_plan_batches_packs_sorted_lengths_under_the_paid_budget():
    """Batches mix lengths; what is budgeted is rows x the longest row, padding included."""
    gen = _gen_module()
    rng = torch.Generator().manual_seed(3)
    lengths = (torch.randint(30, 300, (500,), generator=rng) * 160).tolist()  # 0.3-3 s, whole frames
    budget = 3200
    batches = gen.plan_batches(lengths, frame_samples=160, max_batch_frames=budget)

    seen = sorted(i for b in batches for i in b)
    assert seen == list(range(len(lengths))), "every clip exactly once"
    for b in batches:
        widest = max(-(-lengths[i] // 160) for i in b)
        assert len(b) == 1 or len(b) * widest <= budget, "padded size must fit the budget"
        assert lengths[b[-1]] >= lengths[b[0]], "sorted, so neighbours pad each other little"
    paid = sum(len(b) * max(-(-lengths[i] // 160) for i in b) for b in batches)
    useful = sum(-(-n // 160) for n in lengths)
    assert (paid - useful) / paid < 0.05, "padding waste stays small on sorted lengths"

    # equal-length grouping of the same clips: the long tail of one-of-a-kind lengths ran
    # a batch each; mixed packing must launch far fewer
    by_len = {}
    for i, n in enumerate(lengths):
        by_len.setdefault(n, []).append(i)
    exact = sum(-(-len(m) // max(1, budget // (n // 160))) for n, m in by_len.items())
    assert len(batches) < 0.6 * exact

    assert batches == gen.plan_batches(lengths, frame_samples=160, max_batch_frames=budget), "deterministic"

    # a lone clip over the budget still gets a batch of its own rather than being dropped
    assert gen.plan_batches([160 * 50], frame_samples=160, max_batch_frames=10) == [[0]]


def test_sample_with_lens_matches_generating_each_row_alone():
    """A row in a mixed-length batch is the clip it would be on its own at that length.

    The mask keeps padded keys out of attention, the conv position embedding and the
    entropy scaling, so the padded lane never reaches the valid region. Row 0's noise is
    the same stream prefix whether the tensor is [2, long] or [2, short], which is what
    makes the two calls comparable at a fixed seed.
    """
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    from test_ddo import _make_cfm

    torch.manual_seed(0)
    model = _make_cfm(dit_kwargs={"attn_mask_enabled": True, "logn_ref_len": 500}).eval()
    long, short = 1600, 960

    mixed, _ = model.sample(lens=[long, short], steps=2, seed=11)
    alone, _ = model.sample(long, batch=2, steps=2, seed=11)
    assert mixed.shape == (2, long)
    assert torch.allclose(mixed[0], alone[0], atol=1e-5)

    mixed, _ = model.sample(lens=[short, long], steps=2, seed=11)
    alone, _ = model.sample(short, batch=2, steps=2, seed=11)
    assert torch.allclose(mixed[0, :short], alone[0], atol=1e-5)

    mixed, _ = model.sample(lens=[short, long], steps=2, seed=11, solver="dpmpp")
    alone, _ = model.sample(short, batch=2, steps=2, seed=11, solver="dpmpp")
    assert torch.allclose(mixed[0, :short], alone[0], atol=1e-5)

    with pytest.raises(ValueError):
        model.sample(long, lens=[long], steps=2)
    with pytest.raises(ValueError):
        model.sample(steps=2)


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
