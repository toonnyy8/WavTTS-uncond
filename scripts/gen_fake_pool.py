"""Generate the offline fake-sample pool p_ref draws for DDO finetuning.

Online generation is not an option: one fake sample costs 32 NFE of a 664M model, an
order of magnitude more than the training step it would feed. The paper's diffusion
setting is offline too (50k images regenerated per round). So: generate a pool once per
round, reuse it, regenerate when the reference model changes.

The output directory is byte-for-byte the layout of a real prepared corpus --
`<out>/raw` (a HF Dataset with audio_path/duration/text) plus `<out>/duration.json` --
so that `load_dataset()` reads it back through the *same* CustomDataset with the *same*
waveform_kwargs as the real data. That is not a convenience. DDO turns the model into
its own discriminator, and the discriminator will separate real from fake on whatever
feature is cheapest, including features that say nothing about speech quality. Sharing
the loading path neutralises three of them (spec 3.5):

  * loudness   -- both halves get RMS-normalised to target_rms at load time
  * frame grid -- both halves get the same sub-frame jitter at load time
  * precision  -- see the PCM_F note on _write_wav()

The fourth, the length distribution, has to be handled here: durations are drawn with
replacement from the real corpus' own duration.json.

Usage:

    uv run python scripts/gen_fake_pool.py \\
      --ckpt ckpts/<run>/model_last.pt \\
      --config src/wavtts/configs/WavTTS_clean.yaml \\
      --ref_durations data/LibriTTS_460/duration.json \\
      --out data/LibriTTS_460_fake_r1 \\
      --hours 50 --steps 32 --cfg_strength 0.0 --solver euler \\
      --max_batch_frames 3200 --seed 1234 --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import numpy as np
import soundfile as sf
import torch
from datasets import Dataset as Dataset_
from hydra.utils import get_class
from omegaconf import OmegaConf
from tqdm import tqdm

from wavtts.model import CFM
from wavtts.model.ddo import load_ref_state_dict


# CustomDataset.__getitem__ silently skips clips outside this window, so anything we
# generate outside it would either never be trained on or, worse, push the loader onto a
# neighbouring index. Matching the filter here keeps the drawn distribution equal to the
# distribution the real half actually presents.
MIN_DURATION = 0.3
MAX_DURATION = 30.0


def read_reference_durations(path: str) -> list[float]:
    with open(path, "r", encoding="utf-8") as f:
        durations = json.load(f)["duration"]
    kept = [float(d) for d in durations if MIN_DURATION <= float(d) <= MAX_DURATION]
    if not kept:
        raise ValueError(f"{path} has no durations inside [{MIN_DURATION}, {MAX_DURATION}] s")
    return kept


def sample_clip_lengths(
    ref_durations: list[float],
    total_seconds: float,
    *,
    frame_samples: int,
    sample_rate: int,
    generator: torch.Generator,
) -> list[int]:
    """Clip lengths in samples, drawn *with replacement* from the empirical distribution.

    Not a fixed length and not a uniform range. Length is the single cheapest shortcut
    available to the discriminator: if the fake pool were, say, all 5 s, then `len(x)`
    alone separates the two halves perfectly and Delta can be driven arbitrarily far
    apart without the model ever learning anything about speech (spec 3.5, shortcut 1).
    An empirical resample makes the marginal over length carry no information at all.

    Every draw is rounded to a whole number of frames, because that is the only length
    CFM.sample() can produce (it ceils to the frame grid internally); rounding here means
    the duration we record is the duration the file actually has.
    """
    if not ref_durations:
        raise ValueError("no reference durations to draw from")

    ref = torch.tensor(ref_durations, dtype=torch.float64)
    min_samples = frame_samples
    lengths: list[int] = []
    cumulative = 0.0
    # draw in blocks rather than one at a time: 4096 multinomial draws cost the same as one
    block = 4096
    while cumulative < total_seconds:
        idx = torch.randint(0, ref.numel(), (block,), generator=generator)
        for d in ref[idx].tolist():
            n_frames = max(1, round(d * sample_rate / frame_samples))
            n_samples = max(min_samples, n_frames * frame_samples)
            lengths.append(n_samples)
            cumulative += n_samples / sample_rate
            if cumulative >= total_seconds:
                break
    return lengths


def plan_batches(lengths: list[int], *, frame_samples: int, max_batch_frames: int) -> list[list[int]]:
    """Pack clip indices, sorted by length, into batches whose *padded* size fits the budget.

    A batch holds clips of different lengths, generated together the way training runs
    them: padded to the longest row, each row carrying its own length through a mask, so
    every row comes out exactly as it would have alone (CFM.sample's `lens`). What the GPU
    pays for is rows x longest row -- padding included -- so that is what is budgeted, and
    sorting first keeps neighbours within a frame or two of each other: on the real pool
    the padding waste is 0.2% and the launch count drops 2.3x against grouping by exact
    length, whose long tail of one-of-a-kind lengths ran 42% full on average.

    This is not "generate long and trim": trimming a longer utterance would leave every
    fake clip ending on an abrupt cut mid phone, a shortcut recorded speech never offers.
    The model is told each row's length and plans that much speech; only the padding lane,
    which it never reads, is discarded.

    The plan is computed from the full clip list and never from "what is still missing",
    so a resumed or sharded run reproduces the same batches, the same seeds and therefore
    the same waveforms as an uninterrupted one.
    """
    order = sorted(range(len(lengths)), key=lambda i: (lengths[i], i))
    batches: list[list[int]] = []
    current: list[int] = []
    longest = 0
    for i in order:
        n_frames = -(-lengths[i] // frame_samples)  # ceil: the padded width is a whole frame count
        widest = max(longest, n_frames)
        if current and (len(current) + 1) * widest > max_batch_frames:
            batches.append(current)
            current, widest = [], n_frames
        current.append(i)
        longest = widest
    if current:
        batches.append(current)
    return batches


def build_model(ckpt_path: str, cfg, device: str) -> CFM:
    """The reference model, loaded from its EMA weights.

    EMA, not the raw weights: "the fakes came from p_ref" is the premise the whole
    likelihood-ratio identity rests on, and p_ref in CFM's Delta is the EMA copy. Two
    different weight sets on the two sides of that identity is a silent bug, not an
    approximation. load_ref_state_dict is shared with the training path for that reason.
    """
    model_cls = get_class(f"wavtts.model.{cfg.model.backbone}")
    model = CFM(
        transformer=model_cls(
            **OmegaConf.to_container(cfg.model.arch, resolve=True),
            wav_frame_len=cfg.model.waveform.wav_frame_len,
        ),
        waveform_kwargs=OmegaConf.to_container(cfg.model.waveform, resolve=True),
        **OmegaConf.to_container(cfg.model.cfm, resolve=True),
    ).to(device)
    model.load_state_dict(load_ref_state_dict(ckpt_path))
    model.eval()
    return model


def _write_wav(path: str, wav: torch.Tensor, sample_rate: int) -> None:
    """32-bit float WAV, no peak normalisation, no clipping anywhere on the way out.

    The model generates at target_rms with speech's crest factor of ~7, so its output
    leaves +-1 on essentially every clip. Writing 16-bit PCM would clip those peaks, and
    "has been clipped" is a property no recorded clip in the corpus has -- it would be
    handed to the discriminator as a perfect, free separator (spec 3.5, shortcut 4).
    peak_normalize() is the same mistake in a quieter form: it would make peak level a
    constant on the fake half and a variable on the real half.

    soundfile rather than torchaudio.save(encoding="PCM_F", bits_per_sample=32):
    torchaudio >= 2.9 routes save() through TorchCodec, which *ignores* both of those
    arguments (it warns, it does not fail) and encodes 16-bit PCM, clipping to +-1. The
    exact failure this function exists to prevent, delivered by the API that names it.
    soundfile is already a declared dependency and writes WAVE_FORMAT_IEEE_FLOAT, which
    torchaudio.load() reads back unclipped.
    """
    data = wav.detach().to(torch.float32).cpu().numpy()
    if data.ndim > 1:
        data = data.squeeze(0)
    # Written beside the target and renamed into place. main() takes an existing file as
    # a finished clip when it resumes, so a run killed mid-write would otherwise leave a
    # truncated wav that every later resume steps over as done.
    tmp = f"{path}.tmp"  # soundfile reads the container off the extension, hence format=
    sf.write(tmp, data.astype(np.float32), sample_rate, format="WAV", subtype="FLOAT")
    os.replace(tmp, path)


def _quantile_report(name: str, values: list[float]) -> str:
    t = torch.tensor(values, dtype=torch.float64)
    qs = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], dtype=torch.float64)
    q = torch.quantile(t, qs).tolist()
    return (
        f"{name:>10}: n={t.numel():>7d}  total={t.sum().item() / 3600:7.2f} h  mean={t.mean().item():5.2f} s  "
        f"p05={q[0]:5.2f}  p25={q[1]:5.2f}  p50={q[2]:5.2f}  p75={q[3]:5.2f}  p95={q[4]:5.2f}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an offline fake-sample pool for DDO")
    parser.add_argument("--ckpt", required=True, help="reference checkpoint (.pt); EMA weights are used")
    parser.add_argument("--config", required=True, help="the training yaml the checkpoint was trained with")
    parser.add_argument("--ref_durations", required=True, help="duration.json of the real corpus")
    parser.add_argument("--out", required=True, help="output dir, e.g. data/LibriTTS_460_fake_r1")
    parser.add_argument("--hours", type=float, default=50.0, help="total generated duration")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument(
        "--cfg_strength",
        type=float,
        default=0.0,
        help=(
            "keep at 0. DDO takes p_ref and p_theta to be the guidance-free model; a guided "
            "sampler draws from a different distribution than the p_ref that appears in Delta, "
            "which invalidates the likelihood-ratio identity the loss is built on. Non-zero is "
            "an ablation knob only. (The official VAR script passes cfg=1.0 and calls it "
            "guidance-free, but under VAR's t = cfg*ratio that is a weak 0->1 ramp, not none -- "
            "paper and code disagree there; we follow the paper.)"
        ),
    )
    parser.add_argument("--solver", choices=["euler", "dpmpp"], default="euler")
    parser.add_argument(
        "--sway_sampling_coef",
        type=float,
        default=-1.0,
        help="matches the trainer's built-in sampling, so the pool looks like what we listen to",
    )
    parser.add_argument("--max_batch_frames", type=int, default=3200)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default=None, help="cpu | cuda | cuda:N (default: auto)")
    parser.add_argument(
        "--autocast",
        choices=["bf16", "none"],
        default="bf16",
        help=(
            "run the backbone under bf16 autocast (cuda only; ~2x on an RTX 4090, whose bf16 tensor "
            "cores do twice its fp32 rate). The ODE state and the noise stay fp32. This is also the "
            "precision the trainer evaluates p_ref at inside Delta, so bf16 samples are samples of "
            "the p_ref the loss actually sees. 'none' reproduces the round-1 pool's fp32 path"
        ),
    )
    parser.add_argument(
        "--shard",
        default="0/1",
        help=(
            "i/n: this process generates only the batches with index %% n == i. The clip list and "
            "the clip -> batch plan are a pure function of --seed, so n shards launched with the "
            "same arguments (any order, any devices) produce disjoint slices of one identical pool; "
            "the shard that finds every clip on disk writes raw/ and duration.json"
        ),
    )
    return parser


def parse_shard(spec: str) -> tuple[int, int]:
    try:
        idx, n = (int(x) for x in spec.split("/"))
    except ValueError:
        raise SystemExit(f"--shard expects i/n, got {spec!r}") from None
    if n < 1 or not 0 <= idx < n:
        raise SystemExit(f"--shard {spec!r}: need 0 <= i < n")
    return idx, n


def _write_metadata(out: str, paths: list[str], durations: list[float]) -> None:
    """raw/ and duration.json, each staged and renamed so two shards finishing at the same
    moment cannot half-write the same files; the content is identical either way."""
    raw = os.path.join(out, "raw")
    staged = f"{raw}.tmp.{os.getpid()}"
    Dataset_.from_dict(
        {
            "audio_path": [os.path.abspath(p) for p in paths],
            "duration": durations,
            "text": [""] * len(paths),
        }
    ).save_to_disk(staged)
    try:
        os.rename(staged, raw)  # fails, rather than merges, if another shard got there first
    except OSError:
        shutil.rmtree(staged, ignore_errors=True)

    tmp = os.path.join(out, f"duration.json.tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"duration": durations}, f)
    os.replace(tmp, os.path.join(out, "duration.json"))


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    shard_idx, shard_n = parse_shard(args.shard)

    cfg = OmegaConf.load(args.config)
    sample_rate = int(cfg.model.cfm.sample_rate)
    frame_samples = int(cfg.model.waveform.wav_frame_len)

    ref_durations = read_reference_durations(args.ref_durations)

    # One generator, seeded once: the clip list is a pure function of (--seed,
    # --ref_durations, --hours), so a re-run -- or a resumed run -- plans the same pool.
    generator = torch.Generator().manual_seed(int(args.seed))
    lengths = sample_clip_lengths(
        ref_durations,
        args.hours * 3600.0,
        frame_samples=frame_samples,
        sample_rate=sample_rate,
        generator=generator,
    )
    batches = plan_batches(lengths, frame_samples=frame_samples, max_batch_frames=args.max_batch_frames)

    wav_dir = os.path.join(args.out, "wavs")
    os.makedirs(wav_dir, exist_ok=True)
    paths = [os.path.join(wav_dir, f"fake_{i:07d}.wav") for i in range(len(lengths))]

    model = build_model(args.ckpt, cfg, device)
    use_autocast = args.autocast == "bf16" and str(device).startswith("cuda")

    mine = [(i, b) for i, b in enumerate(batches) if i % shard_n == shard_idx]
    desc = f"Generating {len(lengths)} clips" + (f" (shard {shard_idx}/{shard_n}, {len(mine)} batches)" if shard_n > 1 else "")
    n_written = 0
    with torch.inference_mode():
        for batch_idx, members in tqdm(mine, desc=desc):
            # Resume at batch granularity: a partially written batch is regenerated whole so
            # that its shared seed still produces the waveforms the complete run would have.
            if all(os.path.exists(paths[i]) for i in members):
                continue

            # Per-batch derived seed rather than per-clip: CFM.sample() seeds one generator
            # for the whole batch, so per-clip seeding would force batch size 1 and ~n_clips
            # times the wall clock. Determinism is preserved instead by fixing the
            # clip -> batch assignment in plan_batches().
            batch_seed = (int(args.seed) * 1_000_003 + batch_idx) % (2**31 - 1)
            row_lens = [lengths[i] for i in members]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_autocast):
                wavs, _ = model.sample(
                    lens=row_lens,
                    steps=args.steps,
                    cfg_strength=args.cfg_strength,
                    sway_sampling_coef=args.sway_sampling_coef,
                    seed=batch_seed,
                    solver=args.solver,
                )
            for row, i in enumerate(members):
                _write_wav(paths[i], wavs[row, : row_lens[row]], sample_rate)
                n_written += 1

    durations = [n / sample_rate for n in lengths]

    missing = sum(not os.path.exists(p) for p in paths)
    if missing:
        # another shard's slice is still being generated: the pool is not a dataset yet,
        # and a raw/ listing clips that do not exist would make load_dataset() crash mid-epoch
        print(
            f"\nwrote {n_written} new clips; {missing} of {len(paths)} still missing (other shards "
            f"unfinished?). raw/ and duration.json not written -- re-run any shard once they are."
        )
        return

    # The real corpus' format exactly: load_dataset() must read this back with no special
    # casing. text is present and empty -- nothing in an unconditional model reads it, but
    # the column exists in the real arrow and the schemas should not diverge.
    _write_metadata(args.out, paths, durations)

    print(f"\nwrote {n_written} new clips ({len(paths)} in pool) to {args.out}")
    print("duration distributions, for eyeballing against each other:")
    print(_quantile_report("reference", ref_durations))
    print(_quantile_report("generated", durations))


if __name__ == "__main__":
    main()
