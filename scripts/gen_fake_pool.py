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
    """Group clip indices into batches of *identical* frame count.

    Deliberately not length-bucketing with a crop at the end: generating a bucket at its
    maximum length and trimming would leave every fake clip ending on an abrupt cut mid
    phone, which recorded speech never does -- a fifth shortcut, and a self-inflicted one.
    CFM.sample() takes one duration per call, so equal-length grouping is also simply what
    the API wants.

    The grouping is computed from the full clip list and never from "what is still
    missing", so a resumed run reproduces the same batches, the same seeds and therefore
    the same waveforms as an uninterrupted one.
    """
    by_len: dict[int, list[int]] = {}
    for i, n in enumerate(lengths):
        by_len.setdefault(n, []).append(i)

    batches: list[list[int]] = []
    for n_samples in sorted(by_len):
        n_frames = n_samples // frame_samples
        per_batch = max(1, max_batch_frames // n_frames)
        members = by_len[n_samples]
        for start in range(0, len(members), per_batch):
            batches.append(members[start : start + per_batch])
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
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

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

    n_written = 0
    with torch.inference_mode():
        for batch_idx, members in enumerate(tqdm(batches, desc=f"Generating {len(lengths)} clips")):
            # Resume at batch granularity: a partially written batch is regenerated whole so
            # that its shared seed still produces the waveforms the complete run would have.
            if all(os.path.exists(paths[i]) for i in members):
                continue

            # Per-batch derived seed rather than per-clip: CFM.sample() seeds one generator
            # for the whole batch, so per-clip seeding would force batch size 1 and ~n_clips
            # times the wall clock. Determinism is preserved instead by fixing the
            # clip -> batch assignment in plan_batches().
            batch_seed = (int(args.seed) * 1_000_003 + batch_idx) % (2**31 - 1)
            wavs, _ = model.sample(
                lengths[members[0]],
                batch=len(members),
                steps=args.steps,
                cfg_strength=args.cfg_strength,
                sway_sampling_coef=args.sway_sampling_coef,
                seed=batch_seed,
                solver=args.solver,
            )
            for row, i in enumerate(members):
                _write_wav(paths[i], wavs[row], sample_rate)
                n_written += 1

    durations = [n / sample_rate for n in lengths]

    # The real corpus' format exactly: load_dataset() must read this back with no special
    # casing. text is present and empty -- nothing in an unconditional model reads it, but
    # the column exists in the real arrow and the schemas should not diverge.
    Dataset_.from_dict(
        {
            "audio_path": [os.path.abspath(p) for p in paths],
            "duration": durations,
            "text": [""] * len(paths),
        }
    ).save_to_disk(os.path.join(args.out, "raw"))
    with open(os.path.join(args.out, "duration.json"), "w", encoding="utf-8") as f:
        json.dump({"duration": durations}, f)

    print(f"\nwrote {n_written} new clips ({len(paths)} in pool) to {args.out}")
    print("duration distributions, for eyeballing against each other:")
    print(_quantile_report("reference", ref_durations))
    print(_quantile_report("generated", durations))


if __name__ == "__main__":
    main()
