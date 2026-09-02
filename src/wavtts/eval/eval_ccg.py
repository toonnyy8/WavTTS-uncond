"""How good, and how coherent, is a long unconditional generation?

No reference and no speaker conditioning: the model generates a clip from noise and the
question is whether it sounds good and holds together over its whole length. Coherence
here is a property of the clip, not a match to anyone — a clip whose voice, room and
recording character wander partway through is incoherent regardless of who it started as.
This is groundwork for zero-shot editing, where a prior that stays consistent over a long
window is what makes an edit blend into its surroundings.

  quality    UTMOS over the whole clip — the headline
  drift      each segment against the first, and the worst point it reaches
  adjacent   neighbouring segments, plus the switch rate (fraction of joins below
             `--switch_threshold`): a clip can hold a steady average while flipping
             character at the seams

Coherence is measured with a speaker embedding because nothing else is as sensitive to
"this stopped being the same recording", not because identity is the goal.

Conditions:

  real      a genuine continuous recording chopped the same way — the ceiling, and where
            the embedder's own noise floor shows
  full      one single-pass generation of the whole clip. The baseline worth beating, and
            the one that degrades as the clip outgrows the training length
  ola_w0    overlapping windows, velocities overlap-added, every ODE step taken on the
            whole waveform. Each sample sees only its own window
  ola_w<S>  the same, guided toward the full pass: v = v_ola + S·(v_full − v_ola), so
            S=0 is `ola_w0`, S=1 is `full`, and in between is a blend
  naive     independent segments concatenated, no context at all — the floor
  w<S>      the segment-by-segment sampler with the previous tail pinned in (plan §4)

  python -m wavtts.eval.eval_ccg --ckpt CKPT --config CFG --conditions real full ola_w0 ola_w0.5
"""

from __future__ import annotations

import argparse
import os
import re
from importlib.resources import files
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from omegaconf import OmegaConf

from wavtts.eval.eval_spk_cond import SpeakerEmbedder
from wavtts.infer.sample_uncond import load_model
from wavtts.infer.spk_cond import CCG, generate_long, generate_ola
from wavtts.model.utils import peak_normalize
from wavtts.train.metrics import GenMetrics


def parse_condition(cond: str, args) -> CCG | None:
    """Condition name -> guidance config, or None for the two reference points.

    "w2_lp8_apg0" is guidance strength 2, Φ_LP factor 8, Φ_APG eta 0. Unsuffixed
    operators fall back to the command line, so one flag sweeps a whole run.
    """
    if cond in ("naive", "real", "full"):
        return None
    cond = cond.removeprefix("ola_")
    m = re.fullmatch(r"w([0-9.]+)(?:_lp(\d+))?(?:_apg([0-9.]+))?(?:_k([0-9.]+))?(?:_win([0-9.]+)-([0-9.]+))?", cond)
    if m is None:
        raise ValueError(f"Unknown condition: {cond}")
    window = (float(m.group(5)), float(m.group(6))) if m.group(5) else args.ccg_window
    return CCG(
        w=float(m.group(1)),
        window=tuple(window) if window else None,
        lp_k=int(m.group(2)) if m.group(2) else args.ccg_lp_k,
        eta_apg=float(m.group(3)) if m.group(3) else args.ccg_eta_apg,
        rescale=args.repl_cfg_rescale,
        kappa=float(m.group(4)) if m.group(4) else args.ccg_kappa,
    )


def continuous_recordings(root: str, min_sec: float, sample_rate: int, target_rms: float):
    """One continuous stretch per chapter: consecutive files are the same person reading
    on, so concatenating them gives a real long recording to compare rollouts against."""
    for chapter in sorted(Path(root).glob("*/*")):
        parts, total = [], 0
        for w in sorted(chapter.glob("*.wav")):
            audio, sr = torchaudio.load(str(w))
            audio = audio.mean(dim=0, keepdim=True)
            if sr != sample_rate:
                audio = torchaudio.functional.resample(audio, sr, sample_rate)
            parts.append(audio)
            total += audio.shape[1]
            if total >= min_sec * sample_rate:
                break
        if total < min_sec * sample_rate:
            continue
        audio = torch.cat(parts, dim=-1)
        if target_rms > 0:  # same domain the model's own output lives in
            audio = audio - audio.mean()
            rms = audio.pow(2).mean().sqrt()
            if rms > 1e-5:
                audio = audio * (target_rms / rms)
        yield chapter.parent.name, audio


def segment_scores(wav: torch.Tensor, embed, seg: int, threshold: float):
    """Similarity of every segment to the first, and between neighbours."""
    segs = [wav[:, i : i + seg] for i in range(0, wav.shape[1] - seg // 2, seg)]
    embs = [embed(s) for s in segs if s.shape[1] > seg // 2]
    cos = F.cosine_similarity
    to_first = [float(cos(embs[0], e, dim=-1)) for e in embs[1:]]
    adjacent = [float(cos(a, b, dim=-1)) for a, b in zip(embs, embs[1:])]
    switches = sum(1 for a in adjacent if a < threshold) / max(len(adjacent), 1)
    return to_first, adjacent, switches


def summarize(name: str, rows: dict[str, list[float]]) -> str:
    def stat(key, fmt="{:.3f}"):
        return fmt.format(float(torch.tensor(rows[key]).mean()))

    return (
        f"{name:<14} drift {stat('first')} (worst {stat('first_min')})  "
        f"adjacent {stat('adjacent')} (worst {stat('adjacent_min')})  "
        f"switch {stat('switch', '{:.2f}')}  UTMOS {stat('utmos', '{:.2f}')}"
    )


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Context consistency of long unconditional rollouts")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--num_clips", type=int, default=8, help="independent rollouts per condition")
    parser.add_argument("--total_sec", type=float, default=30.0)
    parser.add_argument("--seg_sec", type=float, default=5.0)
    parser.add_argument("--ctx_sec", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--cfg_strength", type=float, default=0.0)
    parser.add_argument("--resample_rounds", type=int, default=4)
    parser.add_argument("--resample_range", type=float, nargs=2, default=(0.15, 0.5), metavar=("LO", "HI"))
    parser.add_argument("--conditions", nargs="+", default=["real", "full", "ola_w0", "ola_w0.5"])
    parser.add_argument("--repl_cfg_rescale", type=float, default=1.0)
    parser.add_argument("--ccg_window", type=float, nargs=2, default=None, metavar=("T_LO", "T_HI"))
    parser.add_argument("--ccg_lp_k", type=int, default=1)
    parser.add_argument("--ccg_eta_apg", type=float, default=1.0)
    parser.add_argument("--ccg_kappa", type=float, default=1.0)
    parser.add_argument("--win_sec", type=float, default=10.0, help="ola_* conditions: window the model sees")
    parser.add_argument("--ola_hop_sec", type=float, default=5.0, help="ola_* conditions: spacing between windows")
    parser.add_argument("--switch_threshold", type=float, default=0.8)
    parser.add_argument("--real_root", default="/media/8tsp/dataset/LibriTTS/test-other")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    config_path = args.config or str(files("wavtts").joinpath("configs/WavTTS.yaml"))
    cfg = OmegaConf.load(config_path)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, cfg, device)
    embed = SpeakerEmbedder(device)
    metrics = GenMetrics(sample_rate=cfg.model.cfm.sample_rate)

    sr = cfg.model.cfm.sample_rate
    t = model.timesteps(args.steps, device=device)
    seg, total = int(args.seg_sec * sr), int(args.total_sec * sr)
    ccgs = {c: parse_condition(c, args) for c in args.conditions}

    real = []
    if "real" in ccgs:
        gen = continuous_recordings(args.real_root, args.total_sec, sr, cfg.model.waveform.target_rms)
        real = [next(gen) for _ in range(args.num_clips)]

    print(f"{args.num_clips} rollouts of {args.total_sec:g}s in {args.seg_sec:g}s segments, no reference\n")
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    keys = ("first", "first_min", "adjacent", "adjacent_min", "switch", "utmos")
    rows = {c: {k: [] for k in keys} for c in args.conditions}
    for idx in range(args.num_clips):
        for cond, ccg in ccgs.items():
            name = f"clip{idx}"
            if cond == "real":
                name, wav = real[idx]
                wav = wav[:, :total].to(device)
            elif cond == "full" or cond.startswith("ola_"):
                wav = generate_ola(
                    model,
                    total,
                    t,
                    # win >= total collapses the short branch onto the long one, which
                    # is exactly the single-pass baseline
                    win=total if cond == "full" else int(args.win_sec * sr),
                    hop=total if cond == "full" else int(args.ola_hop_sec * sr),
                    cfg_strength=args.cfg_strength,
                    resample_rounds=args.resample_rounds,
                    resample_range=tuple(args.resample_range),
                    ccg=ccg or CCG(),
                    seed=args.seed * 1000 + idx,
                )
            else:
                wav = generate_long(
                    model,
                    total,
                    t,
                    seg_samples=seg,
                    # "naive" carries nothing between segments: every one comes from the
                    # bare prior, which is the drift everything else is measured against
                    ctx_samples=0 if ccg is None else int(args.ctx_sec * sr),
                    init_ref=None,  # the model picks its own speaker in segment 1
                    cfg_strength=args.cfg_strength,
                    resample_rounds=args.resample_rounds,
                    resample_range=tuple(args.resample_range),
                    ccg=ccg or CCG(),
                    seed=args.seed * 1000 + idx,
                )
            to_first, adjacent, switch = segment_scores(wav, embed, seg, args.switch_threshold)
            mos = metrics.utmos(wav.reshape(1, -1), device)
            r = rows[cond]
            r["first"].append(sum(to_first) / len(to_first))
            r["first_min"].append(min(to_first))
            r["adjacent"].append(sum(adjacent) / max(len(adjacent), 1))
            r["adjacent_min"].append(min(adjacent, default=1.0))
            r["switch"].append(switch)
            r["utmos"].append(mos if mos is not None else float("nan"))
            if args.save_dir:
                suffix = f"_{name}" if cond == "real" else ""
                out = Path(args.save_dir) / f"{idx:02d}_{cond}{suffix}.wav"
                torchaudio.save(str(out), peak_normalize(wav.reshape(1, -1).float().cpu()), sr)

        print(
            f"[{idx + 1:3d}/{args.num_clips}] "
            + "  ".join(f"{c}={rows[c]['first'][-1]:.3f}/{rows[c]['adjacent'][-1]:.3f}" for c in args.conditions)
        )

    print("\n(drift = later segments vs the first, adjacent = neighbouring segments)")
    for cond in args.conditions:
        print(summarize(cond, rows[cond]))


if __name__ == "__main__":
    main()
