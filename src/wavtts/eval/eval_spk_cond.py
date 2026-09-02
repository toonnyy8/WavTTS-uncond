"""Does inversion/replacement conditioning actually carry the speaker?

Scores SIM-o (cosine similarity of speaker embeddings, generated vs. reference) for the
conditioned sampler against the two bounds that make the number readable:

  real_same   a different real utterance by the same speaker  — the ceiling any method
              can reach, and where the embedder's own noise floor shows
  uncond      a plain unconditional sample                    — the floor, i.e. what a
              random speaker scores against this reference
  real_diff   a real utterance by a different speaker         — chance, as a sanity check

Speakers come from LibriTTS test-other, which is disjoint from the train-clean-100/360
the checkpoints are trained on, so this is zero-shot on unseen speakers.

The embedder is microsoft/wavlm-base-plus-sv (WavLM + x-vector head, the model card's
same-speaker threshold is 0.86). That is a different network from the WavLM-large ECAPA
in eval/utils_eval.py, which needs a checkpoint this repo does not ship — the absolute
numbers are therefore not comparable with published SIM-o, but the ordering across
conditions is what the question is about.

  python -m wavtts.eval.eval_spk_cond --ckpt CKPT --config CFG --num_speakers 20
"""

from __future__ import annotations

import argparse
import os
import random
import re
from importlib.resources import files
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from omegaconf import OmegaConf

from wavtts.infer.sample_uncond import load_model
from wavtts.infer.spk_cond import CCG, prepare_ref, speaker_conditioned_sample
from wavtts.model.utils import peak_normalize
from wavtts.train.metrics import GenMetrics


SV_MODEL = "microsoft/wavlm-base-plus-sv"


class SpeakerEmbedder:
    def __init__(self, device, model_name: str = SV_MODEL):
        from transformers import AutoFeatureExtractor, WavLMForXVector

        self.fe = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = WavLMForXVector.from_pretrained(model_name).to(device).eval()
        self.device = device

    @torch.inference_mode()
    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        # the model's waveforms live at target_rms, outside +-1; pretrained audio models
        # want the conventional domain (same reason train/metrics.py peak-normalizes)
        wav = peak_normalize(wav.reshape(-1).float().cpu())
        inputs = self.fe(wav.numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return self.model(**inputs).embeddings[0]


def sample_speakers(root: str, num_speakers: int, min_sec: float, seed: int):
    """Pick (speaker, ref, held-out utterance by the same speaker) triples from a
    LibriTTS split laid out as <root>/<speaker>/<chapter>/*.wav."""
    rng = random.Random(seed)
    picks = []
    for spk_dir in sorted(Path(root).iterdir()):
        if not spk_dir.is_dir():
            continue
        wavs = sorted(spk_dir.rglob("*.wav"))
        long_enough = [w for w in wavs if _duration(w) >= min_sec]
        if len(long_enough) < 2:
            continue
        ref, other = rng.sample(long_enough, 2)
        picks.append((spk_dir.name, ref, other))
    rng.shuffle(picks)
    return picks[:num_speakers]


def _parse_condition(cond: str) -> tuple[str | None, int, float]:
    """Condition name -> (trajectory mode, resampling rounds, replacement guidance).

    "noise_u4_g1.5" is the forward-noising path with U=4 and repl_cfg=1.5."""
    m = re.fullmatch(r"(noise|invert)(?:_u(\d+))?(?:_g([0-9.]+))?", cond)
    if m is None:
        return None, 1, 0.0
    return m.group(1), int(m.group(2) or 1), float(m.group(3) or 0.0)


def _duration(path: Path) -> float:
    # torchaudio 2.11 dropped `info`; the header is 44 bytes of RIFF and reading the
    # whole file for a duration would dominate the scan
    size = os.path.getsize(path)
    return (size - 44) / (24000 * 2)  # LibriTTS is 24 kHz 16-bit mono


def load_wav(path, sample_rate: int, max_sec: float | None = None) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    if max_sec is not None:
        wav = wav[:, : int(max_sec * sample_rate)]
    return wav


def summarize(name: str, sims: list[float], floor: list[float] | None, ceil: list[float] | None) -> str:
    x = torch.tensor(sims)
    line = f"{name:<12} n={len(sims):3d}  SIM {x.mean():.3f} +- {x.std():.3f}  [{x.min():.3f}, {x.max():.3f}]"
    if floor is not None and len(floor) == len(sims):
        f = torch.tensor(floor)
        # paired against the same references, so the per-speaker delta is the readable
        # number: an unpaired mean hides how much of the spread is just voice-to-voice
        line += f"  |  vs uncond {(x - f).mean():+.3f} ({int((x > f).sum())}/{len(sims)} wins)"
        if ceil is not None and len(ceil) == len(sims):
            c = torch.tensor(ceil)
            line += f"  gap closed {float(((x - f).mean() / (c - f).mean())) * 100:5.1f}%"
    return line


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Speaker consistency of replacement-conditioned WavTTS")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--test_root", default="/media/8tsp/dataset/LibriTTS/test-other")
    parser.add_argument("--num_speakers", type=int, default=20)
    parser.add_argument("--ref_sec", type=float, default=5.0)
    parser.add_argument("--duration_sec", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--cfg_strength", type=float, default=0.0)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["noise", "uncond", "real_same", "real_diff"],
        help="noise[_uN][_gS] / invert[_uN][_gS] (N resampling rounds, S replacement guidance) "
        "/ uncond / real_same / real_diff",
    )
    parser.add_argument("--resample_range", type=float, nargs=2, default=(0.15, 0.5), metavar=("LO", "HI"))
    parser.add_argument("--repl_cfg_rescale", type=float, default=1.0)
    parser.add_argument("--ccg_window", type=float, nargs=2, default=None, metavar=("T_LO", "T_HI"))
    parser.add_argument("--ccg_lp_k", type=int, default=1)
    parser.add_argument("--ccg_eta_apg", type=float, default=1.0)
    parser.add_argument("--ccg_kappa", type=float, default=1.0)
    parser.add_argument("--invert_t_start", type=float, default=0.95)
    parser.add_argument("--invert_steps", type=int, default=None)
    parser.add_argument("--invert_renorm", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--utmos", action="store_true", help="also score naturalness (downloads SpeechMOS)")
    parser.add_argument("--save_dir", default=None, help="write the generated wavs here for listening")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    config_path = args.config or str(files("wavtts").joinpath("configs/WavTTS.yaml"))
    cfg = OmegaConf.load(config_path)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, cfg, device)
    embed = SpeakerEmbedder(device)

    sr = cfg.model.cfm.sample_rate
    t = model.timesteps(args.steps, device=device)
    duration = int(args.duration_sec * sr)
    dtype = next(model.parameters()).dtype

    picks = sample_speakers(args.test_root, args.num_speakers, args.ref_sec + 0.5, args.seed)
    if not picks:
        raise SystemExit(f"no usable speakers under {args.test_root}")
    print(f"{len(picks)} unseen speakers from {args.test_root}\n")
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    sims = {c: [] for c in args.conditions}
    moss = {c: [] for c in args.conditions}
    metrics = GenMetrics(sample_rate=sr) if args.utmos else None
    snrs = []
    for idx, (spk, ref_path, other_path) in enumerate(picks):
        ref = prepare_ref(
            str(ref_path),
            sample_rate=sr,
            target_rms=cfg.model.waveform.target_rms,
            hop=model.wav_frame_hop,
            max_sec=args.ref_sec,
            device=device,
            dtype=dtype,
        )
        ref_emb = embed(ref)
        seed = args.seed * 1000 + idx
        if args.save_dir:
            # the reference every score in this row is measured against, saved in the
            # same domain the model saw it in, so a listener compares like with like
            ref_out = Path(args.save_dir) / f"{spk}_ref.wav"
            torchaudio.save(str(ref_out), peak_normalize(ref.reshape(1, -1).float().cpu()), sr)

        for cond in args.conditions:
            traj_mode, rounds, repl_cfg = _parse_condition(cond)
            if traj_mode is not None:
                wav, snr = speaker_conditioned_sample(
                    model,
                    ref,
                    duration,
                    t,
                    ref_traj=traj_mode,
                    cfg_strength=args.cfg_strength,
                    resample_rounds=rounds,
                    resample_range=tuple(args.resample_range),
                    ccg=CCG(
                        w=repl_cfg,
                        window=tuple(args.ccg_window) if args.ccg_window else None,
                        lp_k=args.ccg_lp_k,
                        eta_apg=args.ccg_eta_apg,
                        rescale=args.repl_cfg_rescale,
                        kappa=args.ccg_kappa,
                    ),
                    seed=seed,
                    invert_t_start=args.invert_t_start,
                    invert_steps=args.invert_steps,
                    invert_renorm=args.invert_renorm,
                )
                if snr is not None:
                    snrs.append(snr)
            elif cond == "uncond":
                with torch.inference_mode():
                    wav, _ = model.sample(
                        duration, batch=1, steps=args.steps, cfg_strength=args.cfg_strength, seed=seed
                    )
            elif cond == "real_same":
                wav = load_wav(other_path, sr, args.duration_sec).to(device)
            elif cond == "real_diff":
                _, _, other_spk_wav = picks[(idx + 1) % len(picks)]
                wav = load_wav(other_spk_wav, sr, args.duration_sec).to(device)
            else:
                raise ValueError(f"Unknown condition: {cond}")

            sim = float(F.cosine_similarity(ref_emb, embed(wav), dim=-1))
            sims[cond].append(sim)
            if metrics is not None:
                mos = metrics.utmos(wav.reshape(1, -1), device)
                if mos is not None:
                    moss[cond].append(mos)
            if args.save_dir and cond != "real_same" and cond != "real_diff":
                out = Path(args.save_dir) / f"{spk}_{cond}.wav"
                torchaudio.save(str(out), peak_normalize(wav.reshape(1, -1).float().cpu()), sr)

        print(
            f"[{idx + 1:3d}/{len(picks)}] spk {spk:<6} " + "  ".join(f"{c}={sims[c][-1]:.3f}" for c in args.conditions)
        )

    print()
    if snrs:
        s = torch.tensor(snrs)
        print(f"inversion reconstruction SNR: {s.mean():.1f} +- {s.std():.1f} dB (plan's gate: > 25 dB)")
    floor = sims.get("uncond")
    ceil = sims.get("real_same")
    for cond in args.conditions:
        print(summarize(cond, sims[cond], None if cond == "uncond" else floor, ceil))
    for cond in args.conditions:
        if moss[cond]:
            m = torch.tensor(moss[cond])
            print(f"{cond:<12} UTMOS {m.mean():.2f} +- {m.std():.2f}")


if __name__ == "__main__":
    main()
