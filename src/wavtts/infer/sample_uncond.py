"""Minimal CLI for unconditional speech sampling."""

import argparse
import os
from importlib.resources import files

import torch
import torchaudio
from hydra.utils import get_class
from omegaconf import OmegaConf

from wavtts.model import CFM
from wavtts.model.utils import peak_normalize


def load_model(ckpt_path: str, cfg, device: str) -> CFM:
    model_cls = get_class(f"wavtts.model.{cfg.model.backbone}")
    model = CFM(
        transformer=model_cls(
            **OmegaConf.to_container(cfg.model.arch, resolve=True),
            wav_frame_len=cfg.model.waveform.wav_frame_len,
        ),
        waveform_kwargs=OmegaConf.to_container(cfg.model.waveform, resolve=True),
        **OmegaConf.to_container(cfg.model.cfm, resolve=True),
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "ema_model_state_dict" in ckpt:
        state = {
            k.replace("ema_model.", ""): v
            for k, v in ckpt["ema_model_state_dict"].items()
            if k not in ("initted", "update", "step")
        }
    elif "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="WavTTS unconditional sampling")
    parser.add_argument("--ckpt", required=True, help="checkpoint .pt path")
    parser.add_argument("--config", default=None, help="training yaml (default: packaged WavTTS.yaml)")
    parser.add_argument("--duration_sec", type=float, default=5.0)
    parser.add_argument("--num", type=int, default=4, help="number of samples to generate")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--cfg_strength", type=float, default=2.0)
    parser.add_argument("--solver", choices=["euler", "dpmpp"], default="euler")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out_dir", default="samples_uncond")
    parser.add_argument("--device", default=None, help="cpu | cuda | cuda:N (default: auto)")
    args = parser.parse_args(argv)

    config_path = args.config or str(files("wavtts").joinpath("configs/WavTTS.yaml"))
    cfg = OmegaConf.load(config_path)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.ckpt, cfg, device)

    sr = cfg.model.cfm.sample_rate
    duration = int(args.duration_sec * sr)
    os.makedirs(args.out_dir, exist_ok=True)

    with torch.inference_mode():
        wavs, _ = model.sample(
            duration,
            batch=args.num,
            steps=args.steps,
            cfg_strength=args.cfg_strength,
            seed=args.seed,
            solver=args.solver,
        )

    for i in range(args.num):
        out_path = os.path.join(args.out_dir, f"sample_{i}.wav")
        # the model generates at target_rms, outside +-1; a file has to come back inside
        wav = peak_normalize(wavs[i : i + 1].to(torch.float32).cpu())
        torchaudio.save(out_path, wav, sr)
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
