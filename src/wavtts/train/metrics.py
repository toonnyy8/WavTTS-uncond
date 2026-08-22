"""Lightweight generation-quality metrics for training-time monitoring.

Signal statistics (silence_ratio / clipping_rate / rms) are pure torch and always
available. Heavyweight metrics (UTMOS, speaker self-similarity) are lazy-loaded and
return None on any failure — metrics must never crash a training run.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from wavtts.model.utils import peak_normalize


def silence_ratio(wav: torch.Tensor, frame_len: int = 400, threshold_rel: float = 1e-2) -> float:
    """Fraction of frames whose RMS falls below threshold_rel times the clip's own RMS.

    Relative rather than absolute: the loudness the model trains at is a config knob
    (`waveform.target_rms`), and an absolute threshold silently changes meaning the
    moment it moves. 1e-2 reproduces the old absolute 1e-3 at the old target_rms of 0.1,
    so the logged curve keeps its scale across the change.

    wav: [n] or [1, n].
    """
    wav = wav.reshape(-1)
    n = (wav.numel() // frame_len) * frame_len
    if n == 0:
        return 1.0
    level = wav.pow(2).mean().sqrt()
    if level <= 0:
        return 1.0
    frames = wav[:n].reshape(-1, frame_len)
    frame_rms = frames.pow(2).mean(dim=-1).sqrt()
    return (frame_rms < threshold_rel * level).float().mean().item()


def clipping_rate(wav: torch.Tensor, crest: float = 9.9) -> float:
    """Fraction of samples whose |x| exceeds `crest` times the clip's own RMS.

    Relative for the same reason as silence_ratio, and 9.9 reproduces the old absolute
    0.99 at the old target_rms of 0.1. Speech sits near a crest factor of 7, so a healthy
    clip still reads close to zero while a saturating model stands out.
    """
    wav = wav.reshape(-1)
    level = wav.pow(2).mean().sqrt()
    if level <= 0:
        return 0.0
    return (wav.abs() > crest * level).float().mean().item()


def rms(wav: torch.Tensor) -> float:
    return wav.reshape(-1).pow(2).mean().sqrt().item()


def mel_figure(wav: torch.Tensor, sample_rate: int = 16000):
    """Log-mel spectrogram matplotlib figure, for SummaryWriter.add_figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torchaudio

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=1024, hop_length=256, n_mels=80
    )(wav.reshape(1, -1).float().cpu())
    mel_db = mel.clamp(min=1e-5).log10().squeeze(0)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.imshow(mel_db.numpy(), origin="lower", aspect="auto", interpolation="none")
    ax.set_xlabel("frame")
    ax.set_ylabel("mel bin")
    fig.tight_layout()
    return fig


class GenMetrics:
    """Lazy holders for heavyweight metric models."""

    def __init__(self, sample_rate: int = 16000, spk_ckpt_path: str | None = None):
        self.sample_rate = sample_rate
        self.spk_ckpt_path = spk_ckpt_path
        self._utmos = None
        self._utmos_failed = False
        self._spk = None
        self._spk_failed = False

    def utmos(self, wav: torch.Tensor, device) -> float | None:
        """Predicted MOS (1-5) via UTMOS. None if the hub model is unavailable.

        Peak-normalized first: UTMOS was trained on conventional +-1 audio, and this
        model's output sits at whatever `target_rms` says.
        """
        if self._utmos_failed:
            return None
        try:
            if self._utmos is None:
                self._utmos = (
                    torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(device).eval()
                )
            wav = peak_normalize(wav.float())
            with torch.inference_mode():
                score = self._utmos(wav.reshape(1, -1).float().to(device), self.sample_rate)
            return float(score.item())
        except Exception as e:  # noqa: BLE001 — metrics must never crash training
            print(f"[metrics] UTMOS unavailable, skipping: {e}")
            self._utmos_failed = True
            return None

    def spk_sim_self(self, wav: torch.Tensor, device) -> float | None:
        """Intra-utterance speaker consistency: cosine similarity between speaker
        embeddings of the clip's two halves. Requires spk_ckpt_path (WavLM-large
        ECAPA checkpoint, same one the eval scripts use); None when unset/unavailable."""
        if self.spk_ckpt_path is None or self._spk_failed:
            return None
        try:
            if self._spk is None:
                from wavtts.eval.ecapa_tdnn import ECAPA_TDNN_SMALL

                model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large", config_path=None)
                state = torch.load(self.spk_ckpt_path, map_location="cpu", weights_only=True)
                model.load_state_dict(state["model"], strict=False)
                self._spk = model.to(device).eval()
            # same reason as UTMOS: WavLM/ECAPA expect the conventional +-1 domain
            wav = peak_normalize(wav.reshape(-1).float()).to(device)
            half = wav.numel() // 2
            with torch.inference_mode():
                emb = self._spk(torch.stack([wav[:half], wav[half : 2 * half]]))
            return float(F.cosine_similarity(emb[0], emb[1], dim=-1).item())
        except Exception as e:  # noqa: BLE001 — metrics must never crash training
            print(f"[metrics] speaker self-sim unavailable, skipping: {e}")
            self._spk_failed = True
            return None
