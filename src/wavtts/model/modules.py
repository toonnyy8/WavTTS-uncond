"""
ein notation:
b - batch
n - sequence
nw - raw wave length
d - dimension
"""
# ruff: noqa: F722 F821

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import torchaudio
from torch import nn
from x_transformers.x_transformers import apply_rotary_pos_emb


# sinusoidal position embedding


class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# convolutional position embedding


class ConvPositionEmbedding(nn.Module):
    def __init__(self, dim, kernel_size=31, groups=16):
        super().__init__()
        assert kernel_size % 2 != 0
        self.conv1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
        )
        self.layer_need_mask_idx = [i for i, layer in enumerate(self.conv1d) if isinstance(layer, nn.Conv1d)]

    def forward(self, x: float["b n d"], mask: bool["b n"] | None = None):
        if mask is not None:
            mask = mask.unsqueeze(1)  # [B 1 N]
        x = x.permute(0, 2, 1)  # [B D N]

        if mask is not None:
            x = x.masked_fill(~mask, 0.0)
        for i, block in enumerate(self.conv1d):
            x = block(x)
            if mask is not None and i in self.layer_need_mask_idx:
                x = x.masked_fill(~mask, 0.0)

        x = x.permute(0, 2, 1)  # [B N D]

        return x


# RMSNorm


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.native_rms_norm = float(torch.__version__[:3]) >= 2.4

    def forward(self, x):
        if self.native_rms_norm:
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                x = x.to(self.weight.dtype)
            x = F.rms_norm(x, normalized_shape=(x.shape[-1],), weight=self.weight, eps=self.eps)
        else:
            variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                x = x.to(self.weight.dtype)
            x = x * self.weight

        return x


# AdaLayerNorm
# return with modulated x for attn input, and params for later mlp modulation


class AdaLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 6)

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb=None):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(emb, 6, dim=1)

        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


# AdaLayerNorm for final layer
# return only with modulated x for attn input, cuz no more mlp modulation


class AdaLayerNorm_Final(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        emb = self.linear(self.silu(emb))
        scale, shift = torch.chunk(emb, 2, dim=1)

        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


# FeedForward


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, dropout=0.0, approximate: str = "none"):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        activation = nn.GELU(approximate=approximate)
        project_in = nn.Sequential(nn.Linear(dim, inner_dim), activation)
        self.ff = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, x):
        return self.ff(x)


# Attention
# modified from diffusers/src/diffusers/models/attention_processor.py


class Attention(nn.Module):
    def __init__(
        self,
        processor: AttnProcessor,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        qk_norm: str | None = None,
    ):
        super().__init__()

        self.processor = processor

        self.dim = dim
        self.heads = heads
        self.inner_dim = dim_head * heads
        self.dropout = dropout

        self.to_q = nn.Linear(dim, self.inner_dim)
        self.to_k = nn.Linear(dim, self.inner_dim)
        self.to_v = nn.Linear(dim, self.inner_dim)

        if qk_norm is None:
            self.q_norm = None
            self.k_norm = None
        elif qk_norm == "rms_norm":
            self.q_norm = RMSNorm(dim_head, eps=1e-6)
            self.k_norm = RMSNorm(dim_head, eps=1e-6)
        else:
            raise ValueError(f"Unimplemented qk_norm: {qk_norm}")

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(self.inner_dim, dim))
        self.to_out.append(nn.Dropout(dropout))

    def forward(
        self,
        x: float["b n d"],
        mask: bool["b n"] | None = None,
        rope=None,  # rotary position embedding
    ) -> torch.Tensor:
        return self.processor(self, x, mask=mask, rope=rope)


# Attention processor


class AttnProcessor:
    def __init__(
        self,
        attn_mask_enabled: bool = True,
        logn_ref_len: int | None = None,  # entropy-invariant scaling reference; None disables
    ):
        self.attn_mask_enabled = attn_mask_enabled
        self.logn_ref_len = logn_ref_len

    def _logit_scale(self, seq_len: int) -> float:
        """Entropy-invariant attention temperature, folded into the softmax scale.

        Entropy invariance (Su, 2021): softmax entropy grows with the number of keys, so
        `log_m(n)` holds it steady as n moves. Every query attends over the same `n` here,
        so one scalar covers the whole batch.

        Clamped at 1: the job is to sharpen attention on sequences longer than the
        reference, never to flatten it on shorter ones, which is how the reference
        implementations (Qwen) apply it too. Without the clamp a 0.4 s clip runs at 0.46,
        pushing a 40-key softmax toward uniform for no reason.
        """
        if not self.logn_ref_len:
            return 1.0
        return max(1.0, math.log(max(seq_len, 2)) / math.log(self.logn_ref_len))

    def __call__(
        self,
        attn: Attention,
        x: float["b n d"],  # noised input x
        mask: bool["b n"] | None = None,
        rope=None,  # rotary position embedding
    ) -> torch.FloatTensor:
        batch_size = x.shape[0]

        # `sample` projections
        query = attn.to_q(x)
        key = attn.to_k(x)
        value = attn.to_v(x)

        # attention
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # qk norm
        if attn.q_norm is not None:
            query = attn.q_norm(query)
        if attn.k_norm is not None:
            key = attn.k_norm(key)

        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)
            query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
            key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)

        # attention temperature, folded into the softmax scale rather than the query:
        # scaling the query would allocate another [b, h, n, d] tensor in every block
        softmax_scale = self._logit_scale(query.shape[-2]) / math.sqrt(head_dim)

        # mask e.g. an inference batch with different target durations: drop the padding
        if self.attn_mask_enabled and mask is not None:
            attn_mask = mask[:, None, None, :].expand(batch_size, attn.heads, query.shape[-2], key.shape[-2])
        else:
            attn_mask = None

        x = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False, scale=softmax_scale
        )
        x = x.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        x = x.to(query.dtype)

        # linear proj
        x = attn.to_out[0](x)
        # dropout
        x = attn.to_out[1](x)

        if mask is not None:
            mask = mask.unsqueeze(-1)
            x = x.masked_fill(~mask, 0.0)

        return x


# DiT Block


class DiTBlock(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        dim_head,
        ff_mult=4,
        dropout=0.1,
        qk_norm=None,
        attn_mask_enabled=True,
        logn_ref_len=None,
    ):
        super().__init__()

        self.attn_norm = AdaLayerNorm(dim)
        self.attn = Attention(
            processor=AttnProcessor(attn_mask_enabled=attn_mask_enabled, logn_ref_len=logn_ref_len),
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            qk_norm=qk_norm,
        )

        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")

    def forward(self, x, t, mask=None, rope=None):  # x: noised input, t: time embedding
        # pre-norm & modulation for attention input
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)

        # attention
        attn_output = self.attn(x=norm, mask=mask, rope=rope)

        # process attention output for input x
        x = x + gate_msa.unsqueeze(1) * attn_output

        norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm)
        x = x + gate_mlp.unsqueeze(1) * ff_output

        return x

# time step conditioning embedding


class TimestepEmbedding(nn.Module):
    def __init__(self, dim, freq_embed_dim=256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, timestep: float["b"]):
        time_hidden = self.time_embed(timestep)
        time_hidden = time_hidden.to(timestep.dtype)
        time = self.time_mlp(time_hidden)  # b d
        return time

# auxiliary loss for mel spectrogram
class MelSpectrogramLoss(nn.Module):
    """
    General PyTorch implementation of the Multi-Scale Mel Spectrogram Loss from DAC.
    Reference: https://github.com/descriptinc/lyrebird-audiotools
    """
    def __init__(
        self,
        sample_rate: int = 24000,
        n_mels: List[int] = [5, 10, 20, 40, 80, 160, 320],
        window_lengths: List[int] = [32, 64, 128, 256, 512, 1024, 2048],
        mel_fmin: List[float] = [0, 0, 0, 0, 0, 0, 0],
        mel_fmax: List[Optional[float]] = [None, None, None, None, None, None, None],
        pow: float = 1.0,
        clamp_eps: float = 1e-5,
        weight: float = 1.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.weight = weight
        self.pow = pow
        self.clamp_eps = clamp_eps

        
        assert len(n_mels) == len(window_lengths), "n_mels and window_lengths must have the same length"
        
        self.mel_transforms = nn.ModuleList()
        
        for i, (nm, wl) in enumerate(zip(n_mels, window_lengths)):
            fmin = mel_fmin[i] if i < len(mel_fmin) else 0.0
            fmax = mel_fmax[i] if i < len(mel_fmax) else None
            
            # DAC default logic: hop_length = window_length // 4
            hop_length = wl // 4
            
            transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=wl,        # using window_length as n_fft usually
                win_length=wl,
                hop_length=hop_length,
                n_mels=nm,
                f_min=fmin,
                f_max=fmax,
                power=1.0,       # computing Magnitude Mel (power=1.0) initially
                normalized=False,
                center=True,
                pad_mode="reflect"
            )
            self.mel_transforms.append(transform)

    def forward(
        self,
        x_pred: torch.Tensor,
        x_true: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
        frame_lengths: torch.Tensor | None = None,
    ):
        """
        Args:
            x_pred: [B, T] or [B, 1, T] Estimated Waveform
            x_true: [B, T] or [B, 1, T] Ground Truth Waveform
        Returns:
            Weighted scalar loss
        """
        # Ensure correct shape [B, T] for torchaudio, or [B, 1, T] is also fine but usually squeeze
        if x_pred.ndim == 3 and x_pred.shape[1] == 1:
            x_pred = x_pred.squeeze(1)
        if x_true.ndim == 3 and x_true.shape[1] == 1:
            x_true = x_true.squeeze(1)
            
        total_loss = x_pred.new_tensor(0.0)

        use_mask = frame_mask is not None
        if use_mask:
            span_starts, span_ends = self._get_span_bounds_from_mask(frame_mask, frame_lengths)
            span_starts = span_starts.to(device=x_pred.device)
            span_ends = span_ends.to(device=x_pred.device)
        
        for mel_transform in self.mel_transforms:
            x_mels = mel_transform(x_pred)
            y_mels = mel_transform(x_true)

            mel_mask = None
            if use_mask:
                hop = int(mel_transform.hop_length)
                mel_t = x_mels.shape[-1]
                mel_mask = torch.zeros(
                    (x_mels.shape[0], mel_t),
                    device=x_mels.device,
                    dtype=torch.bool,
                )
                for i in range(x_mels.shape[0]):
                    s = int(span_starts[i].item())
                    e = int(span_ends[i].item())
                    if e <= s:
                        continue
                    ms = s // hop
                    me = (e + hop - 1) // hop
                    ms = max(0, min(ms, mel_t))
                    me = max(0, min(me, mel_t))
                    if me > ms:
                        mel_mask[i, ms:me] = True

                if not mel_mask.any():
                    continue

                mel_mask = mel_mask.unsqueeze(1)
            
            # Log Magnitude Loss
            # Formula: L1( log10(x^pow + eps), log10(y^pow + eps) )
            x_log = x_mels.clamp(min=self.clamp_eps).pow(self.pow).log10()
            y_log = y_mels.clamp(min=self.clamp_eps).pow(self.pow).log10()

            diff_full = (x_log - y_log).abs()

            if mel_mask is None:
                diff = diff_full.mean(dim=(1, 2))
            else:
                diff = []
                for i in range(diff_full.shape[0]):
                    m = mel_mask[i]
                    if m.any():
                        diff.append(diff_full[i].masked_select(m).mean())
                    else:
                        diff.append(diff_full.new_tensor(0.0))
                diff = torch.stack(diff, dim=0)

            total_loss += diff.mean()

        return total_loss * self.weight

    @staticmethod
    def _get_span_bounds_from_mask(mask: torch.Tensor, lengths: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert boolean frame mask [B, T] into per-sample [start, end) bounds.

        This helper assumes one contiguous masked span per sample (matching current
        random-span masking strategy). If a sample has no masked frame, start=end=0.
        """
        if mask.dtype != torch.bool:
            mask = mask.bool()

        bsz, t = mask.shape
        device = mask.device

        if lengths is None:
            lengths = torch.full((bsz,), t, device=device, dtype=torch.long)
        else:
            lengths = lengths.to(device=device, dtype=torch.long)

        starts = torch.zeros((bsz,), device=device, dtype=torch.long)
        ends = torch.zeros((bsz,), device=device, dtype=torch.long)

        for i in range(bsz):
            li = int(lengths[i].item())
            row = mask[i, :li]
            idx = row.nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue
            starts[i] = idx[0]
            ends[i] = idx[-1] + 1

        return starts, ends

    @staticmethod
    def _aligned_random_span_mask(
        lengths: torch.Tensor,
        frac_lengths: torch.Tensor,
        align_to: int,
        max_length: int | None = None,
    ) -> torch.Tensor:
        """Sample one contiguous mask span per sample with boundary alignment.

        Args:
            lengths: [B] valid sequence lengths in model frames.
            frac_lengths: [B] desired masked length fraction.
            align_to: span start/end alignment in model frames.
            max_length: optional mask width. If None, use lengths.max().
        Returns:
            bool mask with shape [B, max_length].
        """
        assert align_to >= 1
        device = lengths.device
        lengths = lengths.to(dtype=torch.long)
        frac_lengths = frac_lengths.to(device=device, dtype=torch.float32)

        max_len = int(lengths.max().item()) if max_length is None else int(max_length)
        mask = torch.zeros((lengths.shape[0], max_len), device=device, dtype=torch.bool)

        for i in range(lengths.shape[0]):
            li = int(lengths[i].item())
            if li <= 0:
                continue

            target = int((frac_lengths[i].item() * li))
            target = max(1, min(target, li))

            span_len = ((target + align_to - 1) // align_to) * align_to
            span_len = min(span_len, li)

            max_start = li - span_len
            if max_start <= 0:
                start = 0
            else:
                candidate = torch.randint(0, max_start + 1, (1,), device=device).item()
                start = (candidate // align_to) * align_to
                start = min(start, max_start)

            end = start + span_len
            mask[i, start:end] = True

        return mask
