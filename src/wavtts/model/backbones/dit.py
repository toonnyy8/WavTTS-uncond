"""
ein notation:
b - batch
n - sequence
nw - raw wave length
d - dimension
"""
# ruff: noqa: F722 F821

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from x_transformers.x_transformers import RotaryEmbedding

from wavtts.model.modules import (
    AdaLayerNorm_Final,
    ConvPositionEmbedding,
    DiTBlock,
    TimestepEmbedding,
)
from wavtts.model.rope import (
    YaRNRotaryEmbedding,
    randomized_positions,
    rpe_max_len,
    yarn_attention_factor,
    yarn_inv_freq,
)


# speech-state conditioning: the only condition this model has
STATE_CLEAN = 0  # single consistent speaker
STATE_MIXED = 1  # speaker-inconsistent (overlap or concat augmentation)
STATE_NULL = 2  # unconditional
NUM_STATES = 3


# waveform patch embedding


class InputEmbedding(nn.Module):
    def __init__(
        self,
        wav_frame_len,
        out_dim,
        use_audio_proj: bool = False,
        audio_proj_dim: int | None = None,
        audio_proj_hidden: int | None = None,
    ):
        super().__init__()

        self.use_audio_proj = use_audio_proj

        if not use_audio_proj:
            self.proj = nn.Linear(wav_frame_len, out_dim)
        else:
            audio_proj_dim = out_dim if audio_proj_dim is None else audio_proj_dim
            audio_proj_hidden = audio_proj_dim if audio_proj_hidden is None else audio_proj_hidden
            self.x_proj = nn.Sequential(
                nn.Linear(wav_frame_len, audio_proj_hidden, bias=False),
                nn.Linear(audio_proj_hidden, audio_proj_dim),
            )
            self.fuse = nn.Linear(audio_proj_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(self, x: float["b n d"], audio_mask: bool["b n"] | None = None):
        if not self.use_audio_proj:
            h = self.proj(x)
        else:
            h = self.fuse(self.x_proj(x))

        h = self.conv_pos_embed(h, mask=audio_mask) + h
        return h


# Transformer backbone using DiT blocks


class DiT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        wav_frame_len=160,
        qk_norm=None,
        pe_attn_head=None,
        attn_backend="torch",  # "torch" | "flash_attn"
        attn_mask_enabled=False,
        long_skip_connection=False,
        checkpoint_activations=False,
        use_audio_proj: bool = False,
        audio_proj_dim: int | None = None,
        audio_proj_hidden: int | None = None,
        # length extrapolation (see wavtts/model/rope.py); defaults reproduce the
        # original vanilla-RoPE behaviour exactly
        rope_type: str = "default",  # "default" | "yarn"
        yarn_scale: float = 1.0,  # s during training; s' at inference via set_yarn_scale
        yarn_native_ctx: int = 3000,  # frames the architecture is expected to cover unaided
        yarn_alpha: float = 1.0,
        yarn_beta: float = 32.0,
        rpe: str = "off",  # "off" | "relative" | "absolute"; training only
        rpe_length_scale: float = 1.0,  # curriculum value, advanced by the trainer
        rpe_per_sample: bool = True,  # one position draw per sample (reference impl) vs per batch (paper)
        logn_ref_len: int | None = None,  # entropy-invariant attention scaling; None disables
    ):
        super().__init__()

        self.time_embed = TimestepEmbedding(dim)
        self.state_embed = nn.Embedding(NUM_STATES, dim)
        self.input_embed = InputEmbedding(
            wav_frame_len,
            dim,
            use_audio_proj=use_audio_proj,
            audio_proj_dim=audio_proj_dim,
            audio_proj_hidden=audio_proj_hidden,
        )

        if rope_type == "default":
            self.rotary_embed = RotaryEmbedding(dim_head)
        elif rope_type == "yarn":
            self.rotary_embed = YaRNRotaryEmbedding(
                dim_head, scale=yarn_scale, native_ctx=yarn_native_ctx, alpha=yarn_alpha, beta=yarn_beta
            )
        else:
            raise ValueError(f"Unknown rope_type: {rope_type}")

        self.rope_type = rope_type
        self.rope_dim_head = dim_head
        self.yarn_scale = yarn_scale
        self.yarn_native_ctx = yarn_native_ctx
        self.yarn_alpha = yarn_alpha
        self.yarn_beta = yarn_beta
        self.rpe = rpe
        self.rpe_length_scale = rpe_length_scale
        self.rpe_per_sample = rpe_per_sample

        self.dim = dim
        self.depth = depth

        self.transformer_blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    ff_mult=ff_mult,
                    dropout=dropout,
                    qk_norm=qk_norm,
                    pe_attn_head=pe_attn_head,
                    attn_backend=attn_backend,
                    attn_mask_enabled=attn_mask_enabled,
                    logn_ref_len=logn_ref_len,
                    attn_temperature=yarn_attention_factor(yarn_scale) if rope_type == "yarn" else 1.0,
                )
                for _ in range(depth)
            ]
        )
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None

        self.norm_out = AdaLayerNorm_Final(dim)  # final modulation
        self.proj_out_dim = wav_frame_len
        self.proj_out = nn.Linear(dim, wav_frame_len)
        self.proj_out_output_layer = self.proj_out

        self.checkpoint_activations = checkpoint_activations

        self.initialize_weights()

        # raw waveform tokenization.
        self.wav_frame_len = wav_frame_len

    def initialize_weights(self):
        # State (class) embedding:
        nn.init.normal_(self.state_embed.weight, std=0.02)

        # Zero-out AdaLN layers in DiT blocks:
        for block in self.transformer_blocks:
            nn.init.constant_(block.attn_norm.linear.weight, 0)
            nn.init.constant_(block.attn_norm.linear.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out_output_layer.weight, 0)
        if self.proj_out_output_layer.bias is not None:
            nn.init.constant_(self.proj_out_output_layer.bias, 0)

    def set_wav_frame_len(self, wav_frame_len: int):
        self.wav_frame_len = int(wav_frame_len)

        if self.wav_frame_len != self.proj_out_dim:
            raise ValueError(
                f"wav_frame_len ({self.wav_frame_len}) must equal proj_out_dim ({self.proj_out_dim}) "
                "for reshape wav front-end."
            )

    def set_rpe_length_scale(self, length_scale: float):
        """Advance the length curriculum. Called per update by the trainer."""
        self.rpe_length_scale = float(length_scale)

    def set_yarn_scale(self, scale: float):
        """Retune YaRN to an inference scale `s'`.

        The paper trains at `s` and infers at `s' >= s`; `s' > s` reaches past
        `s * native_ctx`. Frequencies and attention temperature both depend on the
        scale, so both are rebuilt here — no parameters change.
        """
        if self.rope_type != "yarn":
            raise ValueError(f"set_yarn_scale requires rope_type='yarn', got {self.rope_type!r}")

        self.yarn_scale = float(scale)
        inv_freq = yarn_inv_freq(
            self.rope_dim_head, 10000.0, self.yarn_scale, self.yarn_native_ctx, self.yarn_alpha, self.yarn_beta
        )
        self.rotary_embed.inv_freq = inv_freq.to(self.rotary_embed.inv_freq.device)
        self.rotary_embed.scale = self.yarn_scale
        for block in self.transformer_blocks:
            block.attn.processor.attn_temperature = yarn_attention_factor(self.yarn_scale)

    def _wav_to_tokens(
        self,
        wav: torch.Tensor,
        mask: torch.Tensor | None = None,
        lens: torch.Tensor | None = None,
    ):
        assert wav.ndim == 2, f"Expected [B, N] wav input, got {tuple(wav.shape)}"

        bsz, num_samples = wav.shape
        frame_len = self.wav_frame_len
        pad_len = (frame_len - (num_samples % frame_len)) % frame_len

        if pad_len > 0:
            wav = F.pad(wav, (0, pad_len), value=0.0)
            if mask is not None:
                mask = F.pad(mask, (0, pad_len), value=False)

        tokens = wav.view(bsz, -1, frame_len)

        token_mask = None
        if mask is not None:
            token_mask = mask.view(bsz, -1, frame_len).any(dim=-1)

        token_lens = None
        if lens is not None:
            token_lens = (lens.to(dtype=torch.long, device=wav.device) + frame_len - 1) // frame_len

        return tokens, token_mask, token_lens

    def _tokens_to_wav(self, tokens: torch.Tensor, target_num_samples: int):
        wav = tokens.reshape(tokens.shape[0], -1)
        return wav[:, :target_num_samples]

    def ckpt_wrapper(self, module):
        # https://github.com/chuanyangjin/fast-DiT/blob/main/models.py
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs

        return ckpt_forward

    def forward(
        self,
        x: float["b nw"],  # noised waveform
        state: int["b"],  # speech-state condition: STATE_CLEAN / STATE_MIXED / STATE_NULL
        time: float["b"] | float[""],  # time step
        mask: bool["b nw"] | None = None,
        cfg_infer: bool = False,  # pack positive & negative state forward
        neg_state: int["b"] | None = None,  # negative branch state for cfg_infer
        lens: int["b"] | None = None,
    ):
        if x.ndim != 2:
            raise ValueError(f"WavTTS DiT expects raw waveform x [B, N], got {x.ndim}D.")

        target_num_samples = x.shape[1]
        x, token_mask, _token_lens = self._wav_to_tokens(x, mask=mask, lens=lens)
        mask = token_mask

        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        t = self.time_embed(time)
        h = self.input_embed(x, audio_mask=mask)

        if cfg_infer:  # pack positive & negative state forward: b n d -> 2b n d
            if neg_state is None:
                neg_state = torch.full_like(state, STATE_NULL)
            h = torch.cat((h, h), dim=0)
            t = torch.cat((t, t), dim=0)
            state = torch.cat((state, neg_state), dim=0)
            mask = torch.cat((mask, mask), dim=0) if mask is not None else None

        t = t + self.state_embed(state)

        # randomized positional encoding is a training-time augmentation: the clip keeps
        # its token order but is told it spans a longer stretch, so short training audio
        # still exercises the rotations only long audio would produce
        if self.rpe != "off" and self.training:
            max_len = rpe_max_len(self.rpe, seq_len, self.rpe_length_scale, self.yarn_native_ctx)
            rope = self.rotary_embed(
                randomized_positions(h.shape[0], seq_len, max_len, h.device, per_sample=self.rpe_per_sample)
            )
        else:
            rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if self.long_skip_connection is not None:
            residual = h

        for block in self.transformer_blocks:
            if self.checkpoint_activations:
                # https://pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.checkpoint
                h = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), h, t, mask, rope, use_reentrant=False)
            else:
                h = block(h, t, mask=mask, rope=rope)

        if self.long_skip_connection is not None:
            h = self.long_skip_connection(torch.cat((h, residual), dim=-1))

        h = self.norm_out(h, t)
        output = self.proj_out(h)

        return self._tokens_to_wav(output, target_num_samples=target_num_samples)
