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

from wavtts.model.modules import (
    AdaLayerNorm_Final,
    ConvPositionEmbedding,
    DiTBlock,
    TimestepEmbedding,
)


# speech-state conditioning: the only condition this model has
STATE_CLEAN = 0  # speech
STATE_NULL = 1  # unconditional, the CFG negative branch
NUM_STATES = 2


# waveform patch embedding


class InputEmbedding(nn.Module):
    def __init__(
        self,
        patch_len,
        out_dim,
        use_audio_proj: bool = False,
        audio_proj_dim: int | None = None,
        audio_proj_hidden: int | None = None,
    ):
        super().__init__()

        self.use_audio_proj = use_audio_proj

        if not use_audio_proj:
            self.proj = nn.Linear(patch_len, out_dim)
        else:
            audio_proj_dim = out_dim if audio_proj_dim is None else audio_proj_dim
            audio_proj_hidden = audio_proj_dim if audio_proj_hidden is None else audio_proj_hidden
            self.x_proj = nn.Sequential(
                nn.Linear(patch_len, audio_proj_hidden, bias=False),
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
        wav_frame_len=160,  # hop between patches; also the frame rate the dataset batches on
        patch_overlap=0,  # samples each patch reaches into either neighbour; 0 = disjoint patches
        qk_norm=None,
        attn_mask_enabled=False,
        long_skip_connection=False,
        checkpoint_activations=False,
        use_audio_proj: bool = False,
        audio_proj_dim: int | None = None,
        audio_proj_hidden: int | None = None,
        logn_ref_len: int | None = None,  # entropy-invariant attention scaling; None disables
    ):
        super().__init__()

        if patch_overlap < 0 or 2 * patch_overlap > wav_frame_len:
            # beyond half a hop a sample would land in more than two patches at once, and
            # the analysis window would carry more of its neighbours than of itself
            raise ValueError(f"patch_overlap must be in [0, {wav_frame_len // 2}], got {patch_overlap}")
        self.patch_overlap = int(patch_overlap)
        self.patch_len = wav_frame_len + 2 * self.patch_overlap
        if self.patch_overlap:
            # not persistent: derived from patch_len, so it never belongs in a checkpoint
            self.register_buffer(
                "synthesis_window", torch.hann_window(self.patch_len, periodic=True), persistent=False
            )

        self.time_embed = TimestepEmbedding(dim)
        self.state_embed = nn.Embedding(NUM_STATES, dim)
        self.input_embed = InputEmbedding(
            self.patch_len,
            dim,
            use_audio_proj=use_audio_proj,
            audio_proj_dim=audio_proj_dim,
            audio_proj_hidden=audio_proj_hidden,
        )

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
                    attn_mask_enabled=attn_mask_enabled,
                    logn_ref_len=logn_ref_len,
                )
                for _ in range(depth)
            ]
        )
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None

        self.norm_out = AdaLayerNorm_Final(dim)  # final modulation
        self.proj_out_dim = self.patch_len
        self.proj_out = nn.Linear(dim, self.patch_len)
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

        if self.wav_frame_len + 2 * self.patch_overlap != self.proj_out_dim:
            raise ValueError(
                f"wav_frame_len ({self.wav_frame_len}) + 2*patch_overlap ({self.patch_overlap}) must equal "
                f"proj_out_dim ({self.proj_out_dim}) for the wav front-end."
            )

    def _wav_to_tokens(
        self,
        wav: torch.Tensor,
        mask: torch.Tensor | None = None,
        lens: torch.Tensor | None = None,
    ):
        """One token per `wav_frame_len` hop, each reading `patch_overlap` into its neighbours."""
        assert wav.ndim == 2, f"Expected [B, N] wav input, got {tuple(wav.shape)}"

        bsz, num_samples = wav.shape
        hop = self.wav_frame_len
        pad_len = (hop - (num_samples % hop)) % hop

        if pad_len > 0:
            wav = F.pad(wav, (0, pad_len), value=0.0)
            if mask is not None:
                mask = F.pad(mask, (0, pad_len), value=False)

        # mask and lengths stay hop-aligned whatever the overlap, so the token count and
        # what counts as padding are the same as with disjoint patches
        token_mask = mask.view(bsz, -1, hop).any(dim=-1) if mask is not None else None
        token_lens = (lens.to(dtype=torch.long, device=wav.device) + hop - 1) // hop if lens is not None else None

        o = self.patch_overlap
        if o == 0:
            tokens = wav.view(bsz, -1, hop)
        else:
            # padding both ends by `o` centres patch i on its own hop, so the first and
            # last patches reach into silence rather than being shifted inward
            tokens = F.pad(wav, (o, o), value=0.0).unfold(1, self.patch_len, hop)

        return tokens, token_mask, token_lens

    def _tokens_to_wav(self, tokens: torch.Tensor, target_num_samples: int):
        if self.patch_overlap == 0:
            return tokens.reshape(tokens.shape[0], -1)[:, :target_num_samples]

        bsz, num_tokens, patch_len = tokens.shape
        hop, o = self.wav_frame_len, self.patch_overlap
        padded_len = (num_tokens - 1) * hop + patch_len  # == num_tokens * hop + 2 * o

        def fold(x):
            return F.fold(x, output_size=(1, padded_len), kernel_size=(1, patch_len), stride=(1, hop))

        # Weighted overlap-add. Folding the windowed patches sums them; folding the window
        # alone gives the weight each output sample actually received, so the ratio is an
        # exact weighted mean. That is what handles the edges: the outermost `o` samples
        # are reached by only one patch, at the taper of its window, and a plain COLA sum
        # would leave them attenuated. Dividing by the true envelope restores full
        # amplitude there instead of fading the clip in and out.
        window = self.synthesis_window.to(tokens.dtype)
        numer = fold((tokens * window).transpose(1, 2)).reshape(bsz, padded_len)
        envelope = fold(self.synthesis_window[None, :, None].expand(1, patch_len, num_tokens))
        envelope = envelope.reshape(1, padded_len).clamp(min=1e-8).to(tokens.dtype)

        return (numer / envelope)[:, o : o + target_num_samples]

    def ckpt_wrapper(self, module):
        # https://github.com/chuanyangjin/fast-DiT/blob/main/models.py
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs

        return ckpt_forward

    def forward(
        self,
        x: float["b nw"],  # noised waveform
        state: int["b"],  # speech-state condition: STATE_CLEAN / STATE_NULL
        time: float["b"] | float[""],  # time step
        mask: bool["b nw"] | None = None,
        cfg_infer: bool = False,  # pack clean & null state forward
        lens: int["b"] | None = None,
    ):
        if x.ndim != 2:
            raise ValueError(f"WavTTS DiT expects raw waveform x [B, N], got {x.ndim}D.")

        target_num_samples = x.shape[1]
        x, token_mask, _token_lens = self._wav_to_tokens(x, mask=mask, lens=lens)
        mask = token_mask

        batch = x.shape[0]
        if time.ndim == 0:
            time = time.repeat(batch)

        t = self.time_embed(time)
        h = self.input_embed(x, audio_mask=mask)

        if cfg_infer:  # pack positive & negative state forward: b n d -> 2b n d
            neg_state = torch.full_like(state, STATE_NULL)
            h = torch.cat((h, h), dim=0)
            t = torch.cat((t, t), dim=0)
            state = torch.cat((state, neg_state), dim=0)
            mask = torch.cat((mask, mask), dim=0) if mask is not None else None

        t = t + self.state_embed(state)

        if self.long_skip_connection is not None:
            residual = h

        # NoPE: there is no positional encoding here at all. Long-range position comes
        # from the bidirectional causal masks inside the attention (a query can tell
        # where it sits from how much it can see), local position from the input
        # embedding's ConvPositionEmbedding.
        for block in self.transformer_blocks:
            if self.checkpoint_activations:
                # https://pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.checkpoint
                h = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), h, t, mask, use_reentrant=False)
            else:
                h = block(h, t, mask=mask)

        if self.long_skip_connection is not None:
            h = self.long_skip_connection(torch.cat((h, residual), dim=-1))

        h = self.norm_out(h, t)
        output = self.proj_out(h)

        return self._tokens_to_wav(output, target_num_samples=target_num_samples)
