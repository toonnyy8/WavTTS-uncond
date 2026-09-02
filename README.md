<div align="center">
  <h1>
  WavTTS-Uncond: Unconditional Raw-Waveform Speech Generation with a Mixed-Speech Null Branch
  </h1>

  <p align="center">
    <i>A research fork of <a href="https://github.com/cwx-worst-one/WavTTS">WavTTS</a> that turns the zero-shot TTS model into an unconditional speech generator, folding speaker-inconsistent audio into the CFG null branch so guidance points away from it without ever overshooting.</i>
  </p>
</div>

## 📖 Introduction

This fork rewrites WavTTS into an **unconditional pure speech generation model** operating directly on raw 16 kHz waveforms with flow matching + DiT. All text and audio-prompt conditioning is removed; the only condition is a 2-value state embedding:

- `clean` — single, speaker-consistent speech
- `null` — the CFG negative branch

**The label comes first, the augmentation follows it.** Each sample is labelled `null` with probability `state_null_prob` (0.5), and *only* null samples may be mixed with a batch-roll partner (prob `p_mix`), in one of two equal-power forms:

- **overlap** — whole-utterance blend `√(1−λ)·x + √λ·partner` (simultaneous speakers)
- **concat** — switch to the partner at a random point with an equal-power cos/sin crossfade (temporal speaker switch)

Crossfading leaves no boundary artifact the model could cheat on — hence "no-leaky".

So the clean branch is pure single-speaker speech, while the null branch models a mixture:

```
p_null = (1 − p_mix) · p_clean + p_mix · p_mixed
```

**Mixed audio has no state of its own** — it lives inside `null`, so ordinary CFG carries the speaker-consistency signal for free:

```
v = v_clean + w · (v_clean − v_null)
```

This points away from speaker inconsistency *and* self-extinguishes: where `x` is unambiguously clean, `p_mixed(x)` vanishes faster than any mixing weight, so `∇log p_null → ∇log p_clean` and the guidance term goes to zero — for any `p_mix < 1`. Giving mixed its own state and subtracting it directly would keep the first property and lose the second: that term is a likelihood *ratio* gradient and pushes harder the further `x` gets from the mixed manifold, the mechanism behind negative-prompt oversaturation. `p_mix` only sets how deep into the clean region guidance survives.

Design documents live under [`docs/superpowers/specs/`](docs/superpowers/specs/) with the full rationale, defaults, and known limitations.

**Note:** the upstream TTS inference/eval scripts (`infer_cli.py`, `utils_infer.py`, `src/wavtts/eval/`) are kept for reference but are incompatible with this model and no longer maintained here.

## ⚙️ Installation

Managed with [uv](https://docs.astral.sh/uv/) — `uv.lock` pins the whole environment.

```bash
git clone https://github.com/toonnyy8/WavTTS
cd WavTTS

uv sync                 # creates .venv from uv.lock, incl. the dev group (pytest)
uv sync --extra eval    # add the (unmaintained) upstream eval deps
uv add <package>        # add a dependency: updates pyproject.toml + uv.lock + .venv
```

Run anything through `uv run` (e.g. `uv run pytest`), or activate `.venv` directly.

## 🏋️ Training

Prepare a raw-waveform dataset (e.g., [Emilia](https://huggingface.co/datasets/amphion/Emilia-Dataset)) with the scripts under `src/wavtts/train/datasets/`, then:

```bash
# single GPU works out of the box; run `uv run accelerate config` for multi-GPU
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run accelerate launch --mixed_precision bf16 src/wavtts/train/train.py --config-name WavTTS.yaml
```

`WavTTS.yaml` (664.5M params) is tuned for a single 24 GB GPU: bf16, 3200 frames/GPU ×6 grad-accum
(= 19200 frames/update, ~22 GiB, 1.25 s/update on an RTX 4090). With more GPUs, raise
`batch_size_per_gpu` and drop `grad_accumulation_steps` accordingly. `WavTTS_small.yaml` (22.9M) is the quick sanity-check run.

Key config entries in `src/wavtts/configs/WavTTS.yaml`:

| Key | Default | Meaning |
|---|---|---|
| `seed` | 666 | run-level reproducibility (python/torch/cuda + dataset shuffling, cudnn deterministic) |
| `model.cfm.state_null_prob` | 0.5 | prob a sample is labelled `null`; the rest are `clean` |
| `model.cfm.p_mix` | 0.5 | among `null` samples, prob of the mixing augmentation |
| `model.cfm.p_concat` | 0.5 | among mixed: concat (temporal switch) vs overlap |
| `ckpts.logger` | tensorboard | `wandb` \| `tensorboard` \| `null` |
| `ckpts.log_samples_seeds` | [0, 1, 2, 3] | fixed seeds for checkpoint sampling — same clips evolve across training |
| `ckpts.log_samples_secs` | [5, 15, 30, 60] | one clip length per seed; 60 s is past the 30 s training maximum |
| `ckpts.spk_ckpt_path` | null | ECAPA (WavLM-large) ckpt enabling `gen/spk_sim_self` |

### Length extrapolation

Training clips top out at 29.8 s, so generating longer needs help. Two mechanisms, both
trained in rather than bolted on at inference, and both leaving the architecture
untouched — no extra parameters, no extra buffers, nothing to set at generation time:

**Randomized positional encoding** ([Ruoss et al.
2023](https://arxiv.org/abs/2305.16843)) — training positions are
`randperm(L_t)[:n].sort()` instead of `arange(n)`. A short clip keeps its token order
but is told it spans a longer stretch, so it exercises rotations only long audio would
produce. Training only; inference uses contiguous positions.

At `γ = 4` the longest clip's positions reach 11916, so generating up to **119 s** uses
positions the model trained on. Vanilla RoPE at `base=10000` turns its slowest dim once
every 471 s, so nothing wraps along the way. Reaching past 119 s means raising `γ` and
retraining — length generalization comes from the positions trained on.

*Why not YaRN.* An earlier revision of this branch paired the above with YaRN
frequencies at a fixed `s`, following [arXiv:2606.23687](https://arxiv.org/abs/2606.23687).
That was a category error: YaRN is a post-training method, and every piece of it —
NTK-by-parts sparing the high-frequency dims, the `0.1·ln(s) + 1` attention temperature,
the short adaptation run — exists to preserve structure a model already learned under a
different spectrum. Training from scratch there is nothing to preserve, so a fixed
interpolated spectrum is just an oddly-shaped one picked for no reason — and for this
data a bad one: over the longest position the augmentation produces, `base=10000`
already leaves 5 of 32 dim pairs short of a single turn, and `s=4` makes that 10 of 32.
It pushes the slow tail slower (471 s → 1885 s) when speech at 100 Hz frames lives
between 0.25 and 3000 frames. The Randomized YaRN implementation and its checkpoints
remain on the `randomized-yarn-uniform` branch.

Two deviations from Ruoss et al.

`L_t` is relative. Their sequences all sit near the training cap, so a fixed `L_t`
stretches every sample about equally. Ours span 0.3–30 s, and a fixed `L_t` would
stretch a median 4.5 s clip ~13× while barely touching a 30 s one. In waveform
modelling relative position *is* physical time — pitch period, formant transitions — so
`L_t` is a multiple of each clip's own length.

`L_t` is drawn per sample rather than stepped through a curriculum:
`L_t ~ U[n, n·γ]`, fresh every forward, with `γ = 4`. A curriculum makes the stretch
factor a property of the *update* — every row in a batch stretched alike, and once the
schedule leaves `k=1` the model never sees a contiguous sequence again. The random
bound makes it a property of the *sample*, so from the first update every batch spans
contiguous through `γ`. Measured over 4096 draws at `n=2979`, the realized stretch is
uniform on `[1, 4]`: min 1.00, quartiles 1.76 / 2.50 / 3.27, max 4.00. It also removes
a discrete regime change from the middle of training — one less thing for a weight
average to straddle.

**Entropy invariance** — softmax entropy grows with the number of keys, so logits are
scaled by `max(1, log(n) / log(logn_ref_len))`. The clamp matters: the job is to sharpen
attention on sequences longer than the reference, never to flatten it on shorter ones.
Trained in, because this model's clips already span two orders of magnitude of `n`; the
model learns the relationship rather than extrapolating it at inference.

```bash
uv run python src/wavtts/infer/sample_uncond.py \
  --ckpt ckpts/.../model_last.pt --duration_sec 60
```

| Key | Default | Meaning |
|---|---|---|
| `arch.rpe_gamma` | 4.0 | per-sample stretch bound: `L_t ~ U[n, n·γ]`; `1.0` disables |
| `arch.logn_ref_len` | 500 | entropy-invariant scaling reference (5 s), clamped at 1; `null` disables |

### Framing-phase augmentation

The front end slices the waveform on a grid anchored at sample 0, and `input_embed` is a
`Linear` over the raw samples, so the model has no shift equivariance: delay a waveform by
one sample and its token decomposition changes completely. Left alone, every clip is seen
under one arbitrary alignment for the whole run, and the grid phase becomes something to
memorize.

`random_frame_offset` crops a random `0..wav_frame_hop-1` samples off the front of each
clip at load time, so it meets the grid at a different phase every epoch — `wav_frame_hop`
distinct views per clip, at no perceptual cost, since a pure delay cannot change what the
audio sounds like. The range is the hop, not the frame: framing is periodic in the hop.
Cropping rather than zero-padding, so the token count can only shrink and the sampler's
frame budget stays an upper bound; the cost is at most 9.9 ms off the head of a clip.

This is a different target from the overlapping-frames arm. Overlap-add makes a frame
boundary a crossfade rather than a splice — neighbouring tokens disagreeing on the samples
they share. The offset is about the model tying anything to absolute grid phase. They
compose; `WavTTS_clean_ola_offset.yaml` is the arm that measures the second on top of the
first, against `WavTTS_clean_ola.yaml` as the baseline.

| Key | Default | Meaning |
|---|---|---|
| `waveform.random_frame_offset` | `False` | crop a random `0..hop-1` samples off each clip at load time; training only |

**Do not compare loss curves across the two arms.** The augmentation widens the target
distribution, so `flow_loss` and `aux_mel_loss` both sit higher by construction. Judge it
on the checkpoint samples — `gen/utmos`, the mel images, and the strength of the frame-rate
line in a long-term average spectrum of the generated clips.

### Monitoring

```bash
uv run tensorboard --logdir runs
```

At every checkpoint the trainer generates one fixed-seed clip per length in
`log_samples_secs` and logs audio (`gen/audio_{sec}s_seed{k}`), log-mel images
(`gen/mel_{sec}s_seed{k}`), and quality metrics — both per length (`gen_{sec}s/*`) and
averaged (`gen/*`). Read the per-length curves: a model that extrapolates badly keeps a
healthy 5 s clip while the 60 s one collapses, and the average hides that. The four
clips cost ~45 s per checkpoint on an RTX 4090; a clip that will not fit is skipped with
a warning rather than taking the run down.

Metrics logged for each clip:

- `gen/utmos` — predicted MOS (1–5), naturalness at a glance
- `gen/spk_sim_self` — cosine similarity of the clip's two halves' speaker embeddings — the direct speaker-consistency signal this design targets (opt-in via `ckpts.spk_ckpt_path`)
- `gen/silence_ratio`, `gen/clipping_rate`, `gen/rms` — instant alarms for silent collapse, clipping, and energy drift

All metrics are fail-safe: they skip (with a warning) rather than interrupt training.

## 🎧 Sampling

```bash
uv run python src/wavtts/infer/sample_uncond.py \
  --ckpt ckpts/.../model_last.pt \
  --duration_sec 5 --num 4 \
  --steps 32 --cfg_strength 2.0 \
  --solver euler        # euler | dpmpp (DPM-Solver++(2M))
```

Useful flags: `--solver dpmpp` (multistep DPM-Solver++ adapted to the rectified-flow interpolant), `--seed N` (deterministic, does not touch the global RNG), `--device cpu`. Raise `--cfg_strength` for a stronger push away from speaker inconsistency; there is no second guidance weight to tune.

## ✅ Tests

```bash
uv run pytest tests/test_uncond_smoke.py -v
```

CPU-only smoke suite covering the mixing math (including the no-leak prefix property), CFG paths, both solvers, seed isolation, metrics, and the CLI end-to-end.

## 🙏 Acknowledgements

This fork is built on [WavTTS](https://github.com/cwx-worst-one/WavTTS) (Chen et al., 2026), which itself builds on [F5-TTS](https://github.com/SWivid/F5-TTS), [DAC](https://github.com/descriptinc/descript-audio-codec), and [JiT](https://github.com/LTH14/JiT). If you use the waveform-domain flow-matching backbone, please cite the original paper:

```bibtex
@article{chen2026wavtts,
  title={WavTTS: Towards High-Quality Zero-Shot TTS via Direct Raw Waveform Modeling},
  author={Chen, Wenxi and Jia, Dongya and Chen, Yushen and Niu, Zhikang and Liang, Yuzhe and Li, Xiquan and Yan, Ruiqi and Ma, Ziyang and Yang, Guanrou and Chen, Sanyuan and others},
  journal={arXiv preprint arXiv:2606.03455},
  year={2026}
}
```

## 📜 License

Code is released under the MIT License (inherited from upstream). Upstream pre-trained weights are CC BY-NC 4.0 due to Emilia dataset licensing.
