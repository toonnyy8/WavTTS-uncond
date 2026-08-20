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

Training clips top out at 29.8 s, so generating longer needs help. Two mechanisms,
both trained in rather than bolted on at inference, both off by `rope_type: default`:

**Randomized YaRN** ([arXiv:2606.23687](https://arxiv.org/abs/2606.23687)) — YaRN
frequencies at a fixed scale `s` during training, randomized positional encoding
(`randperm(L_t)[:n].sort()` instead of `arange(n)`, training only), and a length
curriculum that grows `L_t` over the run. A short clip keeps its token order but is
told it spans a longer stretch, so it exercises rotations only long audio would
produce. Inference runs plain YaRN at `s'`, and `s' > s` reaches past `s · native_ctx`:
`s=2, s'=4` covers 120 s.

Unlike the paper's text corpora, our clips span 0.3–30 s, so a fixed `L_t` would
stretch a median 4.5 s clip ~13× while barely touching a 30 s one. In waveform
modelling relative position *is* physical time — pitch period, formant transitions —
so `rpe: relative` sets `L_t = k · clip length` for a constant stretch instead.
`rpe: absolute` restores the paper's literal behaviour.

**Entropy invariance** — softmax entropy grows with the number of keys, so logits are
scaled by `max(1, log(n) / log(logn_ref_len))`. The clamp matters: the job is to sharpen
attention on sequences longer than the reference, never to flatten it on shorter ones.
Trained in, because this model's clips already span two orders of magnitude of `n`; the
model learns the relationship rather than extrapolating it at inference.

```bash
uv run python src/wavtts/infer/sample_uncond.py \
  --ckpt ckpts/.../model_last.pt --duration_sec 120 --yarn_scale 4
```

| Key | Default | Meaning |
|---|---|---|
| `arch.rope_type` | yarn | `default` \| `yarn` — `default` disables everything below |
| `arch.yarn_scale` | 2.0 | `s` during training |
| `arch.yarn_native_ctx` | 3000 | frames (30 s) the architecture should cover unaided |
| `arch.rpe` | relative | `off` \| `relative` (`L_t = k ·` clip length) \| `absolute` (`k · native_ctx`) |
| `arch.rpe_per_sample` | True | independent position draw per sample (reference impl) vs one per batch (paper) |
| `arch.logn_ref_len` | 500 | entropy-invariant scaling reference (5 s), clamped at 1; `null` disables |
| `optim.rpe_curriculum` | `[[0,1.0],[20000,1.25],[40000,1.5],[60000,2.0]]` | `[update, k]` milestones |

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
