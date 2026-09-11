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

## 🥊 DDO finetuning

[Direct Discriminative Optimization](https://arxiv.org/abs/2503.01103) (Zheng et al., ICML
2025 oral) is a short finetuning stage that trades a little likelihood for sample quality.
Maximum-likelihood training minimizes the *forward* KL, which barely penalizes putting mass
where the data has none — at finite capacity the model covers modes rather than choosing
them, and here that is audible as mumbling, speaker drift and non-speech hum. DDO freezes a
copy of the model as a reference `p_ref`, reads the pair `(p_θ, p_ref)` as a GAN
discriminator `d_θ = σ(log p_θ/p_ref)`, and optimizes the discriminative objective. No extra
network, no alternating training, no architecture change: the loss only needs the two
models' flow losses on the same `(x_t, t, ε)`.

**This study runs no CFG on either side.** The baseline is the *clean arm pretrained
checkpoint, sampled without guidance*, and the comparison is the *same arm after DDO, also
sampled without guidance*, at the same NFE. That is the point of the method rather than a
simplification: Theorem 3.3 makes DDO's optimum `p_θ* ∝ p_ref^(1−1/β) · p_data^(1/β)`, which
is guidance-shaped but lives in the weights instead of costing a second forward at every
sampling step. **DDO is meant to replace guidance, not to stack on top of it.**

### One round, three steps

```bash
# 1. weights-only init, so the round starts at theta == p_ref with a fresh optimizer
#    (rounds deliberately do not carry optimizer state across)
uv run python scripts/make_pretrained_init.py \
  ckpts/WavTTS_Uncond_Large_RPE_Clean_LibriTTS_460/model_last.pt \
  ckpts/WavTTS_Uncond_Large_RPE_Clean_DDO_R1_LibriTTS_460 \
  src/wavtts/configs/WavTTS_ddo_r1.yaml

# 2. the offline fake pool: ~50 h sampled from the SAME weights p_ref will be
#    (the checkpoint's EMA half), guidance-free, 32-bit float wavs, durations drawn
#    from the real corpus' own duration.json
uv run python scripts/gen_fake_pool.py \
  --ckpt ckpts/WavTTS_Uncond_Large_RPE_Clean_LibriTTS_460/model_last.pt \
  --config src/wavtts/configs/WavTTS_clean.yaml \
  --ref_durations data/LibriTTS_460/duration.json \
  --out data/LibriTTS_460_fake_r1 \
  --hours 50 --steps 32 --cfg_strength 0.0 --solver euler \
  --max_batch_frames 19200 --seed 1234 --device cuda:0 --shard 0/4
#    ... and the same command with --shard 1/4 .. 3/4 on cuda:1..3: the plan is a pure
#    function of --seed, so the shards write disjoint slices of one pool and whichever
#    finishes last writes raw/ and duration.json. Rows of different lengths share a batch
#    under a mask (exactly what training does; each row comes out as it would alone) and
#    the backbone runs under bf16 autocast (--autocast none for the fp32 path). Measured:
#    50 h in 40 min on four 4090s (round 2), against 86 min for the round-1 pool's fp32
#    run -- bf16 is the 2x; the packing saves launches, not FLOPs

# 3. the round itself (~2.4 h on 4x RTX 4090 for 9000 updates)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run accelerate launch --mixed_precision bf16 --num_processes 4 \
    src/wavtts/train/train.py --config-name WavTTS_ddo_r1.yaml
```

The `ddo:` block in `src/wavtts/configs/WavTTS_ddo_r1.yaml` is what switches the objective
on; every other config omits it and behaves exactly as before.

| Key | Default | Meaning |
|---|---|---|
| `ddo.ref_ckpt` | — | the frozen `p_ref`; its **EMA** weights are read, and they must be the weights the pool was generated from |
| `ddo.fake_dataset` | — | `data/<name>` written by `gen_fake_pool.py`, in the real corpus' own directory format |
| `ddo.alpha` | 1.0 | weight on the fake term; the paper sweeps `[0.5, 6.0]`. The loss is divided by `max(α, 1)` so an α sweep is not also an LR sweep |
| `ddo.beta` | 1.0 | scale on the log-ratio. **Calibrate it — see below.** Not a stability knob: Theorem 3.3's `β < 1` extrapolates past `p_data` along `p_data/p_ref`, which is where the quality comes from and why a round must be short. The theorem's β is *this* β divided by the row's element count (Δ is a per-element mean), so a calibrated value near 300 is a theorem-β of ~0.004 on a 5 s clip — the paper's regime |
| `ddo.delta_normalize` | mean | `mean` \| `sum`. `mean` because clips span 0.3–30 s and one global β cannot suit both ends of a 100× dimension range — otherwise length becomes the one feature the discriminator needs |
| `ddo.anchor_weight` | 1.0 | weight on the MLE anchor kept on `null` rows. Never reached in this arm (`state_null_prob` is 0); it exists for a future CFG-arm port |
| `ddo.real_mel_weight` | 0.0 | the auxiliary mel term on the **real** rows, outside Δ (times `aux_mel_loss_weight`). The DDO real term saturates within a few hundred updates, after which nothing holds θ to real speech and silence is a cheap way "away from `p_ref`"; this anchors `x_pred`'s spectrum on real rows for the rest of the round. 0 in round 1; try 1.0 from round 2 and read `gen/silence_ratio` against the corpus' 0.034 |
| `ddo.real_fake_ratio` | 1.0 | target real:fake **frame** ratio per batch, enforced by repeating the pool's indices rather than by pool size |
| `ddo.fake_cfg_strength` | 0.0 | recorded only. A guided sample is not a sample of `p_ref`, so the likelihood-ratio identity would stop holding |
| `optim.max_updates` | 9000 | the round length, ~1% of pretraining; the run stops here regardless of `epochs` |
| `optim.lr_decay_end_factor` | 0.3 | the LR floor after warmup, as a fraction of the peak. The paper never anneals a round to zero (CIFAR: warmup only; EDM2: inverse-sqrt to ~0.4× by the end); the old floor of 1e-8 would leave the last third of a round standing still |
| `optim.ema_kwargs` | `beta: 0.999, update_every: 1, update_after_step: 0, power: 1.0` | ~1000 updates of averaging from update 1000 on. `make_pretrained_init.py` zeroes the EMA step, so `ema_pytorch`'s decay ramp sets the window rather than `beta`: under the default ramp `beta` is never reached inside a round and the window ends near 4300 updates, half the round — and every sample and `gen/*` metric is read off the EMA weights |

### Calibrating β

**Do not copy the paper's `0.01–0.1`.** That range is stated for a `Δ` summed over 3072
dimensions, magnitude `O(10³)`. This port's `Δ` is a per-element masked mean over
variable-length rows, magnitude `O(10⁻³)` — five to six orders of magnitude smaller. Pasting
`0.02` in puts `βΔ` at `~1e-5`, `logsigmoid` never leaves its linear midpoint, and the DDO
term is off in all but name. Nothing errors.

1. Run ~300 updates at `beta: 1.0` and read `ddo/delta_real`, `ddo/delta_fake`,
   `ddo/delta_std`. At update 0 `θ == p_ref` exactly, so `Δ ≡ 0` and `delta_std == 0` — the
   numbers only mean anything once the run has moved.
2. Set `β ≈ 1 / delta_std`, which puts `βΔ` at `O(1)`.
3. Trim with `ddo/acc`, **target 0.6–0.75**. `acc > 0.9` reached quickly ⇒ β too large,
   sigmoid saturated, gradients gone. `acc ≈ 0.5` while `delta_std` keeps growing ⇒ β too
   small, the DDO term is not doing anything.

### What to watch

`gen/utmos` and `gen/spk_sim_self` are this repo's stand-in for FID and the only real
verdict — the paper's author sums up "as long as the FID is decreasing, the training is
normal". Alongside them:

- `ddo/delta_real`, `ddo/delta_fake` — healthy is the two drifting slowly apart
- `ddo/margin` — their difference, the quantity the official trainer logs
- `ddo/delta_std` — what β is calibrated against
- `ddo/acc` — target 0.6–0.75; ties count a half, and a one-sided batch logs `nan` rather
  than a number that looks like an accuracy but is only half of one
- `ddo/loss_real`, `ddo/loss_fake`, `anchor_loss` — logged apart, so it is visible which term
  is driving the update
- `flow_loss` — **a divergence alarm, not a quality metric.** DDO trades likelihood for
  quality, so it is *expected* to rise; past roughly 2× the pretraining value the round has
  been pushed too far

Failure looks like one of two things, and the 300-update β probe shows both without burning
a round: `ddo/acc` stuck at 0.5 (signal too weak) or racing to 0.99 (saturated β, or the
discriminator found a shortcut — check that the pool's durations, loudness, frame grid and
32-bit float precision all match the real corpus).

### Multiple rounds

One round is not the method. Diffusion models in the paper need **12–28 rounds**, each
short (0.3–0.8% of pretraining). Round `n+1` is this config with a **new `model.name`**, and with `ddo.ref_ckpt` and
`ddo.fake_dataset` pointing at round `n`'s **best** checkpoint and a pool regenerated from
it. In practice round 1 peaked at update 300 on 144 paired clips (UTMOS 3.03 → 3.60) and
was *below* the baseline by 9000, so the useful part of a round here is its first few
hundred updates: `WavTTS_ddo_r2.yaml` runs 1500 with a checkpoint every 250 and starts
from round 1's update-300 checkpoint. The name is not cosmetic: `save_dir` derives from it and the trainer resumes whatever
`model_last.pt` it finds there, so under round `n`'s name round `n+1` would full-state-resume
round `n` at its final update — optimizer state and all — and stop after one batch. The
trainer refuses a finished run, and `train.py` refuses a `save_dir` with no `pretrained_*.pt`
in it (that would train a random init against a pretrained `p_ref`). Best, not last: a round provably does not converge — quality bottoms out mid-round and
then gets worse again, which is why `save_per_updates` is 1000 here and nothing is rotated
away. Rounds do not carry optimizer state, and α/β are worth re-sweeping each time.

Full rationale, the four deliberate departures from the paper, and the cost model:
[`docs/superpowers/specs/2026-09-11-ddo-design.md`](docs/superpowers/specs/2026-09-11-ddo-design.md)
and [`docs/superpowers/plans/2026-09-11-ddo.md`](docs/superpowers/plans/2026-09-11-ddo.md).

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
uv run pytest tests/ -v
```

CPU-only. `test_uncond_smoke.py` covers the mixing math (including the no-leak prefix
property), CFG paths, both solvers, seed isolation, metrics, and the CLI end-to-end;
`test_ddo.py`, `test_ddo_data.py` and `test_ddo_train.py` cover the DDO loss, the fake-pool
data path and the training wiring.

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
