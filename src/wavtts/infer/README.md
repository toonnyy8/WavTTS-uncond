# Inference

WavTTS supports zero-shot TTS from a reference audio prompt via either command-line or script-based inference.

> **💡 Tip for Reference Audio:**
> For optimal results, use a short reference clip, preferably under 12 seconds. We also recommend leaving a brief silence at the end of the clip to avoid truncating the prompt mid-word.

The pretrained checkpoint and matching vocabulary are available on [🤗 Hugging Face](https://huggingface.co/worstchan/WavTTS) and the checkpoint will be downloaded automatically when using the default configuration. To use a local checkpoint, specify its path with the `ckpt_file` parameter.


## CLI Inference

> **Note:** `wavtts_infer-cli` is an alias for `python src/wavtts/infer/infer_cli.py`. You can use either command interchangeably.

CLI inference can be run either with direct arguments or with a TOML configuration file. Command-line arguments always override values defined in the TOML config.

### A. Direct Arguments

Run inference by passing the required arguments directly:

```bash
wavtts_infer-cli \
  --model WavTTS \
  --ref_audio "infer/examples/basic_ref_en.wav" \
  --ref_text "Some call me nature, others call me mother nature." \
  --gen_text "The text you want WavTTS to synthesize."
```
*(Optional: Use `--model_cfg` instead of `--model` to provide an explicit YAML model config path.)*

### B. TOML Configuration

For reproducible runs, we recommend storing inference settings in a `.toml` file. If no config is provided, `wavtts_infer-cli` uses the default example config at `src/wavtts/infer/examples/basic.toml`.

```bash
# Use the default example config
wavtts_infer-cli -c src/wavtts/infer/examples/basic.toml

# Use a custom config with optional argument overrides
wavtts_infer-cli -c custom.toml --gen_text "Override text here."
```

**Example `custom.toml`:**
```toml
# Model settings
model = "WavTTS"
ckpt_file = "" # Leave empty to use the Hugging Face default
vocab_file = "infer/examples/vocab.txt"

# Prompt and generation
ref_audio = "infer/examples/basic_ref_en.wav"
ref_text = "Some call me nature, others call me mother nature."
gen_text = "The text you want WavTTS to synthesize."

# Output settings
output_dir = "output"
output_file = "infer_cli_basic.wav"
remove_silence = false

# Generation hyperparameters
nfe_step = 50
cfg_strength = 3.0
timestep_mapping = "power"
timestep_power = 2.0
shift = 3.0
speed = 1.0
```

## Script-based Inference

For customized pipelines or evaluation, you can modify and run the provided bash scripts.

### Single-sample Inference
Edit the paths and text in `src/wavtts/infer/infer.sh`, then execute:

```bash
bash src/wavtts/infer/infer.sh
```

### Batch Inference
For batch inference, configure the task and dataset paths in `src/wavtts/eval/eval_infer_batch.sh`, then run:

```bash
bash src/wavtts/eval/eval_infer_batch.sh --infer-only
```

> **💡 Tip:**
> If you notice obvious background noise in the synthesized speech, try lowering the `shift` value to mitigate it (e.g., set `shift = 1.0`).
---

## Zero-shot speaker conditioning on the unconditional model

`spk_cond.py` conditions a purely **unconditional** WavTTS checkpoint on a speaker at
inference time, with no training and no architecture change. Generation runs on a
concatenated `[ref | gen]` sequence and the ref half is pinned back onto a reference
trajectory after every ODE step; attention is the only path by which identity reaches the
generated half. This is RePaint-style replacement adapted to the rectified-flow ODE —
see `docs/superpowers/plans/wavtts_inversion_replacement_spk_cond.md`.

```bash
python -m wavtts.infer.spk_cond \
  --ckpt ckpts/<run>/model_last.pt --config ckpts/<run>/<date>/<time>/.hydra/config.yaml \
  --ref path/to/reference.wav --ref_sec 5 --duration_sec 5 \
  --resample_rounds 4 --repl_cfg 1 --with_ref --out_dir samples_spk_cond   # best setting
```

`--ref_traj` picks how the reference trajectory is built:

* **`noise`** (default) — `x^ref_t = t·x_ref + (1-t)·eps` with one fixed `eps`. Every
  pinned state is an exact sample of the training marginal `p(x_t | x_1 = x_ref)`, which
  is all the replacement needs, and it costs nothing.
* **`invert`** — the plan's §3.1 ODE inversion. The flow map is **not numerically
  invertible** on these checkpoints: round-trip reconstruction sits near 0 dB against the
  plan's 25 dB gate at every NFE from 16 to 256. Three knobs make it usable anyway:
  `--invert_t_start` (default 0.95; `1.0` is the plan verbatim) enters the ODE below the
  untrained top of the `t` range, `--invert_steps` decouples inversion NFE from sampling
  NFE, and `--invert_renorm` moment-matches the implied noise path. Still behind `noise`,
  so it stays the ablation — see the table below.

  Note the 25 dB gate is only a statement about invertibility. The forward-noising path
  scores −3 dB on it and conditions best of all, so a low number there is not by itself
  a reason to distrust a sample.

`--resample_rounds N` is RePaint's time-travel loop over `--resample_range` (default
`t ∈ [0.15, 0.5]`, where global structure is decided and the ref half is still mostly
noise).

`--repl_cfg S` guides between the replaced and un-replaced branches: each step also
evaluates the velocity of the generated half **on its own**, and extrapolates the full
sequence's velocity away from it by `S`. Their difference is the reference's entire
contribution through attention, so this amplifies the conditioning with no extra
conditioning signal — classifier-free guidance with "replacement" in place of a class
label. It costs a second forward per step and is unrelated to `--cfg_strength`, which
guides on the model's own trained null branch and is a no-op on any checkpoint with
`state_null_prob = 0`.

The combination is computed on the data prediction, not the velocity, and
`--repl_cfg_rescale` (default 1) then puts the conditional branch's mean and standard
deviation back before converting to velocity. The domain change alone is a no-op — the
extrapolation is affine and so is x ↔ v at fixed t — but the rescale is not affine, and a
standard deviation only means anything in the data domain.

### Measured speaker consistency

`python -m wavtts.eval.eval_spk_cond --ckpt ... --config ... --num_speakers 20 --utmos`

20 unseen speakers from LibriTTS `test-other` (disjoint from the train-clean-100/360
these checkpoints are trained on), 5 s reference, 5 s generated, 32 steps, embedder
`microsoft/wavlm-base-plus-sv`. `WavTTS_Uncond_Large_RPE_Clean_OLA_Init100_LibriTTS_460`
@ `model_last`:

| condition | SIM ↑ | Δ vs uncond | wins | gap closed | UTMOS |
|---|---|---|---|---|---|
| `real_same` (same speaker, real) — ceiling | 0.949 ± 0.029 | +0.375 | 20/20 | 100 % | 3.49 |
| `noise_u4_g1` (U=4, guidance 1) — **best** | **0.840 ± 0.075** | +0.265 | 17/20 | 71 % | 3.43 |
| `noise_g2` (U=1, guidance 2) | 0.825 ± 0.070 | +0.251 | 19/20 | 67 % | 3.03 |
| `noise_u4` (replacement, U=4) | 0.811 ± 0.098 | +0.236 | 19/20 | 63 % | 3.62 |
| `noise` (replacement, U=1) | 0.791 ± 0.084 | +0.217 | 17/20 | 58 % | 2.79 |
| `invert` (ODE inversion, U=1, `t_start=1.0`) | 0.614 ± 0.136 | +0.039 | 13/20 | 10 % | 1.36 |
| `uncond` (no conditioning) — floor | 0.575 ± 0.178 | — | — | — | 2.59 |
| `real_diff` (different speaker, real) — chance | 0.590 ± 0.215 | +0.015 | 10/20 | 4 % | 3.49 |

The conditioning works: unconditional sampling scores at chance against the reference
(0.575, against the 0.590 different-speaker floor), while replacement closes 58–69 % of
the distance to a real same-speaker utterance, on 17–20 of 20 references. `invert` at the
plan's literal `t_start = 1.0` is barely above chance; the next section takes it to 0.745,
which is still short of forward noising.

The UTMOS column says replacement adds no artifacts — but part of `noise_u4`'s 3.62 is
just the extra NFE (68 vs 32: `--resample_rounds 4` triples the 12 steps that fall inside
`--resample_range`). An NFE-matched unconditional baseline at 68 steps scores UTMOS 2.99
and SIM 0.598, still at chance, so the identity gain is not a compute artifact.

Identity drifts with generated length (`noise_u4`, gap closed): **67 % at 3 s → 63 % at
5 s → 47 % at 10 s → 41 % at 15 s**, and the per-reference win rate falls with it (17/20
→ 19/20 → 15/20 → 14/20). That curve is the case for the plan's §3.5 rolling generation.

### Replacement guidance (`--repl_cfg`)

Same 20 speakers. U is `--resample_rounds`, S is `--repl_cfg`; NFE counts model forwards
per generated sample (guidance doubles them, resampling triples the 12 steps inside
`--resample_range`).

Rescale off (φ=0) is plain extrapolation; rescale on (φ=1) restores the conditional
branch's moments in the data domain.

| U | S | φ | NFE | SIM | gap closed | wins | UTMOS |
|---|---|---|---|---|---|---|---|
| 1 | 0 | — | 32 | 0.791 | 58 % | 17/20 | 2.79 |
| 1 | 2 | 0 | 64 | 0.826 | 67 % | **20/20** | 2.83 |
| 1 | 2 | 1 | 64 | 0.825 | 67 % | 19/20 | **3.03** |
| 1 | 4 | 0 | 64 | 0.752 | 47 % | 16/20 | 1.67 |
| 1 | 4 | 1 | 64 | 0.767 | 51 % | 15/20 | 1.46 |
| 1 | 6 | 1 | 64 | 0.745 | 45 % | 15/20 | 1.28 |
| 4 | 0 | — | 68 | 0.811 | 63 % | 19/20 | 3.62 |
| 4 | 1 | 0 | 136 | 0.833 | 69 % | 19/20 | 3.46 |
| 4 | 1 | 1 | 136 | **0.840** | **71 %** | 17/20 | 3.43 |
| 4 | 2 | 0 | 136 | 0.813 | 64 % | 19/20 | 2.09 |
| 4 | 2 | 1 | 136 | 0.792 | 58 % | 18/20 | 3.00 |
| invert (`t_start` 0.95) | 2 | 0 | 96 | 0.734 | 42 % | 17/20 | 1.59 |

Guidance is worth about as much as resampling, but the two are **partial substitutes,
not additive**: the best strength falls as U rises (S=2 at U=1, S=1 at U=4), and S=2 on
top of U=4 is already over-guided. Both push the generated half toward agreeing with the
reference, so stacking them at full strength overshoots — the failure is textbook CFG
over-extrapolation, SIM falling *and* UTMOS collapsing together.

The rescale buys naturalness at the useful strengths (+0.20 UTMOS at U=1 S=2, +0.91 at
U=4 S=2) for no loss of identity, and **cannot rescue over-guidance**. The reason is
measurable: the guidance direction `x_cond − x_free` has cosine similarity **0.75–0.83**
with `x_cond` itself over `t ≥ 0.25`, so about four fifths of the reference's contribution
is collinear with the prediction — a gain change. Moment matching removes exactly that
part, cancelling most of the guidance along with the inflation it caused (which reaches
3.4× at S=6) and leaving the orthogonal fifth amplified sixfold. That fifth is the
artifact. Absolute normalization to `target_rms` would be worse still: the conditional
prediction's own scale runs 0.14 at t=0 up to 1.08 at t≈0.6, so there is no fixed target
to normalize to.

`noise_u4_g1` with the rescale is the best setting found: SIM 0.840, 71 % of the gap
closed, UTMOS 3.43 against real speech's 3.49. `noise_g2` gets 98 % of that identity at
64 NFE instead of 136.

On `invert` (`t_start` 0.95, φ=1) guidance helps only in a narrow band, and resampling
turns actively harmful:

| U | S | SIM | gap closed | wins | UTMOS |
|---|---|---|---|---|---|
| 1 | 0 | 0.731 | 42 % | 15/20 | 1.94 |
| 1 | 1 | **0.769** | **52 %** | 18/20 | **2.36** |
| 1 | 2 | 0.712 | 37 % | 13/20 | 1.49 |
| 4 | 1 | 0.612 | 10 % | 10/20 | 1.86 |

Both knobs amplify how hard the generated half is pushed to agree with what is pinned
next to it, so both are only as good as the thing pinned. With the forward-noising path
that is a valid marginal sample and U=4 is the best setting there; with the inverted path
it is a state the model never visits, and U=4 drags the output all the way back to chance
(0.612, spread [0.109, 0.928]). S=1 is mild enough to still net out positive; S=2 is not.
Best inverted result is 0.769 against 0.840 for forward noising, at higher cost.

### Rescuing the inversion

Same 20 speakers. `t_start` is `--invert_t_start`, `K_inv` is `--invert_steps`, `renorm`
is `--invert_renorm`; `n_ref rms` is the noise endpoint's RMS, which the training marginal
puts at 1.0.

| t_start | K_inv | renorm | n_ref rms | SIM | gap closed | UTMOS |
|---|---|---|---|---|---|---|
| 1.00 (plan §3.1) | 32 | — | 0.66 | 0.614 | 10 % | 1.36 |
| 0.95 | 32 | — | 0.98 | 0.731 | 42 % | — |
| 0.95 | 32 | ✓ | 1.00 | 0.720 | 39 % | — |
| 0.90 | 128 | — | 0.98 | 0.736 | 43 % | 2.22 |
| 0.90 | 128 | ✓ | 1.00 | 0.745 | 46 % | — |
| 0.90 | 128, U=4 | — | 0.98 | 0.740 | 44 % | 3.24 |

Entering below `t = 1` does nearly all of it. That one change takes the noise endpoint
from RMS 0.66 to 0.98 and SIM from 0.614 (chance) to 0.731 — the drift was produced
entirely by integrating an unlearned velocity field through a diverging `1/(1-t)` in the
last few percent of the schedule. Raising `K_inv` adds a little; `renorm` adds nothing
once `t_start < 1`, because by then there is no miscalibration left to correct. Note also
that resampling barely moves `invert` (+0.004) where it clearly moves `noise` (+0.020).

The rescued inversion still loses to plain forward noising on identity (0.745 vs 0.811)
and naturalness (3.24 vs 3.62) while costing 128 extra forward passes. Which is the
expected outcome: conditioning needs the pinned states to be plausible marginal samples,
not to lie on the model's own trajectory — that is an *editing* requirement. RePaint uses
forward noising for exactly this reason.

Not implemented from the plan: speaker-embedding guidance (§3.4) and rolling long-form
generation (§3.5).


---

## Context-consistency guidance for long rollouts (CCG)

`docs/superpowers/plans/context_consistency_guidance.md` implemented in `spk_cond.py`
(`CCG`, `generate_long`) and evaluated by `wavtts.eval.eval_ccg`.

No reference speaker is involved. The model invents a voice in its first segment and each
later segment is conditioned on the tail of what came before; the question is whether the
voice survives to the end of the clip.

```bash
python -m wavtts.infer.spk_cond --ckpt ... --config ... --ref anything.wav \
  --long_sec 30 --seg_sec 5 --ctx_sec 5 --repl_cfg 1 --out_dir samples_long
```

The plan's two branches are the construction the speaker-conditioning guidance above
already uses: the **long branch** runs the model on `[ctx | cur]` and takes the current
slice, the **short branch** runs it on `cur` alone, and `Δ = v_long − v_short` is the
context's whole contribution through attention. `--repl_cfg` is the plan's `w` minus one
(0 is the plain long branch, i.e. ordinary replacement). What the plan adds:

* `--ccg_window T_LO T_HI` — the limited-interval schedule, full strength below `T_LO`
  tapering to 0 at `T_HI`. **This repo's t runs the other way from the plan's**: the
  plan's τ_lo=0.5, τ_hi=0.9 is `--ccg_window 0.1 0.5` here.
* `--ccg_lp_k` — Φ_LP, decimate-and-interpolate low-pass on Δ
* `--ccg_eta_apg` — Φ_APG, how much of Δ along the conditional prediction survives
* `--ccg_kappa` — context noise level `t_ctx = 1 − κ(1−t)`; 1 is the plan's design (A)
* `--long_sec` — generate a clip this long, with `--sampler` choosing how:
  * `rolling` — the plan's §4 segment-by-segment rollout (`--seg_sec`, `--ctx_sec`). The
    context is pinned rather than regenerated, so segments abut on audio the model
    actually produced and no crossfade is needed to join them.
  * `ola` (default) — every ODE step is taken on the whole waveform at once. With
    `--win_sec 0` (the default) that is a single pass; with a window set, the short
    branch is the plain average of overlapping `--win_sec` windows spaced
    `--ola_hop_sec` apart and the long branch is one forward on the entire clip, so
    `--repl_cfg S` blends `v = v_windowed + S·(v_full − v_windowed)`.

### Results

8 unconditional 30 s rollouts, 5 s segments, 5 s context, U=4, rescale on. `drift` is
each segment against the first, `adjacent` is neighbouring segments, `switch` is the
fraction of joins below 0.8. `real` is a genuine continuous LibriTTS recording chopped
the same way — the ceiling, and where the embedder's own noise floor shows.

| condition | drift | worst | adjacent | switch ↓ | UTMOS | gap closed |
|---|---|---|---|---|---|---|
| `real` (continuous recording) — ceiling | 0.964 | 0.943 | 0.962 | 0.00 | 3.46 | 100 % |
| `w2_win0.1-0.5` | **0.893** | **0.845** | 0.921 | 0.00 | 2.75 | **74 %** |
| `w1` | 0.881 | 0.805 | **0.932** | 0.00 | **2.79** | 70 % |
| `w2_lp8` | 0.879 | 0.798 | 0.924 | 0.03 | 2.72 | 69 % |
| `w0` (replacement AR, no guidance) | 0.847 | 0.759 | 0.900 | 0.08 | 2.71 | 58 % |
| `w2` | 0.793 | 0.643 | 0.760 | 0.38 | 2.25 | 38 % |
| `naive` (independent segments) — floor | 0.686 | 0.533 | 0.734 | 0.60 | 1.75 | 0 % |

Rolling context is the large effect: independent segments drift to 0.686 and switch voice
at 60 % of the joins, and simply pinning the previous tail takes that to 0.847 / 8 %.
**Guidance is then a real further gain, not noise** — w=1 closes 70 % of the naive→real
gap against w=0's 58 %, removes the last switches, lifts the worst segment from 0.759 to
0.805, and costs nothing in UTMOS. The best drift number belongs to w=2 with the plan's
limited-interval schedule (0.893, 74 %), which also has the best worst-segment; w=1 has
the better adjacent similarity and naturalness. Either is a defensible default.

**w=2 unguarded collapses** (0.793, switch 0.38) even though the same strength is fine
for single-segment conditioning. Errors compound: a segment's artifacts become the next
segment's context, so over-guidance is self-amplifying here in a way it is not in one
shot. Both the limited-interval schedule and Φ_LP recover it (0.893 / 0.879), which is
the clearest evidence in this set that the plan's §3.2 operators do what they claim.

Also measured, in the reference-anchored variant of this rollout: `--ccg_kappa 0.5`
(design C, fractional context noise) **fails hard** — every join a switch, UTMOS 1.32.
The plan flags this: a prior trained with one noise level per sample sees a context
cleaner than the segment beside it as out of distribution and needs Diffusion-Forcing
style per-token noise training. Do not use κ<1 here. Φ_APG (η=0) changed nothing there
either.

### The windowed sampler is solving a problem this model does not have

Once the guidance is set up as "windows versus the whole clip", the obvious control is to
skip the windows entirely. 6 unconditional 60 s clips, same metrics, `full` being one
single-pass generation of the whole 60 s:

| condition | drift | worst | adjacent | switch ↓ | UTMOS |
|---|---|---|---|---|---|
| `real` (continuous recording) — ceiling | 0.965 | 0.943 | 0.964 | 0.00 | 3.37 |
| **`full`** (single pass over 60 s) | **0.951** | **0.932** | **0.956** | 0.00 | **3.05** |
| `ola_w0.5` (windows guided toward the full pass) | 0.936 | 0.874 | 0.938 | 0.02 | 2.74 |
| `ola_w0` (windows alone) | 0.781 | 0.580 | 0.899 | 0.08 | 2.41 |

Doing nothing wins. A single 60 s forward lands within 0.014 of a real recording on drift
and holds every metric above the windowed sampler; chopping the clip into 10 s windows
costs 0.17 of drift and 0.6 UTMOS, and guiding back toward the full pass recovers most
but not all of that. At 30 s the same ordering holds with a smaller spread (`full` 0.935 /
3.11, `ola_w0` 0.806 / 2.77).

`rpe_gamma: 4.0` is why. Training tops out at 30 s clips, but randomized positional
encoding shows the model positions equivalent to 119 s, so a 60 s forward is still inside
the range it was trained on — 12001 tokens at attention temperature 1.400, 0.35 s and
2.8 GiB. The windowed machinery is worth reaching for only past that, where the full pass
runs on positions the model has never seen. Untested here.

### The plan's own validation, and what it says about the ceiling

Running §5.7's checks on this model, comparing Δ from a correct-speaker context against Δ
from a *different speaker's* context on the same segment:

| t (noise → data) | 0.00 | 0.25 | 0.50 | 0.75 | 0.88 |
|---|---|---|---|---|---|
| ‖Δ‖ / ‖v_short‖ | 0.08 | 0.53 | 0.88 | 0.77 | 0.61 |
| cos(Δ_correct, Δ_wrong) | 1.00 | 0.96 | 0.99 | 0.99 | 0.97 |
| cos(Δ, v_short) | 0.18 | −0.83 | −0.86 | −0.70 | −0.74 |

Check 3 fails: ‖Δ‖/‖v_short‖ should sit in 0.05–0.5 and reaches 0.88 over most of the
schedule. Check 2 fails harder and is the informative one — Δ computed against the wrong
speaker's context has cosine **0.96–0.99** with Δ computed against the right one. The
plan's premise is that Δ ≈ ∇ log p(ctx | cur) is identity evidence; measured, most of its
magnitude is generic "there is context to my left, continue an utterance" and only about
a quarter (‖Δ_r − Δ_w‖ ≈ 0.28‖Δ‖) is speaker-specific.

That is consistent with what the rollouts show. Guidance helps because the generic part
genuinely is what long-sequence coherence needs, and it stops helping at w=2 because
amplifying that part further overwhelms the model rather than sharpening identity. It
also explains Φ_APG doing nothing: with cos(Δ, v_short) ≈ −0.8, η=0 discards four fifths
of Δ's magnitude and the result is unchanged, so the discarded part was neither the
damage nor the benefit. Φ_LP, which keeps low-frequency structure rather than a
direction, is the operator that actually works.

Closing the last 26 % to a real recording likely needs a Δ that isolates identity —
contrasting the true context against a *decoy* context rather than against no context at
all. That is a different algorithm and is not implemented here.
