# WavTTS 無條件語音生成 + 混合語音負樣本 CFG — 設計文件

日期：2026-08-19
狀態：已與作者於對話中確認方向（從零訓練、batch-roll 混合、就地修改）

## 目標

把 WavTTS 從 zero-shot TTS 改造為**無條件純語音生成模型**：

- 移除 text conditioning 與 audio prompt（infilling）conditioning。
- 模型唯一的條件輸入是一個三值狀態標記：`clean=0`（單一語者乾淨語音）、
  `mixed=1`（兩位語者混合語音）、`null=2`（無條件）。
- 訓練時以 **no-leaky 混合增強**產生 mixed 樣本，兩種型態各半：
  (a) **overlap**——整段波形等功率比例混入 batch 內另一條語音（多人疊音）；
  (b) **concat**——前後段由不同語者串接、切換處等功率 crossfade（時間軸換人）。
  標記與內容完全一致，且不留下可作弊的邊界 artifact，不洩漏乾淨答案。
- 推論時以 CFG 做**負樣本引導**：正分支 `clean`、負分支 `mixed`（預設）或 `null`，
  `v = v_clean + w · (v_clean − v_neg)`，把生成推離「多語者混雜」方向，
  得到語者一致的單人語音。

## 非目標

- 不保留 TTS 功能。`src/wavtts/infer/infer_cli.py`、`utils_infer.py`、`eval/` 目錄
  與新模型不相容，**保留原檔但不維護**（不再被 import）。
- 不做 speaker prompt / voice cloning；模型沒有任何語者身分輸入。
- 不微調既有 checkpoint；從零訓練。

## 架構變更（就地修改）

### `src/wavtts/model/backbones/dit.py`

- 刪除 `TextEmbedding`、text cache（`text_cond`/`text_uncond`/`clear_cache`）、
  `get_input_embed` 的 text 邏輯。
- `InputEmbedding` 簡化為單一音訊分支：`x [b, n, wav_frame_len] → dim`
  （保留 `use_audio_proj` 兩層投影選項；刪除 `cond_proj` 與 text 串接），
  `ConvPositionEmbedding` 照舊。
- 新增 `self.state_embed = nn.Embedding(3, dim)`；forward 時
  `t = time_embed(time) + state_embed(state)`（state 為 `[b]` int tensor），
  即標準 DiT class-conditioning。
- `forward(x, state, time, mask=None, cfg_infer=False, neg_state=None, lens=None)`：
  `cfg_infer=True` 時打包正分支（傳入的 `state`）與負分支（`neg_state`）
  成 2b batch 一次 forward。
- waveform patchify（`wav_frame_len=160`、`_wav_to_tokens`/`_tokens_to_wav`）、
  DiT blocks、AdaLN、zero-init、`proj_out` 全部不變。

### `src/wavtts/model/cfm.py`

移除：`vocab_char_map`、text 處理、`frac_lengths_mask`、`rand_span_mask` infilling、
`audio_drop_prob` / `cond_drop_prob` / `joint_cond_drop_prob`、`mask_align_to` 對齊機制。

新增 CFM 參數（含預設值）：

| 參數 | 預設 | 意義 |
|---|---|---|
| `p_mix` | 0.5 | 每個樣本被做成 mixed 的機率 |
| `p_concat` | 0.5 | mixed 樣本中採 concat 型態的比例（其餘為 overlap） |
| `mix_lambda_range` | (0.3, 0.7) | overlap 混合能量比 λ 的均勻取樣範圍 |
| `concat_point_range` | (0.3, 0.7) | concat 切換點位置（佔有效長度比例）的均勻取樣範圍 |
| `concat_xfade_ms` | 20 | concat 切換處等功率 crossfade 長度（毫秒） |
| `state_drop_prob` | 0.1 | 標記被 drop 成 `null` 的機率（保留純無條件採樣能力） |

#### 訓練 `forward(inp, lens=None)`

1. 對每個樣本抽 Bernoulli(`p_mix`) 決定是否混合；混合來源一律為
   `partner = inp.roll(1, dims=0)`（batch-roll，零 IO 成本）。
2. 混合樣本再抽 Bernoulli(`p_concat`) 決定型態：
   - **overlap**：λ ~ U(`mix_lambda_range`)，等功率混合
     `x1 = sqrt(1−λ)·inp + sqrt(λ)·partner`（多人疊音負樣本）。
   - **concat**：切換點 `s = u·len`，u ~ U(`concat_point_range`)；
     `x1[:s] = inp[:s]`、`x1[s:] = partner[s:]`，切換處以 `concat_xfade_ms`
     的等功率（cos/sin）crossfade 銜接（時間軸換人負樣本）。
     Crossfade 是 no-leaky 的必要條件：硬接的 click artifact 會讓模型
     以邊界瑕疵而非語者身分辨識 mixed，CFG 引導就會失效。
   兩種型態共用同一個 `mixed` 標記——負方向是「語者不一致」的統稱，
   不分 class、不增加推論期旋鈕。
   `# ponytail: 1 switch point per concat sample; go 1-2 random switches if
   the model only learns late-utterance consistency.`
3. 標記：混合樣本 `mixed`，其餘 `clean`；再以 `state_drop_prob` 逐樣本改為 `null`。
4. Flow matching 與現況相同（`prediction=x_pred`、`loss_space=v`、
   logistic-normal t 取樣、`latents_scale`），但 loss 遮罩改為整段有效長度
   （`lens_to_mask(lens)`），不再有 infilling span。
5. Aux mel loss 照舊，`frame_mask` 傳長度遮罩、`frame_lengths=lens`。

已知限制（接受）：partner 比本樣本短時，overlap 的 padding 零值區混不到干擾語音、
concat 的後段可能含 partner 的尾端靜音，mixed 標記在該區段偏「乾淨」；
dynamic batch sampler 依長度排序，同 batch 長度相近，影響有限。`# ponytail: batch-roll mixing; switch to dataset-level pair loading if
label purity ever matters.`

#### 推論 `sample(...)`

新簽名：

```python
sample(duration, *, batch=1, steps=32, cfg_strength=2.0,
       negative="mixed",  # "mixed" | "null"
       sway_sampling_coef=None, timestep_mapping="sway_sampling",
       timestep_power=None, shift=1.0, use_epss=True, seed=None,
       solver="euler")  # "euler" | "dpmpp"
# duration: 生成樣本點數（int，16kHz）
# seed 使用獨立 torch.Generator，不會動到全域 RNG（訓練中途採樣安全）
```

- `y0 = randn(batch, duration)`，無任何 cond 填充/遮罩/裁切邏輯。
- ODE fn：`cfg_strength < 1e-5` 時單分支 `clean`；否則 packed forward
  正分支 `clean`、負分支 `negative` 對應的 state，
  `v = v_pos + cfg_strength · (v_pos − v_neg)`。
- timestep mapping（sway/EPSS/power/shift）與 Euler odeint 照舊。
- `solver="dpmpp"` 改用 DPM-Solver++(2M)（data-prediction 多步法）：對 rectified-flow
  插值取 `α_t=t, σ_t=1−t, λ_t=log(t/(1−t))`，端點以 `t_eps` 夾住奇異點，
  末步直接輸出 x_pred；每步由 CFG 後的 v 轉 `x_pred = x + (1−t)v`，
  CFG 組合程式碼與 Euler 路徑完全共用。可搭配任一 timestep 排程。
- 回傳 `out / latents_scale` 與 trajectory。

### `src/wavtts/model/dataset.py`

- `CustomDataset.__getitem__` 只回傳 `wav`（資料集磁碟格式不變，text 欄位忽略）。
- `collate_fn` 只回傳 `wav`、`wav_lengths`。
- `load_dataset` 不再需要 tokenizer 參數（保留參數位置相容或直接刪除，採直接刪除）。

### `src/wavtts/model/trainer.py`

- 移除 text 相關批次欄位與 `duration_predictor` 呼叫中的 text。
- `self.model(wav, lens=wav_lengths)`。
- `log_samples`：改為無條件生成——取 batch 第一條的長度（上限 10 秒）生成一條，
  存 `update_{n}_gen.wav`；不再存 ref。不再 import `utils_infer` 的常數，
  直接用固定值（`steps=32`、`cfg_strength=2.0`、`sway_sampling_coef=-1.0`）。

### `src/wavtts/train/train.py`

- 刪除 tokenizer/vocab 邏輯；`model_cls(**arch, wav_frame_len=...)`。

### `src/wavtts/configs/WavTTS.yaml`（就地修改）

- 刪除 `tokenizer`、`tokenizer_path`、arch 的 `text_dim`/`text_mask_padding`/
  `conv_layers`（text conv）欄位。
- `cfm` 新增 `p_mix: 0.5`、`p_concat: 0.5`、`mix_lambda_range: [0.3, 0.7]`、
  `concat_point_range: [0.3, 0.7]`、`concat_xfade_ms: 20`、`state_drop_prob: 0.1`；
  刪除 `joint_cond_drop_prob`。
- 其餘超參不變。

### 新增 `src/wavtts/infer/sample_uncond.py`

極簡 CLI：`--ckpt --duration_sec 5 --num 4 --steps 32 --cfg_strength 2.0
--negative mixed --solver euler|dpmpp --seed --out_dir --device`。
載入 checkpoint（EMA state dict 優先）、建模型、`sample()`、`torchaudio.save`。

## 訓練監控、可重現性與 DPM++ 採樣器（2026-08-19 追加）

### TensorBoard 監控（`trainer.py`、`train/metrics.py`）

- config 預設 `ckpts.logger: tensorboard`（wandb 仍可用，切回後 gen 指標記 scalar、
  音訊/圖僅存檔）；訓練結束 `writer.close()`。看板：`tensorboard --logdir runs`。
- 每次 checkpoint（`save_per_updates`）在主進程生成固定樣本並記錄：
  - `gen/audio_seed{k}`：音訊；`gen/mel_seed{k}`：log-mel 圖
    （torchaudio 80 mels + matplotlib，`add_figure`）。
  - wav 檔另存 `samples/update_{n}_seed{k}.wav`。

### 量化指標（`src/wavtts/train/metrics.py`，對所有 clip 取平均記 scalar）

| 指標 | 說明 | 依賴 |
|---|---|---|
| `gen/utmos` | UTMOS 預測 MOS（1–5），自然度單一數字 | torch.hub lazy 載入，失敗自動跳過 |
| `gen/silence_ratio` | 低能量 frame 佔比（崩成靜音的警報） | 純 torch |
| `gen/clipping_rate` | \|x\|>0.99 樣本比例（爆音警報） | 純 torch |
| `gen/rms` | 整體能量漂移 | 純 torch |
| `gen/spk_sim_self` | 前後半 ECAPA embedding cosine——語者一致性 直接指標 | opt-in：`ckpts.spk_ckpt_path`（WavLM-large ECAPA ckpt，同 eval 用），null 停用 |

指標一律 fail-safe：任何載入/計算失敗只印警告並跳過，不中斷訓練。

### 可重現性

- config 頂層 `seed: 666`：`train.py` 進 main 即呼叫既有 `seed_everything(seed)`
  （python/torch/cuda seed + `cudnn.deterministic`，訓練略慢但 run 級可重現），
  dataset shuffle 的 `resumable_with_seed` 同一值。
- **固定 seed 中間結果**：`ckpts.log_samples_seeds: [0, 1, 2, 3]`、
  `ckpts.log_samples_sec: 5`——每個 checkpoint 用同一組 seed、同一長度各生成一條，
  seed↔clip 對應固定，跨 checkpoint 可直接對照同一條 clip 的演化。
- `CFM.sample(seed=...)` 改用獨立 `torch.Generator`（原 `torch.manual_seed` 會
  重置全域 RNG，破壞訓練可重現性；已修並有測試鎖定）。

### DPM-Solver++(2M)

見上方 `sample(...)` 一節的 `solver="dpmpp"` 說明；trainer 的 checkpoint 採樣
維持 Euler（32 步）。

## 驗證

`tests/test_uncond_smoke.py`（不依賴資料集與 checkpoint）：

- 建一個縮小的 DiT（dim=64, depth=2）+ CFM。
- 隨機 `[4, 16000]` batch（含不等長 lens）跑 `forward` + `backward`，
  assert loss 有限、梯度存在。
- `sample(duration=8000, steps=2)`，assert 輸出 shape `[batch, 8000]` 且有限，
  分別測 `negative="mixed"` 與 `"null"` 與 `cfg_strength=0`。
- 追加：dpmpp solver（有/無 CFG）shape 與有限性、未知 solver 拒絕、
  `sample(seed=...)` 不動全域 RNG 且同 seed 輸出確定、訊號指標
  （silence/clipping/rms）數值正確、mel 圖可產生、CLI 端到端
  （model_state_dict 與 EMA checkpoint 兩種載入路徑）。

## 風險與備註

- CFG 負分支用 `mixed` 是本設計的核心假設：`(v_clean − v_mixed)` 方向近似
  「單語者純度」梯度。若實驗顯示引導過強造成 artifacts，可下調 `cfg_strength`
  或改 `negative="null"` 做傳統 CFG 對照。

### 負分支選擇：自我限制性質（2026-08-20 補記）

標準 CFG 的引導項是**分類器梯度**：

```
v_clean + w·(v_clean − v_null) = v_clean + w·∇log p(clean | x)
```

`null` 是真正的邊際分佈（`state_drop_prob` 不分 clean/mixed 均勻 drop，
所以 `p_null ≈ 0.5·p_clean + 0.5·p_mixed`）。當 x 已明確落在 clean 區域時
`p(clean|x) → 1`、梯度飽和趨近 0、`v_null → v_clean`，**引導自動熄火**。
這是 CFG 穩定的關鍵性質。

負樣本版本則是**似然比梯度**：

```
v_clean + w·(v_clean − v_mixed) = v_clean + w·∇log [ p(x|clean) / p(x|mixed) ]
```

這一項不熄火。x 越深入 clean 區域、離 mixed 流形越遠，`∇log p(x|mixed)` 反而
越強地指回 mixed 流形，減掉它等於越推越用力，可能衝出 clean 流形——
與 negative prompt 在高權重下過飽和是同一個機制。

反向的權衡：正因為 `null` 是 50% clean 的混合物，`v_clean − v_null` 對
「語者不一致」這個特定失效模式的推力較弱，要同樣強度就得開大 `w`，
而大 `w` 有自己的 artifact。兩邊都有代價。

**決定一**：不引入三分支加權形式
`v = v_clean + w·(v_clean − v_null) + w_neg·(v_clean − v_mixed)`。
它雖然能保留主引導的自我限制性質、又補上針對性的抗混合推力，但多一個
需要獨立調參的旋鈕，而且推論成本 +50%（packed forward 從 2b 變 3b）。
要加強度就調 `w`。

**決定二**：`mixed` 不再有自己的 state，**併入 `null`**。state 從三個縮為兩個
（`STATE_CLEAN = 0`、`STATE_NULL = 1`、`NUM_STATES = 2`），`sample()` 與 CLI
的 `negative` 參數移除。

標籤規則**先決定標籤、再決定增強**，兩個旋鈕因此正交：

1. 每個樣本以 `state_null_prob` 的機率標為 `null`，其餘 `clean`。
2. **只有 null 樣本**可能被混合增強，機率 `p_mix`。

於是 clean 分支是純粹的單語者語音，而
`p_null = (1−p_mix)·p_clean + p_mix·p_mixed`。
預設 `state_null_prob: 0.5`、`p_mix: 0.5` → 標籤剛好各半，null 內部也是
一半 clean、一半 mixed。

先前的參數化（混合樣本一律 null、再用 `state_drop_prob` 補一些 clean 進 null）
會讓兩個旋鈕互相綁死：要湊出 50/50 的標籤比例就得解一條方程式，而且
`p_mix = 0.5, state_drop_prob = 0` 這個看似自然的解會讓 null 裡**只有** mixed，
退化回被否決的似然比形式。改成上面的兩階段後不會發生這種事。

關鍵在於**自我熄火只需要 null 裡有 clean 成分，不需要它佔多數**：
x 深入 clean 區域時 `p_mixed(x)/p_clean(x) → 0` 的速度快過任何固定的混合權重，
所以 `p_null(x) → (1−p_mix)·p_clean(x)`、`∇log p_null → ∇log p_clean`、引導項歸零。
`p_mix` 只決定熄火發生的深度——大則抗混合推力維持得久、晚熄；小則早熄、溫和。
`p_mix = 1` 是唯一的退化點（null 只剩 mixed）。

這個安排同時保住兩件事：混合增強提供的語者一致性訊號仍在（藏在 null 裡），
而引導形式是自我限制的標準 CFG。刪掉混合增強則會兩者皆失——訓練資料只剩單語者語音時
`p_clean = p_null = p_data`，`v_clean − v_null → 0`，CFG 沒有東西可以放大，
只剩下 null 因為只看到少量資料而估得較差所產生的殘差（即 Karras 等人的 autoguidance，
有效但不受控、也不是這裡要的東西）。

`state_drop_prob` 因此被 `state_null_prob` 取代，語意也不同：前者是「額外把標籤
丟成 null」的補丁，後者是標籤分佈本身；而 `p_mix` 從「多少樣本被混合」變成
「null 樣本中多少被混合」，也就是決定引導何時熄火的旋鈕。
- `p_mix=0.5` 給負分支足夠的訓練訊號；若 clean 品質受影響可降至 0.3。
  overlap 與 concat 各教一件事（不疊音／不換人），`p_concat` 可依實驗調整比重；
  concat 型態與語者漂移失效模式最直接對應。
- 混合用等功率係數（`sqrt(1−λ)`, `sqrt(λ)`）避免破音；不做響度對齊
  （Emilia 已大致正規化）。
- batch-roll partner 可能偶爾是同語者（batch 依長度排序、非 speaker-disjoint），
  少量實質乾淨樣本會掛 mixed 標記；升級路徑同 padding 尾端限制——dataset 層配對載入。
