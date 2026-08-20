# WavTTS 無條件語音生成 — 長度外推（Randomized YaRN + 熵不變性）設計文件

日期：2026-08-20
狀態：已實作並開始訓練（分支 `length-extrapolation`）
前置文件：[`2026-08-19-uncond-speech-cfg-design.md`](2026-08-19-uncond-speech-cfg-design.md)
（其中「負分支選擇：自我限制性質」一節記錄了 `mixed` state 併入 `null` 的決定，
與本文件的長度外推改動同期，但兩者互相獨立）

## 問題

訓練資料（LibriTTS train-clean-100，33,188 條、53.7 小時）的長度分佈：

| | 秒 | frames @100Hz |
|---|---|---|
| max | 29.79 | 2979 |
| p99 | 21.29 | 2129 |
| p95 | 15.17 | 1517 |
| median | 4.55 | 455 |

架構本身沒有硬性長度上限——`ConvPositionEmbedding` 是 depthwise conv（純局部、與長度無關），
RoPE 是純相對、沒有 learned absolute position，`attn_mask_enabled: False` 也沒有 mask 限制。
`sample(duration=N)` 給多長都跑得動。真正的限制是**訓練時看過的相對距離範圍**：
超過 ~30s 後，attention 面對的是沒見過的旋轉角度，全域結構會崩（重複、漂移）。

目標：生成 120 秒。

## 方法

兩個機制，都**在訓練時就套用**而非只在推論期外掛，都可用 config 完全關掉
（`rope_type: default` 回到原行為，有測試逐位元鎖定）。

### 1. Randomized YaRN

出處：Mehta, Yin & Durrett, *Randomized YaRN Improves Length Generalization for
Long-Context Reasoning*（[arXiv:2606.23687](https://arxiv.org/abs/2606.23687)，NYU）。
參考實作：[Manas-Mehta/Randomized-YaRN](https://github.com/Manas-Mehta/Randomized-YaRN)。

論文名稱裡的 "Randomized" 指的是**隨機的位置索引，不是隨機的 scale factor**。方法是三件事疊加：

**(a) YaRN，scale 固定。** NTK-by-parts 插值：對每個 RoPE 維度算它在 native context 內
轉幾圈，轉超過 `beta` 圈的高頻維度原樣外推、不到 `alpha` 圈的低頻維度整個除以 `s`、
中間線性 ramp。訓練用固定的 `s`，推論用 `s'`。

```
λ_d = 2π / inv_freq_d                            # 該維度的波長
r_d = native_ctx / λ_d                           # 在 native context 內轉幾圈
γ_d = clamp((r_d − alpha) / (beta − alpha), 0, 1)
inv_freq_d' = inv_freq_d · (γ_d + (1 − γ_d) / s)
```

YaRN 自帶的 attention temperature `0.1·ln(s) + 1` 保留。訓練期 `s` 固定，這一項是常數、
會被權重吸收；它只有在推論 `s' ≠ s` 時才真的起作用。參考實作把它同時乘在 query 與 key 上，
等價於 logits 乘它的平方——我們的實作直接乘平方，數學上相同。

**(b) 隨機位置編碼（RPE，Ruoss et al. 2023）。** 訓練時位置不是 `arange(n)`，而是
從 `[0, L_t)` 抽 `n` 個不重複的索引、排序後使用。序列本身沒變長，token 順序也沒變，
只是**被告知彼此隔得比較遠**——短音檔因此也能操練到只有長音檔才會產生的旋轉角度。
只在 `self.training` 時生效，推論一律連續位置。

**(c) 長度課程。** `L_t` 隨訓練進度遞增。論文的 ablation 顯示拿掉 curriculum 會掉最多
18.3 分，這一項不是裝飾。

推論時跑標準 YaRN，scale 用 `s'`。論文 Appendix B 指出 `s' > s` 能解鎖超過 `s · native_ctx`
的長度，所以 `s=2` 訓練、`s'=4` 推論可覆蓋 120 秒。

### 2. 熵不變性（logn attention scaling）

softmax 的熵隨 key 數量成長，同一組權重在長序列上注意力會被攤平。對 logits 乘
`log(n) / log(logn_ref_len)` 讓熵維持穩定。

與論文無關，是獨立開關，也**訓練時就套用**。理由是這個模型跟 LLM 的處境不同：LLM 訓練長度
固定，只能在推論期硬套一個沒學過的係數；而我們的 clip 長度天然橫跨 30–2979 frames，
而且 batch 依長度排序，所以模型有機會在跨兩個數量級的 `n` 上**學到** `n ↔ scale` 的對應，
外推出去的那個值才有意義。`logn_ref_len = 3000` 時訓練期係數落在 0.43–1.0（median 0.76），
超過訓練長度才 > 1。

與 YaRN 的 attention temperature 正交：後者只看 `s`，前者只看實際 token 數 `n`；
兩者相乘後一起折進 softmax scale。RPE 只改位置、不改 `n`，所以三者互不干擾。

## 實作

新檔 `src/wavtts/model/rope.py`：

| 函式／類別 | 職責 |
|---|---|
| `yarn_inv_freq(dim, base, scale, native_ctx, alpha, beta)` | NTK-by-parts 插值後的 inverse frequencies |
| `yarn_attention_factor(scale)` | `0.1·ln(s) + 1`，`s ≤ 1` 時回 1.0 |
| `YaRNRotaryEmbedding` | 與 x_transformers `RotaryEmbedding` 同介面，`forward` 收任意位置張量 |
| `randomized_positions(batch, seq_len, max_len, device, per_sample)` | 排序後的不重複隨機位置 |
| `rpe_max_len(mode, seq_len, length_scale, native_ctx)` | 單一 batch 的 `L_t` |

改動落點比預期小，因為 x_transformers 的 `apply_rotary_pos_emb` 本來就支援 `[b, n, d]`
形狀的 freqs（會自己 rearrange 成 `b 1 n d`），而 `RotaryEmbedding.forward(t)` 收的就是
任意位置張量——**per-sample 隨機位置不需要動 attention 任何一行**。

- `backbones/dit.py`：依 config 選 rope 型別；forward 內建位置；
  `set_rpe_length_scale()`（課程）與 `set_yarn_scale()`（推論期 `s'`，重建 inv_freq
  與 attention temperature，不動任何參數）。
- `modules.py`：`AttnProcessor._logit_scale()` 把 logn 與 YaRN temperature 合成一個乘數，
  **折進 SDPA 的 `scale` 參數**而非乘在 query 上——後者會在 28 個 block 裡各配置一個
  `[b, h, n, d]` 新張量。`flash_attn` 路徑走 `softmax_scale`，兩個 backend 等價。
- `trainer.py`：`_advance_rpe_curriculum(update)` 按 update 推進；resume 時會接回正確的
  `k` 而不是從頭。
- `infer/sample_uncond.py`：`--yarn_scale` 設定 `s'`。

### 超參數對應

| 論文（Qwen2.5 / BABILong） | 本專案 |
|---|---|
| `L_pre` 32768 | `yarn_native_ctx: 3000` frames（30s，最長 utterance） |
| 訓練序列 ≤9216 | 實際 utterance 長度，≤2979 |
| `s = 2` | `yarn_scale: 2.0` |
| `s' = 4` | `--yarn_scale 4` → 120s |
| `L_t` 課程 8192→16384（≈2× 訓練長度） | `k` 課程 1.0→2.0 |
| `alpha = 1, beta = 32` | 同 |

課程：`optim.rpe_curriculum: [[0, 1.0], [20000, 1.25], [40000, 1.5], [60000, 2.0]]`。
`k = 1.0` 就是連續位置，所以 20k warmup 期間模型先學正常語音，之後才開始見到被拉伸的。
論文按 epoch 推進，但我們一個 epoch 只有 ~202 updates、總共十幾萬 updates，
按 epoch 走會抖得沒意義，因此改按 update 計。

## 三項刻意偏離參考來源的決定

### `rpe: relative`（預設）而非論文的固定 `L_t`

論文的序列長度都貼近訓練上限，固定 `L_t` 對每個樣本的拉伸倍率大致相同（≈1.8×）。
我們的 clip 是 0.3–30s，固定 `L_t = 6000` 的話：

| clip | 固定 `L_t` 的拉伸 | `L_t = k·n` 的拉伸 |
|---|---|---|
| 4.5s（median） | 13.2× | 2.0× |
| 15s | 4.0× | 2.0× |
| 29.8s | 2.0× | 2.0× |

文字上這沒問題——attention 主要靠內容、相對*順序*才是重點。但 100Hz 的波形建模裡，
相對位置**就是物理時間**，基頻週期、共振峰轉折、音素時長全靠它。中位數樣本被隨機拉伸
13 倍，很可能正好毀掉這個模型賴以為生的局部結構。（`ConvPositionEmbedding` 另外提供
真實的相鄰資訊，所以不至於致命，但 RoPE 是精細時序進入 attention 的唯一管道。）

因此 `rpe: relative` 讓 `L_t = k × 該樣本自己的長度`，拉伸倍率恆定。長度固定時
與論文完全等價。`rpe: absolute` 保留論文的字面行為（`L_t = k × native_ctx`）。

### `rpe_per_sample: True`（預設）

論文正文（§2.2「we draw a set of positions」）與 Ruoss et al. 都是**每個 batch 一組**位置；
參考實作 `ryarn/patching.py:79` 逐樣本 stack。兩者的差別是 freqs 從 `[1, n, d]` 變成
`[b, n, d]`，在 28 層裡各自算 cos/sin。

實測代價（frame budget 把 `b × n` 卡在 3200，所以 b 大時 n 必小，兩者不會同時大）：

| batch | per-batch | per-sample | delta |
|---|---|---|---|
| b=64 n=50 | 1.38 MiB | 88.28 MiB | +86.90 |
| b=8 n=400 | 11.04 MiB | 88.28 MiB | +77.25 |
| b=1 n=3200 | 88.28 MiB | 88.28 MiB | +0.00 |

88 MiB / 22 GiB = 0.4%，且是 micro-batch 內的暫時量。計算成本原本是 b 次 `randperm`
的 Python 迴圈（b=64 時 1.58 ms），改用 `torch.rand(draws, max_len).argsort(-1)[:, :n]`
（uniform noise 的 argsort 就是 permutation）後降到 0.069 ms，與 per-batch 同級。

代價既然接近零，就取 per-sample：每個 update 看到 `b` 組獨立的位置佈局，而不是 64 個
樣本共用一組，隨機化的多樣性差 64 倍。`rpe_per_sample: False` 可切回論文正文行為。

### `k = 1.0` 時走連續位置的快路徑

課程停在 `k = 1.0` 時 `L_t = n`，位置其實就是 `arange(n)`，但若仍走 RPE 路徑，
freqs 會白白 materialize 成 `[b, n, d]`。實測前 20000 updates（約 5 小時）因此慢約 8%。
`dit.py` 改成 `max_len > seq_len` 才走 RPE，否則用 `forward_from_seq_len` 的 `[1, n, d]`。

## Config

`model.arch`：

| 鍵 | 預設 | 意義 |
|---|---|---|
| `rope_type` | `yarn` | `default` \| `yarn`；`default` 停用以下全部 |
| `yarn_scale` | 2.0 | 訓練期 `s` |
| `yarn_native_ctx` | 3000 | frames（30s），架構應能自行覆蓋的長度 |
| `yarn_alpha` | 1.0 | 轉不到 alpha 圈的維度：完全插值 |
| `yarn_beta` | 32.0 | 轉超過 beta 圈的維度：純外推 |
| `rpe` | `relative` | `off` \| `relative`（`L_t = k · clip 長度`）\| `absolute`（`k · native_ctx`） |
| `rpe_length_scale` | 1.0 | 初始 `k`，由課程推進 |
| `rpe_per_sample` | `True` | 逐樣本獨立抽位置 |
| `logn_ref_len` | 3000 | 熵不變性的參考長度；`null` 停用 |

`optim.rpe_curriculum`：`[[update, k], ...]` 里程碑。

`model.name` 在本分支改為 `WavTTS_Uncond_Large_YaRN`，讓 checkpoint 與 TensorBoard
目錄和 baseline run 分開。兩者**不能互相 resume**——vanilla RoPE 會把
`rotary_embed.inv_freq` 存進 state_dict，YaRN 版註冊為 `persistent=False` 不存，
`load_state_dict` 會直接報 unexpected key。

## 推論

```bash
uv run python src/wavtts/infer/sample_uncond.py \
  --ckpt ckpts/WavTTS_Uncond_Large_YaRN_LibriTTS_100/model_last.pt \
  --duration_sec 120 --yarn_scale 4
```

## 驗證

`tests/test_uncond_smoke.py` 追加（總計 36 個測試，純 CPU、不依賴資料集）：

- YaRN 頻率：最高頻維度不動、最低頻維度整個除以 `s`、中間 ramp 單調；`s=1` 等於原始 RoPE。
- `yarn_attention_factor`：`s=1` 回 1.0。
- 隨機位置：嚴格遞增（順序保留、無重複）、落在範圍內、per-sample 各不相同、
  `per_sample=False` 回 `[1, n]`、無空間可攤時退回連續位置。
- `rpe_max_len` 三種模式與未知模式拒絕。
- YaRN DiT 在 4× native_ctx 的長度上 forward 正常。
- RPE 只在 `training=True` 生效，且每次 forward 位置都重抽。
- `set_yarn_scale` 同時重建頻率與 temperature；`rope_type=default` 時拒絕。
- logn scale 在參考長度上等於 1.0、長於參考 > 1、短於參考 < 1、預設停用；
  且折進 softmax scale 後仍確實改變 DiT 輸出。
- 課程按 update 推進，每個里程碑只觸發一次。
- **`rope_type: default` + `rpe: off` 與原本的 DiT 輸出逐位元相同**。

## 風險與備註

- 兩個機制都改變了模型在**每個**長度下計算的函數，不只是外推時。baseline run 的權重
  無法沿用，必須從零訓練。
- `rpe: relative` 是對論文的領域適配，沒有文獻背書。若語音品質受損，第一個該關的是它
  （`rpe: off`），保留 YaRN 與 logn。
- `logn_ref_len` 與 padding 有交互作用：`attn_mask_enabled: False` 表示 padding frame
  也參與 attention，`n` 取的是 padded 長度。實測 packing efficiency ≈100%（batch 依長度
  排序），影響可忽略。
- 論文是在已預訓練的 LLM 上做 LoRA 微調，我們是從零訓練。從零訓練時 `native_ctx` 不是
  一個既成事實而是一個宣告，`yarn_native_ctx: 3000` 是照資料的最長 utterance 選的。
- 外推效果的驗收方式：用同一組固定 seed 在 30/60/120s 生成，看 `gen/utmos` 與 mel 圖是否
  在超過 30s 後崩壞。若 Randomized YaRN 不足以支撐 120s，備案是 outpainting 滑動窗
  （每一窗把前一窗尾端當已知區、逐步覆寫成解析加噪值），長度無上限但必須循序生成。

## 同期未記錄於前份文件的訓練迴圈改動

以下在 2026-08-19 至 08-20 之間陸續加入，一併補記：

- **響度正規化**（`dataset.py`）：`CustomDataset` 新增 `target_rms: float = 0.1`，
  每條 utterance 載入時正規化到同一 RMS，峰值超過 0.99 則整體縮放（不裁切，保留波形形狀）。
  動機是 no-leaky 混合增強的等功率混合需要兩個來源音量可比，否則一方會蓋掉另一方。
  `target_rms: 0` 停用。
- **EMA 取樣**：checkpoint 取樣改用 `self.ema_model.ema_model` 而非線上權重。線上權重
  每個 batch 都在抖，EMA 才是推論實際會用的；固定 seed 的 clip 也因此在 checkpoint 之間
  可比。`CFM.sample()` 內部本來就會 `self.eval()`，所以取樣期間沒有 dropout 污染。
- **RNG state 存進 checkpoint**：原本 resume 會從 run 起始的 seed 重播 noise/t/mixing 的
  抽樣，而不是接續。現存 python `random` 與 torch CPU/CUDA 三份狀態。刻意不存 numpy——
  這個訓練迴圈沒用到 numpy 隨機，而 `np.random.get_state()` 含 ndarray，會讓現有的
  `torch.load(weights_only=True)` 載入路徑失效。舊 checkpoint 缺這個 key 時照常 resume。
  資料集狀態不需要存：批次順序完全由 `resumable_with_seed + epoch` 決定，resume 再用
  `skip_first_batches` 跳過已消耗的批次，而 `__getitem__` 沒有任何隨機性。
- **記錄間隔**：`ckpts.log_per_updates`（scalar 記錄間隔）；取樣與 metrics 改為跟著
  `last_per_updates`（`model_last.pt`）而非 `save_per_updates`，品質在每個 last-checkpoint
  間隔就看得到。
