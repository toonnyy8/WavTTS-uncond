# WavTTS — 逐 frame 噪聲條件與 prompt training 設計文件

日期：2026-09-15
狀態：**討論中，未定案**。本文記錄到目前為止的決策與待決問題，不是可執行的規格。
條件注入方案（§4）待 wall-clock 量測後定案，§6 的問題尚未討論。
前置文件：
[`context_consistency_guidance.md`](../plans/context_consistency_guidance.md)（`kappa` = 推論期的 context noise level）、
[`wavtts_inversion_replacement_spk_cond.md`](../plans/wavtts_inversion_replacement_spk_cond.md)（training-free replacement）

## 1. 動機

目前這個模型是**完全無條件**的：唯一的條件是 `state`（`STATE_CLEAN` / `STATE_NULL`），
用來做 CFG 的正負分支。沒有文字、沒有 speaker embedding。所有語者條件都發生在**推論期**，
而且全是 training-free 的：

- `infer/spk_cond.py` 用 RePaint 式 replacement——每個 ODE step 之後把參考軌跡釘回序列的
  一半，靠 attention 讓生成的一半讀到參考並繼承語者。
- `context_consistency_guidance.md` 的 `kappa` 進一步讓被釘住的 context 帶噪而非全乾淨
  （`t_ctx = 1 - κ(1-t)`），但同樣只是 sampling 時的 heuristic。
- 下游的 zero-shot source separation（`zsss.run`）已經在手調這個旋鈕：
  `--ref_clean 0.05 --ref_guide 1 --refs enroll --ref_sec 3`。

也就是說，**「context 帶多少噪聲」這件事已經是一個被實際使用、且必須 grid search 的
推論期參數，但模型從未在訓練時見過它**。本設計要把它變成模型原生學過的能力。

目的優先序（使用者確認）：

1. **A — 語者條件**（主）：取代／補強 training-free replacement，讓 prompt 條件成為
   訓練過的能力。直接受益者是上面那條 separation pipeline。
2. **C — 長音訊跨段一致性**（次）：接續前段已生成內容往下長，前段當作帶噪 context。
3. **B — 語音編輯 / infilling**（再次）：給定前後文重新生成中間一段。

## 2. 核心構想

兩件事疊在一起：

**(a) 逐 frame 獨立的噪聲條件。** 目前 `time` 是 per-sample 的純量，整條序列共用同一個
噪聲量級。改為每個 frame 有自己的噪聲量級，`time` 從 `[b]` 變成 `[b, n]`。

**(b) 中間切一段，兩區只差在噪聲量級。** 參照 F5-TTS / E2-TTS 一系的 prompt training，
但不採用它們的硬性二分（prompt 區直接餵乾淨 `x1`、target 區才是 flow 變數）。這裡
**兩區都是 flow 變數，只是噪聲量級不同**——這正是 `kappa` 那個想法的訓練版本。

### 角色不固定（已定案）

`t_in`（中間段）與 `t_out`（外側）**各自獨立抽樣，不強制誰大誰小**。

討論過的另外兩種擺法及其取捨：

- *中間 = 低噪 context、外側 = 高噪 target*：推論時參考音檔可放在序列任意位置、兩側都能
  往外長；每個樣本同時給出「往左長」與「往右長」兩種監督訊號（目前 `spk_cond.py` 只有
  `[ref | gen]` 一種擺法，模型從未學過往左長）；中間段貼齊端點時自動退化成現有用法。
- *中間 = 高噪 target、外側 = 低噪 context*：標準 infilling，直接服務 B，但對 A 而言是
  「參考被切成前後兩半、生成夾在中間」這種少見的推論擺法。

**選定不固定角色**：最通用，A/B/C 全涵蓋，上述兩種都是它的特例。代價是每個 batch 只有
一部分樣本落在「context 明顯比 target 乾淨」這個對 A 最有用的組態上，訓練訊號被稀釋——
若實測發現 A 的收斂太慢，抽樣分布是第一個該調的地方（見 §6.2）。

## 3. 現行實作的相關事實

條件注入路徑（`backbones/dit.py:279-315`）：

```
time [b] ──TimestepEmbedding──> t [b, d] ──+ state_embed[state]──> t [b, d]
                                                                    │
            每個 DiTBlock: attn_norm(x, emb=t) ────────────────────┘
            最後: norm_out(h, t)
```

`t` 是 `[b, d]`，經 `AdaLayerNorm.linear`（`nn.Linear(dim, dim*6)`）產生
`shift/scale/gate`，再用 `scale_msa[:, None]`（`modules.py:113`）broadcast 到所有 token。
**`time` 和 `state` 兩個條件都只走 AdaLN 這一條路，模型裡沒有任何 additive conditioning。**

flow 變數定義在 **waveform domain**（`cfm.py:387-389`）：

```python
t = time.unsqueeze(-1)          # [b, 1]
φ = (1 - t) * x0 + t * x1       # [b, nw]
```

而 frame／token 是 `[b, n]`，`n = nw / hop`。兩個 domain 的落差是 §5 的主題。

## 4. 條件注入方案（待定案）

以 `dim=1152, depth=28, ff_mult=4` 計，每 token 每層的 MACs：

| 項目 | MACs |
|---|---|
| attn 投影（qkv + out） | 5.31M |
| ff（`d→4d→d`） | 10.62M |
| attn matmul（QK^T + AV） | `2·n·1152`（n=600 → 1.38M；n=3000 → 6.9M） |
| **AdaLN linear（`d→6d`）** | **7.96M** |

AdaLN 目前每個 *sample* 只算一次；改成 per-token 就是每個 *token* 算一次，佔總量
**35~46%**（視序列長度）。

| | 機制 | FLOPs 成本 | 表達力 | checkpoint 相容性 |
|---|---|---|---|---|
| **1. Additive per-token** | 每 frame 的噪聲量級過 `SinusPositionEmbedding`+Linear，加到 `input_embed` 之後的 token hidden state；AdaLN 維持 per-sample | ≈0 | 只在輸入端注入一次 | 新增模組（zero-init 則初始等價） |
| **2. Per-token AdaLN** | 每層用該 frame 自己的 shift/scale/gate | +35~46% | 每層重新注入 | **權重形狀完全不變** |
| **3. 不明確條件化** | 只改 `x_t` 的構造，模型自己從統計推斷 | 0 | 兩區 `t` 接近時（0.6 vs 0.7）幾乎分辨不出來 | 不變 |

**方案 3 已排除**：本設計的前提就是「兩區只差在噪聲量級」——沒有其他訊號可供分辨，
那就必須明確告知。

方案 2 的一個關鍵性質：**它是現有機制的嚴格推廣，不是新東西**。程式面只要把
`torch.chunk(emb, 6, dim=1)` 改成 `dim=-1`、拿掉 `scale_msa[:, None]` 與
`gate_msa.unsqueeze(1)` 這幾處 broadcast（`modules.py:111-114, 132-134, 346-359`），
`emb` 傳 `[b, 1, d]` 時結果與現在**逐位元相同**——per-sample AdaLN 是 per-token AdaLN 在
「`t` 沿序列為常數」時的特例。`nn.Linear(dim, dim*6)` 的權重形狀一個都沒動，
現有 checkpoint 可以直接續訓。對這個 repo 大量使用 init-from-pretrained 的做法
（`WavTTS_clean_ola_offset_init460.yaml` 等）這點很值錢。

**待辦**：方案 1 vs 2 的取捨完全落在那 35~46% 是不是真的要付。FLOPs 比例不等於
wall-clock 比例——AdaLN 的 `d→6d` 是規整 GEMM、GPU 效率高，可能被 attention 那些
memory-bound 的 softmax/mask/rope 稀釋；反過來，per-token 後中間 activation 從
`[b, 6d]` 變 `[b, n, 6d]`（n=3000、bf16 下每層約 83 MB），配上開啟的
`checkpoint_activations`，也可能在記憶體頻寬上咬回來。**需實測 forward+backward
wall-clock**，量測時 GPU 必須是乾淨的（2026-09-15 當下有 `zsss.run` 的 separation eval
佔滿唯一一張 4090，量測已排到其後）。

若實測確認成本過高而又需要每層注入，折衷是**只讓前 N 層用 per-token AdaLN**——噪聲量級
是低階資訊，早期層注入通常就夠，成本可調且相容性不變。屬 ad hoc，非必要不採用。

## 5. 技術陷阱：OLA 重疊 frame 的噪聲歸屬

現行的 overlapping 設定是 `wav_frame_len=320, wav_frame_hop=160`：相鄰 frame 重疊 160
samples，**一個 waveform sample 同時被兩個 frame 覆蓋**。而 flow 變數 `x_t` 活在 waveform
domain。若噪聲量級以 frame 為單位定義，重疊區的 sample 會被兩個不同的 `t` 主張所有權——
這不是實作 bug，是兩個 domain 的真實落差。

處理方向（未定案）：噪聲場定義在 **waveform／hop 層級**（每個 hop 一個 `t` 值，無歧義），
token 層級要餵給模型的條件 `time` 則從該 token 覆蓋範圍 pool 下來（平均或取中心）。
因為只有兩個切割邊界，受影響的只有 2 個 token；它們確實橫跨兩種噪聲量級，
把「該 token 的平均 `t`」如實餵進去是誠實的表述，不是近似。

## 6. 待決問題

以下都還沒討論，且與 §4 的方案選擇正交。

**6.1 loss 算在哪一區。** 全部 frame 都算？只算高噪區（原始 prompt training 的做法——
prompt 區是 ground truth，沒東西可學）？還是加權？角色不固定意味著「高噪區」是動態的。

**6.2 `t_in` / `t_out` 的抽樣分布。** 兩者都用現行的 `logistic_normal`
（`P_mean=-0.8, P_std=0.8`）獨立抽？還是其中一個依 `kappa` 由另一個導出（
`t_ctx = 1 - κ(1-t)`，直接對應現有推論參數）？分布形狀會決定 §2 提到的訊號稀釋程度。

**6.3 區段幾何。** 中間段的位置與長度怎麼隨機化（最短／最長比例、是否允許貼齊端點而退化成
`[ctx | gen]`）？是否允許零長度（退化回現行的全域單一 `t`，可作為 A/B 的對照組）？

**6.4 推論路徑。** `sample()` 與 `velocity()` 目前只吃純量 `t`。要讓訓練出來的能力真的
被用到，推論端需要能指定 per-frame 的噪聲排程——這也是取代 `spk_cond.py` 那套 replacement
的接口。

**6.5 既有機制的互動。** `prediction="x_pred"` 的 `_x_to_v`、`loss_space="v"` 的
`(1-time).clamp_min(t_eps)` 都要改成 per-frame broadcast；mixing augmentation 與
`state_null_prob`（clean-only config 下為 0，不啟用）的互動；aux mel loss 作用在整條
waveform 上，與逐 frame 噪聲的關係。

## 7. 與 uncertainty weighting 的合流

同分支上已實作 EDM2 §B.6 / Kendall et al. 2018 的可學習 per-timestep loss 權重
（`CFM.use_uncertainty_loss_weight`）：`uncertainty_net` 吃 `time` 吐 `s(t)`，
loss 變成 `loss·exp(-s) + s`。

目前 `time` 是 `[b]`，所以 `s(t)` 是 per-sample 的。**`time` 一旦變成 `[b, n]`，
`s(t)` 自動變成 per-frame**——每個 frame 依自己的噪聲量級拿到自己的權重。這正是那套
weighting 在逐 frame 設定下該有的樣子，不需要額外設計。實作上 `TimestepEmbedding(dim=1)`
吃 `[b, n]` 吐 `[b, n, 1]`，squeeze 即可。

這也讓 §6.1 有了第三個答案：不是「算或不算」的二分，而是讓 `s(t)` 自己學出各噪聲量級
該有的權重。
