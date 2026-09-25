# WavTTS 無條件語音生成 — Self-Flow（雙時間步排程 + 自監督表徵損失）設計文件

日期：2026-09-18
狀態：已實作於 `self-flow` 分支
論文：Chefer, Esser, Lorenz, Podell, Raja, Tong, Torralba, Rombach,
*Self-Supervised Flow Matching for Scalable Multi-Modal Synthesis*, 2026
（Black Forest Labs，<https://bfl.ai/research/self-flow>）
前置文件：[`2026-08-20-length-extrapolation-design.md`](2026-08-20-length-extrapolation-design.md)
（`make_rope()` 的公開化直接來自該文的隨機位置編碼）

## 問題

Flow matching 的訓練目標本身不獎勵語意表徵。均勻加噪之下，去噪多半能靠局部相關性
解決——鄰近樣本點已經給了答案，模型沒有理由去學跨越整段語音的結構。既有補救是
**外部對齊**（REPA 系列）：借一個已經學會表徵的凍結編碼器來對齊中間層特徵。

論文指出外部對齊有兩個實證上的病徵。其一，它不遵守預期的 scaling law：把 DINOv2-B
換成更強的 DINOv3-H+，生成品質反而*更差*——模型被綁在一個與生成目標未必對齊的固定
表徵上。其二，它跨模態不泛化：影片與音訊實驗中，多數外部編碼器（V-JEPA 2、Depth
Anything 3、MERT）相對 vanilla flow matching 是**負收益**。

對本專案而言還有第三點：這是**無條件原始波形**模型。要挑一個「適合」的語音編碼器，
本身就是一個沒有明確答案、且會把外部模型的歸納偏置寫進生成分佈的決定。

## 方法

Self-Flow 不引入任何外部模型，兩個部件都只在訓練期存在。

### 1. 雙時間步排程（Dual-Timestep Scheduling）

每個樣本抽**兩個**時間步 `t, s ~ p(t)`，再抽一個 token 遮罩 `M`（比例 `R_M`，
音訊用 0.5；本實作是以 `mask_block` 個 token 為一塊抽，見〈遮罩的時間尺度〉）：

```
τⁱ = s   if i ∈ M
     t   otherwise

x_τ = diag(1−τ)·x₀ + diag(τ)·x₁
```

輸入因此同時帶著兩個噪聲水準。**這個資訊不對稱就是全部的機制**：一個 token 若其
鄰居比它乾淨，局部去噪救不了它，模型只能去建立全域連結。

論文比較過兩個更直覺的做法，兩者都顯著變差（Fig. 2b）：把一部分 token 設為
`t=1`（完全遮蔽），或每個 token 獨立抽噪聲（diffusion forcing）。原因是
**train–inference gap**——推論時輸入是均勻加噪的，那兩種做法在訓練中幾乎不produce
那個情境。雙時間步保住了 per-token 的邊際時間步分佈，落在兩者之間。

值得注意的是，即使**不加**表徵損失，單獨的雙時間步排程就已經小幅改善生成品質。

### 2. 表徵損失

維護一份 EMA 教師 `f_θ'`（衰減 0.9999）。教師看的是**同一段音訊、同樣的噪聲實現**，
但用兩個時間步中較乾淨的那個**均勻**加噪。學生從自己的淺層特徵預測教師的深層特徵：

```
L_rep = −E cos( h_θ^(l)(x_τ, τ), sg[ f_θ'^(k)(x_τ_clean, τ_clean) ] )
L     = L_gen + γ · L_rep
```

`l = 0.3D`、`k = 0.7D`（本專案 D=28 → 第 8 與第 20 層，恰為論文 ImageNet 的
`l=8, k=20`），`γ = 0.8`，`h_θ` 是一個約 10M 參數的三層 MLP。`l < k` 的方向性來自
擴散模型中語意特徵沿深度演化的既有觀察：讓學生用**部分、被破壞的**輸入去預測教師
用**較乾淨**輸入算出的、**更深**的特徵。

論文的層選擇消融（App. H）顯示這個超參不敏感：`l` 從 8 動到 4 或 12 影響很小。退化
發生在偏離太遠時——教師層太淺則語意尚未成形，學生層太深則干擾生成。

## 移植到 WavTTS 的四個實際問題

### 時間軸方向相反

論文的慣例是 `t=1` 為噪聲、`t=0` 為資料，故教師看的是 `τ_min = min{t,s}`。本專案的
`φ = (1−t)·x₀ + t·x₁` 中 `x₀` 是噪聲、`x₁` 是資料，`t=1` 是*資料*。因此本實作的
教師時間步是 `max(t, s)`。`_dual_timestep()` 的 docstring 記下了這件事，因為這是
照抄論文公式就會靜默寫錯的第一處——錯了不會報錯，只會讓教師看到比學生更*髒*的輸入，
整個資訊不對稱反向。

### 條件必須變成 per-token

原本 adaLN 路徑假設每個樣本一個純量時間步：`AdaLayerNorm` 把 `[b, d]` 的調變參數用
`[:, None]` 攤開到 token 軸，`DiTBlock` 的 gate 也 `unsqueeze(1)`。雙時間步要求
`[b, n, d]`。

改法是讓這些模組在收到 2D 條件時補上 token 軸、收到 3D 時原樣使用，`chunk` 從
`dim=1` 改為 `dim=-1`。純量路徑因此是嚴格的 no-op——
`test_per_token_time_matches_scalar_time_when_uniform` 釘住這個等價性（把同一個純量
時間步展開成 per-token 常數向量，輸出必須逐元素相同）。推論路徑完全不受影響。

`SinusPositionEmbedding` 的 `x.unsqueeze(1)` 同樣改為 `unsqueeze(-1)`：對 1D 輸入
兩者相同，對 2D 輸入才是正確的。

**Token 對齊**：遮罩是在 100 Hz 的 token 上抽的，不是在 16 kHz 的樣本上。
`tau_wav = tau_tok.repeat_interleave(wav_frame_len)` 把每個 token 的噪聲水準攤到它
那 160 個樣本上。在樣本層級抽遮罩會讓噪聲水準在一個 token 內部跳動，而 DiT 是把整個
frame 當一個 token 看的——模型沒有任何管道能表達那種結構。

### 學生與教師必須共用位置

隨機位置編碼每次 forward 重抽一次拉伸倍率（見前置文件）。兩次獨立的 forward 會拿到
**不同的位置幾何**，那樣比對出來的特徵沒有意義。`make_rope()` 因此從 `forward` 內
的分支抽成公開方法，CFM 抽一次、兩個 pass 共用。

這是實作中第二個「不會報錯只會變差」的陷阱，也是為什麼它被寫成一個公開方法而不是
一個註解。

### 遮罩的時間尺度

論文把遮罩寫成 per-token 的 i.i.d. 抽樣。照抄是第三個靜默出錯的地方——因為
**token 的時間長度不同**。論文音訊用 Songbloom AE，每秒 25 個 latent，一個 token
是 40 ms；本專案是 100 Hz 的原始波形 frame，一個 token 是 10 ms。

i.i.d. Bernoulli(`R_M`) 之下，一段連續被遮罩的 token 平均長 `1/(1−R_M)` 個 token。
`R_M = 0.5` 時是 2 個：

| | token 長度 | 平均遮罩段 |
|---|---|---|
| 論文（Songbloom, 25 Hz） | 40 ms | 2 token = **80 ms** |
| 本專案 i.i.d.（100 Hz） | 10 ms | 2 token = **20 ms** |
| 本專案 `mask_block=4` | 10 ms | 8 token = **80 ms** |

同一條公式在這裡給出快 4 倍的噪聲交替。這會直接侵蝕機制本身：整套方法靠的是
「鄰居比我乾淨，所以局部去噪救不了我」，而 20 ms 只有 2–4 個基頻週期——內插就處理掉
了，全域連結的壓力消失。

`R_M` 調不出這件事：i.i.d. 之下遮罩比例與遮罩段長是同一個參數的兩面（`1/(1−R_M)`），
要拉長段落就得同時拉高比例。**只有分塊能把兩者解耦**，這是 `mask_block` 存在的理由。

`mask_block=4` 復原論文的時間尺度，但它更像**下界而非目標**：HuBERT 在語音上遮的是
10 個 frame × 20 ms = **200 ms** 的 span，我們未分塊的 20 ms 只等於它的一個 token。
合理的搜尋區間是 4–20 token（40–200 ms）。

`mask_block=1` 是預設，且與原本的逐 token 抽樣**位元等價**
（`test_mask_block_1_is_the_unblocked_draw` 釘住這點），所以這個旋鈕不會影響舊設定。

#### `mask_run_len`：同一個幾何族，把尺度從比例解開

i.i.d. 抽法的段長**本來就是幾何分布**——`1/(1−R_M)` 就是它的期望值。所以「改成幾何
分布」不會得到新的分布族，真正該做的是把**均值**從 `R_M` 解開。`mask_block` 是用「乘
以 k」達成，代價是邊界卡在 k 的格點上、而且最短段被墊到 k。

`mask_run_len` 直接抽段長：遮罩段 `~ Geom(1/mask_run_len)`，未遮罩段的均值由 `R_M`
導出（`mask_run_len·(1−R_M)/R_M`），所以 per-token 邊際維持不變、`R_M` 仍是主旋鈕
（那是論文掃的量，也是「preserves the per-token marginal」指的量）。

`mask_block=k` ≡ `mask_run_len = k/(1−R_M)`（`R_M=0.5` 時 k=4 → 8.0）。兩者互斥，
同時設會丟 `ValueError`。

實作是 renewal 形式而不是 per-token 鏈：不對稱的兩態鏈需要 sequential scan，抽段長再
攤平則完全向量化（`_mask_tokens`）。三個會靜默出錯的地方：

- **起始狀態要從穩態抽**（`rand < mask_ratio`），不是擲硬幣。幾何分布無記憶，所以在
  index 0 開一段全新的 run 已經在平衡態，邊際才會精確落在 `R_M`；擲硬幣在
  `R_M ≠ 0.5` 時序列開頭會偏。
- **`torch.rand` 會回傳正好 0**，`log(0) = −inf` → `.long()` 是垃圾值 → `cumsum` 變負
  → scatter 越界。小批量的 CPU 測試抓不到，GPU 上跑幾百次才炸。
- **段數取 `K = n_tok`**：段長 ≥1，所以 n 段必定覆蓋 n 個 token，完全免掉覆蓋率分析。
  代價是多一個 `(b, n)` 張量，實測 670 µs 對 `mask_block` 的 30 µs——對照 1.46 s 的
  單步是 0.05%，不值得為此補邊界判斷。

**保留下來的差異是那個下界。** `mask_block=4` 最短段是 4，幾何鏈有 8%（按 token 計）
落在 ≤3。語音自監督的先例兩邊都有：HuBERT / wav2vec 2.0 用固定長度 span（等於有下界），
SpanBERT 則明確測出幾何段長勝過固定長度。目前沒有本專案上的證據可以裁決。

## 成本

教師**就是 trainer 本來就在維護的 checkpoint EMA**。所以代價是每步多一次 forward，
不是多一份 664M 權重。那次 forward 還在第 k 層截斷（`hidden_only=True`），省下
28 層中的 8 層。

代價是 EMA 現在每個 rank 都要有一份：否則非主 rank 算不出表徵損失，梯度會在 group
內不一致。單卡無影響；多卡時每個 rank 多一份 EMA，而那本來就是主 rank 獨有的開銷。

EMA 的更新排程也跟著改：`ema_pytorch` 預設 `update_every=10`（每 10 步才真的動一次），
對 checkpoint 平滑無所謂，對教師則不是論文要的東西。config 的 `ckpts.ema` 把它設回
`beta=0.9999, update_every=1, update_after_step=0`。

顯存：多出來的 forward 讓 `batch_size_per_gpu` 從 3200 降到 2400、
`grad_accumulation_steps` 從 6 升到 8。每次 update 仍是 19200 frames，只是切得更細。
兩張卡時 `grad_accumulation_steps` 是 4（2 × 4 × 2400 同樣是 19200），速度從約
0.55 update/s 變成約 1.1，177 epoch 的預算從約 17 天變成約 10 天。

換卡數續訓有個陷阱。accelerate 的 `AcceleratedScheduler` 每個 update 會把底層
scheduler 走 `num_processes` 步（`scheduler.py` 的 `split_batches=False` 分支），所以
`SequentialLR` 存進 checkpoint 的 milestone 與兩個 `total_iters` 都是「`num_processes`
× update」的單位。照原樣還原就是用舊視野配新步速：單卡存的狀態在雙卡續訓，lr 會在
預定 update 數的一半觸底，後半段以趨近 0 的 lr 訓練。排程是 update 數的純函數，所以
`trainer.py` 在 resume 時改成重建本輪的排程、再快轉到 checkpoint 的 update。

投影頭 `rep_proj` 是唯一新增的參數：dim=1152 下 10.62M（總參數 664.5M → 675.1M），
對應論文的「約 10M」。它只在訓練期參與，推論時完全不碰。

## 決定與取捨

**時間步分佈不動，而且不改成 uniform。** 論文 Fig. 11b 有一組 uniform 勝過
logit-normal 的消融，但正文限定那是 **text-to-image**；音訊實驗（App. B）**全部**跑
logit-normal，最佳設定是 trainshift `α = 1.0`、`R_M = 0.5`。T2I 之所以用 uniform，
App. A.2 自己說了是「to maintain comparability to previous works」，不是調出來的。
論文的收尾建議也是別另外調：「the optimal scheduler for flow matching works well」。

換算本專案的設定：論文的 timeshift 把 logit-normal 的位移寫成 `µ' = µ + log α`；本
實作是 `t = sigmoid(N(P_mean, P_std))`，且時間軸相反（`logit(t_論文) = −logit(t_本專案)`），
所以 `P_mean = −0.8` 對應論文座標的 `µ' = +0.8`，即 **`α ≈ e^0.8 ≈ 2.23`**。
（config 裡的 `time_shift: 1.0` 是另一個獨立旋鈕且預設等於關閉，不是論文的 `α`。）

論文觀察到 Self-Flow 偏好比 baseline **更高**的 shift（音訊 0.75 → 1.0），推測是雙
時間步加噪把整體 SNR 往平均拉、需要更多低 SNR 覆蓋。本專案的 2.23 已經在那一側，
方向無誤。要追絕對品質的話，該掃的是把 `P_mean` 再往負推，而不是換成 uniform；這件事
後來做了，見下面〈把 `P_mean` 往負推到 −1.2〉。

**沒有教師就報錯，不靜默退回。** `self_flow=True` 而 `forward()` 沒收到 teacher 時
丟 `ValueError`。靜默退回 vanilla flow matching 意味著跑一週才發現訓的是別的東西，
而 loss 曲線看起來會完全正常。

**遮罩比例沿用論文的音訊值 0.5，用 `mask_block` 而不是 `R_M` 去對齊時間尺度。**
論文為音訊掃過 `R_M ∈ {0.05, 0.1, 0.25, 0.5}` 並選出 0.5（影像 0.25、影片 0.1——影片
取 0.1 是因為時間冗餘太高）。語音同屬時序模態，且論文既然是在音訊上選出 0.5 的，就
從 0.5 起跑；時間尺度的差異交給 `mask_block` 處理，理由見〈遮罩的時間尺度〉。

**`mask_block=4` 勝出。** clean-460 上兩條線各跑到 update 55000 的同步比較下，
`mask_block=4` 優於 `mask_block=1`，兩者的 checkpoint 都保留著。這份文件不記錄評估
指標，因為判讀是在 repo 之外做的。這個結果也把前面「4 比較像下界」的推測降級成尚未
驗證——已知的只有 4 > 1，4 與 10/20 還沒比過。

**把 `P_mean` 往負推到 −1.2。** 對應 `α ≈ e^1.2 ≈ 3.32`，比原本的 2.23 更深入低
SNR：中位數 `t` 從 0.310 降到 0.232、q25 從 0.208 降到 0.149、`t < 0.1` 的比例從 4%
升到 11%。上面警告過這會和 `mask_block` 的對照混成兩個變因，避開的方式是把平均遮罩段
長釘在 80 ms——這條線用 `mask_run_len: 8`，而 `8 = 4/(1−0.5)`，與 `mask_block: 4` 的
平均段長相同（實際抽樣量到 7.97 token，邊際 0.4998）。沒被釘住的是段長分布的形狀
（格點與下界消失），所以這條線對 MB4 的差異是「`P_mean` + 段長律的形狀」，而不是單獨
的 `P_mean`。

**負向條件與引導在 2026-09-26 從程式移除，所以有了第四條線。**
`WavTTS_selfflow_rl8_pm12_nocfg.yaml`（`..._SelfFlow_RL8_PM12_NoCFG_LibriTTS_460`）沿用
RL8_PM12 的 self-flow 旋鈕，但模型已經沒有 state embedding、沒有混合增強、也沒有
`cfg_strength`——唯一的條件輸入是 timestep。它從零訓練，因為前三條的 checkpoint 都帶
`state_embed.weight`、在新程式下載不進來（詳見 CFG 設計文件的〈移除紀錄〉）。這條線與
前三條之間因此不只差一個旋鈕，而是差一個架構，**不能當成 RL8_PM12 的延續來讀**。

**前三條對照線的配置。** `WavTTS_selfflow_mb1.yaml`（`mask_block: 1`，ckpt 目錄
`WavTTS_Uncond_Large_SelfFlow_LibriTTS_460`）與 `WavTTS_selfflow.yaml`
（`mask_block: 4`，`..._SelfFlow_MB4_LibriTTS_460`）除了這一個參數之外完全相同。
`WavTTS_selfflow_rl8_pm12.yaml`（`mask_run_len: 8` + `P_mean: −1.2`，
`..._SelfFlow_RL8_PM12_LibriTTS_460`）是第三條，而它不是從頭訓練：MB4 在 update
92500 的 `model_last.pt` 被複製進這條線自己的 ckpt 目錄後接著跑，所以 92500 之前是
共用歷史，之後才分叉。複製而不是 hardlink，否則新線存檔會把 MB4 的檔案一起截斷。
分成三份 config 而不是就地改一份，是因為共用一份檔案時各條線無法各自啟停，而且 ckpt
目錄由 `model.name` 決定——名字沒改就會靜默續訓到另一條線的權重上。

## 已知限制

- **舊 checkpoint 無法直接載入。** `rep_proj` 是新參數，`load_checkpoint` 的兩條路徑
  都是 `strict=True`。本輪從頭訓練，不受影響；若要從 `randomized-rope` 的權重暖啟，
  得走 `scripts/make_pretrained_init.py` 那條容許缺漏模組的路。
- **表徵損失飽和得快。** 小模型（24M）在 ~180 個 update 內 cosine 就爬到 0.93；
  664M 模型在 clean-460 上跑到 5 萬多個 update 時 cosine 在 0.91–0.94 徘徊（兩條線
  皆然）；RL8_PM12 跑到 40 萬 update 後均值仍在那個區間，只是單步的散佈更寬——update
  35 萬之後的 105k 個值均值 0.938、範圍 0.85–0.98。這是 BYOL 式目標的常態（EMA 教師
  + stop-grad 是標準的防塌陷結構，遮罩保證學生仍有實質工作要做）。論文沒有給健康的
  cosine 區間，也只用 FID 消融來判斷，所以這個數字本身不構成結論——真正該看的是生成
  品質，而那還沒有對照數字。
- **還沒有與 vanilla flow matching 的同步長對照數字。** 跑過的是 LibriTTS
  clean-100（53.7 h）的迴圈驗證，以及 clean-460（244.6 h、149510 clips）上
  `mask_block` 1 對 4 的兩條線（4 勝）；缺的是 `self_flow: False` 的那一條，所以
  Self-Flow 本身相對於 vanilla 的增益在本專案上仍未驗證。
- **`mask_run_len` 跑過了，但沒有評估數字。** `WavTTS_selfflow_rl8_pm12.yaml` 從 MB4
  的 update 92500 接到 402500（Epoch 72/177）後停下，177 epoch 的預算沒有跑完。實際
  抽樣驗過邊際 0.4998（目標 0.5）與平均遮罩段 7.97 token = 80 ms（目標 8），但生成
  品質還沒和 MB4 對照過。要注意 `flow_loss` 的絕對值在這兩條線之間不可比——`t` 更小
  使 v-target 更難，同一個 fork 點上 RL8_PM12 讀 0.4–0.75、MB4 讀 0.19–0.32——能比的
  是固定 seed 的取樣與 mel 指標。而且這條線同時動了 `P_mean`，所以就算有了數字，也分
  不出 `mask_run_len` 與 `P_mean` 各自的貢獻。
