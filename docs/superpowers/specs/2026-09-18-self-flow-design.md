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
音訊用 0.5）：

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

## 移植到 WavTTS 的三個實際問題

### 時間軸方向相反

論文的慣例是 `t=1` 為噪聲、`t=0` 為資料，故教師看的是 `τ_min = min{t,s}`。本專案的
`φ = (1−t)·x₀ + t·x₁` 中 `x₀` 是噪聲、`x₁` 是資料，`t=1` 是*資料*。因此本實作的
教師時間步是 `max(t, s)`。`_dual_timestep()` 的 docstring 記下了這件事，因為它是
唯一一處照抄論文公式就會靜默寫錯的地方——錯了不會報錯，只會讓教師看到比學生更*髒*
的輸入，整個資訊不對稱反向。

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

投影頭 `rep_proj` 是唯一新增的參數：dim=1152 下 10.62M（總參數 664.5M → 675.1M），
對應論文的「約 10M」。它只在訓練期參與，推論時完全不碰。

## 決定與取捨

**時間步分佈不動。** 論文觀察到 Self-Flow 偏好比 baseline 略高的 shift（音訊實驗
0.75 → 1.0），推測是雙時間步加噪把整體 SNR 往平均拉。本專案的 baseline 已經在
shift 1.0（`logistic_normal`, `P_mean=-0.8`, `time_shift=1.0`），而且這一輪的目的是
**乾淨的對照**——只動一個變因。若之後要追絕對品質，這是第一個該掃的旋鈕。

**沒有教師就報錯，不靜默退回。** `self_flow=True` 而 `forward()` 沒收到 teacher 時
丟 `ValueError`。靜默退回 vanilla flow matching 意味著跑一週才發現訓的是別的東西，
而 loss 曲線看起來會完全正常。

**遮罩比例沿用論文的音訊值 0.5。** 論文為音訊掃過 `R_M ∈ {0.05, 0.1, 0.25, 0.5}`
並選出 0.5（影像 0.25、影片 0.1——影片取 0.1 是因為時間冗餘太高）。語音同屬時序模態，
但論文既然是在音訊上選出 0.5 的，就從 0.5 起跑。

## 已知限制

- **舊 checkpoint 無法直接載入。** `rep_proj` 是新參數，`load_checkpoint` 的兩條路徑
  都是 `strict=True`。本輪從頭訓練，不受影響；若要從 `randomized-rope` 的權重暖啟，
  得走 `scripts/make_pretrained_init.py` 那條容許缺漏模組的路。
- **表徵損失飽和得快。** 小模型（24M）在 ~180 個 update 內 cosine 就爬到 0.93。
  這是 BYOL 式目標的常態（EMA 教師 + stop-grad 是標準的防塌陷結構，遮罩保證學生仍有
  實質工作要做），但大模型上值得盯著：若 cosine 很早就貼到 1，表示 `γ` 或 `R_M`
  給錯了，該項已經不提供梯度。
- **只在 LibriTTS clean-100（53.7 h）上驗證過整條迴圈跑得動**，還沒有與 baseline 的
  同步長對照數字。
