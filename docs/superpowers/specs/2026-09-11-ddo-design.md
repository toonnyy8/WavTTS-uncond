# WavTTS 無條件語音生成 — Direct Discriminative Optimization (DDO) 微調設計文件

日期：2026-09-11
狀態：設計，尚未實作
實作分支：`ddo`（切自 `randomized-rope` @ `746cc17`）
論文：Zheng et al., *Direct Discriminative Optimization: Your Likelihood-Based Visual
Generative Model is Secretly a GAN Discriminator*（[arXiv:2503.01103](https://arxiv.org/abs/2503.01103)，
ICML 2025 oral），官方實作 [NVlabs/DDO](https://github.com/NVlabs/DDO)
前置文件：
[`2026-08-19-uncond-speech-cfg-design.md`](2026-08-19-uncond-speech-cfg-design.md)、
[`2026-08-20-length-extrapolation-design.md`](2026-08-20-length-extrapolation-design.md)

> **關於「官方實作」的範圍，先講清楚。** NVlabs/DDO 整個 repo 只有 VAR（自迴歸）的訓練碼
> 加兩支取樣腳本。**擴散版的訓練碼不存在，也從來沒有公開過**（issue #1，作者：EDM/EDM2
> 那份需要更久的重構）。擴散版唯一的權威來源是論文 v2/v3 的 **Appendix D 程式碼片段**，
> 全文轉錄於本文 §附錄 A。本設計的每一處「論文怎麼做」都指向那段程式碼或 Appendix C 的
> 超參表；凡是 VAR-only 的做法會標明。

---

## 1. 問題

本 repo 的模型用 flow matching 做最大概似訓練。MLE 最小化的是 forward KL
`D_KL(p_data ‖ p_θ)`，這個方向對「模型在資料沒有質量的地方放質量」幾乎不罰——
`p_data(x)=0` 的 `x` 對積分沒有貢獻。於是有限容量下它會 **mode-covering**：把質量抹開去
蓋住所有模態，寧可多生一些不像語音的東西，也不願漏掉任何一種語音。

在這個模型上，那正是聽得到的失效樣態：低 CFG 時含糊、語者游移、非語音的嗡鳴；要壓下去
只能把 `cfg_strength` 拉高，而 CFG 是推論期的補救，不是訓練期的修正，代價是多一倍 NFE
與過飽和。

DDO 提供的是訓練期的修正：**不動架構、不加判別器網路、不做 GAN 的交替訓練**，
用一個凍結的參考模型把目前模型自己變成判別器，再用判別式目標把它往 reverse KL
的方向推一段。論文在 CIFAR-10 / ImageNet 上把 FID 砍掉一半以上，且不需要推論期 guidance。

對本 repo 的吸引力在於它與現有結構的相容性：目標函數只吃「兩個模型在同一個
`(x, t, ε)` 上的 flow loss 之差」，而那正是 `CFM.forward` 已經在算的量。

---

## 2. DDO 的形式

### 2.1 判別器的重參數化

GAN 的判別器最優解是 `d*(x) = p_data(x) / (p_data(x) + p_g(x))`。DDO 反過來用：
若拿一個**凍結的參考模型** `p_ref`（生成器）與**可訓練的目標模型** `p_θ` 組出

```
d_θ(x) := σ( log p_θ(x) / p_ref(x) )                                    (Eq. 8)
```

則把 `d_θ` 丟進標準的 GAN 判別器目標，就得到一個**只對 `θ` 做最小化**的單一損失：

```
min_θ  L(θ) = − E_{x∼p_data}[ log σ( log p_θ(x)/p_ref(x) ) ]
              − E_{x∼p_ref} [ log (1 − σ( log p_θ(x)/p_ref(x) )) ]      (Eq. 9)
```

Theorem 3.1：容量無限時最優解 `p_θ* = p_data`。沒有交替訓練、沒有第二組參數、
沒有 GAN 的不穩定性——參考模型永遠不更新。

實務版本加兩個超參（Eq. 13）：

```
L_{α,β}(θ) = − E_{p_data}[ log σ( β·r_θ(x) ) ]
             − α · E_{p_ref} [ log σ( −β·r_θ(x) ) ] ,   r_θ(x) := log p_θ(x) − log p_ref(x)
```

**`β` 不只是數值穩定的縮放。** Theorem 3.3 說最優解是
`p_θ* ∝ p_ref^{1−1/β} · p_data^{1/β}`：`β < 1` 會**故意衝過 `p_data`**，沿
`p_data/p_ref` 的方向外插。`β = 0.02` 的指數是 50——那是極強的銳化。
這同時解釋了兩件事：為什麼 DDO 的效果像「內建 guidance」，以及**為什麼一輪必須很短**
（見 §2.3）。`α` 則調整「壓低假樣本」相對於「抬高真樣本」的權重。

論文給的典型範圍是 `α ∈ [0.5, 50]`、`β ∈ [0.01, 0.1]`——**那是在他們的 Δ 定義下的值，
本 repo 不能照抄**，見 §3.2、§3.3。

### 2.2 擴散/流模型的代理量

`log p_θ(x)` 對擴散模型算不動，論文改用 ELBO 的差當代理（Eq. 14–15）：

```
log p_θ(x)/p_ref(x)  ≈  E_{t,ε}[ Δ_{x_t,t,ε} ] ,
Δ_{x_t,t,ε} = − w(t) ( ‖ε_θ(x_t,t) − ε‖²  −  ‖ε_ref(x_t,t) − ε‖² )
```

再用 Jensen 把期望值搬到 `log σ` 外面（Eq. 16），換來**每個樣本只要一次前向**：

```
L(θ) ≤ − E_{t,ε}[ E_{p_data} log σ(βΔ) + α · E_{p_ref} log σ(−βΔ) ]
```

Appendix D 的實作（§附錄 A）證實三件關鍵細節：

1. `w(t)` 就是 EDM 預訓練用的加權，**但丟掉 EDM2 學習式的 `logvar` 不確定度加權**。
2. `Δ` 是 **`torch.sum` over C,H,W，沒有任何維度正規化**。
3. **`t` 與 `ε` 抽一次，真假兩批共用**（common random numbers）。
   `θ` 與 `ref` 當然也共用——那是變異數縮減的主要來源，缺了它 Δ 就是兩份獨立雜訊相減。

### 2.3 多輪 self-play，而且輪數比直覺多得多

一輪跑完後，把**依評估指標挑出的最佳** `θ*_n`（不是最後一個 checkpoint）冷凍成下一輪的
`p_ref`，重新用它生一批假樣本，再跑一輪。

實際輪數（Appendix C）：

| 模型 | 輪數 | 每輪長度 | 佔預訓練 |
|---|---|---|---|
| EDM CIFAR-10 uncond / cond | **12 / 16** | 1.5M images = 30 epochs | 0.75% |
| EDM2-S ImageNet-64 | **24** | 6.4M images = 5 epochs | 0.6% |
| EDM2-L ImageNet-512 | **28** | 6.4M images = 5 epochs | 0.34% |
| VAR-d16 / d30 ImageNet-256 | **2** | ~80–98 optimizer steps | < 0.03% |

**擴散模型要 12–28 輪**，不是兩三輪。每輪很短、輪數很多，這是方法的形狀，不是可調的偏好。

而且論文 §3.3 明說：*"the optimization process of DDO provides useful gradient information
in the early stage but does not converge to the data distribution in the final."*
**輪內不收斂是設計使然**：品質會在輪中觸底然後回頭變差。因此
（a）checkpoint 與評估要密，（b）**挑最後一個 checkpoint 是錯的**，
（c）每輪都要重掃 `α`/`β`（論文每輪掃 ~20 個節點）。

輪與輪之間**不承接 optimizer state**（官方 VAR 腳本直接把 resume 關掉）。

---

## 3. 移植到本 repo

### 3.1 Δ 用 repo 自己的 flow loss

本 repo 的插值是 rectified flow `x_t = (1−t)x₀ + t·x₁`，模型出 `x_pred`
（`prediction: x_pred`），損失算在 v 空間（`loss_space: v`）：

```
v̂ = (x̂ − x_t)/(1−t) ,  目標 = (x₁ − x_t)/(1−t) ,  ℓ = ‖v̂ − 目標‖² = ‖x̂ − x₁‖² / (1−t)²
```

嚴格說，rectified flow 的 MSE 只在特定加權下等於 ELBO（Kingma & Gao 2023）。
本設計**不去重建那個加權**，而是直接令

```
Δ(x; t, x₀) := − ( ℓ_θ(x; t, x₀) − ℓ_ref(x; t, x₀) )
```

其中 `ℓ` 就是 `CFM.forward` 現行的 per-sample 損失（依 `loss_space` 決定空間，
v 空間即隱含 `w(t) ∝ 1/(1−t)²`）。理由：

- **那是模型實際被擬合的目標。** 用它當代理，`θ = θ_ref` 時 `Δ ≡ 0` 恰好成立，
  訓練起點在 sigmoid 的線性區正中央，`L = (1+α)·log 2`。這是實作的解析檢查點。
- `w(t)` 的任何常數倍都被 `β` 吸收，而 `β` 本來就要 grid search。留下來的只有
  `w(t)` 的**形狀**，而 v 空間的形狀正是預訓練用的形狀。
- 論文自己也是這樣選的：EDM 的 `weight·(D−x)²` 恰等於 `‖F_θ − F̂‖²`，
  也就是預訓練損失本身。**論文明確警告**：換了 preconditioning 就要重推 `w(t)`，
  否則 `β` 的標定範圍毫無意義。我們的做法（直接用預訓練損失）正是避開這個坑的方式。

**aux mel loss 不進 Δ。** 它不是 ELBO 的一部分，是感知正則項。它只留在 §3.4 的 anchor 項。

### 3.2 Δ 的長度正規化：取 mean 而非 sum（**明知故犯的偏離**）

論文的 Δ 是 `torch.sum` over 所有維度，且**明文反對**改成平均：
*"Do not normalize the summed MSE by dimension instead — the reported α/β ranges assume an
unnormalized sum."* 它也說 `β` 應該大致隨資料維度 `1/D` 縮放。

本 repo 不能照做，原因是**維度不固定**。訓練資料長度跨 0.3–30 s，
即 48 到 4768 個有效 token、維度差 **100 倍**。`β` 是單一全域純量：
在 5 s 片段上調到 `βΔ ~ O(1)` 的 `β`，到 30 s 片段上就是 `βΔ ~ O(100)`，
sigmoid 直接飽和、梯度歸零；反過來調則短片段完全沒有訊號。
更糟的是**長度本身會變成判別器唯一需要的特徵**。

所以：**`Δ` 對每個樣本的有效元素取平均**（`delta_normalize: mean`，預設）。

這其實不違反論文的精神，而是它的自然延伸：論文說 `β ∝ 1/D`，而我們的 `D` 逐樣本不同，
**取平均恰好就是讓 `β` 逐樣本自適應 `β_eff = β/D`**。代價是丟掉「長樣本的 log-ratio
本來就該更大」這個外延性；換來的是一個 `β` 能同時適用整個長度分佈。

`delta_normalize: sum` 保留為 ablation，但在 `batch_size_type: frame` 下不建議使用。
padding 必須排除——`lens_to_mask` 給的 mask 已經在 `CFM.forward` 裡了，沿用即可。

### 3.3 β 必須重新標定，不能抄論文

論文的 `β ∈ [0.01, 0.1]` 對應的是**對 3072 維求和**、量級 O(10³) 的 Δ。
本設計的 Δ 是 per-element 平均、量級 O(10⁻³)～O(10⁻²)。**兩者差 5–6 個數量級，
直接照抄 β 等於把 DDO 項關掉。**

標定流程（寫進 plan 的 Task）：

1. `β` 設 1.0 跑 300 updates，記錄 `ddo/delta_real`、`ddo/delta_fake` 與
   `ddo/delta_std`（batch 內 Δ 的標準差）。
2. 取 `β₀ ≈ 1 / delta_std`，使 `βΔ` 落在 O(1)。
3. 用 `ddo/acc`（判別器準確率）微調：**目標 0.6–0.75**。
   - `acc > 0.9` 且很快到達 → `β` 太大，sigmoid 飽和，梯度消失。
   - `acc ≈ 0.5` 且 `delta_std` 持續變大 → `β` 太小，DDO 項沒在起作用。

起點在 `θ = θ_ref`，Δ 恆為 0、`delta_std = 0`，所以第 1 步必須**跑起來之後**才讀數。

**梯度尺度的正規化。** 官方 VAR 實作在最後做 `loss = loss / max(alpha, 1.0)`，
讓梯度量級不隨 `α` 變動，於是掃 `α` 時不必重調 LR。本設計照抄，理由相同。

### 3.4 CFG 幾何的保護：null 分支用 MLE anchor（與論文一致）

模型的唯一條件是 `clean` / `null` 兩值 state，推論靠 `v = v_clean + w·(v_clean − v_null)`。
`null` 分支承載混合增強，它定義了「要往哪裡推開」。

論文對 CFG 模型的處理（VAR 實作、issue #3 作者說明）是：`p_θ`、`p_ref` 都取
guidance-free 模型；**無條件分支的假樣本項權重設 0，改用 MLE 訓練它**。作者的理由是
*"the unconditional part itself in CFG serves as a negative signal. Compared to the
conditional part, it should be more mode-covering, and training with MLE is a stable choice."*

這正是本 repo 需要的：`null` 分支**應該**保持 mode-covering，它是負分佈的模型，
銳化它等於把 CFG 的減項變窄，guidance 會指向錯的地方。加上 `clean` 與 `null` 共用整個
trunk，`null` 若無人監督就會隨 trunk 漂移——clean 分支被銳化、null 分支爛掉，
`w=2` 的 guidance 指向垃圾。

所以 batch 依照預訓練的比例拆三份：

| 列 | 來源 | state | 目標 |
|---|---|---|---|
| real-positive | 真實語料，`1 − state_null_prob` | `clean` | DDO 正項 `−log σ(βΔ)` |
| anchor | 真實語料，`state_null_prob`（含混合增強） | `null` | 原本的 CFM flow loss ×`anchor_weight` |
| fake-negative | `p_ref` 生成的假樣本池 | `clean` | DDO 負項 `−α·log σ(−βΔ)` |

`anchor_weight` 預設 1.0，也就是與預訓練完全相同的權重——調的是 `β`，不是它。
**假樣本永遠是 `clean`、永遠不過混合增強**。

若拿 `state_null_prob: 0` 的 clean arm（`WavTTS_clean.yaml`）做 DDO，就沒有 anchor 列。
**第一輪建議先跑這個 arm**：少一個 CFG 幾何的干擾變數。

### 3.5 假樣本：離線生成，且必須與真實資料統計對齊

線上生成不可行：每個假樣本要 32 次 NFE 的 664M 模型前向，比訓練本身貴一個數量級。
（VAR 能線上生是因為它的取樣只有 10 個 scale；論文的擴散版本也是**離線**：
每輪離線生 50k 張。）**離線生成一個池子**，重複使用。

**真假比例。** 論文的擴散設定是真假 **1:1 配對**：50k 假樣本對 50k 張 CIFAR，
每個 batch 一半真一半假。本 repo 的假樣本池不可能做到 244.6 h，所以
**比例由取樣強制，不由資料集大小決定**：`TaggedConcatDataset` 依兩邊的總 frame 數
重複假樣本池的索引，讓每個 batch 的真假 frame 數接近 `ddo.real_fake_ratio`（預設 1.0）。
重複的只有波形，每次抽到配的 `(t, ε)`、frame offset 都是新的。

**共用隨機數。** 論文真假共用同一組 `(t, ε)`。本 repo 做不到完整共用——真假兩列長度不同，
`ε` 的形狀就不同。可做的是**共用 `t`**（純量，恆可配對）：batch 內真假列按索引配對重用
同一批 `t` 抽樣。`ε` 的配對放棄，並記錄為與論文的偏離。
（真正關鍵的配對是 `θ` 與 `ref` 之間的，那個我們完整保留。）

離線化引入一個論文沒有的風險：判別器可以靠**任何**能分開真假的捷徑把 Δ 拉開，
而那些捷徑對生成品質毫無幫助。本 repo 至少有四條：

1. **長度分佈。** 假樣本長度必須從真實資料的 `duration.json` 經驗分佈抽樣，
   不能用固定長度、不能用均勻分佈。
2. **音量。** 真實資料在 `CustomDataset.__getitem__` 被 RMS 正規化到 `target_rms`；
   假樣本從模型出來是模型自己的量級。
3. **frame grid。** 真實資料有 `rand_frame_offset` 的次 frame 抖動；假樣本是在 grid 上生成的。
4. **存檔精度。** 模型輸出超出 ±1，存成 16-bit PCM 會削頂——那是真實語料絕對沒有的特徵。
   存 `PCM_F`（32-bit float WAV）。

第 2–4 點的統一解法是：假樣本池的**目錄格式與真實資料集完全相同**
（`data/<name>/raw` + `duration.json`），用同一個 `CustomDataset`、同一組
`waveform_kwargs` 載入。這不是方便，是正確性。

（對照論文：VAR 版把生成的 latent 丟掉、解碼成 pixel 再重新編碼，就為了讓真假樣本
待在同一個「pixel-encoded space」裡，作者說實測更好——issue #5。同一個顧慮，
我們的版本是讓真假走同一條載入管線。）

生成設定：

- 權重用 `p_ref` 的 **EMA**——必須與 Δ 裡的 ref 是同一組權重，否則「假樣本來自 `p_ref`」
  這個前提不成立。
- `cfg_strength = 0`。論文取 guidance-free 模型當 `p_ref`/`p_θ`。
  （註：官方 VAR 腳本用 `cfg=1.0` 並稱之為 guidance-free，但 VAR 的 `t = cfg·ratio`
  公式下那其實是逐 scale 由 0 爬到 1.0 的弱 guidance——論文與程式碼在此不一致。
  我們採理論上正確的 0，並保留 `fake_cfg_strength` 旋鈕做 ablation。）
- `steps=32`、`solver=euler`、`sway_sampling_coef=-1.0`，與 trainer 內建取樣一致。
  32 步的 ODE 解不等於 `p_ref` 的精確樣本，這是論文同樣接受的近似。

池子大小見 §5。

### 3.6 兩個模型必須看到同一組 RoPE 位置（關鍵陷阱）

`DiT.forward` 在訓練模式下會抽隨機化位置：

```python
if self.training and self.rpe_gamma > 1.0:
    rope = self.rotary_embed(randomized_positions(h.shape[0], seq_len, self.rpe_gamma, h.device))
```

`randomized_positions` 每次呼叫都從全域 RNG 抽新的 `randperm`。θ 前向一次、ref 前向一次
＝ **兩組不同的位置**，於是 `Δ = ℓ_θ(positions A) − ℓ_ref(positions B)` 變成
「同一個模型在兩組不同位置上的損失差」，純雜訊，DDO 訊號完全被蓋掉。而且它**不會報錯**，
只會表現成訓練沒有效果。

把 ref 設成 `eval()` 也不對：那樣 ref 吃連續位置、θ 吃隨機位置，一樣不同，
而且偏差是系統性的（Δ 恆偏向一邊）。

解法：**在 `DiT.forward` 加一個 optional `positions` 參數**，給了就直接用、無視
`self.training`。DDO 路徑抽一次位置，兩個模型都吃它。

這就是論文那條「共用 `(t, ε)`」的規則在本 repo 的第三個分量。同理，論文**關掉所有
dropout**（*"to ensure steady improvement"*，因為 dropout 會讓同一個資料點的 `log p_θ`
每次前向都不同，而 `log p_θref` 是在另一組 mask 下算的）——本 repo 所有 config 的
`dropout` 已經是 0.0，這條自動滿足，但不得在 DDO 的 config 裡把它打開。

`rpe_gamma` 在 DDO 微調期間**保持開啟**——它是訓練期增強，關掉等於只在連續位置上微調，
會侵蝕長度外推能力。假樣本則是用連續位置（推論設定）生成的，那沒有問題：它們是資料。

`checkpoint_activations` 對 ref 應該關掉——ref 跑在 `no_grad` 下，重算 activation 純屬浪費。

### 3.7 數值精度：論文說整個 loss 都要 fp32

論文 Appendix C 的原話：*"We also find numerical precision crucial for the diffusion DDO
loss and **disable mixed-precision training**."*（VAR 版反而保留 fp16，因為 AR 的
log-softmax 比「加權 MSE 的差」條件數好太多。）

原因是災難性抵消：Δ 是**兩個幾乎相同的模型**的損失之差，微調初期相對差可能是 10⁻³ 量級，
bf16 只有 8 bit 尾數，這個減法會把有效位數吃光，Δ 退化成量化雜訊。

本 repo 的完整 fp32 不可行：README 記載 fp32 在 24 GB 卡上只塞得下 2400 frames/GPU
（bf16 是 3200，且 clean-460 arm 跑到 4800）。所以：

- **硬性要求**：`v_pred` / `x_pred` 出爐後先 `.float()`，平方誤差、mask 平均、相減、
  `logsigmoid` 全部在 fp32。`F.logsigmoid(βΔ)` 與 `F.logsigmoid(−βΔ)`，
  **不得**寫成 `log(1 − sigmoid(x))`。
- **已知偏離**：trunk 仍跑 bf16 autocast。若標定階段看到 `ddo/delta_std` 呈現量化階梯、
  或 Δ 的直方圖離散化，**第一件要試的事就是整輪改 fp32**
  （`accelerate launch` 拿掉 `--mixed_precision bf16`，`batch_size_per_gpu` 減半、
  `grad_accumulation_steps` 加倍以維持更新尺寸）。這條記在 §6 的風險表裡。

### 3.8 EMA 的半衰期必須縮短（否則整輪的學習被平均掉）

論文對擴散版特別點名 EMA 有助於穩定，而且用的是**很短**的平均長度：
EDM CIFAR 傳統 EMA 半衰期 **0.25M images**（bs 512 → 約 500 步）；
EDM2 的 power-function EMA **length 0.05**。

本 repo 的 `EMA(model, include_online_model=False)` 吃 `ema_pytorch` 的預設值
（`beta=0.9999`、`update_every=10`），有效平均長度約 **100k updates**。
一輪 DDO 只有 5–9k updates——**EMA 會把整輪的改變幾乎完全平均掉**，
而 trainer 的取樣與指標全部讀 EMA 權重，於是曲線會平得像沒訓練。

所以 DDO 的 config 必須顯式給 `ema_kwargs`，目標有效平均長度落在輪長的 5–10%
（例如 `beta: 0.999`、`update_every: 1` → 約 1000 updates）。
`make_pretrained_init.py` 已經會把 EMA 的 `step` 歸零，暖機斜坡會重新開始。

### 3.9 參考模型的擺放

ref 不能出現在 `nn.Module` 的樹裡，否則：optimizer 會收它的參數、EMA 會平均它、
DDP 會同步它的梯度、`state_dict()` 會把 checkpoint 撐成兩倍、
`make_pretrained_init.py` 與 `sample_uncond.py` 的 key 全部對不上。

作法：存在**單元素 list** 裡（`self._ddo_ref = [ref]`），`nn.Module.__setattr__`
不會登記它。代價是 `.to(device)` 不會傳遞，由 trainer 明確搬一次。

同理，DDO 的前向必須走 `self.model(...)`（DDP 包好的那個 `__call__`），不能繞過去呼叫
unwrapped model 的方法，否則多卡梯度不同步。所以 **DDO 路徑做在 `CFM.forward` 裡面**，
以 `is_fake` 旗標分流，而不是另做一個 wrapper module。這樣 state dict 的 key 一個都不變。

---

## 4. 訓練配置

新增 config 區塊 `ddo:`（缺席即為現行行為，完全向後相容）：

```yaml
ddo:
  ref_ckpt: ckpts/<pretrained_run>/model_last.pt  # 凍結參考模型；讀 EMA 權重
  fake_dataset: LibriTTS_460_fake_r1              # data/<name>，gen_fake_pool.py 產出
  alpha: 1.0              # 假樣本項權重；論文擴散版掃 [0.5, 6.0]
  beta: 1.0               # log-ratio 縮放；**必須依 §3.3 標定**，勿抄論文的 0.01–0.1
  delta_normalize: mean   # mean | sum，見 §3.2
  anchor_weight: 1.0      # null 列保留的 CFM flow loss 權重
  real_fake_ratio: 1.0    # 每個 batch 的真:假 frame 比；論文擴散版是 1:1
  fake_cfg_strength: 0.0  # 記錄用；實際值在生成池子時決定
```

`optim` / `ckpts` 的改動（DDO 是短程微調，預訓練的排程完全不適用）：

| 鍵 | 預訓練 | DDO | 理由 |
|---|---|---|---|
| `learning_rate` | 7.5e-5 | **1e-5**（掃 1e-5 … 5e-5） | 論文擴散版在 bs 512 下用 5e-5（EDM2-S）到 1.5e-4（CIFAR）；本 repo 的更新是 19200 frames，先保守 |
| `num_warmup_updates` | 20000 | **200** | 20000 步暖機在一個 ~9000 步的輪次裡永遠走不完 |
| `max_updates` | —（新增） | **~9000** | 明確的輪次長度（≈1% 預訓練）；到了就停 |
| `ema_kwargs` | 預設（~100k updates） | **`beta: 0.999, update_every: 1`** | §3.8；不改的話整輪的學習會被平均掉 |
| `save_per_updates` / `last_per_updates` | 10000 / 2500 | **1000 / 500** | 品質會在輪中觸底再回頭變差，挑點要密 |
| `grad_accumulation_steps` | 視卡數 | 維持 19200 frames/update | 不動更新尺寸，變因只有目標函數 |

`max_updates` 是新增的 trainer 選項：現行排程是 warmup + 線性衰減到
`len(dataloader)×epochs`，DDO 輪次太短、而且 dataloader 因為併入假樣本池而變長，
用 `epochs` 湊步數會湊不準。

**後期輪次的 `P_std`。** 論文在晚期輪次把 `P_std` 調寬
（EDM2-S：1.6 → 3.0 於第 17 輪；EDM2-L：1.6 → 2.0 → 3.0），讓訊號覆蓋更極端的雜訊水平。
本 repo 的 `P_std` 是 0.8。這是唯一有紀錄的輪間排程變化，第 1 輪不動，之後視情況調。

### 監看指標

既有的 `gen/utmos`、`gen/spk_sim_self`、`gen/silence_ratio`、`gen/rms` 全部留著，
它們是這裡的 FID 替身。新增（對應官方 `DDO_trainer.py` 記的那組）：

- `ddo/delta_real`、`ddo/delta_fake`：兩側 Δ 的均值。健康的曲線是兩者緩慢分開。
- `ddo/margin = delta_real − delta_fake`：官方記的量。
- `ddo/delta_std`：batch 內 Δ 的標準差，`β` 標定讀的就是它。
- `ddo/acc`：`mean(Δ_real > 0)/2 + mean(Δ_fake < 0)/2`，目標 0.6–0.75。
- `ddo/loss_real`、`ddo/loss_fake`、`anchor_loss`：三項分開記，才知道誰在主導。
- `flow_loss`（真實列）：**這是發散警報，不是品質指標**。DDO 本來就是拿 likelihood
  換品質，它上升是預期行為；漲超過預訓練值的 ~2× 就是這一輪推過頭了。

作者對「訓練正不正常」的判準只有一句（issue #4）：*"As long as the FID is decreasing,
the training is normal."* 本 repo 的對應物是 `gen/utmos` 與 `gen/spk_sim_self`。

---

## 5. 成本估算

以 clean-460 arm（4×RTX 4090，19200 frames/update，~0.35 s/update）為基準，
以下是量級估算，不是量測值：

**每次 update。** 預訓練 ≈ 1 前向 + 1 反向 ≈ 3 單位。DDO 每列多一次 ref 前向，
且 batch 同時含真假兩側，≈ `2 × (3 + 1) = 8` 單位 → **約 2.7×**，即 ~0.95 s/update。

**一輪。** 9k updates ≈ **2.4 小時**（4 卡）。

**假樣本池。** 論文的比例是真假 1:1，對本 repo 就是 244.6 h 的假音訊——
一次全池前向 ≈ `244.6h×3600×100 frames / 19200 × 0.42 s` ≈ 1930 s，×32 NFE ≈ **17 小時單卡**、
~4.3 小時四卡。每輪都要重生。

這太貴了，所以**實務上池子定在 50–60 h（真實語料的 ~20–25%）**，
靠 `real_fake_ratio` 的索引重複把每個 batch 的真假比例拉回 1:1；
代價是假樣本在一輪內被重複使用的次數是真樣本的 ~5 倍。生成成本降到
**~3.5 小時單卡 / ~1 小時四卡**。池子大小是第一批要做的 ablation 之一。

**整體。** 論文擴散版要 12–28 輪。以 24 輪、四卡計：
`24 × (1 h 生成 + 2.4 h 訓練) ≈ 82 GPU·小時`（以四卡計是 ~3.5 天 wall-clock）。
先跑 3–4 輪看斜率，再決定要不要走完。

**顯存。** ref 的 bf16 權重 +1.33 GiB，ref 前向在 `no_grad` 下不留 activation。
24 GB 卡上把 `batch_size_per_gpu` 減半、`grad_accumulation_steps` 加倍即可維持更新尺寸。
若要依 §3.7 改跑全 fp32，再減半一次。

---

## 6. 風險與否證條件

| 風險 | 徵兆 | 處置 |
|---|---|---|
| 判別器走捷徑（長度/音量/削頂/frame grid） | `ddo/acc` 幾步內衝到 >0.95，但 `gen/utmos` 不動 | 檢查 §3.5 四點；真假各抽 20 條人耳/頻譜比對 |
| RoPE 位置沒對齊（§3.6） | Δ 在 `θ=θ_ref` 時不為 0；`acc` 黏在 0.5；`delta_std` 一開始就很大 | 單元測試會擋下來：`θ=θ_ref` 時 `Δ ≡ 0` |
| bf16 抵消（§3.7） | `delta_std` 呈現量化階梯；Δ 直方圖離散 | Δ 全走 fp32；仍不行就整輪改 fp32（論文的做法） |
| EMA 太長（§3.8） | 所有 `gen/*` 曲線平得像沒訓練，但 `ddo/*` 明顯在動 | 縮短 `ema_kwargs.beta` |
| β 飽和 | `acc > 0.9`，`ddo/loss_*` 趨近 0，梯度範數塌陷 | 降 β，見 §3.3 |
| 推過頭（**預期會發生**） | `flow_loss` 暴衝、`silence_ratio`/`clipping_rate` 異常、聽起來過度銳化 | 這是設計使然（§2.3）：挑輪中最佳 checkpoint，不是最後一個 |
| CFG 幾何被破壞 | `gen/spk_sim_self` 下降，或最佳 `cfg_strength` 大幅偏移 | 檢查 anchor 是否生效；先跑 `state_null_prob: 0` 的 arm |
| 假樣本池過擬合 | 後期 `ddo/delta_fake` 持續壓低但 `gen/utmos` 停滯 | 加大池子或中途重生 |

**這個方法在本 repo 上失敗長什麼樣：** 一輪跑完 `gen/utmos` 沒有可見上升，
而 `ddo/acc` 停在 0.5（訊號太弱）或衝到 0.99（捷徑或飽和）。兩者都能在
標定階段的 300 步探針裡看出來，不必燒掉整輪。

**副作用預期：** DDO 之後最佳的 `cfg_strength` 應該會**下降**——論文的主要賣點就是
「不需要 guidance 也能達到更好的 FID」，而 Theorem 3.3 說 `β<1` 的最優解本身就是
往 `p_data/p_ref` 方向的外插，形式上與 guidance 同構。評估時必須重掃 `cfg_strength`，
拿舊的 2.0 去比是把改善算進誤差裡。

---

## 7. 交付範圍

新增：

- `src/wavtts/model/ddo.py` — Δ 與 DDO 損失（純函數，不持有狀態）
- `scripts/gen_fake_pool.py` — 離線假樣本池生成，輸出與 `data/<name>/` 同格式
- `src/wavtts/configs/WavTTS_ddo_r1.yaml` — 第一輪配置
- `tests/test_ddo.py`、`tests/test_ddo_data.py`、`tests/test_ddo_train.py` — CPU-only

修改：

- `src/wavtts/model/backbones/dit.py` — `forward` 加 optional `positions`
- `src/wavtts/model/cfm.py` — `attach_ddo_ref()`、`forward` 的 DDO 分流
- `src/wavtts/model/dataset.py` — 假樣本池的 tagged concat 與 `is_fake` collate 欄位
- `src/wavtts/model/trainer.py` — ref 搬卡、`max_updates`、`ema_kwargs`、`loss_dict` 泛化記錄
- `src/wavtts/train/train.py` — 建 ref、組資料集
- `README.md` — DDO 章節

不動：`infer/`、`eval/`、`modules.py`、`utils.py`、`rope.py`、`train/datasets/`。

**向後相容是硬性要求**：`ddo` 區塊缺席時，`CFM.forward` 的行為、state dict 的 key、
既有 40+ 個測試全部不變。

---

## 附錄 A：論文 Appendix D 的擴散版 DDO loss（逐字轉錄）

這是擴散版唯一的權威實作來源——NVlabs/DDO 的 repo 裡**沒有**這段程式碼。

```python
# Diffusion DDO loss of EDM
class EDMLoss_DDO:
    def __init__(self, P_mean=-0.4, P_std=1.0, sigma_data=0.5, alpha=1.0, beta=0.02):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.alpha = alpha
        self.beta = beta

    def __call__(self, net, ref_net, images, fake_images, labels=None, fake_labels=None):
        # Sample diffusion time
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        # Diffusion loss weighting
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        # Sample Gaussian noise
        noise = torch.randn_like(images) * sigma
        # Denoise by the target model
        D = net(images + noise, sigma, labels)
        net.eval()
        D_fake = net(fake_images + noise, sigma, fake_labels)
        net.train()
        D_logp = -torch.sum(weight * (D - images) ** 2, dim=(1, 2, 3))
        D_fake_logp = -torch.sum(weight * (D_fake - fake_images) ** 2, dim=(1, 2, 3))
        # Denoise by the reference model
        with torch.no_grad():
            ref_D = ref_net(images + noise, sigma, labels)
            ref_D_fake = ref_net(fake_images + noise, sigma, fake_labels)
        ref_D_logp = -torch.sum(weight * (ref_D - images) ** 2, dim=(1, 2, 3))
        ref_D_fake_logp = -torch.sum(weight * (ref_D_fake - fake_images) ** 2, dim=(1, 2, 3))
        # Compute loss
        loss = -F.logsigmoid(self.beta * (D_logp - ref_D_logp)) \
               -self.alpha * F.logsigmoid(-self.beta * (D_fake_logp - ref_D_fake_logp))
        return loss.mean()
```

讀這段時要注意的四件事：

1. `sigma` 與 `noise` **抽一次，四次前向共用**（真/假 × θ/ref）。
2. `*_logp` 是 `-torch.sum(...)`，**沒有除以維度**。
3. 只有 ref 的前向在 `torch.no_grad()` 下；θ 的兩次前向都要梯度，**沒有任何 `.detach()`**。
4. `net.eval()` / `net.train()` 只包住 θ 對假樣本的前向——在 dropout 全關的情況下
   幾乎是 no-op，但官方片段就是這樣寫的。

官方 VAR 實作（`DDO_trainer.py`）與此結構相同，差別是 Δ 用精確的 token log-prob 和，
真假兩批 `torch.cat` 成一個 `2B` batch 一次前向，最後多一個 `loss / max(alpha, 1.0)`。
