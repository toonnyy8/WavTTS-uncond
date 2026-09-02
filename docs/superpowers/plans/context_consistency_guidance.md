# 上下文一致性引導（Context-Consistency Guidance, CCG）
### 用長/短 clip velocity 差分做 CFG 式引導，改善分段 posterior sampling 的語者/音色一致性 — 實作參照技術文檔

- 版本：v0.1（2026-09-02）
- 狀態：**研究提案，未經實驗驗證**。本文件為實作規格與風險清單，非已證結果。
- 適用場景：zero-shot diffusion/flow-matching source separation 的分段（segment-wise）posterior sampling；亦適用於一般長音訊生成的跨段一致性。
- 相依前提：一個在固定長度 L_train 音訊片段上訓練的 unconditional diffusion 或 flow-matching prior（waveform 或 latent 皆可），如 UnDiff / MSDM / 單源歌聲 DM。

---

## 1. 問題與動機

固定長度訓練的 unconditional prior 在長序列上分段做 posterior sampling 時，各段之間只透過（可選的）重疊區弱耦合，導致 **source identity 漂移**：分離出的音軌每隔數秒從一個語者/樂器切換到另一個（Duet Singing Separation [5] 明確記錄此失敗模式）。既有解法（overlapping segments + autoregressive conditioning [5]）把前段結果當 inpainting 條件硬性延伸，屬「replacement 式」約束，強度不可調、且無法在去噪早期就把身分資訊注入梯度方向。

CCG 的核心觀察：**「有上下文」與「無上下文」兩個 velocity/score 的差分，方向上近似指向「與上下文同一身分」的證據梯度**，可以用 classifier-free guidance（CFG）[1] 的形式加權放大。這與 History Guidance（DFoT）[2] 在影片領域的做法同構——他們證明對「歷史條件」做 CFG 式引導能顯著提升長 rollout 的時序一致性——差別在於 DFoT 的模型原生支援可變長 context（需重訓），而 CCG 的目標是 **training-free**，直接用固定長度 prior 拼裝出兩個分支。

## 2. 記號與前置假設

音訊表徵 x ∈ R^n（waveform 或 latent 序列；下述所有操作沿「序列時間軸」進行）。採 rectified-flow / CondOT 路徑約定：

```
x_t = (1 − t)·x_0 + t·ε,   ε ~ N(0, I),   t ∈ [0, 1]
```

- t = 1 為純噪聲、t = 0 為資料；取樣沿 t: 1 → 0 積分 ODE dx/dt = v_θ(x_t, t)。
- 最優 velocity v* = ε − x_0；clean estimate（flow 版 Tweedie）：**x̂_0 = x_t − t·v̂**。
- 與 score 模型互換（若 prior 是 score/ε-prediction）：由 x_t − x_0 = t·(ε − x_0) = t·v* 得三個等價轉換，實作上一律以 x̂_0 為中介：
  - `x0 = xt - t * v`（velocity → clean）
  - `v = (xt - x0) / t`（clean → velocity）
  - `eps = (xt - (1 - t) * x0) / t`，score `s = -eps / t`（clean → ε/score）
  請以單元測試驗證：對已知 (x_0, ε, t) 合成 x_t，檢查各轉換往返誤差 < 1e-6，避免不同 codebase 的 t 方向（t=0 是資料或噪聲）與符號約定不一致。
- 序列切成段 {x^(1), …, x^(M)}，段長 L ≤ L_train，相鄰段重疊 O 個樣本（overlap ratio r = O/L）。
- 混合觀測 y = Σ_k s_k（單通道分離），likelihood 引導沿用 DPS [7] / FlowDPS [8] / MSDM Dirac [6] 任一。

## 3. 方法定義

### 3.1 兩個分支

對第 m 段、去噪時刻 t：

**短分支（無上下文，"unconditional"）**
```
v_short = v_θ(x_t^(m), t)
```
只看當前段。**注意**：此分支必須各段獨立、不重疊計算。若用 MultiDiffusion [4] 式的重疊平均當短分支，重疊區已隱含弱上下文耦合，差分 Δ 會被稀釋（見 §8 風險二）。

**長分支（有上下文，"conditional"）**

構造一個含上下文的輸入，長度仍 ≤ L_train：

```
c_t   = (1 − t_ctx)·x̃_prev + t_ctx·ε_ctx          # 前文（已分離的前段尾端，長度 C = L − L_cur）
x_ctx = concat(c_t, x_t^(m)[cur])                   # 總長 ≤ L_train
v_long_full = v_θ(x_ctx, t)                         # 對整個拼接輸入 forward
v_long = v_long_full[cur]                           # 只取當前段對應的切片
```

其中 x̃_prev 是前段（同一源軌）已完成的分離估計。t_ctx 是上下文的噪聲等級，有三種設計：

| 設計 | t_ctx | 性質 | 依據 |
|---|---|---|---|
| (A) 同步噪聲 | t_ctx = t，ε_ctx 固定（每段取樣開始時抽一次，整條軌跡重用） | replacement/inpainting 式；固定長度 prior 完全 in-distribution | Duet [5] 的 noised-context 條件化 |
| (B) 零噪聲 | t_ctx = 0 | context 資訊最強，但對「訓練時所有位置同噪聲」的 prior 是 OOD 輸入 | 需 Diffusion Forcing 式 prior [3] 才安全 |
| (C) 部分噪聲 | t_ctx = κ·t，κ ∈ (0,1) | 折衷；等效對歷史做低通（Fractional History Guidance） | DFoT HG-f [2] |

**預設用 (A)**；若觀察到 context 資訊不足（Δ 範數過小），改試 (C) 並掃 κ ∈ {0.3, 0.5, 0.7}。(B) 只在 prior 經 per-token noise 訓練時使用。

**關鍵：ε_ctx 必須在整條去噪軌跡中固定**（同一 realization），否則 context 每步抖動，Δ 方向噪聲極大。

### 3.2 引導組合

```
Δ_t = v_long − v_short
ṽ_t = v_short + w(t) · Φ(Δ_t)
```

貝氏解讀：v_short ≈ −∇ 對應 log p(x^(m))、v_long ≈ log p(x^(m) | ctx)，故 Δ ≈ ∇ log p(ctx | x^(m))，即「本段與上下文同一身分」的證據方向；w > 1 銳化該後驗，與 CFG [1] 完全同構。

**w(t)：限區間排程。** 語者/音色身分屬全局低頻結構，在高噪聲步即已決定；低噪聲步的 Δ 主要含 OOD 退化與高頻誤導（§8 風險一）。依 Kynkäänniemi et al. [12]（guidance 只在中高噪聲區間有益）：

```
w(t) = w_max · smoothstep((t − τ_lo)/(τ_hi − τ_lo)),  預設 τ_lo = 0.5, τ_hi = 0.9
     = 0                                              當 t < τ_lo
```

即只在去噪前段（約前 30–50% 步數）施加引導，之後 ṽ = v_short 保細節。

**Φ(·)：差分後處理，兩個可組合的算子。**

(i) **低通投影**（處理 OOD 高頻污染，動機：FreeLong [11] 觀察到長輸入下全局低頻結構可靠、高頻退化；FDG [15] 顯示頻域分解 guidance 有效）：
```
Φ_LP(Δ) = LP_fc(Δ)          # 沿序列時間軸的 low-pass
```
waveform 域建議 fc 對應「音色/能量包絡」尺度（~50–200 Hz 以下的調變頻率，即對 envelope 做濾波而非對訊號本身）；latent 域直接對 latent 序列做 depthwise 1-D 低通（kernel 長度 ≈ latent frame rate × 0.1–0.3 s）。實作最簡版：`Δ_lp = upsample(downsample(Δ, k), k)`，k = 4–16。

(ii) **APG 式平行分量抑制 + 範數控制**（處理過度引導的飽和/嗡鳴，依 APG [13]、CFG-Rescale [14]）：
```
Δ∥ = (⟨Δ, v_short⟩ / ‖v_short‖²) · v_short
Δ⊥ = Δ − Δ∥
Φ_APG(Δ) = η·Δ∥ + Δ⊥,   η ∈ [0, 0.5]（預設 0；即只保留正交分量）
```
之後對 ṽ 做 rescale：`ṽ ← ṽ · (‖v_short‖/‖ṽ‖)^φ`，φ ∈ [0.5, 1]（音訊上過度 guidance 的症狀是動態壓縮與 buzzing，φ 越大抑制越強）。

組合順序：`Φ = rescale ∘ Φ_APG ∘ Φ_LP`。消融時三者獨立開關。

### 3.3 與 posterior sampling 疊加（分離設定）

CCG 修改的是 **prior drift**；likelihood 引導照常疊加。以 DPS/FlowDPS 式 Gaussian likelihood 為例，對每個源 k：

```
x̂_0,k = x_t,k − t·ṽ_t,k                              # 用 CCG 後的 velocity 算 clean estimate
g_k    = ∇_{x_t,k} ‖ y − Σ_j x̂_0,j ‖²                 # sum-constraint（Jayaram & Thickstun [16] / MSDM Gaussian [6]）
v_post,k = ṽ_t,k + ζ(t)·g_k
```

若用 MSDM Dirac [6]（第 N 源設為殘差），CCG 只施加在自由源的 velocity 上，殘差源自動繼承。**每個源 k 各自維護自己的上下文 x̃_prev,k 與 ε_ctx,k**——這是多源情形的硬性要求（§5.4）。

## 4. 演算法（單段、pseudocode）

```python
def sample_segment_with_ccg(
    prior,            # v_theta(x, t) -> velocity, input length <= L_train
    y_seg,            # 當前段混合觀測
    prev_tails,       # dict: source_k -> 前段已分離結果的尾端 (長度 C)，第一段為 None
    n_sources, n_steps, L_cur, C,
    w_max=2.0, tau_lo=0.5, tau_hi=0.9,
    t_ctx_mode="sync",         # "sync" | "fractional"
    kappa=0.5, lp_k=8, eta_apg=0.0, phi_rescale=1.0, zeta_fn=...,
):
    ts = timesteps(1.0, 0.0, n_steps)                       # t: 1 -> 0
    x = {k: randn(L_cur) for k in range(n_sources)}         # 各源初始噪聲（t=1）
    eps_ctx = {k: randn(C) for k in range(n_sources)}       # 每段抽一次、整段軌跡固定

    for t, t_next in pairs(ts):
        v_tilde, x0 = {}, {}
        for k in range(n_sources):
            # ---- 短分支：只看當前段 ----
            v_s = prior(x[k], t)

            # ---- 長分支：拼接前文 ----
            if prev_tails[k] is not None:
                t_c = t if t_ctx_mode == "sync" else kappa * t
                c   = (1 - t_c) * prev_tails[k] + t_c * eps_ctx[k]
                v_l = prior(concat(c, x[k]), t)[C:]          # 只取當前段切片
                delta = v_l - v_s
                delta = lowpass(delta, lp_k)                              # Φ_LP
                d_par = project(delta, onto=v_s); d_orth = delta - d_par  # Φ_APG
                delta = eta_apg * d_par + d_orth
                w = w_max * smoothstep((t - tau_lo) / (tau_hi - tau_lo)) if t > tau_lo else 0.0
                v = v_s + w * delta
                v = v * (norm(v_s) / max(norm(v), 1e-8)) ** phi_rescale   # rescale
            else:
                v = v_s                                                    # 第一段：純 prior
            v_tilde[k] = v
            x0[k] = x[k] - t * v

        # ---- likelihood 引導（sum constraint；可換成 MSDM Dirac）----
        residual = y_seg - sum(x0.values())
        for k in range(n_sources):
            g = grad(lambda xk: norm(y_seg - sum_x0_wrt(xk))**2, x[k])    # 需對 prior 反傳
            v_tilde[k] = v_tilde[k] + zeta_fn(t) * g

        # ---- ODE/SDE 步進 ----
        for k in range(n_sources):
            x[k] = x[k] + (t_next - t) * v_tilde[k]                       # Euler；可換 Heun

    return {k: x[k] for k in range(n_sources)}                            # t≈0 的分離估計
```

全曲流程：段 m=1…M 依序呼叫上式，`prev_tails[k] = 上一段輸出的最後 C 個樣本`；相鄰段輸出在重疊區用 crossfade 或 MultiDiffusion 式平均合併（**只在最終波形合併時**，見 §3.1 注意事項）。

## 5. 實作細節與陷阱

**5.1 每步 NFE 成本。** 每源每步 = 2 次 prior forward（短+長）+ 1 次反傳（likelihood）。相對 Duet 式 baseline（1 forward + 1 反傳）約 +60–80% 時間。長分支輸入較長，若 prior 是 attention 架構，成本按序列長度平方增長——C 不必貪大，語者身分約 2–4 秒上下文即足，建議 C ≈ 0.25–0.5·L_train。

**5.2 批次化。** 短分支與長分支長度不同，無法同 batch；但 K 個源的同分支可以 batch。若 prior 支援 attention mask/padding，可把兩分支 pad 到同長合併成一個 batch，實測是否比兩次 forward 快。

**5.3 位置編碼。** 長分支輸入若超過 L_train 會觸發 positional encoding 外推問題——本設計刻意讓 `C + L_cur ≤ L_train` 避開。**不要**為了更長 context 超過 L_train，除非 prior 用 RoPE 且驗證過外推。

**5.4 多源必須各自條件化。** Δ 方向除了語者身分，也編碼響度、節奏、混響等全局屬性。單源情形無害；duet/同音色多源情形，若共用 context 或用混合當 context，引導會把各軌拉向彼此（identity 縮並，恰是要解的問題的反面）。每源獨立的 prev_tail + ε_ctx 是硬性要求。

**5.5 第一段的身分錨定。** 第一段無 context，permutation 由 prior + likelihood 隨機決定。可選增強：以外部 speaker/instrument embedding 對第一段做 oracle-free 檢查（兩軌 embedding 相似度過高 → 重抽初始噪聲重跑），或人工指定參考片段當 pseudo-context。

**5.6 latent prior 的注意事項。** 若 prior 在 VAE latent 上（如 MSLDM 系），CCG 全程在 latent 序列軸操作，Φ_LP 的 kernel 以 latent frame 為單位；likelihood 的 sum constraint 在 latent 空間不嚴格成立（encoder 非線性），需沿用該系統原本的 data-consistency 方案，CCG 不解決此問題也不使其惡化。

**5.7 數值驗證清單（實作後必跑）。**
1. w_max = 0 時輸出應與 Duet 式 baseline（無 CCG）逐位一致（容差內）。
2. 對合成資料（已知兩源）檢查 Δ 的符號：把 prev_tail 換成「錯誤語者」時，⟨Δ_correct, Δ_wrong⟩ 應顯著為負或近正交——這驗證 Δ 真的攜帶身分資訊而非純 OOD 噪聲。
3. 逐步記錄 ‖Δ‖/‖v_short‖ 曲線：健康值約 0.05–0.5；若 t < 0.3 後仍 > 0.5，代表 OOD 污染主導，收緊 τ_lo 或加強 Φ_LP。

## 6. 超參數預設與掃描範圍

| 參數 | 預設 | 掃描範圍 | 說明 |
|---|---|---|---|
| w_max | 2.0 | {1, 1.5, 2, 3, 5} | w=1 即純長分支（無外推）；>3 需搭配 APG/rescale |
| τ_lo, τ_hi | 0.5, 0.9 | τ_lo ∈ {0.3, 0.5, 0.6} | 引導區間 [12]；τ_lo 越低越激進 |
| C（context 長） | 0.5·L_train | {0.25, 0.5}·L_train | 身分資訊 2–4 s 即飽和 |
| overlap ratio r | 0.25 | {0.1, 0.25, 0.5} | 與 Duet [5] 對齊以便公平比較 |
| t_ctx 模式 | sync (A) | A / C(κ=0.5) | §3.1 表 |
| lp_k | 8 | {1(關), 4, 8, 16} | 低通下採樣因子 |
| η_apg | 0.0 | {0, 0.25, 0.5, 1(關)} | 1 = 不投影 |
| φ_rescale | 1.0 | {0, 0.5, 1} | 0 = 關 |

消融優先序：w_max 與 τ_lo（核心）→ Φ_LP 開關 → APG/rescale 開關 → t_ctx 模式。

## 7. 評估協定

主指標三組，缺一不可（單看 SI-SDR 會漏掉本方法的主要效果）：

1. **身分一致性**：對每段分離結果抽 speaker embedding（如 ECAPA-TDNN）或樂器 embedding（CLAP audio encoder），量測 (a) 段間相似度序列的均值/最小值；(b) identity switch rate = 相鄰段相似度低於門檻的比率。這是 CCG 的直接目標。
2. **分離品質**：SI-SDRi（全曲拼接後計算，不只逐段），另報 permutation-invariant 版本以分離「排列錯誤」與「品質下降」兩種失敗。
3. **感知/生成品質**：DNSMOS 或 FAD，監測過度引導的 buzzing/動態壓縮 artifact。

對照組：(a) naive 分段（無 overlap、無 conditioning）；(b) Duet 式 AR posterior sampling [5]（同 overlap ratio）；(c) CCG w_max=1（純長分支，驗證「外推」本身的貢獻）。

## 8. 已知風險與失敗模式

**風險一：v_long 的 OOD 污染。** 固定長度 prior 對「context 噪聲模式與訓練不同」（設計 C）或「拼接邊界」敏感，Δ 中混有 OOD 退化方向，w>1 一併放大。緩解：設計 (A) 的同步噪聲 + 限區間 w(t) + Φ_LP。若 §5.7-3 的診斷曲線異常，優先收緊 τ_lo。

**風險二：短分支被污染導致 Δ 稀釋。** 若短分支計算時混入重疊平均（MultiDiffusion 式），uncond 分支已含上下文，Δ 變小，被迫調大 w_max 而放大風險一。規格已規定短分支各段獨立。

**風險三：錯誤身分的自我強化。** AR 傳遞的固有問題：若第 m 段分錯（或第一段 permutation 不佳），CCG 會忠實地把錯誤身分傳下去且比 baseline 更「堅定」。CCG 不解決此問題，只降低「無故切換」；解決錯誤鎖死需要多假設機制（→ §9 SMC）。

**風險四：引導過強的音訊 artifact。** 症狀為頻譜過度平滑、動態壓縮、嗡鳴。監測 DNSMOS/FAD，用 APG + rescale + 限區間三重保險。

**風險五：Δ 攜帶非身分屬性。** 響度/混響/節奏也在 Δ 裡；跨段錄音條件變化大的素材（現場錄音）可能被強行「平滑化」。若成為問題，考慮在 embedding 空間定義更窄的 potential（超出本文件範圍，屬 SMC 權重設計）。

## 9. 與 block-wise SMC 的整合（展望）

CCG 與時間軸 SMC 正交且互補：CCG 把一致性放進 **proposal 的 drift**，SMC 把一致性放進**粒子權重**。整合方式：

- 每個粒子 = 一條「至今為止的分離假設」；新段用 CCG-guided posterior sampling 當 proposal（每粒子各自的 prev_tails）。
- 權重 potential = 重疊區重建一致性 + embedding 段間相似度 + 混合 likelihood。
- 預期效益：CCG 降低 proposal 與目標的差距 → 權重變異數下降 → 緩解 weight degeneracy（時間軸 SMC 最大的失敗風險）；SMC 的多假設與 resampling 則補上 CCG 的風險三（錯誤鎖死）。
- 建議實驗順序：先單獨驗證 CCG（§7 協定）；確認一致性提升且 SI-SDR 不降後，再包 SMC 外殼，比較 K ∈ {1, 4, 8} 粒子下的 identity switch rate 與權重 ESS。

## 10. 參考文獻

| # | 文獻 | 出處 | 在本文件中的角色 |
|---|---|---|---|
| [1] | Ho & Salimans, *Classifier-Free Diffusion Guidance* | NeurIPS 2021 Workshop; arXiv:2207.12598 | CFG 外推式的原型（§3.2） |
| [2] | Song, Chen, Simchowitz, Du, Tedrake, Sitzmann, *History-Guided Video Diffusion* (DFoT) | ICML 2025; arXiv:2502.06764 | 最近前例：對可變長歷史做 CFG 式引導提升時序一致性；Fractional History Guidance 對應 §3.1 設計 (C) |
| [3] | Chen et al., *Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion* | NeurIPS 2024; arXiv:2407.01392 | per-token 獨立噪聲訓練，設計 (B) 的前提 |
| [4] | Bar-Tal, Yariv, Lipman, Dekel, *MultiDiffusion* | ICML 2023; arXiv:2302.08113 | 重疊窗合併（僅用於最終波形合併；§3.1 警告勿用於短分支） |
| [5] | Yu et al., *Zero-Shot Duet Singing Voices Separation with Diffusion Models* | arXiv:2311.07345 | 分段 AR posterior sampling baseline；identity 漂移的問題定義 |
| [6] | Mariani, Postolache et al., *Multi-Source Diffusion Models* (MSDM) | ICLR 2024; arXiv:2302.02257 | Dirac/Gaussian likelihood 分離框架（§3.3） |
| [7] | Chung et al., *Diffusion Posterior Sampling* (DPS) | ICLR 2023; arXiv:2209.14687 | likelihood 引導骨架 |
| [8] | Kim, Kim, Ye, *FlowDPS* | ICCV 2025; arXiv:2503.08136 | flow-matching 版 posterior sampling（velocity/Tweedie 拆解） |
| [9] | Wu, Trippe et al., *Practical and Asymptotically Exact Conditional Sampling* (TDS) | NeurIPS 2023; arXiv:2306.17775 | §9 SMC 外殼的 twisting 框架 |
| [10] | Lipman et al., *Flow Matching for Generative Modeling* | ICLR 2023; arXiv:2210.02747 | flow 路徑與 velocity 約定（§2） |
| [11] | Lu et al., *FreeLong* | NeurIPS 2024; arXiv:2407.19918 | 長輸入下「低頻可靠、高頻退化」的實證依據（§3.2 Φ_LP） |
| [12] | Kynkäänniemi et al., *Applying Guidance in a Limited Interval Improves Sample and Distribution Quality* | NeurIPS 2024; arXiv:2404.07724 | 限區間 guidance 排程 w(t)（§3.2） |
| [13] | Sadat, Hilliges, Weber, *Eliminating Oversaturation and Artifacts of High Guidance Scales* (APG) | ICLR 2025; arXiv:2410.02416 | 平行分量抑制 + rescale + momentum（§3.2 Φ_APG） |
| [14] | Lin et al., *Common Diffusion Noise Schedules and Sample Steps are Flawed* | WACV 2024; arXiv:2305.08891 | CFG-Rescale 範數控制 |
| [15] | *Guidance in the Frequency Domain Enables High-Fidelity Sampling at Low CFG Scales* (FDG) | arXiv:2506.19713 | 頻域分解 guidance 的獨立佐證（§3.2 Φ_LP） |
| [16] | Jayaram & Thickstun, *Source Separation with Deep Generative Priors* | ICML 2020; arXiv:2002.07942 | Gaussian sum-constraint likelihood 的原始出處 |

---
*附註：[15] 為 2025 年 arXiv 預印本，作者與正式發表場合請於引用前再確認；其餘條目的 arXiv 編號已於 2026-09-02 核對。*
