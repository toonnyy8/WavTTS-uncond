# 以 Inversion + Replacement 在 Unconditional WavTTS 上實現 Training-Free 語者條件生成

**技術文檔 v1.0** · 實作參照用

---

## 1. 目標與定位

本文檔描述一種 training-free 的語者條件生成方法：在一個純 unconditional 的 WavTTS（raw-waveform rectified flow）上，不改架構、不微調，僅靠推論期操作達成 speaker-conditioned 生成。

核心思路：

1. 將參考語音 `x_ref` 經 ODE inversion 映回噪聲端，得到 `n_ref` 與整條反演軌跡。
2. 將 `n_ref` 與新取樣的噪聲 `n` 沿時間軸拼接為初始狀態。
3. 反向 ODE 的每一步，把 ref 區段替換回其參考軌跡上的對應狀態，生成區則正常積分。attention 使生成區能持續讀取一個 on-trajectory 的 ref，從而繼承語者身份。

方法學上這等價於用 unconditional prior 做 outpainting 式的近似 posterior sampling（近似 `p(x_gen | x_ref)`），屬於 RePaint / score-SDE inpainting 一族在確定性 flow ODE 上的適配。預期用途是作為 zero-shot source separation 研究中「speaker-conditioned 單源 prior」的快速原型：先驗證語者條件對 permutation / identity 漂移的緩解效果，再決定是否投入訓練 embedding 條件版模型。

**非目標**：本方法不是 TTS（無文字條件、內容不可控），也不追求超越原生 prompt 串接的音色相似度。

---

## 2. 前置：模型與記號

### 2.1 底層模型

假設已有一個 unconditional WavTTS：DiT 骨幹上的 rectified flow，x-prediction 目標，訓練細節沿用原論文（Chen et al., 2026）：

- 波形 16 kHz，non-overlapping patchify，patch 長度 `F = 160`（patch 序列率 100 Hz）。
- Signal-Noise Variance Alignment：訓練目標為縮放後波形 `x' = k·x`（Emilia 上 `k ≈ 9`）。**本文所有波形空間操作（含 inversion 的輸入）都在縮放域進行**，輸出再除以 `k` 還原。
- 訓練時 timestep 採 LogitNormal(μ=−0.8, σ=0.8)。
- 無文字、無 x_ctx、無 infilling mask——這是與原版 WavTTS 最大的差異，訓練退化為乾淨的 unconditional flow matching。若模型是從帶 CFG dropout 的條件版取其 uncond 分支，需注意該分支僅在約 10% 訓練步被優化，品質上限較低；建議專門訓練 uncond 模型。

### 2.2 Rectified flow 記號

內插路徑（t=0 為噪聲端、t=1 為資料端）：

```
x_t = (1 − t)·x_0 + t·x_1,    x_0 ~ N(0, I)
```

模型輸出 x-prediction `x_θ(x_t, t)`，對應 velocity：

```
v_θ(x_t, t) = (x_θ(x_t, t) − x_t) / (1 − t)
```

反向（生成）Euler 步，時間表 `0 = t_0 < t_1 < … < t_K = 1`（建議沿用 PolyShift，p=2, s=3）：

```
x_{t_{i+1}} = x_{t_i} + (t_{i+1} − t_i) · v_θ(x_{t_i}, t_i)
```

### 2.3 序列佈局

以 patch 為單位。ref 佔前 `L_r` 個 patch、生成區佔後 `L_g` 個 patch，總長 `L = L_r + L_g`。定義二值遮罩 `M ∈ {0,1}^{L×F}`：ref 區為 1、生成區為 0。與原版 WavTTS 不同，這個 mask 純粹是推論期的簿記工具，模型本身看不到它。

`L_g` 需事先指定（NAR 模型的固有限制）。uncond 模型無文字，長度沒有語意錨點，直接按需求秒數設定即可。

---

## 3. 方法

### 3.1 階段一：Inversion（取得參考軌跡）

對縮放後的 `x'_ref` 反向積分 ODE（把生成方向倒過來跑，t 從 1 走到 0）：

```
x^ref_{t_{i−1}} = x^ref_{t_i} − (t_i − t_{i−1}) · v_θ(x^ref_{t_i}, t_i),    x^ref_{t_K} = x'_ref
```

得到 `n_ref = x^ref_{t_0}` 以及**整條中間狀態序列 `{x^ref_{t_i}}`，全部快取**。

兩點注意：

- **快取軌跡，而非只存 n_ref。** 線性內插 `t·x_ref + (1−t)·n_ref` 假設模型 ODE 軌跡是直線；rectified flow 只有在完美 rectify 時才成立，實際模型有曲率。用快取的 `x^ref_{t_i}` 做替換，ref 區走的是模型自己的軌跡，替換零近似誤差。記憶體成本為 K 份 ref 區狀態（K=50、10 秒 ref 約 50 × 10s × 16kHz × 4B ≈ 32 MB，可接受）。
- **Inversion 誤差控制。** 一階 Euler inversion 有截斷誤差，重建檢查（拿 n_ref 正向積分回來與 x_ref 比 SNR）應 > 25 dB 才算合格；不足時提高 inversion 的 NFE（inversion 與生成的 NFE 不必相同，替換時對 t 做最近鄰或內插對齊即可）。也可換用高階 solver（Heun）或 rectified-flow 專用 inversion（RF-Inversion、FireFlow）改善。inversion 只對 ref 區做一次，成本可攤提到多次生成。

**Inversion 時的長度陷阱**：inversion 應在「ref 單獨成序列」的設定下做，還是「ref + 零填充到全長」？建議前者——單獨反演 ref，快取的軌跡是「模型認為這段語音自己的軌跡」。後者會讓零填充區污染 attention。代價是生成時 ref 區的 positional 範圍與 inversion 時一致（ref 放在序列開頭），RoPE 的相對位置性質使這個安排自然成立。

### 3.2 階段二：拼接與替換生成

初始化：

```
x_{t_0} = M ⊙ x^ref_{t_0} + (1 − M) ⊙ n,    n ~ N(0, I)
```

每步 Euler 更新後，立即做投影（replacement）：

```
x̃_{t_{i+1}} = x_{t_i} + Δt · v_θ(x_{t_i}, t_i)          # 全序列一起算 velocity
x_{t_{i+1}} = M ⊙ x^ref_{t_{i+1}} + (1 − M) ⊙ x̃_{t_{i+1}}   # ref 區釘回參考軌跡
```

velocity 必須對全序列計算（不能只餵生成區），語者資訊正是經由 attention 從 ref 區流向生成區。ref 區被模型「重建」的輸出直接丟棄。

最終取 `x_{t_K}` 的生成區，除以 `k` 還原振幅。

### 3.3 強化一：Resampling（time-travel）

Replacement 條件化的已知弱點：高噪聲步（t 小）時 ref 區幾乎是純噪聲，語者訊息最稀薄，而全域結構（含 identity）恰在此階段成形。RePaint 的 resampling 迴圈是標準解法——在選定的時間步上，把當前狀態部分重新加噪回退 j 步再前進，重複 U 次，給生成區與 ref 區更多協調機會。

Flow 版本的回退操作（從 t_b 退回 t_a < t_b）：

```
x_{t_a} = (t_a / t_b) · x_{t_b} + sqrt(1 − (t_a / t_b)²) · ε · c,   ε ~ N(0,I)
```

其中係數依線性內插路徑的邊際分布匹配推導；更簡單的做法是用當前的 `x_θ` 估計重新內插：`x_{t_a} = (1−t_a)·ε + t_a·x_θ(x_{t_b}, t_b)`。建議只在中段噪聲區（t ∈ [0.15, 0.5] 左右）做 2–4 輪 resampling，成本線性增加，收益集中在 identity 穩定度。

### 3.4 強化二：Speaker guidance（可選，從 inpainting 走向 inpainting + DPS）

若純 replacement 的音色相似度不足，可對生成區疊加語者相似度梯度。取凍結的 speaker encoder `E`（WavLM-based 或 ECAPA），每步：

```
g = ∇_{x_t} [ cos( E(x_θ(x_t, t)[gen]/k), E(x_ref) ) ]
x_{t_{i+1}} ← x_{t_{i+1}} + η_t · (1 − M) ⊙ g
```

即 DPS 式 guidance：把 likelihood 換成 speaker embedding 相似度，作用範圍限生成區。`η_t` 需隨 t 排程（高噪聲段小、中段大、近資料端歸零），過強會產生 adversarial artifact（騙過 encoder 但聽感劣化）。這一項使方法從純 inpainting 變成 inpainting + posterior guidance 的混合。

### 3.5 強化三：長序列滾動生成

attention 對遠距 ref 的依賴隨距離衰減，生成長度超過訓練片段典型長度後 identity 會漂移。解法沿 Duet Singing 的 overlapping autoregressive posterior sampling：

1. 第一段：`[ref | gen_1]`，照 §3.2 生成。
2. 之後每段：`[ref | tail(gen_{i−1}) | gen_i]`——把上一段的尾部（例如 1–2 秒）也 inversion 後當作額外的 ref 區釘住，滾動前進。
3. 段間交疊區用 crossfade 拼接。

上一段尾部同時攜帶語者與局部聲學連續性，比只靠原始 ref 更能抑制漂移。

---

## 4. 演算法總表

```
輸入: uncond 模型 v_θ, 參考語音 x_ref, 生成長度 L_g,
      NFE K, 縮放 k, PolyShift 時間表 {t_i},
      resample 輪數 U, resample 區間 [t_lo, t_hi],
      guidance 強度排程 {η_t}（可為 0）

# ---- 階段一：inversion（一次性，可快取重用）----
x'_ref = k · x_ref
traj = {}
z = x'_ref
for i = K, …, 1:
    traj[t_i] = z
    z = z − (t_i − t_{i−1}) · v_θ(z, t_i)
traj[t_0] = z                      # z 即 n_ref
assert 重建SNR(traj) > 25 dB       # 品質閘門

# ---- 階段二：生成 ----
x = concat(traj[t_0], n ~ N(0,I))  # 沿 patch 軸拼接
for i = 0, …, K−1:
    repeat (U if t_i ∈ [t_lo, t_hi] else 1) times:
        v = v_θ(x, t_i)                          # 全序列
        x = x + (t_{i+1} − t_i) · v
        x[ref] = traj[t_{i+1}]                   # 投影回參考軌跡
        if η_{t_i} > 0:
            x[gen] += η_{t_i} · ∇ cos(E(x_θ[gen]/k), E(x_ref))
        if resampling 且未到最後一輪:
            x = 回退(x, t_{i+1} → t_i)            # §3.3
return x[gen] / k
```

---

## 5. 實作細節與陷阱

**縮放域一致性。** inversion 輸入、軌跡快取、替換、guidance 的波形重建全部在 `k` 縮放域；只有 speaker encoder 的輸入和最終輸出要除回 `k`。混用是最容易犯的 silent bug。

**Patch 邊界對齊。** ref 長度取 `F=160` 的整數倍，避免 ref/gen 邊界落在 patch 內部。邊界處若有可聽接縫，可在波形域對邊界 patch 做短 crossfade，或在 ref 尾端保留半秒「犧牲區」不計入輸出。

**時間表對齊。** 生成用 PolyShift 時，inversion 若用不同離散化，`traj` 的鍵與生成步的 t 不重合——對 traj 做鄰近兩狀態的線性內插即可，誤差遠小於重跑 inversion。

**velocity 的邊界行為。** t→1 時 x-prediction 轉 velocity 有 1/(1−t) 放大，沿用訓練時的 t 裁剪（≤0.98）。

**成本。** 相對單次 uncond 生成：inversion 一次 K 步（可攤提）+ 生成 K·U̅ 步 + guidance 每步一次 encoder 前傳與反傳。無 guidance、U=1 時約為 2× 單次生成。

**何時放棄本路線。** 若 (a) inversion 重建 SNR 拉不上去、(b) resampling 加到 4 輪 SIM-o 仍不及原生 prompt 串接的 7 成、(c) 生成區內容出現 ref 的語意內容洩漏（模型把 ref 當「上文」續寫語意而非只繼承音色），則轉向 speaker-embedding 條件版訓練。(c) 無法用本方法根治，因為 uncond prior 學到的序列內一致性同時涵蓋語者與內容連貫。

---

## 6. 評估方案

| 面向 | 指標 | 說明 |
|---|---|---|
| 語者相似度 | SIM-o（WavLM-based SV，cos） | 主指標；與 (i) 原生 prompt 串接 WavTTS、(ii) 純 uncond 生成隨機語者 兩個上下界比較 |
| 自然度 | UTMOS / DNSMOS | 確認 replacement 未引入 artifact |
| 長度穩健性 | SIM-o vs 生成長度曲線 | 量測 identity 漂移的距離衰減；驗證 §3.5 的必要性 |
| 消融 | 線性內插 vs 軌跡快取；U ∈ {1,2,4}；η=0 vs η>0 | 定位每個組件的貢獻 |

WER 不適用（無文字條件）。若後續接分離任務，再加 SI-SDRi 與 permutation 錯誤率。

---

## 7. 與分離研究的銜接

本方法產出的等效物是「可依語者條件化的單源 prior 取樣器」。接入 DPS/SMC 分離框架時：

- 對 N 個源各持一段 ref 與其反演軌跡，聯合狀態 `(s_1,…,s_N)` 中每源的 ref 區各自釘住，sum-constraint likelihood（Gaussian 或 Dirac，見 MSDM）作用於生成區。
- 語者條件打破 prior 的 exchangeability，permutation 峰不再等價，理論上直接緩解 DPS 單峰近似在分離後驗上的塌陷；若仍需處理殘餘多峰性，本取樣器可整體作為 SMC（TDS 式）的提議核。

---

## 8. 參考文獻

**底層模型與訓練框架**

1. Chen, W., et al. "WavTTS: Towards High-Quality Zero-Shot TTS via Direct Raw Waveform Modeling." arXiv:2606.03455, 2026.（DiT + rectified flow + x-prediction、patchify、variance alignment k、PolyShift）
2. Chen, Y., et al. "F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching." ACL 2025. arXiv:2410.06885.（infilling 式 flow TTS、Sway Sampling、字元比例估長）
3. Le, M., et al. "Voicebox: Text-Guided Multilingual Universal Speech Generation at Scale." NeurIPS 2023. arXiv:2306.15687.（text-conditioned speech infilling 任務定義）
4. Lipman, Y., et al. "Flow Matching for Generative Modeling." ICLR 2023. arXiv:2210.02747.
5. Liu, X., Gong, C., Liu, Q. "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow." ICLR 2023. arXiv:2209.03003.
6. Ho, J., Salimans, T. "Classifier-Free Diffusion Guidance." arXiv:2207.12598, 2022.

**Replacement / inpainting 式條件化**

7. Lugmayr, A., et al. "RePaint: Inpainting using Denoising Diffusion Probabilistic Models." CVPR 2022. arXiv:2201.09865.（每步替換 + resampling/time-travel 迴圈的來源）
8. Song, Y., et al. "Score-Based Generative Modeling through Stochastic Differential Equations." ICLR 2021. arXiv:2011.13456.（以 unconditional score 模型做 inpainting 條件生成的理論源頭）
9. Song, J., Meng, C., Ermon, S. "Denoising Diffusion Implicit Models." ICLR 2021. arXiv:2010.02502.（確定性取樣與 inversion 的基礎）

**Rectified flow 的 inversion**

10. Rout, L., et al. "Semantic Image Inversion and Editing using Rectified Stochastic Differential Equations." arXiv:2410.10792, 2024.（RF-Inversion）
11. Deng, Y., et al. "FireFlow: Fast Inversion of Rectified Flow for Image Semantic Editing." arXiv:2412.07517, 2024.（低成本高保真 flow inversion）

**Posterior guidance 與音訊分離脈絡**

12. Chung, H., et al. "Diffusion Posterior Sampling for General Noisy Inverse Problems." ICLR 2023. arXiv:2209.14687.（DPS；§3.4 guidance 的形式來源）
13. Mariani, G., Postolache, E., et al. "Multi-Source Diffusion Models for Simultaneous Music Generation and Separation." ICLR 2024. arXiv:2302.02257.（MSDM；Dirac/Gaussian likelihood、joint prior 與 inpainting 模式）
14. Yu, C.-Y., et al. "Zero-Shot Duet Singing Voices Separation with Diffusion Models." arXiv:2311.07345, 2023.（overlapping autoregressive posterior sampling，§3.5 的直接前身）
15. Iashchenko, A., et al. "UnDiff: Unsupervised Voice Restoration with Unconditional Diffusion Model." Interspeech 2023. arXiv:2306.00721.（unconditional 語音 prior 的 post-training conditioning）
16. Wu, L., et al. "Practical and Asymptotically Exact Conditional Sampling in Diffusion Models." NeurIPS 2023. arXiv:2306.17775.（TDS；§7 SMC 銜接）

*註：文獻 10、11 的 arXiv 編號建議在引用前複查；flow inversion 是快速迭代的方向，可能已有更新版本或更佳替代。*
