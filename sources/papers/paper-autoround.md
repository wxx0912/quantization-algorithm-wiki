---
{
  "id": "paper-autoround",
  "title": "Optimize Weight Rounding via Signed Gradient Descent for the Quantization of LLMs",
  "category": "paper",
  "url": "https://arxiv.org/abs/2309.05516",
  "version": "v5（2024-10-08）",
  "version_kind": "paper-version",
  "retrieved_at": "2026-10-24",
  "read_form": [
    "latex",
    "html"
  ],
  "read_locator": "摘要、§1（Figure 1）、§3（Eq.1–6、Algorithm 1、Table 1）、§4.1（设置、Table 2–3）、§4.2（Table 4–5）、§4.4（Table 6–7 的浮动体）、§4.5（Table 7 的讨论）、§5 结论与 Limitations、附录 A（Table 8、Table 9 的讨论）、附录 B（Figure 2、Table 9 的浮动体、Table 10–14）、附录 C（Table 15）",
  "topic": "reconstruction",
  "requires_external_verification": true,
  "excerpt_sha256": null
}
---
## 这份来源是什么

SignRound 原始论文（arXiv 2309.05516，v5；方法后来以 AutoRound 之名进入工具链）。它沿用 AdaRound 的"可训练舍入偏移 `V`"，再叠加可训练裁剪系数 `α, β`，用 **SignSGD** 在块级输出重构目标上只调这几组参数（每 block 200 步、lr 固定 5e-3），不做权重回归。论文主张以 **11 个 0-shot 任务的平均准确率**为主指标，把困惑度（PPL）降为参考，理由是 PPL 对个别 token 的极端概率过于敏感。

它是 `algorithm-autoround` 的算法依据，也与 `recipe-composition`、`guide-algorithm-selection` 相关。

范围边界：weight-only、**非对称**量化，只作用于 transformer block 内的线性层（embedding 与 lm-head 排除，原文措辞是 typically）；主配置是 W4G-1/W4G128/W3G128/W2G128（论文只定义 `G` 为更细粒度分组、未定义 `G-1`，本页按外部惯例把 `G-1` 理解为"不分组"，属推断）。不做激活量化，也不做 KV cache 压缩。本页只覆盖这篇 v5，不包含后续版本或工具仓库（SignRoundV2 / auto-round）的能力声明。

## 已读定位与阅读形式

- 阅读形式：arXiv e-print 的 LaTeX 源（单文件 `acl_latex.tex`，正文与附录同文件），另读 arXiv HTML 渲染版交叉核对编号；v5 = 2024-10-08。
- 定位：§3 方法（Eq.1–6、Algorithm 1、Table 1 的浮动体）、§4 实验（4.1 设置与 Table 2–3 浮动体、4.2 Table 4–5 浮动体、4.4 Table 6 与 Table 7 的浮动体、4.5 Table 7 的讨论）、§5 结论与 Limitations、附录 A（Table 8、Table 9 的讨论）、B（Figure 2、Table 9 的浮动体、Table 10–14）、C（Table 15）。编号：Figure 1–2、Table 1–15、Eq.1–6、Algorithm 1；**附录表沿用主文序号，不是 A 编号**。

## 关键事实

- **量化与可训练参数**（§3，Eq.1–4）：基础式 `W̃ = s·clip(⌊W/s + zp⌉, n, m)`（Eq.1，`s = (max(W) − min(W))/(2^bit − 1)`，Eq.2）。**原文如此**：Eq.1 在反量化侧未显式减 `zp`，按标准 affine 定义应为 `s·(clip(⌊W/s⌉+zp, n, m) − zp)`。可训练舍入偏移 `V`（做法来自 AdaRound）：`W̃ = s·clip(⌊W/s + zp + V⌉, n, m)`（Eq.3）；可训练裁剪系数 `α, β ∈ [0,1]`：`s = (max(W)·α − min(W)·β)/(2^bit − 1)`（Eq.4）。
- **优化目标**（§3，Eq.5）：块级输出重构 `min_{α,β,V} ‖WX − W̃X‖²_F`，不优化任务损失。
- **SignSGD**（§3，Eq.6）：`W_{t+1} = W_t − lr_t·sign(g_t)`。四条理由：① 舍入最优值落在一个区间、只需阈值正确，梯度幅度不必精确；② 参数有界（V 在 [-0.5,0.5]、α/β 在 [0,1]），符号步长足以走遍；③ 步长可推算——200 步、lr 5e-3、线性衰减时单参数最大调整量 `200×0.005/2 = 0.5`，正好覆盖 V 的 [-0.5,0.5] 与裁剪系数的 [0.5,1.0]；④ 比 Adam 省显存与算力。Table 4（`tab:signgd_adam`）：W2G128/Mistral-7B 上 SignSGD 52.71，Adam 扫 lr 后最好只有 43.55。
- **Algorithm 1**：每步取 `bs` 条样本，算原 block 输出 `y_f` 与量化 block 输出 `y_q`、用二者 MSE 反向，**记录 loss 最低时的参数**，再用 SignSGD 更新 `α, β, V`；输出 `best_V, best_α, best_β`。
- **设置**（§4.1）：weight-only、非对称、只量化 block 内线性层；校准集是 pile-10k 的 **512 条随机样本**（所有校准方法共用同一 seed）、序列长 2048；默认每 block **200 步**、lr **5e-3**、batch **8**、线性衰减、开 AMP。评测用 lm-eval-harness 的 11 个 0-shot 任务取平均，PPL（WikiText2/PTB/C4）只作补充。`Ours*` 指在 8 组超参（步数 200/1000 × 裁剪 lr × 是否用前一已量化 block 的输出作为输入）中挑最好的一组，70B 只试少数选项。
- **主结果**（Table 1；11 任务平均准确率，↑）：W2G128 的 Mistral-7B——16-bit 63.30、RTN 30.52、GPTQ 39.61、AWQ 30.06、HQQ 31.41、OmniQuant 32.17、**Ours 52.71**、Ours* 53.01；同配置 LLaMA-V2 7B/13B/70B 的 Ours 为 48.64/53.46/61.69（16-bit 57.98/61.42/66.12）。W4G128 下 Ours 与 GPTQ/AWQ/RTN 都在 60–63 区间；W3G128/Mistral-7B 的 Ours 60.43。
- **与舍入类方法对比**（§4.3、Table 3）：W4G-1/Mistral-7B——FlexRound 200/1000/5000 步 58.93/60.62/60.94、AdaRound 200/1000 步 58.30/58.38、**SignRound 200 步 62.33**；W2G128/Mistral-7B——FlexRound 30.10/30.16、AdaRound 30.74/30.84、**SignRound 200 步 52.71**。论文结论是"200 步就优于它们 5000 步"，并指出这些实现缺 AMP、速度慢（Table 9）。
- **舍入 vs 裁剪消融**（Table 5）：W4G-1 下 RTN 58.84 → 只调裁剪 61.10 → 只调舍入 61.62 → 联合 62.33；W2G128 下"只调舍入"的 LLaMA-V2-13B 为 54.41，而联合为 **53.46**。即两者各自有增益、舍入贡献更大，但联合在某些配置上反而更低。
- **超参敏感性与泛化**（Table 6–7）：W4G-1/Mistral-7B 上默认 62.33，序列 512 → 60.32、样本 128 → 61.82、batch 4 → 61.78、100 步 → 61.06、**1000 步 → 62.58**、lr 1e-2 → 61.27，默认整体均衡。W4G128 下 BF16/FP16 → Ours 的平均准确率变化都在 1% 以内（Gemma-2b 53.69→53.40、Llama-3-8B-Instruct 63.52→63.12、Mixtral-8x7B 66.98→66.33、Phi-3-mini 66.62→66.33 等）。
- **胜率与成本**（§4.2、§5、附录 A）：在 LLaMA-V1/V2/Mistral-7B 的 W4G-1/W4G128/W3G128/W2G128 共 32 组中，SignRound 对 GPTQ 胜 30/32、对 OmniQuant 29/32、对 AWQ 27/32、对 HQQ 15/16；W2G128 上的绝对平均准确率提升为 6.91%–33.22%。量化 LLaMA-V2-70B 约需 **2.5 小时**。对照方法是被论文**配置过**的版本（GPTQ 开 act-order/true-sequential、OmniQuant 关 LET 且 70B 只用 128 条样本）。Limitations：2-bit 超低比特下与原模型仍有明显差距，非均匀量化与混合精度留作 future work。

## 不能推出的结论

- `Ours*` 是 8 组超参里最好的一组（70B 只试少数选项），不能与固定超参的 `Ours` 混比，也不能忽略"每组都要跑一遍"的搜索成本。
- 只做 weight-only、非对称、block 内线性层量化；不能推断 KV cache、W4A4 激活量化或 embedding/lm-head 的处理效果。源码里被注释掉的 KV cache 段落与 future work 都不构成实验证据。
- 主指标是 11 个 0-shot 任务平均准确率，论文自己认为 PPL 受 outlier 影响、不宜作主判据；附录 C 也用 PPL 做了正面声明（称 124 个场景里 83 个优于其它方法），引用时不能把两种口径互相替代。
- 对照方法是被论文配置过的版本，不能代表它们各自的最优表现，也不能当作对其全部实现的普适排名。
- "在多数场景更优"是均值口径：Table 5 就显示 W2G128 的 LLaMA-V2-13B 上"只调舍入"（54.41）高于联合（53.46），单个配置不能想当然认为 SignRound 一定更好。
- 2-bit 仍与原模型有明显差距；"200 步完成量化"指该方法的调参成本，不含校准数据准备、评测与部署开销。
