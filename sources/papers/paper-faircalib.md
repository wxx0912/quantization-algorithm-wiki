---
{
  "id": "paper-faircalib",
  "title": "FAIR-Calib: Frontier-Aware Instability-Reweighted Calibration for Post-Training Quantization of Diffusion Large Language Models",
  "category": "paper",
  "url": "https://arxiv.org/abs/2606.06547",
  "version": "v2（2026-07-19）",
  "version_kind": "paper-version",
  "retrieved_at": "2026-10-28",
  "read_form": [
    "html"
  ],
  "read_locator": "摘要、§1 引言、§3.1–3.4（Eq.1–10）、§4.1–4.5 理论（Lemma 4.1–4.6、Prop.4.4）、§5.1–5.4、Table 1–4、§6 结论",
  "topic": "mixed-architecture",
  "excerpt_sha256": null
}
---
## 这份来源是什么

FAIR-Calib 论文（arXiv 2606.06547，v2），研究扩散大语言模型（dLLM）的 W4A4 训练后量化。它提出 dLLM 解码会**不可逆地提交（commit）** token，产生"stability lag"：低位量化容易在写入前沿翻转临界决策，翻转被永久锁定并沿后续去噪步放大。方法分两阶段——Stage I 用全精度教师探测出"前沿感知 + 掩码期可靠性"的位置先验 `w̄`，Stage II 用该先验做**离策略的静态 teacher-forcing 加权隐状态 MSE 校准**。在 LLaDA 与 Dream 上评估。

它是 `algorithm-faircalib` 的一手依据。与同族来源分工：`paper-dllmquant` 从校准采样与权重补偿切入，`paper-starquant` 从激活变换与注意力补偿切入，本篇从**校准权重（哪些位置更重要）**切入；三者都属 dLLM 的 W4A4 路线。

范围边界：主结果 W4A4，PTQ；采用 FlatQuant 的逐层可学习仿射变换，对称量化（`z=0`）；不含微调。附录 E 另有 W3A4 与 W8A8（QDrop-style）的泛化实验，本页未纳入。

## 已读定位与阅读形式

- 阅读形式：arXiv HTML 渲染版（`arxiv.org/html/2606.06547v2`），正文与表格按 HTML 结构解析；未读 LaTeX 源码或 PDF。
- 定位：§3.1 预备（Eq.1–8）、§3.2 总览、§3.3 前沿感知时间×位置权重（Eq.9）、§3.4 离策略加权隐状态 MSE（Eq.10）、§4.1–4.5 理论（Lemma 4.1–4.6、Prop.4.4）、§5.1 设置、§5.2 Table 1–2、§5.3 Table 3–4、§5.4 诊断、§6 结论。附录 A–F 未逐条读（E 含 W3A4/W8A8 泛化实验、F 为效率分析）。

## 关键事实

- **问题定义**（§1）：dLLM 反复对所有位置预测分布、并把一部分 mask 位置写成具体 token（commit），写入不可逆；即便全精度，许多位置也存在"稳定性滞后"`δ_lag`（commit 后 top-1 预测仍持续变化）。量化扰动一旦在写入前沿翻转临界决策，错误就被锁定并沿后续步放大（Figure 1 的 `mse_prob` 逐步上升）。
- **Stage I：教师探测**（§3.2–3.3，Eq.9）：在**随机 commit** 的教师 rollout 上累积位置权重
  `w_i ← w_i + λ0(t)·1{i∈Ĉ_t} + λ1·c̃_{t,i}·1{i∈M(S_t)}`，
  其中 `Ĉ_t` 是本步实际写入前沿（frontier-hit 指示），`c̃_{t,i}` 是掩码位置的教师"锐度/可靠性"分数（行归一化的 token 概率/负熵/margin 之一），`λ0(t)` 采用 **early-boost** 调度（越早的 commit 影响越多后续步）。权重按步累加，再对齐到末尾 `K`（如 256）的生成窗口，窗口外给一个小的 floor。
- **Stage II：离策略静态校准**（§3.4，Eq.10）：不用 end-to-end 扩散 rollout，而是喂**完全可见的真实 token**，逐层最小化加权的隐状态 MSE：`argmin_θℓ E[Σ_i w̄_i ‖h^q_{ℓ,i}−h^⋆_{ℓ,i}‖²]`，`θ≤ℓ` 表示更浅层已校准并冻结。
- **理论**（§4，Prop.4.4、Lemma 4.6）：在模型无关随机 commit 与温和平滑假设下，最终输出 KL `KL(μ⋆‖μ^q)` 有"时间×位置"可加上界，且在 Assumption 4.3（单调掩码、提交集与掩码模式一一对应）下**只有被 commit 的位置贡献**；每个 token 的 KL 可由 logit 平方误差、再由隐状态平方误差控制——这为加权隐状态 MSE 这个代理目标提供了依据。
- **设置**（§5.1–5.3）：W4A4，模型 LLaDA 家族与 Dream 家族；基线 RTN/QuaRot/FlatQuant 在同等设置下比较；默认校准序列长度 **1024**（比基线更短）；Stage I 探测预算默认 **N_probe=512**。10 项评测：PIQA、BoolQ、WinoGrande、ARC-E、ARC-C、HellaSwag、TruthfulQA-MC2、MMLU、HumanEval、GSM8K。
- **主结果**（Table 1，LLaDA 家族，10 项平均）：LLaDA-Base FP 62.12、RTN 49.13、QuaRot 58.78、FlatQuant 59.37、FAIR-Calib **61.09**；LLaDA-Instruct FP 73.81、RTN 66.83、QuaRot 71.34、FlatQuant 71.38、FAIR-Calib **72.40**；LLaDA-1.5 FP 73.53、RTN 67.40、QuaRot 71.05、FlatQuant 71.94、FAIR-Calib **72.75**。
- **主结果**（Table 2，Dream 家族，10 项平均）：Dream-Base FP 70.01、RTN 42.00、QuaRot 57.49、FlatQuant 62.08、FAIR-Calib **64.64**；Dream-Instruct FP 71.01、RTN 45.76、QuaRot 61.14、FlatQuant 63.98、FAIR-Calib **66.66**。
- **消融**（§5.3，Table 3，Dream-Base，10 项平均）：baseline 61.76、仅 frontier-hit 63.12、仅 masked-stage reliability 62.89、FAIR-Calib **64.64**（两信号互补）。Table 4(a) 探测预算 128/256/512/1024 → 63.15/63.56/64.64/64.63；Table 4(b) `λ0(t)` 调度：无 62.89、uniform 63.21、late-boost 62.58、early-boost **64.64**。
- **机制诊断**（§5.4）：teacher-forced commit-step flips 由 2.9±1.4 降到 1.9±0.9（N=32 样本）；位置先验 `w̄` 在 GSM8K-CoT 与 WikiText2 间有较高秩一致性（Spearman），作者称 masked-stage reliability 门对跨语料可迁移性重要，而 frontier-hit-only 变体跨语料几乎不相关。

## 不能推出的结论

- 本页只读正文与所列表格，未复现实验、未运行；准确率与诊断数字均为论文报告值。
- 结论限于 LLaDA 家族与 Dream 家族两类**掩码去噪** dLLM、W4A4 与论文的校准设置；不能外推到 AR LLM、块扩散模型、图像/视频 DiT 或其它 dLLM。
- 主结果只报告 W4A4；附录 E 另有 W3A4 与 W8A8（QDrop-style）结果，但本页未纳入，不能据此认为 W3A3/W2A2 或 weight-only 有效。
- 理论界在"模型无关随机 commit + 温和平滑"假设下成立；真实推理用模型相关 commit 策略（Dream 熵驱动、LLaDA 置信度驱动），论文承认会多出一个 policy-shift 项，界不直接等于实际误差。
- `K=256` 窗口、`λ1`、`N_probe=512`、early-boost 调度、可靠性分数选择等为经验设定；换模型/数据集需重确。
- 论文未给出可核验的官方代码来源页；本库无对应 repo 来源页，不能核对实现细节。
