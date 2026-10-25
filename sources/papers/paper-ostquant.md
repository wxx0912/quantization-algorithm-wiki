---
{
  "id": "paper-ostquant",
  "title": "OstQuant: Refining Large Language Model Quantization with Orthogonal and Scaling Transformations for Better Distribution Fitting",
  "category": "paper",
  "url": "https://arxiv.org/abs/2501.13987",
  "version": "v1（2025-01-23）",
  "version_kind": "paper-version",
  "retrieved_at": "2026-10-25",
  "read_form": [
    "html"
  ],
  "read_locator": "摘要、§1、§2、§3（Lemma 1、Definition 1、Eq.1–8）、§4.1（Eq.9–11、WOMI、Inter/Intra-Block Learning）、§4.2 KL-Top（Eq.12–13、Table 1）、§5.1 总结果（Table 2）",
  "topic": "transforms",
  "excerpt_sha256": null
}
---
## 这份来源是什么

OSTQuant 原始论文（arXiv 2501.13987，v1），研究 LLM 的 PTQ。它先提出 Quantization Space Utilization Rate（QSUR）度量数据在量化空间中的利用率，并据此把"缩放类"与"旋转类"等价变换统一分析，再给出可学习的正交 + 缩放变换（OSTQuant）与 KL-Top 损失，在 LLaMA 系列上评估。

它是 `algorithm-ostquant` 的一手依据，也是 `guide-search-space` 中"等价变换"类别与 `recipe-composition` 相关讨论的补充材料。

范围边界：**PTQ**；覆盖 W4-only、W4A4KV4 等配置；模型为 LLaMA-1/2/3 家族；变换在部署时可融合进权重。

## 已读定位与阅读形式

- 阅读形式：arXiv HTML 渲染版（`arxiv.org/html/2501.13987v1`），正文与大型结果表按 HTML 结构解析；未读 LaTeX 源码或 PDF。
- 定位：§3 QSUR 定义与"线性变换对 QSUR 的影响"（Eq.1–8）、§4.1 等价变换对与 WOMI 初始化（Eq.9–11；Eq.9 是前向重写式）、§4.2 KL-Top（Eq.12–13、Table 1）、§5.1 总结果（Table 2，8 个模型列）。

## 关键事实

- **QSUR**（§3，Definition 1、Eq.1–7）：定义为数据超体积 `V_X` 与其量化超立方体体积 `V_S` 之比，用协方差特征值表示；论文据此主张 scaling 类方法（如 SmoothQuant）对离群与均值偏移敏感，rotation 类方法（如 QuaRot）靠增大维度降低离群；最大 QSUR 对应 `T = c·Λ^{-1/2}Q^T`（Eq.8）。
- **等价变换对**（§4.1）：论文把变换对定义为 `T = ΛO`（正文无编号），前向重写式为 Eq.9（`y = Q(xW₁OΛ)·Q(Λ⁻¹OᵀW₂)`）；`Λ` 为可学习对角缩放、`O` 为单位正交矩阵；`O` 在 Stiefel 流形上用一阶优化器（RiemannAdam）优化；忽略量化时前向等价，优化后 `O`、`Λ` 可合并进权重，**部署无额外参数或计算**。每个 block 学四对；残差通路的全局正交矩阵 `R_res` 可跨 RMSNorm 传递（酉矩阵保范）。
- **WOMI 初始化**（§4.1）：对接收残差输入的线性层权重沿输入通道拼接后做协方差特征分解，令 `R_res = Q_W·H^T`（`H` 为归一化 Hadamard 矩阵）；out/value 投影的正交矩阵按 head 维切分后同法初始化；缩放矩阵初始化为单位阵。
- **KL-Top 损失**（§4.2，Eq.12–13）：PTQ 校准集小（约 1,000 条），直接最小化 CE 会使模型在小数据集上过拟合；改为只对 top-k 预测概率算 KL，保留更丰富语义。Table 1 显示 KL-Top 在 LLaMA-2-7B 上 Wiki PPL 由 5.38 升到 5.94，但 Arc-Easy/Arc-Challenge 由 69.87/42.41 升到 72.69/44.62——即 PPL 变差而零样本任务更好。
- **总结果**（§5.1，Table 2，`0-shot 9 Avg.`/`Wiki PPL`）：LLaMA-3-8B FP 68.09/6.14；**W4A16KV16** OSTQuant **67.80**/6.53（QuaRot 67.27、SpinQuant 66.54）；**W4A4KV4** OSTQuant **65.37**/7.29（QuaRot 61.38/8.18、SpinQuant 64.10/7.35）。摘要亦称 W4-only 保留 99.5% 浮点精度、W4A4KV4 相对当时 SOTA 缩小 32% 差距（对应 LLaMA-3-8B：FP 68.09，SpinQuant 64.10，OSTQuant 65.37）；§1 另称 W4A4KV4 维持至少 96% 原始性能（按 Table 2 核算：LLaMA-3-8B 96.0%、LLaMA-2-7B 96.9%、LLaMA-2-13B 96.7%）。

## 不能推出的结论

- 本页未读附录 A（含 QSUR 推导 A.2、WOMI/KL-Top 消融 A.3、推理开销 A.4、future work A.5），未核验其数学推导，也未复现实验；Eq.7/8 形式按 HTML 渲染记录。
- "W4-only 保留 99.5% 浮点精度"是摘要口径，按 Table 2 逐值核算并非对每个模型都成立（如 LLaMA-2-7B 的 4-16-16 为 64.37/65.21≈98.7%），不能当作每个模型都满足的保证。
- 结果只在 LLaMA-1/2/3 家族上给出，不能外推到 MoE、DiT、dLLM、VLM 或其它非 LLaMA 结构，也不能证明变换在这些结构上仍保持浮点等价。
- "部署无额外开销"指变换可离线融合进权重；不含量化本身的误差、也不保证目标后端存在对应低比特 kernel。
- QSUR 是论文自提度量，它与最终精度是相关性而非因果预测器；QSUR 高不代表任务精度一定好。
- KL-Top 在 LLaMA-2-7B 上使 Wiki PPL 变差却提升零样本分数，说明单看 PPL 会误导；不能据 PPL 判断该方法的优劣。
- Table 2 中部分单元格为 `-`（例如个别基线在 LLaMA-3-70B、LLaMA-2-70B 上的结果缺失），不能把空值当零或当作已支持。
