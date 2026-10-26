---
{
  "id": "algorithm-ostquant",
  "title": "OSTQuant：正交与缩放的可学习等价变换",
  "type": "algorithm",
  "tags": [
    "equivalent-transformation",
    "rotation",
    "scaling",
    "learned-transform"
  ],
  "aliases": [
    "OSTQuant",
    "QSUR",
    "正交缩放变换",
    "可学习变换"
  ],
  "related": [
    "algorithm-quarot",
    "algorithm-spinquant",
    "algorithm-smoothquant",
    "algorithm-flatquant",
    "guide-search-space"
  ],
  "sources": [
    "paper-ostquant"
  ],
  "confidence": "source-reported",
  "evidence_scope": "source-reported",
  "updated": "2026-10-26"
}
---

# OSTQuant：正交与缩放的可学习等价变换

## 一句话理解

OSTQuant 把"缩放类"与"旋转类"等价变换统一在一个可学习框架下：每个变换对由一个对角缩放和一个正交矩阵组成，用梯度在 Stiefel 流形上端到端优化，目标是让权重与激活的分布在整个量化空间里被更充分地利用（QSUR）。忽略量化时前向完全等价，优化后变换可并进权重，部署不增加参数。

## 核心机制

- **QSUR**：以数据超体积与量化超立方体体积之比度量"量化空间利用率"；论文据此解释为什么 scaling 方法对离群/均值敏感、rotation 方法在维度大时更能压低离群，并推出使 QSUR 最大的变换形式 `T = c·Λ^{-1/2}Q^T`。
- **等价变换对** `T = ΛO`：`Λ` 可学习对角缩放，`O` 单位正交；`O` 用 RiemannAdam 等一阶优化器在 Stiefel 流形上优化；每个 block 学四对，残差通路的全局正交矩阵 `R_res` 可跨 RMSNorm 传播。
- **WOMI 初始化**：对残差通路线性层的权重协方差做特征分解得到初始化正交矩阵，再配合归一化 Hadamard；缩放矩阵初始化为单位阵。
- **KL-Top 损失**：小校准集下只对 top-k 预测概率算 KL，避免直接最小化 CE 造成的过拟合。

## 基本流程

1. 用 WOMI 初始化正交矩阵、单位阵初始化缩放矩阵。
2. 在冻结权重的前提下，以量化网络输出与浮点输出的任务损失为目标优化变换参数（含 KL-Top）。
3. 把学到的 `O`、`Λ` 合并进相邻权重，得到部署模型。
4. 在合并后的模型上做常规量化（论文配置覆盖 W4A16 与 W4A4KV4）。

## 适用条件

- 论文验证对象是 LLaMA-1/2/3 家族；变换依赖结构的等价关系（残差通路、head 布局、RoPE、RMSNorm），换架构需重新证明等价性。
- 需要可导优化与校准数据；KL-Top 需要模型输出概率。
- 变换可融合进权重，但**量化本身**与**后端 kernel 支持**不在该方法范围内。

## 结果边界

- 论文报告（Table 2，LLaMA-3-8B，`0-shot 9 Avg.`）：FP 68.09；W4A16KV16 OSTQuant 67.80（QuaRot 67.27、SpinQuant 66.54）；W4A4KV4 OSTQuant 65.37（QuaRot 61.38、SpinQuant 64.10）。摘要称 W4-only 保留 99.5% 浮点精度、W4A4KV4 相对当时 SOTA 缩小 32% 差距；另称 W4A4KV4 维持至少 96% 原始性能。其中 99.5% 是摘要口径，逐模型核算并非都达到。
- QSUR 是论文自提度量，与精度是相关性，不能当精度预测器。
- KL-Top 会让 Wiki PPL 变差但零样本分数变好（LLaMA-2-7B：PPL 5.38→5.94，Arc-C 42.41→44.62），说明不能用 PPL 单指标判断。
- "部署无额外开销"只指变换可离线融合进权重；Q/K 输出仍需在线做一次 Hadamard（同 QuaRot），也不含量化误差、不保证后端支持。
- 结果限于 LLaMA 家族，不能外推到 MoE/DiT/dLLM/VLM；当前 Wiki 未复现。

## Agent 如何使用

当需要在旋转（QuaRot/SpinQuant）与缩放（SmoothQuant）之间取舍、或想引入可学习变换时读本页，再按需展开：

| 需要确认的内容 | 继续读取 |
| --- | --- |
| QSUR 推导、Eq.9–13、Table 1–2 数值与未评测列 | [`paper-ostquant`](../../sources/papers/paper-ostquant.md) |
| 旋转类方法的插入位置与假设 | [QuaRot](quarot.md)、[SpinQuant](spinquant.md)、[FlatQuant](flatquant.md) |
| 缩放类方法如何迁移激活量化难度 | [SmoothQuant](smoothquant.md) |
| 组合顺序与统计失效 | [算法组合](../recipes/composition.md) |

回答时要区分"等价变换（不改浮点结果）"与"量化误差（真正掉点来源）"，并说明该变换只对已验证的 LLaMA 结构成立。

## 详细来源

- [OSTQuant 原论文来源页](../../sources/papers/paper-ostquant.md)
