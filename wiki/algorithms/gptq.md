---
{
  "id": "algorithm-gptq",
  "title": "GPTQ：基于输入二阶统计的权重重构",
  "type": "algorithm",
  "tags": ["weight-reconstruction", "hessian", "weight-only", "calibration"],
  "aliases": ["GPTQ", "二阶量化", "权重误差补偿", "掉点"],
  "related": ["algorithm-quarot", "algorithm-smoothquant"],
  "sources": ["paper-gptq", "repo-gptq"],
  "confidence": "source-reported",
  "evidence_scope": "mixed",
  "updated": "2026-10-26"
}
---

# GPTQ

## 一句话理解

GPTQ 是一种权重后训练量化方法。它用代表性输入估计每层输入之间的相关性，在把权重列量化到低比特网格时，调整尚未量化的列来补偿输出误差。它主要解决 **weight-only PTQ** 的精度问题，不负责激活量化、训练或具体后端部署。

## 核心机制

对线性层 `Y = WX`，GPTQ 近似求解：

```text
min ||WX - W_hat X||²_F
```

其中 `X` 来自校准数据。对应的局部二阶统计是 `H = 2XXᵀ`。它描述的是该层输出重构目标的 Hessian，不是完整语言模型任务损失的 Hessian。

量化一列权重后，GPTQ 用二阶统计把这次量化误差传播到还未处理的列，从而减少后续输出误差。相对 OBQ，计算量的下降主要来自固定列顺序；分块更新（lazy batch-updates）解决的是访存瓶颈而非理论计算量；数值稳定性来自阻尼（平均对角值的 1%）与 Cholesky 的组合，Cholesky 另带来速度收益。

## 基本流程

1. 用代表性校准数据运行模型，收集目标线性层输入并累积 `XXᵀ`。
2. 在实现层固定位宽、粒度、scale、zero-point 和导出布局；论文主实验使用 per-row asymmetric min-max 网格，不能把其他格式当成论文统一支持。
3. 对输入统计加阻尼并做 Cholesky 预处理。
4. 按列量化权重，同时用二阶统计补偿尚未量化的列。
5. 以分块方式更新后续列，并保存量化权重与对应的 scale/zero-point（列序在实现内已还原）。

## 适用条件

- 主要用于权重后训练量化，激活通常保持浮点。
- 需要校准输入；“校准数据应覆盖目标模型的实际输入分布”是工程使用原则，不是 GPTQ 论文对所有任务的保证。
- 必须固定并记录位宽、量化粒度、scale、zero-point 与列顺序（导出布局属工程事项，本库来源未覆盖）。
- 经过 SmoothQuant 或 QuaRot 等变换后，输入坐标和权重已经改变；通常需要重新采集对应统计，这是迁移建议，不是 GPTQ 原论文实验结论。

## 结果边界

- 论文结果绑定其模型、校准集、量化网格和评测协议，不能直接当作任意模型的质量保证。
- 论文速度结果主要针对 batch-1 自回归生成的特定 kernel，不能直接推广到大 batch prefill、通用低比特 GEMM 或其他 runtime。
- GPTQ 本身不证明 W/A/KV 同时低比特，也不证明导出的权重会被目标后端以真实低比特 kernel 执行。
- 当前 Wiki 没有运行 GPTQ 量化、目标模型评测或 GPU benchmark。

## Agent 如何使用

首轮只读本页即可判断 GPTQ 是否属于候选方法；公式、默认参数与论文数字见来源页。目标模型上的质量与速度需要独立验证，不能由本页承诺。若需要精确证据，再按下面路径展开：

| 需要确认的内容 | 继续读取 |
| --- | --- |
| 数学公式、算法步骤、论文实验数字 | [`paper-gptq`](../../sources/papers/paper-gptq.md) |
| 默认参数、函数行为、量化器实现 | [`repo-gptq`](../../sources/repos/repo-gptq.md) |
| 目标模型或后端是否真的可用 | 目标模型源码、导出器和运行时的独立验证 |

回答 GPTQ 相关问题时，必须同时说明量化对象、位宽/粒度、校准数据、证据来源和尚未验证的部署条件。没有这些条件时，只能给出候选方法说明，不能承诺最终质量或速度。

## 详细来源

- [GPTQ 原论文来源页](../../sources/papers/paper-gptq.md)
- [GPTQ 官方源码来源页](../../sources/repos/repo-gptq.md)
