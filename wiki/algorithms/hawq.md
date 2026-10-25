---
{
  "id": "algorithm-hawq",
  "title": "HAWQ：用 Hessian 敏感度选择混合精度",
  "type": "algorithm",
  "tags": [
    "mixed-precision",
    "hessian",
    "sensitivity",
    "qat"
  ],
  "aliases": [
    "HAWQ-I",
    "HAWQ-II",
    "HAWQ-V2",
    "Hessian AWare Quantization"
  ],
  "related": [
    "algorithm-modelopt-autoquantize",
    "guide-architecture-transfer"
  ],
  "sources": [
    "paper-hawq2",
    "source-hawq2",
    "code-hawq"
  ],
  "confidence": "source-reported",
  "evidence_scope": "mixed",
  "updated": "2026-10-24"
}
---

# HAWQ：按曲率敏感度分配混合精度

## 一句话理解

本页以 HAWQ-V2 为主：它用任务损失的平均 Hessian trace 与量化扰动估计层敏感度，在模型大小预算下选择不同位宽，再进行量化感知微调。它负责精度分配，论文准确率不能当作无训练 PTQ 结果。

## 核心机制

对参数块 $W_i$，以 $\bar h_i=\mathrm{Tr}(\nabla^2_{W_i}L)/n_i$ 表示平均曲率，候选配置的扰动分数为：

$$\Omega=\sum_i\bar h_i\|Q_{b_i}(W_i)-W_i\|_2^2.$$

平均 trace 通过 Hessian-vector product 和 Hutchinson 随机估计获得，不需构造完整 Hessian。这是带有局部驻点及扰动方向假设的敏感度代理，不是全模型真实损失的精确预测（与 GPTQ 层输出重构 Hessian 的差异是本库对照，不在本页 sources 内）。

## 基本流程

1. 固定模型、任务损失、代表数据与候选位宽。
2. 估计各块平均 trace，用相对敏感度缩小分配空间。
3. 计算候选的量化扰动，在模型大小与扰动的 Pareto 前沿选配置。
4. 按配置量化感知微调，评估最终任务质量。

## 适用条件与结果边界

- 论文直接证据是 ImageNet CNN 和 COCO 检测模型；LLM/MoE 迁移需另行验证。
- HAWQ-I、V2、V3 的指标和部署证据不同；本页不将当前仓库 V3 的结果归给 V2。
- 表格中的 `MP` 指混精量化（mixed-precision），标注的数字是所选混精配置的最低位宽，不是平均位宽或全网固定格式。
- 模型大小、trace 估计耗时、量化后推理速度是不同指标。
- 工程建议：模型变换、数据或量化器改变后重新检查敏感度；还需确认后端支持所选混精配置。

## Agent 如何使用

当需要在大小预算下分配各层位宽时，把 HAWQ-V2 作为候选。需要公式假设与会议版结果时读 [论文来源](../../sources/papers/paper-hawq2.md)；量化器和预印本差异读 [LaTeX 来源](../../sources/papers/source-hawq2.md)；训练与版本边界读 [官方源码来源](../../sources/repos/code-hawq.md)。回答时明确有无微调、采用哪个版本，以及成本是模型大小还是实际延迟。本库未执行训练或部署实验。
