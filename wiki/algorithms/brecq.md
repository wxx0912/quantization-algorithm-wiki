---
{
  "id": "algorithm-brecq",
  "title": "BRECQ：块级重构与输出敏感度",
  "type": "algorithm",
  "tags": [
    "ptq",
    "block-reconstruction",
    "rounding",
    "fisher",
    "vision"
  ],
  "aliases": [
    "Block Reconstruction",
    "Pushing the Limit of Post-Training Quantization"
  ],
  "related": [
    "algorithm-adaround",
    "algorithm-autoround"
  ],
  "sources": [
    "paper-brecq",
    "repo-brecq"
  ],
  "confidence": "source-reported",
  "evidence_scope": "source-reported",
  "updated": "2026-10-24"
}
---

# BRECQ：块级重构与输出敏感度

BRECQ 是以网络块为重构单元的后训练量化方法：共同优化块内权重的舍入，量化激活时还学习其步长，以兼顾块内误差依赖和小校准集上的泛化。原论文证据来自视觉 CNN，不直接证明 Transformer block 也是最优重构粒度。

## 核心机制

论文用块输出误差近似量化造成的任务损失变化，并以对角 Fisher（输出梯度平方）为输出元素加权。它是二阶近似，不是精确任务 Hessian，也没有全局最优保证。权重优化采用 AdaRound 式连续松弛与舍入正则；激活量化采用 learned step-size 机制，不是继续训练所有浮点权重。

官方实现同时支持 MSE 与 Fisher 相关目标，`block_reconstruction` 默认使用 MSE。因此引用 BRECQ 不等于实际使用了 Fisher 加权（本库推断）。

## 基本流程

1. 固定教师模型、块边界、权重与激活位宽及量化粒度。
2. 缓存块输入和浮点目标输出；Fisher 模式额外获取梯度信息。
3. 优化块内权重舍入变量，最终转为硬舍入；需要激活量化时再优化量化步长。
4. 逐块完成重构；不属于任何块的首尾等层单独重构。
5. 评估模型质量，再结合实际后端导出和验证部署。

论文 Algorithm 1 与当前代码的优化阶段安排应分别查阅 Source；不能仅凭流程摘要假定两者完全一致。

## 适用条件与边界

- 论文用少量训练图像校准，主要覆盖 ResNet、MobileNet、RegNet 等 CNN；block 粒度的优势是这些实验中的结果。
- 论文同时包含 weight-only 和 W/A 量化，主分类对比的首末层保留 8-bit；低至 2-bit 并不意味着无损。
- 需要块缓存和反向优化预算；Fisher 模式还有梯度成本。
- 重构不直接产生部署 kernel；论文中的 FPGA 模拟和移动 CPU 延迟不能替代目标 GPU 实测。
- 迁移到 LLM、MoE 或 dLLM 属工程研究：需重新确定块边界、校准分布、教师输入和损失，不能沿用图像实验精度结论。

## Agent 如何使用

先把 BRECQ 作为“块级重构优化”的候选，确认是否接受梯度优化预算。需要公式、校准设置、准确率和混合精度结果时读论文 Source；需要默认参数、输入缓存或 MSE/Fisher 差异时读源码 Source。工程上改变变换、位宽或分组后应重新验证重构，不能默认旧舍入解仍适用。本项目未运行 BRECQ 实验。

## 详细来源

- [论文：方法、实验与边界](../../sources/papers/paper-brecq.md)
- [官方源码：优化阶段与目标](../../sources/repos/repo-brecq.md)
