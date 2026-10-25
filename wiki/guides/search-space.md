---
{
  "id": "guide-search-space",
  "title": "量化搜索空间：算法类别与可搜索变量",
  "type": "guide",
  "tags": [
    "search",
    "selection",
    "format",
    "quantizer",
    "mixed-precision",
    "equivalent-transformation",
    "compensation",
    "calibration"
  ],
  "aliases": [
    "搜索空间",
    "量化算法类别",
    "可搜索变量",
    "量化格式"
  ],
  "related": [
    "guide-algorithm-selection",
    "foundation-quantization-contract",
    "recipe-composition",
    "algorithm-modelopt-autoquantize",
    "guide-architecture-transfer",
    "foundation-number-formats"
  ],
  "sources": [
    "paper-gptq",
    "paper-adaround",
    "paper-autoround",
    "paper-awq",
    "paper-smoothquant",
    "paper-quarot",
    "paper-spinquant",
    "paper-flatquant",
    "paper-hawq2",
    "code-modelopt-autoquantize",
    "paper-svdquant",
    "paper-brecq",
    "paper-omniquant",
    "paper-viditq",
    "paper-llada-architecture"
  ],
  "confidence": "inferred",
  "evidence_scope": "mixed",
  "updated": "2026-10-25"
}
---

# 量化搜索空间：算法类别与可搜索变量

把"要试什么"显式列出来，而不是对算法名做无约束笛卡尔积。本页是候选搜索的类别与变量清单；每类的事实、边界与证据以对应算法页和来源页为准，本页不自证效果。

## 搜索轴

先固定格式与部署路径，再沿下列轴取候选。一个候选通常只在一个轴上做主要改动。

| 轴 | 决定什么 | 入口 |
| --- | --- | --- |
| 数值格式与粒度 | 位宽、INT/FP、group size、per-channel/tensor、对称性 | [数值格式](../foundations/number-formats.md)、[数值契约](../foundations/quantization-contract.md) |
| 算法类别 | 固定格式上如何降低误差 | 本页类别表 |
| 混合精度分配 | 哪些模块/层保持高精度 | [HAWQ](../algorithms/hawq.md)、[AutoQuantize](../algorithms/modelopt-autoquantize.md) |
| 校准数据与统计 | 样本分布、序列长度、统计量重算时机 | [baselines](../algorithms/baselines.md)、[组合规则](../recipes/composition.md) |
| 变换与位置 | 缩放/旋转/置换/低秩作用在何处 | [QuaRot](../algorithms/quarot.md)、[SpinQuant](../algorithms/spinquant.md)、[FlatQuant](../algorithms/flatquant.md)、[SVDQuant](../algorithms/svdquant.md) |
| runtime 与部署 | 融合契约、kernel、fallback、端到端成本 | [部署证据](./deployment-evidence.md) |

## 算法类别

| 类别 | 本库页面 | 优化变量 | 可搜索的内容 |
| --- | --- | --- | --- |
| Quantizer design | [RTN 与范围校准](../algorithms/baselines.md) | 数值格式、scale、zero-point、粒度 | 位宽、group size、per-channel/tensor、对称性 |
| Calibration | [SmoothQuant](../algorithms/smoothquant.md)、[AWQ](../algorithms/awq.md) | 样本分布、采样权重、校准统计 | 校准集、序列长度、统计位置与重算时机 |
| Equivalent transformation | [QuaRot](../algorithms/quarot.md)、[SpinQuant](../algorithms/spinquant.md)、[FlatQuant](../algorithms/flatquant.md) | 变换类型、位置、参数 | 用哪种变换、放在哪、是否浮点等价 |
| Weight reconstruction | [GPTQ](../algorithms/gptq.md)、[AdaRound](../algorithms/adaround.md)、[AutoRound](../algorithms/autoround.md) | 局部目标、block、舍入、优化预算 | 目标函数、block 大小、迭代预算 |
| Mixed precision | [HAWQ](../algorithms/hawq.md)、[ModelOpt AutoQuantize](../algorithms/modelopt-autoquantize.md) | 每模块精度、bit allocation | 保留哪些模块高精度、敏感度阈值 |
| Error compensation | [SVDQuant](../algorithms/svdquant.md)（低秩）、[偏置校正](../algorithms/bias-correction.md) | 补偿分支的秩、位置、目标；有偏误差的期望 | 低秩预算、补偿位置、目标；是否校正输出偏置 |
| Joint optimization | [BRECQ](../algorithms/brecq.md)、[OmniQuant](../algorithms/omniquant.md) | 跨层/跨模块重构参数、损失 | 联合范围、损失、更新块 |
| Generation-aware | [Q-DiT](../algorithms/qdit.md)（图像 DiT）、[DLLMQuant](../algorithms/dllmquant.md)（文本 dLLM） | 生成阶段或采样策略 | timestep、mask 比例、去噪/解码轨迹、校准覆盖步数 |

类别是导航用的组织方式，不代表同一行方法可互换，也不代表跨行组合已成立。

## 覆盖缺口

- Generation-aware 已补 [Q-DiT](../algorithms/qdit.md)（图像 DiT）与 [DLLMQuant](../algorithms/dllmquant.md)（文本 dLLM）两页；`Q-Schedule` 未定位到明确论文，暂缺。图像/视频方向另有 [架构迁移](./architecture-transfer.md) 中的 ViDiT-Q 迁移假设。
- [OSTQuant](../algorithms/ostquant.md)（正交+缩放可学习变换）与 [偏置校正](../algorithms/bias-correction.md)（DFQ）已补页。
- 数值格式已有独立页 [数值格式：INT、FP8、MX 与 NVFP4](../foundations/number-formats.md)；[数值契约](../foundations/quantization-contract.md) 继续负责术语与产物定义。

## Agent 如何使用

1. 先用 [数值契约](../foundations/quantization-contract.md) 与 [部署证据](./deployment-evidence.md) 确定 W/A/KV、粒度与目标后端。
2. 用上表把掉点问题映射到类别与搜索轴，选 1–3 个候选，而不是全组合。
3. 把候选写成"观察 → 假设 → 单一主要改动 → 预期可区分结果"，见 [算法选择](./algorithm-selection.md)。
4. 组合前查 [组合规则](../recipes/composition.md)，跨架构前查 [架构迁移](./architecture-transfer.md)。
5. 本页只是候选清单；本项目未运行量化实验，不代表任何组合已可用。
