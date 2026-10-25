---
{
  "id": "algorithm-omniquant",
  "title": "OmniQuant：联合学习 clipping 与等价变换",
  "type": "algorithm",
  "tags": ["joint-optimization", "clipping", "scaling", "block-reconstruction"],
  "aliases": ["OmniQuant", "LET", "LWC", "可学习等价变换"],
  "related": ["algorithm-smoothquant", "algorithm-gptq"],
  "sources": ["paper-omniquant", "repo-omniquant"],
  "confidence": "source-reported",
  "evidence_scope": "mixed",
  "updated": "2026-10-24"
}
---

# OmniQuant：联合学习 clipping 与等价变换

## 一句话理解

OmniQuant 是一种需要梯度优化的 block-wise PTQ 方法。它冻结原始模型权重，只学习量化 clipping 参数和可吸收的等价变换参数，以降低量化 block 的输出误差；它不是全模型 QAT，也不是 one-shot 量化。

## 核心机制

OmniQuant 包含两个主要变量：

- **LWC**：学习每个量化组的 clipping 强度，调整可表示范围；
- **LET**：学习可吸收的通道 scale/shift，使激活和权重更适合量化，同时保持浮点计算等价。

优化目标是让量化 block 的输出接近浮点 block 输出。权重保持冻结，优化的是量化相关参数。

## 基本流程

1. 缓存 block 的浮点参考输入和输出。
2. 用 SmoothQuant/OS+ 等方法初始化 LET，初始化 LWC 的范围参数。
3. 在校准数据上联合优化 LET 和 LWC。
4. 融合可吸收变换，执行最终量化，并把结果传给下一 block。
5. 对后续 block 重复处理。

LET 的 scale/shift 只能在计算图允许的位置吸收；论文排除 FFN 第二线性层等不稳定位置，具体模型仍需检查。

## 适用条件

- 论文覆盖 weight-only W2A16/W3A16/W4A16，以及部分 W4A4/W6A6 weight-activation 配置。
- 需要校准数据、反向传播和优化器状态；论文常用 WikiText-2 的 128×2048 校准样本，通常 20 epochs，W2A16 使用 40 epochs。
- 论文明确：weight-activation 下 LWC 与 LET 同时启用；weight-only 下 OPT 两者都用、LLaMA 只用 LWC（LET 对 LLaMA 增益可忽略）。
- Softmax 输出保持高精度，不能把方法理解为所有算子都低比特。

## 结果边界

- 论文的 W4A4/W6A6 质量结果不是部署 kernel 证据；真实设备部署实验主要是 weight-only。
- 论文结果绑定模型、校准集、位宽、粒度、优化步数和后端，不能直接形成普遍算法排名。
- 对比里 SmoothQuant 与 OS+ 是论文在同一 pipeline 下自行复现，LLM-QAT 属 QAT 范式；跨范式数字不能直接横比。
- 修改 LET、LWC、校准数据或分组后，旧的 clipping 参数、校准统计与质量结果不能直接复用。
- 当前 Wiki 没有运行 OmniQuant 优化、导出或 GPU benchmark。

## Agent 如何使用

首轮读取本页即可判断目标是否需要学习 clipping 或等价变换。需要精确证据时再展开：

| 需要确认的内容 | 继续读取 |
| --- | --- |
| LWC/LET 公式、优化设置、论文结果和部署条件 | [`paper-omniquant`](../../sources/papers/paper-omniquant.md) |
| 参数化、逐 block 流程、学习率和量化器实现 | [`repo-omniquant`](../../sources/repos/repo-omniquant.md) |
| 目标模型、导出器和 kernel 是否可用 | 目标模型源码、导出器和 runtime 独立验证 |

回答 OmniQuant 相关问题时，必须说明是 weight-only 还是 W/A、优化预算、校准数据、学习率来源以及是否存在真实部署证据。

## 详细来源

- [OmniQuant 原论文来源页](../../sources/papers/paper-omniquant.md)
- [OmniQuant 官方源码来源页](../../sources/repos/repo-omniquant.md)
