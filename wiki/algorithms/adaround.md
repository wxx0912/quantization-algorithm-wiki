---
{
  "id": "algorithm-adaround",
  "title": "AdaRound：固定量化网格上的自适应舍入",
  "type": "algorithm",
  "tags": ["ptq", "rounding", "reconstruction", "vision"],
  "aliases": ["Adaptive Rounding", "Up or Down"],
  "related": ["algorithm-brecq", "algorithm-autoround"],
  "sources": ["paper-adaround", "repo-adaround"],
  "confidence": "source-reported",
  "evidence_scope": "mixed",
  "updated": "2026-10-24"
}
---

# AdaRound：固定量化网格上的自适应舍入

## 一句话理解

AdaRound 在量化网格已经确定后，学习每个权重应向下还是向上舍入，以降低层输出误差。它不训练原始权重，也不联合学习量化 scale；核心对象是舍入决策。

## 核心机制

对固定步长 `s`，权重可写成 floor 值加一个 0/1 舍入变量。AdaRound 用连续变量和退火正则近似优化这些离散选择，使软舍入最终接近 0 或 1：

```text
W_hat = s · clip(floor(W / s) + h(V), n, p)
```

最终目标是浮点层输出与量化层输出的重构误差，并在逐层顺序量化时使用前层已量化输入。论文指出改变量化步长（scale）会得到不同的 QUBO 问题，因此不联合优化步长与舍入；据此，改变 scale、clipping 或通道变换后也需要重新优化舍入（本库推断）。

## 基本流程

1. 固定量化位宽、scale、范围和粒度。
2. 准备无标签校准数据、教师输出和量化层输入。
3. 优化舍入变量，使层输出重构误差下降。
4. 通过退火正则将软变量硬化为向下或向上舍入。
5. 固化量化权重并继续处理下一层。

## 适用条件

- 原论文主要验证视觉 CNN 的逐层对称 W4 权重量化，激活通常保持 FP32；附带研究了 8-bit 激活。
- 需要校准数据和离线优化，但不需要标签或端到端任务微调。
- 论文默认使用预先确定的量化步长；AdaRound 本身不负责联合优化 scale。
- KV cache、LLM、MoE、DiT 和文本 dLLM 不属于原论文的直接实验范围。

## 结果边界

- 论文结果主要来自 ResNet、Inception、MobileNet 和 DeepLab 等视觉模型，不能直接作为 LLM 或其他生成模型的质量证据。
- 论文中的 W4A32、W4A8、校准图像数量和优化迭代数绑定具体模型与实验协议，不能直接外推到其他模型。
- AdaRound 的局部重构目标是近似任务损失后的层级目标，不提供全局最优性保证。
- 当前 Wiki 没有运行 AdaRound 优化、目标模型评测或 GPU benchmark。

## Agent 如何使用

首轮读取本页即可判断问题是否属于固定网格上的舍入优化。需要精确证据时再展开：

| 需要确认的内容 | 继续读取 |
| --- | --- |
| 连续松弛、正则退火、实验设置和视觉结果 | [`paper-adaround`](../../sources/papers/paper-adaround.md) |
| 维护实现的软/硬舍入流程与默认超参 | [`repo-adaround`](../../sources/repos/repo-adaround.md) |
| 目标模型和后端是否适配 | 目标模型计算图、量化器和 runtime 独立验证 |

回答 AdaRound 相关问题时，必须说明固定的量化网格、舍入对象、校准数据、优化预算和模型范围。不能把它描述成同时学习 scale 的通用量化器。

## 详细来源

- [AdaRound 原论文来源页](../../sources/papers/paper-adaround.md)
- [AdaRound 官方维护实现来源页](../../sources/repos/repo-adaround.md)
