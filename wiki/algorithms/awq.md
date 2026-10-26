---
{
  "id": "algorithm-awq",
  "title": "AWQ：激活感知的权重量化",
  "type": "algorithm",
  "tags": ["ptq", "scaling", "weight-only", "calibration", "outlier"],
  "aliases": ["Activation-aware Weight Quantization", "激活感知权重量化", "掉点"],
  "related": ["algorithm-gptq", "algorithm-smoothquant", "algorithm-autoround"],
  "sources": ["paper-awq", "repo-awq"],
  "confidence": "source-reported",
  "evidence_scope": "mixed",
  "updated": "2026-10-24"
}
---

# AWQ：激活感知的权重量化

## 一句话理解

AWQ 是一种利用输入激活统计来保护重要权重通道的 weight-only PTQ 方法。它通过等价缩放降低重要通道的相对量化误差，再搜索 clipping，主要面向 INT3/INT4 权重、FP16 激活的 LLM 压缩与推理。

## 核心机制

对线性层 `Y = WX`，AWQ 选择正对角缩放矩阵 `D`，使浮点计算保持等价：

```text
WX = (WD)(D⁻¹X)
```

它从校准输入中统计每个输入通道的激活幅度，用 `s = s_X^α` 搜索缩放强度，并在缩放后的权重上进行量化和 clipping。AWQ 关注的是缩放后模块输出误差，不是单独最小化权重误差。

“激活感知”表示用激活分布指导权重量化，不表示 AWQ 原始方法同时量化激活。

## 基本流程

1. 用代表性数据运行模型，收集目标线性层输入激活。
2. 根据激活幅度生成候选通道 scale，搜索使模块输出误差较小的缩放。
3. 将逆缩放吸收到允许吸收的相邻算子，并同步调整目标权重。
4. 在最终浮点变换上搜索 clipping，执行权重量化和打包。

缩放只能沿计算图中的等价位置移动，不能跨任意非线性或共享消费者直接传播。

## 适用条件

- 原始论文主要验证 weight-only W4A16/W3A16，激活保持高精度。
- 论文主实验使用 group size 128，INT2 组合实验使用过 group size 64；其他格式和粒度需单独核对。
- 需要校准数据、激活缓存和多次模块前向，但不需要梯度训练。
- AWQ 不自动包含 KV cache 量化，导出 packing 和后端支持必须单独确认。

## 结果边界

- 论文结果绑定其模型、Pile 校准集、位宽、分组和评测协议，不能直接推广到 W4A4、KV 量化或任意模型。
- 论文中的 INT2 结果是 AWQ 与 GPTQ 的组合证据，不能当作单独 AWQ 的普遍结论。
- TinyChat 的速度来自量化格式、packing、kernel 融合和平台实现，不能归因于 scale 搜索本身，也不能外推到其他 runtime。
- 当前 Wiki 没有运行 AWQ 量化、目标模型评测或 GPU benchmark。

## Agent 如何使用

首轮读取本页即可判断 AWQ 是否适合作为 weight-only 候选。需要精确证据时再展开：

| 需要确认的内容 | 继续读取 |
| --- | --- |
| 缩放公式、论文设置、实验数字和组合证据 | [`paper-awq`](../../sources/papers/paper-awq.md) |
| scale/clipping 搜索候选、源码入口和实现差异 | [`repo-awq`](../../sources/repos/repo-awq.md) |
| 目标模型、导出器或 runtime 是否真的支持 | 目标项目源码和运行时独立验证 |

回答 AWQ 相关问题时，必须说明权重/激活对象、位宽、group size、校准数据、是否包含 GPTQ 组合，以及性能证据的实际范围。

## 详细来源

- [AWQ 原论文来源页](../../sources/papers/paper-awq.md)
- [AWQ 官方源码来源页](../../sources/repos/repo-awq.md)
