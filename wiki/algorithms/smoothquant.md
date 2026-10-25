---
{
  "id": "algorithm-smoothquant",
  "title": "SmoothQuant：在权重与激活之间迁移量化难度",
  "type": "algorithm",
  "tags": ["equivalent-transformation", "scaling", "activation", "calibration"],
  "aliases": ["SmoothQuant", "平滑量化", "激活离群值"],
  "related": ["algorithm-gptq", "algorithm-omniquant"],
  "sources": ["paper-smoothquant", "repo-smoothquant"],
  "confidence": "source-reported",
  "evidence_scope": "mixed",
  "updated": "2026-10-24"
}
---

# SmoothQuant：在权重与激活之间迁移量化难度

## 一句话理解

SmoothQuant 是面向 LLM 的 W8A8 后训练量化方法。它利用激活通道的幅度统计，把部分激活量化难度迁移到权重，使激活更适合 INT8 GEMM；它本身不是 weight-only 量化，也不是 QAT。

## 核心机制

对线性层 `Y = XW`，SmoothQuant 使用逐输入通道的等价缩放：

```text
XW = (X D⁻¹)(D W)
```

其中 `D` 根据校准数据中的激活范围和权重范围计算，迁移强度由 `α` 控制。浮点变换本身保持等价，分别量化变换后的激活和权重后才产生误差。

## 基本流程

1. 用校准数据统计激活和对应权重通道的范围。
2. 根据 `α` 计算通道 scale。
3. 将逆 scale 吸收到兼容的上游归一化或线性层，并将正 scale 乘到下游权重。
4. 在变换后的模型上确定静态/动态量化器和量化步长。
5. 使用支持 W8A8 的后端执行量化计算。

共享同一输入的多个消费者必须同步变换；论文指出输入来自残差 add 时需要对残差分支额外缩放，其它分叉与非线性消费者论文未讨论。

## 适用条件

- 原论文主要验证 INT8 W8A8 training-free PTQ。
- 论文的 O1/O2/O3 配置使用相同的权重粒度、不同的激活粒度或静态性：O1 为 per-tensor weight + per-token dynamic activation，O2 为 per-tensor weight + per-tensor dynamic activation，O3 为 per-tensor weight + per-tensor static activation；不能混成一个统一默认配置。
- 论文常用 Pile 校准数据和 512 条随机句子；`α=0.5`、`α=0.75` 是特定模型设置，不是普适最优值。
- 目标模型需要有可吸收缩放的计算图；具体 RMSNorm、LayerNorm、Q/K/V、FFN 和 MoE 映射必须逐模型检查。

## 结果边界

- 论文证据主要是 W8A8 INT8，不支持把 SmoothQuant 直接当作 W4A4、weight-only 或 KV cache 量化方法。
- 论文性能绑定 A100、指定模型、序列长度和 PyTorch/FasterTransformer 后端，不能直接推广到其他硬件或 runtime。
- 平滑只解决量化前的范围分配，不保证目标模型最终质量，也不自动保证真实 INT8 kernel 命中。
- 论文只在附录 A 定性认为 weight-only（GPTQ）与 W8A8 "somewhat orthogonal"、未来可整合以争取 W4A4，没有组合实验；组合会改变输入坐标与权重、需要重新校准属于工程推断。
- 当前 Wiki 没有运行 SmoothQuant 量化或性能实验。

## Agent 如何使用

首轮读取本页即可判断目标是否需要 W8A8 的激活范围迁移。需要精确证据时再展开：

| 需要确认的内容 | 继续读取 |
| --- | --- |
| 公式、O1/O2/O3、α 设置、论文结果和性能条件 | [`paper-smoothquant`](../../sources/papers/paper-smoothquant.md) |
| 缩放吸收位置、Llama/Mixtral 分组和源码行为 | [`repo-smoothquant`](../../sources/repos/repo-smoothquant.md) |
| 目标模型是否可安全融合、是否命中 INT8 kernel | 目标模型计算图、量化器和 runtime 独立验证 |

回答 SmoothQuant 相关问题时，必须说明 W/A 对象、激活粒度、静态/动态范围、α、校准数据和后端条件。

## 详细来源

- [SmoothQuant 原论文来源页](../../sources/papers/paper-smoothquant.md)
- [SmoothQuant 官方源码来源页](../../sources/repos/repo-smoothquant.md)
