---
{
  "id": "algorithm-quarot",
  "title": "QuaRot：旋转坐标中的低比特推理",
  "type": "algorithm",
  "tags": ["rotation", "equivalent-transformation", "activation", "kv-cache"],
  "aliases": ["QuaRot", "Hadamard", "随机旋转"],
  "related": ["algorithm-gptq", "algorithm-omniquant"],
  "sources": ["paper-quarot", "repo-quarot"],
  "confidence": "source-reported",
  "evidence_scope": "mixed",
  "updated": "2026-10-24"
}
---

# QuaRot：旋转坐标中的低比特推理

## 一句话理解

QuaRot 使用随机 Hadamard 旋转重新分布激活、权重和 KV 中的异常值，使 LLaMA 类 Transformer 可以在统一旋转坐标中进行低比特量化。旋转本身是浮点等价变换，量化仍然会产生误差。

## 核心机制

对线性层，正交矩阵 `R` 给出如下等价改写（本库归纳）：

```text
XW = (XR)(RᵀW)
```

QuaRot 把旋转吸收进相邻权重（pre-norm 下 RMSNorm 的缩放因子也随之一并折进权重），并在不能离线吸收的位置保留在线 Hadamard。它还处理 FFN、attention 的 Q/K/V、RoPE 后的 Q/K 以及 KV cache，因此不是“给权重乘一次随机矩阵”。

一般旋转不能随意穿过非线性、门控乘法、RoPE 或残差分叉；具体插入位置必须和计算图匹配。

## 基本流程

1. 在浮点模型中融合可吸收的归一化和权重变换。
2. 在 FFN、attention 和 KV 路径插入配对旋转，保持浮点前向等价。
3. 在旋转后的模型上重新收集量化统计。
4. 使用 GPTQ 或 RTN 量化权重，在线量化激活，并按指定格式写入 KV cache。
5. 部署时同时保留在线 Hadamard、反量化和 KV 读取规则。

## 适用条件

- 原论文主要验证 LLaMA 系列的 PTQ，权重、激活和 KV 可以组合为低比特配置。
- 论文主实验使用 INT4；权重通常使用 GPTQ，激活使用在线 RTN，KV 使用分组非对称量化。
- 旋转改变了输入与权重的坐标，据此需要重新计算输入统计、scale 与 GPTQ 相关量，不能复用旋转前的统计（本库工程推断）。
- 非 LLaMA 结构、不同 head 布局、MoE 分支和共享权重需要重新证明变换的等价性。

## 结果边界

- “全 4-bit”描述的是论文特定 W/A/KV 配置，不等于所有 attention 运算都以 INT4 执行；论文的 KV 在读取后会反量化并以 FP16 做点积。
- 论文性能绑定 RTX 3090、自定义 CUTLASS/FlashInfer 路径和单 Transformer block 测量，不能直接当作整模型端到端速度。
- 旋转不自动保证目标模型质量，也不证明任意后端支持在线变换或真实低比特 kernel。
- QuaRot 与 GPTQ 的组合是论文验证过的特定路径，不能据此推断与任意重构器或导出器都兼容。
- 当前 Wiki 没有运行 QuaRot 量化、目标模型评测或 GPU benchmark。

## Agent 如何使用

首轮读取本页即可判断问题是否涉及旋转、异常值迁移或 KV 量化。需要精确证据时再展开：

| 需要确认的内容 | 继续读取 |
| --- | --- |
| 旋转位置、量化配置、论文结果和性能范围 | [`paper-quarot`](../../sources/papers/paper-quarot.md) |
| 旋转吸收、伪量化流程和实现入口 | [`repo-quarot`](../../sources/repos/repo-quarot.md) |
| 目标架构、导出格式和真实 kernel 支持 | 目标模型源码、导出器和 runtime 独立验证 |

回答 QuaRot 相关问题时，必须区分浮点等价变换、量化误差、在线变换开销、KV 反量化路径和实际部署证据。

## 详细来源

- [QuaRot 原论文来源页](../../sources/papers/paper-quarot.md)
- [QuaRot 官方源码来源页](../../sources/repos/repo-quarot.md)
