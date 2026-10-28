---
{
  "id": "algorithm-quipsharp",
  "title": "QuIP#：Hadamard 非相干处理 + E8 格子码本的极低比特权重量化",
  "type": "algorithm",
  "tags": [
    "weight-only",
    "rotation",
    "reconstruction",
    "quantizer",
    "ptq"
  ],
  "aliases": [
    "QuIP#",
    "QuIP# 2-bit",
    "Hadamard 非相干处理",
    "随机 Hadamard 变换",
    "E8P 码本",
    "格子码本向量量化"
  ],
  "related": [
    "algorithm-gptq",
    "algorithm-awq",
    "algorithm-quarot",
    "algorithm-omniquant",
    "foundation-quantization-contract",
    "guide-search-space"
  ],
  "sources": [
    "paper-quipsharp"
  ],
  "confidence": "source-reported",
  "evidence_scope": "source-reported",
  "updated": "2026-10-28"
}
---

# QuIP#：Hadamard 非相干处理 + E8 格子码本的极低比特权重量化

## 一句话理解

QuIP# 是权重-only 的极低比特 PTQ（每权重 ≤4 bit，主打 2 bit）：先用**随机 Hadamard 变换（RHT）**把权重打散成近似球状高斯分布（非相干处理），再用基于 **E8 格子**的向量量化码本去贴合这个球状分布，最后可选微调补偿层间误差。它把"逐权重舍入"升级为"向量量化"，在 2-bit 区间取得同类最优。

## 核心机制

1. **RHT 非相干处理（§3）**：用随机 Hadamard 变换替代 QuIP 的两因子 Kronecker，非相干理论界更好、结构化矩阵乘法由 `Θ(n√n)` 降到 `Θ(n log n)`，且 Hadamard 的 ±1 条目可免浮点乘。
2. **BlockLDLQ（§4.1）**：把 LDLQ 的线性反馈自适应舍入从"逐列标量舍入"扩展到"按块向量舍入"，使可表示权重集合从超立方体变成更贴球状分布的球。
3. **E8P 码本（§4.2）**：基于 `R⁸` 中最优单位球填充的 `E8` 格子；2-bit E8P 码本有 `2^16` 个条目但只需查 `2^8` 项表，其余 8 bit 存符号与平移，查表小、解码快。
4. **RVQ 扩比特（§4.3）**：4 bit = 用 2-bit E8P 量化两遍；3 bit = 2-bit E8P + 1-bit `E8` 码本。
5. **量化期微调（§5，可选）**：块内先微调未量化层补偿已量化层，全部量化后再微调剩余未量化参数；符号向量按实数优化以塑形权重。论文称 70B 约 50 GPU-hours，符号存 FP16 后仅增加 <0.01 bit/weight。

## 基本流程

1. 用校准集的激活估计 proxy Hessian `H=E[xxᵀ]`（论文：RedPajama 1T 6144 条、原生上下文长度），对权重 `W` 与 `H` 一起做 RHT 非相干处理（随机符号 + Hadamard）。
2. 用 BlockLDLQ（以变换后的 `H` 做线性反馈）做块级自适应舍入，向量量化步使用 E8P 码本（高比特用 RVQ 叠加）。
3. （可选）块内与整模微调未量化参数，并允许符号向量更新。
4. 推理时按 RHT + 码本查表解码；码本仅 1KiB、每条权重解码指令数很少，便于低开销服务。

## 适用条件

- 面向**权重-only**的极端压缩（2–4 bit/weight），尤其想在 2-bit 保住可用质量。
- 需要校准集估计 Hessian `H=E[xxᵀ]`（与 GPTQ 同类，**即使不开微调也要**），并离线跑 RHT/码本量化与（若启用）微调；推理后端需要支持码本查表解码（论文用 CUDA kernel）。
- 不改动激活精度：不覆盖 W4A4/W8A8、KV cache 或激活离群。

## 结果边界

- 论文报告（Table 2，WikiText2 PPL，上下文长度 2048，越低越好）：2-bit QuIP# 在 Llama-1-7B 为 **6.86**、Llama-2-70B 为 **4.16**，同档 OmniQuant 为 15.5 / 7.81；LLaMA 1/2 七个模型在 3-bit、4-bit、FP16 下的完整数字见 Source。
- 论文称 Llama 2 上 3-bit 优于"理论无损 4-bit 模型"，并称首次出现 2-bit PTQ 具备与更高比特类似的缩放趋势。
- 消融：去掉微调与格子码本后仍优于 OmniQuant/AWQ，各组件均有增益。
- 推理（Table 5–6）：2-bit 在 RTX 4090 上 2-7B 170.50 tok/s、2-70B 32.74 tok/s；HuggingFace 实现下 2-bit Llama-2-7B 106.3 tok/s vs AQLM 20.6、FP16 33.1。
- 本页只核对了 Llama 1/2 × WikiText2/C4 的正文结果；头条结果含微调；吞吐绑定实现与硬件。未复现、未在目标后端核验。

## Agent 如何使用

处理"2-bit 权重能不能不掉点、有没有比逐权重舍入更好的办法、码本/变换怎么选"时先读本页，再按需展开：

| 需要确认的内容 | 继续读取 |
| --- | --- |
| RHT、BlockLDLQ、E8P、RVQ、Table 2–6 与设置 | [`paper-quipsharp`](../../sources/papers/paper-quipsharp.md) |
| 旋转类方法的另一种路线 | [QuaRot](quarot.md) |
| 权重-only 的其它范式 | [GPTQ](gptq.md)、[AWQ](awq.md)、[OmniQuant](omniquant.md) |
| 量化契约与搜索空间 | [量化契约](../foundations/quantization-contract.md)、[量化搜索空间](../guides/search-space.md) |

回答时要区分"非相干变换（RHT）""码本/向量量化（E8P + RVQ）""量化期微调"三类贡献，并说明这是权重-only、且头条结果用了微调，不能直接等同于训练-free 的 W4A4 方案。

## 详细来源

- [QuIP# 原论文来源页](../../sources/papers/paper-quipsharp.md)
