---
{
  "id": "algorithm-llmint8",
  "title": "LLM.int8()：8-bit 矩阵乘法的向量级量化与混精分解",
  "type": "algorithm",
  "tags": [
    "activation",
    "quantizer",
    "mixed-precision",
    "outlier",
    "deployment"
  ],
  "aliases": [
    "LLM.int8()",
    "8-bit 矩阵乘法",
    "向量级量化",
    "混精分解",
    "离群特征维拆分"
  ],
  "related": [
    "algorithm-smoothquant",
    "algorithm-gptq",
    "algorithm-awq",
    "foundation-number-formats",
    "foundation-quantization-contract",
    "guide-deployment-evidence"
  ],
  "sources": [
    "paper-llmint8"
  ],
  "confidence": "source-reported",
  "evidence_scope": "source-reported",
  "updated": "2026-10-28"
}
---

# LLM.int8()：8-bit 矩阵乘法的向量级量化与混精分解

## 一句话理解

LLM.int8() 把 transformer 的前馈层与注意力投影层做成 Int8 矩阵乘法：绝大多数值用"每个内积一组 scale"的**向量级量化**，把极少数量级 ≥6 的**离群特征维**单独用 FP16 相乘再相加，从而在 175B 规模也做到零退化。它是 W8A8 的经典基线，量化对象是权重+激活，不压 embedding/norm/bias。

## 核心机制

1. **向量级量化（Eq.7）**：把矩阵乘视为独立内积序列，给 `X` 每行、`W` 每列各一个归一化常数；Int8×Int8 在 Int32 累加，再用常数外积 `c_x⊗c_w` 反量化。相比"整张量一个 scale"，离群只影响所在的那一组内积。
2. **混精分解（Eq.8）**：检测 `X` 中量级超过阈值 `α`（论文取 `α=6.0`）的特征维，记为集合 `O`；`O` 对应的列（与 `W` 对应行）用 FP16 相乘，其余用 Int8 相乘，最后在 FP16 相加。13B 内 `|O|≤7`、仅约 0.1% 额外内存，>99.9% 的值仍走 8-bit。
3. **为什么必须拆**：离群特征极稀疏但极关键——它在 6.7B 附近"相变"，出现在所有层与约 75% 的序列维，每条序列约 15 万个离群值却集中在约 6 个特征维；把这些维置零可让 PPL 上升 600–1000%，而随机置零同等数量特征几乎无影响。

## 基本流程

1. 逐层（FFN 与 attention 的投影层）离线把 `W` 按列 absmax 量化到 Int8。
2. 推理时按行 absmax 把 `X` 量化到 Int8。
3. 扫描 `X` 各特征维的最大量级，量级 ≥ `α` 的维进入集合 `O`。
4. `O` 的列走 FP16 相乘；其余走 Int8×Int8→Int32 并用 scale 外积反量化；两部分在 FP16 相加作为该层输出。
5. 全模型逐层执行；embedding、norm、bias 保持 FP16。

## 适用条件

- 面向自回归 LLM 的 **W8A8** 推理加速与显存减半；对已出现系统性离群特征的大模型（≥6.7B）收益最明确。
- 需要后端支持 Int8 矩阵乘与 FP16 混合相加（论文用 CUDA/bitsandbytes，并有 HuggingFace 集成）。
- 只保留 8-bit：不解决 4-bit 及以下；不量化 attention function 与 KV cache 等投影层之外的张量。

## 结果边界

- 论文报告（Table 1，C4 验证 PPL，NVIDIA A40）：13B 上 FP32 12.45，LLM.int8() (absmax/zeropoint) 均为 **12.45**（完全恢复）；同期基线 Int8 absmax 19.08、zeropoint 13.94、absmax vector-wise 16.48。2.7B 上 FP32 14.43、LLM.int8() 14.44。
- zeroshot：论文称在 OPT 125M–175B 上保持 FP16 水平（Figure 1）。
- 显存约减半；对等效 175B 的大矩阵乘约 2× 加速，BLOOM-176B 端到端运行时保持；**<6.7B 可能因量化开销变慢**。
- 质量结论基于 fairseq 稠密 AR transformer / OPT 与 C4、EleutherAI harness；端到端运行时另测 BLOOM-176B（附录 D）。`α=6.0` 等为经验设定。未复现、未在目标后端核验。

## Agent 如何使用

处理"8-bit 权重+激活怎么量化、离群通道怎么处理、为什么大模型 8-bit 会崩"时先读本页，再按需展开：

| 需要确认的内容 | 继续读取 |
| --- | --- |
| 向量级量化/混精分解公式、Table 1 与设置 | [`paper-llmint8`](../../sources/papers/paper-llmint8.md) |
| 把离群从激活迁到权重的另一条路 | [SmoothQuant](smoothquant.md) |
| 权重-only 的误差补偿与缩放方法 | [GPTQ](gptq.md)、[AWQ](awq.md) |
| 数值格式与量化契约 | [数值格式](../foundations/number-formats.md)、[量化契约](../foundations/quantization-contract.md) |
| 显存/延迟证据的读法 | [部署证据](../guides/deployment-evidence.md) |

回答时要区分"向量级量化（per-inner-product scale）"与"混精分解（离群维 FP16 旁路）"两类贡献，并说明它只覆盖 8-bit、只量化投影层，不能推出更低比特或整模型量化的结论。

## 详细来源

- [LLM.int8() 原论文来源页](../../sources/papers/paper-llmint8.md)
