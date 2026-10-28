---
{
  "id": "algorithm-skvq",
  "title": "SKVQ：滑动窗口 + 通道重排的极低比特 KV cache 量化",
  "type": "algorithm",
  "tags": [
    "kv-cache",
    "activation",
    "quantizer",
    "clipping",
    "mixed-precision"
  ],
  "aliases": [
    "SKVQ",
    "滑动窗口 KV cache 量化",
    "channel reorder 量化",
    "KV cache 2bit",
    "KV cache 1.5bit"
  ],
  "related": [
    "algorithm-kvquant",
    "algorithm-baselines",
    "foundation-quantization-contract",
    "guide-search-space"
  ],
  "sources": [
    "paper-skvq"
  ],
  "confidence": "source-reported",
  "evidence_scope": "source-reported",
  "updated": "2026-10-28"
}
---

# SKVQ：滑动窗口 + 通道重排的极低比特 KV cache 量化

## 一句话理解

SKVQ 把 KV cache 压到极低比特：先用**通道重排**把分布相似的通道编入同一量化组，再做**分组裁剪的动态量化**，同时用**滑动窗口**把最近 `w` 个 token 的 KV 保持全精度、并用 attention sink 规则额外保住开头几个 token。它只压 KV cache，不含权重；论文把它做到 Key 2-bit / Value 1.5-bit（group 64，Figure 4）仍无明显掉点。

## 核心机制

1. **通道重排（channel reorder）**：利用置换不变性重排 KV cache 的通道（Key 与 Value），让分布相近的通道进入同一组共享 scale/zero-point，避免离群通道把整组范围拉大。
2. **分组裁剪动态量化**：在重排后的组内做 clipping，缓解“离群是相对的、非离群通道随 token 波动大”的问题。
3. **滑动窗口**：只保留最近 `w` 个 token 的 KV 全精度，滑出窗口的 token 才量化；量化误差不会累积到最近窗口。
4. **重要 KV 过滤**：把开头若干 token（attention sink）保留高精度；论文讨论过按累计注意力分数保留 heavy hitters，但明确未实现（收益不显著、FlashAttention 下难以取注意力分数），最终采用固定位置的 sink 方案，并把过滤规则留作接口。
5. **混合位宽**：Key 2-bit、Value 1.5-bit 的组合。

## 基本流程

1. 离线校准（论文：Wikitext2-v1 训练集 256 条、长度 4096）确定通道重排顺序与分组裁剪参数；scale/zero-point 可用 FP8(E4M3) 存储。
2. prefill：先用全精度 KV 算注意力，再量化 cache 并保留最后 `w` 个 token 全精度。
3. decode：每步只处理滑出窗口的那个 token，按过滤规则决定是否保留高精度。

## 适用条件

- 面向长上下文 LLM 的 KV cache 显存/带宽优化，尤其是需要极低比特的场景。
- 需要离线校准（论文：256 条 × 4096）与运行时 KV 量化/反量化实现；论文未报告自定义 kernel；窗口大小与过滤规则需按模型/长度确认。
- 只压 cache、不压权重；论文未做与权重量化的联用实验，「可与权重量化分别处理」是工程假设（对照 KVQuant 的联用结论有 Table 5 直接支撑）。

## 结果边界

- 论文（LongBench，Table 1，平均组大小 128、K/V 2-bit、窗口 128）：SKVQ 平均分接近 FP16 并优于 RPTQ/SmoothQuant/RTN/KIVI（如 Llama-2-7B-chat 平均 37.50 对 FP16 38.50；Mistral-7B-Instruct 46.23 对 48.66）。
- 摘要中“Key 2-bit / Value 1.5-bit”的极低比特结果来自 Figure 4（group size 64，仅 Llama-2-7B-chat 与 Mistral-7B-Instruct-v0.2），与 Table 1 的 K/V 2-bit、平均组大小 128 不是同一配置。
- 内存/延迟为 LLM-Viewer roof-line 分析（LLaMA-7B，A100-80G）：单卡支持 1M 上下文，batch 128 / 200k 序列理论约 7× 解码加速。
- 未复现、未在目标后端核验；结论不能外推到 MLA / 线性注意力 / MoE / DiT / dLLM；“1M 上下文”“7× 加速”依赖硬件与 batch/序列假设。

## Agent 如何使用

处理“KV cache 想压到 2-bit 甚至更低、以及长上下文掉点”时，先读本页，再按需展开：

| 需要确认的内容 | 继续读取 |
| --- | --- |
| 通道重排、分组裁剪、窗口与过滤规则、Table 1/6 | [`paper-skvq`](../../sources/papers/paper-skvq.md) |
| 同为 KV cache 量化、非均匀数据类型与稀疏路线 | [KVQuant](kvquant.md) |
| RTN 与分组量化的数值口径 | [量化契约](../foundations/quantization-contract.md)、[基线](baselines.md) |
| cache 量化在搜索空间中的位置 | [量化搜索空间](../guides/search-space.md) |

回答时要区分“重排+裁剪”“滑动窗口/过滤”“混合位宽”三类贡献，并把 roof-line 分析值与真实后端端到端收益分开陈述。

## 详细来源

- [SKVQ 原论文来源页](../../sources/papers/paper-skvq.md)
