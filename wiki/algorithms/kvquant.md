---
{
  "id": "algorithm-kvquant",
  "title": "KVQuant：面向百万级上下文的低比特 KV cache 量化",
  "type": "algorithm",
  "tags": [
    "kv-cache",
    "activation",
    "quantizer",
    "calibration",
    "outlier"
  ],
  "aliases": [
    "KVQuant",
    "KV cache 量化",
    "sub-4-bit KV cache 量化",
    "pre-RoPE key 量化",
    "per-channel key 量化"
  ],
  "related": [
    "algorithm-skvq",
    "algorithm-baselines",
    "foundation-quantization-contract",
    "guide-search-space"
  ],
  "sources": [
    "paper-kvquant"
  ],
  "confidence": "source-reported",
  "evidence_scope": "source-reported",
  "updated": "2026-10-28"
}
---

# KVQuant：面向百万级上下文的低比特 KV cache 量化

## 一句话理解

KVQuant 是一套把 KV cache 压到 4-bit 以下的组合技术：Key 沿通道（per-channel）量化、在 RoPE 之前量化 Key、用按层敏感度加权的非均匀数据类型（nuqX）表示分布、并按向量隔离约 1% 离群值（dense-and-sparse），再把第一个 token 保留 fp16。它不是权重量化方法，也不含微调。

## 核心机制

1. **per-channel key 量化**：Key 存在离群通道，scale/zero-point 沿通道共享可把同量级元素聚在一起；Value 仍用 per-token：per-channel 对 Value 不但无收益，反而更差——附录 G/Table 9（LLaMA-7B、seqlen 128K、int3）为 PT/PT 10.87、PC/PT 7.05、PC/PC 223 PPL（fp16 5.68）。
2. **pre-RoPE key 量化**：在旋转位置编码前量化 Key，避免 RoPE 按位置把成对通道混合后更难量化。
3. **nuqX 非均匀数据类型**：离线用对角 Fisher 加权的最小化目标（Eq.1）经 k-means 求出每层的 signposts，再按通道/ token 重缩放；因 KV cache 读取是带宽瓶颈，非线性反量化的额外算力不增加延迟。
4. **per-vector dense-and-sparse**：按通道（Key）或按 token（Value）各自设离群阈值，隔离约 1% 离群值，缩窄稠密部分的范围。
5. **attention sink 与校准分工**：第一个 token 的 KV 保留 fp16；Key 的 scale/zero-point 离线校准，Value 的离群阈值在线计算。

## 基本流程

1. 用校准集（论文：16 条 × 2K，Wikitext-2）离线求每层 nuqX 的 signposts 与 Key 的 scale/zero-point（校准时排除第一个 token）。
2. 推理时 Key 在 RoPE 前量化；读取 cache 后先按 per-channel 反量化、再应用 RoPE，然后与 Query 做 QKᵀ（论文用融合 kernel 把反量化与 on-the-fly RoPE 一起做）；Value 进入 cache 时在线求阈值并 per-token 量化。
3. 第一个 token 恒为 fp16；自定义 kernel 融合反量化与矩阵乘。

## 适用条件

- 面向长上下文 LLM 的 KV cache 显存优化，可与 weight-only 量化叠加。
- 需要离线校准数据与目标模型的 Key/Value 分布；需要自定义 kernel 才能在论文水平上兑现延迟收益。
- 只压 cache、不压权重；不能解决长上下文模型的训练问题。

## 结果边界

- 论文（Wikitext-2，LLaMA/Llama-2/Llama-3/Mistral）：4-bit 退化 <0.02、3-bit <0.1、2-bit <0.5 PPL，分别对应 nuq4-1% 的 3.7×、nuq3-1% 的 4.8×、nuq2-1% 的 6.9× KV 显存节省（平均位宽 4.33 / 3.33 / 2.33）。
- 长上下文（附录 A Table 8 的**显存容量估算**）：按 nuq2（无离群）相对 fp16 的 8× KV cache 压缩估算（128K 序列 64.0GB → 8.0GB），LLaMA-7B 单 A100-80GB 可容纳 1M 上下文的 KV cache、8-GPU 达 10M——这是容量估算，不是对应长度的质量或吞吐实测；kernel 在 A6000（batch 1）报告 Key 1.2–1.6×、Value 1.3–1.7× 延迟节省。
- 与 weight-only（SqueezeLLM）联用：KV 分别用 nuq4-1%（配 4-bit 权重）/ nuq3-1%（配 3-bit 权重）时额外退化 0.02 / 0.1 PPL。
- 未复现、未在目标后端核验；结论不能外推到 MLA / 线性注意力 / MoE / DiT / dLLM；延迟基准不含 prompt processing。

## Agent 如何使用

处理“KV cache 要不要压、压到几比特、Key/Value 怎么分维度量化”时，先读本页，再按需展开：

| 需要确认的内容 | 继续读取 |
| --- | --- |
| 四项技术、Eq.1、Table 1/5/6 与附录口径 | [`paper-kvquant`](../../sources/papers/paper-kvquant.md) |
| 同为 KV cache 量化、更极端的窗口+重排路线 | [SKVQ](skvq.md) |
| RTN 与分组量化的数值口径 | [量化契约](../foundations/quantization-contract.md)、[基线](baselines.md) |
| cache 量化在搜索空间中的位置 | [量化搜索空间](../guides/search-space.md) |

回答时要区分“量化维度（per-channel/per-token）”“数据类型（均匀/nuq）”“离群处理（稀疏/窗口）”三类贡献，并把论文报告值与目标后端的存储/计算收益分开陈述。

## 详细来源

- [KVQuant 原论文来源页](../../sources/papers/paper-kvquant.md)
