---
{
  "id": "paper-kvquant",
  "title": "KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization",
  "category": "paper",
  "url": "https://arxiv.org/abs/2401.18079",
  "version": "v6（2025-05-28）",
  "version_kind": "paper-version",
  "retrieved_at": "2026-10-28",
  "read_form": [
    "html"
  ],
  "read_locator": "摘要、§2 Background、§3.1–3.7（含 Eq.1）、§4.1–4.4（Table 1/5/6）、§5 结论、附录 A（Table 7/8）、附录 G（Table 9）、附录 S；附录 D/E/F/H/L 仅查阅结论口径",
  "topic": "cache",
  "excerpt_sha256": null
}
---
## 这份来源是什么

KVQuant 论文（arXiv 2401.18079，v6），面向长上下文 LLM 推理的 KV cache 低比特量化。它指出既有方法在 4-bit 以下难以准确表示 KV 激活，提出四项技术组合——per-channel key 量化、pre-RoPE key 量化、按层敏感度加权的非均匀数据类型 nuqX、per-vector dense-and-sparse——并配以 attention sink 处理与离线校准。它是 `algorithm-kvquant` 的一手依据。

范围边界：只量化 **KV cache**，可与 weight-only 方法联用；需要离线校准与自定义 CUDA kernel，不含微调。

## 已读定位与阅读形式

- 阅读形式：arXiv HTML 渲染版（`arxiv.org/html/2401.18079v6`），正文与表格按 HTML 结构解析；未读 LaTeX 源码或 PDF。
- 定位：§3.1–3.7 方法（含 Eq.1 敏感度加权目标）、§4.1–4.4 结果（Table 1 主表、Table 5 与权重量化联用、Table 6 kernel 计时）、§5 结论、附录 A（Table 7/8 容量估算）、附录 G（Table 9 per-channel 消融）、附录 S（局限）；附录 D/E/F/H/L 仅查阅其结论口径，其余附录的推导与逐项消融未逐条读。

## 关键事实

- **分布观察**（§3.1，Figure 2）：Key 矩阵存在明显的离群通道；Value 既有离群通道也有离群 token，但程度弱于 Key 的离群通道。
- **四项技术**：① **per-channel key 量化**（scale/zero-point 沿通道共享；Value 仍 per-token）；② **pre-RoPE key 量化**（在旋转位置编码前量化 Key，避免 RoPE 把成对通道按位置角度混合）；③ **nuqX**：按层推导的敏感度加权非均匀数据类型，用对角 Fisher 构造目标 `min Σ F_ii (A_i − Q(A_i))²`（Eq.1），离线用 k-means 求 signposts；④ **per-vector dense-and-sparse**：每个向量（per-channel 或 per-token）用各自的离群阈值隔离约 1% 离群值。
- **附加处理**：attention sink——把第一个 token 的 KV 保留 fp16，在低比特下带来稳定收益，且校准时不纳入该 token；**离线校准** Key 的 scale/zero-point，Value 的离群阈值在线计算（CPU offload）。
- **per-channel 的收益**（§3.1）：Key per-channel + Value per-token 相比两者都 per-token，在 LLaMA-7B 3-bit Wikitext-2 上改善 3.82 PPL。
- **校准设置**（§4.1）：16 条、序列长度 2K，取自 Wikitext-2 训练集。
- **主结果**（§4.1，Wikitext-2，LLaMA/Llama-2/Llama-3/Mistral）：4-bit 退化 <0.02、3-bit <0.1、2-bit <0.5 PPL，对应 3.7× / 4.8× / 6.9× 的 KV 显存节省；nuq2 提供 8× 压缩。
- **长上下文**（附录 A，Table 8 的**显存容量估算**）：LLaMA-7B 序列 128K 时 fp16 需 64.0GB、nuq2（无离群）为 8.0GB（8×）、nuq2-1% 为 9.3GB；按此估算，单张 A100-80GB 可容纳 1M 上下文的 KV cache、8-GPU 系统可达 10M；这是显存可容纳量的估算，不是对应长度的质量或吞吐实测。
- **per-channel 的边界**（§3.1、附录 G/Table 9，LLaMA-7B、seqlen 128K）：per-channel 量化只对 Key 有收益；int3 下 PT/PT 10.87、PC/PT 7.05、PC/PC 223 PPL（fp16 5.68，KV cache 64.0GB 对 12.0GB），即对 Value 用 per-channel 比 per-token 更差。论文给出两点实现口径：所有 per-channel 实验用 fp16 zero-point（低比特整数 zero-point 会落到可表示范围外）；并推测 per-channel Value 会把误差集中到输出向量的单个分量、KIVI 也报告了类似现象。
- **与权重量化联用**（§4.3，Table 5）：KV 分别用 nuq4-1%（配 4-bit 权重）/ nuq3-1%（配 3-bit 权重）时，叠加 SqueezeLLM weight-only 仅额外退化 0.02 / 0.1 PPL（LLaMA-7B/13B）。
- **kernel**（§4.4，Table 6，A6000，batch 1）：Key 乘 1.2–1.6×、Value 乘 1.3–1.7× 延迟节省。
- **基线**（§4.1、§4.2）：均匀 intX、非均匀 nfX（NormalFloat）、Atom、FlexGen，以及 KIVI（2-bit，group 32 + 128 fp16 残差，§4.2/Table 2–3）；Atom 与 FlexGen 的分组大小在 §4.1 正文（64/128）与 Table 1 表注（128/64）中互相矛盾，本页不采信单一值。

## 不能推出的结论

- 本页只读正文与限定的附录，未复现实验，也无项目内量化或部署实测；PPL 与显存数字均为论文报告值。
- 结论建立在 LLaMA / Llama-2 / Llama-3 / Mistral（MHA/GQA）上，**不能外推**到 MLA、线性注意力、SSM 的 cache，或 MoE/DiT/dLLM 的中间激活。
- “1M / 10M 上下文”“1.2–1.7× 延迟节省”是论文在其实现与硬件（A100-80GB / A6000，batch 1）上的报告值，不等于任意后端的端到端收益；且延迟基准只覆盖 memory-bound 的解码阶段，不含 prompt processing（附录 S）。
- 方法依赖离线校准与自定义 kernel；不能据此认为目标推理后端已支持低比特 KV 计算。
- nuq 数据类型、离群阈值与稀疏比例、第一 token 保留等超参都会影响结果，不能当作通用常数。
