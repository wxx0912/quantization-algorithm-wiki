---
{
  "id": "paper-skvq",
  "title": "SKVQ: Sliding-window Key and Value Cache Quantization for Large Language Models",
  "category": "paper",
  "url": "https://arxiv.org/abs/2405.06219",
  "version": "v3（2024-11-12）",
  "version_kind": "paper-version",
  "retrieved_at": "2026-10-28",
  "read_form": [
    "html"
  ],
  "read_locator": "摘要、§2 Related Work、§3.1–3.2（Figure 2–3）、§4.1–4.3（Table 1 与消融）、§5 结论、§6 Detailed Implementations、§9 Memory and Latency（Table 6）",
  "topic": "cache",
  "excerpt_sha256": null
}
---
## 这份来源是什么

SKVQ 论文（arXiv 2405.06219，v3），面向极低比特 KV cache 量化：用 **channel reorder + group clipping 的 clipped dynamic quantization** 缓解通道离群，再用 **滑动窗口** 把最近一段 token 的 KV 保持全精度，并叠加 attention sink 过滤规则。它把 KV cache 压到 Key 2-bit / Value 1.5-bit。它是 `algorithm-skvq` 的一手依据。

范围边界：只量化 **KV cache**（实验里把 SmoothQuant、RPTQ 当作 KV 量化器对照，不涉及模型权重）；需要离线校准与运行时 KV 量化/反量化实现（论文未报告自定义 kernel）。论文对内存与延迟的分析基于 LLM-Viewer 的 roof-line 模型，不是端到端实测。

## 已读定位与阅读形式

- 阅读形式：arXiv HTML 渲染版（`arxiv.org/html/2405.06219v3`），正文与表格按 HTML 结构解析；未读 LaTeX 源码或 PDF。
- 定位：§3.1 通道重排与分组裁剪、§3.2 滑动窗口与过滤规则、§4.1 设置、§4.2 LongBench 主结果（Table 1）、§4.3 消融、§5 结论、§6 实现细节与 §9 内存/延迟分析（Table 6）。NIAH 与更多 LongBench 附表（§7–8）未逐条读。

## 关键事实

- **通道重排**（§3.1）：利用置换不变性，对 KV cache（Key 与 Value）的通道重排，把分布相似的通道排到同一量化组，再对每组做 clipping 动态量化；作者指出“离群”是相对的，且非离群通道随 token 的幅值波动大，单靠 smoothing 在极低比特下不够。
- **滑动窗口与过滤**（§3.2）：每个时间步只保留最近 `w` 个 token 的 KV 全精度（论文 `w=128`），滑出窗口的 token 按过滤规则决定是否仍保留高精度；过滤规则加入 attention sink，即把开头若干 token 保留高精度；论文讨论过按累计注意力分数保留 heavy hitters，但因收益不显著、且 FlashAttention 下难以直接取到注意力分数，明确未实现，只把它留作过滤规则的接口。
- **设置**（§4.1）：模型含 Llama-2-13B、Llama-2-7B/13B-chat、Llama-2-7B-80k、Vicuna-v1.5-7b-16k、LongChat-v1.5-32k、Mistral-7B-v0.1（滑窗注意力；SKVQ 原文记为 MQA，但 Mistral 官方论文与配置为 **GQA**，本库以官方为准，此处记为来源差异）、Mistral-7B-Instruct-v0.2（原文记为 MQA，同为该差异）；全实验用非对称量化；校准集为 Wikitext2-v1 训练集 256 条、长度 4096，离线进行；scale/zero-point 可用 FP8(E4M3) 存储，Table 3 显示几乎无额外退化。
- **主结果**（§4.2，Table 1，LongBench，平均组大小 128、K/V 2-bit、窗口 128）：SKVQ 平均分接近 FP16，优于 RPTQ、SmoothQuant、RTN 与 KIVI。例：Llama-2-7B-chat 37.50（FP16 38.50，KIVI 35.91）；Mistral-7B-Instruct 46.23（FP16 48.66，KIVI 45.48）。摘要给出的“Key 2-bit / Value 1.5-bit 无明显掉点”来自 Figure 4（group size 64，仅 Llama-2-7B-chat 与 Mistral-7B-Instruct-v0.2），与 Table 1 的 K/V 2-bit、平均组大小 128 不是同一配置。
- **内存与延迟**（§9，Table 6，LLaMA-7B，A100-80G，假设 flash-attention）：SKVQ 使单卡支持 1M 上下文；batch 128、序列 200k 时论文给出理论约 7× 解码加速（KV2 推理时间 469.7ms 对 FP16 3300ms）。

## 不能推出的结论

- 本页只读正文与限定的附录；未读 LaTeX 源码、未复现实验，也无项目内量化或部署实测。
- 结论建立在 Llama-2 / Vicuna / LongChat / Mistral（Llama 族为 MHA；Mistral 官方为 GQA、SKVQ 原文记为 MQA，部分含滑窗注意力）上，**不能外推**到 MLA、线性注意力、SSM 或 MoE/DiT/dLLM 的 cache。
- 内存/延迟是 LLM-Viewer roof-line 模型的分析结果，不是真实后端端到端实测；“1M 上下文”“7× 加速”依赖 batch/序列长度与硬件假设，不能当作通用收益。
- 依赖离线校准（256×4096）与运行时 KV 量化实现；论文未报告自定义 kernel，也不能据此认为目标推理后端已支持低比特 KV 计算。
- 窗口大小 `w=128`、过滤规则与分组策略是论文设定，换模型/长度需重新确认。
