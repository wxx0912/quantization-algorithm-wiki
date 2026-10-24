---
{
  "id": "paper-awq-architecture",
  "title": "AWQ 论文中 MoE 与多模态部分（v5，架构迁移范围）",
  "category": "paper",
  "url": "https://arxiv.org/abs/2306.00978",
  "version": "v5（2024-07-18）",
  "version_kind": "paper-version",
  "retrieved_at": "2026-10-24",
  "read_form": [
    "latex",
    "html"
  ],
  "read_locator": "§5.1 Settings、§5 开头（Table 4–5 浮动体）、§5.2 Evaluation（Table 4–5 讨论、Table 6 OpenFlamingo、Table 7 VILA、Table 8 码/数学、Figure 5 Vicuna、Figure 6–7 定性）、§5.4 Speedup（Table 10 VILA latency）；编号口径 Table 1–10、Figure 1–10、Eq.1–5（`S3.T1`–`S3.T3`、`S5.T4`–`S5.T10`、`S1.F1`–`S1.F2`、`S3.F3`、`S4.F4`、`S5.F5`–`S5.F10`、`S3.E1`–`S3.E5`）",
  "topic": "mixed-architecture",
  "requires_external_verification": true,
  "excerpt_sha256": null
}
---
## 这份来源是什么

同一篇 AWQ 论文（arXiv 2306.00978，v5）中与**架构迁移**有关的已读范围：Mixtral 的 MoE 结果、Vicuna 指令微调、OpenFlamingo-9B 与 VILA 的视觉语言实验、VILA 部署吞吐。与 `paper-awq` 同源同版本，两个 ID 都保留。

范围边界：只覆盖正文（`text/*.tex`、`figure_text/*`）；该 v5 的 e-print 缺 `text/appendix.tex`，HTML 渲染版也不含附录，因此**不覆盖任何附录表或消融**。

## 已读定位与阅读形式

- 阅读形式：arXiv e-print 的 LaTeX 源，另读 arXiv HTML 渲染版交叉核对锚点；v5 = 2024-07-18。
- 定位：Table 1–10、Figure 1–10、Eq.1–5 全在正文；本条目只关心 Table 5（Mistral/Mixtral）、Table 6（OpenFlamingo）、Table 7（VILA）、Table 8（码/数学）、Figure 5（Vicuna GPT-4 评测）、Figure 6–7（定性）、Table 10（VILA 吞吐）。源文件提示：`tab_opt_ppl.tex` 与 `tab_mistral_mixtral_acc.tex` 共用 `\label{tab:opt_ppl}`（原文如此），正文本该指向 Table 3 的引用会错解析到 Table 5。

## 关键事实

- **设置**（§5.1）：weight-only grouped、group size 128、INT4/INT3、Pile 小样本校准、α 网格 20。
- **MoE / GQA**（Table 5）。Mistral-7B-Instruct-v0.2 / Mixtral-8x7B-Instruct-v0.1 的 WikiText-2 PPL：FP16 4.14/5.94 → INT4-g128 4.30/6.05 → INT3-g128 4.83/6.52，表注称覆盖 GQA 与 MoE 两种架构。
- **指令微调**（Figure 5）。Vicuna-7B/13B 的 INT3-g128 用 GPT-4 协议评测：80 题、正反序共 160 次比较，AWQ 胜率高于 RTN 与 GPTQ。
- **VLM**（Table 6–7）。OpenFlamingo-9B COCO Captioning CIDEr 32-shot（仅量化**语言部分**）：INT4-g128 相对 FP16 退化 RTN −4.57、GPTQ −6.72、AWQ −1.17。VILA-7B/13B 的 INT4-g128 在 11 个视觉语言 benchmark 上接近 FP16（VQAv2 80.3→80.1、80.5→80.4）；论文自称"VLM 低比特量化"首个研究。
- **码/数学与吞吐**（Table 8、Table 10）。CodeLlama-7b-Instruct-hf 的 MBPP 与 Llama-2 的 GSM8K 在 INT4-g128 下 AWQ 优于对照；VILA W4A16 相对 FP16，VILA-7B 在 Orin 上 11.5 → 35.6 tokens/s（表注最高 3.1×）、VILA-13B 6.1 → 17.5（最高 2.9×）。

## 不能推出的结论

- OpenFlamingo 只量化语言部分，不能当作视觉编码器/投影器/跨模态层已量化的证据；VILA 结论也限于其视觉塔、分辨率与任务集合。
- 单个 Mixtral-8x7B 结果不覆盖任意 router、shared expert 或 fused expert 布局；Table 5 是 PPL，不是 MoE 真低比特 GEMM、专家并行或路由算子的性能证据。
- Vicuna 的 Figure 5 是 GPT-4 自动评测，不是人工评测，不能推广到其他模型或协议。
- 结论绑定 AWQ 固定设置（weight-only、g128、INT3/INT4、Pile 校准），不能推断 W/A/KV 同时低比特或更小 group 下同样成立；Table 10 的吞吐也只在 A100/4090/Orin 与特定实现下测过。
- 本条目与 `paper-awq` 共享同一批实验，不能拼成新的独立实验；附录（缺失）中的消融不在范围。
