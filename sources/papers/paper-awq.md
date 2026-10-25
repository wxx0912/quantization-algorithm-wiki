---
{
  "id": "paper-awq",
  "title": "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration",
  "category": "paper",
  "url": "https://arxiv.org/abs/2306.00978",
  "version": "v5（2024-07-18）",
  "version_kind": "paper-version",
  "retrieved_at": "2026-10-24",
  "read_form": [
    "latex",
    "html"
  ],
  "read_locator": "摘要、§1（Figure 1）、§2（Figure 2，HTML 锚在 §1）、§3.1（Table 1）、§3.2（Eq.1–5、Table 2–3、Figure 3）、§4.1（Figure 4）、§4.2、§5 开头（Table 4–5）、§5.1（设置）、§5.2（Table 6–9、Figure 5–8）、§5.3（Figure 8 的校准效率讨论）、§5.4（Figure 9–10 的加速讨论、Table 10）",
  "topic": "reconstruction",
  "excerpt_sha256": null
}
---
## 这份来源是什么

AWQ 原始论文（arXiv 2306.00978，v5；MLSys 2024）。它提出 **weight-only**（W4A16/W3A16）PTQ 的等价变换：只用校准集上的**激活幅度统计**给每个输入通道解缩放因子 `s = s_X^α`，把显著权重通道放大以降低其相对量化误差，再用 `1/s` 抵消回激活侧（可离线融进前序算子）；搜索空间只有标量 α，叠加 weight clipping 最小化量化 MSE。论文后半介绍 **TinyChat**（on-the-fly 反量化、SIMD weight packing、kernel 融合）。

它是 `algorithm-awq` 的原理与实验依据，也与 `guide-algorithm-selection`、`recipe-composition` 相关；同一论文的 MoE/多模态部分另登记为 `paper-awq-architecture`。

范围边界：只量化权重、激活保持 FP16（无激活量化、无 QAT）；主实验是 group size 128 的 INT4/INT3，另有 INT2 组合实验。主论文 e-print 缺 `text/appendix.tex`，**补充材料/附录未被读取**。

## 已读定位与阅读形式

- 阅读形式：arXiv e-print 的 LaTeX 源（`main.tex` + `text/*.tex` + `figure_text/*.tex`，缺 `text/appendix.tex`），另读 arXiv HTML 渲染版交叉核对编号；v5 = 2024-07-18。
- 定位：§3 AWQ（Table 1–3、Eq.1–5）、§3.2（Figure 3）、§4 TinyChat（Figure 4）、§5 实验（Table 4–10、Figure 5–10）；Table 1–10、Figure 1–10、Eq.1–5 全在正文。源码瑕疵：`tab_mistral_mixtral_acc.tex` 与 `tab_opt_ppl.tex` 共用 `\label{tab:opt_ppl}`（原文如此）。

## 关键事实

- **观察：权重重要性由激活决定**（§3.1、Table 1）。INT3-g128 下按**激活分布**保留 0.1%–1% 通道为 FP16 远好于按权重范数或随机：OPT-6.7B 上按激活保留 1% 得 PPL 11.39，而 RTN 23.54、按权重范数选 22.37、随机 24.23。混合精度有效但硬件不友好，故引出等价缩放。
- **缩放与搜索空间**（§3.2，Eq.1–5）。权重乘 `s>1`、激活除 `s`，在 `Q(w·s)·(x/s) = Δ'·Round(ws/Δ')·x/s` 与 `Δ'≈Δ` 假设下显著权重相对误差按 `1/s` 变小（Eq.2–3）；最优缩放要平衡两侧，故把空间限制为 `s = s_X^α`（`s_X` 为每通道激活平均幅度），只搜 `α ∈ [0,1]`（Eq.4–5），另加 weight clipping。Table 2 显示 OPT-6.7B 上 s=2 最优（11.92），s=4 时 21.2% 通道的 `Δ'` 被抬高、非显著通道误差反而变大。
- **设置**（§5.1）：weight-only grouped、**group size 128**（INT2 实验 g64）、INT4/INT3、Pile 校准、α 20 点网格；基线 RTN 与 GPTQ（含 reorder 变体 GPTQ-R）；论文排除 ZeroQuant/AdaRound/BRECQ（依赖反向传播）。
- **主结果**（§5.2、Table 4；WikiText-2 PPL，INT3-g128）：Llama-2 7B/13B/70B 的 AWQ **6.24/5.32/3.74**（FP16 5.47/4.88/3.32，RTN 6.66/5.52/3.98，GPTQ 6.43/5.48/3.88）；LLaMA-1 7B/13B/30B/65B 的 AWQ **6.35/5.52/4.61/3.95**（FP16 5.68/5.09/4.10/3.53）。LLaMA-7B 上 GPTQ 记 8.81、须加 reorder（GPTQ-R 6.53）才正常，AWQ 优于两者。Mixtral-8x7B / Mistral-7B（Table 5）：FP16 5.94/4.14 → INT4-g128 6.05/4.30 → INT3-g128 6.52/4.83。
- **其他任务与数据效率**（§5.2–5.3）。OpenFlamingo-9B COCO（仅量化语言部分）INT4-g128 较 FP16 只掉 1.17（RTN −4.57、GPTQ −6.72），VILA-7B/13B 基本无损，MBPP/GSM8K 与 FP16 相当。INT2-g64 下 RTN 失效，**AWQ+GPTQ** 在 OPT 1.3B–30B 上 35.71/25.70/15.71/13.25/11.38，论文据此称二者**正交**；AWQ 用 16 条序列即超过 GPTQ 192 条，Pile 的 PubMed/Enron 域互换时 AWQ PPL 只升 0.5–0.6（GPTQ 差 2.3–4.9）。
- **TinyChat 与加速**（§4–5.4）。LLaMA-7B 在 4090 batch 1 下生成 20 token 需 310 ms（200-token prompt 仅 10 ms），属 memory-bound（正文写 LLaMA-7B、Figure 3 图注写 Llama-2-7B，论文自身口径不一）。手段：on-the-fly 反量化融入 MM/MV kernel、SIMD 感知 weight packing、kernel 融合。TinyChat 相对 HF FP16 在 4090 上 2.7–3.9×，Orin 上对 llama.cpp 最高 1.7×、对 AutoGPTQ ≥2.6×。

## 不能推出的结论

- 所有精度与加速结论都绑定 **W4A16/W3A16**（不量化激活），不能当作 W4A4、W8A8 或 KV cache 量化的证据。
- 数字绑定 group 128（INT2 为 g64）、α 的 20 点网格与 Pile 校准集；换 group size/校准数据/α 范围都会改变结果。
- "AWQ 优于 GPTQ"是**相对论文所选的 GPTQ 配置与实现**（LLaMA-7B 上 GPTQ 需 reorder trick），不是对 GPTQ 全部实现的普适排名。
- Table 9 的 INT2 结论来自 **AWQ 与 GPTQ 组合**，不能推出"单用 AWQ 可用 INT2"或"任意组合/顺序都有效"。
- TinyChat 的加速来自位宽、packing、kernel 融合与平台专用实现，不是缩放搜索本身；不能外推到其他后端（llama.cpp/TensorRT-LLM）或训练场景。多模态/指令结论只覆盖 OpenFlamingo-9B（仅语言部分）、VILA-7B/13B、Vicuna 7B/13B、LLaVA-13B。
