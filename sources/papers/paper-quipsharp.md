---
{
  "id": "paper-quipsharp",
  "title": "QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks",
  "category": "paper",
  "url": "https://arxiv.org/abs/2402.04396",
  "version": "v2（2024-06-04）",
  "version_kind": "paper-version",
  "retrieved_at": "2026-10-28",
  "read_form": [
    "html"
  ],
  "read_locator": "摘要、§3 RHT 非相干处理（Alg.3）、§4 BlockLDLQ 与格子码本（4.1 自适应舍入、4.2 E8P、4.3 RVQ）、§5 量化期微调、§6.1–6.4（Table 2–6）、§7 结论、附录 F.2 Hessian 生成",
  "topic": "transforms",
  "excerpt_sha256": null
}
---
## 这份来源是什么

QuIP# 论文（arXiv 2402.04396，v2），是 QuIP 的改进版**权重-only** PTQ 方法，面向极端压缩（每权重 ≤4 bit，主打 2 bit）。三项技术：① 用**随机 Hadamard 变换（RHT）**替换 QuIP 的两因子 Kronecker 做非相干处理；② 用**向量量化**匹配非相干权重的球状分布，码本基于 **E8 格子**（2-bit E8P 码本）；③ **量化期微调**。在 Llama 1/2 上评估。

它是 `algorithm-quipsharp` 的一手依据；在本库中作为"2-bit 权重-only + 格子码本 + 非相干变换"的代表，与 `paper-gptq`/`paper-awq`（重构/缩放）、`paper-quarot`（旋转）、`paper-omniquant`（混精）对照。

范围边界：**权重-only**，≤4 bit；含可选微调；不含激活量化。

## 已读定位与阅读形式

- 阅读形式：arXiv HTML 渲染版（`arxiv.org/html/2402.04396v2`），正文与表格按 HTML 结构解析；未读 LaTeX 源码或 PDF。
- 定位：§3 非相干处理与 RHT（Alg.3）、§4 BlockLDLQ（4.1 自适应舍入、4.2 E8P 码本、4.3 RVQ 扩到高比特）、§5 量化期微调、§6.1 Llama 结果（Table 2、Table 4）、§6.2 比特缩放、§6.3 高效推理（Table 5–6）、§6.4 消融、§7 结论。附录 A–E 未逐条读。

## 关键事实

- **校准与 Hessian**（§2.2、Alg.1、附录 F.2）：目标是最小化逐层 proxy loss `tr((Ŵ−W)H(Ŵ−W)ᵀ)`，`H=E_x[xxᵀ]` 由校准集估计；Alg.1 的输入即 `W`、`H` 与码本 `C`。论文用 RedPajama 1T 的 6144 条原生上下文序列生成 Hessian（Llama 1 用 2048、Llama 2 用 4096），生成一次可跨实验复用、不计入量化耗时。
- **RHT 非相干处理**（§3，Alg.3 IP-RHT）：用随机 Hadamard 变换替换 QuIP 的两因子 Kronecker，改进非相干参数的理论界，把结构化随机正交矩阵乘法从 `Θ(n√n)` 降到 `Θ(n log n)`，且 Hadamard 条目属 `{−1,+1}`，可无浮点乘实现。
- **BlockLDLQ**（§4、§4.1）：把 Chee et al. (2023) 的 LDLQ 自适应舍入（逐列、线性反馈）扩展到**向量量化**——一次对一块权重做向量舍入，使可表示集合从"超立方体"变成更贴合 RHT 后球状分布的"球"。
- **E8P 码本**（§4.2）：基于 `E8` 格子（`R⁸` 中最优单位球填充）；2-bit E8P 码本含 `2^16` 个条目，但查表只需 `2^8` 项，其余 8 bit 存符号与平移，因而查表小、解码快（`<5` 条指令/权重、码本 1KiB，§6.3）。
- **扩到高比特**（§4.3）：用**残差向量量化（RVQ）**——4 bit 用 2-bit E8P 量化两遍；3 bit 用 2-bit E8P + 1-bit `E8` 码本。
- **量化期微调**（§5）：先在每个 transformer block 内微调未量化层以补偿已量化层（块内可并行），全部量化后再微调剩余未量化参数以最小化整模激活误差；把符号向量当实数向量优化以让非相干处理"塑形"权重。论文称 70B 模型约需 50 GPU-hours；符号向量存 FP16 后仍只增加 **<0.01 bit/weight**。
- **设置**（§6.1）：WikiText2 与 C4 困惑度（Table 2 上下文长度 2048；与 AQLM 对比的 Table 4 用 4096）；Llama 1（7/13/30/65B）与 Llama 2（7/13/70B）；基线含 AWQ、OmniQuant、AQLM、QuIP；Table 3 为 zeroshot。
- **主结果**（Table 2，WikiText2 PPL，越低越好）：2-bit 下 QuIP# 在 Llama-1-7B 为 **6.86**、Llama-2-70B 为 **4.16**，而 OmniQuant 分别为 15.5、7.81；3-bit QuIP# 为 5.98/5.31/4.36/3.78/5.79/5.10/3.56；4-bit 为 5.76/5.17/4.18/3.60/5.56/4.95/3.38。C4 上 2-bit QuIP# 为 8.36/7.48/6.71/6.19/8.35/7.45/6.12。FP16 WikiText2 为 5.68/5.09/4.10/3.53/5.47/4.88/3.32。
- **比特缩放**（§6.2）：论文称 Llama 2 上 QuIP# 3-bit 优于"理论无损 4-bit 模型（FP16 按 4 bit 计）"，且首次出现 2-bit PTQ 具备与更高比特类似的缩放趋势。
- **消融**（§6.4，Table 2/4）：即使去掉微调与格子码本（"no FT & no E₈"），QuIP# 仍显著优于 OmniQuant 与 AWQ；各组件在各比特下都有增益；与 QuIP 的差距也体现 RHT 相对 Kronecker 的更强非相干性。
- **推理**（§6.3，Table 5–6）：2-bit QuIP# 在 RTX 4090（FlashAttention Llama 实现）上 2-7B 达 **170.50 tok/s**（峰值内存带宽 29.60%）、2-70B **32.74 tok/s**（56.84%）；在 RTX 4090（HuggingFace 实现）上 2-bit Llama-2-7B 为 **106.3 tok/s**，而 AQLM 2-bit 为 20.6、FP16 为 33.1。论文称在 A6000 上吞吐约为 QuIP 的 2×，比 SpQR/SqueezeLLM 高 >40%。
- **代码**：`https://github.com/Cornell-RelaxML/quip-sharp`。

## 不能推出的结论

- 本页只读正文与所列表格，未复现实验、未运行；PPL、吞吐均为论文报告值。
- 是**权重-only**方法：不量化激活，不能据此推断 W4A4/W8A8 等权重+激活设置的表现。
- 头条结果包含**微调**（非纯训练-free）；"no FT" 行才是无微调配置，不能把两者混说。
- 本页只核对正文 §6.1–6.4 的 Llama 1/2 与 WikiText2/C4（含 Table 3 zeroshot）；附录 E.2 另报告 Mixtral-8x7B（MoE）与 Falcon-180B，本页未逐条核对，故不在此引用。
- 码本/格子（E8P）、RVQ 组合、微调超参与 "no FT" 的具体配置都是设计选择；2-bit 的最优码本未必适合其它位宽或后端。
- 吞吐数字绑定具体 kernel/库（FlashAttention 或 HuggingFace 实现）与硬件（RTX 4090 / A6000），不等于任意后端收益。
