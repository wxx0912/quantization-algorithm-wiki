---
{
  "id": "paper-llmint8",
  "title": "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale",
  "category": "paper",
  "url": "https://arxiv.org/abs/2208.07339",
  "version": "v2（2022-11-10）",
  "version_kind": "paper-version",
  "retrieved_at": "2026-10-28",
  "read_form": [
    "html"
  ],
  "read_locator": "摘要、§2 Background、§3.1 向量级量化（Eq.7）、§3.2 混精分解（Eq.8）、§3.3 设置、§3.4 主结果（Table 1）、§4.1–4.3 离群特征分析、§6 讨论与局限、附录 D 运行时",
  "topic": "core",
  "excerpt_sha256": null
}
---
## 这份来源是什么

LLM.int8() 论文（arXiv 2208.07339，v2），提出把 transformer 的**前馈层与注意力投影层**（权重与激活都用 Int8）做成 8-bit 矩阵乘法的量化流程，使 175B 参数模型零退化加载与推理。方法分两段：**向量级量化**（vector-wise，每个内积一组独立归一化常数）与**混合精度分解**（mixed-precision decomposition，把极少数量级 ≥6 的离群特征维用 FP16 单独相乘，其余 >99.9% 的值走 Int8）。

它是 `algorithm-llmint8` 的一手依据；在本库中作为 **W8A8** 与"用混精分解处理离群通道"的经典基线，与 `paper-smoothquant`（通道缩放）、`paper-gptq`（二阶补偿）对照。

范围边界：**8-bit（W8A8 / Int8）**，只量化 FFN 与注意力投影层（embedding、norm、bias 保持 FP16）；不含亚 8-bit 结果。

## 已读定位与阅读形式

- 阅读形式：arXiv HTML 渲染版（`arxiv.org/html/2208.07339v2`），正文与表格按 HTML 结构解析；未读 LaTeX 源码或 PDF。
- 定位：§2 Background（absmax 与 zeropoint 量化）、§3.1 向量级量化（Eq.7）、§3.2 混精分解（Eq.8）、§3.3 实验设置、§3.4 主结果（Table 1）、§4.1 Finding Outlier Features、§4.2 Measuring the Effect、§4.3 Interpretation、§6 Discussion and Limitations、附录 D（运行时）。§5 Related work、§7 Broader Impacts 与附录 B/C/E/F 未逐条读。

## 关键事实

- **向量级量化**（§3.1，Eq.7）：把矩阵乘看成独立内积序列，给 `X` 每行、`W` 每列各一个归一化常数，反量化用常数外积 `c_x⊗c_w`，即 `C_f16 ≈ S·C_i32 = S·Q(A_f16)Q(B_f16)`，Int8 乘法在 **Int32** 累积；相比"整张量一个 scale"，它把离群影响限制在每个内积内。
- **混精分解**（§3.2，Eq.8）：定义离群特征维集合 `O`（该维至少有一个元素量级超过阈值 `α`）；论文称 `α=6.0` 足以把退化压到接近零。`O` 的列（及 `W` 对应行）用 FP16 相乘，其余用 Int8 相乘，最后在 FP16 相加：`C ≈ Σ_{h∈O}X_f16^h W_f16^h + S_f16·Σ_{h∉O}X_i8^h W_i8^h`。13B 以内 `|O|≤7`，只占约 0.1% 额外内存，>99.9% 的值仍在 8-bit 相乘。
- **离群特征判据与相变**（§4.1，Figure 3）：判据为"量级 ≥ 6.0、影响 ≥ 25% 的层、影响 ≥ 6% 的序列维"。相变发生在 **6.7B** 附近——离群出现在所有层与约 **75%** 的序列维；6.7B 规模下每条序列约 **150,000** 个离群值，却集中在**仅 6 个特征维**。
- **离群特征的作用**（§4.2）：把这些离群维置零，mean top-1 softmax 概率由约 40% 降到约 20%、验证 PPL 上升 **600–1000%**（尽管它们仅占约 0.1% 的输入特征）；对照实验随机置零同等数量的特征只使 top-1 概率下降 0.02–0.3%、PPL 上升约 0.1%。
- **设置**（§3.3–3.4）：PPL 用 fairseq 预训练的稠密 AR transformer（125M–13B，C4 验证集，NVIDIA A40）；zeroshot 用 OPT 模型（EleutherAI lm-eval harness，WinoGrande/HellaSwag/PIQA/LAMBADA），规模到 175B。
- **主结果**（Table 1，C4 验证 PPL，越低越好）：13B 上 FP32 **12.45**，Absmax LLM.int8() **12.45**、Zeropoint LLM.int8() **12.45**（完全恢复）；同期基线 Int8 absmax 19.08、Int8 zeropoint 13.94、absmax row-wise 16.49、absmax vector-wise 16.48、zeropoint vector-wise 13.47（论文指出 13B 的 8-bit PPL 比 6.7B 更差）。2.7B 上 FP32 14.43、LLM.int8() 14.44/14.43。
- **zeroshot**（§3.4、Figure 1）：LLM.int8() 在 125M–175B 上保持 FP16 水平；基线 8-bit absmax vector-wise 随规模退化到接近随机。
- **显存与速度**（摘要、§3.4、附录 D）：显存约减半，175B 模型可在单台服务器（消费级 GPU）上跑；量化开销会拖慢 <6.7B 的推理，但对等效 175B 的大矩阵乘约快 **2×**，BLOOM-176B 的端到端运行时保持。
- **代码**：`https://github.com/TimDettmers/bitsandbytes`，并有 HuggingFace Transformers 集成。

## 不能推出的结论

- 本页只读正文与所列表格，未复现实验、未运行；PPL 与加速均为论文报告值。
- 方法是 **8-bit（W8A8）**；不能推出 4-bit 及以下的结论，论文也未报告 int4 结果。
- 只量化 FFN 与注意力投影层，embedding/norm/bias 仍为 FP16；不能理解为"整模型 8-bit"。
- 质量结论基于 fairseq 稠密 AR transformer / OPT（125M–175B）、C4 与 EleutherAI harness；端到端运行时另测 BLOOM-176B（附录 D）。不能外推到 MoE、dLLM、DiT、多模态或 LLaMA 系列（论文未测）。
- "约 2× 加速"只对等效 175B 的大矩阵乘成立；<6.7B 模型因量化开销可能变慢。显存与加速绑定其实现（bitsandbytes / HF）与硬件，不等于任意后端收益。
- 离群阈值 `α=6.0`、判据（≥25% 层、≥6% 序列维）与"6 个特征维"均为该论文的设定与观测；换模型或规模需重新确认。
