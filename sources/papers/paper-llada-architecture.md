---
{
  "id": "paper-llada-architecture",
  "title": "LLaDA: Large Language Diffusion Models（架构与过程范围）",
  "category": "paper",
  "url": "https://arxiv.org/abs/2502.09992",
  "version": "v2（2025-02-18）",
  "version_kind": "paper-version",
  "retrieved_at": "2026-10-24",
  "read_form": [
    "latex",
    "html"
  ],
  "read_locator": "摘要、§1（Eq.1–2）、§2 Approach（2.1 Probabilistic Formulation，Eq.3–4；2.2 Pre-training；2.3 SFT，Eq.5；2.4 Inference，Eq.6）、附录 A Formulation（Training/Inference/Algorithms，Algorithm 1–5、Eq.7–16、Figure 4）、附录 B.2 Architectures and Scaling Experiments（Table 5）；编号口径正文 Eq.1–6、Figure 1–3、Table 1–4，附录 Eq.7–16、Figure 4–5、Table 5–14、Algorithm 1–5",
  "topic": "mixed-architecture",
  "excerpt_sha256": null
}
---
# LLaDA: Large Language Diffusion Models（架构与过程范围）

## 这份来源是什么

LLaDA 原始论文（arXiv 2502.09992，v2）。本条目只登记与**架构和生成过程**有关的已读范围：LLaDA 是掩码式扩散语言模型（masked diffusion model，MDM），用前向加噪（mask）与反向去噪（remasking）替代自回归的 next-token prediction。它是 `guide-architecture-transfer` 里"dLLM 与 AR 数据流不同"这一判断的架构依据，不作为任何量化质量或性能证据。

论文自述为 ICML 2025（v2 的 `main.tex` 用 `icml2025` 的 accepted 选项），但该发表信息未经会议论文集或 arXiv 元数据独立确认，按外部核实项处理。

范围边界：只读架构与过程相关章节；§3 Experiments 的 benchmark、iGSM、案例分析等实验数值不在本条目范围内、未逐条复核。

## 已读定位与阅读形式

- 阅读形式：arXiv e-print 的 LaTeX 源（单文件 `main.tex`，正文与附录同文件）；版本号取自 arXiv abs 页；v1 = 2025-02-14、v2 = 2025-02-18（其后还有 v3）。
- 定位：§2 Approach（2.1 概率形式 Eq.1–4、2.2 预训练、2.3 SFT Eq.5、2.4 Inference Eq.6）、附录 A Formulation（Algorithm 1–5、Eq.7–16、Figure 4）、附录 B.2（Table 5）。编号：正文 Eq.1–6、Figure 1–3、Table 1–4；附录 Eq.7–16、Figure 4–5、Table 5–14、Algorithm 1–5（算法全在附录）。

## 关键事实

- **前向/反向过程**（§2.1）：前向把 `x_0` 的 token 独立逐步 mask，`t ∈ (0,1)` 时每个 token 以概率 `t` 被 mask；反向令 `t` 从 1 走到 0 迭代恢复，在 `t=0` 得到模型分布 `p_θ(x_0)`。
- **mask predictor 与训练目标**（§2.1、Eq.3–4）：核心是同时预测所有 masked token 的 `p_θ(·|x_t)`，损失为只在 masked 位置计算的交叉熵（Eq.3，`t` 从 `[0,1]` 均匀采样），已被证明是模型分布负对数似然的上界（Eq.4）；与 BERT 式掩码语言模型的差别是**掩码比例在 0–1 间随机变化**而非固定。
- **不用因果掩码、无时间嵌入**（§2.2、附录 A）：Transformer 是 mask predictor、结构接近现有 LLM，但**不用 causal mask**（公式允许看到整个输入）；反向过程的条件分布只依赖干净数据、与 `t` 无关，故**时间 `t` 不作为输入、模型不含时间嵌入**。
- **与 KV cache / GQA 的关系**（§2.2）：论文明确写 "LLaDA is incompatible with KV caching"，因此用 **vanilla multi-head attention 而非 grouped query attention**、key/value head 数不同；注意力层参数因此更多，作者缩小 FFN 维度以保持可比模型大小。
- **规模与架构**（§2.2、附录 B.2）：预训练用 **2.3T tokens**、固定序列长度 **4096**（1% 数据随机取 `[1,4096]` 长度）；采用 LLaMA 类似 Transformer，用 **RMSNorm + SwiGLU + RoPE**。附录 B.2（Table 5）给出 LLaDA 8B 与 LLaMA3 8B 的架构对照，说明超参尽量一致、只做必要改动。
- **SFT 与预训练同构**（§2.3、Eq.5）：SFT 只 mask response 部分 `r_t`、保持 prompt 不变，损失形式与预训练一致（Eq.5）；把 `(p_0, r_0)` 视作干净数据、`(p_0, r_t)` 视作加噪数据后，流程与预训练完全相同。
- **采样与 remasking**（§2.4、附录 A，Algorithm 4–5）：从全 mask 的 response 出发，把反向过程离散成若干采样步（步数是超参，权衡效率与质量）；每步先同时预测所有 masked token、再按期望 remask 掉 `s/t` 的 token。除纯随机外给出两种确定性策略：**low-confidence**（remask 置信度最低的部分）与 **semi-autoregressive**（分块从左到右采样；SFT 后的模型用它，附录指出 base model 用它与随机策略差别很小）。
- **似然评估**（§2.4、Eq.6）：除 Eq.5 的上界外，另给一个方差更低的形式（均匀采样 `l` 个 token 去 mask、按 `L/l` 缩放），并采用无监督 classifier-free guidance。
- **与 AR 的关系**（附录 A，Eq.15）：附录指出该训练目标与 any-order autoregressive 模型的目标等价，解释了 LLaDA 的双向推理能力；但它用的是双向 Transformer（结论亦指出未设计专门的注意力或位置编码），与带因果掩码的同类方法不同。

## 不能推出的结论

- 本条目未复核 §3 的实验数值，不能把该论文的任何精度、吞吐或显存数字当作 LLaDA 量化后的证据。
- **不能把 KV cache 量化的结论搬到 LLaDA**：论文明确说它与 KV caching 不兼容，因此 KV cache 量化、KV 相关的算子/访存优化对它不直接适用。
- 不能由"双向注意力、无因果掩码"推断 LLaDA 的激活分布与 AR 模型相同，也不能推断为 AR 模型设计的异常值/旋转/平滑等变换可原样迁移。
- mask 阶段（大量 mask token 作为输入）的激活分布与完整序列不同，"逐步去噪"意味着同一权重在不同步看到不同掩码比例的输入，不能按单次前向的激活统计外推。
- LLaDA 8B 与 LLaMA3 8B 的架构对照只说明网络形状差别（MHA vs GQA、FFN 维度调整），不能据此推断量化敏感性相同。
- `semi-autoregressive` 只是采样/remasking 策略，不改变模型本身是双向 diffusion LM 这一事实，不能据此把它当成 AR 模型来讨论。
