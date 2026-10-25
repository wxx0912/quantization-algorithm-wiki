---
{
  "id": "paper-dllmquant",
  "title": "DLLMQuant: Quantizing Diffusion-based Large Language Models",
  "category": "paper",
  "url": "https://arxiv.org/abs/2508.14090",
  "version": "v1（2025-08-14）",
  "version_kind": "paper-version",
  "retrieved_at": "2026-10-25",
  "read_form": [
    "html"
  ],
  "read_locator": "摘要、Introduction、Related Work、Method（TMAS/Alg.1、IA-AQ 含 Eq.6–9、CGQ 含 Eq.10）、Experiments（Setup、Results Table 1–3、Memory and Speedup Table 4）",
  "topic": "mixed-architecture",
  "excerpt_sha256": null
}
---
## 这份来源是什么

DLLMQuant 论文（arXiv 2508.14090，v1），研究扩散式语言模型（DLLM）的 PTQ。它指出 DLLM 的三个机制——动态 mask、迭代生成、双向注意力——与现有 LLM PTQ 假设冲突，并给出三项针对性技术（TMAS 校准采样、IA-AQ 激活量化、CGQ 权重量化补偿），在 LLaDA-8B / LLaDA-1.5-8B / DREAM-7B 上评估。

它是 `algorithm-dllmquant` 的一手依据，也是本项目关注文本 dLLM 量化时最直接的一篇论文来源。

范围边界：**PTQ**（论文强调无需微调）；主结果 W4A4（权重 4-bit、激活 4-bit）；基线 RTN、AWQ、QuaRot（QuaRot 用 GPTQ 补偿权重）；硬件 NVIDIA A6000。

## 已读定位与阅读形式

- 阅读形式：arXiv HTML 渲染版（`arxiv.org/html/2508.14090v1`），正文与表格按 HTML 结构解析；未读 LaTeX 源码或 PDF。
- 定位：Method 三节（TMAS 的 Alg.1、IA-AQ 的 Eq.6–9、CGQ 的 Eq.10）、Experiments 的 Setup、Results Table 1–3、Memory and Speedup Table 4。未读附录 A（整体消融）与附录 B（逐层输出分布）。

## 关键事实

- **问题**（摘要、Introduction）：把 AWQ 直接用于 LLaDA-8B 的 W4A4 会掉 **16%** 准确率（摘要），Introduction 表述为"超过 16%"。论文归因于三点：① 迭代生成 + 动态 mask ratio 使各解码步的 token 分布不同，常规随机/均匀采样校准覆盖不到；② 每步输出作为下一步输入，量化误差随迭代累积放大；③ 已解码（unmasked）token 稳定、masked token 仍具概率性，GPTQ 用统一 Hessian 等价对待所有 token 的假设失效。
- **TMAS**（Temporal-Mask Adaptive Sampling，Alg.1）：按 block 数 `B` 与总步数 `T` 分层采样；每 block 取 `⌊512/B⌋` 条样本；把 mask ratio 分 4 档、目标比例 `[0.3,0.2,0.2,0.3]`；相邻步分布相似、远离步差异大，故按区间跨步采样。论文称该采样能恢复 INT4 模型大部分性能。
- **IA-AQ**（Interaction-Aware Activation Quantization，Eq.6–9）：定位到 attention 中 softmax 输出与 value 矩阵相乘这一步是误差累积主因；重设 `V` 的量化误差度量为以 softmax 输出加权的 Frobenius 误差（Eq.7），再对标准 scale 乘系数 `α`（从 `{1.0,0.8}` 搜索）以支持交互强/重要的 token（Eq.8–9）。
- **CGQ**（Certainty-Guided Quantization，Eq.10）：用 mask 指示与 token 最终分数的平方根加权 Hessian（`(X⊙(1[mask]+√sc))` 的 Gram），只在被解码且高置信的区域上加重，使权重量化补偿更贴合 DLLM 迭代解码。
- **设置**：权重对称均匀量化、per-channel；激活非对称均匀、per-token；校准取 WinoGrande 的 128 段；评测 TruthfulQA-MC2、Arc-C、HellaSwag、WinoGrande、PIQA、MMLU、C-EVAL、HumanEval、GSM8K 共九项取平均。
- **结果**（Table 1，LLaDA，W4A4，九任务平均）：FP **59.87**；RTN 44.23；AWQ 48.09；QuaRot 51.03；AWQ+DLLMQuant 49.26；QuaRot+DLLMQuant **54.29**。GSM8K：QuaRot 44.57 → DLLMQuant **56.25**（提升约 11.7 分）。
- **消融**（Table 2，LLaDA，W4A4 平均）：GPTQ 51.03 → +TMAS 52.36 → +CGQ 53.16 → +IA-AQ **54.29**；AWQ 48.09 → 49.26。Table 3 显示 CGQ 的 mask 与 score 两个因子需同时用。
- **效率**（Table 4）：论文报告平均推理加速 **>1.6×**、显存节省 **>3.2×**。

## 不能推出的结论

- 本页只读正文；未读附录 A/B，未复现实验，也未核验论文所称"将发布代码"是否已有公开仓库，故本库暂无对应 repo 来源页。
- 结论建立在 LLaDA-8B、LLaDA-1.5-8B、DREAM-7B 三个文本 DLLM 上，不能外推到 AR LLM、图像/视频 DiT 或其它 DLLM 实现。
- ">1.6× 加速 / >3.2× 显存节省"是论文在其实现与 A6000 上报告的值，属特定 batch/实现，不等于任意后端或端到端的收益。
- IA-AQ 的 `α` 只在 `{1.0,0.8}` 内搜索、TMAS 的比例 `[0.3,0.2,0.2,0.3]` 是经验设定，均属启发式；换数据集/模型需重新确认，不能当作通用常数。
- 论文对 W4A4 的报告不等于该模型在目标推理后端上有真实低比特 kernel 或加速；表内分数是准确率指标，不能读成性能证据。
