---
{
  "id": "paper-smoothquant",
  "title": "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models",
  "category": "paper",
  "url": "https://arxiv.org/abs/2211.10438",
  "version": "v7（2024-03-29）",
  "version_kind": "paper-version",
  "retrieved_at": "2026-10-23",
  "read_form": [
    "latex",
    "html"
  ],
  "read_locator": "摘要、§1（Figure 1–2）、§2（Eq.1、Figure 3–4）、§3（Eq.2、Table 1）、§4（Eq.3–4、Figure 5–6）、§5.1–5.5（Table 2–11、Figure 7–10）、§6–7、附录 A",
  "topic": "core",
  "requires_external_verification": true,
  "excerpt_sha256": null
}
---
## 这份来源是什么

SmoothQuant（ICML 2023 论文；本页读 arXiv v7，2024-03-29，在会议版基础上补了 Llama-2/Falcon/Mistral/Mixtral 等结果）。它解决**权重与激活同为 INT8（W8A8）**时激活异常值导致的精度崩塌：用数学等价的逐通道缩放 `diag(s)` 把量化难度从激活迁移到权重，从而能用上 INT8 GEMM kernel，而不是像 LLM.int8() 那样保留 FP16 异常值通道。

它是 `algorithm-smoothquant`、`foundation-quantization-contract`（静态/动态、粒度、校准口径）与 `algorithm-baselines` 的一手材料。

范围边界：只做 INT8 的 training-free PTQ；不做 4-bit/低比特、没有 weight-only 实验、没有 QAT 或精度恢复微调；实验覆盖 OPT/BLOOM/GLM-130B/MT-NLG-530B 与 Llama-1/2、Falcon、Mistral、Mixtral。

## 已读定位与阅读形式

- 阅读形式：arXiv e-print 的 LaTeX 源（`main.tex` 加 `\input` 的 text 与 fig_text，含表格数值本体），另读 arXiv HTML 渲染版交叉核对编号。v7 = 2024-03-29。
- 定位：§2 量化器与粒度（Eq.1）、§3 量化难度（Eq.2、Table 1）、§4 SmoothQuant（Eq.3–4）、§5 实验（Table 2–11、Figure 7–10）、§6–7、附录 A（与 weight-only 的定性讨论，无图表）。

## 关键事实

- **量化器**（§2，Eq.1）：对称均匀 INT8，`Δ = max(|X|)/(2^{N−1}−1)`；按统计来源分 static/dynamic，按粒度分 per-tensor / per-token（激活）/ per-channel（权重）/ group-wise。
- **核心约束与观察**（§3，Eq.2、Table 1）：硬件 GEMM 只允许在矩阵乘的外层维度缩放，所以激活只能用 per-token；三条观察是"权重好量化""激活异常值比常规值大约 100×""异常值固定在少数通道上"。Table 1 显示只有 per-channel 激活量化能保住精度，但它与 INT8 GEMM 不兼容（原表标灰）。
- **等价变换与迁移强度**（§4，Eq.3–4）：`s_j = max(|X_j|)^α / max(|W_j|)^{1−α}`；`s` 可离线融合进前一层、不新增 kernel 调用。**α=0.5** 对 OPT/BLOOM 通用，**α=0.75** 用于异常值更严重的 GLM-130B。
- **校准口径**（§5.1）：smoothing 因子与 static 量化步长一次性用 Pile 的 512 条随机句子标定，α 在 Pile 验证集上做 grid search；同一个量化模型用于全部下游任务。
- **量化档位**（§5.1，Table 2）：SmoothQuant 提供 O1（per-tensor 权重 + per-token dynamic 激活）、O2（再加 per-tensor dynamic）、O3（再加 per-tensor static）三档，O1→O3 效率递增。
- **主结果**（§5.2，Table 3–4）：OPT-175B 七任务平均 66.9%（FP16）→ O1 66.5%；BLOOM-176B 68.2% → O2 68.4%；GLM-130B 73.8% → O1 73.7%。同表的 W8A8/ZeroQuant/Outlier Suppression 全部接近随机、PPL 爆炸，LLM.int8() 保精度但延迟更高。
- **加速与显存**（§5.3）：PyTorch-HF 单卡最高 1.51× 加速、1.96× 显存节省；FasterTransformer 单卡最高 1.56×，并把 OPT-66B 从 2 卡降到 1 卡、175B 从 8 卡降到 4 卡；全部实验在 A100 80GB。MT-NLG 530B（Table 9–10）精度不变（73.1%），显存 1040GB→527GB、16→8 卡。
- **实现契约**（§5.1）：两个后端（PyTorch HuggingFace 概念验证、FasterTransformer 生产），线性层与 bmm 用 CUTLASS INT8 GEMM，其余算子保持 FP16。

## 不能推出的结论

- 只覆盖 W8A8 INT8：不做低比特、weight-only 或 QAT；附录 A 的 W4A4 是 future work，不是实验结论。
- α=0.5/0.75 是论文在其模型与校准集上的经验选择，`max(|X_j|)` 是校准样本统计量；不能当作普适最优，也不能推断换校准集后仍是同一 α。
- Table 4 **只有 GLM-130B 一列**改用 LAMBADA/MMLU/MNLI/QNLI（原表用 `{*}` 注明不可横比），且额外裁掉 top 2% token；不能据此做跨模型定量比较。该表的 LLM.int8() 属混合精度（保留异常值通道为高精度），不是纯 INT8 GEMM 对照。
- 1.51×/1.56×/1.96× 是特定后端 + A100 80GB + 指定模型与序列长度下的测量（论文亦指出 FasterTransformer 的 FP16 基线本身就快 3× 以上），不能外推到其它后端、硬件或长度。
- 论文声明只看量化前后的相对变化：Table 3/4 的数值不能当作这些模型在对应任务上的能力基线。
