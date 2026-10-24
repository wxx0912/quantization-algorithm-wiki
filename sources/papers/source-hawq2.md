---
{
  "id": "source-hawq2",
  "title": "HAWQ-V2: Hessian Aware trace-Weighted Quantization of Neural Networks（arXiv 1911.03852v1 预印本 LaTeX）",
  "category": "paper",
  "url": "https://arxiv.org/abs/1911.03852",
  "version": "v1（2019-11-10）",
  "version_kind": "paper-version",
  "retrieved_at": "2026-10-24",
  "read_form": [
    "latex",
    "html"
  ],
  "read_locator": "s2_method.tex（Assumption 1、Lemma 1、敏感度与 Ω 定义）、s3_results.tex（ResNet50/InceptionV3/SqueezeNext/RetinaNet 结果数值与图注）、s5_conclusion.tex、s6_appendix.tex（量化器与 STE、搜索空间、Extra Results 图注）；arxiv.org/abs/1911.03852（Submission history 确认只有 v1）",
  "topic": "mixed-architecture",
  "requires_external_verification": true,
  "excerpt_sha256": null
}
---
## 这份来源是什么

HAWQ-V2 预印本（arXiv 1911.03852）v1 的 LaTeX 源码，用于核对会议版之外的量化器定义、STE 使用、搜索空间公式与图注设置。该 arXiv 条目**只有 v1**（2019-11-10），没有更新版本。与 `paper-hawq2` 是同一工作的不同版本，两个 ID 均保留。

## 已读定位与阅读形式

- 阅读形式：arXiv e-print 的 LaTeX 源（`arxiv.tex` + `s0_abstract`…`s6_appendix`；`s4_ablation_experiments.tex` 只有 110 字节且未被 `arxiv.tex` 引入）；版本信息另见 arxiv.org/abs/1911.03852。
- 编号提示：源码用 `leqno` 把公式号放左侧；`s6_appendix.tex` 里的量化器与搜索空间公式为无编号的 `align*`/`equation*`，不计入正文编号。

## 关键事实

- **量化器**（s6_appendix.tex §A）：前向先 `Clamp(X, q_0, q_{2^k−1})`，再以 `Δ=(q_{2^k−1}−q_0)/(2^k−1)` 舍入到整数索引 `X^I`，再反量化 `Q(X)=ΔX^I+q_0`；量化范围 `[q_0, q_{2^k−1}]` 可取 `[min,max]` 的子区间以裁剪 outlier。推理时用整数索引做矩阵乘再统一反量化；因为量化感知微调，反向用 STE。
- **搜索空间**（s6_appendix.tex §B）：位宽组合数 `B = m^L`，逐层微调顺序数用第二类 Stirling 数表示；只按平均 Hessian trace 排序时，混精问题可归约为整数划分。
- **结果数值**（s3_results.tex）：预印本报告 InceptionV3 **75.68%**、ResNet50 **75.76%**、SqueezeNext **68.38%**（基线 69.38%/10.1 MB，量化后 1.07 MB），与会议版数值不同。
- **校准收敛证据**（s3_results.tex、s6_appendix.tex）：§3.1 **正文**称约 **512 个数据点**后 trace 收敛、50 次 Hutchinson 迭代足以达到低方差近似（此说法在预印本与会议版正文中均有，**并非预印本独有、也非图注**）；RetinaNet 的激活 trace **图注**另写明其左图用 **128 个数据点、50 次 Hutchinson 迭代**，这一**组合**才只在预印本 LaTeX（`s6_appendix.tex`）图注里可定位。会议版的 trace 成本数字（30 分钟/54 块/33s）与加速口径见 `paper-hawq2`，本页不重复。

## 不能推出的结论

- 本页与 `paper-hawq2` 是同一工作的两个版本；版本数值差异与引用口径只在 `paper-hawq2` 维护一处，本页只补充预印本 LaTeX 侧独有的量化器/搜索空间/图注证据，两页不重复维护同一结论。
- 128 数据点 / 50 迭代只出现在 RetinaNet 的图注，不是统一校准默认值，也不代表其它模型的收敛要求（同文 §3.1 另报告约 512 个数据点收敛）。
- 附录的量化器与 STE 描述是 2019 年的 min/max 子区间裁剪 + 直通估计方案（正文对 RetinaNet 权重用 channel-wise 量化），不能外推为现代 group-128、per-channel 或 per-token 量化配置。
- 本条目只有 LaTeX 静态证据，未运行训练或推理；`s4_ablation_experiments.tex` 未生效，不能当作论文内容。
