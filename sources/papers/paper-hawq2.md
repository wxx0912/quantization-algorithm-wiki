---
{
  "id": "paper-hawq2",
  "title": "HAWQ-V2: Hessian Aware trace-Weighted Quantization of Neural Networks（NeurIPS 2020 会议版）",
  "category": "paper",
  "url": "https://proceedings.neurips.cc/paper/2020/file/d77c703536718b95308130ff2e5cf9ee-Paper.pdf",
  "version": "NeurIPS 2020 会议版（2020-12-06）",
  "version_kind": "paper-version",
  "retrieved_at": "2026-10-24",
  "read_form": [
    "pdf"
  ],
  "read_locator": "摘要、§1（Figure 1）、§2（Assumption 1、Lemma 1、Eq.(1)–(10)；Figure 2 在 §2.1、Figure 3 在 §2.2、Figure 4 在 §2.3）、§3.1（Figure 5；正文交叉引用 Table 4 在附录 C）、§3.2（Table 1）、§3.3（Table 2）、§3.4（Table 3）、§4；Table 1/2/3 经 PDF 第 8/9 页文本抽取与页面图像双重核对",
  "topic": "mixed-architecture",
  "requires_external_verification": true,
  "excerpt_sha256": null
}
---
## 这份来源是什么

HAWQ-V2 的正式会议版（NeurIPS 2020），给出用平均 Hessian trace 与权重量化扰动构成敏感度、在 Pareto 前沿自动选混合精度配置的流程与 CNN/检测结果。它是 `algorithm-hawq` 与 `algorithm-modelopt-autoquantize` 混精部分的对照依据。

与预印本 `source-hawq2`（arXiv 1911.03852v1）是同一工作，但会议版数值与预印本不同（如 ResNet50 75.92 vs 75.76），两者不能混用。

范围边界：模型是视觉 CNN（ResNet50、InceptionV3、SqueezeNext）与检测模型 RetinaNet-ResNet50；不含任何 LLM、Transformer 或 KV cache 证据。所有报告结果都经过量化感知微调。

## 已读定位与阅读形式

- 阅读形式：NeurIPS 官方 PDF（正文 12 页，含参考文献，**不含附录**）；第 8/9 页的 Table 1–3 另用页面图像逐格目视复核，正文其余部分通过 PDF 文本抽取阅读。
- 定位：§1 Introduction（Figure 1）、§2 Methodology（Assumption 1、Lemma 1、Eq.(1)–(10)；Figure 2–4 分属 §2.1–2.3）、§3 Empirical Results（3.1 Hutchinson trace 估计 Figure 5、3.2 Table 1、3.3 Table 2、3.4 Table 3）、§4 Conclusions。编号：正文 Eq.(1)–(10)、Figure 1–5、Table 1–3；附录（Table 4 在附录 C、Table 5 在附录 D、Figure 6–8）仅由正文交叉引用推知，**未读到附录本体**。

## 关键事实

- **敏感度定义与排序**（§2、Eq.(4)–(10)）：在 Assumption 1（模型二次可微且收敛）下由 Lemma 1 给出，比"最大 Hessian 特征值"更好的度量是**平均 Hessian trace**：`Ω_i = Tr(H_i)·‖ΔW_i‖²`，整网按 `Ω(b)=Σ_i Ω_i` 排序、取扰动最小的一组。`Tr(H_i)` 对参数量归一化，否则大层仅因参数多就显得更敏感。
- **trace 的估计**（§3.1、Figure 5）：用 Hutchinson 算法（矩阵无关）近似，不显式构造大矩阵；约 **512 个数据点**之后 trace 收敛、**50 次 Hutchinson 迭代**即足够，4 张 GPU 上 **30 分钟**内算完 ResNet50 的 **54 个 block**（平均 33s/block）；相对基于搜索的方法最高快 **120×**（Table 4，附录 C）。
- **ImageNet 结果**（§3.2、Table 1）：ResNet50 FP32 77.39%/97.8 MB → HAWQ-V2（2-bit 权重 + 4-bit 激活混精）**75.92%**/7.99 MB；InceptionV3 77.45%/91.2 MB → **75.98%**/7.57 MB；SqueezeNext 69.38%/10.1 MB → **68.68%**/1.07 MB。表注明确 `MP` 表示该混精配置里**最低**的位宽。
- **检测结果**（§3.3、Table 2；RetinaNet-ResNet50，MS COCO）：FP32 基线 35.6 mAP/145 MB；Direct 4-bit 31.5 mAP；HAWQ-V2 3-bit 权重混精 + 4-bit 激活 **34.1 mAP**/17.90 MB，激活也混精得 **34.4 mAP**、激活提到 6-bit 得 **34.8 mAP**；论文指出检测任务对激活量化更敏感。
- **消融**（§3.4、Table 3；SqueezeNext）：(a) "总扰动大" 67.46%/1.09 MB vs "总扰动小（HAWQ-V2）" **68.68%/1.07 MB**；(b) 只用 `‖ΔW‖²` 的 L2 敏感度 67.83%/1.10 MB，加平均 Hessian trace 后 **68.68%/1.07 MB**（论文称 trace 版本高 0.85% 且模型更小）。
- **Pareto 前沿与自动选位宽**（§2、Figure 4）：给定目标模型大小选 `Ω` 最小的位宽设置；相对 HAWQ 的手工位宽选择，HAWQ-V2 能自动生成整网精度设置并取得更好精度。

## 不能推出的结论

- 论文所有结果都经过**量化感知微调**，不能把准确率读成"无训练 PTQ"的结果。
- Table 1–3 的 `MP` 只是最低位宽，不是平均位宽或全网固定格式；同一模型大小不代表相同延迟，多位宽切换、kernel 与激活存储需单独核算。
- 实验对象是 CNN 与检测网络、没有 LLM/Transformer 证据；不能把"平均 Hessian trace 排序"或 30 分钟预算直接外推到 LLM 的 KV cache、attention 或 MoE。
- trace 估计用"约 512 数据点 + 50 次 Hutchinson 迭代"是经验值、不是通用校准默认值；换模型或任务需重新验证收敛。
- 会议版与预印本 `source-hawq2` 数值不同（ResNet50 75.92 vs 75.76、InceptionV3 75.98 vs 75.68、SqueezeNext 68.68 vs 68.38），引用时须写明版本、不能拼接。
- 论文报告的 **120×** 是与搜索式方法（如 HAQ）比较的**整条选位宽流程**加速，30 分钟/54 块/33s/block 才是 trace 计算本身；两者都不是量化后推理的加速。
