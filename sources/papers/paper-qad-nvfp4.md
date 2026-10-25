---
{
  "id": "paper-qad-nvfp4",
  "title": "Quantization-Aware Distillation for NVFP4 Inference Accuracy Recovery",
  "category": "paper",
  "url": "https://arxiv.org/abs/2601.20088",
  "version": "v1（2026-01-27）",
  "version_kind": "paper-version",
  "retrieved_at": "2026-10-25",
  "read_form": [
    "html"
  ],
  "read_locator": "摘要、§2.1 NVFP4 Format（NVFP4 位宽、块大小、两级 scaling 与相对 FP8 的收益）、§2.1 Post-Training Quantization 段（PTQ 背景）",
  "topic": "mixed-architecture",
  "excerpt_sha256": null
}
---
## 这份来源是什么

NVIDIA 的技术报告《Quantization-Aware Distillation for NVFP4 Inference Accuracy Recovery》（arXiv 2601.20088，v1）。报告的主题是用量化感知蒸馏（QAD）恢复 NVFP4 量化后 LLM/VLM 的精度；其 §2.1 给出了 NVFP4 格式的定义，是本库 `foundation-number-formats` 中 NVFP4 部分的依据。

范围边界与已知限制：**本页只读了 §2.1 的格式定义与 PTQ 背景段**，未读 QAD 方法（§3）、消融（§4）与附录实验；因此本页只支持"NVFP4 是什么"这一层事实，不支持任何 QAD 效果结论。

## 已读定位与阅读形式

- 阅读形式：arXiv HTML 渲染版（`arxiv.org/html/2601.20088v1`）；未读 LaTeX 源码或 PDF。
- 定位：§2.1 的 "NVFP4 Format" 段（4-bit 浮点、块大小 16、两级 scaling、相对 FP8 的吞吐与显存收益）与同节的 "Post-Training Quantization" 背景段。

## 关键事实

- **NVFP4 是 4-bit 浮点格式**，面向现代 GPU 的训练与推理（§2.1）。
- **相对 FP8**：NVFP4 提供约 **2–3× 的算术吞吐**与约 **1.8× 的显存**下降（§2.1）。
- **NVFP4 扩展自 MXFP4**，改动是两处（§2.1）：块大小由 **32 缩小到 16**，并采用**两级 scaling**——**每块 E4M3 scale 加每 tensor FP32 scale**。
- 论文给出的三点理由（§2.1）：更小的块能更局部地适配数据分布；E4M3 scale 可表达非 2 的幂的缩放因子、降低量化误差；第二级的 FP32 scale 用于扩展整体动态范围。
- 报告指出 PTQ 对多数 LLM 的 FP8 工作良好；对 NVFP4，PTQ 在非常大的模型上也可用（其结论范围见该报告正文与附录）。

## 不能推出的结论

- 本页只读 §2.1，**未读 QAD 方法与实验**；引用本页不能说"QAD 有效""NVFP4 精度已恢复"，也不能引用该报告的任何实验数字。
- NVFP4 的规范定义来自 NVIDIA 官方格式文档（本页未读到），本页记录的是该报告 §2.1 的转述口径；两者若有冲突应以官方规范为准。
- "2–3× 吞吐、1.8× 显存"是报告相对 FP8 的表述，属特定硬件/实现下的比较，不等于任意模型、任意后端的端到端收益。
- NVFP4 的 4-bit 元素格式（E2M1）由"扩展自 MXFP4"推得；报告 §2.1 未逐字给出元素位布局，本页不把它当作已逐位核实的规范。
- 块大小 16 与两级 scale 是 NVFP4 相对 MXFP4 的差异点；不能据此推断所有"FP4"格式都有相同结构。
