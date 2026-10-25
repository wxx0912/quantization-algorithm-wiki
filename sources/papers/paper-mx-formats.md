---
{
  "id": "paper-mx-formats",
  "title": "Microscaling Data Formats for Deep Learning",
  "category": "paper",
  "url": "https://arxiv.org/abs/2310.10537",
  "version": "v1（2023-10-16）",
  "version_kind": "paper-version",
  "retrieved_at": "2026-10-25",
  "read_form": [
    "html"
  ],
  "read_locator": "摘要、§1 动机、§2 Microscaling（MX 块定义、共享 scale 规则）、§2.2 具体 MX 格式与 Table 1（块大小/scale 格式/元素格式）、§3 转换算法、§4.2 转换方法与舍入模式、§4.3–§4.5 结果章结论",
  "topic": "core",
  "excerpt_sha256": null
}
---
## 这份来源是什么

《Microscaling Data Formats for Deep Learning》（arXiv 2310.10537，v1）。它定义 Microscaling（MX）格式族：一个"MX 块"由固定数量的元素共享一个 scale，块由**块大小、scale 数据格式、元素数据格式**三者共同定义；论文给出 MXFP8/MXFP6/MXFP4/MXINT8 的参数与转换方法，并评估其作为 FP32 替代的可行性与"sub-8-bit 训练"。

它是 `foundation-number-formats` 中 MX 部分与"为什么会出现块内共享 scale"的依据，也是理解 NVFP4（MXFP4 的扩展）的前置。

范围边界：**格式规范与实证评估**；规范的规范文本在 OCP Microscaling Specification，本页只读了论文。

## 已读定位与阅读形式

- 阅读形式：arXiv HTML 渲染版（`arxiv.org/html/2310.10537v1`），正文与表格按 HTML 结构解析；未读 LaTeX 源码、PDF 或 OCP 规范原文。
- 定位：§2 的 MX 块定义与共享 scale 规则、§2.2 与 Table 1 的具体格式参数、§3 的转换算法（Algorithm 1）、§4.2 的舍入模式、§4.3–§4.5 的结论段落。

## 关键事实

- **MX 块**（§2）：一个基本数据单元表示 `k` 个元素的向量，由**一个共享 scale `X`** 与 `k` 个元素组成；块由"块大小 `k`、scale 数据格式、元素数据格式"定义，两种格式相互独立，`k` 个元素共享同一元素格式。
- **具体格式**（Table 1）：MXFP8（块 32、scale E8M0、元素 FP8 E4M3/E5M2）、MXFP6（块 32、E8M0、FP6 E2M3/E3M2）、MXFP4（块 32、E8M0、FP4 E2M1）、MXINT8（块 32、E8M0、INT8）。所有具体 MX 格式都用 **E8M0**（8-bit 指数）作为共享 scale。
- **E8M0 与特殊值**（§2）：共享 scale `X` 不编码 Inf；元素可各自编码 NaN；部分元素格式可由元素编码 Inf。E8M0 可表示的指数是 FP32 指数范围的超集。
- **转换**：§3 给出 Algorithm 1 的转换流程，并说明转换不必沿用它；推理实验的转换用 round-half-to-nearest-even（§4.2），生成式训练实验改用 round-half-away-from-zero（§4.5）。
- **结论**（结果章）：MXINT8 可作为 FP32 的近无损直接替代；MXFP8/MXFP6 的直接转换推理中"尾数更多的变体更好"；论文报告了 sub-8-bit 权重、激活、梯度训练生成式语言模型的首次结果。

## 不能推出的结论

- 本页只读论文，**未读 OCP Microscaling Specification 原文**；MX 的规范性定义以 OCP 规范为准，本页只记录论文口径。
- 论文只称 E8M0 为 8-bit exponent，并把其细节外推到 OCP Microscaling Specification；"只有指数、无尾数、scale 是 2 的幂"是按 E8M0 命名推得，不是本页读到的论文原文结论。它与 NVFP4 用 E4M3 做块内 scale（可含尾数、非 2 的幂）不同，不能把两者混为一谈。
- 论文的精度结论绑定其转换算法（round-half-to-nearest-even）与实验设置；换量化器或换粒度不能直接沿用。
- "块 32"是论文 Table 1 的取值；块大小是格式参数之一，不能当成 MX 的固定定义，也不能据此推断某硬件一定支持 32 之外的块。
- 论文报告的是研究结果，不构成"某后端已有真实低比特 kernel"的证据；实际吞吐与 kernel 需另找部署证据。
