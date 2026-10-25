---
{
  "id": "source-viditq",
  "title": "ViDiT-Q LaTeX 源码包（arXiv src 2406.02540v2）",
  "category": "paper",
  "url": "https://arxiv.org/src/2406.02540v2",
  "version": "v2（2024-06-30）",
  "version_kind": "paper-version",
  "retrieved_at": "2026-10-24",
  "read_form": [
    "latex"
  ],
  "read_locator": "源码包内 ms.tex 全文（正文 + \\appendix 之后的三节附录）、ms.tex:320 的 Eq.1、tab:main_vbench / tab:main_ucf / tab:main_quant_comparison / tab:ablation、ms.tex:621 的 fig:hardware 内嵌表格、ms.tex:488 的 fig:main_images 内嵌表格、figs/ 目录清单",
  "topic": "mixed-architecture",
  "excerpt_sha256": null
}
---
## 这份来源是什么

ViDiT-Q 的 **LaTeX 源码包**（与 `paper-viditq` 同论文、同版本 v2）。单独登记为一条来源，是因为它提供论文正式版式之外的信息：公式与表格的源码写法、哪些内容是注释掉的、源码包到底包含哪些文件。**论文本身的内容与结论以 `paper-viditq` 为准**，本页不重复。

## 已读定位与阅读形式

- 阅读形式：arXiv e-print（`arxiv.org/src/2406.02540v2`）解包后的 `ms.tex`（1399 行，正文与附录同文件）与 `figs/`（27 个条目，含 `imagediff/`、`jpgs/`、`pngs/` 三个子目录）。
- 定位口径：按源码行号与浮动体标签（`\label`）引用，不引用行号之外的外部页码。

## 关键事实

- **单文件结构**：`ms.tex` 里正文与附录同文件；`\appendix` 在 731 行，其后依次是 §A Hardware Experiments Details（A.1 profiling 设置、A.2 只量化线性层的理由、A.3 量化成本、A.4 资源节省估计）、§B 评测指标详述、§C 实验结果详细分析。
- **Eq.1 的源码形态**：`ms.tex:320` 只有一个 `equation` 环境、内部用 `\\` 折成两行（`Y = (X diag(s)^{-1})·(diag(s)W) = X̂·Ŵ`；`s_i = max(|X_i|)^α / max(|W_i|)^{1-α}`）。它**编号为一个公式（Eq.1）**，不要按"两行"数成两个。
- **表格清单**：编号表只有 4 个，全在正文——`tab:main_vbench`（Table 1，VBench）、`tab:main_ucf`（Table 2，UCF-101）、`tab:main_quant_comparison`（Table 3）、`tab:ablation`（Table 4，消融）；**附录没有任何编号表**。
- **只在源码里才看得出来的表格**：`ms.tex:621` 的 `fig:hardware`（Figure 8）是一个 `figure` 环境，内嵌一张**四列**表（`Bit-width (W/A)` + BitOps/Memory/Latency；值 4.00×/1.99×/1.47×、7.11×/2.67×/–、5.05×/2.42×/–，后两行的 Latency 列为 `-`）；`ms.tex:488` 的 `fig:main_images`（Figure 7）的 `minipage` 里也嵌了一张无编号表（PixArt-α/Σ 的 FID/CLIP/IR，其 `\caption` 在 495 行被注释）。两者都**没有 Table 编号**，引用时不能写成"Table X"。
- **注释块不计入正文**：源码有多处被 `%` 注释掉的浮动体与段落，例如 `tab:main_ucf`（Table 2）表体里 PTQ4DM / ViDiT-Q-MP `6/6` 的两行（`ms.tex:403-404`；同类 `6/6` 行在 Table 3 中有活动版本，见 `ms.tex:473-475`，`:475` 为 ViDiT-Q-MP）、一份被注释的硬件节省表（`ms.tex:786-796`，是活动表的变体——`6/6-MP` 与 `4/8-MP` 的 Latency 列都填 1.47×，活动表填 `-`）、被注释的 `\section{Analysis}`/`\subsection{Ablation Studies}` 废弃小节标题（`ms.tex:664`、`667`）、以及若干被整段注释的表格与图。这些都不属于论文内容。
- **图表计数提示**：`ms.tex` 里**活动**的 `\begin{subfigure}` 只有 15 个（含被注释的共 20 个），subfigure **不推进 figure 计数器**；图的编号按 `\begin{figure}` 计是 19 个（正文 8、附录 11），含被注释的浮动体共 22 个。
- **数值口径与缺失物**：`tab:main_vbench` 的 STDiT 段、`tab:main_ucf` 的 FVD/FVD-FP16/VQA/Flow 与正文叙述一致；`tab:main_quant_comparison` 的 baseline 行是作者把 PTQ4DM/Q-Diffusion/ZeroQuant/SmoothQuant 等**自行适配到 DiT** 后测的，表中没有说明这些 baseline 是否用了各自最优配置。源码包解包后是 `ms.tex`、`neurips_2024.sty`、`.bib`/`.bbl`、`figs/` 图片，另有一份编译产物 `neurips_2024.pdf`，**没有量化 kernel 或训练/推理代码**。

## 不能推出的结论

- 源码包不等于论文的最终版式：`ms.tex` 里被 `%` 注释的表格、段落与小节标题都不属于论文内容，不能当作论文主张引用。
- 源码包里没有实现代码，**不能据此推断**任何实现细节（如动态量化的 minmax kernel、混精调度的具体写法），这些只有官方代码库才能回答。
- 本页只确认"源码怎么写的"，不独立复算任何图像/视频指标；论文口径的结论请查 `paper-viditq`。
- 与 `paper-viditq` 是同版本同论文，两条来源不能当作两份独立证据互相印证。
