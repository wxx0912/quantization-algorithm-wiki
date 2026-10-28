---
{
  "id": "paper-repqvit",
  "title": "RepQ-ViT: Scale Reparameterization for Post-Training Quantization of Vision Transformers",
  "category": "paper",
  "url": "https://arxiv.org/abs/2212.08254",
  "version": "v1（2022-12-16）",
  "version_kind": "paper-version",
  "retrieved_at": "2026-10-28",
  "read_form": [
    "html"
  ],
  "read_locator": "摘要、§1 引言、§3 总览（Alg.1）、§3.1 预备（Eq.1–11）、§3.2 post-LayerNorm 缩放重参数（Eq.12–17、Figure 2）、§3.3 post-Softmax 缩放重参数（Eq.18–20、Figure 3）、§4.1 设置、§4.2 Table 1、§4.3 Table 2、§4.4 Table 3–4、§4.5 Table 5、§5 结论",
  "topic": "transforms",
  "excerpt_sha256": null
}
---
## 这份来源是什么

RepQ-ViT 论文（arXiv 2212.08254，v1；中科院自动化所 / 国科大），提出面向 **ViT 的训练后量化**框架，核心是**量化-推理解耦**：校准阶段用复杂量化器（post-LayerNorm 走 channel-wise、post-Softmax 走 log√2）拿到好精度，再用**缩放重参数**把它们改写成硬件友好的简单量化器（分别为 layer-wise 均匀量化、log2 量化）用于推理。两种重参数都是等价变换，不增加推理结构。

它是 `algorithm-repqvit` 的一手依据；在本库中作为"把高精度但难落地的量化器，重参数成硬件好实现的量化器"的代表，被 `paper-ptq4dit` 当作 DiT 的强基线（记为 RepQ*）与其 CSB/SSC 对照。

范围边界：权重与激活都量化（论文主打 W4/A4、W6/A6）；权重 channel-wise、激活推理时 layer-wise；校准只用少量样本、无重建、无超参。

## 已读定位与阅读形式

- 阅读形式：arXiv HTML 渲染版（`arxiv.org/html/2212.08254v1`），正文、公式与表格按 HTML 结构解析；未读 LaTeX 源码或 PDF。
- 定位：§3.1 预备（ViT 结构 Eq.1–6、uniform/log2 量化器 Eq.7–11）、§3.2 post-LayerNorm 重参数（Eq.12–17、Figure 2）、§3.3 post-Softmax 重参数（Eq.18–20、Figure 3）、§4.1 设置、§4.2 Table 1（ImageNet）、§4.3 Table 2（COCO）、§4.4 Table 3–4（消融）、§4.5 Table 5（效率）。References 未读。

## 关键事实

- **两类难点**：① **post-LayerNorm 激活**存在严重的通道间幅度差异（Figure 2：DeiT-S 第一模块第 300–350 通道的最小/均值/最大为 3.94/7.11/22.2），单一 scale 的 layer-wise 量化无法覆盖，但 channel-wise 激活量化又不被通用硬件高效支持；② **post-Softmax 激活**是幂律分布（Figure 3：DeiT-S 第一个 MSA 里 99.2% 的激活小于 0.3，少数接近 1），log2 量化器对大幅值描述过粗（举例：`[2^-1.5, 2^-0.5]` 全被舍入到 `2^-1`），而 log√2 量化器更准却无位运算效率。
- **post-LayerNorm 缩放重参数**（§3.2，Eq.12–17）：先做 channel-wise 量化得 `s`、`z`，再重参数为标量 `s̃=E[s]`、`z̃=E[z]`（layer-wise）；定义 `r1=s/s̃`、`r2=z−z̃`，把它们折进 LayerNorm 仿射因子 `β̃=(β+s⊙r2)/r1`、`γ̃=γ/r1`（Eq.15），并调整下一层权重 `W̃_{:,j}=r1⊙W_{:,j}`、`b̃_j=b_j−(s⊙r2)W_{:,j}`（Eq.17）。因 `r1` 与 `W` 的量化维度不同，`W̃` 需**重新校准**，带来轻微精度损失。
- **post-Softmax 缩放重参数**（§3.3，Eq.18–20）：用换底公式把 log√2 的量化改成对 log2 量化乘常数因子 2（Eq.18：`-log_√2(A/s) = -2·log_2(A/s)`）；反量化时 `A^=s·√2^{-A^(Z)}=s·2^{-A^(Z)/2}`，按 `A^(Z)` 的奇偶讨论（Eq.19），最终把系数并入 scale `s̃=s·[1(A^(Z))·(√2−1)+1]`（Eq.20，`1(·)` 为奇偶指示函数，FPGA 上查最低位即可）。这是**精确等价变换**，精度与 log√2 相同，只有极小的额外开销。
- **硬件友好量化器与粒度**（§3.1）：均匀量化 Eq.7–9、log2 量化 Eq.10–11（log2 与 2 的幂可用移位实现）；推理过程权重 channel-wise、激活 layer-wise。
- **设置**（§4.1）：图像分类 ImageNet（ViT、DeiT、Swin），检测/分割 COCO（Mask R-CNN、Cascade Mask R-CNN，Swin-T/S backbone）；分类取 32 张 ImageNet 样本、检测取 1 张 COCO 样本校准；校准用 Percentile，权重 channel-wise、激活 layer-wise；post-LayerNorm 重参数作用在所有 block（含 Swin 的 PatchMerging），post-Softmax 作用在所有 MSA。**无超参数、无重建**。
- **ImageNet 结果**（Table 1，Top-1）：FP 为 ViT-S 81.39 / ViT-B 84.54 / DeiT-T 72.21 / DeiT-S 79.85 / DeiT-B 81.80 / Swin-S 83.23 / Swin-B 85.27。W4/A4 下 FQ-ViT 全部 0.10、PTQ4ViT 42.57/30.69/36.96/34.08/64.39/76.09/74.02、APQ-ViT 47.95/41.41/47.94/43.55/67.48/77.15/76.48，**RepQ-ViT 65.05/68.48/57.43/69.03/75.61/79.45/78.32**；论文称在 ViT-B、DeiT-S 上分别比 APQ-ViT 高 27.07%、25.48%。W6/A6 下 RepQ-ViT 为 80.43/83.62/70.76/78.90/81.27/82.79/84.57（DeiT-B、Swin-S 只掉 0.53%、0.44%）。
- **COCO 结果**（Table 2，W4/A4）：RepQ-ViT 在 Swin-T backbone 上比 APQ-ViT 高 12.4 box AP 与 13.4 mask AP（Mask R-CNN）、19.8 box AP 与 17.0 mask AP（Cascade Mask R-CNN）；W6/A6 下 Cascade Mask R-CNN + Swin-T 达 50.0 box AP / 43.5 mask AP（FP 50.4/43.7）。论文指出 PTQ4ViT 的 twin-scale 搜索在高阶任务上失效、APQ-ViT 对 backbone 不鲁棒。
- **消融**（Table 3–4，W4/A4）：post-LayerNorm 上 DeiT-S 从 FP 79.85 到 layer-wise 33.17 / channel-wise 70.28 / 缩放重参数 69.03（Swin-S 依次 83.23/57.63/80.52/79.45）；post-Softmax 上 DeiT-S 从 log2 67.71 / log√2 69.03 到重参数 69.03（Swin-S 77.87/79.45/79.45）——重参数与 log√2 精度一致。
- **效率**（Table 5，单张 3090 GPU，W4/A4）：DeiT-S RepQ-ViT 69.03%（32 样本、1.3 分钟），对比 PTQ4ViT 34.08%（32 样本、3.2 分钟）；Swin-S 79.45%（32 样本、2.9 分钟），对比 PTQ4ViT 76.09%（32 样本、7.7 分钟）。论文称是首个把 ViT 的 4-bit PTQ 做到可用水平的工作。

## 不能推出的结论

- 本页只读正文与所列表格，未复现实验、未运行；精度、数据量与耗时均为论文报告值。
- 结论只在 ViT/DeiT/Swin 与 ImageNet（分类）、COCO（检测/分割）上验证；不能外推到 LLM、DiT/dLLM 或视频模型的表现（DiT 上的效果见 `paper-ptq4dit` 的 RepQ* 复现）。
- 缩放重参数是**等价变换**，但它把"高精度量化器"换成"硬件友好量化器"，收益建立在后端确实支持 layer-wise 均匀量化与 log2 量化之上；不能当作任意后端都开箱加速。
- post-LayerNorm 的 `W̃` 需重新校准，相对 channel-wise 有轻微精度损失（论文报告约 1.25%，DeiT-S）；不能宣称与 channel-wise 完全无损。
- 校准规模（32 张 ImageNet / 1 张 COCO）与 Percentile 方法是论文设定；换数据集或分辨率需重确。
- 论文用 Top-1 精度与 AP 报告，未报告端到端延迟或硬件 kernel 实测；耗时是量化（校准）耗时，不是推理加速。
