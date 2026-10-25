---
{
  "id": "paper-svdquant",
  "title": "SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models",
  "category": "paper",
  "url": "https://arxiv.org/abs/2411.05007",
  "version": "v1（2024-11-07）；v3（ICLR 2025 camera-ready）",
  "version_kind": "paper-version",
  "retrieved_at": "2026-10-24",
  "read_form": [
    "latex"
  ],
  "read_locator": "sections/method.tex（§4，Eq.3–7、Proposition 4.1/4.2）、sections/experiments.tex（§5）、table_texts/quality.tex（Table 1）、figure_texts/{distribution,singular_values,svd_fusion,visual,efficiency,lora,ablation,ablation-rank}.tex（Figure 4–11）、sections/appendix.tex（Eq.8–10、Table 2、Figure 12–17）；编号口径 Figure 1–17、Table 1–2、Eq.1–10",
  "topic": "transforms",
  "excerpt_sha256": null
}
---
# SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models

## 这份来源是什么

SVDQuant 原始论文（arXiv 2411.05007；本页以 v1 为基线，并标注 v3 camera-ready 的关键差异），给出"通道缩放 + 截断 SVD 低秩分支 + 4-bit 残差分支"的 W4A4 分解、误差分析、配套推理引擎与扩散模型结果，是 `algorithm-svdquant`、`guide-deployment-evidence` 与 `recipe-composition` 的依据。

## 已读定位与阅读形式

- 本轮复核：实际读取 v1 LaTeX 的方法、实验设置、Table 1 和性能图注，核对本页公式、格式与引用数字；未运行推理。
- 阅读形式：arXiv e-print 的 LaTeX 源（`main.tex` + `sections/*` + `figure_texts/*` + `table_texts/*`），逐文件读取，未运行代码或复现实验。
- 定位：§4 `sections/method.tex`（Eq.3–7、Proposition 4.1/4.2、引擎融合 §4.3）、§5 `sections/experiments.tex`、`table_texts/quality.tex`（Table 1）、`sections/appendix.tex`（Eq.8–10、Table 2、Figure 12–17）。编号：Figure 1–17（正文 1–11、附录 12–17）、Table 1–2、Eq.1–10（正文 1–7、附录 8–10，附录 `:36` 的 align 两行都编号），**无 Algorithm**。

## 关键事实

- **误差分解**（§4.1，Eq.3–4）：`E(X, W) = ‖XW − Q(X)Q(W)‖_F`；Proposition 4.1 给出上界，由权重/输入的量级（`‖W‖_F`、`‖X‖_F`）与各自的量化误差四项共同界定。
- **异常值从激活迁到权重**（§4.2，Figure 4）：采用 `X∈R^{b×m}`、`W∈R^{m×n}` 时，逐通道缩放为 `X̂ = X·diag(λ)⁻¹`、`Ŵ = diag(λ)·W`（原文此处右乘写法与其维度定义不一致，本页按维度和 `XW=X̂Ŵ` 恒等式更正） 降低激活侧误差，但权重侧量级与异常值同时上升，整体收益受限。
- **低秩分解与 SVD 求解**（§4.2，Eq.5，Figure 5）：把 `Ŵ` 分解为 `Ŵ = L₁L₂ + R`，低秩支路以 16-bit 算 `X̂L₁L₂`、残差支路用 4-bit `Q(X̂)Q(R)`；秩 `r` 通常取 16 或 32，低秩分支的额外参数/计算占比为 `(mr+nr)/mn`。以 `min‖Ŵ − L₁L₂‖_F` 为目标，最优解为 `L₁ = UΣ_{:,:r}`、`L₂ = V_{:r,:}`；由 Eckart–Young–Mirsky 得 `‖R‖_F = sqrt(Σ_{i>r} σ_i²)`。实践上迭代"分解 `W − Q(R)` 并调整 `R`"若干轮，取误差最小的一轮。
- **误差上界**（§4.2，Proposition 4.2 / Eq.7）：在**假设 `R` 元素服从正态分布**的前提下，`E‖R − Q(R)‖_F ≤ (sqrt(log(size(R))π)/q_max)·E‖R‖_F`，即量化误差被残差量级 `‖R‖_F` 控制。
- **引擎融合**（§4.3，Figure 6）：单独运行低秩分支约增加 **50%** 的 4-bit 分支延迟（同图图注写作 57%，瓶颈从算力转为访存）；把 down-projection 与量化 kernel、up-projection 与 4-bit 计算 kernel 融合后，低秩分支只额外增加 **5–10%** 延迟。
- **实现细节**（§5.1）：8-bit 用 per-token 动态激活 + per-channel 权重 + rank 16；4-bit 用 per-group 对称 + rank 32；INT4 用 group 64 + 16-bit scales；v1 的 FP4 用 group 32 + FP8 scales，v3 camera-ready 改用 NVFP4（group 16 + FP8 scales，Blackwell 原生）并对残差权重采用 GPTQ。FLUX.1 自适应归一化层的 linear 输入保 16-bit（该层为 W4A16），其他模型 cross-attention 的 K/V 投影保 16-bit（延迟占比 <5%）。
- **模型与数据**（§5.1）：FLUX.1（19 joint + 38 parallel attention，约 12B，50 步 dev / 4 步 schnell）、PixArt-Σ（600M，20 步）、SDXL（2.6B，30 步）与 SDXL-Turbo（4 步）；校准用 COCO prompts（原文写作 COCO Captions 2024，但引用为 Chen et al. 2015；不据此认定存在独立的 2024 数据集版本），评测用 MJHQ-30K 的 5K prompts 与 sDCI 的 5K prompts；指标为 FID（越低越好）、Image Reward（越高越好）、LPIPS、PSNR。
- **质量与效率结果**（§5.2，Table 1、Figure 8）：8-bit 的 PSNR 均高于 21；FLUX.1-dev 50 步 MJHQ——BF16 FID 20.3 / IR 0.953，INT W4A4 为 19.9 / 0.932（LPIPS 0.254、PSNR 20.1）。FLUX.1 transformer 模型大小从 **22.2 GiB 降到 6.1 GiB**（含低秩分支 0.3 GiB），约 **3.6×** 模型大小压缩；Figure 8 图注另报告相对 16-bit 模型 **3.5× 运行显存缩减**，两者不是同一指标。桌面及笔记本 RTX 4090 上相对 NF4 W4A16 约 **3.0×** 加速；16GB 笔记本 4090 上 BF16 需逐层 CPU offload、INT4 全量驻留，端到端约 **10.1×** 加速。
- **基线与消融**（§5.1–5.2，Figure 10–11）：基线为 NF4、ViDiT-Q、MixDQ、TensorRT；消融显示 SVD-only、naive、仅 smoothing、LoRC 均不佳；秩从 16 增到 64 可显著提升质量，论文取 rank 32。

## 不能推出的结论

- Proposition 4.2 的上界建立在"`R` 元素服从正态分布"的假设上，不是任意网络或量化器下的无条件保证；以 `min‖R‖_F` 为目标只是该假设下的代理标准，不等于全局最优。
- 全部质量结论只覆盖文本到图像扩散模型（FLUX.1、PixArt-Σ、SDXL/SDXL-Turbo），不能外推到文本 dLLM、自回归语言模型或 MoE。
- FID/IR/LPIPS/PSNR 是分布与相似度指标，单个指标略优不能解释为逐张图像更好，需与定性样例（Figure 7、12–17）一并看。
- 3.0× 与 10.1× 是绑定 NVIDIA RTX 4090、特定基线（NF4/逐层 offload）与实现下的结果，不是 W4A4 的通用速度或显存预期。
- LoRA 集成（Figure 9）与秩消融（Figure 11）只在其给定配置下成立，不能当作其它秩、其它 LoRA 或其它模型的行为。
- 低秩分支 5–10% 的额外开销来自该引擎的 kernel 融合，未融合实现不能按此数字估计。
