---
{
  "id": "paper-brecq",
  "title": "BRECQ: Pushing the Limit of Post-Training Quantization by Block Reconstruction",
  "category": "paper",
  "url": "https://arxiv.org/abs/2102.05426",
  "version": "v2（2021-07-25）",
  "version_kind": "paper-version",
  "retrieved_at": "2026-10-24",
  "read_form": [
    "latex",
    "html"
  ],
  "read_locator": "§2（Eq.1–4）、§3.1（Eq.5–8、Figure 1）、§3.2（Algorithm 1）、§3.3（Eq.9–10）、§3.4（Eq.11）、§4.1（Table 1）、§4.2（Table 2–3）、§4.3（Table 4）、§4.4（Table 5、Figure 2）、§4.5、附录 A（Eq.12–15）、附录 B（Table 6、Figure 3–5、Eq.16–18、Algorithm 2）",
  "topic": "reconstruction",
  "requires_external_verification": true,
  "excerpt_sha256": null
}
---
## 这份来源是什么

BRECQ 原始论文（arXiv 2102.05426，v2；ICLR 2021）。它回答"PTQ 的重构粒度该选多大"：先把二阶误差从权重空间搬到某个粒度的输出空间，再比较网络级/stage 级/block 级/层级，实验选中 **block 级**（块对角 Hessian，保留块内跨层依赖、忽略块间依赖）；重构目标用**对角 Fisher 信息矩阵**代替预激活 Hessian 做加权，以体现输出敏感度。另加混合精度（2/4/8 bit 搜索）与激活量化。

它是 `algorithm-brecq` 的算法依据，也常被当作 `algorithm-adaround` 的后续（内部用 AdaRound 的舍入参数化 + LSQ 的步长更新）。

范围边界：实验是视觉 CNN（ImageNet 分类 + MS COCO 检测），**无 LLM/Transformer 证据**；混合精度主文用模型体积与 FPGA 延迟（模拟器）约束，附录 B.3 另在移动 CPU 延迟上验证（该 CPU 的低位 GEMM 只支持普通卷积）。

## 已读定位与阅读形式

- 阅读形式：arXiv e-print 的 LaTeX 源（单文件 `iclr2021_conference.tex`，正文与附录同文件），另读 arXiv HTML 渲染版交叉核对编号；v2 = 2021-07-25。
- 定位：§3.1 跨层依赖（Eq.5–8）、§3.2 block 重构（Algorithm 1）、§3.3 预激活 Hessian 近似（Eq.9–10）、§3.4 混合精度（Eq.11）、§4 实验（Table 1–5、Figure 1–2）、附录 A（Eq.12–15）与 B（Table 6、Figure 3–5、Eq.16–18、Algorithm 2）。注意 Table 1 来自 `wraptable`，主结果表是 Table 2/3。

## 关键事实

- **跨层依赖与误差搬迁**（§3.1，Eq.5–8）：由 Gauss-Newton 近似 `H^(θ) ≈ Jᵀ H^(z) J`（Eq.6），二阶误差可近似写成某输出层的 `argmin E[Δzᵀ H^(z) Δz]`（Eq.7，Theorem 3.1）；一阶泰勒 `Δz ≈ J·Δθ`（Eq.8）给出直观解释，同样适用于激活量化。
- **四种粒度**（§3.2、Figure 1）：层级（忽略跨层依赖，等价 AdaRound/AdaQuant 一类）、网络级（完整 Hessian，1024 张校准图上易过拟合泛化差）、stage 级与 block 级（中间）。论文选 **block 级**，理由是 Hessian 的主要非对角损失集中在 block 内（残差连接也加剧块内依赖）、块间可忽略；论文明确说该分析**不给"最优粒度"的理论结论**，选择来自实验，好处是无超参、跨模型/任务通用。
- **粒度消融**（§4.1、Table 1；2-bit 权重、首末层除外，ImageNet 准确率）：ResNet-18 层级 65.19、**block 66.39**、stage 66.01、网络级 54.15；MobileNetV2 层级 52.13、**block 59.67**、stage 54.23、网络级 40.76。
- **对角 Fisher 加权**（§3.3，Eq.9–10）：取 `H^(z) = c·I` 丢失太多信息，改用对角 Fisher `F̅ = E[∇log p·∇log pᵀ]`（仅在模型分布=真实数据分布时才等于任务 Hessian，论文承认不可能严格成立），其对角为逐元素梯度平方，目标变为 `min E[Δzᵀ diag((∂L/∂z)²) Δz]`（Eq.10）；取单位权重就退化为输出 MSE。
- **优化器与算法**（§3.2、Algorithm 1、附录 B.4.1）：不绑定优化方法，论文选 **AdaRound 舍入参数化**（`ŵ = s·clip(⌊w/s⌋ + σ(v), n, p)`，Eq.16）优化权重、**LSQ 学习步长**（Eq.18）优化激活量化步长。逐 block：采集输入与 FP 输出 `z` 及其梯度 `g^z`，做 `T` 轮重构（开启激活量化时同步更新步长），再算层内/层间敏感度（2-bit 时才做）供混合精度用。校准只需训练集的一个小子集（论文用 1024 张），单张 GTX 1080 Ti 上约 **20 分钟**得到一个量化 ResNet-18。
- **ImageNet 结果**（§4.2，Table 2–3；首末层 8-bit）。BRECQ **W4A32** 在 ResNet-18/50、MobileNetV2、RegNet-600MF、RegNet-3.2GF、MnasNet-2.0 上为 70.70/76.29/71.66/73.02/78.04/76.00（FP16 71.08/77.00/72.49/73.71/78.36/76.68），**W2A32** 为 66.30/72.40/59.67/65.83/73.88/67.13；同表里 AdaRound\* 在 W2A32 上只剩 55.96–24.70，Bias Correction\*/AdaQuant\* 只剩 0.11–0.49。激活也量化到 4-bit 时 **W4A4** 为 69.60/75.05/66.57/68.33/74.21/73.56、**W2A4** 为 64.80/70.29/53.34/59.31/67.15/63.01，而 ZeroQ/LAPQ/AdaQuant 在 W2A4 上都只有 0.05–0.21。
- **与 QAT 对比**（§4.3，Table 4）：ResNet-18 4/4 下 BRECQ 69.60（1024 样本、0.4 GPU·h）对 PACT 69.2 / DSQ 69.56 / LSQ 71.1（均 1.2M 样本、100 GPU·h）；MobileNetV2 4/4 下 BRECQ 66.57（0.8 h）好于 PACT 61.40、DSQ 64.80（均 192 h）。论文称这种 PTQ 生产量化模型比 QAT 快约 240 倍，并自称首次在不显著掉精度下把权重压到 INT2（W2A32）。
- **检测与混合精度**（§4.4、Table 5；§3.4、Eq.11）：Faster R-CNN 4-bit 权重 + 8-bit 激活，ResNet-18 34.55 → 34.34、ResNet-50 38.55 → 38.29、MobileNetV2 33.44 → 33.18；RetinaNet-ResNet50 36.82 → 36.65；降到 2-bit 权重仍有 31.82 / 34.75，而 AdaRound\* 的 2/8 只有 23.01 / 19.97。混合精度目标 `min_c L(ŵ,c) s.t. H(c) ≤ δ, c ∈ {2,4,8}ⁿ`（Eq.11），因 4/8-bit 几乎无损、只搜 2-bit 排列（含 4 层的 block 也只有 3⁴=81 种），再用遗传算法（Algorithm 2）在硬件阈值下搜索。
- **首末层**（附录 B.1、Table 6；ResNet-18）：首末层也量化几乎无损（W4/8 70.76 vs 不量化 70.58；W2/8 66.30 vs 64.53），与"首末层敏感"的常见说法不同。

## 不能推出的结论

- 论文主体是视觉 CNN 的 ImageNet/COCO，不能据此断言 Transformer/LLM 的最优重构粒度也是 block，也不能外推 Table 2/3 的绝对精度。
- "block 级最好"是实验结论，论文自己说不给理论最优粒度；换模型族、block 定义或任务需重新验证。
- 对角 Fisher 是近似：FIM 等于任务 Hessian 需要模型分布与真实分布一致（论文也承认不可能严格成立），Eq.10 是启发式加权，不能宣称它精确刻画任务损失二阶信息。
- Table 2/3 里带 `*` 的对照（AdaRound/AdaQuant/ZeroQ/Bias Correction）是作者基于开源代码的复现，且各表全精度基线不同（AdaRound 原文报 ResNet-18 69.68、本文报 71.08）；跨论文比"相对增益"会失真。
- "2-bit 近乎无损"要分任务：ImageNet W2A32 上 ResNet-18 从 71.08 掉到 66.30（约 4.8 点），COCO 2/8 上 Faster R-CNN ResNet-18 从 34.55 掉到 31.82。
- 混合精度/延迟结论绑定具体硬件路径（FPGA 模拟器、只支持普通卷积的移动 CPU），不能推断任意硬件或端到端 INT4 加速。
- 论文不含 KV cache 量化，也没有激活 8-bit 之外的系统级实验，不能作为 KV cache 或注意力算子的量化依据。
