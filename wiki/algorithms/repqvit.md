---
{
  "id": "algorithm-repqvit",
  "title": "RepQ-ViT：ViT 训练后量化的缩放重参数（channel-wise→layer-wise、log√2→log2）",
  "type": "algorithm",
  "tags": [
    "vision",
    "ptq",
    "activation",
    "quantizer",
    "equivalent-transformation",
    "scaling",
    "deployment"
  ],
  "aliases": [
    "RepQ-ViT",
    "缩放重参数",
    "量化推理解耦",
    "post-LayerNorm 重参数",
    "post-Softmax 重参数"
  ],
  "related": [
    "algorithm-ptq4dit",
    "algorithm-baselines",
    "foundation-number-formats",
    "foundation-quantization-contract",
    "guide-search-space"
  ],
  "sources": [
    "paper-repqvit"
  ],
  "confidence": "source-reported",
  "evidence_scope": "source-reported",
  "updated": "2026-10-28"
}
---

# RepQ-ViT：ViT 训练后量化的缩放重参数（channel-wise→layer-wise、log√2→log2）

## 一句话理解

RepQ-ViT 用"量化-推理解耦"解决 ViT 里两个难量化的点位：post-LayerNorm 激活通道间幅度差异大、post-Softmax 激活是幂律分布。它在校准阶段用高精度但难落地的量化器（channel-wise、log√2），再用**缩放重参数**把它们等价改写成硬件友好的量化器（layer-wise 均匀量化、log2 量化）用于推理——不增加推理结构，配少量校准样本、无重建、无超参。

## 核心机制

1. **post-LayerNorm 重参数**：先对激活做 channel-wise 量化得 `s`、`z`，再折成标量 `s̃=E[s]`、`z̃=E[z]`；把差量 `r1=s/s̃`、`r2=z−z̃` 并进 LayerNorm 仿射因子 `β̃=(β+s⊙r2)/r1`、`γ̃=γ/r1`，同时把 `r1⊙W`、`b−(s⊙r2)W` 写进下一层权重。因 `r1` 与权重量化维度不同，权重需重新校准（轻微损失）。
2. **post-Softmax 重参数**：用换底公式把 log√2 量化改成"对 log2 量化乘 2"，反量化时按整数索引奇偶拆 `√2` 项并并入 scale `s̃=s·[1(A^Z)·(√2−1)+1]`，从而既保住 log√2 的精度、又能用位运算。
3. **粒度**：推理时权重 channel-wise、激活 layer-wise 均匀量化；log2 量化器只用于 post-Softmax。

## 基本流程

1. 用少量校准样本（论文：32 张 ImageNet / 1 张 COCO，Percentile）初始化：post-LayerNorm 走 channel-wise 量化，post-Softmax 走 log√2 量化。
2. 对 post-LayerNorm：算 `r1`、`r2` → 改写 LayerNorm 的 `γ/β` → 改写并重新校准下一层 `W/b`，得到 layer-wise 量化。
3. 对 post-Softmax：把量化公式换成 log2（×2），反量化 scale 按奇偶重参数为 `s̃`。
4. 输出量化模型，推理全程只用硬件友好的 layer-wise 均匀量化与 log2 量化。

## 适用条件

- 面向 **ViT 系列**（ViT / DeiT / Swin）的 W4/A4、W6/A6 PTQ；当 post-LayerNorm 存在严重通道间差异、post-Softmax 呈幂律分布时收益最明显。
- 需要能改写 LayerNorm 仿射因子与相邻线性层权重、并能重校准该层权重；校准只需少量样本，无重建、无超参。
- 收益建立在后端支持 layer-wise 激活均匀量化与 log2 量化之上；只做量化器重参数，不含混合精度或 kernel 实现。

## 结果边界

- 论文报告（Table 1，ImageNet Top-1，W4/A4）：ViT-S 65.05、ViT-B 68.48、DeiT-T 57.43、DeiT-S 69.03、DeiT-B 75.61、Swin-S 79.45、Swin-B 78.32（FP 81.39/84.54/72.21/79.85/81.80/83.23/85.27；基线 PTQ4ViT、APQ-ViT 明显更低）。
- W6/A6 接近满精度：DeiT-B 81.27、Swin-S 82.79（相对 FP 只掉 0.53%、0.44%）。
- COCO（Table 2，W4/A4）：Swin-T backbone 上比 APQ-ViT 高 12.4 box / 13.4 mask AP（Mask R-CNN）、19.8 box / 17.0 mask AP（Cascade Mask R-CNN）。
- 消融（Table 3–4）：post-LayerNorm 重参数（DeiT-S 69.03）接近 channel-wise（70.28）、远高于 layer-wise（33.17）；post-Softmax 重参数（DeiT-S 69.03）与 log√2 一致、高于 log2（67.71）。
- 效率（Table 5，单张 3090）：DeiT-S 32 样本 / 1.3 分钟，Swin-S 32 样本 / 2.9 分钟；只报校准耗时，未报推理延迟。未复现、未在目标后端核验。

## Agent 如何使用

处理"ViT 的 post-LayerNorm / post-Softmax 难量化怎么办""怎么把 channel-wise 或 log√2 换成硬件友好的量化器"时先读本页，再按需展开：

| 需要确认的内容 | 继续读取 |
| --- | --- |
| 重参数公式、Table 1–5 与设置 | [`paper-repqvit`](../../sources/papers/paper-repqvit.md) |
| 把同一思路用到 DiT 的后续工作（CSB/SSC） | [PTQ4DiT](ptq4dit.md) |
| 数值格式与 log2/幂律量化口径 | [数值格式](../foundations/number-formats.md)、[量化契约](../foundations/quantization-contract.md) |
| ViT 量化在搜索空间中的位置 | [量化搜索空间](../guides/search-space.md) |

回答时要区分"用复杂量化器拿精度"与"缩放重参数换成简单量化器"两步；重参数是等价变换而非近似，但它依赖后端支持对应量化器，不能当作任意后端都开箱加速。

## 详细来源

- [RepQ-ViT 原论文来源页](../../sources/papers/paper-repqvit.md)
