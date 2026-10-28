---
{
  "id": "algorithm-ptq4dit",
  "title": "PTQ4DiT：用通道显著性平衡与时间步感知校准量化 Diffusion Transformer",
  "type": "algorithm",
  "tags": [
    "dit",
    "diffusion",
    "ptq",
    "activation",
    "quantizer",
    "equivalent-transformation",
    "calibration"
  ],
  "aliases": [
    "PTQ4DiT",
    "CSB",
    "SSC",
    "通道显著性平衡",
    "Spearman 引导显著性校准"
  ],
  "related": [
    "algorithm-repqvit",
    "algorithm-qdit",
    "algorithm-viditq",
    "algorithm-svdquant",
    "guide-architecture-transfer",
    "guide-search-space"
  ],
  "sources": [
    "paper-ptq4dit"
  ],
  "confidence": "source-reported",
  "evidence_scope": "source-reported",
  "updated": "2026-10-28"
}
---

# PTQ4DiT：用通道显著性平衡与时间步感知校准量化 Diffusion Transformer

## 一句话理解

PTQ4DiT 针对 DiT 量化的两个难点——线性层里激活与权重各有一段极端幅值的显著通道、且显著激活随去噪时间步变化——提出 **CSB**（用对角矩阵把激活与权重的通道显著性互补摊平）与 **SSC**（按时间步与权重的 Spearman 相关性加权聚合激活显著性），再用重参数把平衡矩阵折进 adaLN 与量化/反量化，推理无额外开销。

## 核心机制

1. **CSB（通道显著性平衡）**：通道显著性取 `max|·|`，用激活与权重的几何平均 `s̃=(s(X_j)s(W_j))^{1/2}` 作目标，构造 `B^X=diag(s̃/s(X_j))`、`B^W=diag(s̃/s(W_j))`，使 `X̃=XB^X`、`W̃=B^WW` 后两边通道显著性相等、整体显著性上界下降。
2. **SSC（时间步感知）**：激活显著性随步变化、权重固定；用 Spearman ρ 的归一化指数权重 `η_t∝exp[−ρ(s(X^(t)),s(W))]` 加权聚合得 `s_ρ`，让"与权重越不相关"的时间步权重越大，从而更充分发挥互补。
3. **重参数**：`B^X_ρ`、`B^W_ρ` 互为逆，`X̃·W̃=X·W` 保持等价；离线更新 `W̃`，把 `B^X_ρ` 折进前驱 adaLN 的 `γ/β` 或前驱反量化，故推理无额外计算。

## 基本流程

1. 跨多个时间步生成校准样本（论文：256 分辨率 25 步、512 分辨率 10 步，每步 32 样本），统计激活与权重的通道显著性。
2. 用 SSC 聚合激活显著性，再由 CSB 构造平衡矩阵 `B^X_ρ`、`B^W_ρ`。
3. 重参数：离线把 `B^W_ρ` 并进权重、把 `B^X_ρ` 并进前驱 adaLN/反量化。
4. 量化参数的优化沿用 Q-Diffusion 的实现；再用均匀量化（权重 channel-wise、激活 tensor-wise）量化，重参数在量化前完成等价改写，推理无额外开销——这里的“等价”指重参数改写，不是说量化本身无误差，也不改变位宽。

## 适用条件

- 面向 **DiT（diffusion transformer）** 的 W8A8 / W4A8 PTQ；当显著通道集中在少数通道、且其幅值随时间步变化时收益最明显。
- 需要能访问并改写 adaLN 的 `γ/β` 与前驱线性层/反量化；校准需跨时间步。
- 只做显著性重分配与重参数，不含混合精度与真实低位 kernel；重参数不改变位宽。

## 结果边界

- 论文报告（Table 1，ImageNet 256，FID↓）：250 步 W8A8 下 FP 4.53 → PTQ4DiT 4.63；W4A8 下 Q-Diffusion 15.31、RepQ* 23.21、PTQ4DiT 7.09（论文称 W4A8 的 FID 比 FP 仅增 2.56）。
- 512 分辨率（Table 2，W4A8，250 步 FID）：FP 8.39、Q-Diffusion 58.81、RepQ* 59.65、PTQ4DiT 17.55；论文称比次优的 Q-Diffusion 低 41.26 FID。
- 消融（Table 3，256，W4A8，250 步 FID）：Baseline 22.54、+CSB 8.17、+CSB+SSC 7.09。
- 只用 DiT-XL/2 在 ImageNet 256/512、DDPM 250/100/50 步上验证；校准需跨步，未复现、未在目标后端核验。

## Agent 如何使用

处理"DiT 有哪些显著通道难量化、去噪时间步让分布漂移怎么办"时先读本页，再按需展开：

| 需要确认的内容 | 继续读取 |
| --- | --- |
| CSB/SSC 公式、Table 1–3 与设置 | [`paper-ptq4dit`](../../sources/papers/paper-ptq4dit.md) |
| ViT 上的缩放重参数（本页基线 RepQ* 来源） | [RepQ-ViT](repqvit.md) |
| 同为扩散量化的其它路线 | [Q-DiT](qdit.md)、[ViDiT-Q](viditq.md)、[SVDQuant](svdquant.md) |
| 结构迁移与搜索空间 | [结构迁移](../guides/architecture-transfer.md)、[量化搜索空间](../guides/search-space.md) |

回答时要区分"通道显著性重分配（CSB）""时间步加权（SSC）""重参数（无额外推理开销）"三类贡献；本方法只做重分配，不能推出它能替代混合精度或真实低位 kernel。

## 详细来源

- [PTQ4DiT 原论文来源页](../../sources/papers/paper-ptq4dit.md)
