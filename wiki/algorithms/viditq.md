---
{
  "id": "algorithm-viditq",
  "title": "ViDiT-Q：视频/图像 Diffusion Transformer 的量化（动态激活 + 时间步感知通道平衡 + 指标解耦混精）",
  "type": "algorithm",
  "tags": [
    "dit",
    "diffusion",
    "mixed-precision",
    "activation",
    "equivalent-transformation",
    "deployment"
  ],
  "aliases": [
    "ViDiT-Q",
    "动态激活量化",
    "时间步感知通道平衡",
    "指标解耦混合精度"
  ],
  "related": [
    "algorithm-ptq4dit",
    "algorithm-qdit",
    "algorithm-smoothquant",
    "guide-architecture-transfer",
    "guide-deployment-evidence",
    "case-paper-evidence"
  ],
  "sources": [
    "paper-viditq",
    "source-viditq",
    "code-viditq"
  ],
  "confidence": "source-reported",
  "evidence_scope": "mixed",
  "updated": "2026-10-28"
}
---

# ViDiT-Q：视频/图像 Diffusion Transformer 的量化（动态激活 + 时间步感知通道平衡 + 指标解耦混精）

## 一句话理解

ViDiT-Q 把扩散模型量化推到 **DiT（diffusion transformer）**：它发现 DiT 激活在 token、CFG 两路、时间步、输入通道四个维度上方差都很大，而常规方法用固定粗粒度量化参数；于是用 per-token 量化、激活参数在线动态计算、时间步感知的通道平衡，再叠加按层类型分配敏感度指标的混合精度，把视频/图像 DiT 做到 W8A8 接近 FP16。

## 核心机制

1. **per-token 量化参数**：线性层不需要 token 间求和，激活按 token 给各自 scale（参数量约为激活的千分之一）。
2. **动态量化**：激活的 min-max 参数每次前向在线计算，直接消掉 CFG 两路与时间步带来的方差。
3. **时间步感知通道平衡**：沿用 `s_i=max(|X_i|)^α / max(|W_i|)^{1−α}` 的通道缩放（与 SmoothQuant/AWQ 同源），但把扩散过程等分成前后两半分别给 `α₁<α`、`α₂>α`，用分布先验在约 10–30 次 PTQ（主文写 20 次，<1 GPU 小时）内调好。
4. **指标解耦混合精度**：先把层影响解耦成"cross-attention≈内容变化、spatial attention 与 FFN≈视觉质量、temporal attention≈时间一致性"，用对应指标做敏感度分析；W6A6-MP 把少量 FFN 保持 FP16，W4A8-MP 把全部 FFN 设为 W8A8、并把前 1/4 时间步也设为 W8A8。

## 基本流程

1. 只量化**线性层**（t/y embedding 与 final layer 因通道小保持 FP16；attention 计算不量化，论文口径）；权重静态、激活动态。
2. 用校准数据在线统计激活范围，per-token 求量化参数。
3. 由校准数据调通道平衡的 `α₁/α₂`（时间步前后两半）。
4. 做指标解耦的层敏感度分析，确定混精方案（哪些层/哪些时间步升位宽）。
5. 用 min-max 量化落地；低比特配置用 INT8 kernel 代理估计延迟。

## 适用条件

- 面向**视频/图像 DiT**（论文：STDiT/OpenSORA、Latte、PixArt-α、PixArt-Σ）；主设置 W8A8、W6A6-MP、W4A8-MP。
- 需要能访问去噪时间步与 CFG 两路前向、能做少量 PTQ 调参与层敏感度分析。
- 量化方案是 min-max、只量化线性层；混精方案按具体模型层名定制，换模型要重跑敏感度分析。

## 结果边界

- 论文报告（VBench 8 维）：ViDiT-Q **W8A8 与 FP16 接近**，而 naive W8A8 在 Imaging、Aesthetic、Overall Consistency 等维度明显掉分。
- UCF-101（STDiT）：FP16 FVD 136.87；naive W8A8 154.92；ViDiT-Q W8A8 141.13；**naive W4A8 崩到 544.34**，而 ViDiT-Q-MP W4A8 做到 129.10。论文提示 101 个视频时 FVD 波动大、FVD-FP16 与视觉感知相关性更好。
- 资源：显存峰值降 2×（W8A8）/2.42×（W4A8-MP）/2.67×（W6A6-MP，主文口径；附录归属不一致）；延迟用 INT8 kernel **估计**，端到端约 1.47×（W8A8）。
- **W6A6/W4A8 没有 GPU kernel**，延迟与 BitOps 是代理口径；只在视频/图像 DiT 上验证，未复现、未在目标后端核验。

## Agent 如何使用

处理"DiT 的激活为什么难量化、低比特为什么崩、混精怎么分配"时先读本页，再按需展开：

| 需要确认的内容 | 继续读取 |
| --- | --- |
| 方法与全部数字（Table 1–4、附录） | [`paper-viditq`](../../sources/papers/paper-viditq.md) |
| LaTeX 源码包口径（公式/表格写法、注释块） | [`source-viditq`](../../sources/papers/source-viditq.md) |
| 官方实现的量化层与 kernel 分层 | [`code-viditq`](../../sources/repos/code-viditq.md) |
| 同为扩散/DiT 量化的其它路线 | [PTQ4DiT](ptq4dit.md)、[Q-DiT](qdit.md) |
| 结构迁移与部署证据 | [结构迁移](../guides/architecture-transfer.md)、[部署证据](../guides/deployment-evidence.md) |

回答时要区分"per-token + 动态激活量化""时间步感知通道平衡""指标解耦混合精度"三类贡献，并明确 **W6A6/W4A8 的延迟是代理估计、没有对应 kernel**，不能当作实测加速。

## 详细来源

- [ViDiT-Q 原论文来源页](../../sources/papers/paper-viditq.md)
- [ViDiT-Q LaTeX 源码包来源页](../../sources/papers/source-viditq.md)
- [ViDiT-Q 官方实现来源页](../../sources/repos/code-viditq.md)
