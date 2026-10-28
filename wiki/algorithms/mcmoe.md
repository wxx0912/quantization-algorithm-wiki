---
{
  "id": "algorithm-mcmoe",
  "title": "MC-MoE：MoE 专家级混合精度量化 + 动态剪枝的免训练压缩",
  "type": "algorithm",
  "tags": [
    "moe",
    "mixed-precision",
    "weight-only",
    "sensitivity",
    "calibration",
    "cost-constraint",
    "deployment",
    "ptq"
  ],
  "aliases": [
    "MC-MoE",
    "Mixture Compressor",
    "PMQ",
    "ODP",
    "专家级混合精度量化",
    "MoE 动态专家剪枝"
  ],
  "related": [
    "algorithm-moequant",
    "algorithm-gptq",
    "algorithm-baselines",
    "foundation-quantization-contract",
    "guide-search-space"
  ],
  "sources": [
    "paper-mcmoe"
  ],
  "confidence": "source-reported",
  "evidence_scope": "source-reported",
  "updated": "2026-10-28"
}
---

# MC-MoE：MoE 专家级混合精度量化 + 动态剪枝的免训练压缩

## 一句话理解

MC-MoE 抓住"MoE 各专家的重要性差异很大、且只有少量 token 关键"，做两段免训练压缩：加载前用 **PMQ** 按专家重要度做专家级混合精度量化（专家位宽取自 `{1,2,3}`，用整数规划在目标平均位宽下最小化加权重构误差），推理时用 **ODP** 剪掉低置信专家、但保护约 2% 的重要 token 以免注意力塌陷。基座量化器是 GPTQ。

## 核心机制

1. **专家显著性**：单个专家的显著性为 `φ^α·w^β`，`φ` 是激活频率、`w` 是平均路由权重；论文用 Mixtral-8×7b 的逐专家热图显示二者与重构误差都高度不均。
2. **PMQ（位宽分配）**：对"仅把专家 `e_i` 量化到 `j` 位"测输出 F-norm 误差 `ε_{i,j}`，解整数规划 `min Σ_i Σ_j φ^α w^β (ε_{i,j}x_{ij})^γ`，约束平均位宽 `ΣΣ j·x_{ij}=n·k`、每个专家一个位宽、且至少各 1 个专家取 3-bit 与 2-bit；解完用 GPTQ 量化，attention/gating 统一 4-bit（平均位宽增量 ≤0.05 bit）。
3. **ODP（动态剪枝 + token 保护）**：先按纯路由权重剪枝（`w1/w0<μ` 时剪掉较弱的 top-2 专家），会导致重要 token 的"注意力衰减"；再用 `I_j=‖t_j‖_1·注意力分数均值`（Eq.6）打分保护约 2% 的关键 token，使其专家不被剪。

## 基本流程

1. 用一批校准数据（论文：C4、128×2048 token）跑 16-bit 前向，统计每个专家的激活频率与平均路由权重，并测各专家量化到 1/2/3 位的输出误差。
2. 解整数规划得到每个 MoE block 内各专家的位宽，用 GPTQ 按位宽量化专家；其余层 4-bit。
3. 推理时对每个 token 做 gating 加权剪枝，同时对重要 token 跳过剪枝；量化权重由 HQQ 保存/反量化。

## 适用条件

- 面向 **MoE LLM 的权重压缩**（专家 1/2/3-bit 混精 + 动态剪枝）；专家重要性差异大、存在长期不被激活的专家时收益最明显。
- 需要一小段校准数据，且能访问 gating 权重与每层注意力图（token 保护用 Eq.6 的注意力分数）。
- 只压权重、免训练；ODP 会改变推理时的激活专家路径，需要后端支持动态专家选择（工程约束）。

## 结果边界

- 论文报告（Table 2，Mixtral-8×7b，8 任务 avg.）：16-bit FP 71.29、均匀 2-bit 42.67、BSP 2.54-bit 49.07、PMQ 2.54-bit 67.50（−3.8%），1.57-bit 仍为 54.49。
- MMLU 5-shot（Table 3）：FP 70.60、均匀 2-bit 34.05、PMQ 2.54-bit 61.19。
- 显存/效率（Table 4）：Mixtral-8×7b 从 96.80GB（16-bit）压到 16.24GB（2.54-bit）；叠加 ODP 后激活参数 4.53GB→3.96GB、加速 1.63×→1.71×，平均精度下降不到 1%。
- ODP 消融：保护 2% 关键 token 使 PPL 由 6.46 降到 6.24，激活专家参数仅由 15.1% 降到 14.8%。
- 平均位宽是专家层口径且其余层固定 4-bit；只做权重、免训练，未复现、未在目标后端核验。

## Agent 如何使用

处理"MoE 专家这么多、能不能按重要性给不同位宽""MoE 推理能不能少激活专家"时先读本页，再按需展开：

| 需要确认的内容 | 继续读取 |
| --- | --- |
| PMQ/ODP 公式、Table 1–4 与设置 | [`paper-mcmoe`](../../sources/papers/paper-mcmoe.md) |
| 同样针对 MoE、但只做加权校准的路线 | [MoEQuant](moequant.md) |
| 混合精度与成本预算的通用口径 | [量化契约](../foundations/quantization-contract.md)、[基线](baselines.md) |
| MoE 压缩在搜索空间中的位置 | [量化搜索空间](../guides/search-space.md) |

回答时要区分"专家级位宽分配（PMQ）""推理时动态剪枝（ODP）""基座量化器（GPTQ）"三类贡献；本方法只针对权重，平均位宽是专家层预算，不能当成整模型位宽或真实低比特 GEMM 证据。

## 详细来源

- [MC-MoE 原论文来源页](../../sources/papers/paper-mcmoe.md)
