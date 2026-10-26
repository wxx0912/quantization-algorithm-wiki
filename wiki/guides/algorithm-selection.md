---
{
  "id": "guide-algorithm-selection",
  "title": "从量化问题选择候选方法",
  "type": "guide",
  "tags": [
    "selection",
    "calibration",
    "diagnosis"
  ],
  "aliases": [
    "GPTQ 与 AWQ",
    "算法选择",
    "掉点",
    "权重重构",
    "离群值",
    "怎么选算法",
    "校准"
  ],
  "related": [
    "algorithm-gptq",
    "algorithm-awq",
    "algorithm-baselines",
    "algorithm-smoothquant",
    "algorithm-quarot",
    "algorithm-omniquant"
  ],
  "sources": [
    "paper-awq",
    "repo-awq",
    "paper-gptq",
    "paper-smoothquant",
    "paper-quarot",
    "paper-omniquant",
    "paper-adaround",
    "paper-autoround",
    "paper-flatquant",
    "paper-spinquant",
    "paper-svdquant"
  ],
  "confidence": "source-reported",
  "evidence_scope": "mixed",
  "updated": "2026-10-24"
}
---

# 从量化问题选择候选方法

先确定目标格式和可部署路径，再选择算法。下面是基于算法机制的实验建议，不是从统计现象直接推导出的必胜规则。

| 观察到的问题 | 有理由考虑的候选 | 下一步应验证什么 |
| --- | --- | --- |
| 固定低比特网格下 RTN 误差大 | GPTQ、AdaRound、AutoRound、块重构 | 同数据、同粒度下局部与任务误差是否改善，预算是否可接受 |
| 激活少数通道持续存在大值 | SmoothQuant、AWQ 类缩放、旋转/可学习变换 | 变换浮点等价，权重/激活误差是否只是互相迁移 |
| 固定变换仍不足 | OmniQuant、FlatQuant、SpinQuant | 可学习变量、训练目标、数据覆盖、在线变换成本 |
| 离群分量可由小秩表达 | SVDQuant | 低秩预算与低比特残差误差，融合 kernel 是否抵消分支开销 |
| 少量模块影响明显 | 合法后端精度分配与受控恢复实验 | 在当前候选上恢复模块的收益，不能复用一份永久敏感层名单 |
| 分数正常但不加速 | 部署核验 | 是否 fake quant、fallback、未融合变换或 workload 不匹配 |

## GPTQ 与 AWQ 如何比较

[GPTQ](../algorithms/gptq.md) 利用输入 Gram 矩阵，固定量化列后补偿剩余权重，目标是局部输出重构。[AWQ](../algorithms/awq.md) 利用激活重要性寻找通道缩放，改善组内权重量化，官方实现还包含 clipping 搜索。两者不是“一个用数据、一个不用数据”：它们都使用校准信息，但优化变量与计算成本不同。

比较时固定模型、W/A 格式、group size、排除层、校准样本、导出布局和评测后端。记录各自预处理、搜索和量化耗时。如果一个方案使用更细粒度或特殊 kernel，结果只能归因于整个 Recipe，不能单独归因于算法。

AWQ 原文存在 AWQ+GPTQ 组合实验，见 [组合规则](../recipes/composition.md)。这说明二者并非必须互斥，也不说明该组合在任意架构或后端都可直接用。

## 局部诊断与全局结论

单层权重误差小，不能证明整模型得分高；在浮点模型中只量化某层，也不等于在当前量化候选中恢复该层的收益。为定位问题，可先做同输入 tensor 回放，再做完整自由生成。生成分布改变时，应记录输出长度、异常率和轨迹变化。

若变换前后浮点结果已经不一致，先修结构映射和融合，而不是继续搜索 clipping。如果参考量化和部署结果不一致，先检查数值契约；若二者对齐但质量差，才将问题归因于当前量化方案。

## 形成一项可证伪的候选

记录“观察 → 假设 → 单一主要改动 → 预计可区分的结果 → 质量与成本门槛”。例如保持 W4A16 group 128 不变，仅比较 RTN 和 GPTQ；如果收益只出现在校准数据上，需调查分布覆盖，而不是立即认定重构方法优越。

这里给的是实验设计建议。所有方法的原文证据、预算和局限以各算法页为准；本项目未进行模型实验。
