---
{
  "id": "algorithm-baselines",
  "title": "基础基线：RTN、MinMax、AbsMax 与 MSE 范围校准",
  "type": "algorithm",
  "tags": [
    "rtn",
    "calibration",
    "clipping",
    "quantizer"
  ],
  "aliases": [
    "RTN",
    "MinMax",
    "AbsMax",
    "MSE",
    "最邻近舍入",
    "范围校准"
  ],
  "related": [
    "algorithm-gptq",
    "algorithm-smoothquant",
    "algorithm-omniquant"
  ],
  "sources": [
    "repo-gptq",
    "paper-gptq",
    "paper-smoothquant",
    "paper-omniquant"
  ],
  "confidence": "source-reported",
  "evidence_scope": "mixed",
  "updated": "2026-10-24"
}
---

# 基础基线：RTN 与范围校准

## 一句话理解

MinMax、AbsMax 和 MSE 范围搜索决定量化网格；RTN 在给定网格上取最近格点。它们可以组合，例如 MSE 范围搜索后再 RTN，而不是四个互斥的完整量化算法。

## 核心机制

对均匀 affine 整数量化，步长为 $s$、整数零点为 $z$：

$$q=\mathrm{clip}(\mathrm{round}(x/s)+z,q_{min},q_{max}),\qquad\hat x=s(q-z).$$

- **MinMax**：用组内极值确定范围，affine 编码通常需包含零。
- **AbsMax**：用最大绝对值定义对称范围；具体整数端点依实现而定。
- **MSE 范围搜索**：枚举裁剪范围，选择量化反量化后误差最小的候选；通常按平方误差，实现也可能换用其它误差指数（见下方 caveat）。
- **RTN**：执行最近舍入和饱和截断，不做跨权重误差补偿。

## 基本流程与条件

固定整数格式、分组轴和范围规则 → 收集权重或激活统计 → 确定 scale/zero-point → 舍入与反量化检查 → 评估任务质量。

静态激活范围需要校准，动态范围从当前输入计算；weight-only MinMax+RTN 可直接读取权重。恒定组、tie-breaking 和 scale dtype 按量化器实现处理。更细粒度增加元数据和部署约束，不能只看 bit 数。

## 结果边界

- 公式适用于均匀整数量化，不能直接套到 FP4、FP8 等非均匀码本。
- 名为 `mse` 的实现未必使用平方误差：GPTQ `Quantizer` 默认误差指数为 2.4。
- 局部 tensor MSE 低不保证最终任务质量高；范围搜索也不保证全局最优。
- 工程上应在同模型、同格式、同数据和同评测协议下比较基础基线与更复杂方法。

## Agent 如何使用

使用本页建立可解释基线。量化器公式与范围搜索实现见 [GPTQ 源码](../../sources/repos/repo-gptq.md)；RTN 论文对比见 [GPTQ 论文](../../sources/papers/paper-gptq.md)；静态/动态粒度见 [SmoothQuant](../../sources/papers/paper-smoothquant.md)，学习 clipping 见 [OmniQuant](../../sources/papers/paper-omniquant.md)。本库没有运行模型基线实验。
