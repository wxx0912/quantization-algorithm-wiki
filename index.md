# 按问题与机制阅读

先读 [数值契约与术语](wiki/foundations/quantization-contract.md)，确定 W/A/KV、粒度、动态/静态参数和部署目标。类别可交叉；这里的机制是导航，格式、算法、工具和搜索策略仍需分开。

| 机制 / 问题 | 入口 |
| --- | --- |
| 校准与基线 | [RTN、MinMax/AbsMax、MSE](wiki/algorithms/baselines.md) |
| 固定网格上的权重与舍入重构 | [GPTQ](wiki/algorithms/gptq.md)、[AdaRound](wiki/algorithms/adaround.md)、[AutoRound](wiki/algorithms/autoround.md) |
| 激活感知缩放 | [AWQ](wiki/algorithms/awq.md)、[SmoothQuant](wiki/algorithms/smoothquant.md) |
| 旋转与可学习变换 | [QuaRot](wiki/algorithms/quarot.md)、[SpinQuant](wiki/algorithms/spinquant.md)、[FlatQuant](wiki/algorithms/flatquant.md) |
| 块重构与联合优化 | [BRECQ](wiki/algorithms/brecq.md)、[OmniQuant](wiki/algorithms/omniquant.md) |
| 低秩与误差补偿 | [SVDQuant](wiki/algorithms/svdquant.md) |
| 混合精度与成本搜索 | [HAWQ](wiki/algorithms/hawq.md)、[ModelOpt AutoQuantize](wiki/algorithms/modelopt-autoquantize.md) |

## 从问题到方案

1. [如何比较 GPTQ 与 AWQ、诊断掉点](wiki/guides/algorithm-selection.md)。
2. [缩放/旋转后的统计失效与合法组合](wiki/recipes/composition.md)：含 QuaRot+GPTQ、AWQ+GPTQ、OmniQuant LET+LWC 的证据与实施约束。
3. [AR、MoE、dLLM、DiT、多模态迁移](wiki/guides/architecture-transfer.md)：包含 ViDiT-Q 的公式、轨迹与混精案例。
4. [W4A4 是否意味着加速](wiki/guides/deployment-evidence.md)：区分存储、伪量化、kernel、局部与整模性能。
5. [论文案例与实现证据导航](wiki/cases/paper-evidence.md)：按证据范围阅读，具体数值保留在算法页。

[完整稳定 ID 与标签索引](index.generated.md) · [来源索引](sources/README.md) · [来源页规范](sources/SPEC.md) · [Agent 入口](SKILL.md)
