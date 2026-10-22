---
{"id":"algorithm-adaround","title":"AdaRound：固定量化网格上的自适应舍入","type":"algorithm","tags":["ptq","rounding","reconstruction","vision"],"aliases":["Adaptive Rounding","Up or Down"],"related":["algorithm-brecq","algorithm-autoround"],"sources":["paper-adaround","repo-adaround"],"updated":"2026-10-23"}
---

# AdaRound：固定量化网格上的自适应舍入

AdaRound 解决的是：给定一个量化网格，每个权重应向上还是向下舍入，才能降低层输出误差。逐权重 round-to-nearest 只考虑独立的权重误差；多个误差对输出可能互相抵消，也可能同向累积。AdaRound 学习舍入变量，不对原模型进行端到端任务训练。

## 算法计算

[论文 v2 §3.1–3.3](https://arxiv.org/pdf/2004.10568v2)从任务损失二阶近似出发，将问题简化为层级重构，并通过连续松弛优化离散舍入。对固定步长 $s$、整数上下界 $n,p$：

$$\widetilde W=s\,\operatorname{clip}(\lfloor W/s\rfloor+h(V),n,p),$$
$$h(V)=\operatorname{clip}(\sigma(V)(\zeta-\gamma)+\gamma,0,1),\quad\zeta=1.1,\ \gamma=-0.1.$$

$V$ 与权重同形；每个 $h(V)$ 是“从 floor 向 ceil 移动多少”的连续量。初始化令软量化权重接近原始权重，再最小化

$$\mathcal L=\|f_a(Wx)-f_a(\widetilde W\hat x)\|_F^2+
\lambda\sum_{i,j}\left(1-|2h(V_{ij})-1|^{\beta}\right).$$

这里 $x$ 来自全精度前缀，$\hat x$ 来自已量化的前缀，$f_a$ 是该层后的激活函数。正则促使 $h$ 最终接近 0 或 1，$\beta$ 按训练过程退火。以上分别对应式 (22)–(25)。这种“非对称重构”指教师与学生输入不同，不是指 zero-point 非零的非对称量化。

处理一个网络时，先折叠可折叠的 BatchNorm、选择网格；再逐层缓存教师输出和学生输入，优化 $V$，硬化为上下舍入选择，最后传给下一层。卷积场景将矩阵乘法换为卷积。最终不需要保留 $V$ 或 sigmoid 分支，推理权重仍在原量化网格上。

## 固定什么，学习什么

原方法固定原浮点权重、量化 scale 和整数范围，只优化舍入变量。论文 §5 明确指出更改步长会改变 QUBO 问题，因此没有联合优化 scale 与 AdaRound。这个约束对算法组合非常重要：先改变 clipping 或通道 scale，必须重新确定网格并重新做舍入优化。

论文主要实验采用逐层对称 W4、FP32 激活；说明方法也可用于逐通道，但不能据此声称论文已验证所有 group-wise LLM 配置。KV cache 不在该方法的原始实验对象中。

## 校准与成本

论文 §5 默认 1024 张无标签 ImageNet 训练图像，batch size 32、Adam、每层 10k 次迭代，报告 ResNet18 在单张 GTX 1080 Ti 上约 10 分钟。该耗时对应当时视觉模型与实现，不可外推到十亿参数 LLM。缓存教师激活、逐权重松弛参数与优化器状态都会增加离线内存；没有任务标签不等于没有数据需求。

## 实验和可迁移范围

论文表 5 在 ResNet18 / ImageNet / W4A32 下报告：最近舍入 23.99%，STE 66.63±0.06%，AdaRound 68.60±0.09% top-1；该实验 FP32 基线为 69.68%。这些数值属于特定模型权重、逐层量化和网格选择，不能与使用不同预训练权重的 BRECQ 表格直接做跨论文排名。

表 6 比较 MinMax、weight MSE、输出 MSE 三种网格；网格选择显著影响最终结果，说明“只换舍入就能修复所有误差”并不成立。原始实验主体为传统视觉网络与任务；LLM、MoE、DiT、文本 dLLM 需要新的块定义、数据覆盖和生成质量证据。

## 组合与失效条件

- **范围校准 → AdaRound**是自然顺序，原文提供多个固定网格实验。后续再次 clipping 使已有舍入选择不再对应原优化问题。
- **AdaRound 与 BRECQ**并非两个完全独立的流水线开关：BRECQ 本身将自适应舍入放到块重构中，重复跑层级和块级优化需明确是否重置变量及额外收益。
- **缩放/旋转 → AdaRound**只有固定变换后重建网格、输入缓存和教师输出才具有清楚语义；此为组合条件分析，不是原论文已验证的所有组合。
- 如果浮点与量化分支的非线性、残差或 BN 状态不一致，局部重构会优化错误目标。应先验证无量化时两分支一致。

## 官方实现与版本

论文作者机构的 [AIMET `ec4bc578681beaa52043bc83a228d5e5a855dd15`](https://github.com/quic/aimet/tree/ec4bc578681beaa52043bc83a228d5e5a855dd15) 提供维护中的实现。已读 `TrainingExtensions/torch/src/python/aimet_torch/_base/adaround/adaround_optimizer.py`：`AdaroundOptimizer.adaround_module` 比较软/硬舍入误差，调用 `_optimize_rounding`，最后设置 `use_soft_rounding=False`。这是当前维护实现，不是论文实验的原始 commit；具体默认迭代数和 wrapper 支持应以所用 AIMET 版本为准。

## Agent 最小验证

先在相同网格上比较 RTN 与 AdaRound，再单独测试网格选择；同时报告校准集、留出集重构误差和任务指标。区分 soft 输出与最终 hard 输出，检查收益是否在硬化后保留。迁移到生成模型时不能只以单层 MSE 验收，需要保持解码配置一致的任务评测。

本项目尚未执行本算法实验。
