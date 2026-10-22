---
{"id":"algorithm-gptq","title":"GPTQ：基于输入二阶统计的权重重构","type":"algorithm","tags":["weight-reconstruction","hessian","weight-only","calibration"],"aliases":["GPTQ","二阶量化","权重误差补偿"],"related":["algorithm-quarot","algorithm-smoothquant"],"sources":["paper-gptq","repo-gptq"],"updated":"2026-10-23"}
---

# GPTQ

## 问题与核心机制

直接把每个权重舍入到最近格点，只优化权重本身的距离，没有考虑输入通道之间的相关性。GPTQ 在小型校准集上逐层重构输出：固定一个权重列的量化值后，调整尚未量化的列来补偿输出误差。它优化的是量化权重，不是对激活做平滑，也不是一种 checkpoint 格式。

本页采用原论文的列样本约定：`W ∈ R^(d_out×d_in)`，`X ∈ R^(d_in×n)`，输出为 `WX`。目标是

$$\min_{\widehat W}\|WX-\widehat W X\|_F^2.$$

对每个输出行，该局部平方误差的 Hessian 是 `H=2XXᵀ`。它是层重构目标的精确二阶信息，不能称为完整语言模型任务损失的精确 Hessian。[论文 §3，Eq.1–3](https://arxiv.org/html/2210.17323v2#S3)

## 计算流程

1. 使用代表性输入运行模型，收集当前 Linear 的输入，累积 Gram 矩阵。权重的量化网格由位宽、粒度、scale、zero-point 决定。
2. 对 `H` 加阻尼，计算逆矩阵的 Cholesky 上三角因子 `U`，满足 `UᵀU=(H+λI)⁻¹`。
3. 按共享列顺序处理所有输出行。对列 `j`，计算 `q_j=Q(w_j)`，归一化误差 `e_j=(w_j-q_j)/U_jj`，更新剩余列：`W[:,j:] -= e_j U[j,j:]`。
4. 为减少内存访问，在列块内部处理更新；块结束后把该块误差一次矩阵乘法传播到后续列。
5. 保存量化权重，并将本层结果用于后续层的处理流程。导出打包是单独步骤，需保留分组、scale、zero-point、列排列等元数据。

注意：论文 Algorithm 1 为方便把 Cholesky 因子继续记作 `H⁻¹`；上式改名 `U`，避免将其误当成原逆矩阵。`blocksize` 是算法批量更新块大小，`groupsize` 是共享量化参数的组大小，二者不是同一参数。[论文 §4、Algorithm 1](https://arxiv.org/html/2210.17323v2#S4)

## 数据、配置与成本

原论文 §5 使用 C4 中 128 个随机 2048-token 片段，在单张 A100-80GB 上逐 Transformer block 量化。默认网格是 uniform per-row asymmetric min-max（在本页 `[out,in]` 布局中，每个输出行一组参数）；主要研究 OPT/BLOOM 的 3/4-bit 权重量化，激活维持浮点。前一 block 量化后重新运行，为下一 block 提供实际量化路径的输入。2-bit/三值结果属于更激进的特定设置，不能据此承诺任意 W2 配置。局部 Gram 存储量随输入通道数平方增长；块更新改善实际效率，不消除矩阵分解成本。

当前官方 `fasterquant` 默认 `blocksize=128`、`percdamp=.01`、`groupsize=-1`，同时提供 `actorder` 和 `static_groups`。这些是所读源码快照的配置，不能将后续实现支持直接当成原论文所有实验设置。`actorder` 按 Hessian 对角线重排列；导出与运行端必须一致处理这一排列。

## 前提与组合

- 校准输入必须覆盖目标使用分布；局部输出 MSE 小不保证最终 logits 或生成质量最好。
- 在 SmoothQuant 缩放或 QuaRot 旋转后，输入坐标和权重均已改变。重新采集对应统计，或严格推导并验证统计变换，不能直接复用旧 `H`。
- QuaRot 论文明确以旋转后 GPTQ 为默认权重量化步骤，是有原文证据的组合。
- 同一权重接着运行另一个重构器可能覆盖 GPTQ 的结果。没有明确联合目标和实现证据时，不把串联作为常规 Recipe。
- 对 MoE 专家、不同生成阶段和新架构的适用性需另行检查采样覆盖与模块映射；矩阵形式适用不等于完整链路适配。

## 实现与部署

固定实现：[gptq.py，2d65066](https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/gptq.py)。`GPTQ.add_batch` 累积输入统计，`fasterquant` 执行阻尼、排列、Cholesky 和误差更新。函数末尾写回的是模型权重 tensor；仅调用该函数不能证明已有低比特存储或 kernel。

原论文附录 A.2.2 的速度实验针对 batch size 1、自回归生成和量化矩阵乘浮点向量路径，收益主要来自访存。论文明确不声称通过通用 FP16×INT4 算术支持获得乘法加速；不能把该结果推广到大 batch prefill 或任意现代后端。

## 原文结果与实验建议

论文 Table 11 的 OPT-175B / C4 perplexity：FP16 为 10.13，4-bit RTN 为 11.61，4-bit GPTQ 为 10.28。该表同时注明校准来自 C4 训练集，因此这项结果不是完全零样本；评测采用附录 A.2.1 的 2048-token 非重叠片段协议。这里只转述论文结果，本项目未复跑。

Agent 可先在固定网格、相同校准集和模块范围下比较 RTN 与 GPTQ，再分别改变 group size、阻尼或列顺序。观察层输出误差、logits 差异和完整任务质量。如果改进只来自更细粒度或不同后端，不能归因于二阶重构本身。

## 来源与边界

原理以论文 v2 §3–4 为准；实现以固定 commit 为准；实验与性能分别绑定 §5 和附录 A.2。未核验目标 checkpoint、导出器和部署 kernel，不作部署可用性承诺。
