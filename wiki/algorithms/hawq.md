---
{"id":"algorithm-hawq","title":"HAWQ：用 Hessian 敏感度选择混合精度","type":"algorithm","tags":["mixed-precision","hessian","sensitivity","qat"],"aliases":["HAWQ-I","HAWQ-II","HAWQ-V2","Hessian AWare Quantization"],"related":["algorithm-modelopt-autoquantize","guide-architecture-transfer"],"sources":["paper-hawq2","source-hawq2","code-hawq"],"updated":"2026-10-23"}
---

# HAWQ：用 Hessian 敏感度选择混合精度

HAWQ 先估计各层受到量化扰动时的损失敏感度，再分配不同位宽。它不是一种新的舍入器。本文以 NeurIPS 2020 的 **HAWQ-V2** 为代表；该版本也常被称为 HAWQ-II。HAWQ-I 用最大 Hessian 特征值排序，V2 用平均 Hessian trace，并加入 Pareto 精度选择和混合精度激活分析。[会议论文 §1–2](https://proceedings.neurips.cc/paper/2020/file/d77c703536718b95308130ff2e5cf9ee-Paper.pdf)

**分类边界：原论文的结果包含 quantization-aware fine-tuning。** 可以借用它的敏感度思想为 PTQ 提供候选配置，但不能把论文准确率标成无训练 PTQ 的结果。本项目仅核实文献和源码，没有运行 HAWQ。

## 从输入到输出

输入是已训练模型、带标签的代表数据、可求二阶导的任务损失、候选位宽及模型大小预算。

1. 按层或块划分参数 $W_i$，固定浮点模型与损失定义。
2. 用 Hessian-vector product 与随机向量估计每块 Hessian 的 trace，不显式构造大矩阵。
3. 除以该块参数维度，得到平均 trace；根据敏感度次序缩小候选位宽分配空间。
4. 对每个候选位宽实际量化 $W_i$，计算权重扰动平方范数，不能只看位宽标签。
5. 对每种整网配置累加 trace 加权扰动，在模型大小与扰动的 Pareto 前沿上选择目标配置。
6. 按选定的权重/激活位宽执行量化感知微调，并做任务评估。最终产物是位宽分配、量化模型及评估结果；敏感度分数本身不是准确率。

激活混合精度还需对激活变量计算二阶信息。论文利用不同样本之间的块对角结构，逐样本估计并平均，避免把所有样本激活拼成一个巨大 Hessian。[§2.2–2.3]

## 数学机制与变量

令任务损失为 $L$，块维度为 $n_i$，定义：

$$
H_i=\nabla^2_{W_i}L,\qquad
\bar h_i=\frac{\operatorname{Tr}(H_i)}{n_i},\qquad
\delta_i(b)=Q_b(W_i)-W_i.
$$

近似驻点时，一阶项较小，有

$$
\Delta L_i\approx\tfrac12\delta_i^T H_i\delta_i.
$$

若扰动在 Hessian 各特征方向具有相同二阶矩，方向平均后得到与 $\bar h_i\|\delta_i\|_2^2$ 成比例的敏感度。V2 按

$$
\Omega(\mathbf b)=\sum_i\bar h_i\|Q_{b_i}(W_i)-W_i\|_2^2
$$

排序整网候选。论文 Eq. (10) 的 trace 符号表示**平均** trace，不能漏掉维度归一化。否则大层仅因参数多就得到更高敏感度。[§2.1 Assumption 1、Eq. (4)–(5)、§2.3 Eq. (10)]

Hutchinson 估计使用满足 $\mathbb E[zz^T]=I$ 的随机向量：

$$
\operatorname{Tr}(H_i)\approx\frac1m\sum_{k=1}^m z_k^TH_i z_k.
$$

V2 先用相对敏感度约束候选，再在剩余配置里按 $\Omega$ 与大小选择；它不等于对所有层任意位宽组合求出真实任务损失的全局最优解。搜索变量是每层位宽；后续微调还更新模型参数。不存在这里统一定义的“只优化 scale 的 HAWQ”。

## 假设、数据与成本

- 理论条件包括局部极小点、零梯度、半正定 Hessian、扰动方向假设及可忽略高阶项。远离驻点、强非线性或跨层耦合明显时，分数只作代理。[§2.1]
- 数据需要反映实际任务且能计算损失；无标签 LLM 校准可以构造自监督损失，但那是迁移方案，不能直接继承 CNN 原论文的验证结论。
- 会议版 §3.1/Fig. 5 报告：ResNet50 的估计在约 512 个数据点后趋于稳定，并使用 50 次 Hutchinson 迭代；54 个块在 4 张 GPU 上少于 30 分钟。另一个版本——arXiv v1 的 `s6_appendix.tex`——对 RetinaNet 激活 trace 图注明使用 128 个数据点和 50 次迭代，不能把前者当成统一 calibration 默认。这是统计估计成本，**不是完整 QAT 时间，也不是 LLM 成本承诺**。
- 相同模型大小不意味着相同延迟。多位宽切换、实际 kernel、激活存储和未量化层均需单独核算。

## 原文格式与有条件结果

论文研究 CNN 的均匀低位宽权重/激活量化，并未验证 LLM 的 KV-cache 量化。预印本附录 A 给出先 clip 到 $[q_0,q_{2^k-1}]$、再以 $\Delta=(q_{2^k-1}-q_0)/(2^k-1)$ 舍入到整数索引、最后反量化的量化器；范围允许取原 min/max 的子区间以裁剪 outlier。附录明确 backward 使用 STE。不能据此把所有实验解释成固定的现代 group-128 量化，具体权重/激活范围与粒度需随模型配置核对。

以下结果统一采用 **NeurIPS 2020 会议版**（`paper-hawq2`），已核对 PDF 第 8 页 Table 1 和第 9 页 Table 3(b)。`MP` 标记报告混合配置的**最低位宽**，不是平均位宽或全网固定格式。[Table 1 表注]

| 原文条件 | 浮点基线 | HAWQ-V2 | 能支持的结论 |
|---|---:|---:|---|
| ResNet50，ImageNet，Table 1(a) | Top-1 77.39%，97.8 MB | 最低 W2/A4 MP，75.92%，7.99 MB | 经论文流程得到混合精度模型；不能称为全网 W2A4 PTQ |
| InceptionV3，ImageNet，Table 1(b) | 77.45%，91.2 MB | 最低 W2/A4 MP，75.98%，7.57 MB | 相同架构和任务下的大小/质量点 |
| SqueezeNext，ImageNet，Table 3(b) | Top-1 69.38%，10.1 MB | W3-MP/A8，trace 敏感度 68.68%，1.07 MB；L2 敏感度 67.83%，1.10 MB | 同一会议版的敏感度消融支持加入曲率；大小并非完全相等，不能只归因于单一变量 |

**预印本版本差异：** arXiv `1911.03852v1` 的 `s3_results.tex` 分别报告 ResNet50 75.76%、InceptionV3 75.68%、SqueezeNext 68.38%。这些数字与上表会议版不同，且表格编排也不同。本页引用 v1 LaTeX 的量化器与 STE 描述，不将其结果替换到会议版 Table 1/3 的位置。会议版 Table 1(c) 也独立报告 SqueezeNext 68.68%，与 Table 3(b) 一致。

## 实现与版本边界

官方 [Zhen-Dong/HAWQ](https://github.com/Zhen-Dong/HAWQ/tree/1616df69fdd99f100a8d8a6e78742e0a89292634) 的锁定版本为 `1616df69fdd99f100a8d8a6e78742e0a89292634`。已读 README 与 [`quant_train.py`](https://github.com/Zhen-Dong/HAWQ/blob/1616df69fdd99f100a8d8a6e78742e0a89292634/quant_train.py)：`train`/`train_kd` 包含 backward 与 `optimizer.step()`，印证训练边界。

该仓库 README 当前主展示 **HAWQ-V3** 的模型与 TVM 推理结果；不能用这些表格代替 V2 论文实验，也不能声称当前默认训练命令精确复现 V2。README 说明所发 checkpoint 以浮点保存，真正缩小文件还涉及权重整数打包；论文大小数字、checkpoint 文件大小与运行显存应分开。V2 每项配置的精确端到端复现尚未在本 Wiki 执行。

## 组合顺序与失效条件

在新的缩放、旋转、剪裁或重构之后，量化扰动与参数坐标系均可能变化。应在最终候选变换的模型状态上重新估计敏感度和扰动，不直接复用变换前的 trace 排序。此规则来自公式依赖关系，属于工程推导；不是论文已经验证所有变换组合。

把 HAWQ 作为 PTQ 搜索启发时，先确定可部署的位宽候选，再估计分数、选择配置，最后用实际 PTQ 生成模型并评估。若省略原方法的微调，应明确记录为衍生方案。对融合 QKV/共享专家等执行单元还要施加实际后端的同格式约束。

## Agent 使用与消融

适合的问题是“统一位宽超出质量容忍度，在同一大小预算下哪些层应保留精度”。至少比较统一位宽、按误差选位宽、按平均 trace 加权误差选位宽；固定候选格式、大小预算、校准集、任务指标，并分别报告有无微调。

需要额外检查 trace 估计对样本与随机向量的稳定性、代理分数与真实单块损失变化的相关性，以及最终全模型质量。如果代理排名不稳定或后端不支持选出的混精模式，停止将分数直接转成部署 Recipe。

来源登记见 [混合精度与架构证据](../../sources/mixed-architecture.md)。与 [ModelOpt AutoQuantize](modelopt-autoquantize.md) 对比时，应区别 Hessian trace 选位宽和输出扰动/Fisher 代理搜索。
