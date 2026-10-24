---
{
  "id": "algorithm-brecq",
  "title": "BRECQ：块级重构与输出敏感度",
  "type": "algorithm",
  "tags": [
    "ptq",
    "block-reconstruction",
    "rounding",
    "fisher",
    "vision"
  ],
  "aliases": [
    "Block Reconstruction",
    "Pushing the Limit of Post-Training Quantization"
  ],
  "related": [
    "algorithm-adaround",
    "algorithm-autoround"
  ],
  "sources": [
    "paper-brecq",
    "repo-brecq"
  ],
  "confidence": "source-reported",
  "evidence_scope": "source-reported",
  "updated": "2026-10-23"
}
---

# BRECQ：块级重构与输出敏感度

BRECQ 在一组有结构关联的层上共同优化量化误差。原论文的主要对象是 ResNet、MobileNet、RegNet 等视觉网络，block 通常指残差块等 CNN 组件。其研究支持“重构粒度需要权衡依赖关系与小样本泛化”，不直接证明任意 Transformer block 是最优粒度。

## 为什么选择块

单层重构忽略相邻层量化误差之间的相互作用；整个网络联合重构又容易在少量校准图像上过拟合。[论文 v2 §3.2](https://arxiv.org/pdf/2102.05426v2)用块对角 Hessian 近似解释中间粒度：保留块内的跨层依赖，忽略块间依赖。作者明确说明理论没有给出普遍最优粒度，选择 block 来自其所测试模型的实验。

令 $z$ 是浮点块输出，$\hat z$ 是量化块输出，$\Delta z=z-\hat z$。二阶输出误差代理为

$$\mathbb E[\Delta z^T H_z\Delta z].$$

论文 §3.3 式 (9)–(10)用对角 Fisher 近似输出 Hessian：

$$\mathcal L_{rec}=\mathbb E\sum_i g_i^2(z_i-\hat z_i)^2,\quad
g_i={\partial\mathcal L_{task}\over\partial z_i}.$$

它给对任务敏感的输出元素更大权重。Fisher 与任务损失 Hessian 的等价需要分布与模型假设；不能把“梯度平方”称作精确 Hessian，也不能把直接 MSE 实现称为已经使用 Fisher。取单位权重则退化为普通输出 MSE。

## 从校准数据到输出

1. 准备浮点教师和量化网络，确定 block 边界、weight/activation bit 与粒度。
2. 缓存当前块输入、浮点输出；使用 Fisher 目标时另外收集梯度相关信息。
3. 同时优化块内各量化层的权重舍入变量。论文采用 AdaRound 式连续松弛与二值化正则，最后硬化舍入。
4. 若量化激活，学习激活量化 step size，采用 learned step-size 机制；不把这个步骤混同于继续更新浮点模型权重。
5. 逐块推进，必要时采集敏感度供混合精度配置使用。

论文算法 1 给出整体顺序，附录 B.4.1 解释舍入与激活 step size。原始浮点教师保持固定；导出的舍入权重和量化步长取代训练时松弛变量。重构优化不直接提供 packed checkpoint 或部署 kernel。

## 实现不等于论文的所有选项

已读 [官方仓库 `e455d62e93c70351961f8991c913b59435bd165f`](https://github.com/yhhhli/BRECQ/tree/e455d62e93c70351961f8991c913b59435bd165f) 的 README 和 `quant/block_recon.py`。

`block_reconstruction` 默认 `batch_size=32`、`iters=20000`、`opt_mode='mse'`；权重阶段把 quantizer 替换成 `AdaRoundQuantizer` 并优化其 `alpha`。激活阶段优化 `delta`。`LossFunction` 支持 `mse`、`fisher_diag`、`fisher_full`；只有非 MSE 模式才走 `save_grad_data`。因此复现记录必须写实际 `opt_mode`，不能因为引用 BRECQ 就声称运行使用了式 (10)。

`fisher_full` 是源码模式名称，不代表精确完整 Fisher 二次型：代码令 $a=|\hat z-z|\odot|g|$，使用按样本归约的 `((sum(a) * a).mean()) / 100` 代理。它与论文式 (10) 的对角 Fisher 不同，不能仅凭模式名声称复现完整 FIM。

`asym` 控制是否使用量化前缀输入重构浮点输出；函数默认与实际启动参数也要区分。README 展示 `--channel_wise`、`--n_bits_w 2 --n_bits_a 4 --act_quant` 的 ImageNet 示例，不能把示例覆盖范围推广为所有网络结构。

## 数据、成本与结果

论文使用 1024 张训练样本校准，§3.3 报告单张 GTX 1080 Ti 上约 20 分钟得到量化 ResNet18。成本包含块输入输出缓存、舍入参数和反向传播；Fisher 相关模式还需要梯度缓存。预算不能仅按模型总参数数目估算，也依赖块大小与特征图尺寸。

论文表 2 的 ImageNet ResNet18：FP32 top-1 为 71.08%，BRECQ W4A32 为 70.70±0.07%，W2A32 为 66.30±0.12%。表 3 同模型 W4A4 为 69.60±0.04%，W2A4 为 64.80±0.08%。这些是不同量化对象的结果，不能合并成“INT2 几乎无损”。原文全精度 baseline 与 AdaRound 原论文不同，不能直接用跨论文百分比计算算法优势。

论文还探索混合精度与目标检测，但首版这里只将其作为来源范围说明，没有把它们转换成已验证的 LLM recipe。A4 的视觉网络实验不涉及 KV cache；也不能证明目标 GPU 上的真实 INT4 GEMM 与端到端加速。

## 组合与迁移约束

- **AdaRound + BRECQ**：BRECQ 的权重优化已采用自适应舍入机制。应表达为“块级目标下的舍入优化”，而非两个独立算法的机械串联。
- **缩放/旋转前处理**：先固定浮点等价图，再重采输入、教师输出及 Fisher 相关梯度；变换改变坐标系后旧梯度权重不能直接复用。能组合是条件分析，不是本页原文的全面实验结论。
- **混合精度搜索**：更改位宽或 group 后会改变量化网格和重构难度，旧舍入解不能被视为新配置最优。候选敏感度只在原数据和目标定义下有意义。
- **LLM/dLLM 迁移**：需要明确 block 边界、attention mask、残差路径、teacher 输入、任务损失或无标签代理、序列缓存预算。文本去噪阶段的数据覆盖不能由 CNN 图像校准结果替代。

## Agent 最小验证

固定模型、格式、校准集和总优化预算，比较 layer/block/stage 粒度；记录校准与留出数据误差，识别大粒度过拟合。独立比较 MSE 与 Fisher 目标，注明梯度来源和额外成本。对 W/A 分别消融，避免把激活 step-size 学习收益归到权重舍入上。

本项目没有执行 BRECQ 实验；页面结论来自上述论文与固定源码。
