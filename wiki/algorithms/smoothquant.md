---
{
  "id": "algorithm-smoothquant",
  "title": "SmoothQuant：在权重与激活之间迁移量化难度",
  "type": "algorithm",
  "tags": [
    "equivalent-transformation",
    "scaling",
    "activation",
    "calibration"
  ],
  "aliases": [
    "SmoothQuant",
    "平滑量化",
    "激活离群值"
  ],
  "related": [
    "algorithm-gptq",
    "algorithm-omniquant"
  ],
  "sources": [
    "paper-smoothquant",
    "repo-smoothquant"
  ],
  "confidence": "source-reported",
  "evidence_scope": "source-reported",
  "updated": "2026-10-23"
}
---

# SmoothQuant

## 问题与计算流程

当激活少数通道存在持续的大值时，per-token 或 per-tensor scale 会被这些通道主导，使其余值的有效分辨率下降。SmoothQuant 对输入通道做离线缩放，将部分动态范围迁移到权重，而不是删除异常值或保留一组浮点通道。

采用行样本约定：`X ∈ R^(n×d_in)`，`W ∈ R^(d_in×d_out)`，`Y=XW`。对正对角矩阵 `D=diag(s)`：

$$XW=(XD^{-1})(DW),\qquad s_j=\frac{\max|X_j|^\alpha}{\max|W_j|^{1-\alpha}}.$$

`max|X_j|` 来自校准样本第 `j` 个输入通道；`max|W_j|` 在对应权重输入通道上取最大值。`α` 控制迁移程度，增大它通常把更多激活动态范围转移给权重。公式是浮点等价变换；再分别量化 `XD⁻¹` 和 `DW` 才产生近似。[论文 §4 Eq.3–4](https://arxiv.org/html/2211.10438v7#S4)

实际步骤：采集激活范围 → 根据权重范围求 `s` → 将逆缩放吸收到上游归一化或兼容线性模块 → 将正缩放吸收到所有对应下游投影 → 对变换后的模型校准量化器 → 执行 W/A 量化。

## 吸收条件与修改对象

RMSNorm 的逐通道增益可除以 `s`；LayerNorm 的增益和偏置都需相应调整。下游所有共享该输入的 Linear 必须乘同一 `s`。PyTorch Linear 权重存储为 `[out,in]`，因此源码沿列乘 scale，与上面数学布局相反但运算一致。

固定源码 `smooth_ln_fcs` 对多个下游投影的权重取联合通道范围。`smooth_lm` 的 Llama 路径对 Q/K/V 共同处理，对 gate/up 共同处理；Mixtral 路径把 router 与专家的输入投影纳入同组。它们证明了当前实现的特定映射，不证明任意 MoE 的路由和共享路径都可直接套用。

残差分叉、非线性以及未被同步变换的消费者可能破坏等价性。不能声称所有缩放在任何模型上都能无开销融合。需要先核对计算图和所有消费者，再比较浮点变换前后的输出。

## 校准、格式与预算

原论文 §5.1 使用 Pile 中 512 个随机句子采集平滑因子和静态量化步长。OPT/BLOOM 常用 `α=.5`，GLM-130B 使用 `.75`；这些是特定模型的论文设置，不是普适最优值。

主要目标是 INT8 W8A8。Table 2 的设置如下：

| 方案 | 权重 | 激活 |
| --- | --- | --- |
| O1 | per-tensor | per-token dynamic |
| O2 | per-tensor | per-tensor dynamic |
| O3 | per-tensor | per-tensor static |

v7 Table 7 后续 Llama-2/Mixtral 实验改为 per-channel 权重、per-token 激活，不能与上表混为一套配置。平滑因子离线固定不等于激活量化器必定静态。激活量化范围应在变换后的分布上确定。平滑本身不需要梯度训练，成本主要是校准前向和参数变换。

论文 Figure 6 对 Linear 与 attention BMM 使用 INT8，而 LayerNorm、Softmax 等轻量算子保留浮点。W8A8 不表示整张计算图每个 tensor 都是 INT8，也不自动规定 KV cache 的持久存储格式。

## 方法关系与失效规则

SmoothQuant 解决 W/A 动态范围平衡；GPTQ 解决给定输入统计下的权重重构。二者数学上可能组合，但要定义先变换、后采集统计和重构的具体路径，并验证导出格式。不能仅凭机制互补就视为现成部署方案。

OmniQuant 使用类似缩放作为初始化，再优化可学习变换和 clipping，最终参数不等于 SmoothQuant 固定公式。调整 `α`、校准数据或共享模块分组后，受影响的激活范围、权重 scale、重构统计和下游质量结果需要重新确认。

## 实现与证据

[smoothquant/smooth.py，c61476d](https://github.com/mit-han-lab/smoothquant/blob/c61476d728e42ae0d8a35e7e78494edcac3237b5/smoothquant/smooth.py) 的 `smooth_ln_fcs`、`smooth_ln_fcs_llama_like` 和 `smooth_lm` 已核读。这个文件执行平滑，不是完整 INT8 kernel 实现。

论文 §5.1–5.3 的部署证据来自 Hugging Face/PyTorch 和 FasterTransformer 集成，配套 CUTLASS INT8 GEMM/BMM。目标后端若只有 fake quant，不能复用论文速度结论。

论文 §5.2、Table 4：BLOOM-176B 上 O1/O2 保持接近浮点质量，静态 per-tensor 的 O3 平均准确率下降约 0.8 个百分点，作者归因于校准统计与实际输入范围差异。这体现的是给定评测设置下的观察，不意味着所有模型应固定选择 O1。

## Agent 实验建议与边界

在固定数据、格式和粒度下，对比不平滑、多个 `α`、变换前后仅浮点输出。再比较权重与激活的误差是否互相迁移。若浮点等价检查失败，先排查融合位置，不能通过扩大 clipping 搜索掩盖计算图错误。

本项目核对原文和缩放源码，未运行量化或性能实验。4-bit、dLLM mask 轨迹或视频时间步上的适用性需独立验证。
