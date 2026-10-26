---
{
  "id": "algorithm-spinquant",
  "title": "SpinQuant：学习正交旋转以降低量化损失",
  "type": "algorithm",
  "tags": [
    "ptq",
    "rotation",
    "learned-transform",
    "w4a4",
    "kv-cache"
  ],
  "aliases": [
    "SpinQuant",
    "learned rotation",
    "学习旋转",
    "可学习变换"
  ],
  "related": [
    "algorithm-quarot",
    "algorithm-gptq",
    "algorithm-flatquant"
  ],
  "sources": [
    "paper-spinquant",
    "repo-spinquant"
  ],
  "confidence": "source-reported",
  "evidence_scope": "source-reported",
  "updated": "2026-10-24"
}
---

# SpinQuant：学习正交旋转以降低量化损失

SpinQuant 冻结预训练权重，利用校准数据学习正交旋转，降低低比特权重、激活和 KV 的量化损失。它优化的是旋转参数，不是全部模型权重；适合希望在随机旋转基础上进一步改善低比特精度、且能承担反向传播校准成本的场景。

## 核心机制

对线性层，把权重乘以正交旋转矩阵、同时以同一旋转作用到输入，未量化的前向输出不变；但量化结果会随旋转改变（两者的矩阵记法为本库归纳）。因此可以在保持未量化网络数学等价的前提下，寻找更容易量化的表示。

论文学习残差流旋转 $R_1$ 和各 attention 层的 head 维旋转 $R_2$，使用量化网络的任务损失与 Cayley 约束优化保持正交。RoPE 之后 Q/K 的 $R_3$ 和 FFN down-projection 输入的 $R_4$ 保留固定 Hadamard 在线变换。不是所有旋转都能离线吸收，也不能将旋转任意穿过非线性。

## 基本流程

1. 按模型结构融合 RMSNorm 缩放，并插入成对旋转，建立数学等价的浮点计算图。
2. 冻结原权重，以随机 Hadamard 初始化可学习旋转，使用校准任务损失优化 $R_1/R_2$。
3. 固定旋转，吸收可融合部分，再进行权重量化；在线 Hadamard 保留在部署图中。
4. 导出旋转后的量化权重和格式信息，检查目标后端对在线变换及 W/A/KV 格式的支持。

与 GPTQ 联用时，论文验证过“学习阶段权重保留 16 bit、量化激活及相应 KV，再使用 GPTQ 量化权重”的顺序；不能把它和学习阶段也使用 RTN 权重量化的设置混为一谈。

## 适用条件与结果范围

论文 v2 在 LLaMA-2/3 上评估，主要涉及 W4A4KV16、W4A4KV4，也报告 weight-only。其设置为权重 per-channel symmetric、激活 per-token asymmetric dynamic、KV asymmetric dynamic group 128。旋转学习采用 800 条 WikiText-2 样本与 100 次迭代；GPTQ 阶段另用 128 条长度 2048 的 WikiText-2 样本。这些是论文设置，不是所有实现的默认值。

结果支持在这些模型和设置中学习旋转能改善随机旋转的量化效果，并展示与 GPTQ 的兼容性；不代表每层误差都会降低，也不保证 MoE、DiT、文本 dLLM 或新后端具有同样收益。浮点等价是代数性质，不等于有限精度运行逐位一致。论文精度和训练成本也不能替代目标系统的推理性能测试。

## Agent 如何使用

先据此判断是否需要学习旋转，再按问题展开来源：

- 旋转位置、优化公式、校准预算、RTN/GPTQ 对比和实验数字：读[论文来源](../../sources/papers/paper-spinquant.md)。
- 训练入口、旋转产物、tied embedding 处理：读[官方源码来源](../../sources/repos/repo-spinquant.md)。
- 工程迁移时，先检查模型结构和浮点变换一致性；改变旋转后应重新采集统计或实现严格的统计变换，再量化。该建议属于工程推导。
- 是否能导出、实际调用何种 kernel、最终质量和速度：继续检查目标项目，不由本文承诺。
