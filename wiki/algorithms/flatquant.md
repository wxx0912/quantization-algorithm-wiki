---
{
  "id": "algorithm-flatquant",
  "title": "FlatQuant：可学习可逆变换与块重构",
  "type": "algorithm",
  "tags": [
    "ptq",
    "learned-transform",
    "reconstruction",
    "w4a4",
    "kv-cache"
  ],
  "aliases": [
    "FlatQuant",
    "flatness",
    "平坦化",
    "可学习变换"
  ],
  "related": [
    "algorithm-spinquant",
    "algorithm-omniquant",
    "algorithm-gptq"
  ],
  "sources": [
    "paper-flatquant",
    "repo-flatquant"
  ],
  "confidence": "source-reported",
  "evidence_scope": "source-reported",
  "updated": "2026-10-24"
}
---

# FlatQuant：可学习可逆变换与块重构

FlatQuant 通过学习可逆线性变换、通道缩放和裁剪范围，让权重与激活更易量化。它属于需要校准优化的 PTQ，原论文主要验证 LLaMA 系列的低比特 W/A/KV 量化。

## 核心机制

对线性层，浮点等价关系为 $XW^T=(XP)(P^{-1}W^T)$。FlatQuant 学习使变换后量化误差较小的可逆 $P$，不要求它正交；以 Kronecker 结构把大变换拆成两个小矩阵，控制存储和在线计算成本。

实际优化以整个 Transformer block 的浮点与量化输出差异为目标，联合学习变换、通道缩放和裁剪参数。这里的 SVD 是可逆矩阵的参数化，不是 SVDQuant 的低秩补偿分支。

## 基本流程

1. 在适用的 attention、FFN 和 KV 边界设置变换及量化器。
2. 用校准数据逐块优化变换、缩放和裁剪，保持原模型权重不作为全模型训练变量。
3. 固定变换，将可融合参数吸收到权重或相邻算子；按最终变换图完成 RTN 或 GPTQ 权重量化。
4. 推理时保留必要的在线激活变换；通过相应 kernel 执行变换、量化和低比特计算。

论文只声明 LayerNorm、RoPE、attention score 与量化前变换保留 FP16；据此，Q/K 的一般可逆变换需要配对补偿，归一化、RoPE、非线性与残差位置会限制融合，不能把任意矩阵直接穿过这些算子（除论文声明外的部分为本库工程推断）。

## 适用条件与边界

- 论文主设置为权重 per-channel symmetric、激活 per-token symmetric；KV cache 量化另用 group-wise asymmetric（group size 128、§4.1）；具体位宽、分组及预算见 Source。
- 适合愿意承担块级校准优化、并能部署在线变换的候选场景；只支持普通 packed weights 的后端未必能够直接承接。
- 原论文以 LLaMA-2/3 和 LLaMA-3.1-Instruct 为实验对象，不能直接证明 MoE、多模态或 dLLM 的效果。
- 论文的加速依赖特定 RTX 3090、Triton/CUTLASS/FlashInfer 实现及 batch；小 batch 解码并不一定更快。
- 官方代码区分 fake quant 与真实量化路径，vLLM 接入声明是 fake quant。仅有模型接入或 `F.linear` 结果不能证明真实 INT4 加速。

## Agent 如何使用

先判断任务是否同时涉及 W/A 误差，以及目标后端能否承接在线变换。需要公式、训练预算、精度或速度条件时读论文 Source；需要 SVD 参数化、块输入缓存和导出路径时读源码 Source。工程上修改变换或量化格式后需重新校准并验证，不能复用旧整数权重来推断质量。本项目未运行 FlatQuant 训练或 GPU 性能测试。

## 详细来源

- [论文：变换、训练与实验条件](../../sources/papers/paper-flatquant.md)
- [官方源码：参数化、缓存及部署路径](../../sources/repos/repo-flatquant.md)
