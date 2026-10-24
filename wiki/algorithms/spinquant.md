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
    "学习旋转"
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
  "updated": "2026-10-23"
}
---

# SpinQuant：学习正交旋转以降低量化损失

SpinQuant 固定预训练权重，通过校准数据学习正交旋转，使**浮点等价的计算图在加入量化后损失更小**。它优化旋转，不是对全部模型权重做 QAT；它也不等于任意可逆变换。随机旋转能分散异常值，却不保证不同随机种子的量化精度一致，这是论文引入学习的直接动机。[论文 §2–3](https://arxiv.org/html/2405.16406v2#S3)

## 从输入到输出

采用行向量记法，线性层为 $Y=XW^T$。正交矩阵 $R$ 满足 $RR^T=I$，因此 $XW^T=(XR)(WR)^T$；实际量化计算为 $Q_a(XR)Q_w(WR)^T$，量化误差会随 $R$ 改变。

1. 将 RMSNorm 的逐通道缩放融合到后继权重，建立残差流可旋转的图。不是把旋转任意穿过非线性函数。
2. 引入共享的残差旋转 $R_1\in\mathbb R^{d\times d}$，吸收到 embedding、读写残差流的线性权重与输出头的相应位置。
3. 每层学习 head 维度的 $R_2\in\mathbb R^{d_h\times d_h}$，在 value 与 attention 输出投影之间成对旋转。注意 softmax attention 对 value 的线性作用允许此等价变换；论文不是为每个 head 独立学习一个任意大矩阵。
4. 对 RoPE **之后**的 Q/K 施加固定 Hadamard $R_3$，利用 $(QR_3)(KR_3)^T=QK^T$；对 FFN down-projection 输入施加固定 Hadamard $R_4$。这两处保留在线变换，不能声称全部旋转都可离线融合。
5. 冻结原权重，经过带 fake quant 的网络计算校准任务损失，更新 $R_1,R_2$。最终固定旋转，再做权重量化与评测。

上述位置见论文 Figure 1、5、6 和 §3.1。共享权重、非 RMSNorm、不同 head 布局需要重新检查等价性；官方当前实现会解除 `tie_word_embeddings` 并复制输出头，不能忽略其存储影响。

## 目标函数与优化

论文 Eq.1 优化 $\min_{R_1,R_2\in\mathcal M}\mathcal L_Q(R_1,R_2\mid W,X)$，$\mathcal M$ 为正交矩阵集合，$W$ 固定，$X$ 为校准样本，$\mathcal L_Q$ 是含量化的任务损失，例如 next-token cross entropy。旋转采用随机 Hadamard 初始化。

Eq.2 的 Cayley 更新为

$$R'=(I-\tfrac\alpha2Y)^{-1}(I+\tfrac\alpha2Y)R,\qquad Y^T=-Y.$$

$Y$ 由梯度投影构造，$\alpha$ 是步长；反对称性保证更新保留正交约束。实现用固定点迭代计算 Cayley 更新。不要将普通 SGD 后直接得到的任意矩阵代入 $R^{-1}=R^T$。

与 GPTQ 联用时论文 §4.3.1 有重要细节：学习旋转阶段让权重保持 16 bit、只量化激活和所需 KV，随后用 GPTQ 处理权重误差；在该实验中比同时针对 RTN 权重误差学习更好。这个顺序是已有实验，不是自由组合的推断。

## 校准、格式与预算

| 项目 | 论文 v2 的设置 |
| --- | --- |
| 旋转优化 | WikiText-2 的 800 样本、100 iterations；Cayley SGD，学习率 1.5 线性降至 0 |
| 原权重 | 固定；优化变量为共享 $R_1$ 和各层 $R_2$ |
| 权重量化 | per-channel symmetric；用权重 MSE 搜索范围；RTN 或 GPTQ |
| 激活 | per-token asymmetric min-max dynamic quantization |
| KV | asymmetric dynamic，group size 128 |
| GPTQ 校准 | 另用 WikiText-2 128 条、长度 2048 |
| 预算 | §4.1 报告 8×A100 上 LLaMA-2 7B 约 1.25 h、LLaMA-3 8B 约 1.39 h、70B 约 12 h；不能当成本机预估 |

主表覆盖 W4A4KV16 与 W4A4KV4 等配置；不是所有层、算子或参数都变成 4 bit。具体实验须同时保存旋转学习配置与最终 GPTQ/RTN 配置。

## 原文结果及解释边界

论文 Table 2 在 LLaMA-2 7B、最终 W4A4KV4 下比较两种旋转学习设置：在 W4A4KV4 上学习得到 8 项 zero-shot 平均准确率 $60.9\pm0.6$、WikiText-2 PPL $6.8\pm0.15$；在 W16A4KV4 上学习再 GPTQ 得到 $64.0\pm0.3$、PPL $5.9\pm0.01$。这支持“让旋转负责激活误差、GPTQ 负责权重误差”的具体流程，不证明所有模型都能提升同样幅度。

Table 1 覆盖 LLaMA-2 7B/13B/70B 与 LLaMA-3 8B/70B；不能外推为 MoE、dLLM 或 DiT 的已验证结果。论文精度证据不自动证明任意后端存在相应融合 kernel。

## 官方实现与产物

固定源码版本：`8f47aa3f00e8662caf1a484153920a07e5281c3a`。

- [`optimize_rotation.py: train`](https://github.com/facebookresearch/SpinQuant/blob/8f47aa3f00e8662caf1a484153920a07e5281c3a/optimize_rotation.py#L42)：冻结原参数，构造 $R_1/R_2$，使用 WikiText 数据、Trainer 和 SGDG，输出旋转字典。
- [`train_utils/optimizer.py: Cayley_loop / SGDG`](https://github.com/facebookresearch/SpinQuant/blob/8f47aa3f00e8662caf1a484153920a07e5281c3a/train_utils/optimizer.py#L33)：约束优化；不要将类名误解为普通 SGD。
- [`ptq.py: train`](https://github.com/facebookresearch/SpinQuant/blob/8f47aa3f00e8662caf1a484153920a07e5281c3a/ptq.py#L23)：调用 PTQ 流程和 PPL 评测。文件名、函数名不代表其重新训练全部权重。

以上是源码静态核验；本 Wiki 未运行校准、导出或 kernel benchmark。当前源码晚于论文 v2，复现论文须明确使用的脚本参数和模型版本，不能假定当前默认值就是原文配置。

## 组合顺序与失效条件

已有依据的流程：结构等价改写 → 学习旋转 → 固定旋转 → 在旋转图上采集 GPTQ 输入统计 → GPTQ → 导出与验证。若改变旋转、残差融合或 head 布局，原来的激活范围、GPTQ Hessian、权重 scale 与量化权重均不可原样复用。部分统计可严格变换，例如正交基变换后的二阶矩，但必须实现并验证正确变换；默认重新采集更易审计。

与 FlatQuant 不能简单叠加两个“变换开关”：二者改变同一输入基底，训练目标、在线图和 kernel 均需重新定义。与 dLLM 的数学线性层等价性可能仍成立，但 mask 阶段、去噪步和非因果上下文造成的激活分布及任务损失差异没有由 LLaMA 实验解决。

## Agent 使用与消融

当 W4A4 的主要问题来自激活异常值、随机旋转种子波动且预算允许任务损失反传时，SpinQuant 是候选。先验证浮点旋转前后 logits 一致，再讨论量化收益。

有区分力的实验：固定样本、量化范围、最终 GPTQ 配置和评测集，比较无旋转、随机旋转多种子、学习旋转多种子；另比较学习时 W4 与 W16、最终都使用同样 GPTQ。记录正交误差、量化前后层输出误差、最终 PPL/任务准确率，以及旋转训练成本。若只比较不同后端、不同 KV 位宽的最终分数，无法归因于学习旋转。

来源登记见 [transforms](../../sources/topics/transforms.md)。
