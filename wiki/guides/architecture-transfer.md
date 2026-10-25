---
{
  "id": "guide-architecture-transfer",
  "title": "从 AR 到 MoE、dLLM、DiT 与 VLM：量化迁移需要的证据",
  "type": "guide",
  "tags": [
    "architecture",
    "ar",
    "moe",
    "dllm",
    "dit",
    "vlm",
    "calibration",
    "transfer",
    "denoising",
    "mask"
  ],
  "aliases": [
    "ViDiT-Q",
    "ViDiT-Q-MP",
    "架构适配",
    "生成过程量化"
  ],
  "related": [
    "algorithm-qdit",
    "algorithm-dllmquant",
    "algorithm-hawq",
    "algorithm-modelopt-autoquantize",
    "algorithm-awq",
    "algorithm-smoothquant"
  ],
  "sources": [
    "paper-awq-architecture",
    "paper-llada-architecture",
    "paper-viditq",
    "source-viditq",
    "code-viditq",
    "code-modelopt-autoquantize"
  ],
  "confidence": "source-reported",
  "evidence_scope": "mixed",
  "updated": "2026-10-24"
}
---

# 从 AR 到 MoE、dLLM、DiT 与 VLM：量化迁移需要的证据

架构相似只能提供候选线索。量化方案能否迁移，还取决于实际执行图、激活分布、生成过程、任务指标与后端。本文把直接实验、源码支持和迁移假设分开；没有进行新的模型实验。

## 先按证据范围判断

| 目标 | 已读直接证据 | 不能外推的部分 | 迁移前要补什么 |
|---|---|---|---|
| AR 文本 Transformer | AWQ v5 Table 4 的 LLaMA/Llama-2，weight-only INT3/INT4、group 128 | 不能证明 W/A/KV 同时低比特、所有层均低比特或所有上下文长度有效 | 量化对象、校准语料、序列长度、prefill/decode 与目标评测 |
| MoE | AWQ v5 Table 5 的 Mixtral-8x7B-Instruct-v0.1；ModelOpt 当前源码有专家分组 | 某个 Mixtral 结果不覆盖任意 router、shared expert、fused expert 布局 | 校准 token 的专家覆盖、路由变化、专家实现与融合约束，分别核算总权重/活跃权重 |
| 文本 dLLM | LLaDA v2 §2 直接证明随机 mask 训练、双向依赖和迭代去 mask 生成；它不是量化论文 | 不支持“AR 量化已在 LLaDA 有效”，也不能拿图像 DiT 结果代替文本证据 | 真实 mask 比例与轨迹、remasking、块长/生成步数、评分目标、缓存语义与逐样本生成 |
| 图像/视频 DiT | ViDiT-Q v2 对 STDiT/Latte/PixArt 的直接量化研究，见下文 | 视频感知指标不等价于文本正确率；该版 W4A8 延迟不是真实 W4A8 kernel 实测 | 目标 scheduler、CFG、分辨率/帧数、时间步分布、质量指标与实际 kernel |
| VLM | AWQ v5 §5/Table 6–7：OpenFlamingo 与 VILA | OpenFlamingo 实验仅量化语言部分，不能推广为视觉编码器、投影器全部已量化 | 模块覆盖、图像/视频输入预处理、视觉 token 数、跨模态数据与评测 |

AWQ Table 5 在 WikiText2 上报告 Mixtral FP16 PPL 5.94、INT4-g128 6.05、INT3-g128 6.52；这是该权重量化配置的质量证据，不是 MoE 真低比特 GEMM 或专家并行加速证据。[AWQ v5](https://arxiv.org/html/2306.00978v5#S5)

## 深读案例：ViDiT-Q 如何处理生成过程

原文为 [ViDiT-Q: Efficient and Accurate Quantization of Diffusion Transformers for Image and Video Generation](https://arxiv.org/html/2406.02540v2)，版本 v2，2024-06-30。已读方法、实验与硬件附录，并核对 [LaTeX 源码](https://arxiv.org/src/2406.02540v2) 的 `ms.tex` 公式、表格与附录口径。

### 问题不是只有通道 outlier

§3.1 分析四类变化：token 间、CFG 条件/无条件分支间、去噪时间步间、输入通道间。原文主要量化 linear，保留 attention 的 FlashAttention 路径以及少量首尾嵌入/输出层；因此“模型 W8A8”不是每个运算和张量都转成 INT8。

### 实际流程

1. 运行代表性浮点生成轨迹，采集激活通道幅值与不同阶段的分布。
2. 对激活采用**在线、逐 token**的 MinMax 量化参数，使不同 token、CFG 分支和时间步不必共享同一个静态范围。
3. 用输入通道缩放在权重与激活之间转移幅值差异；前后半程分别选缩放强度。
4. 对更低位宽，进一步按层类型和时间步做敏感度分析，为关键层或阶段保留更高位宽。
5. 用视频的语义、视觉质量、时间一致性等指标评估整条生成轨迹，不能只凭某一层输出 MSE 排名决定分配。

论文 Eq. (1) 使用输入为行向量的记号：

$$
Y=XW=(XD^{-1})(DW),\qquad
D=\operatorname{diag}(s),\quad
s_i=\frac{\max|X_i|^\alpha}{\max|W_i|^{1-\alpha}}.
$$

这个等价关系在量化前成立；分别量化两侧后不保证相同误差。§3.2.3 根据生成阶段把流程分为两半，使用不同的 $\alpha$。必须对应保存/选择权重变换与激活变换；不能仅在 runtime 改激活除数而沿用不匹配的权重。论文报告在其试验中 20 次 PTQ 以内（$<$1 GPU 小时）可选到有效 $\alpha$，不是对任意 DiT 的通用预算。

动态量化消除了固定激活范围的一部分失配，但**没有消除离线校准需求**：通道平衡与混精计划仍基于校准数据。原文离线计划与 runtime 动态 scale 是两个不同状态。

### 混精分配与条件

§3.3 将视频质量影响分开考察：cross-attention 对内容，spatial attention/FFN 对视觉质量，temporal attention 对时间一致性。该关联是作者模型上的观察，不是所有 DiT 的恒定规律。时间步敏感度通过将轨迹分为四段分别量化来检查。

§4.1 所述方案中：W6A6 保留 blocks 6、26 的敏感 FFN 为 FP16；W4A8 将 FFN 保持 W8A8，并将前四分之一生成阶段设为 W8A8。这些层号和比例依赖实验模型，**不应直接硬编码到另一架构**。

视频实验包括 Latte 的 20 步 DDIM、CFG 7.0，与 STDiT 的 100 步 DDIM、CFG 4.0。改变 scheduler、步数或 CFG 后，原阶段敏感度和缩放计划需要重新验证。

### 有条件的质量与性能结果

Table 1 的 STDiT/OpenSORA VBench 结果摘录如下，保留多维指标，不把“lossless”解释成每项指标严格不变：

| 配置 | Imaging quality | Subject consistency | Overall consistency |
|---|---:|---:|---:|
| FP16 | 63.68 | 90.28 | 26.21 |
| Naive PTQ W8A8 | 56.52 | 85.30 | 25.42 |
| ViDiT-Q W8A8 | 63.48 | 90.24 | 26.06 |
| ViDiT-Q-MP W4A8 | 61.07 | 88.72 | 25.94 |

硬件证据必须另看 §4.3、Appendix A.1/A.4：RTX 4080、CUDA 12.1、batch 1；作者使用 CUTLASS 静态 INT8 kernel 计时与额外动态范围计算开销分析。该版缺少 W6A6/W4A8 kernel，表中相关延迟由 W8A8 代表；1.47× 是量化层、未量化层和转换开销分项计时后汇总的收益。不能据此声称已实测 W4A8 端到端 1.47×。BitOps、内存估计与实测延迟应分别标注。

### 论文与当前官方实现不同

锁定官方仓库 commit [`a8add8be17ad91069aba7a7209aa88e04b2fe41e`](https://github.com/A-suozhang/ViDiT-Q/tree/a8add8be17ad91069aba7a7209aa88e04b2fe41e)。README 说明 2025-01 重组代码并新增 CUDA kernel 与端到端路径，旧实现保留在 `viditq_old` 分支。本页没有执行这些路径，也不把 2025 代码支持反写为 2024 v2 论文证据。

已读具体入口：

- [`quant_utils/qdiff/base/base_quantizer.py`](https://github.com/A-suozhang/ViDiT-Q/blob/a8add8be17ad91069aba7a7209aa88e04b2fe41e/quant_utils/qdiff/base/base_quantizer.py)：`DynamicQuantizer` 在线沿二维输入每一行计算 min/max 或 absmax，并返回反量化值；这条 Python 路径是伪量化。
- [`quant_utils/qdiff/viditq/viditq_quant_layer.py`](https://github.com/A-suozhang/ViDiT-Q/blob/a8add8be17ad91069aba7a7209aa88e04b2fe41e/quant_utils/qdiff/viditq/viditq_quant_layer.py)：把输入 reshape 为 `[B*N_token,C]`，实现逐 token 激活范围。当前类还加入 Hadamard 旋转，不能称为论文 Eq. (1) 的逐行复现。
- 当前源码实现为 `weight_mask.abs()**alpha / act_mask.abs()**(1-alpha)`，输入乘 mask、权重除 mask；这与论文记号的倒数方向和 $\alpha\leftrightarrow1-\alpha$ 约定需要逐项对齐。源码中的 `abs()` 会先消除负值影响，但零值、极小值和幂运算的数值稳定性仍需检查，不能照搬论文超参数值。
- README 区分 `software_simulation` 和 `hardware` 配置；后者导出整数权重并使用扩展 kernel。是否覆盖目标模型与所有混精模式，仍需实际配置和运行验证。

### 对 Agent 的可迁移启发

迁移的是实验问题：是否存在阶段分布变化、静态范围是否失配、不同层影响哪些任务维度。不能把 ViDiT-Q 的视频指标、具体高精度层号或阶段划分直接移到 LLaDA。

建议的消融是：固定生成输入与轨迹配置，比较静态 tensor-wise、动态 token-wise、加通道平衡、加阶段/层混精；同时测局部误差和完整生成指标。所有阶段对应的权重变换、格式和 runtime 开销都应记录。

## 文本 dLLM：最小迁移证据清单

[LLaDA v2 §2 与 Fig. 2](https://arxiv.org/html/2502.09992v2#S2) 描述从完全 mask 到逐渐去 mask 的生成，mask predictor 同时预测 masked tokens，并使用 remasking 策略。它提供架构与过程证据，**不是低比特效果证据**。

把 AR 上的 AWQ/GPTQ/旋转等作为候选时，至少补齐：

1. **覆盖数据状态。** 采集真实采样过程中的 mask 比例、已生成 token、提示部分与目标部分，按阶段记录范围、outlier 和误差；仅用完整自然文本一次 forward 不能自动代表生成分布。
2. **损失与统计定义。** 明确 masked-token 目标、位置归约、提示 token 是否计入、步间权重；AR next-token loss 的敏感度排序不能直接复用。
3. **计算图契约。** 核对实际 attention mask、归一化、共享权重、旋转可交换位置和缓存策略。不能仅因名字包含 Transformer 就认为 AR KV-cache 或 RoPE 处理全部成立。
4. **轨迹控制。** 保持模型、tokenizer、seed、样本顺序、采样步数、block size、remasking 与评分一致；先比较逐样本输出和分阶段误差，再看总体分数。
5. **部署证据。** 确认实际走低比特 kernel，报告每步开销、生成步数与完整任务吞吐；AR decode 的收益不能直接换算为 dLLM 加速。

这些是根据过程差异推导的验证要求。第一版没有收集到并核实一项可在此页直接宣称“已验证 AR→LLaDA 量化迁移”的实验，未知项应保持未知。

## MoE 与 VLM 的额外边界

对 MoE，记录 router 是否量化、专家命中分布、稀疏/共享专家、量化前后路由变化；这些是目标模型的待测条件。ModelOpt 的专家同格式分组和 `active_moe` 成本模型是实现支持，不能替代路由质量测量。总模型存储与单 token 活跃参数成本必须分开。

对 VLM，AWQ OpenFlamingo 实验明确只量化语言部分；其 COCO captioning 5k 样本结果不能覆盖视觉塔和 connector。迁移至图像数量/分辨率可变的 VLM 时，应核对校准输入是否覆盖视觉 token 分布与文本交互，并单独评估视觉任务。文本 PPL 通过不足以证明图像理解保持。

## 形成可执行 Recipe 的停止条件

只有代数等价、或只有相近架构的论文分数时，状态应为“候选迁移”。至少在目标计算图、数据状态、格式/后端和完整生成指标上完成验证，才标记为目标场景可执行。若某一步变换改变了统计坐标或生成轨迹，重采相关统计；如果性能仅有估算，保留“估算”标签而不写加速承诺。

相关：[AutoQuantize](../algorithms/modelopt-autoquantize.md)、[HAWQ](../algorithms/hawq.md)、生成过程的直接量化见 [Q-DiT](../algorithms/qdit.md)（图像 DiT）与 [DLLMQuant](../algorithms/dllmquant.md)（文本 dLLM）、[来源登记](../../sources/topics/mixed-architecture.md)。
