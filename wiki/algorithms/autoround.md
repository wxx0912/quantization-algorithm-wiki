---
{
  "id": "algorithm-autoround",
  "title": "AutoRound / SignRound：用符号梯度联合优化舍入与裁剪",
  "type": "algorithm",
  "tags": [
    "ptq",
    "rounding",
    "reconstruction",
    "clipping",
    "llm"
  ],
  "aliases": [
    "AutoRound",
    "SignRound",
    "SignRoundV1",
    "signed gradient descent"
  ],
  "related": [
    "algorithm-adaround",
    "algorithm-awq",
    "algorithm-brecq"
  ],
  "sources": [
    "paper-autoround",
    "repo-autoround"
  ],
  "confidence": "source-reported",
  "evidence_scope": "source-reported",
  "updated": "2026-10-23"
}
---

# AutoRound / SignRound：用符号梯度联合优化舍入与裁剪

**AutoRound 是持续演进的工具，SignRound 是其中的算法。** 本页原理与实验以 [SignRound 论文 `2309.05516v5`](https://arxiv.org/pdf/2309.05516v5) 为准；当前仓库同时提供 SignRound V2、更多格式及组合功能，不能将它们全部归入 2024 年论文的结果。

## 输入、变量与计算

给定预训练 LLM、校准文本、目标 bit/group/zero-point 配置，按 Transformer block 缓存输入和浮点输出。保持原始浮点权重不变，为每个权重引入舍入偏移 $V$，同时学习量化范围上下端的收缩系数 $\alpha,\beta$。

为避免混淆整数编码与反量化，下面采用标准 affine 写法（对应原文 §3 式 (2)–(4) 的舍入偏移与范围缩放机制）：

$$s={\alpha\max(W)-\beta\min(W)\over 2^b-1},\quad
q=\operatorname{clip}\!\left(\operatorname{round}(W/s+z+V),q_{min},q_{max}\right),$$
$$\widetilde W=s(q-z).$$

$z$ 是与新范围相匹配的整数 zero-point；范围统计按实际量化 group 计算。核对 [v5 LaTeX 源码](https://arxiv.org/src/2309.05516v5) `acl_latex.tex` 后，原文式 (1)、(3) 的书写确实省略了反量化的 zero-point 减法，本页显式写出标准 affine 反量化，不逐字照抄该简写。公式中的 $\alpha,\beta$ 与 AWQ 的 scale 搜索指数不是同一变量。论文考虑 $V$ 的有效舍入偏移范围和 $\alpha,\beta\in[0,1]$，通过块输出重构优化三者：

$$\mathcal L(\alpha,\beta,V)=\|F_{\widetilde W}(X)-F_W(X)\|_F^2.$$

原文式 (5) 以 $WX$ 简写重构；实现中一个 block 含非线性、attention 和多个 linear，不能把整个块当作单个矩阵。算法 1 用量化反量化后的权重前向，再通过近似梯度更新。其核心是 SignSGD：

$$\theta_{t+1}=\theta_t-\eta_t\operatorname{sign}(\nabla_\theta\mathcal L),\quad
\theta\in\{V,\alpha,\beta\}.$$

只使用梯度方向，避免不同量级的梯度导致部分变量在有限步数内移动不足。论文使用线性衰减学习率，记录重构损失最优状态，最后固化为普通量化权重；推理不需要 SignSGD、$V$ 或优化器。使用反向传播不等于对原模型做全量端到端 QAT：本方法优化的是离线量化参数，目标是局部输出重构。

## 数据、预算与格式

论文 §4.1：pile-10k 随机抽取 512 条校准样本，SignRound 序列长度 2048；每块 200 步、batch size 8、学习率 $5\times10^{-3}$、线性衰减，使用 AMP。其对比说明其他算法使用各自官方序列长度等设置，所以不能宣称论文所有方法的计算预算严格相等。

原始研究重点为非对称 weight-only W4G-1、W4G128、W3G128、W2G128；G-1 表示不分子 group 的配置，不能当作 group size 1。它不直接证明 KV cache 或 W4A4 的质量。训练成本包含块输入输出缓存、逐权重舍入变量、前后向与最佳状态保存，不能只用“200 步”推断固定显存或耗时。

## 条件明确的结果

论文表 1 在 11 个零样本任务平均准确率上比较 LLaMA 与 Mistral，表 2 展示各任务详情。对 Mistral-7B-v0.1 / W2G128：FP16 63.30%，GPTQ 39.61%，AWQ 30.06%，默认 SignRound 52.71%。这些是论文报告，包含其模型、校准和评测设置，不表示当前实现或本仓库复现结果。

表中的 `Ours*` 是多个超参候选中选出的最好结果，不能与固定超参的 `Ours` 混为一谈，也不能忽略搜索成本。W4G128 下该模型默认 SignRound 为 62.62%，GPTQ 为 62.32%，HQQ 为 62.75%；这也说明算法不存在脱离配置的统一排名。原文舍入/裁剪消融（LaTeX 表标签 `tab:round_weight_clip_tuning`）还显示 LLaMA-2-13B / W2G128 下仅舍入为 54.41%，联合为 53.46%，因此联合优化并非每个配置都优于单项。

## 当前实现与论文的边界

已读 [intel/auto-round `a7656a27d020e636a961b35f84c551f453efe434`](https://github.com/intel/auto-round/tree/a7656a27d020e636a961b35f84c551f453efe434) 的 README、以下源码和组合说明：

- `auto_round/algorithms/quantization/sign_round/quantizer.py`：`SignRoundQuantizer.quantize_block`、最佳 loss 状态和迭代流程。
- `auto_round/algorithms/quantization/sign_round/sign_sgd.py`：`SignSGD` 更新实现。
- `auto_round/data_type/int.py::quant_tensor_asym`：范围缩放、zero-point、`round_ste` 和 `scale * (q - zp)` 的量化反量化路径。
- `docs/algorithm_combinations.md`：组合精度与部署两维评价。

当前 README 同时列出 SignRound V1/V2、WNA16、MXFP4、NVFP4、FP8 和 GGUF 等能力；这只是该工具版本的能力声明，实际启用算法、scheme、export 和目标 runtime 要分别记录。本页没有审计所有格式 kernel，也没有阅读 SignRound V2 全文，因此不展开其算法结论。

该 commit 的组合文档将 AWQ+SignRound 列为可部署，将 Hadamard/SpinQuant 组合的部署列为未就绪，并注明评价受实现成熟度和内部主观评估影响。它不是跨后端的支持承诺。导出文件成功不能证明目标 runtime 的真实低比特计算路径已使用。

## 组合与失败条件

缩放、旋转或低秩分解应先确定最终浮点图，再缓存输入/教师输出、初始化 $V$ 与 clipping 范围。变换之后不能继续使用旧 cache 或旧 rounding 参数。若与 AWQ 组合，两者都可能调整 clipping，必须明确前者输出是初始化还是固定约束，避免重复修改同一范围却仍声称复现原方法。

相较 AdaRound 固定网格的上下舍入，SignRound 联合调整 clipping 和 rounding；不能将 AdaRound 的固定-grid 假设直接套到这里。迁移到 MoE 时要覆盖专家路由和低频专家；迁移到 dLLM 时需要 mask/去噪阶段样本。当前工具出现相应模块，不替代任务与生成质量验证。

## Agent 消融

固定格式与校准集，比较 RTN、仅 clipping、仅 rounding、两者联合；再比较相同更新步数的 SignSGD 与其他优化器，记录 wall-clock 与显存。单独比较默认参数和额外搜索，检查校准与留出数据误差，以及 hard quantized 权重的下游质量。最小部署检查还应包括导出重载后的数值一致性和后端实际 kernel。

本项目仅完成资料核读，未运行算法或推理实验。
