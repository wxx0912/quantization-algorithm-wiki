---
{
  "id": "algorithm-faircalib",
  "title": "FAIR-Calib：面向 dLLM 写入前沿的加权校准",
  "type": "algorithm",
  "tags": [
    "dllm",
    "calibration",
    "reconstruction",
    "mask",
    "activation"
  ],
  "aliases": [
    "FAIR-Calib",
    "前沿感知校准",
    "写入前沿翻转",
    "dLLM 校准重加权"
  ],
  "related": [
    "algorithm-dllmquant",
    "algorithm-starquant",
    "algorithm-quantdllm",
    "algorithm-flatquant",
    "foundation-quantization-contract",
    "guide-search-space"
  ],
  "sources": [
    "paper-faircalib"
  ],
  "confidence": "source-reported",
  "evidence_scope": "source-reported",
  "updated": "2026-10-28"
}
---

# FAIR-Calib：面向 dLLM 写入前沿的加权校准

## 一句话理解

FAIR-Calib 认为 dLLM 掉点的关键不在"模型整体"，而在**不可逆的写入前沿**：一旦某步把临界 token 写错，就再也改不回来，还会沿后续去噪步放大。于是它先用全精度教师探测出"哪些位置容易被写错、又影响后续"的位置先验 `w̄`，再用这份先验去加权逐层的隐状态对齐校准。它只改**校准怎么加权**，不改量化器本身。

## 核心机制

1. **问题刻画**：dLLM 每步预测所有位置分布、提交一部分 mask 位置；写入不可逆，产生了全精度下也存在的"稳定性滞后"。量化扰动在写入前沿翻转临界决策后，错误被锁定并被后续步放大（论文用 `mse_prob` 逐 step 上升展示）。
2. **Stage I 教师探测（Eq.9）**：在随机 commit 的全精度 rollout 上按步累加两种信号——**frontier-hit** 指示（本步是否在该位置 commit，配 early-boost 的时间权重 `λ0(t)`）与**掩码期可靠性** `c̃_{t,i}`（教师分布的锐度/margin 类分数，乘 `λ1`）。累加后对齐到末尾生成窗口、窗口外给小 floor，得到静态位置先验 `w̄`。
3. **Stage II 离策略加权校准（Eq.10）**：不做昂贵的端到端扩散 rollout，而是喂完全可见的真实 token，逐层最小化 `E[Σ_i w̄_i‖h^q_{ℓ,i}−h^⋆_{ℓ,i}‖²]`——把"前沿脆弱位置"保护起来，其余位置按普通重建对齐。
4. **理论基础**：在随机 commit 与温和平滑假设下，输出分布 KL 有"时间×位置"可加上界，且只由被 commit 的位置贡献；token KL 可经 logit 平方误差桥接到隐状态平方误差——这解释了为什么用加权的隐状态 MSE 而非直接对齐概率。

## 基本流程

1. 跑少量全精度教师 rollout（随机 commit），按 Eq.9 累积 `w̄`，并对齐到生成窗口。
2. 采用 FlatQuant 式的逐层可学习仿射变换，把权重/激活摊平后再做均匀对称量化。
3. 逐层、用完全可见的文本做 teacher-forcing，按 `w̄` 加权的隐状态 MSE 校准（更浅层冻结）。
4. 得到 W4A4 模型，推理时按各模型默认的 commit 策略执行。

## 适用条件

- 面向**掩码去噪、不可逆 commit** 的 dLLM 的 W4A4 量化；论文验证 LLaDA 家族与 Dream 家族。
- 需要有全精度教师可探测（一小段生成 rollout）与一小段校准文本（论文默认长度 1024）；沿用 FlatQuant 的仿射变换框架。
- 该方法只调整"校准损失里每个位置多重要"，不改变量化格式、位宽或后端 kernel。

## 结果边界

- 论文报告（W4A4，10 项平均；论文未报告评测硬件）：LLaDA-Base 由 QuaRot 58.78 / FlatQuant 59.37 提到 **61.09**（FP 62.12）；LLaDA-Instruct **72.40**；LLaDA-1.5 **72.75**；Dream-Base **64.64**（FlatQuant 62.08）；Dream-Instruct **66.66**（FlatQuant 63.98）。
- 消融：仅 frontier-hit 63.12、仅 masked-stage reliability 62.89、合并 64.64（Dream-Base）；探测预算约 512 后饱和；early-boost 的时间权重最优。
- 诊断：teacher-forced commit-step flip 由 2.9±1.4 降到 1.9±0.9（N=32）。
- 主结果覆盖 W4A4 与掩码去噪 dLLM（附录 E 另有 W3A4/W8A8，本页未纳入）；理论界依赖随机 commit 与平滑假设，真实模型相关 commit 会引入 policy-shift。`K`、`λ1`、探测预算、时间调度等为经验设定；未复现、未在目标后端核验。

## Agent 如何使用

处理 dLLM 校准掉点、想区分"校准数据不够"与"前沿位置没被保护"时先读本页，再按需展开：

| 需要确认的内容 | 继续读取 |
| --- | --- |
| Stage I/II 公式、Eq.9–10、理论 Lemma/Prop、Table 1–4 | [`paper-faircalib`](../../sources/papers/paper-faircalib.md) |
| 同族 dLLM 量化的其它切入 | [DLLMQuant](dllmquant.md)、[STaR-Quant](starquant.md) |
| 仿射变换摊平的基线 | [FlatQuant](flatquant.md) |
| 量化契约与搜索空间 | [量化契约](../foundations/quantization-contract.md)、[量化搜索空间](../guides/search-space.md) |

回答时要强调这是"校准重加权"方法：它叠加在既有逐层 PTQ/仿射变换之上，收益来自把重建误差集中到脆弱的前沿位置，而不是新增量化格式。

## 详细来源

- [FAIR-Calib 原论文来源页](../../sources/papers/paper-faircalib.md)
