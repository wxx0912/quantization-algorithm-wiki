---
{
  "id": "algorithm-quantdllm",
  "title": "Quant-dLLM：扩散语言模型的 2-bit 权重量化",
  "type": "algorithm",
  "tags": [
    "dllm",
    "calibration",
    "quantizer",
    "weight-only",
    "mixed-precision",
    "mask"
  ],
  "aliases": [
    "Quant-dLLM",
    "dLLM 2-bit 量化",
    "dLLM 极限低位权重量化",
    "掩码校准仿真"
  ],
  "related": [
    "algorithm-dllmquant",
    "algorithm-starquant",
    "algorithm-faircalib",
    "algorithm-baselines",
    "foundation-quantization-contract",
    "guide-search-space"
  ],
  "sources": [
    "paper-quantdllm"
  ],
  "confidence": "source-reported",
  "evidence_scope": "source-reported",
  "updated": "2026-10-28"
}
---

# Quant-dLLM：扩散语言模型的 2-bit 权重量化

## 一句话理解

Quant-dLLM 把 dLLM 的掩码去噪机制转成量化设计：用 MCS 让校准数据覆盖"不同 timestep × 不同 mask 比例"，用 DAQ 把权重拟合成"多个二值矩阵 + 行列 scale"的任意阶表示，用 ABMP 在严格 2-bit 平均预算下把少量敏感块升到 3-bit、等量块降到 1-bit。它只压权重、不做激活低比特量化，也不需要微调。

## 核心机制

1. **MCS（校准）**：固定一段可见前缀 `P={1..⌊γL⌋}`（默认 `γ=0.25`），其余位置按 `Bernoulli(α(t))` 采样可见；扫过离散的 `T` 个 timestep 生成校准批，使激活统计贴近掩码去噪的真实分布，而不是"全可见"文本。
2. **DAQ（量化器）**：每个权重矩阵表示为 `K` 个二值矩阵 `B_k∈{±1}` 之和，每项带自己的可分行列 scale `(α_r^{(k)}α_c^{(k)T})`，`K` 即该块的等效位宽。参数化分两步学：DOR 用掩码校准的二阶矩 `S_MCS` 推出重要性矩阵 `Z`，按 3σ 选离群并加权成 `Λ`，把目标改写为加权 Frobenius 重构；RSR 交替闭式更新 `α_r`/`α_c`、逐元素搜索 `B`。高阶通过拟合残差贪心叠加。
3. **ABMP（混合精度）**：按块重要性 `s_g=Σ Z_ij` 排序，层内 top-k 块给 3-bit、bottom-k 块给 1-bit、其余 2-bit，从而在 `(1/|G|)Σ b_g=2` 的严格平均预算下把表达能力集中到敏感块。

## 基本流程

1. 用 MCS 从校准文本生成覆盖各 timestep 与 mask 比例的校准批，聚合激活二阶统计 `S_MCS`。
2. 逐层处理：由校准统计算重要性矩阵 `Z` → ABMP 在 2-bit 平均预算下给出每块位宽 `{b_g}` → DAQ 按该位宽量化该层权重。
3. 重复到所有目标层；无梯度、无需训练，部署时按 DAQ 的二值分解与行列 scale 执行。

## 适用条件

- 面向**掩码迭代去噪**的 dLLM 的**权重压缩**；论文验证 LLaDA-8B-Base/Instruct、LLaDA-1.5、Dream-7B-Base/Instruct。
- 需要一小段校准文本（论文：128 条 C4、长度 4096）与对去噪调度的访问（MCS 需要 `α(t)`）。
- 只降权重位宽；能省权重存储，但不能替代激活量化、KV cache 量化或后端 kernel。

## 结果边界

- 论文报告（Table 1，2-bit 权重-only，group size 128，单卡 A800-80GB，7 项通用任务平均）：LLaDA-8B-Base FP 61.46、GPTQ 35.34、GPTAQ 35.87、Slim-LLM 42.39、Quant-dLLM 54.06；LLaDA-8B-Instruct 55.53、LLaDA-1.5 54.21、Dream-7B-Base 44.75、Dream-7B-Instruct 47.99；五模型平均由 Slim-LLM 40.9 提升到 51.3。
- 消融（§4.3，MMLU 5-shot）：MCS 使 LLaDA 52.10→56.87；ABMP 在 LLaDA 5% 最优、Dream 10% 最优；DAQ 由 baseline 39.26 经 RSR 48.32 到 RSR w/ DOR 56.87。
- 模型体积（LLaDA-8B-Base）：2-bit 下 3.69GB（GPTQ 3.70GB、Slim-LLM 3.72GB），相对 FP16 16.09GB。
- 只覆盖权重 2-bit 与 C4 校准；`γ`、timestep 档数、ABMP 比例、3σ 与 `λ` 都是经验超参。未复现、未在目标后端核验；数理/代码增益只有 Figure 3 概述口径，不能当确定数值。

## Agent 如何使用

处理"dLLM 权重能不能压到 2-bit、怎么压"时先读本页，再按需展开：

| 需要确认的内容 | 继续读取 |
| --- | --- |
| MCS/DAQ/ABMP 公式、Alg.1–3、Table 1–3 与设置 | [`paper-quantdllm`](../../sources/papers/paper-quantdllm.md) |
| 同为 dLLM、但做 W4A4 权重+激活的路线 | [DLLMQuant](dllmquant.md) |
| 训练后量化的数值口径与粒度 | [量化契约](../foundations/quantization-contract.md)、[基线](baselines.md) |
| dLLM 量化在搜索空间中的位置 | [量化搜索空间](../guides/search-space.md) |

回答时要区分"校准采样（MCS）""权重表示与补偿（DAQ/DOR/RSR）""位宽分配（ABMP）"三类贡献，并说明本方法只针对权重，不能推出激活或 KV 也能低位。

## 详细来源

- [Quant-dLLM 原论文来源页](../../sources/papers/paper-quantdllm.md)
