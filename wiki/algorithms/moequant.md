---
{
  "id": "algorithm-moequant",
  "title": "MoEQuant：用专家均衡采样与 affinity 加权校准量化 MoE LLM",
  "type": "algorithm",
  "tags": [
    "moe",
    "weight-only",
    "calibration",
    "hessian",
    "sensitivity",
    "llm",
    "ptq"
  ],
  "aliases": [
    "MoEQuant",
    "EBSS",
    "AGQ",
    "专家均衡自采样",
    "affinity 引导量化",
    "MoE LLM 权重量化"
  ],
  "related": [
    "algorithm-mcmoe",
    "algorithm-gptq",
    "algorithm-awq",
    "algorithm-baselines",
    "foundation-quantization-contract",
    "guide-search-space"
  ],
  "sources": [
    "paper-moequant"
  ],
  "confidence": "source-reported",
  "evidence_scope": "source-reported",
  "updated": "2026-10-28"
}
---

# MoEQuant：用专家均衡采样与 affinity 加权校准量化 MoE LLM

## 一句话理解

MoEQuant 认为 MoE LLM 量化掉点的根因是两处 MoE 专有的"失衡"：校准样本在专家间分布长尾（冷门专家校准不足），以及 gating 让同一专家对不同 token 的相关性不同而常规校准把所有 token 一视同仁。它用 **EBSS** 让校准集在各专家间均衡、用 **AGQ** 把 gating 的 token-expert affinity 写进量化损失与 Hessian，且可叠加在 AWQ 或 GPTQ 之上。

## 核心机制

1. **inter-expert 失衡（EBSS）**：把校准集选择写成"低困惑度 + 专家均衡"的联合目标 `D*=argmin PPL(M,D)·exp(σ(M,D)/τ)`；用模型自采样（只依赖词表、贴近模型自分布）加概率引导路径剪枝（每轮保留 `w` 条分支）把搜索从 `O(m^n)` 降到 `O(wn)`；专家均衡项延迟到分支级计算，避免逐 token 遍历全词表。
2. **intra-expert 失衡（AGQ）**：把 gating 系数 `c_i` 展开进 `y_i=c_i·E(x_i)` 的线性运算，据此把逐层量化损失改成 affinity 加权 `L(Ŵ)=Σ_i c_i·‖Wx_i−Ŵx_i‖²`，Hessian 随之变成 `H=(X·√c)(X·√c)ᵀ`——高 affinity token 在敏感度与权重更新里权重更大。
3. **即插即用**：两项技术都不绑定量化器；论文分别在 AWQ 与 GPTQ 上验证（记 MoEQuant⁺ / MoEQuant⁺⁺）。

## 基本流程

1. 用 EBSS 自采样构造专家均衡的校准集：按累计对数概率打分生成候选分支 → 按 `score` 取 top-`w` 分支 → 评估专家均衡 → 产出校准集。
2. 用该校准集逐层量化（论文用 AWQ 或 GPTQ），其中误差/Hessian 按 AGQ 的 affinity 权重计算。
3. 只量化权重（对称均匀、per-channel）；其余走向与所选基座量化器一致，无微调。

## 适用条件

- 面向 **MoE LLM 的权重量化**（论文主打 4-bit 并报告 3-bit）；校准集在多数专家间分布长尾、或冷门专家校准不足时收益最明确。
- 需要能拿到 gating 权重（AGQ）与做模型自采样（EBSS）；校准集由模型自己生成，不依赖外部语料。
- 只压权重，不改激活精度；对 dense LLM 没有 MoE 失衡可治，本库认为收益有限（工程推断，原文未述）。

## 结果边界

- 论文报告（Table 1，4-bit，7 项任务 avg.）：Qwen-MoE-14B FP 51.20 → MoEQuant⁺⁺ 49.59；DeepSeek-MoE-16B FP 40.86 → 40.01；Mixtral-8x7B FP 56.80 → 55.58（对应 GPTQ 分别 49.00/39.01/53.42）。论文指出 GPTQ 在 WikiText2 上 PPL 更低是校准集过拟合，C4 更能反映泛化。
- 低比特（Table 3，3-bit avg.）：Mixtral-8x7B GPTQ 45.03 → MoEQuant⁺⁺ 49.75；DeepSeek-MoE-16B GPTQ 35.85 → 36.47（Table 9：3-bit 下 RTN 两模型 PPL 崩到数万、DeepSeek 的 AWQ 崩到数千，Mixtral 的 AWQ 仍约 7–13）。
- 消融（Table 5）：DeepSeek-MoE-16B 基线 39.01、仅 EBSS 39.87、仅 AGQ 39.50、叠加 40.01；`τ=1.2`、`w=4` 为默认最优/性价比点。
- 效率（Table 4，A6000）：4-bit 相对 FP16，解码加速 1.17×–2.08×、显存节省 3.27×–3.74×。速度为论文报告值，绑定实现与硬件。
- 只做权重 4-bit/3-bit 与 MoE 模型；`τ`、`w`、校准规模都是论文设定，未复现、未在目标后端核验。

## Agent 如何使用

处理"MoE 模型量化掉点比 dense 多、校准集要不要专门构造、专家不均衡怎么办"时先读本页，再按需展开：

| 需要确认的内容 | 继续读取 |
| --- | --- |
| EBSS/AGQ 公式、Table 1–9 与设置 | [`paper-moequant`](../../sources/papers/paper-moequant.md) |
| 另一条 MoE 压缩路线（专家级混精 + 动态剪枝） | [MC-MoE](mcmoe.md) |
| 权重量化的基线方法与数值口径 | [GPTQ](gptq.md)、[AWQ](awq.md)、[量化契约](../foundations/quantization-contract.md) |
| MoE 量化在搜索空间中的位置 | [量化搜索空间](../guides/search-space.md) |

回答时要区分三类贡献：校准集构造（EBSS）、误差/Hessian 加权（AGQ）、以及所选基座量化器本身；本方法只针对权重，不能推出激活或 KV 也能低位。

## 详细来源

- [MoEQuant 原论文来源页](../../sources/papers/paper-moequant.md)
