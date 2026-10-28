---
{
  "id": "paper-moequant",
  "title": "MoEQuant: Enhancing Quantization for Mixture-of-Experts Large Language Models via Expert-Balanced Sampling and Affinity Guidance",
  "category": "paper",
  "url": "https://arxiv.org/abs/2505.03804",
  "version": "v1（2025-05-02）",
  "version_kind": "paper-version",
  "retrieved_at": "2026-10-28",
  "read_form": [
    "html"
  ],
  "read_locator": "摘要、§1 引言（Figure 1–2）、§3 预备、§4.1 MoEQuant、§4.2 EBSS（Eq.9–14、Figure 4–5）、§4.3 AGQ（Eq.15–19）、§5.1 设置、§5.2 结果（Table 1–4）、§6 结论、附录 A.1 消融（Table 5–7）、附录 A.2 完整结果（Table 8–9）",
  "topic": "mixed-architecture",
  "excerpt_sha256": null
}
---
## 这份来源是什么

MoEQuant 论文（arXiv 2505.03804，v1；Houmo AI 与东南大学），研究 **MoE LLM 的训练后权重量化**，把问题归到 MoE 的两个独有失衡：**inter-expert imbalance**（校准样本在各专家间分布不均，长尾里冷门专家校准不足）与 **intra-expert imbalance**（gating 让同一专家对不同 token 的相关性不同，而常规逐层校准把所有 token 一视同仁）。对策是两项即插即用技术：**EBSS**（Expert-Balanced Self-Sampling，用模型自采样 + 累计概率与专家均衡指标构造专家均衡的校准集）与 **AGQ**（Affinity-Guided Quantization，把 gating 的 token-expert affinity 写进量化损失与 Hessian）。

它是 `algorithm-moequant` 的一手依据；在本库中作为"MoE 因其动态路由需要专门校准与加权误差"的代表，与通用 LLM 的 `paper-gptq`、`paper-awq`、`paper-smoothquant` 对照，与同为 MoE 压缩的 `paper-mcmoe` 互补（本篇加权校准，后者专家级混精 + 动态剪枝）。

范围边界：**weight-only**（对称均匀、per-channel 权重），论文主打 4-bit 并报告 3-bit（平均在正文 §5.2、完整在附录 A.2）；PTQ、无微调；方法与具体量化器正交（论文在 AWQ 与 GPTQ 上各叠加一版）。

## 已读定位与阅读形式

- 阅读形式：arXiv HTML 渲染版（`arxiv.org/html/2505.03804v1`），正文与表格按 HTML 结构解析；未读 LaTeX 源码或 PDF。
- 定位：§3 预备（Eq.1 MoE 前向、Eq.2 PPL、Eq.5–8 量化映射/反量化/损失/Hessian）、§4.2 EBSS（Eq.9–14）、§4.3 AGQ（Eq.15–19）、§5.1 设置、§5.2 Table 1（4-bit 主结果）、Table 2（指令微调）、Table 3（3-bit）、Table 4（速度/显存）、§A.1 Table 5–7（消融与超参）、§A.2 Table 8–9。References 未读。

## 关键事实

- **两个失衡**（摘要、§1）：① 样本在专家间分布不均——论文给出 Qwen-MoE-A2.7B-14B 第一层 MoE 的样本分布（C4、WikiText2 各 128×512 token）呈长尾；② 同一专家对不同 token 的 affinity（gating 权重）不同，GPTQ 收集 Hessian 时忽略 gating，扭曲了样本重要性。MoE 层占 transformer block 参数的约 80%（按激活专家）/最多约 97%（含全部专家）。
- **EBSS**（§4.2）：把校准集选择写成"低困惑度 + 专家均衡"的联合目标 `D* = argmin_D PPL(M,D)·exp(σ(M,D)/τ)`（Eq.9–10），属 NP-hard；实际用**模型自采样**（只依赖词表、天然贴近模型自分布）加**概率引导路径剪枝**（Eq.13–14，每轮只保留 `w` 条分支）把搜索复杂度从 `O(m^n)` 降到 `O(wn)`；专家均衡项**延迟到分支级**计算（Eq.14 的专家均衡按已定序列 `S` 计算、不并入候选 token `v` 的专家分布），避免逐 token 遍历全词表。默认 `τ=1.2`、`w=4`。
- **AGQ**（§4.3，Eq.15–19）：把 gating 系数 `c_i` 从 `y_i=c_i·E(x_i)` 逐层展开成恒等形式——`((c_i·x·W^up)⊙f(x·W^gate))W^down`、`((x·W^up)⊙f(x·W^gate))(c_i·W^down)`；论文再给出近似式 `((x·W^up)⊙f(c_i·x·W^gate))W^down`（Eq.17 第三行用 ≈，不是恒等），用以说明 `c_i` 可折进该层线性运算；据此把逐层量化损失改写为 **affinity 加权** `L(Ŵ)=Σ_i c_i·‖Wx_i−Ŵx_i‖²`（Eq.18），对应 Hessian 变为 `H=(X·√c)(X·√c)ᵀ`（Eq.19，亦等价写作 `(X·c)Xᵀ`），高 affinity token 在敏感度与权重更新中权重更大。
- **设置**（§5.1）：对称均匀 per-channel 的**权重**量化；NVIDIA A6000；模型 DeepSeek-MoE-16B、Qwen-MoE-14B、Mixtral-8x7B 及各自 chat 版；基线默认 128 条 WikiText2 校准（MoEQuant 改用 EBSS 校准）；评测 WikiText2/C4 PPL + MMLU/HumanEval/GSM8K/BoolQ/HellaSwag/OpenBookQA/MathQA（BoolQ/HellaSwag/OpenBookQA/MathQA 等用 lm-evaluation-harness 0.4.4，MMLU/GSM8K/HumanEval 用各自官方 repo）。GPTQ 实现先用等价 Hadamard 变换去权重离群（与 QuaRot 一致）、不做在线变换。
- **主结果**（Table 1，4-bit，7 项任务 avg.）：Qwen-MoE-14B FP 51.20、RTN 36.89、AWQ 42.52、GPTQ 49.00、MoEQuant⁺(基于 AWQ) 44.27、MoEQuant⁺⁺(基于 GPTQ) 49.59；DeepSeek-MoE-16B FP 40.86、RTN 35.85、AWQ 39.23、GPTQ 39.01、MoEQuant⁺ 39.68、MoEQuant⁺⁺ 40.01；Mixtral-8x7B FP 56.80、RTN 47.18、AWQ 48.64、GPTQ 53.42、MoEQuant⁺ 51.19、MoEQuant⁺⁺ 55.58。论文指出 GPTQ 在 WikiText2 上 PPL 更低是校准集过拟合，C4 才反映泛化。
- **指令微调**（Table 2，4-bit）：Qwen-MoE-14B-Chat 的 MoEQuant⁺⁺ 在 MMLU/HumanEval/GSM8K 为 58.00/21.95/29.11（FP 59.00/21.34/30.71），论文称保持 >94% 满精度；摘要"HumanEval 提升 >10 点"只写 DeepSeekMoE-16B；与 chat 版数字（RTN 10.41 → MoEQuant⁺⁺ 21.95）吻合，本页据此推断指 chat 版。
- **低比特**（Table 3，3-bit avg.）：DeepSeek-MoE-16B FP 40.86、RTN 20.17、AWQ 22.20、GPTQ 35.85、MoEQuant⁺ 26.65、MoEQuant⁺⁺ 36.47；Mixtral-8x7B FP 56.80、RTN 18.64、AWQ 36.05、GPTQ 45.03、MoEQuant⁺ 39.30、MoEQuant⁺⁺ 49.75。Table 9 显示 3-bit 下 RTN 在两模型上 PPL 崩到数万、DeepSeek 的 AWQ 崩到数千，而 Mixtral 的 AWQ 仍约 7–13。
- **速度与显存**（Table 4，4-bit 相对 FP16，A6000）：解码速度 1.27×（Qwen-MoE-14B）/1.17×（DeepSeek-MoE-16B）/2.08×（Mixtral-8x7B）；显存节省 3.28×/3.27×/3.74×（论文称平均 >1.2× 加速、>3.2× 显存节省）。
- **消融**（§A.1，Table 5，GPTQ 基线，7 项任务 avg.）：DeepSeek-MoE-16B 基线 39.01、仅 EBSS 39.87、仅 AGQ 39.50、两者叠加 40.01；Mixtral-8x7B 基线 53.42、仅 EBSS 55.15、仅 AGQ 54.24、叠加 55.58。超参 `τ=1.2` 最优（Table 6），`w=4` 性价比最优（Table 7）。

## 不能推出的结论

- 本页只读正文与所列表格，未复现实验、未运行；PPL、任务分、速度与显存均为论文报告值。
- 只做**权重**量化：论文未报告激活低位量化，不能据此推断 W4A4/W8A8 有效，也不能当作真实低比特 GEMM 的证据。
- EBSS 依赖**模型自采样**与对 gating/专家分布的访问，默认 `τ=1.2`、`w=4`；换模型或任务需重调，不能当通用校准默认值。
- 结论限于 DeepSeek-MoE-16B、Qwen-MoE-14B、Mixtral-8x7B(-Chat) 与 WikiText2 校准（WikiText2/C4 评测）；不能外推到 dense LLM、其它 MoE 或 dLLM/DiT。
- Table 1 的 "+/++" 表示基于 AWQ / 基于 GPTQ 的 MoEQuant，不是位宽差异；不能把 MoEQuant⁺ 与 MoEQuant⁺⁺ 当成同一实现的两次结果。
- 论文 §A.1 正文称 EBSS 约 +1.3%、AGQ 约 +2%、两者约 +2.6%，与 Table 5 的平均分列（DeepSeek 39.01→39.87/39.50/40.01）对不上；引用增益时以表为准。
