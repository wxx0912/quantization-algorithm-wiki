---
{
  "id": "paper-mcmoe",
  "title": "MC-MoE: Mixture Compressor for Mixture-of-Experts LLMs Gains More",
  "category": "paper",
  "url": "https://arxiv.org/abs/2410.06270",
  "version": "v1（2024-10-08）",
  "version_kind": "paper-version",
  "retrieved_at": "2026-10-28",
  "read_form": [
    "html"
  ],
  "read_locator": "摘要、§1 引言（Figure 1–2）、§3.1 预备（Eq.1–2）、§3.2 PMQ（3.2.1 显著性分析、3.2.2 位宽分配 Eq.3–4、Figure 3）、§3.3 ODP（3.3.1 注意力衰减 Eq.5、3.3.2 token 保护 Eq.6、Figure 4）、§4.1 Table 1–3、§4.2 Table 4、§5 结论、附录 A.1–A.5",
  "topic": "mixed-architecture",
  "excerpt_sha256": null
}
---
## 这份来源是什么

MC-MoE 论文（arXiv 2410.06270，v1；HKU/CUHK/北航），提出一套**免训练**的 MoE LLM"混合压缩"框架，把压缩分成两段：加载前的 **PMQ**（Pre-Loading Mixed-precision Quantization，按专家重要度做专家级混合精度量化）与推理时的 **ODP**（Online Dynamic Pruning，按 gating 权重动态剪掉低置信专家、并保护少量重要 token）。两项观察支撑其设计：不同专家在激活重构误差、路由分数与激活频率上差异很大；只有一小部分 token 是关键的。

它是 `algorithm-mcmoe` 的一手依据；在本库中作为"MoE 专家级混合精度 + 动态剪枝"的代表，与只做加权校准的 `paper-moequant`、以及通用混精页 `paper-hawq2` 对照。

范围边界：**weight-only**（专家权重 1/2/3-bit 混精，其余层 4-bit）；基座量化器是 GPTQ；免训练、无微调；PMQ 与 ODP 可分开使用。

## 已读定位与阅读形式

- 阅读形式：arXiv HTML 渲染版（`arxiv.org/html/2410.06270v1`），正文与表格按 HTML 结构解析；未读 LaTeX 源码或 PDF。
- 定位：§3.1 预备（Eq.1 MoE 前向、Eq.2 量化重构）、§3.2.1 专家显著性（Figure 3）、§3.2.2 位宽分配（Eq.3–4）、§3.3.1 注意力衰减（Eq.5、Figure 4）、§3.3.2 token 保护（Eq.6）、§4.1 Table 1–3、§4.2 Table 4、§5 结论、附录 A.1–A.5。References 未读。

## 关键事实

- **专家显著性**（§3.2.1）：用两项指标合成单个专家的显著性 `φ_i^α·w_i^β`，其中 `φ_i=n_i/N` 是访问频率（该专家被激活次数/数据集规模）、`w_i` 是激活权重均值（该专家路由权重之和/样本数）；`α`、`β` 是平衡两者的超参。
- **位宽分配（IP 模型）**（§3.2.2，Eq.3–4）：对"仅把专家 `e_i` 量化到 `j` 位"测输出 F-norm 误差 `ε_{i,j}`；在**整数规划**里最小化 `Σ_i Σ_j φ_i^α w_i^β (ε_{i,j} x_{ij})^γ`，约束平均位宽 `ΣΣ j·x_{ij}=n·k`、每个专家只取一个位宽、且至少 1 个专家取 3-bit、至少 1 个取 2-bit；位宽选项限定 `{1,2,3}`。解出后按 GPTQ 量化。作者称该 IP 求解只需约 1 秒（原文摘要与 §1 亦称 Linear Programming，正文 §3.1–§3.2 定为 Integer Programming）。其余权重（attention、gating）统一 4-bit，对平均位宽的增量不超过 0.05 bit。
- **ODP**（§3.3）：先用纯路由权重剪枝——对 top-2 的两个专家，当 `w1/w0<μ`（`μ` 取校准数据上 `w1/w0` 的中位数）时剪掉较小的那个（Eq.5）；论文报告这能减少约 15% 计算量但带来约 10% 性能下降，原因是"**注意力衰减**"（被剪专家对应的重要 token 在后续层的注意力分数塌陷，Figure 4）。
- **token 保护**（§3.3.2，Eq.6）：定义 token 重要度 `I_j=‖t_j‖_1·(Σ_{j≤i≤L} A_{j,i})/(L−j)`，同时用 ℓ1 范数与注意力分数；对被判为重要的 token，其所属专家不参与剪枝。论文称只保护 **2%** 的 token 就能显著缓解注意力衰减，同时维持约 15% 的计算压缩。
- **设置**（§4）：模型 Mixtral-8×7b（总量 49B、激活 13B、32 层、H=4096、8 专家）与 Mixtral-8×22b（141B、39B、56 层、H=6144、8 专家）；专家混精用 **C4、128 条随机序列、每条 2048 token** 校准，随后用 GPTQ 完成最终量化；其余非 MoE 层 4-bit；8×7b 在 2×A100-80GB、8×22b 在 4×A100-80GB 上压；用 HQQ 保存量化权重与反量化。评测 WikiText2 PPL + 8 项 zero-shot（PIQA/ARC-e/ARC-c/BoolQ/HellaSwag/Winogrande/MathQA/MMLU，EleutherAI LM Harness）。
- **主结果**（Table 2，Mixtral-8×7b，8 任务 avg.）：16-bit FP 71.29；均匀 3-bit GPTQ 69.09（−2.2%）、均匀 2-bit 42.67（−28.6%）；BSP 2.54-bit 49.07（−22.2%）；Hessian 分配 2.54-bit 67.18、2.05-bit 58.85、1.57-bit 45.91；**PMQ** 2.54-bit **67.50**（−3.8%）、2.05-bit 63.25、1.57-bit 54.49（−16.8%）。论文称 PMQ 在 2.54-bit 超过 BSP 18.4%，在 1.57-bit 仍高于 BSP 的 2.54-bit 结果 5.4%。
- **MMLU 5-shot**（Table 3，Mixtral-8×7b）：FP 70.60；均匀 2-bit 34.05；BSP 2.54-bit 31.57；Hessian 2.54/2.05/1.57-bit 58.22/43.51/31.96；PMQ 2.54/2.05/1.57-bit **61.19**/49.84/33.44。
- **消融**（§4.1 讨论 + Figure 5–6）：随机分配位宽明显掉点；只用路由分数好于随机但仍偏高；激活频率优于路由权重；F-norm 与 PMQ 都优于 Hessian 分配，且在 **2-bit 以下** PMQ 的领先随位宽降低而扩大。
- **显存/效率**（Table 4）：Mixtral-8×7b 16-bit 需 96.80GB 参数、激活参数 26.31GB；PMQ 2.54-bit 总参 16.24GB、激活参数 4.53GB、加速 1.63×，叠加 ODP 后激活参数降到 3.96GB、1.71×；2.05-bit + ODP 为 3.23GB、1.80×；1.57-bit + ODP 为 2.55GB、1.89×。Mixtral-8×22b 16-bit 281.24GB/76.49GB，2.54-bit PMQ 46.58GB/12.66GB、1.77×，叠加 ODP 后 10.96GB、1.82×。ODP 使激活参数约减 15%、平均精度下降不到 1%（速度与激活参数在单张 A100-80GB、C4 上测）。
- **token 保护消融**（§4.2）：2.05-bit Mixtral-8×7b 下，保护 2% 关键 token 使 PPL 由 6.46 降到 6.24，而激活专家参数只由 15.1% 降到 14.8%。
- **与 dense 对比**：论文称 2.54-bit 压缩后的 Mixtral-8×7b 在 MMLU 上超过 16-bit LLaMA2-13b 约 8%；Table 4 中 LLaMA2-7b/13b 的 LM-Eval 为 61.52/65.19，PMQ 2.54-bit 时 Mixtral-8×7b 为 67.50（Table 2 与 4）、Mixtral-8×22b 为 72.08（Table 4）；两者指标口径不同（MMLU 5-shot 与 LM-Eval 均值），不要混比。

## 不能推出的结论

- 本页只读正文与所列表格，未复现实验、未运行；PPL、任务分、显存与加速均为论文报告值。
- 只做**权重**量化（专家 1/2/3-bit 混精、其余 4-bit）：论文未报告激活低位量化，不能据此推断 W4A4/W8A8 有效。
- 平均位宽是**专家层**的预算口径，且"其余层统一 4-bit"；不能把某个平均位宽直接当成整模型位宽或整模型压缩比。
- ODP 是**推理时动态剪枝**，不是量化：它减少的是被激活的专家数（约 15%），会放宽数值精度但改变的是计算路径；不能与纯量化收益混算。
- 结论限于 Mixtral-8×7b/8×22b 与 C4 校准；不能外推到其它 MoE、dense LLM 或 dLLM/DiT。
- 论文 §4.2 正文写 2.54-bit Mixtral-8×7b 为 16.65GB 总参数 / 3.69GB 激活参数，与 Table 4 的 16.24GB / 4.53GB 不一致（正文另称整体压缩到 10.82–16.65GB）；引用体积与激活参数时以 Table 4 为准并注意该处出入。
- 各方法（BSP、Hessian）的对比行由作者复现/改编，配置是否各取最优未在表中说明；不能当严格的跨方法排名。
