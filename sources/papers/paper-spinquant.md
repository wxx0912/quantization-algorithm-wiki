---
{
  "id": "paper-spinquant",
  "title": "SpinQuant: LLM Quantization with Learned Rotations",
  "category": "paper",
  "url": "https://arxiv.org/abs/2405.16406",
  "version": "v2（2024-05-28）",
  "version_kind": "paper-version",
  "retrieved_at": "2026-10-24",
  "read_form": [
    "latex",
    "html"
  ],
  "read_locator": "摘要、§1（Figure 1）、§2（Figure 2–4）、§3.1（Figure 5–6）、§3.2（Eq.1–3）、§4.1（设置）、§4.2（Table 1）、§4.3（Table 2–4）、§6 Conclusions、§7 Limitations and Broader Impacts、附录 A（Table 5–8）、附录 B（Eq.4、Figure 7、Table 9）、附录 C（Figure 8–11）",
  "topic": "transforms",
  "requires_external_verification": true,
  "excerpt_sha256": null
}
---
## 这份来源是什么

SpinQuant 原始论文（arXiv 2405.16406，v2；Meta）。出发点是"旋转能降低异常值"这一已知性质（QuIP/QuIP#、QuaRot），但论文先指出一个常被忽略的事实：**不同随机旋转的量化效果差异很大**（LLaMA-2 7B W4A4 下最好与最差差 13 个点，随机 Hadamard 也有 6 个点的波动）。于是它把旋转本身变成**可学习参数**，用 Cayley SGD 在 Stiefel 流形上优化，目标直接是量化网络的**任务损失**。网络全精度输出不变，因此是纯粹的"让权重/激活更好量化"的离线变换。

它是 `algorithm-spinquant` 与 `recipe-composition` 的依据：既给出"学旋转"的目标与更新式，也给出四个旋转位置各自能否被吸收进权重、哪些必须在线计算的结论。

范围边界：模型是 LLaMA-2(7B/13B/70B) 与 LLaMA-3(8B/70B)；主设置 W4A4 与 W4A4KV4，也报 weight-only（W4A16KV16）；量化用 RTN（权重 per-channel 对称、激活 per-token 非对称、KV group 128），并与 GPTQ 组合验证兼容性。方法本身是 PTQ，不改权重、不做 QAT。

## 已读定位与阅读形式

- 阅读形式：arXiv e-print 的 LaTeX 源（`arxiv.tex` + `sections/*` + `tables/*` + `figures/*` + `appendix.tex` + `appendix_tables/*` + `appendix_figures/*`），另读 arXiv HTML 渲染版交叉核对编号；v2 = 2024-05-28。
- 定位：§2 动机（Figure 2–4）、§3.1 旋转参数化（Figure 5–6）、§3.2 Cayley 优化（Eq.1–3）、§4 实验（Table 1–4）、§7 Limitations、附录 A（Table 5–8）、B（Eq.4、Figure 7、Table 9）、C（Figure 8–11）。编号：Figure 1–11、Table 1–9、Eq.1–4，**无 Algorithm**。

## 关键事实

- **为什么用旋转**（§2.1、Figure 2–3）：异常值会拉大量化范围、挤压其余值的有效位宽；随机旋转把大小权重/激活统计上混合。用 kurtosis `κ` 度量尾部厚度：旋转前很多层激活的 `κ` 超过 **200**，乘随机旋转后全部层降到 **≈3**（接近高斯），激活量化误差也明显下降。
- **随机旋转的方差**（§2.2、Figure 4）：LLaMA-2 7B、W4A4、100 次随机试验，最好与最差的 zero-shot 平均准确率差 **13 个点**；随机 Hadamard（对 `±1` 对角阵取 `H S` 的变体）整体更好，但方差仍有 **6 个点**——"随便找一个随机旋转"不能当作方法保证。
- **四个旋转位置**（§3.1、Figure 1、Figure 5–6）：`R₁` 残差流（`D_token × D_token`，旋转 embedding 输出、进入 attention/FFN 前用 `R₁ᵀ` 抵消，可吸收进相邻权重）；`R₂` attention 内按 head（`D_head × D_head`，逐层独立，也可吸收）；`R₃` Q/K 输出上的在线 Hadamard（RoPE 插在中间、无法吸收，保持固定 Hadamard）；`R₄` FFN down-projection 输入上的在线 Hadamard。可吸收意味着**不引入新参数**、全精度网络数值不变（pre-norm LLaMA 里靠把 RMSNorm 缩放折进后面的权重实现）。
- **优化目标与更新**（§3.2、Eq.1–3）：只学 `R₁`、`R₂`（`R₃`/`R₄` 固定为 Hadamard）：`argmin_{R ∈ M} L_Q(R₁, R₂ | W, X)`，`M` 是 Stiefel 流形、`L_Q` 是带伪量化的任务损失、权重 `W` 冻结；`R₁`/`R₂` 参数量只占权重总量的 **0.26%**。更新用 Cayley SGD：`R' = ΔR(Y)·R`，`ΔR(Y) = (I − αY/2)⁻¹(I + αY/2)`，`Y = Ĝ − Ĝᵀ`、`Ĝ = G Rᵀ − ½ R Rᵀ G Rᵀ`；`ΔR` 恒正交故 `R'` 保持正交，每步约为朴素 SGD 的 **2 倍**计算量。
- **训练与量化设置**（§4.1）：`R₁`、`R₂` 以随机 Hadamard 初始化，lr 从 **1.5** 线性衰减到 0，校准用 **800 条 WikiText-2 样本**、**100 iterations**；8×A100 上 LLaMA-3 8B ≈ **1.39 h**、LLaMA-2 7B ≈ **1.25 h**、13B ≈ **2.36 h**、70B ≈ **12 h**。权重用 **RTN** per-channel 对称（范围用 MSE 线性搜索），激活/KV 用非对称 min-max 动态量化（激活 per-token、KV group 128）；与 GPTQ 兼容时 GPTQ 用 128 条 WikiText-2、序列 2048 校准。
- **与 GPTQ 的组合**（§4.3.1、Table 2）：在"只有激活被量化"的网络（W16A4KV4）上优化 `R₁`/`R₂`、把权重量化交给 GPTQ：W4A4KV4 的 8 任务平均从 **60.9±0.6**（在 W4A4KV4 上优化）提到 **64.0±0.3**、WikiText-2 PPL 从 **6.8±0.15** 降到 **5.9±0.01**；W4A4KV16 同理从 61.0±1.0 / 6.7 提到 64.1±0.4 / 5.9。这是全篇后续采用的主配置。
- **主结果**（§4.2、Table 1；8 个 zero-shot 常识任务平均准确率，`*` 表示权重也用 RTN）：W4A4KV4——LLaMA-2 7B FP 66.9 → **SpinQuant 64.0**（差 **2.9** 点，QuaRot 62.5、QuaRot\* 58.7、LLM-QAT 44.9）、13B 68.3 → **66.9**、70B 72.9 → **71.2**；LLaMA-3 8B 69.6 → **65.2**（QuaRot 63.3）、70B 74.5 → **SpinQuant\* 70.1**（RTN）/ 69.3（GPTQ）。W4A4KV16 下 LLaMA-2 7B/13B/70B 的差距为 **2.8/1.1/1.8** 点。weight-only（W4A16KV16）：LLaMA-3 8B 69.6 → 68.5、LLaMA-2 7B 66.9 → 65.9（≈1 点），LLaMA-2 13B 的 68.5 略高于 FP16 68.3。
- **消融**（§4.3.2–4.3.3、Table 3–4）：LLaMA-2 7B W4A4KV4 不做旋转 PPL 2382.5 / 平均 37.1，只加 `R₁` → 9.9/51.8，`R₁+R₄` → 6.4/60.6，`R₁+R₂+R₄` → 6.3/61.0，四个都加 → 6.2/61.5；`R₁` 单独使用会让**权重量化**掉 3–5 点（W4A16KV16 63.6→60.2），加 `R₄` 后被抵消，`R₂` 主要贡献在 4-bit 激活（再涨 ≈0.4 点）。Cayley 优化**之前**普通浮点正交旋转明显不如 Hadamard（48.3±2.7 vs 58.7±1.0），**之后**几乎无差别（61.5±0.8 vs 61.5±0.3）。附录 A（Table 7–8）：128 与 800 样本的 PPL 都是 6.2±0.03，迭代 10/25/50/100/200 对应 6.6/6.4/6.3/6.2/6.2；4-4-16 上激活非对称优于对称（GPTQ 平均 63.3→64.0）。
- **机制分析**（附录 B）：梯度分析（Eq.4）说明全精度下旋转不影响输出、但**带量化时**该梯度一般非零，这正是"能学、且只在量化模型里值得学"的依据；loss 分析（Table 9，WikiText-2 测试集）用端到端量化信噪比说明收益：不旋转 −2.9 dB → 随机旋转 0.9 dB（+3.8）→ 学习后 **6.8 dB**（再 +5.9）；但逐层 SNR 并非普遍改善（少数层显著变好、多数层变化不大、个别层变差）。Limitations（§7）自述只在学术语境下评测，真实分布可能不同、泛化性待验证。

## 不能推出的结论

- 随机旋转的方差（13 点，Hadamard 也有 6 点）意味着**单次随机旋转不能当作方法保证**；本页收益都绑定"用 Cayley SGD 在 800 条 WikiText-2 样本上跑 100 iterations 学出来的"那一组 `R₁`/`R₂`。
- 覆盖范围只有 LLaMA-2/3（7B–70B）与 W4A4/W4A4KV4/W4A4KV16/W4A16KV16（GPTQ 兼容实验另用 W16A4KV4）；不能外推为 MoE、dLLM 或 DiT 上的已验证结论，也不含 chat/instruct 版模型或 QAT 场景。
- 旋转的数值等价性依赖具体结构前提：`R₁`/`R₂` 能被吸收进相邻权重（pre-norm + 可把 RMSNorm 缩放折进权重）、`R₃` 在 RoPE 之后、`R₂` 按 head 维切分；换网络布局（共享权重、tied embedding、非 RMSNorm、不同 head 划分）就要重新检查等价性。
- Table 1 中 `*`（权重也是 RTN）与不带 `*`（权重用 GPTQ）是**不同设置**、不能混比；论文正文报的"gap ≈4.4 点"在 LLaMA-3 70B 上取的是 70.1（`*`，RTN）而非 69.3（GPTQ）。
- 精度证据不提供后端证据：`R₃`/`R₄` 是在线变换，能否低开销落地取决于部署端是否有对应的融合实现；本仓库未运行旋转学习、校准、导出或 benchmark。
- 论文报告的时间（1.25 h / 1.39 h / 12 h）是 **8×A100 上的旋转学习耗时**，不是端到端量化或推理成本；§1 的"1.3 小时"与 §4.1 的"1.25 小时"指同一件事。
- Table 1 里 SmoothQuant/LLM-QAT/GPTQ/QuaRot 是论文用公开代码复现的数字，OmniQuant/AQLM/AWQ/QuIP/QuIP# 则引自各自论文，两类数字口径不同，不能直接横比。
