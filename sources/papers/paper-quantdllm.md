---
{
  "id": "paper-quantdllm",
  "title": "Quant-dLLM: Post-Training Extreme Low-Bit Quantization for Diffusion Large Language Models",
  "category": "paper",
  "url": "https://arxiv.org/abs/2510.03274",
  "version": "v1（2025-09-27）",
  "version_kind": "paper-version",
  "retrieved_at": "2026-10-28",
  "read_form": [
    "html"
  ],
  "read_locator": "摘要、§1 引言、§3.1–3.5（Eq.2–11、Alg.1–3）、§4.1 设置、§4.2 Table 1、§4.3 Table 2(a–d)、§4.4 Table 3、§5 结论",
  "topic": "mixed-architecture",
  "excerpt_sha256": null
}
---
## 这份来源是什么

Quant-dLLM 论文（arXiv 2510.03274，v1），研究扩散大语言模型（dLLM）的**权重 2-bit** 训练后量化。它指出把自回归 LLM 的 PTQ 直接搬到 dLLM 的 2-bit 会明显掉点，提出三项技术：Masked Calibration Simulation（MCS，掩码校准仿真）、Data-aware Any-order Quantizer（DAQ，数据感知任意阶量化器）、Adaptive Blockwise Mixed Precision（ABMP，自适应分块混合精度），并在 LLaDA-8B-Base/Instruct、LLaDA-1.5、Dream-7B-Base/Instruct 上评估。

它是 `algorithm-quantdllm` 的一手依据。与 `paper-dllmquant`、`paper-starquant`、`paper-faircalib` 的分工：本篇是**权重-only、2-bit 平均预算**，后三者是权重+激活（W4A4）路线。

范围边界：PTQ、无训练/无梯度、只量化权重；不含激活低比特量化。

## 已读定位与阅读形式

- 阅读形式：arXiv HTML 渲染版（`arxiv.org/html/2510.03274v1`），正文与表格按 HTML 结构解析；未读 LaTeX 源码或 PDF。
- 定位：方法 §3.1 预备（二值矩阵参数化、Eq.2）、§3.2 MCS（Alg.1）、§3.3 DAQ（Eq.3–10、Alg.2）、§3.4 ABMP（Eq.11、Alg.3）、§3.5 pipeline；实验 §4.1 设置、§4.2 Table 1 主结果、§4.3 Table 2 消融、§4.4 Table 3 模型体积、§5 结论。References 未读。

## 关键事实

- **问题**（摘要、§1）：dLLM 的掩码去噪激活不同于标准 PTQ 假设的"全可见"信号；作者称 2-bit 下误差来自 ① timestep/mask 使激活分布偏离全可见校准，② 量化误差跨去噪步累积、越到后期越大。
- **MCS**（§3.2，Alg.1）：给定全可见序列，固定确定性可见前缀 `P={1,…,⌊γL⌋}`，其余位置按 `Bernoulli(α(t))` 采样可见，再置为 `[MASK]`；把 `[0,1]` 离散成 `T` 个 timestep 扫一遍得到校准集。默认 `γ=0.25`，固定随机种子。
- **DAQ**（§3.3，Eq.3–10）：把每个权重矩阵写成 K 个二值矩阵之和、每个二值矩阵带可分行列 scale：`Ŵ=Σ_{k=1..K}(α_r^{(k)}α_c^{(k)T})⊙B_k`，`B_k∈{−1,+1}`，K 即该块位宽。含两步：**DOR**（数据感知目标重写，Eq.5–6）把 MCS 的未中心二阶矩 `S_MCS` 蒸馏成重要性矩阵 `Z=(WD⁻¹)⊙(WD⁻¹)`，按 3σ 选离群得掩码 `Λ=1+(λ−1)Π`，把目标变成加权 Frobenius `‖Λ⊙(W−Ŵ)‖_F²`；**RSR**（行列交替重缩放，Eq.8–9）对 `α_r`/`α_c` 闭式交替更新、`B` 用逐元素最小误差搜索（Eq.10）。K 阶用拟合残差的贪心扩展（Alg.2）。
- **ABMP**（§3.4，Eq.11）：按块重要性 `s_g=Σ_{(i,j)∈g}Z_ij` 在层内排序，top-k 块给 3-bit、bottom-k 块给 1-bit、其余 2-bit，满足严格平均预算 `(1/|G|)Σ_g b_g=2`；默认重分配比例 5%（`k=⌊0.05|G|⌋`）。
- **pipeline**（§3.5）：先 MCS 造校准集，再逐层——由校准统计算重要性矩阵 Z → ABMP 定位宽 `{b_g}` → DAQ 按位宽量化该层权重；逐层进行、无梯度。
- **设置**（§4.1）：单卡 NVIDIA A800-80GB，PyTorch + HuggingFace；PTQ 无需训练或反向传播；**128 条 C4 校准样本、长度 4096、group size 128**；模型 LLaDA-8B-Base、LLaDA-8B-Instruct、LLaDA-1.5、Dream-7B-Base、Dream-7B-Instruct；基线 GPTQ、GPTAQ、Slim-LLM（同校准数据与长度）；评测 7 项通用任务（MMLU/ARC-E/ARC-C/HellaSwag/WinoGrande/PIQA/BBH，通用任务 n-shot 128 MC 采样）+ 数理（GSM8K/MATH/GPQA）+ 代码（HumanEval/MBPP，Pass@1）。
- **主结果**（Table 1，2-bit，7 项通用任务平均）：LLaDA-8B-Base FP 61.46、GPTQ 35.34、GPTAQ 35.87、Slim-LLM 42.39、Quant-dLLM **54.06**；LLaDA-8B-Instruct Quant-dLLM **55.53**（Slim-LLM 49.41）；LLaDA-1.5 **54.21**（Slim-LLM 45.44）；Dream-7B-Base **44.75**（Slim-LLM 31.09）；Dream-7B-Instruct **47.99**（Slim-LLM 32.86）。论文称五模型平均由 GPTQ 36.5 / GPTAQ 35.6 / Slim-LLM 40.9 提升到 **51.3**。
- **数理/代码**（§4.2，Figure 3）：论文称其方法在 LLaDA-Instruct 上数理与科学平均 >30%（基线 <12%）、代码在 LLaDA-Instruct 与 LLaDA-1.5 上 >15%（基线近 0%）；未给逐值表。
- **消融**（§4.3，Table 2，LLaDA-8B-Base / Dream-7B-Base，MMLU 5-shot，group 与 block 均 128）：MCS 使 LLaDA 52.10→56.87、Dream 37.81→40.22；ABMP 在 LLaDA 5% 最优 56.87（无 ABMP 54.32）、Dream 10% 最优 40.22（无 ABMP 32.75）；DAQ 拆分 baseline 39.26→+RSR 48.32→+RSR w/ DOR 56.87（LLaDA）；校准样本 64/128/256→54.49/56.87/55.59（LLaDA）。
- **模型体积**（§4.4，Table 3，LLaDA-8B-Base）：FP16 16.09GB、GPTQ 2-bit 3.70GB、Slim-LLM 3.72GB、Quant-dLLM **3.69GB**；作者称行列 scale 比常规 zero-point+scale 的内存开销更小。

## 不能推出的结论

- 本页只读正文与所列表格，未复现实验、未运行；准确率与体积均为论文报告值。
- 结论限于 LLaDA-8B(-Base/Instruct)、LLaDA-1.5、Dream-7B(-Base/Instruct) 与 C4 校准；不能外推到其它 dLLM、AR LLM 或图像/视频 DiT。
- 只做**权重-only 2-bit**：论文未报告激活低位量化，不能据此认为 W2A4/W4A4 有效或该模型可做真实低比特 GEMM。
- Table 1 为 7 项通用任务平均；数理与代码结果只有 Figure 3 的概述口径，未给逐值表，不能当作确定数值引用。
- MCS 的 `γ=0.25`、timestep 档数、ABMP 的 5%/10% 重分配比例、DAQ 的 3σ 阈值与 `λ` 均为经验设定，换模型或数据需重确。
- 论文声明代码与模型将发布于 `https://github.com/ZTA2785/Quant-dLLM`（本轮核验该地址返回 200，但未读仓库内容）；本库无对应 repo 来源页，不能核对实现细节。
