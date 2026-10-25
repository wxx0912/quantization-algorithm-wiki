---
{
  "id": "paper-omniquant",
  "title": "OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models",
  "category": "paper",
  "url": "https://arxiv.org/abs/2308.13137",
  "version": "v3（2024-03-18）",
  "version_kind": "paper-version",
  "retrieved_at": "2026-10-24",
  "read_form": [
    "latex"
  ],
  "read_locator": "摘要、§1（Figure 1–2）、§2 Related Work、§3（Eq.1 目标、Eq.2 LWC、Eq.3–5 LET；Figure 3 在源码中位于 OmniQuant 章节之前，官方渲染归 §2）、§4（4.1 Settings、4.2 Table 1、4.3 Table 2、4.4 instruction-tuned Figure 4、4.5 Table 3）、§5 Conclusion、附录（Algorithm 1、Table A1–A25、Figure A1–A6）",
  "topic": "core",
  "excerpt_sha256": null
}
---
## 这份来源是什么

OmniQuant 原始论文（arXiv 2308.13137，v3；ICLR 2024）。它把量化参数（权重 clipping 阈值、激活等价变换）从手工设定改为**可微学习**：在 block-wise 量化误差最小化框架下冻结 FP 权重，只学少量量化参数，同时覆盖 weight-only（W2A16/W3A16/W4A16）与 weight-activation（W6A6/W4A4）。

它是 `algorithm-omniquant` 的原理依据，也是 `algorithm-baselines`（对 RTN/GPTQ/AWQ、SmoothQuant/OS+/RPTQ/LLM-QAT）与 `foundation-quantization-contract` 的一手材料。

范围边界：方法本身是 PTQ 但要**梯度优化**（20–40 epochs）；只处理线性层与 attention 的 Q/K/V，SoftMax 输出保持全精度；实测部署只做 weight-only。

## 已读定位与阅读形式

- 阅读形式：arXiv e-print 的 LaTeX 源（单文件 `iclr2024_conference.tex`，正文与附录同文件）；v3 = 2024-03-18。**未读 HTML/PDF**，编号以 LaTeX 源为准。
- 定位：§3.1 目标（Eq.1）、§3.2 LWC（Eq.2）、§3.3 LET（Eq.3–5）、§4 实验（4.2 Table 1、4.3 Table 2、4.4 Figure 4、4.5 Table 3）、附录 Algorithm 1 与 Table A1–A25、Figure A1–A6。注意正文表打印为 1–3，`\appendix` 后重置并加 A 前缀（附录表 A1–A25，全文共 28 张）。

## 关键事实

- **优化目标**（§3.1，Eq.1）：`argmin_{Θ1,Θ2} ‖F(W,X) − F(Q_w(W;Θ1,Θ2), Q_a(X;Θ2))‖`，**逐 block 顺序优化**；冻结 FP 权重是因为全权重优化解空间过大、难以优化。
- **LWC**（§3.2，Eq.2）：`W_q = clamp(⌊W/h⌉ + z, 0, 2^N−1)`，`h = (γ·max(W) − β·min(W))/(2^N−1)`、`z = −⌊β·min(W)/h⌉`；`γ, β ∈ [0,1]` 是可学习的**裁剪强度**（sigmoid 实例化），`γ=β=1` 退化为 MinMax。论文刻意学"强度"而非直接学阈值，以降低优化难度。
- **LET**（§3.3，Eq.3–5）：线性层 `Y = [(X−δ) ⊘ s]·[s ⊙ W] + [B + δW]`，`s`、`δ` 可学习且能**离线吸收**进前序 LayerNorm/线性层，不引入额外参数或运行时开销（只对除 FFN 第二线性层外的所有线性层施加）。attention 对 Q/K 做 `P = Softmax((Q ⊘ s_a)(s_a ⊙ Kᵀ))`，`s_a` 可吸收进 Q/K 投影；V 的显式变换被省略。
- **设置**（§4.1）：weight-only 默认 INT4/INT3/INT2 per-channel 权重；weight-activation 默认 INT6/INT4 权重 + per-token 激活；除 SoftMax 输出外中间激活都量化（softmax 输出长尾、不适合均匀量化）。缩放因子用 SmoothQuant 初始化、平移用 Outlier Suppression+ 初始化；AdamW、零 weight decay、lr LWC 5e-3 / LET 1e-2；校准集 WikiText2 **128×2048**；单张 A100、batch 1、**20 epochs**（W2A16 用 40）。weight-only 下 OPT 两者都用、**LLaMA 只用 LWC**。
- **weight-only 结果**（§4.2，Table 1，WikiText2 PPL）。W2A16 反差最大：LLaMA-1-7B 上 RTN 1.1e5、GPTQ 2.1e3、**OmniQuant 15.47**；LLaMA-2-70B 上 GPTQ 77.95、OmniQuant 7.81。W3A16g128：OmniQuant 6.15/5.44/4.56/3.94（LLaMA-1 7B/13B/30B/65B）与 6.03/5.28/3.78（LLaMA-2 7B/13B/70B），优于 GPTQ 与 AWQ（AWQ 在 LLaMA-2-70B 行未给数值，表中为"-"）；W4A16g128 已接近无损。
- **weight-activation 结果**（§4.3，Table 2，6 个 zero-shot 平均）。LLaMA-1-7B W4A4：SmoothQuant 38.41、LLM-QAT 41.27、OS+ 48.43、**OmniQuant 52.65**（FP16 64.09）；W6A6：OS+ 61.13、SmoothQuant 62.81、**OmniQuant 63.17**。LLaMA-1-13B/30B/65B 的 W4A4 为 54.37/56.63/59.22。论文把 W8A8 排除在对比外（SmoothQuant 在 W8A8 上几乎无损）。
- **真实设备部署**（§4.5，Table 3；MLC-LLM，A100-80G，生成 512 token）。LLaMA-7B：FP16 12.6G 权重/69.2 tok/s → `W4A16g128` 3.8G/**134.2** tok/s；65B 的 FP16 版本 OOM → `W4A16g128` 33.0G/24.3；论文称接近**翻倍**。**部署只做 weight-only**（W4A4/W6A6 缺硬件支持）。
- **训练时间与消融**（打印为 Table A12，在附录 A.3 末尾、A.4 讨论）。单张 A100-80G、128×2048、20 epochs、batch 1：LLaMA 7B/13B/30B/65B 的 weight-only 为 1.1/2.2/4.5/8.9 小时，weight-activation 为 1.6/3.3/7.3/14.4 小时，约为 **GPTQ 的 5 倍**，但远快于 QAT。附录 A.3：weight-activation 下去掉 LET 会让 W4A4 的 PPL 涨到 `e3` 量级；weight-only 下 LET 对 OPT 提升明显、对 LLaMA 微弱（naive W3A16 时 LLaMA 10.68、OPT 4.6e3），这是"LLaMA 只用 LWC"的依据。代码 `github.com/OpenGVLab/OmniQuant`。

## 不能推出的结论

- LET 可离线吸收不等于零成本：整个过程需在 128×2048 校准样本上做 20–40 epochs 梯度优化，约为 GPTQ 的 5 倍耗时，不能当作 one-shot PTQ。
- Table 3 的加速/显存收益是 **weight-only**、MLC-LLM + A100-80G、生成 512 token 的测量；论文没做 W4A4/W6A6 部署，不能把表内加速当作 W4A4 端到端收益。
- Table 2 的对比来自不同来源：SmoothQuant 与 OS+ 是论文自己复现、LLM-QAT 是 QAT，不能当作同一 pipeline 的严格公平对照。
- W4A4 不是无损：LLaMA-1-7B 上 52.65 vs FP16 64.09（差 11.4 分），论文结论只是"显著优于同期方法"，不能读成"W4A4 可直接生产"。
- 校准集是 WikiText2 128×2048；论文附录 A.3 用 WikiText2/C4/Pile 做过校准集消融，PPL 波动 0.0006–0.17、并称 16 条样本即可收敛，故对校准分布不敏感；但 lr（5e-3/1e-2）与 20–40 epochs 等超参仍非普适。
- 方法只覆盖"除 FFN 第二线性层外的线性层 + attention Q/K/V"，SoftMax 输出不量化；不能当作"任意算子都能 W4A4"的依据，也不能推出 V 需要显式变换。
- Falcon-180B、OPT 全家族、LLaMA-1/2 的完整结果已逐值核对（附录 A.8 Full Results）：Table A18 的 Falcon-180B 从 BF16/FP16 3.29/6.64/6.31 到 W3A16g512 的 OmniQuant 3.71/6.95/6.71；Table A20–A22（OPT weight-only）、Table A25（OPT weight-activation）、Table A19/A23/A24（LLaMA-1/2）亦逐值读到，未见与前文不一致。这些仍只是论文给定配置下的结果。
