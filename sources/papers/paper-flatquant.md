---
{
  "id": "paper-flatquant",
  "title": "FlatQuant: Flatness Matters for LLM Quantization",
  "category": "paper",
  "url": "https://arxiv.org/abs/2410.09426",
  "version": "v1（2024-10-12）",
  "version_kind": "paper-version",
  "retrieved_at": "2026-10-24",
  "read_form": [
    "latex",
    "html"
  ],
  "read_locator": "摘要、§1、§2（Figure 1–2）、§2.1（Eq.1）、§3.1（Eq.2–4、Figure 3）、§3.2、§3.3、§4.1（设置；Table 1–2 浮动体在此）、§4.2（Main Results，讨论 Table 1–3）、§4.3（Figure 4–6）、§4.4（Table 4–6）、§5 结论、附录 A–E（Eq.5–7、Table 7–15、Figure 7–18、kernel 与 FLOP 分析）",
  "topic": "transforms",
  "excerpt_sha256": null
}
---
## 这份来源是什么

FlatQuant 原始论文（arXiv 2410.09426，v1）。它把量化困难归到权重与激活分布的**平坦度**，主张现有 pre-quantization 变换（per-channel scaling、Hadamard/正交旋转）仍不足，于是为**每个线性层**学一个可逆仿射变换，并用 Kronecker 分解把在线变换的开销压到很低。名字取自 "Fast and Learnable Affine Transformation"。

它是 `algorithm-flatquant` 与 `recipe-composition` 的原理依据：关键结论是"变换只需要可逆、不需要正交"，因此比 QuaRot/SpinQuant 的旋转类变换更灵活，代价是每层独立矩阵的存储与在线计算。

范围边界：主结果是 **W4A4**（权重 per-channel 对称、激活 per-token 对称），也覆盖 KV cache 量化（§4.1：group-wise asymmetric、group size 128）、weight-only 与 W3A3；模型只用到 LLaMA-2(7B/13B/70B) 与 LLaMA-3(8B/70B)，另在 LLaMA-3.1-8B-Instruct 上做 MT-Bench（Table 3）与附录 Table 10 的 PPL/QA。全篇是 PTQ，逐 block 顺序训练，不做端到端微调。

## 已读定位与阅读形式

- 阅读形式：arXiv e-print 的 LaTeX 源（`iclr2025_conference.tex` + `texes/*` + `tables/*` + `figures/*`，附录第五节来自 `texes/limits.tex`），另读 arXiv HTML 渲染版交叉核对编号；v1 = 2024-10-12。**无 Algorithm。**
- 定位：§2.1 量化预备（Eq.1）、§3.1 仿射变换与 Kronecker（Eq.2–4）、§3.2 接入 Transformer、§3.3 高效 kernel、§4 实验（Table 1–6、Figure 4–6）、§5 结论、附录 A–E（Eq.5–7、Table 7–15、Figure 7–18）。编号 Eq.1–7、Table 1–15、Figure 1–18。渲染差异：`weight_only.tex` 与 `decomposed_size.tex` 各在一个浮动体内塞了两个表/图（Table 5+6、Figure 5+6），HTML 缺 `S4.T5`/`S4.F5` 锚点；按源码顺序 Table 5 = weight-only、6 = ab_sep、Figure 5 = 分解尺寸、6 = online_trans。

## 关键事实

- **动机**（§1–2、Figure 1–2）：以"通道幅值降序曲线是否接近水平包络"刻画平坦度。观察：per-channel scaling（SmoothQuant/OmniQuant）把激活异常值挪到权重、权重包络反而变陡；Hadamard/正交旋转（QuaRot/QServe）整体更好，但对所有层共用同一变换、不看每层特性，且线性变换本身带来开销。最严重的量化误差出在**开头几个含 massive outlier 的 pivot token**，per-channel scaling 与 Hadamard 都压不住。
- **方法：逐层可逆仿射 + Kronecker**（§3.1、Eq.2–4）：目标 `P* = argmin_P ‖Y − Q(XP)·Q(P⁻¹Wᵀ)‖²_F`（Eq.2），`P` **只要求可逆、不要求正交**；权重侧 `P⁻¹Wᵀ` 可离线并入相邻层，激活侧 `P` 必须在线算（未分解的大矩阵方案会增加前向计算、访存和模型存储，这正是分解及融合 kernel 的动机）。为省显存/算子，把 `P` 分解为 `P = P₁ ⊗ P₂`（Eq.3），`XP` 变成两次小矩阵模式乘，按 `n₁n₂ = n`、`n₁ ≤ n₂` 最小化 `n₁+n₂` 选尺寸（如 `n=8192` 取 64×128）；理论内存最多省 `n/2`、算力最多省 `√n/2`。
- **另两组可学参数**（§3.1）：每通道缩放 `diag(c)`（可离线并入前序 LayerNorm/linear）；可学裁剪阈值 `α_w, α_a ∈ (0,1)`（sigmoid 约束）同时作用于权重、激活与 KV cache。
- **训练目标与预算**（§3.1、Eq.4）：逐 block 最小化块输出重构误差 `min_Θ ‖F_l(X) − F̂_l(X;Θ)‖²_F`（`Θ = {P, c, α_a, α_w}`），不是任务损失。校准用 WikiText-2 **128 条 × 2048 token**、batch 4、AdamW、**15 epochs**、变换 lr 5e-3 / 裁剪 lr 5e-2；LLaMA-3-8B 单卡约 26 GB / 0.9 h。求 `P⁻¹` 用 SVD 参数化 + AMP：FP16 直接求逆会让 `PP⁻¹` 非对角元到 ~1e-3 而伤训练，SVD 压到 ~1e-6（附录 Table 7 对比四种配方）。
- **接入 Transformer 与 kernel**（§3.2–3.3）：注意力用四个变换 `{P_a, P_o, P_h, P_v}`（只对 `P_a`/`P_o` 做 Kronecker，逐 head 矩阵本就小；`P_o` 与 `P_v` 融合，借鉴 QuaRot），FFN 用 `P_ug`、`P_d`；**保留 LayerNorm**（pre-norm 残差下 RMSNorm 之后的变换必须全模型共享，保留才能让每层不同）；论文同时声明 pre-quantization 变换、RoPE embedding 与 attention score 也保留 FP16。在线变换与量化都是 memory-bound，于是用 Triton 把 `Q(P₁ᵀ ×₁ X̃ ×₂ P₂)` 融成单 kernel（`P₁`/`P₂` 装进 SRAM，中间结果不落 global memory）；INT4 GEMM 用 CUTLASS、KV cache 量化用 FlashInfer。
- **主结果**（§4.2，Table 1–3）：W4A4 WikiText-2 PPL——LLaMA-3-8B FP16 6.14、SpinQuant 7.96、QuaRot 10.60、**FlatQuant(RTN) 6.98**、GPTQ 6.90；LLaMA-3-70B FP16 2.86、SpinQuant 7.58、QuaRot 55.44、**FlatQuant(RTN) 3.78**。6 个 0-shot 任务平均：LLaMA-2-70B 77.05 → 76.62（掉 0.43）、LLaMA-3-70B 79.95 → 79.01（0.94）、LLaMA-3-8B 73.23 → 71.23（2.00）。MT-Bench（LLaMA-3.1-8B-Instruct）：FP16 7.60、QuaRot 5.99、**FlatQuant 6.94**（Math 项 7.20 ≈ FP16 7.00）。论文强调 FlatQuant+RTN 已与 +GPTQ 相当但校准时间少得多。
- **推理延迟**（§4.3；RTX 3090，prefill 2048、decode 256）：batch 64 下 prefill 最高 **2.30×**、decode **1.76×**（相对 FP16）；附录 C.4 补充 batch < 16 时 decode 加速不足 1×（量化开销 > KV cache 省下的访存）。Figure 6 拆解五个在线变换：FlatQuant 端到端只多 **0.07×**，QuaRot 三个 Hadamard 多 0.26×。Figure 5 显示分解尺寸对 PPL 影响很小、对加速影响很大，`n₁ = n₂` 时最高。
- **消融与其他方案**（§4.4，Table 4–6）：LLaMA-3-8B W4A4 以纯 RTN 为基线（PPL 1266.60、QA 30.99），加可学变换 → 8.50/66.82、LT + per-channel scaling → 7.95/67.08、LT + 可学裁剪（不含 scaling）→ 7.11/70.72、三者齐全 → **6.98/71.23**。weight-only（Table 5）：FlatQuant-GPTQ W4A16/W3A16 为 6.48/7.52，对照 RTN 8.70/2.2E3、GPTQ 7.00/13.00、AWQ 7.10/12.80。Table 6：W4A4KV4 训出的同一套变换直接套到 W4/A4/KV4 分别为 6.56/6.49/6.23（全量 W4A4KV4 为 6.98），接近。
- **附录补充**（B–D）：KV-only 4-bit 时 PPL 6.20（**K 比 V 更敏感**），LLaMA-2 上 K/V 都 2-bit 时 FlatQuant 6.66/5.69 对 QuaRot 9.23/7.07；W3A3KV3 下 FlatQuant 10.82 而 QuaRot 崩到 686.54。校准集换 C4/Pile 时 PPL 只在 6.98–7.04；"先变换后裁剪"(6.98) 优于"先裁剪后变换"(7.37)、固定阈值(7.25)、不裁剪(7.95)。校准时间：LLaMA-2-7B 1.15 h（W4A4）/0.67 h（weight-only），LLaMA-3-70B 5.94 h/4.89 h。LLaMA-2-7B 上在线变换只占 FP16 前向约 2.61% 计算量。附录 E 自述：校准集最优选择仍是开放问题，未考察 MXFP4 等新数据类型。

## 不能推出的结论

- 模型只有 LLaMA-2/3（7B–70B）与 LLaMA-3.1-8B-Instruct，位宽只到 W4A4/W3A3 与 KV 2–4 bit；不能外推到 MoE/MQA/多模态或其他位宽——论文附录 E 自述未考察 MXFP4 等新数据类型。
- 变换矩阵逐层学习、绑定具体模型/bit 配置/校准数据；Table 6 的"一次训练多场景复用"只在 LLaMA-3-8B 的 W4/A4/KV4 上验证过，不是无代价迁移。
- Table 1–3 的主结果是 **W4A4**，不能当作 weight-only（Table 5）或 KV-only（Table 11–12）的证据。
- "首个在 LLaMA-3-70B 上用纯 RTN 的 W4A4 把精度损失压到 ≤1%"是论文自述，口径限定为 RTN 权重量化器 + 表 2 的六个 QA 任务（LLaMA-3-70B QA 掉 0.94、WikiText-2 PPL 升 0.92），不是所有任务都在 1% 内。
- 延迟/加速绑定 RTX 3090 与 Triton 融合 kernel、CUTLASS INT4、FlashInfer；Figure 5 的分解尺寸实验反而用朴素 PyTorch 矩阵乘测；没有这些 kernel 的后端不能直接套用倍率。
- 26 GB / 0.9 h 是论文报告的 LLaMA-3-8B 单卡校准成本；本仓库未运行 FlatQuant 训练，也未复算任何加速倍率。校准集消融只覆盖 WikiText-2/C4/Pile 三个英文文本源，不能推广到模态差异很大的校准数据。

## 本轮原文核对补充

2026-10-24 实际读取 v1 LaTeX：`texes/method.tex` 的 `eq:affinequant`、`eq:kronecker`、`eq:loss`，`texes/exp.tex` 的设置与延迟讨论，`tables/ppl.tex` 的 `tab:wikitext-ppl`，`tables/ablation_main.tex` 的 `tab:ab_main`，以及 `texes/appendix.tex` 的 SVD 与延迟小节。Table 1 的 SpinQuant 7.96/7.58 和 QuaRot 10.60/55.44 是 RTN 行，不能省略该权重量化器条件。主文 2.30×/1.76× 为 RTX 3090、batch 64、prefill 2048 和 decode 256 的论文报告；附录明确 batch <16 时解码不足 1×。未做本地复现。
