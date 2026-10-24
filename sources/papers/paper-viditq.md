---
{
  "id": "paper-viditq",
  "title": "ViDiT-Q: Efficient and Accurate Quantization of Diffusion Transformers for Image and Video Generation",
  "category": "paper",
  "url": "https://arxiv.org/abs/2406.02540",
  "version": "v2（2024-06-30）",
  "version_kind": "paper-version",
  "retrieved_at": "2026-10-24",
  "read_form": [
    "latex"
  ],
  "read_locator": "摘要、§1（Figure 1–2）、§3（Figure 3–6）、§3.1、§3.2.1、§3.2.2、§3.2.3（Eq.1）、§3.3.1、§3.3.2、§4.1（设置）、§4.2（Table 1–3）、§4.3（Table 4、Figure 8）、§5 结论与 Limitations、附录 A（Figure 9–10、资源节省估计）、附录 C（Figure 11–19 的定性结果与讨论）",
  "topic": "mixed-architecture",
  "requires_external_verification": true,
  "excerpt_sha256": null
}
---
## 这份来源是什么

ViDiT-Q 原始论文（arXiv 2406.02540，v2；清华 + 无问芯穹）。它把"扩散模型量化"从 UNet 时代推到 **DiT（diffusion transformer）**，核心观察是 DiT 的激活在**四个维度**上方差很大——token、CFG 条件/无条件两路、时间步、输入通道——而现有方法用固定且粗粒度的量化参数。对策是三件事：per-token 量化参数、激活量化参数在线动态计算、**时间步感知的通道平衡**；再叠加"指标解耦的混合精度"（按层类型分配敏感度指标）。

它是 `guide-architecture-transfer`、`guide-deployment-evidence` 与 `case-paper-evidence` 的依据：给出"把 LLM/UNet 的量化经验迁到 DiT/dLLM"时需要补的证据类型，也给出一个教训——缺少 kernel 的低比特配置只能用代理口径报延迟。

范围边界：模型是视频 DiT（STDiT/OpenSORA、Latte）与图像 DiT（PixArt-α、PixArt-Σ）；主设置 W8A8、W6A6-MP、W4A8-MP；量化方案是 **min-max**，激活用 per-token 动态量化、权重静态，只量化**线性层**（t/y embedding 与 final layer 保持 FP16，attention 计算不量化）。权重不是逐层学习的，α 与混精方案离线由校准数据确定。

## 已读定位与阅读形式

- 阅读形式：arXiv e-print 的 LaTeX 源（单文件 `ms.tex` + `figs/`，正文与附录同文件）；v2 = 2024-06-30。同一版本另有一份以"源码包"口径登记的来源 `source-viditq`。
- 定位：§3 方法（3.1 挑战、3.2.1 token-wise、3.2.2 dynamic、3.2.3 通道平衡 Eq.1、3.3 指标解耦混精）、§4 实验（Table 1–4、Figure 8）、§5 结论与 Limitations、附录 A（Figure 9–10、资源节省估计）、C（Figure 11–19）。编号：Figure 1–19（正文 8、附录 11）、Table 1–4（全在正文）、Eq.1，**无 Algorithm**。注意 `ms.tex` 活动的 `subfigure` 有 15 个（含注释 20 个）但**不推进 figure 计数器**；Figure 7/8 的浮动体内嵌的是**无编号表**，引用时不要写成"Table X"。

## 关键事实

- **失败现象**（§1、Figure 2）：把现有扩散量化方法直接套到 DiT 会明显退化。论文自身对失败配置口径不一致——§3.1、§4.2、附录 C.2 写 W4A8/W6A6 生成空白或纯噪声，§1 写 **W4A16** 生成单色空白帧；本页按各处原文分别标注。原因归于"数据方差存在于多个层级，而现有方法用固定且粗粒度的量化参数"。
- **四类方差**（§3.1、Figure 4；示例层 `blocks.26.ffn.fc1`）：① token-wise；② CFG-wise（条件/无条件两次前向的激活范围不同）；③ timestep-wise；④ input channel-wise（权重与激活都在少数输入通道上偏大，导致多数值被量化成 0；">100×" 出自 §3.2.3）。
- **三个对策**（§3.2）：① **per-token 量化参数**（线性层不需 token 间求和，参数量约为激活的 **1/1000**）；② **动态量化**（激活参数在线 min-max 计算，消掉 CFG-wise 与 timestep-wise 方差；附录 A.3 的 Nsight profiling 显示 minmax 的 `reduce_kernel` 约 **100 µs**、线性层约 **10 ms**）；③ **时间步感知的通道平衡**——掩码 `s_i = max(|X_i|)^α / max(|W_i|)^{1−α}`（Eq.1，沿用 SmoothQuant/AWQ 思路），但单一 `α` 敏感且不够好（后期去噪步异常值更突出），于是把扩散过程等分前后两半分别给 `α₁ < α`、`α₂ > α`，靠分布先验在 **20 次 PTQ（<1 GPU 小时）**内调好（附录 A.3 写 10–30 次）。
- **指标解耦的混合精度**（§3.3）：先指出**瓶颈层**现象（W4A16 生成空白图，只跳过 STDiT 的 `blocks.27.ffn.fc2` 一层就能生成可读内容），再指出纯 MSE 不能代表生成质量（Figure 6）。做法是把层的影响"解耦"成 **cross-attention ≈ 内容变化、spatial attention 与 FFN ≈ 视觉质量、temporal attention ≈ 时间一致性**，对不同层类型用对应指标做敏感度分析，并把时间步等分四段分别量化找最敏感区间，给高敏感部分更高位宽。具体：W6A6-MP 把 `blocks` 6 与 26 的 FFN 保持 FP16（占延迟 <1%）、其余 W6A6；W4A8-MP 把所有 FFN（**15% 的层**）设为 W8A8、其中前 1/4 的时间步也设为 W8A8，其余 W4A8。
- **评测设置**（§4.1）：视频侧 STDiT(OpenSORA) 与 Latte，VBench 取 8 个维度；UCF-101 用 101 条 prompt + open-sora 10 条 prompt，看 CLIPSIM/CLIP-Temp、DOVER 的 VQA-Aesthetic/Technical、Flow-score、Temporal Flickering、FVD 与 **FVD-FP16**（以 FP16 生成的视频作 ground truth）；Latte 用 20 步 DDIM + CFG 7.0，STDiT 用 100 步 DDIM + CFG 4.0。图像侧 PixArt-α/Σ 在 COCO 前 1024 条 prompt 上看 FID、CLIPScore、ImageReward，20 步 DPM-solver + CFG 4.5。
- **主结果**（§4.2，Table 1–2）：VBench 8 维上 **ViDiT-Q W8A8 与 FP16 基本持平**，而 Naive W8A8 在 Imaging（63.68→56.52）、Aesthetic（57.12→53.64）、Overall Consist.（26.21→25.42）等维度明显掉分，Dynamic Degree 反而虚高（56.94→**69.44**，论文指出这是"生成失败导致的假高分"）。UCF-101 上 STDiT：FP16 FVD 136.87，Naive W8A8 154.92（FVD-FP16 50.72），ViDiT-Q W8A8 141.13（15.52），**Naive W4A8 直接崩到 544.34/637.02、VQA-Aesthetic 0.16**，而 ViDiT-Q-MP W4A8 做到 **129.10/60.13**、VQA 33.98/47.65。论文提示 101 个视频时 FVD 波动大、FVD-FP16 与视觉感知相关性更好。
- **与其他方法的对比**（Table 3；baseline 是作者自行适配到 DiT 的 PTQ4DM/Q-Diffusion/ZeroQuant/SmoothQuant 等）：W8A8 下 ViDiT-Q 全面优于它们并接近 FP16；W4A8 下**只有 SmoothQuant 能避免生成失败**，ViDiT-Q 的混精版本还能把 W4A8 做到"可比基线 W8A8"的水平。
- **消融**（Table 4 的浮动体在 §4.3、讨论在附录 C；W4A8）：无技术 → +动态量化 → +通道平衡 → +时间步感知，VQA-Aesthetic / VQA-Technical 从 0.178/0.086（生成失败）经 32.2/11.0、32.0/22.8 升到 **51.8/23.2**；论文明确指出即便全部加上仍有明显退化，这正是需要混精的理由。Table 4 未标注数据集，按数值与 Table 3 的 open-sora 行吻合**推断**为 open-sora prompt set（推断，原文未写明）。
- **硬件资源与成本**（§4.3、Figure 8、附录 A.2–A.4）：显存峰值降低 **2×（W8A8）/ 2.42×（W4A8-MP）/ 2.67×（W6A6-MP）**；因为**没有 W6A6/W4A8（也没有动态量化）的 GPU kernel**，延迟用 INT8 kernel 估计——单层约 2×、端到端 **1.47×**（W8A8），BitOps 理论节省 4.00×/7.11×/5.05×。附录 A.2：FlashAttention 下 attention 只占 **14.3%** 延迟且显存被压掉，故只量化线性层，t/y embedding 与 final layer 因通道小（<1/1000 延迟）保持 FP16。附录 A.3：α 调优需 10–30 次 PTQ、<1 GPU 小时，混精的敏感性分析约 **10 GPU 小时**。注意主文把 2.67× 记在 W6A6-MP 名下、附录另有一句归属口径不一致（原文如此）。
- **Limitations**（§5）：时间步级混精需要额外的系统级优化；敏感性分析的效率与精度待改进；要真正吃满 4-bit 权重的加速潜力还需更低的**激活**位宽（当前 W4A8 的 8-bit 激活是瓶颈）。

## 不能推出的结论

- **W6A6/W4A8 的延迟不是实测**：论文明确用 W8A8 的 1.47× 代表它们、BitOps 是理论值；不能把 1.47× 当作低比特真实加速，也不能据 BitOps 推算端到端性能。
- 结果只在视频/图像 DiT（STDiT、Latte、PixArt）上验证，**不能推断文本 dLLM（如 LLaDA）的迁移**，也不能推断 UNet 扩散或自回归模型的表现。
- 显存/延迟绑定具体硬件与实现：附录 A.1 的 profiling 硬件是单张 **RTX 4080**、CUDA 12.1、batch 1，延迟用基于 Cutlass 的静态 INT8 kernel 估计；§1 出现的 A100 只用于描述 OpenSORA 基线开销。不能外推为部署侧的服务吞吐/延迟。
- 混精方案是**按具体模型层名定制**的（STDiT 的 blocks 6/26 FFN、所有 FFN、前 1/4 时间步），换模型要重跑敏感性分析（约 10 GPU 小时）；不能当通用默认值。
- 读表要与论文的口径说明一起读：FVD 在 101 个视频上波动大、Dynamic Degree 的高分来自生成失败、Flow-score 过高过低都是退化征兆；单独摘一个指标排名会得出相反结论。
- 本仓库未运行任何 DiT 量化/生成实验；本页数值全部来自论文表格，未复算图像/视频指标。
