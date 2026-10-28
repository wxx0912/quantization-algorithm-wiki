---
{
  "id": "paper-soloq",
  "title": "SoloQ: Calibration-Free Quantization for Diffusion Language Models",
  "category": "paper",
  "url": "https://arxiv.org/abs/2610.07121",
  "version": "v1（2026-10-05）",
  "version_kind": "paper-version",
  "retrieved_at": "2026-10-28",
  "read_form": [
    "html"
  ],
  "read_locator": "摘要、§1 引言、§3.1–3.2（Eq.1–3）、§4.1–4.5（Eq.4–12、Prop.1）、§5.1–5.3（Table 1–2）、§5.4 效率（Figure 4）、附录 E FPGA 原型（Table 12）、§6.1–6.4 分析（Table 3–5）、§7 结论",
  "topic": "transforms",
  "excerpt_sha256": null
}
---
## 这份来源是什么

SoloQ 论文（arXiv 2610.07121，v1），提出**无需校准**的扩散大语言模型量化框架：与其跟踪随掩码状态与去噪步变化的激活分布，不如把权重和激活一起旋进一个"分布可预测"的正交基，从而做数据无关量化。核心是 K-RPBH 旋转（Kronecker 扩展的随机置换块 Hadamard）、一个仅缩放（scale-only）的偏差校正，以及块扩散模型的**提交时刻 KV cache 量化**。在 LLaDA/Dream（全序列）与 Fast-dLLM v2/Nemotron-Labs-Diffusion（块扩散）上评估。

它是 `algorithm-soloq` 的一手依据。与本库其它来源的分工：`paper-quantdllm`/`paper-dllmquant`/`paper-starquant`/`paper-faircalib` 都是**依赖校准数据**的 dLLM 量化；本篇是校准无关路线，并把 KV cache 量化扩展到块扩散模型。旋转原理可对照 `paper-quarot`。

范围边界：W4A4 权重+激活（并有 W4A4KV4 变体）；两种量化路径——分布匹配码本（SoloQ-C）与硬件原生 NVFP4（SoloQ-N）；不含微调。

## 已读定位与阅读形式

- 阅读形式：arXiv HTML 渲染版（`arxiv.org/html/2610.07121v1`），正文与表格按 HTML 结构解析；未读 LaTeX 源码或 PDF。
- 定位：§3.2 随机旋转下的可预测边缘分布（Eq.3）、§4.2 K-RPBH（Eq.4、Prop.1）、§4.3 校准无关权重-激活量化（Eq.7–11）、§4.4 仅缩放偏差校正（Eq.12）、§4.5 块扩散 KV cache（提交时刻）、§5.1 设置、§5.2 Table 1、§5.3 Table 2、§5.4 效率（Figure 4）、附录 E FPGA 原型（Table 12）、§6.1–6.4 分析与消融（Table 3–5）、§7 结论。附录 A–G 未逐条读。

## 关键事实

- **原理**（§3.2，Eq.3）：对非零向量 `x`，归一化 `x̃=x/‖x‖₂`；在 Haar 随机旋转 `R` 下 `z=Rx̃` 均匀分布于单位球，与 `x` 方向无关，每个坐标服从固定球面边缘 `f_d(u)=Γ(d/2)/(√π Γ((d−1)/2))·(1−u²)^((d−3)/2)`，`u∈[−1,1]`，`d` 大时近似 `N(0,1/d)`。因此随机旋转能把任意表示映射到可预测的类高斯分布，支撑解析量化器设计。
- **K-RPBH**（§4.2，Eq.4）：`Π=(Q_k⊗I_h)·blkdiag(H_h D_1,…,H_h D_k)·P`，`d=kh`、`h` 为 2 的幂；`H_h` 是归一化 Walsh–Hadamard 矩阵、`D_i` 是对角 Rademacher 符号、`P` 是随机置换、`Q_k` 是 Haar 随机正交的**跨块混合器**。`k=1` 时退化为 RPBH。Prop.1 给出跨块方差平均 `E_{Q_k}[Var_D(ẑ_ℓ^{(r)}|P,Q_k)]=1/d`，修补 RPBH 残留的块间能量不平衡（Dream FFN 维度 `18944=512×37` 这种多块情形尤甚）。作者称其比 QuIP 的 Kronecker 变换便宜、在支持维度上比 QuaRot 需要更少的变换操作。
- **两种校准无关量化路径**（§4.3，Eq.7–11）：先把两算子旋到同一 K-RPBH 基（`W'=WΠᵀ`、`x'=Πx`；权重离线旋转、激活在线旋转），再二选一——**SoloQ-C** 分布匹配码本：激活用 Lloyd–Max 标量量化器、权重用 E8P 点阵量化器（带残差级），scale 取向量/行 L2 范数；**SoloQ-N** 硬件原生 NVFP4：固定 E2M1 元素格式，全局幅度 + 每 16 坐标块幅度（E4M3 类 scale）两级缩放。两种路径的 scale 都直接由张量导出，无需校准数据或预采激活统计。
- **仅缩放偏差校正**（§4.4，Eq.12）：`c_bc(z)=‖z‖₂²/⟨q(z),z⟩`，使 `⟨c_bc(z)q(z),z⟩=‖z‖₂²`，精确恢复沿原向量的分量；它可吸收进已有的 per-token 激活 scale 或 per-row 权重 scale，**不需要额外载荷或额外量化比特**。
- **块扩散 KV cache 量化**（§4.5）：只在**块被提交时**量化 KV cache，正在去噪的活动块保持全精度，避免对同一状态反复量化；Key 用 KIVI 式 per-channel 非对称量化，Value 走 SoloQ 激活路径（K-RPBH + Lloyd–Max 或 NVFP4）。
- **设置**（§5.1）：准确率评测在 NVIDIA H200；全序列模型 LLaDA-8B-Base、LLaDA-1.5、Dream-7B；块扩散模型 Fast-dLLM v2、Nemotron-Labs-Diffusion；保留各模型官方的评测栈而非统一评测器；基线含 RTN、AWQ、QuaRot(+GPTQ)、DLLMQuant、STaR-Quant（全序列），块扩散用 RTN/AWQ/QuaRot。
- **主结果**（Table 1，W4A4，9 项平均）：LLaDA-Base-8B FP16 58.95、RTN 44.23、AWQ 48.09、QuaRot+GPTQ 51.03、DLLMQuant+ 49.26、DLLMQuant++ 54.29、STaR-Quant 57.07、**SoloQ-C 57.74 / SoloQ-N 57.60（均校准无关）**；LLaDA-1.5-8B 上 SoloQ-N **68.48**（论文称比 DLLMQuant++ 高 4.17 分）；Dream-7B 上 SoloQ-N **62.53**（DLLMQuant++ 61.90）。
- **块扩散**（Table 2）：Fast-dLLM v2-7B FP16 60.35；W4A4 下 RTN/AWQ 崩塌（均值约 6.6）、QuaRot 因 Fast-dLLM v2 的维度不支持而为 N/A，而 SoloQ-C **57.00**、SoloQ-N **56.77**；W4A4KV4 下 SoloQ-C **56.19**。论文称在 W4A4 与 W4A4KV4 下都能维持较强精度。
- **效率**（§5.4，Figure 4 及正文）：SoloQ-N 峰值显存降低 **2.31×–2.61×**、端到端加速 **1.94×–2.24×**（相对 BF16）；Nemotron 用固定 75 次前向序列比较得 2.05×。SoloQ-C 用 H200 的 8-bit Tensor Core kernel，SoloQ-N 用 RTX Pro 6000 的原生 NVFP4 kernel。
- **FPGA 原型**（附录 E、Table 12，非 GPU 测量）：在 Zynq ZCU104 上，NVFP4 相对 BF16 在 LLaDA-Base-8B 报告 **1.94×** 端到端加速（0.13→0.252 tok/s）、**2.70×** 能效（15.13→40.90 tokens/kJ）。
- **旋转消融**（Table 3）：SoloQ-N 下 K-RPBH 保持精度同时把单次激活旋转延迟（H200）从 Haar 的 98.135ms 降到 5.779ms（Fast-dLLM v2）、从 0.559ms 降到 0.048ms（LLaDA-1.5）；Table 4 组件消融（Nemotron）KL 由 block-RHT 的 0.11396 降到 K-RPBH 的 0.00413、能量不平衡由 67.89% 降到 22.12%。
- **KV cache 对比**（§6.4，Table 5，Fast-dLLM v2，GSM8K/IFEval 平均）：2-bit KV（约 3 比特/项、含 scale 与元数据开销）、权重与激活保持 FP16 时，SoloQ-C 53.31、SoloQ-N 53.53，优于 KIVI、OScar、TurboQuant。
- **代码**：论文声明代码公开于 `https://github.com/Intelligent-Computing-Lab-Panda/SoloQ`。

## 不能推出的结论

- 本页只读正文与所列表格，未复现实验、未运行；准确率、延迟与显存均为论文报告值。
- 结论限于 LLaDA-8B-Base/LLaDA-1.5/Dream-7B（全序列）与 Fast-dLLM v2/Nemotron-Labs-Diffusion（块扩散）；不能外推到 AR LLM、图像/视频 DiT 或其它 dLLM。
- 报告的是 **W4A4 与 W4A4KV4**；不能据此认为更低比特（如 2-bit 权重+激活）或其它格式同样有效。
- 效率数字绑定到具体 kernel 与硬件（SoloQ-C 的 8-bit Tensor Core / H200、SoloQ-N 的原生 NVFP4 / RTX Pro 6000），不等于任意后端的端到端收益。
- K-RPBH 的块大小 `h`、`Q_k` 抽样、码本/点阵选择、NVFP4 块大小 16、偏差校正的适用条件等为设计设定；换维度或后端需重确。
- 论文声明代码在 `https://github.com/Intelligent-Computing-Lab-Panda/SoloQ`，但本轮核验该地址返回 404，未能确认仓库内容；本库无对应 repo 来源页。
