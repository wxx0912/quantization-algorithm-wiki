---
{
  "id": "algorithm-soloq",
  "title": "SoloQ：dLLM 的校准无关旋转量化",
  "type": "algorithm",
  "tags": [
    "dllm",
    "rotation",
    "equivalent-transformation",
    "format",
    "quantizer",
    "kv-cache"
  ],
  "aliases": [
    "SoloQ",
    "校准无关量化",
    "数据无关量化",
    "K-RPBH",
    "块扩散 KV cache 量化"
  ],
  "related": [
    "algorithm-dllmquant",
    "algorithm-starquant",
    "algorithm-quarot",
    "algorithm-kvquant",
    "algorithm-skvq",
    "foundation-number-formats",
    "foundation-quantization-contract",
    "guide-search-space"
  ],
  "sources": [
    "paper-soloq"
  ],
  "confidence": "source-reported",
  "evidence_scope": "source-reported",
  "updated": "2026-10-28"
}
---

# SoloQ：dLLM 的校准无关旋转量化

## 一句话理解

SoloQ 的出发点是：dLLM 的激活分布随掩码状态和去噪步不断变化，与其用校准数据去追这种变化，不如把权重和激活**一起旋进一个分布可预测的正交基**（K-RPBH 旋转），让各坐标近似固定高斯分布，从而无需校准即可量化。它同时支持分布匹配码本（SoloQ-C）和硬件原生 NVFP4（SoloQ-N），并把同一套思路扩展到块扩散模型的**提交时刻 KV cache 量化**。

## 核心机制

1. **可预测分布**（§3.2）：任意归一化向量经 Haar 随机旋转后在单位球上均匀分布，单坐标服从固定球面边缘分布，维度大时近似 `N(0,1/d)`。于是量化器可以按这个已知分布解析设计，不用采激活统计。
2. **K-RPBH 旋转**（§4.2）：`Π=(Q_k⊗I_h)·blkdiag(H_hD_1,…,H_hD_k)·P`。在随机置换 `P` + 块内随机 Hadamard（`H_hD_i`）之上，再加一个 Haar 随机正交的**跨块混合器** `Q_k`。后者直接重分配块间残留能量，修掉 RPBH 在多块维度（如 `18944=512×37`）上的块间方差波动；`k=1` 时退化为 RPBH。相比 QuIP 的 Kronecker 变换更便宜、在支持维度上比 QuaRot 更省变换操作。
3. **两条校准无关量化路径**（§4.3）：把两算子旋到同一基（权重离线、激活在线），再二选一——SoloQ-C 用分布匹配码本（激活 Lloyd–Max、权重 E8P 点阵，scale 取 L2 范数）；SoloQ-N 用硬件原生 NVFP4（E2M1 + 全局/每 16 坐标块两级缩放）。二者都直接由张量导出 scale，不用校准数据。
4. **仅缩放偏差校正**（§4.4）：`c_bc(z)=‖z‖²/⟨q(z),z⟩` 精确恢复沿原向量的分量，且能吸收进已有的 per-token 激活 scale 或 per-row 权重 scale，不增加额外载荷或比特。
5. **提交时刻 KV cache 量化**（§4.5）：只在块提交时量化持久 KV，活动去噪块保持全精度；Key 用 KIVI 式 per-channel 非对称量化，Value 走 SoloQ 激活路径。

## 基本流程

1. 选块大小 `h`（2 的幂）与跨块维数 `k`（`d=kh`），按 Eq.4 构造 K-RPBH 变换 `Π`。
2. 离线把权重旋到该基并量化；推理时在线旋转激活。
3. 依部署目标选 SoloQ-C（码本）或 SoloQ-N（NVFP4），并对量化结果应用仅缩放偏差校正。
4. 若是块扩散模型，在块提交时对持久 KV 做量化（Key per-channel、Value 走激活路径），活动块保持全精度。

## 适用条件

- 面向 dLLM 的 **W4A4** 权重+激活量化，且希望**免校准**、免预采激活统计；论文验证 LLaDA-8B-Base、LLaDA-1.5、Dream-7B 与块扩散的 Fast-dLLM v2、Nemotron-Labs-Diffusion。
- SoloQ-N 需要目标后端支持 NVFP4（论文用 RTX Pro 6000）；SoloQ-C 用 8-bit Tensor Core kernel（论文用 H200）。旋转维度需支持 `d=kh`（`h` 为 2 的幂）。
- 改动对象是"旋转基 + 量化器 + 可能的 KV cache"，不改变 dLLM 的解码流程；KV cache 部分只适用于**有持久 cache 的块扩散**模型。

## 结果边界

- 论文报告（Table 1，W4A4，H200，9 项平均）：LLaDA-Base-8B 上 SoloQ-C **57.74** / SoloQ-N **57.60**，均校准无关，高于 STaR-Quant 57.07、DLLMQuant++ 54.29；LLaDA-1.5-8B SoloQ-N **68.48**；Dream-7B SoloQ-N **62.53**。
- 块扩散（Table 2，Fast-dLLM v2）：W4A4 下 RTN/AWQ 崩塌、QuaRot 因维度不支持而 N/A，SoloQ-C/N 分别 **57.00 / 56.77**；W4A4KV4 下 SoloQ-C **56.19**。
- KV cache（Table 5，Fast-dLLM v2，GSM8K/IFEval 平均）：2-bit KV（约 3 BPE）、权重激活 FP16 时 SoloQ-C 53.31、SoloQ-N 53.53，优于 KIVI、OScar、TurboQuant。
- 效率：相对 BF16，**SoloQ-N**（RTX Pro 6000 上的原生 NVFP4 kernel）峰值显存降低 2.31×–2.61×、端到端加速 1.94×–2.24×（Nemotron 用固定 75 次前向比较得 2.05×）；**SoloQ-C** 走 H200 的 8-bit Tensor Core kernel。另有一个独立的 NVFP4 FPGA 原型加速器（附录 E，非 GPU 测量），在 LLaDA-Base-8B 上相对 BF16 报告 1.94× 端到端加速、2.70× 能效。以上数字各自绑定对应硬件与 kernel，非通用收益。
- 只覆盖 W4A4/W4A4KV4；`h`、`Q_k`、码本/点阵、NVFP4 块大小等为设计设定。未复现、未在目标后端核验。

## Agent 如何使用

处理"dLLM 能不能免校准量化""激活分布随去噪步漂移怎么办""块扩散 KV cache 怎么压"时先读本页，再按需展开：

| 需要确认的内容 | 继续读取 |
| --- | --- |
| K-RPBH 公式、Eq.3–12、Prop.1、Table 1–5 | [`paper-soloq`](../../sources/papers/paper-soloq.md) |
| 依赖校准的 dLLM 量化对照 | [DLLMQuant](dllmquant.md)、[STaR-Quant](starquant.md)、[FAIR-Calib](faircalib.md) |
| 旋转量化的通用原理 | [QuaRot](quarot.md) |
| KV cache 量化的分布与维度选择 | [KVQuant](kvquant.md)、[SKVQ](skvq.md) |
| 数值格式与契约 | [数值格式](../foundations/number-formats.md)、[量化契约](../foundations/quantization-contract.md) |

回答时要区分"旋转基（K-RPBH）""量化器选择（码本 vs NVFP4）""偏差校正""KV cache 提交时刻量化"四部分，并把免校准的事实（不依赖校准集）与论文的精度/效率报告值分开陈述。

## 详细来源

- [SoloQ 原论文来源页](../../sources/papers/paper-soloq.md)
