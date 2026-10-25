---
{
  "id": "algorithm-svdquant",
  "title": "SVDQuant：低秩分支吸收异常值",
  "type": "algorithm",
  "tags": [
    "ptq",
    "low-rank",
    "compensation",
    "diffusion",
    "dit",
    "w4a4"
  ],
  "aliases": [
    "SVDQuant",
    "Nunchaku",
    "低秩异常值吸收"
  ],
  "related": [
    "algorithm-smoothquant",
    "algorithm-flatquant"
  ],
  "sources": [
    "paper-svdquant",
    "repo-svdquant",
    "repo-nunchaku"
  ],
  "confidence": "source-reported",
  "evidence_scope": "source-reported",
  "updated": "2026-10-24"
}
---

# SVDQuant：低秩分支吸收异常值

SVDQuant 面向图像扩散模型的低比特量化：先把激活异常值通过通道缩放转移到权重，再用高精度低秩分支吸收权重中的主导成分，剩余部分采用低比特计算。Nunchaku 是配套推理引擎，用融合 kernel 降低额外分支的访存开销。

## 核心机制

令线性层为 $Y=XW$，通道缩放保持 $XW=(XD^{-1})(DW)=\hat X\hat W$。将变换后权重分解为 $\hat W=L_1L_2+R$，得到

$$\hat Y=\hat XL_1L_2+Q_a(\hat X)Q_w(R).$$

第一项是高精度低秩分支，第二项是低比特残差分支。低秩分支使用量化前的输入，因此产物不是所有计算都变成 4 bit。截断 SVD 用于提取主导权重方向；它最小化给定 rank 下的权重残差范数，不保证最终任务误差全局最优。

## 基本流程

1. 用校准 prompts 采集扩散去噪过程的激活，确定通道缩放。
2. 对变换后的权重做低秩分解，量化残差；论文还采用迭代更新低秩分支和残差、选取误差较小候选的做法。
3. 保存缩放、低秩因子、残差量化权重与 scales。
4. 通过同时支持两条分支的后端执行并求和；Nunchaku 将低秩 down projection 与激活量化融合、up projection 与低比特计算融合。

## 适用条件与结果范围

论文 v1 覆盖 FLUX.1、PixArt-Σ、SDXL/SDXL-Turbo。4-bit 实验使用 W/A per-group symmetric、rank 32；INT4 为 group 64 加 16-bit scales，v1 的 FP4 为 group 32 加 FP8 scales（v3 camera-ready 改为 NVFP4 group 16 + FP8 scales，并对残差权重用 GPTQ）。部分层保留高精度：FLUX 自适应归一化线性层输入采用 16 bit，其他模型的 cross-attention K/V 投影保留 16 bit。

论文结果支持这些图像扩散模型的低比特质量和特定 Nunchaku 实现的效率收益，不能外推为文本 dLLM 或任意后端的验证。论文 v1 的 FP4 格式（group 32）与当前 Nunchaku 实现的 FP4 group 16 不同，部署必须采用配套导出/runtime 合约。

低秩分支虽然参数少，独立执行仍有访存成本；论文融合开销和加速数字都有硬件及基线条件。模型大小、运行显存和端到端延迟应分别理解，fake-quant 质量不能证明真实低比特 kernel 已运行。

## Agent 如何使用

当图像扩散 W4A4 的 smoothing 引起权重异常值、且部署端支持低秩与低比特联合执行时，可将 SVDQuant 作为候选。

- 误差公式、SVD 假设、格式、实验数字与性能口径：读[论文来源](../../sources/papers/paper-svdquant.md)。
- 低秩校准、搜索目标和默认预算：读[DeepCompressor 来源](../../sources/repos/repo-svdquant.md)。
- 打包格式、scale、分组和融合路径：读[Nunchaku 来源](../../sources/repos/repo-nunchaku.md)，再检查目标部署配置。
- 工程上改变缩放或 rank 后需要重建相应低秩分支与残差；接入 LoRA 或旋转须核对共同输入基底。这是工程推导，不能直接视为论文验证过的通用组合。
