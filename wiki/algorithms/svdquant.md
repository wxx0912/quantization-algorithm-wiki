---
{"id":"algorithm-svdquant","title":"SVDQuant：低秩分支吸收异常值","type":"algorithm","tags":["ptq","low-rank","compensation","diffusion","dit","w4a4"],"aliases":["SVDQuant","Nunchaku","低秩异常值吸收"],"related":["algorithm-smoothquant","algorithm-flatquant"],"sources":["paper-svdquant","repo-svdquant","repo-nunchaku"],"updated":"2026-10-23"}
---

# SVDQuant：低秩分支吸收异常值

SVDQuant 将激活异常值通过通道缩放转移到权重，再用高精度低秩分支表示变换后权重中最显著的部分，剩余权重和激活用低比特计算。**增加一个高精度分支**是方法的一部分，不能将产物称为全部算子均为 4 bit 的网络。Nunchaku 是与其配套的推理引擎，负责融合低秩与低比特分支；算法精度和融合 kernel 的速度是不同证据。[论文 §4](https://arxiv.org/html/2411.05007v1#S4)

## 从平滑到残差量化

为避免源码中 `Linear.weight` 的转置混淆，本文统一令 $X\in\mathbb R^{t\times n}$、$W\in\mathbb R^{n\times m}$，输出 $Y=XW$。

1. 从校准轨迹收集输入通道尺度，选取对角缩放 $D$，令 $\hat X=XD^{-1}$、$\hat W=DW$，保持 $XW=\hat X\hat W$。这是本页行向量约定下的恒等式。
2. 对 $\hat W$ 做截断 SVD。若 $\hat W=U\Sigma V^T$，令 $L_1=U_{:,1:r}\Sigma_{1:r,1:r}$、$L_2=V^T_{1:r,:}$，残差 $R=\hat W-L_1L_2$。
3. 保留 $\hat XL_1L_2$ 的高精度计算，将其余计算替换为 $Q_a(\hat X)Q_w(R)$。
4. 可迭代对 $\hat W-Q_w(R)$ 做低秩分解，重新更新残差和量化参数，选择误差最小的候选。不是只分解一次原始权重后任意量化。
5. 导出低秩因子、缩放、残差量化权重及 scales；部署需同时实现两条分支并正确求和。

核心近似为

$$\hat Y=\underbrace{\hat XL_1L_2}_{\text{高精度低秩分支}}+\underbrace{Q_a(\hat X)Q_w(R)}_{\text{低比特残差分支}}.$$

因此在忽略高精度舍入误差时，输出误差仅剩 $\hat XR-Q_a(\hat X)Q_w(R)$。低秩分支使用量化前的 $\hat X$，如果改为低精度输入，就改变了这个误差分解。

## 为什么 SVD 有用，以及没有保证什么

截断 SVD 在给定 rank 下最小化 $\|\hat W-L_1L_2\|_F$，剩余范数是 $\sqrt{\sum_{i>r}\sigma_i^2}$。这能去除变换后权重的主导方向。论文 Proposition 4.2 的量化误差上界还假设残差元素服从正态分布；不能把它解释为任意网络上最终输出误差的无条件最优保证。

实际输出误差也与输入分布有关。官方低秩校准器支持按 `OutputsError` 搜索、迭代和 early stopping，说明最小权重 Frobenius 范数不是唯一实践标准。rank 越大，高精度参数和计算量越高；两因子增加 $r(n+m)$ 个参数，小 rank 不意味着额外访存自动消失。

## 校准、位宽与范围

| 条件 | 论文 v1 §5.1 |
| --- | --- |
| 模型 | FLUX.1-dev（50 steps）、FLUX.1-schnell（4）、PixArt-Σ（20）、SDXL（30）、SDXL-Turbo（4） |
| 校准数据 | COCO prompts；需采集去噪轨迹激活，不是仅对文本 token 做一次前向 |
| 8 bit | activation per-token dynamic、weight per-channel，rank 16 |
| 4 bit | W/A per-group symmetric，rank 32 |
| INT4 | group size 64、16-bit scales |
| FP4 | group size 32、FP8 scales；不能等同于任意硬件上的另一种 FP4 格式 |
| 保留精度 | FLUX adaptive normalization 线性层输入为 16 bit（W4A16）；其他模型 cross-attention K/V projections 保留 16 bit |

本文来源的 `COCO Captions 2024` 字样来自论文正文，而引用年份为 2015；本 Wiki 不据此定义一个新的数据集版本。复现时以真实下载版本、prompt 清单、scheduler、步数与 seed 为准。

固定代码的默认 diffusion 校准配置是 `num_samples: 128`，SVDQuant 配置 rank 32、`num_iters: 100`、`early_stop: true`、20 格 smoothing 搜索。这些是**当前配置文件**，不等于证明论文所有结果都使用同一预算；实际合并模型/格式/fast 配置后才是运行配置。SVD 分解和搜索需要矩阵、轨迹缓存及多次候选评估，本次未实测耗时/显存。

这里的 K/V 是图像生成模型 cross-attention 投影，不能直接写成 AR 的持久 KV-cache 量化。论文不是文本 dLLM 上的验证。

## 有条件的实验结论

论文 Table 1 的 FLUX.1-dev、50 steps、MJHQ 5K prompts：BF16 的 FID 20.3、ImageReward 0.953；INT W4A4 SVDQuant 为 19.9、0.932，LPIPS 0.254、PSNR 20.1（相对浮点输出）。FID 略低不能抹去 ImageReward 或逐样本相似度变化，也不能解释成每张图更好。

相同表格包含 sDCI 5K prompts，说明跨 prompt 集评测而非只在校准集上比较。所有数据是原文报告；本仓库未生成图片或运行指标。

## Nunchaku：为什么低秩分支需要专门 kernel

论文 §4.3、Figure 6 分析了 rank 32 分支的额外访存：低秩 down projection 读取高精度输入，up projection 写输出；独立运行会重复读取/写回与残差分支相同的数据，图中 naive 分支约增加 57% latency。这不是参数量增加 57%。

Nunchaku 将低秩 **down projection 与 activation quantization** 融合以共享输入，再将低秩 **up projection 与 4-bit compute** 融合以共享输出，减少 DRAM 往返和 kernel launch。论文报告其设置下额外开销约 5–10%；这个数字是特定实现的观察，不是算术恒等式。

论文摘要给出的 FLUX.1 12B、16 GB laptop RTX 4090 相对 4-bit weight-only baseline 的 3.0× 加速具有硬件、基线和内存约束，不能当作所有 W4A4 的速度预期。FP4 质量模拟也不证明该 GPU 原生执行相应 FP4 GEMM。目标部署需核实 packed layout、group size、scales、rank、GPU kernel、fallback 以及端到端工作负载。

当前 Nunchaku 源码也能定位到这两类融合，而不只是论文描述。固定 commit `302e0e97024ebd68688fe890e5df83731edf7b54`：

- [`GEMM_W4A4::quantize`](https://github.com/mit-han-lab/nunchaku/blob/302e0e97024ebd68688fe890e5df83731edf7b54/src/Linear.cpp#L444) 分配 packed activation、scales 和低秩中间结果，并在开启融合时调用 `quantize_w4a4_act_fuse_lora`。
- [`quantize_w4a4_fuse_lora_kernel`](https://github.com/mit-han-lab/nunchaku/blob/302e0e97024ebd68688fe890e5df83731edf7b54/src/kernels/zgemm/gemm_w4a4.cuh#L1098) 包含 `EpilogueLoraDown`；[`launch_lora`](https://github.com/mit-han-lab/nunchaku/blob/302e0e97024ebd68688fe890e5df83731edf7b54/src/kernels/zgemm/gemm_w4a4_launch_impl.cuh#L196) 将低秩 up/down 与 GEMM epilogue 组合。
- 当前 `Linear.cpp` 中 INT4 scales 按输入维度 64 分组，FP4 路径为 16，并用 FP8_E4M3 scales；这与论文 v1 的 FP4 group 32 不同。必须区分论文格式与当前导出/runtime 合约。

这是静态调用链核验，没有编译 kernel 或在目标 GPU 验证。不能由 DeepCompressor 的 fake-quant 评测替代真实 kernel 数值与速度验证。

## 官方实现与固定版本

DeepCompressor commit：`69f3473f5e1c1504bae35cc50c7858ef900a9b17`。

- [`LowRankBranch.reset_parameters`](https://github.com/mit-han-lab/deepcompressor/blob/69f3473f5e1c1504bae35cc50c7858ef900a9b17/deepcompressor/nn/patch/lowrank.py#L29)：`torch.linalg.svd(weight.double())`，保存截断因子；`forward` 为两次 Linear。
- [`QuantLowRankCalibrator`](https://github.com/mit-han-lab/deepcompressor/blob/69f3473f5e1c1504bae35cc50c7858ef900a9b17/deepcompressor/calib/lowrank.py#L20)：`_ask` 对 `self.w - self.qw` 构造低秩分支，并量化剩余权重，保留误差最佳候选。
- [`SVDQuant 配置`](https://github.com/mit-han-lab/deepcompressor/blob/69f3473f5e1c1504bae35cc50c7858ef900a9b17/examples/diffusion/configs/svdquant/__default__.yaml)：smoothing、low-rank 与 skipped modules。
- [`diffusion 默认配置`](https://github.com/mit-han-lab/deepcompressor/blob/69f3473f5e1c1504bae35cc50c7858ef900a9b17/examples/diffusion/configs/__default__.yaml)：校准规模与基础量化设置；运行仍需合并具体模型/精度配置。

## 组合顺序与 Agent 消融

已证实的机制组合是 smoothing → 低秩分解/残差量化迭代 → 两分支部署。缩放变化后低秩因子、残差、权重 scales 均需重建；rank 改变后需要重新量化残差，不能截断旧分支却保持旧残差不变。若加入旋转，也需明确两分支输入基底并重新拟合；没有证据不能直接拼成通用 Recipe。

论文声明可接入 off-the-shelf LoRA；实际组合还要核对 LoRA 的原始权重基底、缩放坐标、rank 与后端格式，不得把 LoRA 权重直接相加到未知基底的残差上。

当图像扩散 W4A4 中 smoothing 导致权重异常值突出、且高精度分支有部署路径时，可以考虑 SVDQuant。用于文本 dLLM 时需重新评估去噪 mask 轨迹、目标函数和生成任务，图像质量表无法替代文本指标。

建议保持 prompt、scheduler、steps、seed、W/A 格式不变，对比无 smoothing、仅 smoothing、smoothing+rank 16/32、一次 SVD 与迭代校准。分别记录权重残差谱、各去噪阶段输出误差、图像质量和相似度；性能另比较 unfused/fused、相同 batch/resolution 的端到端延迟和峰值显存。把误差改善归因到算法、把延迟改善归因到实际 kernel，避免以理论 FLOPs 代替速度。

来源登记见 [transforms](../../sources/transforms.md)。
