---
{"id":"algorithm-flatquant","title":"FlatQuant：可学习可逆变换与块重构","type":"algorithm","tags":["ptq","learned-transform","reconstruction","clipping","w4a4","kv-cache"],"aliases":["FlatQuant","flatness","平坦化"],"related":["algorithm-spinquant","algorithm-omniquant","algorithm-gptq"],"sources":["paper-flatquant","repo-flatquant"],"updated":"2026-10-23"}
---

# FlatQuant：可学习可逆变换与块重构

FlatQuant 在量化之前使用按层学习的可逆变换，让权重与激活更易量化，同时联合学习通道缩放及裁剪范围。与仅学习正交旋转的 SpinQuant 不同，FlatQuant 的变换不要求 $P^{-1}=P^T$；以 Kronecker 结构控制在线变换成本，并用块输出重构优化这些参数。[论文 §3](https://arxiv.org/html/2410.09426v1#S3)

## 计算流程与变量

设 $X\in\mathbb R^{t\times n}$、$W\in\mathbb R^{m\times n}$，原层输出为 $Y=XW^T$。Eq.2 写为

$$\min_{P\ \text{可逆}}\|Y-Q_a(XP)Q_w(P^{-1}W^T)\|_F^2.$$

这说明如何变换单层；实际训练目标是整个 Transformer block 的输出误差，而非分别最小化每个线性层误差。

1. 为 attention 和 FFN 中不同量化边界插入可逆变换。较大矩阵采用 $P=P_1\otimes P_2$，$n=n_1n_2$；head 内小矩阵无需同样分解。
2. 将 $X$ reshape 为 token × $n_1$ × $n_2$，以两个较小矩阵乘法完成 $XP$，避免显式生成 $n\times n$ 矩阵。代价从大矩阵乘法降为两个模式乘法；具体转置取决于行向量和 reshape 约定，需遵守实现。
3. 在适用位置加入逐通道缩放 $c$，并对 W/A/KV 的裁剪阈值使用 sigmoid 约束的可学习系数。
4. 固定原权重，对每个 block 优化 $\Theta=\{P,c,\alpha_a,\alpha_w\}$，目标为 Eq.4：$\|\mathcal F_l(X)-\hat{\mathcal F}_l(X;\Theta)\|_F^2$。前者为浮点教师块，后者包含 fake quant。
5. 固定变换，把逆变换和允许融合的缩放吸收到权重中，产生最终权重参数；在线路径仍须变换激活并量化，再运行低比特 GEMM。

SVD 参数化在这里用于表示**可逆变换矩阵**，不能与 SVDQuant 增加低秩输出分支混为一谈。官方 `SVDSingleTransMatrix` 和 `SVDDecomposeTransMatrix` 学习正交因子及对角奇异值，逆变换使用其倒数；若奇异值接近零，数学等价性不等于有限精度稳定性。

## Transformer 接入与前提

§3.2 的 attention 变换包括 Q/K/V 投影输入 $P_a$、输出投影输入 $P_o$、head 内 K 与 V 变换 $P_h,P_v$；FFN 包括 up/gate 输入 $P_{ug}$ 与 down 输入 $P_d$。一般可逆矩阵作用于 Q/K 时必须使用相配的逆转置，不能照搬正交旋转的“同乘一个矩阵”。RoPE、非线性与 residual 的位置都影响可融合性。

FlatQuant 保留归一化后的在线变换，允许不同层使用不同矩阵。归一化、预量化变换、RoPE 与 attention scores 在文中保留 FP16。缩放只有在相邻算子允许等价吸收时才可融合；论文举例把 down 输入缩放吸收到 up-projection 权重，不是把任意矩阵穿过 SiLU。

## 数据、量化与资源

论文 §4.1：WikiText-2 随机 128 条、每条 2048 tokens，batch size 4、15 epochs；AdamW，变换初始学习率 $5\times10^{-3}$、clipping 学习率 $5\times10^{-2}$，余弦衰减。LLaMA-3 8B 单 GPU 校准约 26 GB、0.9 h 是论文报告，不是本项目测量。

权重采用 per-channel symmetric，激活 per-token symmetric；KV 使用 group-wise asymmetric，group size 128。默认 RTN，另报告训练后 GPTQ，GPTQ 使用相同校准集。论文覆盖 W4A4 和低比特 KV 等设置，实验讨论其他位宽不代表所有后端支持。

已读当前训练实现使用 `fp_inps` 构造教师与量化块输入，逐层更新浮点输入缓存；不能仅因“sequential”就描述为始终使用前层量化输出的非对称重构。实际复现应核对缓存构造。

## 结果与实现证据

论文 Table 1：LLaMA-3 8B 的 WikiText-2 PPL，FP16 为 6.14；W4A4 下 FlatQuant RTN 为 6.98，GPTQ 为 6.90；LLaMA-3 70B 分别为 2.86、3.78、3.77。表中共同条件见 §4.1；这是原文报告，不能跨表与其他论文不同 calibration/任务套件直接排名。

固定源码 `9d88ffcb7d2c6bda59fb5c44dad36adc101aadb1`：

- [`trans_utils.py: SVDSingleTransMatrix / SVDDecomposeTransMatrix`](https://github.com/ruikangliu/FlatQuant/blob/9d88ffcb7d2c6bda59fb5c44dad36adc101aadb1/flatquant/trans_utils.py#L8)：正交因子、可学习对角线、逆变换及 eval 参数固化。
- [`train_utils.py: cali_flat_quant`](https://github.com/ruikangliu/FlatQuant/blob/9d88ffcb7d2c6bda59fb5c44dad36adc101aadb1/flatquant/train_utils.py#L14)：块级缓存与 MSE 训练；注意实际代码中对 loss 的归一化操作。
- [`flat_linear.py: FlatQuantizedLinear`](https://github.com/ruikangliu/FlatQuant/blob/9d88ffcb7d2c6bda59fb5c44dad36adc101aadb1/flatquant/flat_linear.py#L8)：变换、裁剪、量化和 `reparameterize`；该层中 `F.linear` 的 fake-quant 路径不能证明整数 GEMM。
- [`REALQUANT.md`](https://github.com/ruikangliu/FlatQuant/blob/9d88ffcb7d2c6bda59fb5c44dad36adc101aadb1/REALQUANT.md)：当前仓库另有真实量化权重和 kernel 路径。README 明确 vLLM 接入为 fake quant；不能把 vLLM 支持直接写成真 W4A4 加速。

源码包含晚于论文 v1 的模型支持与真实量化更新，两类证据分开：论文原始 LLaMA 实验不能证明当前 DeepSeek 或其他模型支持的精度；README 声明也不能替代执行验证。本 Wiki 仅静态核验，没有 GPU 运行。

## 组合、重校准与部署

有证据的组合是通道缩放 + 可逆变换 + 学习 clipping 的块重构，固定后再 RTN 或 GPTQ。先完成这些优化，再在最终变换图上计算 GPTQ 二阶统计和最终量化范围。变换、clipping 或量化位宽改变后，先前的 scale、重构损失和敏感度结论都需重新验证；重新训练 P 后不能保留旧整数权重。

论文 §3.3 将 Kronecker 变换和量化融合，减少中间写回。若目标后端只接受普通 AWQ/GPTQ packed weights 而无在线变换，算法产物不能直接落地。推理收益必须覆盖变换、量化、GEMM、KV 访问和完整请求，而非只报 GEMM 时间。

## Agent 选择与消融

适用于仅旋转仍无法控制 W/A 误差、能够接受块级优化且有在线变换部署路径的候选场景。预算受限时先计算一次所需缓存与块优化成本；不要默认比 GPTQ 更便宜。

固定数据、位宽、最终量化器，分别移除 clipping、通道缩放、可学习变换，比较块输出误差及最终 PPL；另比较 RTN 与训练后 GPTQ，检验剩余权重误差是否值得额外校准。验证浮点重参数化等价性、逆变换数值稳定性，以及 unfused/fused kernel 数值和端到端时间。AR 结果迁移到 dLLM 需要覆盖 mask 与去噪阶段，并验证输入分布、教师目标和全程生成质量。

来源登记见 [transforms](../../sources/transforms.md)。
