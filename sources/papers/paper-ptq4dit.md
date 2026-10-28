---
{
  "id": "paper-ptq4dit",
  "title": "PTQ4DiT: Post-training Quantization for Diffusion Transformers",
  "category": "paper",
  "url": "https://arxiv.org/abs/2405.16005",
  "version": "v1（2024-05-25）",
  "version_kind": "paper-version",
  "retrieved_at": "2026-10-28",
  "read_form": [
    "html"
  ],
  "read_locator": "摘要、§1 引言、§2 背景（2.1 DiT、2.2 量化）、§3 两大挑战（Figure 3–4）、§4.1 CSB（Eq.3–8）、§4.2 SSC（Eq.9–11）、§4.3 重参数（Eq.12–13）、§5.1 设置、§5.2 Table 1–2、§5.3 Table 3、§6 结论",
  "topic": "transforms",
  "excerpt_sha256": null
}
---
## 这份来源是什么

PTQ4DiT 论文（arXiv 2405.16005，v1；IIT / UW-Madison / UCF），提出**专门面向 DiT 的训练后量化**。它指出 DiT 量化的两个难点：线性层里同时出现在激活与权重上的**极端幅值显著通道**（salient channel），以及显著激活分布随**去噪时间步**变化。对策是 **CSB**（Channel-wise Salience Balancing，用对角矩阵把激活与权重的显著幅度互补地重新分配）与 **SSC**（Spearman's ρ-guided Salience Calibration，按时间步与权重的相关性加权聚合激活显著性），再用**重参数**把平衡矩阵折进 adaLN 与量化/反量化，推理无额外开销。

它是 `algorithm-ptq4dit` 的一手依据；在本库中作为"DiT 需要按时间步与通道共同处理显著性"的代表，与同为扩散量化的 `paper-qdit`、`paper-viditq`、`paper-svdquant` 对照，并把 `paper-repqvit` 作为其 DiT 适配基线（RepQ*）。

范围边界：权重与激活都量化（论文主打 W8A8、W4A8）；均匀量化、权重 channel-wise、激活 tensor-wise；校准需要跨多个时间步；重参数后推理等价、无额外计算。

## 已读定位与阅读形式

- 阅读形式：arXiv HTML 渲染版（`arxiv.org/html/2405.16005v1`），正文、公式与表格按 HTML 结构解析；未读 LaTeX 源码或 PDF。
- 定位：§3 挑战（Figure 3–4）、§4.1 CSB（Eq.3–8）、§4.2 SSC（Eq.9–11、Figure 2 右）、§4.3 重参数（Eq.12–13、post-adaLN / post-matrix-multiplication）、§5.1 设置、§5.2 Table 1–2（Figure 5–6）、§5.3 Table 3。References 与附录未逐一读。

## 关键事实

- **挑战 ❶ 显著通道**（§3；互补性观察见 §1 Figure 1 左 / §4.1）：激活或权重里存在幅值远超常规的通道，均匀量化时为保住多数通道要截断这些极端值，导致系统性误差；论文观察到激活与权重的显著通道**不重合**，因此可互补重分配。
- **挑战 ❷ 显著激活的时间步变化**（§3，Figure 4）：DiT 跨多时间步去噪，显著激活通道的最大幅值随时间步明显变化，某步有效的量化参数在另一步可能不适用。
- **CSB**（§4.1，Eq.3–8）：定义通道显著性 `s(X_j)=max|X_j|`、`s(W_j)=max|W_j|`（Eq.4），用几何平均 `s̃=(s(X_j)·s(W_j))^{1/2}` 作"平衡显著性"（Eq.5），构造对角平衡矩阵 `B^X=diag(s̃/s(X_j))`、`B^W=diag(s̃/s(W_j))`（Eq.6–7），使 `X̃=XB^X`、`W̃=B^WW` 后两边的通道显著性相等，整体显著性上界下降（Eq.8）。
- **SSC**（§4.2，Eq.10–11）：激活显著性向量 `s(X^(t))` 随步变化、权重 `s(W)` 固定；激活与权重显著性**相关性越低**、互补后上界下降越多。于是用 Spearman ρ 的归一化指数权重 `η_t ∝ exp[−ρ(s(X^(t)),s(W))]` 对时间步加权聚合，得 `s_ρ=Σ_t η_t s(X^(t))`，再用 `s_ρ` 替换 Eq.5–7 里的激活显著性。
- **重参数**（§4.3，Eq.12–13）：`B^X_ρ` 与 `B^W_ρ` 互为逆，`X̃·W̃=X·W` 保持等价；离线更新 `W̃`，并把 `B^X_ρ` 折进前驱算子——post-adaLN 里改 `γ̃=γB^X_ρ`、`β̃=βB^X_ρ`（Eq.13），post-matmul 里吸收进前驱反量化，因而推理无额外计算。
- **设置**（§5.1）：ImageNet 上 class-conditional **DiT-XL/2**，分辨率 256×256 与 512×512；DDPM solver 250 步（另测 100、50 步）；统一均匀量化、权重 channel-wise、激活 tensor-wise（除另注）；校准对 256 分辨率取 25 个时间步、512 分辨率取 10 个时间步，每步生成 32 个样本；量化参数优化沿用 Q-Diffusion；RTX A6000；指标 FID/sFID/IS/Precision（ADM toolkit），256 采 10000 张、512 采 5000 张。
- **主结果**（Table 1，ImageNet 256，FID↓）：250 步 W8A8——FP 4.53、PTQ4DM 21.65、Q-Diffusion 5.57、PTQD 5.69、RepQ* 4.51、**PTQ4DiT 4.63**；250 步 W4A8——PTQ4DM 72.58、Q-Diffusion 15.31、PTQD 16.45、RepQ* 23.21、**PTQ4DiT 7.09**。100 步 W8A8 FP 5.00 → 4.73、W4A8 7.75；50 步 W8A8 FP 6.02 → 5.45、W4A8 9.17。论文称 W4A8 下 PTQ4DM FID 增加 68.05，而 PTQ4DiT 只增加 2.56。
- **512 分辨率**（Table 2，W4A8 FID↓）：250 步 FP 8.39、Q-Diffusion 58.81、PTQD 87.53、RepQ* 59.65、**PTQ4DiT 17.55**；100 步 FP 9.06 → 19.00；50 步 FP 11.28 → 19.71。论文称 250 步下比次优的 Q-Diffusion 低 41.26 FID、9.83 sFID。
- **消融**（Table 3，ImageNet 256，W4A8，250 步，FID↓）：FP 4.53、Q-Diffusion 15.31、Baseline 22.54、+CSB 8.17、+CSB+SSC **7.09**。

## 不能推出的结论

- 本页只读正文与所列表格，未复现实验、未运行；FID/sFID/IS/Precision 均为论文报告值。
- 结论只用 **DiT-XL/2** 在 ImageNet 256/512、DDPM 250/100/50 步上验证；不能外推到其它 DiT、视频 DiT、UNet 扩散、LLM 或 dLLM。
- 是**权重+激活**量化（W8A8/W4A8），但论文未报告真实低比特 kernel 的推理延迟；重参数保证"无额外计算"，不等于端到端有对应加速。
- 校准需要**跨多个时间步**生成样本（256 用 25 步、512 用 10 步，每步 32 样本），成本高于常规单步校准；换分辨率或采样步数需重确。
- CSB/SSC 是显著性**重分配**，不改变位宽与量化器；不能据此推断它能替代混合精度或低位 kernel。
- Table 2 的 PTQD 等基线行由作者复现/改编，配置是否各取最优未在表中说明；不能当严格的跨方法排名。
- 论文未给 SSC 超参（Spearman 权重）之外的敏感性分析；不能把某个时间步选择当成通用最优。
