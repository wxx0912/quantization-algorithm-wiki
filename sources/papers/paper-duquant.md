---
{
  "id": "paper-duquant",
  "title": "Rotation and Permutation for Advanced Outlier Management and Efficient Quantization of LLMs（DuQuant）",
  "category": "paper",
  "url": "https://arxiv.org/abs/2406.01721",
  "version": "v1（2024-06-03）",
  "version_kind": "paper-version",
  "retrieved_at": "2026-10-28",
  "read_form": [
    "html"
  ],
  "read_locator": "摘要、§2 Motivation、§3.1 预备（Eq.1）、§3.2 方法（Eq.2–5）、§3.3 理论、§4.1 主结果（Table 1–4）、§4.2 消融（Table 5–7）、§5 结论",
  "topic": "transforms",
  "excerpt_sha256": null
}
---
## 这份来源是什么

DuQuant 论文（arXiv 2406.01721，v1；该版本标题为“Rotation and Permutation for Advanced Outlier Management and Efficient Quantization of LLMs”，后续版本改名“DuQuant: Distributing Outliers via Dual Transformation Makes Stronger Quantized LLMs”），提出面向 LLM 权重+激活低比特量化的**双重变换**（旋转 + 置换）。它区分两类激活离群：**Normal Outliers**（在所有 token 序列里都出现的固定特征维）与 **Massive Outliers**（也叫 attention sinks，只出现在少数 token、量级常 >100、约为其它激活中位数的 1000 倍），并指出现有平滑类方法（SmoothQuant/OmniQuant）难以处理 massive outliers。方法为 smooth → 块对角旋转 → 之字形置换 → 第二次旋转（合称 RAP），在 LLaMA 1/2/3 与 Vicuna 上评估。

它是 `algorithm-duquant` 的一手依据；在本库中作为"旋转 + 置换联合处理离群"的代表，与 `paper-quarot`（纯 Hadamard 旋转）、`paper-smoothquant`（通道缩放）、`paper-omniquant`（可学习等价变换）对照。

范围边界：W4A4 为主（附录含 W6A6）；整数均匀量化，激活 per-token、权重 per-channel；可选 `+LWC`（来自 OmniQuant 的可学习权重裁剪）。校准为随机 128 条 WikiText2 序列、序列长 2048（附录 C）；实验用 RTX 3090（小模型）与 A100（大模型）。

## 已读定位与阅读形式

- 阅读形式：arXiv HTML 渲染版（`arxiv.org/html/2406.01721v1`），正文与表格按 HTML 结构解析；未读 LaTeX 源码或 PDF。
- 定位：§2 Motivation（两类离群、FFN 下投影层的 massive outliers）、§3.1 预备（Eq.1 均匀量化）、§3.2 方法（Eq.2 旋转构造、Eq.3 块对角近似、Eq.4 块间方差、Eq.5 整体变换）、§3.3 理论（Thm.1/2）、§4.1 主结果（Table 1–4）、§4.2 消融（Table 5–7、Figure 3–4）、§5 结论。附录 B–H 未逐条读。

## 关键事实

- **两类离群**（§2）：Normal outliers 在所有 token 序列里持续出现；Massive outliers（attention sinks）只在少数 token 出现，量级常 >100、约为其它激活中位数的 1000 倍。论文称其首次发现 massive outliers 出现在 **FFN 下投影（down-projection）层的输入**（而非仅在 transformer block 输出），例如 LLaMA2-7B 第 1 层该处有一个约 **1200** 的单点激活。
- **为何难处理**（§2）：SmoothQuant 的通道平滑在 massive outlier 处会得到很大的平滑因子，把离群迁移到 down-projection 权重上；OmniQuant/AffineQuant 因梯度不稳而必须排除 down projection 的可学习参数；QUIK 对 down projection 用 INT8、Atom 对 128 个离群通道用 INT8。
- **双重变换（RAP）**（§3.2，Eq.5）：`Y=X·W=[(X·Λ)·R̂_(1)·P·R̂_(2)]·[R̂_(2)ᵀ·Pᵀ·R̂_(1)ᵀ·Λ⁻¹·W]`，即先 smooth（`Λ`，把量化难度从激活迁到权重），再块对角旋转 `R̂`、之字形置换 `P`、第二次旋转 `R̂`，合起来记作 `G`（激活侧在线施加）与 `G⁻¹`（可离线折入权重）。
- **块对角旋转**（§3.2，Eq.2–3）：用带先验的贪心搜索定位离群主维、构造正交旋转，并按块对角近似 `R̂=BlockDiag(R̂_{b1},…,R̂_{bK})`，块大小 `2^n`、块数 `K=C_in/2^n`；为省算力可令所有块复用同一个块旋转矩阵（Remark 2）。
- **之字形置换**（Eq.4）：块对角旋转只做块内混合，块间离群量级方差仍大；之字形置换把激活最大的通道按"大到小、再小到大"往返分配到各块，平衡块间离群幅度，再由第二次旋转进一步平滑。
- **量化设置**（§3.1、§4、附录 C）：整数均匀量化（Eq.1，含 scale `Δ` 与零点 `z`）；激活 per-token、权重 per-channel；`DuQuant` 用 RTN，激活裁剪比 0.9、权重 0.8；`DuQuant+LWC` 叠加 OmniQuant 的可学习权重裁剪；校准为随机 128 条 WikiText2 序列、序列长 2048。平滑矩阵与 LWC 因子可折进量化权重、不增加推理开销。
- **推理开销**（§4.2、Table 9）：相对 W4A4，仅旋转增加约 +1.5%（7B）/+1.8%（13B）计算开销，一个置换+两次旋转增加约 +8.9%/+9.3%；prefill 阶段相对 FP16 约 2.08×（7B）/2.34×（13B）加速（RTX 3090、batch=16；batch=1 时为 1.95×/2.15×）。
- **主结果**（Table 1，W4A4 WikiText2 PPL，越低越好）：LLaMA2-7B FP16 5.47 → DuQuant **6.28** / +LWC **6.08**（Atom 8.40、OmniQuant 14.26、SmoothQuant 83.12）；LLaMA2-70B FP16 3.31 → **3.79** / **3.76**（OmniQuant 与 Atom 在该设置下 NaN）；LLaMA1-7B FP16 5.68 → 6.40 / 6.18；C4 上 LLaMA2-7B FP16 6.97 → 7.90 / 7.79。
- **zero-shot**（Table 2，W4A4，LLaMA1-7B 六任务平均）：FP16 64.09、SmoothQuant 38.41、Outlier Suppression+ 48.43、OmniQuant 52.65、AffineQuant 53.42、QLLM 51.84、Atom 56.88、DuQuant **61.76**（DuQuant+LWC 61.18）；§4.1 称其平均准确率比 QLLM 高约 +9%、比 Atom 高约 +5%。
- **模块消融**（Table 5，LLaMA2-7B/13B 的 WikiText2/C4）：仅 smooth（7B/13B）NaN/1379.46 与 160.30/203.87；仅 rotation 8.48/10.63 与 14.32/21.73；smooth+rotation 7.92/10.64 与 5.96/7.94；rotation+permutation+rotation2 6.79/8.51 与 6.06/8.03；四项全开 **6.28/7.90** 与 **5.42/7.05**。
- **与 QuaRot 对比**（Table 7，W4A4 PPL）：DuQuant 6.40/5.65/4.72/6.28/5.42，QuaRot-GPTQ 6.44/5.63/4.73/6.39/5.75，QuaRot-RTN 7.08/6.57/5.44/9.66/6.73。

## 不能推出的结论

- 本页只读正文与所列表格，未复现实验、未运行；PPL 与 zero-shot 均为论文报告值。
- 主结果限于 **W4A4**（附录另有 W6A6）；不能据此推断 W2A2 或权重-only 设置。
- 结论基于 LLaMA 1（7B–65B）、LLaMA2（7B–70B）、LLaMA3（8B/70B）与 Vicuna-v1.5（7B/13B）；不能外推到 MoE、dLLM、DiT、多模态或其它模型族。
- `DuQuant` 与 `DuQuant+LWC` 是两个配置；`+LWC` 引入来自 OmniQuant 的可学习权重裁剪，不能把两者当作同一方法。
- 块大小 `2^n`、贪心搜索步数 `N`、裁剪比（0.9/0.8）与"块复用同一旋转矩阵"等为该论文设定；换模型/位宽需重新确认。
- 论文称 massive outliers 出现在 down-projection 输入，这是其观测与结论，不能推广为"所有模型结构都如此"。
