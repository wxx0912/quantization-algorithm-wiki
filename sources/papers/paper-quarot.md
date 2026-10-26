---
{
  "id": "paper-quarot",
  "title": "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs",
  "category": "paper",
  "url": "https://arxiv.org/abs/2404.00456",
  "version": "v2（2024-10-29）",
  "version_kind": "paper-version",
  "retrieved_at": "2026-10-23",
  "read_form": [
    "latex",
    "html"
  ],
  "read_locator": "摘要、§1、§3.1–3.4、§4（Stage 1a–1d、Stage 2a–2c、Eq.4–14、Figure 3）、§5（Table 1–4、Figure 4）、附录 A.1–A.10（Table 5–17、Figure 5–7，含 KV cache 与 6/8-bit RTN 细节）",
  "topic": "core",
  "excerpt_sha256": null
}
---
## 这份来源是什么

QuaRot 原始论文（arXiv 2404.00456，v2；NeurIPS 2024）。它用**计算不变性**把随机 Hadamard 旋转插进 LLaMA 式 Transformer 前向图，使隐藏状态、FFN 激活、attention 的 value/key 都变成无异常值分布，从而做到权重、激活、KV cache **全 4-bit** 端到端量化，且不保留任何高精度 outlier 通道。

它是 `algorithm-quarot` 的原理依据，也是 `foundation-quantization-contract`、`recipe-composition` 的一手材料。

范围边界：**PTQ**（无权重重训，但权重默认用 GPTQ、需校准集）；主实验 INT4（正文 Table 3 另给 INT8 RTN 结果，INT6 细节在附录 A.6，KV cache 消融到 2-bit/3-bit）；旋转本身在满精度下完成、数学等价，量化误差另计；实验为 LLaMA-2 家族（附录补 LLaMA-3 与 Phi-3-mini），性能在 RTX 3090 上测。

## 已读定位与阅读形式

- 阅读形式：arXiv e-print 的 LaTeX 源（`main.tex` + `\input` 的 `intro/background/method/experiments/appendix/conclusion`），另读 arXiv HTML 渲染版交叉核对；v2 = 2024-10-29。
- 定位：§3.4 计算不变性、§4 Stage 1a–1d 与 Stage 2a–2c（Eq.4–14）、§5 Table 1–4 与 Figure 4、附录 A.1–A.10（Table 5–17，含 KV cache 与 6/8-bit RTN）。编号：Figure 1–7、Table 1–17；公式按源码为 Eq.1–14（arXiv HTML 渲染版的公式锚点编号与源码编号不一致，核对以 label/内容为准）。

## 关键事实

- **两阶段**（§4）：Stage 1 满精度改权重与图（1a 权重乘正交矩阵、1b FFN 激活旋转、1c value 投影旋转、1d key 旋转），Stage 2 才量化（2a 权重、2b 在线激活、2c attention）。改造后前向输出与原来**完全一致**（§3.4 定理）。
- **旋转插入**（§4，Eq.4–14）：RMSNorm 的线性缩放先融进相邻权重、再与 Hadamard `Q` 相乘（`W_k ← Qᵀ·diag(α)·W_k`）；down-proj 前插入满精度 online Hadamard（`W_down ← H·W_down`）；attention 按 head 插入 `H_{d_h}`，再用 `H_{n_h×d_h} = (I⊗H_{d_h})(H_{n_h}⊗I)` 拆成"融进 `W_out`"+"对激活在线做 `H_{n_h}⊗I`"；key 因 RoPE 无法直接融合，改为 query/key 在线 head-wise Hadamard，并选用 Post-RoPE caching（解码时只对一个 query 做变换）。
- **Stage 2 量化配置**（§5 Setup）：激活 per-token 对称、clipping 0.9；KV cache 非对称、group 128、clipping 0.95；权重 per-column 对称、RTN 或 GPTQ、clipping 由一维线性搜索定；GPTQ 校准集为 WikiText 128×2048。在单张 NVIDIA A100 上，Llama-2-70B 的 QuaRot 改造约 5 分钟、GPTQ 再约 2 小时（RTX 3090 只用于性能实验）。
- **主结果**（§5.1，Table 1–2）。WikiText-2 PPL（4-bit）：FP16 7B/13B/70B = 5.47/4.88/3.32，QuaRot(GPTQ，无 outlier 通道) **6.10/5.40/3.79**，QuaRot-128G 5.93/5.26/3.61；对照 SmoothQuant(RTN) 83.12/35.88、OmniQuant 14.26/12.30、QUIK-4B 8.87/7.78/6.91。零样本平均分（A4W4KV4）：FP16 69.82/72.59/77.07 → QuaRot 65.64/69.79/75.98（论文称平均分最多损失 4.18%，70B 仅 1.09%）。
- **RTN / 分组 / KV**（Table 3–4、附录 A.3）。论文称 6/8-bit 量化"无损"；逐值看 8-bit RTN 已与 FP16 几乎一致（7B PPL 5.50 vs 5.47、0-shot 平均 69.65 vs 69.82）、6-bit 仍有小差距（5.56 vs 5.47），8-bit RTN 无需校准集；4-bit RTN 明显更差（7B 8.37 vs GPTQ 6.10，70B 4.14 vs 3.79），差距随模型变大而缩小。4-bit+GPTQ 下 group size 越小越好（无分组 6.10/5.40/3.79 → 128G 5.93/5.26/3.61 → 64G 5.88/5.25/3.58），但每 group 存 scale、kernel 更复杂。KV cache 消融（group 128，权重/激活高精度）：7B 上 4/4 位 PPL 5.51、3/4 位 5.65、2/2 位 9.23，key 比 value 更敏感。
- **性能与扩展**（§5.2，Figure 4；RTX 3090，单 transformer block）。prefill 上 7B **1.97×–2.16×**、70B 最多 **3.33×**；decoding 峰值显存较 FP16 至少 **3.63×** 节省。用 CUTLASS 做 INT4 TensorCore 乘法（INT32 累加），KV cache 量化基于 FlashInfer。代码 `github.com/spcl/QuaRot`；附录 A.8/A.9 给 LLaMA-3 与 Phi-3-mini 结果（Table 11–13），说明流程不限于 LLaMA-2。

## 不能推出的结论

- "全 4-bit"指权重/激活/KV 都是 4-bit 且不保留高精度通道，但**不等于无需校准**：默认权重用 GPTQ，需要 128 条 WikiText 样本；论文所称"无需校准数据"只适用于 RTN；6/8-bit 的"无损"是论文口径，逐值数字见上（8-bit 7B 5.50 vs 5.47、6-bit 5.56 vs 5.47），4-bit RTN 明显掉点（7B PPL 8.37）。
- 旋转是数学等价变换，本身不引入误差；Table 1/2 的精度损失来自量化，不能归因于旋转，也不能推出"旋转消除量化误差"。
- Table 1 的对照方法不是在统一实现下复现的（SmoothQuant/OmniQuant 取自 OmniQuant 论文，QUIK 用"全部层 4-bit"配置），不能当作同一 pipeline 的公平对照。
- 1.97×–3.33× 加速与 3.63× 显存节省是**单 block**、RTX 3090、自定义 CUTLASS kernel 下的测量，不能外推到整机端到端、A100/H100 或其它 kernel。
- Eq.9 的恒等式 `H_{n_h×d_h} = (I⊗H_{d_h})(H_{n_h}⊗I)` 要求 head 数 `n_h` 与 head 维度 `d_h` 都是 2 的幂，是该实现的适用范围约束。
- 附录 A.8/A.9（LLaMA-3、Phi-3-mini）已逐值核对 Table 11–13：Table 11 的 4-bit WikiText PPL 为 LLaMA-3-8B/70B FP16 6.14/2.86 → QuaRot(GPTQ) 8.16/6.66 → QuaRot-128G 7.36/5.51；Table 12 零样本平均 FP16 73.22/79.94 → QuaRot 65.18/69.21；Table 13 Phi-3-mini FP16 PPL 6.35/平均 73.11 → QuaRot-GPTQ INT4 7.85/66.34、QuaRot-RTN INT4 11.69/56.12。LLaMA-3 附录只报 GPTQ 与 GPTQ-128G、无 4-bit RTN，这些结论仍限于这两类架构，不能视为已在所有架构上验证。
