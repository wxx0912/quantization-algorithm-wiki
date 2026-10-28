---
{
  "id": "paper-starquant",
  "title": "STaR-Quant: State-Time Consistent Post-Training Quantization for Diffusion Large Language Models",
  "category": "paper",
  "url": "https://arxiv.org/abs/2606.04945",
  "version": "v2（2026-06-08）",
  "version_kind": "paper-version",
  "retrieved_at": "2026-10-28",
  "read_form": [
    "html"
  ],
  "read_locator": "摘要、§1 引言、§3.1–3.3（Eq.1–18）、§4.1 设置、§4.2 Table 1、§4.3 Table 2–3、§4.4 Table 4、§5 结论、Limitations",
  "topic": "mixed-architecture",
  "excerpt_sha256": null
}
---
## 这份来源是什么

STaR-Quant 论文（arXiv 2606.04945，v2），研究扩散大语言模型（dLLM）的低比特**权重+激活**训练后量化。它指出两个 dLLM 特有问题——掩码 token 与未掩码 token 的激活分布不同（state-dependent activation disparity）、量化误差沿迭代去噪步累积（temporal error accumulation）——并提出 State-Guided Activation Transformation（SGAT）与 Temporal Attention Compensation（TAC），在 LLaDA-8B、LLaDA-1.5-8B、Dream 7B 上评估。

它是 `algorithm-starquant` 的一手依据。与同族来源的分工：`paper-dllmquant` 用校准采样与权重补偿处理同一类问题，`paper-faircalib` 从"提交脆弱点"重加权校准，本篇从**激活变换空间**与**注意力输出补偿**两处切入；`paper-quantdllm` 是权重-only 2-bit 路线。

范围边界：W4A4（权重 4-bit、激活 4-bit），PTQ、无需微调；硬件 NVIDIA A40。

## 已读定位与阅读形式

- 阅读形式：arXiv HTML 渲染版（`arxiv.org/html/2606.04945v2`），正文与表格按 HTML 结构解析；未读 LaTeX 源码或 PDF。
- 定位：§3.1 预备（Eq.1–6）、§3.2 SGAT（Eq.7–15）、§3.3 TAC（Eq.16–18）、§4.1 设置、§4.2 Table 1 主结果、§4.3 Table 2–3 消融、§4.4 Table 4 效率、§5 结论与 Limitations；附录 B（实现细节：子空间维度划分、AdamW 超参与 15 个校准 epoch、校准后固定权重侧变换）与附录 D 及其引用的 Table 5（W8A8；HTML 抽取中表注排版位置落在附录 C 之后、附录 D 之前，归属按附录 D 正文说明）已读。附录 A（SGAT 一致性证明）与附录 C（TAC 细节）未逐条读，其要点以正文 §3.2/§3.3 为准。

## 关键事实

- **两个挑战**（§1）：① 掩码与未掩码 token 激活的量程与离群模式不同，用统一激活变换/量化空间无法同时平滑两者；② 生成逐步迭代、每步输出作为下一步输入，量化误差跨步累积，其中 attention 里 softmax 与 value 相乘是低比特下的关键易错算子。
- **SGAT**（§3.2）：按 mask 状态把 token 分组（`X_t^s`，`s∈{m,u}`）；把隐藏维分解为共享与状态专属子空间 `d=d_sh+d_m+d_u`、基 `P=[U_sh,U_m,U_u]`，用二值子空间门 `G_m=diag(1_dsh,1_dm,0_du)`、`G_u=diag(1_dsh,0_dm,1_du)` 得 `P_m=PG_m=[U_sh,U_m,0]`、`P_u=PG_u=[U_sh,0,U_u]`；激活变换 `Φ_s(X_t^s)=X_t^s PG_s`。**权重侧保持统一静态变换** `W̃=Q_w(P_e W^T)`（正交基时 `P_e=P^T`），只算一次，推理时只做状态条件化的激活变换，避免重复变换权重。优化目标为逐层 `L_state=E_t Σ_s ‖Y_t^{s,FP}−Q_a(X_t^s PG_s)W̃‖_F²`（Eq.15）；在逐层 PTQ 校准中只更新变换基（用 AdamW，lr 1e-3、batch 4、15 个校准 epoch），原模型参数冻结；校准结束后才用得到的基预计算并固定权重侧变换 `W_f=Q_w(PᵀWᵀ)`。
- **TAC**（§3.3）：不改注意力计算，而是在 attention 输出投影前补偿量化后的注意力表示 `Ẑ_t`（softmax 输出与 V 相乘的结果，Eq.16）。采用**块对角仿射**：把隐藏维分成块，`Z^{tac}_{t,b}=Ẑ_{t,b}M_b^T+1 d_b^T`（Eq.17）；参数由全精度与量化表示的**均值/协方差匹配**闭式给出：`M_b=(Σ_b^FP)^{1/2}(Σ_b^Q+λI)^{−1/2}`、`d_b=μ_b^FP−M_b μ_b^Q`（Eq.18），再用奇异值平滑与向单位阵收缩稳定。默认块大小 `g=16`。
- **设置**（§4.1）：NVIDIA A40；PTQ 无需微调；激活 per-token、权重 per-channel（§3.1）；模型 LLaDA-8B、LLaDA-1.5-8B、Dream 7B；基线 RTN、AWQ、QuaRot（按官方仓库配 GPTQ 权重补偿）、DLLMQuant；校准为 WinoGrande 采样 128 段；评测 9 项——TruthfulQA-MC2、ARC-Challenge、HellaSwag、WinoGrande、PIQA、MMLU、C-EVAL、GSM8K、HumanEval。
- **主结果**（Table 1，W4A4，9 项平均）：LLaDA-8B FP 58.99、RTN 44.23、AWQ 48.09、DLLMQuant+ 49.26、QuaRot 51.03、DLLMQuant++ 54.29、STaR-Quant **57.07**；LLaDA-1.5-8B FP 69.86、QuaRot 61.06、DLLMQuant++ 64.31、STaR-Quant **66.93**；Dream-7B FP 66.94、DLLMQuant++ 61.90、STaR-Quant **63.59**。相对 DLLMQuant++ 分别 +2.78 / +2.62 / +1.69。
- **难任务**（§4.2）：LLaDA-8B 上 HumanEval 28.92→**35.98**、GSM8K 56.25→**57.29**；Dream-7B 上 HumanEval 44.50→**47.43**、GSM8K 66.17→**69.57**。
- **消融**（§4.3，Table 2，W4A4，LLaDA-8B，9 项平均）：FP 58.99；去掉 TAC 55.43；去掉 SGAT 56.39；完整 57.07。TAC 块大小（Table 3）g=4/8/16/32/64 → 52.48/54.66/**57.07**/56.39/56.03。
- **效率**（§4.4，Table 4）：LLaDA/LLaDA-1.5/Dream 分别 1.65×/1.64×/1.69× 加速（平均 1.66×）；显存 LLaDA 与 LLaDA-1.5 由约 15.9GB 降到 5.2GB（3.05×），Dream 由 13.95GB 降到 4.44GB（3.14×）。加速沿用 QuaRot 的 W4A4 kernel 与测量协议。

## 不能推出的结论

- 本页只读正文与所列表格，未复现实验、未运行；准确率、加速与显存均为论文报告值。
- 结论限于 LLaDA-8B、LLaDA-1.5-8B、Dream 7B 三个**掩码去噪** dLLM 与 WinoGrande 校准；不能外推到 AR LLM、图像/视频 DiT、块扩散（block-diffusion）模型或其它 dLLM。
- 主文重点为 **W4A4**（Limitations 明确 3-bit/2-bit 权重+激活仍困难，不能据此认为更低比特有效）；附录 D 引用/讨论的 Table 5 另报告 **W8A8** 结果。
- SGAT 的 `d_sh/d_m/d_u` 划分、状态定义为预定义的 masked/unmasked，TAC 的 `g=16`、阻尼 `λ`、奇异值/收缩细节属经验设定；换模型或调度需重确。
- 加速与显存是其在 A40 + 沿用 kernel 上的报告，不等于任意后端的端到端收益；论文也指出 TAC/SGAT 的实际加速依赖融合 kernel。
