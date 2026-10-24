---
{
  "id": "paper-gptq",
  "title": "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers",
  "category": "paper",
  "url": "https://arxiv.org/abs/2210.17323",
  "version": "v2（2023-03-22）",
  "version_kind": "paper-version",
  "retrieved_at": "2026-10-23",
  "read_form": [
    "latex",
    "html"
  ],
  "read_locator": "摘要、§1（Figure 1）、§3（Eq.1；Eq.2–3）、§4（Figure 2、Step 1–3、Eq.4–5、Algorithm 1）、§5（Table 1–7、Figure 3–4）、§6、附录 A.1（Table 8）、A.2、A.3（Table 9–12）、A.4（Table 13–22）",
  "topic": "core",
  "requires_external_verification": true,
  "excerpt_sha256": null
}
---
## 这份来源是什么

GPTQ 原始论文（arXiv 2210.17323，v2；同时是 ICLR 2023 会议版）。它把 OBQ 的逐层重构改成能在数百亿参数上跑完的 one-shot 权重量化：固定列顺序、按 128 列分块惰性更新、用 Cholesky 预处理 `H⁻¹`，再配一个"解压权重后做矩阵-向量乘"的 kernel 换端到端收益。

它是 `algorithm-gptq` 的原理与实验依据，也是 `algorithm-baselines` 与 `foundation-quantization-contract` 的一手材料。

范围边界：只量化**权重**、不做激活量化，one-shot PTQ 不训练，收益来自减少访存而不是减少计算；实验是 OPT/BLOOM 两个密集模型族，另有 RN18/RN50 与 BERT 小模型对照。

## 已读定位与阅读形式

- 阅读形式：arXiv e-print 的 LaTeX 源（`gptq.tex`，ICLR 2023 版；同一 e-print 内的 `arxiv.tex` 是更早排版，本页一律以 `gptq.tex` 为准），另读 arXiv HTML 渲染版交叉核对编号。v2 = 2023-03-22。
- 定位：§3 逐层重构目标（Eq.1–3）、§4 GPTQ 算法与 Algorithm 1（Eq.4–5）、§5 实验（Table 1–7、Figure 3–4）、§6 局限、附录 A.1–A.4（与 OBQ 对照、实验细节、语言生成与 zero-shot 补充表）。

## 关键事实

- **目标与二阶信息**（§3，Eq.1）：逐层 `argmin_Ŵ ‖WX − ŴX‖²₂`，按输出通道（行）独立并行求解；`H_F = 2X_FX_Fᵀ` 是该层重构目标的二阶信息，不是任务损失 Hessian。
- **三步实现**（§4，Algorithm 1）：① 固定列顺序（greedy 选点在大层上收益很小）；② 每次处理 `B=128` 列、块内只更新本块、块末再全局更新 `W` 与 `H⁻¹`（不减少计算量，但解掉访存瓶颈）；③ 用 Cholesky kernel 一次性预算 `H⁻¹` 相关信息，dampening 取 `λ = 1% × 平均对角值`。
- **设置**（§5 Setup）：单张 A100-80GB 量化到 175B；校准集是 C4 的 128 条随机 2048-token 片段；量化格为 uniform per-row asymmetric（min-max），RTN 用同一网格。量化耗时 OPT-175B 4.2h、BLOOM-176B 3.8h。
- **主结果**（§5，Table 5）：OPT-175B WikiText2 FP16 8.34 → GPTQ-4 8.37、GPTQ-3 8.68，而 RTN-3 崩到 7.3e3；BLOOM-176B 8.11 → 8.21（4bit）/ 8.64（3bit）。论文自述模型越大越好压（OPT-66B 例外，归于该 checkpoint 早期层 dead units）。
- **部署契约**（§5）：3-bit OPT-175B 约 63GB（embedding 与输出层仍 FP16）+ 约 9GB KV，可整体放进单张 80GB A100；加速来自"解压权重后矩阵-向量乘" kernel，batch size 1 生成下 A100 230ms → 71ms（3.24×）。
- **口径提示**：C4 评测与校准数据同源，论文自己在 caption 里写 "not fully zero-shot"（附录 A.3）；PPL 按不重叠 2048 段、next-token log-prob 指数平均计算（附录 A.2）。
- **原文内部差异**：OPT-125M 的 FP16 参考 PPL 在 Table 3 记 27.65、Table 8 记 27.66，引用时注明"原文如此"。

## 不能推出的结论

- 只覆盖静态权重量化：不做激活量化、不训练，也没有导出格式/packing 与具体推理后端（vLLM/TensorRT-LLM）的集成结论；论文自带的 3-bit kernel 不代表其它后端支持同样粒度。
- 逐 block 顺序量化，下一个 block 的输入取自**已被量化**的前序 block（顺序耦合），不能据此推断它优化了端到端任务指标。
- 3.24× 绑定 batch size 1 的生成路径与 HF/accelerate 分层切分（通信 <5%）；论文自己说明大 batch 与非生成场景该 kernel 不直接适用，不能外推为 prefill 或低比特 GEMM 的吞吐。
- PPL 结论只在论文的模型族、per-row asymmetric min-max 网格与上述评测协议下成立；C4 与校准集同源。
- 与 OBQ 的对照只在 BERT-base 与 OPT-125M 上做，"GPTQ 接近 OBQ 精度"不能推广到所有模型规模。
