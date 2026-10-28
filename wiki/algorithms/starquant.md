---
{
  "id": "algorithm-starquant",
  "title": "STaR-Quant：dLLM 的状态-时间一致 W4A4 量化",
  "type": "algorithm",
  "tags": [
    "dllm",
    "activation",
    "equivalent-transformation",
    "compensation",
    "quantizer",
    "calibration"
  ],
  "aliases": [
    "STaR-Quant",
    "状态时间一致量化",
    "SGAT",
    "TAC",
    "dLLM W4A4 量化"
  ],
  "related": [
    "algorithm-dllmquant",
    "algorithm-quantdllm",
    "algorithm-faircalib",
    "algorithm-quarot",
    "foundation-quantization-contract",
    "guide-search-space"
  ],
  "sources": [
    "paper-starquant"
  ],
  "confidence": "source-reported",
  "evidence_scope": "source-reported",
  "updated": "2026-10-28"
}
---

# STaR-Quant：dLLM 的状态-时间一致 W4A4 量化

## 一句话理解

STaR-Quant 针对 dLLM 的两个特有问题出手：对"状态维"，用 SGAT 把掩码 token 与未掩码 token 送进不同的激活变换子空间（但共用同一份变换后的权重）；对"时间维"，用 TAC 在注意力输出投影前对量化后的注意力表示做一次轻量块对角仿射补偿，抑制跨去噪步的误差累积。目标是在 W4A4 下尽量保住知识、推理与代码任务。

## 核心机制

1. **SGAT（激活，Eq.7–15）**：按 mask 状态把 token 分组，各自走不同的激活变换。隐藏维拆成共享子空间与状态专属子空间 `[U_sh,U_m,U_u]`，用二值门得到掩码用基 `[U_sh,U_m,0]`、未掩码用基 `[U_sh,0,U_u]`；两态共享 `U_sh`、各用专属子空间平滑自己的离群。**权重侧只保留一个统一静态变换** `W̃=Q_w(P_e W^T)`（算一次），因此推理只做状态条件化的激活变换，不重复变换权重。
2. **TAC（注意力输出，Eq.16–18）**：不改动注意力本身，而是在软注意力输出（softmax×V）与输出投影之间插入补偿。用块对角仿射 `Z^{tac}=ẐM^T+1d^T`，参数由全精度与量化表示的均值/协方差匹配闭式求解，再做奇异值平滑与向单位阵收缩以稳估计。
3. **逐层校准：SGAT 梯度优化、TAC 闭式求解**：SGAT 冻结原模型参数，用 AdamW（lr 1e-3、batch 4、15 个校准 epoch）逐层最小化量化输出与全精度输出的 Frobenius 误差；TAC 由配对统计量闭式求出。整体是 PTQ（原模型参数不更新、不做微调），但不能理解为全程无梯度。

## 基本流程

1. 在 dLLM 的逐步去噪过程中，按当前 mask 状态把每个 token 标记为 masked / unmasked。
2. 逐层优化 SGAT 的共享与状态专属子空间，把权重侧变换折叠进量化权重（推理只留激活侧变换）。
3. 在校准阶段得到的配对激活上，对每个注意力层估计 TAC 的块对角 `M_b`、`d_b`，默认 `g=16`。
4. 推理：激活按状态变换并量化 → 量化线性 → attention → 输出投影前做 TAC 补偿 → 输出投影。

## 适用条件

- 面向**掩码迭代去噪**的 dLLM 的 **W4A4** 权重+激活量化（激活 per-token、权重 per-channel，§3.1）；论文验证 LLaDA-8B、LLaDA-1.5-8B、Dream 7B。
- 需要能拿到每个去噪步的 mask 状态、并定位 attention 的 softmax×V 输出作为 TAC 插入点；需要一小段校准数据（论文：WinoGrande 128 段）。
- 依赖融合 kernel 才能兑现论文的加速；低比特权重+激活能否在目标后端执行仍由部署侧决定。

## 结果边界

- 论文报告（Table 1，W4A4，9 项平均）：LLaDA-8B 由 QuaRot 51.03 / DLLMQuant++ 54.29 提到 **57.07**（FP 58.99）；LLaDA-1.5-8B **66.93**（DLLMQuant++ 64.31）；Dream-7B **63.59**（DLLMQuant++ 61.90）。
- 难任务增益：LLaDA-8B HumanEval 28.92→35.98、GSM8K 56.25→57.29；Dream HumanEval 44.50→47.43、GSM8K 66.17→69.57。
- 消融：去掉 TAC 降到 55.43、去掉 SGAT 降到 56.39（完整 57.07，LLaDA-8B）；TAC 块大小 `g=16` 最优。
- 效率：论文报告平均 1.66× 加速、约 3.05×/3.14× 显存节省（A40，沿用 QuaRot 的 W4A4 kernel）。
- 只覆盖 W4A4 与掩码去噪 dLLM；2-bit 权重+激活仍困难；状态划分、`g`、阻尼等为经验设定。未复现、未在目标后端核验。

## Agent 如何使用

处理 dLLM 的 W4A4 掉点、尤其想区分"激活状态不一致"与"跨步误差累积"时先读本页，再按需展开：

| 需要确认的内容 | 继续读取 |
| --- | --- |
| SGAT/TAC 公式、Eq.7–18、Table 1–4 与设置 | [`paper-starquant`](../../sources/papers/paper-starquant.md) |
| 同族 dLLM 量化的其它切入（校准采样、提交脆弱点） | [DLLMQuant](dllmquant.md)、[FAIR-Calib](faircalib.md) |
| 旋转/变换类方法的通用前提 | [QuaRot](quarot.md) |
| 量化契约与搜索空间 | [量化契约](../foundations/quantization-contract.md)、[量化搜索空间](../guides/search-space.md) |

回答时把贡献拆成"激活变换空间（SGAT）"与"注意力输出补偿（TAC）"，并说明二者互补、都要在 W4A4 语境下理解。

## 详细来源

- [STaR-Quant 原论文来源页](../../sources/papers/paper-starquant.md)
