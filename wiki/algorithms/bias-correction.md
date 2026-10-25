---
{
  "id": "algorithm-bias-correction",
  "title": "偏置校正：消除量化引入的输出偏置",
  "type": "algorithm",
  "tags": [
    "compensation",
    "activation",
    "scaling",
    "rtn"
  ],
  "aliases": [
    "偏置校正",
    "bias correction",
    "量化偏置",
    "DFQ"
  ],
  "related": [
    "algorithm-baselines",
    "algorithm-gptq",
    "algorithm-svdquant",
    "algorithm-smoothquant",
    "guide-search-space"
  ],
  "sources": [
    "paper-dfq"
  ],
  "confidence": "source-reported",
  "evidence_scope": "source-reported",
  "updated": "2026-10-25"
}
---

# 偏置校正：消除量化引入的输出偏置

## 一句话理解

量化不是"零均值噪声"就一定安全：当权重误差对某输出单元求和不为零时，该单元输出的均值会漂移，并在后续层被放大。偏置校正把期望误差 `E[εx]` 从层输出里减掉（实现上折进 bias），从而保持每个输出的均值；在有 BatchNorm 的网络里还能不用数据算出该期望。

## 核心机制

- **有偏误差**：`ε = W̃ − W`；若 `∑ᵢεᵢⱼ ≠ 0`，输出普遍偏移。论文指出偏差不仅来自量化，也来自权重/激活裁剪、张量分解、通道剪枝。
- **校正**：因 `ỹ = W̃x = y + εx`，从输出减去 `E[εx] = ε·E[x]`，即可保持输出均值；由于形状与该层输出一致，可直接并入 bias，**无运行时开销**（可推广到卷积层）。
- **无数据估计**：需要 `E[x]`。若有 BatchNorm，可用其 `γ/β` 与"裁剪正态分布"闭式求激活均值，从而无数据求出 `E[εx]`。
- **与范围均衡**：跨层范围均衡是保持输出的重参数化，但会增大某些层的 bias、进而增大激活量化范围；论文把均衡（含 bias 吸收）与偏置校正合用构成 DFQ。均衡本身不是有偏误差来源，校正修的是均衡后权重量化产生的有偏误差。

## 基本流程

1. 折叠 BatchNorm 并（如需）先做跨层范围均衡与 bias 吸收——论文附录 D 要求偏置校正在这两步之后执行。
2. 量化（或裁剪/分解）权重，得到 `W̃`。
3. 估计该层输入的期望 `E[x]`：无数据时用 BatchNorm 参数 + 裁剪正态分布；有数据时直接统计激活。
4. 计算期望误差 `E[εx] = ε·E[x]`，把它并入该层 bias。

## 适用条件

- 只要操作引入了"有偏"误差即可用，不限于量化本身；与具体量化格式解耦。
- 无数据版本依赖 BatchNorm 的 `γ/β` 且后接可裁剪的激活（如 ReLU/ReLU6）；无 BN 的模型需用数据估计 `E[x]`。
- 只修正输出**均值**；对偏置不敏感但方差敏感的路径，收益有限。

## 结果边界

- 论文报告（MobileNetV2）：仅偏置校正改善量化模型但不足以接近浮点；完整 DFQ 达近似浮点、top-1 仅损失 0.53%。
- FP32 权重裁剪无校正损失 4.66%，加校正降到 0.57%，说明它对非量化来源的有偏噪声同样有效。
- 语义分割 DeeplabV3+（MobileNetV2 backbone）：原模型 FP32 72.94 → INT8 41.40；DFQ FP32 72.45 → INT8 72.33；per-channel INT8 71.44。
- 这些是 CNN 的 8-bit 整型量化（权重+激活）分类/分割指标，不能直接外推到 LLM 的 PPL、生成质量或端到端任务。
- 当前 Wiki 未复现；无数据版本的前提（BN）在现代 LLM 上通常不成立。

## Agent 如何使用

怀疑"量化后输出整体偏移、误差不是零均值"时读本页，再按需展开：

| 需要确认的内容 | 继续读取 |
| --- | --- |
| 有偏误差定义、Eq.13–16、消融与 Table 3 | [`paper-dfq`](../../sources/papers/paper-dfq.md) |
| 基线量化（RTN/范围校准）与误差来源 | [基础基线](baselines.md) |
| 另一条误差补偿路径（低秩分支） | [SVDQuant](svdquant.md) |
| 误差补偿在搜索空间中的位置 | [量化搜索空间](../guides/search-space.md) |

回答时要区分"偏置校正（修均值）"与"重构/补偿（修逐元素误差）"，并说明无数据前提只在有 BN 的网络上成立。

## 详细来源

- [DFQ 原论文来源页](../../sources/papers/paper-dfq.md)
