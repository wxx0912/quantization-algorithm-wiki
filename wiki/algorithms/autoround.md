---
{
  "id": "algorithm-autoround",
  "title": "AutoRound / SignRound：用符号梯度联合优化舍入与裁剪",
  "type": "algorithm",
  "tags": ["ptq", "rounding", "reconstruction", "clipping", "llm"],
  "aliases": ["AutoRound", "SignRound", "SignRoundV1", "signed gradient descent"],
  "related": ["algorithm-adaround", "algorithm-awq", "algorithm-brecq"],
  "sources": ["paper-autoround", "repo-autoround"],
  "confidence": "source-reported",
  "evidence_scope": "mixed",
  "updated": "2026-10-24"
}
---

# AutoRound / SignRound：用符号梯度联合优化舍入与裁剪

## 一句话理解

SignRound 是论文中的算法，AutoRound 是持续演进的工具。SignRound 在 Transformer block 内冻结原始权重，用 SignSGD 联合优化舍入偏移和 clipping 参数，主要面向非对称 weight-only LLM 量化。

## 核心机制

方法优化舍入和范围参数，并使用 block 重构目标：

- `V`：控制权重向上或向下舍入；
- `α、β`：调整量化范围的上下界；
- block 输出重构损失：作为优化目标，而不是全模型任务损失。

SignSGD 只使用梯度符号更新这些有界参数。优化结束后保存损失最低的状态，把 `V`、`α/β` 折进量化权重；推理阶段不需要 `V`、SignSGD 或优化器（最后一句为本库工程推断，论文算法只输出 `best_V/best_α/best_β`）。

## 基本流程

1. 按 Transformer block 缓存校准输入和浮点输出。
2. 根据目标位宽、分组和 affine zero-point 初始化量化范围。
3. 用 SignSGD 联合优化舍入偏移和 clipping 参数。
4. 保存最佳 block 状态，固化量化权重。
5. 逐 block 处理并记录输入来源与实际导出格式；输入来自前一浮点还是已量化 block 是配置选项。

本页按工具约定将 `G-1` 理解为每行不再细分组；论文未明确定义该记号，不能理解成 group size 1。

## 适用条件

- 论文主要验证非对称 weight-only W4G-1、W4G128、W3G128 和 W2G128。
- 校准集为 pile-10k 的 512 条随机样本，默认序列长度 2048、每 block 200 步、batch size 8、初始学习率 5e-3 并线性衰减，使用 AMP。
- 需要 block 输入/输出缓存、反向传播和最佳状态保存；200 步不等于固定的总耗时或显存成本。
- 论文不直接证明 W4A4、KV cache、embedding/lm-head 或其他激活格式的质量。

## 结果边界

- 论文主指标是 11 个 zero-shot 任务平均准确率，`Ours*` 是额外超参搜索得到的最好结果，不能和固定默认配置混用。
- 联合 clipping+rounding 并非每个模型和位宽都优于只优化其中一项。
- AutoRound 当前工具还包含 SignRound V2、WNA16、MXFP4、NVFP4、FP8 和 GGUF 等能力；这些不能自动归入论文 v5 的 SignRound 结论。
- 工具组合文档或导出成功不能证明目标 runtime 实际使用真实低比特 kernel。
- 当前 Wiki 没有运行 AutoRound 优化、导出或 GPU benchmark。

## Agent 如何使用

首轮读取本页即可判断目标是否需要 LLM weight-only 的舍入/范围联合优化。需要精确证据时再展开：

| 需要确认的内容 | 继续读取 |
| --- | --- |
| SignRound 公式、训练预算、论文结果和限制 | [`paper-autoround`](../../sources/papers/paper-autoround.md) |
| 当前 AutoRound 工具的实现、格式和组合声明 | [`repo-autoround`](../../sources/repos/repo-autoround.md) |
| 目标模型、导出器和 runtime 是否支持 | 目标模型源码、导出器和 runtime 独立验证 |

回答时必须说明是论文 SignRound 还是当前 AutoRound 工具，量化对象、位宽/分组、校准预算和是否做了额外超参搜索。

## 详细来源

- [SignRound 原论文来源页](../../sources/papers/paper-autoround.md)
- [AutoRound 官方源码来源页](../../sources/repos/repo-autoround.md)
