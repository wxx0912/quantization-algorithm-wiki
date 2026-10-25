---
{
  "id": "algorithm-dllmquant",
  "title": "DLLMQuant：扩散语言模型的迭代感知 PTQ",
  "type": "algorithm",
  "tags": [
    "dllm",
    "calibration",
    "activation",
    "quantizer",
    "mask"
  ],
  "aliases": [
    "DLLMQuant",
    "dLLM 量化",
    "扩散语言模型量化"
  ],
  "related": [
    "guide-architecture-transfer",
    "algorithm-gptq",
    "algorithm-awq",
    "algorithm-quarot",
    "guide-search-space"
  ],
  "sources": [
    "paper-dllmquant"
  ],
  "confidence": "source-reported",
  "evidence_scope": "source-reported",
  "updated": "2026-10-25"
}
---

# DLLMQuant：扩散语言模型的迭代感知 PTQ

## 一句话理解

DLLMQuant 把 DLLM 的三个机制特点转化为量化设计：用 TMAS 校准覆盖"不同解码步 × 不同 mask 比例"的数据分布，用 IA-AQ 抑制迭代中放大最快的误差路径，用 CGQ 让权重量化补偿只看"会被解码、且高置信"的 token。它默认叠在已有的 PTQ 前端（AWQ 或 QuaRot+GPTQ）之上。

## 核心机制

1. **TMAS（校准）**：按 block 数与总步数分层，在每个 block 内的不同 mask ratio 区间按比例采样，使校准集覆盖跨 timestep 的分布漂移；论文称该采样能恢复 INT4 模型大部分性能。
2. **IA-AQ（激活）**：定位误差主因是 attention 中 softmax 输出与 value 矩阵相乘这一步；改用"以 softmax 输出加权"的误差度量来定 value 的量化 scale，并在 `α∈{1.0,0.8}` 内搜索，把精度让给交互强的 token。
3. **CGQ（权重）**：GPTQ 类方法用统一 Hessian，而 DLLM 中已解码/低分 token 的误差不进入下一轮；CGQ 用 mask 指示与 token 分数对 Hessian 加权，使补偿集中在会影响下一步解码的区域。

## 基本流程

1. 准备 DLLM 的迭代解码与 block 划分，按 TMAS 采样出覆盖各 mask 比例与步数区间的校准集。
2. 在选定前端（AWQ 或 QuaRot+GPTQ）上叠加 IA-AQ 与 CGQ；权重对称 per-channel、激活非对称 per-token。
3. 用 DLLM 自身评测协议（多任务 + 生成类任务）验证，而不是只看 PPL。

## 适用条件

- 面向**掩码迭代解码**的 DLLM；论文验证 LLaDA-8B、LLaDA-1.5-8B、DREAM-7B。
- 依赖对解码过程的理解：TMAS 需要 block 数与步数，CGQ 需要 mask 状态与 token 分数，IA-AQ 需要定位 attention 的 softmax×V 路径。
- 前端方法与后端 kernel 仍由部署侧决定；本方法不改变"能否在目标后端跑低比特"。

## 结果边界

- 论文报告（LLaDA，W4A4，九任务平均）：FP 59.87、RTN 44.23、AWQ 48.09、QuaRot 51.03；AWQ+DLLMQuant 49.26、QuaRot+DLLMQuant 54.29；GSM8K 由 QuaRot 的 44.57 升到 56.25。
- 消融显示三项技术叠加才达到最好（GPTQ 51.03 → +TMAS 52.36 → +CGQ 53.16 → +IA-AQ 54.29）。
- ">1.6× 加速 / >3.2× 显存节省"是论文在其实现与 A6000 上的报告值，不等于任意后端的端到端收益。
- 结论限于文本 DLLM；不能外推到 AR LLM 或图像/视频 DiT。参数（`α`、TMAS 比例）属启发式，换数据集需重确。
- 当前 Wiki 未复现实验，也无可核验的官方代码来源页。

## Agent 如何使用

处理 dLLM 量化、尤其是 LLaDA 系列掉点时，先读本页，再按需展开：

| 需要确认的内容 | 继续读取 |
| --- | --- |
| 三项技术、Alg.1、Eq.6–10 与 Table 1–4 | [`paper-dllmquant`](../../sources/papers/paper-dllmquant.md) |
| 前端方法（AWQ、QuaRot+GPTQ、GPTQ）的机制与边界 | [AWQ](awq.md)、[QuaRot](quarot.md)、[GPTQ](gptq.md) |
| 文本 dLLM 与 AR/DiT/VLM 的迁移差异 | [架构迁移](../guides/architecture-transfer.md) |
| 生成感知量化在搜索空间中的位置 | [量化搜索空间](../guides/search-space.md) |

回答时要区分"校准采样改进"、"激活路径改进"与"权重补偿改进"三类贡献，并区分准确率与吞吐/显存证据。

## 详细来源

- [DLLMQuant 原论文来源页](../../sources/papers/paper-dllmquant.md)
