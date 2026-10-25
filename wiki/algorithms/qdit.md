---
{
  "id": "algorithm-qdit",
  "title": "Q-DiT：扩散 Transformer 的细粒度与动态量化",
  "type": "algorithm",
  "tags": [
    "dit",
    "vision",
    "diffusion",
    "format",
    "search"
  ],
  "aliases": [
    "Q-DiT",
    "DiT PTQ",
    "扩散 Transformer 量化"
  ],
  "related": [
    "guide-architecture-transfer",
    "algorithm-gptq",
    "guide-search-space"
  ],
  "sources": [
    "paper-qdit"
  ],
  "confidence": "source-reported",
  "evidence_scope": "source-reported",
  "updated": "2026-10-25"
}
---

# Q-DiT：扩散 Transformer 的细粒度与动态量化

## 一句话理解

Q-DiT 针对 DiT 的三处结构特点——权重沿输入通道方差大、激活在特定通道有结构性离群、激活分布随 timestep 漂移——用"沿输入通道的分组量化 + 运行时动态激活 scale + 组大小的进化搜索"来降低量化误差。它是针对生成过程的量化，选择依据是最终生成质量（FID），不是逐层 MSE。

## 核心机制

1. **细粒度分组量化**：权重与激活按输入通道方向切组，同一层权重与激活使用同一 group size。论文认为"沿输出通道"的常规做法在 DiT 上会吃亏。
2. **动态激活量化**：运行时按每组的 min-max 现算 scale，并把 min-max 计算融合进前一个算子；论文称融合后开销相对 matmul 可忽略。
3. **组大小自动分配**：用进化算法逐层搜索 group size，目标函数是量化模型生成样本与真实样本间的 FID，约束是 BitOps 预算；论文发现"group 越小越好"在 DiT 上不成立。

## 基本流程

1. 选定默认 group 与搜索空间，先做一版基线量化（论文默认 group 128，搜索空间 {32,64,128,192,288}）。
2. 对候选 group 配置量化并生成样本，用 FID 评估，进化搜索更新候选。
3. 取最优配置产出量化模型；激活走动态量化路径。

## 适用条件

- 论文验证对象是 DiT-XL/2（ImageNet 256×256 / 512×512），权重用 GPTQ、激活动态量化。
- 搜索目标依赖"能生成样本并算 FID"，需要真实数据、采样器与评测脚本；不适合无法生成/无法评分的场景。
- 变换与量化都绑定具体 DiT 实现；换 DiT 变体、换采样器或换分辨率需重新确认结构假设与搜索空间。

## 结果边界

- 论文报告（ImageNet 256×256，100 步、无 CFG）：W8A8 下 Q-DiT FID 11.93，优于 GPTQ 16.44、RepQ-ViT 13.19、PTQ4DM 20.91；W4A8 下 15.76，优于 GPTQ 25.48、RepQ-ViT 315.85。这些是生成分布指标，不是任务正确率。
- W4A8 的 FID（15.76）仍高于 FP 的 12.40，不能读成"无损"；高分辨率 512×512 的 W8A8 并非最优（FID 16.23，略逊 RepQ-ViT 16.01），优势主要在 W4A8 段。
- 自动搜索以 FID 为目标，换指标/数据会改变最优配置；搜索本身有额外算力成本。
- 动态激活量化的"几乎无开销"依赖算子融合，不能外推到未融合的后端；论文未证明目标后端存在低比特 kernel。
- 当前 Wiki 未运行该量化或复现其 FID。

## Agent 如何使用

首轮读取本页即可判断问题是否涉及 DiT 的权重/激活分布与 timestep 漂移。需要精确证据时再展开：

| 需要确认的内容 | 继续读取 |
| --- | --- |
| 四条观察、三技术公式、Table 1–3 数值 | [`paper-qdit`](../../sources/papers/paper-qdit.md) |
| 生成感知量化在搜索空间中的位置 | [量化搜索空间](../guides/search-space.md) |
| 与文本 dLLM 的差异及迁移前要补的证据 | [架构迁移](../guides/architecture-transfer.md) |
| 目标实现、采样器与后端 kernel | 目标模型源码与 runtime 独立验证 |

回答 DiT 量化问题时，要区分"生成质量指标（FID）"与"任务正确率"，也要区分"搜索得到的配置"与"可迁移的通用规则"。

## 详细来源

- [Q-DiT 原论文来源页](../../sources/papers/paper-qdit.md)
