---
{
  "id": "algorithm-modelopt-autoquantize",
  "title": "ModelOpt AutoQuantize：受成本约束的格式搜索",
  "type": "algorithm",
  "tags": [
    "mixed-precision",
    "search",
    "fisher",
    "modelopt",
    "cost-constraint"
  ],
  "aliases": [
    "AutoQuantize",
    "auto_quantize",
    "自动混精",
    "混合精度"
  ],
  "related": [
    "algorithm-hawq",
    "guide-architecture-transfer"
  ],
  "sources": [
    "code-modelopt-autoquantize"
  ],
  "confidence": "source-reported",
  "evidence_scope": "mixed",
  "updated": "2026-10-24"
}
---

# ModelOpt AutoQuantize：受成本约束的格式搜索

## 一句话理解

AutoQuantize 在用户提供的量化 Recipe 中，为模块或模块组选择格式，使敏感度代理与成本预算之间取得折中。它是搜索接口，不是一种数值格式；本页依据固定版本官方源码。

## 核心机制

先校准候选，再评分、分配。默认 `gradient` 评分累加 $(g\odot\Delta y)^2$，不等于 HAWQ 的 Hessian trace。另有整网输出 `kl_div` 评分和 `aumann_shapley` 路径积分评分；三者的数据、计算成本和搜索目标不能混写。

## 基本流程

1. 提供模型、候选配置、数据、forward 回调和成本约束；gradient 路径还需损失或 backward 回调。
2. 按实际模块和融合约束分组，校准候选并估计敏感度。
3. 在预算内选择 Recipe，检查返回状态和实际成本。
4. 独立评估整模型质量，再导出至支持该混精配置的后端。

## 适用条件与结果边界

- 候选可携带 AWQ 等校准算法；搜索之后随意再改权重或 scale，会使原评分失效。
- `weight` 成本是量化权重成本，`active_moe` 是按专家活跃比例计账，`kv_cache` 是单独的 K/V 搜索路径。它们不等于总显存或真实延迟。
- 默认校准和评分步数指 batch 数，不能直接解释成样本数。
- 模块分组、固定基线和排除项必须检查实际配置，不能假定所有层独立搜索。
- 搜索代理小不保证全模型任务损失小，导出成功不保证后端使用目标 kernel。

## Agent 如何使用

当需要在多个合法量化配置间自动分配格式时读取本页；需要评分公式、默认参数、成本与函数细节时展开 [源码来源](../../sources/repos/code-modelopt-autoquantize.md)。回答应同时给出版本、候选集、评分方法、成本模型、预算和质量验证要求。当前没有本项目的量化或部署实测。
