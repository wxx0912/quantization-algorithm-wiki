---
name: quantization-algorithm-wiki
description: 查询量化算法的机制、校准需求、适用条件、组合关系及部署证据，为 PTQ 方法理解和候选方案判断提供可追溯知识。
argument-hint: "[关键词] | [--tag 标签 --type 类型] | [page-id]"
---

# Quantization Algorithm Wiki

## 何时使用

需要理解量化方法、比较机制、核对校准统计、查阅组合前提或解释论文与部署结果时使用。本库提供知识依据；当前任务的 Recipe、执行、评测与迭代由调用方决定。

## 如何查询

以下命令在本 Skill 所在仓库根目录执行；路径相对于本文件。先用算法名、格式名、机制或短问题词查页，再按需要扩展相关知识。

```bash
python3 scripts/query.py "GPTQ"
python3 scripts/query.py --tag weight-reconstruction --type algorithm
python3 scripts/get_page.py algorithm-gptq
python3 scripts/get_page.py algorithm-gptq --follow-sources
```

检索先匹配 id、标题、aliases 和 tags，无结果才搜索正文并标为弱相关。多词采用 AND；长问题先提炼为短关键词。`--tag` 按词表筛选，忽略大小写；`--type` 限定页面类型，`--limit` 控制数量，`--json` 提供结构化结果。可用分类见 [生成索引](index.generated.md)。未命中时换关键词或标签；仍无相关资料则说明缺口，不把未命中解释成算法不可用。

方法概览先读算法页；格式问题读 foundations，选择或迁移问题读 guides，组合问题读 [组合页](wiki/recipes/composition.md)，性能证据读 [部署指南](wiki/guides/deployment-evidence.md) 或 cases。只展开与当前问题相关的页面。

需要公式、默认参数、实验数字或实现细节时，用 `--follow-sources` 展开登记的本地来源记录；它不会下载原文，也不会自动展开所有 related 页面。需要进一步核对原文时优先可获取的 LaTeX，其次 HTML，必要时补读 PDF；按实际阅读形式引用。

## 如何理解知识

- 区分论文报告、固定版本源码、工程推导与目标模型实测。论文数字保留相关模型、格式、数据和测试条件；静态源码不证明目标运行时命中。
- 页面级 confidence/evidence_scope 和编辑日期不代表每条结论已验证。没有依据的条件保持未知；本库知识不能替代目标模型上的数值与性能验证。
- 组合知识重点看修改对象、执行顺序、适用前提及可能失效的统计量。已验证组合仅在所列条件下成立；缺少组合证据也不等于数学不兼容。
- 分开理解质量、低比特存储、伪量化、真实 kernel 和端到端性能证据，不能从其中一项推导其他项已达标。

## 如何回答

按问题提供相关机制、必要条件、来源与边界。问原理时讲清计算过程；问组合时补充顺序与统计依赖；问目标适用性时说明依据与未验证项。无需每次输出完整 Recipe、消融或验收计划，也不把检索排名当作算法效果排名。

新增或修改知识时读 [AUTHORING](AUTHORING.md)；索引、校验与工具参数见 [README](README.md)，日常查询不需要运行维护流程。
