---
name: quantization-algorithm-wiki
description: 查询 PTQ、权重重构、激活感知缩放、旋转变换、联合优化、低秩补偿和混合精度方法，为量化方案选择、Recipe 设计与部署证据核查提供可追溯资料。
argument-hint: "[自然语言问题] | [--tag 标签 --type 类型] | [page-id]"
---

# Quantization Algorithm Wiki

## 何时使用

当任务需要选择量化算法、解释算法计算过程、核对校准数据和统计量、判断方法组合、设计量化消融，或区分论文结果与真实部署证据时使用。它不替代目标模型源码检查、实际量化运行、训练、kernel 编译或 GPU benchmark。

## 查询路径

```bash
python3 scripts/query.py "<关键词>" --limit 10
python3 scripts/query.py --tag <标签> --type <algorithm|foundation|guide|recipe|case>
python3 scripts/query.py "<关键词>" --json
python3 scripts/get_page.py <page-id>
python3 scripts/get_page.py <page-id> --follow-sources
python3 scripts/validate.py
python3 scripts/build_index.py --check
```

先用方法名、别名、机制标签或短问题词缩小范围，再读取一至三篇页面，并沿 `related` 和 `sources` 展开。空格分隔的关键词采用 AND；查询是词面检索，不是语义问答。检索先匹配页面 id、标题、`aliases` 与 `tags`，没有命中再退回正文并标为弱相关，结果按 `type` 分组；`--tag` 按 `data/vocabulary.yaml` 的小写标签精确匹配，未登记的标签报错。问题短语写在页面 `aliases` 里，本库没有集中映射表。

## 证据和论文阅读顺序

优先阅读可获取的论文 LaTeX 源码，其次是 HTML，PDF 用于补充图表、排版或没有 LaTeX 的内容。来源页记录实际阅读形式、固定论文版本、源码 commit 和已读定位。

回答时区分：

1. **论文报告**：保留模型、数据、位宽、粒度、硬件和评测条件；不能把特定实验推广成通用结论。
2. **源码支持**：保留 commit、路径和实际配置；静态代码存在不等于目标运行时命中。
3. **本项目实测**：当前 Wiki 只验证页面、来源、索引和查询工具，没有运行模型量化或 GPU 实验。
4. **迁移假设**：说明推导依据和缺失证据；AR、MoE、dLLM、DiT 与 VLM 之间不能自动互相迁移。

页面级 `confidence` 与 `evidence_scope` 只描述整体证据强度与范围，不能覆盖章节级差异；`updated` 只是编辑日期，不是算法实测认证。没有来源或无法定位的数字应标记为未知，不用默认配置补齐。

## 适用性与组合检查

形成候选前先确认目标模型结构、W/A/KV 对象、位宽、分组、scale/zero-point、校准数据、生成阶段、计算预算和后端。随后检查：

- 方法修改的是权重、量化器、激活、计算图还是额外补偿分支；
- 变换后的输入坐标、范围、scale、clipping、Gram/Hessian、敏感度和教师输出是否需要重新采集；
- 组合方法是否优化同一变量，执行顺序是否明确，浮点等价是否保持；
- 导出布局、packing、kernel、fallback 和端到端计时是否与论文设置一致。

`wiki/recipes/composition.md` 中标记为“论文有验证”的组合只在所列条件下成立；“迁移假设”不能直接当成可执行 Recipe。未找到组合证据不等于数学不兼容。

停止条件：浮点变换不等价时先检查共享消费者与图变换，不要继续量化；参考量化与部署不一致时先核对数值/导出契约；只有质量结果或只有存储/伪量化证据时不得承诺速度或低比特加速；缺少模型、格式或后端关键条件时不得把方案标为可执行。

## 输出契约

回答量化问题时说明：

1. 候选方法及其具体计算和修改对象；
2. 所需校准数据、统计量、预算和格式粒度；
3. 适用前提、已知失败条件和证据范围；
4. 方法关系、执行顺序和可能失效的产物；
5. 官方实现入口与版本边界；
6. 尚未验证的目标模型、后端或性能问题；
7. 能区分不同解释的消融与验收门槛。

建议按此顺序组织回答：问题与目标 → 候选和依据 → 必要顺序与失效统计 → 实现与部署条件 → 未知项 → 消融与验收门槛。只有质量结果时不能承诺速度；只有 fake quant 或存储结果时不能承诺真实低比特计算。任何目标模型上的最终结论都必须由独立执行和评测工具验证。
