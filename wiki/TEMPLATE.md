# 算法页面模板

以下头部以真实字段替换后放在算法页面最开始。`related` 只填写已存在的页面 ID，`sources` 必须登记对应来源页。`confidence` 与 `evidence_scope` 取值见 `CONVENTIONS.md` 第 2.5 节。

```json
{
  "id": "algorithm-name",
  "title": "算法名称",
  "type": "algorithm",
  "tags": ["weight-reconstruction"],
  "aliases": ["中文别名"],
  "related": [],
  "sources": ["paper-name", "repo-name"],
  "confidence": "source-reported",
  "evidence_scope": "mixed",
  "updated": "2026-10-23"
}
```

## 问题与核心机制

具体解决什么误差，修改哪些变量，保留什么不变。定义符号与张量布局。

## 计算流程与公式

按输入、校准、变换或优化、量化、导出的顺序解释。区分数学理想形式与实际近似实现。

## 数据、配置与成本

校准样本、统计量、粒度、W/A/KV、优化预算和原文适用模型。原文设置不等于当前任务的推荐默认值。

## 前提、局限与方法关系

数学前提、结构吸收条件、已知失败、组合先后、统计失效和需独立验证的迁移假设。

## 实现与部署

官方仓库、固定 revision、函数入口、导出和 kernel 要求。区分已读源码与未验证 runtime。

## 原文实验与可证伪实验

带条件的原文结果；为 Agent 提供能区分原因的消融，不把候选建议当成结论。

## 来源与证据边界

给出具体定位，并列出当前尚未确认的部分。本项目若未跑数值实验，明确说明。
