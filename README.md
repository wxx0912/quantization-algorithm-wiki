# Quantization Algorithm Wiki

面向研究人员与 Coding Agent 的量化算法知识库，为 AnyPTQ 的候选方法选择、Recipe 组织和实验设计提供可追溯依据。当前以 PTQ 为主，说明算法的具体计算、修改变量、校准需求、组合边界和实现证据。

v0.1 已收录 14 篇算法与基线页面，覆盖 RTN/范围校准、GPTQ、AdaRound、AutoRound、AWQ、SmoothQuant、QuaRot、SpinQuant、FlatQuant、BRECQ、OmniQuant、SVDQuant、HAWQ 和 ModelOpt AutoQuantize。另有数值契约、算法选择、架构迁移、部署证据、组合 Recipe 与论文案例导航。

这是知识与检索工具的交付。论文结果、固定版本源码核对、迁移假设分别标明；本项目没有运行模型量化、训练或 GPU benchmark，也不宣称所有 Recipe 已能在任意后端执行。来源按“一来源一文件”登记在 `sources/papers/` 与 `sources/repos/`，每篇带版本锚、阅读定位与“不能推出的结论”；页面用 `confidence`/`evidence_scope` 标注整体证据强度。

## 阅读入口

- [按问题与机制阅读](index.md)：算法、组合、架构与案例导航。
- [完整页面与标签索引](index.generated.md)：由页面元数据生成。
- [量化术语与数值契约](wiki/foundations/quantization-contract.md)。
- [Agent 入口](SKILL.md)：何时使用、查询路径、证据边界与输出契约。
- [来源层入口](sources/README.md) 与[来源页规范](sources/SPEC.md)、[页面模板](wiki/TEMPLATE.md)、[维护约定](CONTRIBUTING.md)。

与 [Model Architecture Wiki](https://github.com/wxx0912/model-architecture-wiki) 协作：结构库描述组件和计算路径，本库描述算法如何作用于这些组件及其证据边界。先取得具体结构和诊断信息，再检索本库；不能由模型名称自动确定最佳量化方法。

## 本地检索

需要 Python 3.10+，只使用标准库。在仓库根目录运行：

```bash
python3 scripts/query.py "GPTQ"
python3 scripts/query.py "统计" --type recipe
python3 scripts/query.py --tag rotation --json
python3 scripts/query.py --alias 怎么选算法
python3 scripts/get_page.py algorithm-gptq --follow-sources
```

检索覆盖名称、别名、标签与正文；空格分隔词采用 AND，名称/别名优先排序。它不是语义问答：长问题未命中时使用较短关键词；`--tag` 使用索引中的小写标签精确匹配。稳定 ID 用于读取，`related` 和 `sources` 用于逐步展开，避免一次加载全库。`--alias <别名键>` 用词表登记的问题级别名展开到页面 ID，未登记的键报错而不是静默返回空。

## 维护与验证

```bash
python3 scripts/build_index.py
python3 scripts/validate.py
python3 scripts/build_index.py --check
python3 -m unittest discover -s tests -v
python3 scripts/check_shared.py --peer ../model-architecture-wiki
```

校验覆盖页面元数据、ID、引用关系、本地链接及基础标题锚点；不验证外网可达性或算法正确性。内容仍需核对原论文与官方源码。来源阅读优先 LaTeX 源码，其次 HTML，必要时补看 PDF；已完成的 HTML 阅读按实际记录，不改写为已读 LaTeX。`check_shared.py` 比对两库共享文件（`CONVENTIONS.md`、`sources/SPEC.md`、`scripts/` 与 `tests/test_shared_tools.py`）的哈希，改共享文件须同步两库。

后续优先补齐文本 dLLM 的直接量化证据、目标后端的具体适配与真实实验案例。当前没有数据库、向量服务、量化执行器或自动兼容判定器。
