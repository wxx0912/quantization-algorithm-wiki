# 按问题与机制阅读

先读 [数值契约与术语](wiki/foundations/quantization-contract.md)，确定 W/A/KV、粒度、动态/静态参数和部署目标；类别与可搜索变量见 [量化搜索空间](wiki/guides/search-space.md)。格式、算法、工具与搜索策略分开看，不能互相替代。

## 从问题到方案

1. [量化搜索空间：算法类别与可搜索变量](wiki/guides/search-space.md)：先定格式与部署路径，再沿搜索轴取候选。
2. [如何比较 GPTQ 与 AWQ、诊断掉点](wiki/guides/algorithm-selection.md)。
3. [缩放/旋转后的统计失效与合法组合](wiki/recipes/composition.md)：含 QuaRot+GPTQ、AWQ+GPTQ、OmniQuant LET+LWC 的证据与实施约束。
4. [AR、MoE、dLLM、DiT、多模态迁移](wiki/guides/architecture-transfer.md)：包含 ViDiT-Q 的公式、轨迹与混精案例。
5. [W4A4 是否意味着加速](wiki/guides/deployment-evidence.md)：区分存储、伪量化、kernel、局部与整模性能。
6. [论文案例与实现证据导航](wiki/cases/paper-evidence.md)：按证据范围阅读，具体数值保留在算法页。

## 检索入口

- [index.generated.md](index.generated.md)：全部页面的稳定 ID、类型与**按算法类别与搜索轴的分组**（生成物，勿手改）。
- `python3 scripts/query.py "<关键词>"`：按名称、别名（aliases）或问题短语检索，结果按类型分组；没有命中时才退回正文并标为弱相关。
- `python3 scripts/query.py --tag <标签>`：按机制标签检索，标签词表见 `data/vocabulary.yaml`；非法标签会直接报错。
- `python3 scripts/validate.py`：一次校验页面契约、引用与链接；索引是否过期由 `build_index.py --check` 负责。
- 来源清单见 [来源索引](sources/INDEX.generated.md)，检索方式见 [SKILL.md](SKILL.md)。
