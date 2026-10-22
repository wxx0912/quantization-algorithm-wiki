# v0.1 知识与检索验收

日期：2026-10-23。以下是本仓库的实际 CLI 检索观察，不是模型量化实验。命令在仓库根目录执行；索引命令输出首批匹配页面及来源 ID，使用稳定 ID 后再读取具体段落和来源登记。

| 情景 | 检索命令与命中页面 | 证据与回答边界 | 尚未知 |
| --- | --- | --- | --- |
| GPTQ 与 AWQ 如何比较和选择 | `python3 scripts/query.py "GPTQ AWQ" --limit 10`；命中 `algorithm-awq`、`guide-algorithm-selection`、`recipe-composition`；通过 selection 的关联读取 `algorithm-gptq` | GPTQ页解释输入Gram驱动的逐列重构补偿；AWQ页解释激活重要性缩放与剪裁。selection页指出需固定模型、格式、group、校准集、排除层、导出及评测后端。组合页保留 AWQ v5 Table 9 的 OPT-6.7B/WikiText-2/INT2-g64 组合证据，不能外推为普遍胜出。来源 `paper-gptq`、`paper-awq`。 | 目标模型实测误差、耗时、工具入口和后端表现。 |
| 缩放或旋转后重采什么 | `python3 scripts/query.py "旋转 统计" --limit 10`；命中 `recipe-composition`、`algorithm-quarot`、相关变换与重构页面（完整排名见下方） | 组合页给出输入坐标变化后的 Gram 关系，并区分单纯线性变换与前序量化造成的输入变化；权重范围、量化 scale/clipping、后续输入、block误差应按状态更新或重新验证。来源 `paper-quarot`、`paper-smoothquant`、`paper-gptq`。 | 具体目标图能否解析吸收变换、需要重采样的层和数据规模。 |
| AR 上有效的方法迁移到 dLLM 缺什么 | `python3 scripts/query.py "AR dLLM mask" --limit 10`；命中 `guide-architecture-transfer` | 页面给出 mask 比例与生成阶段校准覆盖、masked-token 损失、attention/缓存契约、轨迹控制、逐样本质量与部署证据清单。LLaDA v2 是生成过程证据，不是量化效果证据。来源 `paper-llada-architecture`。 | 目标 dLLM 的量化直接论文/实验、mask轨迹与有效后端证据。 |
| 论文 W4A4 是否足以证明目标后端加速 | `python3 scripts/query.py "W4A4 加速" --limit 10`；命中 `guide-deployment-evidence`、`foundation-quantization-contract`、`algorithm-quarot` | 页面区分数值方案、打包存储、正确性、实际kernel路径、计时和任务验收；QuaRot 单 Transformer block 和 SVDQuant融合案例说明性能范围须按原文保留。来源 `paper-quarot`、`paper-svdquant`、`repo-nunchaku`。 | 目标硬件、负载、kernel 命中、fallback、端到端质量/吞吐。 |
| 两个方法看似互补，怎样判断合法 Recipe | `python3 scripts/query.py "组合 顺序" --type recipe --limit 10`；命中 `recipe-composition` | 按操作对象、优化变量、浮点等价、执行顺序、变换后统计、计算图、格式/导出/后端逐项审查；AWQ+GPTQ为有限条件的论文验证，SmoothQuant+任意GPTQ导出器只是迁移假设。 | 对具体架构和工具的组合入口、导出正确性与目标质量/速度实测。 |

## 实际命令结果摘要

以下记录为根据实际命令输出整理的完整命中 ID 顺序；来源追踪使用 `python3 scripts/get_page.py <稳定ID> --follow-sources`，正文包括定位信息。

```text
GPTQ AWQ: guide-algorithm-selection, algorithm-awq, algorithm-autoround, algorithm-flatquant, algorithm-omniquant, foundation-quantization-contract, guide-architecture-transfer, recipe-composition
旋转 统计: algorithm-gptq, algorithm-quarot, algorithm-spinquant, recipe-composition, algorithm-autoround, algorithm-awq, algorithm-baselines, algorithm-flatquant, algorithm-hawq, algorithm-modelopt-autoquantize
AR dLLM mask: guide-architecture-transfer, algorithm-spinquant, algorithm-awq, algorithm-flatquant, algorithm-autoround, algorithm-brecq, algorithm-smoothquant, algorithm-svdquant, foundation-quantization-contract
W4A4 加速: guide-deployment-evidence, foundation-quantization-contract, algorithm-flatquant, algorithm-svdquant, algorithm-brecq, algorithm-quarot, case-paper-evidence
组合 顺序: recipe-composition
```

“未命中”或结果顺序只反映词面检索评分，不构成算法适用性判断。验证程序检查元数据、引用和内部链接；来源链接需要人工追读，工具不会验证论文含义。

## 完整目标审计

| 目标 | 交付与验收依据 |
| --- | --- |
| 阅读与结构借鉴 | design.md 记录关联结构库与 KernelWiki 的采用范围；保留稳定 ID、来源登记与渐进读取 |
| 六层知识组织 | index.md 按基础、机制、算法、架构、Recipe、实现案例导航，20 篇知识页面 |
| 核心覆盖 | 14 篇算法页覆盖目标全部代表方法；ViDiT-Q 在架构指南深入展开；ModelOpt 明确作为源码机制，不虚构统一论文 |
| 计算深度 | 各算法页包含计算步骤、变量/公式、数据预算、对象格式、实现定位、边界和消融；人工交叉审查修正版本与实现语义 |
| 组合边界 | composition.md 区分论文组合、迁移假设和明确冲突，给出状态顺序与统计失效规则 |
| 原始证据 | 32 条来源登记，保留论文版本和代码 commit；性能分开描述存储、模拟、kernel 与计时范围 |
| 导航与机器读取 | README、index、生成索引、来源入口、模板、Agent指南及标准库 CLI；不重复维护参数数据库 |
| 五项问题验收 | 本页记录真实查询和来源展开；词面检索不等于算法判断 |
| 工具验证 | 20 项 unittest 通过；20 页面/32 来源、内部引用与链接校验零错误；生成索引一致 |
| 范围与公开边界 | 仅建设知识库，无模型/训练/GPU实验；未写入内部参考文档或内部实验，关联知识库保持只读 |

## 已知限制与后续优先级

文本 dLLM 尚缺此版本核实的直接量化实验；目标设备的导出、kernel 命中与端到端收益仍需执行证据。部分论文与当前官方实现配置不同，必须按页面分别使用。后续先补直接证据与具体部署案例，再扩充算法数量。当前格式检查不验证外网链接，也不证明研究结论正确。
