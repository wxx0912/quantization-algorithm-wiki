---
{
  "id": "recipe-composition",
  "title": "算法组合：前提、顺序与统计失效",
  "type": "recipe",
  "tags": [
    "composition",
    "calibration",
    "rotation",
    "scaling"
  ],
  "aliases": [
    "算法组合",
    "统计失效",
    "重新校准",
    "组合顺序",
    "合法 Recipe"
  ],
  "related": [
    "algorithm-awq",
    "algorithm-gptq",
    "algorithm-quarot",
    "algorithm-omniquant",
    "algorithm-svdquant"
  ],
  "sources": [
    "paper-awq",
    "paper-gptq",
    "paper-quarot",
    "paper-omniquant",
    "paper-svdquant"
  ],
  "confidence": "source-reported",
  "evidence_scope": "mixed",
  "updated": "2026-10-23"
}
---

# 算法组合：前提、顺序与统计失效

组合是对模型状态依次执行有条件的操作。判断兼容性应看作用对象、修改变量、数学前提和部署契约，不能把算法名作无约束笛卡尔积。

## 已有证据与研究假设

| 组合 | 证据状态 | 顺序与边界 |
| --- | --- | --- |
| QuaRot → GPTQ → 在线 A/KV 量化 | 论文有验证 | 浮点融合和旋转先完成，GPTQ 使用变换后的输入；保留必要在线 Hadamard 与缓存规则 |
| AWQ + GPTQ | 论文有验证 | AWQ v5 Table 9 的 OPT / WikiText-2 / INT2-g64；不是所有格式和模型通用支持 |
| OmniQuant LET + LWC | 联合目标与消融有证据 | 在同一 block 目标下共同优化，不能随意拆成分别优化后拼接 |
| SVDQuant 低秩分支 + 低比特残差 | 论文与实现有证据 | 平滑、分解、残差量化按方法要求配套；性能依赖融合而非普通两次分支调用 |
| 任意 SmoothQuant 实现 + 任意 GPTQ 导出器 | 条件性迁移假设 | 需确认共享输入的变换、重采统计、量化对象、导出布局与后端；本 Wiki 不声明已有通用适配 |
| 已量化权重直接再送入另一个重构器 | 尚未定义的候选 | 可能二次量化、覆盖优化或丢失浮点信息；需明确新的目标与原始状态，不能默认叠加收益 |

没有组合论文不等于数学不兼容；论文存在组合实验也不等于目标后端已经实现。

## Recipe A：QuaRot + GPTQ

**输入状态：** 浮点模型、已核对的 norm/残差/attention/head 结构、校准集、旋转种子、目标 W/A/KV 与后端。

**顺序：** 融合 norm 参数 → 配对旋转权重与残差流 → 配置无法离线吸收的在线变换 → 在新前向中采集 GPTQ 统计 → 权重重构 → 配置 A/KV 量化与导出。

**修改对象：** 权重和计算图坐标先变化，随后离散权重值变化；A/KV 的在线量化器最后与目标 kernel 对齐。原模型 scale、Gram 和敏感度报告不能直接复用。输出需包含变换与缓存约定，而不是仅低比特权重。

**准入：** 浮点变换等价检查；新统计的来源和坐标一致；部署支持在线变换、缓存与实际精度。原文速度针对单 Transformer block，不能直接用作整模型吞吐承诺。[QuaRot v2 §4–5](https://arxiv.org/html/2404.00456v2#S4)

## Recipe B：AWQ + GPTQ

**证据：** [AWQ v5 Table 9](https://arxiv.org/html/2306.00978v5) 比较 OPT / WikiText-2 / INT2-g64。OPT-6.7B：FP16 PPL 10.86，GPTQ 16.65，AWQ+GPTQ 15.71。支持在这一条件下缩放与二阶重构互补，不支持“所有设置都优于单算法”。

**实现检查：** 锁定 AWQ 缩放与 clipping 配置、具体 GPTQ 量化器和组合入口；确认在变换后的模型上生成对应重构统计。这个顺序是保持统计一致所需的实施要求，不能仅凭 Table 9 反推出所有公开工具已有同一执行入口。

**部署检查：** AWQ 产物与 GPTQ 产物可能采用不同 packing、zero-point 和排列；组合优化后的权重需通过一个明确导出器输出，不能直接拼接两个 checkpoint 元数据。

## Recipe C：OmniQuant LET + LWC

**输入：** 浮点权重、block 校准输入/参考、LET 初始化、bit/group 配置、训练预算。

**顺序：** 初始化 LET/LWC → 同一 fake-quant block 前向 → 重构损失联合反向 → 冻结学习参数 → 融合变换/保存量化参数 → 后续 block 与部署验证。LWC 适应 LET 改变后的权重范围，两个模块不是相互独立的最优解。[OmniQuant v3 §3](https://arxiv.org/html/2308.13137v3#S3)

**失效规则：** 修改 LET 初始化、训练目标、校准状态或 group 后，旧 clipping 最优值与损失记录不再证明新候选有效；重新训练或受控验证，不能直接沿用旧质量结论。

## 什么统计会失效

对行样本布局，缩放后 `X'=XD⁻¹`，输入 Gram 满足 `G'=D⁻ᵀGD⁻¹`；旋转后 `X'=XR`，`G'=RᵀGR`。这些是同一批输入、纯局部线性变换下的代数关系，不能用于掩盖前序量化造成的输入变化。

| 改动 | 应重采、严格变换或重新验证的产物 |
| --- | --- |
| 缩放、旋转、置换 | 输入统计坐标、权重范围、量化 scale/clipping、重构参数 |
| 前序层量化或分组改变 | 后续实际输入、敏感度、block 重构与 logits 结果 |
| 校准数据或生成阶段改变 | 静态范围、重要性、专家覆盖、轨迹统计 |
| bit/格式/导出器/后端改变 | packing、zero-point、数值容差、kernel 命中、性能 |

默认重采最直接。只有存在显式统计变换并验证输入未发生其他变化，才有依据复用。经验案例也需绑定上述依赖，不能在配置变化后继续称为同一实验。

## 明确冲突的例子

- 缩放了共享激活但只补偿一个消费者，破坏浮点等价。
- 把一般旋转穿过 SiLU 或 RoPE 而没有配对推导，不能声称等价。
- 更改量化 scale 后沿用 AdaRound 原网格的硬舍入决策，原优化目标不再成立。
- 目标 kernel 要求融合组统一格式而候选逐模块任意分配，部署契约冲突。

最后一项需从具体 kernel 证据确认，不能凭“可能融合”就禁止某组合。本页是候选审查规则，不是已运行的自动兼容检查器。
