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
    "自动混精"
  ],
  "related": [
    "algorithm-hawq",
    "guide-architecture-transfer"
  ],
  "sources": [
    "code-modelopt-autoquantize"
  ],
  "confidence": "verified",
  "evidence_scope": "source-reported",
  "updated": "2026-10-23"
}
---

# ModelOpt AutoQuantize：受成本约束的格式搜索

AutoQuantize 是 **候选量化 Recipe 的评分与分配机制**，不是一种新数值格式。它先按各候选算法校准，再评估每个模块或运行时分组的量化敏感度，受成本预算约束选择格式。格式可以携带量化器、校准算法和参数，不能把它简化为“从 INT4 与 INT8 中选 bit”。

本文基于 NVIDIA 官方 Model-Optimizer commit `11c2a85160d8b3a84e16e45759a842ce4fb1f13b`，核实日期 2026-10-23。主要依据是源码，不为该机制虚构一篇统一算法论文，也未运行量化/部署实验。[API 与实现](https://github.com/NVIDIA/Model-Optimizer/blob/11c2a85160d8b3a84e16e45759a842ce4fb1f13b/modelopt/torch/quantization/model_quant.py#L590)

## 计算流程与产物

1. 提供模型、可重复迭代的数据、候选格式、约束和 forward 回调；默认 gradient 方法还需要任务损失或 forward/backward 回调。
2. 把可量化模块转换为带候选 Recipe 的模块，并按融合执行约束分组。默认规则会把同注意力的 Q/K/V、若干 gate/up 以及匹配命名的 MoE 专家约束为同一选择。
3. 分别校准候选，保存量化器状态和必要变换。接口默认 `num_calib_steps=512`，是 **batch 数**，不是固定 512 条样本；当前源码对 AWQ 候选还使用 `num_calib_steps // 4` 的校准步数，不能把 512 当成每个候选的实际步数。
4. 用 `num_score_steps` 批数据估计各分组、各候选的敏感度，默认 128 批。评分函数由 `method` 决定。
5. 在允许的候选与成本约束内搜索，将选择应用到模型，返回 `(model, state_dict)`，后者含候选分数、成本和搜索信息。
6. 单独验证最终全模型质量，再导出到支持相应混精格式的后端。API 返回不是“部署成功”的证据。

`module_search_spaces` 可限制特定模块组的候选；`allow_no_quant=False` 可排除最终选择中的不量化选项。默认权重搜索会加入不量化候选，但专用 KV 搜索中 BF16 仅作评分参考，不是可选解。`fixed_quantization_config` 定义未搜索模块的固定量化基线，并计入成本与评分上下文。[`model_quant.py::auto_quantize`]

## 评分方法不能混写

### 默认 gradient

设分组输出为 $y_i$，候选量化造成输出变化 $\Delta y_{i,r}$，任务梯度为 $g_i=\partial L/\partial y_i$。源码 `_get_auto_quantize_score` 累加

$$
S_{i,r}=\sum_{\text{batch, elements}}(g_i\odot\Delta y_{i,r})^2.
$$

它是**逐元素乘积平方后求和**，不是 $(g_i^T\Delta y_{i,r})^2$，也不是 HAWQ 的平均 Hessian trace。实现对乘积先做数值 clamp，再平方。`AutoQuantizeGradientSearcher` 把它解释为 Taylor/Fisher 近似；对似然损失有相应动机，对其他任务只是代理。[`algorithms.py`, `_get_auto_quantize_score`, `AutoQuantizeGradientSearcher`]

若用二元选择变量 $z_{i,r}$，其搜索目标可写为：

$$
\min\sum_{i,r}z_{i,r}S_{i,r},\quad
\sum_rz_{i,r}=1,\quad
\sum_{i,r}z_{i,r}C_{i,r}\le B.
$$

实现调用 `LPS` 约束求解，并可能尝试不同预算下界；求解不满足时会警告。因此仍需检查状态和实际成本，不能仅检查函数是否返回。分数是分组代理和，不是全网络真实损失的精确可加分解。

### `kl_div`

对参考模型输出和一次只激活一个候选分组时的输出计算 KL，要求 forward 返回 logits，不需要标签和 backward。该实现采用**阈值二分搜索，最小化最大分组分数的代理目标**，不是 gradient 路径的最小总和 LP。评分会按分组和候选进行整网 forward，通常不应假设与 gradient 路径成本相同。[`AutoQuantizeKLDivSearcher::estimate_sensitivity_scores/run_search_with_stats`]

### `aumann_shapley`

当前锁定源码还提供路径积分评分：在 $y+t(Q(y)-y)$ 的多个中间点收集 KL 梯度与扰动内积，避免仅在参考模型处计算 KL 梯度时零信号的问题；默认 2 个 path nodes。再以实测参考损伤校准分数和 `predicted_damage`。必须检查 `damage_model.valid`、`approximation_flags`、`completeness`，预测值不是部署 KL 上界。硬专家路由的不连续性可能使路径积分解释失效。[`_auto_quantize_shapley.py` 模块说明]

该方法支持 `max_predicted_damage`，与 `effective_bits` 约束互斥。本文核实其实现语义，未独立验证源码引用的理论论文或预测精度；不能把它当作已经实测优于另外两条路径。

## 成本到底约束了什么

默认 `cost_model="weight"`，`effective_bits` 默认 4.8。以分组参数量 $N_i$、其候选有效位宽 $b_{i,r}$ 概括，指标类似 $\sum_iN_ib_{i,r_i}/\sum_iN_i$；实际以源码中纳入计账的模块、固定基线和格式 override 为准。

| 成本模式 | 实际含义 | 不能据此承诺 |
|---|---|---|
| `weight` | 按可量化权重元素数累计压缩成本；格式可显式声明 effective bits | 进程总显存、激活/KV/workspace 大小、真实文件字节数或延迟 |
| `active_moe` | 对 routed expert 权重按 active experts / total experts 比例加权；shared experts 不这样折减 | 未激活专家不需存储，或稀疏路由实测加速 |
| `kv_cache` | 专用成对 K/V 格式与缓存成本统计，分派到独立 forward-KL 评分 | 纯权重搜索已自动优化 KV，或任何模型都支持此路径 |

`estimate_quant_compression` 优先使用 Recipe/量化器显式 effective bits，再用整数位数或浮点字段推断。源码明确其每项取最小值的估计仍会低估混合 W/A 格式的激活成本。因而 `effective_bits=4.8` 既不是所有层 W4.8，也不是包括所有 runtime 内存的精确预算。[`algorithms.py::estimate_quant_compression`; `_auto_quantize_cost.py`]

## 格式、数据与实现限制

- 候选可包含 INT4 AWQ、FP8、NVFP4 等配置，具体粒度和 scale 来自配置定义，不能从 AutoQuantize 名称推断；必须同时锁定候选配置和导出后端。
- QKV 与专家分组服务融合 kernel 约束；不能把内部搜索单位称为任意独立层。也不能根据模块名称外观推测分组成功，需检查实际 state。
- 候选配置中的全模型名称排除项（例如 `*lm_head*`）不一定按普通 `quantize` 语义生效：API 说明候选逐层匹配，需用 `disabled_layers` 等明确限制。
- 分数依赖实际前向、损失归约、数据、序列长度和模型状态。对重放有副作用的模块，需要核实方法能否正确评分；不能把包含变化中缓存的重复 forward 默认为纯函数。
- checkpoint 可复用已算分数并重新求预算，但只有模型、候选、分组、数据与评分语义一致时，旧证据才有意义。改过缩放/旋转/权重后应重新校准与评分。
- 反向用于敏感度估计，不自动等于 QAT。默认流程不以优化器更新原模型来拟合任务；候选校准算法若含局部优化，需分别描述其更新对象。

## 已核实实现入口

均位于上述固定 commit 的 `modelopt/torch/quantization/`：

| 文件/符号 | 已读内容 |
|---|---|
| `model_quant.py::auto_quantize` | API、默认值、评分分派、候选与固定基线语义 |
| `algorithms.py::_AutoQuantizeBaseSearcher` | 分组、候选校准、成本汇总与约束求解 |
| `algorithms.py::_get_auto_quantize_score` | gradient 精确评分表达式 |
| `algorithms.py::AutoQuantizeKLDivSearcher` | 单组扰动 KL 与阈值二分 |
| `_auto_quantize_cost.py` | 权重与 active-MoE 成本模型 |
| `_auto_quantize_shapley.py` | 路径积分、损伤拟合和适用性标志 |
| `kv_cache_auto_quant.py` | 专用 K/V 成本约束、候选校验及搜索入口 |

## 组合与 Agent 验证

AutoQuantize 应在合法候选集合内做选择；AWQ 等是某些候选的量化算法，不是每个候选后面可随意追加的一步。后处理如果改权重或 scale，搜索时的分数和成本可能不再代表最终模型。

最小消融应固定数据、分组与预算，比较统一格式、gradient 搜索、KL 搜索；同时记录实际预算达成值、全模型任务指标、未量化比例、导出成功率。更换成本模型时分别报告存储成本与 active-MoE 成本，不把两者排序直接比较。验证后端时测目标 batch、序列长度和硬件上的真实端到端表现。

本文不提供未经核实的 Nemotron 量化收益数字，也没有独立运行该版本的质量实验。第一版证据是锁定源码机制与约束；具体模型效果仍需案例页或实测支持。参见 [架构迁移](../guides/architecture-transfer.md) 与 [来源登记](../../sources/topics/mixed-architecture.md)。
