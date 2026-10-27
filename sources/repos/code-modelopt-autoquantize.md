---
{
  "id": "code-modelopt-autoquantize",
  "title": "NVIDIA/Model-Optimizer — AutoQuantize 实现（受成本约束的格式搜索）",
  "category": "repo",
  "url": "https://github.com/NVIDIA/Model-Optimizer/tree/11c2a85160d8b3a84e16e45759a842ce4fb1f13b/modelopt/torch/quantization",
  "version": "11c2a85160d8b3a84e16e45759a842ce4fb1f13b",
  "version_kind": "commit",
  "retrieved_at": "2026-10-23",
  "read_form": [
    "repo-tree"
  ],
  "read_locator": "model_quant.py: auto_quantize；algorithms.py: _get_auto_quantize_score、estimate_quant_compression；_auto_quantize_cost.py；_auto_quantize_shapley.py；kv_cache_auto_quant.py",
  "topic": "mixed-architecture",
  "excerpt_sha256": null
}
---
## 这份来源是什么

NVIDIA Model-Optimizer 的 AutoQuantize 实现。本页以实现为依据，未确定可覆盖整个接口的统一算法论文；仅 `aumann_shapley` 子路径引用了 **arXiv:2607.12266**（*Saturation Makes Quantization Error Additive: A Coverage Model with a Certificate*，v1，2026-07-14），该论文已核对。本条目是 `algorithm-modelopt-autoquantize` 的唯一来源：页面结论来自固定 commit 的静态源码核对与该论文的方法语义。

## 已读定位与阅读形式

固定 commit 下读取量化子包的源码与模块说明（该 commit 经 codeload 取 tarball 确认存在，源码在本地 checkout 中检索，未运行）。

## 关键事实

- `auto_quantize` 默认 `num_calib_steps=512`（batch 数）、`num_score_steps=128`；AWQ 候选使用 `max(1, num_calib_steps // 4)` 的校准步数。
- 默认 `cost_model="weight"`、`effective_bits` 默认 4.8；另有 `active_moe` 与 `kv_cache` 成本模式。
- 默认 gradient 评分为 `S=Σ(g⊙Δy)²`（逐元素乘积先做数值 clamp 再平方），不是 `(gᵀΔy)²`，也不是平均 Hessian trace。
- `kl_div` 路径采用阈值二分搜索，最小化最大分组分数的代理目标；评分按分组与候选做整网 forward。
- `aumann_shapley` 路径在 `y+t(Q(y)−y)` 的多个中间点收集 KL 梯度与扰动内积（默认 2 个 path nodes），并需检查 `damage_model.valid`、`approximation_flags`、`completeness`。
- 默认分组会把同注意力的 Q/K/V、若干 gate/up 与匹配命名的 MoE 专家约束为同一选择；`estimate_quant_compression` 优先使用 Recipe 显式 effective bits。

### 评分方法与实现

#### 默认 gradient

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

#### `kl_div`

对参考模型输出和一次只激活一个候选分组时的输出计算 KL，要求 forward 返回 logits，不需要标签和 backward。该实现采用**阈值二分搜索，最小化最大分组分数的代理目标**，不是 gradient 路径的最小总和 LP。评分会按分组和候选进行整网 forward，通常不应假设与 gradient 路径成本相同。[`AutoQuantizeKLDivSearcher::estimate_sensitivity_scores/run_search_with_stats`]

#### `aumann_shapley`

当前锁定源码还提供路径积分评分：在 $y+t(Q(y)-y)$ 的多个中间点收集 KL 梯度与扰动内积，避免仅在参考模型处计算 KL 梯度时零信号的问题；源码默认 2 个 path nodes（论文侧用 4 点中点法）。再以实测参考损伤校准分数和 `predicted_damage`。必须检查 `damage_model.valid`、`approximation_flags`、`completeness`，预测值不是部署 KL 上界。硬专家路由的不连续性可能使路径积分解释失效。[`_auto_quantize_shapley.py` 模块说明]

该方法支持 `max_predicted_damage`，与 `effective_bits` 约束互斥。已核对源码引用的论文 arXiv:2607.12266：其 coverage 模型 `f(S)=c(1−Π(1−a_i))` 与附录 B 的 Aumann–Shapley 路径积分，与源码 `damage=c(1−exp(−Σb))` 的校准语义一致；论文在路径积分时冻结 MoE 路由（argmax 不可微），与源码「硬路由不连续可能使路径积分失效」的提示相符。预测精度本身未在本项目独立测量，也不能据此认为该方法已实测优于另外两条路径。

### 成本口径

默认 `cost_model="weight"`，`effective_bits` 默认 4.8。以分组参数量 $N_i$、其候选有效位宽 $b_{i,r}$ 概括，指标类似 $\sum_iN_ib_{i,r_i}/\sum_iN_i$；实际以源码中纳入计账的模块、固定基线和格式 override 为准。

成本模式分三类：`weight` 按可量化权重元素累计压缩成本；`active_moe` 对 routed expert 按 active experts / total experts 比例加权；`kv_cache` 使用成对 K/V 格式和独立 forward-KL 评分。它们都不能直接承诺进程总显存、激活/KV/workspace 大小、真实文件字节数、稀疏路由加速或端到端延迟。

`estimate_quant_compression` 优先使用 Recipe/量化器显式 effective bits，再用整数位数或浮点字段推断。源码明确其每项取最小值的估计仍会低估混合 W/A 格式的激活成本。因而 `effective_bits=4.8` 既不是所有层 W4.8，也不是包括所有 runtime 内存的精确预算。[`algorithms.py::estimate_quant_compression`; `_auto_quantize_cost.py`]

### 配置使用条件

- 候选可包含 INT4 AWQ、FP8、NVFP4 等配置，具体粒度和 scale 来自配置定义，不能从 AutoQuantize 名称推断；必须同时锁定候选配置和导出后端。
- QKV 与专家分组服务融合 kernel 约束；不能把内部搜索单位称为任意独立层。也不能根据模块名称外观推测分组成功，需检查实际 state。
- 候选配置中的全模型名称排除项（例如 `*lm_head*`）不一定按普通 `quantize` 语义生效：API 说明候选逐层匹配，需用 `disabled_layers` 等明确限制。
- 分数依赖实际前向、损失归约、数据、序列长度和模型状态。对重放有副作用的模块，需要核实方法能否正确评分；不能把包含变化中缓存的重复 forward 默认为纯函数。
- checkpoint 可复用已算分数并重新求预算，但只有模型、候选、分组、数据与评分语义一致时，旧证据才有意义。改过缩放/旋转/权重后应重新校准与评分。
- 反向用于敏感度估计，不自动等于 QAT。默认流程不以优化器更新原模型来拟合任务；候选校准算法若含局部优化，需分别描述其更新对象。


## 不能推出的结论

- 未找到该机制的统一论文披露（不等于不存在），不能为它虚构统一算法论，也不能把源码语义当作已实测结论。
- `effective_bits` 不是精确的 runtime 内存或延迟预算；每项取最小值的估计会低估混精 W/A 格式的激活成本。
- `predicted_damage` 不是部署 KL 上界；硬专家路由的不连续性可能使路径积分解释失效。
- 未运行量化或部署实验；Shapley 路径引用的理论来源已核对（arXiv:2607.12266），预测精度未在本项目独立测量。
