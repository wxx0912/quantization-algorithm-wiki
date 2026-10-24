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
  "requires_external_verification": true,
  "excerpt_sha256": null
}
---

# NVIDIA/Model-Optimizer — AutoQuantize 实现（受成本约束的格式搜索）

## 这份来源是什么

NVIDIA Model-Optimizer 的 AutoQuantize 实现。该机制没有可引用的统一算法论文（仅 `aumann_shapley` 子路径在源码里引用了一篇 arXiv 论文，本页未联网核验），本条目是 `algorithm-modelopt-autoquantize` 的唯一来源：页面结论全部来自固定 commit 的静态源码核对。

## 已读定位与阅读形式

固定 commit 下读取量化子包的源码与模块说明（该 commit 由 `git ls-remote` 确认，源码在本地 checkout 中检索，未运行）。

## 关键事实

- `auto_quantize` 默认 `num_calib_steps=512`（batch 数）、`num_score_steps=128`；AWQ 候选使用 `num_calib_steps // 4` 的校准步数。
- 默认 `cost_model="weight"`、`effective_bits` 默认 4.8；另有 `active_moe` 与 `kv_cache` 成本模式。
- 默认 gradient 评分为 `S=Σ(g⊙Δy)²`（逐元素乘积先做数值 clamp 再平方），不是 `(gᵀΔy)²`，也不是平均 Hessian trace。
- `kl_div` 路径采用阈值二分搜索，最小化最大分组分数的代理目标；评分按分组与候选做整网 forward。
- `aumann_shapley` 路径在 `y+t(Q(y)−y)` 的多个中间点收集 KL 梯度与扰动内积（默认 2 个 path nodes），并需检查 `damage_model.valid`、`approximation_flags`、`completeness`。
- 默认分组会把同注意力的 Q/K/V、若干 gate/up 与匹配命名的 MoE 专家约束为同一选择；`estimate_quant_compression` 优先使用 Recipe 显式 effective bits。

## 不能推出的结论

- 未找到该机制的统一论文披露（不等于不存在），不能为它虚构统一算法论，也不能把源码语义当作已实测结论。
- `effective_bits` 不是精确的 runtime 内存或延迟预算；每项取最小值的估计会低估混精 W/A 格式的激活成本。
- `predicted_damage` 不是部署 KL 上界；硬专家路由的不连续性可能使路径积分解释失效。
- 未运行量化或部署实验；Shapley 路径引用的理论来源与预测精度未独立验证。
