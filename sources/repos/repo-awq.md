---
{
  "id": "repo-awq",
  "title": "mit-han-lab/llm-awq — 官方实现（scale 比率搜索与 auto_clip）",
  "category": "repo",
  "url": "https://github.com/mit-han-lab/llm-awq/tree/d6e797a42b9ef7778de8ee2352116e0f48a78d61",
  "version": "d6e797a42b9ef7778de8ee2352116e0f48a78d61",
  "version_kind": "commit",
  "retrieved_at": "2026-10-24",
  "read_form": [
    "repo-tree"
  ],
  "read_locator": "awq/quantize/auto_scale.py: auto_scale_block、_search_module_scale、get_act_scale、get_weight_scale、apply_scale；awq/quantize/auto_clip.py: auto_clip_layer、auto_clip_block、apply_clip；awq/quantize/quantizer.py: pseudo_quantize_tensor；README（fake/real 两条路径）",
  "topic": "reconstruction",
  "excerpt_sha256": null
}
---
## 这份来源是什么

AWQ 官方实现，用于核对 scale 搜索的候选集合、评价口径与落地方式，以及 local clipping 的比较对象与跳过范围，是 `algorithm-awq` 的源码依据。

## 已读定位与阅读形式

固定 commit 下逐文件静态读取 `awq/quantize/` 的 `auto_scale.py`、`auto_clip.py`、`quantizer.py` 与 README；源码为本地快照（无 `.git`），commit 号取自 URL。未克隆运行，未执行 AWQ 搜索或量化。

## 关键事实

- **scale 比率搜索**（`auto_scale_block._search_module_scale`）：`n_grid=20`，候选 `ratio = k/20 (k=0…19)`，即 0.00–0.95；`scales = x_max.pow(ratio).clamp(min=1e-4)`，再用 `(scales.max()*scales.min()).sqrt()` 归一化，其中 `x_max` 是逐输入通道的激活绝对均值（`get_act_scale`）。每个候选先乘到被搜索的线性权重上、用 `pseudo_quantize_tensor` 伪量化再除回，评价指标是被检查模块（`module2inspect`，可能是 decoder layer，也可能是 `self_attn`/`mlp` 等子模块）输出的 `(org_out − out)²` 均值；候选间恢复 block 权重，取 MSE 最小者。
- **scale 落地**（`apply_scale`）：按前置算子分流——`nn.Linear` 走 `scale_fc_fc`、`LayerNorm`/`LlamaRMSNorm`/`Qwen2RMSNorm` 走 `scale_ln_fcs`、`GELU`/`BloomGelu`/`GELUActivation`/`SiLU` 走 `ScaledActivation` + `scale_gelu_fc`，其他类型抛 `NotImplementedError`。scale 逐输入通道，乘到下游线性权重、其逆写回前置算子（并同步缩放缓存的输入激活）。
- **local clipping**（`auto_clip_layer`，参数 `n_grid=20`、`max_shrink=0.5`、`n_sample_token=512`）：权重 reshape 成 `[co, 1, n_group, group_size]` 并对激活按约 512 token 抽样，输出通道按 256/64 分批以防 OOM；对每个（输出通道 × 分组）在 `max_val = org_max·(1 − k/20) (k=0…9)` 中搜对称阈值（`min_val = −max_val`），按局部输出 `(cur_out − org_out)²` 选最优。
- **跳过 Q/K**（`auto_clip_block`）：名称含 `q_`、`k_`、`query`、`key`、`Wqkv` 的线性层不裁剪，源码注释为“due to qk bmm, it is hard to clip precisely”。
- **量化器口径**（`pseudo_quantize_tensor`）：非对称 zero-point 分支 `scales = (max_val − min_val).clamp(min=1e-5) / max_int`、`zeros = clamp(−round(min_val/scales), 0, max_int)`；对称分支（`zero_point=False`）注释为“we actually never used this”。“只支持 zero_point”的约束不在量化函数里，而在 `real_quantize_model_weight`：`assert q_config["zero_point"], "We only support zero_point quantization now."`。
- **fake 与 real 两条路径**：README 用 `--q_backend fake` 做模拟伪量化评测，用 `--q_backend real --dump_quant` 导出真实 INT4 权重；W4A16 的 CUDA kernel 在 `awq/kernels`，部署经 TinyChat。

## 不能推出的结论

- 候选集合与搜索区间（ratio 0–0.95、clip 上限 0.5；scale 网格 20 档，clip 实际评估 `int(0.5×20)=10` 档）是实现细节，不能反过来当作论文已测试的端点或论文实验配置。
- `auto_clip_block` 跳过 Q/K 是实现取舍（qk bmm 使逐层局部误差代理不成立），不代表这些投影不能被裁剪，也不代表论文使用了相同规则。
- 静态源码不证明目标 runtime 接受其 packing 或启用低比特 kernel；fake 与 real 两条路径的产出不同，不能互相替代或混用。
- 未运行 fake/real quant，也未核对导出产物与后端格式的一致性；README 的 TinyChat 速度声明不是本仓库实测。
