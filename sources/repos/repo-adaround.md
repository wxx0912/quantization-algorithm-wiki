---
{
  "id": "repo-adaround",
  "title": "quic/aimet — AdaRound 的维护实现（只学舍入参数 α）",
  "category": "repo",
  "url": "https://github.com/quic/aimet/tree/ec4bc578681beaa52043bc83a228d5e5a855dd15",
  "version": "ec4bc578681beaa52043bc83a228d5e5a855dd15",
  "version_kind": "commit",
  "retrieved_at": "2026-10-24",
  "read_form": [
    "repo-tree"
  ],
  "read_locator": "TrainingExtensions/torch/src/python/aimet_torch/_base/adaround/adaround_optimizer.py: AdaroundOptimizer.adaround_module、_optimize_rounding、_compute_recons_metrics、_compute_output_with_adarounded_weights、_can_cache_acts_data；.../adaround/adaround_loss.py: AdaroundLoss.compute_recon_loss、compute_round_loss、_compute_beta、AdaroundHyperParameters；.../adaround/adaround_weight.py: AdaroundParameters；.../adaround/adaround_wrapper.py: apply_adaround；TrainingExtensions/common/src/python/_aimet_common/defs.py: AdaroundConstants",
  "topic": "reconstruction",
  "excerpt_sha256": null
}
---
## 这份来源是什么

作者机构（Qualcomm）维护的 AdaRound 实现，用于核对“只学习舍入参数、scale/zero-point 由量化器提供”这一分工，以及 warm start 与 β 退火的具体调度，是 `algorithm-adaround` 的源码依据。

## 已读定位与阅读形式

固定 commit 下逐文件静态读取 Torch 侧 `_base/adaround/` 与常量定义；源码为本地快照（无 `.git`），commit 号取自 URL。未安装 AIMET，未运行量化或训练。

## 关键事实

- **只优化 `alpha`**：`_optimize_rounding` 建 `torch.optim.Adam([quant_module.alpha])`，不优化 scale、zero-point 或 clipping。`broadcasted_delta`/`broadcasted_offset` 取自权重量化器（`_get_weight_quantizer_delta_and_offset`）；`clip_max`/`clip_min` 则在 `_init_param` 中由量化器位宽推出——`clip_max = 2**bitwidth − 1`、`clip_min = 0`（`adaround_wrapper.py`），并非量化器直接给出。
- **软/硬舍入**：`use_soft_rounding=True` 时 `h(α)=clip(σ(α)(ζ−γ)+γ,0,1)`，`AdaroundConstants.ZETA=1.1`、`GAMMA=−0.1`；`False` 时 `h(α)=(α≥0)`，取恰好 0 或 1。`adaround_module` 优化前后各算一次软/硬重建误差，最后把 `use_soft_rounding` 置为 `False` 固化舍入。
- **组合损失**：重建项 `compute_recon_loss` 为逐样本 `‖ada_out − orig_out‖_F²` 再取均值，若该层后继有激活函数则先套 `act_func`；正则项 `compute_round_loss` 为 `reg_param · Σ(1 − |2h(α)−1|^β)`，且 `cur_iter < num_iterations·warm_start` 期间正则项直接为 0。
- **β 退火**：`_compute_beta` 在 warm start 结束后按余弦把 β 从 `beta_range[0]` 衰减到 `beta_range[1]`（`beta = end + 0.5(start−end)(1+cos(rel_iter·π))`）。
- **默认超参**（`AdaroundParameters`）：`default_reg_param=0.01`、`default_beta_range=(20, 2)`、`default_warm_start=0.2`；`default_num_iterations=None` 时按最低权重位宽取 10000（≥8-bit）或 15000（<8-bit）；`num_batches` 建议取 `min(len(data_loader), ceil(2000/batch_size))`。
- **支持层类型**：`_compute_output_with_adarounded_weights` 只处理 `Conv2d`、`ConvTranspose2d`、`Linear`，其他模块直接抛错。
- **工程实现**：模块级常量 `BATCH_SIZE=32`、`EMPIRICAL_THRESHOLD=3/4`；`_can_cache_acts_data` 决定是否把中间激活缓存到 CPU（再尽量放 GPU）；分布式下按 rank 切分数据集并把学习率乘 `world_size`。

## 不能推出的结论

- 这是持续维护的实现，与论文实验所用 commit 不同：AIMET 的默认 `reg_param`/`num_iterations`/`beta_range` 不能当作论文设置，也不能据此复现论文表格。
- 只有舍入被学习；scale/clip 的粒度与取值取决于所用量化方案（per-tensor/per-channel、静态/动态），本页不证明它们与论文设置一致。
- 激活缓存、分布式与 CPU/GPU 放置是 AIMET 的工程实现，不是算法方法本身，不能当作 AdaRound 的贡献。
- 未运行量化或训练，未核对本实现产出的模型精度；静态源码也不能证明目标 runtime 命中低比特 kernel。
