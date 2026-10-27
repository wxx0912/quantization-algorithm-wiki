---
{
  "id": "repo-brecq",
  "title": "yhhhli/BRECQ — 官方实现（block_recon 与 LossFunction）",
  "category": "repo",
  "url": "https://github.com/yhhhli/BRECQ/tree/e455d62e93c70351961f8991c913b59435bd165f",
  "version": "e455d62e93c70351961f8991c913b59435bd165f",
  "version_kind": "commit",
  "retrieved_at": "2026-10-24",
  "read_form": [
    "repo-tree"
  ],
  "read_locator": "README、quant/block_recon.py: block_reconstruction、LossFunction",
  "topic": "reconstruction",
  "excerpt_sha256": null
}
---
## 这份来源是什么

BRECQ 官方实现，用于核对默认优化预算与 Fisher 模式的实际数值口径，是 `algorithm-brecq` 的源码依据。

## 已读定位与阅读形式

本轮读取固定 commit 的源码归档，归档顶层目录确认版本；静态阅读，未执行训练。

## 关键事实

- `block_reconstruction` 默认 `batch_size=32`、`iters=20000`、`opt_mode='mse'`；权重阶段把 quantizer 换成 `AdaRoundQuantizer` 优化 `alpha`，激活阶段优化 `delta`。
- `LossFunction` 支持 `mse`、`fisher_diag`、`fisher_full`，只有非 MSE 模式才走 `save_grad_data`；`fisher_full` 令 `a=|ẑ−z|`、`g=|g|`，按样本先算 `Σ(a⊙g)` 再乘 `a⊙g`，取 `.mean()/100` 作为代理；本轮对照 v2 LaTeX 的 `eq_obj`，确认其不同于论文的对角 Fisher 目标。

## 不能推出的结论

- 模式名 `fisher_full` 不代表精确完整 Fisher 二次型，不能仅凭名称声称复现论文式 (10)。
- 复现记录必须写明实际 `opt_mode`，并区分函数默认值与实际启动参数。
- README 的 ImageNet 示例不能推广为所有网络结构的覆盖范围；未运行该实现。

## 实现补充

`asym` 默认 False，控制是否用量化前缀输入重构浮点输出；默认值不代表启动脚本实参。权重阶段 Adam 优化 `alpha`，结束设 `soft_targets=False`；激活阶段单独优化 `delta`，使用余弦学习率调度，不启用舍入正则。论文 Algorithm 1 写在同一循环中更新激活步长，不能用论文伪码替代当前实现阶段说明。
