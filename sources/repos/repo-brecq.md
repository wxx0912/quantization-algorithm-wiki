---
{
  "id": "repo-brecq",
  "title": "yhhhli/BRECQ — 官方实现（block_recon 与 LossFunction）",
  "category": "repo",
  "url": "https://github.com/yhhhli/BRECQ/tree/e455d62e93c70351961f8991c913b59435bd165f",
  "version": "e455d62e93c70351961f8991c913b59435bd165f",
  "version_kind": "commit",
  "retrieved_at": "2026-10-23",
  "read_form": [
    "repo-tree"
  ],
  "read_locator": "README、quant/block_recon.py: block_reconstruction、LossFunction",
  "topic": "reconstruction",
  "requires_external_verification": true,
  "excerpt_sha256": null
}
---

# yhhhli/BRECQ — 官方实现（block_recon 与 LossFunction）

## 这份来源是什么

BRECQ 官方实现，用于核对默认优化预算与 Fisher 模式的实际数值口径，是 `algorithm-brecq` 的源码依据。

## 已读定位与阅读形式

固定 commit 下读取 GitHub 原始文件，未克隆运行。

## 关键事实

- `block_reconstruction` 默认 `batch_size=32`、`iters=20000`、`opt_mode='mse'`；权重阶段把 quantizer 换成 `AdaRoundQuantizer` 优化 `alpha`，激活阶段优化 `delta`。
- `LossFunction` 支持 `mse`、`fisher_diag`、`fisher_full`，只有非 MSE 模式才走 `save_grad_data`；`fisher_full` 令 `a=|ẑ−z|`、`g=|g|`，按样本先算 `Σ(a⊙g)` 再乘 `a⊙g`，取 `.mean()/100` 作为代理（"与论文式 (10) 的对角 Fisher 不同"是论文侧断言，本页未联网回原文核验）。

## 不能推出的结论

- 模式名 `fisher_full` 不代表精确完整 Fisher 二次型，不能仅凭名称声称复现论文式 (10)。
- 复现记录必须写明实际 `opt_mode`，并区分函数默认值与实际启动参数。
- README 的 ImageNet 示例不能推广为所有网络结构的覆盖范围；未运行该实现。
