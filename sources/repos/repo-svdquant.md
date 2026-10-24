---
{
  "id": "repo-svdquant",
  "title": "mit-han-lab/deepcompressor — SVDQuant 官方实现与配置",
  "category": "repo",
  "url": "https://github.com/mit-han-lab/deepcompressor/tree/69f3473f5e1c1504bae35cc50c7858ef900a9b17",
  "version": "69f3473f5e1c1504bae35cc50c7858ef900a9b17",
  "version_kind": "commit",
  "retrieved_at": "2026-10-23",
  "read_form": [
    "repo-tree"
  ],
  "read_locator": "deepcompressor/nn/patch/lowrank.py: LowRankBranch.reset_parameters、forward；deepcompressor/calib/lowrank.py: QuantLowRankCalibrator；examples/diffusion/configs/svdquant/__default__.yaml、examples/diffusion/configs/__default__.yaml",
  "topic": "transforms",
  "requires_external_verification": true,
  "excerpt_sha256": null
}
---

# mit-han-lab/deepcompressor — SVDQuant 官方实现与配置

## 这份来源是什么

SVDQuant 官方实现与配置，用于核对低秩分支的构造、迭代校准与默认预算，是 `algorithm-svdquant` 的源码依据。

## 已读定位与阅读形式

固定 commit 下读取 GitHub 原始文件与配置，未克隆运行。

## 关键事实

- `LowRankBranch.reset_parameters` 用 double 精度 `torch.linalg.svd` 保存截断因子；`forward` 在默认 `rank>0` 时为两次 Linear（`rank==0` 退化为恒等、`rank<0` 为单个 Linear）。
- `QuantLowRankCalibrator._ask` 对 `self.w − self.qw` 构造低秩分支并量化剩余权重；误差最佳候选的保留与 `early_stop` 判断在 `_tell` 里完成，`_ask` 只负责生成候选。
- diffusion 默认配置的**校准** `num_samples: 128`（评测为 5000）；SVDQuant 配置为 rank 32、`num_iters: 100`、`early_stop: true`、smoothing 网格搜索 `num_grids: 20`。

## 不能推出的结论

- 配置默认值不等于论文所有结果使用同一预算，实际运行配置需合并模型/格式/精度配置后确定。
- 未实测 SVD 分解与候选评估的耗时和显存，也未运行量化或 kernel。
