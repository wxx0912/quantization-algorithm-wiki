---
{
  "id": "repo-svdquant",
  "title": "mit-han-lab/deepcompressor — SVDQuant 官方实现与配置",
  "category": "repo",
  "url": "https://github.com/mit-han-lab/deepcompressor/tree/69f3473f5e1c1504bae35cc50c7858ef900a9b17",
  "version": "69f3473f5e1c1504bae35cc50c7858ef900a9b17",
  "version_kind": "commit",
  "retrieved_at": "2026-10-24",
  "read_form": [
    "repo-tree"
  ],
  "read_locator": "deepcompressor/nn/patch/lowrank.py: LowRankBranch.reset_parameters、forward；deepcompressor/calib/lowrank.py: QuantLowRankCalibrator；examples/diffusion/configs/svdquant/__default__.yaml、examples/diffusion/configs/__default__.yaml",
  "topic": "transforms",
  "excerpt_sha256": null
}
---
## 这份来源是什么

SVDQuant 官方实现与配置，用于核对低秩分支的构造、迭代校准与默认预算，是 `algorithm-svdquant` 的源码依据。

## 已读定位与阅读形式

本轮读取固定 commit 的 GitHub 源码归档，核对归档根目录所标 commit 与本页版本一致；静态读取所列函数与配置，未运行。

## 关键事实

- `LowRankBranch.reset_parameters` 用 double 精度 `torch.linalg.svd` 保存截断因子；`forward` 在默认 `rank>0` 时为两次 Linear（`rank==0` 的分支 `forward` 返回 `None`，表示无分支输出、`rank<0` 为单个 Linear）。
- `QuantLowRankCalibrator._ask` 对 `self.w − self.qw` 构造低秩分支并量化剩余权重；误差最佳候选的保留与 `early_stop` 判断在 `_tell` 里完成，`_ask` 只负责生成候选。
- diffusion 默认配置的**校准** `num_samples: 128`（评测为 5000）；SVDQuant 配置为 rank 32、`num_iters: 100`、`early_stop: true`、smoothing 网格搜索 `num_grids: 20`。

## 数据流与工程使用

- `QuantLowRankCalibrator` 以 `OutputsError` 为可选目标；默认 SVDQuant 配置实际选择它，不能把所有搜索都解释成单纯最小化权重 Frobenius 范数。
- `reset_parameters` 将 PyTorch `[out,in]` 权重分解为 `UΣVᵀ`，输入先经 `Vᵀ` 对应的 Linear，再经 `UΣ` 对应的 Linear；源码存储方向与论文 `Y=XW` 的矩阵方向不同。
- 工程上修改 smoothing 或 rank 后需要重建低秩因子、残差和量化 scales；旧残差不能与截断后的新分支直接配对。

## 不能推出的结论

- 配置默认值不等于论文所有结果使用同一预算，实际运行配置需合并模型/格式/精度配置后确定。
- 未实测 SVD 分解与候选评估的耗时和显存，也未运行量化或 kernel。
