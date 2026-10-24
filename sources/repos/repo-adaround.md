---
{
  "id": "repo-adaround",
  "title": "quic/aimet — AdaRound 的维护实现",
  "category": "repo",
  "url": "https://github.com/quic/aimet/tree/ec4bc578681beaa52043bc83a228d5e5a855dd15",
  "version": "ec4bc578681beaa52043bc83a228d5e5a855dd15",
  "version_kind": "commit",
  "retrieved_at": "2026-10-23",
  "read_form": [
    "repo-tree"
  ],
  "read_locator": "TrainingExtensions/torch/src/python/aimet_torch/_base/adaround/adaround_optimizer.py: AdaroundOptimizer.adaround_module、_optimize_rounding",
  "topic": "reconstruction",
  "requires_external_verification": true,
  "excerpt_sha256": null
}
---

# quic/aimet — AdaRound 的维护实现

## 这份来源是什么

作者机构维护的 AdaRound 实现，用于核对软/硬舍入的比较方式与最终固化步骤，是 `algorithm-adaround` 的源码依据。

## 已读定位与阅读形式

固定 commit 下读取 GitHub 原始文件，未克隆运行。

## 关键事实

- `adaround_module` 比较软/硬舍入误差并调用 `_optimize_rounding`，结束后设置 `use_soft_rounding=False`。
- 该实现由维护方持续更新，与论文实验所用 commit 不同。

## 不能推出的结论

- 具体默认迭代数与 wrapper 支持随 AIMET 版本变化，不能当作论文设置。
- 非论文原始实验版本，不能据此复现表 5 数值；未运行该实现。
