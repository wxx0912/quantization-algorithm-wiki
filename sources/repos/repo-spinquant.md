---
{
  "id": "repo-spinquant",
  "title": "facebookresearch/SpinQuant — 官方实现（旋转训练与 PTQ 流程）",
  "category": "repo",
  "url": "https://github.com/facebookresearch/SpinQuant/tree/8f47aa3f00e8662caf1a484153920a07e5281c3a",
  "version": "8f47aa3f00e8662caf1a484153920a07e5281c3a",
  "version_kind": "commit",
  "retrieved_at": "2026-10-23",
  "read_form": [
    "repo-tree"
  ],
  "read_locator": "optimize_rotation.py: train；train_utils/optimizer.py: Cayley_loop、SGDG；ptq.py: train",
  "topic": "transforms",
  "requires_external_verification": true,
  "excerpt_sha256": null
}
---

# facebookresearch/SpinQuant — 官方实现（旋转训练与 PTQ 流程）

## 这份来源是什么

SpinQuant 官方实现，用于核对旋转参数的构造、约束优化实现与 PTQ 入口，是 `algorithm-spinquant` 的源码依据。

## 已读定位与阅读形式

固定 commit 下读取 GitHub 原始文件，未克隆运行。

## 关键事实

- `optimize_rotation.py::train` 冻结原参数，构造 `R1`/`R2`，用 WikiText 数据与 Trainer 输出旋转字典。
- `train_utils/optimizer.py` 的 `Cayley_loop`/`SGDG` 实现正交约束下的更新。
- `ptq.py::train` 调用 PTQ 流程与 PPL 评测。
- 当前实现会解除 `tie_word_embeddings` 并复制输出头，影响参数存储。

## 不能推出的结论

- 文件名与函数名不代表重新训练全部权重，也不是普通 SGD。
- 未运行校准、导出或 kernel benchmark；同名脚本的默认参数不等于论文 v2 配置。
- 当前源码晚于论文，复现需要明确脚本参数与模型版本。
