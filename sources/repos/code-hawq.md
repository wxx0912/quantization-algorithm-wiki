---
{
  "id": "code-hawq",
  "title": "Zhen-Dong/HAWQ — 官方实现（V3 为主，含训练循环）",
  "category": "repo",
  "url": "https://github.com/Zhen-Dong/HAWQ/tree/1616df69fdd99f100a8d8a6e78742e0a89292634",
  "version": "1616df69fdd99f100a8d8a6e78742e0a89292634",
  "version_kind": "commit",
  "retrieved_at": "2026-10-23",
  "read_form": [
    "repo-tree"
  ],
  "read_locator": "README、quant_train.py: train、train_kd",
  "topic": "mixed-architecture",
  "requires_external_verification": true,
  "excerpt_sha256": null
}
---

# Zhen-Dong/HAWQ — 官方实现（V3 为主，含训练循环）

## 这份来源是什么

HAWQ 官方实现，用于确认论文流程中的训练边界（是否有反向传播与优化器步进）以及当前仓库的版本状态，是 `algorithm-hawq` 的源码依据。

## 已读定位与阅读形式

固定 commit 下读取 GitHub 原始文件，未克隆运行。

## 关键事实

- `quant_train.py::train`/`train_kd` 包含 backward 与 `optimizer.step()`，与论文结果含微调的结论一致。
- README 当前主展示 HAWQ-V3 的模型与 TVM 推理结果。
- README 说明所发布 checkpoint 以浮点保存，真正缩小文件还涉及权重整数打包。

## 不能推出的结论

- 不能声称当前默认训练命令精确复现 V2；README 的 V3 表格不能代替 V2 论文实验。
- 论文大小数字、checkpoint 文件大小与运行显存是三件不同的事。
- 未运行训练或推理。
