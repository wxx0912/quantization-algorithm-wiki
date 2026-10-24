---
{
  "id": "repo-quarot",
  "title": "spcl/QuaRot — 官方参考实现（fake quant 与旋转工具）",
  "category": "repo",
  "url": "https://github.com/spcl/QuaRot/tree/5008669b08c1f11f9b64d52d16fddd47ca754c5a",
  "version": "5008669b08c1f11f9b64d52d16fddd47ca754c5a",
  "version_kind": "commit",
  "retrieved_at": "2026-10-23",
  "read_form": [
    "repo-tree"
  ],
  "read_locator": "README、fake_quant/main.py、fake_quant/rotation_utils.py",
  "topic": "core",
  "requires_external_verification": true,
  "excerpt_sha256": null
}
---

# spcl/QuaRot — 官方参考实现（fake quant 与旋转工具）

## 这份来源是什么

QuaRot 官方参考实现，用于核对旋转构造与量化编排的实际做法，是 `algorithm-quarot` 的源码依据。

## 已读定位与阅读形式

固定 commit 下读取 GitHub 原始文件，未克隆运行。

## 关键事实

- `rotation_utils.py` 提供旋转矩阵构造与把旋转吸收进权重/归一化的工具。
- `fake_quant/main.py` 组织权重与激活的伪量化流程，属于参考实现而非部署 kernel。
- 仓库说明中提到真实量化与 kernel 路径与伪量化入口是不同层次。

## 不能推出的结论

- 伪量化路径不证明整数 GEMM 或真实低比特推理。
- 未运行 CUDA 或任何模型实验；参考实现与论文实验版本可能有差异。
