---
{
  "id": "repo-smoothquant",
  "title": "mit-han-lab/smoothquant — 官方平滑实现",
  "category": "repo",
  "url": "https://github.com/mit-han-lab/smoothquant/tree/c61476d728e42ae0d8a35e7e78494edcac3237b5",
  "version": "c61476d728e42ae0d8a35e7e78494edcac3237b5",
  "version_kind": "commit",
  "retrieved_at": "2026-10-23",
  "read_form": [
    "repo-tree"
  ],
  "read_locator": "smoothquant/smooth.py: smooth_ln_fcs、smooth_ln_fcs_llama_like、smooth_lm",
  "topic": "core",
  "requires_external_verification": true,
  "excerpt_sha256": null
}
---

# mit-han-lab/smoothquant — 官方平滑实现

## 这份来源是什么

官方平滑实现，用于核对缩放的实际吸收位置与多模块成组处理方式，是 `algorithm-smoothquant` 的源码依据。

## 已读定位与阅读形式

固定 commit 下读取 GitHub 原始文件，未克隆运行。

## 关键事实

- `smooth_ln_fcs` 对共享同一输入的下游投影取联合通道范围，并把逆缩放写回归一化层参数。
- `smooth_ln_fcs_llama_like` 处理 RMSNorm 的逐通道增益；`smooth_lm` 的 Llama 路径对 Q/K/V 与 gate/up 成组处理，Mixtral 路径把 router 与专家输入投影纳入同组。

## 不能推出的结论

- 该文件只执行平滑，不含 INT8 GEMM 或 kernel 实现。
- 特定分组方式不能证明任意 MoE 路由、共享专家或残差分叉都可套用同一映射。
- 未核对融合前后的浮点等价性，也未运行量化。
