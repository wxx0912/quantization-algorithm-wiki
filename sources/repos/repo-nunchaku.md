---
{
  "id": "repo-nunchaku",
  "title": "nunchux-ai/nunchaku — SVDQuant 配套推理引擎（低秩与低比特融合 kernel）",
  "category": "repo",
  "url": "https://github.com/mit-han-lab/nunchaku/tree/302e0e97024ebd68688fe890e5df83731edf7b54",
  "version": "302e0e97024ebd68688fe890e5df83731edf7b54",
  "version_kind": "commit",
  "retrieved_at": "2026-10-24",
  "read_form": [
    "repo-tree"
  ],
  "read_locator": "src/Linear.cpp: GEMM_W4A4::quantize；src/kernels/zgemm/gemm_w4a4.cuh: quantize_w4a4_fuse_lora_kernel（使用 EpilogueLoraDown，其定义在 src/kernels/zgemm/lora.cuh）；src/kernels/zgemm/gemm_w4a4_launch_impl.cuh: launch_lora；nunchaku/merge_safetensors.py",
  "topic": "transforms",
  "excerpt_sha256": null
}
---

# nunchux-ai/nunchaku — SVDQuant 配套推理引擎（低秩与低比特融合 kernel）

> 仓库原名 `mit-han-lab/nunchaku`：本条目钉住的 commit 的 README 中 nunchaku 仓库链接已全部指向 `github.com/nunchux-ai/nunchaku`，`url` 中的 `mit-han-lab` 为旧名。

## 这份来源是什么

Nunchaku 是 SVDQuant 的配套推理引擎，论文中两处融合（低秩 down 与激活量化、低秩 up 与 4-bit 计算）需要在这里落地；本条目用于核对这些融合是否存在于当前源码，是 `algorithm-svdquant`、`guide-deployment-evidence` 与 `case-paper-evidence` 的源码依据。

## 已读定位与阅读形式

本轮读取固定 commit 的 GitHub 源码归档（根目录 commit 与本页一致），检索调用链：`Linear.cpp` 的 `GEMM_W4A4::quantize`、`gemm_w4a4.cuh` 的 fusion kernel、`gemm_w4a4_launch_impl.cuh` 的 `launch_lora`；仅静态核验，未编译。

## 关键事实

- `GEMM_W4A4::quantize` 分配 packed activation、scales 与低秩中间结果，开启融合时调用 `quantize_w4a4_act_fuse_lora`。
- `quantize_w4a4_fuse_lora_kernel` 包含 `EpilogueLoraDown`；`launch_lora` 把低秩 up/down 与 GEMM epilogue 组合。
- 当前 `Linear.cpp` 的 scales：INT4 路径按输入维度 64 分组、scale 为模型 dtype；FP4 路径按 16 分组且用 FP8_E4M3 scale（`Linear.cpp:95-99`、`453-457`、`288-292`；导出侧同样按 `group_size = 16 if fp4 else 64`，`nunchaku/merge_safetensors.py:133`）。论文 v1 的 FP4 group 为 32（已由原始 LaTeX `sections/experiments.tex` 的 Implementation details 核实），与当前实现的 group 16 不同——这是 v1 与实现的格式差异；v3 camera-ready 已改用 NVFP4（group 16 + FP8 scales），与当前实现一致。

## 不能推出的结论

- 未编译 kernel，也未在目标 GPU 运行，不能验证数值一致性与速度。
- 不能由 DeepCompressor 的 fake-quant 评测替代真实 kernel 验证。
- 论文格式与当前导出/runtime 合约必须分开记录，不能互相替代。
