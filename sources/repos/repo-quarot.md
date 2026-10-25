---
{
  "id": "repo-quarot",
  "title": "spcl/QuaRot — 官方参考实现（fake_quant 仿真与 4-bit kernel）",
  "category": "repo",
  "url": "https://github.com/spcl/QuaRot/tree/5008669b08c1f11f9b64d52d16fddd47ca754c5a",
  "version": "5008669b08c1f11f9b64d52d16fddd47ca754c5a",
  "version_kind": "commit",
  "retrieved_at": "2026-10-24",
  "read_form": [
    "repo-tree"
  ],
  "read_locator": "README、fake_quant/README、fake_quant/main.py、fake_quant/rotation_utils.py（fuse_layer_norms、rotate_model、get_orthogonal_matrix、QKRotationWrapper）、fake_quant/quant_utils.py（ActQuantizer、ActQuantWrapper、WeightQuantizer、pack_i4）、fake_quant/utils.py（parser_gen 默认值）",
  "topic": "core",
  "excerpt_sha256": null
}
---

# spcl/QuaRot — 官方参考实现（fake_quant 仿真与 4-bit kernel）

## 这份来源是什么

QuaRot 官方参考实现，用于核对旋转的构造与吸收位置、激活量化粒度、在线 Hadamard 的挂载点，以及仿真与真实 kernel 的分层，是 `algorithm-quarot` 的源码依据。

## 已读定位与阅读形式

固定 commit 下静态读取 `fake_quant/` 的 `main.py`、`rotation_utils.py`、`quant_utils.py`、`utils.py` 与两处 README。源码为本地快照（无 `.git`），commit 号取自 URL。未编译 `quarot/` 下的 kernel，未运行 CUDA 或模型实验。

## 关键事实

- **两层结构**：README 把“仿真结果”指向 `fake_quant/`；真实 4-bit kernel 由 `setup.py` 的 `CUDAExtension`（`quarot/kernels/bindings.cpp`、`gemm.cu`、`quant.cu`、`flashinfer.cu`）编译，`CMakeLists.txt` 用于拉取并构建第三方 cutlass 并设置 include 目录（由 `setup.py::third_party_cmake()` 调用；`add_subdirectory(quarot/kernels)` 在文件末尾已注释）。`fake_quant/README` 对外只声明“Currently, we only support **LLaMa-2** models”，但代码层 `fake_quant/utils.py` 的 `supported_models` 已列出 Llama-2 7B/13B/70B、Meta-Llama-3-8B/70B、opt-125m。
- **旋转构造**（`rotation_utils.py`）：`fuse_layer_norms` 先把 LayerNorm/RMSNorm 融进相邻线性层；其中 `bake_mean_into_linear` 只对 OPT 的 `out_proj`/`fc2` 调用（`rotation_utils.py`），embedding 的去均值 `W − mean(W, dim=−1)` 则对所有模型类型执行。`rotate_model` 再按模型类型分别旋转 embedding、attention 输入/输出、MLP 输入/输出、`down_proj` 与 head；`get_orthogonal_matrix(mode)` 的 `hadamard` 走 `random_hadamard_matrix`，`random` 走 `random_orthogonal_matrix`（对高斯矩阵做 QR，再按 `diag(r)` 符号修正）；`QKRotationWrapper` 在 RoPE 之后给 Q/K 套旋转。
- **激活只支持 per-token**：`ActQuantizer` 注释写“We only support (both sym. and asym.) per-token quantization”；`ActQuantWrapper.forward` 先（可选）执行在线 Hadamard，再 `find_params` + 量化输入，随后量化输出。
- **在线 Hadamard 的挂载**：`main.py` 对名字含 `down_proj` 的层置 `online_full_had`、含 `o_proj` 的置 `online_partial_had`（带 `had_K`/`K`/`had_dim`），`--fp32_had` 决定该 Hadamard 以 FP32 还是 FP16 执行。
- **权重量化器**：`WeightQuantizer`（注释标明“From GPTQ Repo”）默认 `perchannel=False`、`sym=True`、`mse=False`、`norm=2.4`、`grid=100`、`maxshrink=.8`；`main.py` 中 `--w_rtn` 走 RTN、否则走 `gptq_utils`；`pack_i4`/`unpack_i4` 用一个 uint8 打包两个 int4。
- **默认即未量化**：`utils.parser_gen` 默认 `rotate=False`、`rotate_mode='hadamard'`、`fp32_had=False`、`w_bits=16`、`a_bits=16`、`k_bits=16`、`v_bits=16`、`w_groupsize=-1`、`a_asym=False`、`k_pre_rope=False`、`nsamples=128`、`cal_dataset='wikitext2'`、`percdamp=0.01`、`act_order=False`；W4A4/KV4 必须显式传参。

## 不能推出的结论

- `fake_quant/` 是 torch 仿真路径，不建立整数 GEMM 或真实低比特推理；真实执行需编译 `quarot/` 下的 kernel，本页未编译、也未验证命中。
- 默认不旋转、默认 16-bit；论文的 W4A4/KV4 配置必须显式设置，不能把默认值当作论文配置。
- `--rotate_mode`、`--k_pre_rope`、`--int8_down_proj`、`fp32_had` 等开关的具体影响未做实验验证，不能凭开关名称推断论文使用的那组设置。
- README 只声明支持 LLaMa-2，代码层 `supported_models` 虽已列出 Llama-3/OPT，但本页未验证这些模型是否真正跑通，不能据此推断 Mistral、Phi-3 等架构已被覆盖。
- 本页只做静态阅读；该 commit 与论文实验版本可能有差异，仿真与 kernel 两条路径的数值一致性也未核对。
