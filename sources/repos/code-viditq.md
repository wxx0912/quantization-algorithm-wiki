---
{
  "id": "code-viditq",
  "title": "A-suozhang/ViDiT-Q — 官方实现（量化层、基础量化器与 CUDA kernel）",
  "category": "repo",
  "url": "https://github.com/A-suozhang/ViDiT-Q/tree/a8add8be17ad91069aba7a7209aa88e04b2fe41e",
  "version": "a8add8be17ad91069aba7a7209aa88e04b2fe41e",
  "version_kind": "commit",
  "retrieved_at": "2026-10-24",
  "read_form": [
    "repo-tree"
  ],
  "read_locator": "README、kernels/README.md、quant_utils/qdiff/viditq/viditq_quant_layer.py: ViDiTQuantizedLinear（get_channel_mask、get_rotation_matrix、update_quantized_weight_rotated_and_scaled、forward）、quant_utils/qdiff/base/base_quantizer.py: BaseQuantizer、StaticQuantizer、DynamicQuantizer（forward_with_quant_params）、base/mixed_precision_quantizer.py: MixedPrecisionBaseQuantizer/Static/Dynamic、base/quant_layer.py、base/quant_attn.py",
  "topic": "mixed-architecture",
  "excerpt_sha256": null
}
---

# A-suozhang/ViDiT-Q — 官方实现（量化层、基础量化器与 CUDA kernel）

## 这份来源是什么

ViDiT-Q 官方实现，用于核对量化层的静态权重/动态激活分工、通道平衡与旋转的落地顺序、混合精度的实现方式，以及仿真与 CUDA kernel 的分层，是 `guide-architecture-transfer`、`guide-deployment-evidence` 与 `case-paper-evidence` 的源码依据。

## 已读定位与阅读形式

克隆固定 commit 后静态读取 README、`kernels/README.md`、`quant_utils/qdiff/viditq/viditq_quant_layer.py` 与 `quant_utils/qdiff/base/base_quantizer.py`；未编译 kernel，未运行量化或推理。

## 关键事实

- **量化层分工**（`ViDiTQuantizedLinear` docstring）：静态权重量化 + 动态逐 token 激活量化；输入形状按 `[B, N_token, C]` 处理，激活先 reshape 成 `[B*N_token, C]` 再交给 `a_quantizer`。
- **通道平衡 + 旋转的落地顺序**：`get_channel_mask(act_mask)` 计算 `channel_mask = |W|_max(dim=0)^α / |act|_max^(1−α)`（α 取自 `quant_config.viditq.alpha`，源码注释提醒对负值取 α 次幂会出 NaN，并断言拦截）；`update_quantized_weight_rotated_and_scaled` 先把权重除以 `channel_mask` 再量化，然后右乘 `random_hadamard_matrix(in_features)` 的旋转矩阵再量化一次；`forward` 对激活「先乘 `channel_mask`、再做（双精度）旋转、再量化」——mask 相消、旋转矩阵正交，故是浮点意义下的等价变换。
- **量化器**（`base_quantizer.py`）：`BaseQuantizer`/`StaticQuantizer`/`DynamicQuantizer`；反量化统一为 `(x_int + zero_point) * delta`，量化 `round(x/delta) − zero_point`；`StaticQuantizer.init_quant_params` 由 min/max 求 `delta` 并可取非对称 `zero_point`；`DynamicQuantizer` 每次 forward 在线求 `delta`/`zero_point`（含近零 `delta` 的 NaN 保护）。
- **混合精度的实现**：仓库里有两条不同机制。① 线性层权重/激活的按分组混精由 `quant_utils/qdiff/base/mixed_precision_quantizer.py` 的 `MixedPrecisionBaseQuantizer`/`Static`/`Dynamic` 实现（`n_bits` 是位宽列表、按 `i_bitwidth` 索引），由 `quant_layer.py` 在 `n_bits` 为 `ListConfig` 时实例化；② `forward_with_quant_params`（`base_quantizer.py`）在 `mixed_precision` 分支用 `zero_bit_mask = (n_levels != 0)` 把「0 位」分组直接置零，这条路径用在 attention map 量化（`quant_attn.py`）。README 说明改 config 即可（示例 `w4a8_mixed_precision.yaml`）。
- **仿真与 kernel 分层**：README 写明设 `hardware=True` 时用 CUDA kernel 做「real」量化推理，读出 PTQ 生成的量化参数并导出缩减体积的 `int_weight.pt`；kernel 在 `kernels/`（`python setup.py develop` 安装，`bench/` 下跑 benchmark）。`kernels/README.md` 只有安装、benchmark、致谢三节。
- **模型覆盖**：README 列举 diffuser pipeline（PixArt-Σ）、OpenSora、图像 DiT，以及 Flux——Flux 处注明「not fully supported all vidtq techniques」。

## 不能推出的结论

- 量化层是 PyTorch 仿真路径；不编译 `kernels/` 就不构成真实低比特推理证据，`hardware=True` 是否命中 kernel 本页未验证。
- `kernels/README.md` 未写覆盖范围与限制，不能据此推断它支持的位宽/粒度或性能上限。
- 旋转用 `random_hadamard_matrix` 构造，本页未核对其与论文或其他实现的 Hadamard 细节是否等价；`channel_mask` 计算两侧先 `.abs()` 消除负值底数，注释所述 NaN 由断言兜底，残余风险是零/极小值产生的 inf（断言只查 NaN），不能假设任意 α 或统计都安全。
- Flux 的「not fully supported all vidtq techniques」说明能力按模型不同，不能外推到其他 DiT 或视频模型。
- 该 commit 与论文 v2 的版本对应关系未在仓库注明；本页未做任何 GPU 验证，也未核对导出权重与后端格式的一致性。
