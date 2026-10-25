---
{
  "id": "repo-autoround",
  "title": "intel/auto-round — AutoRound/SignRound 官方实现与组合文档",
  "category": "repo",
  "url": "https://github.com/intel/auto-round/tree/a7656a27d020e636a961b35f84c551f453efe434",
  "version": "a7656a27d020e636a961b35f84c551f453efe434",
  "version_kind": "commit",
  "retrieved_at": "2026-10-24",
  "read_form": [
    "repo-tree"
  ],
  "read_locator": "README；auto_round/algorithms/quantization/sign_round/quantizer.py、sign_sgd.py；auto_round/data_type/int.py::quant_tensor_asym、auto_round/data_type/utils.py::reshape_pad_tensor_by_group_size；docs/algorithm_combinations.md",
  "topic": "reconstruction",
  "excerpt_sha256": null
}
---

# intel/auto-round — AutoRound/SignRound 官方实现与组合文档

## 这份来源是什么

AutoRound 官方实现，用于核对 SignRound 的迭代与最优状态保存、量化反量化路径以及组合能力声明，是 `algorithm-autoround` 的源码依据。

## 已读定位与阅读形式

读取固定 commit 的源码归档与仓库文档；本轮复查本地归档 auto-round-a7656a27d020e636a961b35f84c551f453efe434 中的相应文件，未运行量化或推理。

## 关键事实

- `SignRoundQuantizer.quantize_block` 组织迭代流程；未启用 `not_use_best_mse` 时保存最低 loss 状态，启用时保存最后迭代状态。仅在 `q_inputs` 存在且 `enable_quanted_input` 启用时使用量化输入，否则使用浮点输入。`sign_sgd.py::_single_tensor_sgd` 实现按梯度符号更新。
- `quant_tensor_asym` 给出范围缩放、zero-point、`round_ste` 与 `scale*(q−zp)` 的量化反量化路径。
- `auto_round/data_type/utils.py::reshape_pad_tensor_by_group_size` 在 `group_size=-1` 时直接返回原张量、不做分组；`quant_tensor_asym` 沿最后一维求范围，因此这里是不再细分组的 per-row 量化，不是整张量共享一个 scale。这是当前工具约定，不能反写为论文对 G-1 的显式定义。
- `docs/algorithm_combinations.md` 把 AWQ+SignRound 列为可部署、Hadamard/SpinQuant 组合列为未就绪（含实现成熟度与主观评估）；README 另列 SignRound V1/V2、WNA16、MXFP4、NVFP4、FP8、GGUF 等能力声明。

## 不能推出的结论

- README 的能力声明是该工具版本的说法；实际启用算法、scheme、导出与目标 runtime 需分别记录。
- 未审计全部格式的 kernel，也未阅读 SignRound V2 全文，因此不展开其算法结论。
- 组合文档不是跨后端的支持承诺；导出文件成功不证明目标 runtime 使用真实低比特计算。
