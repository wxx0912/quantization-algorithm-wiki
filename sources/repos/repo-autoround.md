---
{
  "id": "repo-autoround",
  "title": "intel/auto-round — AutoRound/SignRound 官方实现与组合文档",
  "category": "repo",
  "url": "https://github.com/intel/auto-round/tree/a7656a27d020e636a961b35f84c551f453efe434",
  "version": "a7656a27d020e636a961b35f84c551f453efe434",
  "version_kind": "commit",
  "retrieved_at": "2026-10-23",
  "read_form": [
    "repo-tree"
  ],
  "read_locator": "README；auto_round/algorithms/quantization/sign_round/quantizer.py、sign_sgd.py；auto_round/data_type/int.py::quant_tensor_asym；docs/algorithm_combinations.md",
  "topic": "reconstruction",
  "requires_external_verification": true,
  "excerpt_sha256": null
}
---

# intel/auto-round — AutoRound/SignRound 官方实现与组合文档

## 这份来源是什么

AutoRound 官方实现，用于核对 SignRound 的迭代与最优状态保存、量化反量化路径以及组合能力声明，是 `algorithm-autoround` 的源码依据。

## 已读定位与阅读形式

固定 commit 下读取 GitHub 原始文件与仓库文档，未克隆运行。

## 关键事实

- `SignRoundQuantizer.quantize_block` 组织迭代流程并保存最佳 loss 状态；`sign_sgd.py` 实现符号更新。
- `quant_tensor_asym` 给出范围缩放、zero-point、`round_ste` 与 `scale*(q−zp)` 的量化反量化路径。
- `docs/algorithm_combinations.md` 把 AWQ+SignRound 列为可部署、Hadamard/SpinQuant 组合列为未就绪（含实现成熟度与主观评估）；README 另列 SignRound V1/V2、WNA16、MXFP4、NVFP4、FP8、GGUF 等能力声明。

## 不能推出的结论

- README 的能力声明是该工具版本的说法；实际启用算法、scheme、导出与目标 runtime 需分别记录。
- 未审计全部格式的 kernel，也未阅读 SignRound V2 全文，因此不展开其算法结论。
- 组合文档不是跨后端的支持承诺；导出文件成功不证明目标 runtime 使用真实低比特计算。
