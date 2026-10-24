---
{
  "id": "repo-flatquant",
  "title": "ruikangliu/FlatQuant — 官方实现（变换参数化、块训练与重参数化）",
  "category": "repo",
  "url": "https://github.com/ruikangliu/FlatQuant/tree/9d88ffcb7d2c6bda59fb5c44dad36adc101aadb1",
  "version": "9d88ffcb7d2c6bda59fb5c44dad36adc101aadb1",
  "version_kind": "commit",
  "retrieved_at": "2026-10-23",
  "read_form": [
    "repo-tree"
  ],
  "read_locator": "flatquant/trans_utils.py: SVDSingleTransMatrix、SVDDecomposeTransMatrix；flatquant/train_utils.py: cali_flat_quant；flatquant/flat_linear.py: FlatQuantizedLinear、reparameterize；README、REALQUANT.md",
  "topic": "transforms",
  "requires_external_verification": true,
  "excerpt_sha256": null
}
---

# ruikangliu/FlatQuant — 官方实现（变换参数化、块训练与重参数化）

## 这份来源是什么

FlatQuant 官方实现，用于核对变换的参数化与逆变换、块训练的输入缓存构造以及 fake/real quant 的边界，是 `algorithm-flatquant` 的源码依据。

## 已读定位与阅读形式

固定 commit 下读取 GitHub 原始文件与仓库文档，未克隆运行。

## 关键事实

- `SVDSingleTransMatrix`/`SVDDecomposeTransMatrix` 学习正交因子与对角奇异值，逆变换使用其倒数。
- `cali_flat_quant` 用 `fp_inps` 构造教师与量化块输入并做 MSE 训练，逐层更新浮点输入缓存。
- `flat_linear.py` 的 `FlatQuantizedLinear` 内含变换、裁剪、量化与 `reparameterize`；该文件的 `F.linear` 路径属于 fake quant。
- `REALQUANT.md` 说明仓库另有真实量化权重与 kernel 路径；README 明确 vLLM 接入为 fake quant。

## 不能推出的结论

- fake-quant 路径不能证明整数 GEMM 或真实低比特加速。
- README 的能力声明不替代执行验证；源码含晚于论文 v1 的模型支持，两类证据需分开。
- 未运行训练、导出或 GPU 测试。
