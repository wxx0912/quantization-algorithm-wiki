---
{
  "id": "repo-flatquant",
  "title": "ruikangliu/FlatQuant — 官方实现（变换参数化、块训练与重参数化）",
  "category": "repo",
  "url": "https://github.com/ruikangliu/FlatQuant/tree/9d88ffcb7d2c6bda59fb5c44dad36adc101aadb1",
  "version": "9d88ffcb7d2c6bda59fb5c44dad36adc101aadb1",
  "version_kind": "commit",
  "retrieved_at": "2026-10-24",
  "read_form": [
    "repo-tree"
  ],
  "read_locator": "flatquant/trans_utils.py: SVDSingleTransMatrix、SVDDecomposeTransMatrix；flatquant/train_utils.py: cali_flat_quant；flatquant/flat_linear.py: FlatQuantizedLinear、reparameterize；README、REALQUANT.md",
  "topic": "transforms",
  "excerpt_sha256": null
}
---

# ruikangliu/FlatQuant — 官方实现（变换参数化、块训练与重参数化）

## 这份来源是什么

FlatQuant 官方实现，用于核对变换的参数化与逆变换、块训练的输入缓存构造以及 fake/real quant 的边界，是 `algorithm-flatquant` 的源码依据。

## 已读定位与阅读形式

本轮读取固定 commit 的源码归档及仓库文档，归档顶层目录确认版本；静态阅读，未执行训练。

## 关键事实

- `SVDSingleTransMatrix`/`SVDDecomposeTransMatrix` 学习正交因子与对角奇异值，逆变换使用其倒数。
- `cali_flat_quant` 用 `fp_inps` 构造教师与量化块输入并做 MSE 训练，逐层更新浮点输入缓存。
- `flat_linear.py` 的 `FlatQuantizedLinear` 内含变换、裁剪、量化与 `reparameterize`；该文件的 `F.linear` 路径属于 fake quant。
- `REALQUANT.md` 说明仓库另有真实量化权重与 kernel 路径；README 明确 vLLM 接入为 fake quant。

## 不能推出的结论

- fake-quant 路径不能证明整数 GEMM 或真实低比特加速。
- README 的能力声明不替代执行验证；源码含晚于论文 v1 的模型支持，两类证据需分开。
- 未运行训练、导出或 GPU 测试。

## 实现补充

`cali_flat_quant` 的量化块输入来自 `fp_inps`，每块完成后交换 `fp_inps, fp_outs`，不能写成始终使用前块量化输出。训练先算 MSE，再执行 `loss / loss.clone().detach()`，因此实际梯度带有按当前损失归一化的因子。`SVDSingleTransMatrix`/`SVDDecomposeTransMatrix` 用 Cayley 正交参数化和可学习对角参数；`inv_t` 路径构造逆转置，不能将其名称省略成无转置的逆矩阵。`FlatQuantizedLinear.reparameterize` 固化变换后权重，真实低位执行另见 `REALQUANT.md`。工程上仍需检查近零对角参数的数值稳定性。
