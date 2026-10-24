---
{
  "id": "repo-gptq",
  "title": "IST-DASLab/gptq — 论文官方实现（fasterquant 与 Quant3Linear）",
  "category": "repo",
  "url": "https://github.com/IST-DASLab/gptq/tree/2d65066eeb06a5c9ff5184d8cebdf33662c67faf",
  "version": "2d65066eeb06a5c9ff5184d8cebdf33662c67faf",
  "version_kind": "commit",
  "retrieved_at": "2026-10-23",
  "read_form": [
    "repo-tree"
  ],
  "read_locator": "gptq.py: GPTQ.add_batch、fasterquant；quant.py: Quantizer.find_params、quantize、Quant3Linear、Quantizer 的 mse 路径；llama.py/opt.py: 调用脚本的 perchannel/sym 默认",
  "topic": "core",
  "requires_external_verification": true,
  "excerpt_sha256": null
}
---

# IST-DASLab/gptq — 论文官方实现（fasterquant 与 Quant3Linear）

## 这份来源是什么

论文作者的官方实现快照，用于核对算法落到代码后的默认参数与数值细节（阻尼、列顺序、范围搜索的目标函数），是 `algorithm-gptq`、`algorithm-baselines` 与 `foundation-quantization-contract` 的源码依据。

## 已读定位与阅读形式

固定 commit 下逐文件读取 GitHub 原始文件，未克隆运行。

## 关键事实

- `add_batch` 累积输入统计，`fasterquant` 执行阻尼、`actorder` 重排、Cholesky 分解与误差更新，默认 `blocksize=128`、`percdamp=.01`、`groupsize=-1`。
- `quant.py` 的 `maxq=2^b−1` 由位宽无条件设定（`trits` 时置 −1），`sym=True` 才切换到对称范围规则。
- 调用脚本 `llama.py`/`opt.py` 以 `perchannel=True` 且 `--sym` 默认关闭运行，故 per-row 非对称网格是该仓库的默认实验设置。
- `mse=True` 实际累积 `|error|^norm`，默认 `norm=2.4`，另有 `grid=100`、`maxshrink=.8` 控制离散候选。
- `actorder` 按 Hessian 对角线降序重排列；README 指出配合 `--static-groups` 时推理端无需额外处理，代码在写回权重前用 `invperm` 还原列序。

## 不能推出的结论

- 静态源码的存在不证明目标运行时会命中整数 GEMM 或低比特存储。
- `mse=True` 默认不是数学意义上的平方误差（`norm=2.4`），不能当作 MSE 校准的定义。
- 所读 commit 的默认值不等于论文全部实验设置，也不能当作推荐默认值。
