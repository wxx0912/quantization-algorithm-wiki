---
{
  "id": "repo-omniquant",
  "title": "OpenGVLab/OmniQuant — 官方实现（LWC/LET 与量化器）",
  "category": "repo",
  "url": "https://github.com/OpenGVLab/OmniQuant/tree/feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14",
  "version": "feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14",
  "version_kind": "commit",
  "retrieved_at": "2026-10-23",
  "read_form": [
    "repo-tree"
  ],
  "read_locator": "README、quantize/omniquant.py、quantize/quantizer.py",
  "topic": "core",
  "excerpt_sha256": null
}
---
## 这份来源是什么

OmniQuant 官方实现，用于核对 LWC/LET 在代码中的参数化与融合方式，是 `algorithm-omniquant` 的源码依据。

## 已读定位与阅读形式

固定 commit 下读取 GitHub 原始文件，未克隆运行。

## 关键事实

- 裁剪（LWC）参数以可学习张量实现在 `quantizer.py`（可学习上下界因子）；LET 变换参数（QKV/out/fc1 的 smooth shift/scale）在 `omniquant.py` 中注册为可学习张量，训练目标为 block 输出重构（MSE 口径）。
- `omniquant.py` 组织逐 block 的校准与优化流程，并提供把变换融合回权重的路径。
- README 给出 LLaMA-7B 的运行入口与位宽配置示例；OPT 的运行脚本位于 `scripts/opt/`。

- **论文与 README 学习率不同**：论文 v3 §4.1 Settings 的 Training 段写 LWC/LET 为 `5e-3/1e-2`，固定源码 README 写 `--lwc_lr/--let_lr` 默认 `1e-2/5e-3`。本轮对照 LaTeX 与 README 确认，复现时以显式命令和解析参数为准。

## 不能推出的结论

- 示例配置不代表论文全部实验设置，也不是推荐默认值。
- 融合后的数值等价性与 kernel 命中未在本仓库验证，未运行优化或推理。
