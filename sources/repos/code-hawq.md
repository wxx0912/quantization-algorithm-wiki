---
{
  "id": "code-hawq",
  "title": "Zhen-Dong/HAWQ — 官方实现（QAT 训练循环，V3 为主）",
  "category": "repo",
  "url": "https://github.com/Zhen-Dong/HAWQ/tree/1616df69fdd99f100a8d8a6e78742e0a89292634",
  "version": "1616df69fdd99f100a8d8a6e78742e0a89292634",
  "version_kind": "commit",
  "retrieved_at": "2026-10-24",
  "read_form": [
    "repo-tree"
  ],
  "read_locator": "README、model_zoo.md（量化口径与复现说明）、quant_train.py（argparse 默认值、main/main_worker、train、train_kd、save_checkpoint）",
  "topic": "mixed-architecture",
  "excerpt_sha256": null
}
---

# Zhen-Dong/HAWQ — 官方实现（QAT 训练循环，V3 为主）

## 这份来源是什么

HAWQ 官方实现，用于确认仓库发布路径是量化感知训练（含反向传播与优化器更新）而非纯 PTQ，并记录其量化粒度口径与复现约束，是 `algorithm-hawq` 的源码依据。

## 已读定位与阅读形式

固定 commit 下静态读取 `README.md`、`model_zoo.md` 与 `quant_train.py` 的参数定义与训练循环。源码为本地快照（无 `.git`），commit 号取自 URL。未运行训练或推理，未读 TVM 侧实现。

## 关键事实

- **仓库定位**（README）：HAWQ 的 PyTorch 实现，做 uniform 低精度与混合精度量化，并给出经 TVM 的硬件实现；README 展示的结果表标为 HAWQ-V3（Table I/II）的 ResNet18/50。
- **训练循环**：`quant_train.py` 的 `train` 与 `train_kd` 都执行 `loss.backward()` 与 `optimizer.step()`，优化器为 `torch.optim.SGD(model.parameters(), lr, momentum, weight_decay)`；即仓库发布路径是「量化 + 微调」，不是 one-shot PTQ。
- **量化口径**（`model_zoo.md`）：权重用对称 channel-wise 线性量化，激活用静态非对称 layer-wise 线性量化；8-bit 例外，硬件只支持对称。`--fixed-point-quantization` 会跳过部分面向部署的操作以方便微调，代价是最终 TVM 结果无法与 PyTorch 完全一致。
- **关键默认值**（`quant_train.py` argparse）：`--quant-scheme` 默认 `uniform4`，`--quant-mode` 声明默认 `symmetric`（该脚本未读取 `args.quant_mode`，实际 `quant_mode` 由 `bit_config` 派生）`--bias-bit` 默认 32，`--act-percentile`/`--weight-percentile` 默认 0，`--act-range-momentum` 默认 -1；`--distill-method` 只支持 `None`/`KD_naive`（`--distill-alpha` 0.95、`--temperature` 6）。支持的骨干见 `quantize_arch_dict`：resnet18/50/50b/101、inceptionv3、mobilenetv2_w1。
- **复现约束**（`model_zoo.md`）：为与 TVM 推理精确一致，模型在单 GPU 上训练与评测；DataParallel/DistributedDataParallel 会改变各卡 BN 统计，进而影响 BN folding 与静态量化。
- **checkpoint 与体积**（README）：发布 checkpoint 以浮点精度保存，要压缩体积需另外对 `weight_integer` 张量或 `quantized_checkpoint.pth.tar` 施加 BitPack。

## 不能推出的结论

- 本仓库是量化感知训练实现：其默认训练命令不能当作 HAWQ-V2 论文实验的复现，README 的 V3 表格也不能代替 V2 结论。
- README 的模型大小、BOPS、准确率与 batch=8 延迟都绑定特定硬件与 TVM 流程；论文报告大小、checkpoint 文件大小与运行显存是三件不同的事，不能互相替代。
- 发布的浮点 checkpoint 不等于已打包的整数模型；BitPack 是外部工具，本页未运行，也未验证打包后的推理路径。
- 未运行训练或推理，也未读 TVM 侧实现与量化算子的数值行为；静态源码不证明目标后端命中。
