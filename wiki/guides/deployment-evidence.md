---
{
  "id": "guide-deployment-evidence",
  "title": "从论文 W4A4 到真实部署证据",
  "type": "guide",
  "tags": [
    "deployment",
    "kernel",
    "performance",
    "validation"
  ],
  "aliases": [
    "真实加速",
    "性能",
    "fake quant",
    "W4A4 加速",
    "后端支持"
  ],
  "related": [
    "algorithm-gptq",
    "algorithm-quarot",
    "algorithm-svdquant",
    "guide-architecture-transfer"
  ],
  "sources": [
    "paper-gptq",
    "paper-quarot",
    "paper-svdquant",
    "repo-nunchaku",
    "paper-viditq",
    "code-viditq"
  ],
  "confidence": "source-reported",
  "evidence_scope": "mixed",
  "updated": "2026-10-24"
}
---

# 从论文 W4A4 到真实部署证据

论文证明某个量化模型保留质量，只能回答该配置下的模型能力问题。要判断目标后端能否加速，还需检查以下证据链。

| 层次 | 需要的证据 | 不足以证明什么 |
| --- | --- | --- |
| 数值方案 | 码本、scale、rounding、group、量化覆盖 | 不证明已打包 |
| 存储产物 | 实际位宽、元数据、padding、大小 | 不证明 GPU 以低精度计算 |
| 实现正确性 | 同输入参考量化与部署算子对照、合理误差容差 | 不证明最终任务质量 |
| kernel 路径 | 后端版本、硬件、调用链/运行证据、是否 fallback | 静态源码存在不证明运行时命中 |
| 性能 | 固定负载、计时边界、预热、并发、输出长度和显存 | 局部算子收益不等于整模收益 |
| 模型验收 | 固定协议的质量与成本目标同时满足 | 一次好结果不保证其他负载 |

## 论文与实现案例

**GPTQ**：原论文 §6/附录 A.2.2 指出其生成速度改善主要来自量化权重访存，测试主要为 batch 1。它不把算法本身等同于低比特乘法加速。

**QuaRot**：Linear 使用 INT4 Tensor Core，但 KV 读入后反量化做 FP16 attention 点积。§5.2 的速度以单 Transformer block 测量，原因是整模型在大 batch 下放不进测试设备。

**SVDQuant/Nunchaku**：低秩分支能帮助精度，但普通实现可能引入额外访存。论文用融合把低秩 down 与激活量化、低秩 up 与低比特 GEMM 衔接。当前 Nunchaku 的静态调用链提供实现证据，仍不能替代目标设备的实际运行与计时。[SVDQuant §4.3](https://arxiv.org/html/2411.05007v1#S4.SS3)

**ViDiT-Q 的版本反例**：论文 v2 明确缺少 W6A6/W4A8 的 GPU kernel（原文未加限定），延迟以 W8A8 估算；后续仓库新增 CUDA 路径不能反过来证明论文已经实测这些格式。详见 [架构与轨迹指南](architecture-transfer.md)。

## Agent 如何处理未知

若只有 fake quant 质量结果，输出“质量候选，目标部署尚未验证”。若后端加载回退浮点，记录实际路径，不能按目标低比特配置报告成功。若只测单层，在报告标题中写明单层范围。

为区分不同解释，先固定生成步数、batch、并发、上下文长度和并行配置；分别测固定工作量与真实任务。否则缩短输出或减少去噪步数可能被误记成 kernel 加速。

本 Wiki 提供检查依据，未运行模型或 GPU benchmark，也没有替任何目标格式作部署认证。
