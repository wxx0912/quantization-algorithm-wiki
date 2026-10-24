---
{
  "id": "case-paper-evidence",
  "title": "量化论文性能证据：案例导航与可比边界",
  "type": "case",
  "tags": [
    "paper-evidence",
    "benchmark",
    "latency",
    "kernel",
    "comparison"
  ],
  "aliases": [
    "论文性能案例",
    "单层 benchmark",
    "延迟估算"
  ],
  "related": [
    "algorithm-gptq",
    "algorithm-quarot",
    "algorithm-svdquant",
    "guide-deployment-evidence",
    "guide-architecture-transfer"
  ],
  "sources": [
    "paper-gptq",
    "paper-quarot",
    "paper-svdquant",
    "repo-nunchaku",
    "paper-viditq",
    "source-viditq",
    "code-viditq"
  ],
  "confidence": "source-reported",
  "evidence_scope": "mixed",
  "updated": "2026-10-23"
}
---

# 量化论文性能证据：案例导航与可比边界

本页只定位几种容易误读的性能证据，不复写算法页中的具体结果数字。共同判断框架见[从论文 W4A4 到真实部署证据](../guides/deployment-evidence.md)；下表的“直接能说明”仅限所列论文版本、负载和实现。

| 案例 | 查看页面 | 论文证据范围 | 不能据此推出 |
| --- | --- | --- | --- |
| GPTQ 的生成性能 | [GPTQ：速度与部署段](../algorithms/gptq.md) | 自回归生成、batch 1；量化权重与 FP 激活向量的矩阵向量路径。性能收益主要与少读权重、降低内存流量有关。 | 大 batch prefill、通用低比特乘法加速、任意 kernel 上 GPTQ 都更快。量化方法与论文所用执行 kernel 必须分开记录。 |
| QuaRot kernel 性能 | [QuaRot：推理路径与性能段](../algorithms/quarot.md) | RTX 3090 上 prefill/decode 测量以**单 Transformer block**为单位；大 batch 下整模型无法装入测试设备。质量评测与 kernel 性能评测范围不同。 | 把单 block 的吞吐或 speedup 写成整模型、服务端到端或其他 GPU 的实测。还需计入在线 Hadamard、KV 反量化及未量化模块。 |
| SVDQuant / Nunchaku 分支融合 | [SVDQuant：Nunchaku 与实现段](../algorithms/svdquant.md) | 论文分析独立低秩分支带来的重复访存，再将低秩 down 与激活量化、低秩 up 与低比特 GEMM 融合；当前 Nunchaku 源码有对应静态调用链。论文中特定 GPU 与 weight-only 基线的端到端结果也只适用于该配置。 | 认为任意 W4A4 实现都因低秩分支而加速；静态源码存在也不证明目标设备实际命中融合 kernel。精度收益与融合后的延迟收益是两条独立证据。 |
| ViDiT-Q 部分格式延迟 | [架构与轨迹指南：ViDiT-Q §硬件证据](../guides/architecture-transfer.md) | 论文 v2 的 W6A6/W4A8 缺少相应 kernel，相关延迟按 W8A8 kernel 估算；后续官方仓库增加的 CUDA 路径是另一个代码版本的证据。 | 把估算标成相应位宽的 kernel 实测，或用后续代码支持倒推论文当时已测该格式。 |

## 比较前对齐条件

不同论文结果只有在关键条件对齐后才可作定量比较。阅读或汇总结果时至少记录：

- **工作负载**：prefill 或 decode、batch、上下文/序列长度、输出长度、并发；扩散模型还需分辨率、帧数、scheduler、CFG 与步数。
- **模型与量化覆盖**：模型版本、哪些层/算子实际量化、W/A/KV 位宽、分组和 scale 格式；标出保留高精度的部分。
- **执行路径**：fake quant、真实打包权重、低比特 kernel 或 fallback；写出 kernel、后端、版本、GPU 与计时边界。
- **指标与基线**：延迟、吞吐、显存和生成质量分开报告；确认比较对象使用相同质量协议与相同请求负载。
- **证据状态**：分别标记论文实测、论文估算、源码静态支持和本地运行验证。缺少目标格式 kernel 时保持“估算/未知”，不能从精度结果推断速度。

若一项结果只覆盖单层、单个算子或 kernel microbenchmark，报告标题和摘要也应保留这个范围。整模型服务性能还受未量化算子、调度、数据搬运、batch 与并发影响，不能仅按量化位宽或 GEMM speedup 线性推算。

本页用于定位已有证据，不构成本项目的性能复现实验；没有新增模型或 GPU benchmark。
