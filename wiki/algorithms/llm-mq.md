---
{
  "id": "algorithm-llm-mq",
  "title": "LLM-MQ：预算约束下的权重混合精度量化",
  "type": "algorithm",
  "tags": [
    "mixed-precision",
    "weight-only",
    "outlier",
    "sensitivity",
    "cost-constraint",
    "kernel",
    "deployment",
    "ptq"
  ],
  "aliases": [
    "LLM-MQ",
    "稀疏离群值保护",
    "敏感度位宽分配",
    "权重显存预算下分配位宽"
  ],
  "related": [
    "algorithm-hawq",
    "algorithm-mcmoe",
    "algorithm-baselines",
    "foundation-quantization-contract",
    "guide-search-space"
  ],
  "sources": [
    "paper-llm-mq"
  ],
  "confidence": "source-reported",
  "evidence_scope": "source-reported",
  "updated": "2026-10-28"
}
---

# LLM-MQ：预算约束下的权重混合精度量化

## 一句话理解

LLM-MQ 是 **weight-only** 的混合精度方法，回答的是"给定权重显存预算，每层该给几位"。它用两件事压住极低比特的掉点：**稀疏离群值保护**（每层把绝对值最大的 0.5% 权重留成 FP16，其余才量化）与**一阶敏感度位宽分配**（把每层位宽 ∈{2,3,4} 的选择写成带显存预算约束的整数规划），最后用自研 CUDA kernel 融合反量化与 GEMV 兑现速度。LLaMA2-13B 上 2.8 bit 平均准确率 64.03，4.0 bit 为 64.96（FP16 为 65.08）。

## 核心机制

1. **稀疏离群值保护**：每层只保留绝对值最大的 0.5% 权重为 FP16（CSR 稀疏存储，论文自述沿用 SqueezeLLM 思路），其余权重按**分配到的位宽**量化；论文用 2-bit 展示该保护的作用——2-bit 均匀量化误差 0.0075 对 3-bit 0.0032（约 2.3 倍），保护后 2-bit 降到 0.0048（Figure 2），即降到约 1/1.56，接近 3-bit 水平。
2. **一阶敏感度**：用一阶 Taylor 近似估计权重量化对损失的影响，写作 `ΔL ≈ g_iᵀ(W_i − Q_b(W_i))`（逐字沿用论文 Eq.1；若按 `ΔL = L(Q(W)) − L(W)` 定义，则应为 `ΔL ≈ g_iᵀ(Q(W) − W)`，两者符号相反——敏感度取绝对值后，位宽分配目标不受该符号约定影响），其中 `g_i` 是损失对第 `i` 层权重的梯度；论文指出 LLM 中约 15% 层的 Hessian 非半正定，二阶信息不可靠，因此只取一阶项 `s_{i,b} = |g_iᵀ(W_i − Q_b(W_i))|`。
3. **整数规划位宽分配**：在权重显存预算 `B` 下最小化 `Σ_i Σ_b c_{i,b}·s_{i,b}`，约束为每层恰好一个位宽、位宽取 {2,3,4}，`c` 为 one-hot 指示变量。用求解器在数秒内得到每层位宽——这是"预算约束"而非"固定平均位宽"。
4. **融合 kernel 与位打包**：4-bit 两两一字节、3-bit 八个三字节、2-bit 四个一字节；把反量化融进 GEMV；选 CUDA Core 而非 Tensor Core（Tensor Core 要求每维至少 8，而 GEMV 最小维度为 1，补零低效；且 GEMV 延迟由权重访存主导）；另为奇数位宽设计访存排布。

## 基本流程

1. 用校准数据计算损失并求取逐层权重梯度 `g_i`（需要反向传播；论文未给出校准数据规模与求梯度的实现细节），再对候选项位宽 `b∈{2,3,4}` 计算量化误差 `W_i − Q_b(W_i)`，得到 `s_{i,b}`。
2. 解整数规划，在目标设备的权重显存预算下得到每层位宽。
3. 按分配结果量化：每层保留 top 0.5%（绝对值最大）权重为 FP16（CSR），其余按 group size 128 分组量化并沿用 AWQ 的重参数化。
4. 推理时用融合反量化与 GEMV 的 CUDA kernel 执行。

## 适用条件

- 目标明确为**权重显存预算**（edge/显存受限设备），需要 2–3 bit 量级的平均位宽；只压权重，不含激活与 KV cache。
- 需要校准数据，并通过反向传播计算逐层权重梯度；论文未披露具体实现细节。除梯度外还需有能力估计每层量化误差；论文用 group size 128、AWQ 重参数化，评测用 lm-evaluation-harness。
- 平均位宽口径不含 FP16 稀疏权重（论文称最多使其增加 0.08 bit），对外报位宽时要说明。
- 实现侧需要支持 2/3/4-bit 权重 + FP16 稀疏分量的 GEMV；论文的加速只在自研 kernel 上给出，通用后端支持情况未报告。

## 结果边界

- Table 1（LLaMA2-13B，group 128，平均精度 / Wiki PPL）：FP16 65.08 / 7.89；LLM-MQ 4.0 bit 64.96 / 8.03、2.8 bit 64.03 / 8.83、2.0 bit 60.23 / 12.17（3.0–3.8 bit 各档完整数值见来源页）；对照 RTN 2.0 bit 41.74 / 1056.33、AWQ 2.0 bit 37.06 / 5e6。
- 论文正文称 2-bit 时零样本平均比 AWQ、RTN 高 23.17 与 19.16 个点；23.17 与 Table 1 的 60.23 − 37.06 相符，19.16 与 60.23 − 41.74 = 18.49 不符（按表格计算）。
- 附录 2.0 bit（平均精度 / Wiki PPL）：LLaMA2-7B 55.61 / 15.93，OPT-6.7B 57.63 / 17.09，OPT-13B 57.36 / 17.24。
- 效率：INT4 LLaMA2-7B 在 NVIDIA T4（16 GB）上、输出 32/128/512/1024/2048 token 相对 PyTorch FP16 端到端加速 1.27/1.42/1.65/1.64/1.45 倍（Abstract 写最高 1.6×）；4096×4096 与 4096×11008 GEMV kernel 相对 cuBLAS FP16 为 2.33×/2.48×。论文明确 2-bit、3-bit kernel 相对 4-bit 无显著加速。
- 未复现、未在目标后端核验；对照只含 RTN/AWQ/SqueezeLLM，没有 GPTQ、QuIP# 等其他 2-bit 方案；结论不能外推到激活、KV cache、MoE、DiT/dLLM。

## Agent 如何使用

处理"权重只给一定显存预算、要在 2–3 bit 之间分配位宽"或"2-bit 权重掉点严重、想保住离群值"的问题时先读本页，再按需展开：

| 需要确认的内容 | 继续读取 |
| --- | --- |
| 误差数字、Table 1/2/3/4 全表、kernel 与效率细节 | [`paper-llm-mq`](../../sources/papers/paper-llm-mq.md) |
| 另一种敏感度位宽分配（Hessian trace，含 QAT） | [HAWQ](hawq.md) |
| 专家级混合精度 + 整数规划 + 剪枝 | [MC-MoE](mcmoe.md) |
| RTN 与分组量化的数值口径 | [量化契约](../foundations/quantization-contract.md)、[基线](baselines.md) |
| 位宽/格式在搜索空间中的位置 | [量化搜索空间](../guides/search-space.md) |

回答时要区分三类贡献（离群值保护 / 位宽分配 / kernel），并把"论文报告值"与"目标硬件实测"分开；不要把 4.0 bit 的 1.6× 加速说成 2-bit 的收益。

## 详细来源

- [LLM-MQ 论文来源页](../../sources/papers/paper-llm-mq.md)
