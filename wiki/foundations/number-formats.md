---
{
  "id": "foundation-number-formats",
  "title": "数值格式：INT、FP8、MX 与 NVFP4",
  "type": "foundation",
  "tags": [
    "format",
    "quantizer",
    "mixed-precision",
    "deployment"
  ],
  "aliases": [
    "数值格式",
    "FP8",
    "FP4",
    "NVFP4",
    "MX",
    "量化格式",
    "MXFP4"
  ],
  "related": [
    "foundation-quantization-contract",
    "algorithm-baselines",
    "algorithm-quarot",
    "algorithm-svdquant",
    "guide-deployment-evidence"
  ],
  "sources": [
    "paper-fp8-formats",
    "paper-mx-formats",
    "paper-qad-nvfp4",
    "paper-gptq"
  ],
  "confidence": "source-reported",
  "evidence_scope": "mixed",
  "updated": "2026-10-26"
}
---

# 数值格式：INT、FP8、MX 与 NVFP4

## 一句话理解

"W4A16""W4A4"只说了目标位宽，没说是哪种码本。同为 4-bit，INT4 是均匀整数网格，MXFP4 与 NVFP4 是 4-bit 浮点，且每 32（MX）或 16（NVFP4）个元素共享一个 scale——误差行为和存储占用都不同。选格式是选"能表示什么、scale 什么粒度"，这一步先于选算法。

## 三类格式

1. **均匀整数（INT）**：固定网格，参数是 scale 与 zero-point（对称时可省 zero-point）；粒度可分 per-tensor / per-channel / per-group。工程惯例是权重用 per-channel、激活用 per-tensor（FP8 论文 Table 5 的 int8 推理即如此）。整数是均匀码本，范围校准公式（见 [baselines](../algorithms/baselines.md)）适用。
2. **浮点（FP8 等）**：位由"符号 + 指数 + 尾数"划分，指数决定动态范围、尾数决定精度。FP8 有两种编码：**E4M3**（4-bit 指数 / 3-bit 尾数）范围小精度高，**E5M2**（5-bit 指数 / 2-bit 尾数）范围大精度低。两者按 tensor 用缩放因子把最大幅值挪进可表示范围，溢出默认饱和（论文另说明可提供 non-saturating 模式）。
3. **微缩放（MX）**：一个 "MX 块"是固定数量元素**共享一个 scale**；块由"块大小 + scale 格式 + 元素格式"共同定义。论文的动机是细粒度子块缩放比 per-tensor 更贴合局部分布、在 sub-8-bit 区间更有效；整块共享 scale 也使元数据量可控。

## 格式对照

| 格式 | 元素 | scale | 说明 |
| --- | --- | --- | --- |
| INT8 / INT4 | 整数码 | per-tensor / channel / group，含 zero-point | 均匀网格；常见惯例：权重 per-channel、激活 per-tensor |
| E4M3 (FP8) | 4-bit 指数 + 3-bit 尾数 | per-tensor（论文推理侧权重用 per-channel） | 最大正规值 448；**无 Inf**，只有一种 NaN，动态范围比 IEEE 约定多一个 2 的幂（17→18 binades）。推荐用于权重与激活 |
| E5M2 (FP8) | 5-bit 指数 + 2-bit 尾数 | per-tensor | 最大正规值 57,344；遵循 IEEE 754，含 Inf/NaN，与 FP16 转换直接。推荐用于梯度 |
| MXFP8 / MXFP6 / MXFP4 / MXINT8 | FP8 / FP6 / FP4(E2M1) / INT8 | 每块共享 **E8M0**（论文只称 8-bit 指数，细节见 OCP 规范；"无尾数 → 2 的幂"按命名推得） | 论文 Table 1 的块大小为 32；共享 scale 不编码 Inf |
| **NVFP4** | 4-bit 元素（位布局待官方格式规范核对） | **两级**：每块 E4M3 scale + 每 tensor FP32 scale | 块大小 **16**；相对 FP8 约 2–3× 吞吐、约 1.8× 显存 |

## 与算法、部署的分工

| 层 | 回答什么 | 本库入口 |
| --- | --- | --- |
| 格式 | 能表示什么、scale 什么粒度、占多少元数据 | 本页 |
| 术语与产物 | W4A16 的含义、fake quant vs 打包 vs 真实 kernel | [数值契约](quantization-contract.md) |
| 算法 | 在给定格式上如何选码与 scale | [搜索空间](../guides/search-space.md) 的类别表 |
| 部署 | 目标后端是否有对应 kernel、真实吞吐 | [部署证据](../guides/deployment-evidence.md) |

## 边界与常见误解

- **W4A16 / W4A8 / W4A4 不指定位布局**：同一个 W4A16 可以是 INT4，也可以是 NVFP4；误差异质，不能只凭位宽标签比较方法。
- **谈 "FP8" 必须说清 E4M3 还是 E5M2**：两者范围差两个数量级（448 vs 57,344），必要时要按 tensor 调缩放因子。
- **E4M3 没有 Inf**：溢出是饱和到 448，不是变 Inf，这会改变误差形式。
- **NVFP4 的元素位布局未核实**：来源只确认块大小 16 与两级 scale；"元素为 E2M1"是按"NVFP4 扩展自 MXFP4"推得，须按 NVIDIA 官方格式文档核对。
- **块内共享 scale 的元数据不能忽略**：MX/NVFP4 每 16 或 32 个元素存一个 scale（NVFP4 还有一级 per-tensor FP32 scale），只按元素位宽估体积会低估。
- **FP 格式不是均匀码本**：整数范围校准（RTN/MinMax/MSE）的公式不能直接套到 FP4/FP8，见 [baselines](../algorithms/baselines.md) 的结果边界。
- 格式决定"码本与 scale 结构"，算法决定"如何选"，两者不能互相替代。

## Agent 如何使用

1. 先用 [数值契约](quantization-contract.md) 确定 W/A/KV 各自的目标位宽，再用本页把位宽落到具体格式（如 W4A16 → INT4 还是 NVFP4）。
2. 记录 scale 粒度与是否有二级 scale，这决定导出布局与体积估计。
3. 需要"怎么选码/scale"时转到算法页；需要"后端是否支持"时转到部署证据。
4. 本页只描述格式语义，未运行任何量化，也不证明某后端存在对应 kernel；表中 NVFP4 的元素位布局未经逐位核实。

## 详细来源

- [FP8 Formats for Deep Learning](../../sources/papers/paper-fp8-formats.md)
- [Microscaling Data Formats for Deep Learning](../../sources/papers/paper-mx-formats.md)
- [Quantization-Aware Distillation for NVFP4（§2.1 格式定义）](../../sources/papers/paper-qad-nvfp4.md)
- [GPTQ](../../sources/papers/paper-gptq.md)（INT4 分组与 per-channel 惯例）
