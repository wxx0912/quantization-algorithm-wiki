---
{
  "id": "algorithm-duquant",
  "title": "DuQuant：旋转 + 置换双重变换处理正常与巨量离群",
  "type": "algorithm",
  "tags": [
    "activation",
    "rotation",
    "outlier",
    "quantizer",
    "equivalent-transformation"
  ],
  "aliases": [
    "DuQuant",
    "双重变换",
    "旋转置换量化",
    "zigzag 置换",
    "之字形置换",
    "massive outlier 量化"
  ],
  "related": [
    "algorithm-quarot",
    "algorithm-smoothquant",
    "algorithm-omniquant",
    "foundation-quantization-contract",
    "guide-search-space",
    "guide-architecture-transfer"
  ],
  "sources": [
    "paper-duquant"
  ],
  "confidence": "source-reported",
  "evidence_scope": "source-reported",
  "updated": "2026-10-28"
}
---

# DuQuant：旋转 + 置换双重变换处理正常与巨量离群

## 一句话理解

DuQuant 针对 LLM 激活的两类离群——**Normal Outliers**（所有序列都出现的固定特征维）与 **Massive Outliers**（attention sink，只在少数 token 出现、量级常 >100）——用一套"平滑 + 块对角旋转 + 之字形置换 + 再旋转"的等价变换（RAP）把离群在激活空间里摊平，使 W4A4 的整数均匀量化更稳。平滑矩阵与 LWC 因子可折入量化权重；旋转/置换在激活侧仍有在线开销（论文报告仅旋转 +1.5%/+1.8%、一个置换+两次旋转 +8.9%/+9.3%，7B/13B，相对 W4A4）。

## 核心机制

1. **两类离群**：normal 持续存在；massive 集中在少数 token 且量级比其它激活大三个数量级。论文称其首次发现 massive outliers 出现在 **FFN 下投影层的输入**（如 LLaMA2-7B 第 1 层有约 1200 的单点激活）。
2. **平滑 Λ**：先把量化难度从激活迁到权重（`Y=(XΛ)(Λ⁻¹W)`）；但 massive outlier 会让平滑因子过大、给权重引入新离群，因此单靠平滑不够。
3. **块对角旋转 `R̂`**：带先验的贪心搜索定位离群主维、构造正交旋转，再做块对角近似（块大小 `2^n`，`K=C_in/2^n` 块）；为省算力可让所有块复用同一个块旋转矩阵。
4. **之字形置换 `P`**：块内旋转无法跨块平衡，块间离群幅度方差仍大；之字形置换把最大激活通道"大到小、再小到大"往返分到各块，再叠加第二次旋转进一步平滑。
5. **整体变换（Eq.5）**：`Y=[(XΛ)R̂_(1)·P·R̂_(2)]·[R̂_(2)ᵀ·Pᵀ·R̂_(1)ᵀΛ⁻¹]W`；记激活侧变换为 `G`、权重侧逆变换为 `G⁻¹`——`G⁻¹` 可离线折入量化权重（`W'=G⁻¹W`，不增加推理开销），`G` 仍须在激活侧在线施加。

## 基本流程

1. 对目标线性层先做 smooth（通道平滑因子 `Λ`）。
2. 由激活统计贪心搜索并构造块对角旋转 `R̂`，对激活做第一次旋转。
3. 用之字形置换 `P` 平衡各块的离群幅度。
4. 再做一次块对角旋转 `R̂`；权重侧用 `G⁻¹` 离线折入量化权重，激活侧按 `G` 在线变换。
5. 用整数均匀量化（激活 per-token、权重 per-channel）量化；可选叠加 `+LWC`（可学习权重裁剪）。

## 适用条件

- 面向 LLM 的**权重+激活**低比特量化（论文主打 W4A4，附录有 W6A6），尤其是激活中存在 massive outlier / attention sink 的模型。
- 需要一小段校准数据估计平滑与旋转；变换可离线折入权重，推理时主要开销是激活侧的旋转与置换。
- 只处理激活/权重的分布整形，不改变位宽语义；不能替代后端低比特 kernel。

## 结果边界

- 论文报告（Table 1，W4A4 WikiText2 PPL，越低越好；小模型评测用 RTX 3090、大模型用 A100）：LLaMA2-7B FP16 5.47 → DuQuant **6.28** / +LWC **6.08**（Atom 8.40、OmniQuant 14.26、SmoothQuant 83.12）；LLaMA2-70B FP16 3.31 → **3.79** / **3.76**（该设置下 OmniQuant 与 Atom 为 NaN）；LLaMA1-7B 5.68 → 6.40 / 6.18。
- zero-shot（Table 2，W4A4，LLaMA1-7B 六任务平均）：FP16 64.09、QLLM 51.84、Atom 56.88、DuQuant **61.76**；论文称平均比 QLLM 高约 +9%、比 Atom 高约 +5%。
- 模块消融（Table 5，WikiText2/C4）：7B 仅 smooth NaN/1379.46、仅 rotation 8.48/10.63、smooth+rotation 7.92/10.64、rotation+permutation+rotation2 6.79/8.51、四项全开 6.28/7.90；13B 依次 160.30/203.87、14.32/21.73、5.96/7.94、6.06/8.03、5.42/7.05。与 QuaRot 对比（Table 7）在多数模型上略优或持平。
- 主结果限 W4A4，模型为 LLaMA 1/2/3 与 Vicuna-v1.5；块大小、贪心步数、裁剪比等为论文设定。未复现、未在目标后端核验。

## Agent 如何使用

处理"激活离群（尤其 attention sink / massive outlier）怎么压、旋转不够怎么办、W4A4 上不去"时先读本页，再按需展开：

| 需要确认的内容 | 继续读取 |
| --- | --- |
| Eq.1–5、双变换构造、Table 1–7 与设置 | [`paper-duquant`](../../sources/papers/paper-duquant.md) |
| 纯 Hadamard 旋转路线与对比 | [QuaRot](quarot.md) |
| 通道平滑与可学习等价变换 | [SmoothQuant](smoothquant.md)、[OmniQuant](omniquant.md) |
| 量化契约与搜索空间 | [量化契约](../foundations/quantization-contract.md)、[量化搜索空间](../guides/search-space.md) |

回答时要区分"平滑（Λ）""块对角旋转""之字形置换"三类机制，并说明 `DuQuant` 与 `DuQuant+LWC` 是两个配置；map 到目标模型时需重新确认块大小与裁剪比。

## 详细来源

- [DuQuant 原论文来源页](../../sources/papers/paper-duquant.md)
