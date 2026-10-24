---
{
  "id": "algorithm-omniquant",
  "title": "OmniQuant：联合学习 clipping 与等价变换",
  "type": "algorithm",
  "tags": [
    "joint-optimization",
    "clipping",
    "scaling",
    "block-reconstruction"
  ],
  "aliases": [
    "OmniQuant",
    "LET",
    "LWC",
    "可学习等价变换"
  ],
  "related": [
    "algorithm-smoothquant",
    "algorithm-gptq"
  ],
  "sources": [
    "paper-omniquant",
    "repo-omniquant"
  ],
  "confidence": "source-reported",
  "evidence_scope": "source-reported",
  "updated": "2026-10-23"
}
---

# OmniQuant

## 问题与优化变量

固定范围和固定平滑规则在极低位宽下可能无法同时控制 W/A 误差。OmniQuant 冻结原始模型权重，逐 Transformer block 学习少量 clipping 和等价变换参数，最小化浮点与量化 block 的输出差异。使用梯度不等于全模型 QAT；这里学习的是受限的量化相关变量。

论文 Eq.1 的 block 重构目标，可写为

$$\min_{\Theta_{LWC},\Theta_{LET}}\|F(W,X)-F_q(W,X;\Theta_{LWC},\Theta_{LET})\|_F^2.$$

源码以 MSELoss 实现平方误差。式子概括局部目标；真实实现区分浮点参考输入与经过前序量化 block 的输入，不能假定整个网络始终在原始浮点激活上独立优化。

## LWC：学习范围缩减比例

对一个量化组，令 `l=β min(W)`、`u=γ max(W)`，`β,γ` 由 sigmoid 参数化约束到 `(0,1)`，则

$$h=\frac{u-l}{2^b-1},\quad z=-\operatorname{round}(l/h),\quad q=\operatorname{clip}(\operatorname{round}(W/h)+z,0,2^b-1).$$

反量化为 `h(q-z)`。学习的是上下界相对于当前权重极值的缩减强度，而不是自由训练全部权重。`β=γ=1` 退化为对应 MinMax 范围。[论文 §3.2 Eq.2](https://arxiv.org/html/2308.13137v3#S3.SS2)

## LET：学习可吸收的缩放与平移

行样本布局 `Y=XW+B` 下，`D=diag(s)`：

$$XW+B=((X-\delta)D^{-1})(DW)+(B+\delta W).$$

先对变换后的激活和权重量化，平移补偿体现在偏置中。attention 中还可对 Q/K 做成对逆缩放保持点积不变；具体吸收位置必须符合位置编码与实际前向顺序。[论文 §3.3 Eq.3–5](https://arxiv.org/html/2308.13137v3#S3.SS3)

论文没有在所有线性层上无条件应用 LET：FFN 第二个线性层被排除，作者指出非线性后稀疏特征可能造成不稳定梯度。实际实现还因模型是否支持 shift/bias 而不同。

## 完整流程与预算

采集激活 scale/shift 初始化信息 → 缓存 block 浮点参考 → 插入 LWC/LET → 在校准数据上联合优化 → 融合变换并量化 → 将结果传给下一 block。权重-only 与 W/A 配置不是同一组开关：论文 Llama 权重-only 主要使用 LWC，OPT 可同时使用 LET；W/A 使用两者。

论文 §4.1：WikiText-2 随机 128 段、长度 2048、batch size 1、通常 20 epochs，W2A16 使用 40 epochs；AdamW 零 weight decay。W-only 研究 INT2/3/4 与 per-channel/group-wise；W/A 研究 W4A4/W6A6，默认权重 per-channel、激活 per-token，Softmax 输出保留浮点。

这需要 block 前反向、参考激活缓存和优化器状态，不能按只做一次前向的校准估算预算。附录某些运行时间表使用 A100-80G，摘要提到 A100-40G 可处理范围；引用时间时必须绑定具体表，而不是统一成一个硬件数字。

## 实现与版本差异

[quantize/omniquant.py，feffe8e](https://github.com/OpenGVLab/OmniQuant/blob/feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14/quantize/omniquant.py) 中 `omniquant` 缓存输入、构造 MSE、对 LET/LWC 参数分组优化，再调用 `smooth_and_quant_inplace`。`aug_loss` 另加参考项，不能与论文简式不加区分。

[quantize/quantizer.py](https://github.com/OpenGVLab/OmniQuant/blob/feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14/quantize/quantizer.py) 的 `UniformAffineQuantizer` 使用 sigmoid bound factors 和 `round_ste`；源码名字 `per_token_dynamic_calibration` 不意味着所有调用都只处理激活，应结合输入 reshape 和 group size 判断。

**已发现配置差异：** 论文 §4.1 将 LWC/LET 学习率分别写成 `5e-3/1e-2`，当前 README 则将 `--lwc_lr/--let_lr` 默认值写成 `1e-2/5e-3`。不能将二者混合声称是精确复现参数；运行时应显式记录实际命令与解析值。

README 的 MLC-LLM 示例属于特定 weight-only 路径；有 W4A4 模拟支持不代表目标后端存在 W4A4 高效 kernel。

## 实验、组合与边界

论文 Table 1：Llama-2-7B / WikiText-2 / W4A16 group 128，RTN 5.72、GPTQ 5.61、AWQ 5.62、OmniQuant 5.58 perplexity。这是表内给定配置的比较，不支持普遍的算法排名，也不是本项目实测。

LET 与 LWC 是有联合目标和消融证据的组合，SmoothQuant 式初始化只是起点。改变 LET 后权重分布和输入统计改变，旧 GPTQ Hessian、旧 clipping 最优值以及敏感度结果不可直接复用。把 OmniQuant 后再串联 GPTQ 应另作研究候选，不是默认受支持的组合。

建议固定数据和预算，对比初始化-only、LWC-only、LET-only、联合优化，并同时看搜索集与未参与优化的数据。若层 MSE 改善但生成退化，应检查前序量化输入与完整生成分布，而非仅增加 epochs。本项目未做 GPU 量化或后端验证。
