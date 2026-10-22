---
{"id":"algorithm-baselines","title":"基础基线：RTN、MinMax、AbsMax 与 MSE 范围校准","type":"algorithm","tags":["rtn","calibration","clipping","quantizer"],"aliases":["RTN","MinMax","AbsMax","MSE","最邻近舍入","范围校准"],"related":["algorithm-gptq","algorithm-smoothquant","algorithm-omniquant"],"sources":["repo-gptq","paper-gptq","paper-smoothquant","paper-omniquant"],"updated":"2026-10-23"}
---

# 基础基线：RTN 与范围校准

## 它们分别决定什么

MinMax/AbsMax 选择可表示的数值范围，MSE 校准根据误差选择范围，RTN 在给定网格上选择最近格点。它们不是互斥的完整算法：可以先用 MSE 选择 clipping 范围，再做 RTN。GPTQ 也需要量化网格，但额外改变尚未量化的权重以补偿误差。

## MinMax 与 affine 量化

对一组非恒定数值，令范围 `[l,u]` 包含零，整数码范围 `[q_min,q_max]`：

$$s=\frac{u-l}{q_{max}-q_{min}},\quad z=\operatorname{clip}\left(\operatorname{round}(q_{min}-l/s),q_{min},q_{max}\right),$$
$$q=\operatorname{clip}(\operatorname{round}(x/s)+z,q_{min},q_{max}),\qquad \widehat x=s(q-z).$$

RTN 只做取最近整数和截断。`z` 是整数零点，不能在反量化时漏掉。零范围/恒定组需要实现明确处理，不能直接除以零；四舍五入的 tie-breaking、scale dtype 和零点存储方式也必须进入部署契约。

MinMax 使用样本极值；weight-only 可直接读取权重而无需输入数据，但激活的静态 MinMax 需要校准数据。动态量化则对当前 token、tensor 或 block 求范围，增加在线统计开销。

## AbsMax 与对称网格

令 `a=max|x|`。一种常见 signed 对称约定使用 `[-Q,Q]`，`s=a/Q`、`z=0`；INT4 常取 `Q=7`，其舍入与截断不一定使用全部 16 个码。另一实现可能使用 unsigned 偏移码和非完全对称端点，不能仅凭“symmetric”字符串判断二者数值一致。

具体例子：GPTQ 固定源码 `quant.py` 的 `sym=True` 采用 `maxq=2^b-1`、中点零点与自己的范围规则；QuaRot 论文描述的 INT4 激活按最大值除以 7。共享 bit 数不意味着码本、rounding 和边界完全相同。

## MSE 范围搜索

枚举候选 clipping 范围 `[l_c,u_c]`，构造每个候选网格并反量化：

$$c^*=\arg\min_c\frac1N\sum_i(x_i-\widehat x_i(c))^2.$$

缩小范围能降低步长和主体舍入误差，却增加尾部截断误差。候选可由极值的收缩比例或其他明确规则产生；离散网格搜索只在指定候选集合中选择，不保证全局最优。

输入为权重或激活样本，输出为每组 scale/zero-point/clipping。成本约为候选数乘样本元素数；不需要梯度。若优化的是 block 输出 MSE，必须明确改变了目标和评估成本，不能继续称为单纯 tensor 范围 MSE。

**源码名称陷阱：** [GPTQ quant.py，2d65066](https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/quant.py) 的 `mse=True` 实际累积 `|error|^norm`，默认 `norm=2.4`。只有显式取 2 才是严格平方误差；默认并非数学意义上的 MSE。`grid=100`、`maxshrink=.8` 控制离散候选，不能泛化成所有 MSE 校准的定义。

## 粒度与适用性

per-tensor 一个 scale，per-channel 每个指定通道一组，group-wise 在约定轴上分组。更细粒度通常降低量化误差，但增加元数据和 kernel 约束。粒度必须写明轴与布局，如 Linear `[out,in]` 的“每个输出行，沿输入轴 group 128”。仅写 group size 不够。

本文公式对应均匀整数网格。FP8、FP4 的可表示集合并非均匀整数网格，不能机械套用 `z` 和 INT4 边界；应单独记录具体浮点格式、缩放层级和 kernel。

## 基线实验与局限

GPTQ v2 Table 11 的 OPT-175B/C4，4-bit 相同 per-row 网格下 RTN perplexity 11.61、GPTQ 10.28、FP16 10.13。这是 RTN 在该任务中的比较，不是普遍失败证明。SmoothQuant v7 Table 2–4 展示激活粒度/静态性对 W8A8 结果的影响；OmniQuant v3 Eq.2 展示学习 clipping 如何从 MinMax 出发。

Agent 应先固定格式、粒度、排除模块和评测协议，比较 MinMax+RTN 与范围搜索+RTN，再加入更复杂方法。只看权重 MSE 不能判定最终生成质量；输入分布变化、旋转或缩放后应重新估计受影响范围。本项目只核对公式和源码，不提供真实模型量化结果。
