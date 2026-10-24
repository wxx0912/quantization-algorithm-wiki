---
{
  "id": "algorithm-awq",
  "title": "AWQ：激活感知的权重量化",
  "type": "algorithm",
  "tags": [
    "ptq",
    "scaling",
    "weight-only",
    "calibration",
    "outlier"
  ],
  "aliases": [
    "Activation-aware Weight Quantization",
    "激活感知权重量化"
  ],
  "related": [
    "algorithm-gptq",
    "algorithm-smoothquant",
    "algorithm-autoround"
  ],
  "sources": [
    "paper-awq",
    "repo-awq"
  ],
  "confidence": "source-reported",
  "evidence_scope": "source-reported",
  "updated": "2026-10-23"
}
---

# AWQ：激活感知的权重量化

AWQ 用输入激活识别对输出影响较大的权重通道，通过逐输入通道缩放改变量化误差的分配，再搜索权重 clipping。原始方法以 INT3/INT4 分组 **weight-only** 量化为主，激活保持高精度。名称中的 activation-aware 表示利用激活指导权重量化，不等于激活也被量化。

## 从输入到输出

采用列向量约定，线性层为 $Y=WX$，$W\in\mathbb R^{d_o\times d_i}$，$X\in\mathbb R^{d_i\times N}$。先用代表性文本前向，缓存待量化模块的输入；每个输入通道计算平均绝对激活 $s_{X,j}=\operatorname{mean}_n|X_{j,n}|$。

设正对角矩阵 $D=\operatorname{diag}(s)$，浮点下有

$$WX=(WD)(D^{-1}X).$$

AWQ 搜索的是下面的量化后输出误差，而不是单独的权重误差：

$$s^*=\arg\min_s\|Q(WD)D^{-1}X-WX\|_F,$$
$$s_j=s_{X,j}^{\alpha},\qquad \alpha\in[0,1].$$

这是[原论文 v5 §3.2，式 (4)–(5)](https://arxiv.org/pdf/2306.00978v5)的搜索参数化。它减少了搜索维度：同一待搜索单元只需扫描一个指数，不为每个权重学习自由参数。放大重要输入通道的权重，并在输入侧反向缩放，能使这些通道的量化误差在还原后变小；但 group 的量化步长也可能变大，因此不是所有通道都会获益。

实际流程是：

1. 缓存输入激活，确定共享输入的线性层及可吸收缩放的相邻算子。
2. 扫描 $\alpha$，对临时缩放的权重做 fake quant，比较模块输出误差，保留最佳 scale。
3. 将逆 scale 吸收到上游 normalization/linear 等支持的算子，将正 scale 乘到目标权重；不能跨任意非线性随意移动。
4. 在变换后的输入与权重上搜索 clipping 阈值，再执行最终量化和打包。

官方 `auto_scale.py` 的 `_search_module_scale` 使用 20 个候选，实际 `ratio=0,0.05,...,0.95`，并对 scale 做数值归一化。这里是实现细节；不能把论文写的区间误读为代码一定测试端点 1。`auto_clip.py::auto_clip_layer` 在每个输出通道、每个 weight group 上比较局部输出误差；`auto_clip_block` 跳过名称匹配 Q/K 的投影，原因是 QK 矩阵乘法使单独 clipping 的误差代理不够直接。

## 预算、量化契约与改动

论文 §5.1 使用 Pile 校准数据、20 点 scale 搜索，主要实验 group size 128。它没有梯度更新或逐权重重构优化，但需要缓存激活与多次模块前向；“无反向传播”不代表无校准成本。具体样本数、序列长度与 token 抽样应记录实际入口配置，不能把图中的数据量消融当作所有实验的统一默认值。

输出包括变换后的浮点参数、通道 scale、clipping 阈值及随后生成的量化权重。只保存 packed weight 而遗漏未融合的激活变换，会破坏计算等价性。AWQ 本身不规定 KV cache 量化；W4A16、group size、对称/非对称、zero-point、pack 顺序仍须从最终实现配置核对。

## 证据与适用边界

论文 §5 覆盖 OPT、LLaMA，以及指令、视觉语言、代码和数学任务。表 9 单独研究 **OPT / WikiText-2 / INT2-g64**：OPT-6.7B 的 GPTQ PPL 为 16.65，AWQ+GPTQ 为 15.71，FP16 为 10.86。这证明在该设定下已有组合证据，不证明任意模型、顺序或格式都同样有效。

论文中的 TinyChat 提速同时包含量化格式、weight packing、算子融合和硬件实现，不能将其归因于 scale 搜索本身。官方仓库的真实量化和推理入口也不意味着任意后端都接受其 packing。多模态评测结果不能自动推广至所有视觉塔、cross-attention 或 MoE 专家布局；文本 dLLM 的 mask 状态分布不属于这些 AR 结果的直接证据。

## 组合与统计失效

- **AWQ → GPTQ**：原论文表 9 已有精度证据。若先施加 $D^{-1}$，GPTQ 需要的是新输入统计；解析上 $H'=D^{-1}HD^{-1}$（$H=XX^T$）。实现未显式维护这一变换时，应重新采集，不能沿用旧 Hessian。模型已 packed 时也不能直接把 AWQ 当成后处理。
- **AWQ 与 SmoothQuant**：都可能修改相同输入通道的 scale，但优化目标不同。连续执行会改变第二次搜索的数据分布和最终 scale；需要定义合成变换并重新校准，不是独立开关。
- **AWQ → 学习舍入**：必须先固定最终浮点变换和 group，再优化舍入。后续再改变 scale/clipping 会改变网格，使原舍入结果失效。
- **旋转之后 AWQ**：通道坐标系已改变，原来的重要通道排序不可复用。是否收益、是否能融合，需要单独验证。

## 实现入口

已读版本：[llm-awq `d6e797a42b9ef7778de8ee2352116e0f48a78d61`](https://github.com/mit-han-lab/llm-awq/tree/d6e797a42b9ef7778de8ee2352116e0f48a78d61)。关键位置为 `awq/quantize/auto_scale.py` 的 `auto_scale_block`、`_search_module_scale`、`apply_scale`，以及 `awq/quantize/auto_clip.py` 的 `auto_clip_layer`、`auto_clip_block`。README 展示搜索、fake quant 与 real quant 的不同入口；这三种状态不能混称“已部署”。

## Agent 选择与消融

当 weight-only 低比特基线损失明显，且误差与输入通道激活幅度相关时，AWQ 是有依据的候选。最小比较为：固定模型、数据、位宽、group、zero-point，比较 RTN、仅 scale、仅 clip、完整 AWQ；另用独立数据观察局部输出误差与下游质量是否一致。若比较 GPTQ，应统一格式和校准预算，并单列 AWQ+GPTQ，避免把组合收益算成单算法收益。

本页为原文与源码研究，未在本项目运行量化或性能实验。
