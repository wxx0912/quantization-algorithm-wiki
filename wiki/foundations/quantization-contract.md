---
{
  "id": "foundation-quantization-contract",
  "title": "量化术语与最小数值契约",
  "type": "foundation",
  "tags": [
    "quantizer",
    "format",
    "deployment",
    "calibration"
  ],
  "aliases": [
    "量化基础",
    "W4A4",
    "W4A16",
    "scale",
    "zero-point",
    "静态量化",
    "动态量化"
  ],
  "related": [
    "algorithm-baselines",
    "algorithm-gptq",
    "algorithm-quarot"
  ],
  "sources": [
    "repo-gptq",
    "paper-quarot",
    "paper-smoothquant"
  ],
  "confidence": "source-reported",
  "evidence_scope": "mixed",
  "updated": "2026-10-23"
}
---

# 量化术语与最小数值契约

## 从数值到执行

把浮点权重变为低比特模型至少涉及四件不同的事：选择可表示数值与 scale，寻找近似权重或变换，存储打包后的码，运行支持该布局的算子。算法、格式、文件和 kernel 是不同对象。

| 术语 | 具体含义 | 必须补充的条件 |
| --- | --- | --- |
| W4A16 | 权重目标位宽 4、激活 16 | INT4/NF4/FP4 等具体码本、哪些算子覆盖、累加精度 |
| W4A4 | 权重与指定激活量化为 4 bit | 不是整个模型所有 tensor 都 4 bit；需列 norm、softmax、残差等例外 |
| KV4 | KV cache 的指定部分使用 4-bit 表示 | K/V 各自粒度、RoPE 前后、写入读取与反量化路径 |
| scale / zero-point | 数值与整数码之间的步长与零点 | dtype、布局、轴、是否使用第二级 scale |
| clipping | 限制可表示范围 | 截断与舍入的误差折中、范围选择目标 |
| RTN | 在给定网格上舍入到最近值 | 网格来自何处、tie-breaking、饱和行为 |
| 静态 / 动态 | 参数预先校准 / 根据当前输入求取 | 应针对具体量化器说明，不能给整个方法一刀切标签 |
| group size | 共用参数的元素数量 | 轴、tensor layout、padding、分组与排列顺序 |
| PTQ / QAT | 从已有模型进行量化优化 / 将量化纳入训练 | 还需说明训练哪些参数、数据量、损失和预算；有梯度不等于全模型训练 |

INT4、FP8、NVFP4 等描述格式体系；GPTQ、AWQ 描述优化方法；ModelOpt 是工具体系；AutoQuantize 是配置搜索接口。不能把它们作为同一列可自由替换的算法。

## 三种容易混淆的产物

1. **Fake quant**：执行量化再反量化，但 tensor/运算仍可能为浮点。用于误差模拟，不能由此声称节省存储或加速。
2. **打包低比特权重**：多个低位码装入容器整数，附带 scale/zero-point。容器 dtype 不是有效位宽，体积需计元数据和 padding。
3. **真实部署路径**：kernel 读取打包权重，可能反量化后做浮点乘法，也可能真正执行低位整数或浮点矩阵乘。两者的收益来源不同。

GPTQ 原论文 batch 1 路径主要利用压缩访存；QuaRot Linear 使用 INT4 Tensor Core，而量化 KV 在 attention 中反量化后点积。这些一手案例说明不能仅凭 checkpoint 标签推断执行精度。[GPTQ 附录 A.2.2](https://arxiv.org/html/2210.17323v2#A1.SS2.SSS2)、[QuaRot §4 Stage 2b/2c](https://arxiv.org/html/2404.00456v2#S4)

## 一个候选至少声明什么

模型与源码版本；作用模块；W/A/K/V 各自格式；分组轴与大小；scale/zero-point 规则；rounding/clipping；等价变换与顺序；校准数据和运行状态；排除模块；导出布局；后端、硬件及实际 kernel 证据。

某字段未知时给出具体待验证项。知识库不通过补默认值把未知候选变成合法候选。

## 校准与评测的分工

校准用于选择量化参数或重构；搜索集用于比较候选；独立验收数据不应反复反馈给搜索。模型局部 MSE、logits KL 和最终任务得分回答不同问题，不能相互替代。

新输入阶段会改变分布，例如 AR prefill/decode、dLLM mask ratio、DiT timestep。上述阶段轴是需要采样验证的条件，不是由静态模型名字自动确定的敏感度名单。
