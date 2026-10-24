---
{
  "id": "algorithm-quarot",
  "title": "QuaRot：旋转坐标中的低比特推理",
  "type": "algorithm",
  "tags": [
    "rotation",
    "equivalent-transformation",
    "activation",
    "kv-cache"
  ],
  "aliases": [
    "QuaRot",
    "Hadamard",
    "随机旋转"
  ],
  "related": [
    "algorithm-gptq",
    "algorithm-omniquant"
  ],
  "sources": [
    "paper-quarot",
    "repo-quarot"
  ],
  "confidence": "source-reported",
  "evidence_scope": "source-reported",
  "updated": "2026-10-23"
}
---

# QuaRot

## 机制与数学前提

QuaRot 用随机化 Hadamard 正交变换分散集中在少数坐标上的大值，并调整相邻权重保持浮点计算等价。它包含计算图变换、权重量化、在线激活量化和 KV 路径，不能简化成“对权重乘一次随机矩阵”。

设行向量激活为 `X`，正交矩阵 `R` 满足 `RRᵀ=I`，则 `XW=(XR)(RᵀW)`。若归一化已经吸收了可学习的逐通道增益，RMSNorm 保范性质允许 `RMSNorm(XR)=RMSNorm(X)R`。这不是一般非线性函数都能与旋转交换的结论。[论文 §3.4、§4](https://arxiv.org/html/2404.00456v2#S4)

## 具体流程

1. 将 RMSNorm 等可吸收的线性参数融合入相邻权重，在残差流、embedding、输入/输出投影和输出头之间一致地改变坐标。
2. FFN 的非线性之后、down projection 之前插入在线 Hadamard，逆变换吸收入 down 权重。不能直接把任意旋转穿过 SiLU/门控乘法。
3. Value 与 attention 输出投影做配对变换；跨 head 的部分使用在线结构化 Hadamard。
4. 对 RoPE 之后的 Q、K 同时做 head 内正交变换，保持 `(QR)(KR)ᵀ=QKᵀ`。一般旋转不能无条件穿过 RoPE 吸收到原始 Q/K 权重中。
5. 在上述变换后的前向上收集统计，默认用 GPTQ 量化权重。激活采用在线 RTN；KV 按缓存格式量化。

步骤 1–4 在足够浮点精度下保持输出等价；步骤 5 是有损量化。正交旋转不消除向量范数上的极端值，也不保证每次随机选择都产生相同质量。

## 配置、数据与成本

论文 v2 §5 设置：Linear 输入采用对称 per-token 量化，clipping ratio 0.9；KV 采用非对称 group size 128，clipping ratio 0.95；权重采用对称 per-channel GPTQ/RTN，并搜索权重 clipping。GPTQ 校准为 WikiText-2 训练集 128 段、每段 2048 token。

原文报告在单张 A100 上，Llama-2-70B 的模型变换约 5 分钟，后续 GPTQ 约 2 小时。这里的时间属于该模型与实现，不包括任意部署环境准备，也不是本项目实测。

## 推理路径必须分开解释

Linear 的 INT4 输入和权重通过低比特矩阵乘累加为 INT32，再反量化到 FP16。论文 Stage 2c 的 attention 则保留 Q 为 FP16，从低比特 KV cache 读取后反量化，在 FP16 中做点积和在线 softmax。因此，“W4A4+KV4”不等于所有 attention 乘法都以 INT4 执行。

在线 Hadamard 仍有计算和内存开销；部分归一化也保留高精度。论文使用 CUTLASS 与 FlashInfer 的定制路径，在 RTX 3090 上做性能评测。§5.2 明确测试单位是单个 Transformer block，因为大 batch 下整模型无法放入测试设备；不能当成整模型端到端加速。只有量化模拟器也不能证明这些收益。

## 组合与迁移边界

旋转后 GPTQ 是论文明确验证的默认组合，RTN 是原文消融。旋转改变输入基底，不能沿用原模型 GPTQ Gram 矩阵或 scale。随机种子、head 布局、KV 写入/读取顺序和在线变换位置必须随 Recipe 保存。

对非 Llama 式 norm、MoE 分支、共享权重或不同注意力结构，需要重新证明变换吸收关系。对于 dLLM，Linear 的等价变换可能仍成立，但原文没有证明其去噪轨迹上的量化质量。

## 源码入口与实验建议

固定 [fake_quant/main.py](https://github.com/spcl/QuaRot/blob/5008669b08c1f11f9b64d52d16fddd47ca754c5a/fake_quant/main.py) 先执行 `fuse_layer_norms`、`rotate_model`，再配置在线变换与权重量化；[rotation_utils.py](https://github.com/spcl/QuaRot/blob/5008669b08c1f11f9b64d52d16fddd47ca754c5a/fake_quant/rotation_utils.py) 包含配对权重修改和 Q/K 旋转 wrapper。这里已核对参考路径，未运行 kernels。

论文 §5.1/Table 1 在 WikiText-2、序列长度 2048、W4A4KV4 下报告：

| Llama-2 | FP16 PPL | QuaRot + per-channel GPTQ | QuaRot + group-128 GPTQ |
| --- | --- | --- | --- |
| 7B | 5.47 | 6.10 | 5.93 |
| 13B | 4.88 | 5.40 | 5.26 |
| 70B | 3.32 | 3.79 | 3.61 |

§5.3 比较 GPTQ 与 RTN，附录 A.7 比较在线 Hadamard 精度。选择方法时应将这些整模型质量实验与 §5.2/附录 A.10 的单 block 速度、显存实验分开，不把各自最佳条件拼成统一收益。

可证伪实验顺序：先关闭量化只验证变换等价；再固定 GPTQ 设置比较旋转开关；再分别开启激活/KV 量化；最后计入 Hadamard、打包与反量化成本测端到端性能。本项目未执行上述数值实验。
