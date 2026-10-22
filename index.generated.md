# 页面索引

> 由 `python3 scripts/build_index.py` 生成，请勿手改。

## 基础

- `foundation-quantization-contract` — [量化术语与最小数值契约](wiki/foundations/quantization-contract.md)

## 算法

- `algorithm-adaround` — [AdaRound：固定量化网格上的自适应舍入](wiki/algorithms/adaround.md)
- `algorithm-autoround` — [AutoRound / SignRound：用符号梯度联合优化舍入与裁剪](wiki/algorithms/autoround.md)
- `algorithm-awq` — [AWQ：激活感知的权重量化](wiki/algorithms/awq.md)
- `algorithm-baselines` — [基础基线：RTN、MinMax、AbsMax 与 MSE 范围校准](wiki/algorithms/baselines.md)
- `algorithm-brecq` — [BRECQ：块级重构与输出敏感度](wiki/algorithms/brecq.md)
- `algorithm-flatquant` — [FlatQuant：可学习可逆变换与块重构](wiki/algorithms/flatquant.md)
- `algorithm-gptq` — [GPTQ：基于输入二阶统计的权重重构](wiki/algorithms/gptq.md)
- `algorithm-hawq` — [HAWQ：用 Hessian 敏感度选择混合精度](wiki/algorithms/hawq.md)
- `algorithm-modelopt-autoquantize` — [ModelOpt AutoQuantize：受成本约束的格式搜索](wiki/algorithms/modelopt-autoquantize.md)
- `algorithm-omniquant` — [OmniQuant：联合学习 clipping 与等价变换](wiki/algorithms/omniquant.md)
- `algorithm-quarot` — [QuaRot：旋转坐标中的低比特推理](wiki/algorithms/quarot.md)
- `algorithm-smoothquant` — [SmoothQuant：在权重与激活之间迁移量化难度](wiki/algorithms/smoothquant.md)
- `algorithm-spinquant` — [SpinQuant：学习正交旋转以降低量化损失](wiki/algorithms/spinquant.md)
- `algorithm-svdquant` — [SVDQuant：低秩分支吸收异常值](wiki/algorithms/svdquant.md)

## 指南

- `guide-algorithm-selection` — [从量化问题选择候选方法](wiki/guides/algorithm-selection.md)
- `guide-architecture-transfer` — [从 AR 到 MoE、dLLM、DiT 与 VLM：量化迁移需要的证据](wiki/guides/architecture-transfer.md)
- `guide-deployment-evidence` — [从论文 W4A4 到真实部署证据](wiki/guides/deployment-evidence.md)

## Recipe

- `recipe-composition` — [算法组合：前提、顺序与统计失效](wiki/recipes/composition.md)

## 案例

- `case-paper-evidence` — [量化论文性能证据：案例导航与可比边界](wiki/cases/paper-evidence.md)

## 按机制标签

- `activation`：[QuaRot：旋转坐标中的低比特推理](wiki/algorithms/quarot.md)；[SmoothQuant：在权重与激活之间迁移量化难度](wiki/algorithms/smoothquant.md)
- `ar`：[从 AR 到 MoE、dLLM、DiT 与 VLM：量化迁移需要的证据](wiki/guides/architecture-transfer.md)
- `architecture`：[从 AR 到 MoE、dLLM、DiT 与 VLM：量化迁移需要的证据](wiki/guides/architecture-transfer.md)
- `benchmark`：[量化论文性能证据：案例导航与可比边界](wiki/cases/paper-evidence.md)
- `block-reconstruction`：[BRECQ：块级重构与输出敏感度](wiki/algorithms/brecq.md)；[OmniQuant：联合学习 clipping 与等价变换](wiki/algorithms/omniquant.md)
- `calibration`：[AWQ：激活感知的权重量化](wiki/algorithms/awq.md)；[基础基线：RTN、MinMax、AbsMax 与 MSE 范围校准](wiki/algorithms/baselines.md)；[GPTQ：基于输入二阶统计的权重重构](wiki/algorithms/gptq.md)；[SmoothQuant：在权重与激活之间迁移量化难度](wiki/algorithms/smoothquant.md)；[量化术语与最小数值契约](wiki/foundations/quantization-contract.md)；[从量化问题选择候选方法](wiki/guides/algorithm-selection.md)；[从 AR 到 MoE、dLLM、DiT 与 VLM：量化迁移需要的证据](wiki/guides/architecture-transfer.md)；[算法组合：前提、顺序与统计失效](wiki/recipes/composition.md)
- `clipping`：[AutoRound / SignRound：用符号梯度联合优化舍入与裁剪](wiki/algorithms/autoround.md)；[基础基线：RTN、MinMax、AbsMax 与 MSE 范围校准](wiki/algorithms/baselines.md)；[FlatQuant：可学习可逆变换与块重构](wiki/algorithms/flatquant.md)；[OmniQuant：联合学习 clipping 与等价变换](wiki/algorithms/omniquant.md)
- `comparison`：[量化论文性能证据：案例导航与可比边界](wiki/cases/paper-evidence.md)
- `compensation`：[SVDQuant：低秩分支吸收异常值](wiki/algorithms/svdquant.md)
- `composition`：[算法组合：前提、顺序与统计失效](wiki/recipes/composition.md)
- `cost-constraint`：[ModelOpt AutoQuantize：受成本约束的格式搜索](wiki/algorithms/modelopt-autoquantize.md)
- `denoising`：[从 AR 到 MoE、dLLM、DiT 与 VLM：量化迁移需要的证据](wiki/guides/architecture-transfer.md)
- `deployment`：[量化术语与最小数值契约](wiki/foundations/quantization-contract.md)；[从论文 W4A4 到真实部署证据](wiki/guides/deployment-evidence.md)
- `diagnosis`：[从量化问题选择候选方法](wiki/guides/algorithm-selection.md)
- `diffusion`：[SVDQuant：低秩分支吸收异常值](wiki/algorithms/svdquant.md)
- `dit`：[SVDQuant：低秩分支吸收异常值](wiki/algorithms/svdquant.md)；[从 AR 到 MoE、dLLM、DiT 与 VLM：量化迁移需要的证据](wiki/guides/architecture-transfer.md)
- `dllm`：[从 AR 到 MoE、dLLM、DiT 与 VLM：量化迁移需要的证据](wiki/guides/architecture-transfer.md)
- `equivalent-transformation`：[QuaRot：旋转坐标中的低比特推理](wiki/algorithms/quarot.md)；[SmoothQuant：在权重与激活之间迁移量化难度](wiki/algorithms/smoothquant.md)
- `fisher`：[BRECQ：块级重构与输出敏感度](wiki/algorithms/brecq.md)；[ModelOpt AutoQuantize：受成本约束的格式搜索](wiki/algorithms/modelopt-autoquantize.md)
- `format`：[量化术语与最小数值契约](wiki/foundations/quantization-contract.md)
- `hessian`：[GPTQ：基于输入二阶统计的权重重构](wiki/algorithms/gptq.md)；[HAWQ：用 Hessian 敏感度选择混合精度](wiki/algorithms/hawq.md)
- `joint-optimization`：[OmniQuant：联合学习 clipping 与等价变换](wiki/algorithms/omniquant.md)
- `kernel`：[量化论文性能证据：案例导航与可比边界](wiki/cases/paper-evidence.md)；[从论文 W4A4 到真实部署证据](wiki/guides/deployment-evidence.md)
- `kv-cache`：[FlatQuant：可学习可逆变换与块重构](wiki/algorithms/flatquant.md)；[QuaRot：旋转坐标中的低比特推理](wiki/algorithms/quarot.md)；[SpinQuant：学习正交旋转以降低量化损失](wiki/algorithms/spinquant.md)
- `latency`：[量化论文性能证据：案例导航与可比边界](wiki/cases/paper-evidence.md)
- `learned-transform`：[FlatQuant：可学习可逆变换与块重构](wiki/algorithms/flatquant.md)；[SpinQuant：学习正交旋转以降低量化损失](wiki/algorithms/spinquant.md)
- `llm`：[AutoRound / SignRound：用符号梯度联合优化舍入与裁剪](wiki/algorithms/autoround.md)
- `low-rank`：[SVDQuant：低秩分支吸收异常值](wiki/algorithms/svdquant.md)
- `mask`：[从 AR 到 MoE、dLLM、DiT 与 VLM：量化迁移需要的证据](wiki/guides/architecture-transfer.md)
- `mixed-precision`：[HAWQ：用 Hessian 敏感度选择混合精度](wiki/algorithms/hawq.md)；[ModelOpt AutoQuantize：受成本约束的格式搜索](wiki/algorithms/modelopt-autoquantize.md)
- `modelopt`：[ModelOpt AutoQuantize：受成本约束的格式搜索](wiki/algorithms/modelopt-autoquantize.md)
- `moe`：[从 AR 到 MoE、dLLM、DiT 与 VLM：量化迁移需要的证据](wiki/guides/architecture-transfer.md)
- `outlier`：[AWQ：激活感知的权重量化](wiki/algorithms/awq.md)
- `paper-evidence`：[量化论文性能证据：案例导航与可比边界](wiki/cases/paper-evidence.md)
- `performance`：[从论文 W4A4 到真实部署证据](wiki/guides/deployment-evidence.md)
- `ptq`：[AdaRound：固定量化网格上的自适应舍入](wiki/algorithms/adaround.md)；[AutoRound / SignRound：用符号梯度联合优化舍入与裁剪](wiki/algorithms/autoround.md)；[AWQ：激活感知的权重量化](wiki/algorithms/awq.md)；[BRECQ：块级重构与输出敏感度](wiki/algorithms/brecq.md)；[FlatQuant：可学习可逆变换与块重构](wiki/algorithms/flatquant.md)；[SpinQuant：学习正交旋转以降低量化损失](wiki/algorithms/spinquant.md)；[SVDQuant：低秩分支吸收异常值](wiki/algorithms/svdquant.md)
- `qat`：[HAWQ：用 Hessian 敏感度选择混合精度](wiki/algorithms/hawq.md)
- `quantizer`：[基础基线：RTN、MinMax、AbsMax 与 MSE 范围校准](wiki/algorithms/baselines.md)；[量化术语与最小数值契约](wiki/foundations/quantization-contract.md)
- `reconstruction`：[AdaRound：固定量化网格上的自适应舍入](wiki/algorithms/adaround.md)；[AutoRound / SignRound：用符号梯度联合优化舍入与裁剪](wiki/algorithms/autoround.md)；[FlatQuant：可学习可逆变换与块重构](wiki/algorithms/flatquant.md)
- `rotation`：[QuaRot：旋转坐标中的低比特推理](wiki/algorithms/quarot.md)；[SpinQuant：学习正交旋转以降低量化损失](wiki/algorithms/spinquant.md)；[算法组合：前提、顺序与统计失效](wiki/recipes/composition.md)
- `rounding`：[AdaRound：固定量化网格上的自适应舍入](wiki/algorithms/adaround.md)；[AutoRound / SignRound：用符号梯度联合优化舍入与裁剪](wiki/algorithms/autoround.md)；[BRECQ：块级重构与输出敏感度](wiki/algorithms/brecq.md)
- `rtn`：[基础基线：RTN、MinMax、AbsMax 与 MSE 范围校准](wiki/algorithms/baselines.md)
- `scaling`：[AWQ：激活感知的权重量化](wiki/algorithms/awq.md)；[OmniQuant：联合学习 clipping 与等价变换](wiki/algorithms/omniquant.md)；[SmoothQuant：在权重与激活之间迁移量化难度](wiki/algorithms/smoothquant.md)；[算法组合：前提、顺序与统计失效](wiki/recipes/composition.md)
- `search`：[ModelOpt AutoQuantize：受成本约束的格式搜索](wiki/algorithms/modelopt-autoquantize.md)
- `selection`：[从量化问题选择候选方法](wiki/guides/algorithm-selection.md)
- `sensitivity`：[HAWQ：用 Hessian 敏感度选择混合精度](wiki/algorithms/hawq.md)
- `transfer`：[从 AR 到 MoE、dLLM、DiT 与 VLM：量化迁移需要的证据](wiki/guides/architecture-transfer.md)
- `validation`：[从论文 W4A4 到真实部署证据](wiki/guides/deployment-evidence.md)
- `vision`：[AdaRound：固定量化网格上的自适应舍入](wiki/algorithms/adaround.md)；[BRECQ：块级重构与输出敏感度](wiki/algorithms/brecq.md)
- `vlm`：[从 AR 到 MoE、dLLM、DiT 与 VLM：量化迁移需要的证据](wiki/guides/architecture-transfer.md)
- `w4a4`：[FlatQuant：可学习可逆变换与块重构](wiki/algorithms/flatquant.md)；[SpinQuant：学习正交旋转以降低量化损失](wiki/algorithms/spinquant.md)；[SVDQuant：低秩分支吸收异常值](wiki/algorithms/svdquant.md)
- `weight-only`：[AWQ：激活感知的权重量化](wiki/algorithms/awq.md)；[GPTQ：基于输入二阶统计的权重重构](wiki/algorithms/gptq.md)
- `weight-reconstruction`：[GPTQ：基于输入二阶统计的权重重构](wiki/algorithms/gptq.md)
