# 来源索引

> 由 `python3 scripts/build_index.py` 生成，请勿手改。

来源页规范见 [SPEC.md](SPEC.md)，主题说明见 topics/。

## 基础与样章

主题说明：[topics/core.md](topics/core.md)

- `paper-gptq` — [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](papers/paper-gptq.md)（被 6 个页面引用）
- `paper-omniquant` — [OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models](papers/paper-omniquant.md)（被 4 个页面引用）
- `paper-quarot` — [QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs](papers/paper-quarot.md)（被 6 个页面引用）
- `paper-smoothquant` — [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](papers/paper-smoothquant.md)（被 4 个页面引用）
- `repo-gptq` — [IST-DASLab/gptq — 论文官方实现（fasterquant 与 Quant3Linear）](repos/repo-gptq.md)（被 3 个页面引用）
- `repo-omniquant` — [OpenGVLab/OmniQuant — 官方实现（LWC/LET 与量化器）](repos/repo-omniquant.md)（被 1 个页面引用）
- `repo-quarot` — [spcl/QuaRot — 官方参考实现（fake quant 与旋转工具）](repos/repo-quarot.md)（被 1 个页面引用）
- `repo-smoothquant` — [mit-han-lab/smoothquant — 官方平滑实现](repos/repo-smoothquant.md)（被 1 个页面引用）

## 重构与舍入

主题说明：[topics/reconstruction.md](topics/reconstruction.md)

- `paper-adaround` — [Up or Down? Adaptive Rounding for Post-Training Quantization](papers/paper-adaround.md)（被 1 个页面引用）
- `paper-autoround` — [Optimize Weight Rounding via Signed Gradient Descent for the Quantization of LLMs](papers/paper-autoround.md)（被 1 个页面引用）
- `paper-awq` — [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](papers/paper-awq.md)（被 3 个页面引用）
- `paper-brecq` — [BRECQ: Pushing the Limit of Post-Training Quantization by Block Reconstruction](papers/paper-brecq.md)（被 1 个页面引用）
- `repo-adaround` — [quic/aimet — AdaRound 的维护实现](repos/repo-adaround.md)（被 1 个页面引用）
- `repo-autoround` — [intel/auto-round — AutoRound/SignRound 官方实现与组合文档](repos/repo-autoround.md)（被 1 个页面引用）
- `repo-awq` — [mit-han-lab/llm-awq — 官方实现（auto_scale 与 auto_clip）](repos/repo-awq.md)（被 1 个页面引用）
- `repo-brecq` — [yhhhli/BRECQ — 官方实现（block_recon 与 LossFunction）](repos/repo-brecq.md)（被 1 个页面引用）

## 变换与低秩

主题说明：[topics/transforms.md](topics/transforms.md)

- `paper-flatquant` — [FlatQuant: Flatness Matters for LLM Quantization](papers/paper-flatquant.md)（被 1 个页面引用）
- `paper-spinquant` — [SpinQuant: LLM Quantization with Learned Rotations](papers/paper-spinquant.md)（被 1 个页面引用）
- `paper-svdquant` — [SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models](papers/paper-svdquant.md)（被 4 个页面引用）
- `repo-flatquant` — [ruikangliu/FlatQuant — 官方实现（变换参数化、块训练与重参数化）](repos/repo-flatquant.md)（被 1 个页面引用）
- `repo-nunchaku` — [nunchux-ai/nunchaku — SVDQuant 配套推理引擎（低秩与低比特融合 kernel）](repos/repo-nunchaku.md)（被 3 个页面引用）
- `repo-spinquant` — [facebookresearch/SpinQuant — 官方实现（旋转训练与 PTQ 流程）](repos/repo-spinquant.md)（被 1 个页面引用）
- `repo-svdquant` — [mit-han-lab/deepcompressor — SVDQuant 官方实现与配置](repos/repo-svdquant.md)（被 1 个页面引用）

## 混精与架构

主题说明：[topics/mixed-architecture.md](topics/mixed-architecture.md)

- `code-hawq` — [Zhen-Dong/HAWQ — 官方实现（V3 为主，含训练循环）](repos/code-hawq.md)（被 1 个页面引用）
- `code-modelopt-autoquantize` — [NVIDIA/Model-Optimizer — AutoQuantize 实现（受成本约束的格式搜索）](repos/code-modelopt-autoquantize.md)（被 2 个页面引用）
- `code-viditq` — [A-suozhang/ViDiT-Q — 官方实现（量化层与基础量化器）](repos/code-viditq.md)（被 3 个页面引用）
- `paper-awq-architecture` — [AWQ 论文中 MoE 与多模态部分（v5，架构迁移范围）](papers/paper-awq-architecture.md)（被 1 个页面引用）
- `paper-hawq2` — [HAWQ-V2: Hessian Aware trace-Weighted Quantization of Neural Networks（NeurIPS 2020 会议版）](papers/paper-hawq2.md)（被 1 个页面引用）
- `paper-llada-architecture` — [LLaDA: Large Language Diffusion Models（架构与过程范围）](papers/paper-llada-architecture.md)（被 1 个页面引用）
- `paper-viditq` — [ViDiT-Q: Efficient and Accurate Quantization of Diffusion Transformers for Image and Video Generation](papers/paper-viditq.md)（被 3 个页面引用）
- `source-hawq2` — [HAWQ-V2: Hessian Aware trace-Weighted Quantization of Neural Networks（arXiv 1911.03852v1 预印本 LaTeX）](papers/source-hawq2.md)（被 1 个页面引用）
- `source-viditq` — [ViDiT-Q LaTeX 源码包（arXiv src 2406.02540v2）](papers/source-viditq.md)（被 2 个页面引用）

