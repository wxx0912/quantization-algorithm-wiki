# 混精与架构

HAWQ、ModelOpt AutoQuantize、LLaDA、ViDiT-Q、Q-DiT、DLLMQuant 与 MoE/VLM 相关证据的来源。这一组跨越混合精度搜索、文本扩散语言模型与视频/图像生成模型的架构事实。HAWQ 的结果含量化感知微调，不能当作纯 PTQ；ModelOpt AutoQuantize 未找到可覆盖整个接口的统一算法论文（仅 `aumann_shapley` 评分模块 `_auto_quantize_shapley.py` 引用 arXiv:2607.12266），页面结论来自固定 commit 的源码核对与仓库内文档（其方法脉络引 OBS/LLM-MQ/SqueezeLLM）；LLaDA 来源页只提供结构与过程事实（不含 §3 实验数值），AWQ 架构页则含量化结果（MoE/VLM 等原论文实验），不属独立复现。Q-DiT 与 DLLMQuant 是生成感知量化的直接论文证据，以 arXiv HTML 渲染版读取，未读 LaTeX 源码或复现实验；NVFP4 来源页只摘取 QAD 报告 §2.1 的格式定义，不含其方法与实验。

本主题的来源清单与引用计数见 [../INDEX.generated.md](../INDEX.generated.md)。
