# 混合精度与架构证据

本表登记实际阅读范围。核实日期：2026-10-23。论文 HTML、LaTeX、会议 PDF 与当前代码可能不同版本，不把其中一个版本的支持或结果反写到另一个版本。原始下载文件不随 Wiki 发布。

| ID | 类型 | URL | 版本与已读定位 |
|---|---|---|---|
| `paper-hawq2` | 原始会议论文 | https://proceedings.neurips.cc/paper/2020/file/d77c703536718b95308130ff2e5cf9ee-Paper.pdf | NeurIPS 2020，正式标题 HAWQ-V2；已读 §1–3、Assumption 1、Eq. (4)–(10)、Table 1–3。PDF 第 8 页 Table 1(a)/(b)/(c) 为 75.92/75.98/68.68；第 9 页 Table 3(b) trace/L2 为 68.68/67.83，已渲染目视复核。 |
| `source-hawq2` | 原始 LaTeX | https://arxiv.org/src/1911.03852v1 | arXiv v1；已读 `s2_method.tex` 敏感度与 Pareto 公式、`s6_appendix.tex` Quantization Details/STE、RetinaNet trace 图设置与 Search Space；`s3_results.tex` 报告 75.76/75.68/68.38，与会议版不同，不替代会议版表格数值。 |
| `code-hawq` | 官方源码 | https://github.com/Zhen-Dong/HAWQ/tree/1616df69fdd99f100a8d8a6e78742e0a89292634 | 固定 commit；README、`quant_train.py::train/train_kd`。当前主展示 V3，含 QAT optimizer step；未声称复现 V2。 |
| `code-modelopt-autoquantize` | 官方源码 | https://github.com/NVIDIA/Model-Optimizer/tree/11c2a85160d8b3a84e16e45759a842ce4fb1f13b/modelopt/torch/quantization | 固定 commit；`model_quant.py::auto_quantize`、`algorithms.py` 的分组/评分/LP/KL 搜索与 compression、`_auto_quantize_cost.py`、`_auto_quantize_shapley.py` 模块机制和标志、`kv_cache_auto_quant.py` 候选与成本校验及 searcher 入口；未运行模型。 |
| `paper-awq-architecture` | 原始论文 HTML | https://arxiv.org/html/2306.00978v5 | v5，2024-07-18；已读 §5 settings、Table 4–7、VLM 实验正文；Mixtral 的 Wikitext2、OpenFlamingo 仅语言部分、VILA 实验范围。 |
| `paper-llada-architecture` | 原始论文 HTML | https://arxiv.org/html/2502.09992v2 | v2；已读 §2、Fig. 2 与 mask/remasking 过程；只用于架构与过程事实，不作为量化实验来源。 |
| `paper-viditq` | 原始论文 HTML | https://arxiv.org/html/2406.02540v2 | v2，2024-06-30；已读 §3、§4、Table 1、Appendix A.1/A.4；动态逐 token、时间步缩放/混精与缺少 W4A8 kernel 的性能口径。 |
| `source-viditq` | 原始 LaTeX | https://arxiv.org/src/2406.02540v2 | v2 `ms.tex`：Eq. (1) 缩放公式、`tab:main_vbench`、§4.1 DDIM/CFG 与 MP 设置、附录 Hardware Resource Savings；复核表格数值，忽略注释中的废弃配置。 |
| `code-viditq` | 官方源码 | https://github.com/A-suozhang/ViDiT-Q/tree/a8add8be17ad91069aba7a7209aa88e04b2fe41e | 固定 commit；README、`quant_utils/qdiff/viditq/viditq_quant_layer.py`、`quant_utils/qdiff/base/base_quantizer.py`、`kernels/README.md`；当前含旋转与 CUDA 路径，区别于论文 v2，不声称本项目已跑 kernel。 |
