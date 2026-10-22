# 学习变换与低秩方法：来源登记

阅读日期：2026-10-23。论文读取 arXiv 固定版本 HTML 正文、公式和表格；源码读取固定 commit。下列研究结论未在本仓库运行模型验证。论文版本与当前代码版本的差异应保留，不将后续实现支持回填为原文证据。

| ID | 类型 | URL | 版本与已读定位 |
| --- | --- | --- | --- |
| `paper-spinquant` | 论文 | https://arxiv.org/html/2405.16406v2 | v2；§2–4，Eq.1–3，Figure 1/5/6，Table 1/2；学习旋转、Cayley 约束、RTN/GPTQ 两阶段配置与实验设置 |
| `repo-spinquant` | 官方源码 | https://github.com/facebookresearch/SpinQuant/tree/8f47aa3f00e8662caf1a484153920a07e5281c3a | `optimize_rotation.py` 的参数冻结、R1/R2、数据与产物；`train_utils/optimizer.py` 的 Cayley_loop/SGDG；`ptq.py` 的 PTQ/PPL 流程 |
| `paper-flatquant` | 论文 | https://arxiv.org/html/2410.09426v1 | v1；§3–4，Eq.2–4，Table 1/2；Kronecker 可逆变换、块重构、在线融合、校准与量化设置 |
| `repo-flatquant` | 官方源码 | https://github.com/ruikangliu/FlatQuant/tree/9d88ffcb7d2c6bda59fb5c44dad36adc101aadb1 | `flatquant/trans_utils.py` SVD 参数化；`train_utils.py` 块输入缓存与训练；`flat_linear.py` 重参数化；README/REALQUANT.md 区分 fake/real quant |
| `paper-svdquant` | 论文 | https://arxiv.org/html/2411.05007v1 | v1；§4–5，Eq.5–7，Proposition 4.2，Figure 4–6，Table 1/2；低秩分解、残差迭代、Nunchaku 融合、diffusion 实验与格式 |
| `repo-svdquant` | 官方源码 | https://github.com/mit-han-lab/deepcompressor/tree/69f3473f5e1c1504bae35cc50c7858ef900a9b17 | `nn/patch/lowrank.py` SVD 与分支；`calib/lowrank.py` 候选更新；`examples/diffusion/configs/svdquant/__default__.yaml` 与基础配置 |
| `repo-nunchaku` | 官方源码 | https://github.com/mit-han-lab/nunchaku/tree/302e0e97024ebd68688fe890e5df83731edf7b54 | `src/Linear.cpp` 的 GEMM_W4A4::quantize；`gemm_w4a4.cuh` 的 quantize_w4a4_fuse_lora_kernel；`gemm_w4a4_launch_impl.cuh` 的 launch_lora；仅静态调用链核验 |

## 尚需运行验证的边界

- 三种方法的校准、权重导出与实际模型质量均未在本 Wiki 环境执行。
- 目标后端是否具备在线变换或两分支融合 kernel，需结合目标 GPU、格式和实际导出产物验证。
- FlatQuant 当前仓库新增模型支持与论文 v1 范围不同；SpinQuant 当前处理 tied embeddings 的方式也需要计入部署约束。
- SVDQuant 的图像 diffusion 证据不能证明文本 dLLM、AR 或 MoE 的精度及性能。
