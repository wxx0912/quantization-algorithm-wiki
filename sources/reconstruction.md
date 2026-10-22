# 缩放、舍入与块重构来源

记录实际阅读的原始资料。论文通过全文抽取阅读所列章节、公式和实验表；源码版本用于锁定本次观察，不表示它是原论文实验版本。仓库当前新增功能与论文结论分别记录。

| ID | 类型 | URL | 版本与已读定位 |
|---|---|---|---|
| `paper-awq` | 论文 | https://arxiv.org/pdf/2306.00978v5 | v5，2024-07-18；§3.1–3.2 式 (4)–(5)，§5.1，表 4、7、9，TinyChat 系统说明；同时读取 HTML v5 |
| `repo-awq` | 官方源码 | https://github.com/mit-han-lab/llm-awq/tree/d6e797a42b9ef7778de8ee2352116e0f48a78d61 | README；awq/quantize/auto_scale.py 的 scale 搜索/融合；auto_clip.py 的局部误差与 Q/K 排除 |
| `paper-adaround` | 论文 | https://arxiv.org/pdf/2004.10568v2 | v2，2020-06-30；§3.1–3.3 式 (21)–(25)，§5 实验设置，表 5–6；固定网格与非对称重构；另读 https://arxiv.org/src/2004.10568v2 的 sections/05_experiments.tex 与 tables/exp_05_ste.tex 核对校准预算、基线和表5结果（排除注释与旧表） |
| `repo-adaround` | 作者机构维护实现 | https://github.com/quic/aimet/tree/ec4bc578681beaa52043bc83a228d5e5a855dd15 | TrainingExtensions/torch/src/python/aimet_torch/_base/adaround/adaround_optimizer.py；AdaroundOptimizer.adaround_module、_optimize_rounding；非论文原始实验版本 |
| `paper-autoround` | 论文 | https://arxiv.org/pdf/2309.05516v5 | v5，2024-10-08；标题 Optimize Weight Rounding via Signed Gradient Descent for the Quantization of LLMs；§3 式 (2)–(6)、算法 1，§4.1–4.2、表 1–2；另读 https://arxiv.org/src/2309.05516v5 的 acl_latex.tex 核实公式与舍入/裁剪消融；不是 SignRound V2 |
| `repo-autoround` | 官方源码 | https://github.com/intel/auto-round/tree/a7656a27d020e636a961b35f84c551f453efe434 | README；auto_round/algorithms/quantization/sign_round/quantizer.py、sign_sgd.py；auto_round/data_type/int.py 的 quant_tensor_asym；docs/algorithm_combinations.md；已识别 V1/V2 与当前工具能力边界 |
| `paper-brecq` | 论文 | https://arxiv.org/pdf/2102.05426v2 | v2，2021-07-25；§3.2–3.3 式 (9)–(10)、算法 1，§4 表 1–4，附录 B.4.1 学习策略与 B.4.3 延迟获取；视觉网络证据范围 |
| `repo-brecq` | 官方源码 | https://github.com/yhhhli/BRECQ/tree/e455d62e93c70351961f8991c913b59435bd165f | README、quant/block_recon.py；block_reconstruction、LossFunction；区分 MSE 默认与 Fisher 可选模式 |

上述页面没有项目内量化或部署实测。AutoRound 当前格式与组合说明只证明文档/源码中存在对应接口和声明；实际 kernel 覆盖仍需按目标后端另行核实。
