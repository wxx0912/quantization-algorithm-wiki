---
{
  "id": "paper-llm-mq",
  "title": "LLM-MQ: Mixed-precision Quantization for Efficient LLM Deployment（NeurIPS 2023 ENLSP Workshop）",
  "category": "paper",
  "url": "https://neurips2023-enlsp.github.io/papers/paper_4.pdf",
  "version": "ENLSP 2023 Workshop PDF（正文 5 页 + 附录 2 页；未检索到 arXiv 条目）",
  "version_kind": "paper-version",
  "retrieved_at": "2026-10-28",
  "read_form": [
    "pdf"
  ],
  "read_locator": "正文 5 页（Abstract、§1 Introduction、§2.1 Sparse Outlier Protection、§2.2 Sensitivity-based Precision Allocation、Eq.1–5、§2.3 Efficient GPU Kernel Design、§3 Experiments 含 Table 1 全表与效率段、Acknowledgement、References）；附录 2 页（Table 2 LLaMA2-7B、Table 3 OPT-6.7B、Table 4 OPT-13B 全表）",
  "topic": "mixed-architecture",
  "excerpt_sha256": null
}
---
## 这份来源是什么

LLM-MQ 论文（NeurIPS 2023 Efficient Natural Language and Speech Processing Workshop 相机版），作者 Shiyao Li、Xuefei Ning、Ke Hong、Tengxuan Liu、Luning Wang、Xiuhong Li、Kai Zhong、Guohao Dai、Huazhong Yang、Yu Wang（清华大学电子工程系/BNRist、Infinigence-AI、北京大学、上海交通大学）。它把"给定权重显存预算下的位宽分配"做成一个整数规划问题，并配套稀疏离群值保护与融合反量化-GEMV 的 CUDA kernel。它是 `algorithm-llm-mq` 的一手依据。

- 获取方式：workshop 官网 PDF（正文 `papers/paper_4.pdf`、附录 `papers/paper_4_supp.pdf`）；同一版 PDF 另有清华大学 NICS 实验室站点副本（用户提供，本页未逐字节比对）。文件指纹：正文 sha256 `9a2101ae6befaff5566fc1ed1bd78cbf10ccb4a7e2c3ae2b985e9c3845562dc0`，附录 sha256 `c8c8c784b80a238dacb7fb082bed1e2cee98cfcb8256b6bbd3a1291fb63e258f`。
- 时间线（workshop 官网日程页）：投稿截止 2023-10-02、录用通知 2023-10-27、相机版提交 2023-11-03、workshop 2023-12-16。这些是 workshop 日程，不等于该 PDF 的确切发布日；PDF 没有版本号，本页只以文件指纹锚定版本。
- 范围边界：只做**权重（weight-only）**量化，不涉及激活或 KV cache；实验为 LLaMA2 与 OPT 稠密模型，不含 MoE、DiT、dLLM。
- 与相邻来源的分工：位宽分配只用**一阶**梯度信息（§2.2 明确说明约 15% 层的 Hessian 非半正定），与 `source-hawq2` 的 Hessian-trace 路线不同；稀疏离群值保护复用 SqueezeLLM 的稠密-稀疏（CSR）思路（§2.1 原文写 "akin to the SqueezeLLM approach"）。

## 已读定位与阅读形式

- 阅读形式：PDF（用 pypdf 抽取文本）；未检索到该工作的 arXiv 条目，因此未读 LaTeX 源码或 HTML 渲染版；公式与表格按抽取文本核对，未逐像素比对排版。
- 已读：Abstract、§1、§2.1、§2.2（Eq.1–5）、§2.3、§3 全文含 Table 1 全表与效率段、Acknowledgement、References；附录 A 的 Table 2/3/4 全表逐行读。
- 未做：未查官方代码仓库、未复现、未做 GPU 实测。

## 关键事实

- 三个组成部分（Abstract、§1）：稀疏离群值保护、基于敏感度的位宽分配、融合反量化与 GEMV 的 CUDA kernel。
- 稀疏离群值保护（§2.1）：每层保留绝对值最大的 0.5% 权重为 FP16（CSR 格式），其余权重做 2-bit 均匀量化（§2.1 用它展示保护效果；最终每层位宽由 §2.2 分配）。论文给出的量化误差：2-bit 0.0075、3-bit 0.0032（2-bit 约为 3-bit 的 2.3 倍），保护 0.5% 离群值后 2-bit 误差降到 0.0048，即降到约 1/1.56，接近 3-bit 水平（Figure 2）。Figure 1(a) 用 PPL 示意（直接 2-bit 为 1e+8，保护后为 16.24，图中未标注具体模型）。
- 敏感度位宽分配（§2.2）：先用一阶 Taylor 近似 ΔL ≈ g_iᵀ(W_i − Q_b(W_i))（Eq.1；此处逐字沿用论文写法，若定义 ΔL = L(Q(W)) − L(W)，差值方向应为 Q(W) − W，符号相反；§2.2 对 s_{i,b} 取绝对值，位宽分配目标不受影响）；论文指出 LLM 中约 15% 层的 Hessian 非半正定（未收敛到局部极小），二阶信息不能准确反映损失变化，因此只用一阶项。令 `s_{i,b} = |g_iᵀ(W_i − Q_b(W_i))|`，在权重显存预算 B 的约束下最小化 `Σ_i Σ_b c_{i,b}·s_{i,b}`，每个层只取一个位宽且位宽属于 {2,3,4}（Eq.3–5），用整数规划求解器在数秒内得到分配方案。
- Kernel（§2.3）：4-bit 两个权重存一字节、3-bit 八个存三字节、2-bit 四个存一字节；把反量化融合进 GEMV 以减少访存。选择 CUDA Core 而非 Tensor Core 的两个理由：Tensor Core 要求输入每个维度至少为 8，而 GEMV 的最小维度是 1，零填充扩维低效；GEMV 延迟主要由权重访存决定，Tensor Core 的额外算力无法缓解该瓶颈。此外为奇数位宽设计了访存排布。
- 实验设置（§3）：weight-only、group size 128、沿用 AWQ 的重参数化；模型为 LLaMA2-13B（Table 1）与附录中的 LLaMA2-7B（Table 2）、OPT-6.7B（Table 3）、OPT-13B（Table 4）；指标为 PIQA、HellaSwag、WinoGrande、ARC-e、OpenBookQA 五个零样本任务的平均准确率，加 Wiki benchmark 困惑度（原文记作 "the Wiki benchmark"，参考文献 [9] 为 Merity 等 Pointer Sentinel Mixture Models），评测代码基于 lm-evaluation-harness。论文声明平均位宽不计 FP16 稀疏权重，最多只会使其增加 0.08 bit。
- Table 1（LLaMA2-13B）：FP16 平均 65.08、Wiki 7.89；LLM-MQ 4.0 bit 64.96 / 8.03，3.6 bit 65.10 / 8.23，3.4 bit 64.52 / 8.61，3.0 bit 64.29 / 8.54，2.8 bit 64.03 / 8.83，2.0 bit 60.23 / 12.17；对照 RTN 2.0 bit 41.74 / 1056.33，AWQ 2.0 bit 37.06 / 5e6（原文记作 `5e6`），SqueezeLLM 只报 4.0 bit 64.93 / 8.44 与 3.0 bit 63.50 / 9.29，没有 2-bit 行。
- 论文 §3 正文称 2-bit 时零样本平均准确率比 AWQ、RTN 高 23.17 与 19.16 个点；其中 23.17 与 Table 1 的 60.23 − 37.06 相符，19.16 与 60.23 − 41.74 = 18.49 不相符（按本页读取的表格计算）。
- 附录示例（2.0 bit，平均准确率 / Wiki PPL）：LLaMA2-7B 55.61 / 15.93（FP16 62.31 / 8.79），OPT-6.7B 57.63 / 17.09（FP16 61.22 / 12.29），OPT-13B 57.36 / 17.24（FP16 62.50 / 11.49）。
- 效率（§3 末段）：INT4 量化的 LLaMA2-7B 部署在 NVIDIA T4（16 GB），输出 token 数为 32、128、512、1024、2048 时，相对 PyTorch FP16 基线的端到端加速分别为 1.27、1.42、1.65、1.64、1.45 倍（Abstract 写 "up to 1.6×"）；单算子口径上，4096×4096 与 4096×11008 的 GEMV kernel 相对 cuBLAS FP16 GEMV 分别快 2.33 与 2.48 倍。
- 论文明确说明 2-bit 与 3-bit kernel 相对 4-bit 没有显著加速，原因是每个位宽下每线程处理固定数据量，低比特时带宽利用率更低。

## 不能推出的结论

- 本页只整理 workshop PDF 的内容，未读 LaTeX 源码（未检索到该工作的 arXiv 条目）、未查官方代码、未复现、未做任何 GPU 实测；论文数字不能当作本项目实测。
- 论文只写"整数规划求解器"（"efficient integer programming solver"），未给出具体求解器、建模工具或求解时间数据；不能据外部讨论推断其使用了 CVXPY 之类的工具。
- 效率结论仅在 NVIDIA T4（16 GB）、INT4 LLaMA2-7B、输出 32–2048 token 条件下测得，且论文没有报告 2-bit/3-bit 的端到端加速；不能把这些加速外推到其他硬件、位宽、batch 或推理后端。
- Table 1/附录的对照只有 RTN、AWQ、SqueezeLLM，没有 GPTQ、QuIP# 等 2-bit 方法的横向对比；不能据此认为 LLM-MQ 优于所有低比特方案。
- weight-only：不能推出对激活量化、KV cache、MoE 路由或 DiT/dLLM 模型的收益。
- 稀疏 FP16 分量需要 CSR 稀疏计算与自定义 kernel 支持（§2.1、§2.3）；论文未报告在通用推理后端上的可用性，不能假定目标后端已支持该混合表示。
- 本页未核查 Figure 1(a) 中 PPL 1e+8 / 16.24 对应的具体模型与位宽配置（图中未标注）。
