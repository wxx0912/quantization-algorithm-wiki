---
{
  "id": "paper-qdit",
  "title": "Q-DiT: Accurate Post-Training Quantization for Diffusion Transformers",
  "category": "paper",
  "url": "https://arxiv.org/abs/2406.17343",
  "version": "v1（2024-06-25）",
  "version_kind": "paper-version",
  "retrieved_at": "2026-10-25",
  "read_form": [
    "html"
  ],
  "read_locator": "摘要、§1、§2、§3.1–3.4（Observation ❶–❹、Eq.1–3、Alg.1、Table 1）、§4.1 设置、§4.2 主结果（Table 2–3）",
  "topic": "mixed-architecture",
  "excerpt_sha256": null
}
---
## 这份来源是什么

Q-DiT 原始论文（arXiv 2406.17343，v1），研究 DiT（Diffusion Transformer）的 PTQ。论点是：为 ViT 和 UNet 类扩散模型设计的 PTQ 直接套到 DiT 上会明显掉点，因为 DiT 的权重与激活方差分布不同。它提出细粒度分组量化、动态激活量化和组大小自动搜索三项，在 ImageNet 上量化 DiT-XL/2。

它是 `algorithm-qdit` 的一手依据，也是 `guide-architecture-transfer` 与 `guide-search-space` 中"生成感知量化（generation-aware）"类别的直接证据。

范围边界：**PTQ**（无重训）；权重用 GPTQ、激活动态量化；主实验 W8A8 / W4A8；模型指定为 DiT-XL/2，数据集 ImageNet 256×256 与 512×512。

## 已读定位与阅读形式

- 阅读形式：arXiv HTML 渲染版（`arxiv.org/html/2406.17343v1`），正文与表格按 HTML 结构解析；未读 LaTeX 源码或 PDF。
- 定位：观察 ❶❷（权重输入通道方差、激活结构性离群）在 §3.2、观察 ❸（激活随 timestep 漂移）在 §3.3、观察 ❹（组大小非单调）与 Eq.2–3、Alg.1、Table 1 在 §3.4；§4.1 设置、§4.2 主结果 Table 2–3。

## 关键事实

- **四条观察**（§3.2–3.4）：❶ 权重沿输入通道的方差显著大于输出通道，常用"沿输出通道"量化会吃亏；❷ 激活在特定通道存在结构性离群，per-tensor 量化放大误差；❸ 激活分布随 denoising timestep 漂移；❹ 组大小与质量非单调（减小 group 不一定更好）。
- **三项技术**（§3.2–3.4）：细粒度分组量化（权重与激活同层同 group，沿输入通道方向组织）；动态激活量化（运行时按 min-max 逐组算 scale，并把 min-max 计算融合进前一个算子，故论文称开销相对 matmul 可忽略）；自动组大小分配（用进化算法搜每层 group size，目标函数是量化模型生成样本与真实样本之间的 **FID**，约束 BitOps，见 Eq.2–3、Alg.1）。
- **设置**（§4.1）：DiT-XL/2，ImageNet 256×256/512×512，DDIM 50/100 步，含/不含 classifier-free guidance；权重与激活均非对称量化，权重用 GPTQ；默认 group 128，搜索空间 {32,64,128,192,288}，同一层权重与激活用同一 group；基线 PTQ4DM、RepQ-ViT、GPTQ。
- **主结果**（§4.2，Table 2，256×256，steps=100、无 CFG）：FP FID 12.40；W8A8 下 PTQ4DM 20.91、RepQ-ViT 13.19、GPTQ 16.44、Q-DiT **11.93**（IS 119.86）；W4A8 下 PTQ4DM 252.31、RepQ-ViT 315.85、GPTQ 25.48、Q-DiT **15.76**（IS 98.78）。
- **512×512**（Table 3）：W8A8 下 Q-DiT FID 16.23 与 RepQ-ViT 16.01 接近、略逊；W4A8 与 CFG 配置下 Q-DiT 保持领先（如 4/8 Ours 21.59、cfg 1.5 的 4/8 Ours 7.82）。所以『趋势一致』只对 W4A8 成立，高分辨率 W8A8 并非最优。
- **组大小非单调证据**（Table 1）：256×256 group128 FID 17.87 → group96 19.97（论文称上升约 11.8%）；512×512 group96 20.76 → group64 20.90。据此论文认为存在比"越小越好"更优的配置。

## 不能推出的结论

- 本页只读正文到 §4.2；未读 §4.3 消融、§5 结论与附录，也未复现实验。表格数值按 arXiv HTML 结构的解析结果记录；若与正式版排版冲突，以原文为准。
- FID/sFID/IS 是生成分布指标，不等于任务正确率或人类偏好；W4A8 的 FID 15.76 仍高于 FP 的 12.40，"维持高保真"是相对同配置方法而言，不能读成"W4A8 无损"。
- 自动搜索以 FID 为目标；换评测指标（如 CLIP score）或数据集，最优 group 配置可能不同，不能把搜到的配置当作跨模型通用规则。
- 动态激活量化有逐组 min-max 的运行时成本，论文称融合后可忽略，但这是其特定实现下的结论，不能外推到未做算子融合的后端。
- 结论只在 DiT-XL/2 + ImageNet 上成立，不能自动迁移到文本 dLLM、视频 DiT、别的分辨率或别的采样器。
