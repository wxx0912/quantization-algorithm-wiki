---
{
  "id": "paper-adaround",
  "title": "Up or Down? Adaptive Rounding for Post-Training Quantization",
  "category": "paper",
  "url": "https://arxiv.org/abs/2004.10568",
  "version": "v2（2020-06-30）",
  "version_kind": "paper-version",
  "retrieved_at": "2026-10-24",
  "read_form": [
    "latex",
    "html"
  ],
  "read_locator": "摘要、§1、§2（例 1、Table 1）、§3.1（Eq.9–13）、§3.2（Eq.14–20）、§3.3（Eq.21–25）、§4（相关工作）、§5.1（实验设置、Table 2–6、Figure 4）、§5.1 末（Table 7 的浮动体，讨论在 §5.2）、§5.2（Table 7–9）、附录 A（Table 10）、附录 B",
  "topic": "reconstruction",
  "excerpt_sha256": null
}
---
## 这份来源是什么

AdaRound 原始论文（arXiv 2004.10568，v2；ICML 2020）。核心主张是：给定一个**已经固定的量化网格**，"四舍五入到最近"并非最优。论文把每层的舍入决策建模为二次无约束二值优化（QUBO），再用连续松弛 + 正则退火求近似解，使层输出（进而任务损失）的重构误差最小。它只需少量**无标签**校准数据，不做端到端微调，对卷积与全连接层通用。

它是 `algorithm-adaround` 的直接依据（该页登记 `paper-adaround` 与 `repo-adaround`）。BRECQ、AutoRound 等把它当作前序与对比对象，但这两页没有把本来源登记为 source。

范围边界：实验对象是传统视觉 CNN（ResNet18/50、InceptionV3、MobileNetV2 分类、DeeplabV3+ 分割），主结果是 **W4A32**，激活量化只作为附带实验做到 8-bit；量化步长 `s` 在优化前已确定，论文不联合优化步长与舍入。

## 已读定位与阅读形式

- 阅读形式：arXiv e-print 的 LaTeX 源（`main.tex` + `sections/01_introduction.tex`…`06_conclusion.tex` + `tables/*.tex`，`\appendixtrue` 打开附录），另读 arXiv HTML 渲染版交叉核对编号；v2 = 2020-06-30。
- 定位：§2 动机（Table 1）、§3.1 任务损失舍入（Eq.9–13）、§3.2 局部 MSE 推导（Eq.14–20）、§3.3 AdaRound（Eq.21–25）、§5.1 消融（Table 2–6、Figure 4）、§5.2 文献对比（Table 7–9）、附录 A（Table 10）。编号：主文 Eq.1–25 + §5 的 Eq.26、Table 1–9、Figure 1–4；附录编译后另有 Table 10 与 Eq.27 起的公式。

## 关键事实

- **动机：最近舍入常次优**（§2、Table 1）。二阶泰勒展开下任务损失增量约 `Δwᵀg + ½ΔwᵀHΔw`（Eq.2–4）；只有对角项时最近舍入最优，非对角项 `Δw₁Δw₂` 会奖励"成对朝相反方向舍入"。ResNet18 首层 4-bit 的 100 组随机舍入中 **48 组优于最近舍入**，最好一组 63.06% 对最近舍入 52.29%（全上/全下只有 0.10%）。
- **逐层 QUBO**（§3.1，Eq.9–13）。量化权重取 `w_floor` 或 `w_ceil`（Eq.9–10），把找最优舍入写成最小化任务损失增量（Eq.11）；在块对角 Hessian 与忽略梯度项后得到 `argmin_Δw E[Δwᵀ H^(w) Δw]`（Eq.13）。障碍是 `H` 的内存/计算开销与 QUBO 的 NP-hard 性质。
- **从任务 Hessian 到局部 MSE**（§3.2，Eq.14–20）。`H^(w) = E[x xᵀ ⊗ ∇²_z L]`（Eq.16），把 `∇²_z L` 近似为对角阵（Eq.17）后，逐行最优化等价于最小化**该层输出的 MSE** `E[(ΔW_{k,:}·x)²]`（Eq.18–20）。论文指出该层级目标此前已被压缩/量化工作用过，AdaRound 的差别是**从任务损失推导出来**；附录 B 给卷积层的同构推导。Figure 1 显示 Eq.13 的代价与验证精度明显相关，Figure 3 显示优化后 `h(V)` 都收敛到 0/1。
- **连续松弛**（§3.3，Eq.21–24）。二值舍入松弛为 `argmin_V ‖Wx − W̃x‖²_F + λ·f_reg(V)`（Eq.21），`W̃ = s·clip(⌊W/s⌋ + h(V), n, p)`（Eq.22），`h(V) = clip(σ(V)(ζ−γ)+γ, 0, 1)`（Eq.23，ζ=1.1、γ=−0.1，整流 sigmoid 保证趋近 0/1 时仍有非零梯度），`f_reg(V) = Σ 1 − |2h(V_{i,j}) − 1|^β`（Eq.24），`β` 退火：前期让 h 自由移动降 MSE、后期逼到 {0,1}。论文把该过程归入 Hopfield 型二值优化。
- **非对称重构（最终目标）**（§3.3，Eq.25）。逐层顺序量化时用 `argmin_V ‖f_a(Wx) − f_a(W̃x̂)‖²_F + λ·f_reg(V)`，`x̂` 是前层已量化后的输入、`f_a` 是激活函数，用 SGD 优化，以补偿前层累积误差并考虑激活函数。
- **实验设置**（§5.1）：先吸收 BN；对称 4-bit 权重，逐层 scale `s` 由最小化 `‖W − W̄‖²_F` 事先确定；默认 **1024 张无标签 ImageNet 训练图**、Adam 默认超参、**10k 次迭代**、batch 32；ResNet18 单张 GTX 1080 Ti 约 **10 分钟**；ResNet18 FP32 基线 **69.68%**；报 5 个种子均值与标准差；除特别说明激活保持 FP32。
- **消融证据**（§5.1，Table 2–6）。Table 2（近似链，四档）：最近舍入 52.29/23.99、任务 Hessian 68.62/N/A、局部 MSE 69.39/65.83、连续松弛 69.58/66.56。Table 4：逐层 66.56 → 非对称 68.37 → 非对称 + ReLU 68.60。Table 5：最近舍入 23.99、STE 66.63、**AdaRound 68.60**。Table 6（网格选择，最近舍入/AdaRound）：Min-Max 0.23/61.96、W-MSE 23.99/68.60、X-MSE 42.89/68.62；两个 MSE 网格无明显胜负，论文统一沿用 W-MSE，并明确**不把步长与舍入联合优化**（"任何步长改变都会产生不同的 QUBO 问题"）。
- **校准数据鲁棒性**（§5.1、Figure 4）：只用 **256 张**图即可把模型优化到 FP32 精度 2% 以内；用同域但非训练集的无标签图（Pascal VOC、MS COCO）平均退化小于 **0.2%**。
- **与文献对比**（§5.2、Table 7–9；改用 2048 张图、20k 次迭代）。W4A32：AdaRound ResNet18 **68.71±0.06**、ResNet50 **75.23±0.04**、InceptionV3 **75.76±0.09**、MobileNetV2 **69.78±0.05**（†先 CLE 预处理），ResNet18/50 距 FP32 都在 1% 内，且优于部分对照（OMSE+opt 67.12、bias correction 67.4；后者记 W4/8，与 AdaRound 行的 W4/32 激活位宽不同，不能直接并列）；加 8-bit 激活后为 68.55/75.01/75.72/69.25。Table 8：最近舍入 23.99、bias correction 38.87、**AdaRound 68.60**。Table 9（DeeplabV3+ MobileNetV2，Pascal VOC，512 张校准图）：FP32 mIOU 72.94、DFQ 8/8 72.33、最近舍入 4/8 6.09、**AdaRound W4A32 70.89±0.33、W4A8 70.86±0.37**（约掉 2 点）。
- **附录 A 的 QUBO 求解器对比**（Table 10，只量化 ResNet18 首层）：最近舍入 52.29、cross-entropy method 68.62±0.17、qbsolv 41.98±3.04（比最近舍入还差）；论文把 qbsolv 的失败归因于其 API 无法"从最近舍入邻域初始化"，求解器选择不影响 AdaRound 本身。

## 不能推出的结论

- 实验对象是视觉 CNN、主结果是 W4A32 与 8-bit 激活；不能作为 LLM/Transformer 上 AdaRound 有效性的证据，也不能把"ResNet18 单模型 10 分钟"外推到十亿参数模型。
- 论文固定步长 `s`（W-MSE）且明确不联合优化步长与舍入；不能宣称它"同时优化了量化网格"，也不能把 Table 6 的 X-MSE 网格当最终方案。
- AdaRound 优化的是**层输出 MSE**（对任务损失做了多重近似：块对角 Hessian、忽略梯度项、`∇²_z L` 取对角），最终 0/1 解来自连续松弛 + 正则退火，无最优性保证；"SOTA"限于 2020 年 Table 7 所列对照方法。
- Table 5–9 的绝对精度绑定各自预训练权重、网络与网格设置，不能跨论文直接排名；只能看"相对对照方法的增益"。
- 逐层顺序量化意味着后续层用前层**已量化**输入、误差沿深度累积；非对称重构只是部分补偿，论文没有跨层联合优化的保证。
- 校准数据结论来自同域场景（ImageNet 子集与同域 VOC/COCO），不能假定换模态/严重分布偏移/极小样本时默认超参（10k 次迭代）仍合适；论文未报告 `λ` 与 `β` 调度取值。
- 4-bit 并非无损：分类约掉 1%、分割约 2 点 mIOU；8-bit 激活量化下精度还会再掉一点。
