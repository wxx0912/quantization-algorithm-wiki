# 来源索引

来源表登记稳定 ID、资料类型、固定版本和实际已读定位。正文承载具体论断与对应公式/表格/函数；来源存在不意味着整篇所有结论都已验证。

| 来源组 | 主要内容 |
| --- | --- |
| [基础与样章](core.md) | GPTQ、SmoothQuant、QuaRot、OmniQuant、基线 |
| [重构与舍入](reconstruction.md) | AWQ、AdaRound、AutoRound、BRECQ |
| [变换与低秩](transforms.md) | SpinQuant、FlatQuant、SVDQuant、Nunchaku |
| [混精与架构](mixed-architecture.md) | HAWQ、ModelOpt、LLaDA、ViDiT-Q、MoE/VLM 证据 |

优先论文 LaTeX 源码，其次 HTML；PDF 用于补充无源码内容或图表。登记实际阅读形式，不把之前的 HTML 阅读事后改成 LaTeX 阅读。论文版本与源码 commit 独立管理，浮动 README 链接只作入口，不能替代固定实现证据。

本库不镜像论文全文或上游代码。读取单页及其来源：`python3 scripts/get_page.py algorithm-gptq --follow-sources`。来源表是机器可读输入，新增行遵循 [维护约定](../CONTRIBUTING.md)。
