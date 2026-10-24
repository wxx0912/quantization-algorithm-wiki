# 来源迁移审计

本文件留档 `sources/` 从主题表格迁移到“一来源一文件”的范围，以及每条来源
`retrieved_at` 的依据。会话日志不在仓库内，没有本文件，“已读”在日志清理后不可考。

## 迁移范围

- 32 条来源：17 条论文（`papers/`）、15 条代码库（`repos/`）；迁移前登记在
  `core.md`、`reconstruction.md`、`transforms.md`、`mixed-architecture.md` 四张表。
- 20 个页面补充 `confidence` 与 `evidence_scope`；来源 ID 全部不变，页面 `sources` 字段未改动。
- 原四张表的主题前言与边界移入 `topics/`，表格不再作为维护位置。

## retrieved_at 口径

- 迁移轮统一写 `2026-10-23`（Asia/Shanghai）。抓取发生在 2026-10-22T16:01–16:52Z，
  即北京时间 2026-10-23 的 00:01–00:52。
- 2026-10-24（Asia/Shanghai）有第二轮重读：本轮为把论文回填为 LaTeX/会议版一手材料、
  重写代码库来源页而重新抓取并逐条核对，被重读的来源按实际重读日期写 `2026-10-24`，
  清单见下节。这是对既有日期的更新，按 SPEC §6 在此记明。
- 依据是 `~/.codex/sessions/` 中 `curl`、`wget`、`git clone` 命令的时间戳；
  下表取该轮抓取的首次出现，精确到分钟。
- 当日 2026-10-23T04:01–04:36Z（北京时间 12:01–12:36）还有一次复核读取，不改变日期。
- 迁移前的仓库内记录（原 `transforms.md` 的“阅读日期：2026-10-23”、原
  `mixed-architecture.md` 的“核实日期：2026-10-23”）与下表一致。

## 逐条依据

| 来源 ID | 首次抓取（UTC） |
| --- | --- |
| paper-gptq | 2026-10-22T16:41:13Z |
| repo-gptq | 2026-10-22T16:41:37Z |
| paper-smoothquant | 2026-10-22T16:42:02Z |
| repo-smoothquant | 2026-10-22T16:42:30Z |
| paper-quarot | 2026-10-22T16:41:52Z |
| repo-quarot | 2026-10-22T16:43:10Z |
| paper-omniquant | 2026-10-22T16:42:02Z |
| repo-omniquant | 2026-10-22T16:42:30Z |
| paper-awq | 2026-10-22T16:45:39Z |
| repo-awq | 2026-10-22T16:44:04Z |
| paper-adaround | 2026-10-22T16:46:00Z |
| repo-adaround | 2026-10-22T16:46:41Z |
| paper-autoround | 2026-10-22T16:46:20Z |
| repo-autoround | 2026-10-22T16:46:00Z |
| paper-brecq | 2026-10-22T16:46:00Z |
| repo-brecq | 2026-10-22T16:45:39Z |
| paper-spinquant | 2026-10-22T16:46:11Z |
| repo-spinquant | 2026-10-22T16:46:10Z |
| paper-flatquant | 2026-10-22T16:46:10Z |
| repo-flatquant | 2026-10-22T16:46:10Z |
| paper-svdquant | 2026-10-22T16:46:10Z |
| repo-svdquant | 2026-10-22T16:46:10Z |
| repo-nunchaku | 2026-10-22T16:52:48Z |
| paper-hawq2 | 2026-10-22T16:48:29Z |
| source-hawq2 | 2026-10-22T16:46:50Z |
| code-hawq | 2026-10-22T16:48:12Z |
| code-modelopt-autoquantize | 2026-10-22T16:46:51Z |
| paper-awq-architecture | 2026-10-22T16:45:39Z |
| paper-llada-architecture | 2026-10-22T16:49:40Z |
| paper-viditq | 2026-10-22T16:47:36Z |
| source-viditq | 2026-10-22T16:51:22Z |
| code-viditq | 2026-10-22T16:47:52Z |

## 2026-10-24 重读轮次

第二轮重读发生在北京时间 2026-10-24 凌晨（抓取落盘至 `/tmp/qa_src/`，mtime 2026-10-23
23:43–2026-10-24 00:35）。这些来源改用一手 LaTeX/会议版材料重新核对，`retrieved_at`
由 `2026-10-23` 改为 `2026-10-24`：

| 来源 ID | 重读材料 |
| --- | --- |
| paper-awq | arXiv e-print LaTeX（v5） |
| paper-awq-architecture | arXiv e-print LaTeX（v5） |
| paper-autoround | arXiv e-print LaTeX（v5） |
| paper-adaround | arXiv e-print LaTeX（v2） |
| paper-omniquant | arXiv e-print LaTeX（v3） |
| paper-brecq | arXiv e-print LaTeX（v2） |
| paper-spinquant | arXiv e-print LaTeX（v2） |
| paper-flatquant | arXiv e-print LaTeX（v1） |
| paper-llada-architecture | arXiv e-print LaTeX（v2） |
| paper-viditq | arXiv e-print LaTeX（v2） |
| source-viditq | arXiv e-print LaTeX（v2） |
| paper-hawq2 | HAWQ-V2 会议版 PDF + 预印本 LaTeX（v1） |
| source-hawq2 | HAWQ-V2 预印本 LaTeX（v1） |
| paper-svdquant | arXiv e-print LaTeX（v1） |

其余来源保持迁移轮的 `2026-10-23`（未做二次重读，或仅重读仍未改动）。

## 本库之外的更早抓取

以下记录发生在其他任务中，不作为本库 `retrieved_at` 的依据，仅留档避免混淆：

- repo-smoothquant：2026-09-15T02:51Z（llada25 训练侧阅读 main 分支 smooth.py）
- paper-spinquant：2026-09-24T15:37Z（读 abs 页）
- repo-adaround：2026-10-21T04:43Z / 2026-10-22T10:28Z（读 aimet develop 分支 README）
- repo-autoround：2026-10-21T04:24Z（读 main 分支 docs/algorithm_combinations.md）
- code-modelopt-autoquantize：2026-09-21T06:00Z、2026-10-21T08:33Z（读 main 分支与 git ls-remote）
- paper-flatquant：2026-08-14T03:22Z（arXiv API 元数据查询）
- paper-llada-architecture：2026-10-21T17:09Z、2026-10-22T10:29Z（读 abs 页）

## 已知边界

- 32 条来源都需要外部网络才能复核（抓取本身经 `socks5h://127.0.0.1:13659` 代理完成），
  因此全部标记 `requires_external_verification: true`：离线环境下这些来源的结论按缺口处理，
  不作为已确认事实使用。
- 本次迁移不做格式兼容：来源表格不再是维护位置，历史表格文件已改写为 `topics/`。
