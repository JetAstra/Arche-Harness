# 脚本目录

`scripts/` 根目录只保存与具体 Case 和冻结输入无关的入口：

- `bootstrap_reproduction.sh`：准备仓库标准 Python 环境。
- `run_end_to_end.sh`：按六个正式 action 创建、读取确认、提交确认后继续，或恢复完整
  `investigation_v2` `research_task`。默认 action 是
  `create-standard-research-task`；不会自动恢复历史任务。
- `run_to_hypothesis_top5.sh`：执行 Retrieval、Query、Hypothesis Generation 和 Selection，
  创建 `research_task` 后停在 `hypothesis-confirmation`。
- `run_paper_search_only.py`：只检索并下载论文。
- `monitor_harness_run.sh`：只读查看一个已知 `research_task` 的运行状态；历史
  schema-v10 只能显式使用 `--legacy-run-id`。

其他脚本按用途放置：

- `services/`：启动 9011 Catalog 和 9012 Investigation MiniChem；其中
  `start_minichem.sh` 是两个入口共用的底层启动器。
- `quality/`：代码质量测量工具。
- `tasks/`：依赖特定 Case、作者输入或复现数据的脚本。

作者假设任务 wrapper 当前依赖仓库内已冻结的 Case 定义，因此位于 `tasks/`；通用作者假设
JSON 入口使用根脚本的 `create-author-hypothesis-task` action。

当前公共 action 为 `create-standard-research-task`、`create-author-hypothesis-task`、
`create-top5-confirmation-task`、`get-hypothesis-confirmation`、
`submit-hypothesis-confirmation-and-continue` 和 `resume-research-task`。旧的
`--run-mode`、`--resume-task` 和 `--confirm-task` 不再是脚本或 CLI alias。
