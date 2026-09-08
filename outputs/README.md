# 运行输出

`outputs/` 是 ARCHE 运行结果的统一入口。新 `research_task` 的任务目录位于
`research_tasks/<research_task_id>/`，并由该命名空间自己的 `index.json` 和
`latest.json` 索引。历史 `runs/<run-id>/` 仅供显式迁移或兼容定位器只读访问，
不是新任务的默认回退目录。
