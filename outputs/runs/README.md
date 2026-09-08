# 运行 Bundle

`runs/` 下保存历史兼容 Bundle，目录名是旧运行 ID。只有显式迁移或兼容定位器才能
读取这里的内容；新 `research_task` 必须创建在
`outputs/research_tasks/<research_task_id>/`。进入历史 Bundle 后，先查看
`status.json` 和 `summary.json`，再按需读取阶段摘要、计算产物和内部记录。
