# ARCHE 当前交付与复现说明

更新日期：2026-08-31

本文面向运行当前仓库或核对交付边界的研究人员。安装、命令参数和完整操作步骤以[仓库 README](../../README.md) 为准；状态、身份和服务边界以[代码架构说明](../../ARCHE_CODE_ARCHITECTURE.md)为准。本文只提供当前状态和研究证据入口，不重复源码能够直接给出的接口细节。

## 当前实现

| 项目 | 当前状态 |
|---|---|
| 活动架构 | 新建 Memory On 任务使用 `investigation_v2` |
| 主流程 | `retrieval` → `research-query-formulation` → `hypothesis-generation` → `hypothesis-selection` → `hypothesis-confirmation` → `hypothesis-investigation` → `task-review` → 发布结果或生成下一批假设 |
| MiniChem 服务 | 9011 提供只读 Catalog；9012 执行带 scope 认证的化学作业 |
| MiniChem 规模 | 65 个 Action、28 个 Backend、84 个 Action-Backend 合同 |
| Investigation | 每个入选假设使用独立 Codex thread 和 MiniChem scope |
| Task Review | 只读取当前批次已经验证的科学结果版本，不提交化学作业 |
| 事实与恢复 | Working Memory 保存科学事实；Investigation Context 与 Harness recovery store 保存私有恢复身份；`research_task` 任务目录提供面向研究人员的状态与阶段摘要 |
| 当前等待状态 | `waiting_investigation_job`，表示作业已提交但尚未形成 Investigation Result |
| 旧流程 | `legacy_workflow_v1` 已退出当前生产和恢复入口；历史数据只能走显式只读迁移 |

测试通过只证明相应合同和已覆盖行为可用。新的化学体系仍可能因输入不完整、计算不收敛、依赖缺失或资源不足而失败。

## 运行环境

直接运行仓库 Python 模块时使用标准解释器，并保留 ARCHE 与 MiniChem 的模块搜索路径：

```bash
PYTHONPATH=src:third_party/minichem_toolbox/src \
  .venv/bin/python -m chemistry_multiagent.controllers.chemistry_multiagent_controller --help
```

启动脚本会检查同一 `.venv`；`ARCHE_RUN_STORAGE_ROOT` 是三个入口共享的私有 runtime 根，
正式 `research_task` 任务目录只写入该根下的 `outputs/research_tasks/<research_task_id>/`，不会把服务工作区
或阶段 trace 目录当作输出根。运行产物默认写入被 Git 忽略的 `local_data/runtime/`，也可以通过
`ARCHE_RUN_STORAGE_ROOT` 指定仓库外的私有绝对目录。前置阶段、MiniChem
服务和 scope registry 分别使用 `ARCHE_STAGE_HARNESS_WORKSPACE_ROOT`、
`ARCHE_STAGE_HARNESS_TRACE_ROOT`、`MINICHEM_SERVICE_WORKSPACE`、
`MINICHEM_SCOPE_REGISTRY_ROOT` 和 `MINICHEM_INVESTIGATION_ENDPOINT`。

## 模型职责

- Retrieval 和 Hypothesis 使用任务创建时冻结的 provider 配置。
- Investigation 使用独立的 Codex Responses 配置；默认模型为 `gpt-5.6-sol`、默认
  `reasoning_effort=xhigh`，实际值以任务创建时冻结的 provider 配置为准。
- Task Review 使用独立的 Review provider 和配置模型，并在任务生命周期内复用同一 thread。
- 新建 V2 任务不会创建旧 Planner、`WorkflowExecutor`、Router、General opinion 或 Judge 阶段。兼容环境变量仍然存在，不表示这些阶段进入当前主流程。
- 恢复任务复用 `ResearchTaskCreated` 中保存的模型、Catalog、架构版本和输出位置，不重新选择配置。

Investigation 只读取当前科学问题、明确条件、当前假设和自己的 MiniChem 结果。Task Review 只读取当前批次的科学事实、观测、程序检查和判断。路径、密钥、原始日志、scope、job 身份和 Working Memory 数据库不进入科学审查消息。

## 状态与恢复

V2 的 native 恢复链由 Working Memory、Investigation Context、`HarnessRecoveryStore`、Codex session store 和 MiniChem 共同核对同一个 `research_task_id`、Module Invocation、Context、Session、thread、scope、operation、job、submission 和 receipt 身份。恢复时必须先查询原作业，再按原身份收集结果；身份缺失、冲突或状态不明确时停止，不能重新提交替代作业。

`waiting_execution` 只属于 `legacy_workflow_v1` 中已经实际提交的 Gaussian native 作业。它与 V2 的 `waiting_investigation_job` 是两套不同协议，不能互换。

当前 Case 1 运行与恢复状态见[Harness 评测与变更摘要](CASE1_HARNESS_E2E_TASK_LOG_2026-08-21.md)。恢复前先查看原 `research_task` 任务目录的 `status.json`、`summary.json` 和 `internal/state/`，并继续使用同一私有运行目录；展示文件不能替代上述 owner 的身份核对。

## 运行入口

从[仓库 README 的运行要求](../../README.md#运行要求)开始准备环境。当前端到端任务使用三个进程：

1. `scripts/services/start_catalog_bootstrap_mcp.sh` 启动 9011 Catalog 服务。
2. `scripts/services/start_investigation_minichem.sh` 启动 9012 Investigation 服务。
3. `scripts/run_end_to_end.sh` 完成 readiness 检查并启动 Controller。

三个进程必须使用同一个绝对 `ARCHE_RUN_STORAGE_ROOT` 私有 runtime 根；服务的
`MINICHEM_SERVICE_WORKSPACE`、scope registry 和阶段 workspace/trace 是该根下的独立目录，
不是 `research_task` 输出目录。仓库内的本地评测统一写入 `local_data/runtime/`；外部部署可以使用工作区之外的私有绝对目录。入口不会自动恢复旧任务；恢复必须显式选择 `resume-research-task` action 并提供原 `research_task_id`。历史 `outputs/runs/<run-id>/` 只能由明确的 schema-v10 只读迁移工具定位。

只运行 Retrieval 和 Hypothesis Top 5 时，使用 `scripts/run_to_hypothesis_top5.sh`。准备冻结论文库和恢复检索的方法见[仓库 README](../../README.md#6-单独检索论文)。

## research_task 结果位置

当前任务目录位于：

```text
<ARCHE_RUN_STORAGE_ROOT>/outputs/research_tasks/<research_task_id>/
├── status.json
├── summary.json
├── stages/
├── artifacts/
└── internal/
    ├── state/
    ├── audit/
    └── catalog/
```

先查看 `status.json` 和 `summary.json`，再按需读取阶段摘要和计算产物。当前任务目录的顶层身份字段为
`research_task_id`，阶段为 `literature_retrieval`、`hypothesis_preparation`、
`investigation_and_experiment` 和 `task_review_and_result`。`internal/state/` 保存恢复资料，
`internal/audit/` 保存模型调用审计，`internal/catalog/` 保存任务创建时冻结的工具目录。这些动态
内容不进入 Git；历史 `outputs/runs/<run-id>/` 只供显式 legacy 迁移/兼容定位器读取。

## 研究证据

| 文档 | 能证明什么 | 不能证明什么 |
|---|---|---|
| [Case 1 公开输入的复现限制](../research/CASE1_REPRODUCTION_INPUT_LIMITATIONS.md) | 公开题面、结构输入和计算定义存在明确缺口 | 当前 V2 已完成论文级 Case 1 复现 |
| [Case 1 分步验证](../research/CASE1_STEPWISE_VALIDATION_2026-08-17.md) | 2026-08-17 的发布数据、固定几何和独立发现证据边界 | 文中的旧运行状态仍然有效 |
| [Case 2 分步验证](../research/CASE2_STEPWISE_VALIDATION_2026-08-17.md) | Case 2 的计算映射、实际结果和未完成项 | 从自然语言题面独立发现完整机制 |
| [Case 1/2/3 MiniChem 验证](../research/CASE123_MINICHEM_VALIDATION_2026-08-13.md) | 当次通用 Action 和真实 Gaussian 验证结果 | 当前 Action 数量、路径或资源配置 |
| [Case 2 人工输入审计](../research/CASE2_HUMAN_INPUT_AUDIT_2026-08-19.md) | 公开材料中可见的中途方向提示及证据冲突 | 原始对话未公开时的确定来源判定 |
| [Case 3 人工输入审计](../research/CASE3_HUMAN_INPUT_AUDIT_2026-08-19.md) | 后续结构和计算条件超出原始题面的范围 | 所有条件均由代理自动生成或由人提供 |

发布数据重放、固定发布几何重算和从题面独立发现必须分开表述。旧 Planner 冒烟、手工补全参数的测试和注入参考数值的运行只能证明工程路径，不能作为论文科学结论。

## Git 与私有数据

Git 只管理源码、配置模板、环境声明、测试和对外文档。不要提交：

- `config.local.env`、API key、访问令牌或 Codex 登录状态；
- `local_data/`、`research_task` 任务目录、SQLite 数据库、模型调用记录和 Gaussian 作业目录；
- 模型、软件缓存、Gaussian scratch、checkpoint 和授权软件；
- `outputs/` 下除固定说明文件以外的动态内容。

旧计划、运行过程和故障报告由 Git 历史保存。它们不应重新复制到当前文档，也不能覆盖当前源码和测试所定义的行为。
