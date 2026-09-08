# ARCHE Harness 领域上下文

本文定义当前 `investigation_v2` 源码、测试、脚本和文档共同使用的领域名称。精确行为以公开入口和测试为准，长期架构边界见 [ARCHE_CODE_ARCHITECTURE.md](ARCHE_CODE_ARCHITECTURE.md)。

## 顶层对象

`research_task` 是唯一顶层聚合，表示一个科学问题及其冻结输入、策略、依赖、科学记录和生命周期。公开身份统一为 `research_task_id`。当前任务目录固定位于 `outputs/research_tasks/<research_task_id>/`；`RunBundle` 只是在源码中管理该目录的实现类名，不表示另一个 `run` 聚合或 `run_id` 身份。

以下名称不用于新任务：

- `task`、`inquiry` 或 `run` 作为另一种顶层对象；
- `task_id`、`inquiry_id` 或 `run_id` 作为当前任务身份；
- `outputs/runs/<run-id>/` 作为当前任务目录（历史 schema-v10 只读迁移边界除外）。

schema-v10 只读迁移器可以读取这些历史名称，但必须在边界内立即转换，不得把旧字段写回当前对象。

## 工作流与模块

`ResearchTaskApplication` 是应用入口，`ResearchWorkflowController` 按顺序推进公开模块。一次模块调用称为 `Module Invocation`，身份为 `module_invocation_id`，输入摘要为 `module_input_digest`。

标准流程使用以下模块名：

1. `retrieval`
2. `research-query-formulation`
3. `hypothesis-generation`
4. `hypothesis-selection`
5. `hypothesis-confirmation`
6. `hypothesis-investigation`
7. `task-review`

作者假设流程以 `hypothesis-import` 取代前四个标准模块，然后进入相同的确认、调查和审查阶段。入口策略只使用 `standard_hypothesis_generation` 和 `author_hypothesis_import`。

## Investigation 身份

`investigation_context_id` 表示一轮已确认执行输入及其返修上下文。每条被确认的假设拥有独立 `investigation_session_id`；Session 绑定自己的 workspace、MiniChem scope、模型 thread 和同一条 `HypothesisExecutionInputRecord`。

提交作业后，恢复必须复用原 `research_task_id`、Context、Session、thread、scope、operation、submission 和 `job_id`。`waiting_investigation_job` 是当前唯一等待协议；排队或运行中的同一作业只能持续查询，终态后收集，不能因等待时间长而取消、中断、放弃或重新提交。

## 输入、事实与展示

- `ResearchTaskWorkingMemory` 保存科学事实、模块生命周期和恢复所需的科学身份。
- `HarnessRecoveryStore` 保存模型 Session、turn、作业回执等私有运行事实。
- `research_task` 任务目录位于 `outputs/research_tasks/<research_task_id>/`，只提供研究员可读状态、阶段摘要和产物索引，不是恢复事实来源。
- 研究材料先进入任务级内容寻址存储，再物化到 Session 的 `inputs/materials/<sha256>/<filename>`；外部绝对路径不进入 Working Memory、任务目录或模型输入。

## 兼容边界

`legacy_workflow_v1` 不是生产或恢复入口。schema-v10 历史数据只能由明确的只读迁移工具读取；旧字段、旧阶段、旧等待记录和旧目录只能存在于该读取边界、迁移测试或历史文档中。当前入口、启动脚本和监控脚本不得静默回退到旧布局。
