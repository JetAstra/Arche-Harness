# ARCHE 文档索引

本目录只保留两类会随仓库发布的文档：当前操作与架构说明，以及能够独立核对的论文复现证据。一次性计划、冒烟记录、故障排查过程和已经被实现取代的旧流程说明由 Git 历史保存，不再作为当前文档发布。

## 从这里开始

- [仓库 README](../README.md)：安装、配置、启动、运行、恢复和结果查看。
- [领域上下文与统一命名](../CONTEXT.md)：`research_task` 顶层对象、模块、运行身份、输入输出和兼容边界。
- [代码架构说明](../ARCHE_CODE_ARCHITECTURE.md)：`investigation_v2` 与 `legacy_workflow_v1` 的边界、状态、身份、服务和测试要求。
- [当前交付与复现说明](current/DELIVERY_AND_REPRODUCTION.md)：当前能力、模型职责、恢复状态和证据范围。

## 当前状态

- [Harness 评测与变更摘要](current/CASE1_HARNESS_E2E_TASK_LOG_2026-08-21.md)：当前 Case 1 恢复状态、Case 3 已完成结果、关键架构变化和验证摘要。文件名为兼容仓库协作规则而保留。
- [假设确认与 Investigation 执行输入统一接口规格](current/HYPOTHESIS_EXECUTION_CONFIRMATION_SPEC_2026-08-30.md)：已实施的研究员确认边界，定义 1–5 条假设选择、假设修改、`guidance`、`materials`、持久化和恢复合同；真实服务验收尚未执行。
- [变更历史](history/CHANGELOG.md)：按阶段记录已经完成的能力变化；文件级差异以 Git 为准。

## 论文复现与输入审计

下列文档是带日期的研究证据，不是当前运行指南。文中旧路径、资源数量和测试总数只表示当次验证环境。

- [Case 1 公开输入的复现限制](research/CASE1_REPRODUCTION_INPUT_LIMITATIONS.md)：说明公开题面、结构输入、Gaussian 设置和自由能计算定义的缺口。
- [Case 1 分步验证](research/CASE1_STEPWISE_VALIDATION_2026-08-17.md)：区分发布数据重放、固定几何重算和从题面独立发现。
- [Case 2 分步验证](research/CASE2_STEPWISE_VALIDATION_2026-08-17.md)：记录发布数据、固定几何和实际计算的证据范围。
- [Case 1/2/3 MiniChem 验证](research/CASE123_MINICHEM_VALIDATION_2026-08-13.md)：保存当次通用 Action、真实 Gaussian 计算和已知限制。
- [Case 2 人工输入审计](research/CASE2_HUMAN_INPUT_AUDIT_2026-08-19.md)：区分公开题面、中途用户消息和无法核实的来源。
- [Case 3 人工输入审计](research/CASE3_HUMAN_INPUT_AUDIT_2026-08-19.md)：说明结构、电子态、扫描参数和最终描述符的来源边界。

## 历史文档的查询方式

删除旧文档不会删除审计记录。需要查看旧 Planner、WorkflowExecutor、运行报告或实施计划时，使用：

```bash
git log --all -- docs/
git show <commit>:<历史文档路径>
```

`legacy_workflow_v1` 只用于查看历史资料，不是当前生产或恢复入口。schema-v10 历史数据只能通过明确的只读迁移工具读取；兼容边界写在根目录架构说明和对应测试中，不再由旧计划文档定义。

## 维护规则

1. 同一操作只保留一个当前入口；索引只概括并链接，不重复详细合同。
2. 当前行为写入根 README、架构说明、源码旁参考或测试；一次性过程写入 Git 提交，不新增长期运行报告。
3. 论文复现文档必须标明证据日期、输入来源、完成范围和限制，不能把发布数据重放写成独立发现。
4. 移动或删除文档后，检查根 README、MiniChem README、TOOLBOX 和本目录中的链接。
