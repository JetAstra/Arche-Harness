# ARCHE/MiniChem 变更历史

本文件只保留阶段节点。当前操作方法和验证状态以
[当前交付与复现说明](../current/DELIVERY_AND_REPRODUCTION.md) 为准；完整文件级差异由 Git 保存。

| 日期 | 阶段 | 主要结果 |
|---|---|---|
| 2026-08-11 | MiniChem 接入 | 将原 ARCHE 固定工具映射为可检索的原子 Actions；增加反应映射、端点构建、TS guess、光谱和数据交换能力，并保留有界的 `23_TSPipeline` 组合入口。 |
| 2026-08-12 | 可移植复现 | 合并 MiniChem/UMA/ASH 环境，复制未修改 ASH 0.95 到 `vendor/`，统一相对路径、缓存边界和环境声明；合入 ARCHE Memory 远端历史。 |
| 2026-08-13 | Case 1 验证 | 补强通用 Gaussian TS/Hessian/IRC、状态传递和解析门禁；增加 NCI/RDG、自由能选择性与 energetic-span 能力；完成 Case 1 直接计算与测试。提交 `1e860a6`。 |
| 2026-08-13 | Case 2/3 验证 | 增加通用 TD-DFT/TDA、relaxed coordinate scan 和 Gaussian NBO/Wiberg Actions；完成候选重放和真实 Gaussian 对照。提交 `ae8e94a`。 |
| 2026-08-13 | 文档收敛 | 用统一交付说明、三代工具箱对比和本简表替代相互重复且状态过期的阶段性报告。 |
| 2026-08-14 | 分支合并 | 合并专家角色、作者 Gaussian codegen/格式转换与 Case 1/2/3 通用计算能力；Catalog 更新为 63 个 Action、28 个 Backend 和 82 个 Action-Backend 合同，并按合并后的实现重写当前文档入口。 |
| 2026-08-15 | Memory、专家与恢复 | 专家后端默认启用，并在任务开始时冻结探测与回退选择；Hypothesis、Revision 和 Expert 使用固定阶段字段，未知持久化字段默认不可见；`waiting_execution` 只保存已经提交的 Gaussian native 作业，恢复时查询同一 operation。General fallback 调试临时注入四步 workflow 并到达可恢复等待状态，未重复提交 Gaussian 作业。 |
| 2026-08-18–21 | 旧流程最终验证 | 完成 TS Action 通用输入、native `foreach`、候选身份传播、Codex Planner 原生 MiniChem MCP、确定性 Execution 和长期 session 验证。轻量测试证明失败反馈、修订和再次执行可以完成；Case 1 自主运行仍因科学输入或工具能力缺口停止，没有形成论文级自主结论。 |
| 2026-08-23 | Investigation V2 切换 | 新建 Memory On 任务切换为 Retrieval、Hypothesis、独立 Investigation session、Task Review 和结果发布；旧 Planner/WorkflowExecutor 只保留给 `legacy_workflow_v1` 恢复。 |
| 2026-08-24–25 | 异步作业与恢复 | 统一 Investigation job、operation、submission、thread 和 checkpoint 身份；完成同 turn 轮询、completed receipt 重放、provider 迁移、scope 作业清单和恢复保护。 |
| 2026-08-25–26 | Case 1/3 真实运行 | Case 1 完成多组 TS 优化、Hessian 和振动验证并保留同任务恢复状态；Case 3 完成扫描证据、任务审查和终态更正后的成功关闭。 |
| 2026-08-26 | 架构与结果保护 | 将根目录架构说明设为代码修改前必读；禁止不完整实验形成成功结果，区分作业终态与科学结果，并增加严格的等待恢复和追加式终态更正合同。 |
| 2026-08-27 | 文档整理 | 删除由当前实现和 Git 历史替代的计划、冒烟和故障过程文档；研究证据移入 `docs/research/`；当前任务记录压缩为状态和阶段摘要。 |

历史报告从当前工作树删除，不表示审计信息丢失；它们仍可通过 `git log --all -- docs/` 和 `git show <commit>:<历史文档路径>` 查看。
