# 假设确认与 Investigation 执行输入统一接口规格

- 状态：已实施；确定性回归已通过，真实服务验收尚未执行
- 日期：2026-08-30
- 适用架构：`investigation_v2`
- 目标读者：ARCHE 研究前端、Controller、Working Memory、Investigation Harness 和测试维护者

本规格定义 Hypothesis 阶段与 Investigation and Experiment 阶段之间已经接入生产代码的研究员确认接口。源码和测试定义精确行为，《ARCHE Harness Agent 架构约束与安全修改指南》定义长期架构边界。本规格中的真实服务验收仍需在独立新任务中执行；确定性回归不代表真实模型、MiniChem 或化学作业已经通过验收。

## Problem Statement

本接口实施前，ARCHE 没有一个同时适用于标准 Top 5 和作者假设直入的研究员确认边界。标准完整任务在 Selection 后直接进入 Investigation；`top5_selection` 会在保存五条假设后关闭任务，不能在同一任务中继续执行；作者假设直入则从 Hypothesis Import 直接进入 Investigation。研究员当时无法在统一位置完成以下操作：

- 查看本批次准备执行的假设；
- 从候选中选择最终执行的 1–5 条假设；
- 在执行前修改假设内容；
- 为单个假设补充 `guidance`；
- 为单个或多个假设补充结构文件、图片、矩阵元素文件或其他 `materials`；
- 确认后恢复同一任务，而不是另建任务或复制运行状态。

Case 3 的五文件实验已经证明，模型需要先在自己的临时 workspace 中获得文件副本，才能读取外部研究材料。但是当前实现把这项能力绑定到 Case 3、作者模式和固定文件结构，不能作为通用接口。模型也不应通过系统提示自动获知 `case1`、`case3`、必须使用某个 Pipeline 或其他案例特异信息。

## Solution

在每轮 Hypothesis Selection 或 Hypothesis Import 后增加统一的 Hypothesis Confirmation 模块。该模块先保持任务为 `Open`，并以 `reason_code=awaiting_researcher_confirmation` 返回研究员可读视图。在确认完成前，不创建 Investigation Context、Session、Codex thread、MiniChem scope，也不调用 Investigation 模型或化学工具。

研究员提交一次确认请求，明确处理本批次全部候选：每条候选必须标记为 `execute` 或 `cancel`，最终必须有 1–5 条 `execute`。执行项携带最终假设快照，并可附带 `guidance` 和 `materials`。系统先校验并持久化上传材料，再在同一 Working Memory 事务中提交确认结果和逐假设执行输入。确认成功后，Controller 从同一任务继续 Investigation。

标准流程变为：

`ResearchTaskCreated → Retrieval → Query → Hypothesis → Selection → Hypothesis Confirmation → Investigation and Experiment → Task Review`

作者假设直入变为：

`ResearchTaskCreated → Hypothesis Import → Hypothesis Confirmation → Investigation and Experiment → Task Review`

标准 Review 生成下一批假设时，新批次必须再次经过 Hypothesis Confirmation。作者假设直入仍不允许生成下一批。

### 研究员可见确认视图

任务进入等待状态时，公开结果和 `research_task` 任务目录的投影视图应提供以下内容：

| 字段 | 含义 |
|---|---|
| `research_task_id` | 当前任务的稳定身份 |
| `status` | 固定为 `open`，表示 Working Memory Task 尚未闭合 |
| `reason_code` | 固定为 `awaiting_researcher_confirmation` |
| `expected_research_task_sequence` | 本次确认使用的乐观并发序号，防止提交过期视图 |
| `hypothesis_batch_ref` | 当前待确认批次的稳定引用 |
| `candidates` | 按稳定排名展示的 1–5 条作者假设，或恰好 5 条标准 Selection 假设 |

每个候选向研究员暴露不透明 `hypothesis_candidate_record_id`、排名和完整科学内容。内部 Module Invocation、provider、thread、scope、job 和路径不进入确认视图，也不进入后续模型输入。

### 确认请求

确认接口是一个应用级操作：`submit_hypothesis_confirmation_and_continue`。调用方提交结构化字段和文件流，不提交一段需要后端再次猜测语义的自由文本 JSON 字符串。

| 字段 | 必填 | 约束 |
|---|---:|---|
| `research_task_id` | 是 | 必须指向一个正在等待当前批次确认的 `investigation_v2` ResearchTask |
| `confirmation_id` | 是 | 调用方生成的幂等身份；相同身份只允许精确重放 |
| `expected_research_task_sequence` | 是 | 必须等于确认前读取的最新 Working Memory 序号 |
| `decisions` | 是 | 必须恰好覆盖当前批次的全部候选，不能遗漏或重复 |
| `files` | 否 | 本次请求上传的文件流集合；只由 `materials` 中的上传引用消费 |

每条 `decisions` 项包含：

| 字段 | `execute` | `cancel` | 说明 |
|---|---:|---:|---|
| `hypothesis_candidate_record_id` | 必填 | 必填 | 必须属于当前待确认批次 |
| `decision` | 必填 | 必填 | 只允许 `execute` 或 `cancel` |
| `hypothesis` | 必填 | 不允许 | 最终执行的完整假设快照；未修改时由前端原样回传 |
| `guidance` | 可选 | 不允许 | 研究员提供给该假设 Investigation session 的科学补充说明 |
| `materials` | 可选 | 不允许 | 该假设可以读取的上传材料及描述；同一上传材料可以分配给多个执行项 |

每条 `materials` 项引用一个本次请求中的文件流，并可提供面向模型的简短描述。文件名只保留安全 basename；调用方本地绝对路径、临时上传路径和前端存储路径不得写入 Working Memory、任务目录科学字段或模型输入。

本地命令适配器可以接收研究员提供的文件路径，但必须先打开并转换为同一文件流合同。前端可以使用 multipart 上传实现同一合同。两种入口最终都调用同一个确认服务，不分别实现 Case 逻辑。

### Investigation 模型输入

模型继续使用所有 Investigation 任务共享的 base system prompt。确认内容只进入当前 session 的 user input，不追加 Case 特异 system prompt，也不改变工具选择规则。

送入模型的结构化执行输入只包含：

- 原始 `scientific_question`；
- 该 session 的最终 `hypothesis`；
- 可选的 `guidance`；
- 可选材料的安全相对路径、文件名、媒体类型和研究员描述。

上述字段经过合同校验后序列化为一次结构化 JSON payload。内部保存时使用字段合同，模型调用时由 Prompt builder 统一序列化；不得把整个确认请求作为未经解析的字符串透传，也不得把内部候选引用、取消决定、排名、Case profile 或材料外部路径交给模型。

材料正文不嵌入 prompt。系统在首次启动对应 Session 前，把材料从任务级持久存储复制到该 Session workspace 的受控相对目录，并设置为不可执行文件。模型通过 payload 中的相对路径读取材料。

## User Stories

1. 作为研究员，我希望标准任务生成 Top 5 后暂停，以便在开始计算前检查候选假设。
2. 作为研究员，我希望作者假设直入也停在同一个确认界面，以便使用同一套实验操作。
3. 作为研究员，我希望不修改任何内容时可以直接确认，以便快速继续任务。
4. 作为研究员，我希望选择执行 1–5 个假设，以便取消不值得投入计算资源的候选。
5. 作为研究员，我希望界面明确显示哪些假设被执行、哪些被取消，以免因遗漏产生歧义。
6. 作为研究员，我希望修改某条假设后再执行，以便修正生成结果中的科学表达或边界。
7. 作为研究员，我希望为每条假设分别填写 `guidance`，以便补充该假设独有的实验背景。
8. 作为研究员，我希望为每条假设上传不同材料，以便不同 Investigation session 只看到与自己相关的输入。
9. 作为研究员，我希望同一个材料可以分配给多条假设，以免重复上传公共结构或说明文件。
10. 作为研究员，我希望上传坐标、图片、JSON、文本或其他允许的科学文件，以便接口不绑定某个 Case。
11. 作为研究员，我希望确认成功后继续原任务，以便保留原问题、假设来源和恢复身份。
12. 作为研究员，我希望刷新页面后仍能看到相同待确认批次，以便前端会话丢失不影响科学任务。
13. 作为研究员，我希望重复提交同一确认请求不会重复创建 Session 或计算作业，以便安全处理网络重试。
14. 作为研究员，我希望过期页面的确认请求被明确拒绝，以免覆盖其他已提交决定。
15. 作为研究员，我希望材料上传失败时任务继续保持待确认，以便修正文件后重新提交。
16. 作为研究员，我希望系统在文件损坏或摘要漂移时停止执行，以免模型读取未经确认的内容。
17. 作为研究员，我希望模型只看到文件在自己 workspace 中的相对路径，以免泄漏服务器目录结构。
18. 作为研究员，我希望 Case 3 的四个结构和二面角映射可以作为普通 `materials` 提供，以便不再依赖专用输入开关。
19. 作为研究员，我希望 TS Pipeline 的图片或固定反应物资料可以作为普通 `materials` 提供，以便模型获得作者明确提供的输入，而不是获得隐藏提示。
20. 作为研究员，我希望 base system prompt 不包含 `case1`、`case3` 或强制工具选择，以便真实评估模型和 Pipeline 能力。
21. 作为前端开发者，我希望待确认视图具有稳定字段和并发序号，以便可靠渲染和提交表单。
22. 作为前端开发者，我希望 JSON 决定和文件上传使用同一个确认请求，以便避免材料已上传但选择尚未确认的模糊状态。
23. 作为 CLI 维护者，我希望本地路径只在入口层转换为文件流，以便 CLI 与前端共用后端合同。
24. 作为 Controller 维护者，我希望确认阶段在模块顺序中直接可见，以便恢复路由不依赖前端状态或任务目录展示。
25. 作为 Working Memory 维护者，我希望原始候选、研究员决定和最终执行输入都是不可变记录，以便重建完整科学来源。
26. 作为 Harness 维护者，我希望每个 Session 只接收一条最终执行输入及其材料，以便保持 workspace、scope 和 thread 隔离。
27. 作为 Review 维护者，我希望结果数量由已确认执行批次决定，以便标准任务选择 1–4 条时仍能进行完整 Review。
28. 作为恢复逻辑维护者，我希望未确认任务不会创建外部副作用，以便恢复只需重放同一待确认视图。
29. 作为恢复逻辑维护者，我希望确认后的材料摘要和 Session 输入可重新核对，以便进程重启不需要读取原外部文件。
30. 作为审计人员，我希望 `research_task` 任务目录展示原始候选、执行数量、取消数量和材料摘要，以便检查研究员实际批准了什么。
31. 作为审计人员，我希望取消的假设仍保留在决定记录中，以便不能把“未执行”误写成“模型没有生成”。
32. 作为现有任务维护者，我希望历史已关闭的 `top5_selection` 和旧 Case 3 任务继续按原合同只读解释，以便新功能不重解释旧事实。
33. 作为 Case 1 运行维护者，我希望正在运行的任务不被迁移、恢复或修改，以便本规格开发不影响活动计算。

## Implementation Decisions

1. 新增公开模块类型 Hypothesis Confirmation。它是研究员输入模块，不调用科学模型或 MiniChem。
2. Selection 和 Hypothesis Import 仍分别保存原始 `SelectedHypothesisRecord`；确认操作不修改、替换或删除这些记录。
3. Confirmation Module Invocation 在候选准备完成后创建并保持 `Open`。Controller 返回 `status=open` 和 `reason_code=awaiting_researcher_confirmation`；普通 `resume` 在没有确认事实时精确返回同一状态，不调用下游 adapter。
4. 确认服务是提交研究员决定的唯一写入口。它必须核对当前 ResearchTask、Open Confirmation Module Invocation、候选全集、`expected_research_task_sequence` 和 `confirmation_id`，然后提交一个密封的 `PreparedModuleInvocationOutput`。
5. Confirmation Module Invocation 原子输出一条 `HypothesisExecutionDecisionRecord`、每个执行项一条 `HypothesisExecutionInputRecord`，以及本次使用的 `ResearchMaterialRecord`。输出组必须完整，否则 Working Memory 零写入。
6. `HypothesisExecutionDecisionRecord` 保存源批次、全部候选的 `execute`/`cancel` 决定、最终执行顺序和确认身份。取消项没有执行输入记录，但必须出现在决定记录中。
7. `HypothesisExecutionInputRecord` 保存原 `SelectedHypothesisRecord` 引用、连续执行排名、最终假设快照、可选 `guidance` 和有序材料记录引用。编辑后的假设不回写原候选。
8. `ResearchMaterialRecord` 保存安全文件名、媒体类型、字节数、SHA-256、可选研究员描述和不透明内容引用。它不保存外部绝对路径、临时上传路径或文件正文。
9. Working Memory 仍是研究员决定和科学输入身份的唯一事实来源。材料字节由任务级、内容寻址的持久输入存储拥有；任务目录只投影可读清单，不能反向创建确认事实。
10. 文件接收先写临时普通文件并计算摘要，通过文件类型、文件数量、单文件大小、任务累计大小和安全名称策略后，再原子发布到任务级内容存储。协议使用冻结策略中的限制，不把具体配额硬编码进 prompt 或 Scientific Record。
11. 材料内容相同可以在同一任务内复用底层字节，但每条 `ResearchMaterialRecord` 的科学描述和分配关系仍按确认请求保存。未被成功确认引用的暂存文件不构成任务事实，可由独立清理过程回收。
12. 确认提交使用 `confirmation_id` 和 `expected_research_task_sequence` 双重门禁。相同身份、相同字段和相同文件摘要即使在确认后也可以重放，并返回当前任务状态；相同身份正文漂移、过期序号、候选集合变化，或使用另一个确认身份修改已经确认的批次时返回状态冲突。
13. 确认成功后不可修改。需要变更的研究员必须在提交前完成；本规格不提供已启动 Investigation 的就地换假设或换材料能力。
14. Investigation Context 绑定当前 `HypothesisExecutionDecisionRecord`，而不是原始完整候选批次。每个 Session 由对应 `HypothesisExecutionInputRecord` 派生，取消项不能创建 Session、scope、thread 或 job。
15. Investigation 首次输入闭包必须从执行输入追溯到原 `SelectedHypothesisRecord` 和 Hypothesis Batch。返修继续复用同一 Session，并保留首次确认的假设、`guidance` 和材料；Task Review 的 repair instructions 是另一类已有科学记录，不能覆盖研究员输入。
16. Review 接受的结果数量由当前 `HypothesisExecutionDecisionRecord` 冻结为 1–5，而不再根据 `entry_strategy` 推断标准任务一定有五个结果。Review 仍必须覆盖该确认批次全部当前 Result heads。
17. 标准 Selection 仍必须恰好产生五条候选。允许执行 1–5 条发生在 Confirmation，而不是放宽 Selection 合同。
18. 新任务不再使用“Top 5 完成即关闭”的独立行为。Top 5 公开入口启动标准 Retrieval 流程并停在统一确认边界；确认后可以继续同一任务。历史已经保存的 `top5_selection` Task 继续按原冻结事实只读解释，不重新打开。
19. Top 5 生成和待确认阶段不访问 9011、9012，也不创建 MiniChem scope。`ResearchTaskCreated` 冻结预期端点、profile、资源策略和依赖约束；确认成功后、首次 Investigation 副作用前解析并持久化唯一 Catalog 快照及执行依赖身份。快照一旦形成，后续恢复不得接受漂移。依赖暂时不可用时保留已确认任务为 `Open`，不撤销研究员决定。
20. 新的作者假设任务通过 Hypothesis Import 导入 1–5 条作者假设，再进入统一确认。作者 `main hypothesis` 是候选科学输入，不写入 system prompt，也不由 Case 开关隐式注入 Investigation。
21. 新任务中的 Case 3 五文件、TS Pipeline 图片和其他作者材料都通过通用 `materials` 接口进入。既有任务仍按 `ResearchTaskCreated` 中已经冻结的旧文件合同解释，不在恢复时迁移。
22. base system prompt 对全部 Investigation session 保持一致。`guidance` 和材料清单属于结构化 user input；它们不能提高工具权限、改变 MiniChem scope、覆盖资源策略或绕过现有安全合同。
23. Prompt builder 只读取当前 Session 的 `HypothesisExecutionInputRecord` 和材料相对路径。它不得暴露 Case profile、内部记录引用、取消项、排名、外部路径、上传元数据、凭据或其他 Session 材料。
24. 每个 Session 的副作用顺序固定为：持久化 Session 身份、创建或核对 workspace、物化并验证材料、准备 MiniChem scope、启动 provider。缺失材料返回 `RecoveryEvidenceMissing` 并保持 Module Invocation `Open`；内容摘要漂移返回 `VerifiedIntegrityConflict`，且不能创建 scope、调用模型或提交工具作业。
25. `research_task` 任务目录保持现有四阶段布局。Hypothesis 展示阶段增加待确认、已确认、执行数量、取消数量和材料摘要；根状态使用 `Open` 生命周期和明确的 `reason_code`，不新增含糊的 `running` 状态。
26. Controller lease 规则继续适用。创建 Confirmation Module Invocation、提交确认记录和开始 Investigation 前都必须核对当前 fencing token；失去 lease 不把任务写成失败。
27. 前端不是恢复事实来源。页面关闭、上传连接断开或前端数据库丢失后，待确认视图必须完全从 Working Memory 和任务级材料存储重建。
28. `materials` 和 `guidance` 都是可选字段。不需要补充时，研究员仍必须明确提交全部候选的执行或取消决定。
29. 该变更属于确认后的架构修改。实施时必须同步源码、测试、架构说明、公开入口说明和任务变更摘要，不能只修改 prompt 或前端展示。

## Testing Decisions

### 最高层验收 seam

主要行为测试通过 `ResearchTaskProductionAssembly` 的新任务、确认和恢复公开入口完成，使用真实 V2 Working Memory、真实 Module Controller、真实 Confirmation 合同和任务级材料存储，其他科学模块使用确定性替代实现。测试只观察公开返回、Scientific Record、任务目录投影和外部 adapter 调用次数，不断言私有辅助函数或临时目录实现。

`ResearchTaskApplication` 和 CLI 只增加薄集成测试，证明它们调用同一应用 seam。前端实现以后也必须调用该 seam，不另建确认状态机。

### 必须覆盖的行为

1. 标准任务在 Selection 后返回待研究员确认，Investigation 和 Review 未调用。
2. 作者假设直入在 Import 后返回相同等待状态，Retrieval、Query、Hypothesis 和 Selection 未调用。
3. 普通恢复在未确认时重放同一 Open Confirmation Module Invocation，不创建第二个 Module Invocation。
4. Top 5 生成和待确认阶段不访问 9011、9012；确认后首次进入 Investigation 才冻结执行 Catalog，恢复复用同一快照。
5. 五条候选全部原样执行时，确认后创建五条执行输入并继续 Investigation。
6. 选择 1、2、3、4、5 条时分别创建相同数量的 Session 和 Result，Review 使用确认数量。
7. 取消全部候选、遗漏候选、重复候选、未知候选或取消项携带额外字段时零写入并返回参数错误。
8. 编辑假设时原 `SelectedHypothesisRecord` 不变，执行输入保存最终快照和来源。
9. `guidance` 只进入所属 Session 的 user input，不进入 system prompt、其他 Session 或 Review 输入。
10. 一个材料分配给多个假设时字节可以复用，但每个所属 Session 都获得自己的安全相对文件。
11. 不同假设的材料不能跨 Session 读取或出现在对方 prompt 中。
12. 本地文件路径不会进入 Working Memory、任务目录科学字段或模型 payload。
13. 文件名穿越、符号链接、特殊文件、超限内容、摘要漂移和媒体策略失败均在模型调用前拒绝。
14. 确认请求在材料发布后、Working Memory 提交前发生进程中断时，重试不会形成部分科学记录或重复 Session。
15. Working Memory 已提交但调用方丢失响应时，相同 `confirmation_id` 精确重放，不重复启动 Investigation。
16. 相同 `confirmation_id` 的字段或文件正文漂移返回幂等冲突。
17. 过期 `expected_research_task_sequence` 返回状态冲突，保留最新确认状态。
18. 材料缺失时恢复返回 `RecoveryEvidenceMissing`；摘要漂移时返回 `VerifiedIntegrityConflict`。
19. Review 生成下一批后再次进入确认，而不是直接开始下一轮 Investigation。
20. 确认后的 repair 复用原 Session、thread、材料和研究员 `guidance`。
21. Prompt 泄漏测试确认没有自动出现 `case1`、`case3`、Case profile、强制 TS Pipeline、外部路径、内部身份或其他 Session 内容。
22. `research_task` 任务目录的 Hypothesis 阶段正确投影待确认与已确认状态，并携带最新 `source_task_sequence`。
23. 历史已关闭 `top5_selection`、旧作者导入和旧 Case 3 文件任务仍按原冻结事实读取，不被新接口重新解释。
24. 活动 Case 1 runtime 不被测试读取、写入、恢复或替换；所有测试使用独立临时 runtime。

### 验证层次

- 合同测试验证新 Scientific Record、确认请求、文件清单和成组原子输出。
- Controller 测试验证阶段顺序、Open/Closed 生命周期、lease 和幂等恢复。
- 跨层恢复测试验证确认、材料物化、Context、Session、Harness 和 Review 的连续身份。
- Prompt 测试验证 base prompt 统一、结构化 user input 和字段隔离。
- 公开入口测试验证标准入口、作者直入、Top 5 启动入口和 CLI 的统一路由。
- 真实服务验收在确定性回归完成后单独执行；未经研究员明确要求，不提交集群任务，也不把替代实现通过写成真实模型或 MiniChem 通过。

## Out of Scope

- 本规格不实现研究前端页面、用户认证、权限管理或多人审批流程。
- 本规格不允许确认后、Investigation 已启动时修改假设、替换材料或追加研究员 `guidance`。
- 本规格不增加 Task Review 中途向研究员请求新材料的交互轮次。
- 本规格不改变 MiniChem Catalog、Action、Backend、资源策略或 scope 权限。
- 本规格不规定模型必须选择某个工具、TS Pipeline、计算方法、基组、溶剂、扫描范围或步长。
- 本规格不把 Case 1、Case 2 或 Case 3 的特定结构、结论、能量、坐标或作者结果写入通用 prompt。
- 本规格不自动迁移 `legacy_workflow_v1`、schema-v10 历史数据或已经关闭的 V2 `research_task`。
- 本规格不恢复、重新提交或修改当前正在运行的 Case 1 任务。
- 本规格不把上传材料自动认定为计算证据；材料是研究输入，只有 Investigation 产生并经 Evidence 验证的结果才能成为 Review 证据。
- 本规格不在协议中固定上传文件的生产配额；实现必须通过可冻结策略提供明确限制和错误结果。

## Further Notes

Case 3 的五文件流程曾作为“外部文件先冻结、再复制到 Session workspace、模型只看相对路径”的行为参考；通用材料模块没有复制其固定文件名、Case profile、特殊 prompt 或历史创建事件中的 `structure_input` 结构。

作者假设直入和标准 Top 5 共享的是确认及执行输入合同，不是上游来源合同。标准 Selection 仍产生恰好五条模型候选；作者 Import 仍导入一至五条已经审查的作者假设。两者在 Confirmation 后统一形成一至五条 `HypothesisExecutionInputRecord`。

实现以“标准 Selection → 等待 → 无材料确认一条 → Investigation 收到执行输入”作为首个纵向 tracer bullet，再扩展作者直入、多材料、恢复和兼容性。后续修改不得通过临时 Case 字段、前端状态文件或任务目录反推来绕过 Working Memory 合同。
