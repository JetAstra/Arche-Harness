# Case2 中途人工输入审计

本文是论文材料和当时仓库状态的研究审计，不是当前 ARCHE 运行说明。

审计日期：2026-08-19
审计范围：论文正文、Supplementary Information、`supplementary/case2.json`、`scripts/tasks/case23_reproduce.py` 以及仓库中已有的 Case2 复现记录。
证据边界：只使用本地文件，不联网；本报告不修改 Prompt、模型输入或实验 Run Bundle。

## 结论先行

对“中途是否有人提供了初始 `user input` 之外的信息”，需要分两种含义回答：

1. **若“条件”指新的实验条件或新的数值约束**（例如新的波长、溶剂比例、当量、温度、完整结构、坐标或电荷自旋）：在公开材料中**没有找到一条可核实的中途人工消息提供这些条件**。论文公开的中途 `User:` 消息只增加了一个机制搜索方向，并没有给出新的物理实验条件。
2. **若“人为参与”包括任何中途用户消息**：图 4 明确显示了一条初始任务之外的消息：`User: Continue investigating interactions within CsPPh2 pathway`。它直接把搜索焦点指向 `CsPPh2` 路径，而这正是论文最后保留的 G 机制；按严格的 task-only 复现标准，这应被记录为**方向性人类干预/近似 oracle steering**，不能声称整条 Case2 对话完全没有人工输入。【`local_data/papers/author_papers/references/papers/ARCHE_paper.pdf`，第 11 页，PDF 文本行 55–65】

SI 对同一转折给出了相反的叙述：负结果触发了 ARCHE 的 reflection，系统自行提出 Cs⁺–phosphide–底物协同复合物；并称最终机制不是预先指定而是自主自校正得到。【`local_data/papers/author_papers/references/papers/ARCHE_supplementary_information.pdf`，PDF 第 34 页（页眉 S34），文本行 10–25；PDF 第 35 页（页眉 S35），文本行 1–2】因此，最稳妥的结论是：**论文图示层面存在中途人工方向提示；SI 叙述层面声称该提示对应自主 reflection；由于没有原始对话、每次 Planner 请求或失败日志，无法把它裁决为“实际人工输入”还是“示意性用户回合”。但无论哪种解释，都不能把 Case2 宣称为已被公开材料证明的纯 task-only、零人工交互轨迹。**

## 判定规则与证据等级

本审计把信息分成四类：

- **公开用户输入**：论文正文/图中以 `User:` 或明确的初始任务描述出现的信息。
- **作者公开 workflow**：SI 中由作者列出的工具、SMILES/atom mapping 要求、route、扫描或 TS guess 规则。这些是公开的计算协议，不等于用户在运行中逐条输入。
- **通用建模假设**：例如最低能构象代表反应性预复合物、单一 SMD 溶剂近似、线性插值 TS guess、离子对统一设为 neutral singlet 等。这些会影响结果，但不是实验条件，也没有证据显示由用户中途提供。
- **结果 artifact**：`case2.json` 中的 `atomnos`、`atomcoords_final`、`charge`、`mult`、Gaussian route 和能量/激发态数组。它们是发布的计算结果摘要；不能倒推为用户在中途把同样内容发给了模型。

已有复现记录明确指出，发布 JSON 只有 cclib/Gaussian 摘要，没有 Case 2 专用的独立 ARCHE 对话、失败日志或完整 Planner 消息，因此不能恢复“同一次”智能体轨迹。【`docs/research/CASE2_STEPWISE_VALIDATION_2026-08-17.md`】该限制是本报告把“论文中写了什么”和“运行时用户实际输入了什么”分开的关键依据。

## 按阶段审计

### 阶段 0：初始任务输入

图 4 展示的初始 `User:` 回合给出了反应物/产物的缩写结构、2 当量 Cs₂CO₃、EtOAc/CPME 混合溶剂、450 nm 蓝光，并要求提出机制假设和计算验证；图中原文在 solvent 后以省略号结束，所以不能从图 4 断言省略部分的完整内容。【`local_data/papers/author_papers/references/papers/ARCHE_paper.pdf`，第 11 页，PDF 文本行 1–6】

正文对实验体系的描述还报告了 α-iodoboronate 1 与 HPPh₂、Cs₂CO₃ 在 450 nm 下生成产物（83%），以及光、碱、膦和自由基抑制剂相关的控制现象。【`local_data/papers/author_papers/references/papers/ARCHE_paper.pdf`，第 9 页，PDF 文本行 77–81】但 SI 的 Case2 重建明确把“系统收到的实验描述”概括为底物、试剂、溶剂混合物、照射波长和产物形成，未把这些控制实验列为先验输入。【`local_data/papers/author_papers/references/papers/ARCHE_supplementary_information.pdf`，PDF 第 33 页（页眉 S33），文本行 7–11】

更直接的是，SI 在讨论 Mechanisms E/F 时写明：暗反应无产物、自由基抑制剂抑制反应等实验约束**没有作为 prior information 提供给系统**。【`local_data/papers/author_papers/references/papers/ARCHE_supplementary_information.pdf`，PDF 第 39 页（页眉 S39），文本行 23–30；PDF 第 40 页（页眉 S40），文本行 1–5】因此，本审计将这些控制结果标为“论文公开实验事实/后续一致性判据”，而不是已被证明的初始用户输入或中途人工注入。

**阶段判定：**初始用户输入包含反应与实验设置，但没有公开的完整 Cartesian 结构、TS 初猜、接触构型、原子索引或数值扫描范围。省略号使完整初始 prompt 无法从论文图中恢复。

### 阶段 1：Retrieval 与假设生成

SI 叙述 Case2 先从科学问题检索卤素活化、光化学、自由基和 α-boryl radical 相关背景，先生成 25 个策略，再合并为 6 个代表性假设。【`local_data/papers/author_papers/references/papers/ARCHE_supplementary_information.pdf`，PDF 第 33 页（页眉 S33），文本行 12–23】C.8.2 的公开假设文本使用了缩写结构式（如 `Ph-CH2-CH2-CHI-BPin`、`HPPh2`），并提出 SN2、C–I 均裂、XAT、HAT、SET 等候选；例如第一组假设要求在 B3LYP/def2SVP 和 SMD(EtOAc) 下优化反应物、产物和 putative TS，并以虚频/IRC 验证。【同上 SI，PDF 第 69 页（页眉 S69），文本行 17–27】

这些是模型/作者公开的候选机制和计算目标，不是新增用户消息。没有本地原始对话能证明某个用户在 25→6 的筛选过程中逐条指定了 A–F。论文正文只说检索和假设生成引导了 6 个代表性假设进入定量评估。【`local_data/papers/author_papers/references/papers/ARCHE_paper.pdf`，第 10 页，PDF 文本行 1–10】

**阶段判定：**未发现额外人工输入；候选机制属于系统输出/作者整理后的公开结果。由于完整 25 条策略和每次采样没有发布，无法复演每一次假设生成。

### 阶段 2：初始 workflow 与结构准备

SI C.9.2 的“Initial Workflow”明确规定了工作流需要的输入和假设：

- 第 1 步要求为底物与 Cs₂CO₃ 或 Ph₂P⁻ 的接触离子对生成多种初始几何，加入显式 Cs⁺ 配位与 HPPh₂ 氢键；输入栏写的是三组分的 **SMILES strings** 和相关位点的 atom mapping。【`local_data/papers/author_papers/references/papers/ARCHE_supplementary_information.pdf`，PDF 第 142 页（页眉 S142），文本行 20–30；PDF 第 143 页（页眉 S143），文本行 1–11】
- 该页没有给出这些 SMILES 的完整字符串，也没有公开 atom mapping 表、接触距离或接触位点的实际 Cartesian 坐标；它只规定了后续工具应接收什么类型的输入。
- 第 1 步还把力场最低能构象直接假定为“最具反应性的 pre-complex”，不考虑其他离子对排列。这是作者公开的建模假设，不是化学实验条件。【同上 SI，PDF 第 143 页，文本行 1–6】
- 初始第 2 步把所有复合物无论离子对或阴离子特征都指定为 neutral singlet，并使用 B3LYP-D3(BJ)/6-31G(d)、SMD(EthylEthanoate)、`scf=loose guess=mix`。这是一个明确的协议选择，且可能影响电荷转移激发；它不是论文初始 `User:` 消息中的条件，也没有证据来自人工中途回合。【同上 SI，PDF 第 143 页，文本行 14–26】
- 初始 workflow 还假设最低能反应物/产物构象足以代表路径、气相优化几何可转移到溶液，以及线性插值可以生成 TS guess；这些分别见 SI 第 145 页文本行 16–30 和第 146 页文本行 6–14。它们属于工作流简化，而不是用户提供的额外条件。【同上 SI，PDF 第 145–146 页，文本行 6–30】

发布数据能核对“结果结构”而不能证明“输入结构”来自何处。`case2.json` 的记录保存 `atomnos`、`atomcoords_final`、`charge`、`mult` 和 Gaussian metadata；当前复现脚本只把 `atomnos` 与 `atomcoords_final` 转成结构，并从记录读取 `charge`/`mult`，没有 SMILES、初始坐标或 contact metadata 的读取路径。【`scripts/tasks/case23_reproduce.py:339-362`】因此，公开 JSON 中的完整最终坐标是计算结果 artifact，不应反写进 task-only prompt，也不能当作论文运行中用户补入的条件。

**阶段判定：**作者公开了“需要 SMILES/atom mapping/接触构型”的 workflow 输入规范，并公开了优化后的最终坐标摘要；没有证据显示这些完整结构由用户中途逐条提供。neutral-singlet 和最低能 pre-complex 等是作者/工作流假设，影响科学结果但不属于人为用户条件。

### 阶段 3：初始计算、失败与方法修订

SI 叙述的初始执行链包括构象采样、TS guess、TS 优化/势垒，或对光化学候选做构象采样、激发态计算并与 450 nm 光子能比较。ARCHE-Chem 随后将激发态协议从 B3LYP-D3(BJ)/6-31G(d) 改为 B3LYP-D3(BJ)/def2-SVP，理由是 6-31G(d) 不含适用于 I 的合适基函数。【`local_data/papers/author_papers/references/papers/ARCHE_supplementary_information.pdf`，PDF 第 33 页（页眉 S33），文本行 20–31】这是一个由领域模型/协议审查触发的通用方法修订；没有证据是用户中途提供的新条件。

公开失败信息只有摘要级别：B–D 的 TS 优化没有收敛到预期鞍点，或松弛到不符合假设的结构，因而不再推进；E/F 的激发能与 450 nm 不匹配。【同上 SI，PDF 第 39 页，文本行 6–22；PDF 第 40 页，文本行 1–15】没有 B–D 的初始坐标、每次失败的 Gaussian 输入、尝试的 atom mapping、具体 TS guess、优化日志或人工修订消息。仓库复现记录也明确登记了这一缺口：B–D 只有文字结论，没有机器可读初猜、失败日志或最终几何。【`docs/research/CASE2_STEPWISE_VALIDATION_2026-08-17.md` 的 Step 2 和 Step 6】

**阶段判定：**可确认的是自动/协议层面的 basis-set 修订和失败后淘汰；不能确认某次失败后是否有人手工改了结构或 TS guess。把缺失的初猜或失败路径补造出来，会把新的重算输入误称为论文原始输入。

### 阶段 4：反思与 G（CsPPh₂）路径出现

SI 版本的叙述是：A–F 的负结果触发 reflection，reflection 重新审视碱、膦和底物的协同作用，提出包含 Cs⁺、phosphide 和 α-iodoboronate 的复合物；随后复合物形成自由能约为 −3.0 kcal/mol、激发能约为 64.0 kcal/mol，从而支持 G。【`local_data/papers/author_papers/references/papers/ARCHE_supplementary_information.pdf`，PDF 第 34 页，文本行 10–25】正文也把该机制说成不是预先指定，而是在迭代假设、计算和反思中出现。【`local_data/papers/author_papers/references/papers/ARCHE_paper.pdf`，第 10 页，PDF 文本行 11–27】

但图 4 在这一段明确插入了新的 `User:` 回合：`Continue investigating interactions within CsPPh2 pathway`，之后紧接着出现“Iteration”和 G 假设，且图中将 `CsPPh2` 路径与最终蓝光匹配结果连在一起。【`local_data/papers/author_papers/references/papers/ARCHE_paper.pdf`，第 11 页，PDF 文本行 55–65】这条消息不是初始反应物、溶剂、光源或产物描述中的自然重复，而是对具体机制方向的直接指定。

因此这一阶段必须分开记账：

- **若图 4 是实际对话轨迹：**存在中途人类方向性输入；它没有增加新的实验条件，但把搜索空间压向最终 G，属于 oracle-like steering，可能直接影响结论。
- **若图 4 是交互式示意图：**SI 的 autonomous reflection 叙述可以成立；但论文没有提供原始消息日志来证明图中的 `User:` 只是示意，因此不能据此宣称“零人工输入”已经被证实。

该矛盾不能用“没有人工修改 Gaussian route”消除。Methods 的“no human intervention”表述位于跨模型比较协议中，具体说的是不人工修补 Gaussian route、错误推理或答案后再评分。【`local_data/papers/author_papers/references/papers/ARCHE_supplementary_information.pdf`，PDF 第 29 页（页眉 S29），文本行 19–32】它能支持“没有公开证据表明有人手工修 route/改答案”，但不能否定图 4 明示的额外用户消息。

### 阶段 5：修正 workflow 与 G 的实际计算对象

修正 workflow 重新要求多种接触离子对、显式 Cs⁺、HPPh₂ 氢键和 atom mapping，并在优化后做 binding/free-energy 与 TD-DFT；其 route 采用 B3LYP/def2SVP、GD3BJ、SMD(EthylEthanoate)，并注明“如可用”才加入 mixed-solvent dielectric。【`local_data/papers/author_papers/references/papers/ARCHE_supplementary_information.pdf`，PDF 第 149 页，文本行 1–22；PDF 第 150 页，文本行 7–29】这说明 EtOAc/CPME 的真实二元溶剂没有被 Gaussian route 直接完整表示，而是以单一 SMD EthylEthanoate 近似；这是作者/计算 workflow 选择，不是中途用户提供的新实验事实。

修正 workflow 的 TS 部分要求用 atom mapping 对齐反应物，沿 C–I cleavage 与 P–H transfer 坐标插值生成 TS guess，再做单虚频验证和双向 IRC。【同上 SI，PDF 第 151 页，文本行 12–30；PDF 第 152 页，文本行 5–25；PDF 第 153 页，文本行 8–27】这里给出了 TS guess 的**规则**，但没有给出实际生成的 XYZ、原子映射表、键长范围、扫描步数或每个失败候选的文件。

发布 JSON 中可复查的是优化/解析后的结构记录。例如：

| 记录 | 发布摘要可见的结构/电子态 | 直接证据 | 是否是用户中途输入 |
|---|---|---|---|
| E：`cp1`、`cp1_excited` | `cp1` 记录 Gaussian 优化/频率 route；最终记录为 41 原子、`charge=0`、`mult=1`；`cp1_excited` 也为 41 原子闭壳层 singlet | `supplementary/case2.json:5652-5663`、`:5915-5917`；`supplementary/case2.json:6304-6314`、`:6566-6569` | 否；这是发布结果摘要。完整坐标由 `atomcoords_final` 保存并由脚本读取，未见 SMILES/初始坐标输入 |
| E：`cp1_radical` | 40 原子、`charge=0`、`mult=2`，使用 unrestricted B3LYP route | `supplementary/case2.json:5068-5080`、`:5326-5328` | 否；是作者计算对象/结果，不是公开 `User:` 消息 |
| F：`cp4`、`cp4_excited` | 64 原子、`charge=-1`、`mult=1`；记录包含 B3LYP-D3BJ/def2SVP、SMD EthylEthanoate route | `supplementary/case2.json:7327-7339`、`:7729-7731`；`supplementary/case2.json:8085-8095`、`:8485-8487` | 否；电子态是发布记录字段 |
| G：`cp5`、`cp5_excited` | 65 原子、`charge=0`、`mult=1`；`cp5` route 为 B3LYP/def2SVP、GD3BJ、SMD EthylEthanoate；TD 记录同一 65 原子体系 | `supplementary/case2.json:10751-10763`、`:11159-11161`；`supplementary/case2.json:12010-12020`、`:12416-12418` | 否；这是 G 的发布结果 artifact，但 G 的方向选择本身受上节“reflection vs User 回合”矛盾影响 |
| G 的组分：`CsPPh2` | 24 原子、`charge=0`、`mult=1`，保存优化/频率 route 与 `atomcoords_final` | `supplementary/case2.json:14012-14023`、`:14173-14175` | 否；没有证据用户把完整坐标或电子态发给模型 |
| 自由基/离子片段 | `Iodine_radical` 为单原子 doublet，`Iodide_ion` 为单原子 −1 singlet；`PPh2_radical` 为 23 原子 neutral doublet，`PPh2_ion` 为 23 原子 −1 singlet | `supplementary/case2.json:9958-9982`；`:10017-10041`；`:10070-10081`、`:10225-10227`；`:10410-10421`、`:10565-10567` | 否；是作者为候选机制准备/发布的计算对象 |

上述字段能说明“最终计算算了什么”，不能说明“谁在什么时候把结构交给了模型”。尤其是 `scripts/tasks/case23_reproduce.py` 的 Case2 TD 分支使用 `cp1`、`cp4`、`cp5` 的 ground-state `atomcoords_final` 加上记录中的 charge/multiplicity 生成请求，且只做 E/F/G 三个 TD 任务；它不是对原始 ARCHE 对话的回放。【`scripts/tasks/case23_reproduce.py:518-548`】

### 阶段 6：扫描、TS guess、失败修订与最终判据

**Case2 没有公开的扫描原子和数值范围。** SI 的 Case2 修正 workflow 只说沿 C–I cleavage/P–H transfer 坐标插值 TS guess，没有 `scan_steps`、原子索引、起止值或步长。当前脚本中带有 `scan_steps`、`D(i,j,k,l)`、起止角度和步数的代码属于 Case3 二面角扫描请求；Case2 的真实 Action 分支只有 TD-DFT。【`scripts/tasks/case23_reproduce.py:365-405`；`scripts/tasks/case23_reproduce.py:518-560`】不能把 Case3 的扫描坐标误记成 Case2 中途的人为条件。

**Case2 的 TS guess 只有规则，没有原始 guess 文件。** 初始 workflow 规定反应物/产物优化几何之间做线性插值；修正 workflow 规定先按 atom mapping 对齐，再沿 C–I/P–H 变化插值。【SI 第 146 页，文本行 6–14；SI 第 152 页，文本行 5–13】`case2.json` 的 Concerted R/S 对象只保留最终 TS 的 `atomcoords_final`、route 和解析结果；例如 R/S 记录分别标记 Gaussian TS 优化 route 和最终 `charge=0`、`mult=1`，但没有输入 guess 坐标或 NEB/插值轨迹。【`supplementary/case2.json:1789-1800`、`:2196-2198`；`supplementary/case2.json:4247-4258`、`:4654-4656`】

**失败后的修订可以确认是工作流层面的，但不能确认有人工改结构。** SI 公开了 B–D 未找到预期 TS、E/F 激发能过高、之后加入 G 的高层逻辑；没有逐次失败输入和重试日志。【SI 第 39 页，文本行 6–22；SI 第 34 页，文本行 1–25】Case 2 分步验证也明确说 B–D 不能从论文材料独立重放，因为缺初猜几何、失败日志和逐步轨迹。【`docs/research/CASE2_STEPWISE_VALIDATION_2026-08-17.md`】

**最终使用的实验判据不等于中途用户输入。** 图 4 将 I₂、去碱、暗反应和 TEMPO 的对照列为结果表，并把它们称为验证最终机制的控制实验。【`local_data/papers/author_papers/references/papers/ARCHE_paper.pdf`，第 11 页，PDF 文本行 69–97】结合 SI 明说这些 dark/radical-inhibitor 约束未作为先验信息，当前只能把它们标成“实验观察/后验一致性证据”，不能标成某次用户消息补入的条件。

## 逐项结论表

| 审计对象 | 初始 user input 是否明确给出 | 中途是否发现人为补入 | 证据与分类 | 对结论的影响 |
|---|---|---|---|---|
| 反应物、产物、2 eq Cs₂CO₃、EtOAc/CPME、450 nm | 是，至少图 4 明确显示 | 未发现重复补入 | 公开任务输入；图中 solvent 后有省略号，完整 prompt 不可恢复【论文 p.11 行 1–6】 | 定义问题和光子能门槛，不是人工 oracle |
| 83% 收率、光/碱/膦依赖、暗反应、TEMPO 抑制 | 正文报告；SI 明确说后两类没有作为 prior information 提供 | 未发现具体人工消息 | 论文实验观察/后验判据，而非已证实的 prompt 字段【论文 p.9 行 77–81；SI S39 行 23–30】 | 影响最终一致性判断；若被提前注入会缩小机制搜索空间 |
| `Continue investigating interactions within CsPPh2 pathway` | 否 | **是，图 4 明确显示** | 额外 User 回合；机制方向性 steering，不是新的物理条件【论文 p.11 行 55–65】 | 直接指向最终 G，严格复现应视为潜在 oracle；但 SI 声称同一转折由 reflection 自主提出 |
| 完整 SMILES、Cartesian 初始结构、atom mapping | 否，正文只给缩写结构；SI 只写“输入 SMILES/atom mapping” | 没有可核实人工消息 | 作者 workflow 输入规范；实际字符串、映射和初始坐标未公开【SI S143 行 1–11；脚本 339–362】 | 阻碍从零重放接触复合物/TS guess；不能补造后当原始输入 |
| 最终几何、`atomnos`、`charge`、`mult` | 不属于初始 user input | 未发现人工消息 | `case2.json` 发布计算结果 artifact；脚本从 final coords/字段构造新请求【JSON 上述各对象行；脚本 339–362】 | 可用于数据重放，不能证明人工在中途提供 |
| Cs⁺ 配位、HPPh₂ 氢键、接触构型 | 初始 prompt 未给具体构型 | 没有可核实人工消息；图 4 的 CsPPh₂ 方向提示除外 | SI workflow 要求生成多种接触构型，是作者/Planner 协议；没有实际接触坐标或 mapping【SI S142–S143；S149 行 1–22】 | 接触排列会影响 G 激发能；是重要未公开输入，不应隐式填入 |
| 电荷/自旋 | 初始 prompt 未给计算态逐物种设定 | 没有可核实人工消息 | 初始 workflow 曾统一 neutral singlet；发布 JSON 另有 −1 singlet、neutral doublet 等结果【SI S143 行 14–21；JSON 相关对象行】 | 是模型设定/协议风险，不是用户新增实验条件 |
| 方法、基组、溶剂近似 | 初始 prompt 给实验混合溶剂，但没有完整 Gaussian route | 没有人工消息；有自动/协议层面修订 | B3LYP-D3BJ/def2SVP/SMD EthylEthanoate；6-31G(d)→def2-SVP 因 I 基组修订【论文 p.21 行 11–14；SI S26 行 26–33；SI S33 行 27–30】 | 是作者/ARCHE-Chem 的建模选择；二元溶剂被单一 SMD 近似 |
| 扫描原子、起止值、步长 | 不适用，Case2 公开 workflow 没有一维 scan | 未发现 | Case2 只有 C–I/P–H 插值 TS 规则；`scan_steps` 代码是 Case3 分支【SI S152 行 5–13；脚本 365–405、549–560】 | 不应为 Case2 追加 Case3 扫描条件 |
| TS guess、B–D 失败输入、重试修订 | 未给出 | 未发现可核实人工结构修订 | 仅公开线性/映射插值规则和失败摘要，无原始文件/日志【SI S146 行 6–14；S152 行 5–25；docs Stepwise 103–114】 | 不能声称完整复演原始 TS 搜索；新的 guess 只能标为独立重算 |

## 对当前 task-only 实验的含义

2026-08-20 的旧流程自主复现记录把论文候选、`case2.json` 数值和旧运行产物排除在 task-only 输入之外，并记录执行角色只能从任务描述构造结构；因缺少完整三维结构、键变化坐标和可验证 TS 初猜，后续高成本步骤没有完成。这与本审计的证据边界一致。原记录可通过 `git show cfb12c1:docs/current/CASE2_AUTONOMOUS_REPRODUCTION.md` 查看。

因此，后续对 Planner 的严格验证应至少保留两套标记：

- **纯 task-only：**不把图 4 的 `CsPPh2` 方向提示、`case2.json` 最终结构/数值、SI 的 G 假设或任何未公开 contact/TS guess 注入初始或中途 prompt；只允许模型从实验描述和工具返回结果自发提出 G。
- **作者路径重放：**如果为了测试 workflow 可执行性而选择 G、复制发布坐标或使用 SI 的 refined workflow，必须在运行记录中标明这是“公开候选/发布 artifact 驱动的重放”，不能称为无人工先验的独立发现。

本审计最终答案是：**没有证据表明中途有人补入了新的实验条件或完整量化输入；但论文图 4 明确显示过一条会把搜索指向 CsPPh₂ 的额外 User 消息。SI 同时把该转折描述为自主 reflection，且缺少原始轨迹，因此 Case2 的“零人工参与”不能由现有论文材料单独证明。**
