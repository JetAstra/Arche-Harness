# Case 3 中途人为输入审计（2026-08-19）

本文是论文材料和当时仓库状态的研究审计，不是当前 ARCHE 运行说明。

## 结论

本报告只审计论文 Case 3，不审计 Case 2。

结论需要区分两个问题：

1. **是否发现了 Case 3 运行中途由人通过新的 user message 追加条件？** 没有发现这样的直接证据。论文和补充材料把 Case 3 的起点描述为“实验选择性趋势的自然语言描述”，并说没有预先指定 descriptor；它把检索、候选 descriptor 生成、工作流构造、Gaussian 专家复核和执行归给 ARCHE 的代理链。[主论文 p.12–13；SI S35–S36]
2. **是否存在原始 user input 没有包含、但后续可执行计算必须使用的条件？** 明确存在。补充材料的示例 workflow 和发布的 `case3.json` 出现了具体的 Ni(II)-bipyridine–Br–H 结构、SMILES/XYZ/SDF/GJF 数据流、neutral singlet 约定、方法/基组/溶剂、扫描坐标和范围、原子索引、BDE 解离物种等。它们不是原始题面的字段。[SI S55–S57、S105–S119、S154–S163；`supplementary/case3.json:1–16, 1480–1526, 1938–1958, 22491–22504, 23895–23920`]

因此，严格表述应是：**“未见中途人工 user message”的说法没有被现有公开材料证实；“没有任何额外输入/条件”的说法则与公开 workflow 和计算 artifact 不相容。** 后续条件的来源在论文中没有逐字段审计：它们可能是 Hypothesis/Planner/ARCHE-Chem 产生或修订的，也可能在实际运行前由作者提供；现有 PDF、JSON 和当前仓库文档不能证明是哪一种。

这些后续条件会影响可执行性，也可能影响数值 profile 和排序，但目前没有证据表明作者把最终 descriptor 或最终数值作为 oracle 直接输入给模型。另一方面，示例 workflow 已经写入“ortho 应有更浅能量上升、从而有利于迁移”的预测；如果这段示例文字在计算前原样提供给 Planner，它会构成答案导向信息。论文没有给出 Case 3 的逐阶段原始 prompt/trace，因此这里只能登记为 **oracle 风险和证据缺口**，不能判定为已发生的 oracle 干预。[SI S105–S106、S158、S163]

## 1. 审计范围和证据等级

### 1.1 使用的本地一手材料

- 主论文：`local_data/papers/author_papers/references/papers/ARCHE_paper.pdf`，重点为 p.12–13 的 Case 3 描述和 Figure 5。
- 补充材料：`local_data/papers/author_papers/references/papers/ARCHE_supplementary_information.pdf`，重点为 C.7（S55–S57）、C.8.3（S105–S119）、C.9.3（S154–S163）、B.1.3/B.2.3（S35–S42）和 C.10（S163–S164）。
- 发布机器可读数据：`supplementary/case3.json`。这是 cclib 风格的 Gaussian 输出摘要，不是原始 user prompt 或完整 Gaussian 输入；其结构字段和行号用于核对后续实际出现的条件。
- 当前本地复现代码和文档：`scripts/tasks/case23_reproduce.py`、`docs/research/CASE123_MINICHEM_VALIDATION_2026-08-13.md`。它们只用于辨别本仓库后来如何从发布 artifact 组装复算请求，不被当作作者运行 trace。

未联网，未读取或修改 Prompt、Run Bundle、模型缓存或论文外部网页。

### 1.2 本报告中的分类

- **公开 user input**：明确写在 C.7 题面、主论文 Figure 5 的 User 区域或同一题面的背景条件中。
- **代理/专家生成**：论文明确归给 Retrieval、Hypothesis、Planner 或 ARCHE-Chem 的内容；它不是人通过 user message 追加的条件，但仍是原始题面之外的信息。
- **公开建模假设/协议**：补充材料 workflow 中为使任务可执行而写下的结构、电子态、方法、扫描和判据。它是公开的作者 protocol，但没有逐字段来源证明。
- **执行 artifact**：`case3.json` 中的坐标、能量、频率、扫描点和 parsed metadata。这些是计算结果或结果摘要，不能反向证明它们曾作为人工输入提供。
- **oracle/人为干预**：只有在证据显示人将结果、目标 descriptor、最终数值或人为修订直接送入后续代理时才这样判定。本材料没有达到该证据标准；对缺少 trace 的部分标记为“无法判定”。

## 2. 原始 Case 3 user input 的边界

主论文 p.13 的 User 文本和 SI C.7（S55–S57）给出了以下内容：

| 类别 | 原始题面已有信息 | 是否足以直接生成计算输入 |
|---|---|---|
| 反应与选择性 | Ni 催化的 reductive migratory cross-coupling；TBAB、强碱 LiOH、DMA；邻氮位 methyl 的 2,2′-bipyridine 几乎只给 migratory product；unsubstituted、meta/para-methyl（无 ortho substitution）几乎只给 in situ product。 | 不足。它给出研究对象和目标趋势，但没有具体分子文件。 |
| 抽象催化模型 | Ni(II) 通常为四配位 square-planar；bipyridine 的两个 N 螯合 Ni；R-Br、Br− 和相邻 β-H 形成 N–N、Br、H 的四配位中间体；该抽象模型是迁移的前提。 | 部分足够。它限定了应比较的化学模型，但未给出原子连接和三维构型。 |
| 研究任务 | 提出解释 ortho substitution 反转选择性的 abstract hypotheses，并对每个指标设计低成本、可比较趋势的计算实验；不指定显式 substrate 或完整 TS。 | 足以让代理提出计划，不足以让 Gaussian 直接运行。 |
| descriptor | 题面要求比较 geometric/electronic/energetic indicator，且指标要跨 unsubstituted、ortho、meta、para 产生趋势；没有指定最终 descriptor。 | 不含最终 descriptor。SI S35–S36 也明确写“no descriptor was specified in advance”。 |
| 结构表示 | 没有 2,2′-bipyridine 四种配体的 SMILES、原子连接表、Ni/Br/H 的具体连接或文件。 | 不足。 |
| 坐标和构象 | 没有 Cartesian coordinates、初始构象、Ni–N/Ni–Br/Ni–H 距离、平面取向或扫描起点。 | 不足。 |
| 电荷和 multiplicity | 题面只说 Ni(II) 和四配位模型，没有给总电荷、spin multiplicity 或开放壳层解离物种的电子态。Ni 的氧化态不能替代整分子的 charge/multiplicity。 | 不足。 |
| 计算方法 | 只给 DMA 溶剂背景和“低成本 quantum-chemical methods”的原则；没有 functional、basis、dispersion、SCRF 细节、温度或收敛阈值。 | 不足。 |
| 扫描定义 | 只要求指标随数值变化并跨配体比较；没有 scan coordinate 的原子索引、起止值、步长、是否 relaxed/constrained、是否 NoSymm。 | 不足。 |
| TS/中间体 | 题面明确允许不指定显式 substrate、不计算完整 TS；没有 TS 坐标或中间体文件。 | Case 3 的 descriptor workflow 不需要完整 TS，但仍需具体的抽象复合物。 |

原始题面本身还给了“比较范围、趋势和理论联系”的评价标准，所以这些不是中途人工追加；真正缺失的是可执行量化计算所需的结构和数值设置。[SI S55–S57]

## 3. 逐阶段审计

### 3.1 题面到 Retrieval

SI B.1.3 说系统只接收实验选择性趋势的自然语言描述，随后 Retrieval Agent 检索 ligand effects in nickel catalysis、migratory cross-coupling 和 Ni(II) coordination geometry 等背景。[SI S35]

这些检索主题是**代理阶段产生的上下文**，不是人对原始 user input 的追加。论文没有给出每条检索结果或检索 prompt，因此无法判断检索上下文是否带入了具体配体结构、计算方法或作者答案；但现有材料没有把这些内容描述为人工消息。

### 3.2 Hypothesis Agent：候选 descriptor 和方法初稿

论文/SI 将候选生成归给 Hypothesis Agent：三类领先候选是 N–Ni–N bite angle、Ni–H 电子调控和 Br–N–N–H 二面角的 out-of-plane distortion。[主论文 p.12；SI S35–S36]

SI C.8.3 进一步保存了多个候选对象。它们已经包含原始题面没有的具体条件，例如：

- 将模型写成含 ligand N–N、Br、H 的 tetra-coordinate Ni(II) complex，并建议在 M06-L/def2-TZVP/SMD(DMA) 下扫描 Ni–H 二面角 0–30°。[SI S105–S106]
- 对 bite angle 建议 B3LYP-D3BJ/def2-TZVP/SMD(DMA)、相对平衡值 ±10°、2° 步长；对 Ni–H BDE/NBO 建议 M06/def2-TZVP/SMD(DMA)。[SI S106]
- 其他候选还提出 Br–Ni 距离增加 0.3 Å、二维 out-of-plane 扫描、τ4、NBO/Wiberg、Ni–Br 扫描、singlet–triplet gap 等不同指标和方法。[SI S107–S117]

这些条件是**模型提出的科学实验设计**，不是原始 user input，也没有证据显示它们是人逐项提供的。它们会影响 Planner 接收到的 strategy/reasoning，并决定后续比较哪些候选；但在这一阶段仍是候选方案，不是已经执行的最终设置。

### 3.3 Planner/ARCHE-Chem：从抽象策略到可执行结构

这里出现了最重要的输入鸿沟。C.9.3 的“Initial workflow”把每个体系具体化为“square-planar Ni(II) complex（N–N ligand, Br, H）”的四种配体，并给出如下步骤：

1. 需要每个配体变体的 Ni(II) complex SMILES，执行 `smiles2sdf`；
2. SDF 转 XYZ；初始 workflow 明确把所有 complex 赋成同一个 **neutral singlet**；
3. 对生成结构做初始 geometry optimization；
4. 生成 relaxed dihedral scan 的 Gaussian route；
5. 从 optimized XYZ 生成 GJF，再运行 Gaussian scan。[SI S154–S157]

这段 workflow 还写出“生成的结构假定保留正确 square-planar coordination geometry 和 metal-centered electronic configuration，无需额外验证 spin state 或 coordination mode”。这不是 C.7 user input 中的数值条件，而是使结构生成可以继续的**公开建模假设**。[SI S154–S155]

公开 workflow 给出的初始/修订协议包含：

- H–N–N–Br（等价于 Br–N–N–H）作为 Ni–H out-of-plane proxy；
- 0°–60°、示例 5° 步长的 relaxed dihedral scan；
- `ωB97XD/def2-SVP`、SMD(DMA)、`Opt=ModRedundant`，初始 workflow 还写有 `freq`、`scf=loose`、`guess=mix` 和后续曲线分析；
- 以能量曲线斜率/上升率联系 migratory selectivity，且 initial workflow 的 Step 9 直接写出 ortho-methyl 应有更浅的能量上升。[SI S156–S158]

Refined workflow 仍然需要相同的四类具体 complex、结构转换、优化、0°–60°二面角扫描和跨 ligand 曲线比较；但它删除或简化了部分初始 workflow 的说明，保留 `ωB97XD/def2-SVP/SMD(DMA)` 和 Step 9 的选择性预测。[SI S159–S163]

论文主文只概括说 Planner 将候选转成 targeted computational tasks，随后由 Gaussian-specialized model 验证 setup；SI B.1.3 说 workflows “reviewed by ARCHE-Chem before execution”。[主论文 p.12；SI S35] 这证明有一个代理/专家复核阶段，但没有逐字段的原始 prompt、修改 diff 或人工审批记录，所以不能把这些具体字段归因给人工。

### 3.4 Execution：发布 JSON 中实际出现的结构、电子态和坐标

`supplementary/case3.json` 是 Gaussian/cclib 解析摘要。它显示执行时已经存在原始题面没有的具体结构和电子态：

- `dihedral.opt` 的四个 parent 记录含原子序列、最终 Cartesian coordinates、Gaussian route、`charge: 0`、`mult: 1`；例如 meta 的 route 和原子序列在 `supplementary/case3.json:5–17`，坐标起点在 `:48–52`，电子态字段在 `:195–197`；unsubstituted、ortho、para 的对应 metadata/charge/multiplicity 在 `:452–465, 607–609`、`:720–733, 911–913`、`:1042–1055, 1233–1235`。
- `dihedral.scan` 的记录以 Gaussian `D(i,j,k,l)` 原子索引定义扫描坐标，包含 13 个完成点和“converged geometries aligned to scan energies”。meta 的 route、电子态和 scan 摘要见 `:1480–1526, 1938–1958`；ortho 的对应数据见 `:6003–6049, 6716–6736`；unsubstituted 的 route 和 scan summary 见 `:3994–4005, 4032–4035, 4336–4357`。
- 发布值中的坐标索引并不由原始题面唯一给出：四个二面角记录分别使用 `D(29,4,9,28)`、`D(21,3,8,20)`、`D(19,3,8,18)` 和 `D(19,3,8,18)`；unsubstituted 记录甚至保存了重复的 `scan_names`。具体起止值也不是抽象的“扫描一个角度”，而是从约 −4°/0° 到约 56°/60° 的 13 点 profile。[`supplementary/case3.json:1938–1958, 4336–4357, 6716–6736, 9285–9305`]

这些字段说明执行确实需要一套具体结构、原子映射和 scan 参数，但 JSON 只证明它们出现在执行结果摘要中，不能证明是由人中途发送的 user input。SI C.10 也明确说三个 JSON 是从 Gaussian logs 用 cclib 解析出的机器可读文件，供后续检查和 re-analysis；它们不是 prompt transcript。[SI S163–S164]

### 3.5 Bite-angle workflow 的额外条件

`bite_angle` 分支进一步显示，候选假设执行时使用了另一套具体协议：

- optimized complexes 使用 B3LYP、GD3BJ、def2-TZVP、SMD(N,N-dimethylacetamide)，例如 meta 记录在 `supplementary/case3.json:11343–11356`；
- add/reduce 扫描使用 `opt=modredundant`，同一 B3LYP/def2-TZVP/GD3BJ/SMD(DMA) route，见 `:12736–12748` 和 reduce 分支 `:18103–18114`；
- 扫描是六个点，不是题面指定的某个范围；例如 meta add 的坐标为 `A(4,27,9)`、六点摘要和起始角见 `:12906–12926`。其他 ligand 的原子索引和起止方向同样记录在 JSON 中。

这套 B3LYP-D3BJ bite-angle protocol 不是 C.7 user input 中的固定设置，而是候选策略/Planner workflow/执行 artifact 之间形成的协议。它会影响“coordination-plane flexibility”这一候选的数值结论，但不能据此判定发生了人工注入。

### 3.6 BDE/NBO 分支的额外物种和电子态

原始题面没有要求直接把 Ni–H 断成自由基，也没有给出 H atom 或 parent-H 结构。电子假设在 SI C.8.3 中由 Hypothesis Agent 提出 NBO/Wiberg 与 BDE 测试；SI B.2.3 后来据计算结果说明 Wiberg 指标和 BDE 不能同样直接表示迁移坐标。[SI S40–S42、S105–S106]

发布 JSON 证实 BDE 执行阶段出现了九个额外对象：四个 parent、四个 `ligand-H` radical 和一个 H atom。parent 使用 M062X/def2-TZVP/SMD(DMA)，`pop=nboread`，并以 neutral singlet 记录；meta parent 的 route/坐标/charge/multiplicity 见 `supplementary/case3.json:22491–22504, 22537–22543, 22684–22686`。H atom 被记录为 `charge: 0, mult: 2`，见 `:23895–23920`；meta-H、unsubstituted-H、ortho-H、para-H 的 radical 记录也都为 `charge: 0, mult: 2`，例如 `:23956–23967, 24141–24143`、`:24377–24385, 24526–24529`、`:24714–24725, 24899–24902`、`:25147–25155, 25332–25335`。

这些解离物种是**由 BDE 假设导出的计算对象**，不是题面中的 user input，也没有公开证据证明是作者在中途手动告诉模型的。它们对 BDE 数值和“ortho 较易 Ni–H cleavage”的比较有直接影响，但不等同于已证明的迁移路径。

### 3.7 Reflection/descriptor 选择

SI B.1.3 和 B.2.3 将结果选择叙述为计算后的代理/反思过程：bite-angle 只有部分定性一致但与迁移结构变化不够直接；Ni–H Wiberg 最大值与“ortho 弱化”预期相矛盾，BDE 虽方向相容但只表示断键热化学；Br–N–N–H dihedral profile 则给出 ortho 最低 distortion penalty，因而被保留为最终 descriptor。[SI S35–S36、S40–S42]

这一步的**结论和 descriptor 选择不是原始 user input**。论文将其归因于候选生成、定向计算、排除和选择的 autonomous pipeline；但由于没有原始 reflection prompt、候选结果文件和时间顺序，无法核验是否曾有人在反思前把“dihedral 是正确答案”或数值排名告知系统。现有证据只支持“公开叙述声称由计算比较后选择”，不支持“已经排除了所有人类先验”。

## 4. 当前仓库复现证据的边界

当前仓库明确区分发布数据重放和从结构重新计算。现有文档说明 Case 2/3 JSON 是 cclib 摘要、不含全部原始 Gaussian logs，发布 scan profile 的重放不等同于从零优化所有几何。[`docs/research/CASE123_MINICHEM_VALIDATION_2026-08-13.md` 的「已知边界」]

当前 `scripts/tasks/case23_reproduce.py` 会从发布 JSON 读取坐标和原子序列，再组装真实 Action request：

- `_atomic_structure` 从 `atomnos`、`atomcoords_final`/scan 起始坐标、`charge`、`mult` 构造结构；
- `_dihedral_request` 从 `scan_summary.scan_names` 解析 atom indices，从首尾 `scan_parameters` 解析 start/end，补入 wB97XD/def2SVP/SMD(DMA)、steps、NoSymm 等 settings；
- `_bite_angle_request` 同理补入 B3LYP/def2TZVP/GD3BJ/SMD(DMA) 和 angle settings。[`scripts/tasks/case23_reproduce.py:339–405, 408–452`]

这里还有一个与 provenance 无关、但会直接影响复现可执行性的字段不一致：unsubstituted 的二面角记录在 `supplementary/case3.json:4355–4357` 使用 `scan_parameter_values`，而当前 `_scan_rows`/`_dihedral_request` 只读取 `scan_parameters`（`scripts/tasks/case23_reproduce.py:192–207, 378–381`）。因此不能把“脚本能从 JSON 组装四个 ligand 的完整请求”当作已经证明的事实；至少 unsubstituted 需要兼容性处理或显式人工/程序转换。这是发布数据 schema 与复现脚本的技术缺口，不是 Case 3 原作者中途人工输入的证据。

这说明当前本地复现若没有发布 artifact，就必须另行提供或生成这些字段；但该脚本是后来的复现工具，不是作者 Case 3 运行中人工输入的证据。研究记录还说明：发布 JSON 没有 Figure S4 所述 Wiberg 数值或原始 NBO 日志，仓库后来以新增 `calculate_bond_orders/gaussian` 对四个发布几何重新计算。[`docs/research/CASE123_MINICHEM_VALIDATION_2026-08-13.md` 的「本轮真实 Wiberg 计算」]

因此，不能把当前仓库后来为复算而设置的 method spec、atom indices、NoSymm、BDE 组合或 NBO Action 反推成论文作者在 Case 3 中途发送的 user input。

## 5. “人为参与”逐项判定表

| 后续信息 | 原始题面是否有 | 公开材料中何时出现 | 最稳妥的来源分类 | 对结论的影响 |
|---|---:|---|---|---|
| 四种具体 bipyridine–Ni–Br–H 结构/连接关系 | 否 | C.9.3 workflow；执行 JSON 的 atomnos/coordinates | 公开 workflow 的建模对象；来源 provenance 未公开 | 决定是否真的比较了同一类四配位模型。 |
| SMILES、SDF、XYZ、GJF 数据流 | 否 | SI S154–S157、S159–S161 | 公开 workflow 协议，不是 user message 证据 | 决定结构能否进入 Gaussian。 |
| square-planar、neutral singlet 约定 | 仅有概念性的 square-planar/Ni(II)背景；无数值电子态 | SI S154–S155；JSON parents `charge=0,mult=1` | 一般建模假设/公开 workflow；若实际由人指定则属于额外条件，但无法判定 | 直接影响电子结构和几何优化。 |
| BDE 的 `ligand-H` radical 与 H atom | 否 | SI S106；JSON BDE 分支 | 由电子假设导出的计算对象，不是题面输入 | 直接决定 BDE/BDFE，不能替代迁移坐标。 |
| wB97XD/def2-SVP/SMD(DMA) dihedral route | 否 | SI S156–S163；JSON `dihedral.scan` | Planner/ARCHE-Chem 或协议作者，来源未公开 | 决定 13 点 profile 和 distortion penalty。 |
| `D(i,j,k,l)` 原子索引、起止角、13 点 | 否 | JSON scan summaries；当前复现脚本解析 | 执行 artifact/后续复现输入；不是已证实的人工追加 | 坐标选错会改变 profile 的科学含义。 |
| B3LYP-D3BJ/def2-TZVP/SMD(DMA) bite-angle route、6 点 add/reduce | 否 | SI S106、S154–S163；JSON `bite_angle.scan` | 候选 workflow/执行 protocol，来源未公开 | 影响 bite-angle 候选的定性比较。 |
| M062X/def2-TZVP/SMD(DMA)、NBO/BDE | 否 | SI S106、S40–S42；JSON BDE metadata | 候选 workflow/执行 protocol，来源未公开 | 影响电子候选排除及 BDE 排名。 |
| “dihedral 是最终 descriptor” | 否，题面只要求提出并比较指标 | 计算比较之后的 SI B.1.3/B.2.3 和主论文 p.13 | 反思/作者报告的结果；无证据是预先人工输入 | 直接决定最终机制抽象，但不能把结果倒灌到起点。 |
| 失败后的人工修订、补坐标、补方法 | 未给出 | Case 3 论文/SI 没有逐次 trace | **无法判定**；没有直接人工证据，也没有足够 trace 排除 | 是严格 autonomous claim 的关键证据缺口。 |

## 6. 是否构成 oracle 或人为干预

### 6.1 可以排除的强断言

不能说“Case 3 的每一个中间字段都在原始 user prompt 中”，因为原始题面没有这些字段；公开 workflow 和 JSON 已明确反证这一点。

也不能说“发布 JSON 就是作者在中途给模型的输入”。SI C.10 明确把 JSON 定义为从 Gaussian logs 解析出的机器可读输出，研究记录也把它定义为发布数据重放和后续重新计算的来源。[SI S163–S164；`docs/research/CASE123_MINICHEM_VALIDATION_2026-08-13.md` 的「已知边界」]

### 6.2 目前没有证据确认的事项

以下问题现有材料无法回答：

1. 四种具体 complex 的初始 SMILES 是由 Hypothesis/Planner 自动生成，还是作者在执行前提供给系统；
2. neutral singlet、具体 coordinate topology 和初始 Cartesian coordinates 是代理默认值、ARCHE-Chem 修订，还是人工选择；
3. 最终 `D(...)`/`A(...)` 原子索引和 scan endpoints 是从生成结构自动抽取，还是由人检查后修订；
4. BDE 的 parent/radical/H 输入是否在 workflow 生成阶段由模型提出，还是由人把 BDE 实验补成可执行任务；
5. 任何失败后是否发生过未记录的手工换结构、换方法、换范围或挑选结果。

论文只给最终叙述和部分示例 workflow，没有逐阶段消息、artifact provenance、人工操作日志或可验证的 before/after workflow diff。因此对这些问题应标记“未公开/无法判定”，而不是推断“没有人工参与”。

### 6.3 oracle 风险

“没有预先指定 descriptor”是 SI B.1.3 的明确叙述；候选 A/B/C 的比较和最终 C 的选择也被写成计算后的结果。[SI S35–S36、S40–S42]

但 C.9.3 的 workflow Step 9 已在执行协议中写出“ortho-methyl ligands show shallower energy rise and thus favor migratory products”，并把 scan slope 直接联系到选择性。[SI S158、S163] 这段如果只是作者在结果之后整理的示例，不是 oracle；如果它在计算之前就作为 Planner 输入，则已经包含方向性答案。当前材料没有时间戳和原始调用上下文，所以应记录为 **潜在泄漏面，不能定性为已发生泄漏**。

## 7. 对当前 Case 3 自主实验的建议边界

若目标是验证 Planner 能否从 task-only Case 3 生成合理且可执行 workflow，建议把下列字段全部视为运行时显式 artifact，并记录 provenance：

1. `structure_source`：四种 ligand complex 是由 Hypothesis/Planner 生成、工具构造，还是人工提供；
2. 原子连接/SMILES、Ni/Br/H 的 atom map、初始 Cartesian coordinates、coordination geometry；
3. `charge`/`multiplicity`，以及 parent、H atom、`ligand-H` radical 的电子态；
4. method、basis、dispersion、solvation、temperature、SCF/geometry convergence 和 symmetry 设置；
5. 每个 scan 的 coordinate type、原子索引、起止值、步长/点数、relaxed/constrained 语义；
6. BDE/NBO 是否为 Hypothesis Agent 提出的候选，还是人工追加的辅助验证；
7. 反思前后完整 workflow、失败事实和修订 diff，尤其不能把 Step 9 的最终趋势预测当作 task-only 输入；
8. 任何人工介入都单独写入 `human_input_events`，没有事件时也写明“未发生/未观测到”，而不是从缺少日志推断“无人工参与”。

这样才能把“作者公开的后续 protocol”与“本轮代理自主生成的条件”分开，也能检验当前 Planner 失败究竟来自缺结构/电子态/扫描字段，还是来自 workflow 拓扑和工具合同。

## 8. 最终判定

| 问题 | 判定 |
|---|---|
| Case 3 题面是否已经包含所有可执行计算条件？ | **否。** 结构、坐标、电子态、方法、扫描定义和 BDE 物种均有缺口。 |
| 是否有公开证据显示中途出现了新的人工 user message？ | **未发现直接证据。** |
| 是否有公开证据显示后续阶段使用了题面之外的条件？ | **有。** SI workflow 和 `case3.json` 明确包含这些条件。 |
| 能否把这些条件全部归因于代理自动生成？ | **不能。** 论文没有提供逐阶段原始 trace。 |
| 能否判定存在 oracle/人工把最终 descriptor 提前喂给模型？ | **不能确认；存在 workflow 示例造成的 oracle 风险。** |
| 对论文“Case 3 自主抽象”的最准确复述 | **公开叙述声称从自然语言趋势出发、代理生成并比较 descriptor；但从公开材料无法审计中间结构和协议条件的来源，因此不能把“无额外人为条件”当作已证事实。** |
