# ARCHE Case 1 逐步复现验证（2026-08-17 快照）

更新日期：2026-08-17
代码基线：`4990f88fc17fcc92334a0aa392256df57c5d856f`

本文保存 2026-08-17 代码基线上的研究证据，不是当前运行指南；下文的 `.runtime/` 路径和状态只表示当次验证环境。当前入口见[交付与复现说明](../current/DELIVERY_AND_REPRODUCTION.md)。

本文按 ARCHE 补充材料 A.3.4 的 11 步流程记录 Case 1 复现。每一步都区分目的、输入、实现、
科学门禁、实际结果、遇到的问题和当前代码不足。运行产生的 Gaussian 日志、checkpoint、模型输出和
大型 JSON 放在 Git 忽略的 `.runtime/`；本文只保存必要的数值、哈希和相对路径。

## 复现目标与证据等级

“复现”分为三个不同目标，不能互相替代：

| 等级 | 目标 | 成功含义 |
|---|---|---|
| E1：发布数据重放 | 从 `supplementary/case1.json` 重算频率记录、复合自由能、排序和 ee | 证明发布数据内部的数值计算可以重放，不证明结构是本轮找到的 |
| E2：固定发布几何重算 | 从发布坐标重新执行 Hessian、Gaussian 高层单点和关键 IRC | 证明当前工具能在作者发布几何上重做计算；只有 IRC 端点闭环后才能验证连接关系 |
| E3：从反应输入独立发现 | 从反应物/产物开始做映射、构象、TS guess、TS 优化、频率、IRC、能量排序 | 才能检验 Agent/工具箱能否从头找回论文路径；结果不必逐坐标等同历史运行，但必须通过统一科学门禁 |

论文当时的完整 Agent 消息、初始 TS 坐标、原子映射、Gaussian 原始日志和失败轨迹没有公开，
因此无法恢复“同一次历史运行”；本轮目标是形成一条新的、可审计的计算证据链。

## 来源与固定输入

| 来源 | SHA256 |
|---|---|
| `ARCHE_lidong/manuscript.pdf` | `1712d2e77455b6ecf1e6de1141b764ad60e6c92bb9f2e03c15fff1e6265b845c` |
| `ARCHE_lidong/Supplementary Information.pdf` | `b2f5aacac2b58b62baaa8b43edff451ddffd2520c319da6c39031e2e86eed70f` |
| `supplementary/case1.json` | `28cba09d3ef6e450be88df83d0183f364b78dee8b513ad7a27c89c14fb8e14c4` |

论文报告的比较目标为：TS-2a 是 major、TS-2b 是 minor，主产物方向为 R，backbone RMSD
为 0.01/0.15 Angstrom，竞争路径 `delta-delta G^‡ = 1.9 kcal mol-1`；实验 88% ee 在
303.15 K 对应约 1.66 kcal mol-1。

## SI A.3.4 的 11 步精确覆盖

下表按补充材料给出的原始工具步骤逐项登记。这里的“已有 Action”只表示接口能力存在；只有“本轮证据”
列实际完成的计算才算复现证据。

| SI 步骤 | 补充材料规定的动作 | 当前实现映射 | 本轮证据等级与状态 | 尚缺门禁 |
|---:|---|---|---|---|
| 1 | `TSPipline`：从反应输入生成两条对映路径的 TS 初猜 | 原子 Actions：映射、显式氢转移、反应变化、构象、反应复合物、端点构建；也有 `search_arche_23_transition_state_path` 组合入口 | E3 准备阶段和一次路径搜索尝试已完成：守恒的无催化剂 R/S 产品各筛选 1 对 MMFF94s 端点；R=`0007`、S=`0006` 的 GFN2-xTB/NEB 均运行 100 周期但未收敛，而且路径最高点均为反应物端点，未得到可用 TS guess | 原文字面输入不守恒；端点 Action 不把目标 CIP 当约束；MMFF 端点未先在同一 xTB 势能面优化，路径没有内部势垒；还缺明确催化物种和充分构象覆盖 |
| 2 | `GenerateGaussianCode`：生成 TS 优化和频率 route | `locate_transition_state/gaussian` 与 `calculate_hessian/gaussian` 生成 typed Gaussian 请求/deck | 能力存在；本轮 Step 1 没有通过门禁的内部最高图像，因而按 fail-closed 原则没有提交 E3 Gaussian TS 优化 | 必须绑定候选 ID、原子序列、电荷、多重度、方法和温度 |
| 3 | `xyz_to_gjf`：把 Step 1 XYZ 与 Step 2 route 合成 GJF | Gaussian backend 内部写 GJF | E2 发布坐标的 Hessian GJF 已生成；E3 未开始 | E3 仍没有合格的 Step 1 TS XYZ |
| 4 | `Gaussian`：TS 优化和频率 | `locate_transition_state/gaussian` 后接 `calculate_hessian/gaussian`、`derive_vibrational_modes` | E2 已完成关键候选 TS-2a/TS-2b-1 的 `Opt=TS` 重优化、Hessian 和投影振动：两者均收敛、恰有 1 个虚频；14 个发布几何的固定 Hessian 证据也保留 | 仍需从 E3 产生的 guess 独立通过同一门禁，并把虚频模式与完整反应核心及 IRC 端点绑定 |
| 5 | `generate_gaussian_code`：生成 IRC route | `trace_intrinsic_reaction_coordinate/gaussian` 的单方向 typed 请求 | E2 已为两个关键候选分别生成 forward/backward 请求 | 方向、checkpoint 和失败恢复要逐分支固定 |
| 6 | `xyz_to_gjf`：把优化 TS 转成 IRC GJF | IRC backend 内部写 GJF | E2 已为 TS-2a、TS-2b-1 生成 4 个 GJF；输入仍是发布坐标，而非本轮重优化结构 | 后续正式运行应从通过 Step 4 门禁的 checkpoint 继续 |
| 7 | `Gaussian`：IRC 并检查 TS 连接正确反应物/产物 | 单向 IRC Action，forward/backward 成对调用；另有 `validate_reaction_path` 做轨迹几何检查 | E2 的 4 个原始 Action 在 7200 秒超时；随后从 29 点 checkpoint 副本用原生 `IRC=Restart` 续跑：TS-2a backward 在点 31 检测 PES minimum；TS-2a forward、TS-2b-1 forward/backward 正常达到点 50，但未检测 PES minimum | 几何方向已显示每个候选一条反应物样和一条产品样分支，但三条没有完整 minimum；仍需独立核对两个端点化学图、氢转移和产品 R/S |
| 8 | `generate_gaussian_code`：生成高层单点 route | Gaussian 单点 Action/deck 生成 | E2 已覆盖 14 个候选 | 要与 Step 4 的同一候选几何、原子序列、电荷和多重度绑定 |
| 9 | `xyz_to_gjf`：把 TS XYZ 转成高层单点 GJF | 单点 backend 内部写 GJF | E2 已为 14 个发布坐标生成 GJF | 同上；当前依候选名连接，仍缺强一致性门禁 |
| 10 | `Gaussian`：计算高层电子能 | Gaussian 16 单点 Action | E2 完成：14/14 正常结束，排序与发布数据一致 | 无 E3 候选可供同协议复算 |
| 11 | Python：组合自由能并计算 `delta-delta G^‡`、ee | Case1 重放脚本与通用热化学/选择性计算 | E1/E2 核心数值完成：1.8848504 kcal mol-1、两态 ee 91.613% | major/minor 与 R/S 仍继承论文标签；缺完整路径家族、构象简并度和全系综审计 |

因此，Step 8--11 的发布坐标/数据重放较完整，Step 4 只有固定坐标 Hessian 的 E2 部分证据，Step 7
尚未闭环；Step 1--4 的 E3 独立发现链仍没有完成。不能把后半段的 E1/E2 成功表述为从原始反应
独立找回论文 TS 的 E3 成功。

## 当前总状态

| 阶段 | 证据等级 | 状态 | 当前结论 |
|---|---|---|---|
| 0. 协议与产物盘点 | 全部 | 已完成 | 现有原始产物可复用；本轮重型 TS/NEB/IRC 探索均单独放在 `.runtime/`，不改缓存 |
| 1. 反应输入规范化 | E3 | 部分完成 | 原文产品不守恒；守恒的 R/S 产品分支可映射，但只是本地修正且未包含催化剂 |
| 2. 映射、构象、复合物和 TS 初猜 | E3 | 已完成一次受控尝试，未通过科学门禁 | 固定 MMFF94s 后端点准备完成；R/S 目标 CIP 通过独立过滤选择；两条通用 xTB/NEB 均达到 100 周期上限且无内部势垒，不能产出 Gaussian TS 初猜；预生成烯胺示例仍不是论文 C.5 从头反应 |
| 3. TS 优化和频率 | E2/E3 | E2 已完成，E3 未完成 | TS-2a/TS-2b-1 从发布几何重优化均收敛且各有一个虚频；这仍是 E2，不能冒充从反应输入发现 |
| 4. 双向 IRC 与端点 | E2/E3 | E2 已取得有限续跑证据，端点闭环仍不完整 | 原四分支先在 29 点后超时；checkpoint 副本续跑后，TS-2a backward 在点 31 得到 PES minimum，另外三条到点 50 仍未检测 minimum。每个候选的反应核心距离呈现一条反应物样和一条产品样方向，但不能据此替代化学图/R/S 门禁 |
| 5. 高层单点 | E2 | 已完成 | 14/14 Gaussian 16 C.01 正常结束，排序与发布数据一致 |
| 6. 热化学、选择性和论文对比 | E1/E2 | 核心数值已完成 | `delta-delta G=1.8848504 kcal mol-1`，两态模型 ee=91.613% |
| 7. 从头独立发现结论 | E3 | 未完成 | 受输入歧义、目标 CIP 未由端点 Action 强制、端点未在同一 xTB 势能面最小化、本轮 NEB 无内部势垒以及 IRC 未闭环限制 |

## Case1 十一步测试交接结论

本节是转入 Case2 前的冻结快照。它按 SI A.3.4 的原始顺序列出“实际做了什么”和“能得出什么结论”；
详细证据、参数、哈希和不足见后续各节。

| SI 步骤 | 实际测试结果 | 结论 |
|---:|---|---|
| 1. TSPipeline/TS 初猜 | 原文字面反应因元素不守恒被拒绝；本地守恒 R/S 分支完成映射、构象、MMFF94s 端点，并分别运行 100 周期 GFN2-xTB/NEB；两条 NEB 未收敛且没有内部势垒 | 原子 Actions 可执行输入准备，但本轮没有生成可用的 E3 TS guess；不能声称从头复现论文 TS |
| 2. 生成 TS 优化/频率 route | Gaussian typed Action 可生成论文 B3LYP/6-31G(d)、IEFPCM、`Opt=TS`/Freq 参数 | route 生成能力具备；本轮 E3 因 Step 1 门禁失败而未提交新 guess |
| 3. TS XYZ→GJF | Gaussian 后端成功为发布几何生成并运行 GJF；Hessian 结构与 fchk 坐标帧契约已修复 | 软件调用和输入生成可用；E3 仍无合格 XYZ |
| 4. TS 优化与频率 | TS-2a、TS-2b-1 从发布几何重优化均收敛，能量为 -1130.54000821/-1130.53428598 Eh；各有 1 个虚频，且六项反应核心投影方向全部通过 | 两个发布候选通过 E2 一阶鞍点与目标模式门禁；这不是 E3 独立发现 |
| 5. 生成 IRC route | 两个候选均生成 forward/backward 的 B3LYP/6-31G(d)、IEFPCM、CalcFC/LQA/Recalc=5 请求 | 双向原子调用能力存在；Action 尚未暴露 restart 参数 |
| 6. IRC XYZ/checkpoint→GJF | 四个初始 GJF 和 checkpoint 成功生成；原 Action 运行到 29 点后超时，checkpoint 可恢复 | 基础输入和 checkpoint 可用；长任务恢复仍需管理员操作 |
| 7. 双向 IRC | 四个 checkpoint 副本均续跑结束：TS-2a backward 在点 31 检测 minimum；其余三条到点 50 未检测 minimum。两个候选都各有一条反应物样和一条产品样几何方向 | 反应方向与目标机理一致，但三条未形成完整 minimum，且缺标准端点图/R/S，IRC 科学闭环仍不完整 |
| 8. 生成高层单点 route | 14 个发布候选均生成 M06-2X/def2-TZVPP、IEFPCM(acetone)、UltraFine、TightSCF 输入 | 高层单点输入能力具备 |
| 9. TS XYZ→高层 GJF | 14 个候选逐一绑定发布坐标、电荷、多重度并生成 GJF | 该层输入转换完整，但候选身份仍依赖发布名称 |
| 10. 高层 Gaussian 单点 | 14/14 正常结束；相对排序与发布 JSON 一致 | 发布坐标的 E2 电子能排序复现成功 |
| 11. 自由能、能差和 ee | 组合自由能得到 TS-2b-1 - TS-2a = 1.8848504 kcal mol⁻¹；303.15 K 两态 ee=91.613%，与论文约 1.9 和实验 88% 同趋势同量级 | E1/E2 数值结论复现成功；R 主产物标签和完整构象系综仍继承论文，未被工具独立证明 |

冻结判断：工具箱已能重放 Case1 的发布数据和关键发布几何计算，能支持 TS 优化、频率、单点和 IRC
续跑；但由于题面不守恒、催化物种定义不唯一、端点立体约束不足和本轮 E3 NEB 失败，尚不能表述为
“从原始反应输入独立发现并完整验证 Case1”。

## 步骤 0：协议与现有产物盘点

### 目的

冻结论文协议、源文件哈希和代码提交，确认哪些计算已有可审计原始日志，避免重复运行昂贵任务。

### 实现方式

- 核对正文 Case Study 1、SI A.3.4、A.4.2、B.1.1、B.2.1、C.5、C.8.1 和 C.9.1。
- 检查 `scripts/tasks/case1_reproduce.py`、MiniChem Gaussian/TS/IRC Actions 和 23_TSPipeline 原子链。
- 盘点 `.runtime/case1_direct_20260813/` 的 GJF、Gaussian log、checkpoint 和 ActionResult。
- 检查进程表，确认没有遗留 Gaussian、UMA、ASH 或 Multiwfn 任务。

### 结果

- 14 个 M06-2X/def2-TZVPP/IEFPCM(acetone) 固定几何单点均保留 GJF、log 和 chk。
- 14 个 Hessian 与投影振动 ActionResult 均存在。
- TS-2a 与 TS-2b-1 的四个 IRC ActionResult 都曾达到 29 个 checkpoint 路径点后 `worker_timeout`；checkpoint
  被保留并证明可以用 Gaussian 原生 `IRC=Restart` 续跑，Action 层目前没有 restart/resume 参数。
- 既有预生成烯胺示例的原子准备链与组合 Action 产物字节等价；该示例生成 4 对端点，但没有运行
  UMA/NEB，也没有验证 TS。下文新增的两个无催化剂立体分支则各生成 1 对端点，不能与该示例混算。
- `.runtime/` 由 `.gitignore` 排除，模型缓存和软件缓存没有修改。

### 当前代码不足

- 复现脚本覆盖发布数据、Hessian、TS 重优化、IRC、单点和灵敏度，但没有把论文 11 步整合成一个带检查点的
  E3 runner；这次不把整条论文流程再封装成一个 Action。
- 旧汇总曾使用 `all_candidates_are_first_order_saddles` 描述“固定几何下一个虚频”。本轮 log 有低
  梯度证据，但没有执行 TS 重优化，仍不足以把全部候选称为本轮独立确认的一阶鞍点。当前脚本使用
  较准确的 `all_candidates_have_one_imaginary_mode`，本文也按后者解释。

## 步骤 1：规范化反应输入

### 目的

在任何原子映射和 TS 搜索之前保证元素守恒、明确化学对象、立体分支、总电荷和多重度。输入不唯一时
保留分支，不把本地修正伪装成作者原始输入。

### 论文原始输入的问题

| 项目 | 论文/SI 内容 | 问题 |
|---|---|---|
| 醛 | `O=Cc1ccc(cc1)[N+](=O)[O-]` | 对硝基苯甲醛，可解析 |
| 丙酮 | `CC(=O)C` | 可解析 |
| TfOH | `C(F)(F)(F)S(=O)(=O)O` | 可解析；是否作为显式 TS 组分未说明 |
| C.5/C.9 二胺 | `N1CCC[C@H]1C(N2CCCC2)` | 与 C.8 使用的分子不同 |
| C.8 二胺 | `N1CCCC1[C@H](N2CCCC2)C` | 比 C.5 版本多 `CH2`，不是等价 SMILES |
| C.5 产物 | `CC(=O)C(O)c1ccc([N+](=O)[O-])cc1` | 分子式 `C9H9NO4`，比醛+丙酮总和 `C10H11NO4` 少 `CH2`；也没有 R/S 标记 |

### 复现分支

| 分支 | 定义 | 用途 | 允许得出的结论 |
|---|---|---|---|
| P：原文字面输入 | 完全保留 C.5 SMILES | 验证输入质量门禁 | 应在元素守恒/映射前失败；不能进入 TS 搜索 |
| C：化学守恒修正版 | 产品补上缺失的 `CH2`，分别建立 RDKit 实际 CIP 为 R 和 S 的两个分支 | E3 独立搜索 | 明确标记为本地假设；不能声称是作者唯一输入 |
| E：预生成烯胺示例 | 使用 `examples/arche_23_ts_pipeline_case1.json` 的 +1 烯胺/醛体系 | 测试原子 Actions 和 UMA/ASH/NEB 能力 | 只能验证 TS guess 子链；不是从 C.5 原料出发 |

### 本轮输入质量门禁结果

- 原文四组分字面输入因反应两侧元素不守恒被映射 Action 拒绝；催化剂和 TfOH 只出现在反应物侧，
  也不能作为一条逐原子守恒的单步反应处理。
- 即使只保留醛和丙酮，原文产品仍为 `C9H9NO4`，而反应物总式为 `C10H11NO4`，映射因少一个
  `CH2` 被拒绝。
- 守恒修正版都可映射。显式指定氢由 map 4 转移到 map 6 后，两个分支均识别出 map 4--5 新成
  C--C 键、map 5--6 由 C=O 变为 C--O，以及 H map 16 从 map 4 转移到 map 6；氢计数守恒。

SMILES 的 `@`/`@@` 只表示相对于当前邻接原子顺序的局部手性，不能脱离具体 SMILES 直接当作 R/S。
本轮用 RDKit 重新指派 CIP 后，实际对应如下。两个文件名是早先按 `@`/`@@` 直觉命名的历史路径，
与实际 CIP 正好相反，后续引用必须以“实际 CIP”列为准。

| 实际 CIP | 守恒产品 SMILES | 映射置信度 | 本轮请求/结果文件 |
|---|---|---:|---|
| S | `CC(=O)C[C@H](O)c1ccc([N+](=O)[O-])cc1` | 0.5836500586 | `corrected_R_request.json` / `corrected_R_atomic.json`（历史误名） |
| R | `CC(=O)C[C@@H](O)c1ccc([N+](=O)[O-])cc1` | 0.6252985635 | `corrected_S_request.json` / `corrected_S_atomic.json`（历史误名） |

`analyze_reaction_changes` 只核对映射后的键和氢变化，不负责重新指派或验证 R/S。因此 CIP 标签必须作为
独立元数据保存并在端点验证时再次由立体化学工具核对，不能由候选能量或文件名反推。

修复后的 `stereo_enamine_mmff94s_fixed/R_result.json` 与 `S_result.json` 每个 pair 的坐标逐元素完全相同，
且两份请求各自仍包含 5 个实际 S 和 3 个实际 R。这是端点 Action 的通用能力边界：输入 SMILES 的手性
没有被传播为候选筛选约束。因而本轮只把 CIP 作为外部科学门禁过滤，不能声称端点 Action 已独立生成两条
对映路径。精确 MMFF94s 的审计表和选择记录在 `.runtime/case1_stepwise_20260816/stereo_enamine/`
及 `stereo_enamine_mmff94s_fixed/`；端点能量只用于排除异常构型，不是势垒或选择性证据。

精确 MMFF94s 固定坐标能量（kcal mol⁻¹）给出的筛选依据如下：

| 实际 CIP | pair | 反应物 | 产品 | 选择理由 |
|---|---|---:|---:|---|
| R | `endpoint_pair_0007` | 110.8059 | 103.1446 | R 集合中反应物+产品总和最低且反应核心距离合理 |
| S | `endpoint_pair_0006` | 110.9653 | 107.2783 | S 集合中反应物+产品总和最低且反应核心距离合理 |

这只是为路径搜索选择数值正常的起点；它没有给出过渡态能垒，也不能推出论文的 major/minor 归属。

进入 Gaussian 前还必须把每个候选的 charge、multiplicity、atom map、CIP 标签和候选 ID 固化到
结构产物；目前无催化剂 C 分支尚未形成论文所需的催化 TS 候选。

## 步骤 2：生成构象、反应复合物和 TS 初猜

### 目的

在巨大三维构象空间中生成两条立体路径的合理起始结构。TS guess 只是搜索起点，不是经验证过渡态。

### 实现方式

按原子 Action 组织：

`map_reaction_atoms -> materialize_mapped_hydrogen_transfers -> analyze_reaction_changes ->`
`generate_conformer_ensemble -> build_reactive_complex_ensemble -> optimize_geometry(MMFF) ->`
`apply_reaction_changes_to_structure -> optional UMA endpoint refinement -> optional ASH/KNARR NEB`

### 已有结果

本轮先对 C 分支的两个实际 CIP 对映体运行轻量原子链。两条链都没有催化剂，且明确设置
`run_uma=false`、`run_neb=false`；它们只检查映射、反应变化、构象和端点构建。随后用修复后的
通用 `MMFF94s` Action 重新生成端点，并对实际三维 CIP 做独立审计，再把最低能量且化学距离合理的
R=`endpoint_pair_0007`、S=`endpoint_pair_0006` 交给 `search_reaction_path`（GFN2-xTB/ALPB acetone，
7 images、IDPP、FIRE、climbing NEB）。这一步仍是 E3 的 TS guess 测试，不是 Gaussian TS 证明。

| 实际 CIP | 结果路径 | 总状态 | 原子阶段 | 端点对 | 准备完成 | TS 已验证 |
|---|---|---|---:|---:|---|---|
| S（文件历史名 R） | `.runtime/case1_stepwise_20260816/corrected_R_atomic.json` | `partial_success` | 23（1 partial + 22 success） | 1 | `true` | `false` |
| R（文件历史名 S） | `.runtime/case1_stepwise_20260816/corrected_S_atomic.json` | `partial_success` | 23（1 partial + 22 success） | 1 | `true` | `false` |

首个 `map_reaction_atoms/rxnmapper` 阶段因初始隐式氢处理返回 `partial_success`；显式物化氢转移后，其余
阶段成功。结果哈希如下，便于在 Git 忽略的运行目录中核对原始产物：

| 文件 | SHA256 |
|---|---|
| `.runtime/case1_stepwise_20260816/corrected_R_request.json` | `a2a4693179555042dede1e6ea95708b9feec2687bffe103ad12d72b6bac81913` |
| `.runtime/case1_stepwise_20260816/corrected_S_request.json` | `a6cd49b45ad02eeaffdb12ddbbe187d57dd1e6b32cad657f1a674d540d602c99` |
| `.runtime/case1_stepwise_20260816/corrected_R_atomic.json` | `587d6f9bf121ee583298e3b6a7f078966f7438194811b578a3e671be04dedd65` |
| `.runtime/case1_stepwise_20260816/corrected_S_atomic.json` | `cc700d380eacc853c31ddfbf929feb1fec67b79420d4f252006b152d007cc602` |

另一个既有分支 E（预生成烯胺示例）完成过不含 UMA/NEB 的准备链：36 个原子阶段中 35
`success`、1 `partial_success`，mapping confidence 为 0.674886，形成 4 对端点，
`preparation_complete=true`，并明确记录 `transition_states_validated=false`。这条结果与上表的两个
无催化剂分支是不同输入，不能合并计数或互相补足科学证据。

### xTB/NEB 实测结果

两个筛选后的端点对均已实际提交给通用 `search_reaction_path/pysisyphus`，协议为
GFN2-xTB/ALPB(acetone)、7 images、IDPP、FIRE、climbing NEB、最多 100 周期。几何 validator 对
原子映射、端点、逐图连续性和指定成/断键进度均通过，但原生优化器均未收敛：

| 实际 CIP | 端点 pair | 100 周期末 Max force/au | RMS force/au | 最高内部图相对反应物/kcal mol⁻¹ | 结论 |
|---|---|---:|---:|---:|---|
| R | `endpoint_pair_0007` | 0.004813 | 0.000588 | -7.470 | 未收敛；全路径从反应物端下坡，无内部势垒 |
| S | `endpoint_pair_0006` | 0.010013 | 0.000940 | -7.021 | 未收敛；全路径从反应物端下坡，无内部势垒 |

两个分支的最高总能量图都固定在反应物端点，pysisyphus 因而反复报告
`HEI is first or last image`，climbing image 无法激活；最终均为 `Number of cycles exceeded`。
因此，即使几何连续性检查通过，也没有科学上可用的内部最高图像，不能把任意图像提交给 Gaussian
冒充 TS guess。本轮 fail-closed，没有继续 E3 的 Gaussian TS 优化。

本次运行还暴露并修复了一个通用解析问题：pysisyphus 1.0.0 的 `final_geometries.trj` 可能不在内部图像
注释中写能量，而完整最终周期能量仍保存在 `optimization.h5/opt/energies`。后端现在仅在轨迹能量不完整时
严格读取 HDF5 的最后一周期，检查图像数和有限数值，并记录
`energy_source=optimization_h5_final_cycle`；它不会把 `converged=false` 改成成功。

审计汇总见
`.runtime/case1_stepwise_20260816/pysis_mmff94s_recovery_report.json`（SHA256
`37d6d1da3250e9658e20d11d9da85cdddb90abe16a00e364ba1d0b49abf7c3d5`），原生 run ID 为
R `3f0d5419a371`、S `64641f177e81`。

### 当前不足

- 分支 E 预先给定烯胺、产品新生立体中心和一个氢转移，含较强机理先验。更关键的是，示例烯胺中
  承载不对称诱导的催化剂手性没有用 `@`/`@@` 指定；即使其新生产品中心写了 `[C@@H]`，也不能用
  这个未定义催化剂构型的示例验证论文的催化不对称诱导、R/S 偏好或 ee。
- C 分支还没有唯一的“丙酮+二胺+TfOH如何形成实际 +1 TS 模型”状态定义；直接把四个中性组分映射成
  最终产品会把催化剂周转、脱水/水解和质子平衡混入一个反应式。
- 本轮 MMFF94s 端点没有先分别在与 NEB 相同的 GFN2-xTB/ALPB 势能面收敛到驻点，端点能量在路径中
  最高，导致 climbing image 无法建立；再次搜索前应先增加通用“同层级端点最小化并验证收敛”的门禁。
- UMA/ASH NEB 只产生 TS guess；只有端点全部 `converged=true` 才允许进入 NEB，后续仍必须做
  Gaussian TS 优化、Hessian 和双向 IRC。

## 步骤 3：Gaussian TS 优化与频率

### 目的

将 TS guess 优化到候选驻点，并检查反应方向上是否恰好有一个负曲率模式。

### 论文方法

```text
#p B3LYP/6-31G(d) opt=(ts,noeigentest,calcall,maxstep=5) freq
   scrf=(iefpcm,solvent=acetone) temperature=303.15
```

### 当前实现与结果

MiniChem 提供通用 `locate_transition_state/gaussian` 和 `calculate_hessian/gaussian`。现有 E2 运行在
14 个发布最终坐标上重新计算 Hessian，并在质量加权坐标中投影 3 个平移和 3 个转动自由度：53 原子
的 159 个 Cartesian 自由度得到 153 个振动模式，14/14 各有一个虚频。

本轮另修复了一个与 Case1 无关的通用坐标系契约：Gaussian 可能将输入整体旋转/平移到标准取向，
而 fchk 的 `Cartesian Force Constants` 与 `Current cartesian coordinates` 都位于这个 checkpoint
坐标系。`calculate_hessian/gaussian` 现在从同一 fchk 读取原子序号和 Bohr 坐标、显式换算为 Angstrom，
并在 `result.structure` 中保留输入的电荷、多重度、键图、原子映射和逐原子身份元数据；Hessian 本身
不做反向旋转。Case1 runner 随后将该 `result.structure`（而非原始输入坐标）交给
`derive_vibrational_modes`。因此，在该修复之前生成的 MiniChem Cartesian 模式方向不应直接用于
反应核心判定，需用保存的 fchk 按新契约重新派生；Gaussian 原生 log 中的频率证据不受此接口问题影响。

14 个固定坐标 Freq log 均正常结束并打印了梯度；Maximum Force 约为 `2e-6`--`1e-5`
Hartree/Bohr，均低于对应 Gaussian 收敛阈值。这是“发布坐标附近梯度较低”的实算证据。与此同时：

- 另从 TS-2a 和 TS-2b-1 的发布最终几何重新执行了通用 `locate_transition_state`：两条
  `Opt=(TS,NoEigenTest,CalcAll,MaxStep=5)` 均正常收敛；随后用同一 Gaussian 结果的 fchk 坐标系
  计算 Hessian 和投影振动，53 原子均得到 153 个模式且恰有一个虚频。
- 当前已确认的数值为 TS-2a `-258.1147 cm⁻¹`（MiniChem 投影 `258.004852i`）和 TS-2b-1
  `-241.2542 cm⁻¹`（`241.154565i`）；Gaussian 总能量分别为 `-1130.54000821` 和
  `-1130.53428598 Eh`。两个同帧虚频模式对目标 C--C 成键、O--H 成键、N--H 断键、C=O 伸长、
  烯胺 C=C 伸长和 C--N 缩短的六项方向投影全部通过。
- 汇总报告为 `.runtime/case1_stepwise_20260816/ts_frame_recovery_report.json`（SHA256
  `ad83ddecf324a781621290c78694ebf58eb8c8e7e2738a4ba13d6ad7f4293f97`）；Gaussian 原生日志分别在
  `ts_frame_workspace/outputs/locate_transition_state/gaussian/d230cb211f31/`（TS-2a）和
  `d592b79034a7/`（TS-2b-1）。旧 `ts_reoptimization_report.json` 的 Gaussian 频率数值仍可用，
  但其中笛卡尔模式生成于 Hessian 坐标系修复前，模式方向应以新的 recovery report 为准。
- 这仍是从发布几何出发的 E2 重算；不能把它写成从反应输入独立找到 TS，也不能仅凭一个虚频替代
  双向 IRC 端点门禁。

该结论属于 E2；它不证明这些坐标是本轮从反应输入独立发现的 E3 结果。

## 步骤 4：双向 IRC 和端点验证

### 目的

证明候选 TS 的两个下坡方向分别连接目标反应物侧和产品侧，而不是无关构象或错误反应。

### 论文方法

```text
#p B3LYP/6-31G(d) irc=(calcfc,lqa,maxpoints=50,recalc=5)
   scrf=(iefpcm,solvent=acetone)
```

### 当前实现与结果

`trace_intrinsic_reaction_coordinate/gaussian` 每次只执行一个方向，forward/backward 分开调用。
当前 TS-2a 与 TS-2b-1 的 4 个分支均在每个方向 7200 秒后 `worker_timeout`；Gaussian checkpoint
并非空白，而是每个分支已经保存 29 个路径点。随后把 checkpoint 复制到
`.runtime/case1_stepwise_20260816/irc_restart_to_50/`，设置 Gaussian 运行所需的
`GAUSS_EXEDIR`、`g16root` 和独立 `GAUSS_SCRDIR`，用原生 `IRC=Restart` 续跑到最多第 50 点。
四个续跑日志的 SHA256 分别为：

| 候选/方向 | 续跑解析点数 | 最终状态 | 最后一点的反应核心距离（Å） | 日志 SHA256 |
|---|---:|---|---|---|
| TS-2a / backward | 2（30--31） | 第 31 点检测 `PES minimum`，正常结束 | C3--C38 3.0331；O39--H11 1.7996；N10--H11 1.0424 | `cef666b1fbc08ee65acfc7bdadd3928a9ba1a3fb47ad18d075422aa796a8558b` |
| TS-2a / forward | 21（30--50） | 正常结束，达到 `MaxPoints=50`，未检测 minimum | C3--C38 1.5866；O39--H11 0.9919；N10--H11 1.9570 | `270c4b5c95f58c1b1589649080c46f52860185ba739dd28add86952b0f209a63` |
| TS-2b-1 / backward | 21（30--50） | 正常结束，达到 `MaxPoints=50`，未检测 minimum | C3--C38 1.5894；O39--H11 0.9903；N10--H11 2.0227 | `d699bb8ecaf2045dddb49a9de4cf11ae6d3b30459bb4a5e4c5ff45ae0ebb4415` |
| TS-2b-1 / forward | 21（30--50） | 正常结束，达到 `MaxPoints=50`，未检测 minimum | C3--C38 3.0630；O39--H11 1.8815；N10--H11 1.0359 | `411e58b74952f7a8dba6a9d8a2c6cf0fe06dba0546fe5515bf4e559174390aaf` |

“解析点数”只包括续跑日志中的第 30 点以后；原始 checkpoint 已包含前 29 点。TS-2a forward 和
TS-2b-1 backward 的成键/断键距离进入产品样区间，另两个分支进入反应物样区间。由于发布材料没有
标准反应物/产物坐标、机器可读原子映射和 R/S 标签，不能把“产品样距离”升级为“端点属于目标化学图
且为 R/S”的证明。

因此，IRC 的结论是“checkpoint 恢复和局部反应方向可用，完整 E2 端点门禁仍未通过”：1 条分支有
PES minimum，3 条只到最大点数；四条均正常退出，没有错误终止。

续跑过程中第一次直接调用 `g16` 曾因未设置 Gaussian 的 `GAUSS_EXEDIR`/`g16root` 而在启动阶段
段错误；该次没有读取或修改 checkpoint。随后按 MiniChem 后端的环境契约补齐 `GAUSS_EXEDIR`、
`g16root` 和每分支独立 `GAUSS_SCRDIR` 后，四条任务均正常进入 IRC 并完成上述结果。失败启动的临时
文件已移入 Git 忽略的 `.runtime/case1_stepwise_20260816/failed_startup_artifacts/`。

### 当前不足

- 论文发布 JSON 没有 IRC 轨迹或标准反应物/产品端点。
- 当前 Case1 runner 只汇总每个分支的收敛状态，还没有自动把结果串到通用
  `validate_reaction_path` 做原子身份、连续性、RMSD 和键变化进度检查。
- 即使接入 `validate_reaction_path`，该 Action 也只能检查原子身份、轨迹 RMSD/连续性和指定键距的
  变化趋势；它不能独立判断两个末端是否分别属于目标反应物/产物化学图，不能完整核对成键、断键和
  显式氢转移，也不负责 RDKit CIP/R/S 指派。因此 `converged=true` 或 validator 通过都不能单独充当
  Case1 的端点化学身份与立体化学证明。
- IRC 是当前 E2 科学闭环的主要未完成项。checkpoint/分支级恢复已在本轮验证可用，但 Action 层仍没有
  checkpoint/restart 参数；目前只能由运行管理员在副本目录中手动续跑，不能由 Planner 直接调度恢复。
- 这是当前 Action 的客观缺口：后端超时只返回 `worker_timeout`，没有把 checkpoint 路径和 restart
  参数暴露给调用者；本轮先用原生 Gaussian 续跑能力验证可行性，没有未经审查地修改该核心接口。

## 步骤 5：高层固定几何单点

### 目的

用更适合能量排序的方法，在低层几何上重新评估电子能量。

### 论文方法

```text
#p M062X/def2TZVPP sp scrf=(iefpcm,solvent=acetone) int=ultrafine scf=tight
```

### 当前结果

- 14/14 使用 Gaussian 16 C.01 正常结束。
- TS-2a 仍为最低候选。
- 本轮单点与发布单点的最大绝对偏差为 0.031179 kcal mol-1；该偏差对所有候选近似相同，
  相对排序最大偏差仅约 `5.18e-7 kcal mol-1`。
- 原始 GJF、log 和 chk 位于 `.runtime/case1_direct_20260813/gaussian_exact*`。

## 步骤 6：自由能、选择性和论文对比

### 目的

把低层频率热修正与高层电子能组合，计算竞争路径自由能差，再用过渡态理论得到两态选择性。

### 计算定义

```text
G_composite = E_high-level_SP + (G_low-level_freq - E_low-level_SCF)
ratio_major/minor = exp(delta-delta G / RT)
ee = (ratio - 1) / (ratio + 1) * 100%
```

### 当前结果

| 指标 | 当前复算 | 论文/实验 | 判断 |
|---|---:|---:|---|
| 最低候选 | TS-2a | TS-2a major | 一致 |
| TS-2b-1 - TS-2a | 1.8848504 kcal mol-1 | 约 1.9 | 一致 |
| 两态预测 ee，303.15 K | 91.613% | 实验 88% | 趋势和量级一致 |
| 实验 88% 对应能差 | 1.6576 kcal mol-1 | 论文约 1.6 | 一致 |

这里的 ee 依赖调用者把 TS-2a/TS-2b-1 声明为两个穷尽的相反对映路径。发布 JSON 没有机器可读
R/S 标签，也没有 14 个候选的完整家族归属，所以当前工具不能独立证明“R 是主要产物”或完成全构象
Boltzmann 系综；R 方向仍继承论文标签。

## 已确认的源材料矛盾

| 项目 | 论文/SI | 发布 JSON/当前复算 | 影响 |
|---|---|---|---|
| 候选数量 | 12 个 verified TS | 14 条 opt/SP 记录 | 缺少 14 到 12 的作者映射，不能自行删两条 |
| TS-2c 相对能量 | Figure S2 为 5.15 kcal mol-1 | JSON 为 1.933984 kcal mol-1 | 两个发布来源冲突，需作者说明 |
| IRC | 论文称候选通过 IRC | JSON 没有 IRC 记录 | 无法从发布数据重放连接证明 |
| R/S 标签 | 正文将 major 归为 R | JSON 无机器可读标签和端点 | 当前数值只能复现能量偏好，不能独立归属构型 |

## 当前代码和工具箱不足汇总

| 严重度 | 范围 | 问题 | 是否阻止当前结论 |
|---|---|---|---|
| 高 | 论文输入 | 产品不守恒、催化剂字符串漂移、缺电荷/多重度/原子映射 | 阻止 E3 从原文直接开始 |
| 高 | TS 发现 | 现有 Case1 示例是预生成烯胺子反应，且催化剂手性未指定；端点 Action 对 R/S 请求仍输出同一混合 CIP 集合，不是四组分原始题面 | 阻止把示例/端点结果称为从头发现或不对称诱导验证 |
| 高 | E3 路径搜索 | MMFF94s 端点未先在同一 GFN2-xTB 势能面优化；两条 100 周期 NEB 均未收敛，最高点固定在反应物端且没有内部势垒 | 当前没有可用 E3 TS guess，阻止 NEB→Gaussian 闭环 |
| 高 | IRC | 四分支已从 29 点 checkpoint 副本续跑：1 条检测 PES minimum，3 条在 50 点未检测 minimum；Action 未暴露 checkpoint restart；`validate_reaction_path` 也不能独立验证端点化学图、完整反应核心和 R/S | 已支持“反应物样/产品样方向”判断，但仍阻止完整连接性和 R/S 独立验证 |
| 中 | TS 门禁 | 关键两候选已完成 E2 TS 重优化、Hessian 和单虚频；E3 仍没有从 NEB/原始反应输入独立通过同一门禁 | 不影响发布数据数值重放，影响“独立验证 TS”表述 |
| 中 | 坐标系契约 | 旧 Hessian Action 返回的矩阵可能与调用者原始取向不一致；本轮已改为返回 fchk 同帧结构并加回归测试 | 修复后不阻止；旧运行的 Cartesian 模态需重新派生 |
| 低 | MMFF 变体 | `mmff94s` 曾小写传给 RDKit 并静默退回 MMFF94；本轮已做原生名称映射和回归测试 | 修复后不阻止；旧端点需注明实际使用的力场 |
| 低 | pysisyphus 结果解析 | 轨迹注释缺能量时旧后端会返回空能量；已增加对 `optimization.h5` 最终周期的严格回退读取，并保留原生未收敛状态 | 修复后可审计本轮路径能量，不改变 NEB 失败结论 |
| 中 | 候选语义 | JSON 无候选家族、R/S 和构象简并度 | 两态 ee 可算，完整系综 ee 不可审计 |
| 中 | 运行恢复 | IRC 原生 checkpoint 已证明可恢复，但 Action 未提供分支级 checkpoint/resume 聚合 | 增加长任务失败成本，需运行管理员手动复制和续跑 |
| 低 | 报告接口 | E1/E2/E3 过去分散在多个产物，容易混淆证据等级 | 本文通过分层记录缓解 |

## 修改与运行日志

| 日期 | 类型 | 内容 | 缓存影响 |
|---|---|---|---|
| 2026-08-16 | 文档 | 新建本文，冻结协议、分支、门禁和现有证据 | 无 |
| 2026-08-16 | 轻量实测与文档 | 原文输入守恒门禁；两个守恒 CIP 分支的映射、氢转移、反应变化与 MMFF 端点准备；补记 SI 11 步覆盖、产物哈希和 Gaussian/IRC 证据边界 | 未修改模型或软件缓存；运行产物仅在 `.runtime/` |
| 2026-08-16 | 通用代码修复与轻量回归测试 | 修复 Gaussian fchk Hessian/结构坐标系契约：解析 `Atomic numbers` 与 Bohr 单位的 `Current cartesian coordinates`，返回与 Hessian 同帧且继承分子身份的 `result.structure`；Case1 的 Hessian→振动调用改用该结构；用整体旋转的双原子输入验证返回结构及虚频模式均在 fchk 坐标系 | 未修改模型或软件缓存；未运行重型 Gaussian |
| 2026-08-16 | 通用力场修复 | `rdkit_mmff` 将公开的 `mmff94/mmff94s` 映射为 RDKit 原生大小写，避免 `MMFF94s` 静默退化；新增乙酰胺能量/参数转发回归测试 | 未修改模型或软件缓存 |
| 2026-08-16 | E2 重优化与 E3 路径实测 | TS-2a/TS-2b-1 发布几何 `Opt=TS`→Hessian→同帧投影振动；重新生成精确 MMFF94s 端点并以 CIP 过滤后完成 R=`0007`、S=`0006` 的 100 周期 xTB/NEB，确认均未收敛且无内部势垒；原四 IRC 分支的 29 点 checkpoint 做 `IRC=Restart` 探针 | 未修改模型或软件缓存；所有输出在 `.runtime/` |
| 2026-08-16 | 通用结果解析修复 | pysisyphus 轨迹注释缺内部图像能量时，从 `optimization.h5/opt/energies` 严格读取最终周期；校验图像数与有限数值，记录能量来源且不覆盖原生 `converged=false` | 未修改模型或软件缓存；新增单元测试 |
| 2026-08-17 | E2 IRC checkpoint 续跑 | 四条 29 点 checkpoint 均在副本目录继续到终止：TS-2a backward 点 31 检测 PES minimum；其余三条正常到点 50、未检测 minimum；按六项反应核心距离确认每个 TS 各有一条反应物样和一条产品样方向 | 未修改模型或软件缓存；原始 checkpoint 未覆盖；续跑产物仅在 `.runtime/` |

后续每完成一步，就在本节追加实际命令、产物 SHA256、运行状态和必要的代码修改。任何针对 ARCHE
核心 Planner/Execution 逻辑的改动，先记录问题和方案，再按用户确认执行。
