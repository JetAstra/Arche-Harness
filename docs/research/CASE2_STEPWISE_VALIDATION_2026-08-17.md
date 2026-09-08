# ARCHE Case 2 逐步复现验证（2026-08-17 快照）

更新日期：2026-08-17
复现对象：论文中的蓝光驱动 α-碘硼酸酯脱碘/脱硼氢化反应
证据原则：发布数据重放（E1）、固定发布几何重算（E2）和从题面独立发现（E3）分开记录。

本文按照补充材料 C.9.2 的初始/修正 workflow 和 B.1.2、B.2.2 的实际推理链记录当次
MiniChem 可执行性。本文不是当前运行指南；E1/E2 的成功不等同于 ARCHE 从自然语言题面独立发现了论文机制。当前入口见[交付与复现说明](../current/DELIVERY_AND_REPRODUCTION.md)。

> 当前环境说明：本文中带有 `.venv` 或其他旧环境路径的命令、测试结果和依赖状态，均保留
> 2026-08-17 当次复现使用的命令、路径和资源配置均作为历史证据保留，不表示当前配置。

## 执行计划

1. 冻结论文、SI、`case2.json` 和单位/哈希（Step 0）。
2. 规范化物种、离子、电荷、自旋和溶剂近似（Step 1），并把 A–G 假设与 Action 契约对齐（Step 2）。
3. 检查构象/接触复合物、优化、频率、TS、IRC 和 TD-DFT 的通用 Action 覆盖（Steps 3–4、6、9）。
4. 先用发布 JSON 做低成本 E1 数值重放（Steps 5、7–9），再在 `g16` 可用时执行 E2/E3 重算。
5. 用势垒、激发能、振子强度和实验控制统一判定，最后登记数据/工具缺陷和所需资源（Steps 10–11）。

本轮已完成 1–5 的低成本部分、Steps 7–10 的发布数据重放和 Action 回归；随后发现并启用了项目内
Gaussian 缓存，已经实际完成固定发布几何的 E/F/G/CsPPh₂ 单点和 E/F/G 三组 TD-DFT。A–D 的
固定发布 TS 几何 Hessian 已实际调用 Gaussian 16 并由 `formchk` 解析完成；R/S 均得到 189 个频率，
但各有 3 个虚频，未通过一阶鞍点门禁。当前正在用 `locate_transition_state/gaussian` 对 R/S 做
`CalcFC` TS 精修；只有精修结构通过单虚频门禁后才会调用双向 IRC。

## 1. 论文问题和成功标准

反应是 `Ph-CH2-CH2-CHI-BPin + HPPh2` 在 2 eq Cs₂CO₃、EtOAc/CPME、450 nm 蓝光下生成
`Ph-CH2-CH2-CH2-BPin + IPPh2`（83%）。论文明确使用了以下实验约束：需要光、碱和膦；暗反应不发生；
TEMPO 抑制反应。ARCHE 初始生成 25 个策略，收敛为 A–F 六类候选；反思后增加 G（CsPPh₂/底物/Cs⁺
协同复合物）。成功不是“所有 Gaussian 作业结束”，而是：

1. A–D 的热路径因高势垒或找不到目标 TS 被排除；
2. E、F 的激发能分别高于 450 nm 光子能，因而被排除；
3. G 的可见光激发与实验光源匹配，并且与光/碱/膦/TEMPO 对照相容。

450 nm 光子能按 `E(eV)=1239.841984/λ(nm)`、`1 eV=23.06055 kcal/mol` 计算为
`2.7552 eV = 63.58 kcal/mol`（论文四舍五入为 63.6 kcal/mol）。

## 2. 来源、哈希和固定数据

| 来源 | SHA256 |
|---|---|
| `../ARCHE_lidong/manuscript.pdf` | `1712d2e77455b6ecf1e6de1141b764ad60e6c92bb9f2e03c15fff1e6265b845c` |
| `../ARCHE_lidong/Supplementary Information.pdf` | `b2f5aacac2b58b62baaa8b43edff451ddffd2520c319da6c39031e2e86eed70f` |
| `supplementary/case2.json` | `fa4af5aec29ab5d99d4b419216c8672e0062355d476ffe3e0150e5d526577722` |

`case2.json` 包含 `Concerted_R`、`Concerted_S`、`hypothesis_E`、`hypothesis_F`、
`hypothesis_G` 五组对象；每个对象保存 Gaussian/cclib 解析后的几何、能量、频率或激发态数据。

## 3. workflow 与当前 Action 映射

SI 初始 workflow 的 15 步为：RDKit 复合物/构象 → Gaussian 优化 → 收敛解析 → 热化学/结合能 → TD-DFT
→ UV-Vis 数据处理 → 反应物/产物构象 → 优化 → 插值 TS guess → TS 优化 → 虚频检查 → IRC → 端点优化
→ 频率 → ΔG‡。修正 workflow 保留同一层次，但将基组改为 def2-SVP、要求完整收敛门禁、双向 IRC、
CalcFC/LQA 和显式 Gibbs 自由能组合。

| Case2 步骤/目标 | MiniChem Action 或后端 | 本轮状态 |
|---|---|---|
| 物种和电荷/自旋规范化 | `normalize_qcschema_molecule`、`parse_quantum_chemistry_output` | 可用；需人工给离子/自由基设定 charge/multiplicity |
| 构象采样 | `generate_conformer_ensemble` (`rdkit_etkdg`/`crest`) | 已用系统 RDKit 对 `CCBr` 实际生成 3 个 ETKDG 构象；当时项目 `.venv` 仍需按环境文件安装 RDKit |
| 接触/离子对复合物 | `build_reactive_complex_ensemble` | 已用两个实际生成的构象调用，2/2 候选通过接触/碰撞门禁；只生成有界接触构型，不替代量化优化 |
| 几何优化 | `optimize_geometry` (`gaussian`、xTB、MMFF、UMA 等) | Gaussian Action 已用项目内 `g16` 完成 E/F/G/CsPPh₂ 单点；TS 优化待 Hessian 门禁后执行 |
| 基态能量/Hessian/频率 | `calculate_energy`、`calculate_hessian`、`derive_vibrational_modes` | E/F/G/CsPPh₂ 单点已成功；R/S 固定 TS Hessian 均成功并解析 189 个频率，但均有 3 个虚频 |
| TD-DFT 激发态 | `calculate_excited_states` (`gaussian`) | E/F/G 三组实际 `g16` 作业均成功并解析 20 个 singlet 态 |
| TS/虚频/IRC | `locate_transition_state`、`calculate_hessian`、`trace_intrinsic_reaction_coordinate` | 固定坐标 Hessian 已完成；R/S TS 精修正在真实运行；IRC 等待单虚频门禁 |
| Gaussian 输出解析 | `parse_quantum_chemistry_output` (`cclib`) | 可用；发布 JSON 已是解析结果 |
| 自由能组合和路径排序 | 通用 Python/`derive_thermochemistry`、`analyze_free_energy_selectivity` | 可用；本轮只做通用数值审计，不新增 Case 特制 Action |

## 4. 分步复现结果

### Step 0：冻结协议和证据等级

**目标。** 固定论文/SI、机器可读数据、单位换算和实验光子能，避免把发布结果误报为新计算。

**实现。** 对三份来源计算 SHA256；检查 JSON 的组名、对象名、`metadata.keywords`、`charge`、`mult`、
`freeenergy`、`etenergies` 和 `etoscs`。运行结果只写在 `.runtime/`（若启动新任务），不动模型/软件缓存。

**结果。** 来源和哈希见第 2 节；JSON 记录的 Gaussian 版本为 2016+B.01，论文正文中的方法是
B3LYP-D3(BJ)/SMD，激发态使用 def2-SVP。未发现 Case2 专用的独立 ARCHE 对话、失败日志或完整历史
planner 消息，因此无法复原“同一次”智能体轨迹。

**不足。** 发布 JSON 没有统一的 `unit` 字段；`scfenergy_final`/`freeenergy` 的数值按 Hartree
解释才与论文 kcal/mol 结果一致。这是数据契约不足，后续应在通用 cclib 导出格式中显式保存单位。

### Step 1：规范化底物、产物和反应组分

**目标。** 明确 α-碘硼酸酯、HPPh₂、Cs₂CO₃/CsPPh₂、I⁻/CsI 的元素组成、总电荷、自旋和溶剂模型。

**实现。** 将中性闭壳层底物/HPPh₂、自由基（multiplicity=2）、PPh₂⁻（charge=-1, singlet）、
CsI 和 CsPPh₂ 分开建模；TD-DFT 只对闭壳层 singlet 参考态调用。论文/JSON 中对应对象的 charge/mult
已审计，且 E/F/G 的对象电荷自洽于其各自假设。

**结果。** 物种层面通过；实验的 EtOAc/CPME 混合溶剂在 Gaussian Action 中只能以一个显式 SMD 溶剂
（发布记录为 `EthylEthanoate`）表示，不能声称已经模拟了真实二元溶剂。

**缺陷/影响。** SI 初始 workflow 曾把离子对“一律设为 neutral singlet”，这会改变电荷转移和激发能，
属于科学风险；修正 workflow 虽要求保留离子相互作用，但仍未规定统一的 counterion/聚集态采样协议。

### Step 2：固定 A–F 初始假设和 G 反思假设

**目标。** 防止只计算最终 G 而漏掉“淘汰候选”的证据链。

**实现。** A–D 作为闭壳层/替代 TS 路径；E 为孤立底物直接激发；F 为底物–膦关联体系；G 为
CsPPh₂/底物协同复合物。每类只需要能回答“势垒是否可及”或“激发能是否匹配”，不需要完整中间体网络。

**结果。** 假设定义可由 SI B.2.2、C.8.2 和 `case2.json` 对象对应；G 不是预先写入的唯一机制，而是
在 A–F 失败后由反思步骤提出。

**缺陷/影响。** 论文公开的是整理后的六类/一类假设，不包含全部 25 个原始策略的完整对象和每次失败输入，
所以只能复现候选族的科学门禁，不能复演全部 planner 采样。

### Step 3：构象和接触复合物准备

**目标。** 为热路径 TS 和光路径激发态提供多个底物/膦/ Cs⁺ 接触几何。

**实现。** 使用 `generate_conformer_ensemble`、`build_reactive_complex_ensemble`，并将 atom mapping、
charge/multiplicity 和候选 ID 随结构保存；量化优化应使用 `optimize_geometry` 后再计算能量/TD-DFT。

**结果。** 本轮使用系统 RDKit 对 `CCBr` 实际生成 3 个 ETKDG 构象，并用其中两个构象调用
`build_reactive_complex_ensemble`；结果为 `status=success`、`accepted_count=2`、`rejected_count=0`。
这些构型只是后续 Gaussian 优化的输入候选，没有把 MMFF/几何接触当作论文复合物证据。结果保存在
`.runtime/case2_structure_smoke/`。

**缺陷/影响。** 当前 MiniChem 只保证“生成接触构型”，不保证它是最低自由能或最具反应性的离子对；若只
保留单一最低 MMFF 构型，会复现 SI 初始 workflow 的已知偏差，不能作为 G 的充分证明。

### Step 4：基态优化、频率和 minimum 门禁

**目标。** 对候选复合物和独立物种确认优化收敛、零虚频，并获得统一热化学数据。

**实现。** Gaussian typed Action 生成 B3LYP-D3(BJ)/def2-SVP、SMD、UltraFine、`Opt=CalcFC`/`Freq`；
`parse_quantum_chemistry_output`/cclib 解析收敛和振动频率。最低要求是正常终止、梯度/位移收敛、
minimum 为 0 个虚频；自由基必须保留 unrestricted multiplicity。

**结果。** 发布 JSON 中相关优化对象均有 `metadata.success=true` 和频率数据，说明发布计算记录可读；
本轮用项目内 Gaussian 缓存对固定发布几何实际提交了单点：E `−1057.61997475 Eh`、F
`−1862.02403523 Eh`、G `−1882.15808953 Eh`、CsPPh₂ `−824.510995437 Eh`，均为
Gaussian 16 C.01 `success`。这些是电子能量单点，不能直接替代含频率的 Gibbs 自由能；完整的
R/S 固定 TS 几何的 `calculate_hessian` 也已实际运行（两个 65 原子、B3LYP-D3(BJ)/def2-SVP/SMD、8 核）。
两项作业均在约 1800 秒墙钟上限前后停在 Gaussian `l1002` 的 CPHF/热化学阶段，没有输出
`Normal termination`，因此 Action 没有生成可用的频率/`formchk` 结果；checkpoint 和完整日志均保留在
`.runtime/case2_gaussian_hessian/`，详细状态见 `.runtime/case2_gaussian_hessian/case2_hessian_attempt_summary.json`。
在停止前 Gaussian 已打印电子能；经逐项核对输入 GJF 与 JSON 坐标，Concerted_R 为
`−1862.46962013 Eh`、Concerted_S 为 `−1862.46883913 Eh`（早期运行摘要曾把标签写反，现已更正）；
这两个电子能不能替代频率校验或论文的 ΔG‡。
此前单点作业的 `job.gjf`、`job.log`、`job.chk` 位于
`.runtime/case2_gaussian_run/outputs/calculate_energy/gaussian/`；本次 Hessian 文件位于
`.runtime/case2_gaussian_hessian/outputs/calculate_hessian/gaussian/`。
随后对 R、S 分支分别提高到 32 核/24 GB/3600 秒，两个 `calculate_hessian` 均返回 `success`，
Gaussian 16 C.01 正常终止，`formchk` 成功生成 195×195 Hessian。R 的 189 个频率中有 3 个虚频
（−451.7441、−35.8975、−21.8377 cm⁻¹）；S 也有 3 个虚频（−456.0570、−16.7831、
−7.9037 cm⁻¹）。结果分别位于 `.runtime/case2_gaussian_hessian_long/` 和
`.runtime/case2_gaussian_hessian_long_s/`，各自含 `result.json`、`frequency_gate.json`、GJF、LOG、
CHK 和 FCHK。这证明 g16、formchk 和 Hessian 解析链真实可用，同时证明固定发布坐标在本轮方法下
不是可直接用于 IRC 的一阶鞍点。

**缺陷/影响。** Gaussian 缓存实际存在于 `third_party/minichem_toolbox/.mini_software_cache/gaussian/g16`，
但旧版 `resolve_command()` 只按进程 cwd 检查环境变量中的相对路径，导致从仓库根启动时误报
`Executable 'g16' was not found`。本轮已将相对命令解析为相对于 MiniChem 根目录或 cwd 的可执行文件，
并用真实 E/F/G 单点验证；该修复是通用运行时路径修复，不改变科学计算逻辑。

### Step 5：CsPPh₂ 复合物形成自由能

**目标。** 检查反思后 G 的基态复合是否热力学可行。

**实现。** 从 `case2.json` 取 `hypothesis_G/cp5`、`hypothesis_E/cp1` 和 `hypothesis_G/CsPPh2` 的
`freeenergy`（Hartree），按 `ΔG = G(cp5) − G(cp1) − G(CsPPh2)`，再乘 627.509474 kcal/Hartree。

**结果。** 机器数据重放为 **−4.62 kcal/mol**（−0.007359 Eh），方向上支持复合形成有利；正文/SI
B.2.2 报告为 **−3.0 kcal/mol**。两者不是单位换算误差，而是发布数值与叙述存在约 1.6 kcal/mol 差异。

**缺陷/影响。** 在没有原始 Gaussian 输入、溶剂/标准态修正和构象集合的情况下，无法判定 −3.0 是另一
构象、不同物种组合还是人工四舍五入。当前工具箱应把“复合物定义、标准态、自由能单位”作为强制输入，
否则智能体可能错误地把 −4.62 和 −3.0 混写成同一结果。本轮不修改发布 JSON。

### Step 6：A–D 协同/闭壳层路径 TS

**目标。** 通过 TS、单虚频、IRC 和 `ΔG‡` 判断热路径是否能解释室温反应。

**实现。** 需要 `locate_transition_state` → `calculate_hessian` → `trace_intrinsic_reaction_coordinate`
（forward/backward）→ `parse_quantum_chemistry_output`；修正 workflow 的通用 route 为 B3LYP-D3(BJ)/def2-SVP、
SMD、`Opt=(TS,CalcFC,NoEigenTest)`、双向 `IRC=(CalcFC,LQA,MaxPoints=50,StepSize=10)`。

**结果。** `Concerted_R` 和 `Concerted_S` 发布记录重放：

| 分支 | ΔG‡（JSON，kcal/mol） | 论文代表值 | 判定 |
|---|---:|---:|---|
| Concerted R | 45.995 | 46.0 | 高势垒，排除 |
| Concerted S | 45.980 | 约 46.0 | 高势垒，排除 |

本轮已从发布 TS 坐标真实调用 `calculate_hessian/gaussian`。R/S 在 32 核配置下均成功完成并由
`formchk` 解析出 195×195 Hessian，但两者都各有 **3 个虚频**，具体数值见 Step 4，因此均未通过
单虚频 TS 门禁。前一次 R/S 的 8 核作业因 1800 秒上限超时，详情保存在
`.runtime/case2_gaussian_hessian/`。

该门禁失败后没有盲目对三阶鞍点执行 IRC，而是实际调用 `locate_transition_state/gaussian` 对 R/S
重新精修。调用路线为 `B3LYP-D3(BJ)/def2-SVP`、SMD(EthylEthanoate)、UltraFine、
`Opt=(TS,NoEigenTest,CalcFC,MaxStep=5,MaxCycles=100)`，R/S 各 32 核、24 GB；作业文件保存在
`.runtime/case2_gaussian_ts_refine/{R,S}/`，当前仍在运行。原 Action 只能表达 `CalcAll` 或不指定力常数，
不能表达修正 workflow 所需的一次性 `CalcFC`；本轮为 Gaussian TS Action 增加了通用、可选的
`initial_force_constants=calcfc` 参数，并保留原 `calculate_all_force_constants` 行为。这个补充不含
Case2 专用数据，只扩展 Gaussian 原生 TS route 的可表达范围。精修完成后还必须重新 Hessian；只有
恰好一个虚频的结构才会实际提交 forward/backward IRC。

**缺陷/影响。** MiniChem 的 IRC Action 能生成双向请求和解析轨迹，但长任务恢复依赖 checkpoint；若没有
minimum/端点化学图门禁，只看到“作业正常结束”不能证明 TS 连接了正确的反应物/产物。

### Step 7：E 孤立底物直接激发

**目标。** 检查孤立 α-碘硼酸酯是否能吸收 450 nm 光。

**实现。** 对 `hypothesis_E/cp1_excited` 的 `etenergies`（cclib 单位 cm⁻¹）和 `etoscs` 进行通用换算：
`1 cm⁻¹ = 0.002859144 kcal/mol`，并同时计算波长和振子强度。

**结果。** S₁ = **102.84 kcal/mol**, 278.0 nm, f=0.0059；S₂ = 105.60 kcal/mol, 270.7 nm。
论文报告最低单重激发约 **102.8 kcal/mol**，与发布数据一致；远高于 63.6 kcal/mol，E 排除。

**缺陷/影响。** TD-DFT Action 能表达 `TD(NStates=20) B3LYP/def2SVP EmpiricalDispersion=GD3BJ
SCRF=(SMD,Solvent=EthylEthanoate) Integral=UltraFine`，本轮已由项目内 `g16` 实际执行。还应由后续
通用输出解析保留激发态性质/轨道贡献，不能只比较最低能量。

**本轮真实调用。** 在发现项目内缓存后，使用同一路由和 `number_of_states=20` 对 JSON 中的固定
`cp1` 几何实际调用 `calculate_excited_states/gaussian`。Action 返回 `success`、20 个 singlet 态，
并保留 `.runtime/case2_gaussian_run/outputs/calculate_excited_states/gaussian/6fc97cdf55a4/`
下的 GJF、CHK、LOG。第一态为 `4.4596 eV`、`35969.10 cm⁻¹`、`278.01 nm`、`f=0.0059`，
换算为 **102.84 kcal/mol**，与论文 `102.8 kcal/mol` 一致。F/G 的相同 TD-DFT 作业已完成并已追加到结果文件。

### Step 8：F 底物–膦关联体系激发

**目标。** 检查加入膦后形成的关联体系是否把激发能降到蓝光可及范围。

**实现。** 对 `hypothesis_F/cp4_excited` 采用同一 TD-DFT 数据审计和光子能门禁。

**结果。** 发布数据重放给出 S₁ = **75.80 kcal/mol**、377.2 nm、f=0.0653；本轮真实 Gaussian
TD-DFT 也给出 S₁ = **75.8000 kcal/mol**、377.20 nm、f=0.0653，S₂ = 79.15 kcal/mol、361.24 nm。
论文报告 **75.8 kcal/mol**，与真实 Action 结果一致；仍高于 63.6 kcal/mol，F 排除。真实结果的
ActionResult 和 Gaussian 文件保存在 `.runtime/case2_gaussian_run/case2_td_results.jsonl` 及对应
`outputs/calculate_excited_states/gaussian/4545b9b49332/`。

**缺陷/影响。** 关联体系的“哪个结构/电荷态”必须绑定到输入 artifact；不能只用名称 `cp4`。MiniChem
目录已有结构身份和 charge/multiplicity 字段，但跨 Action 的强一致性校验仍需要调用方保存并核对。

### Step 9：G CsPPh₂ 协同复合物激发

**目标。** 测试反思假设是否同时满足光子能和有效吸收。

**实现。** 对 `hypothesis_G/cp5_excited` 读取前两个单重态能量与振子强度，并与 450 nm 比较；随后用
自由基/碘原子对象检查 C–I 均裂的可表示性。这里只做通用 TD-DFT/物种契约，不增加论文特制 Action。

**结果。** 发布数据和本轮真实 Gaussian TD-DFT 均给出 S₁ = **44.03 kcal/mol**, 649.4 nm, f=0.0044；
S₂ = **64.02 kcal/mol**, 446.6 nm, f=0.0567。论文报告 S₁ 649 nm/f=0.0044、S₂ 447 nm/f=0.0567、约 64.0 kcal/mol；S₂ 与 450 nm
高度匹配，G 通过光物理门禁。C–I 均裂/ISC 是机制解释，发布 JSON 没有直接的激发态 IRC 或 SOC 计算，
因此本轮只能称为“激发能支持”，不能称为已证明完整自由基路径。

**工具状态。** Gaussian route/TD parser 的专项测试通过；E、F、G 三个真实 TD-DFT 均已完成并被
MiniChem 解析为 20 个 singlet 态，不再把它们标记为“仅接口测试”。

### Step 10：与实验控制和论文结论综合

**目标。** 把量化门禁与实验事实合并，而非只按最低能量选择机制。

**结果。** A–D：约 46 kcal/mol，高于室温可行范围；E：102.8 kcal/mol；F：75.8 kcal/mol；G：S₂
64.0 kcal/mol，接近 450 nm 的 63.6 kcal/mol。G 还与“需要蓝光、碱、膦、TEMPO 抑制”一致，因此
当前数据重放恢复论文的定性结论：**CsPPh₂/底物协同复合物的光诱导 C–I 均裂是最可行的候选起始步骤**。

**尚未完成的独立验证。** 没有 G 的激发态 C–I 断裂扫描、没有 SOC/ISC 速率、
没有完整 HAT/链传播能垒。因此不能声称复现了完整反应动力学或 83% 收率。

### Step 11：工具箱回归和阻塞记录

**已执行测试。**

以下命令、通过数和依赖状态保留当次回归的历史证据；当前运行请使用本文开头说明的标准
`arche` Python 和完整 `PYTHONPATH`。

- `PYTHONPATH=third_party/minichem_toolbox/src .venv/bin/pytest -q third_party/minichem_toolbox/tests/test_gaussian_case23_actions.py third_party/minichem_toolbox/tests/test_gaussian_open_execution_contract.py third_party/minichem_toolbox/tests/test_gaussian_typed_backend.py third_party/minichem_toolbox/tests/test_free_energy_selectivity.py`
  ：已通过 **40 项**（包含新增的 CalcFC TS route 回归；完整并行作业输出待 CPU 资源释放后再次确认）。
- Gaussian TD-DFT route 渲染、20 个激发态解析、闭壳层 singlet 门禁、Gaussian Action 发现和分发均通过。
- 新增的 Gaussian TS `initial_force_constants=calcfc` route 回归通过，生成的关键词为
  `Opt=(TS,NoEigenTest,CalcFC,MaxStep=5,MaxCycles=100)`，且不误加 `CalcAll`。
- 在路径修复前，`calculate_excited_states` 会返回结构化 `backend_unavailable`；修复后已实际启动
  Gaussian 并成功完成 E 的 TD-DFT。
- 该 `backend_unavailable` 已定位为旧版相对路径解析缺陷：缓存位于项目内但环境变量路径未按
  MiniChem 根解析；`resolve_command()` 已修复，并用项目内缓存真实启动 Gaussian 验证。
- 系统 Python 的 RDKit 构象 smoke test 可生成简单分子构象；当时项目 `.venv` 没有 RDKit，说明当时的环境复现还需
  按 `environment.yml` 安装 conda 依赖后再运行构象步骤。
- 热化学测试最初因当时的 `.venv` 缺 `ase` 失败；已在当时项目 `.venv` 安装 requirements 中声明的 `ase==3.29.0`，
  重新测试通过。该安装未修改模型/软件缓存，也未修改源码。

## 5. 论文对比总表

| 证据 | 当前重放 | 论文 | 一致性/结论 |
|---|---:|---:|---|
| 450 nm 光子能 | 63.58 kcal/mol | 63.6 | 一致 |
| A/B/C/D 代表势垒 | 45.98–46.00 kcal/mol | 46.0 | 一致 |
| R 固定发布坐标虚频 | 3 个：−451.74、−35.90、−21.84 cm⁻¹ | 未公开逐频率值 | 未通过本轮单虚频 TS 门禁 |
| S 固定发布坐标虚频 | 3 个：−456.06、−16.78、−7.90 cm⁻¹ | 未公开逐频率值 | 未通过本轮单虚频 TS 门禁 |
| E S₁ | 102.84 kcal/mol | 102.8 | 一致 |
| F S₁ | 75.80 kcal/mol | 75.8 | 一致 |
| G S₁ | 44.03 kcal/mol, f=0.0044 | 649 nm, f=0.0044 | 一致 |
| G S₂ | 64.02 kcal/mol, f=0.0567 | 64.0 kcal/mol；447 nm, f=0.0567 | 一致 |
| CsPPh₂ 结合 ΔG | −4.62 kcal/mol（按 JSON） | −3.0 kcal/mol | **不一致，需原始输入/标准态解释** |

## 6. 当前可交付结论和后续必要资源

当前工具箱客观上具备 Case2 所需的原子能力：构象/复合物构建、Gaussian 优化/频率/TD-DFT/TS/IRC、
cclib 解析和通用自由能组合；专项回归通过。通过发布 JSON 的 E1 重放，论文的“46.0 → 102.8 → 75.8
→ 64.0”淘汰/收敛链已经恢复。

但 E2/E3 仍受两个客观条件限制：

1. R/S 长时 Hessian 已成功，但固定发布坐标均为 3 个虚频；当前正在实际 TS 精修，不能在其重新通过
   单虚频门禁前执行或宣称完成 IRC；
2. 论文未公开完整历史 Agent 轨迹、所有失败输入、原始 GJF/log、构象集合和 G 的激发态断裂/ISC 证据，
   因而只能进行发布数据重放，不能逐消息复演历史 workflow。

R/S TS 精修完成后将继续实际调用 Hessian，并在门禁通过时调用双向 IRC，保存每个候选的 GJF、log、
checkpoint、虚频/IRC 端点和单位显式的 JSON；当前 E/F/G TD-DFT、R/S 固定坐标 Hessian 和结构 Action
均属于真实 Gaussian/本地后端证据，不再只是接口测试。
