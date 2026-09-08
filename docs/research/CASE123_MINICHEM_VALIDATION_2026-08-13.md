# ARCHE Case 1/2/3 MiniChem 计算 Action 验证（2026-08-13 快照）

本文保存 2026-08-13 至 2026-08-14 的工具能力和真实计算证据，不是当前 Action 数量、运行目录或操作入口的参考。当前运行方式见[交付与复现说明](../current/DELIVERY_AND_REPRODUCTION.md)，当前架构边界见[根目录架构说明](../../ARCHE_CODE_ARCHITECTURE.md)。下文的 `.runtime/` 路径、资源数量和测试总数只表示当次验证环境。

## 三代工具箱对比

| 比较项 | ARCHE 原始化学工具箱 | MiniChem 补充 Case1/2/3 Action 前 | MiniChem 补充 Case1/2/3 Action 后 |
|---|---|---|---|
| 版本基线 | `ARCHE/src/chemistry_multiagent/tools` 本地原始目录 | GitHub 基线 `05c2e74` | 2026-08-14 合并 `origin/main` 后的当前工作树 |
| 注册规模 | `toolpool.json` 实际 7 项 | 55 Actions / 26 Backends / 70 Action-Backend pairs | 63 Actions / 28 Backends / 82 pairs |
| 工具发现 | 7 个脚本项；Gaussian 执行主要藏在外层 ExecutionAgent | typed Action/backend catalog；Gaussian 仅能量、普通优化、Hessian、偶极等基础合同 | typed catalog 延伸至 Gaussian TS/IRC、TD-DFT/TDA、1D relaxed scan、NBO/NAO Wiberg；所有参数可发现 |
| Case1 核心计算 | 软件类别基本具备，但需手写 deck/外层执行；没有结构化 TS/IRC/选择性门禁 | 无 typed Gaussian TS/IRC；`23_TSPipeline` 只能产生 TS guess | typed Gaussian TS、Hessian、单向 IRC、元数据保留、严格终止解析；Case1 发布数据与 Gaussian 复算已完成 |
| Case2 核心计算 | 原生 Gaussian 可手写 TD deck；实际 registry 没有 excited-state Action，旧 parser 也不提供完整 TD 结构化合同 | Mini profile 未暴露 `calculate_excited_states`，且 Gaussian 无 TD provider | `calculate_excited_states/gaussian`：1 至 100 roots、TD-DFT/TDA、singlet/triplet、激发能/波长/振子强度/跃迁解析 |
| Case3 扭转与咬合角扫描 | 可用 `generate_gaussian_code + xyz_to_gjf + Gaussian` 人工组合；没有结构化 scan Action 结果和点数门禁 | `scan_reaction_coordinates` 仅 pysisyphus provider | 增加 Gaussian provider：bond/angle/dihedral、一维 relaxed scan、显式范围/步数/单位和 `steps+1` 完成门禁 |
| Case3 Wiberg/NBO | 可手写 `Pop=NBORead`，但 toolpool 未注册 NBO/Wiberg 计算与矩阵解析 | `calculate_bond_orders` 只有 xTB/Multiwfn | 增加 Gaussian NBO/NAO Wiberg provider，输出完整矩阵、阈值 pair 列表及 NBO 版本 |
| 论文分析逻辑 | 多由 LLM 或临时 Python 完成 | 有部分 internal analysis Action | 本轮未新增 Case2/3 专用分析 Action；差值、排序、BDE、波长比较仍由普通 Python 脚本完成 |
| 缓存变更 | 不适用 | 既有软件/模型缓存 | 软件安装、模型权重/checkpoint 和 manifest 未变；真实计算写 `.runtime/`，早期临时 scratch/字体索引已清理且不进入 Git |

这里的“补充前”指共同基线提交 `05c2e74a1bd2b255aed0cb46940b88be50a8b358`。“补充后”同时
包含作者 Gaussian codegen/格式转换接入、Case 1 通用能力和 Case 2/3 Gaussian typed Action。

### 原始 ARCHE 工具箱的判断边界

原始 `toolpool.json` 实际只登记 `plot_tools`、`main`（23_TSPipeline）、
`generate_gaussian_code`、`sdf_to_gjf`、`smiles2sdf`、`xyz_to_gjf` 和
`gen_conformation` 七项。Gaussian 的执行位于外层 ExecutionAgent，不是该目录中可被 Planner
发现的原子工具。因此应区分两个结论：

- 从软件类别看，原始仓库有 RDKit/结构转换、Gaussian 输入生成、外层 `g16` 执行以及基础
  cclib 解析；人工写入正确 route 时，确实能运行 Case2 的 TD-DFT 和 Case3 的 relaxed scan、
  NBO 计算。
- 从自主执行合同看，它没有注册 `calculate_excited_states`、结构化 scan、NBO/Wiberg 解析，
  也没有 root 数、扫描点数/约束、矩阵完整性等门禁。Planner 只能依赖自然语言生成 deck、隐式文件
  和临时 Python 后处理，不能仅凭原始 registry 稳定且可审计地完成 Case2/3。

这也是本轮增加 typed provider、而没有再封装一条论文 workflow 的原因。

## Case2/3 计算需求到通用 Action 的最终映射

| 科学计算环节 | Case2 | Case3 | 当前通用 Action/执行层 | 是否仍缺计算 Action |
|---|---|---|---|---|
| 分子/复合物构建与构象 | 热/光候选初始结构 | 四类 Ni 配体复合物 | `generate_conformer_ensemble`、`build_reactive_complex_ensemble` 等既有 Action | 否；论文未发布的具体初猜仍无法逆推出 |
| 普通几何优化 | 反应物、复合物、自由基 | 四类配体和解离端点 | `optimize_geometry` | 否 |
| TS、频率与 IRC | Mechanism A、B-D 搜索 | 不是最终描述符必需项 | `locate_transition_state/gaussian`、`calculate_hessian/gaussian`、`trace_intrinsic_reaction_coordinate/gaussian` | 否；B-D 缺论文输入/失败日志 |
| 垂直激发态 | E/F/G 及 450 nm 判据 | 不需要 | **新增** `calculate_excited_states/gaussian` | 已补齐 |
| 一维 relaxed scan | 不是发布结论核心 | bite angle 与 Br-N-N-H dihedral | **新增 provider** `scan_reaction_coordinates/gaussian` | 已补齐 |
| NBO/Wiberg | 不是发布结论核心 | Ni-H 电子描述符 | **新增 provider** `calculate_bond_orders/gaussian` | 已补齐 |
| BDE/BDFE | 候选路径可用 | Ni-H 断裂热力学 | 复用 `calculate_energy`；需要 Gibbs 时再接优化、Hessian 和既有热化学能力 | 否；不新增论文专用 BDE Action |
| 差值、排序、450 nm 比较 | 势垒/波长/复合能 | 扫描曲线/Wiberg/BDE 排名 | Agent 编写的普通 Python | 保持通用分析边界，不新增论文专用分析 Action |

因此，“补齐”是对 MiniChem 公共计算合同的扩展，不是把论文 Case workflow 固化成工具。任何满足
相同输入合同的分子、光化学体系、内坐标扫描或 NBO 任务都可复用这些 Action。

### 原始 `gen_gaussiancode.py` 在当前工具箱中的映射

原始 ARCHE 的 `tools/gen_gaussiancode.py` 不是 Gaussian 计算程序。它把自然语言依次交给大模型做
需求解释、route 生成、route 提取和 route 修正，最终只返回以 `#` 开头的 Gaussian route section；
完整 GJF 的坐标、电荷/多重度和真正的 `g16` 执行仍由其他工具与外层 ExecutionAgent 负责。

当前 MiniChem 保留了作者公开名称 `generate_gaussian_code`，并把后续执行分为三个边界：

1. `generate_gaussian_code/gaussian_codegen` 调用作者四阶段 ARCHE-Chem HTTP 工具，只返回经过
   校验的单行 route。该 Action 不生成完整 deck，不运行 Gaussian，失败时不回退到通用模型。
2. 对 Catalog 已覆盖的任务，Planner 显式填写 `method_spec` 和 `action_settings`，Gaussian typed
   adapter 确定性渲染完整输入并执行 `g16`。当前共有 9 个 Gaussian Action，覆盖能量、普通优化、
   Hessian、偶极、TS、IRC、垂直激发态、一维 relaxed scan 和 NBO/Wiberg。
3. 对 typed Action 尚未覆盖的合法 Gaussian 路线，Planner 可在 `authored_files` 中编写完整的
   `code/input.com`，选择 open execution 的 `software_id=gaussian`、`executable=g16`。MiniChem
   在提交前校验 route、分段空行、标题、charge/multiplicity、坐标以及 CPU/内存映射，执行后记录
   native 日志和科学收敛轴；Gaussian IRC 只有明确检测到 PES minimum 才标记收敛，单独的
   branch-complete 文案只作为过程证据；`formchk` 也作为同一软件条目中的可发现命令提供。

`smiles2sdf`、`sdf_to_gjf` 和 `xyz_to_gjf` 负责结构与 deck 格式转换。`sdf_to_gjf` 和
`xyz_to_gjf` 可以绑定 codegen 返回的 route，但不会自行调用模型或运行 Gaussian。

## 相比 GitHub 基线的精确 Action 差异

相对 `05c2e74`，当前 Catalog 新增 8 个 Action ID、12 个 Action-Backend pair，没有移除旧 pair：

| Action/backend pair | 阶段 | 通用能力与边界 |
|---|---|---|
| `locate_transition_state/gaussian` | Case1 | 从任意分子 TS guess 做受约束 Gaussian TS 优化；不自动把结果宣称为已验证 TS |
| `trace_intrinsic_reaction_coordinate/gaussian` | Case1 | 单方向 Gaussian IRC；forward/backward 是两个原子 Action，只有检测到 PES minimum 才标记收敛 |
| `analyze_non_covalent_interactions/multiwfn` | Case1 | 通用波函数 NCI/RDG 网格与候选临界点；不是选择性因果或相互作用能 Action |
| `analyze_free_energy_selectivity/internal_thermochemistry` | Case1 | 显式带标签 Gibbs 自由能的 Boltzmann 汇总；不推断 R/S 或 major/minor |
| `analyze_energetic_span/internal_thermochemistry` | Case1 后续完整循环能力 | 显式完整循环的 Kozuch-Shaik energetic span/TOF；不能从缺失循环状态构造网络 |
| `calculate_excited_states/gaussian` | Case2 | 通用闭壳层参考 TD-DFT/TDA 垂直激发；不做 excited-state optimization、SOC、ISC 或 NTO |
| `scan_reaction_coordinates/gaussian` | Case3 | 通用一维 relaxed internal-coordinate scan；不固化任何论文原子索引或扫描范围 |
| `calculate_bond_orders/gaussian` | Case3 | 通用 NBO/NAO Wiberg 矩阵；Wiberg 是 population descriptor，不等于 BDE |
| `generate_gaussian_code/gaussian_codegen` | 作者工具接入 | 调用作者四阶段专家工具并返回单行 Gaussian route；不生成 deck 或运行 Gaussian |
| `smiles2sdf/format_conversion` | 作者工具接入 | 把非空 SMILES 列表转换为带三维坐标的 SDF |
| `sdf_to_gjf/format_conversion` | 作者工具接入 | 把 SDF 结构和受限 route 转换为 Gaussian 输入文件 |
| `xyz_to_gjf/format_conversion` | 作者工具接入 | 把 XYZ 结构和受限 route 转换为 Gaussian 输入文件 |

相对基线新增的公开 Action ID 分为两组。第一组是 `analyze_non_covalent_interactions`、
`analyze_free_energy_selectivity`、`analyze_energetic_span` 和 `calculate_excited_states`。第二组是
作者公开名称 `generate_gaussian_code`、`smiles2sdf`、`sdf_to_gjf` 和 `xyz_to_gjf`。
Case3 复用既有 `scan_reaction_coordinates` 和 `calculate_bond_orders` Action，只增加 Gaussian provider。
本轮遵循“计算能力进 Action，论文分析留给 Python”的边界，没有新增 BDE、论文机制判定、
配体排序或 450 nm 匹配之类 Case 专用 Action。

### Action 数量之外的既有能力强化

相对共同基线，还有一些不会增加 Action ID、但会改变科学可靠性的修复，不能从 `63 vs 55`
这一计数中看出来：

| 既有能力 | 基线问题 | 当前行为 |
|---|---|---|
| Gaussian 结构串联 | 优化后只保留坐标，可能丢 charge、multiplicity、bonds 和 atom map | 所有坐标更新保留分子电子态、键图、候选标识及逐原子映射；method override 与实际 deck/输出一致 |
| Gaussian IRC | 可把达到 MaxPoints 或只有 branch-complete marker 误记为收敛，也可能计入未接受尝试点 | 只收录带接受标记、完整几何和 SCF 能量的点；区分 PES minimum、MaxPoints、branch complete 和缺 marker |
| 振动模式 | 可能按频率绝对值盲删外部模式 | 在质量加权坐标中投影质心平移和主惯性轴转动；线性去 5、非线性去 6，并输出可审计线性判据 |
| 23_TSPipeline UMA→NEB | 端点 `partial_success/converged=false` 仍可能进入 NEB | 任一端点未明确 `converged=true` 就跳过整批 NEB，并返回原因和 `partial_success` |
| Multiwfn NCI/RDG | 过细网格可能无预警产生超大文本/cube；CP/版本完整性不足 | 运行前按坐标包围盒估算并受 `max_grid_points` 硬上限约束，运行后再核点数；版本、完成 marker、CP 数和连续索引均校验 |
| 自由能选择性 | 重复 conformer ID、极端温度和任意 ArtifactRef 可污染结果 | 显式 ID 全局唯一，RT 有限门禁，区分 population/enantiomeric，只有后者输出 ee；输入 Artifact 语义受限 |

这些都是通用合同/验证修复，不包含 Case1/2/3 的特定分子、原子编号或结论阈值。

## 本轮文件级修改记录

| 文件 | 修改内容 |
|---|---|
| `actions/molecular_electronic.py` | 为既有 excited-state 和 bond-order 原子 Action 增加 Gaussian provider |
| `actions/reaction_and_kinetics.py` | 为既有一维 relaxed scan 原子 Action 增加 Gaussian provider |
| `backend_specs.py` | 声明三项 Gaussian capability、必填字段、允许值、资源/验证级别，并修正软件缓存相对路径提示 |
| `parameter_specs.py` | 暴露 TDDFT/TDA root、spin manifold、Wiberg definition/threshold、扫描坐标/范围/单位/步数/优化设置的参数语义 |
| `mini_profile.py` | 将通用 `calculate_excited_states` 加入当前 MiniChem 公开 profile |
| `backends/quantum_legacy.py` | 确定性生成 Gaussian deck，执行并解析 TD/TDA roots、NBO/Wiberg 分块矩阵和完整 relaxed scan；增加 scratch 隔离、失败 core 清理、坐标/点数门禁及 solvent token 边界；真实冒烟发现后将 TDA 修正为 Gaussian 接受的独立 `TDA=(...)` route keyword |
| `backends/electronic.py` | 将 Gaussian excited-state/bond-order 请求分发到 typed adapter |
| `backends/reaction.py` | 将 Gaussian relaxed-scan 请求分发到 typed adapter |
| `minichem_mcp_tools/open_execution.py` | 收紧原生 Gaussian IRC 的科学收敛轴：branch complete 与 PES minimum 分开，只有后者满足反应路径收敛；移除已经不存在的分布式资源异常分支，使资源超限稳定返回结构化错误 |
| `tests/test_gaussian_case23_actions.py` | 新增 route、parser、完整性门禁、catalog/discovery/dispatcher、scratch/core 和输入边界测试 |
| `tests/test_gaussian_open_execution_contract.py` | 冻结原生 Gaussian IRC PES-minimum 收敛语义及资源预算超限错误合同 |
| `scripts/tasks/case23_reproduce.py` | 发布数据确定性重放；可选调用通用 TD/NBO/scan/energy Actions；BDE/排名只用 Python 组合 |
| `tests/test_case23_reproduction.py` | 固定 Case2/3 发布值、请求映射、全候选覆盖、BDE 组合和通用 Action 复用行为 |
| `backends/gaussian_codegen.py`、`vendor/arche_author_tools/` | 把作者四阶段 Gaussian route 生成工具接入 MCP，校验单行 route 并隔离模型配置 |
| `actions/format_conversion.py`、`backends/format_conversion.py` | 接入作者 `smiles2sdf`、`sdf_to_gjf` 和 `xyz_to_gjf`，只转换结构和 deck，不运行 Gaussian |
| `tests/test_gaussian_codegen_action.py`、`tests/test_format_conversion_actions.py` | 固定作者工具名称、请求合同、错误边界和格式转换行为 |
| `README.md`、`TOOLBOX.md`、`docs/` | 更新当前计数、能力边界、复现命令、三代对比和真实计算证据；历史快照正文不改写 |

未修改 `supplementary/case2.json`、`supplementary/case3.json`、软件安装、模型权重/checkpoint
或缓存 manifest。真实测试曾生成可重建的 Gaussian scratch 和 Matplotlib 字体索引，收尾时已按
明确文件名清理；科学结果保留在 Git 忽略的 `.runtime/`。

## 通用计算代码修改

### Gaussian typed adapter

主要实现位于 `third_party/minichem_toolbox/src/minichem_toolbox/backends/quantum_legacy.py`：

1. 垂直激发态：渲染 `TD=(NStates=...,Singlets|Triplets)` 或
   `TDA=(NStates=...,Singlets|Triplets)`，解析 state index、
   symmetry、eV、cm-1、nm、oscillator strength、`<S**2>` 和主要 MO transitions；请求 root
   数与解析数不一致时返回 `partial_success`。
2. 一维 relaxed scan：校验 2/3/4 个唯一零基原子索引、单位、有限端点、步数和角度范围；
   渲染 Gaussian ModRedundant 输入。坐标数值强制带小数点，避免 Gaussian 把整数文本误判为
   额外原子编号。每个 `Optimization completed` 段解析最后结构和 SCF 能量，只有完整
   `steps+1` 点及 scan summary 才返回 `success`。
3. Wiberg/NBO：渲染 `SP Pop=NBORead` 和 `$NBO BNDIDX $END`；支持大分子分块矩阵，校验
   原子顺序、矩阵完整性和打印精度内对称性，输出全矩阵及阈值 pair 列表。
4. 三项能力继续保留输入 `charge`、`multiplicity` 和结构元数据。激发态 provider 当前限制为
   multiplicity=1 的闭壳层参考，避免把未经验证的开放壳层 TD 路径标记为支持。
5. Gaussian 失败时保留 `.gjf/.log/.err`，并仅删除该隔离作业目录中由失败生成的 `core.*`，
   防止一个输入错误留下 GB 级 core dump。
6. Gaussian 临时 scratch 显式放入当前 Action 的 `.runtime` 作业目录，结束时清理；不会把计算
   中间文件写入共享软件缓存。二面角采用 Gaussian `D(i,j,k,l)` 有符号约定，并对每个收敛几何
   与目标坐标的偏差做门禁，不能只凭点数和 summary 宣称扫描完成。
7. Gaussian 不可用时的安装提示修正为仓库真实相对目录
   `.mini_software_cache/gaussian/g16`；这里只改提示和测试，没有移动或写入缓存。
8. 为支持 `N,N-Dimethylacetamide`、`1,2-Dichloroethane` 等真实溶剂名，solvent parser
   接受受限的数字或 N/O/S/P 位次前缀；同时拒绝 `Acetone,Read` 一类把额外 SCRF 选项塞进
   solvent 字段的输入，保持 typed route 边界。
9. TDA 必须渲染为 Gaussian 16 的独立 `TDA=(...)` keyword，不能作为 `TD=(...,TDA)` 的子选项。
   首次真实 triplet/TDA 冒烟在 Link 1 报 route 语法错误并产生 core；工具按设计保存诊断并只清理
   该作业 core。修正 renderer 后，同一 H2O 请求由 Gaussian 16 C.01 成功返回 2/2 个连续 triplet
   roots，说明问题已由真实执行而不是 mock 覆盖闭环。

### Catalog、discovery 与分发

- `actions/molecular_electronic.py`：给 `calculate_excited_states`、`calculate_bond_orders` 声明
  Gaussian provider。
- `actions/reaction_and_kinetics.py`：给 `scan_reaction_coordinates` 声明 Gaussian provider。
- `backend_specs.py`：登记三项 capability、必填 method/settings、允许值、system type 和
  `real_smoke` 验证级别。
- `parameter_specs.py`：让 Planner 可看到每个参数的类型、范围、影响和无默认值语义。
- `backends/electronic.py`、`backends/reaction.py`：把三个 Action 路由到 Gaussian typed adapter。
- `mini_profile.py`：把 `calculate_excited_states` 加入 MiniChem 公开 profile。

## Case2 发布候选重放与论文对比

`scripts/tasks/case23_reproduce.py` 对 `supplementary/case2.json` 的所有发布候选进行了确定性重放：

- `supplementary/case2.json` SHA256：
  `fa4af5aec29ab5d99d4b419216c8672e0062355d476ffe3e0150e5d526577722`；
- `supplementary/case3.json` SHA256：
  `718959095618dd644effbe09478d9dcf3ea3d7225320159015ef24e6ec21b95e`；
- 确定性重放产物 SHA256：
  `5a382cfcfe900b0b09f1a1434ff2bdd350e9825ea3e255055488198734872a5e`。

| 检查 | 本轮发布数据重放 | 论文结论 | 对比 |
|---|---:|---:|---|
| Concerted R 自由能势垒 | 45.9952 kcal/mol | 46.0 kcal/mol | 一致 |
| Concerted S 自由能势垒 | 45.9795 kcal/mol | 46.0 kcal/mol | 一致 |
| Concerted TS 虚频 | 两条路径各 1 个 | 一阶鞍点 | 发布频率记录一致；不是本轮独立 Hessian |
| Mechanism E S1 | 278.016 nm, `f=0.0059` | UV，不匹配 450 nm | 一致 |
| Mechanism F S1 | 377.196 nm, `f=0.0653` | 不匹配 450 nm | 一致 |
| Mechanism G S1 | 649.404 nm, `f=0.0044` | 弱且不匹配 450 nm | 一致 |
| Mechanism G S2 | 446.581 nm, `f=0.0567` | 可由 450 nm 激发且比 S1 更亮 | 一致 |
| `cp5 - cp1 - CsPPh2` Gibbs 形成能 | -4.61784 kcal/mol | -3.0 kcal/mol | 源材料不一致；JSON 无另一构象/修正可解释差值 |

三组 TD 记录均完整包含 20 个 roots。发布数据重放结果在
`.runtime/case23_validation_20260813/case23_reanalysis.json`；该目录被 Git 忽略，关键数值与哈希
均已固化在本文档和回归测试中。

### Case2 候选覆盖边界

| 论文候选 | 本轮可用输入 | 本轮验证状态 |
|---|---|---|
| Mechanism A（Concerted R/S） | 两个 TS、反应物/产物能量和发布频率 | 已重放两条 46 kcal/mol 势垒和各一个发布虚频 |
| Mechanisms B-D | SI 只有“多次 TS 搜索未收敛或落入错误结构”的文字结论 | 无初猜几何、失败日志或逐步轨迹，无法独立重放“未找到 TS”；不能用补造输入冒充论文候选 |
| Mechanism E | 基态结构和 20-root TD 记录 | 已重放，真实 TD Action 另行验证 |
| Mechanism F | 基态结构和 20-root TD 记录 | 已重放，真实 TD Action 另行验证 |
| Mechanism G | 基态结构、复合物能量和 20-root TD 记录 | 已重放 S1/S2 与复合形成能；真实 TD Action 另行验证 |

缺少 B-D 的原始失败输入是论文发布证据缺口，不是当前工具箱缺少计算 Action：现有通用构象、
TS 优化、Hessian 和 IRC Actions 可以接收这些输入并报告失败或收敛，但不能从一段“未收敛”文字
唯一恢复当时尝试过的结构和搜索参数。

### Case2 真实 TD-DFT Action 验证

新增 `calculate_excited_states/gaussian` 用 Gaussian 16 C.01 对 E/F/G 三个发布基态几何分别执行
`B3LYP-D3BJ/def2-SVP/SMD(ethyl acetate), TD(NStates=20,Singlets), Int=UltraFine`。三项均
`success`，每项都严格解析出连续的 20 个 singlet roots：

| 候选/态 | 真实 Action / nm | 发布 JSON / nm | 真实 `f` | 发布 `f` | 结论 |
|---|---:|---:|---:|---:|---|
| E S1 | 278.01 | 278.016 | 0.0059 | 0.0059 | 一致，不匹配 450 nm |
| F S1 | 377.20 | 377.196 | 0.0653 | 0.0653 | 一致，不匹配 450 nm |
| G S1 | 649.40 | 649.404 | 0.0044 | 0.0044 | 一致，弱跃迁且偏离 450 nm |
| G S2 | 446.58 | 446.581 | 0.0567 | 0.0567 | 一致，接近 450 nm 且比 S1 更亮 |

完整 ActionResult 为 `.runtime/case23_validation_20260813/case23_td_all_real.json`，SHA256
`ebbf6162cb9e244bda0eb75b81dd7b5271a361cb9620fd0cb3865449d69d1d85`。这证明新增
provider 不只是解析发布 JSON，而是能从结构重新提交论文尺度 TDDFT，并复现 Case2 的光谱判据；
它仍不把 S2 后续 ISC 解释为已经直接计算。

此外，用不属于论文 Case 的 H2O 小体系真实验证了 `TDA + triplet` 通用分支。Gaussian 16 C.01
返回 2/2 个连续 triplet roots（7.2916 和 9.4256 eV），Action 状态为 `success`。修复后产物为
`.runtime/case23_validation_20260813/gaussian_tda_triplet_smoke_fixed.json`，SHA256
`b4279417e64bb0d46d35bc36294669e1f712cbf839f7f9975fd398923a852c82`。这项额外测试用于证明
接口不是只为论文的 singlet TDDFT 数据硬编码。

## Case3 发布候选重放与论文对比

### 扭转描述符

四套发布 relaxed dihedral profile 均有 13 个点：

| 配体 | 约 60 度最大相对能量 / kcal mol-1 | 排名 |
|---|---:|---:|
| ortho | 12.8263 | 1（最低） |
| meta | 21.9440 | 2 |
| unsubstituted | 22.0758 | 3 |
| para | 22.1260 | 4 |

结果复现论文核心结论：ortho 配体的 Br-N-N-H 扭转畸变代价显著更低。

本轮还从发布的 ortho 起始几何重新运行了完整 13 点 Gaussian relaxed dihedral scan。13/13 个
优化点均正常结束，全部坐标约束通过；实际坐标相对目标的最大偏差为 `0.00070 degree`。实算
最大畸变能为 12.82496 kcal/mol，发布值为 12.82629 kcal/mol，差 0.00133 kcal/mol。经最新版
有符号二面角解析器独立回放的产物为
`.runtime/case23_validation_20260813/case23_scan_real_audited.json`，SHA256
`29d4bc05eaf40be87eda572e210bf343e6bd88ce56850057f4eefbf651cd8e0f`。这同时验证了真实大体系
执行、13 点完成门禁以及 Gaussian `D(i,j,k,l)` 符号约定。

### 咬合角与 BDE/BDFE 候选假设

- 8 套咬合角 profile（add/reduce × 4 配体）均有 6 个点；ortho 在两个方向的最大畸变代价
  都最低（add 6.2437、reduce 5.6162 kcal/mol），与论文定性叙述一致，但该坐标不直接表示迁移。
- 电子 BDE：meta 34.0330、unsubstituted 33.9416、ortho 24.0275、para 34.4768 kcal/mol。
- Gibbs BDFE：meta 21.4495、unsubstituted 21.0147、ortho 10.0653、para 21.0165 kcal/mol。
- 两种解离定义都给出 ortho 最低，与“较易 Ni-H 断裂”的方向一致；它们仍是端点热力学，
  不能替代迁移路径的畸变能。

为确认既有 `calculate_energy/gaussian` 足以支撑而无需新增 BDE Action，本轮对 4 个 parent、4 个
radical 和 H atom 共 9 个发布几何重新执行 M06-2X/def2-TZVP/SMD(DMA) 单点，9/9 均正常终止。
普通 Python 组合 `E(radical) + E(H) - E(parent)` 得到：

| 配体 | 真实 Action 电子 BDE / kcal mol-1 | 发布值 | 差值 |
|---|---:|---:|---:|
| meta | 34.0330287 | 34.0330272 | 0.0000015 |
| unsubstituted | 33.9415817 | 33.9415802 | 0.0000015 |
| ortho | 24.0275470 | 24.0275459 | 0.0000011 |
| para | 34.4768222 | 34.4768207 | 0.0000015 |

恢复后的可审计聚合产物为
`.runtime/case23_validation_20260813/case23_bde_real_recovered.json`，SHA256
`4ddf2211ec98273351a4d9b73803f4b9ac2ab12fb85f9d278573882df4ac4034`。长任务启动时加载的是
修复前脚本，9 个单点完成后因读取旧 `energy_hartree` 字段而未写总报告；这里从 9 份 normal-
termination 日志和标准 `result.energy` 合同恢复，没有重跑或更改科学数据。当前脚本及回归测试已经
固定标准字段，后续整批运行可直接聚合。

### 本轮真实 Wiberg 计算

`supplementary/case3.json` 没有保存论文 Figure S4 所述 Wiberg 数值或原始 NBO 日志。本轮用新增
`calculate_bond_orders/gaussian` 对发布的四个 M06-2X/def2-TZVP/SMD(DMA) 几何重新计算：

| 配体 | Ni-H 距离 / Angstrom | Ni-H Wiberg | 论文陈述 |
|---|---:|---:|---|
| ortho | 1.48117 | 0.8673 | 最大 |
| unsubstituted | 1.49970 | 0.8604 | 非最大 |
| meta | 1.50108 | 0.8586 | 非最大 |
| para | 1.50224 | 0.8580 | 非最大 |

四项均为 Gaussian 16 C.01 `success`、Gaussian NBO 3.1，完整矩阵维度与 23/29 原子数一致。
结果独立支持论文“ortho 的 Ni-H Wiberg 最大，因此电子弱化假设不成立”的陈述。完整 ActionResult：
`.runtime/case23_validation_20260813/case23_nbo_real.json`，SHA256
`244219ccf52bb94d876ccc3ecd333a1d5d79b535cb473e026043b62c9553f1b9`。

## 真实计算与测试状态

- Case2 E/F/G 三套 20-root TD-DFT：3/3 `success`，60/60 个连续 singlet roots，关键波长和
  振子强度与发布 JSON 在打印精度内一致。
- Case3 ortho 13 点 relaxed dihedral scan：13/13 `success`，最大坐标偏差 `0.00070 degree`，
  最大畸变能与发布值相差 0.00133 kcal/mol。
- Case3 NBO：4/4 `success`；电子 BDE 单点：9/9 normal termination，二者均复现 ortho 排名。
- 小体系 Gaussian C.01 服务冒烟：TD 3 roots、TDA triplet 2 roots、NBO/Wiberg 和 3 点 angle
  scan 均成功。
- ortho 咬合角的论文尺度 6 点复算曾用 8 核启动；运行约 94 分钟时，只读 checkpoint 显示
  `A(3,17,8)=86.5782 degree`，约处于第 3 个目标点。由于该早期进程加载了修正前额外带
  `NoSymm` 的请求，且继续运行很可能触及单 Action 7200 秒上限后再启动第二条同规模任务，
  本轮主动终止并确认子进程/scratch 已清理。它没有生成成功 ActionResult，**不作为完成证据**；
  咬合角结论来自 8 条发布 profile 的完整重放，通用 angle Action 的真实执行/解析由上述 3 点
  非论文小体系冒烟覆盖。当前复现脚本已移除咬合角的额外 `NoSymm`。
- Case1/2/3 定向复现脚本测试：14 passed。
- Gaussian typed/open-execution 定向合同：33 passed。
- 2026-08-14 合并验收：MiniChem 全量 283 passed、1 skipped；根仓库 `tests/` 1015 passed。

## 已知边界

1. Case2/3 JSON 是 cclib 摘要，不含全部原始 Gaussian 日志；发布频率/扫描记录的重放不等同于
   从零优化所有几何。真实 Action 复算单列，不与发布数据重放混写。
2. Case2 Mechanisms B-D 没有机器可读初猜、失败日志或最终几何；当前工具能力可以运行新的搜索，
   但无法可审计地重放论文所称的原始失败尝试。
3. Case2 的 `-3.0 kcal/mol` 复合形成能不能由当前 JSON 的 Gibbs 值严格得到；本报告不修改输入
   或引入未记录修正来强行对齐。
4. TD provider 不计算 SOC/ISC rate，因此“随后 ISC”仍是机制解释，不是该 Action 的直接输出。
5. 一维 relaxed scan 只证明指定坐标上的势能趋势；不自动证明波函数连续、配位拓扑相同或机制因果。
   论文尺度 ortho 咬合角实算未完成，不能用于数值对照；最终复现脚本已经按发布 route 将
   `disable_symmetry` 设为 `false`。完整的 Case3 二面角真实扫描仍按论文要求使用 `NoSymm`。
6. Wiberg 和 BDE/BDFE 是互补描述符，都不能替代迁移反应坐标的能垒。
7. 本轮没有改变 `.mini_model_cache/` 的模型权重/checkpoint、`.mini_software_cache/` 的软件安装
   或任何缓存 manifest。早期真实计算在 scratch 隔离修复落地前曾写入 4 组 Gaussian 临时
   `rwf/skr/int/d2e`（约 5.9 GB），测试还生成一个 Matplotlib 字体索引；确认无进程占用后，这些
   可重建临时文件已清理。真实科学输出只保留在 Git 忽略的 `.runtime/`。
8. 全仓静态检查还发现 comprehensive catalog 的历史 DeepMD 分支引用了仓库中不存在的
   `build_mlip_calculator`/`prepare_mlip_atoms`。该问题在 GitHub 基线已经存在；DeepMD 不属于当前
   26-backend MiniChem profile，也不参与 Case1/2/3。由于仓库没有可核对的实现来源，本轮没有
   猜测性补造或删除该 MLIP 分支；本轮新增代码路径与新文件的未定义名称检查均通过。

## 复现命令

以下真实计算命令和其中的 `--cpu-cores 24`、`--memory-mb 48000` 是 2026-08-13
记录的历史运行参数，用于保留当时的实算证据。当前 MiniChem 默认资源预算为 60 CPU、320000 MB、0 GPU；
当前入口、Python 和 `PYTHONPATH` 以 [当前交付与复现说明](../current/DELIVERY_AND_REPRODUCTION.md) 为准。

只重放仓库发布数据：

```bash
scripts/tasks/case23_reproduce.py \
  --output .runtime/case23_validation_20260813/case23_reanalysis.json
```

显式运行真实计算（可按 `td,nbo,scan,bite,bde` 子集选择；论文尺度扫描和 BDE 会运行较久）：

```bash
scripts/tasks/case23_reproduce.py \
  --real-actions td,nbo,scan,bite,bde \
  --cpu-cores 24 \
  --memory-mb 48000 \
  --output .runtime/case23_validation_20260813/case23_all_real.json
```

脚本只负责论文数据映射和普通 Python 分析。真实计算复用公开、通用的 MiniChem Actions；其中
BDE 是三个 `calculate_energy` 结果的普通 Python 差值，没有新增 BDE 或 Case 专用 Action。
