# MiniChem Toolbox

MiniChem Toolbox 是面向分子机理、光化学和结构描述符任务的小型化学工具箱。它以 Gaussian 为核心电子结构软件，同时提供结构准备、构象搜索、低成本量化计算、热化学、反应路径和结果分析能力。

工具箱通过 MCP 暴露三层接口：预定义 Action、原生软件命令、Agent 编写的 Python 分析程序。详细能力见 [TOOLBOX.md](TOOLBOX.md)。

## ARCHE 原子能力扩展状态

当前仓库副本版本为 `0.3.0+arche.20260813`，注册 65 个 Action 和 28 个 Backend。相对复制时的
MiniChem 0.3.0 基线，本副本增加了反应原子映射与变化集、map 对齐端点编辑、reactive-complex
刚体组装与显式候选选择、FairChem UMA 优化、ASH/KNARR NEB、cclib 最终结构提取、
Raman/ROA/VCD/ECD/NMR 数值派生、独立 SVG 绘图、自由能选择性分析、energetic span、
Multiwfn NCI/RDG 分析，以及作者原名的 `generate_gaussian_code`、`smiles2sdf`、
`sdf_to_gjf`、`xyz_to_gjf`。每项能力仍遵循既有
`ActionSpec -> BackendSpec -> backend handler -> service/Artifact` 分层。

这些能力既可由 Agent 组合成原子 Action 序列，也提供边界明确的
`search_transition_state_path/transition_state_path_search` 组合 Action。该 Action
接收反应物/产物 SMILES 和可选的 MMFF/UMA 力场选择；
当前 profile
原有 26 个 Backend 的运行时依赖均可用，Multiwfn 2026.7.15 的三个 Action、FairChem
UMA 优化和 ASH/KNARR/UMA 的短程 NEB 均已真实执行。`uma-s-1p1` 已缓存且离线检查通过；
这些运行证据仍不等于 Case 1 的 TS/frequency/IRC scientific gates 已经通过。当前交付边界见
[`../../docs/current/DELIVERY_AND_REPRODUCTION.md`](../../docs/current/DELIVERY_AND_REPRODUCTION.md)，
详细 Action 差异见
[`../../docs/research/CASE123_MINICHEM_VALIDATION_2026-08-13.md`](../../docs/research/CASE123_MINICHEM_VALIDATION_2026-08-13.md)。

## 目录结构

```text
third_party/minichem_toolbox/
├── README.md
├── TOOLBOX.md
├── environment.yml                 # Conda 环境的可复现声明
├── requirements.txt                # Conda 创建后安装的 pip 依赖
├── environment-lock/               # 当前 Linux 环境的精确 Conda/Pip 锁和关键版本
├── pyproject.toml
├── config/                         # 运行时、软件命令和能力配置
├── docs/software/                  # Agent 可检索的本地软件文档
├── scripts/                        # 启动、验证、清单更新和兼容脚本
├── vendor/ash-0.95/                # 仓库内、未修改的 ASH 0.95 源码
├── src/
│   ├── minichem_toolbox/           # Action、Backend、作者工具源码快照和执行核心
│   └── minichem_mcp_tools/         # MCP 工具、作业管理和软件检索
├── tests/
│   ├── harnesses/                  # Codex、Claude、OpenCode 隔离测试入口
│   └── react/                      # 直接 API ReAct 测试入口
├── .mini_software_cache/           # 原生软件缓存，软件文件 Git 忽略
└── .mini_model_cache/              # 本地模型缓存，模型文件 Git 忽略
```

`environment.yml` 位于工具箱根目录，是应纳入版本管理的依赖声明。ARCHE 与 MiniChem
共用 `仓库根目录的 .venv` 标准环境；依赖声明用于
检查和补充该环境，不创建仓库内的 `.venv` 或 `.envs/minichem`。
MiniChem 默认资源预算为 60 CPU、320000 MB 和 0 GPU；可通过
`RESEARCHCHEMBENCH_AVAILABLE_*` 环境变量显式覆盖。

## 准备环境

从仓库根目录执行：

```bash
bash scripts/bootstrap_reproduction.sh
```

该脚本检查并补充统一标准环境。也可以在工具箱目录单独运行
`bash scripts/bootstrap.sh` 补充 MiniChem 依赖；两个入口都不会创建新的 Python 环境。

全目录合同与运行时冒烟：

```bash
PYTHONPATH=src:third_party/minichem_toolbox/src \
  .venv/bin/python \
  third_party/minichem_toolbox/scripts/smoke_test_catalog.py \
  --output .runtime/minichem_catalog_smoke.json
```

本仓库的完整行为测试使用统一环境：

```bash
PYTHONPATH=src:third_party/minichem_toolbox/src \
  .venv/bin/python \
  -m pytest -q third_party/minichem_toolbox/tests
```

已有 Gibbs 自由能数值时，可用确定性的 `analyze_free_energy_selectivity`
直接做选择性后处理，无需伪造 GoodVibes 输出文件。调用者必须显式提供标签；同标签记录
按构象配分函数汇总，工具不会根据候选名推断 major/minor 或 R/S：

```json
{
  "action_id": "analyze_free_energy_selectivity",
  "inputs": {
    "free_energies": [
      {"label": "major", "conformer_id": "TS-2a", "gibbs_free_energy": 0.0},
      {"label": "minor", "conformer_id": "TS-2b-1", "gibbs_free_energy": 1.8848503031}
    ],
    "energy_unit": "kcal_mol"
  },
  "action_settings": {
    "temperature_kelvin": 303.15,
    "comparison_type": "enantiomeric"
  }
}
```

`comparison_type` 是必填的语义声明。通用 `population` 比较输出组自由能、population、
`delta_delta_g_kcal_mol`、population ratio、signed population excess 和
`absolute_population_excess_percent`；只有显式选择 `enantiomeric`、提供恰好两个穷尽的
相反对映路径标签时才输出 `ee_percent`。所有显式 `conformer_id` 必须全局唯一；不提供 ID
的记录仍作为独立微观态计入配分函数。

完整催化循环的每个 intermediate/transition state Gibbs 自由能、循环顺序和单圈
`overall_reaction_free_energy` 都已知时，可用确定性的 `analyze_energetic_span` 计算
Kozuch-Shaik energetic span、TDI、TDTS 和完整双重求和分母的正向 TOF。该 Action 只做
数值后处理：不会补齐缺失中间体/过渡态、推断循环顺序或从 Case1 的单步 TS 集合构造完整循环。

Gaussian typed adapter 还提供三项通用计算能力：

- `calculate_excited_states/gaussian`：闭壳层单重态参考的 TD-DFT/TDA 垂直激发，显式指定
  singlet/triplet manifold 和 1 至 100 个 roots；输出激发能、波长、振子强度、态对称性和主要
  轨道跃迁。它不做激发态优化、SOC、ISC 或 NTO 分析。
- `scan_reaction_coordinates/gaussian`：一维 bond/angle/dihedral relaxed scan，显式提供零基原子
  索引、起点、终点、步数、单位、优化收敛和 force-constant 策略；仅当所有 `steps+1` 个点和
  Gaussian scan summary 都完成时返回 `success`。
- `calculate_bond_orders/gaussian`：Gaussian NBO 的 NAO 基 Wiberg 矩阵和阈值过滤后的原子对列表；
  输出完整矩阵并校验原子顺序、矩阵完整性和对称性。Wiberg 指标是波函数/基组依赖的 population
  descriptor，不等同于键能。

这三项都复用普通 `method_spec`、`action_settings` 和 `AtomicStructure`，没有论文 Case 专用参数。
论文结论的差值、排序和绘图仍由 Agent 编写 Python 程序完成，不在 Action 中固化。

2026-08-13 至 2026-08-14 的 Action 验证快照见
[`../../docs/research/CASE123_MINICHEM_VALIDATION_2026-08-13.md`](../../docs/research/CASE123_MINICHEM_VALIDATION_2026-08-13.md)。
阶段节点见 [`../../docs/history/CHANGELOG.md`](../../docs/history/CHANGELOG.md)。

### FairChem UMA / ASH 运行时

FairChem 2.21 要求 NumPy 2.x 和 Torch 2.8；原始 ASH 0.95 在 NumPy 2.4 的更严格
数组到标量转换下会失败。统一环境因此固定 NumPy 2.3.5，位于两者兼容交集
`FairChem: >=2.0,<2.5` 与 `ASH: <2.4` 内。`fairchem`、`ash_neb`、组合 TS Action
和基础工具都使用 `仓库根目录的 .venv` 标准环境。

本仓库固定 `fairchem-core==2.21.0`、Torch 2.8 CPU 构建、
`transformers==4.57.6`、`rxnmapper==0.4.2` 和 ASH 0.95。安装脚本从
`vendor/ash-0.95` 复制到临时目录后构建，不修改源目录，也不在源目录生成构建元数据。
`MINICHEM_ASH_SOURCE` 只用于开发者显式覆盖，正常复现无需设置。

环境检查只验证模块与 CUDA build，不会自动下载 gated UMA 模型。获得 `facebook/UMA` 授权后，
用自己的令牌显式预热项目缓存：

```bash
hf auth login        # 或 export HF_TOKEN=...；令牌不写入仓库
MINICHEM_HOME="$PWD/third_party/minichem_toolbox" \
PYTHONPATH=src:third_party/minichem_toolbox/src \
  .venv/bin/python \
  third_party/minichem_toolbox/scripts/prepare_uma_checkpoint.py --device cuda --allow-download
```

无 GPU 节点可把 `--device cuda` 改为 `--device cpu`；这是功能性降级，不代表 GPU 性能复现。
不带 `--allow-download` 时脚本强制离线，只接受 `.mini_model_cache` 中已有 checkpoint。

## 软件缓存

所有工具箱专用软件放在相对路径 `.mini_software_cache/`：

```text
.mini_software_cache/
├── gaussian/g16/install/g16/
├── gaussian/g16/scratch/
└── multiwfn/2026.7.15/Multiwfn_2026.7.15_bin_Linux_noGUI/
```

Conda 环境提供 RDKit、Open Babel、xTB、CREST、cclib、QCElemental 等开源依赖。Gaussian 16 是商业授权软件，必须从已有合法安装中复制，工具箱不会下载或重新分发。当前 Multiwfn 为官方网站下载的 `2026.7.15` Linux noGUI 包，缓存同时保留原始 ZIP 和解压目录。原 ARCHE 的 3.8-dev 目录只作为配置参考：其中 `Multiwfn_noGUI` 文件小于 ELF 声明的可执行段，实测会段错误，未作为当前安装来源。

更新软件和模型哈希清单：

```bash
PYTHONPATH=src:third_party/minichem_toolbox/src \
  .venv/bin/python \
  third_party/minichem_toolbox/scripts/update_manifests.py
```

## 模型缓存

语义检索模型来自 Hugging Face：

```text
sentence-transformers/all-MiniLM-L6-v2
```

模型文件放在：

```text
.mini_model_cache/
├── all-MiniLM-L6-v2/                         # Action/文档语义检索
├── rxnmapper/albert_heads_8_uspto_all_1310k/ # 反应原子映射
└── huggingface/                              # gated UMA 权重（获批后生成）
```

可选下载命令：

```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 \
  --local-dir .mini_model_cache/all-MiniLM-L6-v2
```

没有该模型时，Action 和软件文档仍可使用词法检索；仅混合语义召回不可用。

## 启动 MCP

```bash
PYTHONPATH=src:third_party/minichem_toolbox/src \
  bash third_party/minichem_toolbox/scripts/start_mcp.sh \
  --transport stdio --discovery-mode progressive
```

脚本默认使用 `.venv/bin/python`，并加载
`src/` 中的工具箱包和仓库根目录中的 ARCHE 包。

验证环境、缓存和工具配置：

```bash
PYTHONPATH=src:third_party/minichem_toolbox/src \
  .venv/bin/python \
  third_party/minichem_toolbox/scripts/verify_minichem.py
PYTHONPATH=src:third_party/minichem_toolbox/src \
  .venv/bin/python \
  -m minichem_mcp_tools.tool_manager validate
```

## Agent 接入

### 模型调用边界

大多数 MiniChem Action 和 Backend 不调用 LLM。`generate_gaussian_code/gaussian_codegen` 是
明确的例外：它调用配置好的 ARCHE-Chem HTTP 服务，执行作者四阶段流程，并只返回经过校验的
单行 Gaussian route。typed Gaussian Actions 仍只根据经过校验的结构化参数确定性生成输入；
原生软件层只执行调用者已经写好的完整输入文件。这两类入口不会隐式调用 codegen Action。

下面的 Codex、Claude、OpenCode harness 和 `tests/react/react_agent.py` 都是位于工具箱外侧的
集成测试客户端：模型调用 MiniChem，而不是 MiniChem 调用模型。它们不是 Action、Backend 或
运行依赖，也不会出现在 Catalog 中。

### Codex

在目标项目的 Codex MCP 配置中注册工具箱启动脚本，或直接运行隔离 harness：

```bash
PYTHONPATH=src:third_party/minichem_toolbox/src \
  bash third_party/minichem_toolbox/tests/harnesses/run_codex.sh
```

Harness 会为每次运行创建独立工作区，并通过 Codex 的 `mcp_servers.minichem_toolbox` 配置启动 MCP。

### Claude Code

```bash
PYTHONPATH=src:third_party/minichem_toolbox/src \
  bash third_party/minichem_toolbox/tests/harnesses/run_claude.sh
```

Harness 会生成独立 `.mcp.json`，只向 Claude 开放工作区文件工具和 `mcp__minichem_toolbox__*`。

### OpenCode

OpenCode 使用 OpenAI 兼容接口。先在不纳入 Git 的仓库根 `config.local.env` 中配置：

```bash
export OPENAI_API_KEY=<api-key>
export MINICHEM_AGENT_BASE_URL=<openai-compatible-base-url>
export MINICHEM_AGENT_MODEL=deepseek/deepseek-v4-flash
```

然后执行：

```bash
PYTHONPATH=src:third_party/minichem_toolbox/src \
  bash third_party/minichem_toolbox/tests/harnesses/run_opencode.sh
```

Harness 会生成独立 `opencode.json` 和 MCP 启动器。

### 直接 API ReAct 示例

Claude、Codex 和 OpenCode 不是工具箱的运行依赖。`tests/react/react_agent.py` 展示了一个最小的
ReAct Agent：模型通过 OpenAI 兼容 API 决定下一步，Python 程序直接调用 MiniChem 的
`search_actions`、`inspect_action` 和 `execute_action` 函数，不启动 MCP，也不经过任何 CLI
harness。

```bash
set -a
source config.local.env  # 或自行导出下面三个变量
set +a

export MINICHEM_AGENT_API_KEY="${OPENAI_API_KEY}"
export MINICHEM_AGENT_BASE_URL="https://api.deepseek.com/v1"
export MINICHEM_AGENT_MODEL="deepseek-v4-flash"

PYTHONPATH=src:third_party/minichem_toolbox/src \
  bash third_party/minichem_toolbox/tests/react/run_react.sh
```

`run_react.sh` 会检查标准解释器、自动加载仓库根目录中的 `config.local.env`，并将附加参数原样传递给
`react_agent.py`。

可用参数：

```text
--task TEXT             覆盖默认的水分子 xTB 单点能示例
--workspace PATH        指定隔离工作区
--base-url URL          OpenAI 兼容 API 根地址
--model NAME            API 接受的模型名称
--max-steps N           最大 ReAct 步数，默认 8
--timeout-seconds N     单次模型请求超时，默认 600 秒
```

运行记录默认保存在 `tests/results/react_<timestamp>_<pid>/`：

- `_sessions/react.jsonl`：每轮模型决策、API token 用量和工具 observation；
- `_tool_results/`、`_tool_artifacts/`：MiniChem 的工具调用记录；
- `outputs/`：化学计算产物；
- `report/final.md`：Agent 最终回答。

该示例只覆盖第一层预定义 Action，目的是清楚展示直接 API 接入方式。ARCHE 等自定义 workflow
可以复用同样的 Python 调用层，并按需继续接入原生软件作业和 Python 分析程序接口。

## 工作区隔离与记录

每次 harness 运行默认写入：

```text
tests/results/<cli>/<timestamp_pid>/
```

其中保存：

- `_sessions/chat.jsonl`：CLI 对话和事件流；
- `_sessions/*.stderr.log`：CLI 错误输出；
- `_tool_results/`：MCP 工具结果；
- `_tool_artifacts/`：工具调用输入和产物索引；
- `code/`、`data/`、`outputs/`、`report/`：该次任务的隔离工作区。

`tests/results/` 是运行产物，已被 Git 忽略，可随时删除。

## 迁移

推荐迁移整个 `third_party/minichem_toolbox/`，不要复制 Python 运行环境。目标机器应按仓库根目录的
部署约定提供标准解释器，并运行 `scripts/bootstrap_reproduction.sh` 检查环境。

需要单独确认：

1. `.mini_software_cache/` 中的授权软件可在目标机器合法使用；
2. 目标机器架构与缓存二进制兼容；
3. `.mini_model_cache/` 已复制，或允许重新下载；
4. Codex、Claude、OpenCode CLI 已在目标机器安装并完成各自认证。

`src/minichem_toolbox/vendor/arche_author_tools/` 受其目录内 `LICENSE` 约束，只允许学术和
非商业研究使用。迁移、复制或部署工具箱时必须保留该许可证；商业使用需要另行获得作者授权。
