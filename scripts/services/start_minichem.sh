#!/usr/bin/env bash
set -euo pipefail

# MiniChem 服务的公共底层 adapter：校验 profile、标准 Python、私有 runtime 和
# 固定资源后调用工具箱 MCP 启动脚本；上层 9011/9012 research_task 服务复用本脚本。
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TOOLBOX_ROOT="$ROOT/third_party/minichem_toolbox"
STARTUP="$TOOLBOX_ROOT/scripts/start_mcp.sh"
CONFIG_FILE="${RESEARCHCHEMBENCH_CONFIG:-$ROOT/config.local.env}"
STANDARD_ARCHE_PYTHON="$ROOT/.venv/bin/python"
# 旧 MINICHEM_PRIVATE_RUNTIME_ROOT 只在这个第三方服务启动 adapter 边界兼容；
# 当前 ARCHE 入口统一从 ARCHE_RUN_STORAGE_ROOT 传入私有运行根。
LOCKED_RUNTIME_ROOT="${ARCHE_RUN_STORAGE_ROOT:-${MINICHEM_PRIVATE_RUNTIME_ROOT:-$ROOT/local_data/runtime/case1_harness}}"
# MiniChem 作业的默认固定资源；测试或临时部署可通过下面的服务环境变量显式覆盖。
MINICHEM_DEFAULT_CPU_CORES=60
MINICHEM_DEFAULT_MEMORY_MB=320000
MINICHEM_DEFAULT_GPU_COUNT=1
LOCKED_PROFILE_WORKSPACE="${MINICHEM_PROFILE_WORKSPACE:-}"
LOCKED_PROFILE_HOST="${MINICHEM_PROFILE_HOST:-}"
LOCKED_PROFILE_PORT="${MINICHEM_PROFILE_PORT:-}"
readonly LOCKED_RUNTIME_ROOT LOCKED_PROFILE_WORKSPACE
readonly LOCKED_PROFILE_HOST LOCKED_PROFILE_PORT

if [[ ! -x "$STARTUP" ]]; then
  echo "MiniChem 启动脚本不存在：$STARTUP" >&2
  exit 1
fi

if [[ -f "$CONFIG_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$CONFIG_FILE"
  set +a
elif [[ -n "${RESEARCHCHEMBENCH_CONFIG:-}" ]]; then
  echo "指定的模型配置不存在：$CONFIG_FILE" >&2
  exit 1
fi

MINICHEM_RUNTIME_PYTHON="${MINICHEM_RUNTIME_PYTHON:-${ARCHE_PYTHON:-$STANDARD_ARCHE_PYTHON}}"
if [[ "$MINICHEM_RUNTIME_PYTHON" != "$STANDARD_ARCHE_PYTHON" ]]; then
  echo "MiniChem 必须使用标准 Python：$STANDARD_ARCHE_PYTHON" >&2
  exit 1
fi
if [[ ! -x "$MINICHEM_RUNTIME_PYTHON" ]]; then
  echo "标准 Python 不可执行：$MINICHEM_RUNTIME_PYTHON；请先准备 arche 环境" >&2
  exit 1
fi

private_runtime_directory() {
  local candidate="$1"
  local runtime_root="$2"
  PYTHONPATH="$ROOT/src:$TOOLBOX_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$MINICHEM_RUNTIME_PYTHON" - "$candidate" "$runtime_root" <<'PY'
import sys

from minichem_mcp_tools.workspace import ensure_private_runtime_directory

try:
    print(
        ensure_private_runtime_directory(
            sys.argv[1],
            runtime_root=sys.argv[2],
        )
    )
except Exception:
    raise SystemExit(1)
PY
}

RUNTIME_ROOT="$LOCKED_RUNTIME_ROOT"
if [[ "$RUNTIME_ROOT" != /* ]]; then
  RUNTIME_ROOT="$ROOT/$RUNTIME_ROOT"
fi
if ! RUNTIME_ROOT="$(private_runtime_directory "$RUNTIME_ROOT" "$RUNTIME_ROOT")"; then
  echo "MiniChem 私有 runtime 根无效；服务未启动" >&2
  exit 1
fi
# 源码目录不是运行资料根；允许仓库内的 local_data/runtime 和仓库外临时根。
case "$RUNTIME_ROOT" in
  "$ROOT"|"$ROOT"/*)
    case "$RUNTIME_ROOT" in
      "$ROOT/local_data/runtime"|"$ROOT/local_data/runtime"/*) ;;
      *)
        echo "MiniChem 私有 runtime 根不得位于源码目录" >&2
        exit 1
        ;;
    esac
    ;;
esac

if [[ -n "$LOCKED_PROFILE_WORKSPACE" ]]; then
  WORKSPACE="$LOCKED_PROFILE_WORKSPACE"
else
  WORKSPACE="${MINICHEM_SERVICE_WORKSPACE:-$RUNTIME_ROOT/minichem}"
fi
if [[ -n "$LOCKED_PROFILE_HOST" ]]; then
  HOST="$LOCKED_PROFILE_HOST"
else
  HOST="${MINICHEM_MCP_HOST:-127.0.0.1}"
fi
if [[ -n "$LOCKED_PROFILE_PORT" ]]; then
  PORT="$LOCKED_PROFILE_PORT"
else
  # 直接调用本通用入口时，9010 仅是 legacy/general 兼容默认；Case 1 必须使用
  # start_catalog_bootstrap_mcp.sh 或 start_investigation_minichem.sh 的固定端口。
  PORT="${MINICHEM_MCP_PORT:-9010}"
fi
if [[ "$WORKSPACE" != /* ]]; then
  WORKSPACE="$ROOT/$WORKSPACE"
fi

if ! WORKSPACE="$(private_runtime_directory "$WORKSPACE" "$RUNTIME_ROOT")"; then
  echo "MiniChem 工作区必须是 runtime 根内的私有目录；服务未启动" >&2
  exit 1
fi
export MINICHEM_PRIVATE_RUNTIME_ROOT="$RUNTIME_ROOT"
# 第三方 MiniChem server 仍读取旧 workspace 名称；映射只在此服务 adapter 边界
# 生效，ARCHE 生产配置使用 MINICHEM_SERVICE_WORKSPACE。
export MINICHEM_SERVICE_WORKSPACE="$WORKSPACE"
export MINICHEM_MCP_WORKSPACE="$MINICHEM_SERVICE_WORKSPACE"
export RESEARCHCHEMBENCH_WORKSPACE="$WORKSPACE"
export MINICHEM_RUNTIME_PYTHON
export ARCHE_PYTHON="$MINICHEM_RUNTIME_PYTHON"

# 默认预算包含 1 张 GPU，供目录中声明需要 GPU 的能力使用。显式环境变量仍可
# 覆盖默认值，但必须与 ARCHE 冻结资源政策一致。
export RESEARCHCHEMBENCH_AVAILABLE_CPU_CORES="${RESEARCHCHEMBENCH_AVAILABLE_CPU_CORES:-$MINICHEM_DEFAULT_CPU_CORES}"
export RESEARCHCHEMBENCH_AVAILABLE_MEMORY_MB="${RESEARCHCHEMBENCH_AVAILABLE_MEMORY_MB:-$MINICHEM_DEFAULT_MEMORY_MB}"
export RESEARCHCHEMBENCH_AVAILABLE_GPU_COUNT="${RESEARCHCHEMBENCH_AVAILABLE_GPU_COUNT:-$MINICHEM_DEFAULT_GPU_COUNT}"
export RESEARCHCHEMBENCH_COMPUTE_ACTION_TIMEOUT_SECONDS="${RESEARCHCHEMBENCH_COMPUTE_ACTION_TIMEOUT_SECONDS:-21600}"

# programmable_analysis 的 Python child 只使用仓库固定的 Codex 0.149.0 bwrap。
# 这里显式注入绝对路径和固定摘要；supervisor 仍会在每个 job 启动前重新核验，
# 因此 bwrap 缺失或被替换时只拒绝 Python 作业，不影响 native/TS/Gaussian。
MINICHEM_PROGRAMMABLE_ANALYSIS_BWRAP_RUNTIME_ROOT="${MINICHEM_PROGRAMMABLE_ANALYSIS_BWRAP_RUNTIME_ROOT:-$ROOT/local_data/runtime/codex_cli/0.149.0/linux-x64}"
MINICHEM_PROGRAMMABLE_ANALYSIS_BWRAP="${MINICHEM_PROGRAMMABLE_ANALYSIS_BWRAP:-$MINICHEM_PROGRAMMABLE_ANALYSIS_BWRAP_RUNTIME_ROOT/vendor/x86_64-unknown-linux-musl/codex-resources/bwrap}"
MINICHEM_PROGRAMMABLE_ANALYSIS_BWRAP_SHA256="${MINICHEM_PROGRAMMABLE_ANALYSIS_BWRAP_SHA256:-7df960565a0dece99240ea4b9d0e011307817f9f3b73176c7b71fda44fe84765}"
export MINICHEM_PROGRAMMABLE_ANALYSIS_BWRAP_RUNTIME_ROOT
export MINICHEM_PROGRAMMABLE_ANALYSIS_BWRAP
export MINICHEM_PROGRAMMABLE_ANALYSIS_BWRAP_SHA256

# 在启动服务和冻结公开目录前校验 Gaussian 专用资源策略；校验程序不输出配置值。
if ! PYTHONPATH="$ROOT/src:$TOOLBOX_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$MINICHEM_RUNTIME_PYTHON" -c \
  'from minichem_toolbox.resource_budget import gaussian_resource_policy; gaussian_resource_policy()' \
  >/dev/null 2>&1; then
  echo "Gaussian 资源策略配置无效；服务未启动" >&2
  exit 1
fi

exec "$STARTUP" \
  --transport streamable_http \
  --host "$HOST" \
  --port "$PORT" \
  "$@"
