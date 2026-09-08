#!/usr/bin/env bash
set -euo pipefail

# research_task 的 Catalog 服务 adapter：固定启动只读 `catalog_bootstrap` profile
# 和 9011 端口。脚本会写入指定的私有 runtime，不提交或执行化学计算。
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STARTUP="$ROOT/scripts/services/start_minichem.sh"
STANDARD_ARCHE_PYTHON="$ROOT/.venv/bin/python"
# `MINICHEM_PRIVATE_RUNTIME_ROOT` 仅作为第三方服务 adapter 的历史输入兼容；
# ARCHE 当前公开配置优先使用 ARCHE_RUN_STORAGE_ROOT。
RUNTIME_ROOT="${ARCHE_RUN_STORAGE_ROOT:-${MINICHEM_PRIVATE_RUNTIME_ROOT:-$ROOT/local_data/runtime/case1_harness}}"
WORKSPACE="${MINICHEM_CATALOG_BOOTSTRAP_WORKSPACE:-$RUNTIME_ROOT/minichem_catalog_bootstrap}"
HOST="127.0.0.1"
PORT="${MINICHEM_CATALOG_BOOTSTRAP_MCP_PORT:-9011}"

if [[ ! -x "$STARTUP" ]]; then
  echo "MiniChem 启动脚本不存在：$STARTUP" >&2
  exit 1
fi

if [[ ! -x "$STANDARD_ARCHE_PYTHON" ]]; then
  echo "标准 Python 不可执行；catalog bootstrap 服务未启动" >&2
  exit 1
fi

private_runtime_directory() {
  local candidate="$1"
  local runtime_root="$2"
  PYTHONPATH="$ROOT/src:$ROOT/third_party/minichem_toolbox/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$STANDARD_ARCHE_PYTHON" - "$candidate" "$runtime_root" <<'PY'
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

if [[ "$RUNTIME_ROOT" != /* ]]; then
  RUNTIME_ROOT="$ROOT/$RUNTIME_ROOT"
fi
if ! RUNTIME_ROOT="$(private_runtime_directory "$RUNTIME_ROOT" "$RUNTIME_ROOT")"; then
  echo "catalog bootstrap 私有 runtime 根无效；服务未启动" >&2
  exit 1
fi
case "$RUNTIME_ROOT" in
  "$ROOT"|"$ROOT"/*)
    case "$RUNTIME_ROOT" in
      "$ROOT/local_data/runtime"|"$ROOT/local_data/runtime"/*) ;;
      *)
        echo "catalog bootstrap 私有 runtime 根不得位于源码目录" >&2
        exit 1
        ;;
    esac
    ;;
esac

if [[ "$PORT" != "9011" ]]; then
  echo "catalog bootstrap 只允许端口 9011；服务未启动" >&2
  exit 1
fi

if [[ "$WORKSPACE" != /* ]]; then
  WORKSPACE="$ROOT/$WORKSPACE"
fi
if ! WORKSPACE="$(private_runtime_directory "$WORKSPACE" "$RUNTIME_ROOT")"; then
  echo "catalog bootstrap 工作区必须位于 runtime 根内；服务未启动" >&2
  exit 1
fi

umask 077
export MINICHEM_PRIVATE_RUNTIME_ROOT="$RUNTIME_ROOT"
export MINICHEM_SERVICE_WORKSPACE="$WORKSPACE"
# 第三方 MiniChem server 仍读取旧 workspace 名称；仅在本 adapter 边界映射，
# ARCHE research_task 配置和公开入口使用 MINICHEM_SERVICE_WORKSPACE。
export MINICHEM_MCP_WORKSPACE="$MINICHEM_SERVICE_WORKSPACE"
export MINICHEM_PROFILE_WORKSPACE="$WORKSPACE"
export MINICHEM_PROFILE_HOST="$HOST"
export MINICHEM_PROFILE_PORT="$PORT"

# 将 profile、传输协议和监听地址放在用户参数之后，确保脚本入口始终是只读 HTTP
# bootstrap；服务端仍会再次校验这些边界，防止 CLI 参数错配时降级到完整服务。
exec bash "$STARTUP" \
  "$@" \
  --profile catalog_bootstrap \
  --transport streamable_http \
  --host "$HOST" \
  --port "$PORT"
