#!/usr/bin/env bash
set -euo pipefail

# research_task 的 Investigation MiniChem 服务 adapter：固定启动带 scope 认证的
# 9012 服务，并绑定调用方指定的私有 runtime、scope registry 和共享调度目录。
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STARTUP="$ROOT/scripts/services/start_minichem.sh"
STANDARD_ARCHE_PYTHON="$ROOT/.venv/bin/python"
# `MINICHEM_PRIVATE_RUNTIME_ROOT` 仅作为第三方服务 adapter 的历史输入兼容；
# ARCHE 当前公开配置优先使用 ARCHE_RUN_STORAGE_ROOT。
RUNTIME_ROOT="${ARCHE_RUN_STORAGE_ROOT:-${MINICHEM_PRIVATE_RUNTIME_ROOT:-$ROOT/local_data/runtime/case1_harness}}"

# Investigation 服务必须显式启用 capability scope；注册表和共享调度目录
# 只保存于被忽略的 local_data/runtime，路径本身不包含 capability 明文。
REGISTRY="${MINICHEM_SCOPE_REGISTRY_ROOT:-$RUNTIME_ROOT/minichem_scope_registry}"
SCHEDULER="${MINICHEM_SHARED_SCHEDULER_ROOT:-$RUNTIME_ROOT/minichem_scheduler}"
WORKSPACE="${MINICHEM_SERVICE_WORKSPACE:-$RUNTIME_ROOT/minichem_service}"
ENDPOINT="${MINICHEM_INVESTIGATION_ENDPOINT:-http://127.0.0.1:9012/mcp}"
PORT="9012"
HOST="127.0.0.1"

if [[ ! -x "$STANDARD_ARCHE_PYTHON" ]]; then
  echo "标准 Python 不可执行；Investigation 服务未启动" >&2
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
  echo "Investigation MiniChem 私有 runtime 根无效；服务未启动" >&2
  exit 1
fi
case "$RUNTIME_ROOT" in
  "$ROOT"|"$ROOT"/*)
    case "$RUNTIME_ROOT" in
      "$ROOT/local_data/runtime"|"$ROOT/local_data/runtime"/*) ;;
      *)
        echo "Investigation MiniChem 私有 runtime 根不得位于源码目录" >&2
        exit 1
        ;;
    esac
    ;;
esac

if [[ "$ENDPOINT" != "http://127.0.0.1:9012/mcp" ]]; then
  echo "MINICHEM_INVESTIGATION_ENDPOINT 只允许 http://127.0.0.1:9012/mcp；服务未启动" >&2
  exit 1
fi

for path_name in REGISTRY SCHEDULER WORKSPACE; do
  path_value="${!path_name}"
  if [[ "$path_value" != /* ]]; then
    path_value="$ROOT/$path_value"
  fi
  if ! path_value="$(private_runtime_directory "$path_value" "$RUNTIME_ROOT")"; then
    echo "Investigation MiniChem 私有目录必须位于 runtime 根内；服务未启动" >&2
    exit 1
  fi
  printf -v "$path_name" '%s' "$path_value"
done

umask 077
export MINICHEM_PRIVATE_RUNTIME_ROOT="$RUNTIME_ROOT"
export MINICHEM_SERVICE_WORKSPACE="$WORKSPACE"
# 第三方 MiniChem server 仍读取旧 workspace 名称；仅在本 adapter 边界映射。
export MINICHEM_MCP_WORKSPACE="$MINICHEM_SERVICE_WORKSPACE"
export MINICHEM_PROFILE_WORKSPACE="$WORKSPACE"
export MINICHEM_PROFILE_HOST="$HOST"
export MINICHEM_PROFILE_PORT="$PORT"

exec bash "$STARTUP" \
  "$@" \
  --profile investigation \
  --scope-auth \
  --scope-registry "$REGISTRY" \
  --scheduler-root "$SCHEDULER" \
  --transport streamable_http \
  --host "$HOST" \
  --port "$PORT"
