#!/usr/bin/env bash
set -euo pipefail

# 通用 `research_task` Top 5 入口：执行 `retrieval`、
# `research-query-formulation`、`hypothesis-generation` 和 `hypothesis-selection`
# 后停在 `hypothesis-confirmation`；不创建 `hypothesis-investigation` session，
# 也不要求启动 MiniChem 9011/9012 服务。后续由端到端入口继续或恢复同一任务。
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ARCHE_PYTHON:-$ROOT/.venv/bin/python}"
CONFIG_FILE="${RESEARCHCHEMBENCH_CONFIG:-$ROOT/config.local.env}"

if [[ "$PYTHON" != "$ROOT/.venv/bin/python" || ! -x "$PYTHON" ]]; then
  echo "必须使用仓库标准 Python；流程未启动" >&2
  exit 1
fi
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "模型配置不存在或不是普通文件；流程未启动" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "$CONFIG_FILE"
set +a

RUNTIME_ROOT="${ARCHE_RUN_STORAGE_ROOT:-$ROOT/local_data/runtime/arche_harness}"
if [[ "$RUNTIME_ROOT" != /* ]]; then
  RUNTIME_ROOT="$ROOT/$RUNTIME_ROOT"
fi
# 模型和请求参数由实验启动前加载的本地配置决定，本入口不覆盖它们。
export ARCHE_RUN_STORAGE_ROOT="${ARCHE_RUN_STORAGE_ROOT:-$RUNTIME_ROOT}"
export ARCHE_STAGE_HARNESS_WORKSPACE_ROOT="${ARCHE_STAGE_HARNESS_WORKSPACE_ROOT:-$RUNTIME_ROOT/harness_workspaces}"
export ARCHE_STAGE_HARNESS_TRACE_ROOT="${ARCHE_STAGE_HARNESS_TRACE_ROOT:-$RUNTIME_ROOT/harness_traces}"
export MINICHEM_SERVICE_WORKSPACE="${MINICHEM_SERVICE_WORKSPACE:-$RUNTIME_ROOT/minichem_service}"
export MINICHEM_SCOPE_REGISTRY_ROOT="${MINICHEM_SCOPE_REGISTRY_ROOT:-$RUNTIME_ROOT/minichem_scope_registry}"
export MINICHEM_INVESTIGATION_ENDPOINT="${MINICHEM_INVESTIGATION_ENDPOINT:-http://127.0.0.1:9012/mcp}"
export PYTHONPATH="$ROOT/src:$ROOT/third_party/minichem_toolbox/src${PYTHONPATH:+:$PYTHONPATH}"

exec "$PYTHON" -m chemistry_multiagent.controllers.chemistry_multiagent_controller \
  create-top5-confirmation-task "$@"
