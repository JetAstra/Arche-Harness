#!/usr/bin/env bash
set -euo pipefail

# 通用 `research_task` 入口：核对标准环境和固定服务地址，然后创建
# 新 research_task、停在 hypothesis-confirmation、继续确认后的任务或恢复原任务。
# 9011/9012 只在首个真实 Investigation Session 中按 scope 检查；候选、确认和
# 已有作业恢复不做全局探测。
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STANDARD_ARCHE_PYTHON="$ROOT/.venv/bin/python"
PYTHON="$STANDARD_ARCHE_PYTHON"
VENDORED_CODEX_BIN="$ROOT/third_party/codex_cli/0.149.0/linux-x64/codex"
CONFIG_FILE="${RESEARCHCHEMBENCH_CONFIG:-$ROOT/config.local.env}"

# 运行参数区：研究员可直接编辑这些值，命令行同名参数会覆盖它们。
QUESTION=""
# 仅在调用方需要补充通用计算单元或初始结构时设置；Case 1 作者输入自带冻结计算单元。
SCIENTIFIC_INPUT_JSON=""
CASE_PROFILE=""
CODEX_MODEL_NAME="gpt-5.6-sol"
CODEX_REASONING_EFFORT="xhigh"
# Codex 认证模式必须由启动配置显式选择；默认 API 保留现有 provider 行为。
# local_login 只在 app-server 生命周期内读取这个 source，脚本不会读取或输出其正文。
CODEX_AUTH_MODE="${ARCHE_CODEX_AUTH_MODE:-api}"
CODEX_AUTH_SOURCE="${ARCHE_CODEX_AUTH_SOURCE:-/root/.codex/auth.json}"
CODEX_ALLOW_PROVIDER_MIGRATION="${ARCHE_CODEX_ALLOW_PROVIDER_MIGRATION:-0}"
PDF_DIR=""
INDEX_DIR=""
PAPER_MANIFEST=""
NO_SEARCH=0
RUNTIME_ROOT="${ARCHE_RUN_STORAGE_ROOT:-$ROOT/local_data/runtime/case1_harness}"

if [[ "$PYTHON" != "$STANDARD_ARCHE_PYTHON" || ! -x "$PYTHON" ]]; then
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

if [[ "$RUNTIME_ROOT" != /* ]]; then RUNTIME_ROOT="$ROOT/$RUNTIME_ROOT"; fi

private_runtime_directory() {
  local candidate="$1" runtime_root="$2"
  PYTHONPATH="$ROOT/src:$ROOT/third_party/minichem_toolbox/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON" - "$candidate" "$runtime_root" <<'PY'
import sys
from minichem_mcp_tools.workspace import ensure_private_runtime_directory
try:
    print(ensure_private_runtime_directory(sys.argv[1], runtime_root=sys.argv[2]))
except Exception:
    raise SystemExit(1)
PY
}

if ! RUNTIME_ROOT="$(private_runtime_directory "$RUNTIME_ROOT" "$RUNTIME_ROOT")"; then
  echo "端到端私有 runtime 根无效；流程未启动" >&2
  exit 1
fi
case "$RUNTIME_ROOT" in
  "$ROOT"|"$ROOT"/*)
    case "$RUNTIME_ROOT" in
      "$ROOT/local_data/runtime"|"$ROOT/local_data/runtime"/*) ;;
      *) echo "端到端私有 runtime 根不得位于源码目录；流程未启动" >&2; exit 1 ;;
    esac
    ;;
esac

ARCHE_RUN_STORAGE_ROOT="${ARCHE_RUN_STORAGE_ROOT:-$RUNTIME_ROOT}"
ARCHE_STAGE_HARNESS_WORKSPACE_ROOT="${ARCHE_STAGE_HARNESS_WORKSPACE_ROOT:-$RUNTIME_ROOT/harness_workspaces}"
ARCHE_STAGE_HARNESS_TRACE_ROOT="${ARCHE_STAGE_HARNESS_TRACE_ROOT:-$RUNTIME_ROOT/harness_traces}"
ARCHE_CODEX_HOME="${ARCHE_CODEX_HOME:-$RUNTIME_ROOT/codex_home}"
MINICHEM_SCOPE_REGISTRY_ROOT="${MINICHEM_SCOPE_REGISTRY_ROOT:-$RUNTIME_ROOT/minichem_scope_registry}"
MINICHEM_SHARED_SCHEDULER_ROOT="${MINICHEM_SHARED_SCHEDULER_ROOT:-$RUNTIME_ROOT/minichem_scheduler}"
MINICHEM_SERVICE_WORKSPACE="${MINICHEM_SERVICE_WORKSPACE:-$RUNTIME_ROOT/minichem_service}"
WORK_DIR="$RUNTIME_ROOT/controller"

for path_name in ARCHE_RUN_STORAGE_ROOT ARCHE_STAGE_HARNESS_WORKSPACE_ROOT \
  ARCHE_STAGE_HARNESS_TRACE_ROOT ARCHE_CODEX_HOME MINICHEM_SCOPE_REGISTRY_ROOT \
  MINICHEM_SHARED_SCHEDULER_ROOT MINICHEM_SERVICE_WORKSPACE WORK_DIR; do
  path_value="${!path_name}"
  if [[ "$path_value" != /* ]]; then path_value="$ROOT/$path_value"; fi
  if ! path_value="$(private_runtime_directory "$path_value" "$RUNTIME_ROOT")"; then
    echo "端到端运行目录必须位于私有 runtime 根内；流程未启动" >&2
    exit 1
  fi
  printf -v "$path_name" '%s' "$path_value"
done

export ARCHE_RUN_STORAGE_ROOT ARCHE_STAGE_HARNESS_WORKSPACE_ROOT ARCHE_STAGE_HARNESS_TRACE_ROOT
export ARCHE_CODEX_HOME MINICHEM_SCOPE_REGISTRY_ROOT MINICHEM_SHARED_SCHEDULER_ROOT
export MINICHEM_SERVICE_WORKSPACE
export ARCHE_CODEX_AUTH_MODE="$CODEX_AUTH_MODE"
if [[ "${CODEX_AUTH_MODE,,}" == "api" ]]; then
  # API 新任务只使用 private config 中的 Key/Base URL，不继承本机登录态。
  unset ARCHE_CODEX_AUTH_SOURCE ARCHE_CODEX_LOCAL_LOGIN
else
  export ARCHE_CODEX_AUTH_SOURCE="$CODEX_AUTH_SOURCE"
fi
export ARCHE_CODEX_ALLOW_PROVIDER_MIGRATION="$CODEX_ALLOW_PROVIDER_MIGRATION"
if [[ -z "${ARCHE_CODEX_MODEL_NAME:-}" ]]; then
  export ARCHE_CODEX_MODEL_NAME="$CODEX_MODEL_NAME"
fi
if [[ -z "${ARCHE_CODEX_REASONING_EFFORT:-}" ]]; then
  export ARCHE_CODEX_REASONING_EFFORT="$CODEX_REASONING_EFFORT"
fi
export PYTHONPATH="$ROOT/src:$ROOT/third_party/minichem_toolbox/src${PYTHONPATH:+:$PYTHONPATH}"

# 默认动作是创建标准 research_task；继续确认、读取确认视图和恢复时，调用方把
# 六个正式动作之一作为第一个参数传入，后续参数原样传给 Controller。
ACTION="create-standard-research-task"
case "${1:-}" in
  create-standard-research-task|create-author-hypothesis-task|create-top5-confirmation-task|\
  get-hypothesis-confirmation|submit-hypothesis-confirmation-and-continue|resume-research-task)
    ACTION="$1"
    shift
    ;;
esac

if [[ ! -x "$VENDORED_CODEX_BIN" || -L "$VENDORED_CODEX_BIN" ]]; then
  echo "仓库随附的 Codex 0.149.0 wrapper 不可用；流程未启动" >&2
  exit 1
fi
CONFIGURED_CODEX_BIN="${ARCHE_CODEX_BIN:-$VENDORED_CODEX_BIN}"
if [[ "$CONFIGURED_CODEX_BIN" != /* ]]; then CONFIGURED_CODEX_BIN="$ROOT/$CONFIGURED_CODEX_BIN"; fi
if [[ ! -x "$CONFIGURED_CODEX_BIN" || -L "$CONFIGURED_CODEX_BIN" ]]; then
  echo "ARCHE_CODEX_BIN 不可执行或是符号链接；流程未启动" >&2
  exit 1
fi
export ARCHE_CODEX_BIN="$CONFIGURED_CODEX_BIN"

EXPECTED_CATALOG_ENDPOINT="http://127.0.0.1:9011/mcp"
EXPECTED_INVESTIGATION_ENDPOINT="http://127.0.0.1:9012/mcp"
if [[ "${MINICHEM_CATALOG_ENDPOINT:-$EXPECTED_CATALOG_ENDPOINT}" != "$EXPECTED_CATALOG_ENDPOINT" || \
  "${MINICHEM_INVESTIGATION_ENDPOINT:-$EXPECTED_INVESTIGATION_ENDPOINT}" != "$EXPECTED_INVESTIGATION_ENDPOINT" ]]; then
  echo "端到端 MiniChem 必须使用 loopback 的 9011 catalog 和 9012 Investigation 服务；流程未启动" >&2
  exit 1
fi
export MINICHEM_CATALOG_ENDPOINT="$EXPECTED_CATALOG_ENDPOINT"
export MINICHEM_INVESTIGATION_ENDPOINT="$EXPECTED_INVESTIGATION_ENDPOINT"

argument_has_option() {
  local option="$1"; shift; local argument
  for argument in "$@"; do
    [[ "$argument" == "$option" || "$argument" == "$option="* ]] && return 0
  done
  return 1
}
has_help_action=0; has_question_option=0
for argument in "$@"; do
  case "$argument" in
    --question|--question=*|-q|-q=*) has_question_option=1 ;;
    --help|-h) has_help_action=1 ;;
  esac
done

trim_question() {
  local value="$1"
  value="${value#${value%%[![:space:]]*}}"
  value="${value%${value##*[![:space:]]}}"
  printf '%s' "$value"
}
if [[ "$has_question_option" == "1" ]]; then
  arguments=("$@")
  for ((index = 0; index < ${#arguments[@]}; index++)); do
    case "${arguments[index]}" in
      --question=*) QUESTION="${arguments[index]#--question=}" ;;
      -q=*) QUESTION="${arguments[index]#-q=}" ;;
      --question|-q)
        if (( index + 1 < ${#arguments[@]} )); then QUESTION="${arguments[index + 1]}"; else QUESTION=""; fi ;;
    esac
  done
fi
QUESTION="$(trim_question "$QUESTION")"
if [[ "$ACTION" == "create-standard-research-task" && "$has_help_action" == "0" && -z "$QUESTION" ]]; then
  echo "新任务 question 去除首尾空白后不能为空；请编辑脚本顶部 QUESTION 或提供 --question" >&2
  exit 1
fi

CONTROLLER_ARGS=()
if [[ "$ACTION" == "create-standard-research-task" && "$has_help_action" == "0" ]]; then
  if ! argument_has_option --question "$@" && ! argument_has_option -q "$@"; then CONTROLLER_ARGS+=(--question "$QUESTION"); fi
  if [[ -n "$SCIENTIFIC_INPUT_JSON" ]] && ! argument_has_option --scientific-input-json "$@"; then
    CONTROLLER_ARGS+=(--scientific-input-json "$SCIENTIFIC_INPUT_JSON")
  fi
  if [[ -n "$CASE_PROFILE" ]] && ! argument_has_option --case-profile "$@"; then
    CONTROLLER_ARGS+=(--case-profile "$CASE_PROFILE")
  fi
  if [[ -n "$PDF_DIR" ]] && ! argument_has_option --pdf-dir "$@"; then CONTROLLER_ARGS+=(--pdf-dir "$PDF_DIR"); fi
  if [[ -n "$INDEX_DIR" ]] && ! argument_has_option --index-dir "$@"; then CONTROLLER_ARGS+=(--index-dir "$INDEX_DIR"); fi
  if [[ -n "$PAPER_MANIFEST" ]] && ! argument_has_option --paper-search-manifest "$@"; then
    CONTROLLER_ARGS+=(--paper-search-manifest "$PAPER_MANIFEST")
  fi
  case "$NO_SEARCH" in
    ""|0|false|no) ;;
    1|true|yes) if ! argument_has_option --no-search "$@"; then CONTROLLER_ARGS+=(--no-search); fi ;;
    *) echo "脚本顶部 NO_SEARCH 只能是 0/1 或 false/true；流程未启动" >&2; exit 1 ;;
  esac
  if ! argument_has_option --work-dir "$@"; then CONTROLLER_ARGS+=(--work-dir "$WORK_DIR"); fi
fi
CONTROLLER_ARGS+=("$@")


exec "$PYTHON" -m chemistry_multiagent.controllers.chemistry_multiagent_controller \
  "$ACTION" "${CONTROLLER_ARGS[@]}"
