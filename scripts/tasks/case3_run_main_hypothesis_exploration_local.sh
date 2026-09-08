#!/usr/bin/env bash
set -Eeuo pipefail

# Case 3 本机探索入口：为本次新 research_task 建立独立 runtime，并把已经清理、
# 冻结的五个输入文件交给 Harness，由 Harness 复制到受限工作区后再供模型读取。
# 任务完成前在 hypothesis-confirmation 等待研究员确认，之后可继续同一任务。
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
PYTHONPATH_VALUE="$ROOT/src:$ROOT/third_party/minichem_toolbox/src"
VENDORED_CODEX_BIN="$ROOT/third_party/codex_cli/0.149.0/linux-x64/codex"
CONFIG_FILE="$(printenv RESEARCHCHEMBENCH_CONFIG 2>/dev/null || true)"
if [[ -z "$CONFIG_FILE" ]]; then CONFIG_FILE="$ROOT/config.local.env"; fi

PREPARED_INPUT_DIR="${CASE3_PREPARED_INPUT_DIR:-$ROOT/local_data/runtime/case3_main_hypothesis_exploration_inputs}"
RUNTIME_PARENT="$ROOT/local_data/runtime/case3_main_hypothesis_exploration_local"
CPU_CORES="${CASE3_CPU_CORES:-60}"
MEMORY_MB="${CASE3_MEMORY_MB:-320000}"
GPU_COUNT="${CASE3_GPU_COUNT:-1}"
CATALOG_PORT=9011
INVESTIGATION_PORT=9012
RUN_NAME="${CASE3_RUN_NAME:-}"
DRY_RUN=0

EXPECTED_PREPARED_FILES=(origin.gjf ortho.gjf meta.gjf para.gjf dihedral_atoms.json)

usage() {
  cat <<'EOF'
用法：scripts/tasks/case3_run_main_hypothesis_exploration_local.sh [选项]
  --case3-prepared-input-dir DIR  已清理的 Case 3 五个输入文件所在目录
  --cpu N, --cpu-cores N       本次 MiniChem 可用 CPU 核数；默认 60
  --memory N, --memory-mb N    本次 MiniChem 可用内存（MB）；默认 320000
  --gpu N, --gpu-count N       本次 MiniChem 可用 GPU 数；默认 1
  --run-name NAME              本次运行目录名；默认使用 UTC 时间和 PID
  --dry-run                    只显示脱敏配置、路径和控制器参数，不启动服务
  -h, --help                   显示帮助
EOF
}

fail_usage() {
  echo "$1" >&2
  usage >&2
  exit 2
}

check_positive_integer() {
  if [[ ! "$2" =~ ^[1-9][0-9]*$ ]]; then
    fail_usage "$1 必须是正整数"
  fi
}

check_nonnegative_integer() {
  if [[ ! "$2" =~ ^(0|[1-9][0-9]*)$ ]]; then
    fail_usage "$1 必须是非负整数"
  fi
}

while (($# > 0)); do
  case "$1" in
    --case3-prepared-input-dir)
      (($# >= 2)) || fail_usage "$1 需要一个值"
      PREPARED_INPUT_DIR="$2"; shift 2 ;;
    --case3-prepared-input-dir=*) PREPARED_INPUT_DIR="${1#*=}"; shift ;;
    --cpu|--cpu-cores)
      (($# >= 2)) || fail_usage "$1 需要一个值"
      CPU_CORES="$2"; shift 2 ;;
    --cpu=*|--cpu-cores=*) CPU_CORES="${1#*=}"; shift ;;
    --memory|--memory-mb)
      (($# >= 2)) || fail_usage "$1 需要一个值"
      MEMORY_MB="$2"; shift 2 ;;
    --memory=*|--memory-mb=*) MEMORY_MB="${1#*=}"; shift ;;
    --gpu|--gpu-count)
      (($# >= 2)) || fail_usage "$1 需要一个值"
      GPU_COUNT="$2"; shift 2 ;;
    --gpu=*|--gpu-count=*) GPU_COUNT="${1#*=}"; shift ;;
    --run-name)
      (($# >= 2)) || fail_usage "$1 需要一个值"
      RUN_NAME="$2"; shift 2 ;;
    --run-name=*) RUN_NAME="${1#*=}"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) fail_usage "未知参数：$1" ;;
  esac
done

check_positive_integer "CPU 核数" "$CPU_CORES"
check_positive_integer "内存 MB" "$MEMORY_MB"
check_nonnegative_integer "GPU 数" "$GPU_COUNT"
if [[ "$PREPARED_INPUT_DIR" != /* ]]; then
  PREPARED_INPUT_DIR="$ROOT/$PREPARED_INPUT_DIR"
fi

validate_prepared_input_directory() {
  local expected candidate
  if [[ ! -d "$PREPARED_INPUT_DIR" || -L "$PREPARED_INPUT_DIR" ]]; then
    echo "Case 3 prepared input 目录必须是非符号链接目录：$PREPARED_INPUT_DIR" >&2
    exit 1
  fi
  for expected in "${EXPECTED_PREPARED_FILES[@]}"; do
    candidate="$PREPARED_INPUT_DIR/$expected"
    if [[ ! -f "$candidate" || -L "$candidate" ]]; then
      echo "Case 3 prepared input 缺少普通文件：$candidate" >&2
      exit 1
    fi
  done
}

validate_prepared_input_directory

if [[ -z "$RUN_NAME" ]]; then
  RUN_NAME="case3-$(date -u +%Y%m%dT%H%M%SZ)-$$"
fi
if [[ ! "$RUN_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]]; then
  fail_usage "run name 只能包含 ASCII 字母、数字、点、下划线和连字符"
fi

if [[ "$PYTHON" != "$ROOT/.venv/bin/python" || ! -x "$PYTHON" ]]; then
  echo "必须使用仓库标准 Python；流程未启动" >&2
  exit 1
fi
if [[ ! -x "$VENDORED_CODEX_BIN" || -L "$VENDORED_CODEX_BIN" ]]; then
  echo "仓库随附的 Codex 0.149.0 wrapper 不可用；流程未启动" >&2
  exit 1
fi
if [[ "$CONFIG_FILE" != /* ]]; then CONFIG_FILE="$ROOT/$CONFIG_FILE"; fi
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "模型配置不存在或不是普通文件；流程未启动" >&2
  exit 1
fi

# 读取登记的冻结题面和语料路径；只读，不创建 runtime，也不调用模型。
read_case3_frozen_inputs() {
  PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON" - "$ROOT" <<'PY'
from pathlib import Path
import base64
import sys

from chemistry_multiagent.controllers.case23_reproduction import load_frozen_case23_corpus

frozen = load_frozen_case23_corpus("case3", repository_root=Path(sys.argv[1]))
print("QUESTION_B64\t" + base64.b64encode(frozen.question.encode("utf-8")).decode("ascii"))
print("MANIFEST\t" + str(frozen.manifest_path))
print("PDF_DIR\t" + str(frozen.pdf_dir))
print("INDEX_DIR\t" + str(frozen.index_dir))
print("TARGET\t" + str(frozen.target_papers))
PY
}

CASE3_QUESTION_B64=""
PAPER_MANIFEST=""
PDF_DIR=""
INDEX_DIR=""
TARGET_PAPERS=""
while IFS=$'\t' read -r input_key input_value; do
  case "$input_key" in
    QUESTION_B64) CASE3_QUESTION_B64="$input_value" ;;
    MANIFEST) PAPER_MANIFEST="$input_value" ;;
    PDF_DIR) PDF_DIR="$input_value" ;;
    INDEX_DIR) INDEX_DIR="$input_value" ;;
    TARGET) TARGET_PAPERS="$input_value" ;;
  esac
done < <(read_case3_frozen_inputs)
if [[ -z "$CASE3_QUESTION_B64" || -z "$PAPER_MANIFEST" || -z "$PDF_DIR" ||
  -z "$INDEX_DIR" || "$TARGET_PAPERS" != "50" ]]; then
  echo "Case 3 冻结语料读取结果不完整或目标不是 50；流程未启动" >&2
  exit 1
fi
CASE3_QUESTION="$(printf '%s' "$CASE3_QUESTION_B64" | base64 --decode)"

CATALOG_ENDPOINT="http://127.0.0.1:$CATALOG_PORT/mcp"
INVESTIGATION_ENDPOINT="http://127.0.0.1:$INVESTIGATION_PORT/mcp"
RUNTIME_ROOT="$RUNTIME_PARENT/$RUN_NAME"
WORK_DIR="$RUNTIME_ROOT/controller"
LOG_ROOT="$RUNTIME_ROOT/logs"
STATE_ROOT="$RUNTIME_ROOT/state"
CATALOG_LOG="$LOG_ROOT/catalog_${CATALOG_PORT}.log"
INVESTIGATION_LOG="$LOG_ROOT/investigation_${INVESTIGATION_PORT}.log"
READINESS_LOG="$LOG_ROOT/readiness.log"
CONTROLLER_LOG="$LOG_ROOT/controller.log"

if ((DRY_RUN)); then
  # 题面含有多行化学事实，dry-run 只显示来源标识，不输出题面正文或 GJF 正文。
  printf '%s\n' \
    "dry_run=1" \
    "case=case3" \
    "run_name=$RUN_NAME" \
    "runtime_root=$RUNTIME_ROOT" \
    "case3_prepared_input_dir=$PREPARED_INPUT_DIR" \
    "case3_prepared_input_files=${EXPECTED_PREPARED_FILES[*]}" \
    "case3_prepared_input_delivery=harness_copies_files_into_run_workspace" \
    "catalog_endpoint=$CATALOG_ENDPOINT" \
    "investigation_endpoint=$INVESTIGATION_ENDPOINT" \
    "paper_search_manifest=$PAPER_MANIFEST" \
    "pdf_dir=$PDF_DIR" \
    "index_dir=$INDEX_DIR" \
    "target_papers=50" \
    "resources=cpu:$CPU_CORES,memory_mb:$MEMORY_MB,gpu:$GPU_COUNT" \
    "ARCHE_PLANNER_DRIVER=codex" \
    "ARCHE_CODEX_AUTH_MODE=api" \
    "ARCHE_CODEX_MODEL_NAME=gpt-5.6-sol" \
    "ARCHE_CODEX_REASONING_EFFORT=xhigh" \
    "ARCHE_CODEX_ALLOW_PROVIDER_MIGRATION=0" \
    "question_source=CASE23_AUTHOR_SCIENTIFIC_QUESTION_INPUTS[case3]" \
    "controller=chemistry_multiagent.controllers.chemistry_multiagent_controller" \
    "controller_args=create-standard-research-task --question <frozen_case3_question> --case-profile case3_fixed_reproduction --use-case3-author-main-hypothesis-exploration --case3-prepared-input-dir $PREPARED_INPUT_DIR --no-search --paper-search-target-papers 50 --num-queries 1 --num-hypotheses-per-query 1 --top-n-hypotheses 1 --max-parallel-investigations 1 --max-hypothesis-batches 1 --max-investigation-repairs-per-combination 3" \
    "services=catalog_$CATALOG_PORT,investigation_$INVESTIGATION_PORT" \
    "side_effects=none"
  exit 0
fi

umask 077
mkdir -p "$RUNTIME_PARENT"
if [[ -e "$RUNTIME_ROOT" || -L "$RUNTIME_ROOT" ]]; then
  echo "本次 run name 的 runtime 已存在；为避免重复创建任务，流程拒绝启动：$RUNTIME_ROOT" >&2
  exit 1
fi
mkdir "$RUNTIME_ROOT"
mkdir "$LOG_ROOT" "$STATE_ROOT"
chmod 700 "$RUNTIME_ROOT" "$LOG_ROOT" "$STATE_ROOT"
exec > >(tee -a "$RUNTIME_ROOT/logs/worker.log") 2>&1
write_state() { printf '%s\n' "$1" >"$STATE_ROOT/phase"; }

write_state "loading_private_config"
set -a
# shellcheck disable=SC1090
if ! source "$CONFIG_FILE"; then
  echo "private model config 加载失败；流程未启动" >&2
  exit 1
fi
set +a
for required_model_variable in \
  ARCHE_CHEM_API_KEY \
  ARCHE_CHEM_BASE_URL \
  ARCHE_CODEX_API_KEY \
  ARCHE_CODEX_BASE_URL \
  ARCHE_REVIEW_CODEX_API_KEY \
  ARCHE_REVIEW_CODEX_BASE_URL; do
  if [[ -z "$(printenv "$required_model_variable" 2>/dev/null || true)" ]]; then
    echo "private model config 缺少 $required_model_variable；流程未启动" >&2
    exit 1
  fi
done

export ARCHE_RUN_STORAGE_ROOT="$RUNTIME_ROOT"
export ARCHE_STAGE_HARNESS_WORKSPACE_ROOT="$RUNTIME_ROOT/harness_workspaces"
export ARCHE_STAGE_HARNESS_TRACE_ROOT="$RUNTIME_ROOT/harness_traces"
export ARCHE_CODEX_HOME="$RUNTIME_ROOT/codex_home"
export MINICHEM_SCOPE_REGISTRY_ROOT="$RUNTIME_ROOT/minichem_scope_registry"
export MINICHEM_SHARED_SCHEDULER_ROOT="$RUNTIME_ROOT/minichem_scheduler"
export MINICHEM_SERVICE_WORKSPACE="$RUNTIME_ROOT/minichem_service"
export MINICHEM_CATALOG_ENDPOINT="$CATALOG_ENDPOINT"
export MINICHEM_INVESTIGATION_ENDPOINT="$INVESTIGATION_ENDPOINT"
export CASE3_CATALOG_PORT="$CATALOG_PORT"
export CASE3_INVESTIGATION_PORT="$INVESTIGATION_PORT"
export MINICHEM_CATALOG_BOOTSTRAP_MCP_PORT="$CATALOG_PORT"
export MINICHEM_INVESTIGATION_MCP_PORT="$INVESTIGATION_PORT"
export RESEARCHCHEMBENCH_AVAILABLE_CPU_CORES="$CPU_CORES"
export RESEARCHCHEMBENCH_AVAILABLE_MEMORY_MB="$MEMORY_MB"
export RESEARCHCHEMBENCH_AVAILABLE_GPU_COUNT="$GPU_COUNT"
export ARCHE_PLANNER_DRIVER=codex
export ARCHE_CODEX_AUTH_MODE=api
export ARCHE_CODEX_MODEL_NAME=gpt-5.6-sol
export ARCHE_CODEX_REASONING_EFFORT=xhigh
export ARCHE_CODEX_ALLOW_PROVIDER_MIGRATION=0
export ARCHE_CODEX_BIN="$VENDORED_CODEX_BIN"
# API 模式不读取本机登录凭据；清除可能残留的 local_login 控制变量。
unset ARCHE_CODEX_AUTH_SOURCE ARCHE_CODEX_LOCAL_LOGIN
export PYTHONPATH="$PYTHONPATH_VALUE"

printf 'case3_prepared_input_dir=%s\ncase3_prepared_input_files=%s\ncatalog_endpoint=%s\ninvestigation_endpoint=%s\n' \
  "$PREPARED_INPUT_DIR" "${EXPECTED_PREPARED_FILES[*]}" "$CATALOG_ENDPOINT" "$INVESTIGATION_ENDPOINT" \
  >"$STATE_ROOT/input_source"

CATALOG_PID=""
INVESTIGATION_PID=""
stop_service() {
  local pid="$1" name="$2"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "停止本次运行启动的 $name 服务组（pid=$pid）"
    kill -TERM -- "-$pid" 2>/dev/null || true
    kill -TERM "$pid" 2>/dev/null || true
    for _ in {1..30}; do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
      kill -KILL -- "-$pid" 2>/dev/null || true
      kill -KILL "$pid" 2>/dev/null || true
    fi
    wait "$pid" 2>/dev/null || true
  fi
}
cleanup_services() {
  local exit_code="$?"
  trap - EXIT INT TERM PIPE
  # 即使调用方关闭输出管道，也必须继续回收本次服务组并写完退出状态。
  trap '' PIPE
  set +e
  write_state "cleaning_up"
  stop_service "$INVESTIGATION_PID" "investigation_$INVESTIGATION_PORT"
  stop_service "$CATALOG_PID" "catalog_$CATALOG_PORT"
  printf 'exit_code=%s\n' "$exit_code" >"$STATE_ROOT/exit"
  if ((exit_code == 0)); then write_state "succeeded"; else write_state "failed"; fi
  exit "$exit_code"
}
trap cleanup_services EXIT
trap 'exit 143' INT TERM

start_service() {
  local service_name="$1" service_log="$2"
  shift 2
  setsid "$@" >"$service_log" 2>&1 &
  local pid="$!"
  if [[ "$service_name" == catalog_* ]]; then CATALOG_PID="$pid"; else INVESTIGATION_PID="$pid"; fi
  printf '%s=%s\n' "$service_name" "$pid" >>"$STATE_ROOT/service_pids"
  echo "$service_name 服务已启动（pid=$pid，日志=$service_log）"
  sleep 1
  kill -0 "$pid" 2>/dev/null || { echo "$service_name 服务提前退出" >&2; return 1; }
}

write_state "starting_catalog_$CATALOG_PORT"
start_service "catalog_$CATALOG_PORT" "$CATALOG_LOG" \
  env \
  ARCHE_RUN_STORAGE_ROOT="$RUNTIME_ROOT" \
  MINICHEM_PROFILE_WORKSPACE="$RUNTIME_ROOT/minichem_catalog" \
  MINICHEM_PROFILE_HOST=127.0.0.1 \
  MINICHEM_PROFILE_PORT="$CATALOG_PORT" \
  MINICHEM_SERVICE_WORKSPACE="$RUNTIME_ROOT/minichem_catalog" \
  RESEARCHCHEMBENCH_AVAILABLE_CPU_CORES="$CPU_CORES" \
  RESEARCHCHEMBENCH_AVAILABLE_MEMORY_MB="$MEMORY_MB" \
  RESEARCHCHEMBENCH_AVAILABLE_GPU_COUNT="$GPU_COUNT" \
  bash "$ROOT/scripts/services/start_minichem.sh" --profile catalog_bootstrap

write_state "starting_investigation_$INVESTIGATION_PORT"
start_service "investigation_$INVESTIGATION_PORT" "$INVESTIGATION_LOG" \
  env \
  ARCHE_RUN_STORAGE_ROOT="$RUNTIME_ROOT" \
  MINICHEM_PROFILE_WORKSPACE="$RUNTIME_ROOT/minichem_investigation" \
  MINICHEM_PROFILE_HOST=127.0.0.1 \
  MINICHEM_PROFILE_PORT="$INVESTIGATION_PORT" \
  MINICHEM_SERVICE_WORKSPACE="$RUNTIME_ROOT/minichem_investigation" \
  MINICHEM_SCOPE_REGISTRY_ROOT="$MINICHEM_SCOPE_REGISTRY_ROOT" \
  MINICHEM_SHARED_SCHEDULER_ROOT="$MINICHEM_SHARED_SCHEDULER_ROOT" \
  RESEARCHCHEMBENCH_AVAILABLE_CPU_CORES="$CPU_CORES" \
  RESEARCHCHEMBENCH_AVAILABLE_MEMORY_MB="$MEMORY_MB" \
  RESEARCHCHEMBENCH_AVAILABLE_GPU_COUNT="$GPU_COUNT" \
  bash "$ROOT/scripts/services/start_minichem.sh" \
  --profile investigation \
  --scope-auth \
  --scope-registry "$MINICHEM_SCOPE_REGISTRY_ROOT" \
  --scheduler-root "$MINICHEM_SHARED_SCHEDULER_ROOT"

write_state "waiting_for_mcp_readiness"
if ! "$PYTHON" "$ROOT/scripts/tasks/case3_check_mcp_readiness.py" \
  --catalog-port "$CATALOG_PORT" \
  --investigation-port "$INVESTIGATION_PORT" \
  --runtime-root "$RUNTIME_ROOT" \
  --scope-registry "$MINICHEM_SCOPE_REGISTRY_ROOT" \
  --timeout-seconds 120 \
  >"$READINESS_LOG" 2>&1; then
  echo "MCP readiness 失败；未创建 Case 3 任务，请查看 $READINESS_LOG" >&2
  exit 1
fi

write_state "running_new_case3_task"
set +e
"$PYTHON" -m chemistry_multiagent.controllers.chemistry_multiagent_controller \
  create-standard-research-task \
  --question "$CASE3_QUESTION" \
  --case-profile case3_fixed_reproduction \
  --use-case3-author-main-hypothesis-exploration \
  --case3-prepared-input-dir "$PREPARED_INPUT_DIR" \
  --no-search \
  --paper-search-manifest "$PAPER_MANIFEST" \
  --pdf-dir "$PDF_DIR" \
  --index-dir "$INDEX_DIR" \
  --paper-search-target-papers 50 \
  --num-queries 1 \
  --num-hypotheses-per-query 1 \
  --top-n-hypotheses 1 \
  --max-parallel-investigations 1 \
  --max-hypothesis-batches 1 \
  --max-investigation-repairs-per-combination 3 \
  --work-dir "$WORK_DIR" 2>&1 | tee "$CONTROLLER_LOG"
PIPE_STATUS=("${PIPESTATUS[@]}")
set -e
CONTROLLER_EXIT_CODE="${PIPE_STATUS[0]:-1}"
CONTROLLER_TEE_EXIT_CODE="${PIPE_STATUS[1]:-1}"
printf 'controller_exit_code=%s\n' "$CONTROLLER_EXIT_CODE" >"$STATE_ROOT/controller_exit"
printf 'controller_tee_exit_code=%s\n' "$CONTROLLER_TEE_EXIT_CODE" >>"$STATE_ROOT/controller_exit"
if ((CONTROLLER_EXIT_CODE != 0 || CONTROLLER_TEE_EXIT_CODE != 0)); then
  echo "Case 3 探索任务退出码为 $CONTROLLER_EXIT_CODE；结果和日志保留在 $RUNTIME_ROOT" >&2
  if ((CONTROLLER_EXIT_CODE == 0)); then
    exit "$CONTROLLER_TEE_EXIT_CODE"
  fi
  exit "$CONTROLLER_EXIT_CODE"
fi
echo "Case 3 主假设探索 research_task 已结束；结果目录=$RUNTIME_ROOT/outputs/research_tasks"
exit 0
