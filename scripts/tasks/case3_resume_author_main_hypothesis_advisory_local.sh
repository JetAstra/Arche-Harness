#!/usr/bin/env bash
set -Eeuo pipefail

# 在已有的 Case 3 advisory runtime 内恢复同一 research_task 和同一 Codex session。
# 本入口只新增本次恢复尝试的日志/状态目录，不创建新的 Bundle 或 research_task。
# 旧 schema-v10 历史目录必须改用显式迁移工具，不能从这里回退读取。
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
PYTHONPATH_VALUE="$ROOT/src:$ROOT/third_party/minichem_toolbox/src"
VENDORED_CODEX_BIN="$ROOT/third_party/codex_cli/0.149.0/linux-x64/codex"
CONFIG_FILE="${RESEARCHCHEMBENCH_CONFIG:-$ROOT/config.local.env}"
DEFAULT_RUNTIME_PARENT="$ROOT/local_data/runtime/case3_author_main_hypothesis_advisory_local"

RUNTIME_ROOT=""
RUN_NAME=""
RESEARCH_TASK_ID=""
CPU_CORES="${CASE3_RESUME_CPU_CORES:-60}"
MEMORY_MB="${CASE3_RESUME_MEMORY_MB:-320000}"
GPU_COUNT="${CASE3_RESUME_GPU_COUNT:-1}"
AUTHORIZE_LEGACY_WAITING_RESUME=0
DRY_RUN=0

usage() {
  cat <<'EOF'
用法：scripts/tasks/case3_resume_author_main_hypothesis_advisory_local.sh [选项]
  --runtime-root PATH        已有运行目录；必须包含唯一的 research_task Bundle
  --run-name NAME            已有运行名；未给 runtime-root 时从默认 Case 3 目录定位
  --research-task-id ID      要恢复的原 research_task_id（必填）
  --cpu N, --cpu-cores N     本次 MiniChem 可用 CPU 核数，默认 60
  --memory N, --memory-mb N  本次 MiniChem 可用内存（MB），默认 320000
  --gpu N, --gpu-count N     本次 MiniChem 可用 GPU 数，默认 1
  --authorize-legacy-waiting-resume
                             显式授权恢复已通过身份核对的旧 v1 waiting 记录
  --dry-run                  只显示脱敏配置、路径和恢复命令，不启动服务
  -h, --help                 显示帮助
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
    --runtime-root)
      (($# >= 2)) || fail_usage "$1 需要一个值"
      RUNTIME_ROOT="$2"; shift 2 ;;
    --runtime-root=*) RUNTIME_ROOT="${1#--runtime-root=}"; shift ;;
    --runtime|--existing-runtime)
      (($# >= 2)) || fail_usage "$1 需要一个值"
      RUNTIME_ROOT="$2"; shift 2 ;;
    --runtime=*|--existing-runtime=*) RUNTIME_ROOT="${1#*=}"; shift ;;
    --run-name)
      (($# >= 2)) || fail_usage "$1 需要一个值"
      RUN_NAME="$2"; shift 2 ;;
    --run-name=*) RUN_NAME="${1#--run-name=}"; shift ;;
    --research-task-id)
      (($# >= 2)) || fail_usage "$1 需要一个值"
      RESEARCH_TASK_ID="$2"; shift 2 ;;
    --research-task-id=*) RESEARCH_TASK_ID="${1#--research-task-id=}"; shift ;;
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
    --authorize-legacy-waiting-resume)
      AUTHORIZE_LEGACY_WAITING_RESUME=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) fail_usage "未知参数：$1" ;;
  esac
done

check_positive_integer "CPU 核数" "$CPU_CORES"
check_positive_integer "内存 MB" "$MEMORY_MB"
check_nonnegative_integer "GPU 数" "$GPU_COUNT"
if [[ -z "$RESEARCH_TASK_ID" || ! "$RESEARCH_TASK_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]]; then
  fail_usage "research_task_id 必须是非空的安全标识"
fi

if [[ -z "$RUNTIME_ROOT" ]]; then
  if [[ -z "$RUN_NAME" ]]; then
    fail_usage "必须提供 --runtime-root，或同时提供 --run-name 以使用默认运行目录"
  fi
  RUNTIME_ROOT="$DEFAULT_RUNTIME_PARENT/$RUN_NAME"
fi
if [[ "$RUNTIME_ROOT" != /* ]]; then
  RUNTIME_ROOT="$ROOT/$RUNTIME_ROOT"
fi
if [[ "$PYTHON" != "$ROOT/.venv/bin/python" || ! -x "$PYTHON" ]]; then
  echo "必须使用仓库标准 Python；流程未启动" >&2
  exit 1
fi
if [[ ! -d "$RUNTIME_ROOT" || -L "$RUNTIME_ROOT" ]]; then
  echo "已有运行目录不存在或不是普通目录；流程未启动：$RUNTIME_ROOT" >&2
  exit 1
fi
RUNTIME_ROOT="$($PYTHON - "$RUNTIME_ROOT" <<'PY'
from pathlib import Path
import sys

candidate = Path(sys.argv[1]).expanduser()
if candidate.is_symlink() or not candidate.is_dir():
    raise SystemExit(1)
print(candidate.resolve())
PY
)"

if [[ -z "$RUN_NAME" ]]; then
  RUN_NAME="$(basename "$RUNTIME_ROOT")"
elif [[ ! "$RUN_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]]; then
  fail_usage "run name 只能包含 ASCII 字母、数字、点、下划线和连字符"
elif [[ "$RUN_NAME" != "$(basename "$RUNTIME_ROOT")" ]]; then
  echo "run name 与 runtime 根目录名不一致；流程未启动" >&2
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

RUN_STORAGE_ROOT="$RUNTIME_ROOT"
BUNDLE_ROOT="$RUNTIME_ROOT/outputs/research_tasks/$RESEARCH_TASK_ID"
CODEX_HOME="$RUNTIME_ROOT/codex_home"
STAGE_HARNESS_WORKSPACE_ROOT="$RUNTIME_ROOT/harness_workspaces"
STAGE_HARNESS_TRACE_ROOT="$RUNTIME_ROOT/harness_traces"
MINICHEM_SCOPE_REGISTRY_ROOT="$RUNTIME_ROOT/minichem_scope_registry"
SCHEDULER_ROOT="$RUNTIME_ROOT/minichem_scheduler"
MINICHEM_SERVICE_WORKSPACE="$RUNTIME_ROOT/minichem_service"

# 只读核对唯一 Run Bundle、TaskCreated 身份、V2 架构和 advisory 模式；不读取或打印科学正文。
validate_existing_task_bundle() {
  "$PYTHON" - "$RUNTIME_ROOT" "$RESEARCH_TASK_ID" <<'PY'
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any


runtime = Path(sys.argv[1])
research_task_id = sys.argv[2]
outputs = runtime / "outputs"
research_tasks = outputs / "research_tasks"
if (
    outputs.is_symlink()
    or not outputs.is_dir()
    or research_tasks.is_symlink()
    or not research_tasks.is_dir()
):
    raise SystemExit("已有 runtime 缺少 outputs/research_tasks；流程未启动")
bundles = sorted(
    item for item in research_tasks.iterdir()
    if item.is_dir() and not item.is_symlink()
)
if len(bundles) != 1:
    raise SystemExit("runtime 必须恰好包含一个 research_task Bundle；流程未启动")
bundle = bundles[0]
if bundle.name != research_task_id:
    raise SystemExit("research_task_id 与 runtime 中唯一 Bundle 不匹配；流程未启动")


def read_object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"恢复所需 {label} 缺失或不是普通文件；流程未启动")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise SystemExit(f"恢复所需 {label} 损坏；流程未启动")
    if not isinstance(value, dict):
        raise SystemExit(f"恢复所需 {label} 必须是 JSON 对象；流程未启动")
    return value


run_document = read_object(bundle / "run.json", "run.json")
status_document = read_object(bundle / "status.json", "status.json")
summary_document = read_object(bundle / "summary.json", "summary.json")
audit_document = read_object(bundle / "internal" / "audit" / "task.json", "audit task.json")
for document, label in (
    (run_document, "run.json"),
    (status_document, "status.json"),
    (summary_document, "summary.json"),
):
    if document.get("research_task_id") != research_task_id:
        raise SystemExit(f"{label} 的 research_task 身份与恢复参数不一致；流程未启动")
if audit_document.get("research_task_id") != research_task_id:
    raise SystemExit("audit task.json 的 research_task 身份与恢复参数不一致；流程未启动")
task_config = run_document.get("task_config")
if not isinstance(task_config, dict):
    raise SystemExit("run.json 缺少任务配置；流程未启动")
if task_config.get("research_task_id") != research_task_id:
    raise SystemExit("run.json 的 ResearchTask 身份与恢复参数不一致；流程未启动")
if task_config.get("stage_layout") != "research_task":
    raise SystemExit("目标 research_task 不是 investigation_v2 任务；流程未启动")
if task_config.get("investigation_mode") != "author_case3_advisory_main_hypothesis":
    raise SystemExit("目标 research_task 不是作者主假设 advisory 任务；流程未启动")

database = bundle / "internal" / "state" / "arche_memory.sqlite3"
if database.is_symlink() or not database.is_file():
    raise SystemExit("恢复所需 Working Memory 数据库缺失或不是普通文件；流程未启动")
try:
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT research_task_id, stage_architecture_version FROM research_tasks"
        ).fetchall()
    finally:
        connection.close()
except sqlite3.Error:
    raise SystemExit("恢复所需 Working Memory 数据库无法读取；流程未启动")
if (
    len(rows) != 1
    or rows[0][0] != research_task_id
    or rows[0][1] != "investigation_v2"
):
    raise SystemExit("Working Memory research_task 身份或架构与恢复参数不匹配；流程未启动")
PY
}

validate_existing_task_bundle

# 端口探测只尝试绑定并立即释放本地 socket；已被任意服务占用时拒绝继续。
check_resume_service_ports_available() {
  "$PYTHON" - <<'PY'
import socket

for port in (9011, 9012):
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))
    except OSError:
        raise SystemExit(f"9011/9012 服务端口已被占用（port={port}）；流程未启动")
    finally:
        probe.close()
PY
}

check_resume_service_ports_available

CONTROLLER_ARGS=(resume-research-task "$RESEARCH_TASK_ID")
if ((AUTHORIZE_LEGACY_WAITING_RESUME)); then
  CONTROLLER_ARGS+=(--authorize-legacy-waiting-resume)
fi

if ((DRY_RUN)); then
  CONTROLLER_COMMAND="$PYTHON -m chemistry_multiagent.controllers.chemistry_multiagent_controller resume-research-task $RESEARCH_TASK_ID"
  if ((AUTHORIZE_LEGACY_WAITING_RESUME)); then
    CONTROLLER_COMMAND+=" --authorize-legacy-waiting-resume"
  fi
  printf '%s\n' \
    "dry_run=1" \
    "run_name=$RUN_NAME" \
    "runtime_root=$RUNTIME_ROOT" \
    "research_task_id=$RESEARCH_TASK_ID" \
    "run_storage_root=$RUN_STORAGE_ROOT" \
    "bundle_root=$BUNDLE_ROOT" \
    "codex_home=$CODEX_HOME" \
    "stage_harness_workspace_root=$STAGE_HARNESS_WORKSPACE_ROOT" \
    "stage_harness_trace_root=$STAGE_HARNESS_TRACE_ROOT" \
    "minichem_scope_registry_root=$MINICHEM_SCOPE_REGISTRY_ROOT" \
    "minichem_scheduler_root=$SCHEDULER_ROOT" \
    "minichem_service_workspace=$MINICHEM_SERVICE_WORKSPACE" \
    "resources=cpu:$CPU_CORES,memory_mb:$MEMORY_MB,gpu:$GPU_COUNT" \
    "services=9011_catalog,9012_investigation" \
    "controller_command=$CONTROLLER_COMMAND" \
    "resume_attempt_root=$RUNTIME_ROOT/resume_attempts/<unique-attempt>" \
    "side_effects=none"
  exit 0
fi

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
  if [[ -z "${!required_model_variable:-}" ]]; then
    echo "private model config 缺少 $required_model_variable；流程未启动" >&2
    exit 1
  fi
done

export ARCHE_RUN_STORAGE_ROOT="$RUN_STORAGE_ROOT"
export ARCHE_STAGE_HARNESS_WORKSPACE_ROOT="$STAGE_HARNESS_WORKSPACE_ROOT"
export ARCHE_STAGE_HARNESS_TRACE_ROOT="$STAGE_HARNESS_TRACE_ROOT"
export ARCHE_CODEX_HOME="$CODEX_HOME"
export MINICHEM_SCOPE_REGISTRY_ROOT="$MINICHEM_SCOPE_REGISTRY_ROOT"
export MINICHEM_SHARED_SCHEDULER_ROOT="$SCHEDULER_ROOT"
export MINICHEM_SERVICE_WORKSPACE="$MINICHEM_SERVICE_WORKSPACE"
export MINICHEM_CATALOG_BOOTSTRAP_WORKSPACE="$RUNTIME_ROOT/minichem_catalog_bootstrap"
export MINICHEM_CATALOG_ENDPOINT="http://127.0.0.1:9011/mcp"
export MINICHEM_INVESTIGATION_ENDPOINT="http://127.0.0.1:9012/mcp"
export RESEARCHCHEMBENCH_AVAILABLE_CPU_CORES="$CPU_CORES"
export RESEARCHCHEMBENCH_AVAILABLE_MEMORY_MB="$MEMORY_MB"
export RESEARCHCHEMBENCH_AVAILABLE_GPU_COUNT="$GPU_COUNT"
export ARCHE_PLANNER_DRIVER=codex
export ARCHE_CODEX_AUTH_MODE=api
export ARCHE_CODEX_MODEL_NAME=gpt-5.6-sol
export ARCHE_CODEX_REASONING_EFFORT=xhigh
export ARCHE_CODEX_ALLOW_PROVIDER_MIGRATION=0
export ARCHE_CODEX_BIN="$VENDORED_CODEX_BIN"
# 恢复必须沿用 API 配置，不读取宿主机 local_login 凭据。
unset ARCHE_CODEX_AUTH_SOURCE ARCHE_CODEX_LOCAL_LOGIN
export PYTHONPATH="$PYTHONPATH_VALUE"

RESUME_PARENT="$RUNTIME_ROOT/resume_attempts"
mkdir -p "$RESUME_PARENT"
chmod 700 "$RESUME_PARENT"
RESUME_ATTEMPT_ROOT=""
for _ in {1..40}; do
  candidate="$RESUME_PARENT/resume-$(date -u +%Y%m%dT%H%M%S%N)-$$-$RANDOM"
  if (umask 077; mkdir "$candidate") 2>/dev/null; then
    RESUME_ATTEMPT_ROOT="$candidate"
    break
  fi
done
if [[ -z "$RESUME_ATTEMPT_ROOT" ]]; then
  echo "无法建立唯一恢复尝试目录；流程未启动" >&2
  exit 1
fi
mkdir "$RESUME_ATTEMPT_ROOT/logs" "$RESUME_ATTEMPT_ROOT/state"
chmod 700 "$RESUME_ATTEMPT_ROOT" "$RESUME_ATTEMPT_ROOT/logs" "$RESUME_ATTEMPT_ROOT/state"
LOG_ROOT="$RESUME_ATTEMPT_ROOT/logs"
STATE_ROOT="$RESUME_ATTEMPT_ROOT/state"
CATALOG_LOG="$LOG_ROOT/catalog_9011.log"
INVESTIGATION_LOG="$LOG_ROOT/investigation_9012.log"
READINESS_LOG="$LOG_ROOT/readiness.log"
CONTROLLER_LOG="$LOG_ROOT/controller.log"
exec > >(tee -a "$LOG_ROOT/worker.log") 2>&1

write_resume_state() {
  printf '%s\n' "$1" >"$STATE_ROOT/phase"
}

SERVICE_PIDS=()
SERVICE_NAMES=()
stop_started_service_groups() {
  local index pid name
  for ((index = ${#SERVICE_PIDS[@]} - 1; index >= 0; index--)); do
    pid="${SERVICE_PIDS[index]}"
    name="${SERVICE_NAMES[index]}"
    if kill -0 "$pid" 2>/dev/null; then
      echo "停止本次恢复启动的 $name 服务组（pid=$pid）"
      kill -TERM -- "-$pid" 2>/dev/null || true
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  for ((index = ${#SERVICE_PIDS[@]} - 1; index >= 0; index--)); do
    pid="${SERVICE_PIDS[index]}"
    name="${SERVICE_NAMES[index]}"
    for _ in {1..30}; do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
      echo "$name 服务组未在宽限期内退出，终止本次恢复服务组（pid=$pid）" >&2
      kill -KILL -- "-$pid" 2>/dev/null || true
      kill -KILL "$pid" 2>/dev/null || true
    fi
    wait "$pid" 2>/dev/null || true
  done
}

cleanup_resume_services() {
  local exit_code=$?
  trap - EXIT INT TERM PIPE
  trap '' PIPE
  set +e
  write_resume_state "cleaning_up"
  stop_started_service_groups
  printf 'exit_code=%s\n' "$exit_code" >"$STATE_ROOT/exit"
  if ((exit_code == 0)); then write_resume_state "succeeded"; else write_resume_state "failed"; fi
  exit "$exit_code"
}
trap cleanup_resume_services EXIT
trap 'exit 143' INT TERM

start_private_minichem_service() {
  local service_name="$1" service_script="$2" service_log="$3" pid
  setsid bash "$service_script" >"$service_log" 2>&1 &
  pid=$!
  SERVICE_NAMES+=("$service_name")
  SERVICE_PIDS+=("$pid")
  printf '%s=%s\n' "$service_name" "$pid" >>"$STATE_ROOT/service_pids"
  echo "$service_name 服务已启动（pid=$pid，日志=$service_log）"
  sleep 1
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "$service_name 服务进程已提前退出；请查看 $service_log" >&2
    return 1
  fi
}

write_resume_state "starting_catalog_9011"
start_private_minichem_service \
  "catalog_9011" \
  "$ROOT/scripts/services/start_catalog_bootstrap_mcp.sh" \
  "$CATALOG_LOG"
write_resume_state "starting_investigation_9012"
start_private_minichem_service \
  "investigation_9012" \
  "$ROOT/scripts/services/start_investigation_minichem.sh" \
  "$INVESTIGATION_LOG"
write_resume_state "waiting_for_mcp_readiness"
if ! "$PYTHON" "$ROOT/scripts/tasks/case3_check_mcp_readiness.py" \
  --runtime-root "$RESUME_ATTEMPT_ROOT" \
  --scope-registry "$MINICHEM_SCOPE_REGISTRY_ROOT" \
  --timeout-seconds 120 \
  >"$READINESS_LOG" 2>&1; then
  echo "9011/9012 MCP readiness 失败；原 task 未恢复，请查看 $READINESS_LOG" >&2
  exit 1
fi

write_resume_state "resuming_existing_task"
set +e
"$PYTHON" -m chemistry_multiagent.controllers.chemistry_multiagent_controller \
  "${CONTROLLER_ARGS[@]}" \
  2>&1 | tee "$CONTROLLER_LOG"
PIPE_STATUS=("${PIPESTATUS[@]}")
set -e
CONTROLLER_EXIT_CODE="${PIPE_STATUS[0]:-1}"
CONTROLLER_TEE_EXIT_CODE="${PIPE_STATUS[1]:-1}"
printf 'controller_exit_code=%s\n' "$CONTROLLER_EXIT_CODE" >"$STATE_ROOT/controller_exit"
printf 'controller_tee_exit_code=%s\n' "$CONTROLLER_TEE_EXIT_CODE" >>"$STATE_ROOT/controller_exit"
if ((CONTROLLER_EXIT_CODE != 0 || CONTROLLER_TEE_EXIT_CODE != 0)); then
  echo "原 task 恢复退出码为 $CONTROLLER_EXIT_CODE；原结果和本次恢复日志保留在 $RUNTIME_ROOT" >&2
  if ((CONTROLLER_EXIT_CODE == 0)); then
    exit "$CONTROLLER_TEE_EXIT_CODE"
  fi
  exit "$CONTROLLER_EXIT_CODE"
fi
echo "原 task 恢复已结束；原 Run Bundle=$BUNDLE_ROOT；本次日志=$RESUME_ATTEMPT_ROOT"
exit 0
