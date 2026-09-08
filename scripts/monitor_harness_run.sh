#!/usr/bin/env bash
set -u
set -o pipefail

# 只读监控一个 research_task 的 Harness 恢复任务。输入是私有 runtime 根和
# research_task_id；脚本读取 Run Bundle、恢复 attempt、进程表和监听端口，不调用
# MCP、不改数据库，也不向 Harness 发送信号。历史 schema-v10 只能显式使用
# --legacy-run-id 读取，不能作为 research_task 的默认回退。持续模式在 Harness
# 退出后结束；失败或未知退出返回非零状态。

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
RUNTIME_ROOT=""
RESEARCH_TASK_ID=""
LEGACY_RUN_ID=""
LABEL="Harness"
INTERVAL_SECONDS="${HARNESS_MONITOR_INTERVAL_SECONDS:-30}"
MODEL_OUTPUT_LINES="${HARNESS_MONITOR_MODEL_OUTPUT_LINES:-20}"
MODEL_OUTPUT_MESSAGES="${HARNESS_MONITOR_MODEL_OUTPUT_MESSAGES:-5}"
CATALOG_PORT=9011
INVESTIGATION_PORT=9012
PROCESS_PATTERN=""
LAUNCHER_PID_FILE=""
RUN_ONCE=0
PREVIOUS_MODEL_CURSOR=""

print_usage() {
  printf '%s\n' \
    '用法：' \
    '  bash scripts/monitor_harness_run.sh \' \
    '    --runtime-root PATH \' \
    '    --research-task-id RESEARCH_TASK_ID \' \
    '    （历史只读：--legacy-run-id RUN_ID）\' \
    '    [--interval 秒数] [--once] [--label 名称] \' \
    '    [--model-output-lines 行数] [--model-output-messages 条数] \' \
    '    [--catalog-port 端口] [--investigation-port 端口] \' \
    '    [--process-pattern 文本] [--launcher-pid-file PATH] [--python PATH]' \
    '' \
    '默认每 30 秒只读检查一次，并显示完整任务阶段。进入 Investigation/Experiment' \
    '后显示最多 5 条新增模型回答，每条最多 20 行。Harness 退出后打印终态并结束；' \
    '按 Ctrl-C 只会停止监控脚本。默认进程特征是当前 research_task 的 resume Controller，' \
    '默认 launcher PID 文件是 <runtime>/state/frozen_resume_launcher.pid。'
}

require_option_value() {
  local option="$1" remaining="$2"
  if ((remaining < 2)); then
    printf '%s 缺少参数值\n' "$option" >&2
    exit 64
  fi
}

while (($# > 0)); do
  case "$1" in
    --runtime-root|--research-task-id|--legacy-run-id|--interval|--label|--model-output-lines|\
    --model-output-messages|--catalog-port|--investigation-port|\
    --process-pattern|--launcher-pid-file|--python)
      require_option_value "$1" "$#"
      option="$1"
      value="$2"
      shift 2
      case "$option" in
        --runtime-root) RUNTIME_ROOT="$value" ;;
        --research-task-id) RESEARCH_TASK_ID="$value" ;;
        --legacy-run-id) LEGACY_RUN_ID="$value" ;;
        --interval) INTERVAL_SECONDS="$value" ;;
        --label) LABEL="$value" ;;
        --model-output-lines) MODEL_OUTPUT_LINES="$value" ;;
        --model-output-messages) MODEL_OUTPUT_MESSAGES="$value" ;;
        --catalog-port) CATALOG_PORT="$value" ;;
        --investigation-port) INVESTIGATION_PORT="$value" ;;
        --process-pattern) PROCESS_PATTERN="$value" ;;
        --launcher-pid-file) LAUNCHER_PID_FILE="$value" ;;
        --python) PYTHON="$value" ;;
      esac
      ;;
    --once)
      RUN_ONCE=1
      shift
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      printf '未知参数：%s\n' "$1" >&2
      print_usage >&2
      exit 64
      ;;
  esac
done

if [[ -z "$RUNTIME_ROOT" || ( -z "$RESEARCH_TASK_ID" && -z "$LEGACY_RUN_ID" ) || ( -n "$RESEARCH_TASK_ID" && -n "$LEGACY_RUN_ID" ) ]]; then
  printf '%s\n' '--runtime-root 必填，且必须二选一提供 --research-task-id 或 --legacy-run-id' >&2
  print_usage >&2
  exit 64
fi
MONITORED_ID="$RESEARCH_TASK_ID"
LAYOUT_MODE="research_task"
IDENTITY_LABEL="research_task_id"
if [[ -n "$LEGACY_RUN_ID" ]]; then
  MONITORED_ID="$LEGACY_RUN_ID"
  LAYOUT_MODE="legacy_schema_v10"
  IDENTITY_LABEL="legacy_run_id"
fi
case "$MONITORED_ID" in
  ''|[!A-Za-z0-9]*|*[!A-Za-z0-9._-]*)
    printf '%s 只能包含字母、数字、点、下划线和连字符，且必须以字母或数字开头：%q\n' "$IDENTITY_LABEL" "$MONITORED_ID" >&2
    exit 64
    ;;
esac
case "$INTERVAL_SECONDS" in
  ''|*[!0-9]*)
    printf '轮询间隔必须是正整数：%s\n' "$INTERVAL_SECONDS" >&2
    exit 64
    ;;
esac
if ((INTERVAL_SECONDS < 1)); then
  printf '轮询间隔必须大于 0：%s\n' "$INTERVAL_SECONDS" >&2
  exit 64
fi
for limit_name in MODEL_OUTPUT_LINES MODEL_OUTPUT_MESSAGES; do
  limit_value="${!limit_name}"
  case "$limit_value" in
    ''|*[!0-9]*)
      printf '%s 必须是正整数：%s\n' "$limit_name" "$limit_value" >&2
      exit 64
      ;;
  esac
  if ((limit_value < 1)); then
    printf '%s 必须大于 0：%s\n' "$limit_name" "$limit_value" >&2
    exit 64
  fi
done
if ((MODEL_OUTPUT_LINES > 200)); then
  printf 'MODEL_OUTPUT_LINES 不能大于 200：%s\n' "$MODEL_OUTPUT_LINES" >&2
  exit 64
fi
if ((MODEL_OUTPUT_MESSAGES > 20)); then
  printf 'MODEL_OUTPUT_MESSAGES 不能大于 20：%s\n' "$MODEL_OUTPUT_MESSAGES" >&2
  exit 64
fi
for port_name in CATALOG_PORT INVESTIGATION_PORT; do
  port_value="${!port_name}"
  case "$port_value" in
    ''|*[!0-9]*)
      printf '%s 必须是有效端口：%s\n' "$port_name" "$port_value" >&2
      exit 64
      ;;
  esac
  if ((port_value < 1 || port_value > 65535)); then
    printf '%s 必须在 1..65535：%s\n' "$port_name" "$port_value" >&2
    exit 64
  fi
done

if [[ ! -d "$RUNTIME_ROOT" ]]; then
  printf 'runtime 根不存在：%s\n' "$RUNTIME_ROOT" >&2
  exit 66
fi
RUNTIME_ROOT="$(readlink -f -- "$RUNTIME_ROOT")"
if [[ ! -x "$PYTHON" ]]; then
  printf 'Python 不可执行：%s\n' "$PYTHON" >&2
  exit 66
fi
PYTHON="$(readlink -f -- "$PYTHON")"

if [[ -z "$PROCESS_PATTERN" ]]; then
  PROCESS_PATTERN="chemistry_multiagent_controller resume-research-task $MONITORED_ID"
fi
if [[ -z "$LAUNCHER_PID_FILE" ]]; then
  LAUNCHER_PID_FILE="$RUNTIME_ROOT/state/frozen_resume_launcher.pid"
elif [[ "$LAUNCHER_PID_FILE" != /* ]]; then
  LAUNCHER_PID_FILE="$RUNTIME_ROOT/$LAUNCHER_PID_FILE"
fi

trap 'printf "\n监控已停止；没有向 Harness 发送信号。\n"; exit 130' INT
trap 'printf "\n监控已停止；没有向 Harness 发送信号。\n"; exit 143' TERM

read_port_state() {
  local port="$1"
  if ! command -v ss >/dev/null 2>&1; then
    printf 'unavailable'
    return
  fi
  if ss -ltn 2>/dev/null | awk -v target=":$port" \
    '$4 ~ target "$" {found=1} END {exit !found}'; then
    printf 'up'
  else
    printf 'down'
  fi
}

print_status_sample() {
  local catalog_state="$1" investigation_state="$2"
  "$PYTHON" - \
    "$RUNTIME_ROOT" \
    "$MONITORED_ID" \
    "$LAYOUT_MODE" \
    "$LABEL" \
    "$PROCESS_PATTERN" \
    "$LAUNCHER_PID_FILE" \
    "$$" \
    "$CATALOG_PORT" \
    "$catalog_state" \
    "$INVESTIGATION_PORT" \
    "$investigation_state" \
    "$PREVIOUS_MODEL_CURSOR" \
    "$MODEL_OUTPUT_LINES" \
    "$MODEL_OUTPUT_MESSAGES" <<'PY'
import hashlib
import json
import os
import sqlite3
import sys
import textwrap
from datetime import datetime
from pathlib import Path

(
    runtime_text,
    monitored_id,
    layout_mode,
    label,
    process_pattern,
    launcher_pid_file_text,
    monitor_pid_text,
    catalog_port,
    catalog_state,
    investigation_port,
    investigation_state,
    previous_model_cursor,
    model_output_lines_text,
    model_output_messages_text,
) = sys.argv[1:]
runtime_root = Path(runtime_text)
legacy_mode = layout_mode == "legacy_schema_v10"
monitor_pid = int(monitor_pid_text)
model_output_lines = int(model_output_lines_text)
model_output_messages = int(model_output_messages_text)
bundle_namespace = "runs" if legacy_mode else "research_tasks"
bundle_root = runtime_root / "outputs" / bundle_namespace / monitored_id
state_root = bundle_root / "internal" / "state"

MODULE_LABELS = {
    "retrieval": "Retrieval",
    "research-query-formulation": "Research Query Formulation",
    "hypothesis-generation": "Hypothesis Generation",
    "hypothesis-import": "Hypothesis Import",
    "hypothesis-selection": "Hypothesis Selection",
    "hypothesis-confirmation": "Hypothesis Confirmation",
    "hypothesis-investigation": "Investigation/Experiment (Harness)",
    "task-review": "Task Review",
}
STAGE_LABELS = {
    "literature_retrieval": "Literature Retrieval",
    "hypothesis_preparation": "Hypothesis Preparation",
    "investigation_and_experiment": "Investigation/Experiment (Harness)",
    "task_review_and_result": "Task Review / Result",
}
if legacy_mode:
    # 历史 Bundle 只读展示；旧阶段名不会进入 research_tasks 文件。
    STAGE_LABELS = {
        "retrieval": "Retrieval",
        "hypothesis": "Hypothesis preparation",
        "investigation_and_experiment": "Investigation/Experiment (Harness)",
        "final_result": "Task Review / Final Result",
    }
STATUS_LABELS = {
    "completed": "完成",
    "open": "进行中",
    "running": "进行中",
    "paused": "已暂停",
    "waiting_investigation_job": "等待 Investigation 作业",
    "failed": "失败",
    "not_started": "未开始",
    "disabled": "已禁用",
    "not_applicable": "不适用",
    "skipped": "已跳过",
}
if legacy_mode:
    # 历史 schema-v10 的等待字段只在显式 legacy 只读分支展示。
    STATUS_LABELS["waiting_execution"] = "等待作业"
TERMINAL_RUN_STATUSES = {
    "succeeded",
    "failed",
    "stopped_no_supported_hypothesis",
}


def read_text(path: Path, default: str) -> str:
    try:
        value = " ".join(path.read_text(encoding="utf-8").split())
    except OSError:
        return default
    return value or default


def read_json(path: Path) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def normalize_module_kind(value: object) -> str:
    """规范化正式模块名；历史别名只在只读投影中归并。"""

    normalized = (
        str(value or "")
        .strip()
        .casefold()
        .replace("_", "-")
        .replace(" ", "-")
    )
    aliases = {
        "query": "research-query-formulation",
        "query-formulation": "research-query-formulation",
        "hypothesis": "hypothesis-generation",
        "hypotheses": "hypothesis-generation",
        "selection": "hypothesis-selection",
        "top5": "hypothesis-selection",
        "top-5": "hypothesis-selection",
        "top5-selection": "hypothesis-selection",
        "top-5-selection": "hypothesis-selection",
        "confirmation": "hypothesis-confirmation",
        "hypothesis-confirm": "hypothesis-confirmation",
        "hypothesis-confirmation-module": "hypothesis-confirmation",
        "investigation": "hypothesis-investigation",
        "investigation-and-experiment": "hypothesis-investigation",
        "investigation-experiment": "hypothesis-investigation",
        "review": "task-review",
    }
    if normalized in aliases:
        return aliases[normalized]
    if normalized.startswith("investigation-"):
        return "hypothesis-investigation"
    if normalized.endswith("-review"):
        return "task-review"
    return normalized


def decode_stored_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return value
    return decoded if isinstance(decoded, str) else value


def expected_module_kinds(contract_version: str, entry_strategy: str) -> list[str]:
    has_confirmation = contract_version.endswith(".v2")
    if entry_strategy == "author_hypothesis_import":
        stages = ["hypothesis-import"]
    else:
        stages = [
            "retrieval",
            "research-query-formulation",
            "hypothesis-generation",
            "hypothesis-selection",
        ]
    if has_confirmation:
        stages.append("hypothesis-confirmation")
    stages.extend(("hypothesis-investigation", "task-review"))
    return stages


def quote_identifier(value: str) -> str:
    """引用由 SQLite schema 返回的列名，避免监控查询拼接外部输入。"""

    if not value.replace("_", "").isalnum() or not value[0].isalpha():
        raise ValueError("SQLite schema column name 非法")
    return '"' + value.replace('"', '""') + '"'


def table_columns(connection: sqlite3.Connection, table: str) -> list[str]:
    """返回一张正式表的列名；表名是脚本内固定值。"""

    return [
        str(row[1])
        for row in connection.execute(f"PRAGMA table_info({quote_identifier(table)})")
    ]


def choose_column(columns: list[str], *candidates: str) -> str | None:
    """优先选择 canonical 列名，必要时读取 expand-contract 旧列名。"""

    return next((candidate for candidate in candidates if candidate in columns), None)


def row_value(row: sqlite3.Row, columns: list[str], column: str) -> object:
    """按 schema 列名读取 sqlite 行，兼容列顺序变化。"""

    return row[columns.index(column)]


def build_module_pipeline(
    contract_version: str,
    entry_strategy: str,
    modules: list[dict[str, object]],
) -> tuple[list[tuple[str, str]], str | None, str]:
    """把正式 Module Invocation 读取结果转换为监控展示。"""

    pipeline: list[tuple[str, str]] = []
    for kind in expected_module_kinds(contract_version, entry_strategy):
        matching = [module for module in modules if module["kind"] == kind]
        if any(bool(module["open"]) for module in matching):
            status = "open"
        elif matching:
            latest = matching[-1]
            event_type = str(latest["completed_event_type"]).casefold()
            status = "failed" if "failed" in event_type else "completed"
        else:
            status = "not_started"
        pipeline.append((MODULE_LABELS.get(kind, kind), status))

    open_modules = [module for module in modules if bool(module["open"])]
    if open_modules:
        current = max(open_modules, key=lambda module: int(module["sequence"]))
        current_kind = str(current["kind"])
        current_text = (
            f"{MODULE_LABELS.get(current_kind, current_kind)} "
            f"(module_invocation_id={current['module_invocation_id']})"
        )
        return pipeline, current_kind, current_text
    return pipeline, None, "<阶段切换中或任务已结束>"


def read_formal_module_progress(
    connection: sqlite3.Connection,
) -> tuple[list[tuple[str, str]], str | None, str] | None:
    """只读读取正式 research_task/module_invocation 表。"""

    table_names = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    if not {
        "research_tasks",
        "research_task_events",
        "module_invocations",
    }.issubset(table_names):
        return None

    task_columns = table_columns(connection, "research_tasks")
    event_columns = table_columns(connection, "research_task_events")
    module_columns = table_columns(connection, "module_invocations")
    # 正式表和正式列只接受 research_task/module_invocation 合同；旧物理列
    # 只能由下方显式 legacy schema-v10 分支读取。
    task_identity_column = choose_column(task_columns, "research_task_id")
    task_contract_column = choose_column(task_columns, "contract_version")
    task_policy_column = choose_column(task_columns, "policy_json")
    event_identity_column = choose_column(event_columns, "research_task_id")
    event_id_column = choose_column(event_columns, "event_id")
    event_sequence_column = choose_column(event_columns, "research_task_sequence")
    event_type_column = choose_column(event_columns, "event_type")
    event_payload_column = choose_column(event_columns, "payload_json")
    module_identity_column = choose_column(module_columns, "module_invocation_id")
    module_task_column = choose_column(module_columns, "research_task_id")
    module_kind_column = choose_column(module_columns, "module_kind")
    module_start_column = choose_column(module_columns, "started_event_id")
    module_completion_column = choose_column(module_columns, "completed_event_id")
    if any(
        column is None
        for column in (
            task_identity_column,
            task_contract_column,
            task_policy_column,
            event_identity_column,
            event_id_column,
            event_sequence_column,
            event_type_column,
            event_payload_column,
            module_identity_column,
            module_task_column,
            module_kind_column,
            module_start_column,
            module_completion_column,
        )
    ):
        return None

    task_row = connection.execute(
        f"SELECT * FROM research_tasks WHERE {quote_identifier(task_identity_column)}=?",
        (monitored_id,),
    ).fetchone()
    if task_row is None:
        return None
    contract_version = decode_stored_text(
        row_value(task_row, task_columns, task_contract_column)
    )
    try:
        policy = json.loads(row_value(task_row, task_columns, task_policy_column))
    except (TypeError, json.JSONDecodeError):
        policy = {}
    entry_strategy = (
        str(policy.get("entry_strategy") or "")
        if isinstance(policy, dict)
        else ""
    )

    event_rows = connection.execute("SELECT * FROM research_task_events").fetchall()
    events = {
        row_value(row, event_columns, event_id_column): row
        for row in event_rows
        if row_value(row, event_columns, event_identity_column)
        == monitored_id
    }
    failed_module_invocation_ids: set[str] = set()
    for event in events.values():
        event_type = str(row_value(event, event_columns, event_type_column))
        if event_type != "ModuleInvocationFailed":
            continue
        try:
            payload = json.loads(
                row_value(event, event_columns, event_payload_column)
            )
        except (TypeError, json.JSONDecodeError):
            return None
        failed_module_invocation_id = (
            payload.get("module_invocation_id")
            if isinstance(payload, dict)
            else None
        )
        if (
            not isinstance(failed_module_invocation_id, str)
            or not failed_module_invocation_id.strip()
        ):
            return None
        failed_module_invocation_ids.add(failed_module_invocation_id)

    module_rows = connection.execute(
        f"SELECT * FROM module_invocations WHERE {quote_identifier(module_task_column)}=?",
        (monitored_id,),
    ).fetchall()
    modules: list[dict[str, object]] = []
    for row in module_rows:
        module_invocation_id = str(
            row_value(row, module_columns, module_identity_column)
        )
        started_event_ref = row_value(row, module_columns, module_start_column)
        started_event = events.get(started_event_ref)
        if started_event is None:
            continue
        completed_event_ref = (
            row_value(row, module_columns, module_completion_column)
            if module_completion_column is not None
            else None
        )
        completed_event = events.get(completed_event_ref)
        failed = module_invocation_id in failed_module_invocation_ids
        if failed and completed_event_ref is not None:
            return None
        modules.append(
            {
                "kind": normalize_module_kind(
                    row_value(row, module_columns, module_kind_column)
                ),
                "module_invocation_id": module_invocation_id,
                "open": completed_event_ref is None and not failed,
                "sequence": int(
                    row_value(started_event, event_columns, event_sequence_column)
                ),
                "completed_event_type": str(
                    "ModuleInvocationFailed"
                    if failed
                    else row_value(
                        completed_event,
                        event_columns,
                        event_type_column,
                    )
                    if completed_event is not None
                    else ""
                ),
            }
        )
    if not failed_module_invocation_ids.issubset(
        {str(module["module_invocation_id"]) for module in modules}
    ):
        return None
    modules.sort(key=lambda module: int(module["sequence"]))
    return build_module_pipeline(contract_version, entry_strategy, modules)


def read_legacy_module_progress(
    connection: sqlite3.Connection,
) -> tuple[list[tuple[str, str]], str | None, str] | None:
    """显式 legacy schema-v10 监控分支；正式 research_task 不调用此函数。"""

    if not legacy_mode:
        return None
    try:
        task_row = connection.execute(
            "SELECT contract_version, policy_json FROM v2_tasks WHERE task_id=?",
            (monitored_id,),
        ).fetchone()
        if task_row is None:
            return None
        contract_version = decode_stored_text(task_row[0])
        try:
            policy = json.loads(task_row[1])
        except (TypeError, json.JSONDecodeError):
            policy = {}
        entry_strategy = (
            str(policy.get("entry_strategy") or "")
            if isinstance(policy, dict)
            else ""
        )
        rows = connection.execute(
            """
            SELECT modules.module_kind,
                   modules.module_run_id,
                   modules.completed_event_id,
                   started.task_sequence,
                   completed.event_type
              FROM v2_module_runs AS modules
              JOIN v2_events AS started
                ON started.event_id = modules.started_event_id
         LEFT JOIN v2_events AS completed
                ON completed.event_id = modules.completed_event_id
             WHERE modules.task_id=?
          ORDER BY started.task_sequence
            """,
            (monitored_id,),
        ).fetchall()
    except sqlite3.Error:
        return None
    modules = [
        {
            "kind": normalize_module_kind(row[0]),
            "module_invocation_id": str(row[1]),
            "open": row[2] is None,
            "sequence": int(row[3]),
            "completed_event_type": str(row[4] or ""),
        }
        for row in rows
    ]
    return build_module_pipeline(contract_version, entry_strategy, modules)


def read_research_task_module_progress() -> tuple[list[tuple[str, str]], str | None, str] | None:
    """按显式布局读取正式 Module Invocation，历史只读走专用分支。"""

    database_path = state_root / "arche_memory.sqlite3"
    if database_path.is_symlink() or not database_path.is_file():
        return None
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(
            database_path.resolve().as_uri() + "?mode=ro",
            uri=True,
            timeout=2,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        if legacy_mode:
            return read_legacy_module_progress(connection)
        return read_formal_module_progress(connection)
    except (OSError, sqlite3.Error, ValueError, TypeError):
        return None
    finally:
        if connection is not None:
            connection.close()


def read_summary_progress(
    bundle_status: str,
    reason_code: str,
) -> tuple[list[tuple[str, str]], str | None, str]:
    summary = read_json(bundle_root / "summary.json") or {}
    raw_order = summary.get("stage_order")
    raw_stages = summary.get("stages")
    stage_order = raw_order if isinstance(raw_order, list) else []
    stages = raw_stages if isinstance(raw_stages, dict) else {}
    pipeline: list[tuple[str, str]] = []
    current_stage: str | None = None
    current_text = "<unknown>"
    for raw_stage in stage_order:
        if not isinstance(raw_stage, str):
            continue
        stage = stages.get(raw_stage)
        status = str(stage.get("status") or "not_started") if isinstance(stage, dict) else "not_started"
        label = STAGE_LABELS.get(raw_stage, raw_stage)
        pipeline.append((label, status))
        if current_stage is None and status in {
            "open",
            "running",
            "paused",
            "waiting_investigation_job",
        }:
            current_stage = (
                "hypothesis-investigation"
                if raw_stage == "investigation_and_experiment"
                else raw_stage
            )
            current_text = label
    if current_stage is None and bundle_status not in TERMINAL_RUN_STATUSES:
        for label, status in pipeline:
            if status == "not_started":
                current_text = f"阶段切换中；下一阶段 {label}"
                break
    elif current_stage is None:
        current_text = "任务已结束"
    if reason_code == "awaiting_researcher_confirmation":
        current_stage = "hypothesis-confirmation"
        current_text = MODULE_LABELS[current_stage]
    return pipeline, current_stage, current_text


def read_launcher_pid() -> str:
    pid_path = Path(launcher_pid_file_text)
    try:
        pid_text = pid_path.read_text(encoding="utf-8").strip()
        if not pid_text.isdigit():
            return ""
        pid = int(pid_text)
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(
            "utf-8", errors="replace"
        )
    except OSError:
        return ""
    return pid_text if runtime_text in cmdline else ""


def find_matching_processes() -> str:
    matches: list[int] = []
    try:
        proc_entries = Path("/proc").iterdir()
    except OSError:
        return ""
    for proc_path in proc_entries:
        if not proc_path.name.isdigit():
            continue
        pid = int(proc_path.name)
        if pid in {os.getpid(), os.getppid(), monitor_pid}:
            continue
        try:
            cmdline = (proc_path / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            )
        except OSError:
            continue
        if "monitor_harness_run.sh" in cmdline:
            continue
        if process_pattern in cmdline:
            matches.append(pid)
    return ",".join(str(pid) for pid in sorted(matches))


def read_bundle_status() -> tuple[str, str, str]:
    payload = read_json(bundle_root / "status.json")
    if payload is None:
        return "<unavailable>", "<none>", "<none>"
    blocking = payload.get("blocking")
    error_type = blocking.get("error_type") if isinstance(blocking, dict) else None
    return (
        str(payload.get("status") or "<unknown>"),
        str(payload.get("reason_code") or "<none>"),
        str(error_type or "<none>"),
    )


def read_latest_job() -> tuple[str, str, str]:
    status_files = list(state_root.glob("**/outputs/execution_jobs/*/status.json"))
    if not status_files:
        return "<none>", "<none>", "<none>"
    try:
        path = max(status_files, key=lambda item: item.stat().st_mtime_ns)
    except OSError:
        return "<unreadable>", "<none>", "<none>"
    payload = read_json(path)
    if payload is None:
        return path.parent.name, "<unreadable>", "<none>"
    return (
        str(payload.get("job_id") or path.parent.name),
        str(payload.get("status") or "<unknown>"),
        str(payload.get("started_at") or payload.get("submitted_at") or "<none>"),
    )


def is_investigation_rollout(path: Path) -> bool:
    try:
        parts = path.relative_to(state_root).parts
    except ValueError:
        return False
    return "investigations" in parts or (parts and parts[0] == "codex_sessions")


def session_label(path: Path) -> str:
    for part in path.parts:
        if part.startswith("investigation-session-") or part.startswith("combination:"):
            return part
    return path.stem


def read_rollout_tail(path: Path) -> list[bytes]:
    try:
        with path.open("rb") as stream:
            size = stream.seek(0, 2)
            start = max(0, size - 4 * 1024 * 1024)
            stream.seek(start)
            lines = stream.read().splitlines()
    except OSError:
        return []
    if start > 0 and lines:
        return lines[1:]
    return lines


def read_investigation_events() -> tuple[
    tuple[str, str, str], list[dict[str, str]]
]:
    rollouts_by_session: dict[str, list[tuple[int, Path]]] = {}
    for path in state_root.glob("**/rollout-*.jsonl"):
        if not is_investigation_rollout(path):
            continue
        try:
            modified_at = path.stat().st_mtime_ns
        except OSError:
            continue
        rollouts_by_session.setdefault(session_label(path), []).append(
            (modified_at, path)
        )
    # 每个 Investigation Session 最多读取最近四个 rollout，覆盖当前 turn 和
    # continuation，同时避免长期任务每轮重新扫描全部历史正文。
    rollouts = [
        path
        for session_rollouts in rollouts_by_session.values()
        for _, path in sorted(session_rollouts, reverse=True)[:4]
    ]
    events: list[dict[str, str]] = []
    agent_messages: list[dict[str, str]] = []
    for path in rollouts:
        for raw_line in read_rollout_tail(path):
            try:
                record = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            payload = record.get("payload")
            if not isinstance(payload, dict):
                continue
            event_type = str(payload.get("type") or record.get("type") or "<unknown>")
            if event_type not in {"agent_message", "task_complete", "mcp_tool_call_end"}:
                continue
            timestamp = str(record.get("timestamp") or "<none>")
            relative_path = str(path.relative_to(state_root))
            cursor = hashlib.sha256(
                relative_path.encode("utf-8") + b"\0" + raw_line
            ).hexdigest()
            if event_type == "agent_message":
                detail = str(payload.get("message") or "")
            elif event_type == "task_complete":
                error = payload.get("error")
                detail = str(error.get("message") if isinstance(error, dict) else "completed")
            else:
                invocation = payload.get("invocation")
                detail = (
                    str(invocation.get("tool") or event_type)
                    if isinstance(invocation, dict)
                    else event_type
                )
            event = {
                "timestamp": timestamp,
                "type": event_type,
                "detail": detail,
                "cursor": cursor,
                "session": session_label(path),
                "path": relative_path,
            }
            events.append(event)
            if event_type == "agent_message" and detail:
                agent_messages.append(event)
    sort_key = lambda event: (event["timestamp"], event["path"], event["cursor"])
    events.sort(key=sort_key)
    agent_messages.sort(key=sort_key)
    if not events:
        return ("<none>", "<none>", "<none>"), agent_messages
    latest = events[-1]
    detail = " ".join(latest["detail"].split())[:180] or "<none>"
    return (latest["timestamp"], latest["type"], detail), agent_messages


def select_new_model_messages(messages: list[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
    if not messages:
        return previous_model_cursor, []
    latest_cursor = messages[-1]["cursor"]
    if not previous_model_cursor:
        return latest_cursor, messages[-model_output_messages:]
    previous_index = next(
        (
            index
            for index, message in enumerate(messages)
            if message["cursor"] == previous_model_cursor
        ),
        None,
    )
    if previous_index is None:
        return latest_cursor, messages[-1:]
    return latest_cursor, messages[previous_index + 1 :][-model_output_messages:]


def format_model_message(message: str) -> tuple[list[str], bool]:
    try:
        decoded = json.loads(message)
    except json.JSONDecodeError:
        rendered = message
    else:
        rendered = json.dumps(decoded, ensure_ascii=False, indent=2)
    wrapped: list[str] = []
    for line in rendered.splitlines() or [rendered]:
        indentation = line[: len(line) - len(line.lstrip())]
        content = line[len(indentation) :]
        pieces = textwrap.wrap(
            content,
            width=max(40, 118 - len(indentation)),
            replace_whitespace=False,
            drop_whitespace=False,
        ) or [""]
        wrapped.extend(indentation + piece for piece in pieces)
    truncated = len(wrapped) > model_output_lines
    return wrapped[:model_output_lines], truncated


def print_model_outputs(
    current_kind: str | None,
    messages: list[dict[str, str]],
) -> None:
    if current_kind != "hypothesis-investigation":
        print("  Harness 模型输出: 当前不在 Investigation/Experiment 阶段。")
        return
    if not messages:
        if previous_model_cursor:
            print("  Harness 模型输出: 本轮没有新增 agent_message。")
        else:
            print("  Harness 模型输出: 暂无 agent_message；模型可能正在推理或调用工具。")
        return
    print("  Harness 模型输出:")
    for message in messages:
        print(f"    [{message['timestamp']}] {message['session']}")
        lines, truncated = format_model_message(message["detail"])
        for line in lines:
            print(f"      {line}")
        if truncated:
            print("      ...（已截断，可用 --model-output-lines 调大）")


attempt_path_text = read_text(
    runtime_root / "state" / "latest_resume_attempt_path.txt", ""
)
attempt_root = Path(attempt_path_text) if attempt_path_text else None
if attempt_root is not None and attempt_root.is_dir():
    attempt_name = attempt_root.name
    phase = read_text(attempt_root / "state" / "phase", "<none>")
    controller_exit = read_text(
        attempt_root / "state" / "controller_exit", "<not-written>"
    )
    overall_exit = read_text(attempt_root / "state" / "exit", "<not-written>")
else:
    attempt_name = "<none>"
    phase = "<none>"
    controller_exit = "<not-written>"
    overall_exit = "<not-written>"

launcher_pid = read_launcher_pid()
controller_pids = find_matching_processes()
bundle_status, reason_code, blocking_error = read_bundle_status()
module_progress = read_research_task_module_progress()
if module_progress is None:
    pipeline, current_kind, current_stage = read_summary_progress(
        bundle_status,
        reason_code,
    )
else:
    pipeline, current_kind, current_stage = module_progress
    if current_kind is None and bundle_status in TERMINAL_RUN_STATUSES:
        current_stage = "任务已结束"
job_id, job_status, job_started_at = read_latest_job()
latest_event, model_messages = read_investigation_events()
event_at, event_type, event_detail = latest_event
model_cursor, new_model_messages = select_new_model_messages(model_messages)

print(f"__HARNESS_MONITOR_CURSOR__={model_cursor}")
print(f"[{datetime.now().astimezone().isoformat(timespec='seconds')}]")
print(f"  Launcher:  {'alive(pid=' + launcher_pid + ')' if launcher_pid else 'exited'}")
print(
    f"  Controller: {'alive(pid=' + controller_pids + ')' if controller_pids else 'exited'}"
)
print(
    f"  Services:  {catalog_port}={catalog_state} "
    f"{investigation_port}={investigation_state}"
)
print(f"  Attempt:   {attempt_name} phase={phase}")
print(
    f"  Run Bundle:status={bundle_status} reason={reason_code} "
    f"blocking={blocking_error}"
)
print(f"  当前阶段:  {current_stage}")
print("  任务阶段:")
if pipeline:
    for stage_label, stage_status in pipeline:
        print(f"    [{STATUS_LABELS.get(stage_status, stage_status)}] {stage_label}")
else:
    print("    <unavailable>")
print(f"  Job:       {job_id} status={job_status} started_at={job_started_at}")
print(f"  Harness 最新事件: {event_at} type={event_type} {event_detail}")
print_model_outputs(current_kind, new_model_messages)

if launcher_pid or controller_pids:
    print(f"  RESULT: {label} 当前仍在运行。")
    raise SystemExit(0)
if phase == "succeeded" or bundle_status in {
    "succeeded",
    "stopped_no_supported_hypothesis",
}:
    exit_detail = "" if overall_exit == "<not-written>" else f" {overall_exit}"
    print(f"  RESULT: {label} 已正常退出。{exit_detail}")
    raise SystemExit(10)
print(
    f"  RESULT: {label} 已退出；phase={phase}；"
    f"controller={controller_exit}；overall={overall_exit}"
)
if phase == "failed" or bundle_status == "failed" or controller_exit != "<not-written>":
    raise SystemExit(11)
raise SystemExit(12)
PY
}

printf '%s\n' \
  "$LABEL 只读监控已启动" \
  "$IDENTITY_LABEL：$MONITORED_ID（布局=$LAYOUT_MODE）" \
  "runtime：$RUNTIME_ROOT" \
  "间隔：${INTERVAL_SECONDS}s；按 Ctrl-C 只会停止监控脚本" \
  ""

while true; do
  catalog_state="$(read_port_state "$CATALOG_PORT")"
  investigation_state="$(read_port_state "$INVESTIGATION_PORT")"
  sample_output="$(print_status_sample "$catalog_state" "$investigation_state")"
  sample_exit=$?
  cursor_line="${sample_output%%$'\n'*}"
  if [[ "$cursor_line" == __HARNESS_MONITOR_CURSOR__=* ]]; then
    PREVIOUS_MODEL_CURSOR="${cursor_line#__HARNESS_MONITOR_CURSOR__=}"
    if [[ "$sample_output" == *$'\n'* ]]; then
      sample_output="${sample_output#*$'\n'}"
    else
      sample_output=""
    fi
  fi
  if [[ -n "$sample_output" ]]; then
    printf '%s\n' "$sample_output"
  fi
  case "$sample_exit" in
    0)
      if ((RUN_ONCE == 1)); then
        exit 0
      fi
      printf '\n'
      sleep "$INTERVAL_SECONDS"
      ;;
    10) exit 0 ;;
    11) exit 1 ;;
    12) exit 2 ;;
    *)
      printf '监控采样失败，内部状态码：%s\n' "$sample_exit" >&2
      exit "$sample_exit"
      ;;
  esac
done
