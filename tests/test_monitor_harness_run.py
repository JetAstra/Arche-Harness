"""通用 Harness 监控脚本的只读阶段和模型输出测试。"""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MONITOR_SCRIPT = REPOSITORY_ROOT / "scripts" / "monitor_harness_run.sh"
RESEARCH_TASK_ID = "research-task-monitor-fixture"


def _write_json(path: Path, payload: object) -> None:
    """在测试目录写入一份 JSON 运行投影。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _create_research_task_module_database(
    bundle_root: Path,
    current_kind: str,
    *,
    entry_strategy: str = "standard_hypothesis_generation",
) -> None:
    """创建监控脚本所需的最小 V2 只读账本。"""

    database_path = bundle_root / "internal/state/arche_memory.sqlite3"
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    connection.executescript(
        """
        CREATE TABLE research_tasks (
            research_task_id TEXT PRIMARY KEY,
            contract_version TEXT NOT NULL,
            policy_json TEXT NOT NULL
        );
        CREATE TABLE research_task_events (
            event_id TEXT PRIMARY KEY,
            research_task_id TEXT NOT NULL,
            research_task_sequence INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            payload_json TEXT NOT NULL
        );
        CREATE TABLE module_invocations (
            module_invocation_id TEXT PRIMARY KEY,
            research_task_id TEXT NOT NULL,
            module_kind TEXT NOT NULL,
            started_event_id TEXT NOT NULL,
            completed_event_id TEXT
        );
        """
    )
    connection.execute(
        "INSERT INTO research_tasks(research_task_id, contract_version, policy_json) VALUES (?, ?, ?)",
        (
            RESEARCH_TASK_ID,
            json.dumps("investigation_v2.task.v2"),
            json.dumps({"entry_strategy": entry_strategy}),
        ),
    )
    if entry_strategy == "author_hypothesis_import":
        module_kinds = (
            "hypothesis-import",
            "hypothesis-confirmation",
            "hypothesis-investigation",
            "task-review",
        )
    else:
        module_kinds = (
            "retrieval",
            "research-query-formulation",
            "hypothesis-generation",
            "hypothesis-selection",
            "hypothesis-confirmation",
            "hypothesis-investigation",
            "task-review",
        )
    for sequence, module_kind in enumerate(module_kinds, start=1):
        started_event_id = f"started-{sequence}"
        connection.execute(
            """
            INSERT INTO research_task_events(
                event_id,
                research_task_id,
                research_task_sequence,
                event_type,
                payload_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                started_event_id,
                RESEARCH_TASK_ID,
                sequence * 2,
                "ModuleInvocationStarted",
                json.dumps(
                    {"module_invocation_id": f"invocation-{module_kind}"}
                ),
            ),
        )
        completed_event_id = None
        if module_kind != current_kind:
            if module_kinds.index(module_kind) > module_kinds.index(current_kind):
                continue
            completed_event_id = f"completed-{sequence}"
            connection.execute(
                """
                INSERT INTO research_task_events(
                    event_id,
                    research_task_id,
                    research_task_sequence,
                    event_type,
                    payload_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    completed_event_id,
                    RESEARCH_TASK_ID,
                    sequence * 2 + 1,
                    "ModuleInvocationCompleted",
                    json.dumps(
                        {"module_invocation_id": f"invocation-{module_kind}"}
                    ),
                ),
            )
        connection.execute(
            """
            INSERT INTO module_invocations(
                module_invocation_id,
                research_task_id,
                module_kind,
                started_event_id,
                completed_event_id
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                f"invocation-{module_kind}",
                RESEARCH_TASK_ID,
                module_kind,
                started_event_id,
                completed_event_id,
            ),
        )
    connection.commit()
    connection.close()


def _record_current_module_failure(
    runtime_root: Path,
    current_kind: str,
) -> None:
    """按正式事件协议记录模块失败，不修改模块完成投影。"""

    bundle_root = runtime_root / "outputs/research_tasks" / RESEARCH_TASK_ID
    database_path = bundle_root / "internal/state/arche_memory.sqlite3"
    module_invocation_id = f"invocation-{current_kind}"
    connection = sqlite3.connect(database_path)
    next_sequence = int(
        connection.execute(
            "SELECT MAX(research_task_sequence) + 1 FROM research_task_events"
        ).fetchone()[0]
    )
    connection.execute(
        """
        INSERT INTO research_task_events(
            event_id,
            research_task_id,
            research_task_sequence,
            event_type,
            payload_json
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (
            "failed-current-module",
            RESEARCH_TASK_ID,
            next_sequence,
            "ModuleInvocationFailed",
            json.dumps(
                {
                    "module_invocation_id": module_invocation_id,
                    "failure_code": "verified_integrity_conflict",
                }
            ),
        ),
    )
    connection.commit()
    connection.close()
    _write_json(
        bundle_root / "status.json",
        {
            "schema_version": 1,
            "research_task_id": RESEARCH_TASK_ID,
            "status": "failed",
            "reason_code": "verified_integrity_conflict",
            "blocking": None,
        },
    )


def _create_monitor_runtime(
    tmp_path: Path,
    current_kind: str,
    *,
    entry_strategy: str = "standard_hypothesis_generation",
) -> Path:
    """创建含 V2 Run Bundle、Session rollout 和作业状态的测试 runtime。"""

    bundle_root = tmp_path / "outputs/research_tasks" / RESEARCH_TASK_ID
    _write_json(
        bundle_root / "status.json",
        {
            "schema_version": 1,
            "research_task_id": RESEARCH_TASK_ID,
            "status": "open",
            "reason_code": None,
            "blocking": None,
        },
    )
    _write_json(
        bundle_root / "summary.json",
        {
            "schema_version": 1,
            "research_task_id": RESEARCH_TASK_ID,
            "status": "open",
            "stage_order": [
                "literature_retrieval",
                "hypothesis_preparation",
                "investigation_and_experiment",
                "task_review_and_result",
            ],
            "stages": {},
        },
    )
    _create_research_task_module_database(
        bundle_root,
        current_kind,
        entry_strategy=entry_strategy,
    )

    session_root = (
        bundle_root
        / "internal/state/investigation_runtime/sessions/investigation-session-monitor"
    )
    rollout_path = (
        session_root
        / "codex_sessions/investigations/task-monitor/runtime-monitor/input-monitor"
        / "codex_home/sessions/2026/08/30/rollout-monitor.jsonl"
    )
    rollout_path.parent.mkdir(parents=True, exist_ok=True)
    model_message = json.dumps(
        {
            "narrative": "这是当前 Harness 模型的完整回答。",
            "observations": ["第一项观察", "第二项观察"],
        },
        ensure_ascii=False,
    )
    rollout_path.write_text(
        "\n".join(
            json.dumps(event, ensure_ascii=False)
            for event in (
                {
                    "timestamp": "2026-08-30T10:00:00.000Z",
                    "type": "event_msg",
                    "payload": {
                        "type": "agent_message",
                        "message": model_message,
                    },
                },
                {
                    "timestamp": "2026-08-30T10:00:01.000Z",
                    "type": "event_msg",
                    "payload": {
                        "type": "mcp_tool_call_end",
                        "invocation": {"tool": "get_execution_job"},
                    },
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        session_root / "workspace/outputs/execution_jobs/job_monitor/status.json",
        {
            "job_id": "job_monitor",
            "status": "running",
            "started_at": "2026-08-30T09:59:00.000Z",
        },
    )
    return tmp_path


def _run_monitor(runtime_root: Path, *extra_arguments: str) -> subprocess.CompletedProcess[str]:
    """执行一次监控采样并返回完整输出。"""

    return subprocess.run(
        [
            "bash",
            str(MONITOR_SCRIPT),
            "--runtime-root",
            str(runtime_root),
            "--research-task-id",
            RESEARCH_TASK_ID,
            "--once",
            "--python",
            sys.executable,
            "--process-pattern",
            "monitor-fixture-process-that-does-not-exist",
            *extra_arguments,
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_monitor_displays_complete_v2_pipeline_and_harness_model_output(
    tmp_path: Path,
) -> None:
    """Investigation 期间显示精确模块链、作业和当前模型回答。"""

    runtime_root = _create_monitor_runtime(tmp_path, "hypothesis-investigation")

    result = _run_monitor(runtime_root)

    assert result.returncode == 2
    assert "当前阶段:  Investigation/Experiment (Harness)" in result.stdout
    assert "[完成] Retrieval" in result.stdout
    assert "[完成] Research Query Formulation" in result.stdout
    assert "[完成] Hypothesis Generation" in result.stdout
    assert "[完成] Hypothesis Selection" in result.stdout
    assert "[完成] Hypothesis Confirmation" in result.stdout
    assert "[进行中] Investigation/Experiment (Harness)" in result.stdout
    assert "[未开始] Task Review" in result.stdout
    assert "job_monitor status=running" in result.stdout
    assert "Harness 最新事件:" in result.stdout
    assert "get_execution_job" in result.stdout
    assert "Harness 模型输出:" in result.stdout
    assert "这是当前 Harness 模型的完整回答。" in result.stdout
    assert "第一项观察" in result.stdout
    assert "__HARNESS_MONITOR_CURSOR__" not in result.stdout


def test_monitor_hides_investigation_answer_outside_harness_stage(
    tmp_path: Path,
) -> None:
    """进入 Task Review 后不把旧 Investigation 回答重复显示为当前输出。"""

    runtime_root = _create_monitor_runtime(tmp_path, "task-review")

    result = _run_monitor(runtime_root)

    assert result.returncode == 2
    assert "当前阶段:  Task Review" in result.stdout
    assert "Harness 模型输出: 当前不在 Investigation/Experiment 阶段。" in result.stdout
    assert "这是当前 Harness 模型的完整回答。" not in result.stdout


def test_monitor_displays_author_hypothesis_import_before_confirmation(
    tmp_path: Path,
) -> None:
    """作者假设直入流程显示 hypothesis-import，而不是 hypothesis-generation。"""

    runtime_root = _create_monitor_runtime(
        tmp_path,
        "hypothesis-confirmation",
        entry_strategy="author_hypothesis_import",
    )

    result = _run_monitor(runtime_root)

    assert result.returncode == 2
    assert "[完成] Hypothesis Import" in result.stdout
    assert "[进行中] Hypothesis Confirmation" in result.stdout
    assert "[未开始] Hypothesis Generation" not in result.stdout


def test_monitor_reports_module_failure_from_canonical_event(
    tmp_path: Path,
) -> None:
    """模块失败事件应关闭阶段，不能因完成投影为空而误报进行中。"""

    runtime_root = _create_monitor_runtime(
        tmp_path,
        "hypothesis-investigation",
        entry_strategy="author_hypothesis_import",
    )
    _record_current_module_failure(runtime_root, "hypothesis-investigation")

    result = _run_monitor(runtime_root)

    assert result.returncode == 1
    assert "当前阶段:  任务已结束" in result.stdout
    assert "[失败] Investigation/Experiment (Harness)" in result.stdout
    assert "[进行中] Investigation/Experiment (Harness)" not in result.stdout


def test_monitor_validates_model_output_limits(tmp_path: Path) -> None:
    """模型输出行数和条数必须位于脚本允许的正整数范围内。"""

    runtime_root = _create_monitor_runtime(tmp_path, "hypothesis-investigation")

    result = _run_monitor(runtime_root, "--model-output-lines", "0")

    assert result.returncode == 64
    assert "MODEL_OUTPUT_LINES 必须大于 0" in result.stderr


def test_monitor_uses_run_bundle_terminal_status_without_launcher_attempt(
    tmp_path: Path,
) -> None:
    """没有恢复 launcher 记录时，成功 Run Bundle 仍按正常终态退出。"""

    runtime_root = _create_monitor_runtime(tmp_path, "task-review")
    bundle_root = runtime_root / "outputs/research_tasks" / RESEARCH_TASK_ID
    (bundle_root / "internal/state/arche_memory.sqlite3").unlink()
    _write_json(
        bundle_root / "status.json",
        {
            "schema_version": 1,
            "research_task_id": RESEARCH_TASK_ID,
            "status": "succeeded",
            "reason_code": None,
            "blocking": None,
        },
    )
    _write_json(
        bundle_root / "summary.json",
        {
            "schema_version": 1,
            "research_task_id": RESEARCH_TASK_ID,
            "status": "succeeded",
            "stage_order": [
                "literature_retrieval",
                "hypothesis_preparation",
                "investigation_and_experiment",
                "task_review_and_result",
            ],
            "stages": {
                stage: {"status": "completed"}
                for stage in (
                    "literature_retrieval",
                    "hypothesis_preparation",
                    "investigation_and_experiment",
                    "task_review_and_result",
                )
            },
        },
    )

    result = _run_monitor(runtime_root)

    assert result.returncode == 0
    assert "当前阶段:  任务已结束" in result.stdout
    assert "RESULT: Harness 已正常退出。" in result.stdout
    assert "<not-written>" not in result.stdout.split("RESULT:", maxsplit=1)[1]


def test_monitor_reports_formal_investigation_waiting_status(
    tmp_path: Path,
) -> None:
    """正式 Investigation 作业等待状态不应退化为旧 waiting_execution。"""

    runtime_root = _create_monitor_runtime(tmp_path, "hypothesis-investigation")
    bundle_root = runtime_root / "outputs/research_tasks" / RESEARCH_TASK_ID
    (bundle_root / "internal/state/arche_memory.sqlite3").unlink()
    _write_json(
        bundle_root / "status.json",
        {
            "schema_version": 1,
            "research_task_id": RESEARCH_TASK_ID,
            "status": "open",
            "reason_code": "waiting_investigation_job",
            "blocking": {"kind": "investigation_job"},
        },
    )
    _write_json(
        bundle_root / "summary.json",
        {
            "schema_version": 1,
            "research_task_id": RESEARCH_TASK_ID,
            "status": "open",
            "stage_order": [
                "literature_retrieval",
                "hypothesis_preparation",
                "investigation_and_experiment",
                "task_review_and_result",
            ],
            "stages": {
                "literature_retrieval": {"status": "completed"},
                "hypothesis_preparation": {"status": "completed"},
                "investigation_and_experiment": {
                    "status": "waiting_investigation_job"
                },
                "task_review_and_result": {"status": "not_started"},
            },
        },
    )

    result = _run_monitor(runtime_root)

    assert result.returncode == 2
    assert "当前阶段:  Investigation/Experiment (Harness)" in result.stdout
    assert "[等待 Investigation 作业] Investigation/Experiment (Harness)" in result.stdout
    assert "waiting_execution" not in result.stdout


def test_monitor_formal_reader_does_not_accept_legacy_sql_columns() -> None:
    """正式 research_task 读取器不应偷偷选择 schema-v10 列名。"""

    source = MONITOR_SCRIPT.read_text(encoding="utf-8")
    formal_reader = source.split("def read_legacy_module_progress(", maxsplit=1)[0]
    assert '"task_id"' not in formal_reader
    assert "execution_policy_json" not in formal_reader
    assert "module_run_id" not in formal_reader
    assert "v2_tasks" not in formal_reader
    assert "v2_module_runs" not in formal_reader
