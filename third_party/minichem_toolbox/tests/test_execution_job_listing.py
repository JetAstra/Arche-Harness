"""验证模型可通过原生 MCP 能力读取当前 scope 的 execution job 清单。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from minichem_mcp_tools import open_execution
from minichem_mcp_tools.execution_models import ExecutionJobListRequest
from minichem_mcp_tools.execution_job_registry import ExecutionJobRegistry
from minichem_toolbox.scope import ScopeContext, activate_scope


def _scope(tmp_path: Path) -> ScopeContext:
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    return ScopeContext(
        task_id="task-1",
        batch_id="batch-1",
        combination_id="combo-1",
        scope_id="scope-1",
        workspace=workspace,
        capability_hash="a" * 64,
    )


def _write_status(scope: ScopeContext, job_id: str, job_type: str, status: str) -> None:
    directory = scope.workspace / "outputs" / "execution_jobs" / job_id
    directory.mkdir(parents=True, mode=0o700)
    (directory / "status.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "job_id": job_id,
                "job_type": job_type,
                "status": status,
                "scope_id": scope.scope_id,
                "scope_workspace": str(scope.workspace),
            }
        ),
        encoding="utf-8",
    )


def test_list_execution_jobs_returns_registry_jobs_and_submission_identity(
    tmp_path: Path,
) -> None:
    """清单统一覆盖 TS、native 和 analysis，并从持久状态计算 active。"""

    scope = _scope(tmp_path)
    registry = ExecutionJobRegistry(scope.workspace, scope.scope_id)
    entries = (
        ("job_01020304", "transition_state_path", "queued"),
        ("job_10203040", "native_software", "success"),
        ("job_ab12cd34", "programmable_analysis", "failed"),
    )
    for index, (job_id, job_type, status) in enumerate(entries):
        operation_id = f"operation-{index}"
        reservation = registry.reserve(
            job_type=job_type,
            operation_id=operation_id,
            operation_fingerprint=f"{index + 1}" * 64,
            submission_key=f"submission-{index}",
            preferred_job_id=job_id,
        )
        _write_status(scope, job_id, job_type, status)
        registry.mark(job_id, phase=status)
        assert reservation.job_id == job_id

    with activate_scope(scope):
        result = open_execution.list_execution_jobs(ExecutionJobListRequest())

    assert result["status"] == "success"
    assert result["inventory_complete"] is True
    assert result["active_job_id"] == "job_01020304"
    assert result["active_job_ids"] == ["job_01020304"]
    assert result["jobs"] == [
        {
            "job_id": "job_01020304",
            "job_status": "queued",
            "phase": "queued",
            "terminal": False,
            "operation_id": "operation-0",
            "submission_key": "submission-0",
        },
        {
            "job_id": "job_10203040",
            "job_status": "success",
            "phase": "success",
            "terminal": True,
            "operation_id": "operation-1",
            "submission_key": "submission-1",
        },
        {
            "job_id": "job_ab12cd34",
            "job_status": "failed",
            "phase": "failed",
            "terminal": True,
            "operation_id": "operation-2",
            "submission_key": "submission-2",
        },
    ]
    assert "path" not in json.dumps(result)
    assert "pid" not in json.dumps(result)


def test_list_execution_jobs_reports_explicit_empty_scope(tmp_path: Path) -> None:
    """没有提交记录的 scope 返回明确的空清单和 null active。"""

    scope = _scope(tmp_path)
    with activate_scope(scope):
        result = open_execution.list_execution_jobs(ExecutionJobListRequest())

    assert result == {
        "status": "success",
        "jobs": [],
        "active_job_id": None,
        "active_job_ids": [],
        "inventory_complete": True,
        "automatic_fallback": False,
    }


def test_list_execution_jobs_returns_all_active_jobs_in_stable_order(tmp_path: Path) -> None:
    """多个 active 作业都属于当前 scope 时，清单完整返回并稳定排序 active ID。"""

    scope = _scope(tmp_path)
    registry = ExecutionJobRegistry(scope.workspace, scope.scope_id)
    for index, job_id in enumerate(("job_01020304", "job_10203040")):
        registry.reserve(
            job_type="native_software",
            operation_id=f"operation-{index}",
            operation_fingerprint=f"{index + 1}" * 64,
            submission_key=f"submission-{index}",
            preferred_job_id=job_id,
        )
        _write_status(scope, job_id, "native_software", "running")
        registry.mark(job_id, phase="running")

    with activate_scope(scope):
        result = open_execution.list_execution_jobs(ExecutionJobListRequest())

    assert result["status"] == "success"
    assert result["inventory_complete"] is True
    assert [item["job_id"] for item in result["jobs"]] == [
        "job_01020304",
        "job_10203040",
    ]
    assert result["active_job_id"] is None
    assert result["active_job_ids"] == [
        "job_01020304",
        "job_10203040",
    ]
    assert "scope-1" not in json.dumps(result)


def test_list_execution_jobs_uses_terminal_status_before_registry_collect(
    tmp_path: Path,
) -> None:
    """worker 已写终态但模型尚未 get/collect 时，不再误报 active 或清单冲突。"""

    scope = _scope(tmp_path)
    registry = ExecutionJobRegistry(scope.workspace, scope.scope_id)
    registry.reserve(
        job_type="transition_state_path",
        operation_id="operation-terminal",
        operation_fingerprint="1" * 64,
        submission_key="submission-terminal",
        preferred_job_id="job_01020304",
    )
    registry.mark("job_01020304", phase="running")
    _write_status(scope, "job_01020304", "transition_state_path", "success")

    with activate_scope(scope):
        result = open_execution.list_execution_jobs(ExecutionJobListRequest())

    assert result["status"] == "success"
    assert result["active_job_id"] is None
    assert result["jobs"] == [
        {
            "job_id": "job_01020304",
            "job_status": "success",
            "phase": "success",
            "terminal": True,
            "operation_id": "operation-terminal",
            "submission_key": "submission-terminal",
        }
    ]
    assert registry.get("job_01020304")["phase"] == "running"


def test_list_execution_jobs_accepts_active_phase_progress_lag(tmp_path: Path) -> None:
    """持久状态已 running、registry 尚 queued 时仍返回唯一 active 作业。"""

    scope = _scope(tmp_path)
    registry = ExecutionJobRegistry(scope.workspace, scope.scope_id)
    registry.reserve(
        job_type="native_software",
        operation_id="operation-active",
        operation_fingerprint="2" * 64,
        submission_key="submission-active",
        preferred_job_id="job_01020304",
    )
    registry.mark("job_01020304", phase="queued")
    _write_status(scope, "job_01020304", "native_software", "running")

    with activate_scope(scope):
        result = open_execution.list_execution_jobs(ExecutionJobListRequest())

    assert result["status"] == "success"
    assert result["active_job_id"] == "job_01020304"
    assert result["jobs"][0]["job_status"] == "running"
    assert result["jobs"][0]["phase"] == "running"
    assert registry.get("job_01020304")["phase"] == "queued"


@pytest.mark.parametrize(
    "registry_phase",
    ["reserved", "preparing", "accepted", "queued", "running", "unknown", "legacy"],
)
def test_list_execution_jobs_accepts_active_status_with_preterminal_registry_phase(
    tmp_path: Path,
    registry_phase: str,
) -> None:
    """active 持久状态允许 registry 的保留、恢复和推进阶段短暂滞后。"""

    scope = _scope(tmp_path)
    registry = ExecutionJobRegistry(scope.workspace, scope.scope_id)
    registry.reserve(
        job_type="native_software",
        operation_id="operation-active-preterminal",
        operation_fingerprint="3" * 64,
        submission_key="submission-active-preterminal",
        preferred_job_id="job_01020304",
    )
    registry.mark("job_01020304", phase=registry_phase)
    _write_status(scope, "job_01020304", "native_software", "running")

    with activate_scope(scope):
        result = open_execution.list_execution_jobs(ExecutionJobListRequest())

    assert result["status"] == "success"
    assert result["active_job_id"] == "job_01020304"
    assert result["active_job_ids"] == ["job_01020304"]
    assert result["jobs"][0]["phase"] == "running"


@pytest.mark.parametrize(
    "registry_phase",
    ["reserved", "preparing", "accepted", "queued", "running", "unknown", "legacy"],
)
def test_list_execution_jobs_accepts_terminal_status_with_stale_preterminal_registry_phase(
    tmp_path: Path,
    registry_phase: str,
) -> None:
    """终态持久 status 可以覆盖尚未同步的 registry preterminal 阶段。"""

    scope = _scope(tmp_path)
    registry = ExecutionJobRegistry(scope.workspace, scope.scope_id)
    registry.reserve(
        job_type="native_software",
        operation_id="operation-terminal-preterminal",
        operation_fingerprint="4" * 64,
        submission_key="submission-terminal-preterminal",
        preferred_job_id="job_01020304",
    )
    registry.mark("job_01020304", phase=registry_phase)
    _write_status(scope, "job_01020304", "native_software", "success")

    with activate_scope(scope):
        result = open_execution.list_execution_jobs(ExecutionJobListRequest())

    assert result["status"] == "success"
    assert result["active_job_id"] is None
    assert result["active_job_ids"] == []
    assert result["jobs"][0]["phase"] == "success"


@pytest.mark.parametrize(
    "registry_phase", ["success", "failed", "timeout", "cancelled", "collected"]
)
def test_list_execution_jobs_rejects_active_status_with_terminal_registry_phase(
    tmp_path: Path,
    registry_phase: str,
) -> None:
    """持久 active 与 registry 终态或 collected 冲突时仍 fail-closed。"""

    scope = _scope(tmp_path)
    registry = ExecutionJobRegistry(scope.workspace, scope.scope_id)
    registry.reserve(
        job_type="native_software",
        operation_id="operation-active-terminal-conflict",
        operation_fingerprint="5" * 64,
        submission_key="submission-active-terminal-conflict",
        preferred_job_id="job_01020304",
    )
    registry.mark("job_01020304", phase=registry_phase)
    _write_status(scope, "job_01020304", "native_software", "running")

    with activate_scope(scope):
        result = open_execution.list_execution_jobs(ExecutionJobListRequest())

    assert result["status"] == "failed"
    assert result["inventory_complete"] is False
    assert result["jobs"] == []
    assert result["active_job_ids"] == []
    assert result["error"]["code"] == "execution_job_inventory_inconsistent"


def test_list_execution_jobs_fails_closed_on_registry_status_mismatch(tmp_path: Path) -> None:
    """registry 与持久 status 的作业族不一致时不返回部分清单。"""

    scope = _scope(tmp_path)
    registry = ExecutionJobRegistry(scope.workspace, scope.scope_id)
    registry.reserve(
        job_type="native_software",
        operation_id="operation-0",
        operation_fingerprint="1" * 64,
        submission_key="submission-0",
        preferred_job_id="job_01020304",
    )
    _write_status(scope, "job_01020304", "programmable_analysis", "running")
    registry.mark("job_01020304", phase="running")

    with activate_scope(scope):
        result = open_execution.list_execution_jobs(ExecutionJobListRequest())

    assert result["status"] == "failed"
    assert result["inventory_complete"] is False
    assert result["jobs"] == []
    assert result["error"]["code"] == "execution_job_inventory_inconsistent"


def test_list_execution_jobs_is_registered_in_full_and_investigation_profiles() -> None:
    """清单工具同时位于 open execution 和 Investigation 显式 allowlist。"""

    from minichem_mcp_tools.open_tools import OPEN_EXECUTION_TOOL_NAMES, TOOL_DESCRIPTIONS
    from minichem_mcp_tools.profiles import INVESTIGATION_TOOL_NAMES
    from minichem_mcp_tools.software_catalog import open_execution_prompt

    assert "list_execution_jobs" in OPEN_EXECUTION_TOOL_NAMES
    assert "list_execution_jobs" in INVESTIGATION_TOOL_NAMES
    assert "active_job_id" in TOOL_DESCRIPTIONS["list_execution_jobs"]
    assert "active_job_ids" in TOOL_DESCRIPTIONS["list_execution_jobs"]
    assert "list_execution_jobs" in open_execution_prompt(include_command_index=False)
