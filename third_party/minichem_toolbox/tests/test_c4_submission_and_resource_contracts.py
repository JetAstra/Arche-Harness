"""覆盖 C4 MiniChem 提交键、资源状态和保守进程终止合同。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from minichem_mcp_tools import discovery_tools, job_supervisor, open_execution
from minichem_mcp_tools.discovery_models import (
    ActionBatchRequest,
    ProgressiveActionRequest,
)
from minichem_mcp_tools.execution_models import (
    AnalysisJobRequest,
    ArtifactDeclarationRequest,
    JobCancelRequest,
    JobCollectRequest,
    NativeJobRequest,
    WorkspaceTextWriteRequest,
)
from minichem_toolbox import process_identity, resource_budget
from minichem_toolbox.artifacts import ArtifactStore
from minichem_toolbox.idempotency import (
    OperationIdempotencyStore,
    calculate_request_fingerprint,
    scoped_submission_operation_id,
)
from minichem_toolbox.models import ActionRequest
from minichem_toolbox.process_identity import ProcessIdentity
from minichem_toolbox.resource_budget import (
    ResourceStateUnavailable,
    active_resource_usage,
    enqueue_gaussian_job,
    read_gaussian_queue_snapshot,
)
from minichem_toolbox.scope import ScopeCapabilityRegistry, activate_scope


def _scope(tmp_path: Path, name: str = "one"):
    """创建一个带共享 scheduler 登记的隔离 capability scope。"""

    workspace = tmp_path / f"workspace-{name}"
    workspace.mkdir()
    registry = ScopeCapabilityRegistry(tmp_path / "registry")
    capability = registry.provision_scope_capability(
        task_id=f"task-{name}",
        batch_id=f"batch-{name}",
        combination_id=f"combination-{name}",
        scope_id=f"scope-{name}",
        workspace=workspace,
        token=(name.upper() + "A" * 48)[:48],
    )
    return replace(capability.scope, scheduler_root=tmp_path / "scheduler")


def test_submission_key_schema_and_scope_replay_conflict_and_isolation(
    tmp_path: Path,
) -> None:
    """公开 schema 暴露提交键，重放、冲突和跨 scope 都有确定结果。"""

    schema = ActionRequest.model_json_schema()
    assert "submission_key" in schema["properties"]
    assert "operation_id" not in schema["properties"]
    for request_type in (
        ProgressiveActionRequest,
        ActionBatchRequest,
        NativeJobRequest,
        AnalysisJobRequest,
        WorkspaceTextWriteRequest,
        JobCollectRequest,
        JobCancelRequest,
        ArtifactDeclarationRequest,
    ):
        request_schema = request_type.model_json_schema()
        assert "submission_key" in request_schema["properties"]
        assert "operation_id" not in json.dumps(request_schema, ensure_ascii=False)

    first_scope = _scope(tmp_path, "one")
    second_scope = _scope(tmp_path, "two")
    fingerprint = calculate_request_fingerprint("test_submission", {"value": 1})
    conflicting = calculate_request_fingerprint("test_submission", {"value": 2})
    calls: list[str] = []

    with activate_scope(first_scope):
        store = OperationIdempotencyStore()
        # 丢弃首次响应后重试，副作用仍只能发生一次。
        store.execute_scoped_submission(
            "stable-retry-key",
            fingerprint,
            lambda: calls.append("first") or {"status": "success", "value": 1},
        )
        replay = store.execute_scoped_submission(
            "stable-retry-key",
            fingerprint,
            lambda: calls.append("duplicate") or {"status": "success", "value": 2},
        )
        conflict = store.execute_scoped_submission(
            "stable-retry-key",
            conflicting,
            lambda: calls.append("conflict") or {"status": "success"},
        )
        first_operation_id = scoped_submission_operation_id("stable-retry-key")

    with activate_scope(second_scope):
        second_store = OperationIdempotencyStore()
        isolated = second_store.execute_scoped_submission(
            "stable-retry-key",
            fingerprint,
            lambda: calls.append("second") or {"status": "success", "value": 3},
        )
        second_operation_id = scoped_submission_operation_id("stable-retry-key")

    assert calls == ["first", "second"]
    assert replay["value"] == 1
    assert replay["submission_key"] == "stable-retry-key"
    assert conflict["error"]["code"] == "submission_key_conflict"
    assert isolated["value"] == 3
    assert first_operation_id != second_operation_id


def test_public_scope_side_effect_rejects_missing_or_forged_operation_identity(
    tmp_path: Path,
) -> None:
    """scope 副作用入口必须使用公开提交键，不能提交 operation_id。"""

    scope = _scope(tmp_path)
    with activate_scope(scope):
        missing = open_execution.write_workspace_text(
            WorkspaceTextWriteRequest(path="outputs/input.txt", content="x")
        )
        forged = open_execution.write_workspace_text(
            WorkspaceTextWriteRequest(
                path="outputs/input.txt",
                content="x",
                operation_id="caller-chosen-operation",
            )
        )
        forged_action = discovery_tools.execute_action(
            ProgressiveActionRequest(
                action_id="generate_gaussian_code",
                backend_id="gaussian_codegen",
                inputs={"user_input": "Single point energy."},
                operation_id="caller-chosen-action-operation",
            )
        )
    assert missing["error"]["code"] == "missing_submission_key"
    assert forged["error"]["code"] == "operation_id_not_assignable"
    assert forged_action["error"]["code"] == "operation_id_not_assignable"
    assert not (scope.workspace / "outputs" / "input.txt").exists()


def test_resource_budget_rejects_missing_scope_workspace_and_fake_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """缺工作区、伪造 CPU id 或数量不一致都不能被当成空闲资源。"""

    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_CPU_CORES", "2")
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_MEMORY_MB", "512")
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_GPU_COUNT", "0")
    scope = _scope(tmp_path)
    missing_scope = replace(scope, workspace=tmp_path / "does-not-exist")
    with activate_scope(missing_scope):
        with pytest.raises(ResourceStateUnavailable):
            active_resource_usage()

    with activate_scope(scope):
        assert active_resource_usage() == {
            "cpu_cores": 0,
            "memory_mb": 0,
            "gpu_count": 0,
        }
    index_directory = scope.scheduler_root / "scope_workspaces"
    unknown_entry = index_directory / "unexpected-index-entry"
    unknown_entry.write_text("x", encoding="utf-8")
    with activate_scope(scope):
        with pytest.raises(ResourceStateUnavailable):
            active_resource_usage()
    unknown_entry.unlink()
    missing_workspace = tmp_path / "missing-registered-workspace"
    missing_identity = f"missing-scope\0{missing_workspace}".encode("utf-8")
    missing_entry = index_directory / (
        "workspace_" + hashlib.sha256(missing_identity).hexdigest() + ".json"
    )
    missing_entry.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scope_id": "missing-scope",
                "workspace": str(missing_workspace),
            }
        ),
        encoding="utf-8",
    )
    missing_entry.chmod(0o600)
    with activate_scope(scope):
        with pytest.raises(ResourceStateUnavailable):
            active_resource_usage()
    missing_entry.unlink()

    job_id = "job_" + "1" * 32
    job_directory = scope.workspace / "outputs" / "execution_jobs" / job_id
    job_directory.mkdir(parents=True)
    (job_directory / "status.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "status": "running",
                "scope_id": scope.scope_id,
                "scope_workspace": str(scope.workspace),
                "resource_limits": {
                    "cpu_cores": 1,
                    "memory_mb": 128,
                    "gpu_count": 0,
                },
                "resource_allocation": {"cpu_ids": [999], "gpu_ids": []},
                "metadata": {"software_id": "orca"},
            }
        ),
        encoding="utf-8",
    )
    with activate_scope(scope):
        with pytest.raises(ResourceStateUnavailable):
            active_resource_usage()


def test_scoped_gaussian_ticket_and_job_inventory_fail_closed_on_unknown_state(
    tmp_path: Path,
) -> None:
    """scope ticket 的非法 job 身份和作业根未知条目都阻止恢复。"""

    scope = _scope(tmp_path)
    with activate_scope(scope):
        with pytest.raises(ResourceStateUnavailable):
            enqueue_gaussian_job("missing-job-id", kind="native")
        ticket = enqueue_gaussian_job(
            "owner",
            kind="native",
            job_id="job_" + "2" * 32,
        )
        ticket_path = (
            scope.scheduler_root
            / "resource_budget"
            / "gaussian_queue"
            / f"ticket_{ticket.ticket_id:020d}.json"
        )
        value = json.loads(ticket_path.read_text(encoding="utf-8"))
        value["job_id"] = "../other-job"
        ticket_path.write_text(json.dumps(value), encoding="utf-8")
        assert read_gaussian_queue_snapshot()["status"] == "unavailable"
        ticket.release()

        job_root = scope.workspace / "outputs" / "execution_jobs"
        job_root.mkdir(parents=True, exist_ok=True)
        (job_root / "unexpected-entry").write_text("x", encoding="utf-8")
        result = open_execution.verify_execution_references(
            open_execution.VerifyExecutionReferencesRequest()
        )
    assert result["inventory_complete"] is False


def test_missing_boot_id_is_unknown_and_pidfd_absence_does_not_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """持久记录缺 boot_id 或系统无 pidfd 时均保持未知且不发信号。"""

    monkeypatch.setattr(
        process_identity,
        "_read_proc_stat",
        lambda _pid: ("R", 42, 7, 8),
    )
    monkeypatch.setattr(process_identity, "_read_boot_id", lambda: "boot-a")
    assert process_identity.process_identity_activity(
        {"pid": 123, "start_time_ticks": 42}
    ) == "unknown"

    identity = ProcessIdentity(123, 42, "boot-a", 7, 8)
    monkeypatch.setattr(job_supervisor.os, "pidfd_open", None, raising=False)
    monkeypatch.setattr(
        job_supervisor.os,
        "killpg",
        lambda *_args: pytest.fail("不应在无 pidfd 时调用 killpg"),
        raising=False,
    )
    monkeypatch.setattr(job_supervisor, "process_identity_activity", lambda _value: "alive")
    monkeypatch.setattr(job_supervisor, "capture_process_identity", lambda _pid: identity)
    assert not job_supervisor._verified_process_group_signal(identity, 15)


def test_artifact_load_uses_one_byte_snapshot_for_hash_and_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ArtifactStore 解码的字节必须就是本次 hash 使用的同一快照。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    store = ArtifactStore(root=tmp_path)
    reference = store.put_json(
        {"value": 1},
        semantic_type="TestResult",
        producer_action="test",
        producer_backend="test",
    )
    assert store.load(reference) == {"value": 1}
