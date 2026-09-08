"""TS Pipeline 异步作业的提交、唤醒、收集和恢复导入合同测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from minichem_toolbox.artifacts import ArtifactStore, read_file_snapshot
from minichem_toolbox.models import ActionResult
from minichem_toolbox.scope import ScopeContext, activate_scope
from minichem_mcp_tools import transition_state_jobs as jobs
from minichem_mcp_tools import open_execution, open_tools
from minichem_mcp_tools.execution_job_registry import ExecutionJobRegistry
from minichem_mcp_tools.execution_models import JobCollectRequest, JobStatusRequest
from minichem_mcp_tools.discovery_models import ProgressiveActionRequest


def _scope(root: Path) -> ScopeContext:
    """构造测试用的固定 capability scope。"""

    root.mkdir(parents=True, exist_ok=True)
    return ScopeContext(
        task_id="task-1",
        batch_id="batch-1",
        combination_id="combination-1",
        scope_id="scope-1",
        workspace=root,
        capability_hash="capability-1",
    )


def _request(submission_key: str) -> jobs.TransitionStatePathJobRequest:
    """构造包含最小科学输入和 GPU 约束的 TS request。"""

    return jobs.TransitionStatePathJobRequest.model_validate(
        {
            "submission_key": submission_key,
            "backend_id": jobs.BACKEND_ID,
            "inputs": {"reactants": ["CC"], "products": ["CO"]},
            "resource_limits": {"gpu_count": 1},
        }
    )


def _fake_start_job(
    monkeypatch: pytest.MonkeyPatch,
    scope: ScopeContext,
    captured: dict[str, object] | None = None,
) -> list[str]:
    """用持久控制文件模拟 detached supervisor，避免测试启动科学进程。"""

    started: list[str] = []

    def start_job(**kwargs):
        assert kwargs["resource_limits"]["walltime_seconds"] == jobs.timeout_seconds_for(
            "compute"
        )
        assert {
            key: kwargs["resource_limits"][key]
            for key in ("cpu_cores", "memory_mb", "gpu_count")
        } == {"cpu_cores": 60, "memory_mb": 320000, "gpu_count": 1}
        operation_id = str(kwargs["operation_id"])
        job_id = str(kwargs["job_id"])
        directory = scope.workspace / "outputs" / "execution_jobs" / job_id
        directory.mkdir(parents=True, exist_ok=False)
        metadata = dict(kwargs["metadata"])
        request_record = {
            "schema_version": 1,
            "job_id": job_id,
            "job_type": jobs.JOB_TYPE,
            "runtime": "ash_neb",
            "metadata": metadata,
            "scope_id": scope.scope_id,
            "scope_workspace": str(scope.workspace),
            "operation_id": operation_id,
            "operation_fingerprint": kwargs["operation_fingerprint"],
            "caller_request_fingerprint": kwargs["caller_request_fingerprint"],
            "submission_key": kwargs["submission_key"],
        }
        status = {
            "schema_version": 1,
            "job_id": job_id,
            "job_type": jobs.JOB_TYPE,
            "status": "queued",
            "scope_id": scope.scope_id,
            "scope_workspace": str(scope.workspace),
            "operation_id": operation_id,
            "operation_fingerprint": kwargs["operation_fingerprint"],
            "caller_request_fingerprint": kwargs["caller_request_fingerprint"],
            "submission_key": kwargs["submission_key"],
        }
        jobs._atomic_json(directory / "request.json", request_record)
        jobs._atomic_json(directory / "status.json", status)
        if captured is not None:
            captured["resource_limits"] = dict(kwargs["resource_limits"])
            captured["transition_state_path_request"] = dict(
                metadata["transition_state_path_request"]
            )
        started.append(job_id)
        return {"status": "success", "job_id": job_id, "job_status": "queued"}

    monkeypatch.setattr(jobs, "_start_job", start_job)
    monkeypatch.setattr(jobs, "active_catalog_hash", lambda: "catalog-test")
    monkeypatch.setattr(
        jobs,
        "execute_traced",
        lambda _name, _arguments, function, **_kwargs: function(),
    )
    return started


@pytest.mark.parametrize(
    "request_resource_limits",
    [None, {"cpu_cores": 1, "memory_mb": 4096, "gpu_count": 0}],
)
def test_submit_overrides_worker_request_resources_with_fixed_evaluation_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request_resource_limits: dict[str, int] | None,
) -> None:
    """模型省略或传入旧资源值时，worker request 仍使用服务固定评测预算。"""

    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_CPU_CORES", "60")
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_MEMORY_MB", "320000")
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_GPU_COUNT", "1")
    scope = _scope(tmp_path)
    captured: dict[str, object] = {}
    _fake_start_job(monkeypatch, scope, captured)
    request_value: dict[str, object] = {
        "submission_key": "fixed-worker-resources",
        "backend_id": jobs.BACKEND_ID,
        "inputs": {"reactants": ["CC"], "products": ["CO"]},
    }
    if request_resource_limits is not None:
        request_value["resource_limits"] = request_resource_limits
    request = jobs.TransitionStatePathJobRequest.model_validate(request_value)

    with activate_scope(scope):
        submitted = jobs.submit_transition_state_path_job(request)
        equivalent_without_resource_selection = dict(request_value)
        equivalent_without_resource_selection.pop("resource_limits", None)
        replayed = jobs.submit_transition_state_path_job(
            jobs.TransitionStatePathJobRequest.model_validate(
                equivalent_without_resource_selection
            )
        )

    assert submitted["status"] == "success"
    assert replayed["job_id"] == submitted["job_id"]
    assert replayed["idempotent_reused"] is True
    expected = {"cpu_cores": 60, "memory_mb": 320000, "gpu_count": 1}
    assert captured["resource_limits"] == {
        **expected,
        "walltime_seconds": jobs.timeout_seconds_for("compute"),
    }
    assert captured["transition_state_path_request"]["resource_limits"] == expected


def test_detached_worker_exception_message_survives_collect_and_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """detached worker 的 ValueError 原因贯通 raw、collect 和诊断 Artifact。"""

    scope = _scope(tmp_path)
    _fake_start_job(monkeypatch, scope)
    request = _request("detached-value-error")
    expected_message = "reaction mapping failed: no valid atom correspondence"

    with activate_scope(scope):
        submitted = jobs.submit_transition_state_path_job(request)
        job_id = str(submitted["job_id"])
        directory = scope.workspace / "outputs" / "execution_jobs" / job_id

        def raise_value_error(_request: dict, _job_id: str) -> dict:
            """模拟 TS Pipeline 在 detached worker 中抛出的具体输入错误。"""

            raise ValueError(expected_message)

        monkeypatch.setattr(jobs, "_worker_action_result", raise_value_error)
        monkeypatch.chdir(directory)
        assert jobs._run_worker() == 0
        raw_result = json.loads(
            (directory / jobs.RAW_RESULT_FILENAME).read_text(encoding="utf-8")
        )
        assert raw_result["error"]["exception_message"] == expected_message
        assert raw_result["retryable"] is False

        _set_job_status(scope, job_id, "success")
        collected = jobs.collect_transition_state_path_job(
            jobs.TransitionStatePathJobCollectRequest(job_id=job_id)
        )

    action_result = collected["action_result"]
    assert action_result["error"]["exception_message"] == expected_message
    assert action_result["retryable"] is False
    diagnostic_reference = next(
        item
        for item in action_result["output_artifacts"]
        if item["semantic_type"] == "BackendDiagnostic"
    )
    diagnostic = ArtifactStore(root=scope.workspace, create=False).load(
        diagnostic_reference
    )
    assert diagnostic["error"]["exception_message"] == expected_message
    assert "traceback" not in diagnostic["error"]


def _set_job_status(scope: ScopeContext, job_id: str, status: str) -> None:
    """修改模拟 supervisor 的最小终态文件。"""

    path = scope.workspace / "outputs" / "execution_jobs" / job_id / "status.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["status"] = status
    jobs._atomic_json(path, value)


def test_submit_is_idempotent_and_second_active_submission_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同 key 复用原 job，不同 key 在当前作业收集前被 phase gate 拒绝。"""

    scope = _scope(tmp_path)
    started = _fake_start_job(monkeypatch, scope)
    with activate_scope(scope):
        first = jobs.submit_transition_state_path_job(_request("one"))
        reused = jobs.submit_transition_state_path_job(_request("one"))
        blocked = jobs.submit_transition_state_path_job(_request("two"))
    assert first["job_id"] == reused["job_id"] == started[0]
    assert reused["idempotent_reused"] is True
    assert blocked["valid"] is False
    assert blocked["error"]["code"] == "transition_state_path_phase_gate_closed"
    assert started == [first["job_id"]]


@pytest.mark.parametrize("terminal_status", ["failed", "timeout", "cancelled"])
def test_terminal_job_without_result_can_be_collected_and_releases_active_gate(
    terminal_status: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TS 作业失败、超时或取消后 collect 都必须释放 scope gate。"""

    scope = _scope(tmp_path)
    started = _fake_start_job(monkeypatch, scope)
    with activate_scope(scope):
        first = jobs.submit_transition_state_path_job(_request("terminal-one"))
        _set_job_status(scope, str(first["job_id"]), terminal_status)
        collected = jobs.collect_transition_state_path_job(
            jobs.TransitionStatePathJobCollectRequest(job_id=str(first["job_id"]))
        )
        second = jobs.submit_transition_state_path_job(_request("terminal-two"))

    assert collected["status"] == "success"
    assert collected["job_status"] == terminal_status
    assert collected["error"]["code"] == f"execution_{terminal_status}"
    assert second["status"] == "success"
    assert len(started) == 2


def test_resource_failure_with_job_id_is_queryable_collectable_and_releases_active_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """带 job_id 的资源失败必须可 get/collect，完成后允许新的 submission_key。"""

    scope = _scope(tmp_path)
    calls: list[str] = []

    def start_job(**kwargs):
        job_id = str(kwargs["job_id"])
        calls.append(job_id)
        if len(calls) == 1:
            return {
                "status": "invalid_request",
                "valid": False,
                "job_id": job_id,
                "error": {
                    "code": "resource_budget_exceeded",
                    "message": "GPU budget is already reserved",
                },
            }
        directory = scope.workspace / "outputs" / "execution_jobs" / job_id
        directory.mkdir(parents=True, exist_ok=False)
        jobs._atomic_json(
            directory / "status.json",
            {
                "schema_version": 1,
                "job_id": job_id,
                "job_type": jobs.JOB_TYPE,
                "status": "queued",
                "scope_id": scope.scope_id,
                "scope_workspace": str(scope.workspace),
                "operation_id": kwargs["operation_id"],
                "operation_fingerprint": kwargs["operation_fingerprint"],
                "caller_request_fingerprint": kwargs["caller_request_fingerprint"],
                "submission_key": kwargs["submission_key"],
            },
        )
        return {"status": "success", "job_id": job_id, "job_status": "queued"}

    monkeypatch.setattr(jobs, "_start_job", start_job)
    monkeypatch.setattr(jobs, "active_catalog_hash", lambda: "catalog-test")
    monkeypatch.setattr(
        jobs,
        "execute_traced",
        lambda _name, _arguments, function, **_kwargs: function(),
    )
    with activate_scope(scope):
        failed = jobs.submit_transition_state_path_job(_request("resource-failure"))
        job_id = failed["job_id"]
        assert failed["status"] == "success"
        assert failed["job_status"] == "failed"
        assert failed["terminal"] is True
        assert failed["error"]["code"] == "resource_budget_exceeded"
        queried = jobs.get_transition_state_path_job(
            jobs.TransitionStatePathJobStatusRequest(job_id=job_id)
        )
        assert queried["job_status"] == "failed"
        assert queried["terminal"] is True
        assert queried["error"]["message"] == "GPU budget is already reserved"
        facade_queried = open_tools.get_execution_job(JobStatusRequest(job_id=job_id))
        assert facade_queried["job_status"] == "failed"
        assert facade_queried["terminal"] is True
        assert facade_queried["error"] == {
            "code": "resource_budget_exceeded",
            "message": "GPU budget is already reserved",
        }
        collected = jobs.collect_transition_state_path_job(
            jobs.TransitionStatePathJobCollectRequest(job_id=job_id)
        )
        assert collected == {
            "status": "success",
            "ready": True,
            "job_id": job_id,
            "job_status": "failed",
            "operation_id": failed["operation_id"],
            "submission_key": "resource-failure",
            "action_id": jobs.ACTION_ID,
            "backend_id": jobs.BACKEND_ID,
            "error": {
                "code": "resource_budget_exceeded",
                "message": "GPU budget is already reserved",
            },
            "automatic_fallback": False,
        }
        retried = jobs.submit_transition_state_path_job(_request("resource-retry"))
    assert retried["status"] == "success"
    assert retried["job_status"] == "queued"
    state = json.loads(
        (scope.workspace / "_tool_artifacts" / jobs.STATE_DIRECTORY_NAME / jobs.STATE_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert state["active_submission_key"] == "resource-retry"
    failed_status = json.loads(
        (
            scope.workspace
            / "outputs"
            / "execution_jobs"
            / job_id
            / "status.json"
        ).read_text(encoding="utf-8")
    )
    assert failed_status["status"] == "failed"
    assert failed_status["error"]["code"] == "resource_budget_exceeded"


def test_get_is_available_before_controller_arm_and_returns_minimal_running_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """模型可在同一 turn 凭 job_id 查询，Controller 轮询仍不泄露科学 payload。"""

    scope = _scope(tmp_path)
    _fake_start_job(monkeypatch, scope)
    with activate_scope(scope):
        submitted = jobs.submit_transition_state_path_job(_request("status"))
        job_id = submitted["job_id"]
        before_arm = jobs.get_transition_state_path_job(
            jobs.TransitionStatePathJobStatusRequest(job_id=job_id)
        )
        assert before_arm["job_status"] == "queued"
        assert "action_result" not in before_arm
        _set_job_status(scope, job_id, "running")
        polled = jobs.controller_get_transition_state_path_job(job_id)
        _set_job_status(scope, job_id, "success")
        jobs.controller_arm_transition_state_path_job(job_id)
        after_wake = jobs.get_transition_state_path_job(
            jobs.TransitionStatePathJobStatusRequest(job_id=job_id)
        )
    assert polled["job_status"] == "running"
    assert "action_result" not in polled
    assert "output_artifacts" not in polled
    assert after_wake["job_status"] == "success"
    assert "action_result" not in after_wake


def test_collect_is_available_without_controller_arm_after_terminal_get(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """模型在未 arm 的正常 turn 内完成 terminal get 和 collect。"""

    scope = _scope(tmp_path)
    _fake_start_job(monkeypatch, scope)
    with activate_scope(scope):
        submitted = jobs.submit_transition_state_path_job(_request("same-turn-collect"))
        job_id = submitted["job_id"]
        _set_job_status(scope, job_id, "success")
        terminal = jobs.get_transition_state_path_job(
            jobs.TransitionStatePathJobStatusRequest(job_id=job_id)
        )
        assert terminal["job_status"] == "success"
        directory = scope.workspace / "outputs" / "execution_jobs" / job_id
        output = directory / "outputs" / "worker_result.txt"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("worker-result\n", encoding="utf-8")
        jobs._atomic_json(
            directory / jobs.RAW_RESULT_FILENAME,
            {
                "status": "success",
                "result": {"workflow_id": "same-turn", "artifact": None},
                "artifact_files": [
                    {
                        "path": "outputs/worker_result.txt",
                        "semantic_type": "TSOutput",
                        "media_type": "text/plain",
                        "result_pointer": "/artifact",
                    }
                ],
            },
        )
        collected = jobs.collect_transition_state_path_job(
            jobs.TransitionStatePathJobCollectRequest(job_id=job_id)
        )
    assert collected["status"] == "success"
    assert collected["job_id"] == job_id


def test_generic_submit_fingerprint_survives_collect_after_internal_route_marker_is_persisted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """统一入口转发的隐藏 backend 标记不应阻断同一 TS 作业的 collect。"""

    scope = _scope(tmp_path)
    _fake_start_job(monkeypatch, scope)
    base_request = _request("generic-route-marker")
    request_value = base_request.model_dump(mode="python")
    request_value["resource_limits"] = base_request.resource_limits
    request_value["requested_backend_id"] = jobs.BACKEND_ID
    request = jobs.TransitionStatePathJobRequest.model_construct(**request_value)
    assert jobs._submission_request_fingerprint(
        request
    ) == jobs._submission_request_fingerprint(base_request)

    with activate_scope(scope):
        submitted = jobs.submit_transition_state_path_job(request)
        job_id = str(submitted["job_id"])
        _set_job_status(scope, job_id, "success")
        jobs._atomic_json(
            scope.workspace
            / "outputs"
            / "execution_jobs"
            / job_id
            / jobs.RAW_RESULT_FILENAME,
            {
                "status": "success",
                "result": {"workflow_id": "generic-route-marker"},
                "artifact_files": [],
            },
        )
        collected = jobs.collect_transition_state_path_job(
            jobs.TransitionStatePathJobCollectRequest(job_id=job_id)
        )

    assert collected["status"] == "success"
    assert (
        collected["action_result"]["result"]["workflow_id"]
        == "generic-route-marker"
    )


def test_collect_accepts_persisted_pre_fix_route_marker_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """旧版已落盘的 TS 指纹仍可在隐藏字段丢失后恢复并 collect。"""

    scope = _scope(tmp_path)
    _fake_start_job(monkeypatch, scope)
    base_request = _request("legacy-route-marker")
    request_value = base_request.model_dump(mode="python")
    request_value["resource_limits"] = base_request.resource_limits
    request_value["requested_backend_id"] = jobs.BACKEND_ID
    request = jobs.TransitionStatePathJobRequest.model_construct(**request_value)
    current_fingerprint = jobs._submission_request_fingerprint
    legacy_fingerprint = jobs._legacy_route_marker_submission_request_fingerprint(request)
    monkeypatch.setattr(
        jobs,
        "_submission_request_fingerprint",
        lambda _request: legacy_fingerprint,
    )

    with activate_scope(scope):
        submitted = jobs.submit_transition_state_path_job(request)
        job_id = str(submitted["job_id"])
        _set_job_status(scope, job_id, "success")
        jobs._atomic_json(
            scope.workspace
            / "outputs"
            / "execution_jobs"
            / job_id
            / jobs.RAW_RESULT_FILENAME,
            {
                "status": "success",
                "result": {"workflow_id": "legacy-route-marker"},
                "artifact_files": [],
            },
        )
        monkeypatch.setattr(jobs, "_submission_request_fingerprint", current_fingerprint)
        collected = jobs.collect_transition_state_path_job(
            jobs.TransitionStatePathJobCollectRequest(job_id=job_id)
        )

    assert collected["status"] == "success"
    assert (
        collected["action_result"]["result"]["workflow_id"]
        == "legacy-route-marker"
    )


def test_unified_facade_routes_ts_collect_but_keeps_get_status_type_agnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """统一 get 只返回公共状态；统一 collect 复用 TS canonical 结果和 active 清理。"""

    scope = _scope(tmp_path)
    _fake_start_job(monkeypatch, scope)
    monkeypatch.setattr(
        open_tools,
        "execute_traced",
        lambda _name, _arguments, function, **_kwargs: function(),
    )
    with activate_scope(scope):
        submitted = jobs.submit_transition_state_path_job(_request("facade"))
        job_id = submitted["job_id"]
        status = open_tools.get_execution_job(JobStatusRequest(job_id=job_id))
        assert status == {
            "status": "success",
            "job_id": job_id,
            "job_status": "queued",
            "terminal": False,
            "job": {"job_id": job_id, "status": "queued"},
        }
        wrong_submission = open_tools.collect_execution_job(
            JobCollectRequest(job_id=job_id, submission_key="unrelated")
        )
        assert wrong_submission["error"]["code"] == "submission_key_conflict"
        _set_job_status(scope, job_id, "success")
        directory = scope.workspace / "outputs" / "execution_jobs" / job_id
        output = directory / "outputs" / "facade.txt"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("facade\n", encoding="utf-8")
        jobs._atomic_json(
            directory / jobs.RAW_RESULT_FILENAME,
            {
                "status": "success",
                "result": {"workflow_id": "facade"},
                "artifact_files": [
                    {
                        "path": "outputs/facade.txt",
                        "semantic_type": "TSOutput",
                        "media_type": "text/plain",
                        "result_pointer": "/workflow_id",
                    }
                ],
            },
        )
        collected = open_tools.collect_execution_job(
            JobCollectRequest(job_id=job_id, submission_key="facade")
        )
        replay = open_tools.collect_execution_job(
            JobCollectRequest(job_id=job_id, submission_key="facade")
        )
        after = open_tools.get_execution_job(JobStatusRequest(job_id=job_id))
        state = json.loads(
            (scope.workspace / "_tool_artifacts" / jobs.STATE_DIRECTORY_NAME / jobs.STATE_FILENAME).read_text(
                encoding="utf-8"
            )
        )
        registry_record = ExecutionJobRegistry(
            workspace=scope.workspace,
            scope_id=scope.scope_id,
        ).get(job_id)
    assert collected["status"] == replay["status"] == "success"
    assert collected["job_id"] == replay["job_id"] == job_id
    assert collected["action_result"] == replay["action_result"]
    assert after["job_status"] == "success"
    assert after["terminal"] is True
    assert registry_record["phase"] == "collected"
    assert state["active_submission_key"] is None
    assert state["active_job_id"] is None


def test_collect_nonterminal_is_rejected_after_race_and_terminal_arm_collects_parent_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Controller arm 后若状态回退则 fail-closed；终态收集在父 scope 注册并可读 artifact。"""

    scope = _scope(tmp_path)
    _fake_start_job(monkeypatch, scope)
    with activate_scope(scope):
        submitted = jobs.submit_transition_state_path_job(_request("collect"))
        job_id = submitted["job_id"]
        _set_job_status(scope, job_id, "success")
        armed = jobs.controller_arm_transition_state_path_job(job_id)
        assert armed["wake_armed"] is True
        _set_job_status(scope, job_id, "running")
        raced = jobs.collect_transition_state_path_job(
            jobs.TransitionStatePathJobCollectRequest(job_id=job_id)
        )
        assert raced["error"]["code"] == "transition_state_path_job_not_terminal"
        _set_job_status(scope, job_id, "success")
        directory = scope.workspace / "outputs" / "execution_jobs" / job_id
        output = directory / "outputs" / "worker_result.txt"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("worker-result\n", encoding="utf-8")
        jobs._atomic_json(
            directory / jobs.RAW_RESULT_FILENAME,
            {
                "status": "success",
                "result": {"workflow_id": "test", "artifact": None},
                "artifact_files": [
                    {
                        "path": "outputs/worker_result.txt",
                        "semantic_type": "TSOutput",
                        "media_type": "text/plain",
                        "result_pointer": "/artifact",
                    }
                ],
            },
        )
        collected = jobs.collect_transition_state_path_job(
            jobs.TransitionStatePathJobCollectRequest(job_id=job_id)
        )
    assert collected["status"] == "success"
    action_result = ActionResult.model_validate(collected["action_result"])
    assert action_result.action == jobs.ACTION_ID
    assert action_result.backend == jobs.BACKEND_ID
    assert action_result.output_artifacts
    assert action_result.result["artifact"]["artifact_id"].startswith("art_")
    store = ArtifactStore(root=scope.workspace, create=False)
    for reference in action_result.output_artifacts:
        assert store.find(reference.artifact_id) == reference
        path = scope.workspace / reference.path
        _, digest = read_file_snapshot(path)
        assert digest == reference.sha256


def test_collect_large_action_result_does_not_duplicate_artifact_refs_in_mcp_envelope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """大规模 TS collect 只在 action_result 中保留完整 ArtifactRef 清单。"""

    scope = _scope(tmp_path)
    _fake_start_job(monkeypatch, scope)
    artifact_count = 445
    with activate_scope(scope):
        submitted = jobs.submit_transition_state_path_job(_request("large-collect"))
        job_id = submitted["job_id"]
        _set_job_status(scope, job_id, "success")
        artifact_directory = scope.workspace / "outputs" / "large-collect"
        artifact_directory.mkdir(parents=True, exist_ok=True)
        store = ArtifactStore(root=scope.workspace)
        references = []
        for index in range(artifact_count):
            path = artifact_directory / f"artifact-{index:03d}.json"
            path.write_text(json.dumps({"index": index}), encoding="utf-8")
            references.append(
                store.register_file(
                    path,
                    semantic_type="TSOutput",
                    producer_action=jobs.ACTION_ID,
                    producer_backend=jobs.BACKEND_ID,
                    media_type="application/json",
                ).model_dump(mode="json")
            )
        action_result = ActionResult(
            status="success",
            action=jobs.ACTION_ID,
            action_version="1.0.0",
            requested_backend=jobs.BACKEND_ID,
            backend=jobs.BACKEND_ID,
            selection_source="agent",
            result={"workflow_id": "large-collect"},
            output_artifacts=references,
        ).model_dump(mode="json")
        monkeypatch.setattr(
            jobs,
            "_read_transition_state_path_result",
            lambda _workspace, _job_id: action_result,
        )
        collected = jobs.collect_transition_state_path_job(
            jobs.TransitionStatePathJobCollectRequest(job_id=job_id)
        )

    assert "output_artifacts" not in collected
    assert collected["action_result"]["output_artifacts"] == references
    mcp_envelope = {
        "content": [
            {
                "type": "text",
                "text": json.dumps(collected, ensure_ascii=False, separators=(",", ":")),
            }
        ]
    }
    outer_text = json.dumps(mcp_envelope, ensure_ascii=False, separators=(",", ":"))
    assert len(outer_text.encode("utf-8")) < 1_048_576
    outer_decoded = json.loads(outer_text)
    inner_decoded = json.loads(outer_decoded["content"][0]["text"])
    assert inner_decoded["action_result"]["output_artifacts"] == references
    assert "output_artifacts" not in inner_decoded


def test_adopt_completed_action_result_is_strict_and_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Controller 可导入真实 completed envelope，但不能越权替换请求或 artifact。"""

    scope = _scope(tmp_path)
    monkeypatch.setattr(
        jobs,
        "execute_traced",
        lambda _name, _arguments, function, **_kwargs: function(),
    )
    request = _request("adopt")
    output = scope.workspace / "outputs" / "recovered.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("recovered\n", encoding="utf-8")
    with activate_scope(scope):
        store = ArtifactStore(root=scope.workspace)
        reference = store.register_file(
            output,
            semantic_type="TSOutput",
            producer_action=jobs.ACTION_ID,
            producer_backend=jobs.BACKEND_ID,
            media_type="text/plain",
        )
    result = ActionResult(
        status="partial_success",
        action=jobs.ACTION_ID,
        action_version="1.0.0",
        requested_backend=jobs.BACKEND_ID,
        backend=jobs.BACKEND_ID,
        selection_source="agent",
        result={"recovered": True},
        output_artifacts=[reference],
    ).model_dump(mode="json")
    with activate_scope(scope):
        result["operation_id"] = jobs.scoped_submission_operation_id("adopt")
    result["submission_key"] = "adopt"
    adopted = jobs.controller_adopt_completed_action_result(
        result,
        request,
        scope_id=scope.scope_id,
        workspace=scope.workspace,
    )
    replay = jobs.controller_adopt_completed_action_result(
        result,
        request,
        scope_id=scope.scope_id,
        workspace=scope.workspace,
    )
    replay_without_request = jobs.controller_adopt_completed_action_result(
        result,
        scope_id=scope.scope_id,
        workspace=scope.workspace,
        submission_key="adopt",
        request_fingerprint=jobs._submission_request_fingerprint(request),
    )
    tampered = dict(result)
    tampered["backend"] = "other_backend"
    rejected = jobs.controller_adopt_completed_action_result(
        tampered,
        request,
        scope_id=scope.scope_id,
        workspace=scope.workspace,
    )
    assert adopted["job_id"] == replay["job_id"]
    assert adopted["wake_armed"] is True
    assert replay["idempotent_reused"] is True
    assert replay_without_request["job_id"] == adopted["job_id"]
    assert rejected["error"]["code"] == "transition_state_path_adopt_identity_conflict"


def test_restarting_submit_does_not_start_same_job_again(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """已有持久作业状态在重新进入服务时只复用身份，不重复调用 supervisor。"""

    scope = _scope(tmp_path)
    started = _fake_start_job(monkeypatch, scope)
    request = _request("restart")
    with activate_scope(scope):
        first = jobs.submit_transition_state_path_job(request)
        _set_job_status(scope, first["job_id"], "running")
        second = jobs.submit_transition_state_path_job(request)
    assert first["job_id"] == second["job_id"]
    assert len(started) == 1


@pytest.mark.parametrize(
    ("start_mode", "expected_error_code"),
    [
        ("raises", "transition_state_path_submission_uncertain"),
        ("missing_job_id", "transition_state_path_submission_uncertain"),
        ("conflicting_job_id", "transition_state_path_identity_conflict"),
    ],
)
def test_uncertain_submit_returns_complete_preallocated_identity_and_can_be_woken(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    start_mode: str,
    expected_error_code: str,
) -> None:
    """启动结果不确定时保留完整服务端身份，并把同一记录作为 unknown 唤醒。"""

    scope = _scope(tmp_path)
    request = _request(f"uncertain-{start_mode}")
    start_calls: list[str] = []
    started_job_ids: list[str] = []

    def uncertain_start(**kwargs):
        """模拟 supervisor 启动异常、缺失身份或返回冲突身份。"""

        start_calls.append(start_mode)
        started_job_ids.append(str(kwargs["job_id"]))
        if start_mode == "raises":
            raise OSError("launch outcome unavailable")
        if start_mode == "missing_job_id":
            return {"status": "failed"}
        return {
            "status": "success",
            "job_id": "job_ffffffffffffffffffffffffffffffff",
        }

    monkeypatch.setattr(jobs, "_start_job", uncertain_start)
    monkeypatch.setattr(jobs, "active_catalog_hash", lambda: "catalog-test")
    monkeypatch.setattr(
        jobs,
        "execute_traced",
        lambda _name, _arguments, function, **_kwargs: function(),
    )

    with activate_scope(scope):
        expected_operation_id = jobs.scoped_submission_operation_id(
            str(request.submission_key)
        )
        submitted = jobs.submit_transition_state_path_job(request)
        expected_job_id = started_job_ids[0]
        replayed = jobs.submit_transition_state_path_job(request)
        controller_status = jobs.controller_get_transition_state_path_job(
            expected_job_id,
            scope_id=scope.scope_id,
            workspace=scope.workspace,
        )
        armed = jobs.controller_arm_transition_state_path_job(
            expected_job_id,
            scope_id=scope.scope_id,
            workspace=scope.workspace,
        )
        model_status = jobs.get_transition_state_path_job(
            jobs.TransitionStatePathJobStatusRequest(job_id=expected_job_id)
        )

    expected_identity = {
        "job_id": expected_job_id,
        "operation_id": expected_operation_id,
        "submission_key": request.submission_key,
        "action_id": jobs.ACTION_ID,
        "backend_id": jobs.BACKEND_ID,
        "request_fingerprint": jobs._submission_request_fingerprint(request),
    }
    assert submitted["error"]["code"] == expected_error_code
    for field_name, expected_value in expected_identity.items():
        assert submitted[field_name] == expected_value
        assert replayed[field_name] == expected_value
    assert replayed["error"]["code"] == expected_error_code
    assert start_calls == [start_mode]
    assert controller_status["job_status"] == "unknown"
    assert controller_status["terminal"] is True
    assert armed["job_status"] == "unknown"
    assert armed["wake_armed"] is True
    assert model_status["job_id"] == expected_job_id
    assert model_status["job_status"] == "unknown"
    assert model_status["terminal"] is True


def test_investigation_sync_transition_state_action_requires_async_tool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Investigation profile 的同步入口明确要求异步工具。"""

    scope = _scope(tmp_path)
    monkeypatch.setattr(
        jobs,
        "execute_traced",
        lambda _name, _arguments, function, **_kwargs: function(),
    )
    request = ProgressiveActionRequest(
        action_id=jobs.ACTION_ID,
        backend_id=jobs.BACKEND_ID,
        submission_key="sync-rejected",
        inputs={"reactants": ["CC"], "products": ["CO"]},
    )
    with activate_scope(scope):
        response = jobs.investigation_execute_action(request)
    assert response["error"]["code"] == "async_job_required"
