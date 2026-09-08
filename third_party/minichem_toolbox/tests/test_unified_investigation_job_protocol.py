"""验证 Investigation profile 的统一 submit/get/collect 作业协议。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chemistry_multiagent.investigation.async_jobs import (
    InvestigationJobController,
    build_investigation_job_receipt,
)
from minichem_toolbox import discovery as catalog_discovery
from minichem_toolbox.catalog import action_specs, backend_specs
from minichem_toolbox.models import ActionResult
from minichem_toolbox.scope import ScopeContext, activate_scope
from minichem_mcp_tools import (
    action_jobs,
    investigation_submission_gate,
    open_execution,
    open_tools,
)
from minichem_mcp_tools import transition_state_jobs as ts_jobs
from minichem_mcp_tools.execution_job_registry import ExecutionJobRegistry
from minichem_mcp_tools.execution_models import (
    ExecutionJobListRequest,
    AnalysisJobRequest,
    JobCollectRequest,
    JobStatusRequest,
    NativeJobRequest,
)


STRUCTURE = {
    "atoms": [
        {"element": "C", "position_angstrom": [0.0, 0.0, 0.0]},
        {"element": "C", "position_angstrom": [1.4, 0.0, 0.0]},
        {"element": "N", "position_angstrom": [2.1, 1.1, 0.0]},
        {"element": "H", "position_angstrom": [2.7, 1.2, 0.9]},
    ],
    "bonds": [
        {"atom_index_a": 0, "atom_index_b": 1, "bond_order": 1.0},
        {"atom_index_a": 1, "atom_index_b": 2, "bond_order": 1.0},
        {"atom_index_a": 2, "atom_index_b": 3, "bond_order": 1.0},
    ],
    "charge": 0,
    "multiplicity": 1,
    "pbc": [False, False, False],
}


def _scope(root: Path) -> ScopeContext:
    """构造隔离的 Investigation scope。"""

    root.mkdir(parents=True, exist_ok=True)
    return ScopeContext(
        task_id="task-1",
        batch_id="batch-1",
        combination_id="combination-1",
        scope_id="scope-1",
        workspace=root,
        capability_hash="capability-1",
    )


def _ts_action_request(
    submission_key: str,
    *,
    backend_id: str | None = ts_jobs.BACKEND_ID,
    reactants: list[str] | None = None,
) -> action_jobs.ActionJobRequest:
    """构造统一 submit_action_job 接受的 TS Action 请求。"""

    return action_jobs.ActionJobRequest.model_validate(
        {
            "action_id": ts_jobs.ACTION_ID,
            "submission_key": submission_key,
            "backend_id": backend_id,
            "inputs": {
                "reactants": reactants or ["CC"],
                "products": ["CO"],
            },
        }
    )


def _install_ts_supervisor(
    monkeypatch: pytest.MonkeyPatch,
    scope: ScopeContext,
    *,
    mode: str = "success",
) -> list[str]:
    """用持久控制文件模拟 TS supervisor，不启动真实计算。"""

    started: list[str] = []

    def start_job(**kwargs):
        job_id = str(kwargs["job_id"])
        started.append(job_id)
        if mode == "raises":
            raise RuntimeError("upstream launch failed: scheduler refused request")
        if mode == "non_mapping":
            return ["not-a-json-object"]
        if mode == "missing":
            return {
                "status": "failed",
                "error": {
                    "code": "upstream_missing_job_id",
                    "message": "upstream response omitted job identity",
                },
            }
        if mode == "conflict":
            return {
                "status": "success",
                "job_id": "job_ffffffffffffffffffffffffffffffff",
            }
        directory = scope.workspace / "outputs" / "execution_jobs" / job_id
        directory.mkdir(parents=True, exist_ok=False)
        common = {
            "schema_version": 1,
            "job_id": job_id,
            "job_type": ts_jobs.JOB_TYPE,
            "scope_id": scope.scope_id,
            "scope_workspace": str(scope.workspace),
            "operation_id": kwargs["operation_id"],
            "operation_fingerprint": kwargs["operation_fingerprint"],
            "caller_request_fingerprint": kwargs["caller_request_fingerprint"],
            "submission_key": kwargs["submission_key"],
        }
        ts_jobs._atomic_json(directory / "request.json", common)
        ts_jobs._atomic_json(directory / "status.json", {**common, "status": "queued"})
        return {"status": "success", "job_id": job_id, "job_status": "queued"}

    monkeypatch.setattr(ts_jobs, "_start_job", start_job)
    monkeypatch.setattr(ts_jobs, "active_catalog_hash", lambda: "a" * 64)
    monkeypatch.setattr(
        ts_jobs,
        "execute_traced",
        lambda _name, _arguments, function, **_kwargs: function(),
    )
    monkeypatch.setattr(
        open_tools,
        "execute_traced",
        lambda _name, _arguments, function, **_kwargs: function(),
    )
    return started


def _set_status(scope: ScopeContext, job_id: str, status: str) -> None:
    """修改模拟 supervisor 的作业状态。"""

    path = scope.workspace / "outputs" / "execution_jobs" / job_id / "status.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["status"] = status
    ts_jobs._atomic_json(path, value)


def test_ts_generic_submit_gate_replays_same_key_and_releases_after_collect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TS 统一入口必须使用 TS registry 类型并遵守同键重放和收集释放。"""

    scope = _scope(tmp_path)
    started = _install_ts_supervisor(monkeypatch, scope)
    request = _ts_action_request("ts-gate-one")
    with activate_scope(scope):
        first = investigation_submission_gate.submit_investigation_action_job(request)
        replay = investigation_submission_gate.submit_investigation_action_job(request)
        changed = investigation_submission_gate.submit_investigation_action_job(
            _ts_action_request("ts-gate-one", reactants=["CCC"])
        )
        blocked = investigation_submission_gate.submit_investigation_action_job(
            _ts_action_request("ts-gate-two")
        )
        registry_record = ExecutionJobRegistry(
            workspace=scope.workspace,
            scope_id=scope.scope_id,
        ).get(str(first["job_id"]))
        _set_status(scope, str(first["job_id"]), "failed")
        collected = open_tools.collect_execution_job(
            JobCollectRequest(
                job_id=str(first["job_id"]),
                submission_key="ts-gate-one",
            )
        )
        next_submission = investigation_submission_gate.submit_investigation_action_job(
            _ts_action_request("ts-gate-two")
        )

    assert first["job_id"] == replay["job_id"] == started[0]
    assert replay["idempotent_reused"] is True
    assert registry_record["job_type"] == ts_jobs.JOB_TYPE
    assert changed["error"]["code"] == "submission_key_conflict"
    assert blocked["error"]["code"] == "investigation_job_phase_gate_closed"
    assert collected["job_status"] == "failed"
    assert next_submission["status"] == "success"
    assert next_submission["job_id"] != first["job_id"]
    assert len(started) == 2


def test_generic_ts_backend_error_gets_one_job_id_and_collectable_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """统一 TS submit 的错误 backend 也必须先登记一个可查询失败作业。"""

    scope = _scope(tmp_path)
    started = _install_ts_supervisor(monkeypatch, scope)
    request = _ts_action_request("ts-invalid-backend", backend_id="gaussian")
    with activate_scope(scope):
        submitted = action_jobs.submit_action_job(request)
        status = open_tools.get_execution_job(
            JobStatusRequest(job_id=str(submitted["job_id"]))
        )
        collected = open_tools.collect_execution_job(
            JobCollectRequest(
                job_id=str(submitted["job_id"]),
                submission_key="ts-invalid-backend",
            )
        )

    assert started == []
    assert submitted["status"] == "success"
    assert submitted["job_status"] == "failed"
    assert submitted["error"]["code"] == "unsupported_backend"
    assert status["error"]["code"] == "unsupported_backend"
    assert collected["error"]["code"] == "unsupported_backend"


def test_generic_ts_collect_error_promotes_matching_error_job_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """统一 TS collect 的错误 envelope 顶层必须保留请求 job_id。"""

    job_id = "job_8912052d"
    error_response = {
        "status": "invalid_request",
        "valid": False,
        "error": {
            "code": "transition_state_path_result_unavailable",
            "message": "worker result unavailable",
            "job_id": job_id,
        },
        "automatic_fallback": False,
    }
    monkeypatch.setattr(
        ts_jobs,
        "collect_transition_state_path_job",
        lambda _request: dict(error_response),
    )

    response = open_execution._transition_state_path_facade_collect(job_id)

    assert response["job_id"] == job_id
    assert response["error"]["job_id"] == job_id


def test_generic_collect_retries_transient_action_error_with_new_presentation_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同一作业的展示参数变化不能把可修复的 collect 错误冻结为 submission 冲突。"""

    scope = _scope(tmp_path)

    def start_job(**kwargs):
        job_id = str(kwargs["job_id"])
        directory = scope.workspace / "outputs" / "execution_jobs" / job_id
        directory.mkdir(parents=True, exist_ok=False)
        metadata = dict(kwargs["metadata"])
        metadata.update(
            {
                "caller_request_fingerprint": kwargs["caller_request_fingerprint"],
                "scope_id": scope.scope_id,
                "scope_workspace": str(scope.workspace),
            }
        )
        common = {
            "schema_version": 1,
            "job_id": job_id,
            "job_type": action_jobs.JOB_TYPE,
            "runtime": kwargs["runtime"],
            "metadata": metadata,
            "scope_id": scope.scope_id,
            "scope_workspace": str(scope.workspace),
            "operation_id": kwargs["operation_id"],
            "operation_fingerprint": kwargs["operation_fingerprint"],
            "caller_request_fingerprint": kwargs["caller_request_fingerprint"],
            "submission_key": kwargs["submission_key"],
        }
        action_jobs._atomic_json(directory / "request.json", common)
        action_jobs._atomic_json(
            directory / "status.json",
            {key: value for key, value in common.items() if key != "metadata"}
            | {"status": "queued"},
        )
        return {"status": "success", "job_id": job_id, "job_status": "queued"}

    monkeypatch.setattr(action_jobs, "_start_job", start_job)
    monkeypatch.setattr(action_jobs, "_validate_action_backend_health", lambda _plan: None)
    monkeypatch.setattr(action_jobs, "active_catalog_hash", lambda: "a" * 64)
    monkeypatch.setattr(
        action_jobs,
        "execute_traced",
        lambda _name, _arguments, function, **_kwargs: function(),
    )
    monkeypatch.setattr(
        open_tools,
        "execute_traced",
        lambda _name, _arguments, function, **_kwargs: function(),
    )
    request = action_jobs.ActionJobRequest.model_validate(
        {
            "action_id": "normalize_qcschema_molecule",
            "submission_key": "collect-retry-presentation",
            "inputs": {"structure": STRUCTURE},
            "action_settings": {
                "fix_center_of_mass": True,
                "fix_orientation": True,
            },
        }
    )
    with activate_scope(scope):
        submitted = action_jobs.submit_action_job(request)
        job_id = str(submitted["job_id"])
        _set_status(scope, job_id, "success")
        first = open_tools.collect_execution_job(
            JobCollectRequest(
                job_id=job_id,
                tail_chars=1,
                include_inputs=False,
                max_files=50,
                submission_key=request.submission_key,
            )
        )
        result = ActionResult(
            status="success",
            action="normalize_qcschema_molecule",
            action_version="1.0.0",
            requested_backend="qcelemental",
            backend="qcelemental",
            selection_source="agent",
            result={"normalized": True},
        ).model_dump(mode="json")
        action_jobs._atomic_json(
            scope.workspace
            / "outputs"
            / "execution_jobs"
            / job_id
            / action_jobs.RAW_RESULT_FILENAME,
            result,
        )
        retry = open_tools.collect_execution_job(
            JobCollectRequest(
                job_id=job_id,
                tail_chars=8000,
                include_inputs=True,
                max_files=1000,
                submission_key=request.submission_key,
            )
        )

    assert first["error"]["code"] == "action_job_result_unavailable"
    assert first["job_id"] == job_id
    assert retry["action_result"]["result"]["normalized"] is True
    assert retry["job_id"] == job_id


def test_generic_collect_error_without_identity_gets_requested_job_id(
    tmp_path: Path,
) -> None:
    """generic collect 的错误没有冲突身份时必须保留请求作业身份。"""

    scope = _scope(tmp_path)
    job_id = "job_12345678"
    with activate_scope(scope):
        response = open_tools.collect_execution_job(
            JobCollectRequest(job_id=job_id, submission_key="missing-collect-job")
        )

    assert response["error"]["code"] == "execution_job_not_accessible"
    assert response["job_id"] == job_id


@pytest.mark.parametrize(
    "response",
    [
        {
            "status": "invalid_request",
            "valid": False,
            "error": {"code": "worker_error", "job_id": "job_other"},
        },
        {
            "status": "invalid_request",
            "valid": False,
            "job_id": "job_other",
            "error": {"code": "worker_error"},
        },
        {
            "status": "invalid_request",
            "valid": False,
            "job": {"job_id": "job_other"},
            "error": {"code": "worker_error"},
        },
        {
            "status": "invalid_request",
            "valid": False,
            "jobId": "job_other",
            "error": {"code": "worker_error"},
        },
    ],
)
def test_generic_collect_does_not_promote_conflicting_job_identity(
    response: dict[str, object],
) -> None:
    """已有不同顶层或嵌套 job_id 时不得覆盖错误的作业身份。"""

    job_id = "job_12345678"
    normalized = open_execution._preserve_requested_job_id_in_collection_error(
        response,
        job_id,
    )

    assert normalized == response


@pytest.mark.parametrize(
    ("action_id", "backend_id", "expected_code"),
    [
        ("standardize_structure", "gaussian", "unsupported_backend"),
        ("unknown_catalog_action", "gaussian", "unknown_action"),
    ],
)
def test_catalog_or_backend_selection_error_is_persisted_after_reservation(
    action_id: str,
    backend_id: str,
    expected_code: str,
    tmp_path: Path,
) -> None:
    """已通过请求模型的 Catalog/后端选择错误也必须有 generic job identity。"""

    scope = _scope(tmp_path)
    request = action_jobs.ActionJobRequest.model_validate(
        {
            "action_id": action_id,
            "submission_key": f"selection-error-{expected_code}",
            "backend_id": backend_id,
            "inputs": {},
            "method_spec": {},
            "action_settings": {},
        }
    )
    with activate_scope(scope):
        submitted = action_jobs.submit_action_job(request)
        job_id = str(submitted["job_id"])
        status = open_tools.get_execution_job(JobStatusRequest(job_id=job_id))
        collected = open_tools.collect_execution_job(
            JobCollectRequest(
                job_id=job_id,
                submission_key=str(request.submission_key),
            )
        )

    assert submitted["status"] == "success"
    assert submitted["job_status"] == "failed"
    assert status["error"]["code"] == expected_code
    assert collected["error"]["code"] == expected_code


@pytest.mark.parametrize("mode", ["raises", "non_mapping", "missing", "conflict"])
def test_generic_ts_supervisor_protocol_errors_are_terminal_and_queryable(
    mode: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """统一 TS submit 的 supervisor 异常和畸形响应不得留下永久 active gate。"""

    scope = _scope(tmp_path)
    _install_ts_supervisor(monkeypatch, scope, mode=mode)
    request = _ts_action_request(f"ts-submit-{mode}")
    with activate_scope(scope):
        submitted = action_jobs.submit_action_job(request)
        job_id = str(submitted["job_id"])
        status = open_tools.get_execution_job(JobStatusRequest(job_id=job_id))
        collected = open_tools.collect_execution_job(
            JobCollectRequest(job_id=job_id, submission_key=request.submission_key)
        )

    assert submitted["status"] == "success"
    assert submitted["job_status"] == "failed"
    assert status["job_status"] == "failed"
    assert collected["job_status"] == "failed"
    if mode == "raises":
        assert status["error"]["exception_type"] == "RuntimeError"
        assert "scheduler refused request" in status["error"]["exception_message"]
    elif mode == "non_mapping":
        assert status["error"]["code"] == "transition_state_path_invalid_supervisor_response"
    elif mode == "missing":
        assert status["error"]["code"] == "upstream_missing_job_id"
        assert status["error"]["message"] == "upstream response omitted job identity"
    else:
        assert status["error"]["code"] == "transition_state_path_identity_conflict"
        assert status["error"]["response_job_id"].startswith("job_")


def test_action_backend_auto_deduction_keeps_raw_identity_through_collect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """省略唯一 backend 的 Action 仍能用原始请求指纹完成 generic collect。"""

    scope = _scope(tmp_path)

    def start_job(**kwargs):
        job_id = str(kwargs["job_id"])
        directory = scope.workspace / "outputs" / "execution_jobs" / job_id
        directory.mkdir(parents=True, exist_ok=False)
        metadata = dict(kwargs["metadata"])
        metadata.update(
            {
                "caller_request_fingerprint": kwargs["caller_request_fingerprint"],
                "scope_id": scope.scope_id,
                "scope_workspace": str(scope.workspace),
            }
        )
        common = {
            "schema_version": 1,
            "job_id": job_id,
            "job_type": action_jobs.JOB_TYPE,
            "runtime": kwargs["runtime"],
            "metadata": metadata,
            "scope_id": scope.scope_id,
            "scope_workspace": str(scope.workspace),
            "operation_id": kwargs["operation_id"],
            "operation_fingerprint": kwargs["operation_fingerprint"],
            "caller_request_fingerprint": kwargs["caller_request_fingerprint"],
            "submission_key": kwargs["submission_key"],
        }
        action_jobs._atomic_json(directory / "request.json", common)
        action_jobs._atomic_json(
            directory / "status.json",
            {key: value for key, value in common.items() if key != "metadata"}
            | {"status": "queued"},
        )
        return {"status": "success", "job_id": job_id, "job_status": "queued"}

    monkeypatch.setattr(action_jobs, "_start_job", start_job)
    monkeypatch.setattr(action_jobs, "_validate_action_backend_health", lambda _plan: None)
    monkeypatch.setattr(action_jobs, "active_catalog_hash", lambda: "a" * 64)
    monkeypatch.setattr(
        action_jobs,
        "execute_traced",
        lambda _name, _arguments, function, **_kwargs: function(),
    )
    monkeypatch.setattr(
        open_tools,
        "execute_traced",
        lambda _name, _arguments, function, **_kwargs: function(),
    )
    request = action_jobs.ActionJobRequest.model_validate(
        {
            "action_id": "normalize_qcschema_molecule",
            "submission_key": "action-auto-backend",
            "inputs": {"structure": STRUCTURE},
            "action_settings": {
                "fix_center_of_mass": True,
                "fix_orientation": True,
            },
        }
    )
    with activate_scope(scope):
        submitted = action_jobs.submit_action_job(request)
        job_id = str(submitted["job_id"])
        _set_status(scope, job_id, "success")
        result = ActionResult(
            status="success",
            action="normalize_qcschema_molecule",
            action_version="1.0.0",
            requested_backend="qcelemental",
            backend="qcelemental",
            selection_source="agent",
            result={"normalized": True},
        ).model_dump(mode="json")
        action_jobs._atomic_json(
            scope.workspace
            / "outputs"
            / "execution_jobs"
            / job_id
            / action_jobs.RAW_RESULT_FILENAME,
            result,
        )
        collected = open_tools.collect_execution_job(
            JobCollectRequest(job_id=job_id, submission_key="action-auto-backend")
        )

    assert submitted["backend_id"] == "qcelemental"
    assert collected["action_result"]["result"]["normalized"] is True


def test_investigation_action_projection_scrubs_legacy_routes_and_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """模型可见 Action 投影不得泄露旧路由、资源选择或长短分类。"""

    monkeypatch.setattr(
        action_jobs.discovery_tools,
        "inspect_action",
        lambda _request: {
            "status": "success",
            "execute_with": "execute_action",
            "nested": {
                "resource_limits": {
                    "cpu_cores": 60,
                    "memory_mb": 320000,
                    "gpu_count": 1,
                },
                "routes": [
                    "submit_action_batch",
                    "submit_transition_state_path_job",
                    "get_transition_state_path_job",
                    "collect_transition_state_path_job",
                ],
                "description": "short task then long-running task",
            },
        },
    )
    projected = action_jobs.investigation_inspect_action(
        action_jobs.ActionInspectRequest(
            action_id=ts_jobs.ACTION_ID,
            backend_id=ts_jobs.BACKEND_ID,
            detail_level="summary",
        )
    )
    encoded = json.dumps(projected, ensure_ascii=False)
    assert projected["execute_with"] == "submit_action_job"
    assert projected["investigation_async_execution"]["status_tool"] == "get_execution_job"
    assert projected["investigation_async_execution"]["collect_tool"] == "collect_execution_job"
    assert "execute_action" not in encoded
    assert "submit_action_batch" not in encoded
    assert "submit_transition_state_path_job" not in encoded
    assert "get_transition_state_path_job" not in encoded
    assert "collect_transition_state_path_job" not in encoded
    assert "resource_limits" not in encoded
    assert "cpu_cores" not in encoded
    assert "memory_mb" not in encoded
    assert "gpu_count" not in encoded
    assert "long-running" not in encoded
    assert "short task" not in encoded
    assert "execute_action_request_template" not in projected["nested"]


def test_real_catalog_action_projection_scrubs_all_action_backend_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """真实 65 个 Action/84 个 backend pair 的模型投影统一为 generic 合同。"""

    fixed_snapshot = {
        "schema_version": 7,
        "catalog_hash": "fixed-test-snapshot",
        "backends": [
            {"id": backend_id, "health": {"status": "available"}}
            for backend_id in backend_specs()
        ],
        "resources": [],
    }

    def inspect_from_fixed_snapshot(request):
        return catalog_discovery.inspect_action(
            request.action_id,
            backend_id=request.backend_id,
            detail_level=request.detail_level,
            snapshot=fixed_snapshot,
        )

    monkeypatch.setattr(
        action_jobs.discovery_tools,
        "inspect_action",
        inspect_from_fixed_snapshot,
    )
    forbidden_terms = (
        "execute_action",
        "submit_action_batch",
        "submit_transition_state_path_job",
        "get_transition_state_path_job",
        "collect_transition_state_path_job",
        "resource_limits",
        "memory_mb",
        "cpu_cores",
        "gpu_count",
        "long-running",
        "long task",
        "short task",
        "长任务",
        "短任务",
    )
    pair_count = 0
    for action_id, specification in action_specs().items():
        for backend_id in specification.backend_ids:
            pair_count += 1
            projected = action_jobs.investigation_inspect_action(
                action_jobs.ActionInspectRequest(
                    action_id=action_id,
                    backend_id=backend_id,
                    detail_level="full",
                )
            )
            assert "selected_request_contract" in projected
            assert "execute_action_request_template" not in projected[
                "selected_request_contract"
            ]
            encoded = json.dumps(projected, ensure_ascii=False, sort_keys=True)
            assert not any(term in encoded for term in forbidden_terms), (
                action_id,
                backend_id,
            )
            assert projected["execute_with"] == "submit_action_job"
            assert projected["investigation_async_execution"]["status_tool"] == (
                "get_execution_job"
            )
            assert projected["investigation_async_execution"]["collect_tool"] == (
                "collect_execution_job"
            )
    assert len(action_specs()) == 65
    assert pair_count == 84


def test_programmable_analysis_receipt_recovers_family_from_request_context() -> None:
    """旧分析 supervisor 回执缺少重复字段时仍能进入 generic recovery。"""

    receipt = build_investigation_job_receipt(
        {
            "type": "mcpToolCall",
            "status": "completed",
            "tool": "submit_analysis_program",
            "arguments": {"request": {"runtime": "python"}},
            "result": {
                "structuredContent": {
                    "status": "success",
                    "job_id": "job_12345678",
                    "operation_id": "scope-operation-1",
                    "submission_key": "analysis-one",
                }
            },
        }
    )
    assert receipt is not None
    assert receipt.identity.job_family == "programmable_analysis"
    assert receipt.identity.job_type == "programmable_analysis"
    assert receipt.identity.runtime == "python"


def test_harness_captures_failed_receipt_collects_and_accepts_next_submit() -> None:
    """计算失败是可收集的作业结果，不应让 Harness turn 失败或锁死。"""

    failed_item = {
        "type": "mcpToolCall",
        "status": "completed",
        "tool": "submit_native_job",
        "arguments": {
            "request": {
                "software_id": "gaussian",
                "runtime": "gaussian",
                "submission_key": "native-failed",
            }
        },
        "result": {
            "structuredContent": {
                "status": "success",
                "job_id": "job_0000fail",
                "job_type": "native_software",
                "runtime": "gaussian",
                "operation_id": "scope-native-failed",
                "submission_key": "native-failed",
                "job_status": "failed",
                "terminal": True,
                "error": {
                    "code": "gaussian_input_error",
                    "message": "QPErr: invalid route section",
                    "exception_type": "GaussianInputError",
                    "exception_message": "route section could not be parsed",
                },
            }
        },
    }
    next_item = {
        "type": "mcpToolCall",
        "status": "completed",
        "tool": "submit_analysis_program",
        "arguments": {"request": {"runtime": "python", "submission_key": "analysis-next"}},
        "result": {
            "structuredContent": {
                "status": "success",
                "job_id": "job_0000next",
                "job_type": "programmable_analysis",
                "runtime": "python",
                "operation_id": "scope-analysis-next",
                "submission_key": "analysis-next",
                "job_status": "queued",
                "terminal": False,
            }
        },
    }

    failed_receipt = build_investigation_job_receipt(failed_item)
    next_receipt = build_investigation_job_receipt(next_item)
    assert failed_receipt is not None
    assert next_receipt is not None
    assert failed_receipt.result["job_status"] == "failed"
    assert failed_receipt.result["error"]["code"] == "gaussian_input_error"

    controller = InvestigationJobController(object(), require_status_identity=True)
    failed_state = controller.capture_submit_receipt(
        failed_receipt,
        task_id="task-1",
        batch_id="batch-1",
        combination_id="combination-1",
        codex_thread_id="thread-1",
        turn_id="turn-1",
        scope_id="scope-1",
        verify_status=False,
    )
    # collect_execution_job 的成功回执调用这个 callback；失败作业同样必须释放门禁。
    controller.mark_job_collected(failed_state)
    next_state = controller.capture_submit_receipt(
        next_receipt,
        task_id="task-1",
        batch_id="batch-1",
        combination_id="combination-1",
        codex_thread_id="thread-1",
        turn_id="turn-1",
        scope_id="scope-1",
        verify_status=False,
    )

    assert failed_state.identity.job_id == "job_0000fail"
    assert next_state.identity.job_id == "job_0000next"
    assert next_state.identity.job_family == "programmable_analysis"


def test_scope_job_list_contains_action_ts_native_analysis_without_cross_scope_records(
    tmp_path: Path,
) -> None:
    """scope 清单同时覆盖四类作业，并且不读取另一个 scope 的记录。"""

    scope = _scope(tmp_path / "scope-one")
    other_scope = ScopeContext(
        task_id="task-2",
        batch_id="batch-2",
        combination_id="combination-2",
        scope_id="scope-two",
        workspace=tmp_path / "scope-two",
        capability_hash="capability-2",
    )

    def register(scope_context: ScopeContext, job_id: str, job_type: str, status: str) -> None:
        registry = ExecutionJobRegistry(scope_context.workspace, scope_context.scope_id)
        registry.reserve(
            job_type=job_type,
            operation_id=f"operation-{job_id}",
            operation_fingerprint="a" * 64,
            submission_key=f"submission-{job_id}",
            preferred_job_id=job_id,
        )
        directory = scope_context.workspace / "outputs" / "execution_jobs" / job_id
        directory.mkdir(parents=True, exist_ok=True)
        ts_jobs._atomic_json(
            directory / "status.json",
            {
                "schema_version": 1,
                "job_id": job_id,
                "job_type": job_type,
                "status": status,
                "scope_id": scope_context.scope_id,
                "scope_workspace": str(scope_context.workspace),
                "operation_id": f"operation-{job_id}",
                "operation_fingerprint": "a" * 64,
                "submission_key": f"submission-{job_id}",
            },
        )
        registry.mark(job_id, phase=status)

    for index, job_type in enumerate(
        (
            "action_execution",
            "transition_state_path",
            "native_software",
            "programmable_analysis",
        )
    ):
        register(scope, f"job_{index + 1:08d}", job_type, "running" if index < 2 else "success")
    register(other_scope, "job_ffffffff", "action_execution", "running")

    with activate_scope(scope):
        result = open_tools.list_execution_jobs(ExecutionJobListRequest())

    assert result["status"] == "success"
    assert {item["job_id"] for item in result["jobs"]} == {
        "job_00000001",
        "job_00000002",
        "job_00000003",
        "job_00000004",
    }
    assert result["active_job_ids"] == ["job_00000001", "job_00000002"]
    assert "job_ffffffff" not in json.dumps(result)


def _native_submission_request(submission_key: str) -> NativeJobRequest:
    """构造可由服务端验证器继续检查的 native 请求。"""

    return NativeJobRequest.model_validate(
        {
            "software_id": "xtb",
            "executable": "xtb",
            "submission_key": submission_key,
        }
    )


def _analysis_submission_request(submission_key: str) -> AnalysisJobRequest:
    """构造可由服务端验证器继续检查的 Python 分析请求。"""

    return AnalysisJobRequest.model_validate(
        {
            "runtime": "core",
            "script_path": "code/program.py",
            "submission_key": submission_key,
        }
    )


def test_native_and_analysis_validation_failures_are_receipts_and_release_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """native/analysis 验证失败先登记 job，再经 generic get/collect 释放门禁。"""

    scope = _scope(tmp_path)
    monkeypatch.setattr(
        open_tools,
        "execute_traced",
        lambda _name, _arguments, function, **_kwargs: function(),
    )
    monkeypatch.setattr(
        open_tools,
        "_submit_native_job",
        open_tools._submit_native_job,
    )
    monkeypatch.setattr(
        open_tools,
        "_submit_analysis_program",
        open_tools._submit_analysis_program,
    )
    monkeypatch.setattr(
        open_execution,
        "validate_native_job",
        lambda _request: {
            "status": "invalid_request",
            "valid": False,
            "error": {
                "code": "native_input_contract_failed",
                "message": "native input deck is incomplete",
                "candidate_fixes": ["add the required staged input"],
            },
        },
    )
    monkeypatch.setattr(
        open_execution,
        "validate_analysis_program",
        lambda _request: {
            "status": "invalid_request",
            "valid": False,
            "error": {
                "stage": "preflight",
                "code": "python_syntax_error",
                "message": "invalid syntax",
                "line": 3,
                "candidate_fixes": ["fix the reported syntax"],
            },
        },
    )
    native_request = _native_submission_request("native-validation-failure")
    analysis_request = _analysis_submission_request("analysis-validation-failure")

    with activate_scope(scope):
        native = investigation_submission_gate.submit_investigation_native_job(
            native_request
        )
        native_status = open_tools.get_execution_job(
            JobStatusRequest(job_id=str(native["job_id"]))
        )
        native_list = open_tools.list_execution_jobs(ExecutionJobListRequest())
        native_collect = open_tools.collect_execution_job(
            JobCollectRequest(
                job_id=str(native["job_id"]),
                submission_key=native_request.submission_key,
            )
        )
        analysis = investigation_submission_gate.submit_investigation_analysis_program(
            analysis_request
        )

    assert native["status"] == "success"
    assert native["job_status"] == "failed"
    assert native["terminal"] is True
    assert native_status["status"] == "success"
    assert native_status["job_status"] == "failed"
    assert native_status["error"]["code"] == "native_input_contract_failed"
    assert native_list["jobs"][0]["job_status"] == "failed"
    assert native_list["jobs"][0]["error"]["code"] == "native_input_contract_failed"
    assert native_collect["status"] == "success"
    assert native_collect["ready"] is True
    assert native_collect["job_status"] == "failed"
    assert native_collect["error"]["code"] == "native_input_contract_failed"
    assert analysis["status"] == "success"
    assert analysis["job_status"] == "failed"
    assert analysis["error"]["code"] == "python_syntax_error"


def test_native_and_analysis_start_failures_are_failed_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """supervisor 启动异常不升级为 Harness 错误，generic 查询仍可收集原始错误。"""

    scope = _scope(tmp_path)
    (tmp_path / "code").mkdir(parents=True, exist_ok=True)
    (tmp_path / "code" / "program.py").write_text(
        "from researchchem_job import JobContext\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        open_tools,
        "execute_traced",
        lambda _name, _arguments, function, **_kwargs: function(),
    )
    monkeypatch.setattr(
        open_execution,
        "validate_native_job",
        lambda request: {
            "status": "success",
            "valid": True,
            "software_id": request.software_id,
            "runtime": "core",
            "resolved_executable": "/bin/true",
            "command": ["/bin/true", *request.arguments],
            "staged_targets": [],
            "stdin_target": None,
            "input_deck_validation": None,
            "calculation_intent": "other",
            "resource_limits": {"cpu_cores": 1, "memory_mb": 4096, "gpu_count": 0},
            "invocation_guide": {"synopsis": "test"},
        },
    )
    monkeypatch.setattr(
        open_execution,
        "validate_analysis_program",
        lambda request: {
            "status": "success",
            "valid": True,
            "runtime": request.runtime,
            "runtime_python": "/bin/true",
            "script_source": request.script_path,
            "script_target": request.script_target,
            "script_sha256": "a" * 64,
            "discovered_imports": [],
            "required_modules": [],
            "required_module_versions": {},
            "required_symbols": {},
            "module_status": {},
            "module_details": {},
            "input_inspection": {"status": "success", "inputs": []},
            "external_execution_findings": [],
            "job_context_compliance": {"status": "valid"},
            "analysis_contract": {"schema_version": 1, "inputs": [], "outputs": []},
            "resource_limits": {"cpu_cores": 1, "memory_mb": 4096, "gpu_count": 0},
        },
    )
    monkeypatch.setattr(
        open_execution,
        "_start_job",
        lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("supervisor launch failed: scheduler unavailable")
        ),
    )

    native_request = _native_submission_request("native-start-failure")
    analysis_request = _analysis_submission_request("analysis-start-failure")
    with activate_scope(scope):
        native = investigation_submission_gate.submit_investigation_native_job(
            native_request
        )
        native_status = open_tools.get_execution_job(
            JobStatusRequest(job_id=str(native["job_id"]))
        )
        native_collect = open_tools.collect_execution_job(
            JobCollectRequest(
                job_id=str(native["job_id"]),
                submission_key=native_request.submission_key,
            )
        )
        analysis = investigation_submission_gate.submit_investigation_analysis_program(
            analysis_request
        )

    for response in (native, analysis):
        assert response["status"] == "success"
        assert response["job_status"] == "failed"
        assert response["terminal"] is True
        assert len(response["job_id"]) == 12
    assert native_status["status"] == "success"
    assert native_status["job_status"] == "failed"
    assert native_status["error"]["exception_type"] == "RuntimeError"
    assert "scheduler unavailable" in native_status["error"]["exception_message"]
    assert native_collect["status"] == "success"
    assert native_collect["ready"] is True
    assert native_collect["error"]["exception_type"] == "RuntimeError"


@pytest.mark.parametrize("job_family", ["native", "analysis"])
def test_native_and_analysis_preparation_failures_are_collectable(
    job_family: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """输入快照/staging 准备失败也必须落为 terminal failed 作业。"""

    scope = _scope(tmp_path)
    monkeypatch.setattr(
        open_tools,
        "execute_traced",
        lambda _name, _arguments, function, **_kwargs: function(),
    )
    if job_family == "native":
        request = _native_submission_request("native-preparation-failure")
        monkeypatch.setattr(
            open_execution,
            "validate_native_job",
            lambda _request: {"status": "success", "valid": True},
        )
        monkeypatch.setattr(
            open_execution,
            "_prepare_scoped_native_job",
            lambda _request, _validation: (_ for _ in ()).throw(
                OSError("input snapshot failed: staged source disappeared")
            ),
        )
        submitter = investigation_submission_gate.submit_investigation_native_job
    else:
        request = _analysis_submission_request("analysis-preparation-failure")
        monkeypatch.setattr(
            open_execution,
            "validate_analysis_program",
            lambda _request: {"status": "success", "valid": True},
        )
        monkeypatch.setattr(
            open_execution,
            "_prepare_scoped_analysis_job",
            lambda _request, _validation: (_ for _ in ()).throw(
                OSError("staging failed: source file disappeared")
            ),
        )
        submitter = investigation_submission_gate.submit_investigation_analysis_program

    with activate_scope(scope):
        submitted = submitter(request)
        job_id = str(submitted["job_id"])
        status = open_tools.get_execution_job(JobStatusRequest(job_id=job_id))
        collected = open_tools.collect_execution_job(
            JobCollectRequest(job_id=job_id, submission_key=request.submission_key)
        )

    assert submitted["status"] == "success"
    assert submitted["job_status"] == "failed"
    assert status["status"] == "success"
    assert status["job_status"] == "failed"
    assert status["error"]["code"] == "execution_preparation_failed"
    assert status["error"]["exception_type"] == "OSError"
    assert "disappeared" in status["error"]["exception_message"]
    assert collected["status"] == "success"
    assert collected["ready"] is True
    assert collected["job_status"] == "failed"
