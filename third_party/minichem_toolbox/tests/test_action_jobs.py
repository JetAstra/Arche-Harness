"""统一 Action Job Core 的提交、恢复和收集合同测试。"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from minichem_toolbox.artifacts import ArtifactStore
from minichem_toolbox.catalog import action_specs, backend_specs
from minichem_toolbox.models import ActionResult
from minichem_toolbox.scope import ScopeContext, activate_scope
from minichem_mcp_tools import action_jobs, investigation_submission_gate, open_tools
from minichem_mcp_tools.execution_models import (
    JobCollectRequest,
    JobStatusRequest,
    NativeJobRequest,
)
from chemistry_multiagent.investigation.async_jobs import (
    InvestigationJobSubmissionError,
    build_investigation_job_receipt,
)
from chemistry_multiagent.investigation.prompts import build_investigation_job_continuation_request


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
    """构造隔离的 Investigation capability scope。"""

    root.mkdir(parents=True, exist_ok=True)
    return ScopeContext(
        task_id="task-1",
        batch_id="batch-1",
        combination_id="combination-1",
        scope_id="scope-1",
        workspace=root,
        capability_hash="capability-1",
    )


def _scan_request(submission_key: str) -> action_jobs.ActionJobRequest:
    """构造 Case3 Gaussian scan 的最小结构化请求。"""

    return action_jobs.ActionJobRequest.model_validate(
        {
            "action_id": "scan_reaction_coordinates",
            "submission_key": submission_key,
            "backend_id": "gaussian",
            "inputs": {"structure": STRUCTURE},
            "method_spec": {
                "method": "wB97XD",
                "basis": "def2SVP",
                "solvation_model": "smd",
                "solvent": "N,N-Dimethylacetamide",
            },
            "action_settings": {
                "scf_convergence": "default",
                "coordinate_type": "dihedral",
                "atom_indices": [0, 1, 2, 3],
                "start_value": 0.0,
                "end_value": 60.0,
                "value_unit": "degree",
                "steps": 12,
                "optimization_convergence": "default",
                "max_cycles": 200,
                "force_constant_strategy": "default",
                "disable_symmetry": True,
            },
        }
    )


def _standardize_request(
    submission_key: str,
    *,
    structure: object = STRUCTURE,
) -> action_jobs.ActionJobRequest:
    """构造可由真实 RDKit worker 快速完成的 Action 请求。"""

    return action_jobs.ActionJobRequest.model_validate(
        {
            "action_id": "standardize_structure",
            "submission_key": submission_key,
            "backend_id": "rdkit",
            "inputs": {"structure": structure},
            "method_spec": {},
            "action_settings": {
                "largest_fragment": False,
                "neutralize": False,
                "canonical_tautomer": False,
            },
        }
    )


def _fake_start_job(
    monkeypatch: pytest.MonkeyPatch,
    scope: ScopeContext,
    captured: dict[str, object] | None = None,
) -> list[str]:
    """模拟统一 supervisor，只写入真实控制文件而不启动 Gaussian。"""

    started: list[str] = []

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
        jobs_record = dict(common)
        status = dict(common)
        status.pop("metadata", None)
        status["status"] = "queued"
        action_jobs._atomic_json(directory / "request.json", jobs_record)
        action_jobs._atomic_json(directory / "status.json", status)
        if captured is not None:
            captured["resource_limits"] = dict(kwargs["resource_limits"])
            captured["worker_action_job_request"] = dict(
                kwargs["metadata"]["worker_action_job_request"]
            )
        started.append(job_id)
        return {"status": "success", "job_id": job_id, "job_status": "queued"}

    monkeypatch.setattr(action_jobs, "_start_job", start_job)
    monkeypatch.setattr(action_jobs, "_validate_action_backend_health", lambda _plan: None)
    monkeypatch.setattr(action_jobs, "active_catalog_hash", lambda: "a" * 64)
    monkeypatch.setattr(action_jobs, "execute_traced", lambda _n, _a, fn, **_k: fn())
    return started


def _set_status(scope: ScopeContext, job_id: str, status: str) -> None:
    """修改模拟 supervisor 的状态文件。"""

    path = scope.workspace / "outputs" / "execution_jobs" / job_id / "status.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["status"] = status
    action_jobs._atomic_json(path, value)


def test_action_job_core_submits_scan_immediately_and_gates_second_active_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """scan 与内部 fast Action 必须共享 job_id、幂等键和 scope 单 active gate。"""

    scope = _scope(tmp_path)
    started = _fake_start_job(monkeypatch, scope)
    with activate_scope(scope):
        first = action_jobs.submit_action_job(_scan_request("scan-one"))
        replay = action_jobs.submit_action_job(_scan_request("scan-one"))
        blocked = action_jobs.submit_action_job(_scan_request("scan-two"))

    assert first["status"] == "success"
    assert first["job_id"] == replay["job_id"] == started[0]
    assert replay["idempotent_reused"] is True
    assert blocked["error"]["code"] == "action_job_phase_gate_closed"
    assert len(started) == 1
    assert ("standardize_structure", "rdkit") in action_jobs.ACTION_JOB_ADAPTERS


def test_generic_action_worker_receives_policy_walltime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """通用 Action worker 请求必须携带服务策略墙钟，而非使用后端默认值。"""

    scope = _scope(tmp_path)
    captured: dict[str, object] = {}
    _fake_start_job(monkeypatch, scope, captured)

    with activate_scope(scope):
        submitted = action_jobs.submit_action_job(_scan_request("scan-policy-timeout"))

    assert submitted["status"] == "success"
    assert captured["resource_limits"] == {
        **action_jobs.fixed_job_resource_limits(),
        "walltime_seconds": action_jobs.timeout_seconds_for("compute"),
    }
    worker_request = captured["worker_action_job_request"]
    assert isinstance(worker_request, dict)
    assert worker_request["resource_limits"] == {
        **action_jobs.fixed_job_resource_limits(),
        "walltime_seconds": action_jobs.timeout_seconds_for("compute"),
    }


def test_generic_action_fixes_backend_cpu_cap_without_model_resource_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """固定评测资源必须服从 pysisyphus 后端上限，且不读取模型资源字段。"""

    scope = _scope(tmp_path)
    captured: dict[str, object] = {}
    _fake_start_job(monkeypatch, scope, captured)
    request = action_jobs.ActionJobRequest.model_validate(
        {
            "action_id": "scan_reaction_coordinates",
            "submission_key": "scan-pysisyphus-cap",
            "backend_id": "pysisyphus",
            "inputs": {"structure": STRUCTURE},
            "method_spec": {"calculator_backend": "xtb", "method": "GFN2-xTB"},
            "action_settings": {
                "coordinate_type": "dihedral",
                "atom_indices": [0, 1, 2, 3],
                "start_value": 0.0,
                "end_value": 60.0,
                "value_unit": "degree",
                "steps": 12,
                "optimizer": "rfo",
                "convergence": "gau",
                "max_cycles": 200,
                "hessian_init": "xtb",
            },
            "resource_limits": {"cpu_cores": 1, "memory_mb": 128, "gpu_count": 0},
        }
    )

    with activate_scope(scope):
        submitted = action_jobs.submit_action_job(request)

    assert submitted["status"] == "success"
    assert captured["resource_limits"]["cpu_cores"] == 48
    worker_request = captured["worker_action_job_request"]
    assert isinstance(worker_request, dict)
    assert worker_request["resource_limits"]["cpu_cores"] == 48


def test_action_job_adapter_registry_exactly_covers_catalog_except_ts_pipeline() -> None:
    """通用 adapter 必须覆盖每个 Catalog Action/backend，且只排除 TS 专用链。"""

    backends = backend_specs()
    expected = {
        (action_id, backend_id)
        for action_id, action in action_specs().items()
        if action_id != action_jobs.TS_ACTION_ID
        for backend_id in action.backend_ids
        if backend_id in backends
    }

    assert set(action_jobs.ACTION_JOB_ADAPTERS) == expected


def test_action_job_repair_template_does_not_expose_resource_selection(
    tmp_path: Path,
) -> None:
    """参数错误先登记 job_id，原始修复模板可由 generic get/collect 读取。"""

    scope = _scope(tmp_path)
    request = action_jobs.ActionJobRequest.model_validate(
        {
            "action_id": "standardize_structure",
            "submission_key": "standardize-missing-structure",
            "backend_id": "rdkit",
            "inputs": {},
            "method_spec": {},
            "action_settings": {},
        }
    )

    with activate_scope(scope):
        response = action_jobs.submit_action_job(request)

    assert response["status"] == "success"
    assert response["job_status"] == "failed"
    assert response["terminal"] is True
    assert response["error"]["missing_fields"] == ["inputs.structure"]
    template = response["error"]["corrected_request_template"]
    assert "resource_limits" not in template
    assert "cpu_cores" not in json.dumps(response)
    assert "memory_mb" not in json.dumps(response)
    assert "gpu_count" not in json.dumps(response)

    with activate_scope(scope):
        status = open_tools.get_execution_job(
            JobStatusRequest(job_id=str(response["job_id"]))
        )
        collected = open_tools.collect_execution_job(
            JobCollectRequest(
                job_id=str(response["job_id"]),
                submission_key="standardize-missing-structure",
            )
        )
    assert status["error"]["missing_fields"] == ["inputs.structure"]
    assert collected["error"]["corrected_request_template"] == template


def test_action_job_core_uses_generic_get_collect_facade(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """generic get/collect 必须返回统一状态并交接 canonical ActionResult。"""

    scope = _scope(tmp_path)
    _fake_start_job(monkeypatch, scope)
    with activate_scope(scope):
        submitted = action_jobs.submit_action_job(_scan_request("scan-collect"))
        job_id = str(submitted["job_id"])
        _set_status(scope, job_id, "success")
        raw = ActionResult(
            status="success",
            action="scan_reaction_coordinates",
            action_version="1.0.0",
            requested_backend="gaussian",
            backend="gaussian",
            selection_source="agent",
            result={"completed_points": 13},
        ).model_dump(mode="json")
        action_jobs._atomic_json(
            scope.workspace
            / "outputs"
            / "execution_jobs"
            / job_id
            / action_jobs.RAW_RESULT_FILENAME,
            raw,
        )
        status = open_tools.get_execution_job(JobStatusRequest(job_id=job_id))
        collected = open_tools.collect_execution_job(
            JobCollectRequest(job_id=job_id, submission_key="scan-collect")
        )

    assert status["job_status"] == "success"
    assert collected["action_result"]["result"]["completed_points"] == 13


def test_generic_collect_compacts_large_nested_action_result_and_keeps_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """generic collect 只压缩嵌套大结果，并保留顶层作业身份。"""

    scope = _scope(tmp_path)
    _fake_start_job(monkeypatch, scope)
    hessian = [
        [float(row * 159 + column) for column in range(159)]
        for row in range(159)
    ]
    with activate_scope(scope):
        submitted = action_jobs.submit_action_job(_scan_request("scan-large-collect"))
        job_id = str(submitted["job_id"])
        _set_status(scope, job_id, "success")
        raw = ActionResult(
            status="success",
            action="scan_reaction_coordinates",
            action_version="1.0.0",
            requested_backend="gaussian",
            backend="gaussian",
            selection_source="agent",
            result={"hessian": hessian, "completed_points": 13},
        ).model_dump(mode="json")
        action_jobs._atomic_json(
            scope.workspace
            / "outputs"
            / "execution_jobs"
            / job_id
            / action_jobs.RAW_RESULT_FILENAME,
            raw,
        )
        collected = open_tools.collect_execution_job(
            JobCollectRequest(
                job_id=job_id,
                submission_key="scan-large-collect",
            )
        )

    assert collected["status"] == "success"
    assert collected["job_id"] == job_id
    assert collected["job_status"] == "success"
    assert collected["operation_id"] == submitted["operation_id"]
    assert collected["submission_key"] == "scan-large-collect"
    assert collected["action_id"] == "scan_reaction_coordinates"
    assert collected["backend_id"] == "gaussian"
    compact_hessian = collected["action_result"]["result"]["hessian"]
    assert compact_hessian["inline"] is False
    assert compact_hessian["content"] == "hessian"
    assert compact_hessian["shape"] == [159, 159]
    assert collected["action_result"]["result"]["completed_points"] == 13
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


def test_generic_collection_response_points_large_result_to_complete_artifact(
    tmp_path: Path,
) -> None:
    """通用收集响应在保留顶层身份的同时给大结果返回完整 ArtifactRef 指针。"""

    artifact = {
        "artifact_id": "art_" + "a" * 32,
        "semantic_type": "ActionResult",
        "media_type": "application/json",
        "sha256": "b" * 64,
        "path": "_tool_artifacts/objects/action-result.json",
        "producer_action": "scan_reaction_coordinates",
        "producer_backend": "gaussian",
        "parent_artifact_ids": [],
    }
    hessian = [
        [float(row * 159 + column) for column in range(159)]
        for row in range(159)
    ]
    action_result = ActionResult(
        status="success",
        action="scan_reaction_coordinates",
        action_version="1.0.0",
        requested_backend="gaussian",
        backend="gaussian",
        selection_source="agent",
        result={"hessian": hessian, "completed_points": 13},
        output_artifacts=[artifact],
    ).model_dump(mode="json")
    response = action_jobs._collection_response(
        {
            "job_id": "job_" + "1" * 32,
            "operation_id": "scope-submission-" + "2" * 64,
            "submission_key": "scan-artifact-pointer",
            "action_id": "scan_reaction_coordinates",
            "backend_id": "gaussian",
        },
        job_status="success",
        action_result=action_result,
    )

    pointer = response["action_result"]["result"]["hessian"]
    assert pointer["inline"] is False
    assert pointer["artifact_id"] == artifact["artifact_id"]
    assert pointer["semantic_type"] == "ActionResult"
    assert response["job_id"] == "job_" + "1" * 32
    assert response["operation_id"] == "scope-submission-" + "2" * 64
    assert response["submission_key"] == "scan-artifact-pointer"
    assert response["status"] == "success"
    assert action_result["result"]["hessian"] == hessian


def test_generic_collect_keeps_canonical_action_result_complete_and_small_results_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """模型响应压缩不改写 canonical 文件，小结果仍保持原始结构。"""

    scope = _scope(tmp_path)
    _fake_start_job(monkeypatch, scope)
    hessian = [
        [float(row * 159 + column) for column in range(159)]
        for row in range(159)
    ]
    with activate_scope(scope):
        submitted = action_jobs.submit_action_job(_scan_request("scan-canonical-large"))
        job_id = str(submitted["job_id"])
        _set_status(scope, job_id, "success")
        raw = ActionResult(
            status="success",
            action="scan_reaction_coordinates",
            action_version="1.0.0",
            requested_backend="gaussian",
            backend="gaussian",
            selection_source="agent",
            result={"hessian": hessian, "completed_points": 13},
        ).model_dump(mode="json")
        result_path = (
            scope.workspace
            / "outputs"
            / "execution_jobs"
            / job_id
            / action_jobs.RAW_RESULT_FILENAME
        )
        action_jobs._atomic_json(result_path, raw)
        collected = action_jobs.collect_action_job(
            job_id,
            submission_key="scan-canonical-large",
        )
        canonical_path = (
            scope.workspace
            / "outputs"
            / "execution_jobs"
            / job_id
            / action_jobs.ACTION_JOB_RESULT_FILENAME
        )
        canonical = json.loads(canonical_path.read_text(encoding="utf-8"))

    assert collected["action_result"]["result"]["hessian"]["inline"] is False
    assert len(canonical["result"]["hessian"]) == 159
    assert len(canonical["result"]["hessian"][0]) == 159
    assert canonical["result"]["hessian"][0][0] == 0.0
    assert canonical["result"]["hessian"][-1][-1] == float(158 * 159 + 158)
    assert collected["action_result"]["result"]["completed_points"] == 13


def test_investigation_inspection_projects_generic_submit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Investigation inspect 投影必须把 generic submit 显示为唯一主入口。"""

    monkeypatch.setattr(
        action_jobs.discovery_tools,
        "inspect_action",
        lambda _request: {"status": "success", "execute_with": "execute_action"},
    )
    request = action_jobs.ActionInspectRequest(
        action_id="scan_reaction_coordinates",
        backend_id="gaussian",
        detail_level="summary",
    )
    projected = action_jobs.investigation_inspect_action(request)
    assert projected["execute_with"] == "submit_action_job"
    assert projected["parallel_execution"]["eligible"] is False
    assert "submit_action_batch" not in json.dumps(projected)
    contract = projected["investigation_async_execution"]["contract"]
    assert "terminal=true/failed" in contract
    assert contract.index("get_execution_job") < contract.index("collect_execution_job")
    assert "terminal=true/failed" in action_jobs.TOOL_DESCRIPTIONS["submit_action_job"]


def test_investigation_inspection_resolves_internal_deterministic_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """省略 backend 的单一 deterministic Action 仍必须展示可执行异步合同。"""

    monkeypatch.setattr(
        action_jobs.discovery_tools,
        "inspect_action",
        lambda _request: {"status": "success", "execute_with": "execute_action"},
    )
    projected = action_jobs.investigation_inspect_action(
        action_jobs.ActionInspectRequest(
            action_id="normalize_qcschema_molecule",
            detail_level="summary",
        )
    )

    assert projected["execute_with"] == "submit_action_job"
    async_contract = projected["investigation_async_execution"]
    assert async_contract["eligible"] is True
    assert async_contract["resolved_backend_id"] == "qcelemental"
    assert async_contract["selection_required"] is False


def test_internal_deterministic_action_submit_infers_single_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """内部确定性 Action 不应要求模型重复提供唯一 backend。"""

    scope = _scope(tmp_path)
    _fake_start_job(monkeypatch, scope)
    request = action_jobs.ActionJobRequest.model_validate(
        {
            "action_id": "normalize_qcschema_molecule",
            "submission_key": "deterministic-normalize",
            "inputs": {"structure": STRUCTURE},
            "method_spec": {},
            "action_settings": {
                "fix_center_of_mass": True,
                "fix_orientation": True,
            },
        }
    )
    with activate_scope(scope):
        submitted = action_jobs.submit_action_job(request)

    assert submitted["status"] == "success"
    assert submitted["backend_id"] == "qcelemental"


def test_file_artifact_is_staged_with_manifest_digest(
    tmp_path: Path,
) -> None:
    """非 JSON ArtifactRef 必须复制公开文件并核对 manifest 摘要。"""

    scope = _scope(tmp_path)
    inputs = scope.workspace / "inputs"
    inputs.mkdir()
    xyz_path = inputs / "structure.xyz"
    xyz_path.write_text("1\nfixture\nH 0.0 0.0 0.0\n", encoding="utf-8")
    with activate_scope(scope):
        reference = ArtifactStore(root=scope.workspace).register_file(
            xyz_path,
            semantic_type="XYZFile",
            producer_action="test_fixture",
            producer_backend="test",
            media_type="chemical/x-xyz",
        )
        request = _standardize_request(
            "standardize-file-artifact",
            structure=reference.model_dump(mode="json"),
        )
        adapter = action_jobs._adapter_for(request.action_id, request.backend_id)
        assert isinstance(adapter, action_jobs.ActionJobAdapter)
        prepared = action_jobs._submission_plan(request, adapter)
        assert not isinstance(prepared, dict)
        worker_request, staged_inputs, snapshots = (
            action_jobs._build_staged_worker_request(*prepared)
        )

    assert worker_request["inputs"]["structure"] == "inputs/structure.xyz"
    assert [item.model_dump(mode="json") for item in staged_inputs] == [
        {
            "source_path": "inputs/structure.xyz",
            "target_path": "inputs/structure.xyz",
        }
    ]
    assert snapshots[0]["sha256"] == reference.sha256


def test_compact_artifact_ref_collect_rebuilds_normalized_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """仅携带 artifact_id 的请求完成补全后，generic collect 必须成功。"""

    scope = _scope(tmp_path)
    _fake_start_job(monkeypatch, scope)
    with activate_scope(scope):
        reference = ArtifactStore(root=scope.workspace).put_json(
            STRUCTURE,
            semantic_type="AtomicStructure",
            producer_action="test_fixture",
            producer_backend="test",
        )
        request = _standardize_request(
            "standardize-compact-artifact",
            structure={"artifact_id": reference.artifact_id},
        )
        submitted = action_jobs.submit_action_job(request)
        job_id = str(submitted["job_id"])
        request_path = (
            scope.workspace
            / "outputs"
            / "execution_jobs"
            / job_id
            / "request.json"
        )
        request_record = json.loads(request_path.read_text(encoding="utf-8"))
        metadata = request_record["metadata"]
        assert metadata["caller_action_job_request"]["inputs"]["structure"] == {
            "artifact_id": reference.artifact_id
        }
        assert metadata["action_job_request"]["inputs"]["structure"] == reference.model_dump(
            mode="json"
        )
        _set_status(scope, job_id, "success")
        raw = ActionResult(
            status="success",
            action="standardize_structure",
            action_version="1.0.0",
            requested_backend="rdkit",
            backend="rdkit",
            selection_source="agent",
            result={"standardized": True},
        ).model_dump(mode="json")
        action_jobs._atomic_json(
            scope.workspace
            / "outputs"
            / "execution_jobs"
            / job_id
            / action_jobs.RAW_RESULT_FILENAME,
            raw,
        )
        collected = action_jobs.collect_action_job(
            job_id,
            submission_key="standardize-compact-artifact",
        )

    assert collected["status"] == "success"
    assert collected["action_result"]["status"] == "success"
    assert collected["action_result"]["action"] == "standardize_structure"


@pytest.mark.parametrize(
    "tampered_snapshot",
    [
        "caller_action_job_request",
        "action_job_request",
    ],
)
def test_tampered_action_request_snapshot_rejects_collect_and_keeps_gate(
    tampered_snapshot: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """篡改调用方或规范请求快照时必须拒绝 collect 并保留 active gate。"""

    scope = _scope(tmp_path)
    started = _fake_start_job(monkeypatch, scope)
    with activate_scope(scope):
        reference = ArtifactStore(root=scope.workspace).put_json(
            STRUCTURE,
            semantic_type="AtomicStructure",
            producer_action="test_fixture",
            producer_backend="test",
        )
        request = _standardize_request(
            "standardize-tampered-snapshot",
            structure={"artifact_id": reference.artifact_id},
        )
        submitted = action_jobs.submit_action_job(request)
        job_id = str(submitted["job_id"])
        request_path = (
            scope.workspace
            / "outputs"
            / "execution_jobs"
            / job_id
            / "request.json"
        )
        request_record = json.loads(request_path.read_text(encoding="utf-8"))
        request_record["metadata"][tampered_snapshot]["action_settings"][
            "neutralize"
        ] = True
        action_jobs._atomic_json(request_path, request_record)
        _set_status(scope, job_id, "success")
        raw = ActionResult(
            status="success",
            action="standardize_structure",
            action_version="1.0.0",
            requested_backend="rdkit",
            backend="rdkit",
            selection_source="agent",
            result={"standardized": True},
        ).model_dump(mode="json")
        action_jobs._atomic_json(
            scope.workspace
            / "outputs"
            / "execution_jobs"
            / job_id
            / action_jobs.RAW_RESULT_FILENAME,
            raw,
        )
        collected = action_jobs.collect_action_job(
            job_id,
            submission_key="standardize-tampered-snapshot",
        )
        next_submission = action_jobs.submit_action_job(
            _standardize_request("standardize-after-tamper")
        )

    assert collected["error"]["code"] == "action_job_result_unavailable"
    assert next_submission["error"]["code"] == "action_job_phase_gate_closed"
    assert len(started) == 1


def test_investigation_sync_action_points_to_generic_submit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Investigation 的同步入口不能把任意 Catalog Action 伪装成同步可执行。"""

    monkeypatch.setattr(action_jobs, "execute_traced", lambda _n, _a, fn, **_k: fn())
    request = action_jobs.ProgressiveActionRequest.model_validate(
        _scan_request("sync-rejected").model_dump(mode="json")
    )
    response = action_jobs.investigation_execute_action(request)
    assert response["error"]["code"] == "async_job_required"
    assert response["error"]["required_submit_tool"] == "submit_action_job"

    fast_response = action_jobs.investigation_execute_action(
        action_jobs.ProgressiveActionRequest.model_validate(
            _standardize_request("sync-fast-rejected").model_dump(mode="json")
        )
    )
    assert fast_response["error"]["code"] == "async_job_required"


def test_action_job_state_corruption_fails_closed_without_resubmission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """身份收据损坏后必须停止，不得以同一 submission_key 再启动 worker。"""

    scope = _scope(tmp_path)
    started = _fake_start_job(monkeypatch, scope)
    with activate_scope(scope):
        first = action_jobs.submit_action_job(_scan_request("scan-corrupt"))
        action_jobs._atomic_json(
            scope.workspace
            / "_tool_artifacts"
            / action_jobs.STATE_DIRECTORY_NAME
            / action_jobs.STATE_FILENAME,
            {"schema_version": action_jobs.STATE_SCHEMA_VERSION, "scope_id": "wrong-scope"},
        )
        replay = action_jobs.submit_action_job(_scan_request("scan-corrupt"))

    assert first["status"] == "success"
    assert replay["error"]["code"] == "action_job_state_unavailable"
    assert len(started) == 1


@pytest.mark.parametrize("terminal_status", ["failed", "timeout", "cancelled"])
def test_terminal_action_job_without_result_can_be_collected_and_releases_gate(
    terminal_status: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """失败、超时或取消作业收集后必须允许 scope 继续提交下一任务。"""

    scope = _scope(tmp_path)
    started = _fake_start_job(monkeypatch, scope)
    with activate_scope(scope):
        first = action_jobs.submit_action_job(_scan_request("scan-terminal"))
        _set_status(scope, str(first["job_id"]), terminal_status)
        collected = action_jobs.collect_action_job(
            str(first["job_id"]),
            submission_key="scan-terminal",
        )
        second = action_jobs.submit_action_job(_scan_request("scan-after-terminal"))

    assert collected["status"] == "success"
    assert collected["job_status"] == terminal_status
    assert collected["error"]["code"] == f"execution_{terminal_status}"
    assert second["status"] == "success"
    assert len(started) == 2


def test_fast_action_uses_real_detached_submit_get_collect_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """短 RDKit Action 也必须由真实 supervisor 返回 job_id 后再查询和收集。"""

    scope = _scope(tmp_path)
    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(scope.workspace))
    monkeypatch.setattr(action_jobs, "active_catalog_hash", lambda: "a" * 64)
    with activate_scope(scope):
        structure_ref = ArtifactStore(root=scope.workspace).put_json(
            STRUCTURE,
            semantic_type="AtomicStructure",
            producer_action="test_fixture",
            producer_backend="test",
        )
        request = _standardize_request(
            "standardize-real",
            structure=structure_ref.model_dump(mode="json"),
        )
        submitted = action_jobs.submit_action_job(request)
        assert submitted["status"] == "success"
        assert submitted["terminal"] is False
        job_id = str(submitted["job_id"])
        receipt = build_investigation_job_receipt(
            {
                "type": "mcpToolCall",
                "status": "completed",
                "tool": "submit_action_job",
                "arguments": {
                    "request": request.model_dump(mode="json")
                },
                "result": {"structuredContent": submitted},
            }
        )
        assert receipt is not None
        assert receipt.identity.job_id == job_id
        assert receipt.identity.job_family == "action_execution"

        status: dict[str, object] = {}
        for _ in range(300):
            status = action_jobs.get_action_job(job_id)
            if status.get("terminal") is True:
                break
            time.sleep(0.1)
        assert status["job_status"] == "success"

        collected = action_jobs.collect_action_job(
            job_id,
            submission_key="standardize-real",
        )

    assert collected["status"] == "success"
    assert collected["action_result"]["status"] == "success"
    assert collected["action_result"]["action"] == "standardize_structure"


def test_explicit_start_failure_is_queryable_and_collect_releases_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """资源或启动拒绝必须成为可查询终态，collect 后允许下一项。"""

    scope = _scope(tmp_path)
    monkeypatch.setattr(action_jobs, "_validate_action_backend_health", lambda _plan: None)
    monkeypatch.setattr(action_jobs, "active_catalog_hash", lambda: "a" * 64)
    monkeypatch.setattr(action_jobs, "execute_traced", lambda _n, _a, fn, **_k: fn())

    def reject_start(**kwargs):
        return {
            "status": "unavailable",
            "job_id": kwargs["job_id"],
            "error": {
                "code": "resource_state_unavailable",
                "message": "resource state unavailable",
            },
        }

    monkeypatch.setattr(action_jobs, "_start_job", reject_start)
    with activate_scope(scope):
        submitted = action_jobs.submit_action_job(_scan_request("scan-resource-failed"))
        status = open_tools.get_execution_job(
            JobStatusRequest(job_id=str(submitted["job_id"]))
        )
        collected = open_tools.collect_execution_job(
            JobCollectRequest(
                job_id=str(submitted["job_id"]),
                submission_key="scan-resource-failed",
            )
        )
        next_submission = action_jobs.submit_action_job(
            _scan_request("scan-after-resource-failed")
        )

    assert submitted["status"] == "success"
    assert submitted["job_status"] == "failed"
    assert status["job_status"] == "failed"
    assert status["error"]["code"] == "resource_state_unavailable"
    assert collected["job_status"] == "failed"
    assert next_submission["status"] == "success"
    assert next_submission["job_id"] != submitted["job_id"]


def test_investigation_gate_blocks_cross_family_submit_until_collect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Action 未 collect 时不得并行提交 native；收集后才开放下一类型。"""

    scope = _scope(tmp_path)
    _fake_start_job(monkeypatch, scope)
    native_calls: list[str] = []

    def submit_native(request: NativeJobRequest) -> dict[str, object]:
        native_calls.append(str(request.submission_key))
        return {"status": "success", "job_id": "job_12345678"}

    monkeypatch.setattr(open_tools, "submit_native_job", submit_native)
    native_request = NativeJobRequest(
        software_id="gaussian",
        executable="g16",
        submission_key="native-after-action",
        calculation_intent="single_point",
    )
    with activate_scope(scope):
        action_submission = (
            investigation_submission_gate.submit_investigation_action_job(
                _scan_request("cross-family-action")
            )
        )
        blocked = investigation_submission_gate.submit_investigation_native_job(
            native_request
        )
        _set_status(scope, str(action_submission["job_id"]), "failed")
        action_jobs.collect_action_job(
            str(action_submission["job_id"]),
            submission_key="cross-family-action",
        )
        allowed = investigation_submission_gate.submit_investigation_native_job(
            native_request
        )

    assert blocked["error"]["code"] == "investigation_job_phase_gate_closed"
    assert blocked["error"]["job_type"] == "action_execution"
    assert native_calls == ["native-after-action"]
    assert allowed["status"] == "success"


def test_action_job_family_is_visible_to_recovery_state_machine() -> None:
    """generic submit 回执必须能进入可恢复 Investigation family。"""

    item = {
        "type": "mcpToolCall",
        "status": "completed",
        "tool": "submit_action_job",
        "arguments": {"request": {"action_id": "scan_reaction_coordinates"}},
        "result": {
            "structuredContent": {
                "status": "success",
                "job_id": "job_12345678",
                "operation_id": "scope-operation-1",
                "submission_key": "scan-family",
                "action_id": "scan_reaction_coordinates",
                "backend_id": "gaussian",
                "job_family": "action_execution",
                "job_type": "action_execution",
                "runtime": "gaussian",
                "software_id": "gaussian",
                "operation_fingerprint": "a" * 64,
            }
        },
    }
    receipt = build_investigation_job_receipt(item)
    assert receipt is not None
    assert receipt.identity.job_family == "action_execution"
    continuation = build_investigation_job_continuation_request(
        "task-1",
        "batch-1",
        "combination-1",
        "thread-1",
        job_id=receipt.identity.job_id,
        operation_id=receipt.identity.operation_id,
        submission_key=receipt.identity.submission_key,
        action_id=receipt.identity.action_id,
        backend_id=receipt.identity.backend_id,
        job_family=receipt.identity.job_family,
        job_type=receipt.identity.job_type,
        runtime=receipt.identity.runtime,
        software_id=receipt.identity.software_id,
    )
    assert continuation["metadata"]["job_family"] == "action_execution"
    assert action_jobs.TOOL_NAMES == ("submit_action_job",)


@pytest.mark.parametrize("missing_field", ["job_type", "runtime"])
def test_action_job_receipt_rejects_incomplete_recovery_identity(
    missing_field: str,
) -> None:
    """Action 回执缺少 job type 或 runtime 时不得进入可恢复状态。"""

    result = {
        "status": "success",
        "job_id": "job_12345678",
        "operation_id": "scope-operation-1",
        "submission_key": "scan-family",
        "action_id": "scan_reaction_coordinates",
        "backend_id": "gaussian",
        "job_type": "action_execution",
        "runtime": "gaussian",
    }
    result.pop(missing_field)
    with pytest.raises(InvestigationJobSubmissionError):
        build_investigation_job_receipt(
            {
                "type": "mcpToolCall",
                "status": "completed",
                "tool": "submit_action_job",
                "arguments": {"request": {"action_id": "scan_reaction_coordinates"}},
                "result": {"structuredContent": result},
            }
        )
