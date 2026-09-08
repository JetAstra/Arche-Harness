"""验证 Investigation 只读引用核对的终态、身份和文件完整性。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from minichem_mcp_tools import open_execution
from minichem_mcp_tools import discovery_tools
from minichem_mcp_tools.discovery_models import ActionBatchRequest
from minichem_mcp_tools.execution_models import (
    ScientificArtifactTextReadRequest,
    VerifyExecutionReferencesRequest,
)
from minichem_toolbox.artifacts import ArtifactStore
from minichem_toolbox.idempotency import (
    OperationIdempotencyStore,
    calculate_request_fingerprint,
    scoped_submission_operation_id,
)
from minichem_toolbox.models import ActionResult
from minichem_toolbox.scope import ScopeCapabilityRegistry, activate_scope
from minichem_toolbox import service
from minichem_mcp_tools.execution_job_registry import ExecutionJobRegistry


def _job_id(operation_id: str) -> str:
    return f"job_{hashlib.sha256(operation_id.encode('ascii')).hexdigest()[:32]}"


def _scope_fixture(tmp_path: Path):
    registry = ScopeCapabilityRegistry(tmp_path / "registry")
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    scheduler = tmp_path / "scheduler"
    capability = registry.provision_scope_capability(
        task_id="task-1",
        batch_id="batch-1",
        combination_id="combo-1",
        scope_id="scope-1",
        workspace=workspace,
        token="A" * 48,
    )
    return replace(capability.scope, scheduler_root=scheduler), workspace


def _write_terminal_job(
    workspace: Path,
    scope,
    *,
    operation_id: str,
    fingerprint: str,
    status: str = "success",
) -> str:
    job_id = _job_id(operation_id)
    directory = workspace / "outputs" / "execution_jobs" / job_id
    directory.mkdir(parents=True)
    identity = {
        "schema_version": 1,
        "job_id": job_id,
        "job_type": "native_software",
        "operation_id": operation_id,
        "operation_fingerprint": fingerprint,
        "scope_id": scope.scope_id,
        "scope_workspace": str(scope.workspace),
    }
    (directory / "status.json").write_text(
        json.dumps({**identity, "status": status}), encoding="utf-8"
    )
    (directory / "operation_receipt.json").write_text(
        json.dumps(
            {
                **identity,
                "fingerprint": fingerprint,
                "phase": status if status == "success" else "ready",
            }
        ),
        encoding="utf-8",
    )
    return job_id


def _write_terminal_action_job(
    workspace: Path,
    scope,
    *,
    operation_id: str,
    fingerprint: str,
    artifact,
    job_index: int,
) -> str:
    """写入可核对的 action_execution 终态记录及 canonical ActionResult。"""

    job_id = f"job_{job_index:08x}"
    submission_key = f"action-{job_index}"
    submission_operation_id = scoped_submission_operation_id(submission_key)
    ExecutionJobRegistry(workspace=workspace, scope_id=scope.scope_id).reserve(
        job_type="action_execution",
        operation_id=submission_operation_id,
        operation_fingerprint=fingerprint,
        caller_request_fingerprint=fingerprint,
        submission_key=submission_key,
        preferred_job_id=job_id,
    )
    directory = workspace / "outputs" / "execution_jobs" / job_id
    directory.mkdir(parents=True)
    identity = {
        "schema_version": 1,
        "job_id": job_id,
        "job_type": "action_execution",
        "operation_id": submission_operation_id,
        "operation_fingerprint": fingerprint,
        "caller_request_fingerprint": fingerprint,
        "scope_id": scope.scope_id,
        "scope_workspace": str(scope.workspace),
        "submission_key": submission_key,
    }
    (directory / "status.json").write_text(
        json.dumps(
            {
                **identity,
                "status": "success",
                "metadata": {
                    "action_id": "normalize_qcschema_molecule",
                    "backend_id": "qcelemental",
                },
            }
        ),
        encoding="utf-8",
    )
    (directory / "operation_receipt.json").write_text(
        json.dumps({**identity, "fingerprint": fingerprint, "phase": "success"}),
        encoding="utf-8",
    )
    result = ActionResult(
        status="success",
        action="normalize_qcschema_molecule",
        action_version="1.0.0",
        requested_backend="qcelemental",
        backend="qcelemental",
        selection_source="agent",
        output_artifacts=[artifact],
        provenance={"async_job_id": job_id},
    )
    (directory / "action_result.json").write_text(
        json.dumps(result.model_dump(mode="json")), encoding="utf-8"
    )
    return job_id


def _write_terminal_programmable_job(
    workspace: Path,
    scope,
    *,
    job_index: int,
    producer_id: str = "submission_operation",
) -> tuple[str, object, str]:
    """写入带 canonical programmable 输出清单的成功分析作业。"""

    job_id = f"job_{job_index:08x}"
    submission_key = f"analysis-{job_index}"
    submission_operation_id = scoped_submission_operation_id(submission_key)
    fingerprint = calculate_request_fingerprint(
        "programmable_analysis", {"job_index": job_index}
    )
    ExecutionJobRegistry(workspace=workspace, scope_id=scope.scope_id).reserve(
        job_type="programmable_analysis",
        operation_id=submission_operation_id,
        operation_fingerprint=fingerprint,
        caller_request_fingerprint=fingerprint,
        submission_key=submission_key,
        preferred_job_id=job_id,
    )
    directory = workspace / "outputs" / "execution_jobs" / job_id
    output_path = directory / "outputs" / "summary.json"
    output_path.parent.mkdir(parents=True)
    output_path.write_text('{"value": "programmable"}\n', encoding="utf-8")
    declaration = {
        "name": "summary",
        "path": "outputs/summary.json",
        "semantic_type": "AnalysisSummary",
        "media_type": "application/json",
        "required": True,
        "parent_artifact_ids": [],
    }
    identity = {
        "schema_version": 1,
        "job_id": job_id,
        "job_type": "programmable_analysis",
        "operation_id": submission_operation_id,
        "operation_fingerprint": fingerprint,
        "caller_request_fingerprint": fingerprint,
        "scope_id": scope.scope_id,
        "scope_workspace": str(scope.workspace),
        "submission_key": submission_key,
    }
    request_record = {
        **identity,
        "metadata": {"analysis_contract": {"outputs": [declaration]}},
    }
    (directory / "request.json").write_text(
        json.dumps(request_record), encoding="utf-8"
    )
    (directory / "status.json").write_text(
        json.dumps({**identity, "status": "success", "metadata": request_record["metadata"]}),
        encoding="utf-8",
    )
    (directory / "operation_receipt.json").write_text(
        json.dumps({**identity, "fingerprint": fingerprint, "phase": "success"}),
        encoding="utf-8",
    )
    producer = {
        "job": job_id,
        "submission_operation": submission_operation_id,
    }.get(producer_id, producer_id)
    artifact = ArtifactStore().register_file(
        output_path,
        semantic_type=declaration["semantic_type"],
        media_type=declaration["media_type"],
        producer_action=f"programmable_analysis:{producer}",
        producer_backend=None,
        parent_artifact_ids=declaration["parent_artifact_ids"],
    )
    manifest_artifact = {
        "name": declaration["name"],
        "path": declaration["path"],
        "semantic_type": declaration["semantic_type"],
        "media_type": declaration["media_type"],
        "parent_artifacts": declaration["parent_artifact_ids"],
        "required": declaration["required"],
        "workspace_path": f"outputs/execution_jobs/{job_id}/outputs/summary.json",
        "size_bytes": output_path.stat().st_size,
        "sha256": artifact.sha256,
        "producer": f"programmable_analysis:{job_id}",
        "validation_status": "valid",
        "validation_errors": [],
        "runtime_registration": None,
        "scientific_validation": {"status": "not_applicable", "checks": [], "errors": []},
    }
    (directory / "artifact_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "job_id": job_id,
                "artifact_status": "valid",
                "artifacts": [manifest_artifact],
            }
        ),
        encoding="utf-8",
    )
    return job_id, artifact, submission_operation_id


def test_verify_execution_references_returns_versioned_all_valid_result(
    tmp_path: Path,
) -> None:
    """同一 scope 的终态 job、typed Action receipt 和 artifact 可全部核对。"""

    scope, workspace = _scope_fixture(tmp_path)
    native_operation = "native-operation-1"
    native_fingerprint = "1" * 64
    job_id = _write_terminal_job(
        workspace,
        scope,
        operation_id=native_operation,
        fingerprint=native_fingerprint,
    )
    action_operation = "action-operation-1"
    action_fingerprint = calculate_request_fingerprint(
        "execute_action", {"action_id": "normalize_qcschema_molecule"}
    )
    with activate_scope(scope):
        OperationIdempotencyStore().execute_idempotent_operation(
            action_operation,
            action_fingerprint,
            lambda: {"status": "success"},
        )
        artifact = ArtifactStore().put_json(
            {"value": 1},
            semantic_type="Result",
            producer_action="normalize_qcschema_molecule",
            producer_backend="qcelemental",
        )
        action_job = _write_terminal_action_job(
            workspace,
            scope,
            operation_id=action_operation,
            fingerprint=action_fingerprint,
            artifact=artifact,
            job_index=1,
        )
        action_submission_operation_id = scoped_submission_operation_id("action-1")
        result = open_execution.verify_execution_references(
            VerifyExecutionReferencesRequest(
                jobs=[
                    {
                        "job_id": job_id,
                        "operation_id": native_operation,
                        "operation_fingerprint": native_fingerprint,
                    },
                    action_job,
                ],
                operations=[
                    {
                        "operation_id": action_operation,
                        "fingerprint": action_fingerprint,
                    }
                ],
                artifacts=[{"artifact_id": artifact.artifact_id}],
            )
        )
        id_only = open_execution.verify_execution_references(
            VerifyExecutionReferencesRequest(
                jobs=[job_id, action_job],
                operations=[action_operation],
                artifacts=[artifact.artifact_id],
            )
        )
        text_page = open_execution.read_scientific_artifact_text(
            ScientificArtifactTextReadRequest(
                artifact_id=artifact.artifact_id,
                limit=8,
                expected_sha256=artifact.sha256,
            )
        )

    assert result["schema_version"] == 2
    assert result["all_valid"] is True
    assert [item["status"] for item in result["items"]] == [
        "valid",
        "valid",
        "valid",
        "valid",
    ]
    assert "path" not in json.dumps(result)
    assert "stdout" not in json.dumps(result)
    assert result["scope_job_inventory"] == sorted(
        [
            {"job_id": job_id, "status": "success"},
            {
                "job_id": action_job,
                "status": "success",
                "submission_operation_id": action_submission_operation_id,
            },
        ],
        key=lambda item: item["job_id"],
    )
    assert result["has_active_jobs"] is False
    assert result["inventory_complete"] is True
    assert id_only["all_valid"] is True
    assert "submission_operation_id" not in id_only["items"][0]
    assert id_only["items"][2]["fingerprint"] == action_fingerprint
    assert id_only["items"][3]["producer_job_id"] == action_job
    assert text_page["status"] == "success"
    assert text_page["artifact_id"] == artifact.artifact_id
    assert text_page["content"]
    assert "path" not in json.dumps(text_page)


@pytest.mark.parametrize("producer_id", ("submission_operation", "job"))
def test_verify_execution_references_binds_programmable_output_to_submission_job(
    tmp_path: Path,
    producer_id: str,
) -> None:
    """成功 programmable_analysis 输出必须返回 job 与 submission operation 来源。"""

    scope, workspace = _scope_fixture(tmp_path)
    with activate_scope(scope):
        job_id, artifact, submission_operation_id = _write_terminal_programmable_job(
            workspace,
            scope,
            job_index=11,
            producer_id=producer_id,
        )
        result = open_execution.verify_execution_references(
            VerifyExecutionReferencesRequest(
                jobs=[job_id],
                artifacts=[artifact.artifact_id],
            )
        )

    assert result["all_valid"] is True
    assert result["items"][-1] == {
        "kind": "artifact",
        "status": "valid",
        "artifact_id": artifact.artifact_id,
        "sha256": artifact.sha256,
        "producer_job_id": job_id,
        "producer_submission_operation_id": submission_operation_id,
    }


def test_verify_execution_references_rejects_unrelated_programmable_producer_claim(
    tmp_path: Path,
) -> None:
    """programmable producer claim 不属于当前 job 时不能建立来源。"""

    scope, workspace = _scope_fixture(tmp_path)
    with activate_scope(scope):
        job_id, artifact, _submission_operation_id = _write_terminal_programmable_job(
            workspace,
            scope,
            job_index=12,
            producer_id="job_deadbeef",
        )
        result = open_execution.verify_execution_references(
            VerifyExecutionReferencesRequest(jobs=[job_id], artifacts=[artifact.artifact_id])
        )

    assert result["all_valid"] is False
    assert result["items"][-1] == {
        "kind": "artifact",
        "status": "invalid",
        "reason_code": "producer_binding_unavailable",
    }


@pytest.mark.parametrize(
    "corruption",
    ("manifest", "path", "sha", "file", "registration", "request", "receipt", "registry"),
)
def test_verify_execution_references_fails_closed_on_programmable_binding_corruption(
    tmp_path: Path,
    corruption: str,
) -> None:
    """programmable 的 manifest、身份、路径或摘要损坏时不猜测来源。"""

    scope, workspace = _scope_fixture(tmp_path)
    with activate_scope(scope):
        job_id, artifact, _submission_operation_id = _write_terminal_programmable_job(
            workspace,
            scope,
            job_index=13,
        )
        directory = workspace / "outputs" / "execution_jobs" / job_id
        if corruption in {"manifest", "path", "sha", "registration"}:
            manifest_path = directory / "artifact_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if corruption == "manifest":
                manifest["artifacts"][0]["validation_status"] = "invalid"
            elif corruption == "path":
                manifest["artifacts"][0]["path"] = "outputs/other.json"
            elif corruption == "sha":
                manifest["artifacts"][0]["sha256"] = "f" * 64
            else:
                manifest["artifacts"][0]["runtime_registration"] = {
                    "path": "outputs/other.json"
                }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        elif corruption == "file":
            output_path = directory / "outputs" / "summary.json"
            output_path.write_bytes(output_path.read_bytes() + b"tampered")
        elif corruption == "request":
            request_path = directory / "request.json"
            request = json.loads(request_path.read_text(encoding="utf-8"))
            request["metadata"]["analysis_contract"]["outputs"][0]["path"] = (
                "outputs/other.json"
            )
            request_path.write_text(json.dumps(request), encoding="utf-8")
        elif corruption == "receipt":
            receipt_path = directory / "operation_receipt.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["operation_id"] = scoped_submission_operation_id("other-operation")
            receipt["submission_key"] = "other-operation"
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        else:
            registry = ExecutionJobRegistry(workspace=workspace, scope_id=scope.scope_id)
            registry_path = registry.path
            registry_state = json.loads(registry_path.read_text(encoding="utf-8"))
            registry_state["jobs"][job_id]["operation_id"] = scoped_submission_operation_id(
                "other-operation"
            )
            registry_path.write_text(json.dumps(registry_state), encoding="utf-8")
        result = open_execution.verify_execution_references(
            VerifyExecutionReferencesRequest(jobs=[job_id], artifacts=[artifact.artifact_id])
        )

    assert result["all_valid"] is False
    assert result["items"][-1].get("status") != "valid"


@pytest.mark.parametrize("claim_kind", ("job", "operation"))
def test_verify_execution_references_rejects_cross_programmable_job_or_operation_claim(
    tmp_path: Path,
    claim_kind: str,
) -> None:
    """artifact 的 producer claim 不能借用另一个 programmable job 或提交 operation。"""

    scope, workspace = _scope_fixture(tmp_path)
    with activate_scope(scope):
        first_job, first_artifact, first_operation = _write_terminal_programmable_job(
            workspace,
            scope,
            job_index=14,
        )
        second_job, second_artifact, _second_operation = _write_terminal_programmable_job(
            workspace,
            scope,
            job_index=15,
            producer_id=first_job if claim_kind == "job" else first_operation,
        )
        result = open_execution.verify_execution_references(
            VerifyExecutionReferencesRequest(
                jobs=[first_job, second_job],
                artifacts=[first_artifact.artifact_id, second_artifact.artifact_id],
            )
        )

    assert result["all_valid"] is False
    assert result["items"][-2]["status"] == "valid"
    assert result["items"][-1] == {
        "kind": "artifact",
        "status": "invalid",
        "reason_code": "producer_binding_unavailable",
    }


def test_verify_execution_references_rejects_manual_artifact_without_job_source(
    tmp_path: Path,
) -> None:
    """没有 programmable 作业来源的人工文件不能升级为 evidence artifact。"""

    scope, workspace = _scope_fixture(tmp_path)
    manual_path = workspace / "manual-summary.json"
    manual_path.write_text('{"value": "manual"}\n', encoding="utf-8")
    with activate_scope(scope):
        artifact = ArtifactStore().register_file(
            manual_path,
            semantic_type="AnalysisSummary",
            media_type="application/json",
            producer_action="external_input:manual",
            producer_backend=None,
        )
        result = open_execution.verify_execution_references(
            VerifyExecutionReferencesRequest(artifacts=[artifact.artifact_id])
        )

    assert result["all_valid"] is False
    assert result["items"] == [
        {
            "kind": "artifact",
            "status": "invalid",
            "reason_code": "producer_binding_unavailable",
        }
    ]


@pytest.mark.parametrize(
    "action_result_status",
    (
        "success",
        "partial_success",
        "invalid_request",
        "unsupported",
        "unavailable",
        "failed",
        "timeout",
        "cancelled",
    ),
)
def test_verify_execution_references_accepts_canonical_action_result_after_successful_worker(
    tmp_path: Path,
    action_result_status: str,
) -> None:
    """外层 worker 成功时接受合同内科学状态并核对完整的产物来源。"""

    scope, workspace = _scope_fixture(tmp_path)
    fingerprint = calculate_request_fingerprint(
        "case3_failed_action", {"case": 3}
    )
    with activate_scope(scope):
        diagnostic_artifact = ArtifactStore().put_json(
            {"failure_code": "gaussian_not_converged"},
            semantic_type="Result",
            producer_action="normalize_qcschema_molecule",
            producer_backend="qcelemental",
        )
        job_id = _write_terminal_action_job(
            workspace,
            scope,
            operation_id="unused-typed-operation",
            fingerprint=fingerprint,
            artifact=diagnostic_artifact,
            job_index=1,
        )
        submission_operation_id = scoped_submission_operation_id("action-1")

    result_path = workspace / "outputs" / "execution_jobs" / job_id / "action_result.json"
    action_result = json.loads(result_path.read_text(encoding="utf-8"))
    action_result["status"] = action_result_status
    if action_result_status not in {"success", "partial_success"}:
        action_result["error"] = {"code": "synthetic_scientific_result"}
    result_path.write_text(json.dumps(action_result), encoding="utf-8")

    with activate_scope(scope):
        result = open_execution.verify_execution_references(
            VerifyExecutionReferencesRequest(
                jobs=[job_id],
                artifacts=[diagnostic_artifact.artifact_id],
            )
        )

    assert result["schema_version"] == 2
    assert result["items"] == [
        {
            "kind": "job",
            "status": "valid",
            "job_id": job_id,
            "operation_fingerprint": fingerprint,
            "terminal_status": "success",
            "submission_operation_id": submission_operation_id,
        },
        {
            "kind": "artifact",
            "status": "valid",
            "artifact_id": diagnostic_artifact.artifact_id,
            "sha256": diagnostic_artifact.sha256,
            "producer_job_id": job_id,
            "producer_submission_operation_id": submission_operation_id,
        },
    ]
    assert result["inventory_complete"] is True
    assert result["operation_inventory_complete"] is True
    assert result["all_valid"] is True


def test_verify_execution_references_ignores_unreferenced_failed_action_without_result(
    tmp_path: Path,
) -> None:
    """未引用的正常失败作业不能污染另一组成功引用的核验结果。"""

    scope, workspace = _scope_fixture(tmp_path)
    with activate_scope(scope):
        failed_artifact = ArtifactStore().put_json(
            {"value": "unreferenced-failure"},
            semantic_type="BackendDiagnostic",
            producer_action="normalize_qcschema_molecule",
            producer_backend="qcelemental",
        )
        failed_job_id = _write_terminal_action_job(
            workspace,
            scope,
            operation_id="unreferenced-failed-action",
            fingerprint=calculate_request_fingerprint(
                "action_job", {"status": "failed", "referenced": False}
            ),
            artifact=failed_artifact,
            job_index=20,
        )
        successful_artifact = ArtifactStore().put_json(
            {"value": "referenced-success"},
            semantic_type="Result",
            producer_action="normalize_qcschema_molecule",
            producer_backend="qcelemental",
        )
        successful_job_id = _write_terminal_action_job(
            workspace,
            scope,
            operation_id="referenced-successful-action",
            fingerprint=calculate_request_fingerprint(
                "action_job", {"status": "success", "referenced": True}
            ),
            artifact=successful_artifact,
            job_index=21,
        )

    failed_directory = workspace / "outputs" / "execution_jobs" / failed_job_id
    failed_status_path = failed_directory / "status.json"
    failed_status = json.loads(failed_status_path.read_text(encoding="utf-8"))
    failed_status["status"] = "failed"
    failed_status_path.write_text(json.dumps(failed_status), encoding="utf-8")
    failed_receipt_path = failed_directory / "operation_receipt.json"
    failed_receipt_path.unlink()
    (failed_directory / "action_result.json").unlink()

    with activate_scope(scope):
        result = open_execution.verify_execution_references(
            VerifyExecutionReferencesRequest(
                jobs=[successful_job_id],
                artifacts=[successful_artifact.artifact_id],
            )
        )

    assert [item["status"] for item in result["items"]] == ["valid", "valid"]
    assert {item["job_id"] for item in result["scope_job_inventory"]} == {
        failed_job_id,
        successful_job_id,
    }
    assert result["inventory_complete"] is True
    assert result["operation_inventory_complete"] is True
    assert result["all_valid"] is True


def test_verify_execution_references_ignores_unreferenced_running_typed_operation(
    tmp_path: Path,
) -> None:
    """未被结果引用的探索 operation 不应否决另一组完整的成功证据。"""

    scope, workspace = _scope_fixture(tmp_path)
    exploratory_operation = "unreferenced-exploratory-operation"
    exploratory_fingerprint = calculate_request_fingerprint(
        "execute_action", {"purpose": "unreferenced-exploration"}
    )
    with activate_scope(scope):
        OperationIdempotencyStore().write_running_operation_receipt(
            exploratory_operation,
            exploratory_fingerprint,
        )
        successful_artifact = ArtifactStore().put_json(
            {"value": "referenced-success"},
            semantic_type="Result",
            producer_action="normalize_qcschema_molecule",
            producer_backend="qcelemental",
        )
        successful_job_id = _write_terminal_action_job(
            workspace,
            scope,
            operation_id="referenced-successful-action",
            fingerprint=calculate_request_fingerprint(
                "action_job", {"status": "success", "referenced": True}
            ),
            artifact=successful_artifact,
            job_index=22,
        )
        result = open_execution.verify_execution_references(
            VerifyExecutionReferencesRequest(
                jobs=[successful_job_id],
                artifacts=[successful_artifact.artifact_id],
            )
        )

    assert [item["status"] for item in result["items"]] == ["valid", "valid"]
    assert result["scope_operation_inventory"] == [
        {"operation_id": exploratory_operation, "status": "running"}
    ]
    assert result["has_active_operations"] is True
    assert result["operation_inventory_complete"] is True
    assert result["all_valid"] is True


@pytest.mark.parametrize("worker_status", ("failed", "timeout", "cancelled"))
def test_verify_execution_references_rejects_conflicting_worker_and_action_status(
    tmp_path: Path,
    worker_status: str,
) -> None:
    """外层 worker 失败、超时或取消时，冲突的科学状态必须保持不可用。"""

    scope, workspace = _scope_fixture(tmp_path)
    with activate_scope(scope):
        artifact = ArtifactStore().put_json(
            {"value": "conflicting-status"},
            semantic_type="Result",
            producer_action="normalize_qcschema_molecule",
            producer_backend="qcelemental",
        )
        job_id = _write_terminal_action_job(
            workspace,
            scope,
            operation_id="conflicting-worker-status",
            fingerprint=calculate_request_fingerprint(
                "action_job", {"value": "conflicting-status"}
            ),
            artifact=artifact,
            job_index=2,
        )
        status_path = workspace / "outputs" / "execution_jobs" / job_id / "status.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status["status"] = worker_status
        status_path.write_text(json.dumps(status), encoding="utf-8")
        receipt_path = (
            workspace / "outputs" / "execution_jobs" / job_id / "operation_receipt.json"
        )
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["phase"] = worker_status
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        result = open_execution.verify_execution_references(
            VerifyExecutionReferencesRequest(jobs=[job_id])
        )

    assert result["items"] == [
        {
            "kind": "job",
            "status": "valid",
            "job_id": job_id,
            "operation_fingerprint": status["operation_fingerprint"],
            "terminal_status": worker_status,
            "submission_operation_id": status["operation_id"],
        }
    ]
    assert result["inventory_complete"] is True
    assert result["operation_inventory_complete"] is True
    assert result["all_valid"] is False


@pytest.mark.parametrize("terminal_status", ("failed", "timeout", "cancelled"))
def test_verify_execution_references_accepts_matching_worker_receipt_and_action_status(
    tmp_path: Path,
    terminal_status: str,
) -> None:
    """外层非成功终态与 receipt、科学状态同名时仍可核对产物来源。"""

    scope, workspace = _scope_fixture(tmp_path)
    with activate_scope(scope):
        artifact = ArtifactStore().put_json(
            {"value": f"matching-{terminal_status}"},
            semantic_type="BackendDiagnostic",
            producer_action="normalize_qcschema_molecule",
            producer_backend="qcelemental",
        )
        job_id = _write_terminal_action_job(
            workspace,
            scope,
            operation_id=f"matching-{terminal_status}",
            fingerprint=calculate_request_fingerprint(
                "action_job", {"status": terminal_status}
            ),
            artifact=artifact,
            job_index=3,
        )
    directory = workspace / "outputs" / "execution_jobs" / job_id
    status_path = directory / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["status"] = terminal_status
    status_path.write_text(json.dumps(status), encoding="utf-8")
    receipt_path = directory / "operation_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["phase"] = terminal_status
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    action_result_path = directory / "action_result.json"
    action_result = json.loads(action_result_path.read_text(encoding="utf-8"))
    action_result["status"] = terminal_status
    action_result["error"] = {"code": f"synthetic_{terminal_status}"}
    action_result_path.write_text(json.dumps(action_result), encoding="utf-8")

    with activate_scope(scope):
        result = open_execution.verify_execution_references(
            VerifyExecutionReferencesRequest(
                jobs=[job_id],
                artifacts=[artifact.artifact_id],
            )
        )

    assert [item["status"] for item in result["items"]] == ["valid", "valid"]
    assert result["all_valid"] is True


@pytest.mark.parametrize(
    ("worker_status", "receipt_phase"),
    (("success", "failed"), ("failed", "timeout"), ("cancelled", "success")),
)
def test_verify_execution_references_rejects_conflicting_worker_and_receipt_status(
    tmp_path: Path,
    worker_status: str,
    receipt_phase: str,
) -> None:
    """同一物理作业的 status 与 receipt 终态冲突时不得接受引用。"""

    scope, workspace = _scope_fixture(tmp_path)
    with activate_scope(scope):
        artifact = ArtifactStore().put_json(
            {"value": "receipt-status-conflict"},
            semantic_type="Result",
            producer_action="normalize_qcschema_molecule",
            producer_backend="qcelemental",
        )
        job_id = _write_terminal_action_job(
            workspace,
            scope,
            operation_id="receipt-status-conflict",
            fingerprint=calculate_request_fingerprint(
                "action_job", {"value": "receipt-status-conflict"}
            ),
            artifact=artifact,
            job_index=4,
        )
    directory = workspace / "outputs" / "execution_jobs" / job_id
    status_path = directory / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["status"] = worker_status
    status_path.write_text(json.dumps(status), encoding="utf-8")
    receipt_path = directory / "operation_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["phase"] = receipt_phase
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    action_result_path = directory / "action_result.json"
    action_result = json.loads(action_result_path.read_text(encoding="utf-8"))
    action_result["status"] = worker_status
    if worker_status != "success":
        action_result["error"] = {"code": f"synthetic_{worker_status}"}
    action_result_path.write_text(json.dumps(action_result), encoding="utf-8")

    with activate_scope(scope):
        result = open_execution.verify_execution_references(
            VerifyExecutionReferencesRequest(
                jobs=[job_id],
                artifacts=[artifact.artifact_id],
            )
        )

    assert result["all_valid"] is False


def test_verify_execution_references_recovers_each_action_artifact_operation_binding(
    tmp_path: Path,
) -> None:
    """同 scope 的 canonical ActionResult 必须逐 artifact 返回唯一 operation 归属。"""

    scope, workspace = _scope_fixture(tmp_path)
    first_operation = "action-operation-first"
    second_operation = "action-operation-second"
    first_fingerprint = calculate_request_fingerprint(
        "action_job", {"value": "first"}
    )
    second_fingerprint = calculate_request_fingerprint(
        "action_job", {"value": "second"}
    )
    with activate_scope(scope):
        for operation_id, fingerprint in (
            (first_operation, first_fingerprint),
            (second_operation, second_fingerprint),
        ):
            OperationIdempotencyStore().execute_idempotent_operation(
                operation_id,
                fingerprint,
                lambda: {"status": "success"},
            )
        first_artifact = ArtifactStore().put_json(
            {"value": "first"},
            semantic_type="Result",
            producer_action="normalize_qcschema_molecule",
            producer_backend="qcelemental",
        )
        second_artifact = ArtifactStore().put_json(
            {"value": "second"},
            semantic_type="Result",
            producer_action="normalize_qcschema_molecule",
            producer_backend="qcelemental",
        )
        first_job = _write_terminal_action_job(
            workspace,
            scope,
            operation_id=first_operation,
            fingerprint=first_fingerprint,
            artifact=first_artifact,
            job_index=1,
        )
        second_job = _write_terminal_action_job(
            workspace,
            scope,
            operation_id=second_operation,
            fingerprint=second_fingerprint,
            artifact=second_artifact,
            job_index=2,
        )
        first_submission_operation_id = scoped_submission_operation_id("action-1")
        second_submission_operation_id = scoped_submission_operation_id("action-2")
        result = open_execution.verify_execution_references(
            VerifyExecutionReferencesRequest(
                jobs=[first_job, second_job],
                operations=[first_operation, second_operation],
                # 请求顺序故意与 job 顺序相反，归属不能依赖列表位置。
                artifacts=[second_artifact.artifact_id, first_artifact.artifact_id],
            )
        )

    assert result["all_valid"] is True
    assert result["scope_operation_inventory"] == [
        {"operation_id": first_operation, "status": "completed"},
        {"operation_id": second_operation, "status": "completed"},
    ]
    assert result["items"][-2:] == [
        {
            "kind": "artifact",
            "status": "valid",
            "artifact_id": second_artifact.artifact_id,
                "sha256": second_artifact.sha256,
                "producer_job_id": second_job,
                "producer_submission_operation_id": second_submission_operation_id,
        },
        {
            "kind": "artifact",
            "status": "valid",
            "artifact_id": first_artifact.artifact_id,
                "sha256": first_artifact.sha256,
                "producer_job_id": first_job,
                "producer_submission_operation_id": first_submission_operation_id,
        },
    ]
    assert all("operation_id" not in item for item in result["items"][-2:])


@pytest.mark.parametrize("corruption", ("missing", "duplicate", "mismatch", "hash"))
def test_verify_execution_references_fails_closed_on_action_artifact_binding_corruption(
    tmp_path: Path,
    corruption: str,
) -> None:
    """ActionResult 缺失、重复归属或摘要不一致时不得猜测 artifact operation。"""

    scope, workspace = _scope_fixture(tmp_path)
    first_operation = "action-operation-first"
    second_operation = "action-operation-second"
    first_fingerprint = calculate_request_fingerprint("action_job", {"value": "first"})
    second_fingerprint = calculate_request_fingerprint("action_job", {"value": "second"})
    with activate_scope(scope):
        for operation_id, fingerprint in (
            (first_operation, first_fingerprint),
            (second_operation, second_fingerprint),
        ):
            OperationIdempotencyStore().execute_idempotent_operation(
                operation_id,
                fingerprint,
                lambda: {"status": "success"},
            )
        first_artifact = ArtifactStore().put_json(
            {"value": "first"},
            semantic_type="Result",
            producer_action="normalize_qcschema_molecule",
            producer_backend="qcelemental",
        )
        second_artifact = ArtifactStore().put_json(
            {"value": "second"},
            semantic_type="Result",
            producer_action="normalize_qcschema_molecule",
            producer_backend="qcelemental",
        )
        first_job = _write_terminal_action_job(
            workspace,
            scope,
            operation_id=first_operation,
            fingerprint=first_fingerprint,
            artifact=first_artifact,
            job_index=1,
        )
        second_job = _write_terminal_action_job(
            workspace,
            scope,
            operation_id=second_operation,
            fingerprint=second_fingerprint,
            artifact=second_artifact,
            job_index=2,
        )
        second_result_path = (
            workspace / "outputs" / "execution_jobs" / second_job / "action_result.json"
        )
        if corruption == "missing":
            second_result_path.unlink()
        elif corruption == "duplicate":
            second_result = json.loads(second_result_path.read_text(encoding="utf-8"))
            second_result["output_artifacts"].append(
                json.loads(json.dumps(first_artifact.model_dump(mode="json")))
            )
            second_result_path.write_text(json.dumps(second_result), encoding="utf-8")
        else:
            second_result = json.loads(second_result_path.read_text(encoding="utf-8"))
            if corruption == "mismatch":
                second_result["output_artifacts"][0]["sha256"] = "f" * 64
                second_result_path.write_text(
                    json.dumps(second_result), encoding="utf-8"
                )
            else:
                artifact_path = workspace / second_artifact.path
                artifact_path.write_bytes(artifact_path.read_bytes() + b"tampered")
        result = open_execution.verify_execution_references(
            VerifyExecutionReferencesRequest(
                jobs=[first_job, second_job],
                operations=[first_operation, second_operation],
                artifacts=[first_artifact.artifact_id, second_artifact.artifact_id],
            )
        )

    assert result["all_valid"] is False
    artifact_items = result["items"][-2:]
    assert any(item.get("status") != "valid" for item in artifact_items)


def test_verify_execution_references_rejects_action_result_producer_job_conflict(
    tmp_path: Path,
) -> None:
    """ActionResult 指向另一个 job 时，artifact 来源链不能被 status/receipt 掩盖。"""

    scope, workspace = _scope_fixture(tmp_path)
    typed_operation = "typed-action-operation"
    typed_fingerprint = calculate_request_fingerprint(
        "action_job", {"value": "typed"}
    )
    with activate_scope(scope):
        OperationIdempotencyStore().execute_idempotent_operation(
            typed_operation,
            typed_fingerprint,
            lambda: {"status": "success"},
        )
        artifact = ArtifactStore().put_json(
            {"value": "producer-conflict"},
            semantic_type="Result",
            producer_action="normalize_qcschema_molecule",
            producer_backend="qcelemental",
        )
        job_id = _write_terminal_action_job(
            workspace,
            scope,
            operation_id=typed_operation,
            fingerprint=typed_fingerprint,
            artifact=artifact,
            job_index=7,
        )
        result_path = (
            workspace / "outputs" / "execution_jobs" / job_id / "action_result.json"
        )
        action_result = json.loads(result_path.read_text(encoding="utf-8"))
        action_result["provenance"]["async_job_id"] = "job_00000008"
        result_path.write_text(json.dumps(action_result), encoding="utf-8")
        result = open_execution.verify_execution_references(
            VerifyExecutionReferencesRequest(
                jobs=[job_id],
                operations=[typed_operation],
                artifacts=[artifact.artifact_id],
            )
        )

    assert result["all_valid"] is False
    assert result["items"][-1] == {
        "kind": "artifact",
        "status": "invalid",
        "reason_code": "producer_binding_unavailable",
    }


@pytest.mark.parametrize(
    ("section", "field", "value"),
    (
        ("result", "action", "copied_action"),
        ("result", "backend", "copied_backend"),
        ("artifact", "producer_action", "copied_action"),
        ("artifact", "producer_backend", "copied_backend"),
    ),
)
def test_verify_execution_references_cross_binds_action_and_artifact_producers(
    tmp_path: Path,
    section: str,
    field: str,
    value: str,
) -> None:
    """ActionResult 与 Artifact producer 身份不一致时不能伪造来源。"""

    scope, workspace = _scope_fixture(tmp_path)
    with activate_scope(scope):
        artifact = ArtifactStore().put_json(
            {"value": "producer-identity"},
            semantic_type="Result",
            producer_action="normalize_qcschema_molecule",
            producer_backend="qcelemental",
        )
        job_id = _write_terminal_action_job(
            workspace,
            scope,
            operation_id="typed-action-operation",
            fingerprint=calculate_request_fingerprint(
                "action_job", {"value": "producer-identity"}
            ),
            artifact=artifact,
            job_index=9,
        )
        result_path = (
            workspace / "outputs" / "execution_jobs" / job_id / "action_result.json"
        )
        action_result = json.loads(result_path.read_text(encoding="utf-8"))
        if section == "result":
            action_result[field] = value
        else:
            action_result["output_artifacts"][0][field] = value
        result_path.write_text(json.dumps(action_result), encoding="utf-8")
        result = open_execution.verify_execution_references(
            VerifyExecutionReferencesRequest(
                jobs=[job_id],
                artifacts=[artifact.artifact_id],
            )
        )

    assert result["all_valid"] is False
    assert result["items"][-1] == {
        "kind": "artifact",
        "status": "invalid",
        "reason_code": "producer_binding_unavailable",
    }


def test_verify_execution_references_matches_case3_multi_inventory_shape(
    tmp_path: Path,
) -> None:
    """24 jobs、13 typed operations、14 artifacts 的真实形状保持命名空间分离。"""

    scope, workspace = _scope_fixture(tmp_path)
    typed_operations = [f"typed-operation-{index:02d}" for index in range(13)]
    with activate_scope(scope):
        for index, operation_id in enumerate(typed_operations):
            OperationIdempotencyStore().execute_idempotent_operation(
                operation_id,
                calculate_request_fingerprint("typed_action", {"index": index}),
                lambda index=index: {"status": "success", "index": index},
            )
        native_jobs = [
            _write_terminal_job(
                workspace,
                scope,
                operation_id=f"native-job-operation-{index:02d}",
                fingerprint=calculate_request_fingerprint(
                    "native_job", {"index": index}
                ),
            )
            for index in range(10)
        ]
        artifacts = []
        action_jobs = []
        for index in range(14):
            artifact = ArtifactStore().put_json(
                {"artifact_index": index},
                semantic_type="Result",
                producer_action="normalize_qcschema_molecule",
                producer_backend="qcelemental",
            )
            artifacts.append(artifact)
            action_jobs.append(
                _write_terminal_action_job(
                    workspace,
                    scope,
                    operation_id=f"typed-operation-for-job-{index:02d}",
                    fingerprint=calculate_request_fingerprint(
                        "action_job", {"index": index}
                    ),
                    artifact=artifact,
                    job_index=index + 1,
                )
            )
        result = open_execution.verify_execution_references(
            VerifyExecutionReferencesRequest(
                jobs=[*native_jobs, *action_jobs],
                operations=typed_operations,
                artifacts=[artifact.artifact_id for artifact in artifacts],
            )
        )

    assert result["schema_version"] == 2
    assert result["all_valid"] is True
    assert len(result["scope_job_inventory"]) == 24
    assert len(result["scope_operation_inventory"]) == 13
    assert len(result["items"]) == 24 + 13 + 14
    producer_items = result["items"][-14:]
    typed_operation_ids = set(typed_operations)
    assert all(
        item["producer_submission_operation_id"] not in typed_operation_ids
        and item["producer_job_id"] in set(action_jobs)
        and "operation_id" not in item
        for item in producer_items
    )


def test_verify_execution_references_fails_closed_on_foreign_action_artifact_record(
    tmp_path: Path,
) -> None:
    """复制 foreign scope 的 ActionResult 到当前 scope 时不能建立伪归属。"""

    foreign_scope, foreign_workspace = _scope_fixture(tmp_path / "foreign")
    with activate_scope(foreign_scope):
        foreign_artifact = ArtifactStore().put_json(
            {"foreign": True},
            semantic_type="Result",
            producer_action="normalize_qcschema_molecule",
            producer_backend="qcelemental",
        )
    scope, workspace = _scope_fixture(tmp_path / "current")
    operation_id = "action-operation-current"
    fingerprint = calculate_request_fingerprint("action_job", {"value": "current"})
    with activate_scope(scope):
        OperationIdempotencyStore().execute_idempotent_operation(
            operation_id,
            fingerprint,
            lambda: {"status": "success"},
        )
        job_id = _write_terminal_action_job(
            workspace,
            scope,
            operation_id=operation_id,
            fingerprint=fingerprint,
            artifact=foreign_artifact,
            job_index=1,
        )
        result = open_execution.verify_execution_references(
            VerifyExecutionReferencesRequest(
                jobs=[job_id],
                operations=[operation_id],
                artifacts=[foreign_artifact.artifact_id],
            )
        )

    assert result["all_valid"] is False
    assert result["items"][-1]["status"] != "valid"


def test_verify_execution_references_rejects_forged_native_submission_identity(
    tmp_path: Path,
) -> None:
    """native job 携带无法由 scoped operation 证明的 submission_key 时必须不可用。"""

    scope, workspace = _scope_fixture(tmp_path)
    operation_id = "native-operation-with-forged-submission"
    fingerprint = calculate_request_fingerprint("native_job", {"value": 1})
    job_id = _write_terminal_job(
        workspace,
        scope,
        operation_id=operation_id,
        fingerprint=fingerprint,
    )
    directory = workspace / "outputs" / "execution_jobs" / job_id
    for filename in ("status.json", "operation_receipt.json"):
        path = directory / filename
        value = json.loads(path.read_text(encoding="utf-8"))
        value["submission_key"] = "forged-native-submission"
        path.write_text(json.dumps(value), encoding="utf-8")

    with activate_scope(scope):
        result = open_execution.verify_execution_references(
            VerifyExecutionReferencesRequest(jobs=[job_id])
        )

    assert result["all_valid"] is False
    assert result["items"] == [{"kind": "job", "status": "unavailable"}]


def test_verify_execution_references_rejects_duplicate_job_references(
    tmp_path: Path,
) -> None:
    """同一个 job 在请求中重复出现时不能用重复项通过完整性核对。"""

    scope, workspace = _scope_fixture(tmp_path)
    job_id = _write_terminal_job(
        workspace,
        scope,
        operation_id="duplicate-job-operation",
        fingerprint=calculate_request_fingerprint("native_job", {"value": 1}),
    )
    with activate_scope(scope):
        result = open_execution.verify_execution_references(
            VerifyExecutionReferencesRequest(jobs=[job_id, {"job_id": job_id}])
        )

    assert result["all_valid"] is False
    assert result["items"] == [
        {"kind": "job", "status": "invalid", "reason_code": "duplicate_reference"},
        {"kind": "job", "status": "invalid", "reason_code": "duplicate_reference"},
    ]


def test_verify_execution_references_rejects_duplicate_artifact_references(
    tmp_path: Path,
) -> None:
    """同一个 Artifact 在请求中重复出现时不能重复计数为两份证据。"""

    scope, workspace = _scope_fixture(tmp_path)
    with activate_scope(scope):
        artifact = ArtifactStore().put_json(
            {"value": "duplicate-artifact"},
            semantic_type="Result",
            producer_action="normalize_qcschema_molecule",
            producer_backend="qcelemental",
        )
        job_id = _write_terminal_action_job(
            workspace,
            scope,
            operation_id="duplicate-artifact-operation",
            fingerprint=calculate_request_fingerprint(
                "action_job", {"value": "duplicate-artifact"}
            ),
            artifact=artifact,
            job_index=10,
        )
        result = open_execution.verify_execution_references(
            VerifyExecutionReferencesRequest(
                jobs=[job_id],
                artifacts=[artifact.artifact_id, {"artifact_id": artifact.artifact_id}],
            )
        )

    assert result["all_valid"] is False
    assert result["items"][0]["status"] == "valid"
    assert result["items"][1:] == [
        {
            "kind": "artifact",
            "status": "invalid",
            "reason_code": "duplicate_reference",
        },
        {
            "kind": "artifact",
            "status": "invalid",
            "reason_code": "duplicate_reference",
        },
    ]


def test_verify_execution_references_marks_running_job_invalid(tmp_path: Path) -> None:
    """运行中的 job 不能被当作已核对的终态结果。"""

    scope, workspace = _scope_fixture(tmp_path)
    operation_id = "running-operation-1"
    fingerprint = "2" * 64
    job_id = _write_terminal_job(
        workspace,
        scope,
        operation_id=operation_id,
        fingerprint=fingerprint,
        status="running",
    )
    with activate_scope(scope):
        result = open_execution.verify_execution_references(
            VerifyExecutionReferencesRequest(
                jobs=[
                    {
                        "job_id": job_id,
                        "operation_id": operation_id,
                        "operation_fingerprint": fingerprint,
                    }
                ]
            )
        )

    assert result["all_valid"] is False
    assert result["items"] == [
        {
            "kind": "job",
            "status": "invalid",
            "reason_code": "job_not_terminal",
        }
    ]
    assert result["scope_job_inventory"] == [
        {"job_id": job_id, "status": "running"}
    ]
    assert result["has_active_jobs"] is True
    assert result["inventory_complete"] is True


def test_verify_execution_references_hides_cross_scope_and_unknown_identity(
    tmp_path: Path,
) -> None:
    """未知或另一 scope 的 job/artifact 统一返回 unavailable 且不泄漏路径。"""

    first_scope, first_workspace = _scope_fixture(tmp_path / "first")
    operation_id = "foreign-operation-1"
    fingerprint = "3" * 64
    foreign_job_id = _write_terminal_job(
        first_workspace,
        first_scope,
        operation_id=operation_id,
        fingerprint=fingerprint,
    )
    with activate_scope(first_scope):
        foreign_artifact = ArtifactStore().put_json(
            {"foreign": True},
            semantic_type="Result",
            producer_action="normalize_qcschema_molecule",
            producer_backend="qcelemental",
        )
    second_scope, _second_workspace = _scope_fixture(tmp_path / "second")
    with activate_scope(second_scope):
        result = open_execution.verify_execution_references(
            VerifyExecutionReferencesRequest(
                jobs=[
                    {
                        "job_id": foreign_job_id,
                        "operation_id": operation_id,
                        "operation_fingerprint": fingerprint,
                    }
                ],
                operations=[
                    {
                        "operation_id": operation_id,
                        "fingerprint": fingerprint,
                    }
                ],
                artifacts=[{"artifact_id": foreign_artifact.artifact_id}],
            )
        )
        text_page = open_execution.read_scientific_artifact_text(
            ScientificArtifactTextReadRequest(
                artifact_id=foreign_artifact.artifact_id,
            )
        )

    assert result["all_valid"] is False
    assert all(item["status"] == "unavailable" for item in result["items"])
    serialized = json.dumps(result, ensure_ascii=False)
    assert str(first_workspace) not in serialized
    assert "foreign-operation-1" not in serialized
    assert result["scope_job_inventory"] == []
    assert result["has_active_jobs"] is False
    assert result["inventory_complete"] is True
    assert text_page == {"status": "unavailable", "automatic_fallback": False}


def test_verify_execution_references_rejects_corrupt_receipt_without_guessing(
    tmp_path: Path,
) -> None:
    """损坏 job receipt 只能报告 unavailable，不能猜测 operation 已完成。"""

    scope, workspace = _scope_fixture(tmp_path)
    operation_id = "corrupt-operation-1"
    fingerprint = "4" * 64
    job_id = _write_terminal_job(
        workspace,
        scope,
        operation_id=operation_id,
        fingerprint=fingerprint,
    )
    receipt = workspace / "outputs" / "execution_jobs" / job_id / "operation_receipt.json"
    receipt.write_text("{not-json", encoding="utf-8")
    typed_operation_id = "corrupt-typed-operation-1"
    typed_fingerprint = "5" * 64
    with activate_scope(scope):
        operation_store = OperationIdempotencyStore()
        operation_store.execute_idempotent_operation(
            typed_operation_id,
            typed_fingerprint,
            lambda: {"status": "success"},
        )
        operation_receipt, _lock_path = operation_store._paths(typed_operation_id)
        operation_receipt.write_text("{not-json", encoding="utf-8")
        result = open_execution.verify_execution_references(
            VerifyExecutionReferencesRequest(
                jobs=[
                    {
                        "job_id": job_id,
                        "operation_id": operation_id,
                        "operation_fingerprint": fingerprint,
                    }
                ],
                operations=[
                    {
                        "operation_id": typed_operation_id,
                        "fingerprint": typed_fingerprint,
                    }
                ],
            )
        )

    assert result["all_valid"] is False
    assert result["items"] == [
        {"kind": "job", "status": "unavailable"},
        {"kind": "operation", "status": "unavailable"},
    ]


def test_verify_execution_references_allows_unreferenced_completed_operation_inventory(
    tmp_path: Path,
) -> None:
    """结果只声明实际证据；同 scope 的其他已完成 operation 仍保留在清单中。"""

    scope, _workspace = _scope_fixture(tmp_path)
    first = "inventory-operation-first"
    second = "inventory-operation-second"
    with activate_scope(scope):
        for operation_id, value in ((first, 1), (second, 2)):
            OperationIdempotencyStore().execute_idempotent_operation(
                operation_id,
                calculate_request_fingerprint("execute_action", {"value": value}),
                lambda value=value: {"status": "success", "value": value},
            )
        result = open_execution.verify_execution_references(
            VerifyExecutionReferencesRequest(operations=[first])
        )

    assert result["all_valid"] is True
    assert result["operation_inventory_complete"] is True
    assert result["has_active_operations"] is False
    assert result["scope_operation_inventory"] == [
        {"operation_id": first, "status": "completed"},
        {"operation_id": second, "status": "completed"},
    ]


def test_verify_execution_references_reports_active_operation_inventory(
    tmp_path: Path,
) -> None:
    """遗留 running typed operation 必须保留活动状态并阻止最终核验。"""

    scope, _workspace = _scope_fixture(tmp_path)
    operation_id = "active-inventory-operation"
    fingerprint = calculate_request_fingerprint("execute_action", {"value": 1})
    with activate_scope(scope):
        OperationIdempotencyStore().write_running_operation_receipt(
            operation_id, fingerprint
        )
        result = open_execution.verify_execution_references(
            VerifyExecutionReferencesRequest(operations=[operation_id])
        )

    assert result["all_valid"] is False
    assert result["scope_operation_inventory"] == [
        {"operation_id": operation_id, "status": "running"}
    ]
    assert result["has_active_operations"] is True
    assert result["operation_inventory_complete"] is True
    assert result["items"] == [
        {
            "kind": "operation",
            "status": "invalid",
            "reason_code": "operation_not_completed",
        }
    ]


def test_verify_execution_references_fails_closed_on_corrupt_operation_inventory(
    tmp_path: Path,
) -> None:
    """typed operation 收据损坏时 inventory 不猜测活动状态，也不泄漏身份。"""

    scope, workspace = _scope_fixture(tmp_path)
    operation_id = "corrupt-inventory-operation"
    fingerprint = calculate_request_fingerprint("execute_action", {"value": 1})
    with activate_scope(scope):
        OperationIdempotencyStore().execute_idempotent_operation(
            operation_id,
            fingerprint,
            lambda: {"status": "success"},
        )
    operations_root = workspace / "_tool_artifacts" / "operations"
    receipt = next(operations_root.glob("[0-9a-f]*.json"))
    receipt.write_text("{not-json", encoding="utf-8")

    with activate_scope(scope):
        result = open_execution.verify_execution_references(
            VerifyExecutionReferencesRequest(operations=[operation_id])
        )

    assert result["all_valid"] is False
    assert result["scope_operation_inventory"] == []
    assert result["has_active_operations"] is None
    assert result["operation_inventory_complete"] is False
    assert result["items"] == [{"kind": "operation", "status": "unavailable"}]
    assert operation_id not in json.dumps(result, ensure_ascii=False)


def test_verify_execution_references_rejects_foreign_operation_receipt(
    tmp_path: Path,
) -> None:
    """复制到当前 workspace 的 foreign receipt 不能变成可见 operation。"""

    first_scope, first_workspace = _scope_fixture(tmp_path / "first")
    operation_id = "foreign-inventory-operation"
    fingerprint = calculate_request_fingerprint("execute_action", {"value": 1})
    with activate_scope(first_scope):
        OperationIdempotencyStore().execute_idempotent_operation(
            operation_id,
            fingerprint,
            lambda: {"status": "success"},
        )
    first_receipt = next(
        (first_workspace / "_tool_artifacts" / "operations").glob("[0-9a-f]*.json")
    )

    second_scope, second_workspace = _scope_fixture(tmp_path / "second")
    second_operations = second_workspace / "_tool_artifacts" / "operations"
    second_operations.mkdir(parents=True)
    second_receipt = second_operations / first_receipt.name
    second_receipt.write_bytes(first_receipt.read_bytes())
    with activate_scope(second_scope):
        result = open_execution.verify_execution_references(
            VerifyExecutionReferencesRequest(operations=[operation_id])
        )

    assert result["all_valid"] is False
    assert result["scope_operation_inventory"] == []
    assert result["has_active_operations"] is None
    assert result["operation_inventory_complete"] is False
    assert operation_id not in json.dumps(result, ensure_ascii=False)


def test_verify_execution_references_recovers_operation_inventory_after_store_restart(
    tmp_path: Path,
) -> None:
    """新的 OperationStore 实例可从持久 receipt 恢复完整 typed operation 清单。"""

    scope, _workspace = _scope_fixture(tmp_path)
    operation_id = "restart-recovery-operation"
    fingerprint = calculate_request_fingerprint("execute_action", {"value": 1})
    with activate_scope(scope):
        OperationIdempotencyStore().execute_idempotent_operation(
            operation_id,
            fingerprint,
            lambda: {"status": "success"},
        )
        restarted = OperationIdempotencyStore(create=False)
        inventory, active, complete = restarted.list_operation_inventory()

    assert inventory == [{"operation_id": operation_id, "status": "completed"}]
    assert active is False
    assert complete is True


def test_verify_execution_references_fails_closed_on_orphan_operation_lock(
    tmp_path: Path,
) -> None:
    """没有对应 receipt 的持久 operation 锁表示身份恢复不完整。"""

    scope, workspace = _scope_fixture(tmp_path)
    operation_id = "orphan-lock-operation"
    with activate_scope(scope):
        store = OperationIdempotencyStore()
        _receipt, lock = store._paths(operation_id)
        lock.touch()
        result = open_execution.verify_execution_references(
            VerifyExecutionReferencesRequest()
        )

    assert result["all_valid"] is False
    assert result["scope_operation_inventory"] == []
    assert result["has_active_operations"] is None
    assert result["operation_inventory_complete"] is False
    assert str(workspace) not in json.dumps(result, ensure_ascii=False)


def test_verify_execution_references_includes_batch_parent_and_children(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """真实 batch wrapper 的父、child operation 都必须进入 scope inventory。"""

    scope, _workspace = _scope_fixture(tmp_path)
    specification = SimpleNamespace(batch_safe=True, backend_ids=("test_backend",))
    monkeypatch.setattr(
        discovery_tools,
        "action_specs",
        lambda: {"test_batch_action": specification},
    )
    monkeypatch.setattr(
        service,
        "action_specs",
        lambda: {"test_batch_action": SimpleNamespace(version="1.0.0")},
    )
    monkeypatch.setattr(
        service,
        "_execute_action_once",
        lambda _action_id, _request, resources_bound=False: {
            "status": "success",
            "action": "test_batch_action",
            "output_artifacts": [],
        },
    )
    # 本测试只验证 batch 父子 operation 是否进入引用清单，不依赖真实 Catalog
    # 的 backend 选择和资源约束；显式隔离资源规格解析，保持测试边界不变。
    monkeypatch.setattr(
        service,
        "_resolve_action_resource_backend_specs",
        lambda _action_id, _request: (),
    )

    request = ActionBatchRequest.model_validate(
        {
            "action_id": "test_batch_action",
            "backend_id": "test_backend",
            "items": [
                {"item_id": "item-a", "inputs": {"value": 1}},
                {"item_id": "item-b", "inputs": {"value": 2}},
            ],
            "submission_key": "batch-parent-scope",
        }
    )
    with activate_scope(scope):
        # 让 batch 资源门槛在测试中保持可用；实际服务仍会逐 child 走
        # service.execute_action 的 OperationStore 入口。
        monkeypatch.setattr(
            discovery_tools,
            "evaluation_resource_budget",
            lambda: SimpleNamespace(
                cpu_cores=8,
                memory_mb=32768,
                gpu_count=0,
                as_dict=lambda: {
                    "cpu_cores": 8,
                    "memory_mb": 32768,
                    "gpu_count": 0,
                },
            ),
        )
        monkeypatch.setattr(
            discovery_tools,
            "active_resource_usage",
            lambda: {"cpu_cores": 0, "memory_mb": 0, "gpu_count": 0},
        )
        monkeypatch.setattr(discovery_tools, "active_resource_jobs", lambda: [])
        submitted = discovery_tools.submit_action_batch(request)
        assert submitted["status"] == "success"
        parent_operation_id = scoped_submission_operation_id(
            request.submission_key
        )
        child_ids = [
            scoped_submission_operation_id(
                discovery_tools._batch_child_submission_key(
                    request.submission_key,
                    item.item_id,
                    index,
                )
            )
            for index, item in enumerate(request.items)
        ]
        result = open_execution.verify_execution_references(
            VerifyExecutionReferencesRequest(
                operations=[parent_operation_id, *child_ids]
            )
        )

    assert result["all_valid"] is True
    assert result["has_active_operations"] is False
    assert result["operation_inventory_complete"] is True
    assert result["scope_operation_inventory"] == [
        {"operation_id": operation_id, "status": "completed"}
        for operation_id in sorted([parent_operation_id, *child_ids])
    ]


def test_read_scientific_artifact_text_matches_evidence_media_contract(
    tmp_path: Path,
) -> None:
    """Artifact 文本读取支持 Evidence 的 XYZ/SDF/JSONL 媒体，并拒绝二进制。"""

    scope, workspace = _scope_fixture(tmp_path)
    xyz_path = workspace / "molecule.xyz"
    xyz_path.write_text("1\nH\nH 0.0 0.0 0.0\n", encoding="utf-8")
    jsonl_path = workspace / "facts.jsonl"
    jsonl_path.write_text('{"value": 1}\n', encoding="utf-8")
    sdf_path = workspace / "molecule.sdf"
    sdf_path.write_text("ethanol\n  MiniChem\n\n$$$$\n", encoding="utf-8")
    binary_path = workspace / "image.bin"
    binary_path.write_bytes(b"\\x00\\x01\\x02")
    with activate_scope(scope):
        store = ArtifactStore()
        xyz = store.register_file(
            xyz_path,
            semantic_type="MoleculeGeometry",
            producer_action="investigation",
            producer_backend=None,
            media_type="chemical/x-xyz",
        )
        jsonl = store.register_file(
            jsonl_path,
            semantic_type="ObservationLog",
            producer_action="investigation",
            producer_backend=None,
            media_type="application/jsonl",
        )
        sdf = store.register_file(
            sdf_path,
            semantic_type="AtomicStructure",
            producer_action="investigation",
            producer_backend=None,
            media_type="chemical/x-mdl-sdfile",
        )
        binary = store.register_file(
            binary_path,
            semantic_type="BinaryOutput",
            producer_action="investigation",
            producer_backend=None,
            media_type="application/octet-stream",
        )
        xyz_result = open_execution.read_scientific_artifact_text(
            ScientificArtifactTextReadRequest(artifact_id=xyz.artifact_id)
        )
        jsonl_result = open_execution.read_scientific_artifact_text(
            ScientificArtifactTextReadRequest(artifact_id=jsonl.artifact_id)
        )
        sdf_result = open_execution.read_scientific_artifact_text(
            ScientificArtifactTextReadRequest(artifact_id=sdf.artifact_id)
        )
        binary_result = open_execution.read_scientific_artifact_text(
            ScientificArtifactTextReadRequest(artifact_id=binary.artifact_id)
        )

    assert xyz_result["status"] == "success"
    assert xyz_result["content"].startswith("1\nH")
    assert jsonl_result["status"] == "success"
    assert jsonl_result["content"].startswith('{"value": 1}')
    assert sdf_result["status"] == "success"
    assert sdf_result["content"].endswith("$$$$\n")
    assert binary_result["status"] == "invalid"
    assert binary_result["reason_code"] == "artifact_not_text"
