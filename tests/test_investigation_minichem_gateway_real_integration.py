"""MiniChem verifier 与 Investigation gateway 的真实链路回归。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from minichem_mcp_tools import open_execution
from minichem_mcp_tools.execution_models import VerifyExecutionReferencesRequest
from minichem_mcp_tools.execution_job_registry import ExecutionJobRegistry
from minichem_toolbox.artifacts import ArtifactStore
from minichem_toolbox.idempotency import (
    OperationIdempotencyStore,
    calculate_request_fingerprint,
    scoped_submission_operation_id,
)
from minichem_toolbox.models import ActionResult
from minichem_toolbox.scope import ScopeCapabilityRegistry, activate_scope

from chemistry_multiagent.investigation.domain import (
    InvestigationResult,
    InvestigationSession,
)
from chemistry_multiagent.investigation.evidence import ReviewEvidenceStore
from chemistry_multiagent.investigation.minichem_gateway import (
    InvestigationMiniChemGateway,
    InvestigationScopeBinding,
)


TASK_ID = "task-real-gateway"
TOKEN = "A" * 48


def _real_scope_fixture(root: Path) -> tuple[Any, Path]:
    """创建临时真实 MiniChem scope，返回 capability scope 与其 workspace。"""

    registry = ScopeCapabilityRegistry(root / "registry")
    workspace = root / "workspace"
    workspace.mkdir(mode=0o700)
    capability = registry.provision_scope_capability(
        task_id=TASK_ID,
        batch_id="batch-1",
        combination_id="comb-1",
        scope_id="scope-1",
        workspace=workspace,
        token=TOKEN,
    )
    return replace(capability.scope, scheduler_root=root / "scheduler"), workspace


def _write_native_job(
    workspace: Path,
    scope: Any,
    *,
    operation_id: str,
    fingerprint: str,
) -> str:
    """写入可由真实 verifier 读取的 native job 终态记录。"""

    job_id = "job_" + hashlib.sha256(operation_id.encode("ascii")).hexdigest()[:32]
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
        json.dumps({**identity, "status": "success"}), encoding="utf-8"
    )
    (directory / "operation_receipt.json").write_text(
        json.dumps({**identity, "fingerprint": fingerprint, "phase": "success"}),
        encoding="utf-8",
    )
    return job_id


def _write_action_job(
    workspace: Path,
    scope: Any,
    *,
    artifact: Any,
    job_index: int,
    result_status: str = "success",
    result_error: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """写入 action_execution 终态记录，返回 job 与提交 operation 身份。"""

    job_id = f"job_{job_index:08x}"
    submission_key = f"case3-action-{job_index}"
    submission_operation_id = scoped_submission_operation_id(submission_key)
    fingerprint = calculate_request_fingerprint("case3_action_job", {"index": job_index})
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
    action_result = ActionResult(
        status=result_status,
        action="normalize_qcschema_molecule",
        action_version="1.0.0",
        requested_backend="qcelemental",
        backend="qcelemental",
        selection_source="agent",
        output_artifacts=[artifact],
        provenance={"async_job_id": job_id},
        error=result_error,
    )
    (directory / "action_result.json").write_text(
        json.dumps(action_result.model_dump(mode="json")),
        encoding="utf-8",
    )
    return job_id, submission_operation_id


class _RealVerifierSession:
    """把 gateway 的 MCP 调用转给真实 MiniChem verifier 函数。"""

    def __init__(self, scope: Any) -> None:
        self.scope = scope
        self.responses: list[dict[str, Any]] = []

    def initialize(self) -> dict[str, str]:
        return {"protocolVersion": "2025-06-18"}

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "get_execution_resources":
            return {
                "status": "success",
                "catalog_hash": "c" * 64,
                "catalog_version": "7",
            }
        if name != "verify_execution_references":
            raise AssertionError(f"unexpected tool: {name}")
        request = VerifyExecutionReferencesRequest.model_validate(arguments["request"])
        with activate_scope(self.scope):
            response = open_execution.verify_execution_references(request)
        self.responses.append(response)
        return response

    def close(self) -> None:
        return None


def test_real_verifier_gateway_round_trip_case3_shape(tmp_path: Path) -> None:
    """真实 verifier 生成 10 native+14 action/13/14 响应并保留来源链。"""

    scope, workspace = _real_scope_fixture(tmp_path / "minichem")
    typed_operations = [f"typed-operation-{index:02d}" for index in range(13)]
    with activate_scope(scope):
        for index, operation_id in enumerate(typed_operations):
            OperationIdempotencyStore().execute_idempotent_operation(
                operation_id,
                calculate_request_fingerprint("typed_action", {"index": index}),
                lambda index=index: {"status": "success", "index": index},
            )
        native_jobs = [
            _write_native_job(
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
        action_submission_operations = []
        failed_action_jobs = []
        for index in range(14):
            result_status = "failed" if index < 6 else "success"
            artifact = ArtifactStore().put_json(
                {
                    "artifact_index": index,
                    **(
                        {"failure_code": "gaussian_not_converged"}
                        if result_status == "failed"
                        else {}
                    ),
                },
                semantic_type=(
                    "BackendDiagnostic" if result_status == "failed" else "Result"
                ),
                producer_action="normalize_qcschema_molecule",
                producer_backend="qcelemental",
            )
            artifacts.append(artifact)
            action_job, submission_operation = _write_action_job(
                workspace,
                scope,
                artifact=artifact,
                job_index=index + 1,
                result_status=result_status,
                result_error=(
                    {"code": "gaussian_not_converged"}
                    if result_status == "failed"
                    else None
                ),
            )
            action_jobs.append(action_job)
            action_submission_operations.append(submission_operation)
            if result_status == "failed":
                failed_action_jobs.append(action_job)

    jobs = tuple([*native_jobs, *action_jobs])
    artifact_ids = tuple(artifact.artifact_id for artifact in artifacts)
    sessions: list[_RealVerifierSession] = []

    def session_factory(endpoint: str, headers: dict[str, str]) -> _RealVerifierSession:
        session = _RealVerifierSession(scope)
        sessions.append(session)
        return session

    gateway = InvestigationMiniChemGateway(
        "http://127.0.0.1:9010/mcp",
        {
            "scope-1": InvestigationScopeBinding(
                scope_id="scope-1",
                token=TOKEN,
                task_id=TASK_ID,
                batch_id="batch-1",
                combination_id="comb-1",
            )
        },
        evidence_store=ReviewEvidenceStore(tmp_path / "review-store"),
        source_cache_dir=tmp_path / "source-cache",
        session_factory=session_factory,
    )
    session = InvestigationSession(
        task_id=TASK_ID,
        batch_id="batch-1",
        combination_id="comb-1",
        codex_thread_id="thread-1",
        workspace_ref="workspace/task-real-gateway/comb-1",
        scope_id="scope-1",
        result_version="v1",
        phase="ready_for_batch_validation",
    )
    result = InvestigationResult(
        task_id=TASK_ID,
        batch_id="batch-1",
        combination_id="comb-1",
        result_version="v1",
        codex_thread_id="thread-1",
        scope_id="scope-1",
        turn_id="turn-v1",
        self_judgment="supported",
        narrative="结果叙述",
        job_refs=jobs,
        operation_refs=tuple(typed_operations),
        artifact_refs=artifact_ids,
        observations=("实际观测",),
    )

    with activate_scope(scope):
        verified = gateway.verify_investigation_result_with_evidence(session, result)
    assert len(sessions) == 1
    response = sessions[0].responses[0]
    response_jobs = response["items"][:24]
    assert response["schema_version"] == 2
    assert len(response["items"]) == 24 + 13 + 14
    assert [item["status"] for item in response["items"]] == [
        "valid"
    ] * (24 + 13 + 14)
    assert response["inventory_complete"] is True
    assert response["operation_inventory_complete"] is True
    assert response["all_valid"] is True
    assert len(failed_action_jobs) == 6
    assert [
        json.loads(
            (
                workspace
                / "outputs"
                / "execution_jobs"
                / job_id
                / "action_result.json"
            ).read_text(encoding="utf-8")
        )["status"]
        for job_id in failed_action_jobs
    ] == ["failed"] * 6
    assert [artifact.semantic_type for artifact in artifacts[:6]] == [
        "BackendDiagnostic"
    ] * 6
    assert all(item["terminal_status"] == "success" for item in response_jobs)
    assert all("submission_operation_id" not in item for item in response_jobs[:10])
    assert [
        item["submission_operation_id"] for item in response_jobs[10:]
    ] == action_submission_operations
    assert all(
        item["producer_job_id"] in set(action_jobs)
        and item["producer_submission_operation_id"]
        == action_submission_operations[action_jobs.index(item["producer_job_id"])]
        for item in response["items"][-14:]
    )

    evidence_payload = verified.evidence.to_payload()
    assert evidence_payload["reference_binding_version"] == "producer-job-v2"
    assert all(
        item["producer_submission_operation_id"] not in typed_operations
        for item in evidence_payload["artifacts"]
    )
    gateway.close()
