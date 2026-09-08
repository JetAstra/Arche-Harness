"""Investigation MiniChem 只读网关的公开 interface 合同测试。"""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import json
import stat
import threading
import time
from contextlib import AsyncExitStack
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from chemistry_multiagent.investigation.domain import (
    InvestigationResult,
    InvestigationSession,
    InvestigationState,
)
from chemistry_multiagent.investigation.async_jobs import (
    InvestigationJobIdentity,
    InvestigationJobState,
)
from chemistry_multiagent.investigation import minichem_gateway as gateway_module
from chemistry_multiagent.investigation.evidence import ReviewEvidenceStore
from chemistry_multiagent.investigation.minichem_gateway import (
    InvestigationMiniChemGateway,
    InvestigationMiniChemGatewayError,
    InvestigationMiniChemProtocolError,
    InvestigationMiniChemReferenceRejected,
    InvestigationMiniChemScopeError,
    InvestigationMiniChemUnavailable,
    InvestigationScopeBinding,
    VerifiedInvestigationResult,
)
from chemistry_multiagent.memory import VerifiedInvestigationEvidence


TASK_ID = "task-gateway"
BATCH_ID = "batch-gateway"
TOKEN = "capability-token-for-tests-0123456789"
CONTENT = "第一段\n第二段\n最后一段\n"
CONTENT_BYTES = CONTENT.encode("utf-8")
CONTENT_SHA256 = hashlib.sha256(CONTENT_BYTES).hexdigest()
ARTIFACT_ID = "art_0123456789abcdef0123456789abcdef"
EXPECTED_CATALOG_HASH = "c" * 64
EXPECTED_CATALOG_VERSION = "7"


def _job_id(index: int) -> str:
    return "job_" + f"{index:032x}"


def _short_job_id(index: int) -> str:
    return "job_" + f"{index:08x}"


def _submission_operation_id(job_id: str) -> str:
    """返回测试中与 job 一一对应的异步提交 operation。"""

    return _scope_submission_operation_id(job_id)


def _scope_submission_operation_id(job_id: str) -> str:
    """返回真实 MiniChem scope-submission operation 形状。"""

    return "scope-submission-" + hashlib.sha256(job_id.encode("utf-8")).hexdigest()


def _binding(index: int) -> InvestigationScopeBinding:
    return InvestigationScopeBinding(
        scope_id=f"scope-{index}",
        token=TOKEN + str(index),
        task_id=TASK_ID,
        batch_id=BATCH_ID,
        combination_id=f"combination-{index}",
    )


def _session(index: int, *, phase: str = "ready_for_batch_validation") -> InvestigationSession:
    return InvestigationSession(
        task_id=TASK_ID,
        batch_id=BATCH_ID,
        combination_id=f"combination-{index}",
        codex_thread_id=f"thread-{index}",
        workspace_ref=f"workspace-{index}",
        scope_id=f"scope-{index}",
        result_version="v1",
        phase=phase,
    )


def _result(
    index: int,
    *,
    jobs: tuple[str, ...] = (),
    operations: tuple[str, ...] = (),
    artifacts: tuple[str, ...] = (),
) -> InvestigationResult:
    return InvestigationResult(
        task_id=TASK_ID,
        batch_id=BATCH_ID,
        combination_id=f"combination-{index}",
        result_version="v1",
        codex_thread_id=f"thread-{index}",
        scope_id=f"scope-{index}",
        turn_id=f"turn-{index}",
        self_judgment="supported",
        narrative="结果叙述",
        job_refs=jobs,
        operation_refs=operations,
        artifact_refs=artifacts,
        observations=("实际观测",),
    )


def _state(results: tuple[InvestigationResult, ...]) -> InvestigationState:
    return InvestigationState(
        task_id=TASK_ID,
        batch_id=BATCH_ID,
        sessions=tuple(_session(index) for index in range(5)),
        results=results,
    )


def _waiting_ts(index: int = 1) -> InvestigationJobState:
    """构造一个已持久化 TS waiting 状态，供 Controller unknown 测试使用。"""

    return InvestigationJobState(
        task_id=TASK_ID,
        batch_id=BATCH_ID,
        combination_id=f"combination-{index}",
        codex_thread_id=f"thread-{index}",
        turn_id=f"turn-{index}",
        status="waiting_investigation_job",
        identity=InvestigationJobIdentity(
            job_id=_job_id(index),
            operation_id=f"operation-{index}",
            submission_key=f"submission-{index}",
            action_id="search_transition_state_path",
            backend_id="transition_state_path_search",
        ),
        request_digest="a" * 64,
        scope_id=f"scope-{index}",
    )


class RecordingSession:
    """只实现网关 interface 的同步 MCP session 替身。"""

    def __init__(self, *, mode: str = "normal") -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.mode = mode
        self.initialize_count = 0
        self.resource_call_count = 0

    def initialize(self) -> dict[str, Any]:
        self.initialize_count += 1
        if self.mode == "initialize-error":
            raise RuntimeError("scope authorization rejected")
        return {"protocolVersion": "2025-06-18"}

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        assert set(arguments) == {"request"}
        request = arguments["request"]
        self.calls.append((name, json.loads(json.dumps(request))))
        if name == "get_execution_resources":
            self.resource_call_count += 1
            if self.mode == "transport":
                raise RuntimeError("resource transport failed")
            if self.mode == "is-error":
                return {"isError": True}
            response: dict[str, Any] = {
                "status": "success",
                "catalog_hash": EXPECTED_CATALOG_HASH,
                "catalog_version": EXPECTED_CATALOG_VERSION,
            }
            if self.mode == "bad-status":
                response["status"] = "failed"
            elif self.mode == "missing-hash":
                response.pop("catalog_hash")
            elif self.mode == "malformed-hash":
                response["catalog_hash"] = "not-a-hash"
            elif self.mode == "drift":
                response["catalog_hash"] = "d" * 64
            elif self.mode == "missing-version":
                response.pop("catalog_version")
            elif self.mode == "empty-version":
                response["catalog_version"] = " "
            elif self.mode == "version-drift":
                response["catalog_version"] = "8"
            elif self.mode == "drift-once" and self.resource_call_count == 1:
                response["catalog_hash"] = "d" * 64
            return response
        if name == "verify_execution_references":
            items: list[dict[str, Any]] = []
            for job_id in request["jobs"]:
                items.append(
                    {
                        "kind": "job",
                        "status": "valid",
                        "job_id": job_id,
                        "submission_operation_id": _submission_operation_id(job_id),
                        "operation_fingerprint": "a" * 64,
                        "terminal_status": "success",
                    }
                )
            for operation_id in request["operations"]:
                items.append(
                    {
                        "kind": "operation",
                        "status": "valid",
                        "operation_id": operation_id,
                        "fingerprint": "b" * 64,
                        "operation_status": "completed",
                    }
                )
            for artifact_id in request["artifacts"]:
                items.append(
                    {
                        "kind": "artifact",
                        "status": "valid",
                        "artifact_id": artifact_id,
                        "sha256": CONTENT_SHA256,
                        "producer_job_id": request["jobs"][0]
                        if request["jobs"]
                        else "",
                        "producer_submission_operation_id": (
                            _submission_operation_id(request["jobs"][0])
                            if request["jobs"]
                            else ""
                        ),
                    }
                )
            return {
                "schema_version": 2,
                "status": "success",
                "all_valid": True,
                "items": items,
                "scope_job_inventory": [
                    {
                        "job_id": job_id,
                        "status": "success",
                        "submission_operation_id": _submission_operation_id(job_id),
                    }
                    for job_id in request["jobs"]
                ],
                "has_active_jobs": False,
                "inventory_complete": True,
                "scope_operation_inventory": [
                    {"operation_id": operation_id, "status": "completed"}
                    for operation_id in request["operations"]
                ],
                "has_active_operations": False,
                "operation_inventory_complete": True,
                "automatic_fallback": False,
            }
        if name == "read_scientific_artifact_text":
            offset = request["offset"]
            limit = request["limit"]
            page = CONTENT[offset : offset + limit]
            next_offset = offset + len(page) if offset + len(page) < len(CONTENT) else None
            return {
                "schema_version": 1,
                "status": "success",
                "artifact_id": request["artifact_id"],
                "semantic_type": "scientific-text",
                "media_type": "text/plain",
                "sha256": CONTENT_SHA256,
                "encoding": "utf-8",
                "offset": offset,
                "returned_chars": len(page),
                "next_offset": next_offset,
                "total_chars": len(CONTENT),
                "truncated": next_offset is not None,
                "content": page,
                "automatic_fallback": False,
            }
        raise AssertionError(f"unexpected tool: {name}")

    def close(self) -> None:
        return None


def _gateway(
    tmp_path: Path,
    session_factory: Any,
    *,
    page_size: int = 8192,
    expected_catalog_hash: str | None = None,
    expected_catalog_version: str | None = None,
) -> InvestigationMiniChemGateway:
    return InvestigationMiniChemGateway(
        "http://127.0.0.1:9010/mcp",
        {_binding(index).scope_id: _binding(index) for index in range(5)},
        evidence_store=ReviewEvidenceStore(tmp_path / "review-store"),
        source_cache_dir=tmp_path / "source-cache",
        session_factory=session_factory,
        page_size=page_size,
        expected_catalog_hash=expected_catalog_hash,
        expected_catalog_version=expected_catalog_version,
    )


def test_preflight_reads_catalog_identity_after_auth_and_is_repeatable(
    tmp_path: Path,
) -> None:
    """认证成功后必须读取正式服务身份，重复 preflight 不重建 scope session。"""

    sessions: list[RecordingSession] = []

    def factory(endpoint: str, headers: dict[str, str]) -> RecordingSession:
        assert endpoint == "http://127.0.0.1:9010/mcp"
        assert headers == {"X-MiniChem-Capability": TOKEN + "0"}
        session = RecordingSession()
        sessions.append(session)
        return session

    gateway = _gateway(
        tmp_path,
        factory,
        expected_catalog_hash=EXPECTED_CATALOG_HASH,
        expected_catalog_version=EXPECTED_CATALOG_VERSION,
    )
    first = gateway.preflight_scope_readiness(_session(0))
    second = gateway.preflight_scope_readiness(_session(0))

    assert first == {
        "scope_id": "scope-0",
        "authenticated": True,
        "catalog_hash": EXPECTED_CATALOG_HASH,
        "catalog_version": EXPECTED_CATALOG_VERSION,
    }
    assert second == first
    assert len(sessions) == 1
    assert sessions[0].initialize_count == 1
    assert sessions[0].calls == [
        ("get_execution_resources", {}),
        ("get_execution_resources", {}),
    ]
    gateway.close()


@pytest.mark.parametrize(
    "mode",
    [
        "bad-status",
        "missing-hash",
        "malformed-hash",
        "drift",
        "missing-version",
        "empty-version",
        "version-drift",
        "is-error",
        "transport",
        "initialize-error",
    ],
)
def test_preflight_catalog_identity_fail_closed(
    tmp_path: Path,
    mode: str,
) -> None:
    """身份缺失、漂移、认证或传输失败都必须阻断正式模型阶段。"""

    gateway = _gateway(
        tmp_path / mode,
        lambda mode=mode: RecordingSession(mode=mode),
        expected_catalog_hash=EXPECTED_CATALOG_HASH,
        expected_catalog_version=EXPECTED_CATALOG_VERSION,
    )
    with pytest.raises(InvestigationMiniChemGatewayError) as error:
        gateway.preflight_scope_readiness(_session(0))
    assert TOKEN not in str(error.value)
    assert str(tmp_path) not in str(error.value)
    gateway.close()


def test_preflight_can_retry_identity_after_transient_drift_without_reauth(
    tmp_path: Path,
) -> None:
    """恢复重试只重复只读身份核对，不重复 factory 或 initialize。"""

    sessions: list[RecordingSession] = []

    def factory() -> RecordingSession:
        session = RecordingSession(mode="drift-once")
        sessions.append(session)
        return session

    gateway = _gateway(
        tmp_path,
        factory,
        expected_catalog_hash=EXPECTED_CATALOG_HASH,
        expected_catalog_version=EXPECTED_CATALOG_VERSION,
    )
    with pytest.raises(InvestigationMiniChemProtocolError):
        gateway.preflight_scope_readiness(_session(0))
    manifest = gateway.preflight_scope_readiness(_session(0))

    assert manifest["catalog_hash"] == EXPECTED_CATALOG_HASH
    assert len(sessions) == 1
    assert sessions[0].initialize_count == 1
    assert sessions[0].resource_call_count == 2
    gateway.close()


@pytest.mark.parametrize("expected_hash", ["bad", True, 7])
def test_gateway_rejects_invalid_expected_catalog_hash(
    tmp_path: Path,
    expected_hash: Any,
) -> None:
    """构造时不接受可伪造或非字符串的 expected catalog hash。"""

    with pytest.raises(ValueError, match="expected_catalog_hash"):
        _gateway(
            tmp_path,
            lambda: RecordingSession(),
            expected_catalog_hash=expected_hash,
        )


@pytest.mark.parametrize("expected_version", ["", " 7", "7 ", "7\n", 7, True])
def test_gateway_rejects_invalid_expected_catalog_version(
    tmp_path: Path,
    expected_version: Any,
) -> None:
    """构造时不接受空白、控制字符或非字符串的 expected 版本。"""

    with pytest.raises(ValueError, match="expected_catalog_version"):
        _gateway(
            tmp_path,
            lambda: RecordingSession(),
            expected_catalog_version=expected_version,
        )


def test_verifier_uses_authenticated_scope_wire_and_returns_same_result(tmp_path: Path) -> None:
    """验证请求只使用 ID-only envelope、scope header 和两类只读工具。"""

    sessions: list[RecordingSession] = []

    def factory(endpoint: str, headers: dict[str, str]) -> RecordingSession:
        assert endpoint == "http://127.0.0.1:9010/mcp"
        assert headers == {"X-MiniChem-Capability": TOKEN + "0"}
        session = RecordingSession()
        sessions.append(session)
        return session

    gateway = _gateway(tmp_path, factory)
    result = _result(
        0,
        jobs=(_job_id(1),),
        operations=("operation-1",),
        artifacts=(ARTIFACT_ID,),
    )
    verified = gateway.result_verifier(_session(0), result)
    assert verified is result
    assert sessions[0].calls == [
        (
            "verify_execution_references",
            {
                "jobs": [_job_id(1)],
                "operations": ["operation-1"],
                "artifacts": [ARTIFACT_ID],
            },
        )
    ]
    assert "submit" not in {name for name, _ in sessions[0].calls}
    gateway.close()


@pytest.mark.parametrize("job_id", (_short_job_id(1), _job_id(1)))
def test_verifier_accepts_current_and_legacy_job_ids(
    tmp_path: Path,
    job_id: str,
) -> None:
    """结果核验同时接受当前短 ID 和已有运行中的旧长 ID。"""

    gateway = _gateway(tmp_path / job_id, lambda: RecordingSession())
    result = _result(0, jobs=(job_id,))

    assert gateway.result_verifier(_session(0), result) is result
    gateway.close()


@pytest.mark.parametrize(
    "job_id",
    (
        "job_0123456",
        "job_012345678",
        "job_0123456g",
        "job-01234567",
        " job_01234567",
    ),
)
def test_verifier_rejects_malformed_job_ids(tmp_path: Path, job_id: str) -> None:
    """短 ID 兼容不能放宽前缀、长度、字符集或空白保护。"""

    gateway = _gateway(
        tmp_path / hashlib.sha256(job_id.encode()).hexdigest(),
        lambda: RecordingSession(),
    )
    result = _result(0, jobs=(job_id,))

    with pytest.raises(InvestigationMiniChemProtocolError, match="job_refs 含无效引用"):
        gateway.result_verifier(_session(0), result)
    gateway.close()


def test_verified_result_api_returns_stable_memory_evidence_without_private_data(
    tmp_path: Path,
) -> None:
    """显式核验 API 同时返回原结果和稳定、脱敏的 Working Memory 证据。"""

    class AlignedRecordingSession(RecordingSession):
        def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            response = super().call_tool(name, arguments)
            if name == "verify_execution_references":
                response["items"][0]["submission_operation_id"] = _submission_operation_id(
                    _job_id(1)
                )
            return response

    gateway = _gateway(tmp_path, lambda: AlignedRecordingSession())
    result = _result(
        0,
        jobs=(_job_id(1),),
        operations=("operation-1",),
        artifacts=(ARTIFACT_ID,),
    )

    first = gateway.verify_investigation_result_with_evidence(_session(0), result)
    replay = gateway.verify_investigation_result_with_evidence(_session(0), result)

    assert isinstance(first, VerifiedInvestigationResult)
    assert first.result is result
    assert isinstance(first.evidence, VerifiedInvestigationEvidence)
    assert first.evidence.to_payload() == replay.evidence.to_payload()
    evidence_payload = first.evidence.to_payload()
    assert evidence_payload["task_id"] == TASK_ID
    assert evidence_payload["batch_id"] == BATCH_ID
    assert evidence_payload["combination_id"] == "combination-0"
    assert evidence_payload["codex_thread_id"] == "thread-0"
    assert evidence_payload["scope_id"] == "scope-0"
    assert evidence_payload["turn_id"] == "turn-0"
    assert evidence_payload["jobs"][0]["terminal_status"] == "completed"
    assert evidence_payload["operations"][0]["terminal_status"] == "completed"
    assert evidence_payload["artifacts"][0]["artifact_id"] == ARTIFACT_ID
    encoded = json.dumps(evidence_payload, ensure_ascii=False, sort_keys=True)
    assert TOKEN not in encoded
    assert str(tmp_path) not in encoded
    assert "workspace-0" not in encoded
    assert "transcript" not in encoded
    gateway.close()


def test_verified_result_api_preserves_each_artifact_operation_binding(
    tmp_path: Path,
) -> None:
    """多 operation 结果的 Working Memory evidence 必须逐 artifact 保留归属。"""

    first_artifact = ARTIFACT_ID
    second_artifact = "art_fedcba9876543210fedcba9876543210"
    first_operation = "operation-first"
    second_operation = "operation-second"

    class MultiOperationSession(RecordingSession):
        def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            response = super().call_tool(name, arguments)
            if name != "verify_execution_references":
                return response
            jobs, operations, artifacts = (
                arguments["request"]["jobs"],
                arguments["request"]["operations"],
                arguments["request"]["artifacts"],
            )
            for item, job_id in zip(response["items"][: len(jobs)], jobs):
                item["submission_operation_id"] = _submission_operation_id(job_id)
            for item, artifact_id, operation_id in zip(
                response["items"][len(jobs) + len(operations) :],
                artifacts,
                (first_operation, second_operation),
            ):
                assert item["artifact_id"] == artifact_id
                item["producer_job_id"] = jobs[0 if operation_id == first_operation else 1]
                item["producer_submission_operation_id"] = _submission_operation_id(
                    item["producer_job_id"]
                )
            return response

    gateway = _gateway(tmp_path, lambda: MultiOperationSession())
    verified = gateway.verify_investigation_result_with_evidence(
        _session(0),
        _result(
            0,
            jobs=(_job_id(1), _job_id(2)),
            operations=(first_operation, second_operation),
            artifacts=(first_artifact, second_artifact),
        ),
    )

    artifacts = verified.evidence.to_payload()["artifacts"]
    assert [
        (
            item["artifact_id"],
            item["producer_job_id"],
            item["producer_submission_operation_id"],
        )
        for item in artifacts
    ] == [
        (first_artifact, _job_id(1), _submission_operation_id(_job_id(1))),
        (second_artifact, _job_id(2), _submission_operation_id(_job_id(2))),
    ]
    gateway.close()


def test_verified_result_api_accepts_multiple_artifacts_for_one_operation(
    tmp_path: Path,
) -> None:
    """多 operation 结果允许同一 operation 产生多个已明确归属的 artifact。"""

    artifacts = (
        ARTIFACT_ID,
        "art_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "art_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    )
    operations = ("operation-first", "operation-second")

    class MultiArtifactSession(RecordingSession):
        def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            response = super().call_tool(name, arguments)
            if name != "verify_execution_references":
                return response
            jobs = arguments["request"]["jobs"]
            operation_refs = arguments["request"]["operations"]
            for item, job_id in zip(response["items"][: len(jobs)], jobs):
                item["submission_operation_id"] = _submission_operation_id(job_id)
            artifact_items = response["items"][len(jobs) + len(operation_refs) :]
            for item, job_id in zip(
                artifact_items,
                (jobs[0], jobs[0], jobs[1]),
            ):
                item["producer_job_id"] = job_id
                item["producer_submission_operation_id"] = _submission_operation_id(job_id)
            return response

    gateway = _gateway(tmp_path, lambda: MultiArtifactSession())
    verified = gateway.verify_investigation_result_with_evidence(
        _session(0),
        _result(
            0,
            jobs=(_job_id(1), _job_id(2)),
            operations=operations,
            artifacts=artifacts,
        ),
    )

    evidence_artifacts = verified.evidence.to_payload()["artifacts"]
    assert [
        (item["producer_job_id"], item["producer_submission_operation_id"])
        for item in evidence_artifacts
    ] == [
        (_job_id(1), _submission_operation_id(_job_id(1))),
        (_job_id(1), _submission_operation_id(_job_id(1))),
        (_job_id(2), _submission_operation_id(_job_id(2))),
    ]
    gateway.close()


@pytest.mark.parametrize(
    "mode", ("missing", "unknown", "mismatch", "noncanonical", "extra")
)
def test_verified_result_api_rejects_artifact_operation_protocol_drift(
    tmp_path: Path,
    mode: str,
) -> None:
    """artifact operation 缺失、未知、错配或额外字段都必须按协议拒绝。"""

    class DriftSession(RecordingSession):
        def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            response = super().call_tool(name, arguments)
            if name != "verify_execution_references":
                return response
            jobs = arguments["request"]["jobs"]
            for item, job_id in zip(response["items"][: len(jobs)], jobs):
                item["submission_operation_id"] = _submission_operation_id(job_id)
            artifacts = response["items"][-2:]
            if mode == "missing":
                artifacts[0].pop("producer_job_id")
                artifacts[0].pop("producer_submission_operation_id")
                return response
            if mode == "unknown":
                artifacts[0]["producer_job_id"] = _job_id(99)
                artifacts[0]["producer_submission_operation_id"] = _submission_operation_id(
                    _job_id(99)
                )
            elif mode == "mismatch":
                artifacts[0]["producer_submission_operation_id"] = "submission-wrong"
            elif mode == "noncanonical":
                noncanonical_operation_id = "scope-submission-not-canonical"
                response["items"][0][
                    "submission_operation_id"
                ] = noncanonical_operation_id
                response["scope_job_inventory"][0][
                    "submission_operation_id"
                ] = noncanonical_operation_id
                artifacts[0][
                    "producer_submission_operation_id"
                ] = noncanonical_operation_id
            else:
                artifacts[1]["extra"] = True
            return response

    gateway = _gateway(tmp_path / mode, lambda: DriftSession())
    with pytest.raises(InvestigationMiniChemGatewayError):
        gateway.verify_investigation_result_with_evidence(
            _session(0),
            _result(
                0,
                jobs=(_job_id(1), _job_id(2)),
                operations=("operation-first", "operation-second"),
                artifacts=(ARTIFACT_ID, "art_fedcba9876543210fedcba9876543210"),
            ),
        )
    gateway.close()


def test_verified_result_api_keeps_submission_operation_outside_typed_operations(
    tmp_path: Path,
) -> None:
    """异步提交 operation 不在科学 operation_refs 时仍保留独立来源链。"""

    gateway = _gateway(tmp_path, lambda: RecordingSession())
    result = _result(
        0,
        jobs=(_job_id(1),),
        operations=("operation-1",),
        artifacts=(ARTIFACT_ID,),
    )
    verified = gateway.verify_investigation_result_with_evidence(_session(0), result)
    payload = verified.evidence.to_payload()
    assert payload["reference_binding_version"] == "producer-job-v2"
    assert payload["operations"][0]["operation_id"] == "operation-1"
    assert payload["jobs"][0]["submission_operation_id"] != "operation-1"
    assert payload["artifacts"][0]["producer_job_id"] == _job_id(1)
    assert payload["artifacts"][0]["producer_submission_operation_id"] == (
        _submission_operation_id(_job_id(1))
    )
    gateway.close()


def test_verified_result_api_accepts_job_only_result_without_typed_operation(
    tmp_path: Path,
) -> None:
    """只有 programmable/native job 时，submission identity 不能伪装成 typed operation。"""

    job_id = _job_id(3)
    gateway = _gateway(tmp_path, lambda: RecordingSession())
    result = _result(
        0,
        jobs=(job_id,),
        artifacts=(ARTIFACT_ID,),
    )

    verified = gateway.verify_investigation_result_with_evidence(_session(0), result)

    payload = verified.evidence.to_payload()
    assert payload["operations"] == []
    assert payload["jobs"] == [
        {
            "job_id": job_id,
            "terminal_status": "completed",
            "receipt_digest": "a" * 64,
            "submission_operation_id": _submission_operation_id(job_id),
        }
    ]
    assert payload["artifacts"][0]["producer_job_id"] == job_id
    assert payload["artifacts"][0]["producer_submission_operation_id"] == (
        _submission_operation_id(job_id)
    )
    gateway.close()


def test_verified_result_api_accepts_native_job_without_artifact_or_submission_operation(
    tmp_path: Path,
) -> None:
    """native job 只有自身终态来源时，也不应伪造 typed 或 submission operation。"""

    class NativeJobSession(RecordingSession):
        def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            response = super().call_tool(name, arguments)
            if name == "verify_execution_references":
                response["items"][0].pop("submission_operation_id")
                response["scope_job_inventory"][0].pop("submission_operation_id")
            return response

    job_id = _job_id(4)
    gateway = _gateway(tmp_path, lambda: NativeJobSession())
    result = _result(0, jobs=(job_id,))

    verified = gateway.verify_investigation_result_with_evidence(_session(0), result)

    payload = verified.evidence.to_payload()
    assert payload["operations"] == []
    assert payload["artifacts"] == []
    assert payload["jobs"] == [
        {
            "job_id": job_id,
            "terminal_status": "completed",
            "receipt_digest": "a" * 64,
        }
    ]
    gateway.close()


def test_verified_result_api_keeps_typed_operation_inventory_strict(
    tmp_path: Path,
) -> None:
    """Result 声明 typed operation 时，缺失的完整 operation inventory 仍拒绝。"""

    class MissingOperationInventorySession(RecordingSession):
        def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            response = super().call_tool(name, arguments)
            if name == "verify_execution_references":
                response["scope_operation_inventory"] = []
            return response

    job_id = _job_id(5)
    gateway = _gateway(tmp_path, lambda: MissingOperationInventorySession())
    result = _result(
        0,
        jobs=(job_id,),
        operations=("typed-operation",),
        artifacts=(ARTIFACT_ID,),
    )

    with pytest.raises(InvestigationMiniChemProtocolError, match="operation inventory"):
        gateway.verify_investigation_result_with_evidence(_session(0), result)
    gateway.close()


def test_realistic_multi_job_artifact_producer_chain_is_not_typed_operation_membership(
    tmp_path: Path,
) -> None:
    """24 个 job、13 个 typed operation 和 14 个 artifact 的真实形状可通过。

    其中每个 artifact 的 submission operation 都不属于 13 个科学 operation，
    但都能通过 artifact→producer job→submission operation 的严格链路核对。
    """

    jobs = tuple(_short_job_id(index) for index in range(24))
    operations = tuple(f"scientific-operation-{index}" for index in range(13))
    artifacts = tuple(f"art_{index:032x}" for index in range(14))
    gateway = _gateway(tmp_path, lambda: RecordingSession())
    verified = gateway.verify_investigation_result_with_evidence(
        _session(0),
        _result(0, jobs=jobs, operations=operations, artifacts=artifacts),
    )
    payload = verified.evidence.to_payload()
    assert len(payload["jobs"]) == 24
    assert len(payload["operations"]) == 13
    assert len(payload["artifacts"]) == 14
    assert all(
        item["producer_submission_operation_id"] not in operations
        for item in payload["artifacts"]
    )
    gateway.close()


def test_realistic_artifact_producer_chain_mismatch_fails_closed(tmp_path: Path) -> None:
    """producer job 存在但提交 operation 错配时不能接受 verifier 的 all_valid。"""

    jobs = tuple(_short_job_id(index) for index in range(24))
    operations = tuple(f"scientific-operation-{index}" for index in range(13))
    artifacts = tuple(f"art_{index:032x}" for index in range(14))

    class MismatchedProducerSession(RecordingSession):
        def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            response = super().call_tool(name, arguments)
            if name == "verify_execution_references":
                artifact_items = response["items"][-len(artifacts) :]
                artifact_items[0]["producer_submission_operation_id"] = (
                    _submission_operation_id(jobs[1])
                )
            return response

    gateway = _gateway(tmp_path, lambda: MismatchedProducerSession())
    with pytest.raises(InvestigationMiniChemProtocolError, match="producer|一致"):
        gateway.verify_investigation_result_with_evidence(
            _session(0),
            _result(0, jobs=jobs, operations=operations, artifacts=artifacts),
        )
    gateway.close()


def test_schema1_single_job_legacy_artifact_binding_has_controlled_compatibility(
    tmp_path: Path,
) -> None:
    """旧 schema1 只有一个 job 且 operation 完全一致时才可补出 producer job。"""

    class LegacySingleJobSession(RecordingSession):
        def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            response = super().call_tool(name, arguments)
            if name != "verify_execution_references":
                return response
            response["schema_version"] = 1
            for inventory_item in response["scope_job_inventory"]:
                inventory_item.pop("submission_operation_id", None)
            jobs = arguments["request"]["jobs"]
            operations = arguments["request"]["operations"]
            items = response["items"]
            for item, job_id in zip(items[: len(jobs)], jobs):
                item["operation_id"] = item.pop("submission_operation_id")
            artifacts = items[len(jobs) + len(operations) :]
            for item in artifacts:
                item.pop("producer_job_id")
                item.pop("producer_submission_operation_id")
            return response

    gateway = _gateway(tmp_path, lambda: LegacySingleJobSession())
    verified = gateway.verify_investigation_result_with_evidence(
        _session(0),
        _result(0, jobs=(_short_job_id(1),), operations=("scientific-operation",), artifacts=(ARTIFACT_ID,)),
    )
    artifact = verified.evidence.to_payload()["artifacts"][0]
    assert artifact["producer_job_id"] == _short_job_id(1)
    assert artifact["producer_submission_operation_id"] == _submission_operation_id(
        _short_job_id(1)
    )
    gateway.close()


def test_schema1_multi_job_legacy_artifact_binding_is_not_guessed(tmp_path: Path) -> None:
    """旧 schema1 多 job artifact 没有 producer job 时必须拒绝猜测。"""

    class AmbiguousLegacySession(RecordingSession):
        def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            response = super().call_tool(name, arguments)
            if name != "verify_execution_references":
                return response
            response["schema_version"] = 1
            for inventory_item in response["scope_job_inventory"]:
                inventory_item.pop("submission_operation_id", None)
            jobs = arguments["request"]["jobs"]
            operations = arguments["request"]["operations"]
            items = response["items"]
            for item in items[: len(jobs)]:
                item["operation_id"] = item.pop("submission_operation_id")
            artifacts = items[len(jobs) + len(operations) :]
            for item in artifacts:
                item["operation_id"] = "submission-unknown"
                item.pop("producer_job_id")
                item.pop("producer_submission_operation_id")
            return response

    gateway = _gateway(tmp_path, lambda: AmbiguousLegacySession())
    with pytest.raises(InvestigationMiniChemProtocolError, match="唯一|producer"):
        gateway.verify_investigation_result_with_evidence(
            _session(0),
            _result(
                0,
                jobs=(_short_job_id(1), _short_job_id(2)),
                operations=("scientific-operation",),
                artifacts=(ARTIFACT_ID,),
            ),
        )
    gateway.close()


def test_schema1_multi_job_artifact_without_operation_is_not_guessed(
    tmp_path: Path,
) -> None:
    """旧 schema1 artifact 连 operation_id 也缺失时，多 job 仍不能猜来源。"""

    class MissingOperationLegacySession(RecordingSession):
        def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            response = super().call_tool(name, arguments)
            if name != "verify_execution_references":
                return response
            response["schema_version"] = 1
            for inventory_item in response["scope_job_inventory"]:
                inventory_item.pop("submission_operation_id", None)
            request = arguments["request"]
            jobs = request["jobs"]
            operations = request["operations"]
            items = response["items"]
            for item in items[: len(jobs)]:
                item["operation_id"] = item.pop("submission_operation_id")
            for item in items[len(jobs) + len(operations) :]:
                item.pop("producer_job_id")
                item.pop("producer_submission_operation_id")
            return response

    jobs = (_short_job_id(1), _short_job_id(2))
    gateway = _gateway(tmp_path, lambda: MissingOperationLegacySession())
    with pytest.raises(InvestigationMiniChemProtocolError, match="唯一|producer"):
        gateway.verify_investigation_result_with_evidence(
            _session(0),
            _result(
                0,
                jobs=jobs,
                operations=("scientific-operation",),
                artifacts=(ARTIFACT_ID,),
            ),
        )
    gateway.close()


def test_schema1_legacy_artifact_can_match_one_job_in_a_multi_job_response(
    tmp_path: Path,
) -> None:
    """旧 schema1 多 job 只在 operation 恰好唯一匹配时转换来源链。"""

    class LegacyMultiJobSession(RecordingSession):
        def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            response = super().call_tool(name, arguments)
            if name != "verify_execution_references":
                return response
            response["schema_version"] = 1
            for inventory_item in response["scope_job_inventory"]:
                inventory_item.pop("submission_operation_id", None)
            request = arguments["request"]
            jobs = request["jobs"]
            operations = request["operations"]
            items = response["items"]
            for item in items[: len(jobs)]:
                item["operation_id"] = item.pop("submission_operation_id")
            for item in items[len(jobs) + len(operations) :]:
                item["operation_id"] = _submission_operation_id(jobs[1])
                item.pop("producer_job_id")
                item.pop("producer_submission_operation_id")
            return response

    jobs = (_short_job_id(1), _short_job_id(2))
    gateway = _gateway(tmp_path, lambda: LegacyMultiJobSession())
    verified = gateway.verify_investigation_result_with_evidence(
        _session(0),
        _result(
            0,
            jobs=jobs,
            operations=("scientific-operation",),
            artifacts=(ARTIFACT_ID,),
        ),
    )
    artifact = verified.evidence.to_payload()["artifacts"][0]
    assert artifact["producer_job_id"] == jobs[1]
    gateway.close()


def test_verify_scope_execution_inventory_fences_created_and_repair_scopes(
    tmp_path: Path,
) -> None:
    """side-effect fence 只允许初始空清单或返修前的精确引用清单。"""

    session_factory_calls: list[RecordingSession] = []

    def factory() -> RecordingSession:
        session = RecordingSession()
        session_factory_calls.append(session)
        return session

    gateway = _gateway(tmp_path, factory)
    created = InvestigationSession(
        task_id=TASK_ID,
        batch_id=BATCH_ID,
        combination_id="combination-0",
        codex_thread_id=None,
        workspace_ref="workspace-0",
        scope_id="scope-0",
        result_version=None,
        phase="created",
    )
    initial_inventory = gateway.verify_scope_execution_inventory(created)
    assert initial_inventory["scope_job_inventory"] == []
    assert initial_inventory["scope_operation_inventory"] == []
    assert initial_inventory["has_active_jobs"] is False
    assert initial_inventory["has_active_operations"] is False

    repair = _session(0, phase="repair_pending")
    expected_jobs = (_job_id(7),)
    expected_operations = ("operation-7",)
    repair_inventory = gateway.verify_scope_execution_inventory(
        repair,
        expected_job_refs=expected_jobs,
        expected_operation_refs=expected_operations,
    )
    assert repair_inventory["scope_job_inventory"] == [
        {
            "job_id": _job_id(7),
            "status": "success",
            "submission_operation_id": _submission_operation_id(_job_id(7)),
        }
    ]
    assert repair_inventory["scope_operation_inventory"] == [
        {"operation_id": "operation-7", "status": "completed"}
    ]
    verify_requests = [
        request
        for name, request in session_factory_calls[0].calls
        if name == "verify_execution_references"
    ]
    assert verify_requests == [
        {"jobs": [], "operations": [], "artifacts": []},
        {"jobs": [_job_id(7)], "operations": ["operation-7"], "artifacts": []},
    ]
    gateway.close()


def test_verify_scope_execution_inventory_rejects_legacy_active_and_missing_state(
    tmp_path: Path,
) -> None:
    """遗留 job/operation、活动状态、损坏状态和缺字段都阻断 side-effect fence。"""

    repair = _session(0, phase="repair_pending")
    expected_jobs = (_job_id(7),)
    expected_operations = ("operation-7",)

    class InventoryDrift(RecordingSession):
        def __init__(self, mode: str) -> None:
            super().__init__()
            self.mode = mode

        def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            response = super().call_tool(name, arguments)
            if name != "verify_execution_references":
                return response
            if self.mode == "extra-job":
                response["scope_job_inventory"].append(
                    {"job_id": _job_id(99), "status": "success"}
                )
            elif self.mode == "foreign-job":
                response["scope_job_inventory"].append(
                    {"job_id": "job-foreign", "status": "success"}
                )
            elif self.mode == "active-extra-job":
                response["scope_job_inventory"].append(
                    {"job_id": _job_id(99), "status": "running"}
                )
            elif self.mode == "extra-operation":
                response["scope_operation_inventory"].append(
                    {"operation_id": "operation-extra", "status": "completed"}
                )
            elif self.mode == "active":
                response["has_active_jobs"] = True
            elif self.mode == "active-operation":
                response["has_active_operations"] = True
            elif self.mode == "corrupt":
                response["scope_operation_inventory"][0]["status"] = "running"
            elif self.mode == "missing-operation-inventory":
                response.pop("scope_operation_inventory")
            elif self.mode == "missing-active-operation":
                response.pop("has_active_operations")
            elif self.mode == "missing-operation-complete":
                response.pop("operation_inventory_complete")
            return response

    for mode in (
        "extra-job",
        "foreign-job",
        "active-extra-job",
        "extra-operation",
        "active",
        "active-operation",
        "corrupt",
        "missing-operation-inventory",
        "missing-active-operation",
        "missing-operation-complete",
    ):
        gateway = _gateway(
            tmp_path / mode,
            lambda mode=mode: InventoryDrift(mode),
        )
        with pytest.raises(InvestigationMiniChemGatewayError):
            gateway.verify_scope_execution_inventory(
                repair,
                expected_job_refs=expected_jobs,
                expected_operation_refs=expected_operations,
            )
        gateway.close()

    initial_gateway = _gateway(tmp_path / "initial-nonempty", lambda: RecordingSession())
    with pytest.raises(InvestigationMiniChemScopeError):
        initial_gateway.verify_scope_execution_inventory(
            InvestigationSession(
                task_id=TASK_ID,
                batch_id=BATCH_ID,
                combination_id="combination-0",
                codex_thread_id=None,
                workspace_ref="workspace-0",
                scope_id="scope-0",
                result_version=None,
                phase="created",
            ),
            expected_job_refs=expected_jobs,
        )
    initial_gateway.close()


def test_scope_identity_and_unavailable_reference_fail_closed(tmp_path: Path) -> None:
    """跨 scope、unavailable、非终态和顺序漂移不能产生已核验结果。"""

    session = RecordingSession()
    gateway = _gateway(tmp_path, lambda: session)
    result = _result(0, artifacts=(ARTIFACT_ID,))
    wrong = InvestigationSession(
        task_id=TASK_ID,
        batch_id=BATCH_ID,
        combination_id="combination-0",
        codex_thread_id="thread-0",
        workspace_ref="workspace-0",
        scope_id="scope-1",
        result_version="v1",
        phase="ready_for_batch_validation",
    )
    with pytest.raises(InvestigationMiniChemScopeError):
        gateway.result_verifier(wrong, result)

    class Unavailable(RecordingSession):
        def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            return {
                "schema_version": 1,
                "status": "success",
                "all_valid": False,
                "items": [{"kind": "artifact", "status": "unavailable"}],
                "scope_job_inventory": [],
                "has_active_jobs": False,
                "inventory_complete": True,
                "scope_operation_inventory": [],
                "has_active_operations": False,
                "operation_inventory_complete": True,
            }

    unavailable_gateway = _gateway(tmp_path / "unavailable", lambda: Unavailable())
    with pytest.raises(InvestigationMiniChemReferenceRejected) as captured:
        unavailable_gateway.result_verifier(_session(0), result)
    assert captured.value.invalid_references == (
        {
            "kind": "artifact",
            "artifact_id": ARTIFACT_ID,
            "reason_code": "reference_unavailable",
        },
    )

    class NonTerminal(RecordingSession):
        def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            response = super().call_tool(name, arguments)
            if name == "verify_execution_references":
                response["items"][0]["terminal_status"] = "running"
            return response

    non_terminal_gateway = _gateway(tmp_path / "non-terminal", lambda: NonTerminal())
    with pytest.raises(InvestigationMiniChemProtocolError):
        non_terminal_gateway.result_verifier(_session(0), _result(0, jobs=(_job_id(3),)))


def test_reference_rejection_preserves_expected_id_for_same_thread_repair(
    tmp_path: Path,
) -> None:
    """服务端省略拒绝项 ID 时，网关仍按冻结请求顺序返回精确不透明引用。"""

    class RejectedOperation(RecordingSession):
        def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            response = super().call_tool(name, arguments)
            if name == "verify_execution_references":
                response["all_valid"] = False
                response["items"][0] = {
                    "kind": "operation",
                    "status": "invalid",
                    "reason_code": "operation_not_completed",
                }
                response["scope_operation_inventory"][0]["status"] = "running"
                response["has_active_operations"] = True
            return response

    gateway = _gateway(tmp_path, lambda: RejectedOperation())
    operation_id = "operation-running"

    with pytest.raises(InvestigationMiniChemReferenceRejected) as captured:
        gateway.result_verifier(
            _session(0),
            _result(0, operations=(operation_id,)),
        )

    assert captured.value.invalid_references == (
        {
            "kind": "operation",
            "operation_id": operation_id,
            "reason_code": "operation_not_completed",
        },
    )


def test_result_verification_allows_unreferenced_running_scope_operation(
    tmp_path: Path,
) -> None:
    """结果证据不必声明同 scope 中未参与结论的探索 operation。"""

    operation_id = "unreferenced-exploratory-operation"

    class HistoricalOperation(RecordingSession):
        def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            response = super().call_tool(name, arguments)
            if name == "verify_execution_references":
                response["scope_operation_inventory"] = [
                    {"operation_id": operation_id, "status": "running"}
                ]
                response["has_active_operations"] = True
            return response

    gateway = _gateway(tmp_path, lambda: HistoricalOperation())
    result = _result(
        0,
        jobs=(_job_id(1),),
        artifacts=(ARTIFACT_ID,),
    )
    assert gateway.verify_investigation_result(_session(0), result) is result
    gateway.close()


def test_duplicate_reference_and_item_order_drift_fail_closed(tmp_path: Path) -> None:
    """重复引用、verifier 项顺序漂移和 inventory 重复项都必须拒绝。"""

    gateway = _gateway(tmp_path / "duplicates", lambda: RecordingSession())
    valid_result = _result(0)

    def unchecked_result(**updates: Any) -> InvestigationResult:
        """绕过领域模型的早期校验，单独覆盖网关的重复引用防线。"""

        values = valid_result.model_dump()
        values.update(updates)
        return InvestigationResult.model_construct(**values)

    duplicate_results = (
        unchecked_result(job_refs=(_job_id(1), _job_id(1))),
        unchecked_result(operation_refs=("operation-1", "operation-1")),
        unchecked_result(artifact_refs=(ARTIFACT_ID, ARTIFACT_ID)),
    )
    for result in duplicate_results:
        with pytest.raises(InvestigationMiniChemProtocolError):
            gateway.result_verifier(_session(0), result)
    gateway.close()

    class Reordered(RecordingSession):
        def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            response = super().call_tool(name, arguments)
            if name == "verify_execution_references":
                response["items"] = list(reversed(response["items"]))
            return response

    reordered_gateway = _gateway(tmp_path / "reordered", lambda: Reordered())
    with pytest.raises(InvestigationMiniChemProtocolError):
        reordered_gateway.result_verifier(
            _session(0),
            _result(0, jobs=(_job_id(1),), operations=("operation-1",), artifacts=(ARTIFACT_ID,)),
        )
    reordered_gateway.close()

    class InventoryExtra(RecordingSession):
        def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            response = super().call_tool(name, arguments)
            if name == "verify_execution_references":
                response["scope_job_inventory"].append(
                    {
                        "job_id": _job_id(1),
                        "status": "success",
                        "submission_operation_id": _submission_operation_id(_job_id(1)),
                    }
                )
            return response

    inventory_gateway = _gateway(tmp_path / "inventory", lambda: InventoryExtra())
    with pytest.raises(InvestigationMiniChemProtocolError):
        inventory_gateway.result_verifier(
            _session(0), _result(0, jobs=(_job_id(1),))
        )
    inventory_gateway.close()


def test_result_verifier_accepts_case3_historical_terminal_inventory(
    tmp_path: Path,
) -> None:
    """当前 Result 的 90 个 job 外带 11 个历史终态 job 时仍能生成证据。"""

    jobs = tuple(_short_job_id(index) for index in range(1, 91))
    operations = tuple(f"scientific-operation-{index}" for index in range(15))
    artifacts = tuple(f"art_{index:032x}" for index in range(1, 36))
    historical_jobs = tuple(
        {
            "job_id": _short_job_id(index),
            "status": "failed" if index % 2 else "success",
            "submission_operation_id": _submission_operation_id(_short_job_id(index)),
        }
        for index in range(101, 112)
    )

    class HistoricalInventorySession(RecordingSession):
        def __init__(self) -> None:
            super().__init__()
            self.final_inventory: tuple[dict[str, Any], ...] = ()

        def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            response = super().call_tool(name, arguments)
            if name == "verify_execution_references":
                assert arguments["request"]["jobs"] == list(jobs)
                assert len(response["scope_job_inventory"]) == 90
                response["scope_job_inventory"].extend(historical_jobs)
                self.final_inventory = tuple(response["scope_job_inventory"])
            return response

    sessions: list[HistoricalInventorySession] = []

    def factory() -> HistoricalInventorySession:
        session = HistoricalInventorySession()
        sessions.append(session)
        return session

    gateway = _gateway(tmp_path, factory)
    result = _result(0, jobs=jobs, operations=operations, artifacts=artifacts)
    references = gateway._verify_result_references(_session(0), result)
    assert len(references.scope_job_inventory) == 101
    assert {
        item["job_id"] for item in references.scope_job_inventory[-11:]
    } == {item["job_id"] for item in historical_jobs}
    verified = gateway.verify_investigation_result_with_evidence(_session(0), result)
    assert verified.result is result
    assert len(sessions) == 1
    assert len(sessions[0].final_inventory) == 101
    assert set(item["job_id"] for item in sessions[0].final_inventory[-11:]) == {
        item["job_id"] for item in historical_jobs
    }
    assert len(verified.evidence.to_payload()["jobs"]) == 90
    gateway.close()

    baseline_gateway = _gateway(tmp_path / "baseline", lambda: RecordingSession())
    baseline = baseline_gateway.verify_investigation_result_with_evidence(
        _session(0), result
    )
    assert len(baseline.evidence.to_payload()["jobs"]) == 90
    assert (
        verified.evidence.to_payload()["verification_digest"]
        != baseline.evidence.to_payload()["verification_digest"]
    )
    baseline_gateway.close()


def test_async_session_factory_and_read_snapshot_are_idempotent(tmp_path: Path) -> None:
    """异步 session、分页重构、source cache 权限和 snapshot 重放均可工作。"""

    sessions: list[RecordingSession] = []

    class AsyncSession(RecordingSession):
        async def initialize(self) -> dict[str, Any]:
            await asyncio.sleep(0)
            return super().initialize()

        async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            await asyncio.sleep(0)
            return super().call_tool(name, arguments)

        async def close(self) -> None:
            await asyncio.sleep(0)

    async def factory(endpoint: str, headers: dict[str, str]) -> AsyncSession:
        assert endpoint.endswith("/mcp")
        assert headers["X-MiniChem-Capability"].startswith(TOKEN)
        session = AsyncSession()
        sessions.append(session)
        return session

    results = [
        _result(
            0,
            jobs=(_short_job_id(1),),
            operations=("snapshot-operation",),
            artifacts=(ARTIFACT_ID,),
        ),
        *tuple(_result(index) for index in range(1, 5)),
    ]
    state = _state(tuple(results))
    gateway = _gateway(tmp_path, factory, page_size=4)
    vector = tuple((f"combination-{index}", "v1") for index in range(5))
    first = gateway.evidence_snapshot_builder(state, tuple(results), vector)
    replay = gateway.evidence_snapshot_builder(state, tuple(results), tuple(reversed(vector)))
    assert first == replay
    assert len(first["result_pairs"]) == 5
    assert first["result_pairs"][0]["artifacts"]
    encoded = json.dumps(first, ensure_ascii=False)
    assert TOKEN not in encoded
    assert str(tmp_path) not in encoded
    verify_calls = [
        call
        for session in sessions
        for call in session.calls
        if call[0] == "verify_execution_references"
    ]
    assert len(verify_calls) == 10
    assert all(set(call[1]) == {"jobs", "operations", "artifacts"} for call in verify_calls)
    assert any(call[1]["artifacts"] == [] for call in verify_calls)
    cache = tmp_path / "source-cache"
    assert stat.S_IMODE(cache.stat().st_mode) == 0o700
    files = [path for path in cache.iterdir() if path.is_file()]
    assert files and all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in files)
    read_calls = [call for session in sessions for call in session.calls if call[0] == "read_scientific_artifact_text"]
    assert [call[1]["offset"] for call in read_calls] == [0, 4, 8, 12, 0, 4, 8, 12]
    gateway.close()


def test_idle_async_session_reconnects_for_next_readonly_verifier(
    tmp_path: Path,
) -> None:
    """只读 session 长时间闲置失效后，下一次 verifier 应建立新连接。"""

    sessions: list[RecordingSession] = []

    class IdleDisconnectedSession(RecordingSession):
        """第一条只读调用返回后，模拟服务端断开闲置连接。"""

        def __init__(self, *, stale_after_first_call: bool) -> None:
            super().__init__()
            self.stale_after_first_call = stale_after_first_call

        async def initialize(self) -> dict[str, Any]:
            await asyncio.sleep(0)
            return super().initialize()

        async def call_tool(
            self,
            name: str,
            arguments: dict[str, Any],
        ) -> dict[str, Any]:
            if self.stale_after_first_call and self.calls:
                raise RuntimeError("闲置 MCP session 已断开")
            await asyncio.sleep(0)
            return super().call_tool(name, arguments)

        async def close(self) -> None:
            await asyncio.sleep(0)

    async def factory(endpoint: str, headers: dict[str, str]) -> IdleDisconnectedSession:
        assert endpoint.endswith("/mcp")
        assert headers["X-MiniChem-Capability"] == TOKEN + "0"
        session = IdleDisconnectedSession(stale_after_first_call=not sessions)
        sessions.append(session)
        return session

    gateway = _gateway(tmp_path, factory)
    gateway.preflight_scope_readiness(_session(0))
    result = _result(0)

    assert gateway.result_verifier(_session(0), result) is result
    assert len(sessions) == 2
    assert [name for name, _ in sessions[0].calls] == ["get_execution_resources"]
    assert [name for name, _ in sessions[1].calls] == [
        "verify_execution_references"
    ]
    gateway.close()


def test_unresponsive_idle_async_session_reconnects_after_call_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """只读 session 不再消费命令时，超时后应换连接并只重试一次。"""

    monkeypatch.setattr(gateway_module, "_SESSION_CALL_TIMEOUT_SECONDS", 0.01)
    sessions: list[RecordingSession] = []

    class UnresponsiveIdleSession(RecordingSession):
        """第一条调用完成后模拟事件流不再消费后续命令。"""

        def __init__(self, *, unresponsive: bool) -> None:
            super().__init__()
            self.unresponsive = unresponsive

        async def initialize(self) -> dict[str, Any]:
            return super().initialize()

        async def call_tool(
            self,
            name: str,
            arguments: dict[str, Any],
        ) -> dict[str, Any]:
            if self.unresponsive and self.calls:
                await asyncio.Future()
            return super().call_tool(name, arguments)

        async def close(self) -> None:
            return None

    async def factory() -> UnresponsiveIdleSession:
        session = UnresponsiveIdleSession(unresponsive=not sessions)
        sessions.append(session)
        return session

    gateway = _gateway(tmp_path, factory)
    gateway.preflight_scope_readiness(_session(0))
    result = _result(0)

    assert gateway.result_verifier(_session(0), result) is result
    assert len(sessions) == 2
    assert [name for name, _ in sessions[0].calls] == ["get_execution_resources"]
    assert [name for name, _ in sessions[1].calls] == [
        "verify_execution_references"
    ]
    gateway.close()


def test_non_readonly_tool_cannot_trigger_session_reconnect() -> None:
    """非只读工具在建立 session 前就拒绝，不能进入自动重试。"""

    factory_calls = 0

    def factory() -> RecordingSession:
        nonlocal factory_calls
        factory_calls += 1
        return RecordingSession()

    scope_session = gateway_module._ScopeSession(
        "http://127.0.0.1:9010/mcp",
        _binding(0),
        factory,
    )
    with pytest.raises(InvestigationMiniChemProtocolError):
        scope_session.call_tool("submit_execution_job", {"request": {}})
    assert factory_calls == 0


def test_protocol_error_does_not_trigger_session_reconnect(tmp_path: Path) -> None:
    """只读响应的协议错误仍然直接 fail-closed，不重放请求。"""

    sessions: list[RecordingSession] = []

    class ProtocolErrorSession(RecordingSession):
        async def initialize(self) -> dict[str, Any]:
            return super().initialize()

        async def call_tool(
            self,
            name: str,
            arguments: dict[str, Any],
        ) -> dict[str, Any]:
            if name == "verify_execution_references":
                return {"isError": True}
            return super().call_tool(name, arguments)

        async def close(self) -> None:
            return None

    async def factory() -> ProtocolErrorSession:
        session = ProtocolErrorSession()
        sessions.append(session)
        return session

    gateway = _gateway(tmp_path, factory)
    with pytest.raises(InvestigationMiniChemProtocolError):
        gateway.result_verifier(_session(0), _result(0))
    assert len(sessions) == 1
    gateway.close()


def test_second_readonly_connection_failure_remains_unavailable(
    tmp_path: Path,
) -> None:
    """fresh session 仍不可用时必须失败，不能无限重试或伪造结果。"""

    sessions: list[RecordingSession] = []

    class FailingAfterPreflightSession(RecordingSession):
        def __init__(self, *, fail_after_first_call: bool) -> None:
            super().__init__()
            self.fail_after_first_call = fail_after_first_call

        async def initialize(self) -> dict[str, Any]:
            return super().initialize()

        async def call_tool(
            self,
            name: str,
            arguments: dict[str, Any],
        ) -> dict[str, Any]:
            if self.fail_after_first_call or self.calls:
                raise RuntimeError("MCP session 不可用")
            return super().call_tool(name, arguments)

        async def close(self) -> None:
            return None

    async def factory() -> FailingAfterPreflightSession:
        session = FailingAfterPreflightSession(fail_after_first_call=bool(sessions))
        sessions.append(session)
        return session

    gateway = _gateway(tmp_path, factory)
    gateway.preflight_scope_readiness(_session(0))
    with pytest.raises(InvestigationMiniChemUnavailable):
        gateway.result_verifier(_session(0), _result(0))
    assert len(sessions) == 2
    gateway.close()


def test_protocol_drift_pagination_digest_binary_and_size_are_rejected(tmp_path: Path) -> None:
    """分页缺页、摘要篡改、二进制控制字符和大小上限都 fail-closed。"""

    class Drift(RecordingSession):
        def __init__(self, mode: str) -> None:
            super().__init__()
            self.mode = mode

        def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            response = super().call_tool(name, arguments)
            if name != "read_scientific_artifact_text":
                return response
            if self.mode == "offset":
                response["offset"] = response["offset"] + 1
            elif self.mode == "digest":
                response["sha256"] = "f" * 64
            elif self.mode == "media":
                if arguments["offset"] == 0:
                    response["media_type"] = "text/plain; charset=utf-16"
            elif self.mode == "binary":
                response["content"] = "\x00"
                response["returned_chars"] = 1
                response["next_offset"] = None
                response["total_chars"] = 1
                response["truncated"] = False
            return response

    result = _result(
        0,
        jobs=(_short_job_id(1),),
        operations=("snapshot-operation",),
        artifacts=(ARTIFACT_ID,),
    )
    state = _state((result, *tuple(_result(index) for index in range(1, 5))))
    vector = tuple((f"combination-{index}", "v1") for index in range(5))
    for index, mode in enumerate(("offset", "digest", "media", "binary")):
        root = tmp_path / mode
        gateway = _gateway(root, lambda mode=mode: Drift(mode), page_size=4)
        with pytest.raises(InvestigationMiniChemGatewayError):
            gateway.evidence_snapshot_builder(state, state.results, vector)

    too_small = InvestigationMiniChemGateway(
        "http://127.0.0.1:9010/mcp",
        {_binding(index).scope_id: _binding(index) for index in range(5)},
        evidence_store=ReviewEvidenceStore(tmp_path / "small-store"),
        source_cache_dir=tmp_path / "small-cache",
        session_factory=lambda: RecordingSession(),
        max_artifact_bytes=4,
    )
    with pytest.raises(InvestigationMiniChemGatewayError):
        too_small.evidence_snapshot_builder(state, state.results, vector)


def test_source_cache_rejects_symlinked_directory(tmp_path: Path) -> None:
    """source cache 目录不能通过符号链接跳出调用方指定位置。"""

    real_dir = tmp_path / "real-cache"
    real_dir.mkdir()
    symlink_dir = tmp_path / "cache-link"
    symlink_dir.symlink_to(real_dir, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink|符号链接"):
        InvestigationMiniChemGateway(
            "http://127.0.0.1:9010/mcp",
            {_binding(index).scope_id: _binding(index) for index in range(5)},
            evidence_store=ReviewEvidenceStore(tmp_path / "store"),
            source_cache_dir=symlink_dir,
            session_factory=lambda: RecordingSession(),
        )


def test_capability_is_not_in_repr_or_error_trace(tmp_path: Path) -> None:
    """scope capability 不能通过 repr、错误或公开 manifest 泄露。"""

    secret = "secret-capability-value-abcdefghijklmnopqrstuvwxyz"
    binding = InvestigationScopeBinding(scope_id="scope-0", token=secret)
    gateway = InvestigationMiniChemGateway(
        "http://127.0.0.1:9010/mcp",
        {"scope-0": binding},
        evidence_store=ReviewEvidenceStore(tmp_path / "store"),
        source_cache_dir=tmp_path / "cache",
        session_factory=lambda: (_ for _ in ()).throw(RuntimeError(secret)),
    )
    assert secret not in repr(binding)
    assert secret not in repr(gateway)
    with pytest.raises(InvestigationMiniChemUnavailable) as error:
        gateway.result_verifier(
            InvestigationSession(
                task_id="task",
                batch_id="batch",
                combination_id="combination",
                codex_thread_id="thread",
                workspace_ref="workspace",
                scope_id="scope-0",
                result_version="v1",
                phase="ready_for_batch_validation",
            ),
            InvestigationResult(
                task_id="task",
                batch_id="batch",
                combination_id="combination",
                result_version="v1",
                codex_thread_id="thread",
                scope_id="scope-0",
                turn_id="turn",
                self_judgment="supported",
                narrative="narrative",
                observations=("observation",),
            ),
        )
    assert secret not in str(error.value)


def test_async_transport_error_chain_is_detached_from_capability(tmp_path: Path) -> None:
    """异步 MCP/httpx 异常的 cause/context 不能把 capability 带回调用方。"""

    secret = "async-secret-capability-value-abcdefghijklmnopqrstuvwxyz"

    class FailingAsyncSession:
        async def initialize(self) -> None:
            return None

        async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
            raise RuntimeError(f"transport header leaked: {secret}")

        async def close(self) -> None:
            return None

    async def factory(endpoint: str, headers: dict[str, str]) -> FailingAsyncSession:
        assert headers["X-MiniChem-Capability"] == secret
        return FailingAsyncSession()

    gateway = InvestigationMiniChemGateway(
        "http://127.0.0.1:9010/mcp",
        {
            "scope-0": InvestigationScopeBinding(
                scope_id="scope-0",
                token=secret,
            )
        },
        evidence_store=ReviewEvidenceStore(tmp_path / "store"),
        source_cache_dir=tmp_path / "cache",
        session_factory=factory,
    )
    with pytest.raises(InvestigationMiniChemUnavailable) as error:
        gateway.result_verifier(_session(0), _result(0))

    seen: set[int] = set()
    pending: list[BaseException] = [error.value]
    while pending:
        cursor = pending.pop()
        if id(cursor) in seen:
            continue
        seen.add(id(cursor))
        assert secret not in repr(cursor)
        assert secret not in str(cursor)
        for linked in (cursor.__cause__, cursor.__context__):
            if linked is not None:
                pending.append(linked)
    assert error.value.__cause__ is None
    assert error.value.__context__ is None


def test_endpoint_rejects_userinfo_query_and_fragment(tmp_path: Path) -> None:
    """endpoint 不能用 URL userinfo、query 或 fragment 携带 capability。"""

    common = {
        "scope-0": InvestigationScopeBinding(scope_id="scope-0", token=TOKEN),
    }
    kwargs = {
        "scope_capabilities": common,
        "evidence_store": ReviewEvidenceStore(tmp_path / "store"),
        "source_cache_dir": tmp_path / "cache",
        "session_factory": lambda: RecordingSession(),
    }
    for endpoint in (
        "http://user:password@127.0.0.1:9010/mcp",
        "http://127.0.0.1:9010/mcp?capability=secret",
        "http://127.0.0.1:9010/mcp?",
        "http://127.0.0.1:9010/mcp#capability",
        "http://127.0.0.1:9010/mcp#",
    ):
        with pytest.raises(ValueError, match="endpoint"):
            InvestigationMiniChemGateway(endpoint, **kwargs)


def test_real_streamable_http_transport_parses_content_and_authenticates_scope(
    tmp_path: Path,
) -> None:
    """通过真实 FastMCP/Starlette transport 核对 envelope、content 和 header。"""

    import httpx
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client
    from mcp.server.fastmcp import FastMCP
    from sse_starlette.sse import AppStatus

    observed_requests: list[tuple[str, dict[str, Any]]] = []
    observed_headers: list[dict[bytes, bytes]] = []

    def create_application() -> tuple[Any, Any]:
        """为每个异步 runner 创建独立生命周期的 FastMCP ASGI 应用。"""

        server = FastMCP("investigation-gateway-test")

        @server.tool()
        def verify_execution_references(request: dict[str, Any]) -> dict[str, Any]:
            """测试服务记录 verifier 的真实 request envelope。"""

            assert tuple(request) == ("jobs", "operations", "artifacts")
            observed_requests.append(("verify", dict(request)))
            items: list[dict[str, Any]] = []
            for job_id in request["jobs"]:
                items.append(
                    {
                        "kind": "job",
                        "status": "valid",
                        "job_id": job_id,
                        "submission_operation_id": _submission_operation_id(job_id),
                        "operation_fingerprint": "a" * 64,
                        "terminal_status": "success",
                    }
                )
            for operation_id in request["operations"]:
                items.append(
                    {
                        "kind": "operation",
                        "status": "valid",
                        "operation_id": operation_id,
                        "fingerprint": "b" * 64,
                        "operation_status": "completed",
                    }
                )
            for artifact_id in request["artifacts"]:
                items.append(
                    {
                        "kind": "artifact",
                        "status": "valid",
                        "artifact_id": artifact_id,
                        "sha256": CONTENT_SHA256,
                        "producer_job_id": request["jobs"][0],
                        "producer_submission_operation_id": _submission_operation_id(
                            request["jobs"][0]
                        ),
                    }
                )
            return {
                "schema_version": 2,
                "status": "success",
                "all_valid": True,
                "items": items,
                "scope_job_inventory": [
                    {
                        "job_id": job_id,
                        "status": "success",
                        "submission_operation_id": _submission_operation_id(job_id),
                    }
                    for job_id in request["jobs"]
                ],
                "has_active_jobs": False,
                "inventory_complete": True,
                "scope_operation_inventory": [
                    {"operation_id": operation_id, "status": "completed"}
                    for operation_id in request["operations"]
                ],
                "has_active_operations": False,
                "operation_inventory_complete": True,
            }

        @server.tool()
        def read_scientific_artifact_text(request: dict[str, Any]) -> dict[str, Any]:
            """测试服务记录 reader 的真实分页 request envelope。"""

            assert tuple(request) == (
                "artifact_id",
                "offset",
                "limit",
                "expected_sha256",
            )
            observed_requests.append(("read", dict(request)))
            offset = request["offset"]
            page = CONTENT[offset : offset + request["limit"]]
            next_offset = offset + len(page) if offset + len(page) < len(CONTENT) else None
            return {
                "schema_version": 1,
                "status": "success",
                "artifact_id": request["artifact_id"],
                "semantic_type": "scientific-text",
                "media_type": "text/plain",
                "sha256": CONTENT_SHA256,
                "encoding": "utf-8",
                "offset": offset,
                "returned_chars": len(page),
                "next_offset": next_offset,
                "total_chars": len(CONTENT),
                "truncated": next_offset is not None,
                "content": page,
            }

        application = server.streamable_http_app()

        async def capturing_application(
            scope: dict[str, Any], receive: Any, send: Any
        ) -> None:
            """记录 ASGI HTTP headers 后交给真实 Starlette MCP application。"""

            if scope["type"] == "http":
                observed_headers.append(dict(scope.get("headers", ())))
            await application(scope, receive, send)

        return application, capturing_application

    class RealMcpSessionContext:
        """在 gateway 的异步 runner 内运行真实 MCP client/ASGI server。"""

        def __init__(self, endpoint: str, headers: dict[str, str]) -> None:
            self.endpoint = endpoint
            self.headers = headers
            self._stack: AsyncExitStack | None = None
            self._application: Any = None
            self._capturing_application: Any = None

        async def __aenter__(self) -> ClientSession:
            stack = AsyncExitStack()
            self._stack = stack
            await stack.__aenter__()
            self._application, self._capturing_application = create_application()
            await stack.enter_async_context(
                self._application.router.lifespan_context(self._application)
            )
            client = httpx.AsyncClient(
                transport=httpx.ASGITransport(app=self._capturing_application),
                base_url="http://localhost:8000",
                headers=self.headers,
            )
            await stack.enter_async_context(client)
            transport = streamable_http_client(self.endpoint, http_client=client)
            read_stream, write_stream, _ = await stack.enter_async_context(transport)
            session = ClientSession(read_stream, write_stream)
            return await stack.enter_async_context(session)

        async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> Any:
            """释放真实 MCP session、transport、HTTP client 和服务生命周期。"""

            if self._stack is None:
                return None
            previous_shutdown = AppStatus.should_exit
            AppStatus.should_exit = True
            try:
                # EventSourceResponse 会为每个真实流启动 watcher；先让它观察到
                # shutdown，再关闭 runner 的事件循环，避免遗留 pending task。
                await asyncio.sleep(0.6)
            finally:
                AppStatus.should_exit = previous_shutdown
            return await self._stack.__aexit__(exc_type, exc, traceback)

    def factory(endpoint: str, headers: dict[str, str]) -> RealMcpSessionContext:
        """把 endpoint/header 注入真实 Streamable HTTP client。"""

        assert endpoint == "http://localhost:8000/mcp"
        assert headers["X-MiniChem-Capability"].startswith(TOKEN)
        return RealMcpSessionContext(endpoint, headers)

    gateway = InvestigationMiniChemGateway(
        "http://localhost:8000/mcp",
        {_binding(index).scope_id: _binding(index) for index in range(5)},
        evidence_store=ReviewEvidenceStore(tmp_path / "store"),
        source_cache_dir=tmp_path / "cache",
        session_factory=factory,
        page_size=4,
    )
    results = [
        _result(
            0,
            jobs=(_short_job_id(1),),
            operations=("snapshot-operation",),
            artifacts=(ARTIFACT_ID,),
        ),
        *tuple(_result(index) for index in range(1, 5)),
    ]
    state = _state(tuple(results))
    vector = tuple((f"combination-{index}", "v1") for index in range(5))

    manifest = gateway.evidence_snapshot_builder(state, tuple(results), vector)
    assert manifest["result_pairs"][0]["artifacts"][0]["source_artifact_ref"] == ARTIFACT_ID
    assert [kind for kind, _ in observed_requests].count("verify") == 5
    assert [kind for kind, _ in observed_requests].count("read") == 4
    assert all(
        tuple(request) == ("jobs", "operations", "artifacts")
        for kind, request in observed_requests
        if kind == "verify"
    )
    assert all(
        tuple(request) == (
            "artifact_id",
            "offset",
            "limit",
            "expected_sha256",
        )
        for kind, request in observed_requests
        if kind == "read"
    )
    assert [request["offset"] for kind, request in observed_requests if kind == "read"] == [0, 4, 8, 12]
    assert observed_headers
    capabilities = {
        dict(
            (key.decode().lower(), value.decode())
            for key, value in headers.items()
        )["x-minichem-capability"]
        for headers in observed_headers
    }
    assert capabilities == {TOKEN + str(index) for index in range(5)}
    gateway.close()


def test_mcp_object_parser_merges_supported_envelopes_and_fails_closed() -> None:
    """MCP typed object、structured content 和文本 content 必须一致且严格 JSON。"""

    assert gateway_module._parse_mcp_object(
        {"status": "success", "value": 1}, "verify"
    ) == {"status": "success", "value": 1}
    assert gateway_module._parse_mcp_object(
        {"structuredContent": {"value": 2}}, "verify"
    ) == {"value": 2}
    assert gateway_module._parse_mcp_object(
        {"structured_content": {"value": 3}}, "verify"
    ) == {"value": 3}
    assert gateway_module._parse_mcp_object(
        {
            "content": [
                {"resource": {"text": '{"value": 4}'}},
                {"text": '{"value": 4}'},
            ]
        },
        "verify",
    ) == {"value": 4}
    assert gateway_module._parse_mcp_object({"fallback": True}, "verify") == {
        "fallback": True
    }

    invalid_envelopes = (
        {"isError": True},
        {"is_error": True},
        {"content": [{"type": "image"}]},
        {"content": [{"text": "not-json"}]},
        {"content": [{"text": "NaN"}]},
        {"content": [{"text": '{"value": 1}'}, {"text": '{"value": 2}'}]},
        {"structuredContent": {"value": 1}, "content": [{"text": '{"value": 2}'}]},
        {"structuredContent": ["not", "object"]},
        SimpleNamespace(),
    )
    for raw in invalid_envelopes:
        with pytest.raises(InvestigationMiniChemProtocolError):
            gateway_module._parse_mcp_object(raw, "verify")

    with pytest.raises(InvestigationMiniChemProtocolError):
        gateway_module._parse_mcp_object({"structuredContent": {"bad": {1, 2}}}, "verify")


def test_async_session_runner_serves_invalid_and_failing_commands() -> None:
    """异步 runner 忽略未知队列项，并把 session 调用异常交给对应 Future。"""

    class Session:
        def echo(self, value: str) -> str:
            return value

        def fail(self) -> None:
            raise RuntimeError("transport failure")

    runner = gateway_module._AsyncSessionRunner(
        lambda: Session(),
        "http://127.0.0.1:9010/mcp",
        {"X-MiniChem-Capability": TOKEN},
    )
    runner._session = Session()
    runner._commands = asyncio.Queue()
    success = concurrent.futures.Future()
    failure = concurrent.futures.Future()
    success_command = gateway_module._SessionCommand("echo", ("ok",), {}, success)
    failure_command = gateway_module._SessionCommand("fail", (), {}, failure)
    with runner._command_lock:
        runner._pending_commands[id(success_command)] = success_command
        runner._pending_commands[id(failure_command)] = failure_command
    runner._commands.put_nowait(object())
    runner._commands.put_nowait(success_command)
    runner._commands.put_nowait(failure_command)
    runner._commands.put_nowait(gateway_module._CLOSE_COMMAND)

    asyncio.run(runner._serve_session_commands())
    assert success.result() == "ok"
    assert isinstance(failure.exception(), RuntimeError)


def test_async_session_runner_fails_queued_command_after_close_without_consuming_it() -> None:
    """关闭命令先到达时，旧队列命令必须失败且不能在旧 session 执行。"""

    class Session:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def initialize(self) -> None:
            return None

        async def call_tool(self, name: str, _arguments: dict[str, Any]) -> None:
            self.calls.append(name)

        async def close(self) -> None:
            return None

    session = Session()
    runner = gateway_module._AsyncSessionRunner(
        lambda: session,
        "http://127.0.0.1:9010/mcp",
        {"X-MiniChem-Capability": TOKEN},
    )
    runner.start()
    assert runner._loop is not None
    assert runner._commands is not None
    result: concurrent.futures.Future[Any] = concurrent.futures.Future()
    command = gateway_module._SessionCommand(
        "call_tool",
        ("verify_execution_references", {"request": {}}),
        {},
        result,
    )
    with runner._command_lock:
        runner._pending_commands[id(command)] = command
    runner._loop.call_soon_threadsafe(
        runner._commands.put_nowait,
        gateway_module._CLOSE_COMMAND,
    )
    runner._loop.call_soon_threadsafe(runner._commands.put_nowait, command)

    with pytest.raises(InvestigationMiniChemUnavailable):
        result.result(timeout=1)
    assert result.done()
    assert session.calls == []
    assert runner._pending_commands == {}
    runner.close()


def test_async_session_runner_close_fails_queued_call_after_active_call_stops() -> None:
    """旧调用停止时，排队命令的 Future 失败且不会被旧 session 消费。"""

    class Session:
        def __init__(self) -> None:
            self.active_started = threading.Event()
            self.calls: list[str] = []

        async def initialize(self) -> None:
            return None

        async def call_tool(self, name: str, _arguments: dict[str, Any]) -> None:
            self.calls.append(name)
            if name == "active":
                self.active_started.set()
                await asyncio.Future()

        async def close(self) -> None:
            return None

    session = Session()
    runner = gateway_module._AsyncSessionRunner(
        lambda: session,
        "http://127.0.0.1:9010/mcp",
        {"X-MiniChem-Capability": TOKEN},
    )
    runner.start()
    first_error: list[BaseException] = []
    second_error: list[BaseException] = []

    def invoke(
        name: str,
        errors: list[BaseException],
    ) -> None:
        try:
            runner.call("call_tool", name, {})
        except BaseException as error:
            errors.append(error)

    first_thread = threading.Thread(
        target=invoke,
        args=("active", first_error),
        daemon=True,
    )
    second_thread = threading.Thread(
        target=invoke,
        args=("queued", second_error),
        daemon=True,
    )
    first_thread.start()
    assert session.active_started.wait(timeout=1)
    second_thread.start()
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        with runner._command_lock:
            pending_count = len(runner._pending_commands)
        if pending_count == 2:
            break
        time.sleep(0.005)
    assert pending_count == 2

    runner.close(timeout=1)
    first_thread.join(timeout=1)
    second_thread.join(timeout=1)
    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert all(isinstance(error, InvestigationMiniChemUnavailable) for error in first_error)
    assert all(isinstance(error, InvestigationMiniChemUnavailable) for error in second_error)
    assert session.calls == ["active"]
    assert runner._pending_commands == {}


@pytest.mark.parametrize(
    "endpoint",
    [
        None,
        "ftp://127.0.0.1/mcp",
        "http://\x00.example/mcp",
        "http://[invalid/mcp",
        "http:///mcp",
    ],
)
def test_endpoint_normalization_rejects_invalid_public_urls(endpoint: Any) -> None:
    """endpoint 只允许无凭据、无 query/fragment 的 HTTP(S) URL。"""

    with pytest.raises(ValueError):
        gateway_module._require_endpoint(endpoint)


def test_job_inventory_parser_accepts_mapping_and_rejects_shape_drift() -> None:
    """job inventory 的包装、当前引用覆盖、ID 和状态都必须通过。"""

    job_id = _job_id(21)
    item = {"job_id": job_id, "status": "success"}
    gateway_module._validate_job_inventory({"jobs": [item]}, (job_id,))
    gateway_module._validate_job_inventory([item], (job_id,))
    historical_item = {"job_id": _job_id(22), "status": "failed"}
    with pytest.raises(InvestigationMiniChemProtocolError):
        gateway_module._validate_job_inventory([item, historical_item], (job_id,))
    gateway_module._validate_job_inventory(
        [item, historical_item],
        (job_id,),
        allow_historical_terminal_jobs=True,
    )
    with pytest.raises(InvestigationMiniChemProtocolError, match="未完整覆盖"):
        gateway_module._validate_job_inventory(
            [historical_item],
            (job_id,),
            allow_historical_terminal_jobs=True,
        )
    invalid_values = (
        None,
        {"jobs": "not-list"},
        [{"job_id": "bad", "status": "success"}],
        [{"job_id": job_id, "status": "running"}],
        [{"job_id": job_id, "status": "success", "extra": True}],
        [item, item],
    )
    for value in invalid_values:
        with pytest.raises(InvestigationMiniChemProtocolError):
            gateway_module._validate_job_inventory(value, (job_id,))


@pytest.mark.parametrize(
    "historical_item",
    [
        {"job_id": "job-foreign", "status": "success"},
        {"job_id": _job_id(24), "status": "running"},
        {"job_id": _job_id(24), "status": "success", "extra": True},
    ],
)
def test_result_inventory_history_allowance_remains_fail_closed(
    historical_item: dict[str, Any],
) -> None:
    """历史 job 例外只放行合法终态项，漏报、foreign 和损坏项仍拒绝。"""

    current_job = _job_id(23)
    with pytest.raises(InvestigationMiniChemProtocolError):
        gateway_module._validate_job_inventory(
            [
                {"job_id": current_job, "status": "success"},
                historical_item,
            ],
            (current_job,),
            allow_historical_terminal_jobs=True,
        )


def test_schema2_job_inventory_cross_checks_visible_submission_operation() -> None:
    """schema2 scope-submission job 的 inventory 字段必须与 job reference 一致。"""

    job_id = _job_id(22)
    submission_operation_id = "scope-submission-" + "a" * 64
    item = {
        "job_id": job_id,
        "status": "success",
        "submission_operation_id": submission_operation_id,
    }
    gateway_module._validate_job_inventory(
        [item],
        (job_id,),
        schema_version=2,
        expected_submission_operations={job_id: submission_operation_id},
    )
    with pytest.raises(InvestigationMiniChemProtocolError, match="提交 operation"):
        gateway_module._validate_job_inventory(
            [dict(item, submission_operation_id="scope-submission-" + "b" * 64)],
            (job_id,),
            schema_version=2,
            expected_submission_operations={job_id: submission_operation_id},
        )
    with pytest.raises(InvestigationMiniChemProtocolError, match="不应额外"):
        gateway_module._validate_job_inventory(
            [dict(item, submission_operation_id="scope-submission-" + "b" * 64)],
            (job_id,),
            schema_version=2,
            expected_submission_operations={job_id: None},
        )
    with pytest.raises(InvestigationMiniChemProtocolError, match="非法"):
        gateway_module._validate_job_inventory(
            [dict(item, submission_operation_id="plain-operation")],
            (job_id,),
            schema_version=2,
            expected_submission_operations={job_id: None},
        )


def test_artifact_page_helpers_cover_metadata_content_and_cursor_guards() -> None:
    """artifact 分页的字段、正文、连续游标和 EOF 关系必须分别 fail-closed。"""

    artifact_id = ARTIFACT_ID
    digest = "a" * 64
    response = {
        "schema_version": 1,
        "status": "success",
        "artifact_id": artifact_id,
        "semantic_type": "scientific-text",
        "media_type": "text/plain; charset=utf-8",
        "sha256": digest,
        "encoding": "utf-8",
        "offset": 0,
        "returned_chars": 1,
        "next_offset": None,
        "total_chars": 1,
        "truncated": False,
        "content": "x",
        "automatic_fallback": False,
    }
    page = gateway_module._parse_artifact_page_response(
        response,
        artifact_id=artifact_id,
        expected_sha256=digest,
        expected_offset=0,
        page_size=4,
        max_artifact_bytes=16,
    )
    assert page.content == "x"

    sdf_response = dict(response)
    sdf_response["media_type"] = "chemical/x-mdl-sdfile"
    sdf_page = gateway_module._parse_artifact_page_response(
        sdf_response,
        artifact_id=artifact_id,
        expected_sha256=digest,
        expected_offset=0,
        page_size=4,
        max_artifact_bytes=16,
    )
    assert sdf_page.media_type == "chemical/x-mdl-sdfile"

    invalid_responses = []
    missing = dict(response)
    missing.pop("content")
    invalid_responses.append(missing)
    bad_status = dict(response)
    bad_status["status"] = "error"
    invalid_responses.append(bad_status)
    bad_fallback = dict(response)
    bad_fallback["automatic_fallback"] = True
    invalid_responses.append(bad_fallback)
    bad_content = dict(response)
    bad_content["content"] = "\x00"
    invalid_responses.append(bad_content)
    bad_offset = dict(response)
    bad_offset["offset"] = 1
    invalid_responses.append(bad_offset)
    bad_truncated = dict(response)
    bad_truncated["truncated"] = "false"
    invalid_responses.append(bad_truncated)
    bad_eof = dict(response)
    bad_eof["total_chars"] = 2
    invalid_responses.append(bad_eof)
    continuation = dict(response)
    continuation.update(next_offset=2, truncated=True, total_chars=3)
    invalid_responses.append(continuation)
    for invalid in invalid_responses:
        with pytest.raises(InvestigationMiniChemProtocolError):
            gateway_module._parse_artifact_page_response(
                invalid,
                artifact_id=artifact_id,
                expected_sha256=digest,
                expected_offset=0,
                page_size=4,
                max_artifact_bytes=16,
            )


def test_scope_binding_normalizers_cover_mapping_sequence_and_object_inputs() -> None:
    """scope capability 支持的三种输入形状都必须保持身份校验。"""

    binding = _binding(0)
    assert gateway_module._normalize_scope_binding("scope-0", binding) is binding
    assert gateway_module._normalize_scope_binding("scope-0", binding.token).scope_id == "scope-0"
    mapping_binding = gateway_module._normalize_scope_binding(
        "scope-0",
        {
            "token": binding.token,
            "task_id": TASK_ID,
            "batch_id": BATCH_ID,
            "combination_id": "combination-0",
        },
    )
    assert mapping_binding.identity_is_complete
    object_binding = SimpleNamespace(
        scope_id="scope-0",
        token=binding.token,
        task_id=TASK_ID,
        batch_id=BATCH_ID,
        combination_id="combination-0",
    )
    assert gateway_module._normalize_scope_binding("scope-0", object_binding).scope_id == "scope-0"

    for value in (
        SimpleNamespace(scope_id="scope-other", token=binding.token),
        {"scope_id": "scope-other", "token": binding.token},
        object(),
    ):
        with pytest.raises(ValueError):
            gateway_module._normalize_scope_binding("scope-0", value)

    assert gateway_module.InvestigationMiniChemGateway._normalize_scope_capabilities(
        {"scope-0": binding}
    )["scope-0"] is binding
    assert gateway_module.InvestigationMiniChemGateway._normalize_scope_capabilities(
        [binding]
    )["scope-0"] is binding
    assert gateway_module.InvestigationMiniChemGateway._normalize_scope_capabilities(
        object_binding
    )["scope-0"].scope_id == "scope-0"
    for value in (
        {},
        "capability-token",
        [object()],
        [binding, binding],
    ):
        with pytest.raises((TypeError, ValueError)):
            gateway_module.InvestigationMiniChemGateway._normalize_scope_capabilities(value)


def test_ts_unknown_controller_status_is_terminal_and_armable_without_resubmit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TS 状态目录消失时返回完整 unknown 终态，Controller 不会重新提交。"""

    waiting = _waiting_ts()
    binding = InvestigationScopeBinding(
        scope_id=waiting.scope_id,
        token=TOKEN + "1",
        task_id=TASK_ID,
        batch_id=BATCH_ID,
        combination_id=waiting.combination_id,
        workspace=tmp_path / "workspace",
    )
    gateway = InvestigationMiniChemGateway(
        "http://127.0.0.1:9010/mcp",
        {waiting.scope_id: binding},
        evidence_store=ReviewEvidenceStore(tmp_path / "store"),
        source_cache_dir=tmp_path / "cache",
    )
    from minichem_mcp_tools import transition_state_jobs

    monkeypatch.setattr(
        transition_state_jobs,
        "controller_get_transition_state_path_job",
        lambda *_args, **_kwargs: {
            "status": "invalid_request",
            "error": {
                "code": "transition_state_path_job_unavailable",
                "message": "作业状态无法核对",
            },
            "job_id": waiting.identity.job_id,
        },
    )
    monkeypatch.setattr(
        transition_state_jobs,
        "controller_arm_transition_state_path_job",
        lambda *_args, **_kwargs: {
            "status": "invalid_request",
            "error": {
                "code": "transition_state_path_job_unavailable",
                "message": "作业状态无法核对",
            },
        },
    )

    status = gateway.get_investigation_job_status(waiting)
    assert status["job_status"] == "unknown"
    assert status["terminal"] is True
    assert status["job_id"] == waiting.identity.job_id
    assert status["operation_id"] == waiting.identity.operation_id
    assert status["submission_key"] == waiting.identity.submission_key
    assert status["action_id"] == waiting.identity.action_id
    assert status["backend_id"] == waiting.identity.backend_id
    assert "action_result" not in status
    assert "log" not in status

    armed = gateway.arm_investigation_job(waiting, "unknown")
    assert armed["job_status"] == "unknown"
    assert armed["job_id"] == waiting.identity.job_id
    assert armed["phase"] == "unknown"
    assert armed["synthetic_terminal"] == "unknown"


def test_async_context_exit_swallows_cleanup_failures_and_handles_empty_state() -> None:
    """MCP context、transport 和 HTTP client 的关闭异常不能泄露给调用方。"""

    class FailingAsyncExit:
        async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
            raise RuntimeError("close failure")

    class FailingClient:
        async def aclose(self) -> None:
            raise RuntimeError("client failure")

    context = gateway_module._AuthenticatedStreamableHTTPContext(
        "http://127.0.0.1:9010/mcp",
        {"X-MiniChem-Capability": TOKEN},
    )
    context._session_context = FailingAsyncExit()
    context._transport_context = FailingAsyncExit()
    context._http_client = FailingClient()
    asyncio.run(context.__aexit__(None, None, None))
    assert context._session_context is None
    assert context._transport_context is None
    assert context._http_client is None
    asyncio.run(context.__aexit__(None, None, None))


def test_authenticated_mcp_http_client_keeps_stream_open_for_gateway_timeout() -> None:
    """真实 MCP client 不应使用 HTTPX 的 5 秒默认读取超时。"""

    client = gateway_module._create_authenticated_mcp_http_client(
        {"X-MiniChem-Capability": TOKEN}
    )
    try:
        assert client.headers["X-MiniChem-Capability"] == TOKEN
        assert client.timeout.connect == gateway_module._MCP_HTTP_REQUEST_TIMEOUT_SECONDS
        assert client.timeout.write == gateway_module._MCP_HTTP_REQUEST_TIMEOUT_SECONDS
        assert client.timeout.pool == gateway_module._MCP_HTTP_REQUEST_TIMEOUT_SECONDS
        assert client.timeout.read == gateway_module._MCP_HTTP_SSE_READ_TIMEOUT_SECONDS
        assert client.timeout.read == gateway_module._SESSION_CALL_TIMEOUT_SECONDS
    finally:
        asyncio.run(client.aclose())
