"""MiniChem Investigation v2 的 Harness 恢复边界测试。

旧 ``waiting_execution``/restart patch 属于 Legacy Workflow V1，不再作为当前
Controller 的恢复入口。本文件保留原来“暂停后可继续、身份漂移必须停止”的测试
意图，但改为验证 V2 Harness 的公开请求和独立恢复事实存储。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chemistry_multiagent.investigation.harness_runtime import (
    BackendSuspended,
    InvestigationHarness,
    InvestigationHarnessContractError,
    InvestigationHarnessIdentityConflict,
    InvestigationHarnessRequest,
    Suspended,
    _recovery_identity,
)
from chemistry_multiagent.investigation.recovery_contract import (
    RecoveryStatus,
    TemporarilyUnavailable,
)


def _request(
    *,
    research_task_id: str = "research-task-1",
    module_input_digest: str = "a" * 64,
) -> InvestigationHarnessRequest:
    """构造只包含 Controller/Harness 边界字段的当前请求。"""

    return InvestigationHarnessRequest(
        research_task_id=research_task_id,
        module_invocation_id="investigation-module-invocation-1",
        module_kind="hypothesis-investigation",
        input_scientific_record_ids=("hypothesis-execution-input-1",),
        module_input_digest=module_input_digest,
        context_binding={
            "kind": "CreateInvestigationContext",
            "investigation_context_id": "investigation-context-1",
            "hypothesis_execution_input_record_id": "hypothesis-execution-input-1",
        },
    )


class _SuspendingBackend:
    """不调用 MiniChem 的确定性 backend，用于验证 Harness 恢复状态。"""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def start(self, request: InvestigationHarnessRequest, harness_invocation_id: str):
        self.calls.append("start")
        return BackendSuspended(
            "provider_unavailable",
            recovery_verification=TemporarilyUnavailable(
                reason_code="provider_unavailable"
            ),
        )

    def resume(self, request: InvestigationHarnessRequest, harness_invocation_id: str):
        self.calls.append("resume")
        return BackendSuspended(
            "provider_unavailable",
            recovery_verification=TemporarilyUnavailable(
                reason_code="provider_unavailable"
            ),
        )


def test_v2_request_rejects_legacy_restart_patch_before_harness_creation() -> None:
    """V2 请求没有 Legacy patch 动作，也不会因其创建恢复状态。"""

    payload = _request().to_dict()
    payload["action"] = "patch_and_continue"

    with pytest.raises(InvestigationHarnessContractError, match="额外字段"):
        InvestigationHarnessRequest.from_mapping(payload)


def test_v2_harness_resume_reuses_suspended_invocation_without_new_start(
    tmp_path: Path,
) -> None:
    """暂停后恢复只调用同一 invocation 的 resume，不重新创建入口。"""

    backend = _SuspendingBackend()
    harness = InvestigationHarness(tmp_path / "runtime", backend=backend)
    first = harness.start(_request())
    assert isinstance(first, Suspended)
    assert first.recovery_verification.status is RecoveryStatus.TEMPORARILY_UNAVAILABLE

    restarted = InvestigationHarness(tmp_path / "runtime", backend=backend)
    second = restarted.resume(first.harness_invocation_id, _request())
    assert isinstance(second, Suspended)
    assert second.harness_invocation_id == first.harness_invocation_id
    assert backend.calls == ["start", "resume"]

    snapshot = restarted.recovery_store.load(_recovery_identity(_request()))
    assert snapshot is not None
    assert snapshot.state.value == "suspended"


def test_v2_harness_identity_drift_is_not_a_restartable_failure(tmp_path: Path) -> None:
    """Task 或输入摘要漂移必须 fail closed，不能被当作普通暂停重试。"""

    backend = _SuspendingBackend()
    harness = InvestigationHarness(tmp_path / "runtime", backend=backend)
    first = harness.start(_request())
    assert isinstance(first, Suspended)

    with pytest.raises(InvestigationHarnessIdentityConflict, match="漂移|身份"):
        harness.resume(
            first.harness_invocation_id,
            _request(module_input_digest="b" * 64),
        )
    assert backend.calls == ["start"]


def test_v2_recovery_statuses_have_no_implicit_unknown_state() -> None:
    """恢复状态集合明确区分继续、暂停、缺证据和已核实完整性冲突。"""

    assert "unknown" not in {status.value.casefold() for status in RecoveryStatus}
    assert {status.value for status in RecoveryStatus} == {
        "Consistent",
        "Reconciled",
        "TemporarilyUnavailable",
        "RecoveryEvidenceMissing",
        "VerifiedIntegrityConflict",
    }
