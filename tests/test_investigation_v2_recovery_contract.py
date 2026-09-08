"""research_task 恢复核验结果与 Investigation Harness 公开 seam 回归。"""

from __future__ import annotations

from pathlib import Path

import pytest

from chemistry_multiagent.controllers.investigation_v2_module_controller import (
    ControllerSuspended,
    ModuleFailed,
    ModuleInvocationRequest,
    ModuleSuspended,
    ResearchWorkflowController,
    ResearchTaskFailedResult,
)
from chemistry_multiagent.investigation.harness_runtime import (
    BackendCompleted,
    BackendSuspended,
    Completed,
    InvestigationHarness,
    InvestigationHarnessRequest,
    Suspended,
)
from chemistry_multiagent.investigation.recovery_contract import (
    Consistent,
    RecoveryEvidenceMissing,
    RecoveryStatus,
    RecoveryVerification,
    Reconciled,
    TemporarilyUnavailable,
    VerifiedIntegrityConflict,
    reconcile_recovery_identity,
)
from chemistry_multiagent.memory import (
    ResearchTaskWorkingMemory,
    PreparedModuleInvocationOutput,
    ScientificRecord,
)
from chemistry_multiagent.memory.v2_identities import StatelessModule


def _task_kwargs() -> dict[str, object]:
    """返回一份 V2 task.v2 的最小冻结输入。"""

    return {
        "contract_version": "investigation_v2.task.v2",
        "input": {"scientific_question": "recovery contract"},
        "policy": {"entry_strategy": "standard_hypothesis_generation"},
        "dependencies": {"catalog": {"ref": "catalog-1"}},
    }


def _request() -> InvestigationHarnessRequest:
    """构造 canonical Harness 请求。"""

    return InvestigationHarnessRequest(
        research_task_id="research-task-1",
        module_invocation_id="invocation-1",
        module_kind="hypothesis-investigation",
        input_scientific_record_ids=("execution-input-1",),
        module_input_digest="1" * 64,
        context_binding={
            "kind": "CreateInvestigationContext",
            "investigation_context_id": "context-1",
            "hypothesis_batch_record_id": "batch-1",
        },
    )


def _output() -> PreparedModuleInvocationOutput:
    """构造最小合法 Investigation 输出。"""

    evidence = ScientificRecord.from_payload(
        scientific_record_id="evidence-1",
        research_task_id="research-task-1",
        scientific_record_type="VerifiedEvidenceRecord",
        scientific_record_schema_version=1,
        producer_module_invocation_id="invocation-1",
        source_scientific_record_ids=(),
        payload={
            "evidence_items": [{"value": "observed"}],
            "artifact_refs": [],
            "source_manifest_ref": "manifest-1",
            "summary": "evidence",
            "verification": {"verified": True},
        },
    )
    result = ScientificRecord.from_payload(
        scientific_record_id="result-1",
        research_task_id="research-task-1",
        scientific_record_type="InvestigationResultRecord",
        scientific_record_schema_version=1,
        producer_module_invocation_id="invocation-1",
        source_scientific_record_ids=("evidence-1",),
        payload={
            "conclusion": "supported",
            "self_judgment": "supported",
            "observations": ["observed"],
            "limitations": [],
            "evidence_ref": "evidence-1",
        },
    )
    return PreparedModuleInvocationOutput.from_records(
        module_invocation_id="invocation-1",
        module_input_digest="1" * 64,
        output_scientific_records=(evidence, result),
    )


def test_recovery_contract_has_five_explicit_results_and_no_unknown() -> None:
    """恢复结果必须是五个互斥状态，而不是 None/unknown。"""

    assert {item.value for item in RecoveryStatus} == {
        "Consistent",
        "Reconciled",
        "TemporarilyUnavailable",
        "RecoveryEvidenceMissing",
        "VerifiedIntegrityConflict",
    }
    assert isinstance(Consistent(reason_code="identity_match"), RecoveryVerification)
    assert Reconciled(
        reason_code="sealed_output_replayed",
        reconciled_fields=("module_input_digest",),
    ).can_continue
    assert TemporarilyUnavailable(reason_code="provider_unavailable").must_suspend
    assert RecoveryEvidenceMissing(reason_code="manifest_missing").must_suspend
    assert VerifiedIntegrityConflict(reason_code="identity_drift").must_fail_integrity


def test_reconcile_distinguishes_missing_reconciled_and_verified_conflict() -> None:
    """缺失证据和可核准差异不能被误报为身份冲突。"""

    missing = reconcile_recovery_identity(
        {"research_task_id": "research-task-1", "module_invocation_id": "invocation-1"},
        {"research_task_id": "research-task-1"},
        required_fields=("research_task_id", "module_invocation_id"),
    )
    assert missing.status is RecoveryStatus.RECOVERY_EVIDENCE_MISSING

    reconciled = reconcile_recovery_identity(
        {"research_task_id": "research-task-1", "module_invocation_id": "invocation-1"},
        {"research_task_id": "research-task-1", "module_invocation_id": "invocation-2"},
        required_fields=("research_task_id", "module_invocation_id"),
        reconciled_fields=("module_invocation_id",),
    )
    assert reconciled.status is RecoveryStatus.RECONCILED
    assert reconciled.reconciled_fields == ("module_invocation_id",)

    conflict = reconcile_recovery_identity(
        {"research_task_id": "research-task-1", "module_invocation_id": "invocation-1"},
        {"research_task_id": "research-task-1", "module_invocation_id": "invocation-2"},
        required_fields=("research_task_id", "module_invocation_id"),
    )
    assert conflict.status is RecoveryStatus.VERIFIED_INTEGRITY_CONFLICT


def test_harness_replays_sealed_output_without_calling_backend_again(
    tmp_path: Path,
) -> None:
    """已密封输出恢复为 Reconciled/Consistent，不重复进入 backend。"""

    class Backend:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def start(self, request, harness_invocation_id):
            self.calls.append("start")
            return BackendCompleted(_output())

        def resume(self, request, harness_invocation_id):
            self.calls.append("resume")
            raise AssertionError("密封输出恢复不应调用 backend")

    backend = Backend()
    runtime_root = tmp_path / "harness"
    first = InvestigationHarness(runtime_root, backend=backend).start(_request())
    assert isinstance(first, Completed)
    assert first.recovery_verification.can_continue

    restarted = InvestigationHarness(runtime_root, backend=backend)
    replay = restarted.resume(_request())
    assert isinstance(replay, Completed)
    assert replay.prepared_output.to_dict() == _output().to_dict()
    assert backend.calls == ["start"]


def test_missing_harness_recovery_is_suspended_not_failed(tmp_path: Path) -> None:
    """缺少 Harness 恢复证据只能暂停，不能伪造失败终态。"""

    outcome = InvestigationHarness(tmp_path / "missing", backend=None).resume(_request())
    assert isinstance(outcome, Suspended)
    assert outcome.recovery_verification.status is RecoveryStatus.RECOVERY_EVIDENCE_MISSING


def test_backend_temporarily_unavailable_stays_suspended(tmp_path: Path) -> None:
    """backend 返回暂时不可用时，Harness 保留可恢复结果。"""

    class Backend:
        def start(self, request, harness_invocation_id):
            return BackendSuspended(
                "provider_unavailable",
                recovery_verification=TemporarilyUnavailable(
                    reason_code="provider_unavailable"
                ),
            )

        def resume(self, request, harness_invocation_id):
            return self.start(request, harness_invocation_id)

    outcome = InvestigationHarness(tmp_path / "runtime", backend=Backend()).start(_request())
    assert isinstance(outcome, Suspended)
    assert outcome.recovery_verification.status is RecoveryStatus.TEMPORARILY_UNAVAILABLE


def test_controller_maps_suspension_without_private_identity_access(tmp_path: Path) -> None:
    """Controller 只消费公开恢复状态，并保持 Open Module Invocation。"""

    memory = ResearchTaskWorkingMemory(tmp_path / "memory.sqlite3")

    class Module:
        def start(self, request: ModuleInvocationRequest):
            return ModuleSuspended(
                "provider_unavailable",
                recovery_verification=TemporarilyUnavailable(
                    reason_code="provider_unavailable"
                ),
            )

        def resume(self, request: ModuleInvocationRequest):
            return ModuleSuspended(
                "manifest_missing",
                recovery_verification=RecoveryEvidenceMissing(
                    reason_code="manifest_missing"
                ),
            )

    controller = ResearchWorkflowController(memory, {"retrieval": Module()})
    result = controller.run_research_task("research-task-1", **_task_kwargs())
    assert isinstance(result, ControllerSuspended)
    assert result.recovery_verification.status is RecoveryStatus.TEMPORARILY_UNAVAILABLE
    facts = memory.read_research_task_facts("research-task-1")
    assert facts.lifecycle.state == "Open"
    assert facts.open_module_invocation is not None

    resumed = controller.resume_research_task("research-task-1")
    assert isinstance(resumed, ControllerSuspended)
    assert resumed.recovery_verification.status is RecoveryStatus.RECOVERY_EVIDENCE_MISSING


def test_only_verified_integrity_conflict_closes_task_as_failure(tmp_path: Path) -> None:
    """只有已核实的身份冲突才原子写入失败终态。"""

    memory = ResearchTaskWorkingMemory(tmp_path / "memory.sqlite3")

    class ConflictModule:
        def start(self, request: ModuleInvocationRequest):
            return ModuleFailed("verified_integrity_conflict")

        def resume(self, request: ModuleInvocationRequest):
            return self.start(request)

    controller = ResearchWorkflowController(memory, {"retrieval": ConflictModule()})
    result = controller.run_research_task("research-task-1", **_task_kwargs())
    assert isinstance(result, ResearchTaskFailedResult)
    assert result.outcome["failure_code"] == "verified_integrity_conflict"
    assert memory.read_research_task_facts("research-task-1").lifecycle.state == "Closed"
    assert [event.event_type for event in memory.read_research_task_event_log("research-task-1")][-2:] == [
        "ModuleInvocationFailed",
        "ResearchTaskFailed",
    ]
