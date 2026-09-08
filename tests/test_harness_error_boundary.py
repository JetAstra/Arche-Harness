"""Investigation Harness 与公开 workflow controller 的错误边界测试。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from chemistry_multiagent.controllers.investigation_v2_module_controller import (
    ControllerSuspended,
    HarnessModuleRunner,
    ModuleCompleted,
    ModuleFailed,
    ModuleInvocationRequest,
    ModuleSuspended,
    ResearchWorkflowController,
)
from chemistry_multiagent.investigation import (
    BackendCompleted,
    BackendFatalFailure,
    BackendSuspended,
    InvestigationHarness,
    InvestigationHarnessContractError,
    InvestigationHarnessPersistenceError,
    InvestigationHarnessRecoveryStateWriteError,
    InvestigationHarnessRequest,
)
from chemistry_multiagent.investigation.recovery_contract import (
    RecoveryStatus,
    TemporarilyUnavailable,
)
from chemistry_multiagent.memory import ResearchTaskWorkingMemory, ScientificRecord
from chemistry_multiagent.memory.v2_ledger import PreparedModuleInvocationOutput
from chemistry_multiagent.memory.v2_identities import CreateInvestigationContext


def _harness_request() -> InvestigationHarnessRequest:
    """构造当前 Harness 边界的最小请求。"""

    return InvestigationHarnessRequest(
        research_task_id="research-task-1",
        module_invocation_id="invocation-1",
        module_kind="hypothesis-investigation",
        input_scientific_record_ids=("selection-1",),
        module_input_digest="digest-1",
        context_binding={
            "kind": "CreateInvestigationContext",
            "investigation_context_id": "context-1",
            "hypothesis_batch_record_id": "batch-1",
        },
    )


def _module_request() -> ModuleInvocationRequest:
    """构造公开 Controller 到 Harness adapter 的请求。"""

    return ModuleInvocationRequest(
        research_task_id="research-task-1",
        module_invocation_id="invocation-1",
        module_kind="hypothesis-investigation",
        input_scientific_record_ids=("selection-1",),
        module_input_digest="digest-1",
        context_binding=CreateInvestigationContext("context-1", "batch-1"),
    )


def _prepared_output(request: InvestigationHarnessRequest) -> PreparedModuleInvocationOutput:
    """按照公开请求构造确定性的 evidence/result 输出包。"""

    evidence_id = f"{request.module_invocation_id}:evidence"
    result_id = f"{request.module_invocation_id}:result"
    evidence = ScientificRecord.from_payload(
        scientific_record_id=evidence_id,
        research_task_id=request.research_task_id,
        scientific_record_type="VerifiedEvidenceRecord",
        scientific_record_schema_version=1,
        producer_module_invocation_id=request.module_invocation_id,
        source_scientific_record_ids=request.input_scientific_record_ids,
        payload={
            "evidence_items": [{"value": "observed"}],
            "artifact_refs": [],
            "source_manifest_ref": "manifest-1",
            "summary": "evidence",
            "verification": {"verified": True},
        },
    )
    result = ScientificRecord.from_payload(
        scientific_record_id=result_id,
        research_task_id=request.research_task_id,
        scientific_record_type="InvestigationResultRecord",
        scientific_record_schema_version=1,
        producer_module_invocation_id=request.module_invocation_id,
        source_scientific_record_ids=(evidence_id,),
        payload={
            "conclusion": "supported",
            "self_judgment": "supported",
            "observations": ["observed"],
            "limitations": [],
            "evidence_ref": evidence_id,
        },
    )
    return PreparedModuleInvocationOutput.from_records(
        module_invocation_id=request.module_invocation_id,
        module_input_digest=request.module_input_digest,
        output_scientific_records=(evidence, result),
    )


class DeterministicBackend:
    """按预设结果运行，不创建模型、MCP 或化学作业。"""

    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls: list[str] = []

    def start(self, request: InvestigationHarnessRequest, harness_invocation_id: str) -> Any:
        del harness_invocation_id
        self.calls.append("start")
        if self.result == "output":
            return BackendCompleted(_prepared_output(request))
        return self.result

    def resume(self, request: InvestigationHarnessRequest, harness_invocation_id: str) -> Any:
        del harness_invocation_id
        self.calls.append("resume")
        if self.result == "output":
            return BackendCompleted(_prepared_output(request))
        return self.result


def test_harness_persistence_errors_have_distinct_recovery_boundaries() -> None:
    """恢复资料缺失与恢复状态写失败共享基类但不能混用。"""

    from chemistry_multiagent.investigation import (
        InvestigationHarnessRecoveryEvidenceUnavailable,
    )

    assert issubclass(
        InvestigationHarnessRecoveryEvidenceUnavailable,
        InvestigationHarnessPersistenceError,
    )
    assert issubclass(
        InvestigationHarnessRecoveryStateWriteError,
        InvestigationHarnessPersistenceError,
    )
    assert (
        InvestigationHarnessRecoveryEvidenceUnavailable
        is not InvestigationHarnessRecoveryStateWriteError
    )


def test_scientific_action_failure_is_not_a_harness_fatal_outcome() -> None:
    """单个科学作业失败不能伪装为 Harness 的完整性失败。"""

    with pytest.raises(InvestigationHarnessContractError):
        BackendFatalFailure("scientific_failed")


def test_provider_unavailable_is_suspended_through_public_runner(
    tmp_path: Path,
) -> None:
    """provider/MCP 暂不可用只能暂停，不关闭 Harness invocation。"""

    backend = DeterministicBackend(
        BackendSuspended(
            "provider_unavailable",
            recovery_verification=TemporarilyUnavailable(
                reason_code="mcp_disconnected"
            ),
        )
    )
    runner = HarnessModuleRunner(
        InvestigationHarness(tmp_path / "runtime", backend=backend)
    )

    outcome = runner.start(_module_request())

    assert isinstance(outcome, ModuleSuspended)
    assert outcome.recovery_verification.status is RecoveryStatus.TEMPORARILY_UNAVAILABLE
    assert backend.calls == ["start"]


def test_identity_conflict_is_controlled_module_failure(tmp_path: Path) -> None:
    """已核实输出身份冲突返回 ModuleFailed，并停止 Harness 重试。"""

    request = _harness_request()
    wrong = InvestigationHarnessRequest(
        research_task_id=request.research_task_id,
        module_invocation_id="invocation-other",
        module_kind=request.module_kind,
        input_scientific_record_ids=request.input_scientific_record_ids,
        module_input_digest=request.module_input_digest,
        context_binding=request.context_binding,
    )
    backend = DeterministicBackend(BackendCompleted(_prepared_output(wrong)))
    runner = HarnessModuleRunner(
        InvestigationHarness(tmp_path / "runtime", backend=backend)
    )

    outcome = runner.start(_module_request())

    assert isinstance(outcome, ModuleFailed)
    assert outcome.failure_code == "verified_integrity_conflict"
    assert outcome.recovery_verification is not None
    assert outcome.recovery_verification.status is RecoveryStatus.VERIFIED_INTEGRITY_CONFLICT
    assert backend.calls == ["start"]


def test_recovery_read_missing_is_suspended_through_public_runner(tmp_path: Path) -> None:
    """找不到恢复资料时返回 RecoveryEvidenceMissing，不调用 backend。"""

    backend = DeterministicBackend("output")
    runner = HarnessModuleRunner(
        InvestigationHarness(tmp_path / "runtime", backend=backend)
    )

    outcome = runner.resume(_module_request())

    assert isinstance(outcome, ModuleSuspended)
    assert outcome.recovery_verification.status is RecoveryStatus.RECOVERY_EVIDENCE_MISSING
    assert backend.calls == []


def test_unexpected_backend_error_keeps_invocation_recoverable(tmp_path: Path) -> None:
    """未声明 backend 错误向上传播；后续 resume 仍使用同一 invocation。"""

    class FlakyBackend:
        def __init__(self) -> None:
            self.fail = True
            self.calls: list[str] = []

        def start(self, request: InvestigationHarnessRequest, harness_invocation_id: str) -> Any:
            del harness_invocation_id
            self.calls.append("start")
            if self.fail:
                raise RuntimeError("unexpected backend bug")
            return BackendCompleted(_prepared_output(request))

        def resume(self, request: InvestigationHarnessRequest, harness_invocation_id: str) -> Any:
            del harness_invocation_id
            self.calls.append("resume")
            return BackendCompleted(_prepared_output(request))

    backend = FlakyBackend()
    runner = HarnessModuleRunner(
        InvestigationHarness(tmp_path / "runtime", backend=backend)
    )

    with pytest.raises(RuntimeError, match="unexpected backend bug"):
        runner.start(_module_request())

    backend.fail = False
    resumed = runner.resume(_module_request())

    assert isinstance(resumed, ModuleCompleted)
    assert backend.calls == ["start", "resume"]


def test_controller_preserves_open_task_when_module_is_temporarily_unavailable(
    tmp_path: Path,
) -> None:
    """Controller 只把暂不可用映射为暂停，Working Memory 保留 Open 事实。"""

    class SuspendedModule:
        def start(self, request: ModuleInvocationRequest) -> ModuleSuspended:
            del request
            return ModuleSuspended(
                "provider_unavailable",
                recovery_verification=TemporarilyUnavailable(
                    reason_code="mcp_disconnected"
                ),
            )

        def resume(self, request: ModuleInvocationRequest) -> ModuleSuspended:
            return self.start(request)

    memory = ResearchTaskWorkingMemory(tmp_path / "research_task.sqlite3")
    controller = ResearchWorkflowController(
        memory,
        {"retrieval": SuspendedModule()},
        require_entry_strategy=True,
    )
    outcome = controller.run_research_task(
        "research-task-1",
        contract_version="investigation_v2.task.v2",
        input={"scientific_question": "test"},
        policy={"entry_strategy": "standard_hypothesis_generation"},
        dependencies={},
    )

    assert isinstance(outcome, ControllerSuspended)
    assert outcome.recovery_verification.status is RecoveryStatus.TEMPORARILY_UNAVAILABLE
    facts = memory.read_research_task_facts("research-task-1")
    assert facts.lifecycle.state == "Open"
    assert facts.open_module_invocation is not None
