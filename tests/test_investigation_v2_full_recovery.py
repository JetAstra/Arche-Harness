"""ResearchWorkflowController 与 Investigation Harness 的 canonical 恢复契约。"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from chemistry_multiagent.controllers.investigation_v2_module_controller import (
    ControllerSuspended,
    HarnessModuleRunner,
    ModuleSuspended,
    ResearchWorkflowController,
    build_module_input_digest,
)
from chemistry_multiagent.investigation.harness_runtime import (
    BackendCompleted,
    BackendSuspended,
    InvestigationHarness,
    InvestigationHarnessIdentityConflict,
    InvestigationHarnessRequest,
    Completed,
    Suspended,
)
from chemistry_multiagent.investigation.recovery_contract import RecoveryStatus
from chemistry_multiagent.memory import (
    ModuleInvocationRequest,
    PreparedModuleInvocationOutput,
    ResearchTaskWorkingMemory,
    ScientificRecord,
)
from chemistry_multiagent.memory.v2_identities import CreateInvestigationContext


def _request(
    *,
    research_task_id: str = "research-task-1",
    module_invocation_id: str = "investigation-1",
    input_ids: tuple[str, ...] = ("execution-input-1",),
    module_input_digest: str | None = None,
) -> InvestigationHarnessRequest:
    """构造不含 Harness 私有身份的正式 Investigation request。"""

    return InvestigationHarnessRequest(
        research_task_id=research_task_id,
        module_invocation_id=module_invocation_id,
        module_kind="hypothesis-investigation",
        input_scientific_record_ids=input_ids,
        module_input_digest=module_input_digest
        or build_module_input_digest(
            research_task_id,
            "hypothesis-investigation",
            input_ids,
        ),
        context_binding=CreateInvestigationContext(
            "investigation-context-1",
            "hypothesis-batch-1",
        ).to_dict(),
    )


def _prepared_output(request: InvestigationHarnessRequest) -> PreparedModuleInvocationOutput:
    """构造一对合法的 Investigation 科学记录。"""

    evidence_id = f"{request.module_invocation_id}:evidence"
    evidence = ScientificRecord.from_payload(
        scientific_record_id=evidence_id,
        research_task_id=request.research_task_id,
        scientific_record_type="VerifiedEvidenceRecord",
        scientific_record_schema_version=1,
        producer_module_invocation_id=request.module_invocation_id,
        source_scientific_record_ids=request.input_scientific_record_ids,
        payload={
            "evidence_items": [{"status": "verified"}],
            "artifact_refs": [],
            "source_manifest_ref": "manifest-1",
            "summary": "deterministic evidence",
            "verification": {"verified": True},
        },
    )
    result = ScientificRecord.from_payload(
        scientific_record_id=f"{request.module_invocation_id}:result",
        research_task_id=request.research_task_id,
        scientific_record_type="InvestigationResultRecord",
        scientific_record_schema_version=1,
        producer_module_invocation_id=request.module_invocation_id,
        source_scientific_record_ids=(
            *request.input_scientific_record_ids,
            evidence_id,
        ),
        payload={
            "conclusion": "deterministic result",
            "self_judgment": "supported",
            "observations": ["verified"],
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
    """记录 Harness 调用并按配置返回暂停或固定输出。"""

    def __init__(self, *, suspend_start: bool = False, suspend_resume: bool = False) -> None:
        self.suspend_start = suspend_start
        self.suspend_resume = suspend_resume
        self.start_calls: list[tuple[str, str]] = []
        self.resume_calls: list[tuple[str, str]] = []

    def start(self, request: InvestigationHarnessRequest, harness_invocation_id: str) -> object:
        self.start_calls.append((request.module_invocation_id, harness_invocation_id))
        if self.suspend_start:
            return BackendSuspended("provider_temporarily_unavailable")
        return BackendCompleted(_prepared_output(request))

    def resume(self, request: InvestigationHarnessRequest, harness_invocation_id: str) -> object:
        self.resume_calls.append((request.module_invocation_id, harness_invocation_id))
        if self.suspend_resume:
            return BackendSuspended("provider_temporarily_unavailable")
        return BackendCompleted(_prepared_output(request))


def test_harness_persists_canonical_request_and_replays_completed_output(
    tmp_path: Path,
) -> None:
    """完成输出跨 Harness 实例重放，不重新调用内部 backend。"""

    request = _request()
    backend = DeterministicBackend()
    harness = InvestigationHarness(tmp_path / "harness", backend=backend)
    first = harness.start(request)
    assert isinstance(first, Completed)
    assert first.harness_invocation_id == request.harness_invocation_id
    assert first.prepared_output.module_invocation_id == request.module_invocation_id

    restarted = InvestigationHarness(tmp_path / "harness", backend=backend)
    replay = restarted.resume(request)
    assert isinstance(replay, Completed)
    assert replay.prepared_output.to_dict() == first.prepared_output.to_dict()
    assert len(backend.start_calls) == 1
    assert len(backend.resume_calls) == 0

    manifest_path = next((tmp_path / "harness" / "runs").rglob("manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert set(manifest["request"]) == {
        "research_task_id",
        "module_invocation_id",
        "module_kind",
        "input_scientific_record_ids",
        "module_input_digest",
        "context_binding",
    }
    assert "task_id" not in manifest["request"]
    assert "runtime_ref" not in manifest["request"]


def test_harness_suspension_then_resume_uses_same_invocation_identity(
    tmp_path: Path,
) -> None:
    """暂时不可用只写暂停事实，恢复继续同一 module invocation。"""

    request = _request(module_invocation_id="investigation-suspended")
    backend = DeterministicBackend(suspend_start=True)
    harness = InvestigationHarness(tmp_path / "harness", backend=backend)
    paused = harness.start(request)
    assert isinstance(paused, Suspended)
    assert paused.recovery_verification.status is RecoveryStatus.TEMPORARILY_UNAVAILABLE

    backend.suspend_start = False
    restarted = InvestigationHarness(tmp_path / "harness", backend=backend)
    completed = restarted.resume(request)
    assert isinstance(completed, Completed)
    assert backend.start_calls == [
        (request.module_invocation_id, request.harness_invocation_id)
    ]
    assert backend.resume_calls == [
        (request.module_invocation_id, request.harness_invocation_id)
    ]

    replay = restarted.resume(request)
    assert isinstance(replay, Completed)
    assert len(backend.resume_calls) == 1


def test_harness_missing_recovery_is_explicitly_suspended(tmp_path: Path) -> None:
    """没有原始 Harness recovery 事实时不得创建替代运行。"""

    request = _request(module_invocation_id="missing-investigation")
    harness = InvestigationHarness(tmp_path / "empty-harness", backend=None)
    result = harness.resume(request)
    assert isinstance(result, Suspended)
    assert result.reason_code == "recovery_evidence_missing"
    assert result.recovery_verification.status is RecoveryStatus.RECOVERY_EVIDENCE_MISSING


def test_harness_rejects_input_identity_drift_on_resume(tmp_path: Path) -> None:
    """已完成 invocation 不能被另一份输入摘要冒充恢复。"""

    request = _request(module_invocation_id="identity-check")
    backend = DeterministicBackend()
    harness = InvestigationHarness(tmp_path / "harness", backend=backend)
    assert isinstance(harness.start(request), Completed)
    drifted = replace(request, module_input_digest="a" * 64)
    with pytest.raises(InvestigationHarnessIdentityConflict):
        harness.resume(drifted)
    assert len(backend.start_calls) == 1
    assert len(backend.resume_calls) == 0


class AlwaysSuspendedBackend:
    """让 Controller 恢复测试停在同一个 Open invocation。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def start(self, request: InvestigationHarnessRequest, harness_invocation_id: str) -> object:
        self.calls.append(("start", request.module_invocation_id))
        return BackendSuspended("provider_temporarily_unavailable")

    def resume(self, request: InvestigationHarnessRequest, harness_invocation_id: str) -> object:
        self.calls.append(("resume", request.module_invocation_id))
        return BackendSuspended("provider_temporarily_unavailable")


def test_research_workflow_controller_preserves_open_harness_invocation_on_retry(
    tmp_path: Path,
) -> None:
    """正式 Controller 只重试同一 canonical request，不创建第二个 invocation。"""

    memory = ResearchTaskWorkingMemory(tmp_path / "research_task.sqlite3")
    task_id = "research-task-controller-recovery"
    memory.create_research_task(
        research_task_id=task_id,
        contract_version="investigation_v2.task.v2",
        input={"scientific_question": "controller recovery"},
        policy={
            "entry_strategy": "standard_hypothesis_generation",
            "budgets": {"max_hypothesis_batches": 1},
        },
        dependencies={"catalog": {"ref": "catalog-1"}},
    )
    binding = CreateInvestigationContext("controller-context", "batch-1")
    digest = build_module_input_digest(task_id, "hypothesis-investigation", ())
    request = ModuleInvocationRequest(
        research_task_id=task_id,
        module_invocation_id="controller-investigation-1",
        module_kind="hypothesis-investigation",
        input_scientific_record_ids=(),
        module_input_digest=digest,
        context_binding=binding,
    )
    memory.start_module_invocation(task_id, 1, request)
    backend = AlwaysSuspendedBackend()
    harness = InvestigationHarness(tmp_path / "harness", backend=backend)
    # 先形成与生产 Controller 相同的 Harness Open recovery 事实。
    harness_request = InvestigationHarnessRequest(
        research_task_id=task_id,
        module_invocation_id=request.module_invocation_id,
        module_kind="hypothesis-investigation",
        input_scientific_record_ids=(),
        module_input_digest=digest,
        context_binding=binding.to_dict(),
    )
    assert isinstance(harness.start(harness_request), Suspended)

    controller = ResearchWorkflowController(
        memory,
        {"hypothesis-investigation": HarnessModuleRunner(harness)},
        require_entry_strategy=True,
    )
    result = controller.resume_research_task(task_id)
    assert isinstance(result, ControllerSuspended)
    assert result.module_invocation_id == request.module_invocation_id
    facts = memory.read_research_task_facts(task_id)
    assert facts.open_module_invocation is not None
    assert facts.open_module_invocation.module_invocation_id == request.module_invocation_id
    assert backend.calls == [("start", request.module_invocation_id), ("resume", request.module_invocation_id)]
