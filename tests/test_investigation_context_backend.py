"""Investigation Context 对 Top 5 子 Session 的确定性编排测试。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
import time
from typing import Any

from chemistry_multiagent.investigation.context_backend import (
    InvestigationContextRunner,
    InvestigationSessionTemporarilyUnavailable,
)
from chemistry_multiagent.investigation.context_store import (
    InvestigationContextIdentity,
    InvestigationContextStore,
    InvestigationContextStoreError,
    InvestigationSessionIdentity,
)
from chemistry_multiagent.investigation.harness_runtime import (
    BackendCompleted,
    BackendFatalFailure,
    BackendSuspended,
    Completed,
    FatalFailure,
    InvestigationHarness,
    InvestigationHarnessRequest,
    Suspended,
)
from chemistry_multiagent.investigation.research_material_store import (
    ResearchMaterialIntegrityConflict,
    ResearchMaterialMissing,
)
from chemistry_multiagent.investigation.recovery_contract import (
    Consistent,
    TemporarilyUnavailable,
    VerifiedIntegrityConflict,
)
from chemistry_multiagent.memory import (
    ResearchTaskWorkingMemory,
    ModuleInvocationRequest,
    PreparedModuleInvocationOutput,
    ScientificRecord,
)
from chemistry_multiagent.memory.v2_identities import (
    CreateInvestigationContext,
    ReuseInvestigationContext,
    StatelessModule,
)


def _record(
    *,
    research_task_id: str = "research-task-1",
    module_invocation_id: str,
    scientific_record_id: str,
    scientific_record_type: str,
    payload: dict[str, Any],
    source_scientific_record_ids: tuple[str, ...] = (),
    scientific_record_schema_version: str | int = "1",
) -> ScientificRecord:
    """构造测试使用的严格科学记录。"""

    return ScientificRecord.from_payload(
        scientific_record_id=scientific_record_id,
        research_task_id=research_task_id,
        scientific_record_type=scientific_record_type,
        scientific_record_schema_version=scientific_record_schema_version,
        producer_module_invocation_id=module_invocation_id,
        source_scientific_record_ids=source_scientific_record_ids,
        payload=payload,
    )


def _create_selected_hypotheses(
    service: ResearchTaskWorkingMemory,
) -> tuple[ScientificRecord, ...]:
    """写入一个 Batch、Top 5 选择摘要和五条逐假设记录。"""

    service.create_research_task(
        research_task_id="research-task-1",
        contract_version="investigation_v2.task.v1",
        input={"kind": "standard", "scientific_question": "question"},
        policy={"entry_strategy": "standard_hypothesis_generation"},
        dependencies={},
    )
    service.start_module_invocation(
        "research-task-1",
        1,
        ModuleInvocationRequest(
            research_task_id="research-task-1",
            module_invocation_id="hypothesis-generation-1",
            module_kind="hypothesis-generation",
            input_scientific_record_ids=(),
            module_input_digest="1" * 64,
            context_binding=StatelessModule(),
        ),
    )
    batch = _record(
        research_task_id="research-task-1",
        module_invocation_id="hypothesis-generation-1",
        scientific_record_id="hypothesis-batch-1",
        scientific_record_type="HypothesisBatchRecord",
        payload={
            "hypotheses": [
                {"hypothesis_id": f"h-{rank}", "claim": f"claim-{rank}"}
                for rank in range(1, 6)
            ],
            "rationale": "ranked",
            "verification_plan": "investigate",
        },
    )
    service.complete_module_invocation(
        "research-task-1",
        2,
        PreparedModuleInvocationOutput.from_records(
            module_invocation_id="hypothesis-generation-1",
            module_input_digest="1" * 64,
            output_scientific_records=(batch,),
        ),
    )
    service.start_module_invocation(
        "research-task-1",
        4,
        ModuleInvocationRequest(
            research_task_id="research-task-1",
            module_invocation_id="hypothesis-hypothesis-selection-1",
            module_kind="hypothesis-selection",
            input_scientific_record_ids=(batch.scientific_record_id,),
            module_input_digest="2" * 64,
            context_binding=StatelessModule(),
        ),
    )
    selection = _record(
        research_task_id="research-task-1",
        module_invocation_id="hypothesis-hypothesis-selection-1",
        scientific_record_id="selection-record-1",
        scientific_record_type="HypothesisSelectionRecord",
        scientific_record_schema_version="2",
        source_scientific_record_ids=(batch.scientific_record_id,),
        payload={
            "selected_hypotheses": [
                {"hypothesis_ref": f"h-{rank}", "rank": rank}
                for rank in range(1, 6)
            ],
            "selection_reason": "top five",
        },
    )
    selected_records = tuple(
        _record(
            research_task_id="research-task-1",
            module_invocation_id="hypothesis-hypothesis-selection-1",
            scientific_record_id=f"selected-hypothesis-{rank}",
            scientific_record_type="SelectedHypothesisRecord",
            source_scientific_record_ids=(batch.scientific_record_id, selection.scientific_record_id),
            payload={
                "hypothesis_ref": f"h-{rank}",
                "rank": rank,
                "hypothesis": {
                    "hypothesis_id": f"h-{rank}",
                    "claim": f"claim-{rank}",
                },
            },
        )
        for rank in range(1, 6)
    )
    service.complete_module_invocation(
        "research-task-1",
        5,
        PreparedModuleInvocationOutput.from_records(
            module_invocation_id="hypothesis-hypothesis-selection-1",
            module_input_digest="2" * 64,
            output_scientific_records=(selection, *selected_records),
        ),
    )
    return (selection, *selected_records)


@dataclass
class FakeSessionResolver:
    """记录无副作用身份规划与幂等外部资源准备。"""

    fail_ensure_once_for: str | None = None

    def __post_init__(self) -> None:
        self.plan_calls: list[str] = []
        self.ensure_calls: list[str] = []
        self._failed_ensure = False

    def plan_session(
        self,
        context: InvestigationContextIdentity,
        selected_hypothesis: ScientificRecord,
    ) -> InvestigationSessionIdentity:
        self.plan_calls.append(selected_hypothesis.scientific_record_id)
        rank = selected_hypothesis.payload["rank"]
        return InvestigationSessionIdentity(
            investigation_session_id=f"investigation-session-{rank}",
            investigation_context_id=context.investigation_context_id,
            research_task_id=context.research_task_id,
            hypothesis_execution_input_record_id=selected_hypothesis.scientific_record_id,
            workspace_identity=(
                f"internal/state/investigation_runtime/sessions/investigation-session-{rank}/"
                f"codex_sessions/investigations/{context.research_task_id}/"
                f"{context.investigation_context_id}/{selected_hypothesis.scientific_record_id}/workspace"
            ),
            minichem_scope_id=f"scope-{rank}",
            scope_contract_digest=str(rank) * 64,
        )

    def ensure_session(self, session: InvestigationSessionIdentity) -> None:
        self.ensure_calls.append(session.investigation_session_id)
        if (
            not self._failed_ensure
            and session.investigation_session_id == self.fail_ensure_once_for
        ):
            self._failed_ensure = True
            raise InvestigationSessionTemporarilyUnavailable(
                "session resources temporarily unavailable"
            )


class FakeSessionHarness:
    """一个 Session 的可重放 Harness；可在指定次数先返回暂停。"""

    def __init__(self, session_id: str, *, suspend_once: bool = False) -> None:
        self.session_id = session_id
        self.suspend_once = suspend_once
        self.calls: list[InvestigationHarnessRequest] = []

    def start(self, request: InvestigationHarnessRequest) -> Completed | Suspended:
        self.calls.append(request)
        if self.suspend_once and len(self.calls) == 1:
            return Suspended(
                f"child-{self.session_id}",
                "job_not_terminal",
                recovery_verification=TemporarilyUnavailable(
                    reason_code="job_not_terminal"
                ),
            )
        selected_ref = next(
            ref for ref in request.input_scientific_record_ids if ref.startswith("selected-hypothesis-")
        )
        suffix = f"{self.session_id}:{request.module_invocation_id}"
        evidence = _record(
            module_invocation_id=request.module_invocation_id,
            scientific_record_id=f"evidence:{suffix}",
            scientific_record_type="VerifiedEvidenceRecord",
            source_scientific_record_ids=request.input_scientific_record_ids,
            payload={
                "evidence_items": [{"value": self.session_id}],
                "artifact_refs": [],
                "source_manifest_ref": f"manifest:{suffix}",
                "summary": "verified",
                "verification": {"status": "verified"},
            },
        )
        result = _record(
            module_invocation_id=request.module_invocation_id,
            scientific_record_id=f"result:{suffix}",
            scientific_record_type="InvestigationResultRecord",
            source_scientific_record_ids=(evidence.scientific_record_id, *request.input_scientific_record_ids),
            payload={
                "conclusion": f"result for {selected_ref}",
                "self_judgment": "supported",
                "observations": ["observed"],
                "limitations": [],
                "evidence_ref": evidence.scientific_record_id,
            },
        )
        return Completed(
            f"child-{self.session_id}",
            PreparedModuleInvocationOutput.from_records(
                module_invocation_id=request.module_invocation_id,
                module_input_digest=request.module_input_digest,
                output_scientific_records=(evidence, result),
            ),
            recovery_verification=Consistent(reason_code="child_output_ready"),
        )


class FakeSessionHarnessFactory:
    """按持久 Session 身份返回同一个 fake Harness。"""

    def __init__(self, *, suspend_session_id: str | None = None) -> None:
        self.suspend_session_id = suspend_session_id
        self.harnesses: dict[str, FakeSessionHarness] = {}

    def open_session(
        self,
        session: InvestigationSessionIdentity,
    ) -> FakeSessionHarness:
        return self.harnesses.setdefault(
            session.investigation_session_id,
            FakeSessionHarness(
                session.investigation_session_id,
                suspend_once=session.investigation_session_id
                == self.suspend_session_id,
            ),
        )


def _backend(
    tmp_path: Path,
    service: ResearchTaskWorkingMemory,
    resolver: FakeSessionResolver,
    factory: FakeSessionHarnessFactory,
    *,
    max_parallel_investigations: int = 5,
) -> tuple[InvestigationContextRunner, InvestigationContextStore]:
    store = InvestigationContextStore(tmp_path / "context.sqlite3")
    return (
        InvestigationContextRunner(
            context_store=store,
            working_memory=service,
            session_owner=resolver,
            session_harness_owner=factory,
            max_parallel_investigations=max_parallel_investigations,
        ),
        store,
    )


def _initial_request(records: tuple[ScientificRecord, ...]) -> InvestigationHarnessRequest:
    return InvestigationHarnessRequest(
        research_task_id="research-task-1",
        module_invocation_id="investigation-initial",
        module_kind="hypothesis-investigation",
        input_scientific_record_ids=tuple(record.scientific_record_id for record in records),
        module_input_digest="a" * 64,
        context_binding=CreateInvestigationContext(
            investigation_context_id="context-1",
            hypothesis_batch_record_id="hypothesis-batch-1",
        ).to_dict(),
    )


def test_context_runs_one_session_per_selected_hypothesis_and_aggregates_pairs(
    tmp_path: Path,
) -> None:
    service = ResearchTaskWorkingMemory(tmp_path / "memory.sqlite3")
    selection = _create_selected_hypotheses(service)
    resolver = FakeSessionResolver()
    factory = FakeSessionHarnessFactory()
    backend, store = _backend(tmp_path, service, resolver, factory)
    request = _initial_request(selection)

    outcome = backend.start(request, request.harness_invocation_id)

    assert isinstance(outcome, BackendCompleted)
    assert [record.scientific_record_type for record in outcome.prepared_output.output_scientific_records] == [
        item
        for _rank in range(1, 6)
        for item in ("VerifiedEvidenceRecord", "InvestigationResultRecord")
    ]
    assert resolver.plan_calls == [
        f"selected-hypothesis-{rank}" for rank in range(1, 6)
    ]
    assert resolver.ensure_calls == [
        f"investigation-session-{rank}" for rank in range(1, 6)
    ]
    context = InvestigationContextIdentity("context-1", "research-task-1", "hypothesis-batch-1")
    snapshot = store.read_context(context)
    assert len(snapshot.sessions) == 5
    assert snapshot.open_invocation is None
    assert snapshot.closed_invocations[0].module_invocation_id == "investigation-initial"
    assert store.read_completed_output_digest(
        context, "investigation-initial"
    ) == outcome.prepared_output.module_output_digest


def test_context_honors_parallel_limit_and_keeps_target_output_order(
    tmp_path: Path,
) -> None:
    """多个 Session 并发运行但受配置上限约束，输出仍按 target 顺序。"""

    class Probe:
        def __init__(self) -> None:
            self.lock = threading.Lock()
            self.first_three = threading.Barrier(3)
            self.active = 0
            self.max_active = 0
            self.started = 0

    class ConcurrentHarness(FakeSessionHarness):
        def __init__(self, session_id: str, probe: Probe) -> None:
            super().__init__(session_id)
            self.probe = probe

        def start(self, request: InvestigationHarnessRequest) -> Completed | Suspended:
            with self.probe.lock:
                self.probe.active += 1
                self.probe.max_active = max(
                    self.probe.max_active,
                    self.probe.active,
                )
                self.probe.started += 1
                wait_for_first_three = self.probe.started <= 3
            try:
                if wait_for_first_three:
                    self.probe.first_three.wait(timeout=1)
                return super().start(request)
            finally:
                with self.probe.lock:
                    self.probe.active -= 1

    class ConcurrentFactory(FakeSessionHarnessFactory):
        def __init__(self, probe: Probe) -> None:
            super().__init__()
            self.probe = probe

        def open_session(
            self,
            session: InvestigationSessionIdentity,
        ) -> ConcurrentHarness:
            return self.harnesses.setdefault(
                session.investigation_session_id,
                ConcurrentHarness(session.investigation_session_id, self.probe),
            )

    service = ResearchTaskWorkingMemory(tmp_path / "memory.sqlite3")
    selection = _create_selected_hypotheses(service)
    resolver = FakeSessionResolver()
    probe = Probe()
    factory = ConcurrentFactory(probe)
    backend, _store = _backend(
        tmp_path,
        service,
        resolver,
        factory,
        max_parallel_investigations=3,
    )

    request = _initial_request(selection)
    outcome = backend.start(request, request.harness_invocation_id)

    assert isinstance(outcome, BackendCompleted)
    assert probe.max_active == 3
    results = [
        record
        for record in outcome.prepared_output.output_scientific_records
        if record.scientific_record_type == "InvestigationResultRecord"
    ]
    assert [record.payload["conclusion"] for record in results] == [
        f"result for selected-hypothesis-{rank}" for rank in range(1, 6)
    ]


def test_context_selects_multiple_failures_by_stable_target_order(
    tmp_path: Path,
) -> None:
    """多个子 Session 失败时按 target 顺序返回最早失败，而非完成顺序。"""

    class FailureFactory(FakeSessionHarnessFactory):
        def __init__(self) -> None:
            super().__init__()
            self.started: list[str] = []

        def open_session(
            self,
            session: InvestigationSessionIdentity,
        ) -> FakeSessionHarness:
            rank = int(session.investigation_session_id.rsplit("-", 1)[-1])

            class FailureHarness(FakeSessionHarness):
                def start(
                    inner_self,
                    request: InvestigationHarnessRequest,
                ) -> Completed | Suspended | FatalFailure:
                    self.started.append(inner_self.session_id)
                    # rank 4 先完成，但 rank 2 必须成为稳定选择的失败。
                    time.sleep(0.05 if rank == 2 else 0)
                    if rank in {2, 4}:
                        return FatalFailure(
                            f"child-{inner_self.session_id}",
                            "verified_integrity_conflict",
                            recovery_verification=VerifiedIntegrityConflict(
                                reason_code=f"session-{rank}-conflict"
                            ),
                        )
                    return super().start(request)

            return self.harnesses.setdefault(
                session.investigation_session_id,
                FailureHarness(session.investigation_session_id),
            )

    service = ResearchTaskWorkingMemory(tmp_path / "memory.sqlite3")
    selection = _create_selected_hypotheses(service)
    resolver = FakeSessionResolver()
    factory = FailureFactory()
    backend, _store = _backend(
        tmp_path,
        service,
        resolver,
        factory,
        max_parallel_investigations=5,
    )

    request = _initial_request(selection)
    outcome = backend.start(request, request.harness_invocation_id)

    assert isinstance(outcome, BackendFatalFailure)
    assert outcome.recovery_verification.reason_code == "session-2-conflict"
    assert factory.started == [
        f"investigation-session-{rank}" for rank in range(1, 6)
    ]


def test_missing_scientific_input_suspends_without_creating_context(
    tmp_path: Path,
) -> None:
    """输入记录暂不可见时保持可恢复，不创建猜测的 Context 或 Session。"""

    service = ResearchTaskWorkingMemory(tmp_path / "memory.sqlite3")
    service.create_research_task(
        research_task_id="research-task-1",
        contract_version="investigation_v2.task.v1",
        input={"kind": "standard", "scientific_question": "question"},
        policy={"entry_strategy": "standard_hypothesis_generation"},
        dependencies={},
    )
    resolver = FakeSessionResolver()
    factory = FakeSessionHarnessFactory()
    backend, _store = _backend(tmp_path, service, resolver, factory)
    request = InvestigationHarnessRequest(
        research_task_id="research-task-1",
        module_invocation_id="investigation-initial",
        module_kind="hypothesis-investigation",
        input_scientific_record_ids=("selected-hypothesis-missing",),
        module_input_digest="a" * 64,
        context_binding=CreateInvestigationContext(
            investigation_context_id="context-1",
            hypothesis_batch_record_id="hypothesis-batch-1",
        ).to_dict(),
    )

    outcome = backend.start(request, request.harness_invocation_id)

    assert isinstance(outcome, BackendSuspended)
    assert outcome.recovery_verification.reason_code == "scientific_record_missing"
    assert resolver.plan_calls == []
    assert resolver.ensure_calls == []


def test_context_resume_keeps_completed_session_and_does_not_recreate_scope(
    tmp_path: Path,
) -> None:
    service = ResearchTaskWorkingMemory(tmp_path / "memory.sqlite3")
    selection = _create_selected_hypotheses(service)
    resolver = FakeSessionResolver()
    factory = FakeSessionHarnessFactory(suspend_session_id="investigation-session-2")
    backend, store = _backend(tmp_path, service, resolver, factory)
    request = _initial_request(selection)

    first = backend.start(request, request.harness_invocation_id)
    resumed = backend.resume(request, request.harness_invocation_id)

    assert isinstance(first, BackendSuspended)
    assert isinstance(resumed, BackendCompleted)
    assert resolver.plan_calls == [
        f"selected-hypothesis-{rank}" for rank in range(1, 6)
    ]
    assert resolver.ensure_calls == [
        f"investigation-session-{rank}"
        for _attempt in range(2)
        for rank in range(1, 6)
    ]
    assert len(factory.harnesses["investigation-session-1"].calls) == 2
    assert len(factory.harnesses["investigation-session-2"].calls) == 2
    context = InvestigationContextIdentity("context-1", "research-task-1", "hypothesis-batch-1")
    assert store.read_context(context).open_invocation is None


def test_context_resume_does_not_reenter_completed_child_harness_backend(
    tmp_path: Path,
) -> None:
    """父 Context 暂停后，已密封的子 Session 只从 Harness Store 重放。"""

    class ChildBackend:
        """第二个 Session 首次暂停，其余调用形成确定性子输出。"""

        def __init__(self, session_id: str) -> None:
            self.session_id = session_id
            self.calls: list[str] = []

        def _complete(
            self, request: InvestigationHarnessRequest
        ) -> BackendCompleted:
            evidence = _record(
                module_invocation_id=request.module_invocation_id,
                scientific_record_id=f"evidence:{self.session_id}:{request.module_invocation_id}",
                scientific_record_type="VerifiedEvidenceRecord",
                source_scientific_record_ids=request.input_scientific_record_ids,
                payload={
                    "evidence_items": [{"value": self.session_id}],
                    "artifact_refs": [],
                    "source_manifest_ref": f"manifest:{self.session_id}",
                    "summary": "verified",
                    "verification": {"status": "verified"},
                },
            )
            result = _record(
                module_invocation_id=request.module_invocation_id,
                scientific_record_id=f"result:{self.session_id}:{request.module_invocation_id}",
                scientific_record_type="InvestigationResultRecord",
                source_scientific_record_ids=(
                    evidence.scientific_record_id,
                    *request.input_scientific_record_ids,
                ),
                payload={
                    "conclusion": f"result for {self.session_id}",
                    "self_judgment": "supported",
                    "observations": ["observed"],
                    "limitations": [],
                    "evidence_ref": evidence.scientific_record_id,
                },
            )
            return BackendCompleted(
                PreparedModuleInvocationOutput.from_records(
                    module_invocation_id=request.module_invocation_id,
                    module_input_digest=request.module_input_digest,
                    output_scientific_records=(evidence, result),
                )
            )

        def start(
            self,
            request: InvestigationHarnessRequest,
            _harness_run_ref: str,
        ) -> BackendCompleted | BackendSuspended:
            self.calls.append("start")
            if self.session_id == "investigation-session-2":
                return BackendSuspended(
                    "job_not_terminal",
                    recovery_verification=TemporarilyUnavailable(
                        reason_code="job_not_terminal"
                    ),
                )
            return self._complete(request)

        def resume(
            self,
            request: InvestigationHarnessRequest,
            _harness_run_ref: str,
        ) -> BackendCompleted:
            self.calls.append("resume")
            return self._complete(request)

    class HarnessFactory:
        """为每个 Session 保留同一个真实 Harness facade 和 backend。"""

        def __init__(self) -> None:
            self.backends: dict[str, ChildBackend] = {}
            self.harnesses: dict[str, InvestigationHarness] = {}

        def open_session(
            self, session: InvestigationSessionIdentity
        ) -> InvestigationHarness:
            backend = self.backends.setdefault(
                session.investigation_session_id,
                ChildBackend(session.investigation_session_id),
            )
            return self.harnesses.setdefault(
                session.investigation_session_id,
                InvestigationHarness(
                    tmp_path
                    / "sessions"
                    / session.investigation_session_id,
                    backend=backend,
                ),
            )

    service = ResearchTaskWorkingMemory(tmp_path / "memory.sqlite3")
    selection = _create_selected_hypotheses(service)
    resolver = FakeSessionResolver()
    factory = HarnessFactory()
    backend = InvestigationContextRunner(
        context_store=InvestigationContextStore(tmp_path / "context.sqlite3"),
        working_memory=service,
        session_owner=resolver,
        session_harness_owner=factory,
    )
    request = _initial_request(selection)

    interrupted = backend.start(request, request.harness_invocation_id)
    resumed = backend.resume(request, request.harness_invocation_id)

    assert isinstance(interrupted, BackendSuspended)
    assert isinstance(resumed, BackendCompleted)
    assert factory.backends["investigation-session-1"].calls == ["start"]
    assert factory.backends["investigation-session-2"].calls == [
        "start",
        "resume",
    ]
    assert all(
        factory.backends[f"investigation-session-{rank}"].calls == ["start"]
        for rank in range(3, 6)
    )


def test_session_identity_and_invocation_are_persisted_before_scope_ensure(
    tmp_path: Path,
) -> None:
    """scope 准备前中断时，恢复使用已保存 Session 身份而不重新规划。"""

    service = ResearchTaskWorkingMemory(tmp_path / "memory.sqlite3")
    selection = _create_selected_hypotheses(service)
    resolver = FakeSessionResolver(
        fail_ensure_once_for="investigation-session-1"
    )
    factory = FakeSessionHarnessFactory()
    backend, store = _backend(tmp_path, service, resolver, factory)
    request = _initial_request(selection)

    interrupted = backend.start(request, request.harness_invocation_id)
    context = InvestigationContextIdentity(
        "context-1", "research-task-1", "hypothesis-batch-1"
    )
    snapshot_after_failure = store.read_context(context)
    resumed = backend.resume(request, request.harness_invocation_id)

    assert isinstance(interrupted, BackendSuspended)
    assert interrupted.reason_code == "investigation_session_unavailable"
    assert len(snapshot_after_failure.sessions) == 5
    assert snapshot_after_failure.open_invocation is not None
    assert resolver.plan_calls == [
        f"selected-hypothesis-{rank}" for rank in range(1, 6)
    ]
    assert resolver.ensure_calls == [
        "investigation-session-1",
        *(f"investigation-session-{rank}" for rank in range(1, 6)),
    ]
    assert isinstance(resumed, BackendCompleted)
    assert all(len(harness.calls) == 1 for harness in factory.harnesses.values())


def test_missing_research_material_suspends_before_child_harness(
    tmp_path: Path,
) -> None:
    """已确认材料的 CAS 字节缺失时保持可恢复，且不启动任何子 Harness。"""

    class MissingMaterialResolver(FakeSessionResolver):
        def ensure_session(self, session: InvestigationSessionIdentity) -> None:
            self.ensure_calls.append(session.investigation_session_id)
            raise ResearchMaterialMissing("material bytes missing")

    service = ResearchTaskWorkingMemory(tmp_path / "memory.sqlite3")
    selection = _create_selected_hypotheses(service)
    resolver = MissingMaterialResolver()
    factory = FakeSessionHarnessFactory()
    backend, _store = _backend(tmp_path, service, resolver, factory)

    request = _initial_request(selection)
    outcome = backend.start(request, request.harness_invocation_id)

    assert isinstance(outcome, BackendSuspended)
    assert outcome.reason_code == "recovery_evidence_missing"
    assert outcome.recovery_verification.reason_code == "research_material_missing"
    assert factory.harnesses == {}


def test_research_material_digest_drift_fails_before_child_harness(
    tmp_path: Path,
) -> None:
    """材料摘要冲突是已核实完整性失败，不能继续创建或驱动子 Harness。"""

    class DriftedMaterialResolver(FakeSessionResolver):
        def ensure_session(self, session: InvestigationSessionIdentity) -> None:
            self.ensure_calls.append(session.investigation_session_id)
            raise ResearchMaterialIntegrityConflict("material digest drift")

    service = ResearchTaskWorkingMemory(tmp_path / "memory.sqlite3")
    selection = _create_selected_hypotheses(service)
    resolver = DriftedMaterialResolver()
    factory = FakeSessionHarnessFactory()
    backend, _store = _backend(tmp_path, service, resolver, factory)

    request = _initial_request(selection)
    outcome = backend.start(request, request.harness_invocation_id)

    assert isinstance(outcome, BackendFatalFailure)
    assert outcome.failure_code == "verified_integrity_conflict"
    assert outcome.recovery_verification.reason_code == (
        "research_material_integrity_conflict"
    )
    assert factory.harnesses == {}


def test_context_completion_write_failure_resumes_from_child_outputs(
    tmp_path: Path,
) -> None:
    """子 Session 已完成但 Context 收尾写失败时，恢复不重建 Session/scope。"""

    class FailOnceContextStore(InvestigationContextStore):
        """第一次完成 Invocation 前模拟持久化暂时不可用。"""

        failed = False

        def complete_invocation(self, *args: Any, **kwargs: Any):
            if not self.failed:
                self.failed = True
                raise InvestigationContextStoreError(
                    "temporary context completion failure"
                )
            return super().complete_invocation(*args, **kwargs)

    service = ResearchTaskWorkingMemory(tmp_path / "memory.sqlite3")
    selection = _create_selected_hypotheses(service)
    resolver = FakeSessionResolver()
    factory = FakeSessionHarnessFactory()
    store = FailOnceContextStore(tmp_path / "context.sqlite3")
    backend = InvestigationContextRunner(
        context_store=store,
        working_memory=service,
        session_owner=resolver,
        session_harness_owner=factory,
    )
    request = _initial_request(selection)

    interrupted = backend.start(request, request.harness_invocation_id)
    context = InvestigationContextIdentity(
        "context-1", "research-task-1", "hypothesis-batch-1"
    )
    open_after_failure = store.read_context(context).open_invocation
    resumed = backend.resume(request, request.harness_invocation_id)

    assert isinstance(interrupted, BackendSuspended)
    assert interrupted.reason_code == "context_store_unavailable"
    assert open_after_failure is not None
    assert isinstance(resumed, BackendCompleted)
    assert resolver.plan_calls == [
        f"selected-hypothesis-{rank}" for rank in range(1, 6)
    ]
    assert resolver.ensure_calls == [
        f"investigation-session-{rank}"
        for _attempt in range(2)
        for rank in range(1, 6)
    ]
    assert all(
        len(factory.harnesses[f"investigation-session-{rank}"].calls) == 2
        for rank in range(1, 6)
    )
    assert store.read_context(context).open_invocation is None


def test_repair_invocation_targets_only_original_session(tmp_path: Path) -> None:
    service = ResearchTaskWorkingMemory(tmp_path / "memory.sqlite3")
    selection = _create_selected_hypotheses(service)
    resolver = FakeSessionResolver()
    factory = FakeSessionHarnessFactory()
    backend, _store = _backend(tmp_path, service, resolver, factory)
    initial_request = _initial_request(selection)
    initial = backend.start(initial_request, initial_request.harness_invocation_id)
    assert isinstance(initial, BackendCompleted)

    facts = service.read_research_task_facts("research-task-1")
    service.start_module_invocation(
        "research-task-1",
        facts.research_task_sequence,
        ModuleInvocationRequest(
            research_task_id="research-task-1",
            module_invocation_id="investigation-initial",
            module_kind="hypothesis-investigation",
            input_scientific_record_ids=initial_request.input_scientific_record_ids,
            module_input_digest=initial_request.module_input_digest,
            context_binding=CreateInvestigationContext(
                investigation_context_id="context-1",
                hypothesis_batch_record_id="hypothesis-batch-1",
            ),
        ),
    )
    service.complete_module_invocation(
        "research-task-1",
        service.read_research_task_facts("research-task-1").research_task_sequence,
        initial.prepared_output,
    )
    result_refs = tuple(
        record.scientific_record_id
        for record in initial.prepared_output.output_scientific_records
        if record.scientific_record_type == "InvestigationResultRecord"
    )
    service.start_module_invocation(
        "research-task-1",
        service.read_research_task_facts("research-task-1").research_task_sequence,
        ModuleInvocationRequest(
            research_task_id="research-task-1",
            module_invocation_id="review-1",
            module_kind="task-review",
            input_scientific_record_ids=result_refs,
            module_input_digest="3" * 64,
            context_binding=StatelessModule(),
        ),
    )
    review = _record(
        module_invocation_id="review-1",
        scientific_record_id="review-record-1",
        scientific_record_type="ReviewRecord",
        source_scientific_record_ids=result_refs,
        payload={
            "reviewed_result_refs": list(result_refs),
            "findings": [{"result_ref": result_refs[0]}],
            "outcome": "RepairRequired",
        },
    )
    repair = _record(
        module_invocation_id="review-1",
        scientific_record_id="repair-request-1",
        scientific_record_type="RepairRequestRecord",
        source_scientific_record_ids=(review.scientific_record_id, result_refs[0]),
        payload={
            "result_ref": result_refs[0],
            "problem": "needs more evidence",
            "instructions": "continue the same investigation",
        },
    )
    service.complete_module_invocation(
        "research-task-1",
        service.read_research_task_facts("research-task-1").research_task_sequence,
        PreparedModuleInvocationOutput.from_records(
            module_invocation_id="review-1",
            module_input_digest="3" * 64,
            output_scientific_records=(review, repair),
        ),
    )
    service.start_module_invocation(
        "research-task-1",
        service.read_research_task_facts("research-task-1").research_task_sequence,
        ModuleInvocationRequest(
            research_task_id="research-task-1",
            module_invocation_id="investigation-repair-1",
            module_kind="hypothesis-investigation",
            input_scientific_record_ids=(review.scientific_record_id, repair.scientific_record_id, result_refs[0]),
            module_input_digest="b" * 64,
            context_binding=ReuseInvestigationContext(
                investigation_context_id="context-1",
                hypothesis_batch_record_id="hypothesis-batch-1",
                source_module_invocation_id="investigation-initial",
            ),
        ),
    )
    repair_request = InvestigationHarnessRequest(
        research_task_id="research-task-1",
        module_invocation_id="investigation-repair-1",
        module_kind="hypothesis-investigation",
        input_scientific_record_ids=(review.scientific_record_id, repair.scientific_record_id, result_refs[0]),
        module_input_digest="b" * 64,
        context_binding=ReuseInvestigationContext(
            investigation_context_id="context-1",
            hypothesis_batch_record_id="hypothesis-batch-1",
            source_module_invocation_id="investigation-initial",
        ).to_dict(),
    )

    repaired = backend.start(repair_request, repair_request.harness_invocation_id)

    assert isinstance(repaired, BackendCompleted)
    assert resolver.plan_calls == [
        f"selected-hypothesis-{rank}" for rank in range(1, 6)
    ]
    assert resolver.ensure_calls == [
        *(f"investigation-session-{rank}" for rank in range(1, 6)),
        "investigation-session-1",
    ]
    assert len(factory.harnesses["investigation-session-1"].calls) == 2
    assert len(factory.harnesses["investigation-session-2"].calls) == 1
    assert all(
        len(factory.harnesses[f"investigation-session-{rank}"].calls) == 1
        for rank in range(2, 6)
    )
    repaired_result = next(
        record
        for record in repaired.prepared_output.output_scientific_records
        if record.scientific_record_type == "InvestigationResultRecord"
    )
    assert result_refs[0] in repaired_result.source_scientific_record_ids
    assert "selected-hypothesis-1" in repaired_result.source_scientific_record_ids
