"""V2 外层 Controller 与内层 Harness 的中断恢复交叉测试。

这些测试只使用内存 provider、execution 和 scope owner。每个测试都模拟进程在
一个明确的持久化窗口退出，再用新的对象实例从相同目录恢复；不会连接真实模型、
MCP 或提交化学作业。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from chemistry_multiagent.controllers.investigation_v2_module_controller import (
    ControllerSuspended,
    HarnessModuleRunner,
    ResearchWorkflowController,
    ModuleCompleted,
    ModuleInvocationRequest as ControllerModuleInvocationRequest,
    ResearchTaskClosedResult,
)
from chemistry_multiagent.investigation.context_backend import (
    InvestigationContextRunner,
    InvestigationSessionTemporarilyUnavailable,
)
from chemistry_multiagent.investigation.context_store import (
    InvestigationContextIdentity,
    InvestigationContextStore,
    InvestigationSessionIdentity,
)
from chemistry_multiagent.investigation.harness_recovery_store import (
    HarnessLifecycle,
    HarnessRecoveryStore,
    HarnessRunIdentity,
    SessionReference,
)
from chemistry_multiagent.investigation.harness_runtime import (
    BackendCompleted,
    InvestigationHarness,
    InvestigationHarnessRequest,
    Suspended,
)
from chemistry_multiagent.investigation.investigation_backend import (
    NativeAction,
    NativeTurnResult,
    InvestigationSessionRunner,
)
from chemistry_multiagent.investigation.recovery_contract import (
    TemporarilyUnavailable,
)
from chemistry_multiagent.memory import (
    ResearchTaskWorkingMemory,
    ModuleInvocationRequest,
    PreparedModuleInvocationOutput,
    ScientificRecord,
)
from chemistry_multiagent.memory.v2_identities import (
    CreateInvestigationContext,
    StatelessModule,
)


def _record(
    *,
    research_task_id: str,
    module_invocation_id: str,
    scientific_record_id: str,
    scientific_record_type: str,
    payload: dict[str, Any],
    source_scientific_record_ids: tuple[str, ...] = (),
) -> ScientificRecord:
    """构造可写入 V2 ledger 的公开科学记录。"""

    return ScientificRecord.from_payload(
        scientific_record_id=scientific_record_id,
        research_task_id=research_task_id,
        scientific_record_type=scientific_record_type,
        scientific_record_schema_version="2",
        producer_module_invocation_id=module_invocation_id,
        source_scientific_record_ids=source_scientific_record_ids,
        payload=payload,
    )


def _minimal_investigation_output(
    request: ControllerModuleInvocationRequest,
) -> PreparedModuleInvocationOutput:
    """构造一个 Review 可消费的公开 Investigation pair。"""

    evidence = _record(
        research_task_id=request.research_task_id,
        module_invocation_id=request.module_invocation_id,
        scientific_record_id=f"{request.module_invocation_id}:evidence",
        scientific_record_type="VerifiedEvidenceRecord",
        source_scientific_record_ids=request.input_scientific_record_ids,
        payload={
            "evidence_items": [{"status": "verified"}],
            "artifact_refs": [],
            "source_manifest_ref": "manifest-test",
            "summary": "deterministic evidence",
            "verification": {"status": "verified"},
        },
    )
    result = _record(
        research_task_id=request.research_task_id,
        module_invocation_id=request.module_invocation_id,
        scientific_record_id=f"{request.module_invocation_id}:result",
        scientific_record_type="InvestigationResultRecord",
        source_scientific_record_ids=(evidence.scientific_record_id, *request.input_scientific_record_ids),
        payload={
            "conclusion": "deterministic result",
            "self_judgment": "supported",
            "observations": ["observed"],
            "limitations": [],
            "evidence_ref": evidence.scientific_record_id,
        },
    )
    return PreparedModuleInvocationOutput.from_records(
        module_invocation_id=request.module_invocation_id,
        module_input_digest=request.module_input_digest,
        output_scientific_records=(evidence, result),
    )


class _AcceptedReview:
    """不访问外部服务的终态 Review adapter。"""

    def start(self, request: ControllerModuleInvocationRequest) -> ModuleCompleted:
        return self._complete(request)

    def resume(self, request: ControllerModuleInvocationRequest) -> ModuleCompleted:
        return self._complete(request)

    @staticmethod
    def _complete(request: ControllerModuleInvocationRequest) -> ModuleCompleted:
        review = _record(
            research_task_id=request.research_task_id,
            module_invocation_id=request.module_invocation_id,
            scientific_record_id=f"{request.module_invocation_id}:review",
            scientific_record_type="ReviewRecord",
            source_scientific_record_ids=request.input_scientific_record_ids,
            payload={
                "reviewed_result_refs": list(request.input_scientific_record_ids),
                "findings": [{"kind": "accepted"}],
                "outcome": "Accepted",
            },
        )
        answer = _record(
            research_task_id=request.research_task_id,
            module_invocation_id=request.module_invocation_id,
            scientific_record_id=f"{request.module_invocation_id}:answer",
            scientific_record_type="FinalAnswerRecord",
            source_scientific_record_ids=(review.scientific_record_id,),
            payload={"answer": "supported", "answerability": "answerable"},
        )
        return ModuleCompleted(
            PreparedModuleInvocationOutput.from_records(
                module_invocation_id=request.module_invocation_id,
                module_input_digest=request.module_input_digest,
                output_scientific_records=(review, answer),
            )
        )


class _PrefixModule:
    """只用于建立 Selection 前的最小账本，不模拟模型。"""

    def start(self, request: ControllerModuleInvocationRequest) -> ModuleCompleted:
        return ModuleCompleted(self._output(request))

    def resume(self, request: ControllerModuleInvocationRequest) -> ModuleCompleted:
        return ModuleCompleted(self._output(request))

    @staticmethod
    def _output(
        request: ControllerModuleInvocationRequest,
    ) -> PreparedModuleInvocationOutput:
        if request.module_kind == "retrieval":
            records = (
                _record(
                    research_task_id=request.research_task_id,
                    module_invocation_id=request.module_invocation_id,
                    scientific_record_id=f"{request.module_invocation_id}:retrieval",
                    scientific_record_type="RetrievalRecord",
                    payload={"summary": "retrieved", "citations": [], "corpus_snapshot_ref": "corpus"},
                ),
            )
        elif request.module_kind == "research-query-formulation":
            records = (
                _record(
                    research_task_id=request.research_task_id,
                    module_invocation_id=request.module_invocation_id,
                    scientific_record_id=f"{request.module_invocation_id}:query",
                    scientific_record_type="QueryRecord",
                    source_scientific_record_ids=request.input_scientific_record_ids,
                    payload={"query": "query", "research_angle": "angle", "constraints": {}},
                ),
            )
        elif request.module_kind == "hypothesis-generation":
            batch_id = f"{request.module_invocation_id}:batch"
            batch = _record(
                research_task_id=request.research_task_id,
                module_invocation_id=request.module_invocation_id,
                scientific_record_id=batch_id,
                scientific_record_type="HypothesisBatchRecord",
                source_scientific_record_ids=request.input_scientific_record_ids,
                payload={
                    "hypotheses": [
                        {"hypothesis_id": f"h-{rank}", "claim": f"claim-{rank}"}
                        for rank in range(1, 6)
                    ],
                    "rationale": "deterministic",
                    "verification_plan": "investigate",
                },
            )
            records = (batch,)
        elif request.module_kind == "hypothesis-selection":
            batch_ref = request.input_scientific_record_ids[0]
            selection = _record(
                research_task_id=request.research_task_id,
                module_invocation_id=request.module_invocation_id,
                scientific_record_id=f"{request.module_invocation_id}:selection",
                scientific_record_type="HypothesisSelectionRecord",
                source_scientific_record_ids=request.input_scientific_record_ids,
                payload={
                    "selected_hypotheses": [
                        {"hypothesis_ref": f"h-{rank}", "rank": rank}
                        for rank in range(1, 6)
                    ],
                    "selection_reason": "top five",
                },
            )
            selected = tuple(
                _record(
                    research_task_id=request.research_task_id,
                    module_invocation_id=request.module_invocation_id,
                    scientific_record_id=f"{request.module_invocation_id}:selected-{rank}",
                    scientific_record_type="SelectedHypothesisRecord",
                    source_scientific_record_ids=(batch_ref, selection.scientific_record_id),
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
            records = (selection, *selected)
        else:  # pragma: no cover - fixture programming error
            raise AssertionError(request.module_kind)
        return PreparedModuleInvocationOutput.from_records(
            module_invocation_id=request.module_invocation_id,
            module_input_digest=request.module_input_digest,
            output_scientific_records=records,
        )


def _seed_selection_task(service: ResearchTaskWorkingMemory) -> None:
    """通过真实 Controller 建立标准入口的四个已闭合模块。"""

    service.create_research_task(
        research_task_id="task-cross-layer",
        contract_version="investigation_v2.task.v1",
        input={"scientific_question": "cross layer resume"},
        policy={
            "stage_architecture_version": "investigation_v2",
            "entry_strategy": "standard_hypothesis_generation",
        },
        dependencies={},
    )
    refs: tuple[str, ...] = ()
    for ordinal, kind in enumerate(
        (
            "retrieval",
            "research-query-formulation",
            "hypothesis-generation",
            "hypothesis-selection",
        ),
        1,
    ):
        request = ControllerModuleInvocationRequest(
            research_task_id="task-cross-layer",
            module_invocation_id=f"{kind}-{ordinal}",
            module_kind=kind,
            input_scientific_record_ids=refs,
            module_input_digest=(f"{ordinal:064x}"),
            context_binding=StatelessModule(),
        )
        facts = service.read_research_task_facts("task-cross-layer")
        service.start_module_invocation(
            "task-cross-layer",
            facts.research_task_sequence,
            ModuleInvocationRequest(
                research_task_id="task-cross-layer",
                module_invocation_id=request.module_invocation_id,
                module_kind=request.module_kind,
                input_scientific_record_ids=request.input_scientific_record_ids,
                module_input_digest=request.module_input_digest,
                context_binding=StatelessModule(),
            ),
        )
        output = _PrefixModule._output(request)
        service.complete_module_invocation(
            "task-cross-layer",
            service.read_research_task_facts("task-cross-layer").research_task_sequence,
            output,
        )
        refs = tuple(record.scientific_record_id for record in output.output_scientific_records)


@dataclass
class _SessionResolver:
    """只生成稳定 Session 身份；不连接外部服务。"""

    ensure_fail_once: bool = False

    def __post_init__(self) -> None:
        self.plans = 0
        self.ensure_calls: list[str] = []
        self._failed = False

    def plan_session(
        self,
        context: InvestigationContextIdentity,
        selected_hypothesis: ScientificRecord,
    ) -> InvestigationSessionIdentity:
        rank = selected_hypothesis.payload["rank"]
        self.plans += 1
        return InvestigationSessionIdentity(
            investigation_session_id=f"session-{rank}",
            investigation_context_id=context.investigation_context_id,
            research_task_id=context.research_task_id,
            hypothesis_execution_input_record_id=selected_hypothesis.scientific_record_id,
            workspace_identity=(
                f"internal/state/investigation_runtime/sessions/session-{rank}/codex_sessions/"
                f"investigations/{context.research_task_id}/{context.investigation_context_id}/"
                f"{selected_hypothesis.scientific_record_id}/workspace"
            ),
            minichem_scope_id=f"scope-{rank}",
            scope_contract_digest=f"{rank}" * 64,
        )

    def ensure_session(self, session: InvestigationSessionIdentity) -> None:
        self.ensure_calls.append(session.investigation_session_id)
        if self.ensure_fail_once and not self._failed:
            self._failed = True
            raise InvestigationSessionTemporarilyUnavailable("scope owner unavailable")


@dataclass
class _SessionExecution:
    investigation_session_id: str
    status: str = "pending"

    def __post_init__(self) -> None:
        self.submissions: list[str] = []
        self.queries: list[str] = []
        self.collections: list[str] = []

    @property
    def job_id(self) -> str:
        return f"job-{self.investigation_session_id.removeprefix('session-')}"

    def submit_job(self, action: NativeAction, *, session: Any) -> dict[str, Any]:
        del action
        assert session.investigation_session_id == self.investigation_session_id
        self.submissions.append(self.job_id)
        return {
            "job_id": self.job_id,
            "operation_id": f"operation-{self.investigation_session_id}",
            "submission_key": f"submission-{self.investigation_session_id}",
            "request_digest": "b" * 64,
            "status": self.status,
        }

    def query_job(self, job: Any, *, session: Any) -> dict[str, Any]:
        assert session.investigation_session_id == self.investigation_session_id
        self.queries.append(job.minichem_job_id)
        return {"job_id": job.minichem_job_id, "status": self.status}

    def collect_job(self, job: Any, *, session: Any) -> dict[str, Any]:
        assert session.investigation_session_id == self.investigation_session_id
        self.collections.append(job.minichem_job_id)
        return {
            "job_id": job.minichem_job_id,
            "operation_id": job.minichem_operation_id,
            "submission_key": job.minichem_submission_key,
            "status": self.status,
        }


class _SessionProvider:
    """每个 Session 首次提交一次作业，后续 turn 形成 Result。"""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.turn_counts: dict[str, int] = {}

    def run_turn(
        self,
        request: dict[str, Any],
        *,
        session: Any,
        action_results: tuple[dict[str, Any], ...],
        is_resume: bool,
        thread_id: str | None = None,
    ) -> NativeTurnResult:
        del request
        count = self.turn_counts.get(session.investigation_session_id, 0) + 1
        self.turn_counts[session.investigation_session_id] = count
        thread = f"thread-{session.investigation_session_id}"
        self.calls.append(
            {
                "investigation_session_id": session.investigation_session_id,
                "count": count,
                "is_resume": is_resume,
                "thread_id": thread_id,
                "action_results": action_results,
            }
        )
        if count == 1:
            return NativeTurnResult(
                thread_id=thread,
                turn_id=f"{session.investigation_session_id}-turn-1",
                actions=(
                    NativeAction(
                        f"submit-{session.investigation_session_id}",
                        "submit_job",
                        {"kind": "gaussian"},
                    ),
                ),
            )
        assert action_results
        job_id = f"job-{session.investigation_session_id.removeprefix('session-')}"
        return NativeTurnResult(
            thread_id=thread,
            turn_id=f"{session.investigation_session_id}-turn-{count}",
            result={
                "self_judgment": "supported",
                "narrative": f"result for {session.investigation_session_id}",
                "job_refs": [job_id],
                "operation_refs": [f"operation-{session.investigation_session_id}"],
                "artifact_refs": [],
                "observations": ["observed"],
                "limitations": [],
            },
        )


class _ChildHarnessFactory:
    """为 Context 中每个 Session 创建持久 native Harness。"""

    def __init__(self, root: Path, provider: _SessionProvider) -> None:
        self.root = root
        self.provider = provider
        self.executions: dict[str, _SessionExecution] = {}
        self.harnesses: dict[str, InvestigationHarness] = {}

    def open_session(self, session: InvestigationSessionIdentity) -> InvestigationHarness:
        execution = self.executions.setdefault(
            session.investigation_session_id,
            _SessionExecution(session.investigation_session_id),
        )
        if session.investigation_session_id in self.harnesses:
            return self.harnesses[session.investigation_session_id]
        recovery_store = HarnessRecoveryStore(
            self.root / session.investigation_session_id / "recovery.sqlite3"
        )
        backend = InvestigationSessionRunner(
            recovery_store=recovery_store,
            provider_transport=self.provider,
            execution_transport=execution,
            session_binder=_SessionBinder(session),
        )
        harness = InvestigationHarness(
            self.root / session.investigation_session_id / "runtime",
            backend=backend,
            recovery_store=recovery_store,
        )
        self.harnesses[session.investigation_session_id] = harness
        return harness


class _SessionBinder:
    """把 Context 的公开 Session 身份绑定到 Harness 私有恢复引用。"""

    def __init__(self, session: InvestigationSessionIdentity) -> None:
        self.session = session

    def bind_session(
        self,
        request: InvestigationHarnessRequest,
        *,
        harness_invocation_id: str,
        existing_session: SessionReference | None,
    ) -> SessionReference:
        del harness_invocation_id
        if existing_session is not None:
            return existing_session
        return SessionReference(
            research_task_id=request.research_task_id,
            investigation_context_id=self.session.investigation_context_id,
            hypothesis_execution_input_record_id=(
                self.session.hypothesis_execution_input_record_id
            ),
            investigation_session_id=self.session.investigation_session_id,
            workspace_identity=self.session.workspace_identity.replace(
                "/", "-"
            ),
            minichem_scope_id=self.session.minichem_scope_id,
            scope_contract_digest=self.session.scope_contract_digest,
            capability_generation_digest=(
                self.session.capability_generation_digest
                or self.session.scope_contract_digest
            ),
        )


def test_outer_controller_resumes_open_module_after_inner_harness_interruption(
    tmp_path: Path,
) -> None:
    """外层 WM 保持 Open，内层 provider 中断后恢复同一 thread 且不重复提交。"""

    service = ResearchTaskWorkingMemory(tmp_path / "memory.sqlite3")
    _seed_selection_task(service)
    provider = _SessionProvider()
    resolver = _SessionResolver()
    child_factory = _ChildHarnessFactory(tmp_path / "children", provider)
    context_backend = InvestigationContextRunner(
        context_store=InvestigationContextStore(tmp_path / "context.sqlite3"),
        working_memory=service,
        session_owner=resolver,
        session_harness_owner=child_factory,
    )
    parent_store = HarnessRecoveryStore(tmp_path / "parent-recovery.sqlite3")
    parent = InvestigationHarness(
        tmp_path / "parent-runtime",
        backend=context_backend,
        recovery_store=parent_store,
    )

    modules = {
        "hypothesis-investigation": HarnessModuleRunner(parent),
        "task-review": _AcceptedReview(),
    }
    controller = ResearchWorkflowController(service, modules, require_entry_strategy=True)

    first = controller.resume_research_task("task-cross-layer")
    assert isinstance(first, ControllerSuspended)
    facts = service.read_research_task_facts("task-cross-layer")
    assert facts.lifecycle.state == "Open"
    assert facts.open_module_invocation is not None
    assert provider.turn_counts["session-1"] == 1
    assert child_factory.executions["session-1"].submissions == ["job-1"]

    # 外层进程退出后，只有 owner 推进原 job；下一次 controller 使用新对象恢复。
    # 每一轮都把已发现的 owner 状态推进到 completed；新发现的 Session 会在下一轮
    # 再被推进，模拟五条独立调查在不同时间恢复的情况。
    resumed_controller = ResearchWorkflowController(
        service,
        {
            "hypothesis-investigation": HarnessModuleRunner(
                InvestigationHarness(
                    tmp_path / "parent-runtime",
                    backend=context_backend,
                    recovery_store=HarnessRecoveryStore(tmp_path / "parent-recovery.sqlite3"),
                )
            ),
            "task-review": _AcceptedReview(),
        },
        require_entry_strategy=True,
    )
    resumed: Any = None
    for _attempt in range(6):
        for execution in child_factory.executions.values():
            execution.status = "completed"
        resumed = resumed_controller.resume_research_task("task-cross-layer")
        if isinstance(resumed, ResearchTaskClosedResult):
            break
        assert isinstance(resumed, ControllerSuspended)
    assert isinstance(resumed, ResearchTaskClosedResult)
    assert child_factory.executions["session-1"].submissions == ["job-1"]
    assert child_factory.executions["session-1"].queries == ["job-1"]
    assert child_factory.executions["session-1"].collections == ["job-1"]
    session_one_calls = [
        call for call in provider.calls if call["investigation_session_id"] == "session-1"
    ]
    assert len(session_one_calls) == 2
    assert session_one_calls[1]["thread_id"] == "thread-session-1"
    assert session_one_calls[1]["is_resume"] is True
    assert resolver.plans == 5
    assert all(
        execution.submissions == [execution.job_id]
        and execution.queries in ([], [execution.job_id])
        and execution.collections == [execution.job_id]
        for execution in child_factory.executions.values()
    )


def test_controller_resume_after_backend_exception_keeps_same_module_invocation(
    tmp_path: Path,
) -> None:
    """Harness 进程在 backend 返回前退出时，下一次只调用原 Module Run 的 resume。"""

    service = ResearchTaskWorkingMemory(tmp_path / "memory.sqlite3")
    calls: list[tuple[str, str]] = []

    class CrashOnce:
        def start(self, request: ControllerModuleInvocationRequest) -> Any:
            calls.append(("start", request.module_invocation_id))
            raise RuntimeError("simulated process exit")

        def resume(self, request: ControllerModuleInvocationRequest) -> ModuleCompleted:
            calls.append(("resume", request.module_invocation_id))
            return ModuleCompleted(_minimal_investigation_output(request))

    class Review:
        def start(self, request: ControllerModuleInvocationRequest) -> ModuleCompleted:
            return _AcceptedReview._complete(request)

        def resume(self, request: ControllerModuleInvocationRequest) -> ModuleCompleted:
            return _AcceptedReview._complete(request)

    service.create_research_task(
        research_task_id="task-1",
        contract_version="investigation_v2.task.v1",
        input={"scientific_question": "resume exception"},
        policy={
            "stage_architecture_version": "investigation_v2",
            "entry_strategy": "standard_hypothesis_generation",
        },
        dependencies={},
    )
    # 先写一个标准 HypothesisBatch，再写 selection，使 Context owner 有明确来源。
    hypothesis_request = ModuleInvocationRequest(
        research_task_id="task-1",
        module_invocation_id="hypothesis-1",
        module_kind="hypothesis-generation",
        input_scientific_record_ids=(),
        module_input_digest="b" * 64,
        context_binding=StatelessModule(),
    )
    service.start_module_invocation("task-1", 1, hypothesis_request)
    batch = _record(
        research_task_id="task-1",
        module_invocation_id="hypothesis-1",
        scientific_record_id="hypothesis-1:batch",
        scientific_record_type="HypothesisBatchRecord",
        payload={
            "hypotheses": [
                {"hypothesis_id": f"h-{rank}", "claim": f"claim-{rank}"}
                for rank in range(1, 6)
            ],
            "rationale": "test input",
            "verification_plan": "test",
        },
    )
    service.complete_module_invocation(
        "task-1",
        2,
        PreparedModuleInvocationOutput.from_records(
            module_invocation_id="hypothesis-1",
            module_input_digest="b" * 64,
            output_scientific_records=(batch,),
        ),
    )
    selection_request = ModuleInvocationRequest(
        research_task_id="task-1",
        module_invocation_id="selection-1",
        module_kind="hypothesis-selection",
        input_scientific_record_ids=(batch.scientific_record_id,),
        module_input_digest="c" * 64,
        context_binding=StatelessModule(),
    )
    service.start_module_invocation("task-1", 4, selection_request)
    selection = _record(
        research_task_id="task-1",
        module_invocation_id="selection-1",
        scientific_record_id="selection-1:selection",
        scientific_record_type="HypothesisSelectionRecord",
        payload={
            "selected_hypotheses": [
                {"hypothesis_ref": f"h{rank}", "rank": rank}
                for rank in range(1, 6)
            ],
            "selection_reason": "top",
        },
    )
    selected = tuple(
        _record(
            research_task_id="task-1",
            module_invocation_id="selection-1",
            scientific_record_id=f"selection-1:selected-{rank}",
            scientific_record_type="SelectedHypothesisRecord",
            source_scientific_record_ids=(selection.scientific_record_id,),
            payload={
                "hypothesis_ref": f"h{rank}",
                "rank": rank,
                "hypothesis": {"hypothesis_id": f"h{rank}", "claim": f"claim-{rank}"},
            },
        )
        for rank in range(1, 6)
    )
    # 该测试只核对 Controller 边界，Context 的完整五条输入由前一个测试覆盖。
    service.complete_module_invocation(
        "task-1",
        5,
        PreparedModuleInvocationOutput.from_records(
            module_invocation_id="selection-1",
            module_input_digest="c" * 64,
            output_scientific_records=(selection, *selected),
        ),
    )
    controller = ResearchWorkflowController(
        service,
        {
            "hypothesis-investigation": CrashOnce(),
            "task-review": Review(),
        },
        require_entry_strategy=True,
    )
    with pytest.raises(RuntimeError, match="simulated process exit"):
        controller.resume_research_task("task-1")
    facts = service.read_research_task_facts("task-1")
    assert facts.open_module_invocation is not None
    module_id = facts.open_module_invocation.module_invocation_id
    resumed = controller.resume_research_task("task-1")
    assert isinstance(resumed, ResearchTaskClosedResult)
    assert calls == [("start", module_id), ("resume", module_id)]


def test_native_provider_interruption_resume_preserves_thread_and_job_identity(
    tmp_path: Path,
) -> None:
    """Harness 内部 provider interrupted 后，resume 不新建 session 或 job。"""

    class Provider:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def run_turn(self, request: dict[str, Any], **kwargs: Any) -> NativeTurnResult:
            del request
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return NativeTurnResult(
                    thread_id="thread-1",
                    turn_id="turn-1",
                    status="interrupted",
                )
            return NativeTurnResult(
                thread_id="thread-1",
                turn_id="turn-2",
                result={
                    "self_judgment": "supported",
                    "narrative": "resumed",
                    "job_refs": [],
                    "operation_refs": [],
                    "artifact_refs": [],
                    "observations": ["observed"],
                    "limitations": [],
                },
            )

    provider = Provider()
    store = HarnessRecoveryStore(tmp_path / "recovery.sqlite3")
    backend = InvestigationSessionRunner(
        recovery_store=store,
        provider_transport=provider,
        execution_transport=None,
    )
    harness = InvestigationHarness(
        tmp_path / "runtime",
        backend=backend,
        recovery_store=store,
    )
    request = InvestigationHarnessRequest(
        research_task_id="task-1",
        module_invocation_id="module-1",
        module_kind="hypothesis-investigation",
        context_binding=CreateInvestigationContext(
            "context-1", "batch-1"
        ).to_dict(),
        input_scientific_record_ids=("selection-1",),
        module_input_digest="a" * 64,
    )

    first = harness.start(request)
    assert isinstance(first, Suspended)
    second = harness.resume(first.harness_invocation_id, request)
    # 没有收集作业，native backend 会把 provider 的结果视为缺少证据并保持可恢复。
    assert isinstance(second, Suspended)
    assert second.reason_code == "recovery_evidence_missing"
    assert len(provider.calls) == 2
    assert provider.calls[1]["thread_id"] == "thread-1"
    snapshot = store.read_snapshot(
        HarnessRunIdentity(
            research_task_id="task-1",
            module_invocation_id="module-1",
            harness_invocation_id=request.harness_invocation_id,
            module_input_digest="a" * 64,
        )
    )
    assert snapshot.state is HarnessLifecycle.SUSPENDED
    assert snapshot.active_job is None
    assert len(snapshot.attempts) == 2
