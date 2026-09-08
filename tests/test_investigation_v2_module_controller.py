"""research_task Module Invocation Controller 的公开顺序与恢复回归。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from chemistry_multiagent.controllers.investigation_v2_module_controller import (
    ControllerAwaitingResearcherConfirmation,
    ControllerSuspended,
    ControllerFailed,
    ResearchWorkflowControllerIntegrityConflict,
    ModuleCompleted,
    ModuleFailed,
    ModuleInvocationRequest,
    ModuleSuspended,
    ResearchWorkflowController,
    ResearchTaskClosedResult,
    ResearchTaskFailedResult,
    build_module_input_digest,
)
from chemistry_multiagent.memory import (
    IdempotencyConflict,
    ModuleInvocationRequest,
    PreparedModuleInvocationOutput,
    ResearchTaskWorkingMemory,
    ScientificRecord,
    StatelessModule,
)
from chemistry_multiagent.memory.v2_identities import (
    CreateInvestigationContext,
    CreateResearchTaskReviewSession,
)


def _task_kwargs(
    *,
    entry_strategy: str = "standard_hypothesis_generation",
) -> dict[str, Any]:
    """返回 canonical task.v2 创建参数。"""

    return {
        "contract_version": "investigation_v2.task.v2",
        "input": {"scientific_question": "controller module order"},
        "policy": {
            "entry_strategy": entry_strategy,
            "budgets": {"max_hypothesis_batches": 2},
        },
        "dependencies": {"catalog": {"ref": "catalog-1"}},
    }


def _record(
    request: ModuleInvocationRequest,
    *,
    record_id: str,
    record_type: str,
    payload: dict[str, Any],
    source_ids: tuple[str, ...] = (),
    schema_version: str | int = 1,
) -> ScientificRecord:
    """构造属于指定 Module Invocation 的 canonical 科学记录。"""

    return ScientificRecord.from_payload(
        scientific_record_id=record_id,
        research_task_id=request.research_task_id,
        scientific_record_type=record_type,
        scientific_record_schema_version=schema_version,
        producer_module_invocation_id=request.module_invocation_id,
        source_scientific_record_ids=source_ids,
        payload=payload,
    )


def _output(
    request: ModuleInvocationRequest,
    records: tuple[ScientificRecord, ...],
) -> PreparedModuleInvocationOutput:
    """构造带自动摘要的 canonical 模块输出。"""

    return PreparedModuleInvocationOutput.from_records(
        module_invocation_id=request.module_invocation_id,
        module_input_digest=request.module_input_digest,
        output_scientific_records=records,
    )


def _standard_output(request: ModuleInvocationRequest) -> PreparedModuleInvocationOutput:
    """为标准入口的各无状态模块生成严格输出。"""

    kind = request.module_kind
    if kind == "retrieval":
        records = (
            _record(
                request,
                record_id=f"{request.module_invocation_id}:retrieval",
                record_type="RetrievalRecord",
                payload={
                    "summary": "retrieved context",
                    "citations": [],
                    "corpus_snapshot_ref": "corpus-1",
                },
            ),
        )
    elif kind == "research-query-formulation":
        records = (
            _record(
                request,
                record_id=f"{request.module_invocation_id}:query",
                record_type="QueryRecord",
                source_ids=request.input_scientific_record_ids,
                payload={
                    "query": "query",
                    "research_angle": "angle",
                    "constraints": {},
                },
            ),
        )
    elif kind == "hypothesis-generation":
        records = (
            _record(
                request,
                record_id=f"{request.module_invocation_id}:batch",
                record_type="HypothesisBatchRecord",
                source_ids=request.input_scientific_record_ids,
                schema_version=1,
                payload={
                    "hypotheses": [
                        {"hypothesis_id": f"h-{index}", "claim": f"claim-{index}"}
                        for index in range(1, 6)
                    ],
                    "rationale": "generated",
                    "verification_plan": "verify",
                },
            ),
        )
    elif kind == "hypothesis-selection":
        selection_id = f"{request.module_invocation_id}:selection"
        selection = _record(
            request,
            record_id=selection_id,
            record_type="HypothesisSelectionRecord",
            source_ids=request.input_scientific_record_ids,
            schema_version=2,
            payload={
                "selected_hypotheses": [
                    {"hypothesis_ref": f"h-{index}", "rank": index}
                    for index in range(1, 6)
                ],
                "selection_reason": "top five",
            },
        )
        selected = tuple(
            _record(
                request,
                record_id=f"{request.module_invocation_id}:selected-{index}",
                record_type="SelectedHypothesisRecord",
                # Selection 摘要是直接来源，同时保留 Batch 的可追溯引用，
                # 供 Confirmation 的候选闭包核对。
                source_ids=(selection_id, *request.input_scientific_record_ids),
                payload={
                    "hypothesis_ref": f"h-{index}",
                    "rank": index,
                    "hypothesis": {
                        "hypothesis_id": f"h-{index}",
                        "claim": f"claim-{index}",
                    },
                },
            )
            for index in range(1, 6)
        )
        records = (selection, *selected)
    else:  # pragma: no cover - 测试数据错误
        raise AssertionError(f"不支持的标准输出模块：{kind}")
    return _output(request, records)


def _author_import_output(request: ModuleInvocationRequest) -> PreparedModuleInvocationOutput:
    """生成 author_hypothesis_import 的 Batch 与逐假设记录。"""

    batch_id = f"{request.module_invocation_id}:batch"
    hypothesis = {"hypothesis_id": "author-h1", "claim": "author claim"}
    batch = _record(
        request,
        record_id=batch_id,
        record_type="HypothesisBatchRecord",
        schema_version="author-import.v2",
        payload={
            "imported_hypotheses": [
                {
                    "hypothesis_ref": "author-h1",
                    "source_ref": "author-source-1",
                    "source_digest": "a" * 64,
                    "hypothesis": hypothesis,
                    "provenance": {"kind": "author"},
                }
            ]
        },
    )
    selected = _record(
        request,
        record_id=f"{request.module_invocation_id}:selected-1",
        record_type="SelectedHypothesisRecord",
        source_ids=(batch_id,),
        payload={
            "hypothesis_ref": "author-h1",
            "rank": 1,
            "hypothesis": hypothesis,
        },
    )
    return _output(request, (batch, selected))


def _investigation_output(
    request: ModuleInvocationRequest,
    *,
    input_ids: tuple[str, ...] | None = None,
    additional_result_source_ids: tuple[str, ...] = (),
) -> PreparedModuleInvocationOutput:
    """为一个或多个执行输入生成一一配对的 evidence/result。"""

    records: list[ScientificRecord] = []
    effective_input_ids = input_ids or request.input_scientific_record_ids
    for index, input_id in enumerate(
        (effective_input_ids or ("execution-input-1",)),
        start=1,
    ):
        evidence_id = f"{request.module_invocation_id}:evidence-{index}"
        result_id = f"{request.module_invocation_id}:result-{index}"
        evidence = _record(
            request,
            record_id=evidence_id,
            record_type="VerifiedEvidenceRecord",
            source_ids=request.input_scientific_record_ids,
            payload={
                "evidence_items": [{"value": f"observed-{index}"}],
                "artifact_refs": [],
                "source_manifest_ref": f"manifest-{index}",
                "summary": f"evidence-{index}",
                "verification": {"verified": True},
            },
        )
        # 与生产 backend 一致：Result 保留证据和本次请求的全部科学输入。
        # 返修请求包含旧 Result，因此新 Result 会形成显式 Result→Result 替换边。
        result_sources = tuple(
            dict.fromkeys(
                (
                    *additional_result_source_ids,
                    evidence_id,
                    *request.input_scientific_record_ids,
                )
            )
        )
        result = _record(
            request,
            record_id=result_id,
            record_type="InvestigationResultRecord",
            source_ids=result_sources,
            payload={
                "conclusion": f"conclusion-{index}",
                "self_judgment": "supported",
                "observations": [f"observed-{index}"],
                "limitations": [],
                "evidence_ref": evidence_id,
            },
        )
        records.extend((evidence, result))
    return _output(request, tuple(records))


class RecordingModule:
    """不启动模型或 MiniChem 的确定性模块替身。"""

    def __init__(self, factory: Callable[[ModuleInvocationRequest], object]) -> None:
        self.factory = factory
        self.calls: list[tuple[str, ModuleInvocationRequest]] = []

    def start(self, request: ModuleInvocationRequest) -> object:
        self.calls.append(("start", request))
        return self.factory(request)

    def resume(self, request: ModuleInvocationRequest) -> object:
        self.calls.append(("resume", request))
        return self.factory(request)


def _chain_modules(
    *,
    review_outcomes: list[str] | None = None,
) -> dict[str, RecordingModule]:
    """构造标准或作者路径使用的全部确定性 adapter。"""

    outcomes = review_outcomes if review_outcomes is not None else ["Accepted"]

    def review_output(request: ModuleInvocationRequest) -> PreparedModuleInvocationOutput:
        outcome = outcomes.pop(0) if outcomes else "Accepted"
        review_id = f"{request.module_invocation_id}:review"
        review = _record(
            request,
            record_id=review_id,
            record_type="ReviewRecord",
            source_ids=request.input_scientific_record_ids,
            payload={
                "reviewed_result_refs": list(request.input_scientific_record_ids),
                "findings": [{"kind": outcome}],
                "outcome": outcome,
            },
        )
        if outcome == "RepairRequired":
            result_ref = request.input_scientific_record_ids[0]
            repair = _record(
                request,
                record_id=f"{request.module_invocation_id}:repair",
                record_type="RepairRequestRecord",
                source_ids=(review_id, result_ref),
                payload={
                    "result_ref": result_ref,
                    "problem": "verification incomplete",
                    "instructions": "perform additional verification",
                },
            )
            return _output(request, (review, repair))
        final = _record(
            request,
            record_id=f"{request.module_invocation_id}:answer",
            record_type="FinalAnswerRecord",
            source_ids=(review_id,),
            payload={
                "answer": "answer",
                "answerability": "answerable" if outcome == "Accepted" else "unanswerable",
            },
        )
        return _output(request, (review, final))

    investigation = RecordingModule(_investigation_output)

    def investigation_output(request: ModuleInvocationRequest) -> PreparedModuleInvocationOutput:
        repair_refs = tuple(
            item
            for item in request.input_scientific_record_ids
            if ":repair" in item
        )
        if not repair_refs:
            return _investigation_output(request)
        # 返修只针对 RepairRequestRecord 产生一对新结果；首轮确认输入作为
        # 结果直接来源，供 Controller 的当前 execution-input 覆盖核对。
        first_request = investigation.calls[0][1]
        return _investigation_output(
            request,
            input_ids=(repair_refs[0],),
            additional_result_source_ids=(
                first_request.input_scientific_record_ids[0],
            ),
        )

    investigation.factory = investigation_output

    return {
        "retrieval": RecordingModule(_standard_output),
        "research-query-formulation": RecordingModule(_standard_output),
        "hypothesis-generation": RecordingModule(_standard_output),
        "hypothesis-selection": RecordingModule(_standard_output),
        "hypothesis-import": RecordingModule(_author_import_output),
        "hypothesis-investigation": investigation,
        "task-review": RecordingModule(review_output),
    }


def _controller(
    tmp_path: Path,
    modules: dict[str, RecordingModule],
) -> ResearchWorkflowController:
    """创建隔离的 canonical Controller。"""

    return ResearchWorkflowController(
        ResearchTaskWorkingMemory(tmp_path / "research_task.sqlite3"),
        modules,
        require_entry_strategy=True,
    )


def test_module_request_exposes_only_canonical_identity_envelope() -> None:
    """模块请求只携带 research_task、invocation、记录引用和 typed owner。"""

    digest = build_module_input_digest(
        "research-task-1", "hypothesis-investigation", ("input-1",)
    )
    request = ModuleInvocationRequest(
        research_task_id="research-task-1",
        module_invocation_id="invocation-1",
        module_kind="hypothesis-investigation",
        input_scientific_record_ids=("input-1",),
        module_input_digest=digest,
        context_binding=CreateInvestigationContext("context-1", "batch-1"),
    )
    public = request.to_public_dict()
    assert public["research_task_id"] == "research-task-1"
    assert public["module_invocation_id"] == "invocation-1"
    assert public["input_scientific_record_ids"] == ["input-1"]
    assert public["module_input_digest"] == digest
    assert public["context_binding"]["kind"] == "CreateInvestigationContext"
    assert not {
        "task_id",
        "module_run_id",
        "runtime_ref",
        "input_digest",
    }.intersection(public)
    assert ModuleInvocationRequest.from_mapping(public) == request


def test_standard_task_stops_at_confirmation_before_investigation(tmp_path: Path) -> None:
    """标准 task.v2 按正式顺序执行至 Confirmation，尚不创建调查副作用。"""

    modules = _chain_modules()
    controller = _controller(tmp_path, modules)
    result = controller.run_research_task(
        "research-task-1",
        **_task_kwargs(),
    )

    assert isinstance(result, ControllerAwaitingResearcherConfirmation)
    assert result.research_task_id == "research-task-1"
    assert result.view.reason_code == "awaiting_researcher_confirmation"
    assert [kind for kind, module in modules.items() if module.calls] == [
        "retrieval",
        "research-query-formulation",
        "hypothesis-generation",
        "hypothesis-selection",
    ]
    assert not modules["hypothesis-investigation"].calls
    assert not modules["task-review"].calls
    facts = controller.working_memory.read_research_task_facts("research-task-1")
    assert facts.open_module_invocation is not None
    assert facts.open_module_invocation.module_kind == "hypothesis-confirmation"


def test_confirmation_submission_continues_to_investigation_and_review(
    tmp_path: Path,
) -> None:
    """研究员明确决定后才创建 Context、Investigation 和 Review。"""

    modules = _chain_modules()
    controller = _controller(tmp_path, modules)
    waiting = controller.run_research_task("research-task-1", **_task_kwargs())
    assert isinstance(waiting, ControllerAwaitingResearcherConfirmation)
    decisions = [
        {
            "hypothesis_candidate_record_id": candidate[
                "hypothesis_candidate_record_id"
            ],
            "decision": "execute" if index == 1 else "cancel",
            **(
                {"hypothesis": candidate["hypothesis"]}
                if index == 1
                else {}
            ),
        }
        for index, candidate in enumerate(waiting.candidates, start=1)
    ]

    result = controller.submit_hypothesis_confirmation_and_continue(
        "research-task-1",
        confirmation_id="confirmation-1",
        expected_research_task_sequence=waiting.expected_research_task_sequence,
        decisions=decisions,
    )
    assert isinstance(result, ResearchTaskClosedResult)
    assert result.status == "succeeded"
    assert modules["hypothesis-investigation"].calls
    assert modules["task-review"].calls
    investigation_request = modules["hypothesis-investigation"].calls[0][1]
    assert isinstance(
        investigation_request.context_binding,
        CreateInvestigationContext,
    )
    assert investigation_request.input_scientific_record_ids
    assert modules["task-review"].calls[0][1].input_scientific_record_ids
    facts = controller.working_memory.read_research_task_facts("research-task-1")
    assert facts.lifecycle.state == "Closed"
    assert facts.lifecycle.outcome["status"] == "succeeded"


def test_suspended_open_invocation_resumes_same_request(tmp_path: Path) -> None:
    """暂时不可用只保留 Open，恢复调用原 invocation 而不新建模块。"""

    modules = _chain_modules()
    retrieval = modules["retrieval"]
    original = retrieval.factory
    attempts = 0

    def suspend_once(request: ModuleInvocationRequest) -> object:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return ModuleSuspended("provider_unavailable")
        return original(request)

    retrieval.factory = suspend_once
    controller = _controller(tmp_path, modules)
    controller.run_research_task("research-task-1", **_task_kwargs())
    first = retrieval.calls[0][1]
    facts = controller.working_memory.read_research_task_facts("research-task-1")
    assert facts.open_module_invocation is not None
    resumed = controller.resume_research_task("research-task-1")
    assert isinstance(resumed, ControllerAwaitingResearcherConfirmation)
    assert retrieval.calls[1][0] == "resume"
    assert retrieval.calls[1][1] == first


def test_author_entry_skips_standard_generation_and_selection(tmp_path: Path) -> None:
    """作者入口只执行 Hypothesis Import，再进入同一 Confirmation seam。"""

    modules = _chain_modules()
    controller = _controller(tmp_path, modules)
    result = controller.run_research_task(
        "author-research-task",
        **_task_kwargs(entry_strategy="author_hypothesis_import"),
    )
    assert isinstance(result, ControllerAwaitingResearcherConfirmation)
    assert modules["hypothesis-import"].calls
    assert not modules["retrieval"].calls
    assert not modules["research-query-formulation"].calls
    assert not modules["hypothesis-generation"].calls
    assert not modules["hypothesis-selection"].calls


def test_module_failure_writes_failed_task_and_replays_terminal_result(
    tmp_path: Path,
) -> None:
    """只有明确 ModuleFailed 才原子写失败事实；恢复不重复调用模块。"""

    modules = _chain_modules()
    calls = modules["retrieval"].calls
    modules["retrieval"].factory = lambda _request: ModuleFailed(
        "verified_integrity_conflict"
    )
    controller = _controller(tmp_path, modules)
    result = controller.run_research_task("research-task-1", **_task_kwargs())
    assert isinstance(result, ResearchTaskFailedResult)
    assert result.outcome["failure_code"] == "verified_integrity_conflict"
    assert controller.resume_research_task("research-task-1") == result
    assert len(calls) == 1
    assert [event.event_type for event in controller.working_memory.read_research_task_event_log("research-task-1")][-2:] == [
        "ModuleInvocationFailed",
        "ResearchTaskFailed",
    ]


def test_investigation_and_review_use_typed_context(tmp_path: Path) -> None:
    """首次 Investigation 与 Review 都只使用 typed context owner。"""

    modules = _chain_modules()
    controller = _controller(tmp_path, modules)
    result = controller.run_research_task("research-task-1", **_task_kwargs())
    assert isinstance(result, ControllerAwaitingResearcherConfirmation)
    decisions = [
        {
            "hypothesis_candidate_record_id": candidate[
                "hypothesis_candidate_record_id"
            ],
            "decision": "execute" if index == 1 else "cancel",
            **(
                {"hypothesis": candidate["hypothesis"]}
                if index == 1
                else {}
            ),
        }
        for index, candidate in enumerate(result.candidates, start=1)
    ]
    final = controller.submit_hypothesis_confirmation_and_continue(
        "research-task-1",
        confirmation_id="confirmation-context",
        expected_research_task_sequence=result.expected_research_task_sequence,
        decisions=decisions,
    )
    assert isinstance(final, ResearchTaskClosedResult)
    investigations = modules["hypothesis-investigation"].calls
    reviews = modules["task-review"].calls
    assert len(investigations) == 1
    assert len(reviews) == 1
    first_request = investigations[0][1]
    assert isinstance(first_request.context_binding, CreateInvestigationContext)
    assert isinstance(reviews[0][1].context_binding, CreateResearchTaskReviewSession)


def test_repair_result_replaces_reviewed_result_head(tmp_path: Path) -> None:
    """返修结果替换被审查的旧结果，下一轮 Review 只读取新 head。"""

    modules = _chain_modules(review_outcomes=["RepairRequired", "Accepted"])
    controller = _controller(tmp_path, modules)
    waiting = controller.run_research_task("research-task-repair", **_task_kwargs())
    assert isinstance(waiting, ControllerAwaitingResearcherConfirmation)
    decisions = [
        {
            "hypothesis_candidate_record_id": candidate[
                "hypothesis_candidate_record_id"
            ],
            "decision": "execute" if index == 1 else "cancel",
            **({"hypothesis": candidate["hypothesis"]} if index == 1 else {}),
        }
        for index, candidate in enumerate(waiting.candidates, start=1)
    ]

    result = controller.submit_hypothesis_confirmation_and_continue(
        "research-task-repair",
        confirmation_id="confirmation-repair",
        expected_research_task_sequence=waiting.expected_research_task_sequence,
        decisions=decisions,
    )

    assert isinstance(result, ResearchTaskClosedResult)
    assert len(modules["hypothesis-investigation"].calls) == 2
    assert len(modules["task-review"].calls) == 2
    old_result_ref = modules["task-review"].calls[0][1].input_scientific_record_ids[0]
    new_review_refs = modules["task-review"].calls[1][1].input_scientific_record_ids
    assert old_result_ref not in new_review_refs
    assert len(new_review_refs) == 1


def test_controller_rejects_canonical_output_identity_drift(tmp_path: Path) -> None:
    """模块输出 invocation 或输入摘要漂移时不能提交科学事实。"""

    modules = _chain_modules()

    def drifted(request: ModuleInvocationRequest) -> ModuleCompleted:
        output = _standard_output(request)
        drifted_output = PreparedModuleInvocationOutput.from_records(
            module_invocation_id=request.module_invocation_id,
            module_input_digest="drifted-input-digest",
            output_scientific_records=output.output_scientific_records,
        )
        return ModuleCompleted(drifted_output)

    modules["retrieval"].factory = drifted
    controller = _controller(tmp_path, modules)
    with pytest.raises(ResearchWorkflowControllerIntegrityConflict):
        controller.run_research_task("research-task-1", **_task_kwargs())
    facts = controller.working_memory.read_research_task_facts("research-task-1")
    assert facts.open_module_invocation is not None
    assert facts.open_module_invocation.module_kind == "retrieval"
