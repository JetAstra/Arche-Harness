"""research_task Confirmation 深模块的公开输入、输出与幂等合同测试。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from chemistry_multiagent.controllers.investigation_v2_confirmation import (
    HypothesisConfirmationError,
    HypothesisConfirmationStateConflict,
    HypothesisConfirmationView,
    HypothesisConfirmationService,
    ResearchMaterialUpload,
)
from chemistry_multiagent.controllers.investigation_v2_module_controller import (
    ControllerAwaitingResearcherConfirmation,
    ModuleCompleted,
    ModuleInvocationRequest,
    ModuleSuspended,
    ResearchWorkflowController,
    ResearchTaskClosedResult,
    build_module_input_digest,
)
from chemistry_multiagent.investigation.research_material_store import (
    MaterialSizePolicy,
    ResearchMaterialStore,
)
from chemistry_multiagent.memory import (
    CASConflict,
    IdempotencyConflict,
    PreparedModuleInvocationOutput,
    ResearchTaskWorkingMemory,
    ScientificRecord,
)
from chemistry_multiagent.memory.v2_identities import (
    CreateInvestigationContext,
    CreateResearchTaskReviewSession,
)


def _task_kwargs(*, entry_strategy: str = "standard_hypothesis_generation") -> dict[str, Any]:
    """返回 Confirmation 测试使用的 canonical task.v2 配置。"""

    return {
        "contract_version": "investigation_v2.task.v2",
        "input": {"scientific_question": "confirmation contract"},
        "policy": {
            "entry_strategy": entry_strategy,
            "budgets": {"max_hypothesis_batches": 1},
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
    """构造属于当前 Module Invocation 的科学记录。"""

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
    """密封带当前 request 身份和输入摘要的模块输出。"""

    return PreparedModuleInvocationOutput.from_records(
        module_invocation_id=request.module_invocation_id,
        module_input_digest=request.module_input_digest,
        output_scientific_records=records,
    )


def _module_output(request: ModuleInvocationRequest) -> PreparedModuleInvocationOutput:
    """为确认前后的标准模块生成最小合法输出。"""

    kind = request.module_kind
    if kind == "retrieval":
        return _output(
            request,
            (
                _record(
                    request,
                    record_id=f"{request.module_invocation_id}:retrieval",
                    record_type="RetrievalRecord",
                    payload={
                        "summary": "retrieval",
                        "citations": [],
                        "corpus_snapshot_ref": "corpus-1",
                    },
                ),
            ),
        )
    if kind == "research-query-formulation":
        return _output(
            request,
            (
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
            ),
        )
    if kind == "hypothesis-generation":
        return _output(
            request,
            (
                _record(
                    request,
                    record_id=f"{request.module_invocation_id}:batch",
                    record_type="HypothesisBatchRecord",
                    source_ids=request.input_scientific_record_ids,
                    payload={
                        "hypotheses": [
                            {
                                "hypothesis_id": f"h-{rank}",
                                "claim": f"claim-{rank}",
                            }
                            for rank in range(1, 6)
                        ],
                        "rationale": "rationale",
                        "verification_plan": "plan",
                    },
                ),
            ),
        )
    if kind == "hypothesis-selection":
        selection_id = f"{request.module_invocation_id}:selection"
        selection = _record(
            request,
            record_id=selection_id,
            record_type="HypothesisSelectionRecord",
            schema_version=2,
            source_ids=request.input_scientific_record_ids,
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
                request,
                record_id=f"{request.module_invocation_id}:selected:{rank}",
                record_type="SelectedHypothesisRecord",
                source_ids=(selection_id, *request.input_scientific_record_ids),
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
        return _output(request, (selection, *selected))
    if kind == "hypothesis-import":
        batch_id = f"{request.module_invocation_id}:batch"
        imported = [
            {
                "hypothesis_ref": f"author-h-{rank}",
                "source_ref": f"author-source-{rank}",
                "source_digest": "a" * 64,
                "hypothesis": {
                    "hypothesis_id": f"author-h-{rank}",
                    "strategy_name": f"author-strategy-{rank}",
                    "reasoning": f"author reasoning {rank}",
                },
                "provenance": {"kind": "researcher"},
            }
            for rank in range(1, 3)
        ]
        batch = _record(
            request,
            record_id=batch_id,
            record_type="HypothesisBatchRecord",
            schema_version="author-import.v2",
            payload={"imported_hypotheses": imported},
        )
        selected = tuple(
            _record(
                request,
                record_id=f"{request.module_invocation_id}:selected:{rank}",
                record_type="SelectedHypothesisRecord",
                source_ids=(batch_id,),
                payload={
                    "hypothesis_ref": item["hypothesis_ref"],
                    "rank": rank,
                    "hypothesis": item["hypothesis"],
                },
            )
            for rank, item in enumerate(imported, start=1)
        )
        return _output(request, (batch, *selected))
    if kind == "hypothesis-investigation":
        records: list[ScientificRecord] = []
        for rank, input_id in enumerate(request.input_scientific_record_ids, start=1):
            evidence_id = f"{request.module_invocation_id}:evidence:{rank}"
            records.extend(
                (
                    _record(
                        request,
                        record_id=evidence_id,
                        record_type="VerifiedEvidenceRecord",
                        source_ids=(input_id,),
                        payload={
                            "evidence_items": [{"value": "observed"}],
                            "artifact_refs": [],
                            "source_manifest_ref": f"manifest-{rank}",
                            "summary": "evidence",
                            "verification": {"verified": True},
                        },
                    ),
                    _record(
                        request,
                        record_id=f"{request.module_invocation_id}:result:{rank}",
                        record_type="InvestigationResultRecord",
                        source_ids=(input_id, evidence_id),
                        payload={
                            "conclusion": "conclusion",
                            "self_judgment": "supported",
                            "observations": ["observed"],
                            "limitations": [],
                            "evidence_ref": evidence_id,
                        },
                    ),
                )
            )
        return _output(request, tuple(records))
    if kind == "task-review":
        review_id = f"{request.module_invocation_id}:review"
        review = _record(
            request,
            record_id=review_id,
            record_type="ReviewRecord",
            source_ids=request.input_scientific_record_ids,
            payload={
                "reviewed_result_refs": list(request.input_scientific_record_ids),
                "findings": [{"kind": "Accepted"}],
                "outcome": "Accepted",
            },
        )
        final = _record(
            request,
            record_id=f"{request.module_invocation_id}:answer",
            record_type="FinalAnswerRecord",
            source_ids=(review_id,),
            payload={"answer": "answer", "answerability": "answerable"},
        )
        return _output(request, (review, final))
    raise AssertionError(f"unsupported module kind: {kind}")


class RecordingModule:
    """不调用模型或 MiniChem 的确定性模块替身。"""

    def __init__(self, factory: Callable[[ModuleInvocationRequest], object]) -> None:
        self.factory = factory
        self.calls: list[tuple[str, ModuleInvocationRequest]] = []

    def start(self, request: ModuleInvocationRequest) -> object:
        self.calls.append(("start", request))
        return self.factory(request)

    def resume(self, request: ModuleInvocationRequest) -> object:
        self.calls.append(("resume", request))
        return self.factory(request)


def _controller(
    tmp_path: Path,
    *,
    entry_strategy: str = "standard_hypothesis_generation",
    material_store: ResearchMaterialStore | None = None,
    suspend_investigation: bool = False,
) -> tuple[ResearchWorkflowController, dict[str, RecordingModule]]:
    """创建带 canonical 模块名的确认 Controller。"""

    modules = {
        "retrieval": RecordingModule(_module_output),
        "research-query-formulation": RecordingModule(_module_output),
        "hypothesis-generation": RecordingModule(_module_output),
        "hypothesis-selection": RecordingModule(_module_output),
        "hypothesis-import": RecordingModule(_module_output),
        "hypothesis-investigation": RecordingModule(
            lambda request: (
                ModuleSuspended("test_pause")
                if suspend_investigation
                else ModuleCompleted(_module_output(request))
            )
        ),
        "task-review": RecordingModule(_module_output),
    }
    controller = ResearchWorkflowController(
        ResearchTaskWorkingMemory(tmp_path / "research_task.sqlite3"),
        modules,
        require_entry_strategy=True,
        material_store=material_store,
    )
    controller.create_research_task(
        "research-task-1",
        **_task_kwargs(entry_strategy=entry_strategy),
    )
    return controller, modules


def _awaiting_confirmation(
    tmp_path: Path,
    *,
    entry_strategy: str = "standard_hypothesis_generation",
    material_store: ResearchMaterialStore | None = None,
    suspend_investigation: bool = False,
) -> tuple[ResearchWorkflowController, dict[str, RecordingModule], ControllerAwaitingResearcherConfirmation]:
    """推进到唯一 Open Confirmation 并返回其公开视图。"""

    controller, modules = _controller(
        tmp_path,
        entry_strategy=entry_strategy,
        material_store=material_store,
        suspend_investigation=suspend_investigation,
    )
    result = controller.resume_research_task("research-task-1")
    assert isinstance(result, ControllerAwaitingResearcherConfirmation)
    return controller, modules, result


def _decisions(
    waiting: ControllerAwaitingResearcherConfirmation,
    execute_ranks: set[int],
    *,
    include_hypothesis: bool = True,
    materials: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """按公开候选身份构造一组完整确认决定。"""

    result: list[dict[str, Any]] = []
    for candidate in waiting.candidates:
        rank = candidate["rank"]
        item: dict[str, Any] = {
            "hypothesis_candidate_record_id": candidate[
                "hypothesis_candidate_record_id"
            ],
            "decision": "execute" if rank in execute_ranks else "cancel",
        }
        if rank in execute_ranks and include_hypothesis:
            item["hypothesis"] = candidate["hypothesis"]
            if materials is not None:
                item["materials"] = materials
        result.append(item)
    return result


def test_confirmation_view_uses_only_canonical_research_task_fields(tmp_path: Path) -> None:
    """研究员视图只暴露 task、候选科学记录和确认序号。"""

    _controller_instance, _modules, waiting = _awaiting_confirmation(tmp_path)
    view = waiting.view
    assert set(view.to_dict()) == {
        "research_task_id",
        "status",
        "reason_code",
        "expected_research_task_sequence",
        "hypothesis_batch_record_id",
        "candidates",
    }
    assert view.research_task_id == "research-task-1"
    assert view.reason_code == "awaiting_researcher_confirmation"
    assert set(view.candidates[0]) == {
        "hypothesis_candidate_record_id",
        "rank",
        "hypothesis",
    }
    assert not {"task_id", "module_run_id", "runtime_ref"}.intersection(view.to_dict())


def test_standard_confirmation_stops_before_investigation(tmp_path: Path) -> None:
    """标准路径到 Confirmation 即暂停，不产生 Investigation 调用。"""

    controller, modules, waiting = _awaiting_confirmation(tmp_path)
    assert len(waiting.candidates) == 5
    assert not modules["hypothesis-investigation"].calls
    facts = controller.working_memory.read_research_task_facts("research-task-1")
    assert facts.open_module_invocation is not None
    assert facts.open_module_invocation.module_kind == "hypothesis-confirmation"


def test_author_confirmation_exposes_one_to_five_imported_candidates(
    tmp_path: Path,
) -> None:
    """作者入口使用 canonical hypothesis-import，并保持同一 Confirmation seam。"""

    controller, modules, waiting = _awaiting_confirmation(
        tmp_path,
        entry_strategy="author_hypothesis_import",
    )
    assert len(waiting.candidates) == 2
    assert modules["hypothesis-import"].calls
    assert not modules["retrieval"].calls
    assert controller.get_hypothesis_confirmation("research-task-1").to_dict() == (
        waiting.view.to_dict()
    )


def test_confirmation_commits_only_execute_inputs_with_canonical_identity(
    tmp_path: Path,
) -> None:
    """完整决定闭合 Confirmation，并仅为 execute 候选写 Input 记录。"""

    controller, modules, waiting = _awaiting_confirmation(tmp_path)
    decisions = _decisions(waiting, {1, 3})
    submission = controller.confirmation_service.confirm(
        "research-task-1",
        confirmation_id="confirmation-1",
        expected_research_task_sequence=waiting.expected_research_task_sequence,
        decisions=decisions,
    )
    assert submission.confirmation_id == "confirmation-1"
    assert not submission.replayed
    facts = controller.working_memory.read_research_task_facts("research-task-1")
    assert facts.open_module_invocation is None
    records = controller.working_memory.read_research_task_scientific_records(
        "research-task-1",
        submission.module.lifecycle.output_scientific_record_ids
        if hasattr(submission.module.lifecycle, "output_scientific_record_ids")
        else facts.module_invocations[-1].lifecycle.outcome[
            "output_scientific_record_ids"
        ],
    )
    assert [record.scientific_record_type for record in records] == [
        "HypothesisExecutionDecisionRecord",
        "HypothesisExecutionInputRecord",
        "HypothesisExecutionInputRecord",
    ]
    decision = records[0]
    assert set(decision.payload["decisions"][0]) == {
        "hypothesis_candidate_record_id",
        "decision",
        "execution_rank",
    }
    assert all(
        "task_id" not in record.to_dict()
        and "module_run_id" not in record.to_dict()
        for record in records
    )
    assert not modules["hypothesis-investigation"].calls


def test_confirmation_submission_continues_to_investigation_and_review(
    tmp_path: Path,
) -> None:
    """提交确认后才调用 Investigation 与 Review。"""

    controller, modules, waiting = _awaiting_confirmation(tmp_path)
    result = controller.submit_hypothesis_confirmation_and_continue(
        "research-task-1",
        confirmation_id="confirmation-continue",
        expected_research_task_sequence=waiting.expected_research_task_sequence,
        decisions=_decisions(waiting, {1, 3}),
    )
    assert isinstance(result, ResearchTaskClosedResult)
    assert result.status == "succeeded"
    assert len(modules["hypothesis-investigation"].calls) == 1
    assert len(modules["hypothesis-investigation"].calls[0][1].input_scientific_record_ids) == 2
    assert len(modules["task-review"].calls) == 1
    assert isinstance(
        modules["hypothesis-investigation"].calls[0][1].context_binding,
        CreateInvestigationContext,
    )
    assert isinstance(
        modules["task-review"].calls[0][1].context_binding,
        CreateResearchTaskReviewSession,
    )


@pytest.mark.parametrize("execute_count", [1, 3, 5])
def test_confirmation_accepts_one_to_five_execution_candidates(
    tmp_path: Path,
    execute_count: int,
) -> None:
    """执行候选数量在 1–5 内时，Investigaton 输入按确认结果冻结。"""

    controller, modules, waiting = _awaiting_confirmation(tmp_path)
    result = controller.submit_hypothesis_confirmation_and_continue(
        "research-task-1",
        confirmation_id=f"confirmation-count-{execute_count}",
        expected_research_task_sequence=waiting.expected_research_task_sequence,
        decisions=_decisions(waiting, set(range(1, execute_count + 1))),
    )
    assert isinstance(result, ResearchTaskClosedResult)
    assert len(modules["hypothesis-investigation"].calls[0][1].input_scientific_record_ids) == execute_count
    records = controller.working_memory.read_research_task_scientific_records(
        "research-task-1"
    )
    assert len(
        [
            record
            for record in records
            if record.scientific_record_type == "InvestigationResultRecord"
        ]
    ) == execute_count


def test_confirmation_rejects_incomplete_decision_coverage(tmp_path: Path) -> None:
    """缺失候选决定时不写 Confirmation 事实。"""

    controller, _modules, waiting = _awaiting_confirmation(tmp_path)
    with pytest.raises(HypothesisConfirmationError):
        controller.confirmation_service.confirm(
            "research-task-1",
            confirmation_id="confirmation-incomplete",
            expected_research_task_sequence=waiting.expected_research_task_sequence,
            decisions=_decisions(waiting, {1})[:-1],
        )
    facts = controller.working_memory.read_research_task_facts("research-task-1")
    assert facts.open_module_invocation is not None


def test_confirmation_rejects_execute_without_hypothesis(tmp_path: Path) -> None:
    """execute 决定必须显式提交最终假设正文。"""

    controller, _modules, waiting = _awaiting_confirmation(tmp_path)
    with pytest.raises(HypothesisConfirmationError):
        controller.confirmation_service.confirm(
            "research-task-1",
            confirmation_id="confirmation-no-hypothesis",
            expected_research_task_sequence=waiting.expected_research_task_sequence,
            decisions=_decisions(waiting, {1}, include_hypothesis=False),
        )


def test_confirmation_rejects_extra_fields_on_cancel(tmp_path: Path) -> None:
    """cancel 决定不能携带会改变科学输入的字段。"""

    controller, _modules, waiting = _awaiting_confirmation(tmp_path)
    decisions = _decisions(waiting, {1})
    decisions[1]["guidance"] = "unexpected"
    with pytest.raises(HypothesisConfirmationError):
        controller.confirmation_service.confirm(
            "research-task-1",
            confirmation_id="confirmation-cancel-fields",
            expected_research_task_sequence=waiting.expected_research_task_sequence,
            decisions=decisions,
        )


def test_confirmation_rejects_stale_sequence_before_commit(tmp_path: Path) -> None:
    """确认视图序号过期时使用 CASConflict，Open 状态保持不变。"""

    controller, _modules, waiting = _awaiting_confirmation(tmp_path)
    with pytest.raises(CASConflict):
        controller.confirmation_service.confirm(
            "research-task-1",
            confirmation_id="confirmation-stale",
            expected_research_task_sequence=waiting.expected_research_task_sequence - 1,
            decisions=_decisions(waiting, {1}),
        )
    facts = controller.working_memory.read_research_task_facts("research-task-1")
    assert facts.open_module_invocation is not None


def test_confirmation_exact_replay_is_idempotent(tmp_path: Path) -> None:
    """同一 confirmation_id 和正文可以精确重放，不追加事件。"""

    controller, _modules, waiting = _awaiting_confirmation(tmp_path)
    decisions = _decisions(waiting, {1})
    first = controller.confirmation_service.confirm(
        "research-task-1",
        confirmation_id="confirmation-replay",
        expected_research_task_sequence=waiting.expected_research_task_sequence,
        decisions=decisions,
    )
    event_count = len(
        controller.working_memory.read_research_task_event_log("research-task-1")
    )
    second = controller.confirmation_service.confirm(
        "research-task-1",
        confirmation_id="confirmation-replay",
        expected_research_task_sequence=1,
        decisions=decisions,
    )
    assert first.module.module_invocation_id == second.module.module_invocation_id
    assert second.replayed
    assert len(controller.working_memory.read_research_task_event_log("research-task-1")) == event_count


def test_confirmation_replay_rejects_changed_confirmation_body(tmp_path: Path) -> None:
    """同一幂等身份不能替换 execute/cancel 或假设正文。"""

    controller, _modules, waiting = _awaiting_confirmation(tmp_path)
    decisions = _decisions(waiting, {1})
    controller.confirmation_service.confirm(
        "research-task-1",
        confirmation_id="confirmation-drift",
        expected_research_task_sequence=waiting.expected_research_task_sequence,
        decisions=decisions,
    )
    changed = _decisions(waiting, {2})
    with pytest.raises(IdempotencyConflict):
        controller.confirmation_service.confirm(
            "research-task-1",
            confirmation_id="confirmation-drift",
            expected_research_task_sequence=1,
            decisions=changed,
        )


def test_confirmation_ingests_one_material_descriptor_without_path(
    tmp_path: Path,
) -> None:
    """上传材料只以摘要记录进入 Working Memory，不保存调用方路径。"""

    material_store = ResearchMaterialStore(
        tmp_path / "materials",
        MaterialSizePolicy(
            max_file_size_bytes=1024,
            max_material_count=5,
            max_task_size_bytes=4096,
        ),
    )
    controller, _modules, waiting = _awaiting_confirmation(
        tmp_path,
        material_store=material_store,
    )
    decisions = _decisions(
        waiting,
        {1, 3},
        materials=[
            {
                "upload_ref": "matrix-elements",
                "description": "共享矩阵元素文件",
            }
        ],
    )
    controller.confirmation_service.confirm(
        "research-task-1",
        confirmation_id="confirmation-material",
        expected_research_task_sequence=waiting.expected_research_task_sequence,
        decisions=decisions,
        files={
            "matrix-elements": ResearchMaterialUpload(
                filename="matrix-elements.json",
                media_type="application/json",
                content=b'{"atoms":[1,2,3]}',
            )
        },
    )
    records = controller.working_memory.read_research_task_scientific_records(
        "research-task-1"
    )
    materials = [
        record
        for record in records
        if record.scientific_record_type == "ResearchMaterialRecord"
    ]
    assert len(materials) == 1
    assert materials[0].payload["filename"] == "matrix-elements.json"
    assert materials[0].payload["content_ref"].startswith("sha256:")
    assert str(tmp_path) not in repr(materials[0].to_dict())


def test_confirmation_service_requires_current_open_confirmation(tmp_path: Path) -> None:
    """没有 Open Confirmation 时，服务不能猜测候选或创建新任务。"""

    memory = ResearchTaskWorkingMemory(tmp_path / "research_task.sqlite3")
    service = HypothesisConfirmationService(memory)
    memory.create_research_task(
        research_task_id="research-task-no-confirmation",
        **_task_kwargs(),
    )
    with pytest.raises(HypothesisConfirmationStateConflict):
        service.read_view("research-task-no-confirmation")
