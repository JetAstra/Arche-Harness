"""canonical Module Invocation 输出成组合同回归。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from chemistry_multiagent.memory import (
    ContractValidationError,
    IdempotencyConflict,
    ModuleInvocationRequest,
    PreparedModuleInvocationOutput,
    ResearchTaskWorkingMemory,
    ScientificRecord,
    StatelessModule,
    WorkingMemoryIntegrityConflict,
)
from chemistry_multiagent.memory.v2_identities import CreateInvestigationContext


def _task_kwargs(*, entry_strategy: str = "standard_hypothesis_generation") -> dict[str, Any]:
    """返回最小完整的 research_task 创建参数。"""

    return {
        "contract_version": "investigation_v2.task.v2",
        "input": {"scientific_question": "module output contract"},
        "policy": {"entry_strategy": entry_strategy},
        "dependencies": {"catalog": {"ref": "catalog-1"}},
    }


def _memory(tmp_path: Path) -> ResearchTaskWorkingMemory:
    """在测试专用目录创建 Working Memory。"""

    return ResearchTaskWorkingMemory(tmp_path / "research_task.sqlite3")


def _record(
    *,
    record_id: str,
    invocation_id: str,
    record_type: str,
    payload: dict[str, Any],
    source_ids: tuple[str, ...] = (),
    schema_version: str | int = 1,
    research_task_id: str = "research-task-1",
) -> ScientificRecord:
    """用 canonical 字段构造不可变 Scientific Record。"""

    return ScientificRecord.from_payload(
        scientific_record_id=record_id,
        research_task_id=research_task_id,
        scientific_record_type=record_type,
        scientific_record_schema_version=schema_version,
        producer_module_invocation_id=invocation_id,
        source_scientific_record_ids=source_ids,
        payload=payload,
    )


def _output(
    invocation_id: str,
    input_digest: str,
    records: tuple[ScientificRecord, ...],
) -> PreparedModuleInvocationOutput:
    """从记录生成带完整摘要的 canonical 输出包。"""

    return PreparedModuleInvocationOutput.from_records(
        module_invocation_id=invocation_id,
        module_input_digest=input_digest,
        output_scientific_records=records,
    )


def _start(
    memory: ResearchTaskWorkingMemory,
    *,
    module_invocation_id: str,
    module_kind: str,
    input_ids: tuple[str, ...] = (),
    input_digest: str = "input-digest",
    context_binding: object | None = None,
) -> ModuleInvocationRequest:
    """按当前账本序号开始一次 Module Invocation。"""

    facts = memory.read_research_task_facts("research-task-1")
    request = ModuleInvocationRequest(
        research_task_id="research-task-1",
        module_invocation_id=module_invocation_id,
        module_kind=module_kind,
        input_scientific_record_ids=input_ids,
        module_input_digest=input_digest,
        context_binding=context_binding or StatelessModule(),
    )
    memory.start_module_invocation(
        "research-task-1",
        facts.research_task_sequence,
        request,
    )
    return request


def _create(memory: ResearchTaskWorkingMemory) -> None:
    """创建测试任务。"""

    memory.create_research_task(
        research_task_id="research-task-1",
        **_task_kwargs(),
    )


def test_scientific_record_and_prepared_output_are_canonical_and_roundtrip() -> None:
    """Scientific Record 与输出包只暴露 research_task/invocation 字段。"""

    record = _record(
        record_id="retrieval-record-1",
        invocation_id="invocation-1",
        record_type="RetrievalRecord",
        payload={
            "summary": "retrieval",
            "citations": [],
            "corpus_snapshot_ref": "corpus-1",
        },
    )
    output = _output("invocation-1", "digest-1", (record,))

    restored_record = ScientificRecord.from_mapping(record.to_dict())
    restored_output = PreparedModuleInvocationOutput.from_mapping(output.to_dict())
    assert restored_record == record
    assert restored_output == output
    assert not {
        "task_id",
        "record_id",
        "record_type",
        "module_run_id",
        "runtime_ref",
        "input_digest",
    }.intersection(record.to_dict())
    assert not {"module_run_id", "input_digest", "output_records"}.intersection(
        output.to_dict()
    )


def test_retrieval_output_is_committed_atomically_and_replayed(
    tmp_path: Path,
) -> None:
    """有效 Retrieval 输出一次提交全部记录，精确重放不追加事件。"""

    memory = _memory(tmp_path)
    _create(memory)
    request = _start(
        memory,
        module_invocation_id="retrieval-invocation",
        module_kind="retrieval",
    )
    record = _record(
        record_id="retrieval-record",
        invocation_id=request.module_invocation_id,
        record_type="RetrievalRecord",
        payload={
            "summary": "retrieval",
            "citations": [],
            "corpus_snapshot_ref": "corpus-1",
        },
    )
    output = _output(request.module_invocation_id, request.module_input_digest, (record,))
    completed = memory.complete_module_invocation(
        "research-task-1",
        expected_research_task_sequence=2,
        output_bundle=output,
    )
    events = memory.read_research_task_event_log("research-task-1")
    replay = memory.complete_module_invocation(
        "research-task-1", 2, output
    )

    assert completed == replay
    assert completed.lifecycle.state == "Closed"
    assert [event.event_type for event in events] == [
        "ResearchTaskCreated",
        "ModuleInvocationStarted",
        "ScientificRecordCreated",
        "ModuleInvocationCompleted",
    ]
    assert memory.read_research_task_scientific_records(
        "research-task-1", ("retrieval-record",)
    )[0].recorded_at is not None


def test_wrong_record_type_and_digest_are_rejected_without_partial_output(
    tmp_path: Path,
) -> None:
    """输出合同失败时不能留下部分 Scientific Record 或闭合事件。"""

    memory = _memory(tmp_path)
    _create(memory)
    request = _start(
        memory,
        module_invocation_id="retrieval-invocation",
        module_kind="retrieval",
    )
    wrong_record = _record(
        record_id="query-record",
        invocation_id=request.module_invocation_id,
        record_type="QueryRecord",
        payload={"query": "q", "research_angle": "angle", "constraints": {}},
    )
    with pytest.raises(ContractValidationError):
        memory.complete_module_invocation(
            "research-task-1",
            2,
            _output(request.module_invocation_id, request.module_input_digest, (wrong_record,)),
        )
    assert len(memory.read_research_task_event_log("research-task-1")) == 2

    retrieval_record = _record(
        record_id="retrieval-record",
        invocation_id=request.module_invocation_id,
        record_type="RetrievalRecord",
        payload={
            "summary": "retrieval",
            "citations": [],
            "corpus_snapshot_ref": "corpus-1",
        },
    )
    with pytest.raises(IdempotencyConflict):
        memory.complete_module_invocation(
            "research-task-1",
            2,
            _output(request.module_invocation_id, "drifted-digest", (retrieval_record,)),
        )
    assert len(memory.read_research_task_event_log("research-task-1")) == 2


def test_standard_selection_requires_five_selected_records(tmp_path: Path) -> None:
    """标准 Selection 必须在一个输出包中保存恰好五条逐假设记录。"""

    memory = _memory(tmp_path)
    _create(memory)
    generation = _start(
        memory,
        module_invocation_id="generation-invocation",
        module_kind="hypothesis-generation",
    )
    batch = _record(
        record_id="hypothesis-batch",
        invocation_id=generation.module_invocation_id,
        record_type="HypothesisBatchRecord",
        schema_version=1,
        payload={
            "hypotheses": [{"hypothesis_id": f"h-{index}"} for index in range(1, 6)],
            "rationale": "ranked",
            "verification_plan": "verify",
        },
    )
    memory.complete_module_invocation(
        "research-task-1", 2, _output(generation.module_invocation_id, generation.module_input_digest, (batch,))
    )
    selection = _start(
        memory,
        module_invocation_id="selection-invocation",
        module_kind="hypothesis-selection",
        input_ids=("hypothesis-batch",),
    )
    selection_record = _record(
        record_id="selection-record",
        invocation_id=selection.module_invocation_id,
        record_type="HypothesisSelectionRecord",
        schema_version=2,
        source_ids=("hypothesis-batch",),
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
            record_id=f"selected-{index}",
            invocation_id=selection.module_invocation_id,
            record_type="SelectedHypothesisRecord",
            source_ids=("selection-record",),
            payload={
                "hypothesis_ref": f"h-{index}",
                "rank": index,
                "hypothesis": {"hypothesis_id": f"h-{index}"},
            },
        )
        for index in range(1, 6)
    )
    completed = memory.complete_module_invocation(
        "research-task-1",
        memory.read_research_task_facts("research-task-1").research_task_sequence,
        _output(selection.module_invocation_id, selection.module_input_digest, (selection_record, *selected)),
    )
    assert completed.lifecycle.state == "Closed"
    assert len(completed.lifecycle.outcome["output_scientific_record_ids"]) == 6


def test_author_import_outputs_one_to_five_selected_records(tmp_path: Path) -> None:
    """作者入口使用 author-import.v2 Batch 并逐条保存一至五条记录。"""

    memory = _memory(tmp_path)
    memory.create_research_task(
        research_task_id="research-task-1",
        **_task_kwargs(entry_strategy="author_hypothesis_import"),
    )
    invocation = _start(
        memory,
        module_invocation_id="import-invocation",
        module_kind="hypothesis-import",
    )
    batch = _record(
        record_id="author-batch",
        invocation_id=invocation.module_invocation_id,
        record_type="HypothesisBatchRecord",
        schema_version="author-import.v2",
        payload={
            "imported_hypotheses": [
                {
                    "hypothesis_ref": "author-h1",
                    "source_ref": "author-source-1",
                    "source_digest": "a" * 64,
                    "hypothesis": {"hypothesis_id": "author-h1", "claim": "claim"},
                    "provenance": {"kind": "author"},
                }
            ]
        },
    )
    selected = _record(
        record_id="author-selected-1",
        invocation_id=invocation.module_invocation_id,
        record_type="SelectedHypothesisRecord",
        source_ids=("author-batch",),
        payload={
            "hypothesis_ref": "author-h1",
            "rank": 1,
            "hypothesis": {"hypothesis_id": "author-h1", "claim": "claim"},
        },
    )
    completed = memory.complete_module_invocation(
        "research-task-1",
        2,
        _output(invocation.module_invocation_id, invocation.module_input_digest, (batch, selected)),
    )
    assert completed.lifecycle.state == "Closed"


def test_investigation_output_requires_one_evidence_result_pair(tmp_path: Path) -> None:
    """Investigation 输出必须成对保存 Verified Evidence 与 Result。"""

    memory = _memory(tmp_path)
    _create(memory)
    request = _start(
        memory,
        module_invocation_id="investigation-invocation",
        module_kind="hypothesis-investigation",
        context_binding=CreateInvestigationContext("context-1", "batch-1"),
    )
    evidence = _record(
        record_id="evidence-1",
        invocation_id=request.module_invocation_id,
        record_type="VerifiedEvidenceRecord",
        payload={
            "evidence_items": [{"value": "observed"}],
            "artifact_refs": [],
            "source_manifest_ref": "manifest-1",
            "summary": "evidence",
            "verification": {"verified": True},
        },
    )
    result = _record(
        record_id="result-1",
        invocation_id=request.module_invocation_id,
        record_type="InvestigationResultRecord",
        source_ids=("evidence-1",),
        payload={
            "conclusion": "supported",
            "self_judgment": "supported",
            "observations": ["observed"],
            "limitations": [],
            "evidence_ref": "evidence-1",
        },
    )
    completed = memory.complete_module_invocation(
        "research-task-1", 2, _output(request.module_invocation_id, request.module_input_digest, (evidence, result))
    )
    assert completed.lifecycle.state == "Closed"

    invalid_result = _record(
        record_id="result-2",
        invocation_id=request.module_invocation_id,
        record_type="InvestigationResultRecord",
        source_ids=(),
        payload={
            "conclusion": "unsupported",
            "self_judgment": "refuted",
            "observations": [],
            "limitations": [],
            "evidence_ref": "evidence-1",
        },
    )
    with pytest.raises((IdempotencyConflict, WorkingMemoryIntegrityConflict, ContractValidationError)):
        memory.complete_module_invocation(
            "research-task-1", 2, _output(request.module_invocation_id, request.module_input_digest, (evidence, invalid_result))
        )
