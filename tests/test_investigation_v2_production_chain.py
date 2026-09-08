"""research_task 生产装配的完整 Module Invocation 链离线回归。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
from pathlib import Path
from typing import Any

from chemistry_multiagent.controllers.investigation_v2_module_controller import (
    ModuleCompleted,
    ModuleInvocationRequest,
    ModuleSuspended,
)
from chemistry_multiagent.controllers.investigation_v2_production import (
    ResearchTaskProductionAssembly,
)
from chemistry_multiagent.memory import (
    PreparedModuleInvocationOutput,
    ResearchTaskWorkingMemory,
    ScientificRecord,
)


RESEARCH_TASK_ID = "research-task-production-chain"
CANONICAL_REQUEST_FIELDS = {
    "research_task_id",
    "module_invocation_id",
    "module_kind",
    "input_scientific_record_ids",
    "module_input_digest",
    "context_binding",
}


def _task_config() -> dict[str, int]:
    """返回标准五候选 research_task 的冻结预算。"""

    return {
        "num_queries": 1,
        "num_hypotheses_per_query": 5,
        "top_n_hypotheses": 5,
        "max_parallel_investigations": 1,
        "max_investigation_repairs_per_combination": 1,
        "max_hypothesis_batches": 1,
    }


def _task_input(task_config: Mapping[str, int]) -> dict[str, Any]:
    """构造生产装配需要冻结的公开研究输入。"""

    return {
        "scientific_question": "验证 research_task 的完整模块链",
        "retrieval": {
            "pdf_dir": "papers",
            "index_dir": "index",
            "search_papers": False,
            "paper_search": {},
        },
        "task_config": dict(task_config),
        "supplementary_scientific_input": {},
    }


def _task_policy(task_config: Mapping[str, int]) -> dict[str, Any]:
    """构造当前 investigation_v2 的正式任务策略。"""

    return {
        "stage_architecture_version": "investigation_v2",
        "entry_strategy": "standard_hypothesis_generation",
        "task_config": dict(task_config),
    }


def _record(
    request: ModuleInvocationRequest,
    *,
    record_id: str,
    record_type: str,
    payload: Mapping[str, Any],
    source_ids: tuple[str, ...] = (),
    schema_version: str | int = 1,
) -> ScientificRecord:
    """构造绑定到当前 Module Invocation 的科学记录。"""

    return ScientificRecord.from_payload(
        scientific_record_id=record_id,
        research_task_id=request.research_task_id,
        scientific_record_type=record_type,
        scientific_record_schema_version=schema_version,
        producer_module_invocation_id=request.module_invocation_id,
        source_scientific_record_ids=source_ids,
        payload=dict(payload),
    )


def _output(
    request: ModuleInvocationRequest,
    records: tuple[ScientificRecord, ...],
) -> PreparedModuleInvocationOutput:
    """密封当前 Module Invocation 的输出记录。"""

    return PreparedModuleInvocationOutput.from_records(
        module_invocation_id=request.module_invocation_id,
        module_input_digest=request.module_input_digest,
        output_scientific_records=records,
    )


def _preparation_output(request: ModuleInvocationRequest) -> PreparedModuleInvocationOutput:
    """为 Retrieval、Query、Hypothesis 和 Selection 生成合法输出。"""

    if request.module_kind == "retrieval":
        records = (
            _record(
                request,
                record_id=f"{request.module_invocation_id}:retrieval",
                record_type="RetrievalRecord",
                payload={
                    "summary": "deterministic retrieval",
                    "citations": [],
                    "corpus_snapshot_ref": "corpus-production-chain",
                },
            ),
        )
    elif request.module_kind == "research-query-formulation":
        records = (
            _record(
                request,
                record_id=f"{request.module_invocation_id}:query",
                record_type="QueryRecord",
                source_ids=request.input_scientific_record_ids,
                payload={
                    "query": "验证完整模块链",
                    "research_angle": "canonical pipeline",
                    "constraints": {},
                },
            ),
        )
    elif request.module_kind == "hypothesis-generation":
        hypotheses = [
            {"hypothesis_id": f"hypothesis-{rank}", "claim": f"claim-{rank}"}
            for rank in range(1, 6)
        ]
        records = (
            _record(
                request,
                record_id=f"{request.module_invocation_id}:batch",
                record_type="HypothesisBatchRecord",
                source_ids=request.input_scientific_record_ids,
                payload={
                    "hypotheses": hypotheses,
                    "rationale": "deterministic test batch",
                    "verification_plan": "run one selected hypothesis",
                },
            ),
        )
    elif request.module_kind == "hypothesis-selection":
        selection_id = f"{request.module_invocation_id}:selection"
        selection = _record(
            request,
            record_id=selection_id,
            record_type="HypothesisSelectionRecord",
            source_ids=request.input_scientific_record_ids,
            schema_version=2,
            payload={
                "selected_hypotheses": [
                    {"hypothesis_ref": f"hypothesis-{rank}", "rank": rank}
                    for rank in range(1, 6)
                ],
                "selection_reason": "deterministic top five",
            },
        )
        selected = tuple(
            _record(
                request,
                record_id=f"{request.module_invocation_id}:selected:{rank}",
                record_type="SelectedHypothesisRecord",
                source_ids=(selection_id, *request.input_scientific_record_ids),
                payload={
                    "hypothesis_ref": f"hypothesis-{rank}",
                    "rank": rank,
                    "hypothesis": {
                        "hypothesis_id": f"hypothesis-{rank}",
                        "claim": f"claim-{rank}",
                    },
                },
            )
            for rank in range(1, 6)
        )
        records = (selection, *selected)
    else:  # pragma: no cover - 测试数据错误
        raise AssertionError(f"不支持准备阶段模块：{request.module_kind}")
    return _output(request, records)


def _investigation_output(request: ModuleInvocationRequest) -> PreparedModuleInvocationOutput:
    """为每个确认后的执行输入生成一一配对的证据和结果。"""

    records: list[ScientificRecord] = []
    for rank, input_id in enumerate(request.input_scientific_record_ids, start=1):
        evidence_id = f"{request.module_invocation_id}:evidence:{rank}"
        evidence = _record(
            request,
            record_id=evidence_id,
            record_type="VerifiedEvidenceRecord",
            source_ids=(input_id,),
            payload={
                "evidence_items": [{"value": "deterministic observation"}],
                "artifact_refs": [],
                "source_manifest_ref": f"manifest-{rank}",
                "summary": "verified test evidence",
                "verification": {"verified": True},
            },
        )
        result = _record(
            request,
            record_id=f"{request.module_invocation_id}:result:{rank}",
            record_type="InvestigationResultRecord",
            source_ids=(input_id, evidence_id),
            payload={
                "conclusion": "deterministic result",
                "self_judgment": "supported",
                "observations": ["deterministic observation"],
                "limitations": [],
                "evidence_ref": evidence_id,
            },
        )
        records.extend((evidence, result))
    return _output(request, tuple(records))


def _review_output(request: ModuleInvocationRequest) -> PreparedModuleInvocationOutput:
    """生成接受当前结果的终态 Review 输出。"""

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
        payload={
            "answer": "完整 research_task 模块链已完成。",
            "answerability": "answerable",
        },
    )
    return _output(request, (review, final))


class RecordingModule:
    """不访问模型或 MiniChem 的确定性 Module Invocation 替身。"""

    def __init__(
        self,
        factory: Callable[[ModuleInvocationRequest], PreparedModuleInvocationOutput],
    ) -> None:
        self.factory = factory
        self.calls: list[tuple[str, ModuleInvocationRequest]] = []

    def start(self, request: ModuleInvocationRequest) -> ModuleCompleted:
        """记录正式请求并返回准备好的输出。"""

        self.calls.append(("start", request))
        return ModuleCompleted(self.factory(request))

    def resume(self, request: ModuleInvocationRequest) -> ModuleCompleted:
        """记录恢复请求并返回准备好的输出。"""

        self.calls.append(("resume", request))
        return ModuleCompleted(self.factory(request))


def _modules() -> dict[str, RecordingModule]:
    """装配完整标准链所需的正式 module kind。"""

    return {
        "retrieval": RecordingModule(_preparation_output),
        "research-query-formulation": RecordingModule(_preparation_output),
        "hypothesis-generation": RecordingModule(_preparation_output),
        "hypothesis-selection": RecordingModule(_preparation_output),
        "hypothesis-investigation": RecordingModule(_investigation_output),
        "task-review": RecordingModule(_review_output),
    }


class _Application:
    """提供生产装配所需的持久 research_task 根目录配置。"""

    def __init__(self, storage_root: Path) -> None:
        self._run_storage_root = storage_root
        self.memory_db_path = None
        self._memory_db_explicit = False


def test_production_assembly_runs_canonical_research_task_chain(
    tmp_path: Path,
) -> None:
    """生产装配从准备、Top 5 确认到 Investigation 和 Review 使用同一 task。"""

    task_config = _task_config()
    modules = _modules()
    assembly = ResearchTaskProductionAssembly(
        _Application(tmp_path),
        module_adapters=modules,
    )

    waiting = assembly.create_research_task(
        research_task_id=RESEARCH_TASK_ID,
        scientific_question="验证 research_task 的完整模块链",
        task_config=task_config,
        task_input=_task_input(task_config),
        policy=_task_policy(task_config),
        dependencies={"catalog": {"status": "deferred_to_investigation_adapter"}},
    )

    assert waiting["research_task_id"] == RESEARCH_TASK_ID
    assert waiting["status"] == "open"
    assert waiting["reason_code"] == "awaiting_researcher_confirmation"
    assert len(waiting["candidates"]) == 5
    assert [
        module_kind
        for module_kind, module in modules.items()
        if module.calls
    ] == [
        "retrieval",
        "research-query-formulation",
        "hypothesis-generation",
        "hypothesis-selection",
    ]

    confirmation = assembly.get_hypothesis_confirmation(RESEARCH_TASK_ID)
    decisions = [
        {
            "hypothesis_candidate_record_id": candidate[
                "hypothesis_candidate_record_id"
            ],
            "decision": "execute" if candidate["rank"] == 1 else "cancel",
            **(
                {"hypothesis": candidate["hypothesis"]}
                if candidate["rank"] == 1
                else {}
            ),
        }
        for candidate in confirmation["candidates"]
    ]
    completed = assembly.submit_hypothesis_confirmation_and_continue(
        RESEARCH_TASK_ID,
        confirmation_id="confirmation-production-chain",
        expected_research_task_sequence=confirmation[
            "expected_research_task_sequence"
        ],
        decisions=decisions,
    )

    assert completed["research_task_id"] == RESEARCH_TASK_ID
    assert completed["status"] == "succeeded"
    assert isinstance(completed["outcome"], Mapping)

    requests = [
        request
        for module in modules.values()
        for _operation, request in module.calls
    ]
    assert [request.module_kind for request in requests] == [
        "retrieval",
        "research-query-formulation",
        "hypothesis-generation",
        "hypothesis-selection",
        "hypothesis-investigation",
        "task-review",
    ]
    for request in requests:
        public = request.to_public_dict()
        assert set(public) == CANONICAL_REQUEST_FIELDS
        assert public["research_task_id"] == RESEARCH_TASK_ID
        assert public["module_invocation_id"]
        assert isinstance(public["input_scientific_record_ids"], list)
        assert public["module_input_digest"]
        assert isinstance(public["context_binding"], Mapping)

    database_path = (
        tmp_path
        / "outputs"
        / "research_tasks"
        / RESEARCH_TASK_ID
        / "internal"
        / "state"
        / "arche_memory.sqlite3"
    )
    facts = ResearchTaskWorkingMemory(database_path).read_research_task_facts(
        RESEARCH_TASK_ID
    )
    assert isinstance(facts.lifecycle.state, str)
    assert facts.lifecycle.state == "Closed"
    assert [module.module_kind for module in facts.module_invocations] == [
        "retrieval",
        "research-query-formulation",
        "hypothesis-generation",
        "hypothesis-selection",
        "hypothesis-confirmation",
        "hypothesis-investigation",
        "task-review",
    ]
    assert facts.lifecycle.outcome["status"] == "succeeded"

    bundle_root = tmp_path / "outputs" / "research_tasks" / RESEARCH_TASK_ID
    assert (bundle_root / "status.json").is_file()
    assert (bundle_root / "summary.json").is_file()
    assert not (tmp_path / "outputs" / "runs").exists()
    status = json.loads((bundle_root / "status.json").read_text(encoding="utf-8"))
    summary = json.loads((bundle_root / "summary.json").read_text(encoding="utf-8"))
    assert status["research_task_id"] == RESEARCH_TASK_ID
    assert summary["research_task_id"] == RESEARCH_TASK_ID


def test_production_assembly_preserves_open_invocation_for_resume(
    tmp_path: Path,
) -> None:
    """生产入口暂停后恢复同一个 Module Invocation，不重新创建身份。"""

    task_config = _task_config()
    modules = _modules()
    first_attempt = True

    class SuspendedModule(RecordingModule):
        """第一次调用暂停，恢复时返回正常输出。"""

        def start(self, request: ModuleInvocationRequest) -> Any:
            self.calls.append(("start", request))
            nonlocal first_attempt
            if first_attempt:
                first_attempt = False
                return ModuleSuspended("provider_unavailable")
            return ModuleCompleted(self.factory(request))

    modules["retrieval"] = SuspendedModule(_preparation_output)
    assembly = ResearchTaskProductionAssembly(
        _Application(tmp_path),
        module_adapters=modules,
    )

    waiting = assembly.create_research_task(
        research_task_id=RESEARCH_TASK_ID,
        scientific_question="验证暂停恢复",
        task_config=task_config,
        task_input=_task_input(task_config),
        policy=_task_policy(task_config),
        dependencies={},
    )
    assert waiting["status"] == "open"
    first_request = modules["retrieval"].calls[0][1]
    assert waiting["module_invocation_id"] == first_request.module_invocation_id

    resumed = assembly.resume_research_task(RESEARCH_TASK_ID)
    assert resumed["status"] == "open"
    assert resumed["reason_code"] == "awaiting_researcher_confirmation"
    assert modules["retrieval"].calls[1][0] == "resume"
    assert modules["retrieval"].calls[1][1] == first_request
