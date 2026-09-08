"""V2-native Task Review runtime 的公开 seam 回归。

测试只使用确定性 Working Memory、snapshot builder 和 Review runtime 替代实现；
不启动真实模型、MCP、MiniChem 或化学作业。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from chemistry_multiagent.harness.base import HarnessError
from chemistry_multiagent.investigation.codex_runtime import InvestigationTurnResult
from chemistry_multiagent.investigation.contracts import (
    TaskFinalizationResponse,
    TaskReviewResponse,
)
from chemistry_multiagent.investigation.evidence import (
    EvidenceSnapshotNotFound,
    ReviewEvidenceStore,
)
from chemistry_multiagent.investigation.recovery_contract import RecoveryStatus
from chemistry_multiagent.investigation.review_runtime import (
    V2ReviewCompleted,
    V2ReviewFailed,
    V2ReviewSuspended,
    V2TaskReviewModuleAdapter,
    ResearchTaskReviewRunner,
    build_v2_task_review_adapter,
    build_v2_task_review_runtime,
)
from chemistry_multiagent.controllers.investigation_v2_module_controller import (
    ResearchWorkflowController,
    ModuleCompleted,
    ModuleSuspended,
    ResearchTaskClosedResult,
)
import chemistry_multiagent.controllers.investigation_v2_native_adapters as native_adapters
import chemistry_multiagent.investigation.review_runtime as review_runtime_module
from chemistry_multiagent.controllers.investigation_v2_native_adapters import (
    build_native_review_module_adapter,
)
from chemistry_multiagent.memory import (
    ResearchTaskWorkingMemory,
    ModuleInvocationRequest,
    PreparedModuleInvocationOutput,
    ScientificRecord,
)
from chemistry_multiagent.memory.v2_identities import (
    CreateInvestigationContext,
    CreateResearchTaskReviewSession,
    ReuseInvestigationContext,
    StatelessModule,
)


def test_current_review_claim_rejects_schema_v10_identity_field() -> None:
    """当前 V2 Evidence claim 不得把旧 combination_id 当作隐式别名。"""

    with pytest.raises(
        review_runtime_module._ReviewIntegrityConflict,  # noqa: SLF001
        match="review_result_legacy_identity_field_forbidden",
    ):
        review_runtime_module._claim_identity_from_mapping(  # noqa: SLF001
            {
                "combination_id": "legacy-combination-1",
                "result_version": "v1",
            }
        )


def test_review_reads_confirmation_v2_execution_input_refs() -> None:
    """Review 必须从当前 Confirmation v2 Decision 读取执行输入集合。"""

    research_task_id = "task-review-confirmation-v2"
    decision = _record(
        research_task_id=research_task_id,
        module_invocation_id="confirmation-1",
        scientific_record_id="decision-1",
        scientific_record_type="HypothesisExecutionDecisionRecord",
        scientific_record_schema_version="hypothesis-confirmation.v2",
        source_scientific_record_ids=("batch-1",),
        payload={
            "source_hypothesis_batch_record_id": "batch-1",
            "confirmation_id": "confirmation-1",
            "decisions": [
                {
                    "hypothesis_candidate_record_id": "selected-1",
                    "decision": "execute",
                    "execution_rank": 1,
                }
            ],
            "execution_input_scientific_record_ids": ["execution-1"],
        },
    )
    execution_input = _record(
        research_task_id=research_task_id,
        module_invocation_id="confirmation-1",
        scientific_record_id="execution-1",
        scientific_record_type="HypothesisExecutionInputRecord",
        scientific_record_schema_version="hypothesis-confirmation.v2",
        source_scientific_record_ids=("decision-1", "selected-1"),
        payload={
            "source_selected_hypothesis_record_id": "selected-1",
            "execution_rank": 1,
            "hypothesis": {
                "strategy_name": "比较两条对映路径",
                "reasoning": "比较关键过渡态。",
            },
            "guidance": None,
            "material_upload_refs": [],
        },
    )

    class _Memory:
        def read_research_task_scientific_records(
            self, task_id: str, record_ids: tuple[str, ...]
        ) -> tuple[ScientificRecord, ...]:
            assert task_id == research_task_id
            records = {
                decision.scientific_record_id: decision,
                execution_input.scientific_record_id: execution_input,
            }
            return tuple(records[record_id] for record_id in record_ids)

    facts = {
        "module_invocations": [
            {
                "module_kind": "hypothesis-confirmation",
                "lifecycle": {
                    "state": "Closed",
                    "outcome": {
                        "output_scientific_record_ids": [
                            "decision-1",
                            "execution-1",
                        ]
                    },
                },
            }
        ]
    }

    assert review_runtime_module._confirmed_execution_input_refs(  # noqa: SLF001
        facts,
        _Memory(),
        research_task_id,
    ) == ("execution-1",)


def _record(
    *,
    research_task_id: str,
    module_invocation_id: str,
    scientific_record_id: str,
    scientific_record_type: str,
    payload: Mapping[str, Any],
    source_scientific_record_ids: Sequence[str] = (),
    scientific_record_schema_version: str | int = 1,
) -> ScientificRecord:
    """通过 V2 记录工厂构造严格科学事实。"""

    return ScientificRecord.from_payload(
        scientific_record_id=scientific_record_id,
        research_task_id=research_task_id,
        scientific_record_type=scientific_record_type,
        scientific_record_schema_version=scientific_record_schema_version,
        producer_module_invocation_id=module_invocation_id,
        source_scientific_record_ids=tuple(source_scientific_record_ids),
        payload=dict(payload),
    )


def _make_v2_task(
    tmp_path: Path,
    *,
    artifacts: Sequence[str] = (),
    result_count: int = 1,
    entry_strategy: str = "author_main_hypothesis_import",
    contract_version: str | int = 1,
    confirmed_count: int | None = None,
    repair_budget: int = 3,
) -> tuple[
    ResearchTaskWorkingMemory,
    tuple[str, ...],
]:
    """建立一项具有当前 Investigation Result head 的真实 V2 账本。"""

    research_task_id = "task-review"
    module_invocation_id = "investigation-1"
    service = ResearchTaskWorkingMemory(str(tmp_path / "memory.sqlite3"))
    service.create_research_task(
        research_task_id=research_task_id,
        contract_version=contract_version,
        input={
            "scientific_question": "Which route is supported?",
            "investigation_context_id": "context-task-review",
            "task_config": {
                "review_model": "deepseek-v4-pro",
                "max_investigation_repairs_per_combination": repair_budget,
            },
        },
        policy={"entry_strategy": entry_strategy, "review_model": "deepseek-v4-pro"},
        dependencies={"tool_catalog_hash": "catalog"},
    )
    facts = service.read_research_task_facts(research_task_id)
    investigation_input_refs: tuple[str, ...] = ()
    if contract_version == "investigation_v2.task.v2":
        if confirmed_count is None:
            confirmed_count = result_count
        import_run_id = "hypothesis-import-1"
        service.start_module_invocation(
            research_task_id,
            facts.research_task_sequence,
            ModuleInvocationRequest(
                research_task_id=research_task_id,
                module_invocation_id=import_run_id,
                module_kind="hypothesis-import",
                input_scientific_record_ids=(),
                module_input_digest="hypothesis-import-input",
                context_binding=StatelessModule(),
            ),
        )
        batch_id = "batch-1"
        imported = [
            {
                "hypothesis_ref": f"hypothesis-{index}",
                "source_ref": f"author-source-{index}",
                "source_digest": f"{index:064x}",
                "hypothesis": {
                    "hypothesis_id": f"hypothesis-{index}",
                    "claim": f"claim-{index}",
                },
                "provenance": {"kind": "review-test"},
            }
            for index in range(1, 6)
        ]
        batch = _record(
            research_task_id=research_task_id,
            module_invocation_id=import_run_id,
            scientific_record_id=batch_id,
            scientific_record_type="HypothesisBatchRecord",
            scientific_record_schema_version="author-import.v2",
            payload={"imported_hypotheses": imported},
        )
        selected_records = tuple(
            _record(
                research_task_id=research_task_id,
                module_invocation_id=import_run_id,
                scientific_record_id=f"selected-{index}",
                scientific_record_type="SelectedHypothesisRecord",
                source_scientific_record_ids=(batch_id,),
                payload={
                    "hypothesis_ref": item["hypothesis_ref"],
                    "rank": index,
                    "hypothesis": item["hypothesis"],
                },
            )
            for index, item in enumerate(imported, start=1)
        )
        service.complete_module_invocation(
            research_task_id,
            expected_research_task_sequence=facts.research_task_sequence + 1,
            output_bundle=PreparedModuleInvocationOutput.from_records(
                module_invocation_id=import_run_id,
                module_input_digest="hypothesis-import-input",
                output_scientific_records=(batch, *selected_records),
            ),
        )
        facts = service.read_research_task_facts(research_task_id)
        confirmation_run_id = "confirmation-1"
        service.start_module_invocation(
            research_task_id,
            facts.research_task_sequence,
            ModuleInvocationRequest(
                research_task_id=research_task_id,
                module_invocation_id=confirmation_run_id,
                module_kind="hypothesis-confirmation",
                input_scientific_record_ids=(),
                module_input_digest="confirmation-input",
                context_binding=StatelessModule(),
            ),
        )
        decision_id = "confirmation-decision"
        investigation_input_refs = tuple(
            f"execution-input-{index}"
            for index in range(1, confirmed_count + 1)
        )
        decision = _record(
            research_task_id=research_task_id,
            module_invocation_id=confirmation_run_id,
            scientific_record_id=decision_id,
            scientific_record_type="HypothesisExecutionDecisionRecord",
            source_scientific_record_ids=(batch_id,),
            payload={
                "source_hypothesis_batch_ref": batch_id,
                "confirmation_id": "confirmation-request-1",
                "decisions": [
                    {
                        "candidate_ref": f"hypothesis-{index}",
                        "decision": (
                            "execute" if index <= confirmed_count else "cancel"
                        ),
                        **(
                            {"execution_rank": index}
                            if index <= confirmed_count
                            else {}
                        ),
                    }
                    for index in range(1, 6)
                ],
                "execution_input_refs": list(investigation_input_refs),
            },
        )
        execution_inputs = tuple(
            _record(
                research_task_id=research_task_id,
                module_invocation_id=confirmation_run_id,
                scientific_record_id=scientific_record_id,
                scientific_record_type="HypothesisExecutionInputRecord",
                source_scientific_record_ids=(decision_id, f"selected-{index}"),
                payload={
                    "source_selected_hypothesis_ref": f"selected-{index}",
                    "execution_rank": index,
                    "hypothesis": {
                        "hypothesis_id": f"hypothesis-{index}",
                        "claim": f"claim-{index}",
                    },
                    "guidance": "session-only-guidance-do-not-copy-to-review",
                    "material_refs": [],
                },
            )
            for index, scientific_record_id in enumerate(investigation_input_refs, start=1)
        )
        service.complete_module_invocation(
            research_task_id,
            expected_research_task_sequence=facts.research_task_sequence + 1,
            output_bundle=PreparedModuleInvocationOutput.from_records(
                module_invocation_id=confirmation_run_id,
                module_input_digest="confirmation-input",
                output_scientific_records=(decision, *execution_inputs),
            ),
        )
        facts = service.read_research_task_facts(research_task_id)
    service.start_module_invocation(
        research_task_id,
        facts.research_task_sequence,
        ModuleInvocationRequest(
            research_task_id=research_task_id,
            module_invocation_id=module_invocation_id,
                module_kind="hypothesis-investigation",
                input_scientific_record_ids=investigation_input_refs,
                module_input_digest="investigation-input",
                context_binding=CreateInvestigationContext(
                    investigation_context_id="context-task-review",
                    hypothesis_batch_record_id="batch-1",
                ),
        ),
    )
    records: list[ScientificRecord] = []
    result_ids: list[str] = []
    for index in range(1, result_count + 1):
        evidence_id = f"evidence-{index}"
        result_id = f"result-{index}"
        evidence = _record(
            research_task_id=research_task_id,
            module_invocation_id=module_invocation_id,
            scientific_record_id=evidence_id,
            scientific_record_type="VerifiedEvidenceRecord",
            payload={
                "evidence_items": [{"kind": "deterministic"}],
                "artifact_refs": list(artifacts),
                "source_manifest_ref": f"manifest-{index}",
                "summary": "verified evidence",
                "verification": {
                    "status": "verified",
                    "hypothesis_execution_input_record_id": (
                        investigation_input_refs[index - 1]
                        if index <= len(investigation_input_refs)
                        else f"combo-{index}"
                    ),
                    "result_version": "v1",
                },
            },
        )
        result = _record(
            research_task_id=research_task_id,
            module_invocation_id=module_invocation_id,
            scientific_record_id=result_id,
            scientific_record_type="InvestigationResultRecord",
            source_scientific_record_ids=(
                evidence_id,
                *(
                    (investigation_input_refs[index - 1],)
                    if index <= len(investigation_input_refs)
                    else ()
                ),
            ),
            payload={
                "conclusion": f"Route {index} is supported.",
                "self_judgment": "supported",
                "observations": ["deterministic observation"],
                "limitations": [],
                "evidence_ref": evidence_id,
            },
        )
        records.extend((evidence, result))
        result_ids.append(result_id)
    service.complete_module_invocation(
        research_task_id,
        expected_research_task_sequence=facts.research_task_sequence + 1,
        output_bundle=PreparedModuleInvocationOutput.from_records(
            module_invocation_id=module_invocation_id,
            module_input_digest="investigation-input",
            output_scientific_records=tuple(records),
        ),
    )
    return service, tuple(result_ids)


def _append_repaired_result(
    service: ResearchTaskWorkingMemory,
    *,
    source_result_ref: str,
    round_number: int,
) -> str:
    """追加一轮 Review 返修与其结果前驱，用于验证冻结预算。"""

    research_task_id = "task-review"
    source_result = service.read_research_task_scientific_records(
        research_task_id,
        (source_result_ref,),
    )[0]
    execution_input_refs = tuple(
        ref
        for ref in source_result.source_scientific_record_ids
        if ref.startswith("execution-input-")
    )
    facts = service.read_research_task_facts(research_task_id)
    review_run_id = f"review-{round_number}"
    review_digest = f"review-input-{round_number}"
    service.start_module_invocation(
        research_task_id,
        facts.research_task_sequence,
        ModuleInvocationRequest(
            research_task_id=research_task_id,
            module_invocation_id=review_run_id,
            module_kind="task-review",
            input_scientific_record_ids=(source_result_ref,),
            module_input_digest=review_digest,
            context_binding=CreateResearchTaskReviewSession(
                research_task_review_session_id="review-session-task-review",
            ),
        ),
    )
    review_ref = f"{review_run_id}:review"
    repair_ref = f"{review_run_id}:repair"
    review = _record(
        research_task_id=research_task_id,
        module_invocation_id=review_run_id,
        scientific_record_id=review_ref,
        scientific_record_type="ReviewRecord",
        source_scientific_record_ids=(source_result_ref,),
        payload={
            "reviewed_result_refs": [source_result_ref],
            "findings": [{"validity": "fail"}],
            "outcome": "RepairRequired",
        },
    )
    repair = _record(
        research_task_id=research_task_id,
        module_invocation_id=review_run_id,
        scientific_record_id=repair_ref,
        scientific_record_type="RepairRequestRecord",
        source_scientific_record_ids=(review_ref, source_result_ref),
        payload={
            "result_ref": source_result_ref,
            "problem": "verification missing",
            "instructions": "repeat verification",
        },
    )
    service.complete_module_invocation(
        research_task_id,
        expected_research_task_sequence=facts.research_task_sequence + 1,
        output_bundle=PreparedModuleInvocationOutput.from_records(
            module_invocation_id=review_run_id,
            module_input_digest=review_digest,
            output_scientific_records=(review, repair),
        ),
    )

    facts = service.read_research_task_facts(research_task_id)
    investigation_run_id = f"investigation-repair-{round_number}"
    investigation_digest = f"investigation-repair-input-{round_number}"
    service.start_module_invocation(
        research_task_id,
        facts.research_task_sequence,
        ModuleInvocationRequest(
            research_task_id=research_task_id,
            module_invocation_id=investigation_run_id,
            module_kind="hypothesis-investigation",
            input_scientific_record_ids=(
                review_ref,
                repair_ref,
                source_result_ref,
                *execution_input_refs,
            ),
            module_input_digest=investigation_digest,
            context_binding=ReuseInvestigationContext(
                investigation_context_id="context-task-review",
                hypothesis_batch_record_id="batch-1",
                source_module_invocation_id="investigation-1",
            ),
        ),
    )
    evidence_ref = f"repair-evidence-{round_number}"
    result_ref = f"repair-result-{round_number}"
    evidence = _record(
        research_task_id=research_task_id,
        module_invocation_id=investigation_run_id,
        scientific_record_id=evidence_ref,
        scientific_record_type="VerifiedEvidenceRecord",
        source_scientific_record_ids=(
            review_ref,
            repair_ref,
            source_result_ref,
            *execution_input_refs,
        ),
        payload={
            "evidence_items": [{"kind": "repaired"}],
            "artifact_refs": [],
            "source_manifest_ref": f"repair-manifest-{round_number}",
            "summary": "repaired evidence",
            "verification": {
                "status": "verified",
                "hypothesis_execution_input_record_id": (
                    execution_input_refs[0] if execution_input_refs else "combo-1"
                ),
                "result_version": f"v{round_number + 1}",
            },
        },
    )
    result = _record(
        research_task_id=research_task_id,
        module_invocation_id=investigation_run_id,
        scientific_record_id=result_ref,
        scientific_record_type="InvestigationResultRecord",
        source_scientific_record_ids=(
            evidence_ref,
            source_result_ref,
            *execution_input_refs,
        ),
        payload={
            "conclusion": "Repaired route result.",
            "self_judgment": "supported",
            "observations": ["repaired observation"],
            "limitations": ["repair budget consumed"],
            "evidence_ref": evidence_ref,
        },
    )
    service.complete_module_invocation(
        research_task_id,
        expected_research_task_sequence=facts.research_task_sequence + 1,
        output_bundle=PreparedModuleInvocationOutput.from_records(
            module_invocation_id=investigation_run_id,
            module_input_digest=investigation_digest,
            output_scientific_records=(evidence, result),
        ),
    )
    return result_ref


def _append_confirmed_result_round(
    service: ResearchTaskWorkingMemory,
    *,
    round_number: int,
) -> str:
    """追加另一轮 Confirmation 与 Result，保留前一轮不可变结果。"""

    research_task_id = "task-review"
    facts = service.read_research_task_facts(research_task_id)
    hypothesis_run_id = f"hypothesis-import-{round_number}"
    hypothesis_digest = f"hypothesis-import-input-{round_number}"
    batch_id = f"batch-round-{round_number}"
    service.start_module_invocation(
        research_task_id,
        facts.research_task_sequence,
        ModuleInvocationRequest(
            research_task_id=research_task_id,
            module_invocation_id=hypothesis_run_id,
            module_kind="hypothesis-import",
            input_scientific_record_ids=(),
            module_input_digest=hypothesis_digest,
            context_binding=StatelessModule(),
        ),
    )
    imported = [
        {
            "hypothesis_ref": f"hypothesis-round-{round_number}-{index}",
            "source_ref": f"author-source-round-{round_number}-{index}",
            "source_digest": f"{round_number * 10 + index:064x}",
            "hypothesis": {
                "hypothesis_id": f"hypothesis-round-{round_number}-{index}",
                "claim": f"claim-round-{round_number}-{index}",
            },
            "provenance": {"kind": "review-test"},
        }
        for index in range(1, 6)
    ]
    batch = _record(
        research_task_id=research_task_id,
        module_invocation_id=hypothesis_run_id,
        scientific_record_id=batch_id,
        scientific_record_type="HypothesisBatchRecord",
        scientific_record_schema_version="author-import.v2",
        payload={"imported_hypotheses": imported},
    )
    selected_records = tuple(
        _record(
            research_task_id=research_task_id,
            module_invocation_id=hypothesis_run_id,
            scientific_record_id=f"selected-round-{round_number}-{index}",
            scientific_record_type="SelectedHypothesisRecord",
            source_scientific_record_ids=(batch_id,),
            payload={
                "hypothesis_ref": item["hypothesis_ref"],
                "rank": index,
                "hypothesis": item["hypothesis"],
            },
        )
        for index, item in enumerate(imported, start=1)
    )
    service.complete_module_invocation(
        research_task_id,
        expected_research_task_sequence=facts.research_task_sequence + 1,
        output_bundle=PreparedModuleInvocationOutput.from_records(
            module_invocation_id=hypothesis_run_id,
            module_input_digest=hypothesis_digest,
            output_scientific_records=(batch, *selected_records),
        ),
    )

    facts = service.read_research_task_facts(research_task_id)
    confirmation_run_id = f"confirmation-{round_number}"
    confirmation_digest = f"confirmation-input-{round_number}"
    service.start_module_invocation(
        research_task_id,
        facts.research_task_sequence,
        ModuleInvocationRequest(
            research_task_id=research_task_id,
            module_invocation_id=confirmation_run_id,
            module_kind="hypothesis-confirmation",
            input_scientific_record_ids=tuple(record.scientific_record_id for record in selected_records),
            module_input_digest=confirmation_digest,
            context_binding=StatelessModule(),
        ),
    )
    decision_id = f"confirmation-decision-{round_number}"
    execution_input_id = f"execution-input-{round_number}-1"
    decision = _record(
        research_task_id=research_task_id,
        module_invocation_id=confirmation_run_id,
        scientific_record_id=decision_id,
        scientific_record_type="HypothesisExecutionDecisionRecord",
        source_scientific_record_ids=(batch_id,),
        payload={
            "source_hypothesis_batch_ref": batch_id,
            "confirmation_id": f"confirmation-request-{round_number}",
            "decisions": [
                {
                    "candidate_ref": f"hypothesis-round-{round_number}-{index}",
                    "decision": "execute" if index == 1 else "cancel",
                    **({"execution_rank": 1} if index == 1 else {}),
                }
                for index in range(1, 6)
            ],
            "execution_input_refs": [execution_input_id],
        },
    )
    execution_input = _record(
        research_task_id=research_task_id,
        module_invocation_id=confirmation_run_id,
        scientific_record_id=execution_input_id,
        scientific_record_type="HypothesisExecutionInputRecord",
        source_scientific_record_ids=(decision_id, f"selected-round-{round_number}-1"),
        payload={
            "source_selected_hypothesis_ref": f"selected-round-{round_number}-1",
            "execution_rank": 1,
            "hypothesis": {
                "hypothesis_id": f"hypothesis-round-{round_number}",
                "claim": f"claim-round-{round_number}",
            },
            "guidance": None,
            "material_refs": [],
        },
    )
    service.complete_module_invocation(
        research_task_id,
        expected_research_task_sequence=facts.research_task_sequence + 1,
        output_bundle=PreparedModuleInvocationOutput.from_records(
            module_invocation_id=confirmation_run_id,
            module_input_digest=confirmation_digest,
            output_scientific_records=(decision, execution_input),
        ),
    )

    facts = service.read_research_task_facts(research_task_id)
    investigation_run_id = f"investigation-{round_number}"
    investigation_digest = f"investigation-input-{round_number}"
    service.start_module_invocation(
        research_task_id,
        facts.research_task_sequence,
        ModuleInvocationRequest(
            research_task_id=research_task_id,
            module_invocation_id=investigation_run_id,
            module_kind="hypothesis-investigation",
            input_scientific_record_ids=(execution_input_id,),
            module_input_digest=investigation_digest,
            context_binding=CreateInvestigationContext(
                investigation_context_id=f"context-round-{round_number}",
                hypothesis_batch_record_id=batch_id,
            ),
        ),
    )
    evidence_id = f"evidence-{round_number}-1"
    result_id = f"result-{round_number}-1"
    evidence = _record(
        research_task_id=research_task_id,
        module_invocation_id=investigation_run_id,
        scientific_record_id=evidence_id,
        scientific_record_type="VerifiedEvidenceRecord",
        source_scientific_record_ids=(execution_input_id,),
        payload={
            "evidence_items": [{"kind": "deterministic"}],
            "artifact_refs": [],
            "source_manifest_ref": f"manifest-{round_number}",
            "summary": "verified evidence",
            "verification": {
                "status": "verified",
                "hypothesis_execution_input_record_id": execution_input_id,
                "result_version": "v1",
            },
        },
    )
    result = _record(
        research_task_id=research_task_id,
        module_invocation_id=investigation_run_id,
        scientific_record_id=result_id,
        scientific_record_type="InvestigationResultRecord",
        source_scientific_record_ids=(evidence_id, execution_input_id),
        payload={
            "conclusion": f"Round {round_number} is supported.",
            "self_judgment": "supported",
            "observations": ["deterministic observation"],
            "limitations": [],
            "evidence_ref": evidence_id,
        },
    )
    service.complete_module_invocation(
        research_task_id,
        expected_research_task_sequence=facts.research_task_sequence + 1,
        output_bundle=PreparedModuleInvocationOutput.from_records(
            module_invocation_id=investigation_run_id,
            module_input_digest=investigation_digest,
            output_scientific_records=(evidence, result),
        ),
    )
    return result_id


class _ReviewRuntime:
    """不访问外部 provider 的 Review runtime 替代实现。"""

    def __init__(self, response: TaskReviewResponse | None = None) -> None:
        self.calls: list[tuple[dict[str, Any], str]] = []
        self.response = response

    def run_task_review_turn(
        self,
        request: Mapping[str, Any],
        *,
        scope_id: str,
    ) -> InvestigationTurnResult:
        """记录公开请求并返回固定 Review 结果。"""

        self.calls.append((dict(request), scope_id))
        response = self.response
        if response is None:
            payload = json.loads(request["messages"][-1]["content"])
            response = TaskReviewResponse(
                findings=[
                    {
                        "hypothesis_execution_input_record_id": item[
                            "hypothesis_execution_input_record_id"
                        ],
                        "validity": "pass",
                        "hypothesis_judgment": "supported",
                    }
                    for item in payload["current_results"]
                ],
                can_answer_original_question=True,
                final_answer="The route is supported.",
            )
        return InvestigationTurnResult(
            result=response,
            thread_id="review-thread",
            turn_id=f"review-turn-{len(self.calls)}",
        )


class _FinalizationRuntime:
    """记录预算耗尽请求并返回终局成稿，不产生返修建议。"""

    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, Any], str]] = []

    def run_task_review_turn(
        self,
        request: Mapping[str, Any],
        *,
        scope_id: str,
    ) -> InvestigationTurnResult:
        self.calls.append((dict(request), scope_id))
        payload = json.loads(request["messages"][-1]["content"])
        execution_input_id = payload["current_results"][0][
            "hypothesis_execution_input_record_id"
        ]
        return InvestigationTurnResult(
            result=TaskFinalizationResponse(
                findings=[
                    {
                        "hypothesis_execution_input_record_id": execution_input_id,
                        "summary": "The repaired evidence is sufficient for a bounded conclusion.",
                    }
                ],
                answerability="answerable",
                final_answer="The available repaired evidence supports the route.",
            ),
            thread_id="review-thread",
            turn_id=f"finalization-turn-{len(self.calls)}",
        )


def _snapshot_builder(review_input: Any) -> dict[str, Any]:
    """从 V2 公开输入生成无路径 Evidence manifest。"""

    return {
        "snapshot_id": "snapshot-v2",
        "research_task_id": review_input.research_task_id,
        "investigation_context_id": review_input.investigation_context_id,
        "result_pairs": [
            {
                "hypothesis_execution_input_record_id": execution_input_id,
                "result_version": result_version,
                "artifacts": [],
            }
            for execution_input_id, result_version in review_input.result_pairs
        ],
    }


def test_review_reads_exact_v2_result_heads_and_frozen_task_context(tmp_path: Path) -> None:
    """Review 请求只由 V2 heads 和冻结 Task input/policy 构造。"""

    service, result_refs = _make_v2_task(tmp_path)
    runtime = _ReviewRuntime()
    adapter = ResearchTaskReviewRunner(
        working_memory=service,
        review_runtime=runtime,
        review_scope_id="review-scope",
        evidence_snapshot_builder=_snapshot_builder,
    )

    outcome = adapter.review_current_result_heads("task-review", result_refs=result_refs)

    assert isinstance(outcome, V2ReviewCompleted)
    assert len(runtime.calls) == 1
    request, scope_id = runtime.calls[0]
    assert scope_id == "review-scope"
    payload = json.loads(request["messages"][-1]["content"])
    assert payload["research_task_id"] == "task-review"
    assert payload["investigation_context_id"] == "context-task-review"
    assert payload["scientific_question"] == "Which route is supported?"
    assert payload["current_results"][0][
        "hypothesis_execution_input_record_id"
    ] == "combo-1"
    assert payload["current_results"][0]["result_version"] == "v1"
    assert "scope_id" not in str(request)
    assert "turn_id" not in str(request)


def test_native_review_rebuilds_artifact_manifest_after_process_restart(
    tmp_path: Path,
) -> None:
    """重新装配 Review 时，从持久 Evidence store 恢复 source manifest。"""

    source_ref = "art_" + "1" * 32
    service, result_refs = _make_v2_task(tmp_path, artifacts=(source_ref,))
    task_root = tmp_path / "bundle"
    source = tmp_path / "result.txt"
    source.write_text("verified result", encoding="utf-8")
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    ReviewEvidenceStore(task_root / "artifacts" / "evidence").register_source_artifact(
        source,
        source_artifact_ref=source_ref,
        expected_source_sha256=source_digest,
        media_type="text/plain",
    )

    # 模拟新 Controller 进程：不传前一进程的 artifact_manifest_registry 对象。
    runtime = _ReviewRuntime()
    adapter = build_native_review_module_adapter(
        task_root=task_root,
        working_memory=service,
        task_input={"scientific_question": "Which route is supported?"},
            policy={"entry_strategy": "author_hypothesis_import"},
        # 默认 Investigation assembly 会注入该空字典；新进程不能把它误当成
        # 已经含有上一进程 manifest 的显式 provider。
        dependencies={
            "review_runtime": runtime,
            "artifact_manifest_registry": {},
        },
    )

    outcome = adapter.start(
        {
            "research_task_id": "task-review",
            "module_invocation_id": "review-restarted",
            "input_scientific_record_ids": result_refs,
            "module_input_digest": "review-restarted-input",
        }
    )

    assert isinstance(outcome, ModuleCompleted)
    request = json.loads(runtime.calls[0][0]["messages"][-1]["content"])
    artifacts = request["evidence_snapshot"]["result_pairs"][0]["artifacts"]
    assert [item["source_artifact_ref"] for item in artifacts] == [source_ref]
    assert all("source_path" not in item for item in artifacts)


def test_native_review_default_runtime_binds_evidence_lease_before_provider_pause(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """默认 Review provider 使用 task Evidence lease，暂不可用仍保持可恢复挂起。"""

    service, result_refs = _make_v2_task(tmp_path)
    task_root = tmp_path / "task-review"
    leases: list[Any] = []
    runtimes: list[Any] = []

    class RecordingLease:
        """记录公开 lease seam，不启动真实 Evidence MCP 子进程。"""

        def __init__(
            self,
            research_task_id: str,
            store_root: Path,
            runtime_root: Path,
            **kwargs: Any,
        ) -> None:
            self.research_task_id = research_task_id
            self.store_root = Path(store_root)
            self.runtime_root = Path(runtime_root)
            self.endpoint = "http://127.0.0.1:9020/mcp"
            self.capability_token = kwargs["capability_token"]
            self.served: list[str] = []
            leases.append(self)

        @contextmanager
        def serve_evidence_snapshot(self, snapshot_id: str):
            self.served.append(snapshot_id)
            yield self

    class RecordingReviewRuntime:
        """模拟真实 runtime 的 MCP 环境门禁和一次可恢复 provider 中断。"""

        def __init__(self, **kwargs: Any) -> None:
            self.environment = dict(kwargs["environment"])
            self.evidence_mcp = kwargs["evidence_mcp"]
            self.sandbox_spec = kwargs["sandbox_spec"]
            runtimes.append(self)

        def run_task_review_turn(
            self,
            _request: Mapping[str, Any],
            *,
            scope_id: str,
        ) -> InvestigationTurnResult:
            del scope_id
            env_name = next(iter(self.evidence_mcp.env_http_headers.values()))
            if env_name not in self.environment:
                raise HarnessError(
                    "MCP capability environment is missing",
                    kind="transport",
                )
            raise HarnessError("provider temporarily unavailable", kind="transport")

    monkeypatch.delenv("ARCHE_EVIDENCE_CAPABILITY_TOKEN", raising=False)
    monkeypatch.setattr(
        native_adapters,
        "TaskEvidenceMcpLease",
        RecordingLease,
        raising=False,
    )
    monkeypatch.setattr(
        native_adapters,
        "InvestigationCodexRuntime",
        RecordingReviewRuntime,
    )

    adapter = build_native_review_module_adapter(
        task_root=task_root,
        working_memory=service,
        task_input={"scientific_question": "Which route is supported?"},
        policy={"entry_strategy": "author_hypothesis_import"},
        dependencies={"evidence_snapshot_builder": _snapshot_builder},
    )

    outcome = adapter.start(
        {
            "research_task_id": "task-review",
            "module_invocation_id": "review-default-lease",
            "input_scientific_record_ids": result_refs,
            "module_input_digest": "review-default-lease-input",
        }
    )

    assert isinstance(outcome, ModuleSuspended)
    assert len(leases) == 1
    assert leases[0].served == ["snapshot-v2"]
    assert len(runtimes) == 1
    env_name = next(iter(runtimes[0].evidence_mcp.env_http_headers.values()))
    assert runtimes[0].environment[env_name] == leases[0].capability_token
    assert runtimes[0].evidence_mcp.endpoint == leases[0].endpoint
    review_owner_root = (
        task_root
        / "internal"
        / "state"
        / "investigation_runtime"
        / "review"
        / "codex_sessions"
        / "task_reviews"
        / "task-review"
    )
    assert runtimes[0].sandbox_spec.workspace == review_owner_root / "workspace"
    assert runtimes[0].sandbox_spec.codex_home == review_owner_root / "codex_home"


def test_author_review_accepts_three_current_result_heads(tmp_path: Path) -> None:
    """作者入口的三条当前 Result 必须作为同一批次一起交给 DS。"""

    service, result_refs = _make_v2_task(tmp_path, result_count=3)
    runtime = _ReviewRuntime(
        TaskReviewResponse(
            findings=[
                {
                    "hypothesis_execution_input_record_id": f"combo-{index}",
                    "validity": "pass",
                    "hypothesis_judgment": "supported",
                }
                for index in range(1, 4)
            ],
            can_answer_original_question=True,
            final_answer="The author routes are supported.",
        )
    )
    adapter = ResearchTaskReviewRunner(
        working_memory=service,
        review_runtime=runtime,
        review_scope_id="review-scope",
        evidence_snapshot_builder=_snapshot_builder,
    )

    outcome = adapter.review_current_result_heads(
        "task-review", result_refs=result_refs
    )

    assert isinstance(outcome, V2ReviewCompleted)
    payload = json.loads(runtime.calls[0][0]["messages"][-1]["content"])
    assert [
        item["hypothesis_execution_input_record_id"]
        for item in payload["current_results"]
    ] == ["combo-1", "combo-2", "combo-3"]


def test_task_v2_standard_review_uses_confirmed_execution_count(
    tmp_path: Path,
) -> None:
    """标准入口取消四条后，Review 接受与 Decision 一致的一条 Result。"""

    service, result_refs = _make_v2_task(
        tmp_path,
        result_count=1,
        entry_strategy="standard_hypothesis_generation",
        contract_version="investigation_v2.task.v2",
        confirmed_count=1,
    )
    runtime = _ReviewRuntime()
    adapter = ResearchTaskReviewRunner(
        working_memory=service,
        review_runtime=runtime,
        review_scope_id="review-scope",
        evidence_snapshot_builder=_snapshot_builder,
    )

    outcome = adapter.review_current_result_heads(
        "task-review", result_refs=result_refs
    )

    assert isinstance(outcome, V2ReviewCompleted)
    assert len(runtime.calls) == 1
    assert "session-only-guidance-do-not-copy-to-review" not in json.dumps(
        runtime.calls[0][0],
        ensure_ascii=False,
    )


def test_task_v2_review_uses_only_latest_confirmation_round(tmp_path: Path) -> None:
    """重新生成并再次确认后，Review 不得混入上一轮 Result head。"""

    service, old_result_refs = _make_v2_task(
        tmp_path,
        contract_version="investigation_v2.task.v2",
        confirmed_count=1,
    )
    new_result_ref = _append_confirmed_result_round(service, round_number=2)
    facts = service.read_research_task_facts("task-review")
    assert set(facts.scientific_graph.record_refs_by_type["InvestigationResultRecord"]) == {
        *old_result_refs,
        new_result_ref,
    }
    runtime = _ReviewRuntime(
        TaskReviewResponse(
            findings=[
                {
                    "hypothesis_execution_input_record_id": "execution-input-2-1",
                    "validity": "pass",
                    "hypothesis_judgment": "supported",
                }
            ],
            can_answer_original_question=True,
            final_answer="The latest route is supported.",
        )
    )
    adapter = ResearchTaskReviewRunner(
        working_memory=service,
        review_runtime=runtime,
        review_scope_id="review-scope",
        evidence_snapshot_builder=_snapshot_builder,
    )

    outcome = adapter.review_current_result_heads(
        "task-review",
        result_refs=(new_result_ref,),
    )

    assert isinstance(outcome, V2ReviewCompleted)
    payload = json.loads(runtime.calls[0][0]["messages"][-1]["content"])
    assert [
        item["hypothesis_execution_input_record_id"]
        for item in payload["current_results"]
    ] == [
        "execution-input-2-1"
    ]


def test_controller_review_input_uses_only_latest_confirmation_round(
    tmp_path: Path,
) -> None:
    """Controller 传给 Review 的 refs 与 runtime 的最新确认轮次定义一致。"""

    service, _old_result_refs = _make_v2_task(
        tmp_path,
        contract_version="investigation_v2.task.v2",
        confirmed_count=1,
    )
    new_result_ref = _append_confirmed_result_round(service, round_number=2)
    controller = ResearchWorkflowController(
        service,
        {},
    )

    assert controller.input_scientific_record_ids_for(
        "task-review", "task-review"
    ) == (new_result_ref,)


def test_task_v2_review_rejects_result_count_different_from_confirmation(
    tmp_path: Path,
) -> None:
    """Result 数量与已确认执行输入不一致时，不调用 Review provider。"""

    service, result_refs = _make_v2_task(
        tmp_path,
        result_count=1,
        entry_strategy="standard_hypothesis_generation",
        contract_version="investigation_v2.task.v2",
        confirmed_count=2,
    )
    runtime = _ReviewRuntime()
    adapter = ResearchTaskReviewRunner(
        working_memory=service,
        review_runtime=runtime,
        review_scope_id="review-scope",
        evidence_snapshot_builder=_snapshot_builder,
    )

    outcome = adapter.review_current_result_heads(
        "task-review", result_refs=result_refs
    )

    assert isinstance(outcome, V2ReviewFailed)
    assert outcome.recovery_verification.reason_code == (
        "review_result_heads_count_invalid"
    )
    assert runtime.calls == []


def test_standard_review_rejects_fewer_than_five_current_result_heads(
    tmp_path: Path,
) -> None:
    """标准入口不能把一条 Result 错当成合法的作者模式输入。"""

    service, result_refs = _make_v2_task(
        tmp_path,
        result_count=1,
        entry_strategy="standard_retrieval",
    )
    runtime = _ReviewRuntime()
    adapter = ResearchTaskReviewRunner(
        working_memory=service,
        review_runtime=runtime,
        review_scope_id="review-scope",
        evidence_snapshot_builder=_snapshot_builder,
    )

    outcome = adapter.review_current_result_heads(
        "task-review", result_refs=result_refs
    )

    assert isinstance(outcome, V2ReviewFailed)
    assert outcome.recovery_verification.reason_code == (
        "review_result_heads_count_invalid"
    )
    assert runtime.calls == []


def test_top5_only_task_cannot_enter_review(tmp_path: Path) -> None:
    """Top 5-only 入口即使存在异常 Result 事实也不能进入 DS Review。"""

    service, result_refs = _make_v2_task(
        tmp_path,
        result_count=5,
        entry_strategy="top5_selection",
    )
    runtime = _ReviewRuntime()
    adapter = ResearchTaskReviewRunner(
        working_memory=service,
        review_runtime=runtime,
        review_scope_id="review-scope",
        evidence_snapshot_builder=_snapshot_builder,
    )

    outcome = adapter.review_current_result_heads(
        "task-review", result_refs=result_refs
    )

    assert isinstance(outcome, V2ReviewFailed)
    assert outcome.recovery_verification.reason_code == (
        "review_entry_strategy_invalid"
    )
    assert runtime.calls == []


def test_review_rejects_result_refs_that_are_not_current_heads(tmp_path: Path) -> None:
    """显式输入不能替代 Scientific Graph 当前 Result heads。"""

    service, _ = _make_v2_task(tmp_path)
    runtime = _ReviewRuntime()
    adapter = build_v2_task_review_runtime(
        working_memory=service,
        review_runtime=runtime,
        review_scope_id="review-scope",
        evidence_snapshot_builder=_snapshot_builder,
    )

    outcome = adapter.review_current_result_heads(
        "task-review",
        result_refs=("not-a-current-head",),
    )

    assert isinstance(outcome, V2ReviewFailed)
    assert outcome.recovery_verification.status is RecoveryStatus.VERIFIED_INTEGRITY_CONFLICT
    assert runtime.calls == []


def test_review_integrity_gate_does_not_start_evidence_lease_or_catalog_readiness(
    tmp_path: Path,
) -> None:
    """闭合前的 V2 身份屏障不启动 Evidence MCP，也不读取全局 Catalog。"""

    service, _ = _make_v2_task(tmp_path)

    class RecordingLease:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def serve_evidence_snapshot(self, snapshot_id: str):
            self.calls.append(snapshot_id)
            raise AssertionError("integrity conflict 前不应启动 Evidence lease")

    lease = RecordingLease()
    runtime = _ReviewRuntime()
    adapter = ResearchTaskReviewRunner(
        working_memory=service,
        review_runtime=runtime,
        review_scope_id="review-scope",
        evidence_snapshot_builder=_snapshot_builder,
        evidence_lease=lease,
    )

    outcome = adapter.review_current_result_heads(
        "task-review",
        result_refs=("not-a-current-head",),
    )

    assert isinstance(outcome, V2ReviewFailed)
    assert lease.calls == []
    assert runtime.calls == []


def test_closed_review_replay_short_circuits_before_evidence_or_readiness(
    tmp_path: Path,
) -> None:
    """已密封 Review 的恢复只读 ResearchTaskClosedResult，不重启 Evidence 或 MiniChem。"""

    service, result_refs = _make_v2_task(tmp_path)
    facts = service.read_research_task_facts("task-review")
    service.start_module_invocation(
        "task-review",
        facts.research_task_sequence,
        ModuleInvocationRequest(
            research_task_id="task-review",
            module_invocation_id="review-1",
            module_kind="task-review",
            input_scientific_record_ids=result_refs,
            module_input_digest="review-input-digest",
            context_binding=CreateResearchTaskReviewSession(
                research_task_review_session_id="review-session-task-review",
            ),
        ),
    )
    sealing_adapter = build_v2_task_review_adapter(
        working_memory=service,
        review_runtime=_ReviewRuntime(),
        review_scope_id="review-scope",
        evidence_snapshot_builder=_snapshot_builder,
    )
    prepared_result = sealing_adapter.start(
        {
            "research_task_id": "task-review",
            "module_invocation_id": "review-1",
            "input_scientific_record_ids": result_refs,
            "module_input_digest": "review-input-digest",
        }
    )
    assert isinstance(prepared_result, ModuleCompleted)
    facts = service.read_research_task_facts("task-review")
    service.complete_module_invocation(
        "task-review",
        facts.research_task_sequence,
        prepared_result.prepared_output,
    )
    facts = service.read_research_task_facts("task-review")
    service.close_research_task(
        "task-review",
        facts.research_task_sequence,
        {
            "status": "succeeded",
            "review_scientific_record_id": "review-1:review",
            "final_answer_scientific_record_id": "review-1:answer",
        },
    )

    class RecordingLease:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def serve_evidence_snapshot(self, snapshot_id: str):
            self.calls.append(snapshot_id)
            raise AssertionError("closed Review replay 不应启动 Evidence lease")

    lease = RecordingLease()
    runtime = _ReviewRuntime()
    replay_adapter = build_v2_task_review_adapter(
        working_memory=service,
        review_runtime=runtime,
        review_scope_id="review-scope",
        evidence_snapshot_builder=_snapshot_builder,
        evidence_lease=lease,
    )
    controller = ResearchWorkflowController(
        service,
        {"task-review": replay_adapter},
    )

    replay = controller.resume_research_task("task-review")

    assert isinstance(replay, ResearchTaskClosedResult)
    assert lease.calls == []
    assert runtime.calls == []


def test_review_module_invocations_reuse_one_task_review_session(tmp_path: Path) -> None:
    """同一 research_task 的多次 Review invocation 复用首次 Review Session。"""

    service, result_refs = _make_v2_task(tmp_path)
    runtime = _ReviewRuntime()
    adapter = ResearchTaskReviewRunner(
        working_memory=service,
        review_runtime=runtime,
        review_scope_id="review-scope",
        evidence_snapshot_builder=_snapshot_builder,
    )

    first = adapter.review_current_result_heads("task-review", result_refs=result_refs)
    second = adapter.review_current_result_heads("task-review", result_refs=result_refs)

    assert isinstance(first, V2ReviewCompleted)
    assert isinstance(second, V2ReviewCompleted)
    first_payload = json.loads(runtime.calls[0][0]["messages"][-1]["content"])
    second_payload = json.loads(runtime.calls[1][0]["messages"][-1]["content"])
    assert first_payload["research_task_id"] == second_payload["research_task_id"] == "task-review"
    assert runtime.calls[0][0]["metadata"].get("review_thread_id") is None
    assert runtime.calls[1][0]["metadata"]["review_thread_id"] == "review-thread"


def test_review_digest_or_evidence_binding_conflict_is_typed_failure(tmp_path: Path) -> None:
    """Result 的 payload digest 或 Evidence 绑定漂移不能降级为暂停。"""

    service, result_refs = _make_v2_task(tmp_path)
    runtime = _ReviewRuntime()

    original_read = service.read_research_task_scientific_records

    def tampered_read(
        research_task_id: str,
        scientific_record_ids: Sequence[str] | None = None,
    ):
        records = list(original_read(research_task_id, scientific_record_ids))
        if scientific_record_ids and result_refs[0] in scientific_record_ids:
            result = records[0]
            # ScientificRecord 构造器本身拒绝坏摘要；这里模拟只读 adapter
            # 读到被篡改的物化字段，验证 Review runtime 仍会自行重算摘要。
            object.__setattr__(result, "payload_digest", "0" * 64)
            records[0] = result
        return tuple(records)

    service.read_research_task_scientific_records = tampered_read  # type: ignore[method-assign]
    adapter = ResearchTaskReviewRunner(
        working_memory=service,
        review_runtime=runtime,
        review_scope_id="review-scope",
        evidence_snapshot_builder=_snapshot_builder,
    )

    outcome = adapter.review_current_result_heads("task-review", result_refs=result_refs)

    assert isinstance(outcome, V2ReviewFailed)
    assert outcome.recovery_verification.status is RecoveryStatus.VERIFIED_INTEGRITY_CONFLICT
    assert runtime.calls == []


def test_review_provider_interruption_keeps_open_as_typed_suspension(tmp_path: Path) -> None:
    """provider 暂时中断时返回 Suspended，不构造 ReviewRecord。"""

    service, result_refs = _make_v2_task(tmp_path)

    class InterruptedRuntime(_ReviewRuntime):
        def run_task_review_turn(self, request, *, scope_id):
            raise TimeoutError("provider idle timeout")

    adapter = ResearchTaskReviewRunner(
        working_memory=service,
        review_runtime=InterruptedRuntime(),
        review_scope_id="review-scope",
        evidence_snapshot_builder=_snapshot_builder,
    )

    outcome = adapter.review_current_result_heads("task-review", result_refs=result_refs)

    assert isinstance(outcome, V2ReviewSuspended)
    assert outcome.recovery_verification.status is RecoveryStatus.TEMPORARILY_UNAVAILABLE


def test_review_evidence_missing_keeps_open_as_typed_suspension(tmp_path: Path) -> None:
    """Evidence snapshot 尚不可读时保持 Open，而不是伪造 Review 失败。"""

    service, result_refs = _make_v2_task(tmp_path)

    def missing_snapshot(_review_input: Any) -> dict[str, Any]:
        raise EvidenceSnapshotNotFound("snapshot unavailable")

    adapter = ResearchTaskReviewRunner(
        working_memory=service,
        review_runtime=_ReviewRuntime(),
        review_scope_id="review-scope",
        evidence_snapshot_builder=missing_snapshot,
    )

    outcome = adapter.review_current_result_heads("task-review", result_refs=result_refs)

    assert isinstance(outcome, V2ReviewSuspended)
    assert outcome.recovery_verification.status is RecoveryStatus.RECOVERY_EVIDENCE_MISSING


def test_review_fail_is_normal_repair_request_outcome(tmp_path: Path) -> None:
    """DS fail 通过正常 Review outcome 转成 RepairRequest 所需公开字段。"""

    service, result_refs = _make_v2_task(tmp_path)
    runtime = _ReviewRuntime(
        TaskReviewResponse(
            findings=[
                {
                    "hypothesis_execution_input_record_id": "combo-1",
                    "validity": "fail",
                    "repair_advice": "repeat the measurement",
                }
            ],
            can_answer_original_question=False,
        )
    )
    adapter = ResearchTaskReviewRunner(
        working_memory=service,
        review_runtime=runtime,
        review_scope_id="review-scope",
        evidence_snapshot_builder=_snapshot_builder,
    )

    outcome = adapter.review_current_result_heads("task-review", result_refs=result_refs)

    assert isinstance(outcome, V2ReviewCompleted)
    assert outcome.to_public_outcome() == {
        "outcome": "RepairRequired",
        "findings": [
            {
                "hypothesis_execution_input_record_id": "combo-1",
                "validity": "fail",
                "repair_advice": "repeat the measurement",
            }
        ],
        "problem": "Review rejected execution inputs: combo-1",
        "instructions": "repeat the measurement",
    }


@pytest.mark.parametrize("repair_budget", [1, 4])
def test_exhausted_repair_budget_sends_finalization_not_review_request(
    tmp_path: Path,
    repair_budget: int,
) -> None:
    """已用完冻结返修次数时，最后一次 DS 调用直接产生终局结果。"""

    service, (initial_result_ref,) = _make_v2_task(
        tmp_path,
        repair_budget=repair_budget,
        entry_strategy="author_hypothesis_import",
        contract_version="investigation_v2.task.v2",
    )
    repaired_result_ref = initial_result_ref
    for round_number in range(1, repair_budget + 1):
        repaired_result_ref = _append_repaired_result(
            service,
            source_result_ref=repaired_result_ref,
            round_number=round_number,
        )
        if round_number < repair_budget:
            interim_runtime = _ReviewRuntime()
            interim = ResearchTaskReviewRunner(
                working_memory=service,
                review_runtime=interim_runtime,
                review_scope_id="review-scope",
                evidence_snapshot_builder=_snapshot_builder,
            ).review_current_result_heads(
                "task-review",
                result_refs=(repaired_result_ref,),
            )
            assert isinstance(interim, V2ReviewCompleted)
            assert (
                interim_runtime.calls[0][0]["metadata"]["request_kind"]
                == "task_review"
            )
    runtime = _FinalizationRuntime()
    adapter = ResearchTaskReviewRunner(
        working_memory=service,
        review_runtime=runtime,
        review_scope_id="review-scope",
        evidence_snapshot_builder=_snapshot_builder,
    )

    outcome = adapter.review_current_result_heads(
        "task-review",
        result_refs=(repaired_result_ref,),
    )

    assert isinstance(outcome, V2ReviewCompleted)
    request = runtime.calls[0][0]
    payload = json.loads(request["messages"][-1]["content"])
    assert request["metadata"]["request_kind"] == "task_finalization"
    assert payload["request_kind"] == "task_finalization"
    assert outcome.to_public_outcome() == {
        "outcome": "Accepted",
        "findings": [
            {
                "hypothesis_execution_input_record_id": "execution-input-1",
                "summary": "The repaired evidence is sufficient for a bounded conclusion.",
            }
        ],
        "answer": "The available repaired evidence supports the route.",
    }


@pytest.mark.parametrize("repair_budget", [1, 4])
def test_controller_closes_task_from_budget_finalization_without_new_repair(
    tmp_path: Path,
    repair_budget: int,
) -> None:
    """Controller 在返修耗尽后保存最终成稿并正常闭合 Task。"""

    service, (initial_result_ref,) = _make_v2_task(
        tmp_path,
        repair_budget=repair_budget,
        entry_strategy="author_hypothesis_import",
        contract_version="investigation_v2.task.v2",
    )
    repaired_result_ref = initial_result_ref
    for round_number in range(1, repair_budget + 1):
        repaired_result_ref = _append_repaired_result(
            service,
            source_result_ref=repaired_result_ref,
            round_number=round_number,
        )
    runtime = _FinalizationRuntime()
    review_adapter = build_v2_task_review_adapter(
        working_memory=service,
        review_runtime=runtime,
        review_scope_id="review-scope",
        evidence_snapshot_builder=_snapshot_builder,
    )
    controller = ResearchWorkflowController(
        service,
        {"task-review": review_adapter},
    )

    result = controller.resume_research_task("task-review")

    assert isinstance(result, ResearchTaskClosedResult)
    assert result.outcome["status"] == "succeeded"
    assert len(runtime.calls) == 1
    assert runtime.calls[0][0]["metadata"]["request_kind"] == "task_finalization"
    facts = service.read_research_task_facts("task-review")
    assert facts.lifecycle.state == "Closed"
    assert (
        len(facts.scientific_graph.record_refs_by_type["RepairRequestRecord"])
        == repair_budget
    )


@pytest.mark.parametrize("artifact_count", [35, 64])
def test_review_snapshot_accepts_thirty_five_and_sixty_four_artifacts(
    tmp_path: Path,
    artifact_count: int,
) -> None:
    """V2 Review 支持 35 个有效产物和 64 个快照上限。"""

    artifacts = tuple(
        f"artifact-{index:02d}" for index in range(1, artifact_count + 1)
    )
    service, result_refs = _make_v2_task(tmp_path, artifacts=artifacts)
    runtime = _ReviewRuntime()

    def builder(review_input: Any) -> dict[str, Any]:
        return {
            "snapshot_id": "snapshot-35-artifacts",
            "research_task_id": review_input.research_task_id,
            "investigation_context_id": review_input.investigation_context_id,
            "result_pairs": [
                {
                    "hypothesis_execution_input_record_id": "combo-1",
                    "result_version": "v1",
                    "artifacts": [
                        {
                            "artifact_id": f"reader-{index:02d}",
                            "source_artifact_ref": artifact,
                            "size": 1,
                            "sha256": "a" * 64,
                            "media_type": "text/plain",
                        }
                        for index, artifact in enumerate(artifacts, start=1)
                    ],
                }
            ],
        }

    adapter = ResearchTaskReviewRunner(
        working_memory=service,
        review_runtime=runtime,
        review_scope_id="review-scope",
        evidence_snapshot_builder=builder,
    )

    outcome = adapter.review_current_result_heads("task-review", result_refs=result_refs)

    assert isinstance(outcome, V2ReviewCompleted)
    payload = json.loads(outcome.request["messages"][-1]["content"])
    snapshot = payload["evidence_snapshot"]
    assert len(snapshot["result_pairs"][0]["artifacts"]) == artifact_count
    assert all("source_path" not in artifact for artifact in snapshot["result_pairs"][0]["artifacts"])


def test_review_does_not_accept_task_context_drift(tmp_path: Path) -> None:
    """调用方传入的 Task input/policy 只是核对值，不能覆盖冻结事实。"""

    service, result_refs = _make_v2_task(tmp_path)
    adapter = ResearchTaskReviewRunner(
        working_memory=service,
        review_runtime=_ReviewRuntime(),
        review_scope_id="review-scope",
        evidence_snapshot_builder=_snapshot_builder,
    )

    outcome = adapter.review_current_result_heads(
        "task-review",
        result_refs=result_refs,
        task_input={"scientific_question": "tampered"},
    )

    assert isinstance(outcome, V2ReviewFailed)
    assert outcome.recovery_verification.status is RecoveryStatus.VERIFIED_INTEGRITY_CONFLICT


def test_review_module_adapter_returns_atomic_v2_review_output(tmp_path: Path) -> None:
    """公开 adapter 可直接接 Controller，并形成 ReviewRecord/FinalAnswerRecord。"""

    service, result_refs = _make_v2_task(tmp_path)
    adapter = build_v2_task_review_adapter(
        working_memory=service,
        review_runtime=_ReviewRuntime(),
        review_scope_id="review-scope",
        evidence_snapshot_builder=_snapshot_builder,
    )
    outcome = adapter.start(
        {
            "research_task_id": "task-review",
            "module_invocation_id": "review-1",
            "input_scientific_record_ids": result_refs,
            "module_input_digest": "review-input-digest",
        }
    )

    assert isinstance(adapter, V2TaskReviewModuleAdapter)
    assert isinstance(outcome, ModuleCompleted)
    assert [record.scientific_record_type for record in outcome.prepared_output.output_scientific_records] == [
        "ReviewRecord",
        "FinalAnswerRecord",
    ]
    review, answer = outcome.prepared_output.output_scientific_records
    assert review.payload["reviewed_result_refs"] == ["result-1"]
    assert answer.source_scientific_record_ids == (review.scientific_record_id,)


def test_review_module_adapter_emits_repair_request_for_ds_fail(tmp_path: Path) -> None:
    """DS fail 仍形成正常 Review bundle，而不是 ModuleFailed。"""

    service, result_refs = _make_v2_task(tmp_path)
    adapter = V2TaskReviewModuleAdapter(
        ResearchTaskReviewRunner(
            working_memory=service,
            review_runtime=_ReviewRuntime(
                TaskReviewResponse(
                    findings=[
                        {
                            "hypothesis_execution_input_record_id": "combo-1",
                            "validity": "fail",
                            "repair_advice": "repeat the measurement",
                        }
                    ],
                    can_answer_original_question=False,
                )
            ),
            review_scope_id="review-scope",
            evidence_snapshot_builder=_snapshot_builder,
        )
    )
    outcome = adapter.start(
        {
            "research_task_id": "task-review",
            "module_invocation_id": "review-1",
            "input_scientific_record_ids": result_refs,
            "module_input_digest": "review-input-digest",
        }
    )

    assert isinstance(outcome, ModuleCompleted)
    assert [record.scientific_record_type for record in outcome.prepared_output.output_scientific_records] == [
        "ReviewRecord",
        "RepairRequestRecord",
    ]
    repair = outcome.prepared_output.output_scientific_records[1]
    assert repair.payload["result_ref"] == "result-1"
    assert repair.source_scientific_record_ids == ("review-1:review", "result-1")
