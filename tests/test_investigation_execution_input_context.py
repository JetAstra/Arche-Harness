"""已确认执行输入接入 Context 与 Investigation prompt 的行为测试。"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pytest

from chemistry_multiagent.controllers.investigation_v2_native_adapters import (
    InvestigationSessionHarnessOwner,
    InvestigationSessionOwner,
)
from chemistry_multiagent.investigation.context_backend import (
    InvestigationContextRunner,
)
from chemistry_multiagent.investigation.context_store import (
    InvestigationContextIdentity,
    InvestigationContextStore,
    InvestigationSessionIdentity,
)
from chemistry_multiagent.investigation.harness_recovery_store import SessionReference
from chemistry_multiagent.investigation.harness_runtime import (
    BackendCompleted,
    Completed,
    InvestigationHarnessRequest,
)
from chemistry_multiagent.investigation.prompts import (
    build_initial_investigation_request,
)
from chemistry_multiagent.investigation.recovery_contract import Consistent
from chemistry_multiagent.investigation.research_material_store import (
    MaterialSizePolicy,
    ResearchMaterialStore,
)
from chemistry_multiagent.memory import (
    ResearchTaskWorkingMemory,
    PreparedModuleInvocationOutput,
    ScientificRecord,
)
from chemistry_multiagent.memory.v2_identities import (
    CreateInvestigationContext,
    ReuseInvestigationContext,
)


def _workspace_relpath(
    investigation_session_id: str,
    research_task_id: str,
    context_id: str,
    hypothesis_id: str,
) -> str:
    """返回当前生产 Session 的实际 workspace 相对路径。"""

    return (
        f"internal/state/investigation_runtime/sessions/{investigation_session_id}/codex_sessions/"
        f"investigations/{research_task_id}/{context_id}/{hypothesis_id}/workspace"
    )


@dataclass(frozen=True)
class _DuckRecord:
    """等待 Working Memory contract 分支时使用的最小公开科学记录。"""

    scientific_record_id: str
    research_task_id: str
    scientific_record_type: str
    scientific_record_schema_version: str | int
    source_scientific_record_ids: tuple[str, ...]
    payload: dict[str, Any]


class _DuckWorkingMemory(ResearchTaskWorkingMemory):
    """只替换读取 seam，不注册或复制新的 ledger record schema。"""

    def __init__(self, database_path: Path, records: dict[str, _DuckRecord]) -> None:
        super().__init__(database_path)
        self.records = records

    def read_research_task_scientific_records(
        self, research_task_id: str, record_ids: tuple[str, ...]
    ) -> tuple[_DuckRecord, ...]:
        values = tuple(
            self.records[scientific_record_id]
            for scientific_record_id in record_ids
            if scientific_record_id in self.records
        )
        return tuple(value for value in values if value.research_task_id == research_task_id)


def _duck(
    scientific_record_id: str,
    scientific_record_type: str,
    *,
    refs: tuple[str, ...] = (),
    payload: dict[str, Any] | None = None,
    schema_version: str | int = 1,
) -> _DuckRecord:
    return _DuckRecord(
        scientific_record_id=scientific_record_id,
        research_task_id="inquiry-execution",
        scientific_record_type=scientific_record_type,
        scientific_record_schema_version=schema_version,
        source_scientific_record_ids=refs,
        payload={} if payload is None else payload,
    )


def _execution_records(execution_count: int = 2) -> dict[str, _DuckRecord]:
    if not 1 <= execution_count <= 5:
        raise ValueError("execution_count 必须在 1 到 5 之间")
    selected_count = max(3, execution_count)
    batch = _duck("batch-1", "HypothesisBatchRecord")
    decision = _duck(
        "decision-1",
        "HypothesisExecutionDecisionRecord",
        refs=(batch.scientific_record_id,),
        payload={
            "decisions": ["execute"] * execution_count
            + ["cancel"] * (selected_count - execution_count)
        },
    )
    selected = tuple(
        _duck(
            f"selected-{rank}",
            "SelectedHypothesisRecord",
            refs=(batch.scientific_record_id,),
            payload={
                "hypothesis_ref": f"h-{rank}",
                "rank": rank,
                "hypothesis": {
                    "hypothesis_id": f"h-{rank}",
                    "claim": f"original-{rank}",
                },
            },
        )
        for rank in range(1, selected_count + 1)
    )
    materials = {
        "material-record-1": _duck(
            "material-record-1",
            "ResearchMaterialRecord",
            payload={
                "material_id": "material-1",
                "content_ref": "sha256:" + "a" * 64,
                "sha256": "a" * 64,
                "size_bytes": 12,
                "filename": "first.xyz",
                "media_type": "chemical/x-xyz",
                "description": "first session material",
            },
        )
    }
    execution = tuple(
        _duck(
            f"execution-{rank}",
            "HypothesisExecutionInputRecord",
            refs=(
                ("material-record-1", decision.scientific_record_id, selected[rank - 1].scientific_record_id)
                if rank == 1
                else (decision.scientific_record_id, selected[rank - 1].scientific_record_id)
            ),
            payload={
                "source_selected_hypothesis_ref": selected[rank - 1].scientific_record_id,
                "execution_rank": rank,
                "hypothesis": {
                    "hypothesis_id": f"h-{rank}",
                    "claim": f"edited-{rank}",
                },
                "guidance": (
                    "only first session guidance" if rank == 1 else None
                ),
                "material_refs": ["material-1"] if rank == 1 else [],
            },
        )
        for rank in range(1, execution_count + 1)
    )
    records = {
        item.scientific_record_id: item
        for item in (batch, decision, *selected, *execution, *materials.values())
    }
    return records


def _canonical_confirmation_v2_records() -> dict[str, _DuckRecord]:
    """返回真实 Confirmation v2 写出的单条、无材料执行输入。"""

    records = _execution_records(execution_count=1)
    execution = records["execution-1"]
    records["execution-1"] = _duck(
        execution.scientific_record_id,
        execution.scientific_record_type,
        refs=("decision-1", "selected-1"),
        schema_version="hypothesis-confirmation.v2",
        payload={
            "source_selected_hypothesis_record_id": "selected-1",
            "execution_rank": 1,
            "hypothesis": {
                "strategy_name": "比较两条对映路径的关键成键过渡态",
                "reasoning": "分别优化并验证两条路径。",
            },
            "guidance": "使用已给定的 TS Pipeline 反应物和产物。",
            "material_upload_refs": [],
        },
    )
    records["decision-1"] = _duck(
        "decision-1",
        "HypothesisExecutionDecisionRecord",
        refs=("batch-1",),
        schema_version="hypothesis-confirmation.v2",
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
    return records


class _Resolver:
    def __init__(self) -> None:
        self.planned: list[str] = []
        self.ensured: list[str] = []

    def plan_session(
        self,
        context: InvestigationContextIdentity,
        owner: _DuckRecord,
    ) -> InvestigationSessionIdentity:
        self.planned.append(owner.scientific_record_id)
        rank = owner.payload.get("execution_rank", owner.payload.get("rank"))
        return InvestigationSessionIdentity(
            investigation_session_id=f"session-{owner.scientific_record_id}",
            investigation_context_id=context.investigation_context_id,
            research_task_id=context.research_task_id,
            hypothesis_execution_input_record_id=owner.scientific_record_id,
            workspace_identity=_workspace_relpath(
                f"session-{owner.scientific_record_id}",
                context.research_task_id,
                context.investigation_context_id,
                owner.scientific_record_id,
            ),
            minichem_scope_id=f"scope-{owner.scientific_record_id}",
            scope_contract_digest=str(rank) * 64,
        )

    def ensure_session(self, session: InvestigationSessionIdentity) -> None:
        self.ensured.append(session.hypothesis_execution_input_record_id)


class _ChildHarness:
    def __init__(self) -> None:
        self.requests: list[InvestigationHarnessRequest] = []

    def start(self, request: InvestigationHarnessRequest) -> Completed:
        self.requests.append(request)
        owner = request.input_scientific_record_ids[0]
        evidence = ScientificRecord.from_payload(
            scientific_record_id=f"evidence-{owner}",
            research_task_id=request.research_task_id,
            scientific_record_type="VerifiedEvidenceRecord",
            scientific_record_schema_version=1,
            producer_module_invocation_id=request.module_invocation_id,
            source_scientific_record_ids=request.input_scientific_record_ids,
            payload={
                "evidence_items": [{"owner": owner}],
                "artifact_refs": [],
                "source_manifest_ref": f"manifest-{owner}",
                "summary": "verified",
                "verification": {"status": "verified"},
            },
        )
        result = ScientificRecord.from_payload(
            scientific_record_id=f"result-{owner}",
            research_task_id=request.research_task_id,
            scientific_record_type="InvestigationResultRecord",
            scientific_record_schema_version=1,
            producer_module_invocation_id=request.module_invocation_id,
            source_scientific_record_ids=(evidence.scientific_record_id, *request.input_scientific_record_ids),
            payload={
                "conclusion": "completed",
                "self_judgment": "supported",
                "observations": [owner],
                "limitations": [],
                "evidence_ref": evidence.scientific_record_id,
            },
        )
        return Completed(
            f"harness-{owner}",
            PreparedModuleInvocationOutput.from_records(
                module_invocation_id=request.module_invocation_id,
                module_input_digest=request.module_input_digest,
                output_scientific_records=(evidence, result),
            ),
            recovery_verification=Consistent(reason_code="child-completed"),
        )


class _Factory:
    def __init__(self) -> None:
        self.harnesses: dict[str, _ChildHarness] = {}

    def open_session(self, session: InvestigationSessionIdentity) -> _ChildHarness:
        return self.harnesses.setdefault(session.hypothesis_execution_input_record_id, _ChildHarness())


class _MaterialDescriptorResolver:
    """模拟 Session 物化后的 descriptor，不读取 Working Memory 外部路径或正文。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def resolve_material_descriptor(
        self, material: _DuckRecord, session: SessionReference
    ) -> dict[str, Any]:
        self.calls.append((material.scientific_record_id, session.investigation_session_id))
        return {
            "workspace_relpath": (
                f"inputs/materials/{session.hypothesis_execution_input_record_id}/{material.payload['filename']}"
            ),
            "filename": material.payload["filename"],
            "media_type": material.payload["media_type"],
            "size_bytes": material.payload["size_bytes"],
            "sha256": material.payload["sha256"],
            "description": material.payload["description"],
        }


def test_execution_inputs_create_only_confirmed_sessions_and_bind_decision(
    tmp_path: Path,
) -> None:
    records = _execution_records()
    memory = _DuckWorkingMemory(tmp_path / "memory.sqlite3", records)
    resolver = _Resolver()
    factory = _Factory()
    backend = InvestigationContextRunner(
        context_store=InvestigationContextStore(tmp_path / "context.sqlite3"),
        working_memory=memory,
        session_owner=resolver,
        session_harness_owner=factory,
    )
    request = InvestigationHarnessRequest(
        research_task_id="inquiry-execution",
        module_invocation_id="investigation-initial",
        module_kind="hypothesis-investigation",
        input_scientific_record_ids=("execution-2", "execution-1"),
        module_input_digest="b" * 64,
        context_binding=CreateInvestigationContext(
            investigation_context_id="context-execution",
            hypothesis_batch_record_id="batch-1",
        ).to_dict(),
    )

    outcome = backend.start(request, request.harness_invocation_id)

    assert isinstance(outcome, BackendCompleted)
    assert resolver.planned == ["execution-1", "execution-2"]
    assert resolver.ensured == ["execution-1", "execution-2"]
    assert set(factory.harnesses) == {"execution-1", "execution-2"}
    assert all(
        harness.requests[0].input_scientific_record_ids == (owner,)
        for owner, harness in factory.harnesses.items()
    )
    context = InvestigationContextIdentity(
        "context-execution", "inquiry-execution", "batch-1",
        hypothesis_execution_decision_record_id="decision-1",
    )
    snapshot = backend._context_store.read_context(context)
    assert {session.hypothesis_execution_input_record_id for session in snapshot.sessions} == {
        "execution-1",
        "execution-2",
    }


def test_confirmation_v2_execution_input_enters_context_without_field_drift(
    tmp_path: Path,
) -> None:
    """当前 Confirmation 产物必须直接进入 Context，不能按旧字段误判冲突。"""

    records = _canonical_confirmation_v2_records()
    resolver = _Resolver()
    factory = _Factory()
    backend = InvestigationContextRunner(
        context_store=InvestigationContextStore(tmp_path / "context.sqlite3"),
        working_memory=_DuckWorkingMemory(tmp_path / "memory.sqlite3", records),
        session_owner=resolver,
        session_harness_owner=factory,
    )
    request = InvestigationHarnessRequest(
        research_task_id="inquiry-execution",
        module_invocation_id="investigation-confirmation-v2",
        module_kind="hypothesis-investigation",
        input_scientific_record_ids=("execution-1",),
        module_input_digest="f" * 64,
        context_binding=CreateInvestigationContext(
            investigation_context_id="context-confirmation-v2",
            hypothesis_batch_record_id="batch-1",
        ).to_dict(),
    )

    outcome = backend.start(request, request.harness_invocation_id)

    assert isinstance(outcome, BackendCompleted)
    assert resolver.planned == ["execution-1"]
    assert resolver.ensured == ["execution-1"]


def test_native_provider_builder_reads_confirmation_v2_execution_input(
    tmp_path: Path,
) -> None:
    """native Session prompt 必须读取当前字段，且不能要求旧字段别名。"""

    records = _canonical_confirmation_v2_records()
    memory = _DuckWorkingMemory(tmp_path / "memory.sqlite3", records)
    factory = InvestigationSessionHarnessOwner(
        task_root=tmp_path / "bundle",
        working_memory=memory,
        task_input={"scientific_question": "原始科学问题"},
        policy={},
        dependencies={},
    )
    request = InvestigationHarnessRequest(
        research_task_id="inquiry-execution",
        module_invocation_id="investigation-confirmation-v2",
        module_kind="hypothesis-investigation",
        input_scientific_record_ids=("execution-1",),
        module_input_digest="a" * 64,
        context_binding=CreateInvestigationContext(
            investigation_context_id="context-confirmation-v2",
            hypothesis_batch_record_id="batch-1",
        ).to_dict(),
    )

    built = factory._build_provider_request(
        request,
        session=SessionReference(
            research_task_id="inquiry-execution",
            investigation_context_id="context-confirmation-v2",
            hypothesis_execution_input_record_id="execution-1",
            investigation_session_id="session-execution-1",
            workspace_identity="workspace-execution-1",
            minichem_scope_id="scope-execution-1",
            scope_contract_digest="2" * 64,
            capability_generation_digest="1" * 64,
        ),
    )

    payload = json.loads(built["messages"][1]["content"])
    assert payload["hypothesis"] == records["execution-1"].payload["hypothesis"]
    assert payload["guidance"] == records["execution-1"].payload["guidance"]
    assert "materials" not in payload


@pytest.mark.parametrize("execution_count", range(1, 6))
def test_one_to_five_execution_inputs_create_matching_sessions_and_results(
    tmp_path: Path,
    execution_count: int,
) -> None:
    """确认后的 1–5 条执行输入必须各自拥有一个 Session 和一条 Result。"""

    records = _execution_records(execution_count)
    memory = _DuckWorkingMemory(tmp_path / "memory.sqlite3", records)
    resolver = _Resolver()
    factory = _Factory()
    backend = InvestigationContextRunner(
        context_store=InvestigationContextStore(tmp_path / "context.sqlite3"),
        working_memory=memory,
        session_owner=resolver,
        session_harness_owner=factory,
    )
    input_refs = tuple(
        f"execution-{rank}" for rank in range(execution_count, 0, -1)
    )
    request = InvestigationHarnessRequest(
        research_task_id="inquiry-execution",
        module_invocation_id=f"investigation-count-{execution_count}",
        module_kind="hypothesis-investigation",
        input_scientific_record_ids=input_refs,
        module_input_digest=str(execution_count) * 64,
        context_binding=CreateInvestigationContext(
            investigation_context_id=f"context-count-{execution_count}",
            hypothesis_batch_record_id="batch-1",
        ).to_dict(),
    )

    outcome = backend.start(request, request.harness_invocation_id)

    assert isinstance(outcome, BackendCompleted)
    expected_owners = [
        f"execution-{rank}" for rank in range(1, execution_count + 1)
    ]
    assert resolver.planned == expected_owners
    assert resolver.ensured == expected_owners
    assert sorted(factory.harnesses) == expected_owners
    result_records = [
        record
        for record in outcome.prepared_output.output_scientific_records
        if record.scientific_record_type == "InvestigationResultRecord"
    ]
    assert len(result_records) == execution_count


def test_one_execution_input_is_a_valid_context_and_uses_execution_owner(
    tmp_path: Path,
) -> None:
    records = _execution_records()
    memory = _DuckWorkingMemory(tmp_path / "memory.sqlite3", records)
    resolver = _Resolver()
    factory = _Factory()
    backend = InvestigationContextRunner(
        context_store=InvestigationContextStore(tmp_path / "context.sqlite3"),
        working_memory=memory,
        session_owner=resolver,
        session_harness_owner=factory,
    )
    request = InvestigationHarnessRequest(
        research_task_id="inquiry-execution",
        module_invocation_id="investigation-single",
        module_kind="hypothesis-investigation",
        input_scientific_record_ids=("execution-1",),
        module_input_digest="d" * 64,
        context_binding=CreateInvestigationContext(
            investigation_context_id="context-single",
            hypothesis_batch_record_id="batch-1",
        ).to_dict(),
    )

    outcome = backend.start(request, request.harness_invocation_id)

    assert isinstance(outcome, BackendCompleted)
    assert resolver.planned == ["execution-1"]
    assert set(factory.harnesses) == {"execution-1"}

    native_resolver = InvestigationSessionOwner(
        task_root=tmp_path / "bundle",
        policy={},
        dependencies={},
    )
    context = InvestigationContextIdentity(
        "context-single",
        "inquiry-execution",
        "batch-1",
        hypothesis_execution_decision_record_id="decision-1",
    )
    session = native_resolver.plan_session(context, records["execution-1"])
    assert session.hypothesis_execution_input_record_id == "execution-1"
    assert session.workspace_identity == _workspace_relpath(
        session.investigation_session_id,
        context.research_task_id,
        context.investigation_context_id,
        "execution-1",
    )


def test_execution_input_repair_reuses_execution_owner_and_not_selected_candidate(
    tmp_path: Path,
) -> None:
    records = _execution_records()
    result = _duck(
        "result-1",
        "InvestigationResultRecord",
        refs=("execution-1",),
        payload={},
    )
    repair = _duck(
        "repair-1",
        "RepairRequestRecord",
        refs=(result.scientific_record_id,),
        payload={"result_ref": result.scientific_record_id},
    )
    records.update({result.scientific_record_id: result, repair.scientific_record_id: repair})
    memory = _DuckWorkingMemory(tmp_path / "memory.sqlite3", records)
    resolver = _Resolver()
    factory = _Factory()
    backend = InvestigationContextRunner(
        context_store=InvestigationContextStore(tmp_path / "context.sqlite3"),
        working_memory=memory,
        session_owner=resolver,
        session_harness_owner=factory,
    )
    initial = InvestigationHarnessRequest(
        research_task_id="inquiry-execution",
        module_invocation_id="investigation-initial",
        module_kind="hypothesis-investigation",
        input_scientific_record_ids=("execution-1", "execution-2"),
        module_input_digest="b" * 64,
        context_binding=CreateInvestigationContext(
            investigation_context_id="context-execution",
            hypothesis_batch_record_id="batch-1",
        ).to_dict(),
    )
    assert isinstance(
        backend.start(initial, initial.harness_invocation_id), BackendCompleted
    )

    repair_request = InvestigationHarnessRequest(
        research_task_id="inquiry-execution",
        module_invocation_id="investigation-repair",
        module_kind="hypothesis-investigation",
        input_scientific_record_ids=(repair.scientific_record_id,),
        module_input_digest="c" * 64,
        context_binding=ReuseInvestigationContext(
            investigation_context_id="context-execution",
            hypothesis_batch_record_id="batch-1",
            source_module_invocation_id="investigation-initial",
        ).to_dict(),
    )
    repaired = backend.start(repair_request, repair_request.harness_invocation_id)

    assert isinstance(repaired, BackendCompleted)
    assert resolver.planned == ["execution-1", "execution-2"]
    assert factory.harnesses["execution-1"].requests[-1].input_scientific_record_ids == (
        "execution-1",
        "repair-1",
        "result-1",
    )


def test_generic_execution_prompt_keeps_system_prompt_and_isolates_material_fields() -> None:
    baseline = build_initial_investigation_request(
        "task-1",
        "batch-1",
        "combination-1",
        "原始问题",
        {"hypothesis_id": "h-1", "claim": "original"},
        confirmed_execution_input=True,
    )
    request = build_initial_investigation_request(
        "task-1",
        "batch-1",
        "combination-1",
        "原始问题",
        {"hypothesis_id": "h-1", "claim": "edited"},
        guidance="只对当前假设补充的 guidance。",
        materials=[
            {
                "workspace_relpath": "inputs/materials/first.xyz",
                "filename": "first.xyz",
                "media_type": "chemical/x-xyz",
                "size_bytes": 12,
                "sha256": "a" * 64,
                "description": "当前 session 的材料",
            }
        ],
        confirmed_execution_input=True,
    )

    assert request["messages"][0] == baseline["messages"][0]
    payload = json.loads(request["messages"][1]["content"])
    assert payload["hypothesis"]["claim"] == "edited"
    assert payload["guidance"] == "只对当前假设补充的 guidance。"
    assert payload["materials"] == [
        {
            "workspace_relpath": "inputs/materials/first.xyz",
            "filename": "first.xyz",
            "media_type": "chemical/x-xyz",
            "size_bytes": 12,
            "sha256": "a" * 64,
            "description": "当前 session 的材料",
        }
    ]
    visible = json.dumps(request["messages"], ensure_ascii=False)
    assert "case1" not in visible.lower()
    assert "case3" not in visible.lower()
    assert "decision-1" not in visible
    assert "cancel" not in visible.lower()
    assert "content_ref" not in visible
    assert "TS Pipeline" not in visible
    assert "tspipeline" not in visible.lower()


def test_generic_prompt_rejects_store_fields_and_unsafe_material_paths() -> None:
    common = (
        "task-1",
        "batch-1",
        "combination-1",
        "原始问题",
        {"hypothesis_id": "h-1", "claim": "edited"},
    )
    with pytest.raises(ValueError):
        build_initial_investigation_request(
            *common,
            confirmed_execution_input=True,
            materials=[
                {
                    "workspace_relpath": "inputs/materials/first.xyz",
                    "filename": "first.xyz",
                    "media_type": "chemical/x-xyz",
                    "size_bytes": 12,
                    "sha256": "a" * 64,
                    "content_ref": "sha256:" + "a" * 64,
                }
            ],
        )
    with pytest.raises(ValueError):
        build_initial_investigation_request(
            *common,
            confirmed_execution_input=True,
            materials=[
                {
                    "workspace_relpath": "/srv/materials/first.xyz",
                    "filename": "first.xyz",
                    "media_type": "chemical/x-xyz",
                    "size_bytes": 12,
                    "sha256": "a" * 64,
                }
            ],
        )


def test_native_provider_builder_prefers_execution_input_over_cancelled_selected(
    tmp_path: Path,
) -> None:
    records = _execution_records()
    memory = _DuckWorkingMemory(tmp_path / "memory.sqlite3", records)
    material_resolver = _MaterialDescriptorResolver()
    factory = InvestigationSessionHarnessOwner(
        task_root=tmp_path / "bundle",
        working_memory=memory,
        task_input={"scientific_question": "原始问题"},
        policy={},
        dependencies={"material_descriptor_resolver": material_resolver},
    )
    session = InvestigationSessionIdentity(
        investigation_session_id="session-execution-1",
        investigation_context_id="context-execution",
        research_task_id="inquiry-execution",
        hypothesis_execution_input_record_id="execution-1",
        workspace_identity=_workspace_relpath(
            "session-execution-1",
            "inquiry-execution",
            "context-execution",
            "execution-1",
        ),
        minichem_scope_id="scope-execution-1",
        scope_contract_digest="1" * 64,
    )
    request = InvestigationHarnessRequest(
        research_task_id="inquiry-execution",
        module_invocation_id="investigation-initial",
        module_kind="hypothesis-investigation",
        input_scientific_record_ids=("execution-1",),
        module_input_digest="b" * 64,
        context_binding=CreateInvestigationContext(
            investigation_context_id="context-execution",
            hypothesis_batch_record_id="batch-1",
        ).to_dict(),
    )

    built = factory._build_provider_request(
        request,
        session=SessionReference(
            research_task_id="inquiry-execution",
            investigation_context_id="context-execution",
            hypothesis_execution_input_record_id="execution-1",
            investigation_session_id="session-execution-1",
            workspace_identity="workspace-execution-1",
            minichem_scope_id="scope-execution-1",
            scope_contract_digest="2" * 64,
            capability_generation_digest="1" * 64,
        ),
    )

    payload = json.loads(built["messages"][1]["content"])
    assert payload["hypothesis"]["claim"] == "edited-1"
    assert payload["guidance"] == "only first session guidance"
    assert payload["materials"][0]["filename"] == "first.xyz"
    assert "original-3" not in json.dumps(built, ensure_ascii=False)
    visible = json.dumps(built, ensure_ascii=False)
    assert "case1" not in visible.lower()
    assert "case3" not in visible.lower()
    assert "TS Pipeline" not in visible
    assert "tspipeline" not in visible.lower()
    assert "decision-1" not in visible
    assert "content_ref" not in visible
    assert material_resolver.calls == [("material-record-1", "session-execution-1")]


def test_native_session_materializes_confirmed_store_bytes_before_prompt(
    tmp_path: Path,
) -> None:
    """确认材料先进入 Session workspace，prompt 只得到核对后的相对路径。"""

    task_root = tmp_path / "bundle"
    task_root.mkdir()
    store = ResearchMaterialStore(
        task_root,
        MaterialSizePolicy(
            max_file_size_bytes=1024,
            max_material_count=5,
            max_task_size_bytes=4096,
        ),
    )
    descriptor = store.ingest_material_bytes(
        "matrix-elements.json",
        "application/json",
        b'{"atoms":[1,2,3,4]}',
        description="四个结构共用的矩阵元素文件",
    )
    records = _execution_records()
    records["material-record-1"] = _duck(
        "material-record-1",
        "ResearchMaterialRecord",
        payload={"material_id": "material-1", **descriptor},
    )
    memory = _DuckWorkingMemory(tmp_path / "memory.sqlite3", records)
    factory = InvestigationSessionHarnessOwner(
        task_root=task_root,
        working_memory=memory,
        task_input={"scientific_question": "原始问题"},
        policy={},
        dependencies={"research_material_store": store},
    )
    session = InvestigationSessionIdentity(
        investigation_session_id="session-execution-1",
        investigation_context_id="context-execution",
        research_task_id="inquiry-execution",
        hypothesis_execution_input_record_id="execution-1",
        workspace_identity=_workspace_relpath(
            "session-execution-1",
            "inquiry-execution",
            "context-execution",
            "execution-1",
        ),
        minichem_scope_id="scope-execution-1",
        scope_contract_digest="1" * 64,
    )

    factory.prepare_session_materials(session)
    built = factory._build_provider_request(
        InvestigationHarnessRequest(
            research_task_id="inquiry-execution",
            module_invocation_id="investigation-initial",
            module_kind="hypothesis-investigation",
            input_scientific_record_ids=("execution-1",),
            module_input_digest="b" * 64,
            context_binding=CreateInvestigationContext(
                investigation_context_id="context-execution",
                hypothesis_batch_record_id="batch-1",
            ).to_dict(),
        ),
        session=SessionReference(
            research_task_id="inquiry-execution",
            investigation_context_id="context-execution",
            hypothesis_execution_input_record_id="execution-1",
            investigation_session_id="session-execution-1",
            workspace_identity="workspace-execution-1",
            minichem_scope_id="scope-execution-1",
            scope_contract_digest="2" * 64,
            capability_generation_digest="1" * 64,
        ),
    )

    payload = json.loads(built["messages"][1]["content"])
    relative_path = payload["materials"][0]["workspace_relpath"]
    assert relative_path.startswith("inputs/materials/")
    materialized = [
        path
        for path in task_root.rglob("matrix-elements.json")
        if ".research_materials" not in path.parts
    ]
    assert len(materialized) == 1
    assert materialized[0].read_bytes() == b'{"atoms":[1,2,3,4]}'
    assert str(task_root) not in json.dumps(payload, ensure_ascii=False)


def test_distinct_session_materials_are_isolated_in_workspaces_and_prompts(
    tmp_path: Path,
) -> None:
    """不同执行假设的材料不能进入彼此的 workspace 或模型输入。"""

    task_root = tmp_path / "bundle"
    task_root.mkdir()
    store = ResearchMaterialStore(
        task_root,
        MaterialSizePolicy(
            max_file_size_bytes=1024,
            max_material_count=5,
            max_task_size_bytes=4096,
        ),
    )
    first_descriptor = store.ingest_material_bytes(
        "first.xyz",
        "chemical/x-xyz",
        b"first geometry",
        description="first session material",
    )
    second_descriptor = store.ingest_material_bytes(
        "second.xyz",
        "chemical/x-xyz",
        b"second geometry",
        description="second session material",
    )
    records = _execution_records()
    records["material-record-1"] = _duck(
        "material-record-1",
        "ResearchMaterialRecord",
        payload={"material_id": "material-1", **first_descriptor},
    )
    records["material-record-2"] = _duck(
        "material-record-2",
        "ResearchMaterialRecord",
        payload={"material_id": "material-2", **second_descriptor},
    )
    second_input = records["execution-2"]
    records["execution-2"] = _duck(
        "execution-2",
        "HypothesisExecutionInputRecord",
        refs=("material-record-2", *second_input.source_scientific_record_ids),
        payload={**second_input.payload, "material_refs": ["material-2"]},
    )
    memory = _DuckWorkingMemory(tmp_path / "memory.sqlite3", records)
    factory = InvestigationSessionHarnessOwner(
        task_root=task_root,
        working_memory=memory,
        task_input={"scientific_question": "原始问题"},
        policy={},
        dependencies={"research_material_store": store},
    )

    prompts: dict[str, dict[str, Any]] = {}
    for rank in (1, 2):
        owner = f"execution-{rank}"
        session = InvestigationSessionIdentity(
            investigation_session_id=f"session-{owner}",
                investigation_context_id="context-execution",
                research_task_id="inquiry-execution",
                hypothesis_execution_input_record_id=owner,
                workspace_identity=_workspace_relpath(
                    f"session-{owner}",
                    "inquiry-execution",
                    "context-execution",
                    owner,
                ),
            minichem_scope_id=f"scope-{owner}",
            scope_contract_digest=str(rank) * 64,
        )
        factory.prepare_session_materials(session)
        built = factory._build_provider_request(
            InvestigationHarnessRequest(
                research_task_id="inquiry-execution",
                module_invocation_id="investigation-initial",
                module_kind="hypothesis-investigation",
                input_scientific_record_ids=(owner,),
                module_input_digest=str(rank) * 64,
                context_binding=CreateInvestigationContext(
                    investigation_context_id="context-execution",
                    hypothesis_batch_record_id="batch-1",
                ).to_dict(),
            ),
            session=SessionReference(
                research_task_id="inquiry-execution",
                investigation_context_id="context-execution",
                hypothesis_execution_input_record_id=owner,
                investigation_session_id=f"session-{owner}",
                workspace_identity=f"workspace-{owner}",
                minichem_scope_id=f"scope-{owner}",
                scope_contract_digest=f"{rank + 5:064x}",
                capability_generation_digest=str(rank) * 64,
            ),
        )
        prompts[owner] = json.loads(built["messages"][1]["content"])

    assert [item["filename"] for item in prompts["execution-1"]["materials"]] == [
        "first.xyz"
    ]
    assert [item["filename"] for item in prompts["execution-2"]["materials"]] == [
        "second.xyz"
    ]
    assert "second.xyz" not in json.dumps(prompts["execution-1"], ensure_ascii=False)
    assert "first.xyz" not in json.dumps(prompts["execution-2"], ensure_ascii=False)
    first_paths = [
        path
        for path in task_root.rglob("first.xyz")
        if ".research_materials" not in path.parts
    ]
    second_paths = [
        path
        for path in task_root.rglob("second.xyz")
        if ".research_materials" not in path.parts
    ]
    assert len(first_paths) == len(second_paths) == 1
    assert "session-execution-1" in first_paths[0].parts
    assert "session-execution-2" not in first_paths[0].parts
    assert "session-execution-2" in second_paths[0].parts
    assert "session-execution-1" not in second_paths[0].parts
    assert first_paths[0].read_bytes() == b"first geometry"
    assert second_paths[0].read_bytes() == b"second geometry"


def test_session_owner_prepares_materials_before_catalog_and_scope(
    tmp_path: Path,
) -> None:
    """Session 身份保存后必须先物化材料，随后才允许冻结目录和创建 scope。"""

    events: list[str] = []

    class _ScopeManager:
        def ensure_session(self, _session: InvestigationSessionIdentity) -> None:
            events.append("scope")

    resolver = InvestigationSessionOwner(
        task_root=tmp_path / "bundle",
        policy={},
        dependencies={
            "prepare_investigation_session_materials": lambda _session: events.append(
                "materials"
            ),
            "prepare_investigation_session": lambda _session: events.append("catalog"),
            "native_scope_manager": _ScopeManager(),
        },
    )
    resolver.ensure_session(
        InvestigationSessionIdentity(
            investigation_session_id="session-execution-1",
            investigation_context_id="context-execution",
            research_task_id="inquiry-execution",
            hypothesis_execution_input_record_id="execution-1",
            workspace_identity=_workspace_relpath(
                "session-execution-1",
                "inquiry-execution",
                "context-execution",
                "execution-1",
            ),
            minichem_scope_id="scope-execution-1",
            scope_contract_digest="1" * 64,
        )
    )

    assert events == ["materials", "catalog", "scope"]


def test_native_provider_builder_keeps_legacy_selected_prompt_fallback(
    tmp_path: Path,
) -> None:
    records = _execution_records()
    memory = _DuckWorkingMemory(tmp_path / "memory.sqlite3", records)
    factory = InvestigationSessionHarnessOwner(
        task_root=tmp_path / "bundle",
        working_memory=memory,
        task_input={
            "scientific_question": "原始问题",
            "supplementary_scientific_input": {"legacy_fact": "保留"},
        },
        policy={},
        dependencies={},
    )
    request = InvestigationHarnessRequest(
        research_task_id="inquiry-execution",
        module_invocation_id="investigation-legacy",
        module_kind="hypothesis-investigation",
        input_scientific_record_ids=("selected-1",),
        module_input_digest="e" * 64,
        context_binding=CreateInvestigationContext(
            investigation_context_id="context-legacy",
            hypothesis_batch_record_id="batch-1",
        ).to_dict(),
    )

    built = factory._build_provider_request(
        request,
        session=SessionReference(
            research_task_id="inquiry-execution",
            investigation_context_id="context-legacy",
            hypothesis_execution_input_record_id="selected-1",
            investigation_session_id="session-selected-1",
            workspace_identity="workspace-selected-1",
            minichem_scope_id="scope-selected-1",
            scope_contract_digest="2" * 64,
            capability_generation_digest="1" * 64,
        ),
    )

    payload = json.loads(built["messages"][1]["content"])
    assert payload["hypothesis"]["claim"] == "original-1"
    assert payload["frozen_context"] == {"legacy_fact": "保留"}
    assert "必须实际调用 TS Pipeline" in built["messages"][0]["content"]
