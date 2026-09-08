"""V2 默认 native adapter 的确定性跨层恢复回归。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from chemistry_multiagent.controllers.investigation_v2_module_controller import (
    ModuleInvocationRequest as ControllerModuleInvocationRequest,
    ModuleSuspended,
)
from chemistry_multiagent.controllers.investigation_v2_native_adapters import (
    InvestigationSessionHarnessOwner,
    InvestigationContextModuleAdapter,
    InvestigationSessionOwner,
    _NativeEvidenceVerifier,
    _NativeScopeCapabilityManager,
    _session_owner_workspace,
    build_native_investigation_module_adapter,
    build_native_review_module_adapter,
)
from chemistry_multiagent.harness.session_store import CodexSessionOwner, CodexSessionStore
from chemistry_multiagent.investigation.investigation_backend import (
    EvidenceResultRejected,
    NativeAction,
    NativeTurnResult,
)
from chemistry_multiagent.investigation.context_store import (
    InvestigationContextIdentity,
    InvestigationSessionIdentity,
)
from chemistry_multiagent.investigation.harness_recovery_store import SessionReference
from chemistry_multiagent.investigation.harness_runtime import InvestigationHarnessRequest
from chemistry_multiagent.investigation.harness_runtime import Suspended
from chemistry_multiagent.investigation.minichem_gateway import (
    InvestigationMiniChemReferenceRejected,
)
from chemistry_multiagent.memory import (
    ModuleInvocationRequest,
    ResearchTaskWorkingMemory,
    PreparedModuleInvocationOutput,
    ScientificRecord,
)
from chemistry_multiagent.memory.v2_identities import (
    CreateInvestigationContext,
    ReuseInvestigationContext,
    StatelessModule,
)
from chemistry_multiagent.tools.minichem_adapter import MiniChemMcpAdapter
from chemistry_multiagent.tools.minichem_catalog_freeze import FrozenMiniChemCatalog


def _workspace_relpath(
    investigation_session_id: str,
    research_task_id: str,
    context_id: str,
    hypothesis_id: str,
) -> str:
    """返回测试 Session 对应的当前生产 workspace 相对路径。"""

    return (
        f"internal/state/investigation_runtime/sessions/{investigation_session_id}/codex_sessions/"
        f"investigations/{research_task_id}/{context_id}/{hypothesis_id}/workspace"
    )


def test_session_resolver_records_the_actual_run_bundle_workspace(tmp_path: Path) -> None:
    """新 Session 身份必须记录实际使用的 Run Bundle workspace，不写目标旧布局。"""

    context = InvestigationContextIdentity(
        investigation_context_id="context-workspace-path",
        research_task_id="task-workspace-path",
        hypothesis_batch_record_id="batch-workspace-path",
    )
    selected = ScientificRecord.from_payload(
        scientific_record_id="hypothesis-workspace-path",
        research_task_id=context.research_task_id,
        scientific_record_type="SelectedHypothesisRecord",
        scientific_record_schema_version=1,
        payload={
            "hypothesis_ref": "hypothesis-ref",
            "rank": 1,
            "hypothesis": {"statement": "workspace path"},
        },
        source_scientific_record_ids=(context.hypothesis_batch_record_id,),
        producer_module_invocation_id="selection-workspace-path",
    )
    resolver = InvestigationSessionOwner(task_root=tmp_path, policy={}, dependencies={})

    session = resolver.plan_session(context, selected)

    assert session.workspace_identity == _workspace_relpath(
        session.investigation_session_id,
        context.research_task_id,
        context.investigation_context_id,
        selected.scientific_record_id,
    )


def test_legacy_session_workspace_identity_is_rejected(
    tmp_path: Path,
) -> None:
    """生产 Session resolver 不把旧相对目录静默映射成当前路径。"""

    session = InvestigationSessionIdentity(
        investigation_session_id="investigation-session-abcdef",
        investigation_context_id="context-legacy-workspace",
        research_task_id="task-legacy-workspace",
        hypothesis_execution_input_record_id="hypothesis-legacy-workspace",
        workspace_identity="inquiry-runtime/sessions/abcdef",
        minichem_scope_id="scope-legacy-workspace",
        scope_contract_digest="a" * 64,
    )

    with pytest.raises(ValueError, match="workspace 身份"):
        _session_owner_workspace(tmp_path, session)
    assert not (tmp_path / "inquiry-runtime").exists()


def test_native_evidence_verifier_separates_job_submission_from_typed_operations() -> None:
    """job submission identity 不得冒充 typed operation，独立 Action operation 必须保留。"""

    job_submission = "scope-submission-" + "a" * 64
    artifact_declaration = "scope-submission-" + "b" * 64

    class _Gateway:
        def __init__(self) -> None:
            self.results: list[Any] = []

        def verify_investigation_result_with_evidence(
            self,
            _session: Any,
            result: Any,
        ) -> Any:
            self.results.append(result)
            return SimpleNamespace(
                evidence=SimpleNamespace(
                    to_payload=lambda: {
                        "verification_object_id": "verification-native",
                        "verification_digest": "c" * 64,
                        "jobs": [],
                    }
                )
            )

        def materialize_result_artifacts(
            self,
            _session: Any,
            result: Any,
        ) -> dict[str, Any]:
            self.results.append(result)
            return {}

    gateway = _Gateway()
    verifier = _NativeEvidenceVerifier(
        gateway=gateway,  # type: ignore[arg-type]
        session=SessionReference(
            research_task_id="task-native-evidence",
            investigation_context_id="batch-native-evidence",
            hypothesis_execution_input_record_id="combination-native-evidence",
            investigation_session_id="session-native-evidence",
            workspace_identity="workspace-native-evidence",
            minichem_scope_id="scope-native-evidence",
            scope_contract_digest="e" * 64,
            capability_generation_digest="d" * 64,
        ),
        recovery_store=SimpleNamespace(
                read_snapshot=lambda _identity: SimpleNamespace(
                    last_attempt=SimpleNamespace(
                        model_thread_id="thread-native-evidence",
                        model_turn_id="turn-native-evidence",
                    )
                )
        ),
        artifact_manifest_registry={},
    )
    response = {
        "self_judgment": "supported",
        "narrative": "已核对 job 与独立 artifact declaration。",
        "job_refs": ["job_deadbeef"],
        "operation_refs": [job_submission, artifact_declaration],
        "artifact_refs": [],
        "observations": ["job 已完成"],
        "limitations": [],
    }

    verifier.verify_evidence(
        response,
        request=SimpleNamespace(
            research_task_id="task-native-evidence",
            module_invocation_id="module-native-evidence",
            harness_invocation_id="harness-native-evidence",
            module_input_digest="e" * 64,
        ),
        jobs=(SimpleNamespace(operation_id=job_submission),),
        prepared_output=SimpleNamespace(),
    )

    assert len(gateway.results) == 2
    assert all(
        result.operation_refs == (artifact_declaration,)
        for result in gateway.results
    )


def test_native_evidence_verifier_returns_safe_result_repair_feedback() -> None:
    """权威拒绝项应进入同 thread Result 修正，不应退化为普通 owner 缺失。"""

    class _RejectingGateway:
        def verify_investigation_result_with_evidence(self, *_args: Any) -> Any:
            raise InvestigationMiniChemReferenceRejected(
                (
                    {
                        "kind": "operation",
                        "operation_id": "operation-running",
                        "reason_code": "operation_not_completed",
                    },
                )
            )

    verifier = _NativeEvidenceVerifier(
        gateway=_RejectingGateway(),  # type: ignore[arg-type]
        session=SessionReference(
            research_task_id="task-native-rejected",
            investigation_context_id="batch-native-rejected",
            hypothesis_execution_input_record_id="combination-native-rejected",
            investigation_session_id="session-native-rejected",
            workspace_identity="workspace-native-rejected",
            minichem_scope_id="scope-native-rejected",
            scope_contract_digest="e" * 64,
            capability_generation_digest="d" * 64,
        ),
        recovery_store=SimpleNamespace(
                read_snapshot=lambda _identity: SimpleNamespace(
                    last_attempt=SimpleNamespace(
                        model_thread_id="thread-native-rejected",
                        model_turn_id="turn-native-rejected",
                    )
                )
        ),
        artifact_manifest_registry={},
    )

    with pytest.raises(EvidenceResultRejected) as captured:
        verifier.verify_evidence(
            {
                "self_judgment": "supported",
                "narrative": "当前结果仍含未完成 operation。",
                "job_refs": ["job_deadbeef"],
                "operation_refs": ["operation-running"],
                "artifact_refs": [],
                "observations": ["已有部分结果"],
                "limitations": ["一个 operation 未完成"],
            },
                request=SimpleNamespace(
                    research_task_id="task-native-rejected",
                    module_invocation_id="module-native-rejected",
                    harness_invocation_id="harness-native-rejected",
                    module_input_digest="e" * 64,
                ),
            jobs=(),
            prepared_output=SimpleNamespace(),
        )

    assert captured.value.to_action_result() == {
        "kind": "result_validation",
        "status": "failed",
        "reason_code": "evidence_references_rejected",
        "invalid_references": [
            {
                "kind": "operation",
                "operation_id": "operation-running",
                "reason_code": "operation_not_completed",
            }
        ],
    }


def _record(research_task_id: str, module_id: str, scientific_record_id: str, scientific_record_type: str, payload: dict[str, Any], refs: tuple[str, ...] = (), scientific_record_schema_version: str | int = 1) -> ScientificRecord:
    return ScientificRecord.from_payload(
        scientific_record_id=scientific_record_id,
        research_task_id=research_task_id,
        scientific_record_type=scientific_record_type,
        scientific_record_schema_version=scientific_record_schema_version,
        producer_module_invocation_id=module_id,
        source_scientific_record_ids=refs,
        payload=payload,
    )


def _commit(service: ResearchTaskWorkingMemory, research_task_id: str, module_id: str, kind: str, records: tuple[ScientificRecord, ...], digest: str) -> None:
    facts = service.read_research_task_facts(research_task_id)
    context_binding = StatelessModule()
    if kind == "hypothesis-investigation":
        context_binding = CreateInvestigationContext(
            investigation_context_id=f"context-{research_task_id}",
            hypothesis_batch_record_id=records[0].source_scientific_record_ids[0],
        )
    service.start_module_invocation(
        research_task_id,
        facts.research_task_sequence,
        ModuleInvocationRequest(
            research_task_id=research_task_id,
            module_invocation_id=module_id,
            module_kind=kind.replace("_", "-"),
            input_scientific_record_ids=records[0].source_scientific_record_ids,
            module_input_digest=digest,
            context_binding=context_binding,
        ),
    )
    service.complete_module_invocation(
        research_task_id,
        service.read_research_task_facts(research_task_id).research_task_sequence,
        PreparedModuleInvocationOutput.from_records(
            module_invocation_id=module_id,
            module_input_digest=digest,
            output_scientific_records=records,
        ),
    )


def _seed_selection(service: ResearchTaskWorkingMemory, research_task_id: str) -> tuple[str, ...]:
    service.create_research_task(
        research_task_id=research_task_id,
        contract_version="investigation_v2.task.v1",
        input={"scientific_question": "验证 native 恢复", "task_config": {"top_n_hypotheses": 1}},
        policy={"entry_strategy": "author_hypothesis_import"},
        dependencies={},
    )
    batch = _record(research_task_id, "hypothesis-import", "author-batch", "HypothesisBatchRecord", {"imported_hypotheses": [{"hypothesis_ref": "h1", "source_ref": "author-source", "source_digest": "a" * 64, "hypothesis": {"hypothesis_id": "h1", "claim": "claim"}, "provenance": {"kind": "author"}}]}, (), "author-import.v2")
    selected = _record(research_task_id, "hypothesis-import", "selected", "SelectedHypothesisRecord", {"hypothesis_ref": "h1", "rank": 1, "hypothesis": {"hypothesis_id": "h1", "claim": "claim"}}, ("author-batch",))
    _commit(service, research_task_id, "hypothesis-import", "hypothesis-import", (batch, selected), "d" * 64)
    return ("selected",)


def _seed_standard_top5(service: ResearchTaskWorkingMemory, research_task_id: str) -> tuple[str, ...]:
    """写入标准入口的 schema 2 Batch、选择摘要和五条逐假设记录。"""

    service.create_research_task(
        research_task_id=research_task_id,
        contract_version="investigation_v2.task.v1",
        input={"scientific_question": "验证五个 Session", "task_config": {"top_n_hypotheses": 5}},
        policy={"entry_strategy": "standard_hypothesis_generation"},
        dependencies={},
    )
    hypotheses = [
        {
            "hypothesis_ref": f"h{index}",
            "claim": f"claim-{index}",
            "hypothesis": {"hypothesis_id": f"h{index}", "claim": f"claim-{index}"},
        }
        for index in range(1, 6)
    ]
    batch = _record(
        research_task_id,
        "hypothesis-generation",
        "batch-standard",
        "HypothesisBatchRecord",
        {
            "hypotheses": hypotheses,
            "rationale": "deterministic five hypotheses",
            "verification_plan": "one independent session per hypothesis",
        },
        (),
    )
    _commit(service, research_task_id, "hypothesis-generation", "hypothesis-generation", (batch,), "a" * 64)
    selection = _record(
        research_task_id,
        "selection-standard",
        "selection-standard",
        "HypothesisSelectionRecord",
        {
            "selected_hypotheses": [
                {"hypothesis_ref": item["hypothesis_ref"], "rank": index}
                for index, item in enumerate(hypotheses, start=1)
            ],
            "selection_reason": "all five are selected for deterministic recovery test",
        },
        (batch.scientific_record_id,),
        2,
    )
    selected = tuple(
        _record(
            research_task_id,
            "selection-standard",
            f"selected-{index}",
            "SelectedHypothesisRecord",
            {
                "hypothesis_ref": item["hypothesis_ref"],
                "rank": index,
                "hypothesis": item["hypothesis"],
            },
            (selection.scientific_record_id,),
            2,
        )
        for index, item in enumerate(hypotheses, start=1)
    )
    _commit(
        service,
        research_task_id,
        "selection-standard",
        "hypothesis-selection",
        (selection, *selected),
        "b" * 64,
    )
    return tuple(record.scientific_record_id for record in selected)


@dataclass
class _Execution:
    status: str = "pending"

    def __post_init__(self) -> None:
        self.submissions: list[str] = []
        self.queries: list[str] = []
        self.collections: list[str] = []

    def submit_job(self, action: NativeAction, *, session: Any) -> dict[str, Any]:
        self.submissions.append(action.action_id)
        return {"job_id": "job-native-1", "operation_id": "op-native-1", "submission_key": "sub-native-1", "status": self.status}

    def query_job(self, job: Any, *, session: Any) -> dict[str, Any]:
        self.queries.append(job.minichem_job_id)
        return {"job_id": job.minichem_job_id, "status": self.status}

    def collect_job(self, job: Any, *, session: Any) -> dict[str, Any]:
        self.collections.append(job.minichem_job_id)
        return {"job_id": job.minichem_job_id, "operation_id": job.minichem_operation_id, "submission_key": job.minichem_submission_key, "status": self.status, "artifact_refs": ["artifact-native"]}


class _Provider:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[dict[str, Any], ...], bool]] = []

    def run_turn(self, request: dict[str, Any], *, session: Any, action_results: tuple[dict[str, Any], ...], is_resume: bool, thread_id: str | None = None) -> NativeTurnResult:
        self.calls.append((action_results, is_resume))
        if len(self.calls) == 1:
            return NativeTurnResult(thread_id="thread-native", turn_id="turn-native-1", actions=(NativeAction("action-native", "submit_job", {"software_id": "gaussian", "arguments": [], "staged_inputs": [], "stdin_target": "input.gjf", "resource_limits": {}, "calculation_intent": "test", "label": "test"}),))
        return NativeTurnResult(thread_id="thread-native", turn_id="turn-native-2", result={"self_judgment": "supported", "narrative": "native result", "job_refs": ["job-native-1"], "operation_refs": ["op-native-1"], "artifact_refs": ["artifact-native"], "observations": ["observed"], "limitations": []})


class _FiveSessionProvider:
    """每个 Session 第一次提交作业，第二次在同 thread 返回结果。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []
        self._counts: dict[str, int] = {}

    def run_turn(self, request: dict[str, Any], *, session: Any, action_results: tuple[dict[str, Any], ...], is_resume: bool, thread_id: str | None = None) -> NativeTurnResult:
        del request, action_results, thread_id
        investigation_session_id = session.investigation_session_id
        count = self._counts.get(investigation_session_id, 0) + 1
        self._counts[investigation_session_id] = count
        self.calls.append((investigation_session_id, is_resume))
        job_id = f"job-{investigation_session_id}"
        if count == 1:
            return NativeTurnResult(
                thread_id=f"thread-{investigation_session_id}",
                turn_id=f"turn-{investigation_session_id}-1",
                actions=(
                    NativeAction(
                        f"action-{investigation_session_id}",
                        "submit_job",
                        {
                            "software_id": "gaussian",
                            "arguments": [],
                            "staged_inputs": [],
                            "stdin_target": "input.gjf",
                            "resource_limits": {},
                            "calculation_intent": "test",
                            "label": "five-session",
                        },
                    ),
                ),
            )
        return NativeTurnResult(
            thread_id=f"thread-{investigation_session_id}",
            turn_id=f"turn-{investigation_session_id}-2",
            result={
                "self_judgment": "supported",
                "narrative": f"result for {investigation_session_id}",
                "job_refs": [job_id],
                "operation_refs": [f"operation-{investigation_session_id}"],
                "artifact_refs": [f"artifact-{investigation_session_id}"],
                "observations": [f"observed {investigation_session_id}"],
                "limitations": [],
            },
        )


class _FiveSessionExecution:
    """按 Session 隔离作业身份；恢复时只查询和收集同一作业。"""

    def __init__(self) -> None:
        self.status = "pending"
        self.submissions: list[tuple[str, str]] = []
        self.queries: list[str] = []
        self.collections: list[str] = []

    def submit_job(self, action: NativeAction, *, session: Any) -> dict[str, Any]:
        job_id = f"job-{session.investigation_session_id}"
        self.submissions.append((session.investigation_session_id, job_id))
        return {
            "job_id": job_id,
            "operation_id": f"operation-{session.investigation_session_id}",
            "submission_key": f"submission-{session.investigation_session_id}",
            "status": self.status,
        }

    def query_job(self, job: Any, *, session: Any) -> dict[str, Any]:
        self.queries.append(job.minichem_job_id)
        return {"job_id": job.minichem_job_id, "operation_id": job.minichem_operation_id, "submission_key": job.minichem_submission_key, "status": self.status}

    def collect_job(self, job: Any, *, session: Any) -> dict[str, Any]:
        self.collections.append(job.minichem_job_id)
        return {"job_id": job.minichem_job_id, "operation_id": job.minichem_operation_id, "submission_key": job.minichem_submission_key, "status": "completed", "artifact_refs": [f"artifact-{session.investigation_session_id}"]}


def test_default_native_adapter_resumes_inner_harness_without_duplicate_submission(tmp_path: Path) -> None:
    research_task_id = "task-native-adapter"
    service = ResearchTaskWorkingMemory(str(tmp_path / "memory.sqlite3"))
    refs = _seed_selection(service, research_task_id)
    execution = _Execution()
    provider = _Provider()
    adapter = build_native_investigation_module_adapter(
        task_root=tmp_path / "bundle",
        working_memory=service,
        task_input={"scientific_question": "验证 native 恢复", "supplementary_scientific_input": {}},
        policy={"entry_strategy": "author_hypothesis_import"},
        dependencies={
            "native_provider_transport_factory": lambda _session, _runtime: provider,
            "native_execution_transport_factory": lambda _session: execution,
        },
    )
    assert isinstance(adapter, InvestigationContextModuleAdapter)
    request = ControllerModuleInvocationRequest(
        research_task_id=research_task_id,
        module_invocation_id="investigation-1",
        module_kind="hypothesis-investigation",
        input_scientific_record_ids=refs,
        module_input_digest="e" * 64,
        context_binding=CreateInvestigationContext(
            investigation_context_id="context-native",
            hypothesis_batch_record_id="author-batch",
        ),
    )

    first = adapter.start(request)
    assert isinstance(first, ModuleSuspended)
    assert execution.submissions == ["action-native"]

    execution.status = "completed"
    # 模拟 Controller/Harness 进程退出：重新打开 Working Memory、Context store
    # 和子 Harness recovery store，只复用外部 Provider/MiniChem owner。
    restarted_service = ResearchTaskWorkingMemory(
        str(tmp_path / "memory.sqlite3")
    )
    restarted_adapter = build_native_investigation_module_adapter(
        task_root=tmp_path / "bundle",
        working_memory=restarted_service,
        task_input={
            "scientific_question": "验证 native 恢复",
            "supplementary_scientific_input": {},
        },
        policy={"entry_strategy": "author_hypothesis_import"},
        dependencies={
            "native_provider_transport_factory": lambda _session, _runtime: provider,
            "native_execution_transport_factory": lambda _session: execution,
        },
    )
    resumed = restarted_adapter.resume(request)
    assert resumed.status == "Completed"
    assert execution.submissions == ["action-native"]
    assert execution.queries == ["job-native-1"]
    assert execution.collections == ["job-native-1"]
    assert len(provider.calls) == 2


def test_repair_request_recovers_thread_from_session_owner_store(tmp_path: Path) -> None:
    """不同 Module Run 的返修从同一 Session owner 读取原 Codex thread。"""

    research_task_id = "task-native-repair"
    service = ResearchTaskWorkingMemory(str(tmp_path / "memory.sqlite3"))
    factory = InvestigationSessionHarnessOwner(
        task_root=tmp_path / "bundle",
        working_memory=service,
        task_input={"scientific_question": "question"},
        policy={"entry_strategy": "author_hypothesis_import"},
        dependencies={},
    )
    session = InvestigationSessionIdentity(
        investigation_session_id="session-native",
        investigation_context_id="context-native",
        research_task_id=research_task_id,
        hypothesis_execution_input_record_id="hypothesis-native",
        workspace_identity=_workspace_relpath(
            "session-native", research_task_id, "context-native", "hypothesis-native"
        ),
        minichem_scope_id="scope-native",
        scope_contract_digest="a" * 64,
    )
    session_ref = SessionReference(
        research_task_id=research_task_id,
        investigation_context_id=session.investigation_context_id,
        hypothesis_execution_input_record_id=session.hypothesis_execution_input_record_id,
        investigation_session_id=session.investigation_session_id,
        workspace_identity="workspace-session-native",
        minichem_scope_id=session.minichem_scope_id,
        scope_contract_digest=session.scope_contract_digest,
        capability_generation_digest=session.scope_contract_digest,
    )
    owner_store = CodexSessionStore.from_root(
        tmp_path
        / "bundle"
        / "internal"
        / "state"
        / "investigation_runtime"
        / "sessions"
        / session.investigation_session_id
        / "codex_sessions",
        owner=CodexSessionOwner.investigation(
            research_task_id,
            session_ref.investigation_context_id,
            session_ref.hypothesis_execution_input_record_id,
        ),
        scope_id=session_ref.minichem_scope_id,
        model="gpt-5.6-sol",
    )
    owner_store.create("thread-native")
    request = InvestigationHarnessRequest(
        research_task_id=research_task_id,
        module_invocation_id="investigation-repair-1",
        module_kind="hypothesis-investigation",
        input_scientific_record_ids=("repair-request",),
        module_input_digest="b" * 64,
        context_binding=CreateInvestigationContext(
            investigation_context_id=session.investigation_context_id,
            hypothesis_batch_record_id="batch-native",
        ).to_dict(),
    )

    assert factory._load_existing_thread_id(request, session_ref) == "thread-native"


def test_later_repair_prompt_uses_direct_request_instead_of_older_source(
    tmp_path: Path,
) -> None:
    """第二轮 repair 必须使用本轮直接输入，不能从来源闭包误取第一轮建议。"""

    research_task_id = "task-native-later-repair"
    selected = _record(
        research_task_id,
        "selection",
        "selected",
        "SelectedHypothesisRecord",
        {
            "hypothesis_ref": "h1",
            "rank": 1,
            "hypothesis": {"hypothesis_id": "h1", "claim": "claim"},
        },
    )
    first_repair = _record(
        research_task_id,
        "review-1",
        "repair-1",
        "RepairRequestRecord",
        {
            "problem": "first review failure",
            "instructions": "第一轮建议",
            "result_ref": "result-1",
        },
        (selected.scientific_record_id,),
    )
    second_repair = _record(
        research_task_id,
        "review-2",
        "repair-2",
        "RepairRequestRecord",
        {
            "problem": "second review failure",
            "instructions": "第二轮建议",
            "result_ref": "result-2",
        },
        (first_repair.scientific_record_id,),
    )
    current_result = _record(
        research_task_id,
        "investigation-2",
        "result-2",
        "InvestigationResultRecord",
        {
            "conclusion": "current result",
            "self_judgment": "supported",
            "observations": ["current observation"],
            "limitations": [],
            "evidence_ref": "evidence-2",
        },
        (first_repair.scientific_record_id,),
    )
    records = {
        record.scientific_record_id: record
        for record in (selected, first_repair, second_repair, current_result)
    }

    class _WorkingMemory:
        @staticmethod
        def read_research_task_scientific_records(
            _research_task_id: str,
            scientific_record_ids: tuple[str, ...],
        ) -> tuple[ScientificRecord, ...]:
            return tuple(
                records[scientific_record_id]
                for scientific_record_id in scientific_record_ids
            )

    factory = InvestigationSessionHarnessOwner(
        task_root=tmp_path / "bundle",
        working_memory=_WorkingMemory(),  # type: ignore[arg-type]
        task_input={"scientific_question": "question"},
        policy={"entry_strategy": "author_hypothesis_import"},
        dependencies={},
    )
    session = SessionReference(
        research_task_id=research_task_id,
        investigation_context_id="context-native-later-repair",
        hypothesis_execution_input_record_id="selected",
        investigation_session_id="session-native-later-repair",
        workspace_identity="workspace-native-later-repair",
        minichem_scope_id="scope-native-later-repair",
        scope_contract_digest="b" * 64,
        capability_generation_digest="a" * 64,
    )
    request = InvestigationHarnessRequest(
        research_task_id=research_task_id,
        module_invocation_id="investigation-repair-2",
        module_kind="hypothesis-investigation",
        input_scientific_record_ids=(
            selected.scientific_record_id,
            second_repair.scientific_record_id,
            current_result.scientific_record_id,
        ),
        module_input_digest="b" * 64,
        context_binding=CreateInvestigationContext(
            investigation_context_id=session.investigation_context_id,
            hypothesis_batch_record_id="batch-native-later-repair",
        ).to_dict(),
    )

    built = factory._build_provider_request(
        request,
        session=session,
        is_resume=True,
        thread_id="thread-native-later-repair",
    )

    payload = json.loads(built["messages"][-1]["content"])
    assert payload == {
        "repair_advice": "第二轮建议",
        "result_version": "v1",
    }


def test_evidence_result_rejection_builds_same_thread_repair_request(
    tmp_path: Path,
) -> None:
    """Evidence 拒绝反馈必须进入同一 thread 的新 repair turn，不能重发初始请求。"""

    research_task_id = "task-native-evidence-repair"
    service = ResearchTaskWorkingMemory(str(tmp_path / "memory.sqlite3"))
    refs = _seed_selection(service, research_task_id)
    factory = InvestigationSessionHarnessOwner(
        task_root=tmp_path / "bundle",
        working_memory=service,
        task_input={"scientific_question": "核对 Evidence 引用"},
        policy={"entry_strategy": "author_hypothesis_import"},
        dependencies={},
    )
    session = SessionReference(
        research_task_id=research_task_id,
        investigation_context_id="context-native-evidence-repair",
        hypothesis_execution_input_record_id="selected",
        investigation_session_id="session-native-evidence-repair",
        workspace_identity="workspace-native-evidence-repair",
        minichem_scope_id="scope-native-evidence-repair",
        scope_contract_digest="b" * 64,
        capability_generation_digest="a" * 64,
    )
    request = InvestigationHarnessRequest(
        research_task_id=research_task_id,
        module_invocation_id="investigation-native-evidence-repair",
        module_kind="hypothesis-investigation",
        input_scientific_record_ids=refs,
        module_input_digest="b" * 64,
        context_binding=CreateInvestigationContext(
            investigation_context_id=session.investigation_context_id,
            hypothesis_batch_record_id="author-batch",
        ).to_dict(),
    )
    validation = {
        "kind": "result_validation",
        "status": "failed",
        "reason_code": "evidence_references_rejected",
        "invalid_references": [
            {
                "kind": "operation",
                "operation_id": "scope-operation-missing",
                "reason_code": "operation_not_declared",
            }
        ],
    }

    built = factory._build_provider_request(
        request,
        session=session,
        action_results=(validation,),
        is_resume=True,
        thread_id="thread-native-evidence-repair",
    )

    assert built["metadata"]["request_kind"] == "deepseek_repair"
    assert built["thread_id"] == "thread-native-evidence-repair"
    payload = json.loads(built["messages"][-1]["content"])
    assert json.loads(payload["repair_advice"]) == validation


def test_default_native_adapter_drives_five_sessions_and_resumes_only_first_job(tmp_path: Path) -> None:
    """标准 Top 5 并发驱动五个独立 Session，恢复不重复任何提交。"""

    research_task_id = "task-native-top5"
    service = ResearchTaskWorkingMemory(str(tmp_path / "memory.sqlite3"))
    refs = _seed_standard_top5(service, research_task_id)
    provider = _FiveSessionProvider()
    execution = _FiveSessionExecution()
    adapter = build_native_investigation_module_adapter(
        task_root=tmp_path / "bundle",
        working_memory=service,
        task_input={"scientific_question": "验证五个 Session", "supplementary_scientific_input": {}},
        policy={"entry_strategy": "standard_hypothesis_generation"},
        dependencies={
            "native_provider_transport_factory": lambda _session, _runtime: provider,
            "native_execution_transport_factory": lambda _session: execution,
        },
    )
    request = ControllerModuleInvocationRequest(
        research_task_id=research_task_id,
        module_invocation_id="investigation-top5",
        module_kind="hypothesis-investigation",
        input_scientific_record_ids=refs,
        module_input_digest="c" * 64,
        context_binding=CreateInvestigationContext(
            investigation_context_id="context-top5",
            hypothesis_batch_record_id="batch-standard",
        ),
    )

    first = adapter.start(request)
    assert isinstance(first, ModuleSuspended)
    assert len(execution.submissions) == 5
    execution.status = "completed"

    resumed = adapter.resume(request)
    assert resumed.status == "Completed"
    assert len(execution.submissions) == 5
    assert len({investigation_session_id for investigation_session_id, _job_id in execution.submissions}) == 5
    assert execution.submissions.count(execution.submissions[0]) == 1
    expected_jobs = [job_id for _session_id, job_id in execution.submissions]
    assert sorted(execution.queries) == sorted(expected_jobs)
    assert sorted(execution.collections) == sorted(expected_jobs)
    assert len(provider.calls) == 10


def test_native_factory_applies_compact_prompt_to_all_investigation_harnesses_only(
    tmp_path: Path,
) -> None:
    """共享哨兵进入五个 GPT Session runtime，但不进入独立 DeepSeek Review runtime。"""

    research_task_id = "task-native-compact-prompt"
    sentinel = "ARCHE_INVESTIGATION_COMPACT_SENTINEL\n保留科学证据"
    service = ResearchTaskWorkingMemory(str(tmp_path / "memory.sqlite3"))
    captured_runtimes = []

    def capture_runtime(_session, runtime):
        captured_runtimes.append(runtime)
        return _Provider()

    shared_dependencies = {
        "investigation_codex_thread": {
            "compact_prompt": sentinel,
            "compact_prompt_sha256": hashlib.sha256(
                sentinel.encode("utf-8")
            ).hexdigest(),
            "model_context_window": 1_000_000,
        },
        "native_provider_transport_factory": capture_runtime,
        "native_execution_transport": _Execution(),
    }
    factory = InvestigationSessionHarnessOwner(
        task_root=tmp_path / "bundle",
        working_memory=service,
        task_input={"scientific_question": "验证压缩提示词装配"},
        policy={"entry_strategy": "standard_hypothesis_generation"},
        dependencies=shared_dependencies,
    )
    sessions = tuple(
        InvestigationSessionIdentity(
            investigation_session_id=f"session-{index}",
            investigation_context_id="context-1",
            research_task_id=research_task_id,
            hypothesis_execution_input_record_id=f"hypothesis-{index}",
            workspace_identity=_workspace_relpath(
                f"session-{index}", research_task_id, "context-1", f"hypothesis-{index}"
            ),
            minichem_scope_id=f"scope-{index}",
            scope_contract_digest=f"{index:064x}",
        )
        for index in range(1, 6)
    )

    for session in sessions:
        factory.open_session(session)

    review_adapter = build_native_review_module_adapter(
        task_root=tmp_path / "bundle",
        working_memory=service,
        task_input={"scientific_question": "验证压缩提示词装配"},
        policy={"review_scope_id": "task-review"},
        dependencies=shared_dependencies,
    )

    assert len(captured_runtimes) == 5
    assert len({id(runtime) for runtime in captured_runtimes}) == 5
    assert {
        runtime.investigation_compact_prompt for runtime in captured_runtimes
    } == {sentinel}
    assert {
        runtime.investigation_model_context_window for runtime in captured_runtimes
    } == {1_000_000}
    assert review_adapter.runtime.review_runtime.investigation_compact_prompt is None
    assert (
        review_adapter.runtime.review_runtime.investigation_model_context_window
        is None
    )


@pytest.mark.parametrize(
    "thread_policy",
    (
        {
            "compact_prompt": "保留科学事实",
            "compact_prompt_sha256": "0" * 64,
            "model_context_window": 1_000_000,
        },
        {
            "compact_prompt": "保留科学事实",
            "compact_prompt_sha256": hashlib.sha256(
                "保留科学事实".encode("utf-8")
            ).hexdigest(),
        },
        {
            "compact_prompt": "保留科学事实",
            "compact_prompt_sha256": hashlib.sha256(
                "保留科学事实".encode("utf-8")
            ).hexdigest(),
            "model_context_window": True,
        },
    ),
)
def test_native_factory_rejects_drifted_frozen_codex_thread_policy(
    tmp_path: Path,
    thread_policy: dict[str, object],
) -> None:
    """冻结提示词摘要、字段集合或上下文档位损坏时不能创建 Session runtime。"""

    service = ResearchTaskWorkingMemory(str(tmp_path / "memory.sqlite3"))
    with pytest.raises(ValueError, match="摘要漂移|字段不完整|正整数"):
        InvestigationSessionHarnessOwner(
            task_root=tmp_path / "bundle",
            working_memory=service,
            task_input={"scientific_question": "验证冻结 Codex thread 配置"},
            policy={"entry_strategy": "standard_hypothesis_generation"},
            dependencies={"investigation_codex_thread": thread_policy},
        )


@dataclass
class _ScopedCapability:
    token: str = "scope-capability-token"


class _ScopedManager:
    """为 open_session 测试提供最小 scope owner。"""

    def __init__(self) -> None:
        self.sessions: list[str] = []
        self.capability = _ScopedCapability()

    def ensure_session(self, session: InvestigationSessionIdentity) -> None:
        self.sessions.append(session.investigation_session_id)

    def capability_for(self, session: InvestigationSessionIdentity) -> _ScopedCapability:
        if session.investigation_session_id not in self.sessions:
            self.ensure_session(session)
        return self.capability


class _InvestigationProfileMcp:
    """只暴露 9012 Investigation profile 的 discovery 和 execution 工具。"""

    def __init__(self, catalog_hash: str) -> None:
        self.catalog_hash = catalog_hash
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.reject_discovery = False

    def initialize(self) -> dict[str, Any]:
        return {"protocolVersion": "2025-06-18", "serverInfo": {"name": "9012-fake"}}

    def read_resource(self, uri: str) -> None:
        raise AssertionError(f"9012 不应读取 Catalog resource: {uri}")

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = dict(arguments["request"])
        self.calls.append((name, payload))
        if name == "list_action_domains":
            if self.reject_discovery:
                raise AssertionError("恢复既有 job 不应重新读取全局 Catalog")
            return {
                "status": "success",
                "catalog_hash": self.catalog_hash,
                "catalog_version": "7",
                "domains": [{"domain": "native_software", "action_ids": []}],
            }
        if name == "submit_native_job":
            return {
                "status": "success",
                "job_id": "job_deadbeef",
                "job_status": "queued",
                "operation_id": "server-operation-1",
                "submission_key": payload.get("submission_key"),
            }
        if name == "get_execution_job":
            return {
                "status": "success",
                "job_id": "job_deadbeef",
                "job_status": "success",
                "job": {"job_id": "job_deadbeef", "status": "success"},
                "terminal": True,
                "operation_id": "server-operation-1",
                "submission_key": "stable-submission-1",
            }
        if name == "collect_execution_job":
            return {
                "status": "success",
                "job_id": "job_deadbeef",
                "job_status": "success",
                "job": {"job_id": "job_deadbeef", "status": "success"},
                "ready": True,
                "operation_id": "server-operation-1",
                "submission_key": "stable-submission-1",
            }
        raise AssertionError(f"测试未定义 9012 工具: {name}")


class _ScopedProvider:
    def __init__(self) -> None:
        self.calls = 0

    def run_turn(
        self,
        request: dict[str, Any],
        *,
        session: Any,
        action_results: tuple[dict[str, Any], ...],
        is_resume: bool,
        thread_id: str | None = None,
    ) -> NativeTurnResult:
        del request, session, action_results, is_resume, thread_id
        self.calls += 1
        return NativeTurnResult(
            thread_id="thread-scoped",
            turn_id=f"turn-scoped-{self.calls}",
            actions=(
                NativeAction(
                    "action-scoped",
                    "submit_job",
                    {
                        "software_id": "gaussian",
                        "arguments": [],
                        "staged_inputs": [],
                        "stdin_target": None,
                        "resource_limits": {},
                        "calculation_intent": "single_point",
                        "label": "scoped identity",
                    },
                ),
            ),
        )


class _ScopedResultProvider:
    def __init__(self) -> None:
        self.calls = 0

    def run_turn(
        self,
        request: dict[str, Any],
        *,
        session: Any,
        action_results: tuple[dict[str, Any], ...],
        is_resume: bool,
        thread_id: str | None = None,
    ) -> NativeTurnResult:
        del request, session, action_results, is_resume, thread_id
        self.calls += 1
        if self.calls == 1:
            return NativeTurnResult(
                thread_id="thread-scoped-result",
                turn_id="turn-scoped-result-1",
                actions=(
                    NativeAction(
                        "action-scoped-result",
                        "submit_job",
                        {
                            "software_id": "gaussian",
                            "arguments": [],
                            "staged_inputs": [],
                            "stdin_target": None,
                            "resource_limits": {},
                            "calculation_intent": "single_point",
                            "label": "scoped identity",
                            "operation_id": "caller-operation-must-not-cross-boundary",
                            "submission_key": "stable-submission-1",
                        },
                    ),
                ),
            )
        return NativeTurnResult(
            thread_id="thread-scoped-result",
            turn_id="turn-scoped-result-2",
            result={
                "self_judgment": "supported",
                "narrative": "scoped result",
                "job_refs": ["job_deadbeef"],
                "operation_refs": [],
                "artifact_refs": [],
                "observations": ["observed"],
                "limitations": [],
            },
        )


class _ResumeResultProvider:
    """恢复原 job 后直接返回 job-only Result。"""

    def run_turn(
        self,
        request: dict[str, Any],
        *,
        session: Any,
        action_results: tuple[dict[str, Any], ...],
        is_resume: bool,
        thread_id: str | None = None,
    ) -> NativeTurnResult:
        del request, session, action_results, is_resume, thread_id
        return NativeTurnResult(
            thread_id="thread-scoped-result",
            turn_id="turn-scoped-result-2",
            result={
                "self_judgment": "supported",
                "narrative": "scoped result",
                "job_refs": ["job_deadbeef"],
                "operation_refs": [],
                "artifact_refs": [],
                "observations": ["observed"],
                "limitations": [],
            },
        )


def test_open_session_uses_investigation_profile_for_9012_catalog_identity(tmp_path: Path) -> None:
    """9012 首次执行只通过 list_action_domains 核对冻结 Catalog 身份。"""

    catalog_hash = "a" * 64
    mcp = _InvestigationProfileMcp(catalog_hash)
    provider = _ScopedProvider()
    manager = _ScopedManager()
    frozen = FrozenMiniChemCatalog(
        catalog_hash=catalog_hash,
        snapshot={"scientific_record_schema_version": 7},
        package_root=tmp_path,
    )

    def make_adapter(**kwargs: Any) -> MiniChemMcpAdapter:
        return MiniChemMcpAdapter(
            endpoint=kwargs["endpoint"],
            session_factory=lambda: mcp,
            operation_root=kwargs["operation_root"],
            retry_count=0,
        )

    factory = InvestigationSessionHarnessOwner(
        task_root=tmp_path / "bundle",
        working_memory=ResearchTaskWorkingMemory(str(tmp_path / "memory.sqlite3")),
        task_input={"scientific_question": "验证 9012 identity"},
        policy={"entry_strategy": "author_hypothesis_import"},
        dependencies={
            "native_scope_manager": manager,
            "native_provider_transport": provider,
            "native_minichem_adapter_factory": make_adapter,
            "frozen_catalog": frozen,
            "minichem_endpoint": "http://127.0.0.1:1/mcp",
            "catalog_endpoint": "http://127.0.0.1:1/mcp",
            "submission_contract_checker": lambda action, session: True,
            "evidence_verifier": lambda response, **kwargs: {},
        },
    )
    session = InvestigationSessionIdentity(
        investigation_session_id="session-scoped",
        investigation_context_id="context-scoped",
        research_task_id="task-scoped",
        hypothesis_execution_input_record_id="hypothesis-scoped",
        workspace_identity=_workspace_relpath(
            "session-scoped", "task-scoped", "context-scoped", "hypothesis-scoped"
        ),
        minichem_scope_id="scope-scoped",
        scope_contract_digest="b" * 64,
    )
    harness = factory.open_session(session)
    outcome = harness.start(
        InvestigationHarnessRequest(
            research_task_id="task-scoped",
            module_invocation_id="module-scoped",
            module_kind="hypothesis-investigation",
            input_scientific_record_ids=(),
            module_input_digest="c" * 64,
            context_binding=CreateInvestigationContext(
                investigation_context_id="context-scoped",
                hypothesis_batch_record_id="batch-scoped",
            ).to_dict(),
        )
    )

    assert isinstance(outcome, Suspended)
    assert [name for name, _payload in mcp.calls] == [
        "list_action_domains",
        "submit_native_job",
    ]


def test_open_session_does_not_use_global_catalog_hash_as_submit_admission(
    tmp_path: Path,
) -> None:
    """无关 Catalog 内容变化不阻断已由 pair checker 接受的新提交。"""

    frozen_hash = "a" * 64
    mcp = _InvestigationProfileMcp("b" * 64)
    frozen = FrozenMiniChemCatalog(
        catalog_hash=frozen_hash,
        snapshot={"scientific_record_schema_version": 7},
        package_root=tmp_path,
    )

    def make_adapter(**kwargs: Any) -> MiniChemMcpAdapter:
        return MiniChemMcpAdapter(
            endpoint=kwargs["endpoint"],
            session_factory=lambda: mcp,
            operation_root=kwargs["operation_root"],
            retry_count=0,
        )

    factory = InvestigationSessionHarnessOwner(
        task_root=tmp_path / "bundle",
        working_memory=ResearchTaskWorkingMemory(
            str(tmp_path / "memory.sqlite3")
        ),
        task_input={"scientific_question": "验证局部提交合同"},
        policy={"entry_strategy": "author_hypothesis_import"},
        dependencies={
            "native_scope_manager": _ScopedManager(),
            "native_provider_transport": _ScopedProvider(),
            "native_minichem_adapter_factory": make_adapter,
            "frozen_catalog": frozen,
            "minichem_endpoint": "http://127.0.0.1:1/mcp",
            "submission_contract_checker": lambda action, session: True,
            "evidence_verifier": lambda response, **kwargs: {},
        },
    )
    session = InvestigationSessionIdentity(
        investigation_session_id="session-pair-contract",
        investigation_context_id="context-pair-contract",
        research_task_id="task-pair-contract",
        hypothesis_execution_input_record_id="hypothesis-pair-contract",
        workspace_identity=_workspace_relpath(
            "session-pair-contract",
            "task-pair-contract",
            "context-pair-contract",
            "hypothesis-pair-contract",
        ),
        minichem_scope_id="scope-pair-contract",
        scope_contract_digest="e" * 64,
    )

    outcome = factory.open_session(session).start(
        InvestigationHarnessRequest(
            research_task_id=session.research_task_id,
            module_invocation_id="module-pair-contract",
            module_kind="hypothesis-investigation",
            input_scientific_record_ids=(),
            module_input_digest="f" * 64,
            context_binding=CreateInvestigationContext(
                investigation_context_id=session.investigation_context_id,
                hypothesis_batch_record_id="batch-pair-contract",
            ).to_dict(),
        )
    )

    assert isinstance(outcome, Suspended)
    assert [name for name, _payload in mcp.calls] == [
        "list_action_domains",
        "submit_native_job",
    ]


def test_open_session_uses_scoped_submission_key_and_server_job_identity(
    tmp_path: Path,
) -> None:
    """scope submit 不发送 caller operation_id，并可恢复 8 位 job identity。"""

    catalog_hash = "a" * 64
    mcp = _InvestigationProfileMcp(catalog_hash)
    provider = _ScopedResultProvider()
    manager = _ScopedManager()
    frozen = FrozenMiniChemCatalog(
        catalog_hash=catalog_hash,
        snapshot={"scientific_record_schema_version": 7},
        package_root=tmp_path,
    )

    def make_adapter(**kwargs: Any) -> MiniChemMcpAdapter:
        return MiniChemMcpAdapter(
            endpoint=kwargs["endpoint"],
            session_factory=lambda: mcp,
            operation_root=kwargs["operation_root"],
            retry_count=0,
        )

    factory = InvestigationSessionHarnessOwner(
        task_root=tmp_path / "bundle",
        working_memory=ResearchTaskWorkingMemory(str(tmp_path / "memory.sqlite3")),
        task_input={"scientific_question": "验证 scoped identity"},
        policy={"entry_strategy": "author_hypothesis_import"},
        dependencies={
            "native_scope_manager": manager,
            "native_provider_transport": provider,
            "native_minichem_adapter_factory": make_adapter,
            "frozen_catalog": frozen,
            "minichem_endpoint": "http://127.0.0.1:1/mcp",
            "catalog_endpoint": "http://127.0.0.1:1/mcp",
            "submission_contract_checker": lambda action, session: True,
            "evidence_verifier": lambda response, **kwargs: {},
        },
    )
    session = InvestigationSessionIdentity(
        investigation_session_id="session-scoped-result",
        investigation_context_id="context-scoped-result",
        research_task_id="task-scoped-result",
        hypothesis_execution_input_record_id="hypothesis-scoped-result",
        workspace_identity=_workspace_relpath(
            "session-scoped-result",
            "task-scoped-result",
            "context-scoped-result",
            "hypothesis-scoped-result",
        ),
        minichem_scope_id="scope-scoped-result",
        scope_contract_digest="b" * 64,
    )
    harness = factory.open_session(session)
    request = InvestigationHarnessRequest(
        research_task_id="task-scoped-result",
        module_invocation_id="module-scoped-result",
        module_kind="hypothesis-investigation",
        input_scientific_record_ids=(),
        module_input_digest="c" * 64,
        context_binding=CreateInvestigationContext(
            investigation_context_id="context-scoped-result",
            hypothesis_batch_record_id="batch-scoped-result",
        ).to_dict(),
    )

    first = harness.start(request)
    assert isinstance(first, Suspended)
    resumed = harness.resume(request)

    assert resumed.status == "Completed"
    submit_payload = next(
        payload for name, payload in mcp.calls if name == "submit_native_job"
    )
    assert submit_payload["submission_key"] == "stable-submission-1"
    assert "operation_id" not in submit_payload
    assert [name for name, _payload in mcp.calls] == [
        "list_action_domains",
        "submit_native_job",
        "get_execution_job",
        "collect_execution_job",
    ]


def test_open_session_resumes_existing_job_without_global_catalog_admission(
    tmp_path: Path,
) -> None:
    """恢复原 job 只核对持久身份，不要求当前 9012 Catalog 仍与提交时相同。"""

    catalog_hash = "a" * 64
    mcp = _InvestigationProfileMcp(catalog_hash)
    manager = _ScopedManager()
    frozen = FrozenMiniChemCatalog(
        catalog_hash=catalog_hash,
        snapshot={"scientific_record_schema_version": 7},
        package_root=tmp_path,
    )

    def make_adapter(**kwargs: Any) -> MiniChemMcpAdapter:
        return MiniChemMcpAdapter(
            endpoint=kwargs["endpoint"],
            session_factory=lambda: mcp,
            operation_root=kwargs["operation_root"],
            retry_count=0,
        )

    common = {
        "task_root": tmp_path / "bundle",
        "working_memory": ResearchTaskWorkingMemory(
            str(tmp_path / "memory.sqlite3")
        ),
        "task_input": {"scientific_question": "验证恢复不依赖当前 Catalog"},
        "policy": {"entry_strategy": "author_hypothesis_import"},
    }
    dependencies = {
        "native_scope_manager": manager,
        "native_provider_transport": _ScopedResultProvider(),
        "native_minichem_adapter_factory": make_adapter,
        "frozen_catalog": frozen,
        "minichem_endpoint": "http://127.0.0.1:1/mcp",
        "catalog_endpoint": "http://127.0.0.1:1/mcp",
        "submission_contract_checker": lambda action, session: True,
        "evidence_verifier": lambda response, **kwargs: {},
    }
    session = InvestigationSessionIdentity(
        investigation_session_id="session-catalog-independent-resume",
        investigation_context_id="context-catalog-independent-resume",
        research_task_id="task-catalog-independent-resume",
        hypothesis_execution_input_record_id="hypothesis-catalog-independent-resume",
        workspace_identity=_workspace_relpath(
            "session-catalog-independent-resume",
            "task-catalog-independent-resume",
            "context-catalog-independent-resume",
            "hypothesis-catalog-independent-resume",
        ),
        minichem_scope_id="scope-catalog-independent-resume",
        scope_contract_digest="b" * 64,
    )
    request = InvestigationHarnessRequest(
        research_task_id=session.research_task_id,
        module_invocation_id="module-catalog-independent-resume",
        module_kind="hypothesis-investigation",
        input_scientific_record_ids=(),
        module_input_digest="c" * 64,
        context_binding=CreateInvestigationContext(
            investigation_context_id=session.investigation_context_id,
            hypothesis_batch_record_id="batch-catalog-independent-resume",
        ).to_dict(),
    )
    first = InvestigationSessionHarnessOwner(
        dependencies=dependencies,
        **common,
    ).open_session(session).start(request)
    assert isinstance(first, Suspended)

    mcp.reject_discovery = True
    dependencies["native_provider_transport"] = _ResumeResultProvider()
    # 模拟恢复资料中没有可加载的冻结 Catalog；恢复原 job 仍不能去 9011 取一份
    # 当前全局快照来替代提交时事实。
    dependencies.pop("frozen_catalog")
    resumed = InvestigationSessionHarnessOwner(
        dependencies=dependencies,
        **common,
    ).open_session(session).resume(request)

    assert resumed.status == "Completed"
    assert [name for name, _payload in mcp.calls].count("list_action_domains") == 1
    assert [name for name, _payload in mcp.calls][-2:] == [
        "get_execution_job",
        "collect_execution_job",
    ]


def test_open_session_binds_scope_and_codex_to_one_owner_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """真实 scope capability 与 Codex sandbox 必须使用同一 Session workspace。"""

    monkeypatch.setenv("MINICHEM_SCOPE_REGISTRY_ROOT", str(tmp_path / "scope-registry"))
    monkeypatch.setenv(
        "MINICHEM_SCOPE_CREDENTIALS_ROOT",
        str(tmp_path / "scope-credentials"),
    )
    manager = _NativeScopeCapabilityManager(task_root=tmp_path / "bundle")
    captured: dict[str, Any] = {}

    def provider_factory(
        _session: InvestigationSessionIdentity,
        runtime: Any,
    ) -> _Provider:
        captured["runtime"] = runtime
        return _Provider()

    factory = InvestigationSessionHarnessOwner(
        task_root=tmp_path / "bundle",
        working_memory=ResearchTaskWorkingMemory(
            str(tmp_path / "memory.sqlite3")
        ),
        task_input={"scientific_question": "验证唯一 Session workspace"},
        policy={"entry_strategy": "author_hypothesis_import"},
        dependencies={
            "native_scope_manager": manager,
            "native_provider_transport_factory": provider_factory,
            "native_execution_transport": _Execution(),
            "submission_contract_checker": lambda action, session: True,
            "evidence_verifier": lambda response, **kwargs: {},
        },
    )
    session = InvestigationSessionIdentity(
        investigation_session_id="session-one-workspace",
        investigation_context_id="context-one-workspace",
        research_task_id="task-one-workspace",
        hypothesis_execution_input_record_id="hypothesis-one-workspace",
        workspace_identity=_workspace_relpath(
            "session-one-workspace",
            "task-one-workspace",
            "context-one-workspace",
            "hypothesis-one-workspace",
        ),
        minichem_scope_id="scope-one-workspace",
        scope_contract_digest="d" * 64,
    )

    factory.open_session(session)

    runtime = captured["runtime"]
    assert runtime.sandbox_spec is not None
    capability = manager.capability_for(session)
    assert capability.workspace == runtime.sandbox_spec.workspace.resolve()
