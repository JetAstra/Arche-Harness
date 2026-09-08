"""以一个 Investigation Context 编排多个可独立恢复的假设 Session。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Protocol

from chemistry_multiagent.memory import (
    HYPOTHESIS_CONFIRMATION_V2_SCHEMA_VERSION,
    ResearchTaskWorkingMemory,
    PreparedModuleInvocationOutput,
    ScientificRecord,
)

from .context_store import (
    InvestigationContextContractError,
    InvestigationContextIdentity,
    InvestigationContextIdentityConflict,
    InvestigationContextStore,
    InvestigationContextStoreError,
    InvestigationContextTransitionConflict,
    InvestigationInvocationIdentity,
    InvestigationSessionIdentity,
)
from .harness_runtime import (
    BackendCompleted,
    BackendFatalFailure,
    BackendResult,
    BackendSuspended,
    Completed,
    FatalFailure,
    InvestigationHarnessRequest,
    InvestigationHarnessRecoveryEvidenceUnavailable,
    Suspended,
)
from .recovery_contract import (
    Consistent,
    RecoveryEvidenceMissing,
    RecoveryStatus,
    Reconciled,
    TemporarilyUnavailable,
    VerifiedIntegrityConflict,
)
from .research_material_store import (
    ResearchMaterialDescriptorError,
    ResearchMaterialIntegrityConflict,
    ResearchMaterialMissing,
    ResearchMaterialPersistenceError,
    ResearchMaterialReadError,
    ResearchMaterialTargetConflict,
)


class _InvestigationSessionOwnerPort(Protocol):
    """先规划并持久化 Session 身份，再幂等准备其外部资源。"""

    def plan_session(
        self,
        context: InvestigationContextIdentity,
        selected_hypothesis: ScientificRecord,
    ) -> InvestigationSessionIdentity:
        """无外部副作用地返回稳定 Session、workspace 和 scope 身份。"""

    def ensure_session(self, session: InvestigationSessionIdentity) -> None:
        """按已持久化身份幂等创建或核对 workspace 和 MiniChem scope。"""


class _InvestigationSessionHarnessOwnerPort(Protocol):
    """按已有 Session 身份打开对应的持久 Harness。"""

    def open_session(self, session: InvestigationSessionIdentity) -> Any:
        """返回只负责该假设 Session 的 Harness facade。"""


class _ContextInputMissing(RuntimeError):
    """当前模块输入不足以确定可恢复的 Context/Session。"""


class _ContextInputConflict(RuntimeError):
    """不可变科学记录确认了互相冲突的 Context/Session 身份。"""


class InvestigationSessionTemporarilyUnavailable(RuntimeError):
    """Session 外部资源暂时无法按已持久化身份创建或核对。"""


@dataclass(frozen=True, slots=True)
class _SessionTarget:
    """一次 Invocation 对一个最终执行输入 Session 的输入投影。

    ``selected_hypothesis`` 保留旧字段名，但它现在表示 Session owner：旧路径
    是 ``SelectedHypothesisRecord``，确认后的路径是
    ``HypothesisExecutionInputRecord``。这样 resolver 不会误把取消前候选当作
    当前执行身份。
    """

    selected_hypothesis: ScientificRecord
    input_scientific_record_ids: tuple[str, ...]
    selected_candidate: ScientificRecord | Any | None = None
    execution_input: ScientificRecord | Any | None = None
    execution_decision: ScientificRecord | Any | None = None


@dataclass(frozen=True, slots=True)
class _ContextInputPlan:
    """从 Scientific Record 图确定性得到的 Context 调用计划。"""

    hypothesis_batch: ScientificRecord
    targets: tuple[_SessionTarget, ...]
    execution_decision: ScientificRecord | Any | None = None


def _canonical_json(value: Any) -> str:
    """生成稳定 JSON，供非标准输入摘要收敛使用。"""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _digest(value: Any) -> str:
    """计算稳定 SHA-256 摘要。"""

    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _unique_refs(values: Sequence[str]) -> tuple[str, ...]:
    """保留首次出现顺序并去除重复科学记录引用。"""

    return tuple(dict.fromkeys(values))


class InvestigationContextRunner:
    """一个 Context 的多 Session Harness backend。

    Controller 仍只启动一个 Investigation Module Invocation。本 backend 从
    Working Memory 的逐假设科学记录确定 Session 边界，并按冻结并发上限调用对应的
    子 Harness；最后按 target 顺序把所有 Evidence/Result pair 合并成一个原子模块
    输出。任一 Session 暂停时保留 Context Invocation 为 Open；下次恢复重放已完成
    子输出并只继续未完成 Session。
    """

    def __init__(
        self,
        *,
        context_store: InvestigationContextStore,
        working_memory: ResearchTaskWorkingMemory,
        session_owner: _InvestigationSessionOwnerPort | Any,
        session_harness_owner: _InvestigationSessionHarnessOwnerPort | Any,
        max_parallel_investigations: int = 5,
    ) -> None:
        """绑定四个明确 owner 和冻结并发配置；不创建 Session、scope 或化学作业。"""

        if not isinstance(context_store, InvestigationContextStore):
            raise TypeError("context_store 必须是 InvestigationContextStore")
        if not isinstance(working_memory, ResearchTaskWorkingMemory):
            raise TypeError(
                "working_memory 必须是 ResearchTaskWorkingMemory"
            )
        if not callable(getattr(session_owner, "plan_session", None)):
            raise TypeError("session_owner 必须提供 plan_session")
        if not callable(getattr(session_owner, "ensure_session", None)):
            raise TypeError("session_owner 必须提供 ensure_session")
        if not callable(
            getattr(session_harness_owner, "open_session", None)
        ) and not callable(session_harness_owner):
            raise TypeError(
                "session_harness_owner 必须提供 open_session"
            )
        if isinstance(max_parallel_investigations, bool) or not isinstance(
            max_parallel_investigations, int
        ):
            raise TypeError("max_parallel_investigations 必须是正整数")
        if not 1 <= max_parallel_investigations <= 5:
            raise ValueError(
                "max_parallel_investigations 必须位于 1 到 5 之间"
            )
        self._context_store = context_store
        self._working_memory = working_memory
        self._session_owner = session_owner
        self._session_harness_owner = session_harness_owner
        self._max_parallel_investigations = max_parallel_investigations

    def start(
        self,
        request: InvestigationHarnessRequest,
        harness_invocation_id: str,
    ) -> BackendResult:
        """创建或重放 Context Invocation；facade 引用不形成领域身份。"""

        if not isinstance(request, InvestigationHarnessRequest):
            raise TypeError("request 必须是 InvestigationHarnessRequest")
        if harness_invocation_id != request.harness_invocation_id:
            return BackendFatalFailure(
                "verified_integrity_conflict",
                recovery_verification=VerifiedIntegrityConflict(
                    reason_code="investigation_context_harness_invocation_id_conflict"
                ),
            )
        return self._run(request, is_resume=False)

    def resume(
        self,
        request: InvestigationHarnessRequest,
        harness_invocation_id: str,
    ) -> BackendResult:
        """从 Context Store 和各子 Harness 恢复同一 Invocation。"""

        if not isinstance(request, InvestigationHarnessRequest):
            raise TypeError("request 必须是 InvestigationHarnessRequest")
        if harness_invocation_id != request.harness_invocation_id:
            return BackendFatalFailure(
                "verified_integrity_conflict",
                recovery_verification=VerifiedIntegrityConflict(
                    reason_code="investigation_context_harness_invocation_id_conflict"
                ),
            )
        return self._run(request, is_resume=True)

    def _run(
        self,
        request: InvestigationHarnessRequest,
        *,
        is_resume: bool,
    ) -> BackendResult:
        """统一执行身份规划、Session 绑定、子 Harness 驱动和输出聚合。"""

        if not isinstance(request, InvestigationHarnessRequest):
            raise TypeError("request 必须是 InvestigationHarnessRequest")
        try:
            plan = self._build_input_plan(request)
            context = InvestigationContextIdentity(
                investigation_context_id=self._context_binding_id(request),
                research_task_id=request.research_task_id,
                hypothesis_batch_record_id=plan.hypothesis_batch.scientific_record_id,
                hypothesis_execution_decision_record_id=(
                    None
                    if plan.execution_decision is None
                    else plan.execution_decision.scientific_record_id
                ),
            )
            snapshot = self._context_store.create_or_load_context(context)
            sessions_by_hypothesis = {
                session.hypothesis_execution_input_record_id: session
                for session in snapshot.sessions
            }
            target_sessions: list[InvestigationSessionIdentity] = []
            for target in plan.targets:
                session = sessions_by_hypothesis.get(
                    target.selected_hypothesis.scientific_record_id
                )
                if session is None:
                    session = self._plan_session(
                        context, target.selected_hypothesis
                    )
                    session = self._context_store.bind_session(context, session)
                    sessions_by_hypothesis[
                        target.selected_hypothesis.scientific_record_id
                    ] = session
                self._assert_session_target(context, session, target)
                target_sessions.append(session)

            invocation = InvestigationInvocationIdentity(
                module_invocation_id=request.module_invocation_id,
                investigation_context_id=context.investigation_context_id,
                module_input_digest=self._normalized_input_digest(request),
                input_scientific_record_ids=request.input_scientific_record_ids,
            )
            self._context_store.start_invocation(context, invocation)
            for session in target_sessions:
                self._ensure_session(session)
            return self._drive_sessions(
                request,
                context,
                plan.targets,
                tuple(target_sessions),
                is_resume=is_resume,
            )
        except _ContextInputMissing as exc:
            return BackendSuspended(
                "recovery_evidence_missing",
                recovery_verification=RecoveryEvidenceMissing(
                    reason_code=str(exc)
                ),
            )
        except (
            _ContextInputConflict,
            InvestigationContextIdentityConflict,
            InvestigationContextTransitionConflict,
        ):
            return BackendFatalFailure(
                "verified_integrity_conflict",
                recovery_verification=VerifiedIntegrityConflict(
                    reason_code="investigation_context_identity_conflict"
                ),
            )
        except InvestigationContextContractError:
            raise
        except InvestigationContextStoreError:
            return BackendSuspended(
                "context_store_unavailable",
                recovery_verification=TemporarilyUnavailable(
                    reason_code="context_store_unavailable"
                ),
            )
        except InvestigationSessionTemporarilyUnavailable:
            return BackendSuspended(
                "investigation_session_unavailable",
                recovery_verification=TemporarilyUnavailable(
                    reason_code="investigation_session_unavailable"
                ),
            )
        except ResearchMaterialMissing:
            return BackendSuspended(
                "recovery_evidence_missing",
                recovery_verification=RecoveryEvidenceMissing(
                    reason_code="research_material_missing"
                ),
            )
        except (
            ResearchMaterialDescriptorError,
            ResearchMaterialIntegrityConflict,
            ResearchMaterialTargetConflict,
        ):
            return BackendFatalFailure(
                "verified_integrity_conflict",
                recovery_verification=VerifiedIntegrityConflict(
                    reason_code="research_material_integrity_conflict"
                ),
            )
        except (ResearchMaterialReadError, ResearchMaterialPersistenceError):
            return BackendSuspended(
                "investigation_session_unavailable",
                recovery_verification=TemporarilyUnavailable(
                    reason_code="research_material_store_unavailable"
                ),
            )
        except InvestigationHarnessRecoveryEvidenceUnavailable:
            return BackendSuspended(
                "session_harness_recovery_unavailable",
                recovery_verification=RecoveryEvidenceMissing(
                    reason_code="session_harness_recovery_unavailable"
                ),
            )

    def _build_input_plan(
        self,
        request: InvestigationHarnessRequest,
    ) -> _ContextInputPlan:
        """从输入记录闭包识别 Batch、逐假设记录和本轮 repair 目标。"""

        closure = self._read_record_closure(
            request.research_task_id, request.input_scientific_record_ids
        )
        batches = [
            record
            for record in closure.values()
            if record.scientific_record_type == "HypothesisBatchRecord"
        ]
        if len(batches) != 1:
            reason = (
                "hypothesis_batch_missing"
                if not batches
                else "multiple_hypothesis_batches"
            )
            exception = _ContextInputMissing if not batches else _ContextInputConflict
            raise exception(reason)

        direct_records = [
            closure[record_id] for record_id in request.input_scientific_record_ids
        ]
        repair_requests = [
            record
            for record in direct_records
            if record.scientific_record_type == "RepairRequestRecord"
        ]
        if repair_requests:
            targets, execution_decision = self._build_repair_targets(
                closure, repair_requests
            )
        else:
            execution_inputs = [
                record
                for record in direct_records
                if record.scientific_record_type == "HypothesisExecutionInputRecord"
            ]
            selected = [
                record
                for record in direct_records
                if record.scientific_record_type == "SelectedHypothesisRecord"
            ]
            if execution_inputs:
                if selected:
                    raise _ContextInputConflict(
                        "execution_input_mixed_with_selected_hypotheses"
                    )
                targets, execution_decision = self._build_execution_targets(
                    closure, execution_inputs
                )
            elif not selected:
                # schema 1 的 Selection 没有足够信息建立 1..N Session；这里不以
                # 列表位置猜测，交由显式 V2 迁移路径补齐后再恢复。
                raise _ContextInputMissing("selected_hypothesis_records_missing")
            else:
                targets = tuple(
                    _SessionTarget(
                        record,
                        (record.scientific_record_id,),
                        selected_candidate=record,
                    )
                    for record in sorted(
                        selected, key=lambda item: item.payload["rank"]
                    )
                )
                execution_decision = None
        if not targets:
            raise _ContextInputMissing("investigation_session_targets_missing")
        return _ContextInputPlan(
            batches[0], targets, execution_decision=execution_decision
        )

    @staticmethod
    def _execution_input_field_names(
        record: ScientificRecord,
    ) -> tuple[str, str]:
        """按记录 schema 选择当前或历史 Confirmation 字段。"""

        if (
            record.scientific_record_schema_version
            == HYPOTHESIS_CONFIRMATION_V2_SCHEMA_VERSION
        ):
            return (
                "source_selected_hypothesis_record_id",
                "material_upload_refs",
            )
        return "source_selected_hypothesis_ref", "material_refs"

    @staticmethod
    def _context_binding_id(request: InvestigationHarnessRequest) -> str:
        """从正式 context binding 读取 Context 身份，不从模块调用 ID 猜测。"""

        binding = request.context_binding
        if not isinstance(binding, Mapping):
            raise _ContextInputMissing("investigation_context_binding_missing")
        context_id = binding.get("investigation_context_id")
        if not isinstance(context_id, str) or not context_id:
            raise _ContextInputMissing("investigation_context_binding_missing")
        return context_id

    @staticmethod
    def _ancestor_records(
        record: ScientificRecord,
        closure: Mapping[str, ScientificRecord],
        record_type: str,
    ) -> tuple[ScientificRecord, ...]:
        """沿 source_record_refs 读取指定类型的祖先，并去除重复路径。"""

        matches: dict[str, ScientificRecord] = {}
        pending = list(record.source_scientific_record_ids)
        visited: set[str] = set()
        while pending:
            record_id = pending.pop()
            if record_id in visited:
                continue
            visited.add(record_id)
            source = closure.get(record_id)
            if source is None:
                raise _ContextInputMissing("source_record_missing")
            if source.scientific_record_type == record_type:
                matches[source.scientific_record_id] = source
            pending.extend(
                source_ref
                for source_ref in source.source_scientific_record_ids
                if source_ref not in visited
            )
        return tuple(matches.values())

    @staticmethod
    def _payload_value(
        record: ScientificRecord,
        *field_names: str,
    ) -> Any:
        """读取 execution record 的最小公开字段，兼容 contract 分支别名。"""

        payload = record.payload
        if not isinstance(payload, Mapping):
            raise _ContextInputConflict("scientific_record_payload_invalid")
        for field_name in field_names:
            if field_name in payload:
                return payload[field_name]
        return None

    def _selected_for_execution(
        self,
        execution_input: ScientificRecord,
        closure: Mapping[str, ScientificRecord],
    ) -> ScientificRecord:
        """确定执行输入唯一对应的 SelectedHypothesisRecord。"""

        direct_selected = tuple(
            closure[record_id]
            for record_id in execution_input.source_scientific_record_ids
            if record_id in closure
            and closure[record_id].scientific_record_type == "SelectedHypothesisRecord"
        )
        payload = execution_input.payload
        if not isinstance(payload, Mapping):
            raise _ContextInputConflict("scientific_record_payload_invalid")
        selected_field, _material_field = self._execution_input_field_names(
            execution_input
        )
        selected_ref = payload.get(selected_field)
        if not isinstance(selected_ref, str) or not selected_ref:
            raise _ContextInputMissing(
                "execution_input_selected_hypothesis_missing"
            )
        selected = closure.get(selected_ref)
        if selected is None:
            raise _ContextInputMissing("execution_input_selected_hypothesis_missing")
        if selected.scientific_record_type != "SelectedHypothesisRecord":
            raise _ContextInputConflict(
                "execution_input_selected_hypothesis_type_conflict"
            )
        if direct_selected != (selected,):
            raise _ContextInputConflict(
                "execution_input_selected_hypothesis_source_conflict"
            )
        return selected

    def _execution_materials(
        self,
        execution_input: ScientificRecord,
        closure: Mapping[str, ScientificRecord],
    ) -> tuple[ScientificRecord, ...]:
        """校验执行输入的材料引用与直接来源完全一致。"""

        payload = execution_input.payload
        if not isinstance(payload, Mapping):
            raise _ContextInputConflict("scientific_record_payload_invalid")
        _selected_field, material_field = self._execution_input_field_names(
            execution_input
        )
        material_refs = payload.get(material_field)
        if not isinstance(material_refs, (list, tuple)) or isinstance(
            material_refs, (str, bytes)
        ):
            raise _ContextInputConflict("execution_input_material_refs_invalid")
        if any(not isinstance(ref, str) or not ref for ref in material_refs):
            raise _ContextInputConflict("execution_input_material_refs_invalid")
        if len(set(material_refs)) != len(material_refs):
            raise _ContextInputConflict("execution_input_material_refs_duplicate")
        source_refs = execution_input.source_scientific_record_ids
        material_records_by_id: dict[str, ScientificRecord] = {}
        for record in closure.values():
            if record.scientific_record_type != "ResearchMaterialRecord":
                continue
            record_payload = record.payload
            if not isinstance(record_payload, Mapping):
                raise _ContextInputConflict("research_material_payload_invalid")
            material_id = record_payload.get("material_id")
            if not isinstance(material_id, str) or not material_id:
                raise _ContextInputConflict("research_material_id_invalid")
            if material_id in material_records_by_id:
                raise _ContextInputConflict("research_material_id_duplicate")
            material_records_by_id[material_id] = record
        materials: list[ScientificRecord] = []
        for ref in material_refs:
            material = material_records_by_id.get(ref)
            if material is None:
                raise _ContextInputMissing("execution_input_material_missing")
            materials.append(material)
        direct_material_refs = tuple(
            ref
            for ref in source_refs
            if ref in closure and closure[ref].scientific_record_type == "ResearchMaterialRecord"
        )
        expected_material_record_refs = tuple(material.scientific_record_id for material in materials)
        if (
            len(direct_material_refs) != len(expected_material_record_refs)
            or set(direct_material_refs) != set(expected_material_record_refs)
        ):
            raise _ContextInputConflict("execution_input_material_sources_conflict")
        return tuple(materials)

    def _validate_execution_record_sources(
        self,
        execution_input: ScientificRecord,
        closure: Mapping[str, ScientificRecord],
        selected: ScientificRecord,
        decision: ScientificRecord,
    ) -> tuple[ScientificRecord, ...]:
        """核对确认记录的精确 Decision、Selected、Material 直接来源。"""

        source_refs = execution_input.source_scientific_record_ids
        if any(
            ref not in closure
            for ref in source_refs
        ):
            raise _ContextInputMissing("source_record_missing")
        if any(
            closure[ref].scientific_record_type
            not in {"HypothesisExecutionDecisionRecord", "SelectedHypothesisRecord", "ResearchMaterialRecord"}
            for ref in source_refs
        ):
            raise _ContextInputConflict("execution_input_source_type_conflict")
        if sum(
            closure[ref].scientific_record_type == "HypothesisExecutionDecisionRecord"
            for ref in source_refs
        ) != 1:
            raise _ContextInputConflict("execution_input_decision_source_conflict")
        if sum(
            closure[ref].scientific_record_type == "SelectedHypothesisRecord"
            for ref in source_refs
        ) != 1:
            raise _ContextInputConflict("execution_input_selected_source_conflict")
        materials = self._execution_materials(execution_input, closure)
        expected_refs = {
            decision.scientific_record_id,
            selected.scientific_record_id,
            *(material.scientific_record_id for material in materials),
        }
        if len(source_refs) != len(set(source_refs)) or set(source_refs) != expected_refs:
            raise _ContextInputConflict("execution_input_source_refs_conflict")
        return materials

    def _validate_execution_lineage(
        self,
        selected: ScientificRecord,
        decision: ScientificRecord,
        closure: Mapping[str, ScientificRecord],
    ) -> None:
        """确认 Selected 与 Decision 都唯一回溯到同一 Batch。"""

        selected_batches = self._ancestor_records(
            selected,
            closure,
            "HypothesisBatchRecord",
        )
        decision_batches = self._ancestor_records(
            decision,
            closure,
            "HypothesisBatchRecord",
        )
        if not selected_batches or not decision_batches:
            raise _ContextInputMissing("execution_input_batch_missing")
        if len(selected_batches) != 1 or len(decision_batches) != 1:
            raise _ContextInputConflict("execution_input_has_multiple_batches")
        if selected_batches[0].scientific_record_id != decision_batches[0].scientific_record_id:
            raise _ContextInputConflict("execution_input_selected_decision_batch_conflict")

    def _build_execution_targets(
        self,
        closure: Mapping[str, ScientificRecord],
        execution_inputs: Sequence[ScientificRecord],
    ) -> tuple[tuple[_SessionTarget, ...], ScientificRecord]:
        """校验 1..5 条确认输入，并按 execution_rank 规划 Session。"""

        if not 1 <= len(execution_inputs) <= 5:
            raise _ContextInputConflict(
                "hypothesis_execution_inputs_must_contain_one_to_five"
            )
        targets: list[tuple[int, _SessionTarget, ScientificRecord]] = []
        selected_ids: set[str] = set()
        decision_ids: set[str] = set()
        for execution_input in execution_inputs:
            payload = execution_input.payload
            if not isinstance(payload, Mapping):
                raise _ContextInputConflict("scientific_record_payload_invalid")
            selected_field, material_field = self._execution_input_field_names(
                execution_input
            )
            required_fields = {
                selected_field,
                "execution_rank",
                "hypothesis",
                "guidance",
                material_field,
            }
            if set(payload) != required_fields:
                raise _ContextInputConflict("execution_input_payload_contract_conflict")
            execution_rank = payload["execution_rank"]
            if (
                isinstance(execution_rank, bool)
                or not isinstance(execution_rank, int)
                or execution_rank < 1
            ):
                raise _ContextInputConflict("execution_rank_invalid")
            hypothesis = payload["hypothesis"]
            if not isinstance(hypothesis, Mapping):
                raise _ContextInputConflict("execution_input_hypothesis_missing")
            guidance = payload["guidance"]
            if guidance is not None and (
                not isinstance(guidance, str) or not guidance.strip()
            ):
                raise _ContextInputConflict("execution_input_guidance_invalid")
            selected = self._selected_for_execution(execution_input, closure)
            selected_ids.add(selected.scientific_record_id)
            decisions = self._ancestor_records(
                execution_input,
                closure,
                "HypothesisExecutionDecisionRecord",
            )
            if len(decisions) != 1:
                raise (
                    _ContextInputMissing("execution_input_decision_missing")
                    if not decisions
                    else _ContextInputConflict(
                        "execution_input_has_multiple_decisions"
                    )
                )
            decision = decisions[0]
            self._validate_execution_record_sources(
                execution_input,
                closure,
                selected,
                decision,
            )
            self._validate_execution_lineage(selected, decision, closure)
            decision_ids.add(decision.scientific_record_id)
            targets.append(
                (
                    execution_rank,
                    _SessionTarget(
                        execution_input,
                        (execution_input.scientific_record_id,),
                        selected_candidate=selected,
                        execution_input=execution_input,
                        execution_decision=decision,
                    ),
                    decision,
                )
            )
        if len(selected_ids) != len(execution_inputs):
            raise _ContextInputConflict("execution_inputs_share_selected_hypothesis")
        if len(decision_ids) != 1:
            raise _ContextInputConflict("execution_inputs_use_multiple_decisions")
        ranks = sorted(rank for rank, _target, _decision in targets)
        if ranks != list(range(1, len(targets) + 1)):
            raise _ContextInputConflict("execution_ranks_must_be_continuous")
        ordered = tuple(
            target for _rank, target, _decision in sorted(targets, key=lambda item: item[0])
        )
        decisions_by_id = {
            decision.scientific_record_id: decision
            for _rank, _target, decision in targets
        }
        return ordered, next(iter(decisions_by_id.values()))

    def _build_repair_targets(
        self,
        closure: Mapping[str, ScientificRecord],
        repair_requests: Sequence[ScientificRecord],
    ) -> tuple[tuple[_SessionTarget, ...], ScientificRecord | None]:
        """把 Repair Request 映射回原 Selected 或确认后的执行输入 Session。"""

        grouped: dict[
            str,
            tuple[_SessionTarget, list[str], int, ScientificRecord | None],
        ] = {}
        for repair in repair_requests:
            result_ref = repair.payload["result_ref"]
            result = closure.get(result_ref)
            if result is None or result.scientific_record_type != "InvestigationResultRecord":
                raise _ContextInputMissing("repair_result_missing")
            execution_inputs = self._ancestor_records(
                result, closure, "HypothesisExecutionInputRecord"
            )
            if len(execution_inputs) > 1:
                raise _ContextInputConflict("repair_result_has_multiple_execution_inputs")
            if execution_inputs:
                execution_input = execution_inputs[0]
                selected_record = self._selected_for_execution(execution_input, closure)
                decisions = self._ancestor_records(
                    execution_input,
                    closure,
                    "HypothesisExecutionDecisionRecord",
                )
                if len(decisions) != 1:
                    raise _ContextInputMissing("repair_execution_decision_missing")
                execution_decision = decisions[0]
                self._validate_execution_record_sources(
                    execution_input,
                    closure,
                    selected_record,
                    execution_decision,
                )
                self._validate_execution_lineage(
                    selected_record,
                    execution_decision,
                    closure,
                )
                execution_rank = self._payload_value(execution_input, "execution_rank")
                if (
                    isinstance(execution_rank, bool)
                    or not isinstance(execution_rank, int)
                    or execution_rank < 1
                ):
                    raise _ContextInputConflict("repair_execution_rank_invalid")
                target = _SessionTarget(
                    execution_input,
                    (execution_input.scientific_record_id, repair.scientific_record_id, result.scientific_record_id),
                    selected_candidate=selected_record,
                    execution_input=execution_input,
                    execution_decision=execution_decision,
                )
                owner = execution_input
                rank = execution_rank
                refs = [execution_input.scientific_record_id, repair.scientific_record_id, result.scientific_record_id]
            else:
                selected = self._selected_ancestors(result, closure)
                if not selected:
                    raise _ContextInputMissing("repair_selected_hypothesis_missing")
                if len(selected) != 1:
                    raise _ContextInputConflict(
                        "repair_result_has_multiple_selected_hypotheses"
                    )
                selected_record = selected[0]
                target = _SessionTarget(
                    selected_record,
                    (selected_record.scientific_record_id, repair.scientific_record_id, result.scientific_record_id),
                    selected_candidate=selected_record,
                )
                owner = selected_record
                rank = selected_record.payload["rank"]
                execution_decision = None
                refs = [selected_record.scientific_record_id, repair.scientific_record_id, result.scientific_record_id]
            existing = grouped.get(owner.scientific_record_id)
            if existing is None:
                grouped[owner.scientific_record_id] = (target, refs, rank, execution_decision)
            else:
                old_target, old_refs, old_rank, old_decision = existing
                if old_decision is not None and execution_decision is not None and (
                    old_decision.scientific_record_id != execution_decision.scientific_record_id
                ):
                    raise _ContextInputConflict("repair_uses_multiple_decisions")
                if (old_decision is None) != (execution_decision is None):
                    raise _ContextInputConflict("repair_mixes_execution_input_paths")
                if old_rank != rank:
                    raise _ContextInputConflict("repair_execution_rank_conflict")
                old_refs.extend(refs[1:])
                grouped[owner.scientific_record_id] = (
                    old_target,
                    old_refs,
                    old_rank,
                    old_decision,
                )
        values = tuple(grouped.values())
        execution_decisions = {
            decision.scientific_record_id: decision
            for _target, _refs, _rank, decision in values
            if decision is not None
        }
        if len(execution_decisions) > 1:
            # Different decision records cannot share one Context invocation.
            raise _ContextInputConflict("repair_uses_multiple_decisions")
        if execution_decisions and len(values) != len(
            [item for item in values if item[3] is not None]
        ):
            raise _ContextInputConflict("repair_mixes_execution_input_paths")
        ordered = tuple(
            _SessionTarget(target.selected_hypothesis, _unique_refs(refs),
                           selected_candidate=target.selected_candidate,
                           execution_input=target.execution_input,
                           execution_decision=target.execution_decision)
            for target, refs, _rank, _decision in sorted(
                values, key=lambda item: item[2]
            )
        )
        return ordered, next(iter(execution_decisions.values()), None)

    def _read_record_closure(
        self,
        research_task_id: str,
        record_ids: Sequence[str],
    ) -> dict[str, ScientificRecord]:
        """从 Working Memory 读取输入及其全部科学来源记录。"""

        records: dict[str, ScientificRecord] = {}
        pending = list(record_ids)
        while pending:
            record_id = pending.pop()
            if record_id in records:
                continue
            try:
                values = self._working_memory.read_research_task_scientific_records(
                    research_task_id, (record_id,)
                )
            except Exception as exc:
                raise _ContextInputMissing("scientific_record_missing") from exc
            if len(values) != 1:
                raise _ContextInputMissing("scientific_record_missing")
            record = values[0]
            if (
                record.research_task_id != research_task_id
                or record.scientific_record_id != record_id
            ):
                raise _ContextInputConflict("scientific_record_identity_conflict")
            records[record_id] = record
            pending.extend(
                source_ref
                for source_ref in record.source_scientific_record_ids
                if source_ref not in records
            )
        return records

    @staticmethod
    def _selected_ancestors(
        record: ScientificRecord,
        closure: Mapping[str, ScientificRecord],
    ) -> tuple[ScientificRecord, ...]:
        """读取一个 Result 的全部逐假设祖先并按排名排序。"""

        selected: dict[str, ScientificRecord] = {}
        pending = list(record.source_scientific_record_ids)
        visited: set[str] = set()
        while pending:
            record_id = pending.pop()
            if record_id in visited:
                continue
            visited.add(record_id)
            source = closure.get(record_id)
            if source is None:
                raise _ContextInputMissing("result_source_record_missing")
            if source.scientific_record_type == "SelectedHypothesisRecord":
                selected[source.scientific_record_id] = source
            pending.extend(source.source_scientific_record_ids)
        return tuple(
            sorted(selected.values(), key=lambda item: item.payload["rank"])
        )

    def _plan_session(
        self,
        context: InvestigationContextIdentity,
        selected_hypothesis: ScientificRecord,
    ) -> InvestigationSessionIdentity:
        """无副作用规划 Session；只在 Context Store 尚无绑定时发生。"""

        raw = self._session_owner.plan_session(
            context, selected_hypothesis
        )
        if not isinstance(raw, InvestigationSessionIdentity):
            raise InvestigationContextContractError(
                "plan_session 必须返回 InvestigationSessionIdentity"
            )
        return raw

    def _ensure_session(self, session: InvestigationSessionIdentity) -> None:
        """只在 Session 与 Invocation 已持久化后准备外部资源。"""

        result = self._session_owner.ensure_session(session)
        if result is not None:
            raise InvestigationContextContractError(
                "ensure_session 必须只核对副作用并返回 None"
            )

    @staticmethod
    def _assert_session_target(
        context: InvestigationContextIdentity,
        session: InvestigationSessionIdentity,
        target: _SessionTarget,
    ) -> None:
        """核对持久 Session 与本轮逐假设输入没有漂移。"""

        if (
            session.investigation_context_id != context.investigation_context_id
            or session.research_task_id != context.research_task_id
            or session.hypothesis_execution_input_record_id
            != target.selected_hypothesis.scientific_record_id
        ):
            raise _ContextInputConflict("session_target_identity_conflict")

    @staticmethod
    def _normalized_input_digest(request: InvestigationHarnessRequest) -> str:
        """Context Store 只接受 SHA-256；非标准摘要绑定完整公开请求。"""

        value = request.module_input_digest
        if len(value) == 64 and all(char in "0123456789abcdef" for char in value):
            return value
        return _digest(
            {
                "declared_input_digest": value,
                "module_kind": request.module_kind,
                "input_scientific_record_ids": list(
                    request.input_scientific_record_ids
                ),
            }
        )

    def _open_session_harness(
        self, session: InvestigationSessionIdentity
    ) -> Any:
        """按持久 Session 身份打开子 Harness，不读取展示状态。"""

        method = getattr(self._session_harness_owner, "open_session", None)
        harness = (
            method(session)
            if callable(method)
            else self._session_harness_owner(session)
        )
        if not callable(getattr(harness, "start", None)):
            raise InvestigationContextContractError(
                "session_harness_owner 必须返回带 start 的 Harness"
            )
        return harness

    def _drive_sessions(
        self,
        request: InvestigationHarnessRequest,
        context: InvestigationContextIdentity,
        targets: Sequence[_SessionTarget],
        sessions: Sequence[InvestigationSessionIdentity],
        *,
        is_resume: bool,
    ) -> BackendResult:
        """按稳定目标顺序驱动子 Harness，并在全部完成后原子聚合输出。"""

        planned_children = tuple(
            (
                target,
                session,
                InvestigationHarnessRequest(
                    research_task_id=request.research_task_id,
                    module_invocation_id=request.module_invocation_id,
                    module_kind=request.module_kind,
                    input_scientific_record_ids=target.input_scientific_record_ids,
                    module_input_digest=request.module_input_digest,
                    context_binding={
                        "kind": "InvestigationSession",
                        "investigation_context_id": context.investigation_context_id,
                        "investigation_session_id": session.investigation_session_id,
                        "hypothesis_execution_input_record_id": (
                            session.hypothesis_execution_input_record_id
                        ),
                    },
                ),
                self._open_session_harness(session),
            )
            for target, session in zip(targets, sessions, strict=True)
        )
        outcomes = self._drive_planned_sessions(
            planned_children,
            is_resume=is_resume,
        )
        output_records: list[ScientificRecord] = []
        reconciled = False
        # Future 结果按 planned_children 的稳定顺序读取，而不是按完成先后
        # 读取；因此多个失败时总是选择 target 顺序最靠前的那个。
        for (target, _session, _child_request, _child), outcome in zip(
            planned_children,
            outcomes,
            strict=True,
        ):
            if isinstance(outcome, Suspended):
                return BackendSuspended(
                    outcome.reason_code,
                    recovery_verification=outcome.recovery_verification,
                )
            if isinstance(outcome, FatalFailure):
                return BackendFatalFailure(
                    outcome.failure_code,
                    recovery_verification=outcome.recovery_verification,
                )
            if not isinstance(outcome, Completed):
                raise InvestigationContextContractError(
                    "子 Harness 返回了未注册结果"
                )
            self._validate_child_output(
                request, target, outcome.prepared_output
            )
            output_records.extend(outcome.prepared_output.output_scientific_records)
            reconciled = reconciled or (
                outcome.recovery_verification.status
                is RecoveryStatus.RECONCILED
            )

        prepared = PreparedModuleInvocationOutput.from_records(
            module_invocation_id=request.module_invocation_id,
            module_input_digest=request.module_input_digest,
            output_scientific_records=tuple(output_records),
        )
        existing_digest = self._context_store.read_completed_output_digest(
            context, request.module_invocation_id
        )
        if (
            existing_digest is not None
            and existing_digest != prepared.module_output_digest
        ):
            return BackendFatalFailure(
                "verified_integrity_conflict",
                recovery_verification=VerifiedIntegrityConflict(
                    reason_code="investigation_context_output_conflict"
                ),
            )
        self._context_store.complete_invocation(
            context,
            request.module_invocation_id,
            module_output_digest=prepared.module_output_digest,
        )
        verification = (
            Reconciled(
                reason_code="investigation_sessions_replayed",
                reconciled_fields=("investigation_sessions",),
            )
            if reconciled or existing_digest is not None
            else Consistent(reason_code="investigation_context_output_ready")
        )
        return BackendCompleted(
            prepared,
            recovery_verification=verification,
        )

    def _drive_planned_sessions(
        self,
        planned_children: Sequence[
            tuple[
                _SessionTarget,
                InvestigationSessionIdentity,
                InvestigationHarnessRequest,
                Any,
            ]
        ],
        *,
        is_resume: bool,
    ) -> tuple[Completed | Suspended | FatalFailure, ...]:
        """按配置上限并发驱动已规划的 Session，并按 target 顺序返回结果。"""

        if not planned_children:
            return ()
        worker_count = min(
            self._max_parallel_investigations,
            len(planned_children),
        )
        if worker_count == 1:
            return tuple(
                self._drive_one_planned_session(item, is_resume=is_resume)
                for item in planned_children
            )
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = tuple(
                executor.submit(
                    self._drive_one_planned_session,
                    item,
                    is_resume=is_resume,
                )
                for item in planned_children
            )
            return tuple(future.result() for future in futures)

    @staticmethod
    def _drive_one_planned_session(
        planned: tuple[
            _SessionTarget,
            InvestigationSessionIdentity,
            InvestigationHarnessRequest,
            Any,
        ],
        *,
        is_resume: bool,
    ) -> Completed | Suspended | FatalFailure:
        """驱动一个预先绑定身份的 Session；不改变其 Context 或 Session 身份。"""

        _target, _session, child_request, child = planned
        has_recovery = getattr(child, "has_recovery", None)
        should_resume = (
            is_resume
            and callable(getattr(child, "resume", None))
            and callable(has_recovery)
            and bool(has_recovery(child_request))
        )
        if should_resume:
            resume_method = getattr(child, "resume", None)
            # InvestigationHarness.resume 的正式入口同时接收稳定的
            # harness_invocation_id 与公开 request；窄 fake 则可以只接收 request。
            run_ref_for = getattr(child, "harness_invocation_id_for", None)
            if callable(run_ref_for):
                return resume_method(
                    run_ref_for(child_request), child_request
                )
            return resume_method(child_request)
        # 没有独立恢复事实时该 Session 可能尚未轮到；start 会在 Harness
        # 内部根据 Recovery Store 选择首次执行或幂等重放。
        return child.start(child_request)

    @staticmethod
    def _validate_child_output(
        parent_request: InvestigationHarnessRequest,
        target: _SessionTarget,
        output: PreparedModuleInvocationOutput,
    ) -> None:
        """核对一个 Session 只返回一对且明确引用本轮假设与 repair 输入。"""

        output_mapping = output.to_dict()
        if (
            output_mapping["module_invocation_id"]
            != parent_request.module_invocation_id
            or output_mapping["module_input_digest"]
            != parent_request.module_input_digest
        ):
            raise _ContextInputConflict("session_output_identity_conflict")
        evidence = [
            record
            for record in output.output_scientific_records
            if record.scientific_record_type == "VerifiedEvidenceRecord"
        ]
        results = [
            record
            for record in output.output_scientific_records
            if record.scientific_record_type == "InvestigationResultRecord"
        ]
        if len(evidence) != 1 or len(results) != 1 or len(output.output_scientific_records) != 2:
            raise _ContextInputConflict("session_output_pair_invalid")
        for record in output.output_scientific_records:
            if (
                record.research_task_id != parent_request.research_task_id
                or record.producer_module_invocation_id != parent_request.module_invocation_id
            ):
                raise _ContextInputConflict("session_output_record_identity_conflict")
        evidence_record = evidence[0]
        result_record = results[0]
        required_sources = set(target.input_scientific_record_ids)
        if not required_sources.issubset(evidence_record.source_scientific_record_ids):
            raise _ContextInputConflict("session_evidence_source_conflict")
        if not required_sources.issubset(result_record.source_scientific_record_ids):
            raise _ContextInputConflict("session_result_source_conflict")
        if (
            result_record.payload["evidence_ref"] != evidence_record.scientific_record_id
            or evidence_record.scientific_record_id not in result_record.source_scientific_record_ids
        ):
            raise _ContextInputConflict("session_evidence_result_conflict")


__all__ = [
    "InvestigationContextRunner",
    "InvestigationSessionTemporarilyUnavailable",
]
