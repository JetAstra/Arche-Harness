"""向 Controller 提供 Working Memory 读写接口。

数据库事务、重复提交判断和状态推进都封装在本文件中。Agent 不直接操作表，
这样一次业务动作要么完整写入，要么完全不生效。
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .contracts import (
    BATCH_CLOSABLE_COMBINATION_STATES,
    BATCH_TECHNICAL_PAUSE_STATE,
    CAUSAL_SOURCE_CONSUMER_EVENT_TYPES,
    CASConflict,
    ContractValidationError,
    IdempotencyConflict,
    LEGACY_TERMINAL_TASK_STATES,
    MemoryDisabledError,
    OperationConflict,
    OPERATION_TERMINAL_EVENT_TYPES,
    canonical_json,
    empty_runtime_checkpoint,
    event_key_attempt,
    event_key_batch_closed,
    event_key_batch_created,
    event_key_checkpoint,
    event_key_execution_step_completed,
    event_key_execution_step_failed,
    event_key_execution_request_rebound,
    event_key_expert_debate,
    event_key_expert_gaussian_review,
    event_key_expert_pause,
    event_key_retrieval_completed,
    event_key_pause,
    event_key_planning,
    event_key_tool_capability_gap,
    event_key_resume,
    event_key_restart,
    event_key_task_closed,
    event_key_task_created,
    event_key_investigation_batch_review,
    event_key_investigation_repair_completed,
    event_key_investigation_repair_requested,
    event_key_investigation_result,
    event_key_investigation_route_marker,
    event_key_investigation_session,
    event_key_investigation_session_thread_bound,
    event_key_investigation_session_thread_migrated,
    event_key_investigation_turn_started,
    event_key_investigation_turn_waiting,
    event_key_investigation_turn_attempt_termination,
    event_key_investigation_legacy_waiting_resume_authorized,
    event_key_investigation_incomplete_v2_waiting_resume_authorized,
    event_key_investigation_task_terminal,
    event_key_investigation_task_correction,
    event_key_investigation_task_failure,
    event_key_hypothesis_selection_completed,
    event_key_investigation_evidence_verification,
    event_key_task_review_session,
    event_key_task_review_turn,
    event_key_investigation_repair_turn_started,
    investigation_session_id,
    INVESTIGATION_EVENT_TYPES,
    INVESTIGATION_PROTOCOL_VERSION,
    INVESTIGATION_STAGE_ARCHITECTURE_VERSION,
    LEGACY_STAGE_ARCHITECTURE_VERSION,
    model_revision_uuid,
    require_nonempty_string,
    sha256_from_manifest_name,
    validate_attempt_payload,
    validate_batch_closed_payload,
    validate_batch_created_payload,
    validate_checkpoint_payload,
    validate_execution_step_completed_payload,
    validate_execution_step_failed_payload,
    validate_execution_request_rebound_payload,
    validate_expert_debate_payload,
    validate_expert_gaussian_review_payload,
    validate_expert_pause_payload,
    validate_pause_payload,
    validate_retrieval_pause_payload,
    validate_retrieval_completed_payload,
    validate_planning_payload,
    validate_tool_capability_gap_payload,
    validate_resume_payload,
    validate_revision_payload,
    validate_planning_source_refs,
    validate_source_ref_shape,
    validate_structured_hypothesis_claims,
    validate_task_relative_path,
    validate_task_created_payload,
    get_investigation_combination_count,
    validate_investigation_batch_review_payload,
    validate_investigation_repair_completed_payload,
    validate_investigation_repair_requested_payload,
    validate_investigation_result_payload,
    validate_investigation_result_version_vector,
    validate_investigation_route_marker_payload,
    validate_investigation_session_payload,
    validate_investigation_session_thread_bound_payload,
    validate_investigation_session_thread_migrated_payload,
    validate_investigation_turn_started_payload,
    validate_investigation_turn_waiting_payload,
    validate_investigation_turn_attempt_termination_payload,
    validate_investigation_legacy_waiting_resume_authorization_payload,
    validate_investigation_incomplete_v2_waiting_resume_authorization_payload,
    validate_investigation_task_terminal_payload,
    validate_investigation_task_correction_payload,
    validate_hypothesis_selection_completed_payload,
    validate_investigation_task_failure_payload,
    validate_investigation_evidence_payload,
    validate_task_review_session_payload,
    validate_task_review_turn_started_payload,
    validate_investigation_repair_turn_started_payload,
    validate_stage_architecture_version,
)

from .models import (
    VERIFICATION_PENDING,
    new_id,
    utc_now,
)
from .projections import (
    _ledger_fact_count,
    _gaussian_code_receipt_projection,
    _merge_gaussian_review_projection,
    batch_row,
    candidate_row,
    combination_row,
    insert_batch_projection,
    list_projection_rows,
    project_hypothesis_regeneration_context,
    project_hypothesis_terminal_outcome,
    project_judge_fact,
    project_program_gate_fact,
    project_scientific_attempt_ledger,
    project_workflow_for_revision,
    project_workflow_revision_repair,
    rebuild_task_projections,
    clear_investigation_projection_rows,
    insert_investigation_review_projection,
    insert_investigation_repair_completed_projection,
    insert_investigation_repair_requested_projection,
    insert_investigation_result_projection,
    insert_investigation_route_marker_projection,
    insert_investigation_session_projection,
    insert_investigation_session_thread_bound_projection,
    insert_investigation_session_thread_migrated_projection,
    insert_investigation_turn_started_projection,
    insert_investigation_turn_waiting_projection,
    insert_investigation_turn_attempt_termination_projection,
    insert_investigation_task_terminal_projection,
    insert_investigation_task_correction_projection,
    insert_investigation_evidence_verification_projection,
    insert_task_review_session_projection,
    insert_task_review_turn_projection,
    insert_investigation_repair_turn_started_projection,
    evaluate_investigation_batch_route_gate,
    investigation_repair_row,
    investigation_result_row,
    investigation_review_row,
    investigation_route_marker_row,
    investigation_session_row,
    investigation_evidence_row,
    task_review_session_row,
    task_review_turn_row,
    investigation_repair_turn_row,
    investigation_turn_row,
    investigation_turn_wait_row,
    investigation_task_terminal_row,
    task_row,
)
from .investigation_index import read_investigation_index_snapshot
from .investigation_writer import (
    VerifiedInvestigationEvidence,
    _InvestigationFactWriter,
)
from .sqlite_store import SQLiteMemoryStore
from chemistry_multiagent.utils.scientific_input import (
    merge_supplementary_chemistry_context,
)
from chemistry_multiagent.utils.gaussian_routing import GaussianRoutePolicy
from chemistry_multiagent.utils.workflow_contract import (
    project_minichem_stage_tool_projection,
    project_workflow_refs,
)
from chemistry_multiagent.utils.task_artifacts import write_content_addressed_json
from chemistry_multiagent.utils.router_evidence import (
    validate_program_gate_for_evidence_ready,
)


_EXECUTION_MANIFEST_FIELDS = {
    "schema_version",
    "execution_id",
    "workflow_version",
    "step_manifest_refs",
    "execution_result_ref",
}
_MINICHEM_EXECUTION_AUDIT_FIELDS = {
    "catalog_hash",
    "catalog_version",
    "execution_layer_ids",
    "action_ids",
    "backend_ids",
    "software_ids",
    "runtimes",
    "operation_ids",
    "job_ids",
    "action_backend_identities",
    "operation_job_identities",
    "artifact_refs",
    "provenance",
}


def _is_supported_execution_manifest(value: Mapping[str, Any]) -> bool:
    """识别基础清单及当前/上一版 MiniChem 审计清单。

    执行失败不会生成另一种 execution manifest；失败步骤和后续
    ``not_executed`` 状态保存在同一份 ``execution_result`` 中。因此成功和失败的
    全工作流都复用这里的字段合同。``catalog_version`` 是后续加入的可选审计字段，
    读取上一版已落盘清单时仍允许它缺失；其他未知字段继续拒绝。
    """

    if not isinstance(value, Mapping):
        return False
    fields = set(value)
    if not _EXECUTION_MANIFEST_FIELDS <= fields:
        return False
    extra = fields - _EXECUTION_MANIFEST_FIELDS
    full_audit = _MINICHEM_EXECUTION_AUDIT_FIELDS
    previous_audit = full_audit - {"catalog_version"}
    return extra in (set(), previous_audit, full_audit)


# ==================== Working Memory 持久服务 ====================


@dataclass(frozen=True)
class CheckpointResumeContext:
    """保存一次 checkpoint 恢复读事务核对出的不可变快照。

    ``manifest`` 是已经通过数据库引用和内容摘要复核的 checkpoint 清单，
    ``record_ids`` 是当前 operation 在 execution 全局检查点链中的连续后缀。
    其余字段标记组合版本和 checkpoint 身份，恢复器可以在调用边界发现上下文
    被误传；该对象本身不执行工具调用，也不写入 Working Memory。
    """

    manifest: Mapping[str, Any]
    record_ids: Tuple[str, ...]
    combination_id: str
    combination_version: int
    checkpoint_id: str


class WorkingMemoryService:
    """保存不可修改的历史事实，并同步维护便于运行的当前状态。

    历史事件用于审计和恢复；任务、批次、组合等表只是可重建的当前视图。每次
    写入都会比较版本和运行身份，防止两个执行者同时推进同一个组合。
    """

    # 模型类暂停恢复后应回到原来的模型阶段。这里显式映射而不从字符串拼接状态，
    # 可以避免把一次规划失败错误地恢复到工具执行阶段。
    _MODEL_OPERATION_STATES = {
        "workflow_generation": "generating_workflow",
        "workflow_optimization": "generating_workflow",
        "router": "routing",
        "workflow_revision": "revising_workflow",
        "expert_gaussian_review": "running",
        "expert_debate": "routing",
    }
    # 恢复说明中的动作顺序固定为“最小重试”到“放弃当前假设”。稳定顺序既方便
    # 研究员阅读，也防止集合遍历顺序让同一暂停生成不同提示。
    _RESTART_ACTION_ORDER = (
        "retry_failed_operation",
        "retry_current_step",
        "restart_workflow",
        "revise_hypothesis",
    )
    # 只有这些通常不会改变科学输入的运行环境故障，才可能安全地重试当前步骤；
    # 工具目录还必须同时明确声明该工具支持安全重试。
    _RETRY_CURRENT_STEP_ERRORS = {
        "transient_external",
        "resource_enospc",
        "resource_oom",
        "process_killed",
    }
    # 进程去向不明时不能复用旧执行前缀，但仍可保留科学工作流并从第一步重新运行。
    _RESTART_WORKFLOW_ERRORS = _RETRY_CURRENT_STEP_ERRORS | {
        # 冻结目录身份漂移意味着当前 MCP 服务会话不能再安全读取工具；
        # 重启时必须使用隔离且重新冻结的服务目录，不能在原会话上原地重试。
        "catalog_freeze_drift",
        "process_disappeared",
        "unknown",
    }
    # 终态任务只能接受完全相同的网络重放，不能再追加新的科学或恢复事实。
    _TASK_TERMINAL_STATES = {
        "succeeded",
        "failed",
        "stopped_no_supported_hypothesis",
    } | LEGACY_TERMINAL_TASK_STATES

    # ==================== 初始化与基础校验 ====================

    def __init__(
        self,
        db_path: str,
        *,
        enabled: bool,
        scratch_root: str,
    ) -> None:
        """创建 Working Memory 服务并准备暂停恢复所需的数据库连接。

        当前 MiniChem 工作流只接受重试、重新开始和放弃假设等恢复动作；旧版
        ToolSpec 的工作流补丁不再是运行时协议。历史数据库中的旧修订仍由只读
        投影视图识别，但不会再由本服务创建新的补丁事实。
        """

        self.enabled = bool(enabled)
        self.scratch_root = os.path.abspath(os.path.expanduser(scratch_root))
        self.task_id: Optional[str] = None
        self.memory_status = "disabled" if not self.enabled else "available"
        self.db_path = os.path.abspath(os.path.expanduser(db_path))
        self.store: Optional[SQLiteMemoryStore] = None
        if self.enabled:
            self.store = SQLiteMemoryStore(self.db_path)

    def status(self) -> Dict[str, Any]:
        """告诉调用方当前是否启用记忆，而不暴露连接或表结构。"""

        return {"memory_status": self.memory_status, "task_id": self.task_id}

    def _require_enabled(self) -> SQLiteMemoryStore:
        """无记忆基线一旦尝试写库就立即失败，避免实验条件被悄悄改变。"""

        if not self.enabled or self.store is None:
            raise MemoryDisabledError("memory 已关闭，禁止创建 task 或写入持久化事实")
        return self.store

    @staticmethod
    def _decode_json(value: Any, *, field: str) -> Any:
        """读取本服务保存的JSON；内容损坏时停止恢复，不用默认值掩盖。"""

        try:
            return json.loads(value)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ContractValidationError(f"数据库中的 {field} 已损坏") from exc

    def _existing_event_exact(
        self,
        conn,
        *,
        event_key: str,
        task_id: str,
        round_id: int,
        event_type: str,
        payload: Mapping[str, Any],
        source_refs: Sequence[str],
    ) -> Optional[str]:
        """用稳定键识别重复提交。

        已有内容完全相同时返回原记录；稳定键相同但内容不同，说明调用方正在
        改写历史，必须报告冲突。
        """

        row = conn.execute("SELECT * FROM memory_events WHERE event_key=?", (event_key,)).fetchone()
        if row is None:
            return None
        stored_payload = self._decode_json(row["payload_json"], field="payload_json")
        if event_type == "task_created":
            stored_payload = validate_task_created_payload(stored_payload)
        elif event_type == "investigation_turn_waiting":
            # 读取 v9 旧 waiting 事件时把 operation_id 适配为 v10
            # submission_operation_id；数据库中的旧正文不被覆盖。
            stored_payload = validate_investigation_turn_waiting_payload(stored_payload)
        elif event_type == "investigation_incomplete_v2_waiting_resume_authorized":
            stored_payload = (
                validate_investigation_incomplete_v2_waiting_resume_authorization_payload(
                    stored_payload
                )
            )
        same = (
            row["task_id"] == task_id
            and row["round"] == round_id
            and row["event_type"] == event_type
            and canonical_json(stored_payload)
            == canonical_json(payload)
            and canonical_json(self._decode_json(row["source_refs_json"], field="source_refs_json"))
            == canonical_json(list(source_refs))
        )
        if not same:
            raise IdempotencyConflict(f"event_key={event_key} 已绑定到不同内容")
        return row["event_id"]

    def _task_created_payload(self, conn, task_id: str) -> Dict[str, Any]:
        """从任务创建事实读取预算，而不是相信后来可变的缓存列。"""

        row = conn.execute(
            "SELECT event_type, payload_json FROM memory_events WHERE event_key=? AND task_id=?",
            (event_key_task_created(task_id), task_id),
        ).fetchone()
        if row is None or row["event_type"] != "task_created":
            raise ContractValidationError("task 缺少唯一 TaskCreated authority")
        return validate_task_created_payload(
            self._decode_json(row["payload_json"], field="TaskCreated.payload_json")
        )

    def _ensure_task_projection_matches_created(
        self,
        conn,
        task_id: str,
        task_created: Mapping[str, Any],
    ) -> Any:
        """核对并在必要时重建任务当前行，返回与 TaskCreated 对齐的行。

        输入是事务连接、task 身份和已验证的不可变 TaskCreated 正文；动作比较问题、
        化学上下文、反应类别和架构枚举，发现当前行被外部改写时按事件历史重建全部
        可重建当前视图。返回修复后的任务行；缺少任务、TaskCreated 或重建所需事实
        时抛出合同错误，不读取环境配置，也不改变不可变事件。
        """

        row = conn.execute(
            "SELECT * FROM tasks WHERE task_id=?", (task_id,)
        ).fetchone()
        if row is None:
            raise ContractValidationError(f"未知 task_id：{task_id}")
        chemistry_context = task_created["chemistry_context"]
        reaction_class = chemistry_context.get("reaction_type") or chemistry_context.get(
            "reaction_class"
        )
        matches = (
            row["scientific_question"] == task_created["scientific_question"]
            and self._decode_json(
                row["chemistry_context_json"], field="tasks.chemistry_context_json"
            )
            == chemistry_context
            and row["reaction_class"] == reaction_class
            and row["stage_architecture_version"]
            == task_created["stage_architecture_version"]
        )
        if matches:
            return row
        rebuild_task_projections(
            conn,
            task_id,
            task_root=self._task_output_root(conn, task_id),
        )
        repaired = conn.execute(
            "SELECT * FROM tasks WHERE task_id=?", (task_id,)
        ).fetchone()
        if repaired is None:
            raise ContractValidationError("TaskCreated 重建后缺少 tasks 当前行")
        if (
            repaired["scientific_question"] != task_created["scientific_question"]
            or self._decode_json(
                repaired["chemistry_context_json"],
                field="tasks.chemistry_context_json",
            )
            != chemistry_context
            or repaired["stage_architecture_version"]
            != task_created["stage_architecture_version"]
        ):
            raise ContractValidationError("tasks 当前行无法由 TaskCreated 重建")
        return repaired

    def _validated_batch_created_payload(
        self,
        conn,
        task_id: str,
        payload_json: Any,
        *,
        field: str,
    ) -> Dict[str, Any]:
        """读取批次事实时重新验证候选映射和冻结的科学正文。

        创建时校验通过并不代表磁盘内容永远可信。恢复、Router 和最终发布每次
        读取同一事件都重新执行固定契约，数据库被改写或旧进程保存了漂移文本时
        会立即停止，而不是把不一致假设继续传给模型。
        """

        task_created = self._task_created_payload(conn, task_id)
        self._ensure_task_projection_matches_created(conn, task_id, task_created)
        strict_v2 = (
            task_created["stage_architecture_version"]
            == INVESTIGATION_STAGE_ARCHITECTURE_VERSION
        )
        task_config = task_created["task_config"]
        top_n_hypotheses = task_config.get(
            "top_n_workflows", task_config.get("top_n_hypotheses")
        )
        expected_combination_count = (
            get_investigation_combination_count(task_config)
            if strict_v2
            else None
        )
        return validate_batch_created_payload(
            self._decode_json(payload_json, field=field),
            top_n_workflows=top_n_hypotheses,
            # 数据库中的既有事件可能采用旧版多主张正文；这里只读兼容，绝不把
            # 兼容字段复制到新的 BatchCreated 事件或活动模型输入。
            allow_legacy_hypothesis=True,
            required_selected_count=expected_combination_count,
        )

    def _effective_scientific_context(self, conn, task_id: str) -> Dict[str, Any]:
        """联接任务事实与首批 Retrieval，得到后续模块唯一可见的科学上下文。"""

        created = self._task_created_payload(conn, task_id)
        batch = conn.execute(
            """
            SELECT e.payload_json
            FROM hypothesis_batches b
            JOIN memory_events e
              ON e.task_id=b.task_id AND e.event_id=b.created_event_id
            WHERE b.task_id=? AND b.batch_index=1
            """,
            (task_id,),
        ).fetchone()
        effective_chemistry_context = deepcopy(created["chemistry_context"])
        if batch is not None:
            payload = self._validated_batch_created_payload(
                conn,
                task_id,
                batch["payload_json"],
                field="initial HypothesisBatchCreated.payload_json",
            )
            snapshot = payload.get("question_snapshot")
            if not isinstance(snapshot, Mapping):
                raise ContractValidationError("首批缺少冻结 question snapshot")
            if snapshot.get("scientific_question") != created["scientific_question"]:
                raise ContractValidationError(
                    "首批 scientific question 与 TaskCreated 不一致"
                )
            retrieval = snapshot.get("initial_retrieval_phase")
            if not isinstance(retrieval, Mapping):
                raise ContractValidationError("首批缺少冻结 Retrieval 结果")
            retrieved_context = retrieval.get("chemistry_context")
            # 空 object 也是 Retrieval 的明确结果，只有非 object 才回退任务初始值。
            if isinstance(retrieved_context, dict):
                # TaskCreated 中的 calculation_unit 是调用方冻结的计算范围；
                # Retrieval 的派生字段可以补充它，但不能覆盖该字段。
                effective_chemistry_context = merge_supplementary_chemistry_context(
                    retrieved_context,
                    created["chemistry_context"],
                )
        return {
            "scientific_question": created["scientific_question"],
            "chemistry_context": effective_chemistry_context,
        }

    def _task_output_root(self, conn, task_id: str) -> Path:
        """根据任务创建时的相对目录，得到该任务唯一的文件根目录。"""

        task_created = self._task_created_payload(conn, task_id)
        scratch = Path(self.scratch_root).resolve()
        root = (scratch / task_created["task_output_relative_dir"]).resolve()
        try:
            root.relative_to(scratch)
        except ValueError as exc:
            raise ContractValidationError("task_output_relative_dir 越出 scratch_root") from exc
        return root

    def _validate_source_refs(self, conn, task_id: str, source_refs: Sequence[str]) -> List[str]:
        """确认每个证据引用属于当前任务，且引用文件的哈希没有变化。

        引用可以指向数据库事实或任务目录中的最终清单；孤立临时文件不能成为
        恢复依据。
        """

        refs = validate_source_ref_shape(list(source_refs))
        task_root: Optional[Path] = None
        for source_ref in refs:
            if "/" not in source_ref:
                event = conn.execute(
                    "SELECT 1 FROM memory_events WHERE task_id=? AND event_id=?",
                    (task_id, source_ref),
                ).fetchone()
                revision = conn.execute(
                    "SELECT 1 FROM revisions WHERE task_id=? AND revision_id=?",
                    (task_id, source_ref),
                ).fetchone()
                if event is None and revision is None:
                    raise ContractValidationError(
                        f"source_ref={source_ref} 不能解析为当前 task 的 event/revision"
                    )
                continue
            task_root = task_root or self._task_output_root(conn, task_id)
            unresolved = task_root / source_ref
            current = task_root
            for part in Path(source_ref).parts:
                current = current / part
                if current.is_symlink():
                    raise ContractValidationError("artifact source_ref 不能经过符号链接")
            target = unresolved.resolve()
            try:
                target.relative_to(task_root)
            except ValueError as exc:
                raise ContractValidationError("artifact source_ref 越出 task 根") from exc
            if not target.is_file():
                raise ContractValidationError(f"artifact source_ref 不存在：{source_ref}")
            expected_hash = sha256_from_manifest_name(source_ref)
            try:
                parsed = json.loads(target.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ContractValidationError("finalized manifest 必须是可读的严格 JSON") from exc
            actual_hash = hashlib.sha256(canonical_json(parsed).encode("utf-8")).hexdigest()
            if actual_hash != expected_hash:
                raise ContractValidationError(
                    f"finalized manifest hash 不匹配：expected={expected_hash}, actual={actual_hash}"
                )
        return refs

    @staticmethod
    def _resolve_event_reference(
        conn,
        task_id: str,
        reference: str,
        *,
        event_type: str | None = None,
    ) -> str:
        """把任务内 event_id 或幂等 event_key 统一解析为持久 event_id。

        输入是当前任务中的事件引用；动作只查询不可变事件，不写数据库。旧任务的
        失败投影可能把 event_key 保存到 ``execution_failure.event_id``，恢复时将其
        映射回真正的 event_id，避免把幂等键当作可引用的事件记录。找不到事件时
        原样返回，由后续严格的 source_ref 或事件类型校验报告失败。
        """

        conditions = "(event_id=? OR event_key=?)"
        params: list[Any] = [task_id, reference, reference]
        if event_type is not None:
            conditions = f"event_type=? AND {conditions}"
            params.insert(1, event_type)
        row = conn.execute(
            f"SELECT event_id FROM memory_events WHERE task_id=? AND {conditions}",
            tuple(params),
        ).fetchone()
        return row["event_id"] if row is not None else reference

    def _read_finalized_manifest(
        self,
        conn,
        task_id: str,
        relative_path: str,
    ) -> Dict[str, Any]:
        """读取一份已经正式发布且内容未变化的任务清单。

        所谓正式发布，是指文件路径已经进入事件引用，文件名中的摘要也与实际内容
        一致。恢复逻辑不会扫描目录猜测哪个文件最新，更不会把尚未提交的临时文件
        当作已发生事实。
        """

        self._validate_source_refs(conn, task_id, [relative_path])
        target = self._task_output_root(conn, task_id) / relative_path
        try:
            value = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ContractValidationError("finalized manifest 无法读取") from exc
        if not isinstance(value, dict):
            raise ContractValidationError("finalized manifest 根必须是 JSON object")
        return value

    @staticmethod
    def _validate_causal_source(
        conn,
        task_id: str,
        causal_source_ref: Any,
        *,
        allowed_source_event_types: Sequence[str] = ("restart_applied",),
    ) -> None:
        """确认结果只消费允许类型的一条尚未使用的因果来源。

        常规工作流结果只能消费人工 Restart。Retrieval 成功结果还可以消费此前的
        RetrievalPaused，因为论文检索恢复不经过 Hypothesis 的任务级 Restart。
        同一来源仍只能被一条真实结果或 workflow revision 消费，避免重复恢复推动
        两条分支。
        """

        if causal_source_ref is None:
            return
        allowed_types = tuple(dict.fromkeys(allowed_source_event_types))
        if not allowed_types:
            raise ContractValidationError("causal_source_ref 没有允许的来源事件类型")
        row = conn.execute(
            "SELECT event_type FROM memory_events WHERE task_id=? AND event_id=?",
            (task_id, causal_source_ref),
        ).fetchone()
        if row is None or row["event_type"] not in allowed_types:
            expected = "、".join(allowed_types)
            raise ContractValidationError(
                f"causal_source_ref 必须引用当前 task 的 {expected} event"
            )
        placeholders = ",".join(
            "?" for _ in CAUSAL_SOURCE_CONSUMER_EVENT_TYPES
        )
        consumer = conn.execute(
            f"""
            SELECT event_id FROM memory_events
            WHERE task_id=?
              AND event_type IN ({placeholders})
              AND json_extract(payload_json, '$.causal_source_ref')=?
            LIMIT 1
            """,
            (
                task_id,
                *sorted(CAUSAL_SOURCE_CONSUMER_EVENT_TYPES),
                causal_source_ref,
            ),
        ).fetchone()
        if consumer is not None:
            raise OperationConflict("causal_source_ref 已被另一条真实结果消费")
        revision = conn.execute(
            "SELECT revision_id FROM revisions WHERE task_id=? "
            "AND json_extract(revised_result_json, '$.causal_source_ref')=? LIMIT 1",
            (task_id, causal_source_ref),
        ).fetchone()
        if revision is not None:
            raise OperationConflict("causal_source_ref 已被 workflow revision 消费")

    def _pending_task_hypothesis_restart(
        self,
        conn,
        task_id: str,
    ) -> Optional[Dict[str, str]]:
        """找到尚未被下一次假设生成结果消费的唯一任务级恢复授权。"""

        rows = conn.execute(
            "SELECT * FROM memory_events WHERE task_id=? "
            "AND event_type='restart_applied' ORDER BY created_at, event_id",
            (task_id,),
        ).fetchall()
        pending: list[Dict[str, str]] = []
        placeholders = ",".join(
            "?" for _ in CAUSAL_SOURCE_CONSUMER_EVENT_TYPES
        )
        for row in rows:
            refs = validate_source_ref_shape(
                self._decode_json(
                    row["source_refs_json"],
                    field="task Restart.source_refs_json",
                )
            )
            pause_ids = [item for item in refs if "/" not in item]
            if len(pause_ids) != 1:
                continue
            pause = conn.execute(
                "SELECT payload_json, source_refs_json FROM memory_events "
                "WHERE task_id=? AND event_id=? AND event_type='execution_paused'",
                (task_id, pause_ids[0]),
            ).fetchone()
            if pause is None:
                continue
            pause_payload = self._decode_json(
                pause["payload_json"],
                field="task Pause.payload_json",
            )
            if (
                pause_payload.get("combination_id") is not None
                or pause_payload.get("operation") != "hypothesis_generation"
            ):
                continue
            consumer = conn.execute(
                f"SELECT 1 FROM memory_events WHERE task_id=? "
                f"AND event_type IN ({placeholders}) "
                "AND json_extract(payload_json, '$.causal_source_ref')=? LIMIT 1",
                (
                    task_id,
                    *sorted(CAUSAL_SOURCE_CONSUMER_EVENT_TYPES),
                    row["event_id"],
                ),
            ).fetchone()
            if consumer is not None:
                continue
            pause_refs = validate_source_ref_shape(
                self._decode_json(
                    pause["source_refs_json"],
                    field="task Pause.source_refs_json",
                )
            )
            manifests = [item for item in pause_refs if "/" in item]
            if len(manifests) != 1:
                raise ContractValidationError(
                    "任务级 hypothesis Pause 必须恰含一份 finalized manifest"
                )
            pending.append(
                {
                    "restart_record_id": row["event_id"],
                    "pause_record_id": pause_ids[0],
                    "prior_manifest_ref": manifests[0],
                }
            )
        if len(pending) > 1:
            raise ContractValidationError("任务存在多个未消费的 hypothesis Restart")
        return pending[0] if pending else None

    @staticmethod
    def _assert_task_writable(conn, task_id: str) -> None:
        """任务结束后只允许重放已有结果，不能再追加新的业务事实。"""

        row = conn.execute("SELECT status FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        if row is None:
            raise ContractValidationError(f"未知 task_id：{task_id}")
        if row["status"] in WorkingMemoryService._TASK_TERMINAL_STATES:
            raise CASConflict(
                "terminal task 禁止写入新事实",
                expected_version=None,
                current_version=None,
                current_state=row["status"],
            )

    @staticmethod
    def _assert_operation_available(
        conn,
        task_id: str,
        operation_instance_id: str,
        *,
        allow_event_types: Sequence[str] = (),
    ) -> None:
        """确认运行身份未被另一种终态占用，可显式复用同一 debate 事件。"""

        placeholders = ",".join("?" for _ in OPERATION_TERMINAL_EVENT_TYPES)
        rows = conn.execute(
            f"""
            SELECT event_id, event_type FROM memory_events
            WHERE task_id=?
              AND event_type IN ({placeholders})
              AND json_extract(payload_json, '$.operation_instance_id')=?
            """,
            (task_id, *sorted(OPERATION_TERMINAL_EVENT_TYPES), operation_instance_id),
        ).fetchall()
        allowed = set(allow_event_types)
        allowed_rows = []
        for row in rows:
            if row["event_type"] in allowed:
                allowed_rows.append(row)
            else:
                raise OperationConflict(
                    f"operation_instance_id={operation_instance_id} 已由 {row['event_type']} 占用"
                )
        if len(allowed_rows) > 1:
            raise OperationConflict(
                f"operation_instance_id={operation_instance_id} 存在重复允许事件"
            )
        revision = conn.execute(
            """
            SELECT revision_id FROM revisions
            WHERE task_id=?
              AND json_extract(revised_result_json, '$.operation_instance_id')=?
            LIMIT 1
            """,
            (task_id, operation_instance_id),
        ).fetchone()
        if revision is not None:
            raise OperationConflict(
                f"operation_instance_id={operation_instance_id} 已由 revision 占用"
            )

    def _insert_event(
        self,
        conn,
        *,
        event_key: str,
        task_id: str,
        round_id: int,
        event_type: str,
        payload: Mapping[str, Any],
        source_refs: Sequence[str],
        check_operation_occupancy: bool,
        allow_operation_event_types: Sequence[str] = (),
        allow_causal_source_event_types: Sequence[str] = ("restart_applied",),
        created_at: Optional[str] = None,
    ) -> Tuple[str, bool]:
        """先处理安全重放，再把一条新事件追加到历史中。

        事件一旦写入便不更新；当前状态由同一事务中的投影更新完成。
        """

        existing_id = self._existing_event_exact(
            conn,
            event_key=event_key,
            task_id=task_id,
            round_id=round_id,
            event_type=event_type,
            payload=payload,
            source_refs=source_refs,
        )
        if existing_id is not None:
            return existing_id, False
        # 启动探测失败的任务在创建事务中直接落为 failed；仍需先写入唯一的
        # TaskCreated 事实。创建事件成功后，该终态继续由 _assert_task_writable
        # 保护，不能追加任何科学或恢复事实。
        is_first_task_created = (
            event_type == "task_created"
            and conn.execute(
                "SELECT 1 FROM memory_events WHERE task_id=? LIMIT 1", (task_id,)
            ).fetchone()
            is None
        )
        # 维护者更正是唯一允许在 accepted succeeded task 上追加的事实；它本身
        # 必须继续经过 service 的当前 TaskClosed/Review/Result 身份核对，投影成功后
        # 才把任务恢复为 running。其他终态仍由统一写保护拒绝。
        is_explicit_investigation_correction = (
            event_type == "investigation_task_correction_recorded"
        )
        if not (is_first_task_created or is_explicit_investigation_correction):
            self._assert_task_writable(conn, task_id)
        refs = self._validate_source_refs(conn, task_id, source_refs)
        self._validate_causal_source(
            conn,
            task_id,
            payload.get("causal_source_ref"),
            allowed_source_event_types=allow_causal_source_event_types,
        )
        operation_instance_id = payload.get("operation_instance_id")
        if check_operation_occupancy and operation_instance_id:
            self._assert_operation_available(
                conn,
                task_id,
                operation_instance_id,
                allow_event_types=allow_operation_event_types,
            )
        event_id = new_id("evt")
        conn.execute(
            """
            INSERT INTO memory_events(
                event_id, event_key, task_id, round, event_type,
                payload_json, source_refs_json, created_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                event_key,
                task_id,
                round_id,
                event_type,
                canonical_json(payload),
                canonical_json(refs),
                created_at or utc_now(),
            ),
        )
        return event_id, True

    @staticmethod
    def _reduce_task_status_in_transaction(conn, task_id: str) -> str:
        """根据同一事务中看到的全部组合，计算任务对外显示的调度状态。

        某个组合暂停时，同批其他组合仍应继续运行。只有不存在可推进或仍在等待的
        组合时，任务才整体显示为暂停；全部组合到达可归约终态后暂时保持运行中，
        等批次归约器一次性发布成功、失败或下一批结论。
        """

        task = conn.execute(
            "SELECT status FROM tasks WHERE task_id=?", (task_id,)
        ).fetchone()
        if task is None:
            raise ContractValidationError("任务状态归约引用未知 task")
        if task["status"] in WorkingMemoryService._TASK_TERMINAL_STATES:
            return task["status"]
        rows = conn.execute(
            "SELECT state FROM combinations WHERE task_id=?", (task_id,)
        ).fetchall()
        states = [row["state"] for row in rows]
        if any(
            state
            in {"generating_workflow", "running", "routing", "revising_workflow"}
            for state in states
        ):
            reduced = "running"
        elif any(state == "waiting_execution" for state in states):
            reduced = "waiting_execution"
        elif any(state == "paused" for state in states):
            reduced = "paused"
        else:
            reduced = "running"
        if task["status"] != reduced:
            conn.execute(
                "UPDATE tasks SET status=?, version=version+1, updated_at=? "
                "WHERE task_id=?",
                (reduced, utc_now(), task_id),
            )
        return reduced

    @staticmethod
    def _parse_event_timestamp(value: str, *, field: str) -> datetime:
        """读取带时区的 ISO 时间，供历史顺序和最终发布共同使用。

        不带时区的时间在不同机器上含义不确定，不能进入可重放历史。这里返回可直接
        比较的时间对象，批次关闭就能证明自己的发布时间不早于任何来源事实。
        """

        require_nonempty_string(value, field)
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ContractValidationError(f"{field} 必须是带时区的 ISO 时间") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ContractValidationError(f"{field} 必须明确包含时区")
        return parsed

    @staticmethod
    def _strict_empty_action_schema(description: str) -> Dict[str, Any]:
        """描述一种无需附加参数的人工恢复动作，并拒绝偷偷夹带配置。"""

        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "description": description,
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }

    @classmethod
    def _restart_action_schemas(cls) -> Dict[str, Dict[str, Any]]:
        """生成研究员可提交的恢复动作输入说明。

        说明书既用于界面展示，也用于提交前校验。无参数动作必须提交空对象，
        放弃假设必须提交原因；旧版 ToolSpec 工作流补丁不在当前说明书中。
        """

        text_field = {"type": "string", "minLength": 1}
        return {
            "retry_failed_operation": cls._strict_empty_action_schema(
                "重新运行失败的模型或首批检索 operation；不会复用孤立文件。"
            ),
            "retry_current_step": cls._strict_empty_action_schema(
                "保留工作流、execution 和已完成前缀，从当前步骤创建新 operation。"
            ),
            "restart_workflow": cls._strict_empty_action_schema(
                "保留科学工作流和尝试次数，分配新 execution 并从第一步重新运行。"
            ),
            "revise_hypothesis": {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "description": "结束当前组合并记录研究员放弃该假设的原因。",
                "properties": {"reason": text_field},
                "required": ["reason"],
                "additionalProperties": False,
            },
        }

    @staticmethod
    def _extract_error_code(evidence: Sequence[Mapping[str, Any]]) -> Optional[str]:
        """从暂停证据读取错误分类，兼容旧的 ``error_code`` 和 MiniChem 的 ``code``。

        错误分类只用于决定恢复动作，不会改变保存的原始错误文件。若错误内容不是
        JSON object 或没有这两个字段，则返回 ``None``，调用方按最保守规则处理。
        """

        for item in evidence:
            if item.get("kind") != "error" or not isinstance(
                item.get("content"), Mapping
            ):
                continue
            content = item["content"]
            nested_error = content.get("error")
            error_code = content.get("error_code") or content.get("code")
            if not error_code and isinstance(nested_error, Mapping):
                error_code = nested_error.get("error_code") or nested_error.get("code")
            if isinstance(error_code, str) and error_code:
                return error_code
        return None

    def _read_artifact_reference(
        self,
        conn,
        task_id: str,
        reference: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """读取四字段引用指向的 JSON 文件，并确认它从未被替换。

        路径、媒体类型、字节数和摘要必须与记录完全一致。恢复说明和模型操作重放
        因而都读取当时冻结的内容，而不是目录中后来生成的同名文件。
        """

        if set(reference) != {"relative_path", "sha256", "media_type", "size"}:
            raise ContractValidationError("artifact ref 必须恰含四个固定字段")
        if reference["media_type"] != "application/json":
            raise ContractValidationError("artifact ref 当前只支持 JSON 文件")
        path = reference["relative_path"]
        validate_task_relative_path(path, "artifact ref relative_path")
        task_root = self._task_output_root(conn, task_id)
        current = task_root
        for component in Path(path).parts:
            current = current / component
            if current.is_symlink():
                raise ContractValidationError("artifact ref 不能经过符号链接")
        target = (task_root / path).resolve()
        try:
            target.relative_to(task_root)
        except ValueError as exc:
            raise ContractValidationError("artifact ref 越出 task 根") from exc
        if not target.is_file():
            raise ContractValidationError("artifact ref 指向的文件不存在")
        content = target.read_bytes()
        if (
            reference["size"] != len(content)
            or reference["sha256"] != hashlib.sha256(content).hexdigest()
        ):
            raise ContractValidationError("artifact ref 的 size 或 sha256 已变化")
        try:
            value = json.loads(content)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ContractValidationError("artifact ref 指向的内容不是严格 JSON") from exc
        if canonical_json(value).encode("utf-8") != content:
            raise ContractValidationError("artifact ref 的 JSON 不是 canonical 编码")
        return {"artifact_ref": deepcopy(dict(reference)), "content": value}

    def _pause_context(self, conn, task_id: str, pause_id: str) -> Dict[str, Any]:
        """读取一条暂停事实及其当前组合，作为说明和恢复的共同上下文。

        两个入口共用同一读取方法，可确保研究员看到的暂停原因与随后实际处理的
        暂停记录完全相同，不会因各自查询条件不同而恢复错组合。
        """

        row = conn.execute(
            "SELECT * FROM memory_events WHERE task_id=? AND event_id=?",
            (task_id, pause_id),
        ).fetchone()
        if row is None or row["event_type"] != "execution_paused":
            raise ContractValidationError("pause_id 不属于当前任务的 execution_paused")
        event = {
            **dict(row),
            "payload": self._decode_json(row["payload_json"], field="Pause.payload_json"),
            "source_refs": self._decode_json(
                row["source_refs_json"], field="Pause.source_refs_json"
            ),
        }
        combination = None
        if event["payload"]["combination_id"] is not None:
            combination = conn.execute(
                "SELECT * FROM combinations WHERE task_id=? AND combination_id=?",
                (task_id, event["payload"]["combination_id"]),
            ).fetchone()
            if combination is None:
                raise ContractValidationError("Pause 的 combination projection 缺失")
        return {"event": event, "combination": combination}

    def _pause_source_evidence(
        self,
        conn,
        task_id: str,
        pause: Mapping[str, Any],
    ) -> List[Dict[str, Any]]:
        """按步骤清单登记顺序展开错误和专家诊断。

        专家文本只作为研究员判断依据，不能自行增加可执行动作。允许哪些恢复方式
        仍由暂停类型、确定性错误分类和工具的安全重试声明共同决定。
        """

        if pause["payload"]["operation"] != "execute_step":
            return []
        manifest_refs = [item for item in pause["source_refs"] if "/" in item]
        if len(manifest_refs) != 1:
            raise ContractValidationError("execute_step Pause 必须恰含一份 step manifest")
        manifest = self._read_finalized_manifest(conn, task_id, manifest_refs[0])
        evidence = []
        error_ref = manifest.get("error_ref")
        if isinstance(error_ref, Mapping):
            item = self._read_artifact_reference(conn, task_id, error_ref)
            item["kind"] = "error"
            evidence.append(item)
        for reference in manifest.get("artifact_refs", []):
            if not isinstance(reference, Mapping):
                raise ContractValidationError("step manifest.artifact_refs 含非法引用")
            item = self._read_artifact_reference(conn, task_id, reference)
            item["kind"] = "diagnostic"
            evidence.append(item)
        return evidence

    def _allowed_restart_actions(
        self,
        *,
        pause_payload: Mapping[str, Any],
        error_code: Optional[str],
        retry_safe: bool,
    ) -> List[str]:
        """根据失败发生阶段和错误性质，筛出不会破坏历史的恢复动作。

        模型调用失败只能重做该模型调用；工具失败可以改工作流或放弃假设。原地重试
        和从头执行风险更高，只有明确的运行环境故障才会开放。
        """

        if pause_payload["operation"] != "execute_step":
            # provider 已接受但本地尚未保存结果的请求无法安全原地重试；只有
            # ``restart_workflow`` 会生成新的 execution 身份，允许研究员明确
            # 放弃未知在途请求后重新提交。
            if pause_payload.get("reason_code") == "unknown_in_flight_request":
                allowed = {"restart_workflow"}
            else:
                allowed = {"retry_failed_operation"}
        else:
            allowed = {"revise_hypothesis"}
            if error_code in self._RESTART_WORKFLOW_ERRORS:
                allowed.add("restart_workflow")
            if retry_safe and error_code in self._RETRY_CURRENT_STEP_ERRORS:
                allowed.add("retry_current_step")
        return [item for item in self._RESTART_ACTION_ORDER if item in allowed]

    def _restart_blocker(
        self,
        conn,
        *,
        task_id: str,
        pause_id: str,
    ) -> Optional[Dict[str, Any]]:
        """判断此刻是否可以安全应用人工恢复。

        只要还有组合在运行或等待外部程序，人工修改就可能与后台写入交叉，因此
        必须等整批进入静止边界。已处理的暂停和已结束任务同样不能再次恢复。
        """

        task = conn.execute(
            "SELECT status FROM tasks WHERE task_id=?", (task_id,)
        ).fetchone()
        if task is None:
            raise ContractValidationError("Restart 引用未知 task")
        if task["status"] in self._TASK_TERMINAL_STATES:
            return {"reason": "task_terminal", "combination_ids": []}
        resolved = conn.execute(
            "SELECT 1 FROM memory_events WHERE task_id=? AND event_key=?",
            (task_id, event_key_restart(pause_id)),
        ).fetchone()
        if resolved is not None:
            return {"reason": "pause_resolved", "combination_ids": []}
        busy = conn.execute(
            """
            SELECT combination_id FROM combinations
            WHERE task_id=? AND state IN (
                'generating_workflow', 'running', 'waiting_execution',
                'routing', 'revising_workflow'
            )
            ORDER BY batch_id, combination_index
            """,
            (task_id,),
        ).fetchall()
        if busy:
            return {
                "reason": "task_busy",
                "combination_ids": [item["combination_id"] for item in busy],
            }
        return None

    # ==================== 暂停与人工恢复 ====================

    def get_restart_guide(
        self,
        task_id: str,
        pause_id: str,
        *,
        retry_safe: bool = False,
    ) -> Dict[str, Any]:
        """生成一份只读恢复说明，列出当前证据和可以安全选择的动作。

        是否允许原地重试必须来自任务创建时冻结的工具规则；调用方没有提供明确
        声明时按不安全处理。专家意见只供研究员阅读，不会自动增加或重排动作，
        调用本方法本身也不会改变数据库状态。
        """

        store = self._require_enabled()
        require_nonempty_string(pause_id, "pause_id")
        with store.transaction() as conn:
            context = self._pause_context(conn, task_id, pause_id)
            pause = context["event"]
            evidence = self._pause_source_evidence(conn, task_id, pause)
            error_code = self._extract_error_code(evidence)
            actions = self._allowed_restart_actions(
                pause_payload=pause["payload"],
                error_code=error_code,
                retry_safe=bool(retry_safe),
            )
            blocker = self._restart_blocker(
                conn,
                task_id=task_id,
                pause_id=pause_id,
            )
            if blocker is not None and blocker["reason"] in {
                "pause_resolved",
                "task_terminal",
            }:
                actions = []
            schemas = self._restart_action_schemas()
            workflow_view = None
            if (
                pause["payload"]["operation"] == "execute_step"
                and context["combination"] is not None
            ):
                workflow = self._decode_json(
                    context["combination"]["current_workflow_json"],
                    field="current_workflow_json",
                )
                workflow_view = project_workflow_refs(workflow)["workflow"]
            return {
                "pause_id": pause_id,
                "reason_code": pause["payload"]["reason_code"],
                "operation": pause["payload"]["operation"],
                "restart_allowed": blocker is None,
                "allowed_actions": actions,
                "blocked_by": blocker,
                "action_schemas": {item: deepcopy(schemas[item]) for item in actions},
                "current_workflow_ref_view": workflow_view,
                "source_evidence": evidence,
            }

    def get_unresolved_task_pause(self, task_id: str) -> Optional[Dict[str, Any]]:
        """返回当前唯一未处理的任务级暂停；没有时返回空。"""

        store = self._require_enabled()
        require_nonempty_string(task_id, "task_id")
        with store.transaction() as conn:
            task = conn.execute(
                "SELECT status FROM tasks WHERE task_id=?",
                (task_id,),
            ).fetchone()
            if task is None:
                raise ContractValidationError("task-scope Pause 引用未知任务")
            rows = conn.execute(
                "SELECT event_id, payload_json FROM memory_events "
                "WHERE task_id=? AND event_type='execution_paused' "
                "ORDER BY created_at, event_id",
                (task_id,),
            ).fetchall()
            unresolved: List[Dict[str, Any]] = []
            for row in rows:
                payload = validate_pause_payload(
                    self._decode_json(
                        row["payload_json"],
                        field="Pause.payload_json",
                    )
                )
                if payload["combination_id"] is not None:
                    continue
                restart = conn.execute(
                    "SELECT 1 FROM memory_events WHERE task_id=? AND event_key=?",
                    (task_id, event_key_restart(row["event_id"])),
                ).fetchone()
                if restart is None:
                    unresolved.append(
                        {
                            "pause_id": row["event_id"],
                            "operation": payload["operation"],
                            "reason_code": payload["reason_code"],
                        }
                    )
            if len(unresolved) > 1:
                raise ContractValidationError("任务存在多条未处理的 task-scope Pause")
            # 任务状态由全部组合的归约结果决定。组合级 Pause 也会把 task
            # 投影为 paused，因此不能再把 task-scope Pause 当成唯一来源；只有
            # 任务级 Pause 自身残留在非 paused 任务上时，才说明投影已经错位。
            if task["status"] != "paused" and unresolved:
                raise ContractValidationError("运行任务残留未处理的 task-scope Pause")
            return None if not unresolved else unresolved[0]

    def _unresolved_combination_pause(
        self,
        conn,
        task_id: str,
        combination_id: str,
    ) -> Optional[Dict[str, Any]]:
        """在同一数据库快照中找出组合当前尚未处理的唯一暂停。"""

        combination = conn.execute(
            "SELECT state, terminal_reason FROM combinations "
            "WHERE task_id=? AND combination_id=?",
            (task_id, combination_id),
        ).fetchone()
        if combination is None:
            raise ContractValidationError("combination Pause 引用了未知组合")
        rows = conn.execute(
            "SELECT event_id, payload_json, source_refs_json, created_at "
            "FROM memory_events WHERE task_id=? AND event_type='execution_paused' "
            "ORDER BY created_at, event_id",
            (task_id,),
        ).fetchall()
        unresolved = []
        for row in rows:
            payload = validate_pause_payload(
                self._decode_json(
                    row["payload_json"],
                    field="Pause.payload_json",
                )
            )
            if payload["combination_id"] != combination_id:
                continue
            restart = conn.execute(
                "SELECT 1 FROM memory_events WHERE task_id=? AND event_key=?",
                (task_id, event_key_restart(row["event_id"])),
            ).fetchone()
            if restart is not None:
                continue
            unresolved.append(
                {
                    "pause_id": row["event_id"],
                    "operation": payload["operation"],
                    "reason_code": payload["reason_code"],
                    "payload": payload,
                    "source_refs": validate_source_ref_shape(
                        self._decode_json(
                            row["source_refs_json"],
                            field="Pause.source_refs_json",
                        )
                    ),
                    "created_at": row["created_at"],
                }
            )
        if len(unresolved) > 1:
            raise ContractValidationError("组合存在多条尚未处理的 Pause")
        if combination["terminal_reason"] in {
            "expert_gaussian_reject",
            "expert_model_not_configured",
        }:
            if combination["state"] != "paused" or unresolved:
                raise ContractValidationError("专家审查停止状态与技术 Pause 事实冲突")
            return None
        if combination["state"] == "paused" and len(unresolved) != 1:
            raise ContractValidationError("暂停组合缺少唯一未处理 Pause")
        if combination["state"] != "paused" and unresolved:
            raise ContractValidationError("非暂停组合残留未处理 Pause")
        return None if not unresolved else unresolved[0]

    def get_unresolved_combination_pause(
        self,
        task_id: str,
        combination_id: str,
    ) -> Optional[Dict[str, Any]]:
        """返回一个组合当前的暂停身份和原因，不修改任何运行状态。"""

        store = self._require_enabled()
        require_nonempty_string(task_id, "task_id")
        require_nonempty_string(combination_id, "combination_id")
        with store.transaction() as conn:
            pause = self._unresolved_combination_pause(
                conn,
                task_id,
                combination_id,
            )
            return None if pause is None else deepcopy(pause)

    def list_unresolved_combination_pauses(
        self,
        task_id: str,
    ) -> List[Dict[str, Any]]:
        """返回任务下全部尚未处理的技术 Pause，并按组合顺序排列。

        任务可能同时有多个组合暂停；调用方不能只读取第一个组合来生成暂停结果。
        本方法只读取数据库，不创建 Restart，也不改变任务或组合状态。暂停身份、
        原因、来源引用和组合编号均来自同一事务快照；若某个暂停与组合投影不一致，
        立即抛出契约错误而不返回不完整列表。专家 reject 和专家模型未配置由各自的
        专家事实发布，不伪装成可执行 Restart 的技术 Pause，因此不进入本列表。
        """

        store = self._require_enabled()
        require_nonempty_string(task_id, "task_id")
        with store.transaction() as conn:
            rows = conn.execute(
                "SELECT c.combination_id FROM combinations c "
                "JOIN hypothesis_batches b "
                "  ON b.task_id=c.task_id AND b.batch_id=c.batch_id "
                "WHERE c.task_id=? AND c.state='paused' "
                "ORDER BY b.batch_index, c.combination_index, c.combination_id",
                (task_id,),
            ).fetchall()
            pauses: List[Dict[str, Any]] = []
            for row in rows:
                pause = self._unresolved_combination_pause(
                    conn,
                    task_id,
                    row["combination_id"],
                )
                if pause is not None:
                    pauses.append(deepcopy(pause))
            return pauses

    @staticmethod
    def _validate_restart_request(raw: Any, index: int) -> Dict[str, Any]:
        """校验一项人工恢复请求，确保输入恰好满足所选动作。

        不需要参数的动作必须提交空对象，放弃假设必须提交原因。未知动作（包括
        已停用的旧 ToolSpec patch）在打开事务前拒绝，避免错误请求留下任何事实。
        """

        if not isinstance(raw, dict) or set(raw) != {
            "pause_id",
            "action",
            "action_input",
        }:
            raise ContractValidationError(
                f"Restart requests[{index}] 必须恰含 pause_id/action/action_input"
            )
        pause_id = require_nonempty_string(raw["pause_id"], "pause_id")
        action = require_nonempty_string(raw["action"], "action")
        action_input = raw["action_input"]
        if not isinstance(action_input, dict):
            raise ContractValidationError("Restart action_input 必须是 JSON object")
        if action not in WorkingMemoryService._RESTART_ACTION_ORDER:
            raise ContractValidationError(f"Restart action 不受支持：{action}")
        if action == "revise_hypothesis":
            if set(action_input) != {"reason"}:
                raise ContractValidationError(
                    "revise_hypothesis.action_input 必须只含 reason"
                )
            require_nonempty_string(action_input["reason"], "reason")
        elif action_input:
            raise ContractValidationError(f"{action}.action_input 必须是空对象")
        return {
            "pause_id": pause_id,
            "action": action,
            "action_input": deepcopy(action_input),
        }

    @staticmethod
    def _restart_result(event_id: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
        """返回调度器继续运行所需的最小人工恢复结果，不暴露内部事件正文。"""

        result = {"restart_id": event_id, "action": payload["action"]}
        if "execution_id" in payload:
            result["execution_id"] = payload["execution_id"]
        return result

    def apply_restarts(
        self,
        task_id: str,
        payload: Mapping[str, Any],
        *,
        retry_safe_by_pause: Optional[Mapping[str, bool]] = None,
    ) -> List[Dict[str, Any]]:
        """在一个事务中应用一项或多项人工恢复决定。

        所有请求先检查暂停归属、任务静止边界、动作适用性和重复提交情况。只有全部
        是新请求时才一起写入；全部与旧请求相同时直接返回原结果。新旧混合或同一
        暂停改换动作会整体拒绝，研究员不会看到只成功一半的批量恢复。
        """

        store = self._require_enabled()
        if not isinstance(payload, Mapping) or set(payload) != {"requests"}:
            raise ContractValidationError("Restart batch 必须恰含 requests")
        raw_requests = payload["requests"]
        if not isinstance(raw_requests, list) or not raw_requests:
            raise ContractValidationError("Restart requests 必须是非空数组")
        requests = [
            self._validate_restart_request(item, index)
            for index, item in enumerate(raw_requests)
        ]
        pause_ids = [item["pause_id"] for item in requests]
        if len(set(pause_ids)) != len(pause_ids):
            raise ContractValidationError("同一批 Restart 的 pause_id 不得重复")
        retry_safe_by_pause = dict(retry_safe_by_pause or {})

        with store.immediate_transaction() as conn:
            contexts = [
                self._pause_context(conn, task_id, item["pause_id"])
                for item in requests
            ]
            existing_rows = [
                conn.execute(
                    "SELECT * FROM memory_events WHERE task_id=? AND event_key=?",
                    (task_id, event_key_restart(item["pause_id"])),
                ).fetchone()
                for item in requests
            ]
            existing_count = sum(row is not None for row in existing_rows)
            if 0 < existing_count < len(requests):
                raise IdempotencyConflict(
                    "批量 Restart 不能混合已提交请求和新请求"
                )
            if existing_count == len(requests):
                replay_results = []
                for request, row in zip(requests, existing_rows):
                    stored = self._decode_json(
                        row["payload_json"], field="Restart.payload_json"
                    )
                    equivalent = stored.get("action") == request["action"]
                    if request["action"] == "revise_hypothesis":
                        equivalent = equivalent and stored.get("reason") == request[
                            "action_input"
                        ]["reason"]
                    if not equivalent:
                        raise IdempotencyConflict(
                            f"pause_id={request['pause_id']} 的 Restart 内容已变化"
                        )
                    replay_results.append(
                        self._restart_result(row["event_id"], stored)
                    )
                return replay_results

            # 所有新请求必须在同一个“整批已经静止”的时刻应用。只要仍有组合正在
            # 运行或等待外部程序，整批请求就不落库，以免人工修改与后台结果相互覆盖。
            blocker = self._restart_blocker(
                conn,
                task_id=task_id,
                pause_id=requests[0]["pause_id"],
            )
            if blocker is not None:
                raise CASConflict(
                    f"Restart 当前被 {blocker['reason']} 阻止",
                    expected_version=None,
                    current_version=None,
                    expected_state="paused",
                    current_state=blocker["reason"],
                )

            prepared = []
            for request, context in zip(requests, contexts):
                pause = context["event"]
                evidence = self._pause_source_evidence(conn, task_id, pause)
                error_code = self._extract_error_code(evidence)
                allowed = self._allowed_restart_actions(
                    pause_payload=pause["payload"],
                    error_code=error_code,
                    retry_safe=bool(retry_safe_by_pause.get(request["pause_id"], False)),
                )
                if request["action"] not in allowed:
                    raise ContractValidationError(
                        f"action={request['action']} 不适用于当前 Pause"
                    )
                prepared.append((request, context))

            results = []
            for request, context in prepared:
                pause = context["event"]
                combination = context["combination"]
                action = request["action"]
                restart_payload: Dict[str, Any] = {"action": action}
                source_refs = [request["pause_id"]]
                runtime = None
                if action == "restart_workflow":
                    restart_payload["execution_id"] = new_id("execution")
                elif action == "revise_hypothesis":
                    restart_payload["reason"] = request["action_input"]["reason"]
                source_refs = sorted(source_refs)
                event_id, _ = self._insert_event(
                    conn,
                    event_key=event_key_restart(request["pause_id"]),
                    task_id=task_id,
                    round_id=pause["round"],
                    event_type="restart_applied",
                    payload=restart_payload,
                    source_refs=source_refs,
                    check_operation_occupancy=False,
                )

                if combination is None:
                    # 首批假设生成发生在具体组合创建之前，所以任务级恢复没有组合行
                    # 可以存放授权。下一次批次创建会直接引用本条恢复事件，数据库据此
                    # 证明授权已被使用，无需再增加一份容易不同步的任务游标字段。
                    cursor = conn.execute(
                        "UPDATE tasks SET status='running', version=version+1, updated_at=? "
                        "WHERE task_id=? AND status='paused'",
                        (utc_now(), task_id),
                    )
                    if cursor.rowcount != 1:
                        raise CASConflict(
                            "任务级 Restart 应用时状态已经变化",
                            expected_version=None,
                            current_version=None,
                            expected_state="paused",
                            current_state=None,
                        )
                elif action == "revise_hypothesis":
                    cursor = conn.execute(
                        """
                        UPDATE combinations SET state='hypothesis_failed',
                            terminal_record_id=?, terminal_reason='human_revise_hypothesis',
                            causal_source_ref=NULL, version=version+1, updated_at=?
                        WHERE task_id=? AND combination_id=? AND state='paused'
                          AND version=?
                        """,
                        (
                            event_id,
                            utc_now(),
                            task_id,
                            combination["combination_id"],
                            combination["version"],
                        ),
                    )
                    if cursor.rowcount != 1:
                        raise CASConflict(
                            "人工放弃假设时组合状态已经变化",
                            expected_version=combination["version"],
                            current_version=combination["version"],
                            expected_state="paused",
                            current_state=combination["state"],
                        )
                else:
                    runtime = runtime or self._decode_json(
                        combination["runtime_checkpoint_json"],
                        field="runtime_checkpoint_json",
                    )
                    runtime["current_operation_instance_id"] = None
                    runtime["current_request_ref"] = None
                    runtime["checkpoint_id"] = None
                    runtime["checkpoint_manifest_ref"] = None
                    current_workflow = (
                        None
                        if combination["current_workflow_json"] is None
                        else self._decode_json(
                            combination["current_workflow_json"],
                            field="current_workflow_json",
                        )
                    )
                    updates = {
                        "state": (
                            self._MODEL_OPERATION_STATES[pause["payload"]["operation"]]
                            if action == "retry_failed_operation"
                            else "running"
                        ),
                        "execution_id": combination["execution_id"],
                        "workflow_version": combination["workflow_version"],
                        "current_workflow": current_workflow,
                    }
                    if action == "restart_workflow":
                        updates["execution_id"] = restart_payload["execution_id"]
                        runtime = empty_runtime_checkpoint()
                    cursor = conn.execute(
                        """
                        UPDATE combinations SET state=?, execution_id=?, workflow_version=?,
                            current_workflow_json=?, runtime_checkpoint_json=?,
                            terminal_record_id=NULL, terminal_reason=NULL,
                            causal_source_ref=?, version=version+1, updated_at=?
                        WHERE task_id=? AND combination_id=? AND state='paused' AND version=?
                        """,
                        (
                            updates["state"],
                            updates["execution_id"],
                            updates["workflow_version"],
                            (
                                None
                                if updates["current_workflow"] is None
                                else canonical_json(updates["current_workflow"])
                            ),
                            canonical_json(runtime),
                            event_id,
                            utc_now(),
                            task_id,
                            combination["combination_id"],
                            combination["version"],
                        ),
                    )
                    if cursor.rowcount != 1:
                        raise CASConflict(
                            "Restart 输掉 combination CAS",
                            expected_version=combination["version"],
                            current_version=combination["version"],
                            expected_state="paused",
                            current_state=combination["state"],
                        )
                results.append(self._restart_result(event_id, restart_payload))
            self._reduce_task_status_in_transaction(conn, task_id)
            return results

    # ==================== 任务与批次状态写入 ====================

    def create_task(
        self,
        *,
        task_id: str,
        scientific_question: str,
        chemistry_context: Mapping[str, Any],
        task_config: Mapping[str, Any],
        task_output_relative_dir: str,
        tool_catalog_hash: str,
        model_configuration: Mapping[str, Any] | None = None,
        expert_backend_selection: Mapping[str, Any] | None = None,
        case_profile: str | None = None,
        gaussian_route_policy: Mapping[str, Any] | None = None,
        stage_architecture_version: str | None = None,
        hypothesis_import: Mapping[str, Any] | None = None,
    ) -> str:
        """在开始 Retrieval 前创建 TaskCreated 任务事实并冻结运行配置。

        输入是任务标识、科学问题、化学上下文、任务配置、输出相对目录、工具目录哈希，
        以及可选的模型配置、专家后端选择和案例配置。动作校验并规范化这些值后写入
        Working Memory；副作用是创建输出目录、持久化任务记录和 TaskCreated 事件，不
        调用模型、工具或在线服务。返回新任务的事件标识字符串；Working Memory 未启用、
        输入不符合合同、任务标识发生幂等冲突或数据库写入失败时抛出相应合同、幂等或
        数据库异常。
        """

        store = self._require_enabled()
        require_nonempty_string(task_id, "task_id")
        task_created_input = {
            "scientific_question": scientific_question,
            "chemistry_context": dict(chemistry_context),
            "task_config": dict(task_config),
            "task_output_relative_dir": task_output_relative_dir,
            "tool_catalog_hash": tool_catalog_hash,
        }
        if model_configuration is not None:
            task_created_input["model_configuration"] = dict(model_configuration)
        if expert_backend_selection is not None:
            task_created_input["expert_backend_selection"] = dict(
                expert_backend_selection
            )
        if case_profile is not None:
            task_created_input["case_profile"] = case_profile
        if gaussian_route_policy is not None:
            task_created_input["gaussian_route_policy"] = dict(gaussian_route_policy)
        if stage_architecture_version is not None:
            task_created_input["stage_architecture_version"] = (
                validate_stage_architecture_version(stage_architecture_version)
            )
        if hypothesis_import is not None:
            task_created_input["hypothesis_import"] = dict(hypothesis_import)
        if (
            isinstance(task_created_input.get("model_configuration"), Mapping)
            and task_created_input["model_configuration"].get("enabled") is True
            and expert_backend_selection is None
        ):
            raise ContractValidationError(
                "启用专家模型创建任务时必须同时提供 expert_backend_selection"
            )
        payload = validate_task_created_payload(task_created_input)
        expected_output_dir = f"outputs/runs/{task_id}"
        if task_output_relative_dir != expected_output_dir:
            raise ContractValidationError(
                f"task_output_relative_dir 必须逐值等于 {expected_output_dir}"
            )
        output_root = Path(self.scratch_root) / task_output_relative_dir
        output_root.mkdir(parents=True, exist_ok=True)
        event_key = event_key_task_created(task_id)
        with store.immediate_transaction() as conn:
            # 重复请求只有在问题、配置和证据完全一致时才能安全复用原事件。
            existing_id = self._existing_event_exact(
                conn,
                event_key=event_key,
                task_id=task_id,
                round_id=0,
                event_type="task_created",
                payload=payload,
                source_refs=[],
            )
            if existing_id is not None:
                self.task_id = task_id
                return existing_id
            if conn.execute("SELECT 1 FROM tasks WHERE task_id=?", (task_id,)).fetchone() is not None:
                raise IdempotencyConflict("task_id 已存在但没有匹配的 TaskCreated")
            now = utc_now()
            reaction_class = payload["chemistry_context"].get("reaction_type") or payload[
                "chemistry_context"
            ].get("reaction_class")
            initial_status = (
                "failed"
                if isinstance(payload.get("expert_backend_selection"), Mapping)
                and payload["expert_backend_selection"].get("status")
                == "startup_failed"
                else "running"
            )
            # 当前状态与创建事实放在同一事务中，数据库不会出现“有任务但无来源事实”。
            conn.execute(
                """
                INSERT INTO tasks(
                    task_id, scientific_question, reaction_class, chemistry_context_json,
                    execution_mode, status, version, stage_architecture_version,
                    created_at, updated_at
                ) VALUES(?, ?, ?, ?, ?, ?, 0, ?, ?, ?)
                """,
                (
                    task_id,
                    scientific_question,
                    reaction_class,
                    canonical_json(payload["chemistry_context"]),
                    "tool_catalog",
                    initial_status,
                    payload["stage_architecture_version"],
                    now,
                    now,
                ),
            )
            event_id, _ = self._insert_event(
                conn,
                event_key=event_key,
                task_id=task_id,
                round_id=0,
                event_type="task_created",
                payload=payload,
                source_refs=[],
                check_operation_occupancy=False,
            )
        self.task_id = task_id
        return event_id

    def create_investigation_task(
        self,
        *,
        task_id: str,
        scientific_question: str,
        chemistry_context: Mapping[str, Any],
        task_config: Mapping[str, Any],
        task_output_relative_dir: str,
        tool_catalog_hash: str,
        model_configuration: Mapping[str, Any] | None = None,
        expert_backend_selection: Mapping[str, Any] | None = None,
        case_profile: str | None = None,
        gaussian_route_policy: Mapping[str, Any] | None = None,
        hypothesis_import: Mapping[str, Any] | None = None,
    ) -> str:
        """显式创建 Investigation v2 任务，不读取 Planner driver 或环境变量。

        输入与 legacy ``create_task`` 相同；动作只把架构版本固定为
        ``investigation_v2``，并复用同一事务、TaskCreated 合同和输出目录校验。
        旧 Controller 继续调用 ``create_task`` 时不会自动切换到新阶段。
        """

        return self.create_task(
            task_id=task_id,
            scientific_question=scientific_question,
            chemistry_context=chemistry_context,
            task_config=task_config,
            task_output_relative_dir=task_output_relative_dir,
            tool_catalog_hash=tool_catalog_hash,
            model_configuration=model_configuration,
            expert_backend_selection=expert_backend_selection,
            case_profile=case_profile,
            gaussian_route_policy=gaussian_route_policy,
            hypothesis_import=hypothesis_import,
            stage_architecture_version=INVESTIGATION_STAGE_ARCHITECTURE_VERSION,
        )

    def create_hypothesis_batch(
        self,
        task_id: str,
        payload: Mapping[str, Any],
        source_refs: Sequence[str],
    ) -> str:
        """一次性保存整批假设，并为入选候选建立独立运行组合。

        任一候选身份或索引不合法时，整批都不会留下半成品。
        """

        store = self._require_enabled()
        with store.immediate_transaction() as conn:
            return self._create_hypothesis_batch_in_transaction(
                conn,
                task_id=task_id,
                payload=payload,
                source_refs=source_refs,
            )

    def begin_next_investigation_batch(
        self,
        task_id: str,
        payload: Mapping[str, Any],
        source_refs: Sequence[str],
    ) -> str:
        """在五个组合都完成 route marker 后原子创建下一 Investigation batch。

        输入是 investigation_v2 task 的下一批完整 BatchCreated 正文和 hypothesis
        manifest 来源；动作在同一 SQLite 事务核对上一批恰有五个当前 marker，再追加
        BatchCreated 及五个无 thread 的 SessionStarted 事实。返回 BatchCreated 事件 ID；
        架构错误、批次跳跃、marker 不完整、来源过期或任一 session 建立失败都会整体
        回滚，不留下只有批次没有 session 的半成品。
        """

        store = self._require_enabled()
        with store.immediate_transaction() as conn:
            return self._create_hypothesis_batch_in_transaction(
                conn,
                task_id=task_id,
                payload=payload,
                source_refs=source_refs,
                require_previous_route_gate=True,
            )

    def _prepare_hypothesis_batch_creation_context(
        self,
        conn,
        *,
        task_id: str,
        payload: Mapping[str, Any],
        require_previous_route_gate: bool,
    ) -> tuple[Dict[str, Any], Dict[str, Any], bool, Dict[str, Any]]:
        """读取任务事实并规范化待写入的 HypothesisBatchCreated 正文。

        输入是已持有写锁的事务连接、task、批次正文和是否要求下一批路由门禁；动作
        核对任务投影、架构版本、协议、因果 Restart 和首批科学问题，返回任务创建事实、
        task 配置、是否 v2 以及规范化 batch 正文。任何漂移都抛出合同错误并由外层回滚。
        """

        task_created = self._task_created_payload(conn, task_id)
        self._ensure_task_projection_matches_created(conn, task_id, task_created)
        task_config = task_created["task_config"]
        investigation_v2 = (
            task_created["stage_architecture_version"]
            == INVESTIGATION_STAGE_ARCHITECTURE_VERSION
        )
        if require_previous_route_gate and not investigation_v2:
            raise ContractValidationError("只有 investigation_v2 task 可以开始下一批")
        raw_payload = dict(payload)
        if investigation_v2:
            if (
                "investigation_protocol_version" in raw_payload
                and raw_payload["investigation_protocol_version"]
                != INVESTIGATION_PROTOCOL_VERSION
            ):
                raise ContractValidationError("BatchCreated Investigation 协议版本漂移")
            raw_payload["investigation_protocol_version"] = INVESTIGATION_PROTOCOL_VERSION
        top_n_hypotheses = task_config.get(
            "top_n_workflows", task_config.get("top_n_hypotheses")
        )
        expected_combination_count = (
            get_investigation_combination_count(task_config)
            if investigation_v2
            else None
        )
        value = validate_batch_created_payload(
            raw_payload,
            top_n_workflows=top_n_hypotheses,
            required_selected_count=expected_combination_count,
        )
        pending_restart = self._pending_task_hypothesis_restart(conn, task_id)
        required_causal_source = (
            None if pending_restart is None else pending_restart["restart_record_id"]
        )
        if value["causal_source_ref"] != required_causal_source:
            raise ContractValidationError("BatchCreated 必须准确消费当前任务级 hypothesis Restart")
        if (
            value["batch_index"] == 1
            and value["question_snapshot"]["scientific_question"]
            != task_created["scientific_question"]
        ):
            raise ContractValidationError("首批 scientific question 与 TaskCreated 不一致")
        return task_created, task_config, investigation_v2, value

    def _find_hypothesis_batch_initial_replay(
        self,
        conn,
        *,
        task_id: str,
        value: Mapping[str, Any],
        source_refs: Sequence[str],
    ) -> str | None:
        """查找首批 BatchCreated 的完全相同重放。

        输入是事务连接、task、已规范化 batch 正文和来源；动作按稳定事件键核对原事件
        的完整内容，返回原事件 ID 或 None。稳定键内容冲突时抛出幂等冲突，不修改索引。
        """

        if value["batch_index"] != 1:
            return None
        return self._existing_event_exact(
            conn,
            event_key=event_key_batch_created(task_id, value["batch_id"]),
            task_id=task_id,
            round_id=0,
            event_type="hypothesis_batch_created",
            payload=value,
            source_refs=source_refs,
        )

    def _validate_hypothesis_batch_sequence_and_sources(
        self,
        conn,
        *,
        task_id: str,
        value: Dict[str, Any],
        task_config: Mapping[str, Any],
        investigation_v2: bool,
        require_previous_route_gate: bool,
    ) -> Dict[str, Any]:
        """核对 batch 编号、上一批来源和 Investigation 下一批五项 marker 门禁。

        输入是事务连接、task、规范化 batch、task 配置及架构标记；动作只读并核对上一
        批及当前 marker，必要时补齐 generation_source 后返回新正文。编号或来源不合法
        时抛出合同错误，不追加事件。
        """

        if value["batch_index"] > task_config["max_hypothesis_batches"]:
            raise ContractValidationError("batch_index 超过 max_hypothesis_batches")
        if value["batch_index"] == 1:
            prior = conn.execute(
                "SELECT 1 FROM hypothesis_batches WHERE task_id=? LIMIT 1", (task_id,)
            ).fetchone()
            if prior is not None:
                raise CASConflict("初始 batch 已存在", expected_version=None, current_version=None)
            return value
        prior = conn.execute(
            "SELECT batch_id,created_event_id FROM hypothesis_batches "
            "WHERE task_id=? AND batch_index=?",
            (task_id, value["batch_index"] - 1),
        ).fetchone()
        previous_batch_id = value["generation_source"]["previous_batch_id"]
        if prior is None or prior["batch_id"] != previous_batch_id:
            raise ContractValidationError("后续 batch 必须逐值引用直接上一批")
        if investigation_v2:
            # 先用不可变事件重建校验当前索引，再授权读取五项 marker；这样 marker
            # 的事件身份被篡改时不会被门禁误认为合法，并且事务会回滚重建结果。
            self._assert_investigation_index_matches_event_rebuild(conn, task_id)
            marker_refs = self._assert_investigation_batch_ready_for_next(
                conn, task_id, previous_batch_id
            )
            source_record_ids = set(value["generation_source"]["source_record_ids"])
            source_record_ids.update(marker_refs)
            source_record_ids.add(prior["created_event_id"])
            generation_source = dict(value["generation_source"])
            generation_source["source_record_ids"] = sorted(source_record_ids)
            value = validate_batch_created_payload(
                {**value, "generation_source": generation_source},
                top_n_workflows=task_config.get(
                    "top_n_workflows", task_config.get("top_n_hypotheses")
                ),
                required_selected_count=get_investigation_combination_count(
                    task_config
                ),
            )
        elif require_previous_route_gate:
            self._validate_source_refs(
                conn, task_id, value["generation_source"]["source_record_ids"]
            )
        self._validate_source_refs(
            conn, task_id, value["generation_source"]["source_record_ids"]
        )
        return value

    def _insert_hypothesis_batch_event_and_projection(
        self,
        conn,
        *,
        task_id: str,
        value: Mapping[str, Any],
        source_refs: Sequence[str],
        investigation_v2: bool,
        create_investigation_sessions: bool,
    ) -> str:
        """追加 BatchCreated 事实并在需要时建立五个 Investigation session。

        输入是事务连接、task、规范化正文、finalized manifest 来源和 v2 标记；动作在同
        一事务追加事件、批次投影及五个 session，返回事件 ID。来源、数据库或投影失败时
        抛出异常并由外层事务整体回滚。
        """

        if len(source_refs) != 1 or "/" not in source_refs[0]:
            raise ContractValidationError(
                "HypothesisBatchCreated.source_refs 必须恰含 hypothesis finalized manifest"
            )
        batch_event_key = event_key_batch_created(task_id, value["batch_id"])
        existing = self._existing_event_exact(
            conn,
            event_key=batch_event_key,
            task_id=task_id,
            round_id=0,
            event_type="hypothesis_batch_created",
            payload=value,
            source_refs=source_refs,
        )
        if existing is not None:
            return existing
        event_id, inserted = self._insert_event(
            conn,
            event_key=batch_event_key,
            task_id=task_id,
            round_id=0,
            event_type="hypothesis_batch_created",
            payload=value,
            source_refs=source_refs,
            check_operation_occupancy=True,
        )
        if inserted:
            created_at = conn.execute(
                "SELECT created_at FROM memory_events WHERE event_id=?", (event_id,)
            ).fetchone()["created_at"]
            insert_batch_projection(
                conn,
                task_id=task_id,
                event_id=event_id,
                payload=value,
                created_at=created_at,
            )
            if investigation_v2 and create_investigation_sessions:
                self._append_investigation_sessions_for_batch(
                    conn,
                    task_id=task_id,
                    batch_event_id=event_id,
                    payload=value,
                    created_at=created_at,
                )
        return event_id

    def _create_hypothesis_batch_in_transaction(
        self,
        conn,
        *,
        task_id: str,
        payload: Mapping[str, Any],
        source_refs: Sequence[str],
        require_previous_route_gate: bool = False,
    ) -> str:
        """在调用方已持有写锁时写入 batch 及其 Investigation session。

        输入是事务连接、任务、完整 batch 正文和 finalized manifest 来源；动作复用
        legacy/v2 批次合同，v2 首批或下一批都原子建立五个 session，必要时先核对上一批
        marker。返回 BatchCreated 事件 ID；所有校验或索引失败由外层事务回滚。
        """

        _, task_config, investigation_v2, value = (
            self._prepare_hypothesis_batch_creation_context(
                conn,
                task_id=task_id,
                payload=payload,
                require_previous_route_gate=require_previous_route_gate,
            )
        )
        existing = self._find_hypothesis_batch_initial_replay(
            conn,
            task_id=task_id,
            value=value,
            source_refs=source_refs,
        )
        if existing is not None:
            return existing
        value = self._validate_hypothesis_batch_sequence_and_sources(
            conn,
            task_id=task_id,
            value=value,
            task_config=task_config,
            investigation_v2=investigation_v2,
            require_previous_route_gate=require_previous_route_gate,
        )
        return self._insert_hypothesis_batch_event_and_projection(
            conn,
            task_id=task_id,
            value=value,
            source_refs=source_refs,
            investigation_v2=investigation_v2,
            create_investigation_sessions=(
                task_config.get("run_mode", "end_to_end") != "top5_selection"
            ),
        )

    def _assert_investigation_batch_ready_for_next(
        self,
        conn,
        task_id: str,
        batch_id: str,
    ) -> list[str]:
        """在当前写事务内核对下一批所需的五个完整 marker。

        输入是已持有 SQLite 写锁的连接、task 和上一批身份；动作按稳定组合身份核对
        五个入选组合、session、当前 result/review、marker 及 route finding，返回五个
        marker 事件 ID，供下一批事实保存因果来源。
        数量不足、身份重复、来源过期或任一 gate 未通过时抛出合同错误；方法只读当前
        索引和不可变事件，不追加事实。
        """

        task_created = self._task_created_payload(conn, task_id)
        expected_count = get_investigation_combination_count(task_created["task_config"])
        selected = conn.execute(
            "SELECT c.combination_id FROM combinations c JOIN hypothesis_candidates hc "
            "ON hc.batch_id=c.batch_id AND hc.combination_id=c.combination_id "
            "WHERE c.task_id=? AND c.batch_id=? AND hc.selection_status='selected' "
            "ORDER BY c.combination_index,c.combination_id",
            (task_id, batch_id),
        ).fetchall()
        combination_ids = [row["combination_id"] for row in selected]
        if len(combination_ids) != expected_count or len(set(combination_ids)) != expected_count:
            raise ContractValidationError(
                "下一 Investigation batch 必须来自上一批全部入选组合"
            )
        sessions = conn.execute(
            "SELECT combination_id,route_marker,phase,latest_result_event_id,"
            "latest_review_event_id,pending_repair_event_id "
            "FROM investigation_sessions WHERE task_id=? AND batch_id=?",
            (task_id, batch_id),
        ).fetchall()
        markers = conn.execute(
            "SELECT combination_id,marker_event_id,result_event_ref,review_event_ref "
            "FROM investigation_route_markers WHERE task_id=? AND batch_id=?",
            (task_id, batch_id),
        ).fetchall()
        batch_gate = evaluate_investigation_batch_route_gate(
            conn,
            task_id=task_id,
            batch_id=batch_id,
        )
        if not batch_gate["allowed"]:
            raise ContractValidationError(
                batch_gate["reason"] or "上一批尚未满足整批 route gate"
            )
        return self._collect_revise_hypothesis_next_batch_marker_ids(
            combination_ids, sessions, markers
        )

    def _append_investigation_sessions_for_batch(
        self,
        conn,
        *,
        task_id: str,
        batch_event_id: str,
        payload: Mapping[str, Any],
        created_at: str,
    ) -> None:
        """在 BatchCreated 事务中为五个入选组合追加无 thread 的 session 事实。

        输入是同一事务中的 v2 BatchCreated 正文、事件身份和时间；动作为每个入选
        combination 生成稳定 session_id、独立 workspace/scope 引用和 SessionStarted
        事件，随后建立当前索引。返回值为空；候选数、事件重复、合同或数据库错误会
        回滚整批 BatchCreated，不预知或伪造 SDK thread，也不保存 transcript。
        """

        selected = [
            item
            for item in payload["candidates"]
            if item["selection_status"] == "selected"
        ]
        task_created = self._task_created_payload(conn, task_id)
        expected_count = get_investigation_combination_count(task_created["task_config"])
        if len(selected) != expected_count:
            raise ContractValidationError(
                "Investigation v2 BatchCreated 必须为每个入选组合原子创建 session"
            )
        for candidate in sorted(selected, key=lambda item: item["combination_index"]):
            combination_id = candidate["combination_id"]
            session_payload = validate_investigation_session_payload(
                {
                    "task_id": task_id,
                    "session_id": investigation_session_id(
                        task_id, payload["batch_id"], combination_id
                    ),
                    "batch_id": payload["batch_id"],
                    "combination_id": combination_id,
                    "codex_thread_id": None,
                    "workspace_ref": (
                        f"investigation-workspace:{task_id}:{payload['batch_id']}:{combination_id}"
                    ),
                    "scope_id": (
                        f"investigation-scope:{task_id}:{payload['batch_id']}:{combination_id}"
                    ),
                    "operation_instance_id": (
                        f"investigation-session:{task_id}:{payload['batch_id']}:{combination_id}"
                    ),
                    "investigation_protocol_version": INVESTIGATION_PROTOCOL_VERSION,
                }
            )
            event_id, inserted = self._insert_event(
                conn,
                event_key=event_key_investigation_session(
                    task_id, payload["batch_id"], combination_id
                ),
                task_id=task_id,
                round_id=0,
                event_type="investigation_session_started",
                payload=session_payload,
                source_refs=[batch_event_id],
                check_operation_occupancy=True,
            )
            if inserted:
                insert_investigation_session_projection(
                    conn,
                    task_id=task_id,
                    event_id=event_id,
                    payload=session_payload,
                    created_at=created_at,
                )

    def claim_combination(
        self,
        task_id: str,
        combination_id: str,
        operation: str,
        operation_instance_id: str,
        expected_state: str,
        expected_version: int,
        *,
        current_request_ref: str,
        step_index: int,
        expected_identity: Optional[Mapping[str, Any]] = None,
        causal_source_ref: Optional[str] = None,
    ) -> Dict[str, Any]:
        """在真正调用工具前，取得当前步骤的唯一执行权。

        成功后把工具调用身份、步骤位置和已经冻结的请求清单一起写入组合状态。
        其他进程若持有旧版本就必须停止，从而避免一次网络重试或进程重连重复启动
        昂贵的 Gaussian 计算。
        """

        store = self._require_enabled()
        require_nonempty_string(operation_instance_id, "operation_instance_id")
        if operation != "execute_step" or expected_state != "running":
            raise ContractValidationError("持久化 claim 只用于 state=running 的 execute_step")
        require_nonempty_string(current_request_ref, "current_request_ref")
        if isinstance(step_index, bool) or not isinstance(step_index, int) or step_index < 0:
            raise ContractValidationError("step_index 必须是非负 JSON integer")
        expected_identity = dict(expected_identity or {})
        allowed_identity = {"workflow_id", "workflow_version", "attempt_index", "execution_id"}
        if set(expected_identity) - allowed_identity:
            raise ContractValidationError("expected_identity 含未知字段")
        with store.immediate_transaction() as conn:
            self._assert_task_writable(conn, task_id)
            self._assert_operation_available(conn, task_id, operation_instance_id)
            self._validate_source_refs(conn, task_id, [current_request_ref])
            # 组合读取和执行权更新处于同一写事务；从身份核对到写入完成之间，另一
            # 个执行者无法插入相同步骤的声明。
            row = conn.execute(
                "SELECT * FROM combinations WHERE task_id=? AND combination_id=?",
                (task_id, combination_id),
            ).fetchone()
            if row is None:
                raise ContractValidationError("未知 combination")
            runtime = self._decode_json(row["runtime_checkpoint_json"], field="runtime_checkpoint_json")
            # 工具步骤完成后没有再写一条单独的业务事件，已发布的步骤清单就是该调用
            # 身份已经使用过的永久证据。再次声明执行权前必须检查整个完成前缀，不能
            # 只查看当前是否有人正在运行。
            for prior_ref in runtime.get("step_manifest_refs", []):
                prior = self._read_finalized_manifest(conn, task_id, prior_ref)
                if prior.get("operation_instance_id") == operation_instance_id:
                    raise OperationConflict(
                        "execute_step operation 已在完成步骤中使用"
                    )
            identity_matches = all(row[field] == value for field, value in expected_identity.items())
            if (
                row["version"] != expected_version
                or row["state"] != expected_state
                or row["causal_source_ref"] != causal_source_ref
                or not identity_matches
                or runtime.get("next_step_index") != step_index
                or runtime.get("current_operation_instance_id") is not None
                or runtime.get("current_request_ref") is not None
                or runtime.get("checkpoint_id") is not None
            ):
                raise CASConflict(
                    "combination claim 的 CAS 前提已变化",
                    expected_version=expected_version,
                    current_version=row["version"],
                    expected_state=expected_state,
                    current_state=row["state"],
                )
            # 请求清单和工具调用身份一同保存。后续等待、完成或暂停都必须逐值复用，
            # 防止执行者拿着旧声明提交另一份输入的结果。
            runtime["current_operation_instance_id"] = operation_instance_id
            runtime["current_request_ref"] = current_request_ref
            cursor = conn.execute(
                """
                UPDATE combinations SET runtime_checkpoint_json=?, version=version+1, updated_at=?
                WHERE task_id=? AND combination_id=? AND state=? AND version=?
                """,
                (
                    canonical_json(runtime),
                    utc_now(),
                    task_id,
                    combination_id,
                    expected_state,
                    expected_version,
                ),
            )
            if cursor.rowcount != 1:
                raise CASConflict(
                    "combination claim 输掉并发 CAS",
                    expected_version=expected_version,
                    current_version=row["version"],
                    expected_state=expected_state,
                    current_state=row["state"],
                )
            claimed = conn.execute(
                "SELECT * FROM combinations WHERE task_id=? AND combination_id=?",
                (task_id, combination_id),
            ).fetchone()
            return combination_row(claimed)

    def rebind_execution_request(
        self,
        *,
        task_id: str,
        combination_id: str,
        operation_instance_id: str,
        step_index: int,
        previous_request_ref: str,
        request_ref: str,
        expected_version: int,
        expected_identity: Mapping[str, Any],
        causal_source_ref: Optional[str],
    ) -> Dict[str, Any]:
        """在 authored files 物化后以 CAS 更新当前 claim 的实际 request 引用。

        文件写入必须发生在原 claim 成功之后；本动作只在验证新 request artifact 已
        进入任务目录后更新运行游标，并写入不可变历史。验证或工具提交若在此之前
        失败，claim 仍指向旧的逻辑 request，调用方不得产生外部工具副作用。
        """

        store = self._require_enabled()
        require_nonempty_string(operation_instance_id, "operation_instance_id")
        require_nonempty_string(previous_request_ref, "previous_request_ref")
        require_nonempty_string(request_ref, "request_ref")
        if isinstance(step_index, bool) or not isinstance(step_index, int) or step_index < 0:
            raise ContractValidationError("step_index 必须是非负 JSON integer")
        expected_identity = dict(expected_identity)
        if set(expected_identity) != {
            "workflow_id",
            "workflow_version",
            "attempt_index",
            "execution_id",
        }:
            raise ContractValidationError("expected_identity 必须包含完整执行身份")
        with store.immediate_transaction() as conn:
            self._assert_task_writable(conn, task_id)
            self._validate_source_refs(conn, task_id, [request_ref])
            row = conn.execute(
                "SELECT * FROM combinations WHERE task_id=? AND combination_id=?",
                (task_id, combination_id),
            ).fetchone()
            if row is None:
                raise ContractValidationError("request rebound 引用了未知 combination")
            workflow = self._decode_json(
                row["current_workflow_json"], field="current_workflow_json"
            )
            steps = workflow.get("steps") if isinstance(workflow, Mapping) else None
            step = steps[step_index] if isinstance(steps, list) and step_index < len(steps) else None
            if not isinstance(step, Mapping):
                raise ContractValidationError("request rebound 对应的 workflow step 不存在")
            payload = validate_execution_request_rebound_payload(
                {
                    "combination_id": combination_id,
                    "execution_id": expected_identity["execution_id"],
                    "workflow_version": expected_identity["workflow_version"],
                    "step_id": step.get("step_id"),
                    "step_index": step_index,
                    "operation_instance_id": operation_instance_id,
                    "previous_request_ref": previous_request_ref,
                    "request_ref": request_ref,
                    "causal_source_ref": causal_source_ref,
                }
            )
            event_key = event_key_execution_request_rebound(
                task_id,
                payload["execution_id"],
                operation_instance_id,
                request_ref,
            )
            existing = self._existing_event_exact(
                conn,
                event_key=event_key,
                task_id=task_id,
                round_id=row["attempt_index"],
                event_type="execution_request_rebound",
                payload=payload,
                source_refs=[request_ref],
            )
            if existing is not None:
                return combination_row(row)
            runtime = self._decode_json(
                row["runtime_checkpoint_json"], field="runtime_checkpoint_json"
            )
            identity_matches = all(
                row[field] == value for field, value in expected_identity.items()
            )
            if (
                row["version"] != expected_version
                or row["state"] != "running"
                or row["causal_source_ref"] != causal_source_ref
                or not identity_matches
                or runtime.get("next_step_index") != step_index
                or runtime.get("current_operation_instance_id") != operation_instance_id
                or runtime.get("current_request_ref") != previous_request_ref
                or runtime.get("checkpoint_id") is not None
            ):
                raise CASConflict(
                    "execution request rebound 的 CAS 前提已变化",
                    expected_version=expected_version,
                    current_version=row["version"],
                    expected_state="running",
                    current_state=row["state"],
                )
            self._insert_event(
                conn,
                event_key=event_key,
                task_id=task_id,
                round_id=row["attempt_index"],
                event_type="execution_request_rebound",
                payload=payload,
                source_refs=[request_ref],
                check_operation_occupancy=False,
            )
            runtime["current_request_ref"] = request_ref
            cursor = conn.execute(
                """
                UPDATE combinations SET runtime_checkpoint_json=?, version=version+1,
                    updated_at=?
                WHERE task_id=? AND combination_id=? AND state='running' AND version=?
                """,
                (
                    canonical_json(runtime),
                    utc_now(),
                    task_id,
                    combination_id,
                    expected_version,
                ),
            )
            if cursor.rowcount != 1:
                raise CASConflict(
                    "execution request rebound 输掉并发 CAS",
                    expected_version=expected_version,
                    current_version=row["version"],
                    expected_state="running",
                    current_state=row["state"],
                )
            updated = conn.execute(
                "SELECT * FROM combinations WHERE task_id=? AND combination_id=?",
                (task_id, combination_id),
            ).fetchone()
            return combination_row(updated)

    @staticmethod
    def _require_claim(row, *, operation_instance_id: str) -> None:
        """提交等待态或结果前，再次确认工具调用仍持有当前执行权。"""

        try:
            runtime = json.loads(row["runtime_checkpoint_json"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise ContractValidationError("runtime claim JSON 已损坏") from exc
        if (
            runtime.get("current_operation_instance_id") != operation_instance_id
        ):
            raise CASConflict(
                "operation claim 已改变",
                expected_version=None,
                current_version=row["version"],
                current_state=row["state"],
            )

    @staticmethod
    def _manifest_ref_path(value: Any) -> Optional[str]:
        """从正式文件引用中取得任务相对路径，并兼容早期测试使用的字符串。"""

        if isinstance(value, str) and value:
            return value
        if isinstance(value, Mapping):
            path = value.get("relative_path")
            if isinstance(path, str) and path:
                return path
        return None

    def record_execution_checkpoint(
        self,
        task_id: str,
        payload: Mapping[str, Any],
        source_refs: Sequence[str],
        *,
        expected_version: int,
    ) -> str:
        """记录已经由监督器接管的外部作业，并让组合进入可恢复等待。

        第一份检查点必须在监督器接管作业后、开始轮询前提交。此后每次继续等待都
        复用同一个工具调用和请求文件，只递增当前执行的检查点序号；进程中断后便
        能继续观察原作业，而不是再次提交计算。
        """

        store = self._require_enabled()
        value = validate_checkpoint_payload(dict(payload))
        if len(source_refs) != 1 or "/" not in source_refs[0]:
            raise ContractValidationError(
                "ExecutionCheckpointed.source_refs 必须恰含 checkpoint manifest"
            )
        event_key = event_key_checkpoint(
            task_id,
            value["execution_id"],
            value["checkpoint_index"],
        )
        with store.immediate_transaction() as conn:
            existing = self._existing_event_exact(
                conn,
                event_key=event_key,
                task_id=task_id,
                round_id=self._attempt_index(
                    conn, task_id, value["combination_id"]
                ),
                event_type="execution_checkpointed",
                payload=value,
                source_refs=source_refs,
            )
            if existing is not None:
                return existing
            row = conn.execute(
                "SELECT * FROM combinations WHERE task_id=? AND combination_id=?",
                (task_id, value["combination_id"]),
            ).fetchone()
            if row is None:
                raise ContractValidationError("checkpoint 引用未知 combination")
            self._require_claim(
                row,
                operation_instance_id=value["operation_instance_id"],
            )
            runtime = self._decode_json(
                row["runtime_checkpoint_json"], field="runtime_checkpoint_json"
            )
            if (
                row["version"] != expected_version
                or row["state"] != "running"
                or row["execution_id"] != value["execution_id"]
                or row["workflow_version"] != value["workflow_version"]
                or row["causal_source_ref"] != value["causal_source_ref"]
                or runtime.get("next_step_index") != value["step_index"]
                or runtime.get("checkpoint_index", 0) + 1
                != value["checkpoint_index"]
            ):
                raise CASConflict(
                    "execution checkpoint 的 CAS 前提已变化",
                    expected_version=expected_version,
                    current_version=row["version"],
                    expected_state="running",
                    current_state=row["state"],
                )
            manifest = self._read_finalized_manifest(
                conn, task_id, source_refs[0]
            )
            for field in (
                "operation_instance_id",
                "execution_id",
                "workflow_version",
                "step_id",
                "step_index",
                "checkpoint_index",
            ):
                if manifest.get(field) != value[field]:
                    raise ContractValidationError(
                        f"checkpoint manifest.{field} 与提交 payload 不一致"
                    )
            request_path = self._manifest_ref_path(manifest.get("request_ref"))
            if request_path != runtime.get("current_request_ref"):
                raise ContractValidationError(
                    "checkpoint manifest 必须复用当前 claim 的 request_ref"
                )
            event_id, _ = self._insert_event(
                conn,
                event_key=event_key,
                task_id=task_id,
                round_id=row["attempt_index"],
                event_type="execution_checkpointed",
                payload=value,
                source_refs=source_refs,
                check_operation_occupancy=False,
            )
            runtime["checkpoint_id"] = event_id
            runtime["checkpoint_index"] = value["checkpoint_index"]
            runtime["checkpoint_manifest_ref"] = source_refs[0]
            cursor = conn.execute(
                """
                UPDATE combinations SET state='waiting_execution',
                    runtime_checkpoint_json=?, version=version+1, updated_at=?
                WHERE task_id=? AND combination_id=? AND state='running' AND version=?
                """,
                (
                    canonical_json(runtime),
                    utc_now(),
                    task_id,
                    value["combination_id"],
                    expected_version,
                ),
            )
            if cursor.rowcount != 1:
                raise CASConflict(
                    "execution checkpoint 输掉并发 CAS",
                    expected_version=expected_version,
                    current_version=row["version"],
                    expected_state="running",
                    current_state=row["state"],
                )
            self._reduce_task_status_in_transaction(conn, task_id)
            return event_id

    def record_execution_resumed(
        self,
        task_id: str,
        payload: Mapping[str, Any],
        source_refs: Sequence[str],
        *,
        expected_version: int,
    ) -> str:
        """从已提交检查点重新接管同一外部作业，不创建第二次工具调用。

        恢复记录只把组合从等待态带回可推进态，原执行、步骤、请求和监督器句柄均
        保持不变。真正的工具结果仍由后续完成或再次等待操作提交。
        """

        store = self._require_enabled()
        value = validate_resume_payload(dict(payload))
        if list(source_refs) != [value["checkpoint_id"]]:
            raise ContractValidationError(
                "ExecutionResumed.source_refs 必须恰含来源 checkpoint event"
            )
        event_key = event_key_resume(task_id, value["checkpoint_id"])
        with store.immediate_transaction() as conn:
            existing = self._existing_event_exact(
                conn,
                event_key=event_key,
                task_id=task_id,
                round_id=self._attempt_index(
                    conn, task_id, value["combination_id"]
                ),
                event_type="execution_resumed",
                payload=value,
                source_refs=source_refs,
            )
            if existing is not None:
                return existing
            checkpoint = conn.execute(
                "SELECT event_type, payload_json FROM memory_events "
                "WHERE task_id=? AND event_id=?",
                (task_id, value["checkpoint_id"]),
            ).fetchone()
            if checkpoint is None or checkpoint["event_type"] != "execution_checkpointed":
                raise ContractValidationError("checkpoint_id 不属于当前任务的检查点")
            checkpoint_payload = self._decode_json(
                checkpoint["payload_json"], field="checkpoint payload"
            )
            for field in (
                "combination_id",
                "execution_id",
                "workflow_version",
                "step_id",
                "step_index",
                "operation_instance_id",
            ):
                if checkpoint_payload.get(field) != value[field]:
                    raise ContractValidationError(
                        "ExecutionResumed 与来源 checkpoint identity 不一致"
                    )
            row = conn.execute(
                "SELECT * FROM combinations WHERE task_id=? AND combination_id=?",
                (task_id, value["combination_id"]),
            ).fetchone()
            if row is None:
                raise ContractValidationError("resume 引用未知 combination")
            runtime = self._decode_json(
                row["runtime_checkpoint_json"], field="runtime_checkpoint_json"
            )
            if (
                row["version"] != expected_version
                or row["state"] != "waiting_execution"
                or row["execution_id"] != value["execution_id"]
                or row["workflow_version"] != value["workflow_version"]
                or runtime.get("checkpoint_id") != value["checkpoint_id"]
                or runtime.get("current_operation_instance_id")
                != value["operation_instance_id"]
                or runtime.get("next_step_index") != value["step_index"]
            ):
                raise CASConflict(
                    "execution resume 的 CAS 前提已变化",
                    expected_version=expected_version,
                    current_version=None if row is None else row["version"],
                    expected_state="waiting_execution",
                    current_state=None if row is None else row["state"],
                )
            event_id, _ = self._insert_event(
                conn,
                event_key=event_key,
                task_id=task_id,
                round_id=row["attempt_index"],
                event_type="execution_resumed",
                payload=value,
                source_refs=source_refs,
                check_operation_occupancy=False,
            )
            cursor = conn.execute(
                """
                UPDATE combinations SET state='running', version=version+1, updated_at=?
                WHERE task_id=? AND combination_id=?
                  AND state='waiting_execution' AND version=?
                """,
                (
                    utc_now(),
                    task_id,
                    value["combination_id"],
                    expected_version,
                ),
            )
            if cursor.rowcount != 1:
                raise CASConflict(
                    "execution resume 输掉并发 CAS",
                    expected_version=expected_version,
                    current_version=row["version"],
                    expected_state="waiting_execution",
                    current_state=row["state"],
                )
            self._reduce_task_status_in_transaction(conn, task_id)
            return event_id

    def complete_execution_step(
        self,
        *,
        task_id: str,
        combination_id: str,
        operation_instance_id: str,
        step_manifest_ref: str,
        step_index: int,
        expected_version: int,
        expected_identity: Mapping[str, Any],
        causal_source_ref: Optional[str],
        gaussian_code_receipt: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """以当前执行权提交一个完成步骤，并原子推进工作流游标。

        单个工具步骤完成并不代表假设已经得到支持或否定，所以这里不会写入完整
        科学尝试，也不会消费人工恢复授权。不过它会写一条不可修改的步骤完成事实，
        这样投影恰好在两个步骤之间丢失时也不会重复执行已经成功的工具。
        """

        store = self._require_enabled()
        require_nonempty_string(operation_instance_id, "operation_instance_id")
        if isinstance(step_index, bool) or not isinstance(step_index, int) or step_index < 0:
            raise ContractValidationError("step_index 必须是非负 JSON integer")
        expected_identity = dict(expected_identity)
        if set(expected_identity) != {
            "workflow_id",
            "workflow_version",
            "attempt_index",
            "execution_id",
        }:
            raise ContractValidationError("expected_identity 必须包含完整执行身份")
        with store.immediate_transaction() as conn:
            row = conn.execute(
                "SELECT * FROM combinations WHERE task_id=? AND combination_id=?",
                (task_id, combination_id),
            ).fetchone()
            if row is None:
                raise ContractValidationError("completed step 引用未知 combination")
            runtime = self._decode_json(
                row["runtime_checkpoint_json"], field="runtime_checkpoint_json"
            )
            manifest = self._read_finalized_manifest(
                conn, task_id, step_manifest_ref
            )
            if manifest.get("status") != "completed":
                raise ContractValidationError(
                    "complete_execution_step 只接受 completed step manifest"
                )
            for field, expected in (
                ("operation_instance_id", operation_instance_id),
                ("execution_id", expected_identity["execution_id"]),
                ("workflow_version", expected_identity["workflow_version"]),
                ("step_index", step_index),
                ("causal_source_ref", causal_source_ref),
            ):
                if manifest.get(field) != expected:
                    raise ContractValidationError(
                        f"step manifest.{field} 与当前 claim 不一致"
                    )
            prefix = manifest.get("step_manifest_refs")
            if (
                not isinstance(prefix, list)
                or len(prefix) != step_index
                or any(not isinstance(item, str) or not item for item in prefix)
            ):
                raise ContractValidationError(
                    "completed step manifest 必须准确保存提交前的步骤前缀"
                )
            for prior_ref in prefix:
                validate_task_relative_path(prior_ref, "step_manifest_refs")
            event_payload = validate_execution_step_completed_payload(
                {
                    "combination_id": combination_id,
                    "execution_id": expected_identity["execution_id"],
                    "workflow_version": expected_identity["workflow_version"],
                    "step_id": manifest.get("step_id"),
                    "step_index": step_index,
                    "operation_instance_id": operation_instance_id,
                    "causal_source_ref": causal_source_ref,
                }
            )
            event_key = event_key_execution_step_completed(
                task_id,
                expected_identity["execution_id"],
                operation_instance_id,
            )
            # exact replay 必须查不可变事件。当前组合可能已经推进到后续步骤甚至终态，
            # 此时再用可变游标判断会把一个合法重放误报为并发冲突。
            existing_id = self._existing_event_exact(
                conn,
                event_key=event_key,
                task_id=task_id,
                round_id=expected_identity["attempt_index"],
                event_type="execution_step_completed",
                payload=event_payload,
                source_refs=[step_manifest_ref],
            )
            if existing_id is not None:
                return combination_row(row)
            identity_matches = all(
                row[field] == value for field, value in expected_identity.items()
            )
            if (
                row["version"] != expected_version
                or row["state"] != "running"
                or row["causal_source_ref"] != causal_source_ref
                or not identity_matches
                or runtime.get("next_step_index") != step_index
                or runtime.get("current_operation_instance_id")
                != operation_instance_id
            ):
                raise CASConflict(
                    "completed step 的 CAS 前提已变化",
                    expected_version=expected_version,
                    current_version=row["version"],
                    expected_state="running",
                    current_state=row["state"],
                )
            request_path = self._manifest_ref_path(manifest.get("request_ref"))
            if request_path is not None and request_path != runtime.get(
                "current_request_ref"
            ):
                raise ContractValidationError(
                    "step manifest 必须复用当前 claim 的 request_ref"
                )
            if prefix != runtime.get("step_manifest_refs", []):
                raise ContractValidationError(
                    "completed step manifest 的步骤前缀与当前执行游标不一致"
                )
            workflow = self._decode_json(
                row["current_workflow_json"], field="current_workflow_json"
            )
            steps = workflow.get("steps") if isinstance(workflow, Mapping) else None
            if not isinstance(steps, list) or step_index >= len(steps):
                raise ContractValidationError("completed step 对应的 workflow step 不存在")
            current_step = steps[step_index]
            if manifest.get("action_id") == "generate_gaussian_code" and (
                not isinstance(current_step, Mapping)
                or current_step.get("action_id") != "generate_gaussian_code"
            ):
                raise ContractValidationError(
                    "Gaussian codegen manifest 与当前 workflow step 不一致"
                )
            _, inserted = self._insert_event(
                conn,
                event_key=event_key,
                task_id=task_id,
                round_id=expected_identity["attempt_index"],
                event_type="execution_step_completed",
                payload=event_payload,
                source_refs=[step_manifest_ref],
                check_operation_occupancy=True,
            )
            if not inserted:
                raise OperationConflict("步骤完成事实的并发状态不一致")
            runtime["step_manifest_refs"] = [
                *runtime.get("step_manifest_refs", []),
                step_manifest_ref,
            ]
            runtime["next_step_index"] = step_index + 1
            runtime["current_operation_instance_id"] = None
            runtime["current_request_ref"] = None
            runtime["checkpoint_id"] = None
            runtime["checkpoint_manifest_ref"] = None
            code_receipt = _gaussian_code_receipt_projection(
                manifest=manifest,
                manifest_ref=step_manifest_ref,
                task_root=self._task_output_root(conn, task_id),
            )
            if gaussian_code_receipt is not None:
                if not isinstance(gaussian_code_receipt, Mapping):
                    raise ContractValidationError("Gaussian codegen receipt 必须是 object")
                if code_receipt is None or any(
                    gaussian_code_receipt.get(field) != code_receipt.get(field)
                    for field in (
                        "step_id",
                        "step_index",
                        "gaussian_code",
                        "step_manifest_ref",
                        "normalized_step_ref",
                        "route_sha256",
                    )
                ):
                    raise ContractValidationError(
                        "提交的 Gaussian codegen receipt 与 finalized manifest 不一致"
                    )
            if code_receipt is not None:
                code_results = runtime.get("gaussian_code_results")
                if not isinstance(code_results, dict):
                    code_results = {}
                code_results[str(code_receipt["step_id"])] = dict(code_receipt)
                runtime["gaussian_code_results"] = code_results
            cursor = conn.execute(
                """
                UPDATE combinations SET runtime_checkpoint_json=?, version=version+1,
                    updated_at=?
                WHERE task_id=? AND combination_id=? AND state='running' AND version=?
                """,
                (
                    canonical_json(runtime),
                    utc_now(),
                    task_id,
                    combination_id,
                    expected_version,
                ),
            )
            if cursor.rowcount != 1:
                raise CASConflict(
                    "completed step 输掉并发 CAS",
                    expected_version=expected_version,
                    current_version=row["version"],
                    expected_state="running",
                    current_state=row["state"],
                )
            updated = conn.execute(
                "SELECT * FROM combinations WHERE task_id=? AND combination_id=?",
                (task_id, combination_id),
            ).fetchone()
            return combination_row(updated)

    def fail_execution_step(
        self,
        *,
        task_id: str,
        combination_id: str,
        operation_instance_id: str,
        failed_step_manifest_ref: str,
        step_manifest_refs: Sequence[str],
        step_index: int,
        expected_version: int,
        expected_identity: Mapping[str, Any],
        causal_source_ref: Optional[str],
        failure_diagnostic: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """登记工具失败、失败后的未执行步骤，并保持组合进入统一审查。

        输入必须包含从首步到 workflow 末尾的完整步骤清单，其中失败步骤位于
        ``step_index``，后续清单状态为 ``not_executed``。方法在一个事务中核对
        清单、当前执行 claim 和版本，写入不可变失败事实并把执行游标推进到
        workflow 末尾；组合仍为 ``running``，下一阶段会运行 Program Gate 和
        执行后 Expert/General/Debate，而不会进入 ``technical_pause``。
        """

        store = self._require_enabled()
        require_nonempty_string(operation_instance_id, "operation_instance_id")
        require_nonempty_string(failed_step_manifest_ref, "failed_step_manifest_ref")
        if isinstance(step_index, bool) or not isinstance(step_index, int) or step_index < 0:
            raise ContractValidationError("step_index 必须是非负 JSON integer")
        expected_identity = dict(expected_identity)
        if set(expected_identity) != {
            "workflow_id",
            "workflow_version",
            "attempt_index",
            "execution_id",
        }:
            raise ContractValidationError("expected_identity 必须包含完整执行身份")
        refs = list(step_manifest_refs)
        if (
            not refs
            or any(not isinstance(item, str) or not item for item in refs)
            or len(refs) != len(set(refs))
            or step_index >= len(refs)
        ):
            raise ContractValidationError("失败执行必须保存完整且有序的步骤清单")
        if refs[step_index] != failed_step_manifest_ref:
            raise ContractValidationError(
                "failed_step_manifest_ref 必须对应失败步骤在完整清单中的位置"
            )
        payload = validate_execution_step_failed_payload(
            {
                "combination_id": combination_id,
                "execution_id": expected_identity["execution_id"],
                "workflow_version": expected_identity["workflow_version"],
                "step_id": "pending",
                "step_index": step_index,
                "operation_instance_id": operation_instance_id,
                "failed_step_manifest_ref": failed_step_manifest_ref,
                "step_manifest_refs": refs,
                "causal_source_ref": causal_source_ref,
                **(
                    {"failure_diagnostic": deepcopy(dict(failure_diagnostic))}
                    if isinstance(failure_diagnostic, Mapping)
                    else {}
                ),
            }
        )
        with store.immediate_transaction() as conn:
            row = conn.execute(
                "SELECT * FROM combinations WHERE task_id=? AND combination_id=?",
                (task_id, combination_id),
            ).fetchone()
            if row is None:
                raise ContractValidationError("failed step 引用未知 combination")
            workflow = self._decode_json(
                row["current_workflow_json"], field="current_workflow_json"
            )
            workflow_steps = workflow.get("steps") if isinstance(workflow, Mapping) else None
            if not isinstance(workflow_steps, list) or len(workflow_steps) != len(refs):
                raise ContractValidationError(
                    "失败执行清单必须逐项覆盖当前 workflow"
                )
            manifests = [
                self._read_finalized_manifest(conn, task_id, reference)
                for reference in refs
            ]
            failed_manifest = manifests[step_index]
            failed_step = workflow_steps[step_index]
            if not isinstance(failed_step, Mapping):
                raise ContractValidationError("当前 workflow 失败步骤不是 object")
            payload["step_id"] = require_nonempty_string(
                failed_step.get("step_id"), "step_id"
            )
            payload = validate_execution_step_failed_payload(payload)
            for index, (manifest, step) in enumerate(zip(manifests, workflow_steps)):
                if not isinstance(step, Mapping):
                    raise ContractValidationError("workflow step 必须是 object")
                if (
                    manifest.get("step_index") != index
                    or manifest.get("step_id") != step.get("step_id")
                    or manifest.get("execution_id") != expected_identity["execution_id"]
                    or manifest.get("workflow_version") != expected_identity["workflow_version"]
                    or manifest.get("step_manifest_refs") != refs[:index]
                ):
                    raise ContractValidationError("失败执行清单的步骤身份或顺序不一致")
                status = manifest.get("status")
                expected_status = (
                    "completed" if index < step_index
                    else "failed" if index == step_index
                    else "not_executed"
                )
                if status != expected_status:
                    raise ContractValidationError(
                        "失败执行清单必须按 completed/failed/not_executed 顺序排列"
                    )
            if failed_manifest.get("operation_instance_id") != operation_instance_id:
                raise ContractValidationError(
                    "失败步骤清单没有沿用当前执行 claim"
                )
            runtime = self._decode_json(
                row["runtime_checkpoint_json"], field="runtime_checkpoint_json"
            )
            event_key = event_key_execution_step_failed(
                task_id,
                expected_identity["execution_id"],
                operation_instance_id,
            )
            existing = self._existing_event_exact(
                conn,
                event_key=event_key,
                task_id=task_id,
                round_id=expected_identity["attempt_index"],
                event_type="execution_step_failed",
                payload=payload,
                source_refs=sorted(refs),
            )
            if existing is not None:
                return combination_row(row)
            identity_matches = all(
                row[field] == value for field, value in expected_identity.items()
            )
            if (
                row["version"] != expected_version
                or row["state"] != "running"
                or row["causal_source_ref"] != causal_source_ref
                or not identity_matches
                or runtime.get("next_step_index") != step_index
                or runtime.get("current_operation_instance_id") != operation_instance_id
                or runtime.get("current_request_ref")
                != self._manifest_ref_path(failed_manifest.get("request_ref"))
            ):
                raise CASConflict(
                    "failed step 的 CAS 前提已变化",
                    expected_version=expected_version,
                    current_version=row["version"],
                    expected_state="running",
                    current_state=row["state"],
                )
            event_id, inserted = self._insert_event(
                conn,
                event_key=event_key,
                task_id=task_id,
                round_id=expected_identity["attempt_index"],
                event_type="execution_step_failed",
                payload=payload,
                source_refs=sorted(refs),
                check_operation_occupancy=True,
            )
            if not inserted:
                raise OperationConflict("步骤失败事实的并发状态不一致")
            runtime["step_manifest_refs"] = refs
            runtime["next_step_index"] = len(refs)
            runtime["current_operation_instance_id"] = None
            runtime["current_request_ref"] = None
            runtime["checkpoint_id"] = None
            runtime["checkpoint_manifest_ref"] = None
            runtime["execution_failure"] = {
                "step_id": payload["step_id"],
                "step_index": step_index,
                "failed_step_manifest_ref": failed_step_manifest_ref,
                "event_id": event_id,
            }
            if isinstance(failure_diagnostic, Mapping):
                runtime["execution_failure"]["diagnostic"] = deepcopy(dict(failure_diagnostic))
            attempts = runtime.get("step_revision_attempts")
            if not isinstance(attempts, Mapping):
                attempts = {}
            attempts = {
                str(key): int(value)
                for key, value in attempts.items()
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0
            }
            attempts[payload["step_id"]] = attempts.get(payload["step_id"], 0) + 1
            runtime["step_revision_attempts"] = attempts
            exhausted = attempts[payload["step_id"]] >= 5
            diagnostic_code = (
                failure_diagnostic.get("code")
                if isinstance(failure_diagnostic, Mapping)
                else None
            )
            terminal_classification = (
                "tool_capability_gap_after_retry_limit"
                if diagnostic_code in {
                    "unregistered_action",
                    "backend_unavailable",
                    "native_software_unavailable",
                    "unsupported_execution_layer",
                    "missing_required_software",
                }
                else "hypothesis_failed"
            )
            if exhausted:
                runtime["terminal_classification"] = terminal_classification
            cursor = conn.execute(
                """
                UPDATE combinations SET state=?, terminal_record_id=?,
                    terminal_reason=?, runtime_checkpoint_json=?,
                    version=version+1, updated_at=?
                WHERE task_id=? AND combination_id=? AND state='running' AND version=?
                """,
                (
                    "hypothesis_failed" if exhausted else "running",
                    event_key if exhausted else None,
                    "max_revision_attempts_exceeded" if exhausted else None,
                    canonical_json(runtime),
                    utc_now(),
                    task_id,
                    combination_id,
                    expected_version,
                ),
            )
            if cursor.rowcount != 1:
                raise CASConflict(
                    "failed step 输掉并发 CAS",
                    expected_version=expected_version,
                    current_version=row["version"],
                    expected_state="running",
                    current_state=row["state"],
                )
            self._reduce_task_status_in_transaction(conn, task_id)
            updated = conn.execute(
                "SELECT * FROM combinations WHERE task_id=? AND combination_id=?",
                (task_id, combination_id),
            ).fetchone()
            return combination_row(updated)

    def complete_planning(
        self,
        task_id: str,
        payload: Mapping[str, Any],
        source_refs: Sequence[str],
        *,
        expected_version: int,
    ) -> str:
        """保存首版工作流，并同时建立后续使用的工作流与执行身份。"""

        store = self._require_enabled()
        value = validate_planning_payload(dict(payload))
        refs = validate_planning_source_refs(list(source_refs))
        if any("/" not in reference for reference in refs):
            raise ContractValidationError(
                "PlanningCompleted.source_refs 必须引用 finalized manifest"
            )
        event_key = event_key_planning(
            task_id, value["combination_id"], value["operation_instance_id"]
        )
        with store.immediate_transaction() as conn:
            existing = self._existing_event_exact(
                conn,
                event_key=event_key,
                task_id=task_id,
                round_id=0,
                event_type="planning_completed",
                payload=value,
                source_refs=refs,
            )
            if existing is not None:
                return existing
            self._assert_operation_available(conn, task_id, value["operation_instance_id"])
            row = conn.execute(
                "SELECT * FROM combinations WHERE task_id=? AND combination_id=?",
                (task_id, value["combination_id"]),
            ).fetchone()
            if row is None:
                raise ContractValidationError("PlanningCompleted 引用未知 combination")
            if (
                row["version"] != expected_version
                or row["state"] != "generating_workflow"
                or row["hypothesis_id"] != value["hypothesis_id"]
                or row["causal_source_ref"] != value["causal_source_ref"]
            ):
                raise CASConflict(
                    "PlanningCompleted 的 CAS 前提已变化",
                    expected_version=expected_version,
                    current_version=row["version"],
                    expected_state="generating_workflow",
                    current_state=row["state"],
                )
            event_id, _ = self._insert_event(
                conn,
                event_key=event_key,
                task_id=task_id,
                round_id=0,
                event_type="planning_completed",
                payload=value,
                source_refs=refs,
                check_operation_occupancy=True,
            )
            cursor = conn.execute(
                """
                UPDATE combinations SET
                    workflow_id=?, workflow_version=1, attempt_index=1, execution_id=?,
                    state='running', current_workflow_json=?, runtime_checkpoint_json=?,
                    causal_source_ref=NULL, version=version+1, updated_at=?
                WHERE task_id=? AND combination_id=? AND state='generating_workflow'
                  AND version=? AND workflow_id IS NULL AND workflow_version=0
                  AND execution_id IS NULL
                """,
                (
                    value["workflow_id"],
                    value["execution_id"],
                    canonical_json(value["workflow"]),
                    canonical_json(empty_runtime_checkpoint()),
                    utc_now(),
                    task_id,
                    value["combination_id"],
                    expected_version,
                ),
            )
            if cursor.rowcount != 1:
                raise CASConflict(
                    "PlanningCompleted 输掉并发 CAS",
                    expected_version=expected_version,
                    current_version=row["version"],
                    expected_state="generating_workflow",
                    current_state=row["state"],
                )
            return event_id

    def record_expert_gaussian_review(
        self,
        task_id: str,
        payload: Mapping[str, Any],
        source_refs: Sequence[str],
        *,
        expected_version: int,
    ) -> str:
        """保存 Planning 后 Gaussian 审查，并在 reject 时原子暂停组合。"""

        store = self._require_enabled()
        value = validate_expert_gaussian_review_payload(dict(payload))
        refs = validate_source_ref_shape(list(source_refs))
        if not refs or not any("/" in item for item in refs):
            raise ContractValidationError("ExpertGaussianReview 必须引用审查 artifact")
        event_key = event_key_expert_gaussian_review(
            task_id,
            value["combination_id"],
            value["operation_instance_id"],
        )
        with store.immediate_transaction() as conn:
            row = conn.execute(
                "SELECT * FROM combinations WHERE task_id=? AND combination_id=?",
                (task_id, value["combination_id"]),
            ).fetchone()
            if row is None:
                raise ContractValidationError("ExpertGaussianReview 引用了未知 combination")
            existing = self._existing_event_exact(
                conn,
                event_key=event_key,
                task_id=task_id,
                round_id=row["attempt_index"],
                event_type="expert_gaussian_review",
                payload=value,
                source_refs=refs,
            )
            if existing is not None:
                return existing
            self._assert_operation_available(conn, task_id, value["operation_instance_id"])
            if (
                row["version"] != expected_version
                or row["state"] != "running"
                or row["workflow_version"] != value["workflow_version"]
                or row["execution_id"] != value["execution_id"]
            ):
                raise CASConflict(
                    "ExpertGaussianReview 的 CAS 前提已变化",
                    expected_version=expected_version,
                    current_version=row["version"],
                    expected_state="running",
                    current_state=row["state"],
                )
            event_id, _ = self._insert_event(
                conn,
                event_key=event_key,
                task_id=task_id,
                round_id=row["attempt_index"],
                event_type="expert_gaussian_review",
                payload=value,
                source_refs=refs,
                check_operation_occupancy=True,
            )
            runtime = self._decode_json(row["runtime_checkpoint_json"], field="runtime_checkpoint_json")
            runtime["expert_review"] = _merge_gaussian_review_projection(
                runtime.get("expert_review"),
                value,
            )
            approved = value["status"] == "approved"
            new_pre_execution_protocol = value.get("review_phase") == "pre_execution"
            runtime["workflow_frozen"] = approved
            if approved:
                runtime.pop("expert_revision_feedback", None)
            elif new_pre_execution_protocol:
                value.setdefault("requested_transition", "revise_workflow")
                runtime["expert_revision_feedback"] = deepcopy(value)
                runtime["expert_revision_feedback"]["source_event_id"] = event_id
            next_state = (
                "running"
                if approved or (not new_pre_execution_protocol and value["status"] == "revise")
                else "revising_workflow"
                if new_pre_execution_protocol
                else "paused"
            )
            cursor = conn.execute(
                """
                UPDATE combinations SET state=?, terminal_reason=?, runtime_checkpoint_json=?,
                    version=version+1, updated_at=?
                WHERE task_id=? AND combination_id=? AND state='running' AND version=?
                """,
                (
                    next_state,
                    (
                        None
                        if approved
                        or new_pre_execution_protocol
                        or (not new_pre_execution_protocol and value["status"] == "revise")
                        else "expert_gaussian_reject"
                        if value["status"] == "reject"
                        else "expert_model_not_configured"
                    ),
                    canonical_json(runtime),
                    utc_now(),
                    task_id,
                    value["combination_id"],
                    expected_version,
                ),
            )
            if cursor.rowcount != 1:
                raise CASConflict(
                    "ExpertGaussianReview 输掉并发 CAS",
                    expected_version=expected_version,
                    current_version=row["version"],
                    expected_state="running",
                    current_state=next_state,
                )
            self._reduce_task_status_in_transaction(conn, task_id)
            return event_id

    def record_expert_debate(
        self,
        task_id: str,
        payload: Mapping[str, Any],
        source_refs: Sequence[str],
        *,
        expected_version: int,
    ) -> str:
        """保存完整执行后的 debate/judge 事实，保持组合在 Router 边界。"""

        store = self._require_enabled()
        value = validate_expert_debate_payload(dict(payload))
        refs = validate_source_ref_shape(list(source_refs))
        if not refs or not any("/" in item for item in refs):
            raise ContractValidationError("ExpertDebate 必须引用 debate artifact")
        event_key = event_key_expert_debate(
            task_id,
            value["combination_id"],
            value["operation_instance_id"],
        )
        with store.immediate_transaction() as conn:
            row = conn.execute(
                "SELECT * FROM combinations WHERE task_id=? AND combination_id=?",
                (task_id, value["combination_id"]),
            ).fetchone()
            if row is None or row["state"] != "routing":
                raise ContractValidationError("ExpertDebate 只接受 routing 组合")
            existing = self._existing_event_exact(
                conn,
                event_key=event_key,
                task_id=task_id,
                round_id=row["attempt_index"],
                event_type="expert_debate_completed",
                payload=value,
                source_refs=refs,
            )
            if existing is not None:
                return existing
            self._assert_operation_available(conn, task_id, value["operation_instance_id"])
            if (
                row["version"] != expected_version
                or row["workflow_version"] != value["workflow_version"]
                or row["execution_id"] != value["execution_id"]
            ):
                raise CASConflict(
                    "ExpertDebate 的 CAS 前提已变化",
                    expected_version=expected_version,
                    current_version=row["version"],
                    expected_state="routing",
                    current_state=row["state"],
                )
            event_id, _ = self._insert_event(
                conn,
                event_key=event_key,
                task_id=task_id,
                round_id=row["attempt_index"],
                event_type="expert_debate_completed",
                payload=value,
                source_refs=refs,
                check_operation_occupancy=True,
            )
            runtime = self._decode_json(row["runtime_checkpoint_json"], field="runtime_checkpoint_json")
            runtime["expert_debate"] = value
            cursor = conn.execute(
                """
                UPDATE combinations SET runtime_checkpoint_json=?, version=version+1, updated_at=?
                WHERE task_id=? AND combination_id=? AND state='routing' AND version=?
                """,
                (
                    canonical_json(runtime),
                    utc_now(),
                    task_id,
                    value["combination_id"],
                    expected_version,
                ),
            )
            if cursor.rowcount != 1:
                raise CASConflict(
                    "ExpertDebate 输掉并发 CAS",
                    expected_version=expected_version,
                    current_version=row["version"],
                    expected_state="routing",
                    current_state=row["state"],
                )
            return event_id

    def record_expert_model_pause(
        self,
        task_id: str,
        payload: Mapping[str, Any],
        source_refs: Sequence[str],
        *,
        expected_version: int,
    ) -> str:
        """保存 provider 未配置的专家暂停并阻止后续工具或 Router 调用。"""

        store = self._require_enabled()
        value = validate_expert_pause_payload(dict(payload))
        refs = validate_source_ref_shape(list(source_refs))
        if not refs or not any("/" in item for item in refs):
            raise ContractValidationError("ExpertModelPaused 必须引用暂停 artifact")
        event_key = event_key_expert_pause(
            task_id,
            value["combination_id"],
            value["operation_instance_id"],
        )
        with store.immediate_transaction() as conn:
            row = conn.execute(
                "SELECT * FROM combinations WHERE task_id=? AND combination_id=?",
                (task_id, value["combination_id"]),
            ).fetchone()
            if row is None:
                raise ContractValidationError("ExpertModelPaused 引用了未知 combination")
            existing = self._existing_event_exact(
                conn,
                event_key=event_key,
                task_id=task_id,
                round_id=row["attempt_index"],
                event_type="expert_model_paused",
                payload=value,
                source_refs=refs,
            )
            if existing is not None:
                return existing
            self._assert_operation_available(conn, task_id, value["operation_instance_id"])
            if row is None or row["version"] != expected_version or row["state"] not in {"running", "routing"}:
                raise CASConflict(
                    "ExpertModelPaused 的 CAS 前提已变化",
                    expected_version=expected_version,
                    current_version=None if row is None else row["version"],
                    expected_state="running 或 routing",
                    current_state=None if row is None else row["state"],
                )
            event_id, _ = self._insert_event(
                conn,
                event_key=event_key,
                task_id=task_id,
                round_id=row["attempt_index"],
                event_type="expert_model_paused",
                payload=value,
                source_refs=refs,
                check_operation_occupancy=True,
            )
            runtime = self._decode_json(row["runtime_checkpoint_json"], field="runtime_checkpoint_json")
            runtime["expert_pause"] = value
            conn.execute(
                """
                UPDATE combinations SET state='paused', terminal_reason='expert_model_not_configured', runtime_checkpoint_json=?,
                    version=version+1, updated_at=?
                WHERE task_id=? AND combination_id=? AND version=?
                """,
                (
                    canonical_json(runtime),
                    utc_now(),
                    task_id,
                    value["combination_id"],
                    expected_version,
                ),
            )
            self._reduce_task_status_in_transaction(conn, task_id)
            return event_id

    def record_tool_capability_gap(
        self,
        task_id: str,
        payload: Mapping[str, Any],
        source_refs: Sequence[str],
        *,
        expected_version: int,
    ) -> str:
        """原子保存规划或修订阶段确认的工具能力缺口。"""

        store = self._require_enabled()
        value = validate_tool_capability_gap_payload(dict(payload))
        refs = validate_source_ref_shape(list(source_refs))
        path_refs = [item for item in refs if "/" in item]
        record_refs = [item for item in refs if "/" not in item]
        initial_gap = len(refs) == 1 and len(path_refs) == 1
        revision_gap = (
            len(refs) == 3
            and len(path_refs) == 2
            and len(record_refs) == 1
        )
        if not initial_gap and not revision_gap:
            raise ContractValidationError(
                "ToolCapabilityGap.source_refs 必须是 Planner 清单，或来源 Attempt 加两份修订清单"
            )
        event_key = event_key_tool_capability_gap(
            task_id,
            value["combination_id"],
            value["operation_instance_id"],
        )
        with store.immediate_transaction() as conn:
            row = conn.execute(
                "SELECT * FROM combinations WHERE task_id=? AND combination_id=?",
                (task_id, value["combination_id"]),
            ).fetchone()
            if row is None:
                raise ContractValidationError(
                    "ToolCapabilityGap 引用未知 combination"
                )
            round_id = row["attempt_index"]
            existing = self._existing_event_exact(
                conn,
                event_key=event_key,
                task_id=task_id,
                round_id=round_id,
                event_type="tool_capability_gap",
                payload=value,
                source_refs=refs,
            )
            if existing is not None:
                if (
                    row["state"] == "tool_capability_gap"
                    and row["terminal_record_id"] == existing
                    and row["terminal_reason"] == "tool_capability_gap"
                    and row["hypothesis_id"] == value["hypothesis_id"]
                ):
                    return existing
                raise IdempotencyConflict(
                    "ToolCapabilityGap 已存在，但组合投影没有停在对应终态"
                )
            self._assert_operation_available(
                conn,
                task_id,
                value["operation_instance_id"],
            )
            if (
                row["version"] != expected_version
                or row["hypothesis_id"] != value["hypothesis_id"]
                or row["causal_source_ref"] != value["causal_source_ref"]
            ):
                raise CASConflict(
                    "ToolCapabilityGap 的 CAS 前提已变化",
                    expected_version=expected_version,
                    current_version=row["version"],
                    expected_state=(
                        "generating_workflow" if initial_gap else "revising_workflow"
                    ),
                    current_state=row["state"],
                )
            if initial_gap:
                if (
                    row["state"] != "generating_workflow"
                    or row["workflow_id"] is not None
                    or row["workflow_version"] != 0
                    or row["attempt_index"] != 0
                    or row["execution_id"] is not None
                    or row["current_workflow_json"] is not None
                ):
                    raise CASConflict(
                        "Planner 能力缺口的来源组合已经开始执行",
                        expected_version=expected_version,
                        current_version=row["version"],
                        expected_state="generating_workflow",
                        current_state=row["state"],
                    )
            else:
                if (
                    row["state"] != "revising_workflow"
                    or not isinstance(row["workflow_id"], str)
                    or row["workflow_version"] < 1
                    or row["attempt_index"] < 1
                    or not isinstance(row["execution_id"], str)
                    or row["current_workflow_json"] is None
                ):
                    raise CASConflict(
                        "修订能力缺口没有紧跟一轮完整执行",
                        expected_version=expected_version,
                        current_version=row["version"],
                        expected_state="revising_workflow",
                        current_state=row["state"],
                    )
                source_attempt_id = record_refs[0]
                source_attempt_row = conn.execute(
                    "SELECT event_type, payload_json FROM memory_events "
                    "WHERE task_id=? AND event_id=?",
                    (task_id, source_attempt_id),
                ).fetchone()
                if (
                    source_attempt_row is None
                    or source_attempt_row["event_type"] != "attempt"
                ):
                    raise ContractValidationError(
                        "修订能力缺口必须引用当前组合的来源 Attempt"
                    )
                source_attempt = validate_attempt_payload(
                    self._decode_json(
                        source_attempt_row["payload_json"],
                        field="ToolCapabilityGap source Attempt.payload_json",
                    )
                )
                if (
                    source_attempt["combination_id"] != value["combination_id"]
                    or source_attempt["workflow_version"] != row["workflow_version"]
                    or source_attempt["execution_id"] != row["execution_id"]
                    or source_attempt["router_decision"].get("decision")
                    != "revise_workflow"
                ):
                    raise ContractValidationError(
                        "修订能力缺口的来源 Attempt 与当前工作流身份不一致"
                    )

            batch = conn.execute(
                "SELECT created_event_id FROM hypothesis_batches "
                "WHERE task_id=? AND batch_id=?",
                (task_id, row["batch_id"]),
            ).fetchone()
            batch_event = (
                None
                if batch is None
                else conn.execute(
                    "SELECT payload_json FROM memory_events "
                    "WHERE task_id=? AND event_id=? "
                    "AND event_type='hypothesis_batch_created'",
                    (task_id, batch["created_event_id"]),
                ).fetchone()
            )
            if batch_event is None:
                raise ContractValidationError(
                    "ToolCapabilityGap 缺少冻结 HypothesisBatchCreated 事实"
                )
            batch_payload = self._validated_batch_created_payload(
                conn,
                task_id,
                batch_event["payload_json"],
                field="ToolCapabilityGap HypothesisBatchCreated.payload_json",
            )
            ranked = batch_payload["hypothesis_phase"]["ranked_strategies"]
            hypotheses = [
                item
                for item in ranked
                if item.get("hypothesis_id") == value["hypothesis_id"]
            ]
            if len(hypotheses) != 1:
                raise ContractValidationError(
                    "ToolCapabilityGap 没有唯一对应的冻结 hypothesis"
                )
            if "unmet_verification_requirements" in value["gap"]:
                requirements = set(hypotheses[0]["verification_requirements"])
                if any(
                    item not in requirements
                    for item in value["gap"]["unmet_verification_requirements"]
                ):
                    raise ContractValidationError(
                        "旧版 ToolCapabilityGap 引用了当前 hypothesis 之外的验证需求"
                    )

            event_id, _ = self._insert_event(
                conn,
                event_key=event_key,
                task_id=task_id,
                round_id=round_id,
                event_type="tool_capability_gap",
                payload=value,
                source_refs=refs,
                check_operation_occupancy=True,
            )
            now = utc_now()
            if initial_gap:
                cursor = conn.execute(
                    """
                    UPDATE combinations SET
                        state='tool_capability_gap', terminal_record_id=?,
                        terminal_reason='tool_capability_gap', causal_source_ref=NULL,
                        version=version+1, updated_at=?
                    WHERE task_id=? AND combination_id=?
                      AND state='generating_workflow' AND version=?
                      AND workflow_id IS NULL AND workflow_version=0
                      AND attempt_index=0 AND execution_id IS NULL
                      AND current_workflow_json IS NULL
                    """,
                    (
                        event_id,
                        now,
                        task_id,
                        value["combination_id"],
                        expected_version,
                    ),
                )
            else:
                cursor = conn.execute(
                    """
                    UPDATE combinations SET
                        state='tool_capability_gap', terminal_record_id=?,
                        terminal_reason='tool_capability_gap', causal_source_ref=NULL,
                        version=version+1, updated_at=?
                    WHERE task_id=? AND combination_id=?
                      AND state='revising_workflow' AND version=?
                      AND workflow_id=? AND workflow_version=?
                      AND attempt_index=? AND execution_id=?
                      AND current_workflow_json=?
                    """,
                    (
                        event_id,
                        now,
                        task_id,
                        value["combination_id"],
                        expected_version,
                        row["workflow_id"],
                        row["workflow_version"],
                        row["attempt_index"],
                        row["execution_id"],
                        row["current_workflow_json"],
                    ),
                )
            if cursor.rowcount != 1:
                raise CASConflict(
                    "ToolCapabilityGap 输掉并发 CAS",
                    expected_version=expected_version,
                    current_version=row["version"],
                    expected_state=(
                        "generating_workflow" if initial_gap else "revising_workflow"
                    ),
                    current_state=row["state"],
                )
            self._reduce_task_status_in_transaction(conn, task_id)
            return event_id

    def pause_operation(
        self,
        task_id: str,
        payload: Mapping[str, Any],
        source_refs: Sequence[str],
        *,
        expected_version: Optional[int],
    ) -> str:
        """把运行停在可恢复的位置，并只保存一条完整的暂停结果。

        暂停不是科学判断，因此不会顺带写入路由结果或不完整尝试。
        """

        store = self._require_enabled()
        value = validate_pause_payload(dict(payload))
        if not source_refs or not any("/" in item for item in source_refs):
            raise ContractValidationError("Pause 必须至少引用一份 finalized 失败证据 manifest")
        event_key = event_key_pause(task_id, value["operation_instance_id"])
        round_id = 0
        with store.immediate_transaction() as conn:
            existing = self._existing_event_exact(
                conn,
                event_key=event_key,
                task_id=task_id,
                round_id=0 if value["combination_id"] is None else self._attempt_index(
                    conn, task_id, value["combination_id"]
                ),
                event_type="execution_paused",
                payload=value,
                source_refs=source_refs,
            )
            if existing is not None:
                return existing
            if value["combination_id"] is None:
                # 任务级暂停发生在尚未进入具体候选组合的阶段，只推进任务自身版本。
                if expected_version is None:
                    raise ContractValidationError("task-scope Pause 必须提供 expected task version")
                task = conn.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
                if task is None or task["status"] != "running" or task["version"] != expected_version:
                    raise CASConflict(
                        "task-scope Pause 的 CAS 前提已变化",
                        expected_version=expected_version,
                        current_version=None if task is None else task["version"],
                        expected_state="running",
                        current_state=None if task is None else task["status"],
                    )
                round_id = 0
                event_id, _ = self._insert_event(
                    conn,
                    event_key=event_key,
                    task_id=task_id,
                    round_id=round_id,
                    event_type="execution_paused",
                    payload=value,
                    source_refs=source_refs,
                    check_operation_occupancy=True,
                )
                conn.execute(
                    "UPDATE tasks SET status='paused', version=version+1, updated_at=? "
                    "WHERE task_id=? AND status='running' AND version=?",
                    (utc_now(), task_id, expected_version),
                )
                return event_id

            # 组合级暂停要核对工作流、执行身份和来源授权，不能暂停已经推进的新版本。
            row = conn.execute(
                "SELECT * FROM combinations WHERE task_id=? AND combination_id=?",
                (task_id, value["combination_id"]),
            ).fetchone()
            if row is None:
                raise ContractValidationError("Pause 引用未知 combination")
            round_id = row["attempt_index"]
            runtime = self._decode_json(
                row["runtime_checkpoint_json"], field="runtime_checkpoint_json"
            )
            if expected_version is None:
                raise ContractValidationError("combination-scope Pause 必须提供 expected version")
            if value["operation"] == "execute_step":
                self._require_claim(
                    row,
                    operation_instance_id=value["operation_instance_id"],
                )
            if (
                row["version"] != expected_version
                or row["state"]
                != (
                    "running"
                    if value["operation"] == "execute_step"
                    else self._MODEL_OPERATION_STATES[value["operation"]]
                )
                or row["execution_id"] != value["execution_id"]
                or (row["workflow_version"] or None) != value["workflow_version"]
                or row["causal_source_ref"] != value["causal_source_ref"]
            ):
                raise CASConflict(
                    "Pause 的 CAS 前提已变化",
                    expected_version=expected_version,
                    current_version=row["version"],
                    expected_state=(
                        "running"
                        if value["operation"] == "execute_step"
                        else self._MODEL_OPERATION_STATES[value["operation"]]
                    ),
                    current_state=row["state"],
                )
            if value["operation"] == "execute_step":
                manifest_paths = [item for item in source_refs if "/" in item]
                if len(manifest_paths) != 1:
                    raise ContractValidationError(
                        "工具技术暂停必须恰好引用一份步骤清单"
                    )
                pause_manifest = self._read_finalized_manifest(
                    conn,
                    task_id,
                    manifest_paths[0],
                )
                expected_manifest = {
                    "operation_instance_id": value["operation_instance_id"],
                    "execution_id": value["execution_id"],
                    "workflow_version": value["workflow_version"],
                    "status": "technical_pause",
                }
                if any(
                    pause_manifest.get(field) != expected
                    for field, expected in expected_manifest.items()
                ):
                    raise ContractValidationError(
                        "技术暂停步骤清单与当前执行身份不一致"
                    )
                prefix = pause_manifest.get("step_manifest_refs")
                if (
                    not isinstance(prefix, list)
                    or prefix != runtime.get("step_manifest_refs", [])
                    or pause_manifest.get("step_index") != len(prefix)
                ):
                    raise ContractValidationError(
                        "技术暂停步骤清单没有准确记录已完成步骤前缀"
                    )
            # 暂停事实先写入同一事务，随后清空当前 claim，恢复时从暂停记录重新授权。
            event_id, _ = self._insert_event(
                conn,
                event_key=event_key,
                task_id=task_id,
                round_id=round_id,
                event_type="execution_paused",
                payload=value,
                source_refs=source_refs,
                check_operation_occupancy=True,
            )
            if value["operation"] == "execute_step":
                runtime["current_operation_instance_id"] = None
                runtime["current_request_ref"] = None
                runtime["checkpoint_id"] = None
                runtime["checkpoint_manifest_ref"] = None
            cursor = conn.execute(
                """
                UPDATE combinations SET state='paused', runtime_checkpoint_json=?,
                    version=version+1, updated_at=?
                WHERE task_id=? AND combination_id=? AND state=? AND version=?
                """,
                (
                    canonical_json(runtime),
                    utc_now(),
                    task_id,
                    value["combination_id"],
                    row["state"],
                    expected_version,
                ),
            )
            if cursor.rowcount != 1:
                raise CASConflict(
                    "Pause 输掉并发 CAS",
                    expected_version=expected_version,
                    current_version=row["version"],
                    expected_state=row["state"],
                    current_state=row["state"],
                )
            self._reduce_task_status_in_transaction(conn, task_id)
            return event_id

    def record_retrieval_pause(
        self,
        task_id: str,
        status: Mapping[str, Any],
        retrieval_request: Optional[Mapping[str, Any]] = None,
    ) -> str:
        """保存脱敏的 PaperSearch 暂停事实，并把任务 CAS 推进为 ``paused``。

        输入只用于整理固定字段；真正写入前会先读取任务、检查稳定事件键和运行
        版本。这样无效重放或非 running 任务不会创建孤立 artifact。artifact 写入
        虽然位于 SQLite 事务中，但文件系统不会随事务回滚，因此失败时只清理本次
        新建的内容地址文件，已存在的同内容文件保持不动。
        """

        store = self._require_enabled()
        if not isinstance(status, Mapping):
            raise ContractValidationError("Retrieval 暂停状态必须是 object")

        def _logical_basename(raw: Any, field: str, default: str) -> str:
            """将检索路径收敛为 basename，并拒绝父目录跳转。"""

            if raw is None or (isinstance(raw, str) and not raw.strip()):
                text = default
            elif not isinstance(raw, str):
                raise ContractValidationError(f"{field} 必须是非空字符串")
            else:
                text = raw.strip()
            if "\x00" in text:
                raise ContractValidationError(f"{field} 不能包含 NUL 字符")
            posix_text = text.replace("\\", "/")
            parts = PurePosixPath(posix_text).parts
            if ".." in parts:
                raise ContractValidationError(f"{field} 不能包含 ..")
            name = PurePosixPath(posix_text).name
            if not name or name in {".", ".."}:
                raise ContractValidationError(f"{field} 必须是逻辑 basename")
            return name

        raw_operation_id = status.get("operation_instance_id")
        if raw_operation_id is not None:
            operation_instance_id = require_nonempty_string(
                raw_operation_id, "operation_instance_id"
            )
        else:
            operation_instance_id = ""
        base_without_identity = {
            "operation": "paper_search",
            "reason_code": status.get("reason_code", "paper_search_incomplete"),
            "provider": status.get("provider", "pypaperbot_scholar"),
            "manifest_path": _logical_basename(
                status.get("manifest_path"),
                "manifest_path",
                ".paper-search-manifest.json",
            ),
            "resume_state_path": _logical_basename(
                status.get("resume_state_path") or status.get("manifest_path"),
                "resume_state_path",
                ".paper-search-resume.json",
            ),
            "response_cache_dir": _logical_basename(
                status.get("response_cache_dir"),
                "response_cache_dir",
                ".paper-search-cache",
            ),
            "target_papers": status.get("target_papers", 50),
            "actual_papers": status.get("actual_papers", 0),
            "target_reached": bool(status.get("target_reached", False)),
            "stop_reason": status.get("stop_reason"),
        }
        if not operation_instance_id:
            operation_instance_id = (
                "retrieval-paper-search-"
                + hashlib.sha256(canonical_json(base_without_identity).encode("utf-8"))
                .hexdigest()[:24]
            )
        base_value = {
            "operation_instance_id": operation_instance_id,
            **base_without_identity,
        }
        # 占位引用只用于在文件系统动作之前校验所有业务字段。最终引用会由内容
        # 地址文件名产生；状态 artifact 只写入这份已脱敏的固定字段集合。
        validate_retrieval_pause_payload(
            {**base_value, "artifact_ref": "retrieval/pending_" + "0" * 64 + ".json"}
        )
        artifact_payload = {
            "status": "recoverable_incomplete",
            **base_value,
        }
        if retrieval_request is not None:
            artifact_payload["retrieval_request"] = self._sanitize_retrieval_request(
                retrieval_request
            )
        artifact_digest = hashlib.sha256(
            canonical_json(artifact_payload).encode("utf-8")
        ).hexdigest()
        artifact_ref = f"retrieval/paper_search_pause_{artifact_digest}.json"
        # 最终引用同样在写文件前完成契约校验，避免任何非法路径进入任务目录。
        validate_retrieval_pause_payload(
            {**base_value, "artifact_ref": artifact_ref}
        )
        event_key = f"wm:retrieval-pause:{task_id}:{operation_instance_id}"
        artifact_created = False
        artifact_path: Optional[Path] = None
        artifact_directory: Optional[Path] = None
        artifact_directory_created = False
        try:
            with store.immediate_transaction() as conn:
                task = conn.execute(
                    "SELECT status, version FROM tasks WHERE task_id=?", (task_id,)
                ).fetchone()
                if task is None:
                    raise ContractValidationError("Retrieval 暂停引用未知任务")
                existing_row = conn.execute(
                    "SELECT event_id, payload_json, source_refs_json FROM memory_events "
                    "WHERE event_key=?",
                    (event_key,),
                ).fetchone()
                if existing_row is not None:
                    existing_payload = validate_retrieval_pause_payload(
                        self._decode_json(
                            existing_row["payload_json"],
                            field="RetrievalPaused.payload_json",
                        )
                    )
                    if {
                        key: existing_payload[key]
                        for key in base_value
                    } != base_value:
                        raise IdempotencyConflict(
                            f"event_key={event_key} 已绑定到不同 Retrieval 暂停内容"
                        )
                    existing_refs = validate_source_ref_shape(
                        self._decode_json(
                            existing_row["source_refs_json"],
                            field="RetrievalPaused.source_refs_json",
                        )
                    )
                    if existing_refs != [existing_payload["artifact_ref"]]:
                        raise ContractValidationError(
                            "RetrievalPaused.source_refs 必须唯一引用 status artifact"
                        )
                    self._validate_source_refs(conn, task_id, existing_refs)
                    return existing_row["event_id"]
                if task["status"] != "running":
                    raise ContractValidationError(
                        "Retrieval 暂停要求任务仍处于 running"
                    )
                # 先核对 operation identity，避免把一次已由其他终态占用的调用
                # 写成新的暂停事实。
                self._assert_operation_available(
                    conn,
                    task_id,
                    operation_instance_id,
                    allow_event_types=("retrieval_paused",),
                )
                # 先占用任务的 running 版本，再触碰文件系统。该更新仍在同一
                # SQLite 事务中，后续 artifact/event 任一步失败都会回滚。
                cursor = conn.execute(
                    "UPDATE tasks SET status='paused', version=version+1, updated_at=? "
                    "WHERE task_id=? AND status='running' AND version=?",
                    (utc_now(), task_id, task["version"]),
                )
                if cursor.rowcount != 1:
                    raise CASConflict(
                        "Retrieval 暂停输掉任务 CAS",
                        expected_version=task["version"],
                        current_version=task["version"],
                        expected_state="running",
                        current_state="paused",
                )
                task_root = self._task_output_root(conn, task_id)
                artifact_path = task_root / artifact_ref
                artifact_directory = artifact_path.parent
                artifact_directory_created = not artifact_directory.exists()
                artifact_existed = artifact_path.exists()
                write_content_addressed_json(
                    task_root,
                    PurePosixPath("retrieval"),
                    "paper_search_pause",
                    artifact_payload,
                )
                artifact_created = not artifact_existed
                value = validate_retrieval_pause_payload(
                    {**base_value, "artifact_ref": artifact_ref}
                )
                event_id, _ = self._insert_event(
                    conn,
                    event_key=event_key,
                    task_id=task_id,
                    round_id=0,
                    event_type="retrieval_paused",
                    payload=value,
                    source_refs=[artifact_ref],
                    check_operation_occupancy=True,
                    allow_operation_event_types=("retrieval_paused",),
                )
                return event_id
        except Exception:
            if artifact_created and artifact_path is not None:
                try:
                    artifact_path.unlink(missing_ok=True)
                except OSError:
                    # 原始事务异常仍需向调用方抛出；清理失败不会掩盖根因。
                    pass
            if artifact_directory_created and artifact_directory is not None:
                try:
                    artifact_directory.rmdir()
                except OSError:
                    # 目录可能被并发写入或包含其他合法文件，不能递归删除。
                    pass
            raise

    @staticmethod
    def _sanitize_retrieval_request(request: Mapping[str, Any]) -> Dict[str, Any]:
        """整理可用于同一任务重试的 Retrieval 请求，不写入机器绝对路径。"""

        if not isinstance(request, Mapping):
            raise ContractValidationError("Retrieval request 必须是 object")
        required = {"scientific_question", "pdf_dir", "index_dir", "search_papers"}
        if set(request) - (required | {"paper_search"}) or not required <= set(request):
            raise ContractValidationError("Retrieval request 字段不完整")
        question = request["scientific_question"]
        if not isinstance(question, str) or not question.strip():
            raise ContractValidationError("Retrieval request scientific_question 非法")

        def safe_relative_path(value: Any, field: str) -> str:
            if not isinstance(value, str) or not value.strip():
                raise ContractValidationError(f"{field} 必须是非空字符串")
            text = value.strip().replace("\\", "/")
            path = PurePosixPath(text)
            if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
                raise ContractValidationError(
                    f"{field} 必须是可在任务恢复时复用的安全相对路径"
                )
            return path.as_posix()

        search_papers = request["search_papers"]
        if not isinstance(search_papers, bool):
            raise ContractValidationError("Retrieval request search_papers 必须是布尔值")
        value: Dict[str, Any] = {
            "scientific_question": question.strip(),
            "pdf_dir": safe_relative_path(request["pdf_dir"], "pdf_dir"),
            "index_dir": safe_relative_path(request["index_dir"], "index_dir"),
            "search_papers": search_papers,
        }
        config = request.get("paper_search")
        if config is not None:
            if not isinstance(config, Mapping):
                raise ContractValidationError("Retrieval request paper_search 必须是 object")
            # 配置中的缓存、manifest 和 resume 路径在正式请求中可能已被规范化为
            # 绝对路径；暂停 artifact 只保存其逻辑文件名，恢复时由 pdf_dir 重新定位。
            safe_config = dict(config)
            for field in ("manifest_path", "resume_state_path", "response_cache_dir"):
                raw = safe_config.get(field)
                if raw is None:
                    continue
                if not isinstance(raw, str) or not raw.strip():
                    raise ContractValidationError(f"paper_search.{field} 非法")
                safe_config[field] = PurePosixPath(raw.replace("\\", "/")).name
            value["paper_search"] = safe_config
        return value

    def record_retrieval_completed(
        self,
        task_id: str,
        retrieval_result: Mapping[str, Any],
    ) -> str:
        """持久化完整 Retrieval 成功结果，并以事件作为独立恢复事实。"""

        store = self._require_enabled()
        if not isinstance(retrieval_result, Mapping):
            raise ContractValidationError("Retrieval 成功结果必须是 object")
        safe_result = self._sanitize_retrieval_result(retrieval_result)
        result_digest = hashlib.sha256(
            canonical_json(safe_result).encode("utf-8")
        ).hexdigest()
        operation_instance_id = f"retrieval-{result_digest[:24]}"
        event_key = event_key_retrieval_completed(task_id, operation_instance_id)
        artifact_payload = {
            "status": "completed",
            "result_digest": result_digest,
            "retrieval_result": safe_result,
        }
        artifact_digest = hashlib.sha256(
            canonical_json(artifact_payload).encode("utf-8")
        ).hexdigest()
        artifact_ref = f"retrieval/retrieval_completed_{artifact_digest}.json"
        artifact_value = {
            "operation_instance_id": operation_instance_id,
            "operation": "retrieval",
            "status": "completed",
            "result_digest": result_digest,
            "artifact_ref": artifact_ref,
            "causal_source_ref": None,
        }
        validate_retrieval_completed_payload(artifact_value)
        artifact_created = False
        artifact_path: Optional[Path] = None
        try:
            with store.immediate_transaction() as conn:
                task = conn.execute(
                    "SELECT status, version FROM tasks WHERE task_id=?", (task_id,)
                ).fetchone()
                if task is None:
                    raise ContractValidationError("Retrieval 成功事实引用未知任务")
                existing_rows = conn.execute(
                    "SELECT event_id, payload_json FROM memory_events "
                    "WHERE task_id=? AND event_type='retrieval_completed' "
                    "ORDER BY created_at DESC",
                    (task_id,),
                ).fetchall()
                if existing_rows:
                    existing = validate_retrieval_completed_payload(
                        self._decode_json(
                            existing_rows[0]["payload_json"],
                            field="RetrievalCompleted.payload_json",
                        )
                    )
                    if existing["result_digest"] != result_digest:
                        raise IdempotencyConflict(
                            "同一任务的 Retrieval 成功结果不得被不同内容覆盖"
                        )
                    return existing_rows[0]["event_id"]
                causal_source_ref = None
                if task["status"] == "paused":
                    pause = conn.execute(
                        "SELECT event_id FROM memory_events "
                        "WHERE task_id=? AND event_type='retrieval_paused' "
                        "AND NOT EXISTS ("
                        "  SELECT 1 FROM memory_events consumer "
                        "  WHERE consumer.task_id=memory_events.task_id "
                        "    AND json_extract(consumer.payload_json, '$.causal_source_ref')="
                        "        memory_events.event_id"
                        ") "
                        "ORDER BY created_at DESC LIMIT 1",
                        (task_id,),
                    ).fetchone()
                    if pause is None:
                        raise ContractValidationError(
                            "暂停任务缺少可消费的 RetrievalPaused 事实"
                        )
                    causal_source_ref = pause["event_id"]
                elif task["status"] != "running":
                    raise ContractValidationError(
                        "Retrieval 成功事实只接受 running 或 Retrieval paused 任务"
                    )
                value = dict(artifact_value)
                value["causal_source_ref"] = causal_source_ref
                validate_retrieval_completed_payload(value)
                self._assert_operation_available(
                    conn,
                    task_id,
                    operation_instance_id,
                    allow_event_types=("retrieval_completed",),
                )
                task_root = self._task_output_root(conn, task_id)
                artifact_path = task_root / artifact_ref
                artifact_existed = artifact_path.exists()
                write_content_addressed_json(
                    task_root,
                    PurePosixPath("retrieval"),
                    "retrieval_completed",
                    artifact_payload,
                )
                artifact_created = not artifact_existed
                event_id, _ = self._insert_event(
                    conn,
                    event_key=event_key,
                    task_id=task_id,
                    round_id=0,
                    event_type="retrieval_completed",
                    payload=value,
                    source_refs=[artifact_ref],
                    check_operation_occupancy=True,
                    allow_operation_event_types=("retrieval_completed",),
                    allow_causal_source_event_types=(
                        "restart_applied",
                        "retrieval_paused",
                    ),
                )
                if causal_source_ref is not None:
                    cursor = conn.execute(
                        "UPDATE tasks SET status='running', version=version+1, updated_at=? "
                        "WHERE task_id=? AND status='paused' AND version=?",
                        (utc_now(), task_id, task["version"]),
                    )
                    if cursor.rowcount != 1:
                        raise CASConflict(
                            "Retrieval 恢复成功输掉任务 CAS",
                            expected_version=task["version"],
                            current_version=task["version"],
                            expected_state="paused",
                            current_state="paused",
                        )
                return event_id
        except Exception:
            if artifact_created and artifact_path is not None:
                try:
                    artifact_path.unlink(missing_ok=True)
                except OSError:
                    pass
            raise

    @staticmethod
    def _sanitize_retrieval_result(result: Mapping[str, Any]) -> Dict[str, Any]:
        """为 Retrieval receipt 去除本机路径，但保留可供 Hypothesis 的科学正文。"""

        value = deepcopy(dict(result))
        status = value.get("paper_search_status")
        if isinstance(status, Mapping):
            status_copy = deepcopy(dict(status))
            for field in ("manifest_path", "resume_state_path", "response_cache_dir"):
                raw = status_copy.get(field)
                if isinstance(raw, str) and raw:
                    status_copy[field] = PurePosixPath(raw.replace("\\", "/")).name
            value["paper_search_status"] = status_copy
        papers = value.get("retrieved_papers")
        if isinstance(papers, list):
            value["retrieved_papers"] = [
                PurePosixPath(item.replace("\\", "/")).name
                if isinstance(item, str) and item
                else item
                for item in papers
            ]
        return value

    def begin_router_operation(
        self,
        *,
        task_id: str,
        combination_id: str,
        expected_version: int,
        expected_identity: Mapping[str, Any],
        causal_source_ref: Optional[str],
    ) -> Dict[str, Any]:
        """在完整执行或可信负向前缀后，以 CAS 推进到 Router 调用边界。

        Router 没有 execute_step 风格的持久 claim；任务文件锁负责单 runner，版本
        和完整执行身份负责拒绝旧模型结果。这里仍先写入 routing 投影，使模型调用
        失败时可以按既有 Pause 协议原子停住，而不是把失败挂在 running 状态上。
        """

        store = self._require_enabled()
        identity = dict(expected_identity)
        if set(identity) != {
            "workflow_id",
            "workflow_version",
            "attempt_index",
            "execution_id",
        }:
            raise ContractValidationError("Router expected_identity 字段不完整")
        with store.immediate_transaction() as conn:
            self._assert_task_writable(conn, task_id)
            row = conn.execute(
                "SELECT * FROM combinations WHERE task_id=? AND combination_id=?",
                (task_id, combination_id),
            ).fetchone()
            if row is None:
                raise ContractValidationError("Router 引用了未知 combination")
            runtime = self._decode_json(
                row["runtime_checkpoint_json"],
                field="runtime_checkpoint_json",
            )
            workflow = self._decode_json(
                row["current_workflow_json"],
                field="current_workflow_json",
            )
            steps = workflow.get("steps") if isinstance(workflow, dict) else None
            completed_refs = runtime.get("step_manifest_refs")
            next_step = runtime.get("next_step_index")
            identity_matches = all(
                row[field] == value for field, value in identity.items()
            )
            if (
                row["version"] != expected_version
                or row["state"] != "running"
                or row["causal_source_ref"] != causal_source_ref
                or not identity_matches
                or not isinstance(steps, list)
                or not isinstance(completed_refs, list)
                or isinstance(next_step, bool)
                or not isinstance(next_step, int)
                or not (1 <= next_step <= len(steps))
                or next_step != len(completed_refs)
                or runtime.get("current_operation_instance_id") is not None
                or runtime.get("current_request_ref") is not None
                or runtime.get("checkpoint_id") is not None
            ):
                raise CASConflict(
                    "Router 开始边界的 CAS 前提已变化",
                    expected_version=expected_version,
                    current_version=row["version"],
                    expected_state="running",
                    current_state=row["state"],
                )
            cursor = conn.execute(
                "UPDATE combinations SET state='routing', version=version+1, updated_at=? "
                "WHERE task_id=? AND combination_id=? AND state='running' AND version=?",
                (utc_now(), task_id, combination_id, expected_version),
            )
            if cursor.rowcount != 1:
                raise CASConflict(
                    "Router 开始边界输掉并发 CAS",
                    expected_version=expected_version,
                    current_version=row["version"],
                    expected_state="running",
                    current_state=row["state"],
                )
            routed = conn.execute(
                "SELECT * FROM combinations WHERE task_id=? AND combination_id=?",
                (task_id, combination_id),
            ).fetchone()
            return combination_row(routed)

    def complete_attempt(
        self,
        task_id: str,
        payload: Mapping[str, Any],
        source_refs: Sequence[str],
        *,
        expected_version: int,
    ) -> str:
        """提交一次可解释的科学尝试，并按 Router 决定推进组合状态。

        工具技术失败没有形成可解释的科学结果，不能调用本方法。通常需要完成全部
        步骤；若已完成前缀出现可信负向程序检查，也可只携带真实前缀进入工作流
        修订。当前执行权必须已经释放，程序检查也必须给出合法结论。
        """

        store = self._require_enabled()
        value = validate_attempt_payload(dict(payload))
        refs = validate_source_ref_shape(list(source_refs))
        if len(refs) != 2 or any("/" not in item for item in refs):
            raise ContractValidationError(
                "Attempt.source_refs 必须恰含 execution 与 Router finalized manifest"
            )
        # 先复核当前 MiniChem 的 Gate，再查重复事件。这样即使数据库中已有一条
        # 同稳定键的旧记录，损坏或被篡改的 evidence_ready 也不能借幂等重放绕过 Gate。
        self._validate_minichem_attempt_program_gate(value)
        with store.immediate_transaction() as conn:
            row = conn.execute(
                "SELECT * FROM combinations WHERE task_id=? AND combination_id=?",
                (task_id, value["combination_id"]),
            ).fetchone()
            if row is None:
                raise ContractValidationError("Attempt 引用未知 combination")
            runtime = self._decode_json(
                row["runtime_checkpoint_json"], field="runtime_checkpoint_json"
            )
            workflow = self._decode_json(
                row["current_workflow_json"], field="current_workflow_json"
            )
            steps = workflow.get("steps") if isinstance(workflow, dict) else None
            if not isinstance(steps, list):
                raise ContractValidationError("当前 workflow 缺少可执行 steps")
            completed_refs = runtime.get("step_manifest_refs")
            completed_step_count = runtime.get("next_step_index")
            execution_steps = value["execution_result"].get("steps")
            if (
                not isinstance(completed_refs, list)
                or isinstance(completed_step_count, bool)
                or not isinstance(completed_step_count, int)
                or not (1 <= completed_step_count <= len(steps))
                or completed_step_count != len(completed_refs)
                or not isinstance(execution_steps, list)
                or len(execution_steps) != completed_step_count
            ):
                raise ContractValidationError("Attempt 的真实完成步骤前缀不一致")
            stopped_early = completed_step_count < len(steps)
            if stopped_early:
                gate = value["program_validation"].get("program_gate_result")
                if (
                    value["execution_result"].get("workflow_outcome")
                    != "stopped_after_required_check_not_matched"
                    or not isinstance(gate, Mapping)
                    or gate.get("allowed_decisions") != ["revise_workflow"]
                    or value["router_decision"].get("decision")
                    != "revise_workflow"
                ):
                    raise ContractValidationError(
                        "提前结束的 Attempt 只能进入工作流修订"
                    )
            self._validate_attempt_source_manifests(
                conn,
                task_id=task_id,
                row=row,
                runtime=runtime,
                workflow=workflow,
                payload=value,
                source_refs=refs,
            )
            event_key = event_key_attempt(
                task_id,
                value["combination_id"],
                row["attempt_index"],
                value["workflow_version"],
                value["execution_id"],
            )
            existing = self._existing_event_exact(
                conn,
                event_key=event_key,
                task_id=task_id,
                round_id=row["attempt_index"],
                event_type="attempt",
                payload=value,
                source_refs=refs,
            )
            if existing is not None:
                return existing
            if (
                row["version"] != expected_version
                or row["state"] != "routing"
                or row["workflow_version"] != value["workflow_version"]
                or row["execution_id"] != value["execution_id"]
                or row["causal_source_ref"] != value["causal_source_ref"]
                or runtime.get("next_step_index") != len(completed_refs)
                or runtime.get("current_operation_instance_id") is not None
                or runtime.get("current_request_ref") is not None
            ):
                raise CASConflict(
                    "Attempt 的 CAS 前提已变化",
                    expected_version=expected_version,
                    current_version=row["version"],
                    expected_state="routing",
                    current_state=row["state"],
                )
            # ExpertDebate 与随后 Attempt 共享一个业务 operation 身份；来源清单已经
            # 证明它们属于同一 debate 事实，因此这里只允许跳过这一种已提交事件，
            # 其他同身份终态仍然报告冲突。
            self._assert_operation_available(
                conn,
                task_id,
                value["operation_instance_id"],
                allow_event_types={"expert_debate_completed"},
            )
            decision = value["router_decision"]["decision"]
            if decision == "evidence_ready":
                next_state = "evidence_ready"
                terminal_reason = value["terminal_reason"]
            elif decision == "revise_hypothesis":
                next_state = "hypothesis_failed"
                terminal_reason = value["terminal_reason"]
            elif value["terminal_reason"] == "workflow_revision_limit_exhausted":
                next_state = "hypothesis_failed"
                terminal_reason = value["terminal_reason"]
            else:
                next_state = "revising_workflow"
                terminal_reason = None
            event_id, _ = self._insert_event(
                conn,
                event_key=event_key,
                task_id=task_id,
                round_id=row["attempt_index"],
                event_type="attempt",
                payload=value,
                source_refs=refs,
                check_operation_occupancy=True,
                allow_operation_event_types={"expert_debate_completed"},
            )
            terminal_record_id = (
                event_id
                if next_state in {"evidence_ready", "hypothesis_failed"}
                else None
            )
            cursor = conn.execute(
                """
                UPDATE combinations SET state=?, terminal_record_id=?,
                    terminal_reason=?, causal_source_ref=NULL, version=version+1,
                    updated_at=?
                WHERE task_id=? AND combination_id=? AND state='routing' AND version=?
                """,
                (
                    next_state,
                    terminal_record_id,
                    terminal_reason,
                    utc_now(),
                    task_id,
                    value["combination_id"],
                    expected_version,
                ),
            )
            if cursor.rowcount != 1:
                raise CASConflict(
                    "Attempt 输掉并发 CAS",
                    expected_version=expected_version,
                    current_version=row["version"],
                    expected_state="routing",
                    current_state=row["state"],
                )
            self._reduce_task_status_in_transaction(conn, task_id)
            return event_id

    @staticmethod
    def _looks_like_minichem_attempt(payload: Mapping[str, Any]) -> bool:
        """根据执行记录中的 MiniChem 身份判断是否必须使用当前 Gate 契约。

        新执行结果会在 metadata 或步骤清单中保存 catalog、执行层、Action/Backend
        或 MCP 作业身份。历史非 MiniChem Attempt 没有这些字段，仍可由只读投影读取；
        一旦发现任一身份字段，就把缺失的 Program Gate 视为持久化错误而不是旧格式。
        """

        execution_result = payload.get("execution_result")
        if not isinstance(execution_result, Mapping):
            return False
        metadata = execution_result.get("metadata")
        if isinstance(metadata, Mapping) and "catalog_hash" in metadata:
            return True
        steps = execution_result.get("steps")
        if not isinstance(steps, list):
            return False
        for step in steps:
            if not isinstance(step, Mapping):
                continue
            if any(
                field in step
                for field in ("execution_layer", "action_id", "backend_id")
            ):
                return True
            raw_output = step.get("raw_output")
            if isinstance(raw_output, Mapping) and any(
                field in raw_output
                for field in (
                    "output_artifacts",
                    "declared_artifacts",
                    "operation_id",
                    "job_id",
                )
            ):
                return True
        validation = payload.get("program_validation")
        gate = validation.get("program_gate_result") if isinstance(validation, Mapping) else None
        return isinstance(gate, Mapping) and "catalog_hash" in gate

    @classmethod
    def _validate_minichem_attempt_program_gate(
        cls,
        payload: Mapping[str, Any],
    ) -> None:
        """核对 MiniChem Attempt 的持久化 Program Gate 事实。

        旧 Router 和执行后 Expert Debate 都核对同一 Attempt 的路由决定与 Gate
        允许列表；只在 evidence_ready 时复验 Gate 的可靠性，不在这里重新推断
        科学结论。
        """

        if not cls._looks_like_minichem_attempt(payload):
            return
        program_validation = payload.get("program_validation")
        if not isinstance(program_validation, Mapping):
            raise ContractValidationError(
                "当前 MiniChem Attempt 缺少合法 program_validation"
            )
        gate = program_validation.get("program_gate_result")
        if not isinstance(gate, Mapping):
            raise ContractValidationError(
                "当前 MiniChem Attempt 缺少 program_gate_result"
            )
        decision = payload.get("router_decision", {}).get("decision")
        allowed = gate.get("allowed_decisions")
        if not isinstance(allowed, list) or decision not in allowed:
            raise ContractValidationError(
                "MiniChem Router decision 不在 Program Gate allowed_decisions 中"
            )
        if decision == "evidence_ready":
            try:
                validate_program_gate_for_evidence_ready(program_validation)
            except ValueError as exc:
                raise ContractValidationError(
                    "MiniChem Program Gate 不允许 evidence_ready"
                ) from exc

    def _validate_attempt_source_manifests(
        self,
        conn,
        *,
        task_id: str,
        row: Any,
        runtime: Mapping[str, Any],
        workflow: Mapping[str, Any],
        payload: Mapping[str, Any],
        source_refs: Sequence[str],
    ) -> None:
        """拒绝旧 Workflow V1 的 Attempt 清单验证入口。

        当前生产任务只使用 ``investigation_v2`` 模块输出。旧 Attempt/Router
        证据协议依赖已经删除的 V1 agent，不能在 V2 Working Memory 中继续以
        隐式兼容方式执行；历史数据如需处理必须走显式、只读迁移工具。
        """

        del conn, task_id, row, runtime, workflow, payload, source_refs
        raise ContractValidationError(
            "legacy_workflow_v1 Attempt 清单验证已移除；请使用 investigation_v2 模块输出"
        )

    def _validate_minichem_execution_manifest(
        self,
        conn,
        *,
        task_id: str,
        manifest: Mapping[str, Any],
        execution_result: Mapping[str, Any],
    ) -> None:
        """复算 MiniChem 最终清单中的执行层、作业和产物身份。

        输入是当前任务、完整 execution manifest 和同一 Attempt 的执行结果。方法读取
        每份已完成步骤清单，用真实步骤顺序重新构造所有扩展审计数组；目录哈希、层
        身份、operation/job、ArtifactRef 或 provenance 任一不一致都会拒绝 Attempt。
        本方法只读取正式文件和数据库，不写入状态。
        """

        created = self._task_created_payload(conn, task_id)
        if manifest.get("catalog_hash") != created["tool_catalog_hash"]:
            raise ContractValidationError("MiniChem execution manifest 的目录哈希不一致")
        steps = execution_result.get("steps")
        manifest_refs = manifest.get("step_manifest_refs")
        if (
            not isinstance(steps, list)
            or not isinstance(manifest_refs, list)
            or len(steps) != len(manifest_refs)
        ):
            raise ContractValidationError("MiniChem execution manifest 的步骤前缀不完整")

        identities = []
        for index, (step, step_manifest_ref) in enumerate(zip(steps, manifest_refs)):
            if not isinstance(step, Mapping):
                raise ContractValidationError("MiniChem ExecutionResult step 必须是 object")
            step_manifest = self._read_finalized_manifest(
                conn,
                task_id,
                step_manifest_ref,
            )
            layer = step.get("execution_layer")
            if layer not in {
                "predefined_actions",
                "native_software",
                "programmable_analysis",
            }:
                raise ContractValidationError("MiniChem ExecutionResult execution_layer 非法")
            identity = {
                "step_id": step.get("step_id"),
                "status": step.get("status", "completed"),
                "execution_layer": layer,
                "operation_id": None,
                "job_id": None,
                "artifact_refs": deepcopy(step.get("output_artifacts", [])),
                "provenance": None,
            }
            if (
                not isinstance(identity["step_id"], str)
                or not identity["step_id"]
                or step_manifest.get("step_id") != identity["step_id"]
                or step_manifest.get("step_index") != index
                or step_manifest.get("execution_layer") != layer
            ):
                raise ContractValidationError("MiniChem 步骤清单的顺序或执行层不一致")
            if identity["status"] not in {"completed", "failed", "not_executed"}:
                raise ContractValidationError("MiniChem ExecutionResult step status 非法")
            if step_manifest.get("status") != identity["status"]:
                raise ContractValidationError("MiniChem 步骤清单 status 与执行结果不一致")
            if layer == "predefined_actions":
                for field in ("action_id", "backend_id"):
                    value = step.get(field)
                    if (
                        not isinstance(value, str)
                        or not value
                        or step_manifest.get(field) != value
                    ):
                        raise ContractValidationError(
                            f"MiniChem predefined step 的 {field} 不一致"
                        )
                    identity[field] = value
            else:
                field = "software_id" if layer == "native_software" else "runtime"
                value = step.get(field)
                if (
                    not isinstance(value, str)
                    or not value
                    or step_manifest.get(field) != value
                ):
                    raise ContractValidationError(
                        f"MiniChem open execution 的 {field} 不一致"
                    )
                identity[field] = value
            raw_output = step.get("raw_output")
            if not isinstance(raw_output, Mapping):
                raise ContractValidationError("MiniChem ExecutionResult 缺少 raw_output")
            if identity["status"] in {"failed", "not_executed"}:
                operation_id = raw_output.get("operation_id") or step_manifest.get(
                    "operation_instance_id"
                )
                if not isinstance(operation_id, str) or not operation_id:
                    raise ContractValidationError("MiniChem 失败步骤缺少 operation 身份")
                identity["operation_id"] = operation_id
                job_id = raw_output.get("job_id") or step_manifest.get("job_id")
                if job_id is not None and (not isinstance(job_id, str) or not job_id):
                    raise ContractValidationError("MiniChem 失败步骤 job_id 非法")
                identity["job_id"] = job_id
                artifacts = raw_output.get("output_artifacts", [])
                if not isinstance(artifacts, list):
                    raise ContractValidationError("MiniChem 失败步骤 output_artifacts 非法")
                identity["artifact_refs"] = deepcopy(artifacts)
                provenance = raw_output.get("provenance")
                if provenance is not None and not isinstance(provenance, Mapping):
                    raise ContractValidationError("MiniChem 失败步骤 provenance 非法")
                identity["provenance"] = deepcopy(provenance)
                identities.append(identity)
                continue
            operation_id = raw_output.get("operation_id") or step_manifest.get(
                "operation_instance_id"
            )
            if not isinstance(operation_id, str) or not operation_id:
                raise ContractValidationError("MiniChem 步骤缺少 operation 身份")
            identity["operation_id"] = operation_id
            job_id = raw_output.get("job_id")
            if job_id is not None and (not isinstance(job_id, str) or not job_id):
                raise ContractValidationError("MiniChem 步骤 job_id 非法")
            identity["job_id"] = job_id
            artifacts = raw_output.get("output_artifacts", step.get("output_artifacts", []))
            if not isinstance(artifacts, list):
                raise ContractValidationError("MiniChem 步骤 output_artifacts 非法")
            identity["artifact_refs"] = deepcopy(artifacts)
            provenance = raw_output.get("provenance")
            if provenance is not None and not isinstance(provenance, Mapping):
                raise ContractValidationError("MiniChem 步骤 provenance 非法")
            identity["provenance"] = deepcopy(provenance)
            identities.append(identity)

        expected = {
            "execution_layer_ids": [item["execution_layer"] for item in identities],
            "action_ids": [
                item["action_id"]
                for item in identities
                if item["execution_layer"] == "predefined_actions"
            ],
            "backend_ids": [
                item["backend_id"]
                for item in identities
                if item["execution_layer"] == "predefined_actions"
            ],
            "software_ids": [
                item["software_id"]
                for item in identities
                if item["execution_layer"] == "native_software"
            ],
            "runtimes": [
                item["runtime"]
                for item in identities
                if item["execution_layer"] == "programmable_analysis"
            ],
            "operation_ids": [item["operation_id"] for item in identities],
            "job_ids": [item["job_id"] for item in identities if item["job_id"] is not None],
            "action_backend_identities": [
                {
                    "step_id": item["step_id"],
                    "action_id": item["action_id"],
                    "backend_id": item["backend_id"],
                }
                for item in identities
                if item["execution_layer"] == "predefined_actions"
            ],
            "operation_job_identities": [
                {
                    "step_id": item["step_id"],
                    "execution_layer": item["execution_layer"],
                    "operation_id": item["operation_id"],
                    "job_id": item["job_id"],
                    **(
                        {"software_id": item["software_id"]}
                        if item["execution_layer"] == "native_software"
                        else {"runtime": item["runtime"]}
                        if item["execution_layer"] == "programmable_analysis"
                        else {}
                    ),
                }
                for item in identities
                if item["job_id"] is not None
            ],
            "artifact_refs": [
                deepcopy(artifact)
                for item in identities
                for artifact in item["artifact_refs"]
            ],
            "provenance": [
                deepcopy(item["provenance"])
                for item in identities
                if item["provenance"] is not None
            ],
        }
        for field, expected_value in expected.items():
            if manifest.get(field) != expected_value:
                raise ContractValidationError(
                    f"MiniChem execution manifest 的 {field} 与步骤证据不一致"
                )

    def _verify_final_results_manifest(
        self,
        conn,
        *,
        task_id: str,
        batch_id: str,
        batch_index: int,
        close_timestamp: str,
        manifest_ref: str,
        expected_results: Sequence[Mapping[str, Any]],
    ) -> None:
        """复验成功批次的结果索引，以及每份受大小限制的 JSON 结果文件。

        索引中的组合身份、相对路径、大小和摘要都必须与真实文件一致，避免数据库
        宣布任务成功后，用户拿到的却是缺失、越界或后来被改写的结果。
        """

        manifest = self._read_finalized_manifest(conn, task_id, manifest_ref)
        if set(manifest) != {
            "batch_id",
            "batch_index",
            "close_timestamp",
            "files",
        }:
            raise ContractValidationError("final-results manifest 字段不完整")
        if (
            manifest["batch_id"] != batch_id
            or manifest["batch_index"] != batch_index
            or manifest["close_timestamp"] != close_timestamp
        ):
            raise ContractValidationError("final-results manifest 批次身份不一致")
        files = manifest["files"]
        evidence_results = [
            item
            for item in expected_results
            if item["terminal_state"] == "evidence_ready"
        ]
        if not isinstance(files, list) or len(files) != len(evidence_results):
            raise ContractValidationError("final-results manifest 未逐项覆盖成功组合")
        task_root = self._task_output_root(conn, task_id)
        for expected, item in zip(evidence_results, files):
            if not isinstance(item, dict) or set(item) != {
                "combination_id",
                "relative_path",
                "sha256",
                "media_type",
                "size",
            }:
                raise ContractValidationError("final-results file item 字段不完整")
            if (
                item["combination_id"] != expected["combination_id"]
                or item["relative_path"]
                != expected["final_result_relative_path"]
                or item["media_type"] != "application/json"
            ):
                raise ContractValidationError("final-results file 与组合结果不一致")
            target = (task_root / item["relative_path"]).resolve()
            try:
                target.relative_to(task_root)
                content = target.read_bytes()
            except (ValueError, OSError) as exc:
                raise ContractValidationError("bounded result 文件不存在或越界") from exc
            if (
                isinstance(item["size"], bool)
                or not isinstance(item["size"], int)
                or item["size"] != len(content)
                or item["sha256"] != hashlib.sha256(content).hexdigest()
            ):
                raise ContractValidationError("bounded result 的 size 或 sha256 不匹配")

    def _verify_terminal_result_files(
        self,
        conn,
        *,
        task_id: str,
    ) -> List[str]:
        """沿 TaskClosed 和 BatchClosed 事实重新核对最终结果文件。"""

        task_event = conn.execute(
            "SELECT event_id, event_type, round, payload_json, source_refs_json, "
            "created_at FROM memory_events WHERE task_id=? AND event_key=?",
            (task_id, event_key_task_closed(task_id)),
        ).fetchone()
        if task_event is None or task_event["event_type"] != "task_closed":
            raise ContractValidationError("终态 task 缺少唯一 TaskClosed 事实")
        try:
            task_payload = json.loads(task_event["payload_json"])
            task_refs = validate_source_ref_shape(
                json.loads(task_event["source_refs_json"])
            )
        except (TypeError, json.JSONDecodeError) as exc:
            raise ContractValidationError("TaskClosed JSON 已损坏") from exc
        if not isinstance(task_payload, dict) or set(task_payload) != {
            "final_batch_index",
            "successful_result_relative_paths",
            "stop_reason",
        }:
            raise ContractValidationError("TaskClosed payload 字段不匹配")
        final_batch_index = task_payload["final_batch_index"]
        if (
            isinstance(final_batch_index, bool)
            or not isinstance(final_batch_index, int)
            or final_batch_index < 1
            or task_event["round"] != final_batch_index
        ):
            raise ContractValidationError("TaskClosed 最终批次索引不一致")
        successful_paths = task_payload["successful_result_relative_paths"]
        if not isinstance(successful_paths, list):
            raise ContractValidationError("TaskClosed 成功结果路径必须是数组")
        for path in successful_paths:
            validate_task_relative_path(path, "successful_result_relative_paths")
        expected_stop_reason = (
            "supported_hypothesis_found"
            if successful_paths
            else "max_hypothesis_batches_exhausted"
        )
        if task_payload["stop_reason"] != expected_stop_reason:
            raise ContractValidationError("TaskClosed 停止原因与结果路径不一致")

        task_projection = conn.execute(
            "SELECT status FROM tasks WHERE task_id=?",
            (task_id,),
        ).fetchone()
        expected_task_status = (
            "succeeded" if successful_paths else "stopped_no_supported_hypothesis"
        )
        if (
            task_projection is None
            or task_projection["status"] != expected_task_status
        ):
            raise ContractValidationError("TaskClosed 与 task 终态不一致")
        batch_projection = conn.execute(
            "SELECT batch_id, batch_index, state FROM hypothesis_batches "
            "WHERE task_id=? AND batch_index=?",
            (task_id, final_batch_index),
        ).fetchone()
        if batch_projection is None:
            raise ContractValidationError("TaskClosed 引用的最终批次不存在")
        batch_event = conn.execute(
            "SELECT event_id, event_type, round, payload_json, source_refs_json, "
            "created_at FROM memory_events WHERE task_id=? AND event_key=?",
            (
                task_id,
                event_key_batch_closed(task_id, batch_projection["batch_id"]),
            ),
        ).fetchone()
        if (
            batch_event is None
            or batch_event["event_type"] != "hypothesis_batch_closed"
            or batch_event["round"] != final_batch_index
            or batch_event["created_at"] != task_event["created_at"]
        ):
            raise ContractValidationError("TaskClosed 缺少匹配的 BatchClosed 事实")
        try:
            batch_payload = validate_batch_closed_payload(
                json.loads(batch_event["payload_json"])
            )
            batch_refs = validate_source_ref_shape(
                json.loads(batch_event["source_refs_json"])
            )
        except (TypeError, json.JSONDecodeError) as exc:
            raise ContractValidationError("BatchClosed JSON 已损坏") from exc
        if batch_payload["batch_index"] != final_batch_index:
            raise ContractValidationError("BatchClosed 批次索引与 TaskClosed 不一致")
        if task_refs != sorted([batch_event["event_id"], *batch_refs]):
            raise ContractValidationError("TaskClosed 来源与 BatchClosed 不一致")
        self._validate_source_refs(conn, task_id, task_refs)

        expected_paths = [
            item["final_result_relative_path"]
            for item in batch_payload["combination_results"]
            if item["terminal_state"] == "evidence_ready"
        ]
        if successful_paths != expected_paths:
            raise ContractValidationError("TaskClosed 结果路径与 BatchClosed 不一致")
        if batch_projection["state"] != batch_payload["batch_outcome"]:
            raise ContractValidationError("BatchClosed 与批次投影终态不一致")
        if batch_payload["batch_outcome"] == "succeeded":
            if len(batch_refs) != 1 or "/" not in batch_refs[0]:
                raise ContractValidationError(
                    "成功 BatchClosed 必须恰引用 final-results manifest"
                )
            self._verify_final_results_manifest(
                conn,
                task_id=task_id,
                batch_id=batch_projection["batch_id"],
                batch_index=final_batch_index,
                close_timestamp=batch_event["created_at"],
                manifest_ref=batch_refs[0],
                expected_results=batch_payload["combination_results"],
            )
        elif batch_refs:
            raise ContractValidationError("失败 BatchClosed 不得引用结果清单")
        return list(expected_paths)

    def read_terminal_result_paths(
        self,
        task_id: str,
        task_status: str,
    ) -> List[str]:
        """按终态类型读取结果路径，并核对新版终态的完整事实链。

        输入是任务身份和任务当前终态。旧版 ``completed`` 与 ``stopped`` 直接读取
        数据库中按批次和组合序号排列的非空路径，使用普通数据库读取，也不要求新
        版关闭事实，以保留旧恢复行为。新版 ``succeeded`` 与
        ``stopped_no_supported_hypothesis`` 则在同一个数据库事务中，
        先按恢复器原有顺序检查 TaskClosed 的字段、结果路径、任务状态和来源，再按
        批次及组合序号读取路径，复验 BatchClosed、最终结果清单和每个
        结果文件的大小与 SHA-256。方法不写数据库，返回新的 ``list[str]``；新版
        终态没有路径、停止终态含路径、关闭事实缺失或文件被删除/篡改时抛出
        ``ContractValidationError``。
        """

        store = self._require_enabled()
        require_nonempty_string(task_id, "task_id")
        require_nonempty_string(task_status, "task_status")

        if task_status in LEGACY_TERMINAL_TASK_STATES:
            result_rows = store.fetchall(
                "SELECT c.final_result_relative_path FROM combinations c "
                "JOIN hypothesis_batches b ON b.task_id=c.task_id AND b.batch_id=c.batch_id "
                "WHERE c.task_id=? AND c.final_result_relative_path IS NOT NULL "
                "ORDER BY b.batch_index, c.combination_index",
                (task_id,),
            )
            return [row["final_result_relative_path"] for row in result_rows]

        with store.immediate_transaction() as conn:
            # 这一段保留旧恢复器终态字段校验的顺序和错误边界；后面的完整关闭链
            # 复验仍在当前事务内进行，不能让较新的通用文件核对先吞掉字段损坏、
            # 状态冲突等更早错误。
            row = conn.execute(
                "SELECT event_type, payload_json, source_refs_json FROM memory_events "
                "WHERE task_id=? AND event_key=?",
                (task_id, event_key_task_closed(task_id)),
            ).fetchone()
            if row is None or row["event_type"] != "task_closed":
                raise ContractValidationError("终态 task 缺少唯一 TaskClosed 事实")
            try:
                payload = json.loads(row["payload_json"])
                source_refs = json.loads(row["source_refs_json"])
            except (TypeError, json.JSONDecodeError) as exc:
                raise ContractValidationError("TaskClosed JSON 已损坏") from exc
            if not isinstance(payload, dict) or set(payload) != {
                "final_batch_index",
                "successful_result_relative_paths",
                "stop_reason",
            }:
                raise ContractValidationError("TaskClosed payload 字段不匹配")
            successful = payload["successful_result_relative_paths"]
            if not isinstance(successful, list) or any(
                not isinstance(item, str) or not item for item in successful
            ):
                raise ContractValidationError("TaskClosed 成功结果路径字段损坏")
            expected_status = (
                "succeeded" if successful else "stopped_no_supported_hypothesis"
            )
            if task_status != expected_status:
                raise ContractValidationError("TaskClosed 与 task 终态不一致")
            self._validate_source_refs(conn, task_id, source_refs)

            result_rows = conn.execute(
                "SELECT c.final_result_relative_path FROM combinations c "
                "JOIN hypothesis_batches b ON b.task_id=c.task_id AND b.batch_id=c.batch_id "
                "WHERE c.task_id=? AND c.final_result_relative_path IS NOT NULL "
                "ORDER BY b.batch_index, c.combination_index",
                (task_id,),
            ).fetchall()
            paths = [row["final_result_relative_path"] for row in result_rows]
            if task_status == "succeeded" and not paths:
                raise ContractValidationError("成功任务缺少最终结果路径")
            if task_status == "stopped_no_supported_hypothesis" and paths:
                raise ContractValidationError("停止任务不应包含成功结果路径")

            verified_paths = self._verify_terminal_result_files(
                conn,
                task_id=task_id,
            )
            if paths != verified_paths:
                raise ContractValidationError("终态投影路径与最终结果清单不一致")
            return deepcopy(list(verified_paths))

    def verify_terminal_result_files(self, task_id: str) -> List[str]:
        """从不可变关闭事实复验终态文件，并返回固定顺序的结果路径。"""

        require_nonempty_string(task_id, "task_id")
        store = self._require_enabled()
        with store.immediate_transaction() as conn:
            return self._verify_terminal_result_files(conn, task_id=task_id)

    # ==================== 终态结果与任务关闭 ====================

    def close_hypothesis_batch(
        self,
        task_id: str,
        batch_id: str,
        payload: Mapping[str, Any],
        source_refs: Sequence[str],
        *,
        expected_batch_version: int,
        expected_task_version: int,
        close_timestamp: str,
    ) -> Dict[str, Optional[str]]:
        """一次性关闭已经排空的批次，并在应当结束时同时关闭任务。

        受大小限制的结果文件及其总索引必须先完成原子发布。数据库事务随后重新核对
        每个组合的终态和版本；若另一进程已经推进任务，当前进程就不能覆盖已有结论。
        成功批次立即结束任务；整批无成功结果时只在批次预算耗尽后结束任务。
        """

        store = self._require_enabled()
        value = validate_batch_closed_payload(dict(payload))
        require_nonempty_string(batch_id, "batch_id")
        close_time = self._parse_event_timestamp(
            close_timestamp,
            field="close_timestamp",
        )
        refs = validate_source_ref_shape(list(source_refs))
        if value["batch_outcome"] == "succeeded":
            if len(refs) != 1 or "/" not in refs[0]:
                raise ContractValidationError(
                    "成功 BatchClosed 必须恰引用 final-results manifest"
                )
        elif refs:
            raise ContractValidationError("失败 BatchClosed 不得引用空结果清单")
        batch_key = event_key_batch_closed(task_id, batch_id)
        task_key = event_key_task_closed(task_id)
        with store.immediate_transaction() as conn:
            # 最终关闭事件会按这个时间参与投影重放。如果时间早于来源 Attempt 或
            # 暂停记录，排序后就可能先关闭任务、再重新打开组合，所以必须在写入前
            # 与当前任务全部历史逐条比较。
            prior_event_times = conn.execute(
                "SELECT created_at FROM memory_events WHERE task_id=?",
                (task_id,),
            ).fetchall()
            if any(
                close_time
                < self._parse_event_timestamp(
                    row["created_at"],
                    field="memory_events.created_at",
                )
                for row in prior_event_times
            ):
                raise ContractValidationError(
                    "批次关闭时间不能早于已有任务事实"
                )
            existing_batch = self._existing_event_exact(
                conn,
                event_key=batch_key,
                task_id=task_id,
                round_id=value["batch_index"],
                event_type="hypothesis_batch_closed",
                payload=value,
                source_refs=refs,
            )
            if existing_batch is not None:
                batch_event = conn.execute(
                    "SELECT created_at FROM memory_events WHERE event_id=?",
                    (existing_batch,),
                ).fetchone()
                if batch_event["created_at"] != close_timestamp:
                    raise IdempotencyConflict(
                        "BatchClosed 重放使用了不同关闭时间"
                    )
                task_config = self._task_created_payload(conn, task_id)["task_config"]
                should_close = value["batch_outcome"] == "succeeded" or (
                    value["batch_index"] >= task_config["max_hypothesis_batches"]
                )
                task_event_id = None
                if should_close:
                    successful_paths = [
                        item["final_result_relative_path"]
                        for item in value["combination_results"]
                        if item["terminal_state"] == "evidence_ready"
                    ]
                    expected_task_payload = {
                        "final_batch_index": value["batch_index"],
                        "successful_result_relative_paths": successful_paths,
                        "stop_reason": (
                            "supported_hypothesis_found"
                            if successful_paths
                            else "max_hypothesis_batches_exhausted"
                        ),
                    }
                    task_event_id = self._existing_event_exact(
                        conn,
                        event_key=task_key,
                        task_id=task_id,
                        round_id=value["batch_index"],
                        event_type="task_closed",
                        payload=expected_task_payload,
                        source_refs=sorted([existing_batch, *refs]),
                    )
                    if task_event_id is None:
                        raise ContractValidationError(
                            "BatchClosed 已存在但配套 TaskClosed 缺失"
                        )
                    task_event = conn.execute(
                        "SELECT created_at FROM memory_events WHERE event_id=?",
                        (task_event_id,),
                    ).fetchone()
                    if task_event["created_at"] != close_timestamp:
                        raise IdempotencyConflict(
                            "TaskClosed 重放使用了不同关闭时间"
                        )
                else:
                    unexpected = conn.execute(
                        "SELECT event_id FROM memory_events WHERE task_id=? AND event_key=?",
                        (task_id, task_key),
                    ).fetchone()
                    if unexpected is not None:
                        raise ContractValidationError(
                            "未耗尽的失败批次不应存在 TaskClosed"
                        )
                if value["batch_outcome"] == "succeeded":
                    self._verify_final_results_manifest(
                        conn,
                        task_id=task_id,
                        batch_id=batch_id,
                        batch_index=value["batch_index"],
                        close_timestamp=close_timestamp,
                        manifest_ref=refs[0],
                        expected_results=value["combination_results"],
                    )
                return {
                    "batch_closed_event_id": existing_batch,
                    "task_closed_event_id": task_event_id,
                }
            batch = conn.execute(
                "SELECT * FROM hypothesis_batches WHERE task_id=? AND batch_id=?",
                (task_id, batch_id),
            ).fetchone()
            task = conn.execute(
                "SELECT * FROM tasks WHERE task_id=?", (task_id,)
            ).fetchone()
            if batch is None or task is None:
                raise ContractValidationError("BatchClosed 引用未知任务或批次")
            has_technical_pause = any(
                item["terminal_state"] == BATCH_TECHNICAL_PAUSE_STATE
                for item in value["combination_results"]
            )
            expected_task_state = "paused" if has_technical_pause else "running"
            if (
                batch["batch_index"] != value["batch_index"]
                or batch["state"] != "running"
                or batch["version"] != expected_batch_version
                or task["status"] != expected_task_state
                or task["version"] != expected_task_version
            ):
                raise CASConflict(
                    "BatchClosed 的 CAS 前提已变化",
                    expected_version=expected_batch_version,
                    current_version=batch["version"],
                    expected_state="running",
                    current_state=batch["state"],
                )
            rows = conn.execute(
                "SELECT * FROM combinations WHERE task_id=? AND batch_id=? "
                "ORDER BY combination_index",
                (task_id, batch_id),
            ).fetchall()
            if len(rows) != len(value["combination_results"]):
                raise ContractValidationError("BatchClosed 未覆盖全部 selected combinations")
            for row, result in zip(rows, value["combination_results"]):
                if result["terminal_state"] == BATCH_TECHNICAL_PAUSE_STATE:
                    pause = self._unresolved_combination_pause(
                        conn,
                        task_id,
                        row["combination_id"],
                    )
                    if (
                        row["combination_id"] != result["combination_id"]
                        or row["combination_index"] != result["combination_index"]
                        or row["state"] != "paused"
                        or pause is None
                        or pause["pause_id"] != result["terminal_record_id"]
                    ):
                        raise ContractValidationError(
                            "BatchClosed technical_pause 与当前暂停事实不一致"
                        )
                    continue
                if (
                    row["combination_id"] != result["combination_id"]
                    or row["combination_index"] != result["combination_index"]
                    or row["state"] != result["terminal_state"]
                    or row["terminal_reason"] != result["terminal_reason"]
                    or row["terminal_record_id"] != result["terminal_record_id"]
                ):
                    raise ContractValidationError(
                        "BatchClosed combination result 与当前终态不一致"
                    )
            if value["batch_outcome"] == "succeeded":
                self._verify_final_results_manifest(
                    conn,
                    task_id=task_id,
                    batch_id=batch_id,
                    batch_index=value["batch_index"],
                    close_timestamp=close_timestamp,
                    manifest_ref=refs[0],
                    expected_results=value["combination_results"],
                )
            batch_event_id, _ = self._insert_event(
                conn,
                event_key=batch_key,
                task_id=task_id,
                round_id=value["batch_index"],
                event_type="hypothesis_batch_closed",
                payload=value,
                source_refs=refs,
                check_operation_occupancy=False,
                created_at=close_timestamp,
            )
            cursor = conn.execute(
                "UPDATE hypothesis_batches SET state=?, version=version+1, updated_at=? "
                "WHERE task_id=? AND batch_id=? AND state='running' AND version=?",
                (
                    value["batch_outcome"],
                    close_timestamp,
                    task_id,
                    batch_id,
                    expected_batch_version,
                ),
            )
            if cursor.rowcount != 1:
                raise CASConflict(
                    "关闭批次时状态已经被另一进程推进",
                    expected_version=expected_batch_version,
                    current_version=batch["version"],
                    expected_state="running",
                    current_state=batch["state"],
                )
            for result in value["combination_results"]:
                cursor = conn.execute(
                    "UPDATE combinations SET final_result_relative_path=?, updated_at=? "
                    "WHERE task_id=? AND combination_id=?",
                    (
                        result["final_result_relative_path"],
                        close_timestamp,
                        task_id,
                        result["combination_id"],
                    ),
                )
                if cursor.rowcount != 1:
                    raise ContractValidationError(
                        "发布批次结果时组合投影缺失"
                    )

            task_config = self._task_created_payload(conn, task_id)["task_config"]
            should_close = value["batch_outcome"] == "succeeded" or (
                value["batch_index"] >= task_config["max_hypothesis_batches"]
            )
            task_event_id: Optional[str] = None
            if should_close:
                successful_paths = [
                    item["final_result_relative_path"]
                    for item in value["combination_results"]
                    if item["terminal_state"] == "evidence_ready"
                ]
                task_payload = {
                    "final_batch_index": value["batch_index"],
                    "successful_result_relative_paths": successful_paths,
                    "stop_reason": (
                        "supported_hypothesis_found"
                        if successful_paths
                        else "max_hypothesis_batches_exhausted"
                    ),
                }
                task_refs = sorted([batch_event_id, *refs])
                task_event_id, _ = self._insert_event(
                    conn,
                    event_key=task_key,
                    task_id=task_id,
                    round_id=value["batch_index"],
                    event_type="task_closed",
                    payload=task_payload,
                    source_refs=task_refs,
                    check_operation_occupancy=False,
                    created_at=close_timestamp,
                )
                task_status = (
                    "succeeded"
                    if successful_paths
                    else "stopped_no_supported_hypothesis"
                )
                cursor = conn.execute(
                    "UPDATE tasks SET status=?, version=version+1, updated_at=? "
                    "WHERE task_id=? AND status=? AND version=?",
                    (
                        task_status,
                        close_timestamp,
                        task_id,
                        expected_task_state,
                        expected_task_version,
                    ),
                )
                if cursor.rowcount != 1:
                    raise CASConflict(
                        "TaskClosed 输掉并发 CAS",
                        expected_version=expected_task_version,
                        current_version=task["version"],
                        expected_state=expected_task_state,
                        current_state=task["status"],
                    )
            return {
                "batch_closed_event_id": batch_event_id,
                "task_closed_event_id": task_event_id,
            }

    @staticmethod
    def _artifact_paths_in_document(value: Any) -> List[Tuple[str, Optional[Dict[str, Any]]]]:
        """找出清单正文继续引用的内容寻址文件。

        模型清单通常用四字段对象引用请求、响应和错误文件；执行清单还会用
        ``step_manifest_refs`` 这样的字符串数组串起步骤清单。这里只识别这两种
        已有表达，不遍历普通 output path，避免把任意计算结果误当作模型历史附件。
        """

        found: List[Tuple[str, Optional[Dict[str, Any]]]] = []
        reference_keys = {"relative_path", "sha256", "media_type", "size"}

        def visit(item: Any, parent_key: Optional[str] = None) -> None:
            if isinstance(item, Mapping):
                if set(item) == reference_keys:
                    path = item.get("relative_path")
                    if isinstance(path, str) and path:
                        found.append((path, deepcopy(dict(item))))
                    return
                for key, child in item.items():
                    visit(child, str(key))
                return
            if isinstance(item, list):
                for child in item:
                    visit(child, parent_key)
                return
            if (
                isinstance(item, str)
                and item
                and "/" in item
                and isinstance(parent_key, str)
                and (
                    parent_key.endswith("_manifest_ref")
                    or parent_key.endswith("_manifest_refs")
                )
            ):
                found.append((item, None))

        visit(value)
        return found

    @staticmethod
    def _artifact_digest_and_size(target: Path) -> Tuple[str, int, Optional[bytes]]:
        """按执行器保存引用时的同一算法复验文件或目录产物。"""

        if target.is_symlink():
            raise ContractValidationError("历史 artifact 不得是符号链接")
        if target.is_file():
            try:
                content = target.read_bytes()
            except OSError as exc:
                raise ContractValidationError("历史证据文件无法读取") from exc
            return hashlib.sha256(content).hexdigest(), len(content), content
        if target.is_dir():
            records = []
            total_size = 0
            try:
                children = sorted(target.rglob("*"))
                for child in children:
                    if child.is_symlink():
                        raise ContractValidationError(
                            "历史 artifact 目录不得包含符号链接"
                        )
                    if not child.is_file():
                        continue
                    content = child.read_bytes()
                    total_size += len(content)
                    records.append(
                        {
                            "path": child.relative_to(target).as_posix(),
                            "sha256": hashlib.sha256(content).hexdigest(),
                            "size": len(content),
                        }
                    )
            except OSError as exc:
                raise ContractValidationError("历史证据目录无法读取") from exc
            digest = hashlib.sha256(
                canonical_json(records).encode("utf-8")
            ).hexdigest()
            return digest, total_size, None
        raise ContractValidationError("历史 artifact 不是普通文件或目录")

    def _regeneration_evidence_item(
        self,
        conn,
        task_id: str,
        *,
        manifest_ref: str,
        relative_path: str,
        declared_ref: Optional[Mapping[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """读取一份 JSON 或文本证据，并逐值复核路径、摘要和大小。

        二进制计算产物仍由 ExecutionResult 中的结构化事实表示，不直接塞进模型
        上下文。文本 Gaussian 日志和输入文件可以保留原文，JSON 则恢复为原对象。
        """

        task_root = self._task_output_root(conn, task_id)
        media_type = "application/json"
        if declared_ref is None:
            # 顶层来源和字符串形式的子清单都必须是内容寻址 JSON 清单，继续使用
            # Working Memory 的严格清单校验；普通附件则由四字段引用自行证明完整性。
            self._validate_source_refs(conn, task_id, [relative_path])
        else:
            if set(declared_ref) != {
                "relative_path",
                "sha256",
                "media_type",
                "size",
            }:
                raise ContractValidationError("历史 artifact ref 字段不完整")
            media_type = declared_ref.get("media_type")
            if not isinstance(media_type, str) or not media_type:
                raise ContractValidationError("历史 artifact 缺少 media_type")
            validate_task_relative_path(relative_path, "历史 artifact relative_path")
            current = task_root
            for part in Path(relative_path).parts:
                current = current / part
                if current.is_symlink():
                    raise ContractValidationError("历史 artifact 不能经过符号链接")
        target = (task_root / relative_path).resolve()
        try:
            target.relative_to(task_root)
        except ValueError as exc:
            raise ContractValidationError("历史 artifact 越出 task 根") from exc
        digest, size, content = self._artifact_digest_and_size(target)
        if declared_ref is not None:
            if (
                declared_ref.get("relative_path") != relative_path
                or declared_ref.get("sha256") != digest
                or declared_ref.get("size") != size
            ):
                raise ContractValidationError("历史 artifact ref 与真实产物不一致")

        is_json = media_type == "application/json" or media_type.endswith("+json")
        is_text = media_type.startswith("text/") or media_type in {
            "chemical/x-gaussian-input",
            "chemical/x-xyz",
            "chemical/x-mdl-sdfile",
        }
        if not is_json and not is_text:
            return None
        if content is None:
            raise ContractValidationError("声明为 JSON 或文本的历史 artifact 不能是目录")
        try:
            decoded = content.decode("utf-8")
            value = json.loads(decoded) if is_json else decoded
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ContractValidationError("历史 JSON 或文本证据无法解码") from exc
        return {
            "manifest_ref": manifest_ref,
            "relative_path": relative_path,
            "sha256": digest,
            "media_type": media_type,
            "size": len(content),
            "content": value,
        }

    def _regeneration_source_evidence(
        self,
        conn,
        task_id: str,
        source_refs: Sequence[str],
    ) -> List[Dict[str, Any]]:
        """沿每份正式清单展开它实际引用的可读证据。

        每个来源清单单独展开和去重。即使两份清单引用同一文件，模型仍能看出该
        文件分别支撑了哪条事实；同一清单内部的重复引用则只保留一次。
        """

        evidence: List[Dict[str, Any]] = []
        for manifest_ref in sorted(item for item in source_refs if "/" in item):
            pending: List[Tuple[str, Optional[Dict[str, Any]]]] = [
                (manifest_ref, None)
            ]
            visited = set()
            while pending:
                relative_path, declared_ref = pending.pop(0)
                if relative_path in visited:
                    continue
                visited.add(relative_path)
                item = self._regeneration_evidence_item(
                    conn,
                    task_id,
                    manifest_ref=manifest_ref,
                    relative_path=relative_path,
                    declared_ref=declared_ref,
                )
                if item is None:
                    continue
                evidence.append(item)
                if isinstance(item["content"], (Mapping, list)):
                    pending.extend(
                        self._artifact_paths_in_document(item["content"])
                    )
        evidence.sort(key=lambda item: (item["manifest_ref"], item["relative_path"]))
        identities = [
            (item["manifest_ref"], item["relative_path"]) for item in evidence
        ]
        if len(identities) != len(set(identities)):
            raise ContractValidationError("同一历史来源重复展开了相同证据")
        return evidence

    def _regeneration_event_record(
        self,
        conn,
        task_id: str,
        row: Any,
    ) -> Dict[str, Any]:
        """把一条不可变事件包装成 Hypothesis Agent 可审计的历史记录。"""

        source_refs = self._decode_json(
            row["source_refs_json"],
            field="regeneration event.source_refs_json",
        )
        source_refs = validate_source_ref_shape(source_refs)
        return {
            "record_id": row["event_id"],
            "created_at": row["created_at"],
            "payload": self._decode_json(
                row["payload_json"],
                field="regeneration event.payload_json",
            ),
            "source_refs": source_refs,
            "source_evidence": self._regeneration_source_evidence(
                conn,
                task_id,
                source_refs,
            ),
        }

    def _regeneration_revision_record(
        self,
        conn,
        task_id: str,
        row: Any,
    ) -> Dict[str, Any]:
        """把已经应用的工作流修改连同原始模型证据投影给下一批。"""

        source_refs = self._decode_json(
            row["source_refs_json"],
            field="regeneration revision.source_refs_json",
        )
        source_refs = validate_source_ref_shape(source_refs)
        return {
            "record_id": row["revision_id"],
            "created_at": row["created_at"],
            "revision_type": row["revision_type"],
            "payload": self._decode_json(
                row["revised_result_json"],
                field="regeneration revision.revised_result_json",
            ),
            "source_refs": source_refs,
            "source_evidence": self._regeneration_source_evidence(
                conn,
                task_id,
                source_refs,
            ),
        }

    def _bounded_result_subject(
        self,
        conn,
        task_id: str,
        batch_id: str,
        combination_id: str,
    ) -> Dict[str, Any]:
        """读取成功组合、初始 Retrieval 和当前批次假设快照。"""

        task = conn.execute(
            "SELECT * FROM tasks WHERE task_id=?",
            (task_id,),
        ).fetchone()
        batch = conn.execute(
            "SELECT * FROM hypothesis_batches WHERE task_id=? AND batch_id=?",
            (task_id, batch_id),
        ).fetchone()
        combination = conn.execute(
            "SELECT * FROM combinations WHERE task_id=? AND batch_id=? "
            "AND combination_id=?",
            (task_id, batch_id, combination_id),
        ).fetchone()
        if task is None or batch is None or combination is None:
            raise ContractValidationError("bounded result 引用了未知任务、批次或组合")
        if batch["state"] not in {"running", "succeeded"}:
            raise ContractValidationError("只有成功批次中的组合可以生成最终结果")
        if combination["state"] != "evidence_ready":
            raise ContractValidationError("只有 evidence_ready 组合可以生成最终结果")
        if not combination["terminal_record_id"]:
            raise ContractValidationError("成功组合缺少 terminal_record_id")

        task_created = self._task_created_payload(conn, task_id)
        task_created_row = conn.execute(
            "SELECT created_at FROM memory_events WHERE task_id=? "
            "AND event_key=? AND event_type='task_created'",
            (task_id, event_key_task_created(task_id)),
        ).fetchone()
        if task_created_row is None:
            raise ContractValidationError("任务缺少 TaskCreated 时间")

        first_batch = conn.execute(
            "SELECT created_event_id FROM hypothesis_batches "
            "WHERE task_id=? AND batch_index=1",
            (task_id,),
        ).fetchone()
        if first_batch is None:
            raise ContractValidationError("任务缺少初始 batch")
        first_created = conn.execute(
            "SELECT payload_json FROM memory_events WHERE task_id=? "
            "AND event_id=? AND event_type='hypothesis_batch_created'",
            (task_id, first_batch["created_event_id"]),
        ).fetchone()
        current_created = conn.execute(
            "SELECT payload_json FROM memory_events WHERE task_id=? "
            "AND event_id=? AND event_type='hypothesis_batch_created'",
            (task_id, batch["created_event_id"]),
        ).fetchone()
        if first_created is None or current_created is None:
            raise ContractValidationError("批次缺少 HypothesisBatchCreated 事实")
        first_payload = self._validated_batch_created_payload(
            conn,
            task_id,
            first_created["payload_json"],
            field="initial HypothesisBatchCreated.payload_json",
        )
        batch_payload = self._validated_batch_created_payload(
            conn,
            task_id,
            current_created["payload_json"],
            field="HypothesisBatchCreated.payload_json",
        )
        snapshot = first_payload.get("question_snapshot")
        if isinstance(snapshot, Mapping):
            retrieval_phase = snapshot.get("initial_retrieval_phase")
        else:
            retrieval_phase = None
        hypothesis_phase = batch_payload.get("hypothesis_phase")
        if isinstance(hypothesis_phase, Mapping):
            ranked = hypothesis_phase.get("ranked_strategies")
        else:
            ranked = None
        if not isinstance(retrieval_phase, Mapping) or not isinstance(ranked, list):
            raise ContractValidationError("最终结果缺少初始 Retrieval 或当前假设批")
        hypotheses = [
            deepcopy(dict(item))
            for item in ranked
            if isinstance(item, Mapping)
            and item.get("hypothesis_id") == combination["hypothesis_id"]
        ]
        if len(hypotheses) != 1:
            raise ContractValidationError("成功组合没有唯一对应的 hypothesis")
        return {
            "task_created": task_created,
            "workflow_start_time": task_created_row["created_at"],
            "batch": batch,
            "combination": combination,
            "retrieval_phase": retrieval_phase,
            "hypothesis_phase": hypothesis_phase,
            "hypothesis": hypotheses[0],
        }

    def _bounded_result_attempt_history(
        self,
        conn,
        task_id: str,
        combination_id: str,
        terminal_record_id: str,
    ) -> Dict[str, Any]:
        """读取成功组合唯一规划及连续 Attempt 历史。"""

        event_rows = conn.execute(
            "SELECT * FROM memory_events WHERE task_id=? "
            "AND event_type IN ('planning_completed','attempt','execution_paused',"
            "'restart_applied') ORDER BY created_at, event_id",
            (task_id,),
        ).fetchall()
        decoded_payloads = {
            row["event_id"]: self._decode_json(
                row["payload_json"],
                field=f"{row['event_type']}.payload_json",
            )
            for row in event_rows
        }
        planning_rows = [
            row
            for row in event_rows
            if row["event_type"] == "planning_completed"
            and decoded_payloads[row["event_id"]].get("combination_id")
            == combination_id
        ]
        if len(planning_rows) != 1:
            raise ContractValidationError("成功组合必须恰有一条 PlanningCompleted")
        planning_record = self._regeneration_event_record(
            conn,
            task_id,
            planning_rows[0],
        )

        attempt_rows = [
            row
            for row in event_rows
            if row["event_type"] == "attempt"
            and decoded_payloads[row["event_id"]].get("combination_id")
            == combination_id
        ]
        if not attempt_rows:
            raise ContractValidationError("成功组合缺少 Attempt")
        attempt_records = []
        for row in attempt_rows:
            record = self._regeneration_event_record(conn, task_id, row)
            if (
                isinstance(row["round"], bool)
                or not isinstance(row["round"], int)
                or row["round"] < 1
            ):
                raise ContractValidationError("Attempt round 非法")
            record["attempt_index"] = row["round"]
            attempt_records.append(record)
        attempt_records.sort(
            key=lambda item: (item["attempt_index"], item["record_id"])
        )
        if [item["attempt_index"] for item in attempt_records] != list(
            range(1, len(attempt_records) + 1)
        ):
            raise ContractValidationError("Attempt index 不连续")
        if attempt_records[-1]["record_id"] != terminal_record_id:
            raise ContractValidationError("成功组合终态不是最后一条 Attempt")
        terminal_payload = decoded_payloads.get(terminal_record_id)
        if isinstance(terminal_payload, Mapping):
            # 结果包也会被独立重放；不能假设 Attempt 写入时已经通过 Gate，避免
            # 数据库被旧版本或人工修改后仍从终态直接发布 evidence_ready。
            self._validate_minichem_attempt_program_gate(terminal_payload)
        return {
            "event_rows": event_rows,
            "decoded_payloads": decoded_payloads,
            "planning_record": planning_record,
            "attempt_records": attempt_records,
        }

    def _bounded_result_revision_records(
        self,
        conn,
        task_id: str,
        combination_id: str,
        decoded_payloads: Mapping[str, Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        """读取当前组合的模型修订和人工工作流补丁。"""

        revision_rows = conn.execute(
            "SELECT * FROM revisions WHERE task_id=? "
            "ORDER BY created_at, revision_id",
            (task_id,),
        ).fetchall()
        revision_records = []
        for row in revision_rows:
            payload = self._decode_json(
                row["revised_result_json"],
                field="revision.revised_result_json",
            )
            if payload.get("combination_id") != combination_id:
                continue
            refs = validate_source_ref_shape(
                self._decode_json(
                    row["source_refs_json"],
                    field="revision.source_refs_json",
                )
            )
            source_attempt_ids = []
            source_pause_ids = []
            for reference in refs:
                if "/" in reference:
                    continue
                source = conn.execute(
                    "SELECT event_type, payload_json FROM memory_events "
                    "WHERE task_id=? AND event_id=?",
                    (task_id, reference),
                ).fetchone()
                if source is None:
                    continue
                if source["event_type"] == "attempt":
                    source_attempt_ids.append(reference)
                elif source["event_type"] == "execution_paused":
                    source_pause_ids.append(reference)
            if len(source_attempt_ids) == 1:
                source_attempt_record_id = source_attempt_ids[0]
            else:
                source_attempt_record_id = None
            record = {
                "record_id": row["revision_id"],
                "created_at": row["created_at"],
                "revision_type": row["revision_type"],
                "payload": deepcopy(payload),
                "source_attempt_record_id": source_attempt_record_id,
                "applied_attempt_index": row["round"],
                "source_pause_record_id": None,
                "source_pause_reason": None,
            }
            if row["revision_type"] == "model_workflow_revision":
                if len(source_attempt_ids) != 1 or source_pause_ids:
                    raise ContractValidationError("模型修订没有唯一来源 Attempt")
            elif row["revision_type"] == "human_workflow_patch":
                if len(source_pause_ids) != 1 or source_attempt_ids:
                    raise ContractValidationError("人工修订没有唯一来源 Pause")
                pause_payload = decoded_payloads.get(source_pause_ids[0])
                if not isinstance(pause_payload, Mapping):
                    raise ContractValidationError("人工修订来源 Pause 不可读取")
                record["source_pause_record_id"] = source_pause_ids[0]
                record["source_pause_reason"] = pause_payload.get("reason_code")
            else:
                raise ContractValidationError("最终结果遇到未知 revision_type")
            revision_records.append(record)
        return revision_records

    def _bounded_result_pause_restart_records(
        self,
        conn,
        task_id: str,
        combination_id: str,
        event_rows: Sequence[Any],
        decoded_payloads: Mapping[str, Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        """读取成功组合中每条已由唯一 Restart 解决的 Pause。"""

        pause_rows = [
            row
            for row in event_rows
            if row["event_type"] == "execution_paused"
            and decoded_payloads[row["event_id"]].get("combination_id")
            == combination_id
        ]
        restart_rows = [
            row for row in event_rows if row["event_type"] == "restart_applied"
        ]
        restart_source_refs = {
            row["event_id"]: validate_source_ref_shape(
                self._decode_json(
                    row["source_refs_json"],
                    field="Restart.source_refs_json",
                )
            )
            for row in restart_rows
        }
        pause_restart_records = []
        for pause in pause_rows:
            matching_restarts = [
                restart
                for restart in restart_rows
                if pause["event_id"] in restart_source_refs[restart["event_id"]]
            ]
            if len(matching_restarts) != 1:
                raise ContractValidationError(
                    "成功组合的每条 Pause 都必须由唯一 Restart 解决"
                )
            pause_record = self._regeneration_event_record(
                conn,
                task_id,
                pause,
            )
            # 最终结果只需要错误与专家诊断的有序证据，不展开清单中的其他内部文件。
            pause_record["source_evidence"] = self._pause_source_evidence(
                conn,
                task_id,
                {
                    "payload": decoded_payloads[pause["event_id"]],
                    "source_refs": pause_record["source_refs"],
                },
            )
            pause_restart_records.append(
                {
                    "pause": pause_record,
                    "restart": self._regeneration_event_record(
                        conn,
                        task_id,
                        matching_restarts[0],
                    ),
                }
            )
        pause_restart_records.sort(
            key=lambda item: (
                item["pause"]["created_at"],
                item["pause"]["record_id"],
            )
        )
        return pause_restart_records

    def _bounded_result_batch_outcomes(
        self,
        conn,
        task_id: str,
        batch_id: str,
    ) -> List[Dict[str, Any]]:
        """读取当前批次全部组合的可归约终局。"""

        batch_combination_rows = conn.execute(
            "SELECT * FROM combinations WHERE task_id=? AND batch_id=? "
            "ORDER BY combination_index",
            (task_id, batch_id),
        ).fetchall()
        batch_combination_outcomes = []
        for batch_combination in batch_combination_rows:
            state = batch_combination["state"]
            gap = None
            if state == "paused":
                pause = self._unresolved_combination_pause(
                    conn,
                    task_id,
                    batch_combination["combination_id"],
                )
                if pause is None:
                    raise ContractValidationError(
                        "技术暂停组合缺少可追溯的 Pause"
                    )
                batch_combination_outcomes.append(
                    {
                        "combination_index": batch_combination[
                            "combination_index"
                        ],
                        "terminal_state": "technical_pause",
                        "terminal_reason": "technical_pause",
                        "tool_capability_gap": None,
                        "technical_failure": {
                            "status": "technical_pause",
                            "failure_kind": pause["reason_code"],
                            "operation": pause["operation"],
                            "pause_id": pause["pause_id"],
                        },
                    }
                )
                continue
            if state == "tool_capability_gap":
                gap_event = conn.execute(
                    "SELECT event_type, payload_json FROM memory_events "
                    "WHERE task_id=? AND event_id=?",
                    (task_id, batch_combination["terminal_record_id"]),
                ).fetchone()
                if (
                    gap_event is None
                    or gap_event["event_type"] != "tool_capability_gap"
                ):
                    raise ContractValidationError(
                        "能力缺口组合缺少对应 ToolCapabilityGap 事实"
                    )
                gap_payload = validate_tool_capability_gap_payload(
                    self._decode_json(
                        gap_event["payload_json"],
                        field="ToolCapabilityGap.payload_json",
                    )
                )
                if (
                    gap_payload["combination_id"]
                    != batch_combination["combination_id"]
                    or gap_payload["hypothesis_id"]
                    != batch_combination["hypothesis_id"]
                ):
                    raise ContractValidationError(
                        "能力缺口事实与批次组合身份不一致"
                    )
                gap = deepcopy(gap_payload["gap"])
            elif state not in {"evidence_ready", "hypothesis_failed"}:
                raise ContractValidationError(
                    "发布 bounded result 前批次仍有未归约组合"
                )
            batch_combination_outcomes.append(
                {
                    "combination_index": batch_combination["combination_index"],
                    "terminal_state": state,
                    "terminal_reason": batch_combination["terminal_reason"],
                    "tool_capability_gap": gap,
                }
            )
        return batch_combination_outcomes

    def build_bounded_result_package(
        self,
        task_id: str,
        batch_id: str,
        combination_id: str,
    ) -> Dict[str, Any]:
        """联接一个成功组合的完整事实，供最终 JSON 渲染器只读使用。

        结果发布发生在 BatchClosed 事务之前，因此这里要求批次仍在运行、组合已经
        evidence_ready。所有正文都从正式事件和修订读取；目录中同名结果文件、孤立
        模型清单或旧阶段 JSON 不参与选择。
        """

        store = self._require_enabled()
        for value, field in (
            (task_id, "task_id"),
            (batch_id, "batch_id"),
            (combination_id, "combination_id"),
        ):
            require_nonempty_string(value, field)
        with store.transaction() as conn:
            subject = self._bounded_result_subject(
                conn,
                task_id,
                batch_id,
                combination_id,
            )
            task_created = subject["task_created"]
            batch = subject["batch"]
            combination = subject["combination"]
            retrieval_phase = subject["retrieval_phase"]
            hypothesis_phase = subject["hypothesis_phase"]

            attempt_history = self._bounded_result_attempt_history(
                conn,
                task_id,
                combination_id,
                combination["terminal_record_id"],
            )
            event_rows = attempt_history["event_rows"]
            decoded_payloads = attempt_history["decoded_payloads"]
            planning_record = attempt_history["planning_record"]
            attempt_records = attempt_history["attempt_records"]

            revision_records = self._bounded_result_revision_records(
                conn,
                task_id,
                combination_id,
                decoded_payloads,
            )

            pause_restart_records = self._bounded_result_pause_restart_records(
                conn,
                task_id,
                combination_id,
                event_rows,
                decoded_payloads,
            )
            batch_combination_outcomes = self._bounded_result_batch_outcomes(
                conn,
                task_id,
                batch_id,
            )
            effective_context = self._effective_scientific_context(conn, task_id)
            return {
                "scientific_question": task_created["scientific_question"],
                "workflow_start_time": subject["workflow_start_time"],
                "task_id": task_id,
                "task_config": deepcopy(task_created["task_config"]),
                "chemistry_context": deepcopy(effective_context["chemistry_context"]),
                "retrieval_phase": deepcopy(dict(retrieval_phase)),
                "hypothesis_phase": deepcopy(dict(hypothesis_phase)),
                "hypothesis": subject["hypothesis"],
                "planning_record": planning_record,
                "attempt_records": attempt_records,
                "revision_records": revision_records,
                "pause_restart_records": pause_restart_records,
                "terminal_record_id": combination["terminal_record_id"],
                "batch_combination_outcomes": batch_combination_outcomes,
            }

    @staticmethod
    def _regeneration_hypothesis_projection(
        hypothesis: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """只把假设本身的科学内容交给下一批生成模型。

        排名结果中的完整对象还包含程序生成的身份和运行状态。这些字段留在原始
        事实中供恢复核验，但下一批只看到名称和完整 reasoning；执行事实仍由同一
        历史项的其他字段完整提供。
        """

        if not isinstance(hypothesis, Mapping):
            raise ContractValidationError("历史 hypothesis 必须是 object")
        compatibility = validate_structured_hypothesis_claims(
            hypothesis,
            path="历史 hypothesis",
        )
        return {
            "strategy_name": deepcopy(hypothesis["strategy_name"]),
            "reasoning": compatibility["reasoning"],
        }

    def _regeneration_package_context(
        self,
        conn,
        task_id: str,
        next_batch_index: int,
    ) -> Dict[str, Any]:
        """读取连续失败批次、首批冻结上下文和下一批生成配置。"""

        task = conn.execute(
            "SELECT * FROM tasks WHERE task_id=?",
            (task_id,),
        ).fetchone()
        if task is None:
            raise ContractValidationError("未知 task_id")
        if task["status"] != "running":
            raise ContractValidationError("只有 running task 可以生成下一批")
        task_created = self._task_created_payload(conn, task_id)
        task_config = task_created["task_config"]
        if next_batch_index > task_config["max_hypothesis_batches"]:
            raise ContractValidationError("next_batch_index 超过任务批次预算")
        existing_next = conn.execute(
            "SELECT 1 FROM hypothesis_batches WHERE task_id=? AND batch_index=?",
            (task_id, next_batch_index),
        ).fetchone()
        if existing_next is not None:
            raise ContractValidationError("下一批已经创建，不应重复生成历史包")

        batches = conn.execute(
            "SELECT * FROM hypothesis_batches WHERE task_id=? "
            "AND batch_index<? ORDER BY batch_index",
            (task_id, next_batch_index),
        ).fetchall()
        expected_indexes = list(range(1, next_batch_index))
        if [row["batch_index"] for row in batches] != expected_indexes:
            raise ContractValidationError("历史批次不连续")
        if any(row["state"] != "failed" for row in batches):
            raise ContractValidationError("下一批只能在全部历史批次无成功结果后生成")

        created_events = {}
        for batch in batches:
            row = conn.execute(
                "SELECT * FROM memory_events WHERE task_id=? AND event_id=? "
                "AND event_type='hypothesis_batch_created'",
                (task_id, batch["created_event_id"]),
            ).fetchone()
            if row is None:
                raise ContractValidationError("历史批次缺少创建事实")
            payload = self._validated_batch_created_payload(
                conn,
                task_id,
                row["payload_json"],
                field="HypothesisBatchCreated.payload_json",
            )
            created_events[batch["batch_id"]] = (row, payload)

        first_snapshot = created_events[batches[0]["batch_id"]][1].get(
            "question_snapshot"
        )
        if not isinstance(first_snapshot, Mapping):
            raise ContractValidationError("首批缺少冻结科学上下文")
        query_angles = first_snapshot.get("initial_query_angles")
        initial_retrieval = first_snapshot.get("initial_retrieval_phase")
        if not isinstance(query_angles, list) or not isinstance(
            initial_retrieval, Mapping
        ):
            raise ContractValidationError("首批 query 或 Retrieval 快照损坏")
        effective = self._effective_scientific_context(conn, task_id)
        scientific_context = project_hypothesis_regeneration_context(
            scientific_question=task_created["scientific_question"],
            initial_query_angles=deepcopy(query_angles),
            initial_retrieval_phase=deepcopy(dict(initial_retrieval)),
            chemistry_context=deepcopy(effective["chemistry_context"]),
        )
        return {
            "batches": batches,
            "created_events": created_events,
            "scientific_context": scientific_context,
            "task_config": task_config,
        }

    def _regeneration_history_sources(
        self,
        conn,
        task_id: str,
    ) -> Dict[str, Any]:
        """读取并解码下一批需要联接的事件、修订和 TaskCreated 身份。"""

        planning_rows = conn.execute(
            "SELECT * FROM memory_events WHERE task_id=? "
            "AND event_type='planning_completed' ORDER BY created_at, event_id",
            (task_id,),
        ).fetchall()
        attempt_rows = conn.execute(
            "SELECT * FROM memory_events WHERE task_id=? "
            "AND event_type='attempt' ORDER BY created_at, event_id",
            (task_id,),
        ).fetchall()
        pause_rows = conn.execute(
            "SELECT * FROM memory_events WHERE task_id=? "
            "AND event_type='execution_paused' ORDER BY created_at, event_id",
            (task_id,),
        ).fetchall()
        restart_rows = conn.execute(
            "SELECT * FROM memory_events WHERE task_id=? "
            "AND event_type='restart_applied' ORDER BY created_at, event_id",
            (task_id,),
        ).fetchall()
        revision_rows = conn.execute(
            "SELECT * FROM revisions WHERE task_id=? ORDER BY created_at, revision_id",
            (task_id,),
        ).fetchall()

        decoded_events: Dict[str, Dict[str, Any]] = {}
        event_groups = [planning_rows, attempt_rows, pause_rows, restart_rows]
        for rows in event_groups:
            for row in rows:
                decoded_events[row["event_id"]] = self._decode_json(
                    row["payload_json"],
                    field=f"{row['event_type']}.payload_json",
                )
        decoded_revisions = {
            row["revision_id"]: self._decode_json(
                row["revised_result_json"],
                field="revision.revised_result_json",
            )
            for row in revision_rows
        }
        task_created_row = conn.execute(
            "SELECT event_id FROM memory_events WHERE task_id=? "
            "AND event_key=? AND event_type='task_created'",
            (task_id, event_key_task_created(task_id)),
        ).fetchone()
        if task_created_row is None:
            raise ContractValidationError("任务缺少 TaskCreated 事实")
        return {
            "planning_rows": planning_rows,
            "attempt_rows": attempt_rows,
            "pause_rows": pause_rows,
            "restart_rows": restart_rows,
            "revision_rows": revision_rows,
            "decoded_events": decoded_events,
            "decoded_revisions": decoded_revisions,
            "task_created_record_id": task_created_row["event_id"],
        }

    def _regeneration_capability_gap(
        self,
        conn,
        task_id: str,
        candidate: Mapping[str, Any],
        combination: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """核对历史能力缺口，并区分首次规划缺口与修订阶段缺口。"""

        if combination["state"] != "tool_capability_gap":
            return {
                "gap": None,
                "initial": False,
                "source_attempt_record_id": None,
            }
        gap_row = conn.execute(
            "SELECT * FROM memory_events WHERE task_id=? "
            "AND event_id=? AND event_type='tool_capability_gap'",
            (task_id, combination["terminal_record_id"]),
        ).fetchone()
        if gap_row is None:
            raise ContractValidationError(
                "历史能力缺口缺少对应 ToolCapabilityGap 事实"
            )
        gap_payload = validate_tool_capability_gap_payload(
            self._decode_json(
                gap_row["payload_json"],
                field="ToolCapabilityGap.payload_json",
            )
        )
        if (
            gap_payload["combination_id"] != combination["combination_id"]
            or gap_payload["hypothesis_id"] != candidate["hypothesis_id"]
            or combination["terminal_reason"] != "tool_capability_gap"
        ):
            raise ContractValidationError(
                "历史能力缺口事实与组合终态不一致"
            )
        gap_refs = validate_source_ref_shape(
            self._decode_json(
                gap_row["source_refs_json"],
                field="ToolCapabilityGap.source_refs_json",
            )
        )
        gap_path_refs = [item for item in gap_refs if "/" in item]
        gap_record_refs = [item for item in gap_refs if "/" not in item]
        initial_gap = len(gap_refs) == 1 and len(gap_path_refs) == 1
        revision_gap = (
            len(gap_refs) == 3
            and len(gap_path_refs) == 2
            and len(gap_record_refs) == 1
        )
        if not initial_gap and not revision_gap:
            raise ContractValidationError(
                "历史能力缺口使用了无法识别的来源证据组合"
            )
        capability_gap = deepcopy(gap_payload["gap"])
        if initial_gap:
            if (
                combination["workflow_version"] != 0
                or combination["attempt_index"] != 0
                or combination["workflow_id"] is not None
                or combination["execution_id"] is not None
            ):
                raise ContractValidationError(
                    "历史 Planner 能力缺口混入了执行身份"
                )
            return {
                "gap": capability_gap,
                "initial": True,
                "source_attempt_record_id": None,
            }
        if (
            combination["workflow_version"] < 1
            or combination["attempt_index"] < 1
            or not isinstance(combination["workflow_id"], str)
            or not isinstance(combination["execution_id"], str)
        ):
            raise ContractValidationError("历史修订能力缺口缺少已执行工作流")
        return {
            "gap": capability_gap,
            "initial": False,
            "source_attempt_record_id": gap_record_refs[0],
        }

    def _regeneration_pause_restart_records(
        self,
        conn,
        task_id: str,
        combination_id: str,
        pause_rows: Sequence[Any],
        restart_rows: Sequence[Any],
        decoded_events: Mapping[str, Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        """读取历史组合中已经由唯一 Restart 解决的 Pause。"""

        matching_pauses = [
            row
            for row in pause_rows
            if decoded_events[row["event_id"]].get("combination_id")
            == combination_id
        ]
        pause_restart_records = []
        for pause in matching_pauses:
            matching_restarts = []
            for restart in restart_rows:
                refs = self._decode_json(
                    restart["source_refs_json"],
                    field="Restart.source_refs_json",
                )
                if pause["event_id"] in refs:
                    matching_restarts.append(restart)
            if len(matching_restarts) != 1:
                raise ContractValidationError(
                    "终态 combination 的 Pause 必须由唯一 Restart 解决"
                )
            pause_restart_records.append(
                {
                    "pause": self._regeneration_event_record(
                        conn,
                        task_id,
                        pause,
                    ),
                    "restart": self._regeneration_event_record(
                        conn,
                        task_id,
                        matching_restarts[0],
                    ),
                }
            )
        pause_restart_records.sort(
            key=lambda pair: (
                pair["pause"]["created_at"],
                pair["pause"]["record_id"],
            )
        )
        return pause_restart_records

    def _regeneration_combination_history(
        self,
        conn,
        task_id: str,
        batch: Mapping[str, Any],
        candidate: Mapping[str, Any],
        combination: Mapping[str, Any],
        ranked: Sequence[Any],
        sources: Mapping[str, Any],
    ) -> Tuple[Dict[str, Any], set[str]]:
        """把一个已结束组合压缩为下一批可读的终局科学经验。

        Working Memory 仍会读取完整事件来核对因果关系，但返回值只保留 Query、
        假设正文、终局分类、裁判理由、科学账本摘要和工具能力缺口；正式身份、
        workflow、暂停恢复和 artifact 内容只用于内部来源集合。
        """

        if (
            combination["state"] not in BATCH_CLOSABLE_COMBINATION_STATES
            or combination["state"] == "evidence_ready"
            or not combination["terminal_reason"]
            or not combination["terminal_record_id"]
        ):
            raise ContractValidationError("历史 combination 尚未无成功终结")
        hypotheses = [
            deepcopy(dict(item))
            for item in ranked
            if isinstance(item, Mapping)
            and item.get("hypothesis_id") == candidate["hypothesis_id"]
        ]
        if len(hypotheses) != 1:
            raise ContractValidationError("历史 combination 没有唯一 hypothesis")
        hypothesis = self._regeneration_hypothesis_projection(hypotheses[0])
        phase_groups = ranked
        source_query = ""
        batch_payload = sources.get("batch_payload")
        if isinstance(batch_payload, Mapping):
            phase = batch_payload.get("hypothesis_phase")
            phase_groups = (
                phase.get("hypotheses_by_query", [])
                if isinstance(phase, Mapping)
                else []
            )
        if isinstance(phase_groups, list):
            for group in phase_groups:
                if not isinstance(group, Mapping):
                    continue
                group_hypotheses = group.get("hypotheses")
                if not isinstance(group_hypotheses, list):
                    continue
                if any(
                    isinstance(item, Mapping)
                    and item.get("hypothesis_id") == candidate["hypothesis_id"]
                    for item in group_hypotheses
                ):
                    source_query = group.get("query", "")
                    break
        combination_id = combination["combination_id"]
        gap_details = self._regeneration_capability_gap(
            conn,
            task_id,
            candidate,
            combination,
        )
        capability_gap = gap_details["gap"]
        if gap_details["initial"]:
            # 首次规划阶段尚未形成工作流，因此只记录工具能力缺口。
            outcome = project_hypothesis_terminal_outcome(
                source_query=source_query,
                hypothesis=hypothesis,
                terminal_classification="tool_capability_gap",
                terminal_reason=combination["terminal_reason"],
                tool_capability_gap=capability_gap,
            )
            return (
                outcome,
                {combination["terminal_record_id"]},
            )

        planning_rows = sources["planning_rows"]
        attempt_rows = sources["attempt_rows"]
        pause_rows = sources["pause_rows"]
        restart_rows = sources["restart_rows"]
        revision_rows = sources["revision_rows"]
        decoded_events = sources["decoded_events"]
        decoded_revisions = sources["decoded_revisions"]

        matching_planning = [
            row
            for row in planning_rows
            if decoded_events[row["event_id"]].get("combination_id")
            == combination_id
        ]
        if len(matching_planning) != 1:
            raise ContractValidationError("历史 combination 必须恰有一次规划")
        planning_records = [
            self._regeneration_event_record(
                conn,
                task_id,
                matching_planning[0],
            )
        ]
        matching_revisions = [
            row
            for row in revision_rows
            if decoded_revisions[row["revision_id"]].get("combination_id")
            == combination_id
        ]
        revision_records = [
            self._regeneration_revision_record(conn, task_id, row)
            for row in matching_revisions
        ]
        matching_attempts = [
            row
            for row in attempt_rows
            if decoded_events[row["event_id"]].get("combination_id")
            == combination_id
        ]
        if not matching_attempts:
            raise ContractValidationError("历史 combination 缺少 Attempt")
        gap_source_attempt_id = gap_details["source_attempt_record_id"]
        if (
            gap_source_attempt_id is not None
            and gap_source_attempt_id
            not in {row["event_id"] for row in matching_attempts}
        ):
            raise ContractValidationError(
                "历史修订能力缺口没有引用当前组合的 Attempt"
            )
        attempt_records = [
            self._regeneration_event_record(conn, task_id, row)
            for row in matching_attempts
        ]
        pause_restart_records = self._regeneration_pause_restart_records(
            conn,
            task_id,
            combination_id,
            pause_rows,
            restart_rows,
            decoded_events,
        )
        terminal_payload = self._decode_json(
            matching_attempts[-1]["payload_json"],
            field="regeneration terminal Attempt.payload_json",
        )
        planning_payload = self._decode_json(
            matching_planning[0]["payload_json"],
            field="regeneration planning.payload_json",
        )
        decision = terminal_payload.get("router_decision")
        terminal_classification = {
            "hypothesis_failed": "reliably_refuted",
            "evidence_ready": "supported",
            "tool_capability_gap": "tool_capability_gap",
        }.get(combination["state"], "insufficient_evidence")
        if combination["state"] == "hypothesis_failed" and isinstance(decision, Mapping):
            if decision.get("decision") != "revise_hypothesis":
                terminal_classification = "insufficient_evidence"
        record = project_hypothesis_terminal_outcome(
            source_query=source_query,
            hypothesis=hypothesis,
            terminal_classification=terminal_classification,
            terminal_reason=combination["terminal_reason"],
            execution_result=terminal_payload.get("execution_result"),
            program_validation=terminal_payload.get("program_validation"),
            router_decision=decision,
            tool_capability_gap=capability_gap,
            workflow=planning_payload.get("workflow"),
        )
        source_record_ids = {planning_records[0]["record_id"]}
        source_record_ids.update(
            item["record_id"] for item in revision_records
        )
        source_record_ids.update(item["record_id"] for item in attempt_records)
        for pair in pause_restart_records:
            source_record_ids.add(pair["pause"]["record_id"])
            source_record_ids.add(pair["restart"]["record_id"])
        source_record_ids.add(combination["terminal_record_id"])
        return record, source_record_ids

    def _assemble_hypothesis_regeneration_package_in_transaction(
        self,
        conn,
        task_id: str,
        next_batch_index: int,
    ) -> Dict[str, Any]:
        """在已打开的读事务中联接全部历史事实并形成假设重生材料。

        输入是事务连接、任务身份和已经通过入口校验的下一批编号；动作只读取
        TaskCreated、BatchCreated、候选、组合和终态事件，不写快照、不扫描任务目录，
        并把每个已选组合的真实科学结果压缩为 Hypothesis 阶段可读的经验。批次、
        终态或 Pause/Restart 因果链不完整时抛出合同错误，防止模型在缺失事实下重试。
        """

        context = self._regeneration_package_context(
            conn,
            task_id,
            next_batch_index,
        )
        batches = context["batches"]
        created_events = context["created_events"]
        scientific_context = context["scientific_context"]
        task_config = context["task_config"]

        sources = self._regeneration_history_sources(conn, task_id)
        history = []
        source_record_ids = {sources["task_created_record_id"]}

        for batch in batches:
            created_row, batch_payload = created_events[batch["batch_id"]]
            source_record_ids.add(created_row["event_id"])
            # 只供当前事务内部联接 Query；该批次正文不会作为模型输入原样复制。
            sources["batch_payload"] = batch_payload
            phase = batch_payload.get("hypothesis_phase")
            ranked = phase.get("ranked_strategies") if isinstance(phase, Mapping) else None
            if not isinstance(ranked, list):
                raise ContractValidationError("历史批次缺少 ranked_strategies")
            candidate_rows = conn.execute(
                "SELECT * FROM hypothesis_candidates WHERE batch_id=? "
                "ORDER BY candidate_index",
                (batch["batch_id"],),
            ).fetchall()
            selected = [
                row for row in candidate_rows
                if row["selection_status"] == "selected"
            ]
            combinations = {
                row["combination_id"]: row
                for row in conn.execute(
                    "SELECT * FROM combinations WHERE task_id=? AND batch_id=?",
                    (task_id, batch["batch_id"]),
                ).fetchall()
            }
            for candidate in selected:
                combination = combinations.get(candidate["combination_id"])
                if combination is None:
                    raise ContractValidationError("selected candidate 缺少 combination")
                record, record_source_ids = self._regeneration_combination_history(
                    conn,
                    task_id,
                    batch,
                    candidate,
                    combination,
                    ranked,
                    sources,
                )
                history.append(record)
                source_record_ids.update(record_source_ids)

            closed = conn.execute(
                "SELECT event_id FROM memory_events WHERE task_id=? "
                "AND event_type='hypothesis_batch_closed' AND round=?",
                (task_id, batch["batch_index"]),
            ).fetchall()
            if len(closed) != 1:
                raise ContractValidationError("历史批次缺少唯一 BatchClosed 事实")
            source_record_ids.add(closed[0]["event_id"])

        return {
            "scientific_context": scientific_context,
            # 当前 Query 在 Hypothesis Agent 的每次子调用前填入；None 只作为
            # package 骨架占位，正式请求一定使用当前角度的完整文本。
            "current_query": None,
            "prior_hypothesis_outcomes": history,
            "next_batch_index": next_batch_index,
            "generation_config": {
                "num_queries": task_config["num_queries"],
                "num_hypotheses_per_query": task_config[
                    "num_hypotheses_per_query"
                ],
                "top_n": task_config.get(
                    "top_n_workflows", task_config.get("top_n_hypotheses")
                ),
            },
            # source_record_ids 只给 Controller 写入 generation_source；Agent 在
            # 编译模型请求前会移除该字段，避免正式身份进入阶段记忆视图。
            "source_record_ids": sorted(source_record_ids),
        }

    def build_hypothesis_regeneration_package(
        self,
        task_id: str,
        *,
        next_batch_index: int,
    ) -> Dict[str, Any]:
        """从全部已关闭失败批次构造下一批假设的只读历史。

        该方法只在一个 SQLite 读事务中联接事实与当前投影，不写快照表，也不扫描
        任务目录寻找“最新文件”。如果任一历史批次、终态或 Pause/Restart 因果链
        不完整，就拒绝生成下一批，避免模型在缺失前提下重复已经失败的方向。
        """

        store = self._require_enabled()
        require_nonempty_string(task_id, "task_id")
        if (
            isinstance(next_batch_index, bool)
            or not isinstance(next_batch_index, int)
            or next_batch_index < 2
        ):
            raise ContractValidationError("next_batch_index 必须至少为 2")
        with store.transaction() as conn:
            return self._assemble_hypothesis_regeneration_package_in_transaction(
                conn,
                task_id,
                next_batch_index,
            )

    @staticmethod
    def _attempt_index(conn, task_id: str, combination_id: str) -> int:
        """读取组合当前的科学尝试序号；未知组合按尚未开始处理。"""

        row = conn.execute(
            "SELECT attempt_index FROM combinations WHERE task_id=? AND combination_id=?",
            (task_id, combination_id),
        ).fetchone()
        return 0 if row is None else row["attempt_index"]

    # ==================== 工作流修订材料 ====================

    def _workflow_revision_subject(
        self,
        conn,
        task_id: str,
        combination_id: str,
    ) -> Dict[str, Any]:
        """读取待修订组合、当前工作流和任务内批次/组合索引。"""

        current = conn.execute(
            "SELECT * FROM combinations WHERE task_id=? AND combination_id=?",
            (task_id, combination_id),
        ).fetchone()
        if current is None:
            raise ContractValidationError("revision package 找不到组合")
        runtime = self._decode_json(current["runtime_checkpoint_json"], field="runtime_checkpoint_json")
        direct_failure = runtime.get("execution_failure") if isinstance(runtime, Mapping) else None
        if current["state"] != "revising_workflow" and not isinstance(direct_failure, Mapping):
            raise ContractValidationError(
                "revision package 只接受 revising_workflow 或 execution failure 组合"
            )
        current_workflow = self._decode_json(
            current["current_workflow_json"],
            field="current_workflow_json",
        )
        scientific_context = self._effective_scientific_context(conn, task_id)
        combination_rows = {
            row["combination_id"]: row
            for row in conn.execute(
                "SELECT * FROM combinations WHERE task_id=?",
                (task_id,),
            ).fetchall()
        }
        batch_rows = {
            row["batch_id"]: row
            for row in conn.execute(
                "SELECT * FROM hypothesis_batches WHERE task_id=?",
                (task_id,),
            ).fetchall()
        }
        return {
            "current": current,
            "current_workflow": current_workflow,
            "scientific_context": scientific_context,
            "combination_rows": combination_rows,
            "batch_rows": batch_rows,
        }

    def _workflow_revision_hypothesis(
        self,
        conn,
        task_id: str,
        combination: Mapping[str, Any],
        batch_rows: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """沿批次创建事件读取组合当时冻结的完整假设。"""

        batch = batch_rows.get(combination["batch_id"])
        if batch is None:
            raise ContractValidationError("revision package 缺少 batch projection")
        event = conn.execute(
            "SELECT payload_json FROM memory_events "
            "WHERE task_id=? AND event_id=? AND event_type='hypothesis_batch_created'",
            (task_id, batch["created_event_id"]),
        ).fetchone()
        if event is None:
            raise ContractValidationError("revision package 缺少 batch 创建事实")
        payload = self._validated_batch_created_payload(
            conn,
            task_id,
            event["payload_json"],
            field="HypothesisBatchCreated.payload_json",
        )
        phase = payload.get("hypothesis_phase")
        if isinstance(phase, Mapping):
            ranked = phase.get("ranked_strategies")
        else:
            ranked = None
        matches = [
            item
            for item in ranked or []
            if isinstance(item, Mapping)
            and item.get("hypothesis_id") == combination["hypothesis_id"]
        ]
        if len(matches) != 1:
            raise ContractValidationError(
                "revision package 没有唯一冻结 hypothesis"
            )
        matched = matches[0]
        compatibility = validate_structured_hypothesis_claims(
            matched,
            path="revision package hypothesis",
        )
        return {
            "strategy_name": deepcopy(matched["strategy_name"]),
            "reasoning": deepcopy(compatibility["reasoning"]),
        }

    def _workflow_revision_sources(
        self,
        conn,
        task_id: str,
        *,
        allow_direct_execution_failure: bool = False,
    ) -> Dict[str, Any]:
        """读取正式工作流版本及修订来源，并只为已核对的直接执行失败允许空 Attempt。

        输入是当前任务的数据库连接和直接执行失败分支的显式许可。方法读取首版
        工作流、历史修订和 Router Attempt；正常 revising_workflow 仍必须至少有一条
        revise_workflow Attempt，只有调用方已经核对失败事件与组合身份时才能传入
        直接失败许可。数据库事实缺失、重复或 JSON 损坏时抛出契约错误，不跳过坏记录。
        """

        workflow_versions: Dict[Tuple[str, int], Dict[str, Any]] = {}
        planning_rows = conn.execute(
            "SELECT payload_json FROM memory_events "
            "WHERE task_id=? AND event_type='planning_completed'",
            (task_id,),
        ).fetchall()
        for row in planning_rows:
            payload = self._decode_json(
                row["payload_json"],
                field="PlanningCompleted.payload_json",
            )
            key = (payload["combination_id"], payload["workflow_version"])
            if key in workflow_versions:
                raise ContractValidationError("首版 workflow 来源不唯一")
            workflow_versions[key] = deepcopy(payload["workflow"])

        revision_rows = conn.execute(
            "SELECT * FROM revisions WHERE task_id=? "
            "ORDER BY created_at, revision_id",
            (task_id,),
        ).fetchall()
        decoded_revisions = []
        for row in revision_rows:
            payload = self._decode_json(
                row["revised_result_json"],
                field="revision.revised_result_json",
            )
            key = (
                payload.get("combination_id"),
                payload.get("workflow_version_after"),
            )
            if key in workflow_versions:
                raise ContractValidationError("workflow version 来源不唯一")
            workflow_versions[key] = deepcopy(payload["after_workflow"])
            decoded_revisions.append((row, payload))

        attempt_rows = conn.execute(
            "SELECT * FROM memory_events WHERE task_id=? AND event_type='attempt' "
            "ORDER BY created_at, event_id",
            (task_id,),
        ).fetchall()
        revise_attempts = []
        for row in attempt_rows:
            payload = self._decode_json(
                row["payload_json"],
                field="Attempt.payload_json",
            )
            decision = payload.get("router_decision")
            if (
                isinstance(decision, Mapping)
                and decision.get("decision") == "revise_workflow"
            ):
                revise_attempts.append((row, payload))
        if not revise_attempts and not allow_direct_execution_failure:
            raise ContractValidationError(
                "revision package 前没有 revise_workflow Attempt"
            )
        return {
            "workflow_versions": workflow_versions,
            "decoded_revisions": decoded_revisions,
            "revise_attempts": revise_attempts,
        }

    def _validate_direct_execution_failure_source(
        self,
        conn,
        task_id: str,
        combination: Mapping[str, Any],
        runtime: Mapping[str, Any],
    ) -> None:
        """核对直接执行失败的运行投影与不可变失败事件完全一致。

        输入是当前组合行和其 runtime checkpoint。方法查询 ``event_id`` 指向的
        ``execution_step_failed`` 事实，校验执行身份、步骤清单、失败诊断和已清理
        的执行 claim；不写数据库，任一字段缺失、错配或引用文件失效都会抛出契约
        错误，避免把损坏的失败投影当成合法修订来源。
        """

        direct_failure = runtime.get("execution_failure")
        if not isinstance(direct_failure, Mapping):
            raise ContractValidationError("直接执行失败缺少 execution_failure 投影")
        allowed_failure_fields = {
            "step_id",
            "step_index",
            "failed_step_manifest_ref",
            "event_id",
            "diagnostic",
        }
        if set(direct_failure) - allowed_failure_fields:
            raise ContractValidationError("execution_failure 投影包含未知字段")
        event_id = direct_failure.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            raise ContractValidationError("execution_failure 缺少失败事件身份")
        event = conn.execute(
            "SELECT event_id, payload_json, source_refs_json FROM memory_events "
            "WHERE task_id=? AND event_type='execution_step_failed' "
            "AND (event_id=? OR event_key=?)",
            (task_id, event_id, event_id),
        ).fetchone()
        if event is None:
            raise ContractValidationError(
                "execution_failure.event_id 找不到 execution_step_failed 事实"
            )
        payload = validate_execution_step_failed_payload(
            self._decode_json(
                event["payload_json"],
                field="ExecutionStepFailed.payload_json",
            )
        )
        for field in ("combination_id", "workflow_version", "execution_id"):
            if payload[field] != combination[field]:
                raise ContractValidationError(
                    "execution_failure 事件与当前 workflow 身份不一致"
                )
        for field in ("step_id", "step_index", "failed_step_manifest_ref"):
            if direct_failure.get(field) != payload[field]:
                raise ContractValidationError(
                    f"execution_failure.{field} 与失败事件不一致"
                )
        if direct_failure.get("diagnostic") != payload.get("failure_diagnostic"):
            raise ContractValidationError("execution_failure.diagnostic 与失败事件不一致")
        step_manifest_refs = runtime.get("step_manifest_refs")
        if step_manifest_refs != payload["step_manifest_refs"]:
            raise ContractValidationError(
                "execution_failure 的步骤清单与失败事件不一致"
            )
        if runtime.get("next_step_index") != len(payload["step_manifest_refs"]):
            raise ContractValidationError("execution_failure 的执行游标不在 workflow 末尾")
        for field in (
            "current_operation_instance_id",
            "current_request_ref",
            "checkpoint_id",
            "checkpoint_manifest_ref",
        ):
            if runtime.get(field) is not None:
                raise ContractValidationError(
                    "execution_failure 仍保留活动执行 claim 或 checkpoint"
                )
        source_refs = self._decode_json(
            event["source_refs_json"],
            field="ExecutionStepFailed.source_refs_json",
        )
        if source_refs != sorted(payload["step_manifest_refs"]):
            raise ContractValidationError(
                "execution_step_failed.source_refs 与步骤清单不一致"
            )
        self._validate_source_refs(conn, task_id, source_refs)

    def _workflow_failure_trajectories(
        self,
        conn,
        task_id: str,
        revise_attempts: Sequence[Tuple[Any, Mapping[str, Any]]],
        combination_rows: Mapping[str, Any],
        batch_rows: Mapping[str, Any],
        workflow_versions: Mapping[Tuple[str, int], Mapping[str, Any]],
        decoded_revisions: Sequence[Tuple[Any, Mapping[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """按 Attempt 顺序投影所有工作流失败轨迹及其已应用修订。"""

        trajectories = []
        previous_versions: Dict[str, int] = {}
        for attempt_row, attempt in revise_attempts:
            attempt_combination_id = attempt["combination_id"]
            combination = combination_rows.get(attempt_combination_id)
            if combination is None:
                raise ContractValidationError(
                    "revision trajectory 缺少 combination projection"
                )
            version = attempt["workflow_version"]
            workflow = workflow_versions.get((attempt_combination_id, version))
            if workflow is None:
                raise ContractValidationError(
                    "revision trajectory 缺少 workflow 来源"
                )
            lower_version = previous_versions.get(attempt_combination_id, 0)
            applied_revisions = []
            for revision_row, revision in decoded_revisions:
                if (
                    revision.get("combination_id") != attempt_combination_id
                    or not (
                        lower_version
                        < revision.get("workflow_version_after", -1)
                        <= version
                    )
                ):
                    continue
                applied_revisions.append(
                    {
                        "revision_id": revision_row["revision_id"],
                        "revision_type": revision_row["revision_type"],
                        "workflow_version_before": revision[
                            "workflow_version_before"
                        ],
                        "workflow_version_after": revision[
                            "workflow_version_after"
                        ],
                        "before_workflow": deepcopy(revision["before_workflow"]),
                        "normalized_patch": deepcopy(revision["normalized_patch"]),
                        "after_workflow": deepcopy(revision["after_workflow"]),
                    }
                )
            previous_versions[attempt_combination_id] = version
            batch = batch_rows[combination["batch_id"]]
            trajectories.append(
                {
                    "attempt_record_id": attempt_row["event_id"],
                    "batch_index": batch["batch_index"],
                    "combination_index": combination["combination_index"],
                    "hypothesis": self._workflow_revision_hypothesis(
                        conn,
                        task_id,
                        combination,
                        batch_rows,
                    ),
                    "workflow_version": version,
                    "workflow": project_workflow_refs(workflow)["workflow"],
                    "applied_revisions": applied_revisions,
                    "execution_result": deepcopy(attempt["execution_result"]),
                    "program_validation": deepcopy(
                        attempt["program_validation"]
                    ),
                    "router_decision": deepcopy(attempt["router_decision"]),
                }
            )
        return trajectories

    def _workflow_retry_restarts_by_pause(
        self,
        restart_rows: Sequence[Any],
    ) -> Dict[str, List[Any]]:
        """索引模型 Pause 之后用于重试失败操作的新 Restart。"""

        retry_restarts_by_pause: Dict[str, List[Any]] = {}
        for retry_row in restart_rows:
            retry_payload = self._decode_json(
                retry_row["payload_json"],
                field="retry Restart.payload_json",
            )
            if retry_payload.get("action") != "retry_failed_operation":
                continue
            retry_sources = self._decode_json(
                retry_row["source_refs_json"],
                field="retry Restart.source_refs_json",
            )
            for source_ref in retry_sources:
                retry_restarts_by_pause.setdefault(source_ref, []).append(
                    retry_row
                )
        return retry_restarts_by_pause

    def _closed_human_correction_result(
        self,
        result_rows: Sequence[Any],
        restart_record_id: str,
        retry_restarts_by_pause: Mapping[str, Sequence[Any]],
    ) -> Optional[Tuple[Any, Dict[str, Any]]]:
        """沿人工补丁授权及后续模型重试授权找到唯一真实结果。"""

        closed_results = []
        active_restart_id = restart_record_id
        for result_row in result_rows:
            payload = self._decode_json(
                result_row["payload_json"],
                field="human correction result.payload_json",
            )
            if payload.get("causal_source_ref") != active_restart_id:
                continue
            if result_row["event_type"] == "attempt":
                closed_results.append((result_row, payload))
            elif payload.get("reason_code") == "tool_execution_failure":
                closed_results.append((result_row, payload))
            elif payload.get("reason_code") in {
                "model_protocol_failure",
                "model_service_failure",
            }:
                # 模型暂停本身不是补丁的科学结果；已重试则沿新授权继续追踪。
                retry_matches = retry_restarts_by_pause.get(
                    result_row["event_id"],
                    [],
                )
                if len(retry_matches) > 1:
                    raise ContractValidationError(
                        "同一模型 Pause 产生了多条 retry Restart"
                    )
                if not retry_matches:
                    break
                active_restart_id = retry_matches[0]["event_id"]
        if not closed_results:
            return None
        if len(closed_results) != 1:
            raise ContractValidationError(
                "同一 human Restart 形成了多条真实结果"
            )
        return closed_results[0]

    def _human_correction_result_view(
        self,
        conn,
        task_id: str,
        result_row: Any,
        result_payload: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """投影人工补丁后的 Attempt 或真实工具技术暂停。"""

        if result_row["event_type"] != "execution_paused":
            return {
                "kind": "attempt",
                "record_id": result_row["event_id"],
                "attempt": deepcopy(result_payload),
                "tool_pause": None,
            }
        pause_sources = self._decode_json(
            result_row["source_refs_json"],
            field="tool Pause.source_refs_json",
        )
        manifest_refs = sorted(item for item in pause_sources if "/" in item)
        pause_view = {
            **dict(result_row),
            "payload": result_payload,
            "source_refs": pause_sources,
        }
        expanded = self._pause_source_evidence(
            conn,
            task_id,
            pause_view,
        )
        source_evidence = []
        for item in expanded:
            reference = item.get("artifact_ref")
            if not isinstance(reference, Mapping):
                raise ContractValidationError(
                    "tool Pause 展开证据缺少 artifact ref"
                )
            source_evidence.append(
                {
                    "manifest_ref": manifest_refs[0],
                    **deepcopy(dict(reference)),
                    "content": deepcopy(item["content"]),
                }
            )
        return {
            "kind": "tool_pause",
            "record_id": result_row["event_id"],
            "attempt": None,
            "tool_pause": {
                "reason_code": result_payload["reason_code"],
                "operation": result_payload["operation"],
                "evidence_manifest_refs": manifest_refs,
                "source_evidence": source_evidence,
            },
        }

    def _workflow_human_correction_chains(
        self,
        conn,
        task_id: str,
        decoded_revisions: Sequence[Tuple[Any, Mapping[str, Any]]],
        combination_rows: Mapping[str, Any],
        batch_rows: Mapping[str, Any],
    ) -> List[Dict[str, Any]]:
        """只投影已经形成真实 Attempt 或工具暂停的人工纠错链。"""

        restart_rows = conn.execute(
            "SELECT * FROM memory_events WHERE task_id=? "
            "AND event_type='restart_applied' ORDER BY created_at, event_id",
            (task_id,),
        ).fetchall()
        result_rows = conn.execute(
            "SELECT * FROM memory_events WHERE task_id=? "
            "AND event_type IN ('attempt', 'execution_paused') "
            "ORDER BY created_at, event_id",
            (task_id,),
        ).fetchall()
        retry_restarts_by_pause = self._workflow_retry_restarts_by_pause(
            restart_rows
        )
        human_chains = []
        for revision_row, revision in decoded_revisions:
            if revision_row["revision_type"] != "human_workflow_patch":
                continue
            revision_sources = self._decode_json(
                revision_row["source_refs_json"],
                field="human revision.source_refs_json",
            )
            pause_matches = []
            for source_ref in revision_sources:
                event = conn.execute(
                    "SELECT * FROM memory_events WHERE task_id=? AND event_id=? "
                    "AND event_type='execution_paused'",
                    (task_id, source_ref),
                ).fetchone()
                if event is not None:
                    pause_matches.append(event)
            if len(pause_matches) != 1:
                raise ContractValidationError(
                    "human revision 必须引用唯一 source Pause"
                )
            source_pause = pause_matches[0]

            restart_matches = []
            for restart_row in restart_rows:
                refs = self._decode_json(
                    restart_row["source_refs_json"],
                    field="Restart.source_refs_json",
                )
                payload = self._decode_json(
                    restart_row["payload_json"],
                    field="Restart.payload_json",
                )
                if (
                    revision_row["revision_id"] in refs
                    and payload.get("action") == "patch_and_continue"
                ):
                    restart_matches.append(restart_row)
            if len(restart_matches) != 1:
                raise ContractValidationError(
                    "human revision 必须由唯一 patch Restart 采用"
                )
            restart_row = restart_matches[0]
            closed_result = self._closed_human_correction_result(
                result_rows,
                restart_row["event_id"],
                retry_restarts_by_pause,
            )
            if closed_result is None:
                continue
            result_row, result_payload = closed_result
            result = self._human_correction_result_view(
                conn,
                task_id,
                result_row,
                result_payload,
            )

            combination = combination_rows.get(revision["combination_id"])
            if combination is None:
                raise ContractValidationError(
                    "human correction chain 缺少 combination projection"
                )
            batch = batch_rows[combination["batch_id"]]
            human_chains.append(
                {
                    "human_revision_id": revision_row["revision_id"],
                    "batch_index": batch["batch_index"],
                    "combination_index": combination["combination_index"],
                    "hypothesis": self._workflow_revision_hypothesis(
                        conn,
                        task_id,
                        combination,
                        batch_rows,
                    ),
                    "pause_record_id": source_pause["event_id"],
                    "restart_record_id": restart_row["event_id"],
                    "before_workflow": deepcopy(revision["before_workflow"]),
                    "normalized_patch": deepcopy(revision["normalized_patch"]),
                    "after_workflow": deepcopy(revision["after_workflow"]),
                    "result": result,
                }
            )
        return human_chains

    def get_active_workflow_revision(
        self,
        task_id: str,
        combination_id: str,
        *,
        workflow_version: int,
        current_workflow: Mapping[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """读取当前工作流版本唯一对应的修订，不产生任何持久化副作用。

        输入是任务、组合、工作流版本和当前工作流正文。首版工作流在检查 Working
        Memory 或访问数据库前直接返回 ``None``；后续版本在一个 SQLite 读事务中
        解码任务的修订正文，按组合、目标版本和完整工作流正文精确匹配，返回包含
        修订身份与正文的副本。修订 JSON 损坏或匹配结果不是唯一一条时抛出
        ``ContractValidationError``，不会用默认值掩盖历史事实。
        """

        if (
            isinstance(workflow_version, bool)
            or not isinstance(workflow_version, int)
            or workflow_version < 1
        ):
            raise ContractValidationError("workflow_version 必须是正整数")
        if workflow_version == 1:
            return None
        store = self._require_enabled()
        require_nonempty_string(task_id, "task_id")
        require_nonempty_string(combination_id, "combination_id")
        if not isinstance(current_workflow, Mapping):
            raise ContractValidationError("current_workflow 必须是 JSON object")
        with store.transaction() as conn:
            rows = conn.execute(
                "SELECT revision_id, revision_type, revised_result_json FROM revisions "
                "WHERE task_id=? ORDER BY created_at, revision_id",
                (task_id,),
            ).fetchall()
            matches = []
            for row in rows:
                try:
                    payload = json.loads(row["revised_result_json"])
                except (TypeError, json.JSONDecodeError) as exc:
                    raise ContractValidationError("revision payload JSON 已损坏") from exc
                if (
                    isinstance(payload, dict)
                    and payload.get("combination_id") == combination_id
                    and payload.get("workflow_version_after") == workflow_version
                    and payload.get("after_workflow") == current_workflow
                ):
                    matches.append(
                        {
                            "revision_id": row["revision_id"],
                            "revision_type": row["revision_type"],
                            **deepcopy(payload),
                        }
                    )
            if len(matches) != 1:
                raise ContractValidationError("当前 workflow 没有唯一来源 revision")
            return matches[0]

    def count_model_workflow_revisions(
        self,
        task_id: str,
        combination_id: str,
    ) -> int:
        """统计当前组合已经提交的模型工作流修订，纯读且排除人工修订。

        输入是任务和组合身份。方法在一个 SQLite 读事务中读取 model
        workflow revision 的正文，解码后只计入目标组合并返回数量；JSON 损坏时
        抛出 ``ContractValidationError``，不会跳过坏记录或提前缓存计数。
        """

        store = self._require_enabled()
        require_nonempty_string(task_id, "task_id")
        require_nonempty_string(combination_id, "combination_id")
        with store.transaction() as conn:
            rows = conn.execute(
                "SELECT revised_result_json FROM revisions "
                "WHERE task_id=? AND revision_type='model_workflow_revision'",
                (task_id,),
            ).fetchall()
            count = 0
            for row in rows:
                try:
                    payload = json.loads(row["revised_result_json"])
                except (TypeError, json.JSONDecodeError) as exc:
                    raise ContractValidationError("model revision JSON 已损坏") from exc
                if (
                    isinstance(payload, dict)
                    and payload.get("combination_id") == combination_id
                ):
                    count += 1
            return count

    def build_workflow_revision_package(
        self,
        task_id: str,
        combination_id: str,
        *,
        catalog_snapshot: Mapping[str, Any] | None = None,
        gaussian_route_policy: GaussianRoutePolicy | None = None,
    ) -> Dict[str, Any]:
        """构造当前组合的固定 Workflow Revision 阶段记忆视图。

        数据库中仍会联接完整工作流版本和 Attempt 以验证来源，但模型只接收当前
        组合的问题、假设、当前工作流、当前科学账本、Program Gate、裁判理由和
        同一组合的紧凑修复记录。其他组合、正式身份、暂停恢复和原始执行内容不
        会进入返回的模型视图。
        """

        store = self._require_enabled()
        require_nonempty_string(task_id, "task_id")
        require_nonempty_string(combination_id, "combination_id")
        with store.transaction() as conn:
            subject = self._workflow_revision_subject(
                conn,
                task_id,
                combination_id,
            )
            current = subject["current"]
            current_workflow = subject["current_workflow"]
            scientific_context = subject["scientific_context"]
            combination_rows = subject["combination_rows"]
            batch_rows = subject["batch_rows"]

            runtime = self._decode_json(
                current["runtime_checkpoint_json"], field="runtime_checkpoint_json"
            )
            direct_failure = (
                runtime.get("execution_failure")
                if isinstance(runtime, Mapping)
                else None
            )
            if isinstance(direct_failure, Mapping):
                self._validate_direct_execution_failure_source(
                    conn,
                    task_id,
                    current,
                    runtime,
                )
            sources = self._workflow_revision_sources(
                conn,
                task_id,
                allow_direct_execution_failure=isinstance(direct_failure, Mapping),
            )
            workflow_versions = sources["workflow_versions"]
            decoded_revisions = sources["decoded_revisions"]
            revise_attempts = sources["revise_attempts"]

            current_sources = [
                (row, payload)
                for row, payload in revise_attempts
                if payload.get("combination_id") == combination_id
                and payload.get("workflow_version") == current["workflow_version"]
                and payload.get("execution_id") == current["execution_id"]
            ]
            if len(current_sources) > 1:
                raise ContractValidationError("当前 workflow revision source Attempt 不唯一")
            expert_feedback = runtime.get("expert_revision_feedback") if isinstance(runtime, Mapping) else None
            expert_revision = isinstance(expert_feedback, Mapping)
            if (
                not current_sources
                and not isinstance(direct_failure, Mapping)
                and not expert_revision
            ):
                raise ContractValidationError(
                    "当前 workflow revision 必须有唯一 source Attempt"
                )
            source_attempt_row = None
            source_attempt = current_sources[0][1] if current_sources else None
            if source_attempt is None and isinstance(direct_failure, Mapping):
                source_attempt = {}

            current_projection = project_workflow_refs(current_workflow)
            hypothesis = self._workflow_revision_hypothesis(
                conn,
                task_id,
                current,
                batch_rows,
            )
            full_ledger = project_scientific_attempt_ledger(
                source_attempt.get("execution_result") if isinstance(source_attempt, Mapping) else None,
                workflow=current_workflow,
            )
            if isinstance(direct_failure, Mapping):
                diagnostic = direct_failure.get("diagnostic")
                current_attempt = {
                    "scientific_ledger": full_ledger,
                    "program_gate": {
                        "status": "not_run",
                        "reason": "execution_failure_direct_to_planner",
                    },
                    "judge": {
                        "decision": "execution_failure",
                        "reason": (
                            diagnostic.get("message")
                            if isinstance(diagnostic, Mapping) and isinstance(diagnostic.get("message"), str)
                            else "执行步骤失败"
                        ),
                        "evidence_refs": [direct_failure.get("failed_step_manifest_ref")],
                        "execution_failure": deepcopy(dict(direct_failure)),
                        "allowed_patch_scope": "failed_step_and_pending_suffix_only",
                    },
                }
            elif expert_revision:
                current_attempt = {
                    "scientific_ledger": full_ledger,
                    "program_gate": {
                        "status": "not_run",
                        "reason": "expert_review_direct_to_planner",
                    },
                    "judge": {
                        "decision": "expert_review_revision",
                        "reason": expert_feedback.get("reason", "Expert requested Planner revision"),
                        "expert_feedback": deepcopy(dict(expert_feedback)),
                        "allowed_patch_scope": "full_workflow_before_execution",
                    },
                }
            else:
                current_attempt = {
                    "scientific_ledger": full_ledger,
                    "program_gate": project_program_gate_fact(
                        source_attempt.get("program_validation"),
                        evidence_offset=_ledger_fact_count(full_ledger),
                        workflow=current_workflow,
                    ),
                    "judge": project_judge_fact(
                        source_attempt.get("router_decision"),
                        step_count=len(full_ledger["steps"]),
                        ledger=full_ledger,
                        program_validation=source_attempt.get("program_validation"),
                        workflow=current_workflow,
                    ),
                }

            prior_repairs: list[Dict[str, Any]] = []
            # 这里只收集当前 combination 的历史修订；调用方在 Memory Off
            # 时会将数组清空，当前 attempt 与 workflow 保持逐字不变。
            for revision_row, revision in decoded_revisions:
                if revision.get("combination_id") != combination_id:
                    continue
                after_version = revision.get("workflow_version_after")
                followups = [
                    payload
                    for row, payload in revise_attempts
                    if payload.get("combination_id") == combination_id
                    and payload.get("workflow_version") == after_version
                ]
                followup = followups[0] if followups else {}
                repair = project_workflow_revision_repair(
                    normalized_patch=revision.get("normalized_patch", {}),
                    before_workflow=revision.get("before_workflow", {}),
                    after_workflow=revision.get("after_workflow", {}),
                    execution_result=followup.get("execution_result"),
                    program_validation=followup.get("program_validation"),
                    router_decision=followup.get("router_decision"),
                    catalog_snapshot=catalog_snapshot,
                )
                prior_repairs.append(repair)
            context = project_hypothesis_regeneration_context(
                scientific_question=scientific_context.get("scientific_question"),
                initial_query_angles=[],
                initial_retrieval_phase={},
                chemistry_context=scientific_context.get("chemistry_context", {}),
            )
            revision_context = {
                "scientific_question": context["scientific_question"],
                "chemistry_context": context["chemistry_context"],
            }
            return {
                "scientific_context": revision_context,
                "hypothesis": hypothesis,
                "current_workflow": project_workflow_for_revision(
                    current_workflow,
                    catalog_snapshot=catalog_snapshot,
                ),
                "current_attempt": current_attempt,
                "tool_catalog_projection": (
                    project_minichem_stage_tool_projection(
                        catalog_snapshot,
                        gaussian_route_policy=gaussian_route_policy,
                    )
                    if isinstance(catalog_snapshot, Mapping)
                    else []
                ),
                "prior_repairs": prior_repairs,
            }

    def get_workflow_revision_source_attempt_id(
        self,
        task_id: str,
        combination_id: str,
        *,
        workflow_version: int,
        execution_id: str,
    ) -> str:
        """读取当前工作流修订唯一对应的 revise_workflow Attempt 身份。

        输入是任务、组合和当前工作流的执行身份。方法只查询保存过的 Attempt 事件，
        复验其 Router 决定与三项身份，并要求恰好找到一条；它不会把工作流修订所
        依据的非终态 Attempt 写入组合的 ``terminal_record_id``。组合不在修订态、
        Attempt 缺失、重复或正文损坏时抛出契约错误，避免恢复器引用不确定的来源。
        """

        store = self._require_enabled()
        require_nonempty_string(task_id, "task_id")
        require_nonempty_string(combination_id, "combination_id")
        if (
            isinstance(workflow_version, bool)
            or not isinstance(workflow_version, int)
            or workflow_version < 1
        ):
            raise ContractValidationError("workflow_version 必须是正整数")
        require_nonempty_string(execution_id, "execution_id")
        with store.transaction() as conn:
            combination = conn.execute(
                "SELECT state, workflow_version, execution_id FROM combinations "
                "WHERE task_id=? AND combination_id=?",
                (task_id, combination_id),
            ).fetchone()
            if combination is None:
                raise ContractValidationError("未知 combination")
            if combination["state"] != "revising_workflow":
                runtime_row = conn.execute(
                    "SELECT runtime_checkpoint_json FROM combinations WHERE task_id=? AND combination_id=?",
                    (task_id, combination_id),
                ).fetchone()
                runtime = self._decode_json(runtime_row["runtime_checkpoint_json"], field="runtime_checkpoint_json") if runtime_row else {}
                failure = runtime.get("execution_failure") if isinstance(runtime, Mapping) else None
                failure_ref = failure.get("event_id") if isinstance(failure, Mapping) else None
                if isinstance(failure_ref, str) and failure_ref:
                    return self._resolve_event_reference(
                        conn,
                        task_id,
                        failure_ref,
                        event_type="execution_step_failed",
                    )
                raise ContractValidationError(
                    "只有 revising_workflow 或 execution failure 组合才有 Revision 来源"
                )
            runtime_row = conn.execute(
                "SELECT runtime_checkpoint_json FROM combinations WHERE task_id=? AND combination_id=?",
                (task_id, combination_id),
            ).fetchone()
            runtime = self._decode_json(runtime_row["runtime_checkpoint_json"], field="runtime_checkpoint_json") if runtime_row else {}
            expert_feedback = runtime.get("expert_revision_feedback") if isinstance(runtime, Mapping) else None
            if isinstance(expert_feedback, Mapping):
                source = expert_feedback.get("source_event_id") or expert_feedback.get("operation_instance_id")
                if isinstance(source, str) and source:
                    return source
            if (
                combination["workflow_version"] != workflow_version
                or combination["execution_id"] != execution_id
            ):
                raise ContractValidationError(
                    "Revision 来源 Attempt 与当前工作流身份不一致"
                )
            rows = conn.execute(
                "SELECT event_id, payload_json FROM memory_events "
                "WHERE task_id=? AND event_type='attempt' "
                "AND json_extract(payload_json, '$.combination_id')=? "
                "AND json_extract(payload_json, '$.workflow_version')=? "
                "AND json_extract(payload_json, '$.execution_id')=? "
                "ORDER BY created_at, event_id",
                (task_id, combination_id, workflow_version, execution_id),
            ).fetchall()
            matches: list[str] = []
            for row in rows:
                payload = validate_attempt_payload(
                    self._decode_json(
                        row["payload_json"],
                        field="Revision source Attempt.payload_json",
                    ),
                    allow_legacy_router=True,
                )
                if payload["router_decision"].get("decision") == "revise_workflow":
                    matches.append(row["event_id"])
            if len(matches) != 1:
                raise ContractValidationError(
                    "当前 workflow revision 没有唯一 source Attempt"
                )
            return matches[0]

    @staticmethod
    def model_revision_id(task_id: str, combination_id: str, operation_instance_id: str) -> str:
        """让 Controller 在保存文件前即可得到本次模型修订的稳定身份。"""

        return model_revision_uuid(task_id, combination_id, operation_instance_id)

    def record_model_revision(
        self,
        task_id: str,
        payload: Mapping[str, Any],
        source_refs: Sequence[str],
        *,
        expected_version: int,
        source_attempt_id: str,
        expected_execution_id: str,
        new_execution_id: str,
    ) -> str:
        """保存有效模型修订，并让组合从新版本工作流第一步重新开始。

        修订正文、版本、尝试序号和新执行身份在同一事务中更新；任一项冲突都会
        使整个操作回滚。
        """

        store = self._require_enabled()
        value = validate_revision_payload(dict(payload))
        require_nonempty_string(source_attempt_id, "source_attempt_id")
        require_nonempty_string(expected_execution_id, "expected_execution_id")
        require_nonempty_string(new_execution_id, "new_execution_id")
        if expected_execution_id == new_execution_id:
            raise ContractValidationError("model revision 必须分配新的 execution_id")
        # 新执行身份由 Controller 分配，不能要求模型回显。它仍属于修订造成的正式
        # 状态变化，所以写入既有修订 JSON，投影删除后才能精确恢复。
        stored_value = {**value, "execution_id_after": new_execution_id}
        revision_id = model_revision_uuid(
            task_id, value["combination_id"], value["operation_instance_id"]
        )
        # 一次模型修订必须同时引用来源 Attempt、模型清单和工作流修订清单。
        refs = validate_source_ref_shape(list(source_refs))
        path_refs = [item for item in refs if "/" in item]
        id_refs = [item for item in refs if "/" not in item]
        if len(refs) != 3 or len(path_refs) != 2 or len(id_refs) != 1:
            raise ContractValidationError(
                "model revision.source_refs 必须恰含 source Attempt 和两份 finalized manifest"
            )
        with store.immediate_transaction() as conn:
            # 兼容旧失败投影把 event_key 作为 source_attempt_id 的任务；落盘时仍
            # 只保存真实 event_id，使后续恢复和 source_ref 校验保持同一身份。
            resolved_source_attempt_id = self._resolve_event_reference(
                conn,
                task_id,
                source_attempt_id,
            )
            resolved_id_ref = self._resolve_event_reference(
                conn,
                task_id,
                id_refs[0],
            )
            if resolved_id_ref != resolved_source_attempt_id:
                raise ContractValidationError(
                    "model revision.source_refs 的 source Attempt 身份不一致"
                )
            source_attempt_id = resolved_source_attempt_id
            refs = validate_source_ref_shape(sorted([source_attempt_id, *path_refs]))
            # 确定性修订身份使网络重放可返回原结果，但不同内容不能覆盖已有修订。
            existing = conn.execute(
                "SELECT * FROM revisions WHERE revision_id=?", (revision_id,)
            ).fetchone()
            if existing is not None:
                same = (
                    existing["task_id"] == task_id
                    and existing["revision_type"] == "model_workflow_revision"
                    and canonical_json(
                        self._decode_json(existing["revised_result_json"], field="revised_result_json")
                    )
                    == canonical_json(stored_value)
                    and canonical_json(
                        self._decode_json(existing["source_refs_json"], field="source_refs_json")
                    )
                    == canonical_json(refs)
                )
                if not same:
                    raise IdempotencyConflict(f"revision_id={revision_id} 已绑定到不同内容")
                attempt = conn.execute(
                    "SELECT event_type, payload_json FROM memory_events WHERE task_id=? AND event_id=?",
                    (task_id, source_attempt_id),
                ).fetchone()
                attempt_payload = (
                    {}
                    if attempt is None
                    else self._decode_json(attempt["payload_json"], field="Attempt.payload_json")
                )
                if (
                    attempt is None
                    or attempt["event_type"] not in {"attempt", "execution_step_failed", "expert_gaussian_review"}
                    or attempt_payload.get("combination_id") != value["combination_id"]
                    or attempt_payload.get("workflow_version") != value["workflow_version_before"]
                    or attempt_payload.get("execution_id") != expected_execution_id
                ):
                    raise IdempotencyConflict("revision exact replay 的 source Attempt identity 不一致")
                projection = conn.execute(
                    "SELECT execution_id FROM combinations WHERE task_id=? AND combination_id=?",
                    (task_id, value["combination_id"]),
                ).fetchone()
                if projection is None or projection["execution_id"] != new_execution_id:
                    raise IdempotencyConflict("revision exact replay 的 execution identity 不一致")
                return revision_id
            self._assert_task_writable(conn, task_id)
            self._validate_source_refs(conn, task_id, refs)
            attempt = conn.execute(
                "SELECT event_type, payload_json FROM memory_events WHERE task_id=? AND event_id=?",
                (task_id, source_attempt_id),
            ).fetchone()
            if attempt is None or attempt["event_type"] not in {"attempt", "execution_step_failed", "expert_gaussian_review"}:
                raise ContractValidationError("source_attempt_id 必须引用当前 task 的 Attempt、execution_step_failed 或 Expert review")
            attempt_payload = self._decode_json(attempt["payload_json"], field="Revision source payload_json")
            if (
                attempt_payload.get("combination_id") != value["combination_id"]
                or attempt_payload.get("workflow_version") != value["workflow_version_before"]
                or attempt_payload.get("execution_id") != expected_execution_id
            ):
                raise ContractValidationError("Revision source 与 revision target identity 不一致")
            if attempt["event_type"] == "execution_step_failed":
                validate_execution_step_failed_payload(attempt_payload)
            elif attempt["event_type"] == "expert_gaussian_review":
                if attempt_payload.get("status") == "approved":
                    raise ContractValidationError("approved Expert review 不能作为 Revision 来源")
            self._validate_causal_source(conn, task_id, value["causal_source_ref"])
            self._assert_operation_available(conn, task_id, value["operation_instance_id"])
            row = conn.execute(
                "SELECT * FROM combinations WHERE task_id=? AND combination_id=?",
                (task_id, value["combination_id"]),
            ).fetchone()
            if row is None:
                raise ContractValidationError("revision 引用未知 combination")
            if (
                row["version"] != expected_version
                or (
                    row["state"] != "revising_workflow"
                    and not isinstance(
                        self._decode_json(row["runtime_checkpoint_json"], field="runtime_checkpoint_json").get("execution_failure"),
                        Mapping,
                    )
                )
                or row["workflow_version"] != value["workflow_version_before"]
                or row["execution_id"] != expected_execution_id
                or self._decode_json(row["current_workflow_json"], field="current_workflow_json")
                != value["before_workflow"]
                or row["causal_source_ref"] != value["causal_source_ref"]
            ):
                raise CASConflict(
                    "model revision 的 CAS 前提已变化",
                    expected_version=expected_version,
                    current_version=row["version"],
                    expected_state="revising_workflow or execution_failure",
                    current_state=row["state"],
                )
            now = utc_now()
            conn.execute(
                """
                INSERT INTO revisions(
                    revision_id, task_id, decision_id, round, revision_type,
                    instructions_json, target_problem_codes_json, generation_status,
                    verification_status, applied_round, revised_result_json,
                    source_refs_json, created_at, updated_at
                ) VALUES(?, ?, NULL, ?, 'model_workflow_revision', ?, '[]',
                    'completed', ?, ?, ?, ?, ?, ?)
                """,
                (
                    revision_id,
                    task_id,
                    row["attempt_index"],
                    canonical_json(value["normalized_patch"]),
                    VERIFICATION_PENDING,
                    row["attempt_index"] + 1,
                    canonical_json(stored_value),
                    canonical_json(refs),
                    now,
                    now,
                ),
            )
            previous_runtime = self._decode_json(
                row["runtime_checkpoint_json"], field="runtime_checkpoint_json"
            )
            direct_failure = previous_runtime.get("execution_failure") if isinstance(previous_runtime, Mapping) else None
            runtime = empty_runtime_checkpoint()
            if isinstance(direct_failure, Mapping):
                failed_index = direct_failure.get("step_index")
                old_refs = previous_runtime.get("step_manifest_refs")
                before_steps = value["before_workflow"].get("steps")
                after_steps = value["after_workflow"].get("steps")
                if (
                    isinstance(failed_index, bool)
                    or not isinstance(failed_index, int)
                    or failed_index < 0
                    or not isinstance(old_refs, list)
                    or not isinstance(before_steps, list)
                    or not isinstance(after_steps, list)
                    or len(before_steps) != len(old_refs)
                    or len(after_steps) <= failed_index
                    or before_steps[:failed_index] != after_steps[:failed_index]
                ):
                    raise ContractValidationError("workflow revision 不能修改已完成步骤前缀")
                runtime["next_step_index"] = failed_index
                runtime["step_manifest_refs"] = list(old_refs[:failed_index])
                runtime["step_revision_attempts"] = deepcopy(previous_runtime.get("step_revision_attempts", {}))
                prefix_ids = {step.get("step_id") for step in after_steps[:failed_index] if isinstance(step, Mapping)}
                old_code_results = previous_runtime.get("gaussian_code_results", {})
                runtime["gaussian_code_results"] = {
                    str(step_id): deepcopy(item)
                    for step_id, item in (old_code_results.items() if isinstance(old_code_results, Mapping) else [])
                    if step_id in prefix_ids
                }
            cursor = conn.execute(
                """
                UPDATE combinations SET
                    workflow_version=?, attempt_index=attempt_index+1, execution_id=?,
                    state='running', current_workflow_json=?, runtime_checkpoint_json=?,
                    causal_source_ref=NULL, version=version+1, updated_at=?
                WHERE task_id=? AND combination_id=?
                  AND (state='revising_workflow' OR state='running')
                  AND version=? AND workflow_version=?
                """,
                (
                    value["workflow_version_after"],
                    new_execution_id,
                    canonical_json(value["after_workflow"]),
                    canonical_json(runtime),
                    now,
                    task_id,
                    value["combination_id"],
                    expected_version,
                    value["workflow_version_before"],
                ),
            )
            if cursor.rowcount != 1:
                raise CASConflict(
                    "model revision 输掉并发 CAS",
                    expected_version=expected_version,
                    current_version=row["version"],
                    expected_state="revising_workflow",
                    current_state=row["state"],
                )
            return revision_id

    # ==================== 公开读取与恢复投影 ====================

    def get_task_created(self, task_id: str) -> Dict[str, Any]:
        """读取任务创建时冻结的问题、预算、目录和 Tool Catalog 身份。

        恢复器必须以 TaskCreated 事实为准，不能相信调用方重新传入的参数，也不能
        从后来变化的任务投影反推初始配置。这里返回深拷贝，调用方整理 preflight
        参数时不会意外修改服务内部读取到的对象。
        """

        store = self._require_enabled()
        with store.transaction() as conn:
            return deepcopy(self._task_created_payload(conn, task_id))

    def get_memory_event_type(self, task_id: str, event_id: str) -> str:
        """读取当前 task 内一条不可变 memory event 的类型。

        输入是 task 身份和 Working Memory 投影保存的真实 event_id；动作只读取
        ``memory_events``，不改变任何事实、不解析事件正文，也不返回 token、路径或
        transcript。成功返回事件类型；task、event 缺失或引用跨 task 时抛出合同错误，
        让恢复调用方不能把任意字符串当成更正授权。
        """

        store = self._require_enabled()
        require_nonempty_string(task_id, "task_id")
        require_nonempty_string(event_id, "event_id")
        with store.transaction() as conn:
            row = conn.execute(
                "SELECT event_type FROM memory_events WHERE task_id=? AND event_id=?",
                (task_id, event_id),
            ).fetchone()
            if row is None:
                raise ContractValidationError(
                    "memory event 不存在或不属于当前 task"
                )
            event_type = row["event_type"]
            if not isinstance(event_type, str) or not event_type:
                raise ContractValidationError("memory event 类型损坏")
            return event_type

    def get_investigation_repair_event_identity(
        self,
        task_id: str,
        event_id: str,
    ) -> Dict[str, str]:
        """读取一条 Investigation 返修来源事件的类型和业务身份。

        输入是当前任务及 pending repair 保存的真实事件 ID；动作在同一只读事务中
        读取事件类型和正文，并按 correction 或普通 repair 的正式合同重新校验
        task、batch、combination。返回只含这四个公开身份字段的副本；事件缺失、
        类型不允许、正文损坏或 task 身份漂移时抛出合同错误，不返回修复建议等正文。
        """

        store = self._require_enabled()
        require_nonempty_string(task_id, "task_id")
        require_nonempty_string(event_id, "event_id")
        with store.transaction() as conn:
            row = conn.execute(
                "SELECT event_type,payload_json FROM memory_events "
                "WHERE task_id=? AND event_id=?",
                (task_id, event_id),
            ).fetchone()
            if row is None:
                raise ContractValidationError(
                    "Investigation repair event 不存在或不属于当前 task"
                )
            event_type = row["event_type"]
            payload = self._decode_json(
                row["payload_json"],
                field="InvestigationRepairAuthorization.payload_json",
            )
            if event_type == "investigation_task_correction_recorded":
                value = validate_investigation_task_correction_payload(payload)
            elif event_type == "investigation_repair_requested":
                value = validate_investigation_repair_requested_payload(payload)
            else:
                raise ContractValidationError(
                    "memory event 类型不允许作为 Investigation repair 来源"
                )
            payload_task_id = value.get("task_id", task_id)
            if payload_task_id != task_id:
                raise ContractValidationError(
                    "Investigation repair event 的 task 身份已漂移"
                )
            return {
                "event_type": event_type,
                "task_id": task_id,
                "batch_id": value["batch_id"],
                "combination_id": value["combination_id"],
            }

    def get_effective_scientific_context(self, task_id: str) -> Dict[str, Any]:
        """读取 Retrieval 优先的科学上下文，供规划、执行、Router 与修订共用。"""

        store = self._require_enabled()
        require_nonempty_string(task_id, "task_id")
        with store.transaction() as conn:
            return deepcopy(self._effective_scientific_context(conn, task_id))

    def get_pending_hypothesis_restart(
        self,
        task_id: str,
    ) -> Optional[Dict[str, str]]:
        """读取任务级假设生成当前唯一的恢复授权和原失败清单。"""

        store = self._require_enabled()
        require_nonempty_string(task_id, "task_id")
        with store.transaction() as conn:
            return deepcopy(
                self._pending_task_hypothesis_restart(conn, task_id)
            )

    def get_task(self, task_id: str) -> Dict[str, Any]:
        """读取任务当前投影，并叠加首批 Retrieval 冻结的科学上下文。"""

        store = self._require_enabled()
        require_nonempty_string(task_id, "task_id")
        with store.immediate_transaction() as conn:
            task_created = self._task_created_payload(conn, task_id)
            row = self._ensure_task_projection_matches_created(
                conn, task_id, task_created
            )
            result = task_row(row)
            effective = self._effective_scientific_context(conn, task_id)
            chemistry_context = deepcopy(effective["chemistry_context"])
            result["chemistry_context"] = chemistry_context
            result["reaction_class"] = (
                chemistry_context.get("reaction_type")
                or chemistry_context.get("reaction_class")
            )
            return result

    def get_unresolved_retrieval_pause(self, task_id: str) -> Optional[Dict[str, Any]]:
        """读取当前任务唯一未解决的 Retrieval 暂停事实。"""

        store = self._require_enabled()
        with store.transaction() as conn:
            row = conn.execute(
                "SELECT event_id, payload_json, source_refs_json FROM memory_events "
                "WHERE task_id=? AND event_type='retrieval_paused' "
                "AND NOT EXISTS (SELECT 1 FROM memory_events consumer "
                "  WHERE consumer.task_id=memory_events.task_id "
                "    AND json_extract(consumer.payload_json, '$.causal_source_ref')="
                "        memory_events.event_id) "
                "ORDER BY created_at DESC LIMIT 1",
                (task_id,),
            ).fetchone()
            if row is None:
                return None
            payload = validate_retrieval_pause_payload(
                self._decode_json(
                    row["payload_json"], field="RetrievalPaused.payload_json"
                )
            )
            source_refs = validate_source_ref_shape(
                self._decode_json(
                    row["source_refs_json"],
                    field="RetrievalPaused.source_refs_json",
                )
            )
            self._validate_source_refs(conn, task_id, source_refs)
            result = {
                "pause_id": row["event_id"],
                "payload": payload,
                "source_refs": source_refs,
            }
            # 请求只存在于经过 source_ref 校验的 receipt artifact 中；旧版暂停
            # 没有该字段时仍可被读取，但不能被自动重试。
            artifact_ref = source_refs[0] if source_refs else None
            if isinstance(artifact_ref, str) and "/" in artifact_ref:
                task_root = self._task_output_root(conn, task_id)
                artifact_path = (task_root / artifact_ref).resolve()
                try:
                    artifact_path.relative_to(task_root)
                except ValueError as exc:
                    raise ContractValidationError("Retrieval 暂停 artifact 越出任务目录") from exc
                if artifact_path.is_file():
                    try:
                        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
                    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise ContractValidationError("Retrieval 暂停 artifact 无法读取") from exc
                    if isinstance(artifact, Mapping) and isinstance(
                        artifact.get("retrieval_request"), Mapping
                    ):
                        result["retrieval_request"] = deepcopy(
                            dict(artifact["retrieval_request"])
                        )
            return result

    def get_retrieval_completed(self, task_id: str) -> Optional[Dict[str, Any]]:
        """读取并校验最近一次 RetrievalCompleted 的 frozen result receipt。"""

        store = self._require_enabled()
        require_nonempty_string(task_id, "task_id")
        with store.transaction() as conn:
            row = conn.execute(
                "SELECT event_id, payload_json, source_refs_json FROM memory_events "
                "WHERE task_id=? AND event_type='retrieval_completed' "
                "ORDER BY created_at DESC LIMIT 1",
                (task_id,),
            ).fetchone()
            if row is None:
                return None
            payload = validate_retrieval_completed_payload(
                self._decode_json(
                    row["payload_json"], field="RetrievalCompleted.payload_json"
                )
            )
            source_refs = validate_source_ref_shape(
                self._decode_json(
                    row["source_refs_json"],
                    field="RetrievalCompleted.source_refs_json",
                )
            )
            if source_refs != [payload["artifact_ref"]]:
                raise ContractValidationError(
                    "RetrievalCompleted.source_refs 必须唯一引用结果 artifact"
                )
            self._validate_source_refs(conn, task_id, source_refs)
            task_root = self._task_output_root(conn, task_id)
            artifact_path = (task_root / payload["artifact_ref"]).resolve()
            try:
                artifact_path.relative_to(task_root)
            except ValueError as exc:
                raise ContractValidationError("RetrievalCompleted artifact 越出任务目录") from exc
            if not artifact_path.is_file() or artifact_path.is_symlink():
                raise ContractValidationError("RetrievalCompleted artifact 不存在")
            try:
                artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ContractValidationError("RetrievalCompleted artifact 无法读取") from exc
            if not isinstance(artifact, Mapping):
                raise ContractValidationError("RetrievalCompleted artifact 必须是 object")
            result = artifact.get("retrieval_result")
            if not isinstance(result, Mapping) or artifact.get("status") != "completed":
                raise ContractValidationError("RetrievalCompleted artifact 内容不完整")
            if artifact.get("result_digest") != payload["result_digest"]:
                raise ContractValidationError("RetrievalCompleted result_digest 不一致")
            if hashlib.sha256(canonical_json(dict(result)).encode("utf-8")).hexdigest() != payload[
                "result_digest"
            ]:
                raise ContractValidationError("RetrievalCompleted 结果摘要不匹配")
            return {
                "event_id": row["event_id"],
                "payload": deepcopy(payload),
                "source_refs": source_refs,
                "retrieval_result": deepcopy(dict(result)),
            }

    def get_batch(self, task_id: str, batch_id: str) -> Dict[str, Any]:
        """读取一批假设当前状态的只读副本。"""

        store = self._require_enabled()
        row = store.fetchone(
            "SELECT * FROM hypothesis_batches WHERE task_id=? AND batch_id=?",
            (task_id, batch_id),
        )
        if row is None:
            raise ContractValidationError("未知 batch")
        return batch_row(row)

    def find_batch_by_index(
        self,
        task_id: str,
        batch_index: int,
    ) -> Optional[Dict[str, Any]]:
        """按任务内批次序号读取完整批次投影，纯读且找不到时返回 ``None``。

        输入是任务身份和从 1 开始的批次序号。方法在一个 SQLite 读事务中直接
        返回匹配的完整批次行，不先查询身份再开启第二次读取；任务不存在或该序号
        尚未提交时返回 ``None``，Working Memory 未启用或输入非法时抛出相应错误。
        """

        store = self._require_enabled()
        require_nonempty_string(task_id, "task_id")
        if (
            isinstance(batch_index, bool)
            or not isinstance(batch_index, int)
            or batch_index < 1
        ):
            raise ContractValidationError("batch_index 必须是正整数")
        with store.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM hypothesis_batches WHERE task_id=? AND batch_index=?",
                (task_id, batch_index),
            ).fetchone()
            if row is None:
                return None
            return batch_row(row)

    def find_latest_batch(self, task_id: str) -> Optional[Dict[str, Any]]:
        """读取任务内批次序号最大的完整投影，纯读且无批次时返回 ``None``。

        输入是任务身份。方法在一个 SQLite 读事务中按 ``batch_index`` 降序直接
        返回最新批次的完整行；任务尚未提交首批时返回 ``None``，Working Memory
        未启用或输入非法时抛出相应错误。
        """

        store = self._require_enabled()
        require_nonempty_string(task_id, "task_id")
        with store.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM hypothesis_batches WHERE task_id=? "
                "ORDER BY batch_index DESC LIMIT 1",
                (task_id,),
            ).fetchone()
            if row is None:
                return None
            return batch_row(row)

    def get_batch_created(self, task_id: str, batch_id: str) -> Dict[str, Any]:
        """读取创建批次时冻结的完整事实。

        批次投影只保留调度所需的状态和数量，不重复保存假设正文。需要恢复某个
        候选时，必须沿投影记录的创建事件读取原始正文，避免从名称或排名反推。
        返回深拷贝，调用方整理模型输入时不会修改数据库中保存的事实。
        """

        store = self._require_enabled()
        with store.transaction() as conn:
            row = conn.execute(
                """
                SELECT e.event_type, e.payload_json
                FROM hypothesis_batches b
                JOIN memory_events e
                  ON e.task_id=b.task_id AND e.event_id=b.created_event_id
                WHERE b.task_id=? AND b.batch_id=?
                """,
                (task_id, batch_id),
            ).fetchone()
            if row is None:
                raise ContractValidationError("未知 batch")
            if row["event_type"] != "hypothesis_batch_created":
                raise ContractValidationError("batch 创建事件类型不匹配")
            payload = self._validated_batch_created_payload(
                conn,
                task_id,
                row["payload_json"],
                field="HypothesisBatchCreated.payload_json",
            )
        if payload.get("batch_id") != batch_id:
            raise ContractValidationError("batch 创建事件身份不匹配")
        return deepcopy(payload)

    def get_batch_created_receipt(
        self,
        task_id: str,
        batch_id: str,
    ) -> Dict[str, Any]:
        """读取 BatchCreated 的事件、payload 和 source_refs 三者事实。

        恢复器用该窄接口核对跨存储 receipt，不能只读取批次正文而忽略事件身份或
        manifest 来源。输入是任务与批次身份；动作只读已提交事件并重新执行当前
        payload/source_refs 校验，缺失、类型错误或事件类型漂移都会失败。
        """

        store = self._require_enabled()
        with store.transaction() as conn:
            row = conn.execute(
                """
                SELECT e.event_id, e.event_type, e.payload_json, e.source_refs_json
                FROM hypothesis_batches b
                JOIN memory_events e
                  ON e.task_id=b.task_id AND e.event_id=b.created_event_id
                WHERE b.task_id=? AND b.batch_id=?
                """,
                (task_id, batch_id),
            ).fetchone()
            if row is None:
                raise ContractValidationError("未知 batch")
            if row["event_type"] != "hypothesis_batch_created":
                raise ContractValidationError("batch 创建事件类型不匹配")
            payload = self._validated_batch_created_payload(
                conn,
                task_id,
                row["payload_json"],
                field="HypothesisBatchCreated.payload_json",
            )
            source_refs = validate_source_ref_shape(
                self._decode_json(row["source_refs_json"], field="source_refs_json")
            )
        if payload.get("batch_id") != batch_id:
            raise ContractValidationError("batch 创建事件身份不匹配")
        return {
            "event_id": row["event_id"],
            "payload": deepcopy(payload),
            "source_refs": deepcopy(source_refs),
        }

    def get_combination(self, task_id: str, combination_id: str) -> Dict[str, Any]:
        """读取一个假设与工作流组合的当前状态。"""

        store = self._require_enabled()
        row = store.fetchone(
            "SELECT * FROM combinations WHERE task_id=? AND combination_id=?",
            (task_id, combination_id),
        )
        if row is None:
            raise ContractValidationError("未知 combination")
        return combination_row(row)

    @staticmethod
    def _read_expert_debate_event(
        conn,
        task_id: str,
        combination_id: str,
        workflow_version: int,
        execution_id: str,
    ) -> Any:
        """按完整工作流身份读取唯一 ExpertDebate 事件。

        输入是事务连接、任务/组合身份和工作流执行版本；动作只查询不可变事件，
        返回唯一数据库行或 ``None``。没有事件表示可以继续同一边界，出现重复事件
        则抛出合同错误，不把历史重复误认为合法重放。
        """

        rows = conn.execute(
            "SELECT * FROM memory_events "
            "WHERE task_id=? AND event_type='expert_debate_completed' "
            "AND json_extract(payload_json, '$.combination_id')=? "
            "AND json_extract(payload_json, '$.workflow_version')=? "
            "AND json_extract(payload_json, '$.execution_id')=? "
            "ORDER BY created_at, event_id",
            (task_id, combination_id, workflow_version, execution_id),
        ).fetchall()
        if not rows:
            return None
        if len(rows) != 1:
            raise ContractValidationError(
                "同一 workflow/execution 存在多条 ExpertDebate 事实"
            )
        return rows[0]

    def _match_expert_debate_artifact_reference(
        self,
        conn,
        task_id: str,
        payload: Mapping[str, Any],
        source_refs: Sequence[str],
    ) -> str:
        """复验 ExpertDebate 的来源并找出唯一匹配 artifact。

        输入是事务连接、已验证的 debate payload 和事件来源引用；动作核对引用仍
        属于当前任务、复验每个 finalized manifest 的摘要，并比较 operation、debate、
        judge、route 四项正文。返回唯一 artifact 相对路径；缺少、重复或不匹配时
        抛出合同错误，绝不从任务目录猜测最新文件。
        """

        refs = validate_source_ref_shape(list(source_refs))
        self._validate_source_refs(conn, task_id, refs)
        artifact_refs = [item for item in refs if "/" in item]
        if not artifact_refs:
            raise ContractValidationError("ExpertDebate 缺少 artifact source_ref")
        matching_artifacts = []
        for relative_path in artifact_refs:
            artifact = self._read_finalized_manifest(conn, task_id, relative_path)
            if not isinstance(artifact, Mapping):
                continue
            if (
                artifact.get("operation_instance_id") == payload["operation_instance_id"]
                and artifact.get("debate") == payload["debate"]
                and artifact.get("judge") == payload["judge"]
                and artifact.get("route") == payload["route"]
            ):
                matching_artifacts.append(relative_path)
        if len(matching_artifacts) != 1:
            raise ContractValidationError(
                "ExpertDebate artifact 与事件 payload 不唯一匹配"
            )
        return matching_artifacts[0]

    @staticmethod
    def _require_expert_debate_combination_identity(
        conn,
        task_id: str,
        combination_id: str,
        workflow_version: int,
        execution_id: str,
    ) -> None:
        """确认 ExpertDebate 仍绑定当前组合的工作流和执行身份。

        输入是事务连接、任务/组合身份和期望版本；动作只读取当前组合，返回值为空。
        组合缺失或身份漂移时抛出合同错误，避免恢复入口复用旧版本 debate。
        """

        combination = conn.execute(
            "SELECT combination_id, workflow_version, execution_id "
            "FROM combinations WHERE task_id=? AND combination_id=?",
            (task_id, combination_id),
        ).fetchone()
        if combination is None:
            raise ContractValidationError("ExpertDebate 引用了未知 combination")
        if (
            combination["workflow_version"] != workflow_version
            or combination["execution_id"] != execution_id
        ):
            raise ContractValidationError(
                "ExpertDebate 与当前组合 workflow/execution 身份不一致"
            )

    def get_expert_debate(
        self,
        task_id: str,
        combination_id: str,
        *,
        workflow_version: int,
        execution_id: str,
    ) -> Optional[Dict[str, Any]]:
        """读取当前工作流版本已经提交的唯一 Expert debate 事实。

        查询只沿 ``expert_debate_completed`` 事件引用的 payload 和 artifact 进行，
        不扫描任务目录。返回值包含经过契约、身份、来源摘要和 artifact 正文复验的
        ``payload`` 与 ``source_refs``；没有匹配事件时返回 ``None``，可供恢复入口
        继续执行同一 debate 边界。发现重复或损坏历史时抛出契约错误，避免把不确定
        的旧记录当成可重试事实。
        """

        store = self._require_enabled()
        require_nonempty_string(task_id, "task_id")
        require_nonempty_string(combination_id, "combination_id")
        require_nonempty_string(execution_id, "execution_id")
        if (
            isinstance(workflow_version, bool)
            or not isinstance(workflow_version, int)
            or workflow_version < 1
        ):
            raise ContractValidationError("workflow_version 必须是正整数")
        with store.transaction() as conn:
            row = self._read_expert_debate_event(
                conn,
                task_id,
                combination_id,
                workflow_version,
                execution_id,
            )
            if row is None:
                return None
            payload = validate_expert_debate_payload(
                self._decode_json(row["payload_json"], field="ExpertDebate.payload_json")
            )
            refs = validate_source_ref_shape(
                self._decode_json(row["source_refs_json"], field="ExpertDebate.source_refs_json")
            )
            matching_artifact = self._match_expert_debate_artifact_reference(
                conn,
                task_id,
                payload,
                refs,
            )
            self._require_expert_debate_combination_identity(
                conn,
                task_id,
                combination_id,
                workflow_version,
                execution_id,
            )
            return {
                "payload": deepcopy(payload),
                "source_refs": [matching_artifact],
            }

    def get_checkpoint_resume_context(
        self,
        task_id: str,
        combination_id: str,
        *,
        expected_version: int,
    ) -> CheckpointResumeContext:
        """在一次只读事务内核对并返回 checkpoint 恢复所需的完整事实。

        输入是任务、组合和调用方刚读到的组合版本。方法读取当前组合及其运行游标，
        找到唯一被游标引用的 ``execution_checkpointed`` 事件，确认事件只引用该
        checkpoint 清单，再通过统一的 finalized manifest 读取器复验文件内容摘要。
        随后核对事件、清单与运行身份，并验证整个 execution 的编号从 1 到当前检查点
        连续、当前 operation 的事件正好形成末尾区间。方法只打开数据库读事务和读取
        已引用文件，不调用外部工具、不写数据库；版本不一致或任一事实损坏时抛出
        ``CASConflict`` 或 ``ContractValidationError``。
        """

        store = self._require_enabled()
        require_nonempty_string(task_id, "task_id")
        require_nonempty_string(combination_id, "combination_id")
        if (
            isinstance(expected_version, bool)
            or not isinstance(expected_version, int)
            or expected_version < 0
        ):
            raise ContractValidationError("expected_version 必须是非负 JSON integer")

        with store.transaction() as conn:
            # 先在同一只读事务中读取组合当前游标，确认调用方持有的版本仍然有效。
            row = conn.execute(
                "SELECT * FROM combinations WHERE task_id=? AND combination_id=?",
                (task_id, combination_id),
            ).fetchone()
            if row is None:
                raise ContractValidationError("未知 combination")
            if row["version"] != expected_version:
                raise CASConflict(
                    "checkpoint resume context 的 CAS 前提已变化",
                    expected_version=expected_version,
                    current_version=row["version"],
                    current_state=row["state"],
                )

            combination = combination_row(row)
            runtime = combination.get("runtime_checkpoint")
            if not isinstance(runtime, Mapping):
                raise ContractValidationError("runtime_checkpoint 必须是 JSON object")
            checkpoint_id = runtime.get("checkpoint_id")
            manifest_ref = runtime.get("checkpoint_manifest_ref")
            if not isinstance(checkpoint_id, str) or not checkpoint_id:
                raise ContractValidationError("等待组合缺少 checkpoint_id")
            if (
                not isinstance(manifest_ref, str)
                or not manifest_ref
                or "\\" in manifest_ref
            ):
                raise ContractValidationError("等待组合缺少规范 checkpoint_manifest_ref")
            relative = PurePosixPath(manifest_ref)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or "." in relative.parts
            ):
                raise ContractValidationError("checkpoint_manifest_ref 越过 task root")

            # 再读取游标引用的 checkpoint 事件和正式清单，核对唯一来源及身份。
            event = conn.execute(
                "SELECT event_type, payload_json, source_refs_json "
                "FROM memory_events WHERE task_id=? AND event_id=?",
                (task_id, checkpoint_id),
            ).fetchone()
            if event is None or event["event_type"] != "execution_checkpointed":
                raise ContractValidationError("runtime checkpoint 没有对应的已提交事件")
            try:
                payload = validate_checkpoint_payload(
                    self._decode_json(
                        event["payload_json"],
                        field="checkpoint payload_json",
                    )
                )
                source_refs = self._decode_json(
                    event["source_refs_json"],
                    field="checkpoint source_refs_json",
                )
            except ContractValidationError as exc:
                raise ContractValidationError("checkpoint event JSON 已损坏") from exc
            if not isinstance(payload, Mapping) or not isinstance(source_refs, list):
                raise ContractValidationError("checkpoint event JSON 已损坏")
            if source_refs != [manifest_ref]:
                raise ContractValidationError("checkpoint event 没有唯一引用 runtime 清单")

            manifest = self._read_finalized_manifest(conn, task_id, manifest_ref)
            expected_identity = {
                "combination_id": combination["combination_id"],
                "execution_id": combination["execution_id"],
                "workflow_version": combination["workflow_version"],
                "operation_instance_id": runtime.get("current_operation_instance_id"),
                "checkpoint_index": runtime.get("checkpoint_index"),
            }
            for field, expected in expected_identity.items():
                # combination_id 只存在于事件 payload；其余字段同时与事件和清单一致。
                if payload.get(field) != expected:
                    raise ContractValidationError(
                        f"checkpoint event.{field} 与 runtime 不一致"
                    )
                if field != "combination_id" and manifest.get(field) != expected:
                    raise ContractValidationError(
                        f"checkpoint manifest.{field} 与 runtime 不一致"
                    )
            for field in ("step_id", "step_index"):
                if payload.get(field) != manifest.get(field):
                    raise ContractValidationError(
                        f"checkpoint manifest.{field} 与 event 不一致"
                    )
            if self._manifest_ref_path(manifest.get("request_ref")) != runtime.get(
                "current_request_ref"
            ):
                raise ContractValidationError("checkpoint 清单没有复用当前请求引用")

            # 最后重建 execution 全局编号链，并截取当前 operation 的连续后缀。
            rows = conn.execute(
                "SELECT event_id, payload_json FROM memory_events "
                "WHERE task_id=? AND event_type='execution_checkpointed'",
                (task_id,),
            ).fetchall()
            matched: list[tuple[int, str]] = []
            execution_records: list[tuple[int, str]] = []
            for event_row in rows:
                try:
                    raw_event_payload = self._decode_json(
                        event_row["payload_json"],
                        field="checkpoint payload_json",
                    )
                except ContractValidationError as exc:
                    raise ContractValidationError("checkpoint event JSON 已损坏") from exc
                if not isinstance(raw_event_payload, Mapping):
                    raise ContractValidationError("checkpoint event JSON 已损坏")
                same_execution = all(
                    raw_event_payload.get(field) == expected
                    for field, expected in (
                        ("combination_id", combination["combination_id"]),
                        ("execution_id", combination["execution_id"]),
                    )
                )
                if not same_execution:
                    continue
                try:
                    event_payload = validate_checkpoint_payload(raw_event_payload)
                except ContractValidationError as exc:
                    index = raw_event_payload.get("checkpoint_index")
                    if (
                        isinstance(index, bool)
                        or not isinstance(index, int)
                        or index < 1
                    ):
                        raise ContractValidationError(
                            "checkpoint event index 非法"
                        ) from exc
                    operation_id = raw_event_payload.get("operation_instance_id")
                    if not isinstance(operation_id, str) or not operation_id:
                        raise ContractValidationError(
                            "checkpoint event operation identity 非法"
                        ) from exc
                    raise ContractValidationError("checkpoint event JSON 已损坏") from exc
                index = event_payload.get("checkpoint_index")
                if isinstance(index, bool) or not isinstance(index, int) or index < 1:
                    raise ContractValidationError("checkpoint event index 非法")
                operation_id = event_payload.get("operation_instance_id")
                if not isinstance(operation_id, str) or not operation_id:
                    raise ContractValidationError(
                        "checkpoint event operation identity 非法"
                    )
                execution_records.append((index, operation_id))
                if all(
                    event_payload.get(field) == expected
                    for field, expected in (
                        ("workflow_version", combination["workflow_version"]),
                        (
                            "operation_instance_id",
                            manifest["operation_instance_id"],
                        ),
                        ("step_index", manifest["step_index"]),
                    )
                ):
                    matched.append((index, event_row["event_id"]))

            matched.sort()
            execution_records.sort()
            current_index = manifest["checkpoint_index"]
            if (
                isinstance(current_index, bool)
                or not isinstance(current_index, int)
                or current_index < 1
            ):
                raise ContractValidationError("checkpoint event index 非法")
            execution_indexes = [
                index
                for index, _operation_id in execution_records
                if index <= current_index
            ]
            if execution_indexes != list(range(1, current_index + 1)):
                raise ContractValidationError("execution checkpoint 全局链不连续")
            prior_indexes = [
                index
                for index, operation_id in execution_records
                if index <= current_index
                and operation_id != manifest["operation_instance_id"]
            ]
            local_start = (max(prior_indexes) + 1) if prior_indexes else 1
            expected_indexes = list(range(local_start, current_index + 1))
            if [index for index, _event_id in matched] != expected_indexes:
                raise ContractValidationError("checkpoint event 链不连续")
            record_ids = tuple(event_id for _index, event_id in matched)
            if not record_ids or record_ids[-1] != checkpoint_id:
                raise ContractValidationError("runtime 没有引用 checkpoint 链末端")

            return CheckpointResumeContext(
                manifest=deepcopy(manifest),
                record_ids=record_ids,
                combination_id=combination["combination_id"],
                combination_version=combination["version"],
                checkpoint_id=checkpoint_id,
            )

    def list_candidates(self, task_id: str, batch_id: str) -> List[Dict[str, Any]]:
        """按顺序读取候选轻量索引，不复制大段假设正文。"""

        store = self._require_enabled()
        rows = store.fetchall(
            """
            SELECT c.* FROM hypothesis_candidates c
            JOIN hypothesis_batches b ON b.batch_id=c.batch_id
            WHERE b.task_id=? AND c.batch_id=? ORDER BY c.candidate_index
            """,
            (task_id, batch_id),
        )
        return list_projection_rows(rows, candidate_row)

    def list_combinations(self, task_id: str, batch_id: str) -> List[Dict[str, Any]]:
        """按原始组合顺序读取本批所有独立运行状态。"""

        store = self._require_enabled()
        rows = store.fetchall(
            "SELECT * FROM combinations WHERE task_id=? AND batch_id=? ORDER BY combination_index",
            (task_id, batch_id),
        )
        return list_projection_rows(rows, combination_row)

    # ==================== Investigation v2 事实写入 ====================

    @staticmethod
    def _remove_investigation_audit_fields(
        value: Dict[str, Any],
    ) -> Dict[str, Any]:
        """从公开 Investigation 正文移除只属于事件包装层的审计字段。

        输入是已经确认的可变 payload 副本；动作删除 ``source_refs``（如果调用方
        将完整写入请求放在一个对象中），返回同一个副本。该字段不会进入科学事实，
        其他字段保持原样；输入不满足 object 合同时由调用方先行拒绝。
        """

        if "source_refs" in value:
            value.pop("source_refs")
        return value

    @staticmethod
    def _merge_investigation_keyword_fields(
        value: Dict[str, Any],
        fields: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """把公开方法的关键字字段并入 Investigation payload。

        输入是已移除审计字段的 payload 副本和关键字映射；动作忽略 ``None``，对重复
        且不一致的值抛出合同错误，对一致值执行幂等合并，返回可供后续 v8 合同校验的
        payload 副本。该方法不写数据库。
        """

        for key, item in fields.items():
            if item is None:
                continue
            if key in value and value[key] != item:
                raise ContractValidationError(f"Investigation payload.{key} 重复且不一致")
            value[key] = item
        return value

    @staticmethod
    def _coerce_investigation_payload(
        payload: Mapping[str, Any] | None,
        fields: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """合并公开 payload 与关键字字段，并把 source_refs 留在事件包装层。"""

        if payload is not None and not isinstance(payload, Mapping):
            raise ContractValidationError("Investigation payload 必须是 JSON object")
        value = WorkingMemoryService._remove_investigation_audit_fields(
            dict(payload or {})
        )
        return WorkingMemoryService._merge_investigation_keyword_fields(value, fields)

    @staticmethod
    def _set_investigation_protocol_version(value: Dict[str, Any]) -> Dict[str, Any]:
        """固定新 Investigation 写入的协议版本，并拒绝调用方静默漂移。

        输入是公开写入方法拼合后的 payload 字典；动作检查调用方若显式给出协议号
        必须等于当前 v8，再写入固定值。返回同一个字典供后续合同校验使用；版本错误
        立即失败，不追加事件，也不改变任何当前索引。
        """

        if (
            "investigation_protocol_version" in value
            and value["investigation_protocol_version"]
            != INVESTIGATION_PROTOCOL_VERSION
        ):
            raise ContractValidationError("Investigation v8 写入必须使用协议版本 8")
        value["investigation_protocol_version"] = INVESTIGATION_PROTOCOL_VERSION
        return value

    @staticmethod
    def _adapt_waiting_resume_checkpoint(value: Dict[str, Any]) -> Dict[str, Any]:
        """把当前 sidecar 调用面的 resumable 字段适配为扁平等待事件正文。

        输入是 waiting 写入请求的可变副本；动作只消费 ``resumable`` 和分组
        ``resume_checkpoint``，从 checkpoint 提取作业、turn 和 operation 身份，随后
        删除这两个包装字段。Working Memory 不复制 checkpoint 正文，只保存
        ``sidecar_record_digest``；扁平字段仍会经过正式 waiting 合同和 Started/session
        逐值核对。分组缺失、身份重复或提交 Started 引用不一致时抛出合同错误，不写入
        任何事件。
        """

        resumable = value.pop("resumable", None)
        if resumable is not None and (
            not isinstance(resumable, bool) or resumable is not True
        ):
            raise ContractValidationError(
                "InvestigationTurnWaiting.resumable 只能明确为 true"
            )
        checkpoint = value.pop("resume_checkpoint", None)
        if checkpoint is None:
            return value
        if not isinstance(checkpoint, Mapping):
            raise ContractValidationError(
                "InvestigationTurnWaiting.resume_checkpoint 必须是对象"
            )
        expected_groups = {
            "subject",
            "turn_chain",
            "execution",
            "provider",
            "capability",
        }
        if set(checkpoint) != expected_groups:
            raise ContractValidationError(
                "InvestigationTurnWaiting.resume_checkpoint 分组不完整"
            )
        groups = {
            name: checkpoint[name]
            for name in expected_groups
        }
        if any(not isinstance(group, Mapping) for group in groups.values()):
            raise ContractValidationError(
                "InvestigationTurnWaiting.resume_checkpoint 分组必须是对象"
            )
        allowed_fields = {
            "subject": {
                "task_id",
                "batch_id",
                "combination_id",
                "session_id",
                "stage",
                "scope_id",
                "codex_thread_id",
                "logical_operation_instance_id",
                "attempt_operation_instance_id",
                "attempt_number",
                "repair_request_number",
                "workspace_ref_digest",
                "repair_request_event_ref",
                "expected_result_version",
            },
            "turn_chain": {
                "submit_turn_id",
                "submit_turn_started_event_ref",
                "active_continuation_turn_id",
                "active_continuation_turn_started_event_ref",
            },
            "execution": {
                "job_id",
                "submission_operation_id",
                "submission_key",
                "submission_request_digest",
                "job_family",
                "action_id",
                "backend_id",
                "submission_operation_fingerprint",
            },
            "provider": {
                "provider_id",
                "codex_thread_id",
                "codex_turn_id",
                "turn_request_digest",
                "thread_policy_digest",
                "turn_count_before",
            },
            "capability": {
                "scope_id",
                "workspace_ref_digest",
                "scope_credential_generation_digest",
            },
        }
        for group_name, allowed in allowed_fields.items():
            if set(groups[group_name]) - allowed:
                raise ContractValidationError(
                    f"InvestigationTurnWaiting.resume_checkpoint.{group_name} 含未知字段"
                )
        subject = groups["subject"]
        turn_chain = groups["turn_chain"]
        execution = groups["execution"]
        provider = groups["provider"]
        capability = groups["capability"]
        required_fields = {
            "subject": {
                "task_id",
                "batch_id",
                "combination_id",
                "session_id",
                "stage",
                "scope_id",
                "codex_thread_id",
                "logical_operation_instance_id",
                "attempt_operation_instance_id",
                "attempt_number",
                "workspace_ref_digest",
            },
            "turn_chain": {"submit_turn_id", "submit_turn_started_event_ref"},
            "execution": {
                "job_id",
                "submission_operation_id",
                "submission_key",
                "submission_request_digest",
                "job_family",
            },
            "provider": {
                "provider_id",
                "codex_thread_id",
                "codex_turn_id",
                "turn_request_digest",
                "thread_policy_digest",
                "turn_count_before",
            },
            "capability": {
                "scope_id",
                "workspace_ref_digest",
                "scope_credential_generation_digest",
            },
        }
        for group_name, required in required_fields.items():
            missing = required - set(groups[group_name])
            if missing:
                raise ContractValidationError(
                    "InvestigationTurnWaiting.resume_checkpoint.%s 缺少字段: %s"
                    % (group_name, ", ".join(sorted(missing)))
                )
        if turn_chain["submit_turn_started_event_ref"] != value.get(
            "turn_started_event_ref"
        ):
            raise ContractValidationError(
                "resume_checkpoint 提交 Started 引用与 waiting 引用不一致"
            )
        if subject["codex_thread_id"] != provider["codex_thread_id"]:
            raise ContractValidationError("resume_checkpoint thread identity 不一致")
        if subject["scope_id"] != capability["scope_id"]:
            raise ContractValidationError("resume_checkpoint scope identity 不一致")
        if subject["workspace_ref_digest"] != capability["workspace_ref_digest"]:
            raise ContractValidationError("resume_checkpoint workspace 摘要不一致")
        if turn_chain["submit_turn_id"] != provider["codex_turn_id"]:
            raise ContractValidationError("resume_checkpoint turn identity 不一致")
        flattened = {
            "task_id": subject["task_id"],
            "batch_id": subject["batch_id"],
            "combination_id": subject["combination_id"],
            "session_id": subject["session_id"],
            "kind": subject["stage"],
            "scope_id": subject["scope_id"],
            "codex_thread_id": subject["codex_thread_id"],
            "workspace_ref_digest": subject["workspace_ref_digest"],
            "turn_id": provider["codex_turn_id"],
            "attempt_number": subject["attempt_number"],
            "logical_operation_instance_id": subject[
                "logical_operation_instance_id"
            ],
            "attempt_operation_instance_id": subject[
                "attempt_operation_instance_id"
            ],
            "job_id": execution["job_id"],
            "submission_operation_id": execution["submission_operation_id"],
            "submission_key": execution["submission_key"],
            "submission_request_digest": execution["submission_request_digest"],
        }
        for key, expected in flattened.items():
            if key in value and value[key] != expected:
                raise ContractValidationError(
                    f"resume_checkpoint.{key} 与 waiting 字段不一致"
                )
            value[key] = expected
        return value

    def _require_investigation_task(self, conn, task_id: str) -> Any:
        """按 TaskCreated 核对任务当前行，阻止旧任务写入 Investigation 事实。

        输入是已打开的事务连接和 task 身份；动作以不可变 TaskCreated 的架构枚举
        为准，并在 tasks 当前行漂移时先重建可恢复索引。返回重建后的任务行；未知
        任务、创建正文损坏或 legacy 架构写入 v2 时失败，不参考当前环境变量。
        """

        task_created = self._task_created_payload(conn, task_id)
        row = self._ensure_task_projection_matches_created(
            conn, task_id, task_created
        )
        version = validate_stage_architecture_version(
            task_created.get(
                "stage_architecture_version", LEGACY_STAGE_ARCHITECTURE_VERSION
            )
        )
        if version != INVESTIGATION_STAGE_ARCHITECTURE_VERSION:
            raise ContractValidationError(
                "legacy_workflow_v1 任务不能写入 Investigation v2 事实"
            )
        return row

    @staticmethod
    def _assert_investigation_operation_available(
        conn,
        task_id: str,
        operation_instance_id: str | None,
    ) -> None:
        """保证一个 Investigation operation 只占用一种终态事实。"""

        if operation_instance_id is None:
            return
        placeholders = ",".join("?" for _ in INVESTIGATION_EVENT_TYPES)
        rows = conn.execute(
            f"SELECT event_id,event_type FROM memory_events WHERE task_id=? "
            f"AND event_type IN ({placeholders}) "
            "AND json_extract(payload_json,'$.operation_instance_id')=?",
            (task_id, *sorted(INVESTIGATION_EVENT_TYPES), operation_instance_id),
        ).fetchall()
        if rows:
            raise OperationConflict(
                f"operation_instance_id={operation_instance_id} 已占用 Investigation 事实"
            )
        revision = conn.execute(
            "SELECT revision_id FROM revisions WHERE task_id=? "
            "AND json_extract(revised_result_json,'$.operation_instance_id')=? LIMIT 1",
            (task_id, operation_instance_id),
        ).fetchone()
        if revision is not None:
            raise OperationConflict(
                f"operation_instance_id={operation_instance_id} 已占用 workflow revision"
            )

    @staticmethod
    def _investigation_session_identity_event_refs(
        conn,
        *,
        task_id: str,
        batch_id: str,
        combination_id: str,
        session_event_id: str,
    ) -> list[str]:
        """返回 session 及其首次身份绑定事件，保证重建时先恢复身份再读科学事实。

        输入是同一事务中的 task、batch、组合和 SessionStarted 事件 ID；动作只查询
        已存在的 ThreadBound 事件并返回稳定来源列表，不写数据库、不读取 transcript。
        返回列表至少包含 session_event_id；若 thread/workspace/scope 已绑定，也包含绑定
        事件。查询失败或事件身份不属于当前组合时由上层来源校验抛出合同错误。
        """

        refs = [session_event_id]
        bound = conn.execute(
            "SELECT event_id FROM memory_events WHERE task_id=? "
            "AND event_type='investigation_session_thread_bound' "
            "AND json_extract(payload_json,'$.batch_id')=? "
            "AND json_extract(payload_json,'$.combination_id')=? LIMIT 1",
            (task_id, batch_id, combination_id),
        ).fetchone()
        if bound is not None:
            refs.append(bound["event_id"])
        return refs

    def _insert_investigation_event(
        self,
        conn,
        *,
        task_id: str,
        event_key: str,
        event_type: str,
        round_id: int,
        payload: Mapping[str, Any],
        source_refs: Sequence[str],
        projection,
    ) -> str:
        """在同一事务中追加 Investigation 事件并更新当前索引。

        输入是已经通过公开方法校验的事件正文、稳定键、来源和索引回调；动作委托
        给内部事实写入器完成幂等、CAS、来源存在性、operation 占用及索引更新；返回
        原事件或新事件 ID。任何校验/索引失败都会由外层事务回滚，不留下半条事实。
        """

        writer = _InvestigationFactWriter(self)
        return writer.append_event(
            conn,
            task_id=task_id,
            event_key=event_key,
            event_type=event_type,
            round_id=round_id,
            payload=payload,
            source_refs=source_refs,
            update_current_index=projection,
        )

    def get_task_architecture_version(self, task_id: str) -> str:
        """按 TaskCreated 事实读取冻结的架构版本并核对当前任务行。

        输入是 task 身份；动作在写事务中验证 TaskCreated，发现 tasks 架构列或科学
        上下文被篡改时先按事件重建当前行。返回严格枚举；未知任务、损坏事件或无法
        重建时失败，不参考当前环境变量。
        """

        store = self._require_enabled()
        require_nonempty_string(task_id, "task_id")
        with store.immediate_transaction() as conn:
            task_created = self._task_created_payload(conn, task_id)
            self._ensure_task_projection_matches_created(conn, task_id, task_created)
            return validate_stage_architecture_version(
                task_created.get(
                    "stage_architecture_version", LEGACY_STAGE_ARCHITECTURE_VERSION
                )
            )

    @staticmethod
    def _prepare_turn_attempt_payload(
        conn,
        *,
        task_id: str,
        kind: str,
        value: Mapping[str, Any],
    ) -> dict[str, Any]:
        """为一次新 Started 事实补齐 attempt 编号、logical identity 和前次 closure。

        输入是事务内尚未完成合同校验的 initial/repair/review payload；动作只读取同一
        logical turn 的历史 Started/closure，按连续编号填充缺省值，不生成 request digest、
        provider 身份或科学内容。返回新的字典；历史断链、多个活动 attempt 或编号漂移
        不在此处猜测，交由 projection 以 fail-closed 方式拒绝。
        """

        prepared = dict(value)
        previous_rows: list[Any]
        if kind == "initial":
            previous_rows = list(
                conn.execute(
                    "SELECT * FROM investigation_turns WHERE task_id=? AND session_id=? "
                    "ORDER BY attempt_number,created_at,turn_started_event_id",
                    (task_id, prepared.get("session_id")),
                ).fetchall()
            )
        elif kind == "repair":
            previous_rows = list(
                conn.execute(
                    "SELECT * FROM investigation_repair_turns WHERE task_id=? AND request_event_ref=? "
                    "ORDER BY attempt_number,created_at,turn_started_event_id",
                    (task_id, prepared.get("request_event_ref")),
                ).fetchall()
            )
        else:
            session = conn.execute(
                "SELECT turn_count FROM task_review_sessions WHERE task_id=?", (task_id,)
            ).fetchone()
            turn_index = prepared.get("turn_index")
            if turn_index is None and session is not None:
                turn_index = session["turn_count"] + 1
                prepared["turn_index"] = turn_index
            previous_rows = list(
                conn.execute(
                    "SELECT * FROM task_review_turns WHERE task_id=? AND turn_index=? "
                    "ORDER BY attempt_number,created_at,turn_started_event_id",
                    (task_id, turn_index),
                ).fetchall()
            )
        if previous_rows:
            previous = previous_rows[-1]
            prepared.setdefault("attempt_number", int(previous["attempt_number"]) + 1)
            prepared.setdefault(
                "logical_operation_instance_id",
                previous["logical_operation_instance_id"],
            )
            closure = conn.execute(
                "SELECT termination_event_id FROM investigation_turn_attempt_terminations "
                "WHERE task_id=? AND turn_started_event_ref=?",
                (task_id, previous["turn_started_event_id"]),
            ).fetchone()
            if closure is None:
                # waiting 是 replacement 的合法前序收束；它与 terminal closure
                # 一样只通过不可变事件引用连接，不能从当前表的状态猜测。
                closure = conn.execute(
                    "SELECT event_id AS termination_event_id FROM memory_events "
                    "WHERE task_id=? AND event_type='investigation_turn_waiting' "
                    "AND json_extract(payload_json,'$.turn_started_event_ref')=?",
                    (task_id, previous["turn_started_event_id"]),
                ).fetchone()
            if closure is not None:
                prepared.setdefault(
                    "previous_attempt_closure_event_ref", closure["termination_event_id"]
                )
        else:
            prepared.setdefault("attempt_number", 1)
            prepared.setdefault(
                "logical_operation_instance_id",
                prepared.get("operation_instance_id"),
            )
        prepared.setdefault("attempt_status", "in_flight")
        return prepared

    @staticmethod
    def _read_replayed_turn_started_event_id(
        conn,
        *,
        task_id: str,
        kind: str,
        value: Mapping[str, Any],
    ) -> str | None:
        """识别同一 Started 请求的精确重放并拒绝身份漂移。

        输入是事务连接、turn 类型和调用方正文；动作按稳定 turn 身份读取历史 Started
        行，逐项比较 provider、请求、turn 和 operation 字段。完全相同返回原事件 ID，
        其余情况返回空值；同一 turn_id 但正文漂移抛出幂等冲突，不创建新的 attempt。
        数据库读取失败或表结构损坏由外层事务传播，不修改状态。
        """

        if kind == "initial":
            identity_column = "turn_id"
            query = (
                "SELECT * FROM investigation_turns WHERE task_id=? AND session_id=? "
                "AND turn_id=? ORDER BY created_at,turn_started_event_id LIMIT 1"
            )
            params = (task_id, value.get("session_id"), value.get("turn_id"))
            fields = (
                "session_id", "batch_id", "combination_id", "codex_thread_id",
                "scope_id", "turn_id", "operation_instance_id",
            )
        elif kind == "repair":
            identity_column = "turn_id"
            query = (
                "SELECT * FROM investigation_repair_turns WHERE task_id=? "
                "AND request_event_ref=? AND turn_id=? ORDER BY created_at,turn_started_event_id LIMIT 1"
            )
            params = (task_id, value.get("request_event_ref"), value.get("turn_id"))
            fields = (
                "batch_id", "combination_id", "request_event_ref",
                "repair_request_number", "codex_thread_id", "scope_id", "turn_id",
                "request_digest", "operation_instance_id",
            )
        else:
            identity_column = "review_turn_id"
            query = (
                "SELECT * FROM task_review_turns WHERE task_id=? AND review_turn_id=? "
                "ORDER BY created_at,turn_started_event_id LIMIT 1"
            )
            params = (task_id, value.get("review_turn_id"))
            fields = (
                "session_event_ref", "batch_id", "review_thread_id", "review_turn_id",
                "turn_index", "request_digest", "operation_instance_id",
            )
        if not isinstance(value.get(identity_column), str):
            return None
        row = conn.execute(query, params).fetchone()
        if row is None:
            return None
        for field in fields:
            if field in value and value[field] is not None and row[field] != value[field]:
                raise IdempotencyConflict(f"{kind} Started 的 {field} 与既有 turn 不一致")
        if "attempt_number" in value and value["attempt_number"] != row["attempt_number"]:
            raise IdempotencyConflict(f"{kind} Started 的 attempt_number 与既有 turn 不一致")
        if (
            "logical_operation_instance_id" in value
            and value["logical_operation_instance_id"] is not None
            and value["logical_operation_instance_id"] != row["logical_operation_instance_id"]
        ):
            raise IdempotencyConflict(f"{kind} Started 的 logical operation 与既有 turn 不一致")
        return row["turn_started_event_id"]

    def _start_task_review_session_in_transaction(
        self,
        conn,
        *,
        task_id: str,
        value: Mapping[str, Any],
        source_refs: Sequence[str],
    ) -> str:
        """在已有写事务中建立或复用 task ReviewSession。

        输入是事务连接、task 身份、会话正文和来源；动作复用会话合同并追加/重放
        SessionStarted 投影；返回事件 ID，任务、来源或正文错误时抛出合同错误。
        """

        self._require_investigation_task(conn, task_id)
        normalized = validate_task_review_session_payload(dict(value))
        task_created = conn.execute(
            "SELECT event_id FROM memory_events WHERE task_id=? AND event_type='task_created'",
            (task_id,),
        ).fetchone()
        if task_created is None:
            raise ContractValidationError("TaskReviewSession 缺少 TaskCreated 来源")
        refs = list(source_refs) or [task_created["event_id"]]
        if task_created["event_id"] not in refs:
            raise ContractValidationError("TaskReviewSession 必须引用 TaskCreated")
        event_key = event_key_task_review_session(task_id)
        return self._insert_investigation_event(
            conn,
            task_id=task_id,
            event_key=event_key,
            event_type="task_review_session_started",
            round_id=0,
            payload=normalized,
            source_refs=refs,
            projection=lambda event_id: insert_task_review_session_projection(
                conn,
                task_id=task_id,
                event_id=event_id,
                payload=normalized,
                created_at=conn.execute(
                    "SELECT created_at FROM memory_events WHERE event_id=?", (event_id,)
                ).fetchone()[0],
            ),
        )

    def start_task_review_session(
        self,
        task_id: str,
        payload: Mapping[str, Any] | None = None,
        source_refs: Sequence[str] = (),
        **fields: Any,
    ) -> str:
        """为一个 investigation task 冻结唯一的 Review provider 会话。

        输入是 provider/model、sandbox 策略摘要、内部 workspace/CODEX_HOME 引用和
        operation 身份；动作在同一事务追加 TaskReviewSessionStarted 并建立当前会话
        索引，重复同内容幂等、内容漂移冲突。返回事件 ID；legacy task、路径/凭据或
        数据库失败时抛出合同错误，绝不保存 transcript。
        """

        store = self._require_enabled()
        value = self._coerce_investigation_payload(payload, fields)
        value["task_id"] = task_id
        self._set_investigation_protocol_version(value)
        with store.immediate_transaction() as conn:
            return self._start_task_review_session_in_transaction(
                conn,
                task_id=task_id,
                value=value,
                source_refs=source_refs,
            )

    def _start_task_review_turn_in_transaction(
        self,
        conn,
        *,
        task_id: str,
        value: Mapping[str, Any],
        source_refs: Sequence[str],
    ) -> str:
        """在已有写事务中校验并追加 task ReviewTurnStarted。

        输入是事务连接、task 身份、turn 正文和来源；动作读取当前 ReviewSession，
        校验连续编号后追加/重放 turn 投影；返回事件 ID，身份、批次或来源错误时抛出
        合同错误，失败由外层事务回滚。
        """

        self._require_investigation_task(conn, task_id)
        review_session = conn.execute(
            "SELECT * FROM task_review_sessions WHERE task_id=?", (task_id,)
        ).fetchone()
        if review_session is None:
            raise ContractValidationError("Review turn 必须先创建 TaskReviewSession")
        normalized = dict(value)
        normalized.setdefault("session_event_ref", review_session["session_event_id"])
        normalized.setdefault("turn_index", review_session["turn_count"] + 1)
        replay_event_id = self._read_replayed_turn_started_event_id(
            conn, task_id=task_id, kind="review", value=normalized
        )
        if replay_event_id is not None:
            return replay_event_id
        normalized = self._prepare_turn_attempt_payload(
            conn,
            task_id=task_id,
            kind="review",
            value=normalized,
        )
        normalized = validate_task_review_turn_started_payload(normalized)
        if normalized["session_event_ref"] != review_session["session_event_id"]:
            raise ContractValidationError("Review turn 必须引用当前 task ReviewSession")
        batch = conn.execute(
            "SELECT created_event_id FROM hypothesis_batches WHERE task_id=? AND batch_id=?",
            (task_id, normalized["batch_id"]),
        ).fetchone()
        if batch is None:
            raise ContractValidationError("Review turn 引用了未知 batch")
        refs = list(source_refs) or [
            review_session["session_event_id"],
            batch["created_event_id"],
        ]
        self._append_previous_attempt_closure_ref(refs, normalized)
        for required_ref in (review_session["session_event_id"], batch["created_event_id"]):
            if required_ref not in refs:
                raise ContractValidationError("Review turn source_refs 缺少会话或 BatchCreated")
        event_key = event_key_task_review_turn(task_id, normalized["review_turn_id"])
        return self._insert_investigation_event(
            conn,
            task_id=task_id,
            event_key=event_key,
            event_type="task_review_turn_started",
            round_id=normalized["turn_index"],
            payload=normalized,
            source_refs=refs,
            projection=lambda event_id: insert_task_review_turn_projection(
                conn,
                task_id=task_id,
                event_id=event_id,
                payload=normalized,
                created_at=conn.execute(
                    "SELECT created_at FROM memory_events WHERE event_id=?", (event_id,)
                ).fetchone()[0],
            ),
        )

    def start_task_review_turn(
        self,
        task_id: str,
        payload: Mapping[str, Any] | None = None,
        source_refs: Sequence[str] = (),
        **fields: Any,
    ) -> str:
        """记录一次 Review provider turn 已成功开始，并推进 task 会话游标。

        输入是 TaskReviewSession、批次、thread/turn 身份、请求摘要和 operation；动作
        在一个事务中校验冻结 thread 与连续 turn_index，追加启动事实并把会话标为
        turn_in_flight。返回事件 ID；provider 尚未接受、编号跳跃、thread 漂移或事务
        失败时抛出合同错误，不写入 prompt/transcript。
        """

        store = self._require_enabled()
        value = self._coerce_investigation_payload(payload, fields)
        value["task_id"] = task_id
        self._set_investigation_protocol_version(value)
        with store.immediate_transaction() as conn:
            return self._start_task_review_turn_in_transaction(
                conn,
                task_id=task_id,
                value=value,
                source_refs=source_refs,
            )

    def start_task_review_session_and_turn(
        self,
        task_id: str,
        session_payload: Mapping[str, Any] | None,
        turn_payload: Mapping[str, Any] | None,
        session_source_refs: Sequence[str] = (),
        turn_source_refs: Sequence[str] = (),
    ) -> tuple[str, str]:
        """在一个事务中建立或复用 ReviewSession 并追加 ReviewTurnStarted。

        输入是 task 级会话正文、provider 已接受的 turn 正文及各自来源引用；动作复用
        两个独立入口的合同、幂等键和投影回调，但只打开一次事务。返回
        ``(session_event_id, turn_event_id)``；任一校验、投影或故障都会整体回滚，因而
        重试可复用同一身份，不会留下只有 session 没有 turn 的窗口。
        """

        store = self._require_enabled()
        session_value = self._coerce_investigation_payload(session_payload, {})
        session_value["task_id"] = task_id
        self._set_investigation_protocol_version(session_value)
        turn_value = self._coerce_investigation_payload(turn_payload, {})
        turn_value["task_id"] = task_id
        self._set_investigation_protocol_version(turn_value)
        with store.immediate_transaction() as conn:
            session_event = self._start_task_review_session_in_transaction(
                conn,
                task_id=task_id,
                value=session_value,
                source_refs=session_source_refs,
            )
            turn_value.setdefault("session_event_ref", session_event)
            turn_event = self._start_task_review_turn_in_transaction(
                conn,
                task_id=task_id,
                value=turn_value,
                source_refs=turn_source_refs,
            )
            return session_event, turn_event

    def record_investigation_evidence_verification(
        self,
        task_id: str,
        evidence: VerifiedInvestigationEvidence,
        source_refs: Sequence[str] = (),
    ) -> str:
        """登记上层权威适配器生成的安全证据验证事件。

        输入必须是 VerifiedInvestigationEvidence 对象；动作只保存其稳定 ID、终态
        job/operation/artifact 摘要，并在同一事务更新索引。返回验证事件 ID；普通
        字符串、未绑定当前 task/session 的 mapping、receipt 原文、token、路径或重复
        对象都会失败，Working Memory 不访问 MiniChem 或秘密存储。
        """

        store = self._require_enabled()
        if not isinstance(evidence, VerifiedInvestigationEvidence):
            raise ContractValidationError(
                "Result 证据必须是 VerifiedInvestigationEvidence，而不是普通字符串"
            )
        value = evidence.to_payload()
        if value["task_id"] != task_id:
            raise ContractValidationError("证据摘要 task 身份不一致")
        with store.immediate_transaction() as conn:
            self._require_investigation_task(conn, task_id)
            event_key = event_key_investigation_evidence_verification(
                task_id,
                value["batch_id"],
                value["combination_id"],
                value["verification_object_id"],
            )
            replayed_event = self._read_replayed_evidence_event_id(
                conn,
                task_id=task_id,
                event_key=event_key,
                value=value,
                source_refs=source_refs,
            )
            if replayed_event is not None:
                return replayed_event
            session = conn.execute(
                "SELECT * FROM investigation_sessions "
                "WHERE task_id=? AND batch_id=? AND combination_id=?",
                (task_id, value["batch_id"], value["combination_id"]),
            ).fetchone()
            if session is None:
                raise ContractValidationError("证据验证必须绑定已创建 Investigation session")
            identity_refs, turn_ref = self._prepare_evidence_current_turn_claim(
                conn,
                task_id=task_id,
                value=value,
                session=session,
            )
            refs = self._validate_evidence_source_refs(
                source_refs=source_refs,
                identity_refs=identity_refs,
                session_event_id=session["session_event_id"],
                turn_ref=turn_ref,
            )
            return self._insert_investigation_event(
                conn,
                task_id=task_id,
                event_key=event_key,
                event_type="investigation_evidence_verified",
                round_id=1,
                payload=value,
                source_refs=refs,
                projection=lambda event_id: insert_investigation_evidence_verification_projection(
                    conn,
                    task_id=task_id,
                    event_id=event_id,
                    payload=value,
                    created_at=conn.execute(
                        "SELECT created_at FROM memory_events WHERE event_id=?", (event_id,)
                    ).fetchone()[0],
                ),
            )

    def _prepare_evidence_current_turn_claim(
        self,
        conn,
        *,
        task_id: str,
        value: Mapping[str, Any],
        session: Any,
    ) -> tuple[list[str], str]:
        """读取当前 in-flight turn 及其 session 身份绑定来源。

        输入是事务连接、task、Evidence 正文和已找到的 session；动作核对 thread/scope，
        读取身份来源，再只从 initial/repair 的 in-flight 行选择当前 turn。返回来源列表
        与 turn event ref；缺失、终态或 turn 漂移时抛出合同错误，不写数据库。
        """

        if (
            session["codex_thread_id"] != value["codex_thread_id"]
            or session["scope_id"] != value["scope_id"]
        ):
            raise ContractValidationError("证据摘要与 session 身份不一致")
        identity_refs = self._investigation_session_identity_event_refs(
            conn,
            task_id=task_id,
            batch_id=value["batch_id"],
            combination_id=value["combination_id"],
            session_event_id=session["session_event_id"],
        )
        if session["pending_repair_event_id"] is None:
            turn_ref = session["current_turn_event_id"]
            turn = (
                conn.execute(
                    "SELECT * FROM investigation_turns WHERE task_id=? "
                    "AND turn_started_event_id=? AND attempt_status='in_flight'",
                    (task_id, turn_ref),
                ).fetchone()
                if isinstance(turn_ref, str)
                else None
            )
        else:
            turn = conn.execute(
                "SELECT * FROM investigation_repair_turns WHERE task_id=? "
                "AND request_event_ref=? AND attempt_status='in_flight' "
                "ORDER BY attempt_number DESC,created_at DESC,turn_started_event_id DESC LIMIT 1",
                (task_id, session["pending_repair_event_id"]),
            ).fetchone()
            turn_ref = None if turn is None else turn["turn_started_event_id"]
        if turn is None or turn["turn_id"] != value["turn_id"]:
            raise ContractValidationError(
                "证据必须使用当前已成功启动的 initial 或 repair turn；"
                "没有当前 turn 时需先处理 pending repair"
            )
        return identity_refs, turn_ref

    @staticmethod
    def _validate_evidence_source_refs(
        *,
        source_refs: Sequence[str],
        identity_refs: Sequence[str],
        session_event_id: str,
        turn_ref: str,
    ) -> list[str]:
        """确认 Evidence 来源覆盖 session 身份和当前 turn claim。

        输入是调用方来源、session 身份来源及当前 turn event ref；动作选择显式来源或
        默认完整来源并核对三类必需引用。返回可写入事件的列表；来源遗漏时抛出合同错误，
        不访问数据库。
        """

        refs = list(source_refs) or [*identity_refs, turn_ref]
        if session_event_id not in refs:
            raise ContractValidationError("证据验证 source_refs 必须引用 session")
        if not set(identity_refs).issubset(refs):
            raise ContractValidationError("证据验证 source_refs 必须引用 session 身份绑定事件")
        if turn_ref not in refs:
            raise ContractValidationError("证据验证 source_refs 必须引用当前 turn claim")
        return refs

    def _read_replayed_evidence_event_id(
        self,
        conn,
        *,
        task_id: str,
        event_key: str,
        value: Mapping[str, Any],
        source_refs: Sequence[str],
    ) -> str | None:
        """读取并核对 Evidence 的精确 replay，而不要求当前 turn 仍 active。

        输入是事务连接、稳定 event key、规范化 Evidence 正文和可选来源；动作只比较
        历史事件类型、正文及调用方显式来源，返回既有 event ID 或 None。内容/来源漂移
        抛出 IdempotencyConflict；未知 key 不写数据库，让调用方继续核对当前 claim。
        """

        existing = conn.execute(
            "SELECT event_id,event_type,payload_json,source_refs_json "
            "FROM memory_events WHERE task_id=? AND event_key=?",
            (task_id, event_key),
        ).fetchone()
        if existing is None:
            return None
        if existing["event_type"] != "investigation_evidence_verified":
            raise IdempotencyConflict("证据验证 event_key 已绑定其他事实类型")
        stored_value = validate_investigation_evidence_payload(
            self._decode_json(existing["payload_json"], field="InvestigationEvidence.payload")
        )
        if canonical_json(stored_value) != canonical_json(value):
            raise IdempotencyConflict("相同证据验证 event_key 的正文发生漂移")
        if source_refs:
            stored_refs = self._decode_json(
                existing["source_refs_json"],
                field="InvestigationEvidence.source_refs_json",
            )
            if sorted(set(source_refs)) != sorted(set(stored_refs)):
                raise IdempotencyConflict("相同证据验证 event_key 的来源发生漂移")
        return existing["event_id"]

    @staticmethod
    def _validate_investigation_session_batch_membership(
        conn,
        *,
        task_id: str,
        batch_id: str,
        combination_id: str,
    ) -> str:
        """核对组合仍属于可调查批次并返回 BatchCreated 事件引用。

        输入是事务连接、task、batch 和 combination 身份；动作只读取批次/组合当前
        行，不写入状态；返回 BatchCreated 事件 ID，未知或已停止组合时抛出合同错误。
        """

        batch_row = conn.execute(
            "SELECT b.created_event_id,c.investigation_phase,b.state "
            "FROM hypothesis_batches b JOIN combinations c "
            "ON c.task_id=b.task_id AND c.batch_id=b.batch_id "
            "WHERE b.task_id=? AND b.batch_id=? AND c.combination_id=?",
            (task_id, batch_id, combination_id),
        ).fetchone()
        valid_investigation_phases = {
            "pending",
            "investigating",
            "repair_pending",
            "validated",
            "marked_for_hypothesis_revision",
        }
        if (
            batch_row is None
            or batch_row["state"] != "running"
            or batch_row["investigation_phase"] not in valid_investigation_phases
        ):
            raise ContractValidationError(
                "Investigation session 必须引用当前 task 已入选的 combination"
            )
        return batch_row["created_event_id"]

    def _insert_investigation_session_thread_bound_in_transaction(
        self,
        conn,
        *,
        task_id: str,
        value: Mapping[str, Any],
        session: Any,
        source_refs: Sequence[str],
    ) -> str:
        """在已有事务中校验并追加/复用 ThreadBound 事件。

        输入是事务连接、task、已存在 session、真实 thread 身份和来源；动作复用
        ThreadBound 合同、稳定键与投影回调；返回事件 ID，身份漂移、来源缺失或投影
        错误时抛出合同错误，失败由外层事务回滚。
        """

        if session["session_id"] != value["session_id"]:
            raise ContractValidationError("session_id 与当前组合身份不一致")
        if session["codex_thread_id"] is not None and (
            session["workspace_ref"] != value["workspace_ref"]
            or session["scope_id"] != value["scope_id"]
        ):
            raise ContractValidationError("session workspace/scope 不能漂移")
        if not isinstance(value.get("codex_thread_id"), str):
            raise ContractValidationError("ThreadBound 必须提供实际 SDK thread")
        if session["codex_thread_id"] is not None and (
            session["codex_thread_id"] != value["codex_thread_id"]
        ):
            raise ContractValidationError("Investigation session 不能漂移到第二个 thread")
        bound_payload = validate_investigation_session_thread_bound_payload(
            {
                "task_id": task_id,
                "batch_id": value["batch_id"],
                "combination_id": value["combination_id"],
                "session_id": value["session_id"],
                "codex_thread_id": value["codex_thread_id"],
                "workspace_ref": value["workspace_ref"],
                "scope_id": value["scope_id"],
                "operation_instance_id": value["operation_instance_id"],
                "investigation_protocol_version": INVESTIGATION_PROTOCOL_VERSION,
            }
        )
        refs = list(source_refs) or [session["session_event_id"]]
        if session["session_event_id"] not in refs:
            raise ContractValidationError("ThreadBound source_refs 必须引用 session")
        return self._insert_investigation_event(
            conn,
            task_id=task_id,
            event_key=event_key_investigation_session_thread_bound(
                task_id, value["batch_id"], value["combination_id"]
            ),
            event_type="investigation_session_thread_bound",
            round_id=0,
            payload=bound_payload,
            source_refs=refs,
            projection=lambda event_id: insert_investigation_session_thread_bound_projection(
                conn,
                task_id=task_id,
                event_id=event_id,
                payload=bound_payload,
                created_at=conn.execute(
                    "SELECT created_at FROM memory_events WHERE event_id=?", (event_id,)
                ).fetchone()[0],
            ),
        )

    def _start_investigation_session_in_transaction(
        self,
        conn,
        *,
        task_id: str,
        value: Mapping[str, Any],
        source_refs: Sequence[str],
    ) -> str:
        """在已有写事务中建立或复用 Investigation session 身份。

        输入是事务连接、task、会话正文和来源；动作核对入选组合后追加 SessionStarted
        或 ThreadBound 事实并更新投影；返回事件 ID，批次、身份或来源错误时抛出合同
        错误，失败由外层事务回滚。
        """

        batch_event_ref = self._validate_investigation_session_batch_membership(
            conn,
            task_id=task_id,
            batch_id=value["batch_id"],
            combination_id=value["combination_id"],
        )
        session = conn.execute(
            "SELECT * FROM investigation_sessions WHERE task_id=? AND batch_id=? "
            "AND combination_id=?",
            (task_id, value["batch_id"], value["combination_id"]),
        ).fetchone()
        if session is not None:
            return self._insert_investigation_session_thread_bound_in_transaction(
                conn,
                task_id=task_id,
                value=value,
                session=session,
                source_refs=source_refs,
            )
        refs = list(source_refs)
        if not refs:
            refs = [batch_event_ref]
        elif batch_event_ref not in refs:
            raise ContractValidationError(
                "Investigation session source_refs 必须引用 BatchCreated 事实"
            )
        if value.get("codex_thread_id") is None:
            raise ContractValidationError("未绑定 thread 的 session 只能由 BatchCreated 原子创建")
        return self._insert_investigation_event(
            conn,
            task_id=task_id,
            event_key=event_key_investigation_session(
                task_id, value["batch_id"], value["combination_id"]
            ),
            event_type="investigation_session_started",
            round_id=0,
            payload=value,
            source_refs=refs,
            projection=lambda event_id: insert_investigation_session_projection(
                conn,
                task_id=task_id,
                event_id=event_id,
                payload=value,
                created_at=conn.execute(
                    "SELECT created_at FROM memory_events WHERE event_id=?", (event_id,)
                ).fetchone()[0],
            ),
        )

    @staticmethod
    def _append_previous_attempt_closure_ref(
        refs: list[str],
        value: Mapping[str, Any],
    ) -> list[str]:
        """把 replacement Started 的 previous closure 加入事件因果来源。

        输入是待写 Started 的来源列表和已规范化正文；动作只在正文声明 closure 时补入
        该事件引用，返回同一列表。无 previous closure 时不变，不访问数据库；来源列表
        因此能让 projection rebuild 先处理 closure，缺失引用则由后续合同拒绝。
        """

        previous_ref = value.get("previous_attempt_closure_event_ref")
        if isinstance(previous_ref, str) and previous_ref and previous_ref not in refs:
            refs.append(previous_ref)
        return refs

    def start_investigation_session(
        self,
        task_id: str,
        payload: Mapping[str, Any] | None = None,
        source_refs: Sequence[str] = (),
        **fields: Any,
    ) -> str:
        """原子创建或绑定一个组合独占的 Investigation session 身份。

        输入是当前 task 已持久化的入选 batch/combination、独立 thread、workspace、
        scope 和 operation 身份；动作核对 BatchCreated 来源、组合阶段和 task 架构。
        v2 BatchCreated 已预建无 thread 的 session 时，本方法追加不可变 ThreadBound
        事件并在首次真实启动时冻结最终 workspace/scope；否则追加兼容的 SessionStarted
        事件。返回事件 ID；未知或未入选组合、跨组合复用身份、协议漂移、legacy task、
        来源缺失和数据库失败均抛出合同或幂等错误，不保存 transcript、token 或私有正文。
        """

        store = self._require_enabled()
        value = self._coerce_investigation_payload(payload, fields)
        value["task_id"] = task_id
        self._set_investigation_protocol_version(value)
        value.setdefault(
            "session_id",
            investigation_session_id(
                task_id,
                value.get("batch_id", ""),
                value.get("combination_id", ""),
            ),
        )
        value = validate_investigation_session_payload(value)
        with store.immediate_transaction() as conn:
            return self._start_investigation_session_in_transaction(
                conn,
                task_id=task_id,
                value=value,
                source_refs=source_refs,
            )

    def migrate_investigation_session_thread(
        self,
        task_id: str,
        payload: Mapping[str, Any] | None = None,
        source_refs: Sequence[str] = (),
        **fields: Any,
    ) -> str:
        """记录显式 provider 迁移并把 session 当前 thread 切到新线程。

        输入是同一 Investigation session 的 source/target thread、workspace/scope、
        waiting 作业的 job/operation/submission/turn checkpoint 和迁移 operation。动作
        在一个事务中核对 session 仍绑定 source、目标线程未被其它组合占用，追加不可变
        ``investigation_session_thread_migrated`` 事实并更新当前索引；重复调用只复用
        正文和来源完全相同的事件。它不创建、取消或修改任何化学作业，也不删除旧线程
        审计。身份漂移、来源缺失、目标冲突或投影重建失败都会抛出合同错误。
        """

        store = self._require_enabled()
        value = self._coerce_investigation_payload(payload, fields)
        value["task_id"] = task_id
        self._set_investigation_protocol_version(value)
        value = validate_investigation_session_thread_migrated_payload(value)
        with store.immediate_transaction() as conn:
            self._require_investigation_task(conn, task_id)
            session = conn.execute(
                "SELECT * FROM investigation_sessions WHERE task_id=? AND batch_id=? "
                "AND combination_id=? AND session_id=?",
                (
                    task_id,
                    value["batch_id"],
                    value["combination_id"],
                    value["session_id"],
                ),
            ).fetchone()
            if session is None:
                raise ContractValidationError(
                    "ThreadMigrated 必须引用已创建的 Investigation session"
                )
            if (
                session["workspace_ref"] != value["workspace_ref"]
                or session["scope_id"] != value["scope_id"]
            ):
                raise ContractValidationError(
                    "ThreadMigrated workspace/scope 与 session 不一致"
                )
            refs = list(source_refs)
            if not refs:
                refs = self._investigation_session_identity_event_refs(
                    conn,
                    task_id=task_id,
                    batch_id=value["batch_id"],
                    combination_id=value["combination_id"],
                    session_event_id=session["session_event_id"],
                )
                current_turn = session["current_turn_event_id"]
                if isinstance(current_turn, str) and current_turn:
                    refs.append(current_turn)
            refs = sorted(set(refs))
            if session["session_event_id"] not in refs:
                raise ContractValidationError(
                    "ThreadMigrated source_refs 必须引用 session"
                )
            event_key = event_key_investigation_session_thread_migrated(
                task_id,
                value["batch_id"],
                value["combination_id"],
                value["source_codex_thread_id"],
                value["target_codex_thread_id"],
            )
            # 先处理精确重放：第二次恢复时 session 已经是 target，不能把它误报成
            # source 漂移；只有正文/来源完全相同的既有事件允许直接返回。
            replay_event = self._existing_event_exact(
                conn,
                event_key=event_key,
                task_id=task_id,
                round_id=0,
                event_type="investigation_session_thread_migrated",
                payload=value,
                source_refs=refs,
            )
            if replay_event is not None:
                return replay_event
            if session["codex_thread_id"] != value["source_codex_thread_id"]:
                raise ContractValidationError(
                    "ThreadMigrated source thread 不是当前 session thread"
                )
            return self._insert_investigation_event(
                conn,
                task_id=task_id,
                event_key=event_key,
                event_type="investigation_session_thread_migrated",
                round_id=0,
                payload=value,
                source_refs=refs,
                projection=lambda event_id: insert_investigation_session_thread_migrated_projection(
                    conn,
                    task_id=task_id,
                    event_id=event_id,
                    payload=value,
                    created_at=conn.execute(
                        "SELECT created_at FROM memory_events WHERE event_id=?",
                        (event_id,),
                    ).fetchone()[0],
                ),
            )

    def begin_investigation_session(
        self,
        task_id: str,
        payload: Mapping[str, Any] | None = None,
        source_refs: Sequence[str] = (),
        **fields: Any,
    ) -> str:
        """兼容旧装配器，调用 canonical start_investigation_session 写入会话。"""

        return self.start_investigation_session(task_id, payload, source_refs, **fields)

    def _start_investigation_turn_in_transaction(
        self,
        conn,
        *,
        task_id: str,
        value: Mapping[str, Any],
        source_refs: Sequence[str],
    ) -> str:
        """在已有写事务中校验并追加首轮 InvestigationTurnStarted。

        输入是事务连接、task、turn 正文和来源；动作读取已绑定 session，核对 thread、
        scope 与首轮顺序后追加/重放 turn 投影；返回事件 ID，身份或状态错误时抛出
        合同错误，失败由外层事务回滚。
        """

        self._require_investigation_task(conn, task_id)
        normalized = dict(value)
        session = conn.execute(
            "SELECT * FROM investigation_sessions WHERE task_id=? AND batch_id=? "
            "AND combination_id=? AND session_id=?",
            (
                task_id,
                normalized.get("batch_id"),
                normalized.get("combination_id"),
                normalized["session_id"],
            ),
        ).fetchone()
        if session is None:
            raise ContractValidationError("Investigation turn 必须引用已创建 session")
        replay_event_id = self._read_replayed_turn_started_event_id(
            conn, task_id=task_id, kind="initial", value=normalized
        )
        if replay_event_id is not None:
            return replay_event_id
        normalized = self._prepare_turn_attempt_payload(
            conn,
            task_id=task_id,
            kind="initial",
            value=normalized,
        )
        normalized = validate_investigation_turn_started_payload(normalized)
        if (
            session["codex_thread_id"] != normalized["codex_thread_id"]
            or session["scope_id"] != normalized["scope_id"]
        ):
            raise ContractValidationError("Investigation turn 与 session 身份不一致")
        if session["latest_result_event_id"] is not None:
            raise ContractValidationError("首轮 Investigation turn 必须在首版 Result 之前启动")
        refs = list(source_refs) or [session["session_event_id"]]
        if session["session_event_id"] not in refs:
            raise ContractValidationError("Investigation turn source_refs 必须引用 session")
        bound = conn.execute(
            "SELECT event_id FROM memory_events WHERE task_id=? AND event_key=?",
            (
                task_id,
                event_key_investigation_session_thread_bound(
                    task_id, normalized["batch_id"], normalized["combination_id"]
                ),
            ),
        ).fetchone()
        if bound is not None and bound["event_id"] not in refs:
            refs.append(bound["event_id"])
        self._append_previous_attempt_closure_ref(refs, normalized)
        return self._insert_investigation_event(
            conn,
            task_id=task_id,
            event_key=event_key_investigation_turn_started(
                task_id,
                normalized["batch_id"],
                normalized["combination_id"],
                normalized["turn_id"],
            ),
            event_type="investigation_turn_started",
            round_id=1,
            payload=normalized,
            source_refs=sorted(set(refs)),
            projection=lambda event_id: insert_investigation_turn_started_projection(
                conn,
                task_id=task_id,
                event_id=event_id,
                payload=normalized,
                created_at=conn.execute(
                    "SELECT created_at FROM memory_events WHERE event_id=?", (event_id,)
                ).fetchone()[0],
            ),
        )

    def start_investigation_turn(
        self,
        task_id: str,
        payload: Mapping[str, Any] | None = None,
        source_refs: Sequence[str] = (),
        **fields: Any,
    ) -> str:
        """登记首轮 Investigation turn 已成功进入 in-flight。

        输入是已绑定 thread 的 session、batch、组合、scope、turn 和 operation 身份；
        动作在同一事务核对 session 未有 Result/pending/终态，追加首轮 turn 事实并更新
        当前恢复游标。返回 turn 事件 ID；未知 session、thread 漂移、重复 turn、启动
        前调用或数据库失败均抛出合同/幂等错误，不保存 prompt/transcript。
        """

        store = self._require_enabled()
        value = self._coerce_investigation_payload(payload, fields)
        value["task_id"] = task_id
        self._set_investigation_protocol_version(value)
        value.setdefault(
            "session_id",
            investigation_session_id(
                task_id,
                value.get("batch_id", ""),
                value.get("combination_id", ""),
            ),
        )
        with store.immediate_transaction() as conn:
            return self._start_investigation_turn_in_transaction(
                conn,
                task_id=task_id,
                value=value,
                source_refs=source_refs,
            )

    def record_investigation_turn_waiting(
        self,
        task_id: str,
        payload: Mapping[str, Any] | None = None,
        source_refs: Sequence[str] = (),
        **fields: Any,
    ) -> str:
        """登记 initial/repair turn 已提交作业并进入可恢复等待的事实。

        输入是被等待的 Started 事件引用、完整 session/thread/workspace/scope/turn
        身份、logical 与 attempt operation、作业三元身份，以及分别由 submit 参数和
        规范化等待 sidecar 计算的两个 SHA-256 摘要；动作在一个事务中核对 Started、
        当前 active 指针和所有身份后追加不可变 waiting 事件，更新 turn 为
        ``resumable`` 并清除 active pointer，同时保留 Started 和等待历史。相同正文
        重试返回原事件；Review、缺少任一身份/摘要、非当前 attempt、身份篡改或终态
        等情况 fail-closed，不写入 Result/Evidence，也不从缺失字段推断作业事实。

        ``status`` 仅作为旧调用面的兼容校验字段，唯一允许的值是
        ``waiting_investigation_job``，不会进入事件正文；不得使用泛化
        ``checkpoint_digest`` 代替两个明确摘要字段。
        """

        store = self._require_enabled()
        require_nonempty_string(task_id, "task_id")
        keyword_fields = dict(fields)
        status = keyword_fields.pop("status", None)
        value = self._coerce_investigation_payload(payload, keyword_fields)
        if "status" in value:
            status = value.pop("status") if status is None else status
        if status is not None and status != "waiting_investigation_job":
            raise ContractValidationError(
                "InvestigationTurnWaiting.status 只能是 waiting_investigation_job"
            )
        value = self._adapt_waiting_resume_checkpoint(value)
        value["task_id"] = task_id
        self._set_investigation_protocol_version(value)
        turn_started_event_ref = value.get("turn_started_event_ref")
        require_nonempty_string(turn_started_event_ref, "turn_started_event_ref")

        with store.immediate_transaction() as conn:
            self._require_investigation_task(conn, task_id)
            started_event = conn.execute(
                "SELECT event_type,payload_json FROM memory_events "
                "WHERE task_id=? AND event_id=?",
                (task_id, turn_started_event_ref),
            ).fetchone()
            if started_event is None:
                raise ContractValidationError(
                    "InvestigationTurnWaiting 必须引用已存在的 Started 事件"
                )
            if started_event["event_type"] == "investigation_turn_started":
                started = validate_investigation_turn_started_payload(
                    self._decode_json(
                        started_event["payload_json"],
                        field="InvestigationTurnStarted.payload",
                    )
                )
                kind = "initial"
            elif started_event["event_type"] == "investigation_repair_turn_started":
                started = validate_investigation_repair_turn_started_payload(
                    self._decode_json(
                        started_event["payload_json"],
                        field="InvestigationRepairTurnStarted.payload",
                    )
                )
                kind = "repair"
            elif started_event["event_type"] == "task_review_turn_started":
                raise ContractValidationError(
                    "Review turn 禁止写入 InvestigationTurnWaiting"
                )
            else:
                raise ContractValidationError(
                    "InvestigationTurnWaiting 只能引用 initial/repair Started 事件"
                )

            session = conn.execute(
                "SELECT * FROM investigation_sessions WHERE task_id=? AND batch_id=? "
                "AND combination_id=?",
                (
                    task_id,
                    started["batch_id"],
                    started["combination_id"],
                ),
            ).fetchone()
            if session is None:
                raise ContractValidationError(
                    "InvestigationTurnWaiting 缺少对应 Investigation session"
                )

            defaults = {
                "turn_started_event_ref": turn_started_event_ref,
                "kind": kind,
                "batch_id": started["batch_id"],
                "combination_id": started["combination_id"],
                "session_id": session["session_id"],
                "codex_thread_id": started["codex_thread_id"],
                "workspace_ref_digest": hashlib.sha256(
                    canonical_json(session["workspace_ref"]).encode("utf-8")
                ).hexdigest(),
                "scope_id": started["scope_id"],
                "turn_id": started["turn_id"],
                "attempt_number": started["attempt_number"],
                "logical_operation_instance_id": started[
                    "logical_operation_instance_id"
                ],
                "attempt_operation_instance_id": started["operation_instance_id"],
            }
            if kind == "initial" and started["session_id"] != session["session_id"]:
                raise ContractValidationError(
                    "InvestigationTurnWaiting initial session_id 与 session 不一致"
                )
            for key, default in defaults.items():
                if key in value and value[key] != default:
                    raise ContractValidationError(
                        f"InvestigationTurnWaiting.{key} 与 Started/session 身份不一致"
                    )
                value.setdefault(key, default)

            normalized = validate_investigation_turn_waiting_payload(value)
            identity_refs = self._investigation_session_identity_event_refs(
                conn,
                task_id=task_id,
                batch_id=normalized["batch_id"],
                combination_id=normalized["combination_id"],
                session_event_id=session["session_event_id"],
            )
            refs = list(source_refs) or [
                *identity_refs,
                turn_started_event_ref,
            ]
            if kind == "repair":
                refs.append(started["request_event_ref"])
            refs = sorted(set(refs))
            required_refs = {session["session_event_id"], turn_started_event_ref}
            if kind == "repair":
                required_refs.add(started["request_event_ref"])
            if not required_refs.issubset(refs):
                raise ContractValidationError(
                    "InvestigationTurnWaiting source_refs 缺少 Started/session/request 来源"
                )

            event_key = event_key_investigation_turn_waiting(
                task_id, turn_started_event_ref
            )
            replay_event = self._existing_event_exact(
                conn,
                event_key=event_key,
                task_id=task_id,
                round_id=normalized["attempt_number"],
                event_type="investigation_turn_waiting",
                payload=normalized,
                source_refs=refs,
            )
            if replay_event is not None:
                return replay_event

            table = "investigation_turns" if kind == "initial" else "investigation_repair_turns"
            turn_row = conn.execute(
                f"SELECT * FROM {table} WHERE task_id=? AND turn_started_event_id=?",
                (task_id, turn_started_event_ref),
            ).fetchone()
            if turn_row is None:
                raise ContractValidationError(
                    "InvestigationTurnWaiting 缺少 Started 投影行"
                )
            pointer = (
                session["current_turn_event_id"]
                if kind == "initial"
                else session["current_repair_turn_event_id"]
            )
            if pointer != turn_started_event_ref or turn_row["attempt_status"] != "in_flight":
                raise ContractValidationError(
                    "只能把当前 in_flight initial/repair attempt 标记为 waiting"
                )
            return self._insert_investigation_event(
                conn,
                task_id=task_id,
                event_key=event_key,
                event_type="investigation_turn_waiting",
                round_id=normalized["attempt_number"],
                payload=normalized,
                source_refs=refs,
                projection=lambda event_id: insert_investigation_turn_waiting_projection(
                    conn,
                    task_id=task_id,
                    event_id=event_id,
                    payload=normalized,
                    created_at=conn.execute(
                        "SELECT created_at FROM memory_events WHERE event_id=?",
                        (event_id,),
                    ).fetchone()[0],
                ),
            )

    def start_investigation_session_and_turn(
        self,
        task_id: str,
        session_payload: Mapping[str, Any] | None,
        turn_payload: Mapping[str, Any] | None,
        session_source_refs: Sequence[str] = (),
        turn_source_refs: Sequence[str] = (),
    ) -> tuple[str, str]:
        """在一个事务中建立或复用 session/thread 并追加首轮 turn。

        输入是 provider 已接受后得到的 session 身份、thread/scope 和 turn 身份及各自
        来源引用；动作复用 SessionStarted/ThreadBound 与 InvestigationTurnStarted 的
        合同、稳定键、幂等和投影回调，但只打开一次事务。返回
        ``(session_or_thread_bound_event_id, turn_event_id)``；任一校验、投影或故障都
        整体回滚，重试时沿用同一 session/thread/turn 身份，不留下半条生命周期事实。
        """

        store = self._require_enabled()
        session_value = self._coerce_investigation_payload(session_payload, {})
        session_value["task_id"] = task_id
        self._set_investigation_protocol_version(session_value)
        session_value.setdefault(
            "session_id",
            investigation_session_id(
                task_id,
                session_value.get("batch_id", ""),
                session_value.get("combination_id", ""),
            ),
        )
        session_value = validate_investigation_session_payload(session_value)
        turn_value = self._coerce_investigation_payload(turn_payload, {})
        turn_value["task_id"] = task_id
        self._set_investigation_protocol_version(turn_value)
        turn_value.setdefault("session_id", session_value["session_id"])
        with store.immediate_transaction() as conn:
            session_event = self._start_investigation_session_in_transaction(
                conn,
                task_id=task_id,
                value=session_value,
                source_refs=session_source_refs,
            )
            turn_event = self._start_investigation_turn_in_transaction(
                conn,
                task_id=task_id,
                value=turn_value,
                source_refs=turn_source_refs,
            )
            return session_event, turn_event

    def record_investigation_turn_attempt_termination(
        self,
        task_id: str,
        *,
        turn_started_event_ref: str,
        terminal_status: str,
    ) -> str:
        """明确终止一条当前 in-flight provider attempt，并保留其 Started 历史。

        输入是 task 身份、被引用的 initial/repair/review Started 事件和 ``failed`` 或
        ``interrupted`` 状态；动作在一个事务中读取 Started 正文，逐值冻结 kind、thread/
        turn、logical operation、per-attempt operation 与 attempt 编号，追加唯一 closure
        事件并更新对应活动指针。相同内容幂等；未知、已消费、非当前、身份漂移、缺失
        Started 或数据库失败均 fail-closed，不从缺 Result 推断终止，也不删除旧 Start。
        """

        store = self._require_enabled()
        require_nonempty_string(task_id, "task_id")
        require_nonempty_string(turn_started_event_ref, "turn_started_event_ref")
        if terminal_status not in {"failed", "interrupted"}:
            raise ContractValidationError("terminal_status 只能是 failed 或 interrupted")
        with store.immediate_transaction() as conn:
            self._require_investigation_task(conn, task_id)
            event = conn.execute(
                "SELECT event_type,payload_json FROM memory_events "
                "WHERE task_id=? AND event_id=?",
                (task_id, turn_started_event_ref),
            ).fetchone()
            if event is None:
                raise ContractValidationError("turn termination 引用了未知 Started 事件")
            raw = self._decode_json(
                event["payload_json"], field="InvestigationTurnStarted.payload"
            )
            if event["event_type"] == "investigation_turn_started":
                started = validate_investigation_turn_started_payload(raw)
                kind = "initial"
                closure = {
                    "task_id": task_id,
                    "turn_started_event_ref": turn_started_event_ref,
                    "terminal_status": terminal_status,
                    "kind": kind,
                    "batch_id": started["batch_id"],
                    "combination_id": started["combination_id"],
                    "turn_id": started["turn_id"],
                    "attempt_number": started["attempt_number"],
                    "logical_operation_instance_id": started[
                        "logical_operation_instance_id"
                    ],
                    "attempt_operation_instance_id": started["operation_instance_id"],
                    "previous_attempt_closure_event_ref": started.get(
                        "previous_attempt_closure_event_ref"
                    ),
                    "session_id": started["session_id"],
                    "codex_thread_id": started["codex_thread_id"],
                    "scope_id": started["scope_id"],
                    "investigation_protocol_version": INVESTIGATION_PROTOCOL_VERSION,
                }
            elif event["event_type"] == "investigation_repair_turn_started":
                started = validate_investigation_repair_turn_started_payload(raw)
                kind = "repair"
                closure = {
                    "task_id": task_id,
                    "turn_started_event_ref": turn_started_event_ref,
                    "terminal_status": terminal_status,
                    "kind": kind,
                    "batch_id": started["batch_id"],
                    "combination_id": started["combination_id"],
                    "turn_id": started["turn_id"],
                    "attempt_number": started["attempt_number"],
                    "logical_operation_instance_id": started[
                        "logical_operation_instance_id"
                    ],
                    "attempt_operation_instance_id": started["operation_instance_id"],
                    "previous_attempt_closure_event_ref": started.get(
                        "previous_attempt_closure_event_ref"
                    ),
                    "request_event_ref": started["request_event_ref"],
                    "codex_thread_id": started["codex_thread_id"],
                    "scope_id": started["scope_id"],
                    "investigation_protocol_version": INVESTIGATION_PROTOCOL_VERSION,
                }
            elif event["event_type"] == "task_review_turn_started":
                started = validate_task_review_turn_started_payload(raw)
                kind = "review"
                closure = {
                    "task_id": task_id,
                    "turn_started_event_ref": turn_started_event_ref,
                    "terminal_status": terminal_status,
                    "kind": kind,
                    "batch_id": started["batch_id"],
                    "turn_id": started["review_turn_id"],
                    "review_turn_id": started["review_turn_id"],
                    "review_thread_id": started["review_thread_id"],
                    "codex_thread_id": started["review_thread_id"],
                    "attempt_number": started["attempt_number"],
                    "logical_operation_instance_id": started[
                        "logical_operation_instance_id"
                    ],
                    "attempt_operation_instance_id": started["operation_instance_id"],
                    "previous_attempt_closure_event_ref": started.get(
                        "previous_attempt_closure_event_ref"
                    ),
                    "investigation_protocol_version": INVESTIGATION_PROTOCOL_VERSION,
                }
            else:
                raise ContractValidationError("turn termination 只能引用三类 Started 事件")
            closure = validate_investigation_turn_attempt_termination_payload(closure)
            return self._insert_investigation_event(
                conn,
                task_id=task_id,
                event_key=event_key_investigation_turn_attempt_termination(
                    task_id, turn_started_event_ref
                ),
                event_type="investigation_turn_attempt_terminated",
                round_id=closure["attempt_number"],
                payload=closure,
                source_refs=[turn_started_event_ref],
                projection=lambda event_id: insert_investigation_turn_attempt_termination_projection(
                    conn,
                    task_id=task_id,
                    event_id=event_id,
                    payload=closure,
                    created_at=conn.execute(
                        "SELECT created_at FROM memory_events WHERE event_id=?", (event_id,)
                    ).fetchone()[0],
                ),
            )

    def _validate_legacy_waiting_resume_authorization_task(
        self,
        conn: Any,
        task_id: str,
    ) -> None:
        """核对授权事件所属 task 仍可追加 Investigation 事实。

        输入是事务连接和已校验的 task 身份；动作重新读取 task、TaskClosed 和失败
        事实，拒绝终态或失败任务；返回 ``None``，任何状态不一致都会抛出合同错误，
        不修改数据库。
        """

        task = self._require_investigation_task(conn, task_id)
        if task["status"] in self._TASK_TERMINAL_STATES:
            raise ContractValidationError(
                "终态 task 不能追加 legacy waiting resume authorization"
            )
        if conn.execute(
            "SELECT 1 FROM investigation_task_terminals WHERE task_id=?",
            (task_id,),
        ).fetchone() is not None:
            raise ContractValidationError(
                "已有 Investigation task terminal 不能追加 legacy waiting resume authorization"
            )
        if conn.execute(
            "SELECT 1 FROM memory_events WHERE task_id=? "
            "AND event_type='investigation_task_failed' LIMIT 1",
            (task_id,),
        ).fetchone() is not None:
            raise ContractValidationError(
                "已有 Investigation task failure 不能追加 legacy waiting resume authorization"
            )

    def _read_legacy_waiting_resume_authorization_session(
        self,
        conn: Any,
        *,
        task_id: str,
        value: Mapping[str, Any],
        source_refs: Sequence[str],
    ) -> Any:
        """读取并核对授权事件引用的 Investigation session。

        输入是事务连接、task、已通过正文合同的授权映射和来源引用；动作逐值比较
        session、thread、scope 以及逻辑 workspace 摘要，并确认 session identity
        event 在来源中；返回数据库 session 行，缺失或漂移时抛出合同错误。
        """

        session = conn.execute(
            "SELECT * FROM investigation_sessions WHERE task_id=? AND batch_id=? "
            "AND combination_id=? AND session_id=?",
            (
                task_id,
                value["batch_id"],
                value["combination_id"],
                value["session_id"],
            ),
        ).fetchone()
        if session is None:
            raise ContractValidationError(
                "legacy waiting authorization 缺少对应 Investigation session"
            )
        expected_session = {
            "session_id": session["session_id"],
            "session_event_ref": session["session_event_id"],
            "codex_thread_id": session["codex_thread_id"],
            "scope_id": session["scope_id"],
            "workspace_ref_digest": hashlib.sha256(
                canonical_json(session["workspace_ref"]).encode("utf-8")
            ).hexdigest(),
        }
        for field_name, expected in expected_session.items():
            if value[field_name] != expected:
                raise ContractValidationError(
                    f"legacy waiting authorization.{field_name} 与 session 不一致"
                )
        if value["session_event_ref"] not in source_refs:
            raise ContractValidationError(
                "legacy waiting authorization source_refs 必须包含 session identity event"
            )
        return session

    def _read_legacy_waiting_resume_authorization_started(
        self,
        conn: Any,
        *,
        task_id: str,
        value: Mapping[str, Any],
        source_refs: Sequence[str],
    ) -> Mapping[str, Any]:
        """读取并核对授权事件引用的 initial/repair Started 事实。

        输入是事务连接、task、授权映射和来源引用；动作按 kind 选择对应 Started
        合同，逐值核对完整 turn/operation identity，并确认 Started event 在来源中；
        返回已验证 Started 映射，事件类型或身份漂移时抛出合同错误。
        """

        started_row = conn.execute(
            "SELECT event_type,payload_json,source_refs_json FROM memory_events "
            "WHERE task_id=? AND event_id=?",
            (task_id, value["turn_started_event_ref"]),
        ).fetchone()
        if started_row is None:
            raise ContractValidationError(
                "legacy waiting authorization 缺少 Started 事件"
            )
        if value["kind"] == "initial":
            if started_row["event_type"] != "investigation_turn_started":
                raise ContractValidationError(
                    "legacy waiting authorization initial 必须引用 InvestigationTurnStarted"
                )
            started = validate_investigation_turn_started_payload(
                self._decode_json(
                    started_row["payload_json"],
                    field="InvestigationTurnStarted.payload",
                )
            )
        else:
            if started_row["event_type"] != "investigation_repair_turn_started":
                raise ContractValidationError(
                    "legacy waiting authorization repair 必须引用 RepairTurnStarted"
                )
            started = validate_investigation_repair_turn_started_payload(
                self._decode_json(
                    started_row["payload_json"],
                    field="InvestigationRepairTurnStarted.payload",
                )
            )
        started_expected = {
            "task_id": task_id,
            "batch_id": value["batch_id"],
            "combination_id": value["combination_id"],
            "codex_thread_id": value["codex_thread_id"],
            "scope_id": value["scope_id"],
            "turn_id": value["turn_id"],
            "attempt_number": value["attempt_number"],
            "logical_operation_instance_id": value["logical_operation_instance_id"],
        }
        if value["kind"] == "initial":
            started_expected["session_id"] = value["session_id"]
        elif started.get("request_event_ref") != value.get("repair_request_event_ref"):
            raise ContractValidationError(
                "legacy waiting authorization repair request identity 不一致"
            )
        for field_name, expected in started_expected.items():
            if started.get(field_name) != expected:
                raise ContractValidationError(
                    f"legacy waiting authorization Started.{field_name} 身份不一致"
                )
        if value["attempt_operation_instance_id"] != started["operation_instance_id"]:
            raise ContractValidationError(
                "legacy waiting authorization.attempt_operation_instance_id 与 Started.operation_instance_id 不一致"
            )
        if value["turn_started_event_ref"] not in source_refs:
            raise ContractValidationError(
                "legacy waiting authorization source_refs 必须包含 Started event"
            )
        return started

    def _read_legacy_waiting_resume_authorization_closure(
        self,
        conn: Any,
        *,
        task_id: str,
        value: Mapping[str, Any],
        source_refs: Sequence[str],
    ) -> Mapping[str, Any]:
        """读取并核对授权事件引用的 terminal_interrupted closure。

        输入是事务连接、task、授权映射和来源引用；动作只接受真实终止事件，逐值
        核对 Started、session、thread、scope 和 attempt identity，并确认 closure
        自身引用了 Started；返回已验证 closure 映射，failed、foreign 或漂移引用
        均抛出合同错误。
        """

        closure_row = conn.execute(
            "SELECT event_type,payload_json,source_refs_json FROM memory_events "
            "WHERE task_id=? AND event_id=?",
            (task_id, value["terminal_event_ref"]),
        ).fetchone()
        if (
            closure_row is None
            or closure_row["event_type"] != "investigation_turn_attempt_terminated"
        ):
            raise ContractValidationError(
                "legacy waiting authorization 必须引用真实 turn closure"
            )
        closure = validate_investigation_turn_attempt_termination_payload(
            self._decode_json(
                closure_row["payload_json"],
                field="InvestigationTurnAttemptTerminated.payload",
            )
        )
        if closure["terminal_status"] != "interrupted":
            raise ContractValidationError(
                "legacy waiting authorization 只能绑定 terminal_interrupted closure"
            )
        if closure["turn_started_event_ref"] != value["turn_started_event_ref"]:
            raise ContractValidationError(
                "legacy waiting authorization closure 与 Started 引用不一致"
            )
        for field_name in (
            "task_id",
            "batch_id",
            "combination_id",
            "turn_id",
            "attempt_number",
            "logical_operation_instance_id",
        ):
            if closure.get(field_name) != value.get(field_name):
                raise ContractValidationError(
                    f"legacy waiting authorization closure.{field_name} 身份不一致"
                )
        if closure.get("attempt_operation_instance_id") != value["attempt_operation_instance_id"]:
            raise ContractValidationError(
                "legacy waiting authorization closure.attempt_operation_instance_id 身份不一致"
            )
        if value["kind"] == "initial" and closure.get("session_id") != value["session_id"]:
            raise ContractValidationError(
                "legacy waiting authorization closure.session_id 身份不一致"
            )
        if (
            closure.get("codex_thread_id") != value["codex_thread_id"]
            or closure.get("scope_id") != value["scope_id"]
        ):
            raise ContractValidationError(
                "legacy waiting authorization closure provider/scope 身份不一致"
            )
        closure_refs = self._decode_json(
            closure_row["source_refs_json"],
            field="InvestigationTurnAttemptTerminated.source_refs",
        )
        if value["turn_started_event_ref"] not in closure_refs:
            raise ContractValidationError(
                "turn closure source_refs 缺少 Started event"
            )
        if value["terminal_event_ref"] not in source_refs:
            raise ContractValidationError(
                "legacy waiting authorization source_refs 必须包含 closure event"
            )
        return closure

    @staticmethod
    def _normalize_legacy_waiting_resume_authorization_source_refs(
        value: Mapping[str, Any],
        source_refs: Sequence[str],
    ) -> list[str]:
        """规范化授权事件来源引用并确认 session、Started、closure 齐全。

        输入是已通过事实核对的授权映射和来源序列；动作去重并排序稳定引用，返回
        可写入事件的列表；缺少任一必需事实时抛出合同错误，不修改数据库。
        """

        refs = sorted(set(source_refs))
        required_refs = {
            value["session_event_ref"],
            value["turn_started_event_ref"],
            value["terminal_event_ref"],
        }
        if not required_refs.issubset(refs):
            raise ContractValidationError(
                "legacy waiting authorization source_refs 不完整"
            )
        return refs

    def _append_legacy_waiting_resume_authorization_event(
        self,
        conn: Any,
        *,
        task_id: str,
        value: Mapping[str, Any],
        source_refs: Sequence[str],
    ) -> str:
        """追加 legacy waiting authorization 不可变事件并保持幂等键。

        输入是事务连接、task、已完成事实核对的正文和规范化来源；动作按原有
        task/batch/combination/sidecar digest 生成稳定 event key，追加无投影事件并
        返回 event ID。相同正文复用既有事件，键或正文冲突由事件存储抛出异常。
        """

        event_key = event_key_investigation_legacy_waiting_resume_authorized(
            task_id,
            value["batch_id"],
            value["combination_id"],
            value["sidecar_record_digest"],
        )
        return self._insert_investigation_event(
            conn,
            task_id=task_id,
            event_key=event_key,
            event_type="investigation_legacy_waiting_resume_authorized",
            round_id=value["attempt_number"],
            payload=value,
            source_refs=source_refs,
            projection=lambda _event_id: None,
        )

    def record_investigation_legacy_waiting_resume_authorization(
        self,
        task_id: str,
        payload: Mapping[str, Any] | None = None,
        source_refs: Sequence[str] = (),
        *,
        resume: bool = False,
        allow_legacy_waiting_resume_migration: bool = False,
        **fields: Any,
    ) -> str:
        """登记旧 waiting sidecar 的显式、一次性 scope 恢复授权。

        输入必须来自明确的 resume 装配，并包含 sidecar 摘要、同一 session、Started
        和 ``terminal_interrupted`` closure 的完整身份；动作依次完成正文校验、事务
        内重读数据库事实和不可变事件追加。旧 tombstone 和 scope token 不在 Working
        Memory 中保存；缺少显式开关、来源、事件或任一身份漂移时失败，不创建普通
        provision 许可，也不修改已有事实。
        """

        store = self._require_enabled()
        require_nonempty_string(task_id, "task_id")
        if not isinstance(resume, bool) or not resume:
            raise ContractValidationError(
                "legacy waiting resume authorization 必须来自明确 resume=True"
            )
        if (
            not isinstance(allow_legacy_waiting_resume_migration, bool)
            or not allow_legacy_waiting_resume_migration
        ):
            raise ContractValidationError(
                "legacy waiting resume migration 未获得显式授权"
            )
        value = self._coerce_investigation_payload(payload, fields)
        value["task_id"] = task_id
        self._set_investigation_protocol_version(value)
        value = validate_investigation_legacy_waiting_resume_authorization_payload(value)

        with store.immediate_transaction() as conn:
            self._validate_legacy_waiting_resume_authorization_task(conn, task_id)
            if value["task_id"] != task_id:
                raise ContractValidationError(
                    "legacy waiting authorization task 身份不一致"
                )
            self._read_legacy_waiting_resume_authorization_session(
                conn,
                task_id=task_id,
                value=value,
                source_refs=source_refs,
            )
            self._read_legacy_waiting_resume_authorization_started(
                conn,
                task_id=task_id,
                value=value,
                source_refs=source_refs,
            )
            self._read_legacy_waiting_resume_authorization_closure(
                conn,
                task_id=task_id,
                value=value,
                source_refs=source_refs,
            )
            refs = self._normalize_legacy_waiting_resume_authorization_source_refs(
                value,
                source_refs,
            )
            return self._append_legacy_waiting_resume_authorization_event(
                conn,
                task_id=task_id,
                value=value,
                source_refs=refs,
            )

    def list_investigation_legacy_waiting_resume_authorizations(
        self,
        task_id: str,
        batch_id: str,
        *,
        include_internal: bool = False,
    ) -> List[Dict[str, Any]]:
        """读取同一批次已经保存的旧 waiting 恢复授权事实。

        输入是 task/batch 身份；动作只读取并重新校验不可变事件正文，返回完整安全
        身份和事件 ID，不返回 token、原始路径或 provider 正文。事件重复、类型或
        摘要损坏时失败，不创建或修改任何状态。
        """

        store = self._require_enabled()
        require_nonempty_string(task_id, "task_id")
        require_nonempty_string(batch_id, "batch_id")
        with store.transaction() as conn:
            rows = conn.execute(
                "SELECT event_id,created_at,payload_json,source_refs_json FROM memory_events "
                "WHERE task_id=? AND event_type=? "
                "ORDER BY created_at,event_id",
                (task_id, "investigation_legacy_waiting_resume_authorized"),
            ).fetchall()
            result: list[Dict[str, Any]] = []
            for row in rows:
                value = validate_investigation_legacy_waiting_resume_authorization_payload(
                    self._decode_json(
                        row["payload_json"],
                        field="InvestigationLegacyWaitingResumeAuthorized.payload",
                    )
                )
                if value["task_id"] != task_id:
                    raise ContractValidationError(
                        "legacy waiting authorization task_id 与查询任务不一致"
                    )
                if value["batch_id"] != batch_id:
                    continue
                item = dict(value)
                item["event_id"] = row["event_id"]
                item["created_at"] = row["created_at"]
                if include_internal:
                    item["source_refs"] = self._decode_json(
                        row["source_refs_json"],
                        field="InvestigationLegacyWaitingResumeAuthorized.source_refs",
                    )
                result.append(item)
            return result

    def _validate_incomplete_v2_waiting_resume_authorization_task(
        self,
        conn: Any,
        task_id: str,
    ) -> None:
        """核对残缺 v2 waiting 授权所属 task 仍可追加事实。

        输入是事务连接和已校验的 task 身份；动作重新读取 TaskCreated、当前任务
        状态、终态索引和失败事件，拒绝已经结束或确认失败的任务。返回值为空，
        不修改历史；这样恢复授权不会绕过任务终态门禁。
        """

        task = self._require_investigation_task(conn, task_id)
        if task["status"] in self._TASK_TERMINAL_STATES:
            raise ContractValidationError(
                "终态 task 不能追加 incomplete v2 waiting resume authorization"
            )
        if conn.execute(
            "SELECT 1 FROM investigation_task_terminals WHERE task_id=?",
            (task_id,),
        ).fetchone() is not None:
            raise ContractValidationError(
                "已有 Investigation task terminal 不能追加 incomplete v2 waiting resume authorization"
            )
        if conn.execute(
            "SELECT 1 FROM memory_events WHERE task_id=? "
            "AND event_type='investigation_task_failed' LIMIT 1",
            (task_id,),
        ).fetchone() is not None:
            raise ContractValidationError(
                "已有 Investigation task failure 不能追加 incomplete v2 waiting resume authorization"
            )

    @staticmethod
    def _read_incomplete_v2_event(
        conn: Any,
        *,
        task_id: str,
        event_ref: str,
        expected_event_type: str,
        decode_json,
        field: str,
    ) -> tuple[Any, Dict[str, Any]]:
        """读取当前 task 的指定事件并返回数据库行与已解码正文。

        输入是事务连接、task、事件引用和预期类型；动作只读取不可变事件并解码
        JSON，不回退到当前索引或其它 task 的同名事件。事件不存在、类型漂移或正文
        不是 object 时抛出合同错误。
        """

        row = conn.execute(
            "SELECT event_id,event_type,payload_json,source_refs_json,created_at,event_key "
            "FROM memory_events WHERE task_id=? AND event_id=?",
            (task_id, event_ref),
        ).fetchone()
        if row is None or row["event_type"] != expected_event_type:
            raise ContractValidationError(
                f"incomplete v2 waiting authorization 缺少 {expected_event_type} 事件"
            )
        raw = decode_json(row["payload_json"], field=field)
        if not isinstance(raw, Mapping):
            raise ContractValidationError(f"{field} 必须是 JSON object")
        return row, dict(raw)

    def _read_incomplete_v2_legacy_authorization_root(
        self,
        conn: Any,
        *,
        task_id: str,
        value: Mapping[str, Any],
        source_refs: Sequence[str],
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        """读取并逐值核对旧 legacy 授权事件及其 root closure。

        输入是同一事务中的 task、残缺 v2 授权正文和规范来源；动作验证旧事件类型、
        旧授权正文、旧授权来源及 terminal_interrupted 根 closure，并核对两代授权
        共享的 session/thread/scope/workspace/kind 身份。返回旧授权正文和根 closure；
        当前替换作业的 job/request 身份可以不同，不能误把 replacement 当作 root。
        """

        legacy_row, legacy = self._read_incomplete_v2_event(
            conn,
            task_id=task_id,
            event_ref=value["legacy_authorization_event_ref"],
            expected_event_type="investigation_legacy_waiting_resume_authorized",
            decode_json=self._decode_json,
            field="InvestigationLegacyWaitingResumeAuthorized.payload",
        )
        legacy = validate_investigation_legacy_waiting_resume_authorization_payload(
            legacy
        )
        expected_legacy_key = event_key_investigation_legacy_waiting_resume_authorized(
            task_id,
            legacy["batch_id"],
            legacy["combination_id"],
            legacy["sidecar_record_digest"],
        )
        if legacy_row["event_key"] != expected_legacy_key:
            raise ContractValidationError(
                "incomplete v2 waiting authorization legacy event key 与正文不一致"
            )
        if value["legacy_authorization_event_ref"] not in source_refs:
            raise ContractValidationError(
                "incomplete v2 waiting authorization source_refs 必须包含 legacy authorization event"
            )
        legacy_refs = self._decode_json(
            legacy_row["source_refs_json"],
            field="InvestigationLegacyWaitingResumeAuthorized.source_refs",
        )
        legacy_refs = validate_source_ref_shape(legacy_refs)
        self._validate_source_refs(conn, task_id, legacy_refs)
        for required_ref in (
            legacy["session_event_ref"],
            legacy["turn_started_event_ref"],
            legacy["terminal_event_ref"],
        ):
            if required_ref not in legacy_refs:
                raise ContractValidationError(
                    "legacy authorization source_refs 缺少 root session/Started/closure"
                )
        shared_fields = (
            "task_id",
            "batch_id",
            "combination_id",
            "session_id",
            "session_event_ref",
            "kind",
            "codex_thread_id",
            "scope_id",
            "workspace_ref_digest",
            "physical_workspace_digest",
        )
        for field_name in shared_fields:
            if value.get(
                "logical_workspace_digest"
                if field_name == "workspace_ref_digest"
                else field_name
            ) != legacy.get(field_name):
                raise ContractValidationError(
                    f"incomplete v2 waiting authorization 与 legacy authorization 的 {field_name} 不一致"
                )
        if (
            value["current_logical_operation_instance_id"]
            != legacy["logical_operation_instance_id"]
        ):
            raise ContractValidationError(
                "incomplete v2 waiting authorization 与 legacy authorization 的 logical operation 不一致"
            )
        if value["root_attempt_closure_event_ref"] != legacy["terminal_event_ref"]:
            raise ContractValidationError(
                "incomplete v2 waiting authorization root closure 与 legacy authorization 不一致"
            )
        closure_row, raw_closure = self._read_incomplete_v2_event(
            conn,
            task_id=task_id,
            event_ref=value["root_attempt_closure_event_ref"],
            expected_event_type="investigation_turn_attempt_terminated",
            decode_json=self._decode_json,
            field="InvestigationTurnAttemptTerminated.payload",
        )
        root_closure = validate_investigation_turn_attempt_termination_payload(
            raw_closure
        )
        root_closure_refs = validate_source_ref_shape(
            self._decode_json(
                closure_row["source_refs_json"],
                field="InvestigationTurnAttemptTerminated.source_refs",
            )
        )
        self._validate_source_refs(conn, task_id, root_closure_refs)
        if root_closure["turn_started_event_ref"] not in root_closure_refs:
            raise ContractValidationError(
                "incomplete v2 waiting authorization root closure source_refs 缺少 Started"
            )
        if root_closure["terminal_status"] != "interrupted":
            raise ContractValidationError(
                "incomplete v2 waiting authorization root closure 必须为 terminal_interrupted"
            )
        if value["root_attempt_closure_event_ref"] not in source_refs:
            raise ContractValidationError(
                "incomplete v2 waiting authorization source_refs 必须包含 root closure"
            )
        for field_name in (
            "task_id",
            "batch_id",
            "combination_id",
            "kind",
            "turn_id",
            "attempt_number",
            "logical_operation_instance_id",
            "attempt_operation_instance_id",
            "codex_thread_id",
            "scope_id",
        ):
            if field_name in root_closure and field_name in legacy:
                expected = legacy[field_name]
            elif field_name in root_closure and field_name in value:
                expected = value[field_name]
            else:
                continue
            if root_closure.get(field_name) != expected:
                raise ContractValidationError(
                    f"incomplete v2 waiting authorization root closure.{field_name} 身份不一致"
                )
        if value["kind"] == "initial" and root_closure.get("session_id") != value["session_id"]:
            raise ContractValidationError(
                "incomplete v2 waiting authorization root closure.session_id 身份不一致"
            )
        if value["kind"] == "repair" and root_closure.get(
            "request_event_ref"
        ) != value.get("repair_request_event_ref"):
            raise ContractValidationError(
                "incomplete v2 waiting authorization root closure.request_event_ref 身份不一致"
            )
        if root_closure.get("turn_started_event_ref") != legacy["turn_started_event_ref"]:
            raise ContractValidationError(
                "incomplete v2 waiting authorization root closure Started 引用不一致"
            )
        return legacy, root_closure

    def _read_incomplete_v2_session_identity(
        self,
        conn: Any,
        *,
        task_id: str,
        value: Mapping[str, Any],
        source_refs: Sequence[str],
    ) -> Any:
        """读取残缺 v2 授权对应的 session 并逐值核对稳定身份。

        输入是事务连接、task、授权正文和来源；动作只读取 session 当前索引并由其
        workspace_ref 计算逻辑摘要，确认 session event、thread、scope 和组合完全
        一致。返回 session 行；索引缺失或正文漂移时抛出合同错误。
        """

        session = conn.execute(
            "SELECT * FROM investigation_sessions WHERE task_id=? AND batch_id=? "
            "AND combination_id=? AND session_id=?",
            (
                task_id,
                value["batch_id"],
                value["combination_id"],
                value["session_id"],
            ),
        ).fetchone()
        if session is None:
            raise ContractValidationError(
                "incomplete v2 waiting authorization 缺少对应 Investigation session"
            )
        expected = {
            "session_event_ref": session["session_event_id"],
            "session_id": session["session_id"],
            "codex_thread_id": session["codex_thread_id"],
            "scope_id": session["scope_id"],
            "logical_workspace_digest": hashlib.sha256(
                canonical_json(session["workspace_ref"]).encode("utf-8")
            ).hexdigest(),
        }
        for field_name, expected_value in expected.items():
            if value[field_name] != expected_value:
                raise ContractValidationError(
                    f"incomplete v2 waiting authorization.{field_name} 与 session 不一致"
                )
        if value["session_event_ref"] not in source_refs:
            raise ContractValidationError(
                "incomplete v2 waiting authorization source_refs 必须包含 session event"
            )
        return session

    def _read_incomplete_v2_started_event(
        self,
        conn: Any,
        *,
        task_id: str,
        event_ref: str,
        kind: str,
    ) -> Mapping[str, Any]:
        """读取 attempt chain 中的一条 initial/repair Started 事实。"""

        expected_type = (
            "investigation_turn_started"
            if kind == "initial"
            else "investigation_repair_turn_started"
        )
        _row, raw = self._read_incomplete_v2_event(
            conn,
            task_id=task_id,
            event_ref=event_ref,
            expected_event_type=expected_type,
            decode_json=self._decode_json,
            field=f"{expected_type}.payload",
        )
        if kind == "initial":
            return validate_investigation_turn_started_payload(raw)
        return validate_investigation_repair_turn_started_payload(raw)

    def _read_incomplete_v2_closure_event(
        self,
        conn: Any,
        *,
        task_id: str,
        event_ref: str,
    ) -> Mapping[str, Any]:
        """读取 attempt chain 中的一条 terminal_interrupted closure。"""

        _row, raw = self._read_incomplete_v2_event(
            conn,
            task_id=task_id,
            event_ref=event_ref,
            expected_event_type="investigation_turn_attempt_terminated",
            decode_json=self._decode_json,
            field="InvestigationTurnAttemptTerminated.payload",
        )
        value = validate_investigation_turn_attempt_termination_payload(raw)
        if value["terminal_status"] != "interrupted":
            raise ContractValidationError(
                "incomplete v2 waiting authorization attempt closure 必须为 terminal_interrupted"
            )
        return value

    def _read_incomplete_v2_attempt_chain(
        self,
        conn: Any,
        *,
        task_id: str,
        value: Mapping[str, Any],
        source_refs: Sequence[str],
        root_closure: Mapping[str, Any],
    ) -> None:
        """逐事件核对 root closure 到当前 Started 的连续 attempt 链。

        输入是同一事务中的 task、完整授权正文、规范来源和已校验根 closure；动作
        读取每个 Started/closure，确认编号连续、前驱 closure 串联、同 session/thread/
        scope/logical operation，且只有末项是当前 in-flight Started。返回值为空；
        任一缺失、断链、身份漂移、closure 类型错误或当前 Started 已有 closure 都
        抛出合同错误。
        """

        chain = value["attempt_chain"]
        root_number = root_closure["attempt_number"]
        previous_closure_ref = value["root_attempt_closure_event_ref"]
        previous_attempt_number = root_number
        stable_fields = {
            "task_id": task_id,
            "batch_id": value["batch_id"],
            "combination_id": value["combination_id"],
            "codex_thread_id": value["codex_thread_id"],
            "scope_id": value["scope_id"],
            "logical_operation_instance_id": value[
                "current_logical_operation_instance_id"
            ],
        }
        for index, item in enumerate(chain):
            started_ref = item["started_event_ref"]
            if started_ref not in source_refs:
                raise ContractValidationError(
                    f"incomplete v2 waiting authorization source_refs 缺少 chain Started[{index}]"
                )
            if item["attempt_number"] != previous_attempt_number + 1:
                raise ContractValidationError(
                    "incomplete v2 waiting authorization attempt chain 编号不连续"
                )
            started = self._read_incomplete_v2_started_event(
                conn,
                task_id=task_id,
                event_ref=started_ref,
                kind=value["kind"],
            )
            for field_name, expected in stable_fields.items():
                if started.get(field_name) != expected:
                    raise ContractValidationError(
                        f"incomplete v2 waiting authorization chain Started.{field_name} 身份漂移"
                    )
            if value["kind"] == "initial" and started.get("session_id") != value["session_id"]:
                raise ContractValidationError(
                    "incomplete v2 waiting authorization chain Started.session_id 身份漂移"
                )
            if value["kind"] == "repair" and started.get("request_event_ref") != value.get(
                "repair_request_event_ref"
            ):
                raise ContractValidationError(
                    "incomplete v2 waiting authorization chain repair request identity 漂移"
                )
            if started["attempt_number"] != item["attempt_number"]:
                raise ContractValidationError(
                    "incomplete v2 waiting authorization chain Started.attempt_number 不一致"
                )
            if started.get("previous_attempt_closure_event_ref") != previous_closure_ref:
                raise ContractValidationError(
                    "incomplete v2 waiting authorization Started.previous_attempt_closure_event_ref 断裂"
                )
            closure_ref = item["closure_event_ref"]
            if index == len(chain) - 1:
                if closure_ref is not None:
                    raise ContractValidationError(
                        "incomplete v2 waiting authorization 当前 Started 不得已有 closure"
                    )
                if started.get("attempt_status", "in_flight") != "in_flight":
                    raise ContractValidationError(
                        "incomplete v2 waiting authorization 当前 Started 必须处于 in_flight"
                    )
                existing_closure = conn.execute(
                    "SELECT 1 FROM memory_events "
                    "WHERE task_id=? AND event_type='investigation_turn_attempt_terminated' "
                    "AND json_extract(payload_json,'$.turn_started_event_ref')=? LIMIT 1",
                    (task_id, started_ref),
                ).fetchone()
                if existing_closure is not None:
                    raise ContractValidationError(
                        "incomplete v2 waiting authorization 当前 Started 已有 closure"
                    )
                if started_ref != value["current_turn_started_event_ref"]:
                    raise ContractValidationError(
                        "incomplete v2 waiting authorization 当前 Started 引用不一致"
                    )
                current_fields = {
                    "turn_id": value["current_turn_id"],
                    "attempt_number": value["current_attempt_number"],
                    "logical_operation_instance_id": value[
                        "current_logical_operation_instance_id"
                    ],
                    "attempt_operation_instance_id": value[
                        "current_attempt_operation_instance_id"
                    ],
                }
                for field_name, expected in current_fields.items():
                    actual_field = (
                        "operation_instance_id"
                        if field_name == "attempt_operation_instance_id"
                        else field_name
                    )
                    if started.get(actual_field) != expected:
                        raise ContractValidationError(
                            f"incomplete v2 waiting authorization 当前 Started.{actual_field} 身份不一致"
                        )
            else:
                if closure_ref is None:
                    raise ContractValidationError(
                        "incomplete v2 waiting authorization 中间 attempt 缺少 closure"
                    )
                if closure_ref not in source_refs:
                    raise ContractValidationError(
                        f"incomplete v2 waiting authorization source_refs 缺少 chain closure[{index}]"
                    )
                closure = self._read_incomplete_v2_closure_event(
                    conn,
                    task_id=task_id,
                    event_ref=closure_ref,
                )
                expected_closure = {
                    "task_id": task_id,
                    "batch_id": value["batch_id"],
                    "combination_id": value["combination_id"],
                    "kind": value["kind"],
                    "turn_started_event_ref": started_ref,
                    "turn_id": started["turn_id"],
                    "attempt_number": item["attempt_number"],
                    "logical_operation_instance_id": value[
                        "current_logical_operation_instance_id"
                    ],
                    "attempt_operation_instance_id": started[
                        "operation_instance_id"
                    ],
                    "previous_attempt_closure_event_ref": previous_closure_ref,
                    "codex_thread_id": value["codex_thread_id"],
                    "scope_id": value["scope_id"],
                }
                if value["kind"] == "initial":
                    expected_closure["session_id"] = value["session_id"]
                else:
                    expected_closure["request_event_ref"] = value[
                        "repair_request_event_ref"
                    ]
                for field_name, expected in expected_closure.items():
                    if closure.get(field_name) != expected:
                        raise ContractValidationError(
                            f"incomplete v2 waiting authorization chain closure.{field_name} 身份漂移"
                        )
                closure_refs_row = conn.execute(
                    "SELECT source_refs_json FROM memory_events WHERE task_id=? AND event_id=?",
                    (task_id, closure_ref),
                ).fetchone()
                if closure_refs_row is None:
                    raise ContractValidationError("attempt closure 历史记录缺失")
                closure_refs = validate_source_ref_shape(
                    self._decode_json(
                        closure_refs_row["source_refs_json"],
                        field="InvestigationTurnAttemptTerminated.source_refs",
                    )
                )
                if started_ref not in closure_refs:
                    raise ContractValidationError(
                        "incomplete v2 waiting authorization chain closure source_refs 缺少 Started"
                    )
                previous_closure_ref = closure_ref
            previous_attempt_number = item["attempt_number"]

    def _validate_incomplete_v2_waiting_resume_authorization_history(
        self,
        conn: Any,
        *,
        task_id: str,
        value: Mapping[str, Any],
        source_refs: Sequence[str],
    ) -> None:
        """在一个事务中完成残缺 v2 waiting 授权的历史核对。

        输入是事务连接、task、已通过 payload validator 的正文和来源；动作按固定
        顺序核对 session、legacy root、attempt chain、当前 Started 与来源集合，
        不创建事件、不更新当前索引。任一证据来源的身份不一致时停止。
        """

        refs = validate_source_ref_shape(sorted(set(source_refs)))
        self._validate_source_refs(conn, task_id, refs)
        session = self._read_incomplete_v2_session_identity(
            conn,
            task_id=task_id,
            value=value,
            source_refs=refs,
        )
        _legacy, root_closure = self._read_incomplete_v2_legacy_authorization_root(
            conn,
            task_id=task_id,
            value=value,
            source_refs=refs,
        )
        self._read_incomplete_v2_attempt_chain(
            conn,
            task_id=task_id,
            value=value,
            source_refs=refs,
            root_closure=root_closure,
        )
        current_pointer = (
            session["current_turn_event_id"]
            if value["kind"] == "initial"
            else session["current_repair_turn_event_id"]
        )
        if current_pointer != value["current_turn_started_event_ref"]:
            raise ContractValidationError(
                "incomplete v2 waiting authorization 当前 Started 不是 session active attempt"
            )

    def _append_incomplete_v2_waiting_resume_authorization_event(
        self,
        conn: Any,
        *,
        task_id: str,
        value: Mapping[str, Any],
        source_refs: Sequence[str],
    ) -> str:
        """追加残缺 v2 waiting 授权事实，并明确声明不更新当前索引。"""

        event_key = event_key_investigation_incomplete_v2_waiting_resume_authorized(
            task_id,
            value["batch_id"],
            value["combination_id"],
            value["sidecar_record_digest"],
        )
        return self._insert_investigation_event(
            conn,
            task_id=task_id,
            event_key=event_key,
            event_type="investigation_incomplete_v2_waiting_resume_authorized",
            round_id=value["current_attempt_number"],
            payload=value,
            source_refs=source_refs,
            projection=lambda _event_id: None,
        )

    def record_investigation_incomplete_v2_waiting_resume_authorization(
        self,
        task_id: str,
        payload: Mapping[str, Any] | None = None,
        source_refs: Sequence[str] = (),
        *,
        resume: bool = False,
        allow_incomplete_v2_waiting_resume_migration: bool = False,
        **fields: Any,
    ) -> str:
        """登记残缺 v2 waiting sidecar 的显式历史恢复授权。

        输入必须来自明确的 ``resume=True`` 装配，并由专用 migration 开关显式允许；
        payload 绑定当前 v2 sidecar、旧 legacy 授权根、连续 Started/closure、当前
        作业和所有工作区/scope/receipt 摘要。动作在同一 SQLite 事务内重新读取并核对
        全部不可变历史，只追加不更新当前索引的事件；缺少开关、来源、连续 attempt、当前 task
        可写条件或任一身份漂移时失败，不修改任何已有事实。
        """

        store = self._require_enabled()
        require_nonempty_string(task_id, "task_id")
        if not isinstance(resume, bool) or not resume:
            raise ContractValidationError(
                "incomplete v2 waiting resume authorization 必须来自明确 resume=True"
            )
        if (
            not isinstance(allow_incomplete_v2_waiting_resume_migration, bool)
            or not allow_incomplete_v2_waiting_resume_migration
        ):
            raise ContractValidationError(
                "incomplete v2 waiting resume migration 未获得显式授权"
            )
        value = self._coerce_investigation_payload(payload, fields)
        if "task_id" in value and value["task_id"] != task_id:
            raise ContractValidationError(
                "incomplete v2 waiting authorization task_id 与参数不一致"
            )
        value["task_id"] = task_id
        self._set_investigation_protocol_version(value)
        value = validate_investigation_incomplete_v2_waiting_resume_authorization_payload(
            value
        )
        with store.immediate_transaction() as conn:
            self._validate_incomplete_v2_waiting_resume_authorization_task(conn, task_id)
            self._validate_incomplete_v2_waiting_resume_authorization_history(
                conn,
                task_id=task_id,
                value=value,
                source_refs=source_refs,
            )
            refs = sorted(set(source_refs))
            required_refs = {
                value["session_event_ref"],
                value["legacy_authorization_event_ref"],
                value["root_attempt_closure_event_ref"],
                *(
                    ref
                    for item in value["attempt_chain"]
                    for ref in (item["started_event_ref"], item["closure_event_ref"])
                    if isinstance(ref, str)
                ),
                value["current_turn_started_event_ref"],
            }
            if not required_refs.issubset(refs):
                raise ContractValidationError(
                    "incomplete v2 waiting authorization source_refs 不完整"
                )
            return self._append_incomplete_v2_waiting_resume_authorization_event(
                conn,
                task_id=task_id,
                value=value,
                source_refs=refs,
            )

    def list_investigation_incomplete_v2_waiting_resume_authorizations(
        self,
        task_id: str,
        batch_id: str,
        *,
        include_internal: bool = False,
    ) -> List[Dict[str, Any]]:
        """读取并重新验证残缺 v2 waiting 授权历史。

        输入是 task/batch 身份；动作重新执行 payload、event key、来源数组和每个来源
        存在性校验，返回事件正文及事件时间；默认不返回 source_refs，内部读取可显式
        请求来源。该事件不更新可变当前索引，读取不会修复或改写数据库。
        """

        store = self._require_enabled()
        require_nonempty_string(task_id, "task_id")
        require_nonempty_string(batch_id, "batch_id")
        with store.transaction() as conn:
            rows = conn.execute(
                "SELECT event_id,event_key,created_at,payload_json,source_refs_json "
                "FROM memory_events WHERE task_id=? AND event_type=? "
                "ORDER BY created_at,event_id",
                (
                    task_id,
                    "investigation_incomplete_v2_waiting_resume_authorized",
                ),
            ).fetchall()
            result: list[Dict[str, Any]] = []
            for row in rows:
                value = validate_investigation_incomplete_v2_waiting_resume_authorization_payload(
                    self._decode_json(
                        row["payload_json"],
                        field="InvestigationIncompleteV2WaitingResumeAuthorized.payload",
                    )
                )
                if value["task_id"] != task_id:
                    raise ContractValidationError(
                        "incomplete v2 waiting authorization task_id 与查询任务不一致"
                    )
                if value["batch_id"] != batch_id:
                    continue
                expected_key = event_key_investigation_incomplete_v2_waiting_resume_authorized(
                    task_id,
                    value["batch_id"],
                    value["combination_id"],
                    value["sidecar_record_digest"],
                )
                if row["event_key"] != expected_key:
                    raise ContractValidationError(
                        "incomplete v2 waiting authorization event key 与正文不一致"
                    )
                refs = validate_source_ref_shape(
                    self._decode_json(
                        row["source_refs_json"],
                        field="InvestigationIncompleteV2WaitingResumeAuthorized.source_refs",
                    )
                )
                self._validate_source_refs(conn, task_id, refs)
                required_refs = {
                    value["session_event_ref"],
                    value["legacy_authorization_event_ref"],
                    value["root_attempt_closure_event_ref"],
                    *(
                        ref
                        for item in value["attempt_chain"]
                        for ref in (item["started_event_ref"], item["closure_event_ref"])
                        if isinstance(ref, str)
                    ),
                    value["current_turn_started_event_ref"],
                }
                if not required_refs.issubset(refs):
                    raise ContractValidationError(
                        "incomplete v2 waiting authorization source_refs 不完整"
                    )
                item = dict(value)
                item["event_id"] = row["event_id"]
                item["created_at"] = row["created_at"]
                if include_internal:
                    item["source_refs"] = refs
                result.append(item)
            return result

    @staticmethod
    def _validate_investigation_result_session(
        conn,
        *,
        task_id: str,
        value: Mapping[str, Any],
    ) -> tuple[Any, str]:
        """核对 Result 的 v8 证据引用、组合 session 和稳定身份。

        输入是当前事务、task 身份和已通过正文合同的 Result；动作读取对应 session，
        拒绝 marker 后写入、thread/scope 漂移和无证据事件；返回 session 行及证据事件
        引用。未知 session、协议或引用不完整时抛出合同错误，不写入数据库。
        """

        if value.get("investigation_protocol_version") != INVESTIGATION_PROTOCOL_VERSION:
            raise ContractValidationError("v8 InvestigationResult 必须声明协议版本 8")
        evidence_ref = value.get("evidence_verification_event_ref")
        if not isinstance(evidence_ref, str):
            raise ContractValidationError("v8 InvestigationResult 必须引用验证证据事件")
        session = conn.execute(
            "SELECT * FROM investigation_sessions WHERE task_id=? AND batch_id=? AND combination_id=?",
            (task_id, value["batch_id"], value["combination_id"]),
        ).fetchone()
        if session is None:
            raise ContractValidationError("InvestigationResult 缺少对应 session")
        if session["route_marker"] is not None:
            raise ContractValidationError("已写入 revise_hypothesis marker 的组合不能再登记 result")
        if (
            value.get("session_event_ref") is not None
            and value["session_event_ref"] != session["session_event_id"]
        ):
            raise ContractValidationError("InvestigationResult 的 session_event_ref 与当前 session 不一致")
        if (
            value["codex_thread_id"] != session["codex_thread_id"]
            or value["scope_id"] != session["scope_id"]
        ):
            raise ContractValidationError("InvestigationResult 与 session 身份不一致")
        prior_result_ref = session["latest_result_event_id"]
        pending_repair_ref = session["pending_repair_event_id"]
        if pending_repair_ref is not None and not isinstance(prior_result_ref, str):
            raise ContractValidationError("pending repair 必须保留可引用的前一版 result")
        return session, evidence_ref

    @staticmethod
    def _validate_investigation_result_replay_state(
        *,
        prior_result_ref: Any,
        pending_repair_ref: Any,
        replaying_existing_result: bool,
        value: Mapping[str, Any],
    ) -> None:
        """核对 Result 是否按首版或 pending repair 顺序提交。

        输入是当前结果引用、pending repair 引用、replay 标记和 Result 正文；动作只
        检查顺序与返修 turn 绑定，不写入状态；越过当前 repair 或缺少 turn 引用时抛出
        合同错误。
        """

        if prior_result_ref is not None and pending_repair_ref is None and not replaying_existing_result:
            raise ContractValidationError("非首版 Result 必须先有当前 pending repair")
        if pending_repair_ref is not None and value.get("repair_turn_event_ref") is None:
            raise ContractValidationError("返修 Result 必须引用已成功启动的 repair turn")

    @staticmethod
    def _validate_investigation_result_turn_reference(
        conn,
        *,
        task_id: str,
        value: Mapping[str, Any],
        session: Any,
        pending_repair_ref: Any,
        replaying_existing_result: bool,
    ) -> str | None:
        """核对 Result 使用的 initial 或 repair turn，并返回 turn 事件引用。

        输入是同一事务中的 Result、session 和当前 repair 引用；动作只读取已成功启动
        的 turn，拒绝自报 turn、请求不匹配和身份漂移；合法的既有 Result replay 返回
        None，不改变任何数据库状态，坏的 turn 引用抛出合同错误。
        """

        if replaying_existing_result:
            return None
        if pending_repair_ref is None:
            turn_ref = session["current_turn_event_id"]
            turn = (
                conn.execute(
                    "SELECT turn_id FROM investigation_turns WHERE task_id=? "
                    "AND turn_started_event_id=?",
                    (task_id, turn_ref),
                ).fetchone()
                if isinstance(turn_ref, str)
                else None
            )
            if turn is None or turn["turn_id"] != value["turn_id"]:
                raise ContractValidationError(
                    "首版 Result 必须使用当前 initial turn 的 turn_id"
                )
            return turn_ref
        turn_ref = value.get("repair_turn_event_ref")
        turn = (
            conn.execute(
                "SELECT turn_id,request_event_ref FROM investigation_repair_turns "
                "WHERE task_id=? AND turn_started_event_id=?",
                (task_id, turn_ref),
            ).fetchone()
            if isinstance(turn_ref, str)
            else None
        )
        if (
            turn is None
            or turn["request_event_ref"] != pending_repair_ref
            or turn["turn_id"] != value["turn_id"]
        ):
            raise ContractValidationError("返修 Result 必须使用当前 repair turn 的 turn_id")
        return turn_ref

    def _load_investigation_result_replay_source_refs(
        self,
        *,
        existing_event: Any,
        source_refs: Sequence[str],
    ) -> list[str]:
        """读取 Result replay 的来源数组，合法缺省保持为空。

        输入是可选既有事件和调用方来源；动作在 replay 缺少来源时读取原事件 JSON，
        不写入状态；返回来源列表，JSON 语法或形状损坏时抛出合同错误。
        """

        refs = list(source_refs)
        if existing_event is not None and not refs:
            refs = self._decode_json(
                existing_event["source_refs_json"],
                field="InvestigationResult.source_refs_json",
            )
            if not isinstance(refs, list):
                raise ContractValidationError("InvestigationResult.source_refs_json 已损坏")
        return refs

    @staticmethod
    def _add_investigation_result_default_source_refs(
        refs: list[str],
        *,
        identity_refs: Sequence[str],
        evidence_ref: str,
        turn_ref: str | None,
        prior_result_ref: Any,
        pending_repair_ref: Any,
        value: Mapping[str, Any],
        replaying_existing_result: bool,
    ) -> list[str]:
        """为缺少调用方来源的 Result 写入请求补齐确定性来源。

        输入是已有来源、session/证据/turn/版本引用和 Result 正文；动作只在来源为空
        时补齐稳定引用，不写入数据库；返回可排序的来源列表，不吞异常或生成默认事实。
        """

        if refs:
            return refs
        refs = list(identity_refs)
        refs.append(evidence_ref)
        if isinstance(turn_ref, str) and not replaying_existing_result:
            refs.append(turn_ref)
        if isinstance(prior_result_ref, str):
            refs.append(prior_result_ref)
        if isinstance(pending_repair_ref, str):
            refs.append(pending_repair_ref)
        if isinstance(value.get("repair_turn_event_ref"), str):
            refs.append(value["repair_turn_event_ref"])
        return refs

    @staticmethod
    def _validate_investigation_result_required_source_refs(
        refs: Sequence[str],
        *,
        identity_refs: Sequence[str],
        evidence_ref: str,
        turn_ref: str | None,
        prior_result_ref: Any,
        pending_repair_ref: Any,
        value: Mapping[str, Any],
        replaying_existing_result: bool,
    ) -> None:
        """核对 Result 来源是否覆盖 session、证据、前版结果与当前 turn。

        输入是来源列表、当前 Result 相关引用和 replay 标记；动作只检查必需引用集合，
        不改变状态；来源缺项时抛出合同错误。
        """

        required_refs = {
            *identity_refs,
            evidence_ref,
            *(
                ref
                for ref in (prior_result_ref, pending_repair_ref)
                if isinstance(ref, str)
                and not (replaying_existing_result and ref == prior_result_ref)
            ),
        }
        if isinstance(value.get("repair_turn_event_ref"), str):
            required_refs.add(value["repair_turn_event_ref"])
        if isinstance(turn_ref, str) and not replaying_existing_result:
            required_refs.add(turn_ref)
        if not required_refs.issubset(refs):
            raise ContractValidationError(
                "InvestigationResult source_refs 必须保留 session、前版 result 和 pending repair"
            )

    def _collect_investigation_result_source_refs(
        self,
        *,
        existing_event: Any,
        source_refs: Sequence[str],
        identity_refs: Sequence[str],
        evidence_ref: str,
        turn_ref: str | None,
        prior_result_ref: Any,
        pending_repair_ref: Any,
        value: Mapping[str, Any],
        replaying_existing_result: bool,
    ) -> list[str]:
        """收集并核对 Result 的 session、证据、turn 与版本来源引用。

        输入是同一事务中已经核对的 Result 事实和来源；动作在 replay 缺少调用方来源
        时读取原事件来源，随后补齐首版/返修所需来源并排序去重；返回可追加事件的
        来源列表。既有来源 JSON 损坏或来源缺项时抛出合同错误，不使用默认引用。
        """

        refs = self._load_investigation_result_replay_source_refs(
            existing_event=existing_event,
            source_refs=source_refs,
        )
        refs = self._add_investigation_result_default_source_refs(
            refs,
            identity_refs=identity_refs,
            evidence_ref=evidence_ref,
            turn_ref=turn_ref,
            prior_result_ref=prior_result_ref,
            pending_repair_ref=pending_repair_ref,
            value=value,
            replaying_existing_result=replaying_existing_result,
        )
        refs = sorted(set(refs))
        self._validate_investigation_result_required_source_refs(
            refs,
            identity_refs=identity_refs,
            evidence_ref=evidence_ref,
            turn_ref=turn_ref,
            prior_result_ref=prior_result_ref,
            pending_repair_ref=pending_repair_ref,
            value=value,
            replaying_existing_result=replaying_existing_result,
        )
        return refs

    def record_investigation_result(
        self,
        task_id: str,
        payload: Mapping[str, Any] | None = None,
        source_refs: Sequence[str] = (),
        **fields: Any,
    ) -> str:
        """原子登记一版 InvestigationResult，并推进当前结果游标。

        输入是组合结果正文和 VerifiedInvestigationEvidence、session/repair turn 来源；
        动作在同一事务核对当前身份、首版/返修顺序并追加事件与当前索引。返回 result
        事件 ID；v8 缺少验证证据、非首版绕过 pending repair、marker 后写入或来源漂移
        时失败，旧历史只支持读取/重建，不被伪造为新事实。
        """

        store = self._require_enabled()
        value = self._coerce_investigation_payload(payload, fields)
        value["task_id"] = task_id
        value = validate_investigation_result_payload(value)
        with store.immediate_transaction() as conn:
            self._require_investigation_task(conn, task_id)
            session, evidence_ref = self._validate_investigation_result_session(
                conn,
                task_id=task_id,
                value=value,
            )
            prior_result_ref = session["latest_result_event_id"]
            pending_repair_ref = session["pending_repair_event_id"]
            event_key = event_key_investigation_result(
                task_id,
                value["batch_id"],
                value["combination_id"],
                value["result_version"],
            )
            refs = list(source_refs)
            identity_refs = self._investigation_session_identity_event_refs(
                conn,
                task_id=task_id,
                batch_id=value["batch_id"],
                combination_id=value["combination_id"],
                session_event_id=session["session_event_id"],
            )
            existing_event = conn.execute(
                "SELECT event_id,source_refs_json FROM memory_events WHERE event_key=?",
                (event_key,),
            ).fetchone()
            replaying_existing_result = existing_event is not None
            self._validate_investigation_result_replay_state(
                prior_result_ref=prior_result_ref,
                pending_repair_ref=pending_repair_ref,
                replaying_existing_result=replaying_existing_result,
                value=value,
            )
            turn_ref = self._validate_investigation_result_turn_reference(
                conn,
                task_id=task_id,
                value=value,
                session=session,
                pending_repair_ref=pending_repair_ref,
                replaying_existing_result=replaying_existing_result,
            )
            refs = self._collect_investigation_result_source_refs(
                existing_event=existing_event,
                source_refs=refs,
                identity_refs=identity_refs,
                evidence_ref=evidence_ref,
                turn_ref=turn_ref,
                prior_result_ref=prior_result_ref,
                pending_repair_ref=pending_repair_ref,
                value=value,
                replaying_existing_result=replaying_existing_result,
            )
            return self._insert_investigation_event(
                conn,
                task_id=task_id,
                event_key=event_key,
                event_type="investigation_result_recorded",
                round_id=1,
                payload=value,
                source_refs=refs,
                projection=lambda event_id: insert_investigation_result_projection(
                    conn,
                    task_id=task_id,
                    event_id=event_id,
                    payload=value,
                    created_at=conn.execute(
                        "SELECT created_at FROM memory_events WHERE event_id=?", (event_id,)
                    ).fetchone()[0],
                ),
            )

    @staticmethod
    def _validate_investigation_review_protocol(value: Mapping[str, Any]) -> None:
        """确认批次 Review 使用当前 v8 协议。

        输入是已通过 Review 正文合同的 mapping；动作只检查协议字段，返回值为空，
        缺失或漂移时抛出合同错误，不修改数据库。
        """

        if value.get("investigation_protocol_version") != INVESTIGATION_PROTOCOL_VERSION:
            raise ContractValidationError("v8 Review 必须声明协议版本 8")

    @staticmethod
    def _load_investigation_review_turn_context(
        conn,
        *,
        task_id: str,
        value: Mapping[str, Any],
    ) -> tuple[Any, Any]:
        """读取并核对当前 task ReviewSession 与 Review turn 身份。

        输入是事务连接、task 和规范化 Review 正文；动作读取冻结的会话/turn，核对
        batch、thread、turn 和会话引用，返回两行数据库记录。身份缺失或漂移时抛出
        合同错误，不消费 turn，也不写入状态。
        """

        task_review_session = conn.execute(
            "SELECT * FROM task_review_sessions WHERE task_id=?", (task_id,)
        ).fetchone()
        task_review_turn = conn.execute(
            "SELECT * FROM task_review_turns WHERE task_id=? AND turn_started_event_id=?",
            (task_id, value.get("task_review_turn_event_ref")),
        ).fetchone()
        if task_review_session is None or task_review_turn is None:
            raise ContractValidationError("v8 Review 必须先启动 TaskReviewSession 和 Review turn")
        if value.get("task_review_session_event_ref") != task_review_session["session_event_id"]:
            raise ContractValidationError("Review 必须引用当前 TaskReviewSession")
        if (
            task_review_turn["batch_id"] != value["batch_id"]
            or task_review_turn["review_thread_id"] != value["review_thread_id"]
            or task_review_turn["review_turn_id"] != value["review_turn_id"]
            or task_review_session["codex_thread_id"] != value["review_thread_id"]
        ):
            raise ContractValidationError("Review thread/turn 与 task 会话冻结身份不一致")
        return task_review_session, task_review_turn

    @staticmethod
    def _validate_investigation_review_turn_available(
        conn,
        *,
        task_id: str,
        task_review_session: Any,
        task_review_turn: Any,
    ) -> None:
        """确认 Review turn 仍处于可消费的 in-flight 状态。

        输入是事务连接、task、会话行和 turn 行；动作检查阶段和不可变事件消费记录，
        返回值为空，已消费或不在 flight 状态时抛出合同错误，不修改数据库。
        """

        if task_review_session["phase"] != "turn_in_flight":
            raise ContractValidationError("Review 必须消费当前 in-flight turn")
        consumed = conn.execute(
            "SELECT 1 FROM memory_events WHERE task_id=? "
            "AND event_type='investigation_batch_reviewed' "
            "AND json_extract(payload_json,'$.task_review_turn_event_ref')=? LIMIT 1",
            (task_id, task_review_turn["turn_started_event_id"]),
        ).fetchone()
        if consumed is not None:
            raise ContractValidationError("Review turn 已被消费，不能重用")

    @staticmethod
    def _collect_investigation_review_result_event_refs(
        conn,
        *,
        task_id: str,
        value: Mapping[str, Any],
    ) -> list[str]:
        """读取 Review 绑定的五个当前 Result 事件身份。

        输入是事务连接、task 和规范化 Review；动作逐项核对 session 当前版本与
        latest Result，返回五个事件 ID。缺 session、版本过期或结果缺失时抛出合同错误，
        不写入数据库。
        """

        result_event_refs: list[str] = []
        for combination_id, result_version in zip(
            value["combination_ids"], value["result_versions"]
        ):
            session = conn.execute(
                "SELECT * FROM investigation_sessions WHERE task_id=? AND batch_id=? AND combination_id=?",
                (task_id, value["batch_id"], combination_id),
            ).fetchone()
            if session is None:
                raise ContractValidationError("Review 必须完整包含五个已创建 session")
            if (
                session["current_result_version"] != result_version
                or session["latest_result_event_id"] is None
            ):
                raise ContractValidationError("Review 必须绑定五个当前 InvestigationResult")
            result_event_refs.append(session["latest_result_event_id"])
        return result_event_refs

    def _prepare_investigation_review_source_refs(
        self,
        conn,
        *,
        task_id: str,
        source_refs: Sequence[str],
        result_event_refs: Sequence[str],
        task_review_session: Any,
        task_review_turn: Any,
    ) -> list[str]:
        """规范化并核对 Review 必需的结果、会话和 turn 来源引用。

        输入是事务连接、task、调用方来源和已读取的 Review 事实身份；动作排序去重、
        核对来源存在性及完整覆盖，返回可追加事件的引用列表。来源缺失或格式错误时
        抛出合同错误，不写入数据库。
        """

        required_review_refs = {
            *result_event_refs,
            task_review_session["session_event_id"],
            task_review_turn["turn_started_event_id"],
        }
        refs = sorted(set(source_refs)) if source_refs else sorted(required_review_refs)
        refs = validate_source_ref_shape(refs)
        refs = self._validate_source_refs(conn, task_id, refs)
        if not required_review_refs.issubset(set(refs)):
            raise ContractValidationError(
                "Review source_refs 必须引用五项当前结果、task 会话和 turn"
            )
        return refs

    def _find_investigation_review_exact_replay(
        self,
        conn,
        *,
        task_id: str,
        value: Mapping[str, Any],
        source_refs: Sequence[str],
    ) -> tuple[str | None, str]:
        """按稳定 Review 事件键核对完全相同的重放。

        输入是事务连接、task、规范化 Review 和已核对来源；动作返回原事件 ID（若有）
        及幂等键。正文或来源与历史冲突时抛出幂等冲突，不依赖当前 in-flight 阶段。
        """

        event_key = event_key_investigation_batch_review(
            task_id, value["batch_id"], value["result_version_vector"]
        )
        existing = self._existing_event_exact(
            conn,
            event_key=event_key,
            task_id=task_id,
            round_id=1,
            event_type="investigation_batch_reviewed",
            payload=value,
            source_refs=source_refs,
        )
        return existing, event_key

    def _append_investigation_batch_review_event(
        self,
        conn,
        *,
        task_id: str,
        value: Mapping[str, Any],
        source_refs: Sequence[str],
        event_key: str,
        idempotency_key: str,
    ) -> str:
        """追加 Review 事件并更新完整五项投影。

        输入是事务连接、task、Review 正文、来源、稳定事件键和幂等键；动作在同一事务
        写入不可变事实并调用 Review 投影，返回事件 ID。数据库或投影失败由外层事务回滚。
        """

        return self._insert_investigation_event(
            conn,
            task_id=task_id,
            event_key=event_key,
            event_type="investigation_batch_reviewed",
            round_id=1,
            payload=value,
            source_refs=source_refs,
            projection=lambda event_id: insert_investigation_review_projection(
                conn,
                task_id=task_id,
                event_id=event_id,
                payload=value,
                created_at=conn.execute(
                    "SELECT created_at FROM memory_events WHERE event_id=?", (event_id,)
                ).fetchone()[0],
                idempotency_key=idempotency_key,
            ),
        )

    def record_investigation_batch_review(
        self,
        task_id: str,
        payload: Mapping[str, Any] | None = None,
        source_refs: Sequence[str] = (),
        **fields: Any,
    ) -> str:
        """原子登记绑定当前五项结果版本向量的批次 Review。

        输入是五个当前 Result、完整 finding、TaskReviewSession/turn 和 review thread；
        动作校验全批身份及同 task thread 后，在一个事务追加 Review 并更新五个当前
        索引。返回 review 事件 ID；缺少完整向量、使用过期结果、thread 漂移或绕过
        Review turn 时失败，fail advice 仍只作为待返修建议保存。
        """

        store = self._require_enabled()
        value = self._coerce_investigation_payload(payload, fields)
        value["task_id"] = task_id
        value = validate_investigation_batch_review_payload(value)
        with store.immediate_transaction() as conn:
            self._require_investigation_task(conn, task_id)
            self._validate_investigation_review_protocol(value)
            task_review_session, task_review_turn = self._load_investigation_review_turn_context(
                conn, task_id=task_id, value=value
            )
            result_event_refs = self._collect_investigation_review_result_event_refs(
                conn, task_id=task_id, value=value
            )
            refs = self._prepare_investigation_review_source_refs(
                conn,
                task_id=task_id,
                source_refs=source_refs,
                result_event_refs=result_event_refs,
                task_review_session=task_review_session,
                task_review_turn=task_review_turn,
            )
            existing, event_key = self._find_investigation_review_exact_replay(
                conn, task_id=task_id, value=value, source_refs=refs
            )
            if existing is not None:
                return existing
            self._validate_investigation_review_turn_available(
                conn,
                task_id=task_id,
                task_review_session=task_review_session,
                task_review_turn=task_review_turn,
            )
            return self._append_investigation_batch_review_event(
                conn,
                task_id=task_id,
                value=value,
                source_refs=refs,
                event_key=event_key,
                idempotency_key=event_key.rsplit(":", 1)[-1],
            )

    def record_batch_review(
        self,
        task_id: str,
        payload: Mapping[str, Any] | None = None,
        source_refs: Sequence[str] = (),
        **fields: Any,
    ) -> str:
        """兼容旧装配器，调用 canonical record_investigation_batch_review。"""

        return self.record_investigation_batch_review(
            task_id, payload, source_refs, **fields
        )

    def record_investigation_task_terminal(
        self,
        task_id: str,
        payload: Mapping[str, Any] | None = None,
        source_refs: Sequence[str] = (),
        **fields: Any,
    ) -> str:
        """原子记录 Investigation v2 的任务终态和回答性来源。

        输入是当前 batch 的 accepted/can_answer、最终回答或不可回答说明、完整 Review
        事件及可选的五项 revise_hypothesis marker 事件；动作在同一事务核对五个当前
        session/result、Review 版本和 marker 身份，追加 TaskClosed 事件并更新任务状态。
        返回终态事件 ID；缺少来源、仍有 pending repair、结果过期、重复关闭、legacy
        task 或数据库失败都会回滚，不保存 transcript、token、路径或私有 receipt。
        """

        store = self._require_enabled()
        value = self._coerce_investigation_payload(payload, fields)
        # 终态合同把两项回答正文定义为必需但可为空。关键字合并会忽略 None，
        # 因此在本接口边界补齐键，再由终态合同校验 accepted/can_answer 的组合。
        value.setdefault("final_answer", None)
        value.setdefault("answerability_explanation", None)
        value["task_id"] = task_id
        self._set_investigation_protocol_version(value)
        value = validate_investigation_task_terminal_payload(value)
        with store.immediate_transaction() as conn:
            self._require_investigation_task(conn, task_id)
            required_refs = {
                value["review_event_ref"],
                *value["marker_event_refs"],
            }
            refs = sorted(set(source_refs)) if source_refs else sorted(required_refs)
            if not required_refs.issubset(refs):
                raise ContractValidationError(
                    "TaskClosed source_refs 必须引用当前 Review 及全部 marker"
                )
            current_terminal = conn.execute(
                "SELECT terminal_event_id FROM investigation_task_terminals "
                "WHERE task_id=?",
                (task_id,),
            ).fetchone()
            if current_terminal is not None:
                current_event = conn.execute(
                    "SELECT event_key FROM memory_events "
                    "WHERE task_id=? AND event_id=?",
                    (task_id, current_terminal["terminal_event_id"]),
                ).fetchone()
                if current_event is None:
                    raise ContractValidationError(
                        "当前 TaskClosed 投影缺少不可变事件来源"
                    )
                # 精确重试必须复用当前投影指向的事件键；更正后的新终态不能退回
                # 首次关闭使用的 base key，否则会把新正文误判为改写旧事件。
                terminal_key = current_event["event_key"]
            else:
                terminal_key = event_key_investigation_task_terminal(
                    task_id, value["batch_id"]
                )
                legacy_terminal = conn.execute(
                    "SELECT 1 FROM memory_events WHERE event_key=?",
                    (terminal_key,),
                ).fetchone()
                # 当前终态已被显式更正撤销、但首次 TaskClosed 仍在历史中时，新的
                # Review 身份必须生成追加键；历史事件保持不可变。
                if legacy_terminal is not None:
                    terminal_key = event_key_investigation_task_terminal(
                        task_id,
                        value["batch_id"],
                        value["review_event_ref"],
                    )
            return self._insert_investigation_event(
                conn,
                task_id=task_id,
                event_key=terminal_key,
                event_type="investigation_task_closed",
                round_id=1,
                payload=value,
                source_refs=refs,
                projection=lambda event_id: insert_investigation_task_terminal_projection(
                    conn,
                    task_id=task_id,
                    event_id=event_id,
                    payload=value,
                    created_at=conn.execute(
                        "SELECT created_at FROM memory_events WHERE event_id=?", (event_id,)
                    ).fetchone()[0],
                ),
            )

    def record_investigation_task_closed(
        self,
        task_id: str,
        payload: Mapping[str, Any] | None = None,
        source_refs: Sequence[str] = (),
        **fields: Any,
    ) -> str:
        """兼容旧装配器，调用 canonical record_investigation_task_terminal。"""

        return self.record_investigation_task_terminal(
            task_id, payload, source_refs, **fields
        )

    def record_hypothesis_selection_completed(
        self,
        task_id: str,
        payload: Mapping[str, Any],
        source_refs: Sequence[str],
    ) -> str:
        """原子记录只完成 Top5 筛选的任务终态。

        输入是已发布的五项 BatchCreated 选择、finalized manifest 来源和模型 operation
        身份；动作只在 ``run_mode=top5_selection`` 任务中追加一条幂等终态事实并把
        tasks.status 更新为 ``succeeded``，不会创建 Investigation session、Review 或
        执行事实。重复提交相同正文返回原事件；来源、批次、选择数量或任务模式漂移时
        整体回滚并抛出合同错误。
        """

        store = self._require_enabled()
        value = validate_hypothesis_selection_completed_payload(payload)
        if value["task_id"] != task_id:
            raise ContractValidationError(
                "HypothesisSelectionCompleted.task_id 与参数不一致"
            )
        refs = validate_source_ref_shape(list(source_refs))
        with store.immediate_transaction() as conn:
            task_created = self._task_created_payload(conn, task_id)
            self._require_investigation_task(conn, task_id)
            task_config = task_created["task_config"]
            if task_config.get("run_mode", "end_to_end") != "top5_selection":
                raise ContractValidationError(
                    "只有 run_mode=top5_selection 任务可以记录 Top5 终态"
                )
            batch = conn.execute(
                "SELECT batch_id,batch_index,selected_count,created_event_id,state "
                "FROM hypothesis_batches WHERE task_id=? AND batch_id=?",
                (task_id, value["batch_id"]),
            ).fetchone()
            if batch is None:
                raise ContractValidationError(
                    "HypothesisSelectionCompleted 引用的 batch 不存在"
                )
            if batch["batch_index"] != value["batch_index"]:
                raise ContractValidationError("Top5 终态 batch_index 与 BatchCreated 不一致")
            if batch["selected_count"] != 5:
                raise ContractValidationError("Top5 终态要求 BatchCreated 选择五项")
            if batch["created_event_id"] not in refs:
                raise ContractValidationError(
                    "Top5 终态 source_refs 必须包含 BatchCreated event"
                )
            self._validate_source_refs(conn, task_id, refs)
            selected_rows = conn.execute(
                "SELECT c.combination_id,c.combination_index,c.hypothesis_id,hc.rank "
                "FROM combinations c JOIN hypothesis_candidates hc "
                "ON hc.batch_id=c.batch_id AND hc.combination_id=c.combination_id "
                "WHERE c.task_id=? AND c.batch_id=? AND hc.selection_status='selected' "
                "ORDER BY c.combination_index",
                (task_id, value["batch_id"]),
            ).fetchall()
            selected_projection = [dict(row) for row in selected_rows]
            if selected_projection != value["selected_combinations"]:
                raise ContractValidationError(
                    "Top5 终态选择与 Working Memory 当前投影不一致"
                )
            event_key = event_key_hypothesis_selection_completed(
                task_id,
                value["batch_id"],
            )
            event_id, inserted = self._insert_event(
                conn,
                event_key=event_key,
                task_id=task_id,
                round_id=1,
                event_type="hypothesis_selection_completed",
                payload=value,
                source_refs=refs,
                check_operation_occupancy=False,
            )
            if inserted:
                # 任务终态由这条不可变事实驱动；组合行保持候选投影，但任务已不可写，
                # 因而不会被恢复器误认为可以继续推进 Investigation。
                conn.execute(
                    "UPDATE tasks SET status='succeeded', final_outcome=?, "
                    "version=version+1, updated_at=? WHERE task_id=?",
                    (canonical_json(value), utc_now(), task_id),
                )
            return event_id

    def record_investigation_task_failure(
        self,
        task_id: str,
        *,
        reason_code: str,
        error_type: str,
    ) -> str:
        """记录协调器或 Controller 已确认的 Investigation 执行失败。

        输入是已创建的 Investigation task 身份、固定失败原因和异常类型；动作在一
        个事务中追加幂等的 ``InvestigationTaskFailed`` 事实并把 tasks.status 设为
        ``failed``，只保存安全标识，不保存异常正文、路径、模型输出或凭据。返回事件
        ID；未知 task、终态冲突、标识不安全或数据库失败时抛出合同/并发异常。该接口
        只供协调器或 Controller 在已经决定返回 terminal ``failed`` 时调用，等待
        作业、暂停和真实进程崩溃路径不得调用它。
        """

        store = self._require_enabled()
        payload = validate_investigation_task_failure_payload(
            {
                "task_id": task_id,
                "reason_code": reason_code,
                "error_type": error_type,
            }
        )
        if payload["task_id"] != task_id:
            raise ContractValidationError("InvestigationTaskFailed task_id 与参数不一致")
        with store.immediate_transaction() as conn:
            task = self._require_investigation_task(conn, task_id)
            if task["status"] in {"waiting_execution", "paused"}:
                raise CASConflict(
                    "waiting_execution/paused task 禁止写入 InvestigationTaskFailed",
                    expected_version=task["version"],
                    current_version=task["version"],
                    expected_state="running",
                    current_state=task["status"],
                )
            event_key = event_key_investigation_task_failure(task_id)
            existing = self._existing_event_exact(
                conn,
                event_key=event_key,
                task_id=task_id,
                round_id=1,
                event_type="investigation_task_failed",
                payload=payload,
                source_refs=[],
            )
            if existing is not None:
                return existing
            return self._insert_investigation_event(
                conn,
                task_id=task_id,
                event_key=event_key,
                event_type="investigation_task_failed",
                round_id=1,
                payload=payload,
                source_refs=[],
                projection=lambda event_id: conn.execute(
                    "UPDATE tasks SET status='failed', final_outcome=?, "
                    "version=version+1, updated_at=? WHERE task_id=?",
                    (
                        canonical_json(payload),
                        conn.execute(
                            "SELECT created_at FROM memory_events WHERE event_id=?",
                            (event_id,),
                        ).fetchone()[0],
                        task_id,
                    ),
                ),
            )

    def reopen_investigation_task_for_repair(
        self,
        task_id: str,
        payload: Mapping[str, Any] | None = None,
        source_refs: Sequence[str] = (),
        *,
        terminal_event_ref: str | None = None,
        review_event_ref: str | None = None,
        result_event_ref: str | None = None,
        combination_id: str | None = None,
        repair_advice: str | None = None,
        batch_id: str | None = None,
        result_version: str | None = None,
        operation_instance_id: str | None = None,
        investigation_protocol_version: int | None = None,
        **fields: Any,
    ) -> str:
        """由维护者显式更正一个已接受终态，并恢复同一组合的返修。

        输入必须明确提供（或在 payload 中提供）当前 TaskClosed、Review、Result、组合
        和修复建议；batch、结果版本、operation 与协议号可由当前不可变索引补齐，显式
        提供时逐值核对。动作只在一个事务中追加不可变更正事件，撤销旧终态当前索引，
        并建立同一 session 的 pending repair；旧 Result、Review、TaskClosed 和 marker
        事件永不删除或改写。返回更正事件 ID；身份错误、当前任务不是 accepted
        succeeded、已有 pending、预算耗尽、来源漂移或缺字段时抛出合同错误，不自动
        判断科学结论。
        """

        store = self._require_enabled()
        keyword_fields = {
            "terminal_event_ref": terminal_event_ref,
            "review_event_ref": review_event_ref,
            "result_event_ref": result_event_ref,
            "combination_id": combination_id,
            "repair_advice": repair_advice,
            "batch_id": batch_id,
            "result_version": result_version,
            "operation_instance_id": operation_instance_id,
            "investigation_protocol_version": investigation_protocol_version,
            **fields,
        }
        value = self._coerce_investigation_payload(payload, keyword_fields)
        if "task_id" in value and value["task_id"] != task_id:
            raise ContractValidationError(
                "InvestigationTaskCorrection.task_id 与参数不一致"
            )
        value["task_id"] = task_id
        # 先校验维护请求自身的六个不可缺少的绑定字段；其余运行身份在同一事务中
        # 读取当前索引后补齐，防止调用方用猜测的 batch/version 形成事实。
        value = validate_investigation_task_correction_payload(value)
        with store.immediate_transaction() as conn:
            self._require_investigation_task(conn, task_id)
            event_key = event_key_investigation_task_correction(
                task_id,
                value["terminal_event_ref"],
                value["review_event_ref"],
                value["result_event_ref"],
                value["combination_id"],
            )
            existing_row = conn.execute(
                "SELECT event_id,event_type,round,payload_json,source_refs_json "
                "FROM memory_events WHERE event_key=?",
                (event_key,),
            ).fetchone()
            if existing_row is not None:
                stored = validate_investigation_task_correction_payload(
                    self._decode_json(
                        existing_row["payload_json"],
                        field="InvestigationTaskCorrection.payload",
                    )
                )
                if stored["task_id"] != task_id:
                    raise IdempotencyConflict(
                        f"event_key={event_key} 已绑定到不同 task"
                    )
                for field in (
                    "batch_id",
                    "result_version",
                    "operation_instance_id",
                    "investigation_protocol_version",
                ):
                    if field not in value:
                        value[field] = stored[field]
                value = validate_investigation_task_correction_payload(value)
                stored_refs = self._decode_json(
                    existing_row["source_refs_json"],
                    field="InvestigationTaskCorrection.source_refs",
                )
                stored_refs = validate_source_ref_shape(stored_refs)
                stored_refs = self._validate_source_refs(conn, task_id, stored_refs)
                required_refs = {
                    value["terminal_event_ref"],
                    value["review_event_ref"],
                    value["result_event_ref"],
                }
                if not required_refs.issubset(stored_refs):
                    raise ContractValidationError(
                        "历史更正 source_refs 缺少 TaskClosed、Review 或 Result"
                    )
                self._validate_investigation_correction_source_identity(
                    conn,
                    task_id,
                    value,
                )
                refs = (
                    sorted(set(source_refs))
                    if source_refs
                    else stored_refs
                )
                existing = self._existing_event_exact(
                    conn,
                    event_key=event_key,
                    task_id=task_id,
                    round_id=existing_row["round"],
                    event_type="investigation_task_correction_recorded",
                    payload=value,
                    source_refs=refs,
                )
                if existing is None:  # pragma: no cover - stable key 查询保证存在
                    raise IdempotencyConflict(
                        f"event_key={event_key} 无法重放已有更正事实"
                    )
                return existing

            terminal = conn.execute(
                "SELECT batch_id FROM investigation_task_terminals WHERE task_id=?",
                (task_id,),
            ).fetchone()
            if "batch_id" not in value:
                if terminal is None:
                    raise ContractValidationError(
                        "更正缺少可读取 batch_id 的当前 TaskClosed"
                    )
                value["batch_id"] = terminal["batch_id"]
            if "result_version" not in value:
                result = conn.execute(
                    "SELECT result_version FROM investigation_results "
                    "WHERE task_id=? AND result_event_id=?",
                    (task_id, value["result_event_ref"]),
                ).fetchone()
                if result is None:
                    raise ContractValidationError(
                        "更正缺少可读取 result_version 的当前 Result"
                    )
                value["result_version"] = result["result_version"]
            if "operation_instance_id" not in value:
                digest = event_key.rsplit(":", 1)[-1]
                value["operation_instance_id"] = (
                    f"investigation-correction-{digest[:48]}"
                )
            self._set_investigation_protocol_version(value)
            value = validate_investigation_task_correction_payload(value)
            required_refs = {
                value["terminal_event_ref"],
                value["review_event_ref"],
                value["result_event_ref"],
            }
            refs = sorted(set(source_refs)) if source_refs else sorted(required_refs)
            if not required_refs.issubset(refs):
                raise ContractValidationError(
                    "更正 source_refs 必须引用当前 TaskClosed、Review 和 Result"
                )
            return self._insert_investigation_event(
                conn,
                task_id=task_id,
                event_key=event_key,
                event_type="investigation_task_correction_recorded",
                round_id=1,
                payload=value,
                source_refs=refs,
                projection=lambda correction_event_id: insert_investigation_task_correction_projection(
                    conn,
                    task_id=task_id,
                    event_id=correction_event_id,
                    payload=value,
                    created_at=conn.execute(
                        "SELECT created_at FROM memory_events WHERE event_id=?",
                        (correction_event_id,),
                    ).fetchone()[0],
                ),
            )

    def _validate_investigation_correction_source_identity(
        self,
        conn,
        task_id: str,
        value: Mapping[str, Any],
    ) -> None:
        """在更正重放前复验三条被引用历史事件的类型和业务身份。

        输入是事务连接、任务身份和已经补齐运行字段的更正正文；动作只读取并重新
        校验 TaskClosed、Review、Result 事件，确认批次、组合、版本和相互引用没有
        漂移。返回值为空；来源缺失、类型错误或正文身份变化时抛出合同错误，不改变
        数据库，避免把篡改后的事件当作幂等重放。
        """

        def read_payload(event_ref: str, event_type: str) -> Dict[str, Any]:
            """读取当前任务中指定类型的历史事件正文。"""

            row = conn.execute(
                "SELECT event_type,payload_json FROM memory_events "
                "WHERE task_id=? AND event_id=?",
                (task_id, event_ref),
            ).fetchone()
            if row is None or row["event_type"] != event_type:
                raise ContractValidationError(
                    f"更正引用的来源不是 {event_type}"
                )
            return self._decode_json(
                row["payload_json"],
                field=f"InvestigationTaskCorrection.{event_type}.payload",
            )

        terminal = validate_investigation_task_terminal_payload(
            read_payload(value["terminal_event_ref"], "investigation_task_closed")
        )
        if (
            terminal["task_id"] != task_id
            or terminal["batch_id"] != value["batch_id"]
            or terminal["review_event_ref"] != value["review_event_ref"]
            or not terminal["accepted"]
        ):
            raise ContractValidationError("更正引用的 TaskClosed 身份已漂移")

        review = validate_investigation_batch_review_payload(
            read_payload(value["review_event_ref"], "investigation_batch_reviewed")
        )
        review_vector = dict(
            zip(review["combination_ids"], review["result_versions"])
        )
        if (
            review["task_id"] != task_id
            or review["batch_id"] != value["batch_id"]
            or review_vector.get(value["combination_id"]) != value["result_version"]
        ):
            raise ContractValidationError("更正引用的 Review 身份已漂移")

        result = validate_investigation_result_payload(
            read_payload(value["result_event_ref"], "investigation_result_recorded")
        )
        if (
            result["task_id"] != task_id
            or result["batch_id"] != value["batch_id"]
            or result["combination_id"] != value["combination_id"]
            or result["result_version"] != value["result_version"]
        ):
            raise ContractValidationError("更正引用的 Result 身份已漂移")

    def request_investigation_repair(
        self,
        task_id: str,
        payload: Mapping[str, Any] | None = None,
        source_refs: Sequence[str] = (),
        **fields: Any,
    ) -> str:
        """原子保存 Review 的 fail advice，建立 pending repair 请求。

        输入是当前 Review 的 fail finding、当前 result 版本和请求编号；动作只追加
        request 事实并把组合置为 pending，v8 不在 request 或 Started 时递增
        repair_count，只有对应 repair Result 成功登记才消耗预算。返回 request 事件 ID；
        Review/Result 过期、marker 已写或三次上限超出时失败，fail advice 不被改写。
        """

        store = self._require_enabled()
        value = self._coerce_investigation_payload(payload, fields)
        value["task_id"] = task_id
        self._set_investigation_protocol_version(value)
        if "repair_request_number" not in value:
            value["repair_request_number"] = value.get("repair_count")
        value = validate_investigation_repair_requested_payload(value)
        with store.immediate_transaction() as conn:
            self._require_investigation_task(conn, task_id)
            session = conn.execute(
                "SELECT * FROM investigation_sessions WHERE task_id=? AND batch_id=? AND combination_id=?",
                (task_id, value["batch_id"], value["combination_id"]),
            ).fetchone()
            if session is None:
                raise ContractValidationError("repair request 缺少 session")
            if session["route_marker"] is not None:
                raise ContractValidationError("已写入 revise_hypothesis marker 的组合不能再次 repair")
            prior_result_ref = session["latest_result_event_id"]
            if not isinstance(prior_result_ref, str):
                raise ContractValidationError(
                    "repair request 必须绑定当前已登记的 result"
                )
            if session["latest_review_event_id"] != value["review_event_ref"]:
                raise ContractValidationError("repair request 必须引用组合当前 Review")
            refs = sorted(set(source_refs or [value["review_event_ref"], prior_result_ref]))
            if prior_result_ref not in refs or value["review_event_ref"] not in refs:
                raise ContractValidationError("repair request 必须引用当前 result 和 Review")
            event_key = event_key_investigation_repair_requested(
                task_id,
                value["batch_id"],
                value["combination_id"],
                value["repair_count"],
            )
            return self._insert_investigation_event(
                conn,
                task_id=task_id,
                event_key=event_key,
                event_type="investigation_repair_requested",
                round_id=value["repair_count"],
                payload=value,
                source_refs=refs,
                projection=lambda event_id: insert_investigation_repair_requested_projection(
                    conn,
                    task_id=task_id,
                    event_id=event_id,
                    payload=value,
                    created_at=conn.execute(
                        "SELECT created_at FROM memory_events WHERE event_id=?", (event_id,)
                    ).fetchone()[0],
                ),
            )

    def start_investigation_repair_turn(
        self,
        task_id: str,
        payload: Mapping[str, Any] | None = None,
        source_refs: Sequence[str] = (),
        **fields: Any,
    ) -> str:
        """登记 deepseek_repair turn 已实际成功开始但不消耗返修预算。

        输入是当前 pending request、repair 编号、同一 Investigation session 的
        thread/scope/turn、请求摘要和 operation；动作在一个事务中追加启动事实，只有
        对应新 Result 成功登记时才递增 repair_count。返回 turn 事件 ID；provider 启动前崩溃、重复启动、编号
        越界或身份漂移均失败且不增加预算，数据库不保存 transcript。
        """

        store = self._require_enabled()
        value = self._coerce_investigation_payload(payload, fields)
        value["task_id"] = task_id
        self._set_investigation_protocol_version(value)
        with store.immediate_transaction() as conn:
            self._require_investigation_task(conn, task_id)
            replay_event_id = self._read_replayed_turn_started_event_id(
                conn, task_id=task_id, kind="repair", value=value
            )
            if replay_event_id is not None:
                return replay_event_id
            value = self._prepare_turn_attempt_payload(
                conn,
                task_id=task_id,
                kind="repair",
                value=value,
            )
            value = validate_investigation_repair_turn_started_payload(value)
            session = conn.execute(
                "SELECT * FROM investigation_sessions WHERE task_id=? AND batch_id=? "
                "AND combination_id=?",
                (task_id, value["batch_id"], value["combination_id"]),
            ).fetchone()
            if session is None:
                raise ContractValidationError("Repair turn 缺少 Investigation session")
            if session["pending_repair_event_id"] != value["request_event_ref"]:
                raise ContractValidationError("Repair turn 必须绑定当前 pending request")
            request = conn.execute(
                "SELECT created_at FROM investigation_repairs WHERE task_id=? "
                "AND repair_request_event_id=?",
                (task_id, value["request_event_ref"]),
            ).fetchone()
            if request is None:
                raise ContractValidationError("Repair turn 引用了未知 repair request")
            identity_refs = self._investigation_session_identity_event_refs(
                conn,
                task_id=task_id,
                batch_id=value["batch_id"],
                combination_id=value["combination_id"],
                session_event_id=session["session_event_id"],
            )
            refs = list(source_refs) or [value["request_event_ref"], *identity_refs]
            self._append_previous_attempt_closure_ref(refs, value)
            for required_ref in (value["request_event_ref"], *identity_refs):
                if required_ref not in refs:
                    raise ContractValidationError(
                        "Repair turn source_refs 缺少 request 或 session 身份绑定事件"
                    )
            event_key = event_key_investigation_repair_turn_started(
                task_id, value["request_event_ref"], value["attempt_number"]
            )
            return self._insert_investigation_event(
                conn,
                task_id=task_id,
                event_key=event_key,
                event_type="investigation_repair_turn_started",
                round_id=value["repair_request_number"],
                payload=value,
                source_refs=refs,
                projection=lambda event_id: insert_investigation_repair_turn_started_projection(
                    conn,
                    task_id=task_id,
                    event_id=event_id,
                    payload=value,
                    created_at=conn.execute(
                        "SELECT created_at FROM memory_events WHERE event_id=?", (event_id,)
                    ).fetchone()[0],
                ),
            )

    def complete_investigation_repair(
        self,
        task_id: str,
        payload: Mapping[str, Any] | None = None,
        source_refs: Sequence[str] = (),
        **fields: Any,
    ) -> str:
        """原子核对返修 request、成功 turn 与新 result，并结束 pending repair。

        输入是当前 request、对应 repair turn 和新 Result 事件；动作只在同一事务中
        验证新结果确实引用 request/turn 后清除 pending。返回完成事件 ID；旧 result、
        异批来源、时间戳伪造或缺少因果引用时失败，不改变已有事实。
        """

        store = self._require_enabled()
        value = self._coerce_investigation_payload(payload, fields)
        value["task_id"] = task_id
        self._set_investigation_protocol_version(value)
        value = validate_investigation_repair_completed_payload(value)
        with store.immediate_transaction() as conn:
            self._require_investigation_task(conn, task_id)
            turn_ref = value.get("repair_turn_event_ref")
            if not isinstance(turn_ref, str):
                raise ContractValidationError("repair completed 必须引用 repair turn")
            refs = sorted(set(source_refs or [
                value["request_event_ref"], value["result_event_ref"], turn_ref
            ]))
            if (
                value["request_event_ref"] not in refs
                or value["result_event_ref"] not in refs
                or turn_ref not in refs
            ):
                raise ContractValidationError(
                    "repair completed source_refs 必须引用 request、result 和 repair turn"
                )
            event_key = event_key_investigation_repair_completed(
                task_id, value["request_event_ref"]
            )
            return self._insert_investigation_event(
                conn,
                task_id=task_id,
                event_key=event_key,
                event_type="investigation_repair_completed",
                round_id=value["repair_count"],
                payload=value,
                source_refs=refs,
                projection=lambda event_id: insert_investigation_repair_completed_projection(
                    conn,
                    task_id=task_id,
                    event_id=event_id,
                    payload=value,
                    created_at=conn.execute(
                        "SELECT created_at FROM memory_events WHERE event_id=?", (event_id,)
                    ).fetchone()[0],
                ),
            )

    def mark_revise_hypothesis_route(
        self,
        task_id: str,
        payload: Mapping[str, Any] | None = None,
        source_refs: Sequence[str] = (),
        **fields: Any,
    ) -> str:
        """原子写入只含来源引用的 revise_hypothesis marker。

        输入是组合身份和可选当前 Result/Review 来源；动作在同一事务核对五项 Review
        gate、当前结果和 marker 终态，再追加 marker 事实。返回 marker 事件 ID；marker
        不承载科学原因，来源过期、pending repair、finding 不满足 gate 或重复写入时
        失败，无法修改组合的已标记终态。
        """

        store = self._require_enabled()
        raw = dict(payload or {})
        embedded_refs = raw.pop("source_refs", ())
        if not source_refs:
            source_refs = embedded_refs
        value = self._coerce_investigation_payload(raw, fields)
        value["task_id"] = task_id
        self._set_investigation_protocol_version(value)
        with store.immediate_transaction() as conn:
            self._require_investigation_task(conn, task_id)
            session = conn.execute(
                "SELECT * FROM investigation_sessions WHERE task_id=? AND batch_id=? AND combination_id=?",
                (task_id, value.get("batch_id"), value.get("combination_id")),
            ).fetchone()
            if session is None:
                # 用 validator 生成稳定的“五个组合”前置错误，而不是 SQL 错误。
                value = validate_investigation_route_marker_payload(value)
                raise ContractValidationError("route marker 缺少对应 session")
            if isinstance(session["latest_result_event_id"], str):
                value.setdefault("result_event_ref", session["latest_result_event_id"])
            if isinstance(session["latest_review_event_id"], str):
                value.setdefault("review_event_ref", session["latest_review_event_id"])
            value = validate_investigation_route_marker_payload(value)
            refs = sorted(set(source_refs or [value["result_event_ref"], value["review_event_ref"]]))
            if not value["result_event_ref"] in refs or not value["review_event_ref"] in refs:
                raise ContractValidationError("route marker source_refs 必须引用 result 和 review")
            event_key = event_key_investigation_route_marker(
                task_id, value["batch_id"], value["combination_id"]
            )
            return self._insert_investigation_event(
                conn,
                task_id=task_id,
                event_key=event_key,
                event_type="investigation_route_marked",
                round_id=1,
                payload=value,
                source_refs=refs,
                projection=lambda event_id: insert_investigation_route_marker_projection(
                    conn,
                    task_id=task_id,
                    event_id=event_id,
                    payload=value,
                    created_at=conn.execute(
                        "SELECT created_at FROM memory_events WHERE event_id=?", (event_id,)
                    ).fetchone()[0],
                ),
            )

    def mark_investigation_route_marker(
        self,
        task_id: str,
        payload: Mapping[str, Any] | None = None,
        source_refs: Sequence[str] = (),
        **fields: Any,
    ) -> str:
        """兼容旧装配器，调用 canonical mark_revise_hypothesis_route。"""

        return self.mark_revise_hypothesis_route(
            task_id, payload, source_refs, **fields
        )

    def mark_revise_hypothesis(
        self,
        task_id: str,
        payload: Mapping[str, Any] | None = None,
        source_refs: Sequence[str] = (),
        **fields: Any,
    ) -> str:
        """兼容旧路由调用点，调用 canonical mark_revise_hypothesis_route。"""

        return self.mark_revise_hypothesis_route(
            task_id, payload, source_refs, **fields
        )

    def check_revise_hypothesis_route_gate(
        self,
        task_id: str,
        batch_id: str,
        combination_id: str,
        *,
        result_event_ref: str | None = None,
        review_event_ref: str | None = None,
    ) -> Dict[str, Any]:
        """只读核对 revise_hypothesis 是否已有五项审查和真实结果依据。

        输入是任务、批次、组合以及可选的结果/审查事件身份。方法不写数据库，返回
        ``allowed``、来源身份、审查结论和失败原因；未知 session 或非法身份直接报错，
        尚未具备足够事实时返回 ``allowed=False``，供路由器在写 marker 前先行核对。
        """

        store = self._require_enabled()
        require_nonempty_string(task_id, "task_id")
        require_nonempty_string(batch_id, "batch_id")
        require_nonempty_string(combination_id, "combination_id")
        for field, value in (("result_event_ref", result_event_ref), ("review_event_ref", review_event_ref)):
            if value is not None:
                require_nonempty_string(value, field)
        with store.transaction() as conn:
            self._require_investigation_task(conn, task_id)
            session = conn.execute(
                "SELECT * FROM investigation_sessions WHERE task_id=? AND batch_id=? "
                "AND combination_id=?",
                (task_id, batch_id, combination_id),
            ).fetchone()
            if session is None:
                raise ContractValidationError("未知 Investigation session")
            batch_gate = evaluate_investigation_batch_route_gate(
                conn,
                task_id=task_id,
                batch_id=batch_id,
            )
            current_result_ref = session["latest_result_event_id"]
            current_review_ref = session["latest_review_event_id"]
            response: Dict[str, Any] = {
                "allowed": bool(batch_gate["allowed"]),
                "marker": "revise_hypothesis",
                "task_id": task_id,
                "batch_id": batch_id,
                "combination_id": combination_id,
                "result_event_ref": current_result_ref,
                "review_event_ref": current_review_ref,
                "repair_count": session["repair_count"],
                "result_version": session["current_result_version"],
                "validity": None,
                "hypothesis_judgment": None,
                "reason": batch_gate["reason"],
            }
            if result_event_ref is not None and result_event_ref != current_result_ref:
                response["allowed"] = False
                response["reason"] = "marker 来源必须正好对应当前 result 和 review"
            if review_event_ref is not None and review_event_ref != current_review_ref:
                response["allowed"] = False
                response["reason"] = "marker 来源必须正好对应当前 result 和 review"
            finding = next(
                (
                    item
                    for item in batch_gate["findings"]
                    if item["combination_id"] == combination_id
                ),
                None,
            )
            if finding is not None:
                response["validity"] = finding["validity"]
                response["hypothesis_judgment"] = finding["hypothesis_judgment"]
            if finding is None and response["allowed"]:
                response["allowed"] = False
                response["reason"] = "当前 Review 缺少指定组合 finding"
            return response

    def validate_revise_hypothesis_route_gate(
        self,
        task_id: str,
        batch_id: str,
        combination_id: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """核对并要求 revise_hypothesis gate 已满足，失败时不写入任何事实。"""

        result = self.check_revise_hypothesis_route_gate(
            task_id,
            batch_id,
            combination_id,
            **kwargs,
        )
        if not result["allowed"]:
            raise ContractValidationError(result["reason"] or "route marker gate 未满足")
        return result

    def check_investigation_route_marker_gate(
        self,
        task_id: str,
        batch_id: str,
        combination_id: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """兼容旧读取调用点，调用 canonical check_revise_hypothesis_route_gate。"""

        return self.check_revise_hypothesis_route_gate(
            task_id, batch_id, combination_id, **kwargs
        )

    def validate_investigation_route_marker_gate(
        self,
        task_id: str,
        batch_id: str,
        combination_id: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """兼容旧校验调用点，调用 canonical validate_revise_hypothesis_route_gate。"""

        return self.validate_revise_hypothesis_route_gate(
            task_id, batch_id, combination_id, **kwargs
        )

    @staticmethod
    def _read_revise_hypothesis_batch_gate_rows(
        conn,
        task_id: str,
        batch_id: str,
    ) -> tuple[list[sqlite3.Row], list[sqlite3.Row], list[sqlite3.Row]]:
        """读取批次组合、marker 和 session 的当前索引行。"""

        combinations = conn.execute(
            "SELECT c.combination_id,c.investigation_phase,"
            "c.investigation_ready_state,c.investigation_stage_state "
            "FROM combinations c JOIN hypothesis_candidates hc "
            "ON hc.batch_id=c.batch_id AND hc.combination_id=c.combination_id "
            "WHERE c.task_id=? AND c.batch_id=? AND hc.selection_status='selected' "
            "ORDER BY c.combination_index,c.combination_id",
            (task_id, batch_id),
        ).fetchall()
        markers = conn.execute(
            "SELECT combination_id,marker_event_id,result_event_ref,review_event_ref "
            "FROM investigation_route_markers WHERE task_id=? AND batch_id=? "
            "ORDER BY combination_id",
            (task_id, batch_id),
        ).fetchall()
        sessions = conn.execute(
            "SELECT combination_id,route_marker,phase,latest_result_event_id,"
            "latest_review_event_id,pending_repair_event_id "
            "FROM investigation_sessions WHERE task_id=? AND batch_id=?",
            (task_id, batch_id),
        ).fetchall()
        return combinations, markers, sessions

    @staticmethod
    def _format_revise_hypothesis_marker_rows(
        markers: Sequence[sqlite3.Row],
    ) -> list[Dict[str, Any]]:
        """把 marker 索引行转换成 route gate 的稳定公开摘要。"""

        return [
            {
                "combination_id": row["combination_id"],
                "marker_event_id": row["marker_event_id"],
                "result_event_ref": row["result_event_ref"],
                "review_event_ref": row["review_event_ref"],
            }
            for row in markers
        ]

    @staticmethod
    def _validate_revise_hypothesis_marker_rows(
        combination_ids: Sequence[str],
        combinations: Sequence[sqlite3.Row],
        markers: Sequence[sqlite3.Row],
        sessions: Sequence[sqlite3.Row],
    ) -> str | None:
        """核对当前批次 marker、组合阶段和 session 当前来源是否逐项一致。"""

        marker_ids = [row["combination_id"] for row in markers]
        expected_ids = set(combination_ids)
        if len(markers) not in {1, 5} or set(marker_ids) != expected_ids:
            return "入选组合必须各有一个 revise_hypothesis marker"
        if any(
            row["investigation_phase"] != "marked_for_hypothesis_revision"
            or row["investigation_ready_state"] != "marked_for_hypothesis_revision"
            or row["investigation_stage_state"] != "marked_for_hypothesis_revision"
            for row in combinations
        ):
            return "marker 与组合当前阶段不一致"
        return WorkingMemoryService._validate_revise_hypothesis_marker_bindings(
            combination_ids, expected_ids, markers, sessions
        )

    @staticmethod
    def _validate_revise_hypothesis_marker_bindings(
        combination_ids: Sequence[str],
        expected_ids: set[str],
        markers: Sequence[sqlite3.Row],
        sessions: Sequence[sqlite3.Row],
    ) -> str | None:
        """核对当前批次 session 的 marker 阶段、pending 和当前来源。"""

        session_by_id = {row["combination_id"]: row for row in sessions}
        if len(sessions) not in {1, 5} or set(session_by_id) != expected_ids:
            return "入选组合必须各有一个当前 Investigation session"
        marker_by_id = {row["combination_id"]: row for row in markers}
        for combination_id in combination_ids:
            session = session_by_id[combination_id]
            marker = marker_by_id[combination_id]
            if (
                session["route_marker"] != "revise_hypothesis"
                or session["phase"] != "marked_for_hypothesis_revision"
                or session["pending_repair_event_id"] is not None
                or marker["result_event_ref"] != session["latest_result_event_id"]
                or marker["review_event_ref"] != session["latest_review_event_id"]
            ):
                return "marker 必须对应组合当前 result 和终态"
        return None

    @staticmethod
    def _collect_revise_hypothesis_next_batch_marker_ids(
        combination_ids: Sequence[str],
        sessions: Sequence[sqlite3.Row],
        markers: Sequence[sqlite3.Row],
    ) -> list[str]:
        """核对下一批来源 marker，并返回当前批次 marker 事件身份。"""

        expected_ids = set(combination_ids)
        session_by_id = {row["combination_id"]: row for row in sessions}
        marker_by_id = {row["combination_id"]: row for row in markers}
        if (
            len(sessions) not in {1, 5}
            or len(markers) != len(sessions)
            or set(session_by_id) != expected_ids
            or set(marker_by_id) != expected_ids
        ):
            raise ContractValidationError(
                "上一批必须为当前组合各自写入 revise_hypothesis marker"
            )
        marker_event_ids: list[str] = []
        for combination_id in combination_ids:
            session = session_by_id[combination_id]
            marker = marker_by_id[combination_id]
            if (
                session["route_marker"] != "revise_hypothesis"
                or session["phase"] != "marked_for_hypothesis_revision"
                or session["pending_repair_event_id"] is not None
                or marker["result_event_ref"] != session["latest_result_event_id"]
                or marker["review_event_ref"] != session["latest_review_event_id"]
            ):
                raise ContractValidationError(
                    "上一批 marker 必须对应每个组合当前 result/review"
                )
            marker_event_ids.append(marker["marker_event_id"])
        return marker_event_ids

    def check_revise_hypothesis_batch_gate(
        self,
        task_id: str,
        batch_id: str,
    ) -> Dict[str, Any]:
        """只读核对当前批次入选组合是否都已完成合法路由标记。

        输入是任务和批次身份。方法读取当前入选组合及其 marker 事件，返回 ``allowed``、
        组合顺序和每项 marker/result/review 事件身份；没有组合、只有部分 marker、重复或
        异批身份均返回 ``allowed=False``，不修改数据库，未知批次直接报合同错误。
        """

        store = self._require_enabled()
        require_nonempty_string(task_id, "task_id")
        require_nonempty_string(batch_id, "batch_id")
        with store.transaction() as conn:
            self._require_investigation_task(conn, task_id)
            batch = conn.execute(
                "SELECT batch_id,state FROM hypothesis_batches WHERE task_id=? AND batch_id=?",
                (task_id, batch_id),
            ).fetchone()
            if batch is None:
                raise ContractValidationError("未知 Investigation batch")
            # 只有不可变事件重建结果与当前索引逐值一致时，marker 才能授权下一步；
            # marker_event_id 等内部来源被篡改时必须 fail-closed，而不是信任缓存行。
            self._assert_investigation_index_matches_event_rebuild(conn, task_id)
            combinations, markers, sessions = (
                self._read_revise_hypothesis_batch_gate_rows(
                    conn, task_id, batch_id
                )
            )
            combination_ids = [row["combination_id"] for row in combinations]
            batch_gate = evaluate_investigation_batch_route_gate(
                conn,
                task_id=task_id,
                batch_id=batch_id,
            )
            response: Dict[str, Any] = {
                "allowed": False,
                "task_id": task_id,
                "batch_id": batch_id,
                "combination_ids": combination_ids,
                "markers": [],
                "review_event_ref": batch_gate["review_event_ref"],
                "result_version_vector": batch_gate["result_version_vector"],
                "reason": batch_gate["reason"],
            }
            if len(combinations) not in {1, 5} or len(set(combination_ids)) != len(combination_ids):
                response["reason"] = "当前批次必须恰有一个或五个入选组合"
                return response
            response["markers"] = self._format_revise_hypothesis_marker_rows(markers)
            if not batch_gate["allowed"]:
                return response
            response["reason"] = self._validate_revise_hypothesis_marker_rows(
                combination_ids, combinations, markers, sessions
            )
            if response["reason"] is not None:
                return response
            response["allowed"] = True
            return response

    def validate_revise_hypothesis_batch_gate(
        self,
        task_id: str,
        batch_id: str,
    ) -> Dict[str, Any]:
        """核对并要求当前批次已经形成完整的 revise_hypothesis 标记。"""

        result = self.check_revise_hypothesis_batch_gate(task_id, batch_id)
        if not result["allowed"]:
            raise ContractValidationError(result["reason"] or "批次 route marker gate 未满足")
        return result

    def check_investigation_batch_route_gate(
        self,
        task_id: str,
        batch_id: str,
    ) -> Dict[str, Any]:
        """兼容旧读取调用点，调用 canonical check_revise_hypothesis_batch_gate。"""

        return self.check_revise_hypothesis_batch_gate(task_id, batch_id)

    def validate_investigation_batch_route_gate(
        self,
        task_id: str,
        batch_id: str,
    ) -> Dict[str, Any]:
        """兼容旧校验调用点，调用 canonical validate_revise_hypothesis_batch_gate。"""

        return self.validate_revise_hypothesis_batch_gate(task_id, batch_id)

    # ==================== Investigation v2 事实读取 ====================

    def get_investigation_session(
        self,
        task_id: str,
        batch_id: str,
        combination_id: str,
        *,
        include_internal: bool = False,
    ) -> Dict[str, Any]:
        """读取一个 session；只有恢复代码显式请求时才返回内部身份。"""

        store = self._require_enabled()
        row = store.fetchone(
            "SELECT * FROM investigation_sessions WHERE task_id=? AND batch_id=? AND combination_id=?",
            (task_id, batch_id, combination_id),
        )
        if row is None:
            raise ContractValidationError("未知 Investigation session")
        return investigation_session_row(row, include_internal=include_internal)

    def list_investigation_sessions(
        self,
        task_id: str,
        batch_id: str,
        *,
        include_internal: bool = False,
    ) -> List[Dict[str, Any]]:
        """按组合顺序读取当前批次 session 索引。"""

        store = self._require_enabled()
        rows = store.fetchall(
            "SELECT * FROM investigation_sessions WHERE task_id=? AND batch_id=? ORDER BY combination_id",
            (task_id, batch_id),
        )
        return [investigation_session_row(row, include_internal=include_internal) for row in rows]

    def _read_investigation_recovery_rows(
        self,
        conn,
        task_id: str,
        batch_id: str,
    ) -> Dict[str, Any]:
        """读取已完成索引核对后的 Investigation 恢复事实集合。

        输入是事务连接、任务身份和已经确认存在的批次；动作按稳定业务顺序读取
        session、turn、repair、Result、Review、marker 和任务终态，并保留恢复所需
        的内部 thread/workspace/scope 身份。返回值只来自当前批次的数据库投影，缺失
        行不会被环境或任务目录内容补全。
        """

        sessions = [
            investigation_session_row(row, include_internal=True)
            for row in conn.execute(
                "SELECT * FROM investigation_sessions WHERE task_id=? AND batch_id=? "
                "ORDER BY combination_id",
                (task_id, batch_id),
            ).fetchall()
        ]
        turns = [
            investigation_turn_row(row, include_internal=True)
            for row in conn.execute(
                "SELECT * FROM investigation_turns WHERE task_id=? AND batch_id=? "
                "ORDER BY combination_id,created_at,turn_started_event_id",
                (task_id, batch_id),
            ).fetchall()
        ]
        repair_turns = [
            investigation_repair_turn_row(row, include_internal=True)
            for row in conn.execute(
                "SELECT * FROM investigation_repair_turns WHERE task_id=? AND batch_id=? "
                "ORDER BY combination_id,repair_request_number,turn_started_event_id",
                (task_id, batch_id),
            ).fetchall()
        ]
        turn_waits = [
            investigation_turn_wait_row(row, include_internal=True)
            for row in conn.execute(
                "SELECT * FROM investigation_turn_waits WHERE task_id=? AND batch_id=? "
                "ORDER BY combination_id,created_at,waiting_event_id",
                (task_id, batch_id),
            ).fetchall()
        ]
        results = [
            investigation_result_row(row, include_internal=True)
            for row in conn.execute(
                "SELECT * FROM investigation_results WHERE task_id=? AND batch_id=? "
                "ORDER BY combination_id,created_at,result_event_id",
                (task_id, batch_id),
            ).fetchall()
        ]
        reviews = [
            investigation_review_row(row, include_internal=True)
            for row in conn.execute(
                "SELECT * FROM investigation_reviews WHERE task_id=? AND batch_id=? "
                "ORDER BY created_at,review_event_id",
                (task_id, batch_id),
            ).fetchall()
        ]
        repairs = [
            investigation_repair_row(row, include_internal=True)
            for row in conn.execute(
                "SELECT * FROM investigation_repairs WHERE task_id=? AND batch_id=? "
                "ORDER BY combination_id,repair_count",
                (task_id, batch_id),
            ).fetchall()
        ]
        markers = [
            investigation_route_marker_row(row)
            for row in conn.execute(
                "SELECT * FROM investigation_route_markers WHERE task_id=? AND batch_id=? "
                "ORDER BY combination_id",
                (task_id, batch_id),
            ).fetchall()
        ]
        terminal = conn.execute(
            "SELECT * FROM investigation_task_terminals WHERE task_id=?",
            (task_id,),
        ).fetchone()
        failure_rows = conn.execute(
            "SELECT event_id,created_at,payload_json FROM memory_events "
            "WHERE task_id=? AND event_type='investigation_task_failed' "
            "ORDER BY created_at,event_id",
            (task_id,),
        ).fetchall()
        if len(failure_rows) > 1:
            raise ContractValidationError("InvestigationTaskFailed 事件不得重复")
        failure = None
        if failure_rows:
            failure = validate_investigation_task_failure_payload(
                self._decode_json(
                    failure_rows[0]["payload_json"],
                    field="InvestigationTaskFailed.payload",
                )
            )
            if failure["task_id"] != task_id:
                raise ContractValidationError(
                    "InvestigationTaskFailed task_id 与恢复任务不一致"
                )
            failure["event_id"] = failure_rows[0]["event_id"]
            failure["created_at"] = failure_rows[0]["created_at"]
        return {
            "sessions": sessions,
            "turns": turns,
            "repair_turns": repair_turns,
            "turn_waits": turn_waits,
            "results": results,
            "reviews": reviews,
            "repairs": repairs,
            "markers": markers,
            "terminal": (
                investigation_task_terminal_row(terminal)
                if terminal is not None and terminal["batch_id"] == batch_id
                else None
            ),
            "failure": failure,
        }

    def get_investigation_recovery_snapshot(
        self,
        task_id: str,
        batch_id: str,
    ) -> Dict[str, Any]:
        """读取经过事件重建核对的 Investigation 恢复快照。

        输入是 task 和 batch 身份；动作先验证 TaskCreated 与 tasks 当前行，再在同一
        事务保存稳定业务键快照、按不可变事件重建当前索引并逐值比较，随后返回恢复
        所需的 session、turn、Result、Review、repair、marker 和终态事实。当前索引被
        篡改、事件因果不完整或批次未知时失败并回滚；这是内部恢复专用接口，保留
        thread/workspace/scope 等安全身份但不返回 transcript、token 或原始路径文本。
        """

        store = self._require_enabled()
        require_nonempty_string(task_id, "task_id")
        require_nonempty_string(batch_id, "batch_id")
        with store.immediate_transaction() as conn:
            self._require_investigation_task(conn, task_id)
            batch = conn.execute(
                "SELECT 1 FROM hypothesis_batches WHERE task_id=? AND batch_id=?",
                (task_id, batch_id),
            ).fetchone()
            if batch is None:
                raise ContractValidationError("未知 Investigation batch")
            before = self._read_investigation_index_snapshot(conn, task_id)
            rebuild_task_projections(
                conn,
                task_id,
                task_root=self._task_output_root(conn, task_id),
            )
            after = self._read_investigation_index_snapshot(conn, task_id)
            if before != after:
                raise ContractValidationError(
                    "Investigation 当前索引未经事件验证，不能作为恢复快照"
                )
            rows = self._read_investigation_recovery_rows(conn, task_id, batch_id)
            return {
                "task_id": task_id,
                "batch_id": batch_id,
                **rows,
            }

    def get_investigation_result(
        self,
        task_id: str,
        result_event_id: str,
        *,
        include_internal: bool = False,
    ) -> Dict[str, Any]:
        """按不可变 result event 身份读取一版科学事实。"""

        store = self._require_enabled()
        row = store.fetchone(
            "SELECT * FROM investigation_results WHERE task_id=? AND result_event_id=?",
            (task_id, result_event_id),
        )
        if row is None:
            raise ContractValidationError("未知 Investigation result")
        return investigation_result_row(row, include_internal=include_internal)

    def list_investigation_results(
        self,
        task_id: str,
        batch_id: str,
        *,
        include_internal: bool = False,
    ) -> List[Dict[str, Any]]:
        """按组合和版本顺序读取 Investigation 结果历史。"""

        store = self._require_enabled()
        rows = store.fetchall(
            "SELECT * FROM investigation_results WHERE task_id=? AND batch_id=? "
            "ORDER BY combination_id,created_at,result_event_id",
            (task_id, batch_id),
        )
        return [investigation_result_row(row, include_internal=include_internal) for row in rows]

    def get_investigation_review(
        self,
        task_id: str,
        review_event_id: str,
        *,
        include_internal: bool = False,
    ) -> Dict[str, Any]:
        """读取一条完整 Review 事实，默认隐藏 Review thread/turn。"""

        store = self._require_enabled()
        row = store.fetchone(
            "SELECT * FROM investigation_reviews WHERE task_id=? AND review_event_id=?",
            (task_id, review_event_id),
        )
        if row is None:
            raise ContractValidationError("未知 Investigation Review")
        return investigation_review_row(row, include_internal=include_internal)

    @staticmethod
    def _require_investigation_replay_batch(
        conn,
        *,
        task_id: str,
        batch_id: str,
    ) -> None:
        """确认 replay 请求的 batch 已由 BatchCreated 事实建立。

        输入是事务连接、task 和 batch 身份；动作只查询不可变批次投影，返回值为空，
        未知 batch 时抛出合同错误，避免把未知对象误报为合法 replay miss。
        """

        if conn.execute(
            "SELECT 1 FROM hypothesis_batches WHERE task_id=? AND batch_id=?",
            (task_id, batch_id),
        ).fetchone() is None:
            raise ContractValidationError("未知 Investigation batch")

    @staticmethod
    def _load_investigation_replay_review_row(
        conn,
        *,
        task_id: str,
        batch_id: str,
    ) -> Any:
        """读取指定批次最近一条 Review 当前索引。

        输入是事务连接、task 和已确认 batch；动作按创建时间和事件身份取最新 Review，
        返回数据库行或 None，不修改当前索引。
        """

        return conn.execute(
            "SELECT * FROM investigation_reviews WHERE task_id=? AND batch_id=? "
            "ORDER BY created_at DESC,review_event_id DESC LIMIT 1",
            (task_id, batch_id),
        ).fetchone()

    def _validate_investigation_replay_session_vector(
        self,
        *,
        sessions: Sequence[Any],
        vector: Sequence[Sequence[str]],
        review_event_id: str,
    ) -> None:
        """核对当前 replay session 仍覆盖请求向量且 Review 仍为最新。

        输入是 session 行、规范化当前批次向量和 Review 事件身份；动作只比较版本、pending
        和最新 Review 指针，返回值为空，索引不完整或过期时抛出合同错误。
        """

        current_vector = sorted(
            [item["combination_id"], item["current_result_version"]]
            for item in sessions
        )
        if (
            len(sessions) not in {1, 5}
            or len(sessions) != len(vector)
            or current_vector != vector
            or any(item["pending_repair_event_id"] is not None for item in sessions)
            or any(item["latest_review_event_id"] != review_event_id for item in sessions)
        ):
            raise ContractValidationError(
                "replay Review 必须覆盖当前 session 的 result 且仍为最新"
            )

    @staticmethod
    def _validate_investigation_replay_result_rows(
        conn,
        *,
        task_id: str,
        batch_id: str,
        sessions: Sequence[Any],
    ) -> None:
        """逐项核对 replay session 指向的 Result 事件身份和版本。

        输入是事务连接、task、batch 和已通过向量校验的五个 session；动作读取每个
        latest Result 及事件类型，返回值为空，缺失、异批或版本漂移时抛出合同错误。
        """

        for session in sessions:
            result_ref = session["latest_result_event_id"]
            if not isinstance(result_ref, str) or not result_ref:
                raise ContractValidationError("replay session 缺少 latest_result_event_id")
            result = conn.execute(
                "SELECT r.result_event_id,r.batch_id,r.combination_id,"
                "r.result_version,e.event_type "
                "FROM investigation_results AS r "
                "LEFT JOIN memory_events AS e ON e.task_id=r.task_id "
                "AND e.event_id=r.result_event_id "
                "WHERE r.task_id=? AND r.result_event_id=?",
                (task_id, result_ref),
            ).fetchone()
            if (
                result is None
                or result["event_type"] != "investigation_result_recorded"
                or result["batch_id"] != batch_id
                or result["combination_id"] != session["combination_id"]
                or result["result_version"] != session["current_result_version"]
            ):
                raise ContractValidationError(
                    "replay session 的 latest Result identity/version 不一致"
                )

    def _read_investigation_review_replay_in_transaction(
        self,
        conn,
        *,
        task_id: str,
        batch_id: str,
        vector: Sequence[Sequence[str]],
        include_internal: bool,
    ) -> Optional[Dict[str, Any]]:
        """在同一事务中完成 Review replay 的 batch、索引和 Result 核对。

        输入是已打开事务、task、batch、规范化向量和内部字段开关；动作先验证 batch
        存在并按事件重建当前索引，再读取并逐项核对 Review。合法的无 Review 或向量不匹配
        返回 None，损坏或过期索引抛出合同错误，不访问 provider 或私有正文。
        """

        self._require_investigation_replay_batch(
            conn, task_id=task_id, batch_id=batch_id
        )
        self._assert_investigation_index_matches_event_rebuild(conn, task_id)
        row = self._load_investigation_replay_review_row(
            conn, task_id=task_id, batch_id=batch_id
        )
        if row is None:
            return None
        stored_vector = validate_investigation_result_version_vector(
            self._decode_json(
                row["result_version_vector_json"],
                field="InvestigationReview.result_version_vector_json",
            )
        )
        if stored_vector != vector:
            return None
        sessions = conn.execute(
            "SELECT combination_id,current_result_version,latest_review_event_id,"
            "latest_result_event_id,pending_repair_event_id FROM investigation_sessions "
            "WHERE task_id=? AND batch_id=? ORDER BY combination_id",
            (task_id, batch_id),
        ).fetchall()
        self._validate_investigation_replay_session_vector(
            sessions=sessions,
            vector=vector,
            review_event_id=row["review_event_id"],
        )
        self._validate_investigation_replay_result_rows(
            conn,
            task_id=task_id,
            batch_id=batch_id,
            sessions=sessions,
        )
        return investigation_review_row(row, include_internal=include_internal)

    def get_investigation_review_for_replay(
        self,
        task_id: str,
        batch_id: str,
        result_version_vector: Any,
        *,
        include_internal: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """按完整五项当前 Result 向量读取可供模型重放的 Review。

        输入是 task、batch 和五项组合/版本向量；动作在一个写事务中先按不可变事件
        重建并逐值核对当前索引，再校验 Review、五个 session 和每个 latest Result
        的身份/版本。没有已保存 Review，或请求向量与已保存 Review 不同，返回 None；
        向量缺项、存在 pending repair、索引或事件不一致、正文损坏时抛出合同错误。
        成功时返回已保存 Review 的安全副本；默认隐藏 thread/turn 等内部恢复身份，
        不访问 provider、transcript、token 或路径。
        """

        store = self._require_enabled()
        require_nonempty_string(task_id, "task_id")
        require_nonempty_string(batch_id, "batch_id")
        with store.immediate_transaction() as conn:
            self._require_investigation_task(conn, task_id)
            task_created = self._task_created_payload(conn, task_id)
            expected_count = get_investigation_combination_count(
                task_created["task_config"]
            )
            vector = validate_investigation_result_version_vector(result_version_vector)
            if len(vector) != expected_count:
                detail = (
                    "（标准任务要求完整五项）"
                    if expected_count == 5
                    else "（作者实验任务要求一项）"
                )
                raise ContractValidationError(
                    "Review replay 向量数量与 TaskCreated 冻结数量不一致" + detail
                )
            return self._read_investigation_review_replay_in_transaction(
                conn,
                task_id=task_id,
                batch_id=batch_id,
                vector=vector,
                include_internal=include_internal,
            )

    def list_investigation_reviews(
        self,
        task_id: str,
        batch_id: str,
        *,
        include_internal: bool = False,
    ) -> List[Dict[str, Any]]:
        """按持久顺序读取批次 Review 历史。"""

        store = self._require_enabled()
        rows = store.fetchall(
            "SELECT * FROM investigation_reviews WHERE task_id=? AND batch_id=? ORDER BY created_at,review_event_id",
            (task_id, batch_id),
        )
        return [investigation_review_row(row, include_internal=include_internal) for row in rows]

    def list_investigation_repairs(
        self,
        task_id: str,
        batch_id: str,
        *,
        include_internal: bool = False,
    ) -> List[Dict[str, Any]]:
        """按组合和返修次数读取返修请求/完成索引。"""

        store = self._require_enabled()
        rows = store.fetchall(
            "SELECT * FROM investigation_repairs WHERE task_id=? AND batch_id=? "
            "ORDER BY combination_id,repair_count",
            (task_id, batch_id),
        )
        return [investigation_repair_row(row, include_internal=include_internal) for row in rows]

    def list_investigation_route_markers(
        self,
        task_id: str,
        batch_id: str,
    ) -> List[Dict[str, Any]]:
        """读取只含事实来源引用的 revise_hypothesis marker。"""

        store = self._require_enabled()
        rows = store.fetchall(
            "SELECT * FROM investigation_route_markers WHERE task_id=? AND batch_id=? ORDER BY combination_id",
            (task_id, batch_id),
        )
        return [investigation_route_marker_row(row) for row in rows]

    def list_investigation_turns(
        self,
        task_id: str,
        batch_id: str,
        *,
        include_internal: bool = False,
    ) -> List[Dict[str, Any]]:
        """读取首轮 Investigation turn 的 in-flight 游标，不返回 prompt 或 transcript。

        输入是 task、batch 身份和是否查看 thread/scope/operation 的开关；动作只读由
        ``InvestigationTurnStarted`` 事件重建的当前索引，按组合与启动时间排序。返回
        turn 字典列表；未知索引或数据库损坏抛出异常，不修改任何事实。
        """

        store = self._require_enabled()
        rows = store.fetchall(
            "SELECT * FROM investigation_turns WHERE task_id=? AND batch_id=? "
            "ORDER BY combination_id,created_at,turn_started_event_id",
            (task_id, batch_id),
        )
        return [investigation_turn_row(row, include_internal=include_internal) for row in rows]

    def list_investigation_turn_waits(
        self,
        task_id: str,
        batch_id: str,
        *,
        include_internal: bool = False,
    ) -> List[Dict[str, Any]]:
        """读取 initial/repair waiting 事实历史，不删除已消费的旧等待记录。

        输入是 task、batch 身份和是否查看恢复内部字段的开关；动作只读由不可变
        ``InvestigationTurnWaiting`` 事件重建的历史索引，按组合、attempt 和事件排序。
        返回等待事实副本；Review 不会出现在此列表，数据库损坏时抛出异常且不修改事实。
        """

        store = self._require_enabled()
        rows = store.fetchall(
            "SELECT * FROM investigation_turn_waits WHERE task_id=? AND batch_id=? "
            "ORDER BY combination_id,attempt_number,created_at,waiting_event_id",
            (task_id, batch_id),
        )
        return [
            investigation_turn_wait_row(row, include_internal=include_internal)
            for row in rows
        ]

    def list_investigation_turn_attempt_terminations(
        self,
        task_id: str,
        batch_id: str,
        *,
        include_internal: bool = False,
    ) -> List[Dict[str, Any]]:
        """读取指定批次的 turn attempt closure 历史。

        输入是 task/batch 身份；动作只读取已由 Started 事实生成的终止投影，并返回
        作业恢复所需的安全身份和 ``termination_event_id``，不返回 transcript、路径或
        凭据。数据库索引缺失或任务身份未知时失败，不修改任何状态。
        """

        store = self._require_enabled()
        require_nonempty_string(task_id, "task_id")
        require_nonempty_string(batch_id, "batch_id")
        with store.transaction() as conn:
            self._require_investigation_task(conn, task_id)
            rows = conn.execute(
                "SELECT termination_event_id,task_id,turn_started_event_ref,kind,batch_id,"
                "combination_id,turn_id,attempt_number,terminal_status,"
                "logical_operation_instance_id,attempt_operation_instance_id,"
                "previous_attempt_closure_event_ref,created_at "
                "FROM investigation_turn_attempt_terminations "
                "WHERE task_id=? AND batch_id=? "
                "ORDER BY combination_id,kind,attempt_number,created_at,termination_event_id",
                (task_id, batch_id),
            ).fetchall()
            result: list[Dict[str, Any]] = []
            for row in rows:
                item = dict(row)
                if not include_internal:
                    # 这些字段仍是安全的恢复身份，但默认视图隐藏 operation 细节，
                    # 与其它 Investigation list API 的公开/内部分层保持一致。
                    item.pop("logical_operation_instance_id", None)
                    item.pop("attempt_operation_instance_id", None)
                    item.pop("previous_attempt_closure_event_ref", None)
                result.append(item)
            return result

    def get_investigation_task_terminal(
        self,
        task_id: str,
        *,
        include_internal: bool = False,
    ) -> Dict[str, Any]:
        """读取任务唯一 Investigation 终态及其 Review/marker 来源。

        输入是 task 身份和是否查看 operation 内部字段的开关；动作只读由 TaskClosed
        事件重建的当前索引，不访问 provider、路径或 transcript。返回终态字典；尚未
        关闭的 task、未知索引或数据库损坏抛出合同/数据库异常，不产生写副作用。
        """

        store = self._require_enabled()
        row = store.fetchone(
            "SELECT * FROM investigation_task_terminals WHERE task_id=?", (task_id,)
        )
        if row is None:
            raise ContractValidationError("未知 Investigation task terminal")
        value = investigation_task_terminal_row(row)
        if include_internal:
            # 当前 converter 默认隐藏 operation；终态只把 operation 作为恢复索引，
            # 不将 provider transcript 或其他私有字段暴露给模型。
            value["operation_instance_id"] = row["operation_instance_id"]
        return value

    def get_hypothesis_selection_completed(
        self,
        task_id: str,
        *,
        include_internal: bool = False,
    ) -> Dict[str, Any]:
        """读取只完成 Top5 筛选的唯一终态事实。

        输入是任务身份；动作只读取并校验不可变
        ``hypothesis_selection_completed`` 事件及其成功任务状态，不访问 provider、
        MiniChem 或 Investigation 索引。任务尚未完成筛选时抛出合同异常；重复或损坏
        的事实、模式漂移和状态不一致也会失败，避免恢复器重新提交已完成模型操作。
        """

        store = self._require_enabled()
        require_nonempty_string(task_id, "task_id")
        with store.transaction() as conn:
            task = conn.execute(
                "SELECT status FROM tasks WHERE task_id=?",
                (task_id,),
            ).fetchone()
            if task is None:
                raise ContractValidationError(f"未知 task_id：{task_id}")
            rows = conn.execute(
                "SELECT event_id,created_at,payload_json,source_refs_json "
                "FROM memory_events WHERE task_id=? "
                "AND event_type='hypothesis_selection_completed' "
                "ORDER BY created_at,event_id",
                (task_id,),
            ).fetchall()
            if not rows:
                raise ContractValidationError("任务尚无 HypothesisSelectionCompleted 终态")
            if len(rows) != 1:
                raise ContractValidationError(
                    "HypothesisSelectionCompleted 事件不得重复"
                )
            payload = validate_hypothesis_selection_completed_payload(
                self._decode_json(
                    rows[0]["payload_json"],
                    field="HypothesisSelectionCompleted.payload",
                )
            )
            if payload["task_id"] != task_id:
                raise ContractValidationError(
                    "HypothesisSelectionCompleted.task_id 与查询任务不一致"
                )
            created = self._task_created_payload(conn, task_id)
            if created["task_config"].get("run_mode", "end_to_end") != "top5_selection":
                raise ContractValidationError(
                    "HypothesisSelectionCompleted 只能属于 top5_selection 任务"
                )
            if task["status"] != "succeeded":
                raise ContractValidationError(
                    "HypothesisSelectionCompleted 存在但任务状态不是 succeeded"
                )
            value = dict(payload)
            if include_internal:
                value["event_id"] = rows[0]["event_id"]
                value["created_at"] = rows[0]["created_at"]
                value["source_refs"] = self._decode_json(
                    rows[0]["source_refs_json"],
                    field="HypothesisSelectionCompleted.source_refs",
                )
            return value

    def get_investigation_task_failure(
        self,
        task_id: str,
        *,
        include_internal: bool = False,
    ) -> Dict[str, Any] | None:
        """读取任务唯一的安全失败事实，兼容 TaskCreated 启动失败来源。

        输入是 task 身份和是否查看事件身份的开关；动作只读并校验
        ``investigation_task_failed`` 事件及其 ``tasks.status``，没有该事件时再核对
        TaskCreated 的 ``expert_backend_selection.status=startup_failed``。返回安全的
        task、reason_code、error_type 映射；任务尚无失败事实返回 ``None``，重复事件、
        状态不一致、来源损坏或数据库错误抛出合同异常，不写入事实、不访问模型或路径。
        """

        store = self._require_enabled()
        require_nonempty_string(task_id, "task_id")
        with store.transaction() as conn:
            task_row = conn.execute(
                "SELECT status FROM tasks WHERE task_id=?",
                (task_id,),
            ).fetchone()
            if task_row is None:
                raise ContractValidationError(f"未知 task_id：{task_id}")
            failure_rows = conn.execute(
                "SELECT event_id,created_at,payload_json FROM memory_events "
                "WHERE task_id=? AND event_type='investigation_task_failed' "
                "ORDER BY created_at,event_id",
                (task_id,),
            ).fetchall()
            if len(failure_rows) > 1:
                raise ContractValidationError("InvestigationTaskFailed 事件不得重复")
            if failure_rows:
                row = failure_rows[0]
                payload = validate_investigation_task_failure_payload(
                    self._decode_json(
                        row["payload_json"],
                        field="InvestigationTaskFailed.payload",
                    )
                )
                if payload["task_id"] != task_id:
                    raise ContractValidationError(
                        "InvestigationTaskFailed task_id 与查询任务不一致"
                    )
                if task_row["status"] != "failed":
                    raise ContractValidationError(
                        "InvestigationTaskFailed 存在但 tasks.status 不是 failed"
                    )
                value: Dict[str, Any] = dict(payload)
                if include_internal:
                    value["event_id"] = row["event_id"]
                    value["created_at"] = row["created_at"]
                return value
            created_row = conn.execute(
                "SELECT event_id,created_at,payload_json FROM memory_events "
                "WHERE task_id=? AND event_type='task_created'",
                (task_id,),
            ).fetchone()
            if created_row is None:
                if task_row["status"] == "failed":
                    raise ContractValidationError("failed task 缺少可核对的失败来源")
                return None
            created = validate_task_created_payload(
                self._decode_json(
                    created_row["payload_json"],
                    field="TaskCreated.payload",
                )
            )
            selection = created.get("expert_backend_selection")
            if not (
                isinstance(selection, Mapping)
                and selection.get("status") == "startup_failed"
            ):
                if task_row["status"] == "failed":
                    raise ContractValidationError("failed task 缺少可核对的失败来源")
                return None
            if task_row["status"] != "failed":
                raise ContractValidationError(
                    "TaskCreated startup_failed 来源与 tasks.status 不一致"
                )
            value = {
                "task_id": task_id,
                "reason_code": "expert_model_startup_failed",
                "error_type": "ExpertModelStartupFailure",
            }
            if include_internal:
                value["event_id"] = created_row["event_id"]
                value["created_at"] = created_row["created_at"]
                value["source_event_type"] = "task_created"
            return value

    def get_task_review_session(
        self,
        task_id: str,
        *,
        include_internal: bool = False,
    ) -> Dict[str, Any]:
        """读取 task 唯一 Review 会话，并按开关隐藏恢复身份。

        输入是 task 身份和是否供恢复代码查看内部字段的开关；动作只读会话当前索引，
        默认隐藏 workspace、CODEX_HOME、scope、thread 和策略摘要；返回会话状态与
        turn 游标字典。未知 task/session 或数据库错误抛出合同/数据库异常，不产生
        写副作用，也不返回 transcript。
        """

        store = self._require_enabled()
        row = store.fetchone(
            "SELECT * FROM task_review_sessions WHERE task_id=?", (task_id,)
        )
        if row is None:
            raise ContractValidationError("未知 TaskReviewSession")
        return task_review_session_row(row, include_internal=include_internal)

    def list_task_review_turns(
        self,
        task_id: str,
        *,
        include_internal: bool = False,
    ) -> List[Dict[str, Any]]:
        """按 turn_index 读取 task Review 会话游标，不返回 transcript。

        输入是 task 身份和是否允许内部恢复字段的开关；动作只读并排序已成功启动的
        turn，默认隐藏 thread、请求摘要和 operation；返回 turn 字典列表。未知 task、
        索引损坏或数据库失败抛出异常，不修改任何事实。
        """

        store = self._require_enabled()
        rows = store.fetchall(
            "SELECT * FROM task_review_turns WHERE task_id=? "
            "ORDER BY turn_index,turn_started_event_id",
            (task_id,),
        )
        return [task_review_turn_row(row, include_internal=include_internal) for row in rows]

    def list_investigation_evidence_verifications(
        self,
        task_id: str,
        batch_id: str,
        *,
        include_internal: bool = False,
    ) -> List[Dict[str, Any]]:
        """读取已核验的安全证据摘要，不访问 MiniChem 或私有文件。

        输入是 task、batch 身份和是否查看内部恢复字段的开关；动作只读已由窄合同
        保存的作业/操作/artifact 摘要，默认隐藏 thread、scope、turn 与 operation；
        返回按组合和验证对象排序的字典列表。数据库或 JSON 损坏时抛出异常，不扫描
        工作区、不读取 transcript，也不产生写副作用。
        """

        store = self._require_enabled()
        rows = store.fetchall(
            "SELECT * FROM investigation_evidence_verifications "
            "WHERE task_id=? AND batch_id=? "
            "ORDER BY combination_id,verification_object_id,verification_event_id",
            (task_id, batch_id),
        )
        return [
            investigation_evidence_row(row, include_internal=include_internal)
            for row in rows
        ]

    def list_investigation_repair_turns(
        self,
        task_id: str,
        batch_id: str,
        *,
        include_internal: bool = False,
    ) -> List[Dict[str, Any]]:
        """按组合和返修编号读取真正启动的 repair turn。

        输入是 task、batch 身份和是否查看内部 thread/scope/摘要的开关；动作只读已
        成功启动的 turn 记录，默认隐藏恢复身份；返回排序后的 turn 字典列表。未知
        索引、数据库错误或结构损坏会抛出异常，不修改 repair_count，也不返回 transcript。
        """

        store = self._require_enabled()
        rows = store.fetchall(
            "SELECT * FROM investigation_repair_turns WHERE task_id=? AND batch_id=? "
            "ORDER BY combination_id,repair_request_number,turn_started_event_id",
            (task_id, batch_id),
        )
        return [
            investigation_repair_turn_row(row, include_internal=include_internal)
            for row in rows
        ]

    def rebuild_projections(self, task_id: str) -> None:
        """删除可丢弃的当前状态，并完全依据历史事实重新计算。

        该入口既验证缓存不是事实来源，也可在缓存损坏后恢复任务。
        """

        store = self._require_enabled()
        with store.immediate_transaction() as conn:
            # 重建前逐条复验事件所引用的 finalized 证据。当前索引计算随后只读取这些
            # 已获数据库授权的路径，不会扫描目录把 orphan 文件升级成事实。
            event_rows = conn.execute(
                "SELECT source_refs_json FROM memory_events WHERE task_id=?",
                (task_id,),
            ).fetchall()
            for event in event_rows:
                refs = self._decode_json(
                    event["source_refs_json"], field="source_refs_json"
                )
                self._validate_source_refs(conn, task_id, refs)
            revision_rows = conn.execute(
                "SELECT source_refs_json FROM revisions WHERE task_id=?",
                (task_id,),
            ).fetchall()
            for revision in revision_rows:
                refs = self._decode_json(
                    revision["source_refs_json"],
                    field="revision.source_refs_json",
                )
                self._validate_source_refs(conn, task_id, refs)
            rebuild_task_projections(
                conn,
                task_id,
                task_root=self._task_output_root(conn, task_id),
            )

    @staticmethod
    def _read_investigation_index_snapshot(conn, task_id: str) -> tuple[Any, ...]:
        """读取按业务主键排序的 Investigation 当前视图快照。

        输入是事务连接和 task 身份；动作委托内部索引模块按稳定业务键读取，不使用
        SQLite rowid；返回可比较 tuple，索引缺失或结构漂移时直接失败，避免把拓扑
        等价的重放误报为数据损坏。
        """

        return read_investigation_index_snapshot(conn, task_id)

    def _assert_investigation_index_matches_event_rebuild(
        self,
        conn,
        task_id: str,
    ) -> None:
        """核对 Investigation 当前索引与不可变事件的完整语义重建结果。

        输入是已打开的写事务连接和 task 身份；动作保存所有当前 Investigation
        视图的稳定业务字段，按不可变事件重建，再逐值比较前后快照。当前索引被篡改、
        事件正文损坏或因果记录不完整时抛出合同错误并由外层事务回滚；成功时不返回
        数据，也不访问 provider、transcript 或外部路径。
        """

        before = self._read_investigation_index_snapshot(conn, task_id)
        rebuild_task_projections(
            conn,
            task_id,
            task_root=self._task_output_root(conn, task_id),
        )
        after = self._read_investigation_index_snapshot(conn, task_id)
        if before != after:
            raise ContractValidationError(
                "Investigation 当前索引与不可变事件重算结果不一致"
            )

    def verify_investigation_current_index(self, task_id: str) -> bool:
        """重算并逐值核对 Investigation 当前索引，成功返回 True，漂移则回滚并报错。

        输入是任务身份。方法在一个写事务中保存重算前快照、按不可变事件重算，再比较
        session/result/review/repair/marker 与组合阶段字段；任何当前索引被篡改或无法
        重算都会回滚，不返回不可信的恢复状态。
        """

        store = self._require_enabled()
        require_nonempty_string(task_id, "task_id")
        with store.immediate_transaction() as conn:
            before = self._read_investigation_index_snapshot(conn, task_id)
            rebuild_task_projections(
                conn,
                task_id,
                task_root=self._task_output_root(conn, task_id),
            )
            after = self._read_investigation_index_snapshot(conn, task_id)
            if before != after:
                raise ContractValidationError(
                    "Investigation 当前索引与不可变事件重算结果不一致"
                )
        return True
