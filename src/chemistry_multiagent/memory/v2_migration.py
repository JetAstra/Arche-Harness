"""把 schema v10 的 ``investigation_v2`` 事实导入新的 V2 深账本。

迁移器是一个独立的深模块：调用方只需要提供 task 身份和一个只读的 Harness
runtime resolver，内部负责读取旧不可变事件、构造完整的内存计划、核对科学记录
图，并在同一 SQLite 事务中追加新合同事件。旧投影、Run Bundle 和当前默认配置
永远不是迁移事实来源；``legacy_workflow_v1`` 在构造新版账本之前就会被拒绝。
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable, Iterator, Mapping, Sequence
import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sqlite3
from typing import Any, Protocol, TypeAlias

from .contracts import (
    ContractValidationError,
    canonical_json,
)
from .v2_ledger import (
    ResearchTaskLedger,
    ModuleInvocationRequest,
    PreparedModuleInvocationOutput,
    ScientificRecord,
    WorkingMemoryIntegrityConflict,
    _prepared_output_digest,
)
from .v2_identities import (
    ContextBinding,
    CreateInvestigationContext,
    CreateResearchTaskReviewSession,
    ReuseInvestigationContext,
    ReuseResearchTaskReviewSession,
    StatelessModule,
)


MIGRATION_VERSION = "investigation_v2.v10-to-ledger.v1"
_SUPPORTED_SOURCE_SCHEMA = 10
_SOURCE_EVENT_TYPES = frozenset(
    {
        "task_created",
        "retrieval_completed",
        "hypothesis_batch_created",
        "hypothesis_selection_completed",
        "investigation_session_started",
        "investigation_session_thread_bound",
        "investigation_session_thread_migrated",
        "investigation_turn_started",
        "investigation_turn_waiting",
        "investigation_turn_attempt_terminated",
        "investigation_legacy_waiting_resume_authorized",
        "investigation_incomplete_v2_waiting_resume_authorized",
        "investigation_evidence_verified",
        "investigation_result_recorded",
        "task_review_session_started",
        "task_review_turn_started",
        "investigation_batch_reviewed",
        "investigation_repair_requested",
        "investigation_repair_turn_started",
        "investigation_repair_completed",
        "investigation_route_marked",
        "investigation_task_closed",
        "investigation_task_failed",
        "investigation_task_correction_recorded",
    }
)
_INTERNAL_SOURCE_EVENT_TYPES = _SOURCE_EVENT_TYPES - {
    "task_created",
    "hypothesis_batch_created",
    "hypothesis_selection_completed",
    "investigation_evidence_verified",
    "investigation_result_recorded",
    "investigation_batch_reviewed",
    "investigation_repair_requested",
    "investigation_route_marked",
    "investigation_task_closed",
    "investigation_task_failed",
    "investigation_task_correction_recorded",
}
_TERMINAL_SOURCE_EVENT_TYPES = frozenset(
    {"investigation_task_closed", "investigation_task_failed"}
)
_TARGET_RECORD_SCHEMA_VERSION = "1"
_MIGRATED_RECORD_SCHEMA_VERSION = "v10-migrated"


class MigrationError(ContractValidationError):
    """迁移无法安全证明目标事实时抛出的基类错误。"""


class MigrationUnsupported(MigrationError):
    """旧事实缺少目标合同所需的确定性信息。"""


class MigrationSourceDrift(MigrationError):
    """同一迁移版本看到的旧不可变 source digest 发生变化。"""


class MigrationRecoveryEvidenceMissing(MigrationError):
    """Open Harness/Review runtime 的恢复证据无法核验。"""


class MigrationIntegrityConflict(MigrationError):
    """旧事件的引用、顺序或分支无法组成合法目标图。"""


@dataclass(frozen=True)
class ExistingHarnessRuntime(Mapping[str, Any]):
    """只读 resolver 返回的既有 Harness 身份。

    ``runtime_ref`` 必须由 manifest 或其他不可变来源明确提供；迁移器不会从
    thread、scope、workspace、job 或 ``harness_run_ref`` 反推它。``state`` 只用于
    诊断，不会被写成 V2 的 ``running`` 状态。
    """

    runtime_ref: str
    harness_run_ref: str | None = None
    state: str = "open"
    prepared_output: PreparedModuleInvocationOutput | Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.runtime_ref, str) or not self.runtime_ref.strip():
            raise MigrationRecoveryEvidenceMissing("runtime resolver 未提供明确 runtime_ref")
        if self.harness_run_ref is not None and (
            not isinstance(self.harness_run_ref, str) or not self.harness_run_ref.strip()
        ):
            raise MigrationRecoveryEvidenceMissing("harness_run_ref 不是有效不透明引用")
        if self.state not in {"open", "suspended", "completed", "failed"}:
            raise MigrationRecoveryEvidenceMissing(
                f"runtime resolver 返回未注册状态：{self.state}"
            )

    def to_dict(self) -> dict[str, Any]:
        """返回不含输出正文的稳定 runtime 身份。"""

        return {
            "runtime_ref": self.runtime_ref,
            "harness_run_ref": self.harness_run_ref,
            "state": self.state,
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return 3


class HarnessRuntimeResolver(Protocol):
    """迁移期间只读核验既有 Harness runtime 的接口。"""

    def resolve(
        self,
        research_task_id: str,
        *,
        source_event_ids: Sequence[str],
        expected_module_kind: str,
        expected_module_input_digest: str,
    ) -> ExistingHarnessRuntime | Mapping[str, Any]:
        """读取 manifest/恢复证据；不得 start、resume 或 submit。"""


RuntimeResolver: TypeAlias = HarnessRuntimeResolver | Callable[..., Any]


@dataclass(frozen=True)
class MigrationResult(Mapping[str, Any]):
    """迁移完成后返回的稳定结果摘要。"""

    research_task_id: str
    migration_version: str
    source_digest: str
    source_event_count: int
    target_event_ids: tuple[str, ...]
    target_scientific_record_ids: tuple[str, ...]
    target_head_digest: str
    migrated: bool = True

    def to_dict(self) -> dict[str, Any]:
        """返回不包含 SQLite 行的公开摘要。"""

        return {
            "research_task_id": self.research_task_id,
            "migration_version": self.migration_version,
            "source_digest": self.source_digest,
            "source_event_count": self.source_event_count,
            "target_event_ids": list(self.target_event_ids),
            "target_scientific_record_ids": list(self.target_scientific_record_ids),
            "target_head_digest": self.target_head_digest,
            "migrated": self.migrated,
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


@dataclass(frozen=True)
class MigrationPlan(Mapping[str, Any]):
    """全部目标事件和映射已经在内存中核对后的计划。"""

    research_task_id: str
    migration_version: str
    source_digest: str
    source_event_count: int
    task_payload: Mapping[str, Any]
    events: tuple[Mapping[str, Any], ...]
    event_map: tuple[Mapping[str, Any], ...]
    scientific_record_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """返回计划的只读可审计摘要。"""

        return {
            "research_task_id": self.research_task_id,
            "migration_version": self.migration_version,
            "source_digest": self.source_digest,
            "source_event_count": self.source_event_count,
            "task_payload": json.loads(canonical_json(self.task_payload)),
            "events": json.loads(canonical_json(list(self.events))),
            "event_map": json.loads(canonical_json(list(self.event_map))),
            "scientific_record_ids": list(self.scientific_record_ids),
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


@dataclass(frozen=True)
class _SourceEvent:
    event_id: str
    event_key: str
    task_id: str
    round_id: int
    event_type: str
    payload: Mapping[str, Any]
    source_refs: tuple[str, ...]
    created_at: str


@dataclass(frozen=True)
class _SourceSnapshot:
    task_id: str
    task_row: Mapping[str, Any]
    task_created: _SourceEvent
    events: tuple[_SourceEvent, ...]
    by_id: Mapping[str, _SourceEvent]
    source_digest: str


@dataclass(frozen=True)
class _TargetRecord:
    source_event_id: str
    record: ScientificRecord
    event_id: str
    recorded_at: str


@dataclass(frozen=True)
class _TargetEvent:
    event_id: str
    event_type: str
    payload: Mapping[str, Any]
    recorded_at: str
    source_event_ids: tuple[str, ...]
    module_invocation_id: str | None = None
    record: _TargetRecord | None = None


@dataclass
class _TargetModule:
    module_invocation_id: str
    module_kind: str
    input_scientific_record_ids: tuple[str, ...]
    module_input_digest: str
    context_binding: ContextBinding
    source_anchor: str
    records: list[_TargetRecord] = field(default_factory=list)
    closed: bool = True
    failure_code: str | None = None


@dataclass(frozen=True)
class _ResultInfo:
    source_event: _SourceEvent
    batch_id: str
    combination_id: str
    result_version: str
    evidence_event_id: str
    predecessor_event_id: str | None
    repair_request_event_id: str | None


def _require_text(value: Any, field_name: str) -> str:
    """校验迁移源中的身份文本，不做默认填充。"""

    if not isinstance(value, str) or not value.strip():
        raise MigrationIntegrityConflict(f"{field_name} 必须是非空字符串")
    return value.strip()


def _as_object(value: Any, field_name: str) -> dict[str, Any]:
    """校验严格 JSON object 并复制，避免修改 source 快照。"""

    if not isinstance(value, Mapping):
        raise MigrationIntegrityConflict(f"{field_name} 必须是 JSON object")
    try:
        copied = json.loads(canonical_json(dict(value)))
    except ContractValidationError as exc:
        raise MigrationIntegrityConflict(f"{field_name} 不是严格 JSON") from exc
    if not isinstance(copied, dict):  # pragma: no cover
        raise MigrationIntegrityConflict(f"{field_name} 必须是 JSON object")
    return copied


def _as_list(value: Any, field_name: str) -> list[Any]:
    """校验严格 JSON array 并复制。"""

    if not isinstance(value, list):
        raise MigrationIntegrityConflict(f"{field_name} 必须是 JSON array")
    try:
        copied = json.loads(canonical_json(value))
    except ContractValidationError as exc:
        raise MigrationIntegrityConflict(f"{field_name} 不是严格 JSON") from exc
    if not isinstance(copied, list):  # pragma: no cover
        raise MigrationIntegrityConflict(f"{field_name} 必须是 JSON array")
    return copied


def _parse_json(value: Any, field_name: str) -> Any:
    """解析旧表 JSON；损坏时立即停止迁移。"""

    try:
        parsed = json.loads(value)
    except (TypeError, ValueError) as exc:
        raise MigrationIntegrityConflict(f"{field_name} 已损坏") from exc
    try:
        return json.loads(canonical_json(parsed))
    except ContractValidationError as exc:
        raise MigrationIntegrityConflict(f"{field_name} 不是严格 JSON") from exc


def _normalise_timestamp(value: Any, field_name: str) -> str:
    """把旧事件时间规范为稳定 UTC ISO 文本。"""

    text = _require_text(value, field_name)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise MigrationIntegrityConflict(f"{field_name} 不是 ISO-8601 时间") from exc
    if parsed.tzinfo is None:
        raise MigrationIntegrityConflict(f"{field_name} 必须包含时区")
    return parsed.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _stable_id(prefix: str, migration_version: str, task_id: str, *parts: Any) -> str:
    """由固定迁移版本和 source 身份生成不透明、跨进程稳定 ID。"""

    body = {
        "prefix": prefix,
        "migration_version": migration_version,
        "task_id": task_id,
        "parts": list(parts),
    }
    digest = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
    return f"v2-migration-{prefix}-{digest[:40]}"


def _input_digest(research_task_id: str, module_kind: str, refs: Sequence[str]) -> str:
    """计算与 Controller 公共模块输入合同一致的稳定摘要。"""

    body = {
        "research_task_id": research_task_id,
        "module_kind": module_kind,
        "input_scientific_record_ids": list(refs),
    }
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


def _is_event_ref(value: str, events: Mapping[str, _SourceEvent]) -> bool:
    """判断 source ref 是否是本 task 的事件身份。"""

    return value in events


class InvestigationV2Migration:
    """读取 v10 source 并一次性导入 V2 ledger。"""

    def __init__(
        self,
        db_path: str | os.PathLike[str],
        *,
        runtime_resolver: RuntimeResolver | None = None,
        migration_version: str = MIGRATION_VERSION,
    ) -> None:
        if not isinstance(db_path, (str, os.PathLike)):
            raise ContractValidationError("migration db_path 必须是路径")
        self.db_path = os.path.abspath(os.path.expanduser(os.fspath(db_path)))
        self.runtime_resolver = runtime_resolver
        self.migration_version = _require_text(migration_version, "migration_version")

    def migrate_existing_task(
        self,
        task_id: str,
        *,
        source_root: str | os.PathLike[str] | None = None,
    ) -> MigrationResult:
        """迁移一个既有 investigation_v2 task，或精确重放已完成迁移。

        ``source_root`` 只作为 resolver 的只读上下文传递；迁移器本身不读取 Run
        Bundle，也不会创建、恢复或提交 Harness/MiniChem 作业。
        """

        task_id = _require_text(task_id, "task_id")
        # 第一个连接只读 source。legacy 判定和完整 MigrationPlan 都在构造
        # InvestigationV2Ledger 之前完成，因而失败路径不会创建 v2_* 表。
        with self._connection() as connection:
            snapshot = self._read_source_snapshot(connection, task_id)
            marker = self._read_marker(connection, task_id)
            if marker is not None:
                return self._replay_completed_migration(snapshot, marker)
            self._reject_partial_target(connection, task_id)
            plan = self._build_plan(snapshot, source_root=source_root)

        # 计划完全通过 source、科学图和模块成组合同核对后才构造新版账本。
        ledger = ResearchTaskLedger(self.db_path)
        with ledger._transaction() as connection:  # 迁移专用单事务写入 seam
            current = self._read_source_snapshot(connection, task_id)
            if current.source_digest != plan.source_digest:
                raise MigrationSourceDrift(
                    f"task {task_id} 在迁移写入前发生 source digest 漂移"
                )
            marker = self._read_marker(connection, task_id)
            if marker is not None:
                return self._replay_completed_migration(current, marker)
            self._reject_partial_target(connection, task_id)
            self._write_plan(connection, ledger, plan)
            return self._result_from_plan(plan)

    def _connection(self) -> sqlite3.Connection:
        """建立只读阶段使用的短连接。"""

        connection = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _read_source_snapshot(
        self,
        connection: sqlite3.Connection,
        task_id: str,
    ) -> _SourceSnapshot:
        """只读取 v10 的 TaskCreated 和 memory_events，不读取 projection。"""

        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if not {"tasks", "memory_events", "schema_meta"}.issubset(tables):
            raise MigrationUnsupported("source 不是受支持的 schema v10 Working Memory")
        schema_row = connection.execute(
            "SELECT value FROM schema_meta WHERE key='schema_version'"
        ).fetchone()
        if schema_row is None or str(schema_row[0]) != str(_SUPPORTED_SOURCE_SCHEMA):
            raise MigrationUnsupported("迁移输入必须是 schema v10")
        task_row = connection.execute(
            "SELECT * FROM tasks WHERE task_id=?", (task_id,)
        ).fetchone()
        if task_row is None:
            raise MigrationUnsupported(f"source task {task_id} 不存在")
        raw_rows = connection.execute(
            "SELECT event_id,event_key,task_id,round,event_type,payload_json,"
            "source_refs_json,created_at FROM memory_events WHERE task_id=?",
            (task_id,),
        ).fetchall()
        if not raw_rows:
            raise MigrationIntegrityConflict(f"task {task_id} 缺少 memory_events")
        events: list[_SourceEvent] = []
        for row in raw_rows:
            event_id = _require_text(row["event_id"], "memory_events.event_id")
            event_type = _require_text(row["event_type"], f"event {event_id}.event_type")
            if event_type not in _SOURCE_EVENT_TYPES:
                raise MigrationUnsupported(
                    f"task {task_id} 含迁移器未注册的 source event {event_type}"
                )
            payload = _as_object(
                _parse_json(row["payload_json"], f"event {event_id}.payload_json"),
                f"event {event_id}.payload",
            )
            refs_value = _parse_json(
                row["source_refs_json"], f"event {event_id}.source_refs_json"
            )
            refs = tuple(_require_text(item, f"event {event_id}.source_refs[]") for item in _as_list(refs_value, "source_refs"))
            created_at = _normalise_timestamp(
                row["created_at"], f"event {event_id}.created_at"
            )
            events.append(
                _SourceEvent(
                    event_id=event_id,
                    event_key=_require_text(row["event_key"], f"event {event_id}.event_key"),
                    task_id=_require_text(row["task_id"], f"event {event_id}.task_id"),
                    round_id=int(row["round"]),
                    event_type=event_type,
                    payload=payload,
                    source_refs=refs,
                    created_at=created_at,
                )
            )
        by_id = {event.event_id: event for event in events}
        if len(by_id) != len(events):
            raise MigrationIntegrityConflict(f"task {task_id} 含重复 source event_id")
        if any(event.task_id != task_id for event in events):
            raise MigrationIntegrityConflict(f"task {task_id} 的 source event 跨 task")
        # 事件引用必须指向本 task；外部 manifest 路径等非事件来源会保留在
        # 科学 payload 中，但不会被伪装成目标 Scientific Record 引用。
        for event in events:
            for ref in event.source_refs:
                if ref in by_id:
                    continue
                foreign = connection.execute(
                    "SELECT task_id FROM memory_events WHERE event_id=?", (ref,)
                ).fetchone()
                if foreign is not None and foreign[0] != task_id:
                    raise MigrationIntegrityConflict(
                        f"event {event.event_id} 引用了另一个 task 的 source event"
                    )
        task_events = [event for event in events if event.event_type == "task_created"]
        if len(task_events) != 1:
            raise MigrationIntegrityConflict(f"task {task_id} 必须恰有一个 TaskCreated")
        task_created = task_events[0]
        if task_created.payload.get("stage_architecture_version") != "investigation_v2":
            # 这里不能调用旧 validator，因为旧 validator 会对缺失 model 配置
            # 填当前默认值，违反迁移只读取冻结事实的合同。
            raise MigrationUnsupported(
                "legacy_workflow_v1 task 在构造新版 ledger 前被拒绝"
            )
        if str(task_row["stage_architecture_version"]) != "investigation_v2":
            raise MigrationIntegrityConflict(
                "tasks.stage_architecture_version 与 TaskCreated 不一致"
            )
        task_payload = task_created.payload
        for required in (
            "scientific_question",
            "chemistry_context",
            "task_config",
            "task_output_relative_dir",
            "tool_catalog_hash",
        ):
            if required not in task_payload:
                raise MigrationUnsupported(f"TaskCreated 缺少冻结字段 {required}")
        _require_text(task_payload["scientific_question"], "TaskCreated.scientific_question")
        _as_object(task_payload["chemistry_context"], "TaskCreated.chemistry_context")
        _as_object(task_payload["task_config"], "TaskCreated.task_config")
        _require_text(task_payload["task_output_relative_dir"], "TaskCreated.task_output_relative_dir")
        _require_text(task_payload["tool_catalog_hash"], "TaskCreated.tool_catalog_hash")
        ordered = self._topologically_order_events(events, by_id, task_id)
        digest_body = {
            "schema_version": _SUPPORTED_SOURCE_SCHEMA,
            "task_id": task_id,
            "task_created": task_created.payload,
            "events": [
                {
                    "event_id": event.event_id,
                    "event_key": event.event_key,
                    "task_id": event.task_id,
                    "round": event.round_id,
                    "event_type": event.event_type,
                    "payload": event.payload,
                    "source_refs": list(event.source_refs),
                    "created_at": event.created_at,
                }
                for event in ordered
            ],
        }
        digest = hashlib.sha256(canonical_json(digest_body).encode("utf-8")).hexdigest()
        return _SourceSnapshot(
            task_id=task_id,
            task_row=dict(task_row),
            task_created=task_created,
            events=tuple(ordered),
            by_id=by_id,
            source_digest=digest,
        )

    @staticmethod
    def _topologically_order_events(
        events: Sequence[_SourceEvent],
        by_id: Mapping[str, _SourceEvent],
        task_id: str,
    ) -> list[_SourceEvent]:
        """按 source 引用拓扑排序，时间和 event_id 只作为稳定并列顺序。"""

        # TaskCreated 是目标账本的唯一根，即使旧 fixture 的时钟早于任务写入时间，
        # 也必须先于所有可达事件；其余并列项再按冻结时间和 event_id 排序。
        order_key = lambda event: (
            0 if event.event_type == "task_created" else 1,
            event.created_at,
            event.event_id,
        )
        indegree = {event.event_id: 0 for event in events}
        children: dict[str, list[str]] = defaultdict(list)
        for event in events:
            for ref in event.source_refs:
                if ref not in by_id:
                    continue
                indegree[event.event_id] += 1
                children[ref].append(event.event_id)
        ready = [event for event in events if indegree[event.event_id] == 0]
        ready.sort(key=order_key)
        result: list[_SourceEvent] = []
        queue: list[_SourceEvent] = ready
        while queue:
            current = queue.pop(0)
            result.append(current)
            for child_id in sorted(children.get(current.event_id, ()), key=lambda item: order_key(by_id[item])):
                indegree[child_id] -= 1
                if indegree[child_id] == 0:
                    queue.append(by_id[child_id])
            queue.sort(key=order_key)
        if len(result) != len(events):
            raise MigrationIntegrityConflict(f"task {task_id} 的 source event 引用存在循环")
        if result[0].event_type != "task_created":
            raise MigrationIntegrityConflict(f"task {task_id} 的 TaskCreated 不是 source 首事件")
        return result

    @staticmethod
    def _read_marker(
        connection: sqlite3.Connection,
        task_id: str,
    ) -> sqlite3.Row | None:
        """读取迁移 marker；表不存在时保持只读空结果。"""

        table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='v2_migration_markers'"
        ).fetchone()
        if table is None:
            return None
        return connection.execute(
            "SELECT * FROM v2_migration_markers WHERE task_id=?", (task_id,)
        ).fetchone()

    @staticmethod
    def _reject_partial_target(connection: sqlite3.Connection, task_id: str) -> None:
        """拒绝没有 marker 的同 task V2 残留，避免猜测半迁移状态。"""

        table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='v2_tasks'"
        ).fetchone()
        if table is None:
            return
        row = connection.execute(
            "SELECT 1 FROM v2_tasks WHERE task_id=?", (task_id,)
        ).fetchone()
        if row is not None:
            raise MigrationIntegrityConflict(
                f"task {task_id} 已有 V2 目标行但缺少完成 marker，拒绝修补"
            )

    def _replay_completed_migration(
        self,
        snapshot: _SourceSnapshot,
        marker: sqlite3.Row,
    ) -> MigrationResult:
        """校验 marker 后返回既有目标身份，不追加任何事件。"""

        if marker["migration_version"] != self.migration_version:
            raise MigrationSourceDrift("同一 task 使用了不同 migration_version")
        if marker["source_digest"] != snapshot.source_digest:
            raise MigrationSourceDrift(
                f"task {snapshot.task_id} 的 source digest 发生漂移"
            )
        if marker["status"] != "complete":
            raise MigrationIntegrityConflict("迁移 marker 不是 complete")
        ledger = ResearchTaskLedger(self.db_path)
        with ledger._connection() as connection:
            target = connection.execute(
                "SELECT 1 FROM research_tasks WHERE research_task_id=?",
                (snapshot.task_id,),
            ).fetchone()
            if target is None:
                raise MigrationIntegrityConflict("完成 marker 缺少 V2 task")
            facts = ledger._load_task_facts_in_connection(connection, snapshot.task_id)
            events = ledger._read_events_in_connection(connection, snapshot.task_id)
            records = ledger._read_scientific_records_from_events(
                snapshot.task_id, events
            )
            head = events[-1].event_digest if events else ""
            if head != marker["target_head_digest"]:
                raise MigrationIntegrityConflict("迁移 marker 的目标摘要已漂移")
            return MigrationResult(
                research_task_id=snapshot.task_id,
                migration_version=self.migration_version,
                source_digest=snapshot.source_digest,
                source_event_count=int(marker["source_event_count"]),
                target_event_ids=tuple(event.event_id for event in events),
                target_scientific_record_ids=tuple(records),
                target_head_digest=head,
            )

    def _build_plan(
        self,
        snapshot: _SourceSnapshot,
        *,
        source_root: str | os.PathLike[str] | None,
    ) -> MigrationPlan:
        """从完整 source 快照构造目标事件，期间不写 SQLite。"""

        self._source_root = None if source_root is None else os.fspath(source_root)
        self._snapshot = snapshot
        self._events: list[_TargetEvent] = []
        self._modules: list[_TargetModule] = []
        self._records: dict[str, _TargetRecord] = {}
        self._source_record_ids: dict[str, str] = {}
        self._source_module_ids: dict[str, str] = {}
        self._source_event_target_events: dict[
            tuple[str, str], tuple[str | None, str | None]
        ] = {}
        self._batches: dict[str, Mapping[str, Any]] = {}
        self._batch_source_events: dict[str, _SourceEvent] = {}
        self._batch_record_ids: dict[str, str] = {}
        self._retrieval_record_id: str | None = None
        self._retrieval_event: _SourceEvent | None = None
        self._query_record_ids: dict[str, str] = {}
        self._selection_record_ids: dict[str, str] = {}
        self._selected_combinations: dict[str, dict[str, str]] = {}
        self._result_infos_by_batch: dict[str, list[_ResultInfo]] = defaultdict(list)
        self._result_info_by_event: dict[str, _ResultInfo] = {}
        self._repair_requests: dict[str, _SourceEvent] = {}
        self._review_record_ids: dict[str, str] = {}
        self._review_module_ids: dict[str, str] = {}
        self._terminal_events = [
            event
            for event in snapshot.events
            if event.event_type in _TERMINAL_SOURCE_EVENT_TYPES
        ]
        if len(self._terminal_events) > 1:
            raise MigrationIntegrityConflict("一个 task 不能有多个 Investigation terminal")

        task_payload = self._target_task_payload(snapshot.task_created.payload)
        task_event_id = _stable_id(
            "task-event", self.migration_version, snapshot.task_id, snapshot.task_created.event_id
        )
        self._events.append(
            _TargetEvent(
                event_id=task_event_id,
                event_type="ResearchTaskCreated",
                payload=task_payload,
                recorded_at=snapshot.task_created.created_at,
                source_event_ids=(snapshot.task_created.event_id,),
            )
        )
        self._map_source(snapshot.task_created.event_id, "task", snapshot.task_id, task_event_id)

        self._collect_batch_and_result_sources(snapshot)
        batches = sorted(
            self._batch_source_events.values(),
            key=lambda event: (
                int(self._batches[event.payload["batch_id"]]["batch_index"]),
                event.created_at,
                event.event_id,
            ),
        )
        if not batches:
            raise MigrationUnsupported("investigation_v2 task 缺少 HypothesisBatchCreated")
        self._append_retrieval_module(snapshot)
        run_mode = snapshot.task_created.payload.get("task_config", {}).get(
            "run_mode", "end_to_end"
        )
        for batch_event in batches:
            batch_id = _require_text(batch_event.payload["batch_id"], "batch_id")
            self._append_query_module(batch_event)
            batch_record = self._append_hypothesis_module(batch_event)
            selection_event = self._selection_event_for_batch(snapshot, batch_id)
            if run_mode == "top5_selection":
                if selection_event is None:
                    raise MigrationUnsupported("Top5 task 缺少 hypothesis_selection_completed")
                self._append_selection_module(selection_event, batch_record)
                if batch_event != batches[-1]:
                    raise MigrationIntegrityConflict("Top5 task 不能继续创建下一批")
                self._append_task_close_for_top5(selection_event)
                self._mark_internal_sources(snapshot)
                return self._freeze_plan(snapshot, task_payload)
            if selection_event is not None:
                self._append_selection_module(selection_event, batch_record)
            self._append_initial_investigation_module(batch_event, batch_record)
            if self._open_module_exists():
                break
            self._append_reviews_and_repairs(batch_id)
            if self._open_module_exists():
                break

        if not self._open_module_exists():
            self._append_terminal_if_present()
        self._mark_internal_sources(snapshot)
        return self._freeze_plan(snapshot, task_payload)

    def _context_binding(
        self,
        module_kind: str,
        owner_id: str,
        *,
        hypothesis_batch_record_id: str | None = None,
        source_module_invocation_id: str | None = None,
    ) -> ContextBinding:
        """把旧 resolver 身份转换为明确的 typed context_binding。"""

        try:
            if module_kind == "hypothesis-investigation":
                if hypothesis_batch_record_id is None:
                    raise MigrationUnsupported(
                        "Investigation context 缺少 hypothesis_batch_record_id"
                    )
                if source_module_invocation_id is None:
                    return CreateInvestigationContext(
                        owner_id,
                        hypothesis_batch_record_id,
                    )
                return ReuseInvestigationContext(
                    owner_id,
                    hypothesis_batch_record_id,
                    source_module_invocation_id,
                )
            if module_kind == "task-review":
                if source_module_invocation_id is None:
                    return CreateResearchTaskReviewSession(owner_id)
                return ReuseResearchTaskReviewSession(
                    owner_id,
                    source_module_invocation_id,
                )
            return StatelessModule()
        except (TypeError, ValueError) as exc:
            raise MigrationUnsupported(
                "旧 runtime 身份不能安全转换为 typed context_binding"
            ) from exc

    def _append_retrieval_module(self, snapshot: _SourceSnapshot) -> None:
        """读取唯一 RetrievalCompleted 指向的显式 artifact 并建立历史记录。

        RetrievalCompleted 事件只保存 artifact 引用；迁移不会扫描运行目录或读取
        Run Bundle。没有 source_root、artifact 不存在、摘要不匹配或事件正文不完整
        时直接停止，避免把当前投影或默认上下文冒充为检索事实。
        """

        retrieval_events = [
            event
            for event in snapshot.events
            if event.event_type == "retrieval_completed"
        ]
        if not retrieval_events:
            return
        if len(retrieval_events) != 1:
            raise MigrationIntegrityConflict(
                "一个 investigation_v2 task 只能迁移唯一 RetrievalCompleted"
            )
        event = retrieval_events[0]
        payload = event.payload
        expected_keys = {
            "operation_instance_id",
            "operation",
            "status",
            "result_digest",
            "artifact_ref",
            "causal_source_ref",
        }
        if set(payload) != expected_keys:
            raise MigrationIntegrityConflict(
                "RetrievalCompleted.payload 字段与 v10 合同不一致"
            )
        artifact_ref = _require_text(payload.get("artifact_ref"), "RetrievalCompleted.artifact_ref")
        if tuple(event.source_refs) != (artifact_ref,):
            raise MigrationIntegrityConflict(
                "RetrievalCompleted 必须唯一引用自身结果 artifact"
            )
        result_digest = _require_text(
            payload.get("result_digest"), "RetrievalCompleted.result_digest"
        )
        if len(result_digest) != 64 or any(char not in "0123456789abcdef" for char in result_digest):
            raise MigrationIntegrityConflict(
                "RetrievalCompleted.result_digest 不是小写 SHA-256"
            )
        if self._source_root is None:
            raise MigrationUnsupported(
                "RetrievalCompleted 缺少用于核验显式 artifact 的 source_root"
            )
        artifact = self._read_retrieval_artifact(
            snapshot,
            artifact_ref=artifact_ref,
            result_digest=result_digest,
        )
        record_id = _stable_id(
            "record", self.migration_version, snapshot.task_id, "retrieval", event.event_id
        )
        module_id = _stable_id(
            "module", self.migration_version, snapshot.task_id, "retrieval", event.event_id
        )
        record = ScientificRecord.from_payload(
            scientific_record_id=record_id,
            research_task_id=snapshot.task_id,
            scientific_record_type="RetrievalRecord",
            scientific_record_schema_version=_MIGRATED_RECORD_SCHEMA_VERSION,
            producer_module_invocation_id=module_id,
            source_scientific_record_ids=(),
            payload={
                "source_artifact_ref": artifact_ref,
                "result_digest": result_digest,
                "retrieval_result": artifact,
            },
        )
        target = _TargetRecord(
            source_event_id=event.event_id,
            record=record,
            event_id=_stable_id(
                "record-event",
                self.migration_version,
                snapshot.task_id,
                event.event_id,
                "retrieval",
            ),
            recorded_at=event.created_at,
        )
        self._append_module(
            _TargetModule(
                module_invocation_id=module_id,
                module_kind="retrieval",
                input_scientific_record_ids=(),
                module_input_digest=_input_digest(snapshot.task_id, "retrieval", ()),
                context_binding=StatelessModule(),
                source_anchor=event.event_id,
                records=[target],
            )
        )
        self._retrieval_event = event
        self._retrieval_record_id = record_id

    def _read_retrieval_artifact(
        self,
        snapshot: _SourceSnapshot,
        *,
        artifact_ref: str,
        result_digest: str,
    ) -> dict[str, Any]:
        """读取并核对 RetrievalCompleted 事件明确指向的 content-addressed 文件。"""

        target = self._resolve_source_file(snapshot, artifact_ref)
        if not target.is_file():
            raise MigrationUnsupported("RetrievalCompleted artifact 不存在")
        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MigrationUnsupported("RetrievalCompleted artifact 无法读取") from exc
        if not isinstance(raw, Mapping) or set(raw) != {
            "status",
            "result_digest",
            "retrieval_result",
        }:
            raise MigrationIntegrityConflict("RetrievalCompleted artifact 字段不完整")
        if raw["status"] != "completed" or raw["result_digest"] != result_digest:
            raise MigrationIntegrityConflict("RetrievalCompleted artifact 摘要或状态不一致")
        result = raw["retrieval_result"]
        if not isinstance(result, Mapping):
            raise MigrationIntegrityConflict("RetrievalCompleted.retrieval_result 必须是 object")
        actual_digest = hashlib.sha256(
            canonical_json(dict(result)).encode("utf-8")
        ).hexdigest()
        if actual_digest != result_digest:
            raise MigrationIntegrityConflict("RetrievalCompleted retrieval_result 摘要不一致")
        return json.loads(canonical_json(dict(result)))

    def _resolve_source_file(
        self, snapshot: _SourceSnapshot, relative_ref: str
    ) -> Path:
        """解析 source event 明确引用的 task 内文件，不扫描目录。"""

        if self._source_root is None:
            raise MigrationUnsupported("缺少核验 source artifact 所需的 source_root")
        root = Path(self._source_root).expanduser().resolve()
        task_relative = _require_text(
            snapshot.task_created.payload.get("task_output_relative_dir"),
            "TaskCreated.task_output_relative_dir",
        )
        task_relative_path = PurePosixPath(task_relative)
        if task_relative_path.is_absolute() or ".." in task_relative_path.parts:
            raise MigrationIntegrityConflict("task_output_relative_dir 不是安全相对路径")
        task_root = (root / Path(*task_relative_path.parts)).resolve()
        try:
            task_root.relative_to(root)
        except ValueError as exc:
            raise MigrationIntegrityConflict("task output 根越出 source_root") from exc
        reference = PurePosixPath(_require_text(relative_ref, "source artifact_ref"))
        if reference.is_absolute() or ".." in reference.parts:
            raise MigrationIntegrityConflict("source artifact_ref 不是安全相对路径")
        current = task_root
        for part in reference.parts:
            current = current / part
            if current.is_symlink():
                raise MigrationIntegrityConflict("source artifact_ref 不能经过符号链接")
        target = current.resolve()
        try:
            target.relative_to(task_root)
        except ValueError as exc:
            raise MigrationIntegrityConflict("source artifact_ref 越出 task 根") from exc
        return target

    def _validate_hypothesis_manifest_ref(self, manifest_ref: str) -> None:
        """核验 BatchCreated 明确引用的 finalized manifest。"""

        target = self._resolve_source_file(self._snapshot, manifest_ref)
        if not target.is_file():
            raise MigrationUnsupported("hypothesis finalized manifest 不存在")
        try:
            parsed = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MigrationUnsupported("hypothesis finalized manifest 无法读取") from exc
        if not isinstance(parsed, Mapping):
            raise MigrationIntegrityConflict("hypothesis finalized manifest 必须是 object")
        # 真实 v10 manifest 使用 manifest_<sha256>.json；若引用含有内容摘要，
        # 必须核对它。早期无摘要文件名仍至少要求明确文件可读。
        for part in target.stem.replace("-", "_").split("_"):
            if len(part) == 64 and all(char in "0123456789abcdef" for char in part):
                actual = hashlib.sha256(
                    canonical_json(dict(parsed)).encode("utf-8")
                ).hexdigest()
                if actual != part:
                    raise MigrationIntegrityConflict(
                        "hypothesis finalized manifest 内容摘要不一致"
                    )
                break

    def _append_query_module(self, batch_event: _SourceEvent) -> None:
        """从当前 Batch 的冻结 question snapshot 建立 QueryRecord。

        作者导入模式明确允许空 query angles：其入口事实是冻结的作者 hypothesis
        manifest，不凭空生成 query。其他任务缺少非空 query 时 fail closed。
        """

        batch_id = _require_text(batch_event.payload.get("batch_id"), "BatchCreated.batch_id")
        snapshot = _as_object(
            batch_event.payload.get("question_snapshot"),
            "BatchCreated.question_snapshot",
        )
        angles_value = snapshot.get("initial_query_angles")
        if not isinstance(angles_value, list):
            raise MigrationUnsupported("BatchCreated.question_snapshot 缺少 initial_query_angles")
        angles = [_require_text(angle, "question_snapshot.initial_query_angles[]") for angle in angles_value]
        if not angles:
            hypothesis_import = self._snapshot.task_created.payload.get("hypothesis_import")
            mode = hypothesis_import.get("mode") if isinstance(hypothesis_import, Mapping) else None
            if not isinstance(mode, str) or not mode.startswith("author_"):
                raise MigrationUnsupported(
                    "非作者入口的 BatchCreated 缺少可迁移的非空 query"
                )
            return
        prior_refs = (self._retrieval_record_id,) if self._retrieval_record_id else ()
        record_id = _stable_id(
            "record", self.migration_version, self._snapshot.task_id, "query", batch_event.event_id
        )
        module_id = _stable_id(
            "module", self.migration_version, self._snapshot.task_id, "query", batch_event.event_id
        )
        record = ScientificRecord.from_payload(
            scientific_record_id=record_id,
            research_task_id=self._snapshot.task_id,
            scientific_record_type="QueryRecord",
            scientific_record_schema_version=_MIGRATED_RECORD_SCHEMA_VERSION,
            producer_module_invocation_id=module_id,
            source_scientific_record_ids=prior_refs,
            payload={
                "question_snapshot": snapshot,
                "query_angles": angles,
            },
        )
        target = _TargetRecord(
            source_event_id=batch_event.event_id,
            record=record,
            event_id=_stable_id(
                "record-event",
                self.migration_version,
                self._snapshot.task_id,
                batch_event.event_id,
                "query",
            ),
            recorded_at=batch_event.created_at,
        )
        self._append_module(
            _TargetModule(
                module_invocation_id=module_id,
                module_kind="research-query-formulation",
                input_scientific_record_ids=prior_refs,
                module_input_digest=_input_digest(self._snapshot.task_id, "query", prior_refs),
                context_binding=StatelessModule(),
                source_anchor=batch_event.event_id,
                records=[target],
            )
        )
        self._query_record_ids[batch_id] = record_id

    def _target_task_payload(self, source: Mapping[str, Any]) -> dict[str, Any]:
        """把旧冻结字段放入 V2 的 input/policy/dependencies，不补默认值。"""

        input_value: dict[str, Any] = {
            "scientific_question": source["scientific_question"],
            "chemistry_context": source["chemistry_context"],
            "task_config": source["task_config"],
            "task_output_relative_dir": source["task_output_relative_dir"],
        }
        if "hypothesis_import" in source:
            input_value["hypothesis_import"] = source["hypothesis_import"]
        # V2 Controller 的入口分流是 TaskCreated 的冻结架构事实。旧 v10
        # TaskCreated 没有该字段，因此只依据同一条不可变 task_config/
        # hypothesis_import 推导一次并写入迁移目标；恢复时不会读取当前默认配置。
        run_mode = source.get("task_config", {}).get("run_mode", "end_to_end")
        hypothesis_import = source.get("hypothesis_import")
        if run_mode == "top5_selection":
            entry_strategy = "top5_selection"
        elif isinstance(hypothesis_import, Mapping) and isinstance(
            hypothesis_import.get("mode"), str
        ) and hypothesis_import["mode"].startswith("author_"):
            entry_strategy = "author_main_hypothesis_import"
        else:
            entry_strategy = "standard_retrieval"
        policy: dict[str, Any] = {
            "stage_architecture_version": source.get(
                "stage_architecture_version", "investigation_v2"
            ),
            "entry_strategy": entry_strategy,
            # 预算也属于被冻结的 TaskCreated 事实；保存在 policy 方便新入口在
            # resume 时直接装配 adapter，不把新默认预算误当成旧任务配置。
            "task_config": copy.deepcopy(dict(source["task_config"])),
        }
        for key in (
            "model_configuration",
            "expert_backend_selection",
            "case_profile",
            "gaussian_route_policy",
        ):
            if key in source:
                policy[key] = source[key]
        dependencies = {
            "source_schema_version": _SUPPORTED_SOURCE_SCHEMA,
            # 旧合同只有 catalog hash。保留原 hash 作为冻结 dependency 内容，
            # 不把它伪装成当前 Catalog 的可查询 ref。
            "tool_catalog_hash": source["tool_catalog_hash"],
        }
        contract_version = source.get("contract_version", "investigation_v2.task.v10")
        if not isinstance(contract_version, (str, int)) or (
            isinstance(contract_version, str) and not contract_version.strip()
        ):
            raise MigrationUnsupported("TaskCreated.contract_version 无法确定")
        return {
            "contract_version": contract_version,
            "input": input_value,
            "policy": policy,
            "dependencies": dependencies,
        }

    def _collect_batch_and_result_sources(self, snapshot: _SourceSnapshot) -> None:
        """解析批次、Result、Evidence、Repair 请求的 source 身份。"""

        for event in snapshot.events:
            if event.event_type == "hypothesis_batch_created":
                payload = event.payload
                batch_id = _require_text(payload.get("batch_id"), "BatchCreated.batch_id")
                if batch_id in self._batch_source_events:
                    raise MigrationIntegrityConflict(f"批次 {batch_id} 重复")
                batch_index = payload.get("batch_index")
                if isinstance(batch_index, bool) or not isinstance(batch_index, int) or batch_index < 1:
                    raise MigrationIntegrityConflict("BatchCreated.batch_index 非法")
                _as_list(payload.get("candidates"), "BatchCreated.candidates")
                phase = _as_object(payload.get("hypothesis_phase"), "BatchCreated.hypothesis_phase")
                ranked = _as_list(phase.get("ranked_strategies"), "hypothesis_phase.ranked_strategies")
                if len(ranked) != len(payload["candidates"]):
                    raise MigrationIntegrityConflict("BatchCreated candidates 与 ranked_strategies 数量不一致")
                if payload.get("task_id", snapshot.task_id) != snapshot.task_id:
                    raise MigrationIntegrityConflict("BatchCreated.task_id 跨 task")
                self._batch_source_events[batch_id] = event
                self._batches[batch_id] = payload
            elif event.event_type == "investigation_repair_requested":
                request_id = event.event_id
                if request_id in self._repair_requests:
                    raise MigrationIntegrityConflict("RepairRequest event 重复")
                payload = event.payload
                _require_text(payload.get("batch_id"), "RepairRequest.batch_id")
                _require_text(payload.get("combination_id"), "RepairRequest.combination_id")
                _require_text(payload.get("result_version"), "RepairRequest.result_version")
                review_event_id = _require_text(
                    payload.get("review_event_ref"), "RepairRequest.review_event_ref"
                )
                review_event = snapshot.by_id.get(review_event_id)
                if review_event is None or review_event.event_type != "investigation_batch_reviewed":
                    raise MigrationIntegrityConflict(
                        "RepairRequest.review_event_ref 必须引用本 task 的 Review"
                    )
                _require_text(payload.get("repair_advice"), "RepairRequest.repair_advice")
                if review_event_id not in event.source_refs:
                    raise MigrationIntegrityConflict(
                        "RepairRequest.source_refs 缺少 Review 前驱"
                    )
                self._repair_requests[request_id] = event
        if not self._batch_source_events:
            return
        batches_by_id = set(self._batch_source_events)
        for event in snapshot.events:
            if event.event_type == "investigation_evidence_verified":
                payload = event.payload
                batch_id = _require_text(payload.get("batch_id"), "Evidence.batch_id")
                combination_id = _require_text(payload.get("combination_id"), "Evidence.combination_id")
                if batch_id not in batches_by_id:
                    raise MigrationIntegrityConflict("Evidence 引用了未知 batch")
                _require_text(
                    payload.get("verification_object_id"),
                    "Evidence.verification_object_id",
                )
                _require_text(
                    payload.get("verification_digest"), "Evidence.verification_digest"
                )
                _require_text(payload.get("verifier_id"), "Evidence.verifier_id")
                _require_text(
                    payload.get("verifier_version"), "Evidence.verifier_version"
                )
                _as_list(payload.get("jobs"), "Evidence.jobs")
                _as_list(payload.get("operations"), "Evidence.operations")
                _as_list(payload.get("artifacts"), "Evidence.artifacts")
                result_ref = payload.get("result_event_ref")
                if result_ref is not None and result_ref not in snapshot.by_id:
                    raise MigrationIntegrityConflict("Evidence.result_event_ref 不存在")
                del combination_id
            elif event.event_type == "investigation_result_recorded":
                info = self._parse_result_info(event, snapshot)
                self._result_infos_by_batch[info.batch_id].append(info)
                self._result_info_by_event[event.event_id] = info
        seen_results: set[tuple[str, str, str]] = set()
        for info in self._result_info_by_event.values():
            key = (info.batch_id, info.combination_id, info.result_version)
            if key in seen_results:
                raise MigrationIntegrityConflict(
                    "同一 batch/combination/result_version 出现多个 Result"
                )
            seen_results.add(key)

    def _parse_result_info(
        self,
        event: _SourceEvent,
        snapshot: _SourceSnapshot,
    ) -> _ResultInfo:
        """解析 Result 与 Evidence/前驱/返修 request 的不可变关系。"""

        payload = event.payload
        batch_id = _require_text(payload.get("batch_id"), "Result.batch_id")
        combination_id = _require_text(payload.get("combination_id"), "Result.combination_id")
        result_version = _require_text(payload.get("result_version"), "Result.result_version")
        if batch_id not in self._batch_source_events:
            raise MigrationIntegrityConflict("Result 引用了未知 batch")
        if payload.get("task_id", snapshot.task_id) != snapshot.task_id:
            raise MigrationIntegrityConflict("Result.task_id 跨 task")
        evidence_event_id = payload.get("evidence_verification_event_ref")
        if evidence_event_id is None:
            evidence_event_ids = [
                ref
                for ref in event.source_refs
                if ref in snapshot.by_id
                and snapshot.by_id[ref].event_type == "investigation_evidence_verified"
            ]
            if len(evidence_event_ids) != 1:
                raise MigrationIntegrityConflict("Result 缺少唯一 Evidence 来源")
            evidence_event_id = evidence_event_ids[0]
        evidence_event_id = _require_text(evidence_event_id, "Result.evidence_event_ref")
        evidence_event = snapshot.by_id.get(evidence_event_id)
        if evidence_event is None or evidence_event.event_type != "investigation_evidence_verified":
            raise MigrationIntegrityConflict("Result 引用的 Evidence event 类型错误")
        if evidence_event.payload.get("batch_id") != batch_id or evidence_event.payload.get("combination_id") != combination_id:
            raise MigrationIntegrityConflict("Result 与 Evidence 的 batch/combination 不一致")
        if evidence_event_id not in event.source_refs:
            raise MigrationIntegrityConflict("Result.source_refs 缺少 Evidence 前驱")
        predecessor_ids = [
            ref
            for ref in event.source_refs
            if ref in snapshot.by_id
            and snapshot.by_id[ref].event_type == "investigation_result_recorded"
        ]
        if len(predecessor_ids) > 1:
            raise MigrationIntegrityConflict("一个 Result 不能有多个 Result 前驱")
        predecessor_event_id = predecessor_ids[0] if predecessor_ids else None
        if predecessor_event_id is not None:
            prior = snapshot.by_id[predecessor_event_id].payload
            if prior.get("batch_id") != batch_id or prior.get("combination_id") != combination_id:
                raise MigrationIntegrityConflict("Result 前驱跨 batch/combination")
        request_ids = [
            ref
            for ref in event.source_refs
            if ref in self._repair_requests
        ]
        explicit_request = payload.get("repair_request_event_ref")
        if explicit_request is not None:
            explicit_request = _require_text(explicit_request, "Result.repair_request_event_ref")
            if explicit_request not in self._repair_requests:
                raise MigrationIntegrityConflict("Result 引用的 RepairRequest 不存在")
            request_ids.append(explicit_request)
        if not request_ids and predecessor_event_id is not None:
            prior_version = snapshot.by_id[predecessor_event_id].payload.get("result_version")
            inferred = [
                request
                for request in self._repair_requests.values()
                if request.payload.get("batch_id") == batch_id
                and request.payload.get("combination_id") == combination_id
                and request.payload.get("result_version") == prior_version
                and request.event_id in event.source_refs
            ]
            if not inferred:
                inferred = [
                    request
                    for request in self._repair_requests.values()
                    if request.payload.get("batch_id") == batch_id
                    and request.payload.get("combination_id") == combination_id
                    and request.payload.get("result_version") == prior_version
                    and predecessor_event_id in request.source_refs
                    and request.created_at < event.created_at
                ]
            if len(inferred) > 1:
                raise MigrationIntegrityConflict("Result 的 RepairRequest 来源不唯一")
            if inferred:
                request_ids.append(inferred[0].event_id)
        if len(set(request_ids)) > 1:
            raise MigrationIntegrityConflict("Result 绑定了多个 RepairRequest")
        repair_request_event_id = request_ids[0] if request_ids else None
        if predecessor_event_id is None and repair_request_event_id is not None:
            raise MigrationIntegrityConflict("初始 Result 不能绑定 RepairRequest")
        return _ResultInfo(
            source_event=event,
            batch_id=batch_id,
            combination_id=combination_id,
            result_version=result_version,
            evidence_event_id=evidence_event_id,
            predecessor_event_id=predecessor_event_id,
            repair_request_event_id=repair_request_event_id,
        )

    def _selection_event_for_batch(
        self,
        snapshot: _SourceSnapshot,
        batch_id: str,
    ) -> _SourceEvent | None:
        """返回 batch 唯一 Top5 selection event。"""

        matches = [
            event
            for event in snapshot.events
            if event.event_type == "hypothesis_selection_completed"
            and event.payload.get("batch_id") == batch_id
        ]
        if len(matches) > 1:
            raise MigrationIntegrityConflict(f"batch {batch_id} 有多个 Top5 selection")
        return matches[0] if matches else None

    def _append_hypothesis_module(self, event: _SourceEvent) -> _TargetRecord:
        """把一个旧 BatchCreated 转成 HypothesisBatchRecord。"""

        payload = event.payload
        batch_id = _require_text(payload["batch_id"], "BatchCreated.batch_id")
        phase = _as_object(payload["hypothesis_phase"], "BatchCreated.hypothesis_phase")
        ranked = _as_list(phase["ranked_strategies"], "ranked_strategies")
        candidates = _as_list(payload["candidates"], "candidates")
        manifest_refs = [
            ref
            for ref in event.source_refs
            if ref not in self._snapshot.by_id
        ]
        if len(manifest_refs) != 1:
            raise MigrationUnsupported("BatchCreated 必须有唯一 finalized hypothesis manifest 来源")
        self._validate_hypothesis_manifest_ref(manifest_refs[0])
        strategy_by_id: dict[str, Mapping[str, Any]] = {}
        for raw_strategy in ranked:
            strategy = _as_object(raw_strategy, "ranked_strategies[]")
            hypothesis_id = _require_text(strategy.get("hypothesis_id"), "strategy.hypothesis_id")
            if hypothesis_id in strategy_by_id:
                raise MigrationIntegrityConflict("BatchCreated hypothesis_id 重复")
            strategy_by_id[hypothesis_id] = strategy
        candidate_entries: list[Mapping[str, Any]] = []
        for raw_candidate in candidates:
            candidate = _as_object(raw_candidate, "candidates[]")
            hypothesis_id = _require_text(candidate.get("hypothesis_id"), "candidate.hypothesis_id")
            if hypothesis_id not in strategy_by_id:
                raise MigrationIntegrityConflict("candidate 缺少对应 ranked strategy")
            candidate_entries.append(candidate)
        module_id = _stable_id("module", self.migration_version, self._snapshot.task_id, "hypothesis", event.event_id)
        record_id = _stable_id("record", self.migration_version, self._snapshot.task_id, "hypothesis", event.event_id)
        prior_record_refs = self._map_batch_generation_sources(payload, batch_id)
        hypotheses = []
        for candidate in candidate_entries:
            hypothesis_id = _require_text(candidate["hypothesis_id"], "candidate.hypothesis_id")
            strategy = strategy_by_id[hypothesis_id]
            hypotheses.append(
                {
                    "hypothesis_ref": f"{record_id}#{hypothesis_id}",
                    "rank": candidate.get("rank"),
                    "strategy": strategy,
                    "candidate": candidate,
                    "source_manifest_ref": manifest_refs[0],
                }
            )
        rationale = phase.get("rationale", payload.get("rationale"))
        verification_plan = phase.get("verification_plan", payload.get("verification_plan"))
        if isinstance(rationale, str) and rationale.strip() and isinstance(
            verification_plan, str
        ) and verification_plan.strip():
            record_schema_version = _TARGET_RECORD_SCHEMA_VERSION
            record_payload: dict[str, Any] = {
                "hypotheses": hypotheses,
                "rationale": rationale,
                "verification_plan": verification_plan,
            }
        else:
            # Case3/author v10 的 BatchCreated 只冻结 ranked_strategies 和
            # candidates；缺少的 rationale/verification_plan 不能从模型或默认值
            # 推导。使用封闭迁移 schema 原样保留可核对的旧字段。
            record_schema_version = _MIGRATED_RECORD_SCHEMA_VERSION
            record_payload = {
                "batch_id": batch_id,
                "batch_index": payload["batch_index"],
                "question_snapshot": _as_object(
                    payload.get("question_snapshot"), "BatchCreated.question_snapshot"
                ),
                "generation_source": _as_object(
                    payload.get("generation_source"), "BatchCreated.generation_source"
                ),
                "hypothesis_phase": {"ranked_strategies": ranked},
                "candidates": candidate_entries,
                "source_manifest_ref": manifest_refs[0],
            }
        record = ScientificRecord.from_payload(
            scientific_record_id=record_id,
            research_task_id=self._snapshot.task_id,
            scientific_record_type="HypothesisBatchRecord",
            scientific_record_schema_version=record_schema_version,
            producer_module_invocation_id=module_id,
            source_scientific_record_ids=tuple(prior_record_refs),
            payload=record_payload,
        )
        target_record = _TargetRecord(
            source_event_id=event.event_id,
            record=record,
            event_id=_stable_id("record-event", self.migration_version, self._snapshot.task_id, event.event_id, "hypothesis"),
            recorded_at=event.created_at,
        )
        module = _TargetModule(
            module_invocation_id=module_id,
            module_kind="hypothesis-generation",
            input_scientific_record_ids=tuple(prior_record_refs),
            module_input_digest=_input_digest(self._snapshot.task_id, "hypothesis", prior_record_refs),
            context_binding=StatelessModule(),
            source_anchor=event.event_id,
            records=[target_record],
        )
        self._append_module(module)
        self._batch_record_ids[batch_id] = record_id
        return target_record

    def _map_batch_generation_sources(
        self,
        payload: Mapping[str, Any],
        batch_id: str,
    ) -> list[str]:
        """把后续 BatchCreated 的旧因果引用映射到目标 Scientific Record。"""

        batch_index = payload.get("batch_index")
        if batch_index == 1:
            refs = []
            if self._retrieval_record_id is not None:
                refs.append(self._retrieval_record_id)
            query_record_id = self._query_record_ids.get(batch_id)
            if query_record_id is not None:
                refs.append(query_record_id)
            return refs
        generation = _as_object(payload.get("generation_source"), "generation_source")
        source_ids = _as_list(generation.get("source_record_ids"), "generation_source.source_record_ids")
        refs: list[str] = []
        previous_batch = generation.get("previous_batch_id")
        if not isinstance(previous_batch, str) or previous_batch not in self._batch_record_ids:
            raise MigrationIntegrityConflict(f"batch {batch_id} 缺少直接上一批目标来源")
        refs.append(self._batch_record_ids[previous_batch])
        if self._retrieval_record_id is not None:
            refs.append(self._retrieval_record_id)
        query_record_id = self._query_record_ids.get(batch_id)
        if query_record_id is not None:
            refs.append(query_record_id)
        for source_id_raw in source_ids:
            source_id = _require_text(source_id_raw, "generation_source.source_record_ids[]")
            target = self._source_record_ids.get(source_id)
            if target is None:
                raise MigrationIntegrityConflict(
                    f"generation_source 引用了无法映射的 source event {source_id}"
                )
            refs.append(target)
        return list(dict.fromkeys(refs))

    def _append_selection_module(
        self,
        event: _SourceEvent,
        batch_record: _TargetRecord,
    ) -> _TargetRecord:
        """把 Top5 selection 转成 HypothesisSelectionRecord。"""

        payload = event.payload
        selected = _as_list(payload.get("selected_combinations"), "selected_combinations")
        if payload.get("selected_count") != 5 or len(selected) != 5:
            raise MigrationUnsupported("Top5 migration 需要旧事件明确保存五项选择")
        reason = payload.get("selection_reason")
        if not isinstance(reason, str) or not reason.strip():
            raise MigrationUnsupported("Top5 event 缺少确定性的 selection_reason")
        batch_id = _require_text(payload.get("batch_id"), "Selection.batch_id")
        candidates_by_id = {
            _require_text(item.get("hypothesis_id"), "candidate.hypothesis_id"): item
            for item in _as_list(self._batches[batch_id].get("candidates"), "candidates")
        }
        selected_hypotheses = []
        for raw in selected:
            item = _as_object(raw, "selected_combinations[]")
            hypothesis_id = _require_text(item.get("hypothesis_id"), "selected.hypotheses.hypothesis_id")
            if hypothesis_id not in candidates_by_id:
                raise MigrationIntegrityConflict("Top5 选择引用未知 hypothesis")
            rank = item.get("rank", candidates_by_id[hypothesis_id].get("rank"))
            if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
                raise MigrationIntegrityConflict("Top5 hypothesis rank 非法")
            selected_hypotheses.append(
                {
                    "hypothesis_ref": f"{batch_record.record.scientific_record_id}#{hypothesis_id}",
                    "rank": rank,
                }
            )
        module_id = _stable_id("module", self.migration_version, self._snapshot.task_id, "selection", event.event_id)
        record_id = _stable_id("record", self.migration_version, self._snapshot.task_id, "selection", event.event_id)
        record = ScientificRecord.from_payload(
            scientific_record_id=record_id,
            research_task_id=self._snapshot.task_id,
            scientific_record_type="HypothesisSelectionRecord",
            scientific_record_schema_version=_TARGET_RECORD_SCHEMA_VERSION,
            producer_module_invocation_id=module_id,
            source_scientific_record_ids=(batch_record.record.scientific_record_id,),
            payload={
                "selected_hypotheses": selected_hypotheses,
                "selection_reason": reason,
            },
        )
        target_record = _TargetRecord(
            source_event_id=event.event_id,
            record=record,
            event_id=_stable_id("record-event", self.migration_version, self._snapshot.task_id, event.event_id, "selection"),
            recorded_at=event.created_at,
        )
        module = _TargetModule(
            module_invocation_id=module_id,
            module_kind="hypothesis-selection",
            input_scientific_record_ids=(batch_record.record.scientific_record_id,),
            module_input_digest=_input_digest(self._snapshot.task_id, "selection", (batch_record.record.scientific_record_id,)),
            context_binding=StatelessModule(),
            source_anchor=event.event_id,
            records=[target_record],
        )
        self._append_module(module)
        self._selection_record_ids[batch_id] = record_id
        return target_record

    def _append_initial_investigation_module(
        self,
        batch_event: _SourceEvent,
        batch_record: _TargetRecord,
    ) -> None:
        """按一个 batch 聚合 1..N 个 initial Evidence/Result 对。"""

        batch_id = _require_text(batch_event.payload["batch_id"], "batch_id")
        infos = self._result_infos_by_batch.get(batch_id, [])
        initial_infos = [info for info in infos if info.predecessor_event_id is None]
        all_combinations = self._selected_combination_ids(batch_id)
        seen_combinations = {info.combination_id for info in initial_infos}
        if seen_combinations - set(all_combinations):
            raise MigrationIntegrityConflict("Result combination 不是本批 selected combination")
        missing = set(all_combinations) - seen_combinations
        later_events = [
            event
            for event in self._snapshot.events
            if event.event_type in {
                "investigation_batch_reviewed",
                "investigation_repair_requested",
                "investigation_task_closed",
                "investigation_task_failed",
            }
            and event.payload.get("batch_id") == batch_id
        ]
        if missing and later_events:
            raise MigrationIntegrityConflict("初始 Investigation 不完整但 source 已进入后续分支")
        module_id = _stable_id(
            "module",
            self.migration_version,
            self._snapshot.task_id,
            "investigation",
            batch_id,
            "initial",
        )
        record_pairs = self._build_result_records(
            initial_infos,
            batch_record,
            module_kind="hypothesis-investigation",
            module_invocation_id=module_id,
        )
        unresolved_source_ids = self._open_source_ids_for_batch(batch_id, missing)
        if missing or not initial_infos:
            # Open Module Run 只表示账本中已有开始事实，不表示部分输出已经提交。
            # 已经形成完整 Evidence/Result 对的历史部分先成为一个闭合模块；待恢复
            # 的 continuation 不携带任何 ScientificRecord，之后由 Harness 一次提交
            # 完整 PreparedModuleInvocationOutput。这样不会让 partial output 看起来像成功事实。
            continuation_refs = tuple(
                dict.fromkeys(
                    self._investigation_input_refs(batch_id, batch_record)
                    + tuple(record.record.scientific_record_id for record in record_pairs)
                )
            )
            resolved_runtime = self._resolve_runtime(
                module_kind="hypothesis-investigation",
                input_scientific_record_ids=continuation_refs,
                source_event_ids=unresolved_source_ids or (batch_event.event_id,),
            )
            if record_pairs:
                closed_module = _TargetModule(
                    module_invocation_id=module_id,
                    module_kind="hypothesis-investigation",
                    input_scientific_record_ids=self._investigation_input_refs(batch_id, batch_record),
                    module_input_digest=_input_digest(
                        self._snapshot.task_id,
                        "investigation",
                        self._investigation_input_refs(batch_id, batch_record),
                    ),
                    # 旧 Result 已经由同一个可核验 Harness runtime 产生；把该
                    # resolver 身份绑定到闭合历史，continuation 才能明确复用它。
                    # 没有 resolver 的路径在上面已经 fail closed。
                    context_binding=self._context_binding(
                        "hypothesis-investigation",
                        resolved_runtime.runtime_ref,
                        hypothesis_batch_record_id=batch_record.record.scientific_record_id,
                    ),
                    source_anchor=batch_event.event_id,
                    records=record_pairs,
                    closed=True,
                )
                self._append_module(closed_module)
            module = _TargetModule(
                module_invocation_id=(
                    module_id
                    if not record_pairs
                    else _stable_id(
                        "module",
                        self.migration_version,
                        self._snapshot.task_id,
                        "investigation",
                        batch_id,
                        "continuation",
                    )
                ),
                module_kind="hypothesis-investigation",
                input_scientific_record_ids=continuation_refs,
                module_input_digest=_input_digest(
                    self._snapshot.task_id, "investigation", continuation_refs
                ),
                context_binding=self._context_binding(
                    "hypothesis-investigation",
                    resolved_runtime.runtime_ref,
                    hypothesis_batch_record_id=batch_record.record.scientific_record_id,
                    source_module_invocation_id=module_id if record_pairs else None,
                ),
                source_anchor=unresolved_source_ids[0] if unresolved_source_ids else batch_event.event_id,
                records=[],
                closed=False,
            )
            self._append_module(module)
            return
        context_binding = self._context_binding(
            "hypothesis-investigation",
            self._synthetic_runtime("investigation", batch_id),
            hypothesis_batch_record_id=batch_record.record.scientific_record_id,
        )
        module = _TargetModule(
            module_invocation_id=module_id,
            module_kind="hypothesis-investigation",
            input_scientific_record_ids=self._investigation_input_refs(batch_id, batch_record),
            module_input_digest=_input_digest(self._snapshot.task_id, "investigation", self._investigation_input_refs(batch_id, batch_record)),
            context_binding=context_binding,
            source_anchor=batch_event.event_id,
            records=record_pairs,
        )
        self._append_module(module)

    def _selected_combination_ids(self, batch_id: str) -> tuple[str, ...]:
        """读取 BatchCreated 中 selected combination 的稳定顺序。"""

        selected: list[tuple[int, str]] = []
        for raw in _as_list(self._batches[batch_id].get("candidates"), "candidates"):
            item = _as_object(raw, "candidates[]")
            if item.get("selection_status") != "selected":
                continue
            combination_id = _require_text(item.get("combination_id"), "candidate.combination_id")
            index = item.get("combination_index")
            if isinstance(index, bool) or not isinstance(index, int) or index < 1:
                raise MigrationIntegrityConflict("candidate.combination_index 非法")
            selected.append((index, combination_id))
        selected.sort()
        if len({item[1] for item in selected}) != len(selected):
            raise MigrationIntegrityConflict("selected combination_id 重复")
        self._selected_combinations[batch_id] = {combination_id: combination_id for _, combination_id in selected}
        return tuple(combination_id for _, combination_id in selected)

    def _investigation_input_refs(
        self,
        batch_id: str,
        batch_record: _TargetRecord,
    ) -> tuple[str, ...]:
        """返回 Investigation ModuleRun 的目标输入引用。"""

        selection_id = self._selection_record_ids.get(batch_id)
        return (selection_id,) if selection_id is not None else (batch_record.record.scientific_record_id,)

    def _build_result_records(
        self,
        infos: Sequence[_ResultInfo],
        batch_record: _TargetRecord,
        *,
        module_kind: str,
        module_invocation_id: str | None = None,
    ) -> list[_TargetRecord]:
        """为一组 Result 先生成 Evidence，再生成 Result 记录。"""

        del module_kind
        if module_invocation_id is None:
            module_invocation_id = _stable_id("module", self.migration_version, self._snapshot.task_id, "result-temp", *(info.source_event.event_id for info in infos))
        result_records: list[_TargetRecord] = []
        for info in sorted(infos, key=lambda item: (item.source_event.created_at, item.source_event.event_id)):
            evidence_event = self._snapshot.by_id[info.evidence_event_id]
            evidence_id = self._source_record_ids.get(evidence_event.event_id)
            if evidence_id is None:
                evidence_id = _stable_id("record", self.migration_version, self._snapshot.task_id, "evidence", evidence_event.event_id)
            evidence_payload, evidence_schema_version = self._evidence_payload(
                evidence_event.payload
            )
            evidence = ScientificRecord.from_payload(
                scientific_record_id=evidence_id,
                research_task_id=self._snapshot.task_id,
                scientific_record_type="VerifiedEvidenceRecord",
                scientific_record_schema_version=evidence_schema_version,
                producer_module_invocation_id=module_invocation_id,
                source_scientific_record_ids=(self._selection_record_ids.get(info.batch_id, batch_record.record.scientific_record_id),),
                payload=evidence_payload,
            )
            evidence_target = _TargetRecord(
                source_event_id=evidence_event.event_id,
                record=evidence,
                event_id=_stable_id("record-event", self.migration_version, self._snapshot.task_id, evidence_event.event_id, "evidence"),
                recorded_at=evidence_event.created_at,
            )
            self._register_record(evidence_target)
            result_records.append(evidence_target)

            result_id = _stable_id("record", self.migration_version, self._snapshot.task_id, "result", info.source_event.event_id)
            source_refs: list[str] = [
                self._selection_record_ids.get(info.batch_id, batch_record.record.scientific_record_id),
                evidence_id,
            ]
            if info.predecessor_event_id is not None:
                prior_id = self._source_record_ids.get(info.predecessor_event_id)
                if prior_id is None:
                    raise MigrationIntegrityConflict("Result 前驱尚未映射")
                source_refs.append(prior_id)
            if info.repair_request_event_id is not None:
                repair_id = self._source_record_ids.get(info.repair_request_event_id)
                if repair_id is None:
                    raise MigrationIntegrityConflict("Result RepairRequest 尚未映射")
                source_refs.append(repair_id)
            result_payload = info.source_event.payload
            narrative = result_payload.get("narrative")
            if not isinstance(narrative, str) or not narrative.strip():
                raise MigrationUnsupported("Result 缺少冻结 narrative")
            self_judgment = result_payload.get("self_judgment")
            if self_judgment not in {"supported", "refuted"}:
                raise MigrationIntegrityConflict("Result.self_judgment 非法")
            observations = _as_list(result_payload.get("observations"), "Result.observations")
            limitations = _as_list(result_payload.get("limitations"), "Result.limitations")
            result = ScientificRecord.from_payload(
                scientific_record_id=result_id,
                research_task_id=self._snapshot.task_id,
                scientific_record_type="InvestigationResultRecord",
                scientific_record_schema_version=_TARGET_RECORD_SCHEMA_VERSION,
                producer_module_invocation_id=module_invocation_id,
                source_scientific_record_ids=tuple(dict.fromkeys(source_refs)),
                payload={
                    "conclusion": narrative,
                    "self_judgment": self_judgment,
                    "observations": observations,
                    "limitations": limitations,
                    "evidence_ref": evidence_id,
                },
            )
            result_target = _TargetRecord(
                source_event_id=info.source_event.event_id,
                record=result,
                event_id=_stable_id("record-event", self.migration_version, self._snapshot.task_id, info.source_event.event_id, "result"),
                recorded_at=info.source_event.created_at,
            )
            self._register_record(result_target)
            result_records.append(result_target)
        return result_records

    def _evidence_payload(
        self, source: Mapping[str, Any]
    ) -> tuple[dict[str, Any], str]:
        """保留旧证据核验摘要，不把内部身份提升到目标顶层。

        v10 的真实 Evidence 没有 source_manifest_ref/summary。此时使用封闭的
        ``v10-migrated`` payload，直接保存旧事件中的 verification_object_id、摘要、
        verifier 和 jobs/operations/artifacts；不把任何旧字段重命名为
        source_manifest_ref，也不声称旧事件曾保存 hypothesis manifest。
        """

        jobs = _as_list(source.get("jobs"), "Evidence.jobs")
        operations = _as_list(source.get("operations"), "Evidence.operations")
        artifacts = _as_list(source.get("artifacts"), "Evidence.artifacts")
        artifact_refs: list[str] = []
        for item in artifacts:
            if not isinstance(item, Mapping):
                raise MigrationIntegrityConflict("Evidence.artifacts item 必须是 object")
            artifact_refs.append(_require_text(item.get("artifact_id"), "Evidence.artifact_id"))
        verification_object_id = _require_text(
            source.get("verification_object_id"), "Evidence.verification_object_id"
        )
        verifier_id = _require_text(source.get("verifier_id"), "Evidence.verifier_id")
        verifier_version = _require_text(
            source.get("verifier_version"), "Evidence.verifier_version"
        )
        verification_digest = _require_text(
            source.get("verification_digest"), "Evidence.verification_digest"
        )
        operation_instance_id = _require_text(
            source.get("operation_instance_id"), "Evidence.operation_instance_id"
        )
        payload: dict[str, Any] = {
            "evidence_items": [
                {"jobs": jobs, "operations": operations, "artifacts": artifacts}
            ],
            "artifact_refs": artifact_refs,
        }
        source_manifest_ref = source.get("source_manifest_ref")
        summary = source.get("summary")
        if isinstance(source_manifest_ref, str) and source_manifest_ref.strip() and isinstance(
            summary, str
        ) and summary.strip():
            payload["source_manifest_ref"] = source_manifest_ref.strip()
            payload["summary"] = summary.strip()
            payload["verification"] = {
                "verification_object_id": verification_object_id,
                "verifier_id": verifier_id,
                "verifier_version": verifier_version,
                "verification_digest": verification_digest,
                "operation_instance_id": operation_instance_id,
                **(
                    {"reference_binding_version": source["reference_binding_version"]}
                    if "reference_binding_version" in source
                    else {}
                ),
            }
            return payload, _TARGET_RECORD_SCHEMA_VERSION
        payload.update(
            {
                "verification_object_id": verification_object_id,
                "verification_digest": verification_digest,
                "verifier_id": verifier_id,
                "verifier_version": verifier_version,
                "operation_instance_id": operation_instance_id,
                **(
                    {"reference_binding_version": source["reference_binding_version"]}
                    if "reference_binding_version" in source
                    else {}
                ),
            }
        )
        return payload, _MIGRATED_RECORD_SCHEMA_VERSION

    def _append_reviews_and_repairs(self, batch_id: str) -> None:
        """按旧 Review→Repair→Result 因果顺序追加目标模块。"""

        reviews = [
            event
            for event in self._snapshot.events
            if event.event_type == "investigation_batch_reviewed"
            and event.payload.get("batch_id") == batch_id
        ]
        reviews.sort(key=lambda event: (event.created_at, event.event_id))
        for review_event in reviews:
            reviewed = self._reviewed_result_refs(review_event)
            findings = self._review_findings(review_event, reviewed)
            request_events = [
                event
                for event in self._snapshot.events
                if event.event_type == "investigation_repair_requested"
                and event.payload.get("review_event_ref") == review_event.event_id
            ]
            route_events = [
                event
                for event in self._snapshot.events
                if event.event_type == "investigation_route_marked"
                and event.payload.get("review_event_ref") == review_event.event_id
            ]
            outcome = self._review_outcome(review_event.payload, findings, request_events, route_events)
            review_module_id = _stable_id("module", self.migration_version, self._snapshot.task_id, "review", review_event.event_id)
            review_record_id = _stable_id("record", self.migration_version, self._snapshot.task_id, "review", review_event.event_id)
            review_record = ScientificRecord.from_payload(
                scientific_record_id=review_record_id,
                research_task_id=self._snapshot.task_id,
                scientific_record_type="ReviewRecord",
                scientific_record_schema_version=_TARGET_RECORD_SCHEMA_VERSION,
                producer_module_invocation_id=review_module_id,
                source_scientific_record_ids=tuple(reviewed),
                payload={
                    "reviewed_result_refs": list(reviewed),
                    "findings": findings,
                    "outcome": outcome,
                },
            )
            review_target = _TargetRecord(
                source_event_id=review_event.event_id,
                record=review_record,
                event_id=_stable_id("record-event", self.migration_version, self._snapshot.task_id, review_event.event_id, "review"),
                recorded_at=review_event.created_at,
            )
            self._register_record(review_target)
            records = [review_target]
            for request_event in sorted(request_events, key=lambda event: (event.created_at, event.event_id)):
                repair_id = _stable_id("record", self.migration_version, self._snapshot.task_id, "repair", request_event.event_id)
                repair_payload = request_event.payload
                result_ref = self._result_ref_for_request(request_event)
                problem = repair_payload.get("problem")
                advice = _require_text(
                    repair_payload.get("repair_advice"),
                    "RepairRequest.repair_advice",
                )
                if isinstance(problem, str) and problem.strip():
                    repair_schema_version = _TARGET_RECORD_SCHEMA_VERSION
                    repair_record_payload: dict[str, Any] = {
                        "result_ref": result_ref,
                        "problem": problem.strip(),
                        "instructions": advice,
                    }
                else:
                    # 真实 v10 RepairRequest 只有 repair_advice 和身份/计数
                    # 字段；不能把 advice 假装成 problem。封闭迁移 schema 保存
                    # 旧正文的科学可用子集，并保留明确的 Result 前驱。
                    repair_schema_version = _MIGRATED_RECORD_SCHEMA_VERSION
                    repair_record_payload = {
                        "result_ref": result_ref,
                        "source_payload": {
                            "batch_id": _require_text(
                                repair_payload.get("batch_id"),
                                "RepairRequest.batch_id",
                            ),
                            "combination_id": _require_text(
                                repair_payload.get("combination_id"),
                                "RepairRequest.combination_id",
                            ),
                            "result_version": _require_text(
                                repair_payload.get("result_version"),
                                "RepairRequest.result_version",
                            ),
                            "repair_advice": advice,
                            "repair_count": repair_payload.get("repair_count"),
                            "review_event_ref": _require_text(
                                repair_payload.get("review_event_ref"),
                                "RepairRequest.review_event_ref",
                            ),
                        },
                    }
                repair = ScientificRecord.from_payload(
                    scientific_record_id=repair_id,
                    research_task_id=self._snapshot.task_id,
                    scientific_record_type="RepairRequestRecord",
                    scientific_record_schema_version=repair_schema_version,
                    producer_module_invocation_id=review_module_id,
                    source_scientific_record_ids=(review_record_id, result_ref),
                    payload={
                        **repair_record_payload,
                    },
                )
                repair_target = _TargetRecord(
                    source_event_id=request_event.event_id,
                    record=repair,
                    event_id=_stable_id("record-event", self.migration_version, self._snapshot.task_id, request_event.event_id, "repair"),
                    recorded_at=request_event.created_at,
                )
                self._register_record(repair_target)
                records.append(repair_target)
            final_target = self._final_answer_for_review(review_event, review_target)
            if final_target is not None:
                self._register_record(final_target)
                records.append(final_target)
            all_results = tuple(reviewed)
            input_refs = all_results
            previous_review = self._previous_review_module()
            review_owner_id = self._synthetic_runtime("review", review_event.event_id)
            previous_review_id = None
            if previous_review is not None:
                previous_binding = previous_review.context_binding
                review_owner_id = getattr(
                    previous_binding,
                    "research_task_review_session_id",
                    review_owner_id,
                )
                previous_review_id = previous_review.module_invocation_id
            module = _TargetModule(
                module_invocation_id=review_module_id,
                module_kind="task-review",
                input_scientific_record_ids=input_refs,
                module_input_digest=_input_digest(self._snapshot.task_id, "review", input_refs),
                context_binding=self._context_binding(
                    "task-review",
                    review_owner_id,
                    source_module_invocation_id=previous_review_id,
                ),
                source_anchor=review_event.event_id,
                records=records,
                closed=not (
                    (outcome == "RepairRequired" and not request_events)
                    or (outcome in {"Accepted", "Unanswerable", "Stopped"} and final_target is None)
                ),
            )
            if not module.closed:
                resolved = self._resolve_runtime(
                    module_kind="task-review",
                    input_scientific_record_ids=input_refs,
                    source_event_ids=(review_event.event_id,),
                )
                module.context_binding = self._context_binding(
                    "task-review",
                    resolved.runtime_ref,
                    source_module_invocation_id=previous_review_id,
                )
            self._append_module(module)
            self._review_record_ids[review_event.event_id] = review_record_id
            self._review_module_ids[review_event.event_id] = review_module_id
            for route_event in route_events:
                self._map_source(route_event.event_id, "audit", review_record_id, None)
            repair_infos = [
                info
                for info in self._result_infos_by_batch.get(batch_id, [])
                if info.repair_request_event_id in {event.event_id for event in request_events}
            ]
            if repair_infos:
                self._append_repair_investigation_module(
                    batch_id,
                    review_event,
                    review_target,
                    request_events,
                    repair_infos,
                )
                if self._open_module_exists():
                    return

    def _reviewed_result_refs(self, event: _SourceEvent) -> tuple[str, ...]:
        """把旧 Review 的 combination/version vector 解析成目标 Result refs。"""

        payload = event.payload
        combinations = _as_list(payload.get("combination_ids"), "Review.combination_ids")
        versions = _as_list(payload.get("result_versions"), "Review.result_versions")
        if len(combinations) != len(versions) or not combinations:
            raise MigrationIntegrityConflict("Review result vector 不完整")
        refs: list[str] = []
        for combination_id, version in zip(combinations, versions):
            combo = _require_text(combination_id, "Review.combination_id")
            version_text = _require_text(version, "Review.result_version")
            candidates = [
                info
                for info in self._result_infos_by_batch.get(payload.get("batch_id"), [])
                if info.combination_id == combo and info.result_version == version_text
            ]
            if len(candidates) != 1:
                raise MigrationIntegrityConflict("Review 引用的 Result 版本缺失或重复")
            target = self._source_record_ids.get(candidates[0].source_event.event_id)
            if target is None:
                raise MigrationIntegrityConflict("Review 引用的 Result 尚未迁移")
            refs.append(target)
        if len(set(refs)) != len(refs):
            raise MigrationIntegrityConflict("Review reviewed_result_refs 重复")
        return tuple(refs)

    def _review_findings(
        self,
        event: _SourceEvent,
        reviewed: Sequence[str],
    ) -> list[dict[str, Any]]:
        """将 finding 的组合身份替换为目标 Result ref，并保留安全字段。"""

        findings = _as_list(event.payload.get("findings"), "Review.findings")
        if len(findings) != len(reviewed):
            raise MigrationIntegrityConflict("Review findings 未覆盖全部 Result")
        result_by_combo: dict[str, str] = {}
        combinations = _as_list(event.payload.get("combination_ids"), "Review.combination_ids")
        for combo, target in zip(combinations, reviewed):
            result_by_combo[_require_text(combo, "Review.combination_id")] = target
        output: list[dict[str, Any]] = []
        for raw in findings:
            finding = _as_object(raw, "Review.findings[]")
            combo = finding.pop("combination_id", None)
            combo = _require_text(combo, "Review.finding.combination_id")
            result_ref = result_by_combo.get(combo)
            if result_ref is None:
                raise MigrationIntegrityConflict("Review finding 引用了未审查 combination")
            finding["result_ref"] = result_ref
            output.append(finding)
        return output

    @staticmethod
    def _review_outcome(
        payload: Mapping[str, Any],
        findings: Sequence[Mapping[str, Any]],
        requests: Sequence[_SourceEvent],
        routes: Sequence[_SourceEvent],
    ) -> str:
        """从完整不可变 Review 分支推导互斥目标结论。"""

        explicit = payload.get("outcome")
        if explicit in {"Accepted", "RepairRequired", "RegenerateHypotheses", "Unanswerable", "Stopped"}:
            return explicit
        has_fail = any(item.get("validity") == "fail" for item in findings)
        can_answer = payload.get("can_answer")
        final_answer = payload.get("final_answer")
        explanation = payload.get("answerability_explanation")
        if has_fail and requests:
            return "RepairRequired"
        if routes:
            return "RegenerateHypotheses"
        if can_answer is True and isinstance(final_answer, str) and final_answer.strip():
            return "Accepted"
        if can_answer is False and (
            (isinstance(explanation, str) and explanation.strip())
            or (isinstance(final_answer, str) and final_answer.strip())
        ):
            return "Unanswerable"
        raise MigrationUnsupported("Review outcome 无法从冻结字段确定")

    def _result_ref_for_request(self, request_event: _SourceEvent) -> str:
        """读取 RepairRequest 对应的当前旧 Result 并返回目标 ref。"""

        payload = request_event.payload
        batch_id = payload.get("batch_id")
        combo = payload.get("combination_id")
        version = payload.get("result_version")
        candidates = [
            info
            for info in self._result_infos_by_batch.get(batch_id, [])
            if info.combination_id == combo and info.result_version == version
        ]
        if len(candidates) != 1:
            raise MigrationIntegrityConflict("RepairRequest 未绑定唯一当前 Result")
        target = self._source_record_ids.get(candidates[0].source_event.event_id)
        if target is None:
            raise MigrationIntegrityConflict("RepairRequest 的 Result 尚未迁移")
        return target

    def _final_answer_for_review(
        self,
        review_event: _SourceEvent,
        review_target: _TargetRecord,
    ) -> _TargetRecord | None:
        """将 Review/TaskClosed 的冻结回答转成 FinalAnswerRecord。"""

        terminal = next(
            (
                event
                for event in self._terminal_events
                if event.payload.get("review_event_ref") == review_event.event_id
            ),
            None,
        )
        source = review_event.payload
        final_answer = source.get("final_answer")
        answerability = "answerable" if source.get("can_answer") is True else "unanswerable"
        if terminal is not None:
            terminal_answer = terminal.payload.get("final_answer")
            terminal_can_answer = terminal.payload.get("can_answer")
            if final_answer is not None and terminal_answer is not None and final_answer != terminal_answer:
                raise MigrationIntegrityConflict("Review 与 TaskClosed final_answer 不一致")
            final_answer = terminal_answer if terminal_answer is not None else final_answer
            answerability = "answerable" if terminal_can_answer is True else "unanswerable"
        if final_answer is None and answerability == "unanswerable":
            final_answer = source.get("answerability_explanation")
            if terminal is not None:
                final_answer = terminal.payload.get("answerability_explanation") or final_answer
        if not isinstance(final_answer, str) or not final_answer.strip():
            return None
        source_event_id = terminal.event_id if terminal is not None else review_event.event_id
        record_id = _stable_id("record", self.migration_version, self._snapshot.task_id, "final-answer", source_event_id)
        module_id = review_target.record.producer_module_invocation_id
        record = ScientificRecord.from_payload(
            scientific_record_id=record_id,
            research_task_id=self._snapshot.task_id,
            scientific_record_type="FinalAnswerRecord",
            scientific_record_schema_version=_TARGET_RECORD_SCHEMA_VERSION,
            producer_module_invocation_id=module_id,
            source_scientific_record_ids=(review_target.record.scientific_record_id,),
            payload={"answer": final_answer, "answerability": answerability},
        )
        target = _TargetRecord(
            source_event_id=source_event_id,
            record=record,
            event_id=_stable_id("record-event", self.migration_version, self._snapshot.task_id, source_event_id, "final-answer"),
            recorded_at=(terminal or review_event).created_at,
        )
        if terminal is not None:
            self._map_source(terminal.event_id, "record", record_id, target.event_id)
        return target

    def _append_repair_investigation_module(
        self,
        batch_id: str,
        review_event: _SourceEvent,
        review_target: _TargetRecord,
        request_events: Sequence[_SourceEvent],
        infos: Sequence[_ResultInfo],
    ) -> None:
        """为一个 Review 的所有 repair Result 建立一个聚合 Investigation run。"""

        del review_target
        request_ids = {event.event_id for event in request_events}
        complete_infos = [info for info in infos if info.repair_request_event_id in request_ids]
        pending_requests = request_ids - {
            info.repair_request_event_id for info in complete_infos if info.repair_request_event_id
        }
        module_id = _stable_id(
            "module", self.migration_version, self._snapshot.task_id, "investigation-repair", review_event.event_id
        )
        prior_module = self._investigation_module_for_result(complete_infos[0]) if complete_infos else self._latest_investigation_module()
        if prior_module is None:
            raise MigrationIntegrityConflict("repair Investigation 缺少可复用的初始 Module Run")
        refs = tuple(dict.fromkeys((self._review_record_ids.get(review_event.event_id, ""), *(
            self._source_record_ids.get(info.predecessor_event_id, "") for info in complete_infos
            if info.predecessor_event_id is not None
        ))))
        refs = tuple(ref for ref in refs if ref)
        prior_binding = prior_module.context_binding
        owner_id = getattr(prior_binding, "investigation_context_id", None)
        if owner_id is None:
            raise MigrationIntegrityConflict(
                "repair Investigation 的前序 Module Invocation 缺少 context owner"
            )
        record_pairs = self._build_result_records(
            complete_infos,
            self._record_by_id(self._review_record_ids.get(review_event.event_id, "")),
            module_kind="hypothesis-investigation",
            module_invocation_id=module_id,
        )
        continuation_source = prior_module.module_invocation_id
        if pending_requests:
            if record_pairs:
                closed_module = _TargetModule(
                    module_invocation_id=module_id,
                    module_kind="hypothesis-investigation",
                    input_scientific_record_ids=refs,
                    module_input_digest=_input_digest(
                        self._snapshot.task_id, "investigation", refs
                    ),
                    context_binding=self._context_binding(
                        "hypothesis-investigation",
                        owner_id,
                        hypothesis_batch_record_id=self._batch_record_ids[batch_id],
                        source_module_invocation_id=prior_module.module_invocation_id,
                    ),
                    source_anchor=review_event.event_id,
                    records=record_pairs,
                    closed=True,
                )
                self._append_module(closed_module)
            open_refs = tuple(
                dict.fromkeys(
                    refs + tuple(record.record.scientific_record_id for record in record_pairs)
                )
            )
            resolved = self._resolve_runtime(
            module_kind="hypothesis-investigation",
                input_scientific_record_ids=open_refs,
                source_event_ids=tuple(sorted(pending_requests)),
            )
            continuation_source = (
                module_id if record_pairs else prior_module.module_invocation_id
            )
            owner_id = resolved.runtime_ref
        module = _TargetModule(
            module_invocation_id=(
                module_id
                if not pending_requests or not record_pairs
                else _stable_id(
                    "module",
                    self.migration_version,
                    self._snapshot.task_id,
                    "investigation-repair",
                    review_event.event_id,
                    "continuation",
                )
            ),
            module_kind="hypothesis-investigation",
            input_scientific_record_ids=(
                refs
                if not pending_requests
                else tuple(
                    dict.fromkeys(
                        refs + tuple(record.record.scientific_record_id for record in record_pairs)
                    )
                )
            ),
            module_input_digest=_input_digest(
                self._snapshot.task_id,
                "investigation",
                refs
                if not pending_requests
                else tuple(
                    dict.fromkeys(
                        refs + tuple(record.record.scientific_record_id for record in record_pairs)
                    )
                ),
            ),
            context_binding=self._context_binding(
                "hypothesis-investigation",
                owner_id,
                hypothesis_batch_record_id=self._batch_record_ids[batch_id],
                source_module_invocation_id=continuation_source,
            ),
            source_anchor=review_event.event_id,
            records=[] if pending_requests else record_pairs,
            closed=not pending_requests,
        )
        if not pending_requests:
            self._append_module(module)
        else:
            self._append_module(module)

    def _append_task_close_for_top5(self, selection_event: _SourceEvent) -> None:
        """Top5-only 没有 FinalAnswer，使用明确 selection closure 终止。"""

        selection_id = self._selection_record_ids.get(selection_event.payload.get("batch_id"))
        if selection_id is None:
            raise MigrationIntegrityConflict("Top5 closure 缺少 selection record")
        self._append_task_close(
            source_event=selection_event,
            outcome={"status": "succeeded", "selection_record_ref": selection_id},
        )

    def _append_terminal_if_present(self) -> None:
        """将显式旧 terminal 变为 V2 TaskClosed/TaskFailed。"""

        if not self._terminal_events:
            return
        terminal = self._terminal_events[0]
        if terminal.event_type == "investigation_task_failed":
            self._append_failed_terminal(terminal)
            return
        review_ref = terminal.payload.get("review_event_ref")
        review_id = self._review_record_ids.get(review_ref)
        if review_id is None:
            raise MigrationIntegrityConflict("TaskClosed 缺少已迁移 Review")
        accepted = terminal.payload.get("accepted")
        can_answer = terminal.payload.get("can_answer")
        final_id = _stable_id("record", self.migration_version, self._snapshot.task_id, "final-answer", terminal.event_id)
        final_exists = final_id in self._records
        if accepted is True and can_answer is True and not final_exists:
            raise MigrationUnsupported("accepted TaskClosed 缺少 FinalAnswer")
        if accepted is False and can_answer is False and not final_exists:
            raise MigrationUnsupported("unanswerable TaskClosed 缺少 answerability explanation")
        status = "succeeded" if accepted is True and can_answer is True else "unanswerable"
        self._append_task_close(
            source_event=terminal,
            outcome={
                "status": status,
                "review_record_ref": review_id,
                **({"final_answer_ref": final_id} if final_exists else {}),
            },
        )

    def _append_failed_terminal(self, terminal: _SourceEvent) -> None:
        """把显式 source failure 关联到唯一失败 Module Run。"""

        reason = terminal.payload.get("reason_code", terminal.payload.get("error_type"))
        reason = _require_text(reason, "InvestigationTaskFailed.reason_code")
        open_modules = [module for module in self._modules if not module.closed]
        if len(open_modules) > 1:
            raise MigrationIntegrityConflict("失败 terminal 前存在多个 Open Module Run")
        if open_modules:
            module = open_modules[0]
            module.closed = True
            module.failure_code = reason
            self._append_target_event(
                event_id=_stable_id(
                    "module-failed-event",
                    self.migration_version,
                    self._snapshot.task_id,
                    module.module_invocation_id,
                ),
                event_type="ModuleInvocationFailed",
                payload={
                    "module_invocation_id": module.module_invocation_id,
                    "failure_code": reason,
                },
                recorded_at=terminal.created_at,
                source_event_ids=(terminal.event_id,),
                module_invocation_id=module.module_invocation_id,
            )
        else:
            module = _TargetModule(
                module_invocation_id=_stable_id("module", self.migration_version, self._snapshot.task_id, "failed", terminal.event_id),
                module_kind="hypothesis-investigation",
                input_scientific_record_ids=(),
                module_input_digest=_input_digest(self._snapshot.task_id, "investigation", ()),
                context_binding=self._context_binding(
                    "hypothesis-investigation",
                    self._synthetic_runtime("failed", terminal.event_id),
                    hypothesis_batch_record_id=next(iter(self._batch_record_ids.values()), None),
                ),
                source_anchor=terminal.event_id,
                records=[],
                closed=True,
                failure_code=reason,
            )
            self._append_module(module)
            # _append_module does not append failure closure; add it after start.
            self._append_target_event(
                event_id=_stable_id(
                    "module-failed-event",
                    self.migration_version,
                    self._snapshot.task_id,
                    module.module_invocation_id,
                ),
                event_type="ModuleInvocationFailed",
                payload={
                    "module_invocation_id": module.module_invocation_id,
                    "failure_code": reason,
                },
                recorded_at=terminal.created_at,
                source_event_ids=(terminal.event_id,),
                module_invocation_id=module.module_invocation_id,
            )
        self._append_target_event(
            event_id=_stable_id("task-failed-event", self.migration_version, self._snapshot.task_id, terminal.event_id),
            event_type="ResearchTaskFailed",
            payload={
                "outcome": {
                    "status": "failed",
                    "module_invocation_id": module.module_invocation_id,
                    "failure_code": reason,
                }
            },
            recorded_at=terminal.created_at,
            source_event_ids=(terminal.event_id,),
        )

    def _append_task_close(self, *, source_event: _SourceEvent, outcome: Mapping[str, Any]) -> None:
        """追加唯一目标 TaskClosed 事件。"""

        self._append_target_event(
            event_id=_stable_id("task-closed-event", self.migration_version, self._snapshot.task_id, source_event.event_id),
            event_type="ResearchTaskClosed",
            payload={"outcome": dict(outcome)},
            recorded_at=source_event.created_at,
            source_event_ids=(source_event.event_id,),
        )

    def _append_module(self, module: _TargetModule) -> None:
        """把计划模块展开成 Start、Records、Completed 事件。"""

        if not module.closed and module.records:
            raise MigrationIntegrityConflict(
                "Open Module Run 不得携带 ScientificRecord；输出必须一次原子提交"
            )
        if module.input_scientific_record_ids:
            for ref in module.input_scientific_record_ids:
                if ref not in self._records:
                    raise MigrationIntegrityConflict(f"Module input 引用了未知目标记录 {ref}")
        if module.closed and module.failure_code is None:
            available = dict(self._records)
            available.update({record.record.scientific_record_id: record.record for record in module.records})
            # V2 output group 的完整合同在内存计划阶段核对，避免失败后才创建
            # v2_* 表；调用私有核对函数只读取领域对象，不写数据库。
            from .v2_ledger import _validate_module_output_group

            try:
                _validate_module_output_group(
                    module.module_kind,
                    tuple(record.record for record in module.records),
                    available,
                )
            except ContractValidationError as exc:
                raise MigrationIntegrityConflict(
                    f"迁移模块 {module.module_invocation_id} 的输出成组合同无效"
                ) from exc
        self._modules.append(module)
        start_event_id = _stable_id(
            "module-start-event", self.migration_version, self._snapshot.task_id, module.module_invocation_id
        )
        module_target_kind = "module"
        continuation_index = 0
        while (module.source_anchor, module_target_kind) in self._source_event_target_events:
            continuation_index += 1
            module_target_kind = f"module-continuation-{continuation_index}"
        self._append_target_event(
            event_id=start_event_id,
            event_type="ModuleInvocationStarted",
            payload={
                "research_task_id": self._snapshot.task_id,
                "module_invocation_id": module.module_invocation_id,
                "module_kind": module.module_kind,
                "input_scientific_record_ids": list(module.input_scientific_record_ids),
                "module_input_digest": module.module_input_digest,
                "context_binding": module.context_binding.to_dict(),
            },
            recorded_at=self._source_event(module.source_anchor).created_at,
            source_event_ids=(module.source_anchor,),
            module_invocation_id=module.module_invocation_id,
            source_target_kind=module_target_kind,
        )
        self._source_module_ids[module.source_anchor] = module.module_invocation_id
        for record in module.records:
            self._source_module_ids.setdefault(
                record.source_event_id, module.module_invocation_id
            )
            self._append_target_event(
                event_id=record.event_id,
                event_type="ScientificRecordCreated",
                payload=record.record.canonical_identity_dict(),
                recorded_at=record.recorded_at,
                source_event_ids=(record.source_event_id,),
                module_invocation_id=module.module_invocation_id,
                record=record,
                source_target_kind=(
                    "query-record"
                    if module.module_kind == "research-query-formulation"
                    else None
                ),
            )
        if module.closed and module.failure_code is None:
            output_records = tuple(record.record for record in module.records)
            output_digest = _prepared_output_digest(output_records)
            completion_sources = tuple(
                dict.fromkeys(
                    record.source_event_id for record in module.records
                )
                or (module.source_anchor,)
            )
            completion_target_kind = "completion"
            completion_index = 0
            while any(
                (source_id, completion_target_kind)
                in self._source_event_target_events
                for source_id in completion_sources
            ):
                completion_index += 1
                completion_target_kind = f"completion-{completion_index}"
            self._append_target_event(
                event_id=_stable_id("module-completed-event", self.migration_version, self._snapshot.task_id, module.module_invocation_id),
                event_type="ModuleInvocationCompleted",
                payload={
                    "module_invocation_id": module.module_invocation_id,
                    "output_scientific_record_ids": [
                        record.record.scientific_record_id for record in module.records
                    ],
                    "module_output_digest": output_digest,
                },
                recorded_at=max(
                    [record.recorded_at for record in module.records]
                    or [self._source_event(module.source_anchor).created_at]
                ),
                source_event_ids=completion_sources,
                module_invocation_id=module.module_invocation_id,
                source_target_kind=completion_target_kind,
            )

    def _append_target_event(
        self,
        *,
        event_id: str,
        event_type: str,
        payload: Mapping[str, Any],
        recorded_at: str,
        source_event_ids: Sequence[str],
        module_invocation_id: str | None = None,
        record: _TargetRecord | None = None,
        source_target_kind: str | None = None,
    ) -> None:
        """追加内存目标事件并记录 source→target 关系。"""

        if any(event.event_id == event_id for event in self._events):
            raise MigrationIntegrityConflict(f"目标 event_id 重复：{event_id}")
        self._events.append(
            _TargetEvent(
                event_id=event_id,
                event_type=event_type,
                payload=json.loads(canonical_json(dict(payload))),
                recorded_at=_normalise_timestamp(recorded_at, "target.recorded_at"),
                source_event_ids=tuple(dict.fromkeys(source_event_ids)),
                module_invocation_id=module_invocation_id,
                record=record,
            )
        )
        if record is not None:
            self._register_record(record)
        for source_event_id in source_event_ids:
            if record is not None:
                target_kind = source_target_kind or "record"
            elif event_type == "ModuleInvocationStarted":
                target_kind = source_target_kind or "module"
            elif event_type == "ModuleInvocationCompleted":
                target_kind = source_target_kind or "completion"
            elif event_type == "ModuleInvocationFailed":
                target_kind = "failure"
            elif event_type in {"ResearchTaskClosed", "ResearchTaskFailed"}:
                target_kind = "terminal"
            else:
                target_kind = "audit"
            target_id = (
                record.record.scientific_record_id
                if record is not None
                else module_invocation_id
            )
            self._map_source(source_event_id, target_kind, target_id, event_id)

    def _register_record(self, record: _TargetRecord) -> None:
        """登记目标记录及 source event 映射，拒绝身份重复。"""

        record_id = record.record.scientific_record_id
        existing = self._records.get(record_id)
        if existing is not None and existing.record.canonical_identity_dict() != record.record.canonical_identity_dict():
            raise MigrationIntegrityConflict(f"目标 Scientific Record {record_id} 发生漂移")
        self._records[record_id] = record
        # 一个旧 BatchCreated 可以同时锚定 QueryRecord 和
        # HypothesisBatchRecord；事件映射通过 target_kind 区分，source_record_ids
        # 只保留第一个稳定身份供旧 generation_source 解析。
        self._source_record_ids.setdefault(record.source_event_id, record_id)

    def _map_source(
        self,
        source_event_id: str,
        target_kind: str,
        target_id: str | None,
        target_event_id: str | None,
    ) -> None:
        """保存内存 source→target 映射；最终同事务落表。"""

        key = (source_event_id, target_kind)
        value = (target_id, target_event_id)
        if key in self._source_event_target_events and self._source_event_target_events[key] != value:
            raise MigrationIntegrityConflict("source→target event map 发生漂移")
        self._source_event_target_events[key] = value

    def _source_event(self, event_id: str) -> _SourceEvent:
        """按 source event ID 取已核对事件。"""

        try:
            return self._snapshot.by_id[event_id]
        except KeyError as exc:
            raise MigrationIntegrityConflict(f"缺少 source event {event_id}") from exc

    def _record_by_id(self, record_id: str) -> _TargetRecord:
        """按目标 record ID 取记录。"""

        try:
            return self._records[record_id]
        except KeyError as exc:
            raise MigrationIntegrityConflict(f"缺少目标 record {record_id}") from exc

    def _synthetic_runtime(self, kind: str, source_id: str) -> str:
        """为已闭合历史模块生成仅供审计的稳定 runtime 身份。"""

        return _stable_id("runtime", self.migration_version, self._snapshot.task_id, kind, source_id)

    def _resolve_runtime(
        self,
        *,
        module_kind: str,
        input_scientific_record_ids: Sequence[str],
        source_event_ids: Sequence[str],
    ) -> ExistingHarnessRuntime:
        """调用只读 resolver；任何缺失或冲突都 fail closed。"""

        resolver = self.runtime_resolver
        if resolver is None:
            raise MigrationRecoveryEvidenceMissing(
                f"Open {module_kind} Module Run 缺少 runtime resolver"
            )
        method = getattr(resolver, "resolve", None)
        if not callable(method):
            if callable(resolver):
                method = resolver
            else:
                raise MigrationRecoveryEvidenceMissing("runtime resolver 没有只读 resolve 接口")
        digest = _input_digest(
            self._snapshot.task_id,
            module_kind,
            input_scientific_record_ids,
        )
        try:
            raw = method(
                self._snapshot.task_id,
                source_event_ids=tuple(source_event_ids),
                expected_module_kind=module_kind,
                expected_module_input_digest=digest,
            )
        except MigrationError:
            raise
        except Exception as exc:
            raise MigrationRecoveryEvidenceMissing("runtime resolver 读取失败") from exc
        if isinstance(raw, ExistingHarnessRuntime):
            runtime = raw
        elif isinstance(raw, Mapping):
            status = raw.get("status")
            if status in {"RecoveryEvidenceMissing", "VerifiedIntegrityConflict", "TemporarilyUnavailable"}:
                error = MigrationRecoveryEvidenceMissing if status != "VerifiedIntegrityConflict" else MigrationIntegrityConflict
                raise error(f"runtime resolver 返回 {status}")
            nested = raw.get("runtime") if isinstance(raw.get("runtime"), Mapping) else raw
            runtime = ExistingHarnessRuntime(
                runtime_ref=nested.get("runtime_ref"),
                harness_run_ref=nested.get("harness_run_ref"),
                state=nested.get("state", "open"),
                prepared_output=nested.get("prepared_output"),
            )
        else:
            raise MigrationRecoveryEvidenceMissing("runtime resolver 返回无法解析的身份")
        if runtime.state == "failed":
            raise MigrationIntegrityConflict("未有明确 source failure 时不能迁移 failed runtime")
        return runtime

    def _open_source_ids_for_batch(
        self,
        batch_id: str,
        missing_combinations: Sequence[str],
    ) -> tuple[str, ...]:
        """收集未形成 Result 的 session/turn/waiting source 身份。"""

        missing = set(missing_combinations)
        ids: list[str] = []
        for event in self._snapshot.events:
            if event.event_type not in {
                "investigation_session_started",
                "investigation_turn_started",
                "investigation_turn_waiting",
                "investigation_incomplete_v2_waiting_resume_authorized",
            }:
                continue
            if event.payload.get("batch_id") != batch_id:
                continue
            if event.payload.get("combination_id") in missing or event.event_type == "investigation_turn_waiting":
                ids.append(event.event_id)
        return tuple(dict.fromkeys(ids))

    def _open_module_exists(self) -> bool:
        """返回计划中是否已经有未闭合模块。"""

        return any(not module.closed for module in self._modules)

    def _latest_investigation_module(self) -> _TargetModule | None:
        """返回最近的 Investigation 模块。"""

        for module in reversed(self._modules):
            if module.module_kind == "hypothesis-investigation":
                return module
        return None

    def _investigation_module_for_result(self, info: _ResultInfo) -> _TargetModule | None:
        """返回某个旧 Result 前驱所在的目标模块。"""

        source_id = info.predecessor_event_id or info.source_event.event_id
        module_id = self._source_module_ids.get(source_id)
        if module_id is None:
            # source result event 被记录事件映射，但 module anchor 是 batch；按
            # runtime 复用要求寻找最近闭合 Investigation 即可。
            return self._latest_investigation_module()
        return next((module for module in self._modules if module.module_invocation_id == module_id), None)

    def _previous_review_module(self) -> _TargetModule | None:
        """返回最近闭合的 Review 模块。"""

        for module in reversed(self._modules):
            if module.module_kind == "task-review" and module.closed:
                return module
        return None

    def _mark_internal_sources(self, snapshot: _SourceSnapshot) -> None:
        """为未生成科学记录的内部 source event 建立 audit map。"""

        for event in snapshot.events:
            if event.event_type in _INTERNAL_SOURCE_EVENT_TYPES:
                if (event.event_id, "audit") in self._source_event_target_events:
                    continue
                self._map_source(event.event_id, "audit", event.event_id, None)

    def _freeze_plan(
        self,
        snapshot: _SourceSnapshot,
        task_payload: Mapping[str, Any],
    ) -> MigrationPlan:
        """校验目标事件顺序和科学引用，并冻结可写计划。"""

        if not self._events or self._events[0].event_type != "ResearchTaskCreated":
            raise MigrationIntegrityConflict("目标计划缺少首个 TaskCreated")
        if any(
            event.event_type in {"ResearchTaskClosed", "ResearchTaskFailed"}
            for event in self._events[:-1]
        ):
            raise MigrationIntegrityConflict("目标 terminal 之后仍有事件")
        target_records = set()
        for event in self._events:
            if event.event_type == "ScientificRecordCreated":
                if event.record is None:
                    raise MigrationIntegrityConflict("目标 ScientificRecord 事件缺少记录")
                record = event.record.record
                if any(ref not in target_records for ref in record.source_scientific_record_ids):
                    raise MigrationIntegrityConflict("目标 ScientificRecord 引用未来或缺失记录")
                target_records.add(record.scientific_record_id)
        serialized_events = tuple(
            {
                "event_id": event.event_id,
                "event_type": event.event_type,
                "payload": dict(event.payload),
                "recorded_at": event.recorded_at,
                "source_event_ids": list(event.source_event_ids),
                "module_invocation_id": event.module_invocation_id,
            }
            for event in self._events
        )
        serialized_map = tuple(
            {
                "source_event_id": source_id,
                "target_kind": target_kind,
                "target_id": target_id,
                "target_event_id": target_event_id,
            }
            for (source_id, target_kind), (target_id, target_event_id) in sorted(
                self._source_event_target_events.items()
            )
        )
        return MigrationPlan(
            research_task_id=snapshot.task_id,
            migration_version=self.migration_version,
            source_digest=snapshot.source_digest,
            source_event_count=len(snapshot.events),
            task_payload=json.loads(canonical_json(task_payload)),
            events=serialized_events,
            event_map=serialized_map,
            scientific_record_ids=tuple(record_id for record_id in self._records),
        )

    @staticmethod
    def _ensure_migration_schema(connection: sqlite3.Connection) -> None:
        """创建同库迁移审计表；调用方已持有唯一写事务。"""

        # 不使用 executescript：sqlite3 会在 executescript 前隐式提交当前事务，
        # 这会破坏“目标行、事件、投影、marker 单个 BEGIN IMMEDIATE”的原子性。
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS v2_migration_markers (
                task_id TEXT PRIMARY KEY,
                migration_version TEXT NOT NULL,
                source_digest TEXT NOT NULL,
                source_event_count INTEGER NOT NULL,
                target_head_digest TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status='complete'),
                migrated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS v2_migration_event_map (
                task_id TEXT NOT NULL,
                source_event_id TEXT NOT NULL,
                source_event_key TEXT NOT NULL,
                target_kind TEXT NOT NULL,
                target_id TEXT,
                target_event_id TEXT,
                PRIMARY KEY(task_id, source_event_id, target_kind)
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS v2_migration_event_map_target_idx
                ON v2_migration_event_map(task_id, target_kind, target_id)
            """
        )

    def _write_plan(
        self,
        connection: sqlite3.Connection,
        ledger: ResearchTaskLedger,
        plan: MigrationPlan,
    ) -> None:
        """在同一个 BEGIN IMMEDIATE 中写目标行、事件、投影和 marker。"""

        self._ensure_migration_schema(connection)
        if connection.execute(
            "SELECT 1 FROM research_tasks WHERE research_task_id=?",
            (plan.research_task_id,),
        ).fetchone() is not None:
            raise MigrationIntegrityConflict("目标 task 在写入前出现")
        connection.execute(
            """
            INSERT INTO research_tasks(
                research_task_id, contract_version, input_json, policy_json,
                dependencies_json, research_task_created_event_id
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                plan.research_task_id,
                canonical_json(plan.task_payload["contract_version"]),
                canonical_json(plan.task_payload["input"]),
                canonical_json(plan.task_payload["policy"]),
                canonical_json(plan.task_payload["dependencies"]),
                plan.events[0]["event_id"],
            ),
        )
        previous_digest: str | None = None
        module_by_id: dict[str, Mapping[str, Any]] = {}
        records_by_id: dict[str, ScientificRecord] = {}
        for sequence, event in enumerate(plan.events, start=1):
            payload = event["payload"]
            digest = ledger._event_digest(
                event_id=event["event_id"],
                research_task_id=plan.research_task_id,
                research_task_sequence=sequence,
                event_type=event["event_type"],
                payload=payload,
                recorded_at=event["recorded_at"],
                previous_event_digest=previous_digest,
            )
            connection.execute(
                """
                INSERT INTO research_task_events(
                    event_id, research_task_id, research_task_sequence, event_type, payload_json,
                    recorded_at, event_digest, previous_event_digest
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event["event_id"],
                    plan.research_task_id,
                    sequence,
                    event["event_type"],
                    canonical_json(payload),
                    event["recorded_at"],
                    digest,
                    previous_digest,
                ),
            )
            if event["event_type"] == "ModuleInvocationStarted":
                request = ModuleInvocationRequest.from_mapping(
                    {"research_task_id": plan.research_task_id, **payload}
                )
                ledger._upsert_module_invocation_projection(
                    connection,
                    research_task_id=plan.research_task_id,
                    request=request,
                    started_event_id=event["event_id"],
                )
                module_by_id[request.module_invocation_id] = payload
            elif event["event_type"] == "ScientificRecordCreated":
                record = ScientificRecord.from_mapping(payload).with_recorded_at(
                    event["recorded_at"]
                )
                records_by_id[record.scientific_record_id] = record
                ledger._upsert_scientific_record_projection(
                    connection,
                    record=record,
                    created_event_id=event["event_id"],
                )
            elif event["event_type"] == "ModuleInvocationCompleted":
                module_id = payload["module_invocation_id"]
                request_payload = module_by_id[module_id]
                ledger._upsert_module_invocation_projection(
                    connection,
                    research_task_id=plan.research_task_id,
                    request=ModuleInvocationRequest.from_mapping(
                        {"research_task_id": plan.research_task_id, **request_payload}
                    ),
                    started_event_id=next(
                        item["event_id"]
                        for item in plan.events
                        if item["event_type"] == "ModuleInvocationStarted"
                        and item["payload"]["module_invocation_id"] == module_id
                    ),
                    completed_event_id=event["event_id"],
                    output_scientific_record_ids=payload["output_scientific_record_ids"],
                    module_output_digest=payload["module_output_digest"],
                )
            elif event["event_type"] == "ModuleInvocationFailed":
                module_id = payload["module_invocation_id"]
                request_payload = module_by_id[module_id]
                ledger._upsert_module_invocation_projection(
                    connection,
                    research_task_id=plan.research_task_id,
                    request=ModuleInvocationRequest.from_mapping(
                        {"research_task_id": plan.research_task_id, **request_payload}
                    ),
                    started_event_id=next(
                        item["event_id"]
                        for item in plan.events
                        if item["event_type"] == "ModuleInvocationStarted"
                        and item["payload"]["module_invocation_id"] == module_id
                    ),
                )
            previous_digest = digest
        facts = ledger._load_task_facts_in_connection(connection, plan.research_task_id)
        ledger._upsert_task_facts_projection(connection, facts)
        head_digest = previous_digest or ""
        connection.execute(
            """
            INSERT INTO v2_migration_markers(
                task_id, migration_version, source_digest, source_event_count,
                target_head_digest, status, migrated_at
            ) VALUES (?, ?, ?, ?, ?, 'complete', ?)
            """,
            (
                plan.research_task_id,
                plan.migration_version,
                plan.source_digest,
                plan.source_event_count,
                head_digest,
                datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z"),
            ),
        )
        source_events = {event.event_id: event for event in self._snapshot.events}
        for item in plan.event_map:
            source = source_events.get(item["source_event_id"])
            if source is None:
                raise MigrationIntegrityConflict("event map 引用了未知 source event")
            connection.execute(
                """
                INSERT INTO v2_migration_event_map(
                    task_id, source_event_id, source_event_key, target_kind,
                    target_id, target_event_id
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    plan.research_task_id,
                    source.event_id,
                    source.event_key,
                    item["target_kind"],
                    item["target_id"],
                    item["target_event_id"],
                ),
            )

    def _result_from_plan(self, plan: MigrationPlan) -> MigrationResult:
        """使用计划的确定性事件身份计算返回值。"""

        previous: str | None = None
        head = ""
        target_event_ids: list[str] = []
        for sequence, event in enumerate(plan.events, start=1):
            head = hashlib.sha256(
                canonical_json(
                    {
                        "event_id": event["event_id"],
                        "research_task_id": plan.research_task_id,
                        "research_task_sequence": sequence,
                        "event_type": event["event_type"],
                        "payload": event["payload"],
                        "recorded_at": event["recorded_at"],
                        "previous_event_digest": previous,
                    }
                ).encode("utf-8")
            ).hexdigest()
            previous = head
            target_event_ids.append(event["event_id"])
        return MigrationResult(
            research_task_id=plan.research_task_id,
            migration_version=plan.migration_version,
            source_digest=plan.source_digest,
            source_event_count=plan.source_event_count,
            target_event_ids=tuple(target_event_ids),
            target_scientific_record_ids=plan.scientific_record_ids,
            target_head_digest=head,
        )


def migrate_existing_task(
    task_id: str,
    *,
    db_path: str | os.PathLike[str],
    source_root: str | os.PathLike[str] | None = None,
    runtime_resolver: RuntimeResolver | None = None,
    migration_version: str = MIGRATION_VERSION,
) -> MigrationResult:
    """函数式迁移入口，等价于 ``InvestigationV2Migration(...).migrate_existing_task``。"""

    return InvestigationV2Migration(
        db_path,
        runtime_resolver=runtime_resolver,
        migration_version=migration_version,
    ).migrate_existing_task(task_id, source_root=source_root)


__all__ = [
    "ExistingHarnessRuntime",
    "HarnessRuntimeResolver",
    "InvestigationV2Migration",
    "MigrationError",
    "MigrationIntegrityConflict",
    "MigrationPlan",
    "MigrationRecoveryEvidenceMissing",
    "MigrationResult",
    "MigrationSourceDrift",
    "MigrationUnsupported",
    "MIGRATION_VERSION",
    "migrate_existing_task",
]
