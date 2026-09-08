"""schema-v10 历史数据的显式只读读取入口。

当前生产任务使用 :mod:`v2_ledger` 的 ``ResearchTaskWorkingMemory``。本模块只
允许迁移工具读取已经存在的 schema-v10 数据；它不构造活动 Working Memory
服务，也不会初始化、迁移或写入 SQLite 数据库。
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Optional

from .memory_service import CheckpointResumeContext
from .memory_service import WorkingMemoryService as _SchemaV10LegacyReaderModel


class SchemaV10ReadOnlyError(RuntimeError):
    """历史 schema-v10 读取器收到写入请求时抛出的错误。"""


class _ReadOnlySQLiteStore:
    """为历史读取方法提供 ``mode=ro`` 的最小 SQLite 存储接口。

    该对象故意只实现旧读取方法使用的连接和查询辅助函数。没有
    ``initialize`` 或可提交写事务的方法，因而构造读取器不会创建目录、建表或
    执行 schema 迁移。
    """

    def __init__(self, path: str) -> None:
        self.path = os.path.abspath(os.path.expanduser(path))
        database = Path(self.path)
        if not database.is_file():
            raise FileNotFoundError(
                f"schema-v10 只读数据库不存在或不是普通文件：{self.path}"
            )

    def connect(self) -> sqlite3.Connection:
        """打开 SQLite URI ``mode=ro`` 连接，不执行会改变数据库的 PRAGMA。"""

        connection = sqlite3.connect(
            f"file:{self.path}?mode=ro",
            uri=True,
            timeout=30,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA query_only = ON")
        return connection

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """在只读连接上执行一个查询事务。"""

        connection = self.connect()
        try:
            connection.execute("BEGIN")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    @contextmanager
    def immediate_transaction(self) -> Iterator[sqlite3.Connection]:
        """拒绝历史服务的所有写事务。"""

        raise SchemaV10ReadOnlyError(
            "schema-v10 读取器不提供写事务；请使用 ResearchTaskWorkingMemory"
        )
        yield  # pragma: no cover - 仅用于满足生成器类型

    def fetchone(
        self,
        sql: str,
        params: Iterable[Any] = (),
    ) -> Optional[sqlite3.Row]:
        """执行单行只读查询。"""

        connection = self.connect()
        try:
            return connection.execute(sql, tuple(params)).fetchone()
        finally:
            connection.close()

    def fetchall(
        self,
        sql: str,
        params: Iterable[Any] = (),
    ) -> List[sqlite3.Row]:
        """执行多行只读查询。"""

        connection = self.connect()
        try:
            return connection.execute(sql, tuple(params)).fetchall()
        finally:
            connection.close()


class SchemaV10ReadOnlyWorkingMemoryReader:
    """schema-v10 历史数据库的 fail-closed 只读 facade。

    读取方法通过显式白名单绑定到历史模型的查询实现。旧模型只作为方法载体由
    ``__new__`` 配置，不调用其构造器；因此不会触发 ``SQLiteMemoryStore`` 的
    ``initialize``、建表、迁移或 ``BEGIN IMMEDIATE``。任何不在白名单中的属性，
    包括所有写入方法，均表现为不存在的接口。
    """

    # 这些方法只读取 schema-v10 的历史事件、投影和恢复快照。名单保持显式，
    # 不用 ``get_``/``list_`` 等前缀推断，避免新写方法意外落入兼容层。
    _READ_METHODS = frozenset(
        {
            "get_restart_guide",
            "get_unresolved_task_pause",
            "get_unresolved_combination_pause",
            "list_unresolved_combination_pauses",
            "read_terminal_result_paths",
            "get_active_workflow_revision",
            "get_workflow_revision_source_attempt_id",
            "get_task_created",
            "get_memory_event_type",
            "get_investigation_repair_event_identity",
            "get_effective_scientific_context",
            "get_pending_hypothesis_restart",
            "get_task",
            "get_unresolved_retrieval_pause",
            "get_retrieval_completed",
            "get_batch",
            "find_batch_by_index",
            "find_latest_batch",
            "get_batch_created",
            "get_batch_created_receipt",
            "get_combination",
            "get_expert_debate",
            "get_checkpoint_resume_context",
            "list_candidates",
            "list_combinations",
            "get_task_architecture_version",
            "list_investigation_legacy_waiting_resume_authorizations",
            "list_investigation_incomplete_v2_waiting_resume_authorizations",
            "check_revise_hypothesis_route_gate",
            "check_investigation_route_marker_gate",
            "check_revise_hypothesis_batch_gate",
            "check_investigation_batch_route_gate",
            "get_investigation_session",
            "list_investigation_sessions",
            "get_investigation_recovery_snapshot",
            "get_investigation_result",
            "list_investigation_results",
            "get_investigation_review",
            "get_investigation_review_for_replay",
            "list_investigation_reviews",
            "list_investigation_repairs",
            "list_investigation_route_markers",
            "list_investigation_turns",
            "list_investigation_turn_waits",
            "list_investigation_turn_attempt_terminations",
            "get_investigation_task_terminal",
            "get_hypothesis_selection_completed",
            "get_investigation_task_failure",
            "get_task_review_session",
            "list_task_review_turns",
            "list_investigation_evidence_verifications",
            "list_investigation_repair_turns",
            "verify_investigation_current_index",
        }
    )

    def __init__(
        self,
        db_path: str,
        *,
        enabled: bool,
        scratch_root: str,
    ) -> None:
        """绑定已有数据库；启用时只打开现有文件的 SQLite 只读连接。"""

        self.db_path = os.path.abspath(os.path.expanduser(db_path))
        self.enabled = bool(enabled)
        self.scratch_root = os.path.abspath(os.path.expanduser(scratch_root))
        self.task_id: Optional[str] = None
        self._reader = _SchemaV10LegacyReaderModel.__new__(
            _SchemaV10LegacyReaderModel
        )
        self._reader.enabled = self.enabled
        self._reader.scratch_root = self.scratch_root
        self._reader.task_id = None
        self._reader.memory_status = "available" if self.enabled else "disabled"
        self._reader.db_path = self.db_path
        self._reader.store = (
            _ReadOnlySQLiteStore(self.db_path) if self.enabled else None
        )

    def status(self) -> dict[str, Any]:
        """返回历史读取器状态；该状态仍属于 schema-v10 兼容形状。"""

        return {
            "memory_status": "available" if self.enabled else "disabled",
            "task_id": self.task_id,
        }

    def close(self) -> None:
        """关闭读取器的短生命周期资源；读取连接按查询自动关闭。"""

        return None

    def __getattr__(self, name: str) -> Any:
        """只绑定白名单查询；未知属性和写入接口均 fail-closed。"""

        if name not in self._READ_METHODS:
            raise AttributeError(f"schema-v10 只读 facade 不提供该接口：{name}")
        reader = object.__getattribute__(self, "_reader")
        return getattr(reader, name)

    def __dir__(self) -> list[str]:
        """只列出只读 facade 的固定接口。"""

        return sorted(
            {
                "close",
                "db_path",
                "enabled",
                "scratch_root",
                "status",
                "task_id",
                *_READ_METHODS,
            }
        )


__all__ = [
    "CheckpointResumeContext",
    "SchemaV10ReadOnlyError",
    "SchemaV10ReadOnlyWorkingMemoryReader",
]
