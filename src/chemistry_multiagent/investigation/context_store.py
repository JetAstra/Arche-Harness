"""Investigation 批次 Context、假设 Session 与 Invocation 的持久身份存储。"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONTEXT_STORE_SCHEMA_VERSION = 2
_CONTEXT_STORE_LEGACY_SCHEMA_VERSION = 1
_CONTEXT_STORE_TABLES = frozenset(
    {
        "investigation_contexts",
        "investigation_sessions",
        "investigation_invocations",
    }
)


class InvestigationContextStoreError(RuntimeError):
    """Investigation Context 持久事实无法读取或写入。"""


class InvestigationContextContractError(InvestigationContextStoreError, ValueError):
    """Context、Session 或 Invocation 身份违反公开合同。"""


class InvestigationContextIdentityConflict(InvestigationContextStoreError):
    """同一不可变身份已经绑定不同内容。"""


class InvestigationContextTransitionConflict(InvestigationContextStoreError):
    """Invocation 转换与当前持久事实不相容。"""


def _require_id(value: object, field_name: str) -> str:
    """校验不包含路径语义的公开身份。"""

    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise InvestigationContextContractError(f"{field_name} 必须是安全的非空标识")
    return value


def _require_digest(value: object, field_name: str) -> str:
    """校验固定格式的 SHA-256 摘要。"""

    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise InvestigationContextContractError(f"{field_name} 必须是 64 位小写 SHA-256")
    return value


def _require_ids(value: object, field_name: str) -> tuple[str, ...]:
    """校验有序且无重复的身份序列。"""

    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        raise InvestigationContextContractError(f"{field_name} 必须是标识序列")
    result = tuple(_require_id(item, f"{field_name}[]") for item in value)
    if not result or len(set(result)) != len(result):
        raise InvestigationContextContractError(f"{field_name} 必须非空且不能重复")
    return result


def _require_relative_directory(value: object) -> str:
    """校验不会越出当前 Run Bundle task root 的 POSIX 相对目录。"""

    if not isinstance(value, str) or not value:
        raise InvestigationContextContractError(
            "workspace_identity 必须是非空相对目录"
        )
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise InvestigationContextContractError(
            "workspace_identity 不能包含绝对路径或父目录"
        )
    return path.as_posix()


def _canonical_json(value: Any) -> str:
    """生成稳定 SQLite identity JSON。"""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True, init=False)
class InvestigationContextIdentity:
    """一个假设批次跨首次调查和返修共享的 Context 身份。

    ``hypothesis_execution_decision_record_id`` 只在确认后的执行输入路径提供，
    用于把 Context 绑定到本轮研究员决定。旧的 SelectedHypothesisRecord 路径
    保持该字段为 ``None``，因而仍使用原有三字段身份。
    """

    investigation_context_id: str
    research_task_id: str
    hypothesis_batch_record_id: str
    hypothesis_execution_decision_record_id: str | None = None

    def __init__(
        self,
        investigation_context_id: str,
        research_task_id: str,
        hypothesis_batch_record_id: str,
        hypothesis_execution_decision_record_id: str | None = None,
    ) -> None:
        """创建正式 Context 身份。"""

        object.__setattr__(self, "investigation_context_id", investigation_context_id)
        object.__setattr__(self, "research_task_id", research_task_id)
        object.__setattr__(self, "hypothesis_batch_record_id", hypothesis_batch_record_id)
        object.__setattr__(
            self,
            "hypothesis_execution_decision_record_id",
            hypothesis_execution_decision_record_id,
        )
        self.__post_init__()

    def __post_init__(self) -> None:
        for field_name in (
            "investigation_context_id",
            "research_task_id",
            "hypothesis_batch_record_id",
        ):
            object.__setattr__(self, field_name, _require_id(getattr(self, field_name), field_name))
        if self.hypothesis_execution_decision_record_id is not None:
            object.__setattr__(
                self,
                "hypothesis_execution_decision_record_id",
                _require_id(
                    self.hypothesis_execution_decision_record_id,
                    "hypothesis_execution_decision_record_id",
                ),
            )


@dataclass(frozen=True, slots=True, init=False)
class InvestigationSessionIdentity:
    """Context 内一个入选假设独占的模型、workspace 与 MiniChem 身份。"""

    investigation_session_id: str
    investigation_context_id: str
    research_task_id: str
    hypothesis_execution_input_record_id: str
    workspace_identity: str
    minichem_scope_id: str
    scope_contract_digest: str
    capability_generation_digest: str | None = None

    def __init__(
        self,
        investigation_session_id: str,
        investigation_context_id: str,
        research_task_id: str,
        hypothesis_execution_input_record_id: str,
        workspace_identity: str,
        minichem_scope_id: str,
        scope_contract_digest: str,
        capability_generation_digest: str | None = None,
    ) -> None:
        """创建只含正式字段的 Session 身份。"""

        for name, value in (
            ("investigation_session_id", investigation_session_id),
            ("investigation_context_id", investigation_context_id),
            ("research_task_id", research_task_id),
            ("hypothesis_execution_input_record_id", hypothesis_execution_input_record_id),
            ("workspace_identity", workspace_identity),
            ("minichem_scope_id", minichem_scope_id),
            ("scope_contract_digest", scope_contract_digest),
            ("capability_generation_digest", capability_generation_digest),
        ):
            object.__setattr__(self, name, value)
        self.__post_init__()

    def __post_init__(self) -> None:
        for field_name in (
            "investigation_session_id",
            "investigation_context_id",
            "research_task_id",
            "hypothesis_execution_input_record_id",
            "minichem_scope_id",
        ):
            object.__setattr__(self, field_name, _require_id(getattr(self, field_name), field_name))
        object.__setattr__(
            self,
            "workspace_identity",
            _require_relative_directory(self.workspace_identity),
        )
        object.__setattr__(
            self,
            "scope_contract_digest",
            _require_digest(
                self.scope_contract_digest,
                "scope_contract_digest",
            ),
        )
        if self.capability_generation_digest is not None:
            object.__setattr__(
                self,
                "capability_generation_digest",
                _require_digest(
                    self.capability_generation_digest,
                    "capability_generation_digest",
                ),
            )

@dataclass(frozen=True, slots=True)
class InvestigationInvocationIdentity:
    """一次 initial 或 repair Module Invocation 的不可变输入边界。"""

    module_invocation_id: str
    investigation_context_id: str
    module_input_digest: str
    input_scientific_record_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "module_invocation_id",
            _require_id(self.module_invocation_id, "module_invocation_id"),
        )
        object.__setattr__(
            self,
            "investigation_context_id",
            _require_id(self.investigation_context_id, "investigation_context_id"),
        )
        object.__setattr__(
            self,
            "module_input_digest",
            _require_digest(self.module_input_digest, "module_input_digest"),
        )
        object.__setattr__(
            self,
            "input_scientific_record_ids",
            _require_ids(
                self.input_scientific_record_ids,
                "input_scientific_record_ids",
            ),
        )


@dataclass(frozen=True, slots=True)
class InvestigationContextSnapshot:
    """从 Context Store 读取的严格恢复视图。"""

    identity: InvestigationContextIdentity
    sessions: tuple[InvestigationSessionIdentity, ...]
    closed_invocations: tuple[InvestigationInvocationIdentity, ...]
    open_invocation: InvestigationInvocationIdentity | None


class SchemaV1LegacyContextReader:
    """schema-v1 Context SQLite 的明确只读读取器。

    v1 表中的 ``inquiry_id``、``hypothesis_record_id`` 和旧 workspace/scope
    字段只在这里被解析；读取结果立即归一为当前正式身份，当前 store 不会向
    v1 表追加或修改记录。
    """

    def __init__(self, database_path: str | os.PathLike[str]) -> None:
        self.database_path = Path(os.path.abspath(os.path.expanduser(os.fspath(database_path))))

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @staticmethod
    def _context_identity(value: Any) -> InvestigationContextIdentity:
        if not isinstance(value, dict):
            raise InvestigationContextStoreError("schema-v1 Context identity 已损坏")
        try:
            return InvestigationContextIdentity(
                investigation_context_id=value["investigation_context_id"],
                research_task_id=value["inquiry_id"],
                hypothesis_batch_record_id=value["hypothesis_batch_record_id"],
                hypothesis_execution_decision_record_id=value.get(
                    "hypothesis_execution_decision_record_id"
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise InvestigationContextStoreError(
                "schema-v1 Context identity 已损坏"
            ) from exc

    @staticmethod
    def _session_identity(value: Any) -> InvestigationSessionIdentity:
        if not isinstance(value, dict):
            raise InvestigationContextStoreError("schema-v1 Session identity 已损坏")
        try:
            return InvestigationSessionIdentity(
                investigation_session_id=value["investigation_session_id"],
                investigation_context_id=value["investigation_context_id"],
                research_task_id=value["inquiry_id"],
                hypothesis_execution_input_record_id=value["hypothesis_record_id"],
                workspace_identity=value["investigation_workspace_relpath"],
                minichem_scope_id=value["minichem_scope_id"],
                scope_contract_digest=value[
                    "minichem_scope_contract_digest"
                ],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise InvestigationContextStoreError(
                "schema-v1 Session identity 已损坏"
            ) from exc

    def read_context(
        self,
        identity: InvestigationContextIdentity,
    ) -> InvestigationContextSnapshot:
        """只读恢复 v1 快照，并归一为当前 dataclass。"""

        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT identity_json FROM investigation_contexts "
                    "WHERE investigation_context_id = ?",
                    (identity.investigation_context_id,),
                ).fetchone()
                if row is None:
                    raise InvestigationContextTransitionConflict(
                        "Investigation Context 不存在"
                    )
                stored_identity = self._context_identity(json.loads(row["identity_json"]))
                if stored_identity != identity:
                    raise InvestigationContextIdentityConflict(
                        "schema-v1 Investigation Context 身份发生漂移"
                    )
                session_rows = connection.execute(
                    "SELECT identity_json FROM investigation_sessions "
                    "WHERE investigation_context_id = ? ORDER BY rowid ASC",
                    (identity.investigation_context_id,),
                ).fetchall()
                invocation_rows = connection.execute(
                    "SELECT identity_json, module_output_digest "
                    "FROM investigation_invocations WHERE investigation_context_id = ? "
                    "ORDER BY invocation_sequence ASC",
                    (identity.investigation_context_id,),
                ).fetchall()
        except InvestigationContextStoreError:
            raise
        except (sqlite3.Error, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise InvestigationContextStoreError(
                "无法读取 schema-v1 Investigation Context"
            ) from exc
        try:
            sessions = tuple(
                self._session_identity(json.loads(row["identity_json"]))
                for row in session_rows
            )
            closed: list[InvestigationInvocationIdentity] = []
            opened: InvestigationInvocationIdentity | None = None
            for row in invocation_rows:
                value = json.loads(row["identity_json"])
                invocation = InvestigationInvocationIdentity(**value)
                if row["module_output_digest"] is None:
                    if opened is not None:
                        raise InvestigationContextStoreError(
                            "schema-v1 Context 存在多个 Open Invocation"
                        )
                    opened = invocation
                else:
                    _require_digest(row["module_output_digest"], "module_output_digest")
                    closed.append(invocation)
        except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise InvestigationContextStoreError(
                "schema-v1 Investigation Context 持久身份已损坏"
            ) from exc
        return InvestigationContextSnapshot(
            identity=identity,
            sessions=sessions,
            closed_invocations=tuple(closed),
            open_invocation=opened,
        )

    def read_completed_output_digest(
        self,
        context: InvestigationContextIdentity,
        module_invocation_id: str,
    ) -> str | None:
        """只读恢复 v1 Invocation 输出摘要。"""

        self.read_context(context)
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT investigation_context_id, module_output_digest "
                    "FROM investigation_invocations WHERE module_invocation_id = ?",
                    (_require_id(module_invocation_id, "module_invocation_id"),),
                ).fetchone()
        except sqlite3.Error as exc:
            raise InvestigationContextStoreError(
                "无法读取 schema-v1 Invocation 输出摘要"
            ) from exc
        if row is None:
            raise InvestigationContextTransitionConflict(
                "Investigation Invocation 不存在"
            )
        if row["investigation_context_id"] != context.investigation_context_id:
            raise InvestigationContextIdentityConflict(
                "Invocation 已绑定到另一个 Investigation Context"
            )
        if row["module_output_digest"] is None:
            return None
        return _require_digest(row["module_output_digest"], "module_output_digest")


class InvestigationContextStore:
    """持久化 Context 与 Session 绑定，封装跨返修复用不变量。"""

    def __init__(self, database_path: str | os.PathLike[str]) -> None:
        """初始化 SQLite 存储；不会创建模型、scope 或化学作业。"""

        if not isinstance(database_path, (str, os.PathLike)):
            raise InvestigationContextContractError("database_path 必须是路径")
        self.database_path = Path(
            os.path.abspath(os.path.expanduser(os.fspath(database_path)))
        )
        if self.database_path.exists() and not self.database_path.is_file():
            raise InvestigationContextContractError("database_path 必须是普通文件")
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._legacy_reader: SchemaV1LegacyContextReader | None = None
        try:
            with self._connect() as connection:
                existing_tables = {
                    row["name"]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    ).fetchall()
                }
                if (
                    _CONTEXT_STORE_TABLES.intersection(existing_tables)
                    and "investigation_context_metadata" not in existing_tables
                ):
                    raise InvestigationContextContractError(
                        "Investigation Context Store 缺少 schema version；必须显式迁移"
                    )
                metadata = connection.execute(
                    "SELECT schema_version FROM investigation_context_metadata "
                    "WHERE singleton_id = 1"
                ).fetchone() if "investigation_context_metadata" in existing_tables else None
                if metadata is not None and metadata["schema_version"] == _CONTEXT_STORE_LEGACY_SCHEMA_VERSION:
                    self._legacy_reader = SchemaV1LegacyContextReader(self.database_path)
                    return
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS investigation_context_metadata (
                        singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
                        schema_version INTEGER NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS investigation_contexts (
                        investigation_context_id TEXT PRIMARY KEY,
                        identity_json TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS investigation_sessions (
                        investigation_session_id TEXT PRIMARY KEY,
                        investigation_context_id TEXT NOT NULL,
                        research_task_id TEXT NOT NULL,
                        hypothesis_execution_input_record_id TEXT NOT NULL,
                        workspace_identity TEXT NOT NULL,
                        minichem_scope_id TEXT NOT NULL,
                        scope_contract_digest TEXT NOT NULL,
                        capability_generation_digest TEXT,
                        identity_json TEXT NOT NULL,
                        UNIQUE (investigation_context_id, hypothesis_execution_input_record_id),
                        FOREIGN KEY (investigation_context_id)
                            REFERENCES investigation_contexts(investigation_context_id)
                    );
                    CREATE TABLE IF NOT EXISTS investigation_invocations (
                        module_invocation_id TEXT PRIMARY KEY,
                        investigation_context_id TEXT NOT NULL,
                        invocation_sequence INTEGER NOT NULL,
                        identity_json TEXT NOT NULL,
                        module_output_digest TEXT,
                        UNIQUE (investigation_context_id, invocation_sequence),
                        FOREIGN KEY (investigation_context_id)
                            REFERENCES investigation_contexts(investigation_context_id)
                    );
                    CREATE UNIQUE INDEX IF NOT EXISTS one_open_investigation_invocation
                        ON investigation_invocations(investigation_context_id)
                        WHERE module_output_digest IS NULL;
                    """
                )
                session_columns = {
                    row[1]
                    for row in connection.execute(
                        "PRAGMA table_info(investigation_sessions)"
                    ).fetchall()
                }
                if (
                    "capability_generation_digest" not in session_columns
                    and "investigation_sessions" in existing_tables
                ):
                    # schema-v2 早期本地 store 没有独立 generation 列；该列
                    # 只增加可核对的 capability owner 摘要，不重写旧 identity。
                    connection.execute(
                        "ALTER TABLE investigation_sessions "
                        "ADD COLUMN capability_generation_digest TEXT"
                    )
                metadata = connection.execute(
                    "SELECT schema_version FROM investigation_context_metadata "
                    "WHERE singleton_id = 1"
                ).fetchone()
                if metadata is None:
                    if "investigation_context_metadata" in existing_tables:
                        raise InvestigationContextContractError(
                            "Investigation Context Store schema version 记录缺失"
                        )
                    connection.execute(
                        "INSERT INTO investigation_context_metadata "
                        "(singleton_id, schema_version) VALUES (1, ?)",
                        (_CONTEXT_STORE_SCHEMA_VERSION,),
                    )
                elif metadata["schema_version"] != _CONTEXT_STORE_SCHEMA_VERSION:
                    raise InvestigationContextContractError(
                        "Investigation Context Store schema version 不受支持"
                    )
        except sqlite3.Error as exc:
            raise InvestigationContextStoreError(
                "无法初始化 Investigation Context Store"
            ) from exc

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @staticmethod
    def _identity_json(value: object) -> str:
        encoded = asdict(value)
        # 兼容在该字段加入前已经落盘的 SelectedHypothesisRecord Context。
        # None 不是旧身份的一部分，不能因为 expand 字段而制造身份漂移。
        for field_name in (
            "hypothesis_execution_decision_record_id",
            "capability_generation_digest",
        ):
            if encoded.get(field_name) is None:
                encoded.pop(field_name, None)
        return _canonical_json(encoded)

    def _ensure_writable(self) -> None:
        if self._legacy_reader is not None:
            raise InvestigationContextContractError(
                "schema-v1 Investigation Context 只能通过 legacy reader 只读恢复"
            )

    def create_or_load_context(
        self,
        identity: InvestigationContextIdentity,
    ) -> InvestigationContextSnapshot:
        """幂等创建 Context；相同 ID 的内容漂移立即失败。"""

        if not isinstance(identity, InvestigationContextIdentity):
            raise InvestigationContextContractError(
                "identity 必须是 InvestigationContextIdentity"
            )
        self._ensure_writable()
        encoded = self._identity_json(identity)
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT identity_json FROM investigation_contexts "
                    "WHERE investigation_context_id = ?",
                    (identity.investigation_context_id,),
                ).fetchone()
                if row is None:
                    connection.execute(
                        "INSERT INTO investigation_contexts "
                        "(investigation_context_id, identity_json) VALUES (?, ?)",
                        (identity.investigation_context_id, encoded),
                    )
                elif row["identity_json"] != encoded:
                    raise InvestigationContextIdentityConflict(
                        "investigation_context_id 已绑定不同 research_task 或 Hypothesis Batch"
                    )
        except InvestigationContextStoreError:
            raise
        except sqlite3.Error as exc:
            raise InvestigationContextStoreError("无法创建 Investigation Context") from exc
        return self.read_context(identity)

    def bind_session(
        self,
        context: InvestigationContextIdentity,
        session: InvestigationSessionIdentity,
    ) -> InvestigationSessionIdentity:
        """幂等绑定假设 Session，并拒绝跨 Context 或跨 research_task 串用。"""

        if not isinstance(context, InvestigationContextIdentity) or not isinstance(
            session, InvestigationSessionIdentity
        ):
            raise InvestigationContextContractError("context/session 类型非法")
        if (
            session.investigation_context_id != context.investigation_context_id
            or session.research_task_id != context.research_task_id
        ):
            raise InvestigationContextIdentityConflict(
                "Investigation Session 与 Context 身份不一致"
            )
        self._ensure_writable()
        self.create_or_load_context(context)
        encoded = self._identity_json(session)
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                by_id = connection.execute(
                    "SELECT identity_json FROM investigation_sessions "
                    "WHERE investigation_session_id = ?",
                    (session.investigation_session_id,),
                ).fetchone()
                by_hypothesis = connection.execute(
                    "SELECT identity_json FROM investigation_sessions "
                    "WHERE investigation_context_id = ? AND hypothesis_execution_input_record_id = ?",
                    (
                        context.investigation_context_id,
                        session.hypothesis_execution_input_record_id,
                    ),
                ).fetchone()
                for row in (by_id, by_hypothesis):
                    if row is not None and row["identity_json"] != encoded:
                        raise InvestigationContextIdentityConflict(
                            "Investigation Session 身份或假设绑定发生漂移"
                        )
                if by_id is None and by_hypothesis is None:
                    connection.execute(
                        "INSERT INTO investigation_sessions "
                        "(investigation_session_id, investigation_context_id, research_task_id, "
                        "hypothesis_execution_input_record_id, workspace_identity, minichem_scope_id, "
                        "scope_contract_digest, capability_generation_digest, identity_json) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            session.investigation_session_id,
                            session.investigation_context_id,
                            session.research_task_id,
                            session.hypothesis_execution_input_record_id,
                            session.workspace_identity,
                            session.minichem_scope_id,
                            session.scope_contract_digest,
                            session.capability_generation_digest,
                            encoded,
                        ),
                    )
        except InvestigationContextStoreError:
            raise
        except sqlite3.Error as exc:
            raise InvestigationContextStoreError("无法绑定 Investigation Session") from exc
        return session

    def start_invocation(
        self,
        context: InvestigationContextIdentity,
        invocation: InvestigationInvocationIdentity,
    ) -> InvestigationInvocationIdentity:
        """开始唯一 Open Invocation；目标 Session 由输入科学记录确定性推导。"""

        if not isinstance(invocation, InvestigationInvocationIdentity):
            raise InvestigationContextContractError(
                "invocation 必须是 InvestigationInvocationIdentity"
            )
        self._ensure_writable()
        if invocation.investigation_context_id != context.investigation_context_id:
            raise InvestigationContextIdentityConflict("Invocation 与 Context 身份不一致")
        self.create_or_load_context(context)
        encoded = self._identity_json(invocation)
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                existing = connection.execute(
                    "SELECT identity_json FROM investigation_invocations "
                    "WHERE module_invocation_id = ?",
                    (invocation.module_invocation_id,),
                ).fetchone()
                if existing is not None:
                    if existing["identity_json"] != encoded:
                        raise InvestigationContextIdentityConflict(
                            "module_invocation_id 的输入身份发生漂移"
                        )
                    return invocation
                opened = connection.execute(
                    "SELECT module_invocation_id FROM investigation_invocations "
                    "WHERE investigation_context_id = ? AND module_output_digest IS NULL",
                    (context.investigation_context_id,),
                ).fetchone()
                if opened is not None:
                    raise InvestigationContextTransitionConflict(
                        "Investigation Context 已有 Open Module Invocation"
                    )
                sequence = connection.execute(
                    "SELECT COALESCE(MAX(invocation_sequence), 0) + 1 AS next_sequence "
                    "FROM investigation_invocations WHERE investigation_context_id = ?",
                    (context.investigation_context_id,),
                ).fetchone()["next_sequence"]
                connection.execute(
                    "INSERT INTO investigation_invocations "
                    "(module_invocation_id, investigation_context_id, invocation_sequence, "
                    "identity_json, module_output_digest) VALUES (?, ?, ?, ?, NULL)",
                    (
                        invocation.module_invocation_id,
                        invocation.investigation_context_id,
                        sequence,
                        encoded,
                    ),
                )
        except InvestigationContextStoreError:
            raise
        except sqlite3.Error as exc:
            raise InvestigationContextStoreError("无法开始 Investigation Invocation") from exc
        return invocation

    def complete_invocation(
        self,
        context: InvestigationContextIdentity,
        module_invocation_id: str,
        *,
        module_output_digest: str,
    ) -> InvestigationContextSnapshot:
        """闭合当前 Invocation；精确重放成功，输出摘要漂移失败。"""

        invocation_id = _require_id(module_invocation_id, "module_invocation_id")
        self._ensure_writable()
        output_digest = _require_digest(module_output_digest, "module_output_digest")
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT investigation_context_id, module_output_digest "
                    "FROM investigation_invocations WHERE module_invocation_id = ?",
                    (invocation_id,),
                ).fetchone()
                if row is None:
                    raise InvestigationContextTransitionConflict(
                        "要完成的 Investigation Invocation 不存在"
                    )
                if row["investigation_context_id"] != context.investigation_context_id:
                    raise InvestigationContextIdentityConflict(
                        "Invocation 已绑定到另一个 Investigation Context"
                    )
                existing_digest = row["module_output_digest"]
                if existing_digest is not None and existing_digest != output_digest:
                    raise InvestigationContextIdentityConflict(
                        "Invocation module output digest 发生漂移"
                    )
                if existing_digest is None:
                    connection.execute(
                        "UPDATE investigation_invocations SET module_output_digest = ? "
                        "WHERE module_invocation_id = ?",
                        (output_digest, invocation_id),
                    )
        except InvestigationContextStoreError:
            raise
        except sqlite3.Error as exc:
            raise InvestigationContextStoreError("无法完成 Investigation Invocation") from exc
        return self.read_context(context)

    def read_context(
        self,
        identity: InvestigationContextIdentity,
    ) -> InvestigationContextSnapshot:
        """读取完整 Context，并重新核对所有持久身份。"""

        if not isinstance(identity, InvestigationContextIdentity):
            raise InvestigationContextContractError(
                "identity 必须是 InvestigationContextIdentity"
            )
        if self._legacy_reader is not None:
            return self._legacy_reader.read_context(identity)
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT identity_json FROM investigation_contexts "
                    "WHERE investigation_context_id = ?",
                    (identity.investigation_context_id,),
                ).fetchone()
                if row is None:
                    raise InvestigationContextTransitionConflict(
                        "Investigation Context 不存在"
                    )
                if row["identity_json"] != self._identity_json(identity):
                    raise InvestigationContextIdentityConflict(
                        "Investigation Context 身份发生漂移"
                    )
                session_rows = connection.execute(
                    "SELECT identity_json FROM investigation_sessions "
                    "WHERE investigation_context_id = ? ORDER BY rowid ASC",
                    (identity.investigation_context_id,),
                ).fetchall()
                invocation_rows = connection.execute(
                    "SELECT identity_json, module_output_digest FROM investigation_invocations "
                    "WHERE investigation_context_id = ? ORDER BY invocation_sequence ASC",
                    (identity.investigation_context_id,),
                ).fetchall()
        except InvestigationContextStoreError:
            raise
        except sqlite3.Error as exc:
            raise InvestigationContextStoreError("无法读取 Investigation Context") from exc
        try:
            sessions = tuple(
                InvestigationSessionIdentity(**json.loads(row["identity_json"]))
                for row in session_rows
            )
            closed: list[InvestigationInvocationIdentity] = []
            opened: InvestigationInvocationIdentity | None = None
            for row in invocation_rows:
                invocation = InvestigationInvocationIdentity(
                    **json.loads(row["identity_json"])
                )
                if row["module_output_digest"] is None:
                    if opened is not None:
                        raise InvestigationContextStoreError(
                            "Investigation Context 存在多个 Open Invocation"
                        )
                    opened = invocation
                else:
                    _require_digest(
                        row["module_output_digest"],
                        "module_output_digest",
                    )
                    closed.append(invocation)
        except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise InvestigationContextStoreError(
                "Investigation Context 持久身份已损坏"
            ) from exc
        return InvestigationContextSnapshot(
            identity=identity,
            sessions=sessions,
            closed_invocations=tuple(closed),
            open_invocation=opened,
        )

    def read_completed_output_digest(
        self,
        context: InvestigationContextIdentity,
        module_invocation_id: str,
    ) -> str | None:
        """读取指定 Invocation 的完成输出摘要；Open Invocation 返回 ``None``。"""

        invocation_id = _require_id(
            module_invocation_id, "module_invocation_id"
        )
        if self._legacy_reader is not None:
            return self._legacy_reader.read_completed_output_digest(
                context, invocation_id
            )
        self.read_context(context)
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT investigation_context_id, module_output_digest "
                    "FROM investigation_invocations WHERE module_invocation_id = ?",
                    (invocation_id,),
                ).fetchone()
        except sqlite3.Error as exc:
            raise InvestigationContextStoreError(
                "无法读取 Investigation Invocation 输出摘要"
            ) from exc
        if row is None:
            raise InvestigationContextTransitionConflict(
                "Investigation Invocation 不存在"
            )
        if row["investigation_context_id"] != context.investigation_context_id:
            raise InvestigationContextIdentityConflict(
                "Invocation 已绑定到另一个 Investigation Context"
            )
        digest = row["module_output_digest"]
        if digest is None:
            return None
        return _require_digest(digest, "module_output_digest")


__all__ = [
    "InvestigationContextContractError",
    "InvestigationContextIdentity",
    "InvestigationContextIdentityConflict",
    "InvestigationContextSnapshot",
    "InvestigationContextStore",
    "InvestigationContextStoreError",
    "InvestigationContextTransitionConflict",
    "InvestigationInvocationIdentity",
    "InvestigationSessionIdentity",
    "SchemaV1LegacyContextReader",
]
