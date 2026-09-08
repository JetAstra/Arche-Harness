"""Investigation and Experiment 领域状态的 SQLite 持久层。

模块接口提供状态保存、恢复、批次验证重放和假设重生 receipt 的 CAS 保存。完整状态
快照是 Investigation 的唯一事实来源；验证重放表和 receipt 表只保存恢复索引与
跨存储写入进度，模型响应在完整状态成功落盘前不会被本模块视为完成。
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

from .domain import BatchValidation, InvestigationState


class InvestigationPersistenceError(ValueError):
    """调查持久层拒绝输入或无法完成可靠保存。"""


class InvestigationRevisionConflict(InvestigationPersistenceError):
    """调用方使用的 expected revision 已经过期。"""

    def __init__(self, task_id: str, expected_revision: int, actual_revision: int) -> None:
        self.task_id = task_id
        self.expected_revision = expected_revision
        self.actual_revision = actual_revision
        super().__init__(
            f"task_id={task_id} 的调查状态版本冲突："
            f"expected_revision={expected_revision}，当前版本={actual_revision}"
        )


class InvestigationStateCorruption(InvestigationPersistenceError):
    """SQLite 中的快照或验证重放记录损坏，不能安全恢复。"""


@dataclass(frozen=True)
class InvestigationStateSnapshot:
    """恢复得到的完整 InvestigationState 及其 CAS revision。"""

    state: InvestigationState
    revision: int


@dataclass(frozen=True)
class InvestigationRegenerationReceiptSnapshot:
    """恢复一个假设重生 receipt 及其独立 CAS revision。

    receipt 是 bridge 在跨 Working Memory 与 Investigation 两个存储之间保存的
    小型进度记录。它不替代 InvestigationState，只用于判断模型生成、BatchCreated
    写入和 BeginNextBatch 是否已经完成，避免恢复时重复调用外部依赖。
    """

    task_id: str
    operation_key: str
    receipt: Mapping[str, Any]
    revision: int


_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS investigation_states (
        task_id TEXT PRIMARY KEY,
        revision INTEGER NOT NULL CHECK (revision >= 1),
        state_json TEXT NOT NULL,
        state_digest TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS investigation_validation_replays (
        task_id TEXT NOT NULL,
        batch_id TEXT NOT NULL,
        idempotency_key TEXT NOT NULL,
        state_revision INTEGER NOT NULL CHECK (state_revision >= 1),
        result_vector_json TEXT NOT NULL,
        validation_json TEXT NOT NULL,
        PRIMARY KEY (task_id, batch_id, idempotency_key)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_investigation_validation_replays_lookup "
    "ON investigation_validation_replays(task_id, batch_id, idempotency_key)",
    """
    CREATE TABLE IF NOT EXISTS investigation_regeneration_receipts (
        task_id TEXT NOT NULL,
        operation_key TEXT NOT NULL,
        revision INTEGER NOT NULL CHECK (revision >= 1),
        receipt_json TEXT NOT NULL,
        receipt_digest TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (task_id, operation_key)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_investigation_regeneration_receipts_task "
    "ON investigation_regeneration_receipts(task_id, operation_key)",
)


def _canonical_json(value: Any) -> str:
    """把 JSON 值编码为稳定文本；非 JSON 值直接失败。"""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise InvestigationPersistenceError("调查数据不能编码为严格 JSON") from exc


def _restore_json_arrays(value: Any) -> Any:
    """将 JSON 数组恢复为领域模型严格要求的 tuple。"""

    if isinstance(value, list):
        return tuple(_restore_json_arrays(item) for item in value)
    if isinstance(value, dict):
        return {key: _restore_json_arrays(item) for key, item in value.items()}
    return value


def _validate_task_id(task_id: str) -> str:
    """校验按任务读取或保存时使用的非空 task_id。"""

    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError("task_id 不能为空")
    return task_id


def _validate_batch_id(batch_id: str) -> str:
    """校验验证重放使用的非空 batch_id。"""

    if not isinstance(batch_id, str) or not batch_id.strip():
        raise ValueError("batch_id 不能为空")
    return batch_id


def _validate_operation_key(operation_key: str) -> str:
    """校验 receipt 的稳定业务幂等键。"""

    if not isinstance(operation_key, str) or not operation_key.strip():
        raise ValueError("operation_key 不能为空")
    return operation_key


def _validate_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    """复制并确认 receipt 是严格 JSON object。"""

    if not isinstance(value, Mapping):
        raise TypeError("receipt 必须是 JSON object")
    try:
        encoded = _canonical_json(dict(value))
        decoded = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise InvestigationPersistenceError("receipt 不能编码为严格 JSON") from exc
    if not isinstance(decoded, dict):  # pragma: no cover - 由 object 输入保证
        raise InvestigationPersistenceError("receipt 必须编码为 JSON object")
    return decoded


_REGENERATION_RECEIPT_STATUS_ORDER = {
    "started": 0,
    "generated": 1,
    "wm_written": 2,
    "completed": 3,
}


def _validate_receipt_status_transition(
    existing_json: object,
    next_status: object,
) -> None:
    """拒绝重生 receipt 的阶段回退或终局状态改写。"""

    if not isinstance(existing_json, str):
        raise InvestigationStateCorruption("重生 receipt 旧 JSON 类型损坏")
    try:
        existing = json.loads(existing_json)
    except json.JSONDecodeError as exc:
        raise InvestigationStateCorruption("重生 receipt 旧 JSON 无法解析") from exc
    if not isinstance(existing, dict):
        raise InvestigationStateCorruption("重生 receipt 旧内容必须是 object")
    previous_status = existing.get("status")
    if previous_status not in {*_REGENERATION_RECEIPT_STATUS_ORDER, "failed"}:
        raise InvestigationStateCorruption("重生 receipt 旧 status 未注册")
    if next_status not in {*_REGENERATION_RECEIPT_STATUS_ORDER, "failed"}:
        raise InvestigationPersistenceError("重生 receipt.status 未注册")
    if previous_status in {"completed", "failed"}:
        if next_status != previous_status:
            raise InvestigationPersistenceError("重生 receipt.status 不能改写终局阶段")
        return
    if next_status == "failed":
        return
    if _REGENERATION_RECEIPT_STATUS_ORDER[next_status] < _REGENERATION_RECEIPT_STATUS_ORDER[
        previous_status
    ]:
        raise InvestigationPersistenceError("重生 receipt.status 不能回退")


def _validate_expected_revision(expected_revision: int) -> int:
    """校验 CAS 版本是非负整数而不是布尔值。"""

    if isinstance(expected_revision, bool) or not isinstance(expected_revision, int):
        raise TypeError("expected_revision 必须是非负整数")
    if expected_revision < 0:
        raise ValueError("expected_revision 不能为负数")
    return expected_revision


def _validate_state(state: InvestigationState) -> InvestigationState:
    """重新构造并严格校验领域状态，避免未校验的 model_copy 进入持久层。"""

    if not isinstance(state, InvestigationState):
        raise TypeError("state 必须是 InvestigationState")
    return InvestigationState.model_validate(state.model_dump(exclude_none=True))


def _serialize_state(state: InvestigationState) -> tuple[str, str]:
    """严格复核状态并返回快照 JSON 与 SHA-256 摘要。"""

    validated = _validate_state(state)
    payload = _canonical_json(validated.model_dump(mode="json", exclude_none=True))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return payload, digest


def _decode_state(task_id: str, raw_json: object, raw_digest: object) -> InvestigationState:
    """解析、摘要校验并严格恢复数据库快照；任何异常都拒绝恢复。"""

    if not isinstance(raw_json, str) or not isinstance(raw_digest, str):
        raise InvestigationStateCorruption(f"task_id={task_id} 的调查快照列类型损坏")
    try:
        decoded = json.loads(raw_json)
    except (TypeError, json.JSONDecodeError) as exc:
        raise InvestigationStateCorruption(f"task_id={task_id} 的调查快照 JSON 无法解析") from exc
    if not isinstance(decoded, dict):
        raise InvestigationStateCorruption(f"task_id={task_id} 的调查快照必须是 JSON 对象")
    try:
        canonical = _canonical_json(decoded)
    except InvestigationPersistenceError as exc:
        raise InvestigationStateCorruption(f"task_id={task_id} 的调查快照 JSON 不规范") from exc
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if canonical != raw_json or digest != raw_digest:
        raise InvestigationStateCorruption(f"task_id={task_id} 的调查快照摘要不匹配")
    try:
        state = InvestigationState.model_validate(_restore_json_arrays(decoded))
    except Exception as exc:
        raise InvestigationStateCorruption(f"task_id={task_id} 的调查快照不符合领域合同") from exc
    if state.task_id != task_id:
        raise InvestigationStateCorruption(f"task_id={task_id} 的快照内部 task_id 不匹配")
    return state


def _collect_validations(state: InvestigationState) -> tuple[BatchValidation, ...]:
    """收集当前批次和全部历史批次的验证记录。"""

    records = list(state.validations)
    for history in state.history:
        records.extend(history.validations)
    return tuple(records)


def _validation_rows(
    state: InvestigationState,
) -> dict[tuple[str, str, str], tuple[str, str]]:
    """从完整快照生成验证重放表的期望内容。"""

    rows: dict[tuple[str, str, str], tuple[str, str]] = {}
    for validation in _collect_validations(state):
        key = (validation.task_id, validation.batch_id, validation.idempotency_key)
        if key in rows:
            raise InvestigationPersistenceError(f"重复的批次验证幂等键: {key}")
        vector = sorted(zip(validation.combination_ids, validation.result_versions))
        rows[key] = (
            _canonical_json(vector),
            _canonical_json(validation.model_dump(mode="json", exclude_none=True)),
        )
    return rows


def _validation_idempotency_key(
    task_id: str,
    batch_id: str,
    vector: tuple[tuple[str, str], ...],
) -> str:
    """复现 BatchValidation 的完整配对向量幂等键算法。"""

    payload = {
        "task_id": task_id,
        "batch_id": batch_id,
        "result_versions": sorted(vector),
    }
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _normalize_vector(
    vector: Mapping[str, str] | tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    """把一个明确的一项或五项向量规范化为组合与版本配对元组。"""

    if isinstance(vector, Mapping):
        pairs = tuple(vector.items())
    elif isinstance(vector, tuple):
        pairs = vector
    else:
        raise TypeError("result_version_vector 必须是映射或一项/五项 tuple 配对")
    if len(pairs) not in {1, 5}:
        raise ValueError("result_version_vector 必须包含一项或五项配对")
    for pair in pairs:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise TypeError("result_version_vector 必须由 (combination_id, result_version) 组成")
        combination_id, result_version = pair
        if (
            not isinstance(combination_id, str)
            or not combination_id.strip()
            or not isinstance(result_version, str)
            or not result_version.strip()
        ):
            raise ValueError("result_version_vector 中的组合和版本必须是非空字符串")
    if len({pair[0] for pair in pairs}) != len(pairs):
        raise ValueError("result_version_vector 的 combination_id 必须唯一")
    return pairs


def _read_replay_rows(
    connection: sqlite3.Connection,
    task_id: str,
    state: InvestigationState,
    revision: int,
) -> dict[tuple[str, str, str], tuple[str, str]]:
    """读取验证重放表并与完整快照逐项核对。"""

    expected = _validation_rows(state)
    rows = connection.execute(
        "SELECT task_id, batch_id, idempotency_key, state_revision, "
        "result_vector_json, validation_json "
        "FROM investigation_validation_replays WHERE task_id = ?",
        (task_id,),
    ).fetchall()
    actual: dict[tuple[str, str, str], tuple[str, str]] = {}
    for row in rows:
        key = (row["task_id"], row["batch_id"], row["idempotency_key"])
        if row["state_revision"] != revision or key in actual:
            raise InvestigationStateCorruption(f"task_id={task_id} 的验证重放索引版本损坏")
        raw_vector = row["result_vector_json"]
        raw_validation = row["validation_json"]
        if not isinstance(raw_vector, str) or not isinstance(raw_validation, str):
            raise InvestigationStateCorruption(f"task_id={task_id} 的验证重放 JSON 类型损坏")
        try:
            vector = json.loads(raw_vector)
            validation_payload = _restore_json_arrays(json.loads(raw_validation))
            validation = BatchValidation.model_validate(validation_payload)
            vector_json = _canonical_json(vector)
            validation_json = _canonical_json(json.loads(raw_validation))
        except Exception as exc:
            raise InvestigationStateCorruption(f"task_id={task_id} 的验证重放 JSON 损坏") from exc
        if not isinstance(vector, list) or len(vector) not in {1, 5}:
            raise InvestigationStateCorruption(f"task_id={task_id} 的验证版本向量不完整")
        if len(vector) != len(validation.combination_ids):
            raise InvestigationStateCorruption(f"task_id={task_id} 的验证版本向量长度漂移")
        expected_vector = _canonical_json(
            sorted(zip(validation.combination_ids, validation.result_versions))
        )
        if (
            validation.task_id != task_id
            or validation.batch_id != row["batch_id"]
            or validation.idempotency_key != row["idempotency_key"]
            or vector_json != expected_vector
        ):
            raise InvestigationStateCorruption(f"task_id={task_id} 的验证重放索引与正文不一致")
        actual[key] = (vector_json, validation_json)
    if actual != expected:
        raise InvestigationStateCorruption(f"task_id={task_id} 的验证重放记录与快照不一致")
    return actual


class InvestigationSQLiteStore:
    """在共享 SQLite 文件中原子保存和恢复 InvestigationState。

    快照表保存全部会话、结果、验证和历史；验证重放表只是可核对索引。建表采用
    ``IF NOT EXISTS``，不会删除已有模块的表；保存使用 ``BEGIN IMMEDIATE``，让
    revision 比较、快照更新和索引更新处在同一个 SQLite 写锁内。
    """

    def __init__(self, database_path: str | os.PathLike[str]) -> None:
        raw_path = os.fspath(database_path)
        if raw_path == ":memory:":
            raise ValueError("InvestigationSQLiteStore 需要可跨连接重启的 SQLite 文件")
        self._database_path = Path(raw_path).expanduser().resolve()
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def initialize(self) -> None:
        """在已有 SQLite 文件中增量创建调查快照和验证重放表。

        输入是构造函数提供的数据库路径；动作只执行 ``CREATE IF NOT EXISTS``，不
        删除其他表。成功返回 ``None``，SQLite 锁或建表错误直接抛出。
        """

        with self._write_transaction() as connection:
            for statement in _SCHEMA_STATEMENTS:
                connection.execute(statement)

    def save_investigation_state(
        self,
        state: InvestigationState,
        expected_revision: int = 0,
    ) -> int:
        """按 expected revision CAS 保存完整状态并返回保存后的 revision。

        输入是严格 ``InvestigationState`` 和上次读取的版本；动作把完整 JSON 快照
        与全部验证重放索引写入同一事务。完全相同的快照即使 expected revision 过期
        也原样幂等返回当前版本；不同内容遇到版本不一致抛出
        ``InvestigationRevisionConflict``。事务提交前发生任何异常都会回滚，调用方
        不能将未落盘的模型响应标为完成。
        """

        expected_revision = _validate_expected_revision(expected_revision)
        validated_state = _validate_state(state)
        state_json, state_digest = _serialize_state(validated_state)
        task_id = _validate_task_id(validated_state.task_id)
        validation_rows = _validation_rows(validated_state)

        with self._write_transaction() as connection:
            row = connection.execute(
                "SELECT revision, state_json, state_digest FROM investigation_states "
                "WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if row is None:
                if expected_revision != 0:
                    raise InvestigationRevisionConflict(task_id, expected_revision, 0)
                revision = 1
                connection.execute(
                    "INSERT INTO investigation_states "
                    "(task_id, revision, state_json, state_digest, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        task_id,
                        revision,
                        state_json,
                        state_digest,
                        _utc_now(),
                        _utc_now(),
                    ),
                )
                self._replace_validation_replays(
                    connection,
                    validated_state,
                    revision,
                    validation_rows,
                )
                return revision

            actual_revision = row["revision"]
            if (
                isinstance(actual_revision, bool)
                or not isinstance(actual_revision, int)
                or actual_revision < 1
            ):
                raise InvestigationStateCorruption(f"task_id={task_id} 的 revision 损坏")
            existing_state = _decode_state(task_id, row["state_json"], row["state_digest"])
            _read_replay_rows(connection, task_id, existing_state, actual_revision)
            existing_json, existing_digest = _serialize_state(existing_state)
            if existing_json == state_json and existing_digest == state_digest:
                return actual_revision
            if expected_revision != actual_revision:
                raise InvestigationRevisionConflict(task_id, expected_revision, actual_revision)

            revision = actual_revision + 1
            update = connection.execute(
                "UPDATE investigation_states SET revision = ?, state_json = ?, "
                "state_digest = ?, updated_at = ? "
                "WHERE task_id = ? AND revision = ?",
                (revision, state_json, state_digest, _utc_now(), task_id, actual_revision),
            )
            if update.rowcount != 1:
                raise InvestigationRevisionConflict(task_id, expected_revision, actual_revision)
            self._replace_validation_replays(
                connection,
                validated_state,
                revision,
                validation_rows,
            )
            return revision

    def _replace_validation_replays(
        self,
        connection: sqlite3.Connection,
        state: InvestigationState,
        revision: int,
        rows: Mapping[tuple[str, str, str], tuple[str, str]],
    ) -> None:
        """在当前事务中替换任务的验证重放索引。"""

        connection.execute(
            "DELETE FROM investigation_validation_replays WHERE task_id = ?",
            (state.task_id,),
        )
        for (task_id, batch_id, idempotency_key), (vector_json, validation_json) in rows.items():
            connection.execute(
                "INSERT INTO investigation_validation_replays "
                "(task_id, batch_id, idempotency_key, state_revision, "
                "result_vector_json, validation_json) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    task_id,
                    batch_id,
                    idempotency_key,
                    revision,
                    vector_json,
                    validation_json,
                ),
            )

    def load_investigation_snapshot(
        self,
        task_id: str,
    ) -> InvestigationStateSnapshot | None:
        """按 task_id 原子恢复完整状态和 revision。

        输入是非空任务标识；动作解析状态摘要、严格恢复 Pydantic 模型并核对验证重放
        表。成功返回快照，不存在返回 ``None``；任何 JSON、revision、索引或领域合同
        损坏都抛出 ``InvestigationStateCorruption``，不返回部分状态。
        """

        task_id = _validate_task_id(task_id)
        connection = self._connect_database()
        try:
            connection.execute("BEGIN")
            row = connection.execute(
                "SELECT revision, state_json, state_digest FROM investigation_states "
                "WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            revision = row["revision"]
            if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
                raise InvestigationStateCorruption(f"task_id={task_id} 的 revision 损坏")
            state = _decode_state(task_id, row["state_json"], row["state_digest"])
            _read_replay_rows(connection, task_id, state, revision)
            connection.commit()
            return InvestigationStateSnapshot(state=state, revision=revision)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def read_saved_batch_validation(
        self,
        task_id: str,
        batch_id: str,
        result_version_vector: Mapping[str, str] | tuple[tuple[str, str], ...],
    ) -> BatchValidation | None:
        """按完整当前批次组合与 result_version 配对恢复已落盘批次验证。

        输入只有一个明确的完整向量（一项或五项映射/tuple 配对）；动作计算稳定幂等
        键并从已核对快照中读取验证，不调用模型。返回原 ``BatchValidation``，找不到
        返回 ``None``；向量不完整、不同配对或持久数据损坏都会失败或不重放。
        """

        task_id = _validate_task_id(task_id)
        batch_id = _validate_batch_id(batch_id)
        vector = _normalize_vector(result_version_vector)
        key = _validation_idempotency_key(task_id, batch_id, vector)
        snapshot = self.load_investigation_snapshot(task_id)
        if snapshot is None:
            return None
        for validation in _collect_validations(snapshot.state):
            if validation.batch_id == batch_id and validation.idempotency_key == key:
                return validation
        return None

    def load_hypothesis_regeneration_receipt(
        self,
        task_id: str,
        operation_key: str,
    ) -> InvestigationRegenerationReceiptSnapshot | None:
        """恢复假设重生 receipt，不返回未核对或部分 JSON。

        输入是任务身份和 bridge 使用的稳定 operation_key；动作只读取 receipt 表，
        核对 revision、规范 JSON 和 SHA-256 摘要。不存在时返回 ``None``，发现数据
        损坏时抛出 ``InvestigationStateCorruption``，不会用默认状态掩盖恢复问题。
        """

        task_id = _validate_task_id(task_id)
        operation_key = _validate_operation_key(operation_key)
        connection = self._connect_database()
        try:
            connection.execute("BEGIN")
            row = connection.execute(
                "SELECT revision, receipt_json, receipt_digest "
                "FROM investigation_regeneration_receipts "
                "WHERE task_id=? AND operation_key=?",
                (task_id, operation_key),
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            revision = row["revision"]
            if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
                raise InvestigationStateCorruption(
                    f"task_id={task_id} operation_key={operation_key} 的 receipt revision 损坏"
                )
            raw_json = row["receipt_json"]
            raw_digest = row["receipt_digest"]
            if not isinstance(raw_json, str) or not isinstance(raw_digest, str):
                raise InvestigationStateCorruption(
                    f"task_id={task_id} operation_key={operation_key} 的 receipt 列类型损坏"
                )
            try:
                decoded = json.loads(raw_json)
            except (TypeError, json.JSONDecodeError) as exc:
                raise InvestigationStateCorruption(
                    f"task_id={task_id} operation_key={operation_key} 的 receipt JSON 无法解析"
                ) from exc
            if not isinstance(decoded, dict):
                raise InvestigationStateCorruption(
                    f"task_id={task_id} operation_key={operation_key} 的 receipt 必须是 object"
                )
            canonical = _canonical_json(decoded)
            digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            if canonical != raw_json or digest != raw_digest:
                raise InvestigationStateCorruption(
                    f"task_id={task_id} operation_key={operation_key} 的 receipt 摘要不匹配"
                )
            connection.commit()
            return InvestigationRegenerationReceiptSnapshot(
                task_id=task_id,
                operation_key=operation_key,
                receipt=decoded,
                revision=revision,
            )
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def save_hypothesis_regeneration_receipt(
        self,
        task_id: str,
        operation_key: str,
        receipt: Mapping[str, Any],
        expected_revision: int = 0,
    ) -> int:
        """按独立 revision CAS 保存一个假设重生 receipt。

        输入是任务身份、稳定 operation_key、严格 JSON receipt 和上次读取版本；动作
        只更新 receipt 表，不修改 InvestigationState。相同 receipt 在过期版本下会
        幂等返回当前版本，不同内容遇到 CAS 冲突抛出明确异常，事务失败会回滚。
        """

        task_id = _validate_task_id(task_id)
        operation_key = _validate_operation_key(operation_key)
        expected_revision = _validate_expected_revision(expected_revision)
        normalized = _validate_receipt(receipt)
        receipt_json = _canonical_json(normalized)
        receipt_digest = hashlib.sha256(receipt_json.encode("utf-8")).hexdigest()

        with self._write_transaction() as connection:
            row = connection.execute(
                "SELECT revision, receipt_json, receipt_digest "
                "FROM investigation_regeneration_receipts "
                "WHERE task_id=? AND operation_key=?",
                (task_id, operation_key),
            ).fetchone()
            if row is None:
                if expected_revision != 0:
                    raise InvestigationRevisionConflict(task_id, expected_revision, 0)
                connection.execute(
                    "INSERT INTO investigation_regeneration_receipts "
                    "(task_id, operation_key, revision, receipt_json, receipt_digest, "
                    "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        task_id,
                        operation_key,
                        1,
                        receipt_json,
                        receipt_digest,
                        _utc_now(),
                        _utc_now(),
                    ),
                )
                return 1

            actual_revision = row["revision"]
            if (
                isinstance(actual_revision, bool)
                or not isinstance(actual_revision, int)
                or actual_revision < 1
            ):
                raise InvestigationStateCorruption(
                    f"task_id={task_id} operation_key={operation_key} 的 receipt revision 损坏"
                )
            _validate_receipt_status_transition(row["receipt_json"], normalized.get("status"))
            if row["receipt_json"] == receipt_json and row["receipt_digest"] == receipt_digest:
                return actual_revision
            if expected_revision != actual_revision:
                raise InvestigationRevisionConflict(task_id, expected_revision, actual_revision)
            revision = actual_revision + 1
            updated = connection.execute(
                "UPDATE investigation_regeneration_receipts SET revision=?, "
                "receipt_json=?, receipt_digest=?, updated_at=? "
                "WHERE task_id=? AND operation_key=? AND revision=?",
                (
                    revision,
                    receipt_json,
                    receipt_digest,
                    _utc_now(),
                    task_id,
                    operation_key,
                    actual_revision,
                ),
            )
            if updated.rowcount != 1:
                raise InvestigationRevisionConflict(task_id, expected_revision, actual_revision)
            return revision

    def claim_hypothesis_regeneration_receipt(
        self,
        task_id: str,
        operation_key: str,
        receipt: Mapping[str, Any],
    ) -> InvestigationRegenerationReceiptSnapshot | None:
        """原子创建重生 receipt，只有首次调用者获得生成权。

        输入是任务身份、稳定 operation_key 和状态为 ``started`` 的完整 receipt；动作
        在一个 SQLite 写事务中执行 insert-if-absent。不存在时返回 revision=1 的新
        snapshot；已存在时只返回 ``None``，绝不把相同 JSON 的幂等更新误当成 owner。
        已存在的内容由调用方随后通过 ``load_hypothesis_regeneration_receipt`` 核对。
        """

        task_id = _validate_task_id(task_id)
        operation_key = _validate_operation_key(operation_key)
        normalized = _validate_receipt(receipt)
        if normalized.get("status") != "started":
            raise InvestigationPersistenceError(
                "重生 receipt claim 只能创建 started 阶段"
            )
        receipt_json = _canonical_json(normalized)
        receipt_digest = hashlib.sha256(receipt_json.encode("utf-8")).hexdigest()

        with self._write_transaction() as connection:
            row = connection.execute(
                "SELECT revision FROM investigation_regeneration_receipts "
                "WHERE task_id=? AND operation_key=?",
                (task_id, operation_key),
            ).fetchone()
            if row is not None:
                return None
            connection.execute(
                "INSERT INTO investigation_regeneration_receipts "
                "(task_id, operation_key, revision, receipt_json, receipt_digest, "
                "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    task_id,
                    operation_key,
                    1,
                    receipt_json,
                    receipt_digest,
                    _utc_now(),
                    _utc_now(),
                ),
            )
            return InvestigationRegenerationReceiptSnapshot(
                task_id=task_id,
                operation_key=operation_key,
                receipt=normalized,
                revision=1,
            )

    def _connect_database(self) -> sqlite3.Connection:
        """打开内部 SQLite 连接并启用外键与忙等待。"""

        connection = sqlite3.connect(
            str(self._database_path),
            timeout=30,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    @contextmanager
    def _write_transaction(self) -> Iterator[sqlite3.Connection]:
        """取得 SQLite 写锁，异常时回滚快照和验证索引。"""

        connection = self._connect_database()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()


def _utc_now() -> str:
    """生成 UTC 更新时间。"""

    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "InvestigationPersistenceError",
    "InvestigationRegenerationReceiptSnapshot",
    "InvestigationRevisionConflict",
    "InvestigationSQLiteStore",
    "InvestigationStateCorruption",
    "InvestigationStateSnapshot",
]
