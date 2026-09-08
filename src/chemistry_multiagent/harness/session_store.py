"""持久化 Planner/Codex 会话元数据并提供每会话锁。

实际对话记录由 Codex CLI 持有；ARCHE 只保存小型 manifest，使重启后的 controller
能够找到同一会话并将其绑定到正确的 Planner 任务和 workspace。manifest 刻意不保存
凭据或 prompt 正文。
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

from .base import HarnessError


_MANIFEST_NAME = ".arche_planner_session.json"
_LOCK_NAME = ".arche_planner_session.lock"
_PLANNER_RECOVERY_NAME = ".arche_planner_recovery.json"
_PLANNER_RECOVERY_POISON_NAME = ".arche_planner_recovery_poison.json"
_PLANNER_VALIDATED_TURN_NAME = ".arche_planner_validated_turn.json"
_PLANNER_RECOVERY_SCHEMA_VERSION = 1
_PLANNER_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_PLANNER_RECOVERY_ACTIONS = {
    "worker_cleanup": "worker_stop_confirmation_required",
    "completion_receipt": "receipt_replay_required",
    "request_intent": "request_intent_confirmation_required",
    "terminal_turn": "terminal_acknowledgement_required",
    "response_contract": "explicit_new_request_required",
}
_PLANNER_RECOVERY_FIELDS = frozenset(
    {
        "schema_version",
        "state",
        "reason",
        "action",
        "stage",
        "request_digest",
        "thread_policy_digest",
        "turn_id",
        "turn_count",
        "created_at",
        "updated_at",
    }
)
_PLANNER_VALIDATED_FIELDS = frozenset(
    {
        "schema_version",
        "thread_id",
        "stage",
        "request_digest",
        "thread_policy_digest",
        "turn_id",
        "turn_count",
        "created_at",
        "updated_at",
    }
)
_SESSION_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,256}$")
_REQUIRED_FIELDS = {
    "schema_version",
    "task_scope_id",
    "planner_scope_id",
    "session_id",
    "workspace",
    "model",
    "last_stage",
    "turn_count",
    "status",
    "created_at",
    "updated_at",
}
_SESSION_PROCESS_LOCKS: dict[str, threading.RLock] = {}
_SESSION_LOCKS_GUARD = threading.Lock()
_SESSION_LOCK_HANDLES: dict[tuple[str, int], tuple[Any, int, Any]] = {}


def _session_process_lock(path: Path) -> threading.RLock:
    """为同一进程的会话线程提供可重入锁，同时保留跨进程文件锁。"""

    key = str(path)
    with _SESSION_LOCKS_GUARD:
        lock = _SESSION_PROCESS_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _SESSION_PROCESS_LOCKS[key] = lock
        return lock


@contextmanager
def _lock_session_file(path: Path, *, session_label: str) -> Iterator[None]:
    """锁住一个会话 turn，支持同一线程嵌套并拒绝无锁继续。

    输入是会话锁文件路径和用于错误信息的会话类型；动作是创建并收紧锁文件、
    获取进程内可重入锁及跨进程 ``fcntl`` 独占锁，并在退出时按相反顺序释放它们。
    返回值是无值上下文；锁文件打开、加锁或释放失败时抛出 ``HarnessError``，
    调用方不会进入未受保护的 turn。
    """

    process_lock = _session_process_lock(path)
    process_lock.acquire()
    state_key = (str(path), threading.get_ident())
    with _SESSION_LOCKS_GUARD:
        nested_state = _SESSION_LOCK_HANDLES.get(state_key)
        if nested_state is not None:
            handle, depth, fcntl = nested_state
            _SESSION_LOCK_HANDLES[state_key] = (handle, depth + 1, fcntl)
            nested = True
        else:
            nested = False

    if nested:
        try:
            yield
        finally:
            with _SESSION_LOCKS_GUARD:
                handle, depth, fcntl = _SESSION_LOCK_HANDLES[state_key]
                if depth <= 1:
                    _SESSION_LOCK_HANDLES.pop(state_key, None)
                else:
                    _SESSION_LOCK_HANDLES[state_key] = (handle, depth - 1, fcntl)
            process_lock.release()
        return

    handle: Any = None
    fcntl: Any = None
    try:
        handle = path.open("a+", encoding="utf-8")
        os.chmod(path, 0o600)
        fcntl = _lock_file(handle, path)
        with _SESSION_LOCKS_GUARD:
            _SESSION_LOCK_HANDLES[state_key] = (handle, 1, fcntl)
        try:
            yield
        finally:
            with _SESSION_LOCKS_GUARD:
                _SESSION_LOCK_HANDLES.pop(state_key, None)
            _unlock_file(handle, path, fcntl)
    except OSError as exc:
        raise HarnessError(
            f"{session_label} session lock cannot be opened: {path}",
            kind="session_state",
        ) from exc
    finally:
        if handle is not None:
            handle.close()
        process_lock.release()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_fcntl() -> Any:
    """加载 Linux 文件锁模块；固定生产环境缺失时必须停止而不是无锁继续。"""

    try:
        import fcntl
    except ImportError as exc:  # pragma: no cover - 生产 Linux 始终提供
        raise HarnessError(
            "Codex session locking requires the Linux fcntl module",
            kind="session_state",
        ) from exc
    return fcntl


def _lock_file(handle: Any, path: Path) -> Any:
    """获取独占文件锁；系统拒绝加锁时不允许进入模型 turn。"""

    fcntl = _require_fcntl()
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    except OSError as exc:
        raise HarnessError(
            f"Codex session lock cannot be acquired: {path}",
            kind="session_state",
        ) from exc
    return fcntl


def _unlock_file(handle: Any, path: Path, fcntl: Any) -> None:
    """释放独占文件锁；释放失败同样保留状态错误。"""

    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError as exc:
        raise HarnessError(
            f"Codex session lock cannot be released: {path}",
            kind="session_state",
        ) from exc


def _fsync_directory(path: Path, *, description: str) -> None:
    """刷盘目录项；目录打开或刷盘任一失败都返回持久化错误。"""

    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_DIRECTORY)
        os.fsync(descriptor)
    except (AttributeError, OSError) as exc:
        raise HarnessError(
            f"{description} directory cannot be durably synced: {path}",
            kind="session_state",
        ) from exc
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                # 若 open/fsync 已经失败，保留原始错误；否则关闭失败也必须失败。
                if descriptor >= 0:
                    raise HarnessError(
                        f"{description} directory descriptor cannot be closed: {path}",
                        kind="session_state",
                    )


def _atomic_write_json_file(
    path: Path,
    value: Mapping[str, Any],
    *,
    description: str,
    mode: int = 0o600,
) -> Path:
    """以临时文件、文件刷盘、原子替换和目录刷盘保存 JSON。"""

    target = Path(path)
    if target.is_symlink():
        raise HarnessError(f"{description} must not be a symlink", kind="session_state")
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    descriptor = -1
    temporary = ""
    try:
        descriptor, temporary = tempfile.mkstemp(
            prefix=f"{target.name}.",
            dir=str(target.parent),
            text=True,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        if target.is_symlink():
            raise HarnessError(f"{description} must not be a symlink", kind="session_state")
        os.replace(temporary, target)
        temporary = ""
        os.chmod(target, mode)
        _fsync_directory(target.parent, description=description)
    except HarnessError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise HarnessError(f"{description} cannot be saved", kind="session_state") from exc
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if temporary:
            try:
                os.unlink(temporary)
            except OSError:
                pass
    return target


def _remove_file_and_fsync_directory(path: Path, *, description: str) -> None:
    """幂等删除状态文件并刷盘父目录，符号链接和刷盘失败均停止流程。"""

    target = Path(path)
    if target.is_symlink():
        raise HarnessError(f"{description} must not be a symlink", kind="session_state")
    if not target.exists():
        return
    try:
        target.unlink()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise HarnessError(f"{description} cannot be cleared", kind="session_state") from exc
    _fsync_directory(target.parent, description=description)


def _validate_session_id(value: Any) -> str:
    if not isinstance(value, str) or _SESSION_ID.fullmatch(value.strip()) is None:
        raise HarnessError("Codex session_id is missing or malformed", kind="session_state")
    return value.strip()


def _validate_scope(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HarnessError(f"Planner session manifest {field} is missing", kind="session_state")
    return value.strip()


def _validate_planner_recovery_header(value: Mapping[str, Any]) -> None:
    """校验 Planner 恢复记录的字段集合、版本和状态动作。"""

    if set(value) != _PLANNER_RECOVERY_FIELDS:
        raise HarnessError(
            "Planner recovery fields are invalid; request and policy digests are required",
            kind="session_state",
        )
    if value.get("schema_version") != _PLANNER_RECOVERY_SCHEMA_VERSION:
        raise HarnessError("Planner recovery schema is invalid", kind="session_state")
    state = value.get("state")
    reason = value.get("reason")
    action = value.get("action")
    if (
        state != "recovery_pending"
        or not isinstance(reason, str)
        or action != _PLANNER_RECOVERY_ACTIONS.get(reason)
    ):
        raise HarnessError("Planner recovery state or action is invalid", kind="session_state")


def _validate_planner_recovery_timestamps_and_digests(
    value: Mapping[str, Any],
) -> None:
    """校验 Planner 恢复记录的时间字段和两个请求摘要。"""

    for field in ("stage", "created_at", "updated_at"):
        _validate_scope(value.get(field), f"recovery.{field}")
    for field in ("request_digest", "thread_policy_digest"):
        digest = value.get(field)
        if not isinstance(digest, str) or _PLANNER_DIGEST.fullmatch(digest) is None:
            raise HarnessError(f"Planner recovery {field} is malformed", kind="session_state")


def _validate_planner_recovery_turn_fields(value: Mapping[str, Any]) -> None:
    """校验 Planner 恢复记录的 turn 标识和非负计数。"""

    turn_id = value.get("turn_id")
    if not isinstance(turn_id, str):
        raise HarnessError("Planner recovery turn_id is malformed", kind="session_state")
    if turn_id and _SESSION_ID.fullmatch(turn_id) is None:
        raise HarnessError("Planner recovery turn_id is malformed", kind="session_state")
    turn_count = value.get("turn_count")
    if isinstance(turn_count, bool) or not isinstance(turn_count, int) or turn_count < 0:
        raise HarnessError("Planner recovery turn_count is invalid", kind="session_state")


class PlannerSessionStore:
    """在隔离 workspace 中持久化一个 Planner 会话 manifest。"""

    schema_version = 1

    def __init__(self, workspace: Path, *, task_scope_id: str, planner_scope_id: str) -> None:
        self.workspace = Path(workspace).resolve()
        self.task_scope_id = _validate_scope(task_scope_id, "task_scope_id")
        self.planner_scope_id = _validate_scope(planner_scope_id, "planner_scope_id")
        self._expected_thread_policy_digest: str | None = None
        self.workspace.mkdir(parents=True, exist_ok=True)

    @property
    def manifest_path(self) -> Path:
        return self.workspace / _MANIFEST_NAME

    @property
    def lock_path(self) -> Path:
        return self.workspace / _LOCK_NAME

    @property
    def recovery_path(self) -> Path:
        """返回 legacy Planner 的恢复记录路径。"""

        return self.workspace / _PLANNER_RECOVERY_NAME

    @property
    def recovery_poison_path(self) -> Path:
        """返回恢复双写失败后的持久阻断标记路径。"""

        return self.workspace / _PLANNER_RECOVERY_POISON_NAME

    @property
    def validated_turn_path(self) -> Path:
        """返回 Planner 已验证完成 turn 的独立凭证路径。"""

        return self.workspace / _PLANNER_VALIDATED_TURN_NAME

    @contextmanager
    def locked(self) -> Iterator[None]:
        """串行化首轮和恢复 turn，包括其中运行的 CLI 子进程。"""

        self.workspace.mkdir(parents=True, exist_ok=True)
        with _lock_session_file(self.lock_path, session_label="Planner"):
            yield

    def _validate_planner_manifest_identity(self, value: Mapping[str, Any]) -> None:
        """核对 Planner manifest 的会话、任务和规划器范围。"""

        _validate_session_id(value.get("session_id"))
        if _validate_scope(value.get("task_scope_id"), "task_scope_id") != self.task_scope_id:
            raise HarnessError("Planner session task_scope_id does not match request", kind="session_state")
        if _validate_scope(value.get("planner_scope_id"), "planner_scope_id") != self.planner_scope_id:
            raise HarnessError("Planner session planner_scope_id does not match request", kind="session_state")

    def _validate_planner_manifest_policy(self, value: Mapping[str, Any]) -> None:
        """核对 Planner manifest 的有效 thread policy 摘要绑定。"""

        stored_policy_digest = value.get("thread_policy_digest")
        if stored_policy_digest is not None and (
            not isinstance(stored_policy_digest, str)
            or _PLANNER_DIGEST.fullmatch(stored_policy_digest) is None
        ):
            raise HarnessError(
                "Planner session thread_policy_digest is malformed",
                kind="session_state",
            )
        if self._expected_thread_policy_digest is None:
            return
        if stored_policy_digest is None:
            raise HarnessError(
                "Planner session manifest lacks effective thread policy digest",
                kind="session_state",
            )
        if stored_policy_digest != self._expected_thread_policy_digest:
            raise HarnessError(
                "Planner session effective thread policy does not match current config",
                kind="session_state",
            )

    def _validate_planner_manifest_state(self, value: Mapping[str, Any]) -> None:
        """核对 Planner manifest 的 schema、计数、状态和策略摘要。"""

        schema_version = value.get("schema_version")
        if isinstance(schema_version, bool) or not isinstance(schema_version, int):
            raise HarnessError("Unsupported Planner session manifest schema", kind="session_state")
        if schema_version != self.schema_version:
            raise HarnessError("Unsupported Planner session manifest schema", kind="session_state")
        turn_count = value.get("turn_count")
        if isinstance(turn_count, bool) or not isinstance(turn_count, int) or turn_count < 0:
            raise HarnessError("Planner session turn_count is invalid", kind="session_state")
        if not isinstance(value.get("status"), str) or not value["status"].strip():
            raise HarnessError("Planner session status is invalid", kind="session_state")
        self._validate_planner_manifest_policy(value)

    def load(self) -> dict[str, Any] | None:
        """读取并校验 Planner manifest；损坏或身份漂移时停止恢复。"""

        if not self.manifest_path.exists():
            return None
        try:
            value = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HarnessError(
                f"Planner session manifest cannot be read: {self.manifest_path}",
                kind="session_state",
            ) from exc
        if not isinstance(value, dict):
            raise HarnessError("Planner session manifest must be a JSON object", kind="session_state")
        missing = _REQUIRED_FIELDS.difference(value)
        if missing:
            raise HarnessError(
                f"Planner session manifest is missing fields: {sorted(missing)}",
                kind="session_state",
            )
        self._validate_planner_manifest_identity(value)
        self._validate_planner_manifest_state(value)
        return dict(value)

    def save(self, manifest: Mapping[str, Any]) -> dict[str, Any]:
        value = dict(manifest)
        value["schema_version"] = self.schema_version
        value["task_scope_id"] = self.task_scope_id
        value["planner_scope_id"] = self.planner_scope_id
        value["session_id"] = _validate_session_id(value.get("session_id"))
        value["workspace"] = str(self.workspace)
        value["updated_at"] = _utc_now()
        stored_policy_digest = value.get("thread_policy_digest")
        if self._expected_thread_policy_digest is not None:
            if stored_policy_digest is None:
                value["thread_policy_digest"] = self._expected_thread_policy_digest
            elif stored_policy_digest != self._expected_thread_policy_digest:
                raise HarnessError(
                    "Planner session effective thread policy does not match current config",
                    kind="session_state",
                )
        elif stored_policy_digest is not None and (
            not isinstance(stored_policy_digest, str)
            or _PLANNER_DIGEST.fullmatch(stored_policy_digest) is None
        ):
            raise HarnessError(
                "Planner session thread_policy_digest is malformed",
                kind="session_state",
            )
        if not isinstance(value.get("turn_count"), int) or value["turn_count"] < 0:
            raise HarnessError("Planner session turn_count is invalid", kind="session_state")
        _atomic_write_json_file(
            self.manifest_path,
            value,
            description=f"Planner session manifest {self.manifest_path}",
        )
        return value

    def _read_planner_json(self, path: Path, *, description: str) -> dict[str, Any] | None:
        """读取 Planner 状态 JSON；损坏、链接和非普通文件均停止恢复。"""

        if path.is_symlink():
            raise HarnessError(f"{description} must not be a symlink", kind="session_state")
        if not path.exists():
            return None
        if not path.is_file():
            raise HarnessError(f"{description} is not a regular file", kind="session_state")
        try:
            os.chmod(path, 0o600)
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HarnessError(f"{description} cannot be read", kind="session_state") from exc
        if not isinstance(value, dict):
            raise HarnessError(f"{description} must be a JSON object", kind="session_state")
        return dict(value)

    def _validate_planner_recovery(self, value: Mapping[str, Any]) -> dict[str, Any]:
        """校验 legacy Planner 恢复记录中的摘要、动作和 turn 计数。"""

        _validate_planner_recovery_header(value)
        _validate_planner_recovery_timestamps_and_digests(value)
        _validate_planner_recovery_turn_fields(value)
        return dict(value)

    def _read_planner_recovery(self) -> dict[str, Any] | None:
        """读取 legacy Planner 恢复记录。"""

        value = self._read_planner_json(self.recovery_path, description="Planner recovery record")
        return None if value is None else self._validate_planner_recovery(value)

    def load_recovery(self) -> dict[str, Any] | None:
        """读取 Planner 恢复记录，不启动新模型 turn。"""

        return self._read_planner_recovery()

    def load_recovery_poison(self) -> dict[str, Any] | None:
        """读取恢复双写失败标记；存在时调用方必须等待人工恢复。"""

        return self._read_planner_json(
            self.recovery_poison_path,
            description="Planner recovery poison",
        )

    def _planner_recovery_identity(
        self,
        *,
        stage: str,
        request_digest: str,
        thread_policy_digest: str,
        turn_id: str | None,
        reason: str,
        turn_count: int,
    ) -> dict[str, Any]:
        """构造不含 prompt 和 secret 的 Planner 恢复身份。"""

        (
            validated_stage,
            validated_request_digest,
            validated_thread_policy_digest,
            validated_turn_id,
            validated_reason,
            validated_turn_count,
        ) = _validate_planner_recovery_identity_arguments(
            stage=stage,
            request_digest=request_digest,
            thread_policy_digest=thread_policy_digest,
            turn_id=turn_id,
            reason=reason,
            turn_count=turn_count,
        )
        now = _utc_now()
        return {
            "schema_version": _PLANNER_RECOVERY_SCHEMA_VERSION,
            "state": "recovery_pending",
            "reason": validated_reason,
            "action": _PLANNER_RECOVERY_ACTIONS[validated_reason],
            "stage": validated_stage,
            "request_digest": validated_request_digest,
            "thread_policy_digest": validated_thread_policy_digest,
            "turn_id": validated_turn_id,
            "turn_count": validated_turn_count,
            "created_at": now,
            "updated_at": now,
        }

    def _persist_planner_recovery_identity_record(
        self,
        existing: Mapping[str, Any] | None,
        expected: dict[str, Any],
    ) -> dict[str, Any]:
        """复用同身份 Planner 恢复记录或原子保存新记录。"""

        if existing is None:
            _atomic_write_json_file(
                self.recovery_path,
                expected,
                description="Planner recovery record",
            )
            return expected
        if any(
            existing.get(key) != expected[key]
            for key in expected
            if key not in {"created_at", "updated_at"}
        ):
            raise HarnessError(
                "Planner recovery record conflicts with the current identity",
                kind="session_state",
            )
        expected["created_at"] = existing["created_at"]
        return expected

    def record_recovery_pending(
        self,
        *,
        stage: str,
        request_digest: str,
        thread_policy_digest: str,
        turn_id: str | None,
        reason: str = "worker_cleanup",
        turn_count_before: int | None = None,
    ) -> dict[str, Any]:
        """保存 worker、receipt 或 request-intent 恢复状态并阻止新 turn。"""

        with self.locked():
            current = self.load()
            if current is None:
                raise HarnessError(
                    "Cannot record Planner recovery before creating a session",
                    kind="session_state",
                )
            existing = self._read_planner_recovery()
            expected = self._planner_recovery_identity(
                stage=stage,
                request_digest=request_digest,
                thread_policy_digest=thread_policy_digest,
                turn_id=turn_id,
                reason=reason,
                turn_count=(
                    current["turn_count"]
                    if turn_count_before is None
                    else turn_count_before
                ),
            )
            self._persist_planner_recovery_identity_record(existing, expected)
            current["last_stage"] = str(stage).strip()
            current["status"] = "recovery_pending"
            return self.save(current)

    def record_recovery_poison(
        self,
        *,
        stage: str,
        request_digest: str,
        thread_policy_digest: str,
        turn_id: str | None,
        reason: str,
        turn_count_before: int | None = None,
    ) -> dict[str, Any]:
        """记录恢复二次持久化失败，跨重启阻止任何新模型 turn。"""

        with self.locked():
            current = self.load()
            if current is None:
                raise HarnessError(
                    "Cannot record Planner recovery poison before creating a session",
                    kind="session_state",
                )
            recovery = self._planner_recovery_identity(
                stage=stage,
                request_digest=request_digest,
                thread_policy_digest=thread_policy_digest,
                turn_id=turn_id,
                reason=reason,
                turn_count=(
                    current["turn_count"]
                    if turn_count_before is None
                    else turn_count_before
                ),
            )
            recovery["state"] = "poisoned"
            recovery["action"] = "manual_recovery_required"
            existing = self.load_recovery_poison()
            if existing is not None:
                if existing != recovery:
                    raise HarnessError(
                        "Planner recovery poison conflicts with current identity",
                        kind="session_state",
                    )
                return existing
            _atomic_write_json_file(
                self.recovery_poison_path,
                recovery,
                description="Planner recovery poison",
            )
            return recovery

    def record_validated_turn(
        self,
        *,
        stage: str,
        request_digest: str,
        thread_policy_digest: str,
        turn_id: str,
    ) -> dict[str, Any]:
        """保存 SDK 已返回且 receipt 身份匹配的 turn 凭证。"""

        with self.locked():
            current = self.load()
            if current is None:
                raise HarnessError(
                    "Cannot record Planner validated turn before creating a session",
                    kind="session_state",
                )
            value = self._build_planner_validated_turn_record(
                current,
                stage=stage,
                request_digest=request_digest,
                thread_policy_digest=thread_policy_digest,
                turn_id=turn_id,
            )
            existing = self._read_planner_json(
                self.validated_turn_path,
                description="Planner validated turn",
            )
            if existing is not None:
                if existing != value:
                    raise HarnessError(
                        "Planner validated turn conflicts with current turn",
                        kind="session_state",
                    )
                return existing
            _atomic_write_json_file(
                self.validated_turn_path,
                value,
                description="Planner validated turn",
            )
            return value

    def _build_planner_validated_turn_record(
        self,
        current: Mapping[str, Any],
        *,
        stage: str,
        request_digest: str,
        thread_policy_digest: str,
        turn_id: str,
    ) -> dict[str, Any]:
        """校验 Planner 凭证身份并构造当前 turn 的持久记录。"""

        if not isinstance(request_digest, str) or _PLANNER_DIGEST.fullmatch(request_digest) is None:
            raise HarnessError("Planner validated request digest is malformed", kind="session_state")
        if not isinstance(thread_policy_digest, str) or _PLANNER_DIGEST.fullmatch(thread_policy_digest) is None:
            raise HarnessError("Planner validated policy digest is malformed", kind="session_state")
        return {
            "schema_version": _PLANNER_RECOVERY_SCHEMA_VERSION,
            "thread_id": _validate_session_id(current["session_id"]),
            "stage": str(stage).strip(),
            "request_digest": request_digest,
            "thread_policy_digest": thread_policy_digest,
            "turn_id": _validate_session_id(turn_id),
            "turn_count": current["turn_count"],
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
        }

    def _validate_planner_recovery_clear_evidence(
        self,
        current: Mapping[str, Any],
        recovery: Mapping[str, Any],
        validated: Mapping[str, Any],
        *,
        stage: str,
        thread_policy_digest: str,
    ) -> None:
        """核对 Planner validated 凭证足以替代旧恢复记录。"""

        if not (
            validated.get("thread_id") == current["session_id"]
            and validated.get("stage") == str(stage).strip()
            and validated.get("thread_policy_digest") == thread_policy_digest
            and validated.get("turn_count") == current["turn_count"]
            and validated.get("turn_count", -1) > recovery["turn_count"]
        ):
            raise HarnessError(
                "Planner validated turn evidence does not match recovery",
                kind="session_state",
            )

    def clear_recovery_after_validated_turn(
        self,
        *,
        stage: str,
        thread_policy_digest: str,
        status: str = "active",
    ) -> dict[str, Any]:
        """只在独立 validated 凭证存在时清理旧恢复记录。"""

        with self.locked():
            current = self.load()
            recovery = self._read_planner_recovery()
            if current is None or recovery is None:
                return current or {}
            validated = self._read_planner_json(
                self.validated_turn_path,
                description="Planner validated turn",
            )
            if validated is None:
                raise HarnessError(
                    "Planner recovery cannot be cleared without validated turn evidence",
                    kind="session_state",
                )
            self._validate_planner_recovery_clear_evidence(
                current,
                recovery,
                validated,
                stage=stage,
                thread_policy_digest=thread_policy_digest,
            )
            current["status"] = str(status).strip()
            saved = self.save(current)
            _remove_file_and_fsync_directory(
                self.recovery_path,
                description="Planner recovery record",
            )
            _remove_file_and_fsync_directory(
                self.validated_turn_path,
                description="Planner validated turn",
            )
            return saved

    def reconcile_recovery_state(
        self,
        *,
        thread_policy_digest: str | None = None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """重启时核对 Planner manifest、recovery 和 validated 凭证，不猜测 turn 结果。

        输入是可选的当前策略摘要；动作是在锁内读取三类持久状态，并调用针对
        manifest 是否存在恢复记录的业务分支。返回最新的 ``(manifest, recovery)``；
        poison、损坏文件或无法确认 turn 结果时抛出状态错误，不启动新的 turn。
        """

        expected_policy = self._resolve_planner_policy_digest(thread_policy_digest)
        with self.locked():
            current, poison, recovery, validated = self._read_planner_recovery_snapshot()
            self._validate_planner_recovery_policy_binding(
                expected_policy,
                poison=poison,
                recovery=recovery,
                validated=validated,
            )
            if poison is not None:
                raise HarnessError(
                    "Planner recovery poison marker blocks recovery",
                    kind="recovery_pending",
                )
            if current is None:
                return self._reconcile_planner_without_manifest(recovery, validated)
            if recovery is None:
                return self._reconcile_planner_without_recovery(current, validated)
            return self._reconcile_planner_with_recovery(current, recovery, validated)

    def _resolve_planner_policy_digest(self, thread_policy_digest: str | None) -> str | None:
        """校验并绑定本次 Planner 恢复使用的策略摘要。"""

        if thread_policy_digest is None:
            return self._expected_thread_policy_digest
        if not isinstance(thread_policy_digest, str) or _PLANNER_DIGEST.fullmatch(thread_policy_digest) is None:
            raise HarnessError(
                "Planner session thread_policy_digest is malformed",
                kind="session_state",
            )
        if (
            self._expected_thread_policy_digest is not None
            and self._expected_thread_policy_digest != thread_policy_digest
        ):
            raise HarnessError(
                "Planner session effective thread policy does not match current config",
                kind="session_state",
            )
        self._expected_thread_policy_digest = thread_policy_digest
        return thread_policy_digest

    def _read_planner_recovery_snapshot(
        self,
    ) -> tuple[
        dict[str, Any] | None,
        dict[str, Any] | None,
        dict[str, Any] | None,
        dict[str, Any] | None,
    ]:
        """在锁内读取 manifest、poison、recovery 和 validated 四份状态。"""

        current = self.load()
        poison = self.load_recovery_poison()
        recovery = self._read_planner_recovery()
        validated = self._read_planner_json(
            self.validated_turn_path,
            description="Planner validated turn",
        )
        return current, poison, recovery, validated

    def _validate_planner_recovery_policy_binding(
        self,
        expected_policy: str | None,
        *,
        poison: Mapping[str, Any] | None,
        recovery: Mapping[str, Any] | None,
        validated: Mapping[str, Any] | None,
    ) -> None:
        """确认三类恢复凭证都绑定当前 effective thread policy。"""

        if expected_policy is None:
            return
        for name, record in (
            ("recovery poison", poison),
            ("recovery", recovery),
            ("validated turn", validated),
        ):
            if record is not None and record.get("thread_policy_digest") != expected_policy:
                raise HarnessError(
                    f"Planner {name} effective thread policy does not match current config",
                    kind="session_state",
                )

    def _reconcile_planner_without_manifest(
        self,
        recovery: Mapping[str, Any] | None,
        validated: Mapping[str, Any] | None,
    ) -> tuple[None, None]:
        """拒绝没有 manifest 却残留恢复或 validated 凭证的状态。"""

        if recovery is not None or validated is not None:
            raise HarnessError(
                "Planner recovery state exists without a session manifest",
                kind="session_state",
            )
        return None, None

    def _reconcile_planner_without_recovery(
        self,
        current: dict[str, Any],
        validated: Mapping[str, Any] | None,
    ) -> tuple[dict[str, Any], None]:
        """恢复记录已经不存在时清除不能证明新 turn 的残留凭证。"""

        if validated is not None:
            _remove_file_and_fsync_directory(
                self.validated_turn_path,
                description="Planner validated turn",
            )
        return current, None

    def _reconcile_planner_with_recovery(
        self,
        current: dict[str, Any],
        recovery: Mapping[str, Any],
        validated: Mapping[str, Any] | None,
    ) -> tuple[dict[str, Any], Mapping[str, Any] | None]:
        """按独立凭证清理已完成 turn，或恢复 manifest 的阻断状态。"""

        if (
            validated is not None
            and current["turn_count"] > recovery["turn_count"]
            and validated.get("thread_id") == current["session_id"]
            and validated.get("turn_count") == current["turn_count"]
            and validated.get("stage") == current["last_stage"]
        ):
            _remove_file_and_fsync_directory(
                self.recovery_path,
                description="Planner recovery record",
            )
            _remove_file_and_fsync_directory(
                self.validated_turn_path,
                description="Planner validated turn",
            )
            return current, None
        expected_status = (
            "contract_failure" if recovery["reason"] == "response_contract" else "recovery_pending"
        )
        if current.get("status") != expected_status:
            current["last_stage"] = recovery["stage"]
            current["status"] = expected_status
            current = self.save(current)
        return current, recovery

    def new_manifest(
        self,
        *,
        session_id: str,
        model: str,
        stage: str,
        turn_count: int = 1,
        status: str = "active",
        thread_policy_digest: str | None = None,
    ) -> dict[str, Any]:
        """构造尚未持久化的 Planner manifest。

        默认值保持旧 Planner 的成功计数兼容；首次 thread 注册由 runner 显式
        使用 ``turn_count=0`` 和 ``status="created"``，因为此时真实 turn 尚未提交。
        """

        if not isinstance(turn_count, int) or turn_count < 0:
            raise HarnessError("Planner session turn_count is invalid", kind="session_state")
        if not isinstance(status, str) or not status.strip():
            raise HarnessError("Planner session status is invalid", kind="session_state")
        if thread_policy_digest is not None:
            if not isinstance(thread_policy_digest, str) or _PLANNER_DIGEST.fullmatch(thread_policy_digest) is None:
                raise HarnessError(
                    "Planner session thread_policy_digest is malformed",
                    kind="session_state",
                )
            if self._expected_thread_policy_digest is not None and self._expected_thread_policy_digest != thread_policy_digest:
                raise HarnessError(
                    "Planner session effective thread policy does not match current config",
                    kind="session_state",
                )
            self._expected_thread_policy_digest = thread_policy_digest
        now = _utc_now()
        value = {
            "schema_version": self.schema_version,
            "task_scope_id": self.task_scope_id,
            "planner_scope_id": self.planner_scope_id,
            "session_id": _validate_session_id(session_id),
            "workspace": str(self.workspace),
            "model": str(model),
            "last_stage": str(stage),
            "turn_count": turn_count,
            "status": status.strip(),
            "created_at": now,
            "updated_at": now,
        }
        if self._expected_thread_policy_digest is not None:
            value["thread_policy_digest"] = self._expected_thread_policy_digest
        return value

    def create_thread_manifest(
        self,
        *,
        session_id: str,
        model: str,
        stage: str,
        thread_policy_digest: str | None = None,
    ) -> dict[str, Any]:
        """原子登记已创建但尚未提交 turn 的 thread，并返回已保存 manifest。

        调用方应在持有 ``locked()`` 的情况下使用本方法；保存采用临时文件、刷盘
        和原子替换，保证首轮 SDK 失败后下一次调用仍能恢复同一 thread。
        """

        return self.save(
            self.new_manifest(
                session_id=session_id,
                model=model,
                stage=stage,
                turn_count=0,
                status="created",
                thread_policy_digest=thread_policy_digest,
            )
        )

    def record_turn(
        self,
        manifest: Mapping[str, Any],
        *,
        stage: str,
        status: str = "active",
        session_id: str | None = None,
    ) -> dict[str, Any]:
        value = dict(manifest)
        if session_id is not None:
            value["session_id"] = session_id
        value["last_stage"] = str(stage)
        value["status"] = str(status)
        value["turn_count"] = int(value.get("turn_count", 0)) + 1
        return self.save(value)


# 新会话使用独立 manifest，避免旧 Planner 主链误读组合级或任务级记录。
_CODEX_MANIFEST_NAME = ".arche_codex_session.json"
_CODEX_LOCK_NAME = ".arche_codex_session.lock"
_CODEX_RECOVERY_NAME = ".arche_codex_recovery.json"
_CODEX_RECOVERY_POISON_NAME = ".arche_codex_recovery_poison.json"
_CODEX_VALIDATED_TURN_NAME = ".arche_codex_validated_turn.json"
_CODEX_THREAD_ROLLBACK_INTENT_NAME = ".arche_codex_thread_rollback_intent.json"
_CODEX_THREAD_ROLLBACK_RECEIPT_NAME = ".arche_codex_thread_rollback_receipt.json"
_LEGACY_CODEX_SESSION_SCHEMA_VERSION = 1
_CODEX_SESSION_SCHEMA_VERSION = 2
_CODEX_THREAD_ROLLBACK_SCHEMA_VERSION = 1
_CODEX_THREAD_ROLLBACK_ERROR_CODES = frozenset({"invalid_encrypted_content"})
_CODEX_THREAD_ROLLBACK_STAGES = frozenset(
    {"initial_investigation", "investigation_job_wake"}
)
_CODEX_THREAD_ROLLBACK_FIELDS = frozenset(
    {
        "schema_version",
        "state",
        "stage",
        "thread_id",
        "request_digest",
        "thread_policy_digest",
        "turn_count",
        "physical_turn_count",
        "provider_error_code",
        "created_at",
        "updated_at",
    }
)
_CODEX_REQUIRED_FIELDS = {
    "schema_version",
    "owner",
    "workspace",
    "scope_id",
    "model",
    "codex_home",
    "thread_id",
    "last_stage",
    "turn_count",
    "status",
    "created_at",
    "updated_at",
}
_CODEX_OPTIONAL_FIELDS = {"thread_policy_digest"}
_OWNER_KINDS = {"investigation", "task_review"}
_RECOVERY_RECORD_SCHEMA_VERSION = 2
_RECOVERY_RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "state",
        "reason",
        "action",
        "stage",
        "request_digest",
        "thread_policy_digest",
        "turn_id",
        "turn_count",
        "created_at",
        "updated_at",
    }
)
_RECOVERY_STATES = frozenset({"contract_failure", "recovery_pending"})
_RECOVERY_REASONS = frozenset(
    {
        "response_contract",
        "worker_cleanup",
        "completion_receipt",
        "request_intent",
        "terminal_turn",
        "goal",
    }
)
_CODEX_RECOVERY_PENDING_ACTIONS = {
    "worker_cleanup": "worker_stop_confirmation_required",
    "completion_receipt": "receipt_replay_required",
    "request_intent": "request_intent_confirmation_required",
    "terminal_turn": "terminal_acknowledgement_required",
    "goal": "goal_recovery_required",
}
_RECOVERY_ACTIONS = frozenset(
    {
        "explicit_new_request_required",
        "worker_stop_confirmation_required",
        "receipt_replay_required",
        "request_intent_confirmation_required",
        "terminal_acknowledgement_required",
        "goal_recovery_required",
    }
)
_RECOVERY_ALLOWED_ACTIONS = {
    "response_contract": frozenset({"explicit_new_request_required"}),
    "worker_cleanup": frozenset(
        {"worker_stop_confirmation_required", "explicit_new_request_required"}
    ),
    "completion_receipt": frozenset({"receipt_replay_required"}),
    "request_intent": frozenset(
        {"request_intent_confirmation_required", "explicit_new_request_required"}
    ),
    "terminal_turn": frozenset({"terminal_acknowledgement_required"}),
    "goal": frozenset({"goal_recovery_required"}),
}
_RECOVERY_POISON_FIELDS = frozenset(
    {
        "schema_version",
        "state",
        "reason",
        "action",
        "stage",
        "request_digest",
        "thread_policy_digest",
        "turn_id",
        "turn_count",
        "created_at",
        "updated_at",
    }
)
_VALIDATED_TURN_FIELDS = frozenset(
    {
        "schema_version",
        "thread_id",
        "stage",
        "request_digest",
        "thread_policy_digest",
        "turn_id",
        "turn_count",
        "created_at",
        "updated_at",
    }
)
_REQUEST_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_THREAD_POLICY_DIGEST = re.compile(r"^[0-9a-f]{64}$")


def _validate_identity_component(value: Any, field: str) -> str:
    """校验会话归属和目录字段，拒绝空值、控制字符及路径分隔符。"""

    if not isinstance(value, str):
        raise HarnessError(f"Codex session {field} must be a non-empty string", kind="session_state")
    normalized = value.strip()
    if _contains_forbidden_identity_characters(normalized):
        raise HarnessError(f"Codex session {field} is malformed", kind="session_state")
    return normalized


def _contains_forbidden_identity_characters(value: str) -> bool:
    """判断会话身份文本是否包含空值、路径分隔符或控制字符。"""

    return (
        not value
        or value in {".", ".."}
        or "\x00" in value
        or "/" in value
        or "\\" in value
        or any(ord(char) < 32 for char in value)
    )


def _validate_thread_id(value: Any) -> str:
    """校验 Codex thread 标识，拒绝空值和不安全字符。"""

    if not isinstance(value, str) or _SESSION_ID.fullmatch(value.strip()) is None:
        raise HarnessError("Codex thread_id is missing or malformed", kind="session_state")
    return value.strip()


def _validate_text(value: Any, field: str) -> str:
    """校验 manifest 中的普通文本字段。"""

    if not isinstance(value, str):
        raise HarnessError(f"Codex session {field} is missing", kind="session_state")
    normalized = value.strip()
    if not normalized or "\x00" in normalized or any(ord(char) < 32 for char in normalized):
        raise HarnessError(f"Codex session {field} is malformed", kind="session_state")
    return normalized


def _validate_thread_policy_digest(value: Any, field: str = "thread_policy_digest") -> str:
    """校验不含 secret 的 effective thread policy 摘要。"""

    if not isinstance(value, str) or _THREAD_POLICY_DIGEST.fullmatch(value) is None:
        raise HarnessError(f"Codex {field} is malformed", kind="session_state")
    return value


def _validate_thread_rollback_record(
    record: Mapping[str, Any],
    *,
    expected_state: str,
) -> dict[str, Any]:
    """校验 Investigation turn rollback 的 intent 或 receipt，拒绝未知字段。"""

    if not isinstance(record, Mapping) or set(record) != _CODEX_THREAD_ROLLBACK_FIELDS:
        raise HarnessError(
            "Codex thread rollback record fields are invalid",
            kind="session_state",
        )
    if record.get("schema_version") != _CODEX_THREAD_ROLLBACK_SCHEMA_VERSION:
        raise HarnessError(
            "Codex thread rollback record schema is invalid",
            kind="session_state",
        )
    state = _validate_text(record.get("state"), "thread_rollback.state")
    if state != expected_state:
        raise HarnessError(
            "Codex thread rollback record state is invalid",
            kind="session_state",
        )
    stage = _validate_text(record.get("stage"), "thread_rollback.stage")
    if stage not in _CODEX_THREAD_ROLLBACK_STAGES:
        raise HarnessError(
            "Codex thread rollback is only valid for eligible Investigation turns",
            kind="session_state",
        )
    thread_id = _validate_thread_id(record.get("thread_id"))
    request_digest = record.get("request_digest")
    if not isinstance(request_digest, str) or _REQUEST_DIGEST.fullmatch(request_digest) is None:
        raise HarnessError(
            "Codex thread rollback request digest is malformed",
            kind="session_state",
        )
    thread_policy_digest = _validate_thread_policy_digest(
        record.get("thread_policy_digest"),
        "thread_rollback.thread_policy_digest",
    )
    turn_count = record.get("turn_count")
    physical_turn_count = record.get("physical_turn_count")
    if isinstance(turn_count, bool) or not isinstance(turn_count, int) or turn_count != 0:
        raise HarnessError(
            "Codex thread rollback successful turn count must be zero",
            kind="session_state",
        )
    if (
        isinstance(physical_turn_count, bool)
        or not isinstance(physical_turn_count, int)
        or physical_turn_count < 1
    ):
        raise HarnessError(
            "Codex thread rollback physical turn count is invalid",
            kind="session_state",
        )
    provider_error_code = record.get("provider_error_code")
    if provider_error_code not in _CODEX_THREAD_ROLLBACK_ERROR_CODES:
        raise HarnessError(
            "Codex thread rollback provider error code is not approved",
            kind="session_state",
        )
    created_at = _validate_text(record.get("created_at"), "thread_rollback.created_at")
    updated_at = _validate_text(record.get("updated_at"), "thread_rollback.updated_at")
    return {
        "schema_version": _CODEX_THREAD_ROLLBACK_SCHEMA_VERSION,
        "state": state,
        "stage": stage,
        "thread_id": thread_id,
        "request_digest": request_digest,
        "thread_policy_digest": thread_policy_digest,
        "turn_count": 0,
        "physical_turn_count": physical_turn_count,
        "provider_error_code": provider_error_code,
        "created_at": created_at,
        "updated_at": updated_at,
    }


def _validate_thread_rollback_arguments(
    *,
    stage: Any,
    thread_id: Any,
    request_digest: Any,
    thread_policy_digest: Any,
    turn_count: Any,
    physical_turn_count: Any,
    provider_error_code: Any,
) -> tuple[str, str, str, str, int, int, str]:
    """校验新建 Investigation turn rollback intent 的身份和固定错误代码。"""

    validated_stage = _validate_text(stage, "thread_rollback.stage")
    if validated_stage not in _CODEX_THREAD_ROLLBACK_STAGES:
        raise HarnessError(
            "Codex thread rollback is only valid for eligible Investigation turns",
            kind="session_state",
        )
    validated_thread_id = _validate_thread_id(thread_id)
    if not isinstance(request_digest, str) or _REQUEST_DIGEST.fullmatch(request_digest) is None:
        raise HarnessError(
            "Codex thread rollback request digest is malformed",
            kind="session_state",
        )
    validated_policy = _validate_thread_policy_digest(
        thread_policy_digest,
        "thread_rollback.thread_policy_digest",
    )
    if isinstance(turn_count, bool) or not isinstance(turn_count, int) or turn_count != 0:
        raise HarnessError(
            "Codex thread rollback successful turn count must be zero",
            kind="session_state",
        )
    if (
        isinstance(physical_turn_count, bool)
        or not isinstance(physical_turn_count, int)
        or physical_turn_count < 1
    ):
        raise HarnessError(
            "Codex thread rollback physical turn count is invalid",
            kind="session_state",
        )
    if provider_error_code not in _CODEX_THREAD_ROLLBACK_ERROR_CODES:
        raise HarnessError(
            "Codex thread rollback provider error code is not approved",
            kind="session_state",
        )
    return (
        validated_stage,
        validated_thread_id,
        request_digest,
        validated_policy,
        0,
        physical_turn_count,
        provider_error_code,
    )


def _validate_codex_recovery_pending_arguments(
    stage: Any,
    request_digest: Any,
    thread_policy_digest: Any,
    turn_id: Any,
    reason: Any,
) -> tuple[str, str, str, str, str]:
    """校验待恢复请求参数并返回规范化身份与对应动作。

    输入是 SDK 阶段、请求摘要、thread policy 摘要、可为空 turn 标识和恢复原因；
    动作是拒绝非法类型/摘要并映射允许原因；返回规范化字段及持久化 action，
    失败时抛出 ``HarnessError``，调用方不得写入恢复记录。
    """

    validated_stage = _validate_text(stage, "last_stage")
    if not isinstance(request_digest, str) or _REQUEST_DIGEST.fullmatch(request_digest) is None:
        raise HarnessError("Codex recovery request digest is malformed", kind="session_state")
    validated_thread_policy_digest = _validate_thread_policy_digest(thread_policy_digest)
    if not isinstance(reason, str) or reason not in _CODEX_RECOVERY_PENDING_ACTIONS:
        raise HarnessError("Codex recovery reason is unsupported", kind="session_state")
    validated_turn_id = "" if turn_id is None else _validate_thread_id(turn_id)
    return (
        validated_stage,
        request_digest,
        validated_thread_policy_digest,
        validated_turn_id,
        _CODEX_RECOVERY_PENDING_ACTIONS[reason],
    )


def _classify_codex_recovery_pending_update(
    existing: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    stage: str,
    request_digest: str,
    thread_policy_digest: str,
    turn_id: str,
    reason: str,
    action: str,
) -> str:
    """判定同一恢复身份应幂等保留、升级为终态或拒绝冲突。"""

    expected_identity = {
        "state": "recovery_pending",
        "stage": stage,
        "request_digest": request_digest,
        "thread_policy_digest": thread_policy_digest,
        "turn_id": turn_id,
        "turn_count": current["turn_count"],
    }
    actual_identity = {field: existing.get(field) for field in expected_identity}
    if actual_identity != expected_identity:
        raise HarnessError(
            "Codex recovery record conflicts with the current recovery state",
            kind="session_state",
        )
    existing_class = (existing.get("reason"), existing.get("action"))
    requested_class = (reason, action)
    if existing_class == requested_class or (
        existing.get("reason") == "terminal_turn" and reason == "worker_cleanup"
    ):
        return "keep"
    if existing.get("reason") == "worker_cleanup" and reason == "terminal_turn":
        return "upgrade_terminal"
    raise HarnessError(
        "Codex recovery record conflicts with the current recovery state",
        kind="session_state",
    )


def _validate_codex_validated_turn_arguments(
    stage: Any,
    request_digest: Any,
    thread_policy_digest: Any,
    turn_id: Any,
) -> tuple[str, str, str, str]:
    """校验已验证 turn 的阶段、摘要和 turn 标识。"""

    validated_stage = _validate_text(stage, "last_stage")
    if not isinstance(request_digest, str) or _REQUEST_DIGEST.fullmatch(request_digest) is None:
        raise HarnessError("Codex validated turn request digest is malformed", kind="session_state")
    validated_thread_policy_digest = _validate_thread_policy_digest(thread_policy_digest)
    validated_turn_id = _validate_thread_id(turn_id)
    return validated_stage, request_digest, validated_thread_policy_digest, validated_turn_id


def _validate_codex_recovery_poison_arguments(
    stage: Any,
    request_digest: Any,
    thread_policy_digest: Any,
    turn_id: Any,
    reason: Any,
) -> tuple[str, str, str, str, str]:
    """校验 Codex poison 身份参数并返回规范字段。"""

    validated_stage = _validate_text(stage, "last_stage")
    if not isinstance(request_digest, str) or _REQUEST_DIGEST.fullmatch(request_digest) is None:
        raise HarnessError("Codex recovery poison request digest is malformed", kind="session_state")
    validated_thread_policy_digest = _validate_thread_policy_digest(thread_policy_digest)
    if not isinstance(reason, str) or reason not in _RECOVERY_REASONS:
        raise HarnessError("Codex recovery poison reason is unsupported", kind="session_state")
    validated_turn_id = "" if turn_id is None else _validate_thread_id(turn_id)
    return validated_stage, request_digest, validated_thread_policy_digest, validated_turn_id, reason


def _validate_codex_contract_failure_arguments(
    stage: Any,
    request_digest: Any,
    thread_policy_digest: Any,
    turn_id: Any,
) -> tuple[str, str, str, str]:
    """校验合同失败记录的阶段、请求摘要、策略摘要和 turn 标识。"""

    validated_stage = _validate_text(stage, "last_stage")
    if not isinstance(request_digest, str) or _REQUEST_DIGEST.fullmatch(request_digest) is None:
        raise HarnessError(
            "Codex contract failure request digest is malformed",
            kind="session_state",
        )
    validated_thread_policy_digest = _validate_thread_policy_digest(thread_policy_digest)
    validated_turn_id = _validate_thread_id(turn_id)
    return validated_stage, request_digest, validated_thread_policy_digest, validated_turn_id


def _validate_planner_recovery_identity_arguments(
    *,
    stage: Any,
    request_digest: Any,
    thread_policy_digest: Any,
    turn_id: Any,
    reason: Any,
    turn_count: Any,
) -> tuple[str, str, str, str, str, int]:
    """校验 Planner 恢复身份参数并返回可持久化的规范字段。"""

    if not isinstance(reason, str) or reason not in _PLANNER_RECOVERY_ACTIONS:
        raise HarnessError("Planner recovery reason is unsupported", kind="session_state")
    if not isinstance(request_digest, str) or _PLANNER_DIGEST.fullmatch(request_digest) is None:
        raise HarnessError("Planner recovery request digest is malformed", kind="session_state")
    if not isinstance(thread_policy_digest, str) or _PLANNER_DIGEST.fullmatch(thread_policy_digest) is None:
        raise HarnessError(
            "Planner recovery thread policy digest is malformed",
            kind="session_state",
        )
    if isinstance(turn_count, bool) or not isinstance(turn_count, int) or turn_count < 0:
        raise HarnessError("Planner recovery turn_count is invalid", kind="session_state")
    validated_turn_id = "" if turn_id is None else _validate_session_id(turn_id)
    return (
        str(stage).strip(),
        request_digest,
        thread_policy_digest,
        validated_turn_id,
        reason,
        turn_count,
    )


def _validate_codex_recovery_digest_identity(
    record: Mapping[str, Any],
    *,
    field_prefix: str,
) -> tuple[str, str, str]:
    """校验恢复记录的阶段、请求摘要和 thread policy 摘要。"""

    stage = _validate_text(record.get("stage"), f"{field_prefix}.stage")
    request_digest = record.get("request_digest")
    if not isinstance(request_digest, str) or _REQUEST_DIGEST.fullmatch(request_digest) is None:
        raise HarnessError(
            f"Codex {field_prefix} request digest is malformed",
            kind="session_state",
        )
    thread_policy_digest = _validate_thread_policy_digest(
        record.get("thread_policy_digest"),
        f"{field_prefix}.thread_policy_digest",
    )
    return stage, request_digest, thread_policy_digest


def _validate_codex_recovery_turn_metadata(
    record: Mapping[str, Any],
    *,
    field_prefix: str,
) -> tuple[str, int, str, str]:
    """校验恢复记录的 turn 标识、计数和两个时间字段。"""

    raw_turn_id = record.get("turn_id")
    if not isinstance(raw_turn_id, str):
        raise HarnessError(f"Codex {field_prefix} turn_id is malformed", kind="session_state")
    turn_id = raw_turn_id.strip()
    if turn_id and _SESSION_ID.fullmatch(turn_id) is None:
        raise HarnessError(f"Codex {field_prefix} turn_id is malformed", kind="session_state")
    turn_count = record.get("turn_count")
    if isinstance(turn_count, bool) or not isinstance(turn_count, int) or turn_count < 0:
        raise HarnessError(f"Codex {field_prefix} turn_count is invalid", kind="session_state")
    created_at = _validate_text(record.get("created_at"), f"{field_prefix}.created_at")
    updated_at = _validate_text(record.get("updated_at"), f"{field_prefix}.updated_at")
    return turn_id, turn_count, created_at, updated_at


def _validate_codex_recovery_record_header(record: Mapping[str, Any]) -> None:
    """校验 Codex 恢复记录的 JSON 类型、字段集合和 schema。"""

    if not isinstance(record, Mapping):
        raise HarnessError("Codex recovery record must be a JSON object", kind="session_state")
    if set(record) != _RECOVERY_RECORD_FIELDS:
        raise HarnessError(
            "Codex recovery record fields are invalid; effective thread policy digest is required",
            kind="session_state",
        )
    schema_version = record.get("schema_version")
    if isinstance(schema_version, bool) or schema_version != _RECOVERY_RECORD_SCHEMA_VERSION:
        raise HarnessError("Codex recovery record schema is invalid", kind="session_state")


def _validate_codex_recovery_state_reason_action(
    record: Mapping[str, Any],
) -> tuple[str, str, str]:
    """校验 Codex 恢复记录的状态、原因和允许动作关系。"""

    state = _validate_text(record.get("state"), "recovery.state")
    if state not in _RECOVERY_STATES:
        raise HarnessError("Codex recovery state is unsupported", kind="session_state")
    reason = _validate_text(record.get("reason"), "recovery.reason")
    if reason not in _RECOVERY_REASONS:
        raise HarnessError("Codex recovery reason is unsupported", kind="session_state")
    action = _validate_text(record.get("action"), "recovery.action")
    if action not in _RECOVERY_ACTIONS:
        raise HarnessError("Codex recovery action is unsupported", kind="session_state")
    if action not in _RECOVERY_ALLOWED_ACTIONS[reason]:
        raise HarnessError("Codex recovery reason and action do not match", kind="session_state")
    if state == "contract_failure" and reason != "response_contract":
        raise HarnessError("Codex contract_failure reason is invalid", kind="session_state")
    return state, reason, action


def _validate_codex_validated_turn_header(record: Mapping[str, Any]) -> None:
    """校验 validated turn 凭证的 JSON 类型、字段集合和 schema。"""

    if not isinstance(record, Mapping) or set(record) != _VALIDATED_TURN_FIELDS:
        raise HarnessError(
            "Codex validated turn fields are invalid; effective thread policy digest is required",
            kind="session_state",
        )
    if record.get("schema_version") != _RECOVERY_RECORD_SCHEMA_VERSION:
        raise HarnessError("Codex validated turn schema is invalid", kind="session_state")


def _validate_codex_validated_turn_identity(
    record: Mapping[str, Any],
) -> tuple[str, str, str, str, str, int, str, str]:
    """校验 validated turn 凭证的 thread、摘要、计数和时间。"""

    thread_id = _validate_thread_id(record.get("thread_id"))
    stage = _validate_text(record.get("stage"), "validated_turn.stage")
    request_digest = record.get("request_digest")
    if not isinstance(request_digest, str) or _REQUEST_DIGEST.fullmatch(request_digest) is None:
        raise HarnessError(
            "Codex validated turn request digest is malformed",
            kind="session_state",
        )
    thread_policy_digest = _validate_thread_policy_digest(
        record.get("thread_policy_digest"),
        "validated_turn.thread_policy_digest",
    )
    turn_id = _validate_thread_id(record.get("turn_id"))
    turn_count = record.get("turn_count")
    if isinstance(turn_count, bool) or not isinstance(turn_count, int) or turn_count < 1:
        raise HarnessError("Codex validated turn count is invalid", kind="session_state")
    created_at = _validate_text(record.get("created_at"), "validated_turn.created_at")
    updated_at = _validate_text(record.get("updated_at"), "validated_turn.updated_at")
    return (
        thread_id,
        stage,
        request_digest,
        thread_policy_digest,
        turn_id,
        turn_count,
        created_at,
        updated_at,
    )


def _validate_model(value: Any) -> str:
    """校验模型名；模型是会话绑定的一部分，不能使用空值。"""

    return _validate_text(value, "model")


def _prepare_directory(path: Path) -> None:
    """创建并收紧会话目录权限；失败时以会话状态错误结束。"""

    try:
        path.mkdir(parents=True, exist_ok=True)
        os.chmod(path, 0o700)
    except OSError as exc:
        raise HarnessError(
            f"Codex session directory cannot be prepared: {path}",
            kind="session_state",
        ) from exc


def _path_is_within(path: Path, parent: Path) -> bool:
    """判断 path 是否落在 parent 内，用于阻止会话目录彼此嵌套。"""

    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


@dataclass(frozen=True, slots=True)
class CodexSessionOwner:
    """Investigation 或 Task Review 会话的不可变归属身份。"""

    kind: str
    research_task_id: str
    investigation_context_id: str | None = None
    hypothesis_execution_input_record_id: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in _OWNER_KINDS:
            raise HarnessError(
                f"Unsupported Codex session owner kind: {self.kind}",
                kind="session_state",
            )
        object.__setattr__(
            self,
            "research_task_id",
            _validate_identity_component(
                self.research_task_id,
                "research_task_id",
            ),
        )
        if self.kind == "investigation":
            if (
                self.investigation_context_id is None
                or self.hypothesis_execution_input_record_id is None
            ):
                raise HarnessError(
                    "Investigation owner requires research_task_id, "
                    "investigation_context_id and "
                    "hypothesis_execution_input_record_id",
                    kind="session_state",
                )
            object.__setattr__(
                self,
                "investigation_context_id",
                _validate_identity_component(
                    self.investigation_context_id,
                    "investigation_context_id",
                ),
            )
            object.__setattr__(
                self,
                "hypothesis_execution_input_record_id",
                _validate_identity_component(
                    self.hypothesis_execution_input_record_id,
                    "hypothesis_execution_input_record_id",
                ),
            )
        elif (
            self.investigation_context_id is not None
            or self.hypothesis_execution_input_record_id is not None
        ):
            raise HarnessError(
                "Task Review owner cannot include investigation_context_id or "
                "hypothesis_execution_input_record_id",
                kind="session_state",
            )

    @classmethod
    def investigation(
        cls,
        research_task_id: str,
        investigation_context_id: str,
        hypothesis_execution_input_record_id: str,
    ) -> "CodexSessionOwner":
        """构造由 research_task、Context 和执行输入确定的会话 owner。"""

        return cls(
            "investigation",
            research_task_id,
            investigation_context_id,
            hypothesis_execution_input_record_id,
        )

    @classmethod
    def task_review(cls, research_task_id: str) -> "CodexSessionOwner":
        """构造由 research_task 唯一确定的 Task Review 会话 owner。"""

        return cls("task_review", research_task_id)

    @property
    def identity(self) -> dict[str, str]:
        """返回 manifest 中唯一的规范化会话归属字段。"""

        identity = {
            "kind": self.kind,
            "research_task_id": self.research_task_id,
        }
        if self.kind == "investigation":
            identity["investigation_context_id"] = (
                self.investigation_context_id or ""
            )
            identity["hypothesis_execution_input_record_id"] = (
                self.hypothesis_execution_input_record_id or ""
            )
        return identity


class SchemaV1CodexSessionOwnerReader:
    """只读解析 schema-v1 Codex Session owner。"""

    @staticmethod
    def read(value: Mapping[str, Any]) -> CodexSessionOwner:
        """把旧 task/batch/combination 字段映射成 canonical owner。"""

        if not isinstance(value, Mapping):
            raise HarnessError(
                "schema-v1 Codex session owner must be an object",
                kind="session_state",
            )
        kind = value.get("kind")
        if kind == "investigation" and set(value) == {
            "kind",
            "task_id",
            "batch_id",
            "combination_id",
        }:
            return CodexSessionOwner.investigation(
                value["task_id"],
                value["batch_id"],
                value["combination_id"],
            )
        if kind == "task_review" and set(value) == {"kind", "task_id"}:
            return CodexSessionOwner.task_review(value["task_id"])
        raise HarnessError(
            "schema-v1 Codex session owner fields are invalid",
            kind="session_state",
        )


class CodexSessionStore:
    """持久化一个归属身份明确的 Codex thread。

    构造出的 workspace 和 CODEX_HOME 只属于当前归属身份。manifest 只记录一组
    规范字段；所有绑定字段在加载和写入时都必须与当前 store 完全一致。
    """

    schema_version = _CODEX_SESSION_SCHEMA_VERSION
    manifest_name = _CODEX_MANIFEST_NAME
    lock_name = _CODEX_LOCK_NAME
    recovery_name = _CODEX_RECOVERY_NAME
    recovery_poison_name = _CODEX_RECOVERY_POISON_NAME
    validated_turn_name = _CODEX_VALIDATED_TURN_NAME
    thread_rollback_intent_name = _CODEX_THREAD_ROLLBACK_INTENT_NAME
    thread_rollback_receipt_name = _CODEX_THREAD_ROLLBACK_RECEIPT_NAME

    def __init__(
        self,
        workspace: Path | str,
        *,
        owner: CodexSessionOwner,
        scope_id: str,
        model: str,
        codex_home: Path | str | None = None,
        thread_policy_digest: str | None = None,
    ) -> None:
        if not isinstance(owner, CodexSessionOwner):
            raise HarnessError("Codex session owner must be CodexSessionOwner", kind="session_state")
        self.owner = owner
        self.workspace = Path(workspace).expanduser().resolve()
        self.scope_id = _validate_identity_component(scope_id, "scope_id")
        self.model = _validate_model(model)
        self._expected_thread_policy_digest = (
            _validate_thread_policy_digest(thread_policy_digest)
            if thread_policy_digest is not None
            else None
        )
        self.codex_home = (
            Path(codex_home).expanduser().resolve()
            if codex_home is not None
            else self.workspace / ".codex_home"
        )
        self._state_root = (
            self.codex_home.parent / "state" if codex_home is not None else self.workspace
        )
        if codex_home is not None and (
            _path_is_within(self.codex_home, self.workspace)
            or _path_is_within(self.workspace, self.codex_home)
        ):
            raise HarnessError(
                "Codex session CODEX_HOME must be separate from workspace",
                kind="session_state",
            )
        _prepare_directory(self.workspace)
        _prepare_directory(self.codex_home)
        if self._state_root != self.workspace and self._state_root != self.codex_home:
            _prepare_directory(self._state_root)

    @classmethod
    def from_root(
        cls,
        root: Path | str,
        *,
        owner: CodexSessionOwner,
        scope_id: str,
        model: str,
        codex_home: Path | str | None = None,
        thread_policy_digest: str | None = None,
    ) -> "CodexSessionStore":
        """按会话归属身份派生并列的 workspace、CODEX_HOME 和 state 目录。"""

        if not isinstance(owner, CodexSessionOwner):
            raise HarnessError("Codex session owner must be CodexSessionOwner", kind="session_state")
        root_path = Path(root).expanduser().resolve()
        if owner.kind == "investigation":
            owner_root = (
                root_path
                / "investigations"
                / owner.research_task_id
                / (owner.investigation_context_id or "")
                / (owner.hypothesis_execution_input_record_id or "")
            )
        else:
            owner_root = root_path / "task_reviews" / owner.research_task_id
        workspace = owner_root / "workspace"
        resolved_codex_home = (
            Path(codex_home).expanduser().resolve()
            if codex_home is not None
            else owner_root / "codex_home"
        )
        _prepare_directory(owner_root)
        return cls(
            workspace,
            owner=owner,
            scope_id=scope_id,
            model=model,
            codex_home=resolved_codex_home,
            thread_policy_digest=thread_policy_digest,
        )

    @property
    def manifest_path(self) -> Path:
        """返回当前会话的规范 manifest 路径。"""

        return self._state_root / self.manifest_name

    @property
    def lock_path(self) -> Path:
        """返回覆盖整个 Codex turn 的锁文件路径。"""

        return self._state_root / self.lock_name

    @property
    def recovery_path(self) -> Path:
        """返回保存合同失败或 worker 清理状态的独立记录路径。"""

        return self._state_root / self.recovery_name

    @property
    def recovery_poison_path(self) -> Path:
        """返回恢复双写都失败时阻断后续调用的独立持久标记路径。"""

        return self._state_root / self.recovery_poison_name

    @property
    def validated_turn_path(self) -> Path:
        """返回响应通过结构合同后保存的独立凭证路径。"""

        return self._state_root / self.validated_turn_name

    @property
    def thread_rollback_intent_path(self) -> Path:
        """返回 Investigation turn rollback 意图的持久路径。"""

        return self._state_root / self.thread_rollback_intent_name

    @property
    def thread_rollback_receipt_path(self) -> Path:
        """返回 Investigation turn rollback 成功回执的持久路径。"""

        return self._state_root / self.thread_rollback_receipt_name

    @contextmanager
    def locked(self) -> Iterator[None]:
        """锁住整个调用方 turn，并支持同一线程嵌套调用 store 方法。"""

        _prepare_directory(self.workspace)
        with _lock_session_file(self.lock_path, session_label="Codex"):
            yield

    def _identity_conflict(self, field: str, expected: Any, actual: Any) -> None:
        """报告任何可能让一个会话归属身份复用另一 thread 的身份冲突。"""

        raise HarnessError(
            f"Codex session identity conflict for {field}: "
            f"expected {expected!r}, got {actual!r}",
            kind="session_state",
        )

    def _validate_manifest_shape(self, manifest: Mapping[str, Any]) -> None:
        """校验 Codex manifest 的 JSON 类型、字段集合和 schema 版本。"""

        if not isinstance(manifest, Mapping):
            raise HarnessError("Codex session manifest must be a JSON object", kind="session_state")
        missing = _CODEX_REQUIRED_FIELDS.difference(manifest)
        if missing:
            raise HarnessError(
                f"Codex session manifest is missing fields: {sorted(missing)}",
                kind="session_state",
            )
        unexpected = set(manifest).difference(_CODEX_REQUIRED_FIELDS | _CODEX_OPTIONAL_FIELDS)
        if unexpected:
            raise HarnessError(
                f"Codex session manifest has unsupported fields: {sorted(unexpected)}",
                kind="session_state",
            )
        schema_version = manifest.get("schema_version")
        if isinstance(schema_version, bool) or not isinstance(schema_version, int):
            raise HarnessError("Codex session manifest schema_version is invalid", kind="session_state")
        if schema_version != self.schema_version:
            raise HarnessError("Unsupported Codex session manifest schema", kind="session_state")

    def _normalize_manifest_schema(
        self,
        manifest: Mapping[str, Any],
    ) -> dict[str, Any]:
        """把 schema-v1 owner 一次性映射为 schema-v2 canonical 结构。"""

        if not isinstance(manifest, Mapping):
            raise HarnessError(
                "Codex session manifest must be a JSON object",
                kind="session_state",
            )
        schema_version = manifest.get("schema_version")
        if schema_version == self.schema_version:
            return dict(manifest)
        if schema_version != _LEGACY_CODEX_SESSION_SCHEMA_VERSION:
            raise HarnessError(
                "Unsupported Codex session manifest schema",
                kind="session_state",
            )
        if set(manifest).difference(_CODEX_REQUIRED_FIELDS | _CODEX_OPTIONAL_FIELDS):
            raise HarnessError(
                "schema-v1 Codex session manifest has unsupported fields",
                kind="session_state",
            )
        normalized = dict(manifest)
        normalized["schema_version"] = self.schema_version
        normalized["owner"] = SchemaV1CodexSessionOwnerReader.read(
            manifest.get("owner")
        ).identity
        return normalized

    def _validate_manifest_owner_and_paths(self, manifest: Mapping[str, Any]) -> None:
        """核对 manifest 的 owner、workspace 和 CODEX_HOME 绑定。"""

        actual_owner = manifest.get("owner")
        if not isinstance(actual_owner, Mapping):
            raise HarnessError("Codex session manifest owner is invalid", kind="session_state")
        if dict(actual_owner) != self.owner.identity:
            self._identity_conflict("owner", self.owner.identity, dict(actual_owner))
        expected_paths = {
            "workspace": str(self.workspace),
            "codex_home": str(self.codex_home),
        }
        for field, expected in expected_paths.items():
            actual = manifest.get(field)
            if not isinstance(actual, str):
                raise HarnessError(
                    f"Codex session manifest {field} is invalid",
                    kind="session_state",
                )
            if str(Path(actual).expanduser().resolve()) != expected:
                self._identity_conflict(field, expected, actual)
        for field, expected in (("scope_id", self.scope_id), ("model", self.model)):
            actual = _validate_text(manifest.get(field), field)
            if actual != expected:
                self._identity_conflict(field, expected, actual)

    def _read_manifest_state_values(
        self,
        manifest: Mapping[str, Any],
    ) -> tuple[str, str, str, int, str, str, str | None]:
        """校验 thread、阶段、计数、时间和有效 thread policy digest。"""

        thread_id = _validate_thread_id(manifest.get("thread_id"))
        stage = _validate_text(manifest.get("last_stage"), "last_stage")
        status = _validate_text(manifest.get("status"), "status")
        turn_count = manifest.get("turn_count")
        if isinstance(turn_count, bool) or not isinstance(turn_count, int) or turn_count < 0:
            raise HarnessError("Codex session manifest turn_count is invalid", kind="session_state")
        created_at = _validate_text(manifest.get("created_at"), "created_at")
        updated_at = _validate_text(manifest.get("updated_at"), "updated_at")
        stored_policy_digest = manifest.get("thread_policy_digest")
        if stored_policy_digest is not None:
            stored_policy_digest = _validate_thread_policy_digest(stored_policy_digest)
        if self._expected_thread_policy_digest is not None:
            if stored_policy_digest is None:
                raise HarnessError(
                    "Codex session manifest lacks effective thread policy digest",
                    kind="session_state",
                )
            if stored_policy_digest != self._expected_thread_policy_digest:
                self._identity_conflict(
                    "thread_policy_digest",
                    self._expected_thread_policy_digest,
                    stored_policy_digest,
                )
        return (
            thread_id,
            stage,
            status,
            turn_count,
            created_at,
            updated_at,
            stored_policy_digest,
        )

    def _validate_manifest(self, manifest: Mapping[str, Any]) -> dict[str, Any]:
        """校验 manifest 并返回只含规范字段的副本。"""

        manifest = self._normalize_manifest_schema(manifest)
        self._validate_manifest_shape(manifest)
        self._validate_manifest_owner_and_paths(manifest)
        (
            thread_id,
            stage,
            status,
            turn_count,
            created_at,
            updated_at,
            stored_policy_digest,
        ) = self._read_manifest_state_values(manifest)
        value = {
            "schema_version": self.schema_version,
            "owner": dict(self.owner.identity),
            "workspace": str(self.workspace),
            "scope_id": self.scope_id,
            "model": self.model,
            "codex_home": str(self.codex_home),
            "thread_id": thread_id,
            "last_stage": stage,
            "turn_count": turn_count,
            "status": status,
            "created_at": created_at,
            "updated_at": updated_at,
        }
        if stored_policy_digest is not None:
            value["thread_policy_digest"] = stored_policy_digest
        return value

    def _read_manifest(self) -> dict[str, Any] | None:
        """读取一个原子替换过的 manifest，并先拒绝符号链接。"""

        if self.manifest_path.is_symlink():
            raise HarnessError("Codex session manifest must not be a symlink", kind="session_state")
        if not self.manifest_path.exists():
            return None
        try:
            os.chmod(self.manifest_path, 0o600)
            parsed = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HarnessError(
                f"Codex session manifest cannot be read: {self.manifest_path}",
                kind="session_state",
            ) from exc
        return self._validate_manifest(parsed)

    def _validate_recovery_record_shape(
        self,
        record: Mapping[str, Any],
    ) -> tuple[str, str, str]:
        """校验恢复记录字段集合、schema、状态、原因和动作关系。"""

        _validate_codex_recovery_record_header(record)
        return _validate_codex_recovery_state_reason_action(record)

    def _validate_recovery_record_identity(
        self,
        record: Mapping[str, Any],
    ) -> tuple[str, str, str, str, int, str, str]:
        """校验恢复记录的阶段、摘要、turn 身份、计数和时间。"""

        stage, request_digest, thread_policy_digest = _validate_codex_recovery_digest_identity(
            record,
            field_prefix="recovery",
        )
        turn_id, turn_count, created_at, updated_at = _validate_codex_recovery_turn_metadata(
            record,
            field_prefix="recovery",
        )
        return stage, request_digest, thread_policy_digest, turn_id, turn_count, created_at, updated_at

    def _validate_recovery_record(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """校验恢复记录，拒绝未知字段、错误摘要和不明恢复动作。"""

        state, reason, action = self._validate_recovery_record_shape(record)
        (
            stage,
            request_digest,
            thread_policy_digest,
            turn_id,
            turn_count,
            created_at,
            updated_at,
        ) = self._validate_recovery_record_identity(record)
        return {
            "schema_version": _RECOVERY_RECORD_SCHEMA_VERSION,
            "state": state,
            "reason": reason,
            "action": action,
            "stage": stage,
            "request_digest": request_digest,
            "thread_policy_digest": thread_policy_digest,
            "turn_id": turn_id,
            "turn_count": turn_count,
            "created_at": created_at,
            "updated_at": updated_at,
        }

    def _read_recovery_record(self) -> dict[str, Any] | None:
        """读取独立恢复记录；任何残留临时文件或符号链接都停止恢复。"""

        if self.recovery_path.is_symlink():
            raise HarnessError("Codex recovery record must not be a symlink", kind="session_state")
        try:
            incomplete = [
                child
                for child in self._state_root.iterdir()
                if child.name.startswith(f"{self.recovery_name}.")
            ]
        except OSError as exc:
            raise HarnessError("Codex recovery record directory cannot be listed", kind="session_state") from exc
        if incomplete:
            raise HarnessError("Codex recovery record has an incomplete entry", kind="session_state")
        if not self.recovery_path.exists():
            return None
        if not self.recovery_path.is_file():
            raise HarnessError("Codex recovery record is not a regular file", kind="session_state")
        try:
            os.chmod(self.recovery_path, 0o600)
            parsed = json.loads(self.recovery_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HarnessError("Codex recovery record cannot be read", kind="session_state") from exc
        return self._validate_recovery_record(parsed)

    def _read_auxiliary_record(
        self,
        path: Path,
        *,
        name: str,
        validator: Callable[[Mapping[str, Any]], dict[str, Any]],
    ) -> dict[str, Any] | None:
        """读取一个状态辅助文件；临时残留、链接、损坏 JSON 都停止恢复。"""

        if path.is_symlink():
            raise HarnessError(f"Codex {name} must not be a symlink", kind="session_state")
        try:
            incomplete = [
                child
                for child in self._state_root.iterdir()
                if child.name.startswith(f"{path.name}.")
            ]
        except OSError as exc:
            raise HarnessError(
                f"Codex {name} directory cannot be listed",
                kind="session_state",
            ) from exc
        if incomplete:
            raise HarnessError(f"Codex {name} has an incomplete entry", kind="session_state")
        if not path.exists():
            return None
        if not path.is_file():
            raise HarnessError(f"Codex {name} is not a regular file", kind="session_state")
        try:
            os.chmod(path, 0o600)
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HarnessError(f"Codex {name} cannot be read", kind="session_state") from exc
        return validator(parsed)

    def _validate_recovery_poison_shape(
        self,
        record: Mapping[str, Any],
    ) -> str:
        """校验 poison 字段集合、schema、状态、原因和人工恢复动作。"""

        if not isinstance(record, Mapping) or set(record) != _RECOVERY_POISON_FIELDS:
            raise HarnessError(
                "Codex recovery poison fields are invalid; effective thread policy digest is required",
                kind="session_state",
            )
        if record.get("schema_version") != _RECOVERY_RECORD_SCHEMA_VERSION:
            raise HarnessError("Codex recovery poison schema is invalid", kind="session_state")
        if _validate_text(record.get("state"), "recovery_poison.state") != "poisoned":
            raise HarnessError("Codex recovery poison state is invalid", kind="session_state")
        reason = _validate_text(record.get("reason"), "recovery_poison.reason")
        if reason not in _RECOVERY_REASONS:
            raise HarnessError("Codex recovery poison reason is unsupported", kind="session_state")
        if _validate_text(record.get("action"), "recovery_poison.action") != "manual_recovery_required":
            raise HarnessError("Codex recovery poison action is invalid", kind="session_state")
        return reason

    def _validate_recovery_poison_identity(
        self,
        record: Mapping[str, Any],
    ) -> tuple[str, str, str, str, int, str, str]:
        """校验 poison 的阶段、摘要、turn 身份、计数和时间。"""

        stage, request_digest, thread_policy_digest = _validate_codex_recovery_digest_identity(
            record,
            field_prefix="recovery_poison",
        )
        turn_id, turn_count, created_at, updated_at = _validate_codex_recovery_turn_metadata(
            record,
            field_prefix="recovery_poison",
        )
        return stage, request_digest, thread_policy_digest, turn_id, turn_count, created_at, updated_at

    def _validate_recovery_poison(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """校验恢复双写失败标记，防止重启时绕过未保存的阻断状态。"""

        reason = self._validate_recovery_poison_shape(record)
        (
            stage,
            request_digest,
            thread_policy_digest,
            turn_id,
            turn_count,
            created_at,
            updated_at,
        ) = self._validate_recovery_poison_identity(record)
        return {
            "schema_version": _RECOVERY_RECORD_SCHEMA_VERSION,
            "state": "poisoned",
            "reason": reason,
            "action": "manual_recovery_required",
            "stage": stage,
            "request_digest": request_digest,
            "thread_policy_digest": thread_policy_digest,
            "turn_id": turn_id,
            "turn_count": turn_count,
            "created_at": created_at,
            "updated_at": updated_at,
        }

    def _read_recovery_poison(self) -> dict[str, Any] | None:
        """读取持久 poison 标记；存在时调用方必须停止新 turn。"""

        return self._read_auxiliary_record(
            self.recovery_poison_path,
            name="recovery poison",
            validator=self._validate_recovery_poison,
        )

    def load_recovery_poison(self) -> dict[str, Any] | None:
        """读取恢复双写失败标记，不改变任何会话状态。"""

        return self._read_recovery_poison()

    def record_recovery_poison(
        self,
        *,
        stage: str,
        request_digest: str,
        thread_policy_digest: str,
        turn_id: str | None,
        reason: str,
    ) -> dict[str, Any]:
        """保存恢复双写均失败的持久阻断标记，阻止重启后盲目重试。

        输入是当前阶段、完整请求摘要、可选 turn 标识和原始恢复原因；动作是
        在会话状态目录原子写入独立 poison 文件，重复写入同一身份幂等返回。副作用
        是后续 ``reconcile_recovery_state`` 一律拒绝新 turn；文件系统写入失败会
        抛出 ``HarnessError(kind="session_state")``，调用方必须继续保留
        ``recovery_pending`` 分类并等待人工恢复。
        """

        (
            validated_stage,
            validated_request_digest,
            validated_thread_policy_digest,
            validated_turn_id,
            validated_reason,
        ) = _validate_codex_recovery_poison_arguments(
            stage,
            request_digest,
            thread_policy_digest,
            turn_id,
            reason,
        )
        self._expected_thread_policy_digest = validated_thread_policy_digest
        with self.locked():
            current = self._read_manifest()
            if current is None:
                raise HarnessError(
                    "Cannot record recovery poison before creating a Codex session",
                    kind="session_state",
                )
            existing = self._read_recovery_poison()
            expected = {
                "schema_version": _RECOVERY_RECORD_SCHEMA_VERSION,
                "state": "poisoned",
                "reason": validated_reason,
                "action": "manual_recovery_required",
                "stage": validated_stage,
                "request_digest": validated_request_digest,
                "thread_policy_digest": validated_thread_policy_digest,
                "turn_id": validated_turn_id,
                "turn_count": current["turn_count"],
            }
            if existing is not None:
                if all(existing.get(key) == value for key, value in expected.items()):
                    return existing
                raise HarnessError(
                    "Codex recovery poison conflicts with the current identity",
                    kind="session_state",
                )
            now = _utc_now()
            return self._write_auxiliary_record(
                self.recovery_poison_path,
                {**expected, "created_at": now, "updated_at": now},
                name="recovery poison",
                validator=self._validate_recovery_poison,
            )

    def record_validated_turn(
        self,
        *,
        stage: str,
        request_digest: str,
        thread_policy_digest: str,
        turn_id: str,
    ) -> dict[str, Any]:
        """保存响应通过结构合同的凭证，供恢复清理核对而非直接推进计数。

        输入是当前阶段、完整请求摘要和已完成 turn 标识；动作是在锁内读取
        manifest 并原子保存同一 thread、当前 turn_count 的凭证。副作用是允许之后
        的 ``clear_recovery_after_validated_turn`` 或重启 reconcile 删除旧恢复记录，
        但不会修改 manifest 计数。身份、计数或落盘失败都会抛出状态错误并保留原记录。
        """

        (
            validated_stage,
            validated_request_digest,
            validated_thread_policy_digest,
            validated_turn_id,
        ) = _validate_codex_validated_turn_arguments(
            stage,
            request_digest,
            thread_policy_digest,
            turn_id,
        )
        with self.locked():
            current = self._read_manifest()
            if current is None:
                raise HarnessError(
                    "Cannot record validated turn before creating a Codex session",
                    kind="session_state",
                )
            if current["status"] not in {"completed", "active"} or current["turn_count"] < 1:
                raise HarnessError(
                    "Codex validated turn requires a completed manifest",
                    kind="session_state",
                )
            expected = {
                "schema_version": _RECOVERY_RECORD_SCHEMA_VERSION,
                "thread_id": current["thread_id"],
                "stage": validated_stage,
                "request_digest": validated_request_digest,
                "thread_policy_digest": validated_thread_policy_digest,
                "turn_id": validated_turn_id,
                "turn_count": current["turn_count"],
            }
            existing = self._read_validated_turn()
            if existing is not None:
                if all(existing.get(key) == value for key, value in expected.items()):
                    return existing
                raise HarnessError(
                    "Codex validated turn conflicts with the current turn",
                    kind="session_state",
                )
            now = _utc_now()
            return self._write_auxiliary_record(
                self.validated_turn_path,
                {**expected, "created_at": now, "updated_at": now},
                name="validated turn",
                validator=self._validate_validated_turn,
            )

    def _validate_validated_turn(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """校验合同通过凭证，确保它只指向当前 thread 的已完成 turn。"""

        _validate_codex_validated_turn_header(record)
        (
            thread_id,
            stage,
            request_digest,
            thread_policy_digest,
            turn_id,
            turn_count,
            created_at,
            updated_at,
        ) = _validate_codex_validated_turn_identity(record)
        return {
            "schema_version": _RECOVERY_RECORD_SCHEMA_VERSION,
            "thread_id": thread_id,
            "stage": stage,
            "request_digest": request_digest,
            "thread_policy_digest": thread_policy_digest,
            "turn_id": turn_id,
            "turn_count": turn_count,
            "created_at": created_at,
            "updated_at": updated_at,
        }

    def _read_validated_turn(self) -> dict[str, Any] | None:
        """读取合同通过凭证；损坏时不删除或猜测其含义。"""

        return self._read_auxiliary_record(
            self.validated_turn_path,
            name="validated turn",
            validator=self._validate_validated_turn,
        )

    def _write_auxiliary_record(
        self,
        path: Path,
        value: Mapping[str, Any],
        *,
        name: str,
        validator: Callable[[Mapping[str, Any]], dict[str, Any]],
    ) -> dict[str, Any]:
        """原子保存状态辅助文件并刷盘目录，失败时保留 fail-closed 语义。"""

        normalized = dict(value)
        normalized["updated_at"] = _utc_now()
        checked = validator(normalized)
        _atomic_write_json_file(path, checked, description=f"Codex {name}")
        return checked

    def _remove_auxiliary_record(self, path: Path, *, name: str) -> None:
        """幂等删除一个已消费的状态辅助文件，并刷盘父目录。"""

        _remove_file_and_fsync_directory(path, description=f"Codex {name}")

    def load_recovery(self) -> dict[str, Any] | None:
        """加载当前会话的合同失败或 worker 清理记录，不修改任何状态。"""

        return self._read_recovery_record()

    def load_thread_rollback_intent(self) -> dict[str, Any] | None:
        """读取 Investigation turn 的 rollback 意图，不启动模型或修改会话。"""

        return self._read_auxiliary_record(
            self.thread_rollback_intent_path,
            name="thread rollback intent",
            validator=lambda record: _validate_thread_rollback_record(
                record,
                expected_state="intent",
            ),
        )

    def load_thread_rollback_receipt(self) -> dict[str, Any] | None:
        """读取已经完成的 thread rollback 回执，供重启幂等判定。"""

        return self._read_auxiliary_record(
            self.thread_rollback_receipt_path,
            name="thread rollback receipt",
            validator=lambda record: _validate_thread_rollback_record(
                record,
                expected_state="rolled_back",
            ),
        )

    def record_thread_rollback_intent(
        self,
        *,
        stage: str,
        thread_id: str,
        request_digest: str,
        thread_policy_digest: str,
        turn_count: int,
        physical_turn_count: int,
        provider_error_code: str,
    ) -> dict[str, Any]:
        """原子保存 Investigation turn rollback 意图，并核对 thread 归属。"""

        (
            validated_stage,
            validated_thread_id,
            validated_request_digest,
            validated_policy,
            validated_turn_count,
            validated_physical_count,
            validated_error_code,
        ) = _validate_thread_rollback_arguments(
            stage=stage,
            thread_id=thread_id,
            request_digest=request_digest,
            thread_policy_digest=thread_policy_digest,
            turn_count=turn_count,
            physical_turn_count=physical_turn_count,
            provider_error_code=provider_error_code,
        )
        self._expected_thread_policy_digest = validated_policy
        with self.locked():
            current = self._read_manifest()
            if current is None:
                raise HarnessError(
                    "Cannot record thread rollback before creating a Codex session",
                    kind="session_state",
                )
            if current["thread_id"] != validated_thread_id:
                raise HarnessError(
                    "Codex thread rollback thread identity does not match session",
                    kind="session_state",
                )
            if current["turn_count"] != validated_turn_count:
                raise HarnessError(
                    "Codex thread rollback successful count does not match session",
                    kind="session_state",
                )
            expected = {
                "schema_version": _CODEX_THREAD_ROLLBACK_SCHEMA_VERSION,
                "state": "intent",
                "stage": validated_stage,
                "thread_id": validated_thread_id,
                "request_digest": validated_request_digest,
                "thread_policy_digest": validated_policy,
                "turn_count": validated_turn_count,
                "physical_turn_count": validated_physical_count,
                "provider_error_code": validated_error_code,
            }
            existing = self.load_thread_rollback_intent()
            if existing is not None:
                if all(existing.get(key) == value for key, value in expected.items()):
                    return existing
                raise HarnessError(
                    "Codex thread rollback intent conflicts with current identity",
                    kind="session_state",
                )
            now = _utc_now()
            return self._write_auxiliary_record(
                self.thread_rollback_intent_path,
                {**expected, "created_at": now, "updated_at": now},
                name="thread rollback intent",
                validator=lambda record: _validate_thread_rollback_record(
                    record,
                    expected_state="intent",
                ),
            )

    def record_thread_rollback_receipt(
        self,
        *,
        stage: str,
        thread_id: str,
        request_digest: str,
        thread_policy_digest: str,
        turn_count: int,
        physical_turn_count: int,
        provider_error_code: str,
    ) -> dict[str, Any]:
        """原子保存官方 rollback 回执；重复恢复只返回同一成功事实。"""

        (
            validated_stage,
            validated_thread_id,
            validated_request_digest,
            validated_policy,
            validated_turn_count,
            validated_physical_count,
            validated_error_code,
        ) = _validate_thread_rollback_arguments(
            stage=stage,
            thread_id=thread_id,
            request_digest=request_digest,
            thread_policy_digest=thread_policy_digest,
            turn_count=turn_count,
            physical_turn_count=physical_turn_count,
            provider_error_code=provider_error_code,
        )
        self._expected_thread_policy_digest = validated_policy
        with self.locked():
            current = self._read_manifest()
            if current is None or current["thread_id"] != validated_thread_id:
                raise HarnessError(
                    "Codex thread rollback receipt thread identity does not match session",
                    kind="session_state",
                )
            if current["turn_count"] != validated_turn_count:
                raise HarnessError(
                    "Codex thread rollback receipt successful count does not match session",
                    kind="session_state",
                )
            existing = self.load_thread_rollback_receipt()
            expected = {
                "schema_version": _CODEX_THREAD_ROLLBACK_SCHEMA_VERSION,
                "state": "rolled_back",
                "stage": validated_stage,
                "thread_id": validated_thread_id,
                "request_digest": validated_request_digest,
                "thread_policy_digest": validated_policy,
                "turn_count": validated_turn_count,
                "physical_turn_count": validated_physical_count,
                "provider_error_code": validated_error_code,
            }
            if existing is not None:
                if all(existing.get(key) == value for key, value in expected.items()):
                    return existing
                raise HarnessError(
                    "Codex thread rollback receipt conflicts with current identity",
                    kind="session_state",
                )
            intent = self.load_thread_rollback_intent()
            if intent is None or any(intent.get(key) != value for key, value in expected.items() if key != "state"):
                raise HarnessError(
                    "Codex thread rollback receipt has no matching intent",
                    kind="session_state",
                )
            now = _utc_now()
            return self._write_auxiliary_record(
                self.thread_rollback_receipt_path,
                {**expected, "created_at": now, "updated_at": now},
                name="thread rollback receipt",
                validator=lambda record: _validate_thread_rollback_record(
                    record,
                    expected_state="rolled_back",
                ),
            )

    # 为调用方保留带 Codex 前缀的明确别名，两个名称都指向同一持久合同。
    record_codex_thread_rollback_intent = record_thread_rollback_intent
    record_codex_thread_rollback_receipt = record_thread_rollback_receipt
    load_codex_thread_rollback_intent = load_thread_rollback_intent
    load_codex_thread_rollback_receipt = load_thread_rollback_receipt

    def _validate_recovery_policy_binding(
        self,
        expected_thread_policy_digest: str | None,
        *,
        poison: Mapping[str, Any] | None,
        recovery: Mapping[str, Any] | None,
        validated: Mapping[str, Any] | None,
    ) -> None:
        """核对恢复辅助记录是否都绑定当前 effective thread policy。"""

        if expected_thread_policy_digest is None:
            return
        for record_name, record in (
            ("recovery poison", poison),
            ("recovery", recovery),
            ("validated turn", validated),
        ):
            if record is not None and record["thread_policy_digest"] != expected_thread_policy_digest:
                raise HarnessError(
                    f"Codex {record_name} effective thread policy does not match current config",
                    kind="session_state",
                )

    def _reconcile_manifest_without_recovery(
        self,
        current: dict[str, Any],
        validated: Mapping[str, Any] | None,
    ) -> tuple[dict[str, Any], None]:
        """恢复记录已删除时清理残留 validated 凭证。"""

        if validated is not None:
            # recovery 已经删除后进程才退出时，凭证只剩清理作用；不把它
            # 当成下一次响应的成功证明。
            self._remove_auxiliary_record(self.validated_turn_path, name="validated turn")
        return current, None

    def _reconcile_manifest_with_recovery(
        self,
        current: dict[str, Any],
        recovery: Mapping[str, Any],
        validated: Mapping[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """根据已验证 turn 证据清理旧恢复，或恢复 manifest 阻断状态。"""

        if (
            recovery["state"] == "recovery_pending"
            and current["status"] in {"completed", "active"}
            and current["turn_count"] > recovery["turn_count"]
        ):
            if (
                validated is not None
                and validated["thread_id"] == current["thread_id"]
                and validated["stage"] == current["last_stage"]
                and validated["turn_count"] == current["turn_count"]
                and validated["turn_count"] > recovery["turn_count"]
            ):
                self._remove_recovery_record()
                self._remove_auxiliary_record(self.validated_turn_path, name="validated turn")
                return current, None
            return current, dict(recovery)
        expected_status = (
            "contract_failure"
            if recovery["state"] == "contract_failure"
            else "recovery_pending"
        )
        if current["status"] != expected_status:
            current["last_stage"] = recovery["stage"]
            current["status"] = expected_status
            current = self._write_atomic(current)
        return current, dict(recovery)

    def reconcile_recovery_state(
        self,
        *,
        thread_policy_digest: str | None = None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """修复恢复记录与 manifest 的已知写入先后崩溃窗口。

        输入是当前 session store 的两个持久文件；动作是在锁内核对其状态，必要时
        补写 ``recovery_pending``/``contract_failure`` manifest，或在已验证新 turn
        已计数而只剩旧恢复记录时幂等删除该记录。返回最新的 ``(manifest, recovery)``；
        若存在独立 poison 标记则直接返回 ``recovery_pending`` 错误；文件损坏、无
        manifest 却有恢复记录或身份无法判断时抛出状态错误，不猜测模型是否完成，
        也不启动新的 turn。
        """

        expected_thread_policy_digest = (
            None if thread_policy_digest is None else _validate_thread_policy_digest(thread_policy_digest)
        )
        if expected_thread_policy_digest is not None:
            if (
                self._expected_thread_policy_digest is not None
                and self._expected_thread_policy_digest != expected_thread_policy_digest
            ):
                raise HarnessError(
                    "Codex session effective thread policy does not match current config",
                    kind="session_state",
                )
            self._expected_thread_policy_digest = expected_thread_policy_digest
        with self.locked():
            current = self._read_manifest()
            poison = self._read_recovery_poison()
            recovery = self._read_recovery_record()
            validated = self._read_validated_turn()
            self._validate_recovery_policy_binding(
                expected_thread_policy_digest,
                poison=poison,
                recovery=recovery,
                validated=validated,
            )
            if poison is not None:
                raise HarnessError(
                    "Codex recovery poison marker blocks recovery",
                    kind="recovery_pending",
                )
            if current is None:
                if recovery is not None:
                    raise HarnessError(
                        "Codex recovery record exists without a session manifest",
                        kind="session_state",
                    )
                return None, None
            if recovery is None:
                return self._reconcile_manifest_without_recovery(current, validated)
            # 仅有 completed manifest 不足以证明响应通过结构合同；必须有
            # 同一 thread、同一计数和阶段的独立凭证，才能补做旧记录删除。
            return self._reconcile_manifest_with_recovery(current, recovery, validated)

    def _write_recovery_record(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """以 0600 临时文件和原子替换保存恢复记录。"""

        value = self._validate_recovery_record(record)
        value["updated_at"] = _utc_now()
        _atomic_write_json_file(
            self.recovery_path,
            value,
            description="Codex recovery record",
        )
        return value

    def _remove_recovery_record(self) -> None:
        """删除已经被成功验证的新 turn 替代的旧恢复记录。"""

        _remove_file_and_fsync_directory(
            self.recovery_path,
            description="Codex recovery record",
        )

    def load(self) -> dict[str, Any] | None:
        """加载当前会话归属身份的 thread；任何身份漂移都 fail-closed。"""

        return self._read_manifest()

    def _write_atomic(self, manifest: Mapping[str, Any]) -> dict[str, Any]:
        """以 0600 临时文件、fsync 和原子替换写入规范 manifest。"""

        value = dict(manifest)
        if self._expected_thread_policy_digest is not None and "thread_policy_digest" not in value:
            value["thread_policy_digest"] = self._expected_thread_policy_digest
        value = self._validate_manifest(value)
        value["updated_at"] = _utc_now()
        _atomic_write_json_file(
            self.manifest_path,
            value,
            description=f"Codex session manifest {self.manifest_path}",
        )
        return value

    def create(
        self,
        thread_id: str,
        *,
        stage: str = "initial",
        status: str = "created",
        thread_policy_digest: str | None = None,
    ) -> dict[str, Any]:
        """原子登记已创建的 thread；已有记录仅允许同 thread 幂等恢复。"""

        validated_thread = _validate_thread_id(thread_id)
        validated_stage = _validate_text(stage, "last_stage")
        validated_status = _validate_text(status, "status")
        if thread_policy_digest is not None:
            self._expected_thread_policy_digest = _validate_thread_policy_digest(
                thread_policy_digest
            )
        with self.locked():
            existing = self._read_manifest()
            recovery = self._read_recovery_record()
            if existing is None and recovery is not None:
                raise HarnessError(
                    "Codex recovery record exists without a session manifest",
                    kind="session_state",
                )
            if existing is not None:
                if existing["thread_id"] != validated_thread:
                    self._identity_conflict("thread_id", existing["thread_id"], validated_thread)
                return existing
            now = _utc_now()
            value = {
                "schema_version": self.schema_version,
                "owner": self.owner.identity,
                "workspace": str(self.workspace),
                "scope_id": self.scope_id,
                "model": self.model,
                "codex_home": str(self.codex_home),
                "thread_id": validated_thread,
                "last_stage": validated_stage,
                "turn_count": 0,
                "status": validated_status,
                "created_at": now,
                "updated_at": now,
            }
            if self._expected_thread_policy_digest is not None:
                value["thread_policy_digest"] = self._expected_thread_policy_digest
            return self._write_atomic(value)

    def replace_unstarted_thread(
        self,
        expected_thread_id: str,
        new_thread_id: str,
        *,
        stage: str = "initial",
        status: str = "created",
        thread_policy_digest: str | None = None,
    ) -> dict[str, Any]:
        """原子替换尚未开始 turn 的 thread，并允许同一替换幂等重入。

        输入是上一次进程登记但尚未提交 turn 的旧 thread 以及 SDK 新建的 thread；
        动作在会话锁内核对旧 manifest 必须仍为 ``created/0``，然后原子替换
        ``thread_id``。副作用只更新当前会话 manifest，不删除恢复、回滚或合同凭证；
        任一凭证已经出现、旧 thread 已有 turn 或身份不一致时抛出状态错误，禁止把
        一个已经开始的 turn 当作可重启的空 thread。
        """

        validated_expected = _validate_thread_id(expected_thread_id)
        validated_new = _validate_thread_id(new_thread_id)
        if validated_expected == validated_new:
            raise HarnessError(
                "Codex unstarted thread replacement must use a new thread",
                kind="session_state",
            )
        validated_stage = _validate_text(stage, "last_stage")
        validated_status = _validate_text(status, "status")
        if thread_policy_digest is not None:
            self._expected_thread_policy_digest = _validate_thread_policy_digest(
                thread_policy_digest
            )
        with self.locked():
            current = self._read_manifest()
            if current is None:
                raise HarnessError(
                    "Cannot replace an unstarted thread before creating a session",
                    kind="session_state",
                )
            if current["thread_id"] == validated_new:
                if current["status"] == "created" and current["turn_count"] == 0:
                    return current
                raise HarnessError(
                    "Codex replacement target already has a turn",
                    kind="session_state",
                )
            if current["thread_id"] != validated_expected:
                self._identity_conflict(
                    "thread_id",
                    validated_expected,
                    current["thread_id"],
                )
            if current["status"] != "created" or current["turn_count"] != 0:
                raise HarnessError(
                    "Codex thread replacement requires an unstarted session",
                    kind="session_state",
                )
            # 这些记录说明 provider 曾经接受过或正在处理一次 turn；即使 manifest
            # 计数尚未推进，也不能通过换 thread 绕过原有恢复合同。
            if any(
                record is not None
                for record in (
                    self._read_recovery_record(),
                    self._read_recovery_poison(),
                    self._read_validated_turn(),
                    self.load_thread_rollback_intent(),
                    self.load_thread_rollback_receipt(),
                )
            ):
                raise HarnessError(
                    "Codex thread replacement is blocked by existing turn recovery state",
                    kind="recovery_pending",
                )
            current["thread_id"] = validated_new
            current["last_stage"] = validated_stage
            current["status"] = validated_status
            return self._write_atomic(current)

    def record_completed_turn(
        self,
        *,
        stage: str,
        status: str = "active",
    ) -> dict[str, Any]:
        """在锁内把同一 thread 的已完成 turn 计数加一并更新阶段。

        合同失败或 worker 无法停止时拒绝直接计数；只有合同失败已经由调用方
        明确登记新请求，且该请求通过后续验证，才允许继续推进会话。
        """

        validated_stage = _validate_text(stage, "last_stage")
        validated_status = _validate_text(status, "status")
        with self.locked():
            current = self._read_manifest()
            if current is None:
                raise HarnessError(
                    "Cannot record a turn before creating a Codex session",
                    kind="session_state",
                )
            recovery = self._read_recovery_record()
            if current["status"] == "contract_failure":
                raise HarnessError(
                    "Codex session has a contract failure; explicit recovery is required",
                    kind="session_state",
                )
            if recovery is not None:
                if not (
                    current["status"] == "recovery_pending"
                    and recovery["state"] == "recovery_pending"
                    and (
                        (
                            recovery["action"] == "explicit_new_request_required"
                            and recovery["reason"]
                            in {"response_contract", "worker_cleanup", "request_intent"}
                        )
                        or (
                            recovery["action"] == "goal_recovery_required"
                            and recovery["reason"] == "goal"
                        )
                        or (
                            recovery["action"] == "receipt_replay_required"
                            and recovery["reason"] == "completion_receipt"
                        )
                    )
                ):
                    raise HarnessError(
                        "Codex session recovery is still pending",
                        kind="session_state",
                    )
            current["last_stage"] = validated_stage
            current["status"] = validated_status
            current["turn_count"] += 1
            return self._write_atomic(current)

    def _build_contract_failure_identity(
        self,
        *,
        stage: str,
        request_digest: str,
        thread_policy_digest: str,
        turn_id: str,
        turn_count: int,
        created_at: str | None = None,
    ) -> dict[str, Any]:
        """构造只含恢复身份的合同失败记录，不保存 prompt 或响应正文。"""

        now = _utc_now()
        return {
            "schema_version": _RECOVERY_RECORD_SCHEMA_VERSION,
            "state": "contract_failure",
            "reason": "response_contract",
            "action": "explicit_new_request_required",
            "stage": stage,
            "request_digest": request_digest,
            "thread_policy_digest": thread_policy_digest,
            "turn_id": turn_id,
            "turn_count": turn_count,
            "created_at": created_at or now,
            "updated_at": now,
        }

    @staticmethod
    def _contract_failure_matches(
        existing: Mapping[str, Any],
        *,
        stage: str,
        request_digest: str,
        thread_policy_digest: str,
        turn_id: str,
        turn_count: int,
    ) -> bool:
        """判断现有恢复记录是否已经是同一合同失败身份。"""

        expected = {
            "state": "contract_failure",
            "reason": "response_contract",
            "action": "explicit_new_request_required",
            "stage": stage,
            "request_digest": request_digest,
            "thread_policy_digest": thread_policy_digest,
            "turn_id": turn_id,
            "turn_count": turn_count,
        }
        return all(existing.get(key) == value for key, value in expected.items())

    @staticmethod
    def _receipt_recovery_matches_contract_failure(
        current: Mapping[str, Any],
        existing: Mapping[str, Any],
        *,
        stage: str,
        request_digest: str,
        thread_policy_digest: str,
        turn_id: str,
    ) -> bool:
        """判断 receipt 重放后合同失败的旧恢复身份。"""

        return (
            current["status"] == "completed"
            and current["turn_count"] == existing["turn_count"] + 1
            and existing["state"] == "recovery_pending"
            and existing["reason"] == "completion_receipt"
            and existing["action"] == "receipt_replay_required"
            and existing["stage"] == stage
            and existing["request_digest"] == request_digest
            and existing["thread_policy_digest"] == thread_policy_digest
            and existing["turn_id"] == turn_id
        )

    @staticmethod
    def _new_request_recovery_matches_contract_failure(
        current: Mapping[str, Any],
        existing: Mapping[str, Any],
        *,
        stage: str,
        request_digest: str,
        thread_policy_digest: str,
        turn_id: str,
    ) -> bool:
        """判断旧恢复已确认后新 turn 合同失败的恢复身份。"""

        return (
            current["status"] == "completed"
            and current["turn_count"] == existing["turn_count"] + 1
            and existing["state"] == "recovery_pending"
            and existing["reason"] in {"response_contract", "worker_cleanup", "request_intent"}
            and existing["action"] == "explicit_new_request_required"
            and existing["stage"] == stage
            and existing["thread_policy_digest"] == thread_policy_digest
            and (
                existing["request_digest"] != request_digest
                or existing["turn_id"] != turn_id
            )
        )

    def _save_contract_failure_manifest(
        self,
        current: dict[str, Any],
        *,
        stage: str,
    ) -> dict[str, Any]:
        """把 manifest 标记为合同失败并原子保存。"""

        current["last_stage"] = stage
        current["status"] = "contract_failure"
        return self._write_atomic(current)

    def _classify_contract_failure_recovery_identity(
        self,
        current: Mapping[str, Any],
        existing: Mapping[str, Any] | None,
        *,
        stage: str,
        request_digest: str,
        thread_policy_digest: str,
        turn_id: str,
        turn_count: int,
    ) -> str:
        """判定合同失败记录应幂等返回、替换旧恢复或新建。"""

        if existing is None:
            return "new"
        if self._contract_failure_matches(
            existing,
            stage=stage,
            request_digest=request_digest,
            thread_policy_digest=thread_policy_digest,
            turn_id=turn_id,
            turn_count=turn_count,
        ):
            return "same"
        if self._receipt_recovery_matches_contract_failure(
            current,
            existing,
            stage=stage,
            request_digest=request_digest,
            thread_policy_digest=thread_policy_digest,
            turn_id=turn_id,
        ) or self._new_request_recovery_matches_contract_failure(
            current,
            existing,
            stage=stage,
            request_digest=request_digest,
            thread_policy_digest=thread_policy_digest,
            turn_id=turn_id,
        ):
            return "replace"
        if current["status"] != "recovery_pending" or existing["action"] != "explicit_new_request_required":
            raise HarnessError(
                "Codex recovery record conflicts with contract failure",
                kind="session_state",
            )
        return "new"

    def record_contract_failure(
        self,
        *,
        stage: str,
        request_digest: str,
        thread_policy_digest: str,
        turn_id: str,
    ) -> dict[str, Any]:
        """持久记录 completed turn 的结构合同失败并保留原收据（receipt）。

        输入是已验证的阶段、完整请求摘要和 completed turn 标识；动作是先写独立
        恢复记录，再把 manifest 状态设为 ``contract_failure``，不会删除已保存的
        completed turn 收据、回退计数或再次调用模型。返回更新后的 manifest。
        记录写入失败时抛出
        ``HarnessError(kind="session_state")``，调用方必须保持失败关闭。
        """

        (
            validated_stage,
            validated_request_digest,
            validated_thread_policy_digest,
            validated_turn_id,
        ) = _validate_codex_contract_failure_arguments(
            stage,
            request_digest,
            thread_policy_digest,
            turn_id,
        )
        with self.locked():
            current = self._read_manifest()
            if current is None:
                raise HarnessError(
                    "Cannot record a contract failure before creating a Codex session",
                    kind="session_state",
                )
            existing = self._read_recovery_record()
            turn_count = current["turn_count"]
            recovery_kind = self._classify_contract_failure_recovery_identity(
                current,
                existing,
                stage=validated_stage,
                request_digest=validated_request_digest,
                thread_policy_digest=validated_thread_policy_digest,
                turn_id=validated_turn_id,
                turn_count=turn_count,
            )
            if recovery_kind == "same":
                return self._save_contract_failure_manifest(current, stage=validated_stage)
            if recovery_kind == "replace":
                assert existing is not None
                self._write_recovery_record(
                    self._build_contract_failure_identity(
                        stage=validated_stage,
                        request_digest=validated_request_digest,
                        thread_policy_digest=validated_thread_policy_digest,
                        turn_id=validated_turn_id,
                        turn_count=turn_count,
                        created_at=existing["created_at"],
                    )
                )
                return self._save_contract_failure_manifest(current, stage=validated_stage)
            self._write_recovery_record(
                self._build_contract_failure_identity(
                    stage=validated_stage,
                    request_digest=validated_request_digest,
                    thread_policy_digest=validated_thread_policy_digest,
                    turn_id=validated_turn_id,
                    turn_count=turn_count,
                )
            )
            return self._save_contract_failure_manifest(current, stage=validated_stage)

    @staticmethod
    def _validate_recovery_confirmation_arguments(
        request_digest: str,
        thread_policy_digest: str,
        turn_id: str | None,
    ) -> tuple[str, str, str]:
        """校验恢复确认调用的请求摘要、策略摘要和 turn 标识。"""

        if not isinstance(request_digest, str) or _REQUEST_DIGEST.fullmatch(request_digest) is None:
            raise HarnessError("Codex recovery request digest is malformed", kind="session_state")
        return (
            request_digest,
            _validate_thread_policy_digest(thread_policy_digest),
            "" if turn_id is None else _validate_thread_id(turn_id),
        )

    @staticmethod
    def _recovery_identity_matches(
        current: Mapping[str, Any],
        recovery: Mapping[str, Any],
        *,
        reason: str,
        action: str,
        request_digest: str,
        thread_policy_digest: str,
        turn_id: str,
    ) -> bool:
        """判断当前 manifest 和恢复记录是否匹配一个确认动作。"""

        return (
            current["status"] == "recovery_pending"
            and recovery["state"] == "recovery_pending"
            and recovery["reason"] == reason
            and recovery["action"] == action
            and recovery["request_digest"] == request_digest
            and recovery["thread_policy_digest"] == thread_policy_digest
            and recovery["turn_id"] == turn_id
        )

    def _classify_contract_failure_recovery_transition(
        self,
        current: Mapping[str, Any],
        recovery: Mapping[str, Any],
        *,
        request_digest: str,
        thread_policy_digest: str,
        turn_id: str,
    ) -> str:
        """核对合同失败恢复身份并返回幂等或状态转换动作。"""

        if current["status"] == "recovery_pending":
            if self._recovery_identity_matches(
                current,
                recovery,
                reason="response_contract",
                action="explicit_new_request_required",
                request_digest=request_digest,
                thread_policy_digest=thread_policy_digest,
                turn_id=turn_id,
            ):
                return "idempotent"
            raise HarnessError(
                "Codex session is already in a different recovery state",
                kind="session_state",
            )
        if current["status"] != "contract_failure":
            raise HarnessError(
                "Codex session is not waiting for contract failure recovery",
                kind="session_state",
            )
        if not (
            recovery["state"] == "contract_failure"
            and recovery["reason"] == "response_contract"
            and recovery["action"] == "explicit_new_request_required"
            and recovery["request_digest"] == request_digest
            and recovery["thread_policy_digest"] == thread_policy_digest
            and recovery["turn_id"] == turn_id
        ):
            raise HarnessError(
                "Codex contract failure identity does not match recovery request",
                kind="session_state",
            )
        return "transition"

    def begin_contract_failure_recovery(
        self,
        *,
        request_digest: str,
        thread_policy_digest: str,
        turn_id: str,
    ) -> dict[str, Any]:
        """显式批准结构合同失败后的新请求恢复，不重放旧收据（receipt）。

        输入必须精确匹配已记录的失败 turn 和请求摘要；动作是把状态改为
        ``recovery_pending``，后续只有摘要不同的新请求可以尝试，旧 completed
        turn 收据仍保留。返回更新后的 manifest；身份不匹配、记录缺失或其它恢复原因均
        fail-closed。
        """

        request_digest, validated_thread_policy_digest, validated_turn_id = (
            self._validate_recovery_confirmation_arguments(
                request_digest,
                thread_policy_digest,
                turn_id,
            )
        )
        with self.locked():
            current = self._read_manifest()
            recovery = self._read_recovery_record()
            if current is None or recovery is None:
                raise HarnessError(
                    "Codex contract failure recovery record is missing",
                    kind="session_state",
                )
            transition = self._classify_contract_failure_recovery_transition(
                current,
                recovery,
                request_digest=request_digest,
                thread_policy_digest=validated_thread_policy_digest,
                turn_id=validated_turn_id,
            )
            if transition == "idempotent":
                return current
            now = _utc_now()
            self._write_recovery_record(
                {
                    **recovery,
                    "state": "recovery_pending",
                    "created_at": recovery["created_at"],
                    "updated_at": now,
                }
            )
            current["status"] = "recovery_pending"
            return self._write_atomic(current)

    def record_recovery_pending(
        self,
        *,
        stage: str,
        request_digest: str,
        thread_policy_digest: str,
        turn_id: str | None,
        reason: str = "worker_cleanup",
    ) -> dict[str, Any]:
        """持久记录 SDK 恢复义务，并阻止同会话发出新 turn。

        输入是阶段、请求摘要、可能为空的 SDK turn 标识和原因
        （worker 清理、receipt、request-intent 或 terminal turn）；动作是保存指定
        原因并把 manifest 设为 ``recovery_pending``。同一 turn 的 terminal 事实比
        worker 清理优先：它可替换较弱记录，反向写入则幂等保留 terminal；其它冲突
        抛出状态错误，不猜测任何恢复步骤已经完成。
        """

        (
            validated_stage,
            validated_request_digest,
            validated_thread_policy_digest,
            validated_turn_id,
            expected_action,
        ) = _validate_codex_recovery_pending_arguments(
            stage,
            request_digest,
            thread_policy_digest,
            turn_id,
            reason,
        )
        with self.locked():
            current = self._read_manifest()
            if current is None:
                raise HarnessError(
                    "Cannot record Codex recovery before creating a Codex session",
                    kind="session_state",
                )
            existing = self._read_recovery_record()
            if (
                existing is not None
                and reason == "terminal_turn"
                and existing.get("reason") == "request_intent"
            ):
                # 旧版本只留下 request-intent 时，adapter 已经从同一 Codex
                # history 证明了终态 turn；把 session recovery 一并升级，随后
                # 的 terminal acknowledgement 才能使用同一个 turn 身份。该
                # 方法在同一可重入 session lock 内执行，并保留其补偿语义。
                return self.migrate_request_intent_recovery_to_terminal_turn(
                    stage=validated_stage,
                    request_digest=validated_request_digest,
                    thread_policy_digest=validated_thread_policy_digest,
                    turn_id=validated_turn_id,
                    legacy_recovery_request_digest=existing.get("request_digest"),
                )
            if existing is not None:
                update = _classify_codex_recovery_pending_update(
                    existing,
                    current,
                    stage=validated_stage,
                    request_digest=validated_request_digest,
                    thread_policy_digest=validated_thread_policy_digest,
                    turn_id=validated_turn_id,
                    reason=reason,
                    action=expected_action,
                )
                if update == "upgrade_terminal":
                    self._write_recovery_record(
                        {
                            **existing,
                            "reason": reason,
                            "action": expected_action,
                            "updated_at": _utc_now(),
                        }
                    )
                current["last_stage"] = validated_stage
                current["status"] = "recovery_pending"
                return self._write_atomic(current)
            now = _utc_now()
            self._write_recovery_record(
                {
                    "schema_version": _RECOVERY_RECORD_SCHEMA_VERSION,
                    "state": "recovery_pending",
                    "reason": reason,
                    "action": expected_action,
                    "stage": validated_stage,
                    "request_digest": validated_request_digest,
                    "thread_policy_digest": validated_thread_policy_digest,
                    "turn_id": validated_turn_id,
                    "turn_count": current["turn_count"],
                    "created_at": now,
                    "updated_at": now,
                }
            )
            current["last_stage"] = validated_stage
            current["status"] = "recovery_pending"
            return self._write_atomic(current)

    def acknowledge_terminal_turn_recovery(
        self,
        *,
        request_digest: str,
        thread_policy_digest: str,
        turn_id: str,
    ) -> dict[str, Any]:
        """确认上层已保存 terminal turn 事实，并清理同一恢复记录。

        输入必须匹配 ``terminal_turn`` 的请求、策略和 turn 身份。调用方应先把
        failed/interrupted 事实写入自己的权威状态；本方法随后把 manifest 恢复为
        未增加成功 turn 计数的正常状态，并删除 recovery。若首次终态回调尚未形成
        recovery 记录则幂等返回；身份不符或任一步骤写盘失败时抛错，让 adapter
        保留 terminal journal 与 request-intent 并在恢复时重放回调。
        """

        request_digest, validated_thread_policy_digest, validated_turn_id = (
            self._validate_recovery_confirmation_arguments(
                request_digest,
                thread_policy_digest,
                turn_id,
            )
        )
        with self.locked():
            current = self._read_manifest()
            recovery = self._read_recovery_record()
            if current is None:
                raise HarnessError(
                    "Codex terminal recovery session is missing",
                    kind="session_state",
                )
            if recovery is None:
                if current["status"] == "recovery_pending":
                    raise HarnessError(
                        "Codex terminal recovery record is missing",
                        kind="session_state",
                    )
                return current
            if not (
                recovery["state"] == "recovery_pending"
                and recovery["reason"] == "terminal_turn"
                and recovery["action"] == "terminal_acknowledgement_required"
                and recovery["request_digest"] == request_digest
                and recovery["thread_policy_digest"] == validated_thread_policy_digest
                and recovery["turn_id"] == validated_turn_id
                and recovery["turn_count"] == current["turn_count"]
            ):
                raise HarnessError(
                    "Codex terminal recovery identity does not match",
                    kind="session_state",
                )
            current["last_stage"] = recovery["stage"]
            current["status"] = "created" if current["turn_count"] == 0 else "completed"
            saved = self._write_atomic(current)
            self._remove_recovery_record()
            return saved

    def migrate_request_intent_recovery_to_terminal_turn(
        self,
        *,
        stage: str,
        request_digest: str,
        thread_policy_digest: str,
        turn_id: str,
        legacy_recovery_request_digest: str | None = None,
    ) -> dict[str, Any]:
        """把旧 request-intent 阻断升级为同一失败 turn 的终态恢复记录。

        输入是旧 session manifest、终态 turn 的原始请求摘要和策略摘要；动作在锁内
        严格核对 manifest/thread/成功计数，并把 ``request_intent`` 恢复记录升级为
        ``terminal_turn``。旧记录的摘要可由调用方显式传入以核对 A/B 漂移；该方法
        不删除 request-intent、terminal journal 或 Working Memory 事实，重复调用同一
        身份幂等，任意身份不一致均失败关闭。
        """

        validated_stage = _validate_text(stage, "last_stage")
        (
            validated_request_digest,
            validated_thread_policy_digest,
            validated_turn_id,
        ) = self._validate_recovery_confirmation_arguments(
            request_digest,
            thread_policy_digest,
            turn_id,
        )
        validated_legacy_digest = None
        if legacy_recovery_request_digest is not None:
            if not isinstance(legacy_recovery_request_digest, str) or _REQUEST_DIGEST.fullmatch(
                legacy_recovery_request_digest
            ) is None:
                raise HarnessError(
                    "Codex legacy recovery request digest is malformed",
                    kind="session_state",
                )
            validated_legacy_digest = legacy_recovery_request_digest
        with self.locked():
            current = self._read_manifest()
            recovery = self._read_recovery_record()
            if current is None or recovery is None:
                raise HarnessError(
                    "Codex legacy request-intent recovery record is missing",
                    kind="session_state",
                )
            if recovery["reason"] == "terminal_turn":
                expected = (
                    current["status"] in {"created", "active", "completed", "recovery_pending"}
                    and recovery["state"] == "recovery_pending"
                    and recovery["action"] == "terminal_acknowledgement_required"
                    and recovery["stage"] == validated_stage
                    and recovery["request_digest"] == validated_request_digest
                    and recovery["thread_policy_digest"] == validated_thread_policy_digest
                    and recovery["turn_id"] == validated_turn_id
                    and recovery["turn_count"] == current["turn_count"]
                )
                if expected:
                    return current
                raise HarnessError(
                    "Codex legacy terminal recovery identity does not match",
                    kind="session_state",
                )
            if current["status"] != "recovery_pending":
                raise HarnessError(
                    "Codex legacy request-intent recovery manifest is not pending",
                    kind="session_state",
                )
            if recovery["state"] != "recovery_pending" or recovery["reason"] != "request_intent":
                raise HarnessError(
                    "Codex legacy recovery reason is not request_intent",
                    kind="session_state",
                )
            if recovery["action"] != "request_intent_confirmation_required":
                raise HarnessError(
                    "Codex legacy request-intent recovery action is invalid",
                    kind="session_state",
                )
            if recovery["stage"] != validated_stage:
                raise HarnessError(
                    "Codex legacy recovery stage does not match",
                    kind="session_state",
                )
            if recovery["thread_policy_digest"] != validated_thread_policy_digest:
                raise HarnessError(
                    "Codex legacy recovery thread policy does not match",
                    kind="session_state",
                )
            if recovery["turn_id"] != "":
                raise HarnessError(
                    "Codex legacy request-intent recovery unexpectedly has a turn id",
                    kind="session_state",
                )
            if recovery["turn_count"] != current["turn_count"]:
                raise HarnessError(
                    "Codex legacy recovery turn count does not match manifest",
                    kind="session_state",
                )
            if recovery["request_digest"] != validated_request_digest:
                raise HarnessError(
                    "Codex legacy recovery digest does not match the terminal turn",
                    kind="session_state",
                )
            if validated_legacy_digest is not None and recovery["request_digest"] != validated_legacy_digest:
                raise HarnessError(
                    "Codex legacy recovery digest does not match the supplied old identity",
                    kind="session_state",
                )
            expected = {
                "schema_version": _RECOVERY_RECORD_SCHEMA_VERSION,
                "state": "recovery_pending",
                "reason": "terminal_turn",
                "action": "terminal_acknowledgement_required",
                "stage": validated_stage,
                "request_digest": validated_request_digest,
                "thread_policy_digest": validated_thread_policy_digest,
                "turn_id": validated_turn_id,
                "turn_count": current["turn_count"],
                "created_at": recovery["created_at"],
                "updated_at": recovery["updated_at"],
            }
            self._write_recovery_record(expected)
            current["last_stage"] = validated_stage
            current["status"] = "recovery_pending"
            return self._write_atomic(current)

    def confirm_worker_stopped_for_recovery(
        self,
        *,
        request_digest: str,
        thread_policy_digest: str,
        turn_id: str | None,
    ) -> dict[str, Any]:
        """在调用方确认 worker 已停止后允许提交一个新的请求。

        输入必须匹配 ``worker_cleanup`` 记录；本方法只改变后续动作要求，不能
        清除记录或直接增加 turn 计数。未确认时任何同会话新 turn 都会被拒绝。
        """

        request_digest, validated_thread_policy_digest, validated_turn_id = (
            self._validate_recovery_confirmation_arguments(
                request_digest,
                thread_policy_digest,
                turn_id,
            )
        )
        with self.locked():
            current = self._read_manifest()
            recovery = self._read_recovery_record()
            if current is None or recovery is None:
                raise HarnessError("Codex worker recovery record is missing", kind="session_state")
            if self._recovery_identity_matches(
                current,
                recovery,
                reason="worker_cleanup",
                action="explicit_new_request_required",
                request_digest=request_digest,
                thread_policy_digest=validated_thread_policy_digest,
                turn_id=validated_turn_id,
            ):
                return current
            if not self._recovery_identity_matches(
                current,
                recovery,
                reason="worker_cleanup",
                action="worker_stop_confirmation_required",
                request_digest=request_digest,
                thread_policy_digest=validated_thread_policy_digest,
                turn_id=validated_turn_id,
            ):
                raise HarnessError(
                    "Codex worker recovery identity does not match",
                    kind="session_state",
                )
            now = _utc_now()
            self._write_recovery_record(
                {
                    **recovery,
                    "action": "explicit_new_request_required",
                    "updated_at": now,
                }
            )
            return current

    def confirm_request_intent_without_new_turn(
        self,
        *,
        request_digest: str,
        thread_policy_digest: str,
        turn_id: str | None,
    ) -> dict[str, Any]:
        """确认 request-intent 没有对应的新 turn，允许提交不同的新请求。

        输入必须精确匹配 ``request_intent`` 恢复记录；动作是把动作字段改为
        ``explicit_new_request_required``，保留原记录和 manifest 计数。副作用是
        同摘要重试仍会被 runtime 拒绝；确认前或身份不匹配时抛出
        ``HarnessError(kind="session_state")``，绝不自行删除记录或重发模型。
        """

        request_digest, validated_thread_policy_digest, validated_turn_id = (
            self._validate_recovery_confirmation_arguments(
                request_digest,
                thread_policy_digest,
                turn_id,
            )
        )
        with self.locked():
            current = self._read_manifest()
            recovery = self._read_recovery_record()
            if current is None or recovery is None:
                raise HarnessError("Codex request-intent recovery record is missing", kind="session_state")
            if self._recovery_identity_matches(
                current,
                recovery,
                reason="request_intent",
                action="explicit_new_request_required",
                request_digest=request_digest,
                thread_policy_digest=validated_thread_policy_digest,
                turn_id=validated_turn_id,
            ):
                return current
            if not self._recovery_identity_matches(
                current,
                recovery,
                reason="request_intent",
                action="request_intent_confirmation_required",
                request_digest=request_digest,
                thread_policy_digest=validated_thread_policy_digest,
                turn_id=validated_turn_id,
            ):
                raise HarnessError(
                    "Codex request-intent recovery identity does not match",
                    kind="session_state",
                )
            self._write_recovery_record(
                {
                    **recovery,
                    "action": "explicit_new_request_required",
                    "updated_at": _utc_now(),
                }
            )
            return current

    def clear_recovery_after_validated_turn(
        self,
        *,
        stage: str,
        thread_policy_digest: str,
        status: str = "completed",
    ) -> dict[str, Any]:
        """在新响应通过结构合同后清除旧恢复记录并保存成功状态。

        输入是已验证新 turn 的阶段和最终 manifest 状态；动作不增加计数，只更新
        manifest 后删除旧恢复记录。任一步骤失败都会抛出状态错误，调用方不能把
        未清理的记录当作成功返回。
        """

        validated_stage = _validate_text(stage, "last_stage")
        validated_thread_policy_digest = _validate_thread_policy_digest(thread_policy_digest)
        validated_status = _validate_text(status, "status")
        with self.locked():
            current = self._read_manifest()
            if current is None:
                raise HarnessError(
                    "Cannot clear recovery before creating a Codex session",
                    kind="session_state",
                )
            recovery = self._read_recovery_record()
            if recovery is None:
                return current
            validated = self._read_validated_turn()
            if validated is None:
                raise HarnessError(
                    "Codex recovery cannot be cleared without validated turn evidence",
                    kind="session_state",
                )
            if not (
                validated["thread_id"] == current["thread_id"]
                and validated["stage"] == validated_stage
                and validated["thread_policy_digest"] == validated_thread_policy_digest
                and recovery["thread_policy_digest"] == validated_thread_policy_digest
                and validated["turn_count"] == current["turn_count"]
                and validated["turn_count"] > recovery["turn_count"]
            ):
                raise HarnessError(
                    "Codex validated turn evidence does not match recovery state",
                    kind="session_state",
                )
            current["last_stage"] = validated_stage
            current["status"] = validated_status
            saved = self._write_atomic(current)
            self._remove_recovery_record()
            self._remove_auxiliary_record(self.validated_turn_path, name="validated turn")
            return saved

    def clear_request_intent_recovery_after_replayed_receipt(
        self,
        *,
        stage: str,
        thread_id: str,
        receipt_turn_count_before: int,
        thread_policy_digest: str,
    ) -> dict[str, Any]:
        """在旧成功回合完成严格重放后清除后续失败 Resume 的恢复阻断。

        输入来自 adapter 已核对并重放的 completed receipt：同一 thread、回执前
        成功计数和策略摘要。动作只接受当前成功计数正好等于回执计数加一，且
        recovery 是同一计数上的 ``request_intent_confirmation_required``；随后把
        manifest 恢复为 completed 并删除该后续失败请求的 recovery。该方法不增加
        turn_count，也不删除 completed receipt；身份、计数、恢复原因或 poison
        任一不一致时失败关闭。
        """

        validated_stage = _validate_text(stage, "last_stage")
        validated_thread_id = _validate_text(thread_id, "thread_id")
        validated_policy = _validate_thread_policy_digest(thread_policy_digest)
        if (
            isinstance(receipt_turn_count_before, bool)
            or not isinstance(receipt_turn_count_before, int)
            or receipt_turn_count_before < 0
        ):
            raise HarnessError(
                "Codex replay receipt successful count is invalid",
                kind="session_state",
            )
        with self.locked():
            current = self._read_manifest()
            recovery = self._read_recovery_record()
            poison = self._read_recovery_poison()
            if current is None or recovery is None:
                raise HarnessError(
                    "Codex replay request-intent recovery record is missing",
                    kind="session_state",
                )
            if poison is not None:
                raise HarnessError(
                    "Codex recovery poison marker blocks replay cleanup",
                    kind="recovery_pending",
                )
            expected_turn_count = receipt_turn_count_before + 1
            if not (
                current.get("thread_id") == validated_thread_id
                and current.get("thread_policy_digest") == validated_policy
                and current.get("turn_count") == expected_turn_count
                and current.get("status") == "recovery_pending"
                and recovery.get("state") == "recovery_pending"
                and recovery.get("reason") == "request_intent"
                and recovery.get("action")
                == "request_intent_confirmation_required"
                and recovery.get("thread_policy_digest") == validated_policy
                and recovery.get("turn_count") == expected_turn_count
                and recovery.get("turn_id") == ""
            ):
                raise HarnessError(
                    "Codex replay request-intent recovery identity does not match",
                    kind="session_state",
                )
            current["last_stage"] = validated_stage
            current["status"] = "completed"
            saved = self._write_atomic(current)
            self._remove_recovery_record()
            return saved


__all__ = ["PlannerSessionStore", "CodexSessionOwner", "CodexSessionStore"]
