"""Execution job 的 scope 私有身份注册表。

注册表只保存可信服务内部的作业路由和幂等身份，不作为 MCP 返回值。所有分配、重放和
状态更新都在 workspace 私有锁内完成；短作业编号与已有目录、永久占用表发生冲突
时重试，旧的 32 位十六进制作业编号可以从持久 receipt 导入恢复。
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import secrets
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from minichem_toolbox.idempotency import validate_submission_key
from minichem_toolbox.scope import current_scope

from .workspace import ensure_private_directory


SHORT_JOB_ID_PATTERN = re.compile(r"^job_[0-9a-f]{8}$")
LEGACY_JOB_ID_PATTERN = re.compile(r"^job_[0-9a-f]{32}$")
EXECUTION_JOB_ID_PATTERN = re.compile(r"^job_(?:[0-9a-f]{8}|[0-9a-f]{32})$")
REGISTRY_SCHEMA_VERSION = 1
REGISTRY_DIRECTORY = Path("_tool_artifacts") / "execution_job_registry"
REGISTRY_FILENAME = "registry.json"
REGISTRY_LOCK_FILENAME = "registry.lock"
GLOBAL_ALLOCATIONS_FILENAME = "allocations.json"
TERMINAL_JOB_STATES = frozenset({"success", "failed", "timeout", "cancelled"})
_REGISTRY_PHASES = frozenset(
    {
        "reserved",
        "preparing",
        "accepted",
        "queued",
        "running",
        "success",
        "failed",
        "timeout",
        "cancelled",
        "collected",
        "unknown",
        "legacy",
    }
)
_IMMUTABLE_RECORD_FIELDS = frozenset(
    {
        "job_id",
        "job_type",
        "operation_id",
        "operation_fingerprint",
        "caller_request_fingerprint",
        "submission_key",
        "scope_id",
        "scope_workspace",
    }
)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_LOCK_FILENAME = re.compile(r"^\.[0-9a-f]{64}\.lock$")


class ExecutionJobRegistryError(RuntimeError):
    """注册表、workspace 或持久身份无法安全核对。"""


class ExecutionJobRegistryConflict(ExecutionJobRegistryError):
    """同一 operation 或 job_id 绑定了不同的可信请求身份。"""

    def __init__(self, message: str, *, record: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.record = dict(record) if isinstance(record, Mapping) else None


@dataclass(frozen=True)
class ExecutionJobReservation:
    """一次 job_id 预留结果；reused 表示同一 operation 已有持久身份。"""

    job_id: str
    reused: bool
    record: dict[str, Any]


def is_execution_job_id(value: Any) -> bool:
    """判断一个值是否是新短 ID 或旧 32 hex ID。"""

    return isinstance(value, str) and EXECUTION_JOB_ID_PATTERN.fullmatch(value) is not None


def is_legacy_execution_job_id(value: Any) -> bool:
    """判断一个值是否是旧版 32 hex 作业 ID。"""

    return isinstance(value, str) and LEGACY_JOB_ID_PATTERN.fullmatch(value) is not None


def validate_execution_job_id(value: Any) -> str:
    """校验模型或持久记录中的作业编号，拒绝路径和未知长度。"""

    if not is_execution_job_id(value):
        raise ValueError("invalid execution job id")
    return str(value)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_operation_id(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value or len(value) > 512:
        raise ExecutionJobRegistryError("execution operation identity is invalid")
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ExecutionJobRegistryError("execution operation identity is not ASCII") from exc
    if any(ord(char) < 0x20 for char in value):
        raise ExecutionJobRegistryError("execution operation identity contains control characters")
    return value


def _safe_fingerprint(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ExecutionJobRegistryError(f"{name} is not a lowercase SHA-256")
    return value


def _safe_job_type(value: Any) -> str:
    if not isinstance(value, str) or not value or len(value) > 128:
        raise ExecutionJobRegistryError("execution job type is invalid")
    if any(ord(char) < 0x20 for char in value):
        raise ExecutionJobRegistryError("execution job type contains control characters")
    return value


def _read_mapping(path: Path) -> dict[str, Any] | None:
    """读取普通 JSON object；符号链接、损坏或非 object 均返回 None。"""

    if path.is_symlink() or not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError):
        return None
    return dict(value) if isinstance(value, Mapping) else None


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """在同一私有目录内原子写入 JSON 控制记录。"""

    directory = path.parent
    ensure_private_directory(directory, create=True)
    temporary = directory / f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    descriptor = os.open(
        temporary,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        payload = (json.dumps(dict(value), ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("execution registry atomic write made no progress")
            offset += written
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o600)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    directory_descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


class ExecutionJobRegistry:
    """管理一个 workspace/scope 私有的 execution job 身份注册表。"""

    def __init__(
        self,
        workspace: str | os.PathLike[str] | Path | None = None,
        scope_id: str | None = None,
        *,
        job_id_factory: Callable[[], str] | None = None,
    ) -> None:
        scoped = current_scope()
        selected_workspace = Path(workspace or (scoped.workspace if scoped is not None else "."))
        self.workspace = selected_workspace.expanduser().resolve()
        self.scope_id = str(scope_id if scope_id is not None else (scoped.scope_id if scoped else ""))
        if scoped is not None and self.scope_id != scoped.scope_id:
            raise ExecutionJobRegistryError("execution registry scope does not match current scope")
        if scoped is not None and self.workspace != Path(scoped.workspace).expanduser().resolve():
            raise ExecutionJobRegistryError("execution registry workspace does not match current scope")
        if not self.scope_id and scoped is not None:
            raise ExecutionJobRegistryError("execution registry scope is empty")
        self._job_id_factory = job_id_factory or (lambda: f"job_{secrets.token_hex(4)}")

    @property
    def root(self) -> Path:
        """返回注册表私有目录，不向 MCP 响应公开。"""

        scope_token = hashlib.sha256(self.scope_id.encode("utf-8")).hexdigest()[:32]
        return ensure_private_directory(
            self.workspace / REGISTRY_DIRECTORY / scope_token,
            create=True,
        )

    @property
    def _workspace_root(self) -> Path:
        """返回所有 scope 共享的分配锁目录。"""

        return ensure_private_directory(self.workspace / REGISTRY_DIRECTORY, create=True)

    @property
    def path(self) -> Path:
        """返回注册表 JSON 的内部路径。"""

        return self.root / REGISTRY_FILENAME

    @contextmanager
    def _lock(self) -> Iterator[None]:
        """在 scope 私有锁内执行一次完整的读改写。"""

        root = self._workspace_root
        path = root / REGISTRY_LOCK_FILENAME
        descriptor = os.open(
            path,
            os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            os.fchmod(descriptor, 0o600)
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _load_global_allocations(self) -> dict[str, dict[str, str]]:
        """读取 workspace 级 job_id 占用表，协调不同 scope 的短 ID 分配。"""

        path = self._workspace_root / GLOBAL_ALLOCATIONS_FILENAME
        if not path.exists():
            return {}
        raw = _read_mapping(path)
        if raw is None or raw.get("schema_version") != REGISTRY_SCHEMA_VERSION:
            raise ExecutionJobRegistryError("execution job allocation table is invalid")
        if raw.get("scope_workspace") != str(self.workspace):
            raise ExecutionJobRegistryError("execution job allocation workspace is invalid")
        jobs = raw.get("jobs")
        if not isinstance(jobs, Mapping):
            raise ExecutionJobRegistryError("execution job allocation entries are invalid")
        allocations: dict[str, dict[str, str]] = {}
        for job_id, value in jobs.items():
            normalized = validate_execution_job_id(job_id)
            if not isinstance(value, Mapping) or not isinstance(value.get("scope_id"), str):
                raise ExecutionJobRegistryError("execution job allocation identity is invalid")
            allocations[normalized] = {"scope_id": value["scope_id"]}
        return allocations

    def _save_global_allocations(self, allocations: Mapping[str, Mapping[str, str]]) -> None:
        """原子保存 workspace 级永久 job_id 占用表。"""

        _atomic_json(
            self._workspace_root / GLOBAL_ALLOCATIONS_FILENAME,
            {
                "schema_version": REGISTRY_SCHEMA_VERSION,
                "scope_workspace": str(self.workspace),
                "jobs": {job_id: dict(value) for job_id, value in allocations.items()},
                "updated_at": _now(),
            },
        )

    def _sync_global_allocations(self, records: Mapping[str, Mapping[str, Any]]) -> None:
        """把当前 scope 的已有记录登记到 workspace 级永久占用表。"""

        allocations = self._load_global_allocations()
        changed = False
        for job_id in records:
            normalized = validate_execution_job_id(job_id)
            existing = allocations.get(normalized)
            if existing is not None and existing.get("scope_id") != self.scope_id:
                raise ExecutionJobRegistryConflict("job_id is occupied by another scope")
            if existing is None:
                allocations[normalized] = {"scope_id": self.scope_id}
                changed = True
        if changed:
            self._save_global_allocations(allocations)

    def _empty_state(self) -> dict[str, Any]:
        return {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "scope_id": self.scope_id,
            "scope_workspace": str(self.workspace),
            "jobs": {},
            "tombstones": {},
            "updated_at": _now(),
        }

    def _validate_record(self, job_id: str, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, Mapping):
            raise ExecutionJobRegistryError("execution registry record is not an object")
        record = dict(raw)
        if record.get("job_id") != job_id or not is_execution_job_id(job_id):
            raise ExecutionJobRegistryError("execution registry job identity is invalid")
        if record.get("scope_id") != self.scope_id or record.get("scope_workspace") != str(self.workspace):
            raise ExecutionJobRegistryError("execution registry scope identity is invalid")
        _safe_job_type(record.get("job_type"))
        _safe_operation_id(record.get("operation_id"))
        _safe_fingerprint(record.get("operation_fingerprint"), "operation_fingerprint")
        _safe_fingerprint(record.get("caller_request_fingerprint"), "caller_request_fingerprint")
        submission_key = record.get("submission_key")
        if submission_key is not None:
            try:
                validate_submission_key(submission_key)
            except ValueError as exc:
                raise ExecutionJobRegistryError("execution registry submission key is invalid") from exc
        if record.get("phase") not in _REGISTRY_PHASES:
            raise ExecutionJobRegistryError("execution registry phase is invalid")
        return record

    def _load(self) -> dict[str, Any]:
        path = self.path
        if not path.exists():
            return self._empty_state()
        if path.is_symlink() or not path.is_file():
            raise ExecutionJobRegistryError("execution registry file is invalid")
        raw = _read_mapping(path)
        if raw is None:
            raise ExecutionJobRegistryError("execution registry cannot be read")
        if raw.get("schema_version") != REGISTRY_SCHEMA_VERSION:
            raise ExecutionJobRegistryError("execution registry schema is invalid")
        if raw.get("scope_id") != self.scope_id or raw.get("scope_workspace") != str(self.workspace):
            raise ExecutionJobRegistryError("execution registry scope does not match")
        jobs = raw.get("jobs")
        tombstones = raw.get("tombstones")
        if not isinstance(jobs, Mapping) or not isinstance(tombstones, Mapping):
            raise ExecutionJobRegistryError("execution registry collections are invalid")
        normalized_jobs: dict[str, dict[str, Any]] = {}
        for job_id, value in jobs.items():
            normalized_jobs[validate_execution_job_id(job_id)] = self._validate_record(job_id, value)
        normalized_tombstones: dict[str, dict[str, Any]] = {}
        for job_id, value in tombstones.items():
            normalized_id = validate_execution_job_id(job_id)
            if not isinstance(value, Mapping):
                raise ExecutionJobRegistryError("execution registry tombstone is invalid")
            normalized_tombstones[normalized_id] = dict(value)
        return {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "scope_id": self.scope_id,
            "scope_workspace": str(self.workspace),
            "jobs": normalized_jobs,
            "tombstones": normalized_tombstones,
            "updated_at": raw.get("updated_at") or _now(),
        }

    def _save(self, state: dict[str, Any]) -> None:
        state["updated_at"] = _now()
        _atomic_json(self.path, state)

    def _job_root(self) -> Path:
        """取得 execution_jobs 目录并核对其中只包含合法作业或 operation lock。"""

        root = ensure_private_directory(self.workspace / "outputs" / "execution_jobs", create=True)
        for child in root.iterdir():
            if _LOCK_FILENAME.fullmatch(child.name):
                if child.is_symlink() or not child.is_file():
                    raise ExecutionJobRegistryError("execution job operation lock is invalid")
                continue
            if not is_execution_job_id(child.name) or child.is_symlink() or not child.is_dir():
                # execution_jobs 根目录的未知项可能改变路径边界，必须对 workspace
                # 整体 fail-closed；合法 job 目录内与目标 operation 无关的损坏 JSON
                # 则由 _record_from_directory 按 operation 过滤后跳过。
                raise ExecutionJobRegistryError("execution job directory contains an unknown entry")
        return root

    def _existing_job_ids(self) -> set[str]:
        return {child.name for child in self._job_root().iterdir() if child.is_dir()}

    def _record_from_directory(
        self,
        job_id: str,
        directory: Path,
        *,
        target_operation_id: str | None = None,
    ) -> dict[str, Any] | None:
        """从旧作业的可信交集读取可导入 registry 的内部身份。"""

        values: list[dict[str, Any]] = []
        unreadable = False
        for filename in ("status.json", "request.json", "operation_receipt.json", "supervisor_spec.json"):
            path = directory / filename
            if not path.exists():
                continue
            value = _read_mapping(path)
            if value is None:
                unreadable = True
                continue
            values.append(value)
        if not values:
            return None
        operation_values = {value.get("operation_id") for value in values if value.get("operation_id") is not None}
        # 旧 workspace 可能残留与本次 operation 无关的损坏目录。只有可读记录已经
        # 指向目标 operation 时，损坏的同目录身份才阻断恢复；否则跳过该目录，避免
        # 一份无关历史垃圾把当前 operation 的唯一 job_id 恢复路径整体锁死。
        if target_operation_id is not None and target_operation_id not in operation_values:
            return None
        if unreadable:
            raise ExecutionJobRegistryError("matching existing execution job identity is unreadable")
        for value in values:
            if value.get("job_id") not in {None, job_id}:
                raise ExecutionJobRegistryError("existing execution job id is inconsistent")
            if value.get("scope_id") != self.scope_id or value.get("scope_workspace") != str(self.workspace):
                # 共享物理 workspace 中另一个 scope 的目录只占用该 ID，不属于本次
                # operation 恢复候选；不能把它误导入当前 registry。
                return None
        fingerprint_values = {
            value.get("operation_fingerprint") or value.get("fingerprint")
            for value in values
            if value.get("operation_fingerprint") is not None or value.get("fingerprint") is not None
        }
        job_type_values = {value.get("job_type") for value in values if value.get("job_type") is not None}
        submission_values = {value.get("submission_key") for value in values if value.get("submission_key") is not None}
        caller_fingerprint_values = {
            value.get("caller_request_fingerprint")
            for value in values
            if value.get("caller_request_fingerprint") is not None
        }
        if (
            len(operation_values) > 1
            or len(fingerprint_values) > 1
            or len(job_type_values) > 1
            or len(submission_values) > 1
            or len(caller_fingerprint_values) > 1
        ):
            raise ExecutionJobRegistryConflict("existing execution job identity records conflict")
        operation_id = next(iter(operation_values), None)
        if operation_id is None:
            return None
        fingerprint = next(iter(fingerprint_values), None)
        job_type = next(iter(job_type_values), None)
        if job_type is None or fingerprint is None:
            raise ExecutionJobRegistryError("existing execution job identity is incomplete")
        caller_fingerprint = next(iter(caller_fingerprint_values), None)
        status_value = next((value.get("status") for value in values if value.get("status") is not None), None)
        receipt_phase = next((value.get("phase") for value in values if value.get("phase") is not None), None)
        phase = status_value if status_value in _REGISTRY_PHASES else receipt_phase
        if phase not in _REGISTRY_PHASES:
            phase = "legacy"
        return {
            "job_id": job_id,
            "job_type": _safe_job_type(job_type),
            "operation_id": _safe_operation_id(operation_id),
            "operation_fingerprint": _safe_fingerprint(fingerprint, "operation_fingerprint"),
            "caller_request_fingerprint": _safe_fingerprint(
                caller_fingerprint,
                "caller_request_fingerprint",
            ),
            "submission_key": next(iter(submission_values), None),
            "scope_id": self.scope_id,
            "scope_workspace": str(self.workspace),
            "phase": phase,
            "created_at": next((value.get("created_at") for value in values if value.get("created_at")), _now()),
            "updated_at": _now(),
        }

    def _find_existing_operation(
        self,
        operation_id: str,
        operation_fingerprint: str,
        *,
        job_type: str,
        submission_key: str | None,
    ) -> dict[str, Any] | None:
        """扫描 registry 缺失的旧目录，恢复同 operation 的原 job_id。"""

        for job_id in sorted(self._existing_job_ids()):
            directory_record = self._record_from_directory(
                job_id,
                self._job_root() / job_id,
                target_operation_id=operation_id,
            )
            if directory_record is None or directory_record.get("operation_id") != operation_id:
                continue
            self._ensure_matching_identity(
                directory_record,
                job_type=job_type,
                operation_id=operation_id,
                operation_fingerprint=operation_fingerprint,
                submission_key=submission_key,
            )
            return directory_record
        return None

    @staticmethod
    def _ensure_matching_identity(
        record: Mapping[str, Any],
        *,
        job_type: str,
        operation_id: str | None,
        operation_fingerprint: str | None,
        submission_key: str | None,
    ) -> None:
        if (
            record.get("job_type") != job_type
            or record.get("operation_id") != operation_id
            or record.get("operation_fingerprint") != operation_fingerprint
            or record.get("submission_key") != submission_key
        ):
            raise ExecutionJobRegistryConflict("same operation is bound to a different execution request", record=record)

    def _new_candidate(self, state: Mapping[str, Any], preferred_job_id: str | None) -> str:
        occupied = (
            set(state["jobs"])
            | set(state["tombstones"])
            | set(self._load_global_allocations())
            | self._existing_job_ids()
        )
        if preferred_job_id is not None:
            selected = validate_execution_job_id(preferred_job_id)
            if selected in occupied:
                raise ExecutionJobRegistryConflict("requested job_id is already occupied")
            return selected
        for _attempt in range(128):
            selected = validate_execution_job_id(self._job_id_factory())
            if not SHORT_JOB_ID_PATTERN.fullmatch(selected):
                raise ExecutionJobRegistryError("new execution job id factory returned a non-short id")
            if selected not in occupied:
                return selected
        raise ExecutionJobRegistryError("unable to allocate a collision-free execution job id")

    def reserve(
        self,
        *,
        job_type: str,
        operation_id: str | None,
        operation_fingerprint: str | None,
        submission_key: str | None,
        caller_request_fingerprint: str | None = None,
        preferred_job_id: str | None = None,
    ) -> ExecutionJobReservation:
        """在 registry 锁内预留或幂等复用一个 job_id。"""

        normalized_type = _safe_job_type(job_type)
        normalized_operation = _safe_operation_id(operation_id)
        normalized_fingerprint = _safe_fingerprint(operation_fingerprint, "operation_fingerprint")
        normalized_caller = _safe_fingerprint(caller_request_fingerprint, "caller_request_fingerprint")
        if normalized_operation is not None and normalized_fingerprint is None:
            raise ExecutionJobRegistryError(
                "operation_fingerprint is required with operation_id"
            )
        normalized_submission = None
        if submission_key is not None:
            try:
                normalized_submission = validate_submission_key(submission_key)
            except ValueError as exc:
                raise ExecutionJobRegistryError("execution registry submission key is invalid") from exc
        with self._lock():
            state = self._load()
            self._sync_global_allocations(state["jobs"])
            if normalized_operation is not None:
                for record in state["jobs"].values():
                    if record.get("operation_id") != normalized_operation:
                        continue
                    self._ensure_matching_identity(
                        record,
                        job_type=normalized_type,
                        operation_id=normalized_operation,
                        operation_fingerprint=normalized_fingerprint,
                        submission_key=normalized_submission,
                    )
                    if normalized_caller is not None and record.get("caller_request_fingerprint") not in {
                        None,
                        normalized_caller,
                    }:
                        raise ExecutionJobRegistryConflict(
                            "same operation caller fingerprint conflicts", record=record
                        )
                    return ExecutionJobReservation(record["job_id"], True, deepcopy(record))
                legacy = self._find_existing_operation(
                    normalized_operation,
                    normalized_fingerprint or "",
                    job_type=normalized_type,
                    submission_key=normalized_submission,
                )
                if legacy is not None:
                    state["jobs"][legacy["job_id"]] = legacy
                    self._sync_global_allocations({legacy["job_id"]: legacy})
                    self._save(state)
                    return ExecutionJobReservation(legacy["job_id"], True, deepcopy(legacy))
            selected = self._new_candidate(state, preferred_job_id)
            record = {
                "job_id": selected,
                "job_type": normalized_type,
                "operation_id": normalized_operation,
                "operation_fingerprint": normalized_fingerprint,
                "caller_request_fingerprint": normalized_caller,
                "submission_key": normalized_submission,
                "scope_id": self.scope_id,
                "scope_workspace": str(self.workspace),
                "phase": "reserved",
                "created_at": _now(),
                "updated_at": _now(),
            }
            state["jobs"][selected] = record
            self._sync_global_allocations({selected: record})
            self._save(state)
            return ExecutionJobReservation(selected, False, deepcopy(record))

    def get(self, job_id: str) -> dict[str, Any] | None:
        """读取当前 scope 的可信内部记录；不存在时返回 None。"""

        normalized = validate_execution_job_id(job_id)
        with self._lock():
            state = self._load()
            value = state["jobs"].get(normalized)
            return deepcopy(value) if isinstance(value, Mapping) else None

    def list_records(self) -> list[dict[str, Any]]:
        """按稳定 job_id 顺序读取当前 scope 的全部可信 job 记录。

        输入不需要额外筛选条件；动作只读取当前 scope 的中央 registry，不返回跨
        scope 记录或 registry 内部路径。返回值是深拷贝，调用方可以用持久 status
        文件逐项核对真实状态，而不会修改注册表；注册表损坏时抛出统一错误。
        """

        with self._lock():
            state = self._load()
            return [
                deepcopy(state["jobs"][job_id])
                for job_id in sorted(state["jobs"])
            ]

    def mark(self, job_id: str, *, phase: str, **fields: Any) -> dict[str, Any]:
        """原子更新 job phase；完整 collected 记录永久保留以防止 ID 重用。"""

        normalized = validate_execution_job_id(job_id)
        if phase not in _REGISTRY_PHASES:
            raise ValueError("invalid execution registry phase")
        attempted_identity_updates = _IMMUTABLE_RECORD_FIELDS.intersection(fields)
        if attempted_identity_updates:
            names = ", ".join(sorted(attempted_identity_updates))
            raise ExecutionJobRegistryConflict(
                f"execution job identity fields are immutable: {names}"
            )
        with self._lock():
            state = self._load()
            record = state["jobs"].get(normalized)
            if not isinstance(record, Mapping):
                raise ExecutionJobRegistryError("execution job is not registered")
            updated = dict(record)
            updated.update(fields)
            updated["phase"] = phase
            updated["updated_at"] = _now()
            validated = self._validate_record(normalized, updated)
            state["jobs"][normalized] = validated
            self._save(state)
            return deepcopy(validated)


__all__ = [
    "EXECUTION_JOB_ID_PATTERN",
    "ExecutionJobRegistry",
    "ExecutionJobRegistryConflict",
    "ExecutionJobRegistryError",
    "ExecutionJobReservation",
    "LEGACY_JOB_ID_PATTERN",
    "SHORT_JOB_ID_PATTERN",
    "is_execution_job_id",
    "is_legacy_execution_job_id",
    "validate_execution_job_id",
]
