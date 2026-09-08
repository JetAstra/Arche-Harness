"""investigation_v2 Controller 的独立运行时 lease。

这个模块只保存并核对 Controller 的进程观察信息。它不打开 Working Memory，
不追加科学事件，也不替代 Working Memory 的 sequence CAS。lease 记录使用短时
文件锁保护读改写；不能把该文件锁在整个任务期间持有，否则接管进程无法检查
前一持有者是否已经过期。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import stat
import tempfile
import threading
import time
from typing import Any


LEASE_SCHEMA_VERSION = 1
DEFAULT_LEASE_TTL_SECONDS = 60.0

_LOCK_REGISTRY_GUARD = threading.Lock()
_LOCK_REGISTRY: dict[str, threading.RLock] = {}


class ControllerLeaseError(RuntimeError):
    """Controller lease 无法安全使用。"""


class ControllerLeaseBusy(ControllerLeaseError):
    """已有未过期的其他 Controller lease。"""


class ControllerLeaseLost(ControllerLeaseError):
    """当前 holder 或 fencing token 已经失效。"""


class ControllerLeaseIntegrityError(ControllerLeaseError, ValueError):
    """lease 文件身份、结构或安全属性不可信。"""


# 允许调用方使用更直观的 Error 后缀名称；异常仍只有一套语义。
ControllerLeaseBusyError = ControllerLeaseBusy
ControllerLeaseLostError = ControllerLeaseLost
ControllerLeaseIntegrityConflict = ControllerLeaseIntegrityError
LeaseBusy = ControllerLeaseBusy
LeaseLost = ControllerLeaseLost


def _require_text(value: Any, field_name: str) -> str:
    """校验 lease 身份字段。"""

    if not isinstance(value, str) or not value.strip():
        raise ControllerLeaseIntegrityError(f"{field_name} 必须是非空字符串")
    return value.strip()


def _require_ttl(value: Any) -> float:
    """校验续期时长必须是有限正数。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("lease ttl 必须是有限正数")
    ttl = float(value)
    if not math.isfinite(ttl) or ttl <= 0:
        raise ValueError("lease ttl 必须是有限正数")
    return ttl


def _clock_seconds(clock: Any) -> float:
    """把注入时钟转换为可比较的 Unix 秒数。"""

    if callable(clock):
        value = clock()
    elif hasattr(clock, "time") and callable(clock.time):
        value = clock.time()
    elif hasattr(clock, "now"):
        value = clock.now() if callable(clock.now) else clock.now
    else:
        raise TypeError("clock 必须可调用或提供 time()/now()")
    if isinstance(value, datetime):
        value = value.timestamp()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("clock 必须返回 Unix 秒数或 datetime")
    seconds = float(value)
    if not math.isfinite(seconds):
        raise ValueError("clock 返回值必须是有限数")
    return seconds


def _thread_lock(path: Path) -> threading.RLock:
    """返回当前进程内同一路径共享的锁。"""

    key = str(path)
    with _LOCK_REGISTRY_GUARD:
        lock = _LOCK_REGISTRY.get(key)
        if lock is None:
            lock = threading.RLock()
            _LOCK_REGISTRY[key] = lock
        return lock


def _ensure_private_parent(path: Path) -> None:
    """创建 lease 父目录并拒绝符号链接或非目录父链。"""

    parent = path.parent
    missing: list[Path] = []
    cursor = parent
    while not cursor.exists() and not cursor.is_symlink():
        missing.append(cursor)
        next_cursor = cursor.parent
        if next_cursor == cursor:
            break
        cursor = next_cursor
    while True:
        if cursor.is_symlink() or (cursor.exists() and not cursor.is_dir()):
            raise ControllerLeaseIntegrityError("lease 父路径必须是普通目录")
        next_cursor = cursor.parent
        if next_cursor == cursor:
            break
        cursor = next_cursor
    for directory in reversed(missing):
        try:
            directory.mkdir(mode=0o700)
        except FileExistsError:
            pass
        if directory.is_symlink() or not directory.is_dir():
            raise ControllerLeaseIntegrityError("lease 父路径必须是普通目录")
        try:
            os.chmod(directory, 0o700)
        except OSError as exc:
            raise ControllerLeaseIntegrityError("无法设置 lease 父目录权限") from exc


def _open_lock_file(path: Path) -> int:
    """以不跟随符号链接的方式打开 lease 临界区锁文件。"""

    flags = os.O_RDWR | os.O_CREAT
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise ControllerLeaseIntegrityError("无法打开 lease 锁文件") from exc
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise ControllerLeaseIntegrityError("lease 锁文件必须是普通文件")
        if stat.S_IMODE(file_stat.st_mode) != 0o600:
            raise ControllerLeaseIntegrityError("lease 锁文件必须是 0600 权限")
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


@contextmanager
def _record_lock(path: Path):
    """在短临界区内独占 lease 记录。"""

    _ensure_private_parent(path)
    thread_lock = _thread_lock(path)
    with thread_lock:
        descriptor = _open_lock_file(path)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


def _validate_timestamp(value: Any, field_name: str) -> float:
    """校验 lease 记录中的时间字段。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ControllerLeaseIntegrityError(f"{field_name} 必须是有限数")
    timestamp = float(value)
    if not math.isfinite(timestamp):
        raise ControllerLeaseIntegrityError(f"{field_name} 必须是有限数")
    return timestamp


@dataclass(frozen=True)
class ControllerLeaseRecord:
    """持久化的 Controller lease 记录。"""

    research_task_id: str
    holder_id: str
    fencing_token: int
    renewed_at: float
    expires_at: float

    def __post_init__(self) -> None:
        """保证记录本身可以安全写回 JSON。"""

        object.__setattr__(
            self,
            "research_task_id",
            _require_text(self.research_task_id, "research_task_id"),
        )
        object.__setattr__(self, "holder_id", _require_text(self.holder_id, "holder_id"))
        if (
            isinstance(self.fencing_token, bool)
            or not isinstance(self.fencing_token, int)
            or self.fencing_token < 1
        ):
            raise ControllerLeaseIntegrityError("fencing_token 必须是正整数")
        renewed_at = _validate_timestamp(self.renewed_at, "renewed_at")
        expires_at = _validate_timestamp(self.expires_at, "expires_at")
        if expires_at < renewed_at:
            raise ControllerLeaseIntegrityError("expires_at 不能早于 renewed_at")
        object.__setattr__(self, "renewed_at", renewed_at)
        object.__setattr__(self, "expires_at", expires_at)

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        expected_research_task_id: str | None = None,
    ) -> "ControllerLeaseRecord":
        """从 canonical JSON 对象读取并核对 research_task 身份。"""

        if not isinstance(value, Mapping):
            raise ControllerLeaseIntegrityError("lease JSON 必须是对象")
        if value.get("schema_version") != LEASE_SCHEMA_VERSION:
            raise ControllerLeaseIntegrityError("lease schema_version 非法")
        if set(value) != {
            "schema_version",
            "research_task_id",
            "holder_id",
            "fencing_token",
            "renewed_at",
            "expires_at",
        }:
            raise ControllerLeaseIntegrityError("lease JSON 字段合同不匹配")
        research_task_id = _require_text(
            value.get("research_task_id"), "research_task_id"
        )
        if (
            expected_research_task_id is not None
            and research_task_id != expected_research_task_id
        ):
            raise ControllerLeaseIntegrityError("lease research_task_id 不一致")
        token = value.get("fencing_token")
        if isinstance(token, bool) or not isinstance(token, int) or token < 1:
            raise ControllerLeaseIntegrityError("fencing_token 必须是正整数")
        return cls(
            research_task_id=research_task_id,
            holder_id=_require_text(value.get("holder_id"), "holder_id"),
            fencing_token=token,
            renewed_at=_validate_timestamp(value.get("renewed_at"), "renewed_at"),
            expires_at=_validate_timestamp(value.get("expires_at"), "expires_at"),
        )

    @classmethod
    def from_schema_v10_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        expected_research_task_id: str | None = None,
    ) -> "ControllerLeaseRecord":
        """只读转换旧 lease JSON；活动写入始终使用 canonical 字段。"""

        if not isinstance(value, Mapping):
            raise ControllerLeaseIntegrityError("schema-v10 lease JSON 必须是对象")
        legacy = dict(value)
        if "research_task_id" not in legacy and "task_id" in legacy:
            legacy["research_task_id"] = legacy.pop("task_id")
        return cls.from_mapping(
            legacy,
            expected_research_task_id=expected_research_task_id,
        )

    def to_dict(self) -> dict[str, Any]:
        """返回可持久化的 lease 记录。"""

        return {
            "schema_version": LEASE_SCHEMA_VERSION,
            "research_task_id": self.research_task_id,
            "holder_id": self.holder_id,
            "fencing_token": self.fencing_token,
            "renewed_at": self.renewed_at,
            "expires_at": self.expires_at,
        }

    def is_active(self, now: float) -> bool:
        """判断在给定时间点 lease 是否仍有效；到期时严格失效。"""

        return self.expires_at > now


class ControllerLease:
    """持久化并校验一个 V2 Controller lease。

    ``path`` 可以是直接的 JSON 文件，也可以是运行目录；后者会使用
    ``<path>/.locks/controller_lease.json``。调用方通过 ``acquire``、``renew``
    和 ``release`` 管理 lease，并在每个新副作用前调用 ``assert_valid``。
    """

    def __init__(
        self,
        path: str | os.PathLike[str] | None = None,
        *,
        research_task_id: str,
        holder_id: str,
        ttl_seconds: float = DEFAULT_LEASE_TTL_SECONDS,
        clock: Any = time.time,
        lease_path: str | os.PathLike[str] | None = None,
        runtime_root: str | os.PathLike[str] | None = None,
    ) -> None:
        """绑定任务、持有者和独立运行时记录路径。"""

        supplied_paths = [item for item in (path, lease_path, runtime_root) if item is not None]
        if len(supplied_paths) != 1:
            raise ValueError("必须且只能提供 path、lease_path 或 runtime_root")
        raw_path = Path(supplied_paths[0]).expanduser()
        if lease_path is not None:
            resolved_path = raw_path
        elif runtime_root is not None or (raw_path.exists() and raw_path.is_dir()) or not raw_path.suffix:
            resolved_path = raw_path / ".locks" / "controller_lease.json"
        else:
            resolved_path = raw_path
        # 只做绝对化，不解析符号链接；后续 open/父链校验必须能看见恶意链接。
        self.lease_path = Path(os.path.abspath(os.fspath(resolved_path)))
        self.lock_path = self.lease_path.with_name(self.lease_path.name + ".lock")
        self.research_task_id = _require_text(
            research_task_id, "research_task_id"
        )
        self.holder_id = _require_text(holder_id, "holder_id")
        self.ttl_seconds = _require_ttl(ttl_seconds)
        self.clock = clock
        self._fencing_token: int | None = None
        self._acquired = False

    @classmethod
    def for_task_root(
        cls,
        task_root: str | os.PathLike[str],
        *,
        research_task_id: str,
        holder_id: str,
        ttl_seconds: float = DEFAULT_LEASE_TTL_SECONDS,
        clock: Any = time.time,
    ) -> "ControllerLease":
        """从任务运行目录构造独立 Controller lease。"""

        return cls(
            runtime_root=task_root,
            research_task_id=research_task_id,
            holder_id=holder_id,
            ttl_seconds=ttl_seconds,
            clock=clock,
        )

    @property
    def fencing_token(self) -> int | None:
        """返回当前句柄持有的 fencing token。"""

        return self._fencing_token

    @property
    def is_acquired(self) -> bool:
        """返回本句柄是否仍认为自己持有 lease。"""

        return self._acquired

    @property
    def record(self) -> ControllerLeaseRecord | None:
        """读取当前持久记录的便捷属性。"""

        return self.read_record()

    @property
    def is_present(self) -> bool:
        """返回记录中是否存在未过期的任意 Controller。"""

        record = self.read_record()
        return record is not None and record.is_active(self._now())

    def _now(self) -> float:
        """读取注入时钟。"""

        return _clock_seconds(self.clock)

    def _read_record_unlocked(self) -> ControllerLeaseRecord | None:
        """在记录锁内读取 lease；缺失文件表示尚未获取。"""

        if not self.lease_path.exists():
            if self.lease_path.is_symlink():
                raise ControllerLeaseIntegrityError("lease 文件不得是符号链接")
            return None
        if self.lease_path.is_symlink() or not self.lease_path.is_file():
            raise ControllerLeaseIntegrityError("lease 文件必须是普通文件")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.lease_path, flags)
        except OSError as exc:
            raise ControllerLeaseIntegrityError("无法读取 lease 文件") from exc
        try:
            file_stat = os.fstat(descriptor)
            if not stat.S_ISREG(file_stat.st_mode):
                raise ControllerLeaseIntegrityError("lease 文件必须是普通文件")
            if stat.S_IMODE(file_stat.st_mode) != 0o600:
                raise ControllerLeaseIntegrityError("lease 文件必须是 0600 权限")
            with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
                descriptor = -1
                try:
                    payload = json.load(stream)
                except (OSError, ValueError, TypeError) as exc:
                    raise ControllerLeaseIntegrityError("lease JSON 无法解析") from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        return ControllerLeaseRecord.from_mapping(
            payload,
            expected_research_task_id=self.research_task_id,
        )

    def read_record(self) -> ControllerLeaseRecord | None:
        """读取当前任务的 lease 记录，不改变任何状态。"""

        _ensure_private_parent(self.lease_path)
        with _record_lock(self.lock_path):
            return self._read_record_unlocked()

    @staticmethod
    def _write_record_unlocked(path: Path, record: ControllerLeaseRecord) -> None:
        """以临时文件、fsync 和原子替换保存 lease。"""

        _ensure_private_parent(path)
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ControllerLeaseIntegrityError("lease 文件必须是普通文件")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary_path = Path(temporary_name)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                descriptor = -1
                json.dump(record.to_dict(), stream, ensure_ascii=False, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, path)
            os.chmod(path, 0o600)
            try:
                directory_descriptor = os.open(path.parent, os.O_RDONLY)
            except OSError:
                directory_descriptor = -1
            if directory_descriptor >= 0:
                try:
                    os.fsync(directory_descriptor)
                finally:
                    os.close(directory_descriptor)
        except Exception:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
            raise

    def acquire(self) -> "ControllerLease":
        """获取 lease；过期记录接管时生成严格递增 token。"""

        if self._acquired:
            raise ControllerLeaseBusy("当前句柄已经持有 lease")
        now = self._now()
        with _record_lock(self.lock_path):
            existing = self._read_record_unlocked()
            if existing is not None and existing.is_active(now):
                raise ControllerLeaseBusy(
                    f"research_task {self.research_task_id} 已由 holder {existing.holder_id} 持有 lease"
                )
            token = 1 if existing is None else existing.fencing_token + 1
            record = ControllerLeaseRecord(
                research_task_id=self.research_task_id,
                holder_id=self.holder_id,
                fencing_token=token,
                renewed_at=now,
                expires_at=now + self.ttl_seconds,
            )
            self._write_record_unlocked(self.lease_path, record)
        self._fencing_token = token
        self._acquired = True
        return self

    def _require_current_record(self, now: float) -> ControllerLeaseRecord:
        """确认本句柄的 holder、token 和 expiry 仍有效。"""

        if not self._acquired or self._fencing_token is None:
            raise ControllerLeaseLost("当前句柄没有有效 lease")
        record = self._read_record_unlocked()
        if record is None:
            self._acquired = False
            raise ControllerLeaseLost("lease 记录已经消失")
        if (
            record.holder_id != self.holder_id
            or record.fencing_token != self._fencing_token
        ):
            self._acquired = False
            raise ControllerLeaseLost("holder 或 fencing token 已失效")
        if not record.is_active(now):
            self._acquired = False
            raise ControllerLeaseLost("lease 已到期")
        return record

    def assert_valid(self) -> ControllerLeaseRecord:
        """在新副作用前核对当前 holder 和 fencing token。"""

        now = self._now()
        with _record_lock(self.lock_path):
            return self._require_current_record(now)

    def is_valid(self) -> bool:
        """返回当前句柄是否仍有效，不向调用方抛出 lease 丢失异常。"""

        try:
            self.assert_valid()
        except ControllerLeaseLost:
            return False
        return True

    assert_fencing_token = assert_valid

    def renew(self, ttl_seconds: float | None = None) -> ControllerLeaseRecord:
        """为当前 holder 续期；失去 token 时 fail-closed。"""

        ttl = self.ttl_seconds if ttl_seconds is None else _require_ttl(ttl_seconds)
        now = self._now()
        with _record_lock(self.lock_path):
            current = self._require_current_record(now)
            renewed = ControllerLeaseRecord(
                research_task_id=current.research_task_id,
                holder_id=current.holder_id,
                fencing_token=current.fencing_token,
                renewed_at=now,
                expires_at=now + ttl,
            )
            self._write_record_unlocked(self.lease_path, renewed)
            return renewed

    # 业务调用方可以使用动作+对象命名；保留短方法作为底层协议入口。
    renew_lease = renew

    def release(self) -> None:
        """释放当前 lease；过期或被接管的旧 holder 无权释放。"""

        if not self._acquired:
            return
        now = self._now()
        with _record_lock(self.lock_path):
            current = self._require_current_record(now)
            released = ControllerLeaseRecord(
                research_task_id=current.research_task_id,
                holder_id=current.holder_id,
                fencing_token=current.fencing_token,
                renewed_at=current.renewed_at,
                expires_at=now,
            )
            self._write_record_unlocked(self.lease_path, released)
        self._acquired = False

    release_lease = release

    def __enter__(self) -> "ControllerLease":
        """获取并返回当前 lease。"""

        return self.acquire()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """离开上下文时释放仍由当前句柄持有的 lease。"""

        try:
            self.release()
        except ControllerLeaseLost:
            # 主流程已经在抛异常时，释放边界不能覆盖原始异常；没有主流程
            # 异常时仍向调用方报告 lease 丢失，避免把失效句柄当成已释放。
            if exc_type is None:
                raise


class ActiveControllerProcess(ControllerLease):
    """V2 领域名称：一个 research_task 的活动 Controller 进程。"""

    def __init__(
        self,
        path: str | os.PathLike[str] | None = None,
        *,
        research_task_id: str,
        holder_id: str,
        ttl_seconds: float = DEFAULT_LEASE_TTL_SECONDS,
        clock: Any = time.time,
        lease_path: str | os.PathLike[str] | None = None,
        runtime_root: str | os.PathLike[str] | None = None,
    ) -> None:
        """绑定 research_task；底层文件协议仍由兼容实现负责。"""

        super().__init__(
            path,
            research_task_id=research_task_id,
            holder_id=holder_id,
            ttl_seconds=ttl_seconds,
            clock=clock,
            lease_path=lease_path,
            runtime_root=runtime_root,
        )

__all__ = [
    "ActiveControllerProcess",
    "ControllerLease",
    "ControllerLeaseBusy",
    "ControllerLeaseBusyError",
    "ControllerLeaseError",
    "ControllerLeaseIntegrityConflict",
    "ControllerLeaseIntegrityError",
    "ControllerLeaseLost",
    "ControllerLeaseLostError",
    "ControllerLeaseRecord",
    "DEFAULT_LEASE_TTL_SECONDS",
    "LEASE_SCHEMA_VERSION",
    "LeaseBusy",
    "LeaseLost",
]
