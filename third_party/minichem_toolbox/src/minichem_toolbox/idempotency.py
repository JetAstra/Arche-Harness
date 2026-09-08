"""MiniChem 有副作用调用的持久幂等收据。"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import re
import secrets
import stat
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, TypeVar

from .artifacts import workspace_root
from .scope import current_scope


_OPERATION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_REQUEST_FINGERPRINT = re.compile(r"^[0-9a-f]{64}$")
_SUBMISSION_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$")
_DIRECTORY_MODE = 0o700
_PRIVATE_FILE_MODE = 0o600
_T = TypeVar("_T")


def _now() -> str:
    """返回 UTC 收据时间。"""

    return datetime.now(timezone.utc).isoformat()


def calculate_request_fingerprint(operation_name: str, request: Mapping[str, Any]) -> str:
    """根据工具名和去除内部身份的请求计算稳定指纹。"""

    request_payload = dict(request)
    for identity_field in (
        "operation_id",
        "caller_request_fingerprint",
        "submission_key",
    ):
        request_payload.pop(identity_field, None)
    payload = {
        "operation": operation_name,
        "request": request_payload,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_submission_key(value: Any) -> str:
    """校验并规范公开提交键；提交键只作为 scope 内语义身份，不是文件路径。"""

    if not isinstance(value, str):
        raise ValueError("submission_key must be a string")
    normalized = value.strip()
    if _SUBMISSION_KEY.fullmatch(normalized) is None:
        raise ValueError("submission_key must be a non-empty path-safe ASCII identity")
    return normalized


def scoped_submission_operation_id(submission_key: str) -> str:
    """由当前认证 scope 和公开提交键生成服务端 operation 身份。"""

    scoped = current_scope()
    if scoped is None:
        raise ValueError("an authenticated scope is required for submission identity")
    normalized = validate_submission_key(submission_key)
    digest = hashlib.sha256(
        f"{scoped.scope_id}\0{normalized}".encode("utf-8")
    ).hexdigest()
    return f"scope-submission-{digest}"


def scoped_submission_request_error(request: Any) -> dict[str, Any] | None:
    """返回 scope 请求缺失或伪造身份时的公开错误；legacy 请求返回 None。"""

    scoped = current_scope()
    if scoped is None:
        return None
    if isinstance(request, Mapping):
        operation_id = request.get("operation_id")
        submission_key = request.get("submission_key")
    else:
        operation_id = getattr(request, "operation_id", None)
        submission_key = getattr(request, "submission_key", None)
    if operation_id is not None:
        return {
            "code": "operation_id_not_assignable",
            "message": "当前 scope 的 operation_id 由 MiniChem 服务端生成，调用方不得提供",
        }
    if submission_key is None:
        return {
            "code": "missing_submission_key",
            "message": "当前 capability scope 的有副作用请求必须携带 submission_key",
        }
    try:
        validate_submission_key(submission_key)
    except ValueError as exc:
        return {"code": "invalid_submission_key", "message": str(exc)}
    return None


class OperationIdempotencyStore:
    """在工作区保存有副作用调用的 running/completed 收据。

    输入目录用于隔离一组 operation 收据；每次执行会在该目录创建锁文件和 JSON
    收据。相同身份和请求可安全读取已完成响应，身份冲突、损坏记录或遗留 running
    状态则返回技术暂停，不会再次执行副作用函数。目录或收据无法读写时直接抛出
    文件系统异常，交由 MCP 边界报告。
    """

    def __init__(self, root: str | Path | None = None, *, create: bool = True) -> None:
        configured_root = (
            Path(root)
            if root is not None
            else workspace_root() / "_tool_artifacts" / "operations"
        )
        self.root = Path(os.path.abspath(configured_root))
        self._prepare_operation_root(create=create)

    @staticmethod
    def _directory_open_flags() -> int:
        """返回逐组件打开私有目录所需的防跟随符号链接标志。"""

        return (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )

    @staticmethod
    def _raise_symlink_path_error(_exc: OSError) -> RuntimeError:
        """把内核拒绝符号链接转换为不泄露路径的 MiniChem 错误。"""

        return RuntimeError("operation 收据路径不得经过符号链接")

    def _open_operation_directory_component(
        self,
        parent_fd: int,
        component: str,
        *,
        create: bool,
    ) -> tuple[int, bool]:
        """以防跟随方式打开一个目录组件，并报告是否刚创建。"""

        flags = self._directory_open_flags()
        try:
            return os.open(component, flags, dir_fd=parent_fd), False
        except FileNotFoundError:
            if not create:
                raise
            created = False
            try:
                os.mkdir(component, _DIRECTORY_MODE, dir_fd=parent_fd)
                created = True
            except FileExistsError:
                pass
            try:
                child_fd = os.open(component, flags, dir_fd=parent_fd)
            except OSError as exc:
                if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                    raise self._raise_symlink_path_error(exc) from exc
                raise
            return child_fd, created
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                raise self._raise_symlink_path_error(exc) from exc
            raise

    def _open_operation_directory(self, *, create: bool) -> int:
        """逐组件打开收据根并返回已核对的目录 fd。

        输入是构造器固定的绝对收据根；每个组件都使用 ``O_NOFOLLOW`` 和
        ``fstat``，缺失组件仅通过 ``mkdirat`` 创建。返回的 fd 在调用方关闭前
        始终固定同一个目录，避免检查路径后再按字符串写入其他位置。
        """

        flags = self._directory_open_flags()
        try:
            directory_fd = os.open(self.root.anchor or os.sep, flags)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise self._raise_symlink_path_error(exc) from exc
            raise
        components = self.root.parts[1:]
        try:
            for index, component in enumerate(components):
                child_fd, created = self._open_operation_directory_component(
                    directory_fd,
                    component,
                    create=create,
                )
                metadata = os.fstat(child_fd)
                if not stat.S_ISDIR(metadata.st_mode):
                    os.close(child_fd)
                    raise RuntimeError("operation 收据根必须是普通目录")
                if created or index == len(components) - 1:
                    os.fchmod(child_fd, _DIRECTORY_MODE)
                os.close(directory_fd)
                directory_fd = child_fd
            return directory_fd
        except BaseException:
            os.close(directory_fd)
            raise

    def _prepare_operation_root(self, *, create: bool = True) -> None:
        """创建或核对收据根，并只把末级目录收紧为 0700。"""

        directory_fd = self._open_operation_directory(create=create)
        os.close(directory_fd)

    def _validate_operation_id(self, operation_id: str) -> str:
        """校验收据身份，防止身份改变文件边界。"""

        if not isinstance(operation_id, str) or _OPERATION_ID.fullmatch(operation_id) is None:
            raise ValueError("operation_id 必须是安全的 ASCII 标识")
        return operation_id

    def _validate_request_fingerprint(self, fingerprint: str) -> str:
        """校验调用方提供的 SHA-256 请求指纹并返回原值。"""

        if (
            not isinstance(fingerprint, str)
            or _REQUEST_FINGERPRINT.fullmatch(fingerprint) is None
        ):
            raise ValueError("fingerprint 必须是 64 位小写 SHA-256 十六进制字符串")
        return fingerprint

    def _paths(self, operation_id: str) -> tuple[Path, Path]:
        operation = self._validate_operation_id(operation_id)
        digest = hashlib.sha256(operation.encode("ascii")).hexdigest()
        return (
            self.root / f"{digest}.json",
            self.root / f".{digest}.lock",
        )

    @contextmanager
    def _open_operation_root(self, *, create: bool = True) -> Iterator[int]:
        """以逐组件防跟随方式打开收据根，并在离开时关闭目录 fd。"""

        directory_fd = self._open_operation_directory(create=create)
        try:
            yield directory_fd
        finally:
            os.close(directory_fd)

    @staticmethod
    def _corrupt_operation_record(message: str) -> dict[str, Any]:
        """构造阻止副作用重放的损坏收据标记。"""

        return {
            "status": "corrupt",
            "error": {
                "code": "operation_record_corrupt",
                "message": message,
            },
        }

    def _validate_child_path(self, child_path: Path) -> str:
        """校验收据子路径只含当前根目录中的单个安全名称。"""

        if child_path.parent != self.root or child_path.name in {"", ".", ".."}:
            raise ValueError("operation 收据路径必须位于收据根目录")
        return child_path.name

    def _read_operation_receipt_from_directory(
        self,
        directory_fd: int,
        receipt_name: str,
    ) -> dict[str, Any] | None:
        """从固定目录 fd 读取普通收据，FIFO、链接和损坏内容均 fail-closed。"""

        receipt_fd: int | None = None
        try:
            flags = (
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0)
            )
            try:
                receipt_fd = os.open(receipt_name, flags, dir_fd=directory_fd)
            except FileNotFoundError:
                return None
            except OSError as exc:
                if exc.errno == errno.ELOOP:
                    raise RuntimeError("operation 收据文件不得是符号链接") from exc
                raise
            if not stat.S_ISREG(os.fstat(receipt_fd).st_mode):
                return self._corrupt_operation_record(
                    "MiniChem operation 收据不是普通文件，已阻止重放"
                )
            with os.fdopen(receipt_fd, "r", encoding="utf-8") as handle:
                receipt_fd = None
                value = json.load(handle)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return self._corrupt_operation_record(
                "MiniChem operation 收据无法读取，已阻止重放"
            )
        finally:
            if receipt_fd is not None:
                os.close(receipt_fd)
        if isinstance(value, Mapping):
            return dict(value)
        return self._corrupt_operation_record(
            "MiniChem operation 收据格式非法，已阻止重放"
        )

    def _read_operation_receipt(
        self,
        receipt_path: Path,
        *,
        directory_fd: int | None = None,
    ) -> dict[str, Any] | None:
        """读取一份收据；目录 fd 已给出时不重新解析可变路径。"""

        receipt_name = self._validate_child_path(receipt_path)
        if directory_fd is not None:
            return self._read_operation_receipt_from_directory(
                directory_fd, receipt_name
            )
        try:
            with self._open_operation_root(create=False) as opened_directory:
                return self._read_operation_receipt_from_directory(
                    opened_directory, receipt_name
                )
        except FileNotFoundError:
            return None

    def read_operation_receipt(self, operation_id: str) -> dict[str, Any] | None:
        """只读读取当前工作区的一份 operation 收据，不创建目录或执行副作用。

        输入是安全的 operation_id；返回完整内部收据供服务端核对，找不到收据返回
        ``None``，损坏收据返回带 ``status=corrupt`` 的受控记录。该方法不把收据
        内容直接暴露给 MCP 调用方，调用方应自行过滤响应字段。
        """

        receipt_path, _lock_path = self._paths(operation_id)
        return self._read_operation_receipt(receipt_path)

    @staticmethod
    def _write_all_bytes(file_descriptor: int, payload: bytes) -> None:
        """完整写入字节串，处理 ``os.write`` 的短写。"""

        offset = 0
        while offset < len(payload):
            written = os.write(file_descriptor, payload[offset:])
            if written <= 0:
                raise OSError("operation 收据写入未前进")
            offset += written

    def _assert_operation_child(
        self,
        directory_fd: int,
        child_name: str,
        *,
        allow_missing: bool,
    ) -> os.stat_result | None:
        """在固定目录 fd 下核对目标不是链接且为普通文件。"""

        try:
            metadata = os.stat(
                child_name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            if allow_missing:
                return None
            raise
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError("operation 收据文件不得是符号链接")
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError("operation 收据文件必须是普通文件")
        return metadata

    def _open_unique_receipt_temporary(
        self,
        directory_fd: int,
        receipt_name: str,
    ) -> tuple[str, int]:
        """在固定收据根创建唯一的 0600 临时文件，不跟随既有链接。"""

        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        for _attempt in range(32):
            temporary_name = (
                f".{receipt_name}.{os.getpid()}.{secrets.token_hex(12)}.tmp"
            )
            try:
                descriptor = os.open(
                    temporary_name,
                    flags,
                    _PRIVATE_FILE_MODE,
                    dir_fd=directory_fd,
                )
            except FileExistsError:
                try:
                    existing = os.stat(
                        temporary_name,
                        dir_fd=directory_fd,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    continue
                if stat.S_ISLNK(existing.st_mode):
                    raise RuntimeError("operation 收据临时文件不得是符号链接")
                continue
            except OSError as exc:
                if exc.errno == errno.ELOOP:
                    raise RuntimeError("operation 收据临时文件不得是符号链接") from exc
                raise
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                os.close(descriptor)
                os.unlink(temporary_name, dir_fd=directory_fd)
                raise RuntimeError("operation 收据临时文件必须是普通文件")
            os.fchmod(descriptor, _PRIVATE_FILE_MODE)
            return temporary_name, descriptor
        raise FileExistsError("operation 收据临时文件名耗尽")

    def _write_operation_receipt(
        self,
        receipt_path: Path,
        value: Mapping[str, Any],
        *,
        directory_fd: int | None = None,
    ) -> None:
        """在固定收据目录内原子写出 0600 收据并同步文件和父目录。"""

        receipt_name = self._validate_child_path(receipt_path)
        if directory_fd is None:
            with self._open_operation_root() as opened_directory:
                self._write_operation_receipt(
                    receipt_path,
                    value,
                    directory_fd=opened_directory,
                )
            return
        self._assert_operation_child(
            directory_fd,
            receipt_name,
            allow_missing=True,
        )
        payload = dict(value)
        # OperationStore 的根目录按 workspace 隔离，但工作区可能在服务重启后被
        # 重新挂载；把 scope 身份写入每一份收据，恢复时才能区分遗留、foreign
        # 或被错误复制进来的记录。legacy/full 调用没有认证 scope，继续保留旧格式。
        scoped = current_scope()
        if scoped is not None:
            payload.update(
                {
                    "scope_id": scoped.scope_id,
                    "scope_workspace": str(scoped.workspace),
                }
            )
        encoded = (
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        ).encode("utf-8")
        temporary_name, temporary_fd = self._open_unique_receipt_temporary(
            directory_fd,
            receipt_name,
        )
        replaced = False
        try:
            self._write_all_bytes(temporary_fd, encoded)
            os.fsync(temporary_fd)
            self._assert_operation_child(
                directory_fd,
                receipt_name,
                allow_missing=True,
            )
            os.close(temporary_fd)
            temporary_fd = -1
            os.replace(
                temporary_name,
                receipt_name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
            replaced = True
            os.fsync(directory_fd)
            receipt_fd = os.open(
                receipt_name,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0),
                dir_fd=directory_fd,
            )
            try:
                metadata = os.fstat(receipt_fd)
                if not stat.S_ISREG(metadata.st_mode):
                    raise RuntimeError("operation 收据文件必须是普通文件")
                os.fchmod(receipt_fd, _PRIVATE_FILE_MODE)
                os.fsync(receipt_fd)
            finally:
                os.close(receipt_fd)
            os.fsync(directory_fd)
        finally:
            if temporary_fd >= 0:
                os.close(temporary_fd)
            if not replaced:
                try:
                    os.unlink(temporary_name, dir_fd=directory_fd)
                except FileNotFoundError:
                    pass

    @contextmanager
    def _hold_operation_lock(
        self,
        operation_id: str,
    ) -> Iterator[tuple[Path, Path, int]]:
        """在固定收据目录 fd 上持有同一 operation 的跨进程锁。"""

        receipt_path, lock_path = self._paths(operation_id)
        lock_name = self._validate_child_path(lock_path)
        with self._open_operation_root() as directory_fd:
            lock_flags = (
                os.O_RDWR
                | os.O_CREAT
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0)
            )
            try:
                lock_fd = os.open(
                    lock_name,
                    lock_flags,
                    _PRIVATE_FILE_MODE,
                    dir_fd=directory_fd,
                )
            except OSError as exc:
                if exc.errno == errno.ELOOP:
                    raise RuntimeError("operation 锁文件不得是符号链接") from exc
                raise
            try:
                metadata = os.fstat(lock_fd)
                if not stat.S_ISREG(metadata.st_mode):
                    raise RuntimeError("operation 锁文件必须是普通文件")
                os.fchmod(lock_fd, _PRIVATE_FILE_MODE)
                os.fsync(lock_fd)
                os.fsync(directory_fd)
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
                yield receipt_path, lock_path, directory_fd
            finally:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                finally:
                    os.close(lock_fd)

    def write_running_operation_receipt(self, operation_id: str, fingerprint: str) -> None:
        """为恢复测试或外部接管写入一份新的 running 收据。

        输入是安全的 operation 身份和规范请求 SHA-256；函数会持有对应跨进程锁并
        原子创建收据。该操作只写状态，不执行科学工具。目标身份已有任意收据时抛出
        ``FileExistsError``，避免覆盖能够阻止重复副作用的既有事实。
        """

        normalized_fingerprint = self._validate_request_fingerprint(fingerprint)

        with self._hold_operation_lock(operation_id) as (
            receipt_path,
            _lock_path,
            directory_fd,
        ):
            existing = self._assert_operation_child(
                directory_fd,
                receipt_path.name,
                allow_missing=True,
            )
            if existing is not None:
                raise FileExistsError(f"operation 收据已存在：{operation_id}")
            self._write_operation_receipt(
                receipt_path,
                {
                    "schema_version": 1,
                    "operation_id": operation_id,
                    "fingerprint": normalized_fingerprint,
                    "status": "running",
                    "response": None,
                    "updated_at": _now(),
                },
                directory_fd=directory_fd,
            )

    def _read_inventory_receipt_record(
        self,
        directory_fd: int,
        receipt_name: str,
    ) -> dict[str, Any] | None:
        """从已打开的收据根读取一份普通 JSON 收据，失败返回不可核对标记。"""

        descriptor: int | None = None
        try:
            descriptor = os.open(
                receipt_name,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0),
                dir_fd=directory_fd,
            )
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                return None
            with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
                descriptor = None
                value = json.load(handle)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError):
            return None
        finally:
            if descriptor is not None:
                os.close(descriptor)
        return dict(value) if isinstance(value, Mapping) else None

    @staticmethod
    def _validate_inventory_operation_id(
        value: Mapping[str, Any],
        receipt_name: str,
        operation_ids: set[str],
    ) -> str | None:
        """核对 inventory 收据中的 operation 身份和哈希文件名。"""

        operation_id = value.get("operation_id")
        if not isinstance(operation_id, str):
            return None
        if _OPERATION_ID.fullmatch(operation_id) is None:
            return None
        if operation_id in operation_ids:
            return None
        if hashlib.sha256(operation_id.encode("ascii")).hexdigest() != Path(
            receipt_name
        ).stem:
            return None
        return operation_id

    @staticmethod
    def _validate_inventory_submission_key(value: Any) -> tuple[bool, str | None]:
        """核对 inventory 收据中的可选 submission_key。"""

        if value is None:
            return True, None
        if not isinstance(value, str) or _SUBMISSION_KEY.fullmatch(value) is None:
            return False, None
        return True, value

    def _validate_inventory_record_header(
        self,
        value: Mapping[str, Any] | None,
        receipt_name: str,
        operation_ids: set[str],
    ) -> tuple[str, str, str | None] | None:
        """核对清单收据的文件名、身份、指纹和状态字段。"""

        if value is None:
            return None
        operation_id = self._validate_inventory_operation_id(
            value,
            receipt_name,
            operation_ids,
        )
        fingerprint = value.get("fingerprint")
        status = value.get("status")
        valid_submission, submission_key = self._validate_inventory_submission_key(
            value.get("submission_key")
        )
        if (
            isinstance(value.get("schema_version"), bool)
            or value.get("schema_version") != 1
            or operation_id is None
            or not isinstance(fingerprint, str)
            or _REQUEST_FINGERPRINT.fullmatch(fingerprint) is None
            or status not in {"running", "completed"}
            or "response" not in value
            or not valid_submission
        ):
            return None
        return operation_id, status, submission_key

    @staticmethod
    def _validate_inventory_record_status(
        value: Mapping[str, Any],
        status: str,
    ) -> bool:
        """核对 running/completed 收据与响应字段是否匹配。"""

        if status == "running":
            return value.get("response") is None
        return isinstance(value.get("response"), Mapping)

    def _validate_operation_inventory_record(
        self,
        value: Mapping[str, Any] | None,
        receipt_name: str,
        scoped: Any,
        operation_ids: set[str],
    ) -> tuple[str, str] | None:
        """核对一份 receipt 的 scope、身份、状态和提交键。"""

        header = self._validate_inventory_record_header(value, receipt_name, operation_ids)
        if header is None or not self._validate_inventory_record_scope(value, scoped, header[0], header[2]):
            return None
        operation_id, status, _submission_key = header
        if not self._validate_inventory_record_status(value, status):
            return None
        return operation_id, status

    def _validate_inventory_record_scope(
        self,
        value: Mapping[str, Any],
        scoped: Any,
        operation_id: str,
        submission_key: str | None,
    ) -> bool:
        """核对当前 scope 身份及 scope submission operation 派生关系。"""

        if (
            value.get("scope_id") != scoped.scope_id
            or value.get("scope_workspace") != str(scoped.workspace)
        ):
            return False
        if operation_id.startswith("scope-submission-"):
            try:
                normalized_key = validate_submission_key(submission_key)
            except ValueError:
                return False
            if (
                submission_key is not None
                and operation_id != scoped_submission_operation_id(normalized_key)
            ):
                return False
        return True

    def _inspect_inventory_child(
        self,
        directory_fd: int,
        child_name: str,
        scoped: Any,
        operation_ids: set[str],
    ) -> tuple[str, str, tuple[str, str] | None] | None:
        """读取并核对 inventory 根目录中的单个锁或 receipt 条目。"""

        try:
            metadata = os.stat(
                child_name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except OSError:
            return None
        if stat.S_ISLNK(metadata.st_mode):
            return None
        if child_name.startswith(".") and child_name.endswith(".lock"):
            if (
                re.fullmatch(r"\.[0-9a-f]{64}\.lock", child_name) is None
                or not stat.S_ISREG(metadata.st_mode)
            ):
                return None
            lock_fd = None
            try:
                lock_fd = os.open(
                    child_name,
                    os.O_RDONLY
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_NONBLOCK", 0),
                    dir_fd=directory_fd,
                )
                if not stat.S_ISREG(os.fstat(lock_fd).st_mode):
                    return None
            except OSError:
                return None
            finally:
                if lock_fd is not None:
                    os.close(lock_fd)
            return "lock", child_name[1:-5], None
        if not stat.S_ISREG(metadata.st_mode) or not child_name.endswith(".json"):
            return None
        if re.fullmatch(r"[0-9a-f]{64}\.json", child_name) is None:
            return None
        record = self._read_inventory_receipt_record(directory_fd, child_name)
        entry = self._validate_operation_inventory_record(
            record,
            child_name,
            scoped,
            operation_ids,
        )
        if entry is None:
            return None
        return "receipt", Path(child_name).stem, entry

    def list_operation_inventory(
        self,
    ) -> tuple[list[dict[str, str]], bool | None, bool]:
        """恢复当前 scope 的完整 typed-operation 清单。

        函数只扫描 OperationStore 自己创建的哈希收据，不读取响应内容之外的工作区
        文件，也不返回路径、日志或进程信息。每份收据都必须包含当前 scope 身份、
        合法 operation/fingerprint、``running`` 或 ``completed`` 状态以及匹配的
        文件名；任一收据损坏、foreign、未知或目录无法核对时返回已确认的前缀、
        ``has_active_operations=None`` 和 ``inventory_complete=False``。目录不存在
        表示该 scope 尚未产生 typed operation，恢复为完整空清单。
        """

        scoped = current_scope()
        if scoped is None:
            return [], None, False

        inventory: list[dict[str, str]] = []
        operation_ids: set[str] = set()
        receipt_digests: set[str] = set()
        lock_digests: set[str] = set()
        has_active = False
        try:
            with self._open_operation_root(create=False) as directory_fd:
                children = sorted(os.listdir(directory_fd))
                for child_name in children:
                    # 每个 operation 只允许一个哈希 JSON 收据和一个普通锁文件。临时文件、
                    # 目录、FIFO、符号链接或任意未知条目都说明恢复边界不完整。
                    inspected = self._inspect_inventory_child(
                        directory_fd,
                        child_name,
                        scoped,
                        operation_ids,
                    )
                    if inspected is None:
                        return inventory, None, False
                    entry_type, digest, entry = inspected
                    if entry_type == "lock":
                        lock_digests.add(digest)
                        continue
                    operation_id, status = entry
                    operation_ids.add(operation_id)
                    receipt_digests.add(digest)
                    inventory.append({"operation_id": operation_id, "status": status})
                    has_active = has_active or status == "running"
                # OperationStore 永久保留每个锁文件；没有对应 receipt 的锁表示进程
                # 可能在落盘身份前崩溃，不能把它当作“没有 operation”。
                if lock_digests != receipt_digests:
                    return inventory, None, False
        except FileNotFoundError:
            # 没有创建过 typed operation 时，空目录就是可核对的完整清单。
            return [], False, True
        except (OSError, RuntimeError):
            return inventory, None, False
        inventory.sort(key=lambda item: item["operation_id"])
        return inventory, has_active, True

    @staticmethod
    def _technical_pause(
        operation_id: str,
        code: str,
        message: str,
        *,
        minichem_status: str | None = None,
    ) -> dict[str, Any]:
        """构造不执行副作用的技术暂停响应。"""

        return {
            "status": "technical_pause",
            "minichem_status": minichem_status or code,
            "operation_id": operation_id,
            "error": {"code": code, "message": message},
        }

    def _validate_scope_operation_identity(
        self,
        operation_id: str,
        submission_key: str | None,
    ) -> dict[str, str] | None:
        """核对 scope 调用只能使用由提交键派生的 operation 身份。"""

        if current_scope() is None:
            return None
        if submission_key is None:
            # 旧的内部恢复/清单接口可能在已认证 scope 中读取历史 operation；
            # 公开副作用入口仍由 scoped_submission_request_error 强制提交键。
            return None
        expected = scoped_submission_operation_id(submission_key)
        if operation_id != expected:
            return {
                "code": "operation_id_not_assignable",
                "message": "当前 scope 的 operation_id 由 MiniChem 服务端生成，调用方不得提供",
            }
        return None

    def _validate_existing_receipt_identity(
        self,
        existing: Mapping[str, Any],
        operation_id: str,
        fingerprint: str,
        submission_key: str | None,
    ) -> dict[str, Any] | None:
        """核对已有收据的 schema、指纹、提交键和 scope 身份。"""

        if (
            isinstance(existing.get("schema_version"), bool)
            or existing.get("schema_version") != 1
            or existing.get("operation_id") != operation_id
            or not isinstance(existing.get("fingerprint"), str)
            or _REQUEST_FINGERPRINT.fullmatch(str(existing.get("fingerprint"))) is None
        ):
            return self._technical_pause(
                operation_id,
                "operation_record_incomplete",
                "operation 收据身份或版本不完整，已阻止重新产生副作用",
            )
        stored_fingerprint = str(existing["fingerprint"])
        if stored_fingerprint != fingerprint:
            code = "submission_key_conflict" if submission_key is not None else "operation_id_conflict"
            message = (
                "同一 scope/submission_key 绑定了不同请求指纹，已阻止执行"
                if submission_key is not None
                else "同一 operation_id 绑定了不同请求指纹，已阻止执行"
            )
            return self._technical_pause(operation_id, code, message)
        if submission_key is not None and existing.get("submission_key") != submission_key:
            return self._technical_pause(
                operation_id,
                "operation_record_incomplete",
                "operation 收据缺少或篡改 submission_key，已阻止重放",
            )
        if submission_key is None and existing.get("submission_key") is not None:
            return self._technical_pause(
                operation_id,
                "operation_record_incomplete",
                "operation 收据属于 submission_key 提交，当前请求未提供该身份",
            )
        return self._validate_existing_scope_identity(existing, operation_id, submission_key)

    def _validate_existing_scope_identity(
        self,
        existing: Mapping[str, Any],
        operation_id: str,
        submission_key: str | None,
    ) -> dict[str, Any] | None:
        """核对已有收据是否属于当前认证 scope。"""

        scoped = current_scope()
        stored_scope_id = existing.get("scope_id")
        stored_scope_workspace = existing.get("scope_workspace")
        has_scope_identity = stored_scope_id is not None or stored_scope_workspace is not None
        if scoped is None:
            if has_scope_identity:
                return self._technical_pause(
                    operation_id,
                    "operation_scope_mismatch",
                    "operation 收据属于认证 scope，当前调用身份无法核对",
                )
            return None
        if (
            stored_scope_id != scoped.scope_id
            or stored_scope_workspace != str(scoped.workspace)
        ):
            code = "operation_scope_mismatch" if has_scope_identity else "operation_record_incomplete"
            message = (
                "operation 收据缺少当前 scope 身份，已阻止重放"
                if code == "operation_record_incomplete"
                else "operation 收据不属于当前 scope，已阻止重放"
            )
            return self._technical_pause(operation_id, code, message)
        if (
            submission_key is not None
            and operation_id != scoped_submission_operation_id(submission_key)
        ):
            return self._technical_pause(
                operation_id,
                "operation_record_incomplete",
                "scope operation 收据缺少可核对的 submission_key，已阻止重放",
            )
        return None

    def _resolve_existing_operation(
        self,
        existing: Mapping[str, Any],
        operation_id: str,
        fingerprint: str,
        submission_key: str | None,
    ) -> dict[str, Any]:
        """把已有收据解析为重放、技术暂停或继续执行标记。"""

        if existing.get("status") == "corrupt":
            error = existing.get("error")
            return self._technical_pause(
                operation_id,
                "operation_record_corrupt",
                "MiniChem operation 收据无法核对，已阻止重放",
                minichem_status="operation_record_corrupt",
            ) | {"error": dict(error) if isinstance(error, Mapping) else {}}
        identity_error = self._validate_existing_receipt_identity(
            existing,
            operation_id,
            fingerprint,
            submission_key,
        )
        if identity_error is not None:
            return identity_error
        if existing.get("status") == "completed" and isinstance(existing.get("response"), Mapping):
            response = deepcopy(dict(existing["response"]))
            if submission_key is not None:
                response.update({"submission_key": submission_key, "operation_id": operation_id})
            return response
        if existing.get("status") == "running":
            return self._technical_pause(
                operation_id,
                "operation_in_progress",
                "同一 operation 仍处于 running，已阻止重新产生副作用",
            )
        return self._technical_pause(
            operation_id,
            "operation_record_incomplete",
            "operation 收据不完整，已阻止重新产生副作用",
        )

    def _write_operation_state(
        self,
        receipt_path: Path,
        directory_fd: int,
        operation_id: str,
        fingerprint: str,
        status: str,
        submission_key: str | None,
        response: Mapping[str, Any] | None,
    ) -> None:
        """写入 running 或 completed 收据，所有字段均绑定同一身份。"""

        stored_response = None
        if response is not None:
            stored_response = dict(response)
            if submission_key is not None:
                stored_response.update(
                    {"submission_key": submission_key, "operation_id": operation_id}
                )
        self._write_operation_receipt(
            receipt_path,
            {
                "schema_version": 1,
                "operation_id": operation_id,
                "fingerprint": fingerprint,
                "status": status,
                "response": stored_response,
                "updated_at": _now(),
                **(
                    {"submission_key": submission_key}
                    if submission_key is not None
                    else {}
                ),
            },
            directory_fd=directory_fd,
        )

    def execute_idempotent_operation(
        self,
        operation_id: str,
        fingerprint: str,
        operation: Callable[[], _T],
        *,
        submission_key: str | None = None,
    ) -> _T | dict[str, Any]:
        """执行一次有幂等身份保护的副作用操作或重放已完成响应。

        输入是 operation 身份、规范请求 SHA-256 和返回 JSON object 的副作用函数。
        函数在跨进程锁内先落盘 running 收据，再执行一次副作用并原子保存 completed
        响应；相同请求的后续调用返回响应副本。身份冲突、损坏或未完成收据返回结构化
        技术暂停。副作用函数或文件系统失败会原样抛出，running 收据会保留以阻止
        无法判断是否已经生效的请求再次执行。
        """

        normalized_operation_id = self._validate_operation_id(operation_id)
        normalized_fingerprint = self._validate_request_fingerprint(fingerprint)
        normalized_submission_key = (
            validate_submission_key(submission_key)
            if submission_key is not None
            else None
        )
        identity_error = self._validate_scope_operation_identity(
            normalized_operation_id,
            normalized_submission_key,
        )
        if identity_error is not None:
            return {
                "status": "invalid_request",
                "error": identity_error,
                "automatic_fallback": False,
            }
        with self._hold_operation_lock(normalized_operation_id) as (
            receipt_path,
            _lock_path,
            directory_fd,
        ):
            existing = self._read_operation_receipt(
                receipt_path,
                directory_fd=directory_fd,
            )
            if existing is not None:
                replay = self._resolve_existing_operation(
                    existing,
                    normalized_operation_id,
                    normalized_fingerprint,
                    normalized_submission_key,
                )
                return replay
            self._write_operation_state(
                receipt_path,
                directory_fd,
                normalized_operation_id,
                normalized_fingerprint,
                "running",
                normalized_submission_key,
                None,
            )
            # operation 可能已经产生远端副作用后进程崩溃；running 收据必须保留，
            # 下一次调用只能暂停，不能把请求误当成首次提交。
            response = operation()
            if not isinstance(response, Mapping):
                raise TypeError("幂等副作用操作必须返回 JSON object")
            self._write_operation_state(
                receipt_path,
                directory_fd,
                normalized_operation_id,
                normalized_fingerprint,
                "completed",
                normalized_submission_key,
                response,
            )
            if normalized_submission_key is not None:
                return {
                    **dict(response),
                    "submission_key": normalized_submission_key,
                    "operation_id": normalized_operation_id,
                }
            return response

    def execute_scoped_submission(
        self,
        submission_key: str,
        fingerprint: str,
        operation: Callable[[], _T],
    ) -> _T | dict[str, Any]:
        """按当前 scope 和公开提交键原子执行或重放一次副作用。"""

        normalized = validate_submission_key(submission_key)
        operation_id = scoped_submission_operation_id(normalized)
        result = self.execute_idempotent_operation(
            operation_id,
            fingerprint,
            operation,
            submission_key=normalized,
        )
        if isinstance(result, Mapping):
            return {
                **dict(result),
                "submission_key": normalized,
                "operation_id": operation_id,
            }
        return result


__all__ = [
    "OperationIdempotencyStore",
    "calculate_request_fingerprint",
    "scoped_submission_operation_id",
    "scoped_submission_request_error",
    "validate_submission_key",
]
