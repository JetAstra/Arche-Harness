"""服务端计算资源预算、Gaussian 固定资源策略和预留。"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import stat
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

from .process_identity import (
    ProcessIdentity,
    capture_process_identity,
    inspect_process_group_membership,
    process_identity_activity,
)
from .scope import SHARED_SCHEDULER_ROOT_ENV, current_scope


AVAILABLE_CPU_CORES_ENV = "RESEARCHCHEMBENCH_AVAILABLE_CPU_CORES"
AVAILABLE_MEMORY_MB_ENV = "RESEARCHCHEMBENCH_AVAILABLE_MEMORY_MB"
AVAILABLE_GPU_COUNT_ENV = "RESEARCHCHEMBENCH_AVAILABLE_GPU_COUNT"

DEFAULT_AVAILABLE_CPU_CORES = 60
DEFAULT_AVAILABLE_MEMORY_MB = 320_000
DEFAULT_AVAILABLE_GPU_COUNT = 1

DEFAULT_REQUEST_CPU_CORES = 1
DEFAULT_REQUEST_MEMORY_MB = 4096
DEFAULT_REQUEST_GPU_COUNT = 0

# Gaussian 的 CPU/内存是服务启动时确定的策略，不属于调用方请求字段。
GAUSSIAN_CPU_CORES_ENV = "MINICHEM_GAUSSIAN_CPU_CORES"
GAUSSIAN_MEMORY_MB_ENV = "MINICHEM_GAUSSIAN_MEMORY_MB"
MINIMUM_GAUSSIAN_MEMORY_MB = 128

ACTIVE_JOB_STATES = {"queued", "running"}
GAUSSIAN_QUEUE_POLL_SECONDS = 0.05
# 提交阶段可能包含大输入 staging；未完成交接只在较长宽限期后清理。
GAUSSIAN_PENDING_GRACE_SECONDS = 300.0

_PRIVATE_DIRECTORY_MODE = 0o700
_PRIVATE_FILE_MODE = 0o600


def _reject_symlink_components(path: Path) -> None:
    """核对路径祖先不是符号链接；缺失的末级允许随后安全创建。"""

    absolute = Path(os.path.abspath(os.fspath(path)))
    cursor = Path(absolute.anchor or os.curdir)
    parts = absolute.parts[1:] if absolute.anchor else absolute.parts
    for part in parts:
        cursor /= part
        try:
            metadata = os.lstat(cursor)
        except FileNotFoundError:
            break
        except OSError as exc:
            raise RuntimeError("MiniChem private path cannot be inspected") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError("MiniChem private path cannot contain a symlink")


def _private_directory_flags() -> int:
    """返回逐级打开私有目录所需的防跟随标志。"""

    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _private_absolute_path(path: Path) -> Path:
    """把内部路径转成不解析符号链接的绝对词法路径。"""

    return Path(os.path.abspath(os.fspath(path)))


def _private_directory_matches(path: Path, descriptor: int) -> bool:
    """比较已打开目录描述符与路径当前身份，发现祖先替换即失败。"""

    try:
        _reject_symlink_components(path)
        by_path = os.stat(path, follow_symlinks=False)
        by_descriptor = os.fstat(descriptor)
    except OSError:
        return False
    return bool(
        stat.S_ISDIR(by_path.st_mode)
        and stat.S_ISDIR(by_descriptor.st_mode)
        and by_path.st_dev == by_descriptor.st_dev
        and by_path.st_ino == by_descriptor.st_ino
    )


def _verify_private_directory_identity(path: Path, descriptor: int) -> None:
    """核对目录描述符仍对应原路径，避免父目录被替换后继续写入。"""

    if not _private_directory_matches(path, descriptor):
        raise RuntimeError("MiniChem private directory identity changed")


def _open_private_directory_descriptor(
    path: Path,
    *,
    create: bool,
    tighten: bool = False,
) -> tuple[int, Path]:
    """从根目录逐级用 dirfd 打开目录，不让任何祖先路径跟随 symlink。"""

    absolute = _private_absolute_path(path)
    _reject_symlink_components(absolute)
    descriptor = os.open(absolute.anchor or os.curdir, _private_directory_flags())
    components = absolute.parts[1:] if absolute.anchor else absolute.parts
    try:
        for index, component in enumerate(components):
            try:
                child = os.open(
                    component,
                    _private_directory_flags(),
                    dir_fd=descriptor,
                )
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(component, _PRIVATE_DIRECTORY_MODE, dir_fd=descriptor)
                except FileExistsError:
                    pass
                child = os.open(
                    component,
                    _private_directory_flags(),
                    dir_fd=descriptor,
                )
            os.close(descriptor)
            descriptor = child
            if tighten and index == len(components) - 1:
                os.fchmod(descriptor, _PRIVATE_DIRECTORY_MODE)
        _verify_private_directory_identity(absolute, descriptor)
        return descriptor, absolute
    except BaseException:
        os.close(descriptor)
        raise


def _open_private_file_at(
    directory_descriptor: int,
    name: str,
    flags: int,
    mode: int = 0,
) -> int:
    """相对稳定目录打开文件，叶子也拒绝符号链接。"""

    return os.open(
        name,
        flags
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        mode,
        dir_fd=directory_descriptor,
    )


def _create_unique_private_temp(
    directory_descriptor: int,
    prefix: str,
) -> tuple[int, str]:
    """在稳定目录描述符下用 O_EXCL 创建唯一临时文件。"""

    for _attempt in range(128):
        name = f"{prefix}{uuid.uuid4().hex}"
        try:
            return (
                _open_private_file_at(
                    directory_descriptor,
                    name,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    _PRIVATE_FILE_MODE,
                ),
                name,
            )
        except FileExistsError:
            continue
    raise RuntimeError("MiniChem private temporary name allocation failed")


def _unlink_private_file(path: Path) -> bool:
    """用稳定父目录删除一个私有文件，并同步删除后的目录项。"""

    try:
        descriptor, absolute_parent = _open_private_directory_descriptor(
            path.parent,
            create=False,
        )
    except FileNotFoundError:
        return False
    try:
        _verify_private_directory_identity(absolute_parent, descriptor)
        try:
            os.unlink(path.name, dir_fd=descriptor)
        except FileNotFoundError:
            return False
        os.fsync(descriptor)
        _verify_private_directory_identity(absolute_parent, descriptor)
        return True
    finally:
        os.close(descriptor)


def _ensure_private_directory(path: Path) -> Path:
    """逐级创建并收紧私有目录，目录或祖先异常时直接失败。"""

    descriptor, _absolute = _open_private_directory_descriptor(
        path,
        create=True,
        tighten=True,
    )
    os.close(descriptor)
    return path


def _validate_private_directory_path(path: Path, *, create: bool) -> Path:
    """返回词法绝对目录路径，并以稳定 dirfd 拒绝任一祖先 symlink。"""

    descriptor, absolute = _open_private_directory_descriptor(
        path,
        create=create,
    )
    try:
        _verify_private_directory_identity(absolute, descriptor)
    finally:
        os.close(descriptor)
    return absolute


def _sync_parent_directory(path: Path) -> None:
    """用稳定目录描述符同步目录项，保证写入或删除在断电后可见。"""

    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        metadata = None
    parent = path if metadata is not None and stat.S_ISDIR(metadata.st_mode) else path.parent
    descriptor, absolute_parent = _open_private_directory_descriptor(
        parent,
        create=False,
    )
    try:
        _verify_private_directory_identity(absolute_parent, descriptor)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _open_private_lock(path: Path) -> int:
    """以稳定父目录和 O_NOFOLLOW 打开私有锁文件并返回描述符。"""

    parent_descriptor, absolute_parent = _open_private_directory_descriptor(
        path.parent,
        create=True,
        tighten=True,
    )
    descriptor = -1
    try:
        _verify_private_directory_identity(absolute_parent, parent_descriptor)
        descriptor = _open_private_file_at(
            parent_descriptor,
            path.name,
            os.O_CREAT | os.O_RDWR,
            _PRIVATE_FILE_MODE,
        )
        os.fchmod(descriptor, _PRIVATE_FILE_MODE)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError("MiniChem lock must be a regular file")
        os.fsync(descriptor)
        os.fsync(parent_descriptor)
        _verify_private_directory_identity(absolute_parent, parent_descriptor)
        return descriptor
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    finally:
        os.close(parent_descriptor)


def _write_private_text_atomically(
    path: Path,
    text: str,
    *,
    encoding: str = "utf-8",
) -> None:
    """用稳定 dirfd、唯一临时文件和双重 fsync 原子写入文本。"""

    _ensure_private_directory(path.parent)
    parent_descriptor, absolute_parent = _open_private_directory_descriptor(
        path.parent,
        create=False,
    )
    descriptor = -1
    temporary_name: str | None = None
    try:
        _verify_private_directory_identity(absolute_parent, parent_descriptor)
        try:
            target_metadata = os.stat(
                path.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            target_metadata = None
        if target_metadata is not None and not stat.S_ISREG(target_metadata.st_mode):
            raise RuntimeError("MiniChem atomic target is not a regular file")
        descriptor, temporary_name = _create_unique_private_temp(
            parent_descriptor,
            f".{path.name}.tmp-",
        )
        os.fchmod(descriptor, _PRIVATE_FILE_MODE)
        with os.fdopen(descriptor, "w", encoding=encoding) as handle:
            descriptor = -1
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        _verify_private_directory_identity(absolute_parent, parent_descriptor)
        try:
            target_metadata = os.stat(
                path.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            target_metadata = None
        if target_metadata is not None and stat.S_ISLNK(target_metadata.st_mode):
            raise RuntimeError("MiniChem atomic target became a symlink")
        os.replace(
            temporary_name,
            path.name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        temporary_name = None
        os.fsync(parent_descriptor)
        _verify_private_directory_identity(absolute_parent, parent_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_name is not None:
            try:
                os.unlink(temporary_name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
        os.close(parent_descriptor)


def _environment_integer(name: str, default: int, *, minimum: int) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < minimum:
        qualifier = "non-negative" if minimum == 0 else f">= {minimum}"
        raise ValueError(f"{name} must be {qualifier}")
    return value


@dataclass(frozen=True)
class ResourceBudget:
    cpu_cores: int
    memory_mb: int
    gpu_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "cpu_cores": self.cpu_cores,
            "memory_mb": self.memory_mb,
            "gpu_count": self.gpu_count,
            "source": "evaluation_policy",
            "agent_controllable": False,
            "scope": "per_task",
        }


@dataclass(frozen=True)
class GaussianResourcePolicy:
    """保存一次服务启动所绑定的 Gaussian CPU 和内存配置。"""

    cpu_cores: int
    memory_mb: int

    def as_dict(self) -> dict[str, int]:
        """返回内部收据使用的实际资源值，不包含环境变量或宿主信息。"""

        return {
            "cpu_cores": self.cpu_cores,
            "memory_mb": self.memory_mb,
        }


def _strict_positive_environment_integer(
    name: str,
    raw: str | None,
    *,
    minimum: int = 1,
) -> int | None:
    """解析可选的正整数配置；缺失返回 None，空值、非整数和小值直接失败。"""

    if raw is None:
        return None
    text = raw
    if not re.fullmatch(r"[1-9][0-9]*", text):
        raise ValueError(f"{name} must be a positive integer")
    value = int(text)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


class ResourceBudgetExceeded(ValueError):
    """Structured rejection when one request exceeds a task resource budget."""

    def __init__(
        self,
        *,
        requested: Mapping[str, int],
        budget: ResourceBudget,
        reserved: Mapping[str, int] | None = None,
        aggregate: bool = False,
    ) -> None:
        self.requested = dict(requested)
        self.budget = budget
        self.reserved = dict(reserved or _zero_resources())
        self.aggregate = aggregate
        available = {
            name: max(0, int(getattr(budget, name)) - int(self.reserved.get(name, 0)))
            for name in ("cpu_cores", "memory_mb", "gpu_count")
        }
        scope = "aggregate active-job" if aggregate else "single-request"
        super().__init__(
            f"Requested resources exceed the evaluator-controlled {scope} budget: "
            f"requested={self.requested}, reserved={self.reserved}, "
            f"available={available}, budget={budget.as_dict()}"
        )

    def as_error(self) -> dict[str, Any]:
        available = {
            name: max(
                0,
                int(getattr(self.budget, name)) - int(self.reserved.get(name, 0)),
            )
            for name in ("cpu_cores", "memory_mb", "gpu_count")
        }
        return {
            "code": (
                "aggregate_resource_budget_exceeded"
                if self.aggregate
                else "resource_budget_exceeded"
            ),
            "message": str(self),
            "requested": self.requested,
            "currently_reserved": self.reserved,
            "available": available,
            "budget": self.budget.as_dict(),
            "retryable": self.aggregate,
        }


class ResourceStateUnavailable(RuntimeError):
    """持久资源状态无法核对时的 fail-closed 错误。"""

    def __init__(self, message: str = "MiniChem resource state is unavailable") -> None:
        super().__init__(message)

    def as_error(self) -> dict[str, Any]:
        """返回不暴露路径、PID 或记录内容的受控资源错误。"""

        return {
            "code": "resource_state_unavailable",
            "message": "Persisted MiniChem resource state cannot be verified",
            "retryable": False,
        }


def evaluation_resource_budget() -> ResourceBudget:
    """读取服务启动时绑定的评测总预算。"""

    return ResourceBudget(
        cpu_cores=_environment_integer(
            AVAILABLE_CPU_CORES_ENV,
            DEFAULT_AVAILABLE_CPU_CORES,
            minimum=1,
        ),
        memory_mb=_environment_integer(
            AVAILABLE_MEMORY_MB_ENV,
            DEFAULT_AVAILABLE_MEMORY_MB,
            minimum=128,
        ),
        gpu_count=_environment_integer(
            AVAILABLE_GPU_COUNT_ENV,
            DEFAULT_AVAILABLE_GPU_COUNT,
            minimum=0,
        ),
    )


def fixed_job_resource_limits(
    *,
    budget: ResourceBudget | None = None,
) -> dict[str, int]:
    """返回所有计算作业统一使用的固定 CPU、内存和 GPU 请求。

    输入是可选的已解析服务预算；省略时从启动脚本导出的服务环境读取。函数不
    接收模型请求中的资源字段，也不按工具类型选择不同组合，调用方拿到的资源
    始终等于当前服务预算。配置无效时保留 ``evaluation_resource_budget`` 的
    fail-closed 异常，不启动一个资源含义不明确的作业。
    """

    selected = budget or evaluation_resource_budget()
    return {
        "cpu_cores": selected.cpu_cores,
        "memory_mb": selected.memory_mb,
        "gpu_count": selected.gpu_count,
    }


def gaussian_resource_policy(
    *,
    budget: ResourceBudget | None = None,
) -> GaussianResourcePolicy:
    """读取 Gaussian 专用 CPU/内存配置并核对其不超过服务总预算。

    输入来自服务进程环境；未设置时分别使用当前服务总 CPU 和内存。配置错误或
    超出服务预算时抛出 ValueError，使启动和目录冻结在不明确策略时失败。
    """

    selected = budget or evaluation_resource_budget()
    cpu_cores = _strict_positive_environment_integer(
        GAUSSIAN_CPU_CORES_ENV,
        os.environ.get(GAUSSIAN_CPU_CORES_ENV),
    )
    memory_mb = _strict_positive_environment_integer(
        GAUSSIAN_MEMORY_MB_ENV,
        os.environ.get(GAUSSIAN_MEMORY_MB_ENV),
        minimum=MINIMUM_GAUSSIAN_MEMORY_MB,
    )
    resolved_cpu = selected.cpu_cores if cpu_cores is None else cpu_cores
    resolved_memory = selected.memory_mb if memory_mb is None else memory_mb
    if resolved_cpu > selected.cpu_cores:
        raise ValueError(
            f"{GAUSSIAN_CPU_CORES_ENV} exceeds the configured service CPU budget"
        )
    if resolved_memory > selected.memory_mb:
        raise ValueError(
            f"{GAUSSIAN_MEMORY_MB_ENV} exceeds the configured service memory budget"
        )
    return GaussianResourcePolicy(
        cpu_cores=resolved_cpu,
        memory_mb=resolved_memory,
    )


def gaussian_resource_policy_record() -> dict[str, Any]:
    """返回不暴露数值、环境名或宿主信息的公开 Gaussian 资源策略。"""

    # 解析策略用于让目录冻结在错误配置下 fail-closed；返回值刻意不包含实际数值。
    gaussian_resource_policy()
    return {
        "mode": "service_managed_full_capacity",
        "concurrency": "serial",
        "agent_controllable": False,
    }


def apply_gaussian_resource_policy(
    resources: Mapping[str, Any] | None,
    *,
    budget: ResourceBudget | None = None,
) -> dict[str, Any]:
    """覆盖调用方 CPU/内存并保留 GPU、墙钟等非 Gaussian 策略字段。"""

    selected = normalize_resource_limits(resources or {})
    policy = gaussian_resource_policy(budget=budget)
    selected["cpu_cores"] = policy.cpu_cores
    selected["memory_mb"] = policy.memory_mb
    if resources:
        for key, value in resources.items():
            if key not in selected:
                selected[key] = value
    return selected


def resource_budget_record() -> dict[str, Any]:
    return evaluation_resource_budget().as_dict()


def normalize_resource_limits(value: Any) -> dict[str, int]:
    if hasattr(value, "model_dump"):
        supplied = value.model_dump(mode="json")
    else:
        supplied = dict(value or {})
    return {
        "cpu_cores": int(supplied.get("cpu_cores") or DEFAULT_REQUEST_CPU_CORES),
        "memory_mb": int(supplied.get("memory_mb") or DEFAULT_REQUEST_MEMORY_MB),
        "gpu_count": int(
            DEFAULT_REQUEST_GPU_COUNT
            if supplied.get("gpu_count") is None
            else supplied["gpu_count"]
        ),
    }


def validate_resource_limits(
    resources: Mapping[str, int],
    *,
    budget: ResourceBudget | None = None,
) -> None:
    selected = budget or evaluation_resource_budget()
    requested = normalize_resource_limits(resources)
    if any(
        requested[name] > int(getattr(selected, name))
        for name in ("cpu_cores", "memory_mb", "gpu_count")
    ):
        raise ResourceBudgetExceeded(requested=requested, budget=selected)


def _workspace_root() -> Path:
    scoped = current_scope()
    if scoped is not None:
        root = Path(scoped.workspace)
        try:
            return _validate_private_directory_path(root, create=False)
        except (FileNotFoundError, OSError, RuntimeError) as exc:
            raise ResourceStateUnavailable(
                "Authenticated MiniChem scope workspace does not exist"
            ) from exc
    raw = (
        os.environ.get("MINICHEM_MCP_WORKSPACE", "").strip()
        or os.environ.get("RESEARCHCHEMBENCH_WORKSPACE", "").strip()
    )
    if not raw:
        raise RuntimeError(
            "MINICHEM_MCP_WORKSPACE or RESEARCHCHEMBENCH_WORKSPACE is required "
            "for resource reservations"
        )
    root = Path(raw).expanduser()
    try:
        return _validate_private_directory_path(root, create=False)
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        raise ResourceStateUnavailable(
            "Configured MiniChem workspace is unavailable"
        ) from exc


def _budget_directory() -> Path:
    scoped = current_scope()
    if scoped is not None:
        # 先核对当前 scope 的实际工作区，再创建共享预算目录；缺失工作区
        # 不能被当成“没有活动作业”。
        _workspace_root()
    scheduler = scoped.scheduler_root if scoped is not None else None
    if scheduler is None:
        configured = os.environ.get(SHARED_SCHEDULER_ROOT_ENV, "").strip()
        if configured:
            candidate = Path(configured).expanduser()
            scheduler = _validate_private_directory_path(candidate, create=True)
    if scheduler is None:
        path = _workspace_root() / "outputs" / ".resource_budget"
    else:
        scheduler = _validate_private_directory_path(scheduler, create=True)
        if scoped is not None:
            _register_scope_workspace(scheduler, scoped)
        path = scheduler / "resource_budget"
    return _ensure_private_directory(path)


@contextmanager
def _scope_workspace_index_lock(scheduler: Path) -> Iterator[None]:
    """锁住共享 scope workspace 索引的完整读写周期。"""

    lock_path = scheduler / "scope_workspaces.lock"
    descriptor = _open_private_lock(lock_path)
    try:
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _register_scope_workspace(scheduler: Path, scope: Any) -> None:
    """在共享 scheduler 内登记 scope workspace，供重启后的资源扫描恢复。"""

    # 多个请求可能同时第一次触发资源扫描；独立的文件锁覆盖目录创建、记录读取
    # 和 O_EXCL 写入，使同一 scope 的合法竞态最终只保留一次幂等登记。
    with _scope_workspace_index_lock(scheduler):
        directory = scheduler / "scope_workspaces"
        if directory.exists() and (directory.is_symlink() or not directory.is_dir()):
            raise RuntimeError("Shared MiniChem scope workspace index is invalid")
        _ensure_private_directory(directory)
        identity = f"{scope.scope_id}\0{scope.workspace}".encode("utf-8")
        digest = hashlib.sha256(identity).hexdigest()
        path = directory / f"workspace_{digest}.json"
        if path.exists():
            if path.is_symlink() or not path.is_file():
                raise RuntimeError("Shared MiniChem scope workspace index entry is invalid")
            if stat.S_IMODE(path.stat().st_mode) & 0o077:
                raise RuntimeError("Shared MiniChem scope workspace index entry is too open")
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    "Shared MiniChem scope workspace index cannot be read"
                ) from exc
            if not isinstance(record, dict) or (
                record.get("scope_id") != scope.scope_id
                or record.get("workspace") != str(scope.workspace)
            ):
                raise RuntimeError("Shared MiniChem scope workspace identity conflict")
            os.chmod(path, _PRIVATE_FILE_MODE)
            return
        descriptor_record = os.open(
            path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
            _PRIVATE_FILE_MODE,
        )
        try:
            with os.fdopen(descriptor_record, "w", encoding="utf-8") as handle:
                descriptor_record = -1
                json.dump(
                    {
                        "schema_version": 1,
                        "scope_id": scope.scope_id,
                        "workspace": str(scope.workspace),
                    },
                    handle,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            if descriptor_record >= 0:
                os.close(descriptor_record)
        os.chmod(path, _PRIVATE_FILE_MODE)
        _sync_parent_directory(directory)


def _read_scope_workspace_index_record(path: Path) -> dict[str, Any]:
    """读取一个索引 JSON，并核对文件名、类型和私有权限。"""

    if re.fullmatch(r"workspace_[0-9a-f]{64}\.json", path.name) is None:
        raise ResourceStateUnavailable(
            "Shared MiniChem scope workspace index contains an unknown entry"
        )
    if path.is_symlink() or not path.is_file():
        raise ResourceStateUnavailable("Shared MiniChem scope workspace index entry is invalid")
    if stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise ResourceStateUnavailable("Shared MiniChem scope workspace index entry is too open")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResourceStateUnavailable(
            "Shared MiniChem scope workspace index cannot be read"
        ) from exc
    if not isinstance(record, dict) or record.get("schema_version") != 1:
        raise ResourceStateUnavailable("Shared MiniChem scope workspace index entry is invalid")
    return record


def _validate_scope_workspace_index_identity(
    path: Path,
    record: dict[str, Any],
) -> Path:
    """核对索引 scope 身份、文件名摘要和绝对 workspace 路径。"""

    raw_workspace = record.get("workspace")
    raw_scope_id = record.get("scope_id")
    if not isinstance(raw_workspace, str) or not isinstance(raw_scope_id, str):
        raise ResourceStateUnavailable("Shared MiniChem scope workspace index entry is invalid")
    if (
        not raw_workspace.strip()
        or not raw_scope_id.strip()
        or raw_workspace.strip() != raw_workspace
        or raw_scope_id.strip() != raw_scope_id
        or _SCOPED_IDENTITY.fullmatch(raw_scope_id) is None
    ):
        raise ResourceStateUnavailable("Shared MiniChem scope workspace index entry is incomplete")
    expected_name = "workspace_" + hashlib.sha256(
        f"{raw_scope_id}\0{raw_workspace}".encode("utf-8")
    ).hexdigest() + ".json"
    if path.name != expected_name:
        raise ResourceStateUnavailable("Shared MiniChem scope workspace identity conflict")
    workspace = Path(raw_workspace).expanduser()
    if not workspace.is_absolute() or any(part in {".", ".."} for part in workspace.parts):
        raise ResourceStateUnavailable("Shared MiniChem scope workspace index path is not canonical")
    return workspace


def _read_registered_scope_workspace(path: Path) -> Path:
    """读取并严格核对一个 scope workspace 索引项。"""

    record = _read_scope_workspace_index_record(path)
    workspace = _validate_scope_workspace_index_identity(path, record)
    try:
        resolved = _validate_private_directory_path(workspace, create=False)
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        raise ResourceStateUnavailable(
            "Shared MiniChem scope workspace cannot be resolved safely"
        ) from exc
    return resolved


def _registered_scope_workspaces(scheduler: Path) -> list[Path]:
    """读取共享 scheduler 的私有 workspace 索引；损坏记录直接失败。"""

    with _scope_workspace_index_lock(scheduler):
        directory = scheduler / "scope_workspaces"
        if directory.is_symlink():
            raise RuntimeError("Shared MiniChem scope workspace index is invalid")
        if not directory.exists():
            return []
        if not directory.is_dir():
            raise RuntimeError("Shared MiniChem scope workspace index is invalid")
        try:
            children = sorted(directory.iterdir(), key=lambda item: item.name)
        except OSError as exc:
            raise ResourceStateUnavailable(
                "Shared MiniChem scope workspace index cannot be read"
            ) from exc
        return [_read_registered_scope_workspace(path) for path in children]


def _validate_scope_workspace_ticket_path(workspace: Any) -> Path:
    """校验 ticket workspace 的类型、绝对形式和稳定目录身份。"""

    if not isinstance(workspace, str) or not workspace.strip():
        raise ResourceStateUnavailable("Gaussian queue ticket workspace is invalid")
    candidate = Path(workspace).expanduser()
    if not candidate.is_absolute():
        raise ResourceStateUnavailable("Gaussian queue ticket workspace is not absolute")
    try:
        resolved = _validate_private_directory_path(candidate, create=False)
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        raise ResourceStateUnavailable(
            "Gaussian queue ticket workspace cannot be resolved"
        ) from exc
    if resolved != Path(os.path.abspath(os.fspath(candidate))):
        raise ResourceStateUnavailable("Gaussian queue ticket workspace is not registered")
    return candidate


def _read_registered_scope_ticket_record(
    scheduler: Path,
    path: Path,
) -> dict[str, Any]:
    """在索引锁内读取 ticket 对应记录并核对文件权限。"""

    with _scope_workspace_index_lock(scheduler):
        if path.is_symlink() or not path.is_file():
            raise ResourceStateUnavailable(
                "Gaussian queue ticket workspace is not registered"
            )
        if stat.S_IMODE(path.stat().st_mode) & 0o077:
            raise ResourceStateUnavailable(
                "Shared MiniChem scope workspace index entry is too open"
            )
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ResourceStateUnavailable(
                "Shared MiniChem scope workspace index cannot be read"
            ) from exc
    if not isinstance(record, dict) or record.get("schema_version") != 1:
        raise ResourceStateUnavailable(
            "Shared MiniChem scope workspace index entry is invalid"
        )
    return record


def _validate_registered_scope_workspace(
    scheduler: Path | None,
    scope_id: Any,
    workspace: Any,
) -> None:
    """核对 ticket 的 scope/workspace 是否是已登记且仍存在的安全目录。"""

    if scheduler is None:
        raise ResourceStateUnavailable(
            "Scoped Gaussian ticket has no shared workspace registry"
        )
    if not isinstance(scope_id, str) or _SCOPED_IDENTITY.fullmatch(scope_id) is None:
        raise ResourceStateUnavailable("Gaussian queue ticket scope is not path-safe")
    candidate = _validate_scope_workspace_ticket_path(workspace)
    # 先完整扫描索引，未知文件、损坏记录或其他 scope 的缺失目标同样阻断恢复。
    _registered_scope_workspaces(scheduler)
    identity = f"{scope_id}\0{candidate}".encode("utf-8")
    path = scheduler / "scope_workspaces" / (
        "workspace_" + hashlib.sha256(identity).hexdigest() + ".json"
    )
    record = _read_registered_scope_ticket_record(scheduler, path)
    if record.get("scope_id") != scope_id or record.get("workspace") != str(candidate):
        raise ResourceStateUnavailable("Gaussian queue ticket scope identity conflicts")


def _execution_job_roots() -> list[Path]:
    """返回当前 scope 和共享 scheduler 索引中的所有执行作业根。"""

    current_root = _workspace_root()
    scope = current_scope()
    if scope is None:
        return [current_root]
    scheduler = scope.scheduler_root
    if scheduler is None:
        return [current_root]
    # _budget_directory 会验证并登记当前 workspace。
    _budget_directory()
    roots = [current_root]
    for workspace in _registered_scope_workspaces(scheduler):
        if workspace not in roots:
            roots.append(workspace)
    return roots


def gaussian_serial_gate_path() -> Path:
    """返回仅供服务内部使用的 Gaussian 串行锁路径。"""

    path = _budget_directory() / "gaussian.serial.lock"
    descriptor = _open_private_lock(path)
    os.close(descriptor)
    _sync_parent_directory(path.parent)
    return path


@contextmanager
def gaussian_serial_gate() -> Iterator[Any]:
    """持有仅限 Gaussian 的跨进程锁，避免等待作业占用非 Gaussian 资源。"""

    gate_path = gaussian_serial_gate_path()
    descriptor = _open_private_lock(gate_path)
    handle = os.fdopen(descriptor, "a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield handle
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _queue_now() -> str:
    """返回服务内部队列记录使用的 UTC 时间。"""

    return datetime.now(timezone.utc).isoformat()


def _gaussian_queue_directory() -> Path:
    """创建并返回跨进程 Gaussian 队列目录。"""

    directory = _budget_directory() / "gaussian_queue"
    return _ensure_private_directory(directory)


@contextmanager
def _gaussian_queue_lock() -> Iterator[None]:
    """锁住队列状态，保证编号、队首转移和释放互斥。"""

    path = _gaussian_queue_directory() / "queue.lock"
    descriptor = _open_private_lock(path)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _queue_atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    """原子写入一个队列条目，避免读取到半条 JSON。"""

    _write_private_text_atomically(
        path,
        json.dumps(dict(value), ensure_ascii=False, indent=2) + "\n",
    )


def _next_gaussian_queue_ticket() -> int:
    """在已持有队列锁时取得一个不会复用的 FIFO 编号。"""

    directory = _gaussian_queue_directory()
    path = directory / "next_ticket"
    current = _read_gaussian_queue_next_ticket(directory)
    if current is None:
        current = 0
    selected = current + 1
    _write_private_text_atomically(path, str(selected) + "\n", encoding="ascii")
    return selected


_GAUSSIAN_TICKET_FILENAME = re.compile(r"ticket_([0-9]{20})\.json")
_SCOPED_JOB_ID = re.compile(r"job_(?:[0-9a-f]{8}|[0-9a-f]{32})")
_SCOPED_IDENTITY = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}")


def _require_positive_integer(value: Any, description: str) -> int:
    """读取一个不接受 bool 的正整数持久字段。"""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ResourceStateUnavailable(f"{description} is invalid")
    if value <= 0:
        raise ResourceStateUnavailable(f"{description} is invalid")
    return value


def _validate_gaussian_ticket_identity(
    value: Mapping[str, Any], expected_ticket_id: int
) -> None:
    """核对 Gaussian ticket 的 schema、编号、归属、类型和状态字段。"""

    if isinstance(value.get("schema_version"), bool) or value.get("schema_version") != 1:
        raise ResourceStateUnavailable("Gaussian queue ticket schema is invalid")
    ticket_id = _require_positive_integer(value.get("ticket_id"), "Gaussian ticket id")
    if ticket_id != expected_ticket_id:
        raise ResourceStateUnavailable("Gaussian queue ticket filename does not match id")
    owner_id = value.get("owner_id")
    if not isinstance(owner_id, str) or not owner_id.strip():
        raise ResourceStateUnavailable("Gaussian queue ticket owner is invalid")
    if value.get("kind") not in {"native", "typed"}:
        raise ResourceStateUnavailable("Gaussian queue ticket kind is invalid")
    if value.get("state") not in {"queued", "running"}:
        raise ResourceStateUnavailable("Gaussian queue ticket state is invalid")


def _configured_scheduler_root() -> Path | None:
    """读取未认证模式使用的共享 scheduler 根，不跟随其符号链接。"""

    configured = os.environ.get(SHARED_SCHEDULER_ROOT_ENV, "").strip()
    if not configured:
        return None
    candidate = Path(configured).expanduser()
    try:
        return _validate_private_directory_path(candidate, create=False)
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        raise ResourceStateUnavailable(
            "Shared MiniChem scheduler root is unavailable"
        ) from exc


def _validate_gaussian_ticket_scope(value: Mapping[str, Any]) -> None:
    """核对 Gaussian ticket 的作业标识和跨 scope workspace 绑定。"""

    job_id = value.get("job_id")
    if job_id is not None and (not isinstance(job_id, str) or not job_id.strip()):
        raise ResourceStateUnavailable("Gaussian queue ticket job id is invalid")
    scope_id = value.get("scope_id")
    workspace = value.get("workspace")
    if scope_id is not None and (not isinstance(scope_id, str) or not scope_id.strip()):
        raise ResourceStateUnavailable("Gaussian queue ticket scope is invalid")
    if workspace is not None and (not isinstance(workspace, str) or not workspace.strip()):
        raise ResourceStateUnavailable("Gaussian queue ticket workspace is invalid")
    if scope_id is None:
        return
    if workspace is None:
        raise ResourceStateUnavailable("Gaussian queue ticket scope workspace is missing")
    if job_id is not None and _SCOPED_JOB_ID.fullmatch(job_id) is None:
        raise ResourceStateUnavailable("Gaussian queue ticket job id is not path-safe")
    scoped = current_scope()
    scheduler = scoped.scheduler_root if scoped is not None else _configured_scheduler_root()
    _validate_registered_scope_workspace(scheduler, scope_id, workspace)


def _validate_gaussian_ticket_process_fields(value: Mapping[str, Any]) -> None:
    """核对 Gaussian ticket 的提交者和 worker 进程身份字段。"""

    for field in ("owner_pid", "submitter_pid"):
        raw_pid = value.get(field)
        if raw_pid is not None:
            _require_positive_integer(raw_pid, f"Gaussian queue {field}")
    identity_fields = {
        "worker_pid",
        "worker_start_time_ticks",
        "worker_start_ticks",
        "worker_boot_id",
    }
    if not any(field in value for field in identity_fields):
        return
    if "worker_pid" not in value or "worker_boot_id" not in value or not (
        "worker_start_time_ticks" in value or "worker_start_ticks" in value
    ):
        raise ResourceStateUnavailable("Gaussian queue worker identity is incomplete")
    _require_positive_integer(value.get("worker_pid"), "Gaussian worker pid")
    _require_positive_integer(
        value.get("worker_start_time_ticks", value.get("worker_start_ticks")),
        "Gaussian worker start ticks",
    )
    if not isinstance(value.get("worker_boot_id"), str) or not value["worker_boot_id"].strip():
        raise ResourceStateUnavailable("Gaussian worker boot id is invalid")


def _validate_gaussian_ticket_record(
    path: Path,
    value: Any,
    expected_ticket_id: int,
) -> dict[str, Any]:
    """严格核对一个 FIFO ticket，任何缺失或越界字段都阻断队列。"""

    if not isinstance(value, dict):
        raise ResourceStateUnavailable("Gaussian queue ticket is not an object")
    _validate_gaussian_ticket_identity(value, expected_ticket_id)
    _validate_gaussian_ticket_scope(value)
    _validate_gaussian_ticket_process_fields(value)
    return value


def _read_gaussian_queue_next_ticket(directory: Path) -> int | None:
    """读取并校验下一个 ticket 计数器；临时或非法内容直接失败。"""

    path = directory / "next_ticket"
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ResourceStateUnavailable("Gaussian queue counter is invalid")
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding="ascii")
    except (OSError, UnicodeDecodeError) as exc:
        raise ResourceStateUnavailable("Gaussian queue counter cannot be read") from exc
    if re.fullmatch(r"[0-9]+\n?", raw) is None:
        raise ResourceStateUnavailable("Gaussian queue counter is invalid")
    try:
        selected = int(raw.strip())
    except ValueError as exc:
        raise ResourceStateUnavailable("Gaussian queue counter is invalid") from exc
    if selected < 0:
        raise ResourceStateUnavailable("Gaussian queue counter is invalid")
    return selected


def _gaussian_queue_entries() -> list[tuple[Path, dict[str, Any]]]:
    """严格读取 queued/running 条目并按 ticket 排序，禁止静默跳过损坏文件。"""

    directory = _gaussian_queue_directory()
    try:
        children = sorted(directory.iterdir(), key=lambda item: item.name)
    except OSError as exc:
        raise ResourceStateUnavailable("Gaussian queue directory cannot be read") from exc
    entries: list[tuple[Path, dict[str, Any]]] = []
    next_ticket = _read_gaussian_queue_next_ticket(directory)
    for path in children:
        if path.name == "queue.lock":
            if path.is_symlink() or not path.is_file():
                raise ResourceStateUnavailable("Gaussian queue lock is invalid")
            continue
        if path.name == "next_ticket":
            continue
        match = _GAUSSIAN_TICKET_FILENAME.fullmatch(path.name)
        if match is None or path.is_symlink() or not path.is_file():
            raise ResourceStateUnavailable("Gaussian queue contains an unknown entry")
        expected_ticket_id = int(match.group(1))
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
            raise ResourceStateUnavailable("Gaussian queue ticket cannot be read") from exc
        value = _validate_gaussian_ticket_record(path, value, expected_ticket_id)
        entries.append((path, value))
    entries.sort(key=lambda item: int(item[1]["ticket_id"]))
    if entries:
        if next_ticket is None or next_ticket < int(entries[-1][1]["ticket_id"]):
            raise ResourceStateUnavailable("Gaussian queue counter is behind a ticket")
    return entries


def read_gaussian_queue_snapshot() -> dict[str, Any]:
    """读取共享 Gaussian FIFO 的脱敏只读视图。

    输入不接受调用方参数；动作是在队列锁内读取当前仍处于 queued/running 的条目，按
    ticket 排序后只返回位置、状态和当前调用者是否拥有该项。当前 scope 的条目可返回
    自己的真实 job_id；其他 scope 只返回位置、状态和 ``owned=false``，不返回稳定匿名
    引用、作业归属、请求、路径、进程号或日志。stdio legacy 没有调用者 scope 时所有条目
    都按未拥有处理，避免把未经认证的请求当成某个任务。
    """

    try:
        with _gaussian_queue_lock():
            entries = _gaussian_queue_entries()
    except (OSError, RuntimeError, ValueError):
        return {
            "status": "unavailable",
            "scope": "global",
            "items": [],
            "item_count": 0,
        }
    scope = current_scope()
    items = []
    for position, (_path, value) in enumerate(entries, start=1):
        owned = bool(
            scope is not None
            and str(value.get("scope_id") or "") == scope.scope_id
            and str(value.get("workspace") or "").strip()
            == str(scope.workspace)
        )
        item: dict[str, Any] = {
            "position": position,
            "status": str(value["state"]),
            "owned": owned,
        }
        if owned and value.get("job_id"):
            item["job_id"] = str(value["job_id"])
        items.append(item)
    return {
        "status": "available",
        "scope": "global",
        "items": items,
        "item_count": len(items),
    }


def _process_is_alive(pid: int) -> bool:
    """判断队列持有者仍然存在；Linux 僵尸进程视为已退出。"""

    if pid <= 0:
        return False
    stat_path = Path("/proc") / str(pid) / "stat"
    try:
        fields = stat_path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        try:
            os.kill(pid, 0)
        except (OSError, ValueError):
            return False
        return True
    suffix = fields[fields.rfind(")") + 2 :].split()
    return bool(suffix) and suffix[0] != "Z"


def _normalize_worker_identity(value: Any) -> dict[str, Any]:
    """规范化 worker 身份，保留无法读取的启动时钟以便恢复时拒绝清理。"""

    if isinstance(value, ProcessIdentity):
        raw = value.as_dict()
    elif hasattr(value, "as_dict"):
        raw = dict(value.as_dict())
    else:
        raw = dict(value or {})
    try:
        pid = int(raw.get("pid") or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError("worker identity pid must be an integer") from exc
    if pid <= 0:
        raise ValueError("worker identity pid must be positive")
    start_time = raw.get("start_time_ticks")
    if start_time is not None:
        try:
            start_time = int(start_time)
        except (TypeError, ValueError) as exc:
            raise ValueError("worker identity start_time_ticks must be an integer") from exc
        if start_time <= 0:
            raise ValueError("worker identity start_time_ticks must be positive")
    boot_id = str(raw.get("boot_id") or "").strip() or None
    return {
        "pid": pid,
        "start_time_ticks": start_time,
        "boot_id": boot_id,
    }


def _worker_identity_from_record(value: Mapping[str, Any]) -> dict[str, Any] | None:
    """从队列或 reservation 记录取出 worker 身份；未绑定时返回 None。"""

    if "worker_pid" not in value:
        return None
    return {
        "pid": value.get("worker_pid"),
        "start_time_ticks": value.get(
            "worker_start_time_ticks", value.get("worker_start_ticks")
        ),
        "boot_id": value.get("worker_boot_id"),
    }


def _worker_identity_activity(value: Mapping[str, Any]) -> str:
    """返回持久 worker 身份的当前状态，未知时必须保留占用。"""

    identity = _worker_identity_from_record(value)
    if identity is None:
        return "unknown"
    if (
        isinstance(identity.get("start_time_ticks"), bool)
        or not isinstance(identity.get("start_time_ticks"), int)
        or identity["start_time_ticks"] <= 0
        or not isinstance(identity.get("boot_id"), str)
        or not identity["boot_id"].strip()
    ):
        # typed Gaussian 的 worker 身份是新记录的一部分；只有 PID 或只有
        # start_ticks 的部分记录不能按 legacy PID 兼容路径清理。
        return "unknown"
    return process_identity_activity(identity)


def _read_queue_json(path: Path) -> dict[str, Any] | None:
    """读取队列清理所需的 JSON 对象，损坏或缺失时返回 None。"""

    if path.is_symlink():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _read_queue_json_state(path: Path) -> tuple[str, dict[str, Any] | None]:
    """区分控制 JSON 缺失与损坏，供恢复逻辑避免把损坏当作 staging。"""

    if path.is_symlink() or (path.exists() and not path.is_file()):
        return "invalid", None
    if not path.exists():
        return "missing", None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError):
        return "invalid", None
    if not isinstance(value, dict):
        return "invalid", None
    return "valid", value


def _native_queue_job_directory(value: Mapping[str, Any]) -> Path | None:
    """根据 native 队列条目的稳定 job_id 定位持久作业目录。"""

    job_id = str(value.get("job_id") or "").strip()
    if not job_id:
        return None
    raw_workspace = str(value.get("workspace") or "").strip()
    if raw_workspace:
        try:
            root = _validate_private_directory_path(
                Path(raw_workspace).expanduser(),
                create=False,
            )
        except (FileNotFoundError, OSError, RuntimeError) as exc:
            raise ResourceStateUnavailable(
                "native queue workspace is unavailable"
            ) from exc
    else:
        root = _workspace_root()
    return root / "outputs" / "execution_jobs" / job_id


def _queue_pid(value: Mapping[str, Any], name: str) -> int:
    """读取清理判断中的 PID，畸形值按未绑定处理。"""

    try:
        return int(value.get(name) or 0)
    except (TypeError, ValueError):
        return 0


def _persisted_process_activity(
    value: Mapping[str, Any],
    prefix: str,
) -> str:
    """核对持久 PID 身份；旧记录无身份字段时走只读兼容判断。"""

    pid_key = f"{prefix}_pid"
    start_key = f"{prefix}_start_time_ticks"
    alias_key = f"{prefix}_start_ticks"
    boot_key = f"{prefix}_boot_id"
    if pid_key not in value:
        return "unknown"
    has_identity = any(key in value for key in (start_key, alias_key, boot_key))
    pid = _queue_pid(value, pid_key)
    if pid <= 0:
        return "unknown"
    if not has_identity:
        # legacy 队列清理不发送信号，只能沿用旧 PID 存活判断；取消路径另行拒绝
        # 没有身份的记录，避免该兼容判断变成危险的 SIGTERM 依据。
        return "alive" if _process_is_alive(pid) else "stopped"
    raw_start = value.get(start_key, value.get(alias_key))
    if (
        isinstance(raw_start, bool)
        or not isinstance(raw_start, int)
        or raw_start <= 0
        or not isinstance(value.get(boot_key), str)
        or not value[boot_key].strip()
    ):
        # 记录已经声明采用新身份协议但字段不完整；不能退回只核对 PID。
        return "unknown"
    identity = {
        "pid": pid,
        "start_time_ticks": raw_start,
        "boot_id": value.get(boot_key),
    }
    return process_identity_activity(identity)


def _persisted_process_group_activity(
    value: Mapping[str, Any],
    prefix: str,
) -> str:
    """核对持久进程组/会话身份，无法证明组状态时保持未知。"""

    group_key = f"{prefix}_process_group_id"
    session_key = f"{prefix}_session_id"
    boot_key = f"{prefix}_boot_id"
    if not any(key in value for key in (group_key, session_key)):
        # 没有新协议字段的旧记录沿用原恢复路径，由调用方决定兼容行为。
        return "legacy"
    raw_group = value.get(group_key)
    raw_session = value.get(session_key)
    raw_boot = value.get(boot_key)
    if (
        isinstance(raw_group, bool)
        or not isinstance(raw_group, int)
        or raw_group <= 0
        or isinstance(raw_session, bool)
        or not isinstance(raw_session, int)
        or raw_session <= 0
        or not isinstance(raw_boot, str)
        or not raw_boot.strip()
    ):
        return "unknown"
    return inspect_process_group_membership(raw_group, raw_session, raw_boot)


def _native_queue_child_activity(
    status: Mapping[str, Any], state: str, child_pid: int
) -> str | None:
    """核对 native child 与其进程组；None 表示继续核对 supervisor。"""

    if child_pid <= 0:
        return None
    child_activity = _persisted_process_activity(status, "child")
    if child_activity == "alive":
        return "child_alive"
    if child_activity == "unknown":
        return "unknown"
    group_activity = _persisted_process_group_activity(status, "child")
    if group_activity == "legacy":
        return "unknown"
    if group_activity == "nonempty":
        return "child_group_alive"
    if group_activity == "unknown":
        return "unknown"
    return None


def _native_queue_supervisor_activity(
    status: Mapping[str, Any], state: str, child_pid: int
) -> str:
    """在 child 已退出或未登记时核对 supervisor 及交接窗口。"""

    supervisor_activity = _persisted_process_activity(status, "supervisor")
    if supervisor_activity == "alive":
        return "supervisor_alive"
    if supervisor_activity == "unknown":
        return "unknown"
    if _queue_pid(status, "supervisor_pid") <= 0:
        return "unknown"
    if state == "running" and child_pid <= 0:
        return "unknown"
    if child_pid <= 0:
        return "unknown"
    return "dead"


def _native_queue_status_activity(status: Mapping[str, Any]) -> str:
    """根据已落盘 status 判断 native 作业是否仍占用 Gaussian FIFO。"""

    state = str(status.get("status") or "").casefold()
    if state in {"success", "failed", "timeout", "cancelled"}:
        return "terminal"
    if state not in {"queued", "running"}:
        return "unknown"
    child_pid = _queue_pid(status, "child_pid")
    child_result = _native_queue_child_activity(status, state, child_pid)
    if child_result is not None:
        return child_result
    return _native_queue_supervisor_activity(status, state, child_pid)


def _native_queue_claim_record_activity(
    claim: Mapping[str, Any],
    status: Mapping[str, Any] | None,
) -> str | None:
    """核对已写入的 supervisor claim；无法决定时交回交接路径继续判断。"""

    claim_activity = _persisted_process_activity(claim, "supervisor")
    if claim_activity == "alive":
        return "supervisor_alive"
    if claim_activity == "unknown":
        return "unknown"
    if _queue_pid(claim, "supervisor_pid") <= 0:
        return None
    active = str((status or {}).get("status") or "").casefold() in {
        "queued",
        "running",
    }
    return "unknown" if active else "dead"


def _native_queue_handoff_activity(
    value: Mapping[str, Any],
    directory: Path,
) -> str:
    """在没有可用 claim 时核对 owner PID 和交接控制文件。"""

    owner_pid = _queue_pid(value, "owner_pid")
    if owner_pid > 0:
        return "supervisor_alive" if _process_is_alive(owner_pid) else "dead"
    control_paths = [
        directory / name
        for name in ("supervisor_spec.json", "status.json", "request.json")
    ]
    if any(
        path.is_symlink() or (path.exists() and not path.is_file())
        for path in control_paths
    ):
        return "unknown"
    if any(
        path.is_file() and not path.is_symlink() for path in control_paths
    ):
        return "handoff_pending"
    return "staging"


def _native_queue_claim_activity(
    value: Mapping[str, Any],
    directory: Path,
    *,
    status: Mapping[str, Any] | None = None,
) -> str:
    """在 status 尚未生成时核对 supervisor claim、owner 和交接文件。"""

    claim_state, claim = _read_queue_json_state(directory / "supervisor_claim.json")
    if claim_state == "invalid":
        return "unknown"
    if claim is not None:
        claim_result = _native_queue_claim_record_activity(claim, status)
        if claim_result is not None:
            return claim_result
    return _native_queue_handoff_activity(value, directory)


def _native_queue_job_activity(value: Mapping[str, Any]) -> str:
    """读取 native 作业状态，区分活 supervisor、活 Gaussian 和失效交接。"""

    directory = _native_queue_job_directory(value)
    if directory is None:
        return "staging"
    status_state, status = _read_queue_json_state(directory / "status.json")
    if status_state == "invalid":
        return "unknown"
    if status is not None:
        return _native_queue_status_activity(status)
    return _native_queue_claim_activity(value, directory)


def _native_queue_entry_age(value: Mapping[str, Any]) -> float:
    """按 handoff、spec 或登记时间计算 native 交接条目的存活时间。"""

    timestamp = str(value.get("handoff_at") or value.get("created_at") or "")
    directory = _native_queue_job_directory(value)
    if not value.get("handoff_at") and directory is not None:
        spec_path = directory / "supervisor_spec.json"
        if spec_path.is_symlink():
            return GAUSSIAN_PENDING_GRACE_SECONDS + 1
        try:
            timestamp = datetime.fromtimestamp(
                spec_path.stat().st_mtime, timezone.utc
            ).isoformat()
        except OSError:
            pass
    try:
        created = datetime.fromisoformat(timestamp)
    except (TypeError, ValueError):
        return GAUSSIAN_PENDING_GRACE_SECONDS + 1
    return (datetime.now(timezone.utc) - created).total_seconds()


def _native_job_recovery_receipt(
    directory: Path,
    status: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]] | None:
    """读取并核对失联作业的 operation 回执；无 operation 时返回 None。"""

    operation_id = status.get("operation_id")
    if operation_id is None:
        return None
    receipt_path = directory / "operation_receipt.json"
    receipt_state, receipt = _read_queue_json_state(receipt_path)
    if receipt_state == "invalid" or receipt is None:
        raise ResourceStateUnavailable(
            "native Gaussian operation receipt is missing or invalid"
        )
    expected = {
        "operation_id": operation_id,
        "job_id": status.get("job_id"),
        "job_type": status.get("job_type"),
        "scope_id": status.get("scope_id"),
        "scope_workspace": status.get("scope_workspace"),
    }
    if any(receipt.get(field) != expected_value for field, expected_value in expected.items()):
        raise ResourceStateUnavailable(
            "native Gaussian operation receipt identity does not match status"
        )
    status_fingerprint = status.get("operation_fingerprint")
    receipt_fingerprint = receipt.get("operation_fingerprint") or receipt.get(
        "fingerprint"
    )
    if status_fingerprint != receipt_fingerprint:
        raise ResourceStateUnavailable(
            "native Gaussian operation receipt fingerprint does not match status"
        )
    return receipt_path, receipt


def _mark_native_job_supervisor_lost(value: Mapping[str, Any]) -> bool:
    """把 supervisor 和子进程都消失的运行记录标记为失败，保留停止原因。"""

    directory = _native_queue_job_directory(value)
    if directory is None:
        return False
    status_path = directory / "status.json"
    status_state, status = _read_queue_json_state(status_path)
    if status_state == "invalid":
        return False
    if status is None or str(status.get("status") or "").casefold() in {
        "success",
        "failed",
        "timeout",
        "cancelled",
    }:
        return True
    receipt_record = _native_job_recovery_receipt(directory, status)
    resource_usage = status.get("resource_usage")
    memory_limit_mb = None
    peak_rss_kb = None
    if isinstance(resource_usage, Mapping):
        memory_limit_mb = resource_usage.get("memory_limit_mb")
        peak_rss_kb = resource_usage.get("peak_process_group_rss_kb")
    memory_exceeded = (
        isinstance(memory_limit_mb, int)
        and not isinstance(memory_limit_mb, bool)
        and memory_limit_mb > 0
        and isinstance(peak_rss_kb, int)
        and not isinstance(peak_rss_kb, bool)
        and peak_rss_kb > memory_limit_mb * 1024
    )
    error = (
        {
            "code": "memory_limit_exceeded",
            "message": (
                f"Process group exceeded the {memory_limit_mb} MB job memory limit"
            ),
        }
        if memory_exceeded
        else {
            "code": "execution_process_missing",
            "message": "The submitted computation can no longer continue",
        }
    )
    status.update(
        {
            "status": "failed",
            "finished_at": _queue_now(),
            "supervisor_exit_pending": False,
            "process_group_activity": "empty",
            "error": error,
        }
    )
    if receipt_record is not None:
        receipt_path, receipt = receipt_record
        receipt.update(
            {
                "phase": "failed",
                "status": "failed",
                "updated_at": status["finished_at"],
                "error": error,
            }
        )
        _queue_atomic_write(receipt_path, receipt)
    # 先写 operation 回执，再发布 status 终态。若第二次写入中断，FIFO 仍在，
    # 下次核对会幂等补写；不能先让公开状态进入终态却留下 running 回执。
    _queue_atomic_write(status_path, status)
    return True


def _execution_job_status_activity(status: Mapping[str, Any]) -> str:
    """核对非 Gaussian 执行作业的 supervisor、child 和进程组活动状态。

    输入是已经通过 schema 和 scope 校验的 active status。只有完整身份明确显示
    supervisor 与 child 都已停止，且两者的进程组/会话为空或完整身份属于前一个 boot
    时才返回 ``dead``；任何身份缺失、权限错误或同一 boot 下扫描结果不确定都返回
    ``unknown``，由调用方继续保留资源占用。Gaussian 记录仍由 FIFO 专用恢复路径处理。
    """

    metadata = status.get("metadata") or {}
    if str(metadata.get("software_id") or "").casefold() == "gaussian":
        return "not_applicable"
    if str(status.get("status") or "").casefold() not in ACTIVE_JOB_STATES:
        return "terminal"
    for prefix in ("supervisor", "child"):
        if _persisted_process_activity(status, prefix) != "stopped":
            return "unknown"
    current_boot = capture_process_identity(os.getpid()).boot_id
    for prefix in ("supervisor", "child"):
        group_activity = _persisted_process_group_activity(status, prefix)
        if group_activity == "empty":
            continue
        raw_group = status.get(f"{prefix}_process_group_id")
        raw_session = status.get(f"{prefix}_session_id")
        persisted_boot = status.get(f"{prefix}_boot_id")
        previous_boot = bool(
            group_activity == "unknown"
            and isinstance(raw_group, int)
            and not isinstance(raw_group, bool)
            and raw_group > 0
            and isinstance(raw_session, int)
            and not isinstance(raw_session, bool)
            and raw_session > 0
            and isinstance(persisted_boot, str)
            and persisted_boot.strip()
            and isinstance(current_boot, str)
            and current_boot.strip()
            and persisted_boot != current_boot
        )
        # boot identity 改变时，旧 boot 的进程组不可能继续存在；仍要求完整
        # group/session 身份，避免把字段损坏误判为 worker 重启。
        if not previous_boot:
            return "unknown"
    return "dead"


def _mark_execution_job_supervisor_lost(
    job_directory: Path,
    status: Mapping[str, Any],
) -> None:
    """把可证明已停止的非 Gaussian 作业持久标记为失败，保留原目录和诊断资料。"""

    status_path = job_directory / "status.json"
    current_state, current = _read_queue_json_state(status_path)
    if current_state == "invalid" or current is None:
        return
    if str(current.get("status") or "").casefold() not in ACTIVE_JOB_STATES:
        return
    if _execution_job_status_activity(current) != "dead":
        return
    current.update(
        {
            "status": "failed",
            "finished_at": _queue_now(),
            "return_code": None,
            "error": {
                "code": "execution_process_missing",
                "message": "The submitted computation can no longer continue",
            },
        }
    )
    _queue_atomic_write(status_path, current)


def _remove_dead_gaussian_queue_entries() -> None:
    """清理明确失效的持有者，同时保护交接中的 native 和活 Gaussian。"""

    for path, value in _gaussian_queue_entries():
        if str(value.get("kind") or "") == "native":
            activity = _native_queue_job_activity(value)
            if activity in {"supervisor_alive", "child_alive", "child_group_alive"}:
                continue
            if activity == "dead":
                if _mark_native_job_supervisor_lost(value):
                    _unlink_private_file(path)
                continue
            if activity == "terminal":
                _unlink_private_file(path)
                continue
            if activity == "unknown":
                # status/claim 损坏或进程身份无法核对时保持 FIFO 占用；超时清理
                # 不能把不可判定状态降级成 staging，否则后续 ticket 会越过前序作业。
                continue
            handoff_at = value.get("handoff_at")
            if not handoff_at:
                submitter_pid = _queue_pid(value, "submitter_pid")
                if submitter_pid > 0 and _process_is_alive(submitter_pid):
                    continue
            if _native_queue_entry_age(value) >= GAUSSIAN_PENDING_GRACE_SECONDS:
                _unlink_private_file(path)
            continue
        # typed Gaussian 的服务进程可能已崩溃而 worker 仍在独立 session 中运行。
        # 新记录只允许在实际 worker 身份明确停止时清理；旧记录没有身份时保持
        # unknown，宁可阻塞后续 Gaussian，也不能按可能已被系统复用的旧服务进程 PID 放行。
        if _worker_identity_activity(value) != "stopped":
            continue
        _unlink_private_file(path)


def reconcile_stopped_native_gaussian_jobs() -> None:
    """核对退出中的 native Gaussian 作业，并释放已经明确停止的队列项。

    输入来自持久 Gaussian 队列和每个作业的进程身份。函数在全局队列锁内核对
    supervisor、child、进程组和会话；只有都明确停止时才写入失败终态并删除对应
    FIFO 项。身份损坏、权限不足或进程仍存在时保持原记录，调用方可以稍后重试。
    """

    with _gaussian_queue_lock():
        _remove_dead_gaussian_queue_entries()


@dataclass(frozen=True)
class GaussianQueueTicket:
    """一个服务内部 Gaussian FIFO 位置。"""

    ticket_id: int
    owner_id: str
    scope_id: str | None = None
    workspace: str | None = None
    release_deferred: bool = False

    def as_dict(self) -> dict[str, Any]:
        """返回私有 supervisor spec 所需的队列身份，不把字段直接作为 MCP 结果。"""

        value = {"ticket_id": self.ticket_id, "owner_id": self.owner_id}
        if self.scope_id is not None:
            value["scope_id"] = self.scope_id
        if self.workspace is not None:
            value["workspace"] = self.workspace
        return value

    def release(self) -> None:
        """幂等删除自己的队列条目，允许下一个作业继续。"""

        if self.release_deferred:
            return

        with _gaussian_queue_lock():
            path = _gaussian_queue_directory() / f"ticket_{self.ticket_id:020d}.json"
            if path.is_symlink() or (path.exists() and not path.is_file()):
                raise ResourceStateUnavailable(
                    "Gaussian queue ticket cannot be released safely"
                )
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                return
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ResourceStateUnavailable(
                    "Gaussian queue ticket cannot be released safely"
                ) from exc
            if not isinstance(value, dict):
                raise ResourceStateUnavailable(
                    "Gaussian queue ticket cannot be released safely"
            )
            if (
                value.get("ticket_id") == self.ticket_id
                and value.get("owner_id") == self.owner_id
                and value.get("scope_id") == self.scope_id
                and (
                    self.workspace is None
                    or str(value.get("workspace") or "") == str(self.workspace)
                )
            ):
                if not _unlink_private_file(path):
                    return

    def defer_release(self) -> None:
        """标记进程树无法核对，暂不释放 FIFO 位置，避免未知作业被越过。"""

        object.__setattr__(self, "release_deferred", True)


def enqueue_gaussian_job(
    owner_id: str,
    *,
    kind: str = "native",
    job_id: str | None = None,
) -> GaussianQueueTicket:
    """按接收顺序登记 Gaussian 作业，并复用同一归属身份的未完成登记。"""

    normalized_owner = str(owner_id).strip()
    if not normalized_owner:
        raise ValueError("Gaussian queue owner_id must be non-empty")
    normalized_kind = "native" if kind == "native" else "typed"
    scope = current_scope()
    scope_id = scope.scope_id if scope is not None else None
    scope_workspace = (
        str(scope.workspace) if scope is not None else str(_workspace_root())
    )
    if scope is not None and (
        not isinstance(job_id, str) or _SCOPED_JOB_ID.fullmatch(job_id) is None
    ):
        raise ResourceStateUnavailable(
            "Scoped Gaussian queue job id is required and must be path-safe"
        )
    with _gaussian_queue_lock():
        _remove_dead_gaussian_queue_entries()
        for _path, value in _gaussian_queue_entries():
            if (
                value.get("owner_id") == normalized_owner
                and value.get("scope_id") == scope_id
                and str(value.get("workspace") or "") == scope_workspace
            ):
                return GaussianQueueTicket(
                    ticket_id=int(value["ticket_id"]),
                    owner_id=normalized_owner,
                    scope_id=scope_id,
                    workspace=scope_workspace,
                )
        ticket_id = _next_gaussian_queue_ticket()
        path = _gaussian_queue_directory() / f"ticket_{ticket_id:020d}.json"
        _queue_atomic_write(
            path,
            {
                "schema_version": 1,
                "ticket_id": ticket_id,
                "owner_id": normalized_owner,
                "owner_pid": os.getpid() if normalized_kind == "typed" else None,
                "submitter_pid": os.getpid() if normalized_kind == "native" else None,
                "handoff_at": None,
                "kind": normalized_kind,
                "job_id": job_id,
                "scope_id": scope_id,
                "workspace": scope_workspace,
                "state": "queued",
                "created_at": _queue_now(),
            },
        )
    return GaussianQueueTicket(
        ticket_id=ticket_id,
        owner_id=normalized_owner,
        scope_id=scope_id,
        workspace=scope_workspace,
    )


def mark_gaussian_queue_handoff(ticket: GaussianQueueTicket) -> None:
    """记录 native supervisor spec 已发布，启动阶段可按 handoff 清理失效条目。"""

    with _gaussian_queue_lock():
        path = _gaussian_queue_directory() / f"ticket_{ticket.ticket_id:020d}.json"
        value = _read_queue_json(path)
        if value is None or value.get("owner_id") != ticket.owner_id:
            return
        value["handoff_at"] = _queue_now()
        _queue_atomic_write(path, value)


def bind_gaussian_queue_owner(
    ticket: GaussianQueueTicket,
    pid: int | None = None,
) -> None:
    """把 native 队列条目绑定到实际 supervisor 进程。"""

    selected_pid = os.getpid() if pid is None else int(pid)
    with _gaussian_queue_lock():
        path = _gaussian_queue_directory() / f"ticket_{ticket.ticket_id:020d}.json"
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ResourceStateUnavailable("Gaussian queue ticket is invalid")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return
        if (
            value.get("owner_id") != ticket.owner_id
            or value.get("scope_id") != ticket.scope_id
            or (
                ticket.workspace is not None
                and str(value.get("workspace") or "") != str(ticket.workspace)
            )
        ):
            return
        value["owner_pid"] = selected_pid
        identity = capture_process_identity(selected_pid)
        # 新绑定即使读取身份失败也要留下“不完整身份”标记；恢复逻辑不能把
        # 这个交接窗口误认作没有身份的 legacy 记录而按普通 PID 放行。
        value.update(
            {
                "owner_start_time_ticks": identity.start_time_ticks,
                "owner_start_ticks": identity.start_time_ticks,
                "owner_boot_id": identity.boot_id,
            }
        )
        _queue_atomic_write(path, value)


def bind_gaussian_queue_worker(
    ticket: GaussianQueueTicket,
    worker_identity: ProcessIdentity | Mapping[str, Any],
) -> bool:
    """把 typed Gaussian FIFO 绑定到实际 worker 的可核对身份。"""

    identity = _normalize_worker_identity(worker_identity)
    with _gaussian_queue_lock():
        path = _gaussian_queue_directory() / f"ticket_{ticket.ticket_id:020d}.json"
        value = _read_queue_json(path)
        if value is None:
            return False
        if (
            value.get("owner_id") != ticket.owner_id
            or value.get("scope_id") != ticket.scope_id
            or (
                ticket.workspace is not None
                and str(value.get("workspace") or "") != str(ticket.workspace)
            )
            or str(value.get("kind") or "") != "typed"
        ):
            return False
        value.update(
            {
                "worker_pid": identity["pid"],
                "worker_start_time_ticks": identity["start_time_ticks"],
                "worker_start_ticks": identity["start_time_ticks"],
                "worker_boot_id": identity["boot_id"],
                "worker_status": "running",
                "worker_started_at": _queue_now(),
            }
        )
        _queue_atomic_write(path, value)
        return True


def wait_for_gaussian_turn(
    ticket: GaussianQueueTicket,
    *,
    should_cancel: Any = None,
    poll_seconds: float = GAUSSIAN_QUEUE_POLL_SECONDS,
) -> bool:
    """等待 ticket 成为队首；取消时删除自身条目并返回 False。"""

    delay = max(0.01, float(poll_seconds))
    while True:
        if should_cancel is not None and bool(should_cancel()):
            ticket.release()
            return False
        with _gaussian_queue_lock():
            _remove_dead_gaussian_queue_entries()
            path = _gaussian_queue_directory() / f"ticket_{ticket.ticket_id:020d}.json"
            if path.is_symlink() or (path.exists() and not path.is_file()):
                raise ResourceStateUnavailable("Gaussian queue ticket is invalid")
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (FileNotFoundError, OSError, json.JSONDecodeError):
                return False
            if (
                value.get("owner_id") != ticket.owner_id
                or value.get("scope_id") != ticket.scope_id
                or (
                    ticket.workspace is not None
                    and str(value.get("workspace") or "") != str(ticket.workspace)
                )
            ):
                return False
            entries = _gaussian_queue_entries()
            if entries and int(entries[0][1]["ticket_id"]) == ticket.ticket_id:
                value.update({"state": "running", "started_at": _queue_now()})
                _queue_atomic_write(path, value)
                return True
        time.sleep(delay)


@contextmanager
def gaussian_execution_slot(
    owner_id: str | None = None,
    *,
    job_id: str | None = None,
) -> Iterator[GaussianQueueTicket]:
    """为同步 typed Gaussian Action 排队，并绑定可核对的 scoped job id。"""

    ticket = enqueue_gaussian_job(
        owner_id or f"typed_{uuid.uuid4().hex}",
        kind="typed",
        job_id=job_id,
    )
    bind_gaussian_queue_owner(ticket)
    try:
        if not wait_for_gaussian_turn(ticket):
            raise RuntimeError("Gaussian queue entry disappeared before execution")
        yield ticket
    finally:
        ticket.release()


def _zero_resources() -> dict[str, int]:
    return {"cpu_cores": 0, "memory_mb": 0, "gpu_count": 0}


def _add_resources(total: dict[str, int], resources: Mapping[str, Any]) -> None:
    normalized = _validate_persisted_resource_limits(resources, "active resource record")
    for name in total:
        total[name] += normalized[name]


def _job_counts_against_resource_budget(status: Mapping[str, Any]) -> bool:
    """判断持久化作业是否已经实际占用服务资源。"""

    state = str(status.get("status", "")).casefold()
    if state not in ACTIVE_JOB_STATES:
        return False
    metadata = status.get("metadata") or {}
    # queued Gaussian 只是 FIFO 等待项，没有 reservation；队首 supervisor
    # 建立 reservation 后会单独从 reservations 记录中计入。
    return not (
        state == "queued"
        and str(metadata.get("software_id") or "").casefold() == "gaussian"
    )


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def _validate_persisted_resource_limits(value: Any, description: str) -> dict[str, int]:
    """严格读取持久资源额度，禁止损坏记录被默认值掩盖。"""

    if not isinstance(value, Mapping):
        raise ResourceStateUnavailable(f"{description} resource limits are invalid")
    result: dict[str, int] = {}
    for name in ("cpu_cores", "memory_mb", "gpu_count"):
        raw = value.get(name)
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise ResourceStateUnavailable(f"{description} resource limits are invalid")
        result[name] = raw
    if result["cpu_cores"] < 1 or result["memory_mb"] < 1:
        raise ResourceStateUnavailable(f"{description} resource limits are invalid")
    budget = evaluation_resource_budget()
    if any(
        result[name] > int(getattr(budget, name))
        for name in ("cpu_cores", "memory_mb", "gpu_count")
    ):
        raise ResourceStateUnavailable(
            f"{description} resource limits exceed the configured budget"
        )
    return result


def _read_persisted_allocation_ids(
    value: Any,
    description: str,
) -> tuple[list[int], list[str]]:
    """读取持久 CPU/GPU 列表并完成类型核验，不检查资源池归属。"""

    if not isinstance(value, Mapping):
        raise ResourceStateUnavailable(f"{description} resource allocation is invalid")
    raw_cpu = value.get("cpu_ids", [])
    raw_gpu = value.get("gpu_ids", [])
    if not isinstance(raw_cpu, list) or not isinstance(raw_gpu, list):
        raise ResourceStateUnavailable(f"{description} resource allocation is invalid")
    cpu_ids: list[int] = []
    for item in raw_cpu:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ResourceStateUnavailable(f"{description} CPU allocation is invalid")
        cpu_ids.append(item)
    gpu_ids: list[str] = []
    for item in raw_gpu:
        if not isinstance(item, str) or not item.strip():
            raise ResourceStateUnavailable(f"{description} GPU allocation is invalid")
        gpu_ids.append(item)
    return cpu_ids, gpu_ids


def _validate_persisted_allocation(
    value: Any,
    description: str,
    *,
    resource_limits: Mapping[str, int] | None = None,
) -> dict[str, list[Any]]:
    """严格读取持久 CPU/GPU 分配，并核对真实资源池和请求数量。"""

    cpu_ids, gpu_ids = _read_persisted_allocation_ids(value, description)
    if len(set(cpu_ids)) != len(cpu_ids) or len(set(gpu_ids)) != len(gpu_ids):
        raise ResourceStateUnavailable(f"{description} resource allocation is duplicated")
    budget = evaluation_resource_budget()
    cpu_pool = set(_cpu_pool(budget))
    gpu_pool = set(_gpu_pool(budget))
    if not set(cpu_ids).issubset(cpu_pool) or not set(gpu_ids).issubset(gpu_pool):
        raise ResourceStateUnavailable(
            f"{description} resource allocation is outside the configured resource pool"
        )
    if resource_limits is not None:
        expected = _validate_persisted_resource_limits(
            resource_limits, f"{description} allocation limits"
        )
        if len(cpu_ids) != expected["cpu_cores"] or len(gpu_ids) != expected["gpu_count"]:
            raise ResourceStateUnavailable(
                f"{description} resource allocation count does not match limits"
            )
    return {"cpu_ids": cpu_ids, "gpu_ids": gpu_ids}


def _require_reservation_object(value: Any) -> dict[str, Any]:
    """确认 reservation 是可继续校验的 JSON 对象，失败时拒绝资源准入。"""

    if not isinstance(value, dict):
        raise ResourceStateUnavailable("resource reservation is not an object")
    return value


def _validate_reservation_header(value: Mapping[str, Any]) -> None:
    """核对 reservation 的版本、进程和业务标签，损坏时立即失败。"""

    schema_version = value.get("schema_version", 1)
    if isinstance(schema_version, bool) or schema_version != 1:
        raise ResourceStateUnavailable("resource reservation schema is invalid")
    _require_positive_integer(value.get("pid"), "resource reservation pid")
    for field in ("kind", "label"):
        if not isinstance(value.get(field), str) or not value[field].strip():
            raise ResourceStateUnavailable(f"resource reservation {field} is invalid")


def _read_reservation_scope_identity(
    value: Mapping[str, Any],
) -> tuple[str | None, str | None]:
    """读取 scope 与 workspace 字段，要求它们成对出现且非空。"""

    scope_id = value.get("scope_id")
    workspace = value.get("workspace")
    if scope_id is not None and (
        not isinstance(scope_id, str)
        or not scope_id.strip()
        or not isinstance(workspace, str)
        or not workspace.strip()
    ):
        raise ResourceStateUnavailable("resource reservation scope is invalid")
    return scope_id, workspace


def _resolve_reservation_scheduler_root() -> Path | None:
    """取得当前 scope 或安全配置中的 scheduler 目录，路径不可信时失败。"""

    scoped = current_scope()
    scheduler = scoped.scheduler_root if scoped is not None else None
    if scheduler is not None:
        return scheduler
    configured = os.environ.get(SHARED_SCHEDULER_ROOT_ENV, "").strip()
    if not configured:
        return None
    try:
        return _validate_private_directory_path(
            Path(configured).expanduser(),
            create=False,
        )
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        raise ResourceStateUnavailable(
            "Shared MiniChem scheduler root is unavailable"
        ) from exc


def _validate_reservation_scope_reference(value: Mapping[str, Any]) -> None:
    """核对 scoped reservation 的作业编号和注册 workspace 身份。"""

    scope_id, workspace = _read_reservation_scope_identity(value)
    if scope_id is None:
        return
    job_id = value.get("job_id")
    if job_id is not None and (
        not isinstance(job_id, str) or _SCOPED_JOB_ID.fullmatch(job_id) is None
    ):
        raise ResourceStateUnavailable("resource reservation job id is not path-safe")
    _validate_registered_scope_workspace(
        _resolve_reservation_scheduler_root(),
        scope_id,
        workspace,
    )


def _validate_typed_worker_identity(value: Mapping[str, Any]) -> None:
    """核对 typed Gaussian worker 的进程、启动时钟和 boot-id。"""

    for field in ("worker_pid", "worker_boot_id"):
        if field not in value:
            raise ResourceStateUnavailable("resource reservation worker identity is incomplete")
    _require_positive_integer(value.get("worker_pid"), "resource worker pid")
    _require_positive_integer(
        value.get("worker_start_time_ticks", value.get("worker_start_ticks")),
        "resource worker start ticks",
    )
    if not isinstance(value.get("worker_boot_id"), str) or not value["worker_boot_id"].strip():
        raise ResourceStateUnavailable("resource worker boot id is invalid")


def _validate_reservation_worker_tracking(value: Mapping[str, Any]) -> None:
    """核对 worker_tracking 类型，并完整核验 typed worker 身份。"""

    tracking = value.get("worker_tracking")
    if tracking is not None and not isinstance(tracking, str):
        raise ResourceStateUnavailable("resource reservation worker tracking is invalid")
    if str(tracking or "").strip().casefold() == "gaussian_typed":
        # reservation 创建到 worker 启动之间仍占用资源，身份缺失不能按零资源处理。
        _validate_typed_worker_identity(value)


def _validate_persisted_reservation(path: Path, value: Any) -> dict[str, Any]:
    """严格核对 reservation JSON；不完整记录必须阻止整个资源准入。"""

    del path
    record = _require_reservation_object(value)
    _validate_reservation_header(record)
    record["resource_limits"] = _validate_persisted_resource_limits(
        record.get("resource_limits"), "resource reservation"
    )
    record["resource_allocation"] = _validate_persisted_allocation(
        record.get("resource_allocation"),
        "resource reservation",
        resource_limits=record["resource_limits"],
    )
    _validate_reservation_scope_reference(record)
    _validate_reservation_worker_tracking(record)
    return record


def _live_resource_reservations() -> list[tuple[Path, dict[str, Any]]]:
    """读取仍由存活进程持有的 reservation，并清理已失效文件。"""

    records: list[tuple[Path, dict[str, Any]]] = []
    reservations = _budget_directory() / "reservations"
    if not reservations.exists():
        return records
    if reservations.is_symlink() or not reservations.is_dir():
        raise ResourceStateUnavailable("resource reservation directory is invalid")
    try:
        children = sorted(reservations.iterdir(), key=lambda item: item.name)
    except OSError as exc:
        raise ResourceStateUnavailable("resource reservation directory cannot be read") from exc
    for path in children:
        if (
            re.fullmatch(r"reservation_[A-Za-z0-9._-]+\.json", path.name) is None
            or path.is_symlink()
            or not path.is_file()
        ):
            raise ResourceStateUnavailable("resource reservation contains an unknown entry")
        try:
            reservation = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
            raise ResourceStateUnavailable("resource reservation cannot be read") from exc
        reservation = _validate_persisted_reservation(path, reservation)
        pid = int(reservation["pid"])
        if _reservation_requires_worker_identity(reservation):
            # Gaussian typed reservation 不再信任服务进程 PID；worker 身份缺失、
            # 启动时钟无法读取或状态无法核对时都保留资源，防止恢复窗口放行下一作业。
            if _worker_identity_activity(reservation) != "stopped":
                records.append((path, reservation))
            else:
                if not _unlink_private_file(path):
                    raise ResourceStateUnavailable("resource reservation disappeared")
            continue
        if pid > 0 and _process_exists(pid):
            records.append((path, reservation))
        else:
            if not _unlink_private_file(path):
                raise ResourceStateUnavailable("resource reservation disappeared")
    return records


def _reservation_requires_worker_identity(value: Mapping[str, Any]) -> bool:
    """判断 reservation 是否是 typed Gaussian，兼容旧记录的可辨识标签。"""

    tracking = str(value.get("worker_tracking") or "").strip().casefold()
    if tracking:
        return tracking == "gaussian_typed"
    # 旧版本没有 tracking 字段，但同步 Gaussian 服务使用 action/gaussian 标签；
    # 只对这个明确后缀启用 fail-closed，非 Gaussian typed 仍沿用原服务进程 PID 兼容路径。
    if str(value.get("kind") or "") != "predefined_action":
        return False
    label = str(value.get("label") or "").strip().casefold()
    return label.rsplit("/", 1)[-1] == "gaussian"


def _status_covered_by_gaussian_reservation(
    status: Mapping[str, Any],
    reservations: list[tuple[Path, Mapping[str, Any]]],
) -> bool:
    """判断 Gaussian 状态是否已有同 job_id 的 live supervisor reservation。"""

    metadata = status.get("metadata") or {}
    job_id = str(status.get("job_id") or "").strip()
    if (
        str(metadata.get("software_id") or "").casefold() != "gaussian"
        or not job_id
    ):
        return False
    status_scope = (
        str(status.get("scope_id") or "").strip(),
        str(status.get("scope_workspace") or "").strip(),
    )
    for _path, reservation in reservations:
        if str(reservation.get("job_id") or "").strip() != job_id:
            continue
        reservation_scope = (
            str(reservation.get("scope_id") or "").strip(),
            str(reservation.get("workspace") or "").strip(),
        )
        # 新记录必须同时绑定 scope；legacy 记录可能只有 workspace 或完全没有
        # scope 元数据。两边都没有 scope_id 时按同一 legacy 作业身份覆盖，避免
        # 一个 native Gaussian 同时按 status 和 reservation 重复计资源。
        if status_scope[0] and reservation_scope[0]:
            matches = status_scope == reservation_scope
        else:
            matches = not status_scope[0] and not reservation_scope[0]
        if matches:
            return True
    return False


def _status_owned_by_scope(
    status: Mapping[str, Any], workspace: Path, scope: Any
) -> bool:
    """判断持久 status 是否属于当前 scope；认证视图缺少绑定字段即拒绝归属。"""

    if scope is None:
        return True
    return (
        str(status.get("scope_id") or "") == scope.scope_id
        and str(status.get("scope_workspace") or "") == str(workspace)
        and str(workspace) == str(scope.workspace)
    )


def _execution_job_directories(root: Path) -> list[Path]:
    """读取一个 workspace 的执行作业目录，拒绝未知项和锁符号链接。"""

    jobs = root / "outputs" / "execution_jobs"
    if jobs.is_symlink():
        raise ResourceStateUnavailable("execution job directory is invalid")
    if not jobs.exists():
        return []
    if not jobs.is_dir():
        raise ResourceStateUnavailable("execution job directory is invalid")
    try:
        children = sorted(jobs.iterdir(), key=lambda item: item.name)
    except OSError as exc:
        raise ResourceStateUnavailable("execution job directory cannot be read") from exc
    result: list[Path] = []
    for child in children:
        if re.fullmatch(r"\.[0-9a-f]{64}\.lock", child.name):
            if child.is_symlink() or not child.is_file():
                raise ResourceStateUnavailable("execution job lock is invalid")
            continue
        if (
            re.fullmatch(r"job[-_][A-Za-z0-9._-]{1,127}", child.name) is None
            or child.is_symlink()
            or not child.is_dir()
        ):
            raise ResourceStateUnavailable("execution job directory contains an unknown entry")
        result.append(child)
    return result


def _validate_execution_job_scope(
    root: Path, status: Mapping[str, Any]
) -> None:
    """核对活动执行作业的 scope/workspace 绑定和路径归属。"""

    scope_id = status.get("scope_id")
    scope_workspace = status.get("scope_workspace")
    if (scope_id is None) != (scope_workspace is None):
        raise ResourceStateUnavailable("execution job scope is incomplete")
    if scope_id is None:
        return
    if (
        not isinstance(scope_id, str)
        or not scope_id.strip()
        or not isinstance(scope_workspace, str)
        or not scope_workspace.strip()
    ):
        raise ResourceStateUnavailable("execution job scope is invalid")
    persisted_job_id = status.get("job_id")
    if persisted_job_id is not None and (
        not isinstance(persisted_job_id, str)
        or _SCOPED_JOB_ID.fullmatch(persisted_job_id) is None
    ):
        raise ResourceStateUnavailable("execution job id is not path-safe")
    scoped = current_scope()
    scheduler = scoped.scheduler_root if scoped is not None else _configured_scheduler_root()
    _validate_registered_scope_workspace(scheduler, scope_id, scope_workspace)
    try:
        scope_root = _validate_private_directory_path(
            Path(scope_workspace).expanduser(),
            create=False,
        )
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        raise ResourceStateUnavailable(
            "execution job scope workspace is unavailable"
        ) from exc
    if scope_root != root:
        raise ResourceStateUnavailable("execution job scope workspace does not match its root")


def _read_execution_job_status_payload(
    path: Path,
    job_directory: Path,
) -> dict[str, Any]:
    """读取执行作业 status，并核对状态、作业身份和 metadata 类型。"""

    if path.is_symlink() or not path.is_file():
        raise ResourceStateUnavailable("execution job status is missing or invalid")
    try:
        status = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        raise ResourceStateUnavailable("execution job status cannot be read") from exc
    if not isinstance(status, dict):
        raise ResourceStateUnavailable("execution job status is not an object")
    state = status.get("status")
    terminal_states = {"success", "failed", "timeout", "cancelled"}
    if state not in ACTIVE_JOB_STATES and state not in terminal_states:
        raise ResourceStateUnavailable("execution job status is invalid")
    persisted_job_id = status.get("job_id")
    if persisted_job_id is not None and persisted_job_id != job_directory.name:
        raise ResourceStateUnavailable("execution job identity is invalid")
    metadata = status.get("metadata")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise ResourceStateUnavailable("execution job metadata is invalid")
    return status


def _validate_active_execution_job_resources(status: dict[str, Any]) -> None:
    """为 active 作业核对资源额度；queued Gaussian 尚未分配资源。"""

    status["resource_limits"] = _validate_persisted_resource_limits(
        status.get("resource_limits"), "execution job"
    )
    if _job_counts_against_resource_budget(status):
        status["resource_allocation"] = _validate_persisted_allocation(
            status.get("resource_allocation"),
            "execution job",
            resource_limits=status["resource_limits"],
        )


def _read_execution_job_status(
    root: Path,
    job_directory: Path,
    reservations: list[tuple[Path, Mapping[str, Any]]],
) -> dict[str, Any] | None:
    """读取并核对一个执行作业；不计入预算的终态返回 None。"""

    status = _read_execution_job_status_payload(
        job_directory / "status.json", job_directory
    )
    state = status.get("status")
    _validate_execution_job_scope(root, status)
    # worker 重启后 CPU affinity 或 GPU 可见集合可能已经变化。先按持久进程
    # 身份核对已停止的非 Gaussian 作业，避免旧资源编号在回收前触发池边界错误。
    if state in ACTIVE_JOB_STATES and _execution_job_status_activity(status) == "dead":
        _mark_execution_job_supervisor_lost(job_directory, status)
        return None
    if state in ACTIVE_JOB_STATES:
        _validate_active_execution_job_resources(status)
    if not _job_counts_against_resource_budget(status):
        return None
    if _status_covered_by_gaussian_reservation(status, reservations):
        return None
    return status


def _active_execution_job_statuses(
    reservations: list[tuple[Path, Mapping[str, Any]]],
) -> list[tuple[Path, dict[str, Any]]]:
    """读取所有已登记 scope 的 active status，供预算和脱敏视图共同使用。"""

    values: list[tuple[Path, dict[str, Any]]] = []
    for root in _execution_job_roots():
        for job_directory in _execution_job_directories(root):
            status = _read_execution_job_status(root, job_directory, reservations)
            if status is not None:
                values.append((root, status))
    return values


def active_resource_usage() -> dict[str, int]:
    total = _zero_resources()
    live_reservations = _live_resource_reservations()
    for _root, status in _active_execution_job_statuses(live_reservations):
        _add_resources(total, status["resource_limits"])
    for _path, reservation in live_reservations:
        _add_resources(total, reservation["resource_limits"])
    return total


def _execution_job_resource_record(
    root: Path, status: Mapping[str, Any], scope: Any
) -> dict[str, Any]:
    """把活动执行作业转换为当前 scope 可见的资源记录。"""

    state = str(status.get("status", "")).casefold()
    if scope is not None and not _status_owned_by_scope(status, root, scope):
        return {"record_type": "execution_job", "status": state, "owned": False}
    record: dict[str, Any] = {
        "record_type": "execution_job",
        "job_id": status.get("job_id") or "",
        "job_type": status.get("job_type"),
        "status": state,
        "label": (status.get("metadata") or {}).get("label"),
        "resource_limits": normalize_resource_limits(status.get("resource_limits") or {}),
        "resource_allocation": dict(status.get("resource_allocation") or {}),
        "submitted_at": status.get("submitted_at"),
        "supervisor_pid": status.get("supervisor_pid"),
        "release_condition": "job reaches success, failed, timeout, or cancelled",
    }
    if scope is not None:
        record["owned"] = True
    return record


def _reservation_resource_record(
    path: Path, reservation: Mapping[str, Any], scope: Any
) -> dict[str, Any]:
    """把活动临时 reservation 转换为当前 scope 可见的资源记录。"""

    owned = scope is not None and (
        str(reservation.get("scope_id") or "") == scope.scope_id
        and str(reservation.get("workspace") or "").strip() == str(scope.workspace)
    )
    if scope is not None and not owned:
        return {"record_type": "submission_reservation", "status": "running", "owned": False}
    record: dict[str, Any] = {
        "record_type": "submission_reservation",
        "reservation_id": path.stem,
        "pid": int(reservation.get("pid") or 0),
        "job_type": reservation.get("kind"),
        "label": reservation.get("label"),
        "resource_limits": normalize_resource_limits(reservation.get("resource_limits") or {}),
        "resource_allocation": dict(reservation.get("resource_allocation") or {}),
        "release_condition": "the in-progress submission creates its job record or fails",
    }
    if scope is not None:
        record.update({"owned": True, "job_id": reservation.get("job_id")})
    return record


def _append_scope_gaussian_queue_records(
    records: list[dict[str, Any]],
    live_reservations: list[tuple[Path, Mapping[str, Any]]],
    scope: Any,
) -> None:
    """把 scope 认证视图中尚未被 reservation 覆盖的 Gaussian FIFO 加入结果。"""

    with _gaussian_queue_lock():
        queue_entries = _gaussian_queue_entries()
    known_job_ids = {str(item["job_id"]) for item in records if item.get("job_id")}
    reserved_job_ids = {
        str(value["job_id"])
        for _path, value in live_reservations
        if value.get("job_id")
    }
    for position, (_path, value) in enumerate(queue_entries, start=1):
        job_id = str(value.get("job_id") or "")
        owned = (
            str(value.get("scope_id") or "") == scope.scope_id
            and str(value.get("workspace") or "") == str(scope.workspace)
        )
        if job_id in reserved_job_ids or (owned and job_id in known_job_ids):
            continue
        item: dict[str, Any] = {
            "record_type": "gaussian_queue",
            "position": position,
            "status": str(value.get("state") or "queued"),
            "owned": owned,
        }
        if owned and job_id:
            item["job_id"] = job_id
        records.append(item)


def active_resource_jobs() -> list[dict[str, Any]]:
    """返回占用任务预算的活动作业和临时资源预留。"""

    records: list[dict[str, Any]] = []
    live_reservations = _live_resource_reservations()
    scope = current_scope()
    for root, status in _active_execution_job_statuses(live_reservations):
        records.append(_execution_job_resource_record(root, status, scope))
    for path, reservation in sorted(live_reservations, key=lambda item: item[0].name):
        records.append(_reservation_resource_record(path, reservation, scope))
    if scope is not None:
        _append_scope_gaussian_queue_records(records, live_reservations, scope)
    return records


@contextmanager
def _budget_lock() -> Iterator[None]:
    lock_path = _budget_directory() / "quota.lock"
    descriptor = _open_private_lock(lock_path)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


@dataclass
class ResourceReservation:
    path: Path
    resource_limits: dict[str, int]
    resource_allocation: dict[str, Any]
    worker_tracking: str | None = None
    identity: dict[str, Any] | None = None

    def release(self) -> None:
        """在配额锁内复核原始记录身份后删除，并同步 reservations 目录。"""

        with _budget_lock():
            state, value = _read_queue_json_state(self.path)
            if state == "missing":
                return
            if state != "valid" or value is None:
                raise ResourceStateUnavailable("resource reservation cannot be released safely")
            expected = self.identity or {
                "pid": os.getpid(),
                "resource_limits": self.resource_limits,
                "resource_allocation": self.resource_allocation,
                "worker_tracking": self.worker_tracking,
            }
            for field, expected_value in expected.items():
                if value.get(field) != expected_value:
                    raise ResourceStateUnavailable(
                        "resource reservation identity changed before release"
                    )
            if not _unlink_private_file(self.path):
                return

    def bind_worker(
        self,
        worker_identity: ProcessIdentity | Mapping[str, Any],
    ) -> bool:
        """持久记录 typed Gaussian worker 身份，失败时保留未知占用。"""

        if self.worker_tracking != "gaussian_typed":
            return False
        identity = _normalize_worker_identity(worker_identity)
        with _budget_lock():
            value = _read_queue_json(self.path)
            if value is None:
                return False
            expected = self.identity or {
                "pid": os.getpid(),
                "resource_limits": self.resource_limits,
                "resource_allocation": self.resource_allocation,
                "worker_tracking": self.worker_tracking,
            }
            if any(value.get(field) != expected_value for field, expected_value in expected.items()):
                raise ResourceStateUnavailable(
                    "resource reservation identity changed before worker binding"
                )
            value.update(
                {
                    "worker_pid": identity["pid"],
                    "worker_start_time_ticks": identity["start_time_ticks"],
                    "worker_start_ticks": identity["start_time_ticks"],
                    "worker_boot_id": identity["boot_id"],
                    "worker_status": "running",
                    "worker_started_at": _queue_now(),
                }
            )
            _queue_atomic_write(self.path, value)
            return True


def _cpu_pool(budget: ResourceBudget) -> list[int]:
    if hasattr(os, "sched_getaffinity"):
        allowed = sorted(int(item) for item in os.sched_getaffinity(0))
    else:
        allowed = list(range(os.cpu_count() or budget.cpu_cores))
    if len(allowed) < budget.cpu_cores:
        raise RuntimeError(
            "The evaluator CPU budget exceeds the process CPU affinity: "
            f"budget={budget.cpu_cores}, affinity={allowed}"
        )
    return allowed[: budget.cpu_cores]


def _gpu_pool(budget: ResourceBudget) -> list[str]:
    configured = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if configured:
        values = [item.strip() for item in configured.split(",") if item.strip()]
    else:
        values = [str(index) for index in range(budget.gpu_count)]
    if len(values) < budget.gpu_count:
        raise RuntimeError(
            "The evaluator GPU budget exceeds CUDA_VISIBLE_DEVICES: "
            f"budget={budget.gpu_count}, devices={values}"
        )
    return values[: budget.gpu_count]


def _active_allocations() -> tuple[set[int], set[str]]:
    used_cpu_ids: set[int] = set()
    used_gpu_ids: set[str] = set()
    live_reservations = _live_resource_reservations()
    records = [
        {
            "resource_allocation": status.get("resource_allocation") or {},
            "resource_limits": status.get("resource_limits") or {},
        }
        for _root, status in _active_execution_job_statuses(live_reservations)
    ]
    # active_resource_jobs 对其他 scope 做了脱敏，资源分配仍必须从全量内部
    # reservation 读取，否则两个 scope 可能错误地分配同一 CPU/GPU。
    for _path, reservation in live_reservations:
        records.append(
            {
                "resource_allocation": reservation.get("resource_allocation") or {},
                "resource_limits": reservation.get("resource_limits") or {},
            }
        )
    for record in records:
        requested = _validate_persisted_resource_limits(
            record.get("resource_limits"), "active resource record"
        )
        allocation = _validate_persisted_allocation(
            record.get("resource_allocation"),
            "active resource record",
            resource_limits=requested,
        )
        cpu_ids = allocation["cpu_ids"]
        gpu_ids = allocation["gpu_ids"]
        if used_cpu_ids.intersection(cpu_ids) or used_gpu_ids.intersection(gpu_ids):
            raise ResourceStateUnavailable(
                "active resource records contain overlapping CPU/GPU allocations"
            )
        used_cpu_ids.update(cpu_ids)
        used_gpu_ids.update(gpu_ids)
    return used_cpu_ids, used_gpu_ids


def reserve_resources(
    resources: Mapping[str, Any],
    *,
    kind: str,
    label: str,
    job_id: str | None = None,
    worker_tracking: str | None = None,
) -> ResourceReservation:
    requested = normalize_resource_limits(resources)
    budget = evaluation_resource_budget()
    validate_resource_limits(requested, budget=budget)
    with _budget_lock():
        reserved = active_resource_usage()
        if any(
            reserved[name] + requested[name] > int(getattr(budget, name))
            for name in requested
        ):
            raise ResourceBudgetExceeded(
                requested=requested,
                budget=budget,
                reserved=reserved,
                aggregate=True,
            )
        used_cpu_ids, used_gpu_ids = _active_allocations()
        available_cpu_ids = [
            item for item in _cpu_pool(budget) if item not in used_cpu_ids
        ]
        available_gpu_ids = [
            item for item in _gpu_pool(budget) if item not in used_gpu_ids
        ]
        if len(available_cpu_ids) < requested["cpu_cores"] or len(
            available_gpu_ids
        ) < requested["gpu_count"]:
            raise ResourceBudgetExceeded(
                requested=requested,
                budget=budget,
                reserved=reserved,
                aggregate=True,
            )
        allocation = {
            "cpu_ids": available_cpu_ids[: requested["cpu_cores"]],
            "gpu_ids": available_gpu_ids[: requested["gpu_count"]],
        }
        directory = _budget_directory() / "reservations"
        _ensure_private_directory(directory)
        path = directory / f"reservation_{uuid.uuid4().hex}.json"
        scope = current_scope()
        reservation_record = {
            "schema_version": 1,
            "pid": os.getpid(),
            "kind": kind,
            "label": label,
            "job_id": job_id,
            "scope_id": scope.scope_id if scope is not None else None,
            "workspace": (
                str(scope.workspace)
                if scope is not None
                else str(_workspace_root())
            ),
            "resource_limits": requested,
            "resource_allocation": allocation,
            "worker_tracking": worker_tracking,
        }
        _queue_atomic_write(
            path,
            reservation_record,
        )
    return ResourceReservation(
        path=path,
        resource_limits=requested,
        resource_allocation=allocation,
        worker_tracking=worker_tracking,
        identity={
            field: reservation_record[field]
            for field in (
                "pid",
                "kind",
                "label",
                "job_id",
                "scope_id",
                "workspace",
                "resource_limits",
                "resource_allocation",
                "worker_tracking",
            )
        },
    )


__all__ = [
    "AVAILABLE_CPU_CORES_ENV",
    "AVAILABLE_GPU_COUNT_ENV",
    "AVAILABLE_MEMORY_MB_ENV",
    "DEFAULT_AVAILABLE_CPU_CORES",
    "DEFAULT_AVAILABLE_GPU_COUNT",
    "DEFAULT_AVAILABLE_MEMORY_MB",
    "DEFAULT_REQUEST_CPU_CORES",
    "DEFAULT_REQUEST_GPU_COUNT",
    "DEFAULT_REQUEST_MEMORY_MB",
    "GAUSSIAN_CPU_CORES_ENV",
    "GAUSSIAN_MEMORY_MB_ENV",
    "GaussianResourcePolicy",
    "GaussianQueueTicket",
    "MINIMUM_GAUSSIAN_MEMORY_MB",
    "ResourceBudget",
    "ResourceBudgetExceeded",
    "ResourceStateUnavailable",
    "ResourceReservation",
    "apply_gaussian_resource_policy",
    "active_resource_usage",
    "active_resource_jobs",
    "evaluation_resource_budget",
    "gaussian_resource_policy",
    "gaussian_resource_policy_record",
    "gaussian_execution_slot",
    "enqueue_gaussian_job",
    "bind_gaussian_queue_owner",
    "bind_gaussian_queue_worker",
    "mark_gaussian_queue_handoff",
    "wait_for_gaussian_turn",
    "gaussian_serial_gate",
    "gaussian_serial_gate_path",
    "normalize_resource_limits",
    "reserve_resources",
    "resource_budget_record",
    "read_gaussian_queue_snapshot",
    "reconcile_stopped_native_gaussian_jobs",
    "validate_resource_limits",
]
