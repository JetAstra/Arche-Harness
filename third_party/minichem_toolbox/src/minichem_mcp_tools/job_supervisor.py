"""Standalone persistent supervisor for one chemistry execution job.

This module intentionally uses only the Python standard library.  The MCP server
starts it in a detached process group and passes a trusted, server-generated JSON
specification.  It never parses shell syntax and always launches an argv vector.
"""

from __future__ import annotations

import ctypes
import hashlib
import fcntl
import json
import os
import re
import resource
import signal
import subprocess
import sys
import stat
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

from minichem_toolbox.resource_budget import (
    GaussianQueueTicket,
    ResourceBudgetExceeded,
    bind_gaussian_queue_owner,
    reserve_resources,
    wait_for_gaussian_turn,
)
from minichem_toolbox.process_identity import (
    ProcessIdentity,
    capture_process_identity,
    inspect_process_group_membership,
    process_identity_activity,
)
from minichem_toolbox.environment_layout import STANDARD_ARCHE_PYTHON
from minichem_toolbox.gaussian_termination import (
    gaussian_archive_only_completion,
    gaussian_archive_warning_provenance,
)


COMPUTE_HARD_TIMEOUT_ENV = "RESEARCHCHEMBENCH_COMPUTE_HARD_TIMEOUT_SECONDS"
DEFAULT_COMPUTE_HARD_TIMEOUT_SECONDS = 7 * 24 * 60 * 60
UNVERIFIED_CHILD_GRACE_SECONDS = 2.0
PRIVATE_DIRECTORY_MODE = 0o700
PRIVATE_FILE_MODE = 0o600
TERMINAL_JOB_STATES = frozenset({"success", "failed", "timeout", "cancelled"})

# 监督进程只向可信的科学程序传递运行时需要的环境。带有 secret/key/token
# 等标记的名称默认拒绝，避免服务进程的认证配置被作业读取；可信 spec 可以
# 通过 child_environment 显式注入单个值，供 Gaussian 等本地程序使用。
CHILD_ENVIRONMENT_NAMES = frozenset(
    {
        "PATH",
        "LD_LIBRARY_PATH",
        "PYTHONPATH",
        "PYTHONNOUSERSITE",
        "PYTHONDONTWRITEBYTECODE",
        "HOME",
        "TMPDIR",
        "MPLCONFIGDIR",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TZ",
        "TERM",
        "CUDA_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "SLURM_JOB_ID",
        "SLURM_JOB_NODELIST",
        "SLURM_CPUS_PER_TASK",
        "SLURM_GPUS",
        "LM_LICENSE_FILE",
        "MLM_LICENSE_FILE",
        "OMP_NUM_THREADS",
        "OMP_STACKSIZE",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "KMP_STACKSIZE",
        "OMPI_MCA_hwloc_base_cpu_list",
        "PRTE_MCA_hwloc_default_cpu_list",
        "MINICHEM_BACKEND_RUNTIME",
        "MINICHEM_MODEL_CACHE",
        "MINICHEM_EXECUTION_JOB_ID",
        "MINICHEM_EXECUTION_JOB_DIRECTORY",
        "MINICHEM_JOB_ROOT",
        "MINICHEM_JOB_INPUTS",
        "MINICHEM_JOB_OUTPUTS",
        "MINICHEM_JOB_REPORT",
        "MINICHEM_WORKER_CPU_IDS",
        "MINICHEM_WORKER_CPU_COUNT",
        "XDG_CACHE_HOME",
        "GAUSS_SCRDIR",
        "GAUSS_EXEDIR",
        "g16root",
    }
)
CHILD_ENVIRONMENT_PREFIXES = (
    "MINICHEM_",
    "GAUSS_",
    "ORCA_",
    "OPENMOLCAS_",
    "MOLCAS_",
    "CREST_",
    "CHEMGRAPH_",
    "OMP_",
    "MKL_",
    "OPENBLAS_",
    "NUMEXPR_",
    "VECLIB_",
    "CUDA_",
    "ROCR_",
    "OMPI_",
    "PRTE_",
    "SLURM_",
)
SENSITIVE_ENV_MARKERS = (
    "API_KEY",
    "ACCESS_KEY",
    "SECRET",
    "TOKEN",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
    "PRIVATE_KEY",
    "AUTHORIZATION",
    "CAPABILITY",
)

# Python 3.11 的标准库可能没有 pidfd API；Linux 仍提供稳定的 syscall。
# 这些编号属于 Linux 通用 syscall 表，非 Linux 主机不尝试调用。
_PIDFD_OPEN_SYSCALL = 434
_PIDFD_SEND_SIGNAL_SYSCALL = 424
_MISSING_ATTRIBUTE = object()
_RUNTIME_CACHE_ENVIRONMENT_NAMES = frozenset(
    {
        "MINICHEM_MODEL_CACHE",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_XET_CACHE",
        "TORCH_HOME",
        "FAIRCHEM_CACHE_DIR",
    }
)

# programmable_analysis 的 bwrap 由仓库启动层注入。这里仍然固定版本、目标
# 位置和 SHA-256，避免 supervisor 被一个任意 PATH 命令或被篡改的环境值绕过。
PROGRAMMABLE_ANALYSIS_BWRAP_ENV = "MINICHEM_PROGRAMMABLE_ANALYSIS_BWRAP"
PROGRAMMABLE_ANALYSIS_BWRAP_RUNTIME_ROOT_ENV = (
    "MINICHEM_PROGRAMMABLE_ANALYSIS_BWRAP_RUNTIME_ROOT"
)
PROGRAMMABLE_ANALYSIS_BWRAP_SHA256_ENV = "MINICHEM_PROGRAMMABLE_ANALYSIS_BWRAP_SHA256"
PROGRAMMABLE_ANALYSIS_BWRAP_VERSION = "0.149.0"
PROGRAMMABLE_ANALYSIS_BWRAP_TARGET = (
    "vendor/x86_64-unknown-linux-musl/codex-resources/bwrap"
)
PROGRAMMABLE_ANALYSIS_BWRAP_SHA256 = (
    "7df960565a0dece99240ea4b9d0e011307817f9f3b73176c7b71fda44fe84765"
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_PROGRAMMABLE_ANALYSIS_LAUNCHER_ENVIRONMENT = frozenset(
    {
        PROGRAMMABLE_ANALYSIS_BWRAP_ENV,
        PROGRAMMABLE_ANALYSIS_BWRAP_RUNTIME_ROOT_ENV,
        PROGRAMMABLE_ANALYSIS_BWRAP_SHA256_ENV,
    }
)
_PROGRAMMABLE_ANALYSIS_GPU_MODULE_PREFIXES = (
    "torch",
    "fairchem",
    "tensorflow",
    "jax",
    "cupy",
    "paddle",
    "numba.cuda",
)


@dataclass
class _SupervisorContext:
    """保存一次 supervisor 执行的可变状态，供各阶段函数共享。"""

    specification: dict[str, Any]
    job_directory: Path
    status_path: Path
    stdout_path: Path
    stderr_path: Path
    command: list[str]
    resources: dict[str, Any]
    resource_allocation: dict[str, Any]
    evaluation_budget: dict[str, Any]
    walltime: int
    execution_class: str
    compute_timeout_policy: bool
    hard_timeout: int
    timeout_policy: dict[str, Any]
    memory_limit_mb: int | None
    execution_started: bool
    started_at: str
    started_monotonic: float
    cancellation_path: Path
    supervisor_identity: ProcessIdentity
    child: subprocess.Popen[bytes] | None = None
    child_identity: ProcessIdentity | None = None
    cancellation_signal: int | None = None
    preserve_queue_state: bool = False
    queue_ticket: GaussianQueueTicket | None = None
    queue_reservation: Any = None


@dataclass
class _ChildObservation:
    """保存子进程监控和清理阶段的结果。"""

    timed_out: bool
    timeout_reason: str | None
    last_progress_at: str
    max_process_group_cpu_ticks: int | None
    memory_exceeded: bool
    child_reap_failed: bool
    peak_process_group_rss_kb: int
    background_process_cleanup: str
    process_group_activity: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _absolute_path_without_following_symlinks(path: str | Path) -> Path:
    """转为绝对路径但保留末端 symlink，交给 O_NOFOLLOW 做最终核验。"""

    return Path(os.path.abspath(os.path.expanduser(str(path))))


def _ensure_private_directory(path: Path) -> None:
    """创建或核验私有目录；目录本身或父目录为 symlink 时抛出异常。"""

    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_dir():
            raise OSError(f"private directory is not a real directory: {path}")
    else:
        path.mkdir(parents=True, mode=PRIVATE_DIRECTORY_MODE, exist_ok=False)
    os.chmod(path, PRIVATE_DIRECTORY_MODE, follow_symlinks=False)


def _fsync_parent_directory(path: Path) -> None:
    """同步目录元数据，确保刚完成的 replace 或新建锁文件可恢复。"""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    """以私有随机临时文件原子写 JSON，并同步文件和父目录。"""

    parent = path.parent
    _ensure_private_directory(parent)
    if path.is_symlink():
        raise OSError(f"JSON destination cannot be a symlink: {path}")
    payload = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, PRIVATE_FILE_MODE)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if path.is_symlink():
            raise OSError(f"JSON destination became a symlink: {path}")
        os.replace(temporary, path)
        _fsync_parent_directory(parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


@contextmanager
def _open_private_append(path: Path):
    """以 O_NOFOLLOW 打开追加日志，并将文件权限固定为 0600。"""

    _ensure_private_directory(path.parent)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_APPEND
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags, PRIVATE_FILE_MODE)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise OSError(f"log destination is not a regular file: {path}")
        os.fchmod(descriptor, PRIVATE_FILE_MODE)
        with os.fdopen(descriptor, "ab", buffering=0) as handle:
            descriptor = -1
            yield handle
    finally:
        if descriptor >= 0:
            os.close(descriptor)


@contextmanager
def _open_private_lock(path: Path):
    """以 O_NOFOLLOW 打开跨进程锁，阻止 symlink 劫持并固定权限。"""

    _ensure_private_directory(path.parent)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, PRIVATE_FILE_MODE)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise OSError(f"lock path is not a regular file: {path}")
        os.fchmod(descriptor, PRIVATE_FILE_MODE)
        _fsync_parent_directory(path.parent)
        with os.fdopen(descriptor, "a+b") as handle:
            descriptor = -1
            yield handle
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _is_sensitive_environment_name(name: str) -> bool:
    """判断环境变量名是否明显包含认证或秘密材料。"""

    upper_name = name.upper()
    return any(marker in upper_name for marker in SENSITIVE_ENV_MARKERS)


def _environment_name_is_allowed(name: str) -> bool:
    """判断环境变量是否属于科学运行时的公开白名单。"""

    # 模型缓存路径不能从 supervisor 父环境或任意 child_environment 直接
    # 继承；它们必须由持久 spec 中的可信 runtime profile 重新注入。
    if name in _RUNTIME_CACHE_ENVIRONMENT_NAMES:
        return False
    return name in CHILD_ENVIRONMENT_NAMES or (
        not _is_sensitive_environment_name(name)
        and name.startswith(CHILD_ENVIRONMENT_PREFIXES)
    )


def _open_pidfd_for_signal(pid: int) -> int | None:
    """打开稳定的进程句柄；标准库缺失时通过 Linux syscall 补齐。"""

    native_open = getattr(os, "pidfd_open", _MISSING_ATTRIBUTE)
    if native_open is None:
        # 测试或受限运行时可以显式关闭 pidfd；此时调用方必须保持未知状态。
        return None
    try:
        if callable(native_open):
            return int(native_open(pid, 0))
        if sys.platform != "linux":
            return None
        libc = ctypes.CDLL(None, use_errno=True)
        descriptor = int(
            libc.syscall(
                ctypes.c_long(_PIDFD_OPEN_SYSCALL),
                ctypes.c_int(pid),
                ctypes.c_uint(0),
            )
        )
        if descriptor < 0:
            error_number = ctypes.get_errno()
            raise OSError(error_number, os.strerror(error_number))
        return descriptor
    except (OSError, TypeError, ValueError):
        return None


def _send_pidfd_signal(descriptor: int, received_signal: signal.Signals) -> bool:
    """通过标准库或 Linux syscall 向已打开的 pidfd 发信号。"""

    native_send = getattr(signal, "pidfd_send_signal", _MISSING_ATTRIBUTE)
    if native_send is None:
        return False
    try:
        if callable(native_send):
            native_send(descriptor, int(received_signal), None, 0)
            return True
        if sys.platform != "linux":
            return False
        libc = ctypes.CDLL(None, use_errno=True)
        result = int(
            libc.syscall(
                ctypes.c_long(_PIDFD_SEND_SIGNAL_SYSCALL),
                ctypes.c_int(descriptor),
                ctypes.c_int(int(received_signal)),
                ctypes.c_void_p(0),
                ctypes.c_uint(0),
            )
        )
        if result < 0:
            error_number = ctypes.get_errno()
            raise OSError(error_number, os.strerror(error_number))
        return True
    except (OSError, TypeError, ValueError):
        return False


def _validated_runtime_cache_environment(
    specification: dict[str, Any],
) -> dict[str, str]:
    """仅保留可信 runtime profile 声明且位于其模型缓存根下的缓存路径。"""

    profile = specification.get("runtime_profile_environment")
    if not isinstance(profile, dict):
        return {}
    root_value = profile.get("MINICHEM_MODEL_CACHE")
    if not isinstance(root_value, str) or not root_value.strip():
        return {}
    root = Path(root_value).expanduser().resolve()
    values: dict[str, str] = {}
    for name in _RUNTIME_CACHE_ENVIRONMENT_NAMES:
        raw_value = profile.get(name)
        if not isinstance(raw_value, str) or not raw_value.strip():
            continue
        candidate = Path(raw_value).expanduser().resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise OSError(
                f"runtime profile cache path escapes MINICHEM_MODEL_CACHE: {name}"
            ) from exc
        values[name] = str(candidate)
    return values


def _inherited_child_environment() -> dict[str, str]:
    """只复制父进程中属于公开科学运行时白名单的变量。"""

    return {
        name: value
        for name, value in os.environ.items()
        if _environment_name_is_allowed(name)
    }


def _explicit_child_environment(specification: dict[str, Any]) -> dict[str, str]:
    """读取可信 spec 的显式变量，并拒绝绕过 profile 的模型缓存路径。"""

    explicit = specification.get("child_environment") or {}
    if not isinstance(explicit, dict):
        return {}
    cache_overrides = {
        str(name)
        for name in explicit
        if str(name) in _RUNTIME_CACHE_ENVIRONMENT_NAMES
    }
    if cache_overrides:
        raise OSError(
            "runtime profile is required for child cache environment: "
            + ", ".join(sorted(cache_overrides))
        )
    return {
        str(name): str(value)
        for name, value in explicit.items()
        if str(name).strip()
    }


def _prepare_child_scratch_directories(
    job_directory: Path,
) -> tuple[Path, Path, Path]:
    """创建作业专用的临时目录、HOME 和 matplotlib 配置目录。"""

    temporary = job_directory / ".tmp"
    home = job_directory / ".home"
    matplotlib = temporary / "matplotlib"
    for directory in (temporary, home, matplotlib):
        _ensure_private_directory(directory)
    return temporary, home, matplotlib


def _job_environment_values(
    specification: dict[str, Any],
    job_directory: Path,
    temporary: Path,
    home: Path,
    matplotlib: Path,
    gaussian_scratch_requested: bool,
) -> dict[str, str]:
    """返回固定的作业路径变量，并把 Gaussian scratch 重定向到私有目录。"""

    values = {
        "TMPDIR": str(temporary),
        "HOME": str(home),
        "MPLCONFIGDIR": str(matplotlib),
        "MINICHEM_EXECUTION_JOB_ID": str(specification.get("job_id") or ""),
        "MINICHEM_EXECUTION_JOB_DIRECTORY": str(job_directory),
        "MINICHEM_JOB_ROOT": str(job_directory),
        "MINICHEM_JOB_INPUTS": str(job_directory / "inputs"),
        "MINICHEM_JOB_OUTPUTS": str(job_directory / "outputs"),
        "MINICHEM_JOB_REPORT": str(job_directory / "report"),
    }
    if gaussian_scratch_requested:
        gaussian_scratch = temporary / "gaussian"
        _ensure_private_directory(gaussian_scratch)
        values["GAUSS_SCRDIR"] = str(gaussian_scratch)
    return values


def _build_child_environment(
    specification: dict[str, Any], job_directory: Path
) -> dict[str, str]:
    """构造最小子进程环境，并固定作业 scratch、HOME 和路径变量。"""

    environment = _inherited_child_environment()
    environment.update(_explicit_child_environment(specification))
    environment.setdefault("PATH", os.defpath)
    environment.update(_validated_runtime_cache_environment(specification))
    temporary, home, matplotlib = _prepare_child_scratch_directories(job_directory)
    environment.update(
        _job_environment_values(
            specification,
            job_directory,
            temporary,
            home,
            matplotlib,
            bool(environment.get("GAUSS_SCRDIR")),
        )
    )
    runtime = str(specification.get("runtime") or "").strip()
    if runtime:
        environment.setdefault("MINICHEM_BACKEND_RUNTIME", runtime)
    return environment


class _ProgrammableAnalysisSandboxError(RuntimeError):
    """表示 programmable_analysis 的 OS 沙箱无法安全建立。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _sandbox_absolute_path(value: str | Path, *, label: str) -> Path:
    """把可信沙箱路径转成绝对词法路径并拒绝空值、相对路径和 NUL。"""

    raw = str(value)
    if not raw or "\x00" in raw:
        raise _ProgrammableAnalysisSandboxError(
            "programmable_analysis_sandbox_path_invalid",
            f"{label} is empty or contains a NUL byte",
        )
    path = Path(raw)
    if not path.is_absolute():
        raise _ProgrammableAnalysisSandboxError(
            "programmable_analysis_sandbox_path_invalid",
            f"{label} must be an absolute path",
        )
    return Path(os.path.abspath(os.fspath(path)))


def _reject_sandbox_symlink_components(
    path: Path,
    *,
    label: str,
    allow_final_symlink: bool = False,
) -> None:
    """逐级核对沙箱根路径，防止提交后通过符号链接替换挂载源。"""

    cursor = Path(path.anchor or os.curdir)
    components = path.parts[1:] if path.anchor else path.parts
    for component in components:
        cursor /= component
        try:
            metadata = os.lstat(cursor)
        except FileNotFoundError:
            break
        except OSError as exc:
            raise _ProgrammableAnalysisSandboxError(
                "programmable_analysis_sandbox_path_unreadable",
                f"{label} cannot be inspected",
            ) from exc
        if stat.S_ISLNK(metadata.st_mode) and not (
            allow_final_symlink and cursor == path
        ):
            raise _ProgrammableAnalysisSandboxError(
                "programmable_analysis_sandbox_symlink_rejected",
                f"{label} contains a symbolic link",
            )


def _require_sandbox_directory(path: Path, *, label: str) -> Path:
    """核对一个非符号链接目录，并返回规范化目录。"""

    path = _sandbox_absolute_path(path, label=label)
    _reject_sandbox_symlink_components(path, label=label)
    try:
        metadata = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise _ProgrammableAnalysisSandboxError(
            "programmable_analysis_sandbox_path_missing",
            f"{label} is unavailable",
        ) from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise _ProgrammableAnalysisSandboxError(
            "programmable_analysis_sandbox_path_invalid",
            f"{label} is not a directory",
        )
    return path


def _path_is_within_sandbox_root(path: Path, root: Path) -> bool:
    """判断规范化 path 是否等于或位于 root 内。"""

    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _require_sandbox_file(path: Path, *, label: str) -> Path:
    """核对一个非符号链接普通文件。"""

    path = _sandbox_absolute_path(path, label=label)
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise _ProgrammableAnalysisSandboxError(
            "programmable_analysis_sandbox_path_missing",
            f"{label} is unavailable",
        ) from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise _ProgrammableAnalysisSandboxError(
            "programmable_analysis_sandbox_path_invalid",
            f"{label} is not a regular file",
        )
    return path


def _verify_programmable_analysis_bwrap() -> tuple[Path, Path]:
    """验证固定 Codex 0.149.0 bwrap 和其运行根；缺失时绝不裸奔执行。"""

    configured_root = os.environ.get(
        PROGRAMMABLE_ANALYSIS_BWRAP_RUNTIME_ROOT_ENV, ""
    ).strip()
    configured_bwrap = os.environ.get(PROGRAMMABLE_ANALYSIS_BWRAP_ENV, "").strip()
    if not configured_root or not configured_bwrap:
        raise _ProgrammableAnalysisSandboxError(
            "programmable_analysis_sandbox_unavailable",
            "Trusted Codex 0.149.0 bwrap configuration is missing",
        )
    runtime_root = _require_sandbox_directory(
        _sandbox_absolute_path(configured_root, label="bwrap runtime root"),
        label="bwrap runtime root",
    )
    bwrap = _require_sandbox_file(
        _sandbox_absolute_path(configured_bwrap, label="bwrap executable"),
        label="bwrap executable",
    )
    expected_relative = Path(PROGRAMMABLE_ANALYSIS_BWRAP_TARGET)
    try:
        relative = bwrap.relative_to(runtime_root)
    except ValueError as exc:
        raise _ProgrammableAnalysisSandboxError(
            "programmable_analysis_sandbox_path_invalid",
            "bwrap executable is outside the trusted 0.149.0 runtime root",
        ) from exc
    if relative != expected_relative:
        raise _ProgrammableAnalysisSandboxError(
            "programmable_analysis_sandbox_path_invalid",
            "bwrap executable is not the fixed 0.149.0 runtime component",
        )
    configured_sha = os.environ.get(PROGRAMMABLE_ANALYSIS_BWRAP_SHA256_ENV, "").strip()
    if configured_sha and (
        not _SHA256_PATTERN.fullmatch(configured_sha)
        or configured_sha != PROGRAMMABLE_ANALYSIS_BWRAP_SHA256
    ):
        raise _ProgrammableAnalysisSandboxError(
            "programmable_analysis_sandbox_integrity_failed",
            "Configured bwrap SHA-256 does not match fixed Codex 0.149.0",
        )
    actual_sha = _digest(bwrap)
    if actual_sha != PROGRAMMABLE_ANALYSIS_BWRAP_SHA256:
        raise _ProgrammableAnalysisSandboxError(
            "programmable_analysis_sandbox_integrity_failed",
            "Vendored bwrap SHA-256 does not match fixed Codex 0.149.0",
        )
    if not os.access(bwrap, os.X_OK):
        raise _ProgrammableAnalysisSandboxError(
            "programmable_analysis_sandbox_unavailable",
            "Vendored bwrap is not executable",
        )
    try:
        completed = subprocess.run(
            [str(bwrap), "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5,
            env={"PATH": "/usr/bin:/bin"},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise _ProgrammableAnalysisSandboxError(
            "programmable_analysis_sandbox_unavailable",
            "Vendored bwrap executable probe failed",
        ) from exc
    if completed.returncode != 0:
        raise _ProgrammableAnalysisSandboxError(
            "programmable_analysis_sandbox_unavailable",
            "Vendored bwrap executable probe returned a non-zero status",
        )
    return runtime_root, bwrap


def _programmable_analysis_job_requires_gpu(specification: dict[str, Any]) -> bool:
    """根据可信 preflight 结果判断脚本是否明确依赖 GPU 科学库。"""

    metadata = specification.get("metadata") or {}
    if bool(metadata.get("gpu_required")):
        return True
    contract = metadata.get("analysis_contract") or {}
    if isinstance(contract, dict) and bool(contract.get("gpu_required")):
        return True
    preflight = metadata.get("preflight") or {}
    if not isinstance(preflight, dict):
        return False
    names: list[str] = []
    for key in ("discovered_imports", "required_modules"):
        values = preflight.get(key) or []
        if isinstance(values, (list, tuple)):
            names.extend(str(value).strip().lower() for value in values)
    return any(
        name == prefix or name.startswith(prefix + ".")
        for name in names
        for prefix in _PROGRAMMABLE_ANALYSIS_GPU_MODULE_PREFIXES
    )


def _runtime_python_root_for_programmable_analysis(
    command: list[str],
) -> Path:
    """只允许服务固定的 ARCHE 标准 Python 环境作为程序解释器。

    ``.venv/bin/python`` 在标准环境中通常是一个预先创建的软链接。这里逐级
    跟随这条固定软链接链，并要求链上的每一个目标都仍在同一个 ``.venv``
    内，避免把“允许软链接”扩大成可替换的宿主路径。
    """

    if not command or not str(command[0]).strip():
        raise _ProgrammableAnalysisSandboxError(
            "programmable_analysis_runtime_invalid",
            "programmable_analysis command has no runtime Python",
        )
    python_path = _sandbox_absolute_path(command[0], label="runtime Python")
    _reject_sandbox_symlink_components(
        python_path,
        label="runtime Python",
        allow_final_symlink=True,
    )
    standard_runtime_root = _sandbox_absolute_path(
        STANDARD_ARCHE_PYTHON.parent.parent,
        label="standard Python runtime root",
    )
    if not _path_is_within_sandbox_root(python_path, standard_runtime_root):
        raise _ProgrammableAnalysisSandboxError(
            "programmable_analysis_runtime_invalid",
            "programmable_analysis runtime Python is outside the fixed .venv",
        )
    standard_python = _require_sandbox_file(
        _sandbox_absolute_path(STANDARD_ARCHE_PYTHON, label="standard Python"),
        label="standard Python",
    )
    current_path = python_path
    visited_links: set[Path] = set()
    while True:
        _reject_sandbox_symlink_components(
            current_path,
            label="runtime Python",
            allow_final_symlink=True,
        )
        try:
            metadata = os.lstat(current_path)
        except OSError as exc:
            raise _ProgrammableAnalysisSandboxError(
                "programmable_analysis_runtime_invalid",
                "programmable_analysis runtime Python is unavailable",
            ) from exc
        if not stat.S_ISLNK(metadata.st_mode):
            if not stat.S_ISREG(metadata.st_mode) or current_path != standard_python:
                raise _ProgrammableAnalysisSandboxError(
                    "programmable_analysis_runtime_invalid",
                    "programmable_analysis must use the fixed ARCHE Python runtime",
                )
            break
        if current_path in visited_links:
            raise _ProgrammableAnalysisSandboxError(
                "programmable_analysis_runtime_invalid",
                "programmable_analysis runtime Python contains a symlink cycle",
            )
        visited_links.add(current_path)
        try:
            target = os.readlink(current_path)
        except OSError as exc:
            raise _ProgrammableAnalysisSandboxError(
                "programmable_analysis_runtime_invalid",
                "programmable_analysis runtime Python symlink cannot be read",
            ) from exc
        target_path = Path(target)
        if target_path.is_absolute():
            next_path = target_path
        else:
            next_path = current_path.parent / target_path
        current_path = _sandbox_absolute_path(next_path, label="runtime Python symlink target")
        if not _path_is_within_sandbox_root(current_path, standard_runtime_root):
            raise _ProgrammableAnalysisSandboxError(
                "programmable_analysis_runtime_invalid",
                "programmable_analysis runtime Python symlink escapes the fixed .venv",
            )
    runtime_root = _require_sandbox_directory(
        standard_runtime_root,
        label="Python runtime root",
    )
    return runtime_root


def _system_read_only_bindings() -> list[tuple[Path, str]]:
    """列出 Python、动态库和 HTTPS 所需的最小系统只读挂载。"""

    bindings: list[tuple[Path, str]] = []
    for destination in ("/usr", "/bin", "/sbin", "/lib", "/lib64"):
        source = Path(destination)
        if not source.exists() or not source.is_dir():
            continue
        bindings.append((source.resolve(), destination))
    for destination in (
        "/etc/passwd",
        "/etc/group",
        "/etc/nsswitch.conf",
        "/etc/hosts",
        "/etc/resolv.conf",
        "/etc/localtime",
    ):
        source = Path(destination)
        if source.exists() and source.is_file():
            bindings.append((source.resolve(), destination))
    for destination in ("/etc/ssl/certs", "/etc/ca-certificates", "/etc/pki"):
        source = Path(destination)
        if source.exists() and source.is_dir():
            bindings.append((source.resolve(), destination))
    return bindings


def _append_unique_read_only_binding(
    bindings: list[tuple[Path, str]],
    source: Path,
    destination: str | Path,
    *,
    label: str,
) -> None:
    """加入一个严格的只读目录挂载，并拒绝作业、仓库或根目录重叠。"""

    source = _require_sandbox_directory(source, label=label)
    destination_path = _sandbox_absolute_path(destination, label=f"{label} destination")
    if destination_path == Path("/"):
        raise _ProgrammableAnalysisSandboxError(
            "programmable_analysis_sandbox_path_invalid",
            f"{label} cannot mount the host root",
        )
    if any(existing_destination == str(destination_path) for _, existing_destination in bindings):
        return
    bindings.append((source, str(destination_path)))


def _candidate_cache_roots(
    child_environment: Mapping[str, str],
    *,
    job_directory: Path,
    fixed_roots: tuple[Path, ...],
) -> list[Path]:
    """收集已由 runtime profile 声明的缓存根，排除整个 local_data 等宽根。"""

    roots: list[Path] = []
    for name in (*_RUNTIME_CACHE_ENVIRONMENT_NAMES, "XDG_CACHE_HOME"):
        raw_value = str(child_environment.get(name) or "").strip()
        if not raw_value:
            continue
        candidate = _sandbox_absolute_path(raw_value, label=name)
        try:
            candidate = candidate.resolve(strict=True)
        except OSError as exc:
            raise _ProgrammableAnalysisSandboxError(
                "programmable_analysis_cache_unavailable",
                f"Trusted runtime cache {name} is unavailable",
            ) from exc
        if not candidate.is_dir():
            raise _ProgrammableAnalysisSandboxError(
                "programmable_analysis_cache_invalid",
                f"Trusted runtime cache {name} is not a directory",
            )
        if candidate == Path("/") or candidate == job_directory:
            raise _ProgrammableAnalysisSandboxError(
                "programmable_analysis_cache_invalid",
                f"Trusted runtime cache {name} is too broad",
            )
        covered_by_fixed_root = False
        for root in fixed_roots:
            if candidate == root or _path_is_within_sandbox_root(candidate, root):
                # 已有固定框架挂载覆盖该路径时不额外增加一个重叠挂载。
                covered_by_fixed_root = True
                break
            if _path_is_within_sandbox_root(root, candidate):
                raise _ProgrammableAnalysisSandboxError(
                    "programmable_analysis_cache_invalid",
                    f"Trusted runtime cache {name} is broader than an approved root",
                )
        if covered_by_fixed_root:
            continue
        if _path_is_within_sandbox_root(candidate, job_directory) or _path_is_within_sandbox_root(
            job_directory, candidate
        ):
            raise _ProgrammableAnalysisSandboxError(
                "programmable_analysis_cache_invalid",
                f"Trusted runtime cache {name} overlaps the job directory",
            )
        if candidate not in roots:
            roots.append(candidate)
    return roots


def _path_value_is_visible(
    value: str,
    *,
    job_directory: Path,
    read_only_roots: tuple[Path, ...],
) -> bool:
    """判断一个环境路径是否落在当前作业或明确只读根内。"""

    try:
        candidate = Path(value).resolve(strict=False)
    except OSError:
        return False
    if _path_is_within_sandbox_root(candidate, job_directory):
        return True
    return any(_path_is_within_sandbox_root(candidate, root) for root in read_only_roots)


def _safe_programmable_analysis_environment(
    source_environment: Mapping[str, str],
    *,
    job_directory: Path,
    python_root: Path,
    toolbox_root: Path,
    read_only_roots: tuple[Path, ...],
    gpu_required: bool,
) -> dict[str, str]:
    """从 supervisor 环境筛出 bwrap 内可见的最小变量，去掉服务认证和宿主路径。"""

    allowed_roots = (job_directory, *read_only_roots)
    environment: dict[str, str] = {
        "PATH": os.pathsep.join((str(python_root / "bin"), "/usr/local/bin", "/usr/bin", "/bin")),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "MINICHEM_HOME": str(toolbox_root),
    }
    for name, raw_value in source_environment.items():
        if name in _PROGRAMMABLE_ANALYSIS_LAUNCHER_ENVIRONMENT:
            continue
        if not isinstance(name, str) or not isinstance(raw_value, str) or "\x00" in raw_value:
            continue
        if not _environment_name_is_allowed(name) and name not in {
            "LD_LIBRARY_PATH",
            "XDG_CACHE_HOME",
        }:
            continue
        if name in {"PATH", "PYTHONPATH", "LD_LIBRARY_PATH"}:
            entries = []
            for entry in raw_value.split(os.pathsep):
                entry = entry.strip()
                if not entry:
                    continue
                if name == "PATH" and not entry.startswith("/"):
                    continue
                if name != "PATH" and not _path_value_is_visible(
                    entry,
                    job_directory=job_directory,
                    read_only_roots=allowed_roots,
                ):
                    continue
                entries.append(entry)
            if name == "PYTHONPATH":
                entries.insert(0, str(job_directory))
            elif name == "LD_LIBRARY_PATH":
                entries = [entry for entry in entries if entry]
            if entries:
                environment[name] = os.pathsep.join(dict.fromkeys(entries))
            continue
        if name in {"HOME", "TMPDIR", "MPLCONFIGDIR", "GAUSS_SCRDIR"} or name.endswith(
            ("_ROOT", "_DIRECTORY", "_WORKSPACE", "_CACHE", "_DIR")
        ):
            if raw_value.startswith("/") and not _path_value_is_visible(
                raw_value,
                job_directory=job_directory,
                read_only_roots=allowed_roots,
            ):
                continue
        environment[name] = raw_value
    if not gpu_required:
        # 固定服务预算可能包含一张 GPU，但普通 Python 分析不能被误导为可见。
        environment["CUDA_VISIBLE_DEVICES"] = ""
        environment["ROCR_VISIBLE_DEVICES"] = ""
    return environment


def _programmable_analysis_gpu_bindings(
    specification: dict[str, Any],
    child_environment: Mapping[str, str],
) -> list[tuple[Path, str]]:
    """按 scheduler 的实际 GPU id 挂载 NVIDIA 设备；无法证明时 fail-closed。"""

    resources = specification.get("resource_limits") or {}
    requested = int(resources.get("gpu_count") or 0)
    if not _programmable_analysis_job_requires_gpu(specification):
        return []
    allocation = specification.get("resource_allocation") or {}
    gpu_ids = [str(item).strip() for item in allocation.get("gpu_ids") or []]
    if requested <= 0 or len(gpu_ids) != requested or len(set(gpu_ids)) != len(gpu_ids):
        raise _ProgrammableAnalysisSandboxError(
            "programmable_analysis_gpu_unavailable",
            "GPU analysis has no complete scheduler allocation",
        )
    visible = [item.strip() for item in str(child_environment.get("CUDA_VISIBLE_DEVICES") or "").split(",") if item.strip()]
    if visible != gpu_ids:
        raise _ProgrammableAnalysisSandboxError(
            "programmable_analysis_gpu_unavailable",
            "GPU visibility does not match the scheduler allocation",
        )
    bindings: list[tuple[Path, str]] = []
    for gpu_id in gpu_ids:
        if not re.fullmatch(r"[0-9]+", gpu_id):
            raise _ProgrammableAnalysisSandboxError(
                "programmable_analysis_gpu_unavailable",
                "GPU allocation id is not a host NVIDIA device id",
            )
        device = Path(f"/dev/nvidia{gpu_id}")
        if not device.is_char_device():
            raise _ProgrammableAnalysisSandboxError(
                "programmable_analysis_gpu_unavailable",
                "Allocated NVIDIA device is unavailable",
            )
        bindings.append((device, str(device)))
    for name in ("nvidiactl", "nvidia-uvm", "nvidia-uvm-tools", "nvidia-modeset"):
        device = Path(f"/dev/{name}")
        if device.exists() and device.is_char_device():
            bindings.append((device, str(device)))
    controls = {destination for _, destination in bindings}
    if "/dev/nvidiactl" not in controls or "/dev/nvidia-uvm" not in controls:
        raise _ProgrammableAnalysisSandboxError(
            "programmable_analysis_gpu_unavailable",
            "Required NVIDIA control devices are unavailable",
        )
    return bindings


def _destination_parent_directories(destinations: Iterable[str]) -> tuple[str, ...]:
    """生成 bwrap bind 目标所需的父目录，避免隐式创建宿主路径。"""

    parents: set[str] = set()
    for raw_destination in destinations:
        current = Path(raw_destination).parent
        while str(current) not in {"", "/", "."}:
            parents.add(str(current))
            current = current.parent
    return tuple(sorted(parents, key=lambda value: (len(Path(value).parts), value)))


def _build_programmable_analysis_sandbox_command(
    context: _SupervisorContext,
    child_environment: Mapping[str, str],
) -> tuple[list[str], dict[str, str]]:
    """构造 Python 作业专用 bwrap argv 和最小环境，不改变 native/TS 启动路径。"""

    bwrap_runtime_root, bwrap = _verify_programmable_analysis_bwrap()
    job_directory = _require_sandbox_directory(
        context.job_directory,
        label="programmable_analysis job directory",
    )
    python_root = _runtime_python_root_for_programmable_analysis(context.command)
    toolbox_root = _require_sandbox_directory(
        Path(__file__).resolve().parents[2],
        label="MiniChem toolbox root",
    )
    repository_root = Path(__file__).resolve().parents[4]
    repository_source = repository_root / "src"
    framework_bindings: list[tuple[Path, str]] = []
    _append_unique_read_only_binding(
        framework_bindings,
        python_root,
        python_root,
        label="Python runtime root",
    )
    _append_unique_read_only_binding(
        framework_bindings,
        toolbox_root,
        toolbox_root,
        label="MiniChem toolbox root",
    )
    if repository_source.is_dir() and not repository_source.is_symlink():
        _append_unique_read_only_binding(
            framework_bindings,
            repository_source,
            repository_source,
            label="ARCHE source root",
        )
    cache_roots = _candidate_cache_roots(
        child_environment,
        job_directory=job_directory,
        fixed_roots=(python_root, toolbox_root, repository_source),
    )
    for index, cache_root in enumerate(cache_roots):
        _append_unique_read_only_binding(
            framework_bindings,
            cache_root,
            cache_root,
            label=f"runtime cache root {index}",
        )
    gpu_required = _programmable_analysis_job_requires_gpu(context.specification)
    safe_environment = _safe_programmable_analysis_environment(
        child_environment,
        job_directory=job_directory,
        python_root=python_root,
        toolbox_root=toolbox_root,
        read_only_roots=tuple(source for source, _ in framework_bindings),
        gpu_required=gpu_required,
    )
    gpu_bindings = _programmable_analysis_gpu_bindings(
        context.specification,
        safe_environment,
    )
    destination_values = [
        str(job_directory),
        *(destination for _, destination in framework_bindings),
        *(destination for _, destination in _system_read_only_bindings()),
        *(destination for _, destination in gpu_bindings),
        "/etc",
        "/tmp",
        "/run",
        "/dev",
        "/proc",
    ]
    arguments: list[str] = [
        str(bwrap),
        "--die-with-parent",
        "--new-session",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--unshare-cgroup",
        "--share-net",
        "--cap-drop",
        "ALL",
        "--clearenv",
    ]
    for name, value in sorted(safe_environment.items()):
        arguments.extend(("--setenv", name, value))
    for destination in _destination_parent_directories(destination_values):
        arguments.extend(("--dir", destination))
    arguments.extend(
        (
            "--tmpfs",
            "/",
            "--tmpfs",
            "/tmp",
            "--tmpfs",
            "/run",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--bind",
            str(job_directory),
            str(job_directory),
        )
    )
    for source, destination in framework_bindings:
        arguments.extend(("--ro-bind", str(source), destination))
    for source, destination in _system_read_only_bindings():
        arguments.extend(("--ro-bind", str(source), destination))
    for source, destination in gpu_bindings:
        arguments.extend(("--dev-bind", str(source), destination))
    arguments.extend(("--chdir", str(job_directory), "--", *context.command))
    # bwrap 自身只是启动器；其固定运行根不应通过环境继续暴露给用户脚本。
    del bwrap_runtime_root
    return arguments, safe_environment


def _process_identity_fields(prefix: str, identity: ProcessIdentity) -> dict[str, Any]:
    """返回 supervisor/status 使用的持久进程身份字段。"""

    return {
        f"{prefix}_pid": identity.pid,
        f"{prefix}_start_time_ticks": identity.start_time_ticks,
        f"{prefix}_start_ticks": identity.start_time_ticks,
        f"{prefix}_boot_id": identity.boot_id,
        f"{prefix}_process_group_id": identity.process_group_id,
        f"{prefix}_session_id": identity.session_id,
    }


def _process_identity_complete(identity: ProcessIdentity) -> bool:
    """判断当前进程身份是否具备恢复和信号核对所需的全部字段。"""

    return (
        isinstance(identity.pid, int)
        and not isinstance(identity.pid, bool)
        and identity.pid > 0
        and isinstance(identity.start_time_ticks, int)
        and not isinstance(identity.start_time_ticks, bool)
        and identity.start_time_ticks > 0
        and bool(identity.boot_id)
    )


def _process_group_identity_complete(identity: ProcessIdentity) -> bool:
    """判断进程身份是否还带有可核对的进程组和会话字段。"""

    return _process_identity_complete(identity) and all(
        isinstance(value, int)
        and not isinstance(value, bool)
        and value > 0
        for value in (identity.process_group_id, identity.session_id)
    )


def _send_verified_pidfd_signal(
    identity: ProcessIdentity,
    received_signal: signal.Signals,
) -> bool:
    """通过稳定的 leader pidfd 发信号；缺少 pidfd 时保持未知且不发信号。"""

    descriptor = _open_pidfd_for_signal(identity.pid)
    if descriptor is None:
        return False
    try:
        current = capture_process_identity(identity.pid)
        if (
            not _process_group_identity_complete(current)
            or current.start_time_ticks != identity.start_time_ticks
            or current.boot_id != identity.boot_id
            or current.process_group_id != identity.process_group_id
            or current.session_id != identity.session_id
        ):
            return False
        return _send_pidfd_signal(descriptor, received_signal)
    except (OSError, ValueError, ProcessLookupError, PermissionError):
        return False
    finally:
        os.close(descriptor)


def _verified_process_group_signal(
    identity: ProcessIdentity | None,
    received_signal: signal.Signals,
) -> bool:
    """仅通过核对后的 leader pidfd 发信号，不对未证明的 PGID 批量发信号。"""

    if identity is None or not _process_group_identity_complete(identity):
        return False
    if process_identity_activity(identity.as_dict()) != "alive":
        return False
    current = capture_process_identity(identity.pid)
    if (
        not _process_group_identity_complete(current)
        or current.start_time_ticks != identity.start_time_ticks
        or current.boot_id != identity.boot_id
        or current.process_group_id != identity.process_group_id
        or current.session_id != identity.session_id
    ):
        return False
    # 仅有 PID/PGID 的“先核对再 os.killpg”仍存在 PID/PGID 复用竞态。
    # 固定环境没有 pidfd 时保持 unknown，并让上层保留资源和 FIFO 占用。
    return _send_verified_pidfd_signal(identity, received_signal)


def _child_process_group_is_empty(identity: ProcessIdentity) -> bool:
    """只有明确看到进程组为空时才允许结束 supervisor。"""

    return (
        inspect_process_group_membership(
            identity.process_group_id,
            identity.session_id,
            identity.boot_id,
        )
        == "empty"
    )


def _finish_verified_unresponsive_child(
    child: subprocess.Popen[bytes], identity: ProcessIdentity
) -> bool | None:
    """尝试终止已有完整身份的 child；None 表示仍可继续观察。"""

    if not _verified_process_group_signal(identity, signal.SIGTERM):
        return _child_process_group_is_empty(identity) if child.poll() is not None else None
    try:
        child.wait(timeout=1)
    except subprocess.TimeoutExpired:
        if not _verified_process_group_signal(identity, signal.SIGKILL):
            return False
        try:
            child.wait(timeout=1)
        except subprocess.TimeoutExpired:
            return False
    return _child_process_group_is_empty(identity)


def _finish_unverified_child_safely(child: subprocess.Popen[bytes]) -> bool:
    """在身份读取失败时有界等待，并只在进程组明确清空后返回成功。

    child 即使已经退出，也不能仅凭 ``poll`` 证明 fork 出的 descendant 已经结束。
    因此调用方只有在这里同时取得完整 PGID/SID/boot 身份并观察到组为空时才能
    写 terminal 状态；其余情况返回 False，由上层保留 FIFO 和资源供恢复处理。
    """

    deadline = time.monotonic() + UNVERIFIED_CHILD_GRACE_SECONDS
    candidate_identity: ProcessIdentity | None = None
    while child.poll() is None and time.monotonic() < deadline:
        candidate = capture_process_identity(child.pid)
        if _process_group_identity_complete(candidate):
            candidate_identity = candidate
            result = _finish_verified_unresponsive_child(child, candidate)
            if result is not None:
                return result
        time.sleep(0.05)
    if child.poll() is None or candidate_identity is None:
        return False
    return _child_process_group_is_empty(candidate_identity)


def _wait_for_child_reap_safely(
    child: subprocess.Popen[bytes],
    child_identity: ProcessIdentity | None,
) -> bool:
    """在终止请求后有界回收 child；身份或信号未知时保留人工恢复状态。"""

    if child.poll() is not None:
        return True
    try:
        child.wait(timeout=5)
        return True
    except subprocess.TimeoutExpired:
        kill_sent = _verified_process_group_signal(child_identity, signal.SIGKILL)
        try:
            # SIGKILL 发送失败或身份核对失败时也不能无限 wait；身份未知时
            # 不再发送任何信号，调用方会保留 FIFO/资源状态供人工恢复。
            child.wait(timeout=5 if kill_sent else 1)
            return True
        except subprocess.TimeoutExpired:
            return False


def _claim_supervisor(specification: dict[str, Any], job_directory: Path) -> bool:
    """原子创建 supervisor claim，确保同一 operation 只有一个监督进程。"""

    operation_id = specification.get("operation_id")
    if operation_id is None:
        return True
    _ensure_private_directory(job_directory)
    claim_path = job_directory / "supervisor_claim.json"
    identity = capture_process_identity(os.getpid())
    claim = {
        "schema_version": 1,
        "operation_id": operation_id,
        "operation_fingerprint": specification.get("operation_fingerprint"),
        "job_id": specification["job_id"],
        "supervisor_pid": os.getpid(),
        **_process_identity_fields("supervisor", identity),
        "claimed_at": _now(),
    }
    try:
        descriptor = os.open(
            claim_path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            PRIVATE_FILE_MODE,
        )
    except FileExistsError:
        return False
    try:
        os.fchmod(descriptor, PRIVATE_FILE_MODE)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            json.dump(claim, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_parent_directory(job_directory)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return True


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _verify_launch_file_snapshots(
    snapshots: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    for expected in snapshots:
        path = Path(str(expected.get("path") or ""))
        try:
            stat = path.stat()
            actual_size = stat.st_size
            actual_sha = _digest(path)
        except (FileNotFoundError, OSError) as exc:
            mismatches.append({"path": str(path), "reason": str(exc)})
            continue
        if actual_size != expected.get("size_bytes") or actual_sha != expected.get(
            "sha256"
        ):
            mismatches.append(
                {
                    "role": expected.get("role"),
                    "path": str(path),
                    "expected_size_bytes": expected.get("size_bytes"),
                    "actual_size_bytes": actual_size,
                    "expected_sha256": expected.get("sha256"),
                    "actual_sha256": actual_sha,
                }
            )
    return mismatches


def _read_operation_receipt(path: Path) -> dict[str, Any] | None:
    """读取 operation receipt；不存在、损坏或不可读时返回未知。"""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _operation_receipt_matches(
    receipt: dict[str, Any], specification: dict[str, Any]
) -> bool:
    """核对 receipt 所属 operation、输入指纹、作业和提交键。"""

    return all(
        receipt.get(receipt_name) == specification.get(specification_name)
        for receipt_name, specification_name in (
            ("operation_id", "operation_id"),
            ("fingerprint", "operation_fingerprint"),
            ("job_id", "job_id"),
            ("submission_key", "submission_key"),
        )
    )


def _update_operation_receipt(
    specification: dict[str, Any], job_directory: Path, phase: str, **extra: Any
) -> None:
    """更新与当前 operation 完全匹配的 receipt，失败时保持原记录不变。"""

    operation_id = specification.get("operation_id")
    if operation_id is None:
        return
    receipt_path = job_directory / "operation_receipt.json"
    receipt = _read_operation_receipt(receipt_path)
    if receipt is None or not _operation_receipt_matches(receipt, specification):
        return
    receipt.update({"phase": phase, "updated_at": _now(), **extra})
    _atomic_json(receipt_path, receipt)


def _preexec(
    resources: dict[str, Any],
    evaluation_budget: dict[str, Any],
    resource_allocation: dict[str, Any],
):
    return partial(_apply_child_cpu_affinity, resources, resource_allocation)


def _apply_child_cpu_affinity(
    resources: dict[str, Any], resource_allocation: dict[str, Any]
) -> None:
    """在 child preexec 阶段校验并应用已分配的 CPU 亲和性。"""

    os.umask(0o077)
    allocated_cpu_ids = [
        int(item) for item in resource_allocation.get("cpu_ids") or []
    ]
    cpu_cores = resources.get("cpu_cores")
    if allocated_cpu_ids and hasattr(os, "sched_setaffinity"):
        allowed = set(os.sched_getaffinity(0))
        selected = set(allocated_cpu_ids)
        if not selected <= allowed:
            raise RuntimeError(
                "Allocated CPU ids are outside the job affinity: "
                f"allocated={sorted(selected)}, allowed={sorted(allowed)}"
            )
        os.sched_setaffinity(0, selected)
    elif cpu_cores is not None and hasattr(os, "sched_getaffinity"):
        allowed = sorted(os.sched_getaffinity(0))
        if allowed:
            os.sched_setaffinity(0, set(allowed[: max(1, int(cpu_cores))]))


def _memory_limit_mb(
    resources: dict[str, Any], evaluation_budget: dict[str, Any]
) -> int | None:
    candidates = [
        int(value)
        for value in (
            resources.get("memory_mb"),
            evaluation_budget.get("memory_mb"),
        )
        if value is not None
    ]
    return min(candidates) if candidates else None


def _process_group_rss_kb(process_group_id: int) -> int | None:
    proc = Path("/proc")
    if not proc.is_dir():
        return None
    total = 0
    observed = False
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="utf-8")
            fields = stat[stat.rfind(")") + 2 :].split()
            if len(fields) < 3 or int(fields[2]) != process_group_id:
                continue
            status = (entry / "status").read_text(encoding="utf-8")
            match = next(
                (line for line in status.splitlines() if line.startswith("VmRSS:")),
                None,
            )
            if match:
                total += int(match.split()[1])
            observed = True
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
    return total if observed else None


def _process_group_cpu_ticks(
    process_group_id: int,
) -> dict[tuple[int, int], int] | None:
    """读取进程组各进程的 CPU ticks 快照；无法读取时返回 None。"""

    proc = Path("/proc")
    if not proc.is_dir():
        return None
    snapshot: dict[tuple[int, int], int] = {}
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="utf-8")
            fields = stat[stat.rfind(")") + 2 :].split()
            if len(fields) < 20 or int(fields[2]) != process_group_id:
                continue
            process_identity = (int(entry.name), int(fields[19]))
            snapshot[process_identity] = int(fields[11]) + int(fields[12])
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
    return snapshot or None


def _output_file_state(paths: tuple[Path, ...]) -> tuple[tuple[str, int, int], ...]:
    """读取输出文件状态并返回路径、大小和纳秒修改时间；读取失败记为 -1。"""

    state: list[tuple[str, int, int]] = []
    for path in paths:
        try:
            stat = path.stat()
        except OSError:
            state.append((str(path), -1, -1))
        else:
            state.append((str(path), stat.st_size, stat.st_mtime_ns))
    return tuple(state)


def _positive_timeout_environment(name: str, default: int) -> int:
    """读取正整数秒数环境变量；缺失时返回默认值，格式错误或非正数时抛出异常。"""

    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _execution_class(specification: dict[str, Any]) -> str:
    """读取已冻结的超时类别；缺失时按旧 spec 的作业类型兼容为 compute 或 legacy。"""

    metadata = specification.get("metadata") or {}
    policy = metadata.get("execution_timeout_policy") or {}
    selected = policy.get("execution_class")
    if selected in {"compute", "fast"}:
        return str(selected)
    if specification.get("job_type") in {"native_software", "programmable_analysis"}:
        return "compute"
    return "legacy"


def _build_timeout_configuration(
    specification: dict[str, Any], resources: dict[str, Any]
) -> tuple[str, bool, int, int, dict[str, Any]]:
    """根据 spec 生成兼容旧协议的超时策略和实际秒数。"""

    walltime = max(1, int(resources.get("walltime_seconds") or 7200))
    execution_class = _execution_class(specification)
    compute_timeout_policy = execution_class == "compute"
    configured_hard_timeout = (
        _positive_timeout_environment(
            COMPUTE_HARD_TIMEOUT_ENV,
            DEFAULT_COMPUTE_HARD_TIMEOUT_SECONDS,
        )
        if compute_timeout_policy
        else walltime
    )
    hard_timeout = max(walltime, configured_hard_timeout)
    if not compute_timeout_policy:
        return (
            execution_class,
            False,
            walltime,
            hard_timeout,
            {
                "scope": "supervisor_actual_execution",
                "base_policy_field": "metadata.execution_timeout_policy",
                "execution_class": execution_class,
                "mode": "wall_clock",
                "walltime_seconds": walltime,
            },
        )
    configured_source = (
        COMPUTE_HARD_TIMEOUT_ENV
        if os.environ.get(COMPUTE_HARD_TIMEOUT_ENV, "").strip()
        else "default"
    )
    hard_timeout_source = (
        f"{configured_source}:clamped_to_idle"
        if configured_hard_timeout < walltime
        else configured_source
    )
    return (
        execution_class,
        True,
        walltime,
        hard_timeout,
        {
            "scope": "supervisor_actual_execution",
            "base_policy_field": "metadata.execution_timeout_policy",
            "execution_class": "compute",
            "mode": "inactivity_with_hard_cap",
            "progress_source": "stdout_stderr_file_or_process_group_cpu_ticks",
            "progress_signals": [
                "stdout_stderr_file_change",
                "process_group_cpu_ticks_increase",
            ],
            "progress_fields": ["size_bytes", "mtime_ns"],
            "cpu_ticks_source": "proc_stat_utime_plus_stime",
            "buffering_semantics": "only_flushed_file_bytes_are_observable",
            "inactivity_timeout_seconds": walltime,
            "inactivity_timeout_source": "resource_limits.walltime_seconds",
            "hard_timeout_seconds": hard_timeout,
            "hard_timeout_source": hard_timeout_source,
        },
    )


def _create_supervisor_context(spec_path: Path) -> _SupervisorContext:
    """读取 spec 并建立监督阶段共享状态，不启动任何子进程。"""

    specification = json.loads(spec_path.read_text(encoding="utf-8"))
    raw_job_directory = _absolute_path_without_following_symlinks(
        specification["job_directory"]
    )
    if raw_job_directory.is_symlink():
        raise OSError(f"job directory cannot be a symlink: {raw_job_directory}")
    job_directory = raw_job_directory
    _ensure_private_directory(job_directory)
    resources = dict(specification.get("resource_limits") or {})
    execution_class, compute_timeout_policy, walltime, hard_timeout, timeout_policy = (
        _build_timeout_configuration(specification, resources)
    )
    queue_managed = isinstance(specification.get("gaussian_queue_ticket"), dict)
    return _SupervisorContext(
        specification=specification,
        job_directory=job_directory,
        status_path=_absolute_path_without_following_symlinks(specification["status_path"]),
        stdout_path=_absolute_path_without_following_symlinks(specification["stdout_path"]),
        stderr_path=_absolute_path_without_following_symlinks(specification["stderr_path"]),
        command=[str(item) for item in specification["command"]],
        resources=resources,
        resource_allocation=dict(specification.get("resource_allocation") or {}),
        evaluation_budget=dict(specification.get("evaluation_resource_budget") or {}),
        walltime=walltime,
        execution_class=execution_class,
        compute_timeout_policy=compute_timeout_policy,
        hard_timeout=hard_timeout,
        timeout_policy=timeout_policy,
        memory_limit_mb=_memory_limit_mb(
            resources, dict(specification.get("evaluation_resource_budget") or {})
        ),
        execution_started=not queue_managed,
        started_at=_now(),
        started_monotonic=time.monotonic(),
        cancellation_path=job_directory / "cancel_requested",
        supervisor_identity=capture_process_identity(os.getpid()),
    )


def _build_job_status_value(
    context: _SupervisorContext, state: str, extra: dict[str, Any]
) -> dict[str, Any]:
    """构造公开状态 JSON，集中保留身份、scope 和 operation 字段。"""

    specification = context.specification
    value = {
        "schema_version": 1,
        "job_id": specification["job_id"],
        "job_type": specification["job_type"],
        "status": state,
        "supervisor_pid": os.getpid(),
        "command": context.command,
        "job_directory": specification["relative_job_directory"],
        "stdout_path": specification["relative_stdout_path"],
        "stderr_path": specification["relative_stderr_path"],
        "resource_limits": context.resources,
        "resource_allocation": context.resource_allocation,
        "submitted_at": specification["submitted_at"],
        "started_at": (
            context.started_at
            if context.execution_started and state != "queued"
            else None
        ),
        "metadata": specification.get("metadata") or {},
        "execution_mode": "local",
        "scratch_directory": os.environ.get("TMPDIR"),
        "timeout_policy": context.timeout_policy,
        **extra,
    }
    value.update(_process_identity_fields("supervisor", context.supervisor_identity))
    if context.child_identity is not None:
        value.update(_process_identity_fields("child", context.child_identity))
    if specification.get("scope_id") is not None:
        value.update(
            {
                "scope_id": specification.get("scope_id"),
                "scope_workspace": specification.get("scope_workspace"),
            }
        )
    if specification.get("operation_id") is not None:
        value.update(
            {
                "operation_id": specification["operation_id"],
                "operation_fingerprint": specification.get("operation_fingerprint"),
                "submission_key": specification.get("submission_key"),
            }
        )
    return value


def _write_job_status(context: _SupervisorContext, state: str, **extra: Any) -> None:
    """写入当前状态并同步 operation receipt，保留公开字段和恢复语义。"""

    value = _build_job_status_value(context, state, extra)
    _atomic_json(context.status_path, value)
    if state in TERMINAL_JOB_STATES or state == "running":
        _update_operation_receipt(
            context.specification,
            context.job_directory,
            state,
            status=state,
            **{
                key: value.get(key)
                for key in (
                    "child_pid",
                    "child_start_time_ticks",
                    "child_start_ticks",
                    "child_boot_id",
                    "child_process_group_id",
                    "child_session_id",
                    "supervisor_pid",
                    "supervisor_start_time_ticks",
                    "supervisor_start_ticks",
                    "supervisor_boot_id",
                    "supervisor_process_group_id",
                    "supervisor_session_id",
                    "error",
                )
            },
        )


def _handle_supervisor_signal(
    context: _SupervisorContext, received_signal: int, _frame: Any
) -> None:
    """记录取消信号并只向已核验身份的 child 请求终止。"""

    context.cancellation_signal = received_signal
    if context.child is not None and context.child.poll() is None:
        _verified_process_group_signal(context.child_identity, signal.SIGTERM)


def _register_supervisor_signal_handlers(context: _SupervisorContext) -> None:
    """注册 supervisor 的可恢复 SIGTERM/SIGINT 处理器。"""

    handler = partial(_handle_supervisor_signal, context)
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


def _release_queue_state(context: _SupervisorContext) -> None:
    """释放 reservation 和 FIFO 条目；未知进程状态时保持资源占用。"""

    if context.preserve_queue_state:
        return
    if context.queue_reservation is not None:
        context.queue_reservation.release()
    if context.queue_ticket is not None:
        context.queue_ticket.release()


def _cancellation_requested(context: _SupervisorContext) -> bool:
    """判断取消信号或可信取消标记是否已出现。"""

    return context.cancellation_signal is not None or (
        context.cancellation_path.is_file()
        and not context.cancellation_path.is_symlink()
    )


def _cancelled_status(context: _SupervisorContext) -> None:
    """写入排队或启动前的取消状态。"""

    _write_job_status(
        context,
        "cancelled",
        finished_at=_now(),
        duration_seconds=0.0,
        return_code=None,
        error={"code": "cancelled", "message": "Job cancellation was requested"},
    )


def _parse_queue_ticket(context: _SupervisorContext) -> GaussianQueueTicket:
    """解析并校验持久 Gaussian FIFO ticket。"""

    value = context.specification["gaussian_queue_ticket"]
    return GaussianQueueTicket(
        ticket_id=int(value["ticket_id"]),
        owner_id=str(value["owner_id"]),
        scope_id=str(value["scope_id"]) if value.get("scope_id") is not None else None,
        workspace=str(value["workspace"]) if value.get("workspace") is not None else None,
    )


def _reserve_queued_gaussian_resources(
    context: _SupervisorContext,
) -> bool | None:
    """等待 Gaussian 队首并取得资源 reservation；容量暂不可用时继续等待。"""

    while True:
        if _cancellation_requested(context):
            return False
        reservation_result = _try_reserve_queued_gaussian_resources(context)
        if reservation_result is None:
            return None
        if reservation_result:
            return True
        time.sleep(0.2)


def _gaussian_request_exceeds_budget(
    context: _SupervisorContext, error: ResourceBudgetExceeded
) -> bool:
    """判断 Gaussian 请求是否超过总预算；超过时不能靠等待解决。"""

    return any(
        int(context.resources.get(name) or 0)
        > int(getattr(error.budget, name))
        for name in ("cpu_cores", "memory_mb", "gpu_count")
    )


def _apply_gaussian_resource_reservation(
    context: _SupervisorContext,
) -> None:
    """把 reservation 的资源和执行计时写入 supervisor 上下文。"""

    assert context.queue_reservation is not None
    context.resources = dict(context.queue_reservation.resource_limits) | {
        "walltime_seconds": context.resources.get(
            "walltime_seconds", context.walltime
        )
    }
    context.resource_allocation = dict(context.queue_reservation.resource_allocation)
    context.memory_limit_mb = _memory_limit_mb(
        context.resources, context.evaluation_budget
    )
    context.execution_started = True
    context.started_at = _now()
    context.started_monotonic = time.monotonic()
    _write_job_status(context, "queued")


def _try_reserve_queued_gaussian_resources(
    context: _SupervisorContext,
) -> bool | None:
    """执行一次 reservation 尝试；True 成功、False 可重试、None 终止。"""

    try:
        context.queue_reservation = reserve_resources(
            context.resources,
            kind=str(context.specification.get("job_type") or "native_software"),
            label=str(
                (context.specification.get("metadata") or {}).get("label")
                or context.command[0]
            ),
            job_id=str(context.specification["job_id"]),
        )
    except ResourceBudgetExceeded as exc:
        if not _gaussian_request_exceeds_budget(context, exc):
            return False
        _write_job_status(
            context,
            "failed",
            finished_at=_now(),
            duration_seconds=0.0,
            return_code=None,
            error=exc.as_error(),
        )
        return None
    _apply_gaussian_resource_reservation(context)
    return True


def _prepare_gaussian_queue(context: _SupervisorContext) -> int | None:
    """处理 Gaussian FIFO，返回提前结束码或 None 继续执行。"""

    if not isinstance(context.specification.get("gaussian_queue_ticket"), dict):
        return None
    try:
        context.queue_ticket = _parse_queue_ticket(context)
    except (KeyError, TypeError, ValueError):
        _write_job_status(
            context,
            "failed",
            finished_at=_now(),
            duration_seconds=0.0,
            return_code=None,
            error={
                "code": "submission_state_invalid",
                "message": "The submitted computation state is invalid",
            },
        )
        return 1
    bind_gaussian_queue_owner(context.queue_ticket)
    _write_job_status(context, "queued")
    acquired = wait_for_gaussian_turn(
        context.queue_ticket,
        should_cancel=lambda: _cancellation_requested(context),
    )
    if not acquired:
        return _finish_failed_gaussian_queue_wait(context)
    reserved = _reserve_queued_gaussian_resources(context)
    if reserved is None:
        return 1
    if not reserved:
        _cancelled_status(context)
        return 2
    return None


def _finish_failed_gaussian_queue_wait(context: _SupervisorContext) -> int:
    """把 FIFO 等待失败路由为取消或提交状态丢失。"""

    if _cancellation_requested(context):
        _cancelled_status(context)
        return 2
    _write_job_status(
        context,
        "failed",
        finished_at=_now(),
        duration_seconds=0.0,
        return_code=None,
        error={
            "code": "submission_state_missing",
            "message": "The submitted computation state is unavailable",
        },
    )
    return 1


def _validate_launch_inputs(context: _SupervisorContext) -> int | None:
    """检查取消标记和 allowlisted 可执行文件快照。"""

    mismatches = _verify_launch_file_snapshots(
        list(context.specification.get("launch_file_snapshots") or [])
    )
    if mismatches:
        _write_job_status(
            context,
            "failed",
            finished_at=_now(),
            duration_seconds=0.0,
            return_code=None,
            error={
                "code": "execution_file_changed_before_launch",
                "message": "An allowlisted executable changed after preflight validation",
                "mismatches": mismatches,
            },
        )
        return 1
    if _cancellation_requested(context):
        _cancelled_status(context)
        return 2
    return None


def _open_stdin_safely(path: Path):
    """以 O_NOFOLLOW 打开普通 stdin 文件，拒绝 symlink 输入。"""

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise OSError(f"stdin path is not a regular file: {path}")
        return os.fdopen(descriptor, "rb")
    except Exception:
        os.close(descriptor)
        raise


def _launch_child_process(context: _SupervisorContext) -> int | None:
    """用最小环境启动 child，并在身份不可核验时进入可恢复状态。"""

    stdin_handle = None
    try:
        stdin_path = context.specification.get("stdin_path")
        if stdin_path:
            stdin_handle = _open_stdin_safely(Path(stdin_path))
        child_environment = _build_child_environment(
            context.specification, context.job_directory
        )
        command = context.command
        if context.specification.get("job_type") == "programmable_analysis":
            command, child_environment = _build_programmable_analysis_sandbox_command(
                context,
                child_environment,
            )
        with _open_private_append(context.stdout_path) as stdout_handle, _open_private_append(
            context.stderr_path
        ) as stderr_handle:
            context.child = subprocess.Popen(
                command,
                cwd=context.job_directory,
                stdin=stdin_handle if stdin_handle is not None else subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                env=child_environment,
                shell=False,
                start_new_session=True,
                close_fds=True,
                preexec_fn=_preexec(
                    context.resources,
                    context.evaluation_budget,
                    context.resource_allocation,
                ),
            )
    except _ProgrammableAnalysisSandboxError as exc:
        _write_job_status(
            context,
            "failed",
            finished_at=_now(),
            duration_seconds=round(time.monotonic() - context.started_monotonic, 6),
            return_code=None,
            error={"code": exc.code, "message": exc.message},
        )
        return 1
    except Exception as exc:
        _write_job_status(
            context,
            "failed",
            finished_at=_now(),
            duration_seconds=round(time.monotonic() - context.started_monotonic, 6),
            return_code=None,
            error={"code": "process_start_failed", "message": str(exc)},
        )
        return 1
    finally:
        if stdin_handle is not None:
            stdin_handle.close()
    context.child_identity = capture_process_identity(context.child.pid)
    if _process_identity_complete(context.child_identity):
        return None
    child_finished = _finish_unverified_child_safely(context.child)
    if not child_finished:
        context.preserve_queue_state = True
        _write_job_status(
            context,
            "running",
            supervisor_exit_pending=True,
            finished_at=None,
            duration_seconds=round(
                time.monotonic() - context.started_monotonic, 6
            ),
            return_code=None,
            error={
                "code": "child_identity_unavailable",
                "message": (
                    "Child process identity cannot be verified; FIFO and resource "
                    "state remain held for manual recovery"
                ),
            },
        )
    else:
        _write_job_status(
            context,
            "failed",
            finished_at=_now(),
            duration_seconds=round(
                time.monotonic() - context.started_monotonic, 6
            ),
            return_code=context.child.returncode,
            error={
                "code": "child_identity_unavailable",
                "message": "Child process identity cannot be verified",
            },
        )
    return 1


def _update_cpu_progress(
    previous: dict[tuple[int, int], int] | None,
    maximum: int | None,
    snapshot: dict[tuple[int, int], int] | None,
) -> tuple[dict[tuple[int, int], int] | None, int | None, bool]:
    """合并进程组 CPU ticks，并返回本轮是否观察到进展。"""

    if snapshot is None:
        return previous, maximum, False
    if previous is None:
        totals = dict(snapshot)
        return totals, sum(totals.values()), False
    previous_maximum = maximum or 0
    totals = dict(previous)
    for identity, cpu_ticks in snapshot.items():
        totals[identity] = max(totals.get(identity, 0), cpu_ticks)
    updated_maximum = sum(totals.values())
    return totals, updated_maximum, updated_maximum > previous_maximum


def _timeout_reason(
    context: _SupervisorContext, elapsed_seconds: float, idle_seconds: float
) -> str | None:
    """根据固定策略判断 hard、idle 或 wall-clock 超时。"""

    if context.compute_timeout_policy:
        if elapsed_seconds > context.hard_timeout:
            return "hard_timeout"
        if idle_seconds > context.walltime:
            return "no_observable_progress"
        return None
    return "walltime_exceeded" if elapsed_seconds > context.walltime else None


def _monitor_child_process(context: _SupervisorContext) -> dict[str, Any]:
    """轮询输出、CPU、内存和取消状态，直到 child 退出或触发停止条件。"""

    child = context.child
    assert child is not None
    last_progress_at = _now()
    last_progress_monotonic = time.monotonic()
    output_paths = (context.stdout_path, context.stderr_path)
    last_output_state = _output_file_state(output_paths)
    process_group_id = (
        context.child_identity.process_group_id
        if context.child_identity is not None
        and context.child_identity.process_group_id is not None
        else child.pid
    )
    cpu_totals = (
        _process_group_cpu_ticks(process_group_id)
        if context.compute_timeout_policy
        else None
    )
    max_cpu_ticks = sum(cpu_totals.values()) if cpu_totals is not None else None
    memory_exceeded = False
    peak_rss = 0
    timeout_reason: str | None = None
    while child.poll() is None:
        if _cancellation_requested(context) and context.cancellation_signal is None:
            context.cancellation_signal = signal.SIGTERM
            _verified_process_group_signal(context.child_identity, signal.SIGTERM)
        if context.cancellation_signal is not None:
            break
        now_monotonic = time.monotonic()
        output_state = _output_file_state(output_paths)
        if output_state != last_output_state:
            last_output_state = output_state
            last_progress_monotonic = now_monotonic
            last_progress_at = _now()
        snapshot = (
            _process_group_cpu_ticks(process_group_id)
            if context.compute_timeout_policy
            else None
        )
        cpu_totals, max_cpu_ticks, cpu_progressed = _update_cpu_progress(
            cpu_totals, max_cpu_ticks, snapshot
        )
        if cpu_progressed:
            last_progress_monotonic = now_monotonic
            last_progress_at = _now()
        timeout_reason = _timeout_reason(
            context,
            now_monotonic - context.started_monotonic,
            now_monotonic - last_progress_monotonic,
        )
        if timeout_reason is not None:
            _verified_process_group_signal(context.child_identity, signal.SIGTERM)
            break
        rss = _process_group_rss_kb(process_group_id)
        if rss is not None:
            peak_rss = max(peak_rss, rss)
            if context.memory_limit_mb is not None and rss > context.memory_limit_mb * 1024:
                memory_exceeded = True
                _verified_process_group_signal(context.child_identity, signal.SIGTERM)
                break
        time.sleep(0.2)
    return {
        "timed_out": timeout_reason is not None,
        "timeout_reason": timeout_reason,
        "last_progress_at": last_progress_at,
        "max_process_group_cpu_ticks": max_cpu_ticks,
        "memory_exceeded": memory_exceeded,
        "peak_process_group_rss_kb": peak_rss,
    }


def _collect_child_observation(
    context: _SupervisorContext, raw: dict[str, Any]
) -> _ChildObservation:
    """有界回收 child，并核验其进程组已清空后生成最终观测。"""

    child = context.child
    assert child is not None
    child_reap_failed = False
    if child.poll() is None and not _wait_for_child_reap_safely(
        child, context.child_identity
    ):
        context.preserve_queue_state = True
        child_reap_failed = True
    background_cleanup = "none_detected"
    if (
        context.child_identity is not None
        and _process_identity_complete(context.child_identity)
        and process_identity_activity(context.child_identity.as_dict()) == "alive"
    ):
        background_cleanup = "terminated_remaining_process_group"
        _verified_process_group_signal(context.child_identity, signal.SIGTERM)
        time.sleep(0.2)
        _verified_process_group_signal(context.child_identity, signal.SIGKILL)
    group_activity = "unknown"
    if context.child_identity is not None and _process_group_identity_complete(
        context.child_identity
    ):
        group_activity = inspect_process_group_membership(
            context.child_identity.process_group_id,
            context.child_identity.session_id,
            context.child_identity.boot_id,
        )
    if group_activity != "empty":
        context.preserve_queue_state = True
        background_cleanup = (
            "remaining_process_group"
            if group_activity == "nonempty"
            else "process_group_identity_unavailable"
        )
    return _ChildObservation(
        timed_out=bool(raw["timed_out"]),
        timeout_reason=raw["timeout_reason"],
        last_progress_at=str(raw["last_progress_at"]),
        max_process_group_cpu_ticks=raw["max_process_group_cpu_ticks"],
        memory_exceeded=bool(raw["memory_exceeded"]),
        child_reap_failed=child_reap_failed,
        peak_process_group_rss_kb=int(raw["peak_process_group_rss_kb"]),
        background_process_cleanup=background_cleanup,
        process_group_activity=group_activity,
    )


def _common_child_status(
    context: _SupervisorContext, observation: _ChildObservation
) -> dict[str, Any]:
    """构造成功、失败、超时和待恢复状态共用的资源字段。"""

    child = context.child
    assert child is not None
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {
        "finished_at": _now(),
        "duration_seconds": round(time.monotonic() - context.started_monotonic, 6),
        "return_code": child.returncode,
        "background_process_cleanup": observation.background_process_cleanup,
        "resource_usage": {
            "user_cpu_seconds": round(usage.ru_utime, 6),
            "system_cpu_seconds": round(usage.ru_stime, 6),
            "max_rss_kb": usage.ru_maxrss,
            "peak_process_group_rss_kb": observation.peak_process_group_rss_kb,
            "max_process_group_cpu_ticks": observation.max_process_group_cpu_ticks,
            "memory_limit_mb": context.memory_limit_mb,
        },
    }


def _read_completed_output_text(path: Path) -> str | None:
    """读取已结束 child 的普通输出文件；路径异常时返回 None。"""

    if path.is_symlink() or not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _build_gaussian_archive_warning_status_extra(
    context: _SupervisorContext,
    observation: _ChildObservation,
    common: dict[str, Any],
) -> dict[str, Any] | None:
    """判断 native Gaussian 归档警告，并返回成功状态的 warning/provenance。"""

    if context.specification.get("job_type") != "native_software":
        return None
    metadata = context.specification.get("metadata")
    if (
        not isinstance(metadata, dict)
        or str(metadata.get("software_id") or "").casefold() != "gaussian"
    ):
        return None
    stdout = _read_completed_output_text(context.stdout_path)
    stderr = _read_completed_output_text(context.stderr_path)
    if stdout is None or stderr is None:
        return None
    child = context.child
    assert child is not None
    completed = {
        **common,
        "returncode": child.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "timed_out": observation.timed_out,
        "memory_exceeded": observation.memory_exceeded,
        "child_reap_failed": observation.child_reap_failed,
        "background_process_cleanup": observation.background_process_cleanup,
        "process_group_activity": observation.process_group_activity,
        "cancelled": context.cancellation_signal is not None,
    }
    if not gaussian_archive_only_completion(completed):
        return None
    return {
        "warning": {
            "code": "gaussian_archive_warning",
            "message": "Gaussian 正常终止，但归档条目写入返回了退出码 1",
        },
        "provenance": gaussian_archive_warning_provenance(child.returncode),
    }


def _timeout_error(
    context: _SupervisorContext, observation: _ChildObservation
) -> dict[str, Any]:
    """生成与旧状态协议一致的超时错误对象。"""

    reason = observation.timeout_reason
    if reason == "no_observable_progress":
        return {
            "code": "no_progress_timeout",
            "termination_reason": reason,
            "message": (
                "Job produced no observable stdout/stderr file or process-group CPU "
                f"progress for {context.walltime} seconds"
            ),
            "last_progress_at": observation.last_progress_at,
            "timeout_policy": context.timeout_policy,
        }
    if reason == "hard_timeout":
        return {
            "code": "hard_timeout_exceeded",
            "termination_reason": reason,
            "message": (
                f"Job exceeded the {context.hard_timeout}-second hard safety limit"
            ),
            "last_progress_at": observation.last_progress_at,
            "timeout_policy": context.timeout_policy,
        }
    return {
        "code": "walltime_exceeded",
        "message": f"Job exceeded {context.walltime} seconds",
    }


def _finalize_child_result(
    context: _SupervisorContext, observation: _ChildObservation
) -> int:
    """根据清理结果写入终态或 waiting 状态，保持原返回码约定。"""

    child = context.child
    assert child is not None
    common = _common_child_status(context, observation)
    if observation.child_reap_failed:
        _write_job_status(
            context,
            "running",
            **common,
            supervisor_exit_pending=True,
            error={
                "code": "child_reap_unavailable",
                "message": (
                    "Child process did not exit after bounded cleanup; process identity "
                    "or signal state requires manual recovery"
                ),
            },
        )
        return 1
    if observation.process_group_activity != "empty":
        pending = dict(common)
        pending["finished_at"] = None
        if observation.process_group_activity == "nonempty":
            error = {
                "code": "process_group_descendants_alive",
                "message": (
                    "Child leader exited but its verified process group still contains "
                    "a descendant; FIFO and resource state remain held"
                ),
            }
        else:
            error = {
                "code": "process_group_identity_unavailable",
                "message": (
                    "Child process-group identity cannot be verified; FIFO and resource "
                    "state remain held for manual recovery"
                ),
            }
        _write_job_status(
            context,
            "running",
            **pending,
            supervisor_exit_pending=True,
            process_group_activity=observation.process_group_activity,
            error=error,
        )
        return 1
    if context.cancellation_signal is not None:
        _write_job_status(
            context,
            "cancelled",
            **common,
            error={"code": "cancelled", "message": "Job cancellation was requested"},
        )
        return 2
    if observation.timed_out:
        _write_job_status(
            context,
            "timeout",
            **common,
            termination_reason=observation.timeout_reason,
            error=_timeout_error(context, observation),
        )
        return 3
    if observation.memory_exceeded:
        _write_job_status(
            context,
            "failed",
            **common,
            error={
                "code": "memory_limit_exceeded",
                "message": (
                    f"Process group exceeded the {context.memory_limit_mb} MB job memory limit"
                ),
            },
        )
        return 5
    if child.returncode == 0:
        _write_job_status(context, "success", **common, error=None)
        return 0
    archive_warning = _build_gaussian_archive_warning_status_extra(
        context,
        observation,
        common,
    )
    if archive_warning is not None:
        _write_job_status(
            context,
            "success",
            **common,
            **archive_warning,
            error=None,
        )
        return 0
    _write_job_status(
        context,
        "failed",
        **common,
        error={
            "code": "nonzero_exit",
            "message": f"Program exited with status {child.returncode}",
        },
    )
    return 4


def _execute_child_job(context: _SupervisorContext) -> int:
    """启动、监控并收束一个已完成排队和预检的 child 作业。"""

    launch_result = _launch_child_process(context)
    if launch_result is not None:
        return launch_result
    assert context.child is not None
    _write_job_status(context, "running", child_pid=context.child.pid, return_code=None)
    raw_observation = _monitor_child_process(context)
    observation = _collect_child_observation(context, raw_observation)
    return _finalize_child_result(context, observation)


def _run_supervised_job(spec_path: Path) -> int:
    """按固定顺序执行 queue、预检、child 监控和终态收束。"""

    context = _create_supervisor_context(spec_path)
    _register_supervisor_signal_handlers(context)
    if not _process_identity_complete(context.supervisor_identity):
        _write_job_status(
            context,
            "failed",
            finished_at=_now(),
            duration_seconds=0.0,
            return_code=None,
            error={
                "code": "supervisor_identity_unavailable",
                "message": "Supervisor process identity cannot be verified",
            },
        )
        return 1
    if not _claim_supervisor(context.specification, context.job_directory):
        return 0
    try:
        queue_result = _prepare_gaussian_queue(context)
        if queue_result is not None:
            return queue_result
        preflight_result = _validate_launch_inputs(context)
        if preflight_result is not None:
            return preflight_result
        return _execute_child_job(context)
    finally:
        _release_queue_state(context)


def _gaussian_gate_path(specification: dict[str, Any]) -> Path | None:
    """读取服务传入的 Gaussian 锁路径；旧 spec 缺失时按作业目录兼容推导。"""

    raw_path = specification.get("gaussian_serial_gate_path")
    if raw_path:
        return _absolute_path_without_following_symlinks(str(raw_path))
    raw_job_directory = specification.get("job_directory")
    if not raw_job_directory:
        return None
    job_directory = _absolute_path_without_following_symlinks(str(raw_job_directory))
    if len(job_directory.parents) < 3:
        return None
    return job_directory.parents[2] / "outputs" / ".resource_budget" / "gaussian.serial.lock"


def supervise(spec_path: Path) -> int:
    """为 Gaussian 子进程持有跨进程串行锁，其他作业沿用原监督流程。"""

    specification = json.loads(spec_path.read_text(encoding="utf-8"))
    metadata = specification.get("metadata") or {}
    if metadata.get("software_id") != "gaussian":
        return _run_supervised_job(spec_path)
    if specification.get("gaussian_queue_ticket") is not None:
        # 新提交通过持久 FIFO 队列取得队首；旧 gate 仅服务没有 queue 身份
        # 的手工/历史 spec。
        return _run_supervised_job(spec_path)
    gate_path = _gaussian_gate_path(specification)
    if gate_path is None:
        # 服务新建的 Gaussian spec 总会带锁路径；旧手工 spec 无法确定共享工作区时，
        # 继续旧行为以保持查询和恢复兼容，不把非 Gaussian 作业误锁住。
        return _run_supervised_job(spec_path)
    _ensure_private_directory(gate_path.parent)
    with _open_private_lock(gate_path) as gate_handle:
        fcntl.flock(gate_handle.fileno(), fcntl.LOCK_EX)
        try:
            return _run_supervised_job(spec_path)
        finally:
            fcntl.flock(gate_handle.fileno(), fcntl.LOCK_UN)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: job_supervisor.py JOB_SPEC.json", file=sys.stderr)
        return 64
    return supervise(Path(sys.argv[1]).resolve())


if __name__ == "__main__":
    raise SystemExit(main())
