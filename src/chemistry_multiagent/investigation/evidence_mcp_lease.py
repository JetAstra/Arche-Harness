"""为一次 Investigation task 管理只读 Evidence MCP 子进程租约。

这个模块只负责一个已经由 ``ReviewEvidenceStore`` 登记的快照服务进程：它在
loopback 端口启动现有 ``evidence_mcp_server``，把 capability 放进私有文件，
用真实 MCP 初始化和工具目录请求确认服务边界，并在结束时只回收能够用 PID、
Linux start time、boot id 以及命令行核对的那个子进程。模块不创建快照、不读取
证据正文、不提供 MiniChem、shell、搜索或写入工具。
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import ipaddress
import json
import os
import re
import signal
import socket
import stat
import subprocess
import sys
import time
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from types import TracebackType
from typing import Any, Iterator, Mapping, Sequence, Type

from .evidence import (
    EvidenceRequestError,
    EvidenceRegistryCorruption,
    EvidenceSnapshotNotFound,
    ReviewEvidenceStore,
)
from .evidence_mcp_server import (
    DEFAULT_EVIDENCE_CAPABILITY_HEADER,
    EvidenceMcpConfigurationError,
    create_evidence_capability_file,
    load_evidence_capability_file,
)

try:
    from minichem_toolbox.process_identity import (
        capture_process_identity,
        process_identity_activity,
    )
except ImportError:  # pragma: no cover - 标准仓库环境总是提供该模块
    capture_process_identity = None  # type: ignore[assignment]
    process_identity_activity = None  # type: ignore[assignment]


__all__ = [
    "EvidenceMcpLeaseError",
    "EvidenceMcpLeaseConfigurationError",
    "EvidenceMcpLeaseIdentityError",
    "EvidenceMcpLeasePortError",
    "EvidenceMcpLeaseStartupError",
    "TaskEvidenceMcpLease",
]


_OPAQUE_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{7,127}$")
_CAPABILITY_RE = re.compile(r"^[\x21-\x7e]{32,512}$")
_HEADER_NAME_RE = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_MAX_RECEIPT_BYTES = 32 * 1024
_SERVER_MODULE = "chemistry_multiagent.investigation.evidence_mcp_server"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_STANDARD_PYTHON = _REPO_ROOT / ".venv" / "bin" / "python"
_STANDARD_PYTHON_REAL = _STANDARD_PYTHON.resolve(strict=False)
_BROAD_PATHS = {Path("/"), Path("/tmp"), Path("/var/tmp")}
_DEFAULT_STARTUP_TIMEOUT = 8.0
_DEFAULT_SHUTDOWN_TIMEOUT = 3.0
_DEFAULT_READINESS_REQUEST_TIMEOUT = 1.5
_PR_SET_PDEATHSIG = 1


class EvidenceMcpLeaseError(RuntimeError):
    """Evidence MCP 租约无法安全启动、核验或回收。"""


class EvidenceMcpLeaseConfigurationError(EvidenceMcpLeaseError):
    """租约构造参数或本地私有运行目录不满足安全约束。"""


class EvidenceMcpLeaseIdentityError(EvidenceMcpLeaseError):
    """无法证明目标 PID 仍然是本租约启动的进程。"""


class EvidenceMcpLeasePortError(EvidenceMcpLeaseError):
    """loopback 端口已被未知监听者占用或无法安全探测。"""


class EvidenceMcpLeaseStartupError(EvidenceMcpLeaseError):
    """Evidence MCP 子进程未在截止时间内通过真实 MCP readiness。"""


def _configure_child_parent_death_signal(parent_pid: int) -> None:
    """在 Linux 子进程中设置父进程退出时的 TERM，避免测试中遗留监听者。"""

    if sys.platform != "linux":
        return
    try:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        result = libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
        if result != 0:
            error_number = ctypes.get_errno()
            raise OSError(error_number, "prctl PR_SET_PDEATHSIG failed")
        if os.getppid() != parent_pid:
            os._exit(143)
    except (ImportError, OSError, AttributeError):
        os._exit(127)


def _reject_symlinked_path_components(path: Path, *, label: str) -> None:
    """拒绝路径本身和所有父目录中的符号链接。"""

    try:
        absolute = Path(os.path.abspath(os.fspath(path)))
        cursor = Path(absolute.anchor or os.curdir)
        components = absolute.parts[1:] if absolute.anchor else absolute.parts
        for component in components:
            cursor /= component
            if cursor.is_symlink():
                raise EvidenceMcpLeaseConfigurationError(f"{label} contains a symlink")
    except EvidenceMcpLeaseConfigurationError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise EvidenceMcpLeaseConfigurationError(f"{label} is invalid") from exc


def _validate_private_directory(path: Path, *, label: str, create: bool) -> Path:
    """准备 0700 私有目录；失败时不返回可继续使用的路径。"""

    if not path.is_absolute():
        raise EvidenceMcpLeaseConfigurationError(f"{label} must be absolute")
    try:
        normalized_path = Path(os.path.abspath(os.fspath(path)))
    except (OSError, TypeError, ValueError) as exc:
        raise EvidenceMcpLeaseConfigurationError(f"{label} is invalid") from exc
    if normalized_path in _BROAD_PATHS or len(normalized_path.parts) <= 2:
        raise EvidenceMcpLeaseConfigurationError(f"{label} is too broad")
    _reject_symlinked_path_components(path, label=label)
    path = normalized_path
    try:
        if create:
            path.mkdir(parents=True, exist_ok=True, mode=0o700)
        if path.is_symlink() or not path.is_dir():
            raise EvidenceMcpLeaseConfigurationError(f"{label} is invalid")
        os.chmod(path, 0o700)
    except EvidenceMcpLeaseConfigurationError:
        raise
    except OSError as exc:
        raise EvidenceMcpLeaseConfigurationError(f"{label} is unavailable") from exc
    return path


def _validate_task_id(value: Any) -> str:
    """确认 task_id 是不含路径和控制字符的稳定任务标识。"""

    if not isinstance(value, str):
        raise EvidenceMcpLeaseConfigurationError("task_id is invalid")
    normalized = value.strip()
    if (
        not normalized
        or _CONTROL_RE.search(normalized)
        or "/" in normalized
        or "\\" in normalized
        or normalized in {".", ".."}
    ):
        raise EvidenceMcpLeaseConfigurationError("task_id is invalid")
    return normalized


def _validate_snapshot_id(value: Any) -> str:
    """确认启动参数只能是 Evidence registry 的 opaque snapshot ID。"""

    if not isinstance(value, str) or _OPAQUE_ID_RE.fullmatch(value) is None:
        raise EvidenceMcpLeaseConfigurationError("snapshot_id is invalid")
    return value


def _validate_capability_token(value: Any) -> str:
    """确认 capability 可安全写入 0600 文件，不在异常中复制原文。"""

    if not isinstance(value, str) or _CAPABILITY_RE.fullmatch(value) is None:
        raise EvidenceMcpLeaseConfigurationError("capability token is invalid")
    return value


def _validate_header_name(value: Any) -> str:
    """规范化 capability header 名称并拒绝控制字符。"""

    if not isinstance(value, str):
        raise EvidenceMcpLeaseConfigurationError("capability header is invalid")
    normalized = value.strip().lower()
    if _HEADER_NAME_RE.fullmatch(normalized) is None:
        raise EvidenceMcpLeaseConfigurationError("capability header is invalid")
    return normalized


def _validate_loopback_host(value: Any) -> str:
    """只允许字面量 IPv4/IPv6 loopback，避免 DNS 解析扩大监听范围。"""

    if not isinstance(value, str) or not value.strip():
        raise EvidenceMcpLeaseConfigurationError("host must be a loopback address")
    normalized = value.strip()
    try:
        address = ipaddress.ip_address(normalized)
    except ValueError as exc:
        raise EvidenceMcpLeaseConfigurationError("host must be a loopback address") from exc
    if not address.is_loopback:
        raise EvidenceMcpLeaseConfigurationError("host must be a loopback address")
    return normalized


def _validate_port(value: Any) -> int:
    """确认使用固定的 TCP 端口而不是让服务随机绑定。"""

    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535:
        raise EvidenceMcpLeaseConfigurationError("port is invalid")
    return value


def _resolve_lease_python_executable(
    python_executable: str | os.PathLike[str] | None,
    python_path: str | os.PathLike[str] | None,
) -> Path:
    """解析并核对租约只能使用仓库标准虚拟环境解释器。"""

    if python_executable is not None and python_path is not None:
        raise EvidenceMcpLeaseConfigurationError("python executable is specified twice")
    selected = python_executable if python_executable is not None else python_path
    selected_path = Path(selected) if selected is not None else _STANDARD_PYTHON
    if not selected_path.is_absolute():
        raise EvidenceMcpLeaseConfigurationError("python executable must be absolute")
    _reject_symlinked_path_components(selected_path.parent, label="python executable")
    try:
        resolved_python = selected_path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise EvidenceMcpLeaseConfigurationError("python executable is unavailable") from exc
    if (
        selected_path.parent != _STANDARD_PYTHON.parent
        or selected_path.name not in {_STANDARD_PYTHON.name, "python3.11"}
        or resolved_python != _STANDARD_PYTHON_REAL
        or not os.access(selected_path, os.X_OK)
    ):
        raise EvidenceMcpLeaseConfigurationError("python executable must be the repository .venv Python")
    return selected_path


def _prepare_lease_roots(
    store_root: str | os.PathLike[str],
    runtime_root: str | os.PathLike[str],
) -> tuple[Path, Path]:
    """规范化并准备相互隔离的私有 store、runtime 目录。"""

    normalized_store = Path(store_root).expanduser()
    normalized_runtime = Path(runtime_root).expanduser()
    if not normalized_store.is_absolute():
        raise EvidenceMcpLeaseConfigurationError("store_root must be absolute")
    if not normalized_runtime.is_absolute():
        raise EvidenceMcpLeaseConfigurationError("runtime_root must be absolute")
    _reject_symlinked_path_components(normalized_store, label="store_root")
    normalized_store = _validate_private_directory(
        normalized_store,
        label="store_root",
        create=False,
    )
    normalized_runtime = _validate_private_directory(
        normalized_runtime,
        label="runtime_root",
        create=True,
    )
    try:
        roots_overlap = normalized_store == normalized_runtime or normalized_runtime.is_relative_to(
            normalized_store
        )
    except AttributeError:  # pragma: no cover - Python 3.11 已提供 is_relative_to
        roots_overlap = str(normalized_runtime).startswith(str(normalized_store) + os.sep)
    if roots_overlap:
        raise EvidenceMcpLeaseConfigurationError("store_root and runtime_root must be separate")
    return normalized_store, normalized_runtime


def _validate_lease_timeouts(
    startup_timeout: float,
    shutdown_timeout: float,
    readiness_request_timeout: float,
) -> tuple[float, float, float]:
    """校验租约启动、停止和 readiness 请求超时。"""

    values = (
        (startup_timeout, "startup_timeout"),
        (shutdown_timeout, "shutdown_timeout"),
        (readiness_request_timeout, "readiness_request_timeout"),
    )
    for value, label in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise EvidenceMcpLeaseConfigurationError(f"{label} is invalid")
    return float(startup_timeout), float(shutdown_timeout), float(readiness_request_timeout)


def _digest_text(value: str) -> str:
    """返回不可逆摘要；只用于 receipt 配置核对，不用于认证。"""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _digest_json(value: Any) -> str:
    """对无 secret 的规范 JSON 配置计算摘要。"""

    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return _digest_text(encoded)


def _safe_file_mode(path: Path, *, label: str) -> None:
    """确认文件是普通非符号链接且权限恰为私有读写。"""

    _reject_symlinked_path_components(path, label=label)
    try:
        if path.is_symlink() or not path.is_file():
            raise EvidenceMcpLeaseConfigurationError(f"{label} is invalid")
        mode = stat.S_IMODE(path.stat().st_mode)
    except EvidenceMcpLeaseConfigurationError:
        raise
    except OSError as exc:
        raise EvidenceMcpLeaseConfigurationError(f"{label} is unavailable") from exc
    if mode != 0o600:
        raise EvidenceMcpLeaseConfigurationError(f"{label} must be mode 0600")


def _write_private_json(path: Path, value: Mapping[str, Any]) -> None:
    """以 0600 临时文件原子写入不含 secret 的身份 receipt。"""

    _reject_symlinked_path_components(path.parent, label="receipt directory")
    temporary = path.with_name(f".{path.name}.tmp")
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
    descriptor = -1
    created_temporary = False
    try:
        descriptor = os.open(
            temporary,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        created_temporary = True
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    except OSError as exc:
        raise EvidenceMcpLeaseError("lease receipt cannot be written") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if created_temporary:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass


def _read_private_json(path: Path) -> dict[str, Any] | None:
    """读取严格 0600 的 receipt；非法内容按身份不可核对处理。"""

    if not path.exists():
        return None
    _safe_file_mode(path, label="lease receipt")
    try:
        if path.stat().st_size > _MAX_RECEIPT_BYTES:
            raise EvidenceMcpLeaseIdentityError("lease receipt is invalid")
        text = path.read_text(encoding="ascii")
        value = json.loads(text)
    except EvidenceMcpLeaseIdentityError:
        raise
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise EvidenceMcpLeaseIdentityError("lease receipt is invalid") from exc
    if not isinstance(value, dict):
        raise EvidenceMcpLeaseIdentityError("lease receipt is invalid")
    return value


def _open_private_log(path: Path) -> int:
    """创建或清空 0600 日志文件，并拒绝符号链接替换。"""

    _reject_symlinked_path_components(path.parent, label="log directory")
    try:
        descriptor = os.open(
            path,
            os.O_CREAT
            | os.O_WRONLY
            | os.O_TRUNC
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        os.fchmod(descriptor, 0o600)
        return descriptor
    except OSError as exc:
        raise EvidenceMcpLeaseConfigurationError("lease log cannot be opened") from exc


def _normalize_identity(identity: Any) -> dict[str, Any]:
    """把 process_identity 的 dataclass 或 mapping 规范成 receipt 字段。"""

    if hasattr(identity, "as_dict"):
        value = identity.as_dict()
    elif isinstance(identity, Mapping):
        value = dict(identity)
    else:
        raise EvidenceMcpLeaseIdentityError("process identity is invalid")
    try:
        pid = int(value.get("pid") or 0)
        start_time_ticks = int(value.get("start_time_ticks") or 0)
        boot_id = str(value.get("boot_id") or "").strip()
    except (TypeError, ValueError) as exc:
        raise EvidenceMcpLeaseIdentityError("process identity is invalid") from exc
    if pid <= 0 or start_time_ticks <= 0 or not boot_id or _CONTROL_RE.search(boot_id):
        raise EvidenceMcpLeaseIdentityError("process identity is unavailable")
    return {
        "pid": pid,
        "start_time_ticks": start_time_ticks,
        "boot_id": boot_id,
    }


def _read_process_command_line(pid: int) -> tuple[str, ...] | None:
    """读取 Linux proc 命令行用于辅助确认 PID 没有被伪造复用。"""

    try:
        raw = (Path("/proc") / str(int(pid)) / "cmdline").read_bytes()
    except (FileNotFoundError, OSError, ValueError):
        return None
    if not raw:
        return None
    try:
        values = tuple(item.decode("utf-8") for item in raw.split(b"\0") if item)
    except UnicodeDecodeError:
        return None
    return values or None


def _same_command_line(actual: Sequence[str] | None, expected: Sequence[str]) -> bool:
    """以完整 argv 比较命令，避免只核对进程名而误收回其他进程。"""

    if actual is None:
        return False
    if len(actual) != len(expected):
        return False
    try:
        return os.path.realpath(actual[0]) == os.path.realpath(expected[0]) and tuple(actual[1:]) == tuple(
            expected[1:]
        )
    except (TypeError, ValueError):
        return False


def _reap_direct_child_process(process: subprocess.Popen[bytes] | None) -> bool:
    """用直接子句柄回收已结束进程，避免向未知 PID 发送信号。"""

    if process is None or process.poll() is None:
        return False
    with contextlib.suppress(Exception):
        process.wait(timeout=0)
    return True


class TaskEvidenceMcpLease:
    """一个 task 在固定 loopback 端口上的单快照 Evidence MCP 进程租约。

    构造参数显式绑定 task、证据 store、私有 runtime 目录、固定 loopback 地址、
    标准仓库 ``.venv/bin/python`` 和 capability。调用
    :meth:`serve_evidence_snapshot` 会先用公开 manifest 核对快照存在且属于该
    task，再启动现有服务并执行真实 MCP ``initialize``/``list_tools`` readiness。
    正常结束只会终止同一 PID、start time、boot id 和完整命令行都能核对的子进程；
    无法核对时保守失败，不盲杀未知监听者。
    """

    def __init__(
        self,
        task_id: str,
        store_root: str | os.PathLike[str],
        runtime_root: str | os.PathLike[str],
        *,
        host: str = "127.0.0.1",
        port: int = 9020,
        python_executable: str | os.PathLike[str] | None = None,
        python_path: str | os.PathLike[str] | None = None,
        capability_token: str,
        capability_header: str = DEFAULT_EVIDENCE_CAPABILITY_HEADER,
        startup_timeout: float = _DEFAULT_STARTUP_TIMEOUT,
        shutdown_timeout: float = _DEFAULT_SHUTDOWN_TIMEOUT,
        readiness_request_timeout: float = _DEFAULT_READINESS_REQUEST_TIMEOUT,
    ) -> None:
        """校验并保存租约配置，不启动进程也不读取证据正文。

        ``store_root`` 必须是当前 ReviewEvidenceStore 的专用目录，``runtime_root``
        会被创建为 0700；二者均拒绝符号链接。``python_executable``/``python_path``
        若提供必须解析到当前仓库的标准 ``.venv/bin/python``。任一配置不安全时
        立即抛出异常，不把 capability 或底层路径复制到异常文本。
        """

        selected_python_path = _resolve_lease_python_executable(python_executable, python_path)

        self._task_id = _validate_task_id(task_id)
        self._store_root, self._runtime_root = _prepare_lease_roots(store_root, runtime_root)

        self._host = _validate_loopback_host(host)
        self._port = _validate_port(port)
        self._capability_token = _validate_capability_token(capability_token)
        self._capability_header = _validate_header_name(capability_header)
        (
            self._startup_timeout,
            self._shutdown_timeout,
            self._readiness_request_timeout,
        ) = _validate_lease_timeouts(
            startup_timeout,
            shutdown_timeout,
            readiness_request_timeout,
        )
        self._python_executable = selected_python_path

        self._receipt_path = self._runtime_root / "evidence-mcp-lease.json"
        self._stdout_path = self._runtime_root / "evidence-mcp.stdout.log"
        self._stderr_path = self._runtime_root / "evidence-mcp.stderr.log"
        self._capability_path = self._runtime_root / "capability.token"
        self._active_snapshot_id: str | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._process_identity: dict[str, Any] | None = None
        self._pending_snapshot_id: str | None = None
        self._serve_context_depth = 0
        self._serve_owned_count = 0

        self._configuration_digest = _digest_json(
            {
                "task_id": self._task_id,
                "store_root": str(self._store_root),
                "host": self._host,
                "port": self._port,
                "python": str(self._python_executable.resolve()),
                "capability_header": self._capability_header,
            }
        )

    def __repr__(self) -> str:
        """返回不含 capability 的诊断表示。"""

        return (
            f"TaskEvidenceMcpLease(task_id={self._task_id!r}, host={self._host!r}, "
            f"port={self._port!r}, active_snapshot_id={self._active_snapshot_id!r})"
        )

    @property
    def task_id(self) -> str:
        """返回绑定 task 标识。"""

        return self._task_id

    @property
    def store_root(self) -> Path:
        """返回证据 store 根目录；该路径不写入 receipt。"""

        return self._store_root

    @property
    def runtime_root(self) -> Path:
        """返回私有运行目录。"""

        return self._runtime_root

    @property
    def host(self) -> str:
        """返回已核对的 loopback host。"""

        return self._host

    @property
    def port(self) -> int:
        """返回固定监听端口。"""

        return self._port

    @property
    def endpoint(self) -> str:
        """返回供 Review runtime 连接的 ``/mcp`` endpoint。"""

        host = f"[{self._host}]" if ":" in self._host else self._host
        return f"http://{host}:{self._port}/mcp"

    @property
    def capability_file(self) -> Path:
        """返回 capability 文件路径；不返回 capability 内容。"""

        return self._capability_path

    @property
    def receipt_path(self) -> Path:
        """返回只含进程身份摘要的私有 receipt 路径。"""

        return self._receipt_path

    @property
    def stdout_path(self) -> Path:
        """返回子进程 stdout 日志路径。"""

        return self._stdout_path

    @property
    def stderr_path(self) -> Path:
        """返回子进程 stderr 日志路径。"""

        return self._stderr_path

    @property
    def active_snapshot_id(self) -> str | None:
        """返回当前 readiness 已通过的 snapshot ID。"""

        return self._active_snapshot_id

    @property
    def process_pid(self) -> int | None:
        """返回当前子进程 PID；租约未启动时返回 None。"""

        return int(self._process.pid) if self._process is not None else None

    @property
    def is_running(self) -> bool:
        """通过保存的 process identity 判断服务是否仍可安全识别为存活。"""

        if self._active_snapshot_id is None or self._process_identity is None:
            return False
        if self._identity_activity(self._process_identity) != "alive":
            return False
        try:
            return self._command_matches_snapshot(
                int(self._process_identity["pid"]),
                self._active_snapshot_id,
            )
        except (KeyError, TypeError, ValueError, EvidenceMcpLeaseError):
            return False

    def _build_server_command(self, snapshot_id: str) -> list[str]:
        """构造不含 capability 明文的 evidence MCP 启动命令。

        顶层 ``chemistry_multiagent`` 包会为普通 Controller 自动读取
        ``config.local.env``。子进程使用一个只预置 package path 的短 bootstrap
        绕过该顶层副作用，再加载同一个 ``evidence_mcp_server`` 模块；因此即使
        仓库有 provider 配置文件，最小 allowlist 环境也不会被重新注入 secret。
        """

        return [
            str(self._python_executable),
            "-c",
            self._build_server_bootstrap(),
            "--store-root",
            str(self._store_root),
            "--snapshot-id",
            snapshot_id,
            "--task-id",
            self._task_id,
            "--capability-file",
            str(self._capability_path),
            "--capability-header",
            self._capability_header,
            "--transport",
            "streamable_http",
            "--host",
            self._host,
            "--port",
            str(self._port),
        ]

    def _build_server_bootstrap(self) -> str:
        """返回只加载 evidence 子包的无 secret Python bootstrap。"""

        package_root = _REPO_ROOT / "src" / "chemistry_multiagent"
        investigation_root = package_root / "investigation"
        return (
            "import runpy,sys,types;"
            "package=types.ModuleType('chemistry_multiagent');"
            f"package.__path__=[{str(package_root)!r}];"
            "sys.modules['chemistry_multiagent']=package;"
            "investigation=types.ModuleType('chemistry_multiagent.investigation');"
            f"investigation.__path__=[{str(investigation_root)!r}];"
            "sys.modules['chemistry_multiagent.investigation']=investigation;"
            f"runpy.run_module({_SERVER_MODULE!r},run_name='__main__')"
        )

    def _build_child_environment(self) -> dict[str, str]:
        """构造最小 allowlist 环境，明确不继承 provider/API secret。"""

        source_root = _REPO_ROOT / "src"
        toolbox_root = _REPO_ROOT / "third_party" / "minichem_toolbox" / "src"
        return {
            "PYTHONPATH": os.pathsep.join((str(source_root), str(toolbox_root))),
            "PYTHONUNBUFFERED": "1",
            "LC_ALL": "C.UTF-8",
        }

    def _manifest_check(self, snapshot_id: str) -> None:
        """通过单个公开 manifest 核对 snapshot 存在且属于当前 task。"""

        normalized = _validate_snapshot_id(snapshot_id)
        try:
            manifest = ReviewEvidenceStore(self._store_root).read_public_snapshot_manifest(normalized)
        except (EvidenceRequestError, EvidenceRegistryCorruption, EvidenceSnapshotNotFound, OSError) as exc:
            raise EvidenceMcpLeaseStartupError("snapshot manifest cannot be verified") from exc
        if (
            not isinstance(manifest, Mapping)
            or manifest.get("snapshot_id") != normalized
            or manifest.get("task_id") != self._task_id
            or not isinstance(manifest.get("result_pairs"), Sequence)
            or isinstance(manifest.get("result_pairs"), (str, bytes, bytearray))
            or not 1 <= len(manifest.get("result_pairs")) <= 5
        ):
            raise EvidenceMcpLeaseStartupError("snapshot does not belong to task")

    def _command_matches_snapshot(self, pid: int, snapshot_id: str) -> bool:
        """确认 PID 的完整命令行与本租约服务该 snapshot 的命令一致。"""

        expected = self._build_server_command(snapshot_id)
        actual = _read_process_command_line(pid)
        return _same_command_line(actual, expected)

    def _identity_activity(self, identity: Mapping[str, Any]) -> str:
        """读取既有 process_identity API 的三态结论。"""

        if process_identity_activity is None:
            raise EvidenceMcpLeaseIdentityError("process identity support is unavailable")
        try:
            return str(process_identity_activity(identity))
        except (OSError, TypeError, ValueError) as exc:
            raise EvidenceMcpLeaseIdentityError("process identity cannot be checked") from exc

    def _capture_child_identity(self, pid: int) -> dict[str, Any]:
        """在子进程尚未退出时捕获完整 PID/start/boot identity。"""

        if capture_process_identity is None:
            raise EvidenceMcpLeaseIdentityError("process identity support is unavailable")
        deadline = time.monotonic() + 0.5
        last_error: BaseException | None = None
        while time.monotonic() < deadline:
            try:
                identity = _normalize_identity(capture_process_identity(pid))
                if self._pending_snapshot_id is not None and self._command_matches_snapshot(
                    pid,
                    self._pending_snapshot_id,
                ):
                    return identity
            except BaseException as exc:  # 身份未知必须保守失败
                last_error = exc
            time.sleep(0.01)
        if last_error is not None:
            raise EvidenceMcpLeaseIdentityError("child process identity is unavailable") from last_error
        raise EvidenceMcpLeaseIdentityError("child process identity is unavailable")

    def _identity_is_owned_and_alive(self, identity: Mapping[str, Any], snapshot_id: str) -> bool:
        """同时核对三态身份和完整命令行，未知时绝不发送信号。"""

        activity = self._identity_activity(identity)
        if activity != "alive":
            return False
        try:
            pid = int(identity["pid"])
        except (KeyError, TypeError, ValueError) as exc:
            raise EvidenceMcpLeaseIdentityError("process identity is invalid") from exc
        if not self._command_matches_snapshot(pid, snapshot_id):
            raise EvidenceMcpLeaseIdentityError("process identity does not match lease")
        return True

    def _assert_loopback_port_free(self) -> None:
        """连接探测固定端口；任何成功监听或异常网络状态都 fail-closed。"""

        family = socket.AF_INET6 if ":" in self._host else socket.AF_INET
        address: tuple[Any, ...]
        if family == socket.AF_INET6:
            address = (self._host, self._port, 0, 0)
        else:
            address = (self._host, self._port)
        client = socket.socket(family, socket.SOCK_STREAM)
        client.settimeout(0.15)
        try:
            try:
                client.connect(address)
            except ConnectionRefusedError:
                return
            except OSError as exc:
                if getattr(exc, "errno", None) in {111, 61, 10061}:  # ECONNREFUSED 各平台值
                    return
                raise EvidenceMcpLeasePortError("loopback port cannot be safely probed") from exc
            raise EvidenceMcpLeasePortError("loopback port is occupied")
        finally:
            client.close()

    def _ensure_capability_file(self) -> None:
        """创建或核对 0600 capability 文件；token 只在此进程内和文件中出现。"""

        if self._capability_path.exists():
            try:
                existing = load_evidence_capability_file(self._capability_path)
            except EvidenceMcpConfigurationError as exc:
                raise EvidenceMcpLeaseConfigurationError("capability file is invalid") from exc
            if not hmac.compare_digest(existing, self._capability_token):
                raise EvidenceMcpLeaseConfigurationError("capability file does not match lease")
            return
        try:
            create_evidence_capability_file(self._capability_path, token=self._capability_token)
        except EvidenceMcpConfigurationError as exc:
            raise EvidenceMcpLeaseConfigurationError("capability file cannot be created") from exc

    def _receipt_payload(self, snapshot_id: str, identity: Mapping[str, Any]) -> dict[str, Any]:
        """构造不含明文 token、路径或 argv 的身份 receipt。"""

        return {
            "schema_version": 1,
            "task_id": self._task_id,
            "snapshot_id": snapshot_id,
            "host": self._host,
            "port": self._port,
            "configuration_digest": self._configuration_digest,
            "command_digest": _digest_json(self._build_server_command(snapshot_id)),
            "identity": dict(identity),
        }

    def _validate_receipt(self, receipt: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
        """验证 receipt 的任务、端口、配置摘要和可回收身份。"""

        if (
            receipt.get("schema_version") != 1
            or receipt.get("task_id") != self._task_id
            or receipt.get("host") != self._host
            or receipt.get("port") != self._port
            or receipt.get("configuration_digest") != self._configuration_digest
        ):
            raise EvidenceMcpLeaseIdentityError("lease receipt configuration mismatch")
        snapshot_id = _validate_snapshot_id(receipt.get("snapshot_id"))
        if receipt.get("command_digest") != _digest_json(self._build_server_command(snapshot_id)):
            raise EvidenceMcpLeaseIdentityError("lease receipt command mismatch")
        raw_identity = receipt.get("identity")
        if not isinstance(raw_identity, Mapping):
            raise EvidenceMcpLeaseIdentityError("lease receipt identity is invalid")
        identity = _normalize_identity(raw_identity)
        return snapshot_id, identity

    def _remove_private_capability(self) -> None:
        """删除租约自己的 capability 文件；不触碰来源 artifact。"""

        try:
            if self._capability_path.exists():
                _safe_file_mode(self._capability_path, label="capability file")
                self._capability_path.unlink()
        except FileNotFoundError:
            pass
        except EvidenceMcpLeaseConfigurationError:
            # 权限/类型异常时保持文件，避免误删攻击者替换的对象。
            pass
        except OSError:
            pass

    def _remove_receipt(self) -> None:
        """删除已经确认结束的本租约 receipt。"""

        try:
            if self._receipt_path.exists():
                _safe_file_mode(self._receipt_path, label="lease receipt")
                self._receipt_path.unlink()
        except FileNotFoundError:
            pass
        except (EvidenceMcpLeaseConfigurationError, OSError):
            pass

    def _wait_for_identity_stop(self, identity: Mapping[str, Any], deadline: float) -> bool:
        """等待同一身份消失；PID 复用或未知状态均不当作可回收。"""

        while time.monotonic() < deadline:
            activity = self._identity_activity(identity)
            if activity == "stopped":
                return True
            if activity == "unknown":
                return False
            time.sleep(0.02)
        return self._identity_activity(identity) == "stopped"

    def _wait_for_direct_child_exit(
        self,
        process: subprocess.Popen[bytes] | None,
        identity: Mapping[str, Any],
    ) -> bool:
        """在命令行短暂消失时等待直接子句柄确认进程确已退出。"""

        if process is None:
            return False
        if process.poll() is not None:
            with contextlib.suppress(Exception):
                process.wait(timeout=0)
            return True
        deadline = time.monotonic() + min(self._shutdown_timeout, 0.5)
        while time.monotonic() < deadline:
            if process.poll() is not None:
                with contextlib.suppress(Exception):
                    process.wait(timeout=0)
                return True
            if self._identity_activity(identity) == "stopped":
                return True
            time.sleep(0.01)
        return False

    def _stop_verified_process_identity(
        self,
        identity: Mapping[str, Any],
        snapshot_id: str,
        process: subprocess.Popen[bytes] | None,
    ) -> None:
        """向已核对身份发送 TERM/KILL 并等待同一身份退出。"""

        pid = int(identity["pid"])
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except OSError as exc:
            raise EvidenceMcpLeaseError("evidence MCP process could not be stopped") from exc
        deadline = time.monotonic() + self._shutdown_timeout
        if not self._wait_for_identity_stop(identity, deadline):
            if not self._identity_is_owned_and_alive(identity, snapshot_id):
                raise EvidenceMcpLeaseIdentityError("process identity changed during shutdown")
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError as exc:
                raise EvidenceMcpLeaseError("evidence MCP process could not be killed") from exc
            if not self._wait_for_identity_stop(identity, time.monotonic() + self._shutdown_timeout):
                raise EvidenceMcpLeaseError("evidence MCP process did not stop")
        if process is not None:
            with contextlib.suppress(Exception):
                process.wait(timeout=0.2)

    def _terminate_owned_identity(
        self,
        identity: Mapping[str, Any],
        snapshot_id: str,
        process: subprocess.Popen[bytes] | None,
    ) -> None:
        """只向已核对身份发送 TERM/KILL，并等待退出。"""

        if _reap_direct_child_process(process):
            return
        if process is not None:
            # SIGKILL 后 /proc/cmdline 可能先短暂消失而 stat 尚未变成 stopped；
            # 只等待直接子句柄或同一 identity 的 stopped，不把命令不匹配当作可杀信号。
            if self._identity_activity(identity) == "alive" and not self._command_matches_snapshot(
                int(identity.get("pid") or 0),
                snapshot_id,
            ):
                if self._wait_for_direct_child_exit(process, identity):
                    return
        if not self._identity_is_owned_and_alive(identity, snapshot_id):
            activity = self._identity_activity(identity)
            if activity == "stopped":
                _reap_direct_child_process(process)
                return
            raise EvidenceMcpLeaseIdentityError("process identity cannot be safely stopped")
        self._stop_verified_process_identity(identity, snapshot_id, process)

    def _recover_prior_receipt(self) -> None:
        """安全回收同一配置遗留进程；伪造或未知身份直接 fail-closed。"""

        receipt = _read_private_json(self._receipt_path)
        if receipt is None:
            return
        snapshot_id, identity = self._validate_receipt(receipt)
        activity = self._identity_activity(identity)
        if activity == "alive":
            if not self._command_matches_snapshot(int(identity["pid"]), snapshot_id):
                raise EvidenceMcpLeaseIdentityError("lease receipt process command mismatch")
            self._terminate_owned_identity(identity, snapshot_id, None)
        elif activity == "unknown":
            raise EvidenceMcpLeaseIdentityError("lease receipt process identity is unknown")
        self._remove_receipt()
        self._remove_private_capability()

    def recover_evidence_mcp_lease(self) -> None:
        """恢复入口专用地回收固定 runtime_root 的遗留进程和私有文件。

        调用方已知同一个 task/runtime 配置但本进程没有活动句柄时使用本方法。它
        先按 receipt 的 PID、启动时间、boot id 和命令行核对遗留子进程，再删除本
        租约的 capability/receipt；无法证明归属时保留文件并抛出异常。
        """

        if self._active_snapshot_id is not None and self._process_identity is not None:
            self.stop_evidence_mcp()
            return
        self._recover_prior_receipt()
        self._remove_private_capability()

    async def _perform_mcp_readiness_async(self) -> None:
        """使用真实 MCP initialize/list_tools 验证 capability 和唯一工具。

        此探针不调用工具，也不读取任何 artifact 正文；它只确认 HTTP endpoint 的
        身份认证、协议协商和工具目录恰好包含 ``read_snapshot_artifact``。
        """

        try:
            import httpx
            from mcp import ClientSession
            from mcp.client.streamable_http import streamable_http_client
        except ImportError as exc:  # pragma: no cover - 标准环境提供依赖
            raise EvidenceMcpLeaseStartupError("MCP readiness dependencies are unavailable") from exc

        timeout = httpx.Timeout(self._readiness_request_timeout)
        headers = {self._capability_header: self._capability_token}
        async with httpx.AsyncClient(headers=headers, timeout=timeout, follow_redirects=False) as client:
            async with streamable_http_client(self.endpoint, http_client=client) as streams:
                read_stream, write_stream, _ = streams
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    result = await session.list_tools()
                    names = tuple(tool.name for tool in result.tools)
                    if names != ("read_snapshot_artifact",):
                        raise EvidenceMcpLeaseStartupError("evidence MCP tool surface is not unique")

    def _perform_mcp_readiness(self) -> None:
        """在同步租约 API 中运行真实 MCP readiness。"""

        try:
            asyncio.run(self._perform_mcp_readiness_async())
        except EvidenceMcpLeaseStartupError:
            raise
        except BaseException as exc:
            raise EvidenceMcpLeaseStartupError("evidence MCP readiness failed") from exc

    def _wait_for_readiness(self, snapshot_id: str) -> None:
        """在截止时间前反复探测服务，失败时不暴露半开租约。"""

        deadline = time.monotonic() + self._startup_timeout
        last_error: BaseException | None = None
        while time.monotonic() < deadline:
            identity = self._process_identity
            if identity is None:
                raise EvidenceMcpLeaseStartupError("child process identity is missing")
            activity = self._identity_activity(identity)
            if activity == "stopped":
                raise EvidenceMcpLeaseStartupError("evidence MCP process exited during startup")
            if activity == "unknown":
                raise EvidenceMcpLeaseIdentityError("child process identity became unknown")
            try:
                self._perform_mcp_readiness()
                return
            except BaseException as exc:
                last_error = exc
                time.sleep(min(0.08, max(0.0, deadline - time.monotonic())))
        if last_error is not None:
            raise EvidenceMcpLeaseStartupError("evidence MCP readiness timed out") from last_error
        raise EvidenceMcpLeaseStartupError("evidence MCP readiness timed out")

    def _cleanup_failed_start(self) -> None:
        """启动失败后回收已核对进程或直接子句柄，并清理 capability。

        身份捕获本身失败时没有足够证据向任意 PID 发信号，但 ``Popen`` 仍然持有
        内核直接子进程句柄；只对这个句柄调用 terminate/kill 并等待，既不会误杀
        PID 复用者，也不会把可能监听端口的启动失败子进程遗留给下一次租约。
        """

        if self._process_identity is None and self._process is not None:
            process = self._process
            try:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=self._shutdown_timeout)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=self._shutdown_timeout)
                else:
                    process.wait(timeout=0)
            except (OSError, subprocess.TimeoutExpired):
                # 句柄异常时不伪造“已结束”；保留 receipt（若已写入）供恢复核对。
                return
        elif self._process_identity is not None and self._active_snapshot_id is not None:
            try:
                self._terminate_owned_identity(
                    self._process_identity,
                    self._active_snapshot_id,
                    self._process,
                )
            except EvidenceMcpLeaseIdentityError:
                # 无法证明归属时保留 receipt，后续恢复仍然 fail-closed。
                return
            except EvidenceMcpLeaseError:
                # 信号或等待失败时不能删除身份凭据，否则下一次会把未知监听者
                # 当作空闲端口；保留 receipt 让显式 stop/recovery 再次核对。
                return
        self._remove_receipt()
        self._remove_private_capability()
        self._process = None
        self._process_identity = None
        self._active_snapshot_id = None

    def _reuse_active_snapshot_if_ready(self, snapshot_id: str) -> bool:
        """复用同一 snapshot 的活动服务，或安全停止旧服务后返回 False。"""

        if self._active_snapshot_id is None:
            return False
        if self._process_identity is None:
            raise EvidenceMcpLeaseIdentityError("active lease identity is missing")
        activity = self._identity_activity(self._process_identity)
        if activity == "alive" and self._active_snapshot_id == snapshot_id:
            if not self._command_matches_snapshot(int(self._process_identity["pid"]), snapshot_id):
                if self._wait_for_direct_child_exit(self._process, self._process_identity):
                    self.stop_evidence_mcp()
                else:
                    raise EvidenceMcpLeaseIdentityError("active process command mismatch")
            else:
                try:
                    # start 的幂等保证同时确认 endpoint 仍然认证且工具表未被替换。
                    self._perform_mcp_readiness()
                except BaseException:
                    # 服务已失去 MCP 响应时，不把旧的半开进程继续交给调用方。
                    self.stop_evidence_mcp()
                    raise
                return True
        if activity == "unknown":
            raise EvidenceMcpLeaseIdentityError("active process identity is unknown")
        self.stop_evidence_mcp()
        return False

    def _launch_snapshot_process(self, snapshot_id: str) -> None:
        """准备私有启动文件、创建子进程并保存其身份 receipt。"""

        self._recover_prior_receipt()
        self._assert_loopback_port_free()
        self._ensure_capability_file()
        self._pending_snapshot_id = snapshot_id
        command = self._build_server_command(snapshot_id)
        child_environment = self._build_child_environment()
        stdout_descriptor = -1
        stderr_descriptor = -1
        try:
            stdout_descriptor = _open_private_log(self._stdout_path)
            stderr_descriptor = _open_private_log(self._stderr_path)
        except BaseException:
            if stdout_descriptor >= 0:
                os.close(stdout_descriptor)
            self._remove_private_capability()
            raise
        child: subprocess.Popen[bytes] | None = None
        try:
            popen_options: dict[str, Any] = {
                "stdin": subprocess.DEVNULL,
                "stdout": stdout_descriptor,
                "stderr": stderr_descriptor,
                "env": child_environment,
                "close_fds": True,
                "start_new_session": True,
                "shell": False,
            }
            if sys.platform == "linux":
                popen_options["preexec_fn"] = partial(
                    _configure_child_parent_death_signal,
                    os.getpid(),
                )
            child = subprocess.Popen(command, **popen_options)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            self._remove_private_capability()
            raise EvidenceMcpLeaseStartupError("evidence MCP process could not start") from exc
        finally:
            if stdout_descriptor >= 0:
                os.close(stdout_descriptor)
            if stderr_descriptor >= 0:
                os.close(stderr_descriptor)

        self._process = child
        self._process_identity = self._capture_child_identity(child.pid)
        self._active_snapshot_id = snapshot_id
        _write_private_json(
            self._receipt_path,
            self._receipt_payload(snapshot_id, self._process_identity),
        )

    def start_evidence_snapshot(self, snapshot_id: str) -> "TaskEvidenceMcpLease":
        """验证并启动一个 snapshot；同 snapshot 重入幂等，不同 snapshot 先停旧服务。

        输入必须是属于构造时 task 的现有一至五项快照。动作创建子进程和私有审计文件，
        readiness 失败或身份不可核对时抛出异常，绝不返回可连接但未认证的半开租约。
        返回当前租约自身，供调用方读取 endpoint。
        """

        normalized_snapshot_id = _validate_snapshot_id(snapshot_id)
        self._manifest_check(normalized_snapshot_id)
        if self._reuse_active_snapshot_if_ready(normalized_snapshot_id):
            return self
        try:
            self._launch_snapshot_process(normalized_snapshot_id)
            self._wait_for_readiness(normalized_snapshot_id)
            return self
        except BaseException:
            self._cleanup_failed_start()
            raise

    def stop_evidence_mcp(self) -> None:
        """停止并清理当前租约；无活动租约时幂等返回。

        只有 process_identity 明确返回同一进程且命令行完全匹配时才发送信号。
        对伪造、陈旧但无法判定或 PID 复用身份抛出异常并保留 receipt，避免误杀。
        """

        if self._active_snapshot_id is None or self._process_identity is None:
            return
        snapshot_id = self._active_snapshot_id
        identity = self._process_identity
        process = self._process
        self._terminate_owned_identity(identity, snapshot_id, process)
        self._remove_receipt()
        self._remove_private_capability()
        self._process = None
        self._process_identity = None
        self._active_snapshot_id = None

    @contextmanager
    def serve_evidence_snapshot(self, snapshot_id: str) -> Iterator["TaskEvidenceMcpLease"]:
        """以 context manager 形式持有当前 snapshot 服务并在退出时回收。

        同一实例嵌套进入同一 snapshot 时共享一个进程，最外层退出后才停止；
        不同 snapshot 会先停止旧服务再启动新服务。异常退出也执行同样的身份核验。
        """

        normalized_snapshot_id = _validate_snapshot_id(snapshot_id)
        previous_snapshot_id = self._active_snapshot_id
        was_running = self.is_running
        self.start_evidence_snapshot(normalized_snapshot_id)
        owns_started_process = (
            previous_snapshot_id is None
            or previous_snapshot_id != normalized_snapshot_id
            or not was_running
        )
        self._serve_context_depth += 1
        if owns_started_process:
            self._serve_owned_count += 1
        try:
            yield self
        except BaseException:
            raise
        finally:
            self._serve_context_depth = max(0, self._serve_context_depth - 1)
            if self._serve_context_depth == 0 and self._serve_owned_count > 0:
                self._serve_owned_count = 0
                self.stop_evidence_mcp()

    def __enter__(self) -> "TaskEvidenceMcpLease":
        """拒绝没有 snapshot 的裸 context，避免意外启动宽泛服务。"""

        raise EvidenceMcpLeaseConfigurationError("use serve_evidence_snapshot(snapshot_id) context")

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        """裸 context 不持有资源；始终让调用方显式选择 snapshot。"""

        return False
