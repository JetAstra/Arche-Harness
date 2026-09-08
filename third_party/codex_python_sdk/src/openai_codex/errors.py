from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class CodexError(Exception):
    """Base exception for SDK errors."""


class GoalNotFoundError(CodexError):
    """服务端明确报告线程没有持久化 Goal。"""


class JsonRpcError(CodexError):
    """Raw JSON-RPC error wrapper from the server."""

    def __init__(self, code: int, message: str, data: Any = None):
        super().__init__(f"JSON-RPC error {code}: {message}")
        self.code = code
        self.message = message
        self.data = data


class TransportPhase(str, Enum):
    """传输诊断允许暴露的稳定阶段名称。"""

    UNKNOWN = "unknown"
    PROCESS_START = "process_start"
    INITIALIZE = "initialize"
    THREAD_START = "thread_start"
    THREAD_RESUME = "thread_resume"
    THREAD_READ = "thread_read"
    TURN_START = "turn_start"
    REQUEST = "request"
    WRITE = "write"
    STDOUT_READ = "stdout_read"
    PROCESS_NOT_RUNNING = "process_not_running"


class StderrIndicator(str, Enum):
    """stderr 仅允许映射出的有限故障类别，不保存原始文本。"""

    NONE = "none"
    SANDBOX = "sandbox"
    PERMISSION = "permission"
    MISSING_DEPENDENCY = "missing_dependency"
    CONFIGURATION = "configuration"
    AUTHENTICATION = "authentication"
    NETWORK = "network"
    RUNTIME = "runtime"


@dataclass(frozen=True, slots=True)
class TransportDiagnostic:
    """保存 transport 关闭时可安全交给调用方的有限诊断字段。

    所有字段都是固定枚举、布尔值或整数；这里不放 stderr、argv、环境、URL
    或 provider 响应。退出码为负数时由 SDK 转换为 ``termination_signal``，
    调用方无需读取平台相关的原始进程对象。
    """

    phase: TransportPhase = TransportPhase.UNKNOWN
    stdout_eof: bool = False
    returncode: int | None = None
    termination_signal: int | None = None
    stderr_indicator: StderrIndicator = StderrIndicator.NONE

    def __post_init__(self) -> None:
        """把构造参数收敛为可安全公开的固定标量。

        输入是 dataclass 已接收的枚举、布尔和整数候选；动作原地规范化冻结字段，
        无 I/O 副作用。非法阶段、类别或整数会降为安全默认值，不向调用方抛出
        校验错误。
        """

        try:
            phase = TransportPhase(self.phase)
        except (TypeError, ValueError):
            phase = TransportPhase.UNKNOWN
        try:
            indicator = StderrIndicator(self.stderr_indicator)
        except (TypeError, ValueError):
            indicator = StderrIndicator.NONE
        returncode = (
            self.returncode
            if isinstance(self.returncode, int) and not isinstance(self.returncode, bool)
            else None
        )
        termination_signal = (
            self.termination_signal
            if isinstance(self.termination_signal, int)
            and not isinstance(self.termination_signal, bool)
            and self.termination_signal > 0
            else None
        )
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "stderr_indicator", indicator)
        object.__setattr__(self, "stdout_eof", self.stdout_eof is True)
        object.__setattr__(self, "returncode", returncode)
        object.__setattr__(self, "termination_signal", termination_signal)

    def as_dict(self) -> dict[str, object]:
        """返回只含稳定标量的诊断副本，供上层错误对象安全携带。

        本方法无输入和外部副作用；返回新的五字段字典，不包含异常正文、stderr、
        路径或环境。对象已在构造时规范化，因此本方法不会因字段值抛出错误。
        """

        return {
            "phase": self.phase.value,
            "stdout_eof": self.stdout_eof,
            "returncode": self.returncode,
            "termination_signal": self.termination_signal,
            "stderr_indicator": self.stderr_indicator.value,
        }


class TransportClosedError(CodexError):
    """Codex transport 意外关闭时抛出的异常。"""

    _SAFE_MESSAGES = frozenset(
        {
            "Codex transport closed",
            "Codex process is not running",
            "Codex process closed stdout",
        }
    )

    def __init__(
        self,
        message: str = "Codex transport closed",
        *,
        diagnostic: TransportDiagnostic | None = None,
    ) -> None:
        # 保留 message 参数以兼容已有调用方式，但只接受 SDK 自己使用的固定文本；
        # 任意调用方正文都会收敛为通用消息，不能借异常 args 保存 stderr 或 secret。
        safe_message = (
            message if message in self._SAFE_MESSAGES else "Codex transport closed"
        )
        super().__init__(safe_message)
        self.diagnostic = diagnostic


class CodexRpcError(JsonRpcError):
    """Base typed error for JSON-RPC failures."""


class ParseError(CodexRpcError):
    """Raised when a request or response cannot be parsed."""


class InvalidRequestError(CodexRpcError):
    """Raised when the runtime rejects the request shape."""


class MethodNotFoundError(CodexRpcError):
    """Raised when the requested operation is unavailable."""


class InvalidParamsError(CodexRpcError):
    """Raised when an operation receives invalid parameters."""


class InternalRpcError(CodexRpcError):
    """Raised when the runtime reports an internal RPC failure."""


class ServerBusyError(CodexRpcError):
    """Server is overloaded / unavailable and caller should retry."""


class RetryLimitExceededError(ServerBusyError):
    """Server exhausted internal retry budget for a retryable operation."""


def _contains_retry_limit_text(message: str) -> bool:
    lowered = message.lower()
    return "retry limit" in lowered or "too many failed attempts" in lowered


def _is_server_overloaded(data: Any) -> bool:
    if data is None:
        return False

    if isinstance(data, str):
        return data.lower() == "server_overloaded"

    if isinstance(data, dict):
        direct = data.get("codex_error_info") or data.get("codexErrorInfo") or data.get("errorInfo")
        if isinstance(direct, str) and direct.lower() == "server_overloaded":
            return True
        if isinstance(direct, dict):
            for value in direct.values():
                if isinstance(value, str) and value.lower() == "server_overloaded":
                    return True
        for value in data.values():
            if _is_server_overloaded(value):
                return True

    if isinstance(data, list):
        return any(_is_server_overloaded(value) for value in data)

    return False


def map_jsonrpc_error(code: int, message: str, data: Any = None) -> JsonRpcError:
    """Map a raw JSON-RPC error into a richer SDK exception class."""

    if code == -32700:
        return ParseError(code, message, data)
    if code == -32600:
        return InvalidRequestError(code, message, data)
    if code == -32601:
        return MethodNotFoundError(code, message, data)
    if code == -32602:
        return InvalidParamsError(code, message, data)
    if code == -32603:
        return InternalRpcError(code, message, data)

    if -32099 <= code <= -32000:
        if _is_server_overloaded(data):
            if _contains_retry_limit_text(message):
                return RetryLimitExceededError(code, message, data)
            return ServerBusyError(code, message, data)
        if _contains_retry_limit_text(message):
            return RetryLimitExceededError(code, message, data)
        return CodexRpcError(code, message, data)

    return JsonRpcError(code, message, data)


def is_retryable_error(exc: BaseException) -> bool:
    """True if the exception is a transient overload-style error."""

    if isinstance(exc, ServerBusyError):
        return True

    if isinstance(exc, JsonRpcError):
        return _is_server_overloaded(exc.data)

    return False
