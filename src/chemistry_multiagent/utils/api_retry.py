"""外部 API 调用的统一瞬时失败重试策略。

该模块只处理传输层和服务端明确可恢复的失败。调用方仍然负责本地参数、协议
和业务内容校验；这些确定性错误不会被本策略重新发送。``retry_count`` 表示
首次调用失败后允许增加的调用次数。通用外部 API 保留原有默认值；模型调用
统一使用六次额外重试、固定等待十秒的独立策略。
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, TypeVar


T = TypeVar("T")

DEFAULT_API_RETRY_COUNT = 5
DEFAULT_API_RETRY_DELAY_SECONDS = 6.0
MODEL_CALL_RETRY_COUNT = 6
MODEL_CALL_RETRY_DELAY_SECONDS = 10.0
_SUPPRESS_NESTED_RETRIES: ContextVar[bool] = ContextVar(
    "suppress_nested_api_retries", default=False
)


@contextmanager
def suppress_nested_api_retries():
    """在一个统一外层 API 边界内关闭适配器的重复重试。"""

    token = _SUPPRESS_NESTED_RETRIES.set(True)
    try:
        yield
    finally:
        _SUPPRESS_NESTED_RETRIES.reset(token)


def is_retryable_http_status(status: Any) -> bool:
    """判断 HTTP 状态是否属于限流或服务端瞬时失败。"""

    if isinstance(status, bool):
        return False
    try:
        numeric_status = int(status)
    except (TypeError, ValueError):
        return False
    return numeric_status == 429 or 500 <= numeric_status <= 599


def response_http_status(value: Any) -> int | None:
    """从常见 HTTP 响应、mapping 或 OpenAI 异常响应中读取状态码。"""

    if isinstance(value, Mapping):
        candidate = value.get("status_code", value.get("status"))
    else:
        candidate = getattr(value, "status_code", None)
        if candidate is None:
            candidate = getattr(value, "status", None)
        if candidate is None:
            response = getattr(value, "response", None)
            candidate = getattr(response, "status_code", None)
            if candidate is None:
                candidate = getattr(response, "status", None)
    if isinstance(candidate, bool):
        return None
    try:
        return int(candidate) if candidate is not None else None
    except (TypeError, ValueError):
        return None


def is_retryable_api_exception(error: BaseException) -> bool:
    """判断异常是否为连接、超时、429 或 5xx 瞬时失败。

    认证、权限、参数、协议和其他 4xx 错误默认立即向上抛出。可选依赖的异常
    类型通过模块名和公开属性识别，使离线测试替身不需要安装对应 SDK。
    """

    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        status = response_http_status(current)
        if status is not None:
            return is_retryable_http_status(status)
        if isinstance(current, (TimeoutError, ConnectionError)):
            return True

        module_name = type(current).__module__.casefold()
        class_name = type(current).__name__.casefold()
        if module_name.startswith("requests"):
            # requests 还有 InvalidURL、InvalidSchema 等本地参数错误；不能把
            # 它们和 ConnectionError/Timeout 一起重发。
            if class_name in {
                "connectionerror",
                "proxyerror",
                "timeout",
                "connecttimeout",
                "readtimeout",
                "chunkedencodingerror",
            }:
                return True
        elif module_name.startswith("httpx"):
            if (
                "timeout" in class_name
                or "connect" in class_name
                or "network" in class_name
                or class_name
                in {
                    "readerror",
                    "writeerror",
                    "closeerror",
                    "remoteprotocolerror",
                }
            ):
                return True
        elif module_name.startswith("aiohttp"):
            if class_name in {
                "clientconnectionerror",
                "clientconnectorerror",
                "clientoserror",
                "serverdisconnectederror",
                "servertimeouterror",
                "clientpayloaderror",
            }:
                return True
        elif module_name.startswith("openai"):
            if class_name in {
                "apiconnectionerror",
                "apitimeouterror",
                "ratelimiterror",
                "internalservererror",
            }:
                return True
        elif class_name in {
            "connectionerror",
            "connecterror",
            "connectionclosederror",
            "timeout",
            "timeouterror",
            "readtimeout",
            "connecttimeout",
            "networkerror",
        }:
            return True
        current = current.__cause__ or current.__context__
    return False


def retry_api_call(
    operation: Callable[[], T],
    *,
    retry_count: int = DEFAULT_API_RETRY_COUNT,
    retry_delay: float = DEFAULT_API_RETRY_DELAY_SECONDS,
    sleeper: Callable[[float], None] | None = None,
    retryable_result: Callable[[T], bool] | None = None,
    is_retryable_exception: Callable[[BaseException], bool] = is_retryable_api_exception,
    on_attempt: Callable[[int, T | None, BaseException | None], None] | None = None,
) -> T:
    """执行一次 API 操作，并只重试已登记的瞬时失败。

    ``retry_count`` 是重试次数而非总尝试次数；``retry_delay`` 固定用于每次重试。
    返回 HTTP 429/5xx 响应时，最后一次响应会原样返回给调用方，使调用方能够保留
    原有的结构化限流或错误处理。异常类失败在达到上限后重新抛出最后一个原异常。
    ``sleeper`` 和 ``on_attempt`` 供测试及审计注入，不改变生产默认行为。
    """

    if isinstance(retry_count, bool) or not isinstance(retry_count, int) or retry_count < 0:
        raise ValueError("retry_count 必须是非负整数，表示额外重试次数")
    if isinstance(retry_delay, bool) or not isinstance(retry_delay, (int, float)):
        raise ValueError("retry_delay 必须是非负数字")
    if not math.isfinite(float(retry_delay)) or retry_delay < 0:
        raise ValueError("retry_delay 必须是非负有限数字")
    if not callable(operation):
        raise TypeError("operation 和 sleeper 必须可调用")
    effective_sleeper = sleeper or time.sleep
    if not callable(effective_sleeper):
        raise TypeError("operation 和 sleeper 必须可调用")
    effective_retry_count = 0 if _SUPPRESS_NESTED_RETRIES.get() else retry_count

    for attempt_index in range(effective_retry_count + 1):
        result: T | None = None
        error: BaseException | None = None
        try:
            result = operation()
            result_is_retryable = (
                retryable_result is not None and retryable_result(result)
            )
        except Exception as exc:
            error = exc
            result_is_retryable = is_retryable_exception(exc)

        if on_attempt is not None:
            on_attempt(attempt_index + 1, result, error)

        if not result_is_retryable:
            if error is not None:
                raise error
            return result  # type: ignore[return-value]
        if attempt_index >= effective_retry_count:
            if error is not None:
                raise error
            return result  # type: ignore[return-value]
        effective_sleeper(float(retry_delay))

    raise AssertionError("API retry loop must return")


__all__ = [
    "DEFAULT_API_RETRY_COUNT",
    "DEFAULT_API_RETRY_DELAY_SECONDS",
    "MODEL_CALL_RETRY_COUNT",
    "MODEL_CALL_RETRY_DELAY_SECONDS",
    "is_retryable_api_exception",
    "is_retryable_http_status",
    "response_http_status",
    "retry_api_call",
    "suppress_nested_api_retries",
]
