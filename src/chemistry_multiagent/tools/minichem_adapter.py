"""MiniChem Toolbox MCP 的唯一生产适配层。

本模块只负责 MCP 会话、目录快照、Action 请求和作业协议的传输与归一化。
它不复制 Action 定义，也不负责规划、重试科学请求或决定 Program Gate；上层
工作流仍然负责这些边界。默认端点是本机回环地址，测试可以注入一个实现相同
调用接口的确定性会话工厂。
"""

from __future__ import annotations

import asyncio
import errno
import fcntl
import hashlib
import inspect
import json
import math
import os
import re
import stat
import threading
import time
from concurrent.futures import Future as ConcurrentFuture
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Protocol

from chemistry_multiagent.utils.api_retry import (
    DEFAULT_API_RETRY_COUNT,
    DEFAULT_API_RETRY_DELAY_SECONDS,
    is_retryable_api_exception,
    is_retryable_http_status,
    response_http_status,
    retry_api_call,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_OPERATION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_JOB_ID = re.compile(r"^job_(?:[0-9a-f]{8}|[0-9a-f]{32})$")
_SUBMISSION_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$")
_CATALOG_FREEZE_DRIFT_MARKERS = (
    "服务冻结目录文件已被删除或替换",
    "服务冻结目录文件缺失或 hash 已变化",
    "服务冻结目录标记与当前 workspace 不一致",
)
_WAITING_STATUSES = frozenset({"waiting", "queued", "running", "submitted", "pending"})
_ACTION_SUCCESS_STATUSES = frozenset({"success", "partial_success"})
_JOB_TERMINAL_SUCCESS_STATUSES = frozenset({"success", "completed", "ready"})
_JOB_TERMINAL_FAILURE_STATUSES = frozenset({"failed", "timeout", "cancelled", "canceled"})
_CATALOG_ID = re.compile(r"^[a-z][a-z0-9_]*$")
_RESOURCE_LIMIT_FIELDS = frozenset({"cpu_cores", "memory_mb", "gpu_count"})
_RESOURCE_POLICY_FIELDS = frozenset(
    {
        "cpu_cores",
        "memory_mb",
        "gpu_count",
        "source",
        "agent_controllable",
        "scope",
    }
)
_PUBLIC_RESOURCE_POLICY_FIELDS = frozenset(
    {"source", "agent_controllable", "scope"}
)
_PUBLIC_TIMEOUT_POLICY_FIELDS = frozenset(
    {"execution_class", "source", "agent_controllable"}
)
_LEGACY_TIMEOUT_POLICY_FIELDS = _PUBLIC_TIMEOUT_POLICY_FIELDS | frozenset(
    {"timeout_seconds"}
)
_RESOURCE_POLICY_FILE = Path(__file__).with_name("minichem_resource_policy.json")
_ACTION_REQUEST_FIELDS = frozenset(
    {
        "action_id",
        "backend_id",
        "component_backends",
        "source_id",
        "inputs",
        "method_spec",
        "action_settings",
        "resource_limits",
    }
)
_ARTIFACT_DECLARATION_FIELDS = frozenset(
    {
        "path",
        "semantic_type",
        "media_type",
        "producer_layer",
        "producer_id",
        "parent_artifact_ids",
    }
)
_JOB_SUBMIT_TOOLS = frozenset({"submit_native_job", "submit_analysis_program"})
_SAFE_FAILURE_DIAGNOSTIC_DETAIL_FIELDS = frozenset(
    {
        "missing_key",
        "rejected_route_field",
        "rejected_value",
        "recommended_value",
    }
)
_CALLER_REQUEST_FINGERPRINT_FIELD = "caller_request_fingerprint"
_AMBIGUOUS_NATIVE_RECEIPT_KIND = "ambiguous_native_submission"
_NATIVE_JOB_EXAMPLE_FIELDS = frozenset(
    {
        "software_id",
        "executable",
        "arguments",
        "staged_inputs",
        "stdin_target",
        "resource_limits",
        "resource_policy",
        "calculation_intent",
        "label",
        "parent_job_id",
    }
)
_NATIVE_JOB_REQUEST_FIELDS = (
    _NATIVE_JOB_EXAMPLE_FIELDS - frozenset({"resource_policy"})
) | frozenset({"operation_id", "submission_key"})
_ANALYSIS_JOB_REQUEST_FIELDS = frozenset(
    {
        "runtime",
        "script_path",
        "script_target",
        "arguments",
        "staged_inputs",
        "required_modules",
        "required_module_versions",
        "required_symbols",
        "inputs",
        "outputs",
        "execution_policy",
        "external_execution",
        "resource_limits",
        "label",
        "parent_job_id",
        "operation_id",
        "submission_key",
    }
)
_OPEN_OUTPUT_PREFIXES = ("outputs/", "report/", "logs/")
_SAFE_RUNTIME_ID = re.compile(r"^[a-z][a-z0-9_]*$")
_SAFE_EXECUTABLE = re.compile(r"^[A-Za-z0-9_.+-]+$")
_SAFE_ARGUMENT_PATH = re.compile(r"^(?:/|~|[A-Za-z]:[\\/])")
_SAFE_MODULE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")
_CATALOG_FIELDS = frozenset(
    {
        "schema_version",
        "execution_layers",
        "discovery_mode",
        "exposure_policy",
        "catalog_visibility_policy",
        "task_specific_tool_filtering",
        "backend_selection_policy",
        "provider_selection_policies",
        "scientific_resource_selection_policy",
        "automatic_fallback",
        "evaluation_resource_budget",
        "actions",
        "backends",
        "resources",
        "catalog_hash",
    }
)
_PUBLIC_CATALOG_FIELDS = _CATALOG_FIELDS | frozenset(
    {
        "catalog_projection_schema_version",
        "catalog_projection_hash",
    }
)
_PUBLIC_ACTION_FIELDS = frozenset(
    {
        "id",
        "version",
        "category",
        "description",
        "input_description",
        "primary_output",
        "backend_ids",
        "required_inputs",
        "optional_inputs",
        "selection_policy",
        "input_semantic_types",
        "output_semantic_types",
        "execution_class",
        "requires_network",
        "capability_tags",
        "data_action",
        "batch_safe",
    }
)
_PUBLIC_BACKEND_FIELDS = frozenset(
    {"id", "display_name", "runtime", "description", "capabilities", "health"}
)
_PUBLIC_HEALTH_FORBIDDEN_KEYS = frozenset(
    {
        "python_modules",
        "missing_python_modules",
        "required_python_modules",
        "modules",
        "runtime",
        "executables",
        "environment",
        "install_notes",
        "conda_packages",
        "pip_packages",
        "required_data_resources",
        "missing_resources",
        "resource_constraints",
        "method_schema",
        "parameter_specs",
        "allowed_method_values",
        "allowed_setting_values",
        "required_input_fields",
        "required_method_fields",
        "required_setting_fields",
        "fixed_parameter_specs",
        "component_backend_options",
        "required_component_roles",
        "supported_system_types",
        "validation_levels",
        "path",
        "base_results_dir",
        "input_path",
        "output_path",
        "source_path",
        "target_path",
        "resource_path",
        "file_path",
        "directory",
        "workspace",
        "workspace_path",
        "base_dir",
        "configuration_id",
        "fixed_input",
        "reactants",
        "products",
        "atomic_action_trace",
    }
)


def _bridge_result_timeout_seconds() -> float:
    """Allow synchronous chemistry Actions to use the same long-job time scale."""

    raw = os.environ.get("MINICHEM_MCP_RESULT_TIMEOUT_SECONDS", "7500")
    try:
        value = float(raw)
    except ValueError as exc:
        raise MiniChemMcpError(
            "MINICHEM_MCP_RESULT_TIMEOUT_SECONDS 必须是数字"
        ) from exc
    if not 30 <= value <= 86400:
        raise MiniChemMcpError(
            "MINICHEM_MCP_RESULT_TIMEOUT_SECONDS 必须在 30 到 86400 秒之间"
        )
    return value


class MiniChemMcpError(RuntimeError):
    """表示 MCP 会话、目录或 Action 协议无法安全完成。"""


class MiniChemMcpProtocolError(MiniChemMcpError):
    """表示服务返回的结构不符合 MiniChem 公开协议。"""


def _load_configured_evaluation_resource_budget() -> dict[str, Any]:
    """读取 ARCHE→MiniChem 的任务级资源政策；配置错误时立即失败。

    该配置只约束服务 Catalog 宣布的评测总预算，不为单次 Action 或 job
    自动填充 ``resource_limits``。单次请求仍必须由上层显式提供并由 MiniChem
    服务按同一预算核对。
    """

    try:
        raw = json.loads(_RESOURCE_POLICY_FILE.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MiniChemMcpProtocolError(
            f"MiniChem 资源政策文件不可读：{_RESOURCE_POLICY_FILE}"
        ) from exc
    if not isinstance(raw, Mapping) or set(raw) != {"schema_version", "evaluation_resource_budget"}:
        raise MiniChemMcpProtocolError("MiniChem 资源政策顶层字段不符合 schema")
    if raw.get("schema_version") != 1:
        raise MiniChemMcpProtocolError("MiniChem 资源政策 schema_version 必须为 1")
    budget = raw.get("evaluation_resource_budget")
    if not isinstance(budget, Mapping) or set(budget) != _RESOURCE_POLICY_FIELDS:
        raise MiniChemMcpProtocolError("MiniChem 资源政策 budget 字段不完整")
    if (
        type(budget.get("cpu_cores")) is not int
        or budget["cpu_cores"] < 1
        or type(budget.get("memory_mb")) is not int
        or budget["memory_mb"] < 128
        or type(budget.get("gpu_count")) is not int
        or budget["gpu_count"] < 0
        or budget.get("source") != "evaluation_policy"
        or budget.get("agent_controllable") is not False
        or budget.get("scope") != "per_task"
    ):
        raise MiniChemMcpProtocolError("MiniChem 资源政策 budget 值不符合 schema")
    return deepcopy(dict(budget))


class MiniChemSession(Protocol):
    """适配器使用的最小会话接口，便于接入真实或确定性 MCP 会话。"""

    def initialize(self) -> Any: ...

    def list_tools(self) -> Any: ...

    def call_tool(self, name: str, arguments: Mapping[str, Any] | None = None) -> Any: ...

    def read_resource(self, uri: str) -> Any: ...


class _SessionFactory(Protocol):
    def __call__(self) -> Any: ...


def _canonical_json(value: Any) -> str:
    """把服务响应编码为稳定 JSON；无法编码时抛出协议错误。"""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise MiniChemMcpProtocolError("MiniChem 响应不是严格 JSON") from exc


def _content_items(value: Any) -> list[Any]:
    """读取 MCP CallToolResult/ReadResourceResult 的内容数组。"""

    if isinstance(value, Mapping):
        contents = value.get("contents")
        if isinstance(contents, list):
            return contents
        content = value.get("content")
        if isinstance(content, list):
            return content
    for name in ("contents", "content"):
        items = getattr(value, name, None)
        if isinstance(items, list):
            return items
    return []


def _content_text(item: Any) -> str | None:
    """从文本资源、文本内容或嵌套 resource 中提取诊断/JSON 文本。"""

    if isinstance(item, Mapping):
        text = item.get("text")
        if isinstance(text, str):
            return text
        blob = item.get("blob")
        if isinstance(blob, str):
            return blob
        resource = item.get("resource")
    else:
        text = getattr(item, "text", None)
        if isinstance(text, str):
            return text
        blob = getattr(item, "blob", None)
        if isinstance(blob, str):
            return blob
        resource = getattr(item, "resource", None)
    if resource is not None and resource is not item:
        return _content_text(resource)
    return None


def _mcp_error_value(
    message: str,
    *,
    raw: Any = None,
    code: str = "mcp_error",
) -> dict[str, Any]:
    """保留 MCP 错误并分类服务冻结目录漂移，供恢复层选择安全动作。

    输入是 MCP 返回的错误文本和可选原始内容；动作只生成 ARCHE 内部的稳定错误
    码，不修改服务端文本或重试请求。返回值保留原始诊断，目录文件被并行服务或
    本地直连调用替换时使用 catalog_freeze_drift，其他错误仍使用调用方传入的
    错误码。函数不会读取或修复 MiniChem workspace；目录漂移必须由服务端以隔离
    workspace 重启后再恢复。
    """

    if code == "mcp_error" and any(
        marker in message for marker in _CATALOG_FREEZE_DRIFT_MARKERS
    ):
        code = "catalog_freeze_drift"
    error: dict[str, Any] = {"code": code, "message": message}
    if raw is not None:
        error["raw"] = raw
    return {"status": "mcp_error", "minichem_status": "mcp_error", "error": error}


def _serialize_mcp_object(value: Any) -> Any:
    """把 MCP Pydantic/dataclass 响应转成可持久化 JSON 值。"""

    if isinstance(value, Mapping):
        return {str(key): _serialize_mcp_object(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize_mcp_object(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _serialize_mcp_object(model_dump(mode="json"))
        except TypeError:
            return _serialize_mcp_object(model_dump())
    as_dict = getattr(value, "dict", None)
    if callable(as_dict):
        try:
            return _serialize_mcp_object(as_dict())
        except TypeError:
            pass
    return deepcopy(value)


def _content_value(value: Any) -> Any:
    """从 MCP 结果、文本内容或普通字典中取出结构化 JSON。

    ``ReadResourceResult`` 使用 ``contents``，``CallToolResult`` 使用
    ``content``；两者都可能只有 plain text。plain text 或 ``isError`` 不被
    静默丢弃，而是转换成带原始诊断的映射，供调用方决定失败边界。
    """

    if isinstance(value, Mapping):
        is_error = bool(value.get("isError", value.get("is_error", False)))
        structured = value.get("structuredContent", value.get("structured_content"))
    else:
        is_error = bool(getattr(value, "isError", getattr(value, "is_error", False)))
        structured = getattr(value, "structuredContent", None)
        if structured is None:
            structured = getattr(value, "structured_content", None)

    # MCP 的 isError 是 envelope 级别的否决信号。即使服务错误地同时填充了
    # structuredContent，也只能把这次调用当作错误；否则会把失败伪装成成功。
    items = _content_items(value)
    texts: list[str] = []
    parsed_text: Any = None
    for item in items:
        text = _content_text(item)
        if not isinstance(text, str):
            continue
        texts.append(text)
        if parsed_text is None:
            try:
                parsed_text = json.loads(text)
            except json.JSONDecodeError:
                pass
    if is_error:
        if texts:
            message = "\n".join(texts).strip() or "MCP 返回空文本错误"
            return _mcp_error_value(message, raw=texts, code="mcp_error")
        raw = structured if structured is not None else repr(value)
        return _mcp_error_value(
            "MCP 工具返回 isError=true 但没有诊断文本",
            raw=_serialize_mcp_object(raw),
        )
    if structured is not None:
        return deepcopy(structured)
    if isinstance(value, Mapping):
        # 一些确定性会话直接返回服务的 object，而不是 MCP envelope。
        if "status" in value or "catalog_hash" in value or "actions" in value:
            return deepcopy(dict(value))
    if parsed_text is not None:
        return parsed_text
    if texts:
        message = "\n".join(texts).strip() or "MCP 返回空文本错误"
        return _mcp_error_value(message, raw=texts, code="plain_text_error")
    if value is None:
        raise MiniChemMcpProtocolError("MCP 结果为空")
    return _serialize_mcp_object(value)


async def _maybe_await(value: Any) -> Any:
    """兼容真实异步 ClientSession 和确定性同步假会话。"""

    return await value if inspect.isawaitable(value) else value


@dataclass
class _SessionCommand:
    """保存同步线程提交给生命周期协程的一次会话调用。"""

    method: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    result: ConcurrentFuture[Any]


class _AsyncSessionBridge:
    """在专用事件循环线程中维持一个单任务 MCP 会话生命周期。

    异步上下文的进入、会话初始化、跨线程提交的所有命令以及退出都由同一
    ``_run_session_lifecycle`` 协程任务执行。同步调用线程只把命令放入该任务
    使用的 asyncio 队列并等待一个 concurrent future，因此不会在另一个 task
    中退出 anyio cancel scope。
    """

    _CLOSE_SENTINEL = object()

    def __init__(self, endpoint: str, factory: _SessionFactory | None = None) -> None:
        self.endpoint = endpoint
        self.factory = factory
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ready = threading.Event()
        self._failure: BaseException | None = None
        self._session: Any = None
        self._context: Any = None
        self._direct_session = False
        self._pending_factory_result: Any = None
        self._command_queue: asyncio.Queue[Any] | None = None
        self._lifecycle_task: asyncio.Task[Any] | None = None
        self._lifecycle_done = threading.Event()
        self._state_lock = threading.RLock()
        self._start_lock = threading.Lock()
        self._pending_commands: dict[int, _SessionCommand] = {}
        self._accept_commands = False
        self._close_requested = False
        self._close_failure: BaseException | None = None
        self._close_failure_reported = False

    def start(self) -> None:
        """建立可接受命令的会话；断连后先回收旧生命周期再重建。"""

        with self._start_lock:
            self._prepare_broken_session_for_restart()
            self._start_unlocked()

    def _prepare_broken_session_for_restart(self) -> None:
        """等待已失效的会话线程退出，避免重试把命令放回旧事件循环。"""

        with self._state_lock:
            thread = self._thread
            if thread is None or not thread.is_alive():
                return
            if (
                self._failure is None
                and self._accept_commands
                and not self._close_requested
            ):
                return
            self._accept_commands = False
            self._close_requested = True
            loop = self._loop
        if loop is not None:
            try:
                # 失效生命周期可能还停在一个未完成 await 中；停止事件循环
                # 让 run() 的 finally 负责取消该 task 并执行上下文退出。
                loop.call_soon_threadsafe(loop.stop)
            except RuntimeError:
                pass
        thread.join(timeout=30)
        if thread.is_alive():
            raise MiniChemMcpError("失效的 MiniChem MCP 会话线程未能退出")
        with self._state_lock:
            if self._thread is thread:
                self._thread = None

    def _start_unlocked(self) -> None:
        """建立会话并完成 MCP initialize；失败时不保留半开连接。"""

        with self._state_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            if self._direct_session and self._session is not None:
                return
            self._thread = None
            self._loop = None
            self._ready.clear()
            self._lifecycle_done.clear()
            self._failure = None
            self._close_failure = None
            self._close_failure_reported = False
            self._session = None
            self._context = None
            self._direct_session = False
            self._pending_factory_result = None
            self._command_queue = None
            self._lifecycle_task = None
            self._pending_commands = {}
            self._accept_commands = False
            self._close_requested = False

        # 确定性同步假会话不需要事件循环；直接调用可避免为测试替身制造
        # Streamable HTTP 线程，也让同步 ClientSession 的 close 生命周期可核对。
        if self.factory is not None:
            try:
                candidate = self.factory()
            except BaseException as exc:
                raise MiniChemMcpError(f"无法建立 MiniChem MCP 会话：{exc}") from exc
            async_methods = any(
                inspect.iscoroutinefunction(getattr(candidate, method, None))
                for method in ("initialize", "list_tools", "call_tool", "read_resource")
            )
            if (
                not inspect.isawaitable(candidate)
                and not hasattr(candidate, "__aenter__")
                and not async_methods
            ):
                self._session = candidate
                try:
                    initialized = self._session.initialize()
                    if inspect.isawaitable(initialized):
                        asyncio.run(initialized)
                except BaseException as exc:
                    self._close_direct_session_after_open_failure(candidate)
                    self._session = None
                    raise MiniChemMcpError(f"无法建立 MiniChem MCP 会话：{exc}") from exc
                self._direct_session = True
                return
            # 异步上下文工厂可能已经执行了有副作用的构造；把同一个对象交给
            # 事件循环线程，避免再次调用工厂而泄漏一个未关闭的 transport。
            self._pending_factory_result = candidate

        def run() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            with self._state_lock:
                self._loop = loop
                self._command_queue = asyncio.Queue()
            lifecycle = loop.create_task(self._run_session_lifecycle())
            with self._state_lock:
                self._lifecycle_task = lifecycle
            try:
                loop.run_until_complete(lifecycle)
            except BaseException as exc:  # 传回调用线程，避免吞掉协议错误
                with self._state_lock:
                    if self._failure is None:
                        self._failure = exc
                    self._accept_commands = False
                    self._close_requested = True
                self._reject_pending_commands(
                    ConnectionError("MiniChem MCP 会话生命周期异常终止，命令未完成")
                )
                self._ready.set()
            finally:
                # 事件循环被 transport 断连提前停止时，生命周期 task 仍可能
                # 处于当前命令的 await 中。取消并重新短暂运行同一个 task，
                # 让它在原上下文中执行 __aexit__，避免遗留异步 task/连接资源。
                if not lifecycle.done():
                    lifecycle.cancel()
                    try:
                        loop.run_until_complete(lifecycle)
                    except BaseException:
                        # 主生命周期异常已在上面保存；清理期间的取消异常
                        # 不应覆盖它，关闭异常仍由 _run_session_lifecycle 记录。
                        pass
                # 事件循环/transport 断连时，当前命令已经从 asyncio.Queue
                # 取出，不能只依赖生命周期协程的队列清理；先完成所有仍在途
                # future，才能让带 operation_id 的上层调用重建会话并安全重放。
                self._reject_pending_commands(
                    ConnectionError("MiniChem MCP 会话已结束，命令未完成")
                )
                self._lifecycle_done.set()
                with self._state_lock:
                    if self._lifecycle_task is lifecycle:
                        self._lifecycle_task = None
                loop.close()
                with self._state_lock:
                    if self._loop is loop:
                        self._loop = None
                    if self._command_queue is not None:
                        self._command_queue = None

        self._thread = threading.Thread(target=run, name="minichem-mcp-session", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=30):
            try:
                self.close()
            except MiniChemMcpError:
                # 超时是打开调用者最直接的诊断；关闭异常会保留在线程状态中。
                pass
            raise MiniChemMcpError("建立 MiniChem MCP 会话超时")
        if self._failure is not None:
            failure = self._failure
            thread = self._thread
            if thread is not None:
                thread.join(timeout=30)
                if thread.is_alive():
                    raise MiniChemMcpError(f"无法释放失败的 MiniChem MCP 会话：{failure}") from failure
            with self._state_lock:
                if self._thread is thread:
                    self._thread = None
            if isinstance(failure, MiniChemMcpError):
                raise failure
            raise MiniChemMcpError(f"无法建立 MiniChem MCP 会话：{failure}") from failure
        if self._session is None:
            raise MiniChemMcpError("MiniChem MCP 会话没有完成 initialize")

    def _close_direct_session_after_open_failure(self, session: Any) -> None:
        """初始化失败后关闭同步会话，避免确定性替身留下半开资源。"""

        close = getattr(session, "close", None)
        if not callable(close):
            return
        try:
            result = close()
            if inspect.isawaitable(result):
                asyncio.run(result)
        except BaseException:
            # 初始化错误是调用者需要修复的主因；清理错误不能让资源继续泄漏，
            # 但也不能覆盖主异常。
            return

    def _mark_transport_failure(self, error: BaseException) -> None:
        """记录 HTTP transport 失效并唤醒所有调用，让上层安全重试。

        Streamable HTTP 的后台 GET 重连在耗尽次数后可能正常结束，而不会让
        ClientSession 的 lifecycle task 退出。此时必须主动拒绝仍在等待的同步
        future，并取消旧 lifecycle；带 operation_id 的上层重试会随后建立新会话，
        没有幂等身份的副作用请求仍由适配器阻止重放。
        """

        with self._state_lock:
            if self._close_requested and self._failure is None:
                return
            if self._failure is None:
                self._failure = error
            self._accept_commands = False
            self._close_requested = True
            lifecycle = self._lifecycle_task
            loop = self._loop
        self._reject_pending_commands(error)
        if lifecycle is None or lifecycle.done() or loop is None:
            return
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        try:
            if running_loop is loop:
                lifecycle.cancel()
            else:
                loop.call_soon_threadsafe(lifecycle.cancel)
        except RuntimeError:
            # lifecycle 已经进入 finally 或事件循环已关闭；run() 仍会完成
            # pending future 的最后一次拒绝和旧线程回收。
            return

    async def _run_session_lifecycle(self) -> None:
        """在同一 asyncio task 中执行打开、命令循环和关闭。"""

        ready_announced = False
        try:
            try:
                await self._open()
            except BaseException as exc:
                with self._state_lock:
                    if self._failure is None:
                        self._failure = exc
            else:
                with self._state_lock:
                    close_requested = self._close_requested
                    self._accept_commands = not close_requested
                self._ready.set()
                ready_announced = True
                if not close_requested:
                    await self._serve_session_commands()
        finally:
            with self._state_lock:
                self._accept_commands = False
            try:
                await self._close_context()
            except BaseException as exc:
                with self._state_lock:
                    self._close_failure = exc
            finally:
                self._reject_pending_commands(
                    MiniChemMcpError("MiniChem MCP 会话已关闭，命令未执行")
                )
                if not ready_announced:
                    self._ready.set()

    async def _serve_session_commands(self) -> None:
        """在生命周期 task 中串行执行同步线程提交的会话命令。"""

        queue = self._command_queue
        if queue is None:
            raise MiniChemMcpError("MiniChem MCP 命令队列未建立")
        while True:
            command = await queue.get()
            if command is self._CLOSE_SENTINEL:
                return
            if not isinstance(command, _SessionCommand):
                continue
            await self._execute_session_command(command)

    async def _execute_session_command(self, command: _SessionCommand) -> None:
        """在生命周期 task 中调用会话方法并完成对应同步 future。"""

        try:
            session = self._session
            if session is None:
                raise MiniChemMcpError("MiniChem MCP 会话未打开")
            function = getattr(session, command.method, None)
            if not callable(function):
                raise MiniChemMcpProtocolError(f"MCP 会话不支持 {command.method}")
            result = await _maybe_await(function(*command.args, **command.kwargs))
        except asyncio.CancelledError as exc:
            # CancelledError 继承 BaseException，不能原样放入同步 future，
            # 否则 call() 不会把它转换为可重试的 MiniChemMcpError。
            self._reject_command(
                command,
                ConnectionError("MiniChem MCP 会话命令因生命周期取消而未完成"),
            )
            raise
        except BaseException as exc:
            self._reject_command(command, exc)
        else:
            self._resolve_command(command, result)

    def _resolve_command(self, command: _SessionCommand, value: Any) -> None:
        """完成会话命令并从在途表移除其同步 future。"""

        with self._state_lock:
            self._pending_commands.pop(id(command), None)
        if not command.result.done():
            command.result.set_result(value)

    def _reject_command(self, command: _SessionCommand, error: BaseException) -> None:
        """拒绝会话命令并从在途表移除其同步 future。"""

        with self._state_lock:
            self._pending_commands.pop(id(command), None)
        if not command.result.done():
            command.result.set_exception(error)

    def _reject_pending_commands(self, error: BaseException) -> None:
        """拒绝队列中及当前执行中的命令，避免同步线程永久等待。"""

        with self._state_lock:
            commands = list(self._pending_commands.values())
            self._pending_commands.clear()
        for command in commands:
            if not command.result.done():
                command.result.set_exception(error)

    async def _open(self) -> None:
        if self.factory is not None:
            context = self._pending_factory_result
            self._pending_factory_result = None
            if context is None:
                context = self.factory()
            if hasattr(context, "__aenter__"):
                self._context = context
                self._session = await context.__aenter__()
            else:
                self._session = await _maybe_await(context)
        else:
            try:
                from mcp import ClientSession
                from mcp.client.streamable_http import streamable_http_client
                from mcp.shared._httpx_utils import create_mcp_http_client
            except Exception as exc:  # pragma: no cover - 标准环境已提供 mcp
                raise MiniChemMcpError("标准环境缺少 MCP Streamable HTTP 客户端") from exc
            http_client = create_mcp_http_client()

            async def observe_response(response: Any) -> None:
                """观察 Streamable HTTP 会话失效响应，不改变正常长请求超时。"""

                request = getattr(response, "request", None)
                method = getattr(request, "method", None)
                if (
                    method in {"GET", "POST"}
                    and getattr(response, "status_code", None) == 409
                ):
                    self._mark_transport_failure(
                        ConnectionError(
                            "MiniChem MCP Streamable HTTP transport 收到 HTTP 409"
                        )
                    )

            http_client.event_hooks["response"].append(observe_response)
            transport = streamable_http_client(
                self.endpoint,
                http_client=http_client,
            )
            self._context = _CombinedMcpContext(
                transport,
                ClientSession,
                owned_client=http_client,
            )
            self._session = await self._context.__aenter__()
        await _maybe_await(self._session.initialize())

    async def _close_context(self) -> None:
        """由生命周期 task 退出上下文并记录完整关闭异常。"""

        if self._context is not None:
            context = self._context
            self._context = None
            try:
                await context.__aexit__(None, None, None)
            finally:
                self._session = None
        elif self._session is not None:
            close = getattr(self._session, "close", None)
            if close is not None:
                await _maybe_await(close())
            self._session = None

    def call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """在线程安全地调用一个会话方法并等待其结果。"""

        self.start()
        if self._direct_session:
            if self._session is None:
                raise MiniChemMcpError("MiniChem MCP 会话未打开")
            function = getattr(self._session, method, None)
            if not callable(function):
                raise MiniChemMcpProtocolError(f"MCP 会话不支持 {method}")
            try:
                result = function(*args, **kwargs)
                if inspect.isawaitable(result):
                    result = asyncio.run(result)
                return result
            except MiniChemMcpError:
                raise
            except Exception as exc:
                raise MiniChemMcpError(f"MiniChem MCP {method} 调用失败：{exc}") from exc
        with self._state_lock:
            loop = self._loop
            if loop is None or not self._accept_commands or self._close_requested:
                raise MiniChemMcpError("MiniChem MCP 会话未打开")
            result: ConcurrentFuture[Any] = ConcurrentFuture()
            command = _SessionCommand(method, args, dict(kwargs), result)
            self._pending_commands[id(command)] = command

            def enqueue() -> None:
                queue = self._command_queue
                if queue is None or self._close_requested:
                    self._reject_command(
                        command,
                        MiniChemMcpError("MiniChem MCP 会话正在关闭"),
                    )
                    return
                queue.put_nowait(command)

            try:
                loop.call_soon_threadsafe(enqueue)
            except RuntimeError as exc:
                self._reject_command(command, MiniChemMcpError("MiniChem MCP 会话未打开"))
                raise MiniChemMcpError("MiniChem MCP 会话未打开") from exc
        try:
            return result.result(timeout=_bridge_result_timeout_seconds())
        except MiniChemMcpError:
            raise
        except Exception as exc:
            raise MiniChemMcpError(f"MiniChem MCP {method} 调用失败：{exc}") from exc

    def close(self) -> None:
        """关闭会话线程并等待所有 Streamable HTTP 资源释放。"""

        if self._direct_session:
            with self._state_lock:
                session = self._session
                self._session = None
                self._direct_session = False
            if session is None:
                return
            close = getattr(session, "close", None)
            if callable(close):
                try:
                    result = close()
                    if inspect.isawaitable(result):
                        asyncio.run(result)
                except BaseException as exc:
                    with self._state_lock:
                        self._close_failure = exc
                        self._close_failure_reported = True
                    raise MiniChemMcpError(f"关闭 MiniChem MCP 会话失败：{exc}") from exc
            return
        with self._state_lock:
            thread = self._thread
            loop = self._loop
            if thread is None:
                failure = self._close_failure
                if failure is not None and not self._close_failure_reported:
                    self._close_failure_reported = True
                    self._raise_close_failure(failure)
                return
            self._close_requested = True
            self._accept_commands = False
            if loop is not None:
                try:
                    loop.call_soon_threadsafe(self._enqueue_close_sentinel)
                except RuntimeError:
                    # 线程已经完成生命周期；join 后仍检查关闭异常。
                    pass
        if thread is None:
            return
        thread.join(timeout=30)
        if thread.is_alive():
            raise MiniChemMcpError("关闭 MiniChem MCP 会话超时")
        with self._state_lock:
            if self._thread is thread:
                self._thread = None
            failure = self._close_failure
            should_report_failure = failure is not None and not self._close_failure_reported
            if should_report_failure:
                self._close_failure_reported = True
        if should_report_failure:
            self._raise_close_failure(failure)

    def _enqueue_close_sentinel(self) -> None:
        """在事件循环线程中请求生命周期 task 退出命令循环。"""

        queue = self._command_queue
        if queue is not None:
            queue.put_nowait(self._CLOSE_SENTINEL)

    @staticmethod
    def _raise_close_failure(failure: BaseException) -> None:
        """把生命周期 task 保存的关闭错误转换为同步调用方异常。"""

        if isinstance(failure, MiniChemMcpError):
            raise failure
        raise MiniChemMcpError(f"关闭 MiniChem MCP 会话失败：{failure}") from failure


class _CombinedMcpContext:
    """把 Streamable HTTP transport 的流元组封装成 ClientSession 上下文。"""

    def __init__(
        self,
        transport: Any,
        client_session: Any,
        *,
        owned_client: Any = None,
    ) -> None:
        self.transport = transport
        self.client_session = client_session
        self.owned_client = owned_client
        self.streams: Any = None
        self.session: Any = None
        self._transport_entered = False
        self._client_entered = False

    async def __aenter__(self) -> Any:
        try:
            if self.owned_client is not None:
                await self.owned_client.__aenter__()
                self._client_entered = True
            self.streams = await self.transport.__aenter__()
            self._transport_entered = True
            receive, send, *_ = self.streams
            self.session = self.client_session(receive, send)
            await self.session.__aenter__()
        except BaseException:
            self.session = None
            try:
                if self._transport_entered:
                    self._transport_entered = False
                    await self.transport.__aexit__(None, None, None)
            finally:
                if self._client_entered:
                    self._client_entered = False
                    await self.owned_client.__aexit__(None, None, None)
            raise
        return self.session

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            if self.session is not None:
                await self.session.__aexit__(exc_type, exc, traceback)
        finally:
            try:
                if self._transport_entered:
                    self._transport_entered = False
                    await self.transport.__aexit__(exc_type, exc, traceback)
            finally:
                if self._client_entered:
                    self._client_entered = False
                    await self.owned_client.__aexit__(exc_type, exc, traceback)


class _ReadOnlySnapshot(Mapping[str, Any]):
    """保存目录副本，并在每次读取时返回新的可修改副本。

    目录对象会被写入 checkpoint 和 Planner 投影；如果直接暴露服务响应的
    dict，调用方改动嵌套列表就会悄悄改变冻结目录，导致 hash 与实际投影不再
    对应。这个 Mapping 只允许读取，读取出的值再深复制一次，避免这种串改。
    """

    __slots__ = ("_value",)

    def __init__(self, value: Mapping[str, Any]) -> None:
        self._value = deepcopy(dict(value))

    def __getitem__(self, key: str) -> Any:
        return deepcopy(self._value[key])

    def __iter__(self):
        return iter(self._value)

    def __len__(self) -> int:
        return len(self._value)


@dataclass(frozen=True)
class MiniChemCatalogSnapshot:
    """保存任务冻结的 MiniChem Action Catalog 及其内容摘要。"""

    catalog_hash: str
    catalog_version: str
    snapshot: Mapping[str, Any]

    def __post_init__(self) -> None:
        """把构造参数复制到只读容器，防止外部字典继续改变冻结内容。"""

        if not isinstance(self.snapshot, Mapping):
            raise TypeError("MiniChem catalog snapshot 必须是 object")
        object.__setattr__(self, "snapshot", _ReadOnlySnapshot(self.snapshot))

    @property
    def actions(self) -> tuple[Mapping[str, Any], ...]:
        """返回目录中的只读 Action 记录。"""

        values = self.snapshot.get("actions")
        return tuple(deepcopy(dict(item)) for item in values) if isinstance(values, list) else ()

    @property
    def backends(self) -> tuple[Mapping[str, Any], ...]:
        """返回目录中的只读 Backend 记录。"""

        values = self.snapshot.get("backends")
        return tuple(deepcopy(dict(item)) for item in values) if isinstance(values, list) else ()

    def planner_projection(self) -> dict[str, Any]:
        """构造 Planner 的 Action 摘要投影，不复制旧 Tool Card 或入口函数定义。

        摘要覆盖冻结目录中的全部 Action。每项同时保留领域、执行层、输入/输出
        语义、Backend 可用性和选择政策；字段级请求合同仍通过 discovery/inspect
        接口按需读取，避免把所有合同一次性拼进模型提示。
        """

        actions = []
        for raw in self.actions:
            action_id = raw.get("id")
            if not isinstance(action_id, str) or not action_id:
                continue
            actions.append(
                {
                    "action_id": action_id,
                    "action_version": raw.get("action_version", raw.get("version")),
                    "category": raw.get("category"),
                    "domain": raw.get("domain", raw.get("category")),
                    "description": raw.get("description"),
                    "backend_ids": list(raw.get("backend_ids", [])),
                    "required_inputs": list(raw.get("required_inputs", [])),
                    "optional_inputs": list(raw.get("optional_inputs", [])),
                    "input_semantic_types": list(
                        raw.get("input_semantic_types", raw.get("required_inputs", []))
                    ),
                    "selection_policy": raw.get("selection_policy"),
                    "primary_output": raw.get("primary_output"),
                    "output_semantic_types": list(
                        raw.get("output_semantic_types", [])
                    ),
                    "execution_layer": raw.get("execution_layer", "predefined_actions"),
                    "execution_class": raw.get("execution_class"),
                    "requires_network": raw.get("requires_network"),
                    "capability_tags": list(raw.get("capability_tags", [])),
                    "data_action": bool(raw.get("data_action", False)),
                }
            )
        return {
            "catalog_hash": self.catalog_hash,
            "catalog_version": self.catalog_version,
            "actions": actions,
            "backends": [deepcopy(dict(item)) for item in self.backends],
        }


@dataclass(frozen=True)
class MiniChemActionRequest:
    """保存一次显式 Action/Backend 请求，禁止隐式 provider 或科学默认值。"""

    action_id: str
    backend_id: str | None
    inputs: Mapping[str, Any]
    method_spec: Mapping[str, Any] = ()
    action_settings: Mapping[str, Any] = ()
    resource_limits: Mapping[str, Any] = ()
    source_id: str | None = None
    component_backends: Mapping[str, str] = ()

    def to_payload(self) -> dict[str, Any]:
        """生成 MCP execute_action 所需的稳定请求 JSON。"""

        return {
            "action_id": self.action_id,
            "backend_id": self.backend_id,
            "component_backends": deepcopy(dict(self.component_backends or {})),
            "source_id": self.source_id,
            "inputs": deepcopy(dict(self.inputs)),
            "method_spec": deepcopy(dict(self.method_spec or {})),
            "action_settings": deepcopy(dict(self.action_settings or {})),
            "resource_limits": deepcopy(dict(self.resource_limits or {})),
        }


def _validate_operation_id(operation_id: str) -> str:
    """校验可持久化且不会改变作业目录含义的 operation identity。"""

    if not isinstance(operation_id, str) or _SAFE_OPERATION.fullmatch(operation_id) is None:
        raise ValueError("operation_id 必须是安全的 ASCII 标识")
    return operation_id


def _validate_job_id(job_id: Any) -> str:
    """校验 MiniChem 服务生成的 job identity，避免查询越界路径或伪造作业。"""

    if not isinstance(job_id, str) or _JOB_ID.fullmatch(job_id) is None:
        raise ValueError("job_id 必须匹配 ^job_(?:[0-9a-f]{8}|[0-9a-f]{32})$")
    return job_id


def _validate_submission_key(submission_key: Any) -> str:
    """校验 scope 提交键；它是稳定语义身份，不是本地文件名。"""

    if not isinstance(submission_key, str):
        raise ValueError("submission_key 必须是字符串")
    normalized = submission_key.strip()
    if _SUBMISSION_KEY.fullmatch(normalized) is None:
        raise ValueError("submission_key 必须是非空的安全 ASCII 标识")
    return normalized


def _validate_response_job_identity(
    value: Any,
    expected_job_id: str,
    *,
    tool_name: str,
) -> None:
    """核对服务响应主体中的 job_id，防止返回错作业却被掩盖。

    get/collect/cancel 的响应既可能在顶层返回 job_id，也可能在 ``job``、
    ``error`` 等嵌套对象中返回。资源余量中的 ``active_jobs`` 是并行作业概览，
    其中出现其他合法 job_id 不表示响应主体串号；这些身份只校验格式，不与
    本次查询目标比较。其他同名字段仍须全部匹配目标作业。
    """

    def visit(current: Any, path: str) -> None:
        if isinstance(current, Mapping):
            for key, nested in current.items():
                nested_path = f"{path}.{key}" if path else str(key)
                if key == "job_id":
                    try:
                        returned_job_id = _validate_job_id(nested)
                    except ValueError as exc:
                        raise MiniChemMcpProtocolError(
                            f"{tool_name} 返回的 {nested_path} 不是合法 job_id"
                        ) from exc
                    is_resource_overview = nested_path.startswith(
                        "resource_availability.active_jobs["
                    )
                    if not is_resource_overview and returned_job_id != expected_job_id:
                        raise MiniChemMcpProtocolError(
                            f"{tool_name} 返回了错误的 job_id："
                            f"expected={expected_job_id!r} actual={returned_job_id!r}"
                        )
                visit(nested, nested_path)
        elif isinstance(current, list):
            for index, nested in enumerate(current):
                visit(nested, f"{path}[{index}]")

    visit(value, "")


def _validate_catalog_id(value: Any, label: str) -> str:
    """核对 Action、Backend 或来源标识使用 MiniChem 的 lower_snake_case 语法。"""

    if not isinstance(value, str) or _CATALOG_ID.fullmatch(value) is None:
        raise MiniChemMcpProtocolError(f"{label} 必须是 lower_snake_case 标识")
    return value


def _validate_public_health_projection(value: Any, label: str) -> None:
    """核对公开健康投影没有重新暴露运行时、安装信息或本机路径。"""

    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key) in _PUBLIC_HEALTH_FORBIDDEN_KEYS:
                raise MiniChemMcpProtocolError(
                    f"MiniChem public catalog {label} 含禁止公开字段：{key}"
                )
            _validate_public_health_projection(nested, f"{label}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _validate_public_health_projection(nested, f"{label}[{index}]")
    elif isinstance(value, str) and (
        any(marker in value for marker in ("/mnt/", "/cpfs04/", "local_data", "src_old"))
        or value.startswith(("/", "~"))
        or re.match(r"^[A-Za-z]:[\\/]", value) is not None
    ):
        raise MiniChemMcpProtocolError(
            f"MiniChem public catalog {label} 含本机路径"
        )


def validate_public_catalog_projection(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """完整校验 MiniChem public catalog projection 及其独立内容摘要。"""

    value = deepcopy(dict(snapshot))
    missing = _PUBLIC_CATALOG_FIELDS - set(value)
    unknown = set(value) - _PUBLIC_CATALOG_FIELDS
    if missing or unknown:
        raise MiniChemMcpProtocolError(
            "MiniChem public catalog projection 顶层字段不匹配："
            f"missing={sorted(missing)} unknown={sorted(unknown)}"
        )
    if value.get("schema_version") != 7:
        raise MiniChemMcpProtocolError(
            "MiniChem public catalog projection schema_version 必须为 7"
        )
    if value.get("catalog_visibility_policy") != "public_contract_projection":
        raise MiniChemMcpProtocolError(
            "MiniChem public catalog projection visibility policy 非法"
        )
    if value.get("catalog_projection_schema_version") != 1:
        raise MiniChemMcpProtocolError(
            "MiniChem public catalog projection schema_version 必须为 1"
        )
    canonical_hash = value.get("catalog_hash")
    projection_hash = value.get("catalog_projection_hash")
    if not isinstance(canonical_hash, str) or _SHA256.fullmatch(canonical_hash) is None:
        raise MiniChemMcpProtocolError(
            "MiniChem public catalog catalog_hash 不是 canonical SHA-256"
        )
    if not isinstance(projection_hash, str) or _SHA256.fullmatch(projection_hash) is None:
        raise MiniChemMcpProtocolError(
            "MiniChem public catalog catalog_projection_hash 不是 SHA-256"
        )
    projection_payload = deepcopy(value)
    projection_payload.pop("catalog_projection_hash", None)
    calculated_projection_hash = hashlib.sha256(
        _canonical_json(projection_payload).encode("utf-8")
    ).hexdigest()
    if projection_hash != calculated_projection_hash:
        raise MiniChemMcpProtocolError(
            "MiniChem public catalog catalog_projection_hash 与公开内容不一致："
            f"expected={calculated_projection_hash} actual={projection_hash}"
        )
    for field in ("execution_layers", "actions", "backends", "resources"):
        if not isinstance(value.get(field), list):
            raise MiniChemMcpProtocolError(
                f"MiniChem public catalog {field} 必须是数组"
            )
    expected_layers = {
        "predefined_actions": (
            "validated common scientific operations",
            False,
        ),
        "native_software": (
            "Agent-authored native inputs and exact allowlisted commands",
            False,
        ),
        "programmable_analysis": (
            "Agent-authored programs in explicitly selected chemistry runtimes",
            False,
        ),
    }
    layer_ids: set[str] = set()
    for index, layer in enumerate(value["execution_layers"]):
        if not isinstance(layer, Mapping) or set(layer) != {"id", "role", "mandatory"}:
            raise MiniChemMcpProtocolError(
                f"MiniChem public catalog execution_layers[{index}] 字段不完整"
            )
        layer_id = layer.get("id")
        if layer_id not in expected_layers or layer_id in layer_ids:
            raise MiniChemMcpProtocolError(
                f"MiniChem public catalog execution_layers[{index}].id 非法或重复"
            )
        expected_role, expected_mandatory = expected_layers[layer_id]
        if (
            layer.get("role") != expected_role
            or layer.get("mandatory") is not expected_mandatory
        ):
            raise MiniChemMcpProtocolError(
                f"MiniChem public catalog execution_layers[{index}] 内容不符合 v7 合同"
            )
        layer_ids.add(layer_id)
    if layer_ids != set(expected_layers):
        raise MiniChemMcpProtocolError(
            "MiniChem public catalog execution_layers 必须完整声明三个执行层"
        )
    mode = value.get("discovery_mode")
    if mode not in {"progressive", "full"}:
        raise MiniChemMcpProtocolError(
            "MiniChem public catalog discovery_mode 必须为 progressive 或 full"
        )
    if value.get("exposure_policy") != {
        "progressive": "progressive_discovery",
        "full": "atomic_all",
    }[mode]:
        raise MiniChemMcpProtocolError(
            "MiniChem public catalog exposure_policy 与 discovery_mode 不一致"
        )
    for field, expected in {
        "task_specific_tool_filtering": False,
        "backend_selection_policy": "per_action_explicit",
        "scientific_resource_selection_policy": "agent_explicit_no_default",
        "automatic_fallback": False,
    }.items():
        if value.get(field) != expected:
            raise MiniChemMcpProtocolError(
                f"MiniChem public catalog {field} 不符合 v7 固定政策"
            )
    provider_policies = value.get("provider_selection_policies")
    allowed_provider_policies = {
        "agent_backend_required",
        "agent_components_required",
        "agent_source_required",
        "fixed_source",
        "internal_deterministic",
    }
    if (
        not isinstance(provider_policies, list)
        or any(not isinstance(item, str) for item in provider_policies)
        or provider_policies != sorted(set(provider_policies))
        or any(item not in allowed_provider_policies for item in provider_policies)
    ):
        raise MiniChemMcpProtocolError(
            "MiniChem public catalog provider_selection_policies 不符合 v7 policy"
        )
    budget = value.get("evaluation_resource_budget")
    budget_fields = {"source", "agent_controllable", "scope"}
    if not isinstance(budget, Mapping) or set(budget) != budget_fields:
        raise MiniChemMcpProtocolError(
            "MiniChem public catalog evaluation_resource_budget 字段不完整"
        )
    if (
        budget.get("source") != "service_policy"
        or budget.get("agent_controllable") is not False
        or budget.get("scope") != "per_task"
    ):
        raise MiniChemMcpProtocolError(
            "MiniChem public catalog evaluation_resource_budget 值不符合 v7 policy"
        )
    action_ids: set[str] = set()
    transition_path_action_id = "search_transition_state_path"
    transition_path_backend_id = "transition_state_path_search"
    for index, action in enumerate(value["actions"]):
        if not isinstance(action, Mapping):
            raise MiniChemMcpProtocolError(
                f"MiniChem public catalog actions[{index}] 必须是 object"
            )
        unknown_action_fields = set(action) - _PUBLIC_ACTION_FIELDS
        if unknown_action_fields:
            raise MiniChemMcpProtocolError(
                f"MiniChem public catalog actions[{index}] 含未知字段："
                f"{sorted(unknown_action_fields)}"
            )
        for required_field in (
            "id",
            "category",
            "description",
            "primary_output",
            "backend_ids",
            "required_inputs",
            "optional_inputs",
            "selection_policy",
        ):
            if required_field not in action:
                raise MiniChemMcpProtocolError(
                    f"MiniChem public catalog actions[{index}] 缺少 {required_field}"
                )
        action_id = _validate_catalog_id(
            action.get("id"), f"public catalog actions[{index}].id"
        )
        if action_id in action_ids:
            raise MiniChemMcpProtocolError(
                f"MiniChem public catalog action_id 重复：{action_id}"
            )
        action_ids.add(action_id)
        backend_choices = action.get("backend_ids")
        if not isinstance(backend_choices, list) or any(
            not isinstance(item, str) for item in backend_choices
        ) or len(backend_choices) != len(set(backend_choices)):
            raise MiniChemMcpProtocolError(
                f"MiniChem public catalog actions[{index}].backend_ids 非法"
            )
        for backend_id in backend_choices:
            _validate_catalog_id(
                backend_id, f"public catalog actions[{index}].backend_ids"
            )
        for field in ("required_inputs", "optional_inputs"):
            if not isinstance(action.get(field), list) or any(
                not isinstance(item, str) for item in action[field]
            ):
                raise MiniChemMcpProtocolError(
                    f"MiniChem public catalog actions[{index}].{field} 必须是字符串数组"
                )
        _validate_public_health_projection(action, f"actions[{index}]")
        if action_id == transition_path_action_id:
            if action.get("backend_ids") != [transition_path_backend_id] or action.get(
                "required_inputs"
            ) != ["reactants", "products"] or action.get("optional_inputs") != [
                "main_force_field", "target_force_field"
            ]:
                raise MiniChemMcpProtocolError(
                    "MiniChem public catalog TS path Action 合同不一致"
                )
    if transition_path_action_id not in action_ids:
        raise MiniChemMcpProtocolError(
            "MiniChem public catalog 缺少 TS path Action"
        )
    backend_ids: set[str] = set()
    for index, backend in enumerate(value["backends"]):
        if not isinstance(backend, Mapping):
            raise MiniChemMcpProtocolError(
                f"MiniChem public catalog backends[{index}] 必须是 object"
            )
        unknown_backend_fields = set(backend) - _PUBLIC_BACKEND_FIELDS
        if unknown_backend_fields:
            raise MiniChemMcpProtocolError(
                f"MiniChem public catalog backends[{index}] 含未知字段："
                f"{sorted(unknown_backend_fields)}"
            )
        for required_field in (
            "id",
            "display_name",
            "runtime",
            "description",
            "capabilities",
            "health",
        ):
            if required_field not in backend:
                raise MiniChemMcpProtocolError(
                    f"MiniChem public catalog backends[{index}] 缺少 {required_field}"
                )
        backend_id = _validate_catalog_id(
            backend.get("id"), f"public catalog backends[{index}].id"
        )
        if backend_id in backend_ids:
            raise MiniChemMcpProtocolError(
                f"MiniChem public catalog backend_id 重复：{backend_id}"
            )
        backend_ids.add(backend_id)
        if (
            not isinstance(backend.get("runtime"), str)
            or _SAFE_RUNTIME_ID.fullmatch(backend["runtime"]) is None
        ):
            raise MiniChemMcpProtocolError(
                f"MiniChem public catalog backends[{index}].runtime 非法"
            )
        capabilities = backend.get("capabilities")
        if not isinstance(capabilities, list) or any(
            not isinstance(item, str) for item in capabilities
        ):
            raise MiniChemMcpProtocolError(
                f"MiniChem public catalog backends[{index}].capabilities 必须是字符串数组"
            )
        # Backend 顶层 runtime 是 Planner 所需的安全执行层身份；禁止字段检查
        # 只作用于 health 及递归健康记录，不能把这个公开身份误判为内部模块信息。
        _validate_public_health_projection(
            backend.get("health"), f"backends[{index}].health"
        )
        _validate_public_health_projection(
            backend.get("description"), f"backends[{index}].description"
        )
    for index, action in enumerate(value["actions"]):
        for backend_id in action.get("backend_ids", []):
            if backend_id not in backend_ids:
                raise MiniChemMcpProtocolError(
                    f"MiniChem public catalog actions[{index}] 引用了未声明 Backend：{backend_id}"
                )
    resource_ids: set[str] = set()
    for index, resource in enumerate(value["resources"]):
        if not isinstance(resource, Mapping) or "id" not in resource:
            raise MiniChemMcpProtocolError(
                f"MiniChem public catalog resources[{index}] 必须包含 id"
            )
        resource_id = _validate_catalog_id(
            resource["id"], f"public catalog resources[{index}].id"
        )
        if resource_id in resource_ids:
            raise MiniChemMcpProtocolError(
                f"MiniChem public catalog resource_id 重复：{resource_id}"
            )
        resource_ids.add(resource_id)
        _validate_public_health_projection(resource, f"resources[{index}]")
    return value


def _validate_action_request_mapping(value: Any) -> dict[str, Any]:
    """核对同步 Action 请求的完整顶层字段，拒绝作业身份和未知字段。"""

    if not isinstance(value, Mapping):
        raise MiniChemMcpProtocolError("execute_action request 必须是 object")
    supplied = set(value)
    unknown = supplied - _ACTION_REQUEST_FIELDS
    if unknown:
        identity = sorted(unknown & {"operation_id", "job_id"})
        if identity:
            raise MiniChemMcpProtocolError(
                "同步 execute_action 不接受 operation_id/job_id："
                f"{identity}"
            )
        raise MiniChemMcpProtocolError(
            f"同步 execute_action request 含未知字段：{sorted(unknown)}"
        )
    payload = deepcopy(dict(value))
    _validate_catalog_id(payload.get("action_id"), "action_id")
    for field in ("backend_id", "source_id"):
        if field not in payload or payload[field] is None:
            continue
        _validate_catalog_id(payload[field], field)
        if payload[field] == "auto":
            raise MiniChemMcpProtocolError(f"{field}=auto 被 MiniChem 禁止")
    for field in ("component_backends", "inputs", "method_spec", "action_settings", "resource_limits"):
        if field not in payload or payload[field] is None:
            continue
        if not isinstance(payload[field], Mapping):
            raise MiniChemMcpProtocolError(f"execute_action.{field} 必须是 object")
    if "resource_limits" in payload and payload["resource_limits"] is not None:
        _validate_resource_limits_mapping(payload["resource_limits"], "execute_action.resource_limits")
    components = payload.get("component_backends")
    if isinstance(components, Mapping):
        for role, backend_id in components.items():
            _validate_catalog_id(role, "component_backends role")
            _validate_catalog_id(backend_id, "component_backends backend")
            if backend_id == "auto":
                raise MiniChemMcpProtocolError("component_backends 不接受 auto")
    return payload


def _validate_resource_limits_mapping(value: Any, label: str) -> dict[str, Any]:
    """核对显式单次请求资源字段，不把任务级预算变成请求默认值。"""

    if not isinstance(value, Mapping):
        raise MiniChemMcpProtocolError(f"{label} 必须是 object")
    unknown = set(value) - _RESOURCE_LIMIT_FIELDS
    if unknown:
        raise MiniChemMcpProtocolError(f"{label} 含未知字段：{sorted(unknown)}")
    checked = deepcopy(dict(value))
    limits = {
        "cpu_cores": (1, "正整数"),
        "memory_mb": (128, "不小于 128 的整数"),
        "gpu_count": (0, "非负整数"),
    }
    for field, (minimum, description) in limits.items():
        if field not in checked:
            continue
        candidate = checked[field]
        if type(candidate) is not int or candidate < minimum:
            raise MiniChemMcpProtocolError(f"{label}.{field} 必须是{description}")
    return checked


def _validate_evaluation_budget_record(value: Any, label: str) -> dict[str, Any]:
    """核对任务级资源政策，并移除旧服务返回的宿主容量数值。"""

    if not isinstance(value, Mapping):
        raise MiniChemMcpProtocolError(f"{label} 必须是 object")
    fields = frozenset(value)
    if fields not in {_PUBLIC_RESOURCE_POLICY_FIELDS, _RESOURCE_POLICY_FIELDS}:
        raise MiniChemMcpProtocolError(
            f"{label} 字段不符合公开资源政策合同"
        )
    checked = deepcopy(dict(value))
    if fields == _RESOURCE_POLICY_FIELDS:
        if (
            type(checked["cpu_cores"]) is not int
            or checked["cpu_cores"] < 1
            or type(checked["memory_mb"]) is not int
            or checked["memory_mb"] < 128
            or type(checked["gpu_count"]) is not int
            or checked["gpu_count"] < 0
            or checked["source"] != "evaluation_policy"
        ):
            raise MiniChemMcpProtocolError(f"{label} 不符合任务资源政策")
    elif checked["source"] != "service_policy":
        raise MiniChemMcpProtocolError(f"{label} 不符合任务资源政策")
    if checked["agent_controllable"] is not False or checked["scope"] != "per_task":
        raise MiniChemMcpProtocolError(f"{label} 不符合任务资源政策")
    return {
        "source": "service_policy",
        "agent_controllable": False,
        "scope": "per_task",
    }


def _validate_timeout_policy_record(value: Any, label: str) -> dict[str, Any]:
    """核对服务超时政策，并移除旧服务返回的具体时限。"""

    if not isinstance(value, Mapping):
        raise MiniChemMcpProtocolError(f"{label} 必须是 object")
    fields = frozenset(value)
    if fields not in {_PUBLIC_TIMEOUT_POLICY_FIELDS, _LEGACY_TIMEOUT_POLICY_FIELDS}:
        raise MiniChemMcpProtocolError(f"{label} 字段不符合公开超时政策合同")
    checked = deepcopy(dict(value))
    if (
        not isinstance(checked["execution_class"], str)
        or checked["execution_class"] not in {"compute", "fast"}
        or checked["agent_controllable"] is not False
    ):
        raise MiniChemMcpProtocolError(f"{label} 不符合执行超时政策")
    if fields == _LEGACY_TIMEOUT_POLICY_FIELDS:
        if (
            type(checked["timeout_seconds"]) is not int
            or checked["timeout_seconds"] < 1
            or checked["source"] != "evaluation_policy"
        ):
            raise MiniChemMcpProtocolError(f"{label} 不符合执行超时政策")
    elif checked["source"] != "service_policy":
        raise MiniChemMcpProtocolError(f"{label} 不符合执行超时政策")
    return {
        "execution_class": checked["execution_class"],
        "source": "service_policy",
        "agent_controllable": False,
    }


def _validate_open_output_path(value: Any, label: str) -> str:
    """校验 Analysis 产物只能发布到任务允许的输出目录。"""

    path = _validate_relative_artifact_path(value, label)
    if not path.startswith(_OPEN_OUTPUT_PREFIXES):
        raise MiniChemMcpProtocolError(
            f"{label} 必须位于 outputs/、report/ 或 logs/ 目录"
        )
    return path


def _validate_analysis_program_request(value: Mapping[str, Any]) -> None:
    """在调用 MCP 前核对 Analysis 输入和输出的最小安全边界。"""

    unknown = set(value) - _ANALYSIS_JOB_REQUEST_FIELDS
    if unknown:
        raise MiniChemMcpProtocolError(
            f"submit_analysis_program request 含真实 AnalysisJobRequest 未声明字段：{sorted(unknown)}"
        )

    inputs = value.get("inputs")
    if inputs is not None:
        if not isinstance(inputs, list):
            raise MiniChemMcpProtocolError("submit_analysis_program.inputs 必须是数组")
        for index, item in enumerate(inputs):
            label = f"submit_analysis_program.inputs[{index}]"
            if not isinstance(item, Mapping):
                raise MiniChemMcpProtocolError(f"{label} 必须是 object")
            source_path = item.get("source_path")
            if not isinstance(source_path, str) or not source_path.strip():
                raise MiniChemMcpProtocolError(
                    f"{label}.source_path 必须是非空任务内路径，不能为 null"
                )
            _validate_relative_artifact_path(source_path, f"{label}.source_path")
            target_path = item.get("target_path")
            if target_path is not None:
                _validate_relative_artifact_path(target_path, f"{label}.target_path")

    outputs = value.get("outputs")
    if outputs is not None:
        if not isinstance(outputs, list):
            raise MiniChemMcpProtocolError("submit_analysis_program.outputs 必须是数组")
        for index, item in enumerate(outputs):
            label = f"submit_analysis_program.outputs[{index}]"
            if not isinstance(item, Mapping):
                raise MiniChemMcpProtocolError(f"{label} 必须是 object")
            if "path" not in item:
                raise MiniChemMcpProtocolError(f"{label}.path 缺失")
            _validate_open_output_path(item["path"], f"{label}.path")


def _validate_contract_text(value: Any, label: str, *, max_chars: int = 100_000) -> str:
    """校验可放入冻结能力合同的说明或模板文本。"""

    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise MiniChemMcpProtocolError(f"{label} 必须是非空安全文本")
    if len(value) > max_chars:
        raise MiniChemMcpProtocolError(f"{label} 超过 {max_chars} 个字符")
    return value


def _project_analysis_program_template(value: Any) -> dict[str, Any]:
    """筛选并校验程序请求模板，不向 Planner 传递资源默认值。"""

    if not isinstance(value, Mapping):
        raise MiniChemMcpProtocolError(
            "submit_analysis_program_request_template 必须是 object"
        )
    required = {
        "runtime",
        "script_path",
        "script_target",
        "arguments",
        "staged_inputs",
        "inputs",
        "outputs",
        "label",
    }
    # 兼容尚未升级的服务：旧模板中的资源值仍需通过格式校验，但不会进入
    # 冻结目录或模型上下文。新服务不再返回该字段。
    allowed = required | {"resource_limits"}
    unknown = set(value) - allowed
    if unknown:
        raise MiniChemMcpProtocolError(
            "submit_analysis_program_request_template 含未知字段："
            f"{sorted(unknown)}"
        )
    missing = required - set(value)
    if missing:
        raise MiniChemMcpProtocolError(
            "submit_analysis_program_request_template 缺少字段："
            f"{sorted(missing)}"
        )
    runtime = _validate_contract_text(value["runtime"], "analysis template.runtime")
    script_path = _validate_code_workspace_path(
        value["script_path"], "analysis template.script_path"
    )
    script_target = _validate_code_workspace_path(
        value["script_target"], "analysis template.script_target"
    )
    arguments = value["arguments"]
    if not isinstance(arguments, list):
        raise MiniChemMcpProtocolError("analysis template.arguments 必须是数组")
    projected_arguments = [
        _validate_example_argument(item, f"analysis template.arguments[{index}]")
        for index, item in enumerate(arguments)
    ]
    staged_inputs = value["staged_inputs"]
    if not isinstance(staged_inputs, list):
        raise MiniChemMcpProtocolError("analysis template.staged_inputs 必须是数组")
    projected_staged: list[dict[str, str]] = []
    for index, item in enumerate(staged_inputs):
        if not isinstance(item, Mapping) or set(item) != {"source_path", "target_path"}:
            raise MiniChemMcpProtocolError(
                f"analysis template.staged_inputs[{index}] 字段不完整"
            )
        projected_staged.append(
            {
                "source_path": _validate_relative_artifact_path(
                    item["source_path"],
                    f"analysis template.staged_inputs[{index}].source_path",
                ),
                "target_path": _validate_relative_artifact_path(
                    item["target_path"],
                    f"analysis template.staged_inputs[{index}].target_path",
                ),
            }
        )
    inputs = value["inputs"]
    if not isinstance(inputs, list):
        raise MiniChemMcpProtocolError("analysis template.inputs 必须是数组")
    projected_inputs: list[dict[str, str]] = []
    for index, item in enumerate(inputs):
        if not isinstance(item, Mapping) or set(item) != {
            "name", "source_path", "target_path", "semantic_type"
        }:
            raise MiniChemMcpProtocolError(f"analysis template.inputs[{index}] 字段不完整")
        projected_inputs.append(
            {
                "name": _validate_open_text(
                    item["name"], f"analysis template.inputs[{index}].name"
                ),
                "source_path": _validate_relative_artifact_path(
                    item["source_path"],
                    f"analysis template.inputs[{index}].source_path",
                ),
                "target_path": _validate_relative_artifact_path(
                    item["target_path"],
                    f"analysis template.inputs[{index}].target_path",
                ),
                "semantic_type": _validate_open_text(
                    item["semantic_type"],
                    f"analysis template.inputs[{index}].semantic_type",
                ),
            }
        )
    outputs = value["outputs"]
    if not isinstance(outputs, list):
        raise MiniChemMcpProtocolError("analysis template.outputs 必须是数组")
    projected_outputs: list[dict[str, Any]] = []
    for index, item in enumerate(outputs):
        if not isinstance(item, Mapping) or set(item) != {
            "name", "path", "semantic_type", "media_type", "required"
        }:
            raise MiniChemMcpProtocolError(f"analysis template.outputs[{index}] 字段不完整")
        required_flag = item["required"]
        if type(required_flag) is not bool:
            raise MiniChemMcpProtocolError(
                f"analysis template.outputs[{index}].required 必须是 bool"
            )
        projected_outputs.append(
            {
                "name": _validate_open_text(
                    item["name"], f"analysis template.outputs[{index}].name"
                ),
                "path": _validate_open_output_path(
                    item["path"], f"analysis template.outputs[{index}].path"
                ),
                "semantic_type": _validate_open_text(
                    item["semantic_type"],
                    f"analysis template.outputs[{index}].semantic_type",
                ),
                "media_type": _validate_open_text(
                    item["media_type"],
                    f"analysis template.outputs[{index}].media_type",
                ),
                "required": required_flag,
            }
        )
    if "resource_limits" in value:
        _validate_resource_limits_mapping(
            value["resource_limits"], "analysis template.resource_limits"
        )
    return {
        "runtime": runtime,
        "script_path": script_path,
        "script_target": script_target,
        "arguments": projected_arguments,
        "staged_inputs": projected_staged,
        "inputs": projected_inputs,
        "outputs": projected_outputs,
        "label": _validate_open_text(value["label"], "analysis template.label"),
    }


def _project_managed_program_contract(value: Any) -> dict[str, Any]:
    """保留 JobContext 输入输出约束，移除运行时路径等内部字段。"""

    if not isinstance(value, Mapping):
        raise MiniChemMcpProtocolError("managed_program_contract 必须是 object")
    expected = {
        "job_context_required_for_compliance",
        "execution_note",
        "minimal_python_template",
        "helper_name_rule",
    }
    if set(value) != expected:
        raise MiniChemMcpProtocolError(
            f"managed_program_contract 字段不完整：需要 {sorted(expected)}"
        )
    if type(value["job_context_required_for_compliance"]) is not bool:
        raise MiniChemMcpProtocolError(
            "managed_program_contract.job_context_required_for_compliance 必须是 bool"
        )
    return {
        "job_context_required_for_compliance": value[
            "job_context_required_for_compliance"
        ],
        "execution_note": _validate_contract_text(
            value["execution_note"], "managed_program_contract.execution_note"
        ),
        "minimal_python_template": _validate_contract_text(
            value["minimal_python_template"],
            "managed_program_contract.minimal_python_template",
        ),
        "helper_name_rule": _validate_contract_text(
            value["helper_name_rule"], "managed_program_contract.helper_name_rule"
        ),
    }


def _validate_artifact_declaration_mapping(value: Any) -> dict[str, Any]:
    """核对 ArtifactDeclarationRequest 的六个公开字段并补齐模型默认值。"""

    if not isinstance(value, Mapping):
        raise MiniChemMcpProtocolError("declare_scientific_artifact request 必须是 object")
    unknown = set(value) - _ARTIFACT_DECLARATION_FIELDS
    if unknown:
        raise MiniChemMcpProtocolError(
            f"declare_scientific_artifact request 含未知字段：{sorted(unknown)}"
        )
    required = {"path", "semantic_type", "producer_layer", "producer_id"}
    missing = required - set(value)
    if missing:
        raise MiniChemMcpProtocolError(
            f"declare_scientific_artifact request 缺少字段：{sorted(missing)}"
        )
    payload = deepcopy(dict(value))
    payload.setdefault("media_type", "application/octet-stream")
    payload.setdefault("parent_artifact_ids", [])
    if not isinstance(payload["semantic_type"], str) or not payload["semantic_type"].strip():
        raise MiniChemMcpProtocolError("semantic_type 必须是非空字符串")
    if not isinstance(payload["media_type"], str) or not payload["media_type"].strip():
        raise MiniChemMcpProtocolError("media_type 必须是非空字符串")
    _validate_relative_artifact_path(payload["path"], "path")
    if payload["producer_layer"] not in {
        "native_software",
        "programmable_analysis",
        "external_input",
    }:
        raise MiniChemMcpProtocolError("producer_layer 不是 MiniChem 允许的值")
    if not isinstance(payload["producer_id"], str) or not payload["producer_id"].strip():
        raise MiniChemMcpProtocolError("producer_id 必须是非空字符串")
    parents = payload["parent_artifact_ids"]
    if not isinstance(parents, list) or any(
        not isinstance(item, str) or not item.strip() for item in parents
    ):
        raise MiniChemMcpProtocolError("parent_artifact_ids 必须是字符串数组")
    return payload


def _validate_relative_artifact_path(value: Any, label: str) -> str:
    """核对 ArtifactRef 路径是任务内相对 POSIX 路径。"""

    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise MiniChemMcpProtocolError(f"{label} 必须是非空 POSIX 路径")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
        raise MiniChemMcpProtocolError(f"{label} 必须是安全相对路径")
    return parsed.as_posix()


def _validate_code_workspace_path(value: Any, label: str = "workspace path") -> str:
    """只接受任务工作区内 ``code/`` 下的安全 POSIX 文件路径。"""

    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or "//" in value
        or value.endswith("/")
        or "\x00" in value
    ):
        raise MiniChemMcpProtocolError(f"{label} 必须是非空 POSIX 路径")
    parsed = PurePosixPath(value)
    if (
        parsed.is_absolute()
        or len(parsed.parts) < 2
        or parsed.parts[0] != "code"
        or any(part in {"", ".", ".."} for part in parsed.parts)
    ):
        raise MiniChemMcpProtocolError(f"{label} 必须是 code/ 下的安全相对路径")
    return parsed.as_posix()


def _require_open_execution_success(result: Mapping[str, Any], tool_name: str) -> None:
    """拒绝 Open Execution discovery/写入工具的错误 envelope。"""

    if result.get("status") != "success":
        raise MiniChemMcpProtocolError(f"{tool_name} 失败：{result}")


def _validate_open_text(value: Any, label: str) -> str:
    """核对公开 discovery 文本为单行非空字符串。"""

    if not isinstance(value, str) or not value.strip() or "\r" in value or "\n" in value:
        raise MiniChemMcpProtocolError(f"{label} 必须是单行非空字符串")
    return value


def _validate_open_description(value: Any, label: str) -> str:
    """校验公开说明文本，并阻止把绝对本机路径传播给 Planner。"""

    text = _validate_open_text(value, label)
    if _SAFE_ARGUMENT_PATH.match(text.strip()):
        raise MiniChemMcpProtocolError(f"{label} 不得包含绝对路径")
    return text


def _validate_example_argument(value: Any, label: str) -> str:
    """核对示例参数不会携带绝对路径、NUL 或超长内容。"""

    if not isinstance(value, str) or "\x00" in value or len(value) > 8192:
        raise MiniChemMcpProtocolError(f"{label} 必须是安全字符串")
    if _SAFE_ARGUMENT_PATH.match(value):
        raise MiniChemMcpProtocolError(f"{label} 不得包含绝对路径")
    return value


def _project_native_job_example(
    value: Any,
    *,
    software_id: str,
    executable: str,
) -> dict[str, Any]:
    """从服务模板筛出 NativeJobRequest 公开字段，不传播运行时内部信息。"""

    if not isinstance(value, Mapping):
        raise MiniChemMcpProtocolError("native_job_request_template 必须是 object")
    for identity_field, expected in (
        ("software_id", software_id),
        ("executable", executable),
    ):
        supplied = value.get(identity_field)
        if supplied is not None and supplied != expected:
            raise MiniChemMcpProtocolError(
                f"example_request.{identity_field} 与 inspect 身份不一致"
            )
    projected: dict[str, Any] = {
        "software_id": software_id,
        "executable": executable,
    }
    for field in sorted(_NATIVE_JOB_EXAMPLE_FIELDS - {"software_id", "executable"}):
        if field not in value:
            continue
        candidate = value[field]
        if field == "arguments":
            if not isinstance(candidate, list):
                raise MiniChemMcpProtocolError("example_request.arguments 必须是数组")
            projected[field] = [
                _validate_example_argument(item, f"example_request.arguments[{index}]")
                for index, item in enumerate(candidate)
            ]
        elif field == "staged_inputs":
            if not isinstance(candidate, list):
                raise MiniChemMcpProtocolError("example_request.staged_inputs 必须是数组")
            staged: list[dict[str, str]] = []
            for index, item in enumerate(candidate):
                if not isinstance(item, Mapping) or set(item) != {"source_path", "target_path"}:
                    raise MiniChemMcpProtocolError(
                        f"example_request.staged_inputs[{index}] 必须只包含 source_path/target_path"
                    )
                source_path = item.get("source_path")
                target_path = item.get("target_path")
                _validate_relative_artifact_path(
                    source_path,
                    f"example_request.staged_inputs[{index}].source_path",
                )
                _validate_relative_artifact_path(
                    target_path,
                    f"example_request.staged_inputs[{index}].target_path",
                )
                staged.append(
                    {
                        "source_path": str(source_path),
                        "target_path": str(target_path),
                    }
                )
            projected[field] = staged
        elif field == "resource_limits":
            if not isinstance(candidate, Mapping):
                raise MiniChemMcpProtocolError("example_request.resource_limits 必须是 object")
            checked_limits = _validate_resource_limits_mapping(
                candidate,
                "example_request.resource_limits",
            )
            projected[field] = checked_limits
        elif field == "resource_policy":
            if software_id != "gaussian" or not isinstance(candidate, Mapping):
                raise MiniChemMcpProtocolError(
                    "example_request.resource_policy 仅允许 Gaussian 且必须是 object"
                )
            expected_policy = {
                "mode": "service_managed_full_capacity",
                "concurrency": "serial",
                "agent_controllable": False,
            }
            if dict(candidate) != expected_policy:
                raise MiniChemMcpProtocolError(
                    "example_request.resource_policy 必须是服务管理的串行策略"
                )
            projected[field] = expected_policy
        elif field in {"stdin_target"}:
            if candidate is not None:
                _validate_relative_artifact_path(candidate, f"example_request.{field}")
            projected[field] = candidate
        elif field == "parent_job_id":
            if candidate is not None and (
                not isinstance(candidate, str) or _JOB_ID.fullmatch(candidate) is None
            ):
                raise MiniChemMcpProtocolError("example_request.parent_job_id 不是合法 job_id")
            projected[field] = candidate
        elif field in {"calculation_intent", "label"}:
            if candidate is not None:
                candidate = _validate_open_text(candidate, f"example_request.{field}")
            projected[field] = candidate
        else:
            raise MiniChemMcpProtocolError(f"example_request 含有未处理字段：{field}")
    return projected


def _workspace_file_missing(result: Mapping[str, Any]) -> bool:
    """识别服务明确报告的文件不存在，避免把其他错误当作可创建。"""

    if result.get("status") not in {"invalid_request", "failed"}:
        return False
    error = result.get("error")
    if not isinstance(error, Mapping):
        return False
    exception_type = str(error.get("exception_type", ""))
    code = str(error.get("code", ""))
    message = str(error.get("message", "")).casefold()
    return (
        exception_type == "FileNotFoundError"
        or code in {"workspace_file_not_found", "file_not_found", "missing"}
        or "not found" in message
        or "不存在" in message
    )


def _workspace_write_race(result: Mapping[str, Any]) -> bool:
    """识别 ``overwrite=false`` 因并发写入而失败的服务响应。"""

    if result.get("status") != "invalid_request":
        return False
    error = result.get("error")
    if not isinstance(error, Mapping):
        return False
    exception_type = str(error.get("exception_type", ""))
    code = str(error.get("code", ""))
    message = str(error.get("message", "")).casefold()
    return (
        exception_type == "FileExistsError"
        or code in {"workspace_file_exists", "file_exists", "already_exists"}
        or "already exists" in message
        or "已存在" in message
    )


def _validate_artifact_ref(
    value: Any,
    *,
    label: str,
    action_id: str | None = None,
    backend_id: str | None = None,
) -> dict[str, Any]:
    """验证 ArtifactRef 的身份、内容摘要、类型和生产来源。"""

    if not isinstance(value, Mapping):
        raise MiniChemMcpProtocolError(f"{label} 必须是 object")
    required = {
        "artifact_id",
        "semantic_type",
        "media_type",
        "sha256",
        "path",
        "producer_action",
        "producer_backend",
        "parent_artifact_ids",
    }
    missing = required - set(value)
    if missing:
        raise MiniChemMcpProtocolError(f"{label} 缺少字段：{sorted(missing)}")
    unknown = set(value) - required
    if unknown:
        raise MiniChemMcpProtocolError(f"{label} 含未知字段：{sorted(unknown)}")
    result = deepcopy(dict(value))
    for field in ("artifact_id", "semantic_type", "media_type", "producer_action"):
        if not isinstance(result[field], str) or not result[field].strip():
            raise MiniChemMcpProtocolError(f"{label}.{field} 必须是非空字符串")
    if not isinstance(result["sha256"], str) or _SHA256.fullmatch(result["sha256"]) is None:
        raise MiniChemMcpProtocolError(f"{label}.sha256 不是小写 SHA-256")
    result["path"] = _validate_relative_artifact_path(result["path"], f"{label}.path")
    producer_backend = result["producer_backend"]
    if producer_backend is not None and (
        not isinstance(producer_backend, str) or not producer_backend.strip()
    ):
        raise MiniChemMcpProtocolError(f"{label}.producer_backend 必须是字符串或 null")
    parents = result["parent_artifact_ids"]
    if not isinstance(parents, list) or any(
        not isinstance(item, str) or not item.strip() for item in parents
    ):
        raise MiniChemMcpProtocolError(f"{label}.parent_artifact_ids 必须是字符串数组")
    if action_id is not None and result["producer_action"] != action_id:
        raise MiniChemMcpProtocolError(
            f"{label}.producer_action 与请求 Action 不一致：{result['producer_action']}"
        )
    if backend_id is not None and producer_backend not in {None, backend_id}:
        raise MiniChemMcpProtocolError(
            f"{label}.producer_backend 与请求 Backend 不一致：{producer_backend}"
        )
    return result


def _validate_artifact_records(
    value: Mapping[str, Any],
    *,
    action_id: str | None = None,
    backend_id: str | None = None,
) -> dict[str, Any]:
    """核对响应中所有 ArtifactRef 和可选来源证明。"""

    result = deepcopy(dict(value))
    if "artifact" in result:
        result["artifact"] = _validate_artifact_ref(
            result["artifact"],
            label="artifact",
            action_id=action_id,
            backend_id=backend_id,
        )
    artifact_refs: list[dict[str, Any]] = []
    if "artifact" in result:
        artifact_refs.append(result["artifact"])
    for field in ("input_artifacts", "output_artifacts"):
        if field not in result:
            continue
        artifacts = result[field]
        # ActionResult 失败时可以没有任何产物；空数组是合法的失败结果，
        # 不能因为没有 BackendDiagnostic 就把协议错误冒充成结构错误。
        if not isinstance(artifacts, list):
            raise MiniChemMcpProtocolError(f"{field} 必须是数组")
        checked = [
            _validate_artifact_ref(
                item,
                label=f"{field}[{index}]",
                # Input artifacts are immutable outputs of an earlier Action and
                # therefore must retain that producer identity. Only artifacts
                # produced by the current response are checked against the
                # current Action/Backend.
                action_id=action_id if field == "output_artifacts" else None,
                backend_id=backend_id if field == "output_artifacts" else None,
            )
            for index, item in enumerate(artifacts)
        ]
        result[field] = checked
        artifact_refs.extend(checked)
    seen_ids: set[str] = set()
    duplicate_ids: set[str] = set()
    for artifact in artifact_refs:
        artifact_id = artifact["artifact_id"]
        if artifact_id in seen_ids:
            duplicate_ids.add(artifact_id)
        seen_ids.add(artifact_id)
    if duplicate_ids:
        raise MiniChemMcpProtocolError(
            "响应中的 ArtifactRef artifact_id 重复：" + ", ".join(sorted(duplicate_ids))
        )
    # collect 的 declared_artifacts/artifact_manifest 是声明记录（路径、producer、
    # parent_artifacts 等），不是 ArtifactRef；这里只保留原样，不能套用输出引用
    # 的 artifact_id/sha256/producer_action 必填校验。
    provenance = result.get("provenance")
    if provenance is not None:
        if not isinstance(provenance, Mapping):
            raise MiniChemMcpProtocolError("provenance 必须是 object")
        provenance_value = deepcopy(dict(provenance))
        catalog_hash = provenance_value.get("catalog_hash")
        if catalog_hash is not None and (
            not isinstance(catalog_hash, str) or _SHA256.fullmatch(catalog_hash) is None
        ):
            raise MiniChemMcpProtocolError("provenance.catalog_hash 不是 SHA-256")
        result["provenance"] = provenance_value
    return result


def _validate_job_file_records(value: Mapping[str, Any]) -> dict[str, Any]:
    """核对 collect_execution_job 返回的文件路径、类型和内容摘要。"""

    result = deepcopy(dict(value))
    files = result.get("files")
    if files is not None:
        if not isinstance(files, list):
            raise MiniChemMcpProtocolError("job.files 必须是数组")
        checked: list[dict[str, Any]] = []
        for index, item in enumerate(files):
            if not isinstance(item, Mapping):
                raise MiniChemMcpProtocolError(f"job.files[{index}] 必须是 object")
            path = _validate_relative_artifact_path(item.get("path"), f"job.files[{index}].path")
            digest = item.get("sha256")
            if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
                raise MiniChemMcpProtocolError(f"job.files[{index}].sha256 不是 SHA-256")
            media_type = item.get("media_type")
            if not isinstance(media_type, str) or not media_type.strip():
                raise MiniChemMcpProtocolError(f"job.files[{index}].media_type 非法")
            checked_item = deepcopy(dict(item))
            checked_item.update({"path": path, "sha256": digest, "media_type": media_type})
            checked.append(checked_item)
        result["files"] = checked
    outputs = result.get("outputs")
    if outputs is not None:
        if not isinstance(outputs, list):
            raise MiniChemMcpProtocolError("job.outputs 必须是数组")
        checked_outputs: list[dict[str, Any]] = []
        for index, item in enumerate(outputs):
            if not isinstance(item, Mapping):
                raise MiniChemMcpProtocolError(f"job.outputs[{index}] 必须是 object")
            output = deepcopy(dict(item))
            raw_path = output.get("path", output.get("job_relative_path"))
            output["path"] = _validate_relative_artifact_path(
                raw_path,
                f"job.outputs[{index}].path",
            )
            digest = output.get("sha256")
            if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
                raise MiniChemMcpProtocolError(f"job.outputs[{index}].sha256 不是 SHA-256")
            size_bytes = output.get("size_bytes")
            if not isinstance(size_bytes, int) or size_bytes < 0:
                raise MiniChemMcpProtocolError(f"job.outputs[{index}].size_bytes 必须是非负整数")
            kind = output.get("kind")
            if not isinstance(kind, str) or not kind.strip():
                raise MiniChemMcpProtocolError(f"job.outputs[{index}].kind 非法")
            checked_outputs.append(output)
        result["outputs"] = checked_outputs
    return result


def _validate_workspace_text_read_result(
    result: Mapping[str, Any],
    *,
    path: str,
    max_chars: int,
    offset: int,
    json_pointer: str | None,
) -> dict[str, Any]:
    """核对工作区文本分页响应，拒绝串页、截断元数据和哈希格式错误。"""

    value = deepcopy(dict(result))
    if value.get("path") != path:
        raise MiniChemMcpProtocolError("read_workspace_text 返回路径与请求不一致")
    returned_hash = value.get("sha256")
    if not isinstance(returned_hash, str) or _SHA256.fullmatch(returned_hash) is None:
        raise MiniChemMcpProtocolError("read_workspace_text.sha256 不是 SHA-256")
    if value.get("encoding") != "utf-8":
        raise MiniChemMcpProtocolError("read_workspace_text.encoding 必须是 utf-8")
    size_bytes = value.get("size_bytes")
    if type(size_bytes) is not int or size_bytes < 0:
        raise MiniChemMcpProtocolError("read_workspace_text.size_bytes 必须是非负整数")
    returned_offset = value.get("offset")
    if type(returned_offset) is not int or returned_offset < 0:
        raise MiniChemMcpProtocolError("read_workspace_text.offset 必须是非负整数")
    if returned_offset != offset:
        raise MiniChemMcpProtocolError(
            "read_workspace_text 返回了错误的 offset："
            f"expected={offset} actual={returned_offset}"
        )
    returned_chars = value.get("returned_chars")
    if type(returned_chars) is not int or returned_chars < 0:
        raise MiniChemMcpProtocolError(
            "read_workspace_text.returned_chars 必须是非负整数"
        )
    total_chars = value.get("total_chars")
    if type(total_chars) is not int or total_chars < 0:
        raise MiniChemMcpProtocolError("read_workspace_text.total_chars 必须是非负整数")
    content = value.get("content")
    if not isinstance(content, str):
        raise MiniChemMcpProtocolError("read_workspace_text.content 必须是字符串")
    if len(content) != returned_chars:
        raise MiniChemMcpProtocolError(
            "read_workspace_text.returned_chars 与 content 字符数不一致"
        )
    available_chars = max(total_chars - offset, 0)
    if returned_chars > available_chars:
        raise MiniChemMcpProtocolError(
            "read_workspace_text.returned_chars 超过 total_chars 可读取范围"
        )
    next_offset = value.get("next_offset")
    if next_offset is not None:
        if type(next_offset) is not int or next_offset <= offset:
            raise MiniChemMcpProtocolError(
                "read_workspace_text.next_offset 必须单调递增或为 null"
            )
        expected_next_offset = offset + returned_chars
        if next_offset != expected_next_offset:
            raise MiniChemMcpProtocolError(
                "read_workspace_text.next_offset 与 returned_chars 不一致"
            )
        if next_offset >= total_chars:
            raise MiniChemMcpProtocolError(
                "read_workspace_text.next_offset 不能超出 total_chars"
            )
    truncated = value.get("truncated")
    if type(truncated) is not bool:
        raise MiniChemMcpProtocolError("read_workspace_text.truncated 必须是布尔值")
    if truncated is not (next_offset is not None):
        raise MiniChemMcpProtocolError(
            "read_workspace_text.truncated 与 next_offset 不一致"
        )
    if next_offset is None and offset + returned_chars < total_chars:
        raise MiniChemMcpProtocolError(
            "read_workspace_text 缺少未完成分页的 next_offset"
        )
    returned_pointer = value.get("json_pointer")
    if returned_pointer != json_pointer:
        raise MiniChemMcpProtocolError(
            "read_workspace_text 返回了错误的 json_pointer："
            f"expected={json_pointer!r} actual={returned_pointer!r}"
        )
    if max_chars < returned_chars:
        raise MiniChemMcpProtocolError(
            "read_workspace_text.returned_chars 超过请求的 max_chars"
        )
    return value


def _job_request_fingerprint(tool_name: str, request: Mapping[str, Any]) -> str:
    """为本地 operation 收据计算不含内部身份字段的规范请求指纹。"""

    payload = deepcopy(dict(request))
    payload.pop("operation_id", None)
    payload.pop("submission_key", None)
    # caller_request_fingerprint 是 adapter 发送给 MiniChem 的隐藏身份，不属于
    # 用户科学请求；保存 receipt 后重算指纹必须继续得到同一值。
    payload.pop(_CALLER_REQUEST_FINGERPRINT_FIELD, None)
    return hashlib.sha256(
        _canonical_json({"tool": tool_name, "request": payload}).encode("utf-8")
    ).hexdigest()


def _derive_minichem_operation_job_id(operation_id: str) -> str:
    """复用 MiniChem 的 operation→job 协议算法，无法复用时拒绝探测。

    当前 MiniChem Toolbox 的 open-execution 协议实现提供了可复用的身份映射函数。
    适配器只借用这段协议映射，不读取服务端目录或状态文件；若部署环境没有同一
    协议函数，继续猜测 ``job_id`` 会把查询发往错误作业，因此必须 fail closed。
    """

    try:
        from minichem_mcp_tools.open_execution import _operation_job_id
    except Exception as exc:
        raise MiniChemMcpProtocolError(
            "MiniChem 未提供可复用的 operation→job 身份算法，拒绝恢复探测"
        ) from exc
    try:
        derived = _operation_job_id(operation_id)
    except Exception as exc:
        raise MiniChemMcpProtocolError(
            "MiniChem operation→job 身份算法执行失败，拒绝恢复探测"
        ) from exc
    try:
        return _validate_job_id(derived)
    except ValueError as exc:
        raise MiniChemMcpProtocolError(
            "MiniChem operation→job 身份算法返回非法 job_id，拒绝恢复探测"
        ) from exc


def _resolve_operation_root(value: str | os.PathLike[str] | None) -> Path | None:
    """解析 operation 收据目录，并拒绝目录自身或既有入口为符号链接。"""

    if value is None:
        return None
    raw = Path(value)
    raw_absolute = raw if raw.is_absolute() else Path.cwd() / raw
    current = Path(raw_absolute.anchor)
    for component in raw_absolute.parts[1:]:
        current = current / component
        if current.is_symlink():
            raise ValueError("operation_root 路径不得经过符号链接")
    root = raw.resolve(strict=False)
    if root.exists() and not root.is_dir():
        raise ValueError("operation_root 必须是普通目录")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _open_operation_directory(operation_root: Path) -> int:
    """以不跟随符号链接的方式打开 operation 父目录并校验其类型。"""

    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if nofollow == 0:
        raise MiniChemMcpError("当前平台不支持安全打开 operation 目录所需的 O_NOFOLLOW")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= nofollow | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(operation_root, flags)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise MiniChemMcpError(
                f"operation 收据目录不得是符号链接：{operation_root}"
            ) from exc
        raise MiniChemMcpError(f"operation 收据目录不可打开：{operation_root}") from exc
    try:
        directory_stat = os.fstat(descriptor)
    except OSError as exc:
        os.close(descriptor)
        raise MiniChemMcpError(
            f"operation 收据目录无法校验：{operation_root}"
        ) from exc
    if not stat.S_ISDIR(directory_stat.st_mode):
        os.close(descriptor)
        raise MiniChemMcpError(f"operation 收据目录必须是目录：{operation_root}")
    return descriptor


def _open_operation_regular_file(
    directory_fd: int,
    name: str,
    *,
    flags: int,
    mode: int,
    label: str,
    missing_is_none: bool = False,
) -> int | None:
    """相对父目录安全打开文件，并确认最终打开的对象是普通文件。"""

    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if nofollow == 0:
        raise MiniChemMcpError(f"{label}无法安全打开：当前平台不支持 O_NOFOLLOW")
    secure_flags = flags | nofollow | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(name, secure_flags, mode, dir_fd=directory_fd)
    except OSError as exc:
        if exc.errno == errno.ENOENT and missing_is_none:
            return None
        if exc.errno == errno.ELOOP:
            raise MiniChemMcpError(f"{label}不得是符号链接：{name}") from exc
        if exc.errno in {errno.EISDIR, errno.ENOTDIR}:
            raise MiniChemMcpError(f"{label}必须是普通文件：{name}") from exc
        raise MiniChemMcpError(f"{label}不可打开：{name}") from exc
    try:
        file_stat = os.fstat(descriptor)
    except OSError as exc:
        os.close(descriptor)
        raise MiniChemMcpError(f"{label}无法校验：{name}") from exc
    if not stat.S_ISREG(file_stat.st_mode):
        os.close(descriptor)
        raise MiniChemMcpError(f"{label}必须是普通文件：{name}")
    return descriptor


class MiniChemMcpAdapter:
    """集中封装 MiniChem MCP discovery、Action、job 和 artifact 协议。"""

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        session_factory: _SessionFactory | None = None,
        operation_root: str | os.PathLike[str] | None = None,
        poll_interval: float = 0.0,
        retry_count: int = DEFAULT_API_RETRY_COUNT,
        retry_delay: float = DEFAULT_API_RETRY_DELAY_SECONDS,
        retry_sleeper: Any = None,
    ) -> None:
        value = endpoint or os.environ.get("MINICHEM_MCP_ENDPOINT", "http://127.0.0.1:9010/mcp")
        if not isinstance(value, str) or not value.startswith(("http://", "https://")):
            raise ValueError("MiniChem MCP endpoint 必须是 HTTP(S) URL")
        if poll_interval < 0:
            raise ValueError("poll_interval 必须是非负数")
        if isinstance(retry_count, bool) or not isinstance(retry_count, int) or retry_count < 0:
            raise ValueError("retry_count 必须是非负整数")
        if (
            isinstance(retry_delay, bool)
            or not isinstance(retry_delay, (int, float))
            or not math.isfinite(float(retry_delay))
            or retry_delay < 0
        ):
            raise ValueError("retry_delay 必须是非负数字")
        if retry_sleeper is not None and not callable(retry_sleeper):
            raise TypeError("retry_sleeper 必须可调用")
        self.endpoint = value
        self.operation_root = _resolve_operation_root(operation_root)
        self.poll_interval = float(poll_interval)
        self.retry_count = retry_count
        self.retry_delay = float(retry_delay)
        self._retry_sleeper = retry_sleeper or time.sleep
        self._bridge = _AsyncSessionBridge(value, session_factory)
        self._catalog: MiniChemCatalogSnapshot | None = None
        self._operations: dict[str, dict[str, Any]] = {}
        self._submission_guard = threading.RLock()
        self._workspace_read_guard = threading.RLock()
        self._workspace_read_hashes: dict[tuple[str, str | None], str] = {}

    def __enter__(self) -> "MiniChemMcpAdapter":
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def open(self) -> None:
        """建立长生命周期 MCP session，并重试连接/初始化瞬时失败。"""

        retry_api_call(
            self._bridge.start,
            retry_count=self.retry_count,
            retry_delay=self.retry_delay,
            sleeper=self._retry_sleeper,
        )

    def close(self) -> None:
        """关闭 MCP session，并保留已冻结目录与 operation 身份。"""

        self._bridge.close()

    def _call_tool(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        retry_safe: bool = True,
    ) -> dict[str, Any]:
        """调用一个 MCP 工具并解析响应；无幂等身份的副作用请求禁止重放。"""

        # MiniChem open_tools/discovery_tools 的公开函数都只接收一个名为
        # ``request`` 的 Pydantic 对象；MCP tools/call 的 arguments 必须保留
        # 这层字段，不能把请求字段平铺到 envelope 顶层。
        def call_remote() -> Any:
            return self._bridge.call(
                "call_tool", name, {"request": dict(arguments or {})}
            )

        try:
            raw = retry_api_call(
                call_remote,
                retry_count=self.retry_count if retry_safe else 0,
                retry_delay=self.retry_delay,
                sleeper=self._retry_sleeper,
                retryable_result=lambda value: is_retryable_http_status(
                    response_http_status(value)
                ),
            )
        except Exception as exc:
            if not retry_safe and is_retryable_api_exception(exc):
                raise MiniChemMcpError(
                    f"MCP 工具 {name} 传输失败；该请求没有 operation/idempotency 身份，"
                    "已阻止不安全重放"
                ) from exc
            raise
        if not retry_safe and is_retryable_http_status(response_http_status(raw)):
            raise MiniChemMcpError(
                f"MCP 工具 {name} 返回瞬时 HTTP 错误；该请求没有 operation/idempotency 身份，"
                "已阻止不安全重放"
            )
        value = _content_value(raw)
        if not isinstance(value, Mapping):
            raise MiniChemMcpProtocolError(f"MCP 工具 {name} 返回值不是 object")
        return deepcopy(dict(value))

    def _validate_discovery_catalog_hash(
        self, result: Mapping[str, Any], tool_name: str
    ) -> dict[str, Any]:
        """核对 discovery 响应与当前冻结目录使用同一 hash/version。

        MiniChem discovery 工具的响应必须带服务端 ``catalog_hash`` 和
        ``catalog_version``。适配器已经冻结目录后，缺失或漂移的身份都立即失败，
        不能用旧合同或另一份目录继续规划/执行。未冻结目录时保留读取入口的兼容
        性，但服务生产响应仍由 MiniChem discovery 层统一补齐这两个字段。
        """

        value = deepcopy(dict(result))
        catalog = self._catalog
        returned_hash = value.get("catalog_hash")
        returned_version = value.get("catalog_version")
        if catalog is not None:
            if returned_hash != catalog.catalog_hash:
                raise MiniChemMcpProtocolError(
                    f"MiniChem {tool_name} 返回的 catalog_hash 与冻结目录不一致："
                    f"expected={catalog.catalog_hash} actual={returned_hash!r}"
                )
            if (
                isinstance(returned_version, bool)
                or not isinstance(returned_version, str)
                or not returned_version.strip()
            ):
                raise MiniChemMcpProtocolError(
                    f"MiniChem {tool_name} 缺少有效 catalog_version"
                )
            if returned_version != catalog.catalog_version:
                raise MiniChemMcpProtocolError(
                    f"MiniChem {tool_name} 返回的 catalog_version 与冻结目录不一致："
                    f"expected={catalog.catalog_version} actual={returned_version!r}"
                )
        return value

    def list_tools(self) -> dict[str, Any]:
        """读取当前 session 的 MCP 工具清单，不把清单当作 Action 定义。"""

        raw = retry_api_call(
            lambda: self._bridge.call("list_tools"),
            retry_count=self.retry_count,
            retry_delay=self.retry_delay,
            sleeper=self._retry_sleeper,
            retryable_result=lambda value: is_retryable_http_status(
                response_http_status(value)
            ),
        )
        value = _content_value(raw)
        return value if isinstance(value, Mapping) else {"tools": value}

    def list_action_domains(self, *, include_action_ids: bool = True) -> dict[str, Any]:
        """读取完整 Action 领域索引，不按任务过滤或截断 Action 身份。"""

        if type(include_action_ids) is not bool:
            raise TypeError("include_action_ids 必须是 bool")
        result = self._validate_discovery_catalog_hash(
            self._call_tool(
                "list_action_domains",
                {"include_action_ids": include_action_ids},
            ),
            "list_action_domains",
        )
        if result.get("status") != "success" or not isinstance(
            result.get("domains"), list
        ):
            raise MiniChemMcpProtocolError(f"list_action_domains 失败：{result}")
        if include_action_ids and self._catalog is not None:
            declared: set[str] = set()
            for index, domain in enumerate(result["domains"]):
                if not isinstance(domain, Mapping):
                    raise MiniChemMcpProtocolError(
                        f"list_action_domains.domains[{index}] 必须是 object"
                    )
                action_ids = domain.get("action_ids")
                if not isinstance(action_ids, list) or any(
                    not isinstance(action_id, str) for action_id in action_ids
                ):
                    raise MiniChemMcpProtocolError(
                        f"list_action_domains.domains[{index}].action_ids 必须是字符串数组"
                    )
                declared.update(action_ids)
            expected = {str(action.get("id")) for action in self._catalog.actions}
            if declared != expected:
                raise MiniChemMcpProtocolError(
                    "list_action_domains 未覆盖冻结目录中的全部 Action："
                    f"expected={len(expected)} actual={len(declared)}"
                )
        return result

    def browse_action_category(
        self,
        category: str,
        *,
        action_kind: str = "all",
        available_only: bool = False,
        detail_level: str = "summary",
    ) -> dict[str, Any]:
        """读取一个明确领域的完整 Action 摘要，不自动换领域或 Backend。"""

        if not isinstance(category, str) or _CATALOG_ID.fullmatch(category) is None:
            raise ValueError("category 必须是 lower_snake_case 标识")
        if action_kind not in {"all", "scientific", "data"}:
            raise ValueError("action_kind 必须是 all、scientific 或 data")
        if type(available_only) is not bool:
            raise TypeError("available_only 必须是 bool")
        if detail_level not in {"summary", "full"}:
            raise ValueError("detail_level 必须是 summary 或 full")
        result = self._validate_discovery_catalog_hash(
            self._call_tool(
                "browse_action_category",
                {
                    "category": category,
                    "action_kind": action_kind,
                    "available_only": available_only,
                    "detail_level": detail_level,
                },
            ),
            "browse_action_category",
        )
        if result.get("status") != "success" or not isinstance(
            result.get("actions"), list
        ):
            raise MiniChemMcpProtocolError(f"browse_action_category 失败：{result}")
        return result

    def search_actions(
        self,
        query: str | None = None,
        *,
        category: str | None = None,
        backend_id: str | None = None,
        action_kind: str = "all",
        retrieval_mode: str = "lexical",
        detail_level: str = "summary",
        available_only: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """按明确条件检索 Action 摘要，返回结果中的每个候选仍可继续 inspect。"""

        if query is not None and not isinstance(query, str):
            raise TypeError("query 必须是字符串或 null")
        for value, label in ((category, "category"), (backend_id, "backend_id")):
            if value is not None and (
                not isinstance(value, str) or _CATALOG_ID.fullmatch(value) is None
            ):
                raise ValueError(f"{label} 必须是 lower_snake_case 标识或 null")
        if action_kind not in {"all", "scientific", "data"}:
            raise ValueError("action_kind 必须是 all、scientific 或 data")
        if retrieval_mode not in {"lexical", "hybrid"}:
            raise ValueError("retrieval_mode 必须是 lexical 或 hybrid")
        if detail_level not in {"summary", "full"}:
            raise ValueError("detail_level 必须是 summary 或 full")
        if type(available_only) is not bool:
            raise TypeError("available_only 必须是 bool")
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("limit 必须在 1 到 100 之间")
        if type(offset) is not int or offset < 0:
            raise ValueError("offset 必须是非负整数")
        request = {
            "query": query,
            "category": category,
            "backend_id": backend_id,
            "action_kind": action_kind,
            "retrieval_mode": retrieval_mode,
            "detail_level": detail_level,
            "available_only": available_only,
            "limit": limit,
            "offset": offset,
        }
        result = self._validate_discovery_catalog_hash(
            self._call_tool("search_actions", request), "search_actions"
        )
        if result.get("status") != "success" or not isinstance(
            result.get("actions"), list
        ):
            raise MiniChemMcpProtocolError(f"search_actions 失败：{result}")
        return result

    def search_all_actions(
        self,
        query: str | None = None,
        *,
        category: str | None = None,
        backend_id: str | None = None,
        action_kind: str = "all",
        retrieval_mode: str = "lexical",
        detail_level: str = "summary",
        available_only: bool = False,
    ) -> dict[str, Any]:
        """分页汇总所有匹配 Action，避免服务端单页上限隐藏目录项。"""

        offset = 0
        pages: list[dict[str, Any]] = []
        while True:
            page = self.search_actions(
                query,
                category=category,
                backend_id=backend_id,
                action_kind=action_kind,
                retrieval_mode=retrieval_mode,
                detail_level=detail_level,
                available_only=available_only,
                limit=100,
                offset=offset,
            )
            pages.append(page)
            actions = page.get("actions")
            if not isinstance(actions, list):
                raise MiniChemMcpProtocolError("search_actions.actions 必须是数组")
            next_offset = page.get("next_offset")
            if next_offset is None:
                break
            if type(next_offset) is not int or next_offset <= offset:
                raise MiniChemMcpProtocolError(
                    "search_actions.next_offset 必须单调递增或为 null"
                )
            offset = next_offset
        if not pages:
            raise MiniChemMcpProtocolError("search_actions 未返回分页结果")
        merged = deepcopy(pages[0])
        merged["actions"] = [
            deepcopy(item)
            for page in pages
            for item in page.get("actions", [])
        ]
        merged["offset"] = 0
        merged["count"] = len(merged["actions"])
        merged["total_matches"] = pages[-1].get("total_matches", len(merged["actions"]))
        merged["next_offset"] = None
        merged["page_count"] = len(pages)
        return merged

    def discover_catalog(self, *, refresh: bool = False) -> MiniChemCatalogSnapshot:
        """读取服务目录并冻结 hash/version；同一 adapter 默认复用首次快照。"""

        if self._catalog is not None and not refresh:
            return self._catalog
        try:
            raw = retry_api_call(
                lambda: self._bridge.call("read_resource", "minichem://catalog"),
                retry_count=self.retry_count,
                retry_delay=self.retry_delay,
                sleeper=self._retry_sleeper,
                retryable_result=lambda value: is_retryable_http_status(
                    response_http_status(value)
                ),
            )
            value = _content_value(raw)
            if isinstance(value, str):
                value = json.loads(value)
        except (MiniChemMcpError, MiniChemMcpProtocolError, json.JSONDecodeError) as exc:
            raise MiniChemMcpProtocolError(
                "MiniChem canonical catalog 资源 minichem://catalog 读取失败；"
                "不能用 list/search 局部结果替代完整目录快照"
            ) from exc
        if not isinstance(value, Mapping):
            raise MiniChemMcpProtocolError("MiniChem catalog 不是 object")
        snapshot = deepcopy(dict(value))
        if snapshot.get("status") == "mcp_error":
            raise MiniChemMcpProtocolError(
                f"MiniChem canonical catalog 资源返回错误：{snapshot.get('error')!r}"
            )
        if snapshot.get("catalog_visibility_policy") == "public_contract_projection":
            snapshot = validate_public_catalog_projection(snapshot)
            catalog = MiniChemCatalogSnapshot(
                catalog_hash=str(snapshot["catalog_hash"]),
                catalog_version=str(snapshot["schema_version"]),
                snapshot=snapshot,
            )
            self._catalog = catalog
            return catalog
        missing = _CATALOG_FIELDS - set(snapshot)
        if missing:
            raise MiniChemMcpProtocolError(
                f"MiniChem canonical catalog 不完整，缺少字段：{sorted(missing)}"
            )
        unknown = set(snapshot) - _CATALOG_FIELDS
        if unknown:
            raise MiniChemMcpProtocolError(
                f"MiniChem canonical catalog 含未知顶层字段：{sorted(unknown)}"
            )
        if snapshot.get("schema_version") != 7:
            raise MiniChemMcpProtocolError(
                "MiniChem canonical catalog schema_version 必须为 7，"
                f"实际为 {snapshot.get('schema_version')!r}"
            )
        for field in ("execution_layers", "actions", "backends", "resources"):
            if not isinstance(snapshot.get(field), list):
                raise MiniChemMcpProtocolError(f"MiniChem catalog.{field} 必须是数组")
        expected_layers = {
            "predefined_actions": (
                "validated common scientific operations",
                False,
            ),
            "native_software": (
                "Agent-authored native inputs and exact allowlisted commands",
                False,
            ),
            "programmable_analysis": (
                "Agent-authored programs in explicitly selected chemistry runtimes",
                False,
            ),
        }
        layer_ids: list[str] = []
        for index, layer in enumerate(snapshot["execution_layers"]):
            if not isinstance(layer, Mapping):
                raise MiniChemMcpProtocolError(
                    f"MiniChem catalog.execution_layers[{index}] 必须是 object"
                )
            if set(layer) != {"id", "role", "mandatory"}:
                raise MiniChemMcpProtocolError(
                    f"MiniChem catalog.execution_layers[{index}] 必须只包含 id/role/mandatory"
                )
            layer_id = layer.get("id")
            if layer_id not in expected_layers or layer_id in layer_ids:
                raise MiniChemMcpProtocolError(
                    f"MiniChem catalog.execution_layers[{index}].id 非法或重复"
                )
            expected_role, expected_mandatory = expected_layers[layer_id]
            if layer.get("role") != expected_role:
                raise MiniChemMcpProtocolError(
                    f"MiniChem catalog.execution_layers[{index}].role 不符合 v7 固定协议"
                )
            if layer.get("mandatory") is not expected_mandatory:
                raise MiniChemMcpProtocolError(
                    f"MiniChem catalog.execution_layers[{index}].mandatory 不符合 v7 固定协议"
                )
            layer_ids.append(layer_id)
        if set(layer_ids) != set(expected_layers):
            raise MiniChemMcpProtocolError(
                "MiniChem catalog.execution_layers 必须完整声明三个执行层"
            )
        discovery_mode = snapshot.get("discovery_mode")
        expected_exposure = {"progressive": "progressive_discovery", "full": "atomic_all"}
        if discovery_mode not in expected_exposure:
            raise MiniChemMcpProtocolError(
                "MiniChem catalog.discovery_mode 必须为 progressive 或 full"
            )
        if snapshot.get("exposure_policy") != expected_exposure[discovery_mode]:
            raise MiniChemMcpProtocolError(
                "MiniChem catalog.exposure_policy 与 discovery_mode 不一致"
            )
        fixed_policies = {
            "catalog_visibility_policy": "focused_task_independent",
            "backend_selection_policy": "per_action_explicit",
            "scientific_resource_selection_policy": "agent_explicit_no_default",
        }
        for field, expected in fixed_policies.items():
            if snapshot.get(field) != expected:
                raise MiniChemMcpProtocolError(
                    f"MiniChem catalog.{field} 必须为 v7 固定策略 {expected!r}"
                )
        if snapshot.get("task_specific_tool_filtering") is not False:
            raise MiniChemMcpProtocolError(
                "MiniChem catalog.task_specific_tool_filtering 必须为 false"
            )
        provider_policies = snapshot.get("provider_selection_policies")
        allowed_provider_policies = {
            "agent_backend_required",
            "agent_components_required",
            "agent_source_required",
            "fixed_source",
            "internal_deterministic",
        }
        if (
            not isinstance(provider_policies, list)
            or any(not isinstance(item, str) for item in provider_policies)
            or provider_policies != sorted(set(provider_policies))
            or any(item not in allowed_provider_policies for item in provider_policies)
        ):
            raise MiniChemMcpProtocolError(
                "MiniChem catalog.provider_selection_policies 必须是排序且去重的 v7 策略数组"
            )
        if snapshot.get("automatic_fallback") is not False:
            raise MiniChemMcpProtocolError("MiniChem catalog.automatic_fallback 必须为 false")
        budget = snapshot.get("evaluation_resource_budget")
        budget_fields = {
            "cpu_cores",
            "memory_mb",
            "gpu_count",
            "source",
            "agent_controllable",
            "scope",
        }
        if not isinstance(budget, Mapping):
            raise MiniChemMcpProtocolError(
                "MiniChem catalog.evaluation_resource_budget 必须是 object"
            )
        if set(budget) != budget_fields:
            raise MiniChemMcpProtocolError(
                "MiniChem catalog.evaluation_resource_budget 字段不符合 v7 schema"
            )
        if (
            type(budget.get("cpu_cores")) is not int
            or budget["cpu_cores"] < 1
            or type(budget.get("memory_mb")) is not int
            or budget["memory_mb"] < 128
            or type(budget.get("gpu_count")) is not int
            or budget["gpu_count"] < 0
            or budget.get("source") != "evaluation_policy"
            or budget.get("agent_controllable") is not False
            or budget.get("scope") != "per_task"
        ):
            raise MiniChemMcpProtocolError(
                "MiniChem catalog.evaluation_resource_budget 值不符合 v7 policy"
            )
        configured_budget = _load_configured_evaluation_resource_budget()
        if dict(budget) != configured_budget:
            raise MiniChemMcpProtocolError(
                "MiniChem catalog.evaluation_resource_budget 与 ARCHE 资源政策不一致："
                f"expected={configured_budget!r} actual={dict(budget)!r}"
            )
        action_ids: set[str] = set()
        for index, action in enumerate(snapshot["actions"]):
            if not isinstance(action, Mapping):
                raise MiniChemMcpProtocolError(f"MiniChem catalog.actions[{index}] 必须是 object")
            action_id = action.get("id")
            _validate_catalog_id(action_id, f"catalog.actions[{index}].id")
            if action_id in action_ids:
                raise MiniChemMcpProtocolError(f"MiniChem catalog action_id 重复：{action_id}")
            action_ids.add(action_id)
            backend_choices = action.get("backend_ids")
            if not isinstance(backend_choices, list) or any(
                not isinstance(item, str) for item in backend_choices
            ):
                raise MiniChemMcpProtocolError(
                    f"MiniChem catalog.actions[{index}].backend_ids 必须是字符串数组"
                )
            if len(backend_choices) != len(set(backend_choices)):
                raise MiniChemMcpProtocolError(
                    f"MiniChem catalog.actions[{index}].backend_ids 不能重复"
                )
            for backend_choice in backend_choices:
                _validate_catalog_id(
                    backend_choice,
                    f"MiniChem catalog.actions[{index}].backend_ids",
                )
        backend_ids: set[str] = set()
        for index, backend in enumerate(snapshot["backends"]):
            if not isinstance(backend, Mapping):
                raise MiniChemMcpProtocolError(f"MiniChem catalog.backends[{index}] 必须是 object")
            backend_id = backend.get("id")
            _validate_catalog_id(backend_id, f"catalog.backends[{index}].id")
            if backend_id in backend_ids:
                raise MiniChemMcpProtocolError(f"MiniChem catalog backend_id 重复：{backend_id}")
            backend_ids.add(backend_id)
        for index, action in enumerate(snapshot["actions"]):
            for backend_id in action.get("backend_ids", []):
                if backend_id not in backend_ids:
                    raise MiniChemMcpProtocolError(
                        f"MiniChem catalog.actions[{index}] 引用了未声明 Backend：{backend_id}"
                    )
        resource_ids: set[str] = set()
        for index, resource in enumerate(snapshot["resources"]):
            if not isinstance(resource, Mapping):
                raise MiniChemMcpProtocolError(
                    f"MiniChem catalog.resources[{index}] 必须是 object"
                )
            if "id" not in resource:
                raise MiniChemMcpProtocolError(
                    f"MiniChem catalog.resources[{index}] 缺少 id"
                )
            _validate_catalog_id(resource["id"], f"catalog.resources[{index}].id")
            if resource["id"] in resource_ids:
                raise MiniChemMcpProtocolError(
                    f"MiniChem catalog resource_id 重复：{resource['id']}"
                )
            resource_ids.add(resource["id"])
        try:
            _canonical_json(snapshot)
        except MiniChemMcpProtocolError as exc:
            raise MiniChemMcpProtocolError("MiniChem canonical catalog 含有非法 JSON 值") from exc
        supplied_hash = snapshot.get("catalog_hash")
        payload = dict(snapshot)
        payload.pop("catalog_hash", None)
        calculated_hash = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
        if not isinstance(supplied_hash, str) or _SHA256.fullmatch(supplied_hash) is None:
            raise MiniChemMcpProtocolError("MiniChem catalog_hash 不是 SHA-256")
        if supplied_hash != calculated_hash:
            raise MiniChemMcpProtocolError(
                "MiniChem catalog_hash 与 canonical snapshot 内容不一致："
                f"expected={calculated_hash} actual={supplied_hash}"
            )
        # 真实 Catalog 没有第二个 catalog_version；以协议 schema 版本作为稳定版本标签。
        version = str(snapshot["schema_version"])
        catalog = MiniChemCatalogSnapshot(
            catalog_hash=supplied_hash,
            catalog_version=str(version),
            snapshot=snapshot,
        )
        self._catalog = catalog
        return catalog

    def inspect_action(
        self,
        action_id: str,
        backend_id: str | None = None,
        *,
        detail_level: str = "contract",
    ) -> dict[str, Any]:
        """读取一个精确 Action/Backend 的请求、输出和资源合同。"""

        _validate_catalog_id(action_id, "action_id")
        if backend_id is not None:
            _validate_catalog_id(backend_id, "backend_id")
        if detail_level not in {"summary", "contract", "full"}:
            raise ValueError("detail_level 必须是 summary、contract 或 full")
        catalog = self._catalog
        if catalog is not None:
            action = next(
                (item for item in catalog.actions if item.get("id") == action_id),
                None,
            )
            if action is None:
                raise MiniChemMcpProtocolError(
                    f"inspect_action 的 action_id 未在冻结目录声明：{action_id}"
                )
            if backend_id is not None and backend_id not in action.get("backend_ids", []):
                raise MiniChemMcpProtocolError(
                    f"inspect_action 的 backend_id 未由 Action 声明：{action_id}/{backend_id}"
                )
        args: dict[str, Any] = {"action_id": action_id, "detail_level": detail_level}
        if backend_id is not None:
            args["backend_id"] = backend_id
        result = self._validate_discovery_catalog_hash(
            self._call_tool("inspect_action", args), "inspect_action"
        )
        if result.get("status") != "success":
            raise MiniChemMcpProtocolError(f"inspect_action 失败：{result}")
        if not isinstance(result.get("action"), Mapping):
            raise MiniChemMcpProtocolError("inspect_action 响应缺少嵌套 action 对象")
        return result

    def inspect_backend(self, backend_id: str, action_id: str | None = None) -> dict[str, Any]:
        """读取一个精确 Backend 及其 Action 能力，不在本地猜测提供者。"""

        _validate_catalog_id(backend_id, "backend_id")
        if action_id is not None:
            _validate_catalog_id(action_id, "action_id")
        catalog = self._catalog
        if catalog is not None:
            backend = next(
                (item for item in catalog.backends if item.get("id") == backend_id),
                None,
            )
            if backend is None:
                raise MiniChemMcpProtocolError(
                    f"inspect_backend 的 backend_id 未在冻结目录声明：{backend_id}"
                )
            if action_id is not None and action_id not in backend.get("capabilities", []):
                raise MiniChemMcpProtocolError(
                    f"inspect_backend 的 action_id 未由 Backend 声明：{backend_id}/{action_id}"
                )
        args: dict[str, Any] = {"backend_id": backend_id}
        if action_id is not None:
            args["action_id"] = action_id
        result = self._validate_discovery_catalog_hash(
            self._call_tool("inspect_backend", args), "inspect_backend"
        )
        if result.get("status") != "success":
            raise MiniChemMcpProtocolError(f"inspect_backend 失败：{result}")
        if not isinstance(result.get("backend"), Mapping):
            raise MiniChemMcpProtocolError("inspect_backend 响应缺少嵌套 backend 对象")
        return result

    def inspect_action_contract(
        self, action_id: str, backend_id: str
    ) -> dict[str, Any]:
        """显式读取一个 Action/Backend 的完整合同，供按需合同阶段使用。"""

        return self.inspect_action(
            action_id,
            backend_id,
            detail_level="full",
        )

    def discover_action_contracts(self) -> dict[str, Any]:
        """逐个读取冻结目录中的 Action/Backend 请求合同。

        canonical Catalog 只说明 Action 可以使用哪些 Backend；真正的必填字段
        仍由 MiniChem 的 ``inspect_action`` 按 pair 返回。该方法把每个 pair 的
        request/output 合同保存为安全 JSON，供任务冻结时校验；不把服务状态、
        本机路径或完整 inspect 响应写入模型目录。
        """

        catalog = getattr(self, "_catalog", None) or self.discover_catalog()
        contracts: list[dict[str, Any]] = []
        for action in sorted(catalog.actions, key=lambda item: str(item.get("id", ""))):
            action_id = _validate_catalog_id(action.get("id"), "catalog.actions[].id")
            backend_ids = action.get("backend_ids")
            if not isinstance(backend_ids, list) or any(
                not isinstance(item, str) for item in backend_ids
            ):
                raise MiniChemMcpProtocolError(
                    f"Action {action_id} 的 backend_ids 必须是字符串数组"
                )
            for backend_id in sorted(set(backend_ids)):
                _validate_catalog_id(backend_id, "catalog.actions[].backend_ids[]")
                inspected = self.inspect_action(action_id, backend_id)
                inspected_action = inspected.get("action")
                if not isinstance(inspected_action, Mapping):
                    raise MiniChemMcpProtocolError(
                        f"inspect_action {action_id}/{backend_id} 缺少 action object"
                    )
                returned_action_id = inspected_action.get(
                    "id", inspected_action.get("action_id")
                )
                if returned_action_id != action_id:
                    raise MiniChemMcpProtocolError(
                        f"inspect_action 返回了错误的 action_id："
                        f"expected={action_id!r} actual={returned_action_id!r}"
                    )
                request_contract = inspected.get("selected_request_contract")
                if request_contract is None:
                    request_contract = inspected_action.get("request_contract")
                if not isinstance(request_contract, Mapping):
                    raise MiniChemMcpProtocolError(
                        f"inspect_action {action_id}/{backend_id} 缺少 selected_request_contract"
                    )
                entry: dict[str, Any] = {
                    "action_id": action_id,
                    "backend_id": backend_id,
                    "request_contract": deepcopy(dict(request_contract)),
                    "output_contract": None,
                }
                output_contract = request_contract.get("output_contract")
                if output_contract is not None:
                    if not isinstance(output_contract, Mapping):
                        raise MiniChemMcpProtocolError(
                            f"inspect_action {action_id}/{backend_id}.output_contract 必须是 object"
                        )
                    entry["output_contract"] = deepcopy(dict(output_contract))
                try:
                    _canonical_json(entry)
                except MiniChemMcpProtocolError as exc:
                    raise MiniChemMcpProtocolError(
                        f"inspect_action {action_id}/{backend_id} 合同不是安全 JSON"
                    ) from exc
                contracts.append(entry)
        return {"schema_version": 1, "contracts": contracts}

    def discover_open_execution_catalog(self) -> dict[str, Any]:
        """读取并脱敏投影 MiniChem 的 native/analysis 开放执行能力。

        输入来自三次真实 MCP discovery 路径：可用 native 软件紧凑清单、每个
        软件的精确 inspect，以及可用分析运行时清单。函数只返回 Planner 需要的
        命令、输入边界和模块名称，不传播解析后的本机路径、解释器、外部命令或
        文档正文；响应结构错误或身份重复时整次 discovery 失败。
        """

        software_result = self._call_tool(
            "list_software",
            {"native_only": True, "available_only": True, "limit": 200},
        )
        _require_open_execution_success(software_result, "list_software")
        software_items = software_result.get("software")
        if not isinstance(software_items, list):
            raise MiniChemMcpProtocolError("list_software.software 必须是数组")
        software_by_id: dict[str, dict[str, Any]] = {}
        for index, item in enumerate(software_items):
            if not isinstance(item, Mapping):
                raise MiniChemMcpProtocolError(f"list_software.software[{index}] 必须是 object")
            software_id = item.get("software_id")
            _validate_catalog_id(software_id, f"list_software.software[{index}].software_id")
            if software_id in software_by_id:
                raise MiniChemMcpProtocolError(f"list_software 软件 id 重复：{software_id}")
            if item.get("available") is not True:
                raise MiniChemMcpProtocolError(
                    f"list_software.software[{index}].available 必须为 true"
                )
            listed_commands: dict[str, tuple[bool, bool]] = {}
            compact_commands = item.get("native_executables")
            if compact_commands is not None:
                if not isinstance(compact_commands, list):
                    raise MiniChemMcpProtocolError(
                        f"list_software.software[{index}].native_executables 必须是数组"
                    )
                for command_index, compact_command in enumerate(compact_commands):
                    if not isinstance(compact_command, Mapping):
                        raise MiniChemMcpProtocolError(
                            f"list_software.software[{index}].native_executables[{command_index}] "
                            "必须是 object"
                        )
                    compact_executable = compact_command.get("executable")
                    if (
                        not isinstance(compact_executable, str)
                        or not _SAFE_EXECUTABLE.fullmatch(compact_executable)
                        or compact_executable in listed_commands
                    ):
                        raise MiniChemMcpProtocolError(
                            f"list_software.software[{index}] native executable 非法或重复"
                        )
                    if (
                        type(compact_command.get("enabled")) is not bool
                        or type(compact_command.get("available")) is not bool
                    ):
                        raise MiniChemMcpProtocolError(
                            f"list_software.software[{index}] native executable enabled/available "
                            "必须是 bool"
                        )
                    listed_commands[compact_executable] = (
                        compact_command["enabled"],
                        compact_command["available"],
                    )
            display_name = _validate_open_text(
                item.get("display_name"),
                f"list_software.software[{index}].display_name",
            )
            software_by_id[str(software_id)] = {
                "software_id": str(software_id),
                "display_name": display_name,
                "listed_commands": listed_commands,
            }

        native_software: list[dict[str, Any]] = []
        for software_id in sorted(software_by_id):
            inspected = self._call_tool("inspect_software", {"software_id": software_id})
            _require_open_execution_success(inspected, "inspect_software")
            if inspected.get("software_id") != software_id:
                raise MiniChemMcpProtocolError(
                    "inspect_software 返回的 software_id 与请求不一致："
                    f"expected={software_id!r} actual={inspected.get('software_id')!r}"
                )
            guides = inspected.get("native_invocation_guides")
            if not isinstance(guides, list):
                raise MiniChemMcpProtocolError(
                    f"inspect_software[{software_id}].native_invocation_guides 必须是数组"
                )
            commands_by_id: dict[str, dict[str, Any]] = {}
            for index, guide in enumerate(guides):
                if not isinstance(guide, Mapping):
                    raise MiniChemMcpProtocolError(
                        f"inspect_software[{software_id}].native_invocation_guides[{index}] 必须是 object"
                    )
                executable = guide.get("executable")
                if (
                    not isinstance(executable, str)
                    or not executable.strip()
                    or not _SAFE_EXECUTABLE.fullmatch(executable)
                ):
                    raise MiniChemMcpProtocolError(
                        f"inspect_software[{software_id}] command executable 非法"
                    )
                if executable in commands_by_id:
                    raise MiniChemMcpProtocolError(
                        f"inspect_software[{software_id}] command 重复：{executable}"
                    )
                if (
                    type(guide.get("enabled")) is not bool
                    or type(guide.get("available")) is not bool
                ):
                    raise MiniChemMcpProtocolError(
                        f"inspect_software[{software_id}] command enabled/available 必须是 bool"
                    )
                listed_status = software_by_id[software_id]["listed_commands"].get(executable)
                if listed_status is not None and listed_status != (True, True):
                    continue
                if software_by_id[software_id]["listed_commands"] and listed_status is None:
                    continue
                if guide["enabled"] and guide["available"]:
                    synopsis = guide.get("synopsis")
                    input_mode = guide.get("input_mode")
                    required_files = guide.get("required_files")
                    synopsis = _validate_open_text(
                        synopsis,
                        f"inspect_software[{software_id}]/{executable}.synopsis",
                    )
                    input_mode = _validate_open_text(
                        input_mode,
                        f"inspect_software[{software_id}]/{executable}.input_mode",
                    )
                    if not isinstance(required_files, list) or any(
                        not isinstance(item, str)
                        or not item.strip()
                        or "\r" in item
                        or "\n" in item
                        for item in required_files
                    ):
                        raise MiniChemMcpProtocolError(
                            f"inspect_software[{software_id}]/{executable}.required_files "
                            "必须是字符串数组"
                        )
                    command_projection = {
                        "executable": executable,
                        "synopsis": synopsis,
                        "input_mode": input_mode,
                        "required_files": sorted(set(required_files)),
                        "example_request": _project_native_job_example(
                            guide.get("native_job_request_template"),
                            software_id=software_id,
                            executable=executable,
                        ),
                    }
                    output_behavior = guide.get("output_behavior")
                    if output_behavior is not None:
                        command_projection["output_behavior"] = _validate_open_description(
                            output_behavior,
                            f"inspect_software[{software_id}]/{executable}.output_behavior",
                        )
                    commands_by_id[executable] = command_projection
            native_software.append(
                {
                    "software_id": software_by_id[software_id]["software_id"],
                    "display_name": software_by_id[software_id]["display_name"],
                    "commands": [commands_by_id[key] for key in sorted(commands_by_id)],
                }
            )

        runtime_result = self._call_tool(
            "list_analysis_runtimes",
            {"available_only": True, "limit": 200, "include_details": True},
        )
        _require_open_execution_success(runtime_result, "list_analysis_runtimes")
        runtimes = runtime_result.get("runtimes")
        if not isinstance(runtimes, list):
            raise MiniChemMcpProtocolError("list_analysis_runtimes.runtimes 必须是数组")
        runtime_by_id: dict[str, dict[str, Any]] = {}
        for index, item in enumerate(runtimes):
            if not isinstance(item, Mapping):
                raise MiniChemMcpProtocolError(f"list_analysis_runtimes.runtimes[{index}] 必须是 object")
            runtime = item.get("runtime")
            if not isinstance(runtime, str) or _SAFE_RUNTIME_ID.fullmatch(runtime) is None:
                raise MiniChemMcpProtocolError(
                    f"list_analysis_runtimes.runtimes[{index}].runtime 非法"
                )
            if runtime in runtime_by_id:
                raise MiniChemMcpProtocolError(f"分析运行时 id 重复：{runtime}")
            if item.get("available") is not True:
                raise MiniChemMcpProtocolError(
                    f"list_analysis_runtimes.runtimes[{index}].available 必须为 true"
                )
            description = _validate_open_text(
                item.get("description"),
                f"list_analysis_runtimes.runtimes[{index}].description",
            )
            modules = item.get("modules")
            if not isinstance(modules, Mapping):
                raise MiniChemMcpProtocolError(
                    f"list_analysis_runtimes.runtimes[{index}].modules 必须是 object"
                )
            module_names: list[str] = []
            for module_name in modules:
                if (
                    not isinstance(module_name, str)
                    or _SAFE_MODULE_NAME.fullmatch(module_name) is None
                ):
                    raise MiniChemMcpProtocolError(
                        f"list_analysis_runtimes.runtimes[{index}].modules 含非法模块名"
                    )
                module_names.append(module_name)
            runtime_by_id[runtime] = {
                "runtime": runtime,
                "description": description,
                "modules": sorted(set(module_names)),
            }
        catalog: dict[str, Any] = {
            "schema_version": 1,
            "native_software": native_software,
            "analysis_runtimes": [runtime_by_id[key] for key in sorted(runtime_by_id)],
            "authoring_contract": {
                "path_prefix": "code/",
                "max_files": 4,
                "max_chars_per_file": 100000,
            },
        }
        if "managed_program_contract" in runtime_result:
            catalog["managed_program_contract"] = _project_managed_program_contract(
                runtime_result["managed_program_contract"]
            )
        if "submit_analysis_program_request_template" in runtime_result:
            catalog["submit_analysis_program_request_template"] = (
                _project_analysis_program_template(
                    runtime_result["submit_analysis_program_request_template"]
                )
            )
        for note_field in ("selection_note", "execution_note"):
            if note_field in runtime_result:
                catalog[note_field] = _validate_open_description(
                    runtime_result[note_field], f"list_analysis_runtimes.{note_field}"
                )
        if "evaluation_resource_budget" in runtime_result:
            catalog["evaluation_resource_budget"] = _validate_evaluation_budget_record(
                runtime_result["evaluation_resource_budget"],
                "list_analysis_runtimes.evaluation_resource_budget",
            )
        if "execution_timeout_policy" in runtime_result:
            catalog["execution_timeout_policy"] = _validate_timeout_policy_record(
                runtime_result["execution_timeout_policy"],
                "list_analysis_runtimes.execution_timeout_policy",
            )
        return catalog

    def ensure_workspace_text(
        self,
        path: str,
        content: str,
        *,
        operation_id: str | None = None,
    ) -> dict[str, Any]:
        """幂等写入 ``code/`` 文本，并拒绝覆盖内容不同的已有文件。

        先读取并比较 SHA-256；文件不存在时使用 ``overwrite=false`` 写入，再次
        读取核对。并发写入导致首次写入失败时只接受最终文件内容与本次内容完全
        相同的情况，其他响应均转换为协议错误；函数不打印或记录文本正文。
        """

        safe_path = _validate_code_workspace_path(path)
        operation = _validate_operation_id(operation_id) if operation_id is not None else None
        if not isinstance(content, str):
            raise MiniChemMcpProtocolError("workspace content 必须是字符串")
        if len(content) > 100000:
            raise MiniChemMcpProtocolError("workspace content 不能超过 100000 个字符")
        expected_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()

        def read_and_check() -> tuple[dict[str, Any], bool]:
            result = self._call_tool(
                "read_workspace_text",
                {"path": safe_path, "max_chars": 100000},
            )
            if result.get("status") != "success":
                return result, False
            if result.get("path") != safe_path:
                raise MiniChemMcpProtocolError("read_workspace_text 返回路径与请求不一致")
            actual_sha256 = result.get("sha256")
            if not isinstance(actual_sha256, str) or _SHA256.fullmatch(actual_sha256) is None:
                raise MiniChemMcpProtocolError("read_workspace_text.sha256 不是 SHA-256")
            return result, actual_sha256 == expected_sha256

        existing, matches = read_and_check()
        if existing.get("status") == "success":
            if not matches:
                raise MiniChemMcpProtocolError(
                    f"workspace 文件 {safe_path} 已存在但内容不一致，拒绝覆盖"
                )
            return {
                key: deepcopy(existing[key])
                for key in ("status", "path", "size_bytes", "sha256", "encoding")
                if key in existing
            } | {"reused": True}
        if not _workspace_file_missing(existing):
            raise MiniChemMcpProtocolError(f"read_workspace_text 失败：{existing}")

        write_payload: dict[str, Any] = {
            "path": safe_path,
            "content": content,
            "overwrite": False,
        }
        if operation is not None:
            write_payload["operation_id"] = operation
        written = self._call_tool(
            "write_workspace_text",
            write_payload,
            retry_safe=operation is not None,
        )
        if written.get("status") != "success" and not _workspace_write_race(written):
            raise MiniChemMcpProtocolError(f"write_workspace_text 失败：{written}")
        verified, matches = read_and_check()
        if verified.get("status") != "success" or not matches:
            raise MiniChemMcpProtocolError(
                f"write_workspace_text 后无法核对 {safe_path} 的内容哈希"
            )
        return {
            key: deepcopy(verified[key])
            for key in ("status", "path", "size_bytes", "sha256", "encoding")
            if key in verified
        } | {"reused": False}

    @staticmethod
    def _inject_operation_id(request: Mapping[str, Any], operation_id: str) -> dict[str, Any]:
        """复制 open-execution 请求并注入适配器负责的 operation identity。"""

        if not isinstance(request, Mapping):
            raise MiniChemMcpProtocolError("MiniChem open-execution request 必须是 object")
        payload = deepcopy(dict(request))
        supplied = payload.get("operation_id")
        if supplied is not None and supplied != operation_id:
            raise MiniChemMcpProtocolError(
                "open-execution request.operation_id 与适配器 operation_id 不一致"
            )
        payload["operation_id"] = operation_id
        return payload

    def validate_native_job(
        self,
        request: Mapping[str, Any],
        operation_id: str,
    ) -> dict[str, Any]:
        """为 native 请求注入安全 operation_id，并调用 MiniChem 原生校验工具。"""

        if not isinstance(request, Mapping):
            raise MiniChemMcpProtocolError("MiniChem native request 必须是 object")
        unknown = set(request) - _NATIVE_JOB_REQUEST_FIELDS
        if unknown:
            raise MiniChemMcpProtocolError(
                "submit_native_job request 含真实 NativeJobRequest 未声明字段："
                f"{sorted(unknown)}"
            )
        operation = _validate_operation_id(operation_id)
        return self._call_tool(
            "validate_native_job",
            self._inject_operation_id(request, operation),
        )

    def validate_analysis_program(
        self,
        request: Mapping[str, Any],
        operation_id: str,
    ) -> dict[str, Any]:
        """为 analysis 请求注入安全 operation_id，并调用 MiniChem 原生校验工具。"""

        if not isinstance(request, Mapping):
            raise MiniChemMcpProtocolError("MiniChem analysis request 必须是 object")
        _validate_analysis_program_request(request)
        operation = _validate_operation_id(operation_id)
        return self._call_tool(
            "validate_analysis_program",
            self._inject_operation_id(request, operation),
        )

    def list_software(
        self, *, native_only: bool = True, available_only: bool = True
    ) -> dict[str, Any]:
        """读取 open-execution 的本机软件清单，不替 Agent 选择软件。"""

        result = self._call_tool(
            "list_software",
            {
                "native_only": native_only,
                "available_only": available_only,
                "offset": 0,
                "limit": 200,
            },
        )
        if result.get("status") != "success" or not isinstance(
            result.get("software"), list
        ):
            raise MiniChemMcpProtocolError(f"list_software 失败：{result}")
        return result

    def inspect_software(self, software_id: str) -> dict[str, Any]:
        """读取一个本机软件的允许命令和 Agent-authored 请求模板。"""

        result = self._call_tool("inspect_software", {"software_id": software_id})
        if result.get("status") != "success":
            raise MiniChemMcpProtocolError(f"inspect_software 失败：{result}")
        return result

    def list_analysis_runtimes(self) -> dict[str, Any]:
        """读取可用于 Agent-authored 分析程序的精确运行时清单。"""

        result = self._call_tool(
            "list_analysis_runtimes",
            {"available_only": True, "include_details": False, "limit": 500},
        )
        if result.get("status") != "success" or not isinstance(
            result.get("runtimes"), list
        ):
            raise MiniChemMcpProtocolError(
                f"list_analysis_runtimes 失败：{result}"
            )
        return result

    def _validate_declared_action_backend(
        self,
        action_id: str,
        backend_id: str | None,
        *,
        source_id: str | None = None,
    ) -> None:
        """在已有冻结目录时核对 Action、Backend 和来源的声明关系。"""

        catalog = self._catalog
        if catalog is None:
            return
        action_record = None
        for candidate in catalog.actions:
            candidate_id = candidate.get("id")
            if candidate_id == action_id:
                action_record = candidate
                break
        if action_record is None:
            raise MiniChemMcpProtocolError(
                f"execute_action 的 action_id 未在冻结 MiniChem Catalog 声明：{action_id}"
            )
        selection_policy = action_record.get("selection_policy")
        if selection_policy in {"agent_backend_required", "agent_components_required"}:
            if backend_id is None:
                raise MiniChemMcpProtocolError(
                    f"execute_action 的 Action {action_id} 必须显式提供 backend_id"
                )
        elif selection_policy == "agent_source_required" and source_id is None:
            raise MiniChemMcpProtocolError(
                f"execute_action 的 Action {action_id} 必须显式提供 source_id"
            )
        if backend_id is None:
            return
        declared = action_record.get("backend_ids", [])
        if not isinstance(declared, list) or backend_id not in declared:
            raise MiniChemMcpProtocolError(
                f"execute_action 的 backend_id 未被 Action 声明：{action_id}/{backend_id}"
            )
        backend_record = next(
            (
                candidate
                for candidate in catalog.backends
                if candidate.get("id") == backend_id
            ),
            None,
        )
        if backend_record is None:
            raise MiniChemMcpProtocolError(
                f"execute_action 的 backend_id 未在冻结 MiniChem Catalog 声明：{backend_id}"
            )

    def _validate_action_result_identity(
        self,
        result: Mapping[str, Any],
        *,
        action_id: str,
        backend_id: str | None,
    ) -> None:
        """核对成功 ActionResult 的 Action/Backend 身份和冻结目录来源。"""

        status = str(result.get("status", "")).lower()
        if status not in _ACTION_SUCCESS_STATUSES:
            return
        returned_action = result.get("action")
        if returned_action != action_id:
            raise MiniChemMcpProtocolError(
                "execute_action 返回的 action 与请求不一致："
                f"expected={action_id!r} actual={returned_action!r}"
            )
        for field in ("requested_backend", "backend"):
            returned_backend = result.get(field)
            if backend_id is not None and returned_backend not in {None, backend_id}:
                raise MiniChemMcpProtocolError(
                    f"execute_action 返回的 {field} 与请求 Backend 不一致："
                    f"expected={backend_id!r} actual={returned_backend!r}"
                )
        if self._catalog is None:
            return
        provenance = result.get("provenance")
        if not isinstance(provenance, Mapping):
            raise MiniChemMcpProtocolError(
                "execute_action 成功响应缺少 provenance.catalog_hash"
            )
        returned_hash = provenance.get("catalog_hash")
        if returned_hash != self._catalog.catalog_hash:
            raise MiniChemMcpProtocolError(
                "execute_action provenance.catalog_hash 与冻结目录不一致："
                f"expected={self._catalog.catalog_hash} actual={returned_hash!r}"
            )

    def execute_action(
        self,
        request: MiniChemActionRequest | Mapping[str, Any],
        *,
        operation_id: str | None = None,
    ) -> dict[str, Any]:
        """同步执行一个显式 Action，并把 ActionResult 归一为 ARCHE 状态。

        内部 operation_id 只由 ARCHE 执行上下文注入，不进入 Planner 的公开请求。
        提供该身份时，服务端可以在传输响应丢失后安全回放已完成响应。
        """

        supplied_operation = None
        if isinstance(request, MiniChemActionRequest):
            payload = request.to_payload()
        else:
            if not isinstance(request, Mapping):
                raise MiniChemMcpProtocolError("execute_action request 必须是 object")
            supplied_operation = request.get("operation_id")
            clean_request = {
                key: value for key, value in request.items() if key != "operation_id"
            }
            payload = _validate_action_request_mapping(clean_request)
        if operation_id is None:
            operation_id = supplied_operation
        elif supplied_operation is not None and supplied_operation != operation_id:
            raise MiniChemMcpProtocolError(
                "execute_action request.operation_id 与适配器 operation_id 不一致"
            )
        if isinstance(request, MiniChemActionRequest):
            payload = _validate_action_request_mapping(payload)
        if operation_id is not None:
            operation_id = _validate_operation_id(operation_id)
            payload["operation_id"] = operation_id
        action_id = payload["action_id"]
        self._validate_declared_action_backend(
            action_id,
            payload.get("backend_id"),
            source_id=payload.get("source_id"),
        )
        result = self._call_tool(
            "execute_action", payload, retry_safe=operation_id is not None
        )
        result = _validate_artifact_records(
            result,
            action_id=action_id,
            backend_id=payload.get("backend_id"),
        )
        self._validate_action_result_identity(
            result,
            action_id=action_id,
            backend_id=payload.get("backend_id"),
        )

        return self._normalize_result(
            result,
            operation_id=operation_id,
            action_id=action_id,
            result_kind="action",
        )

    def write_workspace_text(
        self,
        path: str,
        content: str,
        *,
        overwrite: bool = False,
        operation_id: str | None = None,
    ) -> dict[str, Any]:
        """写入 Agent-authored 文件；内部身份存在时允许安全重试。"""

        payload = {"path": path, "content": content, "overwrite": overwrite}
        if operation_id is not None:
            payload["operation_id"] = _validate_operation_id(operation_id)
        return self._call_tool(
            "write_workspace_text",
            payload,
            retry_safe=operation_id is not None,
        )

    def read_workspace_text(
        self,
        path: str,
        *,
        max_chars: int = 100_000,
        offset: int = 0,
        json_pointer: str | None = None,
        expected_sha256: str | None = None,
    ) -> dict[str, Any]:
        """读取有界 UTF-8 内容，并核对分页游标、JSON Pointer 和文件哈希。

        expected_sha256 可直接传入 ArtifactRef 的文件哈希；同一文件的
        连续分页即使未传该参数，也会在 adapter 内核对前后页哈希一致。
        """

        if type(max_chars) is not int or not 1 <= max_chars <= 2_000_000:
            raise MiniChemMcpProtocolError(
                "read_workspace_text.max_chars 必须是 1 到 2000000 的整数"
            )
        if type(offset) is not int or offset < 0:
            raise MiniChemMcpProtocolError(
                "read_workspace_text.offset 必须是非负整数"
            )
        if json_pointer is not None and (
            not isinstance(json_pointer, str) or len(json_pointer) > 100_000
        ):
            raise MiniChemMcpProtocolError(
                "read_workspace_text.json_pointer 必须是不超过 100000 个字符的字符串"
            )
        if expected_sha256 is not None and (
            not isinstance(expected_sha256, str)
            or _SHA256.fullmatch(expected_sha256) is None
        ):
            raise MiniChemMcpProtocolError(
                "read_workspace_text.expected_sha256 不是 SHA-256"
            )
        payload = {
            "path": path,
            "max_chars": max_chars,
            "offset": offset,
            "json_pointer": json_pointer,
        }
        result = self._call_tool("read_workspace_text", payload)
        if result.get("status") != "success":
            return result
        checked = _validate_workspace_text_read_result(
            result,
            path=path,
            max_chars=max_chars,
            offset=offset,
            json_pointer=json_pointer,
        )
        returned_hash = checked["sha256"]
        if expected_sha256 is not None and returned_hash != expected_sha256:
            raise MiniChemMcpProtocolError(
                "read_workspace_text.sha256 与调用方提供的 expected_sha256 不一致："
                f"expected={expected_sha256} actual={returned_hash}"
            )
        read_key = (path, json_pointer)
        with self._workspace_read_guard:
            known_hash = self._workspace_read_hashes.get(read_key)
            if offset > 0 and known_hash is not None and returned_hash != known_hash:
                raise MiniChemMcpProtocolError(
                    "read_workspace_text.sha256 在连续分页间发生漂移："
                    f"expected={known_hash} actual={returned_hash}"
                )
            if offset == 0:
                self._workspace_read_hashes[read_key] = expected_sha256 or returned_hash
            elif known_hash is None:
                self._workspace_read_hashes[read_key] = expected_sha256 or returned_hash
            if checked["next_offset"] is None:
                self._workspace_read_hashes.pop(read_key, None)
        return checked

    def submit_native_job(
        self,
        request: Mapping[str, Any],
        operation_id: str | None = None,
        *,
        submission_key: str | None = None,
    ) -> dict[str, Any]:
        """提交 native 作业，兼容 legacy operation 与 scope submission 两种合同。"""

        embedded_key = request.get("submission_key") if isinstance(request, Mapping) else None
        scoped_key = submission_key if submission_key is not None else embedded_key
        if scoped_key is not None:
            return self._submit_scoped_job("submit_native_job", request, scoped_key)
        if operation_id is None:
            raise ValueError("legacy submit_native_job 必须提供 operation_id")
        operation = _validate_operation_id(operation_id)
        with self._operation_submission_lock(operation):
            return self._submit_job_locked("submit_native_job", request, operation)

    def submit_analysis_program(
        self,
        request: Mapping[str, Any],
        operation_id: str | None = None,
        *,
        submission_key: str | None = None,
    ) -> dict[str, Any]:
        """提交 programmable analysis，scope 请求只使用稳定 submission_key。"""

        embedded_key = request.get("submission_key") if isinstance(request, Mapping) else None
        scoped_key = submission_key if submission_key is not None else embedded_key
        if scoped_key is not None:
            return self._submit_scoped_job("submit_analysis_program", request, scoped_key)
        if not isinstance(request, Mapping):
            raise MiniChemMcpProtocolError("MiniChem analysis request 必须是 object")
        _validate_analysis_program_request(request)
        if operation_id is None:
            raise ValueError("legacy submit_analysis_program 必须提供 operation_id")
        operation = _validate_operation_id(operation_id)
        with self._operation_submission_lock(operation):
            return self._submit_job_locked("submit_analysis_program", request, operation)

    @staticmethod
    def _scoped_receipt_key(tool_name: str, submission_key: str) -> str:
        """为 scope 提交派生不透明本地收据键，避免把提交键当作路径。"""

        digest = hashlib.sha256(
            f"{tool_name}\0{submission_key}".encode("utf-8")
        ).hexdigest()
        return f"scope-submission-{digest}"

    def _submit_scoped_job(
        self,
        tool_name: str,
        request: Mapping[str, Any],
        submission_key: str,
    ) -> dict[str, Any]:
        """使用服务端生成的 operation identity 提交并保存 scope 收据。"""

        if not isinstance(request, Mapping):
            raise MiniChemMcpProtocolError("MiniChem job request 必须是 object")
        key = _validate_submission_key(submission_key)
        payload = deepcopy(dict(request))
        embedded_key = payload.get("submission_key")
        if embedded_key is not None and _validate_submission_key(embedded_key) != key:
            raise MiniChemMcpProtocolError(
                "MiniChem job request.submission_key 与适配器 submission_key 不一致"
            )
        # scope 合同禁止调用方 operation_id；即使上游误带，也不能把它发送到 MCP。
        payload.pop("operation_id", None)
        payload["submission_key"] = key
        if tool_name == "submit_native_job":
            unknown = set(payload) - _NATIVE_JOB_REQUEST_FIELDS
            if unknown:
                raise MiniChemMcpProtocolError(
                    "submit_native_job request 含真实 NativeJobRequest 未声明字段："
                    f"{sorted(unknown)}"
                )
        elif tool_name == "submit_analysis_program":
            _validate_analysis_program_request(payload)
        else:  # pragma: no cover - 仅由本模块固定工具入口调用
            raise ValueError(f"不支持的 scope submit tool: {tool_name}")

        fingerprint = _job_request_fingerprint(tool_name, payload)
        receipt_key = self._scoped_receipt_key(tool_name, key)
        with self._operation_submission_lock(receipt_key):
            previous = self._operations.get(receipt_key) or self._load_operation(receipt_key)
            if previous is not None:
                previous_tool = previous.get("tool")
                previous_key = previous.get("submission_key")
                previous_fingerprint = previous.get("request_fingerprint")
                previous_request = previous.get("request")
                try:
                    previous_calculated = (
                        _job_request_fingerprint(previous_tool, previous_request)
                        if isinstance(previous_tool, str)
                        and isinstance(previous_request, Mapping)
                        else None
                    )
                except MiniChemMcpProtocolError:
                    previous_calculated = None
                receipt_valid = (
                    previous_tool in _JOB_SUBMIT_TOOLS
                    and previous_key == key
                    and isinstance(previous_fingerprint, str)
                    and _SHA256.fullmatch(previous_fingerprint) is not None
                    and previous.get("operation_fingerprint") == previous_fingerprint
                    and isinstance(previous_request, Mapping)
                    and previous_calculated == previous_fingerprint
                    and isinstance(previous.get("operation_id"), str)
                    and _SAFE_OPERATION.fullmatch(previous["operation_id"]) is not None
                    and isinstance(previous.get("job_id"), str)
                    and _JOB_ID.fullmatch(previous["job_id"]) is not None
                )
                if not receipt_valid:
                    return {
                        "status": "technical_pause",
                        "minichem_status": "submission_receipt_invalid",
                        "minichem_envelope_status": "invalid_request",
                        "operation_id": previous.get("operation_id"),
                        "job_id": previous.get("job_id"),
                        "submission_key": key,
                        "error": {
                            "code": "submission_receipt_invalid",
                            "message": "已有 scope submission 收据缺少或篡改了身份与请求指纹",
                        },
                    }
                if previous_fingerprint != fingerprint or previous_tool != tool_name:
                    return {
                        "status": "technical_pause",
                        "minichem_status": "submission_key_conflict",
                        "minichem_envelope_status": "invalid_request",
                        "operation_id": previous.get("operation_id"),
                        "job_id": previous.get("job_id"),
                        "submission_key": key,
                        "error": {
                            "code": "submission_key_conflict",
                            "message": "同一 submission_key 已绑定不同工具或规范请求，拒绝重复副作用",
                        },
                    }
                return deepcopy(previous)

            result = _validate_artifact_records(
                self._call_tool(tool_name, payload, retry_safe=True)
            )
            normalized = self._normalize_result(result, result_kind="job_submit")
            envelope_status = str(result.get("status", "")).lower()
            if envelope_status == "success":
                server_operation = result.get("operation_id")
                if not isinstance(server_operation, str):
                    raise MiniChemMcpProtocolError(
                        f"{tool_name} scope 响应缺少服务端 operation_id"
                    )
                try:
                    server_operation = _validate_operation_id(server_operation)
                except ValueError as exc:
                    raise MiniChemMcpProtocolError(
                        f"{tool_name} scope 响应的 operation_id 非法"
                    ) from exc
                if result.get("submission_key") != key:
                    raise MiniChemMcpProtocolError(
                        f"{tool_name} scope 响应的 submission_key 与请求不一致"
                    )
                normalized["operation_id"] = server_operation
                normalized["submission_key"] = key
                normalized["request_fingerprint"] = fingerprint
                normalized["operation_fingerprint"] = fingerprint
                normalized["tool"] = tool_name
                normalized["request"] = deepcopy(payload)
                if (
                    normalized.get("status") in {"waiting", "completed"}
                    and isinstance(normalized.get("job_id"), str)
                    and _JOB_ID.fullmatch(normalized["job_id"])
                ):
                    self._operations[receipt_key] = deepcopy(normalized)
                    self._persist_operation(receipt_key, normalized)
            return normalized

    def _submit_job_locked(
        self,
        tool_name: str,
        request: Mapping[str, Any],
        operation: str,
    ) -> dict[str, Any]:
        """在跨进程互斥区内执行一次 native/analysis job 提交。"""

        previous = self._operations.get(operation) or self._load_operation(operation)
        if not isinstance(request, Mapping):
            raise MiniChemMcpProtocolError("MiniChem job request 必须是 object")
        if tool_name == "submit_native_job":
            unknown = set(request) - _NATIVE_JOB_REQUEST_FIELDS
            if unknown:
                raise MiniChemMcpProtocolError(
                    "submit_native_job request 含真实 NativeJobRequest 未声明字段："
                    f"{sorted(unknown)}"
                )
        if tool_name == "submit_analysis_program":
            _validate_analysis_program_request(request)
        payload = deepcopy(dict(request))
        supplied_operation = payload.get("operation_id")
        if supplied_operation is not None and supplied_operation != operation:
            raise MiniChemMcpProtocolError(
                "MiniChem job request.operation_id 与适配器 operation_id 不一致"
            )
        fingerprint = _job_request_fingerprint(tool_name, payload)
        if previous is not None:
            if previous.get("receipt_kind") == _AMBIGUOUS_NATIVE_RECEIPT_KIND:
                previous_tool = previous.get("tool")
                previous_fingerprint = previous.get("request_fingerprint")
                if (
                    previous_tool != tool_name
                    or previous_fingerprint != fingerprint
                ):
                    job_id = previous.get("job_id")
                    return {
                        "status": "technical_pause",
                        "minichem_status": "operation_id_conflict",
                        "minichem_envelope_status": "invalid_request",
                        "operation_id": operation,
                        "job_id": job_id,
                        "error": {
                            "code": "operation_id_conflict",
                            "message": (
                                "同一 operation_id 已进入 native pending recovery，"
                                "请求或工具身份不一致，拒绝任何新副作用"
                            ),
                            "job_id": job_id,
                            "tool": previous_tool,
                            "request_fingerprint": previous_fingerprint,
                        },
                    }
                try:
                    recovered = self.recover_native_operation(
                        operation,
                        request_fingerprint=fingerprint,
                        collect=False,
                    )
                except Exception as recovery_error:
                    return self._native_pending_recovery_pause(
                        operation,
                        previous=previous,
                        request_fingerprint=fingerprint,
                        error=recovery_error,
                    )
                return recovered
            previous_tool = previous.get("tool")
            previous_fingerprint = previous.get("request_fingerprint")
            previous_request = previous.get("request")
            try:
                previous_calculated_fingerprint = (
                    _job_request_fingerprint(previous_tool, previous_request)
                    if isinstance(previous_tool, str) and isinstance(previous_request, Mapping)
                    else None
                )
            except MiniChemMcpProtocolError:
                previous_calculated_fingerprint = None
            receipt_valid = (
                previous.get("operation_id") == operation
                and isinstance(previous_tool, str)
                and previous_tool in _JOB_SUBMIT_TOOLS
                and isinstance(previous_fingerprint, str)
                and _SHA256.fullmatch(previous_fingerprint) is not None
                and previous.get("operation_fingerprint") == previous_fingerprint
                and isinstance(previous_request, Mapping)
                and previous_calculated_fingerprint == previous_fingerprint
                and isinstance(previous.get("job_id"), str)
                and _JOB_ID.fullmatch(previous["job_id"]) is not None
            )
            if not receipt_valid:
                return {
                    "status": "technical_pause",
                    "minichem_status": "operation_receipt_invalid",
                    "minichem_envelope_status": "invalid_request",
                    "operation_id": operation,
                    "job_id": previous.get("job_id"),
                    "error": {
                        "code": "operation_receipt_invalid",
                        "message": "已有 operation 收据缺少或篡改了 tool/request_fingerprint/request",
                        "tool": previous_tool,
                        "request_fingerprint": previous_fingerprint,
                    },
                }
            if (
                previous_tool != tool_name
                or previous_fingerprint != fingerprint
            ):
                job_id = previous.get("job_id")
                return {
                    "status": "technical_pause",
                    "minichem_status": "operation_id_conflict",
                    "minichem_envelope_status": "invalid_request",
                    "operation_id": operation,
                    "job_id": job_id,
                    "error": {
                        "code": "operation_id_conflict",
                        "message": (
                            "同一 operation_id 的 MiniChem tool 或请求指纹不一致，"
                            "拒绝重复副作用"
                        ),
                        "job_id": job_id,
                        "tool": previous_tool,
                        "request_fingerprint": previous_fingerprint,
                    },
                }
            return deepcopy(previous)
        payload["operation_id"] = operation
        if tool_name == "submit_native_job":
            # 该字段由 MiniChem 协议层接收并持久化，但不属于 Planner 的公开
            # NativeJobRequest；它让缺失本地 receipt 时能核对 caller 请求指纹。
            payload[_CALLER_REQUEST_FINGERPRINT_FIELD] = fingerprint
        try:
            # native submit 的响应丢失后只探测原 job，不把同一副作用请求交给
            # adapter 的通用瞬时重试；服务端已有 operation 身份，恢复失败时由
            # 调用方保存技术暂停，不能再盲目发起第二次 submit。
            result = self._call_tool(
                tool_name,
                payload,
                retry_safe=tool_name != "submit_native_job",
            )
        except Exception as submit_error:
            if tool_name != "submit_native_job":
                raise
            try:
                recovered = self.recover_native_operation(
                    operation,
                    request_fingerprint=fingerprint,
                    collect=False,
                )
            except Exception as recovery_error:
                self._persist_ambiguous_native_operation_receipt(
                    operation,
                    payload=payload,
                    request_fingerprint=fingerprint,
                    result=None,
                    error=recovery_error,
                )
                raise MiniChemMcpError(
                    "submit_native_job 传输异常，且只读 operation recovery 未能证明原作业；"
                    "已阻止重复提交"
                ) from recovery_error
            if recovered.get("status") not in {"waiting", "completed"}:
                raise MiniChemMcpError(
                    "submit_native_job 传输异常后 recovery 未返回 waiting/completed；"
                    "已阻止重复提交"
                ) from submit_error
            return self._persist_recovered_native_operation_receipt(
                operation,
                payload=payload,
                request_fingerprint=fingerprint,
                result=recovered,
            )
        result = _validate_artifact_records(result)
        normalized = self._normalize_result(
            result,
            operation_id=operation,
            result_kind="job_submit",
        )
        normalized["request_fingerprint"] = fingerprint
        normalized["operation_fingerprint"] = fingerprint
        normalized["tool"] = tool_name
        normalized["request"] = deepcopy(payload)
        normalized["request"].pop(_CALLER_REQUEST_FINGERPRINT_FIELD, None)
        if (
            tool_name == "submit_native_job"
            and (
                str(result.get("status", "")).lower() == "submission_in_progress"
                or (
                    isinstance(result.get("error"), Mapping)
                    and result["error"].get("code") == "operation_record_incomplete"
                )
            )
        ):
            try:
                recovered = self.recover_native_operation(
                    operation,
                    request_fingerprint=fingerprint,
                    collect=False,
                )
            except Exception as recovery_error:
                normalized["recovery_error"] = {
                    "code": "native_operation_recovery_failed",
                    "message": str(recovery_error),
                }
                self._persist_ambiguous_native_operation_receipt(
                    operation,
                    payload=payload,
                    request_fingerprint=fingerprint,
                    result=normalized,
                    error=recovery_error,
                )
            else:
                if recovered.get("status") in {"waiting", "completed"}:
                    return self._persist_recovered_native_operation_receipt(
                        operation,
                        payload=payload,
                        request_fingerprint=fingerprint,
                        result=recovered,
                    )
        # 只有服务明确返回 success，且适配器能把结果归一为 waiting/completed、
        # 同时拿到合法 job_id 时才建立可恢复 operation 收据。invalid_request、
        # 错误 envelope 或未知 job_status 都不能写成“已提交”，否则后续恢复会
        # 永久阻断安全重试。
        envelope_status = str(result.get("status", "")).lower()
        if (
            envelope_status == "success"
            and normalized.get("status") in {"waiting", "completed"}
            and isinstance(normalized.get("job_id"), str)
            and _JOB_ID.fullmatch(normalized["job_id"])
        ):
            self._operations[operation] = deepcopy(normalized)
            self._persist_operation(operation, normalized)
        return normalized

    def _native_pending_recovery_pause(
        self,
        operation_id: str,
        *,
        previous: Mapping[str, Any],
        request_fingerprint: str,
        error: Exception,
    ) -> dict[str, Any]:
        """返回 pending operation 的只读恢复暂停结果，不重新发起 submit。"""

        result: dict[str, Any] = {
            "status": "technical_pause",
            "minichem_status": "native_operation_recovery_pending",
            "minichem_envelope_status": "technical_pause",
            "operation_id": operation_id,
            "job_id": previous.get("job_id"),
            "tool": "submit_native_job",
            "request_fingerprint": request_fingerprint,
            "operation_fingerprint": request_fingerprint,
            "error": {
                "code": "native_operation_recovery_pending",
                "message": (
                    "native submit 已有未能核对的 pending receipt；"
                    "本次只执行了原 job 的只读 recovery，未再次 submit"
                ),
                "recovery_error": str(error),
            },
        }
        saved_request = previous.get("request")
        if isinstance(saved_request, Mapping):
            result["request"] = deepcopy(dict(saved_request))
        return result

    def _persist_ambiguous_native_operation_receipt(
        self,
        operation_id: str,
        *,
        payload: Mapping[str, Any],
        request_fingerprint: str,
        result: Mapping[str, Any] | None,
        error: Exception,
    ) -> dict[str, Any]:
        """保存无法证明远端身份的 native pending receipt，阻止后续重复 submit。"""

        clean_request = deepcopy(dict(payload))
        clean_request.pop(_CALLER_REQUEST_FINGERPRINT_FIELD, None)
        job_id = self._select_ambiguous_native_job_id(operation_id, result)
        receipt: dict[str, Any] = {
            "schema_version": 1,
            "receipt_kind": _AMBIGUOUS_NATIVE_RECEIPT_KIND,
            "operation_id": operation_id,
            "job_id": job_id,
            "tool": "submit_native_job",
            "request_fingerprint": request_fingerprint,
            "operation_fingerprint": request_fingerprint,
            "request": clean_request,
            "pending_error": {
                "code": "native_operation_recovery_failed",
                "message": str(error),
            },
        }
        self._operations[operation_id] = deepcopy(receipt)
        self._persist_operation(operation_id, receipt)
        return receipt

    @staticmethod
    def _select_ambiguous_native_job_id(
        operation_id: str,
        result: Mapping[str, Any] | None,
    ) -> str | None:
        """选择可继续核对的 job_id；冲突或算法不可用时不猜测作业。"""

        reported_job_id = result.get("job_id") if isinstance(result, Mapping) else None
        if reported_job_id is not None:
            try:
                reported_job_id = _validate_job_id(reported_job_id)
            except ValueError:
                reported_job_id = None
        try:
            derived_job_id = _derive_minichem_operation_job_id(operation_id)
        except MiniChemMcpError:
            derived_job_id = None
        if (
            reported_job_id is not None
            and derived_job_id is not None
            and reported_job_id != derived_job_id
        ):
            return None
        return reported_job_id or derived_job_id

    def _recover_ambiguous_native_operation(
        self,
        operation_id: str,
        saved: Mapping[str, Any],
        *,
        request_fingerprint: str,
        collect: bool,
    ) -> dict[str, Any]:
        """按 pending receipt 只读恢复同一 native job，不能转回 submit。"""

        if saved.get("operation_id") != operation_id:
            raise MiniChemMcpProtocolError(
                f"operation {operation_id} pending receipt 的 operation_id 不一致"
            )
        if saved.get("tool") != "submit_native_job":
            raise MiniChemMcpProtocolError(
                f"operation {operation_id} pending receipt 的 tool 不一致"
            )
        if saved.get("request_fingerprint") != request_fingerprint:
            raise MiniChemMcpProtocolError(
                f"operation {operation_id} pending receipt request_fingerprint 与调用方不一致"
            )
        if saved.get("operation_fingerprint") != request_fingerprint:
            raise MiniChemMcpProtocolError(
                f"operation {operation_id} pending receipt operation_fingerprint 不一致"
            )
        saved_request = saved.get("request")
        if not isinstance(saved_request, Mapping):
            raise MiniChemMcpProtocolError(
                f"operation {operation_id} pending receipt 缺少可核对的 request"
            )
        calculated_fingerprint = _job_request_fingerprint(
            "submit_native_job", saved_request
        )
        if calculated_fingerprint != request_fingerprint:
            raise MiniChemMcpProtocolError(
                f"operation {operation_id} pending receipt request_fingerprint 与 request 不一致"
            )
        job_id = saved.get("job_id")
        if job_id is None:
            job_id = _derive_minichem_operation_job_id(operation_id)
        else:
            _validate_job_id(job_id)
        derived_job_id = _derive_minichem_operation_job_id(operation_id)
        if job_id != derived_job_id:
            raise MiniChemMcpProtocolError(
                f"operation {operation_id} pending receipt 的 job_id 与 operation→job 映射不一致"
            )
        recovered = self._recover_native_operation_from_remote_job(
            operation_id,
            job_id=job_id,
            request_fingerprint=request_fingerprint,
            collect=collect,
        )
        # pending receipt 保留了原始 request；远端身份一旦补齐，就把最小探测
        # receipt 升级为完整 receipt，后续 resume 仍可核对同一请求，不会落入
        # “只有 job_id、无法恢复请求”的技术暂停分支。
        return self._persist_recovered_native_operation_receipt(
            operation_id,
            payload=saved_request,
            request_fingerprint=request_fingerprint,
            result=recovered,
        )

    def _persist_recovered_native_operation_receipt(
        self,
        operation_id: str,
        *,
        payload: Mapping[str, Any],
        request_fingerprint: str,
        result: Mapping[str, Any],
    ) -> dict[str, Any]:
        """把已证明身份的只读恢复结果补齐为可继续核对的本地 receipt。"""

        recovered = deepcopy(dict(result))
        recovered["operation_id"] = operation_id
        recovered["job_id"] = _validate_job_id(recovered.get("job_id"))
        recovered["tool"] = "submit_native_job"
        recovered["request_fingerprint"] = request_fingerprint
        recovered["operation_fingerprint"] = request_fingerprint
        recovered["request"] = deepcopy(dict(payload))
        recovered["request"].pop(_CALLER_REQUEST_FINGERPRINT_FIELD, None)
        self._operations[operation_id] = deepcopy(recovered)
        self._persist_operation(operation_id, recovered)
        return recovered

    @contextmanager
    def _operation_submission_lock(self, operation_id: str):
        """串行化同一 operation 的提交，避免多个进程重复产生远端副作用。"""

        with self._submission_guard:
            if self.operation_root is None:
                yield
                return
            try:
                self.operation_root.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise MiniChemMcpError(
                    f"operation 收据目录不可创建：{self.operation_root}"
                ) from exc
            directory_fd = _open_operation_directory(self.operation_root)
            try:
                descriptor = _open_operation_regular_file(
                    directory_fd,
                    f".{operation_id}.lock",
                    flags=os.O_RDWR | os.O_CREAT,
                    mode=0o600,
                    label="operation 锁文件",
                )
                if descriptor is None:  # pragma: no cover - O_CREAT 下不会发生
                    raise MiniChemMcpError(f"operation 锁文件不可打开：{operation_id}")
                locked = False
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX)
                    locked = True
                    yield
                finally:
                    if locked:
                        fcntl.flock(descriptor, fcntl.LOCK_UN)
                    os.close(descriptor)
            finally:
                os.close(directory_fd)

    def get_execution_job(self, job_id: str) -> dict[str, Any]:
        """查询同一 job 的状态，不创建新的副作用。"""

        _validate_job_id(job_id)
        result = self._call_tool("get_execution_job", {"job_id": job_id})
        _validate_response_job_identity(result, job_id, tool_name="get_execution_job")
        result = _validate_artifact_records(result)
        return self._normalize_result(result, job_id=job_id, result_kind="job_get")

    def collect_execution_job(self, job_id: str, *, include_inputs: bool = False) -> dict[str, Any]:
        """收集终态 job 的文件清单和哈希。"""

        _validate_job_id(job_id)
        result = self._call_tool(
            "collect_execution_job",
            {"job_id": job_id, "include_inputs": include_inputs},
        )
        _validate_response_job_identity(result, job_id, tool_name="collect_execution_job")
        result = _validate_artifact_records(result)
        result = _validate_job_file_records(result)
        return self._normalize_result(result, job_id=job_id, result_kind="job_collect")

    def cancel_execution_job(self, job_id: str) -> dict[str, Any]:
        """取消指定 job，不提交替代作业。"""

        _validate_job_id(job_id)
        # cancel 会向 supervisor 发送终止信号，但当前公开合同没有 operation
        # 身份。若响应在信号生效后丢失，重发会再次产生副作用，因此这里只允许
        # 一次传输尝试；失败由恢复层作为技术暂停处理。
        result = self._call_tool(
            "cancel_execution_job",
            {"job_id": job_id},
            retry_safe=False,
        )
        _validate_response_job_identity(result, job_id, tool_name="cancel_execution_job")
        return self._normalize_result(
            result,
            job_id=job_id,
            result_kind="job_cancel",
        )

    def declare_scientific_artifact(
        self,
        request: Mapping[str, Any],
        *,
        operation_id: str | None = None,
    ) -> dict[str, Any]:
        """登记 ArtifactRef；内部身份存在时允许安全重试并回放声明。"""

        payload = _validate_artifact_declaration_mapping(request)
        if operation_id is not None:
            payload["operation_id"] = _validate_operation_id(operation_id)
        result = _validate_artifact_records(
            self._call_tool(
                "declare_scientific_artifact",
                payload,
                retry_safe=operation_id is not None,
            ),
        )
        if str(result.get("status", "")).lower() in _ACTION_SUCCESS_STATUSES:
            artifact_values: list[Any] = []
            if "artifact" in result:
                artifact_values.append(result["artifact"])
            output_artifacts = result.get("output_artifacts")
            if isinstance(output_artifacts, list):
                artifact_values.extend(output_artifacts)
            if not artifact_values:
                raise MiniChemMcpProtocolError(
                    "declare_scientific_artifact 成功响应缺少有效 ArtifactRef"
                )
        if "artifact" in result and "output_artifacts" not in result:
            # ARCHE 后续 Program Gate 统一读取 output_artifacts；保留原始
            # 顶层 artifact 的同时补出同一引用，不复制或改写 ArtifactRef。
            result["output_artifacts"] = [deepcopy(result["artifact"])]
        return self._normalize_result(result, result_kind="action")

    @staticmethod
    def _remote_operation_identity_value(
        result: Mapping[str, Any],
        field: str,
    ) -> Any:
        """从 get 响应顶层和被查询 job 记录读取同一身份字段。

        只读取响应主体和 ``job``，不把资源概览中的其他作业身份当作当前作业
        证据。两个位置都返回字段时必须相同，否则响应本身已经发生漂移。
        """

        job = result.get("job")
        mappings = [result]
        if isinstance(job, Mapping):
            mappings.append(job)
        values = [mapping[field] for mapping in mappings if field in mapping]
        if not values:
            return None
        first = values[0]
        if any(value != first for value in values[1:]):
            raise MiniChemMcpProtocolError(
                f"get_execution_job 返回的 {field} 在响应层级间不一致"
            )
        return first

    def _validate_remote_native_recovery_identity(
        self,
        result: Mapping[str, Any],
        *,
        operation_id: str,
        job_id: str,
        request_fingerprint: str,
    ) -> None:
        """核对远端 get 响应是否足以证明 native operation/request 身份。

        MiniChem 的恢复证据由 ``operation_id``、确定性 ``job_id``、唯一的
        ``job_type=native_software`` 和 adapter 传入的隐藏 caller fingerprint 组成。
        旧服务或中间代理缺少其中任一字段时，确定性 ``job_id`` 只能定位作业，不能
        据此补写本地收据；适配器立即拒绝恢复，不把作业状态伪造成 waiting 或 completed。
        """

        job = result.get("job")
        if not isinstance(job, Mapping):
            raise MiniChemMcpProtocolError(
                "get_execution_job 缺少可核对的 job 身份，拒绝恢复 native operation"
            )
        returned_operation = self._remote_operation_identity_value(
            result, "operation_id"
        )
        if returned_operation != operation_id:
            raise MiniChemMcpProtocolError(
                "远端 native job 的 operation_id 与待恢复 operation 不一致"
            )
        returned_job = self._remote_operation_identity_value(result, "job_id")
        if returned_job != job_id:
            raise MiniChemMcpProtocolError(
                "远端 native job 的 job_id 与 operation→job 映射不一致"
            )
        returned_tool = self._remote_operation_identity_value(result, "tool")
        returned_job_type = self._remote_operation_identity_value(result, "job_type")
        if returned_tool is not None:
            if returned_tool != "submit_native_job":
                raise MiniChemMcpProtocolError(
                    "远端 native job 的 tool 不是 submit_native_job，拒绝补写 operation 收据"
                )
            if returned_job_type is not None and returned_job_type != "native_software":
                raise MiniChemMcpProtocolError(
                    "远端 job 的 tool 与 job_type 不一致，拒绝补写 operation 收据"
                )
        else:
            # 当前协议没有 tool 字段；协议扩展至少可以用唯一的 native
            # job_type 证明该作业只能由 submit_native_job 创建。
            if returned_job_type != "native_software":
                raise MiniChemMcpProtocolError(
                    "MiniChem get_execution_job 未返回可核对的 native job/tool=submit_native_job；"
                    "确定性 job_id 只能定位作业，拒绝补写 operation 收据"
                )
        returned_fingerprints = [
            value
            for value in (
                self._remote_operation_identity_value(result, "request_fingerprint"),
                self._remote_operation_identity_value(
                    result, "caller_request_fingerprint"
                ),
            )
            if value is not None
        ]
        if returned_fingerprints and any(
            value != returned_fingerprints[0] for value in returned_fingerprints[1:]
        ):
            raise MiniChemMcpProtocolError(
                "get_execution_job 返回了互相冲突的 request fingerprint，"
                "拒绝补写 operation 收据"
            )
        returned_fingerprint = returned_fingerprints[0] if returned_fingerprints else None
        if returned_fingerprint != request_fingerprint:
            if returned_fingerprint is None:
                message = (
                    "MiniChem get_execution_job 未返回可核对的 request_fingerprint；"
                    "确定性 job_id 只能定位作业，拒绝补写 operation 收据"
                )
            else:
                message = "远端 native job 的 request_fingerprint 与待恢复请求不一致"
            raise MiniChemMcpProtocolError(message)

    def _recover_native_operation_from_remote_job(
        self,
        operation_id: str,
        *,
        job_id: str,
        request_fingerprint: str,
        collect: bool,
    ) -> dict[str, Any]:
        """只读探测确定性 job，并在身份完整时归一化其状态。

        本方法不会调用 submit，也不会在身份不足时写入收据。它保留 get/collect
        的原始协议错误；只有服务明确返回 waiting/completed 且 operation、job、tool
        和 request fingerprint 全部可核对时才允许继续。
        """

        latest = self.get_execution_job(job_id)
        if latest.get("status") not in {"waiting", "completed"}:
            error = latest.get("error")
            code = error.get("code") if isinstance(error, Mapping) else None
            detail = code or latest.get("minichem_status") or "unknown"
            raise MiniChemMcpProtocolError(
                f"远端 native job {job_id} 不可恢复：{detail}"
            )
        self._validate_remote_native_recovery_identity(
            latest,
            operation_id=operation_id,
            job_id=job_id,
            request_fingerprint=request_fingerprint,
        )
        if collect and latest.get("status") == "completed":
            latest = self.collect_execution_job(job_id)
        latest["operation_id"] = operation_id
        latest["job_id"] = job_id
        latest["tool"] = "submit_native_job"
        latest["request_fingerprint"] = request_fingerprint
        latest["operation_fingerprint"] = request_fingerprint
        # 只有上面的远端身份核对全部通过后，才允许补写最小 recovery receipt。
        # 缺少原始 request 时不伪造 request object；后续 resume 会继续要求同一
        # 远端身份证明，普通 submit 也会因 receipt 不完整而安全停住。
        receipt = {
            "schema_version": 1,
            "receipt_kind": "minimal_native_recovery",
            "operation_id": operation_id,
            "job_id": job_id,
            "tool": "submit_native_job",
            "request_fingerprint": request_fingerprint,
            "operation_fingerprint": request_fingerprint,
        }
        self._operations[operation_id] = deepcopy(receipt)
        self._persist_operation(operation_id, receipt)
        return latest

    def recover_native_operation(
        self,
        operation_id: str,
        *,
        request_fingerprint: str,
        collect: bool = True,
    ) -> dict[str, Any]:
        """恢复 native operation 的 waiting/completed 状态且绝不重复提交。

        输入是原 operation identity、已由调用方从 request/tool 重算的 SHA-256 指纹
        和是否收集终态文件。若本地收据完整，动作只调用原 job 的 get/collect；若收据
        缺失，则按 MiniChem 公开确定性算法探测同一 job，并要求 get 响应同时证明
        operation_id、job_id、``submit_native_job`` 和 request_fingerprint。当前服务
        旧服务未公开 native 类型或 caller fingerprint 时抛出协议异常，不写猜测收据、
        不调用 submit；带扩展证据的服务响应才允许补写最小收据。收据损坏、漂移或指纹
        不一致同样立即失败，供执行器保存可恢复技术暂停。
        """

        operation = _validate_operation_id(operation_id)
        if (
            not isinstance(request_fingerprint, str)
            or _SHA256.fullmatch(request_fingerprint) is None
        ):
            raise MiniChemMcpProtocolError(
                "native operation recovery request_fingerprint 必须是 SHA-256"
            )
        saved = self._operations.get(operation) or self._load_operation(operation)
        if saved is not None:
            if not isinstance(saved, Mapping):
                raise MiniChemMcpProtocolError(
                    f"operation {operation} 收据不是 object，拒绝恢复"
                )
            if saved.get("receipt_kind") == _AMBIGUOUS_NATIVE_RECEIPT_KIND:
                # pending receipt 只允许沿确定性 job 身份执行 get/collect；不能
                # 交给普通 resume 分支，因为它无法从 pending 记录推导出“已提交”。
                return self._recover_ambiguous_native_operation(
                    operation,
                    saved,
                    request_fingerprint=request_fingerprint,
                    collect=collect,
                )
            if saved.get("tool") != "submit_native_job":
                raise MiniChemMcpProtocolError(
                    f"operation {operation} 收据 tool 不是 submit_native_job，拒绝恢复"
                )
            if saved.get("request_fingerprint") != request_fingerprint:
                raise MiniChemMcpProtocolError(
                    f"operation {operation} 收据 request_fingerprint 与调用方不一致"
                )
            if saved.get("operation_fingerprint") != request_fingerprint:
                raise MiniChemMcpProtocolError(
                    f"operation {operation} 收据 operation_fingerprint 与调用方不一致"
                )
            # resume_operation 会再次核对 operation_id、request、tool 和 job_id，
            # 然后只读取原 job；这里的前置核对保证 native 入口不会误接收 analysis receipt。
            return self.resume_operation(operation, collect=collect)

        job_id = _derive_minichem_operation_job_id(operation)
        return self._recover_native_operation_from_remote_job(
            operation,
            job_id=job_id,
            request_fingerprint=request_fingerprint,
            collect=collect,
        )

    def resume_operation(self, operation_id: str, *, collect: bool = True) -> dict[str, Any]:
        """恢复已有 operation，只查询/收集保存的 job，不重新 submit。"""

        operation = _validate_operation_id(operation_id)
        saved = self._operations.get(operation) or self._load_operation(operation)
        if not isinstance(saved, Mapping):
            raise MiniChemMcpError(f"没有 operation {operation} 的持久身份")
        if saved.get("receipt_kind") == _AMBIGUOUS_NATIVE_RECEIPT_KIND:
            receipt_fingerprint = saved.get("request_fingerprint")
            if (
                not isinstance(receipt_fingerprint, str)
                or _SHA256.fullmatch(receipt_fingerprint) is None
            ):
                raise MiniChemMcpProtocolError(
                    f"operation {operation} pending receipt 缺少可核对的 request_fingerprint"
                )
            return self._recover_ambiguous_native_operation(
                operation,
                saved,
                request_fingerprint=receipt_fingerprint,
                collect=collect,
            )
        receipt_tool = saved.get("tool")
        receipt_fingerprint = saved.get("request_fingerprint")
        if (
            not isinstance(receipt_tool, str)
            or receipt_tool not in _JOB_SUBMIT_TOOLS
            or not isinstance(receipt_fingerprint, str)
            or (
                _SHA256.fullmatch(receipt_fingerprint) is None
            )
        ):
            raise MiniChemMcpProtocolError(
                f"operation {operation} 收据缺少可核对的 tool/request_fingerprint"
            )
        saved_request = saved.get("request")
        if saved_request is None and saved.get("receipt_kind") == "minimal_native_recovery":
            if saved.get("operation_id") != operation:
                raise MiniChemMcpProtocolError(
                    f"operation {operation} minimal recovery 收据中的 operation_id 不一致"
                )
            if saved.get("tool") != "submit_native_job":
                raise MiniChemMcpProtocolError(
                    f"operation {operation} minimal recovery 收据 tool 不一致"
                )
            job_id = saved.get("job_id")
            _validate_job_id(job_id)
            latest = self.get_execution_job(job_id)
            if latest.get("status") in {"waiting", "completed"}:
                self._validate_remote_native_recovery_identity(
                    latest,
                    operation_id=operation,
                    job_id=job_id,
                    request_fingerprint=receipt_fingerprint,
                )
                if collect and latest.get("status") == "completed":
                    latest = self.collect_execution_job(job_id)
            latest["operation_id"] = operation
            latest["job_id"] = job_id
            latest["tool"] = "submit_native_job"
            latest["request_fingerprint"] = receipt_fingerprint
            latest["operation_fingerprint"] = receipt_fingerprint
            self._operations[operation] = deepcopy(dict(saved))
            self._persist_operation(operation, saved)
            return latest
        if not isinstance(saved_request, Mapping):
            raise MiniChemMcpProtocolError(
                f"operation {operation} 收据缺少可核对的原始 request"
            )
        calculated_fingerprint = _job_request_fingerprint(receipt_tool, saved_request)
        if calculated_fingerprint != receipt_fingerprint:
            raise MiniChemMcpProtocolError(
                f"operation {operation} 收据 request_fingerprint 与 request/tool 不一致"
            )
        operation_fingerprint = saved.get("operation_fingerprint")
        if operation_fingerprint != receipt_fingerprint:
            raise MiniChemMcpProtocolError(
                f"operation {operation} 收据 operation_fingerprint 不一致"
            )
        if saved.get("operation_id") != operation:
            raise MiniChemMcpProtocolError(
                f"operation {operation} 收据中的 operation_id 不一致"
            )
        job_id = saved.get("job_id")
        _validate_job_id(job_id)
        latest = self.get_execution_job(job_id)
        status = latest.get("status")
        if collect and status == "completed":
            latest = self.collect_execution_job(job_id)
        latest["operation_id"] = operation
        latest["tool"] = receipt_tool
        latest["request_fingerprint"] = receipt_fingerprint
        latest["operation_fingerprint"] = receipt_fingerprint
        latest["request"] = deepcopy(saved_request)
        self._operations[operation] = deepcopy(latest)
        self._persist_operation(operation, latest)
        return latest

    def wait_for_minichem_operation_completion(
        self,
        operation_id: str,
        *,
        timeout_seconds: float = 600.0,
        collect: bool = True,
    ) -> dict[str, Any]:
        """轮询同一 operation 直到完成或形成技术暂停，不重新提交作业。

        ``resume_operation`` 一次只读取一个状态，供工作流在 checkpoint 边界恢复；
        需要适配器自行等待时调用本方法。超时保留同一 operation/job 身份并返回
        ``technical_pause``，不会把 waiting 误报为 completed。
        """

        if timeout_seconds < 0:
            raise ValueError("timeout_seconds 必须是非负数")
        started = time.monotonic()
        latest = self.resume_operation(operation_id, collect=collect)
        while latest.get("status") == "waiting":
            if time.monotonic() - started >= timeout_seconds:
                latest = deepcopy(latest)
                latest["status"] = "technical_pause"
                latest["minichem_status"] = "poll_timeout"
                latest["minichem_envelope_status"] = "timeout"
                latest["error"] = {
                    "code": "poll_timeout",
                    "message": "MiniChem operation 在限定时间内仍未完成",
                    "operation_id": latest.get("operation_id", operation_id),
                    "job_id": latest.get("job_id"),
                }
                return latest
            if self.poll_interval:
                time.sleep(self.poll_interval)
            latest = self.resume_operation(operation_id, collect=collect)
        return latest

    def _normalize_result(
        self,
        value: Mapping[str, Any],
        *,
        operation_id: str | None = None,
        action_id: str | None = None,
        job_id: str | None = None,
        result_kind: str = "action",
    ) -> dict[str, Any]:
        """按 Action 或 job 的真实 envelope 映射为 completed/waiting/technical_pause。"""

        if not isinstance(value, Mapping):
            raise MiniChemMcpProtocolError("MiniChem 结果不是 object")
        result = deepcopy(dict(value))
        envelope_status = str(result.get("status", "")).lower()
        raw_status = envelope_status
        status = "technical_pause"

        if result_kind == "action":
            if envelope_status in _ACTION_SUCCESS_STATUSES:
                status = "completed"
            else:
                status = "technical_pause"
        elif result_kind == "job_submit":
            job_id_value = result.get("job_id")
            if isinstance(job_id_value, str):
                _validate_job_id(job_id_value)
            elif envelope_status == "success" and job_id is None:
                raise MiniChemMcpProtocolError("job submit 响应缺少合法 job_id")
            job_status = str(result.get("job_status", "")).lower()
            raw_status = job_status or envelope_status
            if envelope_status == "success" and job_status in _WAITING_STATUSES:
                status = "waiting"
            elif envelope_status == "success" and job_status in _JOB_TERMINAL_SUCCESS_STATUSES:
                status = "completed"
            else:
                status = "technical_pause"
        elif result_kind == "job_get":
            job = result.get("job")
            if not isinstance(job, Mapping):
                if envelope_status == "success":
                    raise MiniChemMcpProtocolError("get_execution_job 响应缺少嵌套 job 对象")
                job = {}
            nested_job_id = job.get("job_id")
            if nested_job_id is not None:
                _validate_job_id(nested_job_id)
                if job_id is not None and nested_job_id != job_id:
                    raise MiniChemMcpProtocolError(
                        f"get_execution_job 返回了错误的 job_id：{nested_job_id}"
                    )
            job_status = str(job.get("status", result.get("job_status", ""))).lower()
            raw_status = job_status or envelope_status
            # MiniChem get envelope 把 terminal 放在响应顶层，job 只承载状态和过程信息。
            terminal = result.get("terminal")
            if envelope_status == "success" and job_status in _JOB_TERMINAL_FAILURE_STATUSES:
                status = "technical_pause"
            elif (
                envelope_status == "success"
                and job_status in _JOB_TERMINAL_SUCCESS_STATUSES
                and terminal is not False
            ):
                status = "completed"
            elif (
                envelope_status == "success"
                and terminal is not True
                and job_status in _WAITING_STATUSES
            ):
                status = "waiting"
            elif (
                envelope_status == "success"
                and terminal is False
                and job_status in _WAITING_STATUSES
            ):
                status = "waiting"
            else:
                status = "technical_pause"
        elif result_kind == "job_collect":
            ready = result.get("ready")
            job = result.get("job")
            job_status = str(
                result.get(
                    "job_status",
                    job.get("status", "") if isinstance(job, Mapping) else "",
                )
            ).lower()
            nested_job_id = job.get("job_id") if isinstance(job, Mapping) else None
            if nested_job_id is not None:
                _validate_job_id(nested_job_id)
                if job_id is not None and nested_job_id != job_id:
                    raise MiniChemMcpProtocolError(
                        f"collect_execution_job 返回了错误的 job_id：{nested_job_id}"
                    )
            raw_status = job_status or envelope_status
            # ready=true 只是说明 collect 已到达终态，不代表终态成功；失败、
            # 超时和取消必须保留技术暂停，不能被误报为 completed。
            if envelope_status == "success" and job_status in _JOB_TERMINAL_FAILURE_STATUSES:
                status = "technical_pause"
            elif (
                envelope_status == "success"
                and ready is True
                and job_status in _JOB_TERMINAL_SUCCESS_STATUSES
            ):
                status = "completed"
            elif (
                envelope_status == "success"
                and ready is False
                and job_status in _WAITING_STATUSES
            ):
                status = "waiting"
            else:
                status = "technical_pause"
        elif result_kind == "job_cancel":
            cancellation_sent = result.get("cancellation_sent")
            job_status = str(result.get("job_status", "")).lower()
            raw_status = "cancellation_sent" if cancellation_sent is True else job_status or envelope_status
            if (
                envelope_status == "success"
                and cancellation_sent is False
                and job_status in _JOB_TERMINAL_SUCCESS_STATUSES
            ):
                status = "completed"
            else:
                # cancel 成功只代表已发出取消信号；仍需继续查询原 job。
                status = "technical_pause"
        else:
            raise MiniChemMcpProtocolError(f"未知 MiniChem 结果类型：{result_kind}")

        result["status"] = status
        result["minichem_status"] = raw_status
        result["minichem_envelope_status"] = envelope_status
        if operation_id is not None:
            result["operation_id"] = operation_id
        if action_id is not None:
            result.setdefault("action_id", action_id)
        if job_id is not None:
            result.setdefault("job_id", job_id)
        if status == "waiting" and result_kind != "job_get" and not result.get("job_id"):
            raise MiniChemMcpProtocolError("waiting job 结果缺少合法 job_id")
        if status == "technical_pause":
            error = result.get("error")
            if not isinstance(error, Mapping):
                diagnostic = result.get("failure_diagnostic")
                diagnostic_error: dict[str, Any] | None = None
                if isinstance(diagnostic, Mapping):
                    classification = diagnostic.get("classification")
                    message = diagnostic.get("message")
                    if (
                        isinstance(classification, str)
                        and re.fullmatch(r"[a-z][a-z0-9_.-]{0,127}", classification)
                        and isinstance(message, str)
                        and message.strip()
                    ):
                        diagnostic_error = {
                            "code": classification,
                            "message": message.strip()[:1000],
                        }
                        details = diagnostic.get("details")
                        if isinstance(details, Mapping):
                            safe_details = {
                                str(key): value
                                for key, value in details.items()
                                if key in _SAFE_FAILURE_DIAGNOSTIC_DETAIL_FIELDS
                                and isinstance(value, (str, int, float, bool))
                            }
                            if safe_details:
                                diagnostic_error["details"] = safe_details
                        candidate_fixes = diagnostic.get("candidate_fixes")
                        if isinstance(candidate_fixes, list):
                            safe_fixes = [
                                item.strip()[:500]
                                for item in candidate_fixes[:8]
                                if isinstance(item, str) and item.strip()
                            ]
                            if safe_fixes:
                                diagnostic_error["candidate_fixes"] = safe_fixes
                result["error"] = diagnostic_error or {
                    "code": raw_status or "minichem_protocol_error",
                    "message": "MiniChem 未返回可用的结构化错误；请查看原始响应",
                }
        return result

    def _persist_operation(self, operation_id: str, value: Mapping[str, Any]) -> None:
        """在可选任务目录中原子保存 operation/job 身份，供进程恢复。"""

        if self.operation_root is None:
            return
        self.operation_root.mkdir(parents=True, exist_ok=True)
        target = self.operation_root / f"{operation_id}.json"
        if target.is_symlink():
            raise MiniChemMcpError("operation 收据文件不得是符号链接")
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        if temporary.exists() or temporary.is_symlink():
            raise MiniChemMcpError("operation 收据临时文件已存在")
        payload = (_canonical_json(dict(value)) + "\n").encode("utf-8")
        try:
            with temporary.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            try:
                directory_fd = os.open(self.operation_root, directory_flags)
            except OSError as exc:
                raise MiniChemMcpError(
                    f"operation 收据父目录无法打开并同步：{self.operation_root}"
                ) from exc
            try:
                os.fsync(directory_fd)
            except OSError as exc:
                raise MiniChemMcpError(
                    f"operation 收据父目录同步失败：{self.operation_root}"
                ) from exc
            finally:
                os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)

    def _load_operation(self, operation_id: str) -> dict[str, Any] | None:
        """读取同一任务目录中已保存的 operation 记录。"""

        if self.operation_root is None:
            return None
        directory_fd = _open_operation_directory(self.operation_root)
        descriptor = -1
        try:
            opened_descriptor = _open_operation_regular_file(
                directory_fd,
                f"{operation_id}.json",
                flags=os.O_RDONLY | getattr(os, "O_NONBLOCK", 0),
                mode=0,
                label="operation 收据文件",
                missing_is_none=True,
            )
            if opened_descriptor is None:
                return None
            descriptor = opened_descriptor
            try:
                with os.fdopen(descriptor, "rb") as handle:
                    descriptor = -1
                    raw = handle.read()
            except OSError as exc:
                raise MiniChemMcpError(f"operation 记录不可读：{operation_id}") from exc
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MiniChemMcpError(f"operation 记录不可读：{operation_id}") from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            os.close(directory_fd)
        if not isinstance(value, Mapping):
            raise MiniChemMcpError(f"operation 记录不是 object：{operation_id}")
        return deepcopy(dict(value))


__all__ = [
    "MiniChemActionRequest",
    "MiniChemCatalogSnapshot",
    "MiniChemMcpAdapter",
    "MiniChemMcpError",
    "MiniChemMcpProtocolError",
]
