"""Investigation 专用 MiniChem 只读网关。

本模块用一个小接口统一提供 Investigation 需要的两项 MiniChem 只读能力：
按 scope 建立带 ``X-MiniChem-Capability`` 的 Streamable HTTP MCP 会话，核验
结果中保存的 job/operation/artifact 不透明标识，并把已核验的 UTF-8 artifact
分页重构到 Evidence source cache 后交给 ``ReviewEvidenceStore``。模块从不读取
MiniChem workspace、ArtifactStore 或宿主路径；宿主路径只会在写入调用方明确传入
的 source cache 时出现。

同步和异步 MCP session factory 都可以注入。真实运行路径使用 MCP 官方
Streamable HTTP transport；测试替身只需实现 ``initialize``、``call_tool`` 和
可选的 ``close``，不需要依赖 MiniChem 的内部实现。
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import inspect
import json
import os
import re
import stat
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Protocol, TypeAlias
from urllib.parse import urlsplit

from .domain import InvestigationResult, InvestigationSession, InvestigationState
from .evidence import BatchEvidenceSnapshot, EvidenceArtifactManifest, ReviewEvidenceStore
from ..memory.contracts import ContractValidationError
from ..memory.investigation_writer import VerifiedInvestigationEvidence


__all__ = [
    "InvestigationMiniChemGateway",
    "InvestigationMiniChemGatewayError",
    "InvestigationMiniChemProtocolError",
    "InvestigationMiniChemReferenceRejected",
    "InvestigationMiniChemScopeError",
    "InvestigationMiniChemUnavailable",
    "InvestigationScopeBinding",
    "InvestigationMcpSessionProtocol",
    "VerifiedInvestigationResult",
]


_CAPABILITY_HEADER = "X-MiniChem-Capability"
_VERIFY_TOOL = "verify_execution_references"
_REFERENCE_BINDING_VERSION = "producer-job-v2"
_READ_TOOL = "read_scientific_artifact_text"
_RESOURCE_TOOL = "get_execution_resources"
_NATIVE_STATUS_TOOL = "get_execution_job"
_ALLOWED_TOOLS = frozenset({_VERIFY_TOOL, _READ_TOOL, _RESOURCE_TOOL, _NATIVE_STATUS_TOOL})
_SESSION_CALL_TIMEOUT_SECONDS = 300.0
_MCP_HTTP_REQUEST_TIMEOUT_SECONDS = 30.0
_MCP_HTTP_SSE_READ_TIMEOUT_SECONDS = 300.0
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_JOB_ID_RE = re.compile(r"^job_(?:[0-9a-f]{8}|[0-9a-f]{32})$")
_OPERATION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_REASON_CODE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_ARTIFACT_ID_RE = re.compile(r"^art_[0-9a-f]{32}$")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_TERMINAL_JOB_STATUSES = frozenset(
    {"success", "completed", "ready", "failed", "timeout", "cancelled", "canceled"}
)
_TEXT_MEDIA_PREFIXES = ("text/",)
_TEXT_MEDIA_EXACT = frozenset(
    {
        "application/json",
        "application/jsonl",
        "application/x-ndjson",
        "application/xml",
        "application/yaml",
        "application/x-yaml",
        "chemical/sdf",
        "chemical/x-mdl-sd",
        "chemical/x-mdl-sdfile",
        "chemical/x-sdf",
        "chemical/x-xyz",
    }
)
_SOURCE_CACHE_FILE_MODE = stat.S_IRUSR | stat.S_IWUSR
_SOURCE_CACHE_DIR_MODE = (
    stat.S_IRUSR
    | stat.S_IWUSR
    | stat.S_IXUSR
)


class InvestigationMiniChemGatewayError(RuntimeError):
    """Investigation 只读网关无法安全完成调用。"""


class InvestigationMiniChemProtocolError(InvestigationMiniChemGatewayError):
    """MiniChem MCP 响应或请求违反公开只读协议。"""


class InvestigationMiniChemReferenceRejected(InvestigationMiniChemProtocolError):
    """权威 verifier 已明确拒绝一项或多项 Result 执行引用。"""

    def __init__(self, invalid_references: Sequence[Mapping[str, str]]) -> None:
        normalized = tuple(dict(item) for item in invalid_references)
        if not normalized:
            raise ValueError("invalid_references 不能为空")
        super().__init__("MiniChem reference verification 未完整通过")
        self.invalid_references = normalized


class InvestigationMiniChemScopeError(InvestigationMiniChemGatewayError):
    """请求没有精确绑定到对应 Investigation scope。"""


class InvestigationMiniChemUnavailable(InvestigationMiniChemGatewayError):
    """MiniChem 只读网关或当前 scope 暂不可用。"""


@dataclass(frozen=True, slots=True)
class VerifiedInvestigationResult:
    """一次已由 MiniChem 权威核验的结果及其安全 Working Memory 证据。

    ``result`` 是原样返回的领域结果，``evidence`` 只包含稳定身份、终态引用和
    不可逆摘要。对象没有原始 receipt、token、workspace、路径或 transcript；调用
    方可以直接把两个字段分别交给结果状态机和 Working Memory 写入接口。
    """

    result: InvestigationResult
    evidence: VerifiedInvestigationEvidence

    def __iter__(self) -> Iterator[object]:
        """按 ``(result, evidence)`` 顺序提供安全的解包兼容。"""

        yield self.result
        yield self.evidence


@dataclass(frozen=True, slots=True)
class _VerifiedReferenceSummary:
    """MiniChem reference verifier 返回的脱敏、规范化引用摘要。"""

    jobs: tuple[dict[str, str], ...]
    operations: tuple[dict[str, str], ...]
    artifacts: tuple[dict[str, str], ...]
    scope_job_inventory: tuple[dict[str, str], ...]
    scope_operation_inventory: tuple[dict[str, str], ...]
    reference_binding_version: str

    @property
    def artifact_digests(self) -> dict[str, str]:
        """返回 artifact ID 到 canonical SHA-256 的新字典。"""

        return {item["artifact_id"]: item["artifact_digest"] for item in self.artifacts}


@dataclass(frozen=True, slots=True)
class _ArtifactPage:
    """一个已通过 MCP 文本协议核对的 artifact 分页。"""

    media_type: str
    digest: str
    total_chars: int
    content: str
    next_offset: int | None


class InvestigationMcpSessionProtocol(Protocol):
    """网关所需的最小 MCP 会话 interface。

    ``call_tool`` 的 arguments 必须能承载真实 MCP ``tools/call`` 的
    ``{"request": {...}}`` envelope。实现可以是同步或异步对象；网关不会调用
    工具目录、workspace 读取或任何提交/推进能力。
    """

    def initialize(self) -> Any:
        """完成 MCP initialize 握手。"""

    def call_tool(self, name: str, arguments: Mapping[str, Any] | None = None) -> Any:
        """调用一个公开 MCP 工具。"""


@dataclass(frozen=True, slots=True)
class InvestigationScopeBinding:
    """一个 Investigation scope 的身份与 capability token。

    ``token`` 使用 ``repr=False``，任何公开对象展示、错误文本和快照 manifest
    都不会自动包含 capability。task/batch/combination 是可选的兼容输入；若存在，
    网关会在每次调用前与 ``InvestigationSession`` 的完整身份逐字段比较。
    """

    scope_id: str
    token: str = field(repr=False)
    task_id: str | None = None
    batch_id: str | None = None
    combination_id: str | None = None
    workspace: Path | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """校验 scope 身份与 token 形状；失败信息不输出 token。"""

        _require_identity_text(self.scope_id, "scope_id")
        for value, label in (
            (self.task_id, "task_id"),
            (self.batch_id, "batch_id"),
            (self.combination_id, "combination_id"),
        ):
            if value is not None:
                _require_identity_text(value, label)
        _require_capability_token(self.token)
        if any(
            value is not None
            for value in (self.task_id, self.batch_id, self.combination_id)
        ) and not all(
            value is not None
            for value in (self.task_id, self.batch_id, self.combination_id)
        ):
            raise ValueError("scope binding 的 task/batch/combination 必须同时提供")
        if self.workspace is not None:
            try:
                normalized_workspace = Path(self.workspace).expanduser().resolve()
            except (TypeError, ValueError, OSError):
                raise ValueError("scope binding workspace 无效") from None
            object.__setattr__(self, "workspace", normalized_workspace)

    @property
    def identity_is_complete(self) -> bool:
        """返回是否保存了可与 session 完整比较的三元组身份。"""

        return all(
            value is not None
            for value in (self.task_id, self.batch_id, self.combination_id)
        )


SessionFactory: TypeAlias = Callable[..., Any]


@dataclass(slots=True)
class _SessionCommand:
    """同步网关线程提交给异步 session 生命周期的调用命令。"""

    method_name: str
    arguments: tuple[Any, ...]
    keyword_arguments: dict[str, Any]
    result: concurrent.futures.Future[Any]


_CLOSE_COMMAND = object()


def _require_identity_text(value: Any, label: str) -> str:
    """校验身份标识并返回去除首尾空白的值。"""

    if not isinstance(value, str):
        raise ValueError(f"{label} 必须是非空字符串")
    normalized = value.strip()
    if not normalized or any(ord(char) < 0x20 for char in normalized):
        raise ValueError(f"{label} 必须是非空安全标识")
    return normalized


def _require_capability_token(value: Any) -> str:
    """校验 capability token 的可传输形状，但不在异常中复制其内容。"""

    if not isinstance(value, str) or not 32 <= len(value) <= 512:
        raise ValueError("capability token 无效")
    if any(ord(char) < 0x21 or ord(char) > 0x7E for char in value):
        raise ValueError("capability token 无效")
    return value


def _normalize_endpoint_text(value: Any) -> str:
    """规范化 endpoint 文本并拒绝控制字符。"""

    if not isinstance(value, str):
        raise ValueError("MiniChem endpoint 必须是 HTTP(S) URL")
    normalized = value.strip()
    if not normalized.startswith(("http://", "https://")):
        raise ValueError("MiniChem endpoint 必须是 HTTP(S) URL")
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in normalized):
        raise ValueError("MiniChem endpoint 不能包含控制字符")
    return normalized


def _validate_endpoint_url(normalized: str) -> None:
    """核对 endpoint 的主机、凭据、query 和 fragment。"""

    try:
        parsed = urlsplit(normalized)
    except ValueError:
        raise ValueError("MiniChem endpoint URL 无效") from None
    if not parsed.hostname or parsed.username is not None or parsed.password is not None:
        raise ValueError("MiniChem endpoint 不能包含 URL userinfo")
    if parsed.query or parsed.fragment or "?" in normalized or "#" in normalized:
        raise ValueError("MiniChem endpoint 不能包含 query 或 fragment")


def _require_endpoint(value: Any) -> str:
    """校验 Streamable HTTP endpoint，不把凭据嵌入 URL。"""

    normalized = _normalize_endpoint_text(value)
    _validate_endpoint_url(normalized)
    return normalized


def _detach_exception_chain(error: BaseException) -> BaseException:
    """移除底层 transport 异常链，避免 header/token 出现在公开错误中。"""

    error.__cause__ = None
    error.__context__ = None
    error.__suppress_context__ = True
    return error


def _is_supported_text_media_type(value: Any) -> bool:
    """判断媒体类型是否属于可完整核验的 UTF-8 文本。"""

    if not isinstance(value, str) or not value.strip():
        return False
    media_type = value.strip().lower().split(";", 1)[0].strip()
    return media_type.startswith(_TEXT_MEDIA_PREFIXES) or media_type in _TEXT_MEDIA_EXACT or media_type.endswith(
        "+json"
    )


def _strict_json_loads(value: str) -> Any:
    """解析 MCP 文本内容并拒绝非标准 JSON 常量。"""

    def reject_constant(name: str) -> None:
        raise ValueError(f"非标准 JSON 常量: {name}")

    return json.loads(value, parse_constant=reject_constant)


def _object_field(value: Any, name: str) -> Any:
    """读取 mapping 或 MCP typed object 的同名字段。"""

    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _content_items(value: Any) -> list[Any]:
    """提取 MCP envelope 的 content 数组。"""

    items = _object_field(value, "content")
    return list(items) if isinstance(items, list) else []


def _content_text(value: Any) -> str | None:
    """从 MCP text/resource content 提取 JSON 文本。"""

    text = _object_field(value, "text")
    if isinstance(text, str):
        return text
    nested = _object_field(value, "resource")
    if nested is not None and nested is not value:
        return _content_text(nested)
    return None


def _parse_mcp_content_candidate(raw: Any, tool_name: str) -> Any:
    """解析 MCP content 中的 JSON 文本，并拒绝多页内容漂移。"""

    items = _content_items(raw)
    if not items:
        return None
    texts = [_content_text(item) for item in items]
    if any(text is None for text in texts):
        raise InvestigationMiniChemProtocolError(
            f"MiniChem 工具 {tool_name} 返回了无法解析的 content"
        )
    parsed_values: list[Any] = []
    for text in texts:
        try:
            parsed_values.append(_strict_json_loads(text or ""))
        except (TypeError, ValueError, json.JSONDecodeError):
            raise InvestigationMiniChemProtocolError(
                f"MiniChem 工具 {tool_name} 返回的 content 不是 JSON object"
            ) from None
    first = parsed_values[0]
    if any(value != first for value in parsed_values[1:]):
        raise InvestigationMiniChemProtocolError(
            f"MiniChem 工具 {tool_name} 返回了不一致的 content"
        )
    return first


def _extract_mcp_candidate(raw: Any, tool_name: str) -> Any:
    """从 MCP envelope 中合并 structuredContent、content 和 mapping 候选。"""

    if isinstance(raw, Mapping) and "status" in raw and "content" not in raw and "structuredContent" not in raw:
        return dict(raw)
    if bool(_object_field(raw, "isError") or _object_field(raw, "is_error")):
        raise InvestigationMiniChemProtocolError("MiniChem 只读工具返回错误")
    candidate = _object_field(raw, "structuredContent")
    if candidate is None:
        candidate = _object_field(raw, "structured_content")
    text_candidate = _parse_mcp_content_candidate(raw, tool_name)
    if candidate is not None and text_candidate is not None and candidate != text_candidate:
        raise InvestigationMiniChemProtocolError(
            f"MiniChem 工具 {tool_name} 的 structuredContent 与 content 不一致"
        )
    if candidate is None:
        candidate = text_candidate
    if candidate is None and isinstance(raw, Mapping):
        candidate = dict(raw)
    return candidate


def _serialize_mcp_candidate(candidate: Any, tool_name: str) -> dict[str, Any]:
    """把 MCP candidate 重新编码为严格、独立的 JSON object。"""

    if not isinstance(candidate, Mapping):
        raise InvestigationMiniChemProtocolError(
            f"MiniChem 工具 {tool_name} 返回值不是 JSON object"
        )
    try:
        encoded = json.dumps(
            candidate,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        parsed = _strict_json_loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError):
        raise InvestigationMiniChemProtocolError(
            f"MiniChem 工具 {tool_name} 返回值不是严格 JSON"
        ) from None
    if not isinstance(parsed, dict):
        raise InvestigationMiniChemProtocolError(
            f"MiniChem 工具 {tool_name} 返回值不是 object"
        )
    return parsed


def _parse_mcp_object(raw: Any, tool_name: str) -> dict[str, Any]:
    """把 MCP CallToolResult 或测试替身 object 解析成独立 JSON object。"""

    return _serialize_mcp_candidate(_extract_mcp_candidate(raw, tool_name), tool_name)


def _minimal_native_job_status(response: Mapping[str, Any]) -> dict[str, Any]:
    """从 native get 响应提取身份和状态，丢弃日志、诊断和资源 payload。"""

    if not isinstance(response, Mapping):
        raise InvestigationMiniChemProtocolError("native get_execution_job 返回值不是 object")
    allowed = {
        "status",
        "job_status",
        "terminal",
        "job_id",
        "operation_id",
        "submission_key",
        "operation_fingerprint",
        "caller_request_fingerprint",
        "job_type",
        "runtime",
    }
    result = {key: response[key] for key in allowed if key in response}
    nested = response.get("job")
    if nested is not None:
        if not isinstance(nested, Mapping):
            raise InvestigationMiniChemProtocolError(
                "native get_execution_job.job 不是 object"
            )
        nested_allowed = {
            "job_id",
            "status",
            "job_status",
            "operation_id",
            "submission_key",
            "operation_fingerprint",
            "caller_request_fingerprint",
            "job_type",
            "runtime",
        }
        result["job"] = {
            key: nested[key] for key in nested_allowed if key in nested
        }
    return result


def _synthetic_unknown_transition_state_status(waiting: Any) -> dict[str, Any]:
    """为已持久化但无法找到作业目录的 TS job 返回技术失败终态。

    该状态只保留等待快照中的完整身份和 ``unknown`` 终态，不生成科学结果、日志或
    artifact。它表示 Controller 已确认不能继续核对原 job；后续 wake 让模型按原
    job_id 查询并解释技术失败，任何路径都不会把它当作可重提信号。
    """

    identity = getattr(waiting, "identity", None)
    job_id = getattr(identity, "job_id", None)
    operation_id = getattr(identity, "operation_id", None)
    submission_key = getattr(identity, "submission_key", None)
    action_id = getattr(identity, "action_id", None)
    backend_id = getattr(identity, "backend_id", None)
    if not all(
        isinstance(value, str) and value.strip()
        for value in (job_id, operation_id, submission_key, action_id, backend_id)
    ):
        raise InvestigationMiniChemProtocolError(
            "TS unknown 终态缺少完整等待作业身份"
        )
    return {
        "status": "success",
        "job_id": job_id,
        "operation_id": operation_id,
        "submission_key": submission_key,
        "action_id": action_id,
        "backend_id": backend_id,
        "job_status": "unknown",
        "terminal": True,
        "phase": "unknown",
        # 这是 Controller 无法核对原作业时的显式技术失败标记，不表示作业门禁已
        # 打开；ARCHE 只允许该精确标记绕过正常的 armed/wake_armed 校验，并且不会重提。
        "synthetic_terminal": "unknown",
        "job": {"job_id": job_id, "status": "unknown"},
        "automatic_fallback": False,
    }


def _is_unknown_transition_state_error(
    response: Mapping[str, Any],
    *,
    job_id: str | None = None,
) -> bool:
    """判断 Controller 是否只缺失已持久化 TS job 的本地状态目录。"""

    if response.get("phase") == "unknown":
        return job_id is None or response.get("job_id") == job_id
    error = response.get("error")
    if response.get("status") != "invalid_request" or not isinstance(error, Mapping):
        return False
    if error.get("code") != "transition_state_path_job_unavailable":
        return False
    if job_id is not None and response.get("job_id") != job_id:
        return False
    return True


def _validate_catalog_readiness_response(
    response: Mapping[str, Any],
    *,
    expected_catalog_hash: str | None,
    expected_catalog_version: str | None,
) -> tuple[str, str]:
    """严格核对正式 MiniChem 的冻结目录身份。"""

    if response.get("status") != "success":
        raise InvestigationMiniChemProtocolError(
            "MiniChem get_execution_resources 未返回 success"
        )
    catalog_hash = response.get("catalog_hash")
    if not isinstance(catalog_hash, str) or _SHA256_RE.fullmatch(catalog_hash) is None:
        raise InvestigationMiniChemProtocolError(
            "MiniChem get_execution_resources 缺少合法 catalog_hash"
        )
    catalog_version = response.get("catalog_version")
    if (
        not isinstance(catalog_version, str)
        or not catalog_version
        or catalog_version != catalog_version.strip()
        or _CONTROL_RE.search(catalog_version) is not None
    ):
        raise InvestigationMiniChemProtocolError(
            "MiniChem get_execution_resources 缺少合法 catalog_version"
        )
    if expected_catalog_hash is not None and catalog_hash != expected_catalog_hash:
        raise InvestigationMiniChemProtocolError(
            "正式 MiniChem catalog_hash 与冻结目录不一致"
        )
    if expected_catalog_version is not None and catalog_version != expected_catalog_version:
        raise InvestigationMiniChemProtocolError(
            "正式 MiniChem catalog_version 与冻结目录不一致"
        )
    return catalog_hash, catalog_version


def _invoke_session_factory(factory: SessionFactory, endpoint: str, headers: Mapping[str, str]) -> Any:
    """按兼容顺序调用同步/异步 session factory。

    生产 factory 接收 ``(endpoint, headers)``；为确定性测试保留零参数 factory，
    但无论哪种形式，网关都会在 factory 前构造同一份 capability header。
    """

    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return factory(endpoint, dict(headers))
    candidates = [
        ((endpoint, dict(headers)), {}),
        ((), {"endpoint": endpoint, "headers": dict(headers)}),
    ]
    parameters = tuple(signature.parameters.values())
    positional_parameters = tuple(
        parameter
        for parameter in parameters
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )
    if len(positional_parameters) == 1:
        parameter_name = positional_parameters[0].name.casefold()
        if parameter_name in {"endpoint", "url"}:
            candidates.append(((endpoint,), {}))
        elif parameter_name in {"headers", "http_headers"}:
            candidates.append(((dict(headers),), {}))
    candidates.append(((), {}))
    for args, kwargs in candidates:
        try:
            signature.bind(*args, **kwargs)
        except TypeError:
            continue
        return factory(*args, **kwargs)
    raise TypeError("session_factory 的参数必须支持 endpoint/headers 或零参数")


async def _maybe_await(value: Any) -> Any:
    """等待异步 factory、context manager 或 session 方法。"""

    return await value if inspect.isawaitable(value) else value


def _create_authenticated_mcp_http_client(headers: Mapping[str, str]) -> Any:
    """创建带 scope header 和 MCP 持续响应流超时的 HTTP client。

    普通请求最多等待 30 秒；持续响应流最多 300 秒无数据后才超时。后者与
    gateway 的单次只读调用边界一致，避免 HTTPX 的 5 秒默认读取超时提前断开
    合法的 MiniChem 引用核验。
    """

    import httpx
    from mcp.shared._httpx_utils import create_mcp_http_client

    return create_mcp_http_client(
        headers=dict(headers),
        timeout=httpx.Timeout(
            _MCP_HTTP_REQUEST_TIMEOUT_SECONDS,
            read=_MCP_HTTP_SSE_READ_TIMEOUT_SECONDS,
        ),
    )


class _AuthenticatedStreamableHTTPContext(AbstractAsyncContextManager[Any]):
    """创建带 scope capability header 的真实 Streamable HTTP MCP session。"""

    def __init__(self, endpoint: str, headers: Mapping[str, str]) -> None:
        self._endpoint = endpoint
        self._headers = dict(headers)
        self._http_client: Any = None
        self._transport_context: Any = None
        self._session_context: Any = None

    async def __aenter__(self) -> Any:
        """创建 HTTP client、Streamable transport 和 MCP ClientSession。"""

        try:
            from mcp import ClientSession
            from mcp.client.streamable_http import streamable_http_client

            self._http_client = _create_authenticated_mcp_http_client(self._headers)
            self._transport_context = streamable_http_client(
                self._endpoint,
                http_client=self._http_client,
            )
            read_stream, write_stream, _ = await self._transport_context.__aenter__()
            self._session_context = ClientSession(read_stream, write_stream)
            return await self._session_context.__aenter__()
        except BaseException:
            await self.__aexit__(None, None, None)
            raise

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        """按相反顺序释放 MCP session、transport 和 HTTP client。"""

        if self._session_context is not None:
            context = self._session_context
            self._session_context = None
            try:
                await context.__aexit__(exc_type, exc, traceback)
            except BaseException:
                pass
        if self._transport_context is not None:
            context = self._transport_context
            self._transport_context = None
            try:
                await context.__aexit__(exc_type, exc, traceback)
            except BaseException:
                pass
        if self._http_client is not None:
            client = self._http_client
            self._http_client = None
            try:
                await client.aclose()
            except BaseException:
                pass


class _AsyncSessionRunner:
    """在单独事件循环中维护一个异步 MCP session 生命周期。"""

    def __init__(
        self,
        factory: SessionFactory,
        endpoint: str,
        headers: Mapping[str, str],
        first_candidate: Any = None,
    ) -> None:
        self._factory = factory
        self._endpoint = endpoint
        self._headers = dict(headers)
        self._first_candidate = first_candidate
        self._loop: asyncio.AbstractEventLoop | None = None
        self._commands: asyncio.Queue[_SessionCommand | object] | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._closed = threading.Event()
        self._failure: BaseException | None = None
        self._session: Any = None
        self._context: Any = None
        self._active_command_task: asyncio.Task[Any] | None = None
        self._start_lock = threading.Lock()
        self._command_lock = threading.Lock()
        self._pending_commands: dict[int, _SessionCommand] = {}

    def start(self) -> None:
        """启动事件循环并完成异步 factory、context 和 initialize。"""

        with self._start_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._ready.clear()
            self._closed.clear()
            self._failure = None
            self._loop = asyncio.new_event_loop()
            self._commands = asyncio.Queue()
            self._thread = threading.Thread(
                target=self._run_thread,
                name="investigation-minichem-mcp",
                daemon=True,
            )
            self._thread.start()
        if not self._ready.wait(timeout=60):
            self.close()
            raise InvestigationMiniChemUnavailable("MiniChem scope session 初始化超时")
        if self._failure is not None:
            self.close()
            raise InvestigationMiniChemUnavailable("MiniChem scope session 初始化失败")

    def _run_thread(self) -> None:
        """执行事件循环；失败信息只保存在内部，不复制 token。"""

        loop = self._loop
        if loop is None:
            self._failure = RuntimeError("missing event loop")
            self._ready.set()
            return
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._run_lifecycle())
        except BaseException as exc:
            self._failure = exc
            self._ready.set()
        finally:
            self._mark_lifecycle_unavailable()
            try:
                loop.close()
            except BaseException:
                pass

    async def _open_session_lifecycle(self) -> None:
        """创建候选 session、进入异步 context 并完成 initialize。"""

        candidate = self._first_candidate
        self._first_candidate = None
        if candidate is None:
            candidate = _invoke_session_factory(self._factory, self._endpoint, self._headers)
        candidate = await _maybe_await(candidate)
        self._context = candidate if callable(getattr(candidate, "__aenter__", None)) else None
        if self._context is not None:
            self._session = await _maybe_await(self._context.__aenter__())
        else:
            self._session = candidate
        initialize = getattr(self._session, "initialize", None)
        call_tool = getattr(self._session, "call_tool", None)
        if not callable(initialize) or not callable(call_tool):
            raise TypeError("MCP session 缺少 initialize/call_tool")
        await _maybe_await(initialize())

    async def _serve_session_commands(self) -> None:
        """串行执行事件循环队列中的 MCP session 调用。"""

        commands = self._commands
        if commands is None:
            raise RuntimeError("missing session command queue")
        while True:
            command = await commands.get()
            if command is _CLOSE_COMMAND:
                return
            if not isinstance(command, _SessionCommand):
                continue
            if not self._command_is_pending(command):
                continue
            active_task = asyncio.current_task()
            self._active_command_task = active_task
            try:
                function = getattr(self._session, command.method_name)
                value = await _maybe_await(
                    function(*command.arguments, **command.keyword_arguments)
                )
            except InvestigationMiniChemGatewayError as exc:
                self._set_command_exception(command, exc)
            except BaseException as exc:
                self._set_command_exception(command, exc)
            else:
                self._set_command_result(command, value)
            finally:
                if self._active_command_task is active_task:
                    self._active_command_task = None
                self._forget_pending_command(command)

    def _command_is_pending(self, command: _SessionCommand) -> bool:
        """判断命令是否仍属于当前可接收生命周期。"""

        with self._command_lock:
            return (
                not self._closed.is_set()
                and self._pending_commands.get(id(command)) is command
            )

    @staticmethod
    def _set_command_exception(command: _SessionCommand, error: BaseException) -> None:
        """向仍在等待的调用报告异常，忽略已经被生命周期收束的 Future。"""

        try:
            command.result.set_exception(error)
        except concurrent.futures.InvalidStateError:
            pass

    @staticmethod
    def _set_command_result(command: _SessionCommand, value: Any) -> None:
        """向仍在等待的调用报告结果，忽略已经被生命周期收束的 Future。"""

        try:
            command.result.set_result(value)
        except concurrent.futures.InvalidStateError:
            pass

    def _forget_pending_command(self, command: _SessionCommand) -> None:
        """从待处理登记中移除已完成或已失败的命令。"""

        with self._command_lock:
            self._pending_commands.pop(id(command), None)

    def _fail_pending_commands(self, *, close: bool = False) -> None:
        """让生命周期结束前尚未消费的命令立即收到不可用错误。"""

        with self._command_lock:
            if close:
                self._closed.set()
            pending = tuple(self._pending_commands.values())
            self._pending_commands.clear()
        error = InvestigationMiniChemUnavailable("MiniChem scope session 不可用")
        for command in pending:
            self._set_command_exception(command, error)

    def _mark_lifecycle_unavailable(self) -> None:
        """停止接受新命令，并结束所有未被生命周期消费的命令。"""

        self._fail_pending_commands(close=True)

    async def _close_session_lifecycle(self) -> None:
        """释放异步 context 或直接 session，并吞掉关闭阶段异常。"""

        if self._context is not None:
            try:
                await self._context.__aexit__(None, None, None)
            except BaseException:
                pass
        else:
            close = getattr(self._session, "close", None)
            if callable(close):
                try:
                    await _maybe_await(close())
                except BaseException:
                    pass
        self._session = None
        self._context = None

    async def _run_lifecycle(self) -> None:
        """打开 session、串行执行调用并在关闭时释放 context。"""

        try:
            await self._open_session_lifecycle()
            self._ready.set()
            await self._serve_session_commands()
        except BaseException as exc:
            self._failure = exc
            self._ready.set()
        finally:
            self._mark_lifecycle_unavailable()
            await self._close_session_lifecycle()

    def call(self, method_name: str, *arguments: Any, **keyword_arguments: Any) -> Any:
        """把一次 session 方法调用提交到生命周期事件循环。"""

        self.start()
        loop = self._loop
        commands = self._commands
        if loop is None or commands is None or self._closed.is_set():
            raise InvestigationMiniChemUnavailable("MiniChem scope session 不可用")
        result: concurrent.futures.Future[Any] = concurrent.futures.Future()
        command = _SessionCommand(method_name, arguments, keyword_arguments, result)
        with self._command_lock:
            if self._closed.is_set():
                raise InvestigationMiniChemUnavailable("MiniChem scope session 不可用")
            self._pending_commands[id(command)] = command
            try:
                loop.call_soon_threadsafe(commands.put_nowait, command)
            except RuntimeError:
                self._pending_commands.pop(id(command), None)
                raise InvestigationMiniChemUnavailable("MiniChem scope session 不可用") from None
        try:
            return result.result(timeout=_SESSION_CALL_TIMEOUT_SECONDS)
        except concurrent.futures.TimeoutError:
            self._mark_lifecycle_unavailable()
            raise InvestigationMiniChemUnavailable("MiniChem 只读调用不可用") from None
        except InvestigationMiniChemGatewayError:
            raise
        except BaseException:
            raise InvestigationMiniChemUnavailable("MiniChem 只读调用不可用") from None

    def close(self, timeout: float = 30.0) -> None:
        """请求异步 session 退出并等待其生命周期线程结束。

        ``timeout`` 只限制当前线程等待旧 session 的时间；如果 transport 已经
        卡住，调用方仍然可以在超时后建立新的 session，旧线程会继续按 daemon
        线程回收，不会继续接受新的网关命令。
        """

        # 先关闭接收门并失败所有已登记 Future，再通知事件循环取消当前调用。
        # 这样旧生命周期即使稍后恢复，也不会执行已经交给 fresh session 的请求。
        self._mark_lifecycle_unavailable()
        loop = self._loop
        commands = self._commands
        thread = self._thread
        if loop is None or commands is None or thread is None:
            return
        if thread.is_alive():
            try:
                def cancel_active_call() -> None:
                    """在 session 事件循环内取消卡住的当前调用。"""

                    task = self._active_command_task
                    if task is not None and not task.done():
                        task.cancel()

                loop.call_soon_threadsafe(cancel_active_call)
                loop.call_soon_threadsafe(commands.put_nowait, _CLOSE_COMMAND)
            except RuntimeError:
                pass
            thread.join(timeout=max(0.0, timeout))
        self._thread = None
        self._loop = None
        self._commands = None


class _ScopeSession:
    """绑定单个 scope 的同步或异步 MCP session。"""

    def __init__(
        self,
        endpoint: str,
        binding: InvestigationScopeBinding,
        factory: SessionFactory,
    ) -> None:
        self._endpoint = endpoint
        self._binding = binding
        self._factory = factory
        self._direct_session: Any = None
        self._direct_context: Any = None
        self._runner: _AsyncSessionRunner | None = None
        self._closed = False
        self._start_lock = threading.Lock()
        self._call_lock = threading.Lock()

    def _build_candidate(self) -> Any:
        """创建 factory 候选，不把 token 放入异常文本。"""

        headers = {_CAPABILITY_HEADER: self._binding.token}
        try:
            return _invoke_session_factory(self._factory, self._endpoint, headers)
        except BaseException as exc:
            raise InvestigationMiniChemUnavailable("MiniChem scope session 创建失败") from None

    @staticmethod
    def _requires_async_runner(candidate: Any) -> bool:
        """判断候选是否需要异步生命周期。"""

        if inspect.isawaitable(candidate) or callable(getattr(candidate, "__aenter__", None)):
            return True
        return any(
            inspect.iscoroutinefunction(getattr(candidate, name, None))
            for name in ("initialize", "call_tool", "close")
        )

    def start(self) -> None:
        """建立 direct 或异步 MCP session，并完成 initialize。"""

        with self._start_lock:
            if self._closed:
                raise InvestigationMiniChemUnavailable("MiniChem scope session 已关闭")
            if self._direct_session is not None or self._runner is not None:
                return
            candidate = self._build_candidate()
            if self._requires_async_runner(candidate):
                # 把已创建的候选交给 runner，避免 async factory 被调用两次；
                # 这也保证每个 scope 的认证 session 只有一个生命周期。
                self._runner = _AsyncSessionRunner(
                    self._factory,
                    self._endpoint,
                    {_CAPABILITY_HEADER: self._binding.token},
                    candidate,
                )
                self._runner.start()
                return
            if callable(getattr(candidate, "__enter__", None)):
                try:
                    entered = candidate.__enter__()
                except BaseException:
                    raise InvestigationMiniChemUnavailable("MiniChem scope session 初始化失败") from None
                self._direct_context = candidate
                candidate = entered
            initialize = getattr(candidate, "initialize", None)
            call_tool = getattr(candidate, "call_tool", None)
            if not callable(initialize) or not callable(call_tool):
                raise InvestigationMiniChemProtocolError("MCP session 缺少 initialize/call_tool")
            try:
                initialized = initialize()
                if inspect.isawaitable(initialized):
                    raise InvestigationMiniChemUnavailable("MiniChem session 异步初始化失败")
            except InvestigationMiniChemGatewayError:
                raise
            except BaseException as exc:
                raise InvestigationMiniChemUnavailable("MiniChem scope session 初始化失败") from None
            self._direct_session = candidate

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> Any:
        """调用当前 scope 的一个只读工具，并安全恢复失效的连接。

        只有当前网关白名单中的只读工具会进入恢复路径。首次调用出现传输层
        ``Unavailable`` 时，当前 session 会被隔离并关闭一次，然后用同一 scope
        binding 建立 fresh session 重试同一请求；协议错误、scope 身份错误和
        第二次失败都会原样停止，不能借恢复掩盖不确定状态。
        """

        if name not in _ALLOWED_TOOLS:
            raise InvestigationMiniChemProtocolError("网关拒绝非只读 MiniChem 工具")
        with self._call_lock:
            self.start()
            try:
                return self._call_active_session(name, arguments)
            except InvestigationMiniChemUnavailable:
                self._replace_failed_session()
                self.start()
                return self._call_active_session(name, arguments)

    def _call_active_session(
        self,
        name: str,
        arguments: Mapping[str, Any],
    ) -> Any:
        """在当前 session 中执行一次已经批准的只读调用。"""

        if self._runner is not None:
            return self._runner.call("call_tool", name, dict(arguments))
        session = self._direct_session
        if session is None:
            raise InvestigationMiniChemUnavailable("MiniChem scope session 不可用")
        try:
            return session.call_tool(name, dict(arguments))
        except InvestigationMiniChemGatewayError:
            raise
        except BaseException:
            raise InvestigationMiniChemUnavailable("MiniChem 只读调用失败") from None

    def _replace_failed_session(self) -> None:
        """隔离已失效的 session，保留 scope binding 后允许建立新连接。"""

        with self._start_lock:
            if self._closed:
                raise InvestigationMiniChemUnavailable("MiniChem scope session 已关闭")
            runner = self._runner
            self._runner = None
            session = self._direct_session
            self._direct_session = None
            context = self._direct_context
            self._direct_context = None
        if runner is not None:
            runner.close(timeout=1.0)
            return
        if context is not None:
            close = getattr(context, "__exit__", None)
            if callable(close):
                try:
                    close(None, None, None)
                except BaseException:
                    pass
        elif session is not None:
            close = getattr(session, "close", None)
            if callable(close):
                try:
                    close()
                except BaseException:
                    pass

    def close(self) -> None:
        """关闭当前 scope session。"""

        with self._start_lock:
            if self._closed:
                return
            self._closed = True
            runner = self._runner
            self._runner = None
            session = self._direct_session
            self._direct_session = None
            context = self._direct_context
            self._direct_context = None
        if runner is not None:
            runner.close()
            return
        if context is not None:
            close = getattr(context, "__exit__", None)
            if callable(close):
                try:
                    close(None, None, None)
                except BaseException:
                    pass
        elif session is not None:
            close = getattr(session, "close", None)
            if callable(close):
                try:
                    close()
                except BaseException:
                    pass


def _normalize_scope_binding(scope_id: str, raw: Any) -> InvestigationScopeBinding:
    """把 scope capability、简化 mapping 或 token 转为内部绑定。"""

    if isinstance(raw, InvestigationScopeBinding):
        if raw.scope_id != scope_id:
            raise ValueError("scope capability mapping key 与 scope_id 不一致")
        return raw
    if isinstance(raw, str):
        return InvestigationScopeBinding(scope_id=scope_id, token=raw)
    if isinstance(raw, Mapping):
        values = dict(raw)
        values.setdefault("scope_id", scope_id)
        try:
            binding = InvestigationScopeBinding(**values)
        except (TypeError, ValueError):
            raise ValueError("scope capability mapping 无效") from None
        if binding.scope_id != scope_id:
            raise ValueError("scope capability mapping key 与 scope_id 不一致")
        return binding
    try:
        token = getattr(raw, "token")
        raw_scope_id = getattr(raw, "scope_id")
        binding = InvestigationScopeBinding(
            scope_id=raw_scope_id,
            token=token,
            task_id=getattr(raw, "task_id", None),
            batch_id=getattr(raw, "batch_id", None),
            combination_id=getattr(raw, "combination_id", None),
            workspace=getattr(raw, "workspace", None),
        )
    except (AttributeError, TypeError, ValueError):
        raise ValueError("scope capability 对象无效") from None
    if binding.scope_id != scope_id:
        raise ValueError("scope capability mapping key 与 scope_id 不一致")
    return binding


class InvestigationMiniChemGateway:
    """为 Investigation callback 提供结果核验与证据快照两个 callable。

    构造输入是 endpoint、按 scope 索引的 capability、Evidence source cache 和
    已构造的 ``ReviewEvidenceStore``。网关只暴露 ``result_verifier`` 与
    ``evidence_snapshot_builder`` 两个可注入属性；其它复杂性（scope 选择、认证
    session、MCP envelope、响应严格解析、分页完整性、原子缓存和幂等快照）都留在
    模块内部。任何身份漂移、协议漂移、未终态、不可用、重复/缺页、二进制、超限
    或摘要不匹配都会 fail-closed 抛出网关错误。
    """

    def __init__(
        self,
        endpoint: str,
        scope_capabilities: Mapping[str, Any] | Sequence[Any] | Any,
        *,
        evidence_store: ReviewEvidenceStore,
        source_cache_dir: str | os.PathLike[str] | Path,
        session_factory: SessionFactory | None = None,
        page_size: int = 8192,
        max_pages_per_artifact: int = 1024,
        max_artifact_bytes: int = 16 * 1024 * 1024,
        expected_catalog_hash: str | None = None,
        expected_catalog_version: str | None = None,
    ) -> None:
        """准备 scope 会话工厂、权限收紧的 source cache 和严格资源上限。

        ``scope_capabilities`` 可以是 ``{scope_id: capability}`` mapping，也可以
        是 capability 对象序列；对象需要提供 ``scope_id``、``token``，并可选提供
        task/batch/combination。``session_factory`` 缺省时进入真实 Streamable HTTP
        transport；注入 factory 时只调用目录资源和两项结果只读工具。构造不会连接
        MiniChem、不读取 workspace，参数或 cache 不可用时立即失败。若提供冻结目录
        hash/version，首次 scope readiness 会核对正式服务返回的同一身份。
        """

        self.endpoint = _require_endpoint(endpoint)
        if (
            expected_catalog_hash is not None
            and (
                not isinstance(expected_catalog_hash, str)
                or _SHA256_RE.fullmatch(expected_catalog_hash) is None
            )
        ):
            raise ValueError("expected_catalog_hash 必须是 64 位小写 SHA-256")
        if (
            expected_catalog_version is not None
            and (
                not isinstance(expected_catalog_version, str)
                or not expected_catalog_version
                or expected_catalog_version != expected_catalog_version.strip()
                or _CONTROL_RE.search(expected_catalog_version) is not None
            )
        ):
            raise ValueError("expected_catalog_version 必须是非空安全字符串")
        self._expected_catalog_hash = expected_catalog_hash
        self._expected_catalog_version = expected_catalog_version
        if not isinstance(evidence_store, ReviewEvidenceStore):
            raise TypeError("evidence_store 必须是 ReviewEvidenceStore")
        self._evidence_store = evidence_store
        self._source_cache_dir = _prepare_source_cache_dir(source_cache_dir)
        self._validate_limit(page_size, "page_size", upper=20_000)
        self._validate_limit(max_pages_per_artifact, "max_pages_per_artifact", upper=100_000)
        self._validate_limit(max_artifact_bytes, "max_artifact_bytes", upper=1024 * 1024 * 1024)
        self.page_size = page_size
        self.max_pages_per_artifact = max_pages_per_artifact
        self.max_artifact_bytes = max_artifact_bytes
        self._session_factory = session_factory or (
            lambda endpoint, headers: _AuthenticatedStreamableHTTPContext(endpoint, headers)
        )
        if not callable(self._session_factory):
            raise TypeError("session_factory 必须可调用")
        self._bindings = self._normalize_scope_capabilities(scope_capabilities)
        self._sessions: dict[str, _ScopeSession] = {}
        self._session_lock = threading.RLock()
        self.result_verifier: Callable[
            [InvestigationSession, InvestigationResult], InvestigationResult | None
        ] = self.verify_investigation_result
        self.evidence_snapshot_builder: Callable[
            [InvestigationState, tuple[InvestigationResult, ...], tuple[tuple[str, str], ...]],
            Mapping[str, Any],
        ] = self.build_evidence_snapshot

    @staticmethod
    def _validate_limit(value: Any, label: str, *, upper: int) -> None:
        """校验正整数资源上限。"""

        if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= upper:
            raise ValueError(f"{label} 必须是 1 到 {upper} 的整数")

    @staticmethod
    def _normalize_scope_capabilities(
        value: Mapping[str, Any] | Sequence[Any] | Any,
    ) -> dict[str, InvestigationScopeBinding]:
        """冻结 scope capability mapping，并拒绝重复或漂移身份。"""

        if isinstance(value, Mapping):
            entries = list(value.items())
        elif isinstance(value, (str, bytes, bytearray)):
            raise TypeError("scope_capabilities 必须是 capability mapping 或对象序列")
        elif isinstance(value, Sequence):
            entries = []
            for raw in value:
                raw_scope_id = getattr(raw, "scope_id", None)
                if isinstance(raw, Mapping):
                    raw_scope_id = raw.get("scope_id")
                if not isinstance(raw_scope_id, str):
                    raise ValueError("scope capability 缺少 scope_id")
                entries.append((raw_scope_id, raw))
        else:
            raw_scope_id = getattr(value, "scope_id", None)
            if not isinstance(raw_scope_id, str):
                raise ValueError("scope capability 缺少 scope_id")
            entries = [(raw_scope_id, value)]
        result: dict[str, InvestigationScopeBinding] = {}
        for raw_scope_id, raw in entries:
            scope_id = _require_identity_text(raw_scope_id, "scope_id")
            if scope_id in result:
                raise ValueError("scope_capabilities 不能重复 scope_id")
            try:
                result[scope_id] = _normalize_scope_binding(scope_id, raw)
            except (TypeError, ValueError):
                raise ValueError("scope_capabilities 含无效 capability") from None
        if not result:
            raise ValueError("scope_capabilities 不能为空")
        return result

    def __repr__(self) -> str:
        """返回不含 capability token 的网关摘要。"""

        return (
            f"{type(self).__name__}(endpoint={self.endpoint!r}, "
            f"scope_ids={tuple(self._bindings)!r}, source_cache_dir=<private>)"
        )

    def __enter__(self) -> "InvestigationMiniChemGateway":
        """返回可注入的网关；连接仍按 scope 首次使用时惰性建立。"""

        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        """关闭所有 scope MCP 会话。"""

        self.close()

    def close(self) -> None:
        """关闭所有已建立的 scope session；不删除 source cache。"""

        with self._session_lock:
            sessions = tuple(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            session.close()

    def preflight_scope_readiness(
        self,
        session: InvestigationSession,
    ) -> Mapping[str, Any]:
        """用当前 scope 完成认证握手并核对正式服务目录身份。

        动作先完成带 capability 的 MCP initialize，再以空 request envelope 调用
        ``get_execution_resources``，只读取脱敏的 catalog hash/version。缺失、错误、
        漂移或传输失败都会阻断 runtime/model turn；真正的 job/operation inventory
        核对仍由 receipt-aware 调用路径在即将启动新模型 turn 前执行。
        """

        try:
            if not isinstance(session, InvestigationSession):
                raise TypeError("session 必须是 InvestigationSession")
            binding = self._resolve_scope_binding(session)
            scope_session = self._resolve_scope_session(binding)
            scope_session.start()
            response = self._call_scope_tool(binding, _RESOURCE_TOOL, {})
            catalog_hash, catalog_version = _validate_catalog_readiness_response(
                response,
                expected_catalog_hash=self._expected_catalog_hash,
                expected_catalog_version=self._expected_catalog_version,
            )
            return {
                "scope_id": session.scope_id,
                "authenticated": True,
                "catalog_hash": catalog_hash,
                "catalog_version": catalog_version,
            }
        except InvestigationMiniChemGatewayError as error:
            _detach_exception_chain(error)
            raise

    def _resolve_scope_binding(self, session: InvestigationSession) -> InvestigationScopeBinding:
        """按完整 session scope identity 选择唯一 capability。"""

        if not isinstance(session, InvestigationSession):
            raise TypeError("session 必须是 InvestigationSession")
        binding = self._bindings.get(session.scope_id)
        if binding is None:
            raise InvestigationMiniChemScopeError("session 没有已登记的 MiniChem scope")
        if binding.identity_is_complete and (
            binding.task_id != session.task_id
            or binding.batch_id != session.batch_id
            or binding.combination_id != session.combination_id
        ):
            raise InvestigationMiniChemScopeError("session 与 MiniChem scope identity 漂移")
        return binding

    def _resolve_scope_session(self, binding: InvestigationScopeBinding) -> _ScopeSession:
        """按 scope 惰性创建并缓存独立的认证 MCP session。"""

        with self._session_lock:
            current = self._sessions.get(binding.scope_id)
            if current is None:
                current = _ScopeSession(self.endpoint, binding, self._session_factory)
                self._sessions[binding.scope_id] = current
            return current

    def _call_scope_tool(
        self,
        binding: InvestigationScopeBinding,
        tool_name: str,
        request: Mapping[str, Any],
    ) -> dict[str, Any]:
        """以真实 ``request`` envelope 调用一个已批准只读工具。"""

        if tool_name not in _ALLOWED_TOOLS:
            raise InvestigationMiniChemProtocolError("网关拒绝非只读 MiniChem 工具")
        if not isinstance(request, Mapping):
            raise TypeError("MiniChem request 必须是 JSON object")
        session = self._resolve_scope_session(binding)
        raw = session.call_tool(tool_name, {"request": dict(request)})
        return _parse_mcp_object(raw, tool_name)

    def _resolve_waiting_binding(self, waiting: Any) -> InvestigationScopeBinding:
        """按等待状态的 scope 和组合身份选择唯一 capability。"""

        scope_id = getattr(waiting, "scope_id", None)
        if not isinstance(scope_id, str) or not scope_id.strip():
            raise InvestigationMiniChemScopeError("waiting job 缺少有效 scope_id")
        binding = self._bindings.get(scope_id)
        if binding is None:
            raise InvestigationMiniChemScopeError("waiting job 没有已登记的 MiniChem scope")
        identity = (
            getattr(waiting, "task_id", None),
            getattr(waiting, "batch_id", None),
            getattr(waiting, "combination_id", None),
        )
        if binding.identity_is_complete and identity != (
            binding.task_id,
            binding.batch_id,
            binding.combination_id,
        ):
            raise InvestigationMiniChemScopeError("waiting job 与 MiniChem scope identity 漂移")
        return binding

    def get_investigation_job_status(self, waiting: Any) -> Mapping[str, Any]:
        """查询一个等待作业的最小状态，不读取科学 payload。"""

        try:
            binding = self._resolve_waiting_binding(waiting)
            identity = getattr(waiting, "identity", None)
            job_id = getattr(identity, "job_id", None)
            if not isinstance(job_id, str) or not job_id.strip():
                raise InvestigationMiniChemScopeError("waiting job 缺少有效 job_id")
            family = getattr(waiting, "job_family", None)
            if (
                family == "transition_state_path"
                and getattr(waiting, "status_tool", None)
                == "get_transition_state_path_job"
            ):
                # 只为历史 sidecar 保留旧 Controller 直读；新状态统一走
                # get_execution_job，TS 与其它作业没有模型可见差异。
                workspace = binding.workspace
                if workspace is None:
                    raise InvestigationMiniChemScopeError(
                        "TS Controller status 缺少 capability workspace"
                    )
                from minichem_mcp_tools.transition_state_jobs import (
                    controller_get_transition_state_path_job,
                )

                response = controller_get_transition_state_path_job(
                    job_id,
                    scope_id=binding.scope_id,
                    workspace=workspace,
                )
                if not isinstance(response, Mapping):
                    raise InvestigationMiniChemProtocolError(
                        "TS Controller status 返回值不是 object"
                    )
                if _is_unknown_transition_state_error(
                    response,
                    job_id=job_id,
                ):
                    return _synthetic_unknown_transition_state_status(waiting)
                return dict(response)
            if family in {
                "transition_state_path",
                "native_execution",
                "action_execution",
                "programmable_analysis",
            }:
                response = self._call_scope_tool(
                    binding,
                    _NATIVE_STATUS_TOOL,
                    {"job_id": job_id, "tail_chars": 0},
                )
                return _minimal_native_job_status(response)
            raise InvestigationMiniChemProtocolError("不支持的 Investigation job family")
        except InvestigationMiniChemGatewayError as error:
            _detach_exception_chain(error)
            raise
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
            raise InvestigationMiniChemUnavailable(
                "MiniChem Controller 作业状态不可用"
            ) from None

    def arm_investigation_job(
        self,
        waiting: Any,
        terminal_status: str,
    ) -> Mapping[str, Any]:
        """为中断恢复确认 TS 作业终态并调用专用 Controller arm，不收集结果。"""

        try:
            binding = self._resolve_waiting_binding(waiting)
            identity = getattr(waiting, "identity", None)
            job_id = getattr(identity, "job_id", None)
            if not isinstance(job_id, str) or not job_id.strip():
                raise InvestigationMiniChemScopeError("waiting job 缺少有效 job_id")
            family = getattr(waiting, "job_family", None)
            if family in {
                "transition_state_path",
                "native_execution",
                "action_execution",
                "programmable_analysis",
            } and not (
                family == "transition_state_path"
                and getattr(waiting, "status_tool", None)
                == "get_transition_state_path_job"
            ):
                # Native get_execution_job 本身没有额外的 arm 门禁；返回一次最小
                # status 供 Controller 再次核对，绝不调用 collect。
                return self.get_investigation_job_status(waiting)
            if family != "transition_state_path":
                raise InvestigationMiniChemProtocolError("不支持的 Investigation job family")
            workspace = binding.workspace
            if workspace is None:
                raise InvestigationMiniChemScopeError(
                    "TS Controller arm 缺少 capability workspace"
                )
            from minichem_mcp_tools.transition_state_jobs import (
                controller_arm_transition_state_path_job,
            )

            response = controller_arm_transition_state_path_job(
                job_id,
                scope_id=binding.scope_id,
                workspace=workspace,
            )
            if not isinstance(response, Mapping):
                raise InvestigationMiniChemProtocolError(
                    "TS Controller arm 返回值不是 object"
                )
            if (
                terminal_status == "unknown"
                and _is_unknown_transition_state_error(response)
            ):
                return _synthetic_unknown_transition_state_status(waiting)
            return dict(response)
        except InvestigationMiniChemGatewayError as error:
            _detach_exception_chain(error)
            raise
        except (ImportError, OSError, RuntimeError, TypeError, ValueError):
            raise InvestigationMiniChemUnavailable(
                "MiniChem Controller terminal arm 不可用"
            ) from None

    @staticmethod
    def _validate_result_identity(
        session: InvestigationSession,
        result: InvestigationResult,
    ) -> None:
        """核对 session 与结果的任务、批次、组合和 scope 身份。"""

        if not isinstance(result, InvestigationResult):
            raise TypeError("result 必须是 InvestigationResult")
        if (
            result.task_id != session.task_id
            or result.batch_id != session.batch_id
            or result.combination_id != session.combination_id
            or result.scope_id != session.scope_id
        ):
            raise InvestigationMiniChemScopeError("InvestigationResult 与 session identity 漂移")
        if session.codex_thread_id is None or result.codex_thread_id != session.codex_thread_id:
            raise InvestigationMiniChemScopeError("InvestigationResult 与 session thread 漂移")

    @staticmethod
    def _validate_reference_response(
        response: Mapping[str, Any],
        *,
        expected_jobs: tuple[str, ...],
        expected_operations: tuple[str, ...],
        expected_artifacts: tuple[str, ...],
        inventory_job_ids: Sequence[str] | None = None,
        allow_historical_terminal_jobs: bool = False,
        allow_historical_operations: bool = False,
    ) -> _VerifiedReferenceSummary:
        """严格核对 reference verification，并返回脱敏的引用摘要。

        结果核验可以允许同一认证 scope 中与当前结果无关的历史终态 job 和探索
        operation；side-effect fence 不设置这些开关，仍要求 inventory 与当前引用
        精确一致且没有活动项。
        """

        expected = _expected_reference_items(
            expected_jobs,
            expected_operations,
            expected_artifacts,
        )
        schema_version = _validate_reference_response_header(
            response,
            expected=expected,
        )
        parsed_items = _parse_reference_items(
            response["items"],
            expected,
            schema_version=schema_version,
            expected_operation_ids=expected_operations,
        )
        _validate_reference_response_inventories(
            response,
            expected_jobs if inventory_job_ids is None else tuple(inventory_job_ids),
            expected_operations,
            schema_version=schema_version,
            expected_submission_operations={
                item["job_id"]: item.get("submission_operation_id")
                for item in parsed_items["jobs"]
            },
            allow_historical_terminal_jobs=allow_historical_terminal_jobs,
            allow_historical_operations=allow_historical_operations,
        )
        return _VerifiedReferenceSummary(
            jobs=tuple(parsed_items["jobs"]),
            operations=tuple(parsed_items["operations"]),
            artifacts=tuple(parsed_items["artifacts"]),
            scope_job_inventory=tuple(_inventory_list(response["scope_job_inventory"], "jobs")),
            scope_operation_inventory=tuple(response["scope_operation_inventory"]),
            reference_binding_version=_REFERENCE_BINDING_VERSION,
        )

    def verify_investigation_result(
        self,
        session: InvestigationSession,
        result: InvestigationResult,
    ) -> InvestigationResult:
        """在 session 精确绑定的 authenticated scope 中核验一个结果。

        请求只携带 result 保存的三类不透明 ID，不携带路径、日志、payload 或
        capability。响应必须按 jobs→operations→artifacts 顺序完整覆盖，所有
        job 必须终态、operation 必须 completed、artifact 必须有 canonical
        sha256；scope inventory 可以包含当前 Result 之外的同 scope 历史终态 job，
        但当前引用、producer 来源链和每个 inventory 项仍逐项核对。任何 unavailable、
        身份漂移或协议异常都会抛出网关错误，成功时原样返回同一个
        ``InvestigationResult`` 对象。
        """

        try:
            if not isinstance(session, InvestigationSession):
                raise TypeError("session 必须是 InvestigationSession")
            if not isinstance(result, InvestigationResult):
                raise TypeError("result 必须是 InvestigationResult")
            self._verify_result_references(session, result)
            return result
        except InvestigationMiniChemGatewayError as error:
            _detach_exception_chain(error)
            raise

    def verify_investigation_result_with_evidence(
        self,
        session: InvestigationSession,
        result: InvestigationResult,
    ) -> VerifiedInvestigationResult:
        """核验结果并生成可直接写入 Working Memory 的安全证据摘要。

        输入是同一 Investigation session 产生的领域结果。动作通过当前 scope 的
        MiniChem verifier 逐项核对 job、operation、artifact 的真实终态和摘要，再
        生成强类型 ``VerifiedInvestigationEvidence``；摘要只保存稳定身份、终态和
        SHA-256，不保存 receipt 原文、token、workspace、路径或 transcript。相同
        输入和相同服务端核验结果会产生相同摘要；scope 中额外的历史终态 job 会
        参与摘要哈希，但不会成为 Result 的 job 引用。核验失败、thread/身份漂移或
        无法把 artifact 绑定到其生产作业及该作业提交 operation 时 fail-closed。
        """

        try:
            references = self._verify_result_references(session, result)
            evidence = self._build_verified_evidence(session, result, references)
            return VerifiedInvestigationResult(result=result, evidence=evidence)
        except ContractValidationError:
            raise InvestigationMiniChemProtocolError(
                "MiniChem 核验摘要不满足 Working Memory 证据合同"
            ) from None
        except InvestigationMiniChemGatewayError as error:
            _detach_exception_chain(error)
            raise

    def materialize_result_artifacts(
        self,
        session: InvestigationSession,
        result: InvestigationResult,
    ) -> Mapping[str, EvidenceArtifactManifest]:
        """核验并登记结果 artifact，返回 source ref 到 Evidence manifest 的映射。

        该动作只接受已经绑定到当前 scope 的领域结果。服务端先核对 job、operation
        和 artifact 的完整来源链，再分页读取每个 UTF-8 artifact，最后通过
        ``ReviewEvidenceStore`` 原子登记内容摘要。返回值不含路径、token 或原始
        receipt；任何 scope 漂移、内容摘要不一致或登记失败都会 fail-closed。
        """

        try:
            if not isinstance(session, InvestigationSession):
                raise TypeError("session 必须是 InvestigationSession")
            if not isinstance(result, InvestigationResult):
                raise TypeError("result 必须是 InvestigationResult")
            references = self._verify_result_references(session, result)
            digests = references.artifact_digests
            binding = self._resolve_scope_binding(session)
            manifests: dict[str, EvidenceArtifactManifest] = {}
            for artifact_id in result.artifact_refs:
                expected_sha256 = digests.get(artifact_id)
                if expected_sha256 is None:
                    raise InvestigationMiniChemProtocolError(
                        "artifact canonical sha256 缺失"
                    )
                content, media_type, _ = self._read_complete_artifact_text(
                    binding,
                    artifact_id,
                    expected_sha256,
                )
                source_path = self._atomic_write_source_cache(artifact_id, content)
                try:
                    manifest = self._evidence_store.register_source_artifact(
                        source_path,
                        source_artifact_ref=artifact_id,
                        expected_source_sha256=expected_sha256,
                        media_type=media_type,
                    )
                except BaseException:
                    raise InvestigationMiniChemGatewayError(
                        "Evidence source artifact 登记失败"
                    ) from None
                if not isinstance(manifest, EvidenceArtifactManifest):
                    raise InvestigationMiniChemProtocolError(
                        "Evidence store 返回值类型漂移"
                    )
                manifests[artifact_id] = manifest
            return manifests
        except InvestigationMiniChemGatewayError as error:
            _detach_exception_chain(error)
            raise

    def _verify_result_references(
        self,
        session: InvestigationSession,
        result: InvestigationResult,
    ) -> _VerifiedReferenceSummary:
        """调用一次权威 verifier，返回不含原始响应的规范引用摘要。"""

        if not isinstance(session, InvestigationSession):
            raise TypeError("session 必须是 InvestigationSession")
        if not isinstance(result, InvestigationResult):
            raise TypeError("result 必须是 InvestigationResult")
        self._validate_result_identity(session, result)
        _validate_reference_ids(result)
        response = self._call_scope_tool(
            self._resolve_scope_binding(session),
            _VERIFY_TOOL,
            {
                "jobs": list(result.job_refs),
                "operations": list(result.operation_refs),
                "artifacts": list(result.artifact_refs),
            },
        )
        return self._validate_reference_response(
            response,
            expected_jobs=result.job_refs,
            expected_operations=result.operation_refs,
            expected_artifacts=result.artifact_refs,
            allow_historical_terminal_jobs=True,
            allow_historical_operations=True,
        )

    def _build_verified_evidence(
        self,
        session: InvestigationSession,
        result: InvestigationResult,
        references: _VerifiedReferenceSummary,
    ) -> VerifiedInvestigationEvidence:
        """把 verifier 摘要转换成 memory 层要求的最小安全 payload。"""

        _validate_evidence_reference_bindings(result, references)
        evidence_artifacts = [dict(item) for item in references.artifacts]
        digest_input = {
            "task_id": result.task_id,
            "batch_id": result.batch_id,
            "combination_id": result.combination_id,
            "codex_thread_id": result.codex_thread_id,
            "scope_id": result.scope_id,
            "turn_id": result.turn_id,
            "jobs": references.jobs,
            "operations": references.operations,
            "artifacts": evidence_artifacts,
            "scope_job_inventory": references.scope_job_inventory,
            "scope_operation_inventory": references.scope_operation_inventory,
            "reference_binding_version": references.reference_binding_version,
        }
        verification_digest = _stable_sha256(digest_input)
        payload = {
            "task_id": session.task_id,
            "batch_id": session.batch_id,
            "combination_id": session.combination_id,
            "codex_thread_id": session.codex_thread_id,
            "scope_id": session.scope_id,
            "turn_id": result.turn_id,
            "verification_object_id": f"verification-{verification_digest[:32]}",
            "verifier_id": "minichem-authority-adapter",
            "verifier_version": "reference-v2",
            "verification_digest": verification_digest,
            "jobs": list(references.jobs),
            "operations": list(references.operations),
            "artifacts": evidence_artifacts,
            "operation_instance_id": f"evidence-{verification_digest[:32]}",
            "reference_binding_version": references.reference_binding_version,
        }
        return VerifiedInvestigationEvidence.from_authoritative_summary(payload)

    def verify_scope_execution_inventory(
        self,
        session: InvestigationSession,
        expected_job_refs: Sequence[str] = (),
        expected_operation_refs: Sequence[str] = (),
    ) -> Mapping[str, Any]:
        """核对当前 scope 的终态 inventory，作为 Investigation side-effect fence。

        ``created`` session 只能核对空 job/operation 清单；``repair_pending`` session
        必须由调用方传入 state 中已保存当前 InvestigationResult 的两类引用。方法只
        发送 ID-only verifier 请求，严格核对 job/operation inventory、终态和响应摘要，
        不构造伪造结果，也不读取 artifact 正文。成功返回不含路径、token 的 inventory
        manifest；任何 active、腐坏、漏报、多报、协议漂移或 scope 漂移都会 fail-closed。
        与结果核验不同，side-effect fence 不接受额外历史终态 job。
        """

        try:
            if not isinstance(session, InvestigationSession):
                raise TypeError("session 必须是 InvestigationSession")
            jobs = _normalize_reference_ids(
                expected_job_refs,
                "expected_job_refs",
                _JOB_ID_RE,
            )
            operations = _normalize_reference_ids(
                expected_operation_refs,
                "expected_operation_refs",
                _OPERATION_ID_RE,
            )
            if session.phase == "created":
                if jobs or operations:
                    raise InvestigationMiniChemScopeError(
                        "created session 的 side-effect fence 必须使用空 inventory"
                    )
            elif session.phase != "repair_pending":
                raise InvestigationMiniChemScopeError(
                    "side-effect fence 只接受 created 或 repair_pending session"
                )
            binding = self._resolve_scope_binding(session)
            response = self._call_scope_tool(
                binding,
                _VERIFY_TOOL,
                {
                    "jobs": list(jobs),
                    "operations": list(operations),
                    "artifacts": [],
                },
            )
            self._validate_reference_response(
                response,
                expected_jobs=jobs,
                expected_operations=operations,
                expected_artifacts=(),
            )
            return {
                "scope_id": session.scope_id,
                "scope_job_inventory": response["scope_job_inventory"],
                "has_active_jobs": response["has_active_jobs"],
                "inventory_complete": response["inventory_complete"],
                "scope_operation_inventory": response["scope_operation_inventory"],
                "has_active_operations": response["has_active_operations"],
                "operation_inventory_complete": response["operation_inventory_complete"],
            }
        except InvestigationMiniChemGatewayError as error:
            _detach_exception_chain(error)
            raise

    def _verify_result_artifacts(
        self,
        session: InvestigationSession,
        result: InvestigationResult,
    ) -> dict[str, str]:
        """完整核验一个结果的三类引用并返回 artifact canonical SHA 映射。"""

        _validate_reference_ids(result)
        response = self._call_scope_tool(
            self._resolve_scope_binding(session),
            _VERIFY_TOOL,
            {
                "jobs": list(result.job_refs),
                "operations": list(result.operation_refs),
                "artifacts": list(result.artifact_refs),
            },
        )
        return self._validate_reference_response(
            response,
            expected_jobs=result.job_refs,
            expected_operations=result.operation_refs,
            expected_artifacts=result.artifact_refs,
            allow_historical_terminal_jobs=True,
        ).artifact_digests

    def _read_complete_artifact_text(
        self,
        binding: InvestigationScopeBinding,
        artifact_id: str,
        expected_sha256: str,
    ) -> tuple[bytes, str, str]:
        """从 offset=0 分页读取完整 UTF-8 artifact 并核对稳定元数据。"""

        offset = 0
        page_count = 0
        encoded_size = 0
        chunks: list[str] = []
        expected_metadata: tuple[str, str, int] | None = None
        while True:
            page_count += 1
            if page_count > self.max_pages_per_artifact:
                raise InvestigationMiniChemProtocolError("artifact 分页数量超过上限")
            page = self._read_artifact_page(binding, artifact_id, expected_sha256, offset)
            metadata = (page.media_type, page.digest, page.total_chars)
            if expected_metadata is None:
                expected_metadata = metadata
            elif metadata != expected_metadata:
                raise InvestigationMiniChemProtocolError("artifact canonical metadata 跨页漂移")
            chunks.append(page.content)
            try:
                encoded_size += len(page.content.encode("utf-8"))
            except UnicodeEncodeError:
                raise InvestigationMiniChemProtocolError("artifact 内容不是有效 UTF-8") from None
            if encoded_size > self.max_artifact_bytes:
                raise InvestigationMiniChemProtocolError("artifact 字节大小超过上限")
            if page.next_offset is None:
                break
            offset = page.next_offset
        text = "".join(chunks)
        raw = text.encode("utf-8")
        if (
            expected_metadata is None
            or len(text) != expected_metadata[2]
            or hashlib.sha256(raw).hexdigest() != expected_sha256
        ):
            raise InvestigationMiniChemProtocolError("artifact 重构摘要与 canonical 摘要不一致")
        return raw, expected_metadata[0].split(";", 1)[0].strip(), expected_sha256

    def _read_artifact_page(
        self,
        binding: InvestigationScopeBinding,
        artifact_id: str,
        expected_sha256: str,
        offset: int,
    ) -> _ArtifactPage:
        """读取并校验一个 artifact 分页。"""

        response = self._call_scope_tool(
            binding,
            _READ_TOOL,
            {
                "artifact_id": artifact_id,
                "offset": offset,
                "limit": self.page_size,
                "expected_sha256": expected_sha256,
            },
        )
        return _parse_artifact_page_response(
            response,
            artifact_id=artifact_id,
            expected_sha256=expected_sha256,
            expected_offset=offset,
            page_size=self.page_size,
            max_artifact_bytes=self.max_artifact_bytes,
        )

    def _atomic_write_source_cache(self, artifact_id: str, content: bytes) -> Path:
        """把一个已核验 artifact 原子写入 0700/0600 source cache。"""

        filename = "source-" + hashlib.sha256(artifact_id.encode("utf-8")).hexdigest() + ".bin"
        target = self._source_cache_dir / filename
        try:
            if target.is_symlink() or (target.exists() and not target.is_file()):
                raise InvestigationMiniChemProtocolError("source cache 目标不是普通文件")
            temporary = self._source_cache_dir / f".{filename}.{uuid.uuid4().hex}.tmp"
            descriptor = os.open(
                temporary,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
                _SOURCE_CACHE_FILE_MODE,
            )
            try:
                os.fchmod(descriptor, _SOURCE_CACHE_FILE_MODE)
                with os.fdopen(descriptor, "wb", closefd=True) as handle:
                    descriptor = -1
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, target)
                os.chmod(target, _SOURCE_CACHE_FILE_MODE)
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
                try:
                    if temporary.exists() and not temporary.is_symlink():
                        temporary.unlink()
                except OSError:
                    pass
            return target
        except InvestigationMiniChemGatewayError:
            raise
        except (OSError, ValueError):
            raise InvestigationMiniChemUnavailable("source cache 无法原子写入") from None

    def _normalize_snapshot_inputs(
        self,
        state: InvestigationState,
        current_results: Sequence[InvestigationResult],
        result_version_vector: Sequence[tuple[str, str]],
    ) -> tuple[tuple[InvestigationResult, ...], tuple[tuple[str, str], ...], dict[str, InvestigationSession]]:
        """核对 Review 的五个当前结果、版本向量和 session identity。"""

        _validate_snapshot_input_shape(state, current_results, result_version_vector)
        sessions = _index_snapshot_sessions(state)
        vector = _normalize_snapshot_version_vector(result_version_vector)
        results_by_combination = _index_snapshot_results(current_results, sessions)
        expected_vector = tuple(
            (session.combination_id, session.result_version or "") for session in state.sessions
        )
        if any(not version for _, version in expected_vector) or dict(vector) != dict(expected_vector):
            raise InvestigationMiniChemProtocolError("result_version_vector 与 state 漂移")
        ordered_results = self._validate_ordered_snapshot_results(
            state,
            sessions,
            results_by_combination,
        )
        return ordered_results, expected_vector, sessions

    def _validate_ordered_snapshot_results(
        self,
        state: InvestigationState,
        sessions: Mapping[str, InvestigationSession],
        results_by_combination: Mapping[str, InvestigationResult],
    ) -> tuple[InvestigationResult, ...]:
        """按 state 顺序核对结果版本、thread 和已保存结果内容。"""

        ordered_results: list[InvestigationResult] = []
        saved_results = {
            (item.combination_id, item.result_version): item for item in state.results
        }
        for session in state.sessions:
            result = results_by_combination[session.combination_id]
            self._validate_result_identity(session, result)
            if result.result_version != session.result_version:
                raise InvestigationMiniChemProtocolError("current result version 与 state 漂移")
            saved = saved_results.get((result.combination_id, result.result_version))
            if saved is None or saved != result:
                raise InvestigationMiniChemProtocolError("current result 不是 state 已保存结果")
            ordered_results.append(result)
        return tuple(ordered_results)

    def build_evidence_snapshot(
        self,
        state: InvestigationState,
        current_results: Sequence[InvestigationResult],
        result_version_vector: Sequence[tuple[str, str]],
    ) -> Mapping[str, Any]:
        """核验并缓存五项结果的 artifact，创建 path-free 幂等 evidence snapshot。

        每个结果只通过其 session scope 核验 artifact refs，然后从 offset=0 按
        page size 读取完整 UTF-8 内容；分页元数据、canonical media/SHA、总字符数、
        顺序、大小和页数都严格检查。内容先以 0600 文件原子写入调用方提供的
        0700 source cache，再调用 ``ReviewEvidenceStore.register_source_artifact``
        和 ``create_batch_evidence_snapshot``。返回值只包含公开 manifest，绝不返回
        cache 路径、token 或 workspace。
        """

        try:
            return self._build_evidence_snapshot(
                state,
                current_results,
                result_version_vector,
            )
        except InvestigationMiniChemGatewayError as error:
            _detach_exception_chain(error)
            raise

    def _build_evidence_snapshot(
        self,
        state: InvestigationState,
        current_results: Sequence[InvestigationResult],
        result_version_vector: Sequence[tuple[str, str]],
    ) -> Mapping[str, Any]:
        """执行 evidence snapshot 内部阶段；公开入口负责脱敏异常链。"""

        ordered_results, vector, sessions = self._normalize_snapshot_inputs(
            state, current_results, result_version_vector
        )
        artifact_groups = self._prepare_snapshot_artifact_groups(ordered_results)
        seen_artifacts: set[str] = set()
        for result in ordered_results:
            session = sessions[result.combination_id]
            self._populate_snapshot_artifact_group(
                result,
                session,
                artifact_groups,
                seen_artifacts,
            )
        return self._create_public_evidence_manifest(
            state.task_id,
            state.batch_id,
            vector,
            artifact_groups,
        )

    @staticmethod
    def _prepare_snapshot_artifact_groups(
        ordered_results: Sequence[InvestigationResult],
    ) -> dict[tuple[str, str], list[Any]]:
        """校验五项结果的引用，并建立 pair 到 artifact manifest 的空分组。"""

        for result in ordered_results:
            _validate_reference_ids(result)
        all_artifact_refs = [
            artifact_id
            for result in ordered_results
            for artifact_id in result.artifact_refs
        ]
        if len(all_artifact_refs) != len(set(all_artifact_refs)):
            raise InvestigationMiniChemProtocolError("source artifact ref 不能跨 result pair 复用")
        return {
            (result.combination_id, result.result_version): []
            for result in ordered_results
        }

    def _populate_snapshot_artifact_group(
        self,
        result: InvestigationResult,
        session: InvestigationSession,
        artifact_groups: dict[tuple[str, str], list[Any]],
        seen_artifacts: set[str],
    ) -> None:
        """核验一个结果并把其完整 artifact 内容登记到对应 pair。"""

        pair = (result.combination_id, result.result_version)
        digests = self._verify_result_artifacts(session, result)
        if not result.artifact_refs:
            return
        binding = self._resolve_scope_binding(session)
        for artifact_id in result.artifact_refs:
            self._append_snapshot_artifact(
                binding,
                artifact_id,
                digests,
                seen_artifacts,
                artifact_groups[pair],
            )

    def _append_snapshot_artifact(
        self,
        binding: InvestigationScopeBinding,
        artifact_id: str,
        digests: Mapping[str, str],
        seen_artifacts: set[str],
        target_group: list[Any],
    ) -> None:
        """读取、缓存并登记一个已核验 artifact。"""

        if artifact_id in seen_artifacts:
            raise InvestigationMiniChemProtocolError("source artifact ref 不能跨 result pair 复用")
        seen_artifacts.add(artifact_id)
        expected_sha256 = digests.get(artifact_id)
        if expected_sha256 is None:
            raise InvestigationMiniChemProtocolError("artifact canonical sha256 缺失")
        content, media_type, _ = self._read_complete_artifact_text(
            binding,
            artifact_id,
            expected_sha256,
        )
        source_path = self._atomic_write_source_cache(artifact_id, content)
        try:
            manifest = self._evidence_store.register_source_artifact(
                source_path,
                source_artifact_ref=artifact_id,
                expected_source_sha256=expected_sha256,
                media_type=media_type,
            )
        except BaseException:
            raise InvestigationMiniChemGatewayError("Evidence source artifact 登记失败") from None
        target_group.append(manifest)

    def _create_public_evidence_manifest(
        self,
        task_id: str,
        batch_id: str,
        vector: Sequence[tuple[str, str]],
        artifact_groups: Mapping[tuple[str, str], Sequence[Any]],
    ) -> Mapping[str, Any]:
        """创建 evidence snapshot 并验证公开 manifest 不含 capability。"""

        try:
            snapshot = self._evidence_store.create_batch_evidence_snapshot(
                task_id,
                batch_id,
                vector,
                artifact_groups,
            )
        except BaseException:
            raise InvestigationMiniChemGatewayError("Evidence batch snapshot 创建失败") from None
        if not isinstance(snapshot, BatchEvidenceSnapshot):
            raise InvestigationMiniChemProtocolError("Evidence store 返回值类型漂移")
        manifest = snapshot.to_public_manifest()
        encoded = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if "X-MiniChem-Capability" in encoded or "x-minichem-capability" in encoded:
            raise InvestigationMiniChemProtocolError("公开 evidence manifest 泄露 capability")
        return manifest


def _validate_reference_response_header(
    response: Mapping[str, Any],
    *,
    expected: Sequence[tuple[str, str]],
) -> int:
    """核对 verifier 响应的顶层字段、schema、成功状态和回退标记。"""

    required = {
        "schema_version",
        "status",
        "all_valid",
        "items",
        "scope_job_inventory",
        "has_active_jobs",
        "inventory_complete",
        "scope_operation_inventory",
        "has_active_operations",
        "operation_inventory_complete",
    }
    if set(response) - (required | {"automatic_fallback"}) or not required <= set(response):
        raise InvestigationMiniChemProtocolError("verify_execution_references 响应字段漂移")
    schema_version = response.get("schema_version")
    if type(schema_version) is not int or schema_version not in {1, 2}:
        raise InvestigationMiniChemProtocolError(
            "verify_execution_references schema_version 漂移"
        )
    if response.get("status") != "success" or type(response.get("all_valid")) is not bool:
        raise InvestigationMiniChemProtocolError("verify_execution_references 状态非法")
    if response.get("automatic_fallback", False) is not False:
        raise InvestigationMiniChemProtocolError("只读 reference verification 不允许自动回退")
    if response.get("all_valid") is not True:
        raise InvestigationMiniChemReferenceRejected(
            _parse_rejected_reference_items(response, expected)
        )
    return schema_version


def _parse_rejected_reference_items(
    response: Mapping[str, Any],
    expected: Sequence[tuple[str, str]],
) -> tuple[Mapping[str, str], ...]:
    """把结构正确的 verifier 拒绝项压缩为可回传模型的安全反馈。"""

    items = response.get("items")
    if not isinstance(items, list) or len(items) != len(expected):
        raise InvestigationMiniChemProtocolError(
            "MiniChem reference verification 未完整通过"
        )
    rejected: list[Mapping[str, str]] = []
    id_fields = {
        "job": "job_id",
        "operation": "operation_id",
        "artifact": "artifact_id",
    }
    for (expected_kind, expected_id), raw_item in zip(expected, items):
        if not isinstance(raw_item, Mapping) or raw_item.get("kind") != expected_kind:
            raise InvestigationMiniChemProtocolError(
                "MiniChem reference verification 拒绝项结构非法"
            )
        status = raw_item.get("status")
        if status == "valid":
            continue
        if status == "unavailable":
            if set(raw_item) != {"kind", "status"}:
                raise InvestigationMiniChemProtocolError(
                    "MiniChem reference verification 拒绝原因非法"
                )
            reason_code = "reference_unavailable"
        elif status == "invalid":
            reason_code = raw_item.get("reason_code")
        else:
            reason_code = None
        if (
            not isinstance(reason_code, str)
            or _REASON_CODE_RE.fullmatch(reason_code) is None
        ):
            raise InvestigationMiniChemProtocolError(
                "MiniChem reference verification 拒绝原因非法"
            )
        rejected.append(
            {
                "kind": expected_kind,
                id_fields[expected_kind]: expected_id,
                "reason_code": reason_code,
            }
        )
    if not rejected:
        rejected.extend(_parse_missing_operation_rejections(response, expected))
    if not rejected:
        raise InvestigationMiniChemProtocolError(
            "MiniChem reference verification 拒绝结果与逐项状态不一致"
        )
    return tuple(rejected)


def _parse_missing_operation_rejections(
    response: Mapping[str, Any],
    expected: Sequence[tuple[str, str]],
) -> tuple[Mapping[str, str], ...]:
    """把完整且静止的 scope operation 集合差异转换为 Result 修正反馈。"""

    if (
        response.get("inventory_complete") is not True
        or response.get("has_active_jobs") is not False
        or response.get("operation_inventory_complete") is not True
        or response.get("has_active_operations") is not False
    ):
        return ()
    inventory = response.get("scope_operation_inventory")
    if not isinstance(inventory, list):
        return ()
    inventory_ids: list[str] = []
    for item in inventory:
        if (
            not isinstance(item, Mapping)
            or set(item) != {"operation_id", "status"}
            or item.get("status") != "completed"
        ):
            return ()
        operation_id = item.get("operation_id")
        if (
            not isinstance(operation_id, str)
            or _OPERATION_ID_RE.fullmatch(operation_id) is None
            or operation_id in inventory_ids
        ):
            return ()
        inventory_ids.append(operation_id)
    expected_ids = tuple(
        identity for kind, identity in expected if kind == "operation"
    )
    missing = tuple(
        operation_id
        for operation_id in inventory_ids
        if operation_id not in expected_ids
    )
    unexpected = tuple(
        operation_id
        for operation_id in expected_ids
        if operation_id not in inventory_ids
    )
    return tuple(
        {
            "kind": "operation",
            "operation_id": operation_id,
            "reason_code": reason_code,
        }
        for reason_code, operation_ids in (
            ("operation_not_declared", missing),
            ("operation_not_in_scope_inventory", unexpected),
        )
        for operation_id in operation_ids
    )


def _parse_artifact_page_response(
    response: Mapping[str, Any],
    *,
    artifact_id: str,
    expected_sha256: str,
    expected_offset: int,
    page_size: int,
    max_artifact_bytes: int,
) -> _ArtifactPage:
    """核对一页 artifact 的字段、canonical 元数据和游标关系。"""

    _validate_artifact_page_response_shape(response)
    media_type, digest, total_chars = _parse_artifact_page_metadata(
        response,
        artifact_id=artifact_id,
        expected_sha256=expected_sha256,
        max_artifact_bytes=max_artifact_bytes,
    )
    content, returned_chars = _parse_artifact_page_content(
        response,
        expected_offset=expected_offset,
        page_size=page_size,
    )
    next_offset = response.get("next_offset")
    truncated = response.get("truncated")
    _validate_artifact_page_cursor(
        expected_offset,
        returned_chars,
        next_offset,
        truncated,
        total_chars,
    )
    return _ArtifactPage(media_type, digest, total_chars, content, next_offset)


def _validate_artifact_page_response_shape(response: Mapping[str, Any]) -> None:
    """核对 artifact 分页的公开字段、schema 和自动回退标记。"""

    required = {
        "schema_version",
        "status",
        "artifact_id",
        "media_type",
        "sha256",
        "encoding",
        "offset",
        "returned_chars",
        "next_offset",
        "total_chars",
        "truncated",
        "content",
    }
    allowed = required | {"semantic_type", "automatic_fallback"}
    if set(response) - allowed or not required <= set(response):
        raise InvestigationMiniChemProtocolError("artifact text 响应字段漂移")
    if response.get("schema_version") != 1 or response.get("status") != "success":
        raise InvestigationMiniChemProtocolError("artifact text 响应状态非法")
    if response.get("automatic_fallback", False) is not False:
        raise InvestigationMiniChemProtocolError("artifact text 不允许自动回退")


def _parse_artifact_page_content(
    response: Mapping[str, Any],
    *,
    expected_offset: int,
    page_size: int,
) -> tuple[str, int]:
    """核对 artifact 分页正文、offset 和返回字符数。"""

    content = response.get("content")
    returned_chars = response.get("returned_chars")
    if (
        not isinstance(content, str)
        or _CONTROL_RE.search(content)
        or type(response.get("offset")) is not int
        or response.get("offset") != expected_offset
        or type(returned_chars) is not int
        or returned_chars != len(content)
        or returned_chars > page_size
    ):
        raise InvestigationMiniChemProtocolError("artifact 分页内容或长度非法")
    return content, returned_chars


def _parse_artifact_page_metadata(
    response: Mapping[str, Any],
    *,
    artifact_id: str,
    expected_sha256: str,
    max_artifact_bytes: int,
) -> tuple[str, str, int]:
    """核对 artifact 分页必须跨页保持不变的 canonical 元数据。"""

    media_type = response.get("media_type")
    digest = response.get("sha256")
    total_chars = response.get("total_chars")
    if not _artifact_page_metadata_matches(
        response,
        artifact_id,
        expected_sha256,
        media_type,
        digest,
        total_chars,
    ):
        raise InvestigationMiniChemProtocolError("artifact canonical metadata 漂移")
    if total_chars > max_artifact_bytes:
        raise InvestigationMiniChemProtocolError("artifact 字符大小超过上限")
    return media_type.strip().lower(), digest, total_chars


def _artifact_page_metadata_matches(
    response: Mapping[str, Any],
    artifact_id: str,
    expected_sha256: str,
    media_type: Any,
    digest: Any,
    total_chars: Any,
) -> bool:
    """返回一页 canonical artifact 元数据是否与期望完全一致。"""

    return (
        response.get("artifact_id") == artifact_id
        and isinstance(media_type, str)
        and _is_supported_text_media_type(media_type)
        and isinstance(digest, str)
        and _SHA256_RE.fullmatch(digest) is not None
        and digest == expected_sha256
        and response.get("encoding") == "utf-8"
        and type(total_chars) is int
        and total_chars >= 0
    )


def _validate_artifact_page_cursor(
    offset: int,
    returned_chars: int,
    next_offset: Any,
    truncated: Any,
    total_chars: int,
) -> None:
    """核对一页的 offset、next_offset、truncated 和 EOF 关系。"""

    if type(truncated) is not bool:
        raise InvestigationMiniChemProtocolError("artifact 分页 truncated 类型非法")
    if next_offset is not None:
        _validate_artifact_page_continuation(
            offset,
            returned_chars,
            next_offset,
            truncated,
            total_chars,
        )
        return
    if truncated is not False or offset + returned_chars != total_chars:
        raise InvestigationMiniChemProtocolError("artifact EOF 分页元数据非法")


def _validate_artifact_page_continuation(
    offset: int,
    returned_chars: int,
    next_offset: Any,
    truncated: bool,
    total_chars: int,
) -> None:
    """核对非 EOF 分页的连续游标和截断标记。"""

    if (
        type(next_offset) is not int
        or next_offset != offset + returned_chars
        or next_offset <= offset
        or next_offset >= total_chars
        or truncated is not True
    ):
        raise InvestigationMiniChemProtocolError("artifact 分页存在重叠或缺页")


def _expected_reference_items(
    expected_jobs: Sequence[str],
    expected_operations: Sequence[str],
    expected_artifacts: Sequence[str],
) -> tuple[tuple[str, str], ...]:
    """按公开协议冻结 jobs→operations→artifacts 的响应顺序。"""

    return tuple(
        [("job", value) for value in expected_jobs]
        + [("operation", value) for value in expected_operations]
        + [("artifact", value) for value in expected_artifacts]
    )


def _parse_reference_items(
    items: Any,
    expected: Sequence[tuple[str, str]],
    *,
    schema_version: int = 1,
    expected_operation_ids: Sequence[str] = (),
) -> dict[str, list[dict[str, str]]]:
    """解析 verifier 逐项响应并核对 artifact 的生产作业来源链。

    schema 2 使用 ``submission_operation_id``、``producer_job_id`` 和
    ``producer_submission_operation_id`` 明确区分科学 operation 与异步提交
    operation。schema 1 只作为旧响应兼容入口；旧 artifact 必须能唯一匹配一个
    返回的 job 及其提交 operation，否则即使服务端声称 ``all_valid`` 也拒绝。
    """

    if not isinstance(items, list) or len(items) != len(expected):
        raise InvestigationMiniChemProtocolError("MiniChem reference verification 未完整通过")
    if schema_version not in {1, 2}:
        raise InvestigationMiniChemProtocolError("reference schema_version 不受支持")
    result = {"jobs": [], "operations": [], "artifacts": []}
    for index, ((kind, expected_id), item) in enumerate(zip(expected, items)):
        try:
            parsed = _parse_reference_item(
                kind,
                expected_id,
                item,
                schema_version=schema_version,
            )
        except InvestigationMiniChemProtocolError as error:
            raise InvestigationMiniChemProtocolError(
                f"MiniChem reference verification 第 {index + 1} 项不可用"
            ) from None
        result[parsed.pop("_collection")].append(parsed)
    _validate_producer_reference_chains(
        result,
        expected_job_ids=tuple(value for kind, value in expected if kind == "job"),
        schema_version=schema_version,
    )
    return result


def _parse_reference_item(
    kind: str,
    expected_id: str,
    item: Any,
    *,
    schema_version: int,
) -> dict[str, str]:
    """解析一个 job、operation 或 artifact 的终态摘要。"""

    if not isinstance(item, Mapping) or item.get("kind") != kind or item.get("status") != "valid":
        raise InvestigationMiniChemProtocolError("reference item 状态非法")
    if kind == "job":
        return _parse_job_reference(expected_id, item, schema_version=schema_version)
    if kind == "operation":
        return _parse_operation_reference(expected_id, item)
    return _parse_artifact_reference(expected_id, item, schema_version=schema_version)


def _parse_job_reference(
    expected_id: str,
    item: Mapping[str, Any],
    *,
    schema_version: int,
) -> dict[str, str]:
    """解析 job 的终态、提交 operation 归属和 fingerprint。"""

    base_fields = {
        "kind",
        "status",
        "job_id",
        "operation_fingerprint",
        "terminal_status",
    }
    if schema_version == 2:
        fields = base_fields
        operation_field = "submission_operation_id"
        if set(item) not in (fields, fields | {operation_field}):
            raise InvestigationMiniChemProtocolError("job reference 字段漂移")
    else:
        fields = base_fields | {"operation_id"}
        operation_field = "operation_id"
    if schema_version == 1 and set(item) != fields:
        raise InvestigationMiniChemProtocolError("job reference 字段或 ID 漂移")
    if item.get("job_id") != expected_id:
        raise InvestigationMiniChemProtocolError("job reference 字段或 ID 漂移")
    submission_operation_id = item.get(operation_field)
    fingerprint = item.get("operation_fingerprint")
    terminal_status = item.get("terminal_status")
    if (
        (
            submission_operation_id is not None
            and (
                not isinstance(submission_operation_id, str)
                or _OPERATION_ID_RE.fullmatch(submission_operation_id) is None
                or not submission_operation_id.startswith("scope-submission-")
            )
        )
        or not isinstance(fingerprint, str)
        or _SHA256_RE.fullmatch(fingerprint) is None
        or terminal_status not in _TERMINAL_JOB_STATUSES
    ):
        raise InvestigationMiniChemProtocolError("job reference canonical identity 缺失")
    result = {
        "_collection": "jobs",
        "job_id": expected_id,
        "terminal_status": _normalize_evidence_job_status(terminal_status),
        "receipt_digest": fingerprint,
    }
    if submission_operation_id is not None:
        result["submission_operation_id"] = submission_operation_id
    return result


def _parse_operation_reference(expected_id: str, item: Mapping[str, Any]) -> dict[str, str]:
    """解析 operation 的完成状态和 canonical fingerprint。"""

    fields = {"kind", "status", "operation_id", "fingerprint", "operation_status"}
    if set(item) != fields or item.get("operation_id") != expected_id:
        raise InvestigationMiniChemProtocolError("operation reference 字段或 ID 漂移")
    fingerprint = item.get("fingerprint")
    if item.get("operation_status") != "completed":
        raise InvestigationMiniChemProtocolError("operation reference 未完成")
    if not isinstance(fingerprint, str) or _SHA256_RE.fullmatch(fingerprint) is None:
        raise InvestigationMiniChemProtocolError("operation reference fingerprint 非法")
    return {
        "_collection": "operations",
        "operation_id": expected_id,
        "terminal_status": "completed",
        "operation_digest": fingerprint,
    }


def _normalize_evidence_job_status(status: str) -> str:
    """把 MiniChem job 终态映射到 Working Memory 的有限状态集合。"""

    if status in {"success", "completed", "ready"}:
        return "completed"
    if status in {"failed", "timeout"}:
        return "failed"
    if status in {"cancelled", "canceled"}:
        return "cancelled"
    raise InvestigationMiniChemProtocolError("job reference 不是可保存的终态")


def _parse_artifact_reference(
    expected_id: str,
    item: Mapping[str, Any],
    *,
    schema_version: int,
) -> dict[str, str]:
    """解析 artifact 摘要及其生产 job→提交 operation 来源链。"""

    base_fields = {"kind", "status", "artifact_id", "sha256"}
    new_fields = base_fields | {"producer_job_id", "producer_submission_operation_id"}
    legacy_fields = base_fields | {"operation_id"}
    if schema_version == 2:
        if set(item) != new_fields:
            raise InvestigationMiniChemProtocolError("artifact reference 字段漂移")
        producer_job_id = item.get("producer_job_id")
        producer_submission_operation_id = item.get("producer_submission_operation_id")
        if (
            not isinstance(producer_job_id, str)
            or _JOB_ID_RE.fullmatch(producer_job_id) is None
            or not isinstance(producer_submission_operation_id, str)
            or _OPERATION_ID_RE.fullmatch(producer_submission_operation_id) is None
        ):
            raise InvestigationMiniChemProtocolError("artifact producer identity 非法")
    else:
        item_fields = set(item)
        if item_fields != base_fields and item_fields != legacy_fields:
            raise InvestigationMiniChemProtocolError("artifact reference 字段漂移")
        # schema 1 没有 producer_job_id。这里只记录旧 operation，后面必须
        # 通过唯一 job 且 operation 完全相等的交叉核验才能补出 producer job。
        producer_job_id = ""
        producer_submission_operation_id = item.get("operation_id", "")
        if producer_submission_operation_id and (
            not isinstance(producer_submission_operation_id, str)
            or _OPERATION_ID_RE.fullmatch(producer_submission_operation_id) is None
        ):
            raise InvestigationMiniChemProtocolError("旧 artifact operation 归属非法")
    digest = item.get("sha256")
    if item.get("artifact_id") != expected_id:
        raise InvestigationMiniChemProtocolError("artifact reference 顺序或 ID 漂移")
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        raise InvestigationMiniChemProtocolError("artifact reference canonical sha256 缺失")
    return {
        "_collection": "artifacts",
        "artifact_id": expected_id,
        "producer_job_id": producer_job_id,
        "producer_submission_operation_id": producer_submission_operation_id,
        "artifact_digest": digest,
    }


def _validate_producer_reference_chains(
    parsed: Mapping[str, list[dict[str, str]]],
    *,
    expected_job_ids: Sequence[str],
    schema_version: int,
) -> None:
    """核对每个 artifact 都绑定到返回的唯一 producer job 及其提交 operation。

    InvestigationResult 的科学 operation 集合只用于核对 typed operation inventory，
    不参与 artifact producer operation 的所属判断。这样可以保留真实的异步提交
    operation，即使它不在科学结果声明中。
    """

    jobs = parsed["jobs"]
    job_by_id = {item["job_id"]: item for item in jobs}
    if len(job_by_id) != len(jobs) or set(job_by_id) != set(expected_job_ids):
        raise InvestigationMiniChemProtocolError("reference job producer inventory 不完整")
    submission_ids = [
        item.get("submission_operation_id")
        for item in jobs
        if item.get("submission_operation_id") is not None
    ]
    if len(set(submission_ids)) != len(submission_ids):
        raise InvestigationMiniChemProtocolError("多个 job 不能共用 submission operation")

    for artifact in parsed["artifacts"]:
        producer_job_id = artifact["producer_job_id"]
        producer_submission_operation_id = artifact[
            "producer_submission_operation_id"
        ]
        if schema_version == 1 and not producer_job_id:
            # 旧响应没有 producer_job_id；有旧 operation 时必须在返回 job 集合中
            # 恰好命中一个提交 operation。旧 artifact 连 operation_id 也没有时，
            # 只允许 inventory 恰好只有一个带提交 operation 的 job，不能猜测多 job
            # 响应中的来源。
            if producer_submission_operation_id:
                matching_jobs = [
                    job
                    for job in jobs
                    if job.get("submission_operation_id")
                    == producer_submission_operation_id
                ]
            elif len(jobs) == 1 and jobs[0].get("submission_operation_id"):
                matching_jobs = jobs
            else:
                matching_jobs = []
            if len(matching_jobs) != 1:
                raise InvestigationMiniChemProtocolError(
                    "旧 artifact 缺少唯一可核对的 producer job"
                )
            producer_job_id = matching_jobs[0]["job_id"]
            producer_submission_operation_id = matching_jobs[0][
                "submission_operation_id"
            ]
            artifact["producer_job_id"] = producer_job_id
            artifact["producer_submission_operation_id"] = (
                producer_submission_operation_id
            )
        if (
            not producer_job_id
            or producer_job_id not in job_by_id
            or not producer_submission_operation_id
            or producer_submission_operation_id
            != job_by_id[producer_job_id].get("submission_operation_id")
        ):
            raise InvestigationMiniChemProtocolError(
                "artifact producer job 与 submission operation 不一致"
            )


def _inventory_list(value: Any, field_name: str) -> list[dict[str, str]]:
    """从数组或 ``{"jobs": [...]}` 形式取出已通过协议核验的 inventory。"""

    if isinstance(value, Mapping):
        value = value.get(field_name)
    if not isinstance(value, list):
        raise InvestigationMiniChemProtocolError("inventory 不是数组")
    return [dict(item) for item in value]


def _validate_reference_response_inventories(
    response: Mapping[str, Any],
    expected_job_ids: Sequence[str],
    expected_operation_ids: Sequence[str],
    *,
    schema_version: int,
    expected_submission_operations: Mapping[str, str | None] | None = None,
    allow_historical_terminal_jobs: bool = False,
    allow_historical_operations: bool = False,
) -> None:
    """核对 scope inventory 无活动且覆盖当前引用。

    结果核验可显式允许额外的已终态历史 job 和未参与结论的探索 operation；
    side-effect fence 使用默认的精确集合合同，仍拒绝任何额外或活动项。
    """

    if type(response["has_active_jobs"]) is not bool or response["has_active_jobs"]:
        raise InvestigationMiniChemProtocolError("当前 scope 仍有未终态 job")
    if response["inventory_complete"] is not True:
        raise InvestigationMiniChemProtocolError("当前 scope job inventory 不完整")
    _validate_job_inventory(
        response["scope_job_inventory"],
        expected_job_ids,
        schema_version=schema_version,
        expected_submission_operations=expected_submission_operations or {},
        allow_historical_terminal_jobs=allow_historical_terminal_jobs,
    )
    has_active_operations = response["has_active_operations"]
    if type(has_active_operations) is not bool:
        raise InvestigationMiniChemProtocolError("当前 scope operation 活动状态非法")
    if not allow_historical_operations and has_active_operations:
        raise InvestigationMiniChemProtocolError("当前 scope 仍有活动 operation")
    if response["operation_inventory_complete"] is not True:
        raise InvestigationMiniChemProtocolError("当前 scope operation inventory 不完整")
    _validate_operation_inventory(
        response["scope_operation_inventory"],
        expected_operation_ids,
        allow_historical_operations=allow_historical_operations,
        advertised_has_active=has_active_operations,
    )


def _validate_snapshot_input_shape(
    state: InvestigationState,
    current_results: Sequence[InvestigationResult],
    result_version_vector: Sequence[tuple[str, str]],
) -> None:
    """核对 snapshot 输入与冻结 state 的当前批次数量一致。"""

    if not isinstance(state, InvestigationState):
        raise TypeError("state 必须是 InvestigationState")
    expected_count = state.expected_combination_count
    if (
        isinstance(current_results, (str, bytes, bytearray))
        or not isinstance(current_results, Sequence)
        or len(current_results) != expected_count
    ):
        raise InvestigationMiniChemProtocolError("current_results 数量与 state 不一致")
    if (
        isinstance(result_version_vector, (str, bytes, bytearray))
        or not isinstance(result_version_vector, Sequence)
        or len(result_version_vector) != expected_count
    ):
        raise InvestigationMiniChemProtocolError("result_version_vector 数量与 state 不一致")


def _index_snapshot_sessions(state: InvestigationState) -> dict[str, InvestigationSession]:
    """建立当前批次唯一 combination 到 session 的索引。"""

    sessions = {session.combination_id: session for session in state.sessions}
    if len(sessions) != state.expected_combination_count:
        raise InvestigationMiniChemProtocolError("state 的 Investigation session 数量不符合冻结任务")
    return sessions


def _normalize_snapshot_version_vector(
    result_version_vector: Sequence[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    """规范化当前批次 result version 向量并拒绝重复组合。"""

    vector: list[tuple[str, str]] = []
    for pair in result_version_vector:
        if not isinstance(pair, Sequence) or isinstance(pair, (str, bytes, bytearray)) or len(pair) != 2:
            raise InvestigationMiniChemProtocolError("result_version_vector 项必须是二元组")
        vector.append(
            (
                _require_identity_text(pair[0], "combination_id"),
                _require_identity_text(pair[1], "result_version"),
            )
        )
    if len(vector) not in {1, 5}:
        raise InvestigationMiniChemProtocolError("result_version_vector 必须包含一项或五项")
    if len({item[0] for item in vector}) != len(vector):
        raise InvestigationMiniChemProtocolError("result_version_vector 组合不能重复")
    return tuple(vector)


def _index_snapshot_results(
    current_results: Sequence[InvestigationResult],
    sessions: Mapping[str, InvestigationSession],
) -> dict[str, InvestigationResult]:
    """建立当前结果索引并核对组合覆盖。"""

    results: dict[str, InvestigationResult] = {}
    for result in current_results:
        if not isinstance(result, InvestigationResult):
            raise TypeError("current_results 必须是 InvestigationResult 数组")
        if result.combination_id in results:
            raise InvestigationMiniChemProtocolError("current_results 组合不能重复")
        results[result.combination_id] = result
    if set(results) != set(sessions):
        raise InvestigationMiniChemProtocolError("current_results 未完整覆盖 state session")
    return results


def _stable_sha256(value: Any) -> str:
    """对已脱敏 JSON-like 摘要计算稳定 SHA-256。"""

    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _validate_evidence_reference_bindings(
    result: InvestigationResult,
    references: _VerifiedReferenceSummary,
) -> None:
    """确认 typed operation 与 artifact producer 来源链各自完整且不串义。"""

    operation_ids = {item["operation_id"] for item in references.operations}
    if set(result.operation_refs) != operation_ids:
        raise InvestigationMiniChemProtocolError("权威 operation 集合与结果引用不一致")
    job_by_id = {item["job_id"]: item for item in references.jobs}
    if set(result.job_refs) != set(job_by_id):
        raise InvestigationMiniChemProtocolError("权威 job 集合与结果引用不一致")
    for artifact in references.artifacts:
        producer_job_id = artifact.get("producer_job_id")
        producer_submission_operation_id = artifact.get(
            "producer_submission_operation_id"
        )
        if (
            not isinstance(producer_job_id, str)
            or not producer_job_id
            or producer_job_id not in job_by_id
            or producer_submission_operation_id
            != job_by_id[producer_job_id].get("submission_operation_id")
        ):
            raise InvestigationMiniChemProtocolError(
                "artifact producer 来源链与权威 job 不一致"
            )


def _normalize_reference_ids(
    values: Sequence[str],
    label: str,
    pattern: re.Pattern[str],
) -> tuple[str, ...]:
    """校验并冻结一组 ID-only 引用，拒绝重复、空值和协议外标识。"""

    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise InvestigationMiniChemProtocolError(f"{label} 必须是字符串序列")
    normalized = tuple(values)
    seen: set[str] = set()
    for value in normalized:
        if (
            not isinstance(value, str)
            or not value.strip()
            or pattern.fullmatch(value) is None
        ):
            raise InvestigationMiniChemProtocolError(f"{label} 含无效引用")
        if value in seen:
            raise InvestigationMiniChemProtocolError(f"{label} 不能包含重复引用")
        seen.add(value)
    return normalized


def _validate_reference_ids(result: InvestigationResult) -> None:
    """拒绝同一结果内重复或非字符串的三类不透明引用。"""

    for values, label, pattern in (
        (result.job_refs, "job_refs", _JOB_ID_RE),
        (result.operation_refs, "operation_refs", _OPERATION_ID_RE),
        (result.artifact_refs, "artifact_refs", _ARTIFACT_ID_RE),
    ):
        _normalize_reference_ids(values, label, pattern)


def _extract_job_inventory_entries(value: Any) -> list[Any]:
    """提取 scope job inventory 的数组表示。"""

    if isinstance(value, Mapping):
        if set(value) - {"jobs"} or not isinstance(value.get("jobs"), list):
            raise InvestigationMiniChemProtocolError("job inventory 字段非法")
        return value["jobs"]
    if not isinstance(value, list):
        raise InvestigationMiniChemProtocolError("job inventory 必须是数组")
    return value


def _parse_job_inventory_item(
    item: Any,
    *,
    schema_version: int = 1,
) -> tuple[str, str, str | None]:
    """校验一个 job inventory 项并返回 job ID、终态和可选提交 operation。"""

    if not isinstance(item, Mapping) or not isinstance(item.get("job_id"), str):
        raise InvestigationMiniChemProtocolError("job inventory 项非法")
    fields = set(item)
    if schema_version == 1:
        if fields != {"job_id", "status"}:
            raise InvestigationMiniChemProtocolError("旧 job inventory 字段漂移")
    elif fields not in ({"job_id", "status"}, {"job_id", "status", "submission_operation_id"}):
        raise InvestigationMiniChemProtocolError("job inventory 字段漂移")
    job_id = item["job_id"]
    if _JOB_ID_RE.fullmatch(job_id) is None or not isinstance(item.get("status"), str):
        raise InvestigationMiniChemProtocolError("job inventory 标识或状态非法")
    status = item.get("terminal_status", item.get("status"))
    if status not in _TERMINAL_JOB_STATUSES:
        raise InvestigationMiniChemProtocolError("job inventory 含未终态 job")
    submission_operation_id = item.get("submission_operation_id")
    if submission_operation_id is not None and (
        not isinstance(submission_operation_id, str)
        or _OPERATION_ID_RE.fullmatch(submission_operation_id) is None
        or not submission_operation_id.startswith("scope-submission-")
    ):
        raise InvestigationMiniChemProtocolError("job inventory 提交 operation 非法")
    return job_id, status, submission_operation_id


def _validate_job_inventory(
    value: Any,
    expected_job_ids: Sequence[str],
    *,
    schema_version: int = 1,
    expected_submission_operations: Mapping[str, str | None] | None = None,
    allow_historical_terminal_jobs: bool = False,
) -> None:
    """严格核对 job 集合、终态和可见提交身份。

    默认要求 inventory 与当前引用精确一致；结果核验可显式允许额外的已终态
    历史 job，但仍要求所有当前引用都被覆盖并逐项通过协议检查。
    """

    entries = _extract_job_inventory_entries(value)
    expected_submission_operations = expected_submission_operations or {}
    seen: set[str] = set()
    for item in entries:
        job_id, _, submission_operation_id = _parse_job_inventory_item(
            item,
            schema_version=schema_version,
        )
        if job_id in seen:
            raise InvestigationMiniChemProtocolError("job inventory 不能重复 job")
        seen.add(job_id)
        if schema_version == 2 and job_id in expected_submission_operations:
            expected_submission_operation = expected_submission_operations[job_id]
            if expected_submission_operation is None:
                if submission_operation_id is not None:
                    raise InvestigationMiniChemProtocolError(
                        "无 scope-submission 身份的 job 不应额外声明提交 operation"
                    )
            elif (
                not expected_submission_operation.startswith("scope-submission-")
            ):
                if submission_operation_id is not None:
                    raise InvestigationMiniChemProtocolError(
                        "非 scope-submission job 不应额外声明提交 operation"
                    )
            elif submission_operation_id != expected_submission_operation:
                raise InvestigationMiniChemProtocolError(
                    "job inventory 提交 operation 与 job reference 不一致"
                )
    expected = set(expected_job_ids)
    if not expected.issubset(seen):
        raise InvestigationMiniChemProtocolError("job inventory 未完整覆盖当前引用")
    if not allow_historical_terminal_jobs and seen != expected:
        raise InvestigationMiniChemProtocolError("job inventory 与当前引用集合不一致")


def _validate_operation_inventory(
    value: Any,
    expected_operation_ids: Sequence[str],
    *,
    allow_historical_operations: bool = False,
    advertised_has_active: bool | None = None,
) -> None:
    """核对 operation inventory，并区分结果证据与 side-effect fence。"""

    if not isinstance(value, list):
        raise InvestigationMiniChemProtocolError("operation inventory 不是数组")
    expected = set(expected_operation_ids)
    seen: set[str] = set()
    has_active = False
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {"operation_id", "status"}:
            raise InvestigationMiniChemProtocolError("operation inventory 字段漂移")
        operation_id = item.get("operation_id")
        if not isinstance(operation_id, str) or _OPERATION_ID_RE.fullmatch(operation_id) is None:
            raise InvestigationMiniChemProtocolError("operation inventory 含无效 operation")
        if operation_id in seen:
            raise InvestigationMiniChemProtocolError("operation inventory 不能重复 operation")
        seen.add(operation_id)
        status = item.get("status")
        if status not in {"completed", "running"}:
            raise InvestigationMiniChemProtocolError("operation inventory 状态非法")
        if status == "running":
            has_active = True
            if operation_id in expected or not allow_historical_operations:
                raise InvestigationMiniChemProtocolError("显式引用的 operation 尚未完成")
    if advertised_has_active is not None and advertised_has_active is not has_active:
        raise InvestigationMiniChemProtocolError("operation inventory 活动状态不一致")
    if not expected.issubset(seen):
        raise InvestigationMiniChemProtocolError("operation inventory 未覆盖显式引用")
    if not allow_historical_operations and seen != expected:
        raise InvestigationMiniChemProtocolError("operation inventory 与显式引用不一致")


def _prepare_source_cache_dir(value: str | os.PathLike[str] | Path) -> Path:
    """创建/校验 0700 source cache，拒绝任一符号链接路径组件。"""

    try:
        candidate = Path(value).expanduser()
    except (TypeError, ValueError) as exc:
        raise ValueError("source_cache_dir 无效") from exc
    absolute = Path(os.path.abspath(os.fspath(candidate)))
    cursor = Path(absolute.anchor or os.curdir)
    parts = absolute.parts[1:] if absolute.anchor else absolute.parts
    for part in parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError("source_cache_dir 不能包含符号链接")
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        if candidate.is_symlink() or not candidate.is_dir():
            raise ValueError("source_cache_dir 必须是普通目录")
        os.chmod(candidate, _SOURCE_CACHE_DIR_MODE)
        if stat.S_IMODE(candidate.stat().st_mode) != _SOURCE_CACHE_DIR_MODE:
            raise ValueError("source_cache_dir 必须是 0700")
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError("source_cache_dir 不可用") from exc
    return candidate.resolve(strict=True)
