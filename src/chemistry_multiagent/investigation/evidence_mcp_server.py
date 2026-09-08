"""Investigation/Experiment 任务级只读 Review evidence MCP 服务。

服务只注册 ``read_snapshot_artifact`` 一个工具。来源路径登记和五项快照创建
仍由可信 Controller 调用 ``ReviewEvidenceStore`` 完成；MCP 客户端只能提交
opaque snapshot/artifact ID 以及有界读取参数。Streamable HTTP 入口在每一条
HTTP message 上重新校验 capability，并将请求限制在启动时冻结的 snapshot
allowlist 内，从而避免同一个长连接在切换任务后继续读取旧证据。
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import secrets
import stat
import sys
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from starlette.responses import JSONResponse

from .evidence import (
    EvidenceArtifactIntegrityError,
    EvidenceArtifactNotFound,
    EvidenceBudgetExceeded,
    EvidenceRegistryCorruption,
    EvidenceRequestError,
    EvidenceSnapshotNotFound,
    ReviewEvidenceStore,
    UnsupportedEvidenceArtifact,
)


__all__ = [
    "DEFAULT_EVIDENCE_CAPABILITY_HEADER",
    "EvidenceMcpConfigurationError",
    "EvidenceMcpServerConfig",
    "EvidenceMcpServerError",
    "EvidenceAuthenticatedHTTPApp",
    "create_evidence_capability_file",
    "load_evidence_capability_file",
    "load_snapshot_allowlist_file",
    "create_evidence_mcp_server",
    "create_evidence_http_app",
    "main",
]


DEFAULT_EVIDENCE_CAPABILITY_HEADER = "x-arche-evidence-capability"
_OPAQUE_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{7,127}$")
_CAPABILITY_RE = re.compile(r"^[\x21-\x7e]{32,512}$")
_HEADER_NAME_RE = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")
_MAX_ALLOWLIST_BYTES = 256 * 1024


def _reject_symlinked_path_components(path: Path, *, label: str) -> None:
    """拒绝认证相关路径的符号链接父目录，避免绕过本地权限边界。"""

    try:
        absolute = Path(os.path.abspath(os.fspath(path)))
        cursor = Path(absolute.anchor or os.curdir)
        components = absolute.parts[1:] if absolute.anchor else absolute.parts
        for component in components:
            cursor /= component
            if cursor.is_symlink():
                raise EvidenceMcpConfigurationError(f"{label} is invalid")
    except EvidenceMcpConfigurationError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise EvidenceMcpConfigurationError(f"{label} is invalid") from exc


class EvidenceMcpServerError(RuntimeError):
    """Evidence MCP 服务配置或请求认证失败。"""


class EvidenceMcpConfigurationError(EvidenceMcpServerError):
    """启动参数、快照 allowlist 或 capability 文件不满足安全约束。"""


class _EvidenceMcpRequestError(ValueError):
    """供 MCP 工具内部使用的安全错误，不包含路径或底层异常正文。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.public_message = message


@dataclass(frozen=True, slots=True)
class EvidenceMcpServerConfig:
    """一个 evidence MCP 进程的不可变访问边界。

    ``snapshot_ids`` 是启动时核对过的 path-free snapshot allowlist。stdio 可在
    单一固定 snapshot 下依赖进程管道；任何 Streamable HTTP app 都必须持有
    capability。类实例不保存 capability 明文，只保留 SHA-256 摘要，因此 repr、
    异常和测试输出不会暴露认证信息。
    """

    snapshot_ids: tuple[str, ...]
    task_id: str
    capability_header: str = DEFAULT_EVIDENCE_CAPABILITY_HEADER
    capability_digest: str | None = field(default=None, repr=False, compare=False)
    capability_required: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        normalized_ids = _normalize_server_snapshot_ids(self.snapshot_ids)
        normalized_task_id = _normalize_server_task_id(self.task_id)
        header = _normalize_server_capability_header(self.capability_header)
        digest = _validate_server_capability_binding(
            self.capability_digest,
            capability_required=self.capability_required,
            snapshot_count=len(normalized_ids),
        )
        object.__setattr__(self, "snapshot_ids", normalized_ids)
        object.__setattr__(self, "task_id", normalized_task_id)
        object.__setattr__(self, "capability_header", header)

    @property
    def capability_enabled(self) -> bool:
        """返回是否配置了 HTTP capability；不返回 capability 内容。"""

        return self.capability_digest is not None


def _normalize_server_snapshot_ids(values: Any) -> tuple[str, ...]:
    """去重并校验 MCP server 的 snapshot allowlist。"""

    normalized_ids = tuple(dict.fromkeys(values))
    if not normalized_ids or any(
        _OPAQUE_ID_RE.fullmatch(item) is None for item in normalized_ids
    ):
        raise EvidenceMcpConfigurationError("snapshot allowlist is invalid")
    return normalized_ids


def _normalize_server_task_id(value: Any) -> str:
    """校验并规范化 MCP server 的 task 绑定。"""

    if not isinstance(value, str) or not value.strip():
        raise EvidenceMcpConfigurationError("snapshot task binding is invalid")
    return value.strip()


def _normalize_server_capability_header(value: Any) -> str:
    """校验 capability HTTP header 名称并统一为小写。"""

    header = str(value).strip().lower()
    if _HEADER_NAME_RE.fullmatch(header) is None:
        raise EvidenceMcpConfigurationError("capability header is invalid")
    return header


def _validate_server_capability_binding(
    digest: Any,
    *,
    capability_required: Any,
    snapshot_count: int,
) -> str | None:
    """校验 capability 摘要、强制认证和多快照绑定关系。"""

    if digest is not None and (
        not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
    ):
        raise EvidenceMcpConfigurationError("capability binding is invalid")
    if capability_required and digest is None:
        raise EvidenceMcpConfigurationError("capability binding is missing")
    if snapshot_count > 1 and digest is None:
        raise EvidenceMcpConfigurationError("multiple snapshots require capability")
    return digest


_AUTHORIZED_REQUEST: ContextVar[bool] = ContextVar(
    "arche_evidence_mcp_authorized_request",
    default=False,
)


def _validate_opaque_id(value: Any, field_name: str) -> str:
    """校验工具参数是不透明标识，不接受路径、URL 或任意宿主文件名。"""

    if not isinstance(value, str) or _OPAQUE_ID_RE.fullmatch(value) is None:
        raise _EvidenceMcpRequestError("invalid_request", "evidence request is invalid")
    return value


def _validate_capability(value: Any) -> str:
    """校验可放入本地 0600 文件的 printable capability。"""

    if not isinstance(value, str) or _CAPABILITY_RE.fullmatch(value) is None:
        raise EvidenceMcpConfigurationError("capability file is invalid")
    return value


def _capability_digest(value: str) -> str:
    """只计算 capability 摘要，避免服务配置长期持有明文 token。"""

    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _check_private_file(
    path: Path,
    *,
    label: str,
    required_mode: int | None = None,
) -> None:
    """校验认证/allowlist 文件是非符号链接的私有普通文件。"""

    _reject_symlinked_path_components(path, label=label)
    try:
        if path.is_symlink() or not path.is_file():
            raise EvidenceMcpConfigurationError(f"{label} is invalid")
        mode = stat.S_IMODE(path.stat().st_mode)
    except EvidenceMcpConfigurationError:
        raise
    except OSError as exc:
        raise EvidenceMcpConfigurationError(f"{label} is unavailable") from exc
    if required_mode is not None and mode != required_mode:
        raise EvidenceMcpConfigurationError(f"{label} must be mode 0600")
    if mode & 0o077:
        raise EvidenceMcpConfigurationError(f"{label} must be mode 0600")


def load_evidence_capability_file(path: str | os.PathLike[str]) -> str:
    """从严格 0600 的本地文件读取 capability，不在错误中复制文件路径。

    输入是 Controller 预先写入的认证文件；动作只读取一行 printable token，
    不创建、修改或删除文件。文件不存在、是符号链接、权限过宽或内容非法时
    抛出不含本机路径的配置异常。
    """

    candidate = Path(path).expanduser()
    _reject_symlinked_path_components(candidate, label="capability file")
    _check_private_file(candidate, label="capability file", required_mode=0o600)
    try:
        if candidate.stat().st_size > 512:
            raise EvidenceMcpConfigurationError("capability file is too large")
        content = candidate.read_text(encoding="ascii")
    except EvidenceMcpConfigurationError:
        raise
    except (OSError, UnicodeDecodeError) as exc:
        raise EvidenceMcpConfigurationError("capability file is unavailable") from exc
    token = content.strip()
    if content != token + "\n":
        # 只允许一个末尾换行，不忽略 token 首尾的其他空白。
        raise EvidenceMcpConfigurationError("capability file is invalid")
    return _validate_capability(token)


def create_evidence_capability_file(
    path: str | os.PathLike[str],
    *,
    token: str | None = None,
) -> str:
    """以 0600 原子创建本地 capability 文件并返回明文给调用方。

    输入是专用凭据文件路径和可选 printable token；动作创建私有父目录、以
    ``O_EXCL`` 原子写入并 fsync 文件。已存在文件、符号链接或权限不安全时失败；
    返回值只供同一可信 Controller 立即注入 Codex MCP header，不会写入 prompt。
    """

    candidate = Path(path).expanduser()
    parent = candidate.parent
    try:
        _reject_symlinked_path_components(parent, label="capability directory")
        parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if parent.is_symlink() or not parent.is_dir():
            raise EvidenceMcpConfigurationError("capability directory is invalid")
        os.chmod(parent, 0o700)
    except EvidenceMcpConfigurationError:
        raise
    except OSError as exc:
        raise EvidenceMcpConfigurationError("capability directory is unavailable") from exc
    selected = _validate_capability(token if token is not None else secrets.token_urlsafe(48))
    try:
        descriptor = os.open(
            candidate,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except OSError as exc:
        raise EvidenceMcpConfigurationError("capability file cannot be created") from exc
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            descriptor = -1
            handle.write(selected + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return selected


def _normalize_snapshot_ids(values: Iterable[str]) -> tuple[str, ...]:
    """规范化显式 snapshot allowlist；拒绝空值、路径和重复项。"""

    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str) or _OPAQUE_ID_RE.fullmatch(value) is None:
            raise EvidenceMcpConfigurationError("snapshot allowlist is invalid")
        if value not in normalized:
            normalized.append(value)
    if not normalized:
        raise EvidenceMcpConfigurationError("snapshot allowlist is empty")
    return tuple(normalized)


def load_snapshot_allowlist_file(path: str | os.PathLike[str]) -> tuple[str, ...]:
    """读取明确列出的 snapshot ID allowlist，不枚举 registry 或宿主文件。

    文件必须是私有普通文件，内容可以是 JSON 字符串数组、包含
    ``snapshot_ids`` 数组的 JSON object，或每行一个 opaque ID；最多读取
    256 KiB。返回值只包含去重后的 ID，不返回文件路径或其它元数据。
    """

    candidate = Path(path).expanduser()
    _reject_symlinked_path_components(candidate, label="snapshot allowlist file")
    _check_private_file(candidate, label="snapshot allowlist file")
    try:
        if candidate.stat().st_size > _MAX_ALLOWLIST_BYTES:
            raise EvidenceMcpConfigurationError("snapshot allowlist file is too large")
        text = candidate.read_text(encoding="utf-8")
    except EvidenceMcpConfigurationError:
        raise
    except (OSError, UnicodeDecodeError) as exc:
        raise EvidenceMcpConfigurationError("snapshot allowlist file is unavailable") from exc
    values: Any
    try:
        values = json.loads(text)
    except json.JSONDecodeError:
        values = [line.strip() for line in text.splitlines() if line.strip()]
    if isinstance(values, Mapping):
        values = values.get("snapshot_ids")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise EvidenceMcpConfigurationError("snapshot allowlist file is invalid")
    return _normalize_snapshot_ids(values)


def _read_scope_headers(scope: Mapping[str, Any], header_name: str) -> tuple[str, ...]:
    """从 ASGI scope 读取唯一 capability header，拒绝重复头的歧义。"""

    values: list[str] = []
    wanted = header_name.encode("ascii")
    for raw_name, raw_value in scope.get("headers") or []:
        try:
            if bytes(raw_name).lower() != wanted:
                continue
            decoded = bytes(raw_value).decode("ascii").strip()
        except (TypeError, UnicodeDecodeError):
            return ()
        if decoded:
            values.append(decoded)
    return tuple(values)


def _authorize_capability(config: EvidenceMcpServerConfig, token: str | None) -> bool:
    """以 constant-time 摘要比较验证 capability，不把 token 放入异常或返回值。"""

    digest = config.capability_digest
    if digest is None:
        return not config.capability_required
    if token is None or _CAPABILITY_RE.fullmatch(token) is None:
        return False
    return hmac.compare_digest(_capability_digest(token), digest)


class EvidenceAuthenticatedHTTPApp:
    """在 FastMCP Streamable HTTP app 前执行 capability 认证的 ASGI 包装层。

    对 HTTP lifespan 事件直接交给 FastMCP；对每一条业务 HTTP message 都重验
    header。认证成功只在当前异步上下文设置一个布尔标记，绝不把 token 或 registry
    路径注入 MCP context、prompt、trace 或错误响应。
    """

    def __init__(self, app: Any, *, config: EvidenceMcpServerConfig) -> None:
        if not config.capability_enabled:
            raise EvidenceMcpConfigurationError("Streamable HTTP requires capability")
        self.app = app
        self.config = config

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        """拒绝未授权 HTTP 请求，授权后逐条调用下游 FastMCP app。"""

        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        values = _read_scope_headers(scope, self.config.capability_header)
        token = values[0] if len(values) == 1 else None
        authorized = _authorize_capability(self.config, token)
        if not authorized:
            await JSONResponse(
                {"status": "unauthorized", "error": "invalid capability"},
                status_code=401,
            )(scope, receive, send)
            return
        request_token: Token[bool] = _AUTHORIZED_REQUEST.set(True)
        try:
            await self.app(scope, receive, send)
        finally:
            _AUTHORIZED_REQUEST.reset(request_token)


class _EvidenceFastMCP(FastMCP):
    """把 evidence capability wrapper 固定到该服务实例的 HTTP 入口。"""

    def streamable_http_app(self) -> Any:
        """返回每条 message 都执行 capability 校验的 Streamable HTTP app。"""

        application = super().streamable_http_app()
        config = getattr(self, "_arche_evidence_config", None)
        if not isinstance(config, EvidenceMcpServerConfig):
            raise EvidenceMcpConfigurationError("evidence server configuration is missing")
        return EvidenceAuthenticatedHTTPApp(application, config=config)


def _snapshot_task_binding(
    store: ReviewEvidenceStore,
    snapshot_ids: Sequence[str],
    requested_task_id: str | None,
) -> tuple[tuple[str, ...], str]:
    """启动时逐个核对 snapshot 存在且属于同一 task，不向模型暴露 manifest。"""

    normalized = _normalize_snapshot_ids(snapshot_ids)
    task_ids: set[str] = set()
    for snapshot_id in normalized:
        try:
            manifest = store.read_public_snapshot_manifest(snapshot_id)
        except (EvidenceRequestError, EvidenceRegistryCorruption, OSError) as exc:
            raise EvidenceMcpConfigurationError("snapshot allowlist cannot be verified") from exc
        task_id = manifest.get("task_id") if isinstance(manifest, Mapping) else None
        if not isinstance(task_id, str) or not task_id.strip():
            raise EvidenceMcpConfigurationError("snapshot task binding is invalid")
        task_ids.add(task_id)
    if len(task_ids) != 1:
        raise EvidenceMcpConfigurationError("snapshot allowlist crosses task boundaries")
    task_id = next(iter(task_ids))
    if requested_task_id is not None and requested_task_id != task_id:
        raise EvidenceMcpConfigurationError("snapshot task binding is invalid")
    return normalized, task_id


def _select_server_snapshot_binding(
    snapshot_id: str | None,
    snapshot_allowlist: Sequence[str] | None,
    allowed_snapshot_ids: Sequence[str] | None,
) -> tuple[Sequence[str], bool]:
    """选择唯一 snapshot 来源并标记是否使用多快照 allowlist。"""

    provided = [value is not None for value in (snapshot_allowlist, allowed_snapshot_ids)]
    if sum(provided) > 1:
        raise EvidenceMcpConfigurationError("snapshot allowlist is specified more than once")
    if snapshot_id is not None and (snapshot_allowlist is not None or allowed_snapshot_ids is not None):
        raise EvidenceMcpConfigurationError("snapshot binding is ambiguous")
    if snapshot_id is not None:
        return (snapshot_id,), False
    if snapshot_allowlist is not None:
        return snapshot_allowlist, True
    if allowed_snapshot_ids is not None:
        return allowed_snapshot_ids, True
    raise EvidenceMcpConfigurationError("snapshot binding is required")


def _build_server_capability_digest(capability_token: str | None) -> str | None:
    """校验可选 capability 并只返回其摘要。"""

    if capability_token is None:
        return None
    return _capability_digest(_validate_capability(capability_token))


def _validate_server_capability_requirements(
    digest: str | None,
    *,
    allowlist_mode: bool,
    require_capability: bool,
) -> None:
    """校验 allowlist 和 HTTP 强制认证的 capability 要求。"""

    if digest is None and allowlist_mode:
        raise EvidenceMcpConfigurationError("snapshot allowlist requires capability")
    if digest is None and require_capability:
        raise EvidenceMcpConfigurationError("Streamable HTTP requires capability")


def _build_server_config(
    store: ReviewEvidenceStore,
    *,
    snapshot_id: str | None,
    snapshot_allowlist: Sequence[str] | None,
    allowed_snapshot_ids: Sequence[str] | None,
    task_id: str | None,
    capability_token: str | None,
    capability_header: str,
    require_capability: bool,
) -> EvidenceMcpServerConfig:
    """构造并冻结服务访问边界，避免运行中修改 allowlist 或 capability。"""

    if not isinstance(require_capability, bool):
        raise EvidenceMcpConfigurationError("capability requirement is invalid")
    raw_ids, allowlist_mode = _select_server_snapshot_binding(
        snapshot_id,
        snapshot_allowlist,
        allowed_snapshot_ids,
    )
    normalized, bound_task_id = _snapshot_task_binding(store, raw_ids, task_id)
    digest = _build_server_capability_digest(capability_token)
    _validate_server_capability_requirements(
        digest,
        allowlist_mode=allowlist_mode,
        require_capability=require_capability,
    )
    return EvidenceMcpServerConfig(
        normalized,
        bound_task_id,
        capability_header=capability_header,
        capability_digest=digest,
        capability_required=allowlist_mode or require_capability,
    )


def _safe_tool_error(exc: BaseException) -> dict[str, Any]:
    """将底层异常压缩为无路径、无栈的稳定 MCP 错误 object。"""

    if isinstance(exc, _EvidenceMcpRequestError):
        code = exc.code
        message = exc.public_message
    elif isinstance(exc, EvidenceBudgetExceeded):
        code, message = "budget_exceeded", "evidence read budget exceeded"
    elif isinstance(exc, EvidenceArtifactIntegrityError):
        code, message = "integrity_failure", "evidence could not be verified"
    elif isinstance(exc, (EvidenceArtifactNotFound, EvidenceSnapshotNotFound)):
        code, message = "not_found", "evidence is unavailable"
    elif isinstance(exc, (EvidenceRequestError, UnsupportedEvidenceArtifact)):
        code, message = "invalid_request", "evidence request is invalid"
    elif isinstance(exc, EvidenceRegistryCorruption):
        code, message = "service_unavailable", "evidence service is unavailable"
    else:
        code, message = "service_unavailable", "evidence service is unavailable"
    return {"error": {"code": code, "message": message}}


def create_evidence_mcp_server(
    store: ReviewEvidenceStore,
    *,
    snapshot_id: str | None = None,
    snapshot_allowlist: Sequence[str] | None = None,
    allowed_snapshot_ids: Sequence[str] | None = None,
    task_id: str | None = None,
    capability_token: str | None = None,
    capability_header: str = DEFAULT_EVIDENCE_CAPABILITY_HEADER,
    require_capability: bool = False,
) -> FastMCP:
    """创建只注册一个 read-only evidence 工具的 FastMCP 服务。

    输入是已准备好的 ``ReviewEvidenceStore``、单个 task-level snapshot，或同一
    task 的 snapshot allowlist；allowlist 模式必须额外提供 capability token。stdio
    可在单一固定 snapshot 下使用进程管道；直接调用本地注册的 ``tool.fn`` 不是远程认证
    入口，仅可由可信单元测试直接调用。Streamable HTTP 调用方必须传入
    ``require_capability=True``，并提供 capability token。
    动作在启动时核对快照和任务归属，之后只注册 ``read_snapshot_artifact``，不
    注册路径登记、快照创建、目录枚举、资源或健康检查接口。返回 FastMCP 实例；
    配置非法时抛出不含 registry 路径的异常。
    """

    if not isinstance(store, ReviewEvidenceStore):
        raise TypeError("store must be ReviewEvidenceStore")
    config = _build_server_config(
        store,
        snapshot_id=snapshot_id,
        snapshot_allowlist=snapshot_allowlist,
        allowed_snapshot_ids=allowed_snapshot_ids,
        task_id=task_id,
        capability_token=capability_token,
        capability_header=capability_header,
        require_capability=require_capability,
    )
    server = _EvidenceFastMCP(
        "ARCHE Review Evidence",
        instructions=(
            "Read-only task-scoped evidence reader. Only opaque snapshot/artifact IDs "
            "are accepted; source paths, registration, snapshot creation, filesystem "
            "enumeration, and computation are unavailable."
        ),
        json_response=True,
        stateless_http=False,
    )
    annotations = ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )

    @server.tool(
        name="read_snapshot_artifact",
        description=(
            "Read a bounded UTF-8 text page from an authorized task snapshot by opaque "
            "snapshot_id and artifact_id. Paths and registration are not accepted."
        ),
        annotations=annotations,
        structured_output=True,
    )
    def read_snapshot_artifact(
        snapshot_id: str,
        artifact_id: str,
        offset: int = 0,
        max_chars: int = 8192,
        json_pointer: str | None = None,
    ) -> dict[str, Any]:
        """读取已绑定 task 快照中的一页文本，失败只返回安全错误 object。

        输入只能是 opaque ID、非负 offset、受限 max_chars 和可选 JSON Pointer；
        动作先检查当前服务的快照绑定，再调用 ReviewEvidenceStore 的哈希复核与
        进程预算。返回值是 path-free 读取结果或不含异常栈的稳定错误，不修改来源。
        """

        try:
            normalized_snapshot_id = _validate_opaque_id(snapshot_id, "snapshot_id")
            normalized_artifact_id = _validate_opaque_id(artifact_id, "artifact_id")
            if normalized_snapshot_id not in config.snapshot_ids:
                raise _EvidenceMcpRequestError("unauthorized", "evidence snapshot is not authorized")
            if config.capability_required and not _AUTHORIZED_REQUEST.get():
                raise _EvidenceMcpRequestError("unauthorized", "evidence request is unauthorized")
            # 单一绑定快照允许直接调用；capability allowlist 模式只允许经过已
            # 认证 HTTP message 的调用，测试或本地调用不能绕过 header 边界。
            result = store.read_snapshot_artifact(
                normalized_snapshot_id,
                normalized_artifact_id,
                offset,
                max_chars,
                json_pointer,
            )
            return result.to_public_dict()
        except Exception as exc:
            # 不记录异常正文；FastMCP 将此 object 作为结构化 tool result 返回，
            # 避免默认异常处理把 registry 路径或 traceback 回显给模型。
            return _safe_tool_error(exc)

    # 仅供 create_evidence_http_app 读取；不属于 MCP tool/resource surface。
    server._arche_evidence_config = config  # type: ignore[attr-defined]
    return server


def create_evidence_http_app(server: FastMCP) -> EvidenceAuthenticatedHTTPApp:
    """为 evidence FastMCP 创建带 capability ASGI wrapper 的 Streamable HTTP app。

    输入是 ``create_evidence_mcp_server`` 返回的服务；动作只包装 FastMCP 的
    ``/mcp`` Streamable HTTP 路由，不增加健康、资源、目录或其它工具路由。返回
    值可交给 uvicorn；服务配置缺失时抛出安全配置异常。
    """

    if not isinstance(server, FastMCP):
        raise TypeError("server must be FastMCP")
    config = getattr(server, "_arche_evidence_config", None)
    if not isinstance(config, EvidenceMcpServerConfig):
        raise EvidenceMcpConfigurationError("evidence server configuration is missing")
    if not config.capability_enabled:
        raise EvidenceMcpConfigurationError("Streamable HTTP requires capability")
    application = server.streamable_http_app()
    if isinstance(application, EvidenceAuthenticatedHTTPApp):
        return application
    return EvidenceAuthenticatedHTTPApp(application, config=config)


def _build_argument_parser() -> argparse.ArgumentParser:
    """构造不接受 capability 明文命令行参数的启动解析器。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", default=os.environ.get("ARCHE_EVIDENCE_STORE_ROOT"))
    parser.add_argument(
        "--snapshot-id",
        action="append",
        default=(
            [os.environ["ARCHE_EVIDENCE_SNAPSHOT_ID"]]
            if os.environ.get("ARCHE_EVIDENCE_SNAPSHOT_ID")
            else []
        ),
        help="绑定单个 snapshot；重复传入时形成同一 task 的 allowlist。",
    )
    parser.add_argument(
        "--snapshot-allowlist-file",
        default=os.environ.get("ARCHE_EVIDENCE_SNAPSHOT_ALLOWLIST_FILE"),
    )
    parser.add_argument("--task-id", default=None)
    parser.add_argument(
        "--capability-file",
        default=os.environ.get("ARCHE_EVIDENCE_CAPABILITY_FILE"),
        help="包含 capability 的本地 0600 文件；不会把 token 放进命令行。",
    )
    parser.add_argument("--capability-header", default=DEFAULT_EVIDENCE_CAPABILITY_HEADER)
    parser.add_argument(
        "--transport",
        choices=("streamable_http", "stdio"),
        default="streamable_http",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9020)
    return parser


def _read_cli_snapshot_ids(args: argparse.Namespace) -> tuple[str, ...]:
    """解析显式 ID 或私有 allowlist 文件，不扫描 registry 内容。"""

    ids: list[str] = list(args.snapshot_id or [])
    if args.snapshot_allowlist_file:
        if ids:
            raise EvidenceMcpConfigurationError("snapshot binding is specified more than once")
        ids.extend(load_snapshot_allowlist_file(args.snapshot_allowlist_file))
    return _normalize_snapshot_ids(ids)


def main(argv: Sequence[str] | None = None) -> int:
    """启动 evidence MCP 的 stdio 或 Streamable HTTP 入口。

    输入是显式 store root、snapshot binding 和可选 0600 capability 文件；动作只
    启动已冻结的服务，不生成快照、不登记路径、不枚举宿主目录。失败时输出不含
    registry 路径、token 或 traceback 的一行错误到 stderr，并返回非零状态。
    """

    parser = _build_argument_parser()
    args = parser.parse_args(argv)
    try:
        if not args.store_root:
            raise EvidenceMcpConfigurationError("store root is required")
        snapshot_ids = _read_cli_snapshot_ids(args)
        capability_token = (
            load_evidence_capability_file(args.capability_file)
            if args.capability_file
            else None
        )
        if args.transport == "stdio" and capability_token is not None:
            raise EvidenceMcpConfigurationError("stdio cannot use HTTP capability")
        require_capability = args.transport == "streamable_http"
        if len(snapshot_ids) == 1 and not args.snapshot_allowlist_file:
            server = create_evidence_mcp_server(
                ReviewEvidenceStore(args.store_root),
                snapshot_id=snapshot_ids[0],
                task_id=args.task_id,
                capability_token=capability_token,
                capability_header=args.capability_header,
                require_capability=require_capability,
            )
        else:
            server = create_evidence_mcp_server(
                ReviewEvidenceStore(args.store_root),
                snapshot_allowlist=snapshot_ids,
                task_id=args.task_id,
                capability_token=capability_token,
                capability_header=args.capability_header,
                require_capability=require_capability,
            )
        if args.transport == "stdio":
            server.run(transport="stdio")
            return 0
        import uvicorn

        uvicorn.run(create_evidence_http_app(server), host=args.host, port=args.port)
        return 0
    except Exception as exc:
        print(f"evidence MCP startup failed: {type(exc).__name__}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover - exercised by startup subprocess
    raise SystemExit(main())
