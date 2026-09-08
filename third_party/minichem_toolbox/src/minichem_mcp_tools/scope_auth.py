"""MiniChem scope 的 Streamable HTTP capability 认证与请求上下文绑定。"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

from starlette.responses import JSONResponse

from minichem_toolbox.scope import (
    CAPABILITY_HEADER,
    SHARED_SCHEDULER_ROOT_ENV,
    ScopeCapabilityError,
    ScopeCapabilityRegistry,
    current_scope,
    reset_scope_context,
    scope_context_token,
)


_SCOPE_AUTH_STATE_KEY = "minichem.scope_auth_configuration"


def _scheduler_root(value: str | os.PathLike[str] | Path) -> Path:
    """解析显式共享 scheduler 根，并在服务启动前拒绝不存在或非目录路径。"""

    if value is None:
        raise ScopeCapabilityError("shared scheduler root is required")
    candidate = Path(value).expanduser()
    absolute = Path(os.path.abspath(os.fspath(candidate)))
    cursor = Path(absolute.anchor or os.curdir)
    for part in absolute.parts[1:] if absolute.anchor else absolute.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ScopeCapabilityError("shared scheduler root cannot contain a symlink")
    root = absolute
    if root.exists() and (root.is_symlink() or not root.is_dir()):
        raise ScopeCapabilityError("shared scheduler root must be a regular directory")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if root.is_symlink() or not root.is_dir():
        raise ScopeCapabilityError("shared scheduler root must be a regular directory")
    os.chmod(root, 0o700)
    return root


def resolve_scheduler_root(value: str | os.PathLike[str] | Path | None) -> Path:
    """从显式参数或环境变量解析 scheduler 根；scope auth 下不得隐式回退。"""

    selected = str(value).strip() if value is not None else ""
    selected = selected or os.environ.get(SHARED_SCHEDULER_ROOT_ENV, "").strip()
    if not selected:
        raise ScopeCapabilityError(
            "scope authentication requires an explicit shared scheduler root"
        )
    return _scheduler_root(selected)


def _header_value(scope: dict[str, Any]) -> str | None:
    matches: list[str] = []
    for name, value in scope.get("headers") or []:
        try:
            header_name = (
                name.encode("ascii") if isinstance(name, str) else bytes(name)
            )
            header_value = value.encode("ascii") if isinstance(value, str) else bytes(value)
        except (TypeError, UnicodeEncodeError, ValueError):
            continue
        if header_name.lower() == CAPABILITY_HEADER.encode("ascii"):
            try:
                decoded = header_value.decode("ascii").strip()
            except UnicodeDecodeError:
                return None
            if not decoded:
                return None
            matches.append(decoded)
    return matches[0] if len(matches) == 1 else None


def _request_header_value(request: Any) -> str | None:
    try:
        request_scope = request.scope
    except AttributeError:
        request_scope = None
    if isinstance(request_scope, dict) and "headers" in request_scope:
        return _header_value(request_scope)
    try:
        value = request.headers.get(CAPABILITY_HEADER)
    except (AttributeError, TypeError):
        return None
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or "," in value:
        return None
    return value


def _request_scope_authentication(
    request: Any,
) -> tuple[ScopeCapabilityRegistry, Path] | None:
    """读取外层 ASGI app 写入当前 HTTP message 的认证配置。"""

    try:
        state = request.scope.get("state")
    except (AttributeError, TypeError):
        return None
    if not isinstance(state, dict):
        return None
    configuration = state.get(_SCOPE_AUTH_STATE_KEY)
    if (
        not isinstance(configuration, tuple)
        or len(configuration) != 2
        or not isinstance(configuration[0], ScopeCapabilityRegistry)
        or not isinstance(configuration[1], Path)
    ):
        return None
    return configuration


@contextmanager
def authenticate_current_mcp_request():
    """按 MCP message 携带的原始 HTTP Request 重新绑定 scope。"""

    try:
        from mcp.server.lowlevel.server import request_ctx

        request = request_ctx.get().request
    except LookupError:
        # 直接调用、stdio legacy 或测试替身没有 low-level request context；HTTP
        # 请求本身仍由外层 ASGI wrapper 验证，不能因进程内安装了 HTTP 校验器而
        # 破坏这些 legacy 调用路径。
        yield None
        return
    except AttributeError as exc:
        raise ScopeCapabilityError("authenticated MCP request context is unavailable") from exc
    if request is None:
        # stdio 的 RequestContext 没有 Starlette Request，继续保持 legacy 行为。
        yield None
        return
    configuration = _request_scope_authentication(request)
    if configuration is None:
        # 未启用 scope-auth 的 legacy HTTP 服务没有配置也没有当前 scope。若外层
        # 已绑定 scope 却丢失逐消息配置，必须拒绝，不能静默沿用 initialize 上下文。
        if current_scope() is not None:
            raise ScopeCapabilityError(
                "authenticated MCP request configuration is unavailable"
            )
        yield None
        return
    registry, scheduler_root = configuration
    token = _request_header_value(request)
    if token is None:
        raise ScopeCapabilityError("invalid capability")
    authenticated = registry.authenticate_scope_capability(token)
    selected = replace(authenticated, scheduler_root=scheduler_root)
    context_token = scope_context_token(selected)
    try:
        yield selected
    finally:
        reset_scope_context(context_token)


class ScopeAuthenticatedHTTPApp:
    """在 FastMCP Streamable HTTP app 前验证 capability 并绑定请求 scope。"""

    def __init__(
        self,
        app: Any,
        *,
        registry: ScopeCapabilityRegistry,
        scheduler_root: str | os.PathLike[str] | Path,
    ) -> None:
        self.app = app
        self.registry = registry
        self.scheduler_root = _scheduler_root(scheduler_root)

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        """拒绝缺失/错误 token 的 HTTP 请求，成功请求只在 ContextVar 中选工作区。"""

        if scope.get("type") != "http":
            # uvicorn/Starlette 的 lifespan 不是 MCP 请求，必须交给 FastMCP
            # 初始化 session manager，不能要求它携带 capability header。
            await self.app(scope, receive, send)
            return
        token = _header_value(scope)
        if token is None:
            await JSONResponse(
                {"status": "unauthorized", "error": "invalid capability"},
                status_code=401,
            )(scope, receive, send)
            return
        try:
            authenticated = self.registry.authenticate_scope_capability(token)
            selected = replace(authenticated, scheduler_root=self.scheduler_root)
        except (ScopeCapabilityError, OSError, ValueError):
            await JSONResponse(
                {"status": "unauthorized", "error": "invalid capability"},
                status_code=401,
            )(scope, receive, send)
            return
        context_token = scope_context_token(selected)
        request_scope = dict(scope)
        request_state = dict(scope.get("state") or {})
        request_state[_SCOPE_AUTH_STATE_KEY] = (self.registry, self.scheduler_root)
        request_scope["state"] = request_state
        try:
            await self.app(request_scope, receive, send)
        finally:
            reset_scope_context(context_token)


def create_scope_authenticated_http_app(
    server: Any,
    *,
    registry: ScopeCapabilityRegistry,
    scheduler_root: str | os.PathLike[str] | Path,
) -> ScopeAuthenticatedHTTPApp:
    """为 FastMCP server 创建启用 capability scope 的 Streamable HTTP ASGI app。"""

    if not isinstance(registry, ScopeCapabilityRegistry):
        raise TypeError("registry must be ScopeCapabilityRegistry")
    raw_app_builder = getattr(server, "_streamable_http_app_without_scope_auth", None)
    base_app = (
        raw_app_builder() if callable(raw_app_builder) else server.streamable_http_app()
    )
    return ScopeAuthenticatedHTTPApp(
        base_app,
        registry=registry,
        scheduler_root=scheduler_root,
    )


__all__ = [
    "ScopeAuthenticatedHTTPApp",
    "create_scope_authenticated_http_app",
    "resolve_scheduler_root",
]
