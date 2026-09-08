"""提供 progressive/full 发现服务及 loopback catalog bootstrap 服务。"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading

from mcp.server.fastmcp import FastMCP

from minichem_toolbox.catalog import (
    CatalogFreezeError,
    freeze_catalog_snapshot,
    _publish_catalog_marker,
    resolve_tool_discovery_mode,
    toolbox_overview,
    validate_active_catalog_snapshot,
)

from .registry import load_tool_config, register_catalog_resources, register_public_tools
from .scope_auth import (
    ScopeAuthenticatedHTTPApp,
    create_scope_authenticated_http_app,
    resolve_scheduler_root,
)
from minichem_toolbox.scope import ScopeCapabilityRegistry, ScopeCapabilityError
from .profiles import (
    catalog_bootstrap_profile_instructions,
    register_catalog_bootstrap_resources,
    register_catalog_bootstrap_tools,
    build_public_catalog_identity,
    investigation_profile_instructions,
    validate_catalog_bootstrap_profile_options,
    validate_investigation_profile_options,
    register_investigation_tools,
    resolve_server_profile,
)
from .software_catalog import open_execution_prompt
from .workspace import ensure_private_runtime_directory, workspace_root


_MCP_START_LOCK = threading.RLock()


def _validate_private_scope_storage_paths(
    scope_registry: ScopeCapabilityRegistry | str | os.PathLike[str] | None,
    scheduler_root: str | os.PathLike[str] | None,
) -> tuple[
    ScopeCapabilityRegistry | str | os.PathLike[str] | None,
    str | os.PathLike[str] | None,
]:
    """在显式 runtime 根内校验 scope registry 与 scheduler 路径。

    输入是 scope 认证的注册表和调度器根；动作只在设置了
    ``MINICHEM_PRIVATE_RUNTIME_ROOT`` 时启用 runtime 约束，逐组件拒绝 symlink 并
    创建/收紧 0700 目录。返回值是已规范化的路径，错误不包含宿主路径或认证内容；
    未设置 runtime 根时保留 direct profile 测试和 legacy 调用的既有路径协议。
    """

    runtime_value = os.environ.get("MINICHEM_PRIVATE_RUNTIME_ROOT", "").strip()
    if not runtime_value:
        return scope_registry, scheduler_root
    runtime_root = ensure_private_runtime_directory(
        runtime_value,
        runtime_root=runtime_value,
    )

    normalized_registry = scope_registry
    if isinstance(scope_registry, ScopeCapabilityRegistry):
        ensure_private_runtime_directory(
            scope_registry.root,
            runtime_root=runtime_root,
        )
    elif scope_registry is not None:
        normalized_registry = ensure_private_runtime_directory(
            scope_registry,
            runtime_root=runtime_root,
        )

    selected_scheduler = scheduler_root
    if selected_scheduler is None:
        selected_scheduler = os.environ.get(
            "MINICHEM_SHARED_SCHEDULER_ROOT", ""
        ).strip() or None
    if selected_scheduler is not None:
        selected_scheduler = ensure_private_runtime_directory(
            selected_scheduler,
            runtime_root=runtime_root,
        )
    return normalized_registry, selected_scheduler


class ScopeAuthenticatedFastMCP(FastMCP):
    """为显式 scope-auth 服务固定 capability 认证的 FastMCP 实例。"""

    def __init__(
        self,
        *args,
        scope_registry: ScopeCapabilityRegistry,
        scheduler_root: str | os.PathLike[str],
        **kwargs,
    ) -> None:
        self._scope_registry = scope_registry
        self._scope_scheduler_root = resolve_scheduler_root(scheduler_root)
        super().__init__(*args, **kwargs)

    def _streamable_http_app_without_scope_auth(self):
        """构造未包装的 FastMCP app，仅供认证适配器内部调用。"""

        return super().streamable_http_app()

    def streamable_http_app(self):
        """返回始终经过 capability wrapper 的 Streamable HTTP app。"""

        return ScopeAuthenticatedHTTPApp(
            self._streamable_http_app_without_scope_auth(),
            registry=self._scope_registry,
            scheduler_root=self._scope_scheduler_root,
        )


def _select_profile_instructions(
    selected_profile: str,
    *,
    mode: str,
    snapshot: dict,
) -> str:
    """按 profile 选择公开说明文字，并只在 full 模式追加历史提示。"""

    if selected_profile == "investigation":
        return investigation_profile_instructions()
    if selected_profile == "catalog_bootstrap":
        return catalog_bootstrap_profile_instructions()
    instructions = toolbox_overview(
        discovery_mode=mode,
        include_health=False,
        snapshot=snapshot,
    )
    if mode == "full":
        instructions += open_execution_prompt(include_command_index=True)
    return instructions


def _resolve_scope_server_dependencies(
    *,
    scope_auth: bool,
    scope_registry: ScopeCapabilityRegistry | str | os.PathLike[str] | None,
    scheduler_root: str | os.PathLike[str] | None,
) -> tuple[ScopeCapabilityRegistry | None, str | None]:
    """解析 scope 服务要使用的注册表和 scheduler 根目录。"""

    if not scope_auth:
        return None, None
    registry = (
        scope_registry
        if isinstance(scope_registry, ScopeCapabilityRegistry)
        else ScopeCapabilityRegistry(str(scope_registry))
    )
    return registry, resolve_scheduler_root(scheduler_root)


def _create_profile_server_instance(
    *,
    selected_profile: str,
    config: dict,
    instructions: str,
    mode: str,
    snapshot: dict,
    scope_auth: bool,
    registry: ScopeCapabilityRegistry | None,
    selected_scheduler_root: str | None,
) -> FastMCP:
    """按 profile 创建服务实例并注册其允许的工具和资源。"""

    if selected_profile == "investigation" or scope_auth:
        server = ScopeAuthenticatedFastMCP(
            name=str(
                config.get(
                    "server_name",
                    "MiniChem Investigation Toolbox"
                    if selected_profile == "investigation"
                    else "MiniChem Three-Layer Chemistry Toolbox",
                )
            ),
            instructions=instructions,
            scope_registry=registry,
            scheduler_root=selected_scheduler_root,
        )
        if selected_profile == "investigation":
            register_investigation_tools(server)
        else:
            register_public_tools(server, mode)
            register_catalog_resources(server, mode, snapshot)
        return server
    if selected_profile == "catalog_bootstrap":
        server = FastMCP(
            name=str(
                config.get("server_name", "MiniChem Catalog Bootstrap Toolbox")
            ),
            instructions=instructions,
        )
        register_catalog_bootstrap_tools(server)
        register_catalog_bootstrap_resources(server, snapshot)
        return server
    server = FastMCP(
        name=str(config.get("server_name", "MiniChem Three-Layer Chemistry Toolbox")),
        instructions=instructions,
    )
    register_public_tools(server, mode)
    register_catalog_resources(server, mode, snapshot)
    return server


def _attach_server_metadata(
    server: FastMCP,
    *,
    catalog_identity: dict,
    scope_auth: bool,
    registry: ScopeCapabilityRegistry | None,
    selected_scheduler_root: str | None,
    selected_profile: str,
) -> FastMCP:
    """给服务附加脱敏目录摘要和 scope 配置，兼容不可扩展测试替身。"""

    # 只保存脱敏摘要；完整 snapshot 仍留在 catalog 的受控缓存/资源闭包中。
    try:
        server._minichem_catalog_identity = catalog_identity
    except AttributeError:
        # 测试替身可能是 dict 等不可扩展对象；真实 FastMCP 始终保留该摘要。
        pass
    if scope_auth:
        # FastMCP 对象保留显式配置，CLI/调用方通过
        # create_scope_authenticated_http_app 构造真正的受保护 ASGI app。
        try:
            server._minichem_scope_auth = {
                "registry": registry,
                "scheduler_root": str(selected_scheduler_root),
                "profile": selected_profile,
            }
        except AttributeError:
            # 测试替身可能是 dict 等不可扩展对象；验证已在入口完成，继续返回替身。
            pass
    return server


def create_server(
    profile: str | None = None,
    discovery_mode: str | None = None,
    catalog_snapshot: dict | None = None,
    *,
    host: str | None = None,
    transport: str | None = None,
    scope_auth: bool = False,
    scope_registry: ScopeCapabilityRegistry | str | os.PathLike[str] | None = None,
    scheduler_root: str | os.PathLike[str] | None = None,
) -> FastMCP:
    """创建使用同一冻结目录的 MiniChem 服务实例。

    输入是可选 profile、发现模式和已冻结目录快照；动作在启动锁内校验或创建快照，
    注册公开工具与资源，不修改调用方快照。返回值是已配置的 ``FastMCP`` 实例；
    配置、快照或目录校验失败时抛出启动异常，避免暴露旧目录。
    """

    selected_profile = resolve_server_profile(profile)
    if scope_auth or selected_profile == "investigation":
        scope_registry, scheduler_root = _validate_private_scope_storage_paths(
            scope_registry,
            scheduler_root,
        )
    normalized_host = None
    if selected_profile == "catalog_bootstrap":
        normalized_host = validate_catalog_bootstrap_profile_options(
            host=host,
            transport=transport,
            scope_auth=scope_auth,
            discovery_mode=discovery_mode,
        )
    if selected_profile == "investigation":
        if not scope_auth:
            raise ScopeCapabilityError(
                "investigation profile requires scope authentication"
            )
        normalized_host = validate_investigation_profile_options(
            host=host,
            transport=transport,
            scope_auth=scope_auth,
            discovery_mode=discovery_mode,
        )
    if scope_auth:
        if scope_registry is None:
            raise ScopeCapabilityError(
                "scope authentication requires a capability registry"
            )
        resolve_scheduler_root(scheduler_root)
    # 直接调用 create_server 也可能与 CLI/惰性入口并发；使用 RLock 允许
    # _start_server 在已持锁时复用这里的完整构造过程。
    with _MCP_START_LOCK:
        return _construct_server(
            profile=profile,
            discovery_mode=discovery_mode,
            catalog_snapshot=catalog_snapshot,
            host=normalized_host or host,
            transport=transport,
            scope_auth=scope_auth,
            scope_registry=scope_registry,
            scheduler_root=scheduler_root,
        )


def _construct_server(
    *,
    profile: str | None = None,
    discovery_mode: str | None = None,
    catalog_snapshot: dict | None = None,
    host: str | None = None,
    transport: str | None = None,
    scope_auth: bool = False,
    scope_registry: ScopeCapabilityRegistry | str | os.PathLike[str] | None = None,
    scheduler_root: str | os.PathLike[str] | None = None,
) -> FastMCP:
    """在启动锁已持有时构造服务。

    输入是可选 profile、发现模式和快照；动作加载配置，校验或冻结目录并注册工具、
    资源，不重复获取启动锁。返回值是服务实例；快照模式不一致、目录无效或配置
    失败时抛出启动异常，不会返回半初始化服务。
    """

    selected_profile = resolve_server_profile(profile)
    if scope_auth or selected_profile == "investigation":
        scope_registry, scheduler_root = _validate_private_scope_storage_paths(
            scope_registry,
            scheduler_root,
        )
    if selected_profile == "catalog_bootstrap":
        validate_catalog_bootstrap_profile_options(
            host=host,
            transport=transport,
            scope_auth=scope_auth,
            discovery_mode=discovery_mode,
        )
    if selected_profile == "investigation":
        if not scope_auth:
            raise ScopeCapabilityError(
                "investigation profile requires scope authentication"
            )
        validate_investigation_profile_options(
            host=host,
            transport=transport,
            scope_auth=scope_auth,
            discovery_mode=discovery_mode,
        )
    config = load_tool_config()
    effective_discovery_mode = (
        "progressive"
        if selected_profile in {"investigation", "catalog_bootstrap"}
        else discovery_mode
    )
    if catalog_snapshot is not None:
        if not isinstance(catalog_snapshot, dict):
            raise CatalogFreezeError(
                "显式 MiniChem 目录快照必须是 JSON object"
            )
        snapshot_mode = (
            catalog_snapshot.get("discovery_mode")
        )
        mode = resolve_tool_discovery_mode(
            effective_discovery_mode
            if effective_discovery_mode is not None
            else snapshot_mode
        )
        # 显式快照必须已经安装在当前 workspace；校验函数不改动现有 marker。
        snapshot = validate_active_catalog_snapshot(catalog_snapshot)
        active_mode = resolve_tool_discovery_mode(snapshot.get("discovery_mode"))
        if active_mode != mode:
            raise CatalogFreezeError("显式快照与请求的 discovery_mode 不一致")
    else:
        # 让 freeze 负责解析模式；非法模式也要留下 workspace 失败状态。
        snapshot = freeze_catalog_snapshot(discovery_mode=effective_discovery_mode)
        mode_value = (
            effective_discovery_mode
            if effective_discovery_mode is not None
            else snapshot.get("discovery_mode")
        )
        mode = resolve_tool_discovery_mode(mode_value)
    catalog_identity = build_public_catalog_identity(snapshot)
    instructions = _select_profile_instructions(
        selected_profile,
        mode=mode,
        snapshot=snapshot,
    )
    registry, selected_scheduler_root = _resolve_scope_server_dependencies(
        scope_auth=scope_auth,
        scope_registry=scope_registry,
        scheduler_root=scheduler_root,
    )
    server = _create_profile_server_instance(
        selected_profile=selected_profile,
        config=config,
        instructions=instructions,
        mode=mode,
        snapshot=snapshot,
        scope_auth=scope_auth,
        registry=registry,
        selected_scheduler_root=selected_scheduler_root,
    )
    return _attach_server_metadata(
        server,
        catalog_identity=catalog_identity,
        scope_auth=scope_auth,
        registry=registry,
        selected_scheduler_root=selected_scheduler_root,
        selected_profile=selected_profile,
    )


def _start_server(
    discovery_mode: str | None = None,
    *,
    profile: str | None = None,
    host: str | None = None,
    transport: str | None = None,
    scope_auth: bool = False,
    scope_registry: ScopeCapabilityRegistry | str | os.PathLike[str] | None = None,
    scheduler_root: str | os.PathLike[str] | None = None,
) -> FastMCP:
    """解析 workspace、冻结目录后创建服务。

    输入是可选发现模式；动作锁定 workspace，冻结目录并发布生命周期标记，再创建
    FastMCP 服务。返回值是可供 CLI 和惰性入口使用的服务实例；workspace、冻结或
    标记发布失败时抛出启动异常，不会回退到旧目录。
    """

    # profile 边界先于 workspace/chdir 校验，错配时不冻结目录也不改变进程目录。
    selected_profile = resolve_server_profile(profile)
    if scope_auth or selected_profile == "investigation":
        scope_registry, scheduler_root = _validate_private_scope_storage_paths(
            scope_registry,
            scheduler_root,
        )
    normalized_host = None
    if selected_profile == "catalog_bootstrap":
        normalized_host = validate_catalog_bootstrap_profile_options(
            host=host,
            transport=transport,
            scope_auth=scope_auth,
            discovery_mode=discovery_mode,
        )
    if selected_profile == "investigation":
        if not scope_auth:
            raise ScopeCapabilityError(
                "investigation profile requires scope authentication"
            )
        normalized_host = validate_investigation_profile_options(
            host=host,
            transport=transport,
            scope_auth=scope_auth,
            discovery_mode=discovery_mode,
        )
    if scope_auth:
        if scope_registry is None:
            raise ScopeCapabilityError(
                "scope authentication requires a capability registry"
            )
        resolve_scheduler_root(scheduler_root)

    # freeze、FastMCP 构造和 marker 发布必须共享同一把锁；否则第二个 workspace
    # 可能在第一个服务校验 active 快照后、发布 marker 前替换进程级状态。
    with _MCP_START_LOCK:
        root = workspace_root()
        os.environ.setdefault("MINICHEM_TOOL_LOG_DIR", str(root / "tool_logs"))
        os.chdir(root)
        effective_discovery_mode = (
            "progressive"
            if selected_profile in {"investigation", "catalog_bootstrap"}
            else discovery_mode
        )
        snapshot = freeze_catalog_snapshot(
            workspace=root, discovery_mode=effective_discovery_mode
        )
        mode_value = (
            effective_discovery_mode
            if effective_discovery_mode is not None
            else snapshot.get("discovery_mode")
        )
        mode = resolve_tool_discovery_mode(mode_value)
        create_kwargs: dict[str, object] = {
            "discovery_mode": mode,
            "catalog_snapshot": snapshot,
        }
        if profile is not None:
            create_kwargs["profile"] = selected_profile
        if selected_profile in {"investigation", "catalog_bootstrap"}:
            create_kwargs.update(
                {
                    "host": normalized_host,
                    "transport": transport,
                }
            )
        if scope_auth:
            create_kwargs.update(
                {
                    "scope_auth": True,
                    "scope_registry": scope_registry,
                    "scheduler_root": scheduler_root,
                }
            )
        server = create_server(**create_kwargs)
        _publish_catalog_marker(snapshot, root)
        return server


def __getattr__(name: str):
    """按需提供 FastMCP 的公开 ``mcp`` 入口。

    输入是模块属性名；动作仅对 ``mcp`` 执行一次带目录冻结的惰性启动并缓存服务。
    返回值是 FastMCP 实例；其他属性抛出 ``AttributeError``，启动失败传播原异常，
    不会返回未冻结目录的服务。
    """

    if name != "mcp":
        raise AttributeError(name)
    existing = globals().get("mcp")
    if existing is not None:
        return existing
    global mcp
    with _MCP_START_LOCK:
        existing = globals().get("mcp")
        if existing is None:
            existing = _start_server()
            mcp = existing
    return existing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable_http"],
        default="stdio",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9010)
    parser.add_argument(
        "--profile",
        default=None,
        choices=["full", "investigation", "catalog_bootstrap", "catalog-bootstrap"],
        help=(
            "MCP 工具 profile；investigation 使用 capability scope，"
            "catalog_bootstrap 仅提供 loopback 只读目录冻结能力。"
        ),
    )
    parser.add_argument(
        "--discovery-mode",
        choices=["progressive", "full"],
        default=None,
        help=(
            "MCP surface: progressive discovery (default) or the historical full "
            "one-tool-per-Action compatibility mode."
        ),
    )
    parser.add_argument(
        "--scope-auth",
        action="store_true",
        help="为 Streamable HTTP 启用 MiniChem capability scope 认证。",
    )
    parser.add_argument(
        "--scope-registry",
        default=None,
        help="scope capability 私有注册表目录（与 --scope-auth 一起使用）。",
    )
    parser.add_argument(
        "--scheduler-root",
        default=None,
        help="scope auth 必须显式指定的共享 Gaussian/resource scheduler 根目录。",
    )
    args = parser.parse_args()

    selected_profile = resolve_server_profile(args.profile)
    if selected_profile == "investigation":
        args.scope_auth = True
        try:
            args.host = validate_investigation_profile_options(
                host=args.host,
                transport=args.transport,
                scope_auth=args.scope_auth,
                discovery_mode=args.discovery_mode,
                port=args.port,
            )
        except ValueError as exc:
            parser.error(str(exc))
    if selected_profile == "catalog_bootstrap":
        try:
            args.host = validate_catalog_bootstrap_profile_options(
                host=args.host,
                transport=args.transport,
                scope_auth=args.scope_auth,
                discovery_mode=args.discovery_mode,
                port=args.port,
            )
        except ValueError as exc:
            parser.error(str(exc))
    if args.scope_auth and args.transport != "streamable_http":
        parser.error("--scope-auth 只适用于 streamable_http；stdio 保持 legacy 行为")
    if args.scope_auth and not args.scope_registry:
        parser.error("--scope-auth requires --scope-registry")

    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    start_kwargs: dict[str, object] = {
        "discovery_mode": args.discovery_mode,
        "profile": args.profile,
        "scope_auth": args.scope_auth,
        "scope_registry": args.scope_registry,
        "scheduler_root": args.scheduler_root,
    }
    if selected_profile in {"investigation", "catalog_bootstrap"}:
        start_kwargs.update(
            {
                "host": args.host,
                "transport": args.transport,
            }
        )
    server = _start_server(
        **start_kwargs,
    )
    registry = None
    if args.scope_auth:
        configured_scope = getattr(server, "_minichem_scope_auth", {})
        registry = (
            configured_scope.get("registry")
            if isinstance(configured_scope, dict)
            else None
        )
        if not isinstance(registry, ScopeCapabilityRegistry):
            # 测试替身没有真实服务属性；真实服务在 _start_server 内已经完成路径校验。
            registry = ScopeCapabilityRegistry(args.scope_registry)
    global mcp
    mcp = server
    if args.transport == "streamable_http":
        import uvicorn

        if args.scope_auth and selected_profile != "investigation":
            app = create_scope_authenticated_http_app(
                server,
                registry=registry,
                scheduler_root=args.scheduler_root,
            )
        else:
            app = server.streamable_http_app()
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        server.run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
