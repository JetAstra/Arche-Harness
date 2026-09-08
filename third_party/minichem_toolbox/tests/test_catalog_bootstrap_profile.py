"""验证 MiniChem 初始目录冻结 profile 的最小只读边界。"""

from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest
from mcp.server.fastmcp import FastMCP

from minichem_mcp_tools.profiles import (
    CATALOG_BOOTSTRAP_RESOURCE_URIS,
    CATALOG_BOOTSTRAP_TOOL_NAMES,
    build_public_catalog_identity,
    normalize_loopback_host,
    register_catalog_bootstrap_resources,
    register_catalog_bootstrap_tools,
    resolve_server_profile,
    validate_catalog_bootstrap_profile_options,
    validate_investigation_profile_options,
)
from minichem_mcp_tools.server import create_server
from minichem_toolbox.catalog import CatalogFreezeError, catalog_snapshot
from minichem_toolbox.scope import ScopeCapabilityRegistry


def _walk_strings(value: object):
    """遍历资源公开内容中的字符串，供 token 泄露检查使用。"""

    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)


def test_catalog_bootstrap_registration_is_exact_read_only_surface() -> None:
    """profile 只应注册四个冻结目录工具和一个公开目录资源。"""

    server = FastMCP("catalog-bootstrap-profile-test")
    assert tuple(register_catalog_bootstrap_tools(server)) == CATALOG_BOOTSTRAP_TOOL_NAMES

    snapshot = catalog_snapshot(include_health=False, discovery_mode="progressive")
    assert tuple(register_catalog_bootstrap_resources(server, snapshot)) == (
        "minichem://catalog",
    )
    assert tuple(server._tool_manager._tools) == CATALOG_BOOTSTRAP_TOOL_NAMES
    assert tuple(
        str(resource.uri)
        for resource in server._resource_manager._resources.values()
    ) == CATALOG_BOOTSTRAP_RESOURCE_URIS
    forbidden = {
        "execute_action",
        "submit_action_batch",
        "submit_native_job",
        "get_execution_job",
        "collect_execution_job",
        "cancel_execution_job",
        "submit_analysis_program",
        "validate_analysis_program",
        "write_workspace_text",
        "declare_scientific_artifact",
    }
    assert not forbidden.intersection(server._tool_manager._tools)
    assert "minichem://software" not in server._resource_manager._resources
    assert "minichem://execution-policy" not in server._resource_manager._resources

    resource = server._resource_manager._resources["minichem://catalog"]
    public_payload = json.loads(resource.fn())
    assert all("token" not in text.casefold() for text in _walk_strings(public_payload))


def test_catalog_bootstrap_profile_resolution_is_explicit() -> None:
    """profile 解析应保留 bootstrap 身份，非法值不得回退到 full。"""

    assert resolve_server_profile("catalog_bootstrap") == "catalog_bootstrap"
    assert resolve_server_profile("catalog-bootstrap") == "catalog_bootstrap"
    assert resolve_server_profile("full") == "legacy"
    with pytest.raises(ValueError, match="Unsupported MiniChem server profile"):
        resolve_server_profile("catalog_bootstrap_typo")


@pytest.mark.parametrize(
    ("value", "expected"),
    [("127.0.0.1", "127.0.0.1"), ("localhost", "127.0.0.1"), ("::1", "::1")],
)
def test_catalog_bootstrap_accepts_and_normalizes_loopback_hosts(
    value: str,
    expected: str,
) -> None:
    """允许的 loopback 主机应规范化为安全监听地址。"""

    assert normalize_loopback_host(value) == expected


@pytest.mark.parametrize(
    "value",
    ["", "0.0.0.0", "::", "192.0.2.1", "example.com", "127.0.0.1:9011"],
)
def test_catalog_bootstrap_rejects_non_loopback_hosts(value: str) -> None:
    """通配地址、远程地址、名称和带端口主机都必须拒绝。"""

    with pytest.raises(ValueError):
        validate_catalog_bootstrap_profile_options(host=value)


def test_catalog_bootstrap_rejects_scope_auth_and_transport_mismatch() -> None:
    """bootstrap profile 不得套用 scope-auth、stdio 或 full discovery。"""

    with pytest.raises(ValueError, match="scope authentication"):
        validate_catalog_bootstrap_profile_options(scope_auth=True)
    with pytest.raises(ValueError, match="streamable_http"):
        validate_catalog_bootstrap_profile_options(transport="stdio")
    with pytest.raises(ValueError, match="progressive"):
        validate_catalog_bootstrap_profile_options(discovery_mode="full")


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.0.2.1", "example.com"])
def test_investigation_profile_rejects_non_loopback_hosts(host: str) -> None:
    """Investigation profile 也必须拒绝远程监听地址。"""

    with pytest.raises(ValueError, match="loopback"):
        validate_investigation_profile_options(
            host=host,
            transport="streamable_http",
            scope_auth=True,
        )


def test_investigation_profile_requires_http_scope_auth() -> None:
    """Investigation profile 不得降级到 stdio 或无 scope-auth。"""

    with pytest.raises(ValueError, match="streamable_http"):
        validate_investigation_profile_options(
            host="127.0.0.1",
            transport="stdio",
            scope_auth=True,
        )
    with pytest.raises(ValueError, match="scope authentication"):
        validate_investigation_profile_options(
            host="127.0.0.1",
            transport="streamable_http",
            scope_auth=False,
        )


def test_create_server_rejects_investigation_remote_host_before_construction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """直接构造 Investigation 服务时 host 错配必须先于构造失败。"""

    import minichem_mcp_tools.server as server_module

    monkeypatch.setattr(server_module, "_construct_server", lambda **_: object())
    with pytest.raises(ValueError, match="loopback"):
        create_server(
            profile="investigation",
            host="0.0.0.0",
            transport="streamable_http",
            scope_auth=True,
            scope_registry=tmp_path / "registry",
            scheduler_root=tmp_path / "scheduler",
        )


def test_investigation_scope_storage_stays_inside_private_runtime_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """direct profile 构造也要在 registry/scheduler 产生副作用前锁定 runtime 根。"""

    import minichem_mcp_tools.server as server_module

    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir(mode=0o700)
    monkeypatch.setenv("MINICHEM_PRIVATE_RUNTIME_ROOT", str(runtime_root))
    monkeypatch.setattr(server_module, "_construct_server", lambda **_: object())

    created = create_server(
        profile="investigation",
        host="127.0.0.1",
        transport="streamable_http",
        scope_auth=True,
        scope_registry=runtime_root / "registry",
        scheduler_root=runtime_root / "scheduler",
    )
    assert created is not None
    assert (runtime_root / "registry").stat().st_mode & 0o777 == 0o700
    assert (runtime_root / "scheduler").stat().st_mode & 0o777 == 0o700

    object_registry = ScopeCapabilityRegistry(runtime_root / "object-registry")
    create_server(
        profile="investigation",
        host="127.0.0.1",
        transport="streamable_http",
        scope_auth=True,
        scope_registry=object_registry,
        scheduler_root=runtime_root / "object-scheduler",
    )
    with pytest.raises(ValueError, match="runtime"):
        create_server(
            profile="investigation",
            host="127.0.0.1",
            transport="streamable_http",
            scope_auth=True,
            scope_registry=tmp_path / "outside-registry",
            scheduler_root=runtime_root / "rejected-scheduler",
        )


def test_catalog_bootstrap_resource_rejects_full_snapshot() -> None:
    """bootstrap 资源注册也不接受 full 目录，避免 direct helper 绕过 profile。"""

    server = FastMCP("catalog-bootstrap-full-test")
    snapshot = catalog_snapshot(include_health=False, discovery_mode="full")
    with pytest.raises(CatalogFreezeError, match="progressive"):
        register_catalog_bootstrap_resources(server, snapshot)


def test_catalog_bootstrap_startup_script_uses_separate_runtime_port_and_root() -> None:
    """专用启动脚本应固定独立端口，并把工作区限制在运行资料目录。"""

    repository_root = Path(__file__).resolve().parents[3]
    script = (
        repository_root
        / "scripts"
        / "services"
        / "start_catalog_bootstrap_mcp.sh"
    )
    source = script.read_text(encoding="utf-8")
    assert 'PORT="${MINICHEM_CATALOG_BOOTSTRAP_MCP_PORT:-9011}"' in source
    assert (
        'RUNTIME_ROOT="${ARCHE_RUN_STORAGE_ROOT:-'
        '${MINICHEM_PRIVATE_RUNTIME_ROOT:-$ROOT/local_data/runtime/case1_harness}}"'
        in source
    )
    assert 'WORKSPACE="${MINICHEM_CATALOG_BOOTSTRAP_WORKSPACE:-' in source
    assert "--profile catalog_bootstrap" in source
    assert "--transport streamable_http" in source
    assert '  "$@" \\\n' in source


def test_create_server_rejects_scope_auth_before_catalog_construction(tmp_path: Path) -> None:
    """直接构造 bootstrap server 时 scope-auth 错配必须 fail closed。"""

    with pytest.raises(ValueError, match="scope authentication"):
        create_server(
            profile="catalog_bootstrap",
            host="127.0.0.1",
            transport="streamable_http",
            scope_auth=True,
            scope_registry=tmp_path / "registry",
            scheduler_root=tmp_path / "scheduler",
        )


def test_create_server_builds_bootstrap_surface_from_one_frozen_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """服务构造分支应使用冻结快照，并保持普通 FastMCP 无 scope wrapper。"""

    import minichem_mcp_tools.server as server_module

    snapshot = catalog_snapshot(include_health=False, discovery_mode="progressive")
    monkeypatch.setattr(
        server_module,
        "freeze_catalog_snapshot",
        lambda **_kwargs: snapshot,
    )
    monkeypatch.setattr(
        server_module,
        "validate_active_catalog_snapshot",
        lambda value: value,
    )
    created = create_server(
        profile="catalog_bootstrap",
        host="localhost",
        transport="streamable_http",
    )

    assert tuple(created._tool_manager._tools) == CATALOG_BOOTSTRAP_TOOL_NAMES
    assert tuple(created._resource_manager._resources) == CATALOG_BOOTSTRAP_RESOURCE_URIS
    assert not hasattr(created, "_minichem_scope_auth")


def test_start_server_passes_profile_and_scope_options_after_freeze(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """启动入口在冻结后仍把 profile、loopback 和 scope 参数完整传给构造器。"""

    import minichem_mcp_tools.server as server_module

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    snapshot = catalog_snapshot(include_health=False, discovery_mode="progressive")
    monkeypatch.setattr(server_module, "workspace_root", lambda: workspace)
    monkeypatch.setattr(server_module, "freeze_catalog_snapshot", lambda **_: snapshot)
    monkeypatch.setattr(server_module, "_publish_catalog_marker", lambda *_: None)
    captured: list[dict[str, object]] = []
    sentinel = object()

    def fake_create_server(**kwargs: object) -> object:
        captured.append(kwargs)
        return sentinel

    monkeypatch.setattr(server_module, "create_server", fake_create_server)
    bootstrap = server_module._start_server(
        profile="catalog_bootstrap",
        host="localhost",
        transport="streamable_http",
    )
    registry = ScopeCapabilityRegistry(tmp_path / "registry")
    investigation = server_module._start_server(
        profile="investigation",
        scope_auth=True,
        scope_registry=registry,
        scheduler_root=tmp_path / "scheduler",
    )

    assert bootstrap is sentinel and investigation is sentinel
    assert captured[0]["profile"] == "catalog_bootstrap"
    assert captured[0]["host"] == "127.0.0.1"
    assert captured[0]["transport"] == "streamable_http"
    assert captured[0]["discovery_mode"] == "progressive"
    assert captured[1]["profile"] == "investigation"
    assert captured[1]["scope_auth"] is True
    assert captured[1]["scope_registry"] is registry
    assert captured[1]["scheduler_root"] == tmp_path / "scheduler"


class _FakeCommandServer:
    """CLI 分支测试用的无网络 FastMCP 替身。"""

    def __init__(self) -> None:
        self.run_calls: list[str] = []
        self.app_calls = 0

    def run(self, *, transport: str) -> None:
        self.run_calls.append(transport)

    def streamable_http_app(self) -> str:
        self.app_calls += 1
        return "direct-app"


def test_server_main_successfully_routes_stdio_http_and_scope_http(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """CLI 成功路径覆盖 legacy stdio、bootstrap HTTP 和两种 scope HTTP 包装。"""

    import minichem_mcp_tools.server as server_module
    import uvicorn

    uvicorn_calls: list[tuple[object, str, int]] = []
    monkeypatch.setattr(
        uvicorn,
        "run",
        lambda app, *, host, port: uvicorn_calls.append((app, host, port)),
    )
    monkeypatch.setattr(
        server_module,
        "create_scope_authenticated_http_app",
        lambda server, **_: "scope-app",
    )

    def run_main(argv: list[str]) -> _FakeCommandServer:
        fake = _FakeCommandServer()
        monkeypatch.setattr(server_module, "_start_server", lambda **_: fake)
        monkeypatch.setattr(sys, "argv", argv)
        server_module.__dict__.pop("mcp", None)
        assert server_module.main() == 0
        server_module.__dict__.pop("mcp", None)
        return fake

    legacy = run_main(["server"])
    assert legacy.run_calls == ["stdio"]

    bootstrap = run_main(
        [
            "server",
            "--profile",
            "catalog_bootstrap",
            "--transport",
            "streamable_http",
            "--host",
            "127.0.0.1",
            "--port",
            "9011",
        ]
    )
    assert bootstrap.app_calls == 1

    investigation = run_main(
        [
            "server",
            "--profile",
            "investigation",
            "--transport",
            "streamable_http",
            "--host",
            "127.0.0.1",
            "--port",
            "9012",
            "--scope-registry",
            str(tmp_path / "investigation-registry"),
            "--scheduler-root",
            str(tmp_path / "investigation-scheduler"),
        ]
    )
    assert investigation.app_calls == 1

    scope_legacy = run_main(
        [
            "server",
            "--transport",
            "streamable_http",
            "--scope-auth",
            "--scope-registry",
            str(tmp_path / "legacy-registry"),
            "--scheduler-root",
            str(tmp_path / "legacy-scheduler"),
        ]
    )
    assert scope_legacy.app_calls == 0
    assert len(uvicorn_calls) == 3
    assert all(host == "127.0.0.1" for _app, host, _port in uvicorn_calls)


def test_bootstrap_and_investigation_share_public_identity_from_one_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """两个 profile 从同一 progressive 快照构造时必须暴露同一 canonical 身份。"""

    import minichem_mcp_tools.server as server_module

    snapshot = catalog_snapshot(include_health=False, discovery_mode="progressive")
    monkeypatch.setattr(
        server_module,
        "validate_active_catalog_snapshot",
        lambda value: value,
    )
    bootstrap = create_server(
        profile="catalog_bootstrap",
        host="127.0.0.1",
        transport="streamable_http",
        catalog_snapshot=snapshot,
    )
    investigation = create_server(
        profile="investigation",
        scope_auth=True,
        scope_registry=ScopeCapabilityRegistry(tmp_path / "registry"),
        scheduler_root=tmp_path / "scheduler",
        catalog_snapshot=snapshot,
    )

    public_catalog = json.loads(
        bootstrap._resource_manager._resources["minichem://catalog"].fn()
    )
    expected = build_public_catalog_identity(snapshot)
    assert bootstrap._minichem_catalog_identity == expected
    assert investigation._minichem_catalog_identity == expected
    assert "minichem://catalog" not in investigation._resource_manager._resources
    assert public_catalog["catalog_hash"] == expected["catalog_hash"]
    assert public_catalog["catalog_hash"] == investigation._minichem_catalog_identity[
        "catalog_hash"
    ]
    assert set(expected) == {
        "catalog_hash",
        "catalog_projection_hash",
        "catalog_version",
        "discovery_mode",
    }
    assert str(tmp_path) not in json.dumps(expected, ensure_ascii=False)


@pytest.mark.parametrize("mutation", ["hash", "actions", "action_item"])
def test_catalog_profile_rejects_illegal_or_drifted_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    """profile 构造遇到 canonical 漂移或目录结构损坏时必须在注册前失败。"""

    import minichem_mcp_tools.server as server_module

    snapshot = catalog_snapshot(include_health=False, discovery_mode="progressive")
    broken = deepcopy(snapshot)
    if mutation == "hash":
        broken["catalog_hash"] = "0" * 64
    elif mutation == "actions":
        broken["actions"] = "not-a-list"
        payload = deepcopy(broken)
        payload.pop("catalog_hash")
        broken["catalog_hash"] = hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    else:
        broken["actions"] = ["not-an-action-object"]
        payload = deepcopy(broken)
        payload.pop("catalog_hash")
        broken["catalog_hash"] = hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    monkeypatch.setattr(
        server_module,
        "validate_active_catalog_snapshot",
        lambda value: value,
    )

    with pytest.raises(CatalogFreezeError):
        create_server(
            profile="catalog_bootstrap",
            host="127.0.0.1",
            transport="streamable_http",
            catalog_snapshot=broken,
        )


def test_catalog_profile_rejects_non_object_snapshot() -> None:
    """显式快照不是 JSON object 时不能进入模式解析或工具注册。"""

    with pytest.raises(CatalogFreezeError, match="JSON object"):
        create_server(
            profile="catalog_bootstrap",
            host="127.0.0.1",
            transport="streamable_http",
            catalog_snapshot=["not-a-snapshot"],
        )


@pytest.mark.parametrize(
    "argv",
    [
        ["server", "--profile", "catalog_bootstrap"],
        ["server", "--profile", "catalog_bootstrap", "--transport", "streamable_http", "--scope-auth"],
        ["server", "--profile", "catalog_bootstrap", "--transport", "streamable_http", "--host", "0.0.0.0"],
        ["server", "--profile", "catalog_bootstrap", "--transport", "streamable_http", "--discovery-mode", "full"],
        ["server", "--profile", "catalog_bootstrap", "--transport", "streamable_http", "--port", "9010"],
    ],
)
def test_catalog_bootstrap_cli_profile_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
) -> None:
    """CLI 错配必须在冻结目录或服务构造前退出。"""

    import minichem_mcp_tools.server as server

    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        server.main()


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--host", "0.0.0.0", "--port", "9012"],
        ["--host", "127.0.0.1", "--port", "9010"],
        ["--transport", "stdio", "--port", "9012"],
    ],
)
def test_investigation_cli_profile_mismatch_fails_closed_before_start(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    extra_args: list[str],
) -> None:
    """Investigation CLI 不能被远程地址、旧端口或 stdio 覆盖。"""

    import minichem_mcp_tools.server as server

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "server",
            "--profile",
            "investigation",
            "--scope-registry",
            str(tmp_path / "registry"),
            "--scheduler-root",
            str(tmp_path / "scheduler"),
            *extra_args,
        ],
    )
    with pytest.raises(SystemExit):
        server.main()
