"""验证 Investigation MCP profile 的实例级工具清单和强制认证。"""

from __future__ import annotations

from pathlib import Path

import pytest
from mcp.server.fastmcp import FastMCP
from starlette.testclient import TestClient

from minichem_mcp_tools.profiles import (
    INVESTIGATION_TOOL_NAMES,
    register_investigation_tools,
)
from minichem_mcp_tools.open_tools import OPEN_EXECUTION_TOOL_NAMES
from minichem_mcp_tools.registry import register_public_tools
from minichem_mcp_tools.server import ScopeAuthenticatedFastMCP, create_server
from minichem_toolbox.catalog import action_specs, catalog_snapshot
from minichem_toolbox.discovery import search_actions
from minichem_toolbox.scope import ScopeCapabilityError, ScopeCapabilityRegistry


def _tool_names(server: FastMCP) -> tuple[str, ...]:
    return tuple(server._tool_manager._tools)


def test_investigation_registration_is_exact_explicit_allowlist() -> None:
    """新 profile 只能注册源码常量逐项列出的工具。"""

    server = FastMCP("investigation-profile-test")
    registered = register_investigation_tools(server)

    assert tuple(registered) == INVESTIGATION_TOOL_NAMES
    assert _tool_names(server) == INVESTIGATION_TOOL_NAMES
    assert "declare_scientific_artifact" in _tool_names(server)
    assert "submit_action_job" in _tool_names(server)
    assert "execute_action" not in _tool_names(server)
    assert "submit_action_batch" not in _tool_names(server)
    assert "run_shell" not in _tool_names(server)
    assert not set(action_specs()).intersection(_tool_names(server))


def test_investigation_profile_can_discover_gaussian_codegen_action() -> None:
    """Investigation 通过统一目录发现 route 组合 Action，而不是注册隐藏的专用工具。"""

    snapshot = catalog_snapshot(include_health=False, discovery_mode="progressive")
    result = search_actions(
        backend_id="gaussian_codegen",
        query="Gaussian route composition",
        retrieval_mode="lexical",
        available_only=False,
        snapshot=snapshot,
    )

    codegen = next(
        item for item in result["actions"] if item["action_id"] == "generate_gaussian_code"
    )
    description = str(codegen.get("description") or "")
    assert "DeepSeek Pro" in description
    assert "scientific evidence" in description
    assert "Gaussian input deck" in description


def test_full_registration_keeps_existing_tool_surface() -> None:
    """旧 full 注册仍包含逐 Action 工具和完整 open execution 工具。"""

    server = FastMCP("full-profile-test")
    registered = register_public_tools(server, "full")

    assert set(action_specs()).issubset(registered)
    assert set(OPEN_EXECUTION_TOOL_NAMES).issubset(registered)
    assert "verify_execution_references" not in registered
    assert "read_scientific_artifact_text" not in registered
    assert set(registered) == set(_tool_names(server))


def test_investigation_server_requires_scope_auth(tmp_path: Path, monkeypatch) -> None:
    """Investigation profile 没有 capability registry 时必须在启动前拒绝。"""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(workspace))

    with pytest.raises(ScopeCapabilityError):
        create_server(profile="investigation")


def test_investigation_server_streamable_http_app_is_authenticated(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """直接从 Investigation server 取得 HTTP app 也不能绕过 capability。"""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ScopeCapabilityRegistry(tmp_path / "registry")
    scheduler = tmp_path / "scheduler"
    capability_workspace = tmp_path / "scope-workspace"
    capability_workspace.mkdir()
    registry.provision_scope_capability(
        task_id="task-1",
        batch_id="batch-1",
        combination_id="combo-1",
        scope_id="scope-1",
        workspace=capability_workspace,
        token="A" * 48,
    )
    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(workspace))

    server = create_server(
        profile="investigation",
        scope_auth=True,
        scope_registry=registry,
        scheduler_root=scheduler,
    )

    assert isinstance(server, ScopeAuthenticatedFastMCP)
    assert _tool_names(server) == INVESTIGATION_TOOL_NAMES
    app = server.streamable_http_app()
    with TestClient(app) as client:
        response = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "1"},
                },
            },
            headers={"Accept": "application/json", "Content-Type": "application/json"},
        )

    assert response.status_code == 401
