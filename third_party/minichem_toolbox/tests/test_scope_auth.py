"""MiniChem capability scope、并发工作区和共享调度器回归测试。"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import os
import secrets
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from mcp.server.fastmcp import FastMCP
from starlette.testclient import TestClient

from minichem_mcp_tools import discovery_tools, open_execution
from minichem_mcp_tools.execution_models import JobStatusRequest
from minichem_mcp_tools.scope_auth import ScopeAuthenticatedHTTPApp
from minichem_mcp_tools.tracing import execute_traced
from minichem_toolbox.artifacts import ArtifactStore
from minichem_toolbox import runtime
import minichem_toolbox.catalog as catalog
from minichem_toolbox.resource_budget import (
    active_resource_jobs,
    active_resource_usage,
    enqueue_gaussian_job,
    read_gaussian_queue_snapshot,
)
from minichem_toolbox.scope import (
    ScopeCapabilityConflict,
    ScopeCapabilityError,
    ScopeCapabilityRegistry,
    activate_scope,
    current_scope,
)


def _with_scheduler(capability, scheduler: Path):
    return replace(capability.scope, scheduler_root=scheduler)


def test_scope_provisioning_hashes_token_and_fails_closed_on_conflicts(tmp_path: Path):
    """注册表只写 token 哈希，重复身份和重复工作区均拒绝。"""

    registry = ScopeCapabilityRegistry(tmp_path / "registry")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provisioned = registry.provision_scope_capability(
        task_id="task-1",
        batch_id="batch-1",
        combination_id="combo-1",
        scope_id="scope-1",
        workspace=workspace,
    )
    record_text = provisioned.record_path.read_text(encoding="utf-8")
    assert provisioned.token not in record_text
    assert "token_sha256" in record_text
    assert provisioned.record_path.stat().st_mode & 0o777 == 0o600
    assert provisioned.record_path.parent.stat().st_mode & 0o777 == 0o700
    restored_registry = ScopeCapabilityRegistry(registry.root)
    assert restored_registry.authenticate_scope_capability(provisioned.token).scope_id == "scope-1"

    explicit_workspace = tmp_path / "explicit-workspace"
    explicit_workspace.mkdir()
    explicit_token = secrets.token_urlsafe(32)
    explicit = restored_registry.provision_scope_capability(
        task_id="task-explicit",
        batch_id="batch-explicit",
        combination_id="combo-explicit",
        scope_id="scope-explicit",
        workspace=explicit_workspace,
        token=explicit_token,
    )
    replay = ScopeCapabilityRegistry(registry.root).provision_scope_capability(
        task_id="task-explicit",
        batch_id="batch-explicit",
        combination_id="combo-explicit",
        scope_id="scope-explicit",
        workspace=explicit_workspace,
        token=explicit_token,
    )
    assert replay.record_path == explicit.record_path
    assert replay.scope.scope_id == "scope-explicit"
    with pytest.raises(ScopeCapabilityConflict):
        restored_registry.provision_scope_capability(
            task_id="task-explicit",
            batch_id="batch-explicit",
            combination_id="combo-explicit",
            scope_id="scope-explicit",
            workspace=explicit_workspace,
            token=secrets.token_urlsafe(32),
        )
    with pytest.raises(ScopeCapabilityConflict):
        registry.provision_scope_capability(
            task_id="task-1",
            batch_id="batch-1",
            combination_id="combo-1",
            scope_id="scope-1",
            workspace=workspace,
        )
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ScopeCapabilityConflict):
        registry.provision_scope_capability(
            task_id="task-2",
            batch_id="batch-2",
            combination_id="combo-2",
            scope_id="scope-2",
            workspace=workspace,
        )
    assert registry.authenticate_scope_capability(provisioned.token).scope_id == "scope-1"
    registry.revoke_scope_capability(provisioned.token)
    with pytest.raises(ScopeCapabilityError):
        registry.authenticate_scope_capability(provisioned.token)
    with pytest.raises(ScopeCapabilityConflict):
        registry.provision_scope_capability(
            task_id="task-1",
            batch_id="batch-1",
            combination_id="combo-1",
            scope_id="scope-1",
            workspace=workspace,
        )
    revoked_record = json.loads(provisioned.record_path.read_text(encoding="utf-8"))
    assert revoked_record["revoked"] is True
    assert provisioned.token not in json.dumps(revoked_record)


def test_scope_registry_rejects_unknown_and_malformed_active_records(tmp_path: Path):
    """注册表中的临时条目、未知字段和错误 active 类型都必须阻断管理操作。"""

    registry = ScopeCapabilityRegistry(tmp_path / "registry")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provisioned = registry.provision_scope_capability(
        task_id="task-1",
        batch_id="batch-1",
        combination_id="combo-1",
        scope_id="scope-1",
        workspace=workspace,
        token="A" * 48,
    )
    (registry.root / "capability.tmp").write_text("partial", encoding="utf-8")
    other_workspace = tmp_path / "other"
    other_workspace.mkdir()
    with pytest.raises(ScopeCapabilityError):
        registry.list_scope_records()
    with pytest.raises(ScopeCapabilityError):
        registry.provision_scope_capability(
            task_id="task-2",
            batch_id="batch-2",
            combination_id="combo-2",
            scope_id="scope-2",
            workspace=other_workspace,
            token="B" * 48,
        )

    (registry.root / "capability.tmp").unlink()
    record = json.loads(provisioned.record_path.read_text(encoding="utf-8"))
    record["revoked"] = "false"
    provisioned.record_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ScopeCapabilityError):
        registry.authenticate_scope_capability(provisioned.token)


def test_scope_registry_schema_v2_requires_explicit_correction_event_ref_key(
    tmp_path: Path,
):
    """MiniChem v2 记录缺少更正引用键时必须按篡改记录拒绝读取。"""

    registry = ScopeCapabilityRegistry(tmp_path / "registry")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provisioned = registry.provision_scope_capability(
        task_id="task-1",
        batch_id="batch-1",
        combination_id="combo-1",
        scope_id="scope-1",
        workspace=workspace,
        token="C" * 48,
    )
    record = json.loads(provisioned.record_path.read_text(encoding="utf-8"))
    del record["correction_event_ref"]
    provisioned.record_path.write_text(json.dumps(record), encoding="utf-8")
    os.chmod(provisioned.record_path, 0o600)

    with pytest.raises(ScopeCapabilityError):
        registry.list_scope_records()


def test_scope_http_rejects_missing_or_wrong_capability_and_is_concurrent(tmp_path: Path):
    """HTTP 请求认证失败时拒绝，两个并发 header 选择互不串扰。"""

    registry = ScopeCapabilityRegistry(tmp_path / "registry")
    scheduler = tmp_path / "scheduler"
    workspaces = []
    capabilities = []
    for index in (1, 2):
        workspace = tmp_path / f"workspace-{index}"
        workspace.mkdir(mode=0o700)
        workspaces.append(workspace)
        capabilities.append(
            registry.provision_scope_capability(
                task_id=f"task-{index}",
                batch_id=f"batch-{index}",
                combination_id=f"combo-{index}",
                scope_id=f"scope-{index}",
                workspace=workspace,
            )
        )

    async def echo(scope, receive, send):
        current = __import__("minichem_toolbox.scope", fromlist=["current_scope"]).current_scope()
        payload = json.dumps({"scope_id": current.scope_id, "workspace": str(current.workspace)}).encode()
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": payload})

    app = ScopeAuthenticatedHTTPApp(echo, registry=registry, scheduler_root=scheduler)

    async def run_requests():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://minichem") as client:
            missing = await client.get("/mcp")
            wrong = await client.get("/mcp", headers={"X-MiniChem-Capability": "wrong-token"})
            assert missing.status_code == 401
            assert wrong.status_code == 401

            responses = await asyncio.gather(
                client.get("/mcp", headers={"X-MiniChem-Capability": capabilities[0].token}),
                client.get("/mcp", headers={"X-MiniChem-Capability": capabilities[1].token}),
            )
            return [response.json() for response in responses]

    values = asyncio.run(run_requests())
    assert {value["scope_id"] for value in values} == {"scope-1", "scope-2"}
    assert {value["workspace"] for value in values} == {str(item) for item in workspaces}


def test_shared_scope_workspace_index_registration_is_idempotent_under_race(
    tmp_path: Path,
):
    """同一 scope 首次并发触发资源扫描时，索引最终必须只保存一条记录。"""

    registry = ScopeCapabilityRegistry(tmp_path / "registry")
    scheduler = tmp_path / "scheduler"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    capability = registry.provision_scope_capability(
        task_id="task-race",
        batch_id="batch-race",
        combination_id="combo-race",
        scope_id="scope-race",
        workspace=workspace,
    )
    scoped = _with_scheduler(capability, scheduler)

    def scan() -> dict[str, int]:
        with activate_scope(scoped):
            return active_resource_usage()

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _index: scan(), range(16)))
    assert results == [
        {"cpu_cores": 0, "memory_mb": 0, "gpu_count": 0}
    ] * 16
    entries = list((scheduler / "scope_workspaces").glob("workspace_*.json"))
    assert len(entries) == 1
    assert entries[0].stat().st_mode & 0o777 == 0o600


def test_shared_scope_workspace_index_registers_two_scopes_without_partial_reads(
    tmp_path: Path,
):
    """两个 scope 同时首次扫描时，读取方只看到完整的两个索引记录。"""

    registry = ScopeCapabilityRegistry(tmp_path / "registry")
    scheduler = tmp_path / "scheduler"
    scopes = []
    for index in (1, 2):
        workspace = tmp_path / f"workspace-{index}"
        workspace.mkdir(mode=0o700)
        capability = registry.provision_scope_capability(
            task_id=f"task-index-{index}",
            batch_id=f"batch-index-{index}",
            combination_id=f"combo-index-{index}",
            scope_id=f"scope-index-{index}",
            workspace=workspace,
        )
        scopes.append(_with_scheduler(capability, scheduler))

    def scan(index: int) -> dict[str, int]:
        with activate_scope(scopes[index % 2]):
            return active_resource_usage()

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(scan, range(24)))
    assert results == [
        {"cpu_cores": 0, "memory_mb": 0, "gpu_count": 0}
    ] * 24
    entries = list((scheduler / "scope_workspaces").glob("workspace_*.json"))
    assert len(entries) == 2
    assert all(entry.stat().st_mode & 0o777 == 0o600 for entry in entries)


def test_scope_auth_uses_service_catalog_marker_not_scope_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """scope-auth 请求读取服务级冻结目录，不要求各 scope 复制 marker。"""

    registry = ScopeCapabilityRegistry(tmp_path / "registry")
    service_workspace = tmp_path / "service-workspace"
    scope_workspace = tmp_path / "scope-workspace"
    service_workspace.mkdir()
    scope_workspace.mkdir()
    capability = registry.provision_scope_capability(
        task_id="task-catalog",
        batch_id="batch-catalog",
        combination_id="combo-catalog",
        scope_id="scope-catalog",
        workspace=scope_workspace,
    )
    monkeypatch.setenv(catalog.CATALOG_WORKSPACE_ENV, str(service_workspace))
    monkeypatch.delenv(catalog.CATALOG_HASH_ENV, raising=False)
    with activate_scope(_with_scheduler(capability, tmp_path / "scheduler")):
        assert catalog._catalog_workspace_path() == service_workspace.resolve()
        assert not (scope_workspace / catalog.CATALOG_FILENAME).exists()


def test_shared_gaussian_queue_exposes_only_owned_job_id(tmp_path: Path):
    """两个 scope 共用 FIFO 顺序，但资源视图只暴露当前 scope 的真实 job_id。"""

    registry = ScopeCapabilityRegistry(tmp_path / "registry")
    scheduler = tmp_path / "scheduler"
    scopes = []
    for index in (1, 2):
        workspace = tmp_path / f"workspace-{index}"
        workspace.mkdir()
        capability = registry.provision_scope_capability(
            task_id=f"task-{index}",
            batch_id=f"batch-{index}",
            combination_id=f"combo-{index}",
            scope_id=f"scope-{index}",
            workspace=workspace,
        )
        scopes.append(_with_scheduler(capability, scheduler))

    with activate_scope(scopes[0]):
        first = enqueue_gaussian_job("owner-1", kind="native", job_id="job_11111111111111111111111111111111")
    with activate_scope(scopes[1]):
        second = enqueue_gaussian_job("owner-2", kind="native", job_id="job_22222222222222222222222222222222")

    with activate_scope(scopes[0]):
        snapshot = read_gaussian_queue_snapshot()
    assert snapshot["items"][0] == {
        "position": 1,
        "status": "queued",
        "owned": True,
        "job_id": "job_11111111111111111111111111111111",
    }
    assert snapshot["items"][1] == {"position": 2, "status": "queued", "owned": False}
    assert "owner-2" not in json.dumps(snapshot)
    assert "22222222222222222222222222222222" not in json.dumps(snapshot)
    with activate_scope(scopes[0]):
        first.release()
    with activate_scope(scopes[1]):
        second.release()


def test_shared_scheduler_counts_foreign_active_status_after_reservation_loss(
    tmp_path: Path,
):
    """reservation 失效后，另一 scope 的持久 active status 仍占用共享预算。"""

    registry = ScopeCapabilityRegistry(tmp_path / "registry")
    scheduler = tmp_path / "scheduler"
    scopes = []
    for index in (1, 2):
        workspace = tmp_path / f"workspace-{index}"
        workspace.mkdir()
        capability = registry.provision_scope_capability(
            task_id=f"task-{index}",
            batch_id=f"batch-{index}",
            combination_id=f"combo-{index}",
            scope_id=f"scope-{index}",
            workspace=workspace,
        )
        scopes.append(_with_scheduler(capability, scheduler))

    foreign = scopes[1]
    job_id = "job_" + "f" * 32
    job_directory = foreign.workspace / "outputs" / "execution_jobs" / job_id
    job_directory.mkdir(parents=True)
    (job_directory / "status.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "status": "running",
                "scope_id": foreign.scope_id,
                "scope_workspace": str(foreign.workspace),
                "resource_limits": {
                    "cpu_cores": 2,
                    "memory_mb": 512,
                    "gpu_count": 0,
                },
                "resource_allocation": {
                    "cpu_ids": sorted(os.sched_getaffinity(0))[:2],
                    "gpu_ids": [],
                },
                "metadata": {"software_id": "orca"},
            }
        ),
        encoding="utf-8",
    )

    # 先让共享 scheduler 记录 foreign workspace；这里没有 live reservation，模拟
    # supervisor/controller 重启后 reservation 已丢失但持久 status 仍为 active。
    with activate_scope(foreign):
        assert active_resource_usage() == {
            "cpu_cores": 2,
            "memory_mb": 512,
            "gpu_count": 0,
        }

    with activate_scope(scopes[0]):
        assert active_resource_usage() == {
            "cpu_cores": 2,
            "memory_mb": 512,
            "gpu_count": 0,
        }
        visible = active_resource_jobs()
    assert visible == [
        {
            "record_type": "execution_job",
            "status": "running",
            "owned": False,
        }
    ]


def test_artifacts_and_workspace_context_are_scope_local(tmp_path: Path):
    """ContextVar 选择独立 ArtifactStore，引用不会跨 scope 解析。"""

    registry = ScopeCapabilityRegistry(tmp_path / "registry")
    scopes = []
    for index in (1, 2):
        workspace = tmp_path / f"workspace-{index}"
        workspace.mkdir()
        capability = registry.provision_scope_capability(
            task_id=f"task-{index}",
            batch_id=f"batch-{index}",
            combination_id=f"combo-{index}",
            scope_id=f"scope-{index}",
            workspace=workspace,
        )
        scopes.append(capability.scope)

    with activate_scope(scopes[0]):
        source = scopes[0].workspace / "input.txt"
        source.write_text("one", encoding="utf-8")
        reference = ArtifactStore().register_file(
            source,
            semantic_type="InputFile",
            producer_action="test",
            producer_backend=None,
        )
    with activate_scope(scopes[1]):
        with pytest.raises(KeyError):
            ArtifactStore().load(reference)


def test_execution_status_receipt_is_scope_bound(tmp_path: Path):
    """get_execution_job 只读取同 scope 的 status、receipt 和控制文件。"""

    registry = ScopeCapabilityRegistry(tmp_path / "registry")
    scopes = []
    for index in (1, 2):
        workspace = tmp_path / f"workspace-{index}"
        workspace.mkdir(mode=0o700)
        capability = registry.provision_scope_capability(
            task_id=f"task-{index}",
            batch_id=f"batch-{index}",
            combination_id=f"combo-{index}",
            scope_id=f"scope-{index}",
            workspace=workspace,
        )
        scopes.append(capability.scope)
    owner = scopes[0]
    job_id = "job_" + "a" * 32
    directory = owner.workspace / "outputs" / "execution_jobs" / job_id
    directory.mkdir(parents=True)
    common = {
        "job_id": job_id,
        "job_type": "programmable_analysis",
        "scope_id": owner.scope_id,
        "scope_workspace": str(owner.workspace),
    }
    (directory / "status.json").write_text(
        json.dumps(
            {
                **common,
                "status": "running",
                "supervisor_pid": None,
                "job_directory": f"outputs/execution_jobs/{job_id}",
                "stdout_path": f"outputs/execution_jobs/{job_id}/stdout.log",
                "stderr_path": f"outputs/execution_jobs/{job_id}/stderr.log",
                "resource_limits": {"cpu_cores": 1, "memory_mb": 128, "gpu_count": 0},
                "resource_allocation": {},
                "metadata": {},
                "execution_mode": "local",
            }
        ),
        encoding="utf-8",
    )
    (directory / "stdout.log").touch()
    (directory / "stderr.log").touch()
    for name in ("request.json", "operation_receipt.json", "supervisor_spec.json"):
        (directory / name).write_text(json.dumps(common), encoding="utf-8")

    with activate_scope(owner):
        visible = open_execution.get_execution_job(JobStatusRequest(job_id=job_id))
    assert visible["status"] == "success"
    assert visible["job"]["job_id"] == job_id
    assert "scope_id" not in visible["job"]

    with activate_scope(scopes[1]):
        hidden = open_execution.get_execution_job(JobStatusRequest(job_id=job_id))
    assert hidden == {
        "status": "invalid_request",
        "valid": False,
        "error": {
            "code": "execution_job_not_accessible",
            "message": "Execution job is unavailable in the current capability scope",
        },
        "automatic_fallback": False,
    }


def test_typed_worker_subprocess_receives_scope_without_parent_env_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """typed worker 继承显式 scope 环境，父进程环境不会被临时改写。"""

    registry = ScopeCapabilityRegistry(tmp_path / "registry")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    capability = registry.provision_scope_capability(
        task_id="task-worker",
        batch_id="batch-worker",
        combination_id="combo-worker",
        scope_id="scope-worker",
        workspace=workspace,
    )
    worker_python = tmp_path / "worker-python"
    worker_python.touch()
    monkeypatch.setattr(runtime, "runtime_python", lambda _runtime: worker_python)
    monkeypatch.setattr(
        runtime,
        "runtime_spec",
        lambda _runtime: {
            "name": "test-runtime",
            "environment_variables": {},
            "command_variables": {},
            "use_project_model_cache": False,
        },
    )
    monkeypatch.setattr(runtime, "runtime_path", lambda _runtime: tmp_path)
    captured: dict[str, object] = {}

    class FakeProcess:
        returncode = 0

        def communicate(self, _payload: str, timeout: int):
            captured["timeout"] = timeout
            return '{"status": "success"}', ""

    def fake_popen(*_args, **kwargs):
        captured["environment"] = kwargs["env"]
        return FakeProcess()

    monkeypatch.setattr(runtime.subprocess, "Popen", fake_popen)
    before_workspace = runtime.os.environ.get("MINICHEM_MCP_WORKSPACE")
    with activate_scope(capability.scope):
        result = runtime.invoke_worker(
            runtime="test-runtime",
            payload={"request": {}},
            timeout_seconds=5,
        )
    assert result["status"] == "success"
    environment = captured["environment"]
    assert environment["MINICHEM_MCP_WORKSPACE"] == str(workspace)
    assert environment["MINICHEM_SCOPE_ID"] == "scope-worker"
    assert runtime.os.environ.get("MINICHEM_MCP_WORKSPACE") == before_workspace


def test_stateful_fastmcp_session_reauthenticates_each_http_message(tmp_path: Path):
    """同一 MCP session 切换 capability 时，工具按每条 HTTP message 选择 scope。"""

    registry = ScopeCapabilityRegistry(tmp_path / "registry")
    capabilities = []
    for index in (1, 2):
        workspace = tmp_path / f"workspace-{index}"
        workspace.mkdir()
        capabilities.append(
            registry.provision_scope_capability(
                task_id=f"task-{index}",
                batch_id=f"batch-{index}",
                combination_id=f"combo-{index}",
                scope_id=f"scope-{index}",
                workspace=workspace,
            )
        )
    server = FastMCP("scope-test", json_response=True)

    @server.tool()
    def read_scope() -> dict:
        return execute_traced(
            "read_scope",
            {},
            lambda: {"scope_id": current_scope().scope_id if current_scope() else None},
            capture_artifacts=False,
        )

    app = ScopeAuthenticatedHTTPApp(
        server.streamable_http_app(),
        registry=registry,
        scheduler_root=tmp_path / "scheduler",
    )
    # 构造另一个独立 app 不得覆盖上一个 app 的逐消息认证配置。
    other_workspace = tmp_path / "other-app-workspace"
    other_workspace.mkdir()
    other_registry = ScopeCapabilityRegistry(tmp_path / "other-app-registry")
    other_registry.provision_scope_capability(
        task_id="other-task",
        batch_id="other-batch",
        combination_id="other-combo",
        scope_id="other-scope",
        workspace=other_workspace,
    )
    other_server = FastMCP("other-scope-test", json_response=True)
    other_app = ScopeAuthenticatedHTTPApp(
        other_server.streamable_http_app(),
        registry=other_registry,
        scheduler_root=tmp_path / "other-scheduler",
    )
    assert other_app.registry is other_registry
    with TestClient(app, headers={"host": "localhost:8000"}) as client:
        common = {"Accept": "application/json", "Content-Type": "application/json"}
        initialize = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "scope-test", "version": "1"},
            },
        }
        initialized = client.post(
            "/mcp",
            json=initialize,
            headers={**common, "X-MiniChem-Capability": capabilities[0].token},
        )
        assert initialized.status_code == 200
        session_id = initialized.headers["mcp-session-id"]

        def call_tool(capability, request_id: int) -> dict:
            response = client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": "tools/call",
                    "params": {"name": "read_scope", "arguments": {}},
                },
                headers={
                    **common,
                    "X-MiniChem-Capability": capability.token,
                    "Mcp-Session-Id": session_id,
                    "Mcp-Protocol-Version": "2025-06-18",
                },
            )
            assert response.status_code == 200
            return json.loads(response.json()["result"]["content"][0]["text"])

        assert call_tool(capabilities[0], 2)["scope_id"] == "scope-1"
        assert call_tool(capabilities[1], 3)["scope_id"] == "scope-2"


def test_stateful_batch_handler_reauthenticates_each_http_message(
    tmp_path: Path,
    monkeypatch,
):
    """批量 Action 的原始 handler 也必须使用当前消息而非 initialize 的 scope。"""

    registry = ScopeCapabilityRegistry(tmp_path / "registry")
    capabilities = []
    for index in (1, 2):
        workspace = tmp_path / f"batch-workspace-{index}"
        workspace.mkdir()
        capabilities.append(
            registry.provision_scope_capability(
                task_id=f"batch-task-{index}",
                batch_id=f"batch-{index}",
                combination_id=f"batch-combo-{index}",
                scope_id=f"batch-scope-{index}",
                workspace=workspace,
            )
        )

    def read_authenticated_scope(_request):
        selected = current_scope()
        assert selected is not None
        return {"status": "success", "scope_id": selected.scope_id}

    monkeypatch.setattr(
        discovery_tools,
        "_submit_action_batch_in_scope",
        read_authenticated_scope,
    )
    server = FastMCP("batch-scope-test", json_response=True)
    server.tool(name="submit_action_batch")(discovery_tools.submit_action_batch)
    app = ScopeAuthenticatedHTTPApp(
        server.streamable_http_app(),
        registry=registry,
        scheduler_root=tmp_path / "scheduler",
    )

    with TestClient(app, headers={"host": "localhost:8000"}) as client:
        common = {"Accept": "application/json", "Content-Type": "application/json"}
        initialized = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "batch-scope-test", "version": "1"},
                },
            },
            headers={
                **common,
                "X-MiniChem-Capability": capabilities[0].token,
            },
        )
        assert initialized.status_code == 200
        session_id = initialized.headers["mcp-session-id"]
        called = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "submit_action_batch",
                    "arguments": {
                        "request": {
                            "action_id": "predict_spectrum",
                            "backend_id": "dummy_backend",
                            "items": [{"item_id": "item-1", "inputs": {}}],
                        }
                    },
                },
            },
            headers={
                **common,
                "X-MiniChem-Capability": capabilities[1].token,
                "Mcp-Session-Id": session_id,
                "Mcp-Protocol-Version": "2025-06-18",
            },
        )
        assert called.status_code == 200
        result = json.loads(called.json()["result"]["content"][0]["text"])
        assert result["scope_id"] == "batch-scope-2"
