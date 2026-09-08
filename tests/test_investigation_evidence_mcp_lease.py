"""任务级 Evidence MCP 进程租约的生命周期和安全边界测试。"""

from __future__ import annotations

import asyncio
import hashlib
import os
import signal
import socket
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from chemistry_multiagent.investigation import evidence_mcp_lease as lease_module
from chemistry_multiagent.investigation.evidence import ReviewEvidenceStore
from chemistry_multiagent.investigation.evidence_mcp_lease import (
    EvidenceMcpLeaseConfigurationError,
    EvidenceMcpLeaseIdentityError,
    EvidenceMcpLeasePortError,
    EvidenceMcpLeaseStartupError,
    TaskEvidenceMcpLease,
)


RESULT_VECTOR = tuple((f"combination-{index}", "1") for index in range(1, 6))
CAPABILITY = "lease-capability-" + "x" * 40


def _free_port() -> int:
    """申请一个当前未监听的固定端口；真正启动前仍由租约再次核对。"""

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    port = int(server.getsockname()[1])
    server.close()
    return port


def _create_snapshot(
    store: ReviewEvidenceStore,
    root: Path,
    *,
    task_id: str = "task-1",
    batch_id: str = "batch-1",
    with_artifact: bool = False,
) -> tuple[Any, Any | None]:
    """创建完整五项快照，可选登记一个本地文本 artifact。"""

    groups = {pair: [] for pair in RESULT_VECTOR}
    artifact = None
    if with_artifact:
        source = root / f"{batch_id}.txt"
        source.write_text("lease evidence", encoding="utf-8")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        source_ref = "art_" + hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:32]
        artifact = store.register_source_artifact(
            source,
            source_artifact_ref=source_ref,
            expected_source_sha256=digest,
        )
        groups[RESULT_VECTOR[0]] = [artifact]
    return (
        store.create_batch_evidence_snapshot(task_id, batch_id, RESULT_VECTOR, groups),
        artifact,
    )


def _make_lease(tmp_path: Path, store: ReviewEvidenceStore, *, port: int | None = None) -> TaskEvidenceMcpLease:
    """创建使用标准仓库 Python 和私有 runtime 目录的租约。"""

    return TaskEvidenceMcpLease(
        "task-1",
        store.root_dir,
        tmp_path / "runtime",
        port=port or _free_port(),
        capability_token=CAPABILITY,
        startup_timeout=4.0,
    )


async def _read_one_artifact(
    endpoint: str,
    token: str,
    snapshot_id: str,
    artifact_id: str,
) -> dict[str, Any]:
    """通过真实 MCP client 初始化、列工具并读取一页证据。"""

    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async with httpx.AsyncClient(
        headers={"x-arche-evidence-capability": token},
        timeout=3.0,
    ) as client:
        async with streamable_http_client(endpoint, http_client=client) as streams:
            read_stream, write_stream, _ = streams
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                tools = await session.list_tools()
                assert [tool.name for tool in tools.tools] == ["read_snapshot_artifact"]
                result = await session.call_tool(
                    "read_snapshot_artifact",
                    {"snapshot_id": snapshot_id, "artifact_id": artifact_id},
                )
                assert result.isError is False
                assert result.structuredContent is not None
                return dict(result.structuredContent)


def test_real_subprocess_readiness_and_authenticated_unique_tool(tmp_path: Path) -> None:
    """真实 FastMCP 子进程必须通过认证 readiness，且只允许读取工具。"""

    store = ReviewEvidenceStore(tmp_path / "store")
    snapshot, artifact = _create_snapshot(store, tmp_path, with_artifact=True)
    assert artifact is not None
    lease = _make_lease(tmp_path, store)
    token = CAPABILITY
    try:
        with lease.serve_evidence_snapshot(snapshot.snapshot_id) as running:
            assert running.endpoint.endswith("/mcp")
            assert running.is_running
            public = asyncio.run(
                _read_one_artifact(
                    running.endpoint,
                    token,
                    snapshot.snapshot_id,
                    artifact.artifact_id,
                )
            )
            assert public["content"] == "lease evidence"
            unauthorized = httpx.post(
                running.endpoint,
                headers={
                    "content-type": "application/json",
                    "accept": "application/json",
                    "x-arche-evidence-capability": "wrong-" + "y" * 40,
                },
                json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
                timeout=3.0,
            )
            assert unauthorized.status_code == 401
    finally:
        lease.stop_evidence_mcp()
    assert lease.active_snapshot_id is None


def test_capability_never_enters_command_receipt_or_environment(tmp_path: Path) -> None:
    """命令行、receipt、repr 和子进程环境都不包含 capability 明文。"""

    store = ReviewEvidenceStore(tmp_path / "store")
    snapshot, _ = _create_snapshot(store, tmp_path)
    lease = _make_lease(tmp_path, store)
    lease.start_evidence_snapshot(snapshot.snapshot_id)
    try:
        assert CAPABILITY not in repr(lease)
        assert CAPABILITY not in lease.receipt_path.read_text(encoding="ascii")
        assert CAPABILITY not in " ".join(str(value) for value in lease._process.args)  # type: ignore[union-attr]
        environment = lease._build_child_environment()
        assert set(environment) == {"PYTHONPATH", "PYTHONUNBUFFERED", "LC_ALL"}
        assert CAPABILITY not in "\n".join(f"{key}={value}" for key, value in environment.items())
        assert os.stat(lease.runtime_root).st_mode & 0o777 == 0o700
        for path in (lease.capability_file, lease.receipt_path, lease.stdout_path, lease.stderr_path):
            assert path.stat().st_mode & 0o777 == 0o600
        assert CAPABILITY not in lease.stdout_path.read_text(encoding="utf-8")
        assert CAPABILITY not in lease.stderr_path.read_text(encoding="utf-8")
    finally:
        lease.stop_evidence_mcp()


def test_constructor_rejects_non_loopback_broad_path_and_nonstandard_python(tmp_path: Path) -> None:
    """构造阶段拒绝外部监听地址、宽泛 runtime 和非标准解释器。"""

    store = ReviewEvidenceStore(tmp_path / "store")
    with pytest.raises(EvidenceMcpLeaseConfigurationError, match="loopback"):
        TaskEvidenceMcpLease(
            "task-1",
            store.root_dir,
            tmp_path / "runtime-a",
            host="0.0.0.0",
            port=_free_port(),
            capability_token=CAPABILITY,
        )
    with pytest.raises(EvidenceMcpLeaseConfigurationError, match="too broad"):
        TaskEvidenceMcpLease(
            "task-1",
            store.root_dir,
            "/tmp",
            port=_free_port(),
            capability_token=CAPABILITY,
        )
    with pytest.raises(EvidenceMcpLeaseConfigurationError, match="repository .venv"):
        TaskEvidenceMcpLease(
            "task-1",
            store.root_dir,
            tmp_path / "runtime-b",
            port=_free_port(),
            python_executable="/usr/bin/python3",
            capability_token=CAPABILITY,
        )


def test_wrong_task_or_snapshot_fails_before_process_start(tmp_path: Path) -> None:
    """错误 task、未知 snapshot 不应创建子进程或开放端口。"""

    store = ReviewEvidenceStore(tmp_path / "store")
    own, _ = _create_snapshot(store, tmp_path)
    foreign, _ = _create_snapshot(store, tmp_path, task_id="task-2", batch_id="foreign")
    lease = _make_lease(tmp_path, store)
    with pytest.raises(EvidenceMcpLeaseStartupError):
        lease.start_evidence_snapshot(foreign.snapshot_id)
    with pytest.raises(EvidenceMcpLeaseStartupError):
        lease.start_evidence_snapshot("snap_unknown000000000000000000000000000000000000000000")
    assert lease.process_pid is None
    assert lease.active_snapshot_id is None
    assert not lease.capability_file.exists()

    wrong_task = TaskEvidenceMcpLease(
        "task-2",
        store.root_dir,
        tmp_path / "runtime-foreign",
        port=_free_port(),
        capability_token=CAPABILITY,
    )
    with pytest.raises(EvidenceMcpLeaseStartupError):
        wrong_task.start_evidence_snapshot(own.snapshot_id)


def test_unknown_port_occupant_fails_closed(tmp_path: Path) -> None:
    """未知 TCP 监听者占用固定端口时租约不启动、不盲杀。"""

    store = ReviewEvidenceStore(tmp_path / "store")
    snapshot, _ = _create_snapshot(store, tmp_path)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    lease = _make_lease(tmp_path, store, port=int(listener.getsockname()[1]))
    try:
        with pytest.raises(EvidenceMcpLeasePortError, match="occupied"):
            lease.start_evidence_snapshot(snapshot.snapshot_id)
        assert lease.process_pid is None
    finally:
        listener.close()


def test_same_snapshot_reentry_rechecks_readiness_and_rotation_is_ordered(tmp_path: Path) -> None:
    """同快照重入复用同一 PID 但再次 readiness，不同快照先停旧服务。"""

    store = ReviewEvidenceStore(tmp_path / "store")
    first, _ = _create_snapshot(store, tmp_path, batch_id="first")
    second, _ = _create_snapshot(store, tmp_path, batch_id="second")
    lease = _make_lease(tmp_path, store)
    lease.start_evidence_snapshot(first.snapshot_id)
    first_pid = lease.process_pid
    calls: list[str] = []
    original_readiness = lease._perform_mcp_readiness

    def count_readiness() -> None:
        calls.append("readiness")
        original_readiness()

    lease._perform_mcp_readiness = count_readiness  # type: ignore[method-assign]
    try:
        lease.start_evidence_snapshot(first.snapshot_id)
        assert lease.process_pid == first_pid
        assert calls == ["readiness"]
        lease.start_evidence_snapshot(second.snapshot_id)
        assert lease.active_snapshot_id == second.snapshot_id
        assert lease.process_pid != first_pid
    finally:
        lease.stop_evidence_mcp()
    lease.stop_evidence_mcp()


def test_reentry_does_not_accept_unresponsive_service(tmp_path: Path) -> None:
    """同快照重入 readiness 失败时不返回旧服务，并安全回收它。"""

    store = ReviewEvidenceStore(tmp_path / "store")
    snapshot, _ = _create_snapshot(store, tmp_path)
    lease = _make_lease(tmp_path, store)
    lease.start_evidence_snapshot(snapshot.snapshot_id)
    original = lease._perform_mcp_readiness

    def fail_readiness() -> None:
        raise EvidenceMcpLeaseStartupError("probe failed")

    lease._perform_mcp_readiness = fail_readiness  # type: ignore[method-assign]
    with pytest.raises(EvidenceMcpLeaseStartupError, match="probe failed"):
        lease.start_evidence_snapshot(snapshot.snapshot_id)
    assert lease.active_snapshot_id is None
    assert lease.process_pid is None
    # 恢复方法仅用于避免测试对象保留替身；此处不再启动服务。
    lease._perform_mcp_readiness = original  # type: ignore[method-assign]


def test_capture_identity_failure_closes_direct_child(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """子进程启动后身份捕获失败时用直接子句柄收口，不遗留监听者。"""

    store = ReviewEvidenceStore(tmp_path / "store")
    snapshot, _ = _create_snapshot(store, tmp_path)
    lease = _make_lease(tmp_path, store)

    def fail_capture(_pid: int) -> Any:
        raise RuntimeError("identity unavailable")

    monkeypatch.setattr(lease_module, "capture_process_identity", fail_capture)
    with pytest.raises(EvidenceMcpLeaseIdentityError):
        lease.start_evidence_snapshot(snapshot.snapshot_id)
    assert lease.process_pid is None
    assert lease.active_snapshot_id is None
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.settimeout(0.3)
    try:
        with pytest.raises((ConnectionRefusedError, ConnectionResetError, OSError)):
            probe.connect((lease.host, lease.port))
    finally:
        probe.close()


def test_forged_receipt_identity_is_not_killed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """receipt 指向当前测试进程时，完整命令行不匹配即可拒绝回收。"""

    store = ReviewEvidenceStore(tmp_path / "store")
    snapshot, _ = _create_snapshot(store, tmp_path)
    lease = _make_lease(tmp_path, store)
    identity = lease_module.capture_process_identity(os.getpid()).as_dict()  # type: ignore[union-attr]
    lease_module._write_private_json(
        lease.receipt_path,
        lease._receipt_payload(snapshot.snapshot_id, identity),
    )
    kill_calls: list[int] = []
    monkeypatch.setattr(lease_module.os, "kill", lambda pid, _signal: kill_calls.append(pid))
    with pytest.raises(EvidenceMcpLeaseIdentityError):
        lease.start_evidence_snapshot(snapshot.snapshot_id)
    assert kill_calls == []


def test_crashed_child_can_be_reaped_without_blind_kill(tmp_path: Path) -> None:
    """服务异常退出后 stop 识别 stopped identity 并清理 receipt。"""

    store = ReviewEvidenceStore(tmp_path / "store")
    snapshot, _ = _create_snapshot(store, tmp_path)
    lease = _make_lease(tmp_path, store)
    lease.start_evidence_snapshot(snapshot.snapshot_id)
    assert lease.process_pid is not None
    os.kill(lease.process_pid, signal.SIGKILL)
    deadline = time.monotonic() + 2.0
    while lease.is_running and time.monotonic() < deadline:
        time.sleep(0.03)
    lease.stop_evidence_mcp()
    assert lease.receipt_path.exists() is False
    assert lease.capability_file.exists() is False
