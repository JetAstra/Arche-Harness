"""Review evidence MCP 服务的 surface、隔离、认证和 Streamable HTTP 测试。"""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from chemistry_multiagent.investigation.evidence import ReviewEvidenceStore
from chemistry_multiagent.investigation.evidence_mcp_server import (
    DEFAULT_EVIDENCE_CAPABILITY_HEADER,
    EvidenceAuthenticatedHTTPApp,
    EvidenceMcpConfigurationError,
    create_evidence_capability_file,
    create_evidence_http_app,
    create_evidence_mcp_server,
    load_evidence_capability_file,
    load_snapshot_allowlist_file,
    main,
)


RESULT_VECTOR = tuple((f"combination-{index}", "1") for index in range(1, 6))


def _snapshot(store: ReviewEvidenceStore, source: Path | None = None):
    """创建只含测试 artifact 的完整五项 task 快照。"""

    groups = {pair: [] for pair in RESULT_VECTOR}
    if source is not None:
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        source_ref = "art_" + hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:32]
        groups[RESULT_VECTOR[0]] = [
            store.register_source_artifact(
                source,
                source_artifact_ref=source_ref,
                expected_source_sha256=digest,
            )
        ]
    return store.create_batch_evidence_snapshot("task-1", "batch-1", RESULT_VECTOR, groups)


def test_server_exposes_only_one_read_tool_and_bound_snapshot(tmp_path: Path) -> None:
    """FastMCP surface 只有读取工具，知道另一 task ID 也不能跨快照读取。"""

    source = tmp_path / "evidence.txt"
    source.write_text("authorized", encoding="utf-8")
    store = ReviewEvidenceStore(tmp_path / "store")
    allowed = _snapshot(store, source)
    other_source = tmp_path / "other.txt"
    other_source.write_text("foreign", encoding="utf-8")
    other_groups = {pair: [] for pair in RESULT_VECTOR}
    other_digest = hashlib.sha256(other_source.read_bytes()).hexdigest()
    other_ref = "art_" + hashlib.sha256(str(other_source).encode("utf-8")).hexdigest()[:32]
    other_groups[RESULT_VECTOR[0]] = [
        store.register_source_artifact(
            other_source,
            source_artifact_ref=other_ref,
            expected_source_sha256=other_digest,
        )
    ]
    foreign = store.create_batch_evidence_snapshot("task-2", "batch-1", RESULT_VECTOR, other_groups)

    server = create_evidence_mcp_server(store, snapshot_id=allowed.snapshot_id)
    tools = server._tool_manager.list_tools()  # type: ignore[attr-defined]
    assert [tool.name for tool in tools] == ["read_snapshot_artifact"]
    assert "register" not in json.dumps(tools, default=str).lower()
    tool = tools[0]
    assert tool.annotations.readOnlyHint is True
    assert tool.annotations.destructiveHint is False

    direct = tool.fn(
        snapshot_id=allowed.snapshot_id,
        artifact_id=allowed.result_pairs[0].artifacts[0].artifact_id,
    )
    assert direct["content"] == "authorized"
    source_attempt = tool.fn(
        snapshot_id=allowed.snapshot_id,
        artifact_id=allowed.result_pairs[0].artifacts[0].source_artifact_ref,
    )
    assert source_attempt["error"]["code"] == "not_found"
    denied = tool.fn(
        snapshot_id=foreign.snapshot_id,
        artifact_id=allowed.result_pairs[0].artifacts[0].artifact_id,
    )
    assert denied["error"]["code"] == "unauthorized"
    with pytest.raises(EvidenceMcpConfigurationError, match="Streamable HTTP requires capability"):
        create_evidence_http_app(server)


def test_capability_file_is_private_and_allowlist_is_task_scoped(tmp_path: Path) -> None:
    """capability 只落 0600 文件，allowlist 文件不允许跨 task 或宽权限。"""

    credential = tmp_path / "runtime" / "evidence.cap"
    token = create_evidence_capability_file(credential, token="t" * 48)
    assert token == "t" * 48
    assert oct(credential.stat().st_mode & 0o777) == "0o600"
    assert load_evidence_capability_file(credential) == token

    allowlist = tmp_path / "allowlist.json"
    allowlist.write_text(json.dumps(["snap_abcdefgh"], ensure_ascii=False), encoding="utf-8")
    os.chmod(allowlist, 0o600)
    assert load_snapshot_allowlist_file(allowlist) == ("snap_abcdefgh",)
    os.chmod(allowlist, 0o640)
    with pytest.raises(EvidenceMcpConfigurationError, match="mode 0600"):
        load_snapshot_allowlist_file(allowlist)


def test_streamable_http_reauthenticates_each_message_and_hides_errors(tmp_path: Path) -> None:
    """HTTP 初始化/调用必须带 header，切换 snapshot 或路径参数只返回安全错误。"""

    source = tmp_path / "evidence.txt"
    source.write_text("authorized", encoding="utf-8")
    store = ReviewEvidenceStore(tmp_path / "store")
    snapshot = _snapshot(store, source)
    artifact_id = snapshot.result_pairs[0].artifacts[0].artifact_id
    token = "capability-" + "x" * 40
    server = create_evidence_mcp_server(
        store,
        snapshot_allowlist=[snapshot.snapshot_id],
        capability_token=token,
    )
    app = create_evidence_http_app(server)
    assert isinstance(server.streamable_http_app(), EvidenceAuthenticatedHTTPApp)
    common = {"Accept": "application/json", "Content-Type": "application/json"}
    initialize = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "evidence-test", "version": "1"},
        },
    }
    with TestClient(app, headers={"host": "localhost:9020"}) as client:
        missing = client.post("/mcp", json=initialize, headers=common)
        wrong = client.post(
            "/mcp",
            json=initialize,
            headers={**common, DEFAULT_EVIDENCE_CAPABILITY_HEADER: "wrong-" + "z" * 40},
        )
        assert missing.status_code == 401
        assert wrong.status_code == 401

        initialized = client.post(
            "/mcp",
            json=initialize,
            headers={**common, DEFAULT_EVIDENCE_CAPABILITY_HEADER: token},
        )
        assert initialized.status_code == 200
        session_id = initialized.headers["mcp-session-id"]
        call = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "read_snapshot_artifact",
                    "arguments": {
                        "snapshot_id": snapshot.snapshot_id,
                        "artifact_id": artifact_id,
                    },
                },
            },
            headers={
                **common,
                DEFAULT_EVIDENCE_CAPABILITY_HEADER: token,
                "Mcp-Session-Id": session_id,
                "Mcp-Protocol-Version": "2025-06-18",
            },
        )
        assert call.status_code == 200
        payload = call.json()["result"]["structuredContent"]
        assert payload["content"] == "authorized"

        no_header_call = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "read_snapshot_artifact",
                    "arguments": {"snapshot_id": snapshot.snapshot_id, "artifact_id": artifact_id},
                },
            },
            headers={**common, "Mcp-Session-Id": session_id, "Mcp-Protocol-Version": "2025-06-18"},
        )
        assert no_header_call.status_code == 401

        path_attempt = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {
                    "name": "read_snapshot_artifact",
                    "arguments": {
                        "snapshot_id": snapshot.snapshot_id,
                        "artifact_id": str(tmp_path / "private.txt"),
                    },
                },
            },
            headers={
                **common,
                DEFAULT_EVIDENCE_CAPABILITY_HEADER: token,
                "Mcp-Session-Id": session_id,
                "Mcp-Protocol-Version": "2025-06-18",
            },
        )
        assert path_attempt.status_code == 200
        error_payload = path_attempt.json()["result"]["structuredContent"]
        assert error_payload["error"]["code"] == "invalid_request"
        assert str(tmp_path) not in json.dumps(error_payload)


def test_allowlist_requires_capability_and_cross_task_startup_fails(tmp_path: Path) -> None:
    """多快照服务必须绑定 capability，跨 task allowlist 在启动时拒绝。"""

    store = ReviewEvidenceStore(tmp_path / "store")
    first = _snapshot(store)
    second_groups = {pair: [] for pair in RESULT_VECTOR}
    second = store.create_batch_evidence_snapshot("task-2", "batch-1", RESULT_VECTOR, second_groups)
    with pytest.raises(EvidenceMcpConfigurationError, match="requires capability"):
        create_evidence_mcp_server(store, snapshot_allowlist=[first.snapshot_id])
    with pytest.raises(EvidenceMcpConfigurationError, match="crosses task"):
        create_evidence_mcp_server(
            store,
            snapshot_allowlist=[first.snapshot_id, second.snapshot_id],
            capability_token="capability-" + "x" * 40,
        )


def test_http_cli_rejects_fixed_snapshot_without_capability(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """HTTP CLI 即使绑定单个 snapshot、监听 loopback，也必须配置 capability。"""

    store = ReviewEvidenceStore(tmp_path / "store")
    snapshot = _snapshot(store)
    result = main(
        [
            "--store-root",
            str(store.root_dir),
            "--snapshot-id",
            snapshot.snapshot_id,
            "--transport",
            "streamable_http",
            "--host",
            "127.0.0.1",
        ]
    )
    assert result == 2
    assert "EvidenceMcpConfigurationError" in capsys.readouterr().err
