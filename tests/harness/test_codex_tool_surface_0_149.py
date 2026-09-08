"""用 vendored Codex 0.149.0 app-server 核对真实 Responses 工具面。"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time
from typing import Any

from openai_codex import ApprovalMode, Codex, CodexConfig, Sandbox

from chemistry_multiagent.harness import codex_sdk_adapter as adapter


class _FakeResponsesHandler(BaseHTTPRequestHandler):
    """记录本地 fake Responses 请求并立刻返回受控错误。"""

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", "0"))
        body = json.loads(self.rfile.read(length))
        self.server.captured_bodies.append(body)
        response = b'{"error":{"message":"local surface probe stop"}}'
        self.send_response(400)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, _format: str, *_args: Any) -> None:
        return


class _FakeEvidenceMcpHandler(BaseHTTPRequestHandler):
    """提供本地只读 MCP tools/list 响应，不连接第三方服务。"""

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", "0"))
        request = json.loads(self.rfile.read(length))
        method = request.get("method")
        if method == "initialize":
            result = {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "local-surface-mcp", "version": "0.1"},
            }
        elif method == "tools/list":
            result = {
                "tools": [
                    {
                        "name": self.server.tool_name,
                        "description": "Local surface probe tool",
                        "inputSchema": {
                            "type": "object",
                            "properties": {},
                            "additionalProperties": False,
                        },
                    }
                ]
            }
        elif method == "notifications/initialized":
            self.send_response(202)
            self.end_headers()
            return
        else:
            result = {}
        response = json.dumps(
            {"jsonrpc": "2.0", "id": request.get("id"), "result": result}
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(response)))
        self.send_header("mcp-session-id", "local-surface-session")
        self.end_headers()
        self.wfile.write(response)

    def do_GET(self) -> None:
        self.send_response(405)
        self.end_headers()

    def log_message(self, _format: str, *_args: Any) -> None:
        return


def _successful_sse_response(response_id: str, text: str) -> bytes:
    """构造真实 0.149 可消费的最小 Responses SSE 成功响应。"""

    events = (
        {"type": "response.created", "response": {"id": response_id}},
        {
            "type": "response.output_item.done",
            "item": {
                "type": "message",
                "role": "assistant",
                "id": f"msg-{response_id}",
                "content": [{"type": "output_text", "text": text}],
            },
        },
        {
            "type": "response.completed",
            "response": {
                "id": response_id,
                "usage": {
                    "input_tokens": 1,
                    "input_tokens_details": None,
                    "output_tokens": 1,
                    "output_tokens_details": None,
                    "total_tokens": 2,
                },
            },
        },
    )
    return (
        "\n".join(
            f"event: {event['type']}\ndata: {json.dumps(event)}\n"
            for event in events
        )
        + "\n"
    ).encode("utf-8")


class _FakeCompactionResponsesHandler(BaseHTTPRequestHandler):
    """依次完成普通 turn 和本地压缩，并记录两次请求正文。"""

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", "0"))
        self.server.captured_bodies.append(json.loads(self.rfile.read(length)))
        response_index = len(self.server.captured_bodies)
        response = _successful_sse_response(
            f"response-{response_index}",
            "history" if response_index == 1 else "compacted",
        )
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, _format: str, *_args: Any) -> None:
        return


def _run_real_surface_probe(
    *,
    profile: str,
    tmp_path: Path,
) -> tuple[str, ...]:
    """启动真实 0.149 app-server，向本地 fake Responses 端点取工具名。"""

    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeResponsesHandler)
    server.captured_bodies = []
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    mcp_server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeEvidenceMcpHandler)
    mcp_server.tool_name = (
        "read_snapshot_artifact"
        if profile == adapter.CODEX_PROFILE_TASK_REVIEW
        else "run_scoped_calculation"
    )
    mcp_thread = threading.Thread(target=mcp_server.serve_forever, daemon=True)
    mcp_thread.start()
    workspace = tmp_path / profile / "workspace"
    codex_home = tmp_path / profile / "codex-home"
    workspace.mkdir(parents=True)
    codex_home.mkdir(parents=True)
    provider_url = f"http://127.0.0.1:{server.server_port}/v1"
    raw_config = {
        "model_provider": "surface_probe",
        "model_providers": {
            "surface_probe": {
                "name": "local surface probe",
                "base_url": provider_url,
                "env_key": "SURFACE_PROBE_KEY",
                "wire_api": "responses",
            }
        },
        "mcp_servers": {
            (
                "evidence_reader"
                if profile == adapter.CODEX_PROFILE_TASK_REVIEW
                else "minichem_toolbox"
            ): {
                "url": f"http://127.0.0.1:{mcp_server.server_port}/mcp",
                "env_http_headers": {"Authorization": "MCP_PROBE_TOKEN"},
                "required": True,
                "startup_timeout_sec": 5,
                "tool_timeout_sec": 10,
            }
        },
    }
    effective_config = adapter.build_codex_capability_config(
        thread_config=raw_config,
        workspace=workspace,
        profile=profile,
        env_key="SURFACE_PROBE_KEY",
    )
    wrapper = (
        Path(__file__).resolve().parents[2]
        / "third_party"
        / "codex_cli"
        / "0.149.0"
        / "linux-x64"
        / "codex"
    )
    codex = Codex(
        CodexConfig(
            codex_bin=str(wrapper),
            cwd=str(workspace),
            env={
                "CODEX_HOME": str(codex_home),
                "SURFACE_PROBE_KEY": "local-only-test-value",
                "MCP_PROBE_TOKEN": "local-only-mcp-value",
            },
        )
    )
    try:
        thread = codex.thread_start(
            approval_mode=ApprovalMode.deny_all,
            config=effective_config,
            cwd=str(workspace),
            model="local-surface-probe",
            model_provider="surface_probe",
            sandbox=Sandbox.read_only,
        )
        try:
            for _event in thread.turn("surface probe").stream():
                pass
        except BaseException:
            pass
    finally:
        codex.close()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)
        mcp_server.shutdown()
        mcp_server.server_close()
        mcp_thread.join(timeout=5)
    assert server.captured_bodies
    tools = server.captured_bodies[0].get("tools")
    assert isinstance(tools, list)
    return tuple(
        item.get("name", item.get("type"))
        for item in tools
        if isinstance(item, dict) and isinstance(item.get("name", item.get("type")), str)
    )


def _run_real_compact_prompt_probe(tmp_path: Path, prompt: str) -> tuple[dict[str, Any], ...]:
    """通过真实 0.149 app-server 对本地 custom provider 执行一次手动压缩。"""

    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeCompactionResponsesHandler)
    server.captured_bodies = []
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    workspace = tmp_path / "compact" / "workspace"
    codex_home = tmp_path / "compact" / "codex-home"
    workspace.mkdir(parents=True)
    codex_home.mkdir(parents=True)
    provider_url = f"http://127.0.0.1:{server.server_port}/v1"
    effective_config = adapter.build_codex_capability_config(
        thread_config={
            "compact_prompt": prompt,
            "model_context_window": 1_000_000,
            "model_provider": "compact_probe",
            "model_providers": {
                "compact_probe": {
                    "name": "local compact probe",
                    "base_url": provider_url,
                    "env_key": "COMPACT_PROBE_KEY",
                    "wire_api": "responses",
                    "supports_websockets": False,
                }
            },
        },
        workspace=workspace,
        profile=adapter.CODEX_PROFILE_INVESTIGATION,
        env_key="COMPACT_PROBE_KEY",
    )
    wrapper = (
        Path(__file__).resolve().parents[2]
        / "third_party"
        / "codex_cli"
        / "0.149.0"
        / "linux-x64"
        / "codex"
    )
    codex = Codex(
        CodexConfig(
            codex_bin=str(wrapper),
            cwd=str(workspace),
            env={
                "CODEX_HOME": str(codex_home),
                "COMPACT_PROBE_KEY": "local-only-test-value",
            },
        )
    )
    try:
        thread = codex.thread_start(
            approval_mode=ApprovalMode.deny_all,
            config=effective_config,
            cwd=str(workspace),
            ephemeral=False,
            model="local-compact-probe",
            model_provider="compact_probe",
            sandbox=Sandbox.read_only,
        )
        thread.run("create local history")
        thread.compact()
        deadline = time.monotonic() + 5.0
        while len(server.captured_bodies) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        codex.close()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)
    return tuple(server.captured_bodies)


def test_real_codex_0149_review_surface_is_evidence_only(tmp_path: Path) -> None:
    """真实 Review Responses 请求不得携带任何 Codex 内建工具。"""

    names = _run_real_surface_probe(
        profile=adapter.CODEX_PROFILE_TASK_REVIEW,
        tmp_path=tmp_path,
    )

    assert "mcp__evidence_reader" in names
    assert set(names).issubset(
        {
            "list_mcp_resources",
            "list_mcp_resource_templates",
            "read_mcp_resource",
            "mcp__evidence_reader",
        }
    )
    assert "request_user_input" not in names
    assert "web_search" not in names
    assert "view_image" not in names
    assert "multi_agent_v1" not in names
    assert "mcp__minichem_toolbox" not in names
    assert "exec_command" not in names
    assert "update_plan" not in names


def test_real_codex_0149_investigation_surface_is_shell_and_plan_only(
    tmp_path: Path,
) -> None:
    """真实 Investigation Responses 请求只携带 shell 和 update_plan。"""

    names = _run_real_surface_probe(
        profile=adapter.CODEX_PROFILE_INVESTIGATION,
        tmp_path=tmp_path,
    )

    assert {
        "exec_command",
        "write_stdin",
        "update_plan",
        "mcp__minichem_toolbox",
    }.issubset(names)
    assert set(names).issubset(
        {
            "exec_command",
            "write_stdin",
            "update_plan",
            "list_mcp_resources",
            "list_mcp_resource_templates",
            "read_mcp_resource",
            "mcp__minichem_toolbox",
        }
    )
    assert "request_user_input" not in names
    assert "web_search" not in names
    assert "view_image" not in names
    assert "multi_agent_v1" not in names
    assert "apply_patch" not in names
    assert "mcp__evidence_reader" not in names


def test_real_codex_0149_custom_provider_uses_investigation_compact_prompt(
    tmp_path: Path,
) -> None:
    """真实 0.149 接受 1M 档位，并在 custom provider 压缩时发送哨兵。"""

    prompt = "ARCHE_INVESTIGATION_COMPACT_SENTINEL\n保留科学证据"
    requests = _run_real_compact_prompt_probe(tmp_path, prompt)
    request_texts = [
        content["text"]
        for request in requests
        for item in request.get("input", [])
        if isinstance(item, dict)
        for content in item.get("content", [])
        if isinstance(content, dict) and isinstance(content.get("text"), str)
    ]

    assert len(requests) == 2
    assert request_texts.count(prompt) == 1
    assert any(
        content.get("text") == prompt
        for item in requests[1].get("input", [])
        if isinstance(item, dict)
        for content in item.get("content", [])
        if isinstance(content, dict)
    )
