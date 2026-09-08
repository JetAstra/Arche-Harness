from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from chemistry_multiagent.harness.base import HarnessError, HarnessRequest, HarnessResponse
from chemistry_multiagent.harness.config import DEFAULT_CODEX_BIN, HarnessConfig
from chemistry_multiagent.harness.dispatcher import HarnessDispatcher
from chemistry_multiagent.harness.context import bind_harness_call_context
from chemistry_multiagent.harness import runners as runners_module
from chemistry_multiagent.harness.codex_sdk_adapter import CodexSdkRunResult
from chemistry_multiagent.harness import codex_sdk_adapter as adapter
from chemistry_multiagent.harness.runners import CodexRunner, ReactRunner


def _request(tmp_path: Path) -> HarnessRequest:
    return HarnessRequest(
        messages=(
            {"role": "system", "content": "Return JSON."},
            {"role": "user", "content": '{"ok":true}'},
        ),
        model="fake-model",
        role="planner",
        stage="workflow_generation",
        workspace=tmp_path / "workspace",
        research_task_id="unit-research-task",
        timeout_seconds=10,
    )


def test_harness_request_default_workspace_uses_ignored_runtime() -> None:
    """直接构造请求时也必须使用被忽略的活动运行目录。"""

    request = HarnessRequest(
        messages=({"role": "user", "content": "ok"},),
        model="m",
    )

    assert request.workspace == Path("local_data/runtime/harness_workspaces")


def test_harness_public_identity_is_research_task_id_only() -> None:
    """Harness 请求与路径 helper 不再公开含糊的 run_id 参数。"""

    assert "run_id" not in inspect.signature(HarnessRequest).parameters
    assert "research_task_id" in inspect.signature(HarnessRequest).parameters
    assert "run_id" not in inspect.signature(HarnessRequest.from_compiled).parameters
    assert "research_task_id" in inspect.signature(HarnessRequest.from_compiled).parameters
    assert "run_id" not in inspect.signature(HarnessConfig.workspace_for).parameters
    assert "run_id" not in inspect.signature(HarnessConfig.trace_for).parameters
    request = HarnessRequest(
        messages=({"role": "user", "content": "ok"},),
        model="m",
        research_task_id="research-task-1",
    )
    assert request.research_task_id == "research-task-1"
    assert not hasattr(request, "run_id")
    with pytest.raises(HarnessError, match="research_task_id"):
        HarnessRequest.from_compiled(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "ok"}],
                "run_id": "legacy-run",
            },
            workspace=Path("workspace"),
        )


def test_codex_harness_roles_use_codex_isolated_workspace_and_trace(tmp_path: Path) -> None:
    """Investigation/Review 与旧 Planner 一样使用独立 Codex workspace。"""

    config = HarnessConfig(
        kind="codex",
        workspace_root=tmp_path / "workspaces",
        trace_root=tmp_path / "traces",
    )

    for role in ("planner", "investigation", "review", "task_review"):
        assert config.workspace_for(role=role, research_task_id="task-1").parts[-3] == "codex"
        assert config.trace_for(role=role, research_task_id="task-1").parts[-3] == "codex"


def test_fixed_codex_wrapper_gets_session_sandbox_and_mcp_read_only_roots(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """固定 0.149 wrapper 必须绑定校验后的 runtime 和 MCP 只读根。"""

    runtime_root = tmp_path / "runtime"
    runtime = SimpleNamespace(
        wrapper=DEFAULT_CODEX_BIN,
        runtime_root=runtime_root,
        native_codex_bin=runtime_root / "native-codex",
        bwrap_bin=runtime_root / "bwrap",
    )
    calls: list[Path] = []

    def fake_validate(*, codex_bin):
        calls.append(Path(codex_bin))
        return runtime

    monkeypatch.setattr(runners_module, "validate_vendored_codex_runtime", fake_validate)
    request = _request(tmp_path)
    request = request.__class__(**{**request.__dict__, "workspace": tmp_path / "workspace"})
    config = HarnessConfig(kind="codex", codex_bin=str(DEFAULT_CODEX_BIN))
    spec = runners_module._build_codex_process_sandbox_spec(
        CodexRunner(config),
        request=request,
        codex_home=tmp_path / "codex-home",
    )

    assert spec is not None
    assert calls == [DEFAULT_CODEX_BIN]
    assert spec.codex_bin == DEFAULT_CODEX_BIN
    assert spec.runtime_root == runtime_root
    assert {root.name for root in spec.read_only_roots} == {
        ".venv",
        "src",
        "minichem_toolbox",
    }


def test_codex_turn_default_home_is_sibling_of_workspace(tmp_path: Path) -> None:
    """默认 CODEX_HOME 与 workspace 分离，满足双可写根边界。"""

    workspace = tmp_path / "run" / "planner"
    home = workspace.with_name(workspace.name + ".codex_home")
    assert home.parent == workspace.parent
    assert not home.is_relative_to(workspace)


def _fake_cli(path: Path, output: str) -> Path:
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        f"print({output!r})\n"
        "for arg in sys.argv:\n"
        "    if arg == '--output-last-message':\n"
        "        index = sys.argv.index(arg) + 1\n"
        "        pathlib.Path(sys.argv[index]).write_text('{\\\"ok\\\":true}\\n', encoding='utf-8')\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _write_fake_completed_receipt(
    kwargs: dict[str, object],
    *,
    thread_id: str,
    turn_id: str,
) -> None:
    """测试替身按 SDK 合同写入完整 receipt，避免伪造成功 turn。"""

    adapter._write_completed_turn_receipt(
        kwargs["completed_turn_receipt_root"],
        thread_id=thread_id,
        turn_id=turn_id,
        request_digest=kwargs["request_digest"],
        turn_count_before=kwargs["turn_count_before"],
        thread_policy_digest=kwargs["thread_policy_digest"],
    )


def test_codex_runner_isolated_and_preserves_raw_json(
    tmp_path: Path, monkeypatch
) -> None:
    def fake_sdk_turn(**kwargs):
        request = kwargs["request"]
        kwargs["on_thread_ready"]("session-unit")
        kwargs["trace_path"].write_text(
            '{"type":"codex_notification","method":"turn/completed"}\n',
            encoding="utf-8",
        )
        _write_fake_completed_receipt(
            kwargs,
            thread_id="session-unit",
            turn_id="turn-unit",
        )
        return CodexSdkRunResult(
            HarnessResponse(
                content='{"ok":true}',
                model=request.model,
                raw_trace_path=str(kwargs["trace_path"]),
                metadata={"harness": "codex", "role": request.role, "stage": request.stage},
            ),
            "session-unit",
            "turn-unit",
        )

    monkeypatch.setattr(runners_module, "run_codex_sdk_turn", fake_sdk_turn)
    config = HarnessConfig(
        kind="codex",
        workspace_root=tmp_path / "workspaces",
        trace_root=tmp_path / "traces",
        codex_bin=str(
            _fake_cli(
                tmp_path / "codex",
                '{"type":"thread.started","thread_id":"session-unit"}\n'
                '{"type":"agent_message","text":"{\\"ok\\":true}"}',
            )
        ),
        base_url="https://example.invalid/v1",
        reasoning_effort="high",
    )
    monkeypatch.setenv("ARCHE_CODEX_API_KEY", "unit-test-secret")
    request = _request(tmp_path)
    response = CodexRunner(config).run(request)
    assert response.content == '{"ok":true}'
    assert response.to_provider_envelope()["harness"]["name"] == "codex"
    assert (tmp_path / "workspaces" / "codex" / "unit-research-task" / "planner").is_dir()
    assert (tmp_path / "traces" / "codex" / "unit-research-task" / "planner" / "workflow_generation.jsonl").is_file()
    assert (tmp_path / "traces" / "codex" / "unit-research-task" / "planner" / "workflow_generation.final.md").is_file()


def test_codex_runner_rejects_empty_turn_id_before_manifest_progress(
    tmp_path: Path, monkeypatch
) -> None:
    """session-aware fake 若没有 turn_id，Runner 不能推进成功计数。"""

    def fake_sdk_turn(**kwargs):
        kwargs["on_thread_ready"]("session-empty-turn")
        return CodexSdkRunResult(
            HarnessResponse(content='{"ok":true}', model=kwargs["request"].model),
            "session-empty-turn",
            "",
        )

    monkeypatch.setattr(runners_module, "run_codex_sdk_turn", fake_sdk_turn)
    monkeypatch.setenv("ARCHE_CODEX_API_KEY", "unit-test-secret")
    monkeypatch.setenv("ARCHE_PLANNER_MCP_ENABLED", "0")
    config = HarnessConfig(
        kind="codex",
        workspace_root=tmp_path / "workspaces",
        trace_root=tmp_path / "traces",
        codex_bin=str(_fake_cli(tmp_path / "codex", "")),
        base_url="https://example.invalid/v1",
    )

    with pytest.raises(HarnessError, match="no turn_id"):
        CodexRunner(config).run(_request(tmp_path))

    manifest = json.loads(
        (
            tmp_path
            / "workspaces"
            / "codex"
            / "unit-research-task"
            / "planner"
            / ".arche_planner_session.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["turn_count"] == 0


def test_codex_runner_uses_arche_provider_without_exposing_secret(
    tmp_path: Path, monkeypatch
) -> None:
    """provider/MCP 配置进入 SDK thread config，密钥只保留在进程环境。"""

    capture = tmp_path / "captured.json"
    fake = tmp_path / "codex-capture"
    fake.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, pathlib, sys\n"
        f"pathlib.Path({str(capture)!r}).write_text(json.dumps({{'argv': sys.argv[1:], 'has_key': bool(os.environ.get('ARCHE_CODEX_API_KEY')), 'codex_home': os.environ.get('CODEX_HOME')}}), encoding='utf-8')\n"
        "print('{\"type\":\"thread.started\",\"thread_id\":\"session-capture\"}')\n"
        "for index, arg in enumerate(sys.argv):\n"
        "    if arg == '--output-last-message':\n"
        "        pathlib.Path(sys.argv[index + 1]).write_text('{\\\"ok\\\":true}\\n', encoding='utf-8')\n"
        "print('{\\\"type\\\":\\\"agent_message\\\",\\\"text\\\":\\\"{\\\\\\\"ok\\\\\\\":true}\\\"}')\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    captured_sdk = {}

    def fake_sdk_turn(**kwargs):
        captured_sdk.update(kwargs)
        request = kwargs["request"]
        kwargs["on_thread_ready"]("session-capture")
        _write_fake_completed_receipt(
            kwargs,
            thread_id="session-capture",
            turn_id="turn-capture",
        )
        return CodexSdkRunResult(
            HarnessResponse(content='{"ok":true}', model=request.model),
            "session-capture",
            "turn-capture",
        )

    monkeypatch.setattr(runners_module, "run_codex_sdk_turn", fake_sdk_turn)
    monkeypatch.setenv("ARCHE_CODEX_API_KEY", "do-not-log-this-secret")
    config = HarnessConfig(
        kind="codex",
        workspace_root=tmp_path / "workspaces",
        trace_root=tmp_path / "traces",
        codex_bin=str(fake),
        codex_home=tmp_path / "codex-home",
        model="planner-tool-model",
        base_url="http://127.0.0.1:50917/v1",
        reasoning_effort="high",
    )

    request = _request(tmp_path)
    response = CodexRunner(config).run(request)
    sdk_config = captured_sdk["config"]
    thread_config = captured_sdk["thread_config"]

    assert response.content == '{"ok":true}'
    assert sdk_config.env == {
        "CODEX_HOME": str((tmp_path / "codex-home").resolve()),
        "ARCHE_CODEX_API_KEY": "do-not-log-this-secret",
    }
    assert captured_sdk["secret_environment"] == {
        "ARCHE_CODEX_API_KEY": "do-not-log-this-secret"
    }
    assert captured_sdk["trace_sensitive_values"] == ("do-not-log-this-secret",)
    assert (tmp_path / "codex-home").is_dir()
    assert sdk_config.codex_bin == str((tmp_path / "codex-capture").resolve())
    assert thread_config["model_provider"] == "arche_codex"
    assert thread_config["model_providers"]["arche_codex"]["base_url"] == "http://127.0.0.1:50917/v1"
    assert thread_config["model_providers"]["arche_codex"]["env_key"] == "ARCHE_CODEX_API_KEY"
    assert thread_config["model_providers"]["arche_codex"]["request_max_retries"] == 6
    assert thread_config["model_providers"]["arche_codex"]["stream_max_retries"] == 6
    assert thread_config["model_reasoning_effort"] == "high"
    assert thread_config["model_reasoning_summary"] == "none"
    assert thread_config["model_supports_reasoning_summaries"] is False
    assert response.model == "planner-tool-model"
    assert "start_mcp.sh" in thread_config["mcp_servers"]["minichem_toolbox"]["command"]
    assert thread_config["mcp_servers"]["minichem_toolbox"]["args"] == [
        "--transport", "stdio", "--discovery-mode", "full"
    ]
    assert thread_config["mcp_servers"]["minichem_toolbox"]["required"] is True
    assert thread_config["mcp_servers"]["minichem_toolbox"]["tool_timeout_sec"] == 21600
    assert "arche_planner_control" not in json.dumps(thread_config)
    assert "do-not-log-this-secret" not in json.dumps(thread_config)


def test_codex_runner_replays_receipt_after_completion_recovery_without_new_model_turn(
    tmp_path: Path, monkeypatch
) -> None:
    """Runner 在 receipt 恢复时复用同一 thread，第二次调用不能再次提交模型。"""

    calls: list[dict[str, object]] = []
    model_calls = 0

    def fake_sdk_turn(**kwargs):
        nonlocal model_calls
        calls.append(kwargs)
        request = kwargs["request"]
        if len(calls) == 1:
            kwargs["on_thread_ready"]("session-recovery")
            model_calls += 1
            adapter._write_completed_turn_receipt(
                kwargs["completed_turn_receipt_root"],
                thread_id="session-recovery",
                turn_id="turn-recovery",
                request_digest=kwargs["request_digest"],
                turn_count_before=kwargs["turn_count_before"],
                thread_policy_digest=kwargs["thread_policy_digest"],
            )
            raise adapter.CodexCompletionRecoveryPending(
                "simulated crash after completed turn",
                turn_id="turn-recovery",
                reason="completion_receipt",
            )
        assert calls[1]["session_id"] == "session-recovery"
        assert calls[1]["completion_recovery"]["reason"] == "completion_receipt"
        return CodexSdkRunResult(
            HarnessResponse(content='{"ok":true}', model=request.model),
            "session-recovery",
            "turn-recovery",
            replayed=True,
            receipt_turn_count_before=0,
            request_digest=kwargs["request_digest"],
            thread_policy_digest=kwargs["thread_policy_digest"],
        )

    monkeypatch.setattr(runners_module, "run_codex_sdk_turn", fake_sdk_turn)
    monkeypatch.setenv("ARCHE_CODEX_API_KEY", "unit-test-secret")
    monkeypatch.setenv("ARCHE_PLANNER_MCP_ENABLED", "0")
    config = HarnessConfig(
        kind="codex",
        workspace_root=tmp_path / "workspaces",
        trace_root=tmp_path / "traces",
        codex_bin=str(_fake_cli(tmp_path / "codex", "")),
        base_url="https://example.invalid/v1",
    )
    runner = CodexRunner(config)
    request = _request(tmp_path)

    with pytest.raises(adapter.CodexCompletionRecoveryPending):
        runner.run(request)
    response = runner.run(request)

    assert response.content == '{"ok":true}'
    assert model_calls == 1
    assert not list(
        (
            tmp_path
            / "workspaces"
            / "codex"
            / "unit-research-task"
            / "planner"
            / "completed_turn_receipts"
        ).glob("*.json")
    )


def test_codex_planner_persists_one_session_and_resumes_it(
    tmp_path: Path, monkeypatch
) -> None:
    sdk_calls = []

    def fake_sdk_turn(**kwargs):
        sdk_calls.append(kwargs)
        request = kwargs["request"]
        kwargs["on_thread_ready"]("persistent-session")
        turn_id = f"turn-persistent-{len(sdk_calls)}"
        _write_fake_completed_receipt(
            kwargs,
            thread_id="persistent-session",
            turn_id=turn_id,
        )
        return CodexSdkRunResult(
            HarnessResponse(content='{"ok":true}', model=request.model),
            "persistent-session",
            turn_id,
        )

    monkeypatch.setattr(runners_module, "run_codex_sdk_turn", fake_sdk_turn)
    capture = tmp_path / "calls.jsonl"
    fake = tmp_path / "codex-persistent"
    fake.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys\n"
        f"pathlib.Path({str(capture)!r}).open('a', encoding='utf-8').write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "for index, arg in enumerate(sys.argv):\n"
        "    if arg == '--output-last-message':\n"
        "        pathlib.Path(sys.argv[index + 1]).write_text('{\\\"ok\\\":true}\\n', encoding='utf-8')\n"
        "print('{\\\"type\\\":\\\"thread.started\\\",\\\"thread_id\\\":\\\"persistent-session\\\"}')\n"
        "print('{\\\"type\\\":\\\"agent_message\\\",\\\"text\\\":\\\"{\\\\\"ok\\\\\":true}\\\"}')\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    monkeypatch.setenv("ARCHE_CODEX_API_KEY", "unit-test-secret")
    monkeypatch.setenv("ARCHE_PLANNER_MCP_ENABLED", "0")
    config = HarnessConfig(
        kind="codex",
        workspace_root=tmp_path / "workspaces",
        trace_root=tmp_path / "traces",
        codex_bin=str(fake),
        base_url="https://example.invalid/v1",
    )
    first = _request(tmp_path)
    runner = CodexRunner(config)
    initial = runner.run(first)
    resumed = runner.run(
        HarnessRequest(
            **{
                **first.__dict__,
                "stage": "workflow_optimization",
            }
        )
    )

    assert [call["session_id"] for call in sdk_calls] == [None, "persistent-session"]
    assert sdk_calls[0]["thread_config"]["model_provider"] == "arche_codex"
    assert initial.metadata["session_id"] == "persistent-session"
    assert initial.metadata["session_resumed"] is False
    assert resumed.metadata["session_id"] == "persistent-session"
    assert resumed.metadata["session_resumed"] is True
    manifest = json.loads(
        Path(initial.metadata["session_manifest_path"]).read_text(encoding="utf-8")
    )
    assert manifest["session_id"] == "persistent-session"
    assert manifest["turn_count"] == 2
    assert manifest["last_stage"] == "workflow_optimization"


def test_codex_runner_registers_thread_before_turn_and_recovers_failed_first_turn(
    tmp_path: Path, monkeypatch
) -> None:
    """首轮 turn 失败后，下一次调用必须恢复已登记的同一 thread。"""

    calls: list[str | None] = []
    events: list[tuple[str, int, str]] = []
    fail_first_turn = True

    def fake_sdk_turn(**kwargs):
        nonlocal fail_first_turn
        request = kwargs["request"]
        calls.append(kwargs["session_id"])
        kwargs["on_thread_ready"]("thread-registered")
        manifest_path = request.workspace / ".arche_planner_session.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        events.append(("registered", manifest["turn_count"], manifest["status"]))
        if fail_first_turn:
            fail_first_turn = False
            raise HarnessError("simulated first turn failure", kind="transport")
        events.append(("turn_submitted", manifest["turn_count"], manifest["status"]))
        _write_fake_completed_receipt(
            kwargs,
            thread_id="thread-registered",
            turn_id="turn-1",
        )
        return CodexSdkRunResult(
            HarnessResponse(content='{"ok":true}', model=request.model),
            "thread-registered",
            "turn-1",
        )

    monkeypatch.setattr(runners_module, "run_codex_sdk_turn", fake_sdk_turn)
    monkeypatch.setenv("ARCHE_CODEX_API_KEY", "unit-test-secret")
    monkeypatch.setenv("ARCHE_PLANNER_MCP_ENABLED", "0")
    config = HarnessConfig(
        kind="codex",
        workspace_root=tmp_path / "workspaces",
        trace_root=tmp_path / "traces",
        codex_bin=str(_fake_cli(tmp_path / "codex", "")),
        base_url="https://example.invalid/v1",
    )
    request = _request(tmp_path)
    runner = CodexRunner(config)

    with pytest.raises(HarnessError, match="simulated first turn failure"):
        runner.run(request)

    manifest_path = (
        tmp_path
        / "workspaces"
        / "codex"
        / "unit-research-task"
        / "planner"
        / ".arche_planner_session.json"
    )
    failed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert failed_manifest["session_id"] == "thread-registered"
    assert failed_manifest["turn_count"] == 0
    assert failed_manifest["status"] == "created"

    response = runner.run(request)

    assert response.content == '{"ok":true}'
    assert calls == [None, "thread-registered"]
    assert events == [
        ("registered", 0, "created"),
        ("registered", 0, "created"),
        ("turn_submitted", 0, "created"),
    ]
    completed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert completed_manifest["turn_count"] == 1
    assert completed_manifest["status"] == "active"


def test_codex_runner_does_not_persist_worker_recovery_twice(
    tmp_path: Path, monkeypatch
) -> None:
    """SDK 回调已保存 worker recovery 时，Runner 不应重复写同一状态。"""

    recovery_writes = 0
    original_record = runners_module.PlannerSessionStore.record_recovery_pending

    def count_recovery_write(store, **kwargs):
        nonlocal recovery_writes
        recovery_writes += 1
        return original_record(store, **kwargs)

    monkeypatch.setattr(
        runners_module.PlannerSessionStore,
        "record_recovery_pending",
        count_recovery_write,
    )

    def fake_sdk_turn(**kwargs):
        kwargs["on_thread_ready"]("thread-recovery-once")
        kwargs["on_recovery_pending"]("turn-recovery-once")
        raise adapter.CodexWorkerRecoveryPending(
            "simulated worker cleanup uncertainty",
            turn_id="turn-recovery-once",
            reason="worker_cleanup",
        )

    monkeypatch.setattr(runners_module, "run_codex_sdk_turn", fake_sdk_turn)
    monkeypatch.setenv("ARCHE_CODEX_API_KEY", "unit-test-secret")
    monkeypatch.setenv("ARCHE_PLANNER_MCP_ENABLED", "0")
    config = HarnessConfig(
        kind="codex",
        workspace_root=tmp_path / "workspaces",
        trace_root=tmp_path / "traces",
        codex_bin=str(_fake_cli(tmp_path / "codex", "")),
        base_url="https://example.invalid/v1",
    )

    with pytest.raises(adapter.CodexWorkerRecoveryPending):
        CodexRunner(config).run(_request(tmp_path))

    assert recovery_writes == 1


def test_codex_runner_persists_original_request_intent_digest(
    tmp_path: Path, monkeypatch
) -> None:
    """旧 request-intent 冲突必须按旧摘要阻断，不能登记为当前请求身份。"""

    original_digest = "a" * 64

    def fake_sdk_turn(**kwargs):
        kwargs["on_thread_ready"]("thread-intent-mismatch")
        raise adapter.CodexRequestIntentRecoveryPending(
            "request intent does not match the current request",
            turn_id=None,
            reason="request_intent",
            request_digest=original_digest,
        )

    monkeypatch.setattr(runners_module, "run_codex_sdk_turn", fake_sdk_turn)
    monkeypatch.setenv("ARCHE_CODEX_API_KEY", "unit-test-secret")
    monkeypatch.setenv("ARCHE_PLANNER_MCP_ENABLED", "0")
    config = HarnessConfig(
        kind="codex",
        workspace_root=tmp_path / "workspaces",
        trace_root=tmp_path / "traces",
        codex_bin=str(_fake_cli(tmp_path / "codex", "")),
        base_url="https://example.invalid/v1",
    )

    with pytest.raises(adapter.CodexRequestIntentRecoveryPending):
        CodexRunner(config).run(_request(tmp_path))

    workspace = tmp_path / "workspaces" / "codex" / "unit-research-task" / "planner"
    recovery = json.loads(
        (workspace / ".arche_planner_recovery.json").read_text(encoding="utf-8")
    )
    assert recovery["reason"] == "request_intent"
    assert recovery["request_digest"] == original_digest


def test_codex_runner_rejects_resumed_thread_drift_before_turn(
    tmp_path: Path, monkeypatch
) -> None:
    """恢复返回不同 thread 时必须在真实 turn 前失败并保留原 manifest。"""

    calls: list[str | None] = []
    turn_submissions = 0

    def fake_sdk_turn(**kwargs):
        nonlocal turn_submissions
        request = kwargs["request"]
        calls.append(kwargs["session_id"])
        thread_id = "thread-original" if kwargs["session_id"] is None else "thread-drift"
        kwargs["on_thread_ready"](thread_id)
        turn_submissions += 1
        if kwargs["session_id"] is None:
            _write_fake_completed_receipt(
                kwargs,
                thread_id=thread_id,
                turn_id="turn-1",
            )
        return CodexSdkRunResult(
            HarnessResponse(content='{"ok":true}', model=request.model),
            thread_id,
            "turn-1",
        )

    monkeypatch.setattr(runners_module, "run_codex_sdk_turn", fake_sdk_turn)
    monkeypatch.setenv("ARCHE_CODEX_API_KEY", "unit-test-secret")
    monkeypatch.setenv("ARCHE_PLANNER_MCP_ENABLED", "0")
    config = HarnessConfig(
        kind="codex",
        workspace_root=tmp_path / "workspaces",
        trace_root=tmp_path / "traces",
        codex_bin=str(_fake_cli(tmp_path / "codex", "")),
        base_url="https://example.invalid/v1",
    )
    runner = CodexRunner(config)
    request = _request(tmp_path)
    runner.run(request)

    with pytest.raises(HarnessError, match="different session_id"):
        runner.run(request)

    assert calls == [None, "thread-original"]
    assert turn_submissions == 1
    manifest_path = (
        tmp_path
        / "workspaces"
        / "codex"
        / "unit-research-task"
        / "planner"
        / ".arche_planner_session.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["session_id"] == "thread-original"
    assert manifest["turn_count"] == 1


def test_harness_config_validates_reasoning_effort(monkeypatch) -> None:
    monkeypatch.setenv("ARCHE_CHEM_REASONING_EFFORT", "very-high")
    with pytest.raises(ValueError, match="ARCHE_CHEM_REASONING_EFFORT"):
        HarnessConfig.from_env()


def test_harness_config_uses_planner_specific_codex_model(monkeypatch) -> None:
    monkeypatch.setenv("ARCHE_PLANNER_DRIVER", "codex")
    monkeypatch.setenv("ARCHE_CODEX_API_KEY", "codex-secret")
    monkeypatch.setenv("ARCHE_CODEX_BASE_URL", "https://codex.example/v1")
    monkeypatch.setenv("ARCHE_CODEX_MODEL_NAME", "planner-tool-model")

    config = HarnessConfig.from_env()
    assert config.model == "planner-tool-model"
    assert config.base_url == "https://codex.example/v1"

    monkeypatch.delenv("ARCHE_CODEX_MODEL_NAME")
    monkeypatch.setenv("ARCHE_CHEM_MODEL_NAME", "react-model")
    with pytest.raises(ValueError, match="ARCHE_CODEX_MODEL_NAME"):
        HarnessConfig.from_env()

    monkeypatch.setenv("ARCHE_PLANNER_DRIVER", "react")
    assert HarnessConfig.from_env().model == "react-model"


def test_harness_config_requires_all_independent_codex_provider_values(monkeypatch) -> None:
    monkeypatch.setenv("ARCHE_PLANNER_DRIVER", "codex")
    monkeypatch.setenv("ARCHE_CHEM_API_KEY", "chem-secret")
    monkeypatch.setenv("ARCHE_CHEM_BASE_URL", "https://chem.example/v1")
    monkeypatch.setenv("ARCHE_CHEM_MODEL_NAME", "chem-model")
    for name in ("ARCHE_CODEX_API_KEY", "ARCHE_CODEX_BASE_URL", "ARCHE_CODEX_MODEL_NAME"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ValueError) as error:
        HarnessConfig.from_env()

    message = str(error.value)
    assert "ARCHE_CODEX_API_KEY" in message
    assert "ARCHE_CODEX_BASE_URL" in message
    assert "ARCHE_CODEX_MODEL_NAME" in message


@pytest.mark.parametrize(
    ("config", "environment", "missing_name"),
    [
        (
            {"base_url": None, "model": "codex-model"},
            {"ARCHE_CODEX_API_KEY": "codex-secret"},
            "ARCHE_CODEX_BASE_URL",
        ),
        (
            {"base_url": "https://codex.example/v1", "model": "codex-model"},
            {},
            "ARCHE_CODEX_API_KEY",
        ),
        (
            {"base_url": "https://codex.example/v1", "model": None},
            {"ARCHE_CODEX_API_KEY": "codex-secret"},
            "ARCHE_CODEX_MODEL_NAME",
        ),
    ],
)
def test_codex_runner_rejects_missing_provider_value_before_cli(
    tmp_path: Path,
    monkeypatch,
    config: dict[str, str | None],
    environment: dict[str, str],
    missing_name: str,
) -> None:
    for name in ("ARCHE_CODEX_API_KEY", "ARCHE_CODEX_BASE_URL", "ARCHE_CODEX_MODEL_NAME"):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    runner_config = HarnessConfig(
        kind="codex",
        workspace_root=tmp_path / "workspaces",
        trace_root=tmp_path / "traces",
        model=config["model"],
        base_url=config["base_url"],
    )
    with pytest.raises(HarnessError, match=missing_name):
        CodexRunner(runner_config).run(_request(tmp_path))


def test_compiled_request_rejects_non_object_messages(tmp_path: Path) -> None:
    with pytest.raises(HarnessError, match="only objects"):
        HarnessRequest.from_compiled(
            {"model": "fake-model", "messages": [{"role": "user"}, "bad"]},
            workspace=tmp_path,
        )


def test_harness_prompt_includes_supplied_response_schema(tmp_path: Path) -> None:
    request = HarnessRequest.from_compiled(
        {"model": "fake-model", "messages": [{"role": "user", "content": "go"}]},
        workspace=tmp_path,
        response_schema={
            "type": "object",
            "required": ["decision"],
            "properties": {"decision": {"type": "string"}},
        },
    )

    prompt = request.prompt_text()
    assert "Required response JSON schema:" in prompt
    assert '"decision"' in prompt


def test_dispatcher_infers_role_from_compiled_catalog_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ARCHE_PLANNER_DRIVER", "react")
    calls = []

    def fallback(request, **kwargs):
        calls.append((request, kwargs))
        return {"choices": [{"finish_reason": "stop", "message": {"content": "ok"}}]}

    dispatcher = HarnessDispatcher(
        HarnessConfig(kind="react", workspace_root=tmp_path / "w"),
        react_fallback=fallback,
    )
    result = dispatcher.run_compiled(
        {
            "catalog_key": "workflow_generation",
            "model": "model",
            "messages": [{"role": "user", "content": "go"}],
        }
    )
    assert result["choices"][0]["message"]["content"] == "ok"
    assert calls[0][0]["catalog_key"] == "workflow_generation"


def test_dispatcher_rejects_legacy_run_identity_before_routing(tmp_path: Path) -> None:
    """Dispatcher 不把旧 run_id 伪装成当前 research_task 身份。"""

    dispatcher = HarnessDispatcher(
        HarnessConfig(kind="react", workspace_root=tmp_path / "w"),
        react_fallback=lambda request, **kwargs: {},
    )
    with pytest.raises(HarnessError, match="research_task_id"):
        dispatcher.run_compiled(
            {
                "catalog_key": "router",
                "model": "model",
                "messages": [{"role": "user", "content": "go"}],
                "run_id": "legacy-run",
            }
        )


def test_dispatcher_maps_catalog_operations_to_arche_roles(tmp_path: Path) -> None:
    captured = []

    class FakeRunner:
        def run(self, request):
            captured.append((request.role, request.stage, request.workspace))
            return HarnessResponse(content="ok", model=request.model)

    dispatcher = HarnessDispatcher(
        HarnessConfig(
            kind="codex",
            workspace_root=tmp_path / "workspaces",
            trace_root=tmp_path / "traces",
        ),
        react_fallback=lambda request, **kwargs: {},
    )
    dispatcher._runner = FakeRunner()

    for catalog_key in (
        "workflow_generation",
        "hypothesis_generation.rank",
        "retrieval.keyword",
        "router",
    ):
        dispatcher.run_compiled(
            {
                "catalog_key": catalog_key,
                "model": "fake-model",
                "messages": [{"role": "user", "content": "go"}],
                "research_task_id": "shared-research-task",
            }
        )

    assert [item[:2] for item in captured] == [("planner", "workflow_generation")]
    assert len({item[2] for item in captured}) == 1


def test_non_planner_roles_fall_back_to_react_even_when_harness_selected(
    tmp_path: Path,
) -> None:
    calls = []

    def fallback(request, **kwargs):
        calls.append((request, kwargs))
        return {
            "choices": [
                {"finish_reason": "stop", "message": {"content": "legacy"}}
            ]
        }

    dispatcher = HarnessDispatcher(
        HarnessConfig(
            kind="codex",
            workspace_root=tmp_path / "workspaces",
            trace_root=tmp_path / "traces",
        ),
        react_fallback=fallback,
    )
    result = dispatcher.run_compiled(
        {
            "catalog_key": "hypothesis_generation.rank",
            "model": "fake-model",
            "messages": [{"role": "user", "content": "go"}],
            "research_task_id": "shared-research-task",
        }
    )

    assert result["choices"][0]["message"]["content"] == "legacy"
    assert calls and not (tmp_path / "workspaces" / "codex").exists()


def test_planner_scope_reuses_one_workspace_for_multiple_stages(
    tmp_path: Path,
) -> None:
    captured = []

    class FakeRunner:
        def run(self, request):
            captured.append(request)
            return HarnessResponse(content="ok", model=request.model)

    dispatcher = HarnessDispatcher(
        HarnessConfig(
            kind="codex",
            workspace_root=tmp_path / "workspaces",
            trace_root=tmp_path / "traces",
        ),
        react_fallback=lambda request, **kwargs: {},
    )
    dispatcher._runner = FakeRunner()
    for stage in ("workflow_generation", "workflow_optimization", "workflow_revision"):
        dispatcher.run_compiled(
            {
                "catalog_key": stage,
                "model": "fake-model",
                "messages": [{"role": "user", "content": "go"}],
                "research_task_id": "case1-research-task",
            }
        )

    assert len({item.workspace for item in captured}) == 1
    assert captured[0].workspace == (
        tmp_path / "workspaces" / "codex" / "case1-research-task" / "planner"
    )


def test_bound_planner_context_reuses_task_workspace_and_combination_session(
    tmp_path: Path,
) -> None:
    captured = []

    class FakeRunner:
        def run(self, request):
            captured.append(request)
            return HarnessResponse(content="ok", model=request.model)

    dispatcher = HarnessDispatcher(
        HarnessConfig(
            kind="codex",
            workspace_root=tmp_path / "workspaces",
            trace_root=tmp_path / "traces",
        ),
        react_fallback=lambda request, **kwargs: {},
    )
    dispatcher._runner = FakeRunner()
    with bind_harness_call_context(
        task_scope_id="task-123",
        operation="workflow_generation",
        operation_instance_id="op-1",
        display_context={"batch_index": 1, "combination_index": 2},
    ):
        dispatcher.run_compiled(
            {
                "catalog_key": "workflow_generation",
                "model": "fake-model",
                "messages": [{"role": "user", "content": "go"}],
            }
        )
    with bind_harness_call_context(
        task_scope_id="task-123",
        operation="workflow_revision",
        operation_instance_id="op-2",
        display_context={"batch_index": 1, "combination_index": 3},
    ):
        dispatcher.run_compiled(
            {
                "catalog_key": "workflow_revision",
                "model": "fake-model",
                "messages": [{"role": "user", "content": "go"}],
            }
        )

    assert len({item.workspace for item in captured}) == 1
    assert captured[0].metadata["harness_session_id"] == captured[1].metadata[
        "harness_session_id"
    ]
    assert captured[0].metadata["operation_instance_id"] == "op-1"
    assert captured[1].metadata["operation_instance_id"] == "op-2"


def test_env_config_resolves_cli_roots_before_workspace_cwd_changes(
    tmp_path: Path, monkeypatch
) -> None:
    """运行器切换到 role workspace 后，CLI 输出路径仍保持有效。"""

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ARCHE_PLANNER_DRIVER", "codex")
    monkeypatch.setenv("ARCHE_STAGE_HARNESS_WORKSPACE_ROOT", ".runtime/workspaces")
    monkeypatch.setenv("ARCHE_STAGE_HARNESS_TRACE_ROOT", ".runtime/traces")
    monkeypatch.setenv("ARCHE_CODEX_API_KEY", "unit-test-secret")
    monkeypatch.setenv("ARCHE_CODEX_BASE_URL", "https://codex.example/v1")
    monkeypatch.setenv("ARCHE_CODEX_MODEL_NAME", "codex-model")
    config = HarnessConfig.from_env()

    assert config.workspace_root == (tmp_path / ".runtime/workspaces").resolve()
    assert config.trace_root == (tmp_path / ".runtime/traces").resolve()
    role_workspace = config.workspace_for(role="planner", research_task_id="research-task-1")
    assert role_workspace.is_absolute()


def test_env_config_defaults_to_ignored_local_data_roots(
    tmp_path: Path, monkeypatch
) -> None:
    """未配置 Harness 根目录时，活动产物仍写入被忽略的 local_data。"""

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ARCHE_STAGE_HARNESS_WORKSPACE_ROOT", raising=False)
    monkeypatch.delenv("ARCHE_STAGE_HARNESS_TRACE_ROOT", raising=False)

    config = HarnessConfig.from_env()

    assert config.workspace_root == (
        tmp_path / "local_data/runtime/harness_workspaces"
    ).resolve()
    assert config.trace_root == (
        tmp_path / "local_data/runtime/harness_traces"
    ).resolve()


def test_env_config_resolves_relative_codex_binary_before_workspace_cwd_changes(
    tmp_path: Path, monkeypatch
) -> None:
    """仓库相对的 Codex 路径在切换到 Planner 工作区后仍保持绝对路径。"""

    relative_bin = Path("local_data/tooling/codex")
    (tmp_path / relative_bin).parent.mkdir(parents=True)
    _fake_cli(
        tmp_path / relative_bin,
        '{"type":"thread.started","thread_id":"session-unit"}',
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ARCHE_PLANNER_DRIVER", "codex")
    monkeypatch.setenv("ARCHE_CODEX_BIN", str(relative_bin))
    monkeypatch.setenv("ARCHE_CODEX_API_KEY", "unit-test-secret")
    monkeypatch.setenv("ARCHE_CODEX_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("ARCHE_CODEX_MODEL_NAME", "codex-model")
    monkeypatch.setenv("ARCHE_PLANNER_MCP_ENABLED", "0")
    def fake_sdk_turn(**kwargs):
        kwargs["on_thread_ready"]("session-unit")
        _write_fake_completed_receipt(
            kwargs,
            thread_id="session-unit",
            turn_id="turn-relative",
        )
        return CodexSdkRunResult(
            HarnessResponse(content='{"ok":true}', model=kwargs["request"].model),
            "session-unit",
            "turn-relative",
        )

    monkeypatch.setattr(runners_module, "run_codex_sdk_turn", fake_sdk_turn)

    response = CodexRunner(
        HarnessConfig.from_env(allow_test_codex_binary=True)
    ).run(_request(tmp_path))

    assert response.content == '{"ok":true}'


def test_env_config_requires_repository_wrapper_unless_test_seam_is_explicit(
    tmp_path: Path, monkeypatch
) -> None:
    """生产环境拒绝任意绝对 Codex 文件，测试替身必须显式走参数 seam。"""

    fake = _fake_cli(tmp_path / "arbitrary-codex", "")
    monkeypatch.setenv("ARCHE_PLANNER_DRIVER", "codex")
    monkeypatch.setenv("ARCHE_CODEX_BIN", str(fake))
    monkeypatch.setenv("ARCHE_CODEX_API_KEY", "unit-test-secret")
    monkeypatch.setenv("ARCHE_CODEX_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("ARCHE_CODEX_MODEL_NAME", "codex-model")

    with pytest.raises(ValueError, match="固定.*wrapper"):
        HarnessConfig.from_env()
    assert HarnessConfig.from_env(allow_test_codex_binary=True).codex_bin == str(fake.resolve())


def test_dispatcher_forwards_optional_tools_to_non_react_runner(tmp_path: Path) -> None:
    """未来的原生工具 stage 必须看到同一份显式工具 schema。"""

    class FakeRunner:
        def run(self, request):
            assert request.tools == ({"type": "function", "function": {"name": "probe"}},)
            return HarnessResponse(content="ok", model=request.model)

    dispatcher = HarnessDispatcher(
        HarnessConfig(
            kind="codex",
            workspace_root=tmp_path / "workspaces",
            trace_root=tmp_path / "traces",
        ),
        react_fallback=lambda request, **kwargs: {},
    )
    dispatcher._runner = FakeRunner()
    result = dispatcher.run_compiled(
        {
            "catalog_key": "planner.workflow",
            "model": "fake-model",
            "messages": [{"role": "user", "content": "go"}],
            "tools": [{"type": "function", "function": {"name": "probe"}}],
        }
    )
    assert result["choices"][0]["message"]["content"] == "ok"


def test_compiled_model_entrypoint_restores_non_react_harness_dispatch(monkeypatch) -> None:
    """远端 llm_api 更新不能绕过可选 Harness 边界。"""

    from chemistry_multiagent.harness import dispatcher as dispatcher_module
    from chemistry_multiagent.utils import llm_api
    from chemistry_multiagent.utils.model_catalog import compile_model_request

    calls = []

    class FakeDispatcher:
        kind = "codex"

        def run_compiled(self, request, *, timeout):
            calls.append((request, timeout))
            return {
                "choices": [
                    {"finish_reason": "stop", "message": {"content": '{"ok":true}'}}
                ]
            }

    monkeypatch.setattr(
        dispatcher_module,
        "get_harness_dispatcher",
        lambda **_kwargs: FakeDispatcher(),
    )
    request = compile_model_request(
        operation="hypothesis_generation",
        stage="hypothesis",
        messages=[{"role": "user", "content": "return JSON"}],
        response_contract={
            "validation_kind": "json_schema",
            "schema": {"type": "object"},
            "domain_validator_id": None,
        },
    )

    result = llm_api.call_compiled_model_request(request, timeout=17)

    assert result["choices"][0]["message"]["content"] == '{"ok":true}'
    assert calls == [(request, 17.0)]
    assert callable(llm_api._call_compiled_model_request_react)


def test_non_react_harness_rejects_malformed_compiled_request(monkeypatch) -> None:
    from chemistry_multiagent.harness import dispatcher as dispatcher_module
    from chemistry_multiagent.utils import llm_api

    class FakeDispatcher:
        kind = "codex"

        def run_compiled(self, request, *, timeout):  # pragma: no cover
            raise AssertionError("malformed request must not reach Harness")

    monkeypatch.setattr(
        dispatcher_module,
        "get_harness_dispatcher",
        lambda **_kwargs: FakeDispatcher(),
    )
    with pytest.raises(TypeError, match="compiled model request"):
        llm_api.call_compiled_model_request({"catalog_key": "router"})


def test_react_runner_normalizes_openai_envelope_and_raw_fallback(tmp_path: Path) -> None:
    """Legacy React runner 同时保留 OpenAI 信封和原始文本返回语义。"""

    request = HarnessRequest(
        messages=({"role": "user", "content": "probe"},),
        model="react-model",
        role="generic",
        stage="model",
        workspace=tmp_path,
    )
    envelope = ReactRunner(
        lambda value: {
            "model": "provider-model",
            "choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 3},
        }
    ).run(request)
    assert envelope.content == "answer"
    assert envelope.model == "provider-model"
    assert envelope.usage == {"total_tokens": 3}

    raw = ReactRunner(lambda value: "raw-answer").run(request)
    assert raw.content == "raw-answer"
    assert raw.model == "react-model"


def test_codex_runner_ephemeral_request_keeps_non_session_execution_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """非持久角色继续执行一次性 Codex turn，不创建或回退 legacy exec。"""

    def fake_sdk_turn(**kwargs):
        return CodexSdkRunResult(
            HarnessResponse(content='{"ok":true}', model=kwargs["request"].model),
            None,
            "turn-ephemeral",
        )

    monkeypatch.setattr(runners_module, "run_codex_sdk_turn", fake_sdk_turn)
    binary = _fake_cli(tmp_path / "codex-ephemeral", "")
    config = HarnessConfig(
        kind="codex",
        workspace_root=tmp_path / "workspaces",
        trace_root=tmp_path / "traces",
        codex_bin=str(binary),
        base_url="https://example.invalid/v1",
        model="codex-model",
    )
    monkeypatch.setenv("ARCHE_CODEX_API_KEY", "ephemeral-secret")
    request = HarnessRequest(
        messages=({"role": "user", "content": "probe"},),
        model="request-model",
        role="generic",
        stage="one_shot",
        workspace=tmp_path / "request-workspace",
        research_task_id="ephemeral-research-task",
    )

    response = CodexRunner(config).run(request)

    assert response.content == '{"ok":true}'
    assert not (tmp_path / "request-workspace" / "manifest.json").exists()
