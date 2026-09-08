"""覆盖 vendored Codex SDK 的同步、异步控制流和环境路径分支。"""

from __future__ import annotations

import asyncio
import io
import threading
from pathlib import PosixPath
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from openai_codex import api
from openai_codex import client as client_module
from openai_codex.client import CodexClient
from openai_codex.errors import (
    CodexError,
    InvalidRequestError,
    StderrIndicator,
    TransportClosedError,
    TransportPhase,
)
from openai_codex.generated.v2_all import (
    AccountLoginCompletedNotification,
    AgentMessageDeltaNotification,
    IdleThreadStatus,
    ThreadStatus,
    Turn,
    TurnCompletedNotification,
    TurnStatus,
)
from openai_codex.models import Notification, UnknownNotification


def _turn(turn_id: str, status: TurnStatus = TurnStatus.completed) -> Turn:
    """构造控制流测试所需的最小完整 turn。"""

    return Turn(id=turn_id, items=[], status=status)


def _completion(turn_id: str) -> Notification:
    """构造带有目标 turn 标识的完成通知。"""

    turn = _turn(turn_id)
    return Notification(
        method="turn/completed",
        payload=TurnCompletedNotification(threadId="thread-1", turn=turn),
    )


def _idle_thread(*, ephemeral: bool = False, path: str | None = "/thread.json") -> object:
    """构造 ``_start_goal_operation`` 读取的线程字段。"""

    return SimpleNamespace(
        status=ThreadStatus(root=IdleThreadStatus(type="idle")),
        ephemeral=ephemeral,
        path=path,
    )


def test_interrupt_goal_operation_handles_rollover_and_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """覆盖无活动 turn、成功中断、轮换重试和各类可忽略异常。"""

    client = CodexClient.__new__(CodexClient)
    calls: list[str] = []

    class _State:
        def __init__(self, first: str | None, current: str | None = None) -> None:
            self.thread_id = "thread-1"
            self.first = first
            self.current = current

        def turn_for_interrupt(self) -> str | None:
            return self.first

        def current_turn(self) -> str | None:
            return self.current

    client._interrupt_goal_operation(_State(None))

    def interrupt(thread_id: str, turn_id: str) -> object:
        calls.append(f"{thread_id}:{turn_id}")
        return SimpleNamespace()

    monkeypatch.setattr(client, "turn_interrupt", interrupt)
    client._interrupt_goal_operation(_State("turn-1"))

    def generic_failure(_thread_id: str, _turn_id: str) -> object:
        raise RuntimeError("transport closed")

    monkeypatch.setattr(client, "turn_interrupt", generic_failure)
    client._interrupt_goal_operation(_State("turn-generic"))

    def unrelated_invalid(_thread_id: str, _turn_id: str) -> object:
        raise InvalidRequestError(-32600, "turn is already complete")

    monkeypatch.setattr(client, "turn_interrupt", unrelated_invalid)
    client._interrupt_goal_operation(_State("turn-invalid"))

    rollover_calls: list[str] = []

    def rollover_interrupt(_thread_id: str, turn_id: str) -> object:
        rollover_calls.append(turn_id)
        if len(rollover_calls) == 1:
            raise InvalidRequestError(
                -32600,
                "expected active turn id `old` but found `new`",
            )
        return SimpleNamespace()

    monkeypatch.setattr(client, "turn_interrupt", rollover_interrupt)
    client._interrupt_goal_operation(_State("old", "state-new"))

    same_turn_calls: list[str] = []

    def same_turn_interrupt(_thread_id: str, turn_id: str) -> object:
        same_turn_calls.append(turn_id)
        raise InvalidRequestError(-32600, "expected active turn id but found `old`")

    monkeypatch.setattr(client, "turn_interrupt", same_turn_interrupt)
    client._interrupt_goal_operation(_State("old", "old"))

    retry_failure_calls: list[str] = []

    def retry_failure(_thread_id: str, turn_id: str) -> object:
        retry_failure_calls.append(turn_id)
        raise InvalidRequestError(
            -32600,
            "expected active turn id `old` but found `new`",
        ) if len(retry_failure_calls) == 1 else RuntimeError("closed")

    monkeypatch.setattr(client, "turn_interrupt", retry_failure)
    client._interrupt_goal_operation(_State("old", "state-new"))

    assert calls == ["thread-1:turn-1"]
    assert rollover_calls == ["old", "new"]
    assert same_turn_calls == ["old"]
    assert retry_failure_calls == ["old", "new"]


def test_path_helpers_preserve_platform_specific_path_semantics() -> None:
    """覆盖空路径、重复项、Windows 大小写键和 Linux 默认键。"""

    env = {"PATH": "/usr/bin:/opt/codex:/usr/local/bin"}
    original = dict(env)
    client_module._prepend_path_dirs(env, ())
    assert env == original

    client_module._prepend_path_dirs(
        env,
        (client_module.Path("/opt/codex"), client_module.Path("/runtime")),
    )
    assert env["PATH"].split(client_module.os.pathsep) == [
        "/opt/codex",
        "/runtime",
        "/usr/bin",
        "/usr/local/bin",
    ]
    assert client_module._path_env_key(env) == "PATH"

    with patch.object(client_module.os, "name", "nt"):
        path_key_with_preferred_case = client_module._path_env_key({"Path": "x", "PATH": "y"})
        path_key_with_upper_case = client_module._path_env_key({"PATH": "x", "PathExtra": "y"})
        path_key_without_match = client_module._path_env_key({"PathExtra": "y"})
        windows_env = {"PATH": "old", "Path": "preferred", "other": "value"}
        client_module._prepend_path_dirs(windows_env, (PosixPath("runtime"),))

    assert path_key_with_preferred_case == "Path"
    assert path_key_with_upper_case == "PATH"
    assert path_key_without_match == "PATH"
    assert windows_env["Path"].split(client_module.os.pathsep)[0] == "runtime"
    assert "PATH" not in windows_env
    assert windows_env["other"] == "value"


class _ExitedProcess:
    """为 stdout EOF 诊断提供不启动真实进程的最小 Popen 替身。"""

    def __init__(self, *, stderr: str, returncode: int) -> None:
        self.stdout = io.StringIO("")
        self.stderr = io.StringIO(stderr)
        self._returncode = returncode

    def poll(self) -> int:
        return self._returncode

    def wait(self, *, timeout: float) -> int:
        return self._returncode


def test_transport_eof_diagnostic_drains_stderr_without_retaining_secret() -> None:
    """stdout EOF 等待 stderr drain 后只产生固定诊断，不泄露 secret。"""

    client = CodexClient.__new__(CodexClient)
    client._stderr_indicator = StderrIndicator.NONE
    client._stderr_indicator_lock = threading.Lock()
    client._stderr_drain_done = threading.Event()
    client._stderr_drain_done.clear()
    client._stderr_thread = None
    client._transport_phase = TransportPhase.INITIALIZE
    process = _ExitedProcess(
        stderr="fatal: invalid api key=synthetic-secret\n",
        returncode=-9,
    )
    client._proc = process
    client._start_stderr_drain_thread()

    with pytest.raises(TransportClosedError) as failure:
        client._read_message()

    diagnostic = failure.value.diagnostic
    assert diagnostic is not None
    assert diagnostic.phase is TransportPhase.INITIALIZE
    assert diagnostic.stdout_eof is True
    assert diagnostic.returncode == -9
    assert diagnostic.termination_signal == 9
    assert diagnostic.stderr_indicator is StderrIndicator.AUTHENTICATION
    assert "synthetic-secret" not in str(failure.value)
    assert "synthetic-secret" not in repr(diagnostic)
    assert not hasattr(client, "_stderr_lines")


def test_client_start_is_idempotent_when_process_already_exists() -> None:
    """已有 app-server 时重复 start 不重启进程或清空诊断状态。"""

    client = CodexClient()
    existing_process = object()
    client._proc = existing_process

    client.start()

    assert client._proc is existing_process


def test_client_close_is_noop_without_process() -> None:
    """尚未启动时 close 可安全调用，供构造失败清理路径复用。"""

    client = CodexClient()
    client.close()
    assert client._proc is None


def test_transport_diagnostic_without_process_uses_explicit_phase() -> None:
    """没有可读进程时诊断仍返回固定 phase，不猜测退出码。"""

    client = CodexClient()
    diagnostic = client._transport_diagnostic(
        process=None,
        stdout_eof=False,
        phase=TransportPhase.PROCESS_NOT_RUNNING,
    )

    assert diagnostic.phase is TransportPhase.PROCESS_NOT_RUNNING
    assert diagnostic.stdout_eof is False
    assert diagnostic.returncode is None
    assert diagnostic.termination_signal is None
    assert diagnostic.stderr_indicator is StderrIndicator.NONE


def _patch_goal_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    thread: object,
    wait_result: str | None = "physical-1",
    clear_error: BaseException | None = None,
    set_error: BaseException | None = None,
) -> tuple[CodexClient, object, list[str]]:
    """为目标启动路径安装最小的可观察 fake 依赖。"""

    client = CodexClient.__new__(CodexClient)
    events: list[str] = []

    class _State:
        def activate_turn_routing(self) -> None:
            events.append("activate")

        def wait_for_start(self, _timeout: float) -> str | None:
            events.append("wait")
            return wait_result

        def finish(self) -> None:
            events.append("finish")

    state = _State()
    monkeypatch.setattr(client, "thread_read", lambda _thread_id: SimpleNamespace(thread=thread))
    monkeypatch.setattr(client, "reserve_goal_operation", lambda _thread_id: state)

    def clear(_thread_id: str) -> None:
        events.append("clear")
        if clear_error is not None:
            raise clear_error

    def set_goal(_thread_id: str, **_kwargs: object) -> object:
        events.append("set")
        if set_error is not None:
            raise set_error
        return SimpleNamespace()

    monkeypatch.setattr(client, "thread_goal_clear", clear)
    monkeypatch.setattr(client, "thread_goal_set", set_goal)
    monkeypatch.setattr(client, "cancel_goal_operation", lambda _state: events.append("cancel"))
    monkeypatch.setattr(
        client,
        "unregister_goal_operation",
        lambda _state: events.append("unregister"),
    )
    return client, state, events


def test_start_goal_operation_covers_preconditions_success_and_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证目标启动的线程前置条件、成功返回和失败清理。"""

    idle = _idle_thread()
    client, _state, events = _patch_goal_client(monkeypatch, thread=idle)
    state, turn_id = client._start_goal_operation("thread-1", "investigate")
    assert turn_id == "physical-1"
    assert state is not None
    assert events == ["clear", "activate", "set", "wait"]

    for thread in (
        SimpleNamespace(
            status=SimpleNamespace(root=SimpleNamespace(type="active")),
            ephemeral=False,
            path="x",
        ),
        _idle_thread(ephemeral=True),
        _idle_thread(path=None),
    ):
        client, _state, events = _patch_goal_client(monkeypatch, thread=thread)
        with pytest.raises(InvalidRequestError):
            client._start_goal_operation("thread-1", "investigate")
        assert events == []

    client, _state, events = _patch_goal_client(monkeypatch, thread=idle, wait_result=None)
    with pytest.raises(CodexError, match="timed out"):
        client._start_goal_operation("thread-1", "investigate")
    assert events == ["clear", "activate", "set", "wait", "cancel", "finish", "unregister"]

    client, _state, events = _patch_goal_client(
        monkeypatch,
        thread=idle,
        clear_error=InvalidRequestError(-32600, "clear rejected"),
    )
    with pytest.raises(InvalidRequestError, match="clear rejected"):
        client._start_goal_operation("thread-1", "investigate")
    assert events == ["clear", "finish", "unregister"]

    client, _state, events = _patch_goal_client(
        monkeypatch,
        thread=idle,
        set_error=RuntimeError("goal update failed"),
    )
    with pytest.raises(RuntimeError, match="goal update failed"):
        client._start_goal_operation("thread-1", "investigate")
    assert events == ["clear", "activate", "set", "cancel", "finish", "unregister"]


def test_stream_and_wait_helpers_filter_notifications_and_unregister(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """覆盖同步文本流和 turn/login 完成等待器的过滤循环。"""

    client = CodexClient.__new__(CodexClient)
    turn_events = [
        Notification("other", UnknownNotification(params={})),
        Notification(
            "item/agentMessage/delta",
            AgentMessageDeltaNotification(
                delta="ignored",
                itemId="item-1",
                threadId="thread-1",
                turnId="other-turn",
            ),
        ),
        Notification(
            "item/agentMessage/delta",
            AgentMessageDeltaNotification(
                delta="answer",
                itemId="item-2",
                threadId="thread-1",
                turnId="turn-1",
            ),
        ),
        _completion("turn-1"),
    ]
    registrations: list[tuple[str, str]] = []
    monkeypatch.setattr(
        client,
        "turn_start",
        lambda _thread_id, _text, params=None: SimpleNamespace(
            turn=_turn("turn-1", TurnStatus.in_progress)
        ),
    )
    monkeypatch.setattr(
        client,
        "register_turn_notifications",
        lambda turn_id: registrations.append(("register", turn_id)),
    )
    monkeypatch.setattr(
        client,
        "unregister_turn_notifications",
        lambda turn_id: registrations.append(("unregister", turn_id)),
    )
    monkeypatch.setattr(
        client,
        "next_turn_notification",
        lambda _turn_id: turn_events.pop(0),
    )

    deltas = list(client.stream_text("thread-1", "hello"))
    assert [delta.delta for delta in deltas] == ["answer"]
    assert registrations == [("register", "turn-1"), ("unregister", "turn-1")]

    registrations.clear()
    handle_events = [Notification("other", UnknownNotification(params={})), _completion("turn-h")]
    monkeypatch.setattr(client, "next_turn_notification", lambda _turn_id: handle_events.pop(0))
    handle = api.TurnHandle(client, "thread-1", "turn-h")
    assert [event.method for event in handle.stream()] == ["other", "turn/completed"]
    assert registrations == [("register", "turn-h"), ("unregister", "turn-h")]

    wait_events = [Notification("other", UnknownNotification(params={})), _completion("turn-2")]
    monkeypatch.setattr(client, "next_turn_notification", lambda _turn_id: wait_events.pop(0))
    assert client.wait_for_turn_completed("turn-2").turn.id == "turn-2"

    login_events = [
        Notification("other", UnknownNotification(params={})),
        Notification(
            "account/login/completed",
            AccountLoginCompletedNotification(loginId="login-1", success=True),
        ),
    ]
    login_registrations: list[tuple[str, str]] = []
    monkeypatch.setattr(
        client,
        "register_login_notifications",
        lambda login_id: login_registrations.append(("register", login_id)),
    )
    monkeypatch.setattr(
        client,
        "unregister_login_notifications",
        lambda login_id: login_registrations.append(("unregister", login_id)),
    )
    monkeypatch.setattr(
        client,
        "next_login_notification",
        lambda _login_id: login_events.pop(0),
    )
    assert client.wait_for_login_completed("login-1").login_id == "login-1"
    assert login_registrations == [("register", "login-1"), ("unregister", "login-1")]


class _AsyncNotificationClient:
    """为异步 turn 流提供可观察的队列。"""

    def __init__(self, events: list[Notification]) -> None:
        self.events = list(events)
        self.registrations: list[tuple[str, str]] = []

    def register_turn_notifications(self, turn_id: str) -> None:
        self.registrations.append(("register", turn_id))

    def unregister_turn_notifications(self, turn_id: str) -> None:
        self.registrations.append(("unregister", turn_id))

    async def next_turn_notification(self, _turn_id: str) -> Notification:
        return self.events.pop(0)


class _AsyncCodex:
    """提供 AsyncTurnHandle 所需的初始化和客户端属性。"""

    def __init__(self, client: _AsyncNotificationClient) -> None:
        self._client = client
        self.initialized = 0

    async def _ensure_initialized(self) -> None:
        self.initialized += 1


def test_async_turn_stream_filters_events_and_closes_route() -> None:
    """验证异步流会初始化客户端、过滤通知并在结束时注销路由。"""

    async def collect() -> tuple[list[Notification], _AsyncCodex]:
        fake_client = _AsyncNotificationClient(
            [
                Notification("other", UnknownNotification(params={})),
                _completion("async-turn"),
            ]
        )
        codex = _AsyncCodex(fake_client)
        handle = api.AsyncTurnHandle(codex, "thread-1", "async-turn")
        events = [event async for event in handle.stream()]
        return events, codex

    events, codex = asyncio.run(collect())
    assert [event.method for event in events] == ["other", "turn/completed"]
    assert codex.initialized == 1
    assert codex._client.registrations == [
        ("register", "async-turn"),
        ("unregister", "async-turn"),
    ]
