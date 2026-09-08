"""覆盖 vendored Codex 0.149 协议辅助模块的分支和失败路径。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator

import pytest

from openai_codex import (
    ImageInput,
    LocalImageInput,
    MentionInput,
    SkillInput,
    TextInput,
)
from openai_codex._initialize_metadata import (
    _split_user_agent,
    validate_initialize_metadata,
)
from openai_codex._inputs import _to_wire_input, _to_wire_item
from openai_codex._message_router import MessageRouter
from openai_codex._run import (
    _collect_async_turn_result,
    _collect_turn_result,
    _final_assistant_response_from_items,
)
from openai_codex.errors import (
    CodexRpcError,
    InternalRpcError,
    InvalidParamsError,
    InvalidRequestError,
    JsonRpcError,
    MethodNotFoundError,
    ParseError,
    RetryLimitExceededError,
    ServerBusyError,
    _is_server_overloaded,
    is_retryable_error,
    map_jsonrpc_error,
)
from openai_codex.generated.v2_all import (
    AccountLoginCompletedNotification,
    AgentMessageThreadItem,
    ItemCompletedNotification,
    MessagePhase,
    ThreadGoal,
    ThreadGoalStatus,
    ThreadGoalUpdatedNotification,
    ThreadItem,
    ThreadTokenUsage,
    ThreadTokenUsageUpdatedNotification,
    TokenUsageBreakdown,
    Turn,
    TurnCompletedNotification,
    TurnError,
    TurnStartedNotification,
    TurnStatus,
)
from openai_codex.models import (
    InitializeResponse,
    Notification,
    ServerInfo,
    UnknownNotification,
)
from openai_codex.retry import retry_on_overload


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, False),
        ("server_overloaded", True),
        ("SERVER_OVERLOADED", True),
        ("other", False),
        ({"codex_error_info": "server_overloaded"}, True),
        ({"codexErrorInfo": "other"}, False),
        ({"errorInfo": {"kind": "server_overloaded"}}, True),
        ({"errorInfo": {"kind": "other"}}, False),
        ({"nested": {"deeper": "server_overloaded"}}, True),
        ({"nested": {"deeper": 4}}, False),
        ({"nested": ["other", {"deep": "server_overloaded"}]}, True),
        ({"nested": ["other", {"deep": 4}]}, False),
        ([], False),
        (["server_overloaded"], True),
        (["other", None], False),
        (17, False),
    ],
)
def test_server_overload_detection_handles_wire_shapes(value: object, expected: bool) -> None:
    """确认服务端过载标记可从官方错误数据的常见嵌套形状中恢复。"""

    assert _is_server_overloaded(value) is expected


@pytest.mark.parametrize(
    ("code", "message", "data", "error_type"),
    [
        (-32700, "bad json", None, ParseError),
        (-32600, "bad request", None, InvalidRequestError),
        (-32601, "missing", None, MethodNotFoundError),
        (-32602, "bad params", None, InvalidParamsError),
        (-32603, "internal", None, InternalRpcError),
        (-32000, "retry limit reached", None, RetryLimitExceededError),
        (-32001, "busy", "server_overloaded", ServerBusyError),
        (
            -32002,
            "retry limit reached",
            {"kind": "not-overloaded"},
            RetryLimitExceededError,
        ),
        (-32003, "ordinary", {"kind": "not-overloaded"}, CodexRpcError),
        (-31999, "outside reserved range", None, JsonRpcError),
    ],
)
def test_map_jsonrpc_error_preserves_protocol_error_categories(
    code: int,
    message: str,
    data: object,
    error_type: type[BaseException],
) -> None:
    """确认 JSON-RPC 错误映射覆盖标准码、重试码和未知码。"""

    mapped = map_jsonrpc_error(code, message, data)
    assert type(mapped) is error_type
    assert mapped.code == code
    assert mapped.data == data


def test_retryable_error_uses_overload_marker_for_generic_rpc_errors() -> None:
    """确认未专门分类的 RPC 错误仍可依据过载标记重试。"""

    assert is_retryable_error(JsonRpcError(-32000, "busy", {"x": "server_overloaded"}))
    assert is_retryable_error(ServerBusyError(-32000, "busy"))
    assert not is_retryable_error(ValueError("permanent"))


def test_retry_on_overload_retries_then_returns_and_rejects_bad_budget(monkeypatch) -> None:
    """确认重试等待、成功收敛和非法预算路径都保持确定性。"""

    sleeps: list[float] = []
    monkeypatch.setattr("openai_codex.retry.random.uniform", lambda low, high: 0.0)
    monkeypatch.setattr("openai_codex.retry.time.sleep", sleeps.append)

    attempts = iter([ServerBusyError(-32000, "busy"), "ok"])

    def operation() -> str:
        value = next(attempts)
        if isinstance(value, BaseException):
            raise value
        return value

    assert retry_on_overload(operation, initial_delay_s=0.1) == "ok"
    assert sleeps == [0.1]

    with pytest.raises(ValueError, match="max_attempts"):
        retry_on_overload(lambda: "never", max_attempts=0)

    with pytest.raises(ValueError, match="permanent"):
        retry_on_overload(lambda: (_ for _ in ()).throw(ValueError("permanent")))

    with pytest.raises(ServerBusyError):
        retry_on_overload(
            lambda: (_ for _ in ()).throw(ServerBusyError(-32000, "still busy")),
            max_attempts=2,
            initial_delay_s=0,
        )


@pytest.mark.parametrize(
    ("item", "wire"),
    [
        (TextInput("hello"), {"type": "text", "text": "hello"}),
        (
            ImageInput("data:image/png;base64,AA=="),
            {"type": "image", "url": "data:image/png;base64,AA=="},
        ),
        (
            LocalImageInput("/tmp/image.png"),
            {"type": "localImage", "path": "/tmp/image.png"},
        ),
        (
            SkillInput("chemistry", "/skills/chemistry"),
            {"type": "skill", "name": "chemistry", "path": "/skills/chemistry"},
        ),
        (
            MentionInput("input.xyz", "/work/input.xyz"),
            {"type": "mention", "name": "input.xyz", "path": "/work/input.xyz"},
        ),
    ],
)
def test_input_items_keep_the_0149_wire_shape(item: object, wire: dict[str, object]) -> None:
    """确认五种公开输入项都映射为 Codex 0.149 的线协议对象。"""

    assert _to_wire_item(item) == wire


def test_input_wire_helpers_cover_single_list_and_invalid_items() -> None:
    """确认单项、列表和非法项分别走对应协议分支。"""

    text = TextInput("one")
    assert _to_wire_input(text) == [{"type": "text", "text": "one"}]
    assert _to_wire_input([text, MentionInput("n", "p")]) == [
        {"type": "text", "text": "one"},
        {"type": "mention", "name": "n", "path": "p"},
    ]
    with pytest.raises(TypeError, match="unsupported input item"):
        _to_wire_item(object())


@pytest.mark.parametrize(
    ("user_agent", "server", "expected_name", "expected_version"),
    [
        ("codex/0.149.0", ServerInfo(name="server", version="1.2"), "server", "1.2"),
        ("codex/0.149.0", None, "codex", "0.149.0"),
        ("fallback/9", ServerInfo(name=None, version="1.2"), "fallback", "1.2"),
        ("fallback/9", ServerInfo(name="server", version=None), "server", "9"),
        ("codex 0.149.0", None, "codex", "0.149.0"),
    ],
)
def test_initialize_metadata_normalizes_server_identity(
    user_agent: str,
    server: ServerInfo | None,
    expected_name: str,
    expected_version: str,
) -> None:
    """确认初始化响应优先使用服务端身份并用 user-agent 补全缺失字段。"""

    payload = InitializeResponse(userAgent=user_agent, serverInfo=server)
    assert validate_initialize_metadata(payload) is payload
    assert payload.serverInfo is not None
    assert payload.serverInfo.name == expected_name
    assert payload.serverInfo.version == expected_version


@pytest.mark.parametrize(
    "payload",
    [
        InitializeResponse(userAgent=None, serverInfo=ServerInfo(name="s", version="1")),
        InitializeResponse(userAgent="codex", serverInfo=None),
        InitializeResponse(userAgent=" / ", serverInfo=None),
        InitializeResponse(userAgent="codex/", serverInfo=None),
        InitializeResponse(userAgent="codex", serverInfo=ServerInfo(name=" ", version=" ")),
    ],
)
def test_initialize_metadata_rejects_incomplete_identity(payload: InitializeResponse) -> None:
    """确认缺失 user-agent、服务端名称或版本时明确失败。"""

    with pytest.raises(RuntimeError, match="initialize response missing required metadata"):
        validate_initialize_metadata(payload)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("", (None, None)),
        ("codex/0.149", ("codex", "0.149")),
        ("codex 0.149", ("codex", "0.149")),
        ("codex", ("codex", None)),
        ("/0.149", (None, "0.149")),
    ],
)
def test_user_agent_splitter_handles_delimiters(
    value: str, expected: tuple[str | None, str | None]
) -> None:
    """确认 user-agent 的斜线、空格和空值都按协议规则解析。"""

    assert _split_user_agent(value) == expected


def _unknown(method: str, params: dict[str, object]) -> Notification:
    return Notification(method, UnknownNotification(params))


def test_message_router_routes_login_global_and_turn_queues() -> None:
    """确认登录、全局通知和 turn 通知不会互相消费。"""

    router = MessageRouter()

    pending_login = _unknown("account/login/completed", {"loginId": "login-1"})
    router.route_notification(pending_login)
    router.register_login("login-1")
    assert router.next_login_notification("login-1") is pending_login

    router.register_login("typed-login")
    typed_login = Notification(
        "account/login/completed",
        AccountLoginCompletedNotification(loginId="typed-login", success=True),
    )
    router.route_notification(typed_login)
    assert router.next_login_notification("typed-login") is typed_login
    router.unregister_login("typed-login")

    global_event = _unknown("server/notice", {})
    router.route_notification(global_event)
    assert router.next_global_notification() is global_event

    pending_turn = _unknown("item/started", {"turnId": "turn-1"})
    router.route_notification(pending_turn)
    router.register_turn("turn-1")
    assert router.next_turn_notification("turn-1") is pending_turn

    active_turn = _unknown("item/completed", {"turn": {"id": "turn-2"}})
    router.register_turn("turn-2")
    router.route_notification(active_turn)
    assert router.next_turn_notification("turn-2") is active_turn

    dropped_completion = _unknown("turn/completed", {"turnId": "turn-dropped"})
    router.route_notification(dropped_completion)
    router.register_turn("turn-dropped")
    assert "turn-dropped" not in router._pending_turn_notifications

    no_login_id = _unknown("account/login/completed", {})
    router.route_notification(no_login_id)
    assert router.next_global_notification() is no_login_id


def test_message_router_handles_typed_turns_and_goal_route_lifecycle() -> None:
    """确认 typed turn 与 thread goal 的通知路由和生命周期收束。"""

    router = MessageRouter()
    turn = Turn(id="physical-1", items=[], status=TurnStatus.in_progress)
    started = Notification(
        "turn/started", TurnStartedNotification(threadId="thread-1", turn=turn)
    )

    inactive_goal = router.reserve_goal("thread-inactive")
    router.route_notification(
        Notification(
            "turn/started",
            TurnStartedNotification(threadId="thread-inactive", turn=turn),
        )
    )
    router.register_turn("physical-1")
    assert router.next_turn_notification("physical-1").payload.turn is turn
    router.unregister_goal(inactive_goal)

    active_goal = router.register_goal("thread-1")
    router.route_notification(started)
    completed_turn = turn.model_copy(
        update={"status": TurnStatus.completed, "completedAt": 2}
    )
    router.route_notification(
        Notification(
            "turn/completed",
            TurnCompletedNotification(threadId="thread-1", turn=completed_turn),
        )
    )
    goal = ThreadGoal(
        createdAt=1,
        objective="test",
        status=ThreadGoalStatus.complete,
        threadId="thread-1",
        timeUsedSeconds=1,
        tokensUsed=1,
        updatedAt=2,
    )
    router.route_notification(
        Notification(
            "thread/goal/updated",
            ThreadGoalUpdatedNotification(threadId="thread-1", goal=goal),
        )
    )
    assert not router.has_goal(active_goal.thread_id)


def _agent_item(text: str, phase: MessagePhase | None) -> ThreadItem:
    return ThreadItem(
        root=AgentMessageThreadItem(id=f"item-{text}", text=text, phase=phase, type="agentMessage")
    )


def _usage_notification(turn_id: str) -> Notification:
    breakdown = TokenUsageBreakdown(
        cachedInputTokens=0,
        inputTokens=1,
        outputTokens=2,
        reasoningOutputTokens=0,
        totalTokens=3,
    )
    usage = ThreadTokenUsage(last=breakdown, total=breakdown)
    return Notification(
        "thread/tokenUsage/updated",
        ThreadTokenUsageUpdatedNotification(
            threadId="thread-1", tokenUsage=usage, turnId=turn_id
        ),
    )


def _item_notification(turn_id: str, item: ThreadItem) -> Notification:
    return Notification(
        "item/completed",
        ItemCompletedNotification(
            completedAtMs=1, item=item, threadId="thread-1", turnId=turn_id
        ),
    )


def _completed_notification(
    turn_id: str, status: TurnStatus, error: TurnError | None = None
) -> Notification:
    return Notification(
        "turn/completed",
        TurnCompletedNotification(
            threadId="thread-1",
            turn=Turn(
                id=turn_id,
                items=[],
                status=status,
                error=error,
                startedAt=1,
                completedAt=2,
                durationMs=1000,
            ),
        ),
    )


def test_final_response_selection_prefers_final_then_unknown_phase() -> None:
    """确认最终回答优先级和没有可用回答时的空值行为。"""

    commentary = _agent_item("commentary", MessagePhase.commentary)
    fallback = _agent_item("fallback", None)
    final = _agent_item("final", MessagePhase.final_answer)
    assert _final_assistant_response_from_items([commentary, fallback]) == "fallback"
    assert _final_assistant_response_from_items([fallback, final]) == "final"
    assert _final_assistant_response_from_items([]) is None


def test_collect_turn_result_collects_matching_items_usage_and_final_response() -> None:
    """确认同步收集器只采纳目标 turn 的 item、usage 和 completed 事件。"""

    events = iter(
        [
            _item_notification("other", _agent_item("ignored", MessagePhase.final_answer)),
            _item_notification("turn-1", _agent_item("answer", MessagePhase.final_answer)),
            _usage_notification("other"),
            _usage_notification("turn-1"),
            _completed_notification("other", TurnStatus.completed),
            _completed_notification("turn-1", TurnStatus.completed),
        ]
    )
    result = _collect_turn_result(events, turn_id="turn-1")
    assert result.id == "turn-1"
    assert result.final_response == "answer"
    assert result.usage is not None
    assert len(result.items) == 1


@pytest.mark.parametrize(
    ("error", "match"),
    [(TurnError(message="failed with details"), "failed with details"), (None, "turn failed")],
)
def test_collect_turn_result_rejects_failed_turns(error: TurnError | None, match: str) -> None:
    """确认失败 turn 以服务端错误消息或状态文字抛出。"""

    with pytest.raises(RuntimeError, match=match):
        _collect_turn_result(
            iter([_completed_notification("turn-1", TurnStatus.failed, error)]),
            turn_id="turn-1",
        )


def test_collect_turn_result_rejects_stream_without_completion() -> None:
    """确认流结束却没有目标 completed 事件时不会伪造成功结果。"""

    with pytest.raises(RuntimeError, match="turn completed event not received"):
        _collect_turn_result(
            iter([_item_notification("turn-1", _agent_item("x", None))]),
            turn_id="turn-1",
        )


async def _async_events(events: list[Notification]) -> AsyncIterator[Notification]:
    for event in events:
        yield event


def test_collect_async_turn_result_matches_sync_contract() -> None:
    """确认异步收集器与同步收集器使用同一结果和失败约定。"""

    result = asyncio.run(
        _collect_async_turn_result(
            _async_events(
                [
                    _item_notification("turn-async", _agent_item("async", None)),
                    _usage_notification("turn-async"),
                    _completed_notification("turn-async", TurnStatus.completed),
                ]
            ),
            turn_id="turn-async",
        )
    )
    assert result.final_response == "async"
    assert result.usage is not None

    with pytest.raises(RuntimeError, match="turn completed event not received"):
        asyncio.run(_collect_async_turn_result(_async_events([]), turn_id="missing"))
