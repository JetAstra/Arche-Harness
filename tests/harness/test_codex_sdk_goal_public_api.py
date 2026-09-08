"""覆盖 Codex SDK Goal 公开接口的创建、恢复和终态合同。"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from openai_codex import (
    AsyncGoalHandle,
    AsyncThread,
    CodexError,
    GoalCheckpoint,
    GoalHandle,
    GoalNotFoundError,
    Thread,
)
from openai_codex._goal import _GoalOperationState
from openai_codex.client import CodexClient
from openai_codex.generated.v2_all import (
    ThreadGoalClearedNotification,
    ThreadGoalUpdatedNotification,
    TurnCompletedNotification,
    TurnStartedNotification,
    ThreadGoal,
    ThreadGoalStatus,
    ThreadReadResponse,
    Turn,
    TurnStatus,
)
from openai_codex.models import Notification


def _goal(
    status: ThreadGoalStatus = ThreadGoalStatus.active,
    *,
    created_at: int = 100,
    objective: str = "完成目标",
    token_budget: int | None = 321,
) -> ThreadGoal:
    """构造公开 Goal 测试所需的 provider 目标。"""

    return ThreadGoal.model_construct(
        created_at=created_at,
        objective=objective,
        status=status,
        thread_id="thread-1",
        time_used_seconds=0,
        token_budget=token_budget,
        tokens_used=0,
        updated_at=101,
    )


def _turn(turn_id: str, status: TurnStatus = TurnStatus.completed) -> Turn:
    """构造恢复历史中的物理 turn。"""

    return Turn.model_construct(
        id=turn_id,
        items=[],
        status=status,
        started_at=100,
        completed_at=101 if status is not TurnStatus.in_progress else None,
        duration_ms=1000 if status is not TurnStatus.in_progress else None,
    )


def _started(turn_id: str) -> Notification:
    """构造带 thread 身份的物理 turn 开始通知。"""
    return Notification(
        "turn/started",
        TurnStartedNotification.model_construct(
            thread_id="thread-1", turn=_turn(turn_id, TurnStatus.in_progress)
        ),
    )


def _completed(turn_id: str, status: TurnStatus = TurnStatus.completed) -> Notification:
    """构造带 thread 身份的物理 turn 完成通知。"""
    return Notification(
        "turn/completed",
        TurnCompletedNotification.model_construct(
            thread_id="thread-1", turn=_turn(turn_id, status)
        ),
    )


def _goal_updated(status: ThreadGoalStatus) -> Notification:
    """构造服务端 Goal 状态更新通知。"""
    return Notification(
        "thread/goal/updated",
        ThreadGoalUpdatedNotification.model_construct(
            thread_id="thread-1",
            turn_id=None,
            goal=_goal(status),
        ),
    )


def test_thread_goal_returns_public_handle_with_goal_identity_and_physical_ids() -> None:
    """同步创建应返回稳定逻辑身份并保留有序物理 turn 身份。"""

    provider_goal = _goal()
    state = _GoalOperationState(
        thread_id="thread-1",
        goal=provider_goal,
        physical_turn_ids=("physical-1", "physical-2"),
    )

    class FakeClient:
        def start_goal_operation(
            self,
            thread_id: str,
            objective: str,
            *,
            token_budget: int | None = None,
        ) -> tuple[_GoalOperationState, str]:
            assert (thread_id, objective, token_budget) == ("thread-1", "新目标", 321)
            return state, "physical-1"

        def cancel_goal_operation(self, _state: _GoalOperationState) -> None:
            return None

    handle = Thread(FakeClient(), "thread-1").goal("新目标", token_budget=321)

    assert isinstance(handle, GoalHandle)
    assert handle.thread_id == "thread-1"
    assert handle.id == "physical-1"
    assert handle.logical_goal_id != handle.id
    assert handle.objective == provider_goal.objective
    assert handle.token_budget == provider_goal.token_budget
    assert handle.provider_created_at == provider_goal.created_at
    assert handle.physical_turn_ids == ("physical-1", "physical-2")


def test_resume_goal_validates_history_without_clear_or_set_and_rejects_ambiguity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """恢复只读 GoalGet/history，身份不唯一时必须安全失败。"""

    client = CodexClient.__new__(CodexClient)
    provider_goal = _goal(ThreadGoalStatus.complete)
    history = SimpleNamespace(turns=[_turn("physical-1"), _turn("physical-2")])
    calls: list[str] = []

    monkeypatch.setattr(
        client, "thread_goal_get", lambda _thread_id: SimpleNamespace(goal=provider_goal)
    )
    monkeypatch.setattr(
        client,
        "thread_read",
        lambda _thread_id, include_turns=False: (
            calls.append("read") or ThreadReadResponse.model_construct(thread=history)
        ),
    )
    monkeypatch.setattr(client, "thread_goal_clear", lambda _thread_id: calls.append("clear"))
    monkeypatch.setattr(client, "thread_goal_set", lambda *_args, **_kwargs: calls.append("set"))

    logical_id = CodexClient.goal_identity(provider_goal)
    checkpoint = GoalCheckpoint(
        logical_turn_id="physical-1",
        logical_goal_id=logical_id,
        thread_id="thread-1",
        physical_turn_ids=("physical-1", "physical-2"),
        objective=provider_goal.objective,
        token_budget=provider_goal.token_budget,
        provider_created_at=provider_goal.created_at,
    )
    handle = Thread(client, "thread-1").resume_goal(checkpoint=checkpoint)
    assert isinstance(handle, GoalHandle)
    assert handle.physical_turn_ids == ("physical-1", "physical-2")
    assert calls == ["read"]

    ambiguous = GoalCheckpoint(logical_goal_id=logical_id, thread_id="thread-1")
    with pytest.raises(Exception, match="physical|ambiguous|identity"):
        Thread(client, "thread-1").resume_goal(checkpoint=ambiguous)
    assert calls == ["read", "read"]


def test_resume_goal_raises_precise_error_when_provider_has_no_goal() -> None:
    """只有服务端明确返回 goal=None 时才使用 GoalNotFoundError。"""
    client = CodexClient.__new__(CodexClient)
    client.thread_goal_get = lambda _thread_id: SimpleNamespace(goal=None)  # type: ignore[method-assign]

    with pytest.raises(GoalNotFoundError):
        Thread(client, "thread-1").resume_goal()


def test_resume_goal_extends_checkpoint_with_history_tail_without_clear_or_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """恢复应纳入 checkpoint 之后已落盘的新物理 turn。"""
    client = CodexClient.__new__(CodexClient)
    provider_goal = _goal(ThreadGoalStatus.complete)
    history = SimpleNamespace(turns=[_turn("physical-1"), _turn("physical-2")])
    monkeypatch.setattr(
        client, "thread_goal_get", lambda _thread_id: SimpleNamespace(goal=provider_goal)
    )
    monkeypatch.setattr(
        client,
        "thread_read",
        lambda _thread_id, include_turns=False: ThreadReadResponse.model_construct(thread=history),
    )
    monkeypatch.setattr(
        client,
        "thread_goal_clear",
        lambda _thread_id: pytest.fail("resume must not clear the Goal"),
    )
    monkeypatch.setattr(
        client,
        "thread_goal_set",
        lambda *_args, **_kwargs: pytest.fail("resume must not set the Goal"),
    )
    checkpoint = GoalCheckpoint(
        logical_turn_id="physical-1",
        logical_goal_id=CodexClient.goal_identity(provider_goal),
        thread_id="thread-1",
        physical_turn_ids=("physical-1",),
    )

    handle = Thread(client, "thread-1").resume_goal(checkpoint=checkpoint)

    assert handle.physical_turn_ids == ("physical-1", "physical-2")


def test_resume_goal_extends_to_unique_active_history_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """恢复 active Goal 时应纳入末尾唯一的 inProgress 物理 turn。"""
    client = CodexClient.__new__(CodexClient)
    provider_goal = _goal(ThreadGoalStatus.active)
    history = SimpleNamespace(
        turns=[_turn("physical-1"), _turn("physical-2", TurnStatus.in_progress)]
    )
    monkeypatch.setattr(
        client, "thread_goal_get", lambda _thread_id: SimpleNamespace(goal=provider_goal)
    )
    monkeypatch.setattr(
        client,
        "thread_read",
        lambda _thread_id, include_turns=False: ThreadReadResponse.model_construct(thread=history),
    )
    checkpoint = GoalCheckpoint(
        logical_turn_id="physical-1",
        logical_goal_id=CodexClient.goal_identity(provider_goal),
        thread_id="thread-1",
        physical_turn_ids=("physical-1",),
    )

    handle = Thread(client, "thread-1").resume_goal(checkpoint=checkpoint)

    assert handle.physical_turn_ids == ("physical-1", "physical-2")
    assert handle.current_physical_turn_id == "physical-2"


def test_resume_goal_rejects_noncontiguous_checkpoint_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """checkpoint 物理前缀跳过历史 turn 时必须拒绝恢复。"""
    client = CodexClient.__new__(CodexClient)
    provider_goal = _goal(ThreadGoalStatus.complete)
    history = SimpleNamespace(
        turns=[_turn("physical-1"), _turn("physical-2"), _turn("physical-3")]
    )
    monkeypatch.setattr(
        client, "thread_goal_get", lambda _thread_id: SimpleNamespace(goal=provider_goal)
    )
    monkeypatch.setattr(
        client,
        "thread_read",
        lambda _thread_id, include_turns=False: ThreadReadResponse.model_construct(thread=history),
    )
    checkpoint = GoalCheckpoint(
        logical_turn_id="physical-1",
        logical_goal_id=CodexClient.goal_identity(provider_goal),
        thread_id="thread-1",
        physical_turn_ids=("physical-1", "physical-3"),
    )

    with pytest.raises(Exception, match="contiguous"):
        Thread(client, "thread-1").resume_goal(checkpoint=checkpoint)


def test_async_thread_goal_exposes_same_public_contract() -> None:
    """异步 Thread.goal 应返回异步公开 GoalHandle。"""

    provider_goal = _goal()
    state = _GoalOperationState(thread_id="thread-1", goal=provider_goal)

    class FakeAsyncClient:
        async def start_goal_operation(
            self,
            thread_id: str,
            objective: str,
            *,
            token_budget: int | None = None,
        ) -> tuple[_GoalOperationState, str]:
            assert (thread_id, objective, token_budget) == ("thread-1", "异步目标", 321)
            return state, "physical-1"

        async def cancel_goal_operation(self, _state: _GoalOperationState) -> None:
            return None

    class FakeCodex:
        _client = FakeAsyncClient()

        async def _ensure_initialized(self) -> None:
            return None

    async def run() -> None:
        handle = await AsyncThread(FakeCodex(), "thread-1").goal("异步目标", token_budget=321)
        assert isinstance(handle, AsyncGoalHandle)
        assert handle.id == "physical-1"
        assert handle.logical_goal_id != handle.id
        assert handle.thread_id == "thread-1"

    asyncio.run(run())


def test_goal_wire_get_and_set_include_token_budget() -> None:
    """GoalGet 使用只读 wire，GoalSet 应传递 tokenBudget。"""
    client = CodexClient.__new__(CodexClient)
    calls: list[tuple[str, object]] = []

    def request(method: str, payload: object, *, response_model: object) -> object:
        calls.append((method, payload))
        return SimpleNamespace(goal=_goal())

    client.request = request  # type: ignore[method-assign]
    client.thread_goal_get("thread-1")
    client.thread_goal_set(
        "thread-1",
        objective="目标",
        status=ThreadGoalStatus.active,
        token_budget=321,
    )
    assert calls == [
        ("thread/goal/get", {"threadId": "thread-1"}),
        (
            "thread/goal/set",
            {
                "threadId": "thread-1",
                "objective": "目标",
                "status": "active",
                "tokenBudget": 321,
            },
        ),
    ]


def test_goal_stream_coalesces_ordered_physical_turns_and_keeps_provider_ids() -> None:
    """多物理 turn 只产生一个逻辑完成事件，同时保留完整有序身份。"""
    state = _GoalOperationState(thread_id="thread-1", goal=_goal())
    state.activate_turn_routing()
    for event in (
        _goal_updated(ThreadGoalStatus.active),
        _started("physical-1"),
        _completed("physical-1"),
        _started("physical-2"),
        _completed("physical-2"),
        _goal_updated(ThreadGoalStatus.complete),
    ):
        assert state.observe(event) is True

    class FakeClient:
        def next_goal_notification(self, operation: _GoalOperationState) -> Notification:
            return operation.next_notification()

        def unregister_goal_operation(self, _operation: _GoalOperationState) -> None:
            return None

        def cancel_goal_operation(self, _operation: _GoalOperationState) -> None:
            return None

    handle = GoalHandle(FakeClient(), state)
    events = list(handle.stream())
    assert [event.method for event in events] == ["turn/started", "turn/completed"]
    assert events[0].payload.turn.id == "physical-1"
    assert events[1].payload.turn.id == "physical-1"
    assert handle.id == "physical-1"
    assert handle.logical_goal_id == CodexClient.goal_identity(_goal())
    assert handle.physical_turn_ids == ("physical-1", "physical-2")
    assert handle.current_physical_turn_id is None
    assert handle.terminal_physical_turn_id == "physical-2"
    assert handle.final_turn_status is TurnStatus.completed


def test_prequeued_goal_stream_waits_for_last_physical_turn() -> None:
    """stream 创建前通知已全部到达时，必须消费到最后物理 turn 才结束。"""
    first = _completed("physical-1").payload
    last = _completed("physical-2").payload
    first = first.model_copy(update={"turn": first.turn.model_copy(update={"completed_at": 101})})
    last = last.model_copy(update={"turn": last.turn.model_copy(update={"completed_at": 202})})
    state = _GoalOperationState(thread_id="thread-1", goal=_goal())
    state.activate_turn_routing()
    for event in (
        _goal_updated(ThreadGoalStatus.active),
        _started("physical-1"),
        Notification("turn/completed", first),
        _started("physical-2"),
        Notification("turn/completed", last),
        _goal_updated(ThreadGoalStatus.complete),
    ):
        state.observe(event)

    class FakeClient:
        def next_goal_notification(self, operation: _GoalOperationState) -> Notification:
            return operation.next_notification()

        def unregister_goal_operation(self, _operation: _GoalOperationState) -> None:
            return None

        def cancel_goal_operation(self, _operation: _GoalOperationState) -> None:
            return None

    events = list(GoalHandle(FakeClient(), state).stream())

    assert events[-1].payload.turn.completed_at == 202
    with pytest.raises(Exception):
        state.next_notification()
    assert state._notifications.empty()


def test_goal_state_rejects_unknown_duplicate_and_out_of_order_physical_turns() -> None:
    """状态机对未知、重复和乱序物理身份必须 fail-closed。"""
    state = _GoalOperationState(
        thread_id="thread-1",
        goal=_goal(),
        physical_turn_ids=("physical-1", "physical-2"),
    )
    state.activate_turn_routing()
    state.observe(_started("physical-1"))
    with pytest.raises(ValueError, match="second physical turn"):
        state.observe(_started("physical-2"))

    out_of_order = _GoalOperationState(
        thread_id="thread-1",
        goal=_goal(),
        physical_turn_ids=("physical-1", "physical-2"),
    )
    out_of_order.activate_turn_routing()
    out_of_order.observe(_started("physical-1"))
    with pytest.raises(ValueError, match="out-of-order"):
        out_of_order.observe(_completed("physical-2"))

    duplicate = _GoalOperationState(thread_id="thread-1", goal=_goal())
    duplicate.activate_turn_routing()
    duplicate.observe(_started("physical-1"))
    duplicate.observe(_completed("physical-1"))
    with pytest.raises(ValueError, match="duplicate"):
        duplicate.observe(_completed("physical-1"))

    unknown = _GoalOperationState(thread_id="thread-1", goal=_goal())
    unknown.activate_turn_routing()
    with pytest.raises(ValueError, match="unknown"):
        unknown.observe(_completed("physical-9"))


def test_goal_history_rejects_ambiguous_active_turns() -> None:
    """恢复历史不能同时含多个 inProgress 或在其后继续追加 turn。"""
    state = _GoalOperationState(thread_id="thread-1", goal=_goal())
    in_progress_one = _turn("physical-1", TurnStatus.in_progress)
    in_progress_two = _turn("physical-2", TurnStatus.in_progress)
    with pytest.raises(ValueError, match="multiple in-progress"):
        state.seed_history([in_progress_one, in_progress_two])

    with pytest.raises(ValueError, match="not the final"):
        state.seed_history([in_progress_one, _turn("physical-2")])

    active = _GoalOperationState(thread_id="thread-1", goal=_goal())
    active.seed_history([_turn("physical-1"), in_progress_two])
    assert active.current_turn_id == "physical-2"
    assert active._active_turn_ids == ["physical-2"]


@pytest.mark.parametrize(
    "status",
    [
        ThreadGoalStatus.paused,
        ThreadGoalStatus.blocked,
        ThreadGoalStatus.usage_limited,
        ThreadGoalStatus.budget_limited,
    ],
)
def test_goal_non_complete_terminal_status_is_readable_but_run_fails(
    status: ThreadGoalStatus,
) -> None:
    """paused、blocked、usage/budget 限制不能被句柄伪装为成功。"""
    state = _GoalOperationState(thread_id="thread-1", goal=_goal())
    state.activate_turn_routing()
    for event in (_started("physical-1"), _completed("physical-1"), _goal_updated(status)):
        state.observe(event)

    class FakeClient:
        def next_goal_notification(self, operation: _GoalOperationState) -> Notification:
            return operation.next_notification()

        def unregister_goal_operation(self, _operation: _GoalOperationState) -> None:
            return None

        def cancel_goal_operation(self, _operation: _GoalOperationState) -> None:
            return None

    handle = GoalHandle(FakeClient(), state)
    assert handle.status is status
    assert handle.final_turn_status is TurnStatus.completed
    with pytest.raises(Exception, match="non-complete"):
        handle.run()


def test_goal_interrupt_marks_paused_and_calls_cancel() -> None:
    """中断应调用取消路径并公开 paused/interrupted 事实。"""
    state = _GoalOperationState(thread_id="thread-1", goal=_goal())
    state.current_turn_id = "physical-1"
    cancelled: list[str] = []

    class FakeClient:
        def cancel_goal_operation(self, _operation: _GoalOperationState) -> None:
            cancelled.append("cancel")

    handle = GoalHandle(FakeClient(), state)
    handle.interrupt()
    assert cancelled == ["cancel"]
    assert handle.interrupted is True
    assert handle.status is ThreadGoalStatus.paused


def test_async_resume_goal_is_read_only_and_preserves_checkpoint() -> None:
    """异步恢复应调用 resume_goal_operation，不重建 Goal。"""
    state = _GoalOperationState(thread_id="thread-1", goal=_goal(ThreadGoalStatus.active))
    state.logical_turn_id = "physical-1"
    checkpoint = GoalCheckpoint(
        logical_turn_id="physical-1",
        logical_goal_id=CodexClient.goal_identity(_goal()),
        thread_id="thread-1",
        physical_turn_ids=("physical-1",),
    )
    seen: list[object] = []

    class FakeAsyncClient:
        async def resume_goal_operation(
            self,
            thread_id: str,
            *,
            checkpoint: object | None = None,
        ) -> tuple[_GoalOperationState, str]:
            seen.append((thread_id, checkpoint))
            return state, "physical-1"

    class FakeCodex:
        _client = FakeAsyncClient()

        async def _ensure_initialized(self) -> None:
            return None

    async def run() -> None:
        handle = await AsyncThread(FakeCodex(), "thread-1").resume_goal(checkpoint)
        assert isinstance(handle, AsyncGoalHandle)
        assert handle.id == "physical-1"
        assert seen == [("thread-1", checkpoint)]

    asyncio.run(run())


def test_async_goal_stream_uses_first_physical_id_for_logical_events() -> None:
    """异步事件流应把最终物理完成映射为首个 turn 的逻辑身份。"""
    state = _GoalOperationState(
        thread_id="thread-1",
        goal=_goal(ThreadGoalStatus.active),
    )
    state.activate_turn_routing()
    for event in (
        _started("physical-1"),
        _completed("physical-1"),
        _goal_updated(ThreadGoalStatus.complete),
    ):
        state.observe(event)

    class FakeAsyncClient:
        async def next_goal_notification(
            self, operation: _GoalOperationState
        ) -> Notification:
            return operation.next_notification()

        def unregister_goal_operation(self, _operation: _GoalOperationState) -> None:
            return None

        async def cancel_goal_operation(self, _operation: _GoalOperationState) -> None:
            return None

    class FakeCodex:
        _client = FakeAsyncClient()

    async def run() -> None:
        handle = AsyncGoalHandle(FakeCodex(), state)
        events: list[Notification] = []
        async for event in handle.stream():
            events.append(event)
        assert [event.method for event in events] == ["turn/started", "turn/completed"]
        assert events[0].payload.turn.id == handle.id == "physical-1"
        assert events[1].payload.turn.id == handle.id

    asyncio.run(run())


def test_resume_goal_baseline_allows_no_first_turn_yet() -> None:
    """Goal 已 set 但尚无首 turn 时，baseline 尾部为空仍可返回句柄。"""
    client = CodexClient.__new__(CodexClient)
    provider_goal = _goal(ThreadGoalStatus.active)
    history = SimpleNamespace(turns=[_turn("old-1"), _turn("old-2")])
    client.thread_goal_get = lambda _thread_id: SimpleNamespace(goal=provider_goal)  # type: ignore[method-assign]
    client.thread_read = lambda _thread_id, include_turns=False: ThreadReadResponse.model_construct(  # type: ignore[method-assign]
        thread=history
    )
    checkpoint = GoalCheckpoint(
        logical_goal_id=CodexClient.goal_identity(provider_goal),
        thread_id="thread-1",
        physical_turn_count_before=2,
    )

    handle = Thread(client, "thread-1").resume_goal(checkpoint)

    assert handle.physical_turn_ids == ()
    with pytest.raises(CodexError, match="first physical turn"):
        _ = handle.id


def test_resume_goal_baseline_includes_new_tail_turn() -> None:
    """baseline 后出现的新物理 turn 应被恢复并保留 active 身份。"""
    client = CodexClient.__new__(CodexClient)
    provider_goal = _goal(ThreadGoalStatus.active)
    history = SimpleNamespace(
        turns=[_turn("old-1"), _turn("old-2"), _turn("physical-1", TurnStatus.in_progress)]
    )
    client.thread_goal_get = lambda _thread_id: SimpleNamespace(goal=provider_goal)  # type: ignore[method-assign]
    client.thread_read = lambda _thread_id, include_turns=False: ThreadReadResponse.model_construct(  # type: ignore[method-assign]
        thread=history
    )
    checkpoint = GoalCheckpoint(
        logical_goal_id=CodexClient.goal_identity(provider_goal),
        thread_id="thread-1",
        physical_turn_count_before=2,
    )

    handle = Thread(client, "thread-1").resume_goal(checkpoint)

    assert handle.physical_turn_ids == ("physical-1",)
    assert handle.current_physical_turn_id == "physical-1"


def test_goal_stream_binds_first_turn_for_resumed_goal_without_started_identity() -> None:
    """恢复后首个 TurnStarted 应绑定真实物理身份再生成逻辑事件。"""
    state = _GoalOperationState(thread_id="thread-1", goal=_goal(ThreadGoalStatus.active))

    class FakeClient:
        def next_goal_notification(self, _state: _GoalOperationState) -> Notification:
            return _started("physical-1")

        def unregister_goal_operation(self, _state: _GoalOperationState) -> None:
            return None

        def cancel_goal_operation(self, _state: _GoalOperationState) -> None:
            return None

    handle = GoalHandle(FakeClient(), state)
    stream = handle.stream()
    try:
        event = next(stream)
    finally:
        stream.close()

    assert event.payload.turn.id == "physical-1"
    assert handle.id == "physical-1"
    assert handle.physical_turn_ids == ("physical-1",)


def test_bind_active_goal_clears_stale_goal_cleared_state() -> None:
    """clear 后重新绑定 active Goal 不应让首个完成事件提前收束。"""
    state = _GoalOperationState(thread_id="thread-1")
    state.observe(
        Notification(
            "thread/goal/cleared",
            ThreadGoalClearedNotification.model_construct(thread_id="thread-1"),
        )
    )
    state.bind_goal(_goal(ThreadGoalStatus.active))
    state.activate_turn_routing()
    state.observe(_started("physical-1"))
    state.observe(_completed("physical-1"))

    assert state.cleared is False
    assert state.is_finished() is False


def test_terminal_goal_replay_consumes_last_physical_turn_before_finishing() -> None:
    """complete Goal 的历史回放必须消费全部物理 turn，最终完成来自末尾 turn。"""
    first = _turn("physical-1").model_copy(update={"completed_at": 101})
    last = _turn("physical-2").model_copy(update={"completed_at": 202})
    state = _GoalOperationState(
        thread_id="thread-1",
        goal=_goal(ThreadGoalStatus.complete),
    )
    state.seed_history([first, last])
    state.replay_history_notifications()

    class FakeClient:
        def next_goal_notification(self, operation: _GoalOperationState) -> Notification:
            return operation.next_notification()

        def unregister_goal_operation(self, _operation: _GoalOperationState) -> None:
            return None

        def cancel_goal_operation(self, _operation: _GoalOperationState) -> None:
            return None

    events = list(GoalHandle(FakeClient(), state).stream())

    assert [event.method for event in events] == ["turn/started", "turn/completed"]
    assert events[-1].payload.turn.completed_at == 202


def test_resume_reserves_and_activates_route_before_reading_history() -> None:
    """恢复读请求期间到达的首个物理通知不能因 route 尚未注册而丢失。"""
    client = CodexClient.__new__(CodexClient)
    client._router = object()
    provider_goal = _goal(ThreadGoalStatus.active)
    state = _GoalOperationState(thread_id="thread-1")
    calls: list[str] = []

    def reserve(_thread_id: str) -> _GoalOperationState:
        calls.append("reserve")
        return state

    def goal_get(_thread_id: str) -> SimpleNamespace:
        calls.append("get")
        assert state._turn_routing_active is True
        assert state.observe(_started("physical-1")) is True
        return SimpleNamespace(goal=provider_goal)

    def thread_read(_thread_id: str, include_turns: bool = False) -> ThreadReadResponse:
        calls.append("read")
        assert include_turns is True
        return ThreadReadResponse.model_construct(
            thread=SimpleNamespace(turns=[_turn("physical-1", TurnStatus.in_progress)])
        )

    client.reserve_goal_operation = reserve  # type: ignore[method-assign]
    client.thread_goal_get = goal_get  # type: ignore[method-assign]
    client.thread_read = thread_read  # type: ignore[method-assign]
    client.unregister_goal_operation = lambda _state: calls.append("unregister")  # type: ignore[method-assign]

    handle = Thread(client, "thread-1").resume_goal(
        GoalCheckpoint(
            logical_goal_id=CodexClient.goal_identity(provider_goal),
            thread_id="thread-1",
            physical_turn_ids=("physical-1",),
        )
    )

    assert calls[:3] == ["reserve", "get", "read"]
    assert handle.physical_turn_ids == ("physical-1",)
    assert state.next_notification().method == "turn/started"


def test_goal_interrupt_failure_does_not_fabricate_paused_or_finished_state() -> None:
    """pause 或 turn/interrupt 失败时中断状态必须保持未确认并抛出错误。"""
    client = CodexClient.__new__(CodexClient)
    state = _GoalOperationState(
        thread_id="thread-1",
        goal=_goal(ThreadGoalStatus.active),
    )
    state.current_turn_id = "physical-1"
    client.pause_goal = lambda _thread_id: SimpleNamespace(  # type: ignore[method-assign]
        goal=_goal(ThreadGoalStatus.paused)
    )
    client.turn_interrupt = lambda *_args: (_ for _ in ()).throw(  # type: ignore[method-assign]
        RuntimeError("interrupt failed")
    )

    handle = GoalHandle(client, state)
    with pytest.raises(RuntimeError, match="interrupt failed"):
        handle.interrupt()

    assert handle.status is ThreadGoalStatus.active
    assert handle.interrupted is False
    assert state.is_finished() is False


def test_goal_state_rejects_noncontiguous_physical_turn_start_order() -> None:
    """已核对物理 turn 序列中不能跳过 p2 先启动 p3。"""
    state = _GoalOperationState(
        thread_id="thread-1",
        goal=_goal(),
        physical_turn_ids=("physical-1", "physical-2", "physical-3"),
    )
    state.activate_turn_routing()
    state.observe(_started("physical-1"))
    state.observe(_completed("physical-1"))

    with pytest.raises(ValueError, match="out-of-order"):
        state.observe(_started("physical-3"))


def test_goal_state_rejects_in_progress_turn_completed_status() -> None:
    """turn/completed 不得携带 inProgress 状态。"""
    state = _GoalOperationState(
        thread_id="thread-1",
        goal=_goal(),
        physical_turn_ids=("physical-1",),
    )
    state.activate_turn_routing()
    state.observe(_started("physical-1"))

    with pytest.raises(ValueError, match="inProgress"):
        state.observe(_completed("physical-1", TurnStatus.in_progress))


def test_goal_checkpoint_rejects_string_physical_turn_ids_at_construction() -> None:
    """物理 turn 身份必须是序列，字符串不能被拆成字符。"""
    with pytest.raises(TypeError, match="ordered sequence"):
        GoalCheckpoint(physical_turn_ids="physical-1")  # type: ignore[arg-type]


def test_goal_stream_uses_last_success_after_failed_retry() -> None:
    """失败后重试成功时，逻辑完成事件必须携带最后一次成功结果。"""
    first = _turn("physical-1", TurnStatus.failed).model_copy(update={"completed_at": 101})
    last = _turn("physical-2").model_copy(update={"completed_at": 202})
    state = _GoalOperationState(thread_id="thread-1", goal=_goal())
    state.activate_turn_routing()
    for event in (
        _started("physical-1"),
        Notification("turn/completed", TurnCompletedNotification.model_construct(
            thread_id="thread-1", turn=first
        )),
        _started("physical-2"),
        Notification("turn/completed", TurnCompletedNotification.model_construct(
            thread_id="thread-1", turn=last
        )),
        _goal_updated(ThreadGoalStatus.complete),
    ):
        state.observe(event)

    class FakeClient:
        def next_goal_notification(self, operation: _GoalOperationState) -> Notification:
            return operation.next_notification()

        def unregister_goal_operation(self, _operation: _GoalOperationState) -> None:
            return None

        def cancel_goal_operation(self, _operation: _GoalOperationState) -> None:
            return None

    events = list(GoalHandle(FakeClient(), state).stream())

    assert events[-1].payload.turn.completed_at == 202
