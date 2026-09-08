"""覆盖 Codex SDK goal 状态机的正常、终止和清理路径。"""

from __future__ import annotations

import asyncio
from collections import deque
import threading
import time

import pytest

from openai_codex._goal import (
    _AsyncGoalNotificationStream,
    _GoalOperationState,
    _GoalStreamClosed,
    _GoalStreamCursor,
    _GoalNotificationStream,
    _logical_completion,
    _logical_notification,
    _terminal_goal_status,
)
from openai_codex.generated.v2_all import (
    ThreadGoal,
    ThreadGoalClearedNotification,
    ThreadGoalStatus,
    ThreadGoalUpdatedNotification,
    Turn,
    TurnCompletedNotification,
    TurnStartedNotification,
    TurnStatus,
)
from openai_codex.models import Notification, UnknownNotification


def make_turn(
    turn_id: str,
    status: TurnStatus = TurnStatus.completed,
    *,
    started_at: int | None = 1,
    completed_at: int | None = 4,
    duration_ms: int | None = 20,
) -> Turn:
    """构造只包含 goal 状态机所需字段的生成模型。"""
    return Turn.model_construct(
        id=turn_id,
        items=[],
        status=status,
        started_at=started_at,
        completed_at=completed_at,
        duration_ms=duration_ms,
    )


def make_goal(status: ThreadGoalStatus) -> ThreadGoal:
    """构造目标状态通知使用的最小目标模型。"""
    return ThreadGoal.model_construct(
        created_at=1,
        objective="验证目标",
        status=status,
        thread_id="thread-1",
        time_used_seconds=0,
        tokens_used=0,
    )


def started(turn_id: str = "physical-1") -> Notification:
    """构造 turn/start 通知。"""
    return Notification(
        "turn/started",
        TurnStartedNotification.model_construct(
            thread_id="thread-1",
            turn=make_turn(turn_id, TurnStatus.in_progress),
        ),
    )


def completed(
    turn_id: str = "physical-1",
    status: TurnStatus = TurnStatus.completed,
) -> Notification:
    """构造 turn/completed 通知。"""
    return Notification(
        "turn/completed",
        TurnCompletedNotification.model_construct(
            thread_id="thread-1",
            turn=make_turn(turn_id, status),
        ),
    )


def goal_updated(status: ThreadGoalStatus, turn_id: str | None = None) -> Notification:
    """构造目标更新通知。"""
    return Notification(
        "thread/goal/updated",
        ThreadGoalUpdatedNotification.model_construct(
            thread_id="thread-1",
            turn_id=turn_id,
            goal=make_goal(status),
        ),
    )


def goal_cleared() -> Notification:
    """构造目标清除通知。"""
    return Notification(
        "thread/goal/cleared",
        ThreadGoalClearedNotification.model_construct(thread_id="thread-1"),
    )


def cursor_with_state() -> tuple[_GoalOperationState, _GoalStreamCursor]:
    """构造已经绑定逻辑 turn id 的 cursor。"""
    state = _GoalOperationState(thread_id="thread-1", logical_turn_id="logical-1")
    return state, _GoalStreamCursor(state)


def test_terminal_statuses_and_state_observation_boundaries() -> None:
    """目标终态、预激活路由和物理 turn 记录必须彼此一致。"""
    assert _terminal_goal_status(None) is False
    assert _terminal_goal_status(ThreadGoalStatus.active) is False
    for status in (
        ThreadGoalStatus.paused,
        ThreadGoalStatus.blocked,
        ThreadGoalStatus.usage_limited,
        ThreadGoalStatus.budget_limited,
        ThreadGoalStatus.complete,
    ):
        assert _terminal_goal_status(status) is True

    state = _GoalOperationState(thread_id="thread-1")
    unrelated = Notification(
        "item/started",
        UnknownNotification({"threadId": "thread-1", "turnId": "physical-1"}),
    )
    assert state.observe(unrelated) is False
    assert state.observe(goal_updated(ThreadGoalStatus.active)) is True
    assert state.status == ThreadGoalStatus.active
    state.activate_turn_routing()
    assert state.observe(started("physical-1")) is True
    assert state.observe(started("physical-2")) is True
    assert state.logical_turn_id == "physical-1"
    assert state.started_turn is not None
    assert state.current_turn_id == "physical-2"
    assert state.observe(completed("physical-1")) is True
    assert state.current_turn_id == "physical-2"
    assert state.observe(completed("physical-2")) is True
    assert state.current_turn_id is None
    assert state.observe(goal_cleared()) is True
    assert state.is_finished() is True
    assert [state.next_notification().method for _ in range(6)] == [
        "thread/goal/updated",
        "turn/started",
        "turn/started",
        "turn/completed",
        "turn/completed",
        "thread/goal/cleared",
    ]


def test_operation_waiting_interrupt_and_recovery_paths() -> None:
    """等待、失败、终止和中断控制都应返回稳定结果。"""
    state = _GoalOperationState(thread_id="thread-1")
    assert state.wait_for_start(0) is None
    state.started_turn = make_turn("physical-1", TurnStatus.in_progress)
    assert state._started_logical_turn_id() is None
    state.logical_turn_id = "logical-1"
    assert state.wait_for_start(0) == "logical-1"

    failed = _GoalOperationState(thread_id="thread-1")
    error = RuntimeError("transport stopped")
    failed.fail(error)
    with pytest.raises(RuntimeError, match="transport stopped"):
        failed.wait_for_start(1)
    with pytest.raises(RuntimeError, match="transport stopped"):
        failed.active_turn()

    state.current_turn_id = "physical-2"
    assert state.active_turn(after="physical-1") == "physical-2"
    state.current_turn_id = None
    state.status = ThreadGoalStatus.complete
    assert state.active_turn(after="physical-2") is None
    state.status = ThreadGoalStatus.active
    state.finish()
    assert state.active_turn() is None

    waiting = _GoalOperationState(thread_id="thread-1")
    result: list[str | None] = []
    worker = threading.Thread(target=lambda: result.append(waiting.active_turn()))
    worker.start()
    time.sleep(0.01)
    waiting.resolve_active_turn("expected", "physical-3")
    worker.join(timeout=1)
    assert result == ["physical-3"]
    waiting.current_turn_id = "physical-3"
    assert waiting.current_turn() == "physical-3"
    assert waiting.turn_for_interrupt() == "physical-3"
    assert waiting.begin_interrupt() is True
    waiting.confirm_interrupt()
    assert waiting.explicit_interrupt() is True
    waiting.cancel_interrupt()
    assert waiting.begin_interrupt() is True
    waiting.finish()
    assert waiting.begin_interrupt() is False
    waiting.completed_turn = make_turn("physical-4")
    waiting.current_turn_id = None
    assert waiting.turn_for_interrupt() == "physical-4"
    waiting.started_turn = make_turn("physical-5")
    waiting.completed_turn = None
    assert waiting.turn_for_interrupt() == "physical-5"


def test_logical_notification_and_completion_metadata() -> None:
    """未知、直接 turn id 和嵌套 turn id 都映射到逻辑 id。"""
    unknown = Notification(
        "item/started",
        UnknownNotification(
            {
                "turnId": "physical-1",
                "turn": {"id": "physical-1", "kind": "item"},
            }
        ),
    )
    mapped_unknown = _logical_notification(unknown, "logical-1")
    assert isinstance(mapped_unknown.payload, UnknownNotification)
    assert mapped_unknown.payload.params == {
        "turnId": "logical-1",
        "turn": {"id": "logical-1", "kind": "item"},
    }
    no_turn = Notification("other", UnknownNotification({"value": 1}))
    mapped_no_turn = _logical_notification(no_turn, "logical-1")
    assert mapped_no_turn.payload.params == {"value": 1}
    malformed_nested = Notification(
        "other", UnknownNotification({"turn": {"kind": "item"}})
    )
    assert _logical_notification(malformed_nested, "logical-1").payload.params == {
        "turn": {"kind": "item"}
    }

    direct = goal_updated(ThreadGoalStatus.active, turn_id="physical-1")
    mapped_direct = _logical_notification(direct, "logical-1")
    assert mapped_direct.payload.turn_id == "logical-1"
    nested = started()
    mapped_nested = _logical_notification(nested, "logical-1")
    assert mapped_nested.payload.turn.id == "logical-1"
    no_direct_id = goal_updated(ThreadGoalStatus.active)
    assert _logical_notification(no_direct_id, "logical-1") is no_direct_id

    completed_turn = TurnCompletedNotification.model_construct(
        thread_id="thread-1", turn=make_turn("physical-1", completed_at=5)
    )
    logical = _logical_completion(
        completed_turn,
        logical_turn_id="logical-1",
        started=make_turn("physical-0", started_at=2),
        interrupted=True,
    )
    assert logical.turn.id == "logical-1"
    assert logical.turn.started_at == 2
    assert logical.turn.duration_ms == 3000
    assert logical.turn.status == TurnStatus.interrupted
    no_timing = TurnCompletedNotification.model_construct(
        thread_id="thread-1",
        turn=make_turn("physical-1", started_at=None, completed_at=None, duration_ms=7),
    )
    logical_no_timing = _logical_completion(
        no_timing,
        logical_turn_id="logical-1",
        started=None,
        interrupted=False,
    )
    assert logical_no_timing.turn.duration_ms == 7


def test_cursor_routes_goal_lifecycle_and_failures() -> None:
    """cursor 必须合并物理 turn，并只在目标终止后发出一次完成事件。"""
    state, cursor = cursor_with_state()
    with pytest.raises(RuntimeError, match="not been bound"):
        _GoalStreamCursor(_GoalOperationState(thread_id="thread-1")).process(started())

    events, done = cursor.process(started())
    assert done is False
    assert events[0].payload.turn.id == "logical-1"
    assert cursor.process(started("physical-2")) == ([], False)
    assert cursor.process(goal_updated(ThreadGoalStatus.active)) == ([], False)
    assert cursor.process(completed()) == ([], False)
    events, done = cursor.process(goal_updated(ThreadGoalStatus.complete))
    assert done is True
    assert len(events) == 1
    assert events[0].payload.turn.id == "logical-1"
    assert state.is_finished() is True

    failed_state, failed_cursor = cursor_with_state()
    failed_cursor.process(started())
    failed_cursor.process(goal_updated(ThreadGoalStatus.active))
    assert failed_cursor.process(completed(status=TurnStatus.failed)) == ([], False)
    events, done = failed_cursor.process(goal_updated(ThreadGoalStatus.complete))
    assert done is True
    assert events[0].payload.turn.status == TurnStatus.failed
    assert failed_state.is_finished() is True

    terminal_failed_state, terminal_failed_cursor = cursor_with_state()
    terminal_failed_cursor.status = ThreadGoalStatus.complete
    events, done = terminal_failed_cursor.process(completed(status=TurnStatus.failed))
    assert done is True
    assert events[0].payload.turn.status == TurnStatus.failed
    interrupted_state, interrupted_cursor = cursor_with_state()
    interrupted_cursor.status = ThreadGoalStatus.active
    interrupted_cursor.failed_completion = completed(status=TurnStatus.failed).payload
    events, done = interrupted_cursor.process(completed(status=TurnStatus.interrupted))
    assert done is True
    assert events[0].payload.turn.status == TurnStatus.failed
    assert interrupted_state.is_finished() is False

    missing_mode_state, missing_mode_cursor = cursor_with_state()
    missing_mode_cursor.process(started())
    with pytest.raises(RuntimeError, match="did not activate goal mode"):
        missing_mode_cursor.process(completed())

    cleared_state, cleared_cursor = cursor_with_state()
    cleared_cursor.process(started())
    cleared_cursor.status = ThreadGoalStatus.active
    assert cleared_cursor.process(completed()) == ([], False)
    events, done = cleared_cursor.process(goal_cleared())
    assert done is True
    assert events[0].payload.turn.id == "logical-1"
    assert cleared_state.is_finished() is True


def test_sync_goal_stream_cleans_up_on_completion_close_and_errors() -> None:
    """同步 stream 的完成、关闭、中断和异常路径都释放路由。"""
    state, _ = cursor_with_state()
    values = deque(
        [
            started(),
            goal_updated(ThreadGoalStatus.active),
            completed(),
            goal_updated(ThreadGoalStatus.complete),
        ]
    )
    unregistered: list[bool] = []
    cancelled: list[bool] = []
    stream = _GoalNotificationStream(
        state,
        values.popleft,
        lambda: unregistered.append(True),
        lambda: cancelled.append(True),
    )
    assert iter(stream) is stream
    assert next(stream).payload.turn.id == "logical-1"
    assert next(stream).payload.turn.id == "logical-1"
    with pytest.raises(StopIteration):
        next(stream)
    assert unregistered == [True]
    assert cancelled == []
    with pytest.raises(StopIteration):
        next(stream)

    closed = _GoalNotificationStream(
        _GoalOperationState(thread_id="thread-1", logical_turn_id="logical-1"),
        lambda: (_ for _ in ()).throw(_GoalStreamClosed()),
        lambda: None,
        lambda: None,
    )
    with pytest.raises(StopIteration):
        next(closed)
    assert closed._closed is True

    interrupted = _GoalNotificationStream(
        _GoalOperationState(thread_id="thread-1", logical_turn_id="logical-1"),
        lambda: (_ for _ in ()).throw(KeyboardInterrupt()),
        lambda: None,
        lambda: cancelled.append(True),
    )
    with pytest.raises(KeyboardInterrupt):
        next(interrupted)
    assert cancelled == [True]

    broken = _GoalNotificationStream(
        _GoalOperationState(thread_id="thread-1", logical_turn_id="logical-1"),
        lambda: (_ for _ in ()).throw(ValueError("broken")),
        lambda: None,
        lambda: None,
    )
    with pytest.raises(ValueError, match="broken"):
        next(broken)
    assert broken._closed is True


def test_async_goal_stream_cleans_up_on_completion_close_and_cancel() -> None:
    """异步 stream 与同步 stream 使用同一逻辑完成和异常语义。"""

    async def run() -> None:
        state, _ = cursor_with_state()
        values = deque(
            [
                started(),
                goal_updated(ThreadGoalStatus.active),
                completed(),
                goal_updated(ThreadGoalStatus.complete),
            ]
        )
        unregistered: list[bool] = []

        async def next_value() -> Notification:
            return values.popleft()

        stream = _AsyncGoalNotificationStream(
            state,
            next_value,
            lambda: unregistered.append(True),
            lambda: _completed_awaitable(),
        )
        assert stream.__aiter__() is stream
        assert (await stream.__anext__()).payload.turn.id == "logical-1"
        assert (await stream.__anext__()).payload.turn.id == "logical-1"
        with pytest.raises(StopAsyncIteration):
            await stream.__anext__()
        assert unregistered == [True]
        with pytest.raises(StopAsyncIteration):
            await stream.__anext__()

        closed = _AsyncGoalNotificationStream(
            _GoalOperationState(thread_id="thread-1", logical_turn_id="logical-1"),
            _raise_async(_GoalStreamClosed()),
            lambda: None,
            lambda: _completed_awaitable(),
        )
        with pytest.raises(StopAsyncIteration):
            await closed.__anext__()
        assert closed._closed is True

        cancelled: list[bool] = []
        cancel_stream = _AsyncGoalNotificationStream(
            _GoalOperationState(thread_id="thread-1", logical_turn_id="logical-1"),
            _raise_async(asyncio.CancelledError()),
            lambda: None,
            lambda: _record_async(cancelled),
        )
        with pytest.raises(asyncio.CancelledError):
            await cancel_stream.__anext__()
        assert cancelled == [True]

        broken = _AsyncGoalNotificationStream(
            _GoalOperationState(thread_id="thread-1", logical_turn_id="logical-1"),
            _raise_async(ValueError("broken")),
            lambda: None,
            lambda: _completed_awaitable(),
        )
        with pytest.raises(ValueError, match="broken"):
            await broken.__anext__()
        assert broken._closed is True

    asyncio.run(run())


async def _completed_awaitable() -> None:
    """提供无需外部服务的异步清理回调。"""


def _record_async(values: list[bool]):
    """返回记录异步清理调用的协程函数。"""

    async def record() -> None:
        values.append(True)

    return record()


def _raise_async(error: BaseException):
    """返回抛出指定异常的异步通知读取器。"""

    async def raise_error() -> Notification:
        raise error

    return raise_error
