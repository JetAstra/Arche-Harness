import asyncio
import hashlib
import json
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import AsyncIterator, Awaitable, Callable, Iterator

from .generated.notification_registry import notification_turn_id
from .generated.v2_all import (
    ThreadGoalClearedNotification,
    ThreadGoal,
    ThreadGoalStatus,
    ThreadGoalUpdatedNotification,
    Turn,
    TurnCompletedNotification,
    TurnStartedNotification,
    TurnStatus,
)
from .models import Notification, UnknownNotification


class _GoalStreamClosed(Exception):
    """Wake a notification reader after its logical stream closes."""


def _terminal_goal_status(status: ThreadGoalStatus | None) -> bool:
    return status in {
        ThreadGoalStatus.paused,
        ThreadGoalStatus.blocked,
        ThreadGoalStatus.usage_limited,
        ThreadGoalStatus.budget_limited,
        ThreadGoalStatus.complete,
    }


def _goal_identity(goal: ThreadGoal) -> str:
    """根据服务端 Goal 的稳定事实生成跨物理 turn 的逻辑身份。"""
    facts = {
        "threadId": goal.thread_id,
        "createdAt": goal.created_at,
        "objective": goal.objective,
        "tokenBudget": goal.token_budget,
    }
    digest = hashlib.sha256(
        json.dumps(facts, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:24]
    return f"goal-{goal.thread_id}-{digest}"


@dataclass(slots=True)
class _GoalOperationState:
    """保存一个 Goal 聚合的状态和按顺序接收的物理 turn 事实。"""

    thread_id: str
    logical_turn_id: str | None = None
    logical_goal_id: str | None = None
    goal: ThreadGoal | None = None
    physical_turn_ids: tuple[str, ...] | list[str] = field(default_factory=tuple)
    current_turn_id: str | None = None
    status: ThreadGoalStatus | None = None
    started_turn: Turn | None = None
    completed_turn: Turn | None = None
    interrupted: bool = False
    interrupt_requested: bool = False
    cleared: bool = False
    _condition: threading.Condition = field(default_factory=threading.Condition)
    _notifications: queue.Queue[Notification | BaseException] = field(default_factory=queue.Queue)
    _failure: BaseException | None = None
    _finished: bool = False
    _turn_routing_active: bool = False
    _active_turn_ids: list[str] = field(default_factory=list)
    _completed_turn_ids: set[str] = field(default_factory=set)
    _turn_statuses: dict[str, TurnStatus] = field(default_factory=dict)
    _history_replay_pending: bool = False
    _history_turns: tuple[Turn, ...] = ()
    _history_completed_ids: set[str] = field(default_factory=set)
    _goal_bound: bool = False
    _history_recovery_loading: bool = False
    _observed_notifications: list[Notification] = field(default_factory=list)

    def __post_init__(self) -> None:
        """规范化传入的物理 turn 列表，并从 Goal 计算稳定身份。"""
        physical_turn_ids = list(self.physical_turn_ids)
        if any(not isinstance(turn_id, str) or not turn_id for turn_id in physical_turn_ids):
            raise ValueError("physical turn ids must be non-empty strings")
        if len(set(physical_turn_ids)) != len(physical_turn_ids):
            raise ValueError("duplicate physical turn ids are not allowed")
        self.physical_turn_ids = physical_turn_ids
        if self.goal is not None:
            if self.goal.thread_id != self.thread_id:
                raise ValueError("goal thread identity does not match operation thread")
            self.logical_goal_id = self.logical_goal_id or _goal_identity(self.goal)
            self.status = self.goal.status
            self._goal_bound = True
        if self.logical_turn_id is None and self.physical_turn_ids:
            self.logical_turn_id = self.physical_turn_ids[0]

    def bind_goal(
        self,
        goal: ThreadGoal,
        *,
        logical_turn_id: str | None = None,
        physical_turn_ids: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        """绑定服务端返回的 Goal，并记录调用方核对过的物理 turn 身份。"""
        if goal.thread_id != self.thread_id:
            raise ValueError("goal thread identity does not match operation thread")
        with self._condition:
            if len(self._active_turn_ids) > 1:
                raise ValueError("Goal cannot bind while multiple physical turns are active")
            self.goal = goal
            self._goal_bound = True
            self.logical_goal_id = _goal_identity(goal)
            self.status = goal.status
            # thread/goal/clear 可能先于 set 到达；新的 Goal 绑定后清除旧事实。
            self.cleared = False
            if logical_turn_id is not None:
                self.logical_turn_id = logical_turn_id
            if physical_turn_ids is not None:
                normalized = list(physical_turn_ids)
                if any(not isinstance(turn_id, str) or not turn_id for turn_id in normalized):
                    raise ValueError("physical turn ids must be non-empty strings")
                if len(set(normalized)) != len(normalized):
                    raise ValueError("duplicate physical turn ids are not allowed")
                self.physical_turn_ids = normalized
                if self.logical_turn_id is None and normalized:
                    self.logical_turn_id = normalized[0]
            self._condition.notify_all()

    def seed_history(self, turns: list[Turn]) -> None:
        """把已核对的历史物理 turn 放入待回放队列，不伪造 provider 事件。"""
        with self._condition:
            history_ids = [turn.id for turn in turns]
            if len(set(history_ids)) != len(history_ids):
                raise ValueError("Goal history contains duplicate physical turn ids")
            in_progress_ids = [
                turn.id for turn in turns if turn.status == TurnStatus.in_progress
            ]
            if len(in_progress_ids) > 1:
                raise ValueError("Goal history contains multiple in-progress physical turns")
            in_progress_seen = False
            for turn in turns:
                if turn.status == TurnStatus.in_progress:
                    in_progress_seen = True
                    if turn is not turns[-1]:
                        raise ValueError("in-progress physical turn is not the final history turn")
                elif in_progress_seen:
                    raise ValueError("Goal history contains a turn after an in-progress turn")
            if self.status == ThreadGoalStatus.complete and in_progress_ids:
                raise ValueError("complete Goal history cannot contain an in-progress turn")
            if self._goal_bound and self.physical_turn_ids and history_ids:
                try:
                    positions = [self.physical_turn_ids.index(turn_id) for turn_id in history_ids]
                except ValueError as exc:
                    raise ValueError("Goal history contains an unknown physical turn id") from exc
                if positions != list(range(positions[0], positions[0] + len(positions))):
                    raise ValueError("Goal history physical turn ids are not contiguous")
            # 重新以 thread/read 的快照建立 ledger，避免恢复前 route 已记录的通知
            # 与历史状态重复计算同一 physical turn。
            self._active_turn_ids.clear()
            self._completed_turn_ids.clear()
            self._turn_statuses.clear()
            self._history_completed_ids.clear()
            self.started_turn = None
            self.completed_turn = None
            self.current_turn_id = None
            self._history_replay_pending = True
            self._history_turns = tuple(turns)
            for turn in turns:
                if turn.id not in self.physical_turn_ids:
                    self.physical_turn_ids.append(turn.id)
                self._turn_statuses[turn.id] = turn.status
                if turn.status == TurnStatus.in_progress:
                    self._active_turn_ids.append(turn.id)
                    self.current_turn_id = turn.id
                else:
                    self._completed_turn_ids.add(turn.id)
                    self.completed_turn = turn
                    self._history_completed_ids.add(turn.id)
            if self.physical_turn_ids and self.logical_turn_id is None:
                self.logical_turn_id = self.physical_turn_ids[0]
            self._condition.notify_all()

    def replay_history_notifications(self) -> None:
        """按历史顺序排入已核对的物理 turn 事件，供 complete Goal 读取。"""
        with self._condition:
            turns = self._history_turns
            self._history_replay_pending = False
        for turn in turns:
            self._notifications.put(
                Notification(
                    "turn/started",
                    TurnStartedNotification.model_construct(
                        thread_id=self.thread_id,
                        turn=turn.model_copy(update={"status": TurnStatus.in_progress}),
                    ),
                )
            )
            if turn.status != TurnStatus.in_progress:
                self._notifications.put(
                    Notification(
                        "turn/completed",
                        TurnCompletedNotification.model_construct(
                            thread_id=self.thread_id,
                            turn=turn,
                        ),
                    )
                )

    @property
    def terminal_physical_turn_id(self) -> str | None:
        """返回最后一个已完成物理 turn 的身份。"""
        with self._condition:
            return self.completed_turn.id if self.completed_turn is not None else None

    @property
    def final_turn_status(self) -> TurnStatus | None:
        """返回最后一个物理 turn 的服务端状态。"""
        with self._condition:
            return self.completed_turn.status if self.completed_turn is not None else None

    def observe(self, notification: Notification) -> bool:
        """记录一条路由通知，并在目标完成时唤醒等待者。"""
        payload = notification.payload
        with self._condition:
            if not self._can_observe_payload(payload):
                return False
            try:
                self._record_payload(payload)
            except BaseException as exc:
                self._failure = exc
                self._condition.notify_all()
                self._notifications.put(exc)
                raise
            self._finish_if_goal_done()
            self._observed_notifications.append(notification)
            self._condition.notify_all()
        self._notifications.put(notification)
        return True

    def _can_observe_payload(self, payload: object) -> bool:
        """判断当前路由是否已经允许记录物理 turn 通知。"""
        if self._turn_routing_active:
            return True
        return isinstance(
            payload,
            ThreadGoalClearedNotification | ThreadGoalUpdatedNotification,
        )

    def _record_payload(self, payload: object) -> None:
        """把通知中的物理 turn 或目标状态写入操作状态。"""
        if isinstance(payload, TurnStartedNotification):
            self._record_turn_started(payload)
            return
        if isinstance(payload, TurnCompletedNotification):
            self._record_turn_completed(payload)
            return
        if isinstance(payload, ThreadGoalUpdatedNotification):
            self._record_goal_updated(payload)
            return
        self._record_goal_cleared(payload)

    def _record_goal_cleared(self, payload: object) -> None:
        """记录目标清除事件，其他通知不改变目标状态。"""
        if isinstance(payload, ThreadGoalClearedNotification):
            self.cleared = True
            self.status = None

    def _next_expected_physical_turn_id(self) -> str | None:
        """返回已知物理序列中尚未开始的下一个 turn 身份。"""
        for turn_id in self.physical_turn_ids:
            if turn_id not in self._completed_turn_ids and turn_id not in self._active_turn_ids:
                return turn_id
        return None

    def _record_turn_started(self, payload: TurnStartedNotification) -> None:
        """记录新的物理 turn，重复或跨线程身份会立即失败。"""
        if payload.thread_id != self.thread_id:
            raise ValueError("goal notification thread identity does not match operation")
        turn_id = payload.turn.id
        if not isinstance(turn_id, str) or not turn_id:
            raise ValueError("goal turn started notification has no valid physical id")
        if (
            self._goal_bound
            and _terminal_goal_status(self.status)
            and turn_id not in self.physical_turn_ids
        ):
            raise ValueError("a terminal Goal cannot start another physical turn")
        if self._goal_bound and self._active_turn_ids:
            raise ValueError("a Goal cannot start a second physical turn before completion")
        if self._goal_bound:
            expected_turn_id = self._next_expected_physical_turn_id()
            if expected_turn_id is not None and turn_id != expected_turn_id:
                raise ValueError(
                    "out-of-order physical turn start: "
                    f"expected {expected_turn_id}, got {turn_id}"
                )
        if (
            turn_id in self._active_turn_ids
            or turn_id in self._completed_turn_ids
            or turn_id in self._history_completed_ids
        ):
            raise ValueError(f"duplicate physical turn id: {turn_id}")
        if turn_id not in self.physical_turn_ids:
            self.physical_turn_ids.append(turn_id)
        self._active_turn_ids.append(turn_id)
        if self.logical_turn_id is None:
            self.logical_turn_id = payload.turn.id
        self.current_turn_id = turn_id
        if self.started_turn is None:
            self.started_turn = payload.turn

    def _record_turn_completed(self, payload: TurnCompletedNotification) -> None:
        """按开始顺序记录物理 turn 完成，未知、重复或乱序均失败。"""
        if payload.thread_id != self.thread_id:
            raise ValueError("goal notification thread identity does not match operation")
        if payload.turn.status == TurnStatus.in_progress:
            raise ValueError("turn/completed notification cannot have inProgress status")
        turn_id = payload.turn.id
        if turn_id not in self.physical_turn_ids:
            raise ValueError(f"unknown physical turn id: {turn_id}")
        if turn_id in self._completed_turn_ids:
            raise ValueError(f"duplicate physical turn completion: {turn_id}")
        if self._active_turn_ids and self._active_turn_ids[0] != turn_id:
            raise ValueError(
                "out-of-order physical turn completion: "
                f"expected {self._active_turn_ids[0]}, got {turn_id}"
            )
        if not self._active_turn_ids:
            if turn_id in self._history_completed_ids:
                raise ValueError(f"duplicate physical turn completion: {turn_id}")
            if not any(turn.id == turn_id for turn in self._history_turns):
                raise ValueError(f"completion for inactive physical turn: {turn_id}")
        if turn_id in self._active_turn_ids:
            self._active_turn_ids.remove(turn_id)
        self._completed_turn_ids.add(turn_id)
        self._turn_statuses[turn_id] = payload.turn.status
        self.completed_turn = payload.turn
        self.current_turn_id = self._active_turn_ids[-1] if self._active_turn_ids else None

    def _record_goal_updated(self, payload: ThreadGoalUpdatedNotification) -> None:
        """记录服务端目标状态，并在重新激活时撤销清除标记。"""
        if payload.thread_id != self.thread_id or payload.goal.thread_id != self.thread_id:
            raise ValueError("goal update thread identity does not match operation")
        if self.goal is not None and _goal_identity(payload.goal) != self.logical_goal_id:
            raise ValueError("goal update identity does not match the attached Goal")
        self.goal = payload.goal
        self.status = payload.goal.status
        if self.status == ThreadGoalStatus.active:
            self.cleared = False

    def _goal_is_terminal(self) -> bool:
        """判断目标是否已经清除或进入终态。"""
        if self.cleared:
            return True
        return _terminal_goal_status(self.status)

    def _finish_if_goal_done(self) -> None:
        """在没有活动 turn 且目标终止后标记逻辑操作完成。"""
        if self._history_recovery_loading:
            return
        if self._active_turn_ids:
            return
        if self.completed_turn is None:
            return
        if not self._goal_is_terminal():
            return
        self._finished = True

    def activate_turn_routing(self) -> None:
        """开启物理 turn 通知接收；此前只允许接收 Goal 状态通知。"""
        with self._condition:
            self._turn_routing_active = True

    def begin_history_recovery(self) -> None:
        """在读取 Goal 和历史期间暂缓自动注销 route，保留并发通知。"""
        with self._condition:
            self._history_recovery_loading = True

    def finish_history_recovery(self) -> None:
        """结束历史读取保护；后续通知重新允许按终态标记操作结束。"""
        with self._condition:
            self._history_recovery_loading = False

    def consume_observed_notifications(self) -> tuple[Notification, ...]:
        """取出恢复读请求期间已记录的通知，供历史快照核对一次。"""
        with self._condition:
            notifications = tuple(self._observed_notifications)
            self._observed_notifications.clear()
            return notifications

    def reconcile_history_notifications(
        self,
        history_ids: set[str],
        *,
        drop_history_events: bool,
    ) -> None:
        """删除将由历史回放重建的重复物理事件，保留其余排队通知。"""
        if not drop_history_events or not history_ids:
            return
        with self._notifications.mutex:
            retained: deque[Notification | BaseException] = deque()
            while self._notifications.queue:
                item = self._notifications.queue.popleft()
                if isinstance(item, Notification):
                    turn_id = notification_turn_id(item.payload)
                    if turn_id in history_ids:
                        continue
                retained.append(item)
            self._notifications.queue.extend(retained)

    def wait_for_start(self, timeout: float) -> str | None:
        """Wait for the runtime-generated first turn without consuming its event."""
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                logical_turn_id = self._started_logical_turn_id()
                if logical_turn_id is not None:
                    return logical_turn_id
                self._raise_if_failed()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)

    def _started_logical_turn_id(self) -> str | None:
        """返回已经同时拥有首个物理 turn 和逻辑 id 的结果。"""
        if self.started_turn is None:
            return None
        return self.logical_turn_id

    def _raise_if_failed(self) -> None:
        """把传输线程记录的异常重新交给当前等待者。"""
        if self._failure is not None:
            raise self._failure

    def fail(self, exc: BaseException) -> None:
        with self._condition:
            self._failure = exc
            self._condition.notify_all()
        self._notifications.put(exc)

    def next_notification(self) -> Notification:
        item = self._notifications.get()
        if isinstance(item, BaseException):
            raise item
        return item

    def finish(self) -> None:
        """Mark the logical operation inactive and wake waiting controls."""
        with self._condition:
            self._finished = True
            self.current_turn_id = None
            self._condition.notify_all()

    def is_finished(self) -> bool:
        with self._condition:
            return self._finished

    def begin_interrupt(self) -> bool:
        with self._condition:
            if self._finished:
                return False
            self.interrupt_requested = True
            return True

    def confirm_interrupt(self) -> None:
        with self._condition:
            self.interrupted = True
            self.status = ThreadGoalStatus.paused
            if self.goal is not None:
                self.goal = self.goal.model_copy(update={"status": ThreadGoalStatus.paused})
            self.interrupt_requested = False
            self._condition.notify_all()

    def cancel_interrupt(self) -> None:
        with self._condition:
            self.interrupt_requested = False
            self._condition.notify_all()

    def explicit_interrupt(self) -> bool:
        with self._condition:
            while self.interrupt_requested:
                self._condition.wait()
            return self.interrupted

    def active_turn(self, *, after: str | None = None) -> str | None:
        """Wait for the current turn, or return None once the goal has ended."""
        with self._condition:
            self._raise_if_failed()
            if self._finished:
                return None
            return self._wait_for_active_turn(after)

    def _wait_for_active_turn(self, after: str | None) -> str | None:
        """在已持有条件锁时等待新的物理 turn 或目标终止。"""
        while True:
            self._raise_if_failed()
            current_turn_id = self._active_turn_after(after)
            if current_turn_id is not None:
                return current_turn_id
            if self._goal_is_terminal():
                return None
            self._condition.wait()

    def _active_turn_after(self, after: str | None) -> str | None:
        """返回尚未被调用方消费的物理 turn id。"""
        if self.current_turn_id is None:
            return None
        if self.current_turn_id == after:
            return None
        return self.current_turn_id

    def current_turn(self) -> str | None:
        """Return the current physical turn without waiting for rollover."""
        with self._condition:
            return self.current_turn_id

    def resolve_active_turn(self, expected: str, active: str) -> None:
        """Adopt a server-reported active id when routed state is still stale."""
        with self._condition:
            if self.current_turn_id in {None, expected}:
                self.current_turn_id = active
                self._condition.notify_all()

    def turn_for_interrupt(self) -> str | None:
        """Return an active or stale turn id that can resolve rollover races."""
        with self._condition:
            if self.current_turn_id is not None:
                return self.current_turn_id
            if self.completed_turn is not None:
                return self.completed_turn.id
            if self.started_turn is not None:
                return self.started_turn.id
            return None

    def wake_notification_reader(self) -> None:
        """Release a reader blocked after its stream has been closed."""
        self._notifications.put(_GoalStreamClosed())


def _logical_notification(notification: Notification, logical_turn_id: str) -> Notification:
    """Return a copy whose turn metadata uses the logical operation id."""
    payload = notification.payload
    if isinstance(payload, UnknownNotification):
        return _logical_unknown_notification(notification, logical_turn_id)

    turn_id = notification_turn_id(payload)
    if turn_id is None:
        return notification
    return _logical_typed_notification(notification, logical_turn_id)


def _logical_unknown_notification(
    notification: Notification,
    logical_turn_id: str,
) -> Notification:
    """将未知通知中可识别的物理 turn id 投影为逻辑 id。"""
    payload = notification.payload
    if not isinstance(payload, UnknownNotification):
        return notification
    params = dict(payload.params)
    if isinstance(params.get("turnId"), str):
        params["turnId"] = logical_turn_id
    turn = params.get("turn")
    logical_turn = _logical_unknown_turn(turn, logical_turn_id)
    if logical_turn is not turn:
        params["turn"] = logical_turn
    return Notification(notification.method, UnknownNotification(params))


def _logical_unknown_turn(turn: object, logical_turn_id: str) -> object:
    """复制未知通知中的嵌套 turn，无法识别时保留原值。"""
    if not isinstance(turn, dict):
        return turn
    if not isinstance(turn.get("id"), str):
        return turn
    return {**turn, "id": logical_turn_id}


def _logical_typed_notification(
    notification: Notification,
    logical_turn_id: str,
) -> Notification:
    """将带有生成模型 turn 字段的通知复制为逻辑 turn id。"""
    payload = notification.payload
    if hasattr(payload, "turn_id"):
        return Notification(
            notification.method,
            payload.model_copy(update={"turn_id": logical_turn_id}),
        )
    if hasattr(payload, "turn"):
        logical_turn = payload.turn.model_copy(update={"id": logical_turn_id})
        return Notification(
            notification.method,
            payload.model_copy(update={"turn": logical_turn}),
        )
    return notification


def _logical_completion(
    completed: TurnCompletedNotification,
    *,
    logical_turn_id: str,
    started: Turn | None,
    interrupted: bool,
) -> TurnCompletedNotification:
    """把最后一个物理 turn 的完成事实合并为一次逻辑完成。"""
    final_turn = completed.turn
    started_at, duration_ms = _logical_timing(final_turn, started)
    updates: dict[str, object] = {
        "id": logical_turn_id,
        "started_at": started_at,
        "duration_ms": duration_ms,
    }
    if interrupted:
        updates["status"] = TurnStatus.interrupted
    return completed.model_copy(update={"turn": final_turn.model_copy(update=updates)})


def _logical_timing(final_turn: Turn, started: Turn | None) -> tuple[int | None, int | None]:
    """计算逻辑 turn 的起始时间和跨物理 turn 的总耗时。"""
    started_at = final_turn.started_at
    if started is not None:
        started_at = started.started_at
    duration_ms = final_turn.duration_ms
    if started_at is None:
        return started_at, duration_ms
    completed_at = final_turn.completed_at
    if completed_at is None:
        return started_at, duration_ms
    return started_at, max(0, completed_at - started_at) * 1000


@dataclass(slots=True)
class _GoalStreamCursor:
    """把按顺序到达的物理 Goal 事件聚合为一个逻辑 turn 事件流。"""

    state: _GoalOperationState
    started: Turn | None = None
    last_completed: TurnCompletedNotification | None = None
    failed_completion: TurnCompletedNotification | None = None
    status: ThreadGoalStatus | None = None
    active: bool = False
    cleared: bool = False
    _active_turn_ids: list[str] = field(default_factory=list)
    _physical_turn_ids: list[str] = field(default_factory=list)
    _completed_turn_ids: set[str] = field(default_factory=set)
    _strict_identity: bool = False
    _history_ids: set[str] = field(default_factory=set)
    _history_replay_mode: bool = False

    def __post_init__(self) -> None:
        """从已恢复状态复制身份和 Goal 状态，避免修改共享路由记录。"""
        self.status = self.state.status
        self.cleared = self.state.cleared
        self._physical_turn_ids = list(self.state.physical_turn_ids)
        # state 已观察过的通知仍在队列中，cursor 必须以队列事件重建顺序。
        self._active_turn_ids = []
        self._completed_turn_ids = set()
        self._strict_identity = bool(self._physical_turn_ids) or self.state.goal is not None
        self._history_ids = {turn.id for turn in self.state._history_turns}
        self._history_replay_mode = bool(self._history_ids) and _terminal_goal_status(self.status)
        if not self._history_replay_mode:
            self._completed_turn_ids = {
                turn.id
                for turn in self.state._history_turns
                if turn.status != TurnStatus.in_progress
            }
        self.active = bool(self._active_turn_ids)

    def process(self, notification: Notification) -> tuple[list[Notification], bool]:
        payload = notification.payload
        if isinstance(payload, TurnStartedNotification):
            # 恢复时 Goal 可能已持久化但尚未收到首个 TurnStarted；先用这条合法
            # provider 事件绑定真实物理身份，再生成逻辑 started 事件。
            logical_turn_id = self._bind_first_turn(payload)
            return self._process_turn_started(notification, payload, logical_turn_id)

        if isinstance(payload, TurnCompletedNotification):
            return self._process_turn_completed(notification, payload)

        logical_turn_id = self._logical_turn_id()
        return self._process_goal_or_other(notification, payload, logical_turn_id)

    def _bind_first_turn(self, payload: TurnStartedNotification) -> str:
        """在恢复句柄首次收到合法开始通知时绑定物理和逻辑 turn 身份。"""
        logical_turn_id = self.state.logical_turn_id
        if logical_turn_id is not None:
            return logical_turn_id
        if self.state.goal is None:
            raise RuntimeError("goal operation has not been bound to a logical turn id")
        if payload.thread_id != self.state.thread_id:
            raise RuntimeError("goal turn started notification has mismatched thread id")
        self.state._record_turn_started(payload)
        logical_turn_id = self.state.logical_turn_id
        if logical_turn_id is None:
            raise RuntimeError("goal turn started notification did not bind a physical id")
        return logical_turn_id

    def _logical_turn_id(self) -> str:
        """读取逻辑 turn id，未绑定时拒绝处理物理通知。"""
        logical_turn_id = self.state.logical_turn_id
        if logical_turn_id is None:
            raise RuntimeError("goal operation has not been bound to a logical turn id")
        return logical_turn_id

    def _process_turn_started(
        self,
        notification: Notification,
        payload: TurnStartedNotification,
        logical_turn_id: str,
    ) -> tuple[list[Notification], bool]:
        """把第一个物理 turn 开始事件转换为逻辑事件。"""
        if payload.thread_id != self.state.thread_id:
            raise RuntimeError("goal turn started notification has mismatched thread id")
        turn_id = payload.turn.id
        strict_identity = self._strict_identity
        if (
            strict_identity
            and _terminal_goal_status(self.status)
            and not self._active_turn_ids
            and turn_id not in self._physical_turn_ids
        ):
            raise RuntimeError("a terminal Goal cannot start another physical turn")
        if self.started is not None and not strict_identity:
            self.active = True
            return [], False
        if strict_identity and self._active_turn_ids:
            raise RuntimeError("a Goal cannot start a second physical turn before completion")
        if strict_identity:
            expected_turn_id = self._next_expected_physical_turn_id()
            if expected_turn_id is not None and turn_id != expected_turn_id:
                raise RuntimeError(
                    "out-of-order physical turn start: "
                    f"expected {expected_turn_id}, got {turn_id}"
                )
        if turn_id in self._active_turn_ids or turn_id in self._completed_turn_ids:
            if strict_identity:
                raise RuntimeError(f"duplicate physical turn id: {turn_id}")
            return [], False
        if turn_id not in self._physical_turn_ids:
            self._physical_turn_ids.append(turn_id)
        if (
            isinstance(self.state.physical_turn_ids, list)
            and turn_id not in self.state.physical_turn_ids
        ):
            self.state.physical_turn_ids.append(turn_id)
        self._active_turn_ids.append(turn_id)
        self.active = True
        if self.started is not None:
            return [], False
        self.started = payload.turn
        return [_logical_notification(notification, logical_turn_id)], False

    def _process_turn_completed(
        self,
        notification: Notification,
        payload: TurnCompletedNotification,
    ) -> tuple[list[Notification], bool]:
        """按物理 turn 结果路由中断、失败和成功完成事件。"""
        if payload.thread_id != self.state.thread_id:
            raise RuntimeError("goal turn completed notification has mismatched thread id")
        if payload.turn.status == TurnStatus.in_progress:
            raise RuntimeError("turn/completed notification cannot have inProgress status")
        turn_id = payload.turn.id
        strict_identity = self._strict_identity
        if strict_identity:
            if turn_id not in self._physical_turn_ids:
                raise RuntimeError(f"unknown physical turn id: {turn_id}")
            if turn_id in self._completed_turn_ids:
                raise RuntimeError(f"duplicate physical turn completion: {turn_id}")
            if self._active_turn_ids and self._active_turn_ids[0] != turn_id:
                raise RuntimeError(
                    "out-of-order physical turn completion: "
                    f"expected {self._active_turn_ids[0]}, got {turn_id}"
                )
            if not self._active_turn_ids and turn_id not in self._history_ids:
                raise RuntimeError(f"physical turn completion has no matching start: {turn_id}")
            if turn_id in self._active_turn_ids:
                self._active_turn_ids.remove(turn_id)
            self._completed_turn_ids.add(turn_id)
        elif self._active_turn_ids:
            self._active_turn_ids.remove(turn_id)
        self.active = bool(self._active_turn_ids)
        self.last_completed = payload
        if payload.turn.status == TurnStatus.interrupted:
            self.failed_completion = self.failed_completion or payload
            if strict_identity and (
                self.active
                or not self._all_known_physical_turns_completed()
                or not self._goal_is_terminal()
            ):
                return [], False
            completion_payload = payload if strict_identity else self.failed_completion or payload
            return [self._completion(notification.method, completion_payload)], True
        if payload.turn.status == TurnStatus.failed:
            return self._process_failed_completion(notification.method, payload)
        return self._process_success_completion(notification.method, payload)

    def _next_expected_physical_turn_id(self) -> str | None:
        """返回已知物理序列中尚未开始的下一个 turn 身份。"""
        for turn_id in self._physical_turn_ids:
            if turn_id not in self._completed_turn_ids and turn_id not in self._active_turn_ids:
                return turn_id
        return None

    def _process_failed_completion(
        self,
        method: str,
        payload: TurnCompletedNotification,
    ) -> tuple[list[Notification], bool]:
        """保存失败物理 turn，并在目标已经终止时完成逻辑操作。"""
        self.failed_completion = payload
        if not self._goal_is_terminal() or not self._all_known_physical_turns_completed():
            return [], False
        return self._finish_with_completion(method, payload)

    def _process_success_completion(
        self,
        method: str,
        payload: TurnCompletedNotification,
    ) -> tuple[list[Notification], bool]:
        """处理成功物理 turn，并等待目标状态确认最终完成。"""
        self._assert_goal_mode_activated()
        if not self._goal_is_terminal() or not self._all_known_physical_turns_completed():
            return [], False
        return self._finish_with_completion(
            method,
            payload,
        )

    def _assert_goal_mode_activated(self) -> None:
        """拒绝没有收到目标状态、且也没有清除目标的普通 turn。"""
        if self.status is not None:
            return
        if self.cleared:
            return
        raise RuntimeError(
            "the connected Codex runtime did not activate goal mode for this turn"
        )

    def _process_goal_or_other(
        self,
        notification: Notification,
        payload: object,
        logical_turn_id: str,
    ) -> tuple[list[Notification], bool]:
        """记录目标状态或转发其他通知，并在必要时补发完成事件。"""
        events = self._record_goal_event(notification, payload, logical_turn_id)
        if not self._completion_ready():
            return events, False
        return self._finish_with_completion(
            "turn/completed",
            self.failed_completion or self.last_completed,
            events,
        )

    def _record_goal_event(
        self,
        notification: Notification,
        payload: object,
        logical_turn_id: str,
    ) -> list[Notification]:
        """更新目标状态并返回应继续暴露给调用方的事件。"""
        events = [_logical_notification(notification, logical_turn_id)]
        if isinstance(payload, ThreadGoalUpdatedNotification):
            self.status = payload.goal.status
            self.state.status = self.status
            self.state.goal = payload.goal
            if self.status == ThreadGoalStatus.active:
                self.cleared = False
            return []
        if isinstance(payload, ThreadGoalClearedNotification):
            self.cleared = True
            self.state.cleared = True
            return []
        return events

    def _goal_is_terminal(self) -> bool:
        """判断当前逻辑目标是否已经清除或进入终态。"""
        if self.cleared:
            return True
        return _terminal_goal_status(self.status)

    def _completion_ready(self) -> bool:
        """判断最后一个物理 turn 是否可以收束为逻辑完成。"""
        if self.active:
            return False
        if self.last_completed is None:
            return False
        return self._goal_is_terminal() and self._all_known_physical_turns_completed()

    def _all_known_physical_turns_completed(self) -> bool:
        """确认历史或当前已知的每个物理 turn 都已收到完成事件。"""
        if not self._strict_identity:
            return True
        known_ids = self._history_ids if self._history_replay_mode else set(self._physical_turn_ids)
        return bool(known_ids) and known_ids.issubset(self._completed_turn_ids)

    def _finish_with_completion(
        self,
        method: str,
        payload: TurnCompletedNotification,
        events: list[Notification] | None = None,
    ) -> tuple[list[Notification], bool]:
        """标记逻辑操作完成，并把最终完成通知追加到事件列表。"""
        self.state.finish()
        completed_events = [] if events is None else events
        completed_events.append(self._completion(method, payload))
        return completed_events, True

    def _completion(
        self,
        method: str,
        payload: TurnCompletedNotification,
    ) -> Notification:
        logical_turn_id = self.state.logical_turn_id
        if logical_turn_id is None:
            raise RuntimeError("goal operation has not been bound to a logical turn id")
        return Notification(
            method,
            _logical_completion(
                payload,
                logical_turn_id=logical_turn_id,
                started=self.started,
                interrupted=self.state.explicit_interrupt(),
            ),
        )


@dataclass(slots=True)
class _GoalNotificationStream(Iterator[Notification]):
    """以可关闭的同步事件流读取一次逻辑 Goal 操作。"""

    state: _GoalOperationState
    next_notification: Callable[[], Notification]
    unregister: Callable[[], None]
    cancel_goal: Callable[[], None]
    _cursor: _GoalStreamCursor = field(init=False)
    _pending: deque[Notification] = field(default_factory=deque)
    _closed: bool = False

    def __post_init__(self) -> None:
        self._cursor = _GoalStreamCursor(self.state)

    def __iter__(self) -> "_GoalNotificationStream":
        return self

    def __next__(self) -> Notification:
        self._ensure_open()
        try:
            return self._next_pending_notification()
        except BaseException as exc:
            self._raise_after_error(exc)

    def _ensure_open(self) -> None:
        """在读取前检查同步逻辑流是否已经关闭。"""
        if self._closed:
            raise StopIteration

    def _raise_after_error(self, exc: BaseException) -> None:
        """执行同步流的异常清理，并重新抛出原始错误。"""
        if isinstance(exc, _GoalStreamClosed):
            self.close()
            raise StopIteration from None
        if isinstance(exc, KeyboardInterrupt):
            self.cancel_goal()
        self.close()
        raise exc

    def _next_pending_notification(self) -> Notification:
        """读取并处理通知，直到逻辑事件队列中出现可返回的事件。"""
        while not self._pending:
            notification = self.next_notification()
            events, completed = self._cursor.process(notification)
            self._pending.extend(events)
            if completed:
                self._finish()
        return self._pending.popleft()

    def _finish(self) -> None:
        if self._closed:
            return
        self.state.finish()
        self.state.wake_notification_reader()
        self.unregister()
        self._closed = True

    def close(self) -> None:
        self._finish()


@dataclass(slots=True)
class _AsyncGoalNotificationStream(AsyncIterator[Notification]):
    """Closeable asynchronous view of one logical goal operation."""

    state: _GoalOperationState
    next_notification: Callable[[], Awaitable[Notification]]
    unregister: Callable[[], None]
    cancel_goal: Callable[[], Awaitable[None]]
    _cursor: _GoalStreamCursor = field(init=False)
    _pending: deque[Notification] = field(default_factory=deque)
    _closed: bool = False

    def __post_init__(self) -> None:
        self._cursor = _GoalStreamCursor(self.state)

    def __aiter__(self) -> "_AsyncGoalNotificationStream":
        return self

    async def __anext__(self) -> Notification:
        self._ensure_open()
        try:
            return await self._next_pending_notification()
        except BaseException as exc:
            await self._raise_after_error(exc)

    def _ensure_open(self) -> None:
        """在读取前检查异步逻辑流是否已经关闭。"""
        if self._closed:
            raise StopAsyncIteration

    async def _raise_after_error(self, exc: BaseException) -> None:
        """执行异步流的异常清理，并重新抛出原始错误。"""
        if isinstance(exc, _GoalStreamClosed):
            await self.aclose()
            raise StopAsyncIteration from None
        if isinstance(exc, asyncio.CancelledError):
            await self.cancel_goal()
        await self.aclose()
        raise exc

    async def _next_pending_notification(self) -> Notification:
        """异步读取并处理通知，直到得到一个逻辑事件。"""
        while not self._pending:
            notification = await self.next_notification()
            events, completed = self._cursor.process(notification)
            self._pending.extend(events)
            if completed:
                self._finish()
        return self._pending.popleft()

    def _finish(self) -> None:
        if self._closed:
            return
        self.state.finish()
        self.state.wake_notification_reader()
        self.unregister()
        self._closed = True

    async def aclose(self) -> None:
        self._finish()
