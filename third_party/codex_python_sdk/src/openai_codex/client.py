import json
import os
import re
import subprocess
import threading
import uuid
from _thread import LockType
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, TypeVar

from pydantic import BaseModel

from ._goal import _GoalOperationState, _goal_identity, _terminal_goal_status
from ._message_router import MessageRouter
from ._version import __version__ as SDK_VERSION
from .errors import (
    CodexError,
    GoalNotFoundError,
    InvalidRequestError,
    StderrIndicator,
    TransportClosedError,
    TransportDiagnostic,
    TransportPhase,
)
from .generated.notification_registry import NOTIFICATION_MODELS
from .generated.v2_all import (
    AccountLoginCompletedNotification,
    AgentMessageDeltaNotification,
    CancelLoginAccountResponse,
    ChatgptDeviceCodeLoginAccountResponse,
    ChatgptLoginAccountResponse,
    GetAccountParams as V2GetAccountParams,
    GetAccountResponse,
    IdleThreadStatus,
    LoginAccountParams as V2LoginAccountParams,
    LoginAccountResponse,
    LogoutAccountResponse,
    ModelListResponse,
    ThreadArchiveResponse,
    ThreadCompactStartResponse,
    ThreadForkParams as V2ThreadForkParams,
    ThreadForkResponse,
    ThreadGoalClearResponse,
    ThreadGoalGetResponse,
    ThreadGoal,
    ThreadGoalSetResponse,
    ThreadGoalStatus,
    ThreadListParams as V2ThreadListParams,
    ThreadListResponse,
    ThreadReadResponse,
    ThreadResumeParams as V2ThreadResumeParams,
    ThreadResumeResponse,
    ThreadSetNameResponse,
    ThreadStartParams as V2ThreadStartParams,
    ThreadStartResponse,
    ThreadUnarchiveResponse,
    TurnCompletedNotification,
    TurnInterruptResponse,
    TurnStartParams as V2TurnStartParams,
    TurnStartResponse,
    TurnStartedNotification,
    TurnStatus,
    TurnSteerResponse,
)
from .models import (
    InitializeResponse,
    JsonObject,
    JsonValue,
    Notification,
    UnknownNotification,
)
from .retry import retry_on_overload

ModelT = TypeVar("ModelT", bound=BaseModel)
ApprovalHandler = Callable[[str, JsonObject | None], JsonObject]
RUNTIME_PKG_NAME = "openai-codex-cli-bin"
_GOAL_START_TIMEOUT_S = 30.0
_STDERR_DRAIN_WAIT_S = 0.5


def _transport_phase_for_method(method: str) -> TransportPhase:
    """把 JSON-RPC 方法映射到有限的 transport 阶段枚举。

    输入是即将发送的方法名；动作只查询固定映射，无外部副作用。已知方法返回
    对应阶段，未知方法返回 ``REQUEST``；本函数不抛出协议错误。
    """

    return {
        "initialize": TransportPhase.INITIALIZE,
        "thread/start": TransportPhase.THREAD_START,
        "thread/resume": TransportPhase.THREAD_RESUME,
        "thread/read": TransportPhase.THREAD_READ,
        "turn/start": TransportPhase.TURN_START,
    }.get(method, TransportPhase.REQUEST)


_STDERR_INDICATOR_PATTERNS: tuple[tuple[StderrIndicator, tuple[str, ...]], ...] = (
    (
        StderrIndicator.AUTHENTICATION,
        (
            "unauthorized",
            "authentication failed",
            "invalid api key",
            "api key",
        ),
    ),
    (
        StderrIndicator.PERMISSION,
        ("permission denied", "operation not permitted", "access denied"),
    ),
    (
        StderrIndicator.SANDBOX,
        ("sandbox", "bubblewrap", "bwrap"),
    ),
    (
        StderrIndicator.MISSING_DEPENDENCY,
        ("no such file or directory", "command not found", "executable not found"),
    ),
    (
        StderrIndicator.CONFIGURATION,
        (
            "invalid config",
            "invalid configuration",
            "unknown config",
            "configuration error",
            "failed to parse config",
        ),
    ),
    (
        StderrIndicator.NETWORK,
        (
            "connection refused",
            "network unreachable",
            "connect error",
            "dns",
            "timed out",
            "timeout",
        ),
    ),
    (
        StderrIndicator.RUNTIME,
        ("panic", "fatal", "internal error", "error:"),
    ),
)


def _classify_stderr_indicator(line: str) -> StderrIndicator:
    """把单行 stderr 归类为白名单枚举，不返回或保存原文。

    输入是一行仅在 drain 线程内短暂存在的文本；动作按固定优先级匹配故障词，
    无外部副作用。命中时返回对应枚举，未命中时返回 ``NONE``，不会抛出分类错误。
    """

    lowered = line.casefold()
    for indicator, patterns in _STDERR_INDICATOR_PATTERNS:
        if any(pattern in lowered for pattern in patterns):
            return indicator
    return StderrIndicator.NONE


@dataclass(slots=True)
class _ThreadStartLock:
    lock: LockType = field(default_factory=threading.Lock)
    users: int = 0


def _active_turn_id_from_error(exc: InvalidRequestError) -> str | None:
    match = re.search(r" but found `?([^`]+)`?$", exc.message)
    return match.group(1) if match is not None else None


def _params_dict(
    params: (
        V2ThreadStartParams
        | V2ThreadResumeParams
        | V2ThreadListParams
        | V2ThreadForkParams
        | V2TurnStartParams
        | V2GetAccountParams
        | V2LoginAccountParams
        | JsonObject
        | None
    ),
) -> JsonObject:
    if params is None:
        return {}
    if hasattr(params, "model_dump"):
        dumped = params.model_dump(
            by_alias=True,
            exclude_none=True,
            mode="json",
        )
        if not isinstance(dumped, dict):
            raise TypeError("Expected model_dump() to return dict")
        return dumped
    if isinstance(params, dict):
        return params
    raise TypeError(f"Expected generated params model or dict, got {type(params).__name__}")


def _installed_codex_path() -> Path:
    try:
        from codex_cli_bin import bundled_codex_path
    except ImportError as exc:
        raise FileNotFoundError(
            "Unable to locate the pinned Codex runtime. Install the published SDK build "
            f"with its {RUNTIME_PKG_NAME} dependency, or set CodexConfig.codex_bin "
            "explicitly."
        ) from exc

    return bundled_codex_path()


def _installed_codex_path_dirs() -> tuple[Path, ...]:
    try:
        from codex_cli_bin import bundled_path_dir
    except (ImportError, AttributeError):
        return ()

    path_dir = bundled_path_dir()
    return (path_dir,) if path_dir is not None else ()


def _prepend_path_dirs(env: dict[str, str], path_dirs: tuple[Path, ...]) -> None:
    if not path_dirs:
        return

    path_key = _path_env_key(env)
    if os.name == "nt":
        for key in list(env):
            if key.upper() == "PATH" and key != path_key:
                env.pop(key)

    path_sep = os.pathsep
    existing_path = env.get(path_key, "")
    path_dir_values = [str(path_dir) for path_dir in path_dirs]
    existing_entries = [
        entry for entry in existing_path.split(path_sep) if entry and entry not in path_dir_values
    ]
    env[path_key] = path_sep.join([*path_dir_values, *existing_entries])


def _path_env_key(env: dict[str, str]) -> str:
    if os.name != "nt":
        return "PATH"

    matching_keys = [key for key in env if key.upper() == "PATH"]
    if "Path" in matching_keys:
        return "Path"
    return matching_keys[-1] if matching_keys else "PATH"


@dataclass(frozen=True)
class CodexBinResolverOps:
    installed_codex_path: Callable[[], Path]
    path_exists: Callable[[Path], bool]


def _default_codex_bin_resolver_ops() -> CodexBinResolverOps:
    return CodexBinResolverOps(
        installed_codex_path=_installed_codex_path,
        path_exists=lambda path: path.exists(),
    )


def resolve_codex_bin(config: "CodexConfig", ops: CodexBinResolverOps) -> Path:
    if config.codex_bin is not None:
        codex_bin = Path(config.codex_bin)
        if not ops.path_exists(codex_bin):
            raise FileNotFoundError(
                f"Codex binary not found at {codex_bin}. Set CodexConfig.codex_bin "
                "to a valid binary path."
            )
        return codex_bin

    return ops.installed_codex_path()


def _resolve_codex_bin(config: "CodexConfig") -> Path:
    return resolve_codex_bin(config, _default_codex_bin_resolver_ops())


@dataclass(slots=True)
class CodexConfig:
    """Configuration for launching and identifying the local Codex runtime.

    Most callers can use ``Codex()`` without configuration. Set ``codex_bin``
    only when intentionally using a specific local Codex executable.
    """

    codex_bin: str | None = None
    launch_args_override: tuple[str, ...] | None = None
    config_overrides: tuple[str, ...] = ()
    cwd: str | None = None
    env: dict[str, str] | None = None
    client_name: str = "codex_python_sdk"
    client_title: str = "Codex Python SDK"
    client_version: str = SDK_VERSION
    experimental_api: bool = True


class CodexClient:
    """Synchronous typed JSON-RPC client for `codex app-server` over stdio."""

    def __init__(
        self,
        config: CodexConfig | None = None,
        approval_handler: ApprovalHandler | None = None,
    ) -> None:
        self.config = config or CodexConfig()
        self._approval_handler = approval_handler or self._default_approval_handler
        self._proc: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()
        self._thread_start_locks_guard = threading.Lock()
        self._thread_start_locks: dict[str, _ThreadStartLock] = {}
        self._router = MessageRouter()
        self._stderr_indicator = StderrIndicator.NONE
        self._stderr_indicator_lock = threading.Lock()
        self._stderr_drain_done = threading.Event()
        self._stderr_drain_done.set()
        self._stderr_thread: threading.Thread | None = None
        self._reader_thread: threading.Thread | None = None
        self._transport_phase = TransportPhase.PROCESS_START

    def __enter__(self) -> "CodexClient":
        self.start()
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()

    def start(self) -> None:
        if self._proc is not None:
            return

        self._transport_phase = TransportPhase.PROCESS_START
        self._stderr_indicator = StderrIndicator.NONE
        self._stderr_drain_done.clear()

        path_dirs: tuple[Path, ...] = ()
        if self.config.launch_args_override is not None:
            args = list(self.config.launch_args_override)
        else:
            codex_bin = _resolve_codex_bin(self.config)
            if self.config.codex_bin is None:
                path_dirs = _installed_codex_path_dirs()
            args = [str(codex_bin)]
            for kv in self.config.config_overrides:
                args.extend(["--config", kv])
            args.extend(["app-server", "--listen", "stdio://"])

        env = os.environ.copy()
        if self.config.env:
            env.update(self.config.env)
        _prepend_path_dirs(env, path_dirs)

        self._proc = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            cwd=self.config.cwd,
            env=env,
            bufsize=1,
        )

        self._start_stderr_drain_thread()
        self._start_reader_thread()

    def close(self) -> None:
        if self._proc is None:
            return
        proc = self._proc
        self._proc = None

        if proc.stdin:
            proc.stdin.close()
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            proc.kill()

        if self._stderr_thread and self._stderr_thread.is_alive():
            self._stderr_thread.join(timeout=0.5)
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=0.5)
        self._stderr_drain_done.wait(timeout=_STDERR_DRAIN_WAIT_S)

    def initialize(self) -> InitializeResponse:
        result = self.request(
            "initialize",
            {
                "clientInfo": {
                    "name": self.config.client_name,
                    "title": self.config.client_title,
                    "version": self.config.client_version,
                },
                "capabilities": {
                    "experimentalApi": self.config.experimental_api,
                },
            },
            response_model=InitializeResponse,
        )
        self.notify("initialized", None)
        return result

    def request(
        self,
        method: str,
        params: JsonObject | None,
        *,
        response_model: type[ModelT],
    ) -> ModelT:
        result = self._request_raw(method, params)
        if not isinstance(result, dict):
            raise CodexError(f"{method} response must be a JSON object")
        return response_model.model_validate(result)

    def _request_raw(self, method: str, params: JsonObject | None = None) -> JsonValue:
        """Send a JSON-RPC request and wait for the reader thread to route its response."""
        self._transport_phase = _transport_phase_for_method(method)
        request_id = str(uuid.uuid4())
        waiter = self._router.create_response_waiter(request_id)

        try:
            message: JsonObject = {"id": request_id, "method": method}
            if params is not None:
                message["params"] = params
            self._write_message(message)
        except BaseException:
            self._router.discard_response_waiter(request_id)
            raise

        item = waiter.get()
        if isinstance(item, BaseException):
            raise item
        return item

    def notify(self, method: str, params: JsonObject | None = None) -> None:
        """Send a JSON-RPC notification without waiting for a response."""
        message: JsonObject = {"method": method}
        if params is not None:
            message["params"] = params
        self._write_message(message)

    def next_notification(self) -> Notification:
        """Return the next notification that is not scoped to an active turn."""
        return self._router.next_global_notification()

    def register_login_notifications(self, login_id: str) -> None:
        """Start routing notifications for one interactive login attempt."""
        self._router.register_login(login_id)

    def unregister_login_notifications(self, login_id: str) -> None:
        """Stop routing notifications for one interactive login attempt."""
        self._router.unregister_login(login_id)

    def next_login_notification(self, login_id: str) -> Notification:
        """Return the next routed notification for the requested login id."""
        return self._router.next_login_notification(login_id)

    def register_turn_notifications(self, turn_id: str) -> None:
        """Start routing notifications for one turn into its dedicated queue."""
        self._router.register_turn(turn_id)

    def unregister_turn_notifications(self, turn_id: str) -> None:
        """Stop routing notifications for one turn into its dedicated queue."""
        self._router.unregister_turn(turn_id)

    def next_turn_notification(self, turn_id: str) -> Notification:
        """Return the next routed notification for the requested turn id."""
        return self._router.next_turn_notification(turn_id)

    def register_goal_operation(self, thread_id: str) -> _GoalOperationState:
        """Register a private thread-scoped route for a logical goal turn."""
        return self._router.register_goal(thread_id)

    def reserve_goal_operation(self, thread_id: str) -> _GoalOperationState:
        """Reserve a private thread route before replacing its stored goal."""
        return self._router.reserve_goal(thread_id)

    def unregister_goal_operation(self, state: _GoalOperationState) -> None:
        """Release routing state for one logical goal turn."""
        self._router.unregister_goal(state)

    def next_goal_notification(self, state: _GoalOperationState) -> Notification:
        """Wait for the next notification in a logical goal turn."""
        return state.next_notification()

    def account_login_start(
        self,
        params: V2LoginAccountParams | JsonObject,
    ) -> LoginAccountResponse:
        response = self.request(
            "account/login/start",
            _params_dict(params),
            response_model=LoginAccountResponse,
        )
        response_root = response.root
        if isinstance(
            response_root,
            ChatgptLoginAccountResponse | ChatgptDeviceCodeLoginAccountResponse,
        ):
            self.register_login_notifications(response_root.login_id)
        return response

    def account_login_cancel(self, login_id: str) -> CancelLoginAccountResponse:
        return self.request(
            "account/login/cancel",
            {"loginId": login_id},
            response_model=CancelLoginAccountResponse,
        )

    def account_read(
        self,
        params: V2GetAccountParams | JsonObject | None = None,
    ) -> GetAccountResponse:
        return self.request(
            "account/read",
            _params_dict(params),
            response_model=GetAccountResponse,
        )

    def account_logout(self) -> LogoutAccountResponse:
        return self.request("account/logout", None, response_model=LogoutAccountResponse)

    def thread_start(
        self, params: V2ThreadStartParams | JsonObject | None = None
    ) -> ThreadStartResponse:
        return self.request(
            "thread/start", _params_dict(params), response_model=ThreadStartResponse
        )

    def thread_resume(
        self,
        thread_id: str,
        params: V2ThreadResumeParams | JsonObject | None = None,
    ) -> ThreadResumeResponse:
        payload = {"threadId": thread_id, **_params_dict(params)}
        return self.request("thread/resume", payload, response_model=ThreadResumeResponse)

    def thread_list(
        self, params: V2ThreadListParams | JsonObject | None = None
    ) -> ThreadListResponse:
        return self.request("thread/list", _params_dict(params), response_model=ThreadListResponse)

    def thread_read(self, thread_id: str, include_turns: bool = False) -> ThreadReadResponse:
        return self.request(
            "thread/read",
            {"threadId": thread_id, "includeTurns": include_turns},
            response_model=ThreadReadResponse,
        )

    def thread_fork(
        self,
        thread_id: str,
        params: V2ThreadForkParams | JsonObject | None = None,
    ) -> ThreadForkResponse:
        payload = {"threadId": thread_id, **_params_dict(params)}
        return self.request("thread/fork", payload, response_model=ThreadForkResponse)

    def thread_archive(self, thread_id: str) -> ThreadArchiveResponse:
        return self.request(
            "thread/archive",
            {"threadId": thread_id},
            response_model=ThreadArchiveResponse,
        )

    def thread_unarchive(self, thread_id: str) -> ThreadUnarchiveResponse:
        return self.request(
            "thread/unarchive",
            {"threadId": thread_id},
            response_model=ThreadUnarchiveResponse,
        )

    def thread_set_name(self, thread_id: str, name: str) -> ThreadSetNameResponse:
        return self.request(
            "thread/name/set",
            {"threadId": thread_id, "name": name},
            response_model=ThreadSetNameResponse,
        )

    def thread_compact(self, thread_id: str) -> ThreadCompactStartResponse:
        return self.request(
            "thread/compact/start",
            {"threadId": thread_id},
            response_model=ThreadCompactStartResponse,
        )

    def thread_goal_clear(self, thread_id: str) -> ThreadGoalClearResponse:
        """Clear the persisted goal for a thread before replacing it."""
        return self.request(
            "thread/goal/clear",
            {"threadId": thread_id},
            response_model=ThreadGoalClearResponse,
        )

    def thread_goal_get(self, thread_id: str) -> ThreadGoalGetResponse:
        """读取线程当前持久化 Goal；没有 Goal 时返回 ``goal=None``。"""
        return self.request(
            "thread/goal/get",
            {"threadId": thread_id},
            response_model=ThreadGoalGetResponse,
        )

    def thread_goal_set(
        self,
        thread_id: str,
        *,
        objective: str | None = None,
        status: ThreadGoalStatus | None = None,
        token_budget: int | None = None,
    ) -> ThreadGoalSetResponse:
        """创建或更新线程 Goal，并把可选 token 预算发送给服务端。"""
        payload: JsonObject = {"threadId": thread_id}
        if objective is not None:
            payload["objective"] = objective
        if status is not None:
            payload["status"] = status.value
        if token_budget is not None:
            payload["tokenBudget"] = token_budget
        return self.request(
            "thread/goal/set",
            payload,
            response_model=ThreadGoalSetResponse,
        )

    def pause_goal(self, thread_id: str) -> ThreadGoalSetResponse:
        """Pause the active goal used by a logical goal turn."""
        return self.thread_goal_set(thread_id, status=ThreadGoalStatus.paused)

    def cancel_goal_operation(self, state: _GoalOperationState) -> None:
        """请求暂停并中断 Goal；provider 未确认或 RPC 失败时抛错。"""
        pause_response = self.pause_goal(state.thread_id)
        provider_goal = getattr(pause_response, "goal", None)
        if not isinstance(provider_goal, ThreadGoal) or provider_goal.status != ThreadGoalStatus.paused:
            raise CodexError("provider did not confirm that the Goal is paused")
        interrupt_error = self._interrupt_goal_operation(state)
        if interrupt_error is not None:
            raise interrupt_error

    def _interrupt_goal_operation(self, state: _GoalOperationState) -> BaseException | None:
        """尝试中断当前物理 turn，并返回未能确认的 RPC 错误。"""
        turn_id = state.turn_for_interrupt()
        if turn_id is None:
            return None
        try:
            self.turn_interrupt(state.thread_id, turn_id)
        except InvalidRequestError as exc:
            if not exc.message.startswith("expected active turn id"):
                return exc
            next_turn_id = _active_turn_id_from_error(exc) or state.current_turn()
            if next_turn_id is None or next_turn_id == turn_id:
                return exc
            try:
                self.turn_interrupt(state.thread_id, next_turn_id)
            except BaseException as retry_error:
                return retry_error
        except BaseException as exc:
            return exc
        return None

    def start_goal_operation(
        self,
        thread_id: str,
        objective: str,
        *,
        token_budget: int | None = None,
    ) -> tuple[_GoalOperationState, str]:
        """创建 Goal、等待首个物理 turn，并返回其可路由状态。"""
        with self._thread_start_lock(thread_id):
            return self._start_goal_operation(thread_id, objective, token_budget=token_budget)

    def _start_goal_operation(
        self,
        thread_id: str,
        objective: str,
        *,
        token_budget: int | None = None,
    ) -> tuple[_GoalOperationState, str]:
        thread = self.thread_read(thread_id).thread
        if not isinstance(thread.status.root, IdleThreadStatus):
            raise InvalidRequestError(
                -32600,
                f"thread must be idle before starting a goal: {thread_id}",
            )
        if thread.ephemeral or thread.path is None:
            raise InvalidRequestError(
                -32600,
                f"thread must be persisted before starting a goal: {thread_id}",
            )

        state = self.reserve_goal_operation(thread_id)
        activated = False
        try:
            self.thread_goal_clear(thread_id)
            state.activate_turn_routing()
            set_response = self.thread_goal_set(
                thread_id,
                objective=objective,
                status=ThreadGoalStatus.active,
                token_budget=token_budget,
            )
            provider_goal = getattr(set_response, "goal", None)
            if isinstance(provider_goal, ThreadGoal):
                bind_goal = getattr(state, "bind_goal", None)
                if bind_goal is not None:
                    bind_goal(provider_goal)
            elif isinstance(state, _GoalOperationState):
                raise CodexError("thread/goal/set did not return a ThreadGoal")
            activated = True
            turn_id = state.wait_for_start(_GOAL_START_TIMEOUT_S)
            if turn_id is None:
                raise CodexError(
                    "timed out waiting for goal turn to start after "
                    f"{int(_GOAL_START_TIMEOUT_S)} seconds"
                )
            return state, turn_id
        except BaseException as exc:
            if activated or not isinstance(exc, InvalidRequestError):
                try:
                    self.cancel_goal_operation(state)
                except BaseException:
                    pass
            state.finish()
            self.unregister_goal_operation(state)
            raise

    @staticmethod
    def goal_identity(goal: ThreadGoal) -> str:
        """返回由 thread、createdAt、objective 和 token budget 组成的稳定身份。"""
        return _goal_identity(goal)

    def resume_goal_operation(
        self,
        thread_id: str,
        *,
        checkpoint: object | None = None,
    ) -> tuple[_GoalOperationState, str | None]:
        """只读恢复既有 Goal；身份无法唯一核对时抛错且不修改服务端。"""
        state_factory = self._resume_goal_state_factory()
        state: _GoalOperationState | None = None
        history_recovery = False
        try:
            # 先占住 thread route，再读取服务端快照，避免读请求期间的物理通知
            # 落到未注册的 turn 队列或全局队列。
            if state_factory is not None:
                state = state_factory(thread_id)
                state.activate_turn_routing()
                state.begin_history_recovery()
                history_recovery = True

            response = self.thread_goal_get(thread_id)
            goal = getattr(response, "goal", None)
            if goal is None and hasattr(response, "goal"):
                raise GoalNotFoundError(f"thread {thread_id!r} has no persisted Goal")
            if not isinstance(goal, ThreadGoal):
                raise CodexError(f"thread {thread_id!r} returned an invalid Goal")
            self._validate_goal_checkpoint(thread_id, goal, checkpoint)
            history_response = self.thread_read(thread_id, include_turns=True)
            thread = history_response.thread
            turns = list(getattr(thread, "turns", []))
            physical_turn_count_before = self._checkpoint_physical_turn_count_before(checkpoint)
            if physical_turn_count_before is not None and physical_turn_count_before > len(turns):
                raise CodexError(
                    "Goal checkpoint physical turn count baseline exceeds thread history"
                )
            physical_turn_ids = self._checkpoint_physical_turn_ids(checkpoint)
            if physical_turn_ids:
                physical_turn_ids = self._extend_physical_turn_history(
                    physical_turn_ids,
                    turns,
                    baseline=physical_turn_count_before,
                )
            elif physical_turn_count_before is not None:
                # baseline 是调用方在 Goal 首次 set 前保存的完整历史长度；只把其后的
                # 连续尾部视为该 Goal，旧 turn 不会被误接到新的 Goal 上。
                self._validate_physical_turn_history((), turns)
                physical_turn_ids = tuple(
                    getattr(turn, "id") for turn in turns[physical_turn_count_before:]
                )
            else:
                if len(turns) != 1:
                    raise CodexError(
                        "cannot uniquely resume Goal without persisted physical turn ids"
                    )
                physical_turn_ids = (turns[0].id,)
            self._validate_physical_turn_history(physical_turn_ids, turns)
            selected_turns = [turn for turn in turns if turn.id in physical_turn_ids]
            observed_notifications = (
                state.consume_observed_notifications() if state is not None else ()
            )
            selected_turns = self._merge_resume_history(
                selected_turns,
                turns,
                observed_notifications,
            )
            physical_turn_ids = tuple(turn.id for turn in selected_turns)
            self._validate_goal_turn_sequence(selected_turns)
            expected_turn_id = getattr(checkpoint, "logical_turn_id", None)
            accepted_turn_identities = {_goal_identity(goal)}
            if physical_turn_ids:
                accepted_turn_identities.add(physical_turn_ids[0])
            if expected_turn_id is not None and expected_turn_id not in accepted_turn_identities:
                raise CodexError("Goal checkpoint logical turn identity does not match history")
            if (
                state is not None
                and state.logical_turn_id is not None
                and physical_turn_ids
                and state.logical_turn_id != physical_turn_ids[0]
            ):
                raise CodexError("resume notifications do not match the first physical turn")
            if state is None:
                state = _GoalOperationState(thread_id=thread_id)
            state.bind_goal(goal, physical_turn_ids=physical_turn_ids)
            state.seed_history(selected_turns)
            state.reconcile_history_notifications(
                set(physical_turn_ids),
                drop_history_events=_terminal_goal_status(goal.status),
            )
            if _terminal_goal_status(goal.status):
                state.replay_history_notifications()
            if history_recovery:
                state.finish_history_recovery()
                history_recovery = False
            first_turn_id = state.logical_turn_id
            if first_turn_id is None:
                if _terminal_goal_status(goal.status):
                    raise CodexError(
                        "resumed terminal Goal has no uniquely identified physical turn"
                    )
                # Goal 已持久化但 provider 尚未产生 TurnStarted；调用方可以先拿到可
                # stream 的句柄，等首个通知绑定真实物理 turn 身份。
                return state, None
            return state, first_turn_id
        except BaseException:
            if state is not None and history_recovery:
                state.finish_history_recovery()
            if state is not None and state_factory is not None:
                self.unregister_goal_operation(state)
            raise

    def _resume_goal_state_factory(
        self,
    ) -> Callable[[str], _GoalOperationState] | None:
        """判断恢复是否已有可用 route，并支持测试替换的 reserve 方法。"""
        if hasattr(self, "_router") or "reserve_goal_operation" in self.__dict__:
            return self.reserve_goal_operation
        return None

    @staticmethod
    def _merge_resume_history(
        selected_turns: list[object],
        full_history: list[object],
        observed_notifications: tuple[Notification, ...],
    ) -> list[object]:
        """把读请求期间观察到的连续 turn 事件并入历史尾部并拒绝漂移。"""
        selected_ids = {getattr(turn, "id") for turn in selected_turns}
        full_ids = {getattr(turn, "id") for turn in full_history}
        by_id = {getattr(turn, "id"): index for index, turn in enumerate(selected_turns)}
        observed_started: set[str] = set()
        for notification in observed_notifications:
            payload = notification.payload
            if isinstance(payload, TurnStartedNotification):
                turn_id = payload.turn.id
                if turn_id in full_ids and turn_id not in selected_ids:
                    raise CodexError("resume observed a physical turn outside the Goal history")
                if turn_id not in by_id:
                    by_id[turn_id] = len(selected_turns)
                    selected_turns.append(payload.turn)
                observed_started.add(turn_id)
                continue
            if not isinstance(payload, TurnCompletedNotification):
                continue
            if payload.turn.status == TurnStatus.in_progress:
                raise CodexError("turn/completed notification cannot have inProgress status")
            turn_id = payload.turn.id
            if turn_id not in by_id:
                if turn_id not in observed_started:
                    raise CodexError(
                        "resume observed a physical completion without a matching start"
                    )
                raise CodexError("resume observed an unknown physical turn")
            index = by_id[turn_id]
            existing = selected_turns[index]
            if existing.status == TurnStatus.in_progress:
                selected_turns[index] = payload.turn
            elif existing.status != payload.turn.status:
                raise CodexError("resume history and notification statuses disagree")
        return selected_turns

    @staticmethod
    def _validate_goal_turn_sequence(turns: list[object]) -> None:
        """校验恢复尾部的物理 turn 身份和状态按顺序且只允许末尾活动。"""
        ids = [getattr(turn, "id", None) for turn in turns]
        if any(not isinstance(turn_id, str) or not turn_id for turn_id in ids):
            raise CodexError("Goal history contains a turn without a valid physical id")
        if len(set(ids)) != len(ids):
            raise CodexError("Goal history contains duplicate physical turn ids")
        in_progress = [turn for turn in turns if turn.status == TurnStatus.in_progress]
        if len(in_progress) > 1 or (in_progress and turns[-1] is not in_progress[0]):
            raise CodexError("in-progress physical turn is not the final history turn")

    def _validate_goal_checkpoint(
        self,
        thread_id: str,
        goal: ThreadGoal,
        checkpoint: object | None,
    ) -> None:
        """核对调用方保存的逻辑身份和服务端 Goal 稳定字段。"""
        if checkpoint is None:
            return
        checkpoint_thread = getattr(checkpoint, "thread_id", None)
        if checkpoint_thread is not None and checkpoint_thread != thread_id:
            raise CodexError("Goal checkpoint thread identity does not match")
        expected_identity = getattr(checkpoint, "logical_goal_id", None)
        if expected_identity is None:
            expected_identity = getattr(checkpoint, "goal_id", None)
        if expected_identity is not None and expected_identity != _goal_identity(goal):
            raise CodexError("Goal checkpoint logical identity does not match")
        for name, actual in (
            ("objective", goal.objective),
            ("token_budget", goal.token_budget),
            ("provider_created_at", goal.created_at),
        ):
            expected = getattr(checkpoint, name, None)
            if expected is not None and expected != actual:
                raise CodexError(f"Goal checkpoint {name} does not match provider Goal")

    @staticmethod
    def _checkpoint_physical_turn_ids(checkpoint: object | None) -> tuple[str, ...]:
        """提取 checkpoint 的有序物理 turn 身份并拒绝重复值。"""
        if checkpoint is None:
            return ()
        raw = getattr(checkpoint, "physical_turn_ids", ())
        if raw is None:
            return ()
        if isinstance(raw, str) or not isinstance(raw, (tuple, list)):
            raise CodexError("Goal checkpoint physical turn ids must be an ordered sequence")
        values = tuple(raw)
        if any(not isinstance(value, str) or not value for value in values):
            raise CodexError("Goal checkpoint contains an invalid physical turn id")
        if len(set(values)) != len(values):
            raise CodexError("Goal checkpoint contains duplicate physical turn ids")
        return values

    @staticmethod
    def _checkpoint_physical_turn_count_before(checkpoint: object | None) -> int | None:
        """读取首次 Goal turn 前的历史长度，并拒绝无法核对的基线。"""
        if checkpoint is None:
            return None
        value = getattr(checkpoint, "physical_turn_count_before", None)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CodexError(
                "Goal checkpoint physical turn count baseline must be a non-negative integer"
            )
        return value

    @staticmethod
    def _extend_physical_turn_history(
        checkpoint_ids: tuple[str, ...],
        turns: list[object],
        *,
        baseline: int | None = None,
    ) -> tuple[str, ...]:
        """从 checkpoint 首个 turn 起纳入历史尾部，并拒绝不连续前缀。"""
        history_ids = tuple(getattr(turn, "id", None) for turn in turns)
        if any(not isinstance(turn_id, str) or not turn_id for turn_id in history_ids):
            raise CodexError("thread history contains a turn without a valid physical id")
        if len(set(history_ids)) != len(history_ids):
            raise CodexError("thread history contains duplicate physical turn ids")
        if baseline is not None:
            if baseline >= len(history_ids) or history_ids[baseline] != checkpoint_ids[0]:
                raise CodexError("Goal checkpoint baseline/prefix has drifted")
        try:
            start = history_ids.index(checkpoint_ids[0])
        except ValueError as exc:
            raise CodexError(
                f"unknown physical turn id in Goal checkpoint: {checkpoint_ids[0]}"
            ) from exc
        end = start + len(checkpoint_ids)
        if tuple(history_ids[start:end]) != checkpoint_ids:
            raise CodexError("Goal checkpoint physical turn prefix is not contiguous")
        return history_ids[start:]

    @staticmethod
    def _validate_physical_turn_history(
        physical_turn_ids: tuple[str, ...],
        turns: list[object],
    ) -> None:
        """要求保存的物理 turn 身份是历史中的连续有序子序列。"""
        history_ids = tuple(getattr(turn, "id", None) for turn in turns)
        if any(not isinstance(turn_id, str) or not turn_id for turn_id in history_ids):
            raise CodexError("thread history contains a turn without a valid physical id")
        if len(set(history_ids)) != len(history_ids):
            raise CodexError("thread history contains duplicate physical turn ids")
        positions: list[int] = []
        for turn_id in physical_turn_ids:
            try:
                position = history_ids.index(turn_id)
            except ValueError as exc:
                raise CodexError(f"unknown physical turn id in Goal checkpoint: {turn_id}") from exc
            if positions and position <= positions[-1]:
                raise CodexError("Goal checkpoint physical turn ids are out of order")
            if positions and position != positions[-1] + 1:
                raise CodexError("Goal checkpoint physical turn ids are not contiguous")
            positions.append(position)

    def turn_start(
        self,
        thread_id: str,
        input_items: list[JsonObject] | JsonObject | str,
        params: V2TurnStartParams | JsonObject | None = None,
    ) -> TurnStartResponse:
        """Start a turn and register its notification queue as early as possible."""
        with self._thread_start_lock(thread_id):
            if self._router.has_goal(thread_id):
                raise InvalidRequestError(
                    -32600,
                    f"thread has an active goal operation: {thread_id}",
                )
            payload = {
                **_params_dict(params),
                "threadId": thread_id,
                "input": self._normalize_input_items(input_items),
            }
            started = self.request("turn/start", payload, response_model=TurnStartResponse)
            self.register_turn_notifications(started.turn.id)
            return started

    @contextmanager
    def _thread_start_lock(self, thread_id: str) -> Iterator[None]:
        with self._thread_start_locks_guard:
            entry = self._thread_start_locks.get(thread_id)
            if entry is None:
                entry = _ThreadStartLock()
                self._thread_start_locks[thread_id] = entry
            entry.users += 1
        try:
            with entry.lock:
                yield
        finally:
            with self._thread_start_locks_guard:
                entry.users -= 1
                if entry.users == 0:
                    self._thread_start_locks.pop(thread_id, None)

    def turn_interrupt(self, thread_id: str, turn_id: str) -> TurnInterruptResponse:
        return self.request(
            "turn/interrupt",
            {"threadId": thread_id, "turnId": turn_id},
            response_model=TurnInterruptResponse,
        )

    def turn_steer(
        self,
        thread_id: str,
        expected_turn_id: str,
        input_items: list[JsonObject] | JsonObject | str,
    ) -> TurnSteerResponse:
        return self.request(
            "turn/steer",
            {
                "threadId": thread_id,
                "expectedTurnId": expected_turn_id,
                "input": self._normalize_input_items(input_items),
            },
            response_model=TurnSteerResponse,
        )

    def model_list(self, include_hidden: bool = False) -> ModelListResponse:
        return self.request(
            "model/list",
            {"includeHidden": include_hidden},
            response_model=ModelListResponse,
        )

    def request_with_retry_on_overload(
        self,
        method: str,
        params: JsonObject | None,
        *,
        response_model: type[ModelT],
        max_attempts: int = 3,
        initial_delay_s: float = 0.25,
        max_delay_s: float = 2.0,
    ) -> ModelT:
        return retry_on_overload(
            lambda: self.request(method, params, response_model=response_model),
            max_attempts=max_attempts,
            initial_delay_s=initial_delay_s,
            max_delay_s=max_delay_s,
        )

    def wait_for_turn_completed(self, turn_id: str) -> TurnCompletedNotification:
        """Block on the routed turn stream until the matching completion arrives."""
        self.register_turn_notifications(turn_id)
        try:
            while True:
                notification = self.next_turn_notification(turn_id)
                if (
                    notification.method == "turn/completed"
                    and isinstance(notification.payload, TurnCompletedNotification)
                    and notification.payload.turn.id == turn_id
                ):
                    return notification.payload
        finally:
            self.unregister_turn_notifications(turn_id)

    def wait_for_login_completed(
        self,
        login_id: str,
    ) -> AccountLoginCompletedNotification:
        """Block until the matching interactive login attempt completes."""
        self.register_login_notifications(login_id)
        try:
            while True:
                notification = self.next_login_notification(login_id)
                if (
                    notification.method == "account/login/completed"
                    and isinstance(notification.payload, AccountLoginCompletedNotification)
                    and notification.payload.login_id == login_id
                ):
                    return notification.payload
        finally:
            self.unregister_login_notifications(login_id)

    def stream_text(
        self,
        thread_id: str,
        text: str,
        params: V2TurnStartParams | JsonObject | None = None,
    ) -> Iterator[AgentMessageDeltaNotification]:
        """Start a text turn and yield only its agent-message delta payloads."""
        started = self.turn_start(thread_id, text, params=params)
        turn_id = started.turn.id
        self.register_turn_notifications(turn_id)
        try:
            while True:
                notification = self.next_turn_notification(turn_id)
                if (
                    notification.method == "item/agentMessage/delta"
                    and isinstance(notification.payload, AgentMessageDeltaNotification)
                    and notification.payload.turn_id == turn_id
                ):
                    yield notification.payload
                    continue
                if (
                    notification.method == "turn/completed"
                    and isinstance(notification.payload, TurnCompletedNotification)
                    and notification.payload.turn.id == turn_id
                ):
                    break
        finally:
            self.unregister_turn_notifications(turn_id)

    def _coerce_notification(self, method: str, params: object) -> Notification:
        params_dict = params if isinstance(params, dict) else {}

        model = NOTIFICATION_MODELS.get(method)
        if model is None:
            return Notification(method=method, payload=UnknownNotification(params=params_dict))

        try:
            payload = model.model_validate(params_dict)
        except Exception:  # noqa: BLE001
            return Notification(method=method, payload=UnknownNotification(params=params_dict))
        return Notification(method=method, payload=payload)

    def _normalize_input_items(
        self,
        input_items: list[JsonObject] | JsonObject | str,
    ) -> list[JsonObject]:
        if isinstance(input_items, str):
            return [{"type": "text", "text": input_items}]
        if isinstance(input_items, dict):
            return [input_items]
        return input_items

    def _default_approval_handler(self, method: str, params: JsonObject | None) -> JsonObject:
        """Accept approval requests when the caller did not provide a handler."""
        if method == "item/commandExecution/requestApproval":
            return {"decision": "accept"}
        if method == "item/fileChange/requestApproval":
            return {"decision": "accept"}
        return {}

    def _start_stderr_drain_thread(self) -> None:
        if self._proc is None or self._proc.stderr is None:
            self._stderr_drain_done.set()
            return

        proc = self._proc
        stderr = proc.stderr
        self._stderr_drain_done.clear()

        def _drain() -> None:
            try:
                for line in stderr:
                    indicator = _classify_stderr_indicator(line)
                    if indicator is StderrIndicator.NONE:
                        continue
                    with self._stderr_indicator_lock:
                        # 只保留一个固定枚举；stderr 行本身不进入对象状态。
                        if self._stderr_indicator is StderrIndicator.NONE:
                            self._stderr_indicator = indicator
            finally:
                self._stderr_drain_done.set()

        self._stderr_thread = threading.Thread(target=_drain, daemon=True)
        self._stderr_thread.start()

    def _start_reader_thread(self) -> None:
        """Start the sole stdout reader that fans messages into router queues."""
        if self._proc is None or self._proc.stdout is None:
            return

        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

    def _reader_loop(self) -> None:
        """Continuously classify transport messages into requests, responses, and events."""
        try:
            while True:
                msg = self._read_message()
                if "method" in msg and "id" in msg:
                    response = self._handle_server_request(msg)
                    self._write_message({"id": msg["id"], "result": response})
                    continue
                if "method" in msg and "id" not in msg:
                    method = msg["method"]
                    if isinstance(method, str):
                        self._router.route_notification(
                            self._coerce_notification(method, msg.get("params"))
                        )
                    continue
                self._router.route_response(msg)
        except BaseException as exc:
            self._router.fail_all(exc)

    def _transport_diagnostic(
        self,
        *,
        process: subprocess.Popen[str] | None,
        stdout_eof: bool,
        phase: TransportPhase | None = None,
    ) -> TransportDiagnostic:
        """读取退出状态并构造不含 stderr 原文的固定诊断。

        输入是可选进程、EOF 标记和可选阶段；动作只读取进程退出码及已归类的
        stderr 枚举，不修改进程。返回 ``TransportDiagnostic``；进程 ``poll`` 失败
        时把退出码视为未知，不向上传播读取异常。
        """

        returncode: int | None = None
        if process is not None:
            try:
                candidate = process.poll()
            except BaseException:
                candidate = None
            if isinstance(candidate, int) and not isinstance(candidate, bool):
                returncode = candidate
        termination_signal = (
            -returncode if returncode is not None and returncode < 0 else None
        )
        with self._stderr_indicator_lock:
            indicator = self._stderr_indicator
        return TransportDiagnostic(
            phase=phase or self._transport_phase,
            stdout_eof=stdout_eof,
            returncode=returncode,
            termination_signal=termination_signal,
            stderr_indicator=indicator,
        )

    def _wait_for_process_exit_and_stderr(
        self,
        process: subprocess.Popen[str],
    ) -> None:
        """在 stdout EOF 后短暂等待子进程和 stderr drain，消除读取竞态。

        输入是已出现传输异常的子进程；动作最多各等待固定时长，副作用仅为等待
        进程和 drain 线程完成。返回 ``None``；超时或 OS 等待错误会被吸收，由随后
        的结构化诊断以未知退出状态表示。
        """

        try:
            if process.poll() is None:
                process.wait(timeout=_STDERR_DRAIN_WAIT_S)
        except (subprocess.TimeoutExpired, OSError):
            pass
        self._stderr_drain_done.wait(timeout=_STDERR_DRAIN_WAIT_S)

    def _handle_server_request(self, msg: dict[str, JsonValue]) -> JsonObject:
        method = msg["method"]
        params = msg.get("params")
        if not isinstance(method, str):
            return {}
        return self._approval_handler(
            method,
            params if isinstance(params, dict) else None,
        )

    def _write_message(self, payload: JsonObject) -> None:
        process = self._proc
        if process is None or process.stdin is None:
            raise TransportClosedError(
                "Codex process is not running",
                diagnostic=self._transport_diagnostic(
                    process=None,
                    stdout_eof=False,
                    phase=TransportPhase.PROCESS_NOT_RUNNING,
                ),
            )
        with self._lock:
            try:
                process.stdin.write(json.dumps(payload) + "\n")
                process.stdin.flush()
            except (BrokenPipeError, OSError, ValueError):
                self._wait_for_process_exit_and_stderr(process)
                raise TransportClosedError(
                    "Codex process is not running",
                    diagnostic=self._transport_diagnostic(
                        process=process,
                        stdout_eof=False,
                        phase=self._transport_phase,
                    ),
                ) from None

    def _read_message(self) -> dict[str, JsonValue]:
        process = self._proc
        if process is None or process.stdout is None:
            raise TransportClosedError(
                "Codex process is not running",
                diagnostic=self._transport_diagnostic(
                    process=None,
                    stdout_eof=False,
                    phase=TransportPhase.PROCESS_NOT_RUNNING,
                ),
            )

        line = process.stdout.readline()
        if not line:
            self._wait_for_process_exit_and_stderr(process)
            raise TransportClosedError(
                "Codex process closed stdout",
                diagnostic=self._transport_diagnostic(
                    process=process,
                    stdout_eof=True,
                    phase=self._transport_phase,
                ),
            )

        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CodexError(f"Invalid JSON-RPC line: {line!r}") from exc

        if not isinstance(message, dict):
            raise CodexError(f"Invalid JSON-RPC payload: {message!r}")
        return message


def default_codex_home() -> str:
    return str(Path.home() / ".codex")
