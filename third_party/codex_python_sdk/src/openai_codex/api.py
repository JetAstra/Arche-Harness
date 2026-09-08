from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from typing import AsyncIterator, Iterator

from ._approval_mode import (
    ApprovalMode as ApprovalMode,
    _approval_mode_override_settings,
    _approval_mode_settings,
)
from ._initialize_metadata import validate_initialize_metadata
from ._inputs import (
    ImageInput as ImageInput,
    Input as Input,
    InputItem as InputItem,
    LocalImageInput as LocalImageInput,
    MentionInput as MentionInput,
    RunInput,
    SkillInput as SkillInput,
    TextInput as TextInput,
    _normalize_run_input,
    _to_wire_input,
)
from ._login import (
    AsyncChatgptLoginHandle,
    AsyncDeviceCodeLoginHandle,
    ChatgptLoginHandle,
    DeviceCodeLoginHandle,
    async_start_chatgpt_login,
    async_start_device_code_login,
    start_chatgpt_login,
    start_device_code_login,
)
from ._run import (
    TurnResult,
    _collect_async_turn_result,
    _collect_turn_result,
)
from ._goal import (
    _AsyncGoalNotificationStream,
    _GoalNotificationStream,
    _GoalOperationState,
)
from ._sandbox import Sandbox as Sandbox, _sandbox_mode, _sandbox_policy
from .async_client import AsyncCodexClient
from .client import CodexClient, CodexConfig
from .generated.v2_all import (
    ApiKeyLoginAccountParams,
    GetAccountParams,
    GetAccountResponse,
    LoginAccountParams,
    ModelListResponse,
    Personality,
    ReasoningEffort,
    ReasoningSummary,
    SortDirection,
    ThreadArchiveResponse,
    ThreadCompactStartResponse,
    ThreadForkParams,
    ThreadListCwdFilter,
    ThreadListParams,
    ThreadListResponse,
    ThreadReadResponse,
    ThreadResumeParams,
    ThreadSetNameResponse,
    ThreadSortKey,
    ThreadSource,
    ThreadSourceKind,
    ThreadStartParams,
    ThreadStartSource,
    TurnCompletedNotification,
    TurnInterruptResponse,
    TurnStartParams,
    TurnSteerResponse,
    ThreadGoalStatus,
    TurnStatus,
)
from .models import InitializeResponse, JsonObject, Notification
from .errors import CodexError


class Codex:
    """Synchronous client for creating threads and running Codex turns.

    The client starts its runtime connection during construction. Use it as a
    context manager so resources are closed promptly.
    """

    def __init__(self, config: CodexConfig | None = None) -> None:
        self._client = CodexClient(config=config)
        try:
            self._client.start()
            self._init = validate_initialize_metadata(self._client.initialize())
        except Exception:
            self._client.close()
            raise

    def __enter__(self) -> "Codex":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()

    @property
    def metadata(self) -> InitializeResponse:
        return self._init

    def close(self) -> None:
        self._client.close()

    def login_api_key(self, api_key: str) -> None:
        """Authenticate Codex with an API key."""
        self._client.account_login_start(
            LoginAccountParams(
                root=ApiKeyLoginAccountParams(
                    api_key=api_key,
                    type="apiKey",
                )
            )
        )

    def login_chatgpt(self) -> ChatgptLoginHandle:
        """Start browser-based ChatGPT login and return its live handle."""
        return start_chatgpt_login(self._client)

    def login_chatgpt_device_code(self) -> DeviceCodeLoginHandle:
        """Start device-code ChatGPT login and return its live handle."""
        return start_device_code_login(self._client)

    def account(self, *, refresh_token: bool = False) -> GetAccountResponse:
        """Read the current Codex account state."""
        return self._client.account_read(GetAccountParams(refresh_token=refresh_token))

    def logout(self) -> None:
        """Clear the current Codex account session."""
        self._client.account_logout()

    # BEGIN GENERATED: Codex.flat_methods
    def thread_start(
        self,
        *,
        approval_mode: ApprovalMode = ApprovalMode.auto_review,
        base_instructions: str | None = None,
        config: JsonObject | None = None,
        cwd: str | None = None,
        developer_instructions: str | None = None,
        ephemeral: bool | None = None,
        model: str | None = None,
        model_provider: str | None = None,
        personality: Personality | None = None,
        sandbox: Sandbox | None = None,
        service_name: str | None = None,
        service_tier: str | None = None,
        session_start_source: ThreadStartSource | None = None,
        thread_source: ThreadSource | None = None,
    ) -> Thread:
        """Create a new Codex conversation thread."""
        approval_policy, approvals_reviewer = _approval_mode_settings(approval_mode)
        params = ThreadStartParams(
            approval_policy=approval_policy,
            approvals_reviewer=approvals_reviewer,
            base_instructions=base_instructions,
            config=config,
            cwd=cwd,
            developer_instructions=developer_instructions,
            ephemeral=ephemeral,
            model=model,
            model_provider=model_provider,
            personality=personality,
            sandbox=_sandbox_mode(sandbox),
            service_name=service_name,
            service_tier=service_tier,
            session_start_source=session_start_source,
            thread_source=thread_source,
        )
        started = self._client.thread_start(params)
        return Thread(self._client, started.thread.id)

    def thread_list(
        self,
        *,
        archived: bool | None = None,
        cursor: str | None = None,
        cwd: ThreadListCwdFilter | None = None,
        limit: int | None = None,
        model_providers: list[str] | None = None,
        search_term: str | None = None,
        section_id: str | None = None,
        sort_direction: SortDirection | None = None,
        sort_key: ThreadSortKey | None = None,
        source_kinds: list[ThreadSourceKind] | None = None,
        use_state_db_only: bool | None = None,
    ) -> ThreadListResponse:
        """List saved conversation threads."""
        params = ThreadListParams(
            archived=archived,
            cursor=cursor,
            cwd=cwd,
            limit=limit,
            model_providers=model_providers,
            search_term=search_term,
            section_id=section_id,
            sort_direction=sort_direction,
            sort_key=sort_key,
            source_kinds=source_kinds,
            use_state_db_only=use_state_db_only,
        )
        return self._client.thread_list(params)

    def thread_resume(
        self,
        thread_id: str,
        *,
        approval_mode: ApprovalMode | None = None,
        base_instructions: str | None = None,
        config: JsonObject | None = None,
        cwd: str | None = None,
        developer_instructions: str | None = None,
        model: str | None = None,
        model_provider: str | None = None,
        personality: Personality | None = None,
        sandbox: Sandbox | None = None,
        service_tier: str | None = None,
    ) -> Thread:
        """Resume an existing conversation thread by ID."""
        approval_policy, approvals_reviewer = _approval_mode_override_settings(approval_mode)
        params = ThreadResumeParams(
            thread_id=thread_id,
            approval_policy=approval_policy,
            approvals_reviewer=approvals_reviewer,
            base_instructions=base_instructions,
            config=config,
            cwd=cwd,
            developer_instructions=developer_instructions,
            model=model,
            model_provider=model_provider,
            personality=personality,
            sandbox=_sandbox_mode(sandbox),
            service_tier=service_tier,
        )
        resumed = self._client.thread_resume(thread_id, params)
        return Thread(self._client, resumed.thread.id)

    def thread_fork(
        self,
        thread_id: str,
        *,
        approval_mode: ApprovalMode | None = None,
        base_instructions: str | None = None,
        config: JsonObject | None = None,
        cwd: str | None = None,
        developer_instructions: str | None = None,
        ephemeral: bool | None = None,
        model: str | None = None,
        model_provider: str | None = None,
        sandbox: Sandbox | None = None,
        service_tier: str | None = None,
        thread_source: ThreadSource | None = None,
    ) -> Thread:
        """Create a new thread from an existing thread."""
        approval_policy, approvals_reviewer = _approval_mode_override_settings(approval_mode)
        params = ThreadForkParams(
            thread_id=thread_id,
            approval_policy=approval_policy,
            approvals_reviewer=approvals_reviewer,
            base_instructions=base_instructions,
            config=config,
            cwd=cwd,
            developer_instructions=developer_instructions,
            ephemeral=ephemeral,
            model=model,
            model_provider=model_provider,
            sandbox=_sandbox_mode(sandbox),
            service_tier=service_tier,
            thread_source=thread_source,
        )
        forked = self._client.thread_fork(thread_id, params)
        return Thread(self._client, forked.thread.id)

    def thread_archive(self, thread_id: str) -> ThreadArchiveResponse:
        """Archive a stored conversation thread."""
        return self._client.thread_archive(thread_id)

    def thread_unarchive(self, thread_id: str) -> Thread:
        """Restore an archived conversation thread."""
        unarchived = self._client.thread_unarchive(thread_id)
        return Thread(self._client, unarchived.thread.id)

    # END GENERATED: Codex.flat_methods

    def models(self, *, include_hidden: bool = False) -> ModelListResponse:
        """List available models reported by Codex."""
        return self._client.model_list(include_hidden=include_hidden)


class AsyncCodex:
    """Async mirror of :class:`Codex`.

    Prefer ``async with AsyncCodex()`` so initialization and shutdown are
    explicit and paired. The async client initializes lazily on context entry
    or first awaited API use.
    """

    def __init__(self, config: CodexConfig | None = None) -> None:
        self._client = AsyncCodexClient(config=config)
        self._init: InitializeResponse | None = None
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def __aenter__(self) -> "AsyncCodex":
        await self._ensure_initialized()
        return self

    async def __aexit__(self, _exc_type, _exc, _tb) -> None:
        await self.close()

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            try:
                await self._client.start()
                payload = await self._client.initialize()
                self._init = validate_initialize_metadata(payload)
                self._initialized = True
            except Exception:
                await self._client.close()
                self._init = None
                self._initialized = False
                raise

    @property
    def metadata(self) -> InitializeResponse:
        if self._init is None:
            raise RuntimeError(
                "AsyncCodex is not initialized yet. Prefer `async with AsyncCodex()`; "
                "initialization also happens on first awaited API use."
            )
        return self._init

    async def close(self) -> None:
        await self._client.close()
        self._init = None
        self._initialized = False

    async def login_api_key(self, api_key: str) -> None:
        """Authenticate Codex with an API key."""
        await self._ensure_initialized()
        await self._client.account_login_start(
            LoginAccountParams(
                root=ApiKeyLoginAccountParams(
                    api_key=api_key,
                    type="apiKey",
                )
            )
        )

    async def login_chatgpt(self) -> AsyncChatgptLoginHandle:
        """Start browser-based ChatGPT login and return its live handle."""
        await self._ensure_initialized()
        return await async_start_chatgpt_login(self)

    async def login_chatgpt_device_code(self) -> AsyncDeviceCodeLoginHandle:
        """Start device-code ChatGPT login and return its live handle."""
        await self._ensure_initialized()
        return await async_start_device_code_login(self)

    async def account(self, *, refresh_token: bool = False) -> GetAccountResponse:
        """Read the current Codex account state."""
        await self._ensure_initialized()
        return await self._client.account_read(GetAccountParams(refresh_token=refresh_token))

    async def logout(self) -> None:
        """Clear the current Codex account session."""
        await self._ensure_initialized()
        await self._client.account_logout()

    # BEGIN GENERATED: AsyncCodex.flat_methods
    async def thread_start(
        self,
        *,
        approval_mode: ApprovalMode = ApprovalMode.auto_review,
        base_instructions: str | None = None,
        config: JsonObject | None = None,
        cwd: str | None = None,
        developer_instructions: str | None = None,
        ephemeral: bool | None = None,
        model: str | None = None,
        model_provider: str | None = None,
        personality: Personality | None = None,
        sandbox: Sandbox | None = None,
        service_name: str | None = None,
        service_tier: str | None = None,
        session_start_source: ThreadStartSource | None = None,
        thread_source: ThreadSource | None = None,
    ) -> AsyncThread:
        """Create a new Codex conversation thread."""
        await self._ensure_initialized()
        approval_policy, approvals_reviewer = _approval_mode_settings(approval_mode)
        params = ThreadStartParams(
            approval_policy=approval_policy,
            approvals_reviewer=approvals_reviewer,
            base_instructions=base_instructions,
            config=config,
            cwd=cwd,
            developer_instructions=developer_instructions,
            ephemeral=ephemeral,
            model=model,
            model_provider=model_provider,
            personality=personality,
            sandbox=_sandbox_mode(sandbox),
            service_name=service_name,
            service_tier=service_tier,
            session_start_source=session_start_source,
            thread_source=thread_source,
        )
        started = await self._client.thread_start(params)
        return AsyncThread(self, started.thread.id)

    async def thread_list(
        self,
        *,
        archived: bool | None = None,
        cursor: str | None = None,
        cwd: ThreadListCwdFilter | None = None,
        limit: int | None = None,
        model_providers: list[str] | None = None,
        search_term: str | None = None,
        section_id: str | None = None,
        sort_direction: SortDirection | None = None,
        sort_key: ThreadSortKey | None = None,
        source_kinds: list[ThreadSourceKind] | None = None,
        use_state_db_only: bool | None = None,
    ) -> ThreadListResponse:
        """List saved conversation threads."""
        await self._ensure_initialized()
        params = ThreadListParams(
            archived=archived,
            cursor=cursor,
            cwd=cwd,
            limit=limit,
            model_providers=model_providers,
            search_term=search_term,
            section_id=section_id,
            sort_direction=sort_direction,
            sort_key=sort_key,
            source_kinds=source_kinds,
            use_state_db_only=use_state_db_only,
        )
        return await self._client.thread_list(params)

    async def thread_resume(
        self,
        thread_id: str,
        *,
        approval_mode: ApprovalMode | None = None,
        base_instructions: str | None = None,
        config: JsonObject | None = None,
        cwd: str | None = None,
        developer_instructions: str | None = None,
        model: str | None = None,
        model_provider: str | None = None,
        personality: Personality | None = None,
        sandbox: Sandbox | None = None,
        service_tier: str | None = None,
    ) -> AsyncThread:
        """Resume an existing conversation thread by ID."""
        await self._ensure_initialized()
        approval_policy, approvals_reviewer = _approval_mode_override_settings(approval_mode)
        params = ThreadResumeParams(
            thread_id=thread_id,
            approval_policy=approval_policy,
            approvals_reviewer=approvals_reviewer,
            base_instructions=base_instructions,
            config=config,
            cwd=cwd,
            developer_instructions=developer_instructions,
            model=model,
            model_provider=model_provider,
            personality=personality,
            sandbox=_sandbox_mode(sandbox),
            service_tier=service_tier,
        )
        resumed = await self._client.thread_resume(thread_id, params)
        return AsyncThread(self, resumed.thread.id)

    async def thread_fork(
        self,
        thread_id: str,
        *,
        approval_mode: ApprovalMode | None = None,
        base_instructions: str | None = None,
        config: JsonObject | None = None,
        cwd: str | None = None,
        developer_instructions: str | None = None,
        ephemeral: bool | None = None,
        model: str | None = None,
        model_provider: str | None = None,
        sandbox: Sandbox | None = None,
        service_tier: str | None = None,
        thread_source: ThreadSource | None = None,
    ) -> AsyncThread:
        """Create a new thread from an existing thread."""
        await self._ensure_initialized()
        approval_policy, approvals_reviewer = _approval_mode_override_settings(approval_mode)
        params = ThreadForkParams(
            thread_id=thread_id,
            approval_policy=approval_policy,
            approvals_reviewer=approvals_reviewer,
            base_instructions=base_instructions,
            config=config,
            cwd=cwd,
            developer_instructions=developer_instructions,
            ephemeral=ephemeral,
            model=model,
            model_provider=model_provider,
            sandbox=_sandbox_mode(sandbox),
            service_tier=service_tier,
            thread_source=thread_source,
        )
        forked = await self._client.thread_fork(thread_id, params)
        return AsyncThread(self, forked.thread.id)

    async def thread_archive(self, thread_id: str) -> ThreadArchiveResponse:
        """Archive a stored conversation thread."""
        await self._ensure_initialized()
        return await self._client.thread_archive(thread_id)

    async def thread_unarchive(self, thread_id: str) -> AsyncThread:
        """Restore an archived conversation thread."""
        await self._ensure_initialized()
        unarchived = await self._client.thread_unarchive(thread_id)
        return AsyncThread(self, unarchived.thread.id)

    # END GENERATED: AsyncCodex.flat_methods

    async def models(self, *, include_hidden: bool = False) -> ModelListResponse:
        await self._ensure_initialized()
        return await self._client.model_list(include_hidden=include_hidden)


@dataclass(frozen=True, slots=True)
class GoalCheckpoint:
    """调用方持久化的 Goal 身份和有序物理 turn 校验资料。"""

    logical_turn_id: str | None = None
    logical_goal_id: str | None = None
    thread_id: str | None = None
    physical_turn_count_before: int | None = None
    physical_turn_ids: tuple[str, ...] = ()
    objective: str | None = None
    token_budget: int | None = None
    provider_created_at: int | None = None

    def __post_init__(self) -> None:
        """把调用方传入的物理 turn 序列冻结为不可变 tuple。"""
        if isinstance(self.physical_turn_ids, str):
            raise TypeError("physical_turn_ids must be an ordered sequence, not a string")
        object.__setattr__(self, "physical_turn_ids", tuple(self.physical_turn_ids or ()))

    @property
    def goal_id(self) -> str | None:
        """返回稳定逻辑 Goal 身份的兼容别名。"""
        return self.logical_goal_id

    @property
    def created_at(self) -> int | None:
        """返回服务端 Goal createdAt 的兼容别名。"""
        return self.provider_created_at


def _ensure_goal_start_identity(state: _GoalOperationState, turn_id: str | None) -> None:
    """把启动响应中的首个物理 turn 补入状态，避免丢失 provider 身份。"""
    if not isinstance(turn_id, str) or not turn_id:
        return
    if getattr(state, "logical_turn_id", None) is None:
        state.logical_turn_id = turn_id
    ids = getattr(state, "physical_turn_ids", None)
    if isinstance(ids, list) and turn_id not in ids:
        ids.insert(0, turn_id)


class GoalHandle:
    """同步 Goal 句柄，聚合一个 Goal 的多个物理 turn。"""

    def __init__(self, client: CodexClient, state: _GoalOperationState) -> None:
        """保存客户端和路由状态；不会发起额外请求。"""
        self._client = client
        self._state = state

    @property
    def id(self) -> str:
        """返回 SDK 逻辑 turn 身份，即首个物理 turn 的 provider id。"""
        logical_turn_id = self._state.logical_turn_id
        if logical_turn_id is None:
            raise CodexError("Goal has not received its first physical turn")
        return logical_turn_id

    @property
    def logical_goal_id(self) -> str | None:
        """返回跨多个物理 turn 稳定的逻辑 Goal 身份。"""
        return self._state.logical_goal_id

    @property
    def goal_id(self) -> str | None:
        """返回逻辑 Goal 身份的兼容别名。"""
        return self.logical_goal_id

    @property
    def thread_id(self) -> str:
        """返回 Goal 所属线程身份。"""
        return self._state.thread_id

    @property
    def status(self) -> ThreadGoalStatus | None:
        """返回服务端 Goal 状态，不把非 complete 终态伪装成成功。"""
        return self._state.status

    @property
    def goal_status(self) -> ThreadGoalStatus | None:
        """返回 ``status`` 的公开兼容别名。"""
        return self.status

    @property
    def provider_status(self) -> ThreadGoalStatus | None:
        """返回服务端 Goal 状态的明确别名。"""
        return self.status

    @property
    def objective(self) -> str | None:
        """返回服务端记录的 Goal 目标描述。"""
        goal = self._state.goal
        return getattr(goal, "objective", None)

    @property
    def token_budget(self) -> int | None:
        """返回服务端记录的 token 预算。"""
        goal = self._state.goal
        return getattr(goal, "token_budget", None)

    @property
    def provider_created_at(self) -> int | None:
        """返回服务端 Goal 的 createdAt。"""
        goal = self._state.goal
        return getattr(goal, "created_at", None)

    @property
    def created_at(self) -> int | None:
        """返回 ``provider_created_at`` 的兼容别名。"""
        return self.provider_created_at

    @property
    def provider_goal(self) -> object | None:
        """返回最近一次服务端 Goal 快照；调用方不应修改它。"""
        return self._state.goal

    @property
    def physical_turn_ids(self) -> tuple[str, ...]:
        """按服务端通知顺序返回 Goal 聚合的物理 turn 身份。"""
        return tuple(self._state.physical_turn_ids)

    @property
    def physical_turn_start(self) -> str | None:
        """返回第一个物理 turn 身份。"""
        values = self.physical_turn_ids
        return values[0] if values else None

    @property
    def physical_turn_end(self) -> str | None:
        """返回最后一个已完成物理 turn 身份。"""
        return self._state.terminal_physical_turn_id

    @property
    def current_physical_turn_id(self) -> str | None:
        """返回当前活动物理 turn 身份。"""
        return self._state.current_turn_id

    @property
    def current_turn_id(self) -> str | None:
        """返回当前物理 turn 身份的兼容别名。"""
        return self.current_physical_turn_id

    @property
    def terminal_physical_turn_id(self) -> str | None:
        """返回终止时最后一个物理 turn 身份。"""
        return self._state.terminal_physical_turn_id

    @property
    def terminal_turn_id(self) -> str | None:
        """返回 ``terminal_physical_turn_id`` 的兼容别名。"""
        return self.terminal_physical_turn_id

    @property
    def final_turn_status(self) -> TurnStatus | None:
        """返回最后物理 turn 的实际状态。"""
        return self._state.final_turn_status

    @property
    def final_physical_turn_status(self) -> TurnStatus | None:
        """返回最后物理 turn 状态的明确别名。"""
        return self.final_turn_status

    @property
    def interrupted(self) -> bool:
        """返回是否已确认中断。"""
        return self._state.interrupted

    def stream(self) -> Iterator[Notification]:
        """按逻辑 Goal 事件流返回通知，并在结束时释放路由。"""
        unregister = getattr(self._client, "unregister_goal_operation", lambda: None)
        cancel = getattr(self._client, "cancel_goal_operation", lambda: None)

        def unregister_state() -> None:
            try:
                unregister(self._state)
            except AttributeError:
                pass

        def cancel_state() -> None:
            try:
                cancel(self._state)
            except AttributeError:
                pass

        return _GoalNotificationStream(
            self._state,
            lambda: self._client.next_goal_notification(self._state),
            unregister_state,
            cancel_state,
        )

    def run(self) -> TurnResult:
        """消费 Goal 事件并返回结果；非 complete 终态会抛错。"""
        stream = self.stream()
        try:
            result = _collect_turn_result(stream, turn_id=self.id)
        finally:
            stream.close()
        if self.status != ThreadGoalStatus.complete:
            raise CodexError(f"Goal ended with non-complete status: {self.status}")
        if self.final_turn_status != TurnStatus.completed:
            raise CodexError(f"final physical turn did not complete: {self.final_turn_status}")
        return result

    def interrupt(self) -> None:
        """请求暂停当前 Goal，并保存 interrupted/paused 事实。"""
        if not self._state.begin_interrupt():
            return
        try:
            self._client.cancel_goal_operation(self._state)
        except BaseException as exc:
            self._state.cancel_interrupt()
            self._state.fail(exc)
            raise
        else:
            self._state.confirm_interrupt()
            self._state.finish()
            self._state.wake_notification_reader()


class AsyncGoalHandle:
    """异步 Goal 句柄，属性与同步 GoalHandle 保持一致。"""

    def __init__(self, codex: AsyncCodex, state: _GoalOperationState) -> None:
        """保存异步客户端和路由状态；不会重复创建 Goal。"""
        self._codex = codex
        self._state = state

    @property
    def id(self) -> str:
        """返回首个物理 turn 对应的 SDK 逻辑 turn 身份。"""
        logical_turn_id = self._state.logical_turn_id
        if logical_turn_id is None:
            raise CodexError("Goal has not received its first physical turn")
        return logical_turn_id

    @property
    def logical_goal_id(self) -> str | None:
        """返回跨多个物理 turn 稳定的逻辑 Goal 身份。"""
        return self._state.logical_goal_id

    @property
    def goal_id(self) -> str | None:
        """返回逻辑 Goal 身份的兼容别名。"""
        return self.logical_goal_id

    @property
    def thread_id(self) -> str:
        """返回 Goal 所属线程身份。"""
        return self._state.thread_id

    @property
    def status(self) -> ThreadGoalStatus | None:
        """返回服务端 Goal 状态。"""
        return self._state.status

    @property
    def goal_status(self) -> ThreadGoalStatus | None:
        """返回 ``status`` 的公开兼容别名。"""
        return self.status

    @property
    def provider_status(self) -> ThreadGoalStatus | None:
        """返回服务端 Goal 状态的明确别名。"""
        return self.status

    @property
    def objective(self) -> str | None:
        """返回服务端 Goal 目标描述。"""
        return getattr(self._state.goal, "objective", None)

    @property
    def token_budget(self) -> int | None:
        """返回服务端 token 预算。"""
        return getattr(self._state.goal, "token_budget", None)

    @property
    def provider_created_at(self) -> int | None:
        """返回服务端 Goal createdAt。"""
        return getattr(self._state.goal, "created_at", None)

    @property
    def created_at(self) -> int | None:
        """返回 ``provider_created_at`` 的兼容别名。"""
        return self.provider_created_at

    @property
    def provider_goal(self) -> object | None:
        """返回最近一次服务端 Goal 快照。"""
        return self._state.goal

    @property
    def physical_turn_ids(self) -> tuple[str, ...]:
        """按顺序返回 Goal 聚合的物理 turn 身份。"""
        return tuple(self._state.physical_turn_ids)

    @property
    def physical_turn_start(self) -> str | None:
        """返回第一个物理 turn 身份。"""
        values = self.physical_turn_ids
        return values[0] if values else None

    @property
    def physical_turn_end(self) -> str | None:
        """返回最后一个已完成物理 turn 身份。"""
        return self._state.terminal_physical_turn_id

    @property
    def current_physical_turn_id(self) -> str | None:
        """返回当前活动物理 turn 身份。"""
        return self._state.current_turn_id

    @property
    def current_turn_id(self) -> str | None:
        """返回当前物理 turn 身份的兼容别名。"""
        return self.current_physical_turn_id

    @property
    def terminal_physical_turn_id(self) -> str | None:
        """返回终止时最后一个物理 turn 身份。"""
        return self._state.terminal_physical_turn_id

    @property
    def terminal_turn_id(self) -> str | None:
        """返回 ``terminal_physical_turn_id`` 的兼容别名。"""
        return self.terminal_physical_turn_id

    @property
    def final_turn_status(self) -> TurnStatus | None:
        """返回最后物理 turn 的实际状态。"""
        return self._state.final_turn_status

    @property
    def final_physical_turn_status(self) -> TurnStatus | None:
        """返回最后物理 turn 状态的明确别名。"""
        return self.final_turn_status

    @property
    def interrupted(self) -> bool:
        """返回是否已确认中断。"""
        return self._state.interrupted

    def stream(self) -> AsyncIterator[Notification]:
        """按逻辑 Goal 事件流返回异步通知，并在结束时释放路由。"""
        unregister = getattr(self._codex._client, "unregister_goal_operation", lambda: None)
        cancel = getattr(self._codex._client, "cancel_goal_operation", lambda: None)

        def unregister_state() -> None:
            try:
                unregister(self._state)
            except AttributeError:
                pass

        async def cancel_goal() -> None:
            try:
                result = cancel(self._state)
            except AttributeError:
                return
            if hasattr(result, "__await__"):
                await result

        return _AsyncGoalNotificationStream(
            self._state,
            lambda: self._codex._client.next_goal_notification(self._state),
            unregister_state,
            cancel_goal,
        )

    async def run(self) -> TurnResult:
        """消费异步 Goal 事件并返回结果；非 complete 终态会抛错。"""
        stream = self.stream()
        try:
            result = await _collect_async_turn_result(stream, turn_id=self.id)
        finally:
            await stream.aclose()
        if self.status != ThreadGoalStatus.complete:
            raise CodexError(f"Goal ended with non-complete status: {self.status}")
        if self.final_turn_status != TurnStatus.completed:
            raise CodexError(f"final physical turn did not complete: {self.final_turn_status}")
        return result

    async def interrupt(self) -> None:
        """异步请求暂停当前 Goal，并保存 interrupted/paused 事实。"""
        if not self._state.begin_interrupt():
            return
        try:
            result = self._codex._client.cancel_goal_operation(self._state)
            if hasattr(result, "__await__"):
                await result
        except BaseException as exc:
            self._state.cancel_interrupt()
            self._state.fail(exc)
            raise
        else:
            self._state.confirm_interrupt()
            self._state.finish()
            self._state.wake_notification_reader()


def _coerce_goal_checkpoint(
    checkpoint: GoalCheckpoint | str | Mapping[str, object] | None,
    *,
    thread_id: str,
    logical_turn_id: str | None,
    logical_goal_id: str | None,
    physical_turn_ids: tuple[str, ...] | list[str] | None,
) -> GoalCheckpoint | None:
    """把公开恢复参数收敛为 checkpoint，避免隐式猜测 Goal 身份。"""
    if isinstance(checkpoint, str):
        if (
            logical_turn_id is not None
            or logical_goal_id is not None
            or physical_turn_ids is not None
        ):
            raise ValueError(
                "resume_goal accepts either a checkpoint/id or explicit identity fields"
            )
        return GoalCheckpoint(logical_turn_id=checkpoint, thread_id=thread_id)
    if isinstance(checkpoint, Mapping):
        return GoalCheckpoint(
            logical_turn_id=checkpoint.get("logical_turn_id"),
            logical_goal_id=checkpoint.get("logical_goal_id", checkpoint.get("goal_id")),
            thread_id=checkpoint.get("thread_id", thread_id),
            physical_turn_count_before=checkpoint.get("physical_turn_count_before"),
            physical_turn_ids=checkpoint.get("physical_turn_ids", ()),
            objective=checkpoint.get("objective"),
            token_budget=checkpoint.get("token_budget"),
            provider_created_at=checkpoint.get(
                "provider_created_at", checkpoint.get("created_at")
            ),
        )
    if checkpoint is not None:
        if any(
            value is not None for value in (logical_turn_id, logical_goal_id, physical_turn_ids)
        ):
            raise ValueError(
                "resume_goal cannot combine checkpoint with explicit identity fields"
            )
        return checkpoint
    if logical_turn_id is None and logical_goal_id is None and physical_turn_ids is None:
        return None
    return GoalCheckpoint(
        logical_turn_id=logical_turn_id,
        logical_goal_id=logical_goal_id,
        thread_id=thread_id,
        physical_turn_ids=physical_turn_ids,
    )


@dataclass(slots=True)
class Thread:
    """Synchronous conversation thread used to run one or more turns."""

    _client: CodexClient
    id: str

    def run(
        self,
        input: RunInput,
        *,
        approval_mode: ApprovalMode | None = None,
        cwd: str | None = None,
        effort: ReasoningEffort | None = None,
        model: str | None = None,
        output_schema: JsonObject | None = None,
        personality: Personality | None = None,
        sandbox: Sandbox | None = None,
        service_tier: str | None = None,
        summary: ReasoningSummary | None = None,
    ) -> TurnResult:
        """Run a complete turn and collect its final result."""
        turn = self.turn(
            input,
            approval_mode=approval_mode,
            cwd=cwd,
            effort=effort,
            model=model,
            output_schema=output_schema,
            personality=personality,
            sandbox=sandbox,
            service_tier=service_tier,
            summary=summary,
        )
        stream = turn.stream()
        try:
            return _collect_turn_result(stream, turn_id=turn.id)
        finally:
            stream.close()

    # BEGIN GENERATED: Thread.flat_methods
    def turn(
        self,
        input: RunInput,
        *,
        approval_mode: ApprovalMode | None = None,
        cwd: str | None = None,
        effort: ReasoningEffort | None = None,
        model: str | None = None,
        output_schema: JsonObject | None = None,
        personality: Personality | None = None,
        sandbox: Sandbox | None = None,
        service_tier: str | None = None,
        summary: ReasoningSummary | None = None,
    ) -> TurnHandle:
        """Start a turn and return a handle for streaming or control."""
        wire_input = _to_wire_input(_normalize_run_input(input))
        approval_policy, approvals_reviewer = _approval_mode_override_settings(approval_mode)
        params = TurnStartParams(
            thread_id=self.id,
            input=wire_input,
            approval_policy=approval_policy,
            approvals_reviewer=approvals_reviewer,
            cwd=cwd,
            effort=effort,
            model=model,
            output_schema=output_schema,
            personality=personality,
            sandbox_policy=_sandbox_policy(sandbox),
            service_tier=service_tier,
            summary=summary,
        )
        turn = self._client.turn_start(self.id, wire_input, params=params)
        return TurnHandle(self._client, self.id, turn.turn.id)

    # END GENERATED: Thread.flat_methods

    def goal(self, objective: str, *, token_budget: int | None = None) -> GoalHandle:
        """创建一个 Goal 并返回可流式读取、运行或中断的公开句柄。"""
        if not isinstance(objective, str) or not objective.strip():
            raise ValueError("Goal objective must be a non-empty string")
        state, turn_id = self._client.start_goal_operation(
            self.id,
            objective,
            token_budget=token_budget,
        )
        _ensure_goal_start_identity(state, turn_id)
        return GoalHandle(self._client, state)

    def resume_goal(
        self,
        checkpoint: GoalCheckpoint | str | Mapping[str, object] | None = None,
        *,
        logical_turn_id: str | None = None,
        logical_goal_id: str | None = None,
        physical_turn_ids: tuple[str, ...] | list[str] | None = None,
    ) -> GoalHandle:
        """只读恢复既有 Goal；身份不唯一时失败且不清除或重设 Goal。"""
        checkpoint = _coerce_goal_checkpoint(
            checkpoint,
            thread_id=self.id,
            logical_turn_id=logical_turn_id,
            logical_goal_id=logical_goal_id,
            physical_turn_ids=physical_turn_ids,
        )
        state, turn_id = self._client.resume_goal_operation(
            self.id,
            checkpoint=checkpoint,
        )
        _ensure_goal_start_identity(state, turn_id)
        return GoalHandle(self._client, state)

    def read(self, *, include_turns: bool = False) -> ThreadReadResponse:
        """Read this thread, optionally including its turn history."""
        return self._client.thread_read(self.id, include_turns=include_turns)

    def set_name(self, name: str) -> ThreadSetNameResponse:
        return self._client.thread_set_name(self.id, name)

    def compact(self) -> ThreadCompactStartResponse:
        return self._client.thread_compact(self.id)


@dataclass(slots=True)
class AsyncThread:
    """Asynchronous conversation thread used to run one or more turns."""

    _codex: AsyncCodex
    id: str

    async def run(
        self,
        input: RunInput,
        *,
        approval_mode: ApprovalMode | None = None,
        cwd: str | None = None,
        effort: ReasoningEffort | None = None,
        model: str | None = None,
        output_schema: JsonObject | None = None,
        personality: Personality | None = None,
        sandbox: Sandbox | None = None,
        service_tier: str | None = None,
        summary: ReasoningSummary | None = None,
    ) -> TurnResult:
        """Run a complete turn asynchronously and collect its final result."""
        turn = await self.turn(
            input,
            approval_mode=approval_mode,
            cwd=cwd,
            effort=effort,
            model=model,
            output_schema=output_schema,
            personality=personality,
            sandbox=sandbox,
            service_tier=service_tier,
            summary=summary,
        )
        stream = turn.stream()
        try:
            return await _collect_async_turn_result(stream, turn_id=turn.id)
        finally:
            await stream.aclose()

    # BEGIN GENERATED: AsyncThread.flat_methods
    async def turn(
        self,
        input: RunInput,
        *,
        approval_mode: ApprovalMode | None = None,
        cwd: str | None = None,
        effort: ReasoningEffort | None = None,
        model: str | None = None,
        output_schema: JsonObject | None = None,
        personality: Personality | None = None,
        sandbox: Sandbox | None = None,
        service_tier: str | None = None,
        summary: ReasoningSummary | None = None,
    ) -> AsyncTurnHandle:
        """Start a turn and return a handle for streaming or control."""
        await self._codex._ensure_initialized()
        wire_input = _to_wire_input(_normalize_run_input(input))
        approval_policy, approvals_reviewer = _approval_mode_override_settings(approval_mode)
        params = TurnStartParams(
            thread_id=self.id,
            input=wire_input,
            approval_policy=approval_policy,
            approvals_reviewer=approvals_reviewer,
            cwd=cwd,
            effort=effort,
            model=model,
            output_schema=output_schema,
            personality=personality,
            sandbox_policy=_sandbox_policy(sandbox),
            service_tier=service_tier,
            summary=summary,
        )
        turn = await self._codex._client.turn_start(
            self.id,
            wire_input,
            params=params,
        )
        return AsyncTurnHandle(self._codex, self.id, turn.turn.id)

    # END GENERATED: AsyncThread.flat_methods

    async def goal(self, objective: str, *, token_budget: int | None = None) -> AsyncGoalHandle:
        """异步创建一个 Goal 并返回公开句柄。"""
        if not isinstance(objective, str) or not objective.strip():
            raise ValueError("Goal objective must be a non-empty string")
        await self._codex._ensure_initialized()
        state, turn_id = await self._codex._client.start_goal_operation(
            self.id,
            objective,
            token_budget=token_budget,
        )
        _ensure_goal_start_identity(state, turn_id)
        return AsyncGoalHandle(self._codex, state)

    async def resume_goal(
        self,
        checkpoint: GoalCheckpoint | str | Mapping[str, object] | None = None,
        *,
        logical_turn_id: str | None = None,
        logical_goal_id: str | None = None,
        physical_turn_ids: tuple[str, ...] | list[str] | None = None,
    ) -> AsyncGoalHandle:
        """异步只读恢复既有 Goal；身份不唯一时失败。"""
        await self._codex._ensure_initialized()
        checkpoint = _coerce_goal_checkpoint(
            checkpoint,
            thread_id=self.id,
            logical_turn_id=logical_turn_id,
            logical_goal_id=logical_goal_id,
            physical_turn_ids=physical_turn_ids,
        )
        state, turn_id = await self._codex._client.resume_goal_operation(
            self.id,
            checkpoint=checkpoint,
        )
        _ensure_goal_start_identity(state, turn_id)
        return AsyncGoalHandle(self._codex, state)

    async def read(self, *, include_turns: bool = False) -> ThreadReadResponse:
        """Read this thread, optionally including its turn history."""
        await self._codex._ensure_initialized()
        return await self._codex._client.thread_read(self.id, include_turns=include_turns)

    async def set_name(self, name: str) -> ThreadSetNameResponse:
        await self._codex._ensure_initialized()
        return await self._codex._client.thread_set_name(self.id, name)

    async def compact(self) -> ThreadCompactStartResponse:
        await self._codex._ensure_initialized()
        return await self._codex._client.thread_compact(self.id)


@dataclass(slots=True)
class TurnHandle:
    """Control and consume a synchronous turn after it has started."""

    _client: CodexClient
    thread_id: str
    id: str

    def steer(self, input: RunInput) -> TurnSteerResponse:
        """Send additional input to this active turn."""
        return self._client.turn_steer(
            self.thread_id,
            self.id,
            _to_wire_input(_normalize_run_input(input)),
        )

    def interrupt(self) -> TurnInterruptResponse:
        """Request interruption of this active turn."""
        return self._client.turn_interrupt(self.thread_id, self.id)

    def stream(self) -> Iterator[Notification]:
        """Yield only notifications routed to this turn handle."""
        self._client.register_turn_notifications(self.id)
        try:
            while True:
                event = self._client.next_turn_notification(self.id)
                yield event
                if (
                    event.method == "turn/completed"
                    and isinstance(event.payload, TurnCompletedNotification)
                    and event.payload.turn.id == self.id
                ):
                    break
        finally:
            self._client.unregister_turn_notifications(self.id)

    def run(self) -> TurnResult:
        """Consume the turn stream and return its completed result."""
        stream = self.stream()
        try:
            return _collect_turn_result(stream, turn_id=self.id)
        finally:
            stream.close()


@dataclass(slots=True)
class AsyncTurnHandle:
    """Control and consume an asynchronous turn after it has started."""

    _codex: AsyncCodex
    thread_id: str
    id: str

    async def steer(self, input: RunInput) -> TurnSteerResponse:
        """Send additional input to this active turn."""
        await self._codex._ensure_initialized()
        return await self._codex._client.turn_steer(
            self.thread_id,
            self.id,
            _to_wire_input(_normalize_run_input(input)),
        )

    async def interrupt(self) -> TurnInterruptResponse:
        """Request interruption of this active turn."""
        await self._codex._ensure_initialized()
        return await self._codex._client.turn_interrupt(self.thread_id, self.id)

    async def stream(self) -> AsyncIterator[Notification]:
        """Yield only notifications routed to this async turn handle."""
        await self._codex._ensure_initialized()
        self._codex._client.register_turn_notifications(self.id)
        try:
            while True:
                event = await self._codex._client.next_turn_notification(self.id)
                yield event
                if (
                    event.method == "turn/completed"
                    and isinstance(event.payload, TurnCompletedNotification)
                    and event.payload.turn.id == self.id
                ):
                    break
        finally:
            self._codex._client.unregister_turn_notifications(self.id)

    async def run(self) -> TurnResult:
        """Consume the turn stream and return its completed result."""
        stream = self.stream()
        try:
            return await _collect_async_turn_result(stream, turn_id=self.id)
        finally:
            await stream.aclose()
