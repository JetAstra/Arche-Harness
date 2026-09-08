"""验证 Investigation 使用 Goal 协议及其可恢复物理 turn 账本。"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from openai_codex import CodexConfig

from chemistry_multiagent.harness import codex_sdk_adapter as adapter
from chemistry_multiagent.harness.base import HarnessRequest
from chemistry_multiagent.harness.session_store import CodexSessionOwner, CodexSessionStore


class _FakeGoalHandle:
    """提供最小公开 Goal handle，模拟一个包含两个物理 turn 的完成目标。"""

    def __init__(self, *, status: str = "complete") -> None:
        self.id = "turn-1"
        self.goal_id = "goal-1"
        self.logical_goal_id = "logical-goal-1"
        self.goal_status = status
        self.status = status
        self.final_turn_status = "completed" if status == "complete" else "failed"
        # logical stream 只映射首个物理 turn；末个物理 turn 的失败由 history
        # 和 handle.final_turn_status 提供，避免把两个物理 turn 压成同一事件。
        self.logical_turn_status = "completed"
        self.physical_turn_ids = ("turn-1", "turn-2")
        self.physical_turn_start = "turn-1"
        self.physical_turn_end = "turn-2"
        self.content = '{"ok":true}'
        self.answer = self.content
        self.response = SimpleNamespace(content=self.content)
        self.provider_detail = "provider free text must not enter goal journal"
        self.interrupt_calls = 0

    def run(self):
        """返回已结束的 Goal 结果。"""

        return self

    def stream(self):
        """返回一个逻辑通知流，工具回执来自两个物理 turn。"""

        yield SimpleNamespace(
            method="turn/started",
            payload={"turn": {"id": "turn-1", "status": "inProgress"}},
        )
        yield SimpleNamespace(
            method="item/completed",
            payload={
                "item": {
                    "type": "mcpToolCall",
                    "id": "tool-1",
                    "name": "get_execution_job",
                    "status": "completed",
                }
            },
        )
        yield SimpleNamespace(
            method="item/completed",
            payload={
                "item": {
                    "type": "mcpToolCall",
                    "id": "tool-2",
                    "name": "collect_execution_job",
                    "status": "completed",
                }
            },
        )
        yield SimpleNamespace(
            method="item/completed",
            payload={
                "item": {
                    "type": "agentMessage",
                    "phase": "final_answer",
                    "text": self.content,
                }
            },
        )
        yield SimpleNamespace(
            method="turn/completed",
            payload={
                "turn": {
                    "id": "turn-1",
                    "status": self.logical_turn_status,
                }
            },
        )
        yield SimpleNamespace(
            method="thread/goal/updated",
            payload={"goal": {"status": self.goal_status}},
        )

    def interrupt(self):
        """记录显式中断请求。"""

        self.interrupt_calls += 1


class _FakeGoalHandleWithoutFirstTurnId(_FakeGoalHandle):
    """模拟首个物理 turn 尚未出现时，读取 handle.id 会抛出的 SDK 句柄。"""

    def __init__(self, *, status: str = "complete") -> None:
        self._id: str | None = None
        super().__init__(status=status)
        self._id = None

    @property
    def id(self) -> str:
        if self._id is None:
            raise RuntimeError("Goal has not received its first physical turn")
        return self._id

    @id.setter
    def id(self, value: str | None) -> None:
        self._id = value


class _FakeGoalHandleWithoutStartedEvent(_FakeGoalHandleWithoutFirstTurnId):
    """模拟没有任何逻辑 turn/started 通知的异常 Goal 流。"""

    def stream(self):
        yield SimpleNamespace(
            method="thread/goal/updated",
            payload={"goal": {"status": self.goal_status}},
        )


class _FakeGoalThread:
    """只实现 Goal 入口，缺少普通 turn 入口以确保 adapter 不走普通 turn。"""

    id = "thread-goal-1"

    def __init__(self, *, status: str = "complete", handle_id: str | None = "turn-1") -> None:
        self.handle = (
            _FakeGoalHandleWithoutFirstTurnId(status=status)
            if handle_id is None
            else _FakeGoalHandle(status=status)
        )
        self.handle.id = handle_id
        self.goal_calls: list[tuple[str, int | None]] = []
        self.resume_calls = 0
        self.history_prompt = "objective"

    def goal(self, objective: str, *, token_budget: int | None = None):
        """记录首次 Goal 启动。"""

        self.goal_calls.append((objective, token_budget))
        return self.handle

    def resume_goal(self, *, checkpoint=None):
        """记录恢复入口；恢复不应重新提交目标。"""

        self.resume_calls += 1
        self.resume_checkpoint = checkpoint
        return self.handle

    def read(self, *, include_turns: bool = False):
        """返回固定的两个物理 turn history。"""

        assert include_turns is True
        turns = [
            {
                "id": "turn-1",
                "status": "completed",
                "items_view": "full",
                "items": [
                    {
                        "type": "userMessage",
                        "content": [{"type": "text", "text": self.history_prompt}],
                    },
                    {"type": "mcpToolCall", "id": "tool-1", "status": "completed"},
                ],
            },
            {
                "id": "turn-2",
                "status": self.handle.final_turn_status,
                "items_view": "full",
                "items": [
                    {"type": "mcpToolCall", "id": "tool-2", "status": "completed"},
                    {
                        "type": "agentMessage",
                        "phase": "final_answer",
                        "text": '{"ok":true}',
                    },
                ],
            },
        ]
        return {"thread": {"id": self.id, "turns": turns}}


class _AsyncJobGoalHandle(_FakeGoalHandle):
    """提交非终态异步作业后，模拟 Goal 等待下一次查询。"""

    def stream(self):
        yield SimpleNamespace(
            method="turn/started",
            payload={"turn": {"id": "turn-1", "status": "inProgress"}},
        )
        yield SimpleNamespace(
            method="item/completed",
            payload={
                "item": {
                    "type": "mcpToolCall",
                    "id": "mcp-submit-async-goal",
                    "tool": "submit_action_job",
                    "status": "completed",
                    "result": {
                        "structuredContent": {
                            "status": "success",
                            "job_id": "job-async-goal",
                            "job_status": "queued",
                            "terminal": False,
                        }
                    },
                }
            },
        )
        time.sleep(0.05)
        yield SimpleNamespace(
            method="item/completed",
            payload={
                "item": {
                    "type": "agentMessage",
                    "id": "agent-message-async-goal",
                    "phase": "final_answer",
                    "text": self.content,
                }
            },
        )
        yield SimpleNamespace(
            method="turn/completed",
            payload={
                "turn": {"id": "turn-1", "status": "completed"},
            },
        )
        yield SimpleNamespace(
            method="thread/goal/updated",
            payload={"goal": {"status": self.goal_status}},
        )


class _AsyncJobGoalThread(_FakeGoalThread):
    """把 Goal 流替换为提交异步作业后静默的测试线程。"""

    def __init__(self) -> None:
        super().__init__()
        self.handle = _AsyncJobGoalHandle()


class _FakeGoalCodex:
    """把同一个假线程暴露给 adapter 的启动和恢复入口。"""

    thread = _FakeGoalThread()

    def __init__(self, _config):
        self.closed = False

    def thread_start(self, **_kwargs):
        return self.thread

    def thread_resume(self, *_args, **_kwargs):
        return self.thread

    def close(self):
        self.closed = True


def _goal_request(tmp_path: Path, *, stage: str = "initial_investigation") -> HarnessRequest:
    """构造一个 Investigation Goal 请求。"""

    return HarnessRequest(
        messages=({"role": "user", "content": "objective"},),
        model="codex-model",
        role="investigation",
        stage=stage,
        profile="investigation",
        workspace=tmp_path / "workspace",
        metadata={"request_kind": "initial_investigation"},
    )


def _goal_policy_digest(request: HarnessRequest) -> str:
    """按当前 Investigation 能力配置计算固定 thread 策略摘要。"""

    effective = adapter.build_codex_capability_config(
        thread_config={},
        workspace=request.workspace,
        profile="investigation",
        env_key="ARCHE_CODEX_API_KEY",
        enable_investigation_goals=True,
    )
    return adapter.calculate_codex_thread_policy_digest(
        thread_config=effective,
        workspace=request.workspace,
        provider_id="arche_codex",
        env_key="ARCHE_CODEX_API_KEY",
        model=request.model,
        role=request.role,
        stage=request.stage,
        profile="investigation",
    )


def test_investigation_goal_config_defaults_off_and_requires_explicit_opt_in(
    tmp_path: Path,
) -> None:
    """新 Investigation 默认使用普通 turn，只有显式选择才启用实验性 Goal。"""

    for profile in ("legacy_planner", "task_review"):
        config = adapter.build_codex_capability_config(
            thread_config={},
            workspace=tmp_path / profile,
            profile=profile,
        )
        assert config["features"]["goals"] is False

    investigation = adapter.build_codex_capability_config(
        thread_config={},
        workspace=tmp_path / "investigation",
        profile="investigation",
    )
    assert investigation["features"]["goals"] is False

    goal_investigation = adapter.build_codex_capability_config(
        thread_config={},
        workspace=tmp_path / "goal-investigation",
        profile="investigation",
        enable_investigation_goals=True,
    )
    assert goal_investigation["features"]["goals"] is True

    with pytest.raises(adapter.HarnessError, match="only available for Investigation"):
        adapter.build_codex_capability_config(
            thread_config={},
            workspace=tmp_path / "review-with-goal",
            profile="task_review",
            enable_investigation_goals=True,
        )


def test_legacy_investigation_mode_runs_ordinary_turn_instead_of_goal(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """旧 Investigation session 关闭 Goal 后只调用普通 turn。"""

    class OrdinaryTurn:
        id = "turn-ordinary-1"

        def stream(self):
            item = SimpleNamespace(
                root=SimpleNamespace(
                    type="agentMessage",
                    phase="final_answer",
                    text='{"ok":true}',
                )
            )
            item_payload = SimpleNamespace(item=item)
            item_payload.model_dump = lambda **_kwargs: {
                "item": {
                    "type": "agentMessage",
                    "phase": "final_answer",
                    "text": '{"ok":true}',
                }
            }
            yield SimpleNamespace(
                method="item/completed",
                payload=item_payload,
            )
            turn_payload = SimpleNamespace(
                turn=SimpleNamespace(id=self.id, status="completed")
            )
            turn_payload.model_dump = lambda **_kwargs: {
                "turn": {"id": self.id, "status": "completed"}
            }
            yield SimpleNamespace(
                method="turn/completed",
                payload=turn_payload,
            )

    class DualProtocolThread(_FakeGoalThread):
        def __init__(self) -> None:
            super().__init__()
            self.turn_calls = 0

        def turn(self, *_args, **_kwargs):
            self.turn_calls += 1
            return OrdinaryTurn()

    class DualProtocolCodex(_FakeGoalCodex):
        thread = DualProtocolThread()

    monkeypatch.setattr(adapter, "Codex", DualProtocolCodex)
    request = _goal_request(tmp_path)
    effective = adapter.build_codex_capability_config(
        thread_config={},
        workspace=request.workspace,
        profile="investigation",
        env_key="ARCHE_CODEX_API_KEY",
        enable_investigation_goals=False,
    )
    policy_digest = adapter.calculate_codex_thread_policy_digest(
        thread_config=effective,
        workspace=request.workspace,
        provider_id="arche_codex",
        env_key="ARCHE_CODEX_API_KEY",
        model=request.model,
        role=request.role,
        stage=request.stage,
        profile="investigation",
    )

    result = adapter.run_codex_sdk_turn(
        config=CodexConfig(),
        request=request,
        thread_config={},
        session_id=None,
        trace_path=tmp_path / "ordinary-trace.jsonl",
        completed_turn_receipt_root=tmp_path / "ordinary-receipts",
        request_digest="a" * 64,
        turn_count_before=0,
        thread_policy_digest=policy_digest,
        enable_investigation_goals=False,
    )

    assert DualProtocolCodex.thread.turn_calls == 1
    assert DualProtocolCodex.thread.goal_calls == []
    assert not result.goal_status
    assert result.turn_id == "turn-ordinary-1"
    assert result.thread_policy_digest == policy_digest


def test_goal_initial_requires_the_public_token_budget_parameter() -> None:
    """Goal API 缺少公开 token_budget 参数时立即失败，不能降级重提。"""

    class MissingTokenBudget:
        def goal(self, objective: str):
            raise AssertionError(f"不应调用旧 Goal API: {objective}")

    with pytest.raises(TypeError, match="token_budget"):
        adapter._goal_call_initial(MissingTokenBudget(), "objective", None)


def test_goal_checkpoint_requires_physical_baseline_and_token_budget_fields(monkeypatch) -> None:
    """Goal checkpoint 缺少物理基线或预算字段时必须阻断恢复。"""

    class MissingPhysicalBaseline:
        def __init__(
            self,
            *,
            logical_turn_id=None,
            logical_goal_id=None,
            thread_id=None,
            physical_turn_ids=(),
            objective=None,
            token_budget=None,
        ):
            self.logical_turn_id = logical_turn_id
            self.logical_goal_id = logical_goal_id
            self.thread_id = thread_id
            self.physical_turn_ids = physical_turn_ids
            self.objective = objective
            self.token_budget = token_budget

    monkeypatch.setattr(adapter, "GoalCheckpoint", MissingPhysicalBaseline)
    with pytest.raises(
        adapter.CodexGoalRecoveryPending,
        match="physical baseline or token budget",
    ):
        adapter._build_goal_checkpoint(
            {
                "logical_goal_id": "logical-goal-1",
                "goal_id": "goal-1",
                "thread_id": "thread-goal-1",
                "physical_turn_count_before": 0,
            },
            [{"turn_id": "turn-1"}],
            prompt_text="objective",
            token_budget=None,
        )


def test_goal_turn_journal_rejects_provider_goal_identity_drift(tmp_path: Path) -> None:
    """Goal 物理账本只允许空身份绑定一次，不能改写为另一个 provider Goal。"""

    root = tmp_path / "receipts"
    adapter._write_goal_turn_journal(
        root,
        thread_id="thread-goal-1",
        logical_goal_id="logical-goal-1",
        goal_id="goal-a",
        turn_records=[
            {"turn_id": "turn-1", "status": "completed"},
            {"turn_id": "turn-2", "status": "active"},
        ],
    )
    with pytest.raises(adapter.HarnessError, match="changed identity|provider identity"):
        adapter._write_goal_turn_journal(
            root,
            thread_id="thread-goal-1",
            logical_goal_id="logical-goal-1",
            goal_id="goal-b",
            turn_records=[
                {"turn_id": "turn-1", "status": "completed"},
                {"turn_id": "turn-2", "status": "active"},
            ],
        )


def test_goal_receipt_handle_requires_canonical_store_and_full_identity_on_ack(
    tmp_path: Path,
) -> None:
    """Goal 收据句柄只能绑定当前 SessionStore，并在确认前核对全部固定字段。"""

    owner = CodexSessionOwner.investigation("task-1", "batch-1", "comb-1")
    store = CodexSessionStore.from_root(
        tmp_path / "state",
        owner=owner,
        scope_id="scope-1",
        model="codex-model",
    )
    root = store.manifest_path.parent / "completed_turn_receipts"
    receipt = {
        "schema_version": 1,
        "thread_id": "thread-goal-1",
        "logical_goal_id": "logical-goal-1",
        "goal_id": "goal-1",
        "objective_digest": "b" * 64,
        "request_digest": "a" * 64,
        "turn_count_before": 0,
        "physical_turn_count_before": 0,
        "physical_turn_ids": ["turn-1", "turn-2"],
        "physical_turn_start": "turn-1",
        "physical_turn_end": "turn-2",
        "goal_status": "complete",
        "final_turn_status": "completed",
        "thread_policy_digest": "c" * 64,
    }
    adapter._write_goal_receipt(root, receipt)
    handle = adapter.build_codex_goal_receipt_handle(
        root,
        owner=owner,
        session_store=store,
        thread_id="thread-goal-1",
        turn_id="turn-1",
        request_digest="a" * 64,
        objective_digest="b" * 64,
        thread_policy_digest="c" * 64,
        turn_count_before=0,
        handoff_kind="result",
    )

    tampered = dict(receipt, objective_digest="d" * 64)
    (root.parent / ".arche_codex_goal_receipt.json").write_text(
        json.dumps(tampered), encoding="utf-8"
    )
    with pytest.raises(adapter.HarnessError, match="identity changed"):
        handle.acknowledge()

    with pytest.raises(adapter.HarnessError, match="canonical"):
        adapter.build_codex_goal_receipt_handle(
            tmp_path / "other-receipts",
            owner=owner,
            session_store=store,
            thread_id="thread-goal-1",
            turn_id="turn-1",
            request_digest="a" * 64,
            thread_policy_digest="c" * 64,
            turn_count_before=0,
            handoff_kind="result",
        )


def test_goal_receipt_resolver_missing_record_is_read_only(tmp_path: Path) -> None:
    """缺少 Goal 收据时只读解析不创建目录或修改权限。"""

    owner = CodexSessionOwner.investigation("task-1", "batch-1", "comb-1")
    store = CodexSessionStore.from_root(
        tmp_path / "state",
        owner=owner,
        scope_id="scope-1",
        model="codex-model",
    )
    root = store.manifest_path.parent / "completed_turn_receipts"
    assert not root.exists()
    assert (
        adapter.resolve_codex_goal_receipt_handle(
            root,
            owner=owner,
            session_store=store,
            thread_id="thread-goal-1",
            turn_id="turn-1",
            request_digest=None,
            objective_digest=None,
            turn_count_before=None,
            thread_policy_digest=None,
            handoff_kind="result",
        )
        is None
    )
    assert not root.exists()


def test_goal_orphan_journal_notifies_recovery_before_rejecting_new_thread(
    tmp_path: Path,
) -> None:
    """孤立 Goal 物理账本阻止新 thread，并先保存 Goal 恢复事实。"""

    request = _goal_request(tmp_path)
    root = tmp_path / "receipts"
    adapter._write_goal_turn_journal(
        root,
        thread_id="thread-goal-1",
        logical_goal_id="logical-goal-1",
        goal_id="goal-1",
        turn_records=[{"turn_id": "turn-1", "status": "active"}],
    )
    policy_digest = _goal_policy_digest(request)
    recovery: list[tuple[str | None, str | None]] = []
    with pytest.raises(adapter.CodexGoalRecoveryPending):
        adapter.run_codex_sdk_turn(
            config=CodexConfig(),
            request=request,
            thread_config={},
            session_id=None,
            trace_path=tmp_path / "trace.jsonl",
            completed_turn_receipt_root=root,
            request_digest="a" * 64,
            turn_count_before=0,
            thread_policy_digest=policy_digest,
            on_recovery_pending=lambda turn_id, *, reason=None: recovery.append(
                (turn_id, reason)
            ),
        )
    assert recovery == [("turn-1", "goal")]


def test_investigation_adapter_runs_goal_and_returns_full_physical_range(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Investigation adapter 调用 Goal，返回逻辑身份及完整物理 turn 范围。"""

    _FakeGoalCodex.thread = _FakeGoalThread()
    monkeypatch.setattr(adapter, "Codex", _FakeGoalCodex)
    request = _goal_request(tmp_path)
    _FakeGoalCodex.thread.history_prompt = request.prompt_text()
    receipt_root = tmp_path / "receipts"
    policy_digest = _goal_policy_digest(request)
    completed_items: list[object] = []

    result = adapter.run_codex_sdk_turn(
        config=CodexConfig(),
        request=request,
        thread_config={},
        session_id=None,
        trace_path=tmp_path / "trace.jsonl",
        completed_turn_receipt_root=receipt_root,
        request_digest="a" * 64,
        turn_count_before=0,
        thread_policy_digest=policy_digest,
        enable_investigation_goals=True,
        on_tool_completed=completed_items.append,
    )

    assert _FakeGoalCodex.thread.goal_calls
    assert not hasattr(_FakeGoalCodex.thread, "turn")
    assert result.goal_status == "complete"
    assert result.logical_goal_id == adapter._goal_request_logical_id(
        "thread-goal-1", "a" * 64
    )
    assert result.physical_turn_ids == ("turn-1", "turn-2")
    assert result.turn_id == "turn-1"
    assert result.response.content == '{"ok":true}'
    assert [item["id"] for item in completed_items] == ["tool-1", "tool-2"]
    trace_text = (tmp_path / "trace.jsonl").read_text(encoding="utf-8")
    assert "provider free text must not enter goal journal" not in trace_text


def test_investigation_goal_preserves_external_job_wait_budget_after_submit_status(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Goal 提交非终态作业后，模型短暂无事件不会触发普通空闲检测。"""

    _FakeGoalCodex.thread = _AsyncJobGoalThread()
    monkeypatch.setattr(adapter, "Codex", _FakeGoalCodex)
    request = _goal_request(tmp_path)
    _FakeGoalCodex.thread.history_prompt = request.prompt_text()
    policy_digest = _goal_policy_digest(request)

    result = adapter.run_codex_sdk_turn(
        config=CodexConfig(),
        request=request,
        thread_config={},
        session_id=None,
        trace_path=tmp_path / "trace.jsonl",
        completed_turn_receipt_root=tmp_path / "receipts",
        request_digest="a" * 64,
        turn_count_before=0,
        thread_policy_digest=policy_digest,
        enable_investigation_goals=True,
        idle_timeout_seconds=0.01,
        tool_idle_timeout_seconds=0.2,
    )

    assert result.goal_status == "complete"
    assert result.response.content == '{"ok":true}'


@pytest.mark.parametrize("status", ["paused", "blocked", "usage_limited", "budget_limited"])
def test_investigation_goal_noncomplete_terminal_is_recoverable_and_not_success(
    monkeypatch,
    tmp_path: Path,
    status: str,
) -> None:
    """Goal 非 complete 终态不能生成成功结果，且必须留下可恢复状态。"""

    _FakeGoalCodex.thread = _FakeGoalThread(status=status)
    monkeypatch.setattr(adapter, "Codex", _FakeGoalCodex)
    request = _goal_request(tmp_path)
    _FakeGoalCodex.thread.history_prompt = request.prompt_text()
    policy_digest = _goal_policy_digest(request)

    with pytest.raises(adapter.CodexGoalRecoveryPending) as raised:
        adapter.run_codex_sdk_turn(
            config=CodexConfig(),
            request=request,
            thread_config={},
            session_id=None,
            trace_path=tmp_path / "trace.jsonl",
            completed_turn_receipt_root=tmp_path / "receipts",
            request_digest="a" * 64,
            turn_count_before=0,
            thread_policy_digest=policy_digest,
            enable_investigation_goals=True,
        )

    assert raised.value.goal_status == status
    assert not list((tmp_path / "receipts").parent.glob(".arche_codex_goal_receipt.json"))


def test_goal_recovery_reuses_existing_intent_without_starting_new_goal(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """已有 Goal intent 恢复只调用 resume_goal，不得清除、设置或重提。"""

    _FakeGoalCodex.thread = _FakeGoalThread()
    monkeypatch.setattr(adapter, "Codex", _FakeGoalCodex)
    request = _goal_request(tmp_path)
    policy_digest = _goal_policy_digest(request)
    _FakeGoalCodex.thread.history_prompt = request.prompt_text()
    root = tmp_path / "receipts"
    root.mkdir()
    intent = {
        "schema_version": 1,
        "logical_goal_id": "logical-goal-1",
        "thread_id": "thread-goal-1",
        "goal_id": "goal-1",
        "request_digest": "a" * 64,
        "thread_policy_digest": policy_digest,
        "objective_digest": hashlib.sha256(
            request.prompt_text().encode("utf-8")
        ).hexdigest(),
        "turn_count_before": 0,
        "physical_turn_count_before": 0,
        "profile": "investigation",
        "stage": request.stage,
        "status": "active",
    }
    (root.parent / ".arche_codex_goal_intent.json").write_text(
        json.dumps(intent), encoding="utf-8"
    )

    result = adapter.run_codex_sdk_turn(
        config=CodexConfig(),
        request=request,
        thread_config={},
        session_id="thread-goal-1",
        trace_path=tmp_path / "trace.jsonl",
        completed_turn_receipt_root=root,
        request_digest="a" * 64,
        turn_count_before=0,
        thread_policy_digest=policy_digest,
        enable_investigation_goals=True,
    )

    assert result.goal_status == "complete"
    assert _FakeGoalCodex.thread.resume_calls == 1
    assert _FakeGoalCodex.thread.goal_calls == []


def test_goal_recovery_defers_started_hook_until_stream_supplies_first_turn_id(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """恢复句柄暂时没有 id 时，首个逻辑 started 事件仍只登记一次首物理 turn。"""

    _FakeGoalCodex.thread = _FakeGoalThread(handle_id=None)
    monkeypatch.setattr(adapter, "Codex", _FakeGoalCodex)
    request = _goal_request(tmp_path)
    _FakeGoalCodex.thread.history_prompt = request.prompt_text()
    policy_digest = _goal_policy_digest(request)
    root = tmp_path / "receipts"
    root.parent.joinpath(".arche_codex_goal_intent.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "logical_goal_id": "logical-goal-1",
                "thread_id": "thread-goal-1",
                "goal_id": "goal-1",
                "request_digest": "a" * 64,
                "thread_policy_digest": policy_digest,
                "objective_digest": hashlib.sha256(
                    request.prompt_text().encode("utf-8")
                ).hexdigest(),
                "turn_count_before": 0,
                "physical_turn_count_before": 0,
                "profile": "investigation",
                "stage": request.stage,
                "status": "active",
            }
        ),
        encoding="utf-8",
    )
    started: list[str] = []

    result = adapter.run_codex_sdk_turn(
        config=CodexConfig(),
        request=request,
        thread_config={},
        session_id="thread-goal-1",
        trace_path=tmp_path / "trace.jsonl",
        completed_turn_receipt_root=root,
        request_digest="a" * 64,
        turn_count_before=0,
        thread_policy_digest=policy_digest,
        enable_investigation_goals=True,
        on_turn_started=started.append,
    )

    assert result.turn_id == "turn-1"
    assert started == ["turn-1"]
    assert _FakeGoalCodex.thread.resume_calls == 1
    assert _FakeGoalCodex.thread.goal_calls == []


def test_goal_stream_without_started_identity_fails_closed(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Goal 流结束仍没有首个物理身份时，不能返回结果或伪造 Started。"""

    _FakeGoalCodex.thread = _FakeGoalThread()
    _FakeGoalCodex.thread.handle = _FakeGoalHandleWithoutStartedEvent()
    monkeypatch.setattr(adapter, "Codex", _FakeGoalCodex)
    request = _goal_request(tmp_path)
    _FakeGoalCodex.thread.history_prompt = request.prompt_text()
    policy_digest = _goal_policy_digest(request)
    started: list[str] = []

    with pytest.raises(adapter.CodexGoalRecoveryPending):
        adapter.run_codex_sdk_turn(
            config=CodexConfig(),
            request=request,
            thread_config={},
            session_id=None,
            trace_path=tmp_path / "trace.jsonl",
            completed_turn_receipt_root=tmp_path / "receipts",
            request_digest="a" * 64,
            turn_count_before=0,
            thread_policy_digest=policy_digest,
            enable_investigation_goals=True,
            on_turn_started=started.append,
        )

    assert started == []


@pytest.mark.parametrize("status", ["failed", "interrupted"])
def test_goal_failure_closes_working_memory_logical_turn_with_first_physical_id(
    monkeypatch,
    tmp_path: Path,
    status: str,
) -> None:
    """Goal 末个物理 turn 失败时，终态回调仍使用首个逻辑 turn 身份。"""

    _FakeGoalCodex.thread = _FakeGoalThread(status="blocked")
    _FakeGoalCodex.thread.handle.final_turn_status = status
    _FakeGoalCodex.thread.handle.goal_status = "blocked"
    _FakeGoalCodex.thread.handle.status = "blocked"
    monkeypatch.setattr(adapter, "Codex", _FakeGoalCodex)
    request = _goal_request(tmp_path)
    _FakeGoalCodex.thread.history_prompt = request.prompt_text()
    policy_digest = _goal_policy_digest(request)
    terminal: list[tuple[str, str, str]] = []

    with pytest.raises(adapter.CodexGoalRecoveryPending):
        adapter.run_codex_sdk_turn(
            config=CodexConfig(),
            request=request,
            thread_config={},
            session_id=None,
            trace_path=tmp_path / "trace.jsonl",
            completed_turn_receipt_root=tmp_path / "receipts",
            request_digest="a" * 64,
            turn_count_before=0,
            thread_policy_digest=policy_digest,
            enable_investigation_goals=True,
            on_goal_terminal_turn=lambda thread_id, turn_id, terminal_status: terminal.append(
                (thread_id, turn_id, terminal_status)
            ),
        )

    assert terminal == [("thread-goal-1", "turn-1", status)]


def test_completed_goal_replay_uses_first_prompt_last_answer_and_all_tools(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """完成 Goal 重放不启动新目标，并按完整物理范围恢复结果和工具项。"""

    _FakeGoalCodex.thread = _FakeGoalThread()
    monkeypatch.setattr(adapter, "Codex", _FakeGoalCodex)
    request = _goal_request(tmp_path)
    _FakeGoalCodex.thread.history_prompt = request.prompt_text()
    policy_digest = _goal_policy_digest(request)
    root = tmp_path / "receipts"
    first = adapter.run_codex_sdk_turn(
        config=CodexConfig(),
        request=request,
        thread_config={},
        session_id=None,
        trace_path=tmp_path / "first.jsonl",
        completed_turn_receipt_root=root,
        request_digest="a" * 64,
        turn_count_before=0,
        thread_policy_digest=policy_digest,
        enable_investigation_goals=True,
    )
    assert first.replayed is False
    assert (root.parent / ".arche_codex_goal_receipt.json").is_file()
    assert (root.parent / ".arche_codex_goal_intent.json").is_file()
    _FakeGoalCodex.thread.goal_calls.clear()
    _FakeGoalCodex.thread.resume_calls = 0

    replay = adapter.run_codex_sdk_turn(
        config=CodexConfig(),
        request=request,
        thread_config={},
        session_id="thread-goal-1",
        trace_path=tmp_path / "second.jsonl",
        completed_turn_receipt_root=root,
        request_digest="a" * 64,
        turn_count_before=0,
        thread_policy_digest=policy_digest,
        enable_investigation_goals=True,
    )

    assert replay.replayed is True
    assert replay.turn_id == "turn-1"
    assert replay.physical_turn_ids == ("turn-1", "turn-2")
    assert replay.response.content == '{"ok":true}'
    assert _FakeGoalCodex.thread.goal_calls == []
    assert _FakeGoalCodex.thread.resume_calls == 0
