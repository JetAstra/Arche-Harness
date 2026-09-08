"""验证 Codex SDK turn 的总体时限与 Investigation 配置边界。"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from pathlib import Path
import threading
import time
import traceback
from types import SimpleNamespace

import pytest

from openai_codex import ApprovalMode, CodexConfig, Sandbox
from openai_codex.errors import (
    StderrIndicator,
    TransportClosedError,
    TransportDiagnostic,
    TransportPhase,
)

from chemistry_multiagent.harness import codex_sdk_adapter as adapter
from chemistry_multiagent.harness.base import HarnessRequest, HarnessResponse
from chemistry_multiagent.harness.config import HarnessConfig
from chemistry_multiagent.harness.dispatcher import HarnessDispatcher
from chemistry_multiagent.harness.session_store import CodexSessionOwner, CodexSessionStore


@dataclass
class _FakeItemRoot:
    text: str = ""
    phase: str = ""
    type: str | None = None
    id: str | None = None
    process_id: str | None = None
    tool: str | None = None
    status: str | None = None
    arguments: object | None = None
    result: object | None = None


@dataclass
class _FakeItem:
    root: _FakeItemRoot


@dataclass
class _FakeTurnState:
    id: str
    status: str
    error: object | None = None


@dataclass
class _FakeTurnError:
    message: str


@dataclass
class _FakePayload:
    item: _FakeItem | None = None
    turn: _FakeTurnState | None = None
    provider_request_id: str | None = None


@dataclass
class _FakeEvent:
    method: str
    payload: _FakePayload


class _FakeTurn:
    def stream(self):
        yield _FakeEvent(
            method="item/completed",
            payload=_FakePayload(
                item=_FakeItem(
                    root=_FakeItemRoot(text="{\"ok\":true}", phase="final_answer")
                )
            ),
        )
        yield _FakeEvent(
            method="turn/completed",
            payload=_FakePayload(
                turn=_FakeTurnState(id="turn-1", status="completed")
            ),
        )


class _ToolPauseTurn:
    """先报告 MiniChem 工具开始，再短暂无事件后正常完成。"""

    id = "turn-tool-pause"

    def stream(self):
        yield _FakeEvent(
            method="item/started",
            payload=_FakePayload(
                item=_FakeItem(
                    root=_FakeItemRoot(type="mcpToolCall", id="mcp-tool-1")
                )
            ),
        )
        time.sleep(0.05)
        yield _FakeEvent(
            method="item/completed",
            payload=_FakePayload(
                item=_FakeItem(
                    root=_FakeItemRoot(type="mcpToolCall", id="mcp-tool-1")
                )
            ),
        )
        yield _FakeEvent(
            method="item/completed",
            payload=_FakePayload(
                item=_FakeItem(
                    root=_FakeItemRoot(
                        text='{"ok":true}',
                        phase="final_answer",
                        type="agentMessage",
                        id="agent-message-1",
                    )
                )
            ),
        )
        yield _FakeEvent(
            method="turn/completed",
            payload=_FakePayload(
                turn=_FakeTurnState(id=self.id, status="completed")
            ),
        )


class _AsyncJobWaitTurn:
    """报告异步作业仍在运行后，模拟模型等待下一次查询。"""

    id = "turn-async-job-wait"

    def stream(self):
        yield _FakeEvent(
            method="item/completed",
            payload=_FakePayload(
                item=_FakeItem(
                    root=_FakeItemRoot(
                        type="mcpToolCall",
                        id="mcp-submit-async",
                        tool="submit_action_job",
                        result={
                            "structuredContent": {
                                "status": "success",
                                "job_id": "job-async",
                                "job_status": "queued",
                                "terminal": False,
                                "job": {"job_id": "job-async", "status": "queued"},
                            }
                        },
                    )
                )
            ),
        )
        yield _FakeEvent(
            method="item/completed",
            payload=_FakePayload(
                item=_FakeItem(
                    root=_FakeItemRoot(
                        type="mcpToolCall",
                        id="mcp-get-running",
                        tool="get_execution_job",
                        arguments={"request": {"job_id": "job-async"}},
                        result={
                            "structuredContent": {
                                "status": "success",
                                "job_id": "job-async",
                                "job_status": "running",
                                "terminal": False,
                                "job": {"job_id": "job-async", "status": "running"},
                            }
                        },
                    )
                )
            ),
        )
        time.sleep(0.05)
        yield _FakeEvent(
            method="item/completed",
            payload=_FakePayload(
                item=_FakeItem(
                    root=_FakeItemRoot(
                        text='{"ok":true}',
                        phase="final_answer",
                        type="agentMessage",
                        id="agent-message-async-job",
                    )
                )
            ),
        )
        yield _FakeEvent(
            method="turn/completed",
            payload=_FakePayload(
                turn=_FakeTurnState(id=self.id, status="completed")
            ),
        )


class _AsyncJobSubmitOnlyWaitTurn:
    """提交非终态异步作业后，模拟模型等待下一次查询。"""

    id = "turn-async-job-submit-only"

    def stream(self):
        yield _FakeEvent(
            method="item/completed",
            payload=_FakePayload(
                item=_FakeItem(
                    root=_FakeItemRoot(
                        type="mcpToolCall",
                        id="mcp-submit-async-only",
                        tool="submit_action_job",
                        result={
                            "structuredContent": {
                                "status": "success",
                                "job_id": "job-async-submit-only",
                                "job_status": "queued",
                                "terminal": False,
                            }
                        },
                    )
                )
            ),
        )
        time.sleep(0.05)
        yield _FakeEvent(
            method="item/completed",
            payload=_FakePayload(
                item=_FakeItem(
                    root=_FakeItemRoot(
                        text='{"ok":true}',
                        phase="final_answer",
                        type="agentMessage",
                        id="agent-message-async-submit-only",
                    )
                )
            ),
        )
        yield _FakeEvent(
            method="turn/completed",
            payload=_FakePayload(
                turn=_FakeTurnState(id=self.id, status="completed")
            ),
        )


class _AsyncJobSubmitOnlyCodex:
    def __init__(self, _config):
        self.thread = _FakeThread()

    def thread_start(self, **_kwargs):
        self.thread.turn = lambda *_args, **_kwargs: _AsyncJobSubmitOnlyWaitTurn()
        return self.thread

    def close(self):
        return None


class _AsyncJobWaitCodex:
    def __init__(self, _config):
        self.thread = _FakeThread()

    def thread_start(self, **_kwargs):
        self.thread.turn = lambda *_args, **_kwargs: _AsyncJobWaitTurn()
        return self.thread

    def close(self):
        return None


class _AsyncJobCollectThenOrdinaryPauseTurn:
    """collect 清除作业等待后，再经历非异步工具和模型静默。"""

    id = "turn-async-job-collect-ordinary-pause"

    def __init__(self) -> None:
        self.interrupt_calls = 0
        self.released = threading.Event()

    def interrupt(self):
        self.interrupt_calls += 1
        self.released.set()

    def stream(self):
        yield _FakeEvent(
            method="item/completed",
            payload=_FakePayload(
                item=_FakeItem(
                    root=_FakeItemRoot(
                        type="mcpToolCall",
                        id="mcp-submit-collect-ordinary",
                        tool="submit_action_job",
                        result={
                            "structuredContent": {
                                "status": "success",
                                "job_id": "job-collect-ordinary",
                                "job_status": "queued",
                                "terminal": False,
                            }
                        },
                    )
                )
            ),
        )
        yield _FakeEvent(
            method="item/completed",
            payload=_FakePayload(
                item=_FakeItem(
                    root=_FakeItemRoot(
                        type="mcpToolCall",
                        id="mcp-get-collect-ordinary",
                        tool="get_execution_job",
                        result={
                            "structuredContent": {
                                "status": "success",
                                "job_id": "job-collect-ordinary",
                                "job_status": "running",
                                "terminal": False,
                            }
                        },
                    )
                )
            ),
        )
        yield _FakeEvent(
            method="item/completed",
            payload=_FakePayload(
                item=_FakeItem(
                    root=_FakeItemRoot(
                        type="mcpToolCall",
                        id="mcp-collect-ordinary",
                        tool="collect_execution_job",
                        result={
                            "structuredContent": {
                                "status": "success",
                                "job_id": "job-collect-ordinary",
                                "job_status": "succeeded",
                                "terminal": True,
                            }
                        },
                    )
                )
            ),
        )
        # list_software 是普通 MCP 工具；它不应把之后的模型推理误判为外部作业等待。
        yield _FakeEvent(
            method="item/completed",
            payload=_FakePayload(
                item=_FakeItem(
                    root=_FakeItemRoot(
                        type="mcpToolCall",
                        id="mcp-list-software",
                        tool="list_software",
                    )
                )
            ),
        )
        time.sleep(0.30)
        yield _FakeEvent(
            method="item/completed",
            payload=_FakePayload(
                item=_FakeItem(
                    root=_FakeItemRoot(
                        text='{"ok":true}',
                        phase="final_answer",
                        type="agentMessage",
                        id="agent-message-collect-ordinary",
                    )
                )
            ),
        )
        yield _FakeEvent(
            method="turn/completed",
            payload=_FakePayload(
                turn=_FakeTurnState(id=self.id, status="completed")
            ),
        )


class _AsyncJobCollectThenOrdinaryPauseCodex:
    def __init__(self, _config):
        self.thread = _FakeThread()
        self.turn = _AsyncJobCollectThenOrdinaryPauseTurn()

    def thread_start(self, **_kwargs):
        self.thread.turn = lambda *_args, **_kwargs: self.turn
        return self.thread

    def close(self):
        return None


class _AsyncJobTerminalThenIdleTurn:
    """作业已到终态后保持静默，用于验证会回到普通 watchdog。"""

    id = "turn-async-job-terminal-idle"

    def __init__(self):
        self.interrupt_calls = 0
        self.released = threading.Event()

    def interrupt(self):
        self.interrupt_calls += 1
        self.released.set()

    def stream(self):
        for status, terminal in (("queued", False), ("running", False), ("succeeded", True)):
            yield _FakeEvent(
                method="item/completed",
                payload=_FakePayload(
                    item=_FakeItem(
                        root=_FakeItemRoot(
                            type="mcpToolCall",
                            id=f"mcp-job-{status}",
                            tool=(
                                "submit_action_job"
                                if status == "queued"
                                else "get_execution_job"
                            ),
                            arguments={"request": {"job_id": "job-terminal"}},
                            result={
                                "structuredContent": {
                                    "status": "success",
                                    "job_id": "job-terminal",
                                    "job_status": status,
                                    "terminal": terminal,
                                    "job": {
                                        "job_id": "job-terminal",
                                        "status": status,
                                    },
                                }
                            },
                        )
                    )
                ),
            )
        self.released.wait()


class _AsyncJobTerminalThenIdleCodex:
    def __init__(self, _config):
        self.thread = _FakeThread()
        self.turn = _AsyncJobTerminalThenIdleTurn()

    def thread_start(self, **_kwargs):
        self.thread.turn = lambda *_args, **_kwargs: self.turn
        return self.thread

    def close(self):
        return None


class _AcceptedSubmitWithoutAssistantTurn:
    """只返回已接受的异步作业回执，不生成 assistant 正文。"""

    id = "turn-submit-without-assistant"

    def stream(self):
        item = _FakeItem(
            root=_FakeItemRoot(
                type="mcpToolCall",
                id="mcp-submit-1",
                tool="submit_transition_state_path_job",
                status="completed",
                arguments={"request": "opaque-submit"},
                result={
                    "structuredContent": {
                        "job_id": "job-accepted",
                        "operation_id": "operation-accepted",
                        "submission_key": "submission-accepted",
                        "action_id": "search_transition_state_path",
                        "backend_id": "transition_state_path_search",
                    }
                },
                text="",
                phase=None,
            )
        )
        yield _FakeEvent(method="item/completed", payload=_FakePayload(item=item))
        yield _FakeEvent(
            method="turn/completed",
            payload=_FakePayload(
                turn=_FakeTurnState(id=self.id, status="completed")
            ),
        )


class _CommandPauseTurn:
    """启动 commandExecution 后无 stdout 一小段时间再正常完成。"""

    id = "turn-command-pause"

    def stream(self):
        yield _FakeEvent(
            method="item/started",
            payload=_FakePayload(
                item=_FakeItem(
                    root=_FakeItemRoot(
                        type="commandExecution",
                        id="command-item-1",
                    )
                )
            ),
        )
        time.sleep(0.05)
        yield _FakeEvent(
            method="item/completed",
            payload=_FakePayload(
                item=_FakeItem(
                    root=_FakeItemRoot(
                        type="commandExecution",
                        id="command-item-1",
                        process_id="command-process-1",
                    )
                )
            ),
        )
        yield _FakeEvent(
            method="item/completed",
            payload=_FakePayload(
                item=_FakeItem(
                    root=_FakeItemRoot(
                        text='{"ok":true}',
                        phase="final_answer",
                        type="agentMessage",
                        id="agent-message-command-1",
                    )
                )
            ),
        )
        yield _FakeEvent(
            method="turn/completed",
            payload=_FakePayload(
                turn=_FakeTurnState(id=self.id, status="completed")
            ),
        )


class _CommandStuckTurn:
    """启动 commandExecution 后无输出，用于验证工具局部上限。"""

    id = "turn-command-stuck"

    def __init__(self, *, interruptible: bool) -> None:
        self.interruptible = interruptible
        self.interrupt_calls = 0
        self.released = threading.Event()

    def interrupt(self):
        self.interrupt_calls += 1
        if self.interruptible:
            self.released.set()

    def stream(self):
        yield _FakeEvent(
            method="item/started",
            payload=_FakePayload(
                item=_FakeItem(
                    root=_FakeItemRoot(
                        type="commandExecution",
                        id="command-item-stuck",
                    )
                )
            ),
        )
        self.released.wait()


class _CommandProcessExitedTurn:
    """命令进程异常退出且没有 item/completed，验证映射会清除命令 item。"""

    id = "turn-command-exited"

    def __init__(self) -> None:
        self.interrupt_calls = 0
        self.released = threading.Event()

    def interrupt(self):
        self.interrupt_calls += 1
        self.released.set()

    def stream(self):
        yield _FakeEvent(
            method="item/started",
            payload=_FakePayload(
                item=_FakeItem(
                    root=_FakeItemRoot(
                        type="commandExecution",
                        id="command-item-exited",
                        process_id="process-exited-1",
                    )
                )
            ),
        )
        yield _FakeEvent(
            method="process/outputDelta",
            payload={"params": {"processHandle": "process-exited-1"}},
        )
        yield _FakeEvent(
            method="process/exited",
            payload={"params": {"processHandle": "process-exited-1"}},
        )
        self.released.wait()


class _FakeThread:
    id = "thread-1"

    def turn(self, *_args, **_kwargs):
        return _FakeTurn()


class _IdleTurn:
    """只产生 started 后保持空闲的 turn，可由 interrupt 结束。"""

    id = "turn-idle"

    def __init__(self, *, interruptible: bool) -> None:
        self.interruptible = interruptible
        self.interrupt_calls = 0
        self.released = threading.Event()

    def interrupt(self):
        self.interrupt_calls += 1
        if self.interruptible:
            self.released.set()

    def stream(self):
        self.released.wait()
        if False:  # pragma: no cover - 仅让函数保持生成器形状
            yield None


class _ToolStuckTurn(_IdleTurn):
    """报告工具开始后失去事件，用于验证工具局部上限仍会触发。"""

    def stream(self):
        yield _FakeEvent(
            method="item/started",
            payload=_FakePayload(
                item=_FakeItem(
                    root=_FakeItemRoot(type="mcpToolCall", id="mcp-tool-stuck")
                )
            ),
        )
        self.released.wait()


class _IdleThread(_FakeThread):
    def __init__(self, turn: _IdleTurn) -> None:
        self.turn_handle = turn

    def turn(self, *_args, **_kwargs):
        return self.turn_handle


class _IdleCodex:
    def __init__(self, turn: _IdleTurn) -> None:
        self.thread = _IdleThread(turn)

    def thread_start(self, **_kwargs):
        return self.thread

    def thread_resume(self, *_args, **_kwargs):
        return self.thread

    def close(self):
        return None


class _ToolPauseCodex:
    def __init__(self, _config):
        self.thread = _FakeThread()

    def thread_start(self, **_kwargs):
        self.thread.turn = lambda *_args, **_kwargs: _ToolPauseTurn()
        return self.thread

    def close(self):
        return None


class _AcceptedSubmitWithoutAssistantCodex:
    """返回不含 assistant 正文的已接受 submit turn。"""

    def __init__(self, _config):
        self.thread = _FakeThread()

    def thread_start(self, **_kwargs):
        self.thread.turn = lambda *_args, **_kwargs: _AcceptedSubmitWithoutAssistantTurn()
        return self.thread

    def close(self):
        return None


class _CommandPauseCodex:
    def __init__(self, _config):
        self.thread = _FakeThread()

    def thread_start(self, **_kwargs):
        self.thread.turn = lambda *_args, **_kwargs: _CommandPauseTurn()
        return self.thread

    def close(self):
        return None


class _CommandStuckThread(_FakeThread):
    def __init__(self, turn: _CommandStuckTurn) -> None:
        self.turn_handle = turn

    def turn(self, *_args, **_kwargs):
        return self.turn_handle


class _CommandStuckCodex:
    def __init__(self, turn: _CommandStuckTurn) -> None:
        self.thread = _CommandStuckThread(turn)

    def thread_start(self, **_kwargs):
        return self.thread

    def close(self):
        return None


class _CapturingThread(_FakeThread):
    def __init__(self, calls: list[tuple[str, object]]) -> None:
        self._calls = calls

    def turn(self, *args, **kwargs):
        self._calls.append(("turn", (args, kwargs)))
        return _FakeTurn()


class _CapturingCodex:
    calls: list[tuple[str, object]] = []

    def __init__(self, _config):
        self.thread = _CapturingThread(self.calls)

    def thread_start(self, **kwargs):
        self.calls.append(("thread_start", kwargs))
        return self.thread

    def thread_resume(self, thread_id, **kwargs):
        self.calls.append(("thread_resume", (thread_id, kwargs)))
        return self.thread

    def close(self):
        return None


class _FakeCodex:
    def __init__(self, _config):
        self.thread = _FakeThread()

    def thread_start(self, **_kwargs):
        return self.thread

    def thread_resume(self, *_args, **_kwargs):  # pragma: no cover - 防止错误路由
        return self.thread

    def close(self):
        return None


class _LeakingTurn:
    """发出包含额外敏感值的 notification，验证适配器写 trace 前脱敏。"""

    def stream(self):
        yield SimpleNamespace(
            method="mcpServer/notification",
            payload={"echo": "explicit-header-secret"},
        )
        yield _FakeEvent(
            method="item/completed",
            payload=_FakePayload(
                item=_FakeItem(
                    root=_FakeItemRoot(text='{"ok":true}', phase="final_answer")
                )
            ),
        )
        yield _FakeEvent(
            method="turn/completed",
            payload=_FakePayload(turn=_FakeTurnState(id="turn-1", status="completed")),
        )


class _LeakingThread(_FakeThread):
    def turn(self, *_args, **_kwargs):
        return _LeakingTurn()


class _LeakingCodex:
    def __init__(self, _config):
        self.thread = _LeakingThread()

    def thread_start(self, **_kwargs):
        return self.thread

    def thread_resume(self, *_args, **_kwargs):
        return self.thread

    def close(self):
        return None


class _ReadableThread:
    """提供可重放的最小 thread/read 历史，并统计实际模型 turn 次数。"""

    id = "thread-replay"

    def __init__(self) -> None:
        self.turn_calls = 0
        self.read_calls: list[bool] = []
        self.prompt_text = ""
        self.history_tool_items: tuple[Any, ...] = ()

    def turn(self, prompt, *_args, **_kwargs):
        self.turn_calls += 1
        self.prompt_text = prompt
        return _FakeTurn()

    def read(self, *, include_turns: bool = False):
        self.read_calls.append(include_turns)
        return {
            "thread": {
                "id": self.id,
                "turns": [
                    {
                        "id": "turn-1",
                        "status": "completed",
                        "items_view": "full",
                        "items": [
                            {
                                "type": "userMessage",
                                "content": [{"type": "text", "text": self.prompt_text}],
                            },
                            *self.history_tool_items,
                            {
                                "type": "agentMessage",
                                "phase": "final_answer",
                                "text": '{"ok":true}',
                            },
                        ],
                    }
                ],
            }
        }


class _ReadableCodex:
    thread = _ReadableThread()

    def __init__(self, _config):
        return None

    def thread_start(self, **_kwargs):
        return self.thread

    def thread_resume(self, *_args, **_kwargs):
        return self.thread

    def close(self):
        return None


class _IntentCrashTurn:
    """在请求已提交前模拟 worker 崩溃，history 中不产生新 turn。"""

    id = "turn-intent-crash"

    def stream(self):
        raise RuntimeError("simulated crash after request intent")
        yield None


class _IntentThread:
    """只保留空 history 的 thread，统计是否发生第二次模型 turn。"""

    id = "thread-intent"

    def __init__(self) -> None:
        self.turn_calls = 0
        self.read_calls = 0

    def turn(self, *_args, **_kwargs):
        self.turn_calls += 1
        return _IntentCrashTurn()

    def read(self, *, include_turns: bool = False):
        self.read_calls += 1
        assert include_turns is True
        return {"thread": {"id": self.id, "turns": []}}


class _IntentCodex:
    thread = _IntentThread()

    def __init__(self, _config):
        return None

    def thread_start(self, **_kwargs):
        return self.thread

    def thread_resume(self, *_args, **_kwargs):
        return self.thread

    def close(self):
        return None


class _HistoryCrashTurn:
    """把可选终态写进模拟 history 后，在 notification 前崩溃。"""

    def __init__(self, owner: "_HistoryCrashThread") -> None:
        self.owner = owner
        self.id = f"turn-crash-{owner.turn_calls}"

    def stream(self):
        if self.owner.crash_status is not None:
            self.owner.history.append(
                {"id": self.id, "status": self.owner.crash_status, "items": []}
            )
        raise RuntimeError("simulated crash before terminal notification")
        yield None


class _HistoryCrashThread:
    """保存可变物理 history，用于覆盖 notification 前后的崩溃窗口。"""

    id = "thread-history-crash"

    def __init__(self, history: list[dict[str, object]], crash_status: str | None) -> None:
        self.history = list(history)
        self.crash_status = crash_status
        self.turn_calls = 0

    def turn(self, *_args, **_kwargs):
        self.turn_calls += 1
        return _HistoryCrashTurn(self)

    def read(self, *, include_turns: bool = False):
        assert include_turns is True
        return {"thread": {"id": self.id, "turns": list(self.history)}}


class _HistoryCrashCodex:
    """把测试指定的可变 history thread 暴露给 adapter。"""

    thread = _HistoryCrashThread([], None)

    def __init__(self, _config):
        return None

    def thread_start(self, **_kwargs):
        return self.thread

    def thread_resume(self, *_args, **_kwargs):
        return self.thread

    def close(self):
        return None


class _TerminalFailureTurn:
    """返回明确失败终态；interrupt 会暴露已结束 turn 的竞态。"""

    id = "turn-terminal-failed"

    def __init__(self, status: str) -> None:
        self.status = status
        self.interrupt_calls = 0

    def interrupt(self):
        self.interrupt_calls += 1
        raise RuntimeError("no active turn to interrupt")

    def stream(self):
        yield _FakeEvent(
            method="turn/completed",
            payload=_FakePayload(
                turn=_FakeTurnState(
                    id=self.id,
                    status=self.status,
                    error=_FakeTurnError(message="quota rejected synthetic-secret"),
                ),
                provider_request_id="provider-request-unknown-text",
            ),
        )


class _ErrorThenTerminalFailureTurn(_TerminalFailureTurn):
    """先返回含未知 provider 正文的 error，再返回明确失败终态。"""

    def stream(self):
        yield _FakeEvent(
            method="error",
            payload={
                "error": {
                    "message": "provider-free-text-must-not-persist",
                    "request_id": "provider-error-request-id",
                },
                "thread_id": "thread-terminal-retry",
                "turn_id": self.id,
                "willRetry": True,
                "provider_request_id": "provider-error-outer-request-id",
            },
        )
        yield _FakeEvent(
            method="error",
            payload={
                "error": {"message": "second-provider-free-text"},
                "will_retry": "true",
            },
        )
        yield from super().stream()


class _TerminalThenSuccessThread(_FakeThread):
    """同一 thread 首轮明确失败，下一轮正常完成。"""

    id = "thread-terminal-retry"

    def __init__(self, status: str = "failed") -> None:
        self.failed_turn = _TerminalFailureTurn(status)
        self.turn_calls = 0

    def turn(self, *_args, **_kwargs):
        self.turn_calls += 1
        if self.turn_calls == 1:
            return self.failed_turn
        return _FakeTurn()

    def read(self, *, include_turns: bool = False):
        assert include_turns is True
        return {
            "thread": {
                "id": self.id,
                "turns": [
                    {
                        "id": self.failed_turn.id,
                        "status": self.failed_turn.status,
                        "items_view": "full",
                        "items": [],
                    }
                ],
            }
        }


class _TerminalThenSuccessCodex:
    thread = _TerminalThenSuccessThread()

    def __init__(self, _config):
        return None

    def thread_start(self, **_kwargs):
        return self.thread

    def thread_resume(self, *_args, **_kwargs):
        return self.thread

    def close(self):
        return None


def _compiled_request(*, stage: str) -> dict[str, object]:
    return {
        "catalog_key": stage,
        "role": "planner",
        "stage": stage,
        "model": "codex-model",
        "messages": [{"role": "user", "content": "investigate"}],
    }


def test_dispatcher_only_removes_deadline_for_configured_investigation_stage(
    tmp_path: Path,
) -> None:
    """只有 Codex Planner 的显式 Investigation stage 才使用 None 总时限。"""

    captured: list[HarnessRequest] = []

    class _FakeRunner:
        def run(self, request):
            captured.append(request)
            return HarnessResponse(content="ok", model=request.model)

    dispatcher = HarnessDispatcher(
        HarnessConfig(
            kind="codex",
            workspace_root=tmp_path / "workspaces",
            trace_root=tmp_path / "traces",
            unbounded_turn_stages=frozenset({"investigation_and_experiment"}),
        ),
        react_fallback=lambda request, **kwargs: {},
    )
    dispatcher._runner = _FakeRunner()

    dispatcher.run_compiled(
        _compiled_request(stage="investigation_and_experiment"), timeout=7
    )
    dispatcher.run_compiled(_compiled_request(stage="workflow_generation"), timeout=7)

    assert [request.timeout_seconds for request in captured] == [None, 7.0]


def test_harness_config_reads_explicit_unbounded_stage_list(monkeypatch) -> None:
    """阶段集合通过环境变量显式配置，默认不会放宽旧阶段。"""

    monkeypatch.setenv("ARCHE_PLANNER_DRIVER", "codex")
    monkeypatch.setenv("ARCHE_CODEX_API_KEY", "unit-test-key")
    monkeypatch.setenv("ARCHE_CODEX_BASE_URL", "https://codex.example/v1")
    monkeypatch.setenv("ARCHE_CODEX_MODEL_NAME", "codex-model")
    monkeypatch.setenv(
        "ARCHE_CODEX_UNBOUNDED_TURN_STAGES",
        " Investigation_and_Experiment, ,another_stage ",
    )
    config = HarnessConfig.from_env()

    assert config.unbounded_turn_stages == frozenset(
        {"investigation_and_experiment", "another_stage"}
    )
    assert config.uses_unbounded_turn_timeout(
        role="planner", stage="INVESTIGATION_AND_EXPERIMENT"
    )
    assert not config.uses_unbounded_turn_timeout(
        role="hypothesis", stage="investigation_and_experiment"
    )


def test_sdk_adapter_accepts_explicit_unbounded_turn(monkeypatch, tmp_path: Path) -> None:
    """SDK turn 为 None 总时限时仍消费完整事件并返回最终正文。"""

    monkeypatch.setattr(adapter, "Codex", _FakeCodex)
    request = HarnessRequest(
        messages=(
            {"role": "user", "content": "investigate"},
        ),
        model="codex-model",
        role="planner",
        stage="investigation_and_experiment",
        workspace=tmp_path / "workspace",
        timeout_seconds=None,
    )

    result = adapter.run_codex_sdk_turn(
        config=CodexConfig(),
        request=request,
        thread_config={},
        session_id=None,
        trace_path=tmp_path / "trace.jsonl",
    )

    assert result.response.content == '{"ok":true}'
    assert result.thread_id == "thread-1"
    assert (tmp_path / "workspace").stat().st_mode & 0o077 == 0
    assert (tmp_path / "trace.jsonl").stat().st_mode & 0o077 == 0


def test_sdk_adapter_uses_role_specific_idle_defaults(
    monkeypatch, tmp_path: Path
) -> None:
    """直接调用 adapter 时 Investigation 与 Task Review 使用各自默认值。"""

    observed: list[float] = []
    original_consumer = adapter._consume_codex_turn_events

    def capture_consumer(**kwargs):
        observed.append(kwargs["idle_timeout_seconds"])
        return original_consumer(**kwargs)

    monkeypatch.setattr(adapter, "Codex", _FakeCodex)
    monkeypatch.setattr(adapter, "_consume_codex_turn_events", capture_consumer)
    for index, role in enumerate(("investigation", "task_review")):
        request = HarnessRequest(
            messages=({"role": "user", "content": role},),
            model="codex-model",
            role=role,
            stage="investigation_and_experiment" if role == "investigation" else "task_review",
            workspace=tmp_path / f"workspace-{index}",
            timeout_seconds=None,
        )
        adapter.run_codex_sdk_turn(
            config=CodexConfig(),
            request=request,
            thread_config={},
            session_id=None,
            trace_path=tmp_path / f"trace-{index}.jsonl",
        )

    assert observed == [1800.0, 300.0]


def test_sdk_adapter_rejects_thread_without_protocol_identity(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """app-server 未返回 thread id 时不能继续进入 receipt 或模型 turn。"""

    class _NoIdentityCodex:
        def __init__(self, _config):
            self.closed = False

        def thread_start(self, **_kwargs):
            return SimpleNamespace(id="")

        def close(self):
            self.closed = True

    codex_instances: list[_NoIdentityCodex] = []

    def create_codex(config):
        instance = _NoIdentityCodex(config)
        codex_instances.append(instance)
        return instance

    monkeypatch.setattr(adapter, "Codex", create_codex)
    request = HarnessRequest(
        messages=({"role": "user", "content": "missing thread id"},),
        model="codex-model",
        role="task_review",
        stage="workflow_revision",
        workspace=tmp_path / "workspace",
    )

    with pytest.raises(adapter.HarnessError, match="no thread id"):
        adapter.run_codex_sdk_turn(
            config=CodexConfig(),
            request=request,
            thread_config={},
            session_id=None,
            trace_path=tmp_path / "trace.jsonl",
        )
    assert codex_instances and codex_instances[0].closed


def test_sdk_adapter_idle_timeout_interrupts_and_bounded_joins_worker(
    monkeypatch, tmp_path: Path
) -> None:
    """事件流空闲只触发 transport watchdog，interrupt 可用时不会遗留 worker。"""

    turn = _IdleTurn(interruptible=True)
    monkeypatch.setattr(adapter, "Codex", lambda _config: _IdleCodex(turn))
    request = HarnessRequest(
        messages=({"role": "user", "content": "idle"},),
        model="codex-model",
        role="investigation",
        stage="investigation_and_experiment",
        workspace=tmp_path / "workspace",
        timeout_seconds=None,
    )

    started = time.monotonic()
    with pytest.raises(adapter.HarnessError, match="idle timeout") as failure:
        adapter.run_codex_sdk_turn(
            config=CodexConfig(),
            request=request,
            thread_config={},
            session_id=None,
            trace_path=tmp_path / "trace.jsonl",
            idle_timeout_seconds=0.02,
        )

    assert failure.value.kind == "transport"
    assert turn.interrupt_calls == 1
    assert time.monotonic() - started < 2.5


def test_sdk_adapter_tool_activity_uses_mcp_local_idle_budget(
    monkeypatch, tmp_path: Path
) -> None:
    """已开始的 MCP 工具超过普通空闲阈值时仍可继续，且没有总体 turn 截止时间。"""

    monkeypatch.setattr(adapter, "Codex", _ToolPauseCodex)
    request = HarnessRequest(
        messages=({"role": "user", "content": "long minichem call"},),
        model="codex-model",
        role="investigation",
        stage="investigation_and_experiment",
        workspace=tmp_path / "workspace",
        timeout_seconds=None,
    )

    result = adapter.run_codex_sdk_turn(
        config=CodexConfig(),
        request=request,
        thread_config={},
        session_id=None,
        trace_path=tmp_path / "trace.jsonl",
        idle_timeout_seconds=0.01,
        tool_idle_timeout_seconds=0.2,
    )

    assert result.response.content == '{"ok":true}'
    assert result.turn_id == "turn-tool-pause"


def test_sdk_adapter_preserves_external_job_wait_budget_after_running_status(
    monkeypatch, tmp_path: Path
) -> None:
    """已确认外部作业仍运行时，模型短暂无事件不会触发普通空闲检测。"""

    monkeypatch.setattr(adapter, "Codex", _AsyncJobWaitCodex)
    request = HarnessRequest(
        messages=(
            {"role": "user", "content": "wait for the async job"},
        ),
        model="codex-model",
        role="investigation",
        stage="investigation_and_experiment",
        workspace=tmp_path / "workspace",
        timeout_seconds=None,
    )

    result = adapter.run_codex_sdk_turn(
        config=CodexConfig(),
        request=request,
        thread_config={},
        session_id=None,
        trace_path=tmp_path / "trace.jsonl",
        idle_timeout_seconds=0.01,
        tool_idle_timeout_seconds=0.2,
    )

    assert result.response.content == '{"ok":true}'
    assert result.turn_id == "turn-async-job-wait"


def test_sdk_adapter_preserves_external_job_wait_budget_after_submit_status(
    monkeypatch, tmp_path: Path
) -> None:
    """提交回执已声明作业非终态时，后续长计算不会触发普通空闲检测。"""

    monkeypatch.setattr(adapter, "Codex", _AsyncJobSubmitOnlyCodex)
    request = HarnessRequest(
        messages=(
            {"role": "user", "content": "wait after submitting the async job"},
        ),
        model="codex-model",
        role="investigation",
        stage="investigation_and_experiment",
        workspace=tmp_path / "workspace",
        timeout_seconds=None,
    )

    result = adapter.run_codex_sdk_turn(
        config=CodexConfig(),
        request=request,
        thread_config={},
        session_id=None,
        trace_path=tmp_path / "trace.jsonl",
        idle_timeout_seconds=0.01,
        tool_idle_timeout_seconds=0.2,
    )

    assert result.response.content == '{"ok":true}'
    assert result.turn_id == "turn-async-job-submit-only"


def test_sdk_adapter_collect_then_ordinary_tool_uses_finite_model_idle_budget(
    monkeypatch, tmp_path: Path
) -> None:
    """collect 后的普通工具静默仍受限，但可超过旧 300 秒阈值的缩放回归。"""

    # 测试把真实时间放大 5 倍：0.30 秒模型静默代表 1.5 秒，分别超过旧
    # 0.3 秒但仍低于新 1.8 秒预算，避免为了验证 300/1800 秒而真实等待。
    real_monotonic = time.monotonic

    class _ScaledClock:
        def monotonic(self) -> float:
            return real_monotonic() * 5

    monkeypatch.setattr(adapter, "time", _ScaledClock())
    request = HarnessRequest(
        messages=({"role": "user", "content": "continue after collect"},),
        model="codex-model",
        role="investigation",
        stage="investigation_and_experiment",
        workspace=tmp_path / "workspace-new-budget",
        timeout_seconds=None,
    )

    monkeypatch.setattr(adapter, "Codex", _AsyncJobCollectThenOrdinaryPauseCodex)
    with pytest.raises(adapter.HarnessError, match="idle timeout"):
        adapter.run_codex_sdk_turn(
            config=CodexConfig(),
            request=request,
            thread_config={},
            session_id=None,
            trace_path=tmp_path / "trace-old-budget.jsonl",
            idle_timeout_seconds=0.3,
            tool_idle_timeout_seconds=1.0,
        )

    request = HarnessRequest(
        messages=({"role": "user", "content": "continue after collect"},),
        model="codex-model",
        role="investigation",
        stage="investigation_and_experiment",
        workspace=tmp_path / "workspace-new-budget-2",
        timeout_seconds=None,
    )
    result = adapter.run_codex_sdk_turn(
        config=CodexConfig(),
        request=request,
        thread_config={},
        session_id=None,
        trace_path=tmp_path / "trace-new-budget.jsonl",
        idle_timeout_seconds=1.8,
        tool_idle_timeout_seconds=1.0,
    )

    assert result.response.content == '{"ok":true}'
    assert result.turn_id == "turn-async-job-collect-ordinary-pause"


def test_sdk_adapter_returns_to_normal_idle_budget_after_job_terminal_status(
    monkeypatch, tmp_path: Path
) -> None:
    """外部作业明确终止后，静默模型仍必须触发普通空闲检测。"""

    codex = _AsyncJobTerminalThenIdleCodex(None)
    monkeypatch.setattr(adapter, "Codex", lambda _config: codex)
    request = HarnessRequest(
        messages=({"role": "user", "content": "stop after terminal job"},),
        model="codex-model",
        role="investigation",
        stage="investigation_and_experiment",
        workspace=tmp_path / "workspace",
        timeout_seconds=None,
    )

    with pytest.raises(adapter.HarnessError, match="idle timeout") as failure:
        adapter.run_codex_sdk_turn(
            config=CodexConfig(),
            request=request,
            thread_config={},
            session_id=None,
            trace_path=tmp_path / "trace.jsonl",
            idle_timeout_seconds=0.01,
            tool_idle_timeout_seconds=0.2,
        )

    assert failure.value.kind == "transport"
    assert codex.turn.interrupt_calls == 1


def test_sdk_adapter_rejects_investigation_submit_without_assistant_content(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """提交工具完成后仍必须返回当前 turn 的最终正文。"""

    monkeypatch.setattr(adapter, "Codex", _AcceptedSubmitWithoutAssistantCodex)
    request = HarnessRequest(
        messages=({"role": "user", "content": "submit one job"},),
        model="codex-model",
        role="investigation",
        stage="initial_investigation",
        workspace=tmp_path / "workspace",
        timeout_seconds=None,
    )
    completed_items: list[object] = []

    with pytest.raises(adapter.HarnessError, match="no final assistant content") as failure:
        adapter.run_codex_sdk_turn(
            config=CodexConfig(),
            request=request,
            thread_config={},
            session_id=None,
            trace_path=tmp_path / "trace.jsonl",
            on_tool_completed=completed_items.append,
        )

    assert failure.value.kind == "empty_content"
    assert len(completed_items) == 1


def test_sdk_adapter_notifies_real_turn_identity_after_acceptance(
    monkeypatch, tmp_path: Path
) -> None:
    """只有 SDK 返回真实 turn 句柄后才把其身份交给持久状态层。"""

    monkeypatch.setattr(adapter, "Codex", _ToolPauseCodex)
    request = HarnessRequest(
        messages=({"role": "user", "content": "record accepted turn"},),
        model="codex-model",
        role="investigation",
        stage="investigation_and_experiment",
        workspace=tmp_path / "workspace",
        timeout_seconds=None,
    )
    started_turn_ids: list[str] = []

    result = adapter.run_codex_sdk_turn(
        config=CodexConfig(),
        request=request,
        thread_config={},
        session_id=None,
        trace_path=tmp_path / "trace.jsonl",
        on_turn_started=started_turn_ids.append,
    )

    assert started_turn_ids == ["turn-tool-pause"]
    assert result.turn_id == started_turn_ids[0]


def test_sdk_adapter_command_execution_uses_tool_idle_budget_without_stdout(
    monkeypatch, tmp_path: Path
) -> None:
    """已开始的 commandExecution 无 stdout 时不会触发普通 300 秒检测。"""

    monkeypatch.setattr(adapter, "Codex", _CommandPauseCodex)
    request = HarnessRequest(
        messages=({"role": "user", "content": "long command"},),
        model="codex-model",
        role="investigation",
        stage="investigation_and_experiment",
        workspace=tmp_path / "workspace",
        timeout_seconds=None,
    )

    result = adapter.run_codex_sdk_turn(
        config=CodexConfig(),
        request=request,
        thread_config={},
        session_id=None,
        trace_path=tmp_path / "trace.jsonl",
        idle_timeout_seconds=0.01,
        tool_idle_timeout_seconds=0.2,
    )

    assert result.response.content == '{"ok":true}'
    assert result.turn_id == "turn-command-pause"


def test_sdk_adapter_stuck_mcp_tool_hits_finite_local_idle_limit(
    monkeypatch, tmp_path: Path
) -> None:
    """MCP 工具即使使用较长局部预算，也不能在失联时无限等待。"""

    turn = _ToolStuckTurn(interruptible=True)
    monkeypatch.setattr(adapter, "Codex", lambda _config: _IdleCodex(turn))
    request = HarnessRequest(
        messages=({"role": "user", "content": "stuck minichem call"},),
        model="codex-model",
        role="investigation",
        stage="investigation_and_experiment",
        workspace=tmp_path / "workspace",
        timeout_seconds=None,
    )

    started = time.monotonic()
    with pytest.raises(adapter.HarnessError, match="idle timeout") as failure:
        adapter.run_codex_sdk_turn(
            config=CodexConfig(),
            request=request,
            thread_config={},
            session_id=None,
            trace_path=tmp_path / "trace.jsonl",
            idle_timeout_seconds=0.01,
            tool_idle_timeout_seconds=0.02,
        )

    assert failure.value.kind == "transport"
    assert turn.interrupt_calls == 1
    assert time.monotonic() - started < 2.5


def test_sdk_adapter_stuck_command_execution_hits_tool_idle_limit(
    monkeypatch, tmp_path: Path
) -> None:
    """commandExecution 无输出时最终仍受有限工具局部上限约束。"""

    turn = _CommandStuckTurn(interruptible=True)
    monkeypatch.setattr(adapter, "Codex", lambda _config: _CommandStuckCodex(turn))
    request = HarnessRequest(
        messages=({"role": "user", "content": "stuck command"},),
        model="codex-model",
        role="investigation",
        stage="investigation_and_experiment",
        workspace=tmp_path / "workspace",
        timeout_seconds=None,
    )

    started = time.monotonic()
    with pytest.raises(adapter.HarnessError, match="idle timeout") as failure:
        adapter.run_codex_sdk_turn(
            config=CodexConfig(),
            request=request,
            thread_config={},
            session_id=None,
            trace_path=tmp_path / "trace.jsonl",
            idle_timeout_seconds=0.01,
            tool_idle_timeout_seconds=0.02,
        )

    assert failure.value.kind == "transport"
    assert turn.interrupt_calls == 1
    assert time.monotonic() - started < 2.5


def test_sdk_adapter_process_exit_clears_command_item_mapping(
    monkeypatch, tmp_path: Path
) -> None:
    """process/exited 清除关联 command item 后，失去事件会回到普通空闲上限。"""

    turn = _CommandProcessExitedTurn()
    monkeypatch.setattr(adapter, "Codex", lambda _config: _IdleCodex(turn))
    request = HarnessRequest(
        messages=({"role": "user", "content": "command exited"},),
        model="codex-model",
        role="investigation",
        stage="investigation_and_experiment",
        workspace=tmp_path / "workspace",
        timeout_seconds=None,
    )

    started = time.monotonic()
    with pytest.raises(adapter.HarnessError, match="idle timeout") as failure:
        adapter.run_codex_sdk_turn(
            config=CodexConfig(),
            request=request,
            thread_config={},
            session_id=None,
            trace_path=tmp_path / "trace.jsonl",
            idle_timeout_seconds=0.02,
            tool_idle_timeout_seconds=0.2,
        )

    assert failure.value.kind == "transport"
    assert turn.interrupt_calls == 1
    assert time.monotonic() - started < 0.15


def test_sdk_adapter_unstoppable_worker_returns_recovery_pending(
    monkeypatch, tmp_path: Path
) -> None:
    """interrupt 无法结束 worker 时返回持久化所需的 recovery_pending 错误。"""

    turn = _IdleTurn(interruptible=False)
    monkeypatch.setattr(adapter, "Codex", lambda _config: _IdleCodex(turn))
    request = HarnessRequest(
        messages=({"role": "user", "content": "stuck"},),
        model="codex-model",
        role="investigation",
        stage="investigation_and_experiment",
        workspace=tmp_path / "workspace",
        timeout_seconds=None,
    )

    started = time.monotonic()
    recovery_turn_ids: list[str | None] = []
    try:
        with pytest.raises(adapter.CodexWorkerRecoveryPending) as failure:
            adapter.run_codex_sdk_turn(
                config=CodexConfig(),
                request=request,
                thread_config={},
                session_id=None,
                trace_path=tmp_path / "trace.jsonl",
                idle_timeout_seconds=0.02,
                on_recovery_pending=recovery_turn_ids.append,
            )
        assert failure.value.kind == "recovery_pending"
        assert failure.value.turn_id == "turn-idle"
        assert turn.interrupt_calls == 1
        assert recovery_turn_ids == ["turn-idle"]
        assert time.monotonic() - started < 2.5
    finally:
        turn.released.set()


def test_sdk_adapter_receipt_write_failure_leaves_journal_and_recovers_history(
    monkeypatch, tmp_path: Path
) -> None:
    """receipt 写失败时只留下 completion journal，恢复严格读 history 且不重发模型。"""

    _ReadableCodex.thread = _ReadableThread()
    monkeypatch.setattr(adapter, "Codex", _ReadableCodex)
    monkeypatch.setattr(adapter, "_interrupt_and_join_worker", lambda *_args, **_kwargs: False)
    original_writer = adapter._write_completed_turn_receipt
    failed_once = True

    def fail_first_receipt(*args, **kwargs):
        nonlocal failed_once
        if failed_once:
            failed_once = False
            raise OSError("simulated receipt fsync failure")
        return original_writer(*args, **kwargs)

    monkeypatch.setattr(adapter, "_write_completed_turn_receipt", fail_first_receipt)
    request = HarnessRequest(
        messages=({"role": "user", "content": "journal recovery"},),
        model="codex-model",
        role="investigation",
        stage="investigation_and_experiment",
        workspace=tmp_path / "workspace",
        timeout_seconds=None,
    )
    receipt_root = tmp_path / "receipts"
    digest = "a" * 64

    with pytest.raises(adapter.CodexCompletionRecoveryPending) as failure:
        adapter.run_codex_sdk_turn(
            config=CodexConfig(),
            request=request,
            thread_config={},
            session_id=None,
            trace_path=tmp_path / "first.jsonl",
            completed_turn_receipt_root=receipt_root,
            request_digest=digest,
            turn_count_before=0,
        )

    assert failure.value.kind == "recovery_pending"
    journal = receipt_root.parent / ".arche_codex_completion_pending.json"
    assert journal.is_file()
    assert not list(receipt_root.glob("*.json"))
    assert _ReadableCodex.thread.turn_calls == 1

    replay = adapter.run_codex_sdk_turn(
        config=CodexConfig(),
        request=request,
        thread_config={},
        session_id="thread-replay",
        trace_path=tmp_path / "second.jsonl",
        completed_turn_receipt_root=receipt_root,
        request_digest=digest,
        turn_count_before=0,
    )

    assert replay.replayed is True
    assert replay.turn_id == "turn-1"
    assert _ReadableCodex.thread.turn_calls == 1
    assert (receipt_root / "turn-1.json").is_file()
    assert not journal.exists()


def test_deferred_receipt_handle_requires_matching_handoff_and_is_idempotent(
    tmp_path: Path,
) -> None:
    """延迟句柄按固定 handoff 确认，重复确认安全且跨进程缺失仍拒绝。"""

    owner = CodexSessionOwner.investigation("task-1", "batch-1", "combination-1")
    store = CodexSessionStore.from_root(
        tmp_path / "sessions",
        owner=owner,
        scope_id="scope-1",
        model="investigation-model",
    )
    receipt_root = store.manifest_path.parent / "completed_turn_receipts"
    request_digest = "a" * 64
    policy_digest = "b" * 64
    adapter._write_completed_turn_receipt(
        receipt_root,
        thread_id="thread-deferred",
        turn_id="turn-deferred",
        request_digest=request_digest,
        turn_count_before=3,
        thread_policy_digest=policy_digest,
    )
    handle = adapter.build_codex_completed_turn_receipt_handle(
        receipt_root,
        owner=owner,
        session_store=store,
        thread_id="thread-deferred",
        turn_id="turn-deferred",
        request_digest=request_digest,
        turn_count_before=3,
        thread_policy_digest=policy_digest,
        handoff_kind="result",
    )

    assert set(handle.as_mapping()) == {
        "owner",
        "thread_id",
        "turn_id",
        "turn_request_digest",
        "thread_policy_digest",
        "turn_count_before",
        "handoff_kind",
    }
    with pytest.raises(adapter.HarnessError, match="handoff kind"):
        handle.acknowledge_after_waiting_persisted()
    assert (receipt_root / "turn-deferred.json").is_file()

    handle.acknowledge_after_result_persisted()
    handle.acknowledge_after_result_persisted()
    assert handle.acknowledged is True
    assert not (receipt_root / "turn-deferred.json").exists()

    with pytest.raises(adapter.HarnessError, match="missing"):
        adapter.build_codex_completed_turn_receipt_handle(
            receipt_root,
            owner=owner,
            session_store=store,
            thread_id="thread-deferred",
            turn_id="turn-deferred",
            request_digest=request_digest,
            turn_count_before=3,
            thread_policy_digest=policy_digest,
            handoff_kind="result",
        )


def test_completed_receipt_handle_cannot_be_constructed_from_an_arbitrary_directory(
    tmp_path: Path,
) -> None:
    """直接构造句柄不能绕过 canonical SessionStore 和 owner 绑定。"""

    owner = CodexSessionOwner.investigation("task-1", "batch-1", "combination-1")
    receipt_root = tmp_path / "arbitrary-receipts"
    receipt_root.mkdir()

    with pytest.raises(adapter.HarnessError, match="SessionStore factory"):
        adapter.CodexCompletedTurnReceiptHandle(
            receipt_root=receipt_root,
            owner=owner,
            thread_id="thread-1",
            turn_id="turn-1",
            request_digest="a" * 64,
            thread_policy_digest="b" * 64,
            turn_count_before=0,
            handoff_kind="result",
        )


def test_deferred_receipt_handle_rejects_owner_drift_without_creating_directory(
    tmp_path: Path,
) -> None:
    """句柄构造严格校验 owner，缺失 receipt 目录时保持文件系统不变。"""

    owner = CodexSessionOwner.investigation("task-1", "batch-1", "combination-1")
    store = CodexSessionStore.from_root(
        tmp_path / "sessions",
        owner=owner,
        scope_id="scope-1",
        model="investigation-model",
    )
    receipt_root = store.manifest_path.parent / "completed_turn_receipts"
    with pytest.raises(adapter.HarnessError, match="directory"):
        adapter.build_codex_completed_turn_receipt_handle(
            receipt_root,
            owner=owner,
            session_store=store,
            thread_id="thread-deferred",
            turn_id="turn-deferred",
            request_digest="a" * 64,
            turn_count_before=0,
            thread_policy_digest="b" * 64,
            handoff_kind="waiting",
        )
    assert not receipt_root.exists()

    existing_root = receipt_root
    adapter._write_completed_turn_receipt(
        existing_root,
        thread_id="thread-deferred",
        turn_id="turn-deferred",
        request_digest="a" * 64,
        turn_count_before=0,
        thread_policy_digest="b" * 64,
    )
    with pytest.raises(adapter.HarnessError, match="owner fields"):
        adapter.build_codex_completed_turn_receipt_handle(
            existing_root,
            owner={
                "kind": "investigation",
                "task_id": "task-1",
                "batch_id": "batch-1",
                "combination_id": "combination-1",
                "session_id": "provider-thread",
            },
            session_store=store,
            thread_id="thread-deferred",
            turn_id="turn-deferred",
            request_digest="a" * 64,
            turn_count_before=0,
            thread_policy_digest="b" * 64,
            handoff_kind="waiting",
        )
    assert (existing_root / "turn-deferred.json").is_file()

    with pytest.raises(adapter.HarnessError, match="not canonical"):
        adapter.build_codex_completed_turn_receipt_handle(
            tmp_path / "other-root",
            owner=owner,
            session_store=store,
            thread_id="thread-deferred",
            turn_id="turn-deferred",
            request_digest="a" * 64,
            turn_count_before=0,
            thread_policy_digest="b" * 64,
            handoff_kind="waiting",
        )
    assert (existing_root / "turn-deferred.json").is_file()


def test_deferred_receipt_handle_binds_owner_to_session_store_root(tmp_path: Path) -> None:
    """SessionStore 归属或规范 receipt 根任一漂移都不能构造句柄。"""

    owner = CodexSessionOwner.investigation("task-1", "batch-1", "combination-1")
    store = CodexSessionStore.from_root(
        tmp_path / "sessions",
        owner=owner,
        scope_id="scope-1",
        model="investigation-model",
    )
    receipt_root = store.manifest_path.parent / "completed_turn_receipts"
    adapter._write_completed_turn_receipt(
        receipt_root,
        thread_id="thread-bound",
        turn_id="turn-bound",
        request_digest="a" * 64,
        turn_count_before=0,
        thread_policy_digest="b" * 64,
    )
    handle = adapter.build_codex_completed_turn_receipt_handle(
        receipt_root,
        owner=owner,
        session_store=store,
        thread_id="thread-bound",
        turn_id="turn-bound",
        request_digest="a" * 64,
        turn_count_before=0,
        thread_policy_digest="b" * 64,
        handoff_kind="result",
    )
    assert handle.owner == owner.identity

    with pytest.raises(adapter.HarnessError, match="session store"):
        adapter.build_codex_completed_turn_receipt_handle(
            receipt_root,
            owner=CodexSessionOwner.investigation(
                "task-other",
                "batch-1",
                "combination-1",
            ),
            session_store=store,
            thread_id="thread-bound",
            turn_id="turn-bound",
            request_digest="a" * 64,
            turn_count_before=0,
            thread_policy_digest="b" * 64,
            handoff_kind="result",
        )


def _prepare_completed_receipt_recovery_session(
    tmp_path: Path,
    *,
    thread_id: str = "thread-recovery",
    turn_id: str = "turn-recovery",
    request_digest: str = "a" * 64,
    policy_digest: str = "b" * 64,
    turn_count_before: int = 0,
) -> tuple[CodexSessionStore, Path, CodexSessionOwner, dict[str, str]]:
    """为跨进程 receipt 恢复测试建立已完成 manifest 和可选 receipt。"""

    owner = CodexSessionOwner.investigation("task-1", "batch-1", "combination-1")
    store = CodexSessionStore.from_root(
        tmp_path / "sessions",
        owner=owner,
        scope_id="scope-1",
        model="investigation-model",
        thread_policy_digest=policy_digest,
    )
    store.create(
        thread_id,
        stage="initial_investigation",
        status="created",
        thread_policy_digest=policy_digest,
    )
    for _ in range(turn_count_before + 1):
        store.record_completed_turn(
            stage="initial_investigation",
            status="completed",
        )
    receipt_root = store.manifest_path.parent / "completed_turn_receipts"
    identity = {
        "thread_id": thread_id,
        "turn_id": turn_id,
        "request_digest": request_digest,
        "thread_policy_digest": policy_digest,
        "turn_count_before": turn_count_before,
    }
    return store, receipt_root, owner, identity


def test_completed_receipt_resolver_returns_noop_after_cross_process_ack(
    tmp_path: Path,
) -> None:
    """receipt 已被前一进程删除时，精确 manifest 状态可返回跨进程 no-op。"""

    store, receipt_root, owner, identity = _prepare_completed_receipt_recovery_session(
        tmp_path,
    )
    adapter._write_completed_turn_receipt(receipt_root, **identity)
    handle = adapter.build_codex_completed_turn_receipt_handle(
        receipt_root,
        owner=owner,
        session_store=store,
        handoff_kind="result",
        **identity,
    )
    handle.acknowledge_after_result_persisted()
    assert not (receipt_root / f"{identity['turn_id']}.json").exists()

    recovered = adapter.resolve_codex_completed_turn_receipt_handle(
        receipt_root,
        owner=owner,
        session_store=store,
        handoff_kind="result",
        **identity,
    )

    assert recovered.already_acknowledged is True
    assert recovered.acknowledged is True
    recovered.acknowledge_after_result_persisted()
    assert not (receipt_root / f"{identity['turn_id']}.json").exists()


def test_completed_receipt_resolver_returns_strict_handle_when_receipt_exists(
    tmp_path: Path,
) -> None:
    """receipt 仍在时 resolver 必须返回可严格核对并删除的普通句柄。"""

    store, receipt_root, owner, identity = _prepare_completed_receipt_recovery_session(
        tmp_path,
    )
    adapter._write_completed_turn_receipt(receipt_root, **identity)

    recovered = adapter.resolve_codex_completed_turn_receipt_handle(
        receipt_root,
        owner=owner,
        session_store=store,
        handoff_kind="result",
        **identity,
    )

    assert recovered.already_acknowledged is False
    assert recovered.acknowledged is False
    recovered.acknowledge_after_result_persisted()
    assert not (receipt_root / f"{identity['turn_id']}.json").exists()


def test_completed_receipt_resolver_derives_checkpoint_from_exact_turn_receipt(
    tmp_path: Path,
) -> None:
    """submit 时尚无 checkpoint 的 sidecar 可从同 turn 收据恢复完整句柄。"""

    store, receipt_root, owner, identity = _prepare_completed_receipt_recovery_session(
        tmp_path,
    )
    adapter._write_completed_turn_receipt(receipt_root, **identity)

    recovered = adapter.resolve_codex_completed_turn_receipt_handle(
        receipt_root,
        owner=owner,
        session_store=store,
        thread_id=identity["thread_id"],
        turn_id=identity["turn_id"],
        request_digest=None,
        turn_count_before=None,
        thread_policy_digest=None,
        handoff_kind="result",
    )

    assert recovered is not None
    assert recovered.as_mapping() == {
        "owner": owner.identity,
        "thread_id": identity["thread_id"],
        "turn_id": identity["turn_id"],
        "turn_request_digest": identity["request_digest"],
        "thread_policy_digest": identity["thread_policy_digest"],
        "turn_count_before": identity["turn_count_before"],
        "handoff_kind": "result",
    }


def test_completed_receipt_resolver_accepts_exact_result_after_receipt_ack(
    tmp_path: Path,
) -> None:
    """收据已确认但 sidecar 尚未清理时，无 checkpoint 恢复返回已确认标记。"""

    store, receipt_root, owner, identity = _prepare_completed_receipt_recovery_session(
        tmp_path,
    )
    adapter._write_completed_turn_receipt(receipt_root, **identity)
    handle = adapter.build_codex_completed_turn_receipt_handle(
        receipt_root,
        owner=owner,
        session_store=store,
        handoff_kind="result",
        **identity,
    )
    handle.acknowledge_after_result_persisted()

    recovered = adapter.resolve_codex_completed_turn_receipt_handle(
        receipt_root,
        owner=owner,
        session_store=store,
        thread_id=identity["thread_id"],
        turn_id=identity["turn_id"],
        request_digest=None,
        turn_count_before=None,
        thread_policy_digest=None,
        handoff_kind="result",
    )

    assert recovered is None


def test_completed_receipt_resolver_rejects_partial_optional_checkpoint(
    tmp_path: Path,
) -> None:
    """恢复方只能提供完整 checkpoint，或完全交由精确 turn 收据解析。"""

    store, receipt_root, owner, identity = _prepare_completed_receipt_recovery_session(
        tmp_path,
    )
    adapter._write_completed_turn_receipt(receipt_root, **identity)

    with pytest.raises(adapter.HarnessError, match="checkpoint"):
        adapter.resolve_codex_completed_turn_receipt_handle(
            receipt_root,
            owner=owner,
            session_store=store,
            thread_id=identity["thread_id"],
            turn_id=identity["turn_id"],
            request_digest=identity["request_digest"],
            turn_count_before=None,
            thread_policy_digest=None,
            handoff_kind="result",
        )


@pytest.mark.parametrize("journal", ("completion_pending", "request_intent", "terminal"))
def test_completed_receipt_resolver_blocks_unfinished_journals(
    tmp_path: Path,
    journal: str,
) -> None:
    """任一未清理完成/请求/终态 journal 都不能伪装成已确认 receipt。"""

    store, receipt_root, owner, identity = _prepare_completed_receipt_recovery_session(
        tmp_path,
    )
    if journal == "completion_pending":
        adapter._write_completion_pending(receipt_root, **identity)
    elif journal == "request_intent":
        adapter._write_request_intent(
            receipt_root,
            thread_id=identity["thread_id"],
            request_digest=identity["request_digest"],
            turn_count_before=identity["turn_count_before"],
            physical_turn_count_before=identity["turn_count_before"],
            provider_id="arche_codex",
            env_key="ARCHE_CODEX_API_KEY",
            model="investigation-model",
            role="investigation",
            stage="initial_investigation",
            profile="investigation",
            workspace=store.workspace,
            thread_policy_digest=identity["thread_policy_digest"],
        )
    else:
        adapter._write_terminal_turn_journal(
            receipt_root,
            thread_id=identity["thread_id"],
            turn_id=identity["turn_id"],
            status="interrupted",
            request_digest=identity["request_digest"],
            turn_count_before=identity["turn_count_before"],
            thread_policy_digest=identity["thread_policy_digest"],
        )

    with pytest.raises(adapter.HarnessError, match="journal"):
        adapter.resolve_codex_completed_turn_receipt_handle(
            receipt_root,
            owner=owner,
            session_store=store,
            handoff_kind="result",
            **identity,
        )


@pytest.mark.parametrize(
    "journal_name",
    (
        "recovery_path",
        "recovery_poison_path",
        "validated_turn_path",
        "thread_rollback_intent_path",
        "thread_rollback_receipt_path",
    ),
)
def test_completed_receipt_resolver_blocks_session_store_recovery_journals(
    tmp_path: Path,
    journal_name: str,
) -> None:
    """SessionStore 仍有恢复凭证时不能把 receipt 缺失视为已确认。"""

    store, receipt_root, owner, identity = _prepare_completed_receipt_recovery_session(
        tmp_path,
    )
    receipt_root.mkdir()
    journal_path = getattr(store, journal_name)
    journal_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(adapter.HarnessError, match="recovery journal"):
        adapter.resolve_codex_completed_turn_receipt_handle(
            receipt_root,
            owner=owner,
            session_store=store,
            handoff_kind="result",
            **identity,
        )


def test_completed_receipt_resolver_does_not_allow_missing_receipt_for_waiting(
    tmp_path: Path,
) -> None:
    """waiting handoff 即使 manifest 已完成也不能复用 Result no-op。"""

    store, receipt_root, owner, identity = _prepare_completed_receipt_recovery_session(
        tmp_path,
    )
    receipt_root.mkdir()

    with pytest.raises(adapter.HarnessError, match="non-result"):
        adapter.resolve_codex_completed_turn_receipt_handle(
            receipt_root,
            owner=owner,
            session_store=store,
            handoff_kind="waiting",
            **identity,
        )


@pytest.mark.parametrize(
    "field, value",
    (
        ("thread_id", "thread-other"),
        ("thread_policy_digest", "d" * 64),
        ("turn_count_before", 1),
    ),
)
def test_completed_receipt_resolver_rejects_manifest_or_identity_drift(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    """缺失 receipt 的 no-op 不能绕过 manifest 或 provider 身份漂移。"""

    store, receipt_root, owner, identity = _prepare_completed_receipt_recovery_session(
        tmp_path,
    )
    mismatched = dict(identity)
    mismatched[field] = value

    with pytest.raises(adapter.HarnessError, match="(manifest|receipt|thread)"):
        adapter.resolve_codex_completed_turn_receipt_handle(
            receipt_root,
            owner=owner,
            session_store=store,
            handoff_kind="result",
            **mismatched,
        )


def test_completed_receipt_resolver_rejects_missing_root_without_creating_it(
    tmp_path: Path,
) -> None:
    """receipt 目录缺失时 fail-closed，resolver 不创建该目录。"""

    store, receipt_root, owner, identity = _prepare_completed_receipt_recovery_session(
        tmp_path,
        thread_id="thread-manifest",
    )
    with pytest.raises(adapter.HarnessError, match="receipt directory"):
        adapter.resolve_codex_completed_turn_receipt_handle(
            receipt_root,
            owner=owner,
            session_store=store,
            handoff_kind="result",
            **identity,
        )
    assert not receipt_root.exists()


def test_sdk_adapter_history_recovery_handles_crash_before_completion_journal(
    monkeypatch, tmp_path: Path
) -> None:
    """journal 尚未写入时也严格查已有 thread history，避免再次提交模型。"""

    _ReadableCodex.thread = _ReadableThread()
    monkeypatch.setattr(adapter, "Codex", _ReadableCodex)
    failed_once = True

    def fail_before_journal(*_args, **_kwargs):
        nonlocal failed_once
        if failed_once:
            failed_once = False
            raise OSError("simulated crash before completion journal")
        return None

    monkeypatch.setattr(adapter, "_write_completion_pending", fail_before_journal)
    request = HarnessRequest(
        messages=({"role": "user", "content": "history recovery"},),
        model="codex-model",
        role="investigation",
        stage="investigation_and_experiment",
        workspace=tmp_path / "workspace",
        timeout_seconds=None,
    )
    receipt_root = tmp_path / "receipts"
    digest = "b" * 64

    with pytest.raises(adapter.CodexCompletionRecoveryPending):
        adapter.run_codex_sdk_turn(
            config=CodexConfig(),
            request=request,
            thread_config={},
            session_id=None,
            trace_path=tmp_path / "first.jsonl",
            completed_turn_receipt_root=receipt_root,
            request_digest=digest,
            turn_count_before=0,
        )

    assert not (receipt_root.parent / ".arche_codex_completion_pending.json").exists()
    replay = adapter.run_codex_sdk_turn(
        config=CodexConfig(),
        request=request,
        thread_config={},
        session_id="thread-replay",
        trace_path=tmp_path / "second.jsonl",
        completed_turn_receipt_root=receipt_root,
        request_digest=digest,
        turn_count_before=0,
    )

    assert replay.replayed is True
    assert replay.turn_id == "turn-1"
    assert _ReadableCodex.thread.turn_calls == 1
    assert (receipt_root / "turn-1.json").is_file()


def test_sdk_adapter_pre_turn_intent_blocks_retry_when_history_has_no_new_turn(
    monkeypatch, tmp_path: Path
) -> None:
    """worker 启动后的崩溃若 history 没有新 turn，恢复不得再次提交模型。"""

    _IntentCodex.thread = _IntentThread()
    monkeypatch.setattr(adapter, "Codex", _IntentCodex)
    request = HarnessRequest(
        messages=({"role": "user", "content": "intent crash"},),
        model="codex-model",
        role="investigation",
        stage="investigation_and_experiment",
        workspace=tmp_path / "workspace",
        timeout_seconds=None,
    )
    receipt_root = tmp_path / "receipts"
    digest = "c" * 64
    intent_path = receipt_root.parent / ".arche_codex_request_intent.json"

    with pytest.raises(adapter.HarnessError, match="Codex SDK turn failed"):
        adapter.run_codex_sdk_turn(
            config=CodexConfig(),
            request=request,
            thread_config={"model_provider": "arche_codex"},
            session_id=None,
            trace_path=tmp_path / "first.jsonl",
            provider_id="arche_codex",
            env_key="ARCHE_CODEX_API_KEY",
            completed_turn_receipt_root=receipt_root,
            request_digest=digest,
            turn_count_before=0,
        )

    assert intent_path.is_file()
    assert _IntentCodex.thread.turn_calls == 1

    with pytest.raises(adapter.HarnessError, match="request intent"):
        adapter.run_codex_sdk_turn(
            config=CodexConfig(),
            request=request,
            thread_config={"model_provider": "arche_codex"},
            session_id="thread-intent",
            trace_path=tmp_path / "second.jsonl",
            provider_id="arche_codex",
            env_key="ARCHE_CODEX_API_KEY",
            completed_turn_receipt_root=receipt_root,
            request_digest=digest,
            turn_count_before=0,
        )

    assert _IntentCodex.thread.turn_calls == 1
    assert intent_path.is_file()

    adapter.confirm_codex_request_intent_without_new_turn(
        receipt_root,
        thread=_IntentCodex.thread,
        thread_id="thread-intent",
        request_digest=digest,
        turn_count_before=0,
        provider_id="arche_codex",
        env_key="ARCHE_CODEX_API_KEY",
        model="codex-model",
        role="investigation",
        stage="investigation_and_experiment",
        profile="investigation",
        workspace=request.workspace,
        thread_policy_digest=json.loads(intent_path.read_text(encoding="utf-8"))["thread_policy_digest"],
    )
    assert not intent_path.exists()


@pytest.mark.parametrize("persist_recovery_record", (True, False))
def test_sdk_adapter_recovery_only_then_wake_reuses_thread_without_second_initial_turn(
    monkeypatch,
    tmp_path: Path,
    persist_recovery_record: bool,
) -> None:
    """无论恢复记录是否写入，旧 interrupted turn 都先收束再启动 wake。"""

    class _TwoPhaseThread:
        id = "thread-wake-continuation"

        def __init__(self) -> None:
            self.turn_calls = 0
            self.history = [{"id": "turn-interrupted", "status": "interrupted"}]
            self.resume_calls: list[str] = []

        def turn(self, *_args, **_kwargs):
            self.turn_calls += 1
            assert self.turn_calls == 1
            return _FakeTurn()

        def read(self, *, include_turns: bool = False):
            assert include_turns is True
            return {"thread": {"id": self.id, "turns": list(self.history)}}

    class _TwoPhaseCodex:
        thread = _TwoPhaseThread()

        def __init__(self, _config):
            return None

        def thread_start(self, **_kwargs):
            pytest.fail("recovery/wake must never start a new thread")

        def thread_resume(self, thread_id, **_kwargs):
            self.thread.resume_calls.append(thread_id)
            return self.thread

        def close(self):
            return None

    monkeypatch.setattr(adapter, "Codex", _TwoPhaseCodex)
    initial_request = HarnessRequest(
        messages=({"role": "user", "content": "initial investigation"},),
        model="codex-model",
        role="investigation",
        stage="investigation_and_experiment",
        workspace=tmp_path / "workspace",
        metadata={"request_kind": "initial_investigation"},
    )
    wake_request = HarnessRequest(
        messages=(
            {"role": "system", "content": "resume the active job"},
            {"role": "user", "content": '{"job_id":"job-active"}'},
        ),
        model="codex-model",
        role="investigation",
        stage="investigation_and_experiment",
        workspace=tmp_path / "workspace",
        metadata={
            "request_kind": "investigation_job_wake",
            "job_id": "job-active",
            "operation_id": "operation-active",
            "submission_key": "submission-active",
        },
    )
    receipt_root = tmp_path / "receipts"
    initial_digest = adapter.calculate_codex_request_digest(initial_request)
    wake_digest = adapter.calculate_codex_request_digest(wake_request)
    effective_thread_config = adapter.build_codex_capability_config(
        thread_config={},
        workspace=initial_request.workspace,
        profile="investigation",
        env_key="ARCHE_CODEX_API_KEY",
    )
    policy_digest = adapter.calculate_codex_thread_policy_digest(
        thread_config=effective_thread_config,
        workspace=initial_request.workspace,
        provider_id="arche_codex",
        env_key="ARCHE_CODEX_API_KEY",
        model=initial_request.model,
        role=initial_request.role,
        stage=initial_request.stage,
        profile="investigation",
    )

    # 旧 Harness 已经在 request-intent 后退出；本次夹具直接保留其唯一
    # interrupted history，避免把恢复测试与初次提交混在一起。
    adapter._write_request_intent(
        receipt_root,
        thread_id=_TwoPhaseCodex.thread.id,
        request_digest=initial_digest,
        turn_count_before=0,
        physical_turn_count_before=0,
        provider_id="arche_codex",
        env_key="ARCHE_CODEX_API_KEY",
        model=initial_request.model,
        role=initial_request.role,
        stage=initial_request.stage,
        profile="investigation",
        workspace=initial_request.workspace,
        thread_policy_digest=policy_digest,
    )

    terminal_ack: list[tuple[str, str, str]] = []
    observed_wake_recovery = {
        "state": "recovery_pending",
        "reason": "request_intent",
        "action": "request_intent_confirmation_required",
        "stage": "investigation_job_wake",
        "turn_id": "",
        "turn_count": 0,
        "request_digest": initial_digest,
        "thread_policy_digest": policy_digest,
    }
    with pytest.raises(adapter.CodexTerminalTurnError):
        adapter.run_codex_sdk_turn(
            config=CodexConfig(),
            request=initial_request,
            thread_config={},
            session_id=_TwoPhaseCodex.thread.id,
            trace_path=tmp_path / "recovery.jsonl",
            completed_turn_receipt_root=receipt_root,
            request_digest=initial_digest,
            turn_count_before=0,
            recovery_only=True,
            recovery_turn_id="turn-interrupted",
            completion_recovery=(
                observed_wake_recovery if persist_recovery_record else None
            ),
            on_terminal_turn=lambda *identity: terminal_ack.append(identity),
        )

    assert terminal_ack == [
        (_TwoPhaseCodex.thread.id, "turn-interrupted", "interrupted"),
    ]
    assert _TwoPhaseCodex.thread.turn_calls == 0

    result = adapter.run_codex_sdk_turn(
        config=CodexConfig(),
        request=wake_request,
        thread_config={},
        session_id=_TwoPhaseCodex.thread.id,
        trace_path=tmp_path / "wake.jsonl",
        completed_turn_receipt_root=receipt_root,
        request_digest=wake_digest,
        turn_count_before=0,
    )

    assert result.thread_id == _TwoPhaseCodex.thread.id
    assert result.turn_id == "turn-1"
    assert _TwoPhaseCodex.thread.turn_calls == 1
    assert _TwoPhaseCodex.thread.resume_calls == [
        _TwoPhaseCodex.thread.id,
        _TwoPhaseCodex.thread.id,
    ]


def test_request_intent_confirmation_uses_physical_history_baseline(
    tmp_path: Path,
) -> None:
    """更早 failed 可在物理基线内，新增或未终结 turn 均不能被忽略。"""

    receipt_root = tmp_path / "receipts"
    workspace = tmp_path / "workspace"
    digest = "a" * 64
    policy_digest = "b" * 64
    confirmation = {
        "thread_id": "thread-confirm",
        "request_digest": digest,
        "turn_count_before": 0,
        "provider_id": "arche_codex",
        "env_key": "ARCHE_CODEX_API_KEY",
        "model": "codex-model",
        "role": "investigation",
        "stage": "investigation_and_experiment",
        "profile": "investigation",
        "workspace": workspace,
        "thread_policy_digest": policy_digest,
    }

    def history_thread(status: str):
        return SimpleNamespace(
            read=lambda include_turns=True: {
                "thread": {
                    "id": "thread-confirm",
                    "turns": [{"id": f"turn-{status}", "status": status}],
                }
            }
        )

    adapter._write_request_intent(
        receipt_root,
        physical_turn_count_before=1,
        **confirmation,
    )
    adapter.confirm_codex_request_intent_without_new_turn(
        receipt_root,
        thread=history_thread("failed"),
        **confirmation,
    )
    assert adapter._read_request_intent(receipt_root) is None

    adapter._write_request_intent(
        receipt_root,
        physical_turn_count_before=0,
        **confirmation,
    )
    with pytest.raises(adapter.HarnessError, match="physical history"):
        adapter.confirm_codex_request_intent_without_new_turn(
            receipt_root,
            thread=history_thread("failed"),
            **confirmation,
        )
    assert adapter._read_request_intent(receipt_root) is not None
    adapter._remove_request_intent(receipt_root)

    adapter._write_request_intent(
        receipt_root,
        physical_turn_count_before=1,
        **confirmation,
    )
    with pytest.raises(adapter.HarnessError, match="unfinished") as failure:
        adapter.confirm_codex_request_intent_without_new_turn(
            receipt_root,
            thread=history_thread("inProgress"),
            **confirmation,
        )
    assert failure.value.kind == "recovery_pending"
    assert adapter._read_request_intent(receipt_root) is not None


@pytest.mark.parametrize(
    (
        "initial_history",
        "first_session_id",
        "crash_status",
        "expected_physical_baseline",
    ),
    [
        ([], None, "failed", 0),
        (
            [{"id": "turn-prior-failed", "status": "failed"}],
            "thread-history-crash",
            "interrupted",
            1,
        ),
    ],
)
def test_request_intent_crash_window_recovers_unique_terminal_history(
    monkeypatch,
    tmp_path: Path,
    initial_history: list[dict[str, object]],
    first_session_id: str | None,
    crash_status: str,
    expected_physical_baseline: int,
) -> None:
    """notification 前崩溃后，只把本次基线后的唯一失败 turn 交给同步 ack。"""

    thread = _HistoryCrashThread(initial_history, crash_status)
    _HistoryCrashCodex.thread = thread
    monkeypatch.setattr(adapter, "Codex", _HistoryCrashCodex)
    request = HarnessRequest(
        messages=({"role": "user", "content": "recover terminal crash window"},),
        model="codex-model",
        role="investigation",
        stage="investigation_and_experiment",
        workspace=tmp_path / "workspace",
        timeout_seconds=None,
    )
    receipt_root = tmp_path / "receipts"
    digest = "8" * 64

    with pytest.raises(adapter.HarnessError, match="Codex SDK turn failed"):
        adapter.run_codex_sdk_turn(
            config=CodexConfig(),
            request=request,
            thread_config={},
            session_id=first_session_id,
            trace_path=tmp_path / "crashed.jsonl",
            completed_turn_receipt_root=receipt_root,
            request_digest=digest,
            turn_count_before=0,
        )

    intent_path = tmp_path / ".arche_codex_request_intent.json"
    intent = json.loads(intent_path.read_text(encoding="utf-8"))
    assert intent["physical_turn_count_before"] == expected_physical_baseline
    assert not (tmp_path / ".arche_codex_terminal_turn.json").exists()
    acknowledgements: list[tuple[str, str, str]] = []

    with pytest.raises(adapter.CodexTerminalTurnError) as terminal:
        adapter.run_codex_sdk_turn(
            config=CodexConfig(),
            request=request,
            thread_config={},
            session_id=thread.id,
            trace_path=tmp_path / "recovered.jsonl",
            completed_turn_receipt_root=receipt_root,
            request_digest=digest,
            turn_count_before=0,
            on_terminal_turn=lambda *identity: acknowledgements.append(identity),
        )

    assert terminal.value.status == crash_status
    assert terminal.value.turn_id == "turn-crash-1"
    assert acknowledgements == [(thread.id, "turn-crash-1", crash_status)]
    assert thread.turn_calls == 1
    assert not intent_path.exists()
    assert not (tmp_path / ".arche_codex_terminal_turn.json").exists()


@pytest.mark.parametrize(
    ("recovered_history", "message"),
    [
        (
            [
                {"id": "turn-failed", "status": "failed"},
                {"id": "turn-interrupted", "status": "interrupted"},
            ],
            "multiple",
        ),
        ([{"id": "turn-running", "status": "inProgress"}], "unfinished"),
        ([{"id": "turn-aborted", "status": "aborted"}], "latest"),
        ([{"id": "turn-unknown", "status": "unknown"}], "unfinished"),
    ],
)
def test_request_intent_history_ambiguity_never_starts_replacement_turn(
    monkeypatch,
    tmp_path: Path,
    recovered_history: list[dict[str, object]],
    message: str,
) -> None:
    """多项新增或未终结 history 必须保留 intent，并拒绝猜测 replacement turn。"""

    thread = _HistoryCrashThread([], None)
    _HistoryCrashCodex.thread = thread
    monkeypatch.setattr(adapter, "Codex", _HistoryCrashCodex)
    request = HarnessRequest(
        messages=({"role": "user", "content": "ambiguous request intent"},),
        model="codex-model",
        role="investigation",
        stage="investigation_and_experiment",
        workspace=tmp_path / "workspace",
        timeout_seconds=None,
    )
    receipt_root = tmp_path / "receipts"

    with pytest.raises(adapter.HarnessError, match="Codex SDK turn failed"):
        adapter.run_codex_sdk_turn(
            config=CodexConfig(),
            request=request,
            thread_config={},
            session_id=None,
            trace_path=tmp_path / "crashed.jsonl",
            completed_turn_receipt_root=receipt_root,
            request_digest="6" * 64,
            turn_count_before=0,
        )
    thread.history = recovered_history

    with pytest.raises(adapter.HarnessError, match=message):
        adapter.run_codex_sdk_turn(
            config=CodexConfig(),
            request=request,
            thread_config={},
            session_id=thread.id,
            trace_path=tmp_path / "blocked.jsonl",
            completed_turn_receipt_root=receipt_root,
            request_digest="6" * 64,
            turn_count_before=0,
            on_terminal_turn=lambda *_identity: pytest.fail("ambiguous history cannot ack"),
        )

    assert thread.turn_calls == 1
    assert (tmp_path / ".arche_codex_request_intent.json").is_file()
    assert not (tmp_path / ".arche_codex_terminal_turn.json").exists()


def test_request_intent_rejects_duplicate_terminal_turn_identity(
    monkeypatch, tmp_path: Path
) -> None:
    """本次新增终态若复用基线中的 turn id，不能生成或确认终态 journal。"""

    thread = _HistoryCrashThread(
        [{"id": "turn-duplicate", "status": "failed"}],
        None,
    )
    _HistoryCrashCodex.thread = thread
    monkeypatch.setattr(adapter, "Codex", _HistoryCrashCodex)
    request = HarnessRequest(
        messages=({"role": "user", "content": "duplicate terminal identity"},),
        model="codex-model",
        role="investigation",
        stage="investigation_and_experiment",
        workspace=tmp_path / "workspace",
        timeout_seconds=None,
    )
    receipt_root = tmp_path / "receipts"

    with pytest.raises(adapter.HarnessError, match="Codex SDK turn failed"):
        adapter.run_codex_sdk_turn(
            config=CodexConfig(),
            request=request,
            thread_config={},
            session_id=thread.id,
            trace_path=tmp_path / "crashed.jsonl",
            completed_turn_receipt_root=receipt_root,
            request_digest="4" * 64,
            turn_count_before=0,
        )
    thread.history.append({"id": "turn-duplicate", "status": "failed"})

    with pytest.raises(adapter.CodexRequestIntentRecoveryPending, match="identity"):
        adapter.run_codex_sdk_turn(
            config=CodexConfig(),
            request=request,
            thread_config={},
            session_id=thread.id,
            trace_path=tmp_path / "blocked.jsonl",
            completed_turn_receipt_root=receipt_root,
            request_digest="4" * 64,
            turn_count_before=0,
        )

    assert thread.turn_calls == 1
    assert (tmp_path / ".arche_codex_request_intent.json").is_file()
    assert not (tmp_path / ".arche_codex_terminal_turn.json").exists()


def test_legacy_v2_request_intent_preserves_original_digest_and_blocks_confirmation(
    monkeypatch, tmp_path: Path
) -> None:
    """缺少物理基线的 v2 intent 只能等待显式迁移，不能恢复或通用确认。"""

    thread = _HistoryCrashThread([], None)
    _HistoryCrashCodex.thread = thread
    monkeypatch.setattr(adapter, "Codex", _HistoryCrashCodex)
    request = HarnessRequest(
        messages=({"role": "user", "content": "legacy request intent"},),
        model="codex-model",
        role="investigation",
        stage="investigation_and_experiment",
        workspace=tmp_path / "workspace",
        timeout_seconds=None,
    )
    receipt_root = tmp_path / "receipts"
    digest = "5" * 64

    with pytest.raises(adapter.HarnessError, match="Codex SDK turn failed"):
        adapter.run_codex_sdk_turn(
            config=CodexConfig(),
            request=request,
            thread_config={},
            session_id=None,
            trace_path=tmp_path / "crashed.jsonl",
            completed_turn_receipt_root=receipt_root,
            request_digest=digest,
            turn_count_before=0,
        )
    intent_path = tmp_path / ".arche_codex_request_intent.json"
    legacy_intent = json.loads(intent_path.read_text(encoding="utf-8"))
    legacy_intent["schema_version"] = 2
    legacy_intent.pop("physical_turn_count_before")
    intent_path.write_text(json.dumps(legacy_intent), encoding="utf-8")

    with pytest.raises(adapter.CodexRequestIntentRecoveryPending, match="migration") as blocked:
        adapter.run_codex_sdk_turn(
            config=CodexConfig(),
            request=request,
            thread_config={},
            session_id=thread.id,
            trace_path=tmp_path / "blocked.jsonl",
            completed_turn_receipt_root=receipt_root,
            request_digest=digest,
            turn_count_before=0,
        )
    assert blocked.value.request_digest == digest
    assert thread.turn_calls == 1

    with pytest.raises(adapter.CodexRequestIntentRecoveryPending, match="migration") as confirm:
        adapter.confirm_codex_request_intent_without_new_turn(
            receipt_root,
            thread=thread,
            thread_id=thread.id,
            request_digest=digest,
            turn_count_before=0,
            provider_id="arche_codex",
            env_key="ARCHE_CODEX_API_KEY",
            model=request.model,
            role=request.role,
            stage=request.stage,
            profile="investigation",
            workspace=request.workspace,
            thread_policy_digest=legacy_intent["thread_policy_digest"],
        )
    assert confirm.value.request_digest == digest
    assert intent_path.is_file()


@pytest.mark.parametrize("terminal_status", ["failed", "interrupted"])
def test_terminal_turn_requires_durable_journal_and_explicit_ack_before_retry(
    monkeypatch, tmp_path: Path, capsys, terminal_status: str
) -> None:
    """终态只写白名单证据；显式 ack 前不清理状态，也不启动下一 turn。"""

    _TerminalThenSuccessCodex.thread = _TerminalThenSuccessThread(terminal_status)
    monkeypatch.setattr(adapter, "Codex", _TerminalThenSuccessCodex)
    request = HarnessRequest(
        messages=({"role": "user", "content": "terminal retry"},),
        model="codex-model",
        role="investigation",
        stage="investigation_and_experiment",
        workspace=tmp_path / "workspace",
        timeout_seconds=None,
    )
    receipt_root = tmp_path / "receipts"
    intent_path = receipt_root.parent / ".arche_codex_request_intent.json"
    journal_path = receipt_root.parent / ".arche_codex_terminal_turn.json"
    digest = "f" * 64

    with pytest.raises(adapter.CodexRecoveryPending) as pending:
        adapter.run_codex_sdk_turn(
            config=CodexConfig(),
            request=request,
            thread_config={},
            session_id=None,
            trace_path=tmp_path / "failed.jsonl",
            completed_turn_receipt_root=receipt_root,
            request_digest=digest,
            turn_count_before=0,
            trace_sensitive_values=("synthetic-secret",),
        )

    trace = (tmp_path / "failed.jsonl").read_text(encoding="utf-8")
    trace_record = json.loads(trace)
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    assert pending.value.reason == "terminal_turn"
    assert pending.value.turn_id == "turn-terminal-failed"
    assert trace_record == {
        "type": "codex_notification",
        "method": "turn/completed",
        "params": {
            "thread_id": "thread-terminal-retry",
            "turn_id": "turn-terminal-failed",
            "status": terminal_status,
        },
    }
    assert journal == {
        "schema_version": 1,
        "thread_id": "thread-terminal-retry",
        "turn_id": "turn-terminal-failed",
        "status": terminal_status,
        "request_digest": digest,
        "turn_count_before": 0,
        "thread_policy_digest": journal["thread_policy_digest"],
    }
    assert journal_path.stat().st_mode & 0o777 == 0o600
    for forbidden in ("synthetic-secret", "quota rejected", "provider-request-unknown-text"):
        assert forbidden not in str(pending.value)
        assert forbidden not in trace
        assert forbidden not in journal_path.read_text(encoding="utf-8")
        assert forbidden not in intent_path.read_text(encoding="utf-8")
    assert intent_path.exists()
    assert _TerminalThenSuccessCodex.thread.failed_turn.interrupt_calls == 0
    assert _TerminalThenSuccessCodex.thread.turn_calls == 1
    assert "no active turn" not in capsys.readouterr().err

    acknowledgements: list[tuple[str, str, str]] = []
    with pytest.raises(adapter.CodexTerminalTurnError) as acknowledged:
        adapter.run_codex_sdk_turn(
            config=CodexConfig(),
            request=request,
            thread_config={},
            session_id="thread-terminal-retry",
            trace_path=tmp_path / "ack.jsonl",
            completed_turn_receipt_root=receipt_root,
            request_digest=digest,
            turn_count_before=0,
            on_terminal_turn=lambda *identity: acknowledgements.append(identity),
        )
    assert acknowledged.value.status == terminal_status
    assert acknowledgements == [
        ("thread-terminal-retry", "turn-terminal-failed", terminal_status)
    ]
    assert _TerminalThenSuccessCodex.thread.turn_calls == 1
    assert not intent_path.exists()
    assert not journal_path.exists()

    recovered = adapter.run_codex_sdk_turn(
        config=CodexConfig(),
        request=request,
        thread_config={},
        session_id="thread-terminal-retry",
        trace_path=tmp_path / "success.jsonl",
        completed_turn_receipt_root=receipt_root,
        request_digest=digest,
        turn_count_before=0,
        on_terminal_turn=lambda *identity: acknowledgements.append(identity),
    )
    assert recovered.thread_id == "thread-terminal-retry"
    assert recovered.turn_id == "turn-1"
    assert _TerminalThenSuccessCodex.thread.turn_calls == 2
    assert len(acknowledgements) == 1
    assert not intent_path.exists()


def test_terminal_recovery_replays_started_before_terminal_ack_after_callback_failure(
    monkeypatch, tmp_path: Path
) -> None:
    """terminal journal 恢复先补记同一 Started，再关闭 WM terminal attempt。"""

    _TerminalThenSuccessCodex.thread = _TerminalThenSuccessThread("failed")
    monkeypatch.setattr(adapter, "Codex", _TerminalThenSuccessCodex)
    request = HarnessRequest(
        messages=({"role": "user", "content": "terminal recovery"},),
        model="codex-model",
        role="investigation",
        stage="investigation_and_experiment",
        workspace=tmp_path / "workspace",
        timeout_seconds=None,
    )
    receipt_root = tmp_path / "receipts"
    digest = "b" * 64

    def reject_first_terminal_ack(*_identity: object) -> None:
        """模拟首次终态回调失败，保留 terminal journal。"""

        raise RuntimeError("Started callback has not been durably recorded")

    with pytest.raises(adapter.CodexTerminalTurnRecoveryPending):
        adapter.run_codex_sdk_turn(
            config=CodexConfig(),
            request=request,
            thread_config={},
            session_id=None,
            trace_path=tmp_path / "first.jsonl",
            completed_turn_receipt_root=receipt_root,
            request_digest=digest,
            turn_count_before=0,
            on_terminal_turn=reject_first_terminal_ack,
        )

    started_failures: list[str] = []
    terminal_calls: list[tuple[str, str, str]] = []

    def fail_replayed_started(turn_id: str) -> None:
        """模拟下一次恢复的 Started 补记仍未成功。"""

        started_failures.append(turn_id)
        raise RuntimeError("Started replay failed")

    with pytest.raises(adapter.CodexTerminalTurnRecoveryPending):
        adapter.run_codex_sdk_turn(
            config=CodexConfig(),
            request=request,
            thread_config={},
            session_id="thread-terminal-retry",
            trace_path=tmp_path / "failed-replay.jsonl",
            completed_turn_receipt_root=receipt_root,
            request_digest=digest,
            turn_count_before=0,
            on_turn_started=fail_replayed_started,
            on_terminal_turn=lambda *identity: terminal_calls.append(identity),
        )
    assert started_failures == ["turn-terminal-failed"]
    assert terminal_calls == []
    assert (tmp_path / ".arche_codex_terminal_turn.json").is_file()
    assert _TerminalThenSuccessCodex.thread.turn_calls == 1

    lifecycle: list[tuple[str, str]] = []

    with pytest.raises(adapter.CodexTerminalTurnError) as acknowledged:
        adapter.run_codex_sdk_turn(
            config=CodexConfig(),
            request=request,
            thread_config={},
            session_id="thread-terminal-retry",
            trace_path=tmp_path / "recovered.jsonl",
            completed_turn_receipt_root=receipt_root,
            request_digest=digest,
            turn_count_before=0,
            on_turn_started=lambda turn_id: lifecycle.append(("started", turn_id)),
            on_terminal_turn=lambda thread_id, turn_id, status: lifecycle.append(
                ("terminal", f"{thread_id}:{turn_id}:{status}")
            ),
        )

    assert acknowledged.value.turn_id == "turn-terminal-failed"
    assert lifecycle == [
        ("started", "turn-terminal-failed"),
        ("terminal", "thread-terminal-retry:turn-terminal-failed:failed"),
    ]
    assert _TerminalThenSuccessCodex.thread.turn_calls == 1
    assert not (tmp_path / ".arche_codex_request_intent.json").exists()
    assert not (tmp_path / ".arche_codex_terminal_turn.json").exists()


def test_error_notification_trace_uses_finite_whitelist_before_terminal_failure(
    monkeypatch, tmp_path: Path
) -> None:
    """独立 error 通知只保留有限重试标记，不能持久化 provider 自由正文。"""

    thread = _TerminalThenSuccessThread("failed")
    thread.failed_turn = _ErrorThenTerminalFailureTurn("failed")
    _TerminalThenSuccessCodex.thread = thread
    monkeypatch.setattr(adapter, "Codex", _TerminalThenSuccessCodex)
    trace_path = tmp_path / "error-terminal.jsonl"

    with pytest.raises(adapter.CodexTerminalTurnRecoveryPending) as pending:
        adapter.run_codex_sdk_turn(
            config=CodexConfig(),
            request=HarnessRequest(
                messages=({"role": "user", "content": "error then terminal"},),
                model="codex-model",
                role="investigation",
                stage="investigation_and_experiment",
                workspace=tmp_path / "workspace",
                timeout_seconds=None,
            ),
            thread_config={},
            session_id=None,
            trace_path=trace_path,
            completed_turn_receipt_root=tmp_path / "receipts",
            request_digest="7" * 64,
            turn_count_before=0,
        )

    records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert records[0] == {
        "type": "codex_notification",
        "method": "error",
        "params": {"will_retry": True},
    }
    assert records[1] == {
        "type": "codex_notification",
        "method": "error",
        "params": {},
    }
    serialized = json.dumps(records, ensure_ascii=False)
    for forbidden in (
        "provider-free-text-must-not-persist",
        "provider-error-request-id",
        "provider-error-outer-request-id",
        "second-provider-free-text",
    ):
        assert forbidden not in serialized
        assert forbidden not in str(pending.value)
        assert forbidden not in (tmp_path / ".arche_codex_terminal_turn.json").read_text(
            encoding="utf-8"
        )


def test_terminal_turn_error_exposes_only_whitelisted_provider_error_code() -> None:
    """终态异常只暴露固定 provider code，任意正文必须被拒绝。"""

    error = adapter.CodexTerminalTurnError(
        turn_id="turn-safe-code",
        status="failed",
        error_code="invalid_encrypted_content",
    )
    assert error.error_code == "invalid_encrypted_content"
    assert error.provider_error_code == "invalid_encrypted_content"
    with pytest.raises(adapter.HarnessError, match="error code"):
        adapter.CodexTerminalTurnError(
            turn_id="turn-safe-code",
            status="failed",
            error_code="provider-secret-free-text",
        )


def test_terminal_provider_error_code_parser_requires_exact_nested_json_code() -> None:
    """provider message 只按真实 JSON error.code 精确匹配，不做相似子串扫描。"""

    payload = SimpleNamespace(
        turn=SimpleNamespace(
            status="failed",
            error=SimpleNamespace(
                message=json.dumps(
                    {
                        "error": {
                            "message": "invalid_encrypted_content with details",
                            "code": "invalid_encrypted_content",
                        }
                    }
                )
            ),
        )
    )
    assert adapter._read_safe_terminal_provider_error_code(payload) == (
        "invalid_encrypted_content"
    )
    similar = SimpleNamespace(
        turn=SimpleNamespace(
            status="failed",
            error=SimpleNamespace(
                message=json.dumps(
                    {"error": {"code": "prefix_invalid_encrypted_content_suffix"}}
                )
            ),
        )
    )
    assert adapter._read_safe_terminal_provider_error_code(similar) is None


def test_investigation_wake_rollback_requires_all_physical_turns_failed_or_interrupted(
    tmp_path: Path,
) -> None:
    """官方 rollback helper 只接受成功计数为零且全是失败终态的同一 thread。"""

    class RollbackClient:
        def __init__(self, response):
            self.response = response
            self.calls: list[tuple[str, object]] = []

        def request(self, method, params, *, response_model):
            self.calls.append((method, params))
            return self.response

    class RollbackThread:
        id = "thread-wake"

        def __init__(self, turns, response_turns=()):
            self.turns = turns
            self._client = RollbackClient(
                {"thread": {"id": self.id, "turns": list(response_turns)}}
            )

        def read(self, *, include_turns=False):
            assert include_turns is True
            return {"thread": {"id": self.id, "turns": list(self.turns)}}

    thread = RollbackThread(
        [
            {"id": "failed-1", "status": "failed"},
            {"id": "interrupted-2", "status": "interrupted"},
        ]
    )
    result = adapter.rollback_codex_thread_for_investigation_wake(
        thread=thread,
        thread_id=thread.id,
        successful_turn_count=0,
    )
    assert result["thread_id"] == thread.id
    assert result["num_turns"] == 2
    assert thread._client.calls == [
        (
            "thread/rollback",
            {"threadId": thread.id, "numTurns": 2},
        )
    ]

    with pytest.raises(adapter.HarnessError, match="zero successful turns"):
        adapter.rollback_codex_thread_for_investigation_turn(
            thread=thread,
            thread_id=thread.id,
            successful_turn_count=1,
        )

    completed = RollbackThread([{"id": "completed-1", "status": "completed"}])
    with pytest.raises(adapter.HarnessError, match="completed"):
        adapter.rollback_codex_thread_for_investigation_wake(
            thread=completed,
            thread_id=completed.id,
            successful_turn_count=0,
        )


def test_investigation_wake_terminal_code_invokes_rollback_on_same_thread(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """wake 终态的白名单错误先关闭当前 attempt，再在原 thread 上官方回滚。"""

    class RollbackClient:
        def __init__(self, thread_id: str) -> None:
            self.thread_id = thread_id
            self.calls: list[tuple[str, object]] = []

        def request(self, method, params, *, response_model):
            self.calls.append((method, params))
            return {"thread": {"id": self.thread_id, "turns": []}}

    class RollbackTurn:
        id = "turn-wake-invalid-encrypted"

        def stream(self):
            yield _FakeEvent(
                method="turn/completed",
                payload=SimpleNamespace(
                    turn=SimpleNamespace(
                        id=self.id,
                        status="failed",
                        error=SimpleNamespace(
                            message=json.dumps(
                                {
                                    "error": {
                                        "code": "invalid_encrypted_content",
                                        "message": "provider detail must not persist",
                                    }
                                }
                            )
                        ),
                    )
                ),
            )

    class RollbackThread:
        id = "thread-wake-rollback"

        def __init__(self) -> None:
            self._client = RollbackClient(self.id)
            self.turn_calls = 0

        def turn(self, *_args, **_kwargs):
            self.turn_calls += 1
            return RollbackTurn()

        def read(self, *, include_turns=False):
            assert include_turns is True
            return {
                "thread": {
                    "id": self.id,
                    "turns": [
                        {"id": "old-failed-1", "status": "failed"},
                        {"id": "old-interrupted-2", "status": "interrupted"},
                        {"id": "old-failed-3", "status": "failed"},
                    ],
                }
            }

    class RollbackCodex:
        thread = RollbackThread()

        def __init__(self, _config):
            return None

        def thread_start(self, **_kwargs):
            return self.thread

        def thread_resume(self, *_args, **_kwargs):
            return self.thread

        def close(self):
            return None

    monkeypatch.setattr(adapter, "Codex", RollbackCodex)
    request = HarnessRequest(
        messages=({"role": "user", "content": "wake with full context"},),
        model="codex-model",
        role="investigation",
        stage="investigation_job_wake",
        workspace=tmp_path / "workspace",
        timeout_seconds=None,
    )
    callbacks: list[tuple[str, object]] = []
    with pytest.raises(adapter.CodexTerminalTurnError) as terminal:
        adapter.run_codex_sdk_turn(
            config=CodexConfig(),
            request=request,
            thread_config={},
            session_id=None,
            trace_path=tmp_path / "wake.jsonl",
            completed_turn_receipt_root=tmp_path / "receipts",
            request_digest="9" * 64,
            turn_count_before=0,
            on_turn_started=lambda turn_id: callbacks.append(("started", turn_id)),
            on_terminal_turn=lambda thread_id, turn_id, status: callbacks.append(
                ("terminal", f"{thread_id}:{turn_id}:{status}")
            ),
            on_wake_rollback=lambda thread_id, thread, error_code: (
                callbacks.append(("rollback", thread_id)),
                adapter.rollback_codex_thread_for_investigation_wake(
                    thread=thread,
                    thread_id=thread_id,
                    successful_turn_count=0,
                ),
            ),
        )
    assert terminal.value.error_code == "invalid_encrypted_content"
    assert callbacks.count(("started", "turn-wake-invalid-encrypted")) == 2
    assert (
        "terminal",
        "thread-wake-rollback:turn-wake-invalid-encrypted:failed",
    ) in callbacks
    assert callbacks[-1] == ("rollback", "thread-wake-rollback")
    assert RollbackCodex.thread._client.calls == [
        (
            "thread/rollback",
            {"threadId": "thread-wake-rollback", "numTurns": 3},
        )
    ]

    # 新的 turn 语义对 initial_investigation 使用同一回调；旧 wake
    # 参数的兼容路径在上面的调用中已经覆盖。
    initial_callbacks: list[tuple[str, str]] = []
    initial_request = HarnessRequest(
        messages=({"role": "user", "content": "initial recovery"},),
        model="codex-model",
        role="investigation",
        stage="initial_investigation",
        workspace=tmp_path / "initial-workspace",
        timeout_seconds=None,
    )
    with pytest.raises(adapter.CodexTerminalTurnError):
        adapter.run_codex_sdk_turn(
            config=CodexConfig(),
            request=initial_request,
            thread_config={},
            session_id=RollbackCodex.thread.id,
            trace_path=tmp_path / "initial.jsonl",
            completed_turn_receipt_root=tmp_path / "initial-receipts",
            request_digest="a" * 64,
            turn_count_before=0,
            on_turn_started=lambda turn_id: initial_callbacks.append(("started", turn_id)),
            on_terminal_turn=lambda _thread_id, turn_id, _status: initial_callbacks.append(
                ("terminal", turn_id)
            ),
            on_investigation_rollback=lambda thread_id, thread, error_code: (
                initial_callbacks.append(("rollback", thread_id)),
                adapter.rollback_codex_thread_for_investigation_turn(
                    thread=thread,
                    thread_id=thread_id,
                    successful_turn_count=0,
                ),
            ),
        )
    assert ("rollback", RollbackCodex.thread.id) in initial_callbacks


def test_terminal_journal_write_failure_keeps_intent_and_skips_ack(
    monkeypatch, tmp_path: Path
) -> None:
    """终态 journal 无法 fsync 时必须保留 intent，且不得提前调用 ack。"""

    _TerminalThenSuccessCodex.thread = _TerminalThenSuccessThread("failed")
    monkeypatch.setattr(adapter, "Codex", _TerminalThenSuccessCodex)
    original_atomic_write = adapter._atomic_write_json_file

    def reject_terminal_journal(path, value, *, description, mode=0o600):
        if path.name == ".arche_codex_terminal_turn.json":
            raise adapter.HarnessError(
                "terminal journal fsync failed",
                kind="session_state",
            )
        return original_atomic_write(
            path,
            value,
            description=description,
            mode=mode,
        )

    monkeypatch.setattr(adapter, "_atomic_write_json_file", reject_terminal_journal)
    receipt_root = tmp_path / "receipts"
    acknowledgements: list[tuple[str, str, str]] = []

    with pytest.raises(adapter.CodexRecoveryPending) as failure:
        adapter.run_codex_sdk_turn(
            config=CodexConfig(),
            request=HarnessRequest(
                messages=({"role": "user", "content": "terminal journal failure"},),
                model="codex-model",
                role="investigation",
                stage="investigation_and_experiment",
                workspace=tmp_path / "workspace",
                timeout_seconds=None,
            ),
            thread_config={},
            session_id=None,
            trace_path=tmp_path / "failed-journal.jsonl",
            completed_turn_receipt_root=receipt_root,
            request_digest="e" * 64,
            turn_count_before=0,
            on_terminal_turn=lambda *identity: acknowledgements.append(identity),
        )

    assert failure.value.reason == "terminal_turn"
    assert acknowledgements == []
    intent_path = tmp_path / ".arche_codex_request_intent.json"
    assert intent_path.is_file()
    assert not (tmp_path / ".arche_codex_terminal_turn.json").exists()

    intent = json.loads(intent_path.read_text(encoding="utf-8"))
    monkeypatch.setattr(adapter, "_atomic_write_json_file", original_atomic_write)
    with pytest.raises(adapter.CodexTerminalTurnError) as recovered:
        adapter.run_codex_sdk_turn(
            config=CodexConfig(),
            request=HarnessRequest(
                messages=({"role": "user", "content": "terminal journal failure"},),
                model="codex-model",
                role="investigation",
                stage="investigation_and_experiment",
                workspace=tmp_path / "workspace",
                timeout_seconds=None,
            ),
            thread_config={},
            session_id="thread-terminal-retry",
            trace_path=tmp_path / "recovered-journal.jsonl",
            completed_turn_receipt_root=receipt_root,
            request_digest="e" * 64,
            turn_count_before=0,
            completion_recovery={
                "reason": "terminal_turn",
                "turn_id": "turn-terminal-failed",
                "request_digest": "e" * 64,
                "turn_count": 0,
                "thread_policy_digest": intent["thread_policy_digest"],
            },
            on_terminal_turn=lambda *identity: acknowledgements.append(identity),
        )
    assert recovered.value.status == "failed"
    assert acknowledgements == [
        ("thread-terminal-retry", "turn-terminal-failed", "failed")
    ]
    assert _TerminalThenSuccessCodex.thread.turn_calls == 1
    assert not intent_path.exists()
    assert not (tmp_path / ".arche_codex_terminal_turn.json").exists()


def test_terminal_cleanup_failure_and_worker_timeout_keep_terminal_recovery(
    monkeypatch, tmp_path: Path
) -> None:
    """intent 删除失败与 worker 超时并发时，终态恢复义务不能降级。"""

    released = threading.Event()

    class BlockingTerminalTurn(_TerminalFailureTurn):
        def stream(self):
            yield from super().stream()
            released.wait()

    class BlockingTerminalThread(_TerminalThenSuccessThread):
        def __init__(self) -> None:
            super().__init__("failed")
            self.failed_turn = BlockingTerminalTurn("failed")

    class BlockingTerminalCodex(_TerminalThenSuccessCodex):
        thread = BlockingTerminalThread()

    monkeypatch.setattr(adapter, "Codex", BlockingTerminalCodex)
    original_remove_intent = adapter._remove_request_intent
    remove_calls = 0

    def fail_first_intent_cleanup(root):
        nonlocal remove_calls
        remove_calls += 1
        if remove_calls == 1:
            raise OSError("simulated intent cleanup failure")
        original_remove_intent(root)

    monkeypatch.setattr(adapter, "_remove_request_intent", fail_first_intent_cleanup)
    receipt_root = tmp_path / "receipts"
    request = HarnessRequest(
        messages=({"role": "user", "content": "terminal cleanup race"},),
        model="codex-model",
        role="investigation",
        stage="investigation_and_experiment",
        workspace=tmp_path / "workspace",
        timeout_seconds=None,
    )
    acknowledgements: list[tuple[str, str, str]] = []

    try:
        with pytest.raises(adapter.CodexTerminalTurnRecoveryPending) as pending:
            adapter.run_codex_sdk_turn(
                config=CodexConfig(),
                request=request,
                thread_config={},
                session_id=None,
                trace_path=tmp_path / "first.jsonl",
                completed_turn_receipt_root=receipt_root,
                request_digest="9" * 64,
                turn_count_before=0,
                on_terminal_turn=lambda *identity: acknowledgements.append(identity),
            )
        assert pending.value.reason == "terminal_turn"
        assert pending.value.status == "failed"
        assert (tmp_path / ".arche_codex_request_intent.json").is_file()
        assert (tmp_path / ".arche_codex_terminal_turn.json").is_file()
        assert BlockingTerminalCodex.thread.failed_turn.interrupt_calls == 0
    finally:
        released.set()

    with pytest.raises(adapter.CodexTerminalTurnError):
        adapter.run_codex_sdk_turn(
            config=CodexConfig(),
            request=request,
            thread_config={},
            session_id=BlockingTerminalCodex.thread.id,
            trace_path=tmp_path / "resume.jsonl",
            completed_turn_receipt_root=receipt_root,
            request_digest="9" * 64,
            turn_count_before=0,
            on_terminal_turn=lambda *identity: acknowledgements.append(identity),
        )
    assert acknowledgements == [
        (BlockingTerminalCodex.thread.id, "turn-terminal-failed", "failed"),
        (BlockingTerminalCodex.thread.id, "turn-terminal-failed", "failed"),
    ]
    assert BlockingTerminalCodex.thread.turn_calls == 1
    assert not (tmp_path / ".arche_codex_request_intent.json").exists()
    assert not (tmp_path / ".arche_codex_terminal_turn.json").exists()


def test_interrupt_cleanup_contains_terminal_race_exception(capsys) -> None:
    """残余 interrupt 竞态必须在后台 wrapper 内收束，不能打印线程 traceback。"""

    class EndedTurn:
        def interrupt(self):
            raise RuntimeError("no active turn to interrupt")

    worker = threading.Thread(target=lambda: None)
    worker.start()
    worker.join()

    assert adapter._interrupt_and_join_worker(EndedTurn(), worker) is False
    assert "no active turn" not in capsys.readouterr().err


def test_sdk_adapter_history_read_error_has_no_secret_exception_chain() -> None:
    """history 读取异常必须脱敏，cause/context/traceback 都不能带 provider secret。"""

    class SecretHistory:
        def read(self, *, include_turns: bool = False):
            assert include_turns is True
            raise RuntimeError("provider-secret-history-value")

    with pytest.raises(adapter.HarnessError) as failure:
        adapter._read_thread_turns(SecretHistory(), expected_thread_id="thread-1")

    error = failure.value
    rendered = "".join(traceback.format_exception(error))
    assert "provider-secret-history-value" not in str(error)
    assert "provider-secret-history-value" not in rendered
    assert error.__cause__ is None
    assert error.__context__ is None


def test_sdk_adapter_unreceipted_history_cannot_replay_same_prompt_with_new_tools(
    monkeypatch, tmp_path: Path
) -> None:
    """无 receipt 时即使 prompt 相同，intent 摘要不同也不能 replay。"""

    _ReadableCodex.thread = _ReadableThread()
    monkeypatch.setattr(adapter, "Codex", _ReadableCodex)
    original_writer = adapter._write_completion_pending
    failed_once = True

    def fail_before_completion_journal(*args, **kwargs):
        nonlocal failed_once
        if failed_once:
            failed_once = False
            raise OSError("simulated crash before completion journal")
        return original_writer(*args, **kwargs)

    monkeypatch.setattr(adapter, "_write_completion_pending", fail_before_completion_journal)
    base_request = HarnessRequest(
        messages=({"role": "user", "content": "same prompt"},),
        model="codex-model",
        role="investigation",
        stage="investigation_and_experiment",
        workspace=tmp_path / "workspace",
        tools=({"name": "tool-a"},),
        timeout_seconds=None,
    )
    changed_request = HarnessRequest(
        messages=base_request.messages,
        model=base_request.model,
        role=base_request.role,
        stage=base_request.stage,
        workspace=base_request.workspace,
        tools=({"name": "tool-b"},),
        timeout_seconds=None,
    )
    receipt_root = tmp_path / "receipts"

    with pytest.raises(adapter.CodexCompletionRecoveryPending):
        adapter.run_codex_sdk_turn(
            config=CodexConfig(),
            request=base_request,
            thread_config={},
            session_id=None,
            trace_path=tmp_path / "first.jsonl",
            completed_turn_receipt_root=receipt_root,
            request_digest="d" * 64,
            turn_count_before=0,
        )

    with pytest.raises(adapter.HarnessError, match="request intent"):
        adapter.run_codex_sdk_turn(
            config=CodexConfig(),
            request=changed_request,
            thread_config={},
            session_id="thread-replay",
            trace_path=tmp_path / "second.jsonl",
            completed_turn_receipt_root=receipt_root,
            request_digest="e" * 64,
            turn_count_before=0,
        )

    assert _ReadableCodex.thread.turn_calls == 1


def test_sdk_adapter_rejects_receipt_when_effective_thread_policy_drifts(
    monkeypatch, tmp_path: Path
) -> None:
    """provider/MCP 有效配置变化时，旧 receipt 不能被当作当前请求重放。"""

    _ReadableCodex.thread = _ReadableThread()
    monkeypatch.setattr(adapter, "Codex", _ReadableCodex)
    request = HarnessRequest(
        messages=({"role": "user", "content": "policy drift"},),
        model="codex-model",
        role="investigation",
        stage="investigation_and_experiment",
        workspace=tmp_path / "workspace",
    )
    old_config = {
        "model_provider": "arche_codex",
        "model_providers": {
            "arche_codex": {
                "base_url": "http://old-provider/v1",
                "env_key": "ARCHE_CODEX_API_KEY",
                "wire_api": "responses",
            }
        },
        "mcp_servers": {
            "minichem_toolbox": {
                "url": "http://127.0.0.1:9011/mcp",
                "env_http_headers": {"X-MiniChem": "MINICHEM_TOKEN_ENV"},
                "required": True,
                "startup_timeout_sec": 60,
                "tool_timeout_sec": 21600,
            }
        },
    }
    new_config = copy.deepcopy(old_config)
    new_config["model_providers"]["arche_codex"]["base_url"] = "http://new-provider/v1"
    new_config["mcp_servers"]["minichem_toolbox"]["url"] = "http://127.0.0.1:9012/mcp"
    receipt_root = tmp_path / "receipts"
    request_digest = "a" * 64

    adapter.run_codex_sdk_turn(
        config=CodexConfig(),
        request=request,
        thread_config=old_config,
        session_id=None,
        trace_path=tmp_path / "first.jsonl",
        provider_id="arche_codex",
        env_key="ARCHE_CODEX_API_KEY",
        completed_turn_receipt_root=receipt_root,
        request_digest=request_digest,
        turn_count_before=0,
    )
    receipt_text = (receipt_root / "turn-1.json").read_text(encoding="utf-8")
    assert "header-secret" not in receipt_text
    assert "thread_policy_digest" in receipt_text

    with pytest.raises(adapter.HarnessError, match="policy"):
        adapter.run_codex_sdk_turn(
            config=CodexConfig(),
            request=request,
            thread_config=new_config,
            session_id="thread-replay",
            trace_path=tmp_path / "second.jsonl",
            provider_id="arche_codex",
            env_key="ARCHE_CODEX_API_KEY",
            completed_turn_receipt_root=receipt_root,
            request_digest=request_digest,
            turn_count_before=0,
        )

    assert _ReadableCodex.thread.turn_calls == 1


def test_thread_policy_digest_is_stable_across_stages_and_binds_effective_policy(
    tmp_path: Path,
) -> None:
    """同一 thread 的阶段变化不漂移，provider、模型、profile 和 MCP 变化会漂移。"""

    base_config = {
        "model_provider": "arche_codex",
        "model_providers": {
            "arche_codex": {
                "base_url": "http://provider/v1",
                "env_key": "ARCHE_CODEX_API_KEY",
            }
        },
        "mcp_servers": {"evidence_reader": {"url": "http://evidence/mcp"}},
    }
    digest = adapter.calculate_codex_thread_policy_digest(
        thread_config=base_config,
        workspace=tmp_path / "workspace",
        provider_id="arche_codex",
        env_key="ARCHE_CODEX_API_KEY",
        model="codex-model",
        role="task_review",
        stage="initial_review",
        profile="task_review",
    )
    assert digest == adapter.calculate_codex_thread_policy_digest(
        thread_config=base_config,
        workspace=tmp_path / "workspace",
        provider_id="arche_codex",
        env_key="ARCHE_CODEX_API_KEY",
        model="codex-model",
        role="task_review",
        stage="next_batch_review",
        profile="task_review",
    )
    def digest_for(
        config: dict[str, object],
        *,
        provider_id: str = "arche_codex",
        model: str = "codex-model",
        role: str = "task_review",
        profile: str = "task_review",
    ) -> str:
        return adapter.calculate_codex_thread_policy_digest(
            thread_config=config,
            workspace=tmp_path / "workspace",
            provider_id=provider_id,
            env_key="ARCHE_CODEX_API_KEY",
            model=model,
            role=role,
            stage="next_batch_review",
            profile=profile,
        )

    changed_url = copy.deepcopy(base_config)
    changed_url["model_providers"]["arche_codex"]["base_url"] = "http://other/v1"
    changed_mcp = copy.deepcopy(base_config)
    changed_mcp["mcp_servers"]["evidence_reader"]["url"] = "http://other-evidence/mcp"
    for changed in (
        (changed_url, {"provider_id": "arche_codex"}),
        (changed_mcp, {"provider_id": "arche_codex"}),
        (base_config, {"provider_id": "other_provider"}),
        (base_config, {"model": "other-model"}),
        (base_config, {"profile": "investigation", "role": "investigation"}),
    ):
        config, fields = changed
        assert digest_for(config, **fields) != digest


def test_sdk_adapter_recovery_callback_failure_preserves_worker_classification(
    monkeypatch, tmp_path: Path
) -> None:
    """recovery callback 抛错时仍返回 recovery_pending，而不是覆盖 worker 分类。"""

    turn = _IdleTurn(interruptible=False)
    monkeypatch.setattr(adapter, "Codex", lambda _config: _IdleCodex(turn))
    request = HarnessRequest(
        messages=({"role": "user", "content": "callback failure"},),
        model="codex-model",
        role="investigation",
        stage="investigation_and_experiment",
        workspace=tmp_path / "workspace",
        timeout_seconds=None,
    )

    try:
        with pytest.raises(adapter.CodexWorkerRecoveryPending) as failure:
            adapter.run_codex_sdk_turn(
                config=CodexConfig(),
                request=request,
                thread_config={},
                session_id=None,
                trace_path=tmp_path / "trace.jsonl",
                idle_timeout_seconds=0.01,
                on_recovery_pending=lambda _turn_id: (_ for _ in ()).throw(
                    RuntimeError("callback write failed")
                ),
            )
        assert failure.value.kind == "recovery_pending"
        assert failure.value.reason == "worker_cleanup"
    finally:
        turn.released.set()


def test_sdk_adapter_redacts_explicit_trace_sensitive_values(monkeypatch, tmp_path: Path) -> None:
    """调用方额外提供的 header 值也必须从 SDK notification trace 中清除。"""

    monkeypatch.setattr(adapter, "Codex", _LeakingCodex)
    request = HarnessRequest(
        messages=(
            {"role": "user", "content": "investigate"},
        ),
        model="codex-model",
        role="investigation",
        stage="investigation_and_experiment",
        workspace=tmp_path / "workspace",
        timeout_seconds=None,
    )

    adapter.run_codex_sdk_turn(
        config=CodexConfig(),
        request=request,
        thread_config={},
        session_id=None,
        trace_path=tmp_path / "trace.jsonl",
        trace_sensitive_values=("explicit-header-secret",),
    )

    trace = (tmp_path / "trace.jsonl").read_text(encoding="utf-8")
    assert "explicit-header-secret" not in trace
    assert "[REDACTED]" in trace


def test_sdk_adapter_replays_matching_persisted_turn_without_second_model_call(
    monkeypatch, tmp_path: Path
) -> None:
    """匹配 receipt 和完整 thread/read 历史时只重放正文，不启动第二个 turn。"""

    _ReadableCodex.thread = _ReadableThread()
    monkeypatch.setattr(adapter, "Codex", _ReadableCodex)
    request = HarnessRequest(
        messages=({"role": "user", "content": "replay me"},),
        model="codex-model",
        role="investigation",
        stage="investigation_and_experiment",
        workspace=tmp_path / "workspace",
        timeout_seconds=None,
    )
    receipt_root = tmp_path / "receipts"
    digest = "a" * 64

    adapter.run_codex_sdk_turn(
        config=CodexConfig(),
        request=request,
        thread_config={},
        session_id=None,
        trace_path=tmp_path / "first.jsonl",
        completed_turn_receipt_root=receipt_root,
        request_digest=digest,
        turn_count_before=0,
    )
    replay = adapter.run_codex_sdk_turn(
        config=CodexConfig(),
        request=request,
        thread_config={},
        session_id="thread-replay",
        trace_path=tmp_path / "second.jsonl",
        completed_turn_receipt_root=receipt_root,
        request_digest=digest,
        turn_count_before=0,
    )

    assert replay.replayed is True
    assert replay.response.content == '{"ok":true}'
    assert replay.turn_id == "turn-1"
    assert _ReadableCodex.thread.turn_calls == 1
    assert _ReadableCodex.thread.read_calls == [True]


def test_sdk_adapter_replays_waiting_turn_after_prompt_drift_and_aborted_tail(
    monkeypatch, tmp_path: Path
) -> None:
    """waiting turn 之后只有空 aborted 尾部时，恢复应回滚尾部而不新开模型 turn。"""

    class _TrailingThread:
        id = "thread-trailing"

        def __init__(self) -> None:
            self.turn_calls = 0
            self.read_calls: list[bool] = []
            self.rollback_calls: list[tuple[str, object]] = []
            self.turns: list[dict[str, object]] = []

        def turn(self, *_args, **_kwargs):
            self.turn_calls += 1
            raise AssertionError("waiting replay must not start a new model turn")

        def read(self, *, include_turns: bool = False):
            self.read_calls.append(include_turns)
            return {"thread": {"id": self.id, "turns": list(self.turns)}}

        @property
        def _client(self):
            thread = self

            class _Client:
                def request(self, method, params, *, response_model):
                    thread.rollback_calls.append((method, params))
                    assert method == "thread/rollback"
                    assert params == {"threadId": thread.id, "numTurns": 1}
                    thread.turns = thread.turns[:-1]
                    return {"thread": {"id": thread.id, "turns": list(thread.turns)}}

            return _Client()

    class _TrailingCodex:
        thread = _TrailingThread()

        def __init__(self, _config):
            return None

        def thread_start(self, **_kwargs):
            return self.thread

        def thread_resume(self, *_args, **_kwargs):
            return self.thread

        def close(self):
            return None

    _TrailingCodex.thread.turns = [
        {"id": "turn-failed-1", "status": "failed", "items": []},
        {"id": "turn-failed-2", "status": "interrupted", "items": []},
        {
            "id": "turn-waiting",
            "status": "completed",
            "items_view": "full",
            "items": [
                {
                    "type": "userMessage",
                    "content": [{"type": "text", "text": "original prompt"}],
                },
                {
                    "type": "agentMessage",
                    "phase": "final_answer",
                    "text": '{"ok":true}',
                },
            ],
        },
        {"id": "turn-aborted", "status": "aborted", "items": []},
    ]
    monkeypatch.setattr(adapter, "Codex", _TrailingCodex)
    request = HarnessRequest(
        messages=({"role": "user", "content": "replacement prompt"},),
        model="codex-model",
        role="investigation",
        stage="investigation_job_continuation",
        workspace=tmp_path / "workspace",
    )
    receipt_root = tmp_path / "receipts"
    digest = "a" * 64
    effective_thread_config = adapter.build_codex_capability_config(
        thread_config={},
        workspace=request.workspace,
        profile="investigation",
        env_key="ARCHE_CODEX_API_KEY",
        environment_keys=(),
        auth_mode="api",
    )
    policy_digest = adapter.calculate_codex_thread_policy_digest(
        thread_config=effective_thread_config,
        workspace=request.workspace,
        provider_id="arche_codex",
        env_key="ARCHE_CODEX_API_KEY",
        model=request.model,
        role=request.role,
        stage=request.stage,
        profile="investigation",
    )
    adapter._write_completed_turn_receipt(
        receipt_root,
        thread_id="thread-trailing",
        turn_id="turn-waiting",
        request_digest=digest,
        turn_count_before=0,
        thread_policy_digest=policy_digest,
    )
    adapter._write_request_intent(
        receipt_root,
        thread_id="thread-trailing",
        request_digest="d3" * 32,
        turn_count_before=1,
        physical_turn_count_before=3,
        provider_id="arche_codex",
        env_key="ARCHE_CODEX_API_KEY",
        model=request.model,
        role=request.role,
        stage=request.stage,
        profile="investigation",
        workspace=request.workspace,
        thread_policy_digest=policy_digest,
    )

    replay = adapter.run_codex_sdk_turn(
        config=CodexConfig(),
        request=request,
        thread_config={},
        session_id="thread-trailing",
        trace_path=tmp_path / "replay.jsonl",
        completed_turn_receipt_root=receipt_root,
        request_digest="b" * 64,
        turn_count_before=1,
        thread_policy_digest=policy_digest,
        replay_turn_id="turn-waiting",
    )

    assert replay.replayed is True
    assert replay.turn_id == "turn-waiting"
    assert _TrailingCodex.thread.turn_calls == 0
    assert _TrailingCodex.thread.rollback_calls == [
        ("thread/rollback", {"threadId": "thread-trailing", "numTurns": 1})
    ]
    assert [turn["id"] for turn in _TrailingCodex.thread.turns] == [
        "turn-failed-1",
        "turn-failed-2",
        "turn-waiting",
    ]
    assert not (receipt_root.parent / ".arche_codex_request_intent.json").exists()
    assert (
        receipt_root.parent / ".arche_codex_trailing_rollback_intent.json"
    ).exists()
    assert (
        receipt_root.parent / ".arche_codex_trailing_rollback_receipt.json"
    ).exists()

    replay_again = adapter.run_codex_sdk_turn(
        config=CodexConfig(),
        request=request,
        thread_config={},
        session_id="thread-trailing",
        trace_path=tmp_path / "replay-again.jsonl",
        completed_turn_receipt_root=receipt_root,
        request_digest="c" * 64,
        turn_count_before=1,
        thread_policy_digest=policy_digest,
        replay_turn_id="turn-waiting",
    )
    assert replay_again.replayed is True
    assert _TrailingCodex.thread.turn_calls == 0
    assert len(_TrailingCodex.thread.rollback_calls) == 1

    with pytest.raises(adapter.HarnessError):
        adapter.run_codex_sdk_turn(
            config=CodexConfig(),
            request=request,
            thread_config={},
            session_id="thread-trailing",
            trace_path=tmp_path / "replay-wrong-turn.jsonl",
            completed_turn_receipt_root=receipt_root,
            request_digest="d" * 64,
            turn_count_before=1,
            thread_policy_digest=policy_digest,
            replay_turn_id="turn-not-present",
        )
    with pytest.raises(adapter.HarnessError):
        adapter.run_codex_sdk_turn(
            config=CodexConfig(),
            request=request,
            thread_config={},
            session_id="thread-trailing",
            trace_path=tmp_path / "replay-wrong-policy.jsonl",
            completed_turn_receipt_root=receipt_root,
            request_digest="e" * 64,
            turn_count_before=1,
            thread_policy_digest="f" * 64,
            replay_turn_id="turn-waiting",
        )
    with pytest.raises(adapter.HarnessError):
        adapter.run_codex_sdk_turn(
            config=CodexConfig(),
            request=request,
            thread_config={},
            session_id="thread-trailing",
            trace_path=tmp_path / "replay-wrong-count.jsonl",
            completed_turn_receipt_root=receipt_root,
            request_digest="e" * 64,
            turn_count_before=2,
            thread_policy_digest=policy_digest,
            replay_turn_id="turn-waiting",
        )


def test_sdk_adapter_replay_preserves_completed_mcp_items_for_wake_recovery(
    monkeypatch, tmp_path: Path
) -> None:
    """重启 replay 必须带回同一 turn 的完整工具回执，才能恢复多作业 tracker。"""

    _ReadableCodex.thread = _ReadableThread()
    monkeypatch.setattr(adapter, "Codex", _ReadableCodex)
    request = HarnessRequest(
        messages=({"role": "user", "content": "replay jobs"},),
        model="codex-model",
        role="investigation",
        stage="investigation_and_experiment",
        workspace=tmp_path / "workspace",
        timeout_seconds=None,
    )
    receipt_root = tmp_path / "receipts"
    digest = "c" * 64
    adapter.run_codex_sdk_turn(
        config=CodexConfig(),
        request=request,
        thread_config={},
        session_id=None,
        trace_path=tmp_path / "first.jsonl",
        completed_turn_receipt_root=receipt_root,
        request_digest=digest,
        turn_count_before=0,
    )
    completed_submit = {
        "type": "mcpToolCall",
        "status": "completed",
        "tool": "submit_transition_state_path_job",
        "arguments": {"job_id": "job-secondary"},
        "result": {
            "structuredContent": {
                "status": "success",
                "job_id": "job-secondary",
                "operation_id": "operation-secondary",
                "submission_key": "submission-secondary",
            }
        },
    }
    _ReadableCodex.thread.history_tool_items = (completed_submit,)

    replay = adapter.run_codex_sdk_turn(
        config=CodexConfig(),
        request=request,
        thread_config={},
        session_id="thread-replay",
        trace_path=tmp_path / "second.jsonl",
        completed_turn_receipt_root=receipt_root,
        request_digest=digest,
        turn_count_before=0,
    )

    assert replay.replayed is True
    assert replay.completed_tool_items == (completed_submit,)
    assert _ReadableCodex.thread.turn_calls == 1


def test_sdk_adapter_skips_pre_turn_fence_when_completed_receipt_replays(
    monkeypatch, tmp_path: Path
) -> None:
    """已有 completed receipt 时只重放，不因遗留 inventory 再调用新 turn fence。"""

    _ReadableCodex.thread = _ReadableThread()
    monkeypatch.setattr(adapter, "Codex", _ReadableCodex)
    request = HarnessRequest(
        messages=({"role": "user", "content": "fenced replay"},),
        model="codex-model",
        role="investigation",
        stage="investigation_and_experiment",
        workspace=tmp_path / "workspace",
        timeout_seconds=None,
    )
    receipt_root = tmp_path / "receipts"
    digest = "f" * 64
    adapter.run_codex_sdk_turn(
        config=CodexConfig(),
        request=request,
        thread_config={},
        session_id=None,
        trace_path=tmp_path / "first.jsonl",
        completed_turn_receipt_root=receipt_root,
        request_digest=digest,
        turn_count_before=0,
    )
    fence_calls: list[str] = []

    def fail_if_called() -> None:
        """若执行则说明 receipt-aware 顺序错误。"""

        fence_calls.append("called")
        raise AssertionError("replay 不应调用新 turn fence")

    replay = adapter.run_codex_sdk_turn(
        config=CodexConfig(),
        request=request,
        thread_config={},
        session_id="thread-replay",
        trace_path=tmp_path / "second.jsonl",
        completed_turn_receipt_root=receipt_root,
        request_digest=digest,
        turn_count_before=0,
        before_new_turn=fail_if_called,
    )

    assert replay.replayed is True
    assert fence_calls == []
    assert _ReadableCodex.thread.turn_calls == 1


def test_thread_aware_pre_turn_fence_runs_after_regular_fence() -> None:
    """thread-aware 栅栏在普通 scope 栅栏之后执行，并获得真实 thread 身份。"""

    calls: list[object] = []
    thread = object()
    adapter._run_codex_pre_turn_fence(
        lambda: calls.append("scope"),
        before_new_turn_with_thread=lambda actual_thread, thread_id: calls.append(
            (actual_thread, thread_id)
        ),
        thread=thread,
        thread_id="thread-legacy",
        sensitive_values=(),
    )
    assert calls == ["scope", (thread, "thread-legacy")]


def test_sdk_adapter_refuses_new_thread_when_receipt_exists_without_session_id(
    monkeypatch, tmp_path: Path
) -> None:
    """manifest 丢失但已有完成 receipt 时不能新建另一条 thread。"""

    _ReadableCodex.thread = _ReadableThread()
    monkeypatch.setattr(adapter, "Codex", _ReadableCodex)
    request = HarnessRequest(
        messages=({"role": "user", "content": "orphan receipt"},),
        model="codex-model",
        role="investigation",
        stage="investigation_and_experiment",
        workspace=tmp_path / "workspace",
        timeout_seconds=None,
    )
    receipt_root = tmp_path / "receipts"
    adapter.run_codex_sdk_turn(
        config=CodexConfig(),
        request=request,
        thread_config={},
        session_id=None,
        trace_path=tmp_path / "first.jsonl",
        completed_turn_receipt_root=receipt_root,
        request_digest="a" * 64,
        turn_count_before=0,
    )

    with pytest.raises(adapter.HarnessError, match="original Codex session"):
        adapter.run_codex_sdk_turn(
            config=CodexConfig(),
            request=request,
            thread_config={},
            session_id=None,
            trace_path=tmp_path / "orphan.jsonl",
            completed_turn_receipt_root=receipt_root,
            request_digest="a" * 64,
            turn_count_before=0,
        )

    assert _ReadableCodex.thread.turn_calls == 1


def test_sdk_adapter_runs_pre_turn_fence_before_new_worker_and_fail_closes(
    monkeypatch, tmp_path: Path
) -> None:
    """没有可重放 receipt 时 fence 失败必须发生在 worker/model turn 之前。"""

    _ReadableCodex.thread = _ReadableThread()
    monkeypatch.setattr(adapter, "Codex", _ReadableCodex)
    request = HarnessRequest(
        messages=({"role": "user", "content": "new fenced turn"},),
        model="codex-model",
        role="investigation",
        stage="investigation_and_experiment",
        workspace=tmp_path / "workspace",
        timeout_seconds=None,
    )
    secret = "inventory-secret-should-not-leak"

    def reject_new_turn() -> None:
        """模拟 gateway 发现遗留副作用并拒绝新模型 turn。"""

        raise RuntimeError(secret)

    with pytest.raises(adapter.HarnessError, match="pre-turn fence") as error:
        adapter.run_codex_sdk_turn(
            config=CodexConfig(),
            request=request,
            thread_config={},
            session_id=None,
            trace_path=tmp_path / "blocked.jsonl",
            before_new_turn=reject_new_turn,
        )

    assert secret not in str(error.value)
    assert _ReadableCodex.thread.turn_calls == 0


def test_sdk_adapter_rejects_corrupt_completed_turn_receipt_before_model_call(
    monkeypatch, tmp_path: Path
) -> None:
    """receipt 损坏时直接 fail-closed，不以新模型调用掩盖未知恢复状态。"""

    _ReadableCodex.thread = _ReadableThread()
    monkeypatch.setattr(adapter, "Codex", _ReadableCodex)
    request = HarnessRequest(
        messages=({"role": "user", "content": "replay me"},),
        model="codex-model",
        role="investigation",
        stage="investigation_and_experiment",
        workspace=tmp_path / "workspace",
        timeout_seconds=None,
    )
    receipt_root = tmp_path / "receipts"
    digest = "b" * 64
    adapter.run_codex_sdk_turn(
        config=CodexConfig(),
        request=request,
        thread_config={},
        session_id=None,
        trace_path=tmp_path / "first.jsonl",
        completed_turn_receipt_root=receipt_root,
        request_digest=digest,
        turn_count_before=0,
    )
    (receipt_root / "turn-1.json").write_text("{broken", encoding="utf-8")

    with pytest.raises(adapter.HarnessError, match="receipt"):
        adapter.run_codex_sdk_turn(
            config=CodexConfig(),
            request=request,
            thread_config={},
            session_id="thread-replay",
            trace_path=tmp_path / "second.jsonl",
            completed_turn_receipt_root=receipt_root,
            request_digest=digest,
            turn_count_before=0,
        )

    assert _ReadableCodex.thread.turn_calls == 1


def test_sdk_adapter_rejects_completed_receipt_thread_drift_before_model_call(
    monkeypatch, tmp_path: Path
) -> None:
    """receipt 的 thread 身份漂移时 fail-closed，不把历史结果交给当前 thread。"""

    _ReadableCodex.thread = _ReadableThread()
    monkeypatch.setattr(adapter, "Codex", _ReadableCodex)
    request = HarnessRequest(
        messages=({"role": "user", "content": "replay me"},),
        model="codex-model",
        role="investigation",
        stage="investigation_and_experiment",
        workspace=tmp_path / "workspace",
        timeout_seconds=None,
    )
    receipt_root = tmp_path / "receipts"
    digest = "c" * 64
    adapter.run_codex_sdk_turn(
        config=CodexConfig(),
        request=request,
        thread_config={},
        session_id=None,
        trace_path=tmp_path / "first.jsonl",
        completed_turn_receipt_root=receipt_root,
        request_digest=digest,
        turn_count_before=0,
    )
    (receipt_root / "turn-1.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "thread_id": "different-thread",
                "turn_id": "turn-1",
                "request_digest": digest,
                "turn_count_before": 0,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(adapter.HarnessError, match="policy digest is required"):
        adapter.run_codex_sdk_turn(
            config=CodexConfig(),
            request=request,
            thread_config={},
            session_id="thread-replay",
            trace_path=tmp_path / "second.jsonl",
            completed_turn_receipt_root=receipt_root,
            request_digest=digest,
            turn_count_before=0,
        )

    assert _ReadableCodex.thread.turn_calls == 1


def test_sdk_adapter_rejects_same_digest_receipt_when_full_prompt_differs(
    monkeypatch, tmp_path: Path
) -> None:
    """即使摘要被错误复用，也必须用完整 prompt 核对并拒绝旧正文。"""

    _ReadableCodex.thread = _ReadableThread()
    monkeypatch.setattr(adapter, "Codex", _ReadableCodex)
    first_request = HarnessRequest(
        messages=({"role": "user", "content": "original prompt"},),
        model="codex-model",
        role="investigation",
        stage="investigation_and_experiment",
        workspace=tmp_path / "workspace",
        timeout_seconds=None,
    )
    second_request = HarnessRequest(
        messages=({"role": "user", "content": "different prompt"},),
        model="codex-model",
        role="investigation",
        stage="investigation_and_experiment",
        workspace=tmp_path / "workspace",
        timeout_seconds=None,
    )
    receipt_root = tmp_path / "receipts"
    digest = "d" * 64
    adapter.run_codex_sdk_turn(
        config=CodexConfig(),
        request=first_request,
        thread_config={},
        session_id=None,
        trace_path=tmp_path / "first.jsonl",
        completed_turn_receipt_root=receipt_root,
        request_digest=digest,
        turn_count_before=0,
    )

    with pytest.raises(adapter.HarnessError, match="user prompt"):
        adapter.run_codex_sdk_turn(
            config=CodexConfig(),
            request=second_request,
            thread_config={},
            session_id="thread-replay",
            trace_path=tmp_path / "second.jsonl",
            completed_turn_receipt_root=receipt_root,
            request_digest=digest,
            turn_count_before=0,
        )

    assert _ReadableCodex.thread.turn_calls == 1


def test_acknowledge_codex_receipt_missing_fails_closed(tmp_path: Path) -> None:
    """成功确认不能把缺失 receipt 当作幂等空操作。"""

    with pytest.raises(adapter.HarnessError, match="receipt is missing"):
        adapter.acknowledge_codex_completed_turn_receipt(
            tmp_path / "receipts",
            thread_id="thread-1",
            turn_id="turn-1",
            request_digest="a" * 64,
            turn_count_before=0,
            thread_policy_digest="b" * 64,
        )


def test_sdk_adapter_applies_profile_to_thread_and_keeps_turn_minimal(
    monkeypatch, tmp_path: Path
) -> None:
    """能力策略固定在 thread，turn 不覆盖审批、工作区、模型和沙箱。"""

    _CapturingCodex.calls = []
    monkeypatch.setattr(adapter, "Codex", _CapturingCodex)
    monkeypatch.setenv("ARCHE_CODEX_API_KEY", "parent-secret-must-not-reach-shell")
    investigation = HarnessRequest(
        messages=({"role": "user", "content": "investigate"},),
        model="codex-model",
        role="investigation",
        stage="investigation_and_experiment",
        profile="investigation",
        workspace=tmp_path / "investigation",
        response_schema={"type": "object"},
    )

    adapter.run_codex_sdk_turn(
        config=CodexConfig(),
        request=investigation,
        thread_config={
            "model_provider": "arche_codex",
            "model_reasoning_effort": "max",
            "mcp_servers": {
                "minichem_toolbox": {
                    "url": "http://127.0.0.1:9011/mcp",
                    "env_http_headers": {"X-MiniChem": "MINICHEM_TOKEN_ENV"},
                    "required": True,
                    "startup_timeout_sec": 60,
                    "tool_timeout_sec": 21600,
                }
            },
        },
        session_id=None,
        trace_path=tmp_path / "investigation.jsonl",
    )

    start_kind, start_payload = _CapturingCodex.calls[0]
    assert start_kind == "thread_start"
    assert start_payload["approval_mode"] is ApprovalMode.deny_all
    assert start_payload["ephemeral"] is False
    assert start_payload["sandbox"] is Sandbox.full_access
    assert start_payload["config"]["model_provider"] == "arche_codex"
    assert start_payload["config"]["model_reasoning_effort"] == "max"
    assert start_payload["config"]["mcp_servers"]["minichem_toolbox"]["required"] is True
    assert (
        start_payload["config"]["mcp_servers"]["minichem_toolbox"][
            "default_tools_approval_mode"
        ]
        == "approve"
    )
    rebuilt_config = adapter.build_codex_capability_config(
        thread_config=start_payload["config"],
        workspace=investigation.workspace,
        profile="investigation",
        env_key="ARCHE_CODEX_API_KEY",
        environment_keys=("MINICHEM_TOKEN_ENV",),
    )
    assert (
        rebuilt_config["mcp_servers"]["minichem_toolbox"][
            "default_tools_approval_mode"
        ]
        == "approve"
    )
    assert start_payload["config"]["web_search"] == "disabled"
    assert start_payload["config"]["features"]["multi_agent"] is False
    assert start_payload["config"]["agents"] == {"enabled": False}
    assert start_payload["config"]["features"]["apps"] is False
    assert start_payload["config"]["features"]["plugins"] is False
    assert start_payload["config"]["features"]["browser_use"] is False
    assert start_payload["config"]["features"]["computer_use"] is False
    assert start_payload["config"]["features"]["shell_tool"] is True
    assert start_payload["config"]["shell_environment_policy"] == {
        "inherit": "core",
        "ignore_default_excludes": False,
        "exclude": [
            "ARCHE_CODEX_API_KEY",
            "ARCHE_CHEM_API_KEY",
            "ARCHE_REVIEW_CODEX_API_KEY",
            "DEEPSEEK_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GOOGLE_API_KEY",
            "GEMINI_API_KEY",
            "AZURE_OPENAI_API_KEY",
            "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY",
                "AWS_SESSION_TOKEN",
                "HF_TOKEN",
                "MINICHEM_TOKEN_ENV",
            ],
        }
    assert "parent-secret-must-not-reach-shell" not in str(start_payload["config"])
    assert start_payload["config"]["sandbox_workspace_write"] == {
        "network_access": False,
        "exclude_slash_tmp": True,
        "exclude_tmpdir_env_var": True,
        "writable_roots": [str((tmp_path / "investigation").resolve())],
    }

    turn_kind, turn_payload = _CapturingCodex.calls[1]
    assert turn_kind == "turn"
    turn_args, turn_kwargs = turn_payload
    assert len(turn_args) == 1
    # Investigation 的最终 JSON schema 只通过条件性 prompt 呈现；submit 工具
    # 成功后必须结束 turn，不能让 SDK output_schema 强迫模型继续生成正文。
    assert turn_kwargs == {}
    assert not {"approval_mode", "cwd", "model", "sandbox"} & set(turn_kwargs)

    adapter.run_codex_sdk_turn(
        config=CodexConfig(),
        request=investigation,
        thread_config={
            "model_provider": "arche_codex",
            "model_reasoning_effort": "max",
        },
        session_id="thread-1",
        trace_path=tmp_path / "investigation-resume.jsonl",
    )
    investigation_resume_kind, investigation_resume_payload = _CapturingCodex.calls[2]
    assert investigation_resume_kind == "thread_resume"
    assert investigation_resume_payload[1]["approval_mode"] is ApprovalMode.deny_all
    assert investigation_resume_payload[1]["model"] == "codex-model"
    assert investigation_resume_payload[1]["config"]["model_reasoning_effort"] == "max"
    assert investigation_resume_payload[1]["sandbox"] is Sandbox.full_access

    review = HarnessRequest(
        messages=({"role": "user", "content": "review"},),
        model="codex-model",
        role="review",
        stage="workflow_revision",
        profile="review",
        workspace=tmp_path / "review",
    )
    adapter.run_codex_sdk_turn(
        config=CodexConfig(),
        request=review,
        thread_config={},
        session_id=None,
        trace_path=tmp_path / "review.jsonl",
    )
    review_start_kind, review_start_payload = _CapturingCodex.calls[4]
    assert review_start_kind == "thread_start"
    assert review_start_payload["approval_mode"] is ApprovalMode.deny_all
    assert review_start_payload["ephemeral"] is False
    assert review_start_payload["sandbox"] is Sandbox.workspace_write

    adapter.run_codex_sdk_turn(
        config=CodexConfig(),
        request=review,
        thread_config={"model_reasoning_effort": "max"},
        session_id="thread-1",
        trace_path=tmp_path / "review-resume.jsonl",
    )
    resume_kind, resume_payload = _CapturingCodex.calls[6]
    assert resume_kind == "thread_resume"
    assert resume_payload[1]["approval_mode"] is ApprovalMode.deny_all
    assert resume_payload[1]["sandbox"] is Sandbox.workspace_write


def test_task_review_role_starts_a_persistent_review_thread(monkeypatch, tmp_path: Path) -> None:
    """会话存储使用 task_review 时，首轮也必须创建可恢复的 Review thread。"""

    _CapturingCodex.calls = []
    monkeypatch.setattr(adapter, "Codex", _CapturingCodex)
    request = HarnessRequest(
        messages=({"role": "user", "content": "review"},),
        model="codex-model",
        role="task_review",
        stage="workflow_revision",
        workspace=tmp_path / "task-review",
    )

    adapter.run_codex_sdk_turn(
        config=CodexConfig(),
        request=request,
        thread_config={},
        session_id=None,
        trace_path=tmp_path / "task-review.jsonl",
    )

    kind, payload = _CapturingCodex.calls[0]
    assert kind == "thread_start"
    assert payload["approval_mode"] is ApprovalMode.deny_all
    assert payload["ephemeral"] is False


def test_capability_config_excludes_provider_and_mcp_env_header_names(tmp_path: Path) -> None:
    """shell policy 必须排除 provider key 与 env_http_headers 引用的变量名。"""

    config = adapter.build_codex_capability_config(
        thread_config={
            "mcp_servers": {
                "evidence_reader": {
                    "url": "http://127.0.0.1:9020/mcp",
                    "env_http_headers": {
                        "Authorization": "EVIDENCE_CAPABILITY_TOKEN",
                    },
                    "required": True,
                    "startup_timeout_sec": 60,
                    "tool_timeout_sec": 21600,
                }
            }
        },
        workspace=tmp_path / "workspace",
        profile="review",
        env_key="REVIEW_PROVIDER_SECRET",
    )

    excluded = config["shell_environment_policy"]["exclude"]
    assert "REVIEW_PROVIDER_SECRET" in excluded
    assert "EVIDENCE_CAPABILITY_TOKEN" in excluded
    assert "EVIDENCE_CAPABILITY_TOKEN_VALUE" not in str(config)


def test_capability_profiles_fail_closed_for_unknown_tools_and_limit_mcp(tmp_path: Path) -> None:
    """task_review 只读、investigation 只准 MiniChem，未知工具不能偷偷放行。"""

    review = adapter.build_codex_capability_config(
        thread_config={
            "mcp_servers": {
                "evidence_reader": {
                    "url": "http://127.0.0.1:9020/mcp",
                    "env_http_headers": {"Authorization": "EVIDENCE_TOKEN"},
                    "required": True,
                    "startup_timeout_sec": 60,
                    "tool_timeout_sec": 21600,
                }
            }
        },
        workspace=tmp_path / "review",
        profile="task_review",
    )
    assert review["features"]["shell_tool"] is False
    assert review["features"]["unified_exec"] is False
    assert review["features"]["update_plan"] is False

    with pytest.raises(adapter.HarnessError, match="unsupported MCP"):
        adapter.build_codex_capability_config(
            thread_config={"mcp_servers": {"minichem_toolbox": {}}},
            workspace=tmp_path / "review-bad",
            profile="task_review",
        )
    with pytest.raises(adapter.HarnessError, match="unsupported keys"):
        adapter.build_codex_capability_config(
            thread_config={"tools": {"shell": True}},
            workspace=tmp_path / "review-tool-bad",
            profile="task_review",
        )

    investigation = adapter.build_codex_capability_config(
        thread_config={
            "mcp_servers": {
                "minichem_toolbox": {
                    "url": "http://127.0.0.1:9011/mcp",
                    "env_http_headers": {"X-MiniChem": "MINICHEM_TOKEN"},
                    "required": True,
                    "startup_timeout_sec": 60,
                    "tool_timeout_sec": 21600,
                }
            }
        },
        workspace=tmp_path / "investigation",
        profile="investigation",
    )
    assert investigation["features"]["shell_tool"] is True
    assert investigation["features"]["update_plan"] is False
    assert investigation["tools"] == {
        "experimental_request_user_input": {"enabled": False},
        "update_plan": {"enabled": True},
    }


def test_compact_prompt_is_only_available_to_investigation_profile(tmp_path: Path) -> None:
    """压缩提示词和上下文档位只能进入 GPT Investigation。"""

    sentinel = "ARCHE_INVESTIGATION_COMPACT_SENTINEL"
    investigation = adapter.build_codex_capability_config(
        thread_config={
            "compact_prompt": sentinel,
            "model_context_window": 1_000_000,
        },
        workspace=tmp_path / "investigation",
        profile="investigation",
    )

    assert investigation["compact_prompt"] == sentinel
    assert investigation["model_context_window"] == 1_000_000
    with pytest.raises(adapter.HarnessError, match="only available for Investigation"):
        adapter.build_codex_capability_config(
            thread_config={"compact_prompt": sentinel},
            workspace=tmp_path / "task-review",
            profile="task_review",
        )
    with pytest.raises(adapter.HarnessError, match="only available for Investigation"):
        adapter.build_codex_capability_config(
            thread_config={"model_context_window": 1_000_000},
            workspace=tmp_path / "task-review-window",
            profile="task_review",
        )


@pytest.mark.parametrize(
    "invalid_prompt",
    (None, "", " \n ", ["prompt"], "prompt\x00content"),
)
def test_investigation_compact_prompt_rejects_invalid_text(
    tmp_path: Path,
    invalid_prompt: object,
) -> None:
    """显式压缩提示词必须是非空多行文本，不能携带隐藏控制字符。"""

    with pytest.raises(adapter.HarnessError, match="non-empty string|control characters"):
        adapter.build_codex_capability_config(
            thread_config={"compact_prompt": invalid_prompt},
            workspace=tmp_path / "investigation",
            profile="investigation",
        )


@pytest.mark.parametrize("invalid_window", (None, True, 0, -1, 1.5, "1000000"))
def test_investigation_context_window_rejects_invalid_values(
    tmp_path: Path,
    invalid_window: object,
) -> None:
    """显式上下文档位必须是正整数，不能依赖 TOML 的隐式转换。"""

    with pytest.raises(adapter.HarnessError, match="positive integer"):
        adapter.build_codex_capability_config(
            thread_config={"model_context_window": invalid_window},
            workspace=tmp_path / "investigation-window",
            profile="investigation",
        )


def test_capability_profiles_reject_non_boolean_features_and_unsafe_planner_mcp(
    tmp_path: Path,
) -> None:
    """能力开关必须是布尔值，Planner stdio 描述符不能扩大进程权限。"""

    with pytest.raises(adapter.HarnessError, match="boolean"):
        adapter.build_codex_capability_config(
            thread_config={"features": {"shell_tool": "true"}},
            workspace=tmp_path / "feature-type",
            profile="investigation",
        )

    descriptor = {
        "command": "/usr/bin/python",
        "args": ["server.py"],
        "env": {"ARCHE_PLANNER_CONTROL_CONTEXT": "/tmp/context.json"},
        "required": True,
        "startup_timeout_sec": 30,
        "tool_timeout_sec": 60,
    }
    accepted = adapter.build_codex_capability_config(
        thread_config={
            "mcp_servers": {
                "arche_planner_control": descriptor,
            }
        },
        workspace=tmp_path / "planner-stdio",
        profile="legacy_planner",
    )
    assert accepted["mcp_servers"]["arche_planner_control"] == {
        **descriptor,
        "default_tools_approval_mode": "approve",
    }

    with pytest.raises(adapter.HarnessError, match="unsupported descriptor fields"):
        adapter.build_codex_capability_config(
            thread_config={
                "mcp_servers": {
                    "arche_planner_control": {**descriptor, "cwd": "/tmp/escape"},
                }
            },
            workspace=tmp_path / "planner-cwd",
            profile="legacy_planner",
        )
    for bad_descriptor in (
        {**descriptor, "command": "relative-server"},
        {**descriptor, "args": "not-a-list"},
        {**descriptor, "args": ["bad\x00argument"]},
    ):
        with pytest.raises(adapter.HarnessError, match="command|args"):
            adapter.build_codex_capability_config(
                thread_config={
                    "mcp_servers": {
                        "arche_planner_control": bad_descriptor,
                    }
                },
                workspace=tmp_path / "planner-bad-command",
                profile="legacy_planner",
            )


def test_capability_mcp_descriptors_reject_process_escape_and_remote_evidence(
    tmp_path: Path,
) -> None:
    """Evidence 与 MiniChem 描述符不能借未知字段启动任意本地程序。"""

    evidence = {
        "url": "http://127.0.0.1:9020/mcp",
        "env_http_headers": {"Authorization": "EVIDENCE_TOKEN"},
        "required": True,
        "startup_timeout_sec": 60,
        "tool_timeout_sec": 21600,
    }
    with pytest.raises(adapter.HarnessError, match="unsupported descriptor fields"):
        adapter.build_codex_capability_config(
            thread_config={
                "mcp_servers": {"evidence_reader": {**evidence, "command": "/tmp/evil"}}
            },
            workspace=tmp_path / "review-command",
            profile="task_review",
        )

    minichem = {
        "url": "http://127.0.0.1:9011/mcp",
        "env_http_headers": {"X-MiniChem-Capability": "MINICHEM_TOKEN"},
        "required": True,
        "startup_timeout_sec": 60,
        "tool_timeout_sec": 21600,
    }
    with pytest.raises(adapter.HarnessError, match="unsupported descriptor fields"):
        adapter.build_codex_capability_config(
            thread_config={
                "mcp_servers": {"minichem_toolbox": {**minichem, "cwd": "/tmp/escape"}}
            },
            workspace=tmp_path / "investigation-cwd",
            profile="investigation",
        )
    with pytest.raises(adapter.HarnessError, match="local HTTP /mcp"):
        adapter.build_codex_capability_config(
            thread_config={
                "mcp_servers": {
                    "evidence_reader": {
                        **evidence,
                        "url": "http://evidence.example/mcp",
                    }
                }
            },
            workspace=tmp_path / "review-remote",
            profile="task_review",
        )
    with pytest.raises(adapter.HarnessError, match="local HTTP /mcp"):
        adapter.build_codex_capability_config(
            thread_config={
                "mcp_servers": {
                    "evidence_reader": {
                        **evidence,
                        "url": "http://[malformed/mcp",
                    }
                }
            },
            workspace=tmp_path / "review-malformed-url",
            profile="task_review",
        )


def test_build_codex_config_injects_private_app_server_environment_without_argv_values(
    tmp_path: Path,
) -> None:
    """secret_environment 只进入 SDK 子进程 env，密钥值不能出现在启动参数。"""

    binary = tmp_path / "codex"
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(binary.stat().st_mode | 0o100)
    config = adapter.build_codex_config(
        codex_bin=str(binary),
        workspace=tmp_path / "workspace",
        codex_home=tmp_path / "codex-home",
        environment_keys=("PROVIDER_SECRET", "MCP_SECRET"),
        secret_environment={
            "PROVIDER_SECRET": "provider-value",
            "MCP_SECRET": "mcp-value",
        },
    )

    assert config.env is not None
    assert config.env["PROVIDER_SECRET"] == "provider-value"
    assert config.env["MCP_SECRET"] == "mcp-value"
    assert "provider-value" not in str(config.launch_args_override)
    assert "mcp-value" not in str(config.launch_args_override)


def test_codex_policy_projection_redacts_and_preserves_only_environment_names() -> None:
    """策略摘要只保留变量名，header、token 和普通字段分别走明确分支。"""

    assert adapter._canonicalize_policy_key_value(
        {"Authorization": "MCP_TOKEN", "X-Bad": "literal-secret"},
        "env_http_headers",
    ) == (
        True,
        {"Authorization": "MCP_TOKEN", "X-Bad": "[REDACTED]"},
    )
    assert adapter._canonicalize_policy_key_value("literal", "env_http_headers") == (
        True,
        "[REDACTED]",
    )
    assert adapter._canonicalize_policy_key_value("MCP_TOKEN", "header_env") == (
        True,
        "MCP_TOKEN",
    )
    assert adapter._canonicalize_policy_key_value("literal-value", "header_env") == (
        True,
        "[REDACTED]",
    )
    assert adapter._canonicalize_policy_key_value("literal", "password") == (
        True,
        "[REDACTED]",
    )
    assert adapter._canonicalize_policy_key_value("ordinary", "label") == (
        False,
        "ordinary",
    )


def test_codex_final_content_and_tool_progress_cover_empty_and_identified_items() -> None:
    """最终正文优先级和工具进度的匿名计数都必须可验证。"""

    final = _FakeItem(_FakeItemRoot(text="final", phase="final_answer"))
    draft = _FakeItem(_FakeItemRoot(text="draft", phase="commentary"))
    assert adapter._extract_final_content([final, draft]) == "final"
    assert adapter._extract_final_content([SimpleNamespace(text="fallback")]) == "fallback"
    with pytest.raises(adapter.HarnessError, match="no final"):
        adapter._extract_final_content([SimpleNamespace(text="", phase=None)])

    active: set[str] = set()
    assert (
        adapter._track_item_progress(
            {"item_id": "mcp-1"},
            item_type="mcpToolCall",
            active_tool_ids=active,
            anonymous_active_tool_count=0,
        )
        == 0
    )
    assert active == {"item:mcpToolCall:mcp-1"}
    assert (
        adapter._track_item_progress(
            {},
            item_type="mcpToolCall",
            active_tool_ids=active,
            anonymous_active_tool_count=0,
        )
        == 0
    )
    active.clear()
    assert (
        adapter._track_item_progress(
            {},
            item_type="commandExecution",
            active_tool_ids=active,
            anonymous_active_tool_count=0,
        )
        == 1
    )


def test_codex_provider_identity_rejects_empty_and_path_values() -> None:
    """provider 标识和密钥变量名必须是非空且不含路径控制字符的文本。"""

    assert adapter._validate_codex_provider_identity(" provider ", " KEY ") == (
        "provider",
        "KEY",
    )
    for provider_id, env_key in (("", "KEY"), ("provider/child", "KEY"), ("provider", ""), ("provider", "KEY\\child")):
        with pytest.raises(adapter.HarnessError, match="provider|env_key"):
            adapter._validate_codex_provider_identity(provider_id, env_key)


def test_codex_replay_turn_selection_rejects_ambiguous_or_partial_history() -> None:
    """receipt 只能选择最后一个唯一的 completed/full turn。"""

    receipt = {"turn_id": "turn-1"}
    complete = SimpleNamespace(
        id="turn-1",
        status="completed",
        items_view="full",
    )
    assert adapter._select_replayed_completed_turn([complete], receipt=receipt) is complete
    cases = (
        ([], "no complete turns"),
        ([SimpleNamespace(id="other", status="completed")], "last turn"),
        ([complete, complete], "ambiguous"),
        ([SimpleNamespace(id="turn-1", status="in_progress")], "not completed"),
        ([SimpleNamespace(id="turn-1", status="completed", items_view="summary")], "incomplete"),
    )
    for history, message in cases:
        with pytest.raises(adapter.HarnessError, match=message):
            adapter._select_replayed_completed_turn(history, receipt=receipt)


def test_codex_unreceipted_history_requires_one_matching_completed_turn() -> None:
    """无 receipt 恢复只接受一个同 prompt、已完成且可重放的历史 turn。"""

    def history_thread(turns):
        return SimpleNamespace(
            read=lambda include_turns=True: {
                "thread": {"id": "thread-1", "turns": turns},
            }
        )

    user_item = {"type": "userMessage", "content": [{"type": "text", "text": "prompt"}]}
    final_item = {"type": "agentMessage", "phase": "final_answer", "text": "answer"}
    complete = {
        "id": "turn-1",
        "status": "completed",
        "items_view": "full",
        "items": [user_item, final_item],
    }
    kwargs = {
        "expected_thread_id": "thread-1",
        "request_digest": "a" * 64,
        "prompt_text": "prompt",
        "turn_count_before": 0,
        "physical_turn_count_before": 0,
        "thread_policy_digest": "b" * 64,
    }
    assert adapter._find_unreceipted_completed_turn(
        history_thread([]), **kwargs
    ) is None
    recovered = adapter._find_unreceipted_completed_turn(
        history_thread([complete]), **kwargs
    )
    assert recovered == (
        {
            "schema_version": adapter._RECEIPT_SCHEMA_VERSION,
            "thread_id": "thread-1",
            "turn_id": "turn-1",
            "request_digest": "a" * 64,
            "turn_count_before": 0,
            "thread_policy_digest": "b" * 64,
        },
        "answer",
    )
    failed = {
        "id": "turn-failed",
        "status": "failed",
        "items_view": "full",
        "items": [user_item],
    }
    assert adapter._find_unreceipted_completed_turn(
        history_thread([failed, complete]),
        **{**kwargs, "physical_turn_count_before": 1},
    ) == recovered
    with pytest.raises(adapter.HarnessError, match="unaccounted physical"):
        adapter._find_unreceipted_completed_turn(
            history_thread([failed, complete]),
            **kwargs,
        )
    error_cases = (
        ([complete, complete], "unaccounted"),
        ([{"status": "completed", "items": []}], "no id"),
        ([{"id": "turn-1", "status": "in_progress", "items": []}], "unfinished"),
        (
            [
                {
                    **complete,
                    "items": [
                        {"type": "userMessage", "content": [{"type": "text", "text": "other"}]},
                        final_item,
                    ],
                }
            ],
            "different request",
        ),
    )
    for turns, message in error_cases:
        with pytest.raises(adapter.HarnessError, match=message):
            adapter._find_unreceipted_completed_turn(history_thread(turns), **kwargs)


def test_sdk_adapter_redacts_secret_from_exception_chain(monkeypatch, tmp_path: Path) -> None:
    """app-server/turn 异常链含秘密时不能向调用方回显原始正文。"""

    class SecretTurn:
        def stream(self):
            try:
                raise ValueError("MCP secret=synthetic-mcp-token")
            except ValueError as nested:
                raise RuntimeError("stream failed") from nested

    class SecretThread(_FakeThread):
        def turn(self, *_args, **_kwargs):
            return SecretTurn()

    class SecretCodex(_FakeCodex):
        def __init__(self, _config):
            self.thread = SecretThread()

    monkeypatch.setattr(adapter, "Codex", SecretCodex)
    request = HarnessRequest(
        messages=({"role": "user", "content": "probe"},),
        model="codex-model",
        role="task_review",
        stage="workflow_revision",
        profile="review",
        workspace=tmp_path / "workspace",
    )

    with pytest.raises(adapter.HarnessError) as failure:
        adapter.run_codex_sdk_turn(
            config=CodexConfig(),
            request=request,
            thread_config={},
            session_id=None,
            trace_path=tmp_path / "trace.jsonl",
            secret_environment={"MCP_SECRET": "synthetic-mcp-token"},
        )
    current: BaseException | None = failure.value
    while current is not None:
        assert "synthetic-mcp-token" not in str(current)
        current = current.__cause__ or current.__context__


def test_sdk_adapter_redacts_secret_from_startup_exception_context(monkeypatch, tmp_path: Path) -> None:
    """app-server 启动异常在脱敏后不能通过隐式 context 重新泄漏。"""

    class StartupFailure:
        def __init__(self, _config):
            raise RuntimeError("startup failed: synthetic-provider-token")

    monkeypatch.setattr(adapter, "Codex", StartupFailure)
    request = HarnessRequest(
        messages=({"role": "user", "content": "probe"},),
        model="codex-model",
        role="task_review",
        stage="workflow_revision",
        profile="review",
        workspace=tmp_path / "workspace",
    )

    with pytest.raises(adapter.HarnessError) as failure:
        adapter.run_codex_sdk_turn(
            config=CodexConfig(),
            request=request,
            thread_config={},
            session_id=None,
            trace_path=tmp_path / "trace.jsonl",
            secret_environment={"PROVIDER_SECRET": "synthetic-provider-token"},
        )
    assert failure.value.__cause__ is None
    current: BaseException | None = failure.value
    while current is not None:
        assert "synthetic-provider-token" not in str(current)
        current = current.__cause__ or current.__context__


def test_sdk_adapter_preserves_only_safe_transport_diagnostic(monkeypatch) -> None:
    """startup EOF 的固定诊断可上送，但 stderr secret 和原始正文必须消失。"""

    error = TransportClosedError(
        "Codex process closed stdout: secret=synthetic-secret",
        diagnostic=TransportDiagnostic(
            phase=TransportPhase.INITIALIZE,
            stdout_eof=True,
            returncode=-15,
            termination_signal=15,
            stderr_indicator=StderrIndicator.SANDBOX,
        ),
    )
    sanitized = adapter._sanitize_sdk_error(
        "Codex SDK startup failed",
        error,
        ("synthetic-secret",),
        kind="transport",
    )

    assert isinstance(sanitized, adapter.HarnessError)
    assert "synthetic-secret" not in str(error)
    assert "synthetic-secret" not in str(sanitized)
    assert sanitized.transport_diagnostic == {
        "phase": "initialize",
        "stdout_eof": True,
        "returncode": -15,
        "termination_signal": 15,
        "stderr_indicator": "sandbox",
    }


@pytest.mark.parametrize(
    ("role", "stage", "profile", "metadata", "expected"),
    [
        ("investigation", "workflow", None, {}, adapter.CODEX_PROFILE_INVESTIGATION),
        ("planner", "workflow", None, {}, adapter.CODEX_PROFILE_LEGACY_PLANNER),
        ("review", "workflow", None, {}, adapter.CODEX_PROFILE_TASK_REVIEW),
        ("generic", "investigation_and_experiment", None, {}, adapter.CODEX_PROFILE_INVESTIGATION),
        ("generic", "workflow", "legacy", {}, adapter.CODEX_PROFILE_LEGACY_PLANNER),
        ("generic", "workflow", None, {"codex_profile": "review"}, adapter.CODEX_PROFILE_TASK_REVIEW),
    ],
)
def test_resolve_codex_profile_preserves_role_stage_and_explicit_overrides(
    tmp_path: Path,
    role: str,
    stage: str,
    profile: str | None,
    metadata: dict[str, str],
    expected: str,
) -> None:
    """profile 选择必须保留旧角色、调查阶段和显式覆盖的优先级。"""

    request = HarnessRequest(
        messages=({"role": "user", "content": "probe"},),
        model="codex-model",
        role=role,
        stage=stage,
        workspace=tmp_path / role,
        metadata=metadata,
        profile=profile,
    )

    assert adapter.resolve_codex_profile(request) == expected


def test_codex_tool_configuration_rejects_unknown_shape_and_accepts_boolean_flags() -> None:
    """工具配置只接受白名单名称、enabled 布尔值和空配置。"""

    adapter._validate_codex_tool_configuration(None)
    adapter._validate_codex_tool_configuration(
        {"update_plan": {"enabled": True}, "experimental_request_user_input": {}}
    )
    with pytest.raises(adapter.HarnessError, match="must be an object"):
        adapter._validate_codex_tool_configuration([])
    with pytest.raises(adapter.HarnessError, match="unsupported keys"):
        adapter._validate_codex_tool_configuration({"shell": {"enabled": True}})
    with pytest.raises(adapter.HarnessError, match="must be an object"):
        adapter._validate_codex_tool_configuration({"update_plan": True})
    with pytest.raises(adapter.HarnessError, match="unsupported fields"):
        adapter._validate_codex_tool_configuration({"update_plan": {"disabled": True}})
    with pytest.raises(adapter.HarnessError, match="must be boolean"):
        adapter._validate_codex_tool_configuration({"update_plan": {"enabled": "yes"}})


def test_usage_mapping_supports_sdk_breakdown_aliases_and_missing_values() -> None:
    """SDK token breakdown 缺失或类型错误时不伪造 usage 数字。"""

    assert adapter._usage_mapping(None) == {}
    assert adapter._usage_mapping(SimpleNamespace(last=None)) == {}
    usage = adapter._usage_mapping(
        SimpleNamespace(
            last=SimpleNamespace(
                input_tokens=3,
                output_tokens=4,
                total_tokens=7,
                cached_input_tokens=2,
                reasoning_output_tokens=1,
                cache_write_input_tokens=5,
            )
        )
    )
    assert usage == {
        "input_tokens": 3,
        "prompt_tokens": 3,
        "output_tokens": 4,
        "completion_tokens": 4,
        "total_tokens": 7,
        "cached_input_tokens": 2,
        "reasoning_output_tokens": 1,
        "cache_write_input_tokens": 5,
    }
    assert adapter._usage_mapping(
        SimpleNamespace(last=SimpleNamespace(input_tokens=True, output_tokens="bad"))
    ) == {}
