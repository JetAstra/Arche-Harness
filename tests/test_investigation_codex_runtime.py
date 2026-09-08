"""Investigation Codex SDK runtime 的无网络合同测试。"""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import pytest
from openai_codex.generated.v2_all import (
    McpToolCallResult,
    McpToolCallStatus,
    McpToolCallThreadItem,
)

from chemistry_multiagent.harness.base import HarnessError, HarnessRequest, HarnessResponse
from chemistry_multiagent.harness import codex_sdk_adapter
from chemistry_multiagent.harness.codex_sdk_adapter import (
    CodexCompletionRecoveryPending,
    CodexSdkRunResult,
    CodexTerminalTurnError,
    CodexWorkerRecoveryPending,
)
from chemistry_multiagent.harness.codex_auth import (
    CODEX_INVESTIGATION_PERMISSION_PROFILE,
)
from chemistry_multiagent.harness.session_store import CodexSessionOwner, CodexSessionStore
from chemistry_multiagent.investigation.codex_runtime import (
    CodexProviderSettings,
    EvidenceMcpSettings,
    InvestigationCodexRuntime,
    InvestigationMcpSettings,
    InvestigationTurnResult,
)
from chemistry_multiagent.investigation.async_jobs import (
    InvestigationJobController,
    InvestigationJobIdentity,
    InvestigationJobReceipt,
    InvestigationJobState,
    InvestigationWaitingJobStore,
)
from chemistry_multiagent.investigation.contracts import InvestigationResponse
from chemistry_multiagent.investigation.domain import InvestigationResult, InvestigationSession, InvestigationState
from chemistry_multiagent.investigation import codex_runtime as codex_runtime_module
from chemistry_multiagent.investigation.prompts import (
    build_initial_investigation_request,
    build_investigation_job_continuation_request,
    build_investigation_job_wake_request,
    build_repair_investigation_request,
    build_task_finalization_request,
    build_task_review_request,
)


def _investigation_response(*, combination_id: str = "combination-1", version: str = "1") -> str:
    """返回满足 InvestigationResponse 的最小 JSON。"""

    return json.dumps(
        {
            "self_judgment": "supported",
            "narrative": f"对 {combination_id} 版本 {version} 的实际计算回顾。",
            "job_refs": [f"job-{combination_id}-{version}"],
            "operation_refs": [f"operation-{combination_id}-{version}"],
            "artifact_refs": [f"artifact-{combination_id}-{version}"],
            "observations": [f"observation-{combination_id}-{version}"],
            "limitations": [],
        }
    )


def _review_response() -> str:
    """返回五个唯一组合的最小 Review JSON。"""

    return json.dumps(
        {
            "findings": [
                {
                    "hypothesis_execution_input_record_id": f"combination-{index}",
                    "validity": "pass",
                    "hypothesis_judgment": "supported",
                }
                for index in range(1, 6)
            ],
            "can_answer_original_question": True,
            "final_answer": "五项结果支持对原科学问题的回答。",
        }
    )


def _finalization_response() -> str:
    """返回覆盖五个组合的最小终局成稿 JSON。"""

    return json.dumps(
        {
            "findings": [
                {
                    "hypothesis_execution_input_record_id": f"combination-{index}",
                    "summary": f"组合 {index} 的现有证据摘要。",
                }
                for index in range(1, 6)
            ],
            "answerability": "answerable",
            "final_answer": "现有证据支持有限定的最终结论。",
        }
    )


def _result_mapping(index: int) -> dict[str, object]:
    """返回 Review prompt builder 所需的公开结果摘要。"""

    return {
        "research_task_id": "task-1",
        "investigation_context_id": "batch-1",
        "hypothesis_execution_input_record_id": f"combination-{index}",
        "result_version": "1",
        "self_judgment": "supported",
        "narrative": f"组合 {index} 的计算回顾。",
        "job_refs": [f"job-{index}"],
        "operation_refs": [f"operation-{index}"],
        "artifact_refs": [f"art_{index:032x}"],
        "observations": [f"observation-{index}"],
        "limitations": [],
    }


def _review_request(
    *,
    investigation_context_id: str = "batch-1",
    model: str = "review-model",
) -> dict[str, object]:
    """构造固定五项的 task Review 请求。"""

    results = [_result_mapping(index) for index in range(1, 6)]
    for result in results:
        result["investigation_context_id"] = investigation_context_id
    return build_task_review_request(
        "task-1",
        investigation_context_id,
        "原始科学问题",
        results,
        {
            "snapshot_id": f"snapshot-{investigation_context_id}",
            "research_task_id": "task-1",
            "investigation_context_id": investigation_context_id,
            "result_pairs": [
                {
                    "hypothesis_execution_input_record_id": f"combination-{index}",
                    "result_version": "1",
                    "artifacts": [
                        {
                            "artifact_id": f"evidence_{index:032x}",
                            "source_artifact_ref": f"art_{index:032x}",
                            "size": 1,
                            "sha256": f"{index:064x}",
                            "media_type": "application/json",
                        }
                    ],
                }
                for index in range(1, 6)
            ],
        },
        model=model,
    )


def _finalization_request(*, model: str = "review-model") -> dict[str, object]:
    """构造固定五项的预算耗尽终局成稿请求。"""

    results = [_result_mapping(index) for index in range(1, 6)]
    return build_task_finalization_request(
        "task-1",
        "batch-1",
        "原始科学问题",
        results,
        {
            "snapshot_id": "snapshot-batch-1",
            "research_task_id": "task-1",
            "investigation_context_id": "batch-1",
            "result_pairs": [
                {
                    "hypothesis_execution_input_record_id": f"combination-{index}",
                    "result_version": "1",
                    "artifacts": [
                        {
                            "artifact_id": f"evidence_{index:032x}",
                            "source_artifact_ref": f"art_{index:032x}",
                            "size": 1,
                            "sha256": f"{index:064x}",
                            "media_type": "application/json",
                        }
                    ],
                }
                for index in range(1, 6)
            ],
        },
        model=model,
    )


def _pydantic_mcp_job_item(
    tool: str,
    job_id: str,
    *,
    result_job_id: str | None = None,
) -> McpToolCallThreadItem:
    """构造 openai-codex Pydantic MCP completed item 及 compact collect 结果。"""

    returned_job_id = result_job_id or job_id
    if tool == "submit_action_job":
        payload = {
            "status": "success",
            "job_id": returned_job_id,
            "operation_id": "operation-pydantic",
            "submission_key": "submission-pydantic",
            "action_id": "search_transition_state_path",
            "backend_id": "transition_state_path_search",
        }
        arguments = {"request": {"submission_key": "submission-pydantic"}}
    elif tool == "get_execution_job":
        payload = {
            "status": "success",
            "job_id": returned_job_id,
            "job_status": "succeeded",
            "terminal": True,
            "job": {"job_id": returned_job_id, "status": "succeeded"},
        }
        arguments = {"job_id": job_id}
    else:
        payload = {
            "status": "success",
            "ready": True,
            "job_id": returned_job_id,
            "job_status": "succeeded",
            "action_result": {
                "status": "success",
                "result": {
                    "hessian": {
                        "inline": False,
                        "content": "hessian",
                        "shape": [159, 159],
                    }
                },
            },
        }
        arguments = {"job_id": job_id}
    result = McpToolCallResult(
        content=[SimpleNamespace(text=json.dumps(payload))],
        structuredContent=None,
    )
    return McpToolCallThreadItem(
        id=f"item-{tool}-{job_id}",
        server="minichem_toolbox",
        tool=tool,
        arguments=arguments,
        result=result,
        status=McpToolCallStatus.completed,
        type="mcpToolCall",
    )


def _runtime(
    tmp_path: Path,
    fake_adapter,
    *,
    evidence: bool = False,
    model: str = "investigation-model",
    reasoning_effort: str | None = None,
    investigation_compact_prompt: str | None = None,
    investigation_model_context_window: int | None = None,
    allow_test_doubles: bool = True,
    on_turn_started=None,
    before_new_investigation_turn=None,
    sdk_idle_timeout_seconds: float | None = None,
) -> InvestigationCodexRuntime:
    """输入临时目录、adapter 和可选生命周期/fence 回调，构造单角色 runtime。

    动作是创建本地 runtime 配置和隔离会话根，不启动模型；返回可注入 fake adapter
    的 ``InvestigationCodexRuntime``。参数类型或 MCP 角色不匹配时由 runtime 构造器
    抛出异常。
    """

    environment = {
        "ARCHE_CODEX_API_KEY": "investigation-secret",
        "ARCHE_CODEX_BASE_URL": "https://investigation.example/v1",
        "ARCHE_CODEX_MODEL_NAME": model,
        "ARCHE_REVIEW_CODEX_API_KEY": "review-secret",
        "ARCHE_REVIEW_CODEX_BASE_URL": "https://review.example/v1",
        "ARCHE_REVIEW_CODEX_MODEL_NAME": "review-model",
        "MINICHEM_CAPABILITY_TOKEN": "capability-secret",
        "EVIDENCE_AUTH_TOKEN": "Bearer evidence-token",
        "EVIDENCE_SCOPE_TOKEN": "evidence-scope-secret",
    }
    if reasoning_effort is not None:
        environment["ARCHE_CODEX_REASONING_EFFORT"] = reasoning_effort
    return InvestigationCodexRuntime(
        session_root=tmp_path / "sessions",
        trace_root=tmp_path / "traces",
        role="task_review" if evidence else "investigation",
        minichem=(
            InvestigationMcpSettings(
                "http://127.0.0.1:9011/mcp",
                "MINICHEM_CAPABILITY_TOKEN",
            )
            if not evidence
            else None
        ),
        evidence_mcp=(
            EvidenceMcpSettings(
                "http://127.0.0.1:9020/mcp",
                {
                    "Authorization": "EVIDENCE_AUTH_TOKEN",
                    "X-Evidence-Scope": "EVIDENCE_SCOPE_TOKEN",
                },
            )
            if evidence
            else None
        ),
        investigation_compact_prompt=investigation_compact_prompt,
        investigation_model_context_window=investigation_model_context_window,
        environment=environment,
        adapter=fake_adapter,
        allow_test_doubles=allow_test_doubles,
        on_turn_started=on_turn_started,
        before_new_investigation_turn=before_new_investigation_turn,
        sdk_idle_timeout_seconds=sdk_idle_timeout_seconds,
    )


def _write_test_completed_receipt(
    kwargs: Mapping[str, object],
    *,
    thread_id: str,
    turn_id: str,
) -> None:
    """为成功测试 turn 写入与 runtime 入参完全一致的 completed receipt。"""

    codex_sdk_adapter._write_completed_turn_receipt(
        kwargs["completed_turn_receipt_root"],
        thread_id=thread_id,
        turn_id=turn_id,
        request_digest=kwargs["request_digest"],
        turn_count_before=kwargs["turn_count_before"],
        thread_policy_digest=kwargs["thread_policy_digest"],
    )


class _FakeAdapter:
    """只记录 adapter 入参并模拟 thread-ready 与 completed turn。"""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.next_thread = 1
        self.return_drift = False
        self.invalid_response = False
        self.response_content: str | None = None
        self.fail_first = False
        # 生产 runtime 对每个成功 turn 都要求 completed receipt；测试替身也默认
        # 写入真实回执，只有显式覆盖时才可模拟失败路径。
        self.write_receipt = True
        self.before_turn_manifests: list[dict[str, object]] = []
        self.emit_turn_started = False
        self.started_turn_ids: list[str] = []

    def __call__(self, **kwargs):
        """输入 runtime adapter 参数并按测试开关生成 thread、trace 和响应。

        动作会记录调用，通知 thread/turn 生命周期，并可选模拟身份漂移、失败或
        receipt 写入；副作用仅位于测试临时目录，不访问真实 provider。成功返回
        ``CodexSdkRunResult``；开关要求失败或回调/receipt 合同非法时抛出异常。
        """

        self.calls.append(dict(kwargs))
        request = kwargs["request"]
        session_id = kwargs["session_id"]
        thread_id = session_id or f"thread-{self.next_thread}"
        self.next_thread += 1
        if self.return_drift and session_id is not None:
            thread_id = "drifted-thread"
        kwargs["on_thread_ready"](thread_id)
        if self.emit_turn_started and "on_turn_started" in kwargs:
            started_turn_id = f"turn-{len(self.calls)}"
            self.started_turn_ids.append(started_turn_id)
            kwargs["on_turn_started"](started_turn_id)
        trace_path = kwargs["trace_path"]
        trace_values = tuple(kwargs.get("trace_sensitive_values", ()))
        trace_path.write_text(
            f"notification-{len(self.calls)} " + " ".join(trace_values),
            encoding="utf-8",
        )
        if self.fail_first:
            self.fail_first = False
            raise HarnessError("mock SDK turn failed", kind="transport")
        if self.write_receipt:
            _write_test_completed_receipt(
                kwargs,
                thread_id=thread_id,
                turn_id=f"turn-{len(self.calls)}",
            )
        content = self.response_content or (
            '{"invalid":true}' if self.invalid_response else (
                _review_response() if request.role == "task_review" else _investigation_response(
                    combination_id=str(
                        request.metadata.get(
                            "hypothesis_execution_input_record_id",
                            "combination-1",
                        )
                    )
                )
            )
        )
        return CodexSdkRunResult(
            response=HarnessResponse(
                content=content,
                model=request.model,
                metadata={"turn_id": f"turn-{len(self.calls)}"},
            ),
            thread_id=thread_id,
            turn_id=f"turn-{len(self.calls)}",
        )


class _McpCompletedSubmitAdapter:
    """发出真实形状 ``mcpToolCall/completed`` 事件的最小 adapter。"""

    def __init__(self, sidecar_probe) -> None:
        self.sidecar_probe = sidecar_probe
        self.sidecar_visible_before_return = False
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs):
        """先发 thread/turn 和 submit 完成事件，再返回伪造正文。"""

        self.calls.append(dict(kwargs))
        request = kwargs["request"]
        thread_id = kwargs["session_id"] or "thread-accepted"
        kwargs["on_thread_ready"](thread_id)
        if "on_turn_started" in kwargs:
            kwargs["on_turn_started"]("turn-accepted")
        kwargs["on_tool_completed"](
            {
                "type": "mcpToolCall",
                "status": "completed",
                "tool": "submit_transition_state_path_job",
                "arguments": {"request": "real-submit-payload"},
                "result": {
                    "structuredContent": {
                        "job_id": "job-accepted",
                        "operation_id": "operation-accepted",
                        "submission_key": "submission-accepted",
                        "action_id": "action-accepted",
                        "backend_id": "backend-accepted",
                    }
                },
            }
        )
        self.sidecar_visible_before_return = bool(self.sidecar_probe())
        _write_test_completed_receipt(
            kwargs,
            thread_id=thread_id,
            turn_id="turn-accepted",
        )
        return CodexSdkRunResult(
            response=HarnessResponse(
                # 该正文故意使用另一套身份；等待状态只能来自工具回执。
                content='{"job_id":"assistant-forged","operation_id":"assistant-forged"}',
                model=request.model,
                metadata={"turn_id": "turn-accepted"},
            ),
            thread_id=thread_id,
            turn_id="turn-accepted",
        )


class _ScriptedAsyncInvestigationAdapter:
    """按脚本发出首 job、最终正文或 wake 后 next job 的真实事件。"""

    def __init__(self, phases: tuple[str, ...]) -> None:
        self.phases = list(phases)
        self.calls: list[dict[str, object]] = []

    @staticmethod
    def _submit_item(job_number: int) -> dict[str, object]:
        """构造一个 MCP submit completed item。"""

        suffix = str(job_number)
        return {
            "type": "mcpToolCall",
            "status": "completed",
            "tool": "submit_transition_state_path_job",
            "arguments": {"request": f"scripted-submit-{suffix}"},
            "result": {
                "structuredContent": {
                    "job_id": f"job-{suffix}",
                    "operation_id": f"operation-{suffix}",
                    "submission_key": f"submission-{suffix}",
                    "action_id": "action-scripted",
                    "backend_id": "backend-scripted",
                }
            },
        }

    @staticmethod
    def _progress_item(
        tool: str,
        job_id: str,
        *,
        terminal: bool = True,
    ) -> dict[str, object]:
        """构造 wake turn 中原 job 的 terminal get/collect 事件。"""

        if tool.startswith("get_"):
            payload = {
                "status": "success",
                "job_id": job_id,
                "job_status": "succeeded" if terminal else "running",
                "job": {
                    "job_id": job_id,
                    "status": "succeeded" if terminal else "running",
                },
            }
        else:
            payload = {
                "status": "success",
                "ready": True,
                "job_id": job_id,
                "job_status": "succeeded",
                "action_result": {"job_id": job_id, "status": "succeeded"},
            }
        return {
            "type": "mcpToolCall",
            "status": "completed",
            "tool": tool,
            "arguments": {"job_id": job_id},
            "result": {"structuredContent": payload},
        }

    def __call__(self, **kwargs):
        """在同一 session thread 上执行一段真实 tool/正文序列。"""

        self.calls.append(dict(kwargs))
        phase = self.phases.pop(0)
        request = kwargs["request"]
        thread_id = kwargs["session_id"] or "thread-scripted"
        turn_id = f"turn-scripted-{len(self.calls)}"
        kwargs["on_thread_ready"](thread_id)
        if "on_turn_started" in kwargs:
            kwargs["on_turn_started"](turn_id)
        if phase.startswith("submit"):
            if phase == "submit-2":
                wake_payload = json.loads(request.messages[-1]["content"])
                old_job_id = wake_payload["job_id"]
                kwargs["on_tool_completed"](
                    self._progress_item("get_transition_state_path_job", old_job_id)
                )
                kwargs["on_tool_completed"](
                    self._progress_item("collect_transition_state_path_job", old_job_id)
                )
            job_number = 1 if phase == "submit-1" else 2
            kwargs["on_tool_completed"](self._submit_item(job_number))
            content = '{"job_id":"assistant-text-must-not-drive-waiting"}'
        elif phase == "running":
            wake_payload = json.loads(request.messages[-1]["content"])
            old_job_id = wake_payload["job_id"]
            kwargs["on_tool_completed"](
                self._progress_item(
                    "get_transition_state_path_job",
                    old_job_id,
                    terminal=False,
                )
            )
            content = '{"job_id":"assistant-text-must-not-drive-waiting"}'
        else:
            wake_payload = json.loads(request.messages[-1]["content"])
            old_job_id = wake_payload["job_id"]
            kwargs["on_tool_completed"](
                self._progress_item("get_transition_state_path_job", old_job_id)
            )
            kwargs["on_tool_completed"](
                self._progress_item("collect_transition_state_path_job", old_job_id)
            )
            content = _investigation_response()
        _write_test_completed_receipt(
            kwargs,
            thread_id=thread_id,
            turn_id=turn_id,
        )
        return CodexSdkRunResult(
            response=HarnessResponse(
                content=content,
                model=request.model,
                metadata={"turn_id": turn_id},
            ),
            thread_id=thread_id,
            turn_id=turn_id,
        )


class _SameTurnSequentialJobAdapter:
    """在一个真实 adapter turn 中模拟两个顺序 job 的 submit/get/collect。"""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.events: list[tuple[str, str, str | float]] = []

    @staticmethod
    def _item(tool: str, job_id: str, *, status: str = "succeeded") -> dict[str, object]:
        """构造带 job 身份和状态的真实 MCP completed item。"""

        if tool.startswith("submit_"):
            payload = {
                "job_id": job_id,
                "operation_id": f"operation-{job_id}",
                "submission_key": f"submission-{job_id}",
                "action_id": "action-scripted",
                "backend_id": "backend-scripted",
            }
            arguments = {"request": f"request-{job_id}"}
        elif tool.startswith("get_"):
            payload = {
                # TS get 的外层 status=success 只是 MCP 调用成功；job_status/job.status
                # 才是 queued/running/success 等真实作业状态。
                "status": "success",
                "job_status": status,
                "job_id": job_id,
                "job": {"job_id": job_id, "status": status},
            }
            arguments = {"job_id": job_id}
        else:
            payload = {
                "status": "success",
                "ready": True,
                "job_id": job_id,
                "job_status": status,
                "action_result": {"job_id": job_id, "status": status},
            }
            arguments = {"job_id": job_id}
        return {
            "type": "mcpToolCall",
            "status": "completed",
            "tool": tool,
            "arguments": arguments,
            "result": {"structuredContent": payload},
        }

    def __call__(self, **kwargs):
        """发出两个顺序 job，等待由可注入的 50 秒等价事件表示。"""

        self.calls.append(dict(kwargs))
        request = kwargs["request"]
        thread_id = kwargs["session_id"] or "thread-same-turn"
        turn_id = "turn-same-turn"
        kwargs["on_thread_ready"](thread_id)
        if "on_turn_started" in kwargs:
            kwargs["on_turn_started"](turn_id)
        completed_items: list[dict[str, object]] = []
        for job_id in ("job-1", "job-2"):
            submit_tool = "submit_transition_state_path_job"
            get_tool = "get_transition_state_path_job"
            collect_tool = "collect_transition_state_path_job"
            submit = self._item(submit_tool, job_id)
            kwargs["on_tool_completed"](submit)
            completed_items.append(submit)
            self.events.append(("submit", job_id, len(self.events)))
            self.events.append(("wait", job_id, 50.0))
            running = self._item(get_tool, job_id, status="running")
            kwargs["on_tool_completed"](running)
            completed_items.append(running)
            self.events.append(("get", job_id, "running"))
            terminal = self._item(get_tool, job_id, status="succeeded")
            kwargs["on_tool_completed"](terminal)
            completed_items.append(terminal)
            self.events.append(("get", job_id, "succeeded"))
            collect = self._item(collect_tool, job_id)
            kwargs["on_tool_completed"](collect)
            completed_items.append(collect)
            self.events.append(("collect", job_id, "succeeded"))
        _write_test_completed_receipt(
            kwargs,
            thread_id=thread_id,
            turn_id=turn_id,
        )
        return CodexSdkRunResult(
            response=HarnessResponse(
                content=_investigation_response(),
                model=request.model,
                metadata={"turn_id": turn_id},
            ),
            thread_id=thread_id,
            turn_id=turn_id,
            completed_tool_items=tuple(completed_items),
        )


@pytest.mark.parametrize(
    ("family", "submit_tool", "status_tool", "collect_tool", "identity", "terminal_status"),
    [
        (
            "transition_state_path",
            "submit_transition_state_path_job",
            "get_transition_state_path_job",
            "collect_transition_state_path_job",
            {
                "job_id": "job-ts",
                "operation_id": "operation-ts",
                "submission_key": "submission-ts",
                "action_id": "action-ts",
                "backend_id": "backend-ts",
            },
            "success",
        ),
        (
            "native_execution",
            "submit_native_job",
            "get_execution_job",
            "collect_execution_job",
            {
                "job_id": "job-native",
                "operation_id": "operation-native",
                "submission_key": "submission-native",
                "job_type": "gaussian",
                "runtime": "gaussian",
                "software_id": "gaussian",
            },
            "completed",
        ),
    ],
)
def test_job_progress_uses_real_job_status_and_rejects_transport_success_only(
    family: str,
    submit_tool: str,
    status_tool: str,
    collect_tool: str,
    identity: dict[str, str],
    terminal_status: str,
) -> None:
    """TS/native 的真实状态可归一化，裸 MCP status=success 不能冒充终态或 collect。"""

    callbacks = codex_runtime_module._CodexTurnCallbacks(
        store=SimpleNamespace(),
        manifest=None,
        session_id=None,
        stage="initial_investigation",
        request_digest="request-digest",
        thread_policy_digest="thread-policy",
    )
    submit = {
        "type": "mcpToolCall",
        "status": "completed",
        "tool": submit_tool,
        "arguments": {"request": f"request-{family}"},
        "result": {"structuredContent": identity},
    }
    job_id = identity["job_id"]
    callbacks.capture_tool_completed(submit)
    callbacks.capture_tool_completed(
        {
            "type": "mcpToolCall",
            "status": "completed",
            "tool": status_tool,
            "arguments": {"job_id": job_id},
            "result": {
                "structuredContent": {
                    "status": "success",
                    "job_id": job_id,
                    "job_status": "running",
                    "job": {"job_id": job_id, "status": "running"},
                }
            },
        }
    )
    assert callbacks.active_job_terminal is False
    # 当前 scope 内的其它 job 由 MCP registry 鉴权；本地 active tracker
    # 不应把它们的只读回执误判为当前作业身份冲突。
    callbacks.capture_tool_completed(
        {
            "type": "mcpToolCall",
            "status": "completed",
            "tool": collect_tool,
            "arguments": {"job_id": "job-other"},
            "result": {
                "structuredContent": {
                    "status": "success",
                    "job_id": "job-other",
                    "ready": True,
                }
            },
        }
    )
    assert callbacks.active_job_terminal is False
    with pytest.raises(HarnessError, match="collect 前必须"):
        callbacks.capture_tool_completed(
            {
                "type": "mcpToolCall",
                "status": "completed",
                "tool": collect_tool,
                "arguments": {"job_id": job_id},
                "result": {"structuredContent": {"status": "success", "job_id": job_id}},
            }
        )

    callbacks.capture_tool_completed(
        {
            "type": "mcpToolCall",
            "status": "completed",
            "tool": status_tool,
            "arguments": {"job_id": job_id},
            "result": {
                "structuredContent": {
                    "status": "success" if family == "transition_state_path" else terminal_status,
                    "job_id": job_id,
                    "job_status": terminal_status,
                    "job": {"job_id": job_id, "status": terminal_status},
                }
            },
        }
    )
    with pytest.raises(HarnessError, match="ready、终态或完整结果"):
        callbacks.capture_tool_completed(
            {
                "type": "mcpToolCall",
                "status": "completed",
                "tool": collect_tool,
                "arguments": {"job_id": job_id},
                "result": {"structuredContent": {"status": "success", "job_id": job_id}},
            }
        )
    callbacks.capture_tool_completed(
        {
            "type": "mcpToolCall",
            "status": "completed",
            "tool": collect_tool,
            "arguments": {"job_id": job_id},
            "result": {
                "structuredContent": {
                    "status": "success",
                    "job_id": job_id,
                    "job_status": terminal_status,
                    "ready": True,
                }
            },
        }
    )
    assert callbacks.active_job_collected is True
    assert len(callbacks.completed_job_receipts) == 1


def test_job_progress_requires_get_even_when_submit_returns_terminal_failure() -> None:
    """submit 已返回失败终态时，collect 仍必须先经过同一 job 的 get。"""

    job_id = "job-submit-terminal-failure"
    callbacks = codex_runtime_module._CodexTurnCallbacks(
        store=SimpleNamespace(),
        manifest=None,
        session_id=None,
        stage="initial_investigation",
        request_digest="request-digest",
        thread_policy_digest="thread-policy",
    )
    callbacks.capture_tool_completed(
        {
            "type": "mcpToolCall",
            "status": "completed",
            "tool": "submit_action_job",
            "arguments": {"request": "terminal-failure"},
            "result": {
                "structuredContent": {
                    "status": "success",
                    "job_id": job_id,
                    "operation_id": "operation-submit-terminal-failure",
                    "submission_key": "submission-submit-terminal-failure",
                    "action_id": "scan_reaction_coordinates",
                    "backend_id": "pysisyphus",
                    "job_type": "action_execution",
                    "runtime": "reaction",
                    "job_status": "failed",
                    "terminal": True,
                    "job": {"job_id": job_id, "status": "failed"},
                }
            },
        }
    )

    with pytest.raises(HarnessError, match="collect 前必须"):
        callbacks.capture_tool_completed(
            {
                "type": "mcpToolCall",
                "status": "completed",
                "tool": "collect_execution_job",
                "arguments": {"job_id": job_id},
                "result": {
                    "structuredContent": {
                        "status": "success",
                        "ready": True,
                        "job_id": job_id,
                        "job_status": "failed",
                    }
                },
            }
        )

    assert callbacks.active_job_terminal is False
    assert callbacks.active_job_collected is False

    callbacks.capture_tool_completed(
        {
            "type": "mcpToolCall",
            "status": "completed",
            "tool": "get_execution_job",
            "arguments": {"job_id": job_id},
            "result": {
                "structuredContent": {
                    "status": "success",
                    "job_id": job_id,
                    "job_status": "failed",
                    "terminal": True,
                    "job": {"job_id": job_id, "status": "failed"},
                }
            },
        }
    )
    callbacks.capture_tool_completed(
        {
            "type": "mcpToolCall",
            "status": "completed",
            "tool": "collect_execution_job",
            "arguments": {"job_id": job_id},
            "result": {
                "structuredContent": {
                    "status": "success",
                    "ready": True,
                    "job_id": job_id,
                    "job_status": "failed",
                }
            },
        }
    )

    assert callbacks.active_job_terminal is True
    assert callbacks.active_job_collected is True


def test_job_progress_accepts_pydantic_mcp_result_with_compact_collect_payload() -> None:
    """真实 SDK/Pydantic MCP result 对象的 compact collect 回执可被安全解析。"""

    callbacks = codex_runtime_module._CodexTurnCallbacks(
        store=SimpleNamespace(),
        manifest=None,
        session_id=None,
        stage="initial_investigation",
        request_digest="request-digest",
        thread_policy_digest="thread-policy",
    )
    job_id = "job-pydantic-compact"
    callbacks.capture_tool_completed(_pydantic_mcp_job_item("submit_action_job", job_id))
    callbacks.capture_tool_completed(_pydantic_mcp_job_item("get_execution_job", job_id))
    callbacks.capture_tool_completed(_pydantic_mcp_job_item("collect_execution_job", job_id))

    assert callbacks.active_job_terminal is True
    assert callbacks.active_job_collected is True
    assert [receipt.identity.job_id for receipt in callbacks.completed_job_receipts] == [job_id]


def test_job_progress_rejects_pydantic_mcp_result_job_id_mismatch() -> None:
    """Pydantic MCP result 中与请求不一致的 job_id 仍必须 fail-closed。"""

    callbacks = codex_runtime_module._CodexTurnCallbacks(
        store=SimpleNamespace(),
        manifest=None,
        session_id=None,
        stage="initial_investigation",
        request_digest="request-digest",
        thread_policy_digest="thread-policy",
    )
    job_id = "job-pydantic-mismatch"
    callbacks.capture_tool_completed(_pydantic_mcp_job_item("submit_action_job", job_id))
    callbacks.capture_tool_completed(_pydantic_mcp_job_item("get_execution_job", job_id))

    with pytest.raises(HarnessError, match="没有匹配当前 active job_id"):
        callbacks.capture_tool_completed(
            _pydantic_mcp_job_item(
                "collect_execution_job",
                job_id,
                result_job_id="job-pydantic-other",
            )
        )

    assert callbacks.active_job_collected is False
    assert callbacks.completed_job_receipts == []


def test_job_progress_keeps_active_job_after_failed_get_tool_call() -> None:
    """get 工具调用失败且没有回执身份时保留 active job，允许随后重试同一作业。"""

    job_id = "job-failed-get-retry"
    identity = InvestigationJobIdentity(
        job_id=job_id,
        operation_id="operation-failed-get-retry",
        submission_key="submission-failed-get-retry",
        action_id="locate_transition_state",
        backend_id="gaussian",
        job_family="action_execution",
        job_type="action_execution",
        runtime="gaussian",
        software_id="gaussian",
    )
    callbacks = codex_runtime_module._CodexTurnCallbacks(
        store=SimpleNamespace(),
        manifest=None,
        session_id=None,
        stage="initial_investigation",
        request_digest="request-digest",
        thread_policy_digest="thread-policy",
        active_job_receipt=InvestigationJobReceipt(
            tool_name="submit_action_job",
            identity=identity,
            arguments={"request": "submitted-before-failed-get"},
            result=identity.as_mapping(),
        ),
    )

    callbacks.capture_tool_completed(
        {
            "type": "mcpToolCall",
            "status": "failed",
            "tool": "get_execution_job",
            "arguments": {"job_id": job_id},
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": "工具参数校验失败，未执行作业查询",
                    }
                ],
                "structuredContent": None,
            },
        }
    )

    assert callbacks.active_job_receipt is not None
    assert callbacks.active_job_receipt.identity.job_id == job_id
    assert callbacks.active_job_terminal is False
    assert callbacks.active_job_collected is False
    assert callbacks.completed_job_receipts == []
    with pytest.raises(HarnessError, match="不能并行提交第二个任务"):
        callbacks.capture_tool_completed(
            _SameTurnSequentialJobAdapter._item(
                "submit_transition_state_path_job",
                "job-after-failed-get",
            )
        )

    callbacks.capture_tool_completed(
        _SameTurnSequentialJobAdapter._item(
            "get_execution_job",
            job_id,
            status="succeeded",
        )
    )
    callbacks.capture_tool_completed(
        _SameTurnSequentialJobAdapter._item("collect_execution_job", job_id)
    )

    assert callbacks.active_job_terminal is True
    assert callbacks.active_job_collected is True
    assert [
        receipt.identity.job_id for receipt in callbacks.completed_job_receipts
    ] == [job_id]


def test_payload_candidates_deduplicate_self_referential_protocol_object() -> None:
    """协议对象通过受控 content 字段自引用时必须停止展开。"""

    loop = SimpleNamespace(content=[])
    loop.content.append(loop)

    assert codex_runtime_module._CodexTurnCallbacks._payload_candidates(loop) == ()


def test_resumed_new_turn_rearms_get_before_collect_for_action_job() -> None:
    """恢复到新 turn 时，旧 turn 的终态 get 不能替代当前 turn 的 get。"""

    job_id = "job_7e4c3d01"
    submit_item = {
        "type": "mcpToolCall",
        "status": "completed",
        "tool": "submit_action_job",
        "arguments": {"request": "scope-submission-4561cda431e893d0f5df3e457dcdaf5"},
        "result": {
            "structuredContent": {
                "status": "success",
                "job_id": job_id,
                "operation_id": "scope-submission-4561cda431e893d0f5df3e457dcdaf6",
                "submission_key": "case1-gaussian-tsopt-endpoint_pair_0001-v1",
                "action_id": "locate_transition_state",
                "backend_id": "gaussian",
                "job_type": "action_execution",
                "runtime": "gaussian",
                "software_id": "gaussian",
            }
        },
    }
    terminal_get_item = {
        "type": "mcpToolCall",
        "status": "completed",
        "tool": "get_execution_job",
        "arguments": {"job_id": job_id},
        "result": {
            "structuredContent": {
                "status": "success",
                "job_id": job_id,
                "job_status": "succeeded",
                "terminal": True,
                "job": {"job_id": job_id, "status": "succeeded"},
            }
        },
    }
    collect_item = {
        "type": "mcpToolCall",
        "status": "completed",
        "tool": "collect_execution_job",
        "arguments": {"job_id": job_id},
        "result": {
            "structuredContent": {
                "status": "success",
                "ready": True,
                "job_id": job_id,
                "job_status": "succeeded",
                "action_result": {"job_id": job_id, "status": "succeeded"},
            }
        },
    }

    def new_callbacks() -> codex_runtime_module._CodexTurnCallbacks:
        """建立一个不继承前一 turn 终态布尔值的恢复状态记录器。"""

        return codex_runtime_module._CodexTurnCallbacks(
            store=SimpleNamespace(),
            manifest=None,
            session_id=None,
            stage="investigation_job_continuation",
            request_digest="request-digest",
            thread_policy_digest="thread-policy",
        )

    original_turn = new_callbacks()
    original_turn.capture_tool_completed(submit_item)
    original_turn.capture_tool_completed(terminal_get_item)
    assert original_turn.active_job_terminal is True
    assert original_turn.active_job_receipt is not None

    resumed_turn = new_callbacks()
    resumed_turn.active_job_receipt = original_turn.active_job_receipt
    with pytest.raises(HarnessError, match="collect 前必须"):
        resumed_turn.capture_tool_completed(collect_item)
    assert resumed_turn.active_job_terminal is False
    assert resumed_turn.active_job_collected is False

    resumed_turn.capture_tool_completed(terminal_get_item)
    resumed_turn.capture_tool_completed(collect_item)
    assert resumed_turn.active_job_terminal is True
    assert resumed_turn.active_job_collected is True
    assert [receipt.identity.job_id for receipt in resumed_turn.completed_job_receipts] == [
        job_id
    ]


def test_list_inventory_can_arm_current_action_job_terminal_evidence() -> None:
    """包含历史条目的完整 scope 清单可提供当前 turn 证据，但不能代替 collect。"""

    job_id = "job-inventory-terminal"
    operation_id = "operation-inventory-terminal"
    submission_key = "submission-inventory-terminal"
    callbacks = codex_runtime_module._CodexTurnCallbacks(
        store=SimpleNamespace(),
        manifest=None,
        session_id=None,
        stage="investigation_job_continuation",
        request_digest="request-digest",
        thread_policy_digest="thread-policy",
    )
    callbacks.capture_tool_completed(
        {
            "type": "mcpToolCall",
            "status": "completed",
            "tool": "submit_action_job",
            "arguments": {"request": "inventory-terminal"},
            "result": {
                "structuredContent": {
                    "job_id": job_id,
                    "operation_id": operation_id,
                    "submission_key": submission_key,
                    "action_id": "locate_transition_state",
                    "backend_id": "gaussian",
                    "job_type": "action_execution",
                    "runtime": "gaussian",
                    "software_id": "gaussian",
                }
            },
        }
    )
    inventory_payload = {
        "status": "success",
        "jobs": [
            {
                "job_id": "job-history-ts",
                "job_status": "success",
                "phase": "success",
                "terminal": True,
                "operation_id": "operation-history-ts",
                "submission_key": "submission-history-ts",
            },
            {
                "job_id": job_id,
                "job_status": "success",
                "phase": "success",
                "terminal": True,
                "operation_id": operation_id,
                "submission_key": submission_key,
            },
        ],
        "active_job_id": None,
        "active_job_ids": [],
        "inventory_complete": True,
        "automatic_fallback": False,
    }
    callbacks.capture_tool_completed(
        {
            "type": "mcpToolCall",
            "status": "completed",
            "tool": "list_execution_jobs",
            "arguments": {"request": {}},
            "result": {
                "structuredContent": inventory_payload,
                "content": [
                    {"type": "text", "text": json.dumps(inventory_payload)}
                ],
            },
        }
    )

    assert callbacks.active_job_terminal is True
    assert callbacks.active_job_collected is False
    callbacks.capture_tool_completed(
        {
            "type": "mcpToolCall",
            "status": "completed",
            "tool": "collect_execution_job",
            "arguments": {
                "request": {
                    "job_id": job_id,
                    "submission_key": submission_key,
                }
            },
            "result": {
                "structuredContent": {
                    "status": "success",
                    "ready": True,
                    "job_id": job_id,
                    "job_status": "success",
                    "operation_id": operation_id,
                    "submission_key": submission_key,
                    "action_result": {"job_id": job_id, "status": "success"},
                }
            },
        }
    )
    assert callbacks.active_job_collected is True


@pytest.mark.parametrize(
    "inventory_update",
    [
        {"inventory_complete": False},
        {"jobs": []},
        {
            "jobs": [
                {
                    "job_id": "job-other",
                    "job_status": "success",
                    "terminal": True,
                    "operation_id": "operation-other",
                    "submission_key": "submission-other",
                },
            ]
        },
        {
            "jobs": [
                {
                    "job_id": "job-inventory-terminal",
                    "job_status": "success",
                    "terminal": True,
                    "operation_id": "operation-inventory-terminal",
                    "submission_key": "submission-inventory-terminal",
                },
                {
                    "job_id": "job-inventory-terminal",
                    "job_status": "success",
                    "terminal": True,
                    "operation_id": "operation-inventory-terminal",
                    "submission_key": "submission-inventory-terminal",
                },
            ]
        },
        {
            "jobs": [
                {
                    "job_id": "job-inventory-terminal",
                    "job_status": "running",
                    "terminal": False,
                    "operation_id": "operation-inventory-terminal",
                    "submission_key": "submission-inventory-terminal",
                }
            ]
        },
    ],
)
def test_incomplete_or_nonterminal_list_inventory_cannot_arm_collect(
    inventory_update: dict[str, object],
) -> None:
    """不完整、重复或非终态清单不能替代当前 turn 的 get。"""

    job_id = "job-inventory-terminal"
    operation_id = "operation-inventory-terminal"
    submission_key = "submission-inventory-terminal"
    callbacks = codex_runtime_module._CodexTurnCallbacks(
        store=SimpleNamespace(),
        manifest=None,
        session_id=None,
        stage="investigation_job_continuation",
        request_digest="request-digest",
        thread_policy_digest="thread-policy",
        active_job_receipt=InvestigationJobReceipt.from_mapping(
            {
                "tool_name": "submit_action_job",
                "identity": {
                    "job_id": job_id,
                    "operation_id": operation_id,
                    "submission_key": submission_key,
                    "action_id": "locate_transition_state",
                    "backend_id": "gaussian",
                    "job_family": "action_execution",
                    "job_type": "action_execution",
                    "runtime": "gaussian",
                    "software_id": "gaussian",
                },
                "arguments": {"request": "inventory-terminal"},
                "result": {
                    "job_id": job_id,
                    "operation_id": operation_id,
                    "submission_key": submission_key,
                },
            }
        ),
    )
    payload = {
        "status": "success",
        "jobs": [
            {
                "job_id": job_id,
                "job_status": "success",
                "terminal": True,
                "operation_id": operation_id,
                "submission_key": submission_key,
            }
        ],
        "inventory_complete": True,
    }
    payload.update(inventory_update)
    list_item = {
        "type": "mcpToolCall",
        "status": "completed",
        "tool": "list_execution_jobs",
        "arguments": {"request": {}},
        "result": {"structuredContent": payload},
    }
    duplicate_active = (
        isinstance(payload.get("jobs"), list)
        and sum(
            item.get("job_id") == job_id
            for item in payload["jobs"]
            if isinstance(item, dict)
        )
        > 1
    )
    if duplicate_active:
        with pytest.raises(HarnessError, match="重复返回"):
            callbacks.capture_tool_completed(list_item)
    else:
        callbacks.capture_tool_completed(list_item)

    assert callbacks.active_job_terminal is False
    with pytest.raises(HarnessError, match="collect 前必须"):
        callbacks.capture_tool_completed(
            {
                "type": "mcpToolCall",
                "status": "completed",
                "tool": "collect_execution_job",
                "arguments": {"job_id": job_id},
                "result": {"structuredContent": {"status": "success", "job_id": job_id}},
            }
        )


def test_conflicting_single_list_inventory_item_fails_closed() -> None:
    """单项清单明确给出错误作业身份时必须直接拒绝，而非静默放行。"""

    receipt = InvestigationJobReceipt.from_mapping(
        {
            "tool_name": "submit_action_job",
            "identity": {
                "job_id": "job-inventory-terminal",
                "operation_id": "operation-inventory-terminal",
                "submission_key": "submission-inventory-terminal",
                "action_id": "locate_transition_state",
                "backend_id": "gaussian",
                "job_family": "action_execution",
                "job_type": "action_execution",
                "runtime": "gaussian",
                "software_id": "gaussian",
            },
            "arguments": {"request": "inventory-terminal"},
            "result": {},
        }
    )
    callbacks = codex_runtime_module._CodexTurnCallbacks(
        store=SimpleNamespace(),
        manifest=None,
        session_id=None,
        stage="investigation_job_continuation",
        request_digest="request-digest",
        thread_policy_digest="thread-policy",
        active_job_receipt=receipt,
    )

    with pytest.raises(HarnessError, match="identity 冲突"):
        callbacks.capture_tool_completed(
            {
                "type": "mcpToolCall",
                "status": "completed",
                "tool": "list_execution_jobs",
                "arguments": {"request": {}},
                "result": {
                    "structuredContent": {
                        "status": "success",
                        "jobs": [
                            {
                                "job_id": receipt.identity.job_id,
                                "job_status": "success",
                                "terminal": True,
                                "operation_id": "operation-other",
                                "submission_key": receipt.identity.submission_key,
                            }
                        ],
                        "inventory_complete": True,
                    }
                },
            }
        )
    assert callbacks.active_job_terminal is False


@pytest.mark.parametrize("conflict_field", ["operation_id", "submission_key"])
def test_historical_list_item_cannot_rebind_active_identity(
    conflict_field: str,
) -> None:
    """历史 job 不得复用当前 operation 或 submission 身份。"""

    receipt = InvestigationJobReceipt.from_mapping(
        {
            "tool_name": "submit_action_job",
            "identity": {
                "job_id": "job-active",
                "operation_id": "operation-active",
                "submission_key": "submission-active",
                "action_id": "locate_transition_state",
                "backend_id": "gaussian",
                "job_family": "action_execution",
                "job_type": "action_execution",
                "runtime": "gaussian",
                "software_id": "gaussian",
            },
            "arguments": {"request": "inventory-terminal"},
            "result": {},
        }
    )
    history = {
        "job_id": "job-history",
        "job_status": "success",
        "terminal": True,
        "operation_id": "operation-history",
        "submission_key": "submission-history",
    }
    history[conflict_field] = getattr(receipt.identity, conflict_field)
    callbacks = codex_runtime_module._CodexTurnCallbacks(
        store=SimpleNamespace(),
        manifest=None,
        session_id=None,
        stage="investigation_job_continuation",
        request_digest="request-digest",
        thread_policy_digest="thread-policy",
        active_job_receipt=receipt,
    )

    with pytest.raises(HarnessError, match="绑定到其它 job_id"):
        callbacks.capture_tool_completed(
            {
                "type": "mcpToolCall",
                "status": "completed",
                "tool": "list_execution_jobs",
                "arguments": {"request": {}},
                "result": {
                    "structuredContent": {
                        "status": "success",
                        "jobs": [history],
                        "inventory_complete": True,
                    }
                },
            }
        )
    assert callbacks.active_job_terminal is False


def test_job_progress_rejects_duplicate_job_id_and_status_conflicts() -> None:
    """Codex 回调不允许顶层/嵌套 job 身份或作业状态静默覆盖。"""

    identity = {
        "job_id": "job-conflict",
        "operation_id": "operation-conflict",
        "submission_key": "submission-conflict",
        "action_id": "action-conflict",
        "backend_id": "backend-conflict",
    }

    def callbacks_for_progress() -> codex_runtime_module._CodexTurnCallbacks:
        """建立一个已接受 submit 的单作业 tracker。"""

        callbacks = codex_runtime_module._CodexTurnCallbacks(
            store=SimpleNamespace(),
            manifest=None,
            session_id=None,
            stage="initial_investigation",
            request_digest="request-digest",
            thread_policy_digest="thread-policy",
        )
        callbacks.capture_tool_completed(
            {
                "type": "mcpToolCall",
                "status": "completed",
                "tool": "submit_transition_state_path_job",
                "arguments": {"request": "conflict"},
                "result": {"structuredContent": identity},
            }
        )
        return callbacks

    for conflict, payload_update, expected in (
        (
            "job_id",
            {"job": {"job_id": "job-other", "status": "running"}},
            "job_id 冲突",
        ),
        (
            "status",
            {
                "job_status": "queued",
                "job": {"job_id": "job-conflict", "status": "running"},
            },
            "状态",
        ),
    ):
        callbacks = callbacks_for_progress()
        payload = {
            "status": "success",
            "job_id": "job-conflict",
            "job": {"job_id": "job-conflict", "status": "running"},
        }
        payload.update(payload_update)
        with pytest.raises(HarnessError, match=expected):
            callbacks.capture_tool_completed(
                {
                    "type": "mcpToolCall",
                    "status": "completed",
                    "tool": "get_transition_state_path_job",
                    "arguments": {"job_id": "job-conflict"},
                    "result": {"structuredContent": payload},
                }
            )


def test_job_progress_rejects_conflicting_statuses_across_nested_envelopes() -> None:
    """一次 get 回执的多个 JSON envelope 不能合并出 running+success。"""

    callbacks = codex_runtime_module._CodexTurnCallbacks(
        store=SimpleNamespace(),
        manifest=None,
        session_id=None,
        stage="initial_investigation",
        request_digest="request-digest",
        thread_policy_digest="thread-policy",
    )
    job_id = "job-cross-envelope"
    callbacks.capture_tool_completed(
        {
            "type": "mcpToolCall",
            "status": "completed",
            "tool": "submit_native_job",
            "arguments": {"request": "cross-envelope"},
            "result": {
                "structuredContent": {
                    "job_id": job_id,
                    "operation_id": "operation-cross-envelope",
                    "submission_key": "submission-cross-envelope",
                    "job_type": "native_software",
                    "runtime": "gaussian",
                    "operation_fingerprint": "a" * 64,
                }
            },
        }
    )
    result = {
        "structuredContent": {
            "status": "success",
            "job_id": job_id,
            "job_status": "running",
        },
        "content": [
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "status": "success",
                        "job_id": job_id,
                        "job_status": "success",
                    }
                ),
            }
        ],
    }
    with pytest.raises(HarnessError, match="多个.*状态"):
        callbacks.capture_tool_completed(
            {
                "type": "mcpToolCall",
                "status": "completed",
                "tool": "get_execution_job",
                "arguments": {"job_id": job_id},
                "result": result,
            }
        )
    assert callbacks.active_job_terminal is False


def test_job_progress_decodes_nested_minichem_collect_envelope() -> None:
    """MiniChem collect 的双层 JSON text envelope 仍能核对并完成原作业。"""

    callbacks = codex_runtime_module._CodexTurnCallbacks(
        store=SimpleNamespace(),
        manifest=None,
        session_id=None,
        stage="initial_investigation",
        request_digest="request-digest",
        thread_policy_digest="thread-policy",
    )
    job_id = "job_8912052d"
    callbacks.capture_tool_completed(
        {
            "type": "mcpToolCall",
            "status": "completed",
            "tool": "submit_transition_state_path_job",
            "arguments": {"request": "scripted-submit"},
            "result": {
                "structuredContent": {
                    "job_id": job_id,
                    "operation_id": "operation-8912052d",
                    "submission_key": "submission-8912052d",
                    "action_id": "action-8912052d",
                    "backend_id": "backend-8912052d",
                }
            },
        }
    )
    callbacks.capture_tool_completed(
        {
            "type": "mcpToolCall",
            "status": "completed",
            "tool": "get_transition_state_path_job",
            "arguments": {"job_id": job_id},
            "result": {
                "structuredContent": {
                    "status": "success",
                    "job_id": job_id,
                    "job_status": "success",
                }
            },
        }
    )
    inner_job_result = {
        "status": "success",
        "ready": True,
        "job_id": job_id,
        "job_status": "success",
        "operation_id": "operation-8912052d",
        "submission_id": "submission-8912052d",
        "action": "transition_state_path",
        "backend": "minichem",
        "action_result": {"job_id": job_id, "status": "success"},
        "large_result_marker": "x" * 300_000,
    }
    nested_mcp_envelope = {
        "content": [
            {"type": "text", "text": json.dumps(inner_job_result)}
        ],
        "structuredContent": None,
        "isError": False,
    }
    callbacks.capture_tool_completed(
        {
            "type": "mcpToolCall",
            "status": "completed",
            "tool": "collect_transition_state_path_job",
            "arguments": {"job_id": job_id},
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(nested_mcp_envelope),
                    }
                ],
                "structuredContent": None,
            },
        }
    )

    assert callbacks.active_job_collected is True
    assert [receipt.identity.job_id for receipt in callbacks.completed_job_receipts] == [job_id]


def test_job_progress_rejects_sdk_truncated_nested_text_without_prefix_inference() -> None:
    """内层 text 被 SDK 截断时不得用前缀字段猜测 collect 成功。"""

    job_id = "job_8912052d"
    identity = InvestigationJobIdentity(
        job_id=job_id,
        operation_id="operation-8912052d",
        submission_key="submission-8912052d",
        action_id="action-8912052d",
        backend_id="backend-8912052d",
    )
    callbacks = codex_runtime_module._CodexTurnCallbacks(
        store=SimpleNamespace(),
        manifest=None,
        session_id=None,
        stage="initial_investigation",
        request_digest="request-digest",
        thread_policy_digest="thread-policy",
        active_job_receipt=InvestigationJobReceipt(
            tool_name="submit_transition_state_path_job",
            identity=identity,
            arguments={"request": "scripted-submit"},
            result=identity.as_mapping(),
        ),
        active_job_terminal=True,
    )
    truncated_inner_text = (
        '{"status":"success","ready":true,"job_id":"job_8912052d",'
        '"action_result":{"large":"prefix…166889 chars truncated…'
    )
    outer_envelope = {
        "content": [{"type": "text", "text": truncated_inner_text}],
        "structuredContent": None,
        "isError": False,
    }

    with pytest.raises(HarnessError, match="没有匹配当前 active job_id"):
        callbacks.capture_tool_completed(
            {
                "type": "mcpToolCall",
                "status": "completed",
                "tool": "collect_transition_state_path_job",
                "arguments": {"job_id": job_id},
                "result": {
                    "content": [
                        {"type": "text", "text": json.dumps(outer_envelope)}
                    ],
                    "structuredContent": None,
                },
            }
        )

    assert callbacks.active_job_collected is False
    assert callbacks.completed_job_receipts == []


def test_job_progress_allows_matching_collect_error_for_model_repair() -> None:
    """匹配当前作业的 collect 错误应交给模型，不应升级为 HarnessError。"""

    job_id = "job_8912052d"
    identity = InvestigationJobIdentity(
        job_id=job_id,
        operation_id="operation-8912052d",
        submission_key="submission-8912052d",
        action_id="action-8912052d",
        backend_id="backend-8912052d",
    )
    callbacks = codex_runtime_module._CodexTurnCallbacks(
        store=SimpleNamespace(),
        manifest=None,
        session_id=None,
        stage="initial_investigation",
        request_digest="request-digest",
        thread_policy_digest="thread-policy",
        active_job_receipt=InvestigationJobReceipt(
            tool_name="submit_transition_state_path_job",
            identity=identity,
            arguments={"request": "scripted-submit"},
            result=identity.as_mapping(),
        ),
        active_job_terminal=True,
    )

    callbacks.capture_tool_completed(
        {
            "type": "mcpToolCall",
            "status": "completed",
            "tool": "collect_transition_state_path_job",
            "arguments": {"job_id": job_id},
            "result": {
                "structuredContent": {
                    "status": "invalid_request",
                    "valid": False,
                    "job_id": job_id,
                    "error": {
                        "code": "transition_state_path_result_unavailable",
                        "message": "worker result unavailable",
                        "job_id": job_id,
                    },
                    "automatic_fallback": False,
                }
            },
        }
    )

    assert callbacks.active_job_collected is False
    assert callbacks.completed_job_receipts == []
    with pytest.raises(HarnessError, match="仍有未 collect"):
        callbacks.capture_tool_completed(
            {
                "type": "mcpToolCall",
                "status": "completed",
                "tool": "submit_transition_state_path_job",
                "arguments": {"request": "must-remain-blocked"},
                "result": {
                    "structuredContent": {
                        "job_id": "job-blocked-after-collect-error",
                        "operation_id": "operation-blocked-after-collect-error",
                        "submission_key": "submission-blocked-after-collect-error",
                        "action_id": "action-blocked-after-collect-error",
                        "backend_id": "backend-blocked-after-collect-error",
                    }
                },
            }
        )


@pytest.mark.parametrize(
    ("submit_tool", "status_tool", "collect_tool", "identity"),
    [
        (
            "submit_transition_state_path_job",
            "get_transition_state_path_job",
            "collect_transition_state_path_job",
            {
                "job_id": "job-ts-failed-collect",
                "operation_id": "operation-ts-failed-collect",
                "submission_key": "submission-ts-failed-collect",
                "action_id": "action-ts-failed-collect",
                "backend_id": "backend-ts-failed-collect",
            },
        ),
        (
            "submit_action_job",
            "get_execution_job",
            "collect_execution_job",
            {
                "job_id": "job-action-failed-collect",
                "operation_id": "operation-action-failed-collect",
                "submission_key": "submission-action-failed-collect",
                "action_id": "scan_reaction_coordinates",
                "backend_id": "pysisyphus",
                "job_type": "action_execution",
                "runtime": "reaction",
            },
        ),
        (
            "submit_native_job",
            "get_execution_job",
            "collect_execution_job",
            {
                "job_id": "job_197d61d3",
                "operation_id": "operation-native-failed-collect",
                "submission_key": "submission-native-failed-collect",
                "job_type": "native_software",
                "runtime": "gaussian",
                "software_id": "gaussian",
            },
        ),
    ],
)
def test_job_progress_releases_gate_for_ready_terminal_collect_error(
    submit_tool: str,
    status_tool: str,
    collect_tool: str,
    identity: dict[str, str],
) -> None:
    """失败作业的 ready collect 错误释放 gate，并通知收集回调。"""

    collected_job_ids: list[str] = []
    callbacks = codex_runtime_module._CodexTurnCallbacks(
        store=SimpleNamespace(),
        manifest=None,
        session_id=None,
        stage="initial_investigation",
        request_digest="request-digest",
        thread_policy_digest="thread-policy",
        job_collected_callback=lambda receipt: collected_job_ids.append(
            receipt["identity"]["job_id"]
        ),
    )
    job_id = identity["job_id"]
    callbacks.capture_tool_completed(
        {
            "type": "mcpToolCall",
            "status": "completed",
            "tool": submit_tool,
            "arguments": {"request": f"request-{job_id}"},
            "result": {"structuredContent": identity},
        }
    )
    callbacks.capture_tool_completed(
        {
            "type": "mcpToolCall",
            "status": "completed",
            "tool": status_tool,
            "arguments": {"job_id": job_id},
            "result": {
                "structuredContent": {
                    "status": "success",
                    "job_id": job_id,
                    "job_status": "failed",
                    "terminal": True,
                    "job": {"job_id": job_id, "status": "failed"},
                }
            },
        }
    )
    callbacks.capture_tool_completed(
        {
            "type": "mcpToolCall",
            "status": "completed",
            "tool": collect_tool,
            "arguments": {"job_id": job_id},
            "result": {
                "structuredContent": {
                    "status": "success",
                    "ready": True,
                    "job_id": job_id,
                    "job_status": "failed",
                    "job": {"job_id": job_id, "status": "failed"},
                    "error": {
                        "code": "nonzero_exit",
                        "message": "Program exited with status 1",
                    },
                }
            },
        }
    )

    assert callbacks.active_job_terminal is True
    assert callbacks.active_job_collected is True
    assert [receipt.identity.job_id for receipt in callbacks.completed_job_receipts] == [job_id]
    assert collected_job_ids == [job_id]

    next_job_id = f"{job_id}-next"
    next_identity = dict(identity)
    next_identity.update(
        {
            "job_id": next_job_id,
            "operation_id": f"{identity['operation_id']}-next",
            "submission_key": f"{identity['submission_key']}-next",
        }
    )
    callbacks.capture_tool_completed(
        {
            "type": "mcpToolCall",
            "status": "completed",
            "tool": submit_tool,
            "arguments": {"request": f"request-{next_job_id}"},
            "result": {"structuredContent": next_identity},
        }
    )
    assert callbacks.active_job_receipt is not None
    assert callbacks.active_job_receipt.identity.job_id == next_job_id


def test_job_progress_keeps_gate_when_terminal_error_payload_lacks_active_job_id() -> None:
    """错误 payload 必须自身绑定 active job，不能借用兄弟 envelope 的身份。"""

    job_id = "job-mixed-envelope"
    identity = InvestigationJobIdentity(
        job_id=job_id,
        operation_id="operation-mixed-envelope",
        submission_key="submission-mixed-envelope",
        action_id="native_software",
        backend_id="gaussian",
        job_family="native_execution",
        job_type="native_software",
        runtime="gaussian",
        software_id="gaussian",
    )
    callbacks = codex_runtime_module._CodexTurnCallbacks(
        store=SimpleNamespace(),
        manifest=None,
        session_id=None,
        stage="initial_investigation",
        request_digest="request-digest",
        thread_policy_digest="thread-policy",
        active_job_receipt=InvestigationJobReceipt(
            tool_name="submit_native_job",
            identity=identity,
            arguments={"request": "scripted-submit"},
            result=identity.as_mapping(),
        ),
        active_job_terminal=True,
    )

    callbacks.capture_tool_completed(
        {
            "type": "mcpToolCall",
            "status": "completed",
            "tool": "collect_execution_job",
            "arguments": {"job_id": job_id},
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({"status": "success", "job_id": job_id}),
                    }
                ],
                "structuredContent": {
                    "status": "success",
                    "ready": True,
                    "job_status": "failed",
                    "job": {"status": "failed"},
                    "error": {
                        "code": "nonzero_exit",
                        "message": "Program exited with status 1",
                    },
                },
            },
        }
    )

    assert callbacks.active_job_collected is False
    assert callbacks.completed_job_receipts == []


class _SubmitReceiptStatusGateway:
    """只实现 submit receipt 归属核对所需的 Controller status 接口。"""

    def __init__(self) -> None:
        self.get_calls: list[str] = []
        self.arm_calls: list[str] = []

    def get_investigation_job_status(self, waiting):
        """返回不含科学 payload 的最小 queued 状态。"""

        identity = waiting.identity
        self.get_calls.append(identity.job_id)
        return {
            "status": "queued",
            "job_id": identity.job_id,
            "operation_id": identity.operation_id,
            "submission_key": identity.submission_key,
            "action_id": identity.action_id,
            "backend_id": identity.backend_id,
            "job": {"job_id": identity.job_id, "status": "queued"},
        }


class _ReceiptReplayAdapter:
    """模拟 completed receipt 恢复，并区分 adapter 调用和真实模型调用。"""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.model_calls = 0
        self.fail_first = True

    def __call__(self, **kwargs):
        """输入 runtime adapter 参数，模拟 completed receipt 后的崩溃与重放。

        首轮执行传入 fence、写临时 receipt 后抛出 transport 错误；第二轮返回同一
        turn 的 replay 结果且不再执行 fence。副作用仅是测试 trace/receipt；参数
        或回调合同非法时直接失败，成功的恢复轮返回 ``CodexSdkRunResult``。
        """

        self.calls.append(dict(kwargs))
        request = kwargs["request"]
        session_id = kwargs["session_id"]
        thread_id = session_id or "thread-replay"
        kwargs["on_thread_ready"](thread_id)
        kwargs["trace_path"].write_text("receipt-replay", encoding="utf-8")
        if self.fail_first:
            before_new_turn = kwargs.get("before_new_turn")
            if callable(before_new_turn):
                before_new_turn()
            self.fail_first = False
            self.model_calls += 1
            codex_sdk_adapter._write_completed_turn_receipt(
                kwargs["completed_turn_receipt_root"],
                thread_id=thread_id,
                turn_id="turn-replay",
                request_digest=kwargs["request_digest"],
                turn_count_before=kwargs["turn_count_before"],
                thread_policy_digest=kwargs["thread_policy_digest"],
            )
            raise HarnessError("completed turn process crash", kind="transport")
        return CodexSdkRunResult(
            response=HarnessResponse(
                content=_investigation_response(),
                model=request.model,
                metadata={"turn_id": "turn-replay"},
            ),
            thread_id=thread_id,
            turn_id="turn-replay",
            replayed=True,
            receipt_turn_count_before=0,
            request_digest=kwargs["request_digest"],
        )


def _initial_request(
    *,
    combination_id: str,
    model: str = "investigation-model",
    codex_thread_id: str | None = None,
) -> dict[str, object]:
    """构造一个单组合 Investigation 请求。"""

    return build_initial_investigation_request(
        "task-1",
        "batch-1",
        combination_id,
        "原始科学问题",
        {"method": "test", "charge": 0},
        codex_thread_id=codex_thread_id,
        model=model,
    )


@pytest.mark.parametrize(
    ("recovery_stage", "persist_recovery_record", "request_digest_drift"),
    (
        ("initial_investigation", True, False),
        ("investigation_job_wake", True, False),
        ("initial_investigation", False, False),
        ("initial_investigation", False, True),
    ),
)
def test_runtime_recovery_only_consumes_old_request_intent_before_wake(
    tmp_path: Path,
    recovery_stage: str,
    persist_recovery_record: bool,
    request_digest_drift: bool,
) -> None:
    """runtime 只用旧 intent 摘要确认终态，不要求重建 prompt 内容摘要。"""

    class _RecoveryOnlyAdapter:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def __call__(self, **kwargs):
            self.calls.append(dict(kwargs))
            assert kwargs["recovery_only"] is True
            kwargs["on_thread_ready"]("thread-1")
            kwargs["on_turn_started"]("turn-interrupted")
            kwargs["on_terminal_turn"](
                "thread-1",
                "turn-interrupted",
                "interrupted",
            )
            raise CodexTerminalTurnError(
                turn_id="turn-interrupted",
                status="interrupted",
            )

    adapter = _RecoveryOnlyAdapter()
    runtime = _runtime(tmp_path, adapter)
    request = _initial_request(
        combination_id="combination-1",
        codex_thread_id="thread-1",
    )
    envelope = runtime._extract_request_envelope(
        request,
        expected_profile="investigation",
    )
    provider = runtime._resolve_provider(
        runtime.investigation_provider,
        envelope["model"],
    )
    store = CodexSessionStore.from_root(
        runtime.session_root,
        owner=CodexSessionOwner.investigation(
            "task-1",
            "batch-1",
            "combination-1",
        ),
        scope_id="scope-1",
        model=provider.model,
    )
    harness_request, request_digest, _secret, _config, policy_digest = (
        runtime._build_codex_turn_policy_inputs(
            envelope=envelope,
            store=store,
            provider=provider,
            mcp_config=runtime._investigation_mcp_config(),
            timeout_seconds=None,
        )
    )
    store.create(
        "thread-1",
        stage="initial_investigation",
        thread_policy_digest=policy_digest,
    )
    persisted_request_digest = "f" * 64 if request_digest_drift else request_digest
    if persist_recovery_record:
        store.record_recovery_pending(
            stage=recovery_stage,
            request_digest=persisted_request_digest,
            thread_policy_digest=policy_digest,
            turn_id=None,
            reason="request_intent",
        )
    waiting = InvestigationJobState(
        task_id="task-1",
        batch_id="batch-1",
        combination_id="combination-1",
        codex_thread_id="thread-1",
        turn_id="turn-interrupted",
        status="waiting_investigation_job",
        identity=InvestigationJobIdentity(
            job_id="job-1",
            operation_id="operation-1",
            submission_key="submission-1",
            action_id="action-1",
            backend_id="backend-1",
        ),
        request_digest="job-request",
        scope_id="scope-1",
    )
    codex_sdk_adapter._write_request_intent(
        store.manifest_path.parent / "completed_turn_receipts",
        thread_id="thread-1",
        request_digest=persisted_request_digest,
        turn_count_before=0,
        physical_turn_count_before=0,
        provider_id=provider.settings.provider_id,
        env_key=provider.settings.env_key,
        model=provider.model,
        role=harness_request.role,
        stage=harness_request.stage,
        profile=envelope["profile"],
        workspace=store.workspace,
        thread_policy_digest=policy_digest,
    )
    assert runtime.has_recoverable_interrupted_investigation_turn(
        request,
        scope_id="scope-1",
        active_job=waiting,
    ) is True
    lifecycle: list[tuple[str, str, str]] = []

    runtime.recover_interrupted_investigation_turn(
        request,
        scope_id="scope-1",
        active_job=waiting,
        on_thread_ready=lambda thread_id: lifecycle.append((thread_id, "", "ready")),
        on_turn_started=lambda thread_id, turn_id: lifecycle.append(
            (thread_id, turn_id, "started")
        ),
        on_terminal_turn=lambda thread_id, turn_id, status: lifecycle.append(
            (thread_id, turn_id, status)
        ),
    )

    assert len(adapter.calls) == 1
    assert adapter.calls[0]["session_id"] == "thread-1"
    assert adapter.calls[0]["request_digest"] == persisted_request_digest
    assert (
        adapter.calls[0]["completion_recovery"] is not None
    ) is persist_recovery_record
    assert lifecycle == [
        ("thread-1", "", "ready"),
        ("thread-1", "turn-interrupted", "started"),
        ("thread-1", "turn-interrupted", "interrupted"),
    ]
    assert store.load_recovery() is None
    assert store.load()["status"] == "created"


def test_runtime_recovery_only_consumes_interrupted_repair_before_continuation(
    tmp_path: Path,
) -> None:
    """repair recovery-only 接受原返修 request，且不启动新的模型 turn。"""

    class _RepairRecoveryAdapter:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def __call__(self, **kwargs):
            self.calls.append(dict(kwargs))
            assert kwargs["recovery_only"] is True
            assert kwargs["recovery_turn_id"] == "repair-turn-1"
            kwargs["on_thread_ready"]("thread-1")
            kwargs["on_turn_started"]("repair-turn-1")
            kwargs["on_terminal_turn"](
                "thread-1",
                "repair-turn-1",
                "interrupted",
            )
            raise CodexTerminalTurnError(
                turn_id="repair-turn-1",
                status="interrupted",
            )

    adapter = _RepairRecoveryAdapter()
    runtime = _runtime(tmp_path, adapter)
    request = build_repair_investigation_request(
        "task-1",
        "batch-1",
        "combination-1",
        "thread-1",
        "v1",
        "原样保留的返修建议",
        repair_count=1,
        model="investigation-model",
    )
    envelope = runtime._extract_request_envelope(
        request,
        expected_profile="investigation",
    )
    provider = runtime._resolve_provider(
        runtime.investigation_provider,
        envelope["model"],
    )
    store = CodexSessionStore.from_root(
        runtime.session_root,
        owner=CodexSessionOwner.investigation(
            "task-1",
            "batch-1",
            "combination-1",
        ),
        scope_id="scope-1",
        model=provider.model,
    )
    harness_request, request_digest, _secret, _config, policy_digest = (
        runtime._build_codex_turn_policy_inputs(
            envelope=envelope,
            store=store,
            provider=provider,
            mcp_config=runtime._investigation_mcp_config(),
            timeout_seconds=None,
        )
    )
    store.create(
        "thread-1",
        stage="initial_investigation",
        thread_policy_digest=policy_digest,
    )
    store.record_completed_turn(stage="initial_investigation", status="completed")
    codex_sdk_adapter._write_request_intent(
        store.manifest_path.parent / "completed_turn_receipts",
        thread_id="thread-1",
        request_digest=request_digest,
        turn_count_before=1,
        physical_turn_count_before=1,
        provider_id=provider.settings.provider_id,
        env_key=provider.settings.env_key,
        model=provider.model,
        role=harness_request.role,
        stage=harness_request.stage,
        profile=envelope["profile"],
        workspace=store.workspace,
        thread_policy_digest=policy_digest,
    )
    waiting = InvestigationJobState(
        task_id="task-1",
        batch_id="batch-1",
        combination_id="combination-1",
        codex_thread_id="thread-1",
        turn_id="repair-turn-1",
        status="waiting_investigation_job",
        identity=InvestigationJobIdentity(
            job_id="job-repair-1",
            operation_id="operation-repair-1",
            submission_key="submission-repair-1",
            action_id="action-repair-1",
            backend_id="backend-repair-1",
        ),
        request_digest="a" * 64,
        scope_id="scope-1",
    )
    assert runtime.has_recoverable_interrupted_investigation_turn(
        request,
        scope_id="scope-1",
        active_job=waiting,
    ) is True

    lifecycle: list[tuple[str, str, str]] = []
    runtime.recover_interrupted_investigation_turn(
        request,
        scope_id="scope-1",
        active_job=waiting,
        on_thread_ready=lambda thread_id: lifecycle.append((thread_id, "", "ready")),
        on_turn_started=lambda thread_id, turn_id: lifecycle.append(
            (thread_id, turn_id, "started")
        ),
        on_terminal_turn=lambda thread_id, turn_id, status: lifecycle.append(
            (thread_id, turn_id, status)
        ),
    )

    assert len(adapter.calls) == 1
    assert adapter.calls[0]["session_id"] == "thread-1"
    assert adapter.calls[0]["request_digest"] == request_digest
    assert lifecycle == [
        ("thread-1", "", "ready"),
        ("thread-1", "repair-turn-1", "started"),
        ("thread-1", "repair-turn-1", "interrupted"),
    ]


def test_runtime_recovery_only_uses_exact_continuation_request_intent(
    tmp_path: Path,
) -> None:
    """后续 repair attempt 必须用 continuation request 的严格 digest 收束。"""

    class _ContinuationRecoveryAdapter:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def __call__(self, **kwargs):
            self.calls.append(dict(kwargs))
            assert kwargs["recovery_only"] is True
            assert kwargs["recovery_turn_id"] == "continuation-turn-1"
            kwargs["on_thread_ready"]("thread-1")
            kwargs["on_turn_started"]("continuation-turn-1")
            kwargs["on_terminal_turn"](
                "thread-1",
                "continuation-turn-1",
                "interrupted",
            )
            raise CodexTerminalTurnError(
                turn_id="continuation-turn-1",
                status="interrupted",
            )

    adapter = _ContinuationRecoveryAdapter()
    runtime = _runtime(tmp_path, adapter)
    request = build_investigation_job_continuation_request(
        "task-1",
        "batch-1",
        "combination-1",
        "thread-1",
        job_id="job-continuation-1",
        operation_id="operation-continuation-1",
        submission_key="submission-continuation-1",
        action_id="search_transition_state_path",
        backend_id="transition_state_path_search",
        scope_id="scope-1",
        scientific_question="原始科学问题",
        hypothesis={"method": "test", "charge": 0},
        explicit_conditions={"temperature": 300},
        frozen_context={"fact": "continuation"},
        calculation_unit={"reactants": ["combination-1"]},
        available_tools=(
            "read_artifact",
            "get_transition_state_path_job",
            "collect_transition_state_path_job",
        ),
        model="investigation-model",
    )
    waiting = InvestigationJobState(
        task_id="task-1",
        batch_id="batch-1",
        combination_id="combination-1",
        codex_thread_id="thread-1",
        turn_id="continuation-turn-1",
        status="waiting_investigation_job",
        identity=InvestigationJobIdentity(
            job_id="job-continuation-1",
            operation_id="operation-continuation-1",
            submission_key="submission-continuation-1",
            action_id="search_transition_state_path",
            backend_id="transition_state_path_search",
        ),
        request_digest="c" * 64,
        scope_id="scope-1",
    )
    envelope = runtime._extract_request_envelope(
        request,
        expected_profile="investigation",
    )
    provider = runtime._resolve_provider(
        runtime.investigation_provider,
        envelope["model"],
    )
    store = CodexSessionStore.from_root(
        runtime.session_root,
        owner=CodexSessionOwner.investigation(
            "task-1",
            "batch-1",
            "combination-1",
        ),
        scope_id="scope-1",
        model=provider.model,
    )
    harness_request, request_digest, _secret, _config, policy_digest = (
        runtime._build_codex_turn_policy_inputs(
            envelope=envelope,
            store=store,
            provider=provider,
            mcp_config=runtime._investigation_mcp_config(),
            timeout_seconds=None,
        )
    )
    store.create(
        "thread-1",
        stage="initial_investigation",
        thread_policy_digest=policy_digest,
    )
    store.record_completed_turn(stage="initial_investigation", status="completed")
    store.record_recovery_pending(
        stage="investigation_job_continuation",
        request_digest=request_digest,
        thread_policy_digest=policy_digest,
        turn_id=None,
        reason="request_intent",
    )
    codex_sdk_adapter._write_request_intent(
        store.manifest_path.parent / "completed_turn_receipts",
        thread_id="thread-1",
        request_digest=request_digest,
        turn_count_before=1,
        physical_turn_count_before=1,
        provider_id=provider.settings.provider_id,
        env_key=provider.settings.env_key,
        model=provider.model,
        role=harness_request.role,
        stage=harness_request.stage,
        profile=envelope["profile"],
        workspace=store.workspace,
        thread_policy_digest=policy_digest,
    )

    assert runtime.has_recoverable_interrupted_investigation_turn(
        request,
        scope_id="scope-1",
        active_job=waiting,
    ) is True
    runtime.recover_interrupted_investigation_turn(
        request,
        scope_id="scope-1",
        active_job=waiting,
        on_thread_ready=lambda _thread_id: None,
        on_turn_started=lambda _thread_id, _turn_id: None,
        on_terminal_turn=lambda _thread_id, _turn_id, _status: None,
    )

    assert len(adapter.calls) == 1
    assert adapter.calls[0]["request_digest"] == request_digest
    assert (
        adapter.calls[0]["request"].metadata["request_kind"]
        == "investigation_job_continuation"
    )


def test_existing_goal_investigation_session_preserves_frozen_goal_policy(
    tmp_path: Path,
) -> None:
    """默认关闭 Goal 后，已有 Goal session 仍使用创建时的能力摘要。"""

    fake_adapter = _FakeAdapter()
    runtime = _runtime(tmp_path, fake_adapter)
    request = _initial_request(
        combination_id="combination-1",
        codex_thread_id="thread-1",
    )
    envelope = runtime._extract_request_envelope(
        request,
        expected_profile="investigation",
    )
    provider = runtime._resolve_provider(
        runtime.investigation_provider,
        envelope["model"],
    )
    store = CodexSessionStore.from_root(
        runtime.session_root,
        owner=CodexSessionOwner.investigation(
            "task-1",
            "batch-1",
            "combination-1",
        ),
        scope_id="scope-1",
        model=provider.model,
    )
    harness_request, _request_digest, _secret, _goal_config, _goal_digest = (
        runtime._build_codex_turn_policy_inputs(
            envelope=envelope,
            store=store,
            provider=provider,
            mcp_config=runtime._investigation_mcp_config(),
            timeout_seconds=None,
        )
    )
    goal_config = codex_sdk_adapter.build_codex_capability_config(
        thread_config=runtime._provider_thread_config(
            provider,
            runtime._investigation_mcp_config(),
        ),
        workspace=store.workspace,
        profile=str(envelope["profile"]),
        env_key=provider.settings.env_key,
        environment_keys=tuple(
            {
                **provider.secret_environment,
                **runtime._resolve_mcp_secret_environment(),
            }
        ),
        auth_mode=runtime.auth_mode,
        enable_investigation_goals=True,
    )
    goal_digest = codex_sdk_adapter.calculate_codex_thread_policy_digest(
        thread_config=goal_config,
        workspace=store.workspace,
        provider_id=provider.settings.provider_id,
        env_key=provider.settings.env_key,
        model=provider.model,
        role=harness_request.role,
        stage=harness_request.stage,
        profile=envelope["profile"],
        auth_mode=runtime.auth_mode,
        permission_profile=CODEX_INVESTIGATION_PERMISSION_PROFILE,
    )
    store.create(
        "thread-1",
        stage="initial_investigation",
        thread_policy_digest=goal_digest,
    )

    _request, _digest, _environment, resumed_config, resumed_policy = (
        runtime._build_codex_turn_policy_inputs(
            envelope=envelope,
            store=store,
            provider=provider,
            mcp_config=runtime._investigation_mcp_config(),
            timeout_seconds=None,
        )
    )

    assert resumed_config["features"]["goals"] is True
    assert resumed_policy == goal_digest

    runtime.run_investigation_turn(request, scope_id="scope-1")

    assert fake_adapter.calls[0]["enable_investigation_goals"] is True
    assert fake_adapter.calls[0]["thread_config"]["features"]["goals"] is True
    assert fake_adapter.calls[0]["thread_policy_digest"] == goal_digest


def test_missing_request_intent_yields_to_completed_continuation_receipt(
    tmp_path: Path,
) -> None:
    """成功 continuation 已有完整回执时，缺失 intent 应交给普通 continuation 重放。"""

    runtime = _runtime(tmp_path, _FakeAdapter())
    request = _initial_request(
        combination_id="combination-1",
        codex_thread_id="thread-1",
    )
    envelope = runtime._extract_request_envelope(
        request,
        expected_profile="investigation",
    )
    provider = runtime._resolve_provider(
        runtime.investigation_provider,
        envelope["model"],
    )
    store = CodexSessionStore.from_root(
        runtime.session_root,
        owner=CodexSessionOwner.investigation(
            "task-1",
            "batch-1",
            "combination-1",
        ),
        scope_id="scope-1",
        model=provider.model,
    )
    _harness_request, _request_digest, _secret, _config, policy_digest = (
        runtime._build_codex_turn_policy_inputs(
            envelope=envelope,
            store=store,
            provider=provider,
            mcp_config=runtime._investigation_mcp_config(),
            timeout_seconds=None,
        )
    )
    store.create(
        "thread-1",
        stage="initial_investigation",
        thread_policy_digest=policy_digest,
    )
    store.record_completed_turn(
        stage="investigation_job_continuation",
        status="completed",
    )
    codex_sdk_adapter._write_completed_turn_receipt(
        store.manifest_path.parent / "completed_turn_receipts",
        thread_id="thread-1",
        turn_id="turn-continuation",
        request_digest="a" * 64,
        turn_count_before=0,
        thread_policy_digest=policy_digest,
    )
    waiting = InvestigationJobState(
        task_id="task-1",
        batch_id="batch-1",
        combination_id="combination-1",
        codex_thread_id="thread-1",
        turn_id="turn-continuation",
        status="waiting_investigation_job",
        identity=InvestigationJobIdentity(
            job_id="job-replay",
            operation_id="operation-replay",
            submission_key="submission-replay",
            action_id="action-replay",
            backend_id="backend-replay",
        ),
        request_digest="b" * 64,
        scope_id="scope-1",
    )

    assert runtime.has_recoverable_interrupted_investigation_turn(
        request,
        scope_id="scope-1",
        active_job=waiting,
    ) is False

    (store.manifest_path.parent / "completed_turn_receipts" / "turn-continuation.json").unlink()
    with pytest.raises(HarnessError, match="request-intent is missing"):
        runtime.has_recoverable_interrupted_investigation_turn(
            request,
            scope_id="scope-1",
            active_job=waiting,
        )


def test_later_failed_resume_intent_yields_to_completed_waiting_receipt(
    tmp_path: Path,
) -> None:
    """成功 waiting turn 后的失败 Resume intent 不能冒充原 turn 中断恢复。"""

    runtime = _runtime(tmp_path, _FakeAdapter())
    request = _initial_request(
        combination_id="combination-1",
        codex_thread_id="thread-1",
    )
    envelope = runtime._extract_request_envelope(
        request,
        expected_profile="investigation",
    )
    provider = runtime._resolve_provider(
        runtime.investigation_provider,
        envelope["model"],
    )
    store = CodexSessionStore.from_root(
        runtime.session_root,
        owner=CodexSessionOwner.investigation(
            "task-1",
            "batch-1",
            "combination-1",
        ),
        scope_id="scope-1",
        model=provider.model,
    )
    harness_request, _request_digest, _secret, _config, policy_digest = (
        runtime._build_codex_turn_policy_inputs(
            envelope=envelope,
            store=store,
            provider=provider,
            mcp_config=runtime._investigation_mcp_config(),
            timeout_seconds=None,
        )
    )
    store.create(
        "thread-1",
        stage="initial_investigation",
        thread_policy_digest=policy_digest,
    )
    store.record_completed_turn(
        stage="investigation_job_continuation",
        status="completed",
    )
    receipt_root = store.manifest_path.parent / "completed_turn_receipts"
    codex_sdk_adapter._write_completed_turn_receipt(
        receipt_root,
        thread_id="thread-1",
        turn_id="turn-continuation",
        request_digest="a" * 64,
        turn_count_before=0,
        thread_policy_digest=policy_digest,
    )
    codex_sdk_adapter._write_request_intent(
        receipt_root,
        thread_id="thread-1",
        request_digest="b" * 64,
        turn_count_before=1,
        physical_turn_count_before=3,
        provider_id=provider.settings.provider_id,
        env_key=provider.settings.env_key,
        model=provider.model,
        role=harness_request.role,
        stage=harness_request.stage,
        profile=envelope["profile"],
        workspace=store.workspace,
        thread_policy_digest=policy_digest,
    )
    store.record_recovery_pending(
        stage="initial_investigation",
        request_digest="b" * 64,
        thread_policy_digest=policy_digest,
        turn_id=None,
        reason="request_intent",
    )
    waiting = InvestigationJobState(
        task_id="task-1",
        batch_id="batch-1",
        combination_id="combination-1",
        codex_thread_id="thread-1",
        turn_id="turn-continuation",
        status="waiting_investigation_job",
        identity=InvestigationJobIdentity(
            job_id="job-replay",
            operation_id="operation-replay",
            submission_key="submission-replay",
            action_id="action-replay",
            backend_id="backend-replay",
        ),
        request_digest="c" * 64,
        scope_id="scope-1",
    )

    assert runtime.has_recoverable_interrupted_investigation_turn(
        request,
        scope_id="scope-1",
        active_job=waiting,
    ) is False


@pytest.mark.parametrize(
    ("status", "turn_count"),
    (
        ("active", 0),
        ("recovery_pending", 0),
        ("contract_failure", 0),
        ("completed", 0),
        ("created", 1),
    ),
)
def test_recovery_only_without_record_rejects_unstable_manifest(
    status: str,
    turn_count: int,
) -> None:
    """缺少恢复记录时，活动、失败及状态/计数不一致的 manifest 继续阻断。"""

    with pytest.raises(HarnessError, match="without a recovery record is not stable"):
        InvestigationCodexRuntime._assert_recovery_only_available(
            {
                "status": status,
                "turn_count": turn_count,
            },
            None,
            request_digest="a" * 64,
            thread_policy_digest="b" * 64,
            stage="initial_investigation",
        )


def test_completed_submit_event_persists_sidecar_before_turn_return_and_ignores_text(
    tmp_path: Path,
) -> None:
    """真实 MCP completed 事件先写 waiting，assistant 正文不能伪造作业身份。"""

    store = InvestigationWaitingJobStore(tmp_path / "waiting")
    controller = InvestigationJobController(
        _SubmitReceiptStatusGateway(),
        require_status_identity=True,
    )
    adapter = _McpCompletedSubmitAdapter(
        lambda: store.get("task-1", "batch-1", "combination-1") is not None
    )
    runtime = _runtime(tmp_path, adapter)
    lifecycle: list[tuple[str, str]] = []
    callback_snapshots: list[object] = []

    def on_turn_started(thread_id: str, turn_id: str) -> None:
        """记录真实 turn 身份，供 receipt sidecar 绑定。"""

        lifecycle.append((thread_id, turn_id))

    def on_job_accepted(receipt_mapping) -> None:
        """把 runtime 的真实 receipt 转换为 waiting sidecar。"""

        receipt = InvestigationJobReceipt.from_mapping(receipt_mapping)
        state = controller.capture_submit_receipt(
            receipt,
            task_id="task-1",
            batch_id="batch-1",
            combination_id="combination-1",
            codex_thread_id=lifecycle[-1][0],
            turn_id=lifecycle[-1][1],
            scope_id="scope-1",
            persist_state=store.save_waiting,
        )
        callback_snapshots.append(store.get("task-1", "batch-1", "combination-1"))
        assert state.identity.job_id == "job-accepted"

    result = runtime.run_investigation_turn(
        _initial_request(combination_id="combination-1"),
        scope_id="scope-1",
        on_turn_started=on_turn_started,
        on_job_accepted=on_job_accepted,
    )

    record = store.get("task-1", "batch-1", "combination-1")
    assert adapter.sidecar_visible_before_return is True
    assert callback_snapshots and callback_snapshots[0] is not None
    assert lifecycle == [("thread-accepted", "turn-accepted")]
    assert result.result is None
    assert result.accepted_job is not None
    assert result.accepted_job["identity"]["job_id"] == "job-accepted"
    assert record is not None
    assert record["state"]["identity"]["job_id"] == "job-accepted"
    assert "assistant-forged" not in json.dumps(record, ensure_ascii=False)
    assert "output_schema" not in adapter.calls[0]




def test_repeated_collect_for_completed_previous_job_is_ignored_after_next_submit() -> None:
    """第二个作业已提交后，重复收集已完成的首个作业不能污染当前作业状态。"""

    callbacks = codex_runtime_module._CodexTurnCallbacks(
        store=SimpleNamespace(),
        manifest=None,
        session_id=None,
        stage="initial_investigation",
        request_digest="request-digest",
        thread_policy_digest="thread-policy",
    )

    def capture(item: dict[str, object]) -> None:
        """把固定事件交给真实回调 seam，保持回执形状与 MCP 完成事件一致。"""

        callbacks.capture_tool_completed(item)

    for job_id in ("job-1", "job-2"):
        capture(_SameTurnSequentialJobAdapter._item(
            "submit_transition_state_path_job", job_id
        ))
        capture(_SameTurnSequentialJobAdapter._item(
            "get_transition_state_path_job",
            job_id,
            status="succeeded" if job_id == "job-1" else "running",
        ))
        if job_id == "job-1":
            capture(_SameTurnSequentialJobAdapter._item(
                "collect_transition_state_path_job", job_id
            ))

    # 新 turn/恢复路径不一定把旧 completed receipt 放进当前 callbacks；
    # scope 鉴权由 MCP registry 负责，本地 tracker 仍应允许只读历史 job。
    callbacks.completed_job_receipts.clear()

    # 模型在第二个 submit 后重复核对已完成的第一个 job；同族 get/collect
    # 回执是幂等的，不应被当前 job-2 的 active tracker 当成 job_id 串读。
    capture(_SameTurnSequentialJobAdapter._item(
        "get_transition_state_path_job", "job-1", status="succeeded"
    ))
    capture(_SameTurnSequentialJobAdapter._item(
        "collect_transition_state_path_job", "job-1"
    ))

    assert callbacks.active_job_receipt is not None
    assert callbacks.active_job_receipt.identity.job_id == "job-2"
    assert callbacks.active_job_terminal is False
    assert callbacks.active_job_collected is False

    # 参数声称查询 active job-2、结果却返回历史 job-1 时，不能因两边的
    # job_id 并集包含 active 身份而放行，必须报告回执身份不一致。
    mismatched = _SameTurnSequentialJobAdapter._item(
        "get_transition_state_path_job", "job-1", status="succeeded"
    )
    mismatched["arguments"] = {"job_id": "job-2"}
    with pytest.raises(HarnessError, match="没有匹配当前 active job_id"):
        capture(mismatched)

    # 历史作业可重复查看，但当前未 collect 的 job-2 仍然阻止新 submit。
    with pytest.raises(HarnessError, match="不能并行提交第二个任务"):
        capture(_SameTurnSequentialJobAdapter._item(
            "submit_transition_state_path_job", "job-3"
        ))




@pytest.mark.parametrize("wake_phase", ["final", "submit-2"])
def test_public_runtime_wake_supports_final_result_or_one_next_submit(
    tmp_path: Path,
    wake_phase: str,
) -> None:
    """同一 runtime thread 的 wake 可直接返回 Result，或只提交一个 next job。"""

    adapter = _ScriptedAsyncInvestigationAdapter(("submit-1", wake_phase))
    runtime = _runtime(tmp_path, adapter)
    initial = runtime.run_investigation_turn(
        _initial_request(combination_id="combination-1"),
        scope_id="scope-1",
    )
    assert initial.accepted_job is not None
    initial_identity = initial.accepted_job["identity"]
    wake = build_investigation_job_wake_request(
        "task-1",
        "batch-1",
        "combination-1",
        initial.thread_id,
        job_id=initial_identity["job_id"],
        operation_id=initial_identity["operation_id"],
        submission_key=initial_identity["submission_key"],
        action_id=initial_identity["action_id"],
        backend_id=initial_identity["backend_id"],
        terminal_status="succeeded",
        available_tools=(
            "submit_transition_state_path_job",
            "get_transition_state_path_job",
            "collect_transition_state_path_job",
        ),
        model="investigation-model",
    )
    collected_jobs: list[str] = []
    resumed = runtime.run_investigation_turn(
        wake,
        scope_id="scope-1",
        on_job_collected=lambda receipt: collected_jobs.append(
            receipt["identity"]["job_id"]
        ),
    )

    assert resumed.thread_id == initial.thread_id
    assert [call["session_id"] for call in adapter.calls] == [None, initial.thread_id]
    assert collected_jobs == [initial_identity["job_id"]]
    if wake_phase == "final":
        assert isinstance(resumed.result, InvestigationResponse)
        assert resumed.accepted_job is None
    else:
        assert resumed.result is None
        assert resumed.accepted_job is not None
        assert resumed.accepted_job["identity"]["job_id"] == "job-2"


@pytest.mark.parametrize("resume_phase", ["running", "final"])
def test_public_runtime_continuation_reuses_thread_for_running_or_terminal_job(
    tmp_path: Path,
    resume_phase: str,
) -> None:
    """任意时刻 continuation 都复用原 thread，running 不重复 submit，终态可 collect。"""

    adapter = _ScriptedAsyncInvestigationAdapter(("submit-1", resume_phase))
    runtime = _runtime(tmp_path, adapter)
    initial = runtime.run_investigation_turn(
        _initial_request(combination_id="combination-1"),
        scope_id="scope-1",
    )
    assert initial.accepted_job is not None
    controller = InvestigationJobController(
        _SubmitReceiptStatusGateway(),
        require_status_identity=True,
    )
    waiting = controller.capture_submit_receipt(
        InvestigationJobReceipt.from_mapping(initial.accepted_job),
        task_id="task-1",
        batch_id="batch-1",
        combination_id="combination-1",
        codex_thread_id=initial.thread_id,
        turn_id=initial.turn_id,
        scope_id="scope-1",
    )
    continuation = build_investigation_job_continuation_request(
        "task-1",
        "batch-1",
        "combination-1",
        initial.thread_id,
        job_id=waiting.identity.job_id,
        operation_id=waiting.identity.operation_id,
        submission_key=waiting.identity.submission_key,
        action_id=waiting.identity.action_id,
        backend_id=waiting.identity.backend_id,
        model="investigation-model",
    )
    resumed = runtime.run_investigation_turn(
        continuation,
        scope_id="scope-1",
        active_job=waiting,
    )

    assert resumed.thread_id == initial.thread_id
    assert [call["session_id"] for call in adapter.calls] == [None, initial.thread_id]
    continuation_request = adapter.calls[1]["request"]
    assert continuation_request.metadata["request_kind"] == "investigation_job_continuation"
    assert "terminal_status" not in continuation_request.metadata
    assert "terminal_status" not in json.loads(continuation_request.messages[-1]["content"])
    if resume_phase == "running":
        assert resumed.result is None
        assert resumed.accepted_job is not None
        assert resumed.accepted_job["identity"]["job_id"] == waiting.identity.job_id
    else:
        assert isinstance(resumed.result, InvestigationResponse)
        assert resumed.accepted_job is None


def _action_execution_waiting_state() -> InvestigationJobState:
    """构造带完整身份的 action_execution waiting sidecar。"""

    identity = InvestigationJobIdentity(
        job_id="job-action-recovery",
        operation_id="operation-action-recovery",
        submission_key="submission-action-recovery",
        action_id="scan_reaction_coordinates",
        backend_id="gaussian",
        job_family="action_execution",
        job_type="action_execution",
        runtime="gaussian",
        software_id="gaussian",
        operation_fingerprint="a" * 64,
    )
    return InvestigationJobState(
        task_id="task-1",
        batch_id="batch-1",
        combination_id="combination-1",
        codex_thread_id="thread-action-recovery",
        turn_id="turn-action-submit",
        status="waiting_investigation_job",
        identity=identity,
        request_digest="b" * 64,
        scope_id="scope-1",
    )


def _action_execution_recovery_request(
    waiting: InvestigationJobState,
    *,
    request_kind: str,
) -> dict[str, object]:
    """构造带完整 action_execution 身份的 wake 或 continuation request。"""

    identity = waiting.identity
    common = {
        "job_id": identity.job_id,
        "operation_id": identity.operation_id,
        "submission_key": identity.submission_key,
        "action_id": identity.action_id,
        "backend_id": identity.backend_id,
        "job_family": identity.job_family,
        "job_type": identity.job_type,
        "runtime": identity.runtime,
        "software_id": identity.software_id,
        "operation_fingerprint": identity.operation_fingerprint,
        "scope_id": waiting.scope_id,
        "available_tools": ("get_execution_job", "collect_execution_job"),
        "model": "investigation-model",
    }
    if request_kind == "investigation_job_wake":
        request = build_investigation_job_wake_request(
            "task-1",
            "batch-1",
            "combination-1",
            waiting.codex_thread_id,
            terminal_status="succeeded",
            **common,
        )
    else:
        request = build_investigation_job_continuation_request(
            "task-1",
            "batch-1",
            "combination-1",
            waiting.codex_thread_id,
            **common,
        )
    return request


@pytest.mark.parametrize(
    "request_kind",
    ["investigation_job_wake", "investigation_job_continuation"],
)
@pytest.mark.parametrize(
    "identity_field",
    [
        "job_family",
        "action_id",
        "backend_id",
        "job_type",
        "runtime",
        "software_id",
        "operation_fingerprint",
    ],
)
def test_action_execution_recovery_rejects_complete_identity_drift(
    tmp_path: Path,
    request_kind: str,
    identity_field: str,
) -> None:
    """保持三项作业 ID 不变时，恢复请求仍不能替换 action_execution 身份。"""

    waiting = _action_execution_waiting_state()
    request = _action_execution_recovery_request(waiting, request_kind=request_kind)
    request["metadata"][identity_field] = (
        "native_execution"
        if identity_field == "job_family"
        else f"different-{identity_field}"
    )
    runtime = _runtime(tmp_path, _FakeAdapter())
    runtime._run_validated_turn = lambda **_kwargs: InvestigationTurnResult(
        result=None,
        thread_id=waiting.codex_thread_id,
        turn_id="turn-accepted-drift",
    )

    with pytest.raises(HarnessError, match="identity"):
        runtime.run_investigation_turn(
            request,
            scope_id=waiting.scope_id,
            active_job=waiting,
        )


@pytest.mark.parametrize(
    "request_kind",
    ["investigation_job_wake", "investigation_job_continuation"],
)
def test_action_execution_recovery_accepts_matching_complete_identity(
    tmp_path: Path,
    request_kind: str,
) -> None:
    """完整 action_execution 身份未漂移时，wake/continuation 正常进入 runtime。"""

    waiting = _action_execution_waiting_state()
    request = _action_execution_recovery_request(waiting, request_kind=request_kind)
    runtime = _runtime(tmp_path, _FakeAdapter())
    runtime._run_validated_turn = lambda **_kwargs: InvestigationTurnResult(
        result=None,
        thread_id=waiting.codex_thread_id,
        turn_id="turn-accepted-match",
    )

    result = runtime.run_investigation_turn(
        request,
        scope_id=waiting.scope_id,
        active_job=waiting,
    )

    assert result.turn_id == "turn-accepted-match"








def test_wake_request_can_carry_complete_scientific_context_after_thread_recovery() -> None:
    """thread history 清空后的 wake 必须带完整科学上下文和旧作业终态通知。"""

    request = build_investigation_job_wake_request(
        "task-1",
        "batch-1",
        "combination-1",
        "thread-wake",
        job_id="job-old",
        operation_id="operation-old",
        submission_key="submission-old",
        action_id="action-old",
        backend_id="backend-old",
        terminal_status="succeeded",
        scientific_question="研究能垒变化。",
        hypothesis={"method": "author", "claim": "路径 A 更低"},
        explicit_conditions={"charge": 0, "multiplicity": 1},
        frozen_context={"paper_snapshot": "frozen-1"},
        calculation_unit={"reactant": "R", "product": "P"},
    )
    payload = json.loads(request["messages"][1]["content"])
    assert payload["scientific_question"] == "研究能垒变化。"
    assert payload["hypothesis"]["claim"] == "路径 A 更低"
    assert payload["explicit_conditions"]["charge"] == 0
    assert payload["frozen_context"]["paper_snapshot"] == "frozen-1"
    assert payload["calculation_unit"]["reactant"] == "R"
    assert payload["job_id"] == "job-old"
    assert payload["terminal_status"] == "succeeded"


def test_combinations_get_different_threads_and_independent_homes(tmp_path: Path) -> None:
    """不同组合必须有不同 thread、workspace 和 CODEX_HOME。"""

    fake = _FakeAdapter()
    runtime = _runtime(tmp_path, fake)
    first = runtime.run_investigation_turn(
        _initial_request(combination_id="combination-1"),
        scope_id="scope-1",
    )
    second = runtime.run_investigation_turn(
        _initial_request(combination_id="combination-2"),
        scope_id="scope-2",
    )

    assert first.thread_id != second.thread_id
    first_store = CodexSessionStore.from_root(
        tmp_path / "sessions",
        owner=CodexSessionOwner.investigation("task-1", "batch-1", "combination-1"),
        scope_id="scope-1",
        model="investigation-model",
    )
    second_store = CodexSessionStore.from_root(
        tmp_path / "sessions",
        owner=CodexSessionOwner.investigation("task-1", "batch-1", "combination-2"),
        scope_id="scope-2",
        model="investigation-model",
    )
    assert first_store.workspace != second_store.workspace
    assert first_store.codex_home != second_store.codex_home
    assert first_store.load()["turn_count"] == 1
    assert second_store.load()["turn_count"] == 1


def test_repair_resumes_the_same_combination_thread(tmp_path: Path) -> None:
    """返修必须使用首轮登记的组合 thread，而不能创建第二个 thread。"""

    fake = _FakeAdapter()
    runtime = _runtime(tmp_path, fake)
    initial = runtime.run_investigation_turn(
        _initial_request(combination_id="combination-1"),
        scope_id="scope-1",
    )
    repair = build_repair_investigation_request(
        "task-1",
        "batch-1",
        "combination-1",
        initial.thread_id,
        "2",
        "补足证据",
        model="investigation-model",
    )
    repaired = runtime.run_investigation_turn(repair, scope_id="scope-1")

    assert repaired.thread_id == initial.thread_id
    assert [call["session_id"] for call in fake.calls] == [None, initial.thread_id]
    assert fake.calls[0]["thread_config"] == fake.calls[1]["thread_config"]
    assert [call["enable_investigation_goals"] for call in fake.calls] == [False, False]


def test_investigation_compact_prompt_reaches_every_session_start_and_resume(
    tmp_path: Path,
) -> None:
    """调查压缩提示词必须覆盖不同 Session 的首次调用和同一 Session 的恢复调用。"""

    sentinel = "ARCHE_INVESTIGATION_COMPACT_SENTINEL"
    fake = _FakeAdapter()
    runtime = _runtime(
        tmp_path,
        fake,
        investigation_compact_prompt=sentinel,
        investigation_model_context_window=1_000_000,
    )
    first = runtime.run_investigation_turn(
        _initial_request(combination_id="combination-1"),
        scope_id="scope-1",
    )
    runtime.run_investigation_turn(
        build_repair_investigation_request(
            "task-1",
            "batch-1",
            "combination-1",
            first.thread_id,
            "result-1",
            "补足证据",
            model="investigation-model",
        ),
        scope_id="scope-1",
    )
    runtime.run_investigation_turn(
        _initial_request(combination_id="combination-2"),
        scope_id="scope-2",
    )

    assert [call["session_id"] for call in fake.calls] == [None, first.thread_id, None]
    assert [call["thread_config"]["compact_prompt"] for call in fake.calls] == [
        sentinel,
        sentinel,
        sentinel,
    ]
    assert [call["thread_config"]["model_context_window"] for call in fake.calls] == [
        1_000_000,
        1_000_000,
        1_000_000,
    ]


def test_investigation_compact_prompt_cannot_be_attached_to_task_review(
    tmp_path: Path,
) -> None:
    """调查专用压缩提示词不能进入 DeepSeek Task Review runtime。"""

    with pytest.raises(HarnessError, match="only available for Investigation"):
        _runtime(
            tmp_path,
            _FakeAdapter(),
            evidence=True,
            investigation_compact_prompt="ARCHE_INVESTIGATION_COMPACT_SENTINEL",
        )


def test_investigation_context_window_cannot_be_attached_to_task_review(
    tmp_path: Path,
) -> None:
    """Investigation 的上下文档位不能进入 DeepSeek Task Review runtime。"""

    with pytest.raises(HarnessError, match="only available for Investigation"):
        _runtime(
            tmp_path,
            _FakeAdapter(),
            evidence=True,
            investigation_model_context_window=1_000_000,
        )


@pytest.mark.parametrize("invalid_window", (True, 0, -1, 1.5, "1000000"))
def test_investigation_context_window_requires_positive_integer(
    tmp_path: Path,
    invalid_window: object,
) -> None:
    """Investigation 上下文档位必须以正整数进入 thread config。"""

    with pytest.raises(HarnessError, match="positive integer"):
        _runtime(
            tmp_path,
            _FakeAdapter(),
            investigation_model_context_window=invalid_window,  # type: ignore[arg-type]
        )


def test_investigation_compact_prompt_accepts_multiline_text(tmp_path: Path) -> None:
    """压缩提示词允许自然的多行结构，并原样进入 Investigation thread config。"""

    prompt = "保留以下信息：\n- 科学假设\n- 证据引用"
    fake = _FakeAdapter()
    runtime = _runtime(
        tmp_path,
        fake,
        investigation_compact_prompt=prompt,
    )

    runtime.run_investigation_turn(
        _initial_request(combination_id="combination-1"),
        scope_id="scope-1",
    )

    assert fake.calls[0]["thread_config"]["compact_prompt"] == prompt


def test_investigation_compact_prompt_change_stops_existing_thread_before_provider(
    tmp_path: Path,
) -> None:
    """恢复时改变压缩提示词必须触发 thread policy 漂移，且不能调用 provider。"""

    first_adapter = _FakeAdapter()
    first_runtime = _runtime(
        tmp_path,
        first_adapter,
        investigation_compact_prompt="ARCHE_COMPACT_POLICY_A",
    )
    first = first_runtime.run_investigation_turn(
        _initial_request(combination_id="combination-1"),
        scope_id="scope-1",
    )
    changed_adapter = _FakeAdapter()
    changed_runtime = _runtime(
        tmp_path,
        changed_adapter,
        investigation_compact_prompt="ARCHE_COMPACT_POLICY_B",
    )
    repair = build_repair_investigation_request(
        "task-1",
        "batch-1",
        "combination-1",
        first.thread_id,
        "result-1",
        "补足证据",
        model="investigation-model",
    )

    with pytest.raises(HarnessError, match="provider/auth policy changed|effective thread policy"):
        changed_runtime.run_investigation_turn(repair, scope_id="scope-1")

    assert changed_adapter.calls == []


def test_investigation_context_window_change_stops_existing_thread_before_provider(
    tmp_path: Path,
) -> None:
    """恢复时改变上下文档位必须在调用 provider 前报告 thread policy 漂移。"""

    first_runtime = _runtime(
        tmp_path,
        _FakeAdapter(),
        investigation_model_context_window=1_000_000,
    )
    first = first_runtime.run_investigation_turn(
        _initial_request(combination_id="combination-1"),
        scope_id="scope-1",
    )
    changed_adapter = _FakeAdapter()
    changed_runtime = _runtime(
        tmp_path,
        changed_adapter,
        investigation_model_context_window=900_000,
    )
    repair = build_repair_investigation_request(
        "task-1",
        "batch-1",
        "combination-1",
        first.thread_id,
        "result-1",
        "补足证据",
        model="investigation-model",
    )

    with pytest.raises(HarnessError, match="provider/auth policy changed|effective thread policy"):
        changed_runtime.run_investigation_turn(repair, scope_id="scope-1")

    assert changed_adapter.calls == []


def test_task_review_reuses_one_thread_across_investigation_contexts(
    tmp_path: Path,
) -> None:
    """任务级 Review 的 thread 不随 Investigation Context 改变。"""

    fake = _FakeAdapter()
    runtime = _runtime(tmp_path, fake, evidence=True)
    first = runtime.run_task_review_turn(
        _review_request(investigation_context_id="batch-1"),
        scope_id="review-scope",
    )
    second = runtime.run_task_review_turn(
        _review_request(investigation_context_id="batch-2"),
        scope_id="review-scope",
    )

    assert first.thread_id == second.thread_id
    assert [call["session_id"] for call in fake.calls] == [None, first.thread_id]
    assert fake.calls[0]["request"].timeout_seconds == 900.0
    assert fake.calls[1]["request"].timeout_seconds == 900.0
    assert "minichem_toolbox" not in fake.calls[0]["thread_config"]["mcp_servers"]
    assert fake.calls[0]["thread_config"]["mcp_servers"]["evidence_reader"]["env_http_headers"] == {
        "Authorization": "EVIDENCE_AUTH_TOKEN",
        "X-Evidence-Scope": "EVIDENCE_SCOPE_TOKEN",
    }
    assert "evidence-token" not in Path(fake.calls[0]["trace_path"]).read_text(encoding="utf-8")
    assert "evidence-scope-secret" not in Path(fake.calls[0]["trace_path"]).read_text(encoding="utf-8")
    assert "Bearer evidence-token" in fake.calls[0]["trace_sensitive_values"]
    assert "evidence-scope-secret" in fake.calls[0]["trace_sensitive_values"]


def test_task_review_process_restart_reuses_persisted_thread(tmp_path: Path) -> None:
    """新 Review runtime 进程从持久 manifest 恢复同一 task thread。"""

    fake = _FakeAdapter()
    first_runtime = _runtime(tmp_path, fake, evidence=True)
    first = first_runtime.run_task_review_turn(
        _review_request(investigation_context_id="batch-1"),
        scope_id="review-scope",
    )

    restarted_runtime = _runtime(tmp_path, fake, evidence=True)
    second = restarted_runtime.run_task_review_turn(
        _review_request(investigation_context_id="batch-2"),
        scope_id="review-scope",
    )

    assert second.thread_id == first.thread_id
    assert [call["session_id"] for call in fake.calls] == [None, first.thread_id]


def test_task_finalization_reuses_review_thread_and_its_own_contract(
    tmp_path: Path,
) -> None:
    """终局成稿使用原 Review thread，但不再接受 Review 返修响应。"""

    fake = _FakeAdapter()
    runtime = _runtime(tmp_path, fake, evidence=True)
    first = runtime.run_task_review_turn(
        _review_request(),
        scope_id="review-scope",
    )
    finalization_request = _finalization_request()
    finalization_request["thread_id"] = first.thread_id
    finalization_request["metadata"]["review_thread_id"] = first.thread_id
    fake.response_content = _finalization_response()

    final = runtime.run_task_review_turn(
        finalization_request,
        scope_id="review-scope",
    )

    assert final.thread_id == first.thread_id
    assert final.result.answerability == "answerable"
    assert fake.calls[1]["session_id"] == first.thread_id
    assert (
        fake.calls[1]["request"].metadata["request_kind"]
        == "task_finalization"
    )


def test_task_finalization_after_process_restart_reuses_review_thread(
    tmp_path: Path,
) -> None:
    """Review 进程退出后，终局成稿仍从 task manifest 恢复原 thread。"""

    fake = _FakeAdapter()
    first_runtime = _runtime(tmp_path, fake, evidence=True)
    first = first_runtime.run_task_review_turn(
        _review_request(),
        scope_id="review-scope",
    )

    restarted_runtime = _runtime(tmp_path, fake, evidence=True)
    fake.response_content = _finalization_response()
    final = restarted_runtime.run_task_review_turn(
        _finalization_request(),
        scope_id="review-scope",
    )

    assert final.thread_id == first.thread_id
    assert fake.calls[1]["session_id"] == first.thread_id
    assert fake.calls[1]["request"].metadata["request_kind"] == "task_finalization"


def test_evidence_mcp_settings_require_capability_header() -> None:
    """HTTP evidence reader 必须在启动 Codex 前携带显式 capability。"""

    with pytest.raises(HarnessError, match="capability header is required"):
        EvidenceMcpSettings(
            endpoint="http://127.0.0.1:9020/mcp",
            http_headers={},
        )


def test_provider_environment_bindings_are_separate(tmp_path: Path) -> None:
    """调查和 Review 的 provider_id、env_key、base URL 不能串用。"""

    fake = _FakeAdapter()
    investigation = _runtime(tmp_path / "investigation", fake)
    review = _runtime(tmp_path / "review", fake, evidence=True)
    investigation.run_investigation_turn(
        _initial_request(combination_id="combination-1"),
        scope_id="s1",
    )
    review.run_task_review_turn(_review_request(), scope_id="s2")

    assert [(call["provider_id"], call["env_key"]) for call in fake.calls] == [
        ("arche_investigation", "ARCHE_CODEX_API_KEY"),
        ("arche_review", "ARCHE_REVIEW_CODEX_API_KEY"),
    ]
    assert fake.calls[0]["thread_config"]["model_providers"]["arche_investigation"]["base_url"] == (
        "https://investigation.example/v1"
    )
    assert fake.calls[0]["thread_config"]["model_providers"]["arche_investigation"][
        "request_max_retries"
    ] == 6
    assert fake.calls[0]["thread_config"]["model_providers"]["arche_investigation"][
        "stream_max_retries"
    ] == 6
    assert fake.calls[1]["thread_config"]["model_providers"]["arche_review"]["base_url"] == (
        "https://review.example/v1"
    )
    assert fake.calls[1]["thread_config"]["model_providers"]["arche_review"][
        "request_max_retries"
    ] == 6
    assert fake.calls[1]["thread_config"]["model_providers"]["arche_review"][
        "stream_max_retries"
    ] == 6


def test_investigation_uses_xhigh_reasoning_without_leaking_it_to_review(
    tmp_path: Path,
) -> None:
    """Investigation thread 固定带 xhigh，Task Review 不继承调查推理配置。"""

    fake = _FakeAdapter()
    investigation = _runtime(tmp_path / "investigation", fake)
    review = _runtime(tmp_path / "review", fake, evidence=True)

    investigation.run_investigation_turn(
        _initial_request(combination_id="combination-1"),
        scope_id="investigation-scope",
    )
    review.run_task_review_turn(_review_request(), scope_id="review-scope")

    investigation_config = fake.calls[0]["thread_config"]
    review_config = fake.calls[1]["thread_config"]
    assert investigation_config["model_reasoning_effort"] == "xhigh"
    assert "model_reasoning_effort" not in review_config


def test_production_investigation_start_and_resume_keep_sol_xhigh_identity(
    tmp_path: Path,
) -> None:
    """生产 Investigation 的 start/resume 都保持 Sol 与 xhigh，不因续会话漂移。"""

    fake = _FakeAdapter()
    runtime = _runtime(
        tmp_path,
        fake,
        model="gpt-5.6-sol",
        allow_test_doubles=False,
    )
    initial = runtime.run_investigation_turn(
        _initial_request(combination_id="combination-1", model="gpt-5.6-sol"),
        scope_id="production-scope",
    )
    repair = build_repair_investigation_request(
        "task-1",
        "batch-1",
        "combination-1",
        initial.thread_id,
        "result-1",
        "补足证据",
        model="gpt-5.6-sol",
    )
    runtime.run_investigation_turn(repair, scope_id="production-scope")

    assert [call["request"].model for call in fake.calls] == [
        "gpt-5.6-sol",
        "gpt-5.6-sol",
    ]
    assert [call["session_id"] for call in fake.calls] == [None, initial.thread_id]
    assert [call["thread_config"]["model_reasoning_effort"] for call in fake.calls] == [
        "xhigh",
        "xhigh",
    ]


def test_investigation_reasoning_effort_rejects_unknown_environment_value(
    tmp_path: Path,
) -> None:
    """调查 provider 的独立推理强度环境值非法时必须在模型调用前拒绝。"""

    fake = _FakeAdapter()
    runtime = _runtime(tmp_path, fake)
    settings = runtime.investigation_provider
    assert settings.reasoning_effort_env == "ARCHE_CODEX_REASONING_EFFORT"
    runtime._environment[settings.reasoning_effort_env] = "unsupported"

    with pytest.raises(HarnessError, match="reasoning effort"):
        runtime.run_investigation_turn(
            _initial_request(combination_id="combination-1"),
            scope_id="investigation-scope",
        )
    assert fake.calls == []


@pytest.mark.parametrize(
    ("model", "reasoning_effort"),
    (
        ("gpt-5.6-luna", "xhigh"),
        ("custom-codex-model", "high"),
    ),
)
def test_production_investigation_runtime_accepts_configured_model_and_effort(
    tmp_path: Path,
    model: str,
    reasoning_effort: str,
) -> None:
    """生产 runtime 使用 provider 配置的模型和推理强度，不绑定历史默认值。"""

    fake = _FakeAdapter()
    runtime = _runtime(
        tmp_path,
        fake,
        model=model,
        reasoning_effort=reasoning_effort,
        allow_test_doubles=False,
    )

    result = runtime.run_investigation_turn(
        _initial_request(combination_id="combination-1", model=model),
        scope_id="production-scope",
    )
    assert result.result is not None
    assert fake.calls[0]["request"].model == model
    assert fake.calls[0]["thread_config"]["model_reasoning_effort"] == reasoning_effort


def test_explicit_test_double_boundary_allows_arbitrary_investigation_model(
    tmp_path: Path,
) -> None:
    """任意 Investigation 模型只在显式 test-double 边界内可用。"""

    fake = _FakeAdapter()
    runtime = _runtime(
        tmp_path,
        fake,
        model="test-only-model",
        allow_test_doubles=True,
    )
    result = runtime.run_investigation_turn(
        _initial_request(combination_id="combination-1", model="test-only-model"),
        scope_id="test-scope",
    )

    assert result.thread_id == "thread-1"
    assert fake.calls[0]["thread_config"]["model_reasoning_effort"] == "xhigh"


def test_minichem_header_is_static_config_only(tmp_path: Path) -> None:
    """MiniChem token 只进入 app-server 私有环境，不进入 MCP config、trace 或返回值。"""

    fake = _FakeAdapter()
    runtime = _runtime(tmp_path, fake)
    result = runtime.run_investigation_turn(_initial_request(combination_id="combination-1"), scope_id="s1")

    call = fake.calls[0]
    token = "capability-secret"
    assert call["thread_config"]["mcp_servers"]["minichem_toolbox"]["env_http_headers"] == {
        "X-MiniChem-Capability": "MINICHEM_CAPABILITY_TOKEN"
    }
    assert call["secret_environment"]["MINICHEM_CAPABILITY_TOKEN"] == token
    assert token not in call["request"].prompt_text()
    assert call["secret_environment"]["ARCHE_CODEX_API_KEY"] == "investigation-secret"
    assert token not in json.dumps(call["request"].metadata)
    assert token not in (Path(call["trace_path"]).read_text(encoding="utf-8"))
    assert "investigation-secret" not in Path(call["trace_path"]).read_text(encoding="utf-8")
    manifest = Path(
        tmp_path
        / "sessions/investigations/task-1/batch-1/combination-1/state/.arche_codex_session.json"
    ).read_text(encoding="utf-8")
    assert token not in manifest
    assert token not in repr(result)


def test_all_evidence_header_values_are_rejected_from_request_and_response(tmp_path: Path) -> None:
    """evidence header 值不能进入请求字段、Codex response 或返回对象。"""

    fake = _FakeAdapter()
    runtime = _runtime(tmp_path, fake, evidence=True)
    request = _review_request()
    request["messages"][0]["content"] += " Bearer evidence-token"
    with pytest.raises(HarnessError, match="MCP credential"):
        runtime.run_task_review_turn(request, scope_id="review-scope")
    assert fake.calls == []

    fake.response_content = _review_response()[:-1] + (
        ', "note": "Bearer evidence-token; scope=evidence-scope-secret"}'
    )
    with pytest.raises(HarnessError, match="MCP credential"):
        runtime.run_task_review_turn(_review_request(), scope_id="review-scope")
    assert "evidence-token" not in Path(fake.calls[0]["trace_path"]).read_text(encoding="utf-8")
    assert "evidence-scope-secret" not in Path(fake.calls[0]["trace_path"]).read_text(encoding="utf-8")
    review_store = CodexSessionStore.from_root(
        tmp_path / "sessions",
        owner=CodexSessionOwner.task_review("task-1"),
        scope_id="review-scope",
        model="review-model",
    )
    assert review_store.load()["turn_count"] == 1


def test_first_thread_is_persisted_before_adapter_turn(tmp_path: Path) -> None:
    """adapter callback 返回后、模拟 turn 前必须已经有 count=0 manifest。"""

    observed: list[dict[str, object]] = []

    def observe(**kwargs):
        kwargs["on_thread_ready"]("thread-observed")
        manifest = json.loads(
            (
                tmp_path
                / "sessions/investigations/task-1/batch-1/combination-1/state/.arche_codex_session.json"
            ).read_text(encoding="utf-8")
        )
        observed.append(manifest)
        # 继续完成一次合法 mock turn，但不再次调用 callback。
        request = kwargs["request"]
        _write_test_completed_receipt(
            kwargs,
            thread_id="thread-observed",
            turn_id="turn-observed",
        )
        return CodexSdkRunResult(
            response=HarnessResponse(
                content=_investigation_response(),
                model=request.model,
                metadata={"turn_id": "turn-observed"},
            ),
            thread_id="thread-observed",
            turn_id="turn-observed",
        )

    runtime = _runtime(tmp_path, observe)
    runtime.run_investigation_turn(_initial_request(combination_id="combination-1"), scope_id="s1")

    assert observed[0]["thread_id"] == "thread-observed"
    assert observed[0]["turn_count"] == 0


def test_external_thread_ready_hook_runs_after_manifest_before_turn(tmp_path: Path) -> None:
    """外部 thread hook 看到真实 manifest 后、模型 turn 前执行。"""

    events: list[tuple[str, str | bool]] = []

    def observe(**kwargs):
        kwargs["on_thread_ready"]("thread-bound")
        manifest_path = (
            tmp_path
            / "sessions/investigations/task-1/batch-1/combination-1/state/.arche_codex_session.json"
        )
        events.append(("adapter_after_hook", manifest_path.exists()))
        events.append(("thread", "thread-bound"))
        request = kwargs["request"]
        _write_test_completed_receipt(
            kwargs,
            thread_id="thread-bound",
            turn_id="turn-bound",
        )
        return CodexSdkRunResult(
            response=HarnessResponse(
                content=_investigation_response(),
                model=request.model,
                metadata={"turn_id": "turn-bound"},
            ),
            thread_id="thread-bound",
            turn_id="turn-bound",
        )

    runtime = _runtime(tmp_path, observe)
    runtime.run_investigation_turn(
        _initial_request(combination_id="combination-1"),
        scope_id="s1",
        on_thread_ready=lambda thread_id: events.insert(0, ("external", thread_id)),
    )

    assert events == [
        ("external", "thread-bound"),
        ("adapter_after_hook", True),
        ("thread", "thread-bound"),
    ]


def test_wm_thread_preflight_blocks_replacement_before_manifest_write(tmp_path: Path) -> None:
    """WM 已有 thread 且本地 manifest 缺失时，新 SDK thread 不得写入本地状态。"""

    fake = _FakeAdapter()
    runtime = _runtime(tmp_path, fake)
    with pytest.raises(HarnessError, match="cannot start a missing Codex session"):
        runtime.run_investigation_turn(
            _initial_request(
                combination_id="combination-1",
                codex_thread_id="thread-existing",
            ),
            scope_id="scope-1",
        )
    assert fake.calls == []
    manifest_path = (
        tmp_path
        / "sessions/investigations/task-1/batch-1/combination-1/state/.arche_codex_session.json"
    )
    assert not manifest_path.exists()


def test_invalid_structured_response_is_rejected_after_completed_count(tmp_path: Path) -> None:
    """结构化 JSON 无效时记录合同失败，manifest 仍保留真实完成计数。"""

    fake = _FakeAdapter()
    fake.invalid_response = True
    runtime = _runtime(tmp_path, fake)

    with pytest.raises(HarnessError, match="structured contract"):
        runtime.run_investigation_turn(_initial_request(combination_id="combination-1"), scope_id="s1")

    store = CodexSessionStore.from_root(
        tmp_path / "sessions",
        owner=CodexSessionOwner.investigation("task-1", "batch-1", "combination-1"),
        scope_id="s1",
        model="investigation-model",
    )
    assert store.load()["turn_count"] == 1
    assert store.load()["status"] == "contract_failure"
    assert store.load_recovery()["reason"] == "response_contract"


def test_success_without_completed_receipt_fails_closed(tmp_path: Path) -> None:
    """成功响应缺少 completed receipt 时必须阻断，不能静默返回结果。"""

    fake = _FakeAdapter()
    fake.write_receipt = False
    runtime = _runtime(tmp_path, fake)

    with pytest.raises(HarnessError, match="no durable receipt"):
        runtime.run_investigation_turn(
            _initial_request(combination_id="combination-1"),
            scope_id="s1",
        )
    assert len(fake.calls) == 1


def test_contract_failure_keeps_receipt_blocks_replay_and_allows_explicit_new_request(
    tmp_path: Path,
) -> None:
    """坏正文留下 receipt；同一请求被阻断，显式恢复后只能提交新请求。"""

    fake = _FakeAdapter()
    fake.invalid_response = True
    fake.write_receipt = True
    runtime = _runtime(tmp_path, fake)
    initial = _initial_request(combination_id="combination-1")

    with pytest.raises(HarnessError, match="structured contract"):
        runtime.run_investigation_turn(initial, scope_id="s1")

    store = CodexSessionStore.from_root(
        tmp_path / "sessions",
        owner=CodexSessionOwner.investigation("task-1", "batch-1", "combination-1"),
        scope_id="s1",
        model="investigation-model",
    )
    recovery = store.load_recovery()
    assert recovery is not None
    receipt = store.manifest_path.parent / "completed_turn_receipts" / "turn-1.json"
    assert receipt.is_file()

    fake.invalid_response = False
    with pytest.raises(HarnessError, match="explicit recovery"):
        runtime.run_investigation_turn(initial, scope_id="s1")
    assert len(fake.calls) == 1
    assert receipt.is_file()

    store.begin_contract_failure_recovery(
        request_digest=str(recovery["request_digest"]),
        thread_policy_digest=str(recovery["thread_policy_digest"]),
        turn_id=str(recovery["turn_id"]),
    )
    repair = build_repair_investigation_request(
        "task-1",
        "batch-1",
        "combination-1",
        "thread-1",
        "2",
        "补足结构化证据",
        model="investigation-model",
    )
    result = runtime.run_investigation_turn(repair, scope_id="s1")

    assert result.thread_id == "thread-1"
    assert len(fake.calls) == 2
    assert store.load()["status"] == "completed"
    assert store.load()["turn_count"] == 2
    assert store.load_recovery() is None
    assert receipt.is_file()


def test_task_review_contract_failure_uses_the_same_persistent_block(tmp_path: Path) -> None:
    """任务 Review 的坏 JSON 也必须进入持久合同失败状态而非重复 replay。"""

    fake = _FakeAdapter()
    fake.invalid_response = True
    runtime = _runtime(tmp_path, fake, evidence=True)
    request = _review_request()

    with pytest.raises(HarnessError, match="structured contract"):
        runtime.run_task_review_turn(request, scope_id="review-scope")

    store = CodexSessionStore.from_root(
        tmp_path / "sessions",
        owner=CodexSessionOwner.task_review("task-1"),
        scope_id="review-scope",
        model="review-model",
    )
    assert store.load()["status"] == "contract_failure"
    with pytest.raises(HarnessError, match="explicit recovery"):
        runtime.run_task_review_turn(request, scope_id="review-scope")
    assert len(fake.calls) == 1


def test_request_digest_covers_non_prompt_model_request_fields() -> None:
    """receipt 摘要必须随 schema、工具、采样参数和 metadata 的变化而变化。"""

    first = HarnessRequest(
        messages=({"role": "user", "content": "same"},),
        model="model",
        role="investigation",
        stage="initial_investigation",
        research_task_id="research-task-1",
        invocation_id="initial_investigation",
        max_tokens=100,
        temperature=0.1,
        top_p=0.9,
        timeout_seconds=None,
        response_schema={"type": "object"},
        tools=({"name": "tool-a"},),
        metadata={"request_kind": "initial_investigation"},
        profile="investigation",
    )
    changed = HarnessRequest(
        messages=first.messages,
        model=first.model,
        role=first.role,
        stage=first.stage,
        research_task_id=first.research_task_id,
        invocation_id=first.invocation_id,
        max_tokens=101,
        temperature=first.temperature,
        top_p=first.top_p,
        timeout_seconds=first.timeout_seconds,
        response_schema=first.response_schema,
        tools=first.tools,
        metadata=first.metadata,
        profile=first.profile,
    )

    assert codex_runtime_module._calculate_request_digest(first) != codex_runtime_module._calculate_request_digest(
        changed
    )


def test_request_digest_ignores_only_codex_thread_route_metadata() -> None:
    """首次请求绑定 thread 前后摘要稳定，其他科学与请求元数据仍参与摘要。"""

    request = HarnessRequest(
        messages=({"role": "user", "content": "same"},),
        model="model",
        role="investigation",
        stage="initial_investigation",
        research_task_id="research-task-1",
        invocation_id="initial_investigation",
        timeout_seconds=None,
        response_schema={"type": "object"},
        metadata={
            "request_kind": "initial_investigation",
            "hypothesis_execution_input_record_id": "combination-1",
        },
        profile="investigation",
    )
    routed = HarnessRequest(
        messages=request.messages,
        model=request.model,
        role=request.role,
        stage=request.stage,
        research_task_id=request.research_task_id,
        invocation_id=request.invocation_id,
        timeout_seconds=request.timeout_seconds,
        response_schema=request.response_schema,
        metadata={**request.metadata, "codex_thread_id": "thread-1"},
        profile=request.profile,
    )
    changed = HarnessRequest(
        messages=request.messages,
        model=request.model,
        role=request.role,
        stage=request.stage,
        research_task_id=request.research_task_id,
        invocation_id=request.invocation_id,
        timeout_seconds=request.timeout_seconds,
        response_schema=request.response_schema,
        metadata={
            **request.metadata,
            "hypothesis_execution_input_record_id": "combination-2",
        },
        profile=request.profile,
    )

    for digest in (
        codex_runtime_module._calculate_request_digest,
        codex_sdk_adapter.calculate_codex_request_digest,
    ):
        assert digest(request) == digest(routed)
        assert digest(request) != digest(changed)


def test_failed_turn_keeps_trace_and_retries_same_thread_with_new_attempt_path(tmp_path: Path) -> None:
    """completed 前失败不会占用同一 turn_count 的路径，重试仍恢复原 thread。"""

    fake = _FakeAdapter()
    fake.fail_first = True
    runtime = _runtime(tmp_path, fake)
    request = _initial_request(combination_id="combination-1")

    with pytest.raises(HarnessError, match="mock SDK turn failed"):
        runtime.run_investigation_turn(request, scope_id="s1")

    first_trace = Path(fake.calls[0]["trace_path"])
    assert first_trace.name == "initial_investigation.0.jsonl"
    assert first_trace.is_file()
    assert "capability-secret" not in first_trace.read_text(encoding="utf-8")

    result = runtime.run_investigation_turn(request, scope_id="s1")

    second_trace = Path(fake.calls[1]["trace_path"])
    assert fake.calls[1]["session_id"] == result.thread_id == "thread-1"
    assert second_trace.name == "initial_investigation.0.attempt-1.jsonl"
    assert first_trace.read_text(encoding="utf-8") != second_trace.read_text(encoding="utf-8")
    assert "capability-secret" not in second_trace.read_text(encoding="utf-8")
    store = CodexSessionStore.from_root(
        tmp_path / "sessions",
        owner=CodexSessionOwner.investigation("task-1", "batch-1", "combination-1"),
        scope_id="s1",
        model="investigation-model",
    )
    assert store.load()["turn_count"] == 1


def test_request_intent_recovery_persists_original_intent_digest(tmp_path: Path) -> None:
    """intent mismatch 必须保存旧 intent 摘要，不能把当前请求摘要冒充原身份。"""

    original_digest = "a" * 64

    class IntentMismatchAdapter:
        def __call__(self, **kwargs):
            kwargs["on_thread_ready"]("thread-intent-mismatch")
            kwargs["trace_path"].write_text("intent mismatch", encoding="utf-8")
            error = codex_sdk_adapter.CodexRequestIntentRecoveryPending(
                "request intent does not match the current request",
                turn_id=None,
                reason="request_intent",
                request_digest=original_digest,
            )
            raise error

    runtime = _runtime(tmp_path, IntentMismatchAdapter())
    with pytest.raises(HarnessError, match="request intent") as failure:
        runtime.run_investigation_turn(
            _initial_request(combination_id="combination-1"),
            scope_id="scope-1",
        )

    store = CodexSessionStore.from_root(
        tmp_path / "sessions",
        owner=CodexSessionOwner.investigation("task-1", "batch-1", "combination-1"),
        scope_id="scope-1",
        model="investigation-model",
    )
    recovery = store.load_recovery()
    assert failure.value.kind == "recovery_pending"
    assert recovery is not None
    assert recovery["request_digest"] == original_digest


def test_runtime_preserves_terminal_turn_error_identity(tmp_path: Path) -> None:
    """runtime 脱敏后保留公开 terminal 类型、turn_id 和 status，不携带 provider 正文。"""

    class TerminalAdapter:
        def __call__(self, **kwargs):
            kwargs["on_thread_ready"]("thread-terminal")
            kwargs["trace_path"].write_text(
                "terminal provider detail [REDACTED]",
                encoding="utf-8",
            )
            raise codex_sdk_adapter.CodexTerminalTurnError(
                turn_id="turn-terminal",
                status="failed",
            )

    runtime = _runtime(tmp_path, TerminalAdapter())
    with pytest.raises(codex_sdk_adapter.CodexTerminalTurnError) as failure:
        runtime.run_investigation_turn(
            _initial_request(combination_id="combination-1"),
            scope_id="scope-1",
        )

    assert failure.value.turn_id == "turn-terminal"
    assert failure.value.status == "failed"
    assert "provider detail" not in str(failure.value)


def test_initial_investigation_terminal_error_uses_durable_turn_rollback(
    tmp_path: Path,
) -> None:
    """initial Investigation 的白名单 provider 错误也先回滚原物理 thread。"""

    class RollbackClient:
        def __init__(self, thread_id: str) -> None:
            self.thread_id = thread_id
            self.calls: list[tuple[str, object]] = []

        def request(self, method, params, *, response_model):
            self.calls.append((method, params))
            return {"thread": {"id": self.thread_id, "turns": []}}

    class RollbackThread:
        id = "thread-initial-rollback"

        def __init__(self) -> None:
            self._client = RollbackClient(self.id)

        def read(self, *, include_turns=False):
            assert include_turns is True
            return {"thread": {"id": self.id, "turns": [{"id": "failed-1", "status": "failed"}]}}

    class InitialRollbackAdapter:
        def __init__(self) -> None:
            self.thread = RollbackThread()

        def __call__(self, **kwargs):
            kwargs["on_thread_ready"](self.thread.id)
            kwargs["on_turn_started"]("turn-initial-rollback")
            kwargs["on_terminal_turn"](
                self.thread.id,
                "turn-initial-rollback",
                "failed",
            )
            kwargs["on_investigation_rollback"](
                self.thread.id,
                self.thread,
                "invalid_encrypted_content",
            )
            raise codex_sdk_adapter.CodexTerminalTurnError(
                turn_id="turn-initial-rollback",
                status="failed",
                error_code="invalid_encrypted_content",
            )

    adapter = InitialRollbackAdapter()
    runtime = _runtime(tmp_path, adapter)
    with pytest.raises(codex_sdk_adapter.CodexTerminalTurnError) as failure:
        runtime.run_investigation_turn(
            _initial_request(combination_id="combination-1"),
            scope_id="scope-1",
            on_turn_started=lambda *_identity: None,
            on_terminal_turn=lambda *_identity: "terminal-event",
        )
    assert failure.value.error_code == "invalid_encrypted_content"
    store = CodexSessionStore.from_root(
        tmp_path / "sessions",
        owner=CodexSessionOwner.investigation("task-1", "batch-1", "combination-1"),
        scope_id="scope-1",
        model="investigation-model",
    )
    receipt = store.load_thread_rollback_receipt()
    assert receipt is not None
    assert receipt["stage"] == "initial_investigation"
    assert adapter.thread._client.calls == [
        ("thread/rollback", {"threadId": adapter.thread.id, "numTurns": 1})
    ]


def test_legacy_initial_failure_rolls_back_before_first_recovery_retry(
    tmp_path: Path,
) -> None:
    """补丁前 trace 的初始失败必须在新模型 turn 前清空，而不是带旧历史重试。"""

    class RollbackClient:
        def __init__(self, thread_id: str) -> None:
            self.thread_id = thread_id
            self.calls: list[tuple[str, object]] = []

        def request(self, method, params, *, response_model):
            self.calls.append((method, params))
            return {"thread": {"id": self.thread_id, "turns": []}}

    class LegacyThread:
        id = "thread-legacy-initial"

        def __init__(self) -> None:
            self._client = RollbackClient(self.id)

        def read(self, *, include_turns=False):
            assert include_turns is True
            return {
                "thread": {
                    "id": self.id,
                    "turns": [{"id": "turn-legacy-failed", "status": "failed"}],
                }
            }

    class LegacyRecoveryAdapter:
        def __init__(self) -> None:
            self.thread = LegacyThread()
            self.calls: list[dict[str, object]] = []

        def __call__(self, **kwargs):
            self.calls.append(dict(kwargs))
            kwargs["on_thread_ready"](self.thread.id)
            kwargs["before_new_turn_with_thread"](self.thread, self.thread.id)
            if "on_turn_started" in kwargs:
                kwargs["on_turn_started"]("turn-retry")
            _write_test_completed_receipt(
                kwargs,
                thread_id=self.thread.id,
                turn_id="turn-retry",
            )
            return CodexSdkRunResult(
                response=HarnessResponse(
                    content=_investigation_response(),
                    model=kwargs["request"].model,
                    metadata={"turn_id": "turn-retry"},
                ),
                thread_id=self.thread.id,
                turn_id="turn-retry",
            )

    class SetupAdapter:
        def __call__(self, **kwargs):
            """先建立带真实 thread policy 的旧 manifest，再模拟进程退出。"""

            kwargs["on_thread_ready"]("thread-legacy-initial")
            raise HarnessError("legacy process exited", kind="transport")

    runtime = _runtime(tmp_path, SetupAdapter())
    store = CodexSessionStore.from_root(
        tmp_path / "sessions",
        owner=CodexSessionOwner.investigation("task-1", "batch-1", "combination-1"),
        scope_id="scope-1",
        model="investigation-model",
    )
    with pytest.raises(HarnessError, match="legacy process exited"):
        runtime.run_investigation_turn(
            _initial_request(combination_id="combination-1"),
            scope_id="scope-1",
        )
    trace_directory = (
        tmp_path
        / "traces"
        / "investigation"
        / "task-1"
        / "batch-1"
        / "combination-1"
    )
    trace_directory.mkdir(parents=True, exist_ok=True)
    (trace_directory / "initial_investigation.0.jsonl").write_text(
        json.dumps(
            {
                "type": "codex_notification",
                "method": "turn/completed",
                "params": {
                    "thread_id": "thread-legacy-initial",
                    "turn_id": "turn-legacy-failed",
                    "status": "failed",
                    "error_code": "invalid_encrypted_content",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    active_job = InvestigationJobState(
        task_id="task-1",
        batch_id="batch-1",
        combination_id="combination-1",
        codex_thread_id="thread-legacy-initial",
        turn_id="turn-legacy-failed",
        status="waiting_investigation_job",
        identity=InvestigationJobIdentity(
            job_id="job-legacy",
            operation_id="operation-legacy",
            submission_key="submission-legacy",
            action_id="action-legacy",
            backend_id="backend-legacy",
        ),
        request_digest="legacy-request-digest",
        scope_id="scope-1",
    )
    recovery_adapter = LegacyRecoveryAdapter()
    runtime._adapter = recovery_adapter

    assert runtime.has_recoverable_initial_investigation_failure(
        research_task_id="task-1",
        investigation_context_id="batch-1",
        hypothesis_execution_input_record_id="combination-1",
        scope_id="scope-1",
        thread_id="thread-legacy-initial",
        turn_id="turn-legacy-failed",
    ) is True
    result = runtime.run_investigation_turn(
        _initial_request(
            combination_id="combination-1",
            codex_thread_id="thread-legacy-initial",
        ),
        scope_id="scope-1",
        active_job=active_job,
        recover_existing_terminal_turn=True,
    )
    assert result.turn_id == "turn-retry"
    assert recovery_adapter.calls[0]["before_new_turn_with_thread"]
    assert recovery_adapter.thread._client.calls == [
        (
            "thread/rollback",
            {"threadId": "thread-legacy-initial", "numTurns": 1},
        )
    ]
    receipt = store.load_thread_rollback_receipt()
    assert receipt is not None
    assert receipt["provider_error_code"] == "invalid_encrypted_content"


def test_terminal_recovery_replays_ack_without_poison_or_new_turn(
    tmp_path: Path,
) -> None:
    """终态 recovery 在 hook 缺失时持续阻断，显式 ack 后幂等收束。"""

    class TerminalRecoveryAdapter:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def __call__(self, **kwargs):
            self.calls.append(dict(kwargs))
            kwargs["on_thread_ready"]("thread-terminal-recovery")
            kwargs["trace_path"].write_text("terminal recovery", encoding="utf-8")
            if "on_terminal_turn" not in kwargs:
                raise codex_sdk_adapter.CodexTerminalTurnRecoveryPending(
                    "terminal turn requires acknowledgement",
                    turn_id="turn-terminal-recovery",
                    status="failed",
                    request_digest=kwargs["request_digest"],
                )
            kwargs["on_terminal_turn"](
                "thread-terminal-recovery",
                "turn-terminal-recovery",
                "failed",
            )
            raise codex_sdk_adapter.CodexTerminalTurnError(
                turn_id="turn-terminal-recovery",
                status="failed",
            )

    fake = TerminalRecoveryAdapter()
    runtime = _runtime(tmp_path, fake)
    request = _initial_request(combination_id="combination-1")
    for _ in range(2):
        with pytest.raises(
            codex_sdk_adapter.CodexTerminalTurnRecoveryPending
        ) as pending:
            runtime.run_investigation_turn(request, scope_id="scope-1")
        assert pending.value.reason == "terminal_turn"

    store = CodexSessionStore.from_root(
        tmp_path / "sessions",
        owner=CodexSessionOwner.investigation(
            "task-1",
            "batch-1",
            "combination-1",
        ),
        scope_id="scope-1",
        model="investigation-model",
    )
    recovery = store.load_recovery()
    assert recovery is not None
    assert recovery["reason"] == "terminal_turn"
    assert store.load_recovery_poison() is None
    assert fake.calls[1]["completion_recovery"]["reason"] == "terminal_turn"

    acknowledged: list[tuple[str, str, str]] = []
    with pytest.raises(codex_sdk_adapter.CodexTerminalTurnError) as terminal:
        runtime.run_investigation_turn(
            request,
            scope_id="scope-1",
            on_terminal_turn=lambda *identity: acknowledged.append(identity),
        )
    assert terminal.value.turn_id == "turn-terminal-recovery"
    assert terminal.value.status == "failed"
    assert acknowledged == [
        ("thread-terminal-recovery", "turn-terminal-recovery", "failed")
    ]
    assert len(fake.calls) == 3
    assert store.load_recovery() is None
    assert store.load()["turn_count"] == 0
    assert store.load()["status"] == "created"


def test_unstoppable_worker_persists_recovery_pending_and_blocks_next_turn(tmp_path: Path) -> None:
    """worker 无法停止时持久化 recovery_pending，恢复前不再次调用 adapter。"""

    class _UnstoppableAdapter:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, **kwargs):
            self.calls += 1
            kwargs["on_thread_ready"]("thread-1")
            kwargs["trace_path"].write_text("worker-recovery", encoding="utf-8")
            raise CodexWorkerRecoveryPending(
                "worker remains alive",
                turn_id="turn-1",
                reason="worker_cleanup",
            )

    fake = _UnstoppableAdapter()
    runtime = _runtime(tmp_path, fake)
    request = _initial_request(combination_id="combination-1")

    with pytest.raises(HarnessError, match="worker remains alive"):
        runtime.run_investigation_turn(request, scope_id="s1")

    store = CodexSessionStore.from_root(
        tmp_path / "sessions",
        owner=CodexSessionOwner.investigation("task-1", "batch-1", "combination-1"),
        scope_id="s1",
        model="investigation-model",
    )
    recovery = store.load_recovery()
    assert store.load()["status"] == "recovery_pending"
    assert recovery["reason"] == "worker_cleanup"
    assert recovery["action"] == "worker_stop_confirmation_required"

    with pytest.raises(HarnessError, match="stop confirmation"):
        runtime.run_investigation_turn(request, scope_id="s1")
    assert fake.calls == 1


def test_double_recovery_persistence_failure_writes_durable_poison_and_blocks_restart(
    tmp_path: Path, monkeypatch
) -> None:
    """adapter callback 与 runtime 二次持久化都失败时，poison marker 仍阻断恢复。"""

    class _WorkerRecoveryAdapter:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, **kwargs):
            self.calls += 1
            kwargs["on_thread_ready"]("thread-poison")
            kwargs["trace_path"].write_text("poison", encoding="utf-8")
            raise CodexWorkerRecoveryPending(
                "worker remains alive",
                turn_id="turn-poison",
                reason="worker_cleanup",
            )

    fake = _WorkerRecoveryAdapter()
    runtime = _runtime(tmp_path, fake)
    request = _initial_request(combination_id="combination-1")
    original_record = CodexSessionStore.record_recovery_pending

    def fail_recovery_persistence(*_args, **_kwargs):
        raise HarnessError("simulated recovery persistence failure", kind="session_state")

    monkeypatch.setattr(CodexSessionStore, "record_recovery_pending", fail_recovery_persistence)
    with pytest.raises(HarnessError) as failure:
        runtime.run_investigation_turn(request, scope_id="s1")
    assert failure.value.kind == "recovery_pending"

    monkeypatch.setattr(CodexSessionStore, "record_recovery_pending", original_record)
    store = CodexSessionStore.from_root(
        tmp_path / "sessions",
        owner=CodexSessionOwner.investigation("task-1", "batch-1", "combination-1"),
        scope_id="s1",
        model="investigation-model",
    )
    poison = store.load_recovery_poison()
    assert poison is not None
    assert poison["reason"] == "worker_cleanup"
    assert poison["turn_id"] == "turn-poison"

    with pytest.raises(HarnessError, match="poison") as blocked:
        runtime.run_investigation_turn(request, scope_id="s1")
    assert blocked.value.kind == "recovery_pending"
    assert fake.calls == 1


def test_receipt_recovery_pending_is_persisted_and_replayed_without_model_retry(
    tmp_path: Path,
) -> None:
    """receipt 未确认落盘时上层保留 recovery_pending，第二次只允许严格恢复。"""

    class _CompletionRecoveryAdapter:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def __call__(self, **kwargs):
            self.calls.append(dict(kwargs))
            kwargs["on_thread_ready"]("thread-completion")
            kwargs["trace_path"].write_text("completion-recovery", encoding="utf-8")
            if len(self.calls) == 1:
                codex_sdk_adapter._write_completed_turn_receipt(
                    kwargs["completed_turn_receipt_root"],
                    thread_id="thread-completion",
                    turn_id="turn-completion",
                    request_digest=kwargs["request_digest"],
                    turn_count_before=kwargs["turn_count_before"],
                    thread_policy_digest=kwargs["thread_policy_digest"],
                )
                raise CodexCompletionRecoveryPending(
                    "receipt pending",
                    turn_id="turn-completion",
                    reason="completion_receipt",
                )
            request = kwargs["request"]
            return CodexSdkRunResult(
                response=HarnessResponse(
                    content=_investigation_response(),
                    model=request.model,
                    metadata={"turn_id": "turn-completion"},
                ),
                thread_id="thread-completion",
                turn_id="turn-completion",
                replayed=True,
                receipt_turn_count_before=0,
                request_digest=kwargs["request_digest"],
            )

    fake = _CompletionRecoveryAdapter()
    runtime = _runtime(tmp_path, fake)
    request = _initial_request(combination_id="combination-1")

    with pytest.raises(HarnessError, match="receipt pending"):
        runtime.run_investigation_turn(request, scope_id="s1")

    store = CodexSessionStore.from_root(
        tmp_path / "sessions",
        owner=CodexSessionOwner.investigation("task-1", "batch-1", "combination-1"),
        scope_id="s1",
        model="investigation-model",
    )
    recovery = store.load_recovery()
    assert store.load()["status"] == "recovery_pending"
    assert recovery["reason"] == "completion_receipt"
    assert recovery["action"] == "receipt_replay_required"

    result = runtime.run_investigation_turn(request, scope_id="s1")
    assert result.turn_id == "turn-completion"
    assert len(fake.calls) == 2
    assert store.load()["turn_count"] == 1
    assert store.load_recovery() is None


def test_invalid_replayed_response_transitions_receipt_recovery_to_contract_failure(
    tmp_path: Path,
) -> None:
    """receipt 重放得到坏 JSON 时转为持久合同失败，不重复调用模型或 replay。"""

    class _InvalidReplayAdapter:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def __call__(self, **kwargs):
            self.calls.append(dict(kwargs))
            kwargs["on_thread_ready"]("thread-invalid-replay")
            kwargs["trace_path"].write_text("invalid-replay", encoding="utf-8")
            request = kwargs["request"]
            if len(self.calls) == 1:
                codex_sdk_adapter._write_completed_turn_receipt(
                    kwargs["completed_turn_receipt_root"],
                    thread_id="thread-invalid-replay",
                    turn_id="turn-invalid-replay",
                    request_digest=kwargs["request_digest"],
                    turn_count_before=kwargs["turn_count_before"],
                    thread_policy_digest=kwargs["thread_policy_digest"],
                )
                raise CodexCompletionRecoveryPending(
                    "receipt pending",
                    turn_id="turn-invalid-replay",
                    reason="completion_receipt",
                )
            return CodexSdkRunResult(
                response=HarnessResponse(
                    content='{"invalid":true}',
                    model=request.model,
                    metadata={"turn_id": "turn-invalid-replay"},
                ),
                thread_id="thread-invalid-replay",
                turn_id="turn-invalid-replay",
                replayed=True,
                receipt_turn_count_before=0,
                request_digest=kwargs["request_digest"],
            )

    fake = _InvalidReplayAdapter()
    runtime = _runtime(tmp_path, fake)
    request = _initial_request(combination_id="combination-1")

    with pytest.raises(HarnessError, match="receipt pending"):
        runtime.run_investigation_turn(request, scope_id="s1")
    with pytest.raises(HarnessError, match="structured contract"):
        runtime.run_investigation_turn(request, scope_id="s1")

    store = CodexSessionStore.from_root(
        tmp_path / "sessions",
        owner=CodexSessionOwner.investigation("task-1", "batch-1", "combination-1"),
        scope_id="s1",
        model="investigation-model",
    )
    recovery = store.load_recovery()
    assert recovery is not None
    assert recovery["state"] == "contract_failure"
    assert recovery["reason"] == "response_contract"
    assert recovery["action"] == "explicit_new_request_required"
    assert store.load()["status"] == "contract_failure"
    assert store.load()["turn_count"] == 1
    assert len(fake.calls) == 2

    with pytest.raises(HarnessError, match="explicit recovery"):
        runtime.run_investigation_turn(request, scope_id="s1")
    assert len(fake.calls) == 2


def test_completed_receipt_recovers_crash_before_manifest_record_without_second_model_call(
    tmp_path: Path,
) -> None:
    """receipt 在 manifest 计数前写入时，恢复补记一次且不再次调用模型。"""

    fake = _ReceiptReplayAdapter()
    runtime = _runtime(tmp_path, fake)
    request = _initial_request(combination_id="combination-1")
    with pytest.raises(HarnessError, match="process crash"):
        runtime.run_investigation_turn(request, scope_id="s1")

    result = runtime.run_investigation_turn(request, scope_id="s1")

    assert result.thread_id == "thread-replay"
    assert len(fake.calls) == 2
    assert fake.calls[1]["session_id"] == "thread-replay"
    assert fake.model_calls == 1
    store = CodexSessionStore.from_root(
        tmp_path / "sessions",
        owner=CodexSessionOwner.investigation("task-1", "batch-1", "combination-1"),
        scope_id="s1",
        model="investigation-model",
    )
    assert store.load()["turn_count"] == 1


def test_completed_receipt_recovers_crash_after_manifest_record_without_double_count(
    tmp_path: Path,
) -> None:
    """receipt 在 manifest 已计数后恢复时只重放正文，不重复增加计数。"""

    fake = _ReceiptReplayAdapter()
    runtime = _runtime(tmp_path, fake)
    request = _initial_request(combination_id="combination-1")
    with pytest.raises(HarnessError, match="process crash"):
        runtime.run_investigation_turn(request, scope_id="s1")
    store = CodexSessionStore.from_root(
        tmp_path / "sessions",
        owner=CodexSessionOwner.investigation("task-1", "batch-1", "combination-1"),
        scope_id="s1",
        model="investigation-model",
    )
    store.record_completed_turn(stage="initial_investigation", status="completed")

    runtime.run_investigation_turn(request, scope_id="s1")

    assert fake.model_calls == 1
    assert store.load()["turn_count"] == 1


def test_runtime_result_receipt_resolver_is_cross_process_idempotent(
    tmp_path: Path,
) -> None:
    """runtime 接线在 receipt 已确认后只返回 no-op，不访问 provider。"""

    owner = CodexSessionOwner.investigation("task-1", "batch-1", "combination-1")
    policy_digest = "b" * 64
    request_digest = "a" * 64
    store = CodexSessionStore.from_root(
        tmp_path / "sessions",
        owner=owner,
        scope_id="scope-1",
        model="investigation-model",
        thread_policy_digest=policy_digest,
    )
    store.create(
        "thread-runtime-resolver",
        stage="initial_investigation",
        status="created",
        thread_policy_digest=policy_digest,
    )
    store.record_completed_turn(stage="initial_investigation", status="completed")
    receipt_root = store.manifest_path.parent / "completed_turn_receipts"
    codex_sdk_adapter._write_completed_turn_receipt(
        receipt_root,
        thread_id="thread-runtime-resolver",
        turn_id="turn-runtime-resolver",
        request_digest=request_digest,
        turn_count_before=0,
        thread_policy_digest=policy_digest,
    )
    first = codex_sdk_adapter.build_codex_completed_turn_receipt_handle(
        receipt_root,
        owner=owner,
        session_store=store,
        thread_id="thread-runtime-resolver",
        turn_id="turn-runtime-resolver",
        request_digest=request_digest,
        turn_count_before=0,
        thread_policy_digest=policy_digest,
        handoff_kind="result",
    )
    first.acknowledge_after_result_persisted()

    recovered = InvestigationCodexRuntime.resolve_completed_result_receipt_handle(
        store,
        owner=owner,
        thread_id="thread-runtime-resolver",
        turn_id="turn-runtime-resolver",
        request_digest=request_digest,
        turn_count_before=0,
        thread_policy_digest=policy_digest,
    )

    assert recovered.already_acknowledged is True
    recovered.acknowledge_after_result_persisted()
    assert not (receipt_root / "turn-runtime-resolver.json").exists()


@pytest.mark.parametrize("replay_phase", ("running", "collected"))
def test_waiting_receipt_replay_rebuilds_job_state_before_new_continuation(
    tmp_path: Path,
    replay_phase: str,
) -> None:
    """waiting turn 按工具历史区分未收集作业和已完成科学结果。"""

    job_id = "job-waiting-replay"
    operation_id = "operation-waiting-replay"
    submission_key = "submission-waiting-replay"
    submit_item = {
        "type": "mcpToolCall",
        "status": "completed",
        "tool": "submit_action_job",
        "arguments": {"request": {"submission_key": submission_key}},
        "result": {
            "structuredContent": {
                "status": "success",
                "job_id": job_id,
                "operation_id": operation_id,
                "submission_key": submission_key,
                "action_id": "locate_transition_state",
                "backend_id": "gaussian",
                "job_type": "action_execution",
                "runtime": "gaussian",
                "software_id": "gaussian",
            }
        },
    }
    running_item = {
        "type": "mcpToolCall",
        "status": "completed",
        "tool": "get_execution_job",
        "arguments": {"job_id": job_id},
        "result": {
            "structuredContent": {
                "status": "success",
                "job_id": job_id,
                "job_status": "running",
                "terminal": False,
                "job": {"job_id": job_id, "status": "running"},
            }
        },
    }
    terminal_item = {
        **running_item,
        "result": {
            "structuredContent": {
                "status": "success",
                "job_id": job_id,
                "job_status": "succeeded",
                "terminal": True,
                "job": {"job_id": job_id, "status": "succeeded"},
            }
        },
    }
    collect_item = {
        "type": "mcpToolCall",
        "status": "completed",
        "tool": "collect_execution_job",
        "arguments": {"job_id": job_id},
        "result": {
            "structuredContent": {
                "status": "success",
                "ready": True,
                "job_id": job_id,
                "job_status": "succeeded",
                "action_result": {"job_id": job_id, "status": "succeeded"},
            }
        },
    }
    replay_items = (
        (submit_item, running_item)
        if replay_phase == "running"
        else (submit_item, terminal_item, collect_item)
    )

    class _WaitingReplayAdapter:
        """先制造 completed receipt，再重放旧 turn，最后模拟一个新 continuation。"""

        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def __call__(self, **kwargs):
            self.calls.append(dict(kwargs))
            request = kwargs["request"]
            thread_id = kwargs["session_id"] or "thread-waiting-replay"
            kwargs["on_thread_ready"](thread_id)
            kwargs["trace_path"].write_text("waiting-replay", encoding="utf-8")
            if len(self.calls) == 1:
                codex_sdk_adapter._write_completed_turn_receipt(
                    kwargs["completed_turn_receipt_root"],
                    thread_id=thread_id,
                    turn_id="turn-waiting-replay",
                    request_digest=kwargs["request_digest"],
                    turn_count_before=kwargs["turn_count_before"],
                    thread_policy_digest=kwargs["thread_policy_digest"],
                )
                raise HarnessError("completed waiting turn process crash", kind="transport")
            if len(self.calls) == 2:
                assert kwargs.get("replay_turn_id") == "turn-waiting-replay"
                return CodexSdkRunResult(
                    response=HarnessResponse(
                        content=_investigation_response(),
                        model=request.model,
                        metadata={"turn_id": "turn-waiting-replay"},
                    ),
                    thread_id=thread_id,
                    turn_id="turn-waiting-replay",
                    replayed=True,
                    receipt_turn_count_before=0,
                    request_digest=kwargs["request_digest"],
                    completed_tool_items=replay_items,
                )
            assert kwargs.get("replay_turn_id") == "turn-waiting-replay"
            _write_test_completed_receipt(
                kwargs,
                thread_id=thread_id,
                turn_id="turn-after-waiting-replay",
            )
            return CodexSdkRunResult(
                response=HarnessResponse(
                    content="新 continuation 仍在等待当前作业。",
                    model=request.model,
                    metadata={"turn_id": "turn-after-waiting-replay"},
                ),
                thread_id=thread_id,
                turn_id="turn-after-waiting-replay",
                request_digest=kwargs["request_digest"],
                completed_tool_items=(running_item,),
            )

    adapter = _WaitingReplayAdapter()
    runtime = _runtime(tmp_path, adapter)
    initial_request = _initial_request(combination_id="combination-1")
    with pytest.raises(HarnessError, match="completed waiting turn process crash"):
        runtime.run_investigation_turn(initial_request, scope_id="scope-1")
    store = CodexSessionStore.from_root(
        tmp_path / "sessions",
        owner=CodexSessionOwner.investigation(
            "task-1",
            "batch-1",
            "combination-1",
        ),
        scope_id="scope-1",
        model="investigation-model",
    )
    store.record_completed_turn(stage="initial_investigation", status="completed")
    waiting = InvestigationJobState(
        task_id="task-1",
        batch_id="batch-1",
        combination_id="combination-1",
        codex_thread_id="thread-waiting-replay",
        turn_id="turn-waiting-replay",
        status="waiting_investigation_job",
        identity=InvestigationJobIdentity(
            job_id=job_id,
            operation_id=operation_id,
            submission_key=submission_key,
            action_id="locate_transition_state",
            backend_id="gaussian",
            job_family="action_execution",
            job_type="action_execution",
            runtime="gaussian",
            software_id="gaussian",
        ),
        request_digest="a" * 64,
        scope_id="scope-1",
        job_family="action_execution",
        submit_tool="submit_action_job",
        status_tool="get_execution_job",
        collect_tool="collect_execution_job",
    )
    continuation = build_investigation_job_continuation_request(
        waiting.task_id,
        waiting.batch_id,
        waiting.combination_id,
        waiting.codex_thread_id,
        job_id=waiting.identity.job_id,
        operation_id=waiting.identity.operation_id,
        submission_key=waiting.identity.submission_key,
        action_id=waiting.identity.action_id,
        backend_id=waiting.identity.backend_id,
        job_family=waiting.identity.job_family,
        job_type=waiting.identity.job_type,
        runtime=waiting.identity.runtime,
        software_id=waiting.identity.software_id,
        model="investigation-model",
    )
    accepted_callbacks: list[str] = []

    replayed = runtime.run_investigation_turn(
        continuation,
        scope_id=waiting.scope_id,
        active_job=waiting,
        on_job_accepted=lambda receipt: accepted_callbacks.append(
            receipt["identity"]["job_id"]
        ),
    )

    assert accepted_callbacks == []
    receipt_path = (
        store.manifest_path.parent
        / "completed_turn_receipts"
        / "turn-waiting-replay.json"
    )
    # 回放只证明 provider turn 已完成；调用方尚未把 Result 写入 Working Memory，
    # 因此 completed receipt 必须仍然保留，不能在 runtime 返回前删除。
    assert receipt_path.exists()
    assert replayed.receipt_handle is not None
    assert replayed.receipt_handle.owner == {
        "kind": "investigation",
        "research_task_id": "task-1",
        "investigation_context_id": "batch-1",
        "hypothesis_execution_input_record_id": "combination-1",
    }

    if replay_phase == "collected":
        assert isinstance(replayed.result, InvestigationResponse)
        assert replayed.accepted_job is None
        # 模拟调用方已经把 Evidence/Result 写入 Working Memory 后的显式确认。
        replayed.receipt_handle.acknowledge_after_result_persisted()
        replayed.receipt_handle.acknowledge_after_result_persisted()
        assert not receipt_path.exists()
        assert len(adapter.calls) == 2
        return

    assert replayed.result is None
    assert replayed.accepted_job is not None
    # 模拟调用方已经保存 waiting sidecar 和 resumable waiting 事实。
    replayed.receipt_handle.acknowledge_after_waiting_persisted()
    assert not receipt_path.exists()
    assert replayed.accepted_job["identity"] == waiting.identity.as_mapping()

    continued = runtime.run_investigation_turn(
        continuation,
        scope_id=waiting.scope_id,
        active_job=waiting,
    )

    assert continued.turn_id == "turn-after-waiting-replay"
    assert continued.result is None
    assert continued.accepted_job is not None
    assert len(adapter.calls) == 3


def test_completed_turn_replay_emits_all_ordered_job_callbacks_without_new_turn(
    tmp_path: Path,
) -> None:
    """Harness 部分落盘恢复只重放 receipt history，不创建新的模型 turn。"""

    def tool_items(
        job_id: str,
        operation_id: str,
        submission_key: str,
    ) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
        submit = {
            "type": "mcpToolCall",
            "status": "completed",
            "tool": "submit_action_job",
            "arguments": {"request": {"submission_key": submission_key}},
            "result": {
                "structuredContent": {
                    "status": "success",
                    "job_id": job_id,
                    "operation_id": operation_id,
                    "submission_key": submission_key,
                    "action_id": "locate_transition_state",
                    "backend_id": "gaussian",
                    "job_type": "action_execution",
                    "runtime": "gaussian",
                    "software_id": "gaussian",
                }
            },
        }
        terminal = {
            "type": "mcpToolCall",
            "status": "completed",
            "tool": "get_execution_job",
            "arguments": {"job_id": job_id},
            "result": {
                "structuredContent": {
                    "status": "success",
                    "job_id": job_id,
                    "job_status": "succeeded",
                    "terminal": True,
                    "job": {"job_id": job_id, "status": "succeeded"},
                }
            },
        }
        collect = {
            "type": "mcpToolCall",
            "status": "completed",
            "tool": "collect_execution_job",
            "arguments": {"job_id": job_id},
            "result": {
                "structuredContent": {
                    "status": "success",
                    "ready": True,
                    "job_id": job_id,
                    "job_status": "succeeded",
                    "action_result": {"job_id": job_id, "status": "succeeded"},
                }
            },
        }
        return submit, terminal, collect

    completed_items = (
        *tool_items("job-replay-a", "operation-replay-a", "submission-replay-a"),
        *tool_items("job-replay-b", "operation-replay-b", "submission-replay-b"),
    )

    class _CompletedReplayAdapter:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, **kwargs):
            self.calls += 1
            thread_id = kwargs["session_id"] or "thread-completed-replay"
            kwargs["on_thread_ready"](thread_id)
            if self.calls == 1:
                codex_sdk_adapter._write_completed_turn_receipt(
                    kwargs["completed_turn_receipt_root"],
                    thread_id=thread_id,
                    turn_id="turn-completed-replay",
                    request_digest=kwargs["request_digest"],
                    turn_count_before=kwargs["turn_count_before"],
                    thread_policy_digest=kwargs["thread_policy_digest"],
                )
                raise HarnessError("模拟 completed turn 后进程退出", kind="transport")
            assert kwargs.get("replay_turn_id") == "turn-completed-replay"
            return CodexSdkRunResult(
                response=HarnessResponse(
                    content=json.dumps(
                        {
                            "self_judgment": "supported",
                            "narrative": "已完成两个作业的回顾性核对。",
                            "job_refs": ["job-replay-a", "job-replay-b"],
                            "operation_refs": [
                                "operation-replay-a",
                                "operation-replay-b",
                            ],
                            "artifact_refs": ["artifact-replay"],
                            "observations": ["两个作业均已完成。"],
                            "limitations": [],
                        }
                    ),
                    model=kwargs["request"].model,
                    metadata={"turn_id": "turn-completed-replay"},
                ),
                thread_id=thread_id,
                turn_id="turn-completed-replay",
                replayed=True,
                receipt_turn_count_before=0,
                request_digest=kwargs["request_digest"],
                completed_tool_items=completed_items,
            )

    adapter = _CompletedReplayAdapter()
    runtime = _runtime(tmp_path, adapter)
    request = _initial_request(combination_id="combination-completed-replay")
    with pytest.raises(HarnessError, match="completed turn"):
        runtime.run_investigation_turn(request, scope_id="scope-1")
    store = CodexSessionStore.from_root(
        tmp_path / "sessions",
        owner=CodexSessionOwner.investigation(
            "task-1",
            "batch-1",
            "combination-completed-replay",
        ),
        scope_id="scope-1",
        model="investigation-model",
    )
    store.record_completed_turn(stage="initial_investigation", status="completed")
    events: list[tuple[str, str]] = []
    replay_request = build_initial_investigation_request(
        "task-1",
        "batch-1",
        "combination-completed-replay",
        "恢复时重新构造但不属于原 turn 身份的科学问题",
        {"method": "replay", "charge": 1},
        codex_thread_id="thread-completed-replay",
        model="investigation-model",
    )

    replayed = runtime.replay_completed_turn(
        replay_request,
        scope_id="scope-1",
        replay_turn_id="turn-completed-replay",
        on_job_accepted=lambda receipt: events.append(
            ("accepted", receipt["identity"]["job_id"])
        ),
        on_job_collected=lambda receipt: events.append(
            ("collected", receipt["identity"]["job_id"])
        ),
    )

    assert replayed.result is not None
    assert events == [
        ("accepted", "job-replay-a"),
        ("collected", "job-replay-a"),
        ("accepted", "job-replay-b"),
        ("collected", "job-replay-b"),
    ]
    assert adapter.calls == 2


def test_normal_completed_turn_does_not_override_backend_attempt_digest(
    tmp_path: Path,
) -> None:
    """普通 turn 的 Codex receipt 摘要不能冒充 backend attempt 摘要。"""

    class _NormalCompletedAdapter:
        def __call__(self, **kwargs):
            thread_id = kwargs["session_id"] or "thread-normal-completed"
            kwargs["on_thread_ready"](thread_id)
            _write_test_completed_receipt(
                kwargs,
                thread_id=thread_id,
                turn_id="turn-normal-completed",
            )
            return CodexSdkRunResult(
                response=HarnessResponse(
                    content=_investigation_response(),
                    model=kwargs["request"].model,
                    metadata={"turn_id": "turn-normal-completed"},
                ),
                thread_id=thread_id,
                turn_id="turn-normal-completed",
                request_digest=kwargs["request_digest"],
            )

    runtime = _runtime(tmp_path, _NormalCompletedAdapter())

    completed = runtime.run_investigation_turn(
        _initial_request(combination_id="combination-normal-completed"),
        scope_id="scope-1",
    )

    assert completed.result is not None
    assert completed.request_digest is None
    assert completed.receipt_handle is not None
    assert completed.receipt_handle.request_digest


def test_completed_result_repair_replay_uses_original_receipt_request_digest(
    tmp_path: Path,
) -> None:
    """无新作业的 Result 返修重放应绑定原 turn 摘要，而非重建请求摘要。"""

    class _ResultRepairReplayAdapter:
        def __init__(self) -> None:
            self.calls = 0
            self.original_request_digest: str | None = None

        def __call__(self, **kwargs):
            self.calls += 1
            thread_id = kwargs["session_id"] or "thread-result-repair-replay"
            kwargs["on_thread_ready"](thread_id)
            if self.calls == 1:
                self.original_request_digest = kwargs["request_digest"]
                codex_sdk_adapter._write_completed_turn_receipt(
                    kwargs["completed_turn_receipt_root"],
                    thread_id=thread_id,
                    turn_id="turn-result-repair-replay",
                    request_digest=kwargs["request_digest"],
                    turn_count_before=kwargs["turn_count_before"],
                    thread_policy_digest=kwargs["thread_policy_digest"],
                )
                raise HarnessError("模拟 Result 返修完成后进程退出", kind="transport")
            assert kwargs.get("replay_turn_id") == "turn-result-repair-replay"
            return CodexSdkRunResult(
                response=HarnessResponse(
                    content=_investigation_response(),
                    model=kwargs["request"].model,
                    metadata={"turn_id": "turn-result-repair-replay"},
                ),
                thread_id=thread_id,
                turn_id="turn-result-repair-replay",
                replayed=True,
                receipt_turn_count_before=0,
                request_digest=kwargs["request_digest"],
            )

    adapter = _ResultRepairReplayAdapter()
    runtime = _runtime(tmp_path, adapter)
    original_request = _initial_request(
        combination_id="combination-result-repair-replay"
    )
    with pytest.raises(HarnessError, match="Result 返修"):
        runtime.run_investigation_turn(original_request, scope_id="scope-1")
    store = CodexSessionStore.from_root(
        tmp_path / "sessions",
        owner=CodexSessionOwner.investigation(
            "task-1",
            "batch-1",
            "combination-result-repair-replay",
        ),
        scope_id="scope-1",
        model="investigation-model",
    )
    store.record_completed_turn(stage="deepseek_repair", status="completed")
    replay_request = build_initial_investigation_request(
        "task-1",
        "batch-1",
        "combination-result-repair-replay",
        "恢复时重建的初始问题",
        {"method": "replay", "charge": 1},
        codex_thread_id="thread-result-repair-replay",
        model="investigation-model",
    )

    replayed = runtime.replay_completed_turn(
        replay_request,
        scope_id="scope-1",
        replay_turn_id="turn-result-repair-replay",
    )

    assert replayed.result is not None
    assert replayed.turn_id == "turn-result-repair-replay"
    assert replayed.request_digest == adapter.original_request_digest
    assert replayed.receipt_handle is not None
    assert replayed.request_digest == replayed.receipt_handle.request_digest
    assert adapter.calls == 2


def test_callback_failure_does_not_start_turn(tmp_path: Path) -> None:
    """thread 登记失败时 adapter 必须在真实 turn 之前停止。"""

    calls: list[str] = []

    def failing_adapter(**kwargs):
        kwargs["on_thread_ready"]("")
        calls.append("turn")
        raise AssertionError("不应启动 turn")

    runtime = _runtime(tmp_path, failing_adapter)
    with pytest.raises(HarnessError, match="thread_id"):
        runtime.run_investigation_turn(_initial_request(combination_id="combination-1"), scope_id="s1")
    assert calls == []


def test_investigation_has_no_total_timeout_but_review_is_finite(tmp_path: Path) -> None:
    """调查 turn 使用 None，总体有限 timeout 只属于 Review。"""

    fake = _FakeAdapter()
    investigation = _runtime(tmp_path / "investigation", fake)
    review = _runtime(tmp_path / "review", fake, evidence=True)
    investigation.run_investigation_turn(
        _initial_request(combination_id="combination-1"),
        scope_id="s1",
    )
    review.run_task_review_turn(_review_request(), scope_id="s2")

    assert fake.calls[0]["request"].timeout_seconds is None
    assert fake.calls[1]["request"].timeout_seconds == 900.0
    assert fake.calls[0]["idle_timeout_seconds"] == 1800.0
    assert fake.calls[1]["idle_timeout_seconds"] == 300.0
    assert fake.calls[0]["tool_idle_timeout_seconds"] == 21600.0
    assert fake.calls[1]["tool_idle_timeout_seconds"] == 21600.0
    assert (
        fake.calls[0]["thread_config"]["mcp_servers"]["minichem_toolbox"]["tool_timeout_sec"]
        == 21600
    )

    explicit = _runtime(
        tmp_path / "explicit",
        fake,
        sdk_idle_timeout_seconds=37.0,
    )
    explicit.run_investigation_turn(
        _initial_request(combination_id="combination-2"),
        scope_id="s2",
    )
    assert fake.calls[2]["idle_timeout_seconds"] == 37.0


def test_resume_thread_drift_is_rejected_before_turn(tmp_path: Path) -> None:
    """恢复返回漂移 thread 时 fail closed，且不会产生第二个完成计数。"""

    fake = _FakeAdapter()
    runtime = _runtime(tmp_path, fake)
    first = runtime.run_investigation_turn(_initial_request(combination_id="combination-1"), scope_id="s1")
    fake.return_drift = True
    repair = build_repair_investigation_request(
        "task-1",
        "batch-1",
        "combination-1",
        first.thread_id,
        "2",
        "补足证据",
        model="investigation-model",
    )
    with pytest.raises(HarnessError, match="different thread_id"):
        runtime.run_investigation_turn(repair, scope_id="s1")

    store = CodexSessionStore.from_root(
        tmp_path / "sessions",
        owner=CodexSessionOwner.investigation("task-1", "batch-1", "combination-1"),
        scope_id="s1",
        model="investigation-model",
    )
    assert store.load()["thread_id"] == first.thread_id
    assert store.load()["turn_count"] == 1


def test_custom_provider_settings_are_forwarded_without_key_values(tmp_path: Path) -> None:
    """调用方可显式替换 provider 身份，但 API key 值不进入 adapter 入参。"""

    fake = _FakeAdapter()
    runtime = InvestigationCodexRuntime(
        session_root=tmp_path / "sessions",
        trace_root=tmp_path / "traces",
        minichem=InvestigationMcpSettings("http://127.0.0.1:9011/mcp", "token"),
        investigation_provider=CodexProviderSettings(
            "custom-investigation",
            "CUSTOM_INVESTIGATION_KEY",
            "CUSTOM_INVESTIGATION_URL",
            "CUSTOM_INVESTIGATION_MODEL",
        ),
        environment={
            "CUSTOM_INVESTIGATION_KEY": "secret-value",
            "CUSTOM_INVESTIGATION_URL": "https://custom.example/v1",
            "CUSTOM_INVESTIGATION_MODEL": "custom-model",
        },
        adapter=fake,
        allow_test_doubles=True,
    )
    result = runtime.run_investigation_turn(
        _initial_request(combination_id="combination-1", model="custom-model"),
        scope_id="s1",
    )

    assert result.thread_id == "thread-1"
    assert fake.calls[0]["provider_id"] == "custom-investigation"
    assert fake.calls[0]["env_key"] == "CUSTOM_INVESTIGATION_KEY"
    assert "secret-value" not in json.dumps(fake.calls[0]["thread_config"])


def test_custom_provider_key_is_excluded_from_codex_shell_policy(tmp_path: Path) -> None:
    """自定义 provider 的 env_key 也必须被 Codex shell 动态排除。"""

    config = codex_sdk_adapter.build_codex_capability_config(
        thread_config={},
        workspace=tmp_path / "workspace",
        profile="investigation",
        env_key="CUSTOM_PROVIDER_KEY",
    )

    assert "CUSTOM_PROVIDER_KEY" in config["shell_environment_policy"]["exclude"]


def test_sdk_trace_redacts_static_http_header_values() -> None:
    """SDK notification 即使回显 header，也不能把 token 写入 trace。"""

    trace = StringIO()
    event = SimpleNamespace(
        method="mcpServer/startupStatus/updated",
        payload={"message": "header=capability-secret", "nested": ["capability-secret"]},
    )
    codex_sdk_adapter._write_notification(
        trace,
        event,
        sensitive_values=("capability-secret",),
    )

    assert "capability-secret" not in trace.getvalue()
    assert "[REDACTED]" in trace.getvalue()


def test_runtime_role_composition_and_cross_role_calls_fail_closed(tmp_path: Path) -> None:
    """两个角色的 MCP 组成和公开调用面必须严格互斥。"""

    with pytest.raises(HarnessError, match="forbids evidence"):
        InvestigationCodexRuntime(
            session_root=tmp_path / "bad-investigation",
            trace_root=tmp_path / "bad-investigation-traces",
            role="investigation",
            minichem=InvestigationMcpSettings("http://127.0.0.1:9011/mcp", "MINICHEM_TOKEN"),
            evidence_mcp=EvidenceMcpSettings(
                "http://127.0.0.1:9020/mcp",
                {"Authorization": "EVIDENCE_TOKEN"},
            ),
            environment={"MINICHEM_TOKEN": "mini", "EVIDENCE_TOKEN": "evidence"},
            adapter=_FakeAdapter(),
        )

    investigation = _runtime(tmp_path / "investigation", _FakeAdapter())
    review = _runtime(tmp_path / "review", _FakeAdapter(), evidence=True)
    with pytest.raises(HarnessError, match="cannot call task_review"):
        investigation.run_task_review_turn(_review_request(), scope_id="review")
    with pytest.raises(HarnessError, match="cannot call investigation"):
        review.run_investigation_turn(
            _initial_request(combination_id="combination-1"),
            scope_id="investigation",
        )


def test_review_runtime_has_no_minichem_configuration_or_secret(tmp_path: Path) -> None:
    """Review runtime 和 thread config 不得出现 MiniChem 或其 capability。"""

    fake = _FakeAdapter()
    review = _runtime(tmp_path, fake, evidence=True)
    review.run_task_review_turn(_review_request(), scope_id="review")

    assert review.minichem is None
    encoded_runtime = repr(review.__dict__)
    assert "MINICHEM_CAPABILITY_TOKEN" not in encoded_runtime
    assert "capability-secret" not in encoded_runtime
    thread_config = fake.calls[0]["thread_config"]
    assert "minichem_toolbox" not in thread_config.get("mcp_servers", {})
    assert "MINICHEM_CAPABILITY_TOKEN" not in json.dumps(thread_config)
    assert "capability-secret" not in json.dumps(thread_config)


def test_env_http_headers_are_names_and_shell_excludes_all_private_names(tmp_path: Path) -> None:
    """MCP header 只引用环境变量名，provider/MCP 值均不进 thread config。"""

    fake = _FakeAdapter()
    runtime = _runtime(tmp_path, fake, evidence=True)
    runtime.run_task_review_turn(_review_request(), scope_id="review")

    call = fake.calls[0]
    config = call["thread_config"]
    evidence = config["mcp_servers"]["evidence_reader"]
    assert evidence["env_http_headers"] == {
        "Authorization": "EVIDENCE_AUTH_TOKEN",
        "X-Evidence-Scope": "EVIDENCE_SCOPE_TOKEN",
    }
    assert "http_headers" not in evidence
    excluded = config["shell_environment_policy"]["exclude"]
    assert call["env_key"] in excluded
    assert "EVIDENCE_AUTH_TOKEN" in excluded
    assert "EVIDENCE_SCOPE_TOKEN" in excluded
    assert "Bearer evidence-token" not in json.dumps(config)
    assert "evidence-scope-secret" not in json.dumps(config)


def test_runtime_sanitizes_secret_in_exception_chain(tmp_path: Path) -> None:
    """adapter 异常及其 cause/context 含 token 时，向上抛出的链也不能泄漏。"""

    class SecretFailingAdapter:
        def __call__(self, **kwargs):
            kwargs["on_thread_ready"]("thread-secret-failure")
            try:
                raise ValueError("MCP token=capability-secret")
            except ValueError as nested:
                raise RuntimeError("outer adapter failure") from nested

    runtime = _runtime(tmp_path, SecretFailingAdapter())
    with pytest.raises(HarnessError) as failure:
        runtime.run_investigation_turn(
            _initial_request(combination_id="combination-1"),
            scope_id="scope-1",
        )
    chain: list[BaseException] = []
    current: BaseException | None = failure.value
    while current is not None:
        chain.append(current)
        current = current.__cause__ or current.__context__
    assert all("capability-secret" not in str(item) for item in chain)


def test_turn_started_hook_receives_accepted_sdk_turn_id_before_completion(
    tmp_path: Path,
) -> None:
    """SDK accepted hook 必须在完成结果返回前收到真实 turn id。"""

    fake = _FakeAdapter()
    fake.emit_turn_started = True
    events: list[str] = []
    runtime = _runtime(tmp_path, fake, on_turn_started=events.append)
    result = runtime.run_investigation_turn(
        _initial_request(combination_id="combination-1"),
        scope_id="scope-1",
    )

    assert events == ["turn-1"]
    assert result.turn_id == "turn-1"
    assert fake.started_turn_ids == events


def test_dynamic_turn_started_hook_receives_confirmed_thread_and_turn_identity(
    tmp_path: Path,
) -> None:
    """每次调用注入的 hook 必须收到同一已确认的 thread/turn 二元身份。"""

    fake = _FakeAdapter()
    fake.emit_turn_started = True
    events: list[tuple[str, str]] = []
    runtime = _runtime(tmp_path, fake)
    result = runtime.run_investigation_turn(
        _initial_request(combination_id="combination-1"),
        scope_id="scope-1",
        on_turn_started=lambda thread_id, turn_id: events.append((thread_id, turn_id)),
    )

    assert events == [(result.thread_id, result.turn_id)]


def test_dynamic_turn_started_hook_is_call_local_and_not_reused_by_next_turn(
    tmp_path: Path,
) -> None:
    """连续 turn 使用各自 callback；前一 turn 的 callback 不会收到后一事件。"""

    fake = _FakeAdapter()
    fake.emit_turn_started = True
    first_events: list[tuple[str, str]] = []
    second_events: list[tuple[str, str]] = []
    runtime = _runtime(tmp_path, fake)
    first = runtime.run_investigation_turn(
        _initial_request(combination_id="combination-1"),
        scope_id="scope-1",
        on_turn_started=lambda thread_id, turn_id: first_events.append((thread_id, turn_id)),
    )
    second = runtime.run_investigation_turn(
        _initial_request(combination_id="combination-2"),
        scope_id="scope-2",
        on_turn_started=lambda thread_id, turn_id: second_events.append((thread_id, turn_id)),
    )

    assert first_events == [(first.thread_id, first.turn_id)]
    assert second_events == [(second.thread_id, second.turn_id)]


def test_dynamic_turn_started_hook_is_not_called_for_completed_receipt_replay(
    tmp_path: Path,
) -> None:
    """completed receipt 重放不重复通知 Working Memory 的 turn-start hook。"""

    fake = _ReceiptReplayAdapter()
    runtime = _runtime(tmp_path, fake)
    request = _initial_request(combination_id="combination-1")
    events: list[tuple[str, str]] = []
    with pytest.raises(HarnessError, match="process crash"):
        runtime.run_investigation_turn(
            request,
            scope_id="s1",
            on_turn_started=lambda thread_id, turn_id: events.append((thread_id, turn_id)),
        )
    result = runtime.run_investigation_turn(
        request,
        scope_id="s1",
        on_turn_started=lambda thread_id, turn_id: events.append((thread_id, turn_id)),
    )

    assert events == []
    assert result.turn_id == "turn-replay"


def test_runtime_scope_fence_is_skipped_when_completed_receipt_replays(
    tmp_path: Path,
) -> None:
    """输入带 fence 的 runtime 和一次崩溃/恢复序列，断言 receipt replay 跳过 fence。

    测试动作让 fake adapter 模拟首轮 receipt 生成和第二轮 replay；不调用真实模型，
    只断言 runtime 到 adapter 的 callback 边界及一次性 fence 调用次数。
    """

    fake = _ReceiptReplayAdapter()
    fence_calls: list[str] = []
    runtime = _runtime(
        tmp_path,
        fake,
        before_new_investigation_turn=lambda _envelope: fence_calls.append("fence"),
    )
    request = _initial_request(combination_id="combination-1")
    with pytest.raises(HarnessError, match="process crash"):
        runtime.run_investigation_turn(request, scope_id="s1")
    runtime.run_investigation_turn(request, scope_id="s1")

    # 该 fake adapter 模拟 adapter 的 receipt-aware 顺序：首轮执行 hook，
    # 第二轮已经有 receipt 时直接返回 replay，不执行 hook。
    assert callable(fake.calls[0]["before_new_turn"])
    assert fence_calls == ["fence"]


def test_dynamic_turn_started_hook_failure_fails_closed_before_completion(
    tmp_path: Path,
) -> None:
    """WM start 事实写入失败时不得返回模型结果或增加完成计数。"""

    fake = _FakeAdapter()
    fake.emit_turn_started = True
    runtime = _runtime(tmp_path, fake)

    def reject_lifecycle(_thread_id: str, _turn_id: str) -> None:
        raise RuntimeError("working memory unavailable")

    with pytest.raises(RuntimeError, match="working memory unavailable"):
        runtime.run_investigation_turn(
            _initial_request(combination_id="combination-1"),
            scope_id="scope-1",
            on_turn_started=reject_lifecycle,
        )
    store = CodexSessionStore.from_root(
        tmp_path / "sessions",
        owner=CodexSessionOwner.investigation("task-1", "batch-1", "combination-1"),
        scope_id="scope-1",
        model="investigation-model",
    )
    assert store.load()["turn_count"] == 0




def test_turn_started_hook_rejects_adapter_without_explicit_protocol(
    tmp_path: Path,
) -> None:
    """未实现 accepted hook 的 adapter 不能被误标成已启动。"""

    calls: list[str] = []

    def incompatible_adapter() -> None:
        calls.append("called")

    runtime = _runtime(tmp_path, incompatible_adapter, on_turn_started=lambda _turn_id: None)
    with pytest.raises(HarnessError, match="accepted-turn hook"):
        runtime.run_investigation_turn(
            _initial_request(combination_id="combination-1"),
            scope_id="scope-1",
        )
    assert calls == []



def test_runtime_session_turn_fence_covers_manifest_and_recovery_protocol() -> None:
    """会话 turn fence 覆盖新会话、合同失败、worker 清理和 receipt 重放分支。"""

    runtime = object.__new__(InvestigationCodexRuntime)
    check = runtime._assert_session_turn_available
    request_digest = "request-digest"
    policy_digest = "policy-digest"

    check(None, None, request_digest, policy_digest)
    with pytest.raises(HarnessError, match="without a session manifest") as missing_manifest:
        check(None, {"state": "recovery_pending"}, request_digest, policy_digest)
    assert missing_manifest.value.kind == "session_state"

    with pytest.raises(HarnessError, match="contract failure"):
        check({"status": "contract_failure"}, None, request_digest, policy_digest)
    with pytest.raises(HarnessError, match="recovery_pending record is missing"):
        check({"status": "recovery_pending"}, None, request_digest, policy_digest)
    check({"status": "completed"}, None, request_digest, policy_digest)

    base = {
        "thread_policy_digest": policy_digest,
        "state": "recovery_pending",
        "action": "worker_stop_confirmation_required",
        "reason": "worker_cleanup",
        "request_digest": request_digest,
    }
    with pytest.raises(HarnessError, match="thread policy"):
        check({"status": "recovery_pending"}, {**base, "thread_policy_digest": "old"}, request_digest, policy_digest)
    with pytest.raises(HarnessError, match="stale or incomplete"):
        check({"status": "completed"}, base, request_digest, policy_digest)
    with pytest.raises(HarnessError, match="state does not match"):
        check({"status": "recovery_pending"}, {**base, "state": "other"}, request_digest, policy_digest)

    check(
        {"status": "recovery_pending"},
        {
            **base,
            "action": "receipt_replay_required",
            "reason": "completion_receipt",
        },
        request_digest,
        policy_digest,
    )
    with pytest.raises(HarnessError, match="no-turn confirmation") as intent_error:
        check(
            {"status": "recovery_pending"},
            {
                **base,
                "action": "request_intent_confirmation_required",
                "reason": "request_intent",
            },
            request_digest,
            policy_digest,
        )
    assert intent_error.value.kind == "recovery_pending"
    with pytest.raises(HarnessError, match="stop confirmation"):
        check({"status": "recovery_pending"}, base, request_digest, policy_digest)

    goal_recovery = {
        **base,
        "reason": "goal",
        "action": "goal_recovery_required",
    }
    check(
        {"status": "recovery_pending"},
        goal_recovery,
        request_digest,
        policy_digest,
        allow_goal_recovery=True,
    )
    with pytest.raises(HarnessError, match="original Investigation request"):
        check(
            {"status": "recovery_pending"},
            goal_recovery,
            request_digest,
            policy_digest,
        )
    with pytest.raises(HarnessError, match="original Investigation request"):
        check(
            {"status": "recovery_pending"},
            goal_recovery,
            "different-request-digest",
            policy_digest,
            allow_goal_recovery=True,
        )
    with pytest.raises(HarnessError, match="original Investigation request"):
        check(
            {"status": "recovery_pending"},
            {**goal_recovery, "action": "worker_stop_confirmation_required"},
            request_digest,
            policy_digest,
            allow_goal_recovery=True,
        )

    explicit_completion = {
        **base,
        "action": "explicit_new_request_required",
        "reason": "completion_receipt",
    }
    check({"status": "recovery_pending"}, explicit_completion, request_digest, policy_digest)
    with pytest.raises(HarnessError, match="cannot replay"):
        check(
            {"status": "recovery_pending"},
            {**explicit_completion, "reason": "response_contract"},
            request_digest,
            policy_digest,
        )
    with pytest.raises(HarnessError, match="original request"):
        check(
            {"status": "recovery_pending"},
            {**explicit_completion, "request_digest": "different"},
            request_digest,
            policy_digest,
        )
    check(
        {"status": "recovery_pending"},
        {**explicit_completion, "request_digest": "different", "reason": "worker_cleanup"},
        request_digest,
        policy_digest,
    )



def test_runtime_mcp_composition_accepts_only_one_role_and_its_tools() -> None:
    """runtime 角色推导和 MCP 组合拒绝类型错误、歧义及跨角色工具。"""

    mini = InvestigationMcpSettings("http://127.0.0.1:9011/mcp", "MINICHEM_TOKEN")
    evidence = EvidenceMcpSettings("http://127.0.0.1:9020/mcp", {"Authorization": "TOKEN"})
    validate = codex_runtime_module._validate_runtime_mcp_composition

    with pytest.raises(TypeError, match="minichem"):
        validate(None, object(), None)
    with pytest.raises(TypeError, match="evidence_mcp"):
        validate(None, None, object())
    assert validate(None, mini, None) == "investigation"
    assert validate(None, None, evidence) == "task_review"
    with pytest.raises(HarnessError, match="ambiguous"):
        validate(None, None, None)
    with pytest.raises(HarnessError, match="requires MiniChem"):
        validate("investigation", None, None)
    with pytest.raises(HarnessError, match="requires evidence"):
        validate("task_review", mini, evidence)


def test_runtime_constructor_rejects_non_mapping_environment_and_callbacks(
    tmp_path: Path,
) -> None:
    """runtime 构造器拒绝非 mapping 环境、不可调用 adapter 和生命周期 hook。"""

    mini = InvestigationMcpSettings("http://127.0.0.1:9011/mcp", "MINICHEM_CAPABILITY_TOKEN")
    environment = {
        "ARCHE_CODEX_API_KEY": "investigation-secret",
        "ARCHE_CODEX_BASE_URL": "https://investigation.example/v1",
        "ARCHE_CODEX_MODEL_NAME": "investigation-model",
        "MINICHEM_CAPABILITY_TOKEN": "capability-secret",
    }
    counter = 0

    def construct(**overrides):
        """使用独立路径构造一个待校验 runtime。"""

        nonlocal counter
        counter += 1
        values = {
            "session_root": tmp_path / f"sessions-{counter}",
            "trace_root": tmp_path / f"traces-{counter}",
            "minichem": mini,
            "environment": environment,
            "adapter": _FakeAdapter(),
        }
        values.update(overrides)
        return InvestigationCodexRuntime(**values)

    with pytest.raises(TypeError, match="environment"):
        construct(environment=[])
    with pytest.raises(TypeError, match="adapter"):
        construct(adapter=object())
    with pytest.raises(TypeError, match="before_new_investigation_turn"):
        construct(before_new_investigation_turn=object())
    with pytest.raises(TypeError, match="on_turn_started"):
        construct(on_turn_started=object())


def test_runtime_adapter_result_validation_rejects_thread_turn_and_digest_drift() -> None:
    """adapter 结果必须同时匹配已登记 thread、请求摘要和 thread policy 摘要。"""

    from types import SimpleNamespace

    runtime = object.__new__(InvestigationCodexRuntime)
    runtime._sensitive_values = ()
    prepared = SimpleNamespace(
        session_id=None,
        request_digest="request-digest",
        thread_policy_digest="policy-digest",
    )
    callbacks = SimpleNamespace(registered_thread_id="thread-1")
    response = HarnessResponse(metadata={"turn_id": "turn-1"})
    make_result = lambda **updates: CodexSdkRunResult(
        response=response,
        thread_id=updates.get("thread_id", "thread-1"),
        turn_id=updates.get("turn_id", "turn-1"),
        request_digest=updates.get("request_digest", "request-digest"),
        thread_policy_digest=updates.get("thread_policy_digest", "policy-digest"),
    )

    with pytest.raises(HarnessError, match="invalid result"):
        runtime._validate_adapter_result(prepared, callbacks, object())
    with pytest.raises(HarnessError, match="did not register"):
        runtime._validate_adapter_result(
            prepared,
            SimpleNamespace(registered_thread_id=None),
            make_result(),
        )
    with pytest.raises(HarnessError, match="different from the registered"):
        runtime._validate_adapter_result(prepared, callbacks, make_result(thread_id="thread-2"))
    with pytest.raises(HarnessError, match="resume returned"):
        runtime._validate_adapter_result(
            SimpleNamespace(
                session_id="thread-expected",
                request_digest="request-digest",
                thread_policy_digest="policy-digest",
            ),
            callbacks,
            make_result(),
        )
    with pytest.raises(HarnessError, match="request digest"):
        runtime._validate_adapter_result(
            prepared,
            callbacks,
            make_result(request_digest="different"),
        )
    with pytest.raises(HarnessError, match="thread policy digest"):
        runtime._validate_adapter_result(
            prepared,
            callbacks,
            make_result(thread_policy_digest="different"),
        )
    assert runtime._validate_adapter_result(prepared, callbacks, make_result()) == (
        "thread-1",
        "turn-1",
    )


def test_runtime_request_sequence_helpers_and_provider_resolution_reject_invalid_values(
    tmp_path: Path,
) -> None:
    """请求工具数组和 provider URL/key/model 三元组使用严格 fail-closed 校验。"""

    with pytest.raises(HarnessError, match="tools must be an array"):
        codex_runtime_module._copy_tool_sequence({})
    with pytest.raises(HarnessError, match=r"tools\[0\]"):
        codex_runtime_module._copy_tool_sequence([object()])
    assert codex_runtime_module._copy_tool_sequence(None) == ()
    assert codex_runtime_module._copy_tool_sequence([{"name": "read"}]) == ({"name": "read"},)

    runtime = _runtime(tmp_path, _FakeAdapter())
    settings = runtime.investigation_provider
    runtime._environment = {
        **runtime._environment,
        settings.base_url_env: "https://provider.example/v1",
        settings.env_key: "provider-secret",
        settings.model_env: "investigation-model",
    }
    resolved = runtime._resolve_provider(settings, "investigation-model")
    assert resolved.base_url == "https://provider.example/v1"
    runtime._environment[settings.base_url_env] = "ftp://provider.example/v1"
    with pytest.raises(HarnessError, match="base URL"):
        runtime._resolve_provider(settings, "investigation-model")
    runtime._environment[settings.base_url_env] = "https://provider.example/v1"
    runtime._environment[settings.env_key] = ""
    with pytest.raises(HarnessError, match="missing"):
        runtime._resolve_provider(settings, "investigation-model")
    runtime._environment[settings.env_key] = "provider-secret"
    with pytest.raises(HarnessError, match="does not match"):
        runtime._resolve_provider(settings, "other-model")
