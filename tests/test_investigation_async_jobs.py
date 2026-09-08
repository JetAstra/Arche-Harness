"""Investigation 真实 submit 回执、Controller 轮询和同 thread 唤醒合同测试。"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from chemistry_multiagent.harness.base import HarnessRequest
from chemistry_multiagent.memory import ContractValidationError
from chemistry_multiagent.investigation.async_jobs import (
    InvestigationJobController,
    InvestigationJobIdentity,
    InvestigationJobReceipt,
    InvestigationJobState,
    InvestigationJobSubmissionError,
    InvestigationJobStatusError,
    InvestigationJobTurnClosedError,
    InvestigationWaitingJobStore,
    build_investigation_job_receipt,
)
from chemistry_multiagent.investigation.contracts import InvestigationResponse
from chemistry_multiagent.investigation.codex_runtime import InvestigationTurnResult
from chemistry_multiagent.investigation.domain import (
    InvestigationResult,
    InvestigationSession,
    InvestigationState,
)
from chemistry_multiagent.investigation.prompts import (
    build_initial_investigation_request,
    build_investigation_job_wake_request,
)
from chemistry_multiagent.investigation.resume_contracts import (
    compute_resume_record_auth_digest,
    compute_resume_record_digest,
)
from minichem_toolbox.scope import ScopeContext, activate_scope
from minichem_mcp_tools import transition_state_jobs as transition_state_jobs


class _Gateway:
    """只提供 Controller status/arm，模拟 queued 到 succeeded。"""

    def __init__(self) -> None:
        self.get_calls: list[str] = []
        self.arm_calls: list[str] = []
        self.status = "queued"

    @staticmethod
    def _response(
        waiting: InvestigationJobState,
        status: str,
        *,
        armed: bool = False,
        phase: str | None = None,
    ) -> dict[str, object]:
        response: dict[str, object] = {
            "status": "success",
            "job_id": waiting.identity.job_id,
            "operation_id": waiting.identity.operation_id,
            "submission_key": waiting.identity.submission_key,
            "action_id": waiting.identity.action_id,
            "backend_id": waiting.identity.backend_id,
            "job": {"job_id": waiting.identity.job_id, "status": status},
        }
        if armed:
            response["wake_armed"] = True
            response["phase"] = phase or "armed"
        return response

    def get_investigation_job_status(self, waiting: InvestigationJobState):
        self.get_calls.append(waiting.identity.job_id)
        return self._response(waiting, self.status)

    def arm_investigation_job(self, waiting: InvestigationJobState, terminal_status: str):
        self.arm_calls.append(waiting.identity.job_id)
        return self._response(waiting, terminal_status, armed=True)


class _TransientStatusGateway(_Gateway):
    """第一次 status 暂时失败，随后返回真实作业身份。"""

    def __init__(self) -> None:
        super().__init__()
        self.fail_next_status = True

    def get_investigation_job_status(self, waiting: InvestigationJobState):
        """模拟 Controller status 通道短暂不可用。"""

        if self.fail_next_status:
            self.fail_next_status = False
            raise InvestigationJobStatusError("temporary status outage")
        return super().get_investigation_job_status(waiting)


def _receipt_for(job_id: str) -> InvestigationJobReceipt:
    """构造一个来自真实 submit 工具响应的指定作业回执。"""

    identity = InvestigationJobIdentity(
        job_id=job_id,
        operation_id=f"operation-{job_id.removeprefix('job-')}",
        submission_key=f"submission-{job_id.removeprefix('job-')}",
        action_id="action-1",
        backend_id="backend-1",
    )
    return InvestigationJobReceipt(
        tool_name="submit_transition_state_path_job",
        identity=identity,
        arguments={"request": f"submit-{job_id}"},
        result=identity.as_mapping(),
    )


def _receipt() -> InvestigationJobReceipt:
    """构造默认的首个真实 submit 回执。"""

    return _receipt_for("job-1")


def _native_receipt() -> InvestigationJobReceipt:
    """构造原生 Gaussian 提交回执，供统一状态接口合同测试使用。"""

    identity = InvestigationJobIdentity(
        job_id="job-native",
        operation_id="operation-native",
        submission_key="submission-native",
        job_family="native_execution",
        job_type="native_software",
        runtime="gaussian",
        operation_fingerprint="a" * 64,
    )
    return InvestigationJobReceipt(
        tool_name="submit_native_job",
        identity=identity,
        arguments={"request": "submit-native"},
        result=identity.as_mapping(),
    )


class _MinimalNativeStatusGateway:
    """模拟只返回 job_id 和状态的 scope 绑定统一查询接口。"""

    def __init__(self, *, operation_id: str | None = None) -> None:
        self.operation_id = operation_id

    def get_investigation_job_status(
        self,
        waiting: InvestigationJobState,
    ) -> Mapping[str, Any]:
        """返回最小状态；可选注入冲突 operation_id 验证保护仍生效。"""

        result: dict[str, Any] = {
            "status": "success",
            "job_id": waiting.identity.job_id,
            "job": {"job_id": waiting.identity.job_id, "status": "queued"},
        }
        if self.operation_id is not None:
            result["operation_id"] = self.operation_id
        return result


def _waiting(controller: InvestigationJobController) -> InvestigationJobState:
    """用真实 submit 回执建立可恢复等待状态。"""

    return controller.capture_submit_receipt(
        _receipt(),
        task_id="task-1",
        batch_id="batch-1",
        combination_id="combination-1",
        codex_thread_id="thread-1",
        turn_id="turn-1",
        scope_id="scope-1",
    )




















def _coordinator_state(
    waiting: InvestigationJobState,
    *,
    result_version: str | None = None,
) -> InvestigationState:
    """构造单组合 waiting 路由状态，支持初始和已有 v1 的返修恢复。"""

    session = InvestigationSession(
        task_id=waiting.task_id,
        batch_id=waiting.batch_id,
        combination_id=waiting.combination_id,
        codex_thread_id=waiting.codex_thread_id,
        workspace_ref="workspace-1",
        scope_id=waiting.scope_id,
        result_version=result_version,
        repair_count=1 if result_version is not None else 0,
        phase="repair_pending" if result_version is not None else "created",
    )
    results: tuple[InvestigationResult, ...] = ()
    if result_version is not None:
        results = (
            InvestigationResult(
                task_id=waiting.task_id,
                batch_id=waiting.batch_id,
                combination_id=waiting.combination_id,
                result_version=result_version,
                codex_thread_id=waiting.codex_thread_id,
                scope_id=waiting.scope_id,
                turn_id="turn-v1",
                self_judgment="supported",
                narrative="已有 v1 科学结果。",
                job_refs=("job-1",),
                operation_refs=("operation-1",),
                artifact_refs=("artifact-1",),
                observations=("已有观测。",),
                limitations=(),
            ),
        )
    waiting_mapping = waiting.as_mapping()
    return InvestigationState(
        task_id=waiting.task_id,
        batch_id=waiting.batch_id,
        expected_combination_count=1,
        sessions=(session,),
        results=results,
        route="waiting_investigation_job",
        waiting_investigation_job=waiting_mapping,
        waiting_investigation_jobs=(waiting_mapping,),
    )


def _scientific_response() -> InvestigationResponse:
    """构造 wake 最终科学结果，不含任何等待路由字段。"""

    return InvestigationResponse(
        self_judgment="supported",
        narrative="wake 后读取并解释了已完成作业。",
        job_refs=("job-1",),
        operation_refs=("operation-1",),
        artifact_refs=("artifact-1",),
        observations=("终态观测。",),
        limitations=(),
    )


def test_initial_prompt_uses_unified_job_protocol_and_evidence_boundary() -> None:
    """初始 prompt 统一说明 job_id 查询、收集和证据边界。"""

    request = build_initial_investigation_request(
        "task-1",
        "batch-1",
        "combination-1",
        "判断科学问题。",
        {"strategy": "author"},
    )
    system = request["messages"][0]["content"]
    assert "同一 job_id 协议" in system
    assert "Python" in system
    assert "Gaussian" in system
    assert "提交后约 1、5、10、20、30 分钟查询，之后每约 30 分钟持续查询" in system
    assert "collect" in system
    assert "tool_capability_gap" in system
    assert "Controller" not in system
    assert "sidecar" not in system
    assert "job_family" not in system
    assert "submit_transition_state_path_job" not in system
    assert "get_transition_state_path_job" not in system
    assert "collect_transition_state_path_job" not in system
    assert "awaiting_job" not in system
    wake = build_investigation_job_wake_request(
        "task-1",
        "batch-1",
        "combination-1",
        "thread-1",
        job_id="job-1",
        operation_id="operation-1",
        submission_key="submission-1",
        action_id="action-1",
        backend_id="backend-1",
        terminal_status="succeeded",
    )
    wake_payload = json.loads(wake["messages"][1]["content"])
    assert wake["thread_id"] == "thread-1"
    assert wake_payload["job_id"] == "job-1"
    assert set(wake_payload) == {"job_id", "terminal_status"}
    assert "外部中断" in wake["messages"][0]["content"]
    assert "Controller" not in wake["messages"][0]["content"]
    assert "status_tool" not in wake["messages"][0]["content"]
    assert "collect_tool" not in wake["messages"][0]["content"]


def test_controller_polls_without_model_or_collect_and_wakes_once() -> None:
    """Controller 只查询最小状态，终态 arm 后沿原 thread 唤醒一次。"""

    gateway = _Gateway()
    assert not hasattr(gateway, "submit_job_once")
    assert not hasattr(gateway, "submit_transition_state_path_job")
    assert not hasattr(gateway, "collect_transition_state_path_job")
    assert not hasattr(gateway, "collect_execution_job")
    wakes: list[tuple[dict[str, object], str]] = []
    controller = InvestigationJobController(
        gateway,
        wake_thread=lambda request, thread: wakes.append((dict(request), thread)),
        require_status_identity=True,
    )
    waiting = _waiting(controller)
    assert waiting.status == "waiting_investigation_job"
    pending = controller.poll_waiting_job(waiting)
    assert pending.terminal is False
    assert gateway.arm_calls == []
    gateway.status = "success"
    completed = controller.poll_waiting_job(waiting)
    assert completed.terminal is True
    assert completed.wake_request is not None
    assert gateway.arm_calls == ["job-1"]
    assert len(wakes) == 1
    assert wakes[0][1] == "thread-1"
    assert json.loads(wakes[0][0]["messages"][1]["content"])["job_id"] == "job-1"
    controller.poll_waiting_job(waiting)
    assert len(wakes) == 1


def test_controller_retries_same_wake_after_provider_failure_without_resubmission() -> None:
    """wake provider 失败只重试同一 wake，不新增提交、查询或收集动作。"""

    gateway = _Gateway()
    gateway.status = "success"
    wake_attempts: list[tuple[str, str]] = []

    def wake_thread(_request: object, thread_id: str) -> None:
        """第一次模拟限流，第二次确认仍沿原 thread 唤醒。"""

        wake_attempts.append(("job-1", thread_id))
        if len(wake_attempts) == 1:
            raise RuntimeError("provider rate limited")

    controller = InvestigationJobController(
        gateway,
        wake_thread=wake_thread,
        require_status_identity=True,
    )
    waiting = _waiting(controller)
    with pytest.raises(RuntimeError, match="rate limited"):
        controller.poll_waiting_job(waiting)
    retried = controller.poll_waiting_job(waiting)

    assert retried.terminal is True
    assert wake_attempts == [("job-1", "thread-1"), ("job-1", "thread-1")]
    assert gateway.get_calls == ["job-1", "job-1", "job-1"]
    assert gateway.arm_calls == ["job-1"]
    assert not hasattr(gateway, "submit_job_once")
    assert not hasattr(gateway, "collect_execution_job")


class _CollectedArmGateway(_Gateway):
    """返回已收集而非已 arm 的 TS 状态，验证 Controller 不会误放行。"""

    def arm_investigation_job(self, waiting: InvestigationJobState, terminal_status: str):
        self.arm_calls.append(waiting.identity.job_id)
        response = self._response(waiting, terminal_status)
        response["wake_armed"] = False
        response["phase"] = "collected"
        return response


class _StaleArmGateway(_Gateway):
    """返回另一份作业身份，验证 arm 响应仍受 sidecar identity 约束。"""

    def arm_investigation_job(self, waiting: InvestigationJobState, terminal_status: str):
        self.arm_calls.append(waiting.identity.job_id)
        response = self._response(waiting, terminal_status, armed=True)
        response["job_id"] = "job-stale"
        response["job"] = {"job_id": "job-stale", "status": terminal_status}
        return response


class _SyntheticUnknownArmGateway(_Gateway):
    """只用显式 synthetic marker 表示无法核对原作业的技术失败。"""

    def __init__(self, *, marker: bool = True) -> None:
        super().__init__()
        self.status = "unknown"
        self.marker = marker

    def arm_investigation_job(self, waiting: InvestigationJobState, terminal_status: str):
        self.arm_calls.append(waiting.identity.job_id)
        response = self._response(waiting, terminal_status)
        response["phase"] = "unknown"
        response["job_status"] = "unknown"
        response["job"] = {"job_id": waiting.identity.job_id, "status": "unknown"}
        if self.marker:
            response["synthetic_terminal"] = "unknown"
        return response


def test_controller_rejects_collected_arm_response_before_wake() -> None:
    """TS arm 返回 collected 不能被当成已开放查询门禁。"""

    gateway = _CollectedArmGateway()
    controller = InvestigationJobController(gateway, require_status_identity=True)
    waiting = _waiting(controller)
    gateway.status = "success"

    with pytest.raises(InvestigationJobStatusError, match="wake_armed=true"):
        controller.poll_waiting_job(waiting)
    assert gateway.arm_calls == ["job-1"]


def test_restored_waiting_identity_releases_same_turn_slot_after_collect() -> None:
    """重启后的 wake collect 必须释放旧 turn，随后允许顺序提交 next job。"""

    gateway = _Gateway()
    gateway.status = "queued"
    original = InvestigationJobController(gateway, require_status_identity=True)
    waiting = _waiting(original)

    recovered = InvestigationJobController(gateway, require_status_identity=True)
    restored = recovered.restore_waiting_job(waiting)
    recovered.mark_job_collected(restored)

    next_state = recovered.capture_submit_receipt(
        _receipt_for("job-2"),
        task_id="task-1",
        batch_id="batch-1",
        combination_id="combination-1",
        codex_thread_id=waiting.codex_thread_id,
        turn_id=waiting.turn_id,
        scope_id=waiting.scope_id,
    )

    assert next_state.identity.job_id == "job-2"


def test_controller_rejects_stale_arm_identity_before_wake() -> None:
    """终态 arm 返回另一作业身份时必须先停止，不能唤醒错误作业。"""

    gateway = _StaleArmGateway()
    controller = InvestigationJobController(gateway, require_status_identity=True)
    waiting = _waiting(controller)
    gateway.status = "success"

    with pytest.raises(InvestigationJobStatusError, match="terminal arm.job_id"):
        controller.poll_waiting_job(waiting)
    assert gateway.arm_calls == ["job-1"]


@pytest.mark.parametrize("marker", [True, False])
def test_controller_requires_explicit_synthetic_unknown_exception(marker: bool) -> None:
    """unknown 只有网关显式声明 synthetic 才能绕过正常 arm 字段。"""

    gateway = _SyntheticUnknownArmGateway(marker=marker)
    controller = InvestigationJobController(gateway, require_status_identity=True)
    waiting = _waiting(controller)

    if marker:
        result = controller.poll_waiting_job(waiting)
        assert result.terminal is True
        assert result.job_status == "unknown"
        assert result.wake_request is not None
    else:
        with pytest.raises(InvestigationJobStatusError, match="wake_armed=true"):
            controller.poll_waiting_job(waiting)


def test_investigation_schema_prefix_only_describes_final_json() -> None:
    """最终科学 schema 前导只描述最终 JSON 合同。"""

    compiled = build_initial_investigation_request(
        "task-1",
        "batch-1",
        "combination-1",
        "判断科学问题。",
        {"strategy": "author"},
    )
    request = HarnessRequest.from_compiled(
        compiled,
        role="investigation",
        stage="initial_investigation",
        workspace=Path("/tmp/investigation-prompt-test"),
        response_schema=compiled["response_schema"],
    )
    prompt = request.prompt_text()

    assert "最终结果必须符合以下 JSON schema" in prompt
    assert "self_judgment" in prompt
    assert "submit→wait/get→终态→collect" not in prompt
    assert "仍活跃且未 collect" not in prompt
    assert "awaiting_job" not in prompt
    # HarnessRequest 是传给 adapter 的 turn 请求；Investigation runtime 不会
    # 把 response_schema 转成 SDK output_schema，具体 wire 断言由 adapter seam 覆盖。
    assert request.response_schema is not None


def test_receipt_parser_rejects_missing_job_id_without_model_identity() -> None:
    """submit 回执没有 job_id 时失败关闭，不能伪装等待状态。"""

    item = {
        "type": "mcpToolCall",
        "status": "completed",
        "tool": "submit_transition_state_path_job",
        "arguments": {"request": "submit"},
        "result": {"job_id": None},
    }
    with pytest.raises(InvestigationJobSubmissionError):
        build_investigation_job_receipt(item)


def test_rejected_submit_without_job_identity_keeps_model_turn_open() -> None:
    """提交前参数校验失败没有创建作业时，不应伪造回执或终止模型 turn。"""

    item = {
        "type": "mcpToolCall",
        "status": "completed",
        "tool": "submit_transition_state_path_job",
        "arguments": {"request": {"submission_key": "case-1-combination-1"}},
        "result": {
            "structuredContent": {
                "status": "invalid_request",
                "valid": False,
                "automatic_fallback": False,
                "error": {
                    "code": "gpu_required",
                    "message": "request.resource_limits 必须显式提供 gpu_count=1",
                },
            }
        },
    }

    assert build_investigation_job_receipt(item) is None


def test_uncertain_ts_submit_error_with_complete_identity_persists(
    tmp_path: Path,
) -> None:
    """TS 不确定错误携带完整服务端身份时可先保存可轮询等待状态。"""

    item = {
        "type": "mcpToolCall",
        "status": "completed",
        "tool": "submit_transition_state_path_job",
        "arguments": {
            "request": {
                "submission_key": "case-1-combination-1",
                "backend_id": "transition_state_path_search",
                "scientific_request": "opaque-scientific-input",
            }
        },
        "result": {
            "structuredContent": {
                "status": "invalid_request",
                "valid": False,
                "error": {
                    "code": "transition_state_path_submission_uncertain",
                    "message": "作业启动过程无法确认，已保留持久身份并阻止重新提交",
                },
                "job_id": "job_0123456789abcdef0123456789abcdef",
                "operation_id": "scope-submission-operation-1",
                "submission_key": "case-1-combination-1",
                "action_id": "search_transition_state_path",
                "backend_id": "transition_state_path_search",
                "request_fingerprint": "a" * 64,
            }
        },
    }
    receipt = build_investigation_job_receipt(item)
    assert receipt is not None
    assert receipt.identity.job_id == "job_0123456789abcdef0123456789abcdef"
    assert receipt.identity.operation_id == "scope-submission-operation-1"
    assert receipt.identity.submission_key == "case-1-combination-1"
    assert receipt.identity.action_id == "search_transition_state_path"
    assert receipt.identity.backend_id == "transition_state_path_search"
    assert receipt.result["request_fingerprint"] == "a" * 64
    assert "scientific_result" not in receipt.result
    assert "log" not in receipt.result

    store = InvestigationWaitingJobStore(tmp_path / "waiting")
    controller = InvestigationJobController(
        _Gateway(),
        require_status_identity=True,
    )
    state = controller.capture_submit_receipt(
        receipt,
        task_id="task-1",
        batch_id="batch-1",
        combination_id="combination-1",
        codex_thread_id="thread-1",
        turn_id="turn-1",
        scope_id="scope-1",
        persist_state=store.save_waiting,
    )
    persisted = store.get("task-1", "batch-1", "combination-1")
    assert persisted is not None
    assert state.identity.as_mapping() == persisted["state"]["identity"]
    assert persisted["state"]["status"] == "waiting_investigation_job"


def test_real_uncertain_ts_submit_crosses_receipt_sidecar_and_wake_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """真实 MiniChem 不确定响应可贯通回执、sidecar、Controller arm 和模型 get。"""

    scope_workspace = tmp_path / "scope"
    scope_workspace.mkdir(mode=0o700)
    scope = ScopeContext(
        task_id="task-1",
        batch_id="batch-1",
        combination_id="combination-1",
        scope_id="scope-1",
        workspace=scope_workspace,
        capability_hash="capability-1",
    )
    request = transition_state_jobs.TransitionStatePathJobRequest.model_validate(
        {
            "submission_key": "uncertain-real-boundary",
            "backend_id": transition_state_jobs.BACKEND_ID,
            "inputs": {"reactants": ["CC"], "products": ["CO"]},
            "resource_limits": {"gpu_count": 1},
        }
    )
    start_calls: list[str] = []

    def uncertain_start(**_kwargs: Any) -> object:
        """模拟 supervisor 启动边界失去确认，且不启动科学进程。"""

        start_calls.append("start")
        raise OSError("launch outcome unavailable")

    monkeypatch.setattr(transition_state_jobs, "_start_job", uncertain_start)
    monkeypatch.setattr(
        transition_state_jobs,
        "active_catalog_hash",
        lambda: "catalog-test",
    )
    monkeypatch.setattr(
        transition_state_jobs,
        "execute_traced",
        lambda _name, _arguments, function, **_kwargs: function(),
    )
    with activate_scope(scope):
        submitted = transition_state_jobs.submit_transition_state_path_job(request)
    receipt = build_investigation_job_receipt(
        {
            "type": "mcpToolCall",
            "status": "completed",
            "tool": "submit_transition_state_path_job",
            "arguments": {"request": request.model_dump(mode="json")},
            "result": {"structuredContent": submitted},
        }
    )
    assert receipt is not None

    class _RealTransitionStateGateway:
        """把 ARCHE Controller 调用接到真实 MiniChem scope 状态文件。"""

        @staticmethod
        def get_investigation_job_status(waiting: InvestigationJobState) -> Mapping[str, Any]:
            return transition_state_jobs.controller_get_transition_state_path_job(
                waiting.identity.job_id,
                scope_id=scope.scope_id,
                workspace=scope.workspace,
            )

        @staticmethod
        def arm_investigation_job(
            waiting: InvestigationJobState,
            _terminal_status: str,
        ) -> Mapping[str, Any]:
            return transition_state_jobs.controller_arm_transition_state_path_job(
                waiting.identity.job_id,
                scope_id=scope.scope_id,
                workspace=scope.workspace,
            )

    store = InvestigationWaitingJobStore(tmp_path / "waiting")
    controller = InvestigationJobController(
        _RealTransitionStateGateway(),
        require_status_identity=True,
    )
    waiting = controller.capture_submit_receipt(
        receipt,
        task_id=scope.task_id,
        batch_id=scope.batch_id,
        combination_id=scope.combination_id,
        codex_thread_id="thread-1",
        turn_id="turn-1",
        scope_id=scope.scope_id,
        persist_state=store.save_waiting,
    )
    polled = controller.poll_waiting_job(waiting)
    with activate_scope(scope):
        model_status = transition_state_jobs.get_transition_state_path_job(
            transition_state_jobs.TransitionStatePathJobStatusRequest(
                job_id=waiting.identity.job_id,
            )
        )

    persisted = store.get(scope.task_id, scope.batch_id, scope.combination_id)
    assert persisted is not None
    assert persisted["state"]["identity"] == waiting.identity.as_mapping()
    assert receipt.request_digest == submitted["request_fingerprint"]
    assert polled.terminal is True
    assert polled.job_status == "unknown"
    assert polled.wake_request is not None
    assert model_status["job_id"] == waiting.identity.job_id
    assert model_status["job_status"] == "unknown"
    assert model_status["terminal"] is True
    assert model_status["wake_armed"] is True
    assert start_calls == ["start"]


def test_uncertain_ts_submit_error_without_request_identity_stays_closed() -> None:
    """不确定错误没有真实 submission_key 时仍拒绝伪造等待状态。"""

    item = {
        "type": "mcpToolCall",
        "status": "completed",
        "tool": "submit_transition_state_path_job",
        "arguments": {"request": {"scientific_request": "opaque"}},
        "result": {
            "structuredContent": {
                "status": "invalid_request",
                "error": {"code": "transition_state_path_submission_uncertain"},
                "job_id": "job_0123456789abcdef0123456789abcdef",
                "operation_id": "scope-submission-operation-1",
            }
        },
    }
    with pytest.raises(InvestigationJobSubmissionError):
        build_investigation_job_receipt(item)


@pytest.mark.parametrize(
    "error_code",
    [
        "transition_state_path_submission_uncertain",
        "transition_state_path_identity_conflict",
    ],
)
def test_uncertain_ts_submit_error_with_complete_identity_is_parser_input(
    error_code: str,
) -> None:
    """启动异常或身份冲突只消费服务端完整身份，错误正文不参与身份恢复。"""

    item = {
        "type": "mcpToolCall",
        "status": "completed",
        "tool": "submit_transition_state_path_job",
        "arguments": {
            "request": {
                "submission_key": "case-1-combination-1",
                "backend_id": "transition_state_path_search",
                "scientific_request": "opaque-scientific-input",
            }
        },
        "result": {
            "structuredContent": {
                "status": "invalid_request",
                "error": {
                    "code": error_code,
                    "message": "不确定响应正文不参与身份恢复",
                },
                "job_id": "job_0123456789abcdef0123456789abcdef",
                "operation_id": "scope-submission-operation-1",
                "submission_key": "case-1-combination-1",
                "action_id": "search_transition_state_path",
                "backend_id": "transition_state_path_search",
                "request_fingerprint": "b" * 64,
            }
        },
    }
    receipt = build_investigation_job_receipt(item)
    assert receipt is not None
    assert receipt.identity.job_id == "job_0123456789abcdef0123456789abcdef"
    assert receipt.identity.operation_id == "scope-submission-operation-1"
    assert receipt.identity.submission_key == "case-1-combination-1"
    assert receipt.identity.action_id == "search_transition_state_path"
    assert receipt.identity.backend_id == "transition_state_path_search"
    assert receipt.result["request_fingerprint"] == "b" * 64
    assert "不确定响应正文不参与身份恢复" in receipt.result["error"]["message"]


def test_submit_receipt_status_failure_does_not_poison_same_turn_replay(
    tmp_path: Path,
) -> None:
    """首次 status 暂时失败后，同一真实 receipt 可重试且不会丢失 job identity。"""

    gateway = _TransientStatusGateway()
    store = InvestigationWaitingJobStore(tmp_path / "waiting")
    controller = InvestigationJobController(
        gateway,
        require_status_identity=True,
    )
    receipt = _receipt()
    with pytest.raises(InvestigationJobStatusError, match="temporary status outage"):
        controller.capture_submit_receipt(
            receipt,
            task_id="task-1",
            batch_id="batch-1",
            combination_id="combination-1",
            codex_thread_id="thread-1",
            turn_id="turn-1",
            scope_id="scope-1",
            persist_state=store.save_waiting,
        )

    persisted_after_failure = store.get("task-1", "batch-1", "combination-1")
    assert persisted_after_failure is not None
    assert persisted_after_failure["state"]["identity"]["job_id"] == "job-1"
    retried = controller.capture_submit_receipt(
        receipt,
        task_id="task-1",
        batch_id="batch-1",
        combination_id="combination-1",
        codex_thread_id="thread-1",
        turn_id="turn-1",
        scope_id="scope-1",
        persist_state=store.save_waiting,
    )
    assert retried.identity.job_id == "job-1"
    assert gateway.get_calls == ["job-1"]


def test_production_submit_receipt_can_persist_without_synchronous_status_query(
    tmp_path: Path,
) -> None:
    """生产 self-poll 回调先保存回执，status 瞬时失败不能中断提交 turn。"""

    gateway = _TransientStatusGateway()
    store = InvestigationWaitingJobStore(tmp_path / "waiting")
    controller = InvestigationJobController(
        gateway,
        require_status_identity=True,
    )

    waiting = controller.capture_submit_receipt(
        _receipt(),
        task_id="task-1",
        batch_id="batch-1",
        combination_id="combination-1",
        codex_thread_id="thread-1",
        turn_id="turn-1",
        scope_id="scope-1",
        persist_state=store.save_waiting,
        verify_status=False,
    )

    assert waiting.identity.job_id == "job-1"
    assert gateway.get_calls == []
    persisted = store.get("task-1", "batch-1", "combination-1")
    assert persisted is not None
    assert persisted["state"]["identity"]["job_id"] == "job-1"


def test_status_response_rejects_duplicate_job_id_and_job_status_conflicts() -> None:
    """状态响应重复身份或明确状态冲突时必须 fail closed。"""

    class _ConflictingGateway(_Gateway):
        def __init__(self, conflict: str) -> None:
            super().__init__()
            self.conflict = conflict

        def get_investigation_job_status(self, waiting: InvestigationJobState):
            response = self._response(waiting, "running")
            if self.conflict == "job_id":
                response["job"]["job_id"] = "job-other"
            else:
                response["job_status"] = "queued"
            return response

    for conflict, expected in (
        ("job_id", "job_id"),
        ("status", "job 状态"),
    ):
        controller = InvestigationJobController(
            _ConflictingGateway(conflict),
            require_status_identity=True,
        )
        waiting = controller.capture_submit_receipt(
            _receipt(),
            task_id="task-1",
            batch_id="batch-1",
            combination_id="combination-1",
            codex_thread_id="thread-1",
            turn_id=f"turn-{conflict}",
            scope_id="scope-1",
            verify_status=False,
        )
        with pytest.raises(InvestigationJobStatusError, match=expected):
            controller.poll_waiting_job(waiting)


def test_native_submit_accepts_scope_bound_minimal_status_identity() -> None:
    """统一 get 只回传 job_id/status 时仍接受已持久化的完整 submit 身份。"""

    controller = InvestigationJobController(
        _MinimalNativeStatusGateway(),
        require_status_identity=True,
    )

    waiting = controller.capture_submit_receipt(
        _native_receipt(),
        task_id="task-1",
        batch_id="batch-1",
        combination_id="combination-1",
        codex_thread_id="thread-1",
        turn_id="turn-1",
        scope_id="scope-1",
    )

    assert waiting.identity.job_id == "job-native"
    assert waiting.identity.operation_id == "operation-native"


def test_native_submit_rejects_optional_status_identity_conflict() -> None:
    """状态接口一旦返回 operation_id，冲突值仍必须在接受作业前失败。"""

    controller = InvestigationJobController(
        _MinimalNativeStatusGateway(operation_id="operation-other"),
        require_status_identity=True,
    )

    with pytest.raises(
        InvestigationJobStatusError,
        match="submit status.operation_id 与冻结作业身份不一致",
    ):
        controller.capture_submit_receipt(
            _native_receipt(),
            task_id="task-1",
            batch_id="batch-1",
            combination_id="combination-1",
            codex_thread_id="thread-1",
            turn_id="turn-1",
            scope_id="scope-1",
        )


def test_native_status_rejects_top_level_runtime_conflict_hidden_by_nested_runtime() -> None:
    """native status 的顶层 runtime 冲突不能被 nested runtime 静默覆盖。"""

    class _RuntimeConflictGateway(_MinimalNativeStatusGateway):
        def get_investigation_job_status(
            self,
            waiting: InvestigationJobState,
        ) -> Mapping[str, Any]:
            response = dict(super().get_investigation_job_status(waiting))
            response["runtime"] = "other-runtime"
            response["job"]["runtime"] = waiting.identity.runtime
            return response

    controller = InvestigationJobController(
        _RuntimeConflictGateway(),
        require_status_identity=True,
    )
    waiting = controller.capture_submit_receipt(
        _native_receipt(),
        task_id="task-1",
        batch_id="batch-1",
        combination_id="combination-1",
        codex_thread_id="thread-1",
        turn_id="turn-runtime-conflict",
        scope_id="scope-1",
        verify_status=False,
    )

    with pytest.raises(InvestigationJobStatusError, match="runtime"):
        controller.poll_waiting_job(waiting)


def test_same_turn_receipt_is_idempotent_but_uncollected_second_identity_is_rejected() -> None:
    """同一真实 turn 重放同一回执幂等，未 collect 时的第二身份被拒绝。"""

    gateway = _Gateway()
    controller = InvestigationJobController(gateway, require_status_identity=True)
    first = _waiting(controller)
    replay = controller.capture_submit_receipt(
        _receipt(),
        task_id="task-1",
        batch_id="batch-1",
        combination_id="combination-1",
        codex_thread_id="thread-1",
        turn_id="turn-1",
        scope_id="scope-1",
    )
    assert replay.as_mapping() == first.as_mapping()
    other = InvestigationJobReceipt(
        tool_name="submit_transition_state_path_job",
        identity=InvestigationJobIdentity(
            job_id="job-2",
            operation_id="operation-2",
            submission_key="submission-2",
            action_id="action-1",
            backend_id="backend-1",
        ),
        arguments={"request": "second"},
        result={
            "job_id": "job-2",
            "operation_id": "operation-2",
            "submission_key": "submission-2",
            "action_id": "action-1",
            "backend_id": "backend-1",
        },
    )
    with pytest.raises(InvestigationJobTurnClosedError):
        controller.capture_submit_receipt(
            other,
            task_id="task-1",
            batch_id="batch-1",
            combination_id="combination-1",
            codex_thread_id="thread-1",
            turn_id="turn-1",
            scope_id="scope-1",
        )


def test_same_thread_and_turn_are_isolated_by_task_batch_combination_and_scope() -> None:
    """相同 provider thread/turn 不能把不同任务的 submit 误判为同一作业。"""

    controller = InvestigationJobController(_Gateway(), require_status_identity=True)
    first = controller.capture_submit_receipt(
        _receipt_for("job-task-1"),
        task_id="task-1",
        batch_id="batch-1",
        combination_id="combination-1",
        codex_thread_id="thread-shared",
        turn_id="turn-shared",
        scope_id="scope-1",
    )
    second = controller.capture_submit_receipt(
        _receipt_for("job-task-2"),
        task_id="task-2",
        batch_id="batch-1",
        combination_id="combination-1",
        codex_thread_id="thread-shared",
        turn_id="turn-shared",
        scope_id="scope-1",
    )

    assert first.identity.job_id == "job-task-1"
    assert second.identity.job_id == "job-task-2"


def test_waiting_replacement_requires_verified_terminal_wake_source(
    tmp_path: Path,
) -> None:
    """未有终态 wake 来源的旧作业不能被 replacement 静默覆盖。"""

    controller = InvestigationJobController(_Gateway(), require_status_identity=True)
    previous = _waiting(controller)
    next_state = controller.capture_submit_receipt(
        _receipt_for("job-replacement"),
        task_id="task-1",
        batch_id="batch-1",
        combination_id="combination-1",
        codex_thread_id="thread-1",
        turn_id="turn-replacement",
        scope_id="scope-1",
    )
    store = InvestigationWaitingJobStore(tmp_path)
    store.save_waiting(previous)

    with pytest.raises(InvestigationJobStatusError, match="replacement"):
        store.replace_waiting(previous, next_state)


def test_wake_decision_digest_rejects_tampered_wake_request_and_terminal_status(
    tmp_path: Path,
) -> None:
    """恢复决策字段被篡改时，读取 sidecar 必须停止而不能继续 wake。"""

    controller = InvestigationJobController(_Gateway(), require_status_identity=True)
    waiting = _waiting(controller)
    store = InvestigationWaitingJobStore(tmp_path)
    store.save_waiting(waiting)
    store.arm_terminal(waiting, terminal_status="succeeded")
    store.record_wake_intent(
        waiting,
        {"messages": [{"role": "user", "content": "wake"}]},
    )
    raw = json.loads(store.path.read_text(encoding="utf-8"))
    key = next(iter(raw["records"]))
    raw["records"][key]["terminal_status"] = "failed"
    raw["records"][key]["wake_request"]["messages"][0]["content"] = "tampered"
    store.path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(InvestigationJobStatusError, match="摘要"):
        store.get("task-1", "batch-1", "combination-1")


def test_v2_waiting_record_without_auth_digest_only_upgrades_pure_pending(
    tmp_path: Path,
) -> None:
    """早期 v2 只允许纯 pending 记录补认证摘要，已有恢复决定时必须停止。"""

    controller = InvestigationJobController(_Gateway(), require_status_identity=True)
    waiting = _waiting(controller)
    store = InvestigationWaitingJobStore(tmp_path / "pure-pending")
    store.save_waiting(waiting)
    raw = json.loads(store.path.read_text(encoding="utf-8"))
    key = next(iter(raw["records"]))
    raw["records"][key].pop("record_auth_digest")
    store.path.write_text(json.dumps(raw), encoding="utf-8")

    upgraded = store.get("task-1", "batch-1", "combination-1")
    assert upgraded is not None
    assert isinstance(upgraded["record_auth_digest"], str)

    decided_store = InvestigationWaitingJobStore(tmp_path / "decided")
    decided_store.save_waiting(waiting)
    decided_store.arm_terminal(waiting, terminal_status="succeeded")
    decided = json.loads(decided_store.path.read_text(encoding="utf-8"))
    decided_key = next(iter(decided["records"]))
    decided["records"][decided_key].pop("record_auth_digest")
    decided_store.path.write_text(json.dumps(decided), encoding="utf-8")

    with pytest.raises(InvestigationJobStatusError, match="缺少认证摘要"):
        decided_store.get("task-1", "batch-1", "combination-1")


def test_source_wake_receipt_requires_source_job_thread_and_turn(
    tmp_path: Path,
) -> None:
    """replacement 的 source receipt 必须带原 job、原 thread 和新的 wake turn。"""

    controller = InvestigationJobController(_Gateway(), require_status_identity=True)
    previous = _waiting(controller)
    next_state = controller.capture_submit_receipt(
        _receipt_for("job-replacement"),
        task_id="task-1",
        batch_id="batch-1",
        combination_id="combination-1",
        codex_thread_id="thread-1",
        turn_id="turn-replacement",
        scope_id="scope-1",
    )
    store = InvestigationWaitingJobStore(tmp_path)
    store.save_waiting(previous)
    store.arm_terminal(previous, terminal_status="succeeded")
    store.record_wake_intent(
        previous,
        {"messages": [{"role": "user", "content": "wake"}]},
    )
    store.replace_waiting(previous, next_state)

    with pytest.raises(InvestigationJobStatusError, match="turn"):
        store.record_wake_source_after_replacement(
            previous,
            next_state,
            {"job_id": "job-1", "thread_id": "thread-1"},
        )


def test_two_job_chain_replaces_single_sidecar_key_and_keeps_next_wake_pending(
    tmp_path: Path,
) -> None:
    """wake 后的新 job 复用组合单键，旧 wake 收据不会跳过新 job 唤醒。"""

    gateway = _Gateway()
    controller = InvestigationJobController(gateway, require_status_identity=True)
    first = _waiting(controller)
    store = InvestigationWaitingJobStore(tmp_path)
    store.save_waiting(first)
    second_receipt = InvestigationJobReceipt(
        tool_name="submit_transition_state_path_job",
        identity=InvestigationJobIdentity(
            job_id="job-2",
            operation_id="operation-2",
            submission_key="submission-2",
            action_id="action-1",
            backend_id="backend-1",
        ),
        arguments={"request": "next"},
        result={
            "job_id": "job-2",
            "operation_id": "operation-2",
            "submission_key": "submission-2",
            "action_id": "action-1",
            "backend_id": "backend-1",
        },
    )
    second = controller.capture_submit_receipt(
        second_receipt,
        task_id="task-1",
        batch_id="batch-1",
        combination_id="combination-1",
        codex_thread_id="thread-1",
        turn_id="turn-2",
        scope_id="scope-1",
    )
    store.record_collected_for_replacement(first, _receipt_for("job-1"))
    store.replace_waiting(first, second)
    store.replace_waiting(first, second)
    with pytest.raises(InvestigationJobStatusError, match="wake receipt identity"):
        store.record_wake_receipt(first, {"turn_id": "stale"})
    with pytest.raises(InvestigationJobStatusError, match="清理 waiting"):
        store.clear_after_result(first)
    record = store.get("task-1", "batch-1", "combination-1")
    assert record is not None
    assert record["state"]["identity"]["job_id"] == "job-2"
    assert record["wake_status"] == "pending"
    assert "source_wake_receipt" not in record
    gateway.status = "success"
    wakes: list[tuple[dict[str, object], str]] = []
    controller.wake_thread = lambda request, thread: wakes.append((dict(request), thread))
    controller.poll_waiting_job(second)
    assert len(wakes) == 1
    assert json.loads(wakes[0][0]["messages"][1]["content"])["job_id"] == "job-2"


def test_legacy_waiting_collect_write_adds_explicit_submission_digest(
    tmp_path: Path,
) -> None:
    """旧记录首次保存 collect 许可时应写成可再次读取的 v2，而不是拒绝旧别名。"""

    controller = InvestigationJobController(_Gateway(), require_status_identity=True)
    waiting = _waiting(controller)
    legacy_state = waiting.as_mapping()
    legacy_state.pop("submission_request_digest")
    store = InvestigationWaitingJobStore(tmp_path)
    store.path.write_text(
        json.dumps(
            {
                "schema": store.schema,
                "version": store.legacy_version,
                "records": {
                    store._legacy_key(waiting): {
                        "state": legacy_state,
                        "wake_status": "pending",
                    }
                },
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    source_record = store.list_records()[0]

    record = store.record_collected_for_replacement(waiting, _receipt())

    assert record["state"]["submission_request_digest"] == waiting.request_digest
    assert record["state"]["request_digest"] == waiting.request_digest
    assert record["legacy_source_version"] == store.legacy_version
    assert (
        record["legacy_source_record_digest"] == source_record["record_digest"]
    )
    assert (
        record["legacy_source_record_auth_digest"]
        == source_record["record_auth_digest"]
    )
    assert record["collected_source"]["job_id"] == waiting.identity.job_id
    persisted = json.loads(store.path.read_text(encoding="utf-8"))
    assert persisted["version"] == store.version
    assert next(iter(persisted["records"])).startswith("v2:")
    assert (
        store.get_record_source_version(
            waiting.task_id,
            waiting.batch_id,
            waiting.combination_id,
        )
        == store.legacy_version
    )

def test_legacy_waiting_replacement_preserves_source_lineage_across_sequential_submits(
    tmp_path: Path,
) -> None:
    """旧授权来源在同一 turn 连续 collect→submit 后仍可逐边核对。"""

    controller = InvestigationJobController(_Gateway(), require_status_identity=True)
    first = _waiting(controller)
    legacy_state = first.as_mapping()
    legacy_state.pop("submission_request_digest")
    store = InvestigationWaitingJobStore(tmp_path)
    store.path.write_text(
        json.dumps(
            {
                "schema": store.schema,
                "version": store.legacy_version,
                "records": {
                    store._legacy_key(first): {
                        "state": legacy_state,
                        "wake_status": "pending",
                    }
                },
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    source_record = store.list_records()[0]
    current = first

    for number in range(2, 5):
        store.record_collected_for_replacement(
            current,
            InvestigationJobReceipt(
                tool_name=current.submit_tool,
                identity=current.identity,
                arguments={"job_id": current.identity.job_id},
                result=current.identity.as_mapping(),
            ),
        )
        digest = f"{number}" * 64
        successor = replace(
            current,
            identity=replace(
                current.identity,
                job_id=f"job-{number}",
                operation_id=f"operation-{number}",
                submission_key=f"submission-{number}",
            ),
            request_digest=digest,
            submission_request_digest=digest,
        )
        record = store.replace_waiting(current, successor)
        assert store.replace_waiting(current, successor) == record
        current = successor

    assert record["legacy_source_version"] == store.legacy_version
    assert record["legacy_source_record_digest"] == source_record["record_digest"]
    assert (
        record["legacy_source_record_auth_digest"]
        == source_record["record_auth_digest"]
    )
    history = record["pending_replacement_history"]
    assert len(history) == 3
    assert history[0]["predecessor"]["identity"]["job_id"] == first.identity.job_id
    assert history[-1]["successor"]["identity"]["job_id"] == "job-4"
    assert all(edge["authorization_kind"] == "collect" for edge in history)
    assert (
        store.get_record_source_version(
            current.task_id,
            current.batch_id,
            current.combination_id,
        )
        == store.legacy_version
    )

    tampered = json.loads(store.path.read_text(encoding="utf-8"))
    tampered_record = next(iter(tampered["records"].values()))
    tampered_record["pending_replacement_history"][1]["predecessor"]["identity"][
        "job_id"
    ] = "job-chain-drift"
    tampered_record["record_digest"] = compute_resume_record_digest(tampered_record)
    tampered_record["record_auth_digest"] = compute_resume_record_auth_digest(
        tampered_record
    )
    store.path.write_text(
        json.dumps(tampered, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(InvestigationJobStatusError, match="不连续"):
        store.list_records()


def test_legacy_waiting_write_rejects_non_sha_submission_alias(
    tmp_path: Path,
) -> None:
    """旧请求摘要不是 SHA-256 时不得借首次写回生成 v2 提交身份。"""

    controller = InvestigationJobController(_Gateway(), require_status_identity=True)
    canonical = _waiting(controller)
    legacy_state = canonical.as_mapping()
    legacy_state.pop("submission_request_digest")
    legacy_state["request_digest"] = "legacy-request-digest"
    waiting = InvestigationJobState.from_mapping(legacy_state)
    store = InvestigationWaitingJobStore(tmp_path)
    store.path.write_text(
        json.dumps(
            {
                "schema": store.schema,
                "version": store.legacy_version,
                "records": {
                    store._legacy_key(waiting): {
                        "state": legacy_state,
                        "wake_status": "pending",
                    }
                },
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(InvestigationJobStatusError, match="旧 request_digest"):
        store.record_collected_for_replacement(waiting, _receipt())

    persisted = json.loads(store.path.read_text(encoding="utf-8"))
    assert persisted["version"] == store.legacy_version


def test_legacy_waiting_read_rejects_v2_writeback_source_fields(
    tmp_path: Path,
) -> None:
    """原生 v1 记录不得携带仅由首次受控写回生成的来源字段。"""

    controller = InvestigationJobController(_Gateway(), require_status_identity=True)
    waiting = _waiting(controller)
    legacy_state = waiting.as_mapping()
    legacy_state.pop("submission_request_digest")
    store = InvestigationWaitingJobStore(tmp_path)
    store.path.write_text(
        json.dumps(
            {
                "schema": store.schema,
                "version": store.legacy_version,
                "records": {
                    store._legacy_key(waiting): {
                        "state": legacy_state,
                        "wake_status": "pending",
                        "legacy_source_version": store.legacy_version,
                        "legacy_source_record_digest": "a" * 64,
                        "legacy_source_record_auth_digest": "b" * 64,
                    }
                },
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(InvestigationJobStatusError, match="含未知字段"):
        store.list_records()


def test_legacy_waiting_read_rejects_v2_submission_digest(
    tmp_path: Path,
) -> None:
    """原生 v1 记录不得预先携带 v2 的显式提交摘要。"""

    controller = InvestigationJobController(_Gateway(), require_status_identity=True)
    waiting = _waiting(controller)
    store = InvestigationWaitingJobStore(tmp_path)
    store.path.write_text(
        json.dumps(
            {
                "schema": store.schema,
                "version": store.legacy_version,
                "records": {
                    store._legacy_key(waiting): {
                        "state": waiting.as_mapping(),
                        "wake_status": "pending",
                    }
                },
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        InvestigationJobStatusError,
        match="submission_request_digest 只允许 v2",
    ):
        store.list_records()


def test_waiting_replacement_rejects_reused_submission_identity(tmp_path: Path) -> None:
    """wake 后即使 turn_id 改变，也不能把旧作业身份当作新作业。"""

    controller = InvestigationJobController(_Gateway(), require_status_identity=True)
    previous = _waiting(controller)
    store = InvestigationWaitingJobStore(tmp_path)
    store.save_waiting(previous)
    reused = replace(previous, turn_id="turn-replacement")

    with pytest.raises(InvestigationJobStatusError, match="不能复用旧 job/operation/submission"):
        store.replace_waiting(previous, reused)


def test_waiting_store_serializes_concurrent_distinct_combination_writes(
    tmp_path: Path,
) -> None:
    """五个 Investigation future 并行保存时不能互相覆盖等待记录。"""

    store = InvestigationWaitingJobStore(tmp_path)
    barrier = threading.Barrier(5)

    def make_state(index: int) -> InvestigationJobState:
        """构造一个属于不同 combination 的独立 waiting 快照。"""

        return InvestigationJobState(
            task_id="task-1",
            batch_id="batch-1",
            combination_id=f"combination-{index}",
            codex_thread_id=f"thread-{index}",
            turn_id=f"turn-{index}",
            status="waiting_investigation_job",
            identity=InvestigationJobIdentity(
                job_id=f"job-{index}",
                operation_id=f"operation-{index}",
                submission_key=f"submission-{index}",
                action_id="action-1",
                backend_id="backend-1",
            ),
            request_digest=f"{index:064x}",
            scope_id="scope-1",
        )

    def save(index: int) -> Mapping[str, Any]:
        """让所有 writer 同时进入 store，覆盖 read-modify-write 竞争窗口。"""

        barrier.wait(timeout=10)
        return store.save_waiting(make_state(index))

    with ThreadPoolExecutor(max_workers=5) as executor:
        tuple(executor.map(save, range(5)))
    records = store.list_records()
    assert len(records) == 5
    assert {
        record["state"]["combination_id"]
        for record in records
    } == {f"combination-{index}" for index in range(5)}
