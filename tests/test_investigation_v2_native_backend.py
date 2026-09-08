"""V2-native Investigation Harness backend 的确定性纵向切片测试。

测试只使用内存 fake transport，观察 Harness 的公开结果和
``HarnessRecoveryStore`` typed snapshot；不会启动模型、MCP、MiniChem 或化学作业。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pytest

from chemistry_multiagent.investigation.harness_recovery_store import (
    HarnessLifecycle,
    HarnessRunIdentity,
    HarnessRecoveryStore,
    SuspendInvocation,
)
from chemistry_multiagent.investigation.harness_runtime import (
    InvestigationHarness,
    InvestigationHarnessRequest,
    Suspended,
)
from chemistry_multiagent.investigation.investigation_backend import (
    EvidenceResultRejected,
    NativeAction,
    NativeProviderTransport,
    NativeTurnResult,
    InvestigationSessionRunner,
)
from chemistry_multiagent.memory.v2_identities import CreateInvestigationContext


_INPUT_DIGEST = "a" * 64
_CONTEXT_BINDING = CreateInvestigationContext(
    "context-1", "batch-1"
).to_dict()


def _request() -> InvestigationHarnessRequest:
    return InvestigationHarnessRequest(
        research_task_id="task-1",
        module_invocation_id="module-1",
        module_kind="hypothesis-investigation",
        input_scientific_record_ids=("selection-1",),
        module_input_digest=_INPUT_DIGEST,
        context_binding=_CONTEXT_BINDING,
    )


def test_native_action_rejects_harness_private_identity_payload() -> None:
    with pytest.raises(ValueError, match="私有字段"):
        NativeAction(
            "action-1",
            "submit_job",
            {"kind": "gaussian", "scope_id": "scope-secret"},
        )


def _scientific_result(*, job_refs: list[str] | None = None) -> dict[str, Any]:
    return {
        "self_judgment": "supported",
        "narrative": "验证结果支持当前假设。",
        "job_refs": [] if job_refs is None else job_refs,
        "operation_refs": [] if job_refs is None else ["operation-1"],
        "artifact_refs": [] if job_refs is None else ["artifact-1"],
        "observations": ["观察到稳定的目标信号。"],
        "limitations": ["仅覆盖当前输入组合。"],
    }


@dataclass
class FakeExecutionTransport:
    """记录一次物理提交，并允许测试推进同一个 job 的终态。"""

    status: str = "pending"
    collect_status: str | None = None
    query_error: BaseException | None = None

    def __post_init__(self) -> None:
        self.submissions: list[tuple[NativeAction, str]] = []
        self.queries: list[str] = []
        self.collections: list[str] = []

    def submit_job(self, action: NativeAction, *, session: Any) -> dict[str, Any]:
        self.submissions.append((action, session.investigation_session_id))
        return {
            "job_id": "job-1",
            "operation_id": "operation-1",
            "submission_key": "submission-1",
            "request_digest": "b" * 64,
            "status": self.status,
        }

    def query_job(self, job: Any, *, session: Any) -> dict[str, Any]:
        self.queries.append(job.minichem_job_id)
        if self.query_error is not None:
            raise self.query_error
        return {"job_id": job.minichem_job_id, "status": self.status}

    def collect_job(self, job: Any, *, session: Any) -> dict[str, Any]:
        self.collections.append(job.minichem_job_id)
        status = self.collect_status or self.status
        return {
            "job_id": job.minichem_job_id,
            "operation_id": job.minichem_operation_id,
            "submission_key": job.minichem_submission_key,
            "status": status,
            "artifact_refs": ["artifact-1"] if status == "completed" else [],
            "error_code": "gaussian_failed" if status == "failed" else None,
        }


@dataclass
class ContractGuardExecution(FakeExecutionTransport):
    """只在新 submit 前允许检查局部合同的 fake；恢复路径禁止全局 readiness。"""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.contract_checks: list[str] = []
        self.readiness_checks = 0

    def check_submission_contract(self, action: NativeAction, *, session: Any) -> bool:
        self.contract_checks.append(action.action_id)
        return True

    def preflight_scope_readiness(self, *args: Any, **kwargs: Any) -> None:
        self.readiness_checks += 1
        raise AssertionError("恢复旧 job 不得执行全局 MiniChem readiness")


class ScopeConflictExecution(FakeExecutionTransport):
    """返回不同 scope 的旧 job 状态，模拟权威 scope 漂移。"""

    def query_job(self, job: Any, *, session: Any) -> dict[str, Any]:
        self.queries.append(job.minichem_job_id)
        return {
            "job_id": job.minichem_job_id,
            "scope_id": "scope-other",
            "status": self.status,
        }


class FakeProviderTransport:
    """按调用顺序返回预设 provider turn，并保留内部调用证据。"""

    def __init__(self, turns: list[NativeTurnResult | Callable[..., NativeTurnResult]]) -> None:
        self._turns = turns
        self.calls: list[dict[str, Any]] = []

    def run_turn(
        self,
        request: dict[str, Any],
        *,
        session: Any,
        action_results: tuple[dict[str, Any], ...],
        is_resume: bool,
        thread_id: str | None = None,
    ) -> NativeTurnResult:
        self.calls.append(
            {
                "request": request,
                "investigation_session_id": session.investigation_session_id,
                "action_results": action_results,
                "is_resume": is_resume,
                "thread_id": thread_id,
            }
        )
        next_turn = self._turns[len(self.calls) - 1]
        if callable(next_turn):
            return next_turn(action_results=action_results, session=session)
        return next_turn


class SpecializedProviderTransport(NativeProviderTransport):
    """模拟真实 runtime：继承 generic Protocol，但只实现 V2 专用入口。"""

    def __init__(self) -> None:
        self.calls = 0

    def run_investigation_turn(self, _request: dict[str, Any], **_kwargs: Any) -> NativeTurnResult:
        self.calls += 1
        if self.calls == 1:
            return NativeTurnResult(
                thread_id="thread-specialized",
                turn_id="turn-specialized-1",
                actions=(NativeAction("specialized-action", "submit_job", {"kind": "gaussian"}),),
            )
        return NativeTurnResult(
            thread_id="thread-specialized",
            turn_id="turn-specialized-2",
            result=_scientific_result(job_refs=["job-1"]),
        )


class InterleavedCallbackProvider:
    """在同一 provider turn 内按真实生命周期交错报告两个作业。"""

    def run_investigation_turn(
        self,
        _request: dict[str, Any],
        *,
        scope_id: str,
        on_job_accepted: Callable[[dict[str, Any]], None],
        on_job_collected: Callable[[dict[str, Any]], None],
        **_: Any,
    ) -> NativeTurnResult:
        del scope_id
        for job_id, operation_id, submission_key in (
            ("job-a", "operation-a", "submission-a"),
            ("job-b", "operation-b", "submission-b"),
        ):
            accepted = {
                "identity": {
                    "job_id": job_id,
                    "operation_id": operation_id,
                    "submission_key": submission_key,
                },
                "request_digest": "b" * 64,
                "status": "pending",
            }
            collected = {**accepted, "status": "completed"}
            on_job_accepted(accepted)
            on_job_collected(collected)
        return NativeTurnResult(
            thread_id="thread-interleaved",
            turn_id="turn-interleaved-1",
            result=_scientific_result(job_refs=["job-a", "job-b"]),
        )


class TerminalCallbackProvider:
    """首轮报告 interrupted 终态，恢复轮在同一 thread 返回科学结果。"""

    def __init__(self) -> None:
        self.calls = 0

    def run_investigation_turn(
        self,
        _request: dict[str, Any],
        *,
        on_thread_ready: Callable[[str], None],
        on_turn_started: Callable[[str, str], None],
        on_terminal_turn: Callable[[str, str, str], None],
        on_job_accepted: Callable[[dict[str, Any]], None],
        on_job_collected: Callable[[dict[str, Any]], None],
        **_: Any,
    ) -> NativeTurnResult:
        self.calls += 1
        on_thread_ready("thread-terminal")
        if self.calls == 1:
            on_turn_started("thread-terminal", "turn-terminal-1")
            on_terminal_turn(
                "thread-terminal",
                "turn-terminal-1",
                "interrupted",
            )
            raise RuntimeError("provider 已报告 interrupted 终态")
        receipt = {
            "identity": {
                "job_id": "job-1",
                "operation_id": "operation-1",
                "submission_key": "submission-1",
            },
            "request_digest": "b" * 64,
            "status": "pending",
        }
        on_job_accepted(receipt)
        on_job_collected({**receipt, "status": "completed"})
        return NativeTurnResult(
            thread_id="thread-terminal",
            turn_id="turn-terminal-2",
            result=_scientific_result(job_refs=["job-1"]),
        )


class FailedTerminalCallbackProvider:
    """报告 failed 终态后断流；Harness 应暂停并复用原 thread 恢复。"""

    def __init__(self) -> None:
        self.calls = 0

    def run_investigation_turn(
        self,
        _request: dict[str, Any],
        *,
        on_thread_ready: Callable[[str], None],
        on_turn_started: Callable[[str, str], None],
        on_terminal_turn: Callable[[str, str, str], None],
        on_job_accepted: Callable[[dict[str, Any]], None],
        on_job_collected: Callable[[dict[str, Any]], None],
        **_: Any,
    ) -> NativeTurnResult:
        self.calls += 1
        on_thread_ready("thread-failed-terminal")
        if self.calls == 1:
            on_turn_started("thread-failed-terminal", "turn-failed-terminal-1")
            on_terminal_turn(
                "thread-failed-terminal",
                "turn-failed-terminal-1",
                "failed",
            )
            raise RuntimeError("模拟 provider 报告 failed 后流连接断开")
        receipt = {
            "identity": {
                "job_id": "job-1",
                "operation_id": "operation-1",
                "submission_key": "submission-1",
            },
            "request_digest": "b" * 64,
            "status": "pending",
        }
        on_job_accepted(receipt)
        on_job_collected({**receipt, "status": "completed"})
        return NativeTurnResult(
            thread_id="thread-failed-terminal",
            turn_id="turn-failed-terminal-2",
            result=_scientific_result(job_refs=["job-1"]),
        )


class CallbackCrashRecoveryProvider:
    """首轮在 callback 已产生两个物理作业后中断，恢复同一物理 turn。"""

    def __init__(self) -> None:
        self.calls = 0

    @staticmethod
    def _receipt(
        job_id: str,
        operation_id: str,
        submission_key: str,
        *,
        status: str,
    ) -> dict[str, Any]:
        return {
            "identity": {
                "job_id": job_id,
                "operation_id": operation_id,
                "submission_key": submission_key,
            },
            "request_digest": "b" * 64,
            "status": status,
        }

    def run_investigation_turn(
        self,
        request: dict[str, Any],
        *,
        on_turn_started: Callable[[str, str], None],
        on_job_accepted: Callable[[dict[str, Any]], None],
        on_job_collected: Callable[[dict[str, Any]], None],
        **_: Any,
    ) -> NativeTurnResult:
        self.calls += 1
        assert request.get("thread_id") in {None, "thread-callback-crash"}
        on_turn_started("thread-callback-crash", "turn-callback-crash-1")
        if self.calls == 1:
            first = self._receipt(
                "job-a",
                "operation-a",
                "submission-a",
                status="pending",
            )
            second = self._receipt(
                "job-b",
                "operation-b",
                "submission-b",
                status="pending",
            )
            on_job_accepted(first)
            on_job_collected({**first, "status": "completed"})
            on_job_accepted(second)
            raise RuntimeError("模拟 provider transport 在第二个作业提交后中断")
        return NativeTurnResult(
            thread_id="thread-callback-crash",
            turn_id="turn-callback-crash-1",
            result=_scientific_result(job_refs=["job-a", "job-b"]),
        )


class EvidenceRepairReplayProvider:
    """先重放已完成 turn，再在同一 thread 修正被拒绝的 Result 引用。"""

    def __init__(self) -> None:
        self.model_turn_calls = 0
        self.completed_turn_replay_calls = 0

    @staticmethod
    def _receipt(*, status: str) -> dict[str, Any]:
        return {
            "identity": {
                "job_id": "job-1",
                "operation_id": "operation-1",
                "submission_key": "submission-1",
            },
            "request_digest": "b" * 64,
            "status": status,
        }

    def run_investigation_turn(
        self,
        request: dict[str, Any],
        *,
        on_job_accepted: Callable[[dict[str, Any]], None],
        on_job_collected: Callable[[dict[str, Any]], None],
        **_: Any,
    ) -> NativeTurnResult:
        self.model_turn_calls += 1
        if self.model_turn_calls == 1:
            receipt = self._receipt(status="pending")
            on_job_accepted(receipt)
            on_job_collected({**receipt, "status": "completed"})
            return NativeTurnResult(
                thread_id="thread-evidence-repair",
                turn_id="turn-evidence-repair-1",
                result=_scientific_result(job_refs=["job-1"]),
            )
        assert tuple(request.get("action_results", ())) == (
            {
                "kind": "result_validation",
                "status": "failed",
                "reason_code": "evidence_references_rejected",
                "invalid_references": [
                    {
                        "kind": "operation",
                        "operation_id": "operation-running",
                        "reason_code": "operation_not_completed",
                    }
                ],
            },
        )
        return NativeTurnResult(
            thread_id="thread-evidence-repair",
            turn_id="turn-evidence-repair-2",
            result=_scientific_result(job_refs=["job-1"]),
        )

    def replay_completed_turn(
        self,
        _request: dict[str, Any],
        *,
        replay_turn_id: str,
        on_job_accepted: Callable[[dict[str, Any]], None],
        on_job_collected: Callable[[dict[str, Any]], None],
        **_: Any,
    ) -> NativeTurnResult:
        self.completed_turn_replay_calls += 1
        assert replay_turn_id == "turn-evidence-repair-1"
        receipt = self._receipt(status="pending")
        on_job_accepted(receipt)
        on_job_collected({**receipt, "status": "completed"})
        return NativeTurnResult(
            thread_id="thread-evidence-repair",
            turn_id="turn-evidence-repair-1",
            # 真实 Codex runtime 的普通完成 turn 不把自身 receipt 请求摘要
            # 冒充 Harness attempt 摘要；只读 replay 才返回 receipt 原始摘要。
            request_digest="f" * 64,
            result=_scientific_result(job_refs=["job-1"]),
        )


class CompletedTurnReplayProvider:
    """模拟旧版先报告全部 accepted 后报告 collected 的已完成 turn。"""

    def __init__(self) -> None:
        self.model_turn_calls = 0
        self.completed_turn_replay_calls = 0

    @staticmethod
    def _receipt(
        job_id: str,
        operation_id: str,
        submission_key: str,
        *,
        status: str,
    ) -> dict[str, Any]:
        return {
            "identity": {
                "job_id": job_id,
                "operation_id": operation_id,
                "submission_key": submission_key,
            },
            "request_digest": "b" * 64,
            "status": status,
        }

    def run_investigation_turn(
        self,
        _request: dict[str, Any],
        *,
        on_job_accepted: Callable[[dict[str, Any]], None],
        on_job_collected: Callable[[dict[str, Any]], None],
        **_: Any,
    ) -> NativeTurnResult:
        self.model_turn_calls += 1
        receipts = (
            self._receipt("job-a", "operation-a", "submission-a", status="pending"),
            self._receipt("job-b", "operation-b", "submission-b", status="pending"),
        )
        # 这是合并前留下的真实失败形状：provider turn 已完成，但 callback
        # 按 accepted-all、collected-all 到达，旧 backend 只持久化了第一个 job。
        for receipt in receipts:
            on_job_accepted(receipt)
        for receipt in receipts:
            on_job_collected({**receipt, "status": "completed"})
        return NativeTurnResult(
            thread_id="thread-replay",
            turn_id="turn-replay-1",
            result=_scientific_result(job_refs=["job-a", "job-b"]),
        )

    def replay_completed_turn(
        self,
        _request: dict[str, Any],
        *,
        replay_turn_id: str,
        on_job_accepted: Callable[[dict[str, Any]], None],
        on_job_collected: Callable[[dict[str, Any]], None],
        **_: Any,
    ) -> NativeTurnResult:
        self.completed_turn_replay_calls += 1
        assert replay_turn_id == "turn-replay-1"
        for job_id, operation_id, submission_key in (
            ("job-a", "operation-a", "submission-a"),
            ("job-b", "operation-b", "submission-b"),
        ):
            receipt = self._receipt(
                job_id,
                operation_id,
                submission_key,
                status="pending",
            )
            on_job_accepted(receipt)
            on_job_collected({**receipt, "status": "completed"})
        return NativeTurnResult(
            thread_id="thread-replay",
            turn_id="turn-replay-1",
            result=_scientific_result(job_refs=["job-a", "job-b"]),
        )


class ReplayCapableSequentialProvider:
    """具备 replay 能力，但普通 waiting job 仍应走 query/collect 后续 turn。"""

    def __init__(self) -> None:
        self.model_turn_calls = 0
        self.completed_turn_replay_calls = 0

    def run_investigation_turn(self, _request: dict[str, Any], **_: Any) -> NativeTurnResult:
        self.model_turn_calls += 1
        if self.model_turn_calls == 1:
            return NativeTurnResult(
                thread_id="thread-sequential",
                turn_id="turn-sequential-1",
                actions=(NativeAction("action-1", "submit_job", {"kind": "gaussian"}),),
            )
        return NativeTurnResult(
            thread_id="thread-sequential",
            turn_id="turn-sequential-2",
            result=_scientific_result(job_refs=["job-1"]),
        )

    def replay_completed_turn(self, _request: dict[str, Any], **_: Any) -> NativeTurnResult:
        self.completed_turn_replay_calls += 1
        raise AssertionError("普通 job_not_terminal 恢复不得重放 provider turn")


def _backend(
    tmp_path: Path,
    provider: FakeProviderTransport,
    execution: FakeExecutionTransport,
) -> tuple[InvestigationHarness, HarnessRecoveryStore]:
    store = HarnessRecoveryStore(tmp_path / "harness-recovery.sqlite3")
    backend = InvestigationSessionRunner(
        recovery_store=store,
        provider_transport=provider,
        execution_transport=execution,
    )
    return InvestigationHarness(tmp_path / "runtime", backend=backend, recovery_store=store), store


def test_native_backend_prefers_specialized_runtime_entry_over_protocol_stub(tmp_path: Path) -> None:
    """真实 runtime 的专用入口不能被 Protocol 的空 generic 桩遮蔽。"""

    provider = SpecializedProviderTransport()
    execution = FakeExecutionTransport(status="completed")
    harness, _store = _backend(tmp_path, provider, execution)

    result = harness.start(_request())

    assert result.status == "Completed"
    assert provider.calls == 2


def test_native_start_preserves_interleaved_job_callback_order(tmp_path: Path) -> None:
    """同一 turn 的 accepted/collected 回调按到达顺序推进单活动 job。"""

    provider = InterleavedCallbackProvider()
    execution = FakeExecutionTransport(status="pending")
    harness, store = _backend(tmp_path, provider, execution)  # type: ignore[arg-type]

    outcome = harness.start(_request())

    assert outcome.status == "Completed"
    snapshot = store.read_snapshot(
        HarnessRunIdentity(
            research_task_id="task-1",
            module_invocation_id="module-1",
            harness_invocation_id=_request().harness_invocation_id,
            module_input_digest=_INPUT_DIGEST,
        )
    )
    assert snapshot is not None
    assert [job.minichem_job_id for job in snapshot.jobs] == ["job-a", "job-b"]
    assert snapshot.collected_job_ids == ("job-a", "job-b")
    assert snapshot.active_job is None


def test_native_backend_persists_terminal_callback_before_safe_retry(
    tmp_path: Path,
) -> None:
    """interrupted turn 必须先关闭 attempt，下一次恢复才能安全创建新 turn。"""

    provider = TerminalCallbackProvider()
    harness, store = _backend(  # type: ignore[arg-type]
        tmp_path,
        provider,
        FakeExecutionTransport(),
    )
    request = _request()

    first = harness.start(request)

    assert first.status == "Suspended"
    assert first.reason_code == "provider_interrupted"
    identity = HarnessRunIdentity(
        research_task_id=request.research_task_id,
        module_invocation_id=request.module_invocation_id,
        harness_invocation_id=request.harness_invocation_id,
        module_input_digest=request.module_input_digest,
    )
    snapshot = store.read_snapshot(identity)
    assert snapshot is not None
    assert snapshot.active_attempt is None
    assert snapshot.last_attempt is not None
    assert snapshot.last_attempt.model_thread_id == "thread-terminal"
    assert snapshot.last_attempt.model_turn_id == "turn-terminal-1"

    second = harness.resume(harness.harness_invocation_id_for(request), request)

    assert (
        second.status,
        getattr(second, "reason_code", None),
        getattr(second.recovery_verification, "reason_code", None),
    ) == (
        "Completed",
        None,
        "evidence_verified",
    )
    assert provider.calls == 2


def test_native_backend_failed_terminal_callback_remains_resumable(
    tmp_path: Path,
) -> None:
    """failed turn 已由 callback 关闭时不得因重复关闭升级成完整性冲突。"""

    provider = FailedTerminalCallbackProvider()
    harness, store = _backend(  # type: ignore[arg-type]
        tmp_path,
        provider,
        FakeExecutionTransport(),
    )
    request = _request()

    first = harness.start(request)

    assert first.status == "Suspended"
    assert first.reason_code == "provider_failed"
    identity = HarnessRunIdentity(
        research_task_id=request.research_task_id,
        module_invocation_id=request.module_invocation_id,
        harness_invocation_id=request.harness_invocation_id,
        module_input_digest=request.module_input_digest,
    )
    snapshot = store.read_snapshot(identity)
    assert snapshot is not None
    assert snapshot.state is HarnessLifecycle.SUSPENDED
    assert snapshot.active_attempt is None
    assert snapshot.last_attempt is not None
    assert snapshot.last_attempt.model_thread_id == "thread-failed-terminal"
    assert snapshot.last_attempt.model_turn_id == "turn-failed-terminal-1"

    second = harness.resume(first.harness_invocation_id, request)

    assert second.status == "Completed"
    assert provider.calls == 2


def test_job_callbacks_are_durable_before_provider_transport_returns(
    tmp_path: Path,
) -> None:
    """transport 中断不能丢掉此前已接受或已收集的真实作业身份。"""

    provider = CallbackCrashRecoveryProvider()
    execution = FakeExecutionTransport(status="pending")
    harness, store = _backend(  # type: ignore[arg-type]
        tmp_path,
        provider,
        execution,
    )
    request = _request()
    identity = HarnessRunIdentity(
        research_task_id=request.research_task_id,
        module_invocation_id=request.module_invocation_id,
        harness_invocation_id=request.harness_invocation_id,
        module_input_digest=request.module_input_digest,
    )

    interrupted = harness.start(request)

    assert interrupted.status == "Suspended"
    assert interrupted.reason_code == "provider_contract_unavailable"
    interrupted_snapshot = store.read_snapshot(identity)
    assert [job.minichem_job_id for job in interrupted_snapshot.jobs] == ["job-a", "job-b"]
    assert interrupted_snapshot.collected_job_ids == ("job-a",)
    assert interrupted_snapshot.active_job is not None
    assert interrupted_snapshot.active_job.minichem_job_id == "job-b"
    assert interrupted_snapshot.active_attempt is not None
    assert interrupted_snapshot.active_attempt.model_turn_id == "turn-callback-crash-1"

    execution.status = "completed"
    completed = harness.resume(interrupted.harness_invocation_id, request)

    assert completed.status == "Completed"
    assert provider.calls == 2
    assert execution.submissions == []
    assert execution.queries == ["job-b"]
    assert execution.collections == ["job-b"]
    completed_snapshot = store.read_snapshot(identity)
    assert completed_snapshot.collected_job_ids == ("job-a", "job-b")
    assert completed_snapshot.active_job is None
    assert completed_snapshot.active_attempt is None


def test_replayed_completed_turn_can_continue_after_evidence_rejects_result_refs(
    tmp_path: Path,
) -> None:
    """引用无效属于 Result 修正，不得无限重放同一个无效 completed turn。"""

    class _EvidenceVerifier:
        def __init__(self) -> None:
            self.calls = 0

        def verify_evidence(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            self.calls += 1
            if self.calls == 1:
                raise ValueError("模拟 Evidence owner 首次暂时不可读")
            if self.calls == 2:
                raise EvidenceResultRejected(
                    reason_code="evidence_references_rejected",
                    invalid_references=(
                        {
                            "kind": "operation",
                            "operation_id": "operation-running",
                            "reason_code": "operation_not_completed",
                        },
                    ),
                )
            return {
                "verification_ref": "verification-evidence-repair",
                "verification_digest": "c" * 64,
            }

    provider = EvidenceRepairReplayProvider()
    verifier = _EvidenceVerifier()
    execution = FakeExecutionTransport(
        status="pending",
        query_error=AssertionError("Result 引用修正不得重新查询或提交 MiniChem job"),
    )
    store = HarnessRecoveryStore(tmp_path / "harness-recovery.sqlite3")
    backend = InvestigationSessionRunner(
        recovery_store=store,
        provider_transport=provider,
        execution_transport=execution,
        evidence_verifier=verifier,
        require_evidence_verifier=True,
    )
    harness = InvestigationHarness(
        tmp_path / "runtime",
        backend=backend,
        recovery_store=store,
    )

    first = harness.start(_request())
    assert first.status == "Suspended"
    assert first.reason_code == "evidence_verification_missing"

    repaired = harness.resume(first.harness_invocation_id, _request())

    assert repaired.status == "Completed"
    assert provider.model_turn_calls == 2
    assert provider.completed_turn_replay_calls == 1
    assert verifier.calls == 3
    assert execution.submissions == []
    assert execution.queries == []
    assert execution.collections == []


def test_authoritative_evidence_verifier_owns_scoped_job_refs_missing_from_recovery_snapshot(
    tmp_path: Path,
) -> None:
    """中断 turn 遗漏的 scoped job 应交给 9012 verifier 核验，不能被本地账本抢先拒绝。"""

    class _Provider:
        def run_investigation_turn(
            self,
            _request: dict[str, Any],
            *,
            on_job_accepted: Callable[[dict[str, Any]], None],
            on_job_collected: Callable[[dict[str, Any]], None],
            **_: Any,
        ) -> NativeTurnResult:
            receipt = {
                "identity": {
                    "job_id": "job-recorded",
                    "operation_id": "operation-recorded",
                    "submission_key": "submission-recorded",
                },
                "request_digest": "b" * 64,
                "status": "pending",
            }
            on_job_accepted(receipt)
            on_job_collected({**receipt, "status": "completed"})
            return NativeTurnResult(
                thread_id="thread-authoritative-job-refs",
                turn_id="turn-authoritative-job-refs-1",
                result=_scientific_result(
                    job_refs=["job-recorded", "job-from-interrupted-turn"]
                ),
            )

    class _EvidenceVerifier:
        def __init__(self) -> None:
            self.calls = 0

        def verify_evidence(
            self,
            response: Any,
            *,
            jobs: tuple[Any, ...],
            **_: Any,
        ) -> dict[str, Any]:
            self.calls += 1
            assert tuple(response.job_refs) == (
                "job-recorded",
                "job-from-interrupted-turn",
            )
            assert tuple(job.minichem_job_id for job in jobs) == ("job-recorded",)
            return {
                "verification_ref": "verification-authoritative-job-refs",
                "verification_digest": "c" * 64,
                "evidence_items": [
                    {"job_ref": "job-recorded", "status": "completed"},
                    {
                        "job_ref": "job-from-interrupted-turn",
                        "status": "completed",
                    },
                ],
            }

    verifier = _EvidenceVerifier()
    store = HarnessRecoveryStore(tmp_path / "harness-recovery.sqlite3")
    backend = InvestigationSessionRunner(
        recovery_store=store,
        provider_transport=_Provider(),
        execution_transport=FakeExecutionTransport(status="pending"),
        evidence_verifier=verifier,
        require_evidence_verifier=True,
    )
    harness = InvestigationHarness(
        tmp_path / "runtime",
        backend=backend,
        recovery_store=store,
    )

    completed = harness.start(_request())

    assert completed.status == "Completed"
    assert verifier.calls == 1


def test_unknown_result_job_ref_without_evidence_verifier_remains_suspended(
    tmp_path: Path,
) -> None:
    """没有权威 verifier 时，本地账本之外的 job 引用仍必须 fail-closed。"""

    class _Provider:
        def run_investigation_turn(
            self,
            _request: dict[str, Any],
            *,
            on_job_accepted: Callable[[dict[str, Any]], None],
            on_job_collected: Callable[[dict[str, Any]], None],
            **_: Any,
        ) -> NativeTurnResult:
            receipt = {
                "identity": {
                    "job_id": "job-recorded",
                    "operation_id": "operation-recorded",
                    "submission_key": "submission-recorded",
                },
                "request_digest": "b" * 64,
                "status": "pending",
            }
            on_job_accepted(receipt)
            on_job_collected({**receipt, "status": "completed"})
            return NativeTurnResult(
                thread_id="thread-unverified-job-ref",
                turn_id="turn-unverified-job-ref-1",
                result=_scientific_result(
                    job_refs=["job-recorded", "job-without-authority"]
                ),
            )

    store = HarnessRecoveryStore(tmp_path / "harness-recovery.sqlite3")
    backend = InvestigationSessionRunner(
        recovery_store=store,
        provider_transport=_Provider(),
        execution_transport=FakeExecutionTransport(status="pending"),
    )
    harness = InvestigationHarness(
        tmp_path / "runtime",
        backend=backend,
        recovery_store=store,
    )

    suspended = harness.start(_request())

    assert suspended.status == "Suspended"
    assert suspended.reason_code == "recovery_evidence_missing"
    assert suspended.recovery_verification.reason_code == "result_job_evidence_missing"


def test_completed_turn_replay_with_changed_thread_remains_fatal(
    tmp_path: Path,
) -> None:
    """摘要层允许不同，但 completed turn 的物理 thread 身份仍必须完全一致。"""

    class _ChangedThreadReplayProvider(EvidenceRepairReplayProvider):
        def replay_completed_turn(
            self,
            _request: dict[str, Any],
            *,
            on_job_accepted: Callable[[dict[str, Any]], None],
            on_job_collected: Callable[[dict[str, Any]], None],
            **_: Any,
        ) -> NativeTurnResult:
            receipt = self._receipt(status="pending")
            on_job_accepted(receipt)
            on_job_collected({**receipt, "status": "completed"})
            return NativeTurnResult(
                thread_id="thread-replay-conflict",
                turn_id="turn-evidence-repair-1",
                request_digest="f" * 64,
                result=_scientific_result(job_refs=["job-1"]),
            )

    class _UnavailableEvidenceVerifier:
        def verify_evidence(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            raise ValueError("模拟 Evidence owner 暂时不可读")

    provider = _ChangedThreadReplayProvider()
    store = HarnessRecoveryStore(tmp_path / "harness-recovery.sqlite3")
    backend = InvestigationSessionRunner(
        recovery_store=store,
        provider_transport=provider,
        execution_transport=FakeExecutionTransport(status="pending"),
        evidence_verifier=_UnavailableEvidenceVerifier(),
        require_evidence_verifier=True,
    )
    harness = InvestigationHarness(
        tmp_path / "runtime",
        backend=backend,
        recovery_store=store,
    )

    paused = harness.start(_request())
    failed = harness.resume(paused.harness_invocation_id, _request())

    assert paused.status == "Suspended"
    assert failed.status == "FatalFailure"
    assert failed.failure_code == "verified_integrity_conflict"


def test_result_repair_replay_accepts_jobs_from_only_the_last_attempt(
    tmp_path: Path,
) -> None:
    """多 attempt 重放只回调末次 turn 的 job，不得要求重放整个 invocation。"""

    class _MultiAttemptProvider:
        def __init__(self) -> None:
            self.model_turn_calls = 0
            self.replay_calls = 0

        @staticmethod
        def _receipt(index: int, *, status: str) -> dict[str, Any]:
            return {
                "identity": {
                    "job_id": f"job-{index}",
                    "operation_id": f"operation-{index}",
                    "submission_key": f"submission-{index}",
                },
                "request_digest": chr(ord("a") + index) * 64,
                "status": status,
            }

        def run_investigation_turn(
            self,
            _request: dict[str, Any],
            *,
            on_job_accepted: Callable[[dict[str, Any]], None],
            on_job_collected: Callable[[dict[str, Any]], None],
            **_: Any,
        ) -> NativeTurnResult:
            self.model_turn_calls += 1
            index = self.model_turn_calls
            receipt = self._receipt(index, status="pending")
            on_job_accepted(receipt)
            on_job_collected({**receipt, "status": "completed"})
            return NativeTurnResult(
                thread_id="thread-multi-attempt-repair",
                turn_id=f"turn-multi-attempt-repair-{index}",
                result=_scientific_result(
                    job_refs=[f"job-{item}" for item in range(1, index + 1)]
                ),
            )

        def replay_completed_turn(
            self,
            _request: dict[str, Any],
            *,
            replay_turn_id: str,
            on_job_accepted: Callable[[dict[str, Any]], None],
            on_job_collected: Callable[[dict[str, Any]], None],
            **_: Any,
        ) -> NativeTurnResult:
            self.replay_calls += 1
            assert replay_turn_id == "turn-multi-attempt-repair-2"
            receipt = self._receipt(2, status="pending")
            on_job_accepted(receipt)
            on_job_collected({**receipt, "status": "completed"})
            return NativeTurnResult(
                thread_id="thread-multi-attempt-repair",
                turn_id=replay_turn_id,
                request_digest="f" * 64,
                result=_scientific_result(job_refs=["job-1", "job-2"]),
            )

    class _RejectThenUnavailableVerifier:
        def __init__(self) -> None:
            self.calls = 0

        def verify_evidence(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            self.calls += 1
            if self.calls == 1:
                raise EvidenceResultRejected(
                    reason_code="evidence_references_rejected",
                    invalid_references=(
                        {
                            "kind": "job",
                            "job_id": "job-1",
                            "reason_code": "job_result_missing",
                        },
                    ),
                )
            if self.calls == 2:
                raise ValueError("模拟末次 Result 完成后 Evidence 暂时不可读")
            return {
                "verification_ref": "verification-multi-attempt-repair",
                "verification_digest": "c" * 64,
            }

    provider = _MultiAttemptProvider()
    verifier = _RejectThenUnavailableVerifier()
    store = HarnessRecoveryStore(tmp_path / "harness-recovery.sqlite3")
    backend = InvestigationSessionRunner(
        recovery_store=store,
        provider_transport=provider,
        execution_transport=FakeExecutionTransport(
            status="pending",
            query_error=AssertionError("completed turn 重放不得访问 MiniChem"),
        ),
        evidence_verifier=verifier,
        require_evidence_verifier=True,
    )
    harness = InvestigationHarness(
        tmp_path / "runtime",
        backend=backend,
        recovery_store=store,
    )

    paused = harness.start(_request())
    completed = harness.resume(paused.harness_invocation_id, _request())

    assert paused.status == "Suspended"
    assert paused.reason_code == "evidence_verification_missing"
    assert completed.status == "Completed"
    assert provider.model_turn_calls == 2
    assert provider.replay_calls == 1
    assert verifier.calls == 3


@pytest.mark.parametrize(
    "residual_suspension_reason",
    (None, "recovery_evidence_missing", "provider_contract_unavailable"),
)
def test_resume_replays_result_repair_turn_without_requiring_new_job_callbacks(
    tmp_path: Path,
    residual_suspension_reason: str | None,
) -> None:
    """返修 turn 没有新作业时，Evidence 重试仍应只读重放该 turn。"""

    class _RepairThenReplayProvider:
        def __init__(self) -> None:
            self.model_turn_calls = 0
            self.completed_turn_replay_calls: list[str] = []

        @staticmethod
        def _receipt(*, status: str) -> dict[str, Any]:
            return {
                "identity": {
                    "job_id": "job-1",
                    "operation_id": "operation-1",
                    "submission_key": "submission-1",
                },
                "request_digest": "b" * 64,
                "status": status,
            }

        def run_investigation_turn(
            self,
            _request: dict[str, Any],
            *,
            on_job_accepted: Callable[[dict[str, Any]], None],
            on_job_collected: Callable[[dict[str, Any]], None],
            **_: Any,
        ) -> NativeTurnResult:
            self.model_turn_calls += 1
            if self.model_turn_calls == 1:
                receipt = self._receipt(status="pending")
                on_job_accepted(receipt)
                on_job_collected({**receipt, "status": "completed"})
                return NativeTurnResult(
                    thread_id="thread-result-repair-replay",
                    turn_id="turn-result-repair-replay-1",
                    request_digest="d" * 64,
                    result=_scientific_result(job_refs=["job-1"]),
                )
            return NativeTurnResult(
                thread_id="thread-result-repair-replay",
                turn_id="turn-result-repair-replay-2",
                request_digest="e" * 64,
                result=_scientific_result(job_refs=["job-1"]),
            )

        def replay_completed_turn(
            self,
            _request: dict[str, Any],
            *,
            replay_turn_id: str,
            **_: Any,
        ) -> NativeTurnResult:
            self.completed_turn_replay_calls.append(replay_turn_id)
            assert replay_turn_id == "turn-result-repair-replay-2"
            return NativeTurnResult(
                thread_id="thread-result-repair-replay",
                turn_id="turn-result-repair-replay-2",
                request_digest="e" * 64,
                result=_scientific_result(job_refs=["job-1"]),
            )

    class _RejectThenPauseEvidenceVerifier:
        def __init__(self) -> None:
            self.calls = 0

        def verify_evidence(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            self.calls += 1
            if self.calls == 1:
                raise EvidenceResultRejected(
                    reason_code="evidence_references_rejected",
                    invalid_references=(
                        {
                            "kind": "operation",
                            "operation_id": "operation-running",
                            "reason_code": "operation_not_completed",
                        },
                    ),
                )
            if self.calls == 2:
                raise ValueError("模拟返修 Result 完成后的 Evidence 临时不可读")
            return {
                "verification_ref": "verification-result-repair-replay",
                "verification_digest": "c" * 64,
            }

    provider = _RepairThenReplayProvider()
    verifier = _RejectThenPauseEvidenceVerifier()
    execution = FakeExecutionTransport(
        status="pending",
        query_error=AssertionError("Result 返修和重放不得重新访问 MiniChem job"),
    )
    store = HarnessRecoveryStore(tmp_path / "harness-recovery.sqlite3")
    backend = InvestigationSessionRunner(
        recovery_store=store,
        provider_transport=provider,
        execution_transport=execution,
        evidence_verifier=verifier,
        require_evidence_verifier=True,
    )
    harness = InvestigationHarness(
        tmp_path / "runtime",
        backend=backend,
        recovery_store=store,
    )

    paused = harness.start(_request())

    assert paused.status == "Suspended"
    assert paused.reason_code == "evidence_verification_missing"
    if residual_suspension_reason is not None:
        store.append_transition(
            HarnessRunIdentity(
                research_task_id="task-1",
                module_invocation_id="module-1",
                harness_invocation_id=_request().harness_invocation_id,
                module_input_digest=_INPUT_DIGEST,
            ),
            SuspendInvocation(
                f"test-suspend-{residual_suspension_reason}",
                residual_suspension_reason,
            ),
        )
    completed = harness.resume(paused.harness_invocation_id, _request())

    assert completed.status == "Completed"
    assert provider.model_turn_calls == 2
    assert provider.completed_turn_replay_calls == ["turn-result-repair-replay-2"]
    assert verifier.calls == 3
    assert execution.submissions == []
    assert execution.queries == []
    assert execution.collections == []


def test_resume_reconciles_completed_turn_without_new_model_turn_or_minichem_call(
    tmp_path: Path,
) -> None:
    """旧 callback 顺序造成的部分状态必须只从 completed receipt 重放。"""

    provider = CompletedTurnReplayProvider()
    execution = FakeExecutionTransport(
        status="pending",
        query_error=AssertionError("completed turn 重放不得查询或重新提交 MiniChem"),
    )
    harness, store = _backend(tmp_path, provider, execution)  # type: ignore[arg-type]

    first = harness.start(_request())

    assert first.status == "Suspended"
    assert first.reason_code == "recovery_state_unavailable"
    assert provider.model_turn_calls == 1

    resumed = harness.resume(first.harness_invocation_id, _request())

    assert resumed.status == "Completed"
    assert provider.model_turn_calls == 1
    assert provider.completed_turn_replay_calls == 1
    assert execution.queries == []
    assert execution.collections == []
    assert execution.submissions == []
    snapshot = store.read_snapshot(
        HarnessRunIdentity(
            research_task_id="task-1",
            module_invocation_id="module-1",
            harness_invocation_id=_request().harness_invocation_id,
            module_input_digest=_INPUT_DIGEST,
        )
    )
    assert snapshot is not None
    assert [job.minichem_job_id for job in snapshot.jobs] == ["job-a", "job-b"]
    assert snapshot.collected_job_ids == ("job-a", "job-b")
    assert snapshot.active_job is None


def test_evidence_retry_replays_same_completed_turn_without_new_model_turn(
    tmp_path: Path,
) -> None:
    """作业已 reconciliation 后，Evidence 暂停仍须复用同一 provider receipt。"""

    class _FlakyEvidenceVerifier:
        def __init__(self) -> None:
            self.calls = 0

        def verify_evidence(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            self.calls += 1
            if self.calls == 1:
                raise ValueError("模拟 Evidence owner 首次核验失败")
            return {
                "verification_ref": "verification-replay",
                "verification_digest": "c" * 64,
            }

    provider = CompletedTurnReplayProvider()
    execution = FakeExecutionTransport(
        status="pending",
        query_error=AssertionError("Evidence 重试不得查询或重新提交 MiniChem"),
    )
    verifier = _FlakyEvidenceVerifier()
    store = HarnessRecoveryStore(tmp_path / "harness-recovery.sqlite3")
    backend = InvestigationSessionRunner(
        recovery_store=store,
        provider_transport=provider,
        execution_transport=execution,
        evidence_verifier=verifier,
        require_evidence_verifier=True,
    )
    harness = InvestigationHarness(
        tmp_path / "runtime",
        backend=backend,
        recovery_store=store,
    )

    first = harness.start(_request())
    assert first.status == "Suspended"

    evidence_paused = harness.resume(first.harness_invocation_id, _request())
    assert evidence_paused.status == "Suspended"
    assert evidence_paused.reason_code == "evidence_verification_missing"

    completed = harness.resume(first.harness_invocation_id, _request())

    assert completed.status == "Completed"
    assert provider.model_turn_calls == 1
    assert provider.completed_turn_replay_calls == 2
    assert execution.queries == []
    assert execution.collections == []
    assert execution.submissions == []


def test_replay_capability_does_not_replace_normal_waiting_job_recovery(
    tmp_path: Path,
) -> None:
    """completed replay 只处理部分落盘，不接管普通 waiting job。"""

    provider = ReplayCapableSequentialProvider()
    execution = FakeExecutionTransport(status="pending")
    harness, _store = _backend(tmp_path, provider, execution)  # type: ignore[arg-type]

    first = harness.start(_request())
    assert first.status == "Suspended"
    assert first.reason_code == "job_not_terminal"

    execution.status = "completed"
    completed = harness.resume(first.harness_invocation_id, _request())

    assert completed.status == "Completed"
    assert provider.model_turn_calls == 2
    assert provider.completed_turn_replay_calls == 0
    assert execution.queries == ["job-1"]
    assert execution.collections == ["job-1"]


def test_native_start_suspends_pending_job_and_resume_reuses_job_and_prepared_pair(
    tmp_path: Path,
) -> None:
    provider = FakeProviderTransport(
        [
            NativeTurnResult(
                thread_id="thread-1",
                turn_id="turn-1",
                actions=(NativeAction("action-1", "submit_job", {"kind": "gaussian"}),),
            ),
            NativeTurnResult(
                thread_id="thread-1",
                turn_id="turn-2",
                result=_scientific_result(job_refs=["job-1"]),
            ),
        ]
    )
    execution = FakeExecutionTransport(status="pending")
    harness, store = _backend(tmp_path, provider, execution)

    first = harness.start(_request())

    assert isinstance(first, Suspended)
    assert first.status == "Suspended"
    assert execution.submissions and len(execution.submissions) == 1
    first_investigation_session_id = provider.calls[0]["investigation_session_id"]

    execution.status = "completed"
    resumed = harness.resume(first.harness_invocation_id, _request())

    assert resumed.status == "Completed"
    assert len(execution.submissions) == 1
    assert execution.queries == ["job-1"]
    assert execution.collections == ["job-1"]
    assert provider.calls[1]["investigation_session_id"] == first_investigation_session_id
    assert len(resumed.prepared_output.output_scientific_records) == 2
    assert {
        record.scientific_record_type for record in resumed.prepared_output.output_scientific_records
    } == {"VerifiedEvidenceRecord", "InvestigationResultRecord"}

    snapshot = store.read_snapshot(
        HarnessRunIdentity(
            research_task_id="task-1",
            module_invocation_id="module-1",
            harness_invocation_id=_request().harness_invocation_id,
            module_input_digest=_INPUT_DIGEST,
        )
    )
    assert snapshot.state is HarnessLifecycle.COMPLETED
    assert snapshot.session is not None
    assert snapshot.active_job is None
    assert snapshot.active_attempt is None
    assert snapshot.collected_job_ids == ("job-1",)
    assert len(snapshot.verified_evidence) == 1
    assert snapshot.committed_output is not None


def test_catalog_readiness_is_not_required_for_old_job_resume_but_submit_checks_action_contract(
    tmp_path: Path,
) -> None:
    provider = FakeProviderTransport(
        [
            NativeTurnResult(
                thread_id="thread-1",
                turn_id="turn-1",
                actions=(NativeAction("action-1", "submit_job", {"kind": "gaussian"}),),
            ),
            NativeTurnResult(
                thread_id="thread-1",
                turn_id="turn-2",
                result=_scientific_result(job_refs=["job-1"]),
            ),
        ]
    )
    execution = ContractGuardExecution(status="pending")
    harness, _store = _backend(tmp_path, provider, execution)

    first = harness.start(_request())
    assert first.status == "Suspended"
    assert execution.contract_checks == ["action-1"]
    assert execution.readiness_checks == 0

    execution.status = "completed"
    resumed = harness.resume(first.harness_invocation_id, _request())

    assert resumed.status == "Completed"
    assert execution.contract_checks == ["action-1"]
    assert execution.readiness_checks == 0


def test_completed_runtime_receipt_without_status_is_collected_and_finishes(
    tmp_path: Path,
) -> None:
    accepted = {
        "identity": {
            "job_id": "job-1",
            "operation_id": "operation-1",
            "submission_key": "submission-1",
        },
        "request_digest": "b" * 64,
    }
    provider = FakeProviderTransport(
        [
            NativeTurnResult(
                thread_id="thread-1",
                turn_id="turn-1",
                accepted_job=accepted,
                completed_jobs=(accepted,),
                result=_scientific_result(job_refs=["job-1"]),
            )
        ]
    )
    execution = FakeExecutionTransport(status="pending")
    harness, store = _backend(tmp_path, provider, execution)

    outcome = harness.start(_request())

    assert outcome.status == "Completed"
    assert execution.submissions == []
    assert execution.queries == []
    assert execution.collections == []
    snapshot = store.read_snapshot(
        HarnessRunIdentity(
            research_task_id="task-1",
            module_invocation_id="module-1",
            harness_invocation_id=_request().harness_invocation_id,
            module_input_digest=_INPUT_DIGEST,
        )
    )
    assert snapshot.collected_job_ids == ("job-1",)
    assert snapshot.active_job is None


def test_submission_contract_incompatibility_is_recoverable_and_does_not_submit(
    tmp_path: Path,
) -> None:
    provider = FakeProviderTransport(
        [
            NativeTurnResult(
                thread_id="thread-1",
                turn_id="turn-1",
                actions=(NativeAction("action-1", "submit_job", {"kind": "gaussian"}),),
            )
        ]
    )
    execution = FakeExecutionTransport(status="pending")
    harness, store = _backend(tmp_path, provider, execution)
    backend = InvestigationSessionRunner(
        recovery_store=store,
        provider_transport=provider,
        execution_transport=execution,
        submission_contract_checker=lambda *_args, **_kwargs: False,
    )
    harness = InvestigationHarness(tmp_path / "runtime-contract", backend=backend, recovery_store=store)

    outcome = harness.start(_request())

    assert outcome.status == "Suspended"
    assert outcome.reason_code == "minichem_contract_incompatible"
    assert execution.submissions == []
    assert store.read_snapshot(
        HarnessRunIdentity(
            research_task_id="task-1",
            module_invocation_id="module-1",
            harness_invocation_id=_request().harness_invocation_id,
            module_input_digest=_INPUT_DIGEST,
        )
    ).state is HarnessLifecycle.SUSPENDED


def test_production_native_backend_fails_closed_without_evidence_verifier(
    tmp_path: Path,
) -> None:
    """生产装配开启严格证据模式时不能把合成摘要标成 verified。"""

    provider = FakeProviderTransport(
        [
            NativeTurnResult(
                thread_id="thread-strict",
                turn_id="turn-strict-1",
                actions=(NativeAction("action-strict", "submit_job", {"kind": "gaussian"}),),
            ),
            NativeTurnResult(
                thread_id="thread-strict",
                turn_id="turn-strict-2",
                result=_scientific_result(job_refs=["job-1"]),
            )
        ]
    )
    execution = FakeExecutionTransport(status="completed")
    store = HarnessRecoveryStore(tmp_path / "strict-recovery.sqlite3")
    backend = InvestigationSessionRunner(
        recovery_store=store,
        provider_transport=provider,
        execution_transport=execution,
        require_evidence_verifier=True,
    )
    harness = InvestigationHarness(tmp_path / "strict-runtime", backend=backend, recovery_store=store)

    outcome = harness.start(_request())

    assert isinstance(outcome, Suspended)
    assert outcome.reason_code == "evidence_verification_missing"


def test_authoritative_scope_conflict_on_old_job_resume_is_fatal(
    tmp_path: Path,
) -> None:
    provider = FakeProviderTransport(
        [
            NativeTurnResult(
                thread_id="thread-1",
                turn_id="turn-1",
                actions=(NativeAction("action-1", "submit_job", {"kind": "gaussian"}),),
            ),
            NativeTurnResult(
                thread_id="thread-1",
                turn_id="turn-2",
                result=_scientific_result(job_refs=["job-1"]),
            ),
        ]
    )
    execution = ScopeConflictExecution(status="pending")
    harness, store = _backend(tmp_path, provider, execution)
    first = harness.start(_request())
    assert first.status == "Suspended"

    execution.status = "completed"
    outcome = harness.resume(first.harness_invocation_id, _request())

    assert outcome.status == "FatalFailure"
    assert outcome.failure_code == "verified_integrity_conflict"
    assert execution.submissions and len(execution.submissions) == 1
    assert execution.queries == ["job-1"]
    assert store.read_snapshot(
        HarnessRunIdentity(
            research_task_id="task-1",
            module_invocation_id="module-1",
            harness_invocation_id=_request().harness_invocation_id,
            module_input_digest=_INPUT_DIGEST,
        )
    ).state is HarnessLifecycle.FAILED

def test_terminal_job_failure_is_action_result_and_same_model_loop_can_finish(
    tmp_path: Path,
) -> None:
    def finish_after_failure(*, action_results: tuple[dict[str, Any], ...], **_: Any) -> NativeTurnResult:
        assert action_results
        assert action_results[-1]["status"] == "failed"
        assert action_results[-1]["error_code"] == "gaussian_failed"
        return NativeTurnResult(
            thread_id="thread-1",
            turn_id="turn-2",
            result=_scientific_result(),
        )

    provider = FakeProviderTransport(
        [
            NativeTurnResult(
                thread_id="thread-1",
                turn_id="turn-1",
                actions=(NativeAction("action-1", "submit_job", {"kind": "gaussian"}),),
            ),
            finish_after_failure,
        ]
    )
    execution = FakeExecutionTransport(status="pending", collect_status="failed")
    harness, _store = _backend(tmp_path, provider, execution)

    first = harness.start(_request())
    assert first.status == "Suspended"

    execution.status = "failed"
    resumed = harness.resume(first.harness_invocation_id, _request())

    assert resumed.status == "Completed"
    assert len(execution.submissions) == 1
    assert len(provider.calls) == 2
    assert provider.calls[1]["action_results"][-1]["status"] == "failed"


def test_completed_turn_without_progress_gets_neutral_same_thread_continuation(
    tmp_path: Path,
) -> None:
    provider = FakeProviderTransport(
        [
            NativeTurnResult(thread_id="thread-1", turn_id="turn-1"),
            NativeTurnResult(
                thread_id="thread-1",
                turn_id="turn-2",
                actions=(NativeAction("action-1", "submit_job", {"kind": "gaussian"}),),
            ),
            NativeTurnResult(
                thread_id="thread-1",
                turn_id="turn-3",
                result=_scientific_result(job_refs=["job-1"]),
            ),
        ]
    )
    execution = FakeExecutionTransport(status="completed")
    harness, _store = _backend(tmp_path, provider, execution)

    outcome = harness.start(_request())

    assert outcome.status == "Completed"
    assert len(provider.calls) == 3
    assert provider.calls[1]["is_resume"] is True
    assert provider.calls[1]["investigation_session_id"] == provider.calls[0]["investigation_session_id"]
    assert provider.calls[1]["thread_id"] == "thread-1"
    assert provider.calls[1]["action_results"] == (
        {
            "kind": "neutral_continuation",
            "status": "continue",
            "reason_code": "model_no_progress",
        },
    )
    assert len(execution.submissions) == 1


def test_repeated_completed_turn_without_progress_is_recoverably_suspended(
    tmp_path: Path,
) -> None:
    provider = FakeProviderTransport(
        [
            NativeTurnResult(thread_id="thread-1", turn_id="turn-1"),
            NativeTurnResult(thread_id="thread-1", turn_id="turn-2"),
        ]
    )
    harness, _store = _backend(tmp_path, provider, FakeExecutionTransport())

    outcome = harness.start(_request())

    assert outcome.status == "Suspended"
    assert outcome.reason_code == "model_no_progress"
    assert len(provider.calls) == 2


def test_explicit_budget_terminal_does_not_trigger_neutral_continuation(
    tmp_path: Path,
) -> None:
    provider = FakeProviderTransport(
        [NativeTurnResult(thread_id="thread-1", turn_id="turn-1", status="budget_limited")]
    )
    harness, _store = _backend(tmp_path, provider, FakeExecutionTransport())

    outcome = harness.start(_request())

    assert outcome.status == "Suspended"
    assert outcome.reason_code == "budget_limited"
    assert len(provider.calls) == 1


def test_provider_interruption_and_mcp_unavailability_remain_suspended(
    tmp_path: Path,
) -> None:
    interrupted_provider = FakeProviderTransport(
        [
            NativeTurnResult(
                thread_id="thread-1",
                turn_id="turn-1",
                status="interrupted",
            )
        ]
    )
    interrupted_execution = FakeExecutionTransport()
    harness, store = _backend(tmp_path / "interrupted", interrupted_provider, interrupted_execution)

    interrupted = harness.start(_request())

    assert interrupted.status == "Suspended"
    assert store.read_snapshot(
        HarnessRunIdentity(
            research_task_id="task-1",
            module_invocation_id="module-1",
            harness_invocation_id=_request().harness_invocation_id,
            module_input_digest=_INPUT_DIGEST,
        )
    ).state is HarnessLifecycle.SUSPENDED

    provider = FakeProviderTransport(
        [
            NativeTurnResult(
                thread_id="thread-1",
                turn_id="turn-1",
                actions=(NativeAction("action-1", "submit_job", {"kind": "gaussian"}),),
            )
        ]
    )
    unavailable_execution = FakeExecutionTransport(
        status="pending",
        query_error=ConnectionError("fake MCP unavailable"),
    )
    unavailable_harness, _ = _backend(tmp_path / "unavailable", provider, unavailable_execution)
    first = unavailable_harness.start(_request())
    assert first.status == "Suspended"

    unavailable_execution.status = "completed"
    resumed = unavailable_harness.resume(first.harness_invocation_id, _request())
    assert resumed.status == "Suspended"
    assert resumed.reason_code == "minichem_unavailable"

    idle_provider = FakeProviderTransport(
        [NativeTurnResult(thread_id="thread-1", turn_id="turn-1", status="idle")]
    )
    idle_harness, _ = _backend(
        tmp_path / "idle",
        idle_provider,
        FakeExecutionTransport(),
    )
    idle = idle_harness.start(_request())
    assert idle.status == "Suspended"
    assert idle.reason_code == "idle"


def test_authoritative_thread_conflict_is_the_only_fatal_backend_branch(
    tmp_path: Path,
) -> None:
    provider = FakeProviderTransport(
        [
            NativeTurnResult(
                thread_id="thread-1",
                turn_id="turn-1",
                actions=(NativeAction("action-1", "submit_job", {"kind": "gaussian"}),),
            ),
            NativeTurnResult(
                thread_id="thread-2",
                turn_id="turn-2",
                result=_scientific_result(),
            ),
        ]
    )
    execution = FakeExecutionTransport(status="pending")
    harness, store = _backend(tmp_path, provider, execution)

    first = harness.start(_request())
    execution.status = "completed"
    failed = harness.resume(first.harness_invocation_id, _request())

    assert failed.status == "FatalFailure"
    assert failed.failure_code == "verified_integrity_conflict"
    assert store.read_snapshot(
        HarnessRunIdentity(
            research_task_id="task-1",
            module_invocation_id="module-1",
            harness_invocation_id=_request().harness_invocation_id,
            module_input_digest=_INPUT_DIGEST,
        )
    ).state is HarnessLifecycle.FAILED
