"""V2 Harness 恢复事件在修订/继续执行间的身份和幂等回归。"""

from __future__ import annotations

from pathlib import Path

import pytest

from chemistry_multiagent.investigation.harness_recovery_store import (
    BindSession,
    CloseProviderAttempt,
    CollectJob,
    CommitModuleOutput,
    CompleteInvocation,
    EvidenceVerificationReference,
    HarnessRecoveryStore,
    HarnessRecoveryTransitionConflict,
    HarnessLifecycle,
    HarnessRunIdentity,
    JobReference,
    ModuleOutputReference,
    ProviderAttemptOutcome,
    ProviderAttemptReference,
    ReceiptKind,
    ReceiptReference,
    SessionReference,
    StartProviderAttempt,
    SubmitJob,
    VerifyEvidence,
)

_DIGEST_A = "a" * 64
_DIGEST_B = "b" * 64
_DIGEST_C = "c" * 64
_DIGEST_D = "d" * 64
_DIGEST_E = "e" * 64
_DIGEST_F = "f" * 64


def _identity() -> HarnessRunIdentity:
    return HarnessRunIdentity(
        research_task_id="research-task-1",
        module_invocation_id="module-invocation-1",
        harness_invocation_id="harness-invocation-1",
        module_input_digest=_DIGEST_A,
    )


def _session() -> SessionReference:
    return SessionReference(
        research_task_id="research-task-1",
        investigation_context_id="context-1",
        hypothesis_execution_input_record_id="hypothesis-execution-input-1",
        investigation_session_id="session-1",
        workspace_identity="workspace-1",
        minichem_scope_id="scope-1",
        scope_contract_digest=_DIGEST_B,
        capability_generation_digest=_DIGEST_C,
    )


def _attempt(model_turn_id: str, model_attempt_id: str) -> ProviderAttemptReference:
    return ProviderAttemptReference(
        investigation_session_id="session-1",
        model_thread_id="thread-1",
        model_turn_id=model_turn_id,
        model_attempt_id=model_attempt_id,
        model_request_digest=(
            _DIGEST_C if model_turn_id == "turn-1" else _DIGEST_D
        ),
    )


def _job(
    minichem_job_id: str,
    minichem_operation_id: str,
    minichem_submission_key: str,
) -> JobReference:
    return JobReference(
        investigation_session_id="session-1",
        minichem_job_id=minichem_job_id,
        minichem_operation_id=minichem_operation_id,
        minichem_submission_key=minichem_submission_key,
        minichem_request_digest=(
            _DIGEST_E if minichem_job_id == "job-1" else _DIGEST_F
        ),
    )


def test_revised_provider_attempt_can_continue_without_replaying_old_job(tmp_path: Path) -> None:
    """第一次 attempt 中断后，继续 attempt 可收集各自 job 并提交一次模块输出。"""

    identity = _identity()
    store = HarnessRecoveryStore(tmp_path / "harness-recovery.sqlite3")
    store.create_or_load(identity)
    store.append_transition(identity, BindSession(event_id="bind-1", session=_session()))

    first_attempt = _attempt("turn-1", "attempt-1")
    first_job = _job("job-1", "operation-1", "submission-1")
    store.append_transition(
        identity, StartProviderAttempt(event_id="attempt-start-1", attempt=first_attempt)
    )
    store.append_transition(
        identity, SubmitJob(event_id="job-submit-1", attempt=first_attempt, job=first_job)
    )
    store.append_transition(identity, CollectJob(event_id="job-collect-1", job=first_job))
    store.append_transition(
        identity,
        CloseProviderAttempt(
            event_id="attempt-close-1",
            attempt=first_attempt,
            outcome=ProviderAttemptOutcome.INTERRUPTED,
        ),
    )

    second_attempt = _attempt("turn-2", "attempt-2")
    second_job = _job("job-2", "operation-2", "submission-2")
    store.append_transition(
        identity, StartProviderAttempt(event_id="attempt-start-2", attempt=second_attempt)
    )
    store.append_transition(
        identity, SubmitJob(event_id="job-submit-2", attempt=second_attempt, job=second_job)
    )
    store.append_transition(identity, CollectJob(event_id="job-collect-2", job=second_job))
    store.append_transition(
        identity,
        CloseProviderAttempt(
            event_id="attempt-close-2",
            attempt=second_attempt,
            outcome=ProviderAttemptOutcome.COMPLETED,
            receipt=ReceiptReference(
                kind=ReceiptKind.PROVIDER_TURN,
                receipt_ref="receipt-turn-2",
                receipt_digest=_DIGEST_A,
            ),
        ),
    )
    store.append_transition(
        identity,
        VerifyEvidence(
            event_id="evidence-2",
            verification=EvidenceVerificationReference(
                verification_ref="verification-2",
                verification_digest=_DIGEST_B,
                job_id="job-2",
                operation_id="operation-2",
                submission_key="submission-2",
            ),
        ),
    )
    store.append_transition(
        identity,
        CommitModuleOutput(
            event_id="output-1",
            output=ModuleOutputReference(
                output_ref="module-output-2",
                output_digest=_DIGEST_C,
            ),
        ),
    )
    snapshot = store.append_transition(
        identity, CompleteInvocation(event_id="complete-1")
    )

    assert snapshot.state is HarnessLifecycle.COMPLETED
    assert [attempt.model_attempt_id for attempt in snapshot.attempts] == [
        "attempt-1",
        "attempt-2",
    ]
    assert [job.minichem_job_id for job in snapshot.jobs] == ["job-1", "job-2"]
    assert snapshot.collected_job_ids == ("job-1", "job-2")
    assert snapshot.committed_output is not None


def test_same_transition_is_idempotent_but_output_identity_drift_fails_closed(tmp_path: Path) -> None:
    """恢复重试可复用同一事件，内容漂移不能创建第二份输出。"""

    identity = _identity()
    store = HarnessRecoveryStore(tmp_path / "harness-recovery.sqlite3")
    store.create_or_load(identity)
    first = CommitModuleOutput(
        event_id="output-1",
        output=ModuleOutputReference(output_ref="module-output-1", output_digest=_DIGEST_A),
    )
    store.append_transition(identity, first)
    replay = store.append_transition(identity, first)
    assert replay.committed_output == first.output
    assert replay.event_sequence == 2

    drift = CommitModuleOutput(
        event_id="output-1",
        output=ModuleOutputReference(output_ref="module-output-2", output_digest=_DIGEST_B),
    )
    with pytest.raises(HarnessRecoveryTransitionConflict):
        store.append_transition(identity, drift)
