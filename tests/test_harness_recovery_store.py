"""HarnessRecoveryStore 的公开恢复 seam 测试。

这些测试只通过公开的 typed interface 观察运行身份、恢复事件和快照；不读取
SQLite 表，也不把科学结果、模型 transcript 或凭据放进 Harness 私有存储。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chemistry_multiagent.investigation.harness_recovery_store import (
    BindSession,
    CloseProviderAttempt,
    CollectJob,
    CompleteInvocation,
    FailInvocation,
    HarnessLifecycle,
    HarnessRecoveryIdentityConflict,
    HarnessRecoveryNotFound,
    HarnessRecoveryStateConflict,
    HarnessRecoveryStore,
    HarnessRecoveryTransitionConflict,
    HarnessRunIdentity,
    JobReference,
    ModuleOutputReference,
    ProviderAttemptOutcome,
    ProviderAttemptReference,
    ReceiptKind,
    ReceiptReference,
    CommitModuleOutput,
    SessionReference,
    StartProviderAttempt,
    SubmitJob,
    SuspendInvocation,
    VerifyEvidence,
    EvidenceVerificationReference,
)


_INPUT_DIGEST = "a" * 64
_SCOPE_CONTRACT_DIGEST = "b" * 64
_CAPABILITY_GENERATION_DIGEST = "9" * 64
_TURN_DIGEST = "c" * 64
_SUBMISSION_DIGEST = "d" * 64
_RECEIPT_DIGEST = "e" * 64
_VERIFICATION_DIGEST = "f" * 64
_OUTPUT_DIGEST = "1" * 64


def _identity(
    *,
    research_task_id: str = "research-task-1",
    module_invocation_id: str = "module-invocation-1",
    harness_invocation_id: str = "harness-invocation-1",
    module_input_digest: str = _INPUT_DIGEST,
) -> HarnessRunIdentity:
    return HarnessRunIdentity(
        research_task_id=research_task_id,
        module_invocation_id=module_invocation_id,
        harness_invocation_id=harness_invocation_id,
        module_input_digest=module_input_digest,
    )


def _session(
    *,
    research_task_id: str = "research-task-1",
    investigation_context_id: str = "context-1",
    hypothesis_execution_input_record_id: str = "hypothesis-execution-input-1",
    investigation_session_id: str = "session-1",
    workspace_identity: str = "workspace-1",
    minichem_scope_id: str = "scope-1",
    scope_contract_digest: str = _SCOPE_CONTRACT_DIGEST,
    capability_generation_digest: str = _CAPABILITY_GENERATION_DIGEST,
) -> SessionReference:
    return SessionReference(
        research_task_id=research_task_id,
        investigation_context_id=investigation_context_id,
        hypothesis_execution_input_record_id=hypothesis_execution_input_record_id,
        investigation_session_id=investigation_session_id,
        workspace_identity=workspace_identity,
        minichem_scope_id=minichem_scope_id,
        scope_contract_digest=scope_contract_digest,
        capability_generation_digest=capability_generation_digest,
    )


def _attempt(
    *,
    model_attempt_id: str = "attempt-1",
    model_thread_id: str = "thread-1",
    model_turn_id: str = "turn-1",
    model_request_digest: str = _TURN_DIGEST,
    investigation_session_id: str = "session-1",
) -> ProviderAttemptReference:
    return ProviderAttemptReference(
        investigation_session_id=investigation_session_id,
        model_thread_id=model_thread_id,
        model_turn_id=model_turn_id,
        model_attempt_id=model_attempt_id,
        model_request_digest=model_request_digest,
    )


def _job(
    *,
    minichem_job_id: str = "job-1",
    minichem_operation_id: str = "operation-1",
    minichem_submission_key: str = "submission-1",
    minichem_request_digest: str = _SUBMISSION_DIGEST,
    investigation_session_id: str = "session-1",
) -> JobReference:
    return JobReference(
        investigation_session_id=investigation_session_id,
        minichem_job_id=minichem_job_id,
        minichem_operation_id=minichem_operation_id,
        minichem_submission_key=minichem_submission_key,
        minichem_request_digest=minichem_request_digest,
    )


def _receipt(
    receipt_ref: str,
    kind: ReceiptKind,
) -> ReceiptReference:
    return ReceiptReference(
        kind=kind,
        receipt_ref=receipt_ref,
        receipt_digest=_RECEIPT_DIGEST,
    )


def _provider_receipt(receipt_ref: str = "provider-receipt-1") -> ReceiptReference:
    return _receipt(receipt_ref, ReceiptKind.PROVIDER_TURN)


def _verification(job: JobReference | None = None) -> EvidenceVerificationReference:
    selected_job = job or _job()
    return EvidenceVerificationReference(
        verification_ref="verification-1",
        verification_digest=_VERIFICATION_DIGEST,
        job_id=selected_job.minichem_job_id,
        operation_id=selected_job.minichem_operation_id,
        submission_key=selected_job.minichem_submission_key,
    )


def _prepare_store(store: HarnessRecoveryStore) -> HarnessRunIdentity:
    identity = _identity()
    store.create_or_load(identity)
    store.append_transition(identity, BindSession("event-session", _session()))
    store.append_transition(
        identity,
        StartProviderAttempt("event-attempt", _attempt()),
    )
    store.append_transition(
        identity,
        SubmitJob("event-submit", _attempt(), _job()),
    )
    return identity


def test_create_or_load_is_idempotent_and_read_snapshot_is_required_read(
    tmp_path: Path,
) -> None:
    store = HarnessRecoveryStore(tmp_path / "harness-recovery.sqlite3")
    identity = _identity()

    created = store.create_or_load(identity)
    assert created.identity == identity
    assert created.state is HarnessLifecycle.OPEN
    assert created.event_sequence == 1

    loaded = store.create_or_load(identity)
    assert loaded == created
    assert store.load(identity) == created
    assert store.read_snapshot(identity) == created

    with pytest.raises(HarnessRecoveryNotFound):
        store.read_snapshot(_identity(harness_invocation_id="missing-harness-invocation"))

    with pytest.raises(HarnessRecoveryIdentityConflict):
        store.create_or_load(_identity(research_task_id="another-research-task"))


def test_typed_references_survive_process_restart_and_output_can_be_replayed(
    tmp_path: Path,
) -> None:
    database = tmp_path / "harness-recovery.sqlite3"
    identity = _identity()
    store = HarnessRecoveryStore(database)
    _prepare_store(store)
    job = _job()
    attempt = _attempt()

    store.append_transition(
        identity,
        CollectJob(
            "event-collect",
            job,
            _receipt("job-receipt-1", ReceiptKind.MINICHEM_OPERATION),
        ),
    )
    store.append_transition(
        identity,
        CloseProviderAttempt(
            "event-close",
            attempt,
            ProviderAttemptOutcome.COMPLETED,
            _provider_receipt(),
        ),
    )
    store.append_transition(identity, VerifyEvidence("event-verify", _verification(job)))
    output = ModuleOutputReference(
        output_ref="prepared-output-1",
        output_digest=_OUTPUT_DIGEST,
    )
    store.append_transition(identity, CommitModuleOutput("event-commit-output", output))
    completed = store.append_transition(identity, CompleteInvocation("event-complete"))

    restarted = HarnessRecoveryStore(database)
    replayable = restarted.read_snapshot(identity)
    assert replayable == completed
    assert replayable.state is HarnessLifecycle.COMPLETED
    assert replayable.committed_output == output
    assert replayable.active_job is None
    assert replayable.active_attempt is None
    assert replayable.verified_evidence == (_verification(job),)
    assert replayable.last_receipt == _provider_receipt()
    assert replayable.session is not None
    assert replayable.session.scope_contract_digest == _SCOPE_CONTRACT_DIGEST
    assert replayable.session.capability_generation_digest == _CAPABILITY_GENERATION_DIGEST
    assert (
        replayable.session.scope_contract_digest
        != replayable.session.capability_generation_digest
    )
    assert all(
        not hasattr(replayable, field)
        for field in ("transcript", "scientific_result", "credential", "password")
    )

    # 已提交的终态事件可以精确重放，不产生新的事件序号。
    replayed = restarted.append_transition(identity, CompleteInvocation("event-complete"))
    assert replayed == completed
    assert replayed.event_sequence == completed.event_sequence


def test_transition_idempotency_uses_event_id_and_content_digest(
    tmp_path: Path,
) -> None:
    store = HarnessRecoveryStore(tmp_path / "harness-recovery.sqlite3")
    identity = _prepare_store(store)
    transition = SuspendInvocation("event-suspend", "provider_unavailable")

    first = store.append_transition(identity, transition)
    repeated = store.append_transition(identity, transition)
    assert repeated == first
    assert repeated.event_sequence == first.event_sequence

    with pytest.raises(HarnessRecoveryTransitionConflict):
        store.append_transition(
            identity,
            SuspendInvocation("event-suspend", "receipt_missing"),
        )


def test_process_disappearance_keeps_open_state_and_does_not_infer_failed(
    tmp_path: Path,
) -> None:
    database = tmp_path / "harness-recovery.sqlite3"
    identity = _identity()
    first_process = HarnessRecoveryStore(database)
    _prepare_store(first_process)

    restarted = HarnessRecoveryStore(database)
    snapshot = restarted.read_snapshot(identity)
    assert snapshot.state is HarnessLifecycle.OPEN
    assert snapshot.active_attempt == _attempt()
    assert snapshot.active_job == _job()
    assert snapshot.state is not HarnessLifecycle.FAILED

    suspended = restarted.append_transition(
        identity,
        SuspendInvocation("event-suspend", "provider_receipt_pending"),
    )
    assert suspended.state is HarnessLifecycle.SUSPENDED


def test_failed_is_an_explicit_terminal_transition_and_can_be_replayed(
    tmp_path: Path,
) -> None:
    database = tmp_path / "harness-recovery.sqlite3"
    identity = _identity()
    store = HarnessRecoveryStore(database)
    store.create_or_load(identity)

    failed = store.append_transition(identity, FailInvocation("event-failed", "provider_contract"))
    assert failed.state is HarnessLifecycle.FAILED
    assert HarnessRecoveryStore(database).read_snapshot(identity) == failed

    # 失败之后只能精确重放同一事件，不能把已结束 invocation 改成别的状态。
    assert store.append_transition(identity, FailInvocation("event-failed", "provider_contract")) == failed
    with pytest.raises(HarnessRecoveryStateConflict):
        store.append_transition(identity, SuspendInvocation("event-after-failed", "retryable"))


def test_only_one_attempt_and_one_job_can_be_active(
    tmp_path: Path,
) -> None:
    store = HarnessRecoveryStore(tmp_path / "harness-recovery.sqlite3")
    identity = _prepare_store(store)

    with pytest.raises(HarnessRecoveryStateConflict):
        store.append_transition(
            identity,
            StartProviderAttempt(
                "event-attempt-2",
                _attempt(model_attempt_id="attempt-2", model_turn_id="turn-2"),
            ),
        )
    with pytest.raises(HarnessRecoveryStateConflict):
        store.append_transition(
            identity,
            SubmitJob(
                "event-submit-2",
                _attempt(),
                _job(minichem_job_id="job-2", minichem_submission_key="submission-2"),
            ),
        )

    store.append_transition(identity, CollectJob("event-collect", _job()))
    store.append_transition(
        identity,
        CloseProviderAttempt(
            "event-close",
            _attempt(),
            ProviderAttemptOutcome.INTERRUPTED,
        ),
    )
    resumed = store.append_transition(
        identity,
        StartProviderAttempt(
            "event-attempt-2",
            _attempt(model_attempt_id="attempt-2", model_turn_id="turn-2"),
        ),
    )
    assert resumed.active_attempt == _attempt(model_attempt_id="attempt-2", model_turn_id="turn-2")


def test_task_thread_turn_scope_job_submission_and_digest_drift_fail_closed(
    tmp_path: Path,
) -> None:
    store = HarnessRecoveryStore(tmp_path / "harness-recovery.sqlite3")
    identity = _prepare_store(store)

    with pytest.raises(HarnessRecoveryIdentityConflict):
        store.append_transition(
            _identity(research_task_id="research-task-2"),
            SuspendInvocation("event-task-drift", "provider_unavailable"),
        )
    with pytest.raises(HarnessRecoveryIdentityConflict):
        store.append_transition(
            _identity(module_input_digest="9" * 64),
            SuspendInvocation("event-digest-drift", "provider_unavailable"),
        )
    with pytest.raises(HarnessRecoveryIdentityConflict):
        store.append_transition(
            identity,
            BindSession("event-scope-drift", _session(minichem_scope_id="scope-2")),
        )

    store.append_transition(identity, CollectJob("event-collect", _job()))
    store.append_transition(
        identity,
        CloseProviderAttempt("event-close", _attempt(), ProviderAttemptOutcome.INTERRUPTED),
    )

    with pytest.raises(HarnessRecoveryIdentityConflict):
        store.append_transition(
            identity,
            StartProviderAttempt(
                "event-thread-drift",
                _attempt(
                    model_attempt_id="attempt-2",
                    model_turn_id="turn-2",
                    model_thread_id="thread-2",
                ),
            ),
        )
    with pytest.raises(HarnessRecoveryIdentityConflict):
        store.append_transition(
            identity,
            StartProviderAttempt(
                "event-turn-reuse",
                _attempt(model_attempt_id="attempt-2", model_turn_id="turn-1"),
            ),
        )

    # 新 attempt 可以使用同一 thread 的新 turn，但新 job 的 submission 和摘要必须成组变化。
    store.append_transition(
        identity,
        StartProviderAttempt(
            "event-attempt-2",
            _attempt(model_attempt_id="attempt-2", model_turn_id="turn-2"),
        ),
    )
    with pytest.raises(HarnessRecoveryIdentityConflict):
        store.append_transition(
            identity,
            SubmitJob(
                "event-job-digest-drift",
                _attempt(model_attempt_id="attempt-2", model_turn_id="turn-2"),
                _job(minichem_request_digest="8" * 64),
            ),
        )


def test_invalid_reference_shapes_and_unknown_states_are_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError):
        HarnessRunIdentity(
            research_task_id="research-task-1",
            module_invocation_id="module-invocation-1",
            harness_invocation_id="harness-invocation-1",
            module_input_digest="not-a-sha256",
        )
    with pytest.raises(ValueError):
        SessionReference(
            research_task_id="research-task-1",
            investigation_context_id="context-1",
            hypothesis_execution_input_record_id="hypothesis-execution-input-1",
            investigation_session_id="session-1",
            workspace_identity="/private/workspace",
            minichem_scope_id="scope-1",
            scope_contract_digest=_SCOPE_CONTRACT_DIGEST,
            capability_generation_digest=_CAPABILITY_GENERATION_DIGEST,
        )

    assert {state.value for state in HarnessLifecycle} == {
        "open",
        "suspended",
        "completed",
        "failed",
    }
    assert "unknown" not in {state.value for state in HarnessLifecycle}
