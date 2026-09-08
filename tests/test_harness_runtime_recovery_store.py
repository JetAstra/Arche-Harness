"""Investigation Harness 与 HarnessRecoveryStore 的恢复边界回归。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from chemistry_multiagent.investigation.harness_recovery_store import (
    HarnessLifecycle,
    HarnessRecoveryStore,
    HarnessRecoveryStateConflict,
    HarnessRecoveryStoreError,
    HarnessRunIdentity,
    CommitModuleOutput,
)
from chemistry_multiagent.investigation.harness_runtime import (
    BackendCompleted,
    BackendFatalFailure,
    BackendSuspended,
    Completed,
    FatalFailure,
    InvestigationHarness,
    InvestigationHarnessContractError,
    InvestigationHarnessIdentityConflict,
    InvestigationHarnessRecoveryStateWriteError,
    InvestigationHarnessRequest,
    Suspended,
    SchemaV1LegacyHarnessManifestReader,
    SchemaV1LegacyPreparedModuleOutputReader,
    derive_harness_invocation_id,
)
from chemistry_multiagent.memory import PreparedModuleInvocationOutput, ScientificRecord


_INPUT_DIGEST = "a" * 64


def _request(
    *,
    research_task_id: str = "research-task-1",
    module_invocation_id: str = "module-invocation-1",
    module_input_digest: str = _INPUT_DIGEST,
) -> InvestigationHarnessRequest:
    return InvestigationHarnessRequest(
        research_task_id=research_task_id,
        module_invocation_id=module_invocation_id,
        module_kind="hypothesis-investigation",
        input_scientific_record_ids=("hypothesis-execution-input-1",),
        module_input_digest=module_input_digest,
        context_binding={
            "kind": "CreateInvestigationContext",
            "investigation_context_id": "investigation-context-1",
            "hypothesis_batch_record_id": "hypothesis-batch-1",
        },
    )


def _identity(request: InvestigationHarnessRequest) -> HarnessRunIdentity:
    return HarnessRunIdentity(
        research_task_id=request.research_task_id,
        module_invocation_id=request.module_invocation_id,
        harness_invocation_id=request.harness_invocation_id,
        module_input_digest=request.module_input_digest,
    )


def _output(request: InvestigationHarnessRequest) -> PreparedModuleInvocationOutput:
    evidence_ref = f"{request.module_invocation_id}:evidence"
    result_ref = f"{request.module_invocation_id}:result"
    evidence = ScientificRecord.from_payload(
        scientific_record_id=evidence_ref,
        research_task_id=request.research_task_id,
        scientific_record_type="VerifiedEvidenceRecord",
        scientific_record_schema_version=1,
        producer_module_invocation_id=request.module_invocation_id,
        source_scientific_record_ids=request.input_scientific_record_ids,
        payload={
            "evidence_items": [{"value": "observed"}],
            "artifact_refs": [],
            "source_manifest_ref": "manifest-1",
            "summary": "evidence",
            "verification": {"verified": True},
        },
    )
    result = ScientificRecord.from_payload(
        scientific_record_id=result_ref,
        research_task_id=request.research_task_id,
        scientific_record_type="InvestigationResultRecord",
        scientific_record_schema_version=1,
        producer_module_invocation_id=request.module_invocation_id,
        source_scientific_record_ids=(evidence_ref,),
        payload={
            "conclusion": "supported",
            "self_judgment": "supported",
            "observations": ["observed"],
            "limitations": [],
            "evidence_ref": evidence_ref,
        },
    )
    return PreparedModuleInvocationOutput.from_records(
        module_invocation_id=request.module_invocation_id,
        module_input_digest=request.module_input_digest,
        output_scientific_records=(evidence, result),
    )


class ScriptedBackend:
    """只返回预设结果的本地 backend，不访问任何外部服务。"""

    def __init__(
        self,
        start_result: Any = None,
        resume_result: Any = None,
        *,
        start_error: BaseException | None = None,
    ) -> None:
        self.start_result = start_result
        self.resume_result = resume_result
        self.start_error = start_error
        self.calls: list[str] = []

    @staticmethod
    def _resolve(result: Any, request: InvestigationHarnessRequest) -> Any:
        return result(request) if callable(result) else result

    def start(self, request: InvestigationHarnessRequest, harness_invocation_id: str) -> Any:
        self.calls.append("start")
        if self.start_error is not None:
            raise self.start_error
        result = self.start_result
        if result is None:
            result = BackendCompleted(_output(request))
        return self._resolve(result, request)

    def resume(self, request: InvestigationHarnessRequest, harness_invocation_id: str) -> Any:
        self.calls.append("resume")
        result = self.resume_result
        if result is None:
            result = self.start_result
        if result is None:
            result = BackendCompleted(_output(request))
        return self._resolve(result, request)


def _store(tmp_path: Path) -> HarnessRecoveryStore:
    return HarnessRecoveryStore(tmp_path / "runtime" / "harness-recovery.sqlite3")


def test_completed_output_is_authoritative_and_manifest_is_only_display_data(
    tmp_path: Path,
) -> None:
    request = _request()
    store = _store(tmp_path)
    backend = ScriptedBackend()
    harness = InvestigationHarness(
        tmp_path / "runtime",
        backend=backend,
        recovery_store=store,
    )

    outcome = harness.start(request)

    assert isinstance(outcome, Completed)
    snapshot = store.read_snapshot(_identity(request))
    assert snapshot.state is HarnessLifecycle.COMPLETED
    assert snapshot.committed_output is not None
    assert snapshot.committed_output.output_digest == outcome.prepared_output.module_output_digest
    output_path = next((tmp_path / "runtime").rglob("prepared_output.json"))
    assert json.loads(output_path.read_text(encoding="utf-8"))["module_output_digest"] == (
        outcome.prepared_output.module_output_digest
    )

    # 展示索引即使被改成旧状态，也不能改变独立恢复事实。
    manifest_path = next((tmp_path / "runtime").rglob("manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["state"] = "not-a-lifecycle-state"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    replay_backend = ScriptedBackend(
        start_result=AssertionError("completed invocation must not call backend"),
        resume_result=AssertionError("completed invocation must not call backend"),
    )
    replay = InvestigationHarness(
        tmp_path / "runtime",
        backend=replay_backend,
        recovery_store=HarnessRecoveryStore(store.database_path),
    ).resume(request)

    assert isinstance(replay, Completed)
    assert replay.prepared_output == outcome.prepared_output
    assert replay_backend.calls == []


def test_backend_process_interruption_leaves_open_store_for_resume(
    tmp_path: Path,
) -> None:
    request = _request()
    store = _store(tmp_path)
    backend = ScriptedBackend(start_error=RuntimeError("provider process disappeared"))
    harness = InvestigationHarness(tmp_path / "runtime", backend=backend, recovery_store=store)

    with pytest.raises(RuntimeError, match="process disappeared"):
        harness.start(request)
    assert store.read_snapshot(_identity(request)).state is HarnessLifecycle.OPEN

    backend.start_error = None
    resumed = harness.resume(request)

    assert isinstance(resumed, Completed)
    assert backend.calls == ["start", "resume"]
    assert store.read_snapshot(_identity(request)).state is HarnessLifecycle.COMPLETED


def test_suspended_and_integrity_failure_write_explicit_store_states(
    tmp_path: Path,
) -> None:
    suspended_request = _request(module_invocation_id="module-invocation-suspended")
    suspended_store = _store(tmp_path / "suspended")
    suspended_backend = ScriptedBackend(
        start_result=BackendSuspended("provider_unavailable")
    )
    suspended_harness = InvestigationHarness(
        tmp_path / "suspended" / "runtime",
        backend=suspended_backend,
        recovery_store=suspended_store,
    )

    suspended = suspended_harness.start(suspended_request)

    assert isinstance(suspended, Suspended)
    assert suspended_store.read_snapshot(_identity(suspended_request)).state is (
        HarnessLifecycle.SUSPENDED
    )

    # 科学作业失败应留在 ActionResult/模型 loop 或形成正常科学记录；不得
    # 伪造成 Harness 的不可恢复致命分支。
    with pytest.raises(InvestigationHarnessContractError):
        BackendFatalFailure("scientific_failed")

    failed_request = _request(module_invocation_id="module-invocation-failed")
    failed_store = _store(tmp_path / "failed")
    failed_backend = ScriptedBackend(
        start_result=BackendFatalFailure("verified_integrity_conflict")
    )
    failed_harness = InvestigationHarness(
        tmp_path / "failed" / "runtime",
        backend=failed_backend,
        recovery_store=failed_store,
    )

    failed = failed_harness.start(failed_request)

    assert isinstance(failed, FatalFailure)
    failed_snapshot = failed_store.read_snapshot(_identity(failed_request))
    assert failed_snapshot.state is HarnessLifecycle.FAILED
    assert failed_snapshot.failure_code == "verified_integrity_conflict"
    replay = InvestigationHarness(
        tmp_path / "failed" / "runtime",
        backend=ScriptedBackend(start_result=AssertionError("must not call backend")),
        recovery_store=HarnessRecoveryStore(failed_store.database_path),
    ).resume(failed_request)
    assert isinstance(replay, FatalFailure)
    assert replay.failure_code == "verified_integrity_conflict"


def test_single_job_failure_stays_recoverable_for_backend_model_loop(
    tmp_path: Path,
) -> None:
    """单个作业失败不能被 Harness 误写成 Module/Task 致命失败。"""

    request = _request(module_invocation_id="module-invocation-job-failed")
    store = _store(tmp_path)

    class BackendThatContinuesModelLoop:
        """在 backend 内部消费失败 ActionResult 后继续同一模型回合。"""

        def __init__(self) -> None:
            self.calls: list[str] = []
            self.failed_actions: list[str] = []

        def start(self, request: InvestigationHarnessRequest, harness_invocation_id: str) -> Any:
            self.calls.append("start")
            self.failed_actions.append("minichem_job_failed")
            return BackendCompleted(_output(request))

        def resume(self, request: InvestigationHarnessRequest, harness_invocation_id: str) -> Any:
            raise AssertionError("已在 backend 模型 loop 内继续，不应重复 resume")

    backend = BackendThatContinuesModelLoop()
    harness = InvestigationHarness(
        tmp_path / "runtime",
        backend=backend,
        recovery_store=store,
    )

    first = harness.start(request)

    assert isinstance(first, Completed)
    assert backend.failed_actions == ["minichem_job_failed"]
    assert backend.calls == ["start"]
    assert store.read_snapshot(_identity(request)).state is HarnessLifecycle.COMPLETED


def test_integrity_failure_is_recorded_before_identity_error_propagates(
    tmp_path: Path,
) -> None:
    request = _request()
    wrong_request = _request(module_invocation_id="module-invocation-wrong")
    store = _store(tmp_path)
    backend = ScriptedBackend(start_result=BackendCompleted(_output(wrong_request)))
    harness = InvestigationHarness(tmp_path / "runtime", backend=backend, recovery_store=store)

    with pytest.raises(InvestigationHarnessIdentityConflict):
        harness.start(request)

    snapshot = store.read_snapshot(_identity(request))
    assert snapshot.state is HarnessLifecycle.FAILED
    assert snapshot.failure_code == "verified_integrity_conflict"


def test_explicit_harness_invocation_id_drift_is_not_silently_rewritten(
    tmp_path: Path,
) -> None:
    request = _request()
    store = _store(tmp_path)
    harness = InvestigationHarness(tmp_path / "runtime", backend=ScriptedBackend(), recovery_store=store)
    assert isinstance(harness.start(request), Completed)

    with pytest.raises(InvestigationHarnessIdentityConflict, match="harness_invocation_id"):
        harness.resume("harness-other", request)


def test_output_and_store_commit_are_recovered_without_duplicate_backend_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request()
    store = _store(tmp_path)
    backend = ScriptedBackend()
    harness = InvestigationHarness(tmp_path / "runtime", backend=backend, recovery_store=store)
    original_append = store.append_transition
    commit_failed = False

    def fail_commit(identity: HarnessRunIdentity, transition: Any) -> Any:
        nonlocal commit_failed
        if isinstance(transition, CommitModuleOutput) and not commit_failed:
            commit_failed = True
            raise HarnessRecoveryStoreError("simulated store interruption")
        return original_append(identity, transition)

    monkeypatch.setattr(store, "append_transition", fail_commit)
    with pytest.raises(InvestigationHarnessRecoveryStateWriteError):
        harness.start(request)
    assert backend.calls == ["start"]
    assert store.read_snapshot(_identity(request)).state is HarnessLifecycle.OPEN
    assert next((tmp_path / "runtime").rglob("prepared_output.json")).is_file()

    monkeypatch.setattr(store, "append_transition", original_append)
    replay_backend = ScriptedBackend(
        start_result=AssertionError("staged output should be finalized locally"),
        resume_result=AssertionError("staged output should be finalized locally"),
    )
    replay = InvestigationHarness(
        tmp_path / "runtime",
        backend=replay_backend,
        recovery_store=HarnessRecoveryStore(store.database_path),
    ).resume(request)

    assert isinstance(replay, Completed)
    assert replay_backend.calls == []
    assert store.read_snapshot(_identity(request)).state is HarnessLifecycle.COMPLETED


def test_same_runtime_can_have_independent_module_invocations(tmp_path: Path) -> None:
    first = _request(module_invocation_id="module-invocation-1", module_input_digest="1" * 64)
    second = _request(module_invocation_id="module-invocation-2", module_input_digest="2" * 64)
    store = _store(tmp_path)
    backend = ScriptedBackend()
    harness = InvestigationHarness(tmp_path / "runtime", backend=backend, recovery_store=store)

    assert isinstance(harness.start(first), Completed)
    assert isinstance(harness.start(second), Completed)
    assert store.read_snapshot(_identity(first)).state is HarnessLifecycle.COMPLETED
    assert store.read_snapshot(_identity(second)).state is HarnessLifecycle.COMPLETED
    assert backend.calls == ["start", "start"]


def test_schema_v1_manifest_reader_converts_only_at_explicit_legacy_boundary() -> None:
    """schema-v1 manifest 只能由显式 reader 读取，不能混入活动 request。"""

    request = _request()
    legacy_request = {
        "task_id": request.research_task_id,
        "module_run_id": request.module_invocation_id,
        "module_kind": "investigation-and-experiment",
        "runtime_ref": "legacy-runtime-1",
        "input_record_refs": list(request.input_scientific_record_ids),
        "input_digest": request.module_input_digest,
    }
    legacy_ref = derive_harness_invocation_id("legacy-runtime-1")
    manifest = {
        "schema_version": 1,
        "harness_run_ref": legacy_ref,
        "request": legacy_request,
        "state": "completed",
        "failure_code": None,
    }

    parsed = SchemaV1LegacyHarnessManifestReader.read(
        manifest,
        requested_harness_invocation_id=legacy_ref,
    )

    converted = parsed["request"]
    assert converted.research_task_id == request.research_task_id
    assert converted.module_invocation_id == request.module_invocation_id
    assert converted.module_kind == "hypothesis-investigation"
    assert converted.input_scientific_record_ids == request.input_scientific_record_ids
    assert "task_id" not in converted.to_dict()
    assert "runtime_ref" not in converted.to_dict()


def test_schema_v1_prepared_output_reader_converts_old_envelope_without_writing() -> None:
    """旧 output envelope 仅在显式 reader 中归一，canonical 输出不保留旧键。"""

    request = _request()
    output = _output(request)
    legacy_records = []
    for record in output.output_scientific_records:
        legacy_records.append(
            {
                "record_id": record.scientific_record_id,
                "task_id": record.research_task_id,
                "record_type": record.scientific_record_type,
                "schema_version": record.scientific_record_schema_version,
                "producer_module_run_id": record.producer_module_invocation_id,
                "source_record_refs": list(record.source_scientific_record_ids),
                "payload": dict(record.payload),
                "payload_digest": record.payload_digest,
            }
        )
    legacy_output = {
        "module_run_id": output.module_invocation_id,
        "input_digest": output.module_input_digest,
        "output_records": legacy_records,
        "output_digest": output.module_output_digest,
    }

    converted = SchemaV1LegacyPreparedModuleOutputReader.read(legacy_output)

    assert converted == output
    assert set(converted.to_dict()) == {
        "module_invocation_id",
        "module_input_digest",
        "output_scientific_records",
        "module_output_digest",
    }


def test_recovery_store_write_failure_is_not_reported_as_suspension(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request()
    store = _store(tmp_path)
    backend = ScriptedBackend(start_result=BackendSuspended("provider_unavailable"))
    harness = InvestigationHarness(tmp_path / "runtime", backend=backend, recovery_store=store)

    def fail_write(*args: Any, **kwargs: Any) -> Any:
        raise HarnessRecoveryStoreError("simulated state write failure")

    monkeypatch.setattr(store, "append_transition", fail_write)

    with pytest.raises(InvestigationHarnessRecoveryStateWriteError):
        harness.start(request)


def test_recovery_state_contract_error_is_not_reported_as_identity_conflict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(module_invocation_id="module-invocation-state-contract")
    store = _store(tmp_path)
    harness = InvestigationHarness(
        tmp_path / "runtime",
        backend=ScriptedBackend(start_result=BackendSuspended("provider_unavailable")),
        recovery_store=store,
    )

    def fail_transition(*args: Any, **kwargs: Any) -> Any:
        raise HarnessRecoveryStateConflict("invalid lifecycle transition")

    monkeypatch.setattr(store, "append_transition", fail_transition)

    with pytest.raises(InvestigationHarnessContractError, match="lifecycle transition"):
        harness.start(request)
