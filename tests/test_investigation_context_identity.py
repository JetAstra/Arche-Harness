"""Investigation Harness 的 canonical Module Invocation 恢复边界。"""

from __future__ import annotations

from pathlib import Path

import pytest

from chemistry_multiagent.investigation.harness_runtime import (
    BackendFatalFailure,
    BackendSuspended,
    Completed,
    FatalFailure,
    InvestigationHarness,
    InvestigationHarnessIdentityConflict,
    InvestigationHarnessRequest,
    Suspended,
)
from chemistry_multiagent.memory import PreparedModuleInvocationOutput, ScientificRecord


class RecordingBackend:
    """不启动模型或 MiniChem 的确定性 Harness backend。"""

    def __init__(self, *results: object) -> None:
        self.results = list(results)
        self.calls: list[tuple[str, InvestigationHarnessRequest, str]] = []

    def start(
        self,
        request: InvestigationHarnessRequest,
        harness_invocation_id: str,
    ) -> object:
        self.calls.append(("start", request, harness_invocation_id))
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    def resume(
        self,
        request: InvestigationHarnessRequest,
        harness_invocation_id: str,
    ) -> object:
        self.calls.append(("resume", request, harness_invocation_id))
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


def _request(*, module_input_digest: str = "a" * 64) -> InvestigationHarnessRequest:
    """构造只含模块边界字段的请求。"""

    return InvestigationHarnessRequest(
        research_task_id="research-task-1",
        module_invocation_id="module-invocation-1",
        module_kind="hypothesis-investigation",
        input_scientific_record_ids=("hypothesis-execution-input-1",),
        module_input_digest=module_input_digest,
        context_binding={
            "kind": "CreateInvestigationContext",
            "investigation_context_id": "investigation-context-1",
            "hypothesis_batch_record_id": "hypothesis-batch-1",
        },
    )


def _prepared_output(
    *,
    research_task_id: str = "research-task-1",
    module_invocation_id: str = "module-invocation-1",
    module_input_digest: str = "a" * 64,
) -> PreparedModuleInvocationOutput:
    """构造不含运行时身份的确定性科学输出包。"""

    evidence = ScientificRecord.from_payload(
        scientific_record_id="evidence-1",
        research_task_id=research_task_id,
        scientific_record_type="VerifiedEvidenceRecord",
        scientific_record_schema_version=1,
        producer_module_invocation_id=module_invocation_id,
        source_scientific_record_ids=(),
        payload={
            "evidence_items": [{"kind": "observation", "value": "stable"}],
            "artifact_refs": [],
            "source_manifest_ref": "manifest-1",
            "summary": "确定性测试证据",
            "verification": {"verified": True},
        },
    )
    result = ScientificRecord.from_payload(
        scientific_record_id="result-1",
        research_task_id=research_task_id,
        scientific_record_type="InvestigationResultRecord",
        scientific_record_schema_version=1,
        producer_module_invocation_id=module_invocation_id,
        source_scientific_record_ids=("evidence-1",),
        payload={
            "conclusion": "stable",
            "self_judgment": "supported",
            "observations": ["stable"],
            "limitations": [],
            "evidence_ref": "evidence-1",
        },
    )
    return PreparedModuleInvocationOutput.from_records(
        module_invocation_id=module_invocation_id,
        module_input_digest=module_input_digest,
        output_scientific_records=(evidence, result),
    )


def test_start_persists_prepared_output_before_returning_completed(
    tmp_path: Path,
) -> None:
    """首次 backend 完成时必须先密封输出，公开结果只暴露模块字段。"""

    backend = RecordingBackend(_prepared_output())
    harness = InvestigationHarness(tmp_path / "harness-runtime", backend=backend)

    outcome = harness.start(_request())

    assert isinstance(outcome, Completed)
    assert outcome.harness_invocation_id
    assert outcome.prepared_output == _prepared_output()
    assert len(backend.calls) == 1
    assert set(backend.calls[0][1].to_dict()) == {
        "research_task_id",
        "module_invocation_id",
        "module_kind",
        "input_scientific_record_ids",
        "module_input_digest",
        "context_binding",
    }
    assert tuple(outcome.to_dict()) == ("status", "harness_invocation_id", "prepared_output")
    serialized = str(outcome.to_dict())
    for private_name in ("thread", "turn", "goal", "scope", "job", "receipt", "sidecar", "path"):
        assert private_name not in serialized.lower()

    runtime_files = list((tmp_path / "harness-runtime").rglob("*"))
    assert any(path.name == "prepared_output.json" for path in runtime_files)


def test_new_facade_replays_prepared_output_without_backend_call(
    tmp_path: Path,
) -> None:
    """新进程 facade 读取同一 Module Invocation 时只能重放已密封输出。"""

    runtime_root = tmp_path / "harness-runtime"
    first_backend = RecordingBackend(_prepared_output())
    first = InvestigationHarness(runtime_root, backend=first_backend)
    first_outcome = first.start(_request())

    second_backend = RecordingBackend()
    second = InvestigationHarness(runtime_root, backend=second_backend)
    replay = second.start(_request())
    resumed = second.resume(first_outcome.harness_invocation_id)

    assert isinstance(replay, Completed)
    assert isinstance(resumed, Completed)
    assert replay.prepared_output == first_outcome.prepared_output
    assert resumed.prepared_output == first_outcome.prepared_output
    assert second_backend.calls == []


def test_same_invocation_with_input_drift_fails_closed(
    tmp_path: Path,
) -> None:
    """同一 Module Invocation 不得静默绑定另一份任务、模块或输入。"""

    backend = RecordingBackend(_prepared_output())
    harness = InvestigationHarness(tmp_path / "harness-runtime", backend=backend)
    harness.start(_request())

    drifted_input = _request(module_input_digest="b" * 64)
    with pytest.raises(InvestigationHarnessIdentityConflict):
        harness.start(drifted_input)

    # module_invocation_id 改变会派生独立 Harness invocation，而非覆盖原身份。
    independent_request = InvestigationHarnessRequest(
        research_task_id="research-task-1",
        module_invocation_id="module-invocation-2",
        module_kind="hypothesis-investigation",
        input_scientific_record_ids=("hypothesis-execution-input-1",),
        module_input_digest="c" * 64,
        context_binding=drifted_input.context_binding,
    )
    independent_harness = InvestigationHarness(
        tmp_path / "independent-runtime",
        backend=RecordingBackend(
            _prepared_output(
                module_invocation_id="module-invocation-2",
                module_input_digest="c" * 64,
            )
        ),
    )
    assert isinstance(independent_harness.start(independent_request), Completed)


def test_suspended_reopens_same_identity_and_fatal_failure_is_replayed(
    tmp_path: Path,
) -> None:
    """Suspended 不伪造失败；FatalFailure 终态可安全重放。"""

    runtime_root = tmp_path / "harness-runtime"
    first_backend = RecordingBackend(BackendSuspended("owner_unavailable"))
    first = InvestigationHarness(runtime_root, backend=first_backend)
    suspended = first.start(_request())
    assert isinstance(suspended, Suspended)
    assert "running" not in str(suspended.to_dict()).lower()

    second_backend = RecordingBackend(BackendFatalFailure("verified_integrity_conflict"))
    second = InvestigationHarness(runtime_root, backend=second_backend)
    failed = second.resume(suspended.harness_invocation_id)
    assert isinstance(failed, FatalFailure)
    assert failed.failure_code == "verified_integrity_conflict"
    assert len(second_backend.calls) == 1

    replay = second.resume(suspended.harness_invocation_id)
    assert isinstance(replay, FatalFailure)
    assert replay == failed
    assert len(second_backend.calls) == 1


def test_backend_completed_output_is_sealed_before_completed_outcome(
    tmp_path: Path,
) -> None:
    """输出包存在时恢复不依赖 backend 的进程内状态。"""

    output = _prepared_output()
    backend = RecordingBackend(output)
    harness = InvestigationHarness(tmp_path / "harness-runtime", backend=backend)
    outcome = harness.start(_request())

    assert isinstance(outcome, Completed)
    prepared_files = list((tmp_path / "harness-runtime").rglob("prepared_output.json"))
    assert len(prepared_files) == 1
    assert prepared_files[0].read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("output_kwargs", "expected_message"),
    [
        ({"module_invocation_id": "module-invocation-other"}, "module_invocation_id"),
        ({"research_task_id": "research-task-other"}, "research_task_id"),
        ({"module_input_digest": "b" * 64}, "module_input_digest"),
    ],
)
def test_backend_output_identity_drift_fails_closed_without_prepared_output(
    tmp_path: Path,
    output_kwargs: dict[str, str],
    expected_message: str,
) -> None:
    """错误 backend 输出不得密封，且漂移后不能再次驱动 backend。"""

    backend = RecordingBackend(_prepared_output(**output_kwargs))
    harness = InvestigationHarness(tmp_path / "harness-runtime", backend=backend)

    with pytest.raises(InvestigationHarnessIdentityConflict, match=expected_message):
        harness.start(_request())

    assert not list((tmp_path / "harness-runtime").rglob("prepared_output.json"))
    failed = harness.resume(_request())
    assert isinstance(failed, FatalFailure)
    assert failed.failure_code == "verified_integrity_conflict"
    assert len(backend.calls) == 1


def test_unclassified_backend_error_keeps_runtime_open_for_later_resume(
    tmp_path: Path,
) -> None:
    """暂时性 backend 异常不应被错误写成 FatalFailure。"""

    runtime_root = tmp_path / "harness-runtime"
    first_backend = RecordingBackend(RuntimeError("temporary connection failure"))
    first = InvestigationHarness(runtime_root, backend=first_backend)
    with pytest.raises(RuntimeError, match="temporary connection failure"):
        first.start(_request())

    second_backend = RecordingBackend(_prepared_output())
    second = InvestigationHarness(runtime_root, backend=second_backend)
    outcome = second.resume(_request())

    assert isinstance(outcome, Completed)
    assert first_backend.calls[0][0] == "start"
    assert second_backend.calls[0][0] == "resume"


def test_request_rejects_non_investigation_module_kind() -> None:
    """该 facade 只拥有 Investigation and Experiment 模块边界。"""

    with pytest.raises(ValueError, match="module_kind"):
        InvestigationHarnessRequest(
            research_task_id="research-task-1",
            module_invocation_id="module-invocation-1",
            module_kind="retrieval",
            input_scientific_record_ids=("scientific-record-1",),
            module_input_digest="a" * 64,
            context_binding={"kind": "StatelessModule"},
        )
