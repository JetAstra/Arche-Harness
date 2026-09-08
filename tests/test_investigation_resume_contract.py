"""Investigation 恢复合同、sidecar sealed digest 和 WM waiting 对账测试。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from chemistry_multiagent.investigation.async_jobs import (
    InvestigationJobIdentity,
    InvestigationJobStatusError,
    InvestigationJobState,
    InvestigationWaitingJobStore,
)
from chemistry_multiagent.investigation.resume_contracts import (
    InvestigationResumeCheckpoint,
    ProviderCheckpoint,
    ResumeContractError,
    compute_resume_record_digest,
)
def _canonical_waiting_state() -> InvestigationJobState:
    """构造包含完整 v2 恢复身份的最小 waiting state。"""

    return InvestigationJobState(
        task_id="task-1",
        batch_id="batch-1",
        combination_id="combination-1",
        codex_thread_id="thread-1",
        turn_id="turn-1",
        status="waiting_investigation_job",
        identity=InvestigationJobIdentity(
            job_id="job-1",
            operation_id="submission-op-1",
            submission_key="submission-key-1",
            action_id="action-1",
            backend_id="backend-1",
        ),
        request_digest="a" * 64,
        submission_request_digest="a" * 64,
        turn_request_digest="b" * 64,
        thread_policy_digest="c" * 64,
        turn_count_before=1,
        session_id="session-1",
        stage="initial",
        logical_operation_instance_id="logical-op-1",
        attempt_operation_instance_id="attempt-op-1",
        attempt_number=1,
        scope_id="scope-1",
        workspace_ref_digest="d" * 64,
        submit_turn_id="turn-1",
        submit_turn_started_event_ref="started-event-1",
        turn_started_event_ref="started-event-1",
        scope_credential_generation_digest="e" * 64,
    )


def test_provider_checkpoint_required_fields_round_trip_without_digest_inequality() -> None:
    """必需的策略摘要和 turn_count_before 可往返，摘要值偶然相等也合法。"""

    checkpoint = ProviderCheckpoint(
        provider_id="codex",
        codex_thread_id="thread-1",
        codex_turn_id="turn-1",
        turn_request_digest="a" * 64,
        thread_policy_digest="b" * 64,
        turn_count_before=2,
    )
    assert ProviderCheckpoint.from_mapping(checkpoint.as_mapping()) == checkpoint
    assert ProviderCheckpoint(
        provider_id="codex",
        codex_thread_id="thread-1",
        codex_turn_id="turn-1",
        turn_request_digest="a" * 64,
        thread_policy_digest="b" * 64,
        turn_count_before=0,
    ).as_mapping()["turn_count_before"] == 0


def test_sidecar_save_returns_sealed_digest_and_source_provenance_is_immutable(
    tmp_path: Path,
) -> None:
    """每次写入都返回实际 sealed record，source provenance 进入不可变摘要。"""

    store = InvestigationWaitingJobStore(tmp_path / "waiting")
    state = _canonical_waiting_state()
    saved = store.save_waiting(state)
    reread = store.get(state.task_id, state.batch_id, state.combination_id)
    assert reread == saved
    assert saved["record_digest"] == compute_resume_record_digest(saved)

    without_source = {
        "state": state.as_mapping(),
        "resume_checkpoint": state.to_resume_checkpoint().as_mapping(),
    }
    with_source = dict(without_source)
    with_source["source_wake_state"] = {"turn_id": "wake-source"}
    assert compute_resume_record_digest(with_source) != compute_resume_record_digest(
        without_source
    )


def test_waiting_continuation_rolls_provider_attempt_without_replacing_job(
    tmp_path: Path,
) -> None:
    """同一 job 只前移 continuation attempt，执行身份和稳定主体保持不变。"""

    store = InvestigationWaitingJobStore(tmp_path / "waiting")
    previous = _canonical_waiting_state()
    original = store.save_waiting(previous)
    first_state = replace(
        previous,
        turn_id="turn-2",
        turn_request_digest="f" * 64,
        turn_count_before=2,
        attempt_operation_instance_id="attempt-op-2",
        attempt_number=2,
        turn_started_event_ref="started-event-2",
    )

    rolled = store.roll_forward_waiting_continuation_contract(
        previous,
        first_state,
    )

    assert rolled["record_digest"] != original["record_digest"]
    assert rolled["state"]["identity"] == original["state"]["identity"]
    assert rolled["state"]["turn_id"] == "turn-2"
    assert rolled["state"]["attempt_number"] == 2
    assert rolled["state"]["submit_turn_id"] == "turn-1"
    assert rolled["state"]["submit_turn_started_event_ref"] == "started-event-1"
    assert rolled["resume_checkpoint"]["turn_chain"] == {
        "submit_turn_id": "turn-1",
        "submit_turn_started_event_ref": "started-event-1",
        "active_continuation_turn_id": "turn-2",
        "active_continuation_turn_started_event_ref": "started-event-2",
    }
    assert rolled["resume_checkpoint"]["provider"]["codex_turn_id"] == "turn-2"
    assert rolled["resume_checkpoint"]["execution"] == original[
        "resume_checkpoint"
    ]["execution"]

    next_state = replace(
        first_state,
        turn_id="turn-3",
        turn_request_digest="0" * 64,
        turn_count_before=3,
        attempt_operation_instance_id="attempt-op-3",
        attempt_number=3,
        turn_started_event_ref="started-event-3",
    )
    rolled_twice = store.roll_forward_waiting_continuation_contract(
        first_state,
        next_state,
    )
    assert rolled_twice["resume_checkpoint"]["turn_chain"]["submit_turn_id"] == "turn-1"
    assert rolled_twice["resume_checkpoint"]["turn_chain"]["active_continuation_turn_id"] == "turn-3"
    assert rolled_twice["resume_checkpoint"]["provider"]["codex_turn_id"] == "turn-3"


def test_pending_v2_waiting_rolls_forward_to_first_complete_checkpoint(
    tmp_path: Path,
) -> None:
    """submit 回调后中断的最小 v2 记录可在下一 attempt 补齐合同。"""

    store = InvestigationWaitingJobStore(tmp_path / "waiting")
    canonical = _canonical_waiting_state()
    pending = replace(
        canonical,
        turn_request_digest=None,
        thread_policy_digest=None,
        turn_count_before=None,
        session_id=None,
        stage=None,
        logical_operation_instance_id=None,
        attempt_operation_instance_id=None,
        attempt_number=None,
        workspace_ref_digest=None,
        submit_turn_id=None,
        submit_turn_started_event_ref=None,
        turn_started_event_ref=None,
        scope_credential_generation_digest=None,
    )
    store.save_waiting(pending)
    next_state = replace(
        canonical,
        turn_id="turn-2",
        turn_request_digest="f" * 64,
        turn_count_before=2,
        attempt_operation_instance_id="attempt-op-2",
        attempt_number=2,
        submit_turn_id=pending.turn_id,
        submit_turn_started_event_ref="started-event-1",
        turn_started_event_ref="started-event-2",
    )

    rolled = store.roll_forward_waiting_continuation_contract(
        pending,
        next_state,
    )

    assert rolled["state"]["identity"] == canonical.as_mapping()["identity"]
    assert rolled["state"]["submit_turn_id"] == pending.turn_id
    assert rolled["state"]["turn_id"] == "turn-2"
    assert rolled["resume_checkpoint"]["turn_chain"] == {
        "submit_turn_id": pending.turn_id,
        "submit_turn_started_event_ref": "started-event-1",
        "active_continuation_turn_id": "turn-2",
        "active_continuation_turn_started_event_ref": "started-event-2",
    }


def test_waiting_continuation_rejects_same_turn_even_when_thread_changes(
    tmp_path: Path,
) -> None:
    """同一 turn 即使换 thread 也不能被误认作 continuation。"""

    store = InvestigationWaitingJobStore(tmp_path / "waiting")
    previous = _canonical_waiting_state()
    store.save_waiting(previous)
    same_turn_other_thread = replace(previous, codex_thread_id="thread-2")

    with pytest.raises(InvestigationJobStatusError, match="不能复用同一 provider turn"):
        store.roll_forward_waiting_continuation_contract(
            previous,
            same_turn_other_thread,
        )


def test_waiting_continuation_rejects_thread_migration_without_explicit_evidence(
    tmp_path: Path,
) -> None:
    """换 thread 必须有显式许可和相邻的已核验 thread 链。"""

    store = InvestigationWaitingJobStore(tmp_path / "waiting")
    previous = _canonical_waiting_state()
    store.save_waiting(previous)
    next_state = replace(
        previous,
        codex_thread_id="thread-2",
        turn_id="turn-2",
        turn_request_digest="f" * 64,
        turn_count_before=2,
        attempt_operation_instance_id="attempt-op-2",
        attempt_number=2,
        turn_started_event_ref="started-event-2",
    )

    with pytest.raises(InvestigationJobStatusError, match="默认不能更换 provider thread"):
        store.roll_forward_waiting_continuation_contract(previous, next_state)

    rolled = store.roll_forward_waiting_continuation_contract(
        previous,
        next_state,
        allow_provider_migration=True,
        migration_source_thread_id="thread-1",
        migration_verified_thread_ids=("thread-1", "thread-2"),
    )
    assert rolled["state"]["codex_thread_id"] == "thread-2"
    assert rolled["resume_checkpoint"]["provider"]["codex_thread_id"] == "thread-2"


@pytest.mark.parametrize(
    ("field_name", "replacement", "message"),
    [
        (
            "scope_credential_generation_digest",
            "f" * 64,
            "scope_credential_generation_digest 漂移",
        ),
        ("thread_policy_digest", "f" * 64, "thread_policy_digest 漂移"),
    ],
)
def test_waiting_continuation_rejects_stable_identity_drift(
    tmp_path: Path,
    field_name: str,
    replacement: str,
    message: str,
) -> None:
    """scope 代次和 thread 策略摘要必须在同一物理作业中保持不变。"""

    store = InvestigationWaitingJobStore(tmp_path / field_name)
    previous = _canonical_waiting_state()
    store.save_waiting(previous)
    next_state = replace(
        previous,
        turn_id="turn-2",
        turn_request_digest="f" * 64,
        turn_count_before=2,
        attempt_operation_instance_id="attempt-op-2",
        attempt_number=2,
        turn_started_event_ref="started-event-2",
        **{field_name: replacement},
    )

    with pytest.raises(InvestigationJobStatusError, match=message):
        store.roll_forward_waiting_continuation_contract(previous, next_state)


@pytest.mark.parametrize("field_name", ["attempt_number", "turn_count_before"])
def test_waiting_continuation_rejects_attempt_or_turn_count_jump(
    tmp_path: Path,
    field_name: str,
) -> None:
    """连续恢复的 attempt 与 provider turn count 只能各前进一格。"""

    store = InvestigationWaitingJobStore(tmp_path / field_name)
    previous = _canonical_waiting_state()
    store.save_waiting(previous)
    next_state = replace(
        previous,
        turn_id="turn-2",
        turn_request_digest="f" * 64,
        turn_count_before=3 if field_name == "turn_count_before" else 2,
        attempt_operation_instance_id="attempt-op-3" if field_name == "attempt_number" else "attempt-op-2",
        attempt_number=3 if field_name == "attempt_number" else 2,
        turn_started_event_ref="started-event-2",
    )

    with pytest.raises(InvestigationJobStatusError, match="必须精确前进一格"):
        store.roll_forward_waiting_continuation_contract(previous, next_state)


def test_v1_state_does_not_guess_missing_submit_turn_identity() -> None:
    """旧 v1 缺少原始 submit turn 时只能兼容读取，不能生成 v2 checkpoint。"""

    state = InvestigationJobState(
        task_id="task-1",
        batch_id="batch-1",
        combination_id="combination-1",
        codex_thread_id="thread-1",
        turn_id="turn-1",
        status="waiting_investigation_job",
        identity=InvestigationJobIdentity(
            job_id="job-1",
            operation_id="submission-op-1",
            submission_key="submission-key-1",
            action_id="action-1",
            backend_id="backend-1",
        ),
        request_digest="a" * 64,
        scope_id="scope-1",
    )

    assert state.to_resume_checkpoint(strict=False) is None


def test_waiting_continuation_roll_forward_rejects_wake_handoff(
    tmp_path: Path,
) -> None:
    """已有 wake intent 的 sidecar 不能被普通 continuation 覆盖。"""

    store = InvestigationWaitingJobStore(tmp_path / "waiting")
    previous = _canonical_waiting_state()
    store.save_waiting(previous)
    store.arm_terminal(previous, terminal_status="succeeded")
    store.record_wake_intent(
        previous,
        {"messages": [{"role": "user", "content": "wake"}]},
    )
    next_state = replace(
        previous,
        turn_id="turn-2",
        turn_request_digest="f" * 64,
        turn_count_before=2,
        attempt_operation_instance_id="attempt-op-2",
        attempt_number=2,
        turn_started_event_ref="started-event-2",
    )

    with pytest.raises(InvestigationJobStatusError, match="不能覆盖 wake handoff"):
        store.roll_forward_waiting_continuation_contract(previous, next_state)


def test_provider_checkpoint_rejects_unknown_field() -> None:
    """checkpoint 含未知字段时不能被兼容读取为 canonical。"""

    with pytest.raises(ResumeContractError):
        ProviderCheckpoint.from_mapping(
            {
                "provider_id": "codex",
                "codex_thread_id": "thread-1",
                "codex_turn_id": "turn-1",
                "turn_request_digest": "a" * 64,
                "request_digest": "b" * 64,
            }
        )


def test_active_continuation_provider_turn_is_cross_checked() -> None:
    """分组 checkpoint 的 provider turn 必须指向当前 continuation，而非 T0。"""

    state = _canonical_waiting_state()
    continuation = replace(
        state,
        turn_id="turn-2",
        turn_request_digest="f" * 64,
        turn_count_before=2,
        attempt_operation_instance_id="attempt-op-2",
        attempt_number=2,
        turn_started_event_ref="started-event-2",
    )
    checkpoint = continuation.to_resume_checkpoint()
    assert checkpoint is not None
    mapping = checkpoint.as_mapping()
    mapping["provider"]["codex_turn_id"] = "turn-1"

    with pytest.raises(ResumeContractError, match="active continuation turn"):
        InvestigationResumeCheckpoint.from_mapping(mapping)
