"""Investigation and Experiment 领域合同的确定性测试。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from chemistry_multiagent.investigation import (
    ApplyBatchValidation,
    BatchValidation,
    BeginNextBatch,
    InvestigationResult,
    InvestigationSession,
    InvestigationState,
    RecordInvestigationResult,
    ValidationFinding,
    apply_investigation_event,
)


COMBINATION_IDS = tuple(f"combination-{index}" for index in range(1, 6))


def make_session(
    combination_id: str,
    *,
    batch_id: str = "batch-1",
    phase: str = "created",
    result_version: str | None = None,
    repair_count: int = 0,
    route_marker: str | None = None,
) -> InvestigationSession:
    return InvestigationSession(
        task_id="task-1",
        batch_id=batch_id,
        combination_id=combination_id,
        codex_thread_id=(
            None
            if phase == "created"
            else f"thread-{combination_id}"
        ),
        workspace_ref=f"workspace/{combination_id}",
        scope_id=f"scope-{combination_id}",
        result_version=result_version,
        repair_count=repair_count,
        phase=phase,
        route_marker=route_marker,
    )


def make_state(*, batch_id: str = "batch-1") -> InvestigationState:
    return InvestigationState(
        task_id="task-1",
        batch_id=batch_id,
        sessions=tuple(
            make_session(combination_id, batch_id=batch_id)
            for combination_id in COMBINATION_IDS
        ),
    )


def make_result(
    combination_id: str,
    version: str,
    *,
    batch_id: str = "batch-1",
    judgment: str = "supported",
) -> InvestigationResult:
    return InvestigationResult(
        task_id="task-1",
        batch_id=batch_id,
        combination_id=combination_id,
        result_version=version,
        codex_thread_id=f"thread-{combination_id}",
        scope_id=f"scope-{combination_id}",
        turn_id=f"turn-{version}",
        self_judgment=judgment,
        narrative=f"对 {combination_id} 的完整回顾性计算过程：版本 {version}。",
        job_refs=(f"job-{version}",),
        operation_refs=(f"operation-{version}",),
        artifact_refs=(f"artifact-{version}",),
        observations=(f"observation-{version}",),
    )


def record_all_results(state: InvestigationState, *, prefix: str = "v") -> InvestigationState:
    for index, combination_id in enumerate(COMBINATION_IDS, start=1):
        state = apply_investigation_event(
            state,
            RecordInvestigationResult(
                result=make_result(combination_id, f"{prefix}{index}")
            ),
        )
    return state


def make_validation(
    state: InvestigationState,
    *,
    validity: str = "pass",
    judgment: str = "supported",
    can_answer: bool = True,
    final_answer: str | None = None,
    answerability_explanation: str | None = None,
    advice_suffix: str = "补充或修正该组合的计算。",
    fail_combination_ids: set[str] | None = None,
) -> BatchValidation:
    findings = tuple(
        ValidationFinding(
            combination_id=combination_id,
            validity=(
                "fail"
                if validity == "fail"
                or (fail_combination_ids and combination_id in fail_combination_ids)
                else "pass"
            ),
            hypothesis_judgment=(
                judgment
                if not (
                    validity == "fail"
                    or (fail_combination_ids and combination_id in fail_combination_ids)
                )
                else None
            ),
            repair_advice=(
                f"{combination_id}: {advice_suffix}"
                if (
                    validity == "fail"
                    or (fail_combination_ids and combination_id in fail_combination_ids)
                )
                else None
            ),
        )
        for combination_id in COMBINATION_IDS
    )
    return BatchValidation(
        task_id=state.task_id,
        batch_id=state.batch_id,
        combination_ids=COMBINATION_IDS,
        result_versions=tuple(
            state.session_for(combination_id).result_version  # type: ignore[arg-type]
            for combination_id in COMBINATION_IDS
        ),
        findings=findings,
        can_answer=can_answer,
        **(
            {
                "final_answer": (
                    final_answer
                    if final_answer is not None
                    else "五项结果支持对原科学问题的回答。"
                )
            }
            if can_answer
            else (
                {"answerability_explanation": answerability_explanation}
                if answerability_explanation is not None
                else {}
            )
        ),
        review_thread_id="review-thread-1",
        review_turn_id="review-turn-1",
    )


def test_state_and_records_are_immutable_and_strict() -> None:
    state = record_all_results(make_state())
    original_versions = tuple(
        state.session_for(combination_id).result_version
        for combination_id in COMBINATION_IDS
    )

    with pytest.raises(ValidationError):
        state.task_id = "other-task"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        InvestigationResult(
            task_id="task-1",
            batch_id="batch-1",
            combination_id="combination-1",
            result_version="v-invalid",
            codex_thread_id="thread-combination-1",
            turn_id="turn-invalid",
            self_judgment="inconclusive",
            narrative="invalid scientific state",
        )

    assert original_versions == ("v1", "v2", "v3", "v4", "v5")

    with pytest.raises(ValidationError, match="repair_pending"):
        InvestigationSession(
            **make_session(
                "combination-1",
                result_version="v1",
                phase="repair_pending",
                repair_count=3,
            ).model_dump()
        )
    with pytest.raises(ValidationError, match="created"):
        InvestigationSession(
            **make_session(
                "combination-1",
                repair_count=1,
            ).model_dump()
        )


def test_result_contract_requires_at_least_one_observation() -> None:
    """领域持久记录不能绕过模型响应合同保存无观测的科学判断。"""

    payload = make_result(
        "combination-1",
        "v-empty-observation",
    ).model_dump(mode="python")
    payload["observations"] = ()
    with pytest.raises(ValidationError, match="至少需要一项实际观测"):
        InvestigationResult(**payload)


def test_created_session_binds_real_thread_on_first_result_and_reuses_it_for_repair() -> None:
    unbound = InvestigationSession(
        task_id="task-1",
        batch_id="batch-1",
        combination_id="combination-1",
        codex_thread_id=None,
        workspace_ref="workspace/combination-1",
        scope_id="scope-combination-1",
    )
    assert unbound.codex_thread_id is None

    state = InvestigationState(
        task_id="task-1",
        batch_id="batch-1",
        sessions=(unbound, *(make_session(item) for item in COMBINATION_IDS[1:])),
    )
    first = make_result("combination-1", "v1")
    state = apply_investigation_event(
        state,
        RecordInvestigationResult(result=first),
    )
    assert state.session_for("combination-1").codex_thread_id == first.codex_thread_id
    for combination_id in COMBINATION_IDS[1:]:
        state = apply_investigation_event(
            state,
            RecordInvestigationResult(
                result=make_result(combination_id, "v1"),
            ),
        )

    validation = make_validation(state, validity="fail", can_answer=False)
    state = apply_investigation_event(
        state,
        ApplyBatchValidation(validation=validation),
    )
    wrong_thread = make_result("combination-1", "repair-1").model_copy(
        update={"codex_thread_id": "thread-other"}
    )
    with pytest.raises(ValueError, match="Codex thread"):
        apply_investigation_event(
            state,
            RecordInvestigationResult(result=wrong_thread),
        )


def test_non_created_session_must_have_a_real_thread_id() -> None:
    with pytest.raises(ValidationError, match="非 created"):
        InvestigationSession(
            task_id="task-1",
            batch_id="batch-1",
            combination_id="combination-1",
            codex_thread_id=None,
            workspace_ref="workspace/combination-1",
            scope_id="scope-combination-1",
            result_version="v1",
            phase="ready_for_batch_validation",
        )


def test_pass_supported_and_answerable_accepts_and_replays_by_full_vector() -> None:
    state = record_all_results(make_state())
    validation = make_validation(state)

    accepted = apply_investigation_event(
        state,
        ApplyBatchValidation(validation=validation),
    )
    replayed = apply_investigation_event(
        accepted,
        ApplyBatchValidation(validation=validation),
    )

    assert accepted.route == "accept"
    assert accepted.final_answer == validation.final_answer
    assert accepted.answerability_explanation is None
    assert accepted.validations == (validation,)
    assert replayed == accepted
    assert validation.idempotency_key == replayed.validations[0].idempotency_key


def test_batch_validation_requires_exact_final_answer_or_explanation_branch() -> None:
    """领域验证只保存对应分支的回答文本，并拒绝路径和凭据。"""

    state = record_all_results(make_state())
    accepted = make_validation(state)
    accepted_payload = accepted.model_dump(exclude_none=True)
    accepted_payload.pop("final_answer")
    with pytest.raises(ValidationError, match="final_answer"):
        BatchValidation.model_validate(accepted_payload)

    contradictory_payload = accepted.model_dump(exclude_none=True)
    contradictory_payload["answerability_explanation"] = "不应同时保存说明。"
    with pytest.raises(ValidationError):
        BatchValidation.model_validate(contradictory_payload)

    unsafe_payload = accepted.model_dump(exclude_none=True)
    unsafe_payload["final_answer"] = "结果见 /mnt/private/result.json"
    with pytest.raises(ValidationError):
        BatchValidation.model_validate(unsafe_payload)

    explained = make_validation(
        state,
        can_answer=False,
        answerability_explanation="当前证据不能覆盖原问题。",
    )
    explained_payload = explained.model_dump(exclude_none=True)
    explained_payload["final_answer"] = "不应出现在 explanation 分支。"
    with pytest.raises(ValidationError, match="final_answer"):
        BatchValidation.model_validate(explained_payload)


def test_domain_rejects_accept_when_any_investigation_still_fails() -> None:
    """绕过模型响应合同构造的矛盾验证也不能进入 accept。"""

    state = record_all_results(make_state())
    invalid_source = make_validation(
        state,
        can_answer=False,
        fail_combination_ids={"combination-5"},
    )
    contradictory = invalid_source.model_copy(
        update={
            "can_answer": True,
            "final_answer": "五项结果支持对原科学问题的回答。",
            "answerability_explanation": None,
        }
    )

    with pytest.raises(ValueError, match="全部为 pass"):
        apply_investigation_event(
            state,
            ApplyBatchValidation(validation=contradictory),
        )


def test_pass_refuted_sets_independent_marker_but_does_not_start_new_batch_alone() -> None:
    state = record_all_results(make_state())
    findings = tuple(
        ValidationFinding(
            combination_id=combination_id,
            validity="pass",
            hypothesis_judgment="refuted",
        )
        for combination_id in COMBINATION_IDS
    )
    validation = make_validation(state, judgment="refuted", can_answer=False)
    validation = validation.model_copy(update={"findings": findings})

    revised = apply_investigation_event(
        state,
        ApplyBatchValidation(validation=validation),
    )

    assert revised.route == "regenerate_hypotheses"
    assert all(
        session.route_marker == "revise_hypothesis"
        for session in revised.sessions
    )
    assert revised.can_generate_next_batch is True


def test_fail_advice_uses_same_session_and_stops_after_three_repairs() -> None:
    state = record_all_results(make_state())

    initial_validation = make_validation(
        state,
        validity="fail",
        can_answer=False,
    )
    state = apply_investigation_event(
        state,
        ApplyBatchValidation(validation=initial_validation),
    )
    assert state.session_for("combination-1").repair_count == 0
    assert state.session_for("combination-1").phase == "repair_pending"

    # 响应已经保存但恢复流程重复应用时，只重放原记录，不提前消耗预算。
    replayed = apply_investigation_event(
        state,
        ApplyBatchValidation(validation=initial_validation),
    )
    assert replayed == state
    conflicting_validation = initial_validation.model_copy(
        update={"review_turn_id": "review-turn-retry"}
    )
    with pytest.raises(ValueError, match="相同 result_version"):
        apply_investigation_event(
            state,
            ApplyBatchValidation(validation=conflicting_validation),
        )
    assert state.session_for("combination-1").repair_count == 0

    for repair_round in range(1, 4):
        for combination_id in COMBINATION_IDS:
            state = apply_investigation_event(
                state,
                RecordInvestigationResult(
                    result=make_result(
                        combination_id,
                        f"repair-{repair_round}-{combination_id}",
                    )
                ),
            )
        session = state.session_for("combination-1")
        assert session.repair_count == repair_round
        # 五个组合都在同一轮返修中恢复各自的原会话。
        assert state.session_for("combination-1").phase == (
            "ready_for_batch_validation"
        )
        if repair_round < 3:
            validation = make_validation(
                state,
                validity="fail",
                can_answer=False,
            )
            state = apply_investigation_event(
                state,
                ApplyBatchValidation(validation=validation),
            )
            assert state.session_for("combination-1").repair_count == repair_round
            assert state.session_for("combination-1").phase == "repair_pending"

    # 第三次返修结果仍然失败，才设置独立的 revise_hypothesis 标识。
    state = apply_investigation_event(
        state,
        ApplyBatchValidation(
            validation=make_validation(
                state,
                validity="fail",
                can_answer=False,
            )
        ),
    )
    session = state.session_for("combination-1")
    assert session.repair_count == 3
    assert session.route_marker == "revise_hypothesis"
    assert session.phase == "marked_for_hypothesis_revision"
    assert state.route == "regenerate_hypotheses"

    # 第四次重放只能返回既有批次验证，不能创建新的返修预算。
    assert apply_investigation_event(
        state,
        ApplyBatchValidation(
            validation=make_validation(
                state,
                validity="fail",
                can_answer=False,
            )
        ),
    ) == state


def test_supported_but_not_answerable_returns_explanation_without_new_route() -> None:
    state = record_all_results(make_state())
    validation = make_validation(
        state,
        can_answer=False,
        answerability_explanation="已有支持性结果，但证据范围不能覆盖原科学问题。",
    )

    explained = apply_investigation_event(
        state,
        ApplyBatchValidation(validation=validation),
    )

    assert explained.route == "answerability_explanation"
    assert explained.answerability_explanation == (
        "已有支持性结果，但证据范围不能覆盖原科学问题。"
    )
    assert "continue_investigation" not in explained.model_dump_json()


def test_all_revise_markers_are_required_to_begin_the_next_batch_and_history_is_kept() -> None:
    state = record_all_results(make_state())
    validation = make_validation(state, judgment="refuted", can_answer=False)
    validation = validation.model_copy(
        update={
            "findings": tuple(
                finding.model_copy(update={"hypothesis_judgment": "refuted"})
                for finding in validation.findings
            )
        }
    )
    state = apply_investigation_event(
        state,
        ApplyBatchValidation(validation=validation),
    )

    next_sessions = tuple(
        make_session(combination_id, batch_id="batch-2")
        for combination_id in COMBINATION_IDS
    )
    next_state = apply_investigation_event(
        state,
        BeginNextBatch(batch_id="batch-2", sessions=next_sessions),
    )

    assert next_state.batch_id == "batch-2"
    assert next_state.results == ()
    assert next_state.validations == ()
    assert next_state.history[0].batch_id == "batch-1"
    assert len(next_state.history[0].results) == 5


@pytest.mark.parametrize(
    ("validity", "judgment", "repair_advice"),
    [
        ("pass", None, None),
        ("pass", "supported", "must not accompany pass"),
        ("fail", "supported", "must not accompany fail"),
        ("fail", None, None),
    ],
)
def test_validation_finding_requires_exact_pass_or_fail_payload(
    validity: str,
    judgment: str | None,
    repair_advice: str | None,
) -> None:
    with pytest.raises(ValidationError):
        ValidationFinding(
            combination_id="combination-1",
            validity=validity,
            hypothesis_judgment=judgment,
            repair_advice=repair_advice,
        )


def test_batch_validation_requires_complete_five_item_version_vector() -> None:
    state = record_all_results(make_state())
    validation = make_validation(state)
    with pytest.raises(ValidationError):
        BatchValidation(
            task_id=validation.task_id,
            batch_id=validation.batch_id,
            combination_ids=validation.combination_ids[:4],
            result_versions=validation.result_versions[:4],
            findings=validation.findings[:4],
            can_answer=validation.can_answer,
            final_answer=validation.final_answer,
            answerability_explanation=validation.answerability_explanation,
            review_thread_id=validation.review_thread_id,
            review_turn_id=validation.review_turn_id,
        )


def test_result_versions_are_scoped_by_combination_and_may_repeat() -> None:
    state = make_state()
    for combination_id in COMBINATION_IDS:
        state = apply_investigation_event(
            state,
            RecordInvestigationResult(
                result=make_result(combination_id, "1")
            ),
        )

    validation = make_validation(state)

    assert validation.result_versions == ("1", "1", "1", "1", "1")
    assert apply_investigation_event(
        state,
        ApplyBatchValidation(validation=validation),
    ).route == "accept"


def test_refuted_result_remains_in_later_full_batch_review() -> None:
    state = record_all_results(make_state())
    first_validation = make_validation(
        state,
        can_answer=False,
        fail_combination_ids={"combination-2"},
    )
    first_validation = first_validation.model_copy(
        update={
            "findings": tuple(
                finding.model_copy(update={"hypothesis_judgment": "refuted"})
                if finding.combination_id == "combination-1"
                else finding
                for finding in first_validation.findings
            )
        }
    )
    state = apply_investigation_event(
        state,
        ApplyBatchValidation(validation=first_validation),
    )
    assert state.session_for("combination-1").route_marker == "revise_hypothesis"

    state = apply_investigation_event(
        state,
        RecordInvestigationResult(
            result=make_result("combination-2", "repair-1")
        ),
    )
    second_validation = make_validation(state)
    reviewed = apply_investigation_event(
        state,
        ApplyBatchValidation(validation=second_validation),
    )

    assert reviewed.route == "accept"
    assert reviewed.session_for("combination-1").route_marker is None


def test_next_batch_accepts_new_combination_identities() -> None:
    state = record_all_results(make_state())
    state = apply_investigation_event(
        state,
        ApplyBatchValidation(
            validation=make_validation(state, judgment="refuted", can_answer=False)
        ),
    )
    next_ids = tuple(f"batch-2-combination-{index}" for index in range(1, 6))
    next_sessions = tuple(
        make_session(combination_id, batch_id="batch-2")
        for combination_id in next_ids
    )

    next_state = apply_investigation_event(
        state,
        BeginNextBatch(batch_id="batch-2", sessions=next_sessions),
    )

    assert tuple(item.combination_id for item in next_state.sessions) == next_ids
