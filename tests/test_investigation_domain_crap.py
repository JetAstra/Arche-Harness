"""调查领域拆分后的字段、快照和路由分支测试。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from chemistry_multiagent.investigation import (
    ApplyBatchValidation,
    BeginNextBatch,
    BatchHistory,
    BatchValidation,
    InvestigationResult,
    InvestigationSession,
    InvestigationState,
    RecordInvestigationResult,
    ValidationFinding,
    apply_investigation_event,
)


COMBINATION_IDS = tuple(f"combination-{index}" for index in range(1, 6))


def _make_session(
    combination_id: str,
    *,
    phase: str = "created",
    batch_id: str = "batch-1",
    result_version: str | None = None,
    repair_count: int = 0,
) -> InvestigationSession:
    return InvestigationSession(
        task_id="task-1",
        batch_id=batch_id,
        combination_id=combination_id,
        codex_thread_id=(
            None if phase == "created" else f"thread-{combination_id}"
        ),
        workspace_ref=f"workspace/{combination_id}",
        scope_id=f"scope-{combination_id}",
        result_version=result_version,
        repair_count=repair_count,
        phase=phase,
    )


def _make_state() -> InvestigationState:
    return InvestigationState(
        task_id="task-1",
        batch_id="batch-1",
        sessions=tuple(_make_session(item) for item in COMBINATION_IDS),
    )


def _make_result(combination_id: str, version: str) -> InvestigationResult:
    return InvestigationResult(
        task_id="task-1",
        batch_id="batch-1",
        combination_id=combination_id,
        result_version=version,
        codex_thread_id=f"thread-{combination_id}",
        scope_id=f"scope-{combination_id}",
        turn_id=f"turn-{combination_id}-{version}",
        self_judgment="supported",
        narrative=f"{combination_id} 的调查记录 {version}。",
        observations=(f"{combination_id} 的实际观测 {version}。",),
    )


def _record_all_results(state: InvestigationState) -> InvestigationState:
    for index, combination_id in enumerate(COMBINATION_IDS, start=1):
        state = apply_investigation_event(
            state,
            RecordInvestigationResult(
                result=_make_result(combination_id, f"v{index}"),
            ),
        )
    return state


def _make_validation(
    state: InvestigationState,
    *,
    validity: str = "pass",
    judgment: str = "supported",
    can_answer: bool = True,
    answerability_explanation: str | None = None,
    fail_ids: set[str] | None = None,
) -> BatchValidation:
    fail_ids = fail_ids or set()
    findings = tuple(
        ValidationFinding(
            combination_id=combination_id,
            validity=(
                "fail"
                if validity == "fail" or combination_id in fail_ids
                else "pass"
            ),
            hypothesis_judgment=(
                None
                if validity == "fail" or combination_id in fail_ids
                else judgment
            ),
            repair_advice=(
                f"请返修 {combination_id}。"
                if validity == "fail" or combination_id in fail_ids
                else None
            ),
        )
        for combination_id in COMBINATION_IDS
    )
    payload: dict[str, object] = {
        "task_id": state.task_id,
        "batch_id": state.batch_id,
        "combination_ids": COMBINATION_IDS,
        "result_versions": tuple(
            state.session_for(combination_id).result_version
            for combination_id in COMBINATION_IDS
        ),
        "findings": findings,
        "can_answer": can_answer,
        "review_thread_id": "review-thread-1",
        "review_turn_id": "review-turn-1",
    }
    if can_answer:
        payload["final_answer"] = "五项结果支持回答原科学问题。"
    elif answerability_explanation is not None:
        payload["answerability_explanation"] = answerability_explanation
    return BatchValidation(**payload)


def _make_history(state: InvestigationState) -> BatchHistory:
    return BatchHistory(
        task_id=state.task_id,
        batch_id=state.batch_id,
        sessions=state.sessions,
        results=state.results,
        validations=state.validations,
        route=state.route,
        final_answer=state.final_answer,
        answerability_explanation=state.answerability_explanation,
    )


def test_answerability_field_presence_rejects_conflicting_explicit_none() -> None:
    state = _record_all_results(_make_state())
    payload = _make_validation(state).model_dump(exclude_none=True)
    payload["answerability_explanation"] = None

    with pytest.raises(ValidationError, match="不能携带回答能力说明"):
        BatchValidation.model_validate(payload)


def test_failed_batch_rejects_answer_field_even_when_value_is_none() -> None:
    state = _record_all_results(_make_state())
    failed = _make_validation(state, validity="fail", can_answer=False)
    payload = failed.model_dump(exclude_none=True)
    payload["final_answer"] = None

    with pytest.raises(ValidationError, match="不能携带 final_answer"):
        BatchValidation.model_validate(payload)


def test_batch_payload_rejects_duplicate_finding_identity() -> None:
    state = _record_all_results(_make_state())
    payload = _make_validation(state).model_dump(exclude_none=True)
    findings = list(payload["findings"])
    findings[-1] = dict(findings[0])
    payload["findings"] = tuple(findings)

    with pytest.raises(ValidationError, match="findings 必须完整覆盖 combination_id"):
        BatchValidation.model_validate(payload)


def test_history_accept_requires_latest_validation_and_matching_answer() -> None:
    state = _record_all_results(_make_state())
    accepted = apply_investigation_event(
        state,
        ApplyBatchValidation(validation=_make_validation(state)),
    )
    payload = _make_history(accepted).model_dump(exclude_none=True)
    payload["validations"] = ()

    with pytest.raises(ValidationError, match="accept 历史必须保留批次验证记录"):
        BatchHistory.model_validate(payload)

    payload["validations"] = accepted.model_dump(exclude_none=True)["validations"]
    payload["final_answer"] = "不同的终局回答。"
    with pytest.raises(ValidationError, match="必须来自最新批次验证"):
        BatchHistory.model_validate(payload)


def test_history_explanation_requires_supported_complete_pass_validation() -> None:
    state = _record_all_results(_make_state())
    explanation = "已有支持性结果，但证据范围不足。"
    explained = apply_investigation_event(
        state,
        ApplyBatchValidation(
            validation=_make_validation(
                state,
                can_answer=False,
                answerability_explanation=explanation,
            )
        ),
    )
    payload = _make_history(explained).model_dump(exclude_none=True)
    payload["answerability_explanation"] = "不匹配的说明。"

    with pytest.raises(ValidationError, match="最新批次验证的 explanation"):
        BatchHistory.model_validate(payload)


def test_non_terminal_history_rejects_answer_fields() -> None:
    state = _make_state()
    payload = _make_history(state).model_dump(exclude_none=True)
    payload["final_answer"] = "不应保存在调查中间态。"

    with pytest.raises(ValidationError, match="非终局历史不能保存回答"):
        BatchHistory.model_validate(payload)


def test_mixed_exhausted_and_repairable_failures_keep_repair_route() -> None:
    sessions = tuple(
        _make_session(
            combination_id,
            phase="ready_for_batch_validation",
            result_version=f"v{index}",
            repair_count=3 if index == 1 else 0,
        )
        for index, combination_id in enumerate(COMBINATION_IDS, start=1)
    )
    state = InvestigationState(
        task_id="task-1",
        batch_id="batch-1",
        sessions=sessions,
        results=tuple(
            _make_result(combination_id, f"v{index}")
            for index, combination_id in enumerate(COMBINATION_IDS, start=1)
        ),
    )
    failed = _make_validation(
        state,
        validity="fail",
        can_answer=False,
    )

    routed = apply_investigation_event(
        state,
        ApplyBatchValidation(validation=failed),
    )

    assert routed.route == "repair_required"
    assert routed.session_for("combination-1").route_marker == "revise_hypothesis"
    assert routed.session_for("combination-1").phase == (
        "marked_for_hypothesis_revision"
    )
    assert routed.session_for("combination-2").phase == "repair_pending"
    assert not routed.can_generate_next_batch


def test_all_five_markers_are_required_for_regeneration_route() -> None:
    state = _record_all_results(_make_state())
    validation = _make_validation(
        state,
        can_answer=False,
        judgment="refuted",
    )

    routed = apply_investigation_event(
        state,
        ApplyBatchValidation(validation=validation),
    )

    assert routed.route == "regenerate_hypotheses"
    assert routed.can_generate_next_batch
    assert {
        session.route_marker for session in routed.sessions
    } == {"revise_hypothesis"}


def test_begin_next_batch_rejects_every_identity_and_fresh_context_violation() -> None:
    """下一批入口拒绝未收束路由、身份漂移和继承旧结果。"""

    state = _record_all_results(_make_state())
    refuted = _make_validation(state, can_answer=False, judgment="refuted")
    closed = apply_investigation_event(state, ApplyBatchValidation(validation=refuted))
    next_sessions = tuple(
        _make_session(item, batch_id="batch-2")
        for item in COMBINATION_IDS
    )

    valid_event = BeginNextBatch(batch_id="batch-2", sessions=next_sessions)
    with pytest.raises(ValueError, match="五个组合"):
        apply_investigation_event(state, valid_event)
    with pytest.raises(ValueError, match="必须改变"):
        apply_investigation_event(
            closed,
            BeginNextBatch(batch_id="batch-1", sessions=next_sessions),
        )
    with pytest.raises(ValueError, match="下一批必须保持冻结组合数量"):
        apply_investigation_event(
            closed,
            BeginNextBatch(batch_id="batch-2", sessions=next_sessions[:4]),
        )

    duplicate_sessions = (*next_sessions[:4], next_sessions[0])
    with pytest.raises(ValueError, match="combination_id"):
        apply_investigation_event(
            closed,
            BeginNextBatch(batch_id="batch-2", sessions=duplicate_sessions),
        )

    task_drift = next_sessions[0].model_copy(update={"task_id": "other-task"})
    with pytest.raises(ValueError, match="task_id/batch_id"):
        apply_investigation_event(
            closed,
            BeginNextBatch(
                batch_id="batch-2",
                sessions=(task_drift, *next_sessions[1:]),
            ),
        )

    batch_drift = next_sessions[0].model_copy(update={"batch_id": "other-batch"})
    with pytest.raises(ValueError, match="task_id/batch_id"):
        apply_investigation_event(
            closed,
            BeginNextBatch(
                batch_id="batch-2",
                sessions=(batch_drift, *next_sessions[1:]),
            ),
        )

    old_result = _make_session(
        COMBINATION_IDS[0],
        batch_id="batch-2",
        phase="ready_for_batch_validation",
        result_version="v1",
    )
    with pytest.raises(ValueError, match="无结果"):
        apply_investigation_event(
            closed,
            BeginNextBatch(
                batch_id="batch-2",
                sessions=(old_result, *next_sessions[1:]),
            ),
        )

    marked = next_sessions[0].model_copy(
        update={
            "phase": "marked_for_hypothesis_revision",
            "codex_thread_id": "thread-1",
            "result_version": "v1",
            "route_marker": "revise_hypothesis",
        }
    )
    with pytest.raises(ValueError, match="无结果"):
        apply_investigation_event(
            closed,
            BeginNextBatch(
                batch_id="batch-2",
                sessions=(marked, *next_sessions[1:]),
            ),
        )


def test_state_answer_fields_are_bound_to_exact_terminal_route() -> None:
    """领域状态的回答文本只能出现在对应终局路由。"""

    from types import SimpleNamespace
    from chemistry_multiagent.investigation.domain import (
        _validate_batch_input,
        _validate_investigation_state_accept_route,
        _validate_investigation_state_answer_fields,
        _validate_investigation_state_explanation_route,
    )

    valid_cases = (
        SimpleNamespace(route="accept", final_answer="回答", answerability_explanation=None),
        SimpleNamespace(route="answerability_explanation", final_answer=None, answerability_explanation="说明"),
        SimpleNamespace(route="investigation", final_answer=None, answerability_explanation=None),
    )
    for state in valid_cases:
        _validate_investigation_state_answer_fields(state)

    invalid_cases = (
        (SimpleNamespace(route="answerability_explanation", final_answer=None, answerability_explanation=None), "必须保存说明"),
        (SimpleNamespace(route="answerability_explanation", final_answer="回答", answerability_explanation="说明"), "不能保存 final_answer"),
        (SimpleNamespace(route="investigation", final_answer=None, answerability_explanation="说明"), "只有回答能力说明路由"),
        (SimpleNamespace(route="accept", final_answer=None, answerability_explanation=None), "accept 路由必须保存"),
        (SimpleNamespace(route="investigation", final_answer="回答", answerability_explanation=None), "final_answer 只能"),
    )
    for state, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            _validate_investigation_state_answer_fields(state)


def test_batch_input_requires_current_complete_vector_and_open_phases() -> None:
    """BatchValidation 只能引用当前五项结果，不能跨任务、跨版本或返修中验证。"""

    from types import SimpleNamespace
    from chemistry_multiagent.investigation.domain import (
        _validate_batch_input,
    )

    state = _record_all_results(_make_state())
    validation = _make_validation(state)
    _validate_batch_input(state, validation)
    invalid_cases = (
        (validation.model_copy(update={"task_id": "other-task"}), "任务批次"),
        (validation.model_copy(update={"combination_ids": ("other", *COMBINATION_IDS[1:])}), "五个组合"),
        (validation.model_copy(update={"result_versions": ("other", *validation.result_versions[1:])}), "向量"),
        (
            SimpleNamespace(
                task_id="task-1",
                batch_id="batch-1",
                sessions=tuple(
                    item.model_copy(update={"result_version": None, "phase": "created", "codex_thread_id": None})
                    if item.combination_id == COMBINATION_IDS[0]
                    else item
                    for item in state.sessions
                ),
            ),
            "InvestigationResult",
        ),
        (
            SimpleNamespace(
                task_id="task-1",
                batch_id="batch-1",
                sessions=tuple(
                    item.model_copy(update={"phase": "repair_pending"})
                    if item.combination_id == COMBINATION_IDS[0]
                    else item
                    for item in state.sessions
                ),
            ),
            "返修",
        ),
    )
    for invalid_state, message in invalid_cases:
        candidate = invalid_state if isinstance(invalid_state, SimpleNamespace) else invalid_state
        if isinstance(candidate, SimpleNamespace):
            with pytest.raises(ValueError, match=message):
                _validate_batch_input(candidate, validation)
        else:
            with pytest.raises(ValueError, match=message):
                _validate_batch_input(state, candidate)


def test_accept_and_explanation_routes_require_latest_matching_validation() -> None:
    """accept 与回答能力说明终局都必须由最新完整 Review 事实证明。"""

    from types import SimpleNamespace
    from chemistry_multiagent.investigation.domain import (
        _validate_investigation_state_accept_route,
        _validate_investigation_state_explanation_route,
    )

    state = _record_all_results(_make_state())
    accepted = apply_investigation_event(
        state,
        ApplyBatchValidation(validation=_make_validation(state)),
    )
    _validate_investigation_state_accept_route(accepted)
    for invalid in (
        SimpleNamespace(validations=(), final_answer="回答"),
        accepted.model_copy(update={"validations": (accepted.validations[0].model_copy(update={"can_answer": False}),)}),
        accepted.model_copy(update={"final_answer": "不同回答"}),
    ):
        with pytest.raises(ValueError):
            _validate_investigation_state_accept_route(invalid)

    explanation = apply_investigation_event(
        state,
        ApplyBatchValidation(
            validation=_make_validation(
                state,
                can_answer=False,
                answerability_explanation="证据范围不足。",
            )
        ),
    )
    _validate_investigation_state_explanation_route(explanation)
    for invalid in (
        SimpleNamespace(validations=(), answerability_explanation="说明"),
        explanation.model_copy(update={"answerability_explanation": "不一致"}),
        explanation.model_copy(update={"validations": (explanation.validations[0].model_copy(update={"can_answer": True}),)}),
    ):
        with pytest.raises(ValueError):
            _validate_investigation_state_explanation_route(invalid)



def test_session_phase_contract_covers_thread_marker_result_and_budget_gates() -> None:
    """会话阶段合同同时约束 thread、marker、结果版本和三次返修上限。"""

    valid_cases = (
        _make_session("combination-1"),
        _make_session(
            "combination-1",
            phase="ready_for_batch_validation",
            result_version="v1",
        ),
        InvestigationSession(
            task_id="task-1",
            batch_id="batch-1",
            combination_id="combination-1",
            codex_thread_id="thread-1",
            workspace_ref="workspace/combination-1",
            scope_id="scope-combination-1",
            result_version="v1",
            repair_count=2,
            phase="repair_pending",
        ),
        InvestigationSession(
            task_id="task-1",
            batch_id="batch-1",
            combination_id="combination-1",
            codex_thread_id="thread-1",
            workspace_ref="workspace/combination-1",
            scope_id="scope-combination-1",
            result_version="v1",
            repair_count=3,
            phase="marked_for_hypothesis_revision",
            route_marker="revise_hypothesis",
        ),
        _make_session(
            "combination-1",
            phase="validated",
            result_version="v1",
        ),
    )
    assert len(valid_cases) == 5

    invalid_payloads = (
        {
            "phase": "marked_for_hypothesis_revision",
            "result_version": "v1",
            "route_marker": None,
        },
        {
            "phase": "ready_for_batch_validation",
            "result_version": "v1",
            "route_marker": "revise_hypothesis",
        },
        {
            "phase": "ready_for_batch_validation",
            "result_version": None,
            "route_marker": None,
        },
        {
            "phase": "repair_pending",
            "result_version": "v1",
            "repair_count": 3,
            "route_marker": None,
        },
        {
            "phase": "created",
            "result_version": "v1",
            "route_marker": None,
        },
    )
    for update in invalid_payloads:
        payload = _make_session("combination-1").model_dump()
        payload.update(update)
        if update["phase"] != "created":
            payload["codex_thread_id"] = "thread-1"
        with pytest.raises(ValidationError):
            InvestigationSession.model_validate(payload)
