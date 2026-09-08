"""Investigation Codex 响应到领域事实适配器的确定性测试。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from chemistry_multiagent.investigation import (
    BatchValidation,
    InvestigationResult,
    InvestigationSession,
    InvestigationState,
    InvestigationTurnResult,
    adapt_investigation_turn_to_result,
    adapt_review_turn_to_batch_validation,
)
from chemistry_multiagent.investigation.contracts import (
    FailReviewFinding,
    InvestigationResponse,
    PassReviewFinding,
    TaskReviewResponse,
)


COMBINATION_IDS = tuple(f"combination-{index}" for index in range(1, 6))


def make_investigation_response(
    combination_id: str = "combination-1",
) -> InvestigationResponse:
    """构造包含所有可追溯字段的调查响应。"""

    return InvestigationResponse(
        self_judgment="supported",
        narrative=f"{combination_id} 的实际计算回顾。",
        job_refs=(f"job-{combination_id}",),
        operation_refs=(f"operation-{combination_id}",),
        artifact_refs=(f"artifact-{combination_id}",),
        observations=(f"observation-{combination_id}",),
        limitations=(f"limitation-{combination_id}",),
    )


def make_investigation_session(
    combination_id: str = "combination-1",
    *,
    codex_thread_id: str | None = None,
    result_version: str | None = None,
    phase: str = "created",
) -> InvestigationSession:
    """构造一个调查会话。"""

    return InvestigationSession(
        task_id="task-1",
        batch_id="batch-1",
        combination_id=combination_id,
        codex_thread_id=codex_thread_id,
        workspace_ref=f"workspace-{combination_id}",
        scope_id=f"scope-{combination_id}",
        result_version=result_version,
        phase=phase,
    )


def make_investigation_result(
    combination_id: str,
    result_version: str,
) -> InvestigationResult:
    """构造当前五项状态所需的已保存结果。"""

    return InvestigationResult(
        task_id="task-1",
        batch_id="batch-1",
        combination_id=combination_id,
        result_version=result_version,
        codex_thread_id=f"thread-{combination_id}",
        scope_id=f"scope-{combination_id}",
        turn_id=f"turn-{combination_id}-{result_version}",
        self_judgment="supported",
        narrative=f"{combination_id} 的结果。",
        job_refs=(f"job-{combination_id}",),
        operation_refs=(f"operation-{combination_id}",),
        artifact_refs=(f"artifact-{combination_id}",),
        observations=(f"observation-{combination_id}",),
        limitations=(),
    )


def make_review_state() -> InvestigationState:
    """构造带完整当前版本向量的五项状态。"""

    sessions = tuple(
        make_investigation_session(
            combination_id,
            codex_thread_id=f"thread-{combination_id}",
            result_version=f"v{index}",
            phase="ready_for_batch_validation",
        )
        for index, combination_id in enumerate(COMBINATION_IDS, start=1)
    )
    return InvestigationState(
        task_id="task-1",
        batch_id="batch-1",
        sessions=sessions,
        results=tuple(
            make_investigation_result(combination_id, f"v{index}")
            for index, combination_id in enumerate(COMBINATION_IDS, start=1)
        ),
    )


def make_review_turn_result() -> InvestigationTurnResult:
    """构造顺序打乱且同时包含 pass/fail 的 Review turn。"""

    response = TaskReviewResponse(
        findings=(
            FailReviewFinding(
                hypothesis_execution_input_record_id="combination-5",
                validity="fail",
                repair_advice="补足第五项计算证据。",
            ),
            PassReviewFinding(
                hypothesis_execution_input_record_id="combination-2",
                validity="pass",
                hypothesis_judgment="refuted",
            ),
            PassReviewFinding(
                hypothesis_execution_input_record_id="combination-1",
                validity="pass",
                hypothesis_judgment="supported",
            ),
            PassReviewFinding(
                hypothesis_execution_input_record_id="combination-4",
                validity="pass",
                hypothesis_judgment="supported",
            ),
            PassReviewFinding(
                hypothesis_execution_input_record_id="combination-3",
                validity="pass",
                hypothesis_judgment="refuted",
            ),
        ),
        can_answer_original_question=False,
    )
    return InvestigationTurnResult(
        result=response,
        thread_id="review-thread-1",
        turn_id="review-turn-1",
    )


def test_adapt_investigation_turn_binds_session_runtime_identity_and_preserves_facts() -> None:
    """首次调查把 session 与 runtime 身份合并，不丢失科学字段。"""

    session = make_investigation_session()
    response = make_investigation_response()
    turn = InvestigationTurnResult(
        result=response,
        thread_id="thread-combination-1",
        turn_id="turn-1",
    )

    converted = adapt_investigation_turn_to_result(session, "v1", turn)

    assert converted.task_id == session.task_id
    assert converted.batch_id == session.batch_id
    assert converted.combination_id == session.combination_id
    assert converted.scope_id == session.scope_id
    assert converted.codex_thread_id == turn.thread_id
    assert converted.turn_id == turn.turn_id
    assert converted.result_version == "v1"
    assert converted.job_refs == response.job_refs
    assert converted.operation_refs == response.operation_refs
    assert converted.artifact_refs == response.artifact_refs
    assert converted.observations == response.observations
    assert converted.limitations == response.limitations


def test_adapt_investigation_turn_reuses_bound_thread_and_rejects_drift_or_duplicate_version() -> None:
    """返修必须继续原 thread，且不能复用当前版本。"""

    session = make_investigation_session(
        codex_thread_id="thread-combination-1",
        result_version="v1",
        phase="repair_pending",
    )
    matching_turn = InvestigationTurnResult(
        result=make_investigation_response(),
        thread_id="thread-combination-1",
        turn_id="turn-repair-1",
    )
    converted = adapt_investigation_turn_to_result(session, "v2", matching_turn)
    assert converted.result_version == "v2"

    with pytest.raises(ValueError, match="thread"):
        adapt_investigation_turn_to_result(
            session,
            "v3",
            InvestigationTurnResult(
                result=make_investigation_response(),
                thread_id="thread-other",
                turn_id="turn-repair-2",
            ),
        )
    with pytest.raises(ValueError, match="result_version"):
        adapt_investigation_turn_to_result(session, "v1", matching_turn)
    with pytest.raises(ValueError, match="不能为空"):
        adapt_investigation_turn_to_result(session, "   ", matching_turn)


def test_adapt_investigation_turn_uses_session_scope_without_accepting_scope_override() -> None:
    """scope 只来自 session，适配器接口不允许调用方再传 scope。"""

    session = make_investigation_session()
    turn = InvestigationTurnResult(
        result=make_investigation_response(),
        thread_id="thread-combination-1",
        turn_id="turn-1",
    )

    converted = adapt_investigation_turn_to_result(session, "v1", turn)

    assert converted.scope_id == "scope-combination-1"


def test_adapt_review_turn_orders_vector_and_findings_by_state_sessions() -> None:
    """Review 的输入顺序可变，但输出始终遵循 state.sessions 顺序。"""

    state = make_review_state()
    shuffled_vector = (
        ("combination-5", "v5"),
        ("combination-2", "v2"),
        ("combination-1", "v1"),
        ("combination-4", "v4"),
        ("combination-3", "v3"),
    )

    converted = adapt_review_turn_to_batch_validation(
        state,
        shuffled_vector,
        make_review_turn_result(),
    )

    assert converted.task_id == state.task_id
    assert converted.batch_id == state.batch_id
    assert converted.combination_ids == COMBINATION_IDS
    assert converted.result_versions == ("v1", "v2", "v3", "v4", "v5")
    assert tuple(item.combination_id for item in converted.findings) == COMBINATION_IDS
    assert [(item.validity, item.hypothesis_judgment, item.repair_advice) for item in converted.findings] == [
        ("pass", "supported", None),
        ("pass", "refuted", None),
        ("pass", "refuted", None),
        ("pass", "supported", None),
        ("fail", None, "补足第五项计算证据。"),
    ]
    assert converted.review_thread_id == "review-thread-1"
    assert converted.review_turn_id == "review-turn-1"
    assert converted.can_answer is False
    assert converted.final_answer is None
    assert converted.answerability_explanation is None


def test_adapt_review_turn_preserves_final_answer_on_accept_branch() -> None:
    """Review 适配器必须把最终回答保存到 BatchValidation。"""

    answer = "五项结果支持对原科学问题的回答。"
    response = TaskReviewResponse(
        findings=tuple(
            PassReviewFinding(
                hypothesis_execution_input_record_id=combination_id,
                validity="pass",
                hypothesis_judgment="supported",
            )
            for combination_id in COMBINATION_IDS
        ),
        can_answer_original_question=True,
        final_answer=answer,
    )
    converted = adapt_review_turn_to_batch_validation(
        make_review_state(),
        tuple((f"combination-{index}", f"v{index}") for index in range(1, 6)),
        InvestigationTurnResult(
            result=response,
            thread_id="review-thread-1",
            turn_id="review-turn-accept",
        ),
    )
    assert converted.can_answer is True
    assert converted.final_answer == answer
    assert converted.answerability_explanation is None


def test_adapt_review_turn_allows_same_version_text_for_independent_combinations() -> None:
    """五个组合各自分配版本时可以合法共享同一个版本文本。"""

    original = make_review_state()
    state = InvestigationState(
        task_id=original.task_id,
        batch_id=original.batch_id,
        sessions=tuple(
            session.model_copy(update={"result_version": "1"})
            for session in original.sessions
        ),
        results=tuple(
            result.model_copy(update={"result_version": "1"})
            for result in original.results
        ),
    )
    vector = tuple((combination_id, "1") for combination_id in reversed(COMBINATION_IDS))

    converted = adapt_review_turn_to_batch_validation(
        state,
        vector,
        make_review_turn_result(),
    )

    assert converted.combination_ids == COMBINATION_IDS
    assert converted.result_versions == ("1", "1", "1", "1", "1")


def test_adapt_review_turn_rejects_task_review_thread_drift() -> None:
    """同一任务已有 Review 后不能绑定另一条 Review thread。"""

    state = make_review_state()
    first = adapt_review_turn_to_batch_validation(
        state,
        tuple((f"combination-{index}", f"v{index}") for index in range(1, 6)),
        make_review_turn_result(),
    )
    state_with_review = InvestigationState(
        task_id=state.task_id,
        batch_id=state.batch_id,
        sessions=state.sessions,
        results=state.results,
        validations=(first,),
    )
    drifted_turn = InvestigationTurnResult(
        result=make_review_turn_result().result,
        thread_id="review-thread-other",
        turn_id="review-turn-2",
    )

    with pytest.raises(ValueError, match="Review thread"):
        adapt_review_turn_to_batch_validation(
            state_with_review,
            tuple((f"combination-{index}", f"v{index}") for index in range(1, 6)),
            drifted_turn,
        )


@pytest.mark.parametrize(
    "vector",
    [
        (("combination-1", "v1"),) * 4,
        (
            ("combination-1", "v1"),
            ("combination-2", "v2"),
            ("combination-3", "v3"),
            ("combination-4", "v4"),
            ("unknown", "v5"),
        ),
        (
            ("combination-1", "wrong"),
            ("combination-2", "v2"),
            ("combination-3", "v3"),
            ("combination-4", "v4"),
            ("combination-5", "v5"),
        ),
    ],
)
def test_adapt_review_turn_rejects_incomplete_or_drifted_version_vector(
    vector: tuple[tuple[str, str], ...],
) -> None:
    """Review 只能绑定当前状态完整且一致的五项版本向量。"""

    with pytest.raises(ValueError, match="vector|向量|覆盖|current"):
        adapt_review_turn_to_batch_validation(
            make_review_state(),
            vector,
            make_review_turn_result(),
        )


def test_adapters_reject_wrong_turn_contracts_and_keep_domain_results_frozen() -> None:
    """两个方向不能交叉使用，生成的领域记录保持不可变。"""

    session = make_investigation_session()
    review_turn = make_review_turn_result()
    with pytest.raises(TypeError, match="InvestigationResponse"):
        adapt_investigation_turn_to_result(session, "v1", review_turn)

    state = make_review_state()
    investigation_turn = InvestigationTurnResult(
        result=make_investigation_response(),
        thread_id="review-thread-1",
        turn_id="turn-1",
    )
    with pytest.raises(TypeError, match="TaskReviewResponse"):
        adapt_review_turn_to_batch_validation(
            state,
            tuple((f"combination-{index}", f"v{index}") for index in range(1, 6)),
            investigation_turn,
        )

    converted = adapt_investigation_turn_to_result(
        session,
        "v1",
        InvestigationTurnResult(
            result=make_investigation_response(),
            thread_id="thread-combination-1",
            turn_id="turn-1",
        ),
    )
    with pytest.raises(ValidationError):
        converted.scope_id = "other-scope"  # type: ignore[misc]


def test_review_conversion_output_is_frozen() -> None:
    """BatchValidation 及其 finding 均为不可变领域事实。"""

    converted = adapt_review_turn_to_batch_validation(
        make_review_state(),
        tuple((f"combination-{index}", f"v{index}") for index in range(1, 6)),
        make_review_turn_result(),
    )

    assert isinstance(converted, BatchValidation)
    with pytest.raises(ValidationError):
        converted.review_turn_id = "other-turn"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        converted.findings[0].validity = "fail"  # type: ignore[misc]



def test_adapt_review_turn_rejects_corrupt_state_and_finding_coverage() -> None:
    """适配器对损坏状态、未知/重复/缺失 finding 和说明分支保持 fail-closed。"""

    import copy

    vector = tuple((f"combination-{index}", f"v{index}") for index in range(1, 6))
    turn = make_review_turn_result()
    with pytest.raises(TypeError, match="InvestigationState"):
        adapt_review_turn_to_batch_validation(None, vector, turn)  # type: ignore[arg-type]

    broken_state = copy.deepcopy(make_review_state())
    object.__setattr__(broken_state.sessions[0], "result_version", None)
    with pytest.raises(ValueError, match="必须已有 result_version"):
        adapt_review_turn_to_batch_validation(broken_state, vector, turn)

    base_findings = tuple(
        PassReviewFinding(
            hypothesis_execution_input_record_id=combination_id,
            validity="pass",
            hypothesis_judgment="supported",
        )
        for combination_id in COMBINATION_IDS
    )
    unknown_response = TaskReviewResponse.model_construct(
        findings=(*base_findings[:4], PassReviewFinding(
            hypothesis_execution_input_record_id="unknown",
            validity="pass",
            hypothesis_judgment="supported",
        )),
        can_answer_original_question=False,
        final_answer=None,
        answerability_explanation=None,
    )
    with pytest.raises(ValueError, match="完整覆盖"):
        adapt_review_turn_to_batch_validation(
            make_review_state(),
            vector,
            InvestigationTurnResult(
                result=unknown_response,
                thread_id="review-thread-1",
                turn_id="review-unknown",
            ),
        )

    duplicate_response = TaskReviewResponse.model_construct(
        findings=(*base_findings[:4], base_findings[0]),
        can_answer_original_question=False,
        final_answer=None,
        answerability_explanation=None,
    )
    with pytest.raises(ValueError, match="不能重复"):
        adapt_review_turn_to_batch_validation(
            make_review_state(),
            vector,
            InvestigationTurnResult(
                result=duplicate_response,
                thread_id="review-thread-1",
                turn_id="review-duplicate",
            ),
        )

    missing_response = TaskReviewResponse.model_construct(
        findings=base_findings[:4],
        can_answer_original_question=False,
        final_answer=None,
        answerability_explanation=None,
    )
    with pytest.raises(ValueError, match="完整覆盖"):
        adapt_review_turn_to_batch_validation(
            make_review_state(),
            vector,
            InvestigationTurnResult(
                result=missing_response,
                thread_id="review-thread-1",
                turn_id="review-missing",
            ),
        )

    explained_response = TaskReviewResponse.model_construct(
        findings=base_findings,
        can_answer_original_question=False,
        final_answer=None,
        answerability_explanation="五项结果支持但不足以回答原问题。",
    )
    converted = adapt_review_turn_to_batch_validation(
        make_review_state(),
        vector,
        InvestigationTurnResult(
            result=explained_response,
            thread_id="review-thread-1",
            turn_id="review-explanation",
        ),
    )
    assert converted.answerability_explanation == "五项结果支持但不足以回答原问题。"
