"""Investigation Working Memory 假设重生摘要的确定性和安全边界测试。"""

from __future__ import annotations

import copy
import json

import pytest

from chemistry_multiagent.investigation import (
    ApplyBatchValidation,
    BeginNextBatch,
    BatchValidation,
    BatchHistory,
    InvestigationResult,
    InvestigationSession,
    InvestigationState,
    RecordInvestigationResult,
    ValidationFinding,
    apply_investigation_event,
    build_hypothesis_regeneration_memory,
)


COMBINATION_IDS = tuple(f"combination-{index}" for index in range(1, 6))


def make_session(
    combination_id: str,
    batch_id: str,
    *,
    phase: str = "created",
    result_version: str | None = None,
    repair_count: int = 0,
    route_marker: str | None = None,
) -> InvestigationSession:
    """建立一个与测试结果身份匹配的领域会话。"""

    return InvestigationSession(
        task_id="task-1",
        batch_id=batch_id,
        combination_id=combination_id,
        codex_thread_id=(
            None if phase == "created" else f"thread-{batch_id}-{combination_id}"
        ),
        workspace_ref=f"workspace/{batch_id}/{combination_id}",
        scope_id=f"scope-{batch_id}-{combination_id}",
        result_version=result_version,
        repair_count=repair_count,
        phase=phase,
        route_marker=route_marker,
    )


def make_result(
    batch_id: str,
    combination_id: str,
    version: str,
    *,
    judgment: str = "refuted",
    narrative: str | None = None,
    artifact_refs: tuple[str, ...] | None = None,
) -> InvestigationResult:
    """建立带完整科学事实的 InvestigationResult。"""

    return InvestigationResult(
        task_id="task-1",
        batch_id=batch_id,
        combination_id=combination_id,
        result_version=version,
        codex_thread_id=f"thread-{batch_id}-{combination_id}",
        scope_id=f"scope-{batch_id}-{combination_id}",
        turn_id=f"turn-{batch_id}-{version}-{combination_id}",
        self_judgment=judgment,
        narrative=narrative or f"{combination_id} 的实际计算叙述 {version}。",
        job_refs=(f"job-{batch_id}-{version}-{combination_id}",),
        operation_refs=(f"operation-{batch_id}-{version}-{combination_id}",),
        artifact_refs=artifact_refs or (f"artifact-{batch_id}-{version}-{combination_id}",),
        observations=(f"observation-{batch_id}-{version}-{combination_id}",),
        limitations=(f"limitation-{batch_id}-{version}-{combination_id}",),
    )


def make_initial_state(batch_id: str = "batch-1") -> InvestigationState:
    """建立空的五组合 InvestigationState。"""

    return InvestigationState(
        task_id="task-1",
        batch_id=batch_id,
        sessions=tuple(make_session(combination_id, batch_id) for combination_id in COMBINATION_IDS),
    )


def record_results(
    state: InvestigationState,
    version_prefix: str,
    *,
    judgment: str = "refuted",
    result_overrides: dict[str, dict[str, object]] | None = None,
) -> InvestigationState:
    """为当前五个组合登记一轮结果。"""

    result_overrides = result_overrides or {}
    for combination_id in COMBINATION_IDS:
        values = result_overrides.get(combination_id, {})
        state = apply_investigation_event(
            state,
            RecordInvestigationResult(
                result=make_result(
                    state.batch_id,
                    combination_id,
                    f"{version_prefix}-{combination_id}",
                    judgment=str(values.get("judgment", judgment)),
                    narrative=(
                        str(values["narrative"]) if "narrative" in values else None
                    ),
                    artifact_refs=(
                        tuple(values["artifact_refs"])  # type: ignore[arg-type]
                        if "artifact_refs" in values
                        else None
                    ),
                )
            ),
        )
    return state


def make_validation(
    state: InvestigationState,
    *,
    mode: str,
    fail_ids: set[str] | None = None,
    review_turn_id: str = "review-turn-1",
) -> BatchValidation:
    """建立一轮五项验证，保留 refuted 与 fail advice 两种原始原因。"""

    fail_ids = fail_ids or set()
    findings = tuple(
        ValidationFinding(
            combination_id=combination_id,
            validity="fail" if mode == "fail" or combination_id in fail_ids else "pass",
            hypothesis_judgment=(
                "refuted"
                if mode == "refuted" and combination_id not in fail_ids
                else None
            ),
            repair_advice=(
                f"请补足 {combination_id} 的证据。"
                if mode == "fail" or combination_id in fail_ids
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
        can_answer=False,
        **(
            {}
            if mode in {"refuted", "fail"}
            else {"answerability_explanation": "五项 validity 全部通过但仍不能形成完整回答。"}
        ),
        review_thread_id="review-thread-1",
        review_turn_id=review_turn_id,
    )


def apply_validation(state: InvestigationState, validation: BatchValidation) -> InvestigationState:
    """应用一轮批次验证。"""

    return apply_investigation_event(state, ApplyBatchValidation(validation=validation))


def mark_all_refuted(state: InvestigationState, prefix: str = "initial") -> InvestigationState:
    """登记一轮全部被证伪的结果并进入重生终局。"""

    state = record_results(state, prefix)
    return apply_validation(state, make_validation(state, mode="refuted"))


def exhaust_three_repairs(state: InvestigationState) -> InvestigationState:
    """登记初始结果和三轮返修失败，形成完整重生终局。"""

    state = record_results(state, "initial")
    state = apply_validation(state, make_validation(state, mode="fail", review_turn_id="review-turn-1"))
    for round_index in range(1, 4):
        state = record_results(state, f"repair-{round_index}")
        state = apply_validation(
            state,
            make_validation(state, mode="fail", review_turn_id=f"review-turn-{round_index + 1}"),
        )
    return state


def begin_next_batch(state: InvestigationState, batch_id: str) -> InvestigationState:
    """关闭当前重生批次并建立下一批空会话。"""

    sessions = tuple(make_session(combination_id, batch_id) for combination_id in COMBINATION_IDS)
    return apply_investigation_event(
        state,
        BeginNextBatch(batch_id=batch_id, sessions=sessions),
    )


def make_state_with_closed_history() -> InvestigationState:
    """建立一个带完整前批历史且当前批次已进入重生终局的状态。"""

    state = mark_all_refuted(make_initial_state("batch-1"), "initial")
    state = begin_next_batch(state, "batch-2")
    return mark_all_refuted(state, "current")


def replace_only_history(
    state: InvestigationState,
    history: BatchHistory,
) -> InvestigationState:
    """只替换未重新校验的历史快照，交给公开构造入口拒绝损坏输入。"""

    return state.model_copy(update={"history": (history,)})


def test_projection_preserves_multiple_batches_versions_and_validation_rounds() -> None:
    """多批历史保留每一轮实际结果和完整版本向量。"""

    state = exhaust_three_repairs(make_initial_state("batch-1"))
    state = begin_next_batch(state, "batch-2")
    state = mark_all_refuted(state, "initial")
    state = begin_next_batch(state, "batch-3")
    state = exhaust_three_repairs(state)

    payload = build_hypothesis_regeneration_memory(state)

    assert payload["schema"] == "arche.investigation.regeneration_memory"
    assert payload["schema_version"] == 1
    assert payload["task_id"] == "task-1"
    assert payload["batch_id"] == "batch-3"
    assert [item["batch_id"] for item in payload["closed_batches"]] == [
        "batch-1",
        "batch-2",
    ]
    assert len(payload["closed_batches"][0]["results"]) == 20
    assert len(payload["closed_batches"][0]["validations"]) == 4
    assert len(payload["closed_batches"][1]["results"]) == 5
    assert len(payload["current_batch"]["results"]) == 20
    assert [item["repair_count"] for item in payload["current_batch"]["sessions"]] == [3] * 5
    assert all(
        item["route_marker"] == "revise_hypothesis"
        for item in payload["current_batch"]["sessions"]
    )
    latest = payload["current_batch"]["validations"][-1]
    assert latest["result_versions"] == [
        f"repair-3-combination-{index}" for index in range(1, 6)
    ]
    assert all(finding["validity"] == "fail" for finding in latest["findings"])
    assert all(finding["repair_advice"] for finding in latest["findings"])
    assert latest["can_answer"] is False
    assert latest["answerability_explanation"] is None
    assert "review_thread_id" not in latest
    assert "review_turn_id" not in latest


def test_projection_keeps_real_prior_result_review_repair_and_marker_facts() -> None:
    """前批真实结果、Review 向量、返修历史和 marker 均保留且不互相替代。"""

    payload = build_hypothesis_regeneration_memory(make_state_with_closed_history())
    closed = payload["closed_batches"][0]

    assert closed["batch_id"] == "batch-1"
    assert [item["result_version"] for item in closed["sessions"]] == [
        f"initial-combination-{index}" for index in range(1, 6)
    ]
    assert all(
        result["narrative"] == (
            f"combination-{result['combination_id'].split('-')[-1]} 的实际计算叙述 "
            f"initial-{result['combination_id']}。"
        )
        for result in closed["results"]
    )
    assert len(closed["validations"]) == 1
    assert closed["validations"][0]["result_versions"] == [
        f"initial-combination-{index}" for index in range(1, 6)
    ]
    assert all(
        finding["hypothesis_judgment"] == "refuted"
        and finding["repair_advice"] is None
        for finding in closed["validations"][0]["findings"]
    )
    assert all(
        session["route_marker"] == "revise_hypothesis"
        for session in closed["sessions"]
    )


def test_non_state_input_is_rejected_before_any_projection() -> None:
    """公开入口不接受未通过领域校验的任意对象。"""

    with pytest.raises(TypeError, match="InvestigationState"):
        build_hypothesis_regeneration_memory(object())  # type: ignore[arg-type]


def test_missing_result_history_fails_closed() -> None:
    """缺失前批真实结果时不能只凭 marker 生成下一批输入。"""

    state = make_state_with_closed_history()
    broken = state.history[0].model_copy(update={"results": ()})
    with pytest.raises(ValueError, match="缺少 InvestigationResult"):
        build_hypothesis_regeneration_memory(replace_only_history(state, broken))


def test_missing_review_history_fails_closed() -> None:
    """缺失前批 Review 时不能把结果历史误当作完整验证事实。"""

    state = make_state_with_closed_history()
    broken = state.history[0].model_copy(update={"validations": ()})
    with pytest.raises(ValueError, match="缺少完整 BatchValidation"):
        build_hypothesis_regeneration_memory(replace_only_history(state, broken))


def test_history_route_and_task_identity_are_rechecked() -> None:
    """历史路由和任务身份都必须与当前任务事实一致。"""

    state = make_state_with_closed_history()
    for update, message in (
        ({"route": "investigation"}, "不是已关闭"),
        ({"task_id": "task-other"}, "任务或批次身份不匹配"),
    ):
        broken = state.history[0].model_copy(update=update)
        with pytest.raises(ValueError, match=message):
            build_hypothesis_regeneration_memory(replace_only_history(state, broken))


def test_history_session_identity_marker_and_duplicate_are_rechecked() -> None:
    """历史会话必须属于本批次、保留 marker 且不能重复组合。"""

    state = make_state_with_closed_history()
    history = state.history[0]
    bad_identity = history.sessions[0].model_copy(update={"task_id": "task-other"})
    bad_marker = history.sessions[0].model_copy(
        update={"phase": "validated", "route_marker": None}
    )
    duplicate_sessions = history.sessions[:4] + (history.sessions[0],)
    for sessions, message in (
        (tuple([bad_identity, *history.sessions[1:]]), "会话身份不匹配"),
        (tuple([bad_marker, *history.sessions[1:]]), "marker 未全部保存"),
        (duplicate_sessions, "重复 combination_id"),
    ):
        broken = history.model_copy(update={"sessions": sessions})
        with pytest.raises(ValueError, match=message):
            build_hypothesis_regeneration_memory(replace_only_history(state, broken))


def test_history_result_identity_scope_thread_and_combination_are_rechecked() -> None:
    """历史结果必须保留真实任务、组合、thread 和 scope 绑定。"""

    state = make_state_with_closed_history()
    history = state.history[0]
    result = history.results[0]
    mutations = (
        (result.model_copy(update={"task_id": "task-other"}), "Result 身份不匹配"),
        (result.model_copy(update={"combination_id": "combination-other"}), "未知组合"),
        (result.model_copy(update={"codex_thread_id": "thread-other"}), "thread 引用不匹配"),
        (result.model_copy(update={"scope_id": "scope-other"}), "scope 引用不匹配"),
    )
    for replacement, message in mutations:
        broken = history.model_copy(update={"results": (replacement, *history.results[1:])})
        with pytest.raises(ValueError, match=message):
            build_hypothesis_regeneration_memory(replace_only_history(state, broken))


def test_duplicate_or_missing_current_result_versions_fail_closed() -> None:
    """重复结果版本和会话指向缺失版本都必须拒绝。"""

    state = make_state_with_closed_history()
    history = state.history[0]
    duplicate_result = history.model_copy(
        update={"results": (history.results[0], *history.results)}
    )
    broken_session = history.sessions[0].model_copy(
        update={"result_version": "missing-version"}
    )
    missing_current = history.model_copy(
        update={"sessions": (broken_session, *history.sessions[1:])}
    )
    with pytest.raises(ValueError, match="重复 InvestigationResult"):
        build_hypothesis_regeneration_memory(replace_only_history(state, duplicate_result))
    with pytest.raises(ValueError, match="当前版本没有对应结果"):
        build_hypothesis_regeneration_memory(replace_only_history(state, missing_current))


def test_review_identity_reference_and_duplicate_version_vector_fail_closed() -> None:
    """Review 必须引用当前批次结果，且同一版本向量不能重复保存。"""

    state = make_state_with_closed_history()
    history = state.history[0]
    validation = history.validations[0]
    mutations = (
        (
            validation.model_copy(update={"task_id": "task-other"}),
            "任务或批次身份不匹配",
        ),
        (
            validation.model_copy(
                update={"result_versions": ("missing-version", *validation.result_versions[1:])}
            ),
            "不存在的 InvestigationResult 版本",
        ),
    )
    for replacement, message in mutations:
        broken = history.model_copy(update={"validations": (replacement,)})
        with pytest.raises(ValueError, match=message):
            build_hypothesis_regeneration_memory(replace_only_history(state, broken))

    duplicate = history.model_copy(update={"validations": (validation, validation)})
    with pytest.raises(ValueError, match="重复批次验证版本向量"):
        build_hypothesis_regeneration_memory(replace_only_history(state, duplicate))


def test_latest_review_vector_must_match_current_versions() -> None:
    """最新 Review 的版本向量漂移时，即使引用旧结果也必须拒绝。"""

    state = exhaust_three_repairs(make_initial_state("batch-1"))
    state = begin_next_batch(state, "batch-2")
    state = mark_all_refuted(state, "current")
    history = state.history[0]
    previous_validation = history.validations[-2]
    latest = history.validations[-1].model_copy(
        update={
            "result_versions": (
                previous_validation.result_versions[0],
                *history.validations[-1].result_versions[1:],
            )
        }
    )
    broken = history.model_copy(
        update={"validations": (*history.validations[:-1], latest)}
    )
    with pytest.raises(ValueError, match="最新验证没有绑定"):
        build_hypothesis_regeneration_memory(replace_only_history(state, broken))


def test_marker_cannot_replace_scientific_reason_in_review() -> None:
    """marker 与 Review 原因矛盾时，不能把支持性结果伪装成重生原因。"""

    state = make_state_with_closed_history()
    history = state.history[0]
    finding = history.validations[0].findings[0].model_copy(
        update={
            "validity": "fail",
            "hypothesis_judgment": None,
            "repair_advice": "需要补充证据。",
        }
    )
    validation = history.validations[0].model_copy(
        update={"findings": (finding, *history.validations[0].findings[1:])}
    )
    broken = history.model_copy(update={"validations": (validation,)})
    with pytest.raises(ValueError, match="marker 不一致"):
        build_hypothesis_regeneration_memory(replace_only_history(state, broken))


def test_refuted_and_repair_exhausted_reasons_remain_validation_facts() -> None:
    """相同 marker 下 refuted 与返修耗尽必须由 validation 字段区分。"""

    # 两类状态的 marker 相同，历史摘要必须保留各自 validation 的原始区分。
    refuted_state = mark_all_refuted(make_initial_state(), "refuted")
    exhausted_state = exhaust_three_repairs(make_initial_state("batch-2"))
    refuted_payload = build_hypothesis_regeneration_memory(refuted_state)
    exhausted_payload = build_hypothesis_regeneration_memory(exhausted_state)

    refuted_findings = refuted_payload["current_batch"]["validations"][-1]["findings"]
    exhausted_findings = exhausted_payload["current_batch"]["validations"][-1]["findings"]
    assert refuted_payload["current_batch"]["validations"][-1]["final_answer"] is None
    assert exhausted_payload["current_batch"]["validations"][-1]["final_answer"] is None
    assert all(item["hypothesis_judgment"] == "refuted" for item in refuted_findings)
    assert all(item["validity"] == "fail" for item in exhausted_findings)
    assert all(item["hypothesis_judgment"] is None for item in exhausted_findings)
    assert all(item["repair_advice"] for item in exhausted_findings)
    assert all(
        session["route_marker"] == "revise_hypothesis"
        for session in refuted_payload["current_batch"]["sessions"]
    )
    assert all(
        session["route_marker"] == "revise_hypothesis"
        for session in exhausted_payload["current_batch"]["sessions"]
    )
    assert "reason" not in json.dumps(refuted_payload, ensure_ascii=False)
    assert "reason" not in json.dumps(exhausted_payload, ensure_ascii=False)


def test_projection_is_deterministic_and_does_not_mutate_state() -> None:
    """输出可变副本，重复调用结果稳定且不会改变冻结领域状态。"""

    state = mark_all_refuted(make_initial_state())
    before = state.model_dump(mode="json")
    first = build_hypothesis_regeneration_memory(state)
    second = build_hypothesis_regeneration_memory(state)
    assert first == second
    json.dumps(first, ensure_ascii=False, allow_nan=False)

    changed = copy.deepcopy(first)
    changed["current_batch"]["results"][0]["observations"][0] = "调用方修改"
    assert build_hypothesis_regeneration_memory(state) == first
    assert state.model_dump(mode="json") == before


@pytest.mark.parametrize("route", ["investigation", "repair_required", "accept"])
def test_non_terminal_routes_are_rejected(route: str) -> None:
    """非重生终局不能为下一批模型生成部分记忆。"""

    state = mark_all_refuted(make_initial_state())
    invalid = state.model_copy(update={"route": route})
    with pytest.raises(ValueError, match="终局|regenerate_hypotheses|领域合同"):
        build_hypothesis_regeneration_memory(invalid)


@pytest.mark.parametrize(
    "unsafe",
    [
        ("narrative", "/mnt/host/private/input.log"),
        ("narrative", "api_key=top-secret-value"),
        ("artifact_refs", ("sk-1234567890",)),
    ],
)
def test_paths_and_secrets_fail_closed(unsafe: tuple[str, object]) -> None:
    """领域允许的宽松字符串在模型历史摘要中仍不能带出路径和凭据。"""

    field_name, unsafe_value = unsafe
    overrides = {"combination-1": {field_name: unsafe_value}}
    state = record_results(make_initial_state(), "initial", result_overrides=overrides)
    state = apply_validation(state, make_validation(state, mode="refuted"))
    with pytest.raises(ValueError, match="路径|URI|凭据"):
        build_hypothesis_regeneration_memory(state)


def test_workspace_scope_and_codex_identity_are_not_projected() -> None:
    """工作区、scope 和 Codex thread 只用于内部核对，不会进入 JSON。"""

    state = mark_all_refuted(make_initial_state())
    payload = build_hypothesis_regeneration_memory(state)
    encoded = json.dumps(payload, ensure_ascii=False)
    assert "workspace_ref" not in encoded
    assert "scope_id" not in encoded
    assert "codex_thread_id" not in encoded
    assert "transcript" not in encoded
