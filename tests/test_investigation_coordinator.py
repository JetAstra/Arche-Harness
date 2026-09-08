"""Investigation and Experiment 批次协调器的确定性回归测试。"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

import chemistry_multiagent.investigation.coordinator as coordinator_module
from chemistry_multiagent.investigation.coordinator import (
    run_investigation_and_experiment_batch,
)
from chemistry_multiagent.investigation.domain import ApplyBatchValidation
from chemistry_multiagent.investigation import (
    BatchValidation,
    InvestigationCoordinatorConflict,
    InvestigationResult,
    InvestigationRevisionConflict,
    InvestigationSession,
    InvestigationSQLiteStore,
    InvestigationState,
    ValidationFinding,
    RecordInvestigationResult,
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
        turn_id=f"turn-{combination_id}-{version}",
        self_judgment=judgment,
        narrative=f"{combination_id} 的计算结果 {version}。",
        job_refs=(f"job-{combination_id}-{version}",),
        operation_refs=(f"operation-{combination_id}-{version}",),
        artifact_refs=(f"artifact-{combination_id}-{version}",),
        observations=(f"observation-{combination_id}-{version}",),
    )


def make_validation(
    state: InvestigationState,
    *,
    validity: str = "pass",
    judgment: str = "supported",
    can_answer: bool = True,
    final_answer: str | None = None,
    explanation: str | None = None,
    fail_ids: set[str] | None = None,
) -> BatchValidation:
    fail_ids = fail_ids or set()
    findings = tuple(
        ValidationFinding(
            combination_id=combination_id,
            validity="fail" if validity == "fail" or combination_id in fail_ids else "pass",
            hypothesis_judgment=(
                None
                if validity == "fail" or combination_id in fail_ids
                else judgment
            ),
            repair_advice=(
                f"请补充 {combination_id} 的可核对计算证据。"
                if validity == "fail" or combination_id in fail_ids
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
                {"answerability_explanation": explanation}
                if explanation is not None
                else {}
            )
        ),
        review_thread_id="review-thread-1",
        review_turn_id=f"review-turn-{len(state.validations) + 1}",
    )


def test_parallel_results_cross_barrier_but_are_saved_in_session_order(
    tmp_path: Path,
) -> None:
    store = InvestigationSQLiteStore(tmp_path / "state.sqlite3")
    completion_order: list[str] = []
    lock = threading.Lock()

    def investigate(session: InvestigationSession) -> InvestigationResult:
        # 后面的组合先完成，验证协调器不会按 future 完成顺序追加结果。
        time.sleep((5 - int(session.combination_id.rsplit("-", 1)[1])) * 0.005)
        with lock:
            completion_order.append(session.combination_id)
        return make_result(session.combination_id, "initial-1")

    observed: list[tuple[tuple[str, ...], tuple[tuple[str, str], ...]]] = []

    def review(
        state: InvestigationState,
        results: tuple[InvestigationResult, ...],
        vector: tuple[tuple[str, str], ...],
    ) -> BatchValidation:
        observed.append(
            (tuple(result.combination_id for result in results), vector)
        )
        return make_validation(state)

    outcome = run_investigation_and_experiment_batch(
        store,
        make_state(),
        investigate,
        lambda session, advice: make_result(session.combination_id, "repair-1"),
        review,
    )

    assert completion_order != list(COMBINATION_IDS)
    assert tuple(result.combination_id for result in outcome.state.results) == COMBINATION_IDS
    assert observed == [
        (
            COMBINATION_IDS,
            tuple((combination_id, "initial-1") for combination_id in COMBINATION_IDS),
        )
    ]
    assert outcome.route == "accept"
    assert outcome.state.final_answer == "五项结果支持对原科学问题的回答。"
    assert outcome.review_called is True
    assert outcome.initial_investigator_calls == 5
    assert outcome.revision == 7


def test_initial_investigation_binds_thread_created_by_first_result(tmp_path: Path) -> None:
    store = InvestigationSQLiteStore(tmp_path / "state.sqlite3")
    unbound_sessions = tuple(
        InvestigationSession(
            task_id="task-1",
            batch_id="batch-1",
            combination_id=combination_id,
            codex_thread_id=None,
            workspace_ref=f"workspace/{combination_id}",
            scope_id=f"scope-{combination_id}",
        )
        for combination_id in COMBINATION_IDS
    )
    state = InvestigationState(
        task_id="task-1",
        batch_id="batch-1",
        sessions=unbound_sessions,
    )
    received_thread_ids: list[str | None] = []

    def investigate(session: InvestigationSession) -> InvestigationResult:
        received_thread_ids.append(session.codex_thread_id)
        return make_result(session.combination_id, "initial-1")

    outcome = run_investigation_and_experiment_batch(
        store,
        state,
        investigate,
        lambda session, advice: make_result(session.combination_id, "repair-1"),
        lambda state, results, vector: make_validation(state),
    )
    assert received_thread_ids == [None] * 5
    assert all(session.codex_thread_id == f"thread-{session.combination_id}" for session in outcome.state.sessions)


def test_partial_saved_results_are_recovered_without_recalling_ready_sessions(
    tmp_path: Path,
) -> None:
    store = InvestigationSQLiteStore(tmp_path / "state.sqlite3")
    fail_once = {"enabled": True}
    calls: list[str] = []

    def investigate(session: InvestigationSession) -> InvestigationResult:
        calls.append(session.combination_id)
        if session.combination_id == "combination-3" and fail_once["enabled"]:
            raise RuntimeError("调查器暂时失败")
        return make_result(session.combination_id, "initial-1")

    with pytest.raises(RuntimeError, match="调查器暂时失败"):
        run_investigation_and_experiment_batch(
            store,
            make_state(),
            investigate,
            lambda session, advice: make_result(session.combination_id, "repair-1"),
            lambda state, results, vector: make_validation(state),
        )

    partial = store.load_investigation_snapshot("task-1")
    assert partial is not None
    assert tuple(result.combination_id for result in partial.state.results) == (
        "combination-1",
        "combination-2",
        "combination-4",
        "combination-5",
    )

    fail_once["enabled"] = False
    calls.clear()
    outcome = run_investigation_and_experiment_batch(
        store,
        make_state(),
        investigate,
        lambda session, advice: make_result(session.combination_id, "repair-1"),
        lambda state, results, vector: make_validation(state),
    )
    assert outcome.route == "accept"
    assert calls == ["combination-3"]


def test_terminal_validation_replay_does_not_call_reviewer_again(tmp_path: Path) -> None:
    store = InvestigationSQLiteStore(tmp_path / "state.sqlite3")

    def investigate(session: InvestigationSession) -> InvestigationResult:
        return make_result(session.combination_id, "initial-1")

    first = run_investigation_and_experiment_batch(
        store,
        make_state(),
        investigate,
        lambda session, advice: make_result(session.combination_id, "repair-1"),
        lambda state, results, vector: make_validation(state),
    )

    def reviewer_must_not_run(*args: object) -> BatchValidation:
        raise AssertionError("终局快照恢复不应再次调用 reviewer")

    replay = run_investigation_and_experiment_batch(
        store,
        make_state(),
        reviewer_must_not_run,
        reviewer_must_not_run,  # type: ignore[arg-type]
        reviewer_must_not_run,  # type: ignore[arg-type]
    )
    assert replay.state == first.state
    assert replay.revision == first.revision
    assert replay.review_called is False
    assert replay.review_calls == 0
    assert replay.initial_investigator_calls == 0


class _ValidationReplayStore(InvestigationSQLiteStore):
    """让已保存的完整向量验证命中读取接口而不调用 Review。"""

    def __init__(self, path: Path, validation: BatchValidation) -> None:
        super().__init__(path)
        self.validation = validation
        self.read_count = 0

    def read_saved_batch_validation(self, task_id: str, batch_id: str, vector):
        self.read_count += 1
        return self.validation


def test_saved_validation_vector_is_applied_without_reviewer_call(tmp_path: Path) -> None:
    state = make_state()

    # 先建立完整结果快照，但故意不应用 Review，模拟 Review 前崩溃恢复。
    for combination_id in COMBINATION_IDS:
        state = apply_investigation_event(
            state,
            RecordInvestigationResult(
                result=make_result(combination_id, "initial-1"),
            ),
        )
    validation = make_validation(state)
    store = _ValidationReplayStore(tmp_path / "state.sqlite3", validation)
    store.save_investigation_state(state)

    def reviewer_must_not_run(*args: object) -> BatchValidation:
        raise AssertionError("完整版本向量已命中验证重放")

    outcome = run_investigation_and_experiment_batch(
        store,
        make_state(),
        reviewer_must_not_run,  # type: ignore[arg-type]
        reviewer_must_not_run,  # type: ignore[arg-type]
        reviewer_must_not_run,  # type: ignore[arg-type]
    )
    assert outcome.route == "accept"
    assert outcome.review_called is False
    assert outcome.review_calls == 0
    assert outcome.replay_count == 1
    assert store.read_count == 1


def test_repairs_use_original_session_and_re_review_full_vector_three_times(
    tmp_path: Path,
) -> None:
    store = InvestigationSQLiteStore(tmp_path / "state.sqlite3")
    repair_calls: list[tuple[str, str, str, str]] = []
    reviewed_vectors: list[tuple[tuple[str, str], ...]] = []

    def investigate(session: InvestigationSession) -> InvestigationResult:
        return make_result(session.combination_id, "initial-1")

    def repair(session: InvestigationSession, advice: str) -> InvestigationResult:
        repair_calls.append(
            (session.combination_id, session.codex_thread_id, session.scope_id, advice)
        )
        version = f"repair-{session.repair_count + 1}"
        return make_result(session.combination_id, version)

    def review(
        state: InvestigationState,
        results: tuple[InvestigationResult, ...],
        vector: tuple[tuple[str, str], ...],
    ) -> BatchValidation:
        assert tuple(result.combination_id for result in results) == COMBINATION_IDS
        assert tuple((result.combination_id, result.result_version) for result in results) == vector
        reviewed_vectors.append(vector)
        return make_validation(
            state,
            validity="fail",
            can_answer=False,
        )

    outcome = run_investigation_and_experiment_batch(
        store,
        make_state(),
        investigate,
        repair,
        review,
    )

    assert outcome.route == "regenerate_hypotheses"
    assert outcome.repair_investigator_calls == 15
    assert outcome.review_calls == 4
    assert len(reviewed_vectors) == 4
    assert all(
        (combination_id, "initial-1") in reviewed_vectors[0]
        for combination_id in COMBINATION_IDS
    )
    assert all(
        call[1:] == (
            f"thread-{call[0]}",
            f"scope-{call[0]}",
            "请补充 " + call[0] + " 的可核对计算证据。",
        )
        for call in repair_calls
    )
    assert all(session.repair_count == 3 for session in outcome.state.sessions)
    assert all(session.route_marker == "revise_hypothesis" for session in outcome.state.sessions)


def test_refuted_five_pass_findings_mark_all_combinations_for_regeneration(
    tmp_path: Path,
) -> None:
    store = InvestigationSQLiteStore(tmp_path / "state.sqlite3")

    def investigate(session: InvestigationSession) -> InvestigationResult:
        return make_result(session.combination_id, "initial-1", judgment="refuted")

    def review(
        state: InvestigationState,
        results: tuple[InvestigationResult, ...],
        vector: tuple[tuple[str, str], ...],
    ) -> BatchValidation:
        return make_validation(state, judgment="refuted", can_answer=False)

    outcome = run_investigation_and_experiment_batch(
        store,
        make_state(),
        investigate,
        lambda session, advice: make_result(session.combination_id, "repair-1"),
        review,
    )
    assert outcome.route == "regenerate_hypotheses"
    assert outcome.state.can_generate_next_batch is True


def test_supported_but_not_answerable_returns_only_answerability_explanation(
    tmp_path: Path,
) -> None:
    store = InvestigationSQLiteStore(tmp_path / "state.sqlite3")
    explanation = "支持性结果不能覆盖原问题要求的全部证据范围。"

    def investigate(session: InvestigationSession) -> InvestigationResult:
        return make_result(session.combination_id, "initial-1")

    outcome = run_investigation_and_experiment_batch(
        store,
        make_state(),
        investigate,
        lambda session, advice: make_result(session.combination_id, "repair-1"),
        lambda state, results, vector: make_validation(
            state,
            can_answer=False,
            explanation=explanation,
        ),
    )
    assert outcome.route == "answerability_explanation"
    assert outcome.state.answerability_explanation == explanation


def test_review_exception_leaves_results_saved_for_next_entry(tmp_path: Path) -> None:
    store = InvestigationSQLiteStore(tmp_path / "state.sqlite3")
    investigator_calls: list[str] = []

    def investigate(session: InvestigationSession) -> InvestigationResult:
        investigator_calls.append(session.combination_id)
        return make_result(session.combination_id, "initial-1")

    with pytest.raises(RuntimeError, match="review 失败"):
        run_investigation_and_experiment_batch(
            store,
            make_state(),
            investigate,
            lambda session, advice: make_result(session.combination_id, "repair-1"),
            lambda state, results, vector: (_ for _ in ()).throw(RuntimeError("review 失败")),
        )
    assert len(investigator_calls) == 5
    saved = store.load_investigation_snapshot("task-1")
    assert saved is not None
    assert len(saved.state.results) == 5
    assert saved.state.validations == ()

    outcome = run_investigation_and_experiment_batch(
        store,
        make_state(),
        lambda session: (_ for _ in ()).throw(AssertionError("不应重调调查器")),
        lambda session, advice: (_ for _ in ()).throw(AssertionError("不应重调返修")),
        lambda state, results, vector: make_validation(state),
    )
    assert outcome.route == "accept"
    assert outcome.initial_investigator_calls == 0


class _ConflictingStore:
    """在首次结果保存前写入另一事实，用于验证协调器不覆盖 CAS 冲突。"""

    def __init__(self, path: Path) -> None:
        self.inner = InvestigationSQLiteStore(path)
        self.injected = False

    def load_investigation_snapshot(self, task_id: str):
        return self.inner.load_investigation_snapshot(task_id)

    def read_saved_batch_validation(self, task_id: str, batch_id: str, vector):
        return self.inner.read_saved_batch_validation(task_id, batch_id, vector)

    def save_investigation_state(self, state: InvestigationState, expected_revision: int = 0):
        if expected_revision == 1 and not self.injected:
            self.injected = True
            current = self.inner.load_investigation_snapshot(state.task_id)
            assert current is not None
            changed = InvestigationState.model_validate(
                {
                    **current.state.model_dump(),
                    "sessions": tuple(
                        session.model_copy(update={"workspace_ref": "workspace/other"})
                        if session.combination_id == "combination-1"
                        else session
                        for session in current.state.sessions
                    ),
                }
            )
            self.inner.save_investigation_state(changed, expected_revision=1)
        return self.inner.save_investigation_state(state, expected_revision=expected_revision)


class _ResultReplayStore(_ConflictingStore):
    """在协调器 CAS 失败前写入相同结果，验证不重复调查或覆盖快照。"""

    def save_investigation_state(self, state: InvestigationState, expected_revision: int = 0):
        if expected_revision == 1 and not self.injected:
            self.injected = True
            self.inner.save_investigation_state(state, expected_revision=expected_revision)
            raise InvestigationRevisionConflict(state.task_id, 1, 2)
        return self.inner.save_investigation_state(state, expected_revision=expected_revision)


def test_unrelated_cas_conflict_is_explicit_and_does_not_overwrite_fact(
    tmp_path: Path,
) -> None:
    store = _ConflictingStore(tmp_path / "state.sqlite3")

    with pytest.raises(InvestigationCoordinatorConflict, match="没有发现可重放"):
        run_investigation_and_experiment_batch(
            store,  # type: ignore[arg-type]
            make_state(),
            lambda session: make_result(session.combination_id, "initial-1"),
            lambda session, advice: make_result(session.combination_id, "repair-1"),
            lambda state, results, vector: make_validation(state),
        )

    snapshot = store.load_investigation_snapshot("task-1")
    assert snapshot is not None
    assert snapshot.state.session_for("combination-1").workspace_ref == "workspace/other"
    assert snapshot.state.results == ()


def test_same_result_already_saved_during_cas_is_replayed_without_second_call(
    tmp_path: Path,
) -> None:
    store = _ResultReplayStore(tmp_path / "state.sqlite3")
    calls: list[str] = []

    def investigate(session: InvestigationSession) -> InvestigationResult:
        calls.append(session.combination_id)
        return make_result(session.combination_id, "initial-1")

    outcome = run_investigation_and_experiment_batch(
        store,  # type: ignore[arg-type]
        make_state(),
        investigate,
        lambda session, advice: make_result(session.combination_id, "repair-1"),
        lambda state, results, vector: make_validation(state),
    )
    assert outcome.route == "accept"
    assert outcome.replay_count == 1
    assert calls == list(COMBINATION_IDS)



def _state_with_five_saved_results() -> InvestigationState:
    """构造已保存五项 v1 结果、但尚未 Review 的批次状态。"""

    state = make_state()
    for combination_id in COMBINATION_IDS:
        state = apply_investigation_event(
            state,
            RecordInvestigationResult(result=make_result(combination_id, "initial-1")),
        )
    return state


def test_coordinator_batch_validation_cas_success_replay_and_conflict_paths(
    tmp_path: Path,
) -> None:
    """批次验证 CAS 成功、相同事实重放和不同事实冲突都保持幂等边界。"""

    base_state = _state_with_five_saved_results()
    validation = make_validation(base_state)

    success_store = InvestigationSQLiteStore(tmp_path / "success.sqlite3")
    success_store.save_investigation_state(base_state)
    saved_state, saved_revision, replayed = coordinator_module._persist_batch_validation(
        success_store,
        base_state,
        1,
        validation,
    )
    assert saved_state.validations == (validation,)
    assert saved_revision == 2
    assert replayed is False

    replay_store = InvestigationSQLiteStore(tmp_path / "replay.sqlite3")
    replay_store.save_investigation_state(base_state)
    competing_state = apply_investigation_event(
        base_state,
        ApplyBatchValidation(validation=validation),
    )
    competing_state = competing_state.model_copy(
        update={
            "sessions": tuple(
                session.model_copy(update={"workspace_ref": "workspace/other"})
                if session.combination_id == "combination-1"
                else session
                for session in competing_state.sessions
            )
        }
    )
    replay_store.save_investigation_state(competing_state, expected_revision=1)
    replayed_state, replayed_revision, replayed = coordinator_module._persist_batch_validation(
        replay_store,
        base_state,
        1,
        validation,
    )
    assert replayed_state == competing_state
    assert replayed_revision == 2
    assert replayed is True

    conflict_store = InvestigationSQLiteStore(tmp_path / "conflict.sqlite3")
    conflict_store.save_investigation_state(base_state)
    different = make_validation(
        base_state,
        can_answer=False,
        explanation="同一版本向量的不同回答能力判断。",
    )
    conflict_state = apply_investigation_event(
        base_state,
        ApplyBatchValidation(validation=different),
    )
    conflict_store.save_investigation_state(conflict_state, expected_revision=1)
    with pytest.raises(InvestigationCoordinatorConflict, match="没有发现可重放"):
        coordinator_module._persist_batch_validation(
            conflict_store,
            base_state,
            1,
            validation,
        )

    class _MissingSnapshotStore:
        """模拟 CAS 冲突后连恢复快照也暂时不可读的持久层。"""

        def save_investigation_state(self, *_args, **_kwargs):
            raise InvestigationRevisionConflict("task-1", 0, 1)

        def load_investigation_snapshot(self, _task_id: str):
            return None

    with pytest.raises(InvestigationCoordinatorConflict, match="没有发现可重放"):
        coordinator_module._persist_batch_validation(
            _MissingSnapshotStore(),
            base_state,
            0,
            validation,
        )


def test_coordinator_investigation_helper_rejects_empty_targets_and_missing_adapters(
    tmp_path: Path,
) -> None:
    """并行调查 helper 对空目标、预算和首次/返修依赖 fail-closed。"""

    store = InvestigationSQLiteStore(tmp_path / "helper.sqlite3")
    state = make_state()
    counters = coordinator_module._CallCounters()
    assert coordinator_module._invoke_and_persist_investigations(
        store,
        state,
        0,
        (),
        initial_investigator=None,
        repair_investigator=None,
        counters=counters,
        max_workers=0,
    ) == (state, 0)

    initial_target = ((0, make_session("combination-1"), "initial", None),)
    with pytest.raises(TypeError, match="正整数"):
        coordinator_module._invoke_and_persist_investigations(
            store,
            state,
            0,
            initial_target,
            initial_investigator=lambda _session: make_result("combination-1", "initial-1"),
            repair_investigator=None,
            counters=counters,
            max_workers=True,
        )
    with pytest.raises(ValueError, match="正整数"):
        coordinator_module._invoke_and_persist_investigations(
            store,
            state,
            0,
            initial_target,
            initial_investigator=lambda _session: make_result("combination-1", "initial-1"),
            repair_investigator=None,
            counters=counters,
            max_workers=0,
        )
    with pytest.raises(ValueError, match="initial_investigator"):
        coordinator_module._invoke_and_persist_investigations(
            store,
            state,
            0,
            initial_target,
            initial_investigator=None,
            repair_investigator=None,
            counters=counters,
            max_workers=1,
        )

    repair_target = (
        (0, make_session("combination-1", phase="repair_pending", result_version="v1"), "repair", "补足证据"),
    )
    with pytest.raises(ValueError, match="repair_investigator"):
        coordinator_module._invoke_and_persist_investigations(
            store,
            state,
            0,
            repair_target,
            initial_investigator=None,
            repair_investigator=None,
            counters=counters,
            max_workers=1,
        )

    failing_targets = (
        (0, make_session("combination-1"), "initial", None),
        (1, make_session("combination-2"), "initial", None),
    )

    def fail_investigation(_session: InvestigationSession) -> InvestigationResult:
        """让两个 future 同时失败，覆盖首个错误后的继续收集。"""

        raise RuntimeError("调查器失败")

    with pytest.raises(RuntimeError, match="调查器失败"):
        coordinator_module._invoke_and_persist_investigations(
            store,
            state,
            0,
            failing_targets,
            initial_investigator=fail_investigation,
            repair_investigator=None,
            counters=counters,
            max_workers=2,
        )


def test_coordinator_entry_rejects_invalid_identity_budget_and_orphaned_routes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """公开协调入口拒绝无任务、身份漂移和不可能的未推进路由。"""

    store = InvestigationSQLiteStore(tmp_path / "entry.sqlite3")
    with pytest.raises(ValueError, match="initial_state 或明确的 task_id"):
        run_investigation_and_experiment_batch(store)
    with pytest.raises(ValueError, match="不能为空"):
        run_investigation_and_experiment_batch(store, make_state(), task_id="")
    with pytest.raises(ValueError, match="不匹配"):
        run_investigation_and_experiment_batch(store, make_state(), task_id="other-task")
    with pytest.raises(TypeError, match="正整数"):
        run_investigation_and_experiment_batch(store, make_state(), max_workers=True)
    with pytest.raises(ValueError, match="正整数"):
        run_investigation_and_experiment_batch(store, make_state(), max_workers=0)

    incomplete = _state_with_five_saved_results()
    no_reviewer_store = InvestigationSQLiteStore(tmp_path / "no-reviewer.sqlite3")
    no_reviewer_store.save_investigation_state(incomplete)
    with pytest.raises(ValueError, match="batch_reviewer"):
        run_investigation_and_experiment_batch(
            no_reviewer_store,
            task_id="task-1",
            batch_reviewer=None,
        )

    with monkeypatch.context() as patch:
        patch.setattr(coordinator_module, "_select_investigation_targets", lambda _state: ())
        orphan_store = InvestigationSQLiteStore(tmp_path / "orphan.sqlite3")
        orphan_store.save_investigation_state(make_state())
        with pytest.raises(ValueError, match="既未形成"):
            run_investigation_and_experiment_batch(orphan_store, task_id="task-1")

        repair_state = make_state().model_copy(
            update={
                "sessions": (
                    make_session("combination-1", phase="repair_pending", result_version="v1"),
                    *make_state().sessions[1:],
                ),
                "results": (make_result("combination-1", "v1"),),
                "route": "repair_required",
            }
        )
        repair_store = InvestigationSQLiteStore(tmp_path / "orphan-repair.sqlite3")
        repair_store.save_investigation_state(repair_state)
        with pytest.raises(ValueError, match="repair_pending"):
            run_investigation_and_experiment_batch(repair_store, task_id="task-1")


class _ValidationRaceStore:
    """在批次验证 CAS 写入后模拟另一协调器已保存同一事实。"""

    def __init__(self, path: Path) -> None:
        self.inner = InvestigationSQLiteStore(path)
        self.injected = False

    def load_investigation_snapshot(self, task_id: str):
        return self.inner.load_investigation_snapshot(task_id)

    def read_saved_batch_validation(self, task_id: str, batch_id: str, vector):
        return self.inner.read_saved_batch_validation(task_id, batch_id, vector)

    def save_investigation_state(self, state: InvestigationState, expected_revision: int = 0):
        if expected_revision == 1 and not self.injected:
            self.injected = True
            self.inner.save_investigation_state(state, expected_revision=expected_revision)
            raise InvestigationRevisionConflict(state.task_id, 1, 2)
        return self.inner.save_investigation_state(state, expected_revision=expected_revision)


def test_coordinator_batch_validation_cas_replay_is_counted_without_second_review(
    tmp_path: Path,
) -> None:
    """Review CAS 竞争写入相同验证时只重放已保存事实，不再次调用 Review。"""

    state = _state_with_five_saved_results()
    validation = make_validation(state)
    store = _ValidationRaceStore(tmp_path / "validation-race.sqlite3")
    store.inner.save_investigation_state(state)

    outcome = run_investigation_and_experiment_batch(
        store,  # type: ignore[arg-type]
        task_id="task-1",
        batch_reviewer=lambda _state, _results, _vector: validation,
    )
    assert outcome.route == "accept"
    assert outcome.review_calls == 1
    assert outcome.replay_count == 1



def test_coordinator_validates_result_identity_thread_scope_and_duplicate_version() -> None:
    """结果持久化前逐项核对类型、任务批次、组合、原 thread、scope 和版本唯一性。"""

    state = make_state()
    session = state.sessions[0]
    valid = make_result("combination-1", "initial-1")
    with pytest.raises(TypeError, match="InvestigationResult"):
        coordinator_module._validate_investigation_result_for_session(state, session, object())
    with pytest.raises(ValueError, match="task_id/batch_id"):
        coordinator_module._validate_investigation_result_for_session(
            state,
            session,
            valid.model_copy(update={"task_id": "other-task"}),
        )
    with pytest.raises(ValueError, match="task_id/batch_id"):
        coordinator_module._validate_investigation_result_for_session(
            state,
            session,
            valid.model_copy(update={"batch_id": "other-batch"}),
        )
    with pytest.raises(ValueError, match="combination_id"):
        coordinator_module._validate_investigation_result_for_session(
            state,
            session,
            valid.model_copy(update={"combination_id": "combination-2"}),
        )
    repair_session = make_session(
        "combination-1",
        phase="repair_pending",
        result_version="v1",
    )
    with pytest.raises(ValueError, match="Codex thread"):
        coordinator_module._validate_investigation_result_for_session(
            state.model_copy(update={"sessions": (repair_session, *state.sessions[1:])}),
            repair_session,
            valid.model_copy(update={"codex_thread_id": "thread-other"}),
        )
    with pytest.raises(ValueError, match="scope"):
        coordinator_module._validate_investigation_result_for_session(
            state,
            session,
            valid.model_copy(update={"scope_id": "scope-other"}),
        )

    recorded = apply_investigation_event(
        state,
        RecordInvestigationResult(result=valid),
    )
    with pytest.raises(ValueError, match="result_version"):
        coordinator_module._validate_investigation_result_for_session(
            recorded,
            recorded.sessions[0],
            valid,
        )
