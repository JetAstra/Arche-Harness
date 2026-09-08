"""Investigation SQLite 持久层的确定性回归测试。"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from chemistry_multiagent.investigation.sqlite_store import (
    _canonical_json,
    _decode_state,
    _normalize_vector,
    _serialize_state,
    _validate_receipt_status_transition,
)

from chemistry_multiagent.investigation import (
    ApplyBatchValidation,
    BatchValidation,
    BeginNextBatch,
    InvestigationResult,
    InvestigationSession,
    InvestigationSQLiteStore,
    InvestigationPersistenceError,
    InvestigationState,
    InvestigationStateCorruption,
    InvestigationRevisionConflict,
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
            else f"thread-{batch_id}-{combination_id}"
        ),
        workspace_ref=f"workspace/{batch_id}/{combination_id}",
        scope_id=f"scope-{batch_id}-{combination_id}",
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
        codex_thread_id=f"thread-{batch_id}-{combination_id}",
        scope_id=f"scope-{batch_id}-{combination_id}",
        turn_id=f"turn-{batch_id}-{version}-{combination_id}",
        self_judgment=judgment,
        narrative=f"对 {combination_id} 的完整回顾性计算过程：版本 {version}。",
        job_refs=(f"job-{batch_id}-{version}-{combination_id}",),
        operation_refs=(f"operation-{batch_id}-{version}-{combination_id}",),
        artifact_refs=(f"artifact-{batch_id}-{version}-{combination_id}",),
        observations=(f"observation-{batch_id}-{version}-{combination_id}",),
    )


def record_all_results(state: InvestigationState, *, prefix: str = "v") -> InvestigationState:
    for index, combination_id in enumerate(COMBINATION_IDS, start=1):
        state = apply_investigation_event(
            state,
            RecordInvestigationResult(
                result=make_result(combination_id, f"{prefix}{index}", batch_id=state.batch_id)
            ),
        )
    return state


def make_validation(
    state: InvestigationState,
    *,
    judgment: str = "supported",
    can_answer: bool = True,
    final_answer: str | None = None,
    answerability_explanation: str | None = None,
) -> BatchValidation:
    return BatchValidation(
        task_id=state.task_id,
        batch_id=state.batch_id,
        combination_ids=COMBINATION_IDS,
        result_versions=tuple(
            state.session_for(combination_id).result_version  # type: ignore[arg-type]
            for combination_id in COMBINATION_IDS
        ),
        findings=tuple(
            ValidationFinding(
                combination_id=combination_id,
                validity="pass",
                hypothesis_judgment=judgment,
            )
            for combination_id in COMBINATION_IDS
        ),
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


def make_accepted_state() -> tuple[InvestigationState, BatchValidation]:
    state = record_all_results(make_state())
    validation = make_validation(state)
    return (
        apply_investigation_event(
            state,
            ApplyBatchValidation(validation=validation),
        ),
        validation,
    )


def test_restart_recovers_sessions_results_and_validation(tmp_path: Path) -> None:
    database = tmp_path / "investigation.sqlite3"
    state, validation = make_accepted_state()

    first_store = InvestigationSQLiteStore(database)
    revision = first_store.save_investigation_state(state)
    assert revision == 1

    restarted_store = InvestigationSQLiteStore(database)
    snapshot = restarted_store.load_investigation_snapshot("task-1")
    assert snapshot is not None
    assert snapshot.revision == 1
    assert snapshot.state == state
    assert snapshot.state.validations == (validation,)
    assert snapshot.state.final_answer == validation.final_answer
    assert snapshot.state.validations[0].final_answer == validation.final_answer
    assert len(snapshot.state.sessions) == 5
    assert len(snapshot.state.results) == 5


def test_cas_conflict_and_identical_state_idempotency(tmp_path: Path) -> None:
    store = InvestigationSQLiteStore(tmp_path / "investigation.sqlite3")
    state = make_state()
    assert store.save_investigation_state(state, expected_revision=0) == 1

    replay = store.save_investigation_state(state, expected_revision=0)
    assert replay == 1

    changed = InvestigationState.model_validate(
        {
            **state.model_dump(),
            "sessions": tuple(
                (
                    session.model_copy(
                        update={"workspace_ref": "workspace/changed"}
                    )
                    if session.combination_id == COMBINATION_IDS[0]
                    else session
                )
                for session in state.sessions
            ),
        }
    )
    with pytest.raises(InvestigationRevisionConflict):
        store.save_investigation_state(changed, expected_revision=0)
    assert store.save_investigation_state(changed, expected_revision=1) == 2


def test_regeneration_receipt_status_cannot_roll_back(tmp_path: Path) -> None:
    """重生 receipt 只允许向前推进阶段，不能把完成结果改回旧阶段。"""

    store = InvestigationSQLiteStore(tmp_path / "investigation.sqlite3")
    base = {
        "schema": "arche.investigation.hypothesis_regeneration_receipt",
        "schema_version": 1,
        "status": "started",
        "task_id": "task-1",
        "previous_batch_id": "batch-1",
        "next_batch_id": "batch-2",
    }
    assert (
        store.save_hypothesis_regeneration_receipt(
            "task-1",
            "regenerate:task-1:batch-2",
            base,
        )
        == 1
    )
    generated = {**base, "status": "generated"}
    assert (
        store.save_hypothesis_regeneration_receipt(
            "task-1",
            "regenerate:task-1:batch-2",
            generated,
            expected_revision=1,
        )
        == 2
    )

    with pytest.raises(InvestigationPersistenceError, match="receipt.status"):
        store.save_hypothesis_regeneration_receipt(
            "task-1",
            "regenerate:task-1:batch-2",
            base,
            expected_revision=2,
        )


def test_regeneration_receipt_claim_only_accepts_started(tmp_path: Path) -> None:
    """模型 owner claim 不能伪造 generated/completed 初始状态。"""

    store = InvestigationSQLiteStore(tmp_path / "investigation.sqlite3")
    with pytest.raises(InvestigationPersistenceError, match="claim|started"):
        store.claim_hypothesis_regeneration_receipt(
            "task-1",
            "regenerate:task-1:batch-2",
            {"status": "generated"},
        )


def test_full_review_replays_by_complete_vector_and_different_input_misses(
    tmp_path: Path,
) -> None:
    store = InvestigationSQLiteStore(tmp_path / "investigation.sqlite3")
    state, validation = make_accepted_state()
    store.save_investigation_state(state)

    replayed = store.read_saved_batch_validation(
        "task-1",
        "batch-1",
        tuple(
            zip(
                reversed(validation.combination_ids),
                reversed(validation.result_versions),
            )
        ),
    )
    assert replayed == validation

    different = store.read_saved_batch_validation(
        "task-1",
        "batch-1",
        tuple(
            zip(
                validation.combination_ids,
                tuple(f"different-{index}" for index in range(1, 6)),
            )
        ),
    )
    assert different is None

    by_mapping = store.read_saved_batch_validation(
        "task-1",
        "batch-1",
        dict(zip(validation.combination_ids, validation.result_versions)),
    )
    assert by_mapping == validation
    assert replayed.final_answer == validation.final_answer


def test_full_vector_keeps_shared_result_version_names_per_combination(
    tmp_path: Path,
) -> None:
    store = InvestigationSQLiteStore(tmp_path / "investigation.sqlite3")
    state = make_state()
    for combination_id in COMBINATION_IDS:
        state = apply_investigation_event(
            state,
            RecordInvestigationResult(
                result=make_result(combination_id, "1"),
            ),
        )
    validation = make_validation(state)
    accepted = apply_investigation_event(
        state,
        ApplyBatchValidation(validation=validation),
    )
    store.save_investigation_state(accepted)

    assert validation.result_versions == ("1",) * 5
    assert store.read_saved_batch_validation(
        "task-1",
        "batch-1",
        dict(zip(validation.combination_ids, validation.result_versions)),
    ) == validation


def test_history_retains_all_closed_batch_records(tmp_path: Path) -> None:
    store = InvestigationSQLiteStore(tmp_path / "investigation.sqlite3")
    state = record_all_results(make_state())
    validation = make_validation(state, judgment="refuted", can_answer=False)
    closed = apply_investigation_event(
        state,
        ApplyBatchValidation(validation=validation),
    )
    next_state = apply_investigation_event(
        closed,
        BeginNextBatch(
            batch_id="batch-2",
            sessions=tuple(
                make_session(f"new-combination-{index}", batch_id="batch-2")
                for index in range(1, 6)
            ),
        ),
    )

    assert store.save_investigation_state(closed) == 1
    assert store.save_investigation_state(next_state, expected_revision=1) == 2
    snapshot = store.load_investigation_snapshot("task-1")
    assert snapshot is not None
    recovered = snapshot.state
    assert recovered == next_state
    assert len(recovered.history) == 1  # type: ignore[union-attr]
    assert len(recovered.history[0].sessions) == 5  # type: ignore[union-attr]
    assert len(recovered.history[0].results) == 5  # type: ignore[union-attr]
    assert len(recovered.history[0].validations) == 1  # type: ignore[union-attr]


def test_corrupt_json_fails_closed_without_returning_partial_state(tmp_path: Path) -> None:
    database = tmp_path / "investigation.sqlite3"
    store = InvestigationSQLiteStore(database)
    state, _ = make_accepted_state()
    store.save_investigation_state(state)

    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE investigation_states SET state_json = ? WHERE task_id = ?",
            ("{not-json", "task-1"),
        )

    with pytest.raises(InvestigationStateCorruption):
        store.load_investigation_snapshot("task-1")


def test_corrupt_validation_replay_index_fails_closed(tmp_path: Path) -> None:
    database = tmp_path / "investigation.sqlite3"
    store = InvestigationSQLiteStore(database)
    state, _ = make_accepted_state()
    store.save_investigation_state(state)

    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE investigation_validation_replays "
            "SET validation_json = ? WHERE task_id = ?",
            ("{not-json", "task-1"),
        )

    with pytest.raises(InvestigationStateCorruption):
        store.load_investigation_snapshot("task-1")


def test_replay_index_revision_mismatch_fails_closed(tmp_path: Path) -> None:
    """验证重放索引版本偏离完整状态时不能恢复旧结果。"""

    database = tmp_path / "investigation.sqlite3"
    store = InvestigationSQLiteStore(database)
    state, _ = make_accepted_state()
    store.save_investigation_state(state)

    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE investigation_validation_replays SET state_revision = ? WHERE task_id = ?",
            (99, "task-1"),
        )

    with pytest.raises(InvestigationStateCorruption, match="版本损坏"):
        store.load_investigation_snapshot("task-1")


def test_replay_index_incomplete_vector_fails_closed(tmp_path: Path) -> None:
    """验证重放索引的版本向量必须保持五项完整形状。"""

    database = tmp_path / "investigation.sqlite3"
    store = InvestigationSQLiteStore(database)
    state, _ = make_accepted_state()
    store.save_investigation_state(state)

    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE investigation_validation_replays SET result_vector_json = ? WHERE task_id = ?",
            ("[]", "task-1"),
        )

    with pytest.raises(InvestigationStateCorruption, match="版本向量不完整"):
        store.load_investigation_snapshot("task-1")


def test_investigation_tables_are_additive_to_existing_database(tmp_path: Path) -> None:
    database = tmp_path / "shared.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE existing_component (value TEXT NOT NULL)")
        connection.execute("INSERT INTO existing_component(value) VALUES ('kept')")

    InvestigationSQLiteStore(database)
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT value FROM existing_component"
        ).fetchone() == ("kept",)
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='investigation_states'"
        ).fetchone() == ("investigation_states",)


def _regeneration_receipt() -> dict[str, object]:
    """构造可保存的最小假设重生 receipt。"""

    return {
        "schema": "arche.investigation.hypothesis_regeneration_receipt",
        "schema_version": 1,
        "status": "started",
        "task_id": "task-1",
        "previous_batch_id": "batch-1",
        "next_batch_id": "batch-2",
    }


def test_regeneration_receipt_loads_or_returns_none_without_defaults(tmp_path: Path) -> None:
    """receipt 恢复只返回摘要核对后的事实，缺失时保持 None。"""

    store = InvestigationSQLiteStore(tmp_path / "receipt.sqlite3")
    receipt = _regeneration_receipt()
    assert store.load_hypothesis_regeneration_receipt("task-1", "missing") is None
    assert store.save_hypothesis_regeneration_receipt(
        "task-1", "operation-1", receipt
    ) == 1

    loaded = store.load_hypothesis_regeneration_receipt("task-1", "operation-1")
    assert loaded is not None
    assert loaded.task_id == "task-1"
    assert loaded.operation_key == "operation-1"
    assert loaded.receipt == receipt
    assert loaded.revision == 1


@pytest.mark.parametrize("mutation, message", [
    ("revision", "revision 损坏"),
    ("columns", "列类型损坏"),
    ("json", "JSON 无法解析"),
    ("object", "必须是 object"),
    ("digest", "摘要不匹配"),
])
def test_regeneration_receipt_corruption_fails_closed(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    """receipt 任一持久列损坏都不能回退到默认阶段。"""

    database = tmp_path / f"receipt-{mutation}.sqlite3"
    store = InvestigationSQLiteStore(database)
    store.save_hypothesis_regeneration_receipt(
        "task-1", "operation-1", _regeneration_receipt()
    )
    with sqlite3.connect(database) as connection:
        if mutation == "revision":
            connection.execute("PRAGMA ignore_check_constraints = ON")
            connection.execute(
                "UPDATE investigation_regeneration_receipts SET revision = 0"
            )
        elif mutation == "columns":
            connection.execute(
                "UPDATE investigation_regeneration_receipts SET receipt_json = ?",
                (sqlite3.Binary(b"{}"),),
            )
        elif mutation == "json":
            connection.execute(
                "UPDATE investigation_regeneration_receipts SET receipt_json = ?",
                ("{not-json",),
            )
        elif mutation == "object":
            connection.execute(
                "UPDATE investigation_regeneration_receipts SET receipt_json = ?",
                ("[]",),
            )
        else:
            connection.execute(
                "UPDATE investigation_regeneration_receipts SET receipt_digest = ?",
                ("0" * 64,),
            )

    with pytest.raises(InvestigationStateCorruption, match=message):
        store.load_hypothesis_regeneration_receipt("task-1", "operation-1")


@pytest.mark.parametrize(
    ("existing_json", "next_status", "error_type", "message"),
    [
        ("{not-json", "started", InvestigationStateCorruption, "无法解析"),
        ("[]", "started", InvestigationStateCorruption, "必须是 object"),
        (json.dumps({"status": "unknown"}), "started", InvestigationStateCorruption, "旧 status"),
        (json.dumps({"status": "started"}), "unknown", InvestigationPersistenceError, "status 未注册"),
        (json.dumps({"status": "generated"}), "started", InvestigationPersistenceError, "不能回退"),
        (json.dumps({"status": "completed"}), "failed", InvestigationPersistenceError, "终局阶段"),
    ],
)
def test_receipt_status_transition_rejects_invalid_or_backward_steps(
    existing_json: str,
    next_status: str,
    error_type: type[Exception],
    message: str,
) -> None:
    """receipt 状态只能向前推进，终局不能被改写。"""

    with pytest.raises(error_type, match=message):
        _validate_receipt_status_transition(existing_json, next_status)

    _validate_receipt_status_transition(json.dumps({"status": "started"}), "failed")
    _validate_receipt_status_transition(json.dumps({"status": "completed"}), "completed")
    _validate_receipt_status_transition(json.dumps({"status": "failed"}), "failed")


@pytest.mark.parametrize(
    "vector",
    [
        "not-a-vector",
        (("c1", "v1", "extra"), ("c2", "v1"), ("c3", "v1"), ("c4", "v1"), ("c5", "v1")),
        (("c1", "v1"), ("c1", "v2"), ("c2", "v1"), ("c3", "v1"), ("c4", "v1")),
        (("c1", ""), ("c2", "v1"), ("c3", "v1"), ("c4", "v1"), ("c5", "v1")),
        ((["c1"], "v1"), ("c2", "v1"), ("c3", "v1"), ("c4", "v1"), ("c5", "v1")),
    ],
)
def test_result_version_vector_normalizer_rejects_incomplete_or_ambiguous_input(vector: object) -> None:
    """持久层 replay 只接受非空的一项或完整五项配对向量。"""

    with pytest.raises((TypeError, ValueError)):
        _normalize_vector(vector)  # type: ignore[arg-type]

    valid = tuple((f"c{index}", "v1") for index in range(1, 6))
    assert _normalize_vector(valid) == valid
    assert _normalize_vector(dict(valid)) == valid


def test_decode_state_accepts_canonical_snapshot_and_rejects_digest_types_or_identity(
    tmp_path: Path,
) -> None:
    """快照恢复核对 JSON、摘要和内部 task identity，任何漂移都停止。"""

    state = make_state()
    state_json, digest = _serialize_state(state)
    assert _decode_state("task-1", state_json, digest) == state

    with pytest.raises(InvestigationStateCorruption, match="摘要不匹配"):
        _decode_state("task-1", state_json, "0" * 64)
    with pytest.raises(InvestigationStateCorruption, match="列类型损坏"):
        _decode_state("task-1", None, digest)
    with pytest.raises(InvestigationStateCorruption, match="必须是 JSON 对象"):
        _decode_state("task-1", "[]", "0" * 64)

    payload = state.model_dump(mode="json", exclude_none=True)
    payload["task_id"] = "other-task"
    for session in payload["sessions"]:
        session["task_id"] = "other-task"
    other_json = _canonical_json(payload)
    other_digest = hashlib.sha256(other_json.encode("utf-8")).hexdigest()
    with pytest.raises(InvestigationStateCorruption, match="内部 task_id"):
        _decode_state("task-1", other_json, other_digest)



def test_regeneration_receipt_save_cas_and_corruption_paths_are_explicit(
    tmp_path: Path,
) -> None:
    """receipt 保存覆盖缺失版本、幂等、过期 CAS 和损坏 revision 分支。"""

    database = tmp_path / "receipt-cas.sqlite3"
    store = InvestigationSQLiteStore(database)
    receipt = _regeneration_receipt()

    with pytest.raises(InvestigationRevisionConflict, match="当前版本=0"):
        store.save_hypothesis_regeneration_receipt(
            "task-1",
            "operation-1",
            receipt,
            expected_revision=2,
        )
    assert store.save_hypothesis_regeneration_receipt(
        "task-1", "operation-1", receipt
    ) == 1
    assert store.save_hypothesis_regeneration_receipt(
        "task-1", "operation-1", receipt, expected_revision=0
    ) == 1

    generated = {**receipt, "status": "generated"}
    with pytest.raises(InvestigationRevisionConflict, match="当前版本=1"):
        store.save_hypothesis_regeneration_receipt(
            "task-1", "operation-1", generated, expected_revision=0
        )
    assert store.save_hypothesis_regeneration_receipt(
        "task-1", "operation-1", generated, expected_revision=1
    ) == 2
    assert store.save_hypothesis_regeneration_receipt(
        "task-1", "operation-1", generated, expected_revision=0
    ) == 2

    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA ignore_check_constraints = ON")
        connection.execute(
            "UPDATE investigation_regeneration_receipts SET revision = 0 "
            "WHERE task_id = ? AND operation_key = ?",
            ("task-1", "operation-1"),
        )
    with pytest.raises(InvestigationStateCorruption, match="revision 损坏"):
        store.save_hypothesis_regeneration_receipt(
            "task-1", "operation-1", generated, expected_revision=2
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("vector", "索引与正文不一致"),
        ("task", "索引与正文不一致"),
        ("raw_type", "JSON 类型损坏"),
        ("missing", "记录与快照不一致"),
    ],
)
def test_validation_replay_rows_reject_vector_identity_and_presence_drift(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    """验证重放索引必须和完整快照逐项一致，缺失也不能静默恢复。"""

    database = tmp_path / f"replay-{mutation}.sqlite3"
    store = InvestigationSQLiteStore(database)
    state, _ = make_accepted_state()
    store.save_investigation_state(state)

    with sqlite3.connect(database) as connection:
        row = connection.execute(
            "SELECT batch_id, idempotency_key, result_vector_json, validation_json "
            "FROM investigation_validation_replays WHERE task_id = ?",
            ("task-1",),
        ).fetchone()
        assert row is not None
        if mutation == "vector":
            vector = json.loads(row[2])
            vector[0][1] = "wrong-version"
            connection.execute(
                "UPDATE investigation_validation_replays SET result_vector_json = ? "
                "WHERE task_id = ?",
                (json.dumps(vector, ensure_ascii=False, separators=(",", ":")), "task-1"),
            )
        elif mutation == "task":
            payload = json.loads(row[3])
            payload["task_id"] = "other-task"
            connection.execute(
                "UPDATE investigation_validation_replays SET validation_json = ? "
                "WHERE task_id = ?",
                (json.dumps(payload, ensure_ascii=False, separators=(",", ":")), "task-1"),
            )
        elif mutation == "raw_type":
            connection.execute(
                "UPDATE investigation_validation_replays SET validation_json = ? "
                "WHERE task_id = ?",
                (sqlite3.Binary(b"{}"), "task-1"),
            )
        else:
            connection.execute(
                "DELETE FROM investigation_validation_replays WHERE task_id = ?",
                ("task-1",),
            )

    with pytest.raises(InvestigationStateCorruption, match=message):
        store.load_investigation_snapshot("task-1")
