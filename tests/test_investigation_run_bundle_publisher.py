"""验证 Investigation and Experiment 专用 Run Bundle 发布边界。"""

from __future__ import annotations

import json
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from chemistry_multiagent.investigation import (
    ApplyBatchValidation,
    BatchHistory,
    BatchValidation,
    InvestigationBatchOutcome,
    InvestigationResult,
    InvestigationSession,
    InvestigationState,
    ValidationFinding,
    apply_investigation_event,
)
from chemistry_multiagent.investigation.run_bundle_publisher import (
    _build_current_batch_projection,
    _validate_and_freeze_input,
    publish_investigation_batch_to_run_bundle,
    publish_hypothesis_batch_to_run_bundle,
)
from chemistry_multiagent.memory.contracts import canonical_json
from chemistry_multiagent.utils.run_bundle import (
    INVESTIGATION_STAGE_LAYOUT,
    RunBundle,
)
from tests.fakes.run_bundle import _SchemaV10BundleFixtureWriter


COMBINATION_IDS = tuple(f"combination-{index}" for index in range(1, 6))
AUTHOR_TASK_CONFIG = {
    "num_queries": 1,
    "num_hypotheses_per_query": 1,
    "top_n_hypotheses": 1,
    "max_parallel_investigations": 1,
    "max_investigation_repairs_per_combination": 3,
    "max_hypothesis_batches": 1,
    "investigation_combination_count": 1,
    "investigation_mode": "author_case1_main_hypothesis",
}


def _write_validated_hypothesis_manifest(
    bundle: RunBundle,
    *,
    operation_instance_id: str = "operation-hypothesis-1",
    stage_count: int = 142,
) -> tuple[str, dict[str, object]]:
    """写入 R129 形状的 142-stage 可核验 Hypothesis manifest。"""

    manifest: dict[str, object] = {
        "schema_version": 1,
        "operation": "hypothesis_generation",
        "operation_instance_id": operation_instance_id,
        "stages": [
            {
                "stage_index": index,
                "stage": f"query-{index}",
                "status": "validated",
                "attempts": [{"validation_status": "valid"}],
            }
            for index in range(1, stage_count + 1)
        ],
    }
    raw = json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    relative_path = (
        "working_memory/hypothesis_generation_inputs/"
        f"{operation_instance_id}/manifest_{digest}.json"
    )
    manifest_path = bundle.root / relative_path
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(raw)
    return relative_path, manifest


def _make_hypothesis_batch_payload(
    *,
    operation_instance_id: str = "operation-hypothesis-1",
) -> dict[str, object]:
    """构造 R129 形状的 12 candidates 和五项 selected payload。"""

    return {
        "batch_id": "batch-hypothesis-1",
        "batch_index": 1,
        "operation_instance_id": operation_instance_id,
        "candidates": [
            {
                "combination_id": combination_id,
                "combination_index": index,
                "hypothesis_id": f"hypothesis-{index}",
                "rank": index,
                "selection_status": "selected",
            }
            for index, combination_id in enumerate(COMBINATION_IDS, start=1)
        ]
        + [
            {
                "candidate_index": index,
                "combination_id": f"not-selected-{index}",
                "combination_index": None,
                "hypothesis_id": f"hypothesis-extra-{index}",
                "rank": index,
                "selection_status": "not_selected",
            }
            for index in range(6, 13)
        ],
    }


def _write_author_hypothesis_manifest(
    bundle: RunBundle,
    *,
    operation_instance_id: str = "author-hypothesis-import-1",
) -> tuple[str, dict[str, object], dict[str, object]]:
    """写入最小作者 manifest，并返回对应冻结 TaskCreated 与单候选。"""

    fixture = {
        "fixture_id": "case1_author_main_hypothesis",
        "payload": "frozen-author-fixture",
    }
    fixture_raw = canonical_json(fixture).encode("utf-8")
    fixture_sha256 = hashlib.sha256(fixture_raw).hexdigest()
    fixture_ref = (
        "internal/catalog/author_hypothesis_import/"
        f"fixture_{fixture_sha256}.json"
    )
    fixture_path = bundle.root / fixture_ref
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    fixture_path.write_bytes(fixture_raw)
    manifest = {
        "schema": "arche.investigation.author_hypothesis_import.v1",
        "schema_version": 1,
        "operation": "author_hypothesis_import",
        "operation_instance_id": operation_instance_id,
        "expected_combination_count": 1,
        "fixture_id": "case1_author_main_hypothesis",
        "fixture_sha256": fixture_sha256,
        "fixture_ref": fixture_ref,
        "model_calls": [],
        "generation_called": False,
    }
    manifest_raw = canonical_json(manifest).encode("utf-8")
    manifest_sha256 = hashlib.sha256(manifest_raw).hexdigest()
    manifest_ref = (
        "internal/catalog/author_hypothesis_import/"
        f"manifest_{manifest_sha256}.json"
    )
    manifest_path = bundle.root / manifest_ref
    manifest_path.write_bytes(manifest_raw)
    task_created = {
        "task_config": dict(AUTHOR_TASK_CONFIG),
        "hypothesis_import": {
            "mode": "author_case1_main_hypothesis",
            "fixture_id": "case1_author_main_hypothesis",
            "fixture_sha256": fixture_sha256,
            "manifest_schema": "arche.investigation.author_hypothesis_import.v1",
            "expected_combination_count": 1,
        },
    }
    payload = {
        "batch_id": "batch-author-1",
        "batch_index": 1,
        "operation_instance_id": operation_instance_id,
        "candidates": [
            {
                "combination_id": "combination-1",
                "combination_index": 1,
                "hypothesis_id": "hypothesis-author-1",
                "rank": 1,
                "selection_status": "selected",
            }
        ],
    }
    return manifest_ref, payload, task_created


def _make_sessions(*, batch_id: str = "batch-1") -> tuple[InvestigationSession, ...]:
    """构造已经形成当前结果版本的五组合会话。"""

    return tuple(
        InvestigationSession(
            task_id="task-1",
            batch_id=batch_id,
            combination_id=combination_id,
            codex_thread_id=f"thread-secret-{combination_id}",
            workspace_ref=f"workspace-secret-{combination_id}",
            scope_id=f"scope-secret-{combination_id}",
            result_version=f"v{index}",
            phase="ready_for_batch_validation",
        )
        for index, combination_id in enumerate(COMBINATION_IDS, start=1)
    )


def _make_results(*, batch_id: str = "batch-1") -> tuple[InvestigationResult, ...]:
    """构造五项当前调查结果，引用字段仅使用不透明标识。"""

    return tuple(
        InvestigationResult(
            task_id="task-1",
            batch_id=batch_id,
            combination_id=combination_id,
            result_version=f"v{index}",
            codex_thread_id=f"thread-secret-{combination_id}",
            scope_id=f"scope-secret-{combination_id}",
            turn_id=f"turn-secret-{combination_id}",
            self_judgment="supported",
            narrative=f"{combination_id} 的科学结果。",
            job_refs=(f"job-opaque-{index}",),
            operation_refs=(f"operation-opaque-{index}",),
            artifact_refs=(f"artifact-opaque-{index}",),
            observations=(f"{combination_id} 的观测。",),
            limitations=(f"{combination_id} 的限制。",),
        )
        for index, combination_id in enumerate(COMBINATION_IDS, start=1)
    )


def _make_state(*, batch_id: str = "batch-1") -> InvestigationState:
    """构造 Investigation 路由尚未完成批次验证的完整状态。"""

    return InvestigationState(
        task_id="task-1",
        batch_id=batch_id,
        sessions=_make_sessions(batch_id=batch_id),
        results=_make_results(batch_id=batch_id),
    )


def _make_validation(route: str) -> BatchValidation:
    """构造可把基础状态推进到指定领域路由的批次验证。"""

    if route == "accept":
        findings = tuple(
            ValidationFinding(
                combination_id=combination_id,
                validity="pass",
                hypothesis_judgment="supported",
            )
            for combination_id in COMBINATION_IDS
        )
        return BatchValidation(
            task_id="task-1",
            batch_id="batch-1",
            combination_ids=COMBINATION_IDS,
            result_versions=tuple(f"v{index}" for index in range(1, 6)),
            findings=findings,
            can_answer=True,
            final_answer="五项证据支持该结论。",
            review_thread_id="review-thread-secret",
            review_turn_id="review-turn-secret",
        )
    if route == "answerability_explanation":
        findings = tuple(
            ValidationFinding(
                combination_id=combination_id,
                validity="pass",
                hypothesis_judgment=("supported" if index == 1 else "refuted"),
            )
            for index, combination_id in enumerate(COMBINATION_IDS, start=1)
        )
        return BatchValidation(
            task_id="task-1",
            batch_id="batch-1",
            combination_ids=COMBINATION_IDS,
            result_versions=tuple(f"v{index}" for index in range(1, 6)),
            findings=findings,
            can_answer=False,
            answerability_explanation="证据支持部分判断，但不足以回答原问题。",
            review_thread_id="review-thread-secret",
            review_turn_id="review-turn-secret",
        )
    if route == "repair_required":
        findings = tuple(
            ValidationFinding(
                combination_id=combination_id,
                validity="fail" if index == 1 else "pass",
                hypothesis_judgment=None if index == 1 else "supported",
                repair_advice=("需要补足第一项证据。" if index == 1 else None),
            )
            for index, combination_id in enumerate(COMBINATION_IDS, start=1)
        )
        return BatchValidation(
            task_id="task-1",
            batch_id="batch-1",
            combination_ids=COMBINATION_IDS,
            result_versions=tuple(f"v{index}" for index in range(1, 6)),
            findings=findings,
            can_answer=False,
            review_thread_id="review-thread-secret",
            review_turn_id="review-turn-secret",
        )
    if route == "regenerate_hypotheses":
        findings = tuple(
            ValidationFinding(
                combination_id=combination_id,
                validity="pass",
                hypothesis_judgment="refuted",
            )
            for combination_id in COMBINATION_IDS
        )
        return BatchValidation(
            task_id="task-1",
            batch_id="batch-1",
            combination_ids=COMBINATION_IDS,
            result_versions=tuple(f"v{index}" for index in range(1, 6)),
            findings=findings,
            can_answer=False,
            review_thread_id="review-thread-secret",
            review_turn_id="review-turn-secret",
        )
    raise AssertionError(f"未知测试路由: {route}")


def _make_routed_state(route: str) -> InvestigationState:
    """用领域入口推进基础状态，避免测试绕过领域路由合同。"""

    return apply_investigation_event(
        _make_state(),
        ApplyBatchValidation(validation=_make_validation(route)),
    )


def _make_outcome(state: InvestigationState) -> InvestigationBatchOutcome:
    """构造与冻结 state 绑定的协调结果。"""

    return InvestigationBatchOutcome(
        state=state,
        revision=7,
        route=state.route,
        review_called=True,
        initial_investigator_calls=5,
        repair_investigator_calls=0,
        review_calls=1,
        replay_count=0,
    )


def _read_json(path: Path) -> dict:
    """读取测试用 JSON 对象。"""

    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("route", "expected_stage_status", "expected_run_status"),
    [
        ("investigation", "running", "running"),
        ("repair_required", "running", "running"),
        ("regenerate_hypotheses", "completed", "running"),
        ("accept", "completed", "succeeded"),
        ("answerability_explanation", "completed", "succeeded"),
    ],
)
def test_publish_routes_follow_investigation_terminal_rules(
    tmp_path: Path,
    route: str,
    expected_stage_status: str,
    expected_run_status: str,
) -> None:
    """五种领域路由必须映射到正确的阶段和运行状态。"""

    state = _make_state() if route == "investigation" else _make_routed_state(route)
    bundle = RunBundle(
        tmp_path,
        f"route-{route}",
    )

    publication = publish_investigation_batch_to_run_bundle(
        bundle,
        _make_outcome(state) if route != "investigation" else state,
    )

    investigation_stage = _read_json(
        bundle.root / "stages" / "investigation_and_experiment.json"
    )
    status = _read_json(bundle.status_path)
    assert publication.route == route
    assert investigation_stage["status"] == expected_stage_status
    assert status["status"] == expected_run_status
    assert len(investigation_stage["entries"]) == 1
    assert investigation_stage["summary"]["route"] == route

    final_stage = _read_json(bundle.root / "stages" / "task_review_and_result.json")
    final_summary = _read_json(bundle.summary_path)["final_result"]
    if route in {"accept", "answerability_explanation"}:
        assert publication.final_result_path == bundle.root / "stages" / "task_review_and_result.json"
        assert final_stage["status"] == "completed"
        assert final_summary["outcome_kind"] == route
        if route == "accept":
            assert final_summary["final_answer"] == state.final_answer
            assert "answerability_explanation" not in final_summary
        else:
            assert final_summary["answerability_explanation"] == state.answerability_explanation
            assert "final_answer" not in final_summary
    else:
        assert publication.final_result_path is None
        assert final_stage["status"] == "not_started"
        assert final_stage["entries"] == []
        assert final_summary is None
        if route == "regenerate_hypotheses":
            assert investigation_stage["summary"]["regeneration_request"]["requested"] is True


def test_publish_waiting_job_route_keeps_bundle_running_without_result(
    tmp_path: Path,
) -> None:
    """已提交的长作业尚未返回结果时，发布器必须保留可恢复运行态。"""

    waiting = {"status": "waiting_investigation_job"}
    state = InvestigationState(
        task_id="task-waiting",
        batch_id="batch-waiting",
        expected_combination_count=1,
        sessions=(
            InvestigationSession(
                task_id="task-waiting",
                batch_id="batch-waiting",
                combination_id="combination-waiting",
                codex_thread_id="thread-secret-waiting",
                workspace_ref="workspace-secret-waiting",
                scope_id="scope-secret-waiting",
            ),
        ),
        route="waiting_investigation_job",
        waiting_investigation_job=waiting,
        waiting_investigation_jobs=(waiting,),
    )
    bundle = RunBundle(
        tmp_path,
        "waiting-route",
    )

    publication = publish_investigation_batch_to_run_bundle(bundle, state)

    investigation_stage = _read_json(
        bundle.root / "stages" / "investigation_and_experiment.json"
    )
    assert publication.route == "waiting_investigation_job"
    assert publication.stage_status == "running"
    assert publication.run_status == "running"
    assert publication.call_identity["result_version_vector"] == []
    assert investigation_stage["status"] == "running"
    assert investigation_stage["summary"]["route"] == "waiting_investigation_job"
    assert investigation_stage["summary"]["result_version_vector"] == []
    assert investigation_stage["summary"]["current_results"] == []
    assert _read_json(bundle.status_path)["status"] == "running"


def test_publish_failed_route_preserves_result_review_context_and_marks_bundle_failed(
    tmp_path: Path,
) -> None:
    """失败路由保留当前调查摘要，并同步 Bundle 的失败状态。"""

    state_payload = _make_state().model_dump()
    state_payload.update(
        {
            "route": "failed",
            "failure_reason_code": "investigation_repair_exhausted",
            "failure_error_type": "InvestigationRepairExhausted",
        }
    )
    state = InvestigationState(**state_payload)
    bundle = RunBundle(tmp_path, "failed-route")

    publication = publish_investigation_batch_to_run_bundle(bundle, state)

    investigation_stage = _read_json(
        bundle.root / "stages" / "investigation_and_experiment.json"
    )
    final_stage = _read_json(bundle.root / "stages" / "task_review_and_result.json")
    final_summary = _read_json(bundle.summary_path)["final_result"]
    assert publication.route == "failed"
    assert publication.run_status == "failed"
    assert investigation_stage["status"] == "failed"
    assert investigation_stage["summary"]["failure"]["reason_code"] == (
        "investigation_repair_exhausted"
    )
    assert _read_json(bundle.status_path)["status"] == "failed"
    assert final_stage["status"] == "completed"
    assert final_summary["status"] == "failed"
    assert final_summary["failure"]["failure_kind"] == (
        "investigation_workflow_failure"
    )
    assert final_summary["failure"]["error_type"] == "InvestigationRepairExhausted"


def test_publish_is_idempotent_and_uses_sorted_complete_version_identity(
    tmp_path: Path,
) -> None:
    """相同任务、批次和版本向量的恢复重放不增加 entry。"""

    state = _make_routed_state("accept")
    bundle = RunBundle(tmp_path, "idempotent")

    first = publish_investigation_batch_to_run_bundle(bundle, _make_outcome(state))
    second = publish_investigation_batch_to_run_bundle(bundle, _make_outcome(state))

    expected_vector = [
        [f"combination-{index}", f"v{index}"] for index in range(1, 6)
    ]
    assert first.call_identity == {
        "research_task_id": "task-1",
        "batch_id": "batch-1",
        "result_version_vector": expected_vector,
    }
    assert "task_id" not in first.call_identity
    assert second.call_identity == first.call_identity
    investigation_stage = _read_json(
        bundle.root / "stages" / "investigation_and_experiment.json"
    )
    final_stage = _read_json(bundle.root / "stages" / "task_review_and_result.json")
    assert len(investigation_stage["entries"]) == 1
    assert len(final_stage["entries"]) == 1
    assert _read_json(bundle.status_path)["status"] == "succeeded"


def test_publish_projects_history_and_redacts_runtime_identity(
    tmp_path: Path,
) -> None:
    """阶段摘要保留验证历史和引用，但不泄露会话运行时身份。"""

    accepted = _make_routed_state("accept")
    history = BatchHistory(
        task_id=accepted.task_id,
        batch_id=accepted.batch_id,
        sessions=accepted.sessions,
        results=accepted.results,
        validations=accepted.validations,
        route=accepted.route,
        final_answer=accepted.final_answer,
    )
    current = InvestigationState(
        task_id="task-1",
        batch_id="batch-2",
        sessions=_make_sessions(batch_id="batch-2"),
        results=_make_results(batch_id="batch-2"),
        history=(history,),
    )
    bundle = RunBundle(tmp_path, "redaction")

    publish_investigation_batch_to_run_bundle(bundle, current)

    stage_path = bundle.root / "stages" / "investigation_and_experiment.json"
    stage_text = stage_path.read_text(encoding="utf-8")
    stage = _read_json(stage_path)
    summary = stage["summary"]
    assert len(summary["validations"]) == 1
    assert summary["validations"][0]["batch_id"] == "batch-1"
    assert summary["previous_batches"][0]["batch_id"] == "batch-1"
    assert summary["current_results"][0]["artifact_refs"] == ["artifact-opaque-1"]
    assert summary["current_results"][0]["job_refs"] == ["job-opaque-1"]
    assert summary["current_results"][0]["operation_refs"] == ["operation-opaque-1"]
    for secret in (
        "thread-secret",
        "scope-secret",
        "workspace-secret",
        "turn-secret",
        "review-thread-secret",
        "review-turn-secret",
    ):
        assert secret not in stage_text


def test_publish_rejects_legacy_layout_and_inconsistent_or_incomplete_input(
    tmp_path: Path,
) -> None:
    """布局、outcome 路由、终局文本和版本向量错误都必须 fail-closed。"""

    state = _make_routed_state("accept")
    legacy_bundle = _SchemaV10BundleFixtureWriter.create(
        tmp_path,
        "legacy",
        stage_layout=INVESTIGATION_STAGE_LAYOUT,
    )
    with pytest.raises(ValueError, match="research_task"):
        publish_investigation_batch_to_run_bundle(legacy_bundle, state)

    mismatch_bundle = RunBundle(
        tmp_path,
        "mismatch",
    )
    mismatched = _make_outcome(state)
    mismatched = InvestigationBatchOutcome(
        state=mismatched.state,
        revision=mismatched.revision,
        route="investigation",
        review_called=mismatched.review_called,
        initial_investigator_calls=mismatched.initial_investigator_calls,
        repair_investigator_calls=mismatched.repair_investigator_calls,
        review_calls=mismatched.review_calls,
        replay_count=mismatched.replay_count,
    )
    with pytest.raises(ValueError, match="route"):
        publish_investigation_batch_to_run_bundle(mismatch_bundle, mismatched)

    missing_answer_bundle = RunBundle(
        tmp_path,
        "missing-answer",
    )
    missing_answer = state.model_copy(update={"final_answer": None})
    with pytest.raises(ValueError, match="final_answer"):
        publish_investigation_batch_to_run_bundle(missing_answer_bundle, missing_answer)
    assert _read_json(
        missing_answer_bundle.root / "stages" / "investigation_and_experiment.json"
    )["entries"] == []

    incomplete_bundle = RunBundle(
        tmp_path,
        "incomplete-vector",
    )
    broken_session = state.sessions[0].model_copy(
        update={"phase": "ready_for_batch_validation", "result_version": None}
    )
    incomplete = state.model_copy(
        update={"sessions": (broken_session, *state.sessions[1:])}
    )
    with pytest.raises(ValueError, match="result_version"):
        publish_investigation_batch_to_run_bundle(incomplete_bundle, incomplete)
    assert _read_json(
        incomplete_bundle.root / "stages" / "investigation_and_experiment.json"
    )["entries"] == []


def test_publish_rejects_path_like_opaque_reference_before_writing(
    tmp_path: Path,
) -> None:
    """路径样式的结果引用不能被当作公开不透明引用写入摘要。"""

    state = _make_state()
    unsafe_result = state.results[0].model_copy(
        update={"artifact_refs": ("/mnt/private/result.log",)}
    )
    unsafe_state = state.model_copy(
        update={"results": (unsafe_result, *state.results[1:])}
    )
    bundle = RunBundle(tmp_path, "unsafe-ref")

    with pytest.raises(ValueError, match="不能包含本机路径"):
        publish_investigation_batch_to_run_bundle(bundle, unsafe_state)
    assert _read_json(
        bundle.root / "stages" / "investigation_and_experiment.json"
    )["entries"] == []


def test_current_projection_rejects_incomplete_or_duplicate_sessions() -> None:
    """公开阶段投影在组合数或 combination_id 不唯一时 fail-closed。"""

    state = _make_state()
    with pytest.raises(ValueError, match="完整五组合"):
        _build_current_batch_projection(state.model_copy(update={"sessions": state.sessions[:4]}))

    duplicate_session = state.sessions[0].model_copy(
        update={"combination_id": state.sessions[1].combination_id}
    )
    duplicate_state = state.model_copy(
        update={"sessions": (duplicate_session, *state.sessions[1:])}
    )
    with pytest.raises(ValueError, match="combination_id 必须唯一"):
        _build_current_batch_projection(duplicate_state)


def test_current_projection_rejects_duplicate_or_missing_results() -> None:
    """当前五组合结果必须各自有唯一的组合与版本配对。"""

    state = _make_state()
    duplicate_result = state.results[0].model_copy(
        update={
            "combination_id": state.results[1].combination_id,
            "result_version": state.results[1].result_version,
        }
    )
    duplicate_state = state.model_copy(
        update={"results": (duplicate_result, *state.results[1:])}
    )
    with pytest.raises(ValueError, match="结果版本重复"):
        _build_current_batch_projection(duplicate_state)

    missing_state = state.model_copy(update={"results": state.results[:-1]})
    with pytest.raises(ValueError, match="缺少对应"):
        _build_current_batch_projection(missing_state)


def test_validate_publication_input_rejects_invalid_outcome_counters() -> None:
    """发布前冻结步骤拒绝错误 revision、布尔计数和负计数。"""

    state = _make_routed_state("accept")
    outcome = _make_outcome(state)
    with pytest.raises(TypeError, match="revision"):
        _validate_and_freeze_input(replace(outcome, revision=True))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="revision"):
        _validate_and_freeze_input(replace(outcome, revision=-1))
    with pytest.raises(TypeError, match="review_called"):
        _validate_and_freeze_input(replace(outcome, review_called=1))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="initial_investigator_calls"):
        _validate_and_freeze_input(
            replace(outcome, initial_investigator_calls=True)  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="repair_investigator_calls"):
        _validate_and_freeze_input(
            replace(outcome, repair_investigator_calls=-1)
        )


def test_publish_hypothesis_batch_marks_completed_from_validated_manifest_only(
    tmp_path: Path,
) -> None:
    """五项选择和已验证 manifest 存在时才完成 Hypothesis 阶段，且不泄露科学正文。"""

    bundle = RunBundle(
        tmp_path,
        "hypothesis-projection",
    )
    manifest_ref, _manifest = _write_validated_hypothesis_manifest(bundle)
    payload = _make_hypothesis_batch_payload()

    publication = publish_hypothesis_batch_to_run_bundle(
        bundle,
        batch_payload=payload,
        source_refs=(manifest_ref,),
    )
    replay = publish_hypothesis_batch_to_run_bundle(
        bundle,
        batch_payload=payload,
        source_refs=(manifest_ref,),
    )

    assert publication.stage_status == "completed"
    assert replay.call_identity == publication.call_identity
    assert publication.call_identity["research_task_id"] == bundle.research_task_id
    assert "task_id" not in publication.call_identity
    stage = _read_json(bundle.root / "stages/hypothesis_preparation.json")
    assert stage["status"] == "completed"
    assert len(stage["entries"]) == 1
    summary = stage["entries"][0]["summary"]
    assert summary["status"] == "completed"
    assert summary["selected_count"] == 5
    assert [item["combination_id"] for item in summary["selected_combinations"]] == list(
        COMBINATION_IDS
    )
    assert len(payload["candidates"]) == 12
    assert summary["manifest"]["stage_count"] == 142
    assert summary["manifest"]["validated_stage_count"] == 142
    assert not any(
        key in summary
        for key in ("scientific_question", "hypothesis_text", "provider", "response")
    )
    assert _read_json(bundle.root / "stages/investigation_and_experiment.json")["status"] == (
        "not_started"
    )


def test_publish_hypothesis_batch_rejects_invalid_selection_before_any_write(
    tmp_path: Path,
) -> None:
    """五组合校验失败时不得留下阶段状态或 entry。"""

    bundle = RunBundle(
        tmp_path,
        "hypothesis-invalid",
    )
    manifest_ref, _manifest = _write_validated_hypothesis_manifest(bundle)
    payload = _make_hypothesis_batch_payload()
    invalid_candidate = dict(payload["candidates"][0])  # type: ignore[index]
    invalid_candidate["selection_status"] = "not_selected"
    payload["candidates"] = [invalid_candidate, *payload["candidates"][1:]]  # type: ignore[index]

    with pytest.raises(ValueError, match="五个 selected"):
        publish_hypothesis_batch_to_run_bundle(
            bundle,
            batch_payload=payload,
            source_refs=(manifest_ref,),
        )

    stage = _read_json(bundle.root / "stages/hypothesis_preparation.json")
    assert stage["status"] == "not_started"
    assert stage["entries"] == []


def test_publish_author_hypothesis_requires_frozen_task_created_binding(
    tmp_path: Path,
) -> None:
    """作者单候选不能只凭 manifest operation 发布，必须绑定 TaskCreated 摘要。"""

    bundle = RunBundle(
        tmp_path,
        "author-hypothesis-binding",
    )
    manifest_ref, payload, task_created = _write_author_hypothesis_manifest(bundle)

    with pytest.raises(ValueError, match="冻结 TaskCreated"):
        publish_hypothesis_batch_to_run_bundle(
            bundle,
            batch_payload=payload,
            source_refs=(manifest_ref,),
        )
    assert _read_json(bundle.root / "stages/hypothesis_preparation.json")["entries"] == []

    publication = publish_hypothesis_batch_to_run_bundle(
        bundle,
        batch_payload=payload,
        source_refs=(manifest_ref,),
        task_created=task_created,
    )
    assert publication.selected_count == 1
    summary = _read_json(bundle.root / "stages/hypothesis_preparation.json")["entries"][0][
        "summary"
    ]
    assert summary["selected_count"] == 1
