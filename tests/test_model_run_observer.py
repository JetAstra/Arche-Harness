"""模型阶段观察器的路径、尝试和终态投影测试。"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from chemistry_multiagent.utils.model_run_observer import ModelRunStageObserver
from chemistry_multiagent.utils.model_run_outputs import ModelRunOutputWriter
from chemistry_multiagent.utils.model_run_view_publisher import ModelRunViewPublisher


def _artifact_ref(relative_path: str, *, media_type: str = "application/json") -> dict[str, Any]:
    """构造满足展示协议的稳定 ArtifactRef。"""

    return {
        "relative_path": relative_path,
        "sha256": "a" * 64,
        "media_type": media_type,
        "size": 1,
    }


def _writer(
    tmp_path: Path, research_task_id: str = "observer-test"
) -> ModelRunOutputWriter:
    """创建测试用展示 Writer。"""

    return ModelRunOutputWriter(
        tmp_path, research_task_id, persistence_mode="memory_off"
    )


def _start_event(
    *,
    operation: str,
    raw_stage: str,
    stage_index: int,
    display_context: Mapping[str, Any],
) -> dict[str, Any]:
    """构造观察器 start 事件的最小字段。"""

    return {
        "event_type": "model_stage_started",
        "operation": operation,
        "operation_id": f"{operation}-operation",
        "stage_index": stage_index,
        "raw_stage": raw_stage,
        "stage": raw_stage,
        "stage_kind": "model",
        "query_index": None,
        "validation_kind": "json_schema",
        "domain_validator_id": "test.validator.v1",
        "response_contract": {
            "validation_kind": "json_schema",
            "schema": {"type": "object"},
            "domain_validator_id": "test.validator.v1",
        },
        "stage_input": {"scientific_question": "测试问题"},
        "display_context": deepcopy(dict(display_context)),
        "started_at": "2026-08-11T00:00:00.000Z",
        "finished_at": None,
        "duration_ms": None,
    }


def test_observer_keeps_all_attempts_and_normalises_contract_and_usage(
    tmp_path: Path,
) -> None:
    """同一阶段的成功重答应保留全部回答和完整响应契约。"""

    writer = _writer(tmp_path)
    observer = ModelRunStageObserver(writer, display_context={"batch_index": 1})
    handle = observer.start_model_stage(
        _start_event(
            operation="hypothesis_generation",
            raw_stage="query",
            stage_index=1,
            display_context={"batch_index": 1},
        )
    )
    assert handle is not None
    for attempt_index, content in enumerate(("bad", "good")):
        observer.record_model_attempt(
            handle,
            {
                **_start_event(
                    operation="hypothesis_generation",
                    raw_stage="query",
                    stage_index=1,
                    display_context={"batch_index": 1},
                ),
                "event_type": "model_attempt_recorded",
                "attempt_index": attempt_index,
                "compiled_request": {
                    "model": "test-model",
                    "response_schema_hash": "hash-only",
                    "messages": [{"role": "user", "content": "问题"}],
                },
                "raw_response": {
                    "choices": [{"message": {"content": content}}],
                    "usage": {
                        "prompt_tokens": 11,
                        "completion_tokens": 7,
                        "total_tokens": 18,
                    },
                },
                "parsed_output": {"query": content},
                "validation": {
                    "status": "invalid" if attempt_index == 1 else "valid",
                    "failure_kind": "schema" if attempt_index == 1 else None,
                    "message": "重答" if attempt_index == 1 else None,
                },
                "started_at": "2026-08-11T00:00:00.000Z",
                "finished_at": "2026-08-11T00:00:00.010Z",
                "duration_ms": 10,
            },
        )
    observer.finish_model_stage(
        handle,
        {
            **_start_event(
                operation="hypothesis_generation",
                raw_stage="query",
                stage_index=1,
                display_context={"batch_index": 1},
            ),
            "event_type": "model_stage_finished",
            "status": "validated",
            "final_output": {"query": "good"},
            "finished_at": "2026-08-11T00:00:00.020Z",
            "source_artifacts": {
                "manifest": _artifact_ref("manifest.json"),
                "request_files": [
                    _artifact_ref("requests/001.json"),
                    _artifact_ref("requests/002.json"),
                ],
                "error_files": [_artifact_ref("errors/001.json")],
                "attempt": {"request_ref": _artifact_ref("requests/002.json")},
            },
        },
    )

    path = writer.resolve_output_path(
        "02_query/batch_001/generate_queries.json"
    )
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["status"] == "validated"
    assert len(document["attempts"]) == 2
    request = document["attempts"][0]["request"]
    assert request["response_contract"]["schema"] == {"type": "object"}
    assert document["attempts"][0]["usage"] == {
        "input_tokens": 11,
        "output_tokens": 7,
        "total_tokens": 18,
    }
    assert document["source_artifacts"]["manifest"]["relative_path"] == (
        "manifest.json"
    )
    assert set(document["source_artifacts"]) == {"manifest", "request_files"}


def test_observer_keeps_a_distinct_audit_file_for_terminal_operation_retry(
    tmp_path: Path,
) -> None:
    """新 operation 重试不得撞毁上一份终态 Planning 展示记录。"""

    writer = _writer(tmp_path, "observer-operation-retry")
    context = {"batch_index": 1, "combination_index": 1}
    observer = ModelRunStageObserver(writer, display_context=context)
    first_event = _start_event(
        operation="workflow_generation",
        raw_stage="workflow_generation",
        stage_index=1,
        display_context=context,
    )
    first_event["operation_id"] = "workflow-generation-operation-first"
    first = observer.start_model_stage(first_event)
    assert first is not None and first.writer_handle is not None
    observer.record_model_attempt(
        first,
        {
            **first_event,
            "attempt_index": 0,
            "validation": {
                "status": "transport_failed",
                "failure_kind": "transport",
                "message": "provider transport attempts were exhausted",
            },
        },
    )
    observer.finish_model_stage(
        first,
        {**first_event, "status": "failed", "final_output": None},
    )

    retry_event = deepcopy(first_event)
    retry_event["operation_id"] = "workflow-generation-operation-retry"
    retry = observer.start_model_stage(retry_event)

    assert retry is not None and retry.writer_handle is not None
    assert retry.writer_handle.relative_path.endswith(
        "/plan_workflow.retry_001.json"
    )
    assert writer.result_summary()["status"] == "available"
    assert observer.get_stage_paths() == [
        "04_planning/batch_001/combination_001/plan_workflow.json",
        "04_planning/batch_001/combination_001/plan_workflow.retry_001.json",
    ]


def test_observer_assigns_distinct_dynamic_paths_under_concurrency(tmp_path: Path) -> None:
    """并行关系判断按实际启动顺序分配不重复的比较文件。"""

    writer = _writer(tmp_path, "observer-concurrent")
    observer = ModelRunStageObserver(writer, display_context={"batch_index": 1})

    def start(index: int) -> str:
        handle = observer.start_model_stage(
            _start_event(
                operation="hypothesis_generation",
                raw_stage="relationship",
                stage_index=index,
                display_context={"batch_index": 1},
            )
        )
        assert handle is not None
        observer.finish_model_stage(
            handle,
            {
                **_start_event(
                    operation="hypothesis_generation",
                    raw_stage="relationship",
                    stage_index=index,
                    display_context={"batch_index": 1},
                ),
                "event_type": "model_stage_finished",
                "status": "validated",
                "final_output": "independent",
            },
        )
        return handle.writer_handle.relative_path

    with ThreadPoolExecutor(max_workers=4) as executor:
        paths = list(executor.map(start, range(1, 5)))

    assert sorted(paths) == [
        "internal/audit/model_calls/03_hypothesis/batch_001/relationships/comparison_001.json",
        "internal/audit/model_calls/03_hypothesis/batch_001/relationships/comparison_002.json",
        "internal/audit/model_calls/03_hypothesis/batch_001/relationships/comparison_003.json",
        "internal/audit/model_calls/03_hypothesis/batch_001/relationships/comparison_004.json",
    ]
    assert observer.get_hypothesis_merge_decisions() == []

    merge_handle = observer.start_model_stage(
        _start_event(
            operation="hypothesis_generation",
            raw_stage="merge",
            stage_index=5,
            display_context={"batch_index": 1},
        )
    )
    assert merge_handle is not None
    observer.finish_model_stage(
        merge_handle,
        {
            **_start_event(
                operation="hypothesis_generation",
                raw_stage="merge",
                stage_index=5,
                display_context={"batch_index": 1},
            ),
            "event_type": "model_stage_finished",
            "status": "validated",
            "stage_input": {
                "hypothesis_a": {"core_claim": "甲"},
                "hypothesis_b": {"core_claim": "甲"},
            },
            "final_output": {"core_claim": "合并后的判断"},
        },
    )
    merge_decisions = observer.get_hypothesis_merge_decisions()
    assert len(merge_decisions) == 1
    assert merge_decisions[0]["stage"] == "merge"
    assert all(
        "hypothesis_id" not in json.dumps(item, ensure_ascii=False)
        for item in merge_decisions
    )


def test_observer_projects_revision_final_output_to_patch_only(tmp_path: Path) -> None:
    """Revision 展示阶段只保存模型返回的 patch，不冒充完整工作流。"""

    writer = _writer(tmp_path, "observer-revision")
    observer = ModelRunStageObserver(
        writer,
        display_context={
            "batch_index": 1,
            "combination_index": 2,
            "round_index": 1,
        },
    )
    handle = observer.start_model_stage(
        _start_event(
            operation="workflow_revision",
            raw_stage="workflow_revision",
            stage_index=1,
            display_context={
                "batch_index": 1,
                "combination_index": 2,
                "round_index": 1,
            },
        )
    )
    assert handle is not None
    observer.record_model_attempt(
        handle,
        {
            **_start_event(
                operation="workflow_revision",
                raw_stage="workflow_revision",
                stage_index=1,
                display_context={
                    "batch_index": 1,
                    "combination_index": 2,
                    "round_index": 1,
                },
            ),
            "event_type": "model_attempt_recorded",
            "attempt_index": 0,
            "compiled_request": {"model": "test-model"},
            "raw_response": {"choices": []},
            "parsed_output": {"normalized_patch": {"operations": []}},
            "validation": {"status": "valid"},
        },
    )
    observer.finish_model_stage(
        handle,
        {
            **_start_event(
                operation="workflow_revision",
                raw_stage="workflow_revision",
                stage_index=1,
                display_context={
                    "batch_index": 1,
                    "combination_index": 2,
                    "round_index": 1,
                },
            ),
            "event_type": "model_stage_finished",
            "status": "validated",
            "final_output": {
                "normalized_patch": {"operations": []},
                "after_workflow": {"steps": []},
            },
        },
    )
    path = writer.resolve_output_path(
        "06_revision/batch_001/combination_002/round_001/revise_workflow.json"
    )
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["final_output"] == {"normalized_patch": {"operations": []}}


def test_publisher_rebases_root_stage_paths_for_each_view(tmp_path: Path) -> None:
    """summary、版本和轮次摘要的来源引用都应能从目标目录直接解析。"""

    writer = _writer(tmp_path, "observer-paths")
    publisher = ModelRunViewPublisher(writer)

    query_observer = ModelRunStageObserver(writer, display_context={"batch_index": 1})
    query_handle = query_observer.start_model_stage(
        _start_event(
            operation="hypothesis_generation",
            raw_stage="query",
            stage_index=1,
            display_context={"batch_index": 1},
        )
    )
    assert query_handle is not None
    query_observer.finish_model_stage(
        query_handle,
        {
            **_start_event(
                operation="hypothesis_generation",
                raw_stage="query",
                stage_index=1,
                display_context={"batch_index": 1},
            ),
            "event_type": "model_stage_finished",
            "status": "validated",
            "final_output": ["查询角度"],
        },
    )
    query_source = query_observer.get_stage_paths()[0]
    query_path = writer.resolve_output_path(query_source)

    hypothesis = {
        "hypothesis_id": "h-1",
        "strategy_name": "策略",
        "core_claim": "核心主张",
        "reasoning": "推理",
    }
    batch = {
        "batch_index": 1,
        "hypothesis_phase": {
            "queries": ["查询角度"],
            "hypotheses_by_query": [
                {"query": "查询角度", "hypotheses": [hypothesis]}
            ],
            "ranked_strategies": [hypothesis],
        },
        "candidates": [
            {
                "hypothesis_id": "h-1",
                "selection_status": "selected",
                "combination_id": "comb-1",
                "combination_index": 1,
            }
        ],
    }
    assert publisher.publish_hypothesis_summary(
        batch,
        [query_source],
        [],
        write_ranking=True,
    )
    summary_path = writer.resolve_output_path("03_hypothesis/batch_001/summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary_source = summary["source_files"][0]
    assert (summary_path.parent / summary_source).resolve() == query_path.resolve()

    planning_observer = ModelRunStageObserver(
        writer,
        display_context={"batch_index": 1, "combination_index": 1},
    )
    planning_handle = planning_observer.start_model_stage(
        _start_event(
            operation="workflow_generation",
            raw_stage="workflow_generation",
            stage_index=1,
            display_context={"batch_index": 1, "combination_index": 1},
        )
    )
    assert planning_handle is not None
    planning_observer.finish_model_stage(
        planning_handle,
        {
            **_start_event(
                operation="workflow_generation",
                raw_stage="workflow_generation",
                stage_index=1,
                display_context={"batch_index": 1, "combination_index": 1},
            ),
            "event_type": "model_stage_finished",
            "status": "validated",
            "final_output": {"steps": []},
        },
    )
    planning_source = planning_observer.get_stage_paths()[0]
    planning_path = writer.resolve_output_path(planning_source)
    assert publisher.publish_workflow_version(
        {
            "research_task_id": "observer-paths",
            "batch_index": 1,
            "combination_index": 1,
            "workflow_version": 1,
            "source_type": "initial_planning",
            "source_model_file": planning_source,
            "workflow": {"steps": []},
        }
    )
    version_path = writer.resolve_output_path(
        "04_planning/batch_001/combination_001/workflow_versions/version_001.json"
    )
    version = json.loads(version_path.read_text(encoding="utf-8"))
    assert (version_path.parent / version["source_model_file"]).resolve() == planning_path.resolve()

    router_source = (
        "05_router/batch_001/combination_001/round_001/"
        "evaluate_scientific_evidence.json"
    )
    assert publisher.append_router_round(
        1,
        1,
        {
            "round_index": 1,
            "decision": "accept",
            "problem_type": None,
            "source_file": router_source,
        }
    )
    router_summary_path = writer.resolve_output_path(
        "05_router/batch_001/combination_001/summary.json"
    )
    router_summary = json.loads(router_summary_path.read_text(encoding="utf-8"))
    assert router_summary["rounds"][0]["source_file"] == (
        "round_001/evaluate_scientific_evidence.json"
    )

    revision_source = (
        "06_revision/batch_001/combination_001/round_001/"
        "revise_workflow.json"
    )
    assert publisher.append_revision_round(
        1,
        1,
        {
            "round_index": 1,
            "source_file": revision_source,
            "workflow_version_before": 1,
            "workflow_version_after": 2,
            "normalized_patch": {"operations": []},
        }
    )
    revision_summary_path = writer.resolve_output_path(
        "06_revision/batch_001/combination_001/summary.json"
    )
    revision_summary = json.loads(revision_summary_path.read_text(encoding="utf-8"))
    assert revision_summary["rounds"][0]["source_file"] == "round_001/revise_workflow.json"
