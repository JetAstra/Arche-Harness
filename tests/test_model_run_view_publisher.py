from __future__ import annotations

import json
from pathlib import Path

from chemistry_multiagent.utils.model_run_outputs import ModelRunOutputWriter
from chemistry_multiagent.utils.model_run_view_publisher import ModelRunViewPublisher


def _writer(
    tmp_path: Path, research_task_id: str = "diagnostic-publisher"
) -> ModelRunOutputWriter:
    return ModelRunOutputWriter(
        tmp_path,
        research_task_id,
        persistence_mode="memory_off",
        scientific_question="验证模型视图发布。",
        task_config={"num_queries": 1},
    )


def _hypothesis(hypothesis_id: str) -> dict:
    return {
        "hypothesis_id": hypothesis_id,
        "strategy_name": f"策略-{hypothesis_id}",
        "core_claim": f"{hypothesis_id} 的核心原文  ",
        "reasoning": f"{hypothesis_id} 的理由原文",
    }


def _batch() -> dict:
    h1 = _hypothesis("h-1")
    h2 = _hypothesis("h-2")
    return {
        "batch_index": 1,
        "hypothesis_phase": {
            "queries": ["角度一"],
            "hypotheses_by_query": [{"query": "角度一", "hypotheses": [h1, h2]}],
            "ranked_strategies": [h1, h2],
        },
        "candidates": [
            {
                "hypothesis_id": "h-1",
                "rank": 1,
                "selection_status": "selected",
                "combination_id": "comb-1",
                "combination_index": 1,
            },
            {
                "hypothesis_id": "h-2",
                "rank": 2,
                "selection_status": "not_selected",
                "combination_id": None,
                "combination_index": None,
            },
        ],
    }


def _source_files() -> dict[str, dict[str, str | None]]:
    return {
        "h-1": {"generation": "generation/h-1.json", "planning": "plan.json", "router": "router.json"},
        "h-2": {"generation": "generation/h-2.json", "planning": None, "router": None},
    }


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_publish_summary_and_optional_immutable_ranking_use_three_digit_paths(tmp_path):
    writer = _writer(tmp_path)
    publisher = ModelRunViewPublisher(writer)
    source_files = [
        "02_query/batch_001/generate_queries.json",
        "03_hypothesis/batch_001/generation/query_001.json",
        "03_hypothesis/batch_001/relationships/comparison_001.json",
        "03_hypothesis/batch_001/merges/merge_001.json",
        "03_hypothesis/batch_001/ranking/comparison_001.json",
    ]
    merge_decisions = [
        {
            "stage": "merge",
            "status": "validated",
            "result": {"relationship": "merge"},
            "source_file": "03_hypothesis/batch_001/merges/merge_001.json",
        }
    ]
    for relative_path in {
        *source_files,
        merge_decisions[0]["source_file"],
    }:
        path = writer.resolve_output_path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")

    assert publisher.publish_hypothesis_summary(
        _batch(),
        source_files,
        merge_decisions,
        write_ranking=True,
    )
    summary_path = writer.resolve_output_path(
        "03_hypothesis/batch_001/summary.json"
    )
    ranking_path = writer.resolve_output_path(
        "03_hypothesis/batch_001/ranking/final_ranking.json"
    )
    summary = _read(summary_path)
    ranking = _read(ranking_path)
    expected_ranked = _batch()["hypothesis_phase"]["ranked_strategies"]
    assert summary["ranked_hypotheses"] == expected_ranked
    assert ranking["ranked_hypotheses"] == expected_ranked
    assert [item["hypothesis_id"] for item in ranking["ranked_hypotheses"]] == ["h-1", "h-2"]
    for reference in summary["source_files"]:
        assert (summary_path.parent / reference).resolve().is_file()
    for merge in summary["merge_decisions"]:
        assert (summary_path.parent / merge["source_file"]).resolve().is_file()
    for reference in ranking["source_files"]:
        assert (ranking_path.parent / reference).resolve().is_file()


def test_publish_outcomes_and_workflow_version_keep_gap_and_version_paths(tmp_path):
    writer = _writer(tmp_path, "diagnostic-publisher-outcomes")
    publisher = ModelRunViewPublisher(writer)
    batch = _batch()
    gap = {
        "result_type": "tool_capability_gap",
        "missing_capabilities": ["原始缺口能力"],
        "reason": "Revision 返回的缺口原文。",
        "source_stage": "revision",
    }
    assert publisher.publish_hypothesis_outcomes(
        batch,
        [
            {
                "hypothesis_id": "h-1",
                "combination_index": 1,
                "state": "tool_capability_gap",
                "terminal_reason": "tool_capability_gap",
                "tool_capability_gap": gap,
            }
        ],
        _source_files(),
    )
    assert publisher.publish_workflow_version(
        {
            "research_task_id": "diagnostic-publisher-outcomes",
            "batch_index": 1,
            "combination_index": 1,
            "source_type": "initial_planning",
            "source_model_file": "04_planning/plan_workflow.json",
            "workflow_version": 1,
            "workflow": {"goal": "工作流原文"},
        }
    )
    outcomes = _read(
        writer.resolve_output_path("03_hypothesis/batch_001/outcomes.json")
    )
    version = _read(
        writer.resolve_output_path(
            "04_planning/batch_001/combination_001/workflow_versions/version_001.json"
        )
    )
    item = next(item for item in outcomes["hypotheses"] if item["hypothesis_id"] == "h-1")
    assert item["tool_capability_gap"] == gap
    assert version["workflow_version"] == 1


def test_router_append_resume_and_same_round_update_without_duplicate(tmp_path):
    writer = _writer(tmp_path, "diagnostic-publisher-router")
    publisher = ModelRunViewPublisher(writer)
    assert publisher.append_router_round(
        1,
        1,
        {
            "round_index": 1,
            "decision": "revise_workflow",
            "problem_type": "first_problem",
            "source_file": "round_001/evaluate_scientific_evidence.json",
        },
    )
    assert publisher.append_router_round(
        1,
        1,
        {
            "round_index": 2,
            "decision": "evidence_ready",
            "problem_type": None,
            "source_file": "round_002/evaluate_scientific_evidence.json",
        },
    )

    resumed_writer = ModelRunOutputWriter(
        tmp_path,
        "diagnostic-publisher-router",
        persistence_mode="memory_off",
        resume=True,
    )
    resumed = ModelRunViewPublisher(resumed_writer)
    # 完全相同的恢复重放只返回幂等成功，不再次写文件。
    assert resumed.append_router_round(
        1,
        1,
        {
            "round_index": 2,
            "decision": "evidence_ready",
            "problem_type": None,
            "source_file": "round_002/evaluate_scientific_evidence.json",
        },
    )
    # 同一业务轮次但来源改变属于冲突，不能覆盖历史。
    assert not resumed.append_router_round(
        1,
        1,
        {
            "round_index": 2,
            "decision": "evidence_ready",
            "problem_type": None,
            "source_file": "round_002/replayed_response.json",
        },
    )
    summary = _read(
        writer.resolve_output_path(
            "05_router/batch_001/combination_001/summary.json"
        )
    )
    assert [item["round_index"] for item in summary["rounds"]] == [1, 2]
    assert len(summary["rounds"]) == 2
    assert summary["rounds"][1]["source_file"] == "round_002/evaluate_scientific_evidence.json"


def test_router_first_published_round_can_start_at_two_and_keep_order(tmp_path):
    """Router 首次展示轮次可从 2 开始，后续仍按同轮或下一轮收束。"""

    writer = _writer(tmp_path, "diagnostic-publisher-router-start-two")
    publisher = ModelRunViewPublisher(writer)
    round_two = {
        "round_index": 2,
        "decision": "revise_workflow",
        "problem_type": "execution_evidence_problem",
        "source_file": "round_002/evaluate_scientific_evidence.json",
    }
    assert publisher.append_router_round(1, 1, round_two)
    assert publisher.append_router_round(1, 1, dict(round_two))
    assert not publisher.append_router_round(
        1,
        1,
        {
            "round_index": 1,
            "decision": "revise_workflow",
            "problem_type": "execution_evidence_problem",
            "source_file": "round_001/replayed_response.json",
        },
    )
    assert not publisher.append_router_round(
        1,
        1,
        {
            "round_index": 4,
            "decision": "evidence_ready",
            "problem_type": None,
            "source_file": "round_004/evaluate_scientific_evidence.json",
        },
    )
    assert publisher.append_router_round(
        1,
        1,
        {
            "round_index": 3,
            "decision": "evidence_ready",
            "problem_type": None,
            "source_file": "round_003/evaluate_scientific_evidence.json",
        },
    )
    summary = _read(
        writer.resolve_output_path(
            "05_router/batch_001/combination_001/summary.json"
        )
    )
    assert [item["round_index"] for item in summary["rounds"]] == [2, 3]


def test_revision_first_published_round_still_requires_one(tmp_path):
    """Revision 首次展示轮次仍必须从 1 开始。"""

    writer = _writer(tmp_path, "diagnostic-publisher-revision-start-two")
    publisher = ModelRunViewPublisher(writer)
    assert not publisher.append_revision_round(
        1,
        1,
        {
            "round_index": 2,
            "source_file": "round_002/revise_workflow.json",
            "workflow_version_before": 1,
            "workflow_version_after": 2,
            "normalized_patch": {"operations": []},
        },
    )
    assert not writer.resolve_output_path(
        "06_revision/batch_001/combination_001/summary.json"
    ).exists()


def test_revision_append_preserves_gap_and_bad_view_returns_false(tmp_path):
    writer = _writer(tmp_path, "diagnostic-publisher-revision")
    publisher = ModelRunViewPublisher(writer)
    gap = {
        "result_type": "tool_capability_gap",
        "missing_capabilities": ["修订缺口"],
        "reason": "保留 Revision 的能力缺口理由。",
        "source_stage": "revision",
    }
    assert publisher.append_revision_round(
        2,
        3,
        {
            "round_index": 1,
            "source_file": "round_001/revise_workflow.json",
            "final_output": gap,
            "workflow_version_before": 1,
        },
    )
    summary = _read(
        writer.resolve_output_path(
            "06_revision/batch_002/combination_003/summary.json"
        )
    )
    assert summary["final_status"] == "tool_capability_gap"
    assert summary["rounds"][0]["tool_capability_gap"] == gap

    assert not publisher.append_router_round(
        2,
        3,
        {"round_index": 2, "source_file": "missing-decision.json"},
    )
    assert writer.result_summary()["status"] == "degraded"


def test_revision_append_reloads_projected_round_before_adding_next_round(tmp_path):
    """第二轮 Revision 必须能重读第一轮已经精简过的摘要。"""

    writer = _writer(tmp_path, "diagnostic-publisher-revision-chain")
    publisher = ModelRunViewPublisher(writer)
    assert publisher.append_revision_round(
        1,
        2,
        {
            "round_index": 1,
            "source_file": "round_001/revise_workflow.json",
            "workflow_version_before": 1,
            "workflow_version_after": 2,
            "normalized_patch": {"operations": []},
        },
    )
    assert publisher.append_revision_round(
        1,
        2,
        {
            "round_index": 2,
            "source_file": "round_002/revise_workflow.json",
            "workflow_version_before": 2,
            "workflow_version_after": 3,
            "normalized_patch": {"operations": []},
        },
    )

    summary = _read(
        writer.resolve_output_path(
            "06_revision/batch_001/combination_002/summary.json"
        )
    )
    assert [item["round_index"] for item in summary["rounds"]] == [1, 2]
    assert summary["final_status"] == "revision_applied"
    assert writer.result_summary()["status"] == "available"


def test_publisher_rejects_out_of_order_round_without_creating_a_hole(tmp_path):
    writer = _writer(tmp_path, "diagnostic-publisher-order")
    publisher = ModelRunViewPublisher(writer)
    assert publisher.append_router_round(
        1,
        1,
        {
            "round_index": 1,
            "decision": "revise_workflow",
            "problem_type": "first_problem",
            "source_file": "round_001/evaluate_scientific_evidence.json",
        },
    )
    assert not publisher.append_router_round(
        1,
        1,
        {
            "round_index": 3,
            "decision": "evidence_ready",
            "problem_type": None,
            "source_file": "round_003/evaluate_scientific_evidence.json",
        },
    )
    summary = _read(
        writer.resolve_output_path(
            "05_router/batch_001/combination_001/summary.json"
        )
    )
    assert [item["round_index"] for item in summary["rounds"]] == [1]


def test_publisher_rejects_runtime_drift_and_source_path_escape(tmp_path):
    writer = _writer(tmp_path, "diagnostic-publisher-boundary")
    publisher = ModelRunViewPublisher(writer)

    assert not publisher.publish_workflow_version(
        {
            "research_task_id": "another-research-task",
            "batch_index": 1,
            "combination_index": 1,
            "source_type": "initial_planning",
            "source_model_file": "04_planning/plan_workflow.json",
            "workflow_version": 1,
            "workflow": {"goal": "工作流原文"},
        }
    )

    writer = _writer(tmp_path, "diagnostic-publisher-path")
    publisher = ModelRunViewPublisher(writer)
    batch = _batch()
    assert not publisher.publish_hypothesis_summary(
        batch,
        [str(Path(Path.cwd().anchor) / "tmp" / "outside.json")],
        [],
    )

    writer = _writer(tmp_path, "diagnostic-publisher-path-relative")
    publisher = ModelRunViewPublisher(writer)
    assert not publisher.publish_hypothesis_summary(
        batch,
        ["../../../../../../outside.json"],
        [],
    )

    writer = _writer(tmp_path, "diagnostic-publisher-path-slash")
    publisher = ModelRunViewPublisher(writer)
    assert not publisher.publish_hypothesis_summary(
        batch,
        ["generation\\query_001.json"],
        [],
    )

    writer = _writer(tmp_path, "diagnostic-publisher-model-source-path")
    publisher = ModelRunViewPublisher(writer)
    assert not publisher.publish_workflow_version(
        {
            "research_task_id": "diagnostic-publisher-model-source-path",
            "batch_index": 1,
            "combination_index": 1,
            "source_type": "initial_planning",
            "source_model_file": str(
                Path(Path.cwd().anchor) / "tmp" / "outside-model-stage.json"
            ),
            "workflow_version": 1,
            "workflow": {"goal": "工作流原文"},
        }
    )


def test_workflow_versions_require_same_combination_previous_version(tmp_path):
    writer = _writer(tmp_path, "diagnostic-publisher-version-chain")
    publisher = ModelRunViewPublisher(writer)
    assert publisher.publish_workflow_version(
        {
            "research_task_id": "diagnostic-publisher-version-chain",
            "batch_index": 1,
            "combination_index": 1,
            "source_type": "initial_planning",
            "source_model_file": "04_planning/plan_workflow.json",
            "workflow_version": 1,
            "workflow": {"goal": "首版工作流"},
        }
    )
    assert publisher.publish_workflow_version(
        {
            "research_task_id": "diagnostic-publisher-version-chain",
            "batch_index": 1,
            "combination_index": 1,
            "source_type": "revision_patch_applied",
            "source_model_file": "06_revision/round_001/revise_workflow.json",
            "workflow_version_before": 1,
            "workflow_version_after": 2,
            "normalized_patch": {"replace_steps": [{"step_id": "s1"}]},
            "after_workflow": {"goal": "第二版工作流"},
        }
    )

    # version_004 不能跨过 version_003，且不能把别的 combination 的 version_001 当作前版。
    assert not publisher.publish_workflow_version(
        {
            "research_task_id": "diagnostic-publisher-version-chain",
            "batch_index": 1,
            "combination_index": 1,
            "source_type": "revision_patch_applied",
            "source_model_file": "06_revision/round_003/revise_workflow.json",
            "workflow_version_before": 3,
            "workflow_version_after": 4,
            "normalized_patch": {"replace_steps": [{"step_id": "s1"}]},
            "after_workflow": {"goal": "第四版工作流"},
        }
    )
    assert not publisher.publish_workflow_version(
        {
            "research_task_id": "diagnostic-publisher-version-chain",
            "batch_index": 1,
            "combination_index": 2,
            "source_type": "revision_patch_applied",
            "source_model_file": "06_revision/round_001/revise_workflow.json",
            "workflow_version_before": 1,
            "workflow_version_after": 2,
            "normalized_patch": {"replace_steps": [{"step_id": "s1"}]},
            "after_workflow": {"goal": "另一组合第二版"},
        }
    )
