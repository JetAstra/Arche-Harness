from __future__ import annotations

from copy import deepcopy

import pytest

from chemistry_multiagent.utils.model_run_views import (
    ModelRunViewError,
    construct_hypothesis_outcomes,
    construct_hypothesis_summary,
    construct_revision_summary,
    construct_router_summary,
    construct_workflow_version,
)


def _hypothesis(hypothesis_id: str, strategy_name: str) -> dict:
    """构造带有可识别原文的候选假设。"""

    return {
        "hypothesis_id": hypothesis_id,
        "strategy_name": strategy_name,
        "core_claim": f"  {hypothesis_id} 的核心判断\n",
        "reasoning": f"{hypothesis_id} 的理由原文，保留末尾空格  ",
    }


def _batch() -> dict:
    """构造排名和候选顺序不同的正式批次。"""

    h1 = _hypothesis("h-1", "策略一")
    h2 = _hypothesis("h-2", "策略二")
    h3 = _hypothesis("h-3", "策略三")
    return {
        "batch_index": 1,
        "hypothesis_phase": {
            "queries": ["角度一", "角度二"],
            "hypotheses_by_query": [
                {"query": "角度一", "hypotheses": [h1, h2]},
                {"query": "角度二", "hypotheses": [h3]},
            ],
            "ranked_strategies": [
                deepcopy(h2),
                deepcopy(h1),
                deepcopy(h3),
            ],
        },
        # 故意不按 ranked_strategies 的数组位置填写，视图必须按 hypothesis_id 关联。
        "candidates": [
            {
                "hypothesis_id": "h-3",
                "rank": 3,
                "selection_status": "not_selected",
                "combination_id": None,
                "combination_index": None,
            },
            {
                "hypothesis_id": "h-1",
                "rank": 2,
                "selection_status": "selected",
                "combination_id": "comb-1",
                "combination_index": 1,
            },
            {
                "hypothesis_id": "h-2",
                "rank": 1,
                "selection_status": "selected",
                "combination_id": "comb-2",
                "combination_index": 2,
            },
        ],
        "source_files": [
            "generation/query_001.json",
            "relationships/comparison_001.json",
            "ranking/final_ranking.json",
        ],
    }


def _outcome_sources() -> dict[str, dict[str, str | None]]:
    return {
        hypothesis_id: {
            "generation": f"generation/{hypothesis_id}.json",
            "planning": (
                f"../../04_planning/{hypothesis_id}.json"
                if hypothesis_id != "h-3"
                else None
            ),
            "router": (
                f"../../05_router/{hypothesis_id}.json"
                if hypothesis_id == "h-2"
                else None
            ),
        }
        for hypothesis_id in ("h-1", "h-2", "h-3")
    }


def test_hypothesis_summary_preserves_order_and_scientific_text() -> None:
    batch = _batch()
    original = deepcopy(batch)
    merge = [{"left_id": "h-1", "right_id": "h-2", "decision": "merge"}]

    view = construct_hypothesis_summary(batch, merge_decisions=merge)

    assert view["queries"] == ["角度一", "角度二"]
    assert [item["hypothesis_id"] for item in view["generated_hypotheses"]] == [
        "h-1",
        "h-2",
        "h-3",
    ]
    assert [item["hypothesis_id"] for item in view["ranked_hypotheses"]] == [
        "h-2",
        "h-1",
        "h-3",
    ]
    assert [item["hypothesis_id"] for item in view["selected_hypotheses"]] == [
        "h-2",
        "h-1",
    ]
    assert view["ranked_hypotheses"][0]["core_claim"] == "  h-2 的核心判断\n"
    assert view["ranked_hypotheses"][0]["reasoning"].endswith("  ")
    view["ranked_hypotheses"][0]["core_claim"] = "被调用方修改"
    assert batch == original


def test_hypothesis_outcomes_join_by_id_and_keep_revision_gap() -> None:
    batch = _batch()
    gap = {
        "result_type": "tool_capability_gap",
        "missing_capabilities": ["受控扫描能力原文"],
        "reason": "  Revision 返回的原始能力缺口理由  ",
        "source_stage": "revision",
    }
    facts = [
        {
            "hypothesis_id": "h-1",
            "combination_index": 1,
            "state": "tool_capability_gap",
            "terminal_reason": "tool_capability_gap",
            "tool_capability_gap": gap,
        },
        {
            "hypothesis_id": "h-2",
            "combination_index": 2,
            "state": "evidence_ready",
            "terminal_reason": "router_evidence_ready",
            "tool_capability_gap": None,
        },
    ]

    view = construct_hypothesis_outcomes(
        batch,
        facts,
        source_files=_outcome_sources(),
    )
    by_id = {item["hypothesis_id"]: item for item in view["hypotheses"]}

    assert by_id["h-1"]["selection_status"] == "selected"
    assert by_id["h-1"]["combination_index"] == 1
    assert by_id["h-1"]["terminal_state"] == "tool_capability_gap"
    assert by_id["h-1"]["tool_capability_gap"] == gap
    assert by_id["h-2"]["combination_index"] == 2
    assert by_id["h-3"]["selection_status"] == "not_selected"
    assert by_id["h-3"]["terminal_state"] == "not_executed"
    assert by_id["h-3"]["tool_capability_gap"] is None
    assert by_id["h-1"]["core_claim"] == "  h-1 的核心判断\n"

    by_id["h-1"]["tool_capability_gap"]["reason"] = "被调用方修改"
    assert facts[0]["tool_capability_gap"] == gap


def test_workflow_versions_keep_patch_separate_from_full_workflow() -> None:
    initial_workflow = {
        "workflow_name": "首版路线",
        "goal": "保留这段目标原文  ",
        "steps": [{"step_id": "s1", "description": "首步"}],
    }
    initial = construct_workflow_version(
        {
            "research_task_id": "task-1",
            "batch_index": 1,
            "combination_index": 1,
            "source_type": "initial_planning",
            "source_model_file": "04_planning/plan_workflow.json",
            "workflow_version": 1,
            "workflow": initial_workflow,
        }
    )
    assert initial["workflow_version"] == 1
    assert initial["previous_version"] is None
    assert initial["workflow"] == initial_workflow

    revision = construct_workflow_version(
        {
            "research_task_id": "task-1",
            "batch_index": 1,
            "combination_index": 1,
            "source_type": "revision_patch_applied",
            "source_model_file": "06_revision/round_001/revise_workflow.json",
            "workflow_version_before": 1,
            "workflow_version_after": 2,
            "normalized_patch": {"replace_steps": [{"step_id": "s1"}]},
            "after_workflow": {
                **initial_workflow,
                "steps": [{"step_id": "s1", "description": "修订后的首步"}],
            },
        }
    )
    assert revision["source_type"] == "revision_patch_applied"
    assert revision["workflow_version"] == 2
    assert revision["previous_version"] == "version_001.json"
    assert "normalized_patch" not in revision
    assert revision["workflow"]["goal"] == "保留这段目标原文  "


def test_router_and_revision_rounds_require_program_order_and_keep_gap_verbatim() -> None:
    with pytest.raises(ModelRunViewError, match="实际顺序"):
        construct_router_summary(
            {
                "batch_index": 1,
                "combination_index": 1,
                "rounds": [
                    {
                        "round_index": 2,
                        "router_decision": {
                            "decision": "evidence_ready",
                            "diagnosis": {"problem_type": None},
                        },
                        "source_file": "round_002/evaluate_scientific_evidence.json",
                    },
                    {
                        "round_index": 1,
                        "decision": "revise_workflow",
                        "problem_type": "execution_evidence_problem",
                        "source_file": "round_001/evaluate_scientific_evidence.json",
                    },
                ],
            }
        )

    router = construct_router_summary(
        {
            "batch_index": 1,
            "combination_index": 1,
            "rounds": [
                {
                    "round_index": 1,
                    "decision": "revise_workflow",
                    "problem_type": "execution_evidence_problem",
                    "source_file": "round_001/evaluate_scientific_evidence.json",
                },
                {
                    "round_index": 2,
                    "router_decision": {
                        "decision": "evidence_ready",
                        "diagnosis": {"problem_type": None},
                    },
                    "source_file": "round_002/evaluate_scientific_evidence.json",
                },
            ],
        }
    )
    assert [item["round_index"] for item in router["rounds"]] == [1, 2]
    assert router["final_decision"] == "evidence_ready"

    gap = {
        "result_type": "tool_capability_gap",
        "missing_capabilities": ["新的验证能力"],
        "reason": "Revision 无法完成所需验证，保留原文。",
        "source_stage": "revision",
    }
    with pytest.raises(ModelRunViewError, match="实际顺序"):
        construct_revision_summary(
            {
                "batch_index": 1,
                "combination_index": 1,
                "rounds": [
                    {
                        "round_index": 2,
                        "source_file": "round_002/revise_workflow.json",
                        "final_output": gap,
                        "workflow_version_before": 2,
                    },
                    {
                        "round_index": 1,
                        "source_file": "round_001/revise_workflow.json",
                        "workflow_version_before": 1,
                        "workflow_version_after": 2,
                        "normalized_patch": {"replace_steps": []},
                    },
                ],
            }
        )

    revision = construct_revision_summary(
        {
            "batch_index": 1,
            "combination_index": 1,
            "rounds": [
                {
                    "round_index": 1,
                    "source_file": "round_001/revise_workflow.json",
                    "workflow_version_before": 1,
                    "workflow_version_after": 2,
                    "normalized_patch": {"replace_steps": []},
                },
                {
                    "round_index": 2,
                    "source_file": "round_002/revise_workflow.json",
                    "final_output": gap,
                    "workflow_version_before": 2,
                },
            ],
        }
    )
    assert [item["round_index"] for item in revision["rounds"]] == [1, 2]
    assert revision["final_status"] == "tool_capability_gap"
    assert revision["rounds"][1]["tool_capability_gap"] == gap


def test_incomplete_identity_fails_explicitly() -> None:
    batch = _batch()
    del batch["candidates"][0]["hypothesis_id"]
    with pytest.raises(ModelRunViewError, match="hypothesis_id"):
        construct_hypothesis_summary(batch)


def test_existing_workflow_version_fields_must_form_a_continuous_chain() -> None:
    valid = construct_router_summary(
        {
            "batch_index": 1,
            "combination_index": 1,
            "rounds": [
                {
                    "round_index": 1,
                    "decision": "revise_workflow",
                    "problem_type": "first_problem",
                    "source_file": "round_001/router.json",
                    "workflow_version_before": 1,
                    "workflow_version_after": 2,
                },
                {
                    "round_index": 2,
                    "decision": "evidence_ready",
                    "problem_type": None,
                    "source_file": "round_002/router.json",
                    "workflow_version_before": 2,
                    "workflow_version_after": 3,
                },
            ],
        }
    )
    assert valid["rounds"][1]["workflow_version_before"] == 2

    with pytest.raises(ModelRunViewError, match="同时提供"):
        construct_router_summary(
            {
                "batch_index": 1,
                "combination_index": 1,
                "rounds": [
                    {
                        "round_index": 1,
                        "decision": "evidence_ready",
                        "problem_type": None,
                        "source_file": "round_001/router.json",
                        "workflow_version_before": 1,
                    }
                ],
            }
        )

    with pytest.raises(ModelRunViewError, match="不连续"):
        construct_revision_summary(
            {
                "batch_index": 1,
                "combination_index": 1,
                "rounds": [
                    {
                        "round_index": 1,
                        "source_file": "round_001/revision.json",
                        "workflow_version_before": 1,
                        "workflow_version_after": 2,
                        "normalized_patch": {"replace_steps": []},
                    },
                    {
                        "round_index": 2,
                        "source_file": "round_002/revision.json",
                        "workflow_version_before": 3,
                        "workflow_version_after": 4,
                        "normalized_patch": {"replace_steps": []},
                    },
                ],
            }
        )

    with pytest.raises(ModelRunViewError, match="正整数"):
        construct_revision_summary(
            {
                "batch_index": 1,
                "combination_index": 1,
                "rounds": [
                    {
                        "round_index": 1,
                        "source_file": "round_001/revision.json",
                        "final_output": {
                            "result_type": "tool_capability_gap",
                            "missing_capabilities": ["缺口能力"],
                            "reason": "缺口理由原文。",
                        },
                        "workflow_version_before": 0,
                    }
                ],
            }
        )
