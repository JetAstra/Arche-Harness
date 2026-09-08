from __future__ import annotations

from copy import deepcopy

import pytest

from chemistry_multiagent.controllers.result_view import (
    build_disabled_bounded_result,
    build_enabled_bounded_result,
    strip_internal_claims_for_bounded_result,
)


def _claim_hypothesis(*, reasoning=None) -> dict:
    """构造带原子主张和确定性兼容文本的正式假设。"""

    claims = [
        {
            "statement": "当前策略能够得到可判断的计算证据",
            "rationale": "通过两个版本的计算工作流检验",
        }
    ]
    canonical_reasoning = (
        "[c1] 当前策略能够得到可判断的计算证据\n"
        "[r1] 通过两个版本的计算工作流检验"
    )
    return {
        "hypothesis_id": "hypothesis-1",
        "strategy_name": "当前成功策略",
        "claims": claims,
        "verification_requirements": [
            "执行两个版本的计算工作流并记录支持或反对该主张的观测"
        ],
        "core_claim": "[c1] 当前策略能够得到可判断的计算证据",
        "reasoning": reasoning if reasoning is not None else canonical_reasoning,
        "detailed_reasoning": canonical_reasoning,
        "confidence": 0.7,
        "computation_profile": {"method": "PBE1PBE"},
        "gaussian_constraints": {"charge": 0},
        "metadata": {"legacy": True},
    }


def test_bounded_projection_removes_only_valid_internal_hypothesis_fields() -> None:
    """兼容结果隐藏内部假设字段，但工具同名字段必须原样保留。"""

    value = {
        "final_conclusion": {"selected_strategy": _claim_hypothesis()},
        "tool_result": {
            "hypothesis_id": "工具生成的标签",
            "strategy_name": "工具采用的算法",
            "reasoning": "工具返回的解释文本",
            "measurement": 12.5,
            "units": "kcal/mol",
            "status": "completed",
        },
    }
    original = deepcopy(value)

    projected = strip_internal_claims_for_bounded_result(value)

    selected = projected["final_conclusion"]["selected_strategy"]
    assert "claims" not in selected
    assert "verification_requirements" not in selected
    assert "confidence" not in selected
    assert "computation_profile" not in selected
    assert "gaussian_constraints" not in selected
    assert "metadata" not in selected
    assert projected["tool_result"] == value["tool_result"]
    assert value == original


def test_bounded_projection_rejects_drifted_hypothesis_compatibility_text() -> None:
    value = {
        "final_conclusion": {
            "selected_strategy": _claim_hypothesis(reasoning="已经漂移的兼容文本")
        }
    }

    try:
        strip_internal_claims_for_bounded_result(value)
    except ValueError as exc:
        assert "假设" in str(exc) or "claims" in str(exc)
    else:
        raise AssertionError("派生文本漂移的正式假设不应发布")


def _workflow(version: int) -> dict:
    """构造两个版本可区分、但都符合执行投影需要的最小工作流。"""

    return {
        "workflow_name": f"受控工作流 v{version}",
        "goal": "验证最终结果投影。",
        "strategy_name": "当前成功策略",
        "steps": [
            {
                "step_id": "step-1",
                "step_name": "受控计算",
                "description": "完成一项可验证计算。",
                "tool": "mock_tool",
                "job_type": "none",
                "inputs": '{"bindings":[],"literal":{}}',
                "expected_outputs": '{"outputs":[]}',
                "success_criteria": "取得结构化结果。",
                "error_handling_hint": "技术失败时暂停。",
                "depends_on": [],
            }
        ],
        "expected_outcomes": [],
        "check_bindings": [],
    }


def _execution(
    *,
    overall_status: str,
    success_rate: float,
) -> dict:
    """构造带有 MiniChem 步骤原始输出的最小 ExecutionResult。"""

    step = {
        "step_id": "step-1",
        "step_name": "受控计算",
        "description": "完成一项可验证计算。",
        "tool_name": "mock_tool",
        "status": "completed",
        "raw_output": {"success": overall_status == "success"},
    }
    return {
        "strategy_name": "当前成功策略",
        "total_steps": 1,
        "successful_steps": int(overall_status == "success"),
        "failed_steps": int(overall_status != "success"),
        "success_rate": success_rate,
        "total_duration": 0.0,
        "steps": [step],
        "issues": [] if overall_status == "success" else ["第一轮证据不足"],
        "final_output": {"success": overall_status == "success"},
        "workflow_name": "受控工作流",
        "overall_status": overall_status,
        "summary": {"validation_overview": {"matched": int(overall_status == "success")}},
        "workflow_outcome": "completed" if overall_status == "success" else "structured_failure",
        "intermediate_artifacts": [],
        "metadata": {},
    }


def _package() -> dict:
    """给纯装配函数提供一条先修订、后成功的完整组合历史。"""

    first_execution = _execution(
        overall_status="failed",
        success_rate=0.0,
    )
    second_execution = _execution(overall_status="success", success_rate=1.0)
    return {
        "scientific_question": "验证最终结果投影。",
        "workflow_start_time": "2026-07-20T00:00:00+00:00",
        "task_id": "task-1",
        "task_config": {
            "num_queries": 1,
            "num_hypotheses_per_query": 2,
            "top_n_workflows": 2,
            "max_parallel_combinations": 2,
            "max_workflow_revisions_per_combination": 2,
            "max_hypothesis_batches": 3,
        },
        "chemistry_context": {"charge": 0, "multiplicity": 1},
        "retrieval_phase": {"literature_review": "存在可用文献背景。"},
        "hypothesis_phase": {
            "ranked_strategies": [
                {
                    "hypothesis_id": "hypothesis-1",
                    "strategy_name": "当前成功策略",
                    "reasoning": "通过两个版本的计算工作流检验。",
                    "confidence": 0.7,
                },
                {
                    "hypothesis_id": "hypothesis-2",
                    "strategy_name": "同批其他策略",
                    "reasoning": "用于确认当前结果不能固定取排名第一项。",
                    "confidence": 0.6,
                },
            ]
        },
        "hypothesis": {
            "hypothesis_id": "hypothesis-1",
            "strategy_name": "当前成功策略",
            "reasoning": "通过两个版本的计算工作流检验。",
            "confidence": 0.7,
        },
        "planning_record": {
            "record_id": "planning-record-1",
            "created_at": "2026-07-20T00:01:00+00:00",
            "payload": {"workflow": _workflow(1)},
        },
        "attempt_records": [
            {
                "record_id": "attempt-record-1",
                "attempt_index": 1,
                "created_at": "2026-07-20T00:02:00+00:00",
                "payload": {
                    "workflow_version": 1,
                    "execution_result": first_execution,
                    "router_decision": {
                        "decision": "revise_workflow",
                        "diagnosis": {
                            "problem_type": "workflow_inadequate",
                            "summary": "第一版工作流没有得到足够证据。",
                            "evidence_refs": [],
                            "affected_step_ids": ["step-1"],
                            "claim_assessments": [
                                {
                                    "claim_ref": (
                                        "/combination/hypothesis/claims/0/statement"
                                    ),
                                    "status": "insufficient",
                                    "evidence_refs": [],
                                    "reason": "第一轮缺少直接证据。",
                                }
                            ],
                        },
                        "confidence": 0.8,
                    },
                },
            },
            {
                "record_id": "attempt-record-2",
                "attempt_index": 2,
                "created_at": "2026-07-20T00:03:00+00:00",
                "payload": {
                    "workflow_version": 2,
                    "execution_result": second_execution,
                    "router_decision": {
                        "decision": "evidence_ready",
                        "diagnosis": {
                            "problem_type": "none",
                            "summary": "第二版工作流已经得到充分证据。",
                            "evidence_refs": [
                                "/program_validation/registered_check_results/0/actual"
                            ],
                            "affected_step_ids": [],
                            "claim_assessments": [
                                {
                                    "claim_ref": (
                                        "/combination/hypothesis/claims/0/statement"
                                    ),
                                    "status": "directly_supported",
                                    "evidence_refs": [
                                        "/program_validation/registered_check_results/0/actual"
                                    ],
                                    "reason": "第二轮已有直接登记证据。",
                                }
                            ],
                        },
                        "confidence": 0.9,
                    },
                },
            },
        ],
        "revision_records": [
            {
                "record_id": "revision-record-1",
                "created_at": "2026-07-20T00:02:30+00:00",
                "revision_type": "model_workflow_revision",
                "source_attempt_record_id": "attempt-record-1",
                "payload": {
                    "workflow_version_before": 1,
                    "workflow_version_after": 2,
                    "normalized_patch": {"operations": [{"op": "update_step"}]},
                    "after_workflow": _workflow(2),
                },
            }
        ],
        "pause_restart_records": [],
        "terminal_record_id": "attempt-record-2",
        "batch_combination_outcomes": [
            {
                "combination_index": 1,
                "terminal_state": "evidence_ready",
                "terminal_reason": "router_evidence_ready",
                "tool_capability_gap": None,
            },
            {
                "combination_index": 2,
                "terminal_state": "hypothesis_failed",
                "terminal_reason": "router_revise_hypothesis",
                "tool_capability_gap": None,
            },
            {
                "combination_index": 3,
                "terminal_state": "tool_capability_gap",
                "terminal_reason": "tool_capability_gap",
                "tool_capability_gap": {
                    "result_type": "tool_capability_gap",
                    "unmet_verification_requirements": [
                        "测量候选的实验瞬态光谱并记录可判断观测"
                    ],
                    "missing_capabilities": [
                        "缺少能够测量实验瞬态光谱的已登记工具能力"
                    ],
                    "reason": (
                        "冻结工具目录只能产生计算结果，无法取得所需实验瞬态光谱观测。"
                    ),
                },
            },
        ],
    }


def test_bounded_result_rejects_boolean_combination_index() -> None:
    """组合序号必须是 JSON 整数，不能利用 bool 与 int 相等绕过校验。"""

    package = _package()
    package["batch_combination_outcomes"] = [
        {
            "combination_index": True,
            "terminal_state": "evidence_ready",
            "terminal_reason": "router_evidence_ready",
            "tool_capability_gap": None,
        }
    ]

    try:
        build_enabled_bounded_result(
            package,
            close_timestamp="2026-07-20T00:04:00+00:00",
            summarize_execution=_summarize_execution,
        )
    except ValueError as exc:
        assert "组合序号" in str(exc)
    else:
        raise AssertionError("布尔值不能作为 combination_index")


def _summarize_execution(execution_result: dict) -> dict:
    """模拟 Controller 既有执行摘要接口，保留本测试关心的字段。"""

    status = execution_result["overall_status"]
    return {
        "overall_success_rate": execution_result["success_rate"],
        "workflow_outcome": execution_result["workflow_outcome"],
        "overall_status": status,
        "validation_overview": execution_result["summary"]["validation_overview"],
        "failed_steps": [] if status == "success" else [{"step_id": "step-1"}],
        "issues": list(execution_result["issues"]),
    }


def test_enabled_bounded_result_preserves_original_shape_and_current_combination() -> None:
    """最终 JSON 只绑定当前成功组合，并逐轮展示真实执行与修订。"""

    package = _package()
    original = deepcopy(package)
    result = build_enabled_bounded_result(
        package,
        close_timestamp="2026-07-20T00:04:00+00:00",
        summarize_execution=_summarize_execution,
    )

    assert set(result) == {
        "scientific_question",
        "workflow_start_time",
        "workflow_version",
        "task_id",
        "memory_status",
        "memory_db_path",
        "shared_state",
        "retrieval_phase",
        "hypothesis_phase",
        "planning_rounds",
        "execution_rounds",
        "reflection_rounds",
        "retrieval_followup_rounds",
        "revision_events",
        "stop_conditions",
        "final_conclusion",
        "status",
        "workflow_end_time",
        "total_duration_seconds",
        "batch_combination_outcomes",
    }
    assert result["batch_combination_outcomes"] == package[
        "batch_combination_outcomes"
    ]
    assert "workflow" not in result["batch_combination_outcomes"][2]
    assert "attempt_records" not in result["batch_combination_outcomes"][2]
    assert "tool_results" not in result["batch_combination_outcomes"][2]
    assert result["workflow_version"] == "bounded_closed_loop_v1"
    assert result["task_id"] == "task-1"
    assert result["memory_status"] == "available"
    assert result["memory_db_path"] is None
    assert "selected_strategy_profile" not in result["shared_state"]
    assert "output_files" not in result
    assert "proposal_validation" not in result
    assert [item["round"] for item in result["planning_rounds"]] == [1, 2]
    assert [
        item["result"]["optimized_protocols"][0]["workflow_name"]
        for item in result["planning_rounds"]
    ] == ["受控工作流 v1", "受控工作流 v2"]
    assert [item["status"] for item in result["execution_rounds"]] == [
        "failed",
        "success",
    ]
    assert [item["decision"] for item in result["reflection_rounds"]] == [
        "revise_workflow",
        "accept",
    ]
    assert all(
        "claim_assessments" not in item["result"]["diagnosis"]
        for item in result["reflection_rounds"]
    )
    assert result["execution_rounds"][0]["result"]["steps"][0]["raw_output"] == {
        "success": False
    }
    assert "gaussian_analysis_history" not in result["shared_state"]
    assert "expert_review_history" not in result["shared_state"]
    assert all(
        "expert_backend_audit" not in item
        for item in (
            *result["planning_rounds"],
            *result["execution_rounds"],
            *result["reflection_rounds"],
            *result["revision_events"],
        )
    )
    assert "expert_backend_audit_summary" not in result["final_conclusion"]
    assert package == original
    assert len(result["revision_events"]) == 1
    assert result["revision_events"][0]["round"] == 1
    assert result["stop_conditions"] == {
        "max_reflection_rounds": 3,
        "max_unrecoverable_failures": 0,
        "triggered_condition": "reflection_accept",
        "final_round": 2,
    }
    conclusion = result["final_conclusion"]
    assert conclusion["selected_strategy"]["hypothesis_id"] == "hypothesis-1"
    assert conclusion["conclusion_summary"] == (
        "The selected bounded hypothesis was supported by the recorded computational "
        "evidence; this conclusion does not by itself establish the full scientific "
        "question."
    )
    assert "successfully addressed" not in conclusion["conclusion_summary"]
    assert "confidence" not in conclusion["selected_strategy"]
    assert set(result["hypothesis_phase"]) == {
        "queries",
        "hypotheses_by_query",
        "ranked_strategies",
    }
    assert all(
        "confidence" not in item
        for item in result["hypothesis_phase"]["ranked_strategies"]
    )
    assert conclusion["workflow_outcome"]["total_rounds"] == 2
    assert conclusion["evidence_summary"] == {
        "literature_review_available": True,
        "hypotheses_generated": 2,
        "workflows_executed": 2,
        "successful_executions": 1,
    }
    assert conclusion["total_reflection_rounds"] == 2
    assert result["workflow_start_time"] == 1784505600.0
    assert result["workflow_end_time"] == 1784505840.0
    assert result["total_duration_seconds"] == 240.0


def test_historical_human_revision_renders_without_removed_fields() -> None:
    """旧 SQLite 人工修订可以只读渲染，结果不重新引入旧审计字段。"""

    package = _package()
    revision = package["revision_records"][0]
    revision.update(
        {
            "revision_type": "human_workflow_patch",
            "source_attempt_record_id": None,
            "source_pause_reason": "tool_execution_failure",
            "applied_attempt_index": 1,
        }
    )
    revision["payload"] = {
        "workflow_version_before": 1,
        "workflow_version_after": 2,
        "before_workflow": _workflow(1),
        "normalized_patch": {"operations": [{"op": "update_step"}]},
        "after_workflow": _workflow(2),
    }

    result = build_enabled_bounded_result(
        package,
        close_timestamp="2026-07-20T00:04:00+00:00",
        summarize_execution=_summarize_execution,
    )

    event = result["revision_events"][0]
    assert event["round"] == 1
    assert event["identified_problems"] == ["tool_execution_failure"]
    assert event["workflow_revision_instructions"] == [
        revision["payload"]["normalized_patch"]
    ]
    assert event["revised_result"] == revision["payload"]["after_workflow"]

    def assert_no_removed_fields(value: object) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                assert "gaussian" not in key.lower()
                assert "expert" not in key.lower()
                assert_no_removed_fields(nested)
        elif isinstance(value, list):
            for nested in value:
                assert_no_removed_fields(nested)

    assert_no_removed_fields(result)


def test_disabled_bounded_result_only_changes_memory_identity_fields() -> None:
    """关闭记忆仍保留完整科学结果，只移除正式任务身份和旧校验字段。"""

    package = _package()
    package["task_id"] = None

    result = build_disabled_bounded_result(
        package,
        close_timestamp="2026-07-20T00:04:00+00:00",
        summarize_execution=_summarize_execution,
    )

    assert result["task_id"] is None
    assert result["memory_status"] == "disabled"
    assert result["memory_db_path"] is None
    assert result["planning_rounds"]
    assert result["execution_rounds"]
    assert result["reflection_rounds"]
    assert "selected_strategy_profile" not in result["shared_state"]
    assert "proposal_validation" not in result


def test_bounded_result_rejects_terminal_record_that_is_not_final_attempt() -> None:
    """终态记录若不指向最后一个 evidence_ready Attempt，不能发布结果。"""

    package = _package()
    package["terminal_record_id"] = "attempt-record-1"

    try:
        build_enabled_bounded_result(
            package,
            close_timestamp="2026-07-20T00:04:00+00:00",
            summarize_execution=_summarize_execution,
        )
    except ValueError as exc:
        assert "终态" in str(exc)
    else:
        raise AssertionError("错误终态记录不应生成 bounded JSON")


def test_bounded_result_rejects_malformed_execution_step() -> None:
    """最终发布不得静默丢弃损坏的步骤记录并继续给出成功结论。"""

    package = _package()
    broken = deepcopy(package["attempt_records"][0]["payload"]["execution_result"])
    broken["steps"] = [None]
    package["attempt_records"][0]["payload"]["execution_result"] = broken

    try:
        build_enabled_bounded_result(
            package,
            close_timestamp="2026-07-20T00:04:00+00:00",
            summarize_execution=_summarize_execution,
        )
    except ValueError as exc:
        assert "步骤" in str(exc)
    else:
        raise AssertionError("损坏的执行步骤不应生成 bounded JSON")


def test_bounded_result_rejects_minichem_terminal_attempt_without_program_gate() -> None:
    """当前 MiniChem 终态缺 Gate 时不能沿用 Router 的旧 evidence_ready。"""

    package = _package()
    terminal_execution = package["attempt_records"][-1]["payload"]["execution_result"]
    terminal_execution["steps"][0]["execution_layer"] = "predefined_actions"
    with pytest.raises(ValueError, match="MiniChem|program_validation"):
        build_enabled_bounded_result(
            package,
            close_timestamp="2026-07-20T00:04:00+00:00",
            summarize_execution=_summarize_execution,
        )


def test_bounded_result_rechecks_terminal_gate_before_publishing() -> None:
    """即使 Router 已保存 evidence_ready，受限 Gate 也必须阻止最终发布。"""

    package = _package()
    package["attempt_records"][-1]["payload"]["program_validation"] = {
        "registered_check_results": [],
        "program_gate_result": {
            "allowed_decisions": ["revise_workflow"],
            "constraint_reasons": ["artifact_status_invalid"],
        },
    }
    with pytest.raises(ValueError, match="allowed_decisions|Gate"):
        build_enabled_bounded_result(
            package,
            close_timestamp="2026-07-20T00:04:00+00:00",
            summarize_execution=_summarize_execution,
        )


def test_bounded_result_rejects_invalid_declaration_gate_before_publishing() -> None:
    """声明校验失败不能被旧 Router 决定覆盖。"""

    package = _package()
    package["attempt_records"][-1]["payload"]["program_validation"] = {
        "registered_check_results": [
            {
                "check_ref": "c1",
                "evaluation": "matched",
                "actual": {"artifact_id": "art-1"},
                "evidence_ref": "/execution_result/steps/0/raw_output/result",
                "diagnostic": None,
            }
        ],
        "program_gate_result": {
            "allowed_decisions": ["revise_workflow"],
            "constraint_reasons": ["declaration_validation_failed"],
            "declaration_checks": [
                {"validation_status": "invalid", "errors": ["missing required artifact"]}
            ],
        },
    }
    with pytest.raises(ValueError, match="declaration_checks|Gate"):
        build_enabled_bounded_result(
            package,
            close_timestamp="2026-07-20T00:04:00+00:00",
            summarize_execution=_summarize_execution,
        )
