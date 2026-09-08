"""Memory 阶段视图和 Gaussian 专家视图的跨模式契约测试。

这些测试只检查模型实际能够看到的固定对象，不依赖数据库中的正式身份或本地
运行产物。测试夹具故意放入机器身份、路径、原始输出和控制轨迹，确保 allowlist
投影不会把审计事实误送给科学模型。
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Mapping

from chemistry_multiagent.memory.projections import (
    project_hypothesis_regeneration_context,
    project_hypothesis_terminal_outcome,
    project_judge_fact,
    project_program_gate_fact,
    project_scientific_attempt_ledger,
    project_workflow_for_revision,
    project_workflow_revision_repair,
)
from chemistry_multiagent.utils.expert_views import (
    POST_VIEW_KEYS,
    PRE_VIEW_KEYS,
    STEP_VIEW_KEYS,
    build_gaussian_step_review_view,
    build_workflow_scientific_evidence_view,
    flatten_workflow_evidence_index,
)
from chemistry_multiagent.utils.model_prompts import build_model_messages
from tests.fakes.minichem_mcp import build_catalog_snapshot


_LONG_REASONING = "完整科学推理：" + ("该机理通过结构化观测检验。" * 500)
_LONG_DIAGNOSTIC = "失败诊断中的科学事实：" + ("缺少可判定观测。" * 500)
_FORBIDDEN_MARKERS = (
    "FULL_WORKFLOW_SENTINEL",
    "CONTROL_TRACE_SENTINEL",
    "PERSISTENT_ID_SENTINEL",
    "ABSOLUTE_PATH_SENTINEL",
    "RAW_OUTPUT_SENTINEL",
    "OTHER_COMBINATION_REPAIR_SENTINEL",
    "FORMAL_STEP_ID_SENTINEL",
    "BACKEND_ID_SENTINEL",
)


def _user_base_input(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """解析 build_model_messages 的稳定 user envelope。"""

    user = next(item for item in messages if item["role"] == "user")
    envelope = json.loads(user["content"])
    assert set(envelope) == {
        "base_input",
        "final_check",
        "operation_retry",
        "protocol_error",
        "response_instruction",
    }
    return envelope["base_input"]


def _json_text(value: Any) -> str:
    """把模型消息或视图按稳定 JSON 序列化，便于检查越界字符串。"""

    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _mapping_keys(value: Any) -> set[str]:
    """递归收集对象键，避免用 substring 误判合法字段名。"""

    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, str):
                keys.add(key)
            keys.update(_mapping_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.update(_mapping_keys(item))
    return keys


def test_judge_reason_cannot_leak_raw_log_marker() -> None:
    """裁判派生理由会清理 raw_log 等内部运行标记。"""

    fact = project_judge_fact(
        {
            "decision": "revise_workflow",
            "diagnosis": {"reason": "缺少频率；raw_log 位于内部运行记录。"},
        }
    )

    assert "raw_log" not in fact["reason"]


def test_derived_text_removes_full_runtime_identity_suffix() -> None:
    """派生失败说明不能只移除 identity 字段名而残留 UUID 后缀。"""

    fact = project_judge_fact(
        {
            "decision": "revise_workflow",
            "diagnosis": {
                "reason": (
                    "MiniChem source step step_id-01234567-89ab-cdef-0123-456789abcdef "
                    "selector artifact.semantic_type=SDFFile 命中 2 个 Artifact；"
                    "operation_id-abcdefab-cdef-abcd-efab-cdefabcdefab"
                )
            },
        }
    )

    assert "artifact.semantic_type=SDFFile" in fact["reason"]
    assert "命中 2 个" in fact["reason"]
    assert "step_id-01234567-89ab-cdef-0123-456789abcdef" not in fact["reason"]
    assert "abcdefab-cdef-abcd-efab-cdefabcdefab" not in fact["reason"]


def test_attempt_checks_and_program_gate_clean_derived_internal_markers() -> None:
    """Attempt 检查和 Gate 派生值不能携带运行身份或日志标记。"""

    ledger = project_scientific_attempt_ledger(
        {
            "overall_status": "completed",
            "workflow_outcome": "completed",
            "steps": [
                {
                    "status": "completed",
                    "checks": [
                        {"normal_termination": "job_id=abc raw_log"},
                        {"converged": "runtime=5"},
                    ],
                }
            ],
        }
    )
    gate = project_program_gate_fact(
        {
            "program_gate_result": {
                "allowed_decisions": ["revise_workflow"],
                "constraint_reasons": [
                    "operation_id=abc raw_log runtime=5",
                    "formal-step-id:execution_failed",
                ],
            },
            "registered_check_results": [
                {
                    "evaluation": "raw_log",
                    "requirement": {"expected": "operation_id=abc"},
                    "actual": {"status": "raw_log", "job_id": "abc"},
                }
            ],
        },
        workflow={"steps": [{"step_id": "formal-step-id"}]},
    )

    joined = _json_text({"ledger": ledger, "gate": gate})
    for marker in ("job_id=abc", "operation_id=abc", "raw_log", "runtime=5"):
        assert marker not in joined
    assert gate["checks"][0]["status"] == "not_run"
    assert gate["constraint_reasons"][1] == "s1:execution_failed"
    assert "formal-step-id" not in joined


def test_failed_attempt_projects_actionable_details_with_stable_shape() -> None:
    """失败账本保留可修复诊断，缺失字段仍使用固定空值且不泄漏身份。"""

    workflow = {
        "steps": [
            {
                "step_id": "formal-step",
                "step_name": "检查输入",
                "description": "检查化学输入",
                "action_id": "validate_structure",
            }
        ]
    }
    execution = {
        "overall_status": "failed",
        "workflow_outcome": "stopped_after_error",
        "steps": [
            {
                "step_id": "formal-step",
                "status": "failed",
                "retryable": True,
                "error": {
                    "code": "backend_input_error",
                    "message": "元素不守恒：C 计数不一致；文件=/tmp/failure.log",
                    "failure_stage": "backend_input",
                    "repair_guidance": "修正输入后重试，不要原样提交。",
                    "input_requirements": [
                        "提供完整结构",
                        "/mnt/private/request.json",
                    ],
                    "retryable_after_input_correction": True,
                    "retry_requires_changed_request": True,
                    "status": "invalid_request",
                    "diagnostic_code": "charge_mismatch",
                    "reason_code": "operation_id=hidden",
                    "traceback": "TRACEBACK_SENTINEL",
                    "action_id": "ACTION_ID_SENTINEL",
                    "backend_id": "BACKEND_ID_SENTINEL",
                    "inspect_action_request": {"action_id": "ACTION_ID_SENTINEL"},
                    "unknown_field": "UNKNOWN_SENTINEL",
                },
            }
        ],
    }

    ledger = project_scientific_attempt_ledger(execution, workflow=workflow)
    details = ledger["steps"][0]["failure"]["details"]
    assert set(details) == {
        "stage",
        "message",
        "repair_guidance",
        "input_requirements",
        "retryable",
        "retryable_after_input_correction",
        "retry_requires_changed_request",
        "status",
        "diagnostic_code",
        "reason_code",
    }
    assert details["stage"] == "backend_input"
    assert details["message"].startswith("元素不守恒：C 计数不一致")
    assert details["repair_guidance"] == "修正输入后重试，不要原样提交。"
    assert details["input_requirements"] == ["提供完整结构", "[removed-path]"]
    assert details["retryable"] is True
    assert details["retryable_after_input_correction"] is True
    assert details["retry_requires_changed_request"] is True
    assert details["status"] == "invalid_request"
    assert details["diagnostic_code"] == "charge_mismatch"
    assert details["reason_code"] is None
    joined = _json_text(ledger)
    for marker in (
        "TRACEBACK_SENTINEL",
        "ACTION_ID_SENTINEL",
        "BACKEND_ID_SENTINEL",
        "UNKNOWN_SENTINEL",
        "operation_id=hidden",
        "/tmp/failure.log",
    ):
        assert marker not in joined


def test_revision_ledger_keeps_bounded_runtime_diagnostic_as_repair_guidance() -> None:
    """运行时已批准的错误字段和候选修复必须进入 Revision，不得只剩笼统消息。"""

    workflow = {
        "steps": [
            {
                "step_id": "gaussian-step",
                "step_name": "运行 Gaussian",
                "description": "验证 Gaussian route",
                "execution_layer": "native_software",
                "software_id": "gaussian",
                "request": {},
            }
        ]
    }
    execution = {
        "overall_status": "failed",
        "workflow_outcome": "stopped_after_error",
        "steps": [
            {
                "step_id": "gaussian-step",
                "status": "failed",
                "error": {
                    "code": "gaussian_route_syntax_error",
                    "message": "Gaussian 拒绝了 route section 的输入语法（QPErr）。",
                    "details": {
                        "rejected_route_field": "SCRF.solvent",
                        "rejected_value": "EthylAcetate",
                        "recommended_value": "EthylEthanoate",
                        "internal_path": "/hidden/job/stdout.log",
                    },
                    "candidate_fixes": [
                        "乙酸乙酯使用 Gaussian 原生别名 EthylEthanoate，不使用 EthylAcetate。"
                    ],
                    "traceback": "TRACEBACK_SENTINEL",
                },
            }
        ],
    }

    ledger = project_scientific_attempt_ledger(execution, workflow=workflow)
    repair_guidance = ledger["steps"][0]["failure"]["details"][
        "repair_guidance"
    ]

    assert "rejected_route_field=SCRF.solvent" in repair_guidance
    assert "rejected_value=EthylAcetate" in repair_guidance
    assert "recommended_value=EthylEthanoate" in repair_guidance
    assert "乙酸乙酯使用 Gaussian 原生别名 EthylEthanoate" in repair_guidance
    assert "/hidden/job/stdout.log" not in json.dumps(ledger, ensure_ascii=False)
    assert "TRACEBACK_SENTINEL" not in json.dumps(ledger, ensure_ascii=False)

    revision_input = _revision_input()
    revision_input["current_attempt"]["scientific_ledger"] = ledger
    visible = _user_base_input(
        build_model_messages(
            "workflow_revision",
            revision_input,
            memory_enabled=True,
        )
    )
    visible_guidance = visible["current_attempt"]["scientific_ledger"]["steps"][
        0
    ]["failure"]["details"]["repair_guidance"]
    assert "rejected_route_field=SCRF.solvent" in visible_guidance
    assert "recommended_value=EthylEthanoate" in visible_guidance


def test_tool_capability_gap_reason_cleans_derived_internal_markers() -> None:
    """工具能力缺口理由保留科学说明，但不泄漏内部运行标记。"""

    outcome = project_hypothesis_terminal_outcome(
        source_query="比较溶剂效应",
        hypothesis={"strategy_name": "溶剂策略", "reasoning": "科学推理"},
        terminal_classification="tool_capability_gap",
        terminal_reason="缺少所需工具",
        tool_capability_gap={
            "missing_capabilities": ["solvation_energy"],
            "reason": "缺少能力；operation_id=op raw_log runtime=5",
        },
    )

    reason = outcome["tool_capability_gap"]["reason"]
    assert reason.startswith("缺少能力；")
    for marker in ("operation_id=op", "raw_log", "runtime=5"):
        assert marker not in reason


def test_generic_scientific_observation_cleans_embedded_internal_path() -> None:
    """注册观测值保留 route 科学内容，但移除工具结果中的嵌入路径。"""

    ledger = project_scientific_attempt_ledger(
        {
            "overall_status": "completed",
            "workflow_outcome": "completed",
            "steps": [
                {
                    "step_id": "route-step",
                    "status": "completed",
                    "raw_output": {
                        "result": {
                            "gaussian_code": "#p B3LYP/6-31G(d); output=/tmp/route.gjf"
                        }
                    },
                }
            ],
        },
        workflow={
            "steps": [
                {
                    "step_id": "route-step",
                    "step_name": "生成 route",
                    "description": "生成 Gaussian route",
                    "action_id": "generate_gaussian_code",
                }
            ]
        },
    )

    observation = ledger["steps"][0]["observations"][0]
    assert observation["name"] == "gaussian_route"
    assert observation["value"].startswith("#p B3LYP/6-31G(d)")
    assert "/tmp/route.gjf" not in observation["value"]


def test_stage_derived_failure_paths_cover_all_absolute_path_forms() -> None:
    """失败派生事实统一移除 POSIX、Windows、UNC 和 file URI 路径。"""

    paths = (
        "/private/secret/my run/boom.json",
        "/data/secret/boom.json",
        r"C:\Program Files\Gaussian\g16.exe",
        r"\\server\share\Program Files\g16.exe",
        "file:///private/secret/my file.log",
    )
    message = "；".join(paths) + "；参考 https://server/share；方法 B3LYP/6-31G(d)"
    ledger = project_scientific_attempt_ledger(
        {
            "overall_status": "failed",
            "workflow_outcome": "failed",
            "steps": [
                {
                    "step_id": "formal-step",
                    "status": "failed",
                    "error": {
                        "category": "execution",
                        "code": "failed",
                        "message": message,
                    },
                }
            ],
        },
        workflow={
            "steps": [
                {
                    "step_id": "formal-step",
                    "action_id": "run_native_quantum_chemistry",
                }
            ]
        },
    )
    serialized = _json_text(ledger)
    for path in paths:
        assert path not in serialized
    for leaked_suffix in (
        "run/boom.json",
        r"Files\Gaussian\g16.exe",
        r"Files\g16.exe",
        "file.log",
    ):
        assert leaked_suffix not in serialized
    assert "https://server/share" in serialized
    assert "B3LYP/6-31G(d)" in serialized


def test_stage_derived_path_replacement_preserves_text_after_file_boundary() -> None:
    """含空格路径替换到文件边界后，后续普通句子必须原样保留。"""

    ledger = project_scientific_attempt_ledger(
        {
            "overall_status": "failed",
            "workflow_outcome": "failed",
            "steps": [
                {
                    "step_id": "formal-step",
                    "status": "failed",
                    "error": {
                        "category": "execution",
                        "message": r"读取 C:\Program Files\Gaussian\g16.exe 后继续检查。",
                    },
                }
            ],
        },
        workflow={
            "steps": [
                {
                    "step_id": "formal-step",
                    "action_id": "run_native_quantum_chemistry",
                }
            ]
        },
    )

    message = ledger["steps"][0]["failure"]["details"]["message"]
    assert message.endswith("[removed-path] 后继续检查。")


def test_revision_open_request_paths_are_replaced_without_touching_scientific_slashes() -> None:
    """Revision 的公开 open request 不回显绝对路径，科学 route 和 URL 保持完整。"""

    paths = (
        "/private/secret/input.xyz",
        r"C:\Program Files\Gaussian\g16.exe",
        r"\\server\share\secret\g16.exe",
        "file:///data/secret/output.log",
    )
    projected = project_workflow_for_revision(
        {
            "steps": [
                {
                    "step_id": "formal-native",
                    "step_name": "native step",
                    "description": "执行 B3LYP/6-31G(d)；参考 https://server/share",
                    "execution_layer": "native_software",
                    "request": {
                        "software_id": paths[1],
                        "executable": paths[2],
                        "runtime": paths[0],
                        "arguments": [*paths, "B3LYP/6-31G(d)", "https://server/share"],
                        "calculation_intent": "energy / volume",
                        "label": paths[3],
                    },
                }
            ]
        }
    )
    serialized = _json_text(projected)
    for path in paths:
        assert path not in serialized
    assert "B3LYP/6-31G(d)" in serialized
    assert "https://server/share" in serialized
    assert "energy / volume" in serialized


def _regeneration_context() -> dict[str, Any]:
    """返回冻结生成上下文；其中嵌入只允许保留的科学字段和越界哨兵。"""

    retrieval = {
        "keywords": ["亲核加成", "立体选择性"],
        "literature_review": "检索结论保留完整科学语句。",
        "mechanistic_clues": ["过渡态构象差异"],
        "limitations": ["尚无直接能垒观测"],
        "chemistry_context": {"solvent": "acetone", "temperature_c": 30},
        "workflow": "FULL_WORKFLOW_SENTINEL",
        "control_trace": "CONTROL_TRACE_SENTINEL",
        "source_path": "/mnt/secret/ABSOLUTE_PATH_SENTINEL",
        "raw_output": "RAW_OUTPUT_SENTINEL",
    }
    return {
        "scientific_context": {
            "scientific_question": "解释反应的立体选择性。",
            "initial_query_angles": ["催化剂构象", "溶剂效应"],
            "initial_retrieval_phase": retrieval,
            "chemistry_context": {"substrate": "benzaldehyde", "temperature_c": 30},
            "operation_id": "PERSISTENT_ID_SENTINEL",
        },
        "current_query": "当前 Query：比较两个催化剂构象。",
        "prior_hypothesis_outcomes": [
            {
                "source_query": "构象控制 Query",
                "hypothesis": {
                    "strategy_name": "构象差异策略",
                    "reasoning": _LONG_REASONING,
                    "hypothesis_id": "PERSISTENT_ID_SENTINEL",
                },
                "terminal_classification": "reliably_refuted",
                "final_judge": {
                    "reason": "观察到关键构象能量顺序相反。",
                },
                "cited_evidence": [],
                "attempted_observables": ["相对能量"],
                "missing_evidence": ["溶剂化自由能"],
                "tool_capability_gap": None,
                "workflow": "FULL_WORKFLOW_SENTINEL",
                "control_trace": "CONTROL_TRACE_SENTINEL",
                "path": "/tmp/ABSOLUTE_PATH_SENTINEL",
                "raw_output": "RAW_OUTPUT_SENTINEL",
            },
            {
                "source_query": "溶剂效应 Query",
                "hypothesis": {
                    "strategy_name": "溶剂稳定化策略",
                    "reasoning": "该策略仍缺少溶剂化比较。",
                },
                "terminal_classification": "tool_capability_gap",
                "final_judge": {
                    "reason": "工具无法提供目标溶剂化观测。",
                },
                "cited_evidence": [],
                "attempted_observables": ["气相能量"],
                "missing_evidence": ["溶剂化能量"],
                "tool_capability_gap": {
                    "result_type": "solvation_free_energy",
                    "missing_capabilities": ["solvent_model"],
                    "reason": "当前工具目录没有该能力。",
                },
                "combination_id": "PERSISTENT_ID_SENTINEL",
            },
        ],
        "next_batch_index": 3,
        "generation_config": {
            "num_queries": 1,
            "num_hypotheses_per_query": 2,
            "top_n_workflows": 1,
        },
    }


def _revision_repair(*, marker: str, combination: str) -> dict[str, Any]:
    """构造一条修复记录；other combination 记录不应穿过当前视图。"""

    return {
        "combination": combination,
        "patch": [{"op": "update_step", "step_ref": "s1", "field": "request"}],
        "affected_workflow_before": {"steps": [{"step_ref": "s1", "name": "旧方案"}]},
        "affected_workflow_after": {"steps": [{"step_ref": "s1", "name": marker}]},
        "attempt_ledger": {"steps": [], "judge": {"reason": "证据仍不足"}},
        "judge": {"reason": "需要补充频率观测", "evidence_refs": ["e1"]},
        "combination_id": combination,
        "record_id": "PERSISTENT_ID_SENTINEL",
        "raw_output": "RAW_OUTPUT_SENTINEL",
        "path": "/tmp/ABSOLUTE_PATH_SENTINEL",
    }


def _revision_input() -> dict[str, Any]:
    """返回当前 combination 的 revision 输入及一条应排除的其他组合修复。"""

    return {
        "scientific_context": {
            "scientific_question": "解释反应的立体选择性。",
            "chemistry_context": {"solvent": "acetone", "temperature_c": 30},
        },
        "hypothesis": {
            "strategy_name": "当前构象策略",
            "reasoning": _LONG_REASONING,
            "hypothesis_id": "PERSISTENT_ID_SENTINEL",
        },
        "current_workflow": {
            "steps": [
                {
                    "step_ref": "s1",
                    "step_name": "计算能量",
                    "description": "取得相对能量观测。",
                    "step_id": "FORMAL_STEP_ID_SENTINEL",
                    "action_id": "energy",
                    "backend_id": "gaussian",
                    "request": {"path": "/tmp/ABSOLUTE_PATH_SENTINEL"},
                }
            ],
            "workflow_id": "PERSISTENT_ID_SENTINEL",
        },
        "current_attempt": {
            "scientific_ledger": {
                "steps": [
                    {
                        "step_ref": "s1",
                        "status": "failed",
                        "observations": {"relative_energy": 1.2},
                        "failure": {"category": "execution", "code": "missing_observable"},
                    }
                ]
            },
            "program_gate": {"status": "allowed", "checks": []},
            "judge": {"decision": "insufficient_evidence", "reason": _LONG_DIAGNOSTIC},
            "execution_id": "PERSISTENT_ID_SENTINEL",
            "raw_output": "RAW_OUTPUT_SENTINEL",
        },
        "tool_catalog_projection": [
            {
                "action_id": "energy",
                "action_version": "1.0.0",
                "category": "electronic_structure",
                "description": "计算相对能量",
                "backend_ids": ["gaussian"],
                "required_inputs": [],
                "optional_inputs": [],
                "selection_policy": "agent_backend_required",
                "primary_output": "energy",
                "data_action": False,
            }
        ],
        # Working Memory 在联接阶段先按当前 combination 过滤；模型层只应接收
        # 过滤后的修复数组。另一个 combination 的记录放在未注册字段中，防止
        # adapter 以“差不多的对象”把它重新带回消息。
        "prior_repairs": [_revision_repair(marker="目标修复", combination="target")],
        "other_combination_repairs": [
            _revision_repair(
                marker="OTHER_COMBINATION_REPAIR_SENTINEL", combination="other"
            )
        ],
    }


def test_regeneration_memory_on_off_only_changes_prior_outcomes() -> None:
    """Memory 开关只改变历史数组，当前科学事实必须逐字节一致。"""

    base = _regeneration_context()
    enabled = _user_base_input(
        build_model_messages(
            "hypothesis_generation",
            base,
            stage="regeneration",
            memory_enabled=True,
        )
    )
    disabled = _user_base_input(
        build_model_messages(
            "hypothesis_generation",
            base,
            stage="regeneration",
            memory_enabled=False,
        )
    )

    expected_keys = {
        "scientific_context",
        "current_query",
        "prior_hypothesis_outcomes",
        "next_batch_index",
        "generation_config",
        "hypothesis_output_schema",
    }
    assert set(enabled) == expected_keys
    assert set(disabled) == expected_keys
    enabled_current = deepcopy(enabled)
    disabled_current = deepcopy(disabled)
    enabled_history = enabled_current.pop("prior_hypothesis_outcomes")
    disabled_history = disabled_current.pop("prior_hypothesis_outcomes")
    assert disabled_history == []
    assert enabled_history
    assert json.dumps(enabled_current, ensure_ascii=False, sort_keys=True) == json.dumps(
        disabled_current, ensure_ascii=False, sort_keys=True
    )
    assert _LONG_REASONING in _json_text(enabled)
    joined = _json_text(enabled)
    for marker in _FORBIDDEN_MARKERS:
        assert marker not in joined
    assert "selected_evaluated_history" not in joined
    regeneration_keys = _mapping_keys(enabled)
    assert "workflow" not in regeneration_keys
    assert not regeneration_keys.intersection(
        {"steps", "request", "control_trace", "pause", "restart"}
    )


def test_regeneration_projection_keeps_fixed_outcome_semantics() -> None:
    """再生成历史区分反驳、证据不足和工具能力缺口，且不携带控制轨迹。"""

    scientific_question = "  比较用户指定的 /tmp/input.xyz。  \n"
    context = project_hypothesis_regeneration_context(
        scientific_question=scientific_question,
        initial_query_angles=["催化剂构象"],
        initial_retrieval_phase={
            "keywords": ["构象"],
            "literature_review": _LONG_DIAGNOSTIC,
            "mechanistic_clues": ["过渡态"],
            "limitations": ["观测不足"],
            "workflow": "FULL_WORKFLOW_SENTINEL",
            "raw_output": "RAW_OUTPUT_SENTINEL",
        },
        chemistry_context={"solvent": "acetone", "path": "/tmp/ABSOLUTE_PATH_SENTINEL"},
    )
    outcome = project_hypothesis_terminal_outcome(
        source_query="构象 Query",
        hypothesis={"strategy_name": "策略", "reasoning": _LONG_REASONING},
        terminal_classification="reliably_refuted",
        terminal_reason="关键观测反驳了该策略。",
        execution_result={
            "overall_status": "failed",
            "steps": [
                {
                    "step_id": "FORMAL_STEP_ID_SENTINEL",
                    "status": "failed",
                    "error": {
                        "message": (
                            "source step step_id-01234567-89ab-cdef-0123-456789abcdef "
                            "selector artifact.semantic_type=SDFFile 命中 2 个；"
                            "operation_instance_id-abcdefab-cdef-abcd-efab-cdefabcdefab"
                        )
                    },
                    "raw_output": {
                        "status": "failed",
                        "result": {"energy": -1.2, "raw_output": "RAW_OUTPUT_SENTINEL"},
                        "path": "/tmp/ABSOLUTE_PATH_SENTINEL",
                    },
                }
            ],
            "operation_id": "PERSISTENT_ID_SENTINEL",
        },
        program_validation={
            "program_gate_result": {"allowed_decisions": [], "constraint_reasons": []},
            "registered_check_results": [],
        },
        router_decision={
            "decision": "revise_hypothesis",
            "diagnosis": {
                "summary": _LONG_DIAGNOSTIC,
                "evidence_refs": ["/steps/0/result/energy"],
            },
        },
    )
    assert set(context) == {
        "scientific_question",
        "initial_query_angles",
        "initial_retrieval_phase",
        "chemistry_context",
    }
    assert context["scientific_question"] == scientific_question
    assert outcome["terminal_classification"] == "reliably_refuted"
    assert set(outcome) == {
        "source_query",
        "hypothesis",
        "terminal_classification",
        "final_judge",
        "cited_evidence",
        "attempted_observables",
        "missing_evidence",
        "tool_capability_gap",
    }
    assert set(outcome["final_judge"]) == {"reason"}
    assert outcome["cited_evidence"]
    assert outcome["cited_evidence"][0]["evidence_kind"] == "unavailable"
    assert outcome["cited_evidence"][0]["value"]["status"] == "unavailable"
    assert _LONG_REASONING in _json_text(outcome)
    joined = _json_text({"context": context, "outcome": outcome})
    for marker in _FORBIDDEN_MARKERS:
        assert marker not in joined


def test_revision_memory_on_off_keeps_current_facts_and_same_combination_repairs() -> None:
    """Revision 保留当前事实，只能读取目标组合的修复历史。"""

    base = _revision_input()
    enabled = _user_base_input(
        build_model_messages(
            "workflow_revision",
            base,
            memory_enabled=True,
        )
    )
    disabled = _user_base_input(
        build_model_messages(
            "workflow_revision",
            base,
            memory_enabled=False,
        )
    )
    expected_keys = {
        "scientific_context",
        "hypothesis",
        "current_workflow",
        "current_attempt",
        "tool_catalog_projection",
        "prior_repairs",
        "workflow_revision_output_schema",
    }
    assert set(enabled) == expected_keys
    assert set(disabled) == expected_keys
    enabled_current = deepcopy(enabled)
    disabled_current = deepcopy(disabled)
    enabled_repairs = enabled_current.pop("prior_repairs")
    disabled_repairs = disabled_current.pop("prior_repairs")
    assert disabled_repairs == []
    assert enabled_repairs
    assert all("OTHER_COMBINATION_REPAIR_SENTINEL" not in _json_text(item) for item in enabled_repairs)
    assert all(
            set(item)
            == {
                "patch",
                "affected_workflow_before",
                "affected_workflow_after",
                "attempt_ledger",
                "program_gate",
                "judge",
            }
        for item in enabled_repairs
    )
    assert json.dumps(enabled_current, ensure_ascii=False, sort_keys=True) == json.dumps(
        disabled_current, ensure_ascii=False, sort_keys=True
    )
    assert _LONG_REASONING in _json_text(enabled)
    joined = _json_text(enabled)
    for marker in _FORBIDDEN_MARKERS:
        assert marker not in joined


def test_revision_repair_projector_has_no_machine_identity_or_duplicate_logs() -> None:
    """修复记录只保留 patch、工作流前后值、账本和裁判理由。"""

    repair = project_workflow_revision_repair(
        normalized_patch={
            "operations": [{"op": "update_step", "step_ref": "s1", "field": "request"}],
            "record_id": "PERSISTENT_ID_SENTINEL",
        },
        before_workflow={"steps": [{"step_ref": "s1", "name": "旧方案"}], "path": "/tmp/ABSOLUTE_PATH_SENTINEL"},
        after_workflow={"steps": [{"step_ref": "s1", "name": "新方案"}]},
        execution_result={
            "steps": [
                {
                    "step_id": "FORMAL_STEP_ID_SENTINEL",
                    "status": "failed",
                    "raw_output": {
                        "status": "failed",
                        "result": {"energy": -1.2},
                        "raw_output": "RAW_OUTPUT_SENTINEL",
                    },
                }
            ],
            "operation_id": "PERSISTENT_ID_SENTINEL",
        },
        program_validation={"program_gate_result": {}, "registered_check_results": []},
        router_decision={
            "decision": "insufficient_evidence",
            "diagnosis": {"summary": _LONG_DIAGNOSTIC, "evidence_refs": ["/steps/0"]},
        },
    )
    assert set(repair) == {
        "patch",
        "affected_workflow_before",
        "affected_workflow_after",
        "attempt_ledger",
        "program_gate",
        "judge",
    }
    assert _LONG_DIAGNOSTIC in _json_text(repair)
    joined = _json_text(repair)
    for marker in _FORBIDDEN_MARKERS:
        assert marker not in joined
    for marker in (
        "step_id-01234567-89ab-cdef-0123-456789abcdef",
        "abcdefab-cdef-abcd-efab-cdefabcdefab",
    ):
        assert marker not in joined


def test_revision_repair_maps_canonical_update_without_losing_binding_context() -> None:
    """canonical target_step_id + changes 只影响目标步骤并保留上游短引用。"""

    before_request = {
        "inputs": {"charge": 0},
        "bindings": [
            {
                "source_step_ref": "formal-source",
                "source_field": "/result/energy",
                "target_field": "inputs.energy",
            }
        ],
    }
    after_request = deepcopy(before_request)
    after_request["inputs"]["charge"] = 1
    before = {
        "steps": [
            {"step_id": "formal-source", "step_name": "source"},
            {
                "step_id": "formal-target",
                "step_name": "target",
                "depends_on": ["formal-source"],
                "request": before_request,
            },
            {"step_id": "formal-untouched", "step_name": "untouched"},
        ]
    }
    after = deepcopy(before)
    after["steps"][1]["request"] = after_request
    repair = project_workflow_revision_repair(
        normalized_patch={
            "operations": [
                {
                    "op": "update_step",
                    "target_step_id": "formal-target",
                    "changes": {"request": after_request},
                }
            ]
        },
        before_workflow=before,
        after_workflow=after,
        execution_result={"steps": []},
        program_validation={"program_gate_result": {}, "registered_check_results": []},
        router_decision={},
    )

    assert len(repair["patch"]) == 1
    operation = repair["patch"][0]
    assert operation["operation"] == "update_step"
    assert operation["step_ref"] == "s2"
    assert operation["field"] == "request"
    assert operation["before"]["bindings"][0]["source_step_ref"] == "s1"
    assert operation["after"]["bindings"][0]["source_step_ref"] == "s1"
    assert [item["step_ref"] for item in repair["affected_workflow_before"]] == ["s2"]
    assert [item["step_ref"] for item in repair["affected_workflow_after"]] == ["s2"]
    assert repair["affected_workflow_after"][0]["depends_on"] == ["s1"]
    assert "formal-untouched" not in _json_text(repair)


def _gaussian_step(*, with_structure: bool) -> dict[str, Any]:
    """构造执行前 Gaussian 单元，分别覆盖有结构和不可用结构。"""

    request: dict[str, Any] = {
        "inputs": {
            "charge": 0,
            "multiplicity": 1,
            "title": "催化剂复合物",
            "nprocshared": 8,
            "mem": "16GB",
            "chk_file": "/mnt/jobs/gaussian.chk",
            "user_input": "route intent: " + ("优化结构并计算频率。" * 500),
        },
        "method_spec": {"method": "wB97XD", "basis": "def2TZVP"},
        "action_settings": {"solvent": "acetone", "temperature": 303.15},
        "path": "/tmp/ABSOLUTE_PATH_SENTINEL",
        "raw_output": "RAW_OUTPUT_SENTINEL",
    }
    step: dict[str, Any] = {
        "step_id": "FORMAL_STEP_ID_SENTINEL",
        "step_name": "Gaussian route",
        "description": "准备并审查 Gaussian 输入。" + ("科学目的。" * 500),
        "request": request,
        "validation_requirements": [],
    }
    if with_structure:
        request["inputs"]["structure_summary"] = {
            "summary_status": "available",
            "role": "catalyst",
            "input_format": "xyz",
            "formula": "C10H12N2O2",
            "atom_count": 26,
            "element_counts": {"C": 10, "H": 12, "N": 2, "O": 2},
        }
    return step


def test_gaussian_pre_view_has_stable_schema_and_preserves_long_science_text() -> None:
    """执行前视图固定字段，空结构也不删除字段，且不截断合法文本。"""

    populated = build_gaussian_step_review_view(
        scientific_question="解释反应的立体选择性。",
        hypothesis={"strategy_name": "构象策略", "reasoning": _LONG_REASONING},
        step=_gaussian_step(with_structure=True),
        route="#p wB97XD/def2TZVP opt freq " + ("grid=fine " * 500),
        converter_kind="xyz",
    )
    unavailable = build_gaussian_step_review_view(
        scientific_question="解释反应的立体选择性。",
        hypothesis={"strategy_name": "构象策略", "reasoning": _LONG_REASONING},
        step=_gaussian_step(with_structure=False),
        route="#p wB97XD/def2TZVP opt freq",
        converter_kind="xyz",
    )
    assert set(populated) == set(PRE_VIEW_KEYS)
    assert set(unavailable) == set(PRE_VIEW_KEYS)
    assert set(populated["hypothesis"]) == {"strategy_name", "reasoning"}
    assert set(populated["gaussian_unit"]) == {
        "scientific_purpose",
        "calculation_type",
        "structure",
        "requested_route_intent",
        "generated_route",
        "editable_parameters",
        "validation_requirements",
        "mutable_fields",
    }
    assert set(populated["gaussian_unit"]["editable_parameters"]) == {
        "charge",
        "multiplicity",
        "title",
        "nprocshared",
        "mem",
        "chk_file",
    }
    assert populated["hypothesis"]["reasoning"] == _LONG_REASONING
    assert len(populated["gaussian_unit"]["generated_route"]) > 2048
    assert unavailable["gaussian_unit"]["structure"] == {
        "summary_status": "unavailable",
        "role": None,
        "input_format": "xyz",
        "formula": None,
        "atom_count": None,
        "element_counts": {},
    }
    joined = _json_text({"populated": populated, "unavailable": unavailable})
    for marker in _FORBIDDEN_MARKERS:
        assert marker not in joined
    assert "[omitted:" not in joined


def _evidence_workflow() -> tuple[dict[str, Any], dict[str, Any]]:
    """构造三步执行后证据，其中含成功、失败和未执行步骤。"""

    workflow = {
        "steps": [
            {
                "step_id": "step-route",
                "step_name": "生成 route",
                "description": "生成实际 Gaussian route。",
                "action_id": "generate_gaussian_code",
                "backend_id": "BACKEND_ID_SENTINEL",
                "depends_on": [],
                "request": {"path": "/tmp/ABSOLUTE_PATH_SENTINEL"},
            },
            {
                "step_id": "step-failed",
                "step_name": "解析结果",
                "description": "解析能量和频率。",
                "action_id": "parse_quantum_chemistry_output",
                "backend_id": "BACKEND_ID_SENTINEL",
                "depends_on": ["step-route"],
            },
            {
                "step_id": "step-unknown",
                "step_name": "未注册分析",
                "description": "检查未注册科学结果。",
                "action_id": "unregistered_action",
                "backend_id": "BACKEND_ID_SENTINEL",
                "depends_on": ["step-failed"],
            },
        ]
    }
    execution = {
        "steps": [
            {
                "step_id": "step-route",
                "status": "completed",
                "raw_output": {
                    "status": "completed",
                    "result": {"gaussian_code": "#p B3LYP/6-31G(d) opt freq"},
                    "raw_output": "RAW_OUTPUT_SENTINEL",
                    "artifact_id": "PERSISTENT_ID_SENTINEL",
                },
            },
            {
                "step_id": "step-failed",
                "status": "failed",
                "error": {
                    "category": "parser",
                    "code": "missing_frequency",
                    "details": {"diagnostic": _LONG_DIAGNOSTIC, "path": "/tmp/ABSOLUTE_PATH_SENTINEL"},
                },
                "raw_output": {"status": "failed", "raw_output": "RAW_OUTPUT_SENTINEL"},
            },
            {
                "step_id": "step-unknown",
                "status": "not_executed",
                "raw_output": {"status": "not_executed"},
            },
        ],
        "operation_id": "PERSISTENT_ID_SENTINEL",
    }
    return workflow, execution


def test_gaussian_post_view_is_step_complete_and_uses_embedded_evidence_refs() -> None:
    """执行后每步只出现一次；证据引用内嵌，旧 evidence_index 不进入专家视图。"""

    workflow, execution = _evidence_workflow()
    view, legacy_index = build_workflow_scientific_evidence_view(
        scientific_question="解释反应的立体选择性。",
        hypothesis={"strategy_name": "构象策略", "reasoning": _LONG_REASONING},
        workflow=workflow,
        execution_results=execution,
    )
    assert set(view) == set(POST_VIEW_KEYS)
    steps = view["workflow_evidence"]["steps"]
    assert len(steps) == len(workflow["steps"]) == 3
    assert [step["step_ref"] for step in steps] == ["s1", "s2", "s3"]
    assert all(set(step) == set(STEP_VIEW_KEYS) for step in steps)
    assert len({step["step_ref"] for step in steps}) == len(steps)
    assert steps[0]["observations"]
    assert steps[1]["failure"] is not None
    assert steps[2]["coverage"]["status"] == "unavailable"
    assert steps[2]["not_executed"] is not None
    assert "missing_observables" not in steps[1]["failure"]
    assert "missing_observables" not in steps[2]["not_executed"]
    refs = [
        fact["evidence_ref"]
        for step in steps
        for field in ("observations", "program_checks")
        for fact in step[field]
    ]
    refs.extend(
        step[field]["evidence_ref"]
        for step in steps
        for field in ("failure", "not_executed")
        if isinstance(step[field], Mapping)
    )
    assert refs and len(refs) == len(set(refs))
    assert {item["evidence_ref"] for item in legacy_index} == set(refs)
    assert flatten_workflow_evidence_index(view) == legacy_index
    joined = _json_text(view)
    for marker in _FORBIDDEN_MARKERS:
        assert marker not in joined
    assert "evidence_index" not in view["workflow_evidence"]
    assert "RAW_OUTPUT_SENTINEL" not in joined
    route = "#p B3LYP/6-31G(d) opt freq"
    assert joined.count(route) == 1


def test_case1_scientific_ledger_registers_vibrations_and_spectra_without_dense_arrays() -> None:
    """Hypothesis Regeneration/Revision ledger 保留三个 Action 的科学摘要。"""

    workflow = {
        "steps": [
            {
                "step_id": "vib",
                "step_name": "振动模式",
                "description": "从 Hessian 派生频率",
                "action_id": "derive_vibrational_modes",
            },
            {
                "step_id": "ir",
                "step_name": "IR 光谱",
                "description": "构造 IR 线谱",
                "action_id": "derive_ir_spectrum",
                "depends_on": ["vib"],
            },
            {
                "step_id": "raman",
                "step_name": "Raman 光谱",
                "description": "构造 Raman 线谱",
                "action_id": "derive_raman_spectrum",
                "depends_on": ["ir"],
            },
        ]
    }
    execution = {
        "overall_status": "completed",
        "steps": [
            {
                "step_id": "vib",
                "status": "completed",
                "result": {
                    "frequencies_cm1": [
                        {"real": 100.0, "imaginary": 0.0},
                        {"real": 0.0, "imaginary": 80.0},
                    ],
                    "modes": [["MODE_INTERNAL_SENTINEL"]],
                    "vibrational_energies_ev": [1.0, 2.0],
                },
            },
            {
                "step_id": "ir",
                "status": "completed",
                "result": {
                    "schema_name": "SpectrumResult",
                    "axis": {"unit": "inverse_centimeter", "values": [900.0, 1000.0]},
                    "intensity": {"unit": "kilometer_per_mole", "values": [0.0, 1.0]},
                    "sticks": [{"position": 1000.0, "intensity": 8.0}],
                    "line_shape": {"kind": "gaussian", "fwhm": 20.0},
                    "point_count": 2,
                    "stick_count": 1,
                },
            },
            {
                "step_id": "raman",
                "status": "completed",
                "result": {
                    "wavenumber_cm1": [900.0, 1000.0],
                    "intensity": [0.0, 1.0],
                    "line_frequencies_cm1": [1000.0],
                    "line_intensities": [4.5],
                    "broadening": "lorentzian",
                    "fwhm_cm1": 10.0,
                    "point_count": 2,
                    "stick_count": 1,
                },
            },
        ],
    }
    ledger = project_scientific_attempt_ledger(execution, workflow=workflow)
    steps = ledger["steps"]
    assert [step["missing_observables"] for step in steps] == [[], [], []]
    assert [step["observations"][0]["name"] for step in steps] == [
        "vibrational_mode_summary",
        "ir_line_spectrum",
        "raman_line_spectrum",
    ]
    assert steps[0]["observations"][0]["value"]["imaginary_count"] == 1
    assert steps[1]["observations"][0]["value"]["line_frequencies_cm1"] == [1000.0]
    assert steps[2]["observations"][0]["value"]["line_intensities"] == [4.5]
    assert "MODE_INTERNAL_SENTINEL" not in _json_text(ledger)
    assert "axis" not in _mapping_keys(ledger)
    assert "values" not in _mapping_keys(ledger)


def test_case1_scientific_ledger_marks_invalid_spectrum_summary_missing() -> None:
    """source point_count 或 dense 数组不一致时不产生半真线谱事实。"""

    workflow = {
        "steps": [
            {
                "step_id": "ir",
                "step_name": "IR",
                "description": "IR",
                "action_id": "derive_ir_spectrum",
            }
        ]
    }
    execution = {
        "steps": [
            {
                "step_id": "ir",
                "status": "completed",
                "result": {
                    "wavenumber_cm1": [900.0, 1000.0],
                    "intensity": [0.0, 1.0, 0.0],
                    "line_frequencies_cm1": [1000.0],
                    "line_intensities": [8.0],
                    "point_count": 99,
                    "stick_count": 1,
                },
            }
        ]
    }
    ledger = project_scientific_attempt_ledger(execution, workflow=workflow)
    assert ledger["steps"][0]["observations"] == []
    assert ledger["steps"][0]["missing_observables"] == ["ir_line_spectrum"]


def test_revision_ledger_reads_requested_parser_settings_from_request_sections() -> None:
    """Revision 账本读取请求设置，不把同名 parser 结果块标成已确认设置。"""

    workflow = {
        "steps": [
            {
                "step_id": "parser",
                "step_name": "解析",
                "action_id": "parse_quantum_chemistry_output",
                "request": {
                    "method_spec": {
                        "properties": ["atom_coordinates", "energies"],
                        "coordinate_frames": "last",
                    }
                },
            }
        ]
    }
    execution = {
        "steps": [
            {
                "step_id": "parser",
                "status": "completed",
                "result": {
                    "properties": {
                        "atomcoords": [[0.0, 0.1, 0.2]],
                        "scfenergies": [-100.0],
                    }
                },
            }
        ]
    }

    ledger = project_scientific_attempt_ledger(execution, workflow=workflow)
    settings = ledger["steps"][0]["scientific_settings"]
    assert settings["properties"] == {
        "value": ["atom_coordinates", "energies"],
        "source": "requested_only",
    }
    assert settings["coordinate_frames"] == {
        "value": "last",
        "source": "requested_only",
    }


def test_programmable_ledger_uses_declared_outputs_and_reports_missing_only() -> None:
    """可编程账本只接受 request.outputs 中的注册结果，并准确列出缺失项。"""

    workflow = {
        "steps": [
            {
                "step_id": "analysis",
                "execution_layer": "programmable_analysis",
                "request": {
                    "outputs": [
                        {"name": "observed_support"},
                        {"name": "energy"},
                        {"name": "relative_gibbs_free_energies"},
                    ]
                },
            }
        ]
    }
    execution = {
        "steps": [
            {
                "step_id": "analysis",
                "status": "completed",
                "result": {
                    "observed_support": False,
                    "energy": -1.5,
                    "undeclared_result": "HIDDEN_UNDECLARED_RESULT",
                },
            }
        ]
    }

    ledger = project_scientific_attempt_ledger(execution, workflow=workflow)
    step = ledger["steps"][0]
    assert [item["name"] for item in step["observations"]] == [
        "energy",
        "observed_support",
    ]
    assert step["missing_observables"] == ["relative_gibbs_free_energies"]
    assert "HIDDEN_UNDECLARED_RESULT" not in _json_text(ledger)


def test_real_deck_preflight_observation_is_ready_and_not_missing() -> None:
    """真实 deck 结果的顶层 ready 事实能形成 eN 观测而不被误报缺失。"""

    workflow = {
        "steps": [
            {
                "step_id": "deck",
                "action_id": "sdf_to_gjf",
            }
        ]
    }
    execution = {
        "steps": [
            {
                "step_id": "deck",
                "status": "completed",
                "result": {
                    "native_preflight_ready": True,
                    "native_preflight": {
                        "calculation_intent": "single_point",
                        "link1_segment_count": 1,
                        "checks": ["route_valid"],
                    },
                },
            }
        ]
    }

    ledger = project_scientific_attempt_ledger(execution, workflow=workflow)
    step = ledger["steps"][0]
    assert step["missing_observables"] == []
    assert step["observations"] == [
        {
            "evidence_ref": "e1",
            "name": "deck_preflight_status",
            "value": {"status": "ready", "ready": True},
            "unit": None,
            "conditions": {},
        }
    ]


def test_revision_workflow_maps_only_known_binding_refs_and_keeps_foreach_limit() -> None:
    """公开 workflow wire 不回显未知正式引用，且保留合法 foreach 数值。"""

    projected = project_workflow_for_revision(
        {
            "steps": [
                {"step_id": "formal-source", "step_name": "source"},
                {
                    "step_id": "formal-target",
                    "step_name": "target",
                    "depends_on": ["formal-source"],
                    "request": {
                        "bindings": [
                            {
                                "source_step_ref": "formal-source",
                                "source_field": "/result/energy",
                                "target_field": "inputs.energy",
                            },
                            {
                                "source_step_ref": "FORMAL_BINDING_SENTINEL",
                                "source_field": "/result/energy",
                                "target_field": "inputs.other",
                            },
                        ],
                        "foreach": {
                            "source_step_ref": "formal-source",
                            "source_field": "/result/structures",
                            "item_field": "structure",
                            "item_id_field": "label",
                            "target_field": "inputs.structure",
                            "max_items": 12,
                        },
                    },
                },
            ]
        }
    )
    request = projected["steps"][1]["request"]
    assert request["bindings"][0]["source_step_ref"] == "s1"
    assert request["bindings"][1]["source_step_ref"] is None
    assert request["foreach"]["source_step_ref"] == "s1"
    assert request["foreach"]["max_items"] == 12
    assert "FORMAL_BINDING_SENTINEL" not in _json_text(projected)


def test_revision_workflow_package_and_message_clean_derived_header_fields() -> None:
    """持久 Revision package 与最终消息都清理 workflow 头部派生内部片段。"""

    workflow_name = "完整科学工作流说明：" + ("结构与频率分析。" * 300)
    workflow = {
        "workflow_name": workflow_name + " /tmp/REVISION_PATH_SENTINEL",
        "goal": "判断频率结果；operation_id=REVISION_OPERATION_SENTINEL raw_output",
        "strategy_name": "频率策略；job_id=REVISION_JOB_SENTINEL",
        "steps": [
            {
                "step_id": "formal-step",
                "step_name": "生成频率输入",
                "description": "保留科学目的",
                "action_id": "generate_gaussian_code",
                "request": {},
                "depends_on": [],
            }
        ],
    }
    projected = project_workflow_for_revision(workflow)
    assert projected["workflow_name"].startswith("完整科学工作流说明：")
    assert len(projected["workflow_name"]) > 2048
    joined_package = _json_text(projected)
    for marker in (
        "/tmp/REVISION_PATH_SENTINEL",
        "REVISION_OPERATION_SENTINEL",
        "REVISION_JOB_SENTINEL",
        "raw_output",
    ):
        assert marker not in joined_package

    revision_input = _revision_input()
    revision_input["current_workflow"] = workflow
    messages = build_model_messages(
        "workflow_revision",
        revision_input,
        memory_enabled=False,
    )
    joined_messages = _json_text(messages)
    for marker in (
        "/tmp/REVISION_PATH_SENTINEL",
        "REVISION_OPERATION_SENTINEL",
        "REVISION_JOB_SENTINEL",
        "raw_output",
    ):
        assert marker not in joined_messages


def test_revision_programmable_output_schema_filters_internal_extensions() -> None:
    """Analysis output 的 JSON Schema 保留公开结构并清理路径、身份和密钥。"""

    workflow = {
        "workflow_name": "schema workflow",
        "goal": "schema goal",
        "strategy_name": "schema strategy",
        "steps": [
            {
                "step_id": "analysis-step",
                "step_name": "分析",
                "description": "输出科学标量",
                "execution_layer": "programmable_analysis",
                "request": {
                    "runtime": "python",
                    "outputs": [
                        {
                            "name": "energy",
                            "semantic_type": "scalar",
                            "path": "outputs/energy.json",
                            "json_schema": {
                                "type": "object",
                                "description": "能量；/mnt/private/schema.json operation_id=SCHEMA_OPERATION raw_output",
                                "properties": {
                                    "energy": {"type": "number", "description": "hartree"},
                                    "raw_output": {"type": "string"},
                                    "x-internal-id": {"type": "string"},
                                },
                                "required": ["energy", "raw_output", "x-internal-id"],
                                "x-internal-id": "SCHEMA_EXTENSION",
                                "secret": "SCHEMA_SECRET",
                            },
                        }
                    ],
                },
            }
        ],
    }
    projected = project_workflow_for_revision(workflow)
    schema = projected["steps"][0]["request"]["outputs"][0]["json_schema"]
    assert schema["type"] == "object"
    assert schema["properties"] == {
        "energy": {"type": "number", "description": "hartree"}
    }
    assert schema["required"] == ["energy"]
    serialized = _json_text(projected)
    for marker in (
        "/mnt/private/schema.json",
        "SCHEMA_OPERATION",
        "SCHEMA_EXTENSION",
        "SCHEMA_SECRET",
        "raw_output",
        "x-internal-id",
    ):
        assert marker not in serialized

    revision_input = _revision_input()
    revision_input["current_workflow"] = workflow
    messages = build_model_messages(
        "workflow_revision",
        revision_input,
        memory_enabled=False,
    )
    joined_messages = _json_text(messages)
    for marker in (
        "/mnt/private/schema.json",
        "SCHEMA_OPERATION",
        "SCHEMA_EXTENSION",
        "SCHEMA_SECRET",
        "raw_output",
        "x-internal-id",
    ):
        assert marker not in joined_messages


def test_revision_contract_dynamic_strings_fail_closed_for_internal_markers() -> None:
    """pair 合同允许的字符串仍需清理 operation/raw/path，科学 route 正常保留。"""

    catalog = build_catalog_snapshot(
        actions=[
            {
                "id": "map_reaction_atoms",
                "version": "1.0.0",
                "description": "Map reaction.",
                "backend_ids": ["rxnmapper"],
                "required_inputs": ["reaction"],
                "optional_inputs": [],
                "selection_policy": "agent_backend_required",
                "primary_output": "MappedReaction",
                "output_semantic_types": ["MappedReaction"],
                "data_action": False,
            }
        ],
        backends=[
            {
                "id": "rxnmapper",
                "display_name": "RXNMapper",
                "runtime": "python",
                "capabilities": ["map_reaction_atoms"],
                "health": "healthy",
            }
        ],
    )
    catalog["action_contracts"] = {
        "schema_version": 1,
        "catalog_hash": catalog["catalog_hash"],
        "contracts": [
            {
                "action_id": "map_reaction_atoms",
                "backend_id": "rxnmapper",
                "request_contract": {
                    "sections": {
                        "inputs": {"allowed": ["reaction"], "optional": [], "required": ["reaction"]},
                        "method_spec": {"allowed": ["route"], "optional": [], "required": []},
                        "action_settings": {"allowed": [], "optional": [], "required": []},
                        "resource_limits": {"allowed": [], "optional": [], "required": []},
                    }
                },
                "output_contract": {},
            }
        ],
    }
    workflow = {
        "workflow_name": "workflow",
        "goal": "goal",
        "strategy_name": "strategy",
        "steps": [
            {
                "step_id": "map-step",
                "step_name": "反应映射",
                "description": "保留科学 route",
                "action_id": "map_reaction_atoms",
                "backend_id": "rxnmapper",
                "request": {
                    "inputs": {"reaction": "CCO>>CC=O"},
                    "method_spec": {
                        "route": "#p B3LYP/6-31G(d) operation_id=ROUTE_SENTINEL",
                    },
                    "action_settings": {},
                },
                "depends_on": [],
            }
        ],
    }
    projected = project_workflow_for_revision(workflow, catalog_snapshot=catalog)
    request = projected["steps"][0]["request"]
    assert request["inputs"]["reaction"] == "CCO>>CC=O"
    assert request["method_spec"]["route"] is None
    serialized = _json_text(projected)
    assert "ROUTE_SENTINEL" not in serialized

    revision_input = _revision_input()
    revision_input["current_workflow"] = workflow
    messages = build_model_messages(
        "workflow_revision",
        revision_input,
        memory_enabled=False,
    )
    assert "ROUTE_SENTINEL" not in _json_text(messages)


def test_revision_public_analysis_request_keeps_named_outputs() -> None:
    """Revision 公开 request 稳定保留 AnalysisJobRequest 的具名 outputs。"""

    projected = project_workflow_for_revision(
        {
            "steps": [
                {
                    "step_id": "analysis-step",
                    "step_name": "分析",
                    "execution_layer": "programmable_analysis",
                    "request": {
                        "runtime": "python",
                        "script_path": "code/analyze.py",
                        "outputs": [
                            {
                                "name": "observed_support",
                                "path": "outputs/observed_support.json",
                                "semantic_type": "scalar",
                                "media_type": "application/json",
                                "required": True,
                                "json_schema": {"type": "boolean"},
                                "parent_artifact_ids": ["artifact-1"],
                            }
                        ],
                    },
                }
            ]
        }
    )
    request = projected["steps"][0]["request"]
    assert request["outputs"] == [
        {
            "name": "observed_support",
            "path": None,
            "semantic_type": "scalar",
            "media_type": "application/json",
            "required": True,
            "json_schema": {"type": "boolean"},
            "parent_artifact_ids": [],
        }
    ]


def test_revision_catalog_pair_allowlist_keeps_case1_science_and_hides_paths() -> None:
    """真实 Case1 request 形状按 pair 合同保留科学字段，重复筛选后仍不丢失。"""

    catalog = build_catalog_snapshot(
        actions=[
            {
                "id": "map_reaction_atoms",
                "version": "1.0.0",
                "description": "Map a reaction.",
                "backend_ids": ["rxnmapper"],
                "required_inputs": ["reaction"],
                "optional_inputs": [],
                "selection_policy": "agent_backend_required",
                "primary_output": "MappedReaction",
                "output_semantic_types": ["MappedReaction"],
                "data_action": False,
            }
        ],
        backends=[
            {
                "id": "rxnmapper",
                "display_name": "RXNMapper",
                "runtime": "python",
                "capabilities": ["map_reaction_atoms"],
                "health": "healthy",
            }
        ],
    )
    request_contract = {
        "sections": {
            "inputs": {
                "allowed": [
                    "reaction",
                    "mapped_reaction",
                    "hydrogen_transfers",
                    "candidate_id",
                    "molecule_id",
                    "input_xyz_path",
                    "output_file",
                    "free_energies",
                ],
                "optional": [],
                "required": ["reaction"],
            },
            "method_spec": {
                "allowed": ["energy_unit"],
                "optional": [],
                "required": [],
            },
            "action_settings": {"allowed": [], "optional": [], "required": []},
            "resource_limits": {"allowed": [], "optional": [], "required": []},
        }
    }
    catalog["action_contracts"] = {
        "schema_version": 1,
        "catalog_hash": catalog["catalog_hash"],
        "contracts": [
            {
                "action_id": "map_reaction_atoms",
                "backend_id": "rxnmapper",
                "request_contract": request_contract,
                "output_contract": {},
            }
        ],
    }
    workflow = {
        "workflow_name": "Case1",
        "goal": "保留反应观测",
        "strategy_name": "direct",
        "steps": [
            {
                "step_id": "formal-case1-step",
                "step_name": "反应映射",
                "description": "保留动态反应输入",
                "action_id": "map_reaction_atoms",
                "backend_id": "rxnmapper",
                "request": {
                    "inputs": {
                        "reaction": "CCO>>CC=O",
                        "mapped_reaction": {"reactants": ["CCO"], "products": ["CC=O"]},
                        "hydrogen_transfers": [{"from": 1, "to": 2}],
                        "candidate_id": "candidate-3",
                        "molecule_id": "molecule-reactant",
                        "input_xyz_path": "inputs/reactant.xyz",
                        "output_file": "outputs/mapped.json",
                        "free_energies": [
                            {"conformer_id": "conf-1", "value": -1.2},
                            {"conformer_id": "conf-2", "value": 0.4},
                        ],
                        "unknown_field": "DROP_ME",
                    },
                    "method_spec": {"energy_unit": "kcal/mol", "unknown_field": "DROP_ME"},
                    "action_settings": {"unknown_field": True},
                    "resource_limits": {
                        "cpu_cores": 2,
                        "gpu_count": 0,
                        "memory_mb": 4096,
                        "unknown_field": 99,
                    },
                    "bindings": [
                        {
                            "source_step_ref": "formal-source",
                            "source_field": "job.stdout",
                            "target_field": "inputs.output_file",
                        }
                    ],
                    "outputs": [
                        {
                            "name": "free_energies",
                            "path": "outputs/free_energies.json",
                            "semantic_type": "scalar",
                            "media_type": "application/json",
                            "required": True,
                            "parent_artifact_ids": ["artifact-secret"],
                        }
                    ],
                },
                "depends_on": [],
            }
        ],
    }
    projected = project_workflow_for_revision(workflow, catalog_snapshot=catalog)
    request = projected["steps"][0]["request"]
    assert request["inputs"]["reaction"] == "CCO>>CC=O"
    assert request["inputs"]["mapped_reaction"] == {
        "reactants": ["CCO"],
        "products": ["CC=O"],
    }
    assert request["inputs"]["hydrogen_transfers"] == [{"from": 1, "to": 2}]
    assert request["inputs"]["candidate_id"] == "candidate-3"
    assert request["inputs"]["molecule_id"] == "molecule-reactant"
    assert request["inputs"]["free_energies"] == [
        {"conformer_id": "conf-1", "value": -1.2},
        {"conformer_id": "conf-2", "value": 0.4},
    ]
    assert request["inputs"]["input_xyz_path"] is None
    assert request["inputs"]["output_file"] is None
    assert request["method_spec"]["energy_unit"] == "kcal/mol"
    assert request["resource_limits"] == {"cpu_cores": 2, "gpu_count": 0, "memory_mb": 4096}
    assert request["bindings"][0]["source_field"] == "job.stdout"
    assert request["outputs"][0]["path"] is None
    assert request["outputs"][0]["parent_artifact_ids"] == []
    missing_workflow = deepcopy(workflow)
    del missing_workflow["steps"][0]["request"]["inputs"]["free_energies"]
    missing_request = project_workflow_for_revision(
        missing_workflow,
        catalog_snapshot=catalog,
    )["steps"][0]["request"]
    assert missing_request["inputs"]["free_energies"] is None
    serialized = _json_text(projected)
    assert "DROP_ME" not in serialized
    assert "artifact-secret" not in serialized

    from chemistry_multiagent.utils.workflow_contract import project_minichem_stage_tool_projection

    messages = build_model_messages(
        "workflow_revision",
        {
            "scientific_context": {"scientific_question": "q", "chemistry_context": {}},
            "hypothesis": {"strategy_name": "direct", "reasoning": "r"},
            "current_workflow": projected,
            "current_attempt": {},
            "tool_catalog_projection": project_minichem_stage_tool_projection(catalog),
            "prior_repairs": [],
            "workflow_revision_output_schema": {},
        },
        memory_enabled=True,
    )
    final_workflow = _user_base_input(messages)["current_workflow"]
    final_request = final_workflow["steps"][0]["request"]
    assert final_request["inputs"]["mapped_reaction"] == request["inputs"]["mapped_reaction"]
    assert final_request["inputs"]["free_energies"] == [
        {"conformer_id": "conf-1", "value": -1.2},
        {"conformer_id": "conf-2", "value": 0.4},
    ]
    assert final_request["inputs"]["candidate_id"] == "candidate-3"
    assert final_request["inputs"]["molecule_id"] == "molecule-reactant"
    assert final_request["inputs"]["input_xyz_path"] is None
    assert final_request["bindings"][0]["source_field"] == "job.stdout"
    assert final_request["outputs"][0]["path"] is None
    assert final_request["outputs"][0]["parent_artifact_ids"] == []


def test_revision_native_request_keeps_staged_inputs_and_binding_context() -> None:
    """NativeJobRequest 的 run3 形状在两次筛选后仍可解析且不泄漏路径。"""

    workflow = {
        "workflow_name": "Gaussian",
        "goal": "取得频率",
        "strategy_name": "native",
        "steps": [
            {
                "step_id": "formal-native-step",
                "step_name": "运行 Gaussian",
                "description": "提交 native job",
                "execution_layer": "native_software",
                "request": {
                    "software_id": "gaussian",
                    "executable": "g16",
                    "arguments": ["/mnt/private/job.gjf", "--freq"],
                    "staged_inputs": [
                        {
                            "source_path": "/mnt/private/input.gjf",
                            "target_path": "inputs/job.gjf",
                        }
                    ],
                    "stdin_target": "inputs/job.gjf",
                    "resource_limits": {"cpu_cores": 4, "gpu_count": 0, "memory_mb": 4096},
                    "calculation_intent": "frequency",
                    "label": "case1",
                    "bindings": [
                        {
                            "source_step_ref": "formal-source",
                            "source_field": "job.stdout",
                            "target_field": "staged_inputs.0.source_path",
                        }
                    ],
                    "parent_job_id": "job_0123456789abcdef0123456789abcdef",
                    "authored_files": [
                        {
                            "path": "code/job.gjf",
                            "content": (
                                "%chk=case1.chk\n#p freq\n"
                                "/mnt/private/secret operation_id=op-123 sk-test-secret"
                            ),
                            "overwrite": True,
                        }
                    ],
                },
                "depends_on": [],
            }
        ],
    }
    projected = project_workflow_for_revision(workflow)
    request = projected["steps"][0]["request"]
    assert request["staged_inputs"] == [
        {"source_path": None, "target_path": "inputs/job.gjf"}
    ]
    assert request["bindings"][0]["source_field"] == "job.stdout"
    assert request["bindings"][0]["target_field"] == "staged_inputs.0.source_path"
    assert request["arguments"] == ["--freq"]
    assert "declared_outputs" not in request
    assert request["resource_limits"] == {
        "cpu_cores": 4,
        "gpu_count": 0,
        "memory_mb": 4096,
    }
    assert request["authored_files"] == [
        {"path": "code/job.gjf", "content": None, "overwrite": True}
    ]
    assert request["parent_job_id"] is None
    assert "evaluation_resource_budget" not in request
    assert "execution_timeout_policy" not in request
    serialized = _json_text(projected)
    assert "/mnt/private" not in serialized
    assert "job_0123456789abcdef0123456789abcdef" not in serialized

    messages = build_model_messages(
        "workflow_revision",
        {
            "scientific_context": {"scientific_question": "q", "chemistry_context": {}},
            "hypothesis": {"strategy_name": "native", "reasoning": "r"},
            "current_workflow": projected,
            "current_attempt": {},
            "tool_catalog_projection": [],
            "prior_repairs": [],
            "workflow_revision_output_schema": {},
        },
        memory_enabled=True,
    )
    final_request = _user_base_input(messages)["current_workflow"]["steps"][0]["request"]
    assert final_request["staged_inputs"] == request["staged_inputs"]
    assert final_request["bindings"] == request["bindings"]
    assert final_request["authored_files"] == request["authored_files"]
    assert "/mnt/private" not in messages[1]["content"]

    missing_limits_workflow = deepcopy(workflow)
    del missing_limits_workflow["steps"][0]["request"]["resource_limits"]
    missing_limits_request = project_workflow_for_revision(
        missing_limits_workflow
    )["steps"][0]["request"]
    assert missing_limits_request["resource_limits"] == {
        "cpu_cores": None,
        "gpu_count": None,
        "memory_mb": None,
    }


def test_unregistered_operation_hides_generic_scientific_observation() -> None:
    """未注册 operation 不得把同名科学字段自动升级为阶段事实。"""

    workflow = {
        "steps": [
            {
                "step_id": "formal-energy",
                "step_name": "energy",
                "action_id": "registered_future_energy_action",
            }
        ]
    }
    execution = {
        "steps": [
            {
                "step_id": "formal-energy",
                "status": "completed",
                "result": {
                    "energy": -1.2,
                    "unit": "hartree",
                    "conditions": {
                        "temperature": 300,
                        "unknown_context_field": "HIDDEN_NESTED_SENTINEL",
                    },
                },
            }
        ]
    }
    ledger = project_scientific_attempt_ledger(execution, workflow=workflow)
    observations = ledger["steps"][0]["observations"]
    assert observations == []

    outcome = project_hypothesis_terminal_outcome(
        source_query="energy",
        hypothesis={"strategy_name": "strategy", "reasoning": "reasoning"},
        terminal_classification="insufficient_evidence",
        terminal_reason="reason",
        execution_result=execution,
        workflow=workflow,
        router_decision={
            "decision": "revise_workflow",
            "diagnosis": {
                "summary": "reason",
                "evidence_refs": ["/steps/0/result/energy"],
            },
        },
    )
    cited = outcome["cited_evidence"][0]
    assert cited["value"]["status"] == "unavailable"
    assert "HIDDEN_NESTED_SENTINEL" not in _json_text(outcome)


def test_regeneration_evidence_refs_resolve_values_and_invalid_refs_fail_closed() -> None:
    """再生成可直接解析 eN 的数值、单位和条件，非法引用只形成 unavailable。"""

    workflow = {
        "steps": [
            {
                "step_id": "energy-step",
                "step_name": "energy",
                "action_id": "analyze_thermochemical_selectivity",
            }
        ]
    }
    execution = {
        "steps": [
            {
                "step_id": "energy-step",
                "status": "completed",
                "result": {
                    "delta_delta_g": -1.2,
                    "unit": "hartree",
                    "conditions": {"temperature": 300},
                },
            }
        ]
    }
    outcome = project_hypothesis_terminal_outcome(
        source_query="energy",
        hypothesis={"strategy_name": "strategy", "reasoning": "reasoning"},
        terminal_classification="insufficient_evidence",
        terminal_reason="reason",
        execution_result=execution,
        workflow=workflow,
        router_decision={
            "diagnosis": {"evidence_refs": ["e1", "e99", "e0"]}
        },
    )
    cited = outcome["cited_evidence"]
    assert cited[0]["value"] == -1.2
    assert cited[0]["unit"] == "hartree"
    assert cited[0]["conditions"] == {"temperature": 300}
    assert [item["value"]["status"] for item in cited[1:]] == [
        "unavailable",
        "unavailable",
    ]


def test_revision_repair_uses_n_refs_for_inserted_steps_across_ledger_and_gate() -> None:
    """Revision 新增步骤在 ledger 与 Program Gate 中共享 nN 引用。"""

    before = {
        "workflow_name": "workflow",
        "goal": "goal",
        "strategy_name": "strategy",
        "steps": [
            {
                "step_id": "old-step",
                "step_name": "旧步骤",
                "description": "旧步骤",
                "request": {},
                "depends_on": [],
            }
        ],
    }
    after = {
        "workflow_name": "workflow",
        "goal": "goal",
        "strategy_name": "strategy",
        "steps": [
            before["steps"][0],
            {
                "step_id": "inserted-step",
                "step_name": "新增步骤",
                "description": "新增步骤",
                "request": {},
                "depends_on": ["old-step"],
            },
        ],
    }
    repair = project_workflow_revision_repair(
        normalized_patch={
            "operations": [
                {
                    "op": "insert_step_after",
                    "after_step_id": "old-step",
                    "step": after["steps"][1],
                }
            ]
        },
        before_workflow=before,
        after_workflow=after,
        execution_result={
            "steps": [
                {"step_id": "old-step", "status": "completed"},
                {
                    "step_id": "inserted-step",
                    "status": "failed",
                    "error": {"category": "execution", "code": "failed"},
                },
            ]
        },
        program_validation={
            "program_gate_result": {
                "allowed_decisions": [],
                "constraint_reasons": ["inserted-step:execution_failed"],
            },
            "registered_check_results": [],
        },
        router_decision={},
    )
    assert [item["step_ref"] for item in repair["attempt_ledger"]["steps"]] == [
        "s1",
        "n1",
    ]
    assert repair["program_gate"]["constraint_reasons"] == ["n1:execution_failed"]


def test_revision_strict_pair_drops_unknown_nested_mapping_keys() -> None:
    """严格 Pair 合同只允许注册嵌套科学键，不递归接收未知对象成员。"""

    catalog = {
        "action_contracts": {
            "contracts": [
                {
                    "action_id": "registered_action",
                    "backend_id": "registered_backend",
                    "request_contract": {
                        "sections": {
                            "inputs": {"allowed": ["structured_input"]},
                            "method_spec": {"allowed": []},
                            "action_settings": {"allowed": []},
                            "resource_limits": {"allowed": []},
                        }
                    },
                }
            ]
        },
        "actions": [],
    }
    projected = project_workflow_for_revision(
        {
            "steps": [
                {
                    "step_id": "formal-step",
                    "step_name": "输入",
                    "description": "读取注册输入",
                    "action_id": "registered_action",
                    "backend_id": "registered_backend",
                    "request": {
                        "inputs": {
                            "structured_input": {
                                "value": {"energy": -1.2},
                                "unknown_nested_field": "NESTED_SENTINEL",
                            }
                        }
                    },
                    "depends_on": [],
                }
            ]
        },
        catalog_snapshot=catalog,
    )
    value = projected["steps"][0]["request"]["inputs"]["structured_input"]
    assert value == {"value": {"energy": -1.2}}
    assert "NESTED_SENTINEL" not in _json_text(projected)


def test_scientific_ledger_skips_non_mapping_observations_and_nonfinite_values() -> None:
    """坏账本条目不生成空证据引用，NaN/Infinity 观测按缺失处理。"""

    workflow = {
        "steps": [
            {
                "step_id": "energy-step",
                "step_name": "能量",
                "action_id": "analyze_reaction_free_energy_profile",
            }
        ]
    }
    execution = {
        "steps": [
            {
                "step_id": "energy-step",
                "status": "completed",
                "result": {
                    "relative_gibbs_free_energies": [float("nan")],
                    "activation_free_energies": [float("inf")],
                },
                "observations": [None, "bad-ledger-entry"],
            }
        ]
    }
    ledger = project_scientific_attempt_ledger(execution, workflow=workflow)
    step = ledger["steps"][0]
    assert step["observations"] == []
    assert step["missing_observables"] == [
        "relative_gibbs_free_energies",
        "activation_free_energies",
    ]
    assert [item["evidence_ref"] for item in step["observations"]] == []
    _json_text(ledger)
