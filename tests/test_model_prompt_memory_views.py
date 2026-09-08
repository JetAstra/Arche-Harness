"""阶段 Memory 视图进入模型 Prompt 前的固定合同测试。"""

import json

from chemistry_multiagent.utils.model_prompts import build_model_messages


def _base_input(messages):
    return json.loads(messages[1]["content"])["base_input"]


def _regeneration_input():
    return {
        "scientific_context": {
            "scientific_question": "比较催化反应的选择性来源。",
            "initial_query_angles": ["溶剂化效应"],
            "initial_retrieval_phase": {
                "keywords": ["enantioselectivity"],
                "literature_review": "文献综述",
                "mechanistic_clues": ["过渡态稳定化"],
                "limitations": [],
                "chemistry_context": {"solvent": "acetone"},
            },
            "chemistry_context": {"temperature": "303.15 K"},
        },
        "current_query": "溶剂化效应",
        "prior_hypothesis_outcomes": [
            {
                "source_query": "溶剂化效应",
                "hypothesis": {
                    "strategy_name": "溶剂化差异",
                    "reasoning": "R" * 3074,
                    "hypothesis_id": "FORMAL_HYPOTHESIS_ID",
                },
                "terminal_classification": "insufficient_evidence",
                "final_judge": {
                    "reason": "需要直接比较能垒 /mnt/private/run.log",
                },
                "cited_evidence": [
                    {
                        "evidence_ref": "e1",
                        "step_ref": "s1",
                        "evidence_kind": "observation",
                        "name": "relative_free_energy",
                        "value": -1.2,
                        "unit": "hartree",
                        "conditions": {},
                    }
                ],
                "attempted_observables": ["相对自由能"],
                "missing_evidence": ["过渡态能垒"],
                "tool_capability_gap": None,
                "record_id": "FORMAL_RECORD_ID",
            }
        ],
        "next_batch_index": 2,
        "generation_config": {
            "num_queries": 1,
            "num_hypotheses_per_query": 2,
            "top_n_workflows": 1,
        },
        "hypothesis_output_schema": {"type": "array", "id": "SCHEMA_ID"},
        # 这些旧历史字段必须被阶段视图忽略。
        "selected_evaluated_history": [{"record_id": "OLD_HISTORY"}],
        "unselected_summary": [{"batch_index": 1}],
        "source_record_ids": ["OLD_SOURCE"],
    }


def _revision_input():
    return {
        "scientific_context": {
            "scientific_question": "解释反应选择性。",
            "chemistry_context": {"solvent": "acetone"},
        },
        "hypothesis": {
            "strategy_name": "过渡态差异",
            "reasoning": "比较两个过渡态的自由能。",
            "hypothesis_id": "FORMAL_HYPOTHESIS_ID",
        },
        "current_workflow": {
            "workflow_name": "当前工作流",
            "goal": "取得决定性观测",
            "strategy_name": "过渡态差异",
            "steps": [
                {
                    "step_ref": "s1",
                    "scientific_purpose": "计算反应能垒。",
                    "operation": "gaussian_calculation",
                    "target": {"label": "能垒计算", "role": "观测"},
                    "scientific_settings": {"charge": 0},
                    "expected_observables": [{"name": "barrier", "unit": "kcal/mol"}],
                    "depends_on": [],
                }
            ],
        },
        "current_attempt": {
            "scientific_ledger": {
                "overall_status": "failed",
                "workflow_outcome": "insufficient_evidence",
                "steps": [
                    {
                        "step_ref": "s1",
                        "status": "failed",
                        "observations": [
                            {"name": "energy", "value": -1.2, "unit": "hartree"}
                        ],
                        "program_checks": [],
                        "missing_observables": ["barrier"],
                        "failure": {
                            "category": "execution",
                            "details": "失败日志 /mnt/private/error.log",
                        },
                    }
                ],
            },
            "program_gate": {
                "allowed_decisions": ["revise_workflow"],
                "constraint_reasons": [],
                "checks": [],
            },
            "judge": {
                "decision": "revise_workflow",
                "reason": "证据不足 /mnt/private",
                "evidence_refs": ["e1"],
            },
        },
        "tool_catalog_projection": [
            {
                "action_id": "gaussian_calculation",
                "action_version": "1.0.0",
                "description": "计算化学 Action",
                "backend_ids": ["gaussian_backend"],
                "backends": [
                    {
                        "backend_id": "gaussian_backend",
                        "display_name": "Gaussian",
                        "runtime": "native",
                        "availability": "available",
                        "capabilities": ["gaussian"],
                        "absolute_path": "/mnt/private/FORMAL_BACKEND_INSTANCE",
                    }
                ],
                "required_inputs": [],
                "optional_inputs": [],
                "method_spec": {},
                "primary_output": "Measurement",
                "output_semantic_types": ["measurement"],
                "selection_policy": "agent_backend_required",
                "data_action": False,
            }
        ],
        "prior_repairs": [
            {
                "patch": [
                    {
                        "operation": "update_step",
                        "step_ref": "s1",
                        "field": "scientific_settings",
                        "before": {"charge": 0},
                        "after": {"charge": 0},
                    }
                ],
                "affected_workflow_before": [],
                "affected_workflow_after": [],
                "attempt_ledger": {},
                "judge": {},
            }
        ],
        "workflow_revision_output_schema": {
            "type": "object",
            "id": "SCHEMA_ID",
        },
        "workflow_failure_trajectories": [{"record_id": "OLD_TRAJECTORY"}],
        "human_correction_chains": [{"record_id": "OLD_CORRECTION"}],
    }


def test_regeneration_uses_fixed_roots_and_keeps_long_reasoning():
    enabled = build_model_messages(
        "hypothesis_generation",
        _regeneration_input(),
        stage="regeneration",
        memory_enabled=True,
    )
    projected = _base_input(enabled)
    assert set(projected) == {
        "scientific_context",
        "current_query",
        "prior_hypothesis_outcomes",
        "next_batch_index",
        "generation_config",
        "hypothesis_output_schema",
    }
    assert len(projected["prior_hypothesis_outcomes"][0]["hypothesis"]["reasoning"]) == 3074
    serialized = enabled[1]["content"]
    for hidden in (
        "FORMAL_HYPOTHESIS_ID",
        "FORMAL_RECORD_ID",
        "OLD_HISTORY",
        "OLD_SOURCE",
        "/mnt/private",
        "selected_evaluated_history",
    ):
        assert hidden not in serialized


def test_regeneration_output_schema_is_recursively_allowlisted() -> None:
    """再生成 schema 只保留 JSON Schema 关键字，递归隐藏未知扩展。"""

    source = _regeneration_input()
    source["hypothesis_output_schema"] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "strategy_name": {
                    "type": "string",
                    "x-unknown": {"secret": "SCHEMA_EXTENSION_SENTINEL"},
                },
                "reasoning": {"type": "string"},
            },
            "required": ["strategy_name", "reasoning"],
            "unknown_schema": {"secret": "NESTED_SCHEMA_SENTINEL"},
        },
        "unknown_schema": {"secret": "ROOT_SCHEMA_SENTINEL"},
        "x-unknown": "EXTENSION_SENTINEL",
    }

    projected = _base_input(
        build_model_messages(
            "hypothesis_generation",
            source,
            stage="regeneration",
            memory_enabled=True,
        )
    )

    assert projected["hypothesis_output_schema"] == {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "strategy_name": {"type": "string"},
                "reasoning": {"type": "string"},
            },
            "required": ["strategy_name", "reasoning"],
        },
    }
    serialized = json.dumps(projected, ensure_ascii=False)
    for marker in (
        "SCHEMA_EXTENSION_SENTINEL",
        "NESTED_SCHEMA_SENTINEL",
        "ROOT_SCHEMA_SENTINEL",
        "EXTENSION_SENTINEL",
    ):
        assert marker not in serialized


def test_regeneration_memory_off_changes_only_prior_outcomes():
    source = _regeneration_input()
    enabled = _base_input(
        build_model_messages(
            "hypothesis_generation",
            source,
            stage="regeneration",
            memory_enabled=True,
        )
    )
    disabled = _base_input(
        build_model_messages(
            "hypothesis_generation",
            source,
            stage="regeneration",
            memory_enabled=False,
        )
    )
    assert disabled["prior_hypothesis_outcomes"] == []
    assert {
        key: value
        for key, value in enabled.items()
        if key != "prior_hypothesis_outcomes"
    } == {
        key: value
        for key, value in disabled.items()
        if key != "prior_hypothesis_outcomes"
    }


def test_regeneration_history_lists_hide_runtime_text_and_bad_ledger_entries():
    """再生成历史清单只保留安全科学短语，坏引用不虚构空 eN。"""

    source = _regeneration_input()
    outcome = source["prior_hypothesis_outcomes"][0]
    outcome["attempted_observables"] = [
        "相对自由能",
        "inputs/request.json",
        "action_id=run_gaussian",
        "01234567-89ab-cdef-0123-456789abcdef",
        "artifact_hash=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ]
    outcome["missing_evidence"] = [
        "过渡态能垒",
        "outputs/result.json",
        "backend_id=gaussian",
    ]
    outcome["tool_capability_gap"] = {
        "result_type": "tool_capability_gap",
        "missing_capabilities": [
            "频率",
            "runtime=/tmp/gaussian",
            "action_id=hidden_action",
        ],
        "reason": "缺少频率能力",
    }
    outcome["cited_evidence"] = [None, "坏账本条目", outcome["cited_evidence"][0]]

    projected = _base_input(
        build_model_messages(
            "hypothesis_generation",
            source,
            stage="regeneration",
            memory_enabled=True,
        )
    )
    history = projected["prior_hypothesis_outcomes"][0]
    assert history["attempted_observables"] == ["相对自由能"]
    assert history["missing_evidence"] == ["过渡态能垒"]
    assert history["tool_capability_gap"]["missing_capabilities"] == ["频率"]
    assert [item["evidence_ref"] for item in history["cited_evidence"]] == ["e1"]
    serialized = json.dumps(projected, ensure_ascii=False, allow_nan=False)
    for marker in (
        "inputs/request.json",
        "outputs/result.json",
        "action_id=",
        "backend_id=",
        "01234567-89ab-cdef-0123-456789abcdef",
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ):
        assert marker not in serialized


def test_regeneration_nonfinite_observations_are_treated_as_missing():
    """NaN/Infinity 观测按缺失事实处理，仍可稳定编码模型消息。"""

    source = _regeneration_input()
    source["prior_hypothesis_outcomes"][0]["cited_evidence"][0]["value"] = float("nan")
    messages = build_model_messages(
        "hypothesis_generation",
        source,
        stage="regeneration",
        memory_enabled=True,
    )
    projected = _base_input(messages)
    assert projected["prior_hypothesis_outcomes"][0]["cited_evidence"][0]["value"] is None
    json.dumps(projected, ensure_ascii=False, allow_nan=False)


def test_allowed_scientific_text_is_not_rewritten_by_keyword_or_path_scanning():
    """合法问题、Query 和假设正文按来源放行，派生错误路径仍会被移除。"""

    regeneration = _regeneration_input()
    regeneration["scientific_context"]["scientific_question"] = (
        "比较用户指定的 /tmp/input.xyz；job_id 只是题面中的普通字样。"
    )
    regeneration["current_query"] = "分析 /tmp/input.xyz 的构象"
    regeneration["prior_hypothesis_outcomes"][0]["hypothesis"]["reasoning"] = (
        "reasoning 中完整保留 /tmp/input.xyz、raw_output 和 raw_log 字样。"
    )
    regeneration["prior_hypothesis_outcomes"][0]["final_judge"]["reason"] = (
        "派生理由包含 /mnt/private/run.log job_id=abc operation_id=z "
        "raw_output.log raw_log runtime=5"
    )
    projected = _base_input(
        build_model_messages(
            "hypothesis_generation",
            regeneration,
            stage="regeneration",
            memory_enabled=True,
        )
    )

    assert projected["scientific_context"]["scientific_question"] == regeneration[
        "scientific_context"
    ]["scientific_question"]
    assert projected["current_query"] == regeneration["current_query"]
    assert projected["prior_hypothesis_outcomes"][0]["hypothesis"][
        "reasoning"
    ] == regeneration["prior_hypothesis_outcomes"][0]["hypothesis"]["reasoning"]
    serialized = json.dumps(projected, ensure_ascii=False)
    for internal in (
        "/mnt/private/run.log",
        "job_id=abc",
        "operation_id=z",
        "raw_output.log",
        "raw_log runtime=5",
        "runtime=5",
    ):
        assert internal not in serialized


def test_derived_prompt_paths_cover_all_absolute_path_forms() -> None:
    """模型 Prompt 的派生理由统一替换绝对路径并保留科学斜杠。"""

    source = _regeneration_input()
    paths = (
        "/private/secret/my run/boom.json",
        "/data/secret/boom.json",
        r"C:\Program Files\Gaussian\g16.exe",
        r"\\server\share\Program Files\g16.exe",
        "file:///private/secret/my file.log",
    )
    source["prior_hypothesis_outcomes"][0]["final_judge"]["reason"] = (
        "；".join(paths) + "；参考 https://server/share；方法 B3LYP/6-31G(d)"
    )
    projected = _base_input(
        build_model_messages(
            "hypothesis_generation",
            source,
            stage="regeneration",
            memory_enabled=True,
        )
    )
    serialized = json.dumps(projected, ensure_ascii=False)
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


def test_derived_prompt_path_replacement_preserves_text_after_file_boundary():
    """含空格路径替换到文件边界后，后续普通句子必须原样保留。"""

    source = _regeneration_input()
    source["prior_hypothesis_outcomes"][0]["final_judge"]["reason"] = (
        r"读取 C:\Program Files\Gaussian\g16.exe 后继续检查。"
    )
    projected = _base_input(
        build_model_messages(
            "hypothesis_generation",
            source,
            stage="regeneration",
            memory_enabled=True,
        )
    )
    reason = projected["prior_hypothesis_outcomes"][0]["final_judge"]["reason"]
    assert reason.endswith("[removed-path] 后继续检查。")


def test_revision_uses_fixed_roots_and_deduplicates_attempt_judge():
    messages = build_model_messages(
        "workflow_revision",
        _revision_input(),
        memory_enabled=True,
    )
    projected = _base_input(messages)
    assert set(projected) == {
        "scientific_context",
        "hypothesis",
        "current_workflow",
        "current_attempt",
        "tool_catalog_projection",
        "prior_repairs",
        "workflow_revision_output_schema",
    }
    assert "program_gate" not in projected["current_attempt"]["scientific_ledger"]
    assert "judge" not in projected["current_attempt"]["scientific_ledger"]
    assert set(projected["prior_repairs"][0]) == {
        "patch",
        "affected_workflow_before",
        "affected_workflow_after",
        "attempt_ledger",
        "program_gate",
        "judge",
    }
    step = projected["current_workflow"]["steps"][0]
    assert set(step) == {
        "step_ref",
        "step_name",
        "description",
        "execution_layer",
        "action_id",
        "backend_id",
        "request",
        "depends_on",
    }
    assert "resource_limits" in step["request"]
    assert "authored_files" not in step["request"]
    assert "FORMAL_BACKEND_INSTANCE" not in messages[1]["content"]
    assert "BACKEND_ID_SENTINEL" not in messages[1]["content"]
    serialized = messages[1]["content"]
    for hidden in (
        "FORMAL_HYPOTHESIS_ID",
        "FORMAL_STEP_ID",
        "FORMAL_OPERATION_ID",
        "FORMAL_TRAJECTORY",
        "FORMAL_CORRECTION",
        "OLD_TRAJECTORY",
        "OLD_CORRECTION",
        "/mnt/private",
        "workflow_failure_trajectories",
    ):
        assert hidden not in serialized


def test_revision_output_schema_is_recursively_allowlisted() -> None:
    """Workflow Revision schema 保留合法组合结构并移除未知嵌套键。"""

    source = _revision_input()
    source["workflow_revision_output_schema"] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "oneOf": [
            {
                "type": "object",
                "properties": {
                    "operations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "op": {"enum": ["update_step"]},
                                "changes": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "unknown_schema": "NESTED_REVISION_SENTINEL",
                                },
                            },
                            "required": ["op"],
                            "x-unknown": "ITEM_EXTENSION_SENTINEL",
                        },
                    }
                },
                "required": ["operations"],
                "unknown_schema": "BRANCH_SCHEMA_SENTINEL",
            }
        ],
        "x-unknown": "ROOT_EXTENSION_SENTINEL",
    }

    projected = _base_input(
        build_model_messages("workflow_revision", source, memory_enabled=True)
    )
    assert projected["workflow_revision_output_schema"] == {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "oneOf": [
            {
                "type": "object",
                "properties": {
                    "operations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "op": {"enum": ["update_step"]},
                                "changes": {
                                    "type": "object",
                                    "additionalProperties": False,
                                },
                            },
                            "required": ["op"],
                        },
                    }
                },
                "required": ["operations"],
            }
        ],
    }
    serialized = json.dumps(projected, ensure_ascii=False)
    for marker in (
        "NESTED_REVISION_SENTINEL",
        "ITEM_EXTENSION_SENTINEL",
        "BRANCH_SCHEMA_SENTINEL",
        "ROOT_EXTENSION_SENTINEL",
    ):
        assert marker not in serialized


def test_output_schema_name_maps_filter_sensitive_names_and_preserve_refs() -> None:
    """schema-name map 过滤内部字段并同步 required，同时保留标准引用语义。"""

    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://example.org/schemas/case1",
        "$anchor": "case1",
        "$dynamicAnchor": "case1_dynamic",
        "$dynamicRef": "#case1_dynamic",
        "$ref": "#/$defs/safe_value",
        "$defs": {
            "safe_value": {"type": "number"},
            "raw_output": {"type": "string"},
            "/tmp/x": {"type": "string"},
            "file:///tmp/x": {"type": "string"},
        },
        "type": "object",
        "properties": {
            "safe_value": {"$ref": "#/$defs/safe_value"},
            "raw_output": {"type": "string"},
            "/tmp/x": {"type": "string"},
            "file:///tmp/x": {"type": "string"},
            "B3LYP/6-31G(d)": {"type": "string"},
        },
        "patternProperties": {
            "raw_output": {"type": "string"},
            "^safe_": {"type": "number"},
        },
        "dependentSchemas": {
            "raw_output": {"type": "string"},
            "safe_value": {"$ref": "#/$defs/safe_value"},
        },
        "required": [
            "safe_value",
            "raw_output",
            "/tmp/x",
            "file:///tmp/x",
            "B3LYP/6-31G(d)",
        ],
        "dependentRequired": {
            "safe_value": ["safe_value", "raw_output"],
            "raw_output": ["safe_value"],
        },
        "unknown_schema": {"secret": "UNKNOWN_SCHEMA_SENTINEL"},
    }

    for operation, source, stage, schema_field in (
        (
            "hypothesis_generation",
            _regeneration_input(),
            "regeneration",
            "hypothesis_output_schema",
        ),
        ("workflow_revision", _revision_input(), None, "workflow_revision_output_schema"),
    ):
        source[schema_field] = schema
        kwargs = {"memory_enabled": True}
        if stage is not None:
            kwargs["stage"] = stage
        projected = _base_input(build_model_messages(operation, source, **kwargs))
        output_schema = projected[schema_field]

        assert output_schema["$ref"] == "#/$defs/safe_value"
        assert output_schema["$id"] == "https://example.org/schemas/case1"
        assert output_schema["$anchor"] == "case1"
        assert output_schema["$dynamicRef"] == "#case1_dynamic"
        assert output_schema["$defs"] == {"safe_value": {"type": "number"}}
        assert set(output_schema["properties"]) == {"safe_value", "B3LYP/6-31G(d)"}
        assert set(output_schema["patternProperties"]) == {"^safe_"}
        assert set(output_schema["dependentSchemas"]) == {"safe_value"}
        assert output_schema["required"] == ["safe_value", "B3LYP/6-31G(d)"]
        assert output_schema["dependentRequired"] == {"safe_value": ["safe_value"]}
        serialized = json.dumps(output_schema, ensure_ascii=False)
        for marker in (
            "raw_output",
            "/tmp/x",
            "file:///tmp/x",
            "UNKNOWN_SCHEMA_SENTINEL",
        ):
            assert marker not in serialized


def test_revision_final_message_keeps_scientific_settings_and_preflight_only() -> None:
    """最终 Revision 消息保留设置/预检事实，并过滤结构运行元数据。"""

    source = _revision_input()
    ledger_step = source["current_attempt"]["scientific_ledger"]["steps"][0]
    ledger_step["scientific_settings"] = {
        "method": {"value": "B3LYP", "source": "verified_effective"},
        "server_info": {"value": "HIDDEN_SERVER", "source": "verified_effective"},
    }
    ledger_step["native_preflight"] = {
        "status": "ready",
        "ready": True,
        "calculation_intent": "opt freq",
        "link1_segment_count": 2,
        "checks": ["route_valid"],
        "server_info": "HIDDEN_PREFLIGHT_SERVER",
    }
    ledger_step["observations"] = [
        {
            "evidence_ref": "e1",
            "name": "optimized_geometry",
            "value": {
                "atoms": [
                    {
                        "element": "C",
                        "position_angstrom": [0.0, 0.1, 0.2],
                        "server_info": "HIDDEN_ATOM_SERVER",
                    }
                ],
                "metadata": "HIDDEN_METADATA",
                "custom_field": "HIDDEN_CUSTOM",
                "output_files": ["/tmp/HIDDEN_OUTPUT"],
            },
            "unit": None,
            "conditions": {},
        }
    ]

    messages = build_model_messages("workflow_revision", source, memory_enabled=True)
    projected = _base_input(messages)
    step = projected["current_attempt"]["scientific_ledger"]["steps"][0]
    assert step["scientific_settings"] == {
        "method": {"value": "B3LYP", "source": "verified_effective"},
    }
    assert step["native_preflight"] == {
        "status": "ready",
        "ready": True,
        "calculation_intent": "opt freq",
        "link1_segment_count": 2,
        "checks": ["route_valid"],
    }
    assert step["observations"][0]["value"] == {
        "atoms": [{"element": "C", "position_angstrom": [0.0, 0.1, 0.2]}],
    }
    serialized = messages[1]["content"]
    for hidden in (
        "HIDDEN_SERVER",
        "HIDDEN_PREFLIGHT_SERVER",
        "HIDDEN_ATOM_SERVER",
        "HIDDEN_METADATA",
        "HIDDEN_CUSTOM",
        "HIDDEN_OUTPUT",
        "/tmp/HIDDEN_OUTPUT",
    ):
        assert hidden not in serialized


def test_revision_memory_off_changes_only_prior_repairs():
    source = _revision_input()
    enabled = _base_input(
        build_model_messages("workflow_revision", source, memory_enabled=True)
    )
    disabled = _base_input(
        build_model_messages("workflow_revision", source, memory_enabled=False)
    )
    assert disabled["prior_repairs"] == []
    assert {
        key: value for key, value in enabled.items() if key != "prior_repairs"
    } == {
        key: value for key, value in disabled.items() if key != "prior_repairs"
    }


def test_regeneration_reassigns_duplicate_history_evidence_refs():
    """损坏旧历史中的重复 eN 在最终 outcome 内按事实顺序重新编号。"""

    source = _regeneration_input()
    duplicate = dict(source["prior_hypothesis_outcomes"][0]["cited_evidence"][0])
    duplicate["value"] = -2.4
    source["prior_hypothesis_outcomes"][0]["cited_evidence"].append(duplicate)
    projected = _base_input(
        build_model_messages(
            "hypothesis_generation",
            source,
            stage="regeneration",
            memory_enabled=True,
        )
    )
    refs = [
        item["evidence_ref"]
        for item in projected["prior_hypothesis_outcomes"][0]["cited_evidence"]
    ]
    assert refs == ["e1", "e2"]


def test_revision_final_adapter_closes_refs_and_filters_nested_sentinels():
    """最终 Revision 消息拒绝未知依赖、对象数组项和悬空/歧义 eN。"""

    source = _revision_input()
    source["current_workflow"]["steps"] = [
        {
            "step_ref": "s1",
            "step_name": "source",
            "description": "source",
            "execution_layer": "predefined_actions",
            "action_id": "source_action",
            "backend_id": "source_backend",
            "request": {},
            "depends_on": [],
        },
        {
            "step_ref": "s2",
            "step_name": "target",
            "description": "target",
            "execution_layer": "predefined_actions",
            "action_id": "target_action",
            "backend_id": "target_backend",
            "request": {
                "bindings": [
                    {
                        "source_step_ref": "FORMAL_BINDING_SENTINEL",
                        "source_field": "/result/energy",
                        "target_field": "inputs.energy",
                    }
                ],
                "foreach": {
                    "source_step_ref": "s1",
                    "source_field": "/result/structures",
                    "item_field": "structure",
                    "item_id_field": "label",
                    "target_field": "inputs.structure",
                    "max_items": 9,
                },
            },
            "depends_on": ["s1", "FORMAL_DEPENDENCY_SENTINEL"],
        },
    ]
    source["current_attempt"]["scientific_ledger"]["steps"][0][
        "missing_observables"
    ] = ["barrier", {"secret": "NESTED_MISSING_SENTINEL"}]
    source["current_attempt"]["scientific_ledger"]["steps"][0][
        "observations"
    ] = [
        {
            "evidence_ref": "e1",
            "name": "energy",
            "value": -1.2,
            "unit": "hartree",
            "conditions": {},
        },
        {
            "evidence_ref": "e1",
            "name": "enthalpy",
            "value": -1.1,
            "unit": "hartree",
            "conditions": {},
        },
    ]
    source["current_attempt"]["program_gate"] = {
        "allowed_decisions": [
            "revise_workflow",
            "UNKNOWN_DECISION_SENTINEL",
            {"secret": "NESTED_DECISION_SENTINEL"},
        ],
        "constraint_reasons": [
            "required_reliability_not_matched",
            "formal-old-step:execution_failed",
            {"secret": "NESTED_REASON_SENTINEL"},
        ],
        "checks": [],
    }
    source["current_attempt"]["judge"]["evidence_refs"] = ["e1", "e99"]

    messages = build_model_messages(
        "workflow_revision",
        source,
        memory_enabled=True,
    )
    projected = _base_input(messages)
    request = projected["current_workflow"]["steps"][1]["request"]
    assert request["bindings"][0]["source_step_ref"] is None
    assert request["foreach"]["source_step_ref"] == "s1"
    assert request["foreach"]["max_items"] == 9
    assert projected["current_workflow"]["steps"][1]["depends_on"] == ["s1"]

    attempt = projected["current_attempt"]
    assert attempt["scientific_ledger"]["steps"][0]["missing_observables"] == [
        "barrier"
    ]
    assert attempt["program_gate"]["allowed_decisions"] == ["revise_workflow"]
    assert attempt["program_gate"]["constraint_reasons"] == [
        "required_reliability_not_matched",
        "unresolved_step:execution_failed",
    ]
    assert [
        item["evidence_ref"]
        for item in attempt["scientific_ledger"]["steps"][0]["observations"]
    ] == ["e1", "e2"]
    assert attempt["judge"]["evidence_refs"] == []
    serialized = messages[1]["content"]
    for marker in (
        "FORMAL_BINDING_SENTINEL",
        "FORMAL_DEPENDENCY_SENTINEL",
        "NESTED_MISSING_SENTINEL",
        "UNKNOWN_DECISION_SENTINEL",
        "NESTED_DECISION_SENTINEL",
        "NESTED_REASON_SENTINEL",
        "formal-old-step",
    ):
        assert marker not in serialized
