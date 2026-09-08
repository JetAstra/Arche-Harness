"""固定 Gaussian 专家输入视图的字段、证据和泄漏边界测试。"""

from __future__ import annotations

import json
from typing import Any

import pytest

from chemistry_multiagent.utils.expert_views import (
    POST_VIEW_KEYS,
    PRE_VIEW_KEYS,
    STEP_VIEW_KEYS,
    build_gaussian_step_review_view,
    build_workflow_scientific_evidence_view,
    flatten_workflow_evidence_index,
)


def test_gaussian_review_view_keeps_complete_reasoning_and_fixed_allowlist() -> None:
    """执行前视图保留长正文，同时不携带正式身份和请求对象。"""

    reasoning = "完整科学推理。" * 600
    view = build_gaussian_step_review_view(
        scientific_question="在 303.15 K、acetone 中检验选择性。",
        hypothesis={
            "hypothesis_id": "h-secret",
            "strategy_name": "DFT",
            "reasoning": reasoning,
            "explicit_conditions": [
                {"name": "temperature", "value": 303.15, "unit": "K"}
            ],
        },
        step={
            "step_id": "step-secret",
            "step_name": "XYZ 转 Gaussian",
            "description": "生成频率计算输入。",
            "action_id": "xyz_to_gjf",
            "request": {
                "inputs": {
                    "input_xyz_path": "/mnt/private/input.xyz",
                    "charge": 0,
                    "multiplicity": 1,
                    "title": "complex",
                    "nprocshared": 8,
                    "mem": "16GB",
                },
                "method_spec": {"method": "B3LYP", "secret_path": "/tmp/x"},
            },
        },
        route="#p B3LYP/6-31G(d) opt freq",
        converter_kind="xyz_to_gjf",
    )

    assert tuple(view) == PRE_VIEW_KEYS
    assert tuple(view["hypothesis"]) == ("strategy_name", "reasoning")
    assert view["hypothesis"]["reasoning"] == reasoning
    assert "step_id" not in str(view)
    assert "input_xyz_path" not in str(view)
    assert tuple(view["gaussian_unit"]) == (
        "scientific_purpose",
        "calculation_type",
        "structure",
        "requested_route_intent",
        "generated_route",
        "editable_parameters",
        "validation_requirements",
        "mutable_fields",
    )


def test_derived_paths_are_hidden_for_any_posix_root_and_file_uri_without_route_false_positive() -> None:
    """派生文本拒绝任意绝对路径/ file URI，但保留 Gaussian route 的除法斜线。"""

    route = "#p B3LYP/6-31G(d) opt freq"
    view = build_gaussian_step_review_view(
        scientific_question="问题",
        hypothesis={
            "strategy_name": "策略 /secret/file.gjf",
            "reasoning": "依据 file:///secret/file.gjf 进行比较。",
            "validation_requirements": [
                {"description": "file://localhost/secret/check.json"}
            ],
        },
        step={
            "step_name": "准备",
            "description": "生成 route /secret/file.gjf",
            "request": {"inputs": {}},
        },
        route=route,
        converter_kind="xyz",
    )

    assert view["gaussian_unit"]["generated_route"] == route
    assert view["hypothesis"] == {"strategy_name": "", "reasoning": ""}
    assert view["gaussian_unit"]["scientific_purpose"] == ""
    assert view["gaussian_unit"]["validation_requirements"] == [{}]
    serialized = json.dumps(view, ensure_ascii=False)
    assert "/secret/file.gjf" not in serialized
    assert "file://" not in serialized


def test_gaussian_review_view_reads_xyz_summary_without_exposing_coordinates(tmp_path: Any) -> None:
    """存在 XYZ 输入文件时只发送确定性分子摘要，不发送路径或坐标。"""

    structure = tmp_path / "complex.xyz"
    structure.write_text(
        "3\nwater\nO 0.0 0.0 0.0\nH 0.0 0.0 1.0\nH 0.0 1.0 0.0\n",
        encoding="utf-8",
    )
    view = build_gaussian_step_review_view(
        scientific_question="问题",
        hypothesis={"strategy_name": "策略", "reasoning": "正文"},
        step={
            "step_name": "XYZ 转 Gaussian",
            "description": "转换结构",
            "request": {"inputs": {"input_xyz_path": str(structure)}},
        },
        route="#p opt",
        converter_kind="xyz_to_gjf",
    )
    summary = view["gaussian_unit"]["structure"]
    assert summary["summary_status"] == "available"
    assert summary["formula"] == "H2O"
    assert summary["atom_count"] == 3
    assert summary["element_counts"] == {"O": 1, "H": 2}
    assert str(structure) not in str(view)
    assert "0.0 0.0" not in str(view)


def test_gaussian_review_view_reads_current_sdf_artifact_summary_without_leaks(
    tmp_path: Any,
) -> None:
    """执行前审查从已完成上游 SDF ArtifactRef 提取摘要而不复制内部字段。"""

    sdf = tmp_path / "outputs" / "product.sdf"
    sdf.parent.mkdir(parents=True)
    sdf.write_text(
        "water\n  ARCHE\n\n"
        "  3  0  0  0  0  0  0  0  0  0999 V2000\n"
        "    0.0000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "    0.0000    0.0000    1.0000 H   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "    0.0000    1.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "M  END\n$$$$\n",
        encoding="utf-8",
    )
    step = {
        "step_id": "converter-step-secret",
        "action_id": "sdf_to_gjf",
        "description": "准备 Gaussian 输入",
        "request": {
            "inputs": {},
            "bindings": [
                {
                    "source_step_ref": "source-step-secret",
                    "source_field": "artifact.media_type=chemical/x-mdl-sd",
                    "target_field": "inputs.sdf_path",
                }
            ],
        },
    }
    completed_steps = {
        "source-step-secret": {
            "step_id": "source-step-secret",
            "action_id": "smiles2sdf",
            "backend_id": "format_conversion",
            "output_artifacts": [
                {
                    "artifact_id": "artifact-json-secret",
                    "media_type": "application/json",
                    "semantic_type": "SDFFile",
                    "path": "_tool_artifacts/objects/artifact-json-secret.json",
                },
                {
                    "artifact_id": "artifact-sdf-secret",
                    "media_type": "chemical/x-mdl-sd",
                    "semantic_type": "SDFFile",
                    "path": "outputs/product.sdf",
                },
            ],
            "raw_output": {
                "raw_output": "RAW_SENTINEL",
                "path": "/tmp/raw.sdf",
            },
        }
    }
    view = build_gaussian_step_review_view(
        scientific_question="验证当前产品结构。",
        hypothesis={"strategy_name": "结构策略", "reasoning": "依据当前产物。"},
        step=step,
        route="#p M06-2X/def2-TZVP opt freq",
        converter_kind="sdf_to_gjf",
        completed_steps=completed_steps,
        task_root=tmp_path,
    )
    structure = view["gaussian_unit"]["structure"]
    assert structure["summary_status"] == "available"
    assert structure["formula"] == "H2O"
    assert structure["atom_count"] == 3
    assert structure["element_counts"] == {"H": 2, "O": 1}
    serialized = json.dumps(view, ensure_ascii=False)
    for marker in (
        "converter-step-secret",
        "source-step-secret",
        "artifact-json-secret",
        "artifact-sdf-secret",
        "RAW_SENTINEL",
        "/tmp/raw.sdf",
        "outputs/product.sdf",
        "chemical/x-mdl-sd",
    ):
        assert marker not in serialized


def test_workflow_view_has_one_canonical_step_and_embedded_refs() -> None:
    """执行后每个步骤只出现一次，空字段仍保持固定 schema。"""

    workflow = {
        "steps": [
            {
                "step_id": "route-id",
                "step_name": "route",
                "description": "生成 route",
                "action_id": "generate_gaussian_code",
                "depends_on": [],
            },
            {
                "step_id": "unknown-id",
                "step_name": "unknown",
                "description": "未知分析",
                "action_id": "future_action",
                "depends_on": ["route-id"],
            },
        ]
    }
    execution = {
        "steps": [
            {
                "step_id": "route-id",
                "status": "completed",
                "raw_output": {
                    "status": "completed",
                    "result": {
                        "gaussian_code": "#p opt freq",
                        "raw_output": "should not pass",
                        "path": "/mnt/secret",
                    },
                },
            }
        ]
    }
    view, index = build_workflow_scientific_evidence_view(
        scientific_question="问题",
        hypothesis={"hypothesis_id": "hidden", "strategy_name": "策略", "reasoning": "正文"},
        workflow=workflow,
        execution_results=execution,
    )

    assert tuple(view) == POST_VIEW_KEYS
    steps = view["workflow_evidence"]["steps"]
    assert len(steps) == 2
    assert all(tuple(step) == STEP_VIEW_KEYS for step in steps)
    assert [step["step_ref"] for step in steps] == ["s1", "s2"]
    assert steps[0]["observations"][0]["evidence_ref"] == "e1"
    assert steps[1]["coverage"]["status"] == "unavailable"
    assert steps[1]["not_executed"]["blocked_by"] == ["s1"]
    assert all("path" not in str(item) for item in (view, index))
    assert all("raw_output" not in str(item) for item in (view, index))
    assert "evidence_index" not in view
    assert flatten_workflow_evidence_index(view) == index


def test_deck_projection_does_not_send_full_input_deck() -> None:
    """deck 步骤只提供预检状态，不把含坐标的完整输入文件交给模型。"""

    view, _ = build_workflow_scientific_evidence_view(
        scientific_question="问题",
        hypothesis={"strategy_name": "策略", "reasoning": "正文"},
        workflow={
            "steps": [
                {
                    "step_id": "deck",
                    "step_name": "准备 deck",
                    "description": "转换输入",
                    "action_id": "xyz_to_gjf",
                    "request": {
                        "inputs": {"charge": 0, "multiplicity": 1},
                        "method_spec": {"input_format": "xyz"},
                    },
                }
            ]
        },
        execution_results={
            "steps": [
                {
                    "step_id": "deck",
                    "status": "completed",
                    "raw_output": {
                        "status": "completed",
                        "result": {
                            "gaussian_input": "0 1\nC 0.0 0.0 0.0\n",
                            "preflight_status": "passed",
                            "native_preflight": {
                                "native_preflight_ready": True,
                                "calculation_intent": "opt freq",
                                "link1_segment_count": 2,
                                "checks": ["route_valid", "/tmp/internal"],
                            },
                        },
                    },
                }
            ]
        },
    )
    step = view["workflow_evidence"]["steps"][0]
    assert [item["name"] for item in step["observations"]] == ["deck_preflight_status"]
    assert step["native_preflight"] == {
        "status": "ready",
        "ready": True,
        "calculation_intent": "opt freq",
        "link1_segment_count": 2,
        "checks": ["route_valid"],
    }
    assert "0.0 0.0" not in str(view)


def test_native_preflight_and_settings_keep_actual_and_requested_sources() -> None:
    """预检事实与科学设置按字段保留真实、请求和不可用三种状态。"""

    view, _ = build_workflow_scientific_evidence_view(
        scientific_question="问题",
        hypothesis={"strategy_name": "策略", "reasoning": "正文"},
        workflow={
            "steps": [
                {
                    "step_id": "deck",
                    "step_name": "准备 deck",
                    "description": "转换输入",
                    "action_id": "xyz_to_gjf",
                    "request": {
                        "inputs": {"charge": 0, "multiplicity": 1},
                        "method_spec": {"input_format": "xyz"},
                    },
                }
            ]
        },
        execution_results={
            "steps": [
                {
                    "step_id": "deck",
                    "status": "completed",
                    "result": {
                        "native_preflight": {"native_preflight_ready": True},
                        "effective_settings": {"charge": 0},
                    },
                }
            ]
        },
    )
    step = view["workflow_evidence"]["steps"][0]
    assert step["native_preflight"]["status"] == "ready"
    assert step["scientific_settings"] == {
        "charge": {"value": 0, "source": "verified_effective"},
        "multiplicity": {"value": 1, "source": "requested_only"},
        "input_format": {"value": "xyz", "source": "requested_only"},
    }


def test_native_preflight_not_ready_is_distinct_from_unavailable() -> None:
    """明确失败的 native_preflight 不能被压扁成缺失事实。"""

    view, _ = build_workflow_scientific_evidence_view(
        scientific_question="问题",
        hypothesis={"strategy_name": "策略", "reasoning": "正文"},
        workflow={
            "steps": [
                {
                    "step_id": "deck",
                    "step_name": "准备 deck",
                    "description": "预检输入",
                    "action_id": "xyz_to_gjf",
                }
            ]
        },
        execution_results={
            "steps": [
                {
                    "step_id": "deck",
                    "status": "completed",
                    "result": {
                        "native_preflight": {
                            "native_preflight_ready": False,
                            "checks": ["missing_charge"],
                        }
                    },
                }
            ]
        },
    )
    assert view["workflow_evidence"]["steps"][0]["native_preflight"]["status"] == "not_ready"


def test_native_gaussian_without_action_id_uses_registered_operation() -> None:
    """native_software Gaussian 步骤按软件事实映射为 native_gaussian。"""

    view, _ = build_workflow_scientific_evidence_view(
        scientific_question="问题",
        hypothesis={"strategy_name": "策略", "reasoning": "正文"},
        workflow={
            "explicit_conditions": [{"name": "solvent", "value": "acetone", "unit": None}],
            "steps": [
                {
                    "step_id": "native",
                    "step_name": "Gaussian 优化与频率",
                    "description": "执行 native Gaussian",
                    "execution_layer": "native_software",
                    "request": {"software_id": "gaussian"},
                }
            ],
        },
        execution_results={"steps": [{"step_id": "native", "status": "not_executed"}]},
    )
    step = view["workflow_evidence"]["steps"][0]
    assert step["operation"] == "native_gaussian"
    assert view["explicit_conditions"] == [
        {"name": "solvent", "value": "acetone", "unit": None}
    ]


def test_real_minichem_sdf_to_gjf_result_is_projected_to_safe_preflight_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """真实 MiniChem 格式转换结果只向专家暴露可复核的预检摘要。"""

    from minichem_toolbox.backends import interchange

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    sdf_response = interchange.execute(
        "smiles2sdf",
        "format_conversion",
        {"inputs": {"smiles_list": ["O"]}},
    )
    assert sdf_response["status"] == "success"
    sdf_path = sdf_response["artifact_files"][0]["path"]
    gjf_response = interchange.execute(
        "sdf_to_gjf",
        "format_conversion",
        {
            "inputs": {
                "sdf_path": sdf_path,
                "route_parameters": {
                    "result": {"gaussian_code": "#p B3LYP/6-31G(d) SP"}
                },
            }
        },
    )
    assert gjf_response["status"] == "success"

    view, _ = build_workflow_scientific_evidence_view(
        scientific_question="生成可复核的 Gaussian 输入。",
        hypothesis={"strategy_name": "格式转换", "reasoning": "检查输入预检。"},
        workflow={
            "steps": [
                {
                    "step_id": "deck",
                    "step_name": "准备 Gaussian 输入",
                    "description": "只检查输入预检结果。",
                    "action_id": "sdf_to_gjf",
                }
            ]
        },
        execution_results={
            "steps": [
                {
                    "step_id": "deck",
                    "status": "completed",
                    "result": gjf_response["result"],
                }
            ]
        },
    )
    step = view["workflow_evidence"]["steps"][0]
    assert step["coverage"]["status"] == "complete"
    summary = step["observations"][0]["value"]
    assert summary["status"] == "ready"
    assert summary["ready"] is True
    assert summary["calculation_intent"] == "single_point"
    assert summary["link1_segment_count"] == 1
    assert summary["checks"]
    serialized = json.dumps(view, ensure_ascii=False)
    assert "result.gjf" not in serialized
    assert "inputs/generated.gjf" not in serialized


def test_unknown_result_does_not_fallback_to_raw_output() -> None:
    """未注册操作显式不可用，不把结果字符串或日志当科学事实。"""

    view, _ = build_workflow_scientific_evidence_view(
        scientific_question="问题",
        hypothesis={"strategy_name": "策略", "reasoning": "正文"},
        workflow={
            "steps": [
                {
                    "step_id": "unknown",
                    "step_name": "未知",
                    "action_id": "not_registered",
                    "request": {"arguments": {"path": "/tmp/x"}},
                }
            ]
        },
        execution_results={
            "steps": [
                {
                    "step_id": "unknown",
                    "status": "completed",
                    "raw_output": {
                        "status": "completed",
                        "result": {"raw_output": "unregistered"},
                    },
                }
            ]
        },
    )
    step = view["workflow_evidence"]["steps"][0]
    assert step["operation"] == "unknown"
    assert step["coverage"]["status"] == "unavailable"
    assert step["observations"] == []
    assert "unregistered" not in str(view)


def test_context_is_stable_and_unregistered_custom_observables_are_hidden() -> None:
    """上下文条件保持可见，未注册可编程观测不能充当任意结果入口。"""

    view, _ = build_workflow_scientific_evidence_view(
        scientific_question="问题",
        scientific_context={
            "scientific_question": "问题",
            "chemistry_context": {
                "solvent": "acetone",
                "temperature_c": 30,
                "charge": 0,
            },
        },
        hypothesis={"strategy_name": "策略", "reasoning": "正文", "explicit_conditions": []},
        workflow={
            "explicit_conditions": [
                {"name": "solvent", "value": "acetone", "unit": None}
            ],
            "steps": [
                {
                    "step_id": "s1",
                    "step_name": "IR/Raman",
                    "description": "分析 IR/Raman",
                    "execution_layer": "programmable_analysis",
                    "scientific_observables": [
                        {"name": "ir_peaks", "unit": "cm-1"},
                        {"name": "raman_peaks", "unit": "cm-1"},
                    ],
                }
            ],
        },
        execution_results={
            "steps": [
                {
                    "step_id": "s1",
                    "status": "completed",
                    "result": {
                        "ir_peaks": [1, 2],
                        "raman_peaks": [3],
                        "metrics": {
                            "source_step": "s1",
                            "catalog_version": "secret",
                            "sha256": "secret",
                        },
                    },
                }
            ]
        },
    )

    assert view["explicit_conditions"] == [
        {"name": "solvent", "value": "acetone", "unit": None},
        {"name": "temperature", "value": 30, "unit": "°C"},
        {"name": "charge", "value": 0, "unit": None},
    ]
    step = view["workflow_evidence"]["steps"][0]
    assert step["coverage"]["status"] == "unavailable"
    assert step["observations"] == []
    assert "ir_peaks" not in str(view)
    assert "raman_peaks" not in str(view)
    assert "catalog_version" not in str(view)
    assert "source_step" not in str(view)
    assert "sha256" not in str(view)


def test_programmable_analysis_projects_declared_named_observables() -> None:
    """可编程分析只筛选声明且注册的具名结果，并保留布尔值。"""

    view, _ = build_workflow_scientific_evidence_view(
        scientific_question="判断分析结果。",
        hypothesis={"strategy_name": "分析", "reasoning": "使用具名输出。"},
        workflow={
            "steps": [
                {
                    "step_id": "analysis",
                    "step_name": "可编程分析",
                    "description": "输出支持判断和能量。",
                    "execution_layer": "programmable_analysis",
                    "scientific_observables": [
                        {"name": "ir_peaks", "unit": "cm-1"},
                    ],
                    "request": {
                        "runtime": "python",
                        "script_path": "code/analyze.py",
                        "outputs": [
                            {"name": "observed_support", "semantic_type": "scalar", "path": "outputs/support.json"},
                            {"name": "energy", "semantic_type": "scalar", "path": "outputs/energy.json", "json_schema": {"unit": "hartree"}},
                            {"name": "missing_result", "semantic_type": "scalar", "path": "outputs/missing.json"},
                        ],
                    },
                }
            ]
        },
        execution_results={
            "steps": [
                {
                    "step_id": "analysis",
                    "status": "completed",
                    "result": {
                        "observed_support": False,
                        "energy": -1.5,
                        "metrics": {"source_step": "hidden"},
                    },
                }
            ]
        },
    )
    step = view["workflow_evidence"]["steps"][0]
    assert [item["name"] for item in step["expected_observables"]] == [
        "observed_support",
        "energy",
    ]
    assert {item["name"] for item in step["observations"]} == {
        "observed_support",
        "energy",
    }
    assert step["coverage"] == {"status": "complete", "missing_observables": []}
    assert "hidden" not in json.dumps(view, ensure_ascii=False)


@pytest.mark.parametrize(
    "step",
    [
        {
            "step_id": "native",
            "step_name": "Gaussian 计算",
            "description": "运行 Gaussian",
            "execution_layer": "native_software",
            "backend_id": "xtb",
        },
        {
            "step_id": "native",
            "step_name": "Gaussian 计算",
            "description": "运行 Gaussian",
            "action_id": "run_gaussian",
            "backend_id": "xtb",
        },
    ],
)
def test_non_gaussian_backend_cannot_be_classified_as_native_gaussian(step) -> None:
    """显式非 Gaussian backend 优先于 Action 名称和自然语言描述。"""

    view, _ = build_workflow_scientific_evidence_view(
        scientific_question="问题",
        hypothesis={"strategy_name": "策略", "reasoning": "正文"},
        workflow={"steps": [step]},
        execution_results={"steps": []},
    )

    projected = view["workflow_evidence"]["steps"][0]
    assert projected["operation"] == "unknown"
    assert projected["coverage"]["status"] == "unavailable"


def test_execution_linkage_rejects_duplicate_or_external_step_ids() -> None:
    """执行事实必须能唯一连接到规范步骤，不能静默丢掉额外记录。"""

    kwargs = {
        "scientific_question": "问题",
        "hypothesis": {"strategy_name": "策略", "reasoning": "正文"},
        "workflow": {
            "steps": [
                {"step_id": "s1", "step_name": "一步", "description": "目的"}
            ]
        },
    }
    import pytest

    with pytest.raises(ValueError, match="workflow 外"):
        build_workflow_scientific_evidence_view(
            **kwargs, execution_results={"steps": [{"step_id": "s10"}]}
        )
    with pytest.raises(ValueError, match="重复"):
        build_workflow_scientific_evidence_view(
            **kwargs,
            execution_results={"steps": [{"step_id": "s1"}, {"step_id": "s1"}]},
        )


def test_native_program_checks_do_not_duplicate_termination_observations() -> None:
    """Gaussian termination/convergence 只作为程序检查出现一次。"""

    view, _ = build_workflow_scientific_evidence_view(
        scientific_question="问题",
        hypothesis={"strategy_name": "策略", "reasoning": "正文"},
        workflow={
            "steps": [
                {
                    "step_id": "s1",
                    "step_name": "Gaussian",
                    "description": "计算",
                    "execution_layer": "native_software",
                    "request": {"software_id": "gaussian"},
                }
            ]
        },
        execution_results={
            "steps": [
                {
                    "step_id": "s1",
                    "status": "completed",
                    "result": {
                        "electronic_energy": -1.0,
                        "normal_termination": True,
                        "optimization_convergence": True,
                    },
                }
            ]
        },
    )
    step = view["workflow_evidence"]["steps"][0]
    assert [item["name"] for item in step["observations"]] == ["electronic_energy"]
    assert [item["name"] for item in step["program_checks"]] == [
        "normal_termination",
        "optimization_convergence",
    ]


def test_registered_program_checks_match_formal_step_boundary() -> None:
    """s1 的检查不能因为 startswith 误接到 s10。"""

    view, _ = build_workflow_scientific_evidence_view(
        scientific_question="问题",
        hypothesis={"strategy_name": "策略", "reasoning": "正文"},
        workflow={
            "steps": [
                {"step_id": "s1", "step_name": "一步", "description": "目的"},
                {"step_id": "s10", "step_name": "十步", "description": "目的"},
            ]
        },
        execution_results={"steps": []},
        program_validation={
            "registered_check_results": [
                {
                    "check_instance_id": "s10:normal_termination",
                    "evaluation": "matched",
                    "requirement": {"expected": True},
                    "actual": {"normal_termination": True},
                }
            ]
        },
    )
    steps = view["workflow_evidence"]["steps"]
    assert steps[0]["program_checks"] == []
    assert len(steps[1]["program_checks"]) == 1


def test_registered_program_checks_deduplicate_by_identity_and_science_content() -> None:
    """重复注册检查只生成一个证据，科学内容变化时保留独立事实。"""

    duplicate = {
        "check_instance_id": "action-step:converged",
        "evaluation": "matched",
        "requirement": {"expected": True},
        "actual": {"converged": True},
    }
    changed = {
        **duplicate,
        "evaluation": "not_matched",
        "actual": {"converged": False},
    }
    view, evidence_index = build_workflow_scientific_evidence_view(
        scientific_question="问题",
        hypothesis={"strategy_name": "策略", "reasoning": "正文"},
        workflow={
            "steps": [
                {
                    "step_id": "action-step",
                    "step_name": "优化",
                    "description": "优化结构",
                    "action_id": "optimize_geometry",
                }
            ]
        },
        execution_results={
            "steps": [
                {"step_id": "action-step", "status": "completed", "result": {}}
            ]
        },
        program_validation={
            "registered_check_results": [duplicate, dict(duplicate), changed]
        },
    )

    checks = view["workflow_evidence"]["steps"][0]["program_checks"]
    assert len(checks) == 2
    assert [item["evidence_ref"] for item in checks] == ["e1", "e2"]
    assert [item["status"] for item in checks] == ["passed", "failed"]
    assert [item["evidence_ref"] for item in evidence_index] == ["e1", "e2"]


def _registered_result_view(*, action_id: str, result: dict[str, Any]) -> dict[str, Any]:
    """构造单个 MiniChem Action 的执行后固定视图夹具。"""

    view, _ = build_workflow_scientific_evidence_view(
        scientific_question="比较振动与光谱观测。",
        hypothesis={"strategy_name": "频率策略", "reasoning": "只依据结构化结果。"},
        workflow={
            "steps": [
                {
                    "step_id": "action-step",
                    "step_name": action_id,
                    "description": "Case 1 科学结果",
                    "action_id": action_id,
                    "depends_on": [],
                }
            ]
        },
        execution_results={
            "steps": [
                {
                    "step_id": "action-step",
                    "status": "completed",
                    "result": result,
                }
            ]
        },
    )
    return view["workflow_evidence"]["steps"][0]


def test_free_energy_selectivity_action_projects_numeric_scientific_evidence() -> None:
    """发布自由能重放不能只留下 Artifact 类型检查而丢掉数值结果。"""

    step = _registered_result_view(
        action_id="analyze_free_energy_selectivity",
        result={
            "delta_delta_g_kcal_mol": 1.8848503030603752,
            "populations": [
                {
                    "label": "R-major",
                    "population": 0.9580649124037962,
                    "population_percent": 95.80649124037961,
                },
                {
                    "label": "S-minor",
                    "population": 0.041935087596203846,
                    "population_percent": 4.193508759620385,
                },
            ],
            "population_sum": 1.0,
            "ee_percent": 91.61298248075924,
            "transport": {"raw_output": "must not leak"},
        },
    )

    assert step["operation"] == "free_energy_selectivity"
    assert step["coverage"] == {"status": "complete", "missing_observables": []}
    assert [item["name"] for item in step["observations"]] == [
        "delta_delta_g_kcal_mol",
        "populations",
        "population_sum",
        "ee_percent",
    ]
    assert [item["unit"] for item in step["observations"]] == [
        "kcal/mol",
        None,
        None,
        "%",
    ]
    assert step["observations"][1]["value"] == [
        {
            "label": "R-major",
            "population": 0.9580649124037962,
            "population_percent": 95.80649124037961,
        },
        {
            "label": "S-minor",
            "population": 0.041935087596203846,
            "population_percent": 4.193508759620385,
        },
    ]
    assert "transport" not in json.dumps(step, ensure_ascii=False)


def test_generate_3d_structure_action_projects_registered_structure_summary() -> None:
    """当前 MiniChem 三维结构 Action 应是已知操作，并保留安全的结构摘要。"""

    step = _registered_result_view(
        action_id="generate_3d_structure",
        result={
            "random_seed": 20260821,
            "structure": {
                "atom_count": 26,
                "charge": 0,
                "multiplicity": 1,
                "element_counts": {"C": 10, "H": 11, "N": 1, "O": 4},
                "pbc": [False, False, False],
                "path": "/private/generated.sdf",
            },
        },
    )

    assert step["operation"] == "structure_generation"
    assert step["coverage"] == {"status": "complete", "missing_observables": []}
    assert step["observations"] == [
        {
            "evidence_ref": "e1",
            "name": "generated_structure",
            "value": {
                "atom_count": 26,
                "charge": 0,
                "multiplicity": 1,
                "element_counts": {"C": 10, "H": 11, "N": 1, "O": 4},
                "pbc": [False, False, False],
            },
            "unit": None,
            "conditions": {},
        }
    ]
    assert "/private/generated.sdf" not in json.dumps(step, ensure_ascii=False)


def test_registered_artifact_check_does_not_hide_optimization_convergence() -> None:
    """Artifact 合同检查存在时，独立的优化收敛事实仍须进入专家证据。"""

    view, evidence_index = build_workflow_scientific_evidence_view(
        scientific_question="验证 GPU 几何优化。",
        hypothesis={"strategy_name": "优化", "reasoning": "检查真实收敛事实。"},
        workflow={
            "steps": [
                {
                    "step_id": "optimize",
                    "step_name": "优化结构",
                    "description": "执行有界几何优化。",
                    "action_id": "optimize_geometry",
                }
            ]
        },
        execution_results={
            "steps": [
                {
                    "step_id": "optimize",
                    "status": "completed",
                    "result": {
                        "structure": {"atom_count": 26, "charge": 0, "multiplicity": 1},
                        "converged": True,
                    },
                }
            ]
        },
        program_validation={
            "registered_check_results": [
                {
                    "check_instance_id": "optimize:output:0",
                    "evaluation": "matched",
                    "requirement": {},
                    "actual": {
                        "semantic_type": "AtomicStructure",
                        "media_type": "application/json",
                    },
                }
            ]
        },
    )

    checks = view["workflow_evidence"]["steps"][0]["program_checks"]
    assert [item["name"] for item in checks] == [
        "registered_program_check",
        "optimization_convergence",
    ]
    assert checks[1]["status"] == "passed"
    assert checks[1]["observed"] == {"optimization_convergence": True}
    assert [item["evidence_ref"] for item in evidence_index] == ["e1", "e2", "e3"]


def test_case1_registered_vibrational_modes_is_summary_without_modes_or_energies() -> None:
    """振动 Action 只保留计数和完整虚频数值对，不复制模式坐标大数组。"""

    step = _registered_result_view(
        action_id="derive_vibrational_modes",
        result={
            "frequencies_cm1": [
                {"real": 100.0, "imaginary": 0.0},
                {"real": 0.0, "imaginary": 123.4},
                {"real": 1500.0, "imaginary": 0.0},
            ],
            "vibrational_energies_ev": [1.0, 2.0, 3.0],
            "modes": [["MODE_INTERNAL_SENTINEL"]],
            "path": "/tmp/MODE_PATH_SENTINEL",
        },
    )
    assert step["operation"] == "vibrational_modes"
    assert step["expected_observables"] == [
        {"name": "vibrational_mode_summary", "unit": None}
    ]
    assert step["coverage"]["missing_observables"] == []
    summary = step["observations"][0]["value"]
    assert summary == {
        "total_count": 3,
        "real_count": 2,
        "imaginary_count": 1,
        "frequency_unit": "cm-1",
        "imaginary_frequencies_cm1": [{"real": 0.0, "imaginary": 123.4}],
    }
    assert "MODE_INTERNAL_SENTINEL" not in str(step)
    assert "MODE_PATH_SENTINEL" not in str(step)


def test_case1_registered_ir_and_raman_keep_aligned_line_arrays_only() -> None:
    """IR/Raman 规范化 nested SpectrumResult，丢弃 dense 网格并保留单位。"""

    ir = _registered_result_view(
        action_id="derive_ir_spectrum",
        result={
            "schema_name": "SpectrumResult",
            "axis": {"unit": "inverse_centimeter", "values": [900.0, 1000.0, 1100.0]},
            "intensity": {
                "unit": "kilometer_per_mole",
                "values": [0.0, 1.0, 0.0],
                "path": "/tmp/INTENSITY_PATH_SENTINEL",
            },
            "sticks": [
                {"position": 1000.0, "intensity": 8.0},
                {"position": 1200.0, "intensity": 15.0},
            ],
            "line_shape": {"kind": "gaussian", "fwhm": 20.0},
            "point_count": 3,
            "stick_count": 2,
            "raw_output": "RAW_OUTPUT_SENTINEL",
        },
    )
    raman = _registered_result_view(
        action_id="derive_raman_spectrum",
        result={
            "schema_name": "SpectrumResult",
            "axis": {"unit": "inverse_centimeter", "values": [990.0, 1000.0, 1010.0]},
            "intensity": {"unit": "A^4/amu", "values": [0.0, 4.0, 0.0]},
            "sticks": [{"position": 1000.0, "intensity": 4.5}],
            "line_shape": {"kind": "lorentzian", "fwhm": 10.0},
            "point_count": 3,
            "stick_count": 1,
        },
    )
    for step, name, intensity_unit, broadening in (
        (ir, "ir_line_spectrum", "kilometer_per_mole", "gaussian"),
        (raman, "raman_line_spectrum", "A^4/amu", "lorentzian"),
    ):
        assert step["operation"] in {"ir_spectrum", "raman_spectrum"}
        assert step["expected_observables"] == [{"name": name, "unit": None}]
        assert step["coverage"]["missing_observables"] == []
        summary = step["observations"][0]["value"]
        assert summary["frequency_unit"] == "cm-1"
        assert summary["intensity_unit"] == intensity_unit
        assert summary["broadening"] == broadening
        assert summary["point_count"] == 3
        assert summary["stick_count"] in {1, 2}
        assert "values" not in summary
        assert "RAW_OUTPUT_SENTINEL" not in str(step)
        assert "INTENSITY_PATH_SENTINEL" not in str(step)


def test_case1_registered_spectrum_rejects_mismatched_source_counts() -> None:
    """线谱数组或 source count 不一致时只能报告缺失，不生成半真摘要。"""

    step = _registered_result_view(
        action_id="derive_ir_spectrum",
        result={
            "wavenumber_cm1": [900.0, 1000.0],
            "intensity": [0.0, 1.0, 0.0],
            "line_frequencies_cm1": [1000.0],
            "line_intensities": [8.0],
            "point_count": 99,
            "stick_count": 1,
        },
    )
    assert step["observations"] == []
    assert step["coverage"]["status"] == "unavailable"
    assert step["coverage"]["missing_observables"] == ["ir_line_spectrum"]


def test_case1_registered_structure_hessian_optimization_and_plot_operations() -> None:
    """基线链的结构转换、优化、Hessian 和绘图均有受控分类。"""

    cases = [
        (
            "smiles2sdf",
            {"success": True, "input_count": 1, "molecules_written": 1, "output_sdf_path": "/tmp/x.sdf"},
            "structure_conversion",
            "molecules_written",
        ),
        (
            "optimize_geometry",
            {"structure": {"atoms": [{"element": "O", "position_angstrom": [0.0, 0.0, 0.0]}]}, "converged": True},
            "geometry_optimization",
            "optimized_geometry",
        ),
        (
            "calculate_hessian",
            {"matrix": [[1.0, 0.1], [0.1, 2.0]], "unit": "eV/angstrom^2"},
            "hessian_calculation",
            "hessian_summary",
        ),
        (
            "render_spectrum_plot",
            {"schema_name": "PlotArtifact", "path": "/tmp/plot.svg", "format": "svg"},
            "spectrum_plot",
            None,
        ),
    ]
    for action_id, result, operation, observable in cases:
        step = _registered_result_view(action_id=action_id, result=result)
        assert step["operation"] == operation
        assert step["coverage"]["missing_observables"] == []
        if observable is None:
            assert step["expected_observables"] == []
            assert step["observations"] == []
        else:
            assert [item["name"] for item in step["observations"]] == [observable]
        assert "/tmp/" not in str(step)


def test_real_structure_result_keeps_registered_periodic_and_atomic_fields() -> None:
    """MiniChem AtomicStructure 保留注册字段，同时隐藏解析器和路径元数据。"""

    structure = {
        "atoms": [
            {
                "element": "C",
                "atomic_number": 6,
                "position_angstrom": [0.0, 0.1, 0.2],
            },
            {
                "element": "H",
                "atomic_number": 1,
                "position_angstrom": [1.0, 0.0, 0.0],
            },
        ],
        "cell_angstrom": [
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
        ],
        "pbc": [True, True, True],
        "charge": 0,
        "multiplicity": 1,
        "metadata": {
            "source_path": "outputs/final.xyz",
            "parser": "cclib",
            "normal_termination": True,
        },
    }

    step = _registered_result_view(
        action_id="optimize_geometry",
        result={"structure": structure, "metadata": {"source_path": "/tmp/hidden"}},
    )

    value = step["observations"][0]["value"]
    assert value["cell_angstrom"] == structure["cell_angstrom"]
    assert value["pbc"] == [True, True, True]
    assert value["atoms"] == structure["atoms"]
    assert value["charge"] == 0
    assert value["multiplicity"] == 1
    serialized = json.dumps(step, ensure_ascii=False)
    for marker in ("source_path", "outputs/final.xyz", "parser", "cclib", "/tmp/hidden"):
        assert marker not in serialized


def test_scientific_settings_report_per_field_effective_status() -> None:
    """设置字段逐项区分执行后确认、请求计划和不可用。"""

    view, _ = build_workflow_scientific_evidence_view(
        scientific_question="验证优化设置。",
        hypothesis={"strategy_name": "优化", "reasoning": "比较执行设置。"},
        workflow={
            "steps": [
                {
                    "step_id": "optimization",
                    "step_name": "优化",
                    "description": "执行几何优化。",
                    "action_id": "optimize_geometry",
                    "request": {
                        "method_spec": {"method": "B3LYP", "basis": "6-31G(d)"},
                        "action_settings": {"optimization_level": "tight"},
                    },
                }
            ]
        },
        execution_results={
            "steps": [
                {
                    "step_id": "optimization",
                    "status": "completed",
                    "result": {
                        "effective_settings": {"method": "B3LYP"},
                        "optimized_geometry": {"atoms": []},
                    },
                }
            ]
        },
    )
    step = view["workflow_evidence"]["steps"][0]
    settings = step["scientific_settings"]
    assert "scientific_settings_source" not in step
    assert settings["method"] == {"value": "B3LYP", "source": "verified_effective"}
    assert settings["basis"] == {"value": "6-31G(d)", "source": "requested_only"}
    assert settings["optimization_level"] == {
        "value": "tight",
        "source": "requested_only",
    }
    assert settings["optimizer"] == {"value": None, "source": "unavailable"}
    assert all(
        set(item) == {"value", "source"}
        and item["source"] in {"verified_effective", "requested_only", "unavailable"}
        for item in settings.values()
    )


def test_parser_properties_result_is_not_repeated_as_verified_setting() -> None:
    """cclib properties 是科学结果块，不能冒充已确认的请求设置。"""

    view, _ = build_workflow_scientific_evidence_view(
        scientific_question="解析当前 Gaussian 结果。",
        hypothesis={"strategy_name": "解析", "reasoning": "读取结构和能量。"},
        workflow={
            "steps": [
                {
                    "step_id": "parser",
                    "step_name": "解析",
                    "description": "解析结构、能量和频率。",
                    "action_id": "parse_quantum_chemistry_output",
                    "request": {
                        "method_spec": {
                            "properties": ["atom_coordinates", "energies"]
                        }
                    },
                }
            ]
        },
        execution_results={
            "steps": [
                {
                    "step_id": "parser",
                    "status": "completed",
                    "result": {
                        "properties": {
                            "atomcoords": [[0.0, 0.1, 0.2]],
                            "scfenergies": [-100.0],
                            "server_info": "HIDDEN_SERVER_SENTINEL",
                            "custom_field": "HIDDEN_CUSTOM_SENTINEL",
                        },
                        "attribute_units": {
                            "atomcoords": "angstrom",
                            "scfenergies": "eV",
                        },
                    },
                }
            ]
        },
    )

    step = view["workflow_evidence"]["steps"][0]
    assert step["scientific_settings"]["properties"] == {
        "value": ["atom_coordinates", "energies"],
        "source": "requested_only",
    }
    serialized = json.dumps(step, ensure_ascii=False)
    assert "HIDDEN_SERVER_SENTINEL" not in serialized
    assert "HIDDEN_CUSTOM_SENTINEL" not in serialized


def test_unknown_operation_has_no_scientific_settings() -> None:
    """未知操作不猜测设置字段，固定返回空 object。"""

    view, _ = build_workflow_scientific_evidence_view(
        scientific_question="问题",
        hypothesis={"strategy_name": "策略", "reasoning": "正文"},
        workflow={
            "steps": [
                {
                    "step_id": "unknown",
                    "step_name": "未知",
                    "description": "未知操作",
                    "action_id": "unregistered_action",
                    "request": {"method_spec": {"method": "B3LYP"}},
                }
            ]
        },
        execution_results={"steps": []},
    )
    step = view["workflow_evidence"]["steps"][0]
    assert step["operation"] == "unknown"
    assert step["scientific_settings"] == {}


def test_derived_failure_and_program_check_text_cannot_leak_internal_markers() -> None:
    """失败代码和检查包装清理 job/operation/runtime 等派生内部文本。"""

    view, index = build_workflow_scientific_evidence_view(
        scientific_question="问题",
        hypothesis={"strategy_name": "策略", "reasoning": "合法科学正文。"},
        workflow={
            "steps": [
                {
                    "step_id": "action-step",
                    "step_name": "优化",
                    "description": "优化结构",
                    "action_id": "optimize_geometry",
                }
            ]
        },
        execution_results={
            "steps": [
                {
                    "step_id": "action-step",
                    "status": "failed",
                    "error": {
                        "category": "raw_log",
                        "code": (
                            "step_id-01234567-89ab-cdef-0123-456789abcdef "
                            "raw_output.log runtime=/tmp/x"
                        ),
                        "reason_code": "operation_id=SECRET",
                        "stage": "scf_failed",
                        "validation_errors": ["raw_log", "operation_id=SECRET"],
                    },
                }
            ]
        },
        program_validation={
            "registered_check_results": [
                {
                    "check_instance_id": "action-step:converged",
                    "evaluation": "not_matched",
                    "requirement": {"expected": "operation_id=SECRET", "science": "SCF"},
                    "actual": {
                        "status": "job_id=SECRET",
                        "converged": False,
                        "validation_errors": ["raw_log", "operation_id=SECRET"],
                    },
                }
            ]
        },
    )
    joined = json.dumps({"view": view, "index": index}, ensure_ascii=False)
    for marker in (
        "job_id=SECRET",
        "raw_output.log",
        "runtime=/tmp/x",
        "operation_id=SECRET",
        "step_id-01234567-89ab-cdef-0123-456789abcdef",
    ):
        assert marker not in joined
    failure = view["workflow_evidence"]["steps"][0]["failure"]
    assert failure["category"] == "execution"
    assert failure["code"] == "execution_failed"
    assert failure["details"]["stage"] == "scf_failed"
    assert "validation_errors" not in failure["details"]
    check = view["workflow_evidence"]["steps"][0]["program_checks"][0]
    assert check["observed"]["validation_errors"] == []


def test_failure_view_projects_actionable_details_with_stable_shape() -> None:
    """专家执行后视图读取 MiniChem 失败诊断但不接收运行身份。"""

    view, _ = build_workflow_scientific_evidence_view(
        scientific_question="判断结构输入是否正确。",
        hypothesis={"strategy_name": "输入校验", "reasoning": "检查元素计数。"},
        workflow={
            "steps": [
                {
                    "step_id": "validate",
                    "step_name": "校验结构",
                    "description": "校验输入结构",
                    "action_id": "validate_structure",
                }
            ]
        },
        execution_results={
            "steps": [
                {
                    "step_id": "validate",
                    "status": "failed",
                    "retryable": True,
                    "error": {
                        "code": "backend_input_error",
                        "message": (
                            "元素不守恒：C 计数不一致；文件=/tmp/failure.log；"
                            "source step step_id-01234567-89ab-cdef-0123-456789abcdef；"
                            "operation_instance_id-abcdefab-cdef-abcd-efab-cdefabcdefab"
                        ),
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
            ]
        },
    )
    failure = view["workflow_evidence"]["steps"][0]["failure"]
    details = failure["details"]
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
    joined = json.dumps(view, ensure_ascii=False)
    for marker in (
        "TRACEBACK_SENTINEL",
        "ACTION_ID_SENTINEL",
        "BACKEND_ID_SENTINEL",
        "UNKNOWN_SENTINEL",
        "operation_id=hidden",
        "operation_instance_id-abcdefab-cdef-abcd-efab-cdefabcdefab",
        "step_id-01234567-89ab-cdef-0123-456789abcdef",
        "abcdefab-cdef-abcd-efab-cdefabcdefab",
        "/tmp/failure.log",
    ):
        assert marker not in joined


def test_failure_view_removes_windows_drive_and_unc_paths_without_route_false_positives() -> None:
    """失败诊断清理 Windows 盘符/UNC 路径，同时保留科学 route 和表达式。"""

    failure_messages = [
        r"读取失败：C:\Users\secret\x.log",
        r"读取失败：\\server\share\secret",
        "读取失败：D:/private/x",
        "route B3LYP/6-31G(d)；basis=/",
        "参考文献：https://server/share",
    ]
    view, _ = build_workflow_scientific_evidence_view(
        scientific_question="问题",
        hypothesis={"strategy_name": "策略", "reasoning": "正文"},
        workflow={
            "steps": [
                {
                    "step_id": "windows-paths",
                    "step_name": "路径诊断",
                    "description": "诊断",
                    "action_id": "validate_structure",
                }
            ]
        },
        execution_results={
            "steps": [
                {
                    "step_id": "windows-paths",
                    "status": "failed",
                    "error": {"message": "；".join(failure_messages)},
                }
            ]
        },
    )

    message = view["workflow_evidence"]["steps"][0]["failure"]["details"]["message"]
    assert "C:" not in message
    assert r"\Users\secret\x.log" not in message
    assert r"\\server\share\secret" not in message
    assert "D:" not in message
    assert "B3LYP/6-31G(d)" in message
    assert "basis=/" in message
    assert "https://server/share" in message


def test_source_scientific_question_and_hypothesis_text_is_verbatim() -> None:
    """来源固定的科学正文即使提到协议词也不能被派生文本清理。"""

    question = "请讨论 /tmp 只是示例路径，以及 raw_output 在论文中的字面含义。"
    reasoning = "推理保留 raw_output 和 /tmp 的原文，不代表运行时字段。"
    view, _ = build_workflow_scientific_evidence_view(
        scientific_question=question,
        hypothesis={"strategy_name": "策略", "reasoning": reasoning},
        workflow={
            "steps": [
                {
                    "step_id": "s1",
                    "step_name": "空步骤",
                    "description": "没有执行结果",
                    "action_id": "render_spectrum_plot",
                }
            ]
        },
        execution_results={"steps": []},
    )
    assert view["scientific_question"] == question
    assert view["hypothesis"]["reasoning"] == ""


def test_explicit_conditions_use_scientific_allowlist_and_scalar_values() -> None:
    """专家条件只保留注册科学量，别名归一化且不接受内部对象或路径。"""

    question = "用户问题保留 /tmp/example 字样。"
    hypothesis = {
        "strategy_name": "条件筛选",
        "reasoning": "使用明确实验条件。",
        "conditions": [
            {"name": "temperature_K", "value": 303.15},
            {"name": "solvent", "value": "acetone"},
            {"name": "pressure_pa", "value": 101325},
            {"name": "concentration_m", "value": 0.1},
            {"name": "pH", "value": 7},
            {"name": "time_s", "value": 60},
            {"name": "atmosphere", "value": "N2"},
            {"name": "charge", "value": 0},
            {"name": "multiplicity", "value": 1},
            {"name": "operation_id", "value": "op-secret"},
            {"name": "raw_output", "value": "RAW_SENTINEL"},
            {"name": "temperature", "value": {"raw_output": "RAW_SENTINEL"}},
            {"name": "temperature", "value": "/tmp/secret"},
            {"name": "solvent", "value": "water", "unit": "secret"},
            {"name": "unknown_condition", "value": "drop-me"},
        ],
    }
    pre = build_gaussian_step_review_view(
        scientific_question=question,
        hypothesis=hypothesis,
        step={"step_name": "检查", "description": "检查条件", "request": {"inputs": {}}},
        route="#p opt",
        converter_kind="xyz",
    )
    conditions = pre["explicit_conditions"]
    assert conditions == [
        {"name": "temperature", "value": 303.15, "unit": "K"},
        {"name": "solvent", "value": "acetone", "unit": None},
        {"name": "pressure", "value": 101325, "unit": "Pa"},
        {"name": "concentration", "value": 0.1, "unit": "M"},
        {"name": "pH", "value": 7, "unit": None},
        {"name": "time", "value": 60, "unit": "s"},
        {"name": "atmosphere", "value": "N2", "unit": None},
        {"name": "charge", "value": 0, "unit": None},
        {"name": "multiplicity", "value": 1, "unit": None},
    ]
    serialized = json.dumps(pre, ensure_ascii=False, allow_nan=False)
    for marker in ("op-secret", "RAW_SENTINEL", "/tmp/secret", "drop-me", "secret"):
        assert marker not in serialized
    assert question in serialized


def test_post_conditions_filter_workflow_internal_facts_without_schema_drift() -> None:
    """执行后视图同样只读取固定科学条件，缺失时仍保留 explicit_conditions。"""

    question = "问题中的 /tmp/user-provided 字样不能被改写。"
    view, _ = build_workflow_scientific_evidence_view(
        scientific_question=question,
        scientific_context={
            "chemistry_context": {
                "solvent": "water",
                "temperature": "298.15 K",
                "operation_id": "OP_SENTINEL",
            },
            "conditions": [
                {"name": "atmosphere", "value": "argon"},
                {"name": "transport_id", "value": "TRANSPORT_SENTINEL"},
            ],
        },
        hypothesis={"strategy_name": "策略", "reasoning": "正文"},
        workflow={
            "conditions": [
                {"name": "time", "value": 30, "unit": "s"},
                {"name": "workflow_id", "value": "WORKFLOW_SENTINEL"},
            ],
            "steps": [
                {
                    "step_id": "s1",
                    "step_name": "空步骤",
                    "description": "未执行",
                    "action_id": "render_spectrum_plot",
                }
            ],
        },
        execution_results={"steps": []},
    )
    assert view["explicit_conditions"] == [
        {"name": "solvent", "value": "water", "unit": None},
        {"name": "temperature", "value": "298.15 K", "unit": None},
        {"name": "atmosphere", "value": "argon", "unit": None},
        {"name": "time", "value": 30, "unit": "s"},
    ]
    serialized = json.dumps(view, ensure_ascii=False, allow_nan=False)
    for marker in ("OP_SENTINEL", "TRANSPORT_SENTINEL", "WORKFLOW_SENTINEL"):
        assert marker not in serialized
    assert question in serialized


def test_gaussian_pre_view_cleans_derived_fields_and_structure_keys() -> None:
    """执行前派生字段拒绝内部标记，同时保留合法 Gaussian 科学参数。"""

    view = build_gaussian_step_review_view(
        scientific_question="问题",
        hypothesis={
            "strategy_name": "策略 operation_id=HYPOTHESIS_SENTINEL",
            "reasoning": "科学推理 raw_output=/tmp/reasoning.log",
        },
        step={
            "step_name": "Gaussian 输入",
            "description": "生成输入",
            "request": {
                "calculation_type": "operation_id=CALC_SENTINEL",
                "inputs": {
                    "title": "催化剂复合物",
                    "mem": "16GB",
                    "chk_file": "/mnt/jobs/valid.chk",
                    "structure_summary": {
                        "summary_status": "available",
                        "formula": "C2H3NO",
                        "atom_count": 7,
                        "element_counts": {
                            "C": 2,
                            "N": 1,
                            "O": 1,
                            "operation_id": 99,
                            "Xx": 4,
                            "H": -1,
                        },
                    },
                }
            },
        },
        route="#p opt freq",
        converter_kind="xyz",
    )
    unit = view["gaussian_unit"]
    assert unit["calculation_type"] == "optimization+frequency"
    assert unit["structure"]["formula"] == "C2H3NO"
    assert unit["structure"]["element_counts"] == {"C": 2, "N": 1, "O": 1}
    assert unit["editable_parameters"] == {
        "charge": None,
        "multiplicity": None,
        "title": "催化剂复合物",
        "nprocshared": None,
        "mem": "16GB",
        "chk_file": "valid.chk",
    }
    serialized = json.dumps(view, ensure_ascii=False, allow_nan=False)
    for marker in (
        "HYPOTHESIS_SENTINEL",
        "CALC_SENTINEL",
        "operation_id",
        "raw_output",
        "/tmp/reasoning.log",
    ):
        assert marker not in serialized


def test_expert_safe_values_keep_scientific_words_and_output_settings() -> None:
    """普通科学词和 output_format 保留，嵌入原始响应的字段仍被过滤。"""

    view = build_gaussian_step_review_view(
        scientific_question="问题",
        hypothesis={
            "strategy_name": "例外情况策略",
            "reasoning": "an exception to the rule is scientifically relevant",
            "validation_requirements": [
                {
                    "output_format": "json",
                    "expected_output": "energy",
                    "raw_output": "RAW_VALIDATION_SENTINEL",
                    "operation_id": "OP_VALIDATION_SENTINEL",
                }
            ],
        },
        step={
            "step_name": "准备",
            "description": "准备输入",
            "request": {
                "inputs": {},
                "method_spec": {
                    "output_format": "json",
                    "expected_output": "energy",
                },
            },
        },
        route="#p opt",
        converter_kind="xyz",
    )
    assert view["hypothesis"]["reasoning"] == (
        "an exception to the rule is scientifically relevant"
    )
    assert view["gaussian_unit"]["validation_requirements"] == [
        {"output_format": "json", "expected_output": "energy"}
    ]
    serialized = json.dumps(view, ensure_ascii=False, allow_nan=False)
    assert "output_format" in serialized
    for marker in ("RAW_VALIDATION_SENTINEL", "OP_VALIDATION_SENTINEL"):
        assert marker not in serialized


def test_expert_registered_validation_and_settings_drop_unknown_nested_mappings() -> None:
    """Expert 的注册字段即使是对象，也不能递归接收未知嵌套键。"""

    pre_view = build_gaussian_step_review_view(
        scientific_question="问题",
        hypothesis={
            "strategy_name": "策略",
            "reasoning": "正文",
            "validation_requirements": [
                {
                    "output_format": "json",
                    "expected_output": {"name": "energy", "unknown": "VALIDATION_SENTINEL"},
                    "unknown_requirement": {"secret": "VALIDATION_SENTINEL"},
                }
            ],
        },
        step={"step_name": "准备", "description": "准备输入", "request": {"inputs": {}}},
        route="#p opt",
        converter_kind="xyz",
    )
    validation = pre_view["gaussian_unit"]["validation_requirements"][0]
    assert "unknown_requirement" not in validation
    assert "VALIDATION_SENTINEL" not in json.dumps(pre_view, ensure_ascii=False)

    view, _ = build_workflow_scientific_evidence_view(
        scientific_question="问题",
        hypothesis={"strategy_name": "策略", "reasoning": "正文"},
        workflow={
            "steps": [
                {
                    "step_id": "analysis",
                    "step_name": "分析",
                    "description": "分析",
                    "execution_layer": "programmable_analysis",
                    "request": {
                        "outputs": [
                            {"name": "energy", "semantic_type": "scalar"},
                            {"name": "energy", "semantic_type": "scalar"},
                        ]
                    },
                }
            ]
        },
        execution_results={
            "steps": [
                {
                    "step_id": "analysis",
                    "status": "completed",
                    "result": {"energy": -1.2, "effective_settings": {
                        "analysis_method": {"name": "mean", "unknown": "SETTINGS_SENTINEL"}
                    }},
                }
            ]
        },
        program_validation={
            "registered_check_results": [
                {
                    "check_instance_id": "analysis:check",
                    "evaluation": "matched",
                    "actual": {
                        "status": {"value": "passed", "unknown": "CHECK_SENTINEL"},
                        "validation_errors": [],
                    },
                }
            ]
        },
    )
    step = view["workflow_evidence"]["steps"][0]
    assert step["expected_observables"] == [{"name": "energy", "unit": None}]
    assert "CHECK_SENTINEL" not in json.dumps(view, ensure_ascii=False)
    assert "SETTINGS_SENTINEL" not in json.dumps(view, ensure_ascii=False)


def test_expert_failure_and_derived_text_hide_relative_paths_ids_hashes_and_nonfinite() -> None:
    """失败与派生科学文字清理相对工作区路径、身份串和非有限数。"""

    bare_uuid = "01234567-89ab-cdef-0123-456789abcdef"
    artifact_hash = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    view, _ = build_workflow_scientific_evidence_view(
        scientific_question="问题",
        hypothesis={"strategy_name": "策略", "reasoning": "正文"},
        workflow={
            "steps": [
                {
                    "step_id": "failed",
                    "step_name": "失败",
                    "description": f"action_id=run_gaussian {bare_uuid} {artifact_hash}",
                    "action_id": "optimize_geometry",
                }
            ]
        },
        execution_results={
            "steps": [
                {
                    "step_id": "failed",
                    "status": "failed",
                    "error": {
                        "message": f"失败于 inputs/request.json {bare_uuid} artifact_hash={artifact_hash}",
                        "failure_stage": "execution",
                    },
                }
            ]
        },
    )
    serialized = json.dumps(view, ensure_ascii=False, allow_nan=False)
    for marker in ("inputs/request.json", bare_uuid, artifact_hash, "action_id=run_gaussian"):
        assert marker not in serialized

    nonfinite_view, _ = build_workflow_scientific_evidence_view(
        scientific_question="问题",
        hypothesis={"strategy_name": "策略", "reasoning": "正文"},
        workflow={
            "steps": [
                {
                    "step_id": "energy",
                    "step_name": "能量",
                    "description": "能量",
                    "execution_layer": "programmable_analysis",
                    "request": {"outputs": [{"name": "energy", "semantic_type": "scalar"}]},
                }
            ]
        },
        execution_results={
            "steps": [
                {
                    "step_id": "energy",
                    "status": "completed",
                    "result": {"energy": float("inf")},
                }
            ]
        },
    )
    assert nonfinite_view["workflow_evidence"]["steps"][0]["observations"] == []
    json.dumps(nonfinite_view, ensure_ascii=False, allow_nan=False)
