"""Gaussian 目录驱动工作流与通用连接合同回归。"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from chemistry_multiagent.utils.workflow_contract import (
    WorkflowContractError,
    canonicalize_generation_output,
    validate_canonical_workflow,
)
from tests.fakes.minichem_mcp import build_catalog_snapshot


def _gaussian_catalog() -> dict[str, Any]:
    """构造带 codegen、converter、native Gaussian 和 TS composite 的最小目录。"""

    catalog = build_catalog_snapshot(
        actions=[
            {
                "id": "generate_gaussian_code",
                "version": "1.0.0",
                "category": "molecular_electronic",
                "description": "生成 Gaussian route。",
                "backend_ids": ["gaussian_codegen"],
                "required_inputs": ["user_input"],
                "optional_inputs": [],
                "selection_policy": "agent_backend_required",
                "primary_output": "GaussianCodeResult",
                "output_semantic_types": ["GaussianCodeResult"],
                "data_action": False,
                "batch_safe": False,
            },
            {
                "id": "xyz_to_gjf",
                "version": "1.0.0",
                "category": "scientific_data_interchange",
                "description": "把 XYZ 与 route 转成 Gaussian deck。",
                "backend_ids": ["format_conversion"],
                "required_inputs": ["input_xyz_path"],
                "optional_inputs": ["route_section"],
                "selection_policy": "internal_deterministic",
                "primary_output": "GaussianInputDeck",
                "output_semantic_types": ["GaussianInputDeck"],
                "data_action": False,
                "batch_safe": False,
            },
            {
                "id": "calculate_bond_orders",
                "version": "1.0.0",
                "category": "molecular_electronic",
                "description": "用所选后端计算结构的键级。",
                "backend_ids": ["xtb", "multiwfn", "gaussian"],
                "required_inputs": ["structure"],
                "optional_inputs": [],
                "selection_policy": "agent_backend_required",
                "primary_output": "BondOrderResult",
                "output_semantic_types": ["BondOrderResult"],
                "data_action": False,
                "batch_safe": False,
            },
            {
                "id": "gaussian_calculation",
                "version": "1.0.0",
                "category": "electronic_structure",
                "description": "错误示例：把 Gaussian 计算包装成 predefined Action。",
                "backend_ids": ["gaussian_calculation"],
                "required_inputs": ["input"],
                "optional_inputs": [],
                "selection_policy": "agent_backend_required",
                "primary_output": "GaussianResult",
                "output_semantic_types": ["GaussianResult"],
                "data_action": False,
                "batch_safe": False,
            },
            {
                "id": "gaussian_pipeline",
                "version": "1.0.0",
                "category": "electronic_structure",
                "description": "错误示例：普通 Gaussian pipeline 仍是计算 Action。",
                "backend_ids": ["gaussian_pipeline"],
                "required_inputs": ["input"],
                "optional_inputs": [],
                "selection_policy": "agent_backend_required",
                "primary_output": "GaussianResult",
                "output_semantic_types": ["GaussianResult"],
                "data_action": False,
                "batch_safe": False,
            },
            {
                "id": "ts_pipeline",
                "version": "1.0.0",
                "category": "transition_state",
                "description": "合法示例：TS composite 负责组合路径，不等于 native Gaussian 作业。",
                "backend_ids": ["ts_pipeline"],
                "required_inputs": ["input"],
                "optional_inputs": [],
                "selection_policy": "agent_backend_required",
                "primary_output": "GaussianTSComposite",
                "output_semantic_types": ["GaussianTSComposite"],
                "data_action": False,
                "batch_safe": False,
            },
        ],
        backends=[
            {
                "id": "gaussian_codegen",
                "display_name": "Gaussian route generator",
                "runtime": "native",
                "capabilities": ["generate_gaussian_code"],
                "health": {"status": "healthy"},
            },
            {
                "id": "format_conversion",
                "display_name": "Format conversion",
                "runtime": "native",
                "capabilities": ["xyz_to_gjf"],
                "health": {"status": "healthy"},
            },
            {
                "id": "xtb",
                "display_name": "xTB",
                "runtime": "native",
                "capabilities": ["calculate_bond_orders"],
                "health": {"status": "healthy"},
            },
            {
                "id": "multiwfn",
                "display_name": "Multiwfn",
                "runtime": "native",
                "capabilities": ["calculate_bond_orders"],
                "health": {"status": "healthy"},
            },
            {
                "id": "gaussian",
                "display_name": "Gaussian 键级后端",
                "runtime": "native",
                "capabilities": ["calculate_bond_orders"],
                "health": {"status": "healthy"},
            },
            {
                "id": "gaussian_calculation",
                "display_name": "错误 Gaussian Action backend",
                "runtime": "native",
                "capabilities": ["gaussian_calculation"],
                "health": {"status": "healthy"},
            },
            {
                "id": "gaussian_pipeline",
                "display_name": "错误 Gaussian pipeline backend",
                "runtime": "native",
                "capabilities": ["gaussian_pipeline"],
                "health": {"status": "healthy"},
            },
            {
                "id": "ts_pipeline",
                "display_name": "TS composite backend",
                "runtime": "native",
                "capabilities": ["ts_pipeline"],
                "health": {"status": "healthy"},
            },
        ],
    )
    catalog["open_execution"] = {
        "schema_version": 1,
        "native_software": [
            {
                "software_id": "gaussian",
                "display_name": "Gaussian",
                "commands": [
                    {
                        "executable": "g16",
                        "synopsis": "运行 Gaussian 16",
                        "input_mode": "stdin",
                        "required_files": [],
                        "example_request": {},
                    }
                ],
            }
        ],
        "analysis_runtimes": [],
        "authoring_contract": {
            "path_prefix": "code/",
            "max_files": 4,
            "max_chars_per_file": 100000,
        },
    }
    return catalog


def _resource_limits() -> dict[str, int]:
    """返回固定的最小资源请求。"""

    return {"cpu_cores": 1, "memory_mb": 128, "gpu_count": 0}


def _action_request(inputs: dict[str, Any], bindings: list[dict[str, str]]) -> dict[str, Any]:
    """构造 predefined Action 的最小请求。"""

    return {
        "inputs": inputs,
        "method_spec": {},
        "action_settings": {},
        "resource_limits": _resource_limits(),
        "bindings": bindings,
    }


def _generation_wire() -> dict[str, Any]:
    """构造 codegen→XYZ converter→native Gaussian 三步短引用工作流。"""

    return {
        "steps": [
            {
                "step_ref": "s1",
                "step_name": "生成 Gaussian route",
                "description": "生成固定计算所需的 route。",
                "execution_layer": "predefined_actions",
                "action_id": "generate_gaussian_code",
                "backend_id": "gaussian_codegen",
                "request": _action_request(
                    {"user_input": "优化结构并计算频率"},
                    [],
                ),
                "depends_on": [],
            },
            {
                "step_ref": "s2",
                "step_name": "转换 Gaussian 输入",
                "description": "把结构和 route 转成 Gaussian deck。",
                "execution_layer": "predefined_actions",
                "action_id": "xyz_to_gjf",
                "backend_id": "format_conversion",
                "request": _action_request(
                    {"input_xyz_path": "input.xyz", "route_section": None},
                    [
                        {
                            "source_step_ref": "s1",
                            "source_field": "/result/gaussian_code",
                            "target_field": "inputs.route_section",
                        }
                    ],
                ),
                "depends_on": ["s1"],
            },
            {
                "step_ref": "s3",
                "step_name": "执行 Gaussian native g16",
                "description": "提交绑定 deck 的 Gaussian 作业。",
                "execution_layer": "native_software",
                "request": {
                    "software_id": "gaussian",
                    "executable": "g16",
                    "arguments": ["input.com"],
                    "staged_inputs": [
                        {"source_path": None, "target_path": "input.com"}
                    ],
                    "stdin_target": None,
                    "resource_limits": _resource_limits(),
                    "calculation_intent": "optimization_frequency",
                    "label": "gaussian-contract-test",
                    "parent_job_id": None,
                    "bindings": [
                        {
                            "source_step_ref": "s2",
                            "source_field": "artifact.semantic_type=GaussianInputDeck",
                            "target_field": "staged_inputs.0.source_path",
                        }
                    ],
                },
                "depends_on": ["s2"],
            },
        ]
    }


def _canonical_gaussian_workflow() -> tuple[dict[str, Any], dict[str, Any]]:
    """规范化合法三步链并返回工作流和冻结目录。"""

    catalog = _gaussian_catalog()
    counter = 0

    def allocate(_kind: str) -> str:
        nonlocal counter
        counter += 1
        return f"step-formal-{counter}"

    workflow = canonicalize_generation_output(
        _generation_wire(),
        catalog,
        allocate,
        hypothesis={
            "strategy_name": "Gaussian 固定链",
            "reasoning": "通过 route、deck 和 native 作业的可追踪结果验证计算。",
        },
    )
    return workflow, catalog


def _single_action_workflow(action_id: str, catalog: dict[str, Any]) -> dict[str, Any]:
    """构造一个用于验证 predefined Gaussian Action 边界的工作流。"""

    action = next(item for item in catalog["actions"] if item["id"] == action_id)
    backend_id = action["backend_ids"][0]
    return {
        "workflow_name": "单步 Action 测试",
        "goal": "验证 Action 执行边界",
        "strategy_name": "Action 边界",
        "steps": [
            {
                "step_id": "step-action-1",
                "step_name": action_id,
                "description": "验证 predefined Action 合同。",
                "action_id": action_id,
                "backend_id": backend_id,
                "request": _action_request({"input": "input.xyz"}, []),
                "expected_outputs": {
                    "action_id": action_id,
                    "action_version": action["version"],
                    "primary_output": action["primary_output"],
                    "output_semantic_types": action["output_semantic_types"],
                },
                "depends_on": [],
            }
        ],
    }


def test_catalog_declared_gaussian_workflow_is_valid() -> None:
    """目录声明的 pair 可以通过校验，工作流不要求固定步骤数量。"""

    workflow, catalog = _canonical_gaussian_workflow()

    validate_canonical_workflow(workflow, catalog)
    direct_typed = _single_action_workflow("gaussian_calculation", catalog)
    validate_canonical_workflow(direct_typed, catalog)
    assert all("action_id" in step or step["execution_layer"] == "native_software"
               for step in workflow["steps"])


def test_native_foreach_keeps_generic_bounds_and_identity_checks() -> None:
    """native foreach 只核对通用上限和软件身份，不核对上游 Action。"""

    workflow, catalog = _canonical_gaussian_workflow()
    native = workflow["steps"][2]
    request = native["request"]
    request["bindings"] = []
    request["foreach"] = {
        "source_step_ref": workflow["steps"][1]["step_id"],
        "source_field": "/result/items",
        "item_field": "/result/result/output_gjf_path",
        "item_id_field": "/item_id",
        "target_field": "staged_inputs.0.source_path",
        "max_items": 33,
    }
    with pytest.raises(WorkflowContractError, match="1 到 32"):
        validate_canonical_workflow(workflow, catalog)

    request["foreach"]["max_items"] = 32
    request["software_id"] = "xtb"
    request["executable"] = "xtb"
    catalog["open_execution"]["native_software"].append(
        {
            "software_id": "xtb",
            "display_name": "xTB",
            "commands": [
                {
                    "executable": "xtb",
                    "synopsis": "运行 xTB",
                    "input_mode": "arguments",
                    "required_files": [],
                    "example_request": {},
                }
            ],
        }
    )
    with pytest.raises(WorkflowContractError, match="只允许 native Gaussian"):
        validate_canonical_workflow(workflow, catalog)


def test_native_gaussian_accepts_staged_input_without_converter_binding() -> None:
    """native Gaussian 只需满足通用 staged input 合同，不强制 converter 来源。"""

    workflow, catalog = _canonical_gaussian_workflow()
    native = workflow["steps"][2]
    native["request"]["bindings"] = []
    native["depends_on"] = []
    validate_canonical_workflow(workflow, catalog)


@pytest.mark.parametrize("action_id", ["gaussian_calculation", "gaussian_pipeline"])
def test_catalog_declared_typed_gaussian_action_is_accepted(
    action_id: str,
) -> None:
    """目录声明的 typed Gaussian pair 对所有题面都可规划和校验。"""

    catalog = _gaussian_catalog()
    workflow = _single_action_workflow(action_id, catalog)

    validate_canonical_workflow(workflow, catalog)


def test_ts_composite_pipeline_is_not_treated_as_native_gaussian_job() -> None:
    """TS composite 只表示组合流程，应保留而不触发 native Gaussian 链校验。"""

    catalog = _gaussian_catalog()
    workflow = _single_action_workflow("ts_pipeline", catalog)

    validate_canonical_workflow(workflow, catalog)
