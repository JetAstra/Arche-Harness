"""V2 MiniChem 目录投影与 Gaussian 连接合同回归。"""

from __future__ import annotations

from copy import deepcopy

from chemistry_multiagent.utils.model_prompts import build_model_messages
from chemistry_multiagent.utils.workflow_contract import (
    project_minichem_request_contracts,
    project_minichem_stage_tool_projection,
    resolve_minichem_step_request,
    validate_canonical_workflow,
)
from tests.test_gaussian_workflow_contract import _gaussian_catalog
from tests.test_minichem_toolbox_visibility_e2e import _real_catalog_with_contracts


def test_v2_stage_projection_exposes_every_gaussian_action_backend_pair() -> None:
    """V2 阶段投影保留冻结目录的 Action/Backend 能力，不依赖旧 Planner。"""

    catalog = _gaussian_catalog()
    projection = project_minichem_stage_tool_projection(catalog)
    action_pairs = {
        (item["action_id"], backend_id)
        for item in projection
        for backend_id in item["backend_ids"]
    }

    assert {
        ("generate_gaussian_code", "gaussian_codegen"),
        ("xyz_to_gjf", "format_conversion"),
        ("calculate_bond_orders", "xtb"),
        ("calculate_bond_orders", "multiwfn"),
        ("calculate_bond_orders", "gaussian"),
        ("ts_pipeline", "ts_pipeline"),
    } <= action_pairs
    assert all("planner_constraints" not in item for item in projection)


def test_v2_pair_projection_preserves_contracts_without_private_paths() -> None:
    """V2 pair 合同保留输入/输出交接语义，同时不泄露本机路径。"""

    catalog = _real_catalog_with_contracts()
    projected = project_minichem_request_contracts(catalog)
    by_pair = {(item["action_id"], item["backend_id"]): item for item in projected}

    converter = by_pair[("xyz_to_gjf", "format_conversion")]
    request_contract = converter["request_contract"]
    output_contract = converter["output_contract"]
    assert request_contract["required_contract"]["inputs"]
    assert output_contract["primary_output"] == "GaussianInputDeck"
    assert output_contract["artifact_handoff"]
    assert "generate_gaussian_code" not in str(output_contract)
    assert all(
        token not in str(item)
        for item in projected
        for token in ("/tmp/", "/mnt/", "file://")
    )


def test_gaussian_route_pointer_resolves_to_scalar_route_text() -> None:
    """公开 result 指针应解析为下游转换器需要的 route 字符串。"""

    request = resolve_minichem_step_request(
        {
            "request": {
                "inputs": {"route_section": None},
                "bindings": [
                    {
                        "source_step_ref": "s1",
                        "source_field": "/result/gaussian_code",
                        "target_field": "inputs.route_section",
                    }
                ],
            }
        },
        {
            "s1": {
                "raw_output": {
                    "result": {"gaussian_code": "#p M062X/def2SVP opt"},
                    "output_artifacts": [],
                }
            }
        },
    )
    assert request["inputs"]["route_section"] == "#p M062X/def2SVP opt"


def test_v2_gaussian_workflow_contract_accepts_typed_action_chain() -> None:
    """目录声明的 Gaussian codegen→converter→计算链可通过 V2 合同校验。"""

    catalog = _gaussian_catalog()
    workflow = {
        "workflow_name": "Gaussian V2 合同链",
        "goal": "验证冻结目录中的 Gaussian Action 连接。",
        "strategy_name": "typed Gaussian chain",
        "steps": [
            {
                "step_id": "code",
                "step_name": "生成 route",
                "description": "生成公开 route 结果。",
                "action_id": "generate_gaussian_code",
                "backend_id": "gaussian_codegen",
                "request": {
                    "inputs": {"user_input": "opt"},
                    "method_spec": {},
                    "action_settings": {},
                    "resource_limits": {
                        "cpu_cores": 1,
                        "memory_mb": 128,
                        "gpu_count": 0,
                    },
                    "bindings": [],
                },
                "expected_outputs": {
                    "action_id": "generate_gaussian_code",
                    "action_version": "1.0.0",
                    "primary_output": "GaussianCodeResult",
                    "output_semantic_types": ["GaussianCodeResult"],
                },
                "depends_on": [],
            }
        ],
    }
    validate_canonical_workflow(workflow, catalog)


def test_v2_model_messages_use_current_catalog_projection() -> None:
    """模型消息只接收当前 V2 目录投影，不再依赖 Planner 私有视图。"""

    catalog = _real_catalog_with_contracts()
    stage_projection = project_minichem_stage_tool_projection(catalog)
    task_subgraph = {
        "catalog_hash": catalog["catalog_hash"],
        "catalog_version": str(catalog["schema_version"]),
        "actions": deepcopy(stage_projection),
        "backends": deepcopy(catalog["backends"]),
        "native_software": [],
        "analysis_runtimes": [],
        "managed_program_contract": None,
    }
    messages = build_model_messages(
        "workflow_generation",
        {
            "scientific_context": {"scientific_question": "Q"},
            "hypothesis": {"strategy_name": "H", "reasoning": "R"},
            "task_tool_subgraph": task_subgraph,
            "workflow_generation_output_schema": {"type": "object"},
        },
    )
    user = next(item["content"] for item in messages if item["role"] == "user")
    assert "task_tool_subgraph" in user
    assert "typed Gaussian 计算 Action 不可规划" not in user
