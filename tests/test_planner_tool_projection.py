"""V2 MiniChem 目录能力和请求合同投影回归测试。"""

from __future__ import annotations

import json
from copy import deepcopy

from chemistry_multiagent.utils.gaussian_routing import (
    GaussianRoutePolicy,
    STRUCTURED_EXTENSIONS_POLICY_ID,
)
from chemistry_multiagent.utils.model_prompts import build_model_messages
from chemistry_multiagent.utils.workflow_contract import (
    project_minichem_request_contracts,
    project_minichem_stage_tool_projection,
)
from tests.test_minichem_toolbox_visibility_e2e import _real_catalog_with_contracts


def _declared_pairs(catalog: dict) -> set[tuple[str, str]]:
    """返回冻结目录声明的全部 Action/Backend pair。"""

    return {
        (action["id"], backend_id)
        for action in catalog["actions"]
        for backend_id in action["backend_ids"]
    }


def test_stage_projection_covers_every_declared_pair() -> None:
    """当前 V2 投影完整保留目录能力，不依赖旧 Planner 类。"""

    catalog = _real_catalog_with_contracts()
    projection = project_minichem_stage_tool_projection(catalog)
    projected_pairs = {
        (item["action_id"], backend_id)
        for item in projection
        for backend_id in item["backend_ids"]
    }
    assert projected_pairs == _declared_pairs(catalog)
    assert len(projection) == len(catalog["actions"])
    assert all("planner_constraints" not in item for item in projection)


def test_pair_projection_keeps_artifact_and_input_handoff_contracts() -> None:
    """当前 pair 投影保留 ArtifactRef 和输入交接语义。"""

    catalog = _real_catalog_with_contracts()
    pairs = {
        (item["action_id"], item["backend_id"]): item
        for item in project_minichem_request_contracts(catalog)
    }
    converter = pairs[("xyz_to_gjf", "format_conversion")]
    output = converter["output_contract"]
    assert output["artifact_handoff"]
    assert output["primary_output"] == "GaussianInputDeck"
    input_fields = converter["request_contract"]["required_contract"]["inputs"]
    assert any(field["name"] == "input_xyz_path" for field in input_fields)


def test_pair_projection_redacts_local_paths_and_copies_nested_examples() -> None:
    """目录合同投影不泄露本机路径，且不会共享可变目录对象。"""

    catalog = _real_catalog_with_contracts()
    pairs = project_minichem_request_contracts(catalog)
    serialized = json.dumps(pairs, ensure_ascii=False)
    assert all(token not in serialized for token in ("/tmp/", "/mnt/", "file://"))

    source_contract = next(
        item["request_contract"]
        for item in catalog["action_contracts"]["contracts"]
        if (item["action_id"], item["backend_id"])
        == ("analyze_free_energy_selectivity", "internal_thermochemistry")
    )
    source_example = deepcopy(
        source_contract["required_contract"]["inputs"][0]["canonical_inline_example"]
    )
    projected = next(
        item
        for item in pairs
        if (item["action_id"], item["backend_id"])
        == ("analyze_free_energy_selectivity", "internal_thermochemistry")
    )
    fields = projected["request_contract"]["required_contract"]["inputs"]
    projected_field = next(item for item in fields if "canonical_inline_example" in item)
    assert projected_field["canonical_inline_example"] == source_example
    assert projected_field["canonical_inline_example"] is not source_contract[
        "required_contract"
    ]["inputs"][0]["canonical_inline_example"]


def test_gaussian_route_policy_does_not_remove_catalog_pairs() -> None:
    """任务级 Gaussian 路线策略不会把真实目录能力伪装成不存在。"""

    catalog = _real_catalog_with_contracts()
    policy = GaussianRoutePolicy(
        policy_id=STRUCTURED_EXTENSIONS_POLICY_ID,
        question_fingerprint="0" * 64,
    )
    without_policy = project_minichem_stage_tool_projection(catalog)
    with_policy = project_minichem_stage_tool_projection(
        catalog, gaussian_route_policy=policy
    )
    assert {(x["action_id"], tuple(x["backend_ids"])) for x in with_policy} == {
        (x["action_id"], tuple(x["backend_ids"])) for x in without_policy
    }


def test_model_messages_accept_current_stage_projection() -> None:
    """工作流生成模型读取当前目录投影，不读取旧 Planner 私有结构。"""

    catalog = _real_catalog_with_contracts()
    subgraph = {
        "catalog_hash": catalog["catalog_hash"],
        "catalog_version": str(catalog["schema_version"]),
        "actions": deepcopy(project_minichem_stage_tool_projection(catalog)),
        "backends": deepcopy(catalog["backends"]),
        "native_software": [],
        "analysis_runtimes": [],
        "managed_program_contract": None,
    }
    messages = build_model_messages(
        "workflow_generation",
        {
            "scientific_context": {"scientific_question": "Q"},
            "hypothesis": {"strategy_name": "S", "reasoning": "R"},
            "task_tool_subgraph": subgraph,
            "workflow_generation_output_schema": {"type": "object"},
        },
    )
    user = next(item["content"] for item in messages if item["role"] == "user")
    assert "task_tool_subgraph" in user
    envelope = json.loads(user)
    assert envelope["base_input"]["task_tool_subgraph"]["catalog_hash"] == catalog[
        "catalog_hash"
    ]
