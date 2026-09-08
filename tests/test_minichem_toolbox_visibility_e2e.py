"""验证 MiniChem 完整目录在摘要、合同和执行收据之间保持可见且同 hash。"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from minichem_toolbox.catalog import catalog_snapshot, public_catalog_snapshot
from minichem_toolbox.discovery import inspect_action
from minichem_toolbox import discovery as discovery_module

from chemistry_multiagent.tools.minichem_adapter import (
    MiniChemMcpAdapter,
    MiniChemMcpProtocolError,
)
from chemistry_multiagent.utils.workflow_contract import (
    project_minichem_request_contracts,
    project_minichem_stage_tool_projection,
)
from chemistry_multiagent.utils.model_prompts import build_model_messages
from tests.fakes.minichem_mcp import FakeMcpSession, build_catalog_snapshot


def _real_catalog_with_contracts() -> dict:
    """从固定 MiniChem 规格生成不联网的 65 Action/84 pair fixture。"""

    snapshot = catalog_snapshot(include_health=False)
    contracts = []
    for action in snapshot["actions"]:
        for backend_id in action["backend_ids"]:
            inspected = inspect_action(
                action["id"],
                backend_id=backend_id,
                detail_level="contract",
                snapshot=snapshot,
            )
            request_contract = inspected["selected_request_contract"]
            contracts.append(
                {
                    "action_id": action["id"],
                    "backend_id": backend_id,
                    "request_contract": deepcopy(request_contract),
                    "output_contract": deepcopy(
                        request_contract.get("output_contract")
                    ),
                }
            )
    result = deepcopy(snapshot)
    result["action_contracts"] = {
        "schema_version": 1,
        "catalog_hash": snapshot["catalog_hash"],
        "contracts": contracts,
    }
    return result


def _current_catalog_view(catalog: dict) -> dict:
    """生成当前 V2 的目录摘要与按需 pair 合同视图。"""

    raw_actions = {
        action["id"]: action
        for action in catalog["actions"]
        if isinstance(action, dict) and isinstance(action.get("id"), str)
    }
    stage_actions = project_minichem_stage_tool_projection(catalog)
    # V2 模型输入要求一套稳定的完整 Action 摘要。stage 投影提供执行合同所需
    # 的公共字段；下面只补回 Catalog 的分类元数据，不把 pair 合同嵌回摘要。
    actions = []
    for stage_action in stage_actions:
        raw = raw_actions[stage_action["action_id"]]
        action = {
            key: value
            for key, value in stage_action.items()
            if key != "backend_request_contracts"
        }
        action.update(
            {
                # 模型摘要使用短描述；详细输入合同由按需 pair 投影提供。
                "input_description": raw.get("description", "")[:1000],
                "category": raw.get("category"),
                "domain": raw.get("domain", raw.get("category")),
                "execution_layer": raw.get(
                    "execution_layer", "predefined_actions"
                ),
                "input_semantic_types": list(
                    raw.get("input_semantic_types", raw.get("required_inputs", []))
                ),
                "execution_class": raw.get("execution_class"),
                "requires_network": raw.get("requires_network"),
                "capability_tags": list(raw.get("capability_tags", [])),
            }
        )
        actions.append(action)
    view = {
        "catalog_hash": str(catalog["catalog_hash"]),
        "catalog_version": str(catalog["schema_version"]),
        "actions": actions,
        "backends": deepcopy(catalog["backends"]),
    }
    view["action_backend_contracts"] = project_minichem_request_contracts(catalog)
    return view


def test_real_65_action_catalog_is_visible_as_summary_plus_on_demand_contracts():
    """V2 目录视图保留全部摘要，pair 合同通过独立按需投影获取。"""

    catalog = _real_catalog_with_contracts()
    view = _current_catalog_view(catalog)
    assert len(view["actions"]) == 65
    assert len(view["action_backend_contracts"]) == sum(
        len(action["backend_ids"]) for action in catalog["actions"]
    )
    assert view["actions"][0]["action_id"] == catalog["actions"][0]["id"]
    assert view["actions"][-1]["action_id"] == catalog["actions"][-1]["id"]
    assert len({item["domain"] for item in view["actions"]}) >= 5
    assert len({item["backend_id"] for item in view["action_backend_contracts"]}) >= 5
    assert all("backend_request_contracts" not in item for item in view["actions"])

    summary_view = {
        key: view[key] for key in ("catalog_hash", "catalog_version", "actions", "backends")
    }
    messages = build_model_messages(
        "workflow_generation",
        {
            "scientific_context": {
                "scientific_question": "比较一个给定结构的计算性质。",
                "chemistry_context": {},
            },
            "hypothesis": {
                "strategy_name": "直接计算",
                "core_claim": "目录中的直接观测可以回答问题。",
                "reasoning": "需要按合同选择 Action。",
            },
            "task_tool_subgraph": summary_view,
            "workflow_generation_output_schema": {"type": "object"},
        },
    )
    user = next(item["content"] for item in messages if item["role"] == "user")
    projected = json.loads(user)["base_input"]["task_tool_subgraph"]
    assert projected["catalog_hash"] == catalog["catalog_hash"]
    assert projected["catalog_version"] == "7"
    assert len(projected["actions"]) == 65
    assert all("backend_request_contracts" not in item for item in projected["actions"])
    assert set(projected["actions"][0]) >= {
        "action_id",
        "domain",
        "execution_layer",
        "input_semantic_types",
        "output_semantic_types",
        "backend_ids",
    }
    assert "request_contracts" not in projected["actions"][0]
    assert len(view["action_backend_contracts"]) == 84
    assert all(
        pair["action_id"] in {item["action_id"] for item in projected["actions"]}
        for pair in view["action_backend_contracts"]
    )


def test_public_catalog_projection_keeps_planner_visibility_policy() -> None:
    """Planner 能识别脱敏目录仍是完整、任务无关的公开能力投影。"""

    public = public_catalog_snapshot(catalog_snapshot(include_health=False))
    assert public["catalog_visibility_policy"] == "public_contract_projection"
    view = _current_catalog_view(public)
    messages = build_model_messages(
        "workflow_generation",
        {
            "scientific_context": {
                "scientific_question": "比较一个给定结构的计算性质。",
                "chemistry_context": {},
            },
            "hypothesis": {
                "strategy_name": "直接计算",
                "core_claim": "目录中的直接观测可以回答问题。",
                "reasoning": "需要按合同选择 Action。",
            },
            "task_tool_subgraph": {
                key: view[key]
                for key in ("catalog_hash", "catalog_version", "actions", "backends")
            },
            "workflow_generation_output_schema": {"type": "object"},
        },
    )
    user = next(item["content"] for item in messages if item["role"] == "user")
    projected = json.loads(user)["base_input"]["task_tool_subgraph"]
    assert projected["catalog_hash"] == public["catalog_hash"]


def test_server_discovery_uses_one_active_catalog_for_every_entrypoint(monkeypatch):
    """服务端各 discovery 入口必须返回同一个 active 冻结目录 hash。"""

    active = catalog_snapshot(include_health=False)
    active["catalog_hash"] = "active-frozen-catalog-for-test"
    active["catalog_version"] = "spoofed-version-must-not-be-used"
    monkeypatch.setattr(
        discovery_module,
        "active_catalog_snapshot",
        lambda: deepcopy(active),
    )
    first_action = active["actions"][0]
    category = first_action["category"]
    backend_id = first_action["backend_ids"][0]

    domains = discovery_module.list_action_domains()
    browsed = discovery_module.browse_action_category(
        category=category,
        detail_level="summary",
    )
    searched = discovery_module.search_actions(
        query="energy",
        retrieval_mode="lexical",
        detail_level="summary",
        limit=100,
    )
    inspected = discovery_module.inspect_action(
        first_action["id"],
        backend_id=backend_id,
        detail_level="summary",
    )
    inspected_backend = discovery_module.inspect_backend(backend_id)
    resources = discovery_module.search_resources(limit=100)
    inspected_resource = (
        discovery_module.inspect_resource(resources["resources"][0]["resource_id"])
        if resources["resources"]
        else None
    )

    assert {
        domains["catalog_hash"],
        browsed["catalog_hash"],
        searched["catalog_hash"],
        inspected["catalog_hash"],
        inspected_backend["catalog_hash"],
        resources["catalog_hash"],
        *(
            [inspected_resource["catalog_hash"]]
            if inspected_resource is not None
            else []
        ),
    } == {"active-frozen-catalog-for-test"}
    assert {
        domains["catalog_version"],
        browsed["catalog_version"],
        searched["catalog_version"],
        inspected["catalog_version"],
        inspected_backend["catalog_version"],
        resources["catalog_version"],
        *(
            [inspected_resource["catalog_version"]]
            if inspected_resource is not None
            else []
        ),
    } == {str(active["schema_version"])}


@pytest.mark.parametrize(
    ("identity_field", "identity_value"),
    [
        ("catalog_hash", None),
        ("catalog_hash", "0" * 64),
        ("catalog_version", None),
        ("catalog_version", "8"),
    ],
)
def test_adapter_discovery_rejects_missing_or_drifted_catalog_identity(
    tmp_path: Path,
    identity_field: str,
    identity_value: str | None,
):
    """冻结目录后 discovery 身份缺失或漂移必须立即拒绝。"""

    catalog = build_catalog_snapshot()
    hash_value = catalog["catalog_hash"]
    response = {
        "status": "success",
        "catalog_hash": hash_value,
        "catalog_version": str(catalog["schema_version"]),
        "domains": [
            {
                "category": "structure_and_system",
                "action_ids": ["convert_structure"],
            }
        ],
    }
    if identity_value is None:
        response.pop(identity_field)
    else:
        response[identity_field] = identity_value
    session = FakeMcpSession(
        catalog=catalog,
        tool_results={"list_action_domains": response},
    )
    adapter = MiniChemMcpAdapter(
        session_factory=lambda: session,
        operation_root=tmp_path / "operations",
    )
    adapter.open()
    try:
        adapter.discover_catalog()
        with pytest.raises(MiniChemMcpProtocolError, match=identity_field):
            adapter.list_action_domains()
    finally:
        adapter.close()


def test_adapter_discovery_and_execution_keep_frozen_catalog_identity(tmp_path: Path):
    """search/inspect 只查冻结目录，执行 request/manifest 继续保存同 hash/version。"""

    catalog = build_catalog_snapshot()
    hash_value = catalog["catalog_hash"]
    search_pages = {
        "status": "success",
        "catalog_hash": hash_value,
        "catalog_version": str(catalog["schema_version"]),
        "actions": [{"id": "convert_structure"}],
        "next_offset": None,
        "total_matches": 1,
    }
    session = FakeMcpSession(
        catalog=catalog,
        tool_results={
            "search_actions": lambda request: search_pages,
            "list_action_domains": {
                "status": "success",
                "catalog_hash": hash_value,
                "catalog_version": str(catalog["schema_version"]),
                "domains": [{"category": "structure_and_system", "action_ids": ["convert_structure"]}],
            },
            "inspect_action": {
                "status": "success",
                "catalog_hash": hash_value,
                "catalog_version": str(catalog["schema_version"]),
                "action": {"id": "convert_structure"},
                "selected_request_contract": {
                    "required_contract": {},
                    "optional_field_names": [],
                    "execute_action_request_template": {},
                    "enumerated_options": [],
                    "conditional_requirements": [],
                    "output_contract": {},
                },
            },
        },
    )
    adapter = MiniChemMcpAdapter(
        session_factory=lambda: session,
        operation_root=tmp_path / "operations",
    )
    adapter.open()
    try:
        adapter.discover_catalog()
        domains = adapter.list_action_domains()
        searched = adapter.search_all_actions("structure")
        inspected = adapter.inspect_action_contract("convert_structure", "openbabel")
        assert domains["catalog_hash"] == hash_value
        assert searched["actions"] == [{"id": "convert_structure"}]
        assert inspected["catalog_hash"] == hash_value
        search_call = next(call for call in session.calls if call.name == "search_actions")
        assert search_call.arguments["request"]["limit"] == 100

        result = adapter.execute_action(
            {
                "action_id": "convert_structure",
                "backend_id": "openbabel",
                "inputs": {"structure": {"smiles": "O=Cc1ccccc1"}},
                "method_spec": {},
                "action_settings": {},
                "resource_limits": {},
            },
            operation_id="visibility-operation",
        )
        assert result["status"] == "completed"
        assert result["operation_id"] == "visibility-operation"
        assert result["provenance"]["catalog_hash"] == hash_value
        assert result["output_artifacts"]
    finally:
        adapter.close()
