"""ARCHE 侧 MiniChem 批量交接诊断测试。

本文件锁定 ARCHE 的 `foreach` 数据边、逐项 Artifact 交接和 native 边界，
不启动真实 MiniChem、Gaussian 或外部软件。
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from chemistry_multiagent.tools.minichem_adapter import MiniChemMcpAdapter
from chemistry_multiagent.utils.workflow_contract import (
    WorkflowContractError,
    resolve_minichem_step_request,
    validate_canonical_workflow,
)
from tests.fakes.minichem_mcp import (
    FakeMcpSession,
    build_artifact_ref,
    build_catalog_snapshot,
)


def _batch_catalog() -> dict[str, Any]:
    """构造批量交接测试使用的最小冻结目录。"""

    catalog = build_catalog_snapshot(
        actions=[
            {
                "id": "smiles2sdf",
                "version": "1.0.0",
                "category": "scientific_data_interchange",
                "description": "把一批 SMILES 转为可交接的 SDF 记录。",
                "backend_ids": ["format_conversion"],
                "required_inputs": ["smiles_list"],
                "optional_inputs": [],
                "selection_policy": "internal_deterministic",
                "primary_output": "SDFFile",
                "output_semantic_types": ["SDFFile"],
                "data_action": False,
                "batch_safe": True,
            },
            {
                "id": "sdf_to_gjf",
                "version": "1.0.0",
                "category": "scientific_data_interchange",
                "description": "把一个单记录 SDF 转为 Gaussian 输入文件。",
                "backend_ids": ["format_conversion"],
                "required_inputs": ["sdf_path"],
                "optional_inputs": ["route_parameters"],
                "selection_policy": "internal_deterministic",
                "primary_output": "GaussianInputDeck",
                "output_semantic_types": ["GaussianInputDeck"],
                "data_action": False,
                "batch_safe": False,
            },
        ],
        backends=[
            {
                "id": "format_conversion",
                "display_name": "Format conversion",
                "runtime": "native",
                "capabilities": ["smiles2sdf", "sdf_to_gjf"],
                "health": {"status": "healthy"},
            }
        ],
    )
    catalog["action_contracts"] = {
        "schema_version": 1,
        "catalog_hash": catalog["catalog_hash"],
        "contracts": [
            {
                "action_id": "smiles2sdf",
                "backend_id": "format_conversion",
                "request_contract": {
                    "sections": {
                        "inputs": {
                            "allowed": ["smiles_list"],
                            "required": ["smiles_list"],
                        },
                        "method_spec": {"allowed": [], "required": []},
                        "action_settings": {"allowed": [], "required": []},
                        "resource_limits": {
                            "allowed": ["cpu_cores", "memory_mb", "gpu_count"],
                            "required": [],
                        },
                    }
                },
                "output_contract": {
                    "result_fields": ["successful_records"],
                    "batch_handoff": _smiles_batch_handoff_contract(),
                },
            },
            {
                "action_id": "sdf_to_gjf",
                "backend_id": "format_conversion",
                "request_contract": {
                    "sections": {
                        "inputs": {
                            "allowed": ["sdf_path", "route_parameters"],
                            "required": [
                                {
                                    "name": "sdf_path",
                                    "type": "SDFFile | SDFRecordFile ArtifactRef",
                                    "accepted_artifact_semantic_types": [
                                        "SDFFile",
                                        "SDFRecordFile",
                                    ],
                                }
                            ],
                        },
                        "method_spec": {"allowed": [], "required": []},
                        "action_settings": {"allowed": [], "required": []},
                        "resource_limits": {
                            "allowed": ["cpu_cores", "memory_mb", "gpu_count"],
                            "required": [],
                        },
                    }
                },
                "output_contract": {
                    "result_fields": ["gaussian_input_deck"]
                },
            },
        ],
    }
    return catalog


def _resource_limits() -> dict[str, int]:
    """返回测试用的最小资源声明。"""

    return {"cpu_cores": 1, "memory_mb": 128, "gpu_count": 0}


def _smiles_batch_handoff_contract() -> dict[str, Any]:
    """返回与 MiniChem discovery 对齐的 smiles2sdf 批量交接声明。"""

    return {
        "source_field": "/result/successful_records",
        "item_field": "/sdf_artifact",
        "item_id_field": "/item_id",
        "target_field": "/inputs/sdf_path",
        "downstream_action_id": "sdf_to_gjf",
        "max_items": 32,
        "only_successful_items": True,
        "description": (
            "只把 successful_records 中的 SDFRecordFile 逐项交给 sdf_to_gjf。"
        ),
    }


def _source_artifacts() -> tuple[dict[str, Any], dict[str, Any]]:
    """构造两个带稳定输入身份的单记录 SDF ArtifactRef。"""

    return (
        build_artifact_ref(
            artifact_id="sdf-item-0001",
            semantic_type="SDFRecordFile",
            media_type="chemical/x-mdl-sdfile",
            path="artifacts/sdf-item-0001.sdf",
            producer_action="smiles2sdf",
            producer_backend="format_conversion",
        ),
        build_artifact_ref(
            artifact_id="sdf-item-0002",
            semantic_type="SDFRecordFile",
            media_type="chemical/x-mdl-sdfile",
            path="artifacts/sdf-item-0002.sdf",
            producer_action="smiles2sdf",
            producer_backend="format_conversion",
        ),
    )


def _source_completed_step() -> dict[str, Any]:
    """构造 smiles2sdf 对外公开的逐项结果数组。"""

    first, second = _source_artifacts()
    return {
        "raw_output": {
            "status": "success",
            "result": {
                "successful_records": [
                    {
                        "item_id": "mol-0001",
                        "sdf_artifact": {"artifact_id": first["artifact_id"]},
                    },
                    {
                        "item_id": "mol-0002",
                        "sdf_artifact": {"artifact_id": second["artifact_id"]},
                    },
                ]
            },
        },
        "output_artifacts": [first, second],
    }


def _sdf_to_gjf_foreach_request(source_step_ref: str) -> dict[str, Any]:
    """构造从 successful_records 到 sdf_to_gjf 的静态 foreach 映射。"""

    return {
        "inputs": {"sdf_path": None},
        "method_spec": {},
        "action_settings": {},
        "resource_limits": _resource_limits(),
        "bindings": [],
        "foreach": {
            "source_step_ref": source_step_ref,
            "source_field": "/result/successful_records",
            "item_field": "/sdf_artifact",
            "item_id_field": "/item_id",
            "target_field": "/inputs/sdf_path",
            "max_items": 32,
        },
    }


def _batch_workflow() -> dict[str, Any]:
    """构造一条 smiles2sdf → sdf_to_gjf 的最小静态工作流。"""

    return {
        "workflow_name": "批量 SDF 交接诊断",
        "goal": "每个成功 SMILES 都生成对应的 Gaussian 输入文件。",
        "strategy_name": "逐项 SDF 转换",
        "steps": [
            {
                "step_id": "smiles-step",
                "step_name": "批量生成 SDF",
                "description": "为每个成功 SMILES 暴露单记录 SDF。",
                "action_id": "smiles2sdf",
                "backend_id": "format_conversion",
                "request": {
                    "inputs": {"smiles_list": ["CCO", "CCN"]},
                    "method_spec": {},
                    "action_settings": {},
                    "resource_limits": _resource_limits(),
                    "bindings": [],
                },
                "expected_outputs": {
                    "action_id": "smiles2sdf",
                    "action_version": "1.0.0",
                    "primary_output": "SDFFile",
                    "output_semantic_types": ["SDFFile"],
                    "output_contract": {
                        "result_fields": ["successful_records"],
                        "batch_handoff": _smiles_batch_handoff_contract(),
                    },
                },
                "depends_on": [],
            },
            {
                "step_id": "gjf-step",
                "step_name": "逐项生成 Gaussian 输入",
                "description": "把每个单记录 SDF 交给 sdf_to_gjf。",
                "action_id": "sdf_to_gjf",
                "backend_id": "format_conversion",
                "request": _sdf_to_gjf_foreach_request("smiles-step"),
                "expected_outputs": {
                    "action_id": "sdf_to_gjf",
                    "action_version": "1.0.0",
                    "primary_output": "GaussianInputDeck",
                    "output_semantic_types": ["GaussianInputDeck"],
                    "output_contract": {
                        "result_fields": ["gaussian_input_deck"]
                    },
                },
                "depends_on": ["smiles-step"],
            },
        ],
    }


def test_public_result_array_can_prepare_sdf_to_gjf_foreach_items() -> None:
    """现有最小接口可从公开数组展开两项并保留 item_id 与 ArtifactRef。"""

    workflow = _batch_workflow()
    resolved = resolve_minichem_step_request(
        workflow["steps"][1],
        {"smiles-step": _source_completed_step()},
    )

    assert [item["item_id"] for item in resolved["_foreach_items"]] == [
        "mol-0001",
        "mol-0002",
    ]
    assert [
        item["request"]["inputs"]["sdf_path"]["artifact_id"]
        for item in resolved["_foreach_items"]
    ] == ["sdf-item-0001", "sdf-item-0002"]
    assert all(
        item["lineage"]["origin_source_step_ref"] == "smiles-step"
        for item in resolved["_foreach_items"]
    )


def test_native_foreach_resolves_one_gaussian_deck_request_per_converter_item() -> None:
    """native Gaussian foreach 应按 converter 子结果逐项解析 deck 和来源身份。"""

    first, second = _source_artifacts()
    first_deck = build_artifact_ref(
        artifact_id="gjf-item-0001",
        semantic_type="GaussianInputDeck",
        media_type="chemical/x-gaussian-input",
        path="artifacts/gjf-item-0001.gjf",
        producer_action="sdf_to_gjf",
        producer_backend="format_conversion",
        parent_artifact_ids=[first["artifact_id"]],
    )
    second_deck = build_artifact_ref(
        artifact_id="gjf-item-0002",
        semantic_type="GaussianInputDeck",
        media_type="chemical/x-gaussian-input",
        path="artifacts/gjf-item-0002.gjf",
        producer_action="sdf_to_gjf",
        producer_backend="format_conversion",
        parent_artifact_ids=[second["artifact_id"]],
    )
    native_step = {
        "step_id": "gaussian-step",
        "request": {
            "software_id": "gaussian",
            "executable": "g16",
            "staged_inputs": [
                {"source_path": None, "target_path": "input.gjf"}
            ],
            # 批量 native 合同由 foreach 自己选择每项 deck；不能再用单 deck
            # typed binding 先从聚合结果中做一次全局唯一选择。
            "bindings": [],
            "foreach": {
                "source_step_ref": "gjf-step",
                "source_field": "/result/items",
                # xyz_to_gjf 的真实 child_record 包装完整 ActionResult；稳定的
                # 逐项路径位于 ActionResult.result.output_gjf_path，ArtifactRef
                # 则位于同一 ActionResult.output_artifacts。
                "item_field": "/result/result/output_gjf_path",
                "item_id_field": "/item_id",
                "target_field": "staged_inputs.0.source_path",
                "max_items": 32,
            },
        },
    }
    completed = {
        "gjf-step": {
            "raw_output": {
                "result": {
                    "items": [
                        {
                            "item_id": "pair-a",
                            "pair_id": "pair-a",
                            "result": {
                                "status": "success",
                                "action": "xyz_to_gjf",
                                "result": {"output_gjf_path": first_deck["path"]},
                                "output_artifacts": [first_deck],
                            },
                        },
                        {
                            "item_id": "pair-b",
                            "pair_id": "pair-b",
                            "result": {
                                "status": "success",
                                "action": "xyz_to_gjf",
                                "result": {"output_gjf_path": second_deck["path"]},
                                "output_artifacts": [second_deck],
                            },
                        },
                    ]
                }
            },
            "output_artifacts": [first_deck, second_deck],
        }
    }

    resolved = resolve_minichem_step_request(native_step, completed)

    assert [item["item_id"] for item in resolved["_foreach_items"]] == [
        "pair-a",
        "pair-b",
    ]
    assert [
        item["request"]["staged_inputs"][0]["source_path"] for item in resolved["_foreach_items"]
    ] == [
        "artifacts/gjf-item-0001.gjf",
        "artifacts/gjf-item-0002.gjf",
    ]
    assert [item["provenance"]["pair_id"] for item in resolved["_foreach_items"]] == [
        "pair-a",
        "pair-b",
    ]


def test_foreach_provenance_drops_unbounded_nested_lineage() -> None:
    """foreach manifest 只保留标量身份和 Artifact id，不传播大对象。"""

    deck = build_artifact_ref(
        artifact_id="gjf-provenance-0001",
        semantic_type="GaussianInputDeck",
        media_type="chemical/x-gaussian-input",
        path="artifacts/gjf-provenance-0001.gjf",
        producer_action="xyz_to_gjf",
        producer_backend="format_conversion",
    )
    native_step = {
        "step_id": "gaussian-step",
        "request": {
            "software_id": "gaussian",
            "executable": "g16",
            "staged_inputs": [{"source_path": None, "target_path": "input.gjf"}],
            "bindings": [],
            "foreach": {
                "source_step_ref": "gjf-step",
                "source_field": "/result/items",
                "item_field": "/result/result/output_gjf_path",
                "item_id_field": "/item_id",
                "target_field": "staged_inputs.0.source_path",
                "max_items": 32,
            },
        },
    }
    completed = {
        "gjf-step": {
            "raw_output": {
                "result": {
                    "items": [
                        {
                            "item_id": "pair-a",
                            "provenance": {
                                "pair_id": "pair-a",
                                "lineage": {"raw_log": "x" * 10000},
                                "nested_paths": ["/private/secret"] * 100,
                            },
                            "result": {
                                "result": {"output_gjf_path": deck["path"]},
                                "output_artifacts": [deck],
                            },
                        }
                    ]
                }
            },
            "output_artifacts": [deck],
        }
    }

    resolved = resolve_minichem_step_request(native_step, completed)
    provenance = resolved["_foreach_items"][0]["provenance"]

    assert provenance["pair_id"] == "pair-a"
    assert provenance["gaussian_input_deck_artifact_ids"] == [deck["artifact_id"]]
    assert "lineage" not in provenance
    assert "nested_paths" not in provenance


def test_smiles2sdf_batch_handoff_contract_accepts_declared_mapping() -> None:
    """discovery 声明的 smiles2sdf→sdf_to_gjf 映射可通过静态合同。"""

    validate_canonical_workflow(_batch_workflow(), _batch_catalog())


def test_smiles2sdf_batch_handoff_rejects_tampered_item_field() -> None:
    """篡改 discovery 声明的逐项字段时，静态合同必须拒绝。"""

    catalog = _batch_catalog()
    workflow = _batch_workflow()
    workflow["steps"][1]["request"]["foreach"]["item_field"] = "/wrong"

    with pytest.raises(WorkflowContractError, match="batch_handoff"):
        validate_canonical_workflow(workflow, catalog)


def test_smiles2sdf_batch_handoff_rejects_tampered_target_action() -> None:
    """声明把结果交给另一 Action 时，目标 Action 身份必须被拒绝。"""

    catalog = _batch_catalog()
    workflow = _batch_workflow()
    source_contract = catalog["action_contracts"]["contracts"][0]["output_contract"]
    source_contract["batch_handoff"]["downstream_action_id"] = "smiles2sdf"
    workflow["steps"][0]["expected_outputs"]["output_contract"] = deepcopy(
        source_contract
    )

    with pytest.raises(WorkflowContractError, match="batch_handoff"):
        validate_canonical_workflow(workflow, catalog)


def test_smiles2sdf_batch_handoff_rejects_request_over_contract_limit() -> None:
    """请求 foreach 条数超过 discovery 上限时，静态合同必须拒绝。"""

    catalog = _batch_catalog()
    workflow = _batch_workflow()
    source_contract = catalog["action_contracts"]["contracts"][0]["output_contract"]
    source_contract["batch_handoff"]["max_items"] = 1
    workflow["steps"][0]["expected_outputs"]["output_contract"] = deepcopy(
        source_contract
    )
    workflow["steps"][1]["request"]["foreach"]["max_items"] = 2

    with pytest.raises(WorkflowContractError, match="超过合同上限"):
        validate_canonical_workflow(workflow, catalog)
