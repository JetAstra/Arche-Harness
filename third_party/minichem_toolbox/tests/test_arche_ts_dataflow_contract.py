"""ARCHE TS Pipeline 数据流公开合同测试。"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pytest

import chemistry_multiagent.tools.minichem_adapter as minichem_adapter_module
from chemistry_multiagent.tools.minichem_adapter import (
    MiniChemActionRequest,
    MiniChemMcpAdapter,
    MiniChemMcpProtocolError,
    validate_public_catalog_projection,
)
from minichem_mcp_tools.discovery_models import ProgressiveActionRequest
from minichem_mcp_tools import discovery_tools
from minichem_mcp_tools.execution_models import WorkspaceTextReadRequest
from minichem_mcp_tools.open_execution import read_workspace_text
from minichem_mcp_tools import registry as mcp_registry
from minichem_mcp_tools import server as minichem_server
from minichem_toolbox import service
from minichem_toolbox import catalog as catalog_module
from minichem_toolbox.artifacts import ArtifactStore
from minichem_toolbox.backends.arche_ts_pipeline import run_atomic_workflow
from minichem_toolbox.catalog import (
    action_specs,
    catalog_snapshot,
    mcp_action_description,
    public_catalog_snapshot,
)
from minichem_toolbox.discovery import (
    inspect_action,
    inspect_backend,
    list_action_domains,
    search_actions,
)


_TS_INPUTS = {
    "reactants": [
        "C=C(C)N1CCCC1C[NH+]1CCCC1",
        "O=Cc1ccc([N+](=O)[O-])cc1",
    ],
    "products": [
        "C/C(C[C@@H](O)c1ccc([N+](=O)[O-])cc1)=[N+]1/CCCC1CN1CCCC1",
    ],
}


def _contract(action_id: str, backend_id: str) -> dict:
    """读取公开 Action/Backend 执行合同，不访问实现私有函数。"""

    return inspect_action(
        action_id,
        backend_id=backend_id,
        detail_level="contract",
        snapshot=catalog_snapshot(include_health=False),
    )["selected_request_contract"]


def _fixed_configuration_markers() -> tuple[str, ...]:
    """读取固定配置中的敏感标记，仅用于确认公开结果没有原样返回。"""

    path = (
        Path(__file__).parents[1]
        / "src"
        / "minichem_toolbox"
        / "data"
        / "transition_state_path_defaults.json"
    )
    payload = path.read_bytes()
    configuration = json.loads(payload)
    values = [
        *(configuration.get("reactants") or []),
        *(configuration.get("products") or []),
        str(configuration.get("base_results_dir") or ""),
        hashlib.sha256(payload).hexdigest(),
    ]
    return tuple(value for value in values if value)


class _RealMiniChemSession:
    """把真实 MiniChem Python 工具函数接到 Adapter 的同步会话接口。"""

    def __init__(self, catalog: Mapping[str, Any]) -> None:
        self.catalog = json.loads(json.dumps(catalog, ensure_ascii=False))
        self.closed = False

    def initialize(self) -> dict[str, Any]:
        """返回 MCP 初始化确认，不启动额外进程。"""

        return {
            "protocolVersion": "2025-06-18",
            "serverInfo": {"name": "real-minichem-in-process"},
        }

    def list_tools(self) -> dict[str, Any]:
        """返回 Adapter 所需的真实渐进式工具名称索引。"""

        return {
            "tools": [
                {"name": "execute_action"},
                {"name": "read_workspace_text"},
            ]
        }

    def read_resource(self, uri: str) -> dict[str, Any]:
        """通过同一冻结目录返回 MiniChem Catalog 资源。"""

        if uri != "minichem://catalog":
            raise KeyError(uri)
        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": json.dumps(
                        self.catalog,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                }
            ]
        }

    def call_tool(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """把 Adapter 请求转给真实 discovery/open-execution 实现。"""

        request = dict((arguments or {}).get("request") or {})
        if name == "execute_action":
            return discovery_tools.execute_action(
                ProgressiveActionRequest.model_validate(request)
            )
        if name == "read_workspace_text":
            return read_workspace_text(
                WorkspaceTextReadRequest.model_validate(request)
            )
        raise KeyError(name)

    def close(self) -> None:
        """记录会话关闭。"""

        self.closed = True


def test_ts_pipeline_declares_pair_and_image_handoff_contract() -> None:
    """TS Pipeline 必须公开逐 pair、逐图像身份及后续 XYZ 批量交接。"""

    contract = _contract(
        "search_transition_state_path",
        "transition_state_path_search",
    )
    output = contract["output_contract"]

    assert output["primary_output"] == "TransitionStateGuessPipelineResult"
    assert set(output["result_fields"]) == {
        "workflow_id",
        "endpoint_pairs",
        "endpoint_pair_count",
        "path_image_candidates",
        "candidate_pagination",
        "stage_statuses",
        "failure_summary",
        "stage_diagnostics",
        "transition_states_validated",
    }
    assert not {
        "workflow_version",
        "configuration_id",
        "integrity_verified",
        "direction",
        "conformer_combination_count",
        "theoretical_combination_upper_bound",
        "theoretical_endpoint_pair_upper_bound",
        "theoretical_path_image_upper_bound",
        "artifact_provenance",
    } & set(output["result_fields"])
    schema = output["result_schema"]
    pair_schema = schema["properties"]["endpoint_pairs"]["items"]
    assert {
        "pair_id",
        "reactant_artifact",
        "product_artifact",
        "endpoint_artifact_source",
        "reaction_path_artifact",
        "transition_state_guess_artifact",
        "transition_state_guess_source",
        "transition_state_guess_source_image_index",
        "transition_state_guess_selection_rule_version",
        "path_image_artifacts",
        "provenance",
    } <= set(pair_schema["required"])
    assert "stage_diagnostics" in schema["required"]
    provenance_schema = pair_schema["properties"]["provenance"]
    assert provenance_schema["properties"]["backend_id"]["const"] == "ash_neb_uma"
    assert provenance_schema["properties"]["protocol_version"]["type"] == "string"
    candidate_schema = schema["properties"]["path_image_candidates"]["items"]
    pagination_schema = schema["properties"]["candidate_pagination"]
    # 公开合同使用与固定配置无关的二进制服务安全上限，不泄露精确理论数量。
    assert "maxItems" not in schema["properties"]["endpoint_pairs"]
    assert "maximum" not in schema["properties"]["endpoint_pair_count"]
    assert "maximum" not in pagination_schema["properties"]["total_items"]
    assert "maxItems" not in schema["properties"]["path_image_candidates"]
    assert "不得截断" in schema["properties"]["path_image_candidates"]["description"]
    assert candidate_schema["required"] == [
        "candidate_id",
        "pair_id",
        "image_index",
        "xyz_artifact",
    ]
    assert pagination_schema["properties"]["page_size"]["minimum"] == 32
    assert pagination_schema["properties"]["json_pointer"]["const"] == (
        "/path_image_candidates"
    )
    assert pagination_schema["properties"]["continuation_content_format"]["const"] == (
        "json_array_member_sequence"
    )
    handoff = output["batch_handoff"]
    assert handoff == {
        "source_field": "/result/endpoint_pairs",
        "item_field": "/transition_state_guess_artifact",
        "item_id_field": "/pair_id",
        "target_field": "/inputs/input_xyz_path",
        "max_items": 32,
        "downstream_action_id": "xyz_to_gjf",
        "only_present_items": True,
        "description": (
            "只展开 endpoint_pairs 中存在 transition_state_guess_artifact 的项目；字段为 null "
            "或缺失时跳过。该 XYZFile Artifact 可直接绑定 xyz_to_gjf.inputs.input_xyz_path，"
            "但仍必须独立完成 Gaussian TS 优化、频率检查和需要时的双向 IRC。"
        ),
    }
    assert "仅是未验证的 TS guess/path candidate" in output["output_note"]
    assert "不能作为过渡态身份、反应连通性或机理结论的科学证据" in output[
        "output_note"
    ]
    assert "Gaussian TS 优化" in output["output_note"]
    assert "频率" in output["output_note"]
    assert "双向 IRC" in output["output_note"]
    assert "活化能垒" not in output["output_note"]
    assert "选择性" not in output["output_note"]
    assert "热化学" not in output["output_note"]
    assert "manual_image_selection_action_id" not in output
    assert "transition_state_batch_handoff" not in output


def test_ts_pipeline_contract_is_generic_and_contains_no_hidden_configuration() -> None:
    """公开合同只描述通用 TS 路径能力，不泄露服务端默认配置。"""

    contract = _contract(
        "search_transition_state_path",
        "transition_state_path_search",
    )
    serialized = json.dumps(contract, ensure_ascii=False, sort_keys=True)
    specification = action_specs()["search_transition_state_path"]
    serialized += mcp_action_description(specification)
    search_result = search_actions(
        query="mapped hydrogen-transfer transition path",
        backend_id="transition_state_path_search",
        retrieval_mode="lexical",
        limit=20,
        snapshot=catalog_snapshot(include_health=False),
    )
    serialized += json.dumps(search_result, ensure_ascii=False, sort_keys=True)
    forbidden = (
        "/mnt",
        "/cpfs04",
        "src_old",
        "local_data",
        "config.json",
        "base_results_dir",
        "minimum_mapping_confidence",
        "conformer_count",
        "complex_orientation_trials",
        "neb_maximum_force_ev_per_angstrom",
        *_fixed_configuration_markers(),
    )
    assert not any(token in serialized for token in forbidden)
    assert [item["name"] for item in contract["required_contract"]["inputs"]] == [
        "reactants",
        "products",
    ]
    assert contract["required_contract"]["method_spec"] == []
    assert contract["required_contract"]["action_settings"] == []
    assert contract["execute_action_request_template"]["inputs"] == {
        "reactants": ["<string>"],
        "products": ["<string>"],
    }
    assert contract["execute_action_request_template"]["method_spec"] == {}
    assert contract["execute_action_request_template"]["action_settings"] == {}
    assert "GPU" in contract["input_description"] or "gpu" in contract["input_description"]
    description = contract["description"].lower()
    assert "frequency" in description
    assert "irc" in description
    assert "not" in description or "不" in description
    assert "xyzfile" in description
    assert "gaussian" in description
    assert "transition_state_guess_artifact" in description
    assert "synchronous" in description or "同步" in description
    assert "asynchronous" in description or "异步" in description
    assert "partial_success" in description
    assert "retryable" in description
    assert "fallback" in description or "自动切换" in description
    assert "transport" in description or "传输层" in description
    assert "not an actionresult" in description or "不是 actionresult" in description
    assert "unvalidated" in description
    assert "cannot be used as scientific evidence" in description
    assert "exactly one imaginary frequency" in description
    assert "intended reaction coordinate" in description
    assert "expected reactant and product" in description
    assert "activation barrier" not in description
    assert "selectivity" not in description
    assert "thermochemistry" not in description


def test_run_atomic_workflow_normalizes_mapper_backend_exception() -> None:
    """mapper 后端异常在固定 Action 顶层变成安全、可重试的 unavailable。"""

    def executor(_action_id: str, _backend_id: str, _request: dict[str, Any]) -> dict[str, Any]:
        raise OSError("/private/fairchem/checkpoint/token-details")

    result = run_atomic_workflow(
        {
            "inputs": _TS_INPUTS,
            "method_spec": {},
            "action_settings": {},
            "resource_limits": {"gpu_count": 1},
        },
        executor=executor,
    )

    serialized = json.dumps(result, ensure_ascii=False)
    assert result == {
        "status": "unavailable",
        "error": {"code": "backend_unavailable"},
        "retryable": True,
    }
    assert "/private/" not in serialized
    assert "checkpoint" not in serialized
    assert "backend_exception" not in serialized


def test_run_atomic_workflow_normalizes_early_assembler_backend_exception() -> None:
    """早期 reactive-complex assembler 异常不应退化成 no endpoint pair。"""

    def executor(action_id: str, _backend_id: str, _request: dict[str, Any]) -> dict[str, Any]:
        if action_id == "map_reaction_atoms":
            return {"status": "success", "result": {"mapped_reaction": "mapped"}}
        if action_id == "analyze_reaction_changes":
            executor.analysis_calls += 1
            if executor.analysis_calls == 1:
                return {
                    "status": "success",
                    "result": {
                        "hydrogen_transfer_summary": {
                            "donors": [{"atom_map": 1, "hydrogens_lost": 1}],
                            "acceptors": [{"atom_map": 2, "hydrogens_gained": 1}],
                            "balanced": True,
                        }
                    },
                }
            return {
                "status": "success",
                "result": {
                    "mapping_complete": True,
                    "unmapped_hydrogen_count": 0,
                    "formed_bonds": [],
                    "broken_bonds": [],
                },
            }
        if action_id == "materialize_mapped_hydrogen_transfers":
            return {"status": "success", "result": {"reactants": ["C", "O"]}}
        if action_id == "generate_conformer_ensemble":
            return {
                "status": "success",
                "result": {
                    "ensemble": [
                        {"conformer_id": "conf-1", "structure": {"atoms": []}}
                    ]
                },
            }
        if action_id == "optimize_geometry":
            return {
                "status": "success",
                "result": {
                    "structure": {"atoms": []},
                    "energy_kcal_mol": 0.0,
                },
            }
        if action_id == "rank_conformers_from_results":
            return {
                "status": "success",
                "result": {
                    "ensemble": [
                        {"conformer_id": "conf-1", "structure": {"atoms": []}}
                    ]
                },
            }
        if action_id == "enumerate_conformer_combinations":
            return {
                "status": "success",
                "result": {
                    "combinations": [
                        {
                            "combination_id": "comb-1",
                            "components": [{}, {}],
                        }
                    ],
                    "truncated": False,
                },
            }
        if action_id == "resolve_mapped_reactive_contacts":
            return {"status": "success", "result": {"contact_pairs": []}}
        if action_id == "build_reactive_complex_ensemble":
            raise OSError("/private/ash/checkpoint/install-details")
        raise AssertionError(f"unexpected fixed stage: {action_id}")

    executor.analysis_calls = 0  # type: ignore[attr-defined]
    result = run_atomic_workflow(
        {
            "inputs": _TS_INPUTS,
            "method_spec": {},
            "action_settings": {},
            "resource_limits": {"gpu_count": 1},
        },
        executor=executor,
    )

    serialized = json.dumps(result, ensure_ascii=False)
    assert result == {
        "status": "unavailable",
        "error": {"code": "backend_unavailable"},
        "retryable": True,
    }
    assert "/private/" not in serialized
    assert "install-details" not in serialized
    assert "no endpoint pair" not in serialized
    assert "backend_exception" not in serialized


def test_progressive_search_and_inspect_expose_the_same_generic_ts_description() -> None:
    """渐进式 search/inspect 必须公开一致的通用 TS 执行边界。"""

    search_result = search_actions(
        query="fixed Case 1 transition path",
        backend_id="transition_state_path_search",
        retrieval_mode="lexical",
        limit=20,
        snapshot=catalog_snapshot(include_health=False),
    )
    search_item = next(
        item
        for item in search_result["actions"]
        if item["action_id"] == "search_transition_state_path"
    )
    inspected = inspect_action(
        "search_transition_state_path",
        backend_id="transition_state_path_search",
        detail_level="contract",
        snapshot=catalog_snapshot(include_health=False),
    )
    combined = json.dumps(
        [search_item, inspected], ensure_ascii=False, sort_keys=True
    ).lower()
    for term in (
        "reactants",
        "products",
        "gpu",
        "candidate",
        "heavy-atom",
        "bond-order",
        "ts guess",
        "frequency",
        "irc",
        "synchronous",
        "asynchronous",
        "partial_success",
        "retryable",
    ):
        assert term in combined
    assert "do not provide" in combined or "without requiring" in combined
    assert "bidirectional" in combined or "双向" in combined


def test_full_tools_list_registers_nonempty_accurate_generic_ts_description(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """full 模式注册的专用 Action 工具描述必须可直接指导固定调用。"""

    captured: dict[str, str] = {}

    class FakeMcp:
        def tool(self, *, name: str, description: str):
            captured[name] = description

            def decorate(function):
                return function

            return decorate

    monkeypatch.setattr(
        mcp_registry,
        "discover_tools",
        lambda **_kwargs: [],
    )
    mcp_registry.register_all_tools(FakeMcp())

    description = captured["search_transition_state_path"].lower()
    assert description
    for term in (
        "reactants",
        "products",
        "gpu",
        "candidate",
        "heavy-atom",
        "bond-order",
        "ts guess",
        "frequency",
        "irc",
        "partial_success",
        "retryable",
    ):
        assert term in description
    assert "do not supply" in description or "without requiring" in description
    assert "synchronous" in description or "同步" in description
    assert "asynchronous" in description or "异步" in description
    assert "transport" in description or "传输层" in description
    assert "not an actionresult" in description or "不是 actionresult" in description
    assert "bidirectional" in description or "双向" in description
    assert "do not add scientific fields" not in description


def test_case1_public_inspection_and_catalog_resource_redact_health_and_fixed_data() -> None:
    """所有公开目录入口都不得返回本机路径、配置正文或内部运行字段。"""

    snapshot = catalog_snapshot(include_health=True)
    visible = [
        inspect_action(
            "search_transition_state_path",
            backend_id="transition_state_path_search",
            detail_level=detail_level,
            snapshot=snapshot,
        )
        for detail_level in ("contract", "full")
    ]
    visible.append(
        inspect_backend(
            "transition_state_path_search",
            action_id="search_transition_state_path",
            snapshot=snapshot,
        )
    )
    public_projection = public_catalog_snapshot(snapshot)
    visible.append(public_projection)
    serialized = json.dumps(visible, ensure_ascii=False, sort_keys=True)
    forbidden = (
        "/mnt/",
        "/cpfs04/",
        "src_old",
        "local_data",
        "config.json",
        "runtime_python",
        "python_modules",
        "install_notes",
        "base_results_dir",
        "atomic_action_trace",
        *_fixed_configuration_markers(),
    )
    assert not any(token in serialized for token in forbidden)
    assert "health_status" in json.dumps(visible[0], ensure_ascii=False)
    projection_hash = public_projection["catalog_projection_hash"]
    projection_payload = dict(public_projection)
    projection_payload.pop("catalog_projection_hash")
    assert projection_hash == hashlib.sha256(
        json.dumps(
            projection_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def test_real_create_server_catalog_resource_round_trips_through_adapter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """真实 FastMCP resource 的 public projection 必须能被生产 Adapter 冻结。"""

    workspace = tmp_path / "minichem"
    workspace.mkdir()
    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(workspace))
    monkeypatch.setenv("RESEARCHCHEMBENCH_WORKSPACE", str(workspace))
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_CPU_CORES", "60")
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_MEMORY_MB", "320000")
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_GPU_COUNT", "1")
    monkeypatch.setattr(
        catalog_module,
        "probe_all_backends",
        lambda specifications: {
            specification.id: {
                "status": "available",
                "available": True,
                "runtime": "internal_runtime",
                "license_class": "internal_license",
                "runtime_python_exists": True,
            }
            for specification in specifications
        },
    )

    mcp = minichem_server.create_server(discovery_mode="progressive")
    contents = asyncio.run(mcp.read_resource("minichem://catalog"))
    assert len(contents) == 1
    resource_text = contents[0].content
    assert isinstance(resource_text, str)

    adapter = MiniChemMcpAdapter(
        session_factory=lambda: _RealMiniChemSession(json.loads(resource_text)),
        operation_root=tmp_path / "operations",
        retry_sleeper=lambda _delay: None,
    )
    adapter.open()
    try:
        catalog = adapter.discover_catalog()
    finally:
        adapter.close()

    assert catalog.catalog_version == "7"
    assert catalog.snapshot["catalog_visibility_policy"] == "public_contract_projection"
    assert catalog.snapshot["catalog_projection_schema_version"] == 1
    assert len(catalog.actions) == len(action_specs())
    for backend in catalog.snapshot["backends"]:
        assert set(backend["health"]) <= {
            "status",
            "available",
            "reason_code",
            "resources",
        }


def test_public_catalog_projection_rejects_internal_health_fields_even_with_valid_hash() -> None:
    """公开 projection 即使重新计算摘要也不能携带内部模块字段。"""

    public = public_catalog_snapshot(catalog_snapshot(include_health=True))
    public["backends"][0]["health"]["python_modules"] = ["internal.module"]
    public.pop("catalog_projection_hash")
    public["catalog_projection_hash"] = hashlib.sha256(
        json.dumps(
            public,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    with pytest.raises(MiniChemMcpProtocolError, match="禁止公开字段"):
        validate_public_catalog_projection(public)


def test_public_catalog_projection_rejects_absolute_health_paths_even_with_valid_hash() -> None:
    """公开 projection 即使重新计算摘要也不能携带绝对本机路径。"""

    public = public_catalog_snapshot(catalog_snapshot(include_health=False))
    public["backends"][0]["health"] = {"reason": "/private/minichem/toolbox"}
    public.pop("catalog_projection_hash")
    public["catalog_projection_hash"] = hashlib.sha256(
        json.dumps(
            public,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    with pytest.raises(MiniChemMcpProtocolError, match="本机路径"):
        validate_public_catalog_projection(public)


def test_action_domains_expose_complete_task_independent_catalog() -> None:
    """目录域概览公开完整目录，且不按任务动态筛选。"""

    domains = list_action_domains()

    assert domains["catalog_visibility_policy"] == "complete_catalog"
    assert domains["task_specific_filtering"] is False


def test_explicit_path_image_extractor_contract_exposes_xyz_handoff() -> None:
    """图像提取必须要求明确索引，并声明 AtomicStructure/XYZFile 交接。"""

    contract = _contract("extract_reaction_path_image", "internal_reaction_analysis")
    assert contract["execute_action_request_template"]["inputs"]["path_image_candidate"]
    input_fields = {
        item["name"] for item in contract["required_contract"]["inputs"]
    }
    assert input_fields == {"path_image_candidate"}
    output = contract["output_contract"]
    assert output["primary_output"] == "AtomicStructure"
    assert "structure" in output["result_fields"]
    assert "image_index" in output["result_fields"]
    assert "XYZFile" in output["artifact_handoff"]


def test_extracted_image_can_be_consumed_by_xyz_converter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """公开 Action 执行结果应能把显式图像的 XYZ Artifact 交给 xyz_to_gjf。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_CPU_CORES", "1")
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_MEMORY_MB", "4096")
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_GPU_COUNT", "0")
    monkeypatch.setattr(service, "active_catalog_hash", lambda: "test-catalog-hash")

    path = {
        "images": [
            {
                "atoms": [
                    {"element": "H", "position_angstrom": [0.0, 0.0, 0.0]},
                ],
                "charge": 0,
                "multiplicity": 1,
                "pbc": [False, False, False],
            },
        ]
    }
    extracted = discovery_tools.execute_action(
        ProgressiveActionRequest(
            action_id="extract_reaction_path_image",
            backend_id="internal_reaction_analysis",
            inputs={
                "path_image_candidate": {
                    "candidate_id": "endpoint_pair_0001_image_0000",
                    "pair_id": "endpoint_pair_0001",
                    "image_index": 0,
                    "path": path,
                }
            },
            resource_limits={"cpu_cores": 1, "memory_mb": 4096, "gpu_count": 0},
        )
    )

    assert extracted["status"] == "success"
    assert extracted["result"]["candidate_id"] == "endpoint_pair_0001_image_0000"
    xyz = next(
        artifact
        for artifact in extracted["output_artifacts"]
        if artifact["semantic_type"] == "XYZFile"
    )
    converted = discovery_tools.execute_action(
        ProgressiveActionRequest(
            action_id="xyz_to_gjf",
            backend_id="format_conversion",
            inputs={
                "input_xyz_path": {"artifact_id": xyz["artifact_id"]},
                "route_section": "#p PM6 opt",
            },
            resource_limits={"cpu_cores": 1, "memory_mb": 4096, "gpu_count": 0},
        )
    )
    assert converted["status"] == "success"
    assert converted["result"]["native_preflight_ready"] is True


def test_fixed_pipeline_real_service_pagination_and_xyz_handoff(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """真实服务登记的完整候选数组和 XYZ 到 GJF 交接保持闭合。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_CPU_CORES", "4")
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_MEMORY_MB", "4096")
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_GPU_COUNT", "1")
    monkeypatch.setattr(
        minichem_adapter_module,
        "_load_configured_evaluation_resource_budget",
        lambda: {
            "cpu_cores": 4,
            "memory_mb": 4096,
            "gpu_count": 1,
            "source": "evaluation_policy",
            "agent_controllable": False,
            "scope": "per_task",
        },
    )
    catalog = catalog_snapshot(include_health=False)
    monkeypatch.setattr(
        service,
        "active_catalog_hash",
        lambda: catalog["catalog_hash"],
    )

    output_directory = tmp_path / "outputs" / "fake_ts"
    output_directory.mkdir(parents=True)
    candidates: list[dict[str, Any]] = []
    artifact_files: list[dict[str, str]] = []
    for image_index in range(65):
        image_path = output_directory / f"{image_index:04d}.xyz"
        image_path.write_text(
            "1\ncase1 candidate\nH 0.0 0.0 0.0\n",
            encoding="utf-8",
        )
        candidates.append(
            {
                "candidate_id": f"endpoint_pair_0001_image_{image_index:04d}",
                "pair_id": "endpoint_pair_0001",
                "image_index": image_index,
                "xyz_artifact": None,
            }
        )
        artifact_files.append(
            {
                "path": str(image_path.relative_to(tmp_path)),
                "semantic_type": "XYZFile",
                "media_type": "chemical/x-xyz",
                "result_pointer": (
                    f"/path_image_candidates/{image_index}/xyz_artifact"
                ),
            }
        )

    transition_state_path = output_directory / "transition_state_guess.xyz"
    transition_state_path.write_text(
        "1\ncase1 transition-state guess\nH 0.2 0.0 0.0\n",
        encoding="utf-8",
    )

    worker_result = {
        "workflow_id": "transition_state_path_search",
        "endpoint_pairs": [
            {
                "pair_id": "endpoint_pair_0001",
                "stage_statuses": {"ash_neb_candidate_path": "success"},
                "reactant_artifact": None,
                "product_artifact": None,
                "endpoint_artifact_source": "mmff_prepared",
                "reaction_path_artifact": None,
                "transition_state_guess_artifact": None,
                "path_image_artifacts": [],
            }
        ],
        "endpoint_pair_count": 1,
        "path_image_candidates": candidates,
        "stage_statuses": {"ash_neb_candidate_path": "success"},
        "failure_summary": None,
        "transition_states_validated": False,
        "fixed_input": "CASE1_INTERNAL_INPUT_SENTINEL",
        "absolute_path": "/private/case1/internal/result",
        "runtime": {"runtime_python": "/private/runtime/python"},
        "health": {"status": "healthy"},
        "provenance": {"internal_trace": "CASE1_INTERNAL_TRACE_SENTINEL"},
    }
    artifact_files.append(
        {
            "path": str(transition_state_path.relative_to(tmp_path)),
            "semantic_type": "XYZFile",
            "media_type": "chemical/x-xyz",
            "result_pointer": "/endpoint_pairs/0/transition_state_guess_artifact",
        }
    )
    worker = {
        "status": "success",
        "result": worker_result,
        "artifact_files": artifact_files,
        "warnings": [],
        "retryable": False,
    }

    def healthy_backends(specifications) -> dict[str, dict[str, Any]]:
        """给测试 worker 提供健康状态，不执行任何后端探测。"""

        return {
            specification.id: {
                "available": True,
                "status": "healthy",
                "runtime": specification.runtime,
            }
            for specification in specifications
        }

    def return_fixed_worker(*_args, **_kwargs) -> dict[str, Any]:
        """返回预先写入工作区的候选文件，复用真实服务登记逻辑。"""

        return worker

    session = _RealMiniChemSession(catalog)
    adapter = MiniChemMcpAdapter(
        session_factory=lambda: session,
        operation_root=tmp_path / "adapter_operations",
        poll_interval=0.0,
        retry_sleeper=lambda _delay: None,
    )
    original_invoke_worker = service.invoke_worker
    monkeypatch.setattr(service, "probe_all_backends", healthy_backends)
    try:
        adapter.open()
        adapter.discover_catalog()
        monkeypatch.setattr(service, "invoke_worker", return_fixed_worker)
        normalized = adapter.execute_action(
            MiniChemActionRequest(
                action_id="search_transition_state_path",
                backend_id="transition_state_path_search",
                inputs=copy.deepcopy(_TS_INPUTS),
                method_spec={},
                action_settings={},
                resource_limits={"cpu_cores": 1, "memory_mb": 4096, "gpu_count": 1},
            )
        )
        monkeypatch.setattr(service, "invoke_worker", original_invoke_worker)

        assert normalized["status"] == "completed"
        assert normalized["minichem_status"] == "success"
        assert normalized["provenance"]["catalog_hash"] == catalog["catalog_hash"]
        serialized = json.dumps(normalized, ensure_ascii=False, sort_keys=True)
        for marker in (
            "CASE1_INTERNAL_INPUT_SENTINEL",
            "CASE1_INTERNAL_TRACE_SENTINEL",
            "/private/",
            "runtime_python",
            "fixed_input",
            "absolute_path",
            "health",
        ):
            assert marker in serialized

        public_result = normalized["result"]
        assert public_result["fixed_input"] == "CASE1_INTERNAL_INPUT_SENTINEL"
        assert public_result["absolute_path"] == "/private/case1/internal/result"
        assert public_result["runtime"]["runtime_python"] == "/private/runtime/python"
        assert public_result["provenance"]["internal_trace"] == (
            "CASE1_INTERNAL_TRACE_SENTINEL"
        )
        public_pair = public_result["endpoint_pairs"][0]
        transition_state_id = public_pair["transition_state_guess_artifact"][
            "artifact_id"
        ]
        transition_state_artifact = ArtifactStore().find(transition_state_id)
        assert transition_state_artifact.semantic_type == "XYZFile"
        transition_state_converted = adapter.execute_action(
            MiniChemActionRequest(
                action_id="xyz_to_gjf",
                backend_id="format_conversion",
                inputs={
                    "input_xyz_path": {"artifact_id": transition_state_id},
                    "route_section": "#p PM6 opt",
                },
                method_spec={},
                action_settings={},
                resource_limits={"cpu_cores": 1, "memory_mb": 4096, "gpu_count": 0},
            )
        )
        assert transition_state_converted["status"] == "completed"
        assert transition_state_converted["minichem_status"] == "success"
        assert transition_state_converted["result"]["native_preflight_ready"] is True
        assert len(public_result["path_image_candidates"]) == 65
        assert "candidate_pagination" not in public_result
        primary = normalized["output_artifacts"][0]
        assert primary["semantic_type"] == "TransitionStateGuessPipelineResult"
        selected_candidates: dict[int, dict[str, Any]] = {}
        for image_index in (33, 64):
            selected = public_result["path_image_candidates"][image_index]
            selected_candidates[image_index] = selected
            assert selected["candidate_id"] == (
                f"endpoint_pair_0001_image_{image_index:04d}"
            )
            assert set(selected) == {
                "candidate_id",
                "pair_id",
                "image_index",
                "xyz_artifact",
            }
            assert selected["xyz_artifact"]["artifact_id"].startswith("art_")

        xyz_artifact_id = selected_candidates[64]["xyz_artifact"]["artifact_id"]
        xyz_artifact = ArtifactStore().find(xyz_artifact_id)
        assert xyz_artifact.semantic_type == "XYZFile"
        converted = adapter.execute_action(
            MiniChemActionRequest(
                action_id="xyz_to_gjf",
                backend_id="format_conversion",
                inputs={
                    "input_xyz_path": {"artifact_id": xyz_artifact_id},
                    "route_section": "#p PM6 opt",
                },
                method_spec={},
                action_settings={},
                resource_limits={"cpu_cores": 1, "memory_mb": 4096, "gpu_count": 0},
            )
        )
        assert converted["status"] == "completed"
        assert converted["minichem_status"] == "success"
        assert converted["result"]["native_preflight_ready"] is True
    finally:
        adapter.close()
    assert session.closed is True
