"""验证 MiniChem 实际 MCP 公开文本只服务于能力选择和合法请求构造。"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from mcp.server.fastmcp import FastMCP

from minichem_mcp_tools.execution_models import (
    AnalysisRuntimeListRequest,
    SoftwareInspectRequest,
    SoftwareListRequest,
)
from minichem_mcp_tools.registry import register_catalog_resources, register_public_tools
from minichem_mcp_tools.software_catalog import (
    inspect_software,
    list_analysis_runtimes,
    list_software,
    open_execution_prompt,
    software_resource_snapshot,
)
from minichem_toolbox.catalog import (
    action_specs,
    catalog_snapshot,
    mcp_action_description,
    progressive_toolbox_overview,
    public_catalog_snapshot,
)
from minichem_toolbox.discovery import inspect_action


_PUBLIC_DUMP_PATH = Path(
    "local_data/runtime/case1_harness/probes/public_mcp_descriptions.json"
)
_FORBIDDEN_TEXT = (
    re.compile(r"\bbm25\b", re.IGNORECASE),
    re.compile(r"\b(?:minilm|all-minilm)\b", re.IGNORECASE),
    re.compile(r"cached?\s+manual", re.IGNORECASE),
    re.compile(r"official\s+sources?", re.IGNORECASE),
    re.compile(r"retry\s+once", re.IGNORECASE),
    re.compile(r"\bruntime_python\b", re.IGNORECASE),
    re.compile(r"\bresolved_path\b", re.IGNORECASE),
    re.compile(r"\bcredentials?\b", re.IGNORECASE),
    re.compile(r"\benvironment\s+variables?\b", re.IGNORECASE),
    re.compile(r"\b(?:evaluator|author)\b", re.IGNORECASE),
    re.compile(r"\bARCHE\b"),
    re.compile(r"\bCase\b"),
    re.compile(r"(?:\bcase[\s_-]*[123]\b|案例\s*[123一二三])", re.IGNORECASE),
    re.compile(r"\btraceback\b", re.IGNORECASE),
    re.compile(r"\b(?:install|installation|installed)\b", re.IGNORECASE),
    re.compile(r"\b(?:author_default|case1_candidate_id|initial_centroid_distance_angstrom|random_stream_offset)\b"),
    re.compile(r"/(?:mnt|cpfs|home|root)/"),
    re.compile(r"\b(?:UB3LYP|wB97XD|def2TZVP|EthylEthanoate)\b"),
    re.compile(r"B3LYP/6-31G\(d\)|Temperature=303\.15|solvent=acetone", re.IGNORECASE),
)
_STABLE_ID_FIELDS = {
    "action_id",
    "backend_id",
    "resource_id",
    "software_id",
    "runtime",
    "source_id",
}
_STABLE_ID_VALUES = {
    "agent_backend_required",
    "agent_components_required",
    "agent_source_required",
    "agent_explicit_no_default",
    "agent_controllable",
    "generate_gaussian_code",
}
_FORBIDDEN_FIELD_NAMES = (
    re.compile(r"^(?:author_default|case1_candidate_id|initial_centroid_distance_angstrom|random_stream_offset)$"),
    re.compile(r"^(?:runtime_python|resolved_path|credentials?)$", re.IGNORECASE),
)


def _registered_surface(mode: str) -> dict[str, Any]:
    """收集指定发现模式下 FastMCP 实际注册的公开字段。"""

    server = FastMCP(f"public-contract-{mode}")
    register_public_tools(server, mode)
    register_catalog_resources(server, mode)
    return {
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.parameters,
            }
            for tool in server._tool_manager._tools.values()
        ],
        "resources": [
            {
                "uri": str(resource.uri),
                "name": resource.name,
                "description": resource.description,
            }
            for resource in server._resource_manager._resources.values()
        ],
    }


def _public_values() -> dict[str, Any]:
    """构造真实注册 surface 与公共投影的脱敏审计样本。"""

    snapshot = catalog_snapshot(include_health=False)
    surfaces = {mode: _registered_surface(mode) for mode in ("progressive", "full")}
    selected_contracts = {
        "sdf_to_gjf": inspect_action(
            "sdf_to_gjf",
            backend_id="format_conversion",
            detail_level="full",
            snapshot=snapshot,
        ),
        "xyz_to_gjf": inspect_action(
            "xyz_to_gjf",
            backend_id="format_conversion",
            detail_level="full",
            snapshot=snapshot,
        ),
        "gaussian_scan": inspect_action(
            "scan_reaction_coordinates",
            backend_id="gaussian",
            detail_level="full",
            snapshot=snapshot,
        ),
        "gaussian_codegen": inspect_action(
            "generate_gaussian_code",
            backend_id="gaussian_codegen",
            detail_level="full",
            snapshot=snapshot,
        ),
    }
    compact_contracts = {
        "sdf_to_gjf": inspect_action(
            "sdf_to_gjf",
            backend_id="format_conversion",
            detail_level="contract",
            snapshot=snapshot,
        ),
        "xyz_to_gjf": inspect_action(
            "xyz_to_gjf",
            backend_id="format_conversion",
            detail_level="contract",
            snapshot=snapshot,
        ),
        "gaussian_scan": inspect_action(
            "scan_reaction_coordinates",
            backend_id="gaussian",
            detail_level="contract",
            snapshot=snapshot,
        ),
        "gaussian_codegen": inspect_action(
            "generate_gaussian_code",
            backend_id="gaussian_codegen",
            detail_level="contract",
            snapshot=snapshot,
        ),
    }
    return {
        "fastmcp": surfaces,
        "catalog": public_catalog_snapshot(snapshot),
        "progressive_overview": progressive_toolbox_overview(snapshot=snapshot),
        "action_descriptions": {
            action_id: mcp_action_description(specification)
            for action_id, specification in action_specs().items()
        },
        "selected_contracts": selected_contracts,
        "compact_contracts": compact_contracts,
        "software": {
            "list": list_software(SoftwareListRequest(limit=200)),
            "inspect_gaussian": inspect_software(
                SoftwareInspectRequest(software_id="gaussian")
            ),
            "inspect_format_conversion": inspect_software(
                SoftwareInspectRequest(software_id="format_conversion")
            ),
            "runtimes": list_analysis_runtimes(
                AnalysisRuntimeListRequest(limit=200, include_details=True)
            ),
            "resource": software_resource_snapshot(),
            "prompt": open_execution_prompt(include_command_index=True),
        },
    }


def _walk_strings(value: Any, path: str = "$"):
    """只遍历已构造的公开值，不读取源码、文档文件或 vendor。"""

    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _walk_strings(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_strings(item, f"{path}[{index}]")


def _walk_keys(value: Any, path: str = "$"):
    """遍历公开对象的字段名，检查内部兼容字段是否以 key 形式泄露。"""

    if isinstance(value, dict):
        for key, item in value.items():
            yield path, str(key)
            yield from _walk_keys(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_keys(item, f"{path}[{index}]")


def _assert_orphan_safe(value: Any) -> None:
    """拒绝会把内部过程或宿主细节带给 Planner/模型的公开文本。"""

    findings = []
    key_findings = []
    for path, key in _walk_keys(value):
        if any(pattern.search(key) for pattern in _FORBIDDEN_FIELD_NAMES):
            key_findings.append((path, key))
    for path, text in _walk_strings(value):
        field_name = path.rsplit(".", 1)[-1].split("[", 1)[0]
        if field_name in _STABLE_ID_FIELDS or text in _STABLE_ID_VALUES:
            continue
        for pattern in _FORBIDDEN_TEXT:
            if pattern.search(text):
                findings.append((path, pattern.pattern, text[:240]))
                break
    assert not key_findings, "公开字段名包含内部兼容字段: " + repr(key_findings[:8])
    assert not findings, "公开值包含内部过程或宿主细节: " + repr(findings[:8])


@pytest.fixture(scope="module")
def public_values() -> dict[str, Any]:
    """生成一次公开 dump，供本模块测试和 Who-Read 审计复用。"""

    value = _public_values()
    _PUBLIC_DUMP_PATH.parent.mkdir(parents=True, exist_ok=True)
    _PUBLIC_DUMP_PATH.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return value


def test_registered_progressive_and_full_surface_is_orphan_safe(
    public_values: dict[str, Any],
) -> None:
    """实际 progressive/full 注册的名称、说明、Schema 和资源说明均应可对外。"""

    _assert_orphan_safe(public_values["fastmcp"])
    for mode in ("progressive", "full"):
        surface = public_values["fastmcp"][mode]
        assert surface["tools"]
        assert {item["uri"] for item in surface["resources"]} == {
            "minichem://catalog",
            "minichem://software",
            "minichem://execution-policy",
        }
        assert all(
            item["name"] and item["description"] and isinstance(item["inputSchema"], dict)
            for item in surface["tools"]
        )


def test_registered_input_schema_preserves_machine_and_science_contracts(
    public_values: dict[str, Any],
) -> None:
    """实际工具 Schema 保留科学请求字段，不发布资源选择字段。"""

    full_tools = {
        item["name"]: item for item in public_values["fastmcp"]["full"]["tools"]
    }
    schema = full_tools["generate_gaussian_code"]["inputSchema"]
    assert schema["required"] == ["request"]
    action_request = schema["$defs"]["GaussianActionRequest"]
    assert {
        "backend_id",
        "inputs",
        "method_spec",
        "action_settings",
    } <= set(action_request["properties"])
    assert "resource_limits" not in action_request["properties"]

    shared = full_tools["scan_reaction_coordinates"]["inputSchema"]
    shared_request = shared["$defs"]["GaussianCapableActionRequest"]
    assert "resource_limits" not in shared_request["properties"]

    progressive_tools = {
        item["name"]: item
        for item in public_values["fastmcp"]["progressive"]["tools"]
    }
    progressive_request = progressive_tools["execute_action"]["inputSchema"][
        "$defs"
    ]["ProgressiveActionRequest"]
    assert "resource_limits" not in progressive_request["properties"]

    native_request = full_tools["submit_native_job"]["inputSchema"]["$defs"][
        "NativeJobRequest"
    ]
    assert "resource_limits" not in native_request["properties"]

    gaussian = public_values["selected_contracts"]["gaussian_scan"][
        "selected_request_contract"
    ]
    assert gaussian["resource_policy"] == {
        "mode": "service_managed_full_capacity",
        "concurrency": "serial",
        "agent_controllable": False,
    }
    assert "resource_limits" not in gaussian["execute_action_request_template"]
    assert gaussian["sections"]["resource_limits"] == {
        "service_managed": gaussian["resource_policy"]
    }


def test_public_catalog_and_software_contracts_are_explicit_and_neutral(
    public_values: dict[str, Any],
) -> None:
    """catalog/software 公共投影保留请求语义而不携带固定路线或宿主预算。"""

    _assert_orphan_safe(
        {
            "catalog": public_values["catalog"],
            "overview": public_values["progressive_overview"],
            "actions": public_values["action_descriptions"],
            "contracts": public_values["selected_contracts"],
            "compact_contracts": public_values["compact_contracts"],
            "software": public_values["software"],
        }
    )
    for key in ("sdf_to_gjf", "xyz_to_gjf"):
        contract = public_values["selected_contracts"][key]["selected_request_contract"]
        fields = contract["sections"]["inputs"]["optional"]
        route_field = next(item for item in fields if item["name"] in {"route_parameters", "route_section"})
        assert "default" not in route_field
    gaussian_budget = public_values["software"]["inspect_gaussian"][
        "native_invocation_guides"
    ][0]["native_job_request_template"]["resource_policy"]
    assert gaussian_budget == {
        "mode": "service_managed_full_capacity",
        "concurrency": "serial",
        "agent_controllable": False,
    }
    assert "resource_limits" not in str(
        public_values["software"]["inspect_gaussian"]["native_invocation_guides"]
    )
    non_gaussian_description = public_values["action_descriptions"][
        "derive_ecd_spectrum"
    ]
    assert "service_managed_full_capacity" not in non_gaussian_description
    assert "Gaussian" not in non_gaussian_description


def test_action_descriptions_match_current_input_and_output_boundaries(
    public_values: dict[str, Any],
) -> None:
    """公开 Action 文案必须反映单结构受力和 route-only Gaussian 能力。"""

    forces = action_specs()["calculate_forces"]
    assert forces.description == "Calculate atomic forces for one non-periodic structure."
    assert forces.input_description == "one non-periodic AtomicStructure"
    force_text = public_values["action_descriptions"]["calculate_forces"].casefold()
    assert "one non-periodic structure" in force_text
    assert "aligned batch" not in force_text

    gaussian = action_specs()["generate_gaussian_code"]
    gaussian_text = gaussian.description.casefold()
    assert "optional" in gaussian_text
    assert "non-default" in gaussian_text
    assert "draft one gaussian route line" in gaussian_text
    assert "route text" in gaussian_text
    assert "safe shape" in gaussian_text
    assert "scientific validation" in gaussian_text
    assert "scientific evidence" in gaussian_text
    assert "demand" not in gaussian_text
    assert "generation" not in gaussian_text
    assert "extraction" not in gaussian_text
    assert "correction" not in gaussian_text
    assert "explicit method" in gaussian.input_description.casefold()
    assert "explicit method or settings" in gaussian.input_description.casefold()
    assert "input deck" in gaussian.input_description.casefold()
    assert "run gaussian" in gaussian.input_description.casefold()
    assert "scientific evidence" in gaussian.input_description.casefold()

    gaussian_contract = inspect_action(
        "generate_gaussian_code",
        backend_id="gaussian_codegen",
        detail_level="full",
        snapshot=catalog_snapshot(include_health=False),
    )
    handoff_note = gaussian_contract["selected_request_contract"]["output_contract"][
        "input_handoff_note"
    ]
    assert "不生成完整输入文件" in handoff_note
    assert "不提交或执行 Gaussian" in handoff_note


def test_protonation_parameter_descriptions_are_provider_aware() -> None:
    """inspect_action 的参数说明必须区分 RDKit 与 PDBFixer 的 pH 语义。"""

    rdkit_contract = inspect_action(
        "assign_protonation_states",
        backend_id="rdkit",
        detail_level="full",
        snapshot=catalog_snapshot(include_health=False),
    )
    rdkit_parameters = rdkit_contract["provider_contracts"][0][
        "action_parameter_descriptions"
    ]
    assert "does not use it to choose protonation" in rdkit_parameters["ph"].casefold()
    assert "does not predict pka" in rdkit_parameters["ph"].casefold()
    assert "add_explicit_hydrogens" in rdkit_parameters["rule"]
    assert "without pka inference" in rdkit_parameters["rule"].casefold()

    from minichem_toolbox.backend_specs import BACKEND_SPECS as full_backend_specs

    pdbfixer = next(item for item in full_backend_specs if item.id == "pdbfixer")
    assert "PDBFixer.addMissingHydrogens(ph)" in pdbfixer.method_schema["ph"]
    assert "pH-dependent" in pdbfixer.method_schema["ph"]
