"""MiniChem Action 发现合同的最小请求模板与字段边界回归测试。"""

from __future__ import annotations

import json

import pytest

from minichem_toolbox.backends.cheminformatics import execute as execute_cheminformatics
from minichem_toolbox.catalog import action_specs
from minichem_toolbox.discovery import inspect_action
from minichem_mcp_tools.execution_models import AnalysisJobRequest, NativeJobRequest


def _selected_contract(action_id: str, backend_id: str) -> dict:
    """读取指定 Action/Backend 的紧凑执行合同。"""

    return inspect_action(
        action_id,
        backend_id=backend_id,
        detail_level="contract",
    )["selected_request_contract"]


def _inspected_contract(
    action_id: str,
    backend_id: str,
    detail_level: str,
) -> dict:
    """读取指定发现层级的 Action/Backend 请求合同。"""

    return inspect_action(
        action_id,
        backend_id=backend_id,
        detail_level=detail_level,
    )["selected_request_contract"]


def _required_field(
    contract: dict,
    section: str,
    field_name: str,
) -> dict:
    """从紧凑合同中读取一个必填字段的完整公开描述。"""

    return next(
        item
        for item in contract["required_contract"][section]
        if item["name"] == field_name
    )


@pytest.mark.parametrize("request_type", [NativeJobRequest, AnalysisJobRequest])
def test_execution_request_public_schema_hides_internal_operation_id(request_type) -> None:
    """执行请求公开 Schema 保留描述，但完整 JSON 不泄漏内部身份字段。"""

    schema = request_type.model_json_schema()
    serialized_schema = json.dumps(schema, ensure_ascii=False, sort_keys=True)

    assert schema["description"]
    assert "operation_id" not in serialized_schema


def test_smiles2sdf_template_preserves_registered_array_shape() -> None:
    """smiles_list 的最小模板必须保持非空数组容器。"""

    contract = _selected_contract("smiles2sdf", "format_conversion")

    template_value = contract["execute_action_request_template"]["inputs"][
        "smiles_list"
    ]

    assert isinstance(template_value, list)
    assert template_value


def test_compact_contract_preserves_ordinary_action_descriptions() -> None:
    """普通 Action 的紧凑合同必须保留自身说明，不能只让固定流程有说明。"""

    specification = action_specs()["optimize_geometry"]
    contract = _selected_contract("optimize_geometry", "fairchem_uma")

    assert contract["input_description"] == specification.input_description
    assert contract["description"] == specification.description


def test_transition_state_contract_exposes_generic_backend_metadata_and_handoff() -> None:
    """TS capability 通过普通 Action/Backend 合同公开适用范围和下游交接。"""

    inspected = inspect_action(
        "search_transition_state_path",
        backend_id="transition_state_path_search",
        detail_level="full",
    )
    selected = inspected["selected_request_contract"]
    output = selected["output_contract"]
    provider = inspected["provider_contracts"][0]
    descriptions = inspected["action"]["description"] + provider.get("description", "")

    assert "at least two reactant components" in descriptions
    assert "charge +1" in descriptions
    assert "requires a GPU" in descriptions
    assert {"endpoint_pairs", "path_image_candidates", "transition_states_validated"} <= set(
        output["result_fields"]
    )
    validation_schema = output["result_schema"]["properties"][
        "transition_states_validated"
    ]
    assert validation_schema["const"] is False
    assert "不能作为" in validation_schema["description"]
    assert "过渡态证据" in validation_schema["description"]
    assert "恰好一个" in output["output_note"]
    assert "目标反应坐标" in output["output_note"]
    assert "预期反应物和产物" in output["output_note"]
    assert output["batch_handoff"]["source_field"] == "/result/endpoint_pairs"
    assert output["batch_handoff"]["item_field"] == "/transition_state_guess_artifact"
    assert output["batch_handoff"]["downstream_action_id"] == "xyz_to_gjf"
    assert inspected["execution_timeout_policy"]["agent_controllable"] is False
    assert inspected["evaluation_resource_budget"]["agent_controllable"] is False
    assert provider["resource_constraints"]["minimum_gpu_count"] == 1
    fixed_paths = set(provider["backend_fixed_parameters"])
    assert {
        "backend_runtime.direction",
        "backend_runtime.charge",
        "backend_runtime.multiplicity",
    } <= fixed_paths
    assert not any(
        path.startswith("execution_policy.")
        or path.startswith("evaluation_resource_budget.")
        for path in fixed_paths
    )
    assert "action_parameter_descriptions" in provider


def test_fairchem_contract_pins_the_supported_uma_checkpoint() -> None:
    """Planner 必须看到可执行 checkpoint 枚举，不能把产品族名当模型名。"""

    contract = _selected_contract("optimize_geometry", "fairchem_uma")
    options = {
        item["path"]: item["allowed_values"]
        for item in contract["enumerated_options"]
    }

    assert options["method_spec.model"] == ["uma-s-1p1"]


def test_rdkit_structure_contract_exposes_literal_smiles_syntax_requirement() -> None:
    """RDKit 三维结构合同必须说明字面量 SMILES 需要先满足解析语法。"""

    contract = _selected_contract("generate_3d_structure", "rdkit")

    assert any(
        "inputs.molecule" in note
        and "SMILES" in note
        and "RDKit" in note
        and "可解析" in note
        for note in contract["usage_notes"]
    )


def test_substructure_output_contract_exposes_zero_based_match_indices() -> None:
    """子结构检索合同必须公开后端真实返回的零起始匹配索引。"""

    expected_fields = ["query", "query_format", "matches", "match_count"]

    for detail_level in ("full", "contract"):
        output_contract = _inspected_contract(
            "search_local_substructures",
            "rdkit",
            detail_level,
        )["output_contract"]

        assert output_contract["result_fields"] == expected_fields
        result_schema = output_contract["result_schema"]
        assert result_schema["required"] == expected_fields
        assert set(result_schema["properties"]) == set(expected_fields)
        assert result_schema["properties"]["query"]["type"] == "string"
        assert result_schema["properties"]["query_format"]["enum"] == [
            "smarts",
            "smiles",
        ]
        matches = result_schema["properties"]["matches"]
        assert matches["type"] == "array"
        assert matches["items"]["type"] == "array"
        assert matches["items"]["items"] == {"type": "integer", "minimum": 0}
        assert result_schema["properties"]["match_count"] == {
            "type": "integer",
            "minimum": 0,
        }
        assert "完整索引数组使用 /result/matches/0；" in output_contract[
            "output_note"
        ]
        assert "/result/matches/0/0" in output_contract["output_note"]
        assert "零起始" in output_contract["output_note"]


def test_substructure_output_contract_matches_backend_result_shape() -> None:
    """子结构输出 Schema 的字段和嵌套数组必须与真实 RDKit 结果一致。"""

    response = execute_cheminformatics(
        "search_local_substructures",
        "rdkit",
        {
            "inputs": {
                "molecule": {"smiles": "CC"},
                "query": "C",
            },
            "method_spec": {"query_format": "smarts"},
            "action_settings": {
                "max_matches": 10,
                "unique": True,
                "use_chirality": True,
            },
            "resource_limits": {},
        },
    )
    output_contract = _inspected_contract(
        "search_local_substructures",
        "rdkit",
        "contract",
    )["output_contract"]
    result = response["result"]

    assert response["status"] == "success"
    assert set(result) == set(output_contract["result_fields"])
    assert result["matches"] == [[0], [1]]
    assert result["match_count"] == len(result["matches"])


def test_gaussian_contract_exposes_solvent_keyword_spelling_requirement() -> None:
    """Gaussian 合同说明原生关键字要求，不预选方法、基组或溶剂。"""

    contract = _selected_contract("scan_reaction_coordinates", "gaussian")

    assert any(
        "method_spec.method" in note
        and "method_spec.basis" in note
        and "原生" in note
        for note in contract["usage_notes"]
    )
    assert any(
        "method_spec.solvent" in note
        and "空格" in note
        and "原生拼写" in note
        for note in contract["usage_notes"]
    )
    serialized = json.dumps(contract, ensure_ascii=False)
    assert not any(
        token in serialized
        for token in ("wB97XD", "def2TZVP", "EthylEthanoate")
    )


def test_gaussian_codegen_and_deck_handoff_expose_native_format_requirements() -> None:
    """codegen 与 deck 合同说明格式边界但不泄露固定路线或资源数值。"""

    codegen = _selected_contract("generate_gaussian_code", "gaussian_codegen")
    converter = _selected_contract("xyz_to_gjf", "format_conversion")
    sdf_converter = _selected_contract("sdf_to_gjf", "format_conversion")

    assert any(
        "solvent" in note and "原生关键字" in note and "空格" in note
        for note in codegen["usage_notes"]
    )
    assert [item["name"] for item in codegen["required_contract"]["inputs"]] == [
        "user_input"
    ]
    assert codegen["optional_field_names"]["method_spec"] == []
    assert codegen["optional_field_names"]["action_settings"] == []
    handoff_note = codegen["output_contract"]["input_handoff_note"]
    assert "自然语言" in handoff_note
    assert "不生成完整输入文件" in handoff_note
    assert "不提交或执行 Gaussian" in handoff_note
    assert any(
        "GaussianInputDeck" in note
        and "CPU" in note
        and "内存" in note
        for note in converter["usage_notes"]
    )
    serialized = json.dumps(
        {"codegen": codegen, "converter": converter}, ensure_ascii=False
    )
    assert not any(
        token in serialized
        for token in ("wB97XD", "def2TZVP", "EthylEthanoate", "UB3LYP")
    )
    assert all(
        "resource_limits" not in note
        and "nprocshared" not in note.casefold()
        and "inputs.mem" not in note.casefold()
        for note in converter["usage_notes"]
    )
    assert any(
        "inputs.charge" in note
        and "inputs.multiplicity" in note
        and "SDF" in note
        and "xyz_to_gjf" in note
        for note in sdf_converter["usage_notes"]
    )


def test_array_template_uses_registered_integer_member_shape() -> None:
    """数组成员类型应来自合并后的字段合同，而不是字段名猜测。"""

    contract = _selected_contract("scan_reaction_coordinates", "gaussian")

    template_value = contract["execute_action_request_template"]["action_settings"][
        "atom_indices"
    ]

    assert template_value == ["<integer>"]


def test_smiles2sdf_batch_output_contract_is_shared_by_complete_and_compact() -> None:
    """完整与紧凑发现合同必须共同声明成功项目的有界交接。"""

    expected_result_fields = [
        "input_count",
        "molecules_written",
        "failed_item_count",
        "output_sdf_path",
        "messages",
        "items",
        "successful_records",
    ]
    expected_handoff = {
        "source_field": "/result/successful_records",
        "item_field": "/sdf_artifact",
        "item_id_field": "/item_id",
        "target_field": "/inputs/sdf_path",
        "max_items": 32,
        "downstream_action_id": "sdf_to_gjf",
    }

    for detail_level in ("full", "contract"):
        contract = _inspected_contract(
            "smiles2sdf",
            "format_conversion",
            detail_level,
        )
        output_contract = contract["output_contract"]

        assert output_contract["primary_output"] == "SDFFile"
        assert output_contract["result_fields"] == expected_result_fields
        assert {
            key: output_contract["batch_handoff"][key]
            for key in expected_handoff
        } == expected_handoff
        assert output_contract["batch_handoff"]["only_successful_items"] is True
        assert "失败" in output_contract["batch_handoff"]["description"]

        result_schema = output_contract["result_schema"]
        assert result_schema["type"] == "object"
        assert result_schema["required"] == [
            "input_count",
            "molecules_written",
            "failed_item_count",
            "messages",
            "items",
            "successful_records",
        ]
        assert "output_sdf_path" not in result_schema["required"]
        assert "success" not in result_schema["required"]
        assert result_schema["properties"]["input_count"]["minimum"] == 1
        assert result_schema["properties"]["output_sdf_path"]["type"] == "string"
        assert result_schema["properties"]["success"]["type"] == "boolean"
        successful_records = result_schema["properties"]["successful_records"]
        assert successful_records["type"] == "array"
        assert successful_records["maxItems"] == 32
        item_schema = successful_records["items"]
        assert item_schema["required"] == [
            "item_id",
            "input_index",
            "sdf_artifact",
        ]
        artifact_schema = item_schema["properties"]["sdf_artifact"]
        assert artifact_schema["artifact_type"] == "ArtifactRef"
        assert artifact_schema["semantic_type"] == "SDFRecordFile"

        items_schema = result_schema["properties"]["items"]
        assert items_schema["minItems"] == 1
        assert items_schema["maxItems"] == 32
        assert items_schema["items"]["required"] == [
            "item_id",
            "input_index",
            "smiles",
            "status",
        ]
        assert "error" in items_schema["items"]["properties"]

        handoff_text = " ".join(
            str(value)
            for value in (
                output_contract["artifact_handoff"],
                output_contract["input_handoff_note"],
                output_contract["batch_handoff"]["description"],
            )
        )
        assert "SDFFile" in handoff_text
        assert "SDFRecordFile" in handoff_text
        assert "successful_records" in handoff_text
        assert "失败" in handoff_text
        assert "artifact.semantic_type=SDFFile" in output_contract["artifact_handoff"]
        assert "successful_records/0" not in output_contract["artifact_handoff"]


def test_candidate_selection_contract_documents_stable_candidate_id_format() -> None:
    """候选选择合同应公开稳定 ID 格式，不让 Planner 把数组位置当作 ID。"""

    for detail_level in ("full", "contract"):
        contract = _inspected_contract(
            "select_reactive_complex_candidate",
            "internal_reactive_complex",
            detail_level,
        )
        action_settings = (
            contract["sections"]["action_settings"]["required"]
            if "sections" in contract
            else contract["required_contract"]["action_settings"]
        )
        candidate_id = next(
            item for item in action_settings if item["name"] == "candidate_id"
        )

        assert candidate_id["type"] == "string"
        assert "candidate_0001" in candidate_id["description"]
        assert "数组位置" in candidate_id["description"]


def test_scan_output_contract_exposes_only_fields_shared_by_scan_backends() -> None:
    """一维扫描合同只投影 Gaussian 与 pysisyphus 都实际返回的字段。"""

    expected_fields = [
        "coordinate_type",
        "atom_indices",
        "coordinate_values",
        "coordinate_unit",
        "energies_hartree",
        "structures",
        "data_file",
        "completed_points",
        "requested_points",
    ]

    for backend_id in ("gaussian", "pysisyphus"):
        output_contract = _selected_contract(
            "scan_reaction_coordinates",
            backend_id,
        )["output_contract"]
        assert output_contract["result_fields"] == expected_fields
        result_schema = output_contract["result_schema"]
        assert result_schema["required"] == expected_fields
        assert set(result_schema["properties"]) == set(expected_fields)

    assert "converged" not in expected_fields
    assert "relative_energies_kcal_mol" not in expected_fields


@pytest.mark.parametrize(
    (
        "backend_id",
        "expected_fields",
        "interior_index_field",
        "energies_type",
        "source_field",
    ),
    [
        (
            "pysisyphus",
            [
                "path_file",
                "image_files",
                "image_count",
                "energies_hartree",
                "energy_source",
                "highest_energy_image_index",
                "highest_interior_energy_image_index",
                "highest_energy_image",
                "highest_interior_energy_image",
                "converged",
                "endpoints_consumed",
                "transition_state_validated",
            ],
            "highest_interior_energy_image_index",
            ["array", "null"],
            "/result/highest_interior_energy_image_index",
        ),
        (
            "ash_neb_uma",
            [
                "images",
                "energies_hartree",
                "energy_image_indices",
                "energy_unit",
                "highest_energy_image_index",
                "highest_interior_image_index",
                "transition_state_guess",
                "diagnostic_ash_saddle_geometry",
                "interior_peak_candidate_valid",
                "converged",
                "energy_index_complete",
                "final_path_complete",
                "total_images",
                "intermediate_images_passed_to_ash",
                "climbing_image",
                "maximum_force_ev_per_angstrom",
                "rms_force_ev_per_angstrom",
                "maximum_steps",
                "interpolation",
                "model",
                "task_name",
                "device",
                "allow_model_download",
                "path_file",
                "initial_guess_path",
            ],
            "highest_interior_image_index",
            "array",
            "/result/highest_interior_image_index",
        ),
    ],
)
def test_reaction_path_contract_exposes_backend_stable_candidate_fields(
    backend_id: str,
    expected_fields: list[str],
    interior_index_field: str,
    energies_type: str | list[str],
    source_field: str,
) -> None:
    """路径搜索合同应公开实际字段，并保留未验证候选图像语义。"""

    contract = _selected_contract("search_reaction_path", backend_id)
    output_contract = contract["output_contract"]

    assert output_contract["primary_output"] == "ReactionPath"
    assert output_contract["result_fields"] == expected_fields
    result_schema = output_contract["result_schema"]
    assert result_schema["required"] == expected_fields
    assert result_schema["properties"]["energies_hartree"]["type"] == energies_type
    assert (
        result_schema["properties"][interior_index_field]["type"]
        == ["integer", "null"]
    )
    assert "未验证" in output_contract["output_note"]
    assert output_contract["image_selection_handoff"] == {
        "downstream_action_id": "extract_reaction_path_image",
        "source_field": source_field,
        "target_field": "/inputs/path_image_candidate/image_index",
        "description": (
            "只把显式公开的候选图像索引绑定到 extract_reaction_path_image；"
            "该图像仍是未验证候选，不能据此断言过渡态，也不按数组下标推断。"
        ),
    }


def test_reaction_coordinate_analysis_contract_exposes_highest_image_pointer() -> None:
    """路径分析结果必须公开最高能量候选图像索引供下游显式提取。"""

    contract = _selected_contract(
        "analyze_reaction_coordinate", "internal_reaction_analysis"
    )
    output_contract = contract["output_contract"]

    assert output_contract["primary_output"] == "ReactionCoordinateAnalysisResult"
    assert "highest_energy_image_index" in output_contract["result_fields"]
    assert output_contract["result_schema"]["properties"][
        "highest_energy_image_index"
    ] == {"type": "integer", "minimum": 0}
    assert "未验证" in output_contract["output_note"]
    assert output_contract["image_selection_handoff"] == {
        "downstream_action_id": "extract_reaction_path_image",
        "source_field": "/result/highest_energy_image_index",
        "target_field": "/inputs/path_image_candidate/image_index",
        "description": (
            "只把显式公开的候选图像索引绑定到 extract_reaction_path_image；"
            "该图像仍是未验证候选，不能据此断言过渡态，也不按数组下标推断。"
        ),
    }


@pytest.mark.parametrize(
    ("action_id", "field_name", "semantic_type"),
    [
        (
            "analyze_free_energy_selectivity",
            "free_energies",
            "LabeledFreeEnergyCollection",
        ),
        (
            "analyze_energetic_span",
            "cycle_states",
            "CatalyticCycleFreeEnergyProfile",
        ),
    ],
)
def test_free_energy_templates_use_inline_array_examples(
    action_id: str,
    field_name: str,
    semantic_type: str,
) -> None:
    """自由能输入的最小模板必须展示可直接执行的内联数组。"""

    contract = _selected_contract(action_id, "internal_thermochemistry")
    field_contract = _required_field(contract, "inputs", field_name)
    template_value = contract["execute_action_request_template"]["inputs"][
        field_name
    ]

    assert isinstance(template_value, list)
    assert template_value == field_contract["canonical_inline_example"]
    assert field_contract["accepted_artifact_semantic_types"] == [semantic_type]
    assert "workspace-relative path" not in str(template_value)


@pytest.mark.parametrize(
    ("action_id", "section", "field_name", "expected_bounds"),
    [
        (
            "calculate_excited_states",
            "action_settings",
            "number_of_states",
            {"minimum": 1, "maximum": 100},
        ),
        (
            "scan_reaction_coordinates",
            "action_settings",
            "steps",
            {"minimum": 1, "maximum": 1000},
        ),
        (
            "analyze_free_energy_selectivity",
            "action_settings",
            "temperature_kelvin",
            {"exclusive_minimum": 0.0},
        ),
        (
            "analyze_energetic_span",
            "inputs",
            "overall_reaction_free_energy",
            {"exclusive_maximum": 0.0},
        ),
        (
            "analyze_energetic_span",
            "action_settings",
            "temperature_kelvin",
            {"exclusive_minimum": 0.0},
        ),
    ],
)
def test_compact_contract_preserves_scalar_bounds(
    action_id: str,
    section: str,
    field_name: str,
    expected_bounds: dict[str, float | int],
) -> None:
    """紧凑合同必须保留完整字段合同中的标量边界及其包含关系。"""

    backend_id = (
        "gaussian"
        if action_id in {"calculate_excited_states", "scan_reaction_coordinates"}
        else "internal_thermochemistry"
    )
    contract = _selected_contract(action_id, backend_id)
    field_contract = _required_field(contract, section, field_name)

    assert {
        key: field_contract.get(key)
        for key in expected_bounds
    } == expected_bounds
