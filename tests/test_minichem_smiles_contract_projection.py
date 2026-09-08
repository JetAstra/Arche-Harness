"""验证 MiniChem 分子输入合同与 Planner 投影没有丢失裸 SMILES 能力。"""

from __future__ import annotations

from minichem_toolbox.backends.common import structure_dict
from minichem_toolbox.discovery import inspect_action
from minichem_toolbox.service import (
    _materialize_structure_batch_handoff,
    execute_action,
)
from third_party.minichem_toolbox.tests.test_reactive_complex_backend import _request

from chemistry_multiagent.utils.workflow_contract import (
    project_minichem_request_contracts,
    project_minichem_stage_tool_projection,
    _validate_minichem_batch_handoff_contract,
    _validate_minichem_binding_source_contract,
    resolve_minichem_step_request,
)
from tests.test_workflow_contract_minichem_only import (
    _real_frozen_catalog_with_action_contracts,
)


def _catalog_contract_view(catalog: dict) -> dict:
    """构造当前 V2 使用的 Action 与 pair 合同视图。"""

    return {
        "actions": project_minichem_stage_tool_projection(catalog),
        "planner_contract_projection": {
            "pairs": project_minichem_request_contracts(catalog),
        },
    }


def test_rdkit_molecule_input_accepts_direct_smiles_without_converter() -> None:
    """结构解析器将裸 SMILES 保留为分子输入，不要求虚构的中间转换 Action。"""

    assert structure_dict("CCO") == {"smiles": "CCO"}
    catalog = _real_frozen_catalog_with_action_contracts()

    for backend_id in ("rdkit_etkdg", "crest"):
        contract = inspect_action(
            "generate_conformer_ensemble",
            backend_id=backend_id,
            detail_level="contract",
            snapshot=catalog,
        )["selected_request_contract"]
        assert "SMILES" in contract["input_description"]
        molecule = next(
            item
            for item in contract["required_contract"]["inputs"]
            if item["name"] == "molecule"
        )
        assert "SMILES string" in molecule["accepted_forms"]
        assert molecule["canonical_inline_example"]["atoms"][0]["element"] == "H"


def test_planner_projection_explains_backend_specific_smiles_support() -> None:
    """紧凑 Planner 合同保留裸 SMILES 支持及 CREST 的坐标限制。"""

    catalog = _real_frozen_catalog_with_action_contracts()
    view = _catalog_contract_view(catalog)
    pairs = [
        item
        for item in view["planner_contract_projection"]["pairs"]
        if item["action_id"] == "generate_conformer_ensemble"
    ]

    assert {item["backend_id"] for item in pairs} == {"rdkit_etkdg", "crest"}
    for pair in pairs:
        note = pair["output_contract"]["input_handoff_note"]
        assert "SMILES" in note
        assert "CREST" in note
        assert "不能建立金属配位或离子接触" in note
        assert "build_reactive_complex_ensemble" in note


def test_reactive_selector_contract_names_the_aggregate_input() -> None:
    """ReactiveComplex 选择器不得被通用 ensemble 字段误报成 Conformer 输入。"""

    catalog = _real_frozen_catalog_with_action_contracts()
    contract = inspect_action(
        "select_reactive_complex_candidate",
        backend_id="internal_reactive_complex",
        detail_level="contract",
        snapshot=catalog,
    )["selected_request_contract"]
    ensemble = contract["required_contract"]["inputs"][0]

    assert ensemble["type"] == "ReactiveComplexEnsemble | ArtifactRef"
    assert ensemble["accepted_artifact_semantic_types"] == [
        "ReactiveComplexEnsemble"
    ]
    assert "ConformerEnsemble" in ensemble["rejected_artifact_semantic_types"]
    assert ensemble["rejected_forms"] == [
        "ConformerEnsemble or AtomicStructure ArtifactRef",
        "path to a JSON summary; pass the structured mapping or typed artifact instead",
    ]

    view = _catalog_contract_view(catalog)
    pair = next(
        item
        for item in view["planner_contract_projection"]["pairs"]
        if item["action_id"] == "select_reactive_complex_candidate"
    )
    pair_input = next(
        item
        for item in pair["request_contract"]["required_contract"]["inputs"]
        if item["name"] == "ensemble"
    )
    assert pair_input["accepted_artifact_semantic_types"] == [
        "ReactiveComplexEnsemble"
    ]


def test_ensemble_results_expose_declared_foreach_sources() -> None:
    """真实逐项坐标结果公开后，两个聚合 Action 可作为 foreach 来源。"""

    catalog = _real_frozen_catalog_with_action_contracts()
    view = _catalog_contract_view(catalog)
    for action_id, backend_id in (
        ("build_reactive_complex_ensemble", "internal_reactive_complex"),
        ("generate_conformer_ensemble", "rdkit_etkdg"),
    ):
        pair = next(
            item
            for item in view["planner_contract_projection"]["pairs"]
            if item["action_id"] == action_id and item["backend_id"] == backend_id
        )
        action = next(
            item for item in view["actions"] if item["action_id"] == action_id
        )
        _validate_minichem_binding_source_contract(
            {"source_field": "/result/ensemble"},
            "s1",
            action,
            pair,
            "s2.foreach",
        )


def test_structure_batch_handoff_contract_matches_converter_foreach() -> None:
    """冻结合同的逐项字段与 validator 的 converter 目标保持一致。"""

    catalog = _real_frozen_catalog_with_action_contracts()
    for action_id, backend_id in (
        ("build_reactive_complex_ensemble", "internal_reactive_complex"),
        ("generate_conformer_ensemble", "rdkit_etkdg"),
    ):
        source_action = next(
            item for item in catalog["actions"] if item["id"] == action_id
        )
        source_pair = next(
            item
            for item in catalog["action_contracts"]["contracts"]
            if item["action_id"] == action_id and item["backend_id"] == backend_id
        )
        declaration = source_pair["output_contract"]["batch_handoff"]
        _validate_minichem_batch_handoff_contract(
            foreach=declaration,
            source_action=source_action,
            source_pair_contract=source_pair,
            target_action_id="xyz_to_gjf",
            label=f"{action_id}.batch_handoff",
        )


def test_rdkit_conformer_items_are_public_xyz_artifact_handoff(tmp_path, monkeypatch) -> None:
    """RDKit 逐构象结果应能直接交给已有 XYZ→Gaussian 转换器。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    response = execute_action(
        "generate_conformer_ensemble",
        {
            "backend_id": "rdkit_etkdg",
            "inputs": {"molecule": "CCO"},
            "action_settings": {
                "num_conformers": 2,
                "random_seed": 42,
                "prune_rms_threshold_angstrom": -1.0,
            },
            "resource_limits": {"cpu_cores": 1, "memory_mb": 1024, "gpu_count": 0},
        },
    )

    assert response["status"] == "success"
    assert response["output_artifacts"][0]["semantic_type"] == "ConformerEnsemble"
    items = response["result"]["ensemble"]
    assert [item["item_id"] for item in items] == [item["conformer_id"] for item in items]
    assert all(set(item["structure_artifact"]) == {"artifact_id"} for item in items)
    xyz_artifacts = {
        artifact["artifact_id"]: artifact
        for artifact in response["output_artifacts"]
        if artifact["semantic_type"] == "XYZFile"
    }
    assert len(xyz_artifacts) == len(items)
    assert all(
        item["structure_artifact"]["artifact_id"] in xyz_artifacts
        and (tmp_path / xyz_artifacts[item["structure_artifact"]["artifact_id"]]["path"]).is_file()
        for item in items
    )

    contract = inspect_action(
        "generate_conformer_ensemble",
        backend_id="rdkit_etkdg",
        detail_level="contract",
    )["selected_request_contract"]["output_contract"]
    assert contract["batch_handoff"] == {
        "source_field": "/result/ensemble",
        "item_field": "/structure_artifact",
        "item_id_field": "/item_id",
        "target_field": "/inputs/input_xyz_path",
        "downstream_action_id": "xyz_to_gjf",
        "max_items": 32,
    }
    view = _catalog_contract_view(_real_frozen_catalog_with_action_contracts())
    pair = next(
        item
        for item in view["planner_contract_projection"]["pairs"]
        if item["action_id"] == "generate_conformer_ensemble"
        and item["backend_id"] == "rdkit_etkdg"
    )
    assert pair["output_contract"]["batch_handoff"] == contract["batch_handoff"]


def test_reactive_result_foreach_resolves_each_xyz_ref_for_converter(tmp_path, monkeypatch) -> None:
    """foreach 只携带选中 item 的 ArtifactRef，交给现有 converter 请求。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    built = execute_action(
        "build_reactive_complex_ensemble",
        _request(candidate_count=2),
    )
    completed = {
        "build-step": {
            "raw_output": built,
            "output_artifacts": built["output_artifacts"],
        }
    }
    step = {
        "request": {
            "inputs": {"input_xyz_path": None, "route_section": "#p PM6 SP"},
            "method_spec": {},
            "action_settings": {},
            "resource_limits": {"cpu_cores": 1, "memory_mb": 1024, "gpu_count": 0},
            "bindings": [],
            "foreach": {
                "source_step_ref": "build-step",
                "source_field": "/result/ensemble",
                "item_field": "/structure_artifact",
                "item_id_field": "/item_id",
                "target_field": "/inputs/input_xyz_path",
                "max_items": 32,
            },
        }
    }

    resolved = resolve_minichem_step_request(step, completed)

    assert [item["item_id"] for item in resolved["_foreach_items"]] == [
        item["candidate_id"] for item in built["result"]["ensemble"]
    ]
    assert [
        item["request"]["inputs"]["input_xyz_path"]["artifact_id"]
        for item in resolved["_foreach_items"]
    ] == [
        item["structure_artifact"]["artifact_id"]
        for item in built["result"]["ensemble"]
    ]


def test_crest_multixyz_result_gets_item_records_without_running_crest(
    tmp_path, monkeypatch
) -> None:
    """已有 CREST 多帧文件可离线切分成逐构象交接项。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    (tmp_path / "crest_conformers.xyz").write_text(
        "2\nframe-0\nH 0.0 0.0 0.0\nH 0.0 0.0 0.7\n"
        "2\nframe-1\nH 0.0 0.0 0.0\nH 0.0 0.0 0.8\n",
        encoding="utf-8",
    )
    result = {"ensemble_file": "crest_conformers.xyz"}

    entries = _materialize_structure_batch_handoff(
        action_id="generate_conformer_ensemble",
        backend_id="crest",
        result_payload=result,
        artifact_files=[],
    )

    assert [item["item_id"] for item in result["ensemble"]] == ["0", "1"]
    assert [item["conformer_id"] for item in result["ensemble"]] == ["0", "1"]
    assert len(entries) == 2
    assert [item["result_pointer"] for item in entries] == [
        "/ensemble/0/structure_artifact",
        "/ensemble/1/structure_artifact",
    ]
    assert all((tmp_path / item["path"]).is_file() for item in entries)
