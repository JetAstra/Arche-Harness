from __future__ import annotations

import copy
import math

import pytest

from minichem_mcp_tools.result_transport import compact_action_result
from minichem_toolbox.backends import execute_local
from minichem_toolbox.backends import reactive_complex
from minichem_toolbox.service import execute_action


def _distance(first, second):
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(first, second)))


def _request(*, target=2.0, minimum_distance=0.5, candidate_count=2):
    return {
        "inputs": {
            "components": [
                {
                    "atoms": [
                        {
                            "element": "C",
                            "position_angstrom": [0.0, 0.0, 0.0],
                            "atom_map_number": 1,
                            "label": "anchor",
                        },
                        {
                            "element": "H",
                            "position_angstrom": [1.0, 0.0, 0.0],
                            "atom_map_number": 2,
                        },
                    ],
                    "bonds": [
                        {"atom_index_a": 0, "atom_index_b": 1, "bond_order": 1.0}
                    ],
                    "charge": 0,
                    "multiplicity": 1,
                    "pbc": [False, False, False],
                    "component_label": "electrophile",
                },
                {
                    "atoms": [
                        {
                            "element": "O",
                            "position_angstrom": [0.0, 0.0, 0.0],
                            "atom_map_number": 3,
                        },
                        {
                            "element": "H",
                            "position_angstrom": [0.0, 1.0, 0.0],
                            "atom_map_number": 4,
                        },
                    ],
                    "bonds": [
                        {"atom_index_a": 0, "atom_index_b": 1, "bond_order": 1.0}
                    ],
                    "charge": -1,
                    "multiplicity": 1,
                    "pbc": [False, False, False],
                    "component_label": "nucleophile",
                },
            ],
            "contact_pairs": [
                {
                    "component_a": 0,
                    "atom_a": 0,
                    "component_b": 1,
                    "atom_b": 0,
                    "target_distance_angstrom": target,
                    "tolerance_angstrom": 1.0e-8,
                }
            ],
        },
        "method_spec": {},
        "action_settings": {
            "random_seed": 20260811,
            "candidate_count": candidate_count,
            "max_candidates": candidate_count,
            "orientation_trials": 8,
            "contact_tolerance_angstrom": 0.1,
            "minimum_interatomic_distance_angstrom": minimum_distance,
        },
    }


def test_seeded_assembly_is_reproducible_and_preserves_component_data():
    request = _request()
    original = copy.deepcopy(request)

    first = reactive_complex.execute(
        "build_reactive_complex_ensemble", "internal_reactive_complex", request
    )
    second = reactive_complex.execute(
        "build_reactive_complex_ensemble", "internal_reactive_complex", request
    )

    assert first == second
    assert request == original
    assert first["status"] == "success"
    assert first["result"]["accepted_count"] == 2
    assert first["result"]["state_diagnostics"]["charge"] == -1
    assert first["result"]["state_diagnostics"]["multiplicity"] == 1
    assert first["result"]["graph_edits_applied"] is False
    assert first["result"]["conformers_generated"] is False
    assert first["result"]["geometry_optimization_performed"] is False
    assert "contact_validation" not in first["result"]["assembly_settings"]

    candidate = first["result"]["ensemble"][0]
    structure = candidate["structure"]
    assert structure["atoms"][0]["position_angstrom"] == [0.0, 0.0, 0.0]
    assert [atom["atom_map_number"] for atom in structure["atoms"]] == [1, 2, 3, 4]
    assert structure["atoms"][0]["label"] == "anchor"
    assert candidate["components"][0]["component_label"] == "electrophile"
    assert candidate["components"][1]["component_label"] == "nucleophile"
    assert structure["bonds"] == [
        {"atom_index_a": 0, "atom_index_b": 1, "bond_order": 1.0},
        {"atom_index_a": 2, "atom_index_b": 3, "bond_order": 1.0},
    ]
    assert len(structure["bonds"]) == 2  # The contact was not converted into a bond.

    moving_atoms = candidate["components"][1]["atoms"]
    assert _distance(
        moving_atoms[0]["position_angstrom"], moving_atoms[1]["position_angstrom"]
    ) == pytest.approx(1.0, abs=1.0e-12)
    contact = candidate["contact_diagnostics"][0]
    assert contact["actual_distance_angstrom"] == pytest.approx(2.0, abs=1.0e-12)
    assert contact["within_tolerance"] is True


def test_contact_validation_setting_is_rejected_as_unsupported():
    """失效的 contact_validation 设置必须明确拒绝，不得被静默接受。"""

    request = _request(candidate_count=1)
    request["action_settings"]["contact_validation"] = False

    with pytest.raises(ValueError, match="contact_validation.*not supported"):
        reactive_complex.execute(
            "build_reactive_complex_ensemble", "internal_reactive_complex", request
        )


def test_all_rejected_candidates_fail_without_a_consumable_ensemble():
    result = reactive_complex.execute(
        "build_reactive_complex_ensemble",
        "internal_reactive_complex",
        _request(target=0.4, minimum_distance=0.8, candidate_count=1),
    )

    assert result["status"] == "failed"
    assert result["error"]["code"] == "no_accepted_reactive_complex_candidates"
    assert result["error"]["failure_stage"] == "candidate_acceptance"
    assert result["error"]["attempted_count"] == 1
    assert result["error"]["contact_violation_count"] == 0
    assert result["error"]["collision_candidate_count"] == 1
    assert result["error"]["maximum_collision_count"] >= 1
    assert result["error"]["repair_guidance"]
    assert "result" not in result
    assert "ensemble" not in result["diagnostics"]
    assert result["diagnostics"]["accepted_count"] == 0
    assert result["diagnostics"]["rejected_count"] == 1
    assert result["diagnostics"]["rejected_candidates"][0]["accepted"] is False
    attempt = result["diagnostics"]["attempts"][0]
    assert attempt["contact_diagnostics"][0]["within_tolerance"] is True
    assert attempt["collision_diagnostics"]["collision_count"] >= 1
    assert attempt["accepted"] is False
    assert result["warnings"]


def test_service_preserves_reactive_complex_rejection_summary(monkeypatch, tmp_path):
    """公开 Action 失败应保留可供 Revision 调参的聚合几何诊断。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    result = execute_action(
        "build_reactive_complex_ensemble",
        _request(target=0.4, minimum_distance=0.8, candidate_count=1),
    )

    assert result["status"] == "failed"
    assert result["error"]["failure_stage"] == "candidate_acceptance"
    assert result["error"]["contact_violation_count"] == 0
    assert result["error"]["collision_candidate_count"] == 1
    assert result["error"]["repair_guidance"]

    compact = compact_action_result(result)
    assert compact["error"]["contact_violation_count"] == 0
    assert compact["error"]["collision_candidate_count"] == 1
    assert compact["error"]["repair_guidance"]


def test_mixed_candidates_keep_only_accepted_values_in_primary_ensemble(monkeypatch):
    def fake_candidate(candidate_index, candidate_seed, *_args):
        accepted = candidate_index == 0
        return {
            "candidate_id": f"candidate_{candidate_index + 1:04d}",
            "candidate_seed": candidate_seed,
            "accepted": accepted,
            "structure": {"atoms": []},
            "components": [],
            "component_atom_ranges": [],
            "component_transforms": [],
            "contact_diagnostics": [{"within_tolerance": accepted}],
            "collision_diagnostics": {"collision_count": 0 if accepted else 1},
        }

    monkeypatch.setattr(reactive_complex, "_build_candidate", fake_candidate)
    result = reactive_complex.execute(
        "build_reactive_complex_ensemble",
        "internal_reactive_complex",
        _request(candidate_count=2),
    )

    assert result["status"] == "partial_success"
    assert result["result"]["count"] == 1
    assert result["result"]["attempted_count"] == 2
    assert result["result"]["accepted_count"] == 1
    assert result["result"]["rejected_count"] == 1
    assert [item["accepted"] for item in result["result"]["ensemble"]] == [True]
    assert [item["accepted"] for item in result["result"]["rejected_candidates"]] == [False]
    assert [item["accepted"] for item in result["result"]["attempts"]] == [True, False]


@pytest.mark.parametrize("missing_field", ["charge", "multiplicity"])
def test_component_electronic_state_must_be_explicit(missing_field):
    request = _request(candidate_count=1)
    del request["inputs"]["components"][1][missing_field]

    with pytest.raises(ValueError, match=rf"requires an explicit {missing_field}"):
        reactive_complex.execute(
            "build_reactive_complex_ensemble", "internal_reactive_complex", request
        )


def test_total_atom_and_candidate_output_budgets_are_enforced(monkeypatch):
    total_atom_request = _request(candidate_count=1)
    monkeypatch.setattr(reactive_complex, "HARD_MAX_TOTAL_ATOMS", 3)
    with pytest.raises(ValueError, match="hard total-atom limit"):
        reactive_complex.execute(
            "build_reactive_complex_ensemble", "internal_reactive_complex", total_atom_request
        )

    monkeypatch.setattr(reactive_complex, "HARD_MAX_TOTAL_ATOMS", 4)
    monkeypatch.setattr(
        reactive_complex, "HARD_MAX_CANDIDATE_OUTPUT_ATOM_RECORDS", 15
    )
    output_request = _request(candidate_count=2)
    with pytest.raises(ValueError, match="hard candidate-output atom-record budget"):
        reactive_complex.execute(
            "build_reactive_complex_ensemble", "internal_reactive_complex", output_request
        )


def test_hard_candidate_cap_and_connected_contact_graph_are_enforced():
    request = _request(candidate_count=1)
    request["action_settings"]["max_candidates"] = reactive_complex.HARD_MAX_CANDIDATES + 1
    with pytest.raises(ValueError, match="hard limit"):
        reactive_complex.execute(
            "build_reactive_complex_ensemble", "internal_reactive_complex", request
        )

    disconnected = _request(candidate_count=1)
    disconnected["inputs"]["components"].append(
        {
            "atoms": [
                {
                    "element": "He",
                    "position_angstrom": [0.0, 0.0, 0.0],
                    "atom_map_number": 5,
                }
            ],
            "charge": 0,
            "multiplicity": 1,
        }
    )
    with pytest.raises(ValueError, match="disconnected"):
        reactive_complex.execute(
            "build_reactive_complex_ensemble", "internal_reactive_complex", disconnected
        )


def test_duplicate_atom_maps_and_ambiguous_spin_coupling_are_rejected():
    duplicate = _request(candidate_count=1)
    duplicate["inputs"]["components"][1]["atoms"][0]["atom_map_number"] = 1
    with pytest.raises(ValueError, match="duplicated"):
        reactive_complex.execute(
            "build_reactive_complex_ensemble", "internal_reactive_complex", duplicate
        )

    spin = _request(candidate_count=1)
    spin["inputs"]["components"][0]["multiplicity"] = 2
    spin["inputs"]["components"][1]["multiplicity"] = 2
    with pytest.raises(ValueError, match="Multiple total-spin couplings"):
        reactive_complex.execute(
            "build_reactive_complex_ensemble", "internal_reactive_complex", spin
        )
    spin["action_settings"]["complex_multiplicity"] = 1
    result = reactive_complex.execute(
        "build_reactive_complex_ensemble", "internal_reactive_complex", spin
    )
    assert result["result"]["state_diagnostics"]["multiplicity"] == 1


def test_unknown_action_backend_pair_is_unsupported():
    result = reactive_complex.execute("other_action", "internal_reactive_complex", {})
    assert result["status"] == "unsupported"
    assert result["error"]["code"] == "unsupported_request"


def test_build_mapped_distance_constraints_uses_explicit_maps_and_ratios():
    structure = {
        "atoms": [
            {
                "element": "C",
                "position_angstrom": [0.0, 0.0, 0.0],
                "atom_map_number": 3,
            },
            {
                "element": "H",
                "position_angstrom": [2.0, 0.0, 0.0],
                "atom_map_number": 26,
            },
        ],
        "bonds": [],
        "charge": 0,
        "multiplicity": 1,
        "pbc": [False, False, False],
    }
    result = execute_local(
        "build_mapped_distance_constraints",
        "internal_reactive_complex",
        {
            "inputs": {
                "structure": structure,
                "mapped_bonds": [
                    {"atom_map_a": 3, "atom_map_b": 26, "bond_order": 0.0}
                ],
            },
            "method_spec": {},
            "action_settings": {
                "equilibrium_distance_scale": 1.6,
                "minimum_distance_ratio": 0.8,
                "maximum_distance_ratio": 1.3,
                "force_constant_kcal_mol_angstrom2": 5.0,
            },
        },
    )

    assert result["status"] == "success"
    constraint = result["result"]["constraints"][0]
    assert constraint["atom_indices"] == [0, 1]
    assert constraint["atom_maps"] == [3, 26]
    assert constraint["equilibrium_distance_angstrom"] == pytest.approx(1.09 * 1.6)
    assert constraint["minimum_distance_angstrom"] == pytest.approx(1.09 * 1.6 * 0.8)
    assert constraint["maximum_distance_angstrom"] == pytest.approx(1.09 * 1.6 * 1.3)
    assert constraint["force_constant_kcal_mol_angstrom2"] == 5.0


def test_select_candidate_requires_explicit_accepted_id():
    built = reactive_complex.execute(
        "build_reactive_complex_ensemble",
        "internal_reactive_complex",
        _request(candidate_count=2),
    )
    assert built["status"] == "success"
    candidate_id = built["result"]["ensemble"][1]["candidate_id"]

    selected = reactive_complex.execute(
        "select_reactive_complex_candidate",
        "internal_reactive_complex",
        {
            "inputs": {"ensemble": built["result"]},
            "method_spec": {},
            "action_settings": {"candidate_id": candidate_id},
        },
    )

    assert selected["status"] == "success"
    assert selected["result"]["candidate_id"] == candidate_id
    assert selected["result"]["selection_policy"] == "explicit_candidate_id"
    assert selected["result"]["ranking_performed"] is False
    assert selected["result"]["structure"] == built["result"]["ensemble"][1]["structure"]

    routed = execute_local(
        "select_reactive_complex_candidate",
        "internal_reactive_complex",
        {
            "inputs": {"ensemble": built["result"]},
            "method_spec": {},
            "action_settings": {"candidate_id": candidate_id},
        },
    )
    assert routed["status"] == "success"
    assert routed["result"]["candidate_id"] == candidate_id

    with pytest.raises(ValueError, match="available accepted IDs"):
        reactive_complex.execute(
            "select_reactive_complex_candidate",
            "internal_reactive_complex",
            {
                "inputs": {"ensemble": built["result"]},
                "method_spec": {},
                "action_settings": {"candidate_id": "candidate_9999"},
            },
        )


def test_service_routes_reactive_complex_artifact_to_explicit_selector(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))

    built = execute_action("build_reactive_complex_ensemble", _request(candidate_count=1))
    assert built["status"] == "success"
    assert built["selection_source"] == "internal_deterministic"
    ensemble_artifact = built["output_artifacts"][0]
    assert ensemble_artifact["semantic_type"] == "ReactiveComplexEnsemble"

    candidate_id = built["result"]["ensemble"][0]["candidate_id"]
    selected = execute_action(
        "select_reactive_complex_candidate",
        {
            "inputs": {"ensemble": ensemble_artifact},
            "action_settings": {"candidate_id": candidate_id},
        },
    )

    assert selected["status"] == "success"
    assert selected["selection_source"] == "internal_deterministic"
    assert selected["result"]["candidate_id"] == candidate_id
    structure_artifact = selected["output_artifacts"][0]
    assert structure_artifact["semantic_type"] == "AtomicStructure"
    assert structure_artifact["parent_artifact_ids"] == [
        ensemble_artifact["artifact_id"]
    ]


def test_service_exposes_each_reactive_candidate_as_xyz_artifact(
    monkeypatch, tmp_path
):
    """聚合 ReactiveComplex 保持不变，同时登记每个 accepted candidate 的坐标文件。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    built = execute_action("build_reactive_complex_ensemble", _request(candidate_count=2))

    assert built["status"] == "success"
    items = built["result"]["ensemble"]
    assert [item["item_id"] for item in items] == [
        item["candidate_id"] for item in items
    ]
    assert all(set(item["structure_artifact"]) == {"artifact_id"} for item in items)
    xyz_artifacts = {
        artifact["artifact_id"]: artifact
        for artifact in built["output_artifacts"]
        if artifact["semantic_type"] == "XYZFile"
    }
    assert len(xyz_artifacts) == len(items)
    assert all(
        item["structure_artifact"]["artifact_id"] in xyz_artifacts
        and (
            tmp_path
            / xyz_artifacts[item["structure_artifact"]["artifact_id"]]["path"]
        ).is_file()
        for item in items
    )
    converted = [
        execute_action(
            "xyz_to_gjf",
            {
                "inputs": {
                    "input_xyz_path": item["structure_artifact"],
                    "route_section": "#p PM6 SP",
                },
                "resource_limits": {
                    "cpu_cores": 1,
                    "memory_mb": 1024,
                    "gpu_count": 0,
                },
            },
        )
        for item in items
    ]
    assert all(item["status"] == "success" for item in converted)
    assert all(
        item["output_artifacts"][0]["semantic_type"] == "GaussianInputDeck"
        for item in converted
    )
    assert [
        item["output_artifacts"][0]["parent_artifact_ids"][0]
        for item in converted
    ] == [item["structure_artifact"]["artifact_id"] for item in items]
