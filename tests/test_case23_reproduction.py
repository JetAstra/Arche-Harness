from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "tasks" / "case23_reproduce.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("reproduce_case23", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _payload(name: str):
    return json.loads(
        (REPOSITORY_ROOT / "supplementary" / f"{name}.json").read_text(
            encoding="utf-8"
        )
    )


def test_case2_all_published_candidates_reproduce_paper_photochemical_conclusion():
    module = _load_module()
    path = REPOSITORY_ROOT / "supplementary" / "case2.json"
    report = module.reproduce_case2(_payload("case2"), module._sha256(path))

    assert report["status"] == "passed_with_source_discrepancy"
    assert [row["barrier_kcal_mol"] for row in report["concerted_pathways"]] == pytest.approx(
        [45.995189429891234, 45.97950169300792]
    )
    assert all(
        row["imaginary_frequency_count"] == 1
        for row in report["concerted_pathways"]
    )
    evidence = report["excited_state_evidence"]
    assert {name: value["state_count"] for name, value in evidence.items()} == {
        "mechanism_E": 20,
        "mechanism_F": 20,
        "mechanism_G": 20,
    }
    assert evidence["mechanism_E"]["first_state"]["wavelength_nm"] == pytest.approx(
        278.0163982859057
    )
    assert evidence["mechanism_F"]["first_state"]["wavelength_nm"] == pytest.approx(
        377.1955977474369
    )
    assert evidence["mechanism_G"]["states"][0]["wavelength_nm"] == pytest.approx(
        649.4039020510293
    )
    assert evidence["mechanism_G"]["states"][0]["oscillator_strength"] == pytest.approx(
        0.0044
    )
    assert evidence["mechanism_G"]["states"][1]["wavelength_nm"] == pytest.approx(
        446.58067564594063
    )
    assert evidence["mechanism_G"]["states"][1]["oscillator_strength"] == pytest.approx(
        0.0567
    )
    assert report["complex_formation_free_energy_kcal_mol"] == pytest.approx(
        -4.617842219671111
    )
    assert report["paper_complex_formation_free_energy_kcal_mol"] == -3.0
    assert report["checks"] == {
        "source_sha256_matches": True,
        "both_concerted_paths_have_one_imaginary_frequency": True,
        "concerted_barriers_match_paper_46_kcal_mol": True,
        "all_three_photochemical_candidates_have_20_states": True,
        "mechanism_E_and_F_first_states_are_not_450_nm_matches": True,
        "mechanism_G_S2_matches_450_nm": True,
        "mechanism_G_S2_is_brighter_than_S1": True,
    }


def test_case3_all_published_profiles_and_bond_dissociation_trends_are_reproduced():
    module = _load_module()
    path = REPOSITORY_ROOT / "supplementary" / "case3.json"
    report = module.reproduce_case3(_payload("case3"), module._sha256(path))

    assert report["status"] == "passed_with_missing_published_wiberg_values"
    assert report["dihedral_ranking_by_maximum_penalty"] == [
        "ortho", "meta", "unsubstituted", "para"
    ]
    expected_penalties = {
        "meta": 21.94400534345171,
        "unsubstituted": 22.075782327280727,
        "ortho": 12.82629308631508,
        "para": 22.125983082545723,
    }
    assert {
        name: profile["maximum_penalty_kcal_mol"]
        for name, profile in report["dihedral_profiles"].items()
    } == pytest.approx(expected_penalties)
    assert all(
        profile["point_count"] == 13
        for profile in report["dihedral_profiles"].values()
    )
    assert report["bite_angle_lowest_penalty_ligand"] == {
        "add": "ortho",
        "reduce": "ortho",
    }
    assert all(
        profile["point_count"] == 6
        for direction in report["bite_angle_profiles"].values()
        for profile in direction.values()
    )
    bde = report["bond_dissociation"]
    assert bde["ortho"]["electronic_bde_kcal_mol"] == pytest.approx(
        24.027545917040268
    )
    assert bde["ortho"]["gibbs_bond_dissociation_free_energy_kcal_mol"] == pytest.approx(
        10.06525196424205
    )
    assert report["lowest_electronic_bde_ligand"] == "ortho"
    assert report["lowest_gibbs_bdfe_ligand"] == "ortho"
    assert all(report["checks"].values())


def test_case3_action_requests_are_derived_from_published_geometry_not_case_parameters():
    module = _load_module()
    payload = _payload("case3")
    request = module._dihedral_request(payload, "ortho")

    settings = request["action_settings"]
    assert settings["coordinate_type"] == "dihedral"
    assert settings["atom_indices"] == [18, 2, 7, 17]
    assert settings["steps"] == 12
    assert settings["start_value"] == pytest.approx(-4.04049)
    assert settings["end_value"] == pytest.approx(55.95885)
    assert request["method_spec"] == {
        "method": "wB97XD",
        "basis": "def2SVP",
        "solvation_model": "smd",
        "solvent": "N,N-Dimethylacetamide",
    }

    structure = module._atomic_structure(payload["bde"]["opt"]["ortho"])
    nickel, hydrogen, distance = module._closest_ni_h_pair(structure)
    assert structure["atoms"][nickel]["element"] == "Ni"
    assert structure["atoms"][hydrogen]["element"] == "H"
    assert distance < 2.0


def test_case2_real_td_requests_cover_all_published_photochemical_candidates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    module = _load_module()
    requests = []

    def fake_execute(action_id, request):
        requests.append((action_id, request))
        return {"status": "success", "result": {"completed_states": 20}}

    monkeypatch.setitem(__import__("sys").modules, "minichem_toolbox.service", type(
        "FakeService", (), {"execute_action": staticmethod(fake_execute)}
    ))
    results = module.run_real_actions(
        _payload("case2"),
        _payload("case3"),
        {"td"},
        cpu_cores=2,
        memory_mb=1024,
        workspace=tmp_path,
    )

    assert set(results["case2_vertical_excited_states"]) == {
        "mechanism_E", "mechanism_F", "mechanism_G"
    }
    assert [action_id for action_id, _request in requests] == [
        "calculate_excited_states",
        "calculate_excited_states",
        "calculate_excited_states",
    ]
    assert [request["inputs"]["structure"]["charge"] for _action, request in requests] == [
        0, -1, 0
    ]
    assert [len(request["inputs"]["structure"]["atoms"]) for _action, request in requests] == [
        41, 64, 65
    ]
    for _action, request in requests:
        assert request["method_spec"] == {
            "method": "B3LYP",
            "basis": "def2SVP",
            "excited_state_method": "tddft",
            "dispersion": "gd3bj",
            "solvation_model": "smd",
            "solvent": "EthylEthanoate",
            "integration_grid": "UltraFine",
        }
        assert request["action_settings"]["number_of_states"] == 20


def test_real_action_status_collection_handles_direct_and_wrapped_results():
    module = _load_module()
    payload = {
        "case2_vertical_excited_states": {
            "mechanism_E": {"status": "success", "result": {}},
            "mechanism_F": {"status": "partial_success", "result": {}},
        },
        "case3_ni_h_wiberg": {
            "ortho": {"action_result": {"status": "success", "result": {}}},
        },
    }
    assert module._collect_action_statuses(payload) == [
        "success", "partial_success", "success"
    ]
    assert module._collect_action_statuses({"no_action_result": {}}) == []


def test_case3_real_scan_requests_cover_selected_ligands_with_one_generic_action(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    module = _load_module()
    requests = []

    def fake_execute(action_id, request):
        requests.append((action_id, request))
        return {"status": "success", "result": {"completed_points": 13}}

    monkeypatch.setitem(__import__("sys").modules, "minichem_toolbox.service", type(
        "FakeService", (), {"execute_action": staticmethod(fake_execute)}
    ))
    results = module.run_real_actions(
        _payload("case2"),
        _payload("case3"),
        {"scan"},
        cpu_cores=2,
        memory_mb=1024,
        workspace=tmp_path,
        scan_ligands=("meta", "ortho"),
    )

    assert set(results["case3_dihedral_scans"]) == {"meta", "ortho"}
    assert [action for action, _request in requests] == [
        "scan_reaction_coordinates", "scan_reaction_coordinates"
    ]
    for _action, request in requests:
        assert request["action_settings"]["coordinate_type"] == "dihedral"
        assert request["action_settings"]["steps"] == 12
        assert request["method_spec"]["method"] == "wB97XD"


def test_case3_real_bde_reuses_energy_action_and_python_difference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    module = _load_module()
    requests = []
    artificial_energies = {
        "meta": -100.0,
        "unsubstituted": -200.0,
        "ortho": -300.0,
        "para": -400.0,
        "meta-H": -99.4,
        "unsubstituted-H": -199.4,
        "ortho-H": -299.5,
        "para-H": -399.4,
        "H": -0.5,
    }
    call_order = iter(artificial_energies)

    def fake_execute(action_id, request):
        species = next(call_order)
        requests.append((species, action_id, request))
        return {
            "status": "success",
            "result": {
                "energy": artificial_energies[species],
                "unit": "hartree",
            },
        }

    monkeypatch.setitem(__import__("sys").modules, "minichem_toolbox.service", type(
        "FakeService", (), {"execute_action": staticmethod(fake_execute)}
    ))
    results = module.run_real_actions(
        _payload("case2"),
        _payload("case3"),
        {"bde"},
        cpu_cores=2,
        memory_mb=1024,
        workspace=tmp_path,
    )

    assert len(requests) == 9
    assert all(action_id == "calculate_energy" for _species, action_id, _ in requests)
    assert requests[-1][2]["inputs"]["structure"]["multiplicity"] == 2
    assert all(
        request["method_spec"]["method"] == "M062X"
        for _species, _action, request in requests
    )
    bde = results["case3_bde_from_real_energies"]
    assert bde["meta"]["electronic_bde_kcal_mol"] == pytest.approx(
        0.1 * module.HARTREE_TO_KCAL_MOL
    )
    assert bde["ortho"]["electronic_bde_kcal_mol"] == pytest.approx(0.0)


def test_case3_bde_rejects_non_hartree_energy_results():
    module = _load_module()
    results = {
        species: {"status": "success", "result": {"energy": -1.0, "unit": "eV"}}
        for species in (*module.LIGANDS, *(f"{ligand}-H" for ligand in module.LIGANDS), "H")
    }

    with pytest.raises(ValueError, match="did not return hartree"):
        module._bde_from_action_results(results)


def test_case3_bite_angle_requests_reuse_generic_angle_scan_action(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    module = _load_module()
    requests = []

    def fake_execute(action_id, request):
        requests.append((action_id, request))
        return {"status": "success", "result": {"completed_points": 6}}

    monkeypatch.setitem(__import__("sys").modules, "minichem_toolbox.service", type(
        "FakeService", (), {"execute_action": staticmethod(fake_execute)}
    ))
    results = module.run_real_actions(
        _payload("case2"),
        _payload("case3"),
        {"bite"},
        cpu_cores=2,
        memory_mb=1024,
        workspace=tmp_path,
        scan_ligands=("ortho",),
    )

    assert set(results["case3_bite_angle_scans"]) == {"add", "reduce"}
    assert len(requests) == 2
    add = requests[0][1]
    reduce = requests[1][1]
    for action_id, request in requests:
        assert action_id == "scan_reaction_coordinates"
        assert request["action_settings"]["coordinate_type"] == "angle"
        assert request["action_settings"]["atom_indices"] == [2, 16, 7]
        assert request["action_settings"]["steps"] == 5
        assert request["action_settings"]["disable_symmetry"] is False
        assert request["method_spec"] == {
            "method": "B3LYP",
            "basis": "def2TZVP",
            "dispersion": "gd3bj",
            "solvation_model": "smd",
            "solvent": "N,N-Dimethylacetamide",
        }
    assert add["action_settings"]["end_value"] > add["action_settings"]["start_value"]
    assert reduce["action_settings"]["end_value"] < reduce["action_settings"]["start_value"]
