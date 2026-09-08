#!/usr/bin/env python3
"""重新分析 ARCHE Case 2/3 证据，并可选运行类型化 Gaussian Action。

默认分支读取发布数据并写复算报告；只有调用方显式选择真实 Action 时才提交计算。
本脚本依赖 Case 2/3 冻结输入，不是通用端到端入口。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

HARTREE_TO_KCAL_MOL = 627.5094740631
EV_TO_KCAL_MOL = 23.06054783061903
EV_TO_WAVENUMBER_CM1 = 8065.544005
EXPECTED_CASE2_SHA256 = "fa4af5aec29ab5d99d4b419216c8672e0062355d476ffe3e0150e5d526577722"
EXPECTED_CASE3_SHA256 = "718959095618dd644effbe09478d9dcf3ea3d7225320159015ef24e6ec21b95e"
PAPER_CASE2_CONCERTED_BARRIER_KCAL_MOL = 46.0
PAPER_CASE2_COMPLEX_FORMATION_KCAL_MOL = -3.0
PAPER_CASE2_IRRADIATION_NM = 450.0
CASE2_EXCITED_RECORDS = {
    "mechanism_E": ("hypothesis_E", "cp1_excited"),
    "mechanism_F": ("hypothesis_F", "cp4_excited"),
    "mechanism_G": ("hypothesis_G", "cp5_excited"),
}
LIGANDS = ("meta", "unsubstituted", "ortho", "para")
ELEMENTS = (
    "", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg",
    "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr",
    "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe", "Cs",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    return os.path.relpath(path.resolve(), start=Path.cwd().resolve())


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _excited_states(record: dict[str, Any]) -> list[dict[str, Any]]:
    energies = record.get("etenergies")
    oscillator_strengths = record.get("etoscs")
    symmetries = record.get("etsyms")
    if not all(isinstance(values, list) for values in (energies, oscillator_strengths, symmetries)):
        raise ValueError(f"{record.get('filename')} has no complete excited-state arrays")
    if not (len(energies) == len(oscillator_strengths) == len(symmetries)):
        raise ValueError(f"{record.get('filename')} excited-state arrays have different lengths")
    states = []
    for index, (wavenumber, oscillator, symmetry) in enumerate(
        zip(energies, oscillator_strengths, symmetries, strict=True), 1
    ):
        wavenumber_cm1 = _finite(wavenumber, "excitation wavenumber")
        if wavenumber_cm1 <= 0:
            raise ValueError("excitation wavenumbers must be positive")
        energy_ev = wavenumber_cm1 / EV_TO_WAVENUMBER_CM1
        states.append(
            {
                "state_index": index,
                "state_label": str(symmetry),
                "wavenumber_cm1": wavenumber_cm1,
                "energy_ev": energy_ev,
                "energy_kcal_mol": energy_ev * EV_TO_KCAL_MOL,
                "wavelength_nm": 1.0e7 / wavenumber_cm1,
                "oscillator_strength": _finite(oscillator, "oscillator strength"),
            }
        )
    return states


def reproduce_case2(payload: dict[str, Any], source_sha256: str) -> dict[str, Any]:
    concerted = []
    for stereochemistry in ("R", "S"):
        group = payload[f"Concerted_{stereochemistry}"]
        transition_state = group[f"ts_Concerted_{stereochemistry}"]
        reactant_free_energy = sum(
            _finite(group[name]["freeenergy"], f"{name} freeenergy")
            for name in ("r0_BC15H22IO2", "r1_C12H11P")
        )
        product_free_energy = sum(
            _finite(group[name]["freeenergy"], f"{name} freeenergy")
            for name in ("p0_BC15H23O2", "p1_C12H10IP")
        )
        frequencies = [_finite(value, "transition-state frequency") for value in transition_state["vibfreqs"]]
        concerted.append(
            {
                "path": stereochemistry,
                "barrier_kcal_mol": (
                    _finite(transition_state["freeenergy"], "transition-state freeenergy")
                    - reactant_free_energy
                )
                * HARTREE_TO_KCAL_MOL,
                "reaction_free_energy_kcal_mol": (
                    product_free_energy - reactant_free_energy
                )
                * HARTREE_TO_KCAL_MOL,
                "imaginary_frequency_count": sum(value < 0 for value in frequencies),
                "lowest_frequency_cm1": min(frequencies),
                "source_record": f"Concerted_{stereochemistry}.ts_Concerted_{stereochemistry}",
            }
        )

    excited_state_evidence: dict[str, Any] = {}
    for mechanism, (group_name, record_name) in CASE2_EXCITED_RECORDS.items():
        states = _excited_states(payload[group_name][record_name])
        closest_to_450 = min(
            states, key=lambda state: abs(state["wavelength_nm"] - PAPER_CASE2_IRRADIATION_NM)
        )
        excited_state_evidence[mechanism] = {
            "source_record": f"{group_name}.{record_name}",
            "state_count": len(states),
            "states": states,
            "first_state": states[0],
            "closest_state_to_450_nm": closest_to_450,
        }

    mechanism_g = excited_state_evidence["mechanism_G"]
    mechanism_g_s1 = mechanism_g["states"][0]
    mechanism_g_s2 = mechanism_g["states"][1]
    group_g = payload["hypothesis_G"]
    complex_formation = (
        _finite(group_g["cp5"]["freeenergy"], "cp5 freeenergy")
        - _finite(payload["hypothesis_E"]["cp1"]["freeenergy"], "cp1 freeenergy")
        - _finite(group_g["CsPPh2"]["freeenergy"], "CsPPh2 freeenergy")
    ) * HARTREE_TO_KCAL_MOL
    maximum_concerted_deviation = max(
        abs(row["barrier_kcal_mol"] - PAPER_CASE2_CONCERTED_BARRIER_KCAL_MOL)
        for row in concerted
    )
    checks = {
        "source_sha256_matches": source_sha256 == EXPECTED_CASE2_SHA256,
        "both_concerted_paths_have_one_imaginary_frequency": all(
            row["imaginary_frequency_count"] == 1 for row in concerted
        ),
        "concerted_barriers_match_paper_46_kcal_mol": maximum_concerted_deviation < 0.05,
        "all_three_photochemical_candidates_have_20_states": all(
            evidence["state_count"] == 20 for evidence in excited_state_evidence.values()
        ),
        "mechanism_E_and_F_first_states_are_not_450_nm_matches": all(
            abs(excited_state_evidence[name]["first_state"]["wavelength_nm"] - 450.0) > 50.0
            for name in ("mechanism_E", "mechanism_F")
        ),
        "mechanism_G_S2_matches_450_nm": abs(mechanism_g_s2["wavelength_nm"] - 450.0) < 5.0,
        "mechanism_G_S2_is_brighter_than_S1": (
            mechanism_g_s2["oscillator_strength"] > mechanism_g_s1["oscillator_strength"]
        ),
    }
    return {
        "status": "passed_with_source_discrepancy" if all(checks.values()) else "failed",
        "concerted_pathways": concerted,
        "excited_state_evidence": excited_state_evidence,
        "mechanism_G_key_comparison": {
            "S1": mechanism_g_s1,
            "S2": mechanism_g_s2,
            "paper_conclusion": (
                "S1 is weak and off the 450 nm irradiation window; S2 is brighter and near 450 nm."
            ),
        },
        "complex_formation_free_energy_kcal_mol": complex_formation,
        "paper_complex_formation_free_energy_kcal_mol": PAPER_CASE2_COMPLEX_FORMATION_KCAL_MOL,
        "complex_formation_source_difference_kcal_mol": (
            complex_formation - PAPER_CASE2_COMPLEX_FORMATION_KCAL_MOL
        ),
        "checks": checks,
        "source_discrepancies": [
            (
                "The bundled JSON Gibbs energies give cp5 - cp1 - CsPPh2 = "
                f"{complex_formation:.6f} kcal/mol, whereas Supplementary Section B.2.2 "
                "reports -3.0 kcal/mol without a machine-readable correction or alternate conformer."
            )
        ],
    }


def _scan_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    rows = record.get("scan_steps")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{record.get('filename')} has no scan_steps")
    parsed = []
    first_energy = _finite(rows[0]["energy_ev"], "first scan energy")
    for expected_step, row in enumerate(rows, 1):
        if int(row["step"]) != expected_step:
            raise ValueError("scan step indices must be consecutive and one-based")
        energy_ev = _finite(row["energy_ev"], "scan energy")
        parameters = row.get("scan_parameters") or {}
        coordinate = None
        coordinate_name = None
        if len(parameters) == 1:
            coordinate_name, coordinate_value = next(iter(parameters.items()))
            coordinate = _finite(coordinate_value, "scan coordinate")
        parsed.append(
            {
                "step": expected_step,
                "coordinate_name": coordinate_name,
                "coordinate_value": coordinate,
                "energy_ev": energy_ev,
                "relative_energy_kcal_mol": (energy_ev - first_energy) * EV_TO_KCAL_MOL,
            }
        )
    return parsed


def reproduce_case3(payload: dict[str, Any], source_sha256: str) -> dict[str, Any]:
    dihedral_profiles: dict[str, Any] = {}
    for ligand in LIGANDS:
        record = payload["dihedral"]["scan"][ligand]
        rows = _scan_rows(record)
        dihedral_profiles[ligand] = {
            "source_record": f"dihedral.scan.{ligand}",
            "point_count": len(rows),
            "profile": rows,
            "final_penalty_kcal_mol": rows[-1]["relative_energy_kcal_mol"],
            "maximum_penalty_kcal_mol": max(
                row["relative_energy_kcal_mol"] for row in rows
            ),
            "scan_summary": record.get("scan_summary"),
        }

    bite_angle_profiles: dict[str, dict[str, Any]] = {}
    for direction in ("add", "reduce"):
        bite_angle_profiles[direction] = {}
        for ligand in LIGANDS:
            record = payload["bite_angle"]["scan"][direction][ligand]
            rows = _scan_rows(record)
            bite_angle_profiles[direction][ligand] = {
                "source_record": f"bite_angle.scan.{direction}.{ligand}",
                "point_count": len(rows),
                "profile": rows,
                "final_penalty_kcal_mol": rows[-1]["relative_energy_kcal_mol"],
                "maximum_penalty_kcal_mol": max(
                    row["relative_energy_kcal_mol"] for row in rows
                ),
                "scan_summary": record.get("scan_summary"),
            }

    bde_records = payload["bde"]["opt"]
    hydrogen = bde_records["H"]
    bond_dissociation: dict[str, Any] = {}
    for ligand in LIGANDS:
        parent = bde_records[ligand]
        radical = bde_records[f"{ligand}-H"]
        bond_dissociation[ligand] = {
            "source_records": [f"bde.opt.{ligand}", f"bde.opt.{ligand}-H", "bde.opt.H"],
            "electronic_bde_kcal_mol": (
                _finite(radical["scfenergy_final"], "radical electronic energy")
                + _finite(hydrogen["scfenergy_final"], "hydrogen electronic energy")
                - _finite(parent["scfenergy_final"], "parent electronic energy")
            )
            * EV_TO_KCAL_MOL,
            "gibbs_bond_dissociation_free_energy_kcal_mol": (
                _finite(radical["freeenergy"], "radical freeenergy")
                + _finite(hydrogen["freeenergy"], "hydrogen freeenergy")
                - _finite(parent["freeenergy"], "parent freeenergy")
            )
            * HARTREE_TO_KCAL_MOL,
        }

    lowest_dihedral = min(
        dihedral_profiles,
        key=lambda ligand: dihedral_profiles[ligand]["maximum_penalty_kcal_mol"],
    )
    lowest_bite = {
        direction: min(
            LIGANDS,
            key=lambda ligand: bite_angle_profiles[direction][ligand][
                "maximum_penalty_kcal_mol"
            ],
        )
        for direction in ("add", "reduce")
    }
    lowest_electronic_bde = min(
        LIGANDS,
        key=lambda ligand: bond_dissociation[ligand]["electronic_bde_kcal_mol"],
    )
    lowest_gibbs_bdfe = min(
        LIGANDS,
        key=lambda ligand: bond_dissociation[ligand][
            "gibbs_bond_dissociation_free_energy_kcal_mol"
        ],
    )
    checks = {
        "source_sha256_matches": source_sha256 == EXPECTED_CASE3_SHA256,
        "all_dihedral_profiles_have_13_points": all(
            profile["point_count"] == 13 for profile in dihedral_profiles.values()
        ),
        "ortho_has_lowest_dihedral_distortion_penalty": lowest_dihedral == "ortho",
        "all_bite_angle_profiles_have_6_points": all(
            profile["point_count"] == 6
            for direction in bite_angle_profiles.values()
            for profile in direction.values()
        ),
        "ortho_has_lowest_bite_angle_penalty_in_both_directions": all(
            ligand == "ortho" for ligand in lowest_bite.values()
        ),
        "ortho_has_lowest_electronic_bde": lowest_electronic_bde == "ortho",
        "ortho_has_lowest_gibbs_bdfe": lowest_gibbs_bdfe == "ortho",
    }
    return {
        "status": "passed_with_missing_published_wiberg_values" if all(checks.values()) else "failed",
        "dihedral_profiles": dihedral_profiles,
        "dihedral_ranking_by_maximum_penalty": sorted(
            LIGANDS,
            key=lambda ligand: dihedral_profiles[ligand]["maximum_penalty_kcal_mol"],
        ),
        "bite_angle_profiles": bite_angle_profiles,
        "bite_angle_lowest_penalty_ligand": lowest_bite,
        "bond_dissociation": bond_dissociation,
        "lowest_electronic_bde_ligand": lowest_electronic_bde,
        "lowest_gibbs_bdfe_ligand": lowest_gibbs_bdfe,
        "checks": checks,
        "source_limitations": [
            (
                "Supplementary Section B.2.3 states that the ortho ligand has the largest "
                "Ni-H Wiberg index, but supplementary/case3.json contains no Wiberg/NBO matrix "
                "or original Gaussian logs. The statement can only be independently retested by "
                "running calculate_bond_orders/gaussian on the bundled geometries."
            )
        ],
    }


def _atomic_structure(record: dict[str, Any], *, coordinates: Any = None) -> dict[str, Any]:
    atomic_numbers = record.get("atomnos")
    positions = coordinates if coordinates is not None else record.get("atomcoords_final")
    if not isinstance(atomic_numbers, list) or not isinstance(positions, list):
        raise ValueError("record has no atomic numbers/final coordinates")
    if len(atomic_numbers) != len(positions):
        raise ValueError("record atom and coordinate counts differ")
    atoms = []
    for atomic_number, position in zip(atomic_numbers, positions, strict=True):
        number = int(atomic_number)
        if number <= 0 or number >= len(ELEMENTS):
            raise ValueError(f"unsupported atomic number {number}")
        atoms.append(
            {
                "element": ELEMENTS[number],
                "position_angstrom": [_finite(value, "Cartesian coordinate") for value in position],
            }
        )
    return {
        "atoms": atoms,
        "charge": int(record["charge"]),
        "multiplicity": int(record["mult"]),
        "pbc": [False, False, False],
    }


def _dihedral_request(payload: dict[str, Any], ligand: str) -> dict[str, Any]:
    record = payload["dihedral"]["scan"][ligand]
    rows = record["scan_steps"]
    summary_names = record["scan_summary"]["scan_names"]
    definitions = []
    for name in summary_names:
        match = re.fullmatch(r"D\((\d+),(\d+),(\d+),(\d+)\)", name)
        if match:
            definition = [int(value) - 1 for value in match.groups()]
            if definition not in definitions:
                definitions.append(definition)
    if len(definitions) != 1:
        raise ValueError(f"{ligand} scan does not define exactly one unique dihedral")
    first_parameters = rows[0].get("scan_parameters") or {}
    last_parameters = rows[-1].get("scan_parameters") or {}
    if len(first_parameters) != 1 or len(last_parameters) != 1:
        raise ValueError(f"{ligand} scan endpoints have no unique coordinate value")
    return {
        "inputs": {
            "structure": _atomic_structure(record, coordinates=rows[0]["atomcoords"])
        },
        "method_spec": {
            "method": "wB97XD",
            "basis": "def2SVP",
            "solvation_model": "smd",
            "solvent": "N,N-Dimethylacetamide",
        },
        "action_settings": {
            "scf_convergence": "default",
            "coordinate_type": "dihedral",
            "atom_indices": definitions[0],
            "start_value": float(next(iter(first_parameters.values()))),
            "end_value": float(next(iter(last_parameters.values()))),
            "value_unit": "degree",
            "steps": len(rows) - 1,
            "optimization_convergence": "default",
            "max_cycles": 200,
            "force_constant_strategy": "default",
            "disable_symmetry": True,
        },
    }


def _bite_angle_request(
    payload: dict[str, Any], direction: str, ligand: str
) -> dict[str, Any]:
    record = payload["bite_angle"]["scan"][direction][ligand]
    rows = record["scan_steps"]
    definitions = []
    for name in record["scan_summary"]["scan_names"]:
        match = re.fullmatch(r"A\((\d+),(\d+),(\d+)\)", name)
        if match:
            definition = [int(value) - 1 for value in match.groups()]
            if definition not in definitions:
                definitions.append(definition)
    if len(definitions) != 1:
        raise ValueError(
            f"{direction}/{ligand} scan does not define exactly one unique angle"
        )
    first_parameters = rows[0].get("scan_parameters") or {}
    last_parameters = rows[-1].get("scan_parameters") or {}
    if len(first_parameters) != 1 or len(last_parameters) != 1:
        raise ValueError(f"{direction}/{ligand} scan endpoints have no unique value")
    return {
        "inputs": {
            "structure": _atomic_structure(record, coordinates=rows[0]["atomcoords"])
        },
        "method_spec": {
            "method": "B3LYP",
            "basis": "def2TZVP",
            "dispersion": "gd3bj",
            "solvation_model": "smd",
            "solvent": "N,N-Dimethylacetamide",
        },
        "action_settings": {
            "scf_convergence": "default",
            "coordinate_type": "angle",
            "atom_indices": definitions[0],
            "start_value": float(next(iter(first_parameters.values()))),
            "end_value": float(next(iter(last_parameters.values()))),
            "value_unit": "degree",
            "steps": len(rows) - 1,
            "optimization_convergence": "default",
            "max_cycles": 200,
            "force_constant_strategy": "default",
            "disable_symmetry": False,
        },
    }


def _closest_ni_h_pair(structure: dict[str, Any]) -> tuple[int, int, float]:
    atoms = structure["atoms"]
    nickel = [index for index, atom in enumerate(atoms) if atom["element"] == "Ni"]
    hydrogens = [index for index, atom in enumerate(atoms) if atom["element"] == "H"]
    if len(nickel) != 1 or not hydrogens:
        raise ValueError("expected exactly one Ni atom and at least one H atom")
    nickel_index = nickel[0]
    nickel_position = atoms[nickel_index]["position_angstrom"]
    hydrogen_index = min(
        hydrogens,
        key=lambda index: math.dist(nickel_position, atoms[index]["position_angstrom"]),
    )
    distance = math.dist(nickel_position, atoms[hydrogen_index]["position_angstrom"])
    return nickel_index, hydrogen_index, distance


def _bde_from_action_results(
    energy_results: dict[str, dict[str, Any]]
) -> dict[str, dict[str, float]]:
    def energy(species: str) -> float:
        response = energy_results[species]
        if response.get("status") != "success":
            raise ValueError(f"{species} energy Action did not succeed")
        result = response.get("result") or {}
        if str(result.get("unit", "")).casefold() != "hartree":
            raise ValueError(f"{species} energy Action did not return hartree")
        value = result.get("energy")
        return _finite(value, f"{species} Action energy")

    hydrogen = energy("H")
    return {
        ligand: {
            "parent_energy_hartree": energy(ligand),
            "radical_energy_hartree": energy(f"{ligand}-H"),
            "hydrogen_atom_energy_hartree": hydrogen,
            "electronic_bde_kcal_mol": (
                energy(f"{ligand}-H") + hydrogen - energy(ligand)
            )
            * HARTREE_TO_KCAL_MOL,
        }
        for ligand in LIGANDS
    }


def run_real_actions(
    case2: dict[str, Any],
    case3: dict[str, Any],
    selected: set[str],
    *,
    cpu_cores: int,
    memory_mb: int,
    workspace: Path,
    scan_ligands: tuple[str, ...] = LIGANDS,
) -> dict[str, Any]:
    workspace.mkdir(parents=True, exist_ok=True)
    # 第三方 MiniChem 仍读取旧 workspace 键；映射只在此历史复现 adapter 边界。
    os.environ["MINICHEM_SERVICE_WORKSPACE"] = str(workspace.resolve())
    os.environ["MINICHEM_MCP_WORKSPACE"] = os.environ["MINICHEM_SERVICE_WORKSPACE"]
    toolbox_src = Path(__file__).resolve().parents[2] / "third_party" / "minichem_toolbox" / "src"
    if str(toolbox_src) not in sys.path:
        sys.path.insert(0, str(toolbox_src))
    from minichem_toolbox.service import execute_action

    limits = {"cpu_cores": cpu_cores, "memory_mb": memory_mb, "gpu_count": 0}
    results: dict[str, Any] = {}
    if "td" in selected:
        td_results = {}
        for mechanism, (group_name, excited_record_name) in CASE2_EXCITED_RECORDS.items():
            ground_record_name = excited_record_name.removesuffix("_excited")
            request = {
                "backend_id": "gaussian",
                "inputs": {
                    "structure": _atomic_structure(
                        case2[group_name][ground_record_name]
                    )
                },
                "method_spec": {
                    "method": "B3LYP",
                    "basis": "def2SVP",
                    "excited_state_method": "tddft",
                    "dispersion": "gd3bj",
                    "solvation_model": "smd",
                    "solvent": "EthylEthanoate",
                    "integration_grid": "UltraFine",
                },
                "action_settings": {
                    "scf_convergence": "default",
                    "number_of_states": 20,
                    "spin_symmetry": "singlet",
                },
                "resource_limits": limits,
            }
            td_results[mechanism] = execute_action(
                "calculate_excited_states", request
            )
        results["case2_vertical_excited_states"] = td_results
    if "scan" in selected:
        scan_results = {}
        for ligand in scan_ligands:
            request = {
                "backend_id": "gaussian",
                **_dihedral_request(case3, ligand),
                "resource_limits": limits,
            }
            scan_results[ligand] = execute_action(
                "scan_reaction_coordinates", request
            )
        results["case3_dihedral_scans"] = scan_results
    if "bite" in selected:
        bite_results = {}
        for direction in ("add", "reduce"):
            bite_results[direction] = {}
            for ligand in scan_ligands:
                request = {
                    "backend_id": "gaussian",
                    **_bite_angle_request(case3, direction, ligand),
                    "resource_limits": limits,
                }
                bite_results[direction][ligand] = execute_action(
                    "scan_reaction_coordinates", request
                )
        results["case3_bite_angle_scans"] = bite_results
    if "nbo" in selected:
        nbo_results = {}
        for ligand in LIGANDS:
            structure = _atomic_structure(case3["bde"]["opt"][ligand])
            nickel_index, hydrogen_index, distance = _closest_ni_h_pair(structure)
            response = execute_action(
                "calculate_bond_orders",
                {
                    "backend_id": "gaussian",
                    "inputs": {"structure": structure},
                    "method_spec": {
                        "method": "M062X",
                        "basis": "def2TZVP",
                        "bond_order_definition": "wiberg_nao",
                        "solvation_model": "smd",
                        "solvent": "N,N-Dimethylacetamide",
                    },
                    "action_settings": {
                        "scf_convergence": "default",
                        "minimum_bond_order": 0.0,
                    },
                    "resource_limits": limits,
                },
            )
            matrix = (response.get("result") or {}).get("bond_order_matrix")
            ni_h_wiberg = (
                matrix[nickel_index][hydrogen_index]
                if isinstance(matrix, list) and len(matrix) > nickel_index
                else None
            )
            nbo_results[ligand] = {
                "ni_atom_index": nickel_index,
                "hydride_atom_index": hydrogen_index,
                "ni_h_distance_angstrom": distance,
                "ni_h_wiberg_index": ni_h_wiberg,
                "action_result": response,
            }
        results["case3_ni_h_wiberg"] = nbo_results
    if "bde" in selected:
        energy_results = {}
        bde_records = case3["bde"]["opt"]
        for species in (*LIGANDS, *(f"{ligand}-H" for ligand in LIGANDS), "H"):
            energy_results[species] = execute_action(
                "calculate_energy",
                {
                    "backend_id": "gaussian",
                    "inputs": {"structure": _atomic_structure(bde_records[species])},
                    "method_spec": {
                        "method": "M062X",
                        "basis": "def2TZVP",
                        "solvation_model": "smd",
                        "solvent": "N,N-Dimethylacetamide",
                    },
                    "action_settings": {"scf_convergence": "default"},
                    "resource_limits": limits,
                },
            )
        results["case3_bde_energy_calculations"] = energy_results
        if all(response.get("status") == "success" for response in energy_results.values()):
            results["case3_bde_from_real_energies"] = _bde_from_action_results(
                energy_results
            )
    return results


def _collect_action_statuses(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return []
    status = value.get("status")
    if isinstance(status, str):
        return [status]
    statuses = []
    for child in value.values():
        statuses.extend(_collect_action_statuses(child))
    return statuses


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case2-data", type=Path, default=Path("supplementary/case2.json"))
    parser.add_argument("--case3-data", type=Path, default=Path("supplementary/case3.json"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "local_data/runtime/case23_validation_20260813/case23_report.json"
        ),
    )
    parser.add_argument(
        "--real-actions",
        default="",
        help="Comma-separated real Gaussian calculations: td,nbo,scan,bite,bde. Empty means data reanalysis only.",
    )
    parser.add_argument("--cpu-cores", type=int, default=24)
    parser.add_argument("--memory-mb", type=int, default=48000)
    parser.add_argument(
        "--scan-ligands",
        default=",".join(LIGANDS),
        help="Comma-separated Case3 ligand records used by the validation script; the Action remains case-independent.",
    )
    arguments = parser.parse_args()

    selected = {item.strip().casefold() for item in arguments.real_actions.split(",") if item.strip()}
    unknown = selected - {"td", "nbo", "scan", "bite", "bde"}
    if unknown:
        parser.error(f"unknown --real-actions values: {sorted(unknown)}")
    if arguments.cpu_cores < 1 or arguments.memory_mb < 128:
        parser.error("--cpu-cores must be >=1 and --memory-mb must be >=128")
    scan_ligands = tuple(
        item.strip() for item in arguments.scan_ligands.split(",") if item.strip()
    )
    if not scan_ligands or len(set(scan_ligands)) != len(scan_ligands):
        parser.error("--scan-ligands must contain one or more unique ligand names")
    unknown_scan_ligands = set(scan_ligands) - set(LIGANDS)
    if unknown_scan_ligands:
        parser.error(f"unknown --scan-ligands values: {sorted(unknown_scan_ligands)}")

    case2 = json.loads(arguments.case2_data.read_text(encoding="utf-8"))
    case3 = json.loads(arguments.case3_data.read_text(encoding="utf-8"))
    case2_sha = _sha256(arguments.case2_data)
    case3_sha = _sha256(arguments.case3_data)
    report = {
        "status": "completed",
        "mode": "published_data_reanalysis" if not selected else "published_data_and_real_actions",
        "sources": {
            "case2": {"path": _display_path(arguments.case2_data), "sha256": case2_sha},
            "case3": {"path": _display_path(arguments.case3_data), "sha256": case3_sha},
        },
        "case2": reproduce_case2(case2, case2_sha),
        "case3": reproduce_case3(case3, case3_sha),
        "real_action_results": {},
    }
    if selected:
        workspace = arguments.output.parent / "minichem_workspace"
        report["real_action_results"] = run_real_actions(
            case2,
            case3,
            selected,
            cpu_cores=arguments.cpu_cores,
            memory_mb=arguments.memory_mb,
            workspace=workspace,
            scan_ligands=scan_ligands,
        )
        statuses = _collect_action_statuses(report["real_action_results"])
        expected_status_count = (
            (len(CASE2_EXCITED_RECORDS) if "td" in selected else 0)
            + (len(LIGANDS) if "nbo" in selected else 0)
            + (len(scan_ligands) if "scan" in selected else 0)
            + (2 * len(scan_ligands) if "bite" in selected else 0)
            + (2 * len(LIGANDS) + 1 if "bde" in selected else 0)
        )
        if len(statuses) != expected_status_count or any(
            status != "success" for status in statuses
        ):
            report["status"] = "real_action_failure"
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "output": _display_path(arguments.output),
        "case2_status": report["case2"]["status"],
        "case3_status": report["case3"]["status"],
        "real_actions": sorted(selected),
    }, ensure_ascii=True, indent=2))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
