#!/usr/bin/env python3
"""使用仓库附录数据复算 ARCHE 论文 Case 1。

默认分支只重放发布数据；调用方显式启用 Gaussian 或 MiniChem 选项时，脚本会在
独立工作目录执行对应计算，因此它不是当前 `investigation_v2` 生产入口。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from chemistry_multiagent.tools.minichem_adapter import MiniChemMcpAdapter


HARTREE_TO_EV = 27.211386245988
HARTREE_TO_KCAL_MOL = 627.5094740631
GAS_CONSTANT_KCAL_MOL_K = 0.00198720425864083
EXPECTED_DATA_SHA256 = "28cba09d3ef6e450be88df83d0183f364b78dee8b513ad7a27c89c14fb8e14c4"
PAPER_MAJOR_TS = "TS-2a"
PAPER_MINOR_TS = "TS-2b-1"
PAPER_DELTA_DELTA_G_KCAL_MOL = 1.9
PAPER_VERIFIED_TS_COUNT = 12
FIGURE_S2_TS_2C_KCAL_MOL = 5.15
EXPERIMENTAL_EE_PERCENT = 88.0
GAUSSIAN_SENSITIVITY_ROUTES = {
    "m062x_6311pgdp_gas": "#P M062X/6-311+G(d,p) SP Int=UltraFine SCF=Tight",
    "m062x_6311pgdp_smd_acetone": (
        "#P M062X/6-311+G(d,p) SP SCRF=(SMD,Solvent=Acetone) "
        "Int=UltraFine SCF=Tight"
    ),
    "m062x_6311pgdp_smd_toluene": (
        "#P M062X/6-311+G(d,p) SP SCRF=(SMD,Solvent=Toluene) "
        "Int=UltraFine SCF=Tight"
    ),
    "m062x_6311pgdp_smd_water": (
        "#P M062X/6-311+G(d,p) SP SCRF=(SMD,Solvent=Water) "
        "Int=UltraFine SCF=Tight"
    ),
    "wb97xd_iefpcm_acetone": (
        "#P wB97XD/def2TZVPP SP SCRF=(IEFPCM,Solvent=Acetone) Int=UltraFine SCF=Tight"
    ),
    "b3lyp_gd3bj_iefpcm_acetone": (
        "#P B3LYP/def2TZVPP SP EmpiricalDispersion=GD3BJ "
        "SCRF=(IEFPCM,Solvent=Acetone) Int=UltraFine SCF=Tight"
    ),
}
ELEMENT_SYMBOLS = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    16: "S",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    """Render report paths relative to the caller's working directory."""

    return os.path.relpath(path.resolve(), start=Path.cwd().resolve())


def _last_scf_energy_ev(record: dict[str, Any]) -> float:
    values = record.get("scfenergies")
    if not isinstance(values, list) or not values:
        raise ValueError("record has no SCF energy")
    return float(values[-1])


def _composite_free_energy_hartree(
    optimization: dict[str, Any],
    single_point: dict[str, Any],
) -> float:
    optimization_energy = _last_scf_energy_ev(optimization) / HARTREE_TO_EV
    single_point_energy = _last_scf_energy_ev(single_point) / HARTREE_TO_EV
    thermal_correction = float(optimization["freeenergy"]) - optimization_energy
    return single_point_energy + thermal_correction


def _predicted_ee_percent(delta_delta_g_kcal_mol: float, temperature_kelvin: float) -> float:
    enantiomeric_ratio = math.exp(
        delta_delta_g_kcal_mol / (GAS_CONSTANT_KCAL_MOL_K * temperature_kelvin)
    )
    return 100.0 * (enantiomeric_ratio - 1.0) / (enantiomeric_ratio + 1.0)


def reproduce_supplementary_data(data_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(data_path.read_text(encoding="utf-8"))
    optimizations = payload.get("opt")
    single_points = payload.get("sp")
    if not isinstance(optimizations, dict) or not isinstance(single_points, dict):
        raise ValueError("Case 1 data must contain opt and sp objects")

    optimization_names = set(optimizations)
    single_point_names = set(single_points)
    if optimization_names != single_point_names:
        raise ValueError("Case 1 optimization and single-point candidates do not match")

    rows: list[dict[str, Any]] = []
    for name in sorted(optimization_names):
        optimization = optimizations[name]
        single_point = single_points[name]
        frequencies = [float(value) for value in optimization.get("vibfreqs", [])]
        composite_energy = _composite_free_energy_hartree(optimization, single_point)
        rows.append(
            {
                "candidate": name,
                "composite_free_energy_hartree": composite_energy,
                "imaginary_frequency_count": sum(value < 0.0 for value in frequencies),
                "lowest_frequency_cm-1": min(frequencies) if frequencies else None,
            }
        )

    minimum_energy = min(row["composite_free_energy_hartree"] for row in rows)
    for row in rows:
        row["relative_free_energy_kcal_mol"] = (
            row["composite_free_energy_hartree"] - minimum_energy
        ) * HARTREE_TO_KCAL_MOL
    rows.sort(key=lambda row: (row["relative_free_energy_kcal_mol"], row["candidate"]))

    by_name = {row["candidate"]: row for row in rows}
    delta_delta_g = (
        by_name[PAPER_MINOR_TS]["composite_free_energy_hartree"]
        - by_name[PAPER_MAJOR_TS]["composite_free_energy_hartree"]
    ) * HARTREE_TO_KCAL_MOL
    temperature = float(optimizations[PAPER_MAJOR_TS]["temperature"])
    experimental_ratio = (100.0 + EXPERIMENTAL_EE_PERCENT) / (
        100.0 - EXPERIMENTAL_EE_PERCENT
    )
    experimental_delta_delta_g = (
        GAS_CONSTANT_KCAL_MOL_K * temperature * math.log(experimental_ratio)
    )
    data_sha256 = _sha256(data_path)
    checks = {
        "data_sha256_matches": data_sha256 == EXPECTED_DATA_SHA256,
        "candidate_sets_match": optimization_names == single_point_names,
        "candidate_count_is_14": len(rows) == 14,
        "all_candidate_frequency_records_have_one_imaginary_mode": all(
            row["imaginary_frequency_count"] == 1 for row in rows
        ),
        "lowest_candidate_is_TS-2a": rows[0]["candidate"] == PAPER_MAJOR_TS,
        "major_minor_gap_matches_1.9_kcal_mol": abs(
            delta_delta_g - PAPER_DELTA_DELTA_G_KCAL_MOL
        ) < 0.02,
    }
    source_discrepancies = {
        "paper_verified_ts_count": PAPER_VERIFIED_TS_COUNT,
        "supplementary_json_candidate_count": len(rows),
        "candidate_count_difference": len(rows) - PAPER_VERIFIED_TS_COUNT,
        "figure_s2_TS-2c_kcal_mol": FIGURE_S2_TS_2C_KCAL_MOL,
        "supplementary_json_TS-2c_kcal_mol": by_name["TS-2c"][
            "relative_free_energy_kcal_mol"
        ],
        "TS-2c_difference_kcal_mol": (
            by_name["TS-2c"]["relative_free_energy_kcal_mol"]
            - FIGURE_S2_TS_2C_KCAL_MOL
        ),
        "irc_records_in_supplementary_json": 0,
    }
    report = {
        "status": "passed_with_source_discrepancies" if all(checks.values()) else "failed",
        "mode": "supplementary_data_reanalysis",
        "source": _display_path(data_path),
        "source_sha256": data_sha256,
        "candidate_count": len(rows),
        "temperature_kelvin": temperature,
        "major_transition_state": PAPER_MAJOR_TS,
        "minor_transition_state": PAPER_MINOR_TS,
        "computed_delta_delta_g_kcal_mol": delta_delta_g,
        "paper_delta_delta_g_kcal_mol": PAPER_DELTA_DELTA_G_KCAL_MOL,
        "experimental_ee_percent": EXPERIMENTAL_EE_PERCENT,
        "experimental_delta_delta_g_kcal_mol": experimental_delta_delta_g,
        "computed_ee_percent": _predicted_ee_percent(delta_delta_g, temperature),
        "checks": checks,
        "source_discrepancies": source_discrepancies,
        "ranking": rows,
    }
    return report, payload


def _atomic_structure(record: dict[str, Any]) -> dict[str, Any]:
    atomic_numbers = record.get("atomnos")
    coordinates = record.get("atomcoords_final")
    if not isinstance(atomic_numbers, list) or not isinstance(coordinates, list):
        raise ValueError("candidate has no final atomic structure")
    if len(atomic_numbers) != len(coordinates):
        raise ValueError("candidate atom and coordinate counts differ")
    atoms = []
    for atomic_number, position in zip(atomic_numbers, coordinates, strict=True):
        try:
            symbol = ELEMENT_SYMBOLS[int(atomic_number)]
        except KeyError as exc:
            raise ValueError(f"unsupported atomic number: {atomic_number}") from exc
        atoms.append({"element": symbol, "position_angstrom": position})
    return {
        "atoms": atoms,
        "charge": int(record["charge"]),
        "multiplicity": int(record["mult"]),
        "pbc": [False, False, False],
    }


def _extract_action_energy_hartree(result: dict[str, Any]) -> float:
    result_payload = result.get("result")
    if isinstance(result_payload, dict) and result_payload.get("energy") is not None:
        return float(result_payload["energy"])
    if result.get("energy") is not None:
        return float(result["energy"])
    raise ValueError("MiniChem ActionResult has no scalar energy")


def _run_gaussian_candidate(
    endpoint: str,
    name: str,
    record: dict[str, Any],
    *,
    cpu_cores: int,
    memory_mb: int,
) -> dict[str, Any]:
    request = {
        "action_id": "calculate_energy",
        "backend_id": "gaussian",
        "component_backends": {},
        "source_id": None,
        "inputs": {"structure": _atomic_structure(record)},
        "method_spec": {
            "method": "M062X",
            "basis": "def2TZVPP",
            "solvation_model": "iefpcm",
            "solvent": "Acetone",
            "integration_grid": "UltraFine",
            "charge": int(record["charge"]),
            "multiplicity": int(record["mult"]),
        },
        "action_settings": {"scf_convergence": "Tight"},
        "resource_limits": {
            "cpu_cores": cpu_cores,
            "memory_mb": memory_mb,
            "gpu_count": 0,
        },
    }
    with MiniChemMcpAdapter(endpoint=endpoint) as adapter:
        catalog = adapter.discover_catalog()
        result = adapter.execute_action(request)
    if result.get("status") != "completed":
        raise RuntimeError(f"MiniChem Gaussian calculation failed for {name}: {result.get('error')}")
    energy = _extract_action_energy_hartree(result)
    reference_energy = _last_scf_energy_ev(record) / HARTREE_TO_EV
    return {
        "candidate": name,
        "catalog_hash": catalog.catalog_hash,
        "status": result["status"],
        "energy_hartree": energy,
        "supplementary_energy_hartree": reference_energy,
        "difference_hartree": energy - reference_energy,
        "difference_kcal_mol": (energy - reference_energy) * HARTREE_TO_KCAL_MOL,
        "artifact_count": len(result.get("output_artifacts", [])),
    }


def run_gaussian_validation(
    payload: dict[str, Any],
    *,
    endpoint: str,
    candidates: list[str],
    cpu_cores: int,
    memory_mb: int,
    workers: int,
) -> dict[str, Any]:
    single_points = payload["sp"]
    missing = sorted(set(candidates) - set(single_points))
    if missing:
        raise ValueError(f"unknown Case 1 candidates: {missing}")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _run_gaussian_candidate,
                endpoint,
                name,
                single_points[name],
                cpu_cores=cpu_cores,
                memory_mb=memory_mb,
            )
            for name in candidates
        ]
        results = [future.result() for future in futures]
    results.sort(key=lambda row: row["candidate"])
    return {
        "status": "completed",
        "mode": "minichem_typed_action_validation",
        "endpoint": endpoint,
        "route_equivalence": "exact_reported_single_point_route",
        "route_note": (
            "The typed Gaussian Action explicitly selects IEFPCM acetone, UltraFine, "
            "SCF=Tight, M06-2X, and def2-TZVPP."
        ),
        "calculations": results,
    }


def _local_minichem_executor(
    repository_root: Path,
    *,
    workspace: Path,
    timeout_seconds: float,
):
    """Load the bundled MiniChem service with one explicit local workspace."""

    toolbox_source = repository_root / "third_party/minichem_toolbox/src"
    source_text = str(toolbox_source.resolve())
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    workspace.mkdir(parents=True, exist_ok=True)
    # 第三方 MiniChem 仍读取旧 workspace 键；映射只在此历史复现 adapter 边界。
    os.environ["MINICHEM_SERVICE_WORKSPACE"] = str(workspace.resolve())
    os.environ["MINICHEM_MCP_WORKSPACE"] = os.environ["MINICHEM_SERVICE_WORKSPACE"]
    os.environ["RESEARCHCHEMBENCH_COMPUTE_ACTION_TIMEOUT_SECONDS"] = str(
        max(1, math.ceil(timeout_seconds))
    )
    from minichem_toolbox.service import execute_action

    return execute_action


def _write_action_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def run_minichem_hessian_validation(
    payload: dict[str, Any],
    *,
    repository_root: Path,
    workspace: Path,
    result_root: Path,
    candidates: list[str],
    cpu_cores: int,
    memory_mb: int,
    workers: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Recalculate candidate Hessians and project out external molecular modes."""

    optimizations = payload["opt"]
    missing = sorted(set(candidates) - set(optimizations))
    if missing:
        raise ValueError(f"unknown Case 1 candidates: {missing}")
    execute_action = _local_minichem_executor(
        repository_root,
        workspace=workspace,
        timeout_seconds=timeout_seconds,
    )

    def execute(name: str) -> dict[str, Any]:
        structure = _atomic_structure(optimizations[name])
        hessian = execute_action(
            "calculate_hessian",
            {
                "backend_id": "gaussian",
                "inputs": {"structure": structure},
                "method_spec": {
                    "method": "B3LYP",
                    "basis": "6-31G(d)",
                    "solvation_model": "iefpcm",
                    "solvent": "Acetone",
                    "charge": structure["charge"],
                    "multiplicity": structure["multiplicity"],
                },
                "action_settings": {
                    "scf_convergence": "default",
                    "temperature_kelvin": 303.15,
                },
                "resource_limits": {
                    "cpu_cores": cpu_cores,
                    "memory_mb": memory_mb,
                    "gpu_count": 0,
                },
            },
        )
        hessian_path = result_root / f"hessian_{name}.json"
        _write_action_result(hessian_path, hessian)
        row: dict[str, Any] = {
            "candidate": name,
            "hessian_status": hessian.get("status"),
            "hessian_result_file": _display_path(hessian_path),
            "gaussian_version": hessian.get("backend_version"),
        }
        if hessian.get("status") != "success":
            row["error"] = hessian.get("error")
            return row
        hessian_result = hessian.get("result") or {}
        hessian_structure = hessian_result.get("structure")
        if not isinstance(hessian_structure, dict):
            row["error"] = {
                "code": "missing_hessian_frame_structure",
                "message": (
                    "calculate_hessian succeeded without result.structure in the "
                    "Hessian Cartesian frame"
                ),
            }
            return row

        vibrations = execute_action(
            "derive_vibrational_modes",
            {
                "backend_id": "internal_vibrations",
                "inputs": {
                    "structure": hessian_structure,
                    "hessian": hessian_result,
                },
                "action_settings": {"linearity": "nonlinear"},
                "resource_limits": {
                    "cpu_cores": 1,
                    "memory_mb": min(memory_mb, 8192),
                    "gpu_count": 0,
                },
            },
        )
        vibrations_path = result_root / f"vibrations_{name}.json"
        _write_action_result(vibrations_path, vibrations)
        row.update(
            {
                "vibrations_status": vibrations.get("status"),
                "vibrations_result_file": _display_path(vibrations_path),
            }
        )
        if vibrations.get("status") == "success":
            values = vibrations["result"]["frequencies_cm1"]
            imaginary = [
                float(value["imaginary"])
                for value in values
                if float(value.get("imaginary", 0.0)) > 0.0
            ]
            row.update(
                {
                    "vibrational_mode_count": len(values),
                    "imaginary_frequency_count": len(imaginary),
                    "imaginary_frequencies_cm1": imaginary,
                }
            )
        else:
            row["error"] = vibrations.get("error")
        return row

    with ThreadPoolExecutor(max_workers=workers) as executor:
        rows = list(executor.map(execute, candidates))
    rows.sort(key=lambda row: row["candidate"])
    complete = all(
        row.get("hessian_status") == "success"
        and row.get("vibrations_status") == "success"
        for row in rows
    )
    one_imaginary_mode = all(
        row.get("imaginary_frequency_count") == 1 for row in rows
    )
    return {
        "status": "completed" if complete and one_imaginary_mode else "incomplete",
        "mode": "minichem_case1_hessian_and_projected_vibrations",
        "route": (
            "#P B3LYP/6-31G(d) Freq SCRF=(IEFPCM,Solvent=Acetone) "
            "Temperature=303.15"
        ),
        "candidate_count": len(rows),
        "all_actions_succeeded": complete,
        "all_candidates_have_one_imaginary_mode": one_imaginary_mode,
        "calculations": rows,
    }


def run_minichem_ts_optimization_validation(
    payload: dict[str, Any],
    *,
    repository_root: Path,
    workspace: Path,
    result_root: Path,
    candidates: list[str],
    cpu_cores: int,
    memory_mb: int,
    workers: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Reoptimize released TS geometries, then recalculate Hessians and modes."""

    optimizations = payload["opt"]
    missing = sorted(set(candidates) - set(optimizations))
    if missing:
        raise ValueError(f"unknown Case 1 candidates: {missing}")
    execute_action = _local_minichem_executor(
        repository_root,
        workspace=workspace,
        timeout_seconds=timeout_seconds,
    )

    def execute(name: str) -> dict[str, Any]:
        initial_guess = _atomic_structure(optimizations[name])
        transition_state = execute_action(
            "locate_transition_state",
            {
                "backend_id": "gaussian",
                "inputs": {"initial_guess": initial_guess},
                "method_spec": {
                    "method": "B3LYP",
                    "basis": "6-31G(d)",
                    "solvation_model": "iefpcm",
                    "solvent": "Acetone",
                    "charge": initial_guess["charge"],
                    "multiplicity": initial_guess["multiplicity"],
                },
                "action_settings": {
                    "scf_convergence": "default",
                    "no_eigen_test": True,
                    "calculate_all_force_constants": True,
                    "max_step": 5,
                },
                "resource_limits": {
                    "cpu_cores": cpu_cores,
                    "memory_mb": memory_mb,
                    "gpu_count": 0,
                },
            },
        )
        transition_state_path = result_root / f"ts_optimization_{name}.json"
        _write_action_result(transition_state_path, transition_state)
        transition_state_result = transition_state.get("result") or {}
        row: dict[str, Any] = {
            "candidate": name,
            "transition_state_status": transition_state.get("status"),
            "transition_state_converged": transition_state_result.get("converged"),
            "transition_state_result_file": _display_path(transition_state_path),
            "gaussian_version": transition_state.get("backend_version"),
        }
        if (
            transition_state.get("status") != "success"
            or transition_state_result.get("converged") is not True
            or not isinstance(transition_state_result.get("structure"), dict)
        ):
            row["error"] = transition_state.get("error")
            row["warnings"] = transition_state.get("warnings") or []
            return row

        optimized_structure = transition_state_result["structure"]
        hessian = execute_action(
            "calculate_hessian",
            {
                "backend_id": "gaussian",
                "inputs": {"structure": optimized_structure},
                "method_spec": {
                    "method": "B3LYP",
                    "basis": "6-31G(d)",
                    "solvation_model": "iefpcm",
                    "solvent": "Acetone",
                    "charge": optimized_structure["charge"],
                    "multiplicity": optimized_structure["multiplicity"],
                },
                "action_settings": {
                    "scf_convergence": "default",
                    "temperature_kelvin": 303.15,
                },
                "resource_limits": {
                    "cpu_cores": cpu_cores,
                    "memory_mb": memory_mb,
                    "gpu_count": 0,
                },
            },
        )
        hessian_path = result_root / f"ts_hessian_{name}.json"
        _write_action_result(hessian_path, hessian)
        row.update(
            {
                "hessian_status": hessian.get("status"),
                "hessian_result_file": _display_path(hessian_path),
            }
        )
        if hessian.get("status") != "success":
            row["error"] = hessian.get("error")
            row["warnings"] = hessian.get("warnings") or []
            return row
        hessian_result = hessian.get("result") or {}
        hessian_structure = hessian_result.get("structure")
        if not isinstance(hessian_structure, dict):
            row["error"] = {
                "code": "missing_hessian_frame_structure",
                "message": (
                    "calculate_hessian succeeded without result.structure in the "
                    "Hessian Cartesian frame"
                ),
            }
            return row

        vibrations = execute_action(
            "derive_vibrational_modes",
            {
                "backend_id": "internal_vibrations",
                "inputs": {
                    "structure": hessian_structure,
                    "hessian": hessian_result,
                },
                "action_settings": {"linearity": "nonlinear"},
                "resource_limits": {
                    "cpu_cores": 1,
                    "memory_mb": min(memory_mb, 8192),
                    "gpu_count": 0,
                },
            },
        )
        vibrations_path = result_root / f"ts_vibrations_{name}.json"
        _write_action_result(vibrations_path, vibrations)
        row.update(
            {
                "vibrations_status": vibrations.get("status"),
                "vibrations_result_file": _display_path(vibrations_path),
            }
        )
        if vibrations.get("status") == "success":
            values = vibrations["result"]["frequencies_cm1"]
            imaginary = [
                float(value["imaginary"])
                for value in values
                if float(value.get("imaginary", 0.0)) > 0.0
            ]
            row.update(
                {
                    "vibrational_mode_count": len(values),
                    "imaginary_frequency_count": len(imaginary),
                    "imaginary_frequencies_cm1": imaginary,
                }
            )
        else:
            row["error"] = vibrations.get("error")
            row["warnings"] = vibrations.get("warnings") or []
        return row

    with ThreadPoolExecutor(max_workers=workers) as executor:
        rows = list(executor.map(execute, candidates))
    rows.sort(key=lambda row: row["candidate"])
    complete = all(
        row.get("transition_state_status") == "success"
        and row.get("transition_state_converged") is True
        and row.get("hessian_status") == "success"
        and row.get("vibrations_status") == "success"
        and row.get("imaginary_frequency_count") == 1
        for row in rows
    )
    return {
        "status": "completed" if complete else "incomplete",
        "mode": "released_geometry_ts_reoptimization_and_frequency",
        "evidence_level": "E2",
        "transition_state_route": (
            "#P B3LYP/6-31G(d) Opt=(TS,NoEigenTest,CalcAll,MaxStep=5) "
            "SCRF=(IEFPCM,Solvent=Acetone)"
        ),
        "hessian_route": (
            "#P B3LYP/6-31G(d) Freq SCRF=(IEFPCM,Solvent=Acetone) "
            "Temperature=303.15"
        ),
        "candidate_count": len(rows),
        "all_candidates_reoptimized_and_have_one_imaginary_mode": complete,
        "scope_note": (
            "Each calculation starts from a released final TS geometry. Completion is "
            "independent E2 validation of those supplied candidates, not E3 discovery from "
            "the reaction specification. Imaginary-mode chemical identity and IRC endpoints "
            "remain separate gates."
        ),
        "calculations": rows,
    }


def run_minichem_irc_validation(
    payload: dict[str, Any],
    *,
    repository_root: Path,
    workspace: Path,
    result_root: Path,
    candidates: list[str],
    cpu_cores: int,
    memory_mb: int,
    workers: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Run both atomic Gaussian IRC branches for explicitly selected Case1 TSs."""

    optimizations = payload["opt"]
    missing = sorted(set(candidates) - set(optimizations))
    if missing:
        raise ValueError(f"unknown Case 1 candidates: {missing}")
    execute_action = _local_minichem_executor(
        repository_root,
        workspace=workspace,
        timeout_seconds=timeout_seconds,
    )
    tasks = [
        (candidate, direction)
        for candidate in candidates
        for direction in ("forward", "backward")
    ]

    def execute(task: tuple[str, str]) -> dict[str, Any]:
        name, direction = task
        structure = _atomic_structure(optimizations[name])
        response = execute_action(
            "trace_intrinsic_reaction_coordinate",
            {
                "backend_id": "gaussian",
                "inputs": {"transition_state": structure},
                "method_spec": {
                    "method": "B3LYP",
                    "basis": "6-31G(d)",
                    "solvation_model": "iefpcm",
                    "solvent": "Acetone",
                    "charge": structure["charge"],
                    "multiplicity": structure["multiplicity"],
                },
                "action_settings": {
                    "scf_convergence": "default",
                    "direction": direction,
                    "initial_force_constants": "calcfc",
                    "integrator": "lqa",
                    "max_points": 50,
                    "recalc": 5,
                },
                "resource_limits": {
                    "cpu_cores": cpu_cores,
                    "memory_mb": memory_mb,
                    "gpu_count": 0,
                },
            },
        )
        output_path = result_root / f"irc_{name}_{direction}.json"
        _write_action_result(output_path, response)
        result = response.get("result") or {}
        return {
            "candidate": name,
            "direction": direction,
            "status": response.get("status"),
            "gaussian_branch_completed": result.get("gaussian_branch_completed"),
            "pes_minimum_detected": result.get("pes_minimum_detected"),
            "converged": result.get("converged"),
            "termination_reason": result.get("termination_reason"),
            "completed_points": result.get("completed_points"),
            "path_structure_count": result.get("path_structure_count"),
            "result_file": _display_path(output_path),
            "error": response.get("error"),
            "warnings": response.get("warnings") or [],
        }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        rows = list(executor.map(execute, tasks))
    rows.sort(key=lambda row: (row["candidate"], row["direction"]))
    return {
        "status": (
            "completed"
            if all(row["status"] == "success" and row["converged"] is True for row in rows)
            else "incomplete"
        ),
        "mode": "minichem_case1_bidirectional_gaussian_irc",
        "route": (
            "#P B3LYP/6-31G(d) IRC=(CalcFC,LQA,MaxPoints=50,Recalc=5,{direction}) "
            "SCRF=(IEFPCM,Solvent=Acetone)"
        ),
        "calculations": rows,
    }


def _requested_stage_summary(
    report: dict[str, Any], stage_keys: list[str]
) -> dict[str, Any]:
    statuses = {
        key: (report.get(key) or {}).get("status")
        for key in stage_keys
    }
    incomplete = {
        key: status for key, status in statuses.items() if status != "completed"
    }
    return {
        "all_requested_stages_completed": not incomplete,
        "statuses": statuses,
        "incomplete_stages": incomplete,
    }


def _render_exact_single_point_input(
    name: str,
    record: dict[str, Any],
    *,
    cpu_cores: int,
    memory_mb: int,
    route: str = (
        "#P M062X/def2TZVPP SP SCRF=(IEFPCM,Solvent=Acetone) "
        "Int=UltraFine SCF=Tight"
    ),
    title_prefix: str = "ARCHE paper Case 1 exact-route single point",
) -> str:
    structure = _atomic_structure(record)
    lines = [
        "%Chk=job.chk",
        f"%NProcShared={cpu_cores}",
        f"%Mem={memory_mb}MB",
        route,
        "",
        f"{title_prefix}: {name}",
        "",
        f"{structure['charge']} {structure['multiplicity']}",
    ]
    lines.extend(
        f"{atom['element']} " + " ".join(f"{float(value):.12f}" for value in atom["position_angstrom"])
        for atom in structure["atoms"]
    )
    lines.extend(("", ""))
    return "\n".join(lines)


def _run_exact_gaussian_candidate(
    executable: Path,
    work_root: Path,
    name: str,
    record: dict[str, Any],
    *,
    cpu_cores: int,
    memory_mb: int,
    timeout_seconds: float,
    route: str = "#P M062X/def2TZVPP SP SCRF=(IEFPCM,Solvent=Acetone) Int=UltraFine SCF=Tight",
    title_prefix: str = "ARCHE paper Case 1 exact-route single point",
) -> dict[str, Any]:
    candidate_root = work_root / name
    scratch_root = candidate_root / "scratch"
    candidate_root.mkdir(parents=True, exist_ok=True)
    scratch_root.mkdir(parents=True, exist_ok=True)
    input_text = _render_exact_single_point_input(
        name,
        record,
        cpu_cores=cpu_cores,
        memory_mb=memory_mb,
        route=route,
        title_prefix=title_prefix,
    )
    input_path = candidate_root / "job.gjf"
    log_path = candidate_root / "job.log"
    error_path = candidate_root / "job.err"
    input_path.write_text(input_text, encoding="utf-8")

    executable = executable.resolve(strict=True)
    gaussian_directory = executable.parent
    environment = os.environ.copy()
    environment.update(
        {
            "g16root": str(gaussian_directory.parent),
            "GAUSS_EXEDIR": str(gaussian_directory),
            "GAUSS_SCRDIR": str(scratch_root.resolve()),
            "PATH": f"{gaussian_directory}:{environment.get('PATH', '')}",
        }
    )
    process = subprocess.Popen(
        [str(executable)],
        cwd=candidate_root,
        env=environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(input=input_text, timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = process.communicate()
        log_path.write_text(stdout, encoding="utf-8")
        error_path.write_text(stderr, encoding="utf-8")
        raise RuntimeError(f"Gaussian timed out for {name} after {timeout_seconds} seconds")
    log_path.write_text(stdout, encoding="utf-8")
    error_path.write_text(stderr, encoding="utf-8")
    if process.returncode != 0 or "Normal termination of Gaussian 16" not in stdout:
        raise RuntimeError(f"Gaussian failed for {name}; see {log_path} and {error_path}")

    energy_matches = re.findall(
        r"SCF Done:\s+E\([^)]+\)\s+=\s+(-?\d+(?:\.\d+)?)",
        stdout,
    )
    if not energy_matches:
        raise RuntimeError(f"Gaussian energy was not found for {name}")
    version_match = re.search(r"Gaussian 16, Revision\s+([^,\s]+)", stdout)
    energy = float(energy_matches[-1])
    reference_energy = _last_scf_energy_ev(record) / HARTREE_TO_EV
    return {
        "candidate": name,
        "status": "completed",
        "gaussian_version": version_match.group(1) if version_match else "unknown",
        "energy_hartree": energy,
        "supplementary_energy_hartree": reference_energy,
        "difference_hartree": energy - reference_energy,
        "difference_kcal_mol": (energy - reference_energy) * HARTREE_TO_KCAL_MOL,
        "input_file": _display_path(input_path),
        "log_file": _display_path(log_path),
    }


def run_exact_gaussian_validation(
    payload: dict[str, Any],
    *,
    executable: Path,
    work_root: Path,
    candidates: list[str],
    cpu_cores: int,
    memory_mb: int,
    workers: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    optimizations = payload["opt"]
    single_points = payload["sp"]
    missing = sorted(set(candidates) - set(single_points))
    if missing:
        raise ValueError(f"unknown Case 1 candidates: {missing}")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _run_exact_gaussian_candidate,
                executable,
                work_root,
                name,
                single_points[name],
                cpu_cores=cpu_cores,
                memory_mb=memory_mb,
                timeout_seconds=timeout_seconds,
            )
            for name in candidates
        ]
        results = [future.result() for future in futures]

    for result in results:
        name = result["candidate"]
        optimization_energy = _last_scf_energy_ev(optimizations[name]) / HARTREE_TO_EV
        thermal_correction = float(optimizations[name]["freeenergy"]) - optimization_energy
        result["supplementary_thermal_correction_hartree"] = thermal_correction
        result["recomputed_composite_free_energy_hartree"] = (
            result["energy_hartree"] + thermal_correction
        )
    results.sort(key=lambda row: row["candidate"])
    by_name = {row["candidate"]: row for row in results}
    summary: dict[str, Any] = {}
    if PAPER_MAJOR_TS in by_name and PAPER_MINOR_TS in by_name:
        delta_delta_g = (
            by_name[PAPER_MINOR_TS]["recomputed_composite_free_energy_hartree"]
            - by_name[PAPER_MAJOR_TS]["recomputed_composite_free_energy_hartree"]
        ) * HARTREE_TO_KCAL_MOL
        temperature = float(optimizations[PAPER_MAJOR_TS]["temperature"])
        summary = {
            "recomputed_delta_delta_g_kcal_mol": delta_delta_g,
            "recomputed_ee_percent": _predicted_ee_percent(delta_delta_g, temperature),
        }
    return {
        "status": "completed",
        "mode": "direct_gaussian_exact_reported_route",
        "route": (
            "#P M062X/def2TZVPP SP SCRF=(IEFPCM,Solvent=Acetone) "
            "Int=UltraFine SCF=Tight"
        ),
        "route_equivalence": "exact_reported_single_point_route",
        "reproducibility_note": (
            "The paper does not report CPU or memory directives, and its released data "
            "identifies Gaussian 16 B.01 while this environment provides C.01."
        ),
        **summary,
        "calculations": results,
    }


def run_gaussian_sensitivity_validation(
    payload: dict[str, Any],
    *,
    executable: Path,
    work_root: Path,
    cpu_cores: int,
    memory_mb: int,
    workers: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Compare solvent and functional sensitivity on the released major/minor geometries."""

    optimizations = payload["opt"]
    single_points = payload["sp"]
    candidates = (PAPER_MAJOR_TS, PAPER_MINOR_TS)
    tasks = [
        (route_name, route, candidate)
        for route_name, route in GAUSSIAN_SENSITIVITY_ROUTES.items()
        for candidate in candidates
    ]

    def execute(task: tuple[str, str, str]) -> tuple[str, dict[str, Any]]:
        route_name, route, candidate = task
        result = _run_exact_gaussian_candidate(
            executable,
            work_root / route_name,
            candidate,
            single_points[candidate],
            cpu_cores=cpu_cores,
            memory_mb=memory_mb,
            timeout_seconds=timeout_seconds,
            route=route,
            title_prefix=f"ARCHE Case 1 fixed-geometry sensitivity ({route_name})",
        )
        for reference_key in (
            "supplementary_energy_hartree",
            "difference_hartree",
            "difference_kcal_mol",
        ):
            result.pop(reference_key, None)
        return route_name, result

    with ThreadPoolExecutor(max_workers=workers) as executor:
        task_results = list(executor.map(execute, tasks))

    grouped: dict[str, list[dict[str, Any]]] = {
        route_name: [] for route_name in GAUSSIAN_SENSITIVITY_ROUTES
    }
    for route_name, result in task_results:
        grouped[route_name].append(result)

    route_results = []
    for route_name, calculations in grouped.items():
        calculations.sort(key=lambda row: row["candidate"])
        by_name = {row["candidate"]: row for row in calculations}
        electronic_gap = (
            by_name[PAPER_MINOR_TS]["energy_hartree"]
            - by_name[PAPER_MAJOR_TS]["energy_hartree"]
        ) * HARTREE_TO_KCAL_MOL
        composite_energies = {}
        for candidate in candidates:
            optimization = optimizations[candidate]
            optimization_energy = _last_scf_energy_ev(optimization) / HARTREE_TO_EV
            thermal_correction = float(optimization["freeenergy"]) - optimization_energy
            composite_energies[candidate] = (
                by_name[candidate]["energy_hartree"] + thermal_correction
            )
        composite_gap = (
            composite_energies[PAPER_MINOR_TS] - composite_energies[PAPER_MAJOR_TS]
        ) * HARTREE_TO_KCAL_MOL
        route_results.append(
            {
                "route_id": route_name,
                "route": GAUSSIAN_SENSITIVITY_ROUTES[route_name],
                "electronic_gap_kcal_mol": electronic_gap,
                "fixed_thermal_composite_gap_kcal_mol": composite_gap,
                "fixed_thermal_predicted_ee_percent": _predicted_ee_percent(
                    composite_gap,
                    float(optimizations[PAPER_MAJOR_TS]["temperature"]),
                ),
                "calculations": calculations,
            }
        )
    return {
        "status": "completed",
        "mode": "fixed_geometry_solvent_and_functional_sensitivity",
        "candidates": list(candidates),
        "temperature_kelvin": float(optimizations[PAPER_MAJOR_TS]["temperature"]),
        "scope_note": (
            "All routes use the released IEFPCM-acetone optimized geometries. The composite "
            "comparison reuses each candidate's released B3LYP thermal correction, so this is "
            "a controlled fixed-geometry sensitivity analysis, not a solvent-specific TS reoptimization."
        ),
        "routes": route_results,
    }


def main() -> int:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=repository_root / "supplementary" / "case1.json",
    )
    parser.add_argument(
        "--run-minichem-validation",
        action="store_true",
        help="Run typed MiniChem Gaussian single-point Actions through the configured MCP endpoint",
    )
    parser.add_argument(
        "--run-gaussian-exact",
        action="store_true",
        help="Run the paper's exact reported single-point route directly with copied Gaussian",
    )
    parser.add_argument(
        "--run-gaussian-sensitivity",
        action="store_true",
        help="Run fixed-geometry solvent and functional sensitivity for TS-2a/TS-2b-1",
    )
    parser.add_argument(
        "--run-minichem-hessians",
        action="store_true",
        help="Run typed Gaussian Hessian and projected-vibration Actions for selected candidates",
    )
    parser.add_argument(
        "--run-minichem-ts-optimization",
        action="store_true",
        help=(
            "Reoptimize selected released TS geometries, then run typed Gaussian Hessian "
            "and projected-vibration Actions"
        ),
    )
    parser.add_argument(
        "--run-minichem-irc",
        action="store_true",
        help="Run typed Gaussian forward/backward IRC Actions for selected candidates",
    )
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:9010/mcp",
        help="历史论文复算使用的 legacy/general MiniChem 地址；不是当前 research_task 入口",
    )
    parser.add_argument(
        "--gaussian-candidates",
        nargs="+",
        default=[PAPER_MAJOR_TS, PAPER_MINOR_TS],
    )
    parser.add_argument("--cpu-cores", type=int, default=4)
    parser.add_argument("--memory-mb", type=int, default=8192)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--gaussian-executable",
        type=Path,
        default=(
            repository_root
            / "third_party/minichem_toolbox/.mini_software_cache/gaussian/g16/install/g16/g16"
        ),
    )
    parser.add_argument(
        "--gaussian-work-dir",
        type=Path,
        default=repository_root / "local_data/runtime/case1_gaussian_exact",
    )
    parser.add_argument(
        "--minichem-workspace",
        type=Path,
        default=repository_root / "local_data/runtime/case1_minichem_workspace",
    )
    parser.add_argument(
        "--minichem-result-dir",
        type=Path,
        default=repository_root / "local_data/runtime/case1_minichem_results",
    )
    parser.add_argument("--gaussian-timeout-seconds", type=float, default=21600.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path; runtime artifacts should remain outside Git",
    )
    args = parser.parse_args()

    if args.gaussian_candidates == ["all"]:
        data_preview = json.loads(args.data.read_text(encoding="utf-8"))
        args.gaussian_candidates = sorted(data_preview["sp"])

    if (
        args.cpu_cores < 1
        or args.memory_mb < 128
        or args.workers < 1
        or args.gaussian_timeout_seconds <= 0
    ):
        parser.error("cpu-cores and workers must be positive; memory-mb must be at least 128")

    report, payload = reproduce_supplementary_data(args.data)
    requested_stage_keys: list[str] = []
    if args.run_minichem_validation:
        requested_stage_keys.append("minichem_gaussian_validation")
        report["minichem_gaussian_validation"] = run_gaussian_validation(
            payload,
            endpoint=args.endpoint,
            candidates=args.gaussian_candidates,
            cpu_cores=args.cpu_cores,
            memory_mb=args.memory_mb,
            workers=args.workers,
        )
    if args.run_gaussian_exact:
        requested_stage_keys.append("exact_gaussian_validation")
        report["exact_gaussian_validation"] = run_exact_gaussian_validation(
            payload,
            executable=args.gaussian_executable,
            work_root=args.gaussian_work_dir,
            candidates=args.gaussian_candidates,
            cpu_cores=args.cpu_cores,
            memory_mb=args.memory_mb,
            workers=args.workers,
            timeout_seconds=args.gaussian_timeout_seconds,
        )
    if args.run_gaussian_sensitivity:
        requested_stage_keys.append("gaussian_sensitivity_validation")
        report["gaussian_sensitivity_validation"] = run_gaussian_sensitivity_validation(
            payload,
            executable=args.gaussian_executable,
            work_root=args.gaussian_work_dir,
            cpu_cores=args.cpu_cores,
            memory_mb=args.memory_mb,
            workers=args.workers,
            timeout_seconds=args.gaussian_timeout_seconds,
        )
    if args.run_minichem_hessians:
        requested_stage_keys.append("minichem_hessian_validation")
        report["minichem_hessian_validation"] = run_minichem_hessian_validation(
            payload,
            repository_root=repository_root,
            workspace=args.minichem_workspace,
            result_root=args.minichem_result_dir,
            candidates=args.gaussian_candidates,
            cpu_cores=args.cpu_cores,
            memory_mb=args.memory_mb,
            workers=args.workers,
            timeout_seconds=args.gaussian_timeout_seconds,
        )
    if args.run_minichem_ts_optimization:
        requested_stage_keys.append("minichem_ts_optimization_validation")
        report["minichem_ts_optimization_validation"] = (
            run_minichem_ts_optimization_validation(
                payload,
                repository_root=repository_root,
                workspace=args.minichem_workspace,
                result_root=args.minichem_result_dir,
                candidates=args.gaussian_candidates,
                cpu_cores=args.cpu_cores,
                memory_mb=args.memory_mb,
                workers=args.workers,
                timeout_seconds=args.gaussian_timeout_seconds,
            )
        )
    if args.run_minichem_irc:
        requested_stage_keys.append("minichem_irc_validation")
        report["minichem_irc_validation"] = run_minichem_irc_validation(
            payload,
            repository_root=repository_root,
            workspace=args.minichem_workspace,
            result_root=args.minichem_result_dir,
            candidates=args.gaussian_candidates,
            cpu_cores=args.cpu_cores,
            memory_mb=args.memory_mb,
            workers=args.workers,
            timeout_seconds=args.gaussian_timeout_seconds,
        )
    if requested_stage_keys:
        report["requested_stage_summary"] = _requested_stage_summary(
            report, requested_stage_keys
        )
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    if report["status"] == "failed":
        return 1
    if requested_stage_keys and not report["requested_stage_summary"][
        "all_requested_stages_completed"
    ]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
