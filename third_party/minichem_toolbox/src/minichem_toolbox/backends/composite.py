"""Helpers for composite scientific actions with Agent-selected component backends."""

from __future__ import annotations

import math
from typing import Any

from ..catalog import backend_specs
from ..runtime import invoke_worker
from .common import (
    command_artifacts,
    module_version,
    output_directory,
    partial_success,
    relative_workspace_path,
    request_parts,
    structure_dict,
    structure_from_atoms,
    success,
    write_json,
)


HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANGSTROM = 0.529177210903


def _component_settings(request: dict[str, Any], action_id: str) -> dict[str, Any]:
    settings = dict(request.get("action_settings") or {})
    values = settings.get("calculator_action_settings")
    if not isinstance(values, dict):
        raise ValueError("action_settings.calculator_action_settings must be an action-keyed mapping")
    selected = values.get(action_id)
    if not isinstance(selected, dict):
        raise ValueError(
            f"calculator_action_settings must explicitly provide a {action_id!r} mapping"
        )
    return dict(selected)


def invoke_calculator_component(
    request: dict[str, Any],
    action_id: str,
    structure: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Invoke exactly the calculator backend selected by the Agent for one atomic property."""

    component_backends = dict(request.get("component_backends") or {})
    calculator_id = component_backends.get("calculator")
    if not calculator_id:
        raise ValueError("component_backends.calculator is required")
    specifications = backend_specs()
    if calculator_id not in specifications:
        raise ValueError(f"Unknown calculator component backend: {calculator_id}")
    calculator = specifications[calculator_id]
    if action_id not in calculator.capabilities:
        raise ValueError(f"Calculator backend {calculator_id!r} does not support {action_id!r}")
    method = dict(request.get("method_spec") or {})
    calculator_method = method.get("calculator_method")
    if not isinstance(calculator_method, dict):
        raise ValueError("method_spec.calculator_method must be an explicit mapping")
    nested_request = {
        "backend_id": calculator_id,
        "component_backends": {},
        "source_id": None,
        "inputs": {"structure": structure},
        "method_spec": dict(calculator_method),
        "action_settings": _component_settings(request, action_id),
        "resource_limits": dict(request.get("resource_limits") or {}),
    }
    worker = invoke_worker(
        runtime=calculator.runtime,
        payload={
            "action_id": action_id,
            "backend_id": calculator_id,
            "request": nested_request,
        },
        timeout_seconds=int((request.get("resource_limits") or {}).get("walltime_seconds", 1800)),
    )
    if worker.get("status") not in {"success", "partial_success"}:
        error = worker.get("error") or {}
        raise RuntimeError(
            f"Component {calculator_id}/{action_id} failed: "
            f"{error.get('code', worker.get('status'))}: {error.get('message', error)}"
        )
    result = worker.get("result")
    if not isinstance(result, dict):
        raise RuntimeError(f"Component {calculator_id}/{action_id} returned no structured result")
    provenance = {
        "calculator_backend": calculator_id,
        "calculator_runtime": calculator.runtime,
        "calculator_action": action_id,
        "calculator_backend_version": worker.get("backend_version"),
        "calculator_status": worker.get("status"),
        "calculator_artifact_files": list(worker.get("artifact_files") or []),
        "calculator_provenance": dict(worker.get("provenance") or {}),
    }
    return result, provenance


def energy_hartree(value: dict[str, Any]) -> float:
    if value.get("energy") is not None:
        energy = float(value["energy"])
        unit = str(value.get("unit") or value.get("energy_unit") or "").strip().lower()
    elif value.get("energy_hartree") is not None:
        energy = float(value["energy_hartree"])
        unit = "hartree"
    else:
        raise RuntimeError("Calculator energy result contains no scalar energy")
    if not math.isfinite(energy):
        raise RuntimeError("Calculator returned a non-finite energy")
    if unit in {"hartree", "eh", "a.u.", "au"}:
        return energy
    if unit in {"ev", "electronvolt", "electronvolts"}:
        return energy / HARTREE_TO_EV
    raise RuntimeError(f"Unsupported calculator energy unit for composite optimization: {unit!r}")


def forces_ev_per_angstrom(value: dict[str, Any]) -> list[list[float]]:
    import numpy as np

    forces = np.asarray(value.get("forces"), dtype=float)
    if forces.ndim != 2 or forces.shape[1] != 3 or not np.all(np.isfinite(forces)):
        raise RuntimeError("Calculator force result must be a finite Nx3 matrix")
    unit = str(value.get("unit") or "").strip().lower().replace("å", "angstrom")
    if unit in {"ev/angstrom", "ev/ang", "ev/a"}:
        return forces.tolist()
    if unit in {"hartree/bohr", "eh/bohr"}:
        return (forces * HARTREE_TO_EV / BOHR_TO_ANGSTROM).tolist()
    raise RuntimeError(f"Unsupported calculator force unit for composite optimization: {unit!r}")


def execute_sella(action_id: str, request: dict[str, Any]) -> dict[str, Any]:
    """Run one Sella order-0/order-1 search with an Agent-selected calculator."""

    import numpy as np
    from ase import Atoms
    from ase.calculators.calculator import Calculator, all_changes
    from ase.io import write
    from sella import Sella

    if action_id not in {"optimize_geometry", "locate_transition_state"}:
        raise ValueError(f"The validated Sella adapter does not implement {action_id}")
    inputs, _method, settings = request_parts(request)
    structure_key = "structure" if action_id == "optimize_geometry" else "initial_guess"
    original = structure_dict(inputs[structure_key])
    atom_records = original.get("atoms") or []
    if not atom_records or any("position_angstrom" not in atom for atom in atom_records):
        raise ValueError("Sella requires a non-empty molecular structure with coordinates")
    if any(bool(value) for value in original.get("pbc", [False, False, False])):
        raise ValueError("The validated Sella composite adapter is molecular/non-periodic")
    atoms = Atoms(
        symbols=[str(atom["element"]) for atom in atom_records],
        positions=[atom["position_angstrom"] for atom in atom_records],
        pbc=False,
    )

    class AgentSelectedASECalculator(Calculator):
        implemented_properties = ["energy", "forces"]

        def __init__(self):
            super().__init__()
            self.evaluations: list[dict[str, Any]] = []

        def calculate(self, calculation_atoms=None, properties=("energy", "forces"), system_changes=all_changes):
            super().calculate(calculation_atoms, properties, system_changes)
            value = calculation_atoms if calculation_atoms is not None else self.atoms
            structure = {
                **{key: item for key, item in original.items() if key != "atoms"},
                "atoms": [
                    {
                        **{
                            key: item
                            for key, item in atom_records[index].items()
                            if key != "position_angstrom"
                        },
                        "position_angstrom": value.positions[index].tolist(),
                    }
                    for index in range(len(value))
                ],
            }
            force_result, force_provenance = invoke_calculator_component(
                request, "calculate_forces", structure
            )
            energy_result, energy_provenance = invoke_calculator_component(
                request, "calculate_energy", structure
            )
            energy_ev = energy_hartree(energy_result) * HARTREE_TO_EV
            forces = np.asarray(forces_ev_per_angstrom(force_result), dtype=float)
            if forces.shape != (len(value), 3):
                raise RuntimeError("Calculator force matrix does not match the Sella structure")
            self.results = {"energy": energy_ev, "forces": forces}
            self.evaluations.append(
                {
                    "evaluation_index": len(self.evaluations),
                    "energy_ev": float(energy_ev),
                    "maximum_force_ev_per_angstrom": float(
                        np.max(np.linalg.norm(forces, axis=1))
                    ),
                    "energy_component": energy_provenance,
                    "force_component": force_provenance,
                }
            )

    directory = output_directory(action_id, "sella")
    calculator = AgentSelectedASECalculator()
    atoms.calc = calculator
    internal = settings["internal_coordinates"]
    if not isinstance(internal, bool):
        raise ValueError("internal_coordinates must be an explicit boolean")
    maximum_steps = int(settings["max_steps"])
    force_threshold = float(settings["force_threshold_ev_per_angstrom"])
    if maximum_steps < 1 or force_threshold <= 0:
        raise ValueError("max_steps and force_threshold_ev_per_angstrom must be positive")
    stationary_order = 0 if action_id == "optimize_geometry" else 1
    optimizer = Sella(
        atoms,
        order=stationary_order,
        internal=internal,
        logfile=str(directory / "sella.log"),
        trajectory=str(directory / "optimization.traj"),
        delta0=float(settings["initial_trust_radius"]),
        eta=float(settings["minimum_model_quality"]),
        gamma=float(settings["finite_difference_step"]),
        threepoint=bool(settings["three_point_differences"]),
        nsteps_per_diag=int(settings["steps_per_diagonalization"]),
        diag_every_n=int(settings["diagonalization_interval"]),
        allow_fragments=bool(settings["allow_fragments"]),
        refine_initial_hessian=int(settings["refine_initial_hessian_iterations"]),
    )
    converged = bool(optimizer.run(fmax=force_threshold, steps=maximum_steps))
    optimized_path = directory / (
        "optimized.xyz" if action_id == "optimize_geometry" else "transition_state_candidate.xyz"
    )
    write(str(optimized_path), atoms)
    evaluations_path = write_json(directory, "component_evaluations.json", calculator.evaluations)
    semantic_paths = {
        relative_workspace_path(optimized_path),
        relative_workspace_path(evaluations_path),
    }
    artifacts = [
        item for item in command_artifacts(directory) if item["path"] not in semantic_paths
    ]
    artifacts.extend(
        [
            {
                "path": relative_workspace_path(optimized_path),
                "semantic_type": "AtomicStructure",
                "media_type": "chemical/x-xyz",
            },
            {
                "path": relative_workspace_path(evaluations_path),
                "semantic_type": "ComponentEvaluationTrace",
                "media_type": "application/json",
            },
        ]
    )
    final_structure = structure_from_atoms(atoms)
    final_structure["charge"] = int(original.get("charge", 0))
    final_structure["multiplicity"] = int(original.get("multiplicity", 1))
    result = {
        "structure": final_structure,
        "converged": converged,
        "optimizer": "sella",
        "stationary_point_order": stationary_order,
        "calculator_backend": request["component_backends"]["calculator"],
        "evaluation_count": len(calculator.evaluations),
        "final_energy_ev": (
            calculator.evaluations[-1]["energy_ev"] if calculator.evaluations else None
        ),
    }
    if action_id == "locate_transition_state":
        result["validation_required"] = (
            "Call calculate_hessian and derive_vibrational_modes explicitly to verify exactly "
            "one imaginary mode, then trace_intrinsic_reaction_coordinate if appropriate."
        )
    values = {
        "artifact_files": artifacts,
        "backend_version": module_version("sella"),
        "provenance": {
            "calculator_backend": request["component_backends"]["calculator"],
            "component_evaluation_count": len(calculator.evaluations),
            "automatic_component_selection": False,
        },
    }
    if not converged:
        return partial_success(
            result,
            warnings=["Sella reached max_steps before the requested force threshold."],
            **values,
        )
    return success(result, **values)
