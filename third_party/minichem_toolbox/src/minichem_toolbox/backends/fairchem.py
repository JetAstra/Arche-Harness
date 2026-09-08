"""FairChem UMA molecular geometry optimization backend."""

from __future__ import annotations

import math
import os
import sys
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Iterator

from .common import (
    ase_atoms,
    atoms_and_coordinates,
    command_artifacts,
    module_version,
    output_directory,
    partial_success,
    relative_workspace_path,
    request_parts,
    structure_dict,
    structure_from_atoms,
    success,
    unavailable,
    unsupported,
)


ACTIONS = {"optimize_geometry"}
HARD_MAX_OPTIMIZATION_STEPS = 100_000


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        converted = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{label} must be an integer")
    return converted


def _positive_integer(value: Any, label: str, *, maximum: int | None = None) -> int:
    converted = _integer(value, label)
    if converted < 1:
        raise ValueError(f"{label} must be a positive integer")
    if maximum is not None and converted > maximum:
        raise ValueError(f"{label} cannot exceed the hard limit {maximum}")
    return converted


@contextmanager
def _model_download_policy(allow_download: bool) -> Iterator[None]:
    """Prevent an implicit Hugging Face download unless the request permits it."""

    names = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    previous = {name: os.environ.get(name) for name in names}
    hub_constants = sys.modules.get("huggingface_hub.constants")
    previous_hub_offline = (
        getattr(hub_constants, "HF_HUB_OFFLINE", None)
        if hub_constants is not None
        else None
    )
    if not allow_download:
        for name in names:
            os.environ[name] = "1"
        if hub_constants is not None:
            hub_constants.HF_HUB_OFFLINE = True
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        if hub_constants is not None:
            hub_constants.HF_HUB_OFFLINE = previous_hub_offline


def _apply_constraints(atoms: Any, constraints: Any) -> None:
    if constraints in (None, [], {}):
        return
    if not isinstance(constraints, list):
        raise ValueError("inputs.constraints must be an array of explicit constraint records")

    from ase.constraints import FixAtoms, FixCartesian

    ase_constraints = []
    atom_count = len(atoms)
    for offset, item in enumerate(constraints):
        if not isinstance(item, dict):
            raise ValueError(f"inputs.constraints[{offset}] must be a mapping")
        kind = str(item.get("type") or "").strip().lower()
        if kind == "fix_atoms":
            indices = [int(index) for index in item.get("atom_indices") or []]
            if not indices:
                raise ValueError(
                    f"inputs.constraints[{offset}].atom_indices must be non-empty"
                )
            if any(index < 0 or index >= atom_count for index in indices):
                raise ValueError(f"inputs.constraints[{offset}] contains an invalid atom index")
            ase_constraints.append(FixAtoms(indices=indices))
        elif kind == "fix_cartesian":
            index = int(item.get("atom_index", -1))
            mask = item.get("mask")
            if index < 0 or index >= atom_count:
                raise ValueError(f"inputs.constraints[{offset}].atom_index is invalid")
            if not isinstance(mask, list) or len(mask) != 3 or not all(
                isinstance(value, bool) for value in mask
            ):
                raise ValueError(
                    f"inputs.constraints[{offset}].mask must contain three booleans"
                )
            ase_constraints.append(FixCartesian(index, mask=mask))
        else:
            raise ValueError(
                f"Unsupported inputs.constraints[{offset}].type={kind!r}; "
                "use fix_atoms or fix_cartesian"
            )
    atoms.set_constraint(ase_constraints)


def _preserve_structure_metadata(
    optimized: dict[str, Any], original: dict[str, Any]
) -> dict[str, Any]:
    """Carry graph identity through an optimization that only changes coordinates."""

    result = dict(optimized)
    coordinate_fields = {"atoms", "positions", "coordinates", "cell", "lattice", "pbc"}
    for name, value in original.items():
        if name not in coordinate_fields:
            result[name] = deepcopy(value)
    original_atoms = original.get("atoms") or []
    for index, atom in enumerate(result.get("atoms") or []):
        if index >= len(original_atoms) or not isinstance(original_atoms[index], dict):
            continue
        for name, value in original_atoms[index].items():
            if name not in {"element", "symbol", "position", "position_angstrom"}:
                atom[name] = value
    return result


def _build_calculator(method: dict[str, Any], *, allow_download: bool) -> Any:
    from fairchem.core import FAIRChemCalculator, pretrained_mlip

    model = str(method["model"]).strip()
    task_name = str(method["task_name"]).strip()
    device = str(method["device"]).strip().lower()
    if not model:
        raise ValueError("method_spec.model cannot be empty")
    if not task_name:
        raise ValueError("method_spec.task_name cannot be empty")
    if not device:
        raise ValueError("method_spec.device cannot be empty")
    with _model_download_policy(allow_download):
        predictor = pretrained_mlip.get_predict_unit(model, device=device)
    return FAIRChemCalculator(predictor, task_name=task_name)


def optimize_geometry(request: dict[str, Any]) -> dict[str, Any]:
    try:
        from ase.optimize import BFGS, FIRE, LBFGS
    except (ImportError, ModuleNotFoundError) as exc:
        return unavailable(
            f"FairChem UMA optimization dependencies are unavailable: {exc}",
            install="Install fairchem-core, torch, and ase in the fairchem runtime.",
        )

    inputs, method, settings = request_parts(request)
    allow_download = method.get("allow_model_download")
    if not isinstance(allow_download, bool):
        raise ValueError("method_spec.allow_model_download must be an explicit boolean")
    device = str(method["device"]).strip().lower()
    requested_gpus = int((request.get("resource_limits") or {}).get("gpu_count") or 0)
    if device.startswith("cuda") and requested_gpus < 1:
        raise ValueError(
            "A CUDA FairChem device requires resource_limits.gpu_count >= 1"
        )

    force_threshold = float(settings["fmax_ev_per_angstrom"])
    max_steps = _positive_integer(
        settings["max_steps"],
        "action_settings.max_steps",
        maximum=HARD_MAX_OPTIMIZATION_STEPS,
    )
    optimizer_name = str(settings["optimizer"]).strip().lower()
    if not math.isfinite(force_threshold) or force_threshold <= 0:
        raise ValueError(
            "action_settings.fmax_ev_per_angstrom must be a finite positive number"
        )
    optimizers = {"bfgs": BFGS, "lbfgs": LBFGS, "fire": FIRE}
    if optimizer_name not in optimizers:
        raise ValueError("action_settings.optimizer must be bfgs, lbfgs, or fire")

    original = structure_dict(inputs["structure"])
    if "charge" not in original or "multiplicity" not in original:
        raise ValueError(
            "AtomicStructure must explicitly declare charge and multiplicity for "
            "FairChem OMol optimization"
        )
    if any(bool(value) for value in original.get("pbc", [False, False, False])):
        raise ValueError("FairChem OMol optimization does not accept periodic structures")
    charge = _integer(original["charge"], "AtomicStructure.charge")
    multiplicity = _positive_integer(
        original["multiplicity"], "AtomicStructure.multiplicity"
    )
    atoms = ase_atoms(original)
    atoms.info["charge"] = charge
    # FairChem OMol defines spin as multiplicity, not the number of unpaired electrons.
    atoms.info["spin"] = multiplicity
    _apply_constraints(atoms, inputs.get("constraints"))

    directory = output_directory("optimize_geometry", "fairchem_uma")
    log_path = directory / "optimization.log"
    trajectory_path = directory / "optimization.traj"
    try:
        atoms.calc = _build_calculator(method, allow_download=allow_download)
    except (ImportError, ModuleNotFoundError) as exc:
        return unavailable(
            f"FairChem UMA is unavailable: {exc}",
            install="Install fairchem-core and its compatible torch build.",
        )
    except Exception as exc:
        if not allow_download:
            return unavailable(
                "FairChem model initialization failed while network/model downloads were "
                f"disabled: {type(exc).__name__}: {exc}",
                install=(
                    "Verify the FairChem/torch installation and selected model cache. "
                    "Pre-populate the checkpoint, or explicitly permit a download only "
                    "after checking the runtime compatibility."
                ),
            )
        raise

    optimizer = optimizers[optimizer_name](
        atoms,
        logfile=str(log_path),
        trajectory=str(trajectory_path),
    )
    converged = bool(optimizer.run(fmax=force_threshold, steps=max_steps))
    energy_ev = float(atoms.get_potential_energy())
    if not math.isfinite(energy_ev):
        raise RuntimeError("FairChem returned a non-finite optimized energy")
    optimized = _preserve_structure_metadata(structure_from_atoms(atoms), original)
    atoms_and_coordinates(optimized)
    result = {
        "structure": optimized,
        "converged": converged,
        "energy_ev": energy_ev,
        "optimizer": optimizer_name,
        "optimization_steps": int(optimizer.get_number_of_steps()),
        "force_threshold_ev_per_angstrom": force_threshold,
        "maximum_steps": max_steps,
        "model": str(method["model"]),
        "task_name": str(method["task_name"]),
        "device": device,
        "allow_model_download": allow_download,
        "trajectory_path": relative_workspace_path(trajectory_path),
        "log_path": relative_workspace_path(log_path),
    }
    kwargs = {
        "artifact_files": command_artifacts(directory),
        "backend_version": module_version("fairchem-core") or module_version("fairchem"),
        "provenance": {
            "calculator": "FAIRChemCalculator",
            "model_download_permitted": allow_download,
            "atomic_graph_preserved": True,
        },
    }
    if converged:
        return success(result, **kwargs)
    return partial_success(
        result,
        warnings=[
            "UMA geometry optimization reached max_steps before satisfying the requested "
            "force threshold; the returned structure is not declared converged."
        ],
        **kwargs,
    )


def execute(action_id: str, backend_id: str, request: dict[str, Any]) -> dict[str, Any]:
    if action_id == "optimize_geometry" and backend_id == "fairchem_uma":
        return optimize_geometry(request)
    return unsupported(f"Unsupported FairChem action/backend combination: {action_id}/{backend_id}")
