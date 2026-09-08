"""RDKit MMFF geometry optimization with explicit distance constraints."""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping

from .common import (
    module_version,
    output_directory,
    partial_success,
    relative_workspace_path,
    request_parts,
    structure_dict,
    success,
    unsupported,
    write_xyz,
)


ACTION_ID = "optimize_geometry"
BACKEND_ID = "rdkit_mmff"
HARD_MAX_ITERATIONS = 100_000
HARD_MAX_CONSTRAINTS = 256


def _finite_positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return converted


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return value


def _bond_type(chem: Any, order: float):
    choices = {
        1.0: chem.BondType.SINGLE,
        1.5: chem.BondType.AROMATIC,
        2.0: chem.BondType.DOUBLE,
        3.0: chem.BondType.TRIPLE,
    }
    try:
        return choices[float(order)]
    except KeyError as exc:
        raise ValueError(f"Unsupported molecular bond order for MMFF: {order}") from exc


def _molecule_from_structure(structure: dict[str, Any]):
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    atoms = structure.get("atoms")
    if not isinstance(atoms, list) or not atoms:
        raise ValueError("MMFF optimization requires a non-empty atoms array")
    editable = Chem.RWMol()
    for index, record in enumerate(atoms):
        if not isinstance(record, Mapping):
            raise ValueError(f"atoms[{index}] must be a mapping")
        symbol = str(record.get("element") or record.get("symbol") or "").strip()
        if not symbol:
            raise ValueError(f"atoms[{index}] has no element")
        atom = Chem.Atom(symbol)
        atom.SetFormalCharge(int(record.get("formal_charge", 0)))
        map_number = record.get("atom_map_number")
        if map_number is not None:
            atom.SetAtomMapNum(
                _integer(map_number, f"atoms[{index}].atom_map_number", minimum=1)
            )
        editable.AddAtom(atom)
    for index, record in enumerate(structure.get("bonds") or []):
        if not isinstance(record, Mapping):
            raise ValueError(f"bonds[{index}] must be a mapping")
        first = _integer(record.get("atom_index_a"), f"bonds[{index}].atom_index_a")
        second = _integer(record.get("atom_index_b"), f"bonds[{index}].atom_index_b")
        if first >= len(atoms) or second >= len(atoms) or first == second:
            raise ValueError(f"bonds[{index}] contains invalid atom indices")
        editable.AddBond(
            first,
            second,
            _bond_type(Chem, float(record.get("bond_order", 1.0))),
        )
    molecule = editable.GetMol()
    Chem.SanitizeMol(molecule)
    conformer = Chem.Conformer(len(atoms))
    for index, record in enumerate(atoms):
        coordinates = record.get("position_angstrom")
        if not isinstance(coordinates, list) or len(coordinates) != 3:
            raise ValueError(f"atoms[{index}] requires position_angstrom")
        values = [float(value) for value in coordinates]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"atoms[{index}] coordinates must be finite")
        conformer.SetAtomPosition(index, Point3D(*values))
    molecule.AddConformer(conformer, assignId=True)
    return molecule


def _constraint_indices(
    record: Mapping[str, Any], structure: Mapping[str, Any], offset: int
) -> tuple[int, int]:
    raw_indices = record.get("atom_indices")
    if isinstance(raw_indices, list) and len(raw_indices) == 2:
        first = _integer(raw_indices[0], f"constraints[{offset}].atom_indices[0]")
        second = _integer(raw_indices[1], f"constraints[{offset}].atom_indices[1]")
        if first == second or max(first, second) >= len(structure["atoms"]):
            raise ValueError(f"constraints[{offset}] atom_indices are invalid")
        return first, second
    raw_maps = record.get("atom_maps")
    if not isinstance(raw_maps, list) or len(raw_maps) != 2:
        raise ValueError(
            f"constraints[{offset}] requires atom_indices or atom_maps with two values"
        )
    requested = [
        _integer(value, f"constraints[{offset}].atom_maps", minimum=1)
        for value in raw_maps
    ]
    locations: dict[int, int] = {}
    for atom_index, atom in enumerate(structure["atoms"]):
        map_number = atom.get("atom_map_number")
        if map_number is not None:
            parsed = _integer(
                map_number, f"atoms[{atom_index}].atom_map_number", minimum=1
            )
            if parsed in locations:
                raise ValueError(f"atom map {parsed} is duplicated in the structure")
            locations[parsed] = atom_index
    if requested[0] not in locations or requested[1] not in locations:
        raise ValueError(f"constraints[{offset}] references an absent atom map")
    return locations[requested[0]], locations[requested[1]]


def optimize_geometry(request: dict[str, Any]) -> dict[str, Any]:
    from rdkit.Chem import AllChem

    inputs, method, settings = request_parts(request)
    original = structure_dict(inputs["structure"])
    if any(bool(value) for value in original.get("pbc", [False, False, False])):
        raise ValueError("RDKit MMFF optimization only supports non-periodic molecules")
    variant = str(method.get("force_field") or "").strip().lower()
    if variant not in {"mmff94", "mmff94s"}:
        raise ValueError("method_spec.force_field must be mmff94 or mmff94s")
    rdkit_variant = {"mmff94": "MMFF94", "mmff94s": "MMFF94s"}[variant]
    maximum = _integer(settings.get("max_iterations"), "max_iterations", minimum=1)
    if maximum > HARD_MAX_ITERATIONS:
        raise ValueError(f"max_iterations exceeds hard limit {HARD_MAX_ITERATIONS}")
    force_tolerance = _finite_positive(
        settings.get("force_tolerance"), "force_tolerance"
    )
    energy_tolerance = _finite_positive(
        settings.get("energy_tolerance"), "energy_tolerance"
    )
    molecule = _molecule_from_structure(original)
    properties = AllChem.MMFFGetMoleculeProperties(
        molecule, mmffVariant=rdkit_variant
    )
    if properties is None:
        raise ValueError("RDKit MMFF parameters are unavailable for this molecular graph")
    force_field = AllChem.MMFFGetMoleculeForceField(
        molecule, properties, confId=0, ignoreInterfragInteractions=False
    )
    if force_field is None:
        raise RuntimeError("RDKit could not construct the requested MMFF force field")
    constraints = inputs.get("constraints") or []
    if not isinstance(constraints, list):
        raise ValueError("inputs.constraints must be an array of distance constraints")
    if len(constraints) > HARD_MAX_CONSTRAINTS:
        raise ValueError(f"inputs.constraints cannot exceed {HARD_MAX_CONSTRAINTS}")
    normalized_constraints = []
    for offset, record in enumerate(constraints):
        if not isinstance(record, Mapping):
            raise ValueError(f"constraints[{offset}] must be a mapping")
        first, second = _constraint_indices(record, original, offset)
        minimum = _finite_positive(
            record.get("minimum_distance_angstrom"),
            f"constraints[{offset}].minimum_distance_angstrom",
        )
        maximum_distance = _finite_positive(
            record.get("maximum_distance_angstrom"),
            f"constraints[{offset}].maximum_distance_angstrom",
        )
        if maximum_distance < minimum:
            raise ValueError(f"constraints[{offset}] maximum distance is below minimum")
        force_constant = _finite_positive(
            record.get("force_constant_kcal_mol_angstrom2"),
            f"constraints[{offset}].force_constant_kcal_mol_angstrom2",
        )
        force_field.MMFFAddDistanceConstraint(
            first, second, False, minimum, maximum_distance, force_constant
        )
        normalized_constraints.append(
            {
                "atom_indices": [first, second],
                "minimum_distance_angstrom": minimum,
                "maximum_distance_angstrom": maximum_distance,
                "force_constant_kcal_mol_angstrom2": force_constant,
            }
        )
    convergence_code = int(
        force_field.Minimize(
            maxIts=maximum,
            forceTol=force_tolerance,
            energyTol=energy_tolerance,
        )
    )
    energy = float(force_field.CalcEnergy())
    if not math.isfinite(energy):
        raise RuntimeError("RDKit MMFF returned a non-finite energy")
    optimized = copy.deepcopy(original)
    conformer = molecule.GetConformer(0)
    for index, atom in enumerate(optimized["atoms"]):
        position = conformer.GetAtomPosition(index)
        atom["position_angstrom"] = [float(position.x), float(position.y), float(position.z)]
    directory = output_directory(ACTION_ID, BACKEND_ID)
    path = write_xyz(optimized, directory / "optimized.xyz")
    result = {
        "structure": optimized,
        "converged": convergence_code == 0,
        "convergence_code": convergence_code,
        "energy_kcal_mol": energy,
        "force_field": variant,
        "constraint_count": len(normalized_constraints),
        "constraints": normalized_constraints,
        "max_iterations": maximum,
    }
    kwargs = {
        "artifact_files": [
            {
                "path": relative_workspace_path(path),
                "semantic_type": "AtomicStructure",
                "media_type": "chemical/x-xyz",
            }
        ],
        "backend_version": module_version("rdkit"),
        "provenance": {
            "engine": "RDKit MMFF",
            "coordinates_modified": True,
            "molecular_graph_preserved": True,
            "constraints_explicit": True,
        },
    }
    if convergence_code == 0:
        return success(result, **kwargs)
    return partial_success(
        result,
        warnings=["RDKit MMFF reached max_iterations before convergence."],
        **kwargs,
    )


def execute(action_id: str, backend_id: str, request: dict[str, Any]) -> dict[str, Any]:
    if action_id == ACTION_ID and backend_id == BACKEND_ID:
        return optimize_geometry(request)
    return unsupported(f"Unsupported RDKit MMFF action/backend: {action_id}/{backend_id}")
