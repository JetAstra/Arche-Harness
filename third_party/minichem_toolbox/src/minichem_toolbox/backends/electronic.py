"""Molecular electronic-structure and derived-property actions."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any

from ..artifacts import resolve_workspace_path
from ..orca_contract import normalize_orca_method_basis
from ..parameter_specs import ORCA_DENSITY_DEFAULT_MAXCORE_MB
from .common import (
    ase_atoms,
    atoms_and_coordinates,
    atom_spec,
    command_artifacts,
    module_version,
    output_directory,
    partial_success,
    relative_workspace_path,
    request_parts,
    resolve_input_file,
    run_external,
    structure_dict,
    structure_from_atoms,
    success,
    unavailable,
    unwrap_artifact,
    unsupported,
    write_json,
    write_xyz,
)
from .composite import (
    BOHR_TO_ANGSTROM,
    HARTREE_TO_EV,
    energy_hartree,
    execute_sella,
    forces_ev_per_angstrom,
    invoke_calculator_component,
)
from .goodvibes import execute as execute_goodvibes
from .quantum_legacy import gaussian as _gaussian


ACTIONS = {
    "generate_gaussian_code",
    "calculate_energy", "calculate_forces", "calculate_hessian", "optimize_geometry",
    "calculate_dipole_moment", "calculate_atomic_charges",
    "derive_vibrational_modes", "derive_ir_spectrum", "derive_uv_vis_spectrum",
    "derive_raman_spectrum", "derive_roa_spectrum", "derive_vcd_spectrum",
    "derive_ecd_spectrum", "derive_nmr_spectrum", "derive_thermochemistry",
    "calculate_bond_orders", "calculate_excited_states",
    "scan_thermochemistry_temperature",
    "analyze_thermochemical_ensemble", "validate_thermochemistry_inputs",
    "analyze_free_energy_selectivity", "analyze_energetic_span",
    "calculate_electron_isodensity_surface",
    "analyze_non_covalent_interactions",
}


_MACE_INSTALLED_MODEL_ALIASES = {
    "medium-mpa-0": "mace/macempa0mediummodel",
    "mace-mpa-0-medium": "mace/macempa0mediummodel",
    "medium": "mace/20231203mace128L1_epoch199model",
    "mace-mp-0-medium": "mace/20231203mace128L1_epoch199model",
}
_MACE_0_3_16_MODEL_NAMES = (
    "small",
    "medium",
    "large",
    "small-0b",
    "medium-0b",
    "small-0b2",
    "medium-0b2",
    "large-0b2",
    "medium-0b3",
    "medium-mpa-0",
    "small-omat-0",
    "medium-omat-0",
    "mace-matpes-pbe-0",
    "mace-matpes-r2scan-0",
    "mh-0",
    "mh-1",
)


def _mace_model_cache_root() -> Path:
    configured_cache = os.environ.get("RESEARCHCHEMBENCH_MODEL_CACHE", "").strip()
    return (
        Path(configured_cache).expanduser().resolve()
        if configured_cache
        else (Path(__file__).resolve().parents[4] / ".model_cache").resolve()
    )


def _resolve_mace_model(model_value: Any, *, allow_download: bool) -> str:
    """Resolve exactly the MACE model named by the Agent.

    Installed aliases map to their pinned local cache files.  Unknown labels are
    rejected before MACE interprets them as URLs or paths; downloads are only
    possible when the request explicitly enables them.
    """

    model = str(model_value).strip()
    if not model:
        raise ValueError("MACE method_spec.model cannot be empty")

    alias_path = _MACE_INSTALLED_MODEL_ALIASES.get(model.casefold())
    if alias_path is not None:
        candidate = _mace_model_cache_root() / alias_path
        if candidate.is_file():
            return str(candidate)
        if allow_download:
            return model.casefold()
        raise FileNotFoundError(
            f"The explicitly selected installed MACE alias {model!r} is missing its pinned "
            f"cache file {candidate}"
        )

    if model.startswith("https://"):
        if not allow_download:
            raise RuntimeError("An explicit HTTPS MACE model requires allow_model_download=true")
        return model
    if model.startswith("resource://"):
        return str(resolve_input_file(model))

    candidate = Path(model).expanduser()
    if candidate.is_file():
        resolved = candidate.resolve()
        try:
            resolved.relative_to(_mace_model_cache_root())
        except ValueError:
            return str(resolve_input_file(model))
        return str(resolved)
    if "/" in model or model.endswith(".model"):
        return str(resolve_input_file(model))

    if model in _MACE_0_3_16_MODEL_NAMES:
        if not allow_download:
            raise RuntimeError(
                f"MACE model {model!r} is not one of the installed aliases "
                f"{sorted(_MACE_INSTALLED_MODEL_ALIASES)}; set allow_model_download=true "
                "only if this exact upstream model may be downloaded"
            )
        return model
    raise ValueError(
        f"Unknown MACE model label {model!r}. Installed aliases are "
        f"{sorted(_MACE_INSTALLED_MODEL_ALIASES)}; pinned MACE 0.3.16 upstream names are "
        f"{list(_MACE_0_3_16_MODEL_NAMES)}; an explicit local path or HTTPS URL is also accepted"
    )


def _ase_calculator(backend_id: str, method: dict[str, Any]):
    if backend_id == "ase_emt":
        from ase.calculators.emt import EMT

        return EMT(), module_version("ase")
    if backend_id == "tblite":
        from tblite.ase import TBLite

        name = str(method["method"]).lower().replace("_", "-")
        normalized = {"gfn1": "GFN1-xTB", "gfn1-xtb": "GFN1-xTB", "gfn2": "GFN2-xTB", "gfn2-xtb": "GFN2-xTB"}.get(name)
        if normalized is None:
            raise ValueError("TBLite method must be gfn1 or gfn2")
        return TBLite(method=normalized), module_version("tblite")
    if backend_id == "mace":
        allow_download = bool(method.get("allow_model_download", False))
        model = _resolve_mace_model(method["model"], allow_download=allow_download)
        from mace.calculators import mace_mp

        return (
            mace_mp(
                model=model,
                device=str(method["device"]),
                default_dtype=str(method.get("default_dtype", "float64")),
            ),
            module_version("mace-torch"),
        )
    if backend_id == "chgnet":
        from chgnet.model.dynamics import CHGNetCalculator

        if not bool(method.get("allow_model_download", False)):
            raise RuntimeError("CHGNet pretrained loading requires allow_model_download=true")
        return CHGNetCalculator(use_device=str(method["device"])), module_version("chgnet")
    if backend_id == "deepmd":
        calculator, version, _provenance = build_mlip_calculator(backend_id, method)
        return calculator, version
    raise ValueError(f"No ASE calculator implementation for {backend_id}")


def _ase_property(action_id: str, backend_id: str, request: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    inputs, method, settings = request_parts(request)
    atoms = ase_atoms(inputs["structure"])
    prepare_mlip_atoms(backend_id, atoms, method)
    calculator, version = _ase_calculator(backend_id, method)
    atoms.calc = calculator
    if action_id == "calculate_energy":
        return success(
            {"energy": float(atoms.get_potential_energy()), "unit": "eV", "atom_count": len(atoms)},
            backend_version=version,
        )
    if action_id == "calculate_forces":
        forces = np.asarray(atoms.get_forces(), dtype=float)
        return success(
            {"forces": forces.tolist(), "unit": "eV/angstrom", "atom_count": len(atoms)},
            backend_version=version,
        )
    if action_id == "calculate_dipole_moment":
        dipole = np.asarray(atoms.get_dipole_moment(), dtype=float)
        return success({"dipole": dipole.tolist(), "unit": "e*angstrom"}, backend_version=version)
    if action_id == "optimize_geometry":
        from ase.optimize import BFGS, FIRE, LBFGS

        directory = output_directory(action_id, backend_id)
        optimizer_name = str(settings.get("optimizer", "bfgs")).lower()
        optimizer_class = {"bfgs": BFGS, "lbfgs": LBFGS, "fire": FIRE}.get(optimizer_name)
        if optimizer_class is None:
            raise ValueError("optimizer must be bfgs, lbfgs, or fire")
        log_path = directory / "optimization.log"
        trajectory_path = directory / "optimization.traj"
        optimizer = optimizer_class(atoms, logfile=str(log_path), trajectory=str(trajectory_path))
        converged = bool(
            optimizer.run(
                fmax=float(settings["fmax_ev_per_angstrom"]),
                steps=int(settings.get("max_steps", 500)),
            )
        )
        xyz_path = directory / "optimized.xyz"
        from ase.io import write

        write(str(xyz_path), atoms)
        result = {
            "structure": structure_from_atoms(atoms),
            "converged": converged,
            "energy": float(atoms.get_potential_energy()),
            "energy_unit": "eV",
            "optimizer": optimizer_name,
        }
        if not converged:
            return partial_success(
                result,
                artifact_files=command_artifacts(directory),
                backend_version=version,
                warnings=["Geometry optimization reached its step limit before convergence."],
            )
        return success(
            result,
            artifact_files=command_artifacts(directory),
            backend_version=version,
        )
    if action_id == "calculate_hessian":
        displacement = float(settings.get("displacement_angstrom", 0.01))
        if displacement <= 0:
            raise ValueError("displacement_angstrom must be positive")
        coordinates = atoms.get_positions().copy()
        dimensions = 3 * len(atoms)
        hessian = np.zeros((dimensions, dimensions), dtype=float)
        for column in range(dimensions):
            atom_index, axis = divmod(column, 3)
            plus = coordinates.copy()
            minus = coordinates.copy()
            plus[atom_index, axis] += displacement
            minus[atom_index, axis] -= displacement
            atoms.set_positions(plus)
            forces_plus = np.asarray(atoms.get_forces(), dtype=float).reshape(-1)
            atoms.set_positions(minus)
            forces_minus = np.asarray(atoms.get_forces(), dtype=float).reshape(-1)
            hessian[:, column] = -(forces_plus - forces_minus) / (2 * displacement)
        atoms.set_positions(coordinates)
        hessian = 0.5 * (hessian + hessian.T)
        return success(
            {"matrix": hessian.tolist(), "unit": "eV/angstrom^2", "displacement_angstrom": displacement},
            backend_version=version,
        )
    return unsupported(f"ASE calculator does not implement {action_id}")


def _geometric_constraints(value: Any, atom_count: int, path: Path) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("constraints must be a typed mapping with freeze and/or set lists")
    unknown = sorted(set(value) - {"freeze", "set"})
    if unknown:
        raise ValueError(f"Unknown geomeTRIC constraint sections: {unknown}")
    lines: list[str] = []
    arities = {"distance": 2, "angle": 3, "dihedral": 4}

    def indices(record: dict[str, Any], *, expected: int | None = None) -> list[int]:
        raw = record.get("atom_indices")
        if not isinstance(raw, list) or not raw:
            raise ValueError("Each constraint requires a non-empty atom_indices list")
        values = [int(index) for index in raw]
        if expected is not None and len(values) != expected:
            raise ValueError(f"Constraint requires exactly {expected} atom indices")
        if len(set(values)) != len(values) or min(values) < 0 or max(values) >= atom_count:
            raise ValueError("Constraint atom indices must be unique and inside the structure")
        return values

    freeze = value.get("freeze") or []
    if not isinstance(freeze, list):
        raise ValueError("constraints.freeze must be a list")
    if freeze:
        lines.append("$freeze")
    for record in freeze:
        if not isinstance(record, dict):
            raise ValueError("Each freeze constraint must be an object")
        kind = str(record.get("type") or "").strip().lower()
        if kind in arities:
            atom_indices = indices(record, expected=arities[kind])
            lines.append(kind + " " + " ".join(str(index + 1) for index in atom_indices))
        elif kind in {"x", "y", "z", "xy", "xz", "yz", "xyz"}:
            atom_indices = indices(record)
            selection = ",".join(str(index + 1) for index in atom_indices)
            lines.append(f"{kind} {selection}")
        else:
            raise ValueError(
                "freeze constraint type must be distance, angle, dihedral, x, y, z, xy, xz, yz, or xyz"
            )

    set_constraints = value.get("set") or []
    if not isinstance(set_constraints, list):
        raise ValueError("constraints.set must be a list")
    if set_constraints:
        lines.append("$set")
    for record in set_constraints:
        if not isinstance(record, dict):
            raise ValueError("Each set constraint must be an object")
        kind = str(record.get("type") or "").strip().lower()
        if kind not in arities:
            raise ValueError("set constraint type must be distance, angle, or dihedral")
        atom_indices = indices(record, expected=arities[kind])
        target = float(record["value"])
        if not math.isfinite(target):
            raise ValueError("Constraint target values must be finite")
        lines.append(
            kind + " " + " ".join(str(index + 1) for index in atom_indices) + f" {target:.16g}"
        )
    if not lines:
        return None
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _geometric_optimize(request: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import geometric
    from geometric.engine import Engine
    from geometric.errors import GeomOptNotConvergedError
    from geometric.molecule import Molecule
    from geometric.optimize import run_optimizer

    inputs, method, settings = request_parts(request)
    original = structure_dict(inputs["structure"])
    atoms = original.get("atoms") or []
    if not atoms or any("position_angstrom" not in atom for atom in atoms):
        raise ValueError("geomeTRIC requires a non-empty molecular structure with coordinates")
    if any(bool(value) for value in original.get("pbc", [False, False, False])):
        raise ValueError("The validated geomeTRIC composite adapter is molecular/non-periodic")
    symbols = [str(atom["element"]) for atom in atoms]
    initial = np.asarray([atom["position_angstrom"] for atom in atoms], dtype=float)
    if initial.shape != (len(atoms), 3) or not np.all(np.isfinite(initial)):
        raise ValueError("Initial coordinates must form a finite Nx3 matrix")

    molecule = Molecule()
    molecule.elem = symbols
    molecule.xyzs = [initial]

    class AgentSelectedCalculatorEngine(Engine):
        def __init__(self, value):
            super().__init__(value)
            self.evaluations: list[dict[str, Any]] = []
            self.last_coordinates_angstrom = initial.copy()

        def calc_new(self, coordinates, dirname):
            coordinates_angstrom = (
                np.asarray(coordinates, dtype=float).reshape(-1, 3) * BOHR_TO_ANGSTROM
            )
            structure = {
                **{key: value for key, value in original.items() if key != "atoms"},
                "atoms": [
                    {
                        **{key: value for key, value in atom.items() if key != "position_angstrom"},
                        "position_angstrom": coordinates_angstrom[index].tolist(),
                    }
                    for index, atom in enumerate(atoms)
                ],
            }
            force_result, force_provenance = invoke_calculator_component(
                request, "calculate_forces", structure
            )
            energy_result, energy_provenance = invoke_calculator_component(
                request, "calculate_energy", structure
            )
            energy = energy_hartree(energy_result)
            forces = np.asarray(forces_ev_per_angstrom(force_result), dtype=float)
            if forces.shape != coordinates_angstrom.shape:
                raise RuntimeError("Calculator force matrix does not match the optimized structure")
            gradient = -forces * BOHR_TO_ANGSTROM / HARTREE_TO_EV
            self.last_coordinates_angstrom = coordinates_angstrom.copy()
            self.evaluations.append(
                {
                    "evaluation_index": len(self.evaluations),
                    "energy_hartree": energy,
                    "maximum_force_ev_per_angstrom": float(np.max(np.linalg.norm(forces, axis=1))),
                    "energy_component": energy_provenance,
                    "force_component": force_provenance,
                }
            )
            return {"energy": energy, "gradient": gradient.reshape(-1)}

    coordinate_system = str(settings["coordinate_system"]).strip().lower()
    if coordinate_system not in {"cart", "prim", "dlc", "hdlc", "tric-p", "tric"}:
        raise ValueError("coordinate_system must be cart, prim, dlc, hdlc, tric-p, or tric")
    hessian_strategy = str(settings["hessian_strategy"]).strip().lower()
    if hessian_strategy not in {"never", "first", "each", "last", "first+last"}:
        raise ValueError("hessian_strategy must be never, first, each, last, or first+last")
    projection = str(settings["project_rigid_force_torque"]).strip().lower()
    projection_code = {"never": 0, "auto": 1, "always": 2}.get(projection)
    if projection_code is None:
        raise ValueError("project_rigid_force_torque must be never, auto, or always")
    trust = float(settings["trust_radius_angstrom"])
    minimum_trust = float(settings["minimum_trust_radius_angstrom"])
    maximum_trust = float(settings["maximum_trust_radius_angstrom"])
    if not (0 < minimum_trust <= trust <= maximum_trust):
        raise ValueError("Trust radii must satisfy 0 < minimum <= initial <= maximum")
    maximum_iterations = int(settings["max_iterations"])
    if maximum_iterations < 1:
        raise ValueError("max_iterations must be positive")
    constraint_algorithm = int(settings["constraint_algorithm"])
    if constraint_algorithm not in {0, 1}:
        raise ValueError("constraint_algorithm must be 0 or 1")

    directory = output_directory("optimize_geometry", "geometric")
    input_path = directory / "geometric.input"
    input_path.write_text("", encoding="utf-8")
    constraint_path = _geometric_constraints(
        inputs.get("constraints"), len(atoms), directory / "constraints.txt"
    )
    engine = AgentSelectedCalculatorEngine(molecule)
    options = {
        "customengine": engine,
        "input": str(input_path),
        "prefix": str(directory / "geometric"),
        "coordsys": coordinate_system,
        "maxiter": maximum_iterations,
        "trust": trust,
        "tmin": minimum_trust,
        "tmax": maximum_trust,
        "hessian": hessian_strategy,
        "subfrctor": projection_code,
        "convergence_energy": float(settings["convergence_energy_hartree"]),
        "convergence_grms": float(settings["convergence_grms_hartree_per_bohr"]),
        "convergence_gmax": float(settings["convergence_gmax_hartree_per_bohr"]),
        "convergence_drms": float(settings["convergence_drms_angstrom"]),
        "convergence_dmax": float(settings["convergence_dmax_angstrom"]),
        "rigid": bool(settings["rigid_fragments"]),
        "conmethod": constraint_algorithm,
        "enforce": float(settings["constraint_enforcement_tolerance"]),
    }
    if constraint_path is not None:
        options["constraints"] = str(constraint_path)
    converged = True
    warning = None
    try:
        progress = run_optimizer(**options)
        final_coordinates = np.asarray(progress.xyzs[-1], dtype=float)
    except GeomOptNotConvergedError as exc:
        converged = False
        warning = str(exc)
        final_coordinates = engine.last_coordinates_angstrom
    result_structure = {
        **{key: value for key, value in original.items() if key != "atoms"},
        "atoms": [
            {
                **{key: value for key, value in atom.items() if key != "position_angstrom"},
                "position_angstrom": final_coordinates[index].tolist(),
            }
            for index, atom in enumerate(atoms)
        ],
    }
    optimized_path = write_xyz(result_structure, directory / "optimized.xyz")
    evaluations_path = write_json(directory, "component_evaluations.json", engine.evaluations)
    result = {
        "structure": result_structure,
        "converged": converged,
        "optimizer": "geometric",
        "coordinate_system": coordinate_system,
        "calculator_backend": request["component_backends"]["calculator"],
        "evaluation_count": len(engine.evaluations),
        "final_energy_hartree": (
            engine.evaluations[-1]["energy_hartree"] if engine.evaluations else None
        ),
        "constraint_count": sum(
            len((inputs.get("constraints") or {}).get(section) or [])
            for section in ("freeze", "set")
        ),
    }
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
    values = {
        "artifact_files": artifacts,
        "backend_version": getattr(geometric, "__version__", module_version("geometric")),
        "provenance": {
            "calculator_backend": request["component_backends"]["calculator"],
            "component_evaluation_count": len(engine.evaluations),
            "automatic_component_selection": False,
        },
    }
    if not converged:
        return partial_success(
            result,
            warnings=[f"geomeTRIC reached its iteration limit without convergence: {warning}"],
            **values,
        )
    return success(result, **values)


def _xtb(action_id: str, request: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    inputs, method, settings = request_parts(request)
    directory = output_directory(action_id, "xtb")
    xyz = write_xyz(inputs["structure"], directory / "input.xyz")
    structure = structure_dict(inputs["structure"])
    method_name = str(method["method"]).lower().replace("-", "")
    method_number = {"gfn1": "1", "gfn1xtb": "1", "gfn2": "2", "gfn2xtb": "2"}.get(method_name)
    if method_number is None:
        raise ValueError("xTB method must be gfn1 or gfn2")
    charge = int(method.get("charge", structure.get("charge", 0)))
    unpaired = int(method.get("unpaired_electrons", max(0, int(structure.get("multiplicity", 1)) - 1)))
    resolved_molecular_state = {
        "charge": charge,
        "unpaired_electrons": unpaired,
        "multiplicity": unpaired + 1,
        "charge_source": "method_spec" if "charge" in method else "input_structure",
        "spin_source": (
            "method_spec" if "unpaired_electrons" in method else "input_structure"
        ),
    }
    arguments = [str(xyz), "--gfn", method_number, "--chrg", str(charge), "--uhf", str(unpaired)]
    if action_id == "optimize_geometry":
        arguments.extend(["--opt", str(settings["optimization_level"])])
    elif action_id == "calculate_hessian":
        arguments.append("--hess")
    elif action_id == "calculate_forces":
        arguments.append("--grad")
    elif action_id == "calculate_dipole_moment":
        arguments.append("--dipole")
    elif action_id == "calculate_atomic_charges":
        arguments.append("--pop")
    elif action_id == "calculate_bond_orders":
        arguments.append("--pop")
    completed = run_external(
        executable="xtb",
        environment_variable="CHEMGRAPH_XTB_COMMAND",
        arguments=arguments,
        directory=directory,
        timeout_seconds=int(request.get("resource_limits", {}).get("walltime_seconds", 1800)),
    )
    (directory / "stdout.log").write_text(completed["stdout"], encoding="utf-8")
    (directory / "stderr.log").write_text(completed["stderr"], encoding="utf-8")
    if not completed["available"]:
        return unavailable(completed["stderr"], install="conda install -c conda-forge xtb")
    if completed["returncode"] != 0:
        raise RuntimeError(f"xTB failed: {completed['stderr'][-2000:]}")
    energy_matches = re.findall(r"TOTAL ENERGY\s+(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)", completed["stdout"])
    energy = float(energy_matches[-1]) if energy_matches else None
    if action_id == "calculate_energy":
        if energy is None:
            raise RuntimeError("Could not parse xTB TOTAL ENERGY")
        result = {"energy": energy, "unit": "hartree", "method": f"gfn{method_number}"}
    elif action_id == "optimize_geometry":
        optimized = directory / "xtbopt.xyz"
        if not optimized.is_file():
            raise RuntimeError("xTB optimization completed without xtbopt.xyz")
        optimized_structure = structure_dict(relative_workspace_path(optimized))
        optimized_structure["charge"] = charge
        optimized_structure["multiplicity"] = (
            unpaired + 1
            if "unpaired_electrons" in method
            else int(structure.get("multiplicity", unpaired + 1))
        )
        result = {
            "structure": optimized_structure,
            "converged": True,
            "energy": energy,
            "energy_unit": "hartree",
            "optimization_level": str(settings["optimization_level"]),
        }
    elif action_id == "calculate_hessian":
        hessian_path = directory / "hessian"
        if not hessian_path.is_file():
            raise RuntimeError("xTB Hessian calculation completed without hessian file")
        values = []
        for token in hessian_path.read_text(encoding="utf-8").replace("D", "E").split():
            try:
                values.append(float(token))
            except ValueError:
                continue
        dimensions = 3 * len(structure.get("atoms") or structure.get("symbols") or [])
        if dimensions and len(values) >= dimensions * dimensions:
            matrix = np.asarray(values[-dimensions * dimensions :]).reshape(dimensions, dimensions).tolist()
        else:
            matrix = None
        result = {
            "matrix": matrix,
            "unit": "hartree/bohr^2",
            "raw_hessian_path": relative_workspace_path(hessian_path),
        }
    elif action_id == "calculate_forces":
        gradient_path = directory / "input.engrad"
        if not gradient_path.is_file():
            candidates = sorted(directory.glob("*.engrad"))
            gradient_path = candidates[-1] if candidates else gradient_path
        if not gradient_path.is_file():
            raise RuntimeError("xTB gradient calculation completed without an .engrad file")
        numeric_lines = []
        for line in gradient_path.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = line.strip().replace("D", "E").replace("d", "e")
            if not stripped or stripped.startswith("#"):
                continue
            if len(stripped.split()) != 1:
                continue
            try:
                numeric_lines.append(float(stripped))
            except ValueError:
                continue
        atom_count = len(structure.get("atoms") or structure.get("symbols") or [])
        if atom_count <= 0 or len(numeric_lines) < 2 + 3 * atom_count:
            raise RuntimeError("Could not parse the xTB .engrad gradient block")
        gradient = np.asarray(numeric_lines[2 : 2 + 3 * atom_count], dtype=float).reshape(atom_count, 3)
        conversion = 27.211386245988 / 0.529177210903
        result = {
            "forces": (-gradient * conversion).tolist(),
            "unit": "eV/angstrom",
            "atom_count": atom_count,
            "raw_gradient_path": relative_workspace_path(gradient_path),
        }
    elif action_id == "calculate_dipole_moment":
        match = re.search(
            r"molecular dipole:.*?^\s*full:\s+"
            r"(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s+"
            r"(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s+"
            r"(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s+"
            r"(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)",
            completed["stdout"],
            flags=re.MULTILINE | re.DOTALL,
        )
        if not match:
            raise RuntimeError("Could not parse the xTB molecular dipole")
        dipole_atomic_units = [float(match.group(index)) for index in (1, 2, 3)]
        atomic_unit_to_debye = 2.541746473
        bohr_to_angstrom = 0.529177210903
        dipole_debye = [value * atomic_unit_to_debye for value in dipole_atomic_units]
        dipole_e_angstrom = [value * bohr_to_angstrom for value in dipole_atomic_units]
        result = {
            "dipole": dipole_debye,
            "dipole_debye": dipole_debye,
            "dipole_e_angstrom": dipole_e_angstrom,
            "magnitude": float(match.group(4)),
            "magnitude_e_angstrom": float(match.group(4)) / 4.80320471257,
            "unit": "debye",
            "dipole_atomic_units": dipole_atomic_units,
            "energy_hartree": energy,
        }
    elif action_id == "calculate_atomic_charges":
        charges_path = directory / "charges"
        if not charges_path.is_file():
            raise RuntimeError("xTB population analysis completed without a charges file")
        charges = []
        for token in charges_path.read_text(encoding="utf-8", errors="replace").replace("D", "E").split():
            try:
                charges.append(float(token))
            except ValueError:
                continue
        atom_count = len(structure.get("atoms") or structure.get("symbols") or [])
        if len(charges) != atom_count:
            raise RuntimeError(f"Expected {atom_count} xTB charges, parsed {len(charges)}")
        result = {
            "charges": charges,
            "analysis": "xtb_mulliken",
            "unit": "elementary_charge",
            "energy_hartree": energy,
            "raw_charges_path": relative_workspace_path(charges_path),
        }
    elif action_id == "calculate_bond_orders":
        bond_path = directory / "wbo"
        if not bond_path.is_file():
            raise RuntimeError("xTB population analysis completed without a wbo file")
        minimum = float(settings["minimum_bond_order"])
        if minimum < 0:
            raise ValueError("minimum_bond_order must be nonnegative")
        bonds = []
        for line in bond_path.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                first, second, value = int(parts[0]) - 1, int(parts[1]) - 1, float(parts[2])
            except ValueError:
                continue
            if value >= minimum:
                bonds.append({"atom_index_a": first, "atom_index_b": second, "bond_order": value})
        result = {
            "bonds": bonds,
            "analysis": "xtb_wiberg_bond_order",
            "minimum_bond_order": minimum,
            "raw_bond_order_path": relative_workspace_path(bond_path),
        }
    else:
        return unsupported(f"xTB does not implement {action_id}")
    if action_id == "calculate_hessian" and result.get("matrix") is None:
        return partial_success(
            result,
            artifact_files=command_artifacts(directory),
            provenance={
                "command": completed["command"],
                "resolved_molecular_state": resolved_molecular_state,
            },
            warnings=["xTB produced a Hessian file, but the dense Hessian matrix could not be parsed."],
        )
    return success(
        result,
        artifact_files=command_artifacts(directory),
        provenance={
            "command": completed["command"],
            "resolved_molecular_state": resolved_molecular_state,
        },
    )


def _pyscf(action_id: str, request: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    from pyscf import dft, gto, scf

    inputs, method, settings = request_parts(request)
    structure = structure_dict(inputs["structure"])
    method_name = str(method["method"]).lower()
    molecule = gto.M(
        atom=atom_spec(inputs["structure"]),
        basis=str(method["basis"]),
        charge=int(method.get("charge", structure.get("charge", 0))),
        spin=int(method.get("spin", max(0, int(structure.get("multiplicity", 1)) - 1))),
        unit="Angstrom",
        verbose=0,
    )
    classes = {"rhf": scf.RHF, "uhf": scf.UHF, "rks": dft.RKS, "uks": dft.UKS}
    if method_name not in classes:
        raise ValueError("PySCF method must be rhf, uhf, rks, or uks")
    calculation = classes[method_name](molecule)
    if method_name in {"rks", "uks"}:
        functional = method.get("functional") or method.get("xc")
        if not functional:
            raise ValueError("PySCF RKS/UKS requires an explicit functional in method_spec")
        calculation.xc = str(functional)
    calculation.conv_tol = float(settings.get("scf_convergence", 1e-9))
    energy = float(calculation.kernel())
    common = {
        "converged": bool(calculation.converged),
        "method": method_name,
        "basis": str(method["basis"]),
        "energy_hartree": energy,
    }
    if action_id == "calculate_energy":
        result = {**common, "energy": energy, "unit": "hartree"}
    elif action_id == "calculate_forces":
        gradient = np.asarray(calculation.nuc_grad_method().kernel(), dtype=float)
        conversion = 27.211386245988 / 0.529177210903
        result = {
            **common,
            "forces": (-gradient * conversion).tolist(),
            "unit": "eV/angstrom",
            "atom_count": int(molecule.natm),
        }
    elif action_id == "calculate_hessian":
        hessian = np.asarray(calculation.Hessian().kernel(), dtype=float)
        dimensions = 3 * int(molecule.natm)
        result = {
            **common,
            "matrix": hessian.transpose(0, 2, 1, 3).reshape(dimensions, dimensions).tolist(),
            "unit": "hartree/bohr^2",
        }
    elif action_id == "calculate_dipole_moment":
        dipole = calculation.dip_moment(mol=molecule, dm=calculation.make_rdm1(), unit="Debye", verbose=0)
        result = {**common, "dipole": np.asarray(dipole, dtype=float).tolist(), "unit": "debye"}
    elif action_id == "calculate_atomic_charges":
        population, charges = calculation.mulliken_pop(molecule, calculation.make_rdm1(), verbose=0)
        result = {
            **common,
            "analysis": str(method.get("population_analysis", "mulliken")),
            "charges": np.asarray(charges, dtype=float).tolist(),
            "unit": "elementary_charge",
            "orbital_populations": np.asarray(population, dtype=float).tolist(),
        }
    elif action_id == "calculate_orbitals":
        energies = calculation.mo_energy
        occupations = calculation.mo_occ
        if isinstance(energies, tuple):
            result = {
                **common,
                "alpha_energies_hartree": np.asarray(energies[0]).tolist(),
                "beta_energies_hartree": np.asarray(energies[1]).tolist(),
                "alpha_occupations": np.asarray(occupations[0]).tolist(),
                "beta_occupations": np.asarray(occupations[1]).tolist(),
            }
        else:
            result = {
                **common,
                "energies_hartree": np.asarray(energies).tolist(),
                "occupations": np.asarray(occupations).tolist(),
            }
        if bool(settings.get("include_coefficients", False)):
            result["coefficients"] = np.asarray(calculation.mo_coeff).tolist()
    elif action_id == "calculate_excited_states":
        excited_method = str(method["excited_state_method"]).strip().lower()
        if excited_method == "tda":
            excited = calculation.TDA()
        elif excited_method == "tddft":
            excited = calculation.TDHF() if method_name in {"rhf", "uhf"} else calculation.TDDFT()
        else:
            raise ValueError("PySCF excited_state_method must be tda or tddft")
        spin_symmetry = str(settings["spin_symmetry"]).strip().lower()
        if spin_symmetry not in {"singlet", "triplet"}:
            raise ValueError("spin_symmetry must be singlet or triplet")
        if method_name in {"uhf", "uks"}:
            raise ValueError("This typed PySCF excited-state adapter currently requires a restricted reference")
        excited.singlet = spin_symmetry == "singlet"
        excited.nstates = int(settings["number_of_states"])
        excitation_hartree, _amplitudes = excited.kernel()
        oscillator = np.asarray(excited.oscillator_strength(), dtype=float)
        transition_dipoles = np.asarray(excited.transition_dipole(), dtype=float)
        states = []
        for index, value in enumerate(np.asarray(excitation_hartree, dtype=float)):
            states.append(
                {
                    "state_index": index + 1,
                    "energy_hartree": float(value),
                    "energy_ev": float(value * 27.211386245988),
                    "oscillator_strength": float(oscillator[index]),
                    "transition_dipole_atomic_unit": transition_dipoles[index].tolist(),
                    "spin_symmetry": spin_symmetry,
                }
            )
        result = {
            **common,
            "states": states,
            "excited_state_method": excited_method,
            "spin_symmetry": spin_symmetry,
        }
    else:
        return unsupported(f"PySCF does not implement {action_id}")
    return success(result, backend_version=module_version("pyscf"))


def _nwchem(action_id: str, request: dict[str, Any]) -> dict[str, Any]:
    import json
    import numpy as np
    import qcelemental as qcel
    import qcengine as qcng

    inputs, method, settings = request_parts(request)
    structure = structure_dict(inputs["structure"])
    atoms = structure.get("atoms") or []
    if not atoms:
        raise ValueError("NWChem actions require an explicit 3D AtomicStructure")
    symbols = [str(atom["element"]) for atom in atoms]
    bohr_per_angstrom = 1.0 / 0.529177210903
    geometry = (
        np.asarray([atom["position_angstrom"] for atom in atoms], dtype=float)
        * bohr_per_angstrom
    )
    molecule = qcel.models.Molecule(
        symbols=symbols,
        geometry=geometry,
        molecular_charge=int(method.get("charge", structure.get("charge", 0))),
        molecular_multiplicity=int(method.get("multiplicity", structure.get("multiplicity", 1))),
        fix_com=True,
        fix_orientation=True,
    )
    method_name = str(method["method"]).strip().lower()
    keywords: dict[str, Any] = {"geometry__autosym": 1e-8}
    if method_name == "dft":
        functional = method.get("functional")
        if not functional:
            raise ValueError("NWChem method=dft requires method_spec.functional")
        keywords.update(
            {
                "dft__xc": str(functional),
                "dft__convergence__energy": float(settings["scf_convergence"]),
                "dft__iterations": int(settings["max_scf_cycles"]),
            }
        )
    else:
        keywords.update(
            {
                "scf__thresh": float(settings["scf_convergence"]),
                "scf__maxiter": int(settings["max_scf_cycles"]),
            }
        )
    reference = str(method.get("reference", "")).strip().lower()
    if reference:
        if reference not in {"rhf", "uhf", "rohf"}:
            raise ValueError("NWChem reference must be rhf, uhf, or rohf")
        if reference in {"uhf", "rohf"}:
            keywords[f"scf__{reference}"] = True
    driver = {
        "calculate_energy": "energy",
        "calculate_forces": "gradient",
        "calculate_hessian": "hessian",
        "calculate_dipole_moment": "properties",
        "calculate_atomic_charges": "properties",
    }[action_id]
    if action_id == "calculate_dipole_moment":
        keywords["property__dipole"] = True
    elif action_id == "calculate_atomic_charges":
        keywords["property__mulliken"] = True
    atomic_input = qcel.models.AtomicInput(
        molecule=molecule,
        driver=driver,
        model={"method": method_name, "basis": str(method["basis"])},
        keywords=keywords,
    )
    limits = dict(request.get("resource_limits") or {})
    memory_gib = max(0.5, float(limits.get("memory_mb") or 1024) / 1024.0)
    output = qcng.compute(
        atomic_input,
        "nwchem",
        raise_error=True,
        return_dict=True,
        task_config={
            "memory": memory_gib,
            "ncores": int(limits.get("cpu_cores") or 1),
        },
    )
    directory = output_directory(action_id, "nwchem")
    input_path = directory / "input_qcschema.json"
    input_path.write_text(
        json.dumps(atomic_input.dict(), indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    result_path = directory / "result_qcschema.json"
    result_path.write_text(json.dumps(output, indent=2, default=str) + "\n", encoding="utf-8")
    stdout_path = directory / "stdout.log"
    stdout = str(output.get("stdout") or "")
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path = directory / "stderr.log"
    stderr_path.write_text(str(output.get("stderr") or ""), encoding="utf-8")
    energy = float((output.get("properties") or {}).get("return_energy"))
    common = {
        "energy_hartree": energy,
        "method": method_name,
        "basis": str(method["basis"]),
    }
    if action_id == "calculate_energy":
        result = {**common, "energy": energy, "unit": "hartree"}
    elif action_id == "calculate_forces":
        gradient = np.asarray(output["return_result"], dtype=float)
        conversion = 27.211386245988 / 0.529177210903
        result = {
            **common,
            "forces": (-gradient * conversion).tolist(),
            "unit": "eV/angstrom",
            "atom_count": len(symbols),
        }
    elif action_id == "calculate_hessian":
        hessian = np.asarray(output["return_result"], dtype=float).reshape(3 * len(symbols), 3 * len(symbols))
        result = {**common, "matrix": hessian.tolist(), "unit": "hartree/bohr^2"}
    elif action_id == "calculate_dipole_moment":
        qcvars = (output.get("extras") or {}).get("qcvars") or {}
        vector = np.asarray(qcvars.get("DIPOLE MOMENT"), dtype=float)
        if vector.shape != (3,):
            raise RuntimeError("QCEngine did not return the NWChem dipole vector")
        debye = vector * 2.541746473
        result = {
            **common,
            "dipole": debye.tolist(),
            "magnitude": float(np.linalg.norm(debye)),
            "unit": "debye",
        }
    else:
        block_match = re.search(
            r"Mulliken analysis of the total density\s*-+\s*(.*?)(?:\n\s*\n|\n\s*Multipole analysis)",
            stdout,
            flags=re.S | re.I,
        )
        if not block_match:
            raise RuntimeError("Could not locate the NWChem Mulliken population block")
        populations = []
        charges = []
        for line in block_match.group(1).splitlines():
            match = re.match(
                r"^\s*\d+\s+[A-Za-z][A-Za-z]?\s+([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)",
                line,
            )
            if match:
                nuclear_charge = float(match.group(1))
                population = float(match.group(2))
                populations.append(population)
                charges.append(nuclear_charge - population)
        if len(charges) != len(symbols):
            raise RuntimeError("NWChem Mulliken population count does not match the molecule")
        result = {
            **common,
            "charges": charges,
            "electron_populations": populations,
            "analysis": "mulliken",
            "unit": "elementary_charge",
        }
    version = str((output.get("provenance") or {}).get("version") or "") or None
    return success(
        result,
        artifact_files=command_artifacts(directory),
        backend_version=version,
        provenance={"qcschema_driver": driver, "qcengine_version": module_version("qcengine")},
    )


def _openmolcas_safe_label(value: Any, *, field: str) -> str:
    text = str(value).strip()
    if not text or not re.fullmatch(r"[A-Za-z0-9_.+()/\-]+", text):
        raise ValueError(f"OpenMolcas {field} contains unsupported characters")
    return text


def _render_openmolcas_scf(
    structure: dict[str, Any],
    method: dict[str, Any],
    settings: dict[str, Any],
) -> str:
    atoms = structure.get("atoms") or []
    if not atoms or any("position_angstrom" not in atom for atom in atoms):
        raise ValueError("OpenMolcas requires a non-empty molecular structure with coordinates")
    if any(bool(value) for value in structure.get("pbc", [False, False, False])):
        raise ValueError("The OpenMolcas SCF adapter is molecular/non-periodic")
    method_name = str(method["method"]).strip().lower()
    if method_name not in {"hf", "dft"}:
        raise ValueError("OpenMolcas method must be hf or dft")
    basis = _openmolcas_safe_label(method["basis"], field="basis")
    functional = None
    if method_name == "dft":
        if "functional" not in method:
            raise ValueError("OpenMolcas method=dft requires method_spec.functional")
        functional = _openmolcas_safe_label(method["functional"], field="functional")
    charge = int(method.get("charge", structure.get("charge", 0)))
    multiplicity = int(method.get("multiplicity", structure.get("multiplicity", 1)))
    if multiplicity < 1:
        raise ValueError("multiplicity must be positive")
    use_uhf = settings["use_uhf"]
    use_symmetry = settings["use_symmetry"]
    use_cholesky = settings["use_cholesky"]
    if not all(isinstance(value, bool) for value in (use_uhf, use_symmetry, use_cholesky)):
        raise ValueError("use_uhf, use_symmetry, and use_cholesky must be explicit booleans")
    if multiplicity != 1 and not use_uhf:
        raise ValueError("Open-shell OpenMolcas SCF calculations require use_uhf=true")
    maximum_iterations = int(settings["max_scf_iterations"])
    if not 1 <= maximum_iterations <= 400:
        raise ValueError("max_scf_iterations must be between 1 and 400")
    thresholds = settings["scf_thresholds"]
    if not isinstance(thresholds, (list, tuple)) or len(thresholds) != 4:
        raise ValueError("scf_thresholds must contain EThr, DThr, FThr, and DltNTh")
    threshold_values = [float(value) for value in thresholds]
    if any(not math.isfinite(value) or value <= 0 for value in threshold_values):
        raise ValueError("scf_thresholds values must be positive and finite")
    initial_guess = str(settings["initial_guess"]).strip().lower()
    if initial_guess not in {"default", "core"}:
        raise ValueError("initial_guess must be default or core")

    lines = [
        "&GATEWAY",
        "Title = ResearchChemBench typed OpenMolcas SCF action",
        "Coord",
        str(len(atoms)),
        "Angstrom",
    ]
    lines.extend(
        f"{atom['element']} {float(atom['position_angstrom'][0]):.12f} "
        f"{float(atom['position_angstrom'][1]):.12f} "
        f"{float(atom['position_angstrom'][2]):.12f}"
        for atom in atoms
    )
    lines.extend([f"Basis = {basis}"])
    if not use_symmetry:
        lines.append("Group = Nosym")
    lines.extend(["", "&SEWARD", "", "&SCF", f"Charge = {charge}", f"Spin = {multiplicity}"])
    if use_uhf:
        lines.append("UHF")
    if functional is not None:
        lines.append(f"KSDFT = {functional}")
    if use_cholesky:
        lines.append("Cholesky")
    if initial_guess == "core":
        lines.append("Core")
    lines.extend(
        [
            f"Iter = {maximum_iterations}",
            "Threshold = " + " ".join(f"{value:.12g}" for value in threshold_values),
            "PrOrbitals = 1 1",
            "",
        ]
    )
    return "\n".join(lines)


def _openmolcas_numbers(text: str) -> list[float]:
    return [
        float(value.replace("D", "E").replace("d", "e"))
        for value in re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?", text)
    ]


def _openmolcas(action_id: str, request: dict[str, Any]) -> dict[str, Any]:
    inputs, method, settings = request_parts(request)
    structure = structure_dict(inputs["structure"])
    directory = output_directory(action_id, "openmolcas")
    project = f"job_{directory.name}"
    input_path = directory / f"{project}.input"
    input_path.write_text(
        _render_openmolcas_scf(structure, method, settings),
        encoding="utf-8",
    )
    output_path = directory / f"{project}.log"
    error_path = directory / f"{project}.err"
    cores = max(1, int(request.get("resource_limits", {}).get("cpu_cores") or 1))
    completed = run_external(
        executable="pymolcas",
        environment_variable="CHEMGRAPH_OPENMOLCAS_COMMAND",
        arguments=[
            "-nt", str(cores), "-o", output_path.name, "-e", error_path.name,
            input_path.name,
        ],
        directory=directory,
        timeout_seconds=int(request.get("resource_limits", {}).get("walltime_seconds", 1800)),
    )
    (directory / "driver.stdout.log").write_text(completed["stdout"], encoding="utf-8")
    (directory / "driver.stderr.log").write_text(completed["stderr"], encoding="utf-8")
    if not completed["available"]:
        return unavailable(completed["stderr"], install="Configure OpenMolcas v25.10")
    if completed["returncode"] != 0:
        diagnostic = completed["stderr"]
        if error_path.is_file():
            diagnostic += "\n" + error_path.read_text(encoding="utf-8", errors="replace")
        raise RuntimeError(f"OpenMolcas failed: {diagnostic[-3000:]}")
    output = output_path.read_text(encoding="utf-8", errors="replace") if output_path.is_file() else completed["stdout"]
    energy_match = re.search(
        r"Total SCF energy\s+([-+0-9.eEdD]+)", output, flags=re.I
    )
    if not energy_match:
        raise RuntimeError("Could not parse the OpenMolcas total SCF energy")
    energy = float(energy_match.group(1).replace("D", "E").replace("d", "e"))
    common = {
        "energy_hartree": energy,
        "method": str(method["method"]).strip().lower(),
        "basis": str(method["basis"]),
        "functional": method.get("functional"),
    }
    if action_id == "calculate_energy":
        result = {**common, "energy": energy, "unit": "hartree"}
    elif action_id == "calculate_dipole_moment":
        dipole_block = re.search(
            r"Dipole Moment \(debye\):(.*?)(?:Quadrupole Moment|--)",
            output,
            flags=re.S | re.I,
        )
        if not dipole_block:
            raise RuntimeError("Could not locate the OpenMolcas dipole-moment block")
        match = re.search(
            r"X=\s*([-+0-9.eEdD]+).*?Y=\s*([-+0-9.eEdD]+).*?"
            r"Z=\s*([-+0-9.eEdD]+).*?Total=\s*([-+0-9.eEdD]+)",
            dipole_block.group(1),
            flags=re.S | re.I,
        )
        if not match:
            raise RuntimeError("Could not parse the OpenMolcas dipole vector")
        values = [float(value.replace("D", "E").replace("d", "e")) for value in match.groups()]
        result = {
            **common,
            "dipole": values[:3],
            "magnitude": values[3],
            "unit": "debye",
        }
    elif action_id == "calculate_atomic_charges":
        charge_block = re.search(
            r"Mulliken charges per centre and basis function type(.*?)"
            r"Total electronic charge",
            output,
            flags=re.S | re.I,
        )
        if not charge_block:
            raise RuntimeError("Could not locate the OpenMolcas Mulliken-charge block")
        charge_line = re.search(r"^\s*N-E\s+(.+)$", charge_block.group(1), flags=re.M)
        charges = _openmolcas_numbers(charge_line.group(1)) if charge_line else []
        atom_count = len(structure.get("atoms") or [])
        if len(charges) != atom_count:
            raise RuntimeError("OpenMolcas Mulliken charge count does not match the structure")
        result = {
            **common,
            "charges": charges,
            "total_charge": float(sum(charges)),
            "analysis": "mulliken",
            "unit": "elementary_charge",
        }
    elif action_id == "calculate_orbitals":
        energies: list[float] = []
        occupations: list[float] = []
        for match in re.finditer(
            r"^\s*Orbital\s+.+?\n\s*Energy\s+(.+?)\n\s*Occ\. No\.\s+(.+?)$",
            output,
            flags=re.M,
        ):
            block_energies = _openmolcas_numbers(match.group(1))
            block_occupations = _openmolcas_numbers(match.group(2))
            if len(block_energies) == len(block_occupations):
                energies.extend(block_energies)
                occupations.extend(block_occupations)
        if not energies:
            raise RuntimeError("Could not parse OpenMolcas orbital energies and occupations")
        result = {
            **common,
            "energies_hartree": energies,
            "occupations": occupations,
            "orbital_count": len(energies),
        }
    else:
        return unsupported(f"OpenMolcas does not implement {action_id}")
    result_path = write_json(directory, "result.json", result)
    result_relative = relative_workspace_path(result_path)
    artifacts = [
        item for item in command_artifacts(directory) if item["path"] != result_relative
    ]
    artifacts.append(
        {
            "path": result_relative,
            "semantic_type": {
                "calculate_energy": "EnergyResult",
                "calculate_dipole_moment": "DipoleResult",
                "calculate_atomic_charges": "AtomicChargeResult",
                "calculate_orbitals": "OrbitalResult",
            }[action_id],
            "media_type": "application/json",
        }
    )
    warnings = []
    for line in output.splitlines():
        if "warning" in line.lower() and line.strip() not in warnings:
            warnings.append(line.strip()[:1000])
        if len(warnings) >= 40:
            break
    return success(
        result,
        artifact_files=artifacts,
        backend_version="25.10",
        warnings=warnings,
        provenance={
            "command": completed["command"],
            "parallel_threads": cores,
            "happy_landing": "Happy landing!" in output,
            "generated_input": relative_workspace_path(input_path),
        },
    )


def _multiwfn_wavefunction_analysis(action_id: str, request: dict[str, Any]) -> dict[str, Any]:
    inputs, method, settings = request_parts(request)
    source = resolve_input_file(inputs["structure"])
    if not source.is_file():
        raise ValueError("Multiwfn requires a regular wavefunction file Artifact")
    if source.suffix.lower() not in {
        ".fch", ".fchk", ".wfn", ".wfx", ".mwfn", ".molden", ".47",
    }:
        raise ValueError("Unsupported Multiwfn wavefunction-file extension")
    directory = output_directory(action_id, "multiwfn")
    staged = directory / source.name
    shutil.copy2(source, staged)

    if action_id == "calculate_bond_orders":
        definition = str(method["bond_order_definition"]).strip().lower()
        menu = {"mayer": 1, "wiberg_lowdin": 3, "mulliken": 4}.get(definition)
        if menu is None:
            raise ValueError(
                "bond_order_definition must be mayer, wiberg_lowdin, or mulliken"
            )
        minimum = float(settings["minimum_bond_order"])
        if not math.isfinite(minimum) or minimum < 0.05:
            raise ValueError(
                "Multiwfn minimum_bond_order must be at least its printed-table cutoff 0.05"
            )
        stdin_text = f"9\n{menu}\nn\n0\nq\n"
    elif action_id == "calculate_atomic_charges":
        population = str(method["population_analysis"]).strip().lower()
        if population == "mulliken":
            stdin_text = "7\n5\n1\nn\n0\n0\nq\n"
        elif population == "lowdin":
            stdin_text = "7\n6\n\nn\n0\nq\n"
        else:
            raise ValueError("Multiwfn population_analysis must be mulliken or lowdin")
    else:
        return unsupported(f"Multiwfn does not implement {action_id}")

    completed = run_external(
        executable="Multiwfn_noGUI",
        environment_variable="CHEMGRAPH_MULTIWFN_COMMAND",
        arguments=[staged.name],
        directory=directory,
        stdin_text=stdin_text,
        timeout_seconds=int(request.get("resource_limits", {}).get("walltime_seconds", 1800)),
    )
    (directory / "stdout.log").write_text(completed["stdout"], encoding="utf-8")
    (directory / "stderr.log").write_text(completed["stderr"], encoding="utf-8")
    if not completed["available"]:
        return unavailable(completed["stderr"], install="Configure Multiwfn 2026.7.15 noGUI")
    if completed["returncode"] != 0:
        raise RuntimeError(f"Multiwfn failed: {completed['stderr'][-3000:]}")
    output = re.sub(r"\x1b\[[0-9;]*m", "", completed["stdout"])
    if f"Loaded {staged.name} successfully!" not in output:
        raise RuntimeError("Multiwfn did not confirm loading the supplied wavefunction")
    energy_match = re.search(r"Total energy:\s*([-+0-9.eEdD]+)\s+Hartree", output)
    energy = (
        float(energy_match.group(1).replace("D", "E").replace("d", "e"))
        if energy_match
        else None
    )
    if action_id == "calculate_bond_orders":
        bonds = []
        for match in re.finditer(
            r"^\s*#\s*\d+:\s+(\d+)\(([^)]*)\)\s+(\d+)\(([^)]*)\)\s+"
            r"([-+0-9.eEdD]+)\s*$",
            output,
            flags=re.M,
        ):
            value = float(match.group(5).replace("D", "E").replace("d", "e"))
            if abs(value) < minimum:
                continue
            bonds.append(
                {
                    "atom_index_a": int(match.group(1)) - 1,
                    "element_a": match.group(2).strip(),
                    "atom_index_b": int(match.group(3)) - 1,
                    "element_b": match.group(4).strip(),
                    "bond_order": value,
                }
            )
        if not bonds:
            raise RuntimeError("Multiwfn returned no bond orders at the requested threshold")
        result = {
            "bonds": bonds,
            "analysis": definition,
            "minimum_bond_order": minimum,
            "wavefunction_energy_hartree": energy,
        }
        semantic_type = "BondOrderResult"
    else:
        charges = []
        populations = []
        atoms = []
        for match in re.finditer(
            r"^\s*Atom\s+(\d+)\(([^)]*)\)\s+Population:\s*([-+0-9.eEdD]+)"
            r"\s+Net charge:\s*([-+0-9.eEdD]+)\s*$",
            output,
            flags=re.M,
        ):
            atoms.append(
                {"atom_index": int(match.group(1)) - 1, "element": match.group(2).strip()}
            )
            populations.append(float(match.group(3).replace("D", "E").replace("d", "e")))
            charges.append(float(match.group(4).replace("D", "E").replace("d", "e")))
        if not charges:
            raise RuntimeError("Multiwfn returned no atomic population/charge table")
        result = {
            "atoms": atoms,
            "charges": charges,
            "electron_populations": populations,
            "total_charge": float(sum(charges)),
            "analysis": population,
            "unit": "elementary_charge",
            "wavefunction_energy_hartree": energy,
        }
        semantic_type = "AtomicChargeResult"
    result_path = write_json(directory, "result.json", result)
    result_relative = relative_workspace_path(result_path)
    artifacts = [
        item for item in command_artifacts(directory) if item["path"] != result_relative
    ]
    artifacts.append(
        {
            "path": result_relative,
            "semantic_type": semantic_type,
            "media_type": "application/json",
        }
    )
    return success(
        result,
        artifact_files=artifacts,
        backend_version="2026.7.15",
        provenance={
            "command": completed["command"],
            "menu_sequence": stdin_text.splitlines(),
            "source_wavefunction": relative_workspace_path(source),
            "required_citations": [
                "T. Lu and F. Chen, J. Comput. Chem. 33, 580 (2012), DOI 10.1002/jcc.22885",
                "T. Lu, J. Chem. Phys. 161, 082503 (2024), DOI 10.1063/5.0216272",
            ],
            "arbitrary_menu_input_allowed": False,
        },
    )


_MULTIWFN_WAVEFUNCTION_SUFFIXES = {
    ".fch", ".fchk", ".wfn", ".wfx", ".mwfn", ".molden", ".47",
}
_MULTIWFN_NCI_VERSION = "2026.7.15"
_MULTIWFN_NCI_MAX_GRID_POINTS = 10_000_000
_MULTIWFN_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"


def _multiwfn_loaded(output: str, staged: Path) -> bool:
    return (
        f"Loaded {staged.name} successfully!" in output
        or f"Loaded {staged} successfully!" in output
    )


def _validated_multiwfn_nci_version(output: str) -> str:
    match = re.search(r"\bVersion\s+([0-9]+(?:\.[0-9]+)+)\b", output)
    if match is None:
        raise RuntimeError("Could not verify the Multiwfn version from its output banner")
    version = match.group(1)
    if version != _MULTIWFN_NCI_VERSION:
        raise RuntimeError(
            "Multiwfn NCI/RDG menu contract requires version "
            f"{_MULTIWFN_NCI_VERSION}, but the configured executable reports {version}"
        )
    return version


def _multiwfn_number(value: str, *, context: str) -> float:
    try:
        number = float(value.replace("D", "E").replace("d", "e"))
    except ValueError as exc:
        raise ValueError(f"{context} contains a non-numeric coordinate") from exc
    if not math.isfinite(number):
        raise ValueError(f"{context} contains a non-finite coordinate")
    return number


def _validated_multiwfn_geometry(
    coordinates: list[tuple[float, float, float]],
    *,
    expected_atom_count: int,
    format_name: str,
) -> list[tuple[float, float, float]]:
    if expected_atom_count < 1:
        raise ValueError(f"{format_name} declares no atoms")
    if len(coordinates) != expected_atom_count:
        raise ValueError(
            f"{format_name} coordinate count is incomplete: declared "
            f"{expected_atom_count}, parsed {len(coordinates)}"
        )
    if not all(
        len(coordinate) == 3 and all(math.isfinite(value) for value in coordinate)
        for coordinate in coordinates
    ):
        raise ValueError(f"{format_name} contains an invalid nuclear coordinate")
    return coordinates


def _read_fchk_values(
    lines: Any,
    *,
    count: int,
    label: str,
    integer: bool,
) -> list[int] | list[float]:
    values: list[int] | list[float] = []
    while len(values) < count:
        try:
            line = next(lines)
        except StopIteration as exc:
            raise ValueError(f"FCH/FCHK {label} array is truncated") from exc
        tokens = line.split()
        if len(values) + len(tokens) > count:
            raise ValueError(f"FCH/FCHK {label} array has more values than declared")
        for token in tokens:
            if integer:
                try:
                    values.append(int(token))
                except ValueError as exc:
                    raise ValueError(
                        f"FCH/FCHK {label} array contains a non-integer value"
                    ) from exc
            else:
                values.append(
                    _multiwfn_number(token, context=f"FCH/FCHK {label} array")
                )
    return values


def _parse_fchk_nuclear_coordinates(path: Path) -> list[tuple[float, float, float]]:
    atom_count: int | None = None
    atomic_numbers: list[int] | None = None
    coordinate_values: list[float] | None = None
    with path.open(encoding="utf-8", errors="replace") as handle:
        lines = iter(handle)
        for line in lines:
            atom_match = re.match(r"^\s*Number of atoms\s+I\s+(-?\d+)\s*$", line)
            if atom_match is not None:
                atom_count = int(atom_match.group(1))
                continue
            atomic_number_match = re.match(
                r"^\s*Atomic numbers\s+I\s+N=\s*(\d+)\s*$", line
            )
            if atomic_number_match is not None:
                atomic_numbers = _read_fchk_values(
                    lines,
                    count=int(atomic_number_match.group(1)),
                    label="Atomic numbers",
                    integer=True,
                )
                continue
            coordinate_match = re.match(
                r"^\s*Current cartesian coordinates\s+R\s+N=\s*(\d+)\s*$",
                line,
            )
            if coordinate_match is not None:
                coordinate_values = _read_fchk_values(
                    lines,
                    count=int(coordinate_match.group(1)),
                    label="Current cartesian coordinates",
                    integer=False,
                )
            if (
                atom_count is not None
                and atomic_numbers is not None
                and coordinate_values is not None
            ):
                break
    if atom_count is None or atomic_numbers is None or coordinate_values is None:
        raise ValueError(
            "FCH/FCHK must contain Number of atoms, Atomic numbers, and Current "
            "cartesian coordinates for bounded NCI grid preflight"
        )
    if len(atomic_numbers) != atom_count:
        raise ValueError(
            "FCH/FCHK Atomic numbers count does not match Number of atoms"
        )
    if len(coordinate_values) != 3 * atom_count:
        raise ValueError(
            "FCH/FCHK Current cartesian coordinates count does not match Number of atoms"
        )
    coordinates = [
        tuple(coordinate_values[index:index + 3])
        for index in range(0, len(coordinate_values), 3)
    ]
    return _validated_multiwfn_geometry(
        coordinates, expected_atom_count=atom_count, format_name="FCH/FCHK"
    )


def _parse_wfn_nuclear_coordinates(path: Path) -> list[tuple[float, float, float]]:
    atom_count: int | None = None
    coordinates: list[tuple[float, float, float]] = []
    atom_pattern = re.compile(
        rf"^\s*\S+\s+(\d+)\s+\(CENTRE\s+(\d+)\)\s+"
        rf"({_MULTIWFN_NUMBER})\s+({_MULTIWFN_NUMBER})\s+"
        rf"({_MULTIWFN_NUMBER})\s+CHARGE\s*=",
        flags=re.I,
    )
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if atom_count is None:
                count_match = re.search(r"\b(\d+)\s+NUCLEI\s*$", line, flags=re.I)
                if count_match is not None:
                    atom_count = int(count_match.group(1))
                continue
            if len(coordinates) >= atom_count:
                break
            atom_match = atom_pattern.match(line)
            if atom_match is None:
                raise ValueError("WFN nuclear-coordinate section is malformed or truncated")
            expected_index = len(coordinates) + 1
            if int(atom_match.group(1)) != expected_index or int(
                atom_match.group(2)
            ) != expected_index:
                raise ValueError("WFN nuclear centers are not contiguous and ordered")
            coordinates.append(
                tuple(
                    _multiwfn_number(
                        atom_match.group(index), context="WFN nuclear-coordinate section"
                    )
                    for index in (3, 4, 5)
                )
            )
    if atom_count is None:
        raise ValueError("WFN does not declare its number of nuclei")
    return _validated_multiwfn_geometry(
        coordinates, expected_atom_count=atom_count, format_name="WFN"
    )


def _read_wfx_sections(path: Path) -> dict[str, str]:
    wanted = {
        "number of nuclei",
        "atomic numbers",
        "nuclear cartesian coordinates",
    }
    sections: dict[str, str] = {}
    active: str | None = None
    chunks: list[str] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if active is None:
                opening = re.match(r"^\s*<([^/>]+)>\s*(.*)$", line)
                if opening is None:
                    continue
                name = opening.group(1).strip().casefold()
                if name not in wanted:
                    continue
                active = name
                remainder = opening.group(2)
                closing = re.search(rf"</{re.escape(opening.group(1).strip())}>", remainder, re.I)
                if closing is not None:
                    sections[active] = remainder[:closing.start()].strip()
                    active = None
                else:
                    chunks = [remainder]
            else:
                closing = re.search(rf"</{re.escape(active)}>", line, re.I)
                if closing is not None:
                    chunks.append(line[:closing.start()])
                    sections[active] = "".join(chunks).strip()
                    active = None
                    chunks = []
                else:
                    chunks.append(line)
            if wanted <= sections.keys():
                break
    if active is not None:
        raise ValueError(f"WFX <{active}> section is not closed")
    if not wanted <= sections.keys():
        missing = sorted(wanted - sections.keys())
        raise ValueError(
            "WFX lacks geometry fields required for bounded NCI grid preflight: "
            + ", ".join(missing)
        )
    return sections


def _parse_wfx_nuclear_coordinates(path: Path) -> list[tuple[float, float, float]]:
    sections = _read_wfx_sections(path)
    try:
        atom_count = int(sections["number of nuclei"])
        atomic_numbers = [
            int(value) for value in sections["atomic numbers"].split()
        ]
    except ValueError as exc:
        raise ValueError("WFX nucleus count or Atomic Numbers section is invalid") from exc
    if len(atomic_numbers) != atom_count:
        raise ValueError("WFX Atomic Numbers count does not match Number of Nuclei")
    values = [
        _multiwfn_number(value, context="WFX Nuclear Cartesian Coordinates")
        for value in sections["nuclear cartesian coordinates"].split()
    ]
    if len(values) != 3 * atom_count:
        raise ValueError(
            "WFX Nuclear Cartesian Coordinates count does not match Number of Nuclei"
        )
    coordinates = [
        tuple(values[index:index + 3]) for index in range(0, len(values), 3)
    ]
    return _validated_multiwfn_geometry(
        coordinates, expected_atom_count=atom_count, format_name="WFX"
    )


def _parse_mwfn_nuclear_coordinates(path: Path) -> list[tuple[float, float, float]]:
    atom_count: int | None = None
    in_centers = False
    coordinates: list[tuple[float, float, float]] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            count_match = re.match(r"^\s*Ncenter\s*=\s*(\d+)\s*$", line, re.I)
            if count_match is not None:
                atom_count = int(count_match.group(1))
                continue
            if line.strip().casefold() == "$centers":
                if atom_count is None:
                    raise ValueError("MWFN $Centers appears before Ncenter")
                in_centers = True
                continue
            if not in_centers:
                continue
            if len(coordinates) >= (atom_count or 0):
                break
            fields = line.split()
            if len(fields) != 7:
                raise ValueError("MWFN $Centers record is malformed or truncated")
            expected_index = len(coordinates) + 1
            try:
                center_index = int(fields[0])
                int(fields[2])
                _multiwfn_number(fields[3], context="MWFN $Centers nuclear charge")
            except ValueError as exc:
                raise ValueError("MWFN $Centers identifiers are invalid") from exc
            if center_index != expected_index:
                raise ValueError("MWFN $Centers records are not contiguous and ordered")
            coordinates.append(
                tuple(
                    _multiwfn_number(value, context="MWFN $Centers")
                    for value in fields[4:7]
                )
            )
    if atom_count is None or not in_centers:
        raise ValueError("MWFN must contain Ncenter and $Centers for grid preflight")
    return _validated_multiwfn_geometry(
        coordinates, expected_atom_count=atom_count, format_name="MWFN"
    )


def _parse_molden_nuclear_coordinates(
    path: Path,
) -> tuple[list[tuple[float, float, float]], str]:
    coordinates: list[tuple[float, float, float]] = []
    unit: str | None = None
    in_atoms = False
    following_section: str | None = None
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            header = re.match(
                r"^\s*\[Atoms\]\s*(?:\(([^)]+)\)|(\S+))?\s*$", line, re.I
            )
            if header is not None:
                raw_unit = (header.group(1) or header.group(2) or "").casefold()
                normalized_unit = raw_unit.replace(".", "")
                if normalized_unit in {"au", "bohr"}:
                    unit = "bohr"
                elif normalized_unit in {"angs", "angstrom", "angstroms"}:
                    unit = "angstrom"
                else:
                    raise ValueError(
                        "Molden [Atoms] must explicitly declare AU/Bohr or Angs/Angstrom"
                    )
                in_atoms = True
                continue
            if not in_atoms:
                continue
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", "!", ";")):
                continue
            if stripped.startswith("["):
                section_match = re.match(r"^\[([^]]+)\]", stripped)
                following_section = (
                    section_match.group(1).strip().casefold()
                    if section_match is not None
                    else None
                )
                break
            fields = stripped.split()
            if len(fields) != 6:
                raise ValueError("Molden [Atoms] record is malformed")
            expected_index = len(coordinates) + 1
            try:
                center_index = int(fields[1])
                int(fields[2])
            except ValueError as exc:
                raise ValueError("Molden [Atoms] identifiers are invalid") from exc
            if center_index != expected_index:
                raise ValueError("Molden [Atoms] records are not contiguous and ordered")
            coordinate = tuple(
                _multiwfn_number(value, context="Molden [Atoms]")
                for value in fields[3:6]
            )
            if unit == "angstrom":
                coordinate = tuple(value / BOHR_TO_ANGSTROM for value in coordinate)
            coordinates.append(coordinate)
    if not in_atoms or unit is None:
        raise ValueError("Molden file does not contain a unit-qualified [Atoms] section")
    if following_section != "gto":
        raise ValueError(
            "Molden [Atoms] section is not followed by [GTO]; the geometry or "
            "wavefunction may be truncated"
        )
    gto_centers: set[int] = set()
    in_gto = False
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            section_match = re.match(r"^\s*\[([^]]+)\]", line)
            if section_match is not None:
                section_name = section_match.group(1).strip().casefold()
                if section_name == "gto":
                    in_gto = True
                    continue
                if in_gto:
                    break
                continue
            if not in_gto:
                continue
            center_match = re.match(r"^\s*(\d+)\s+0\s*$", line)
            if center_match is not None:
                gto_centers.add(int(center_match.group(1)))
    if not gto_centers:
        raise ValueError("Molden [GTO] does not declare any basis-function centers")
    if min(gto_centers) < 1 or max(gto_centers) > len(coordinates):
        raise ValueError(
            "Molden [GTO] references an atom missing from [Atoms]; the geometry is "
            "incomplete for bounded NCI grid preflight"
        )
    return (
        _validated_multiwfn_geometry(
            coordinates,
            expected_atom_count=len(coordinates),
            format_name="Molden",
        ),
        unit,
    )


def _parse_nbo47_nuclear_coordinates(path: Path) -> list[tuple[float, float, float]]:
    coordinates: list[tuple[float, float, float]] = []
    atom_count: int | None = None
    in_coordinates = False
    skipped_title = False
    coordinates_closed = False
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not in_coordinates:
                count_match = re.search(r"\bNATOMS\s*=\s*(\d+)", line, re.I)
                if count_match is not None:
                    atom_count = int(count_match.group(1))
                if stripped.casefold() == "$coord":
                    in_coordinates = True
                continue
            if stripped.casefold() == "$end":
                coordinates_closed = True
                break
            if not skipped_title:
                skipped_title = True
                continue
            if not stripped:
                continue
            fields = stripped.split()
            if len(fields) != 5:
                raise ValueError("NBO .47 $COORD record is malformed")
            try:
                int(fields[0])
                _multiwfn_number(fields[1], context="NBO .47 nuclear charge")
            except ValueError as exc:
                raise ValueError("NBO .47 $COORD identifiers are invalid") from exc
            coordinate_angstrom = tuple(
                _multiwfn_number(value, context="NBO .47 $COORD")
                for value in fields[2:5]
            )
            coordinates.append(
                tuple(value / BOHR_TO_ANGSTROM for value in coordinate_angstrom)
            )
    if not in_coordinates:
        raise ValueError("NBO .47 file does not contain a $COORD section")
    if atom_count is None:
        raise ValueError("NBO .47 file does not declare NATOMS")
    if not coordinates_closed:
        raise ValueError("NBO .47 $COORD section is not closed")
    return _validated_multiwfn_geometry(
        coordinates,
        expected_atom_count=atom_count,
        format_name="NBO .47",
    )


def _multiwfn_nci_grid_preflight(
    source: Path,
    *,
    grid_spacing_bohr: float,
    grid_extension_bohr: float,
    max_grid_points: int,
) -> dict[str, Any]:
    suffix = source.suffix.casefold()
    try:
        if suffix in {".fch", ".fchk"}:
            coordinates = _parse_fchk_nuclear_coordinates(source)
            parser = "gaussian_formatted_checkpoint_bohr"
        elif suffix == ".wfn":
            coordinates = _parse_wfn_nuclear_coordinates(source)
            parser = "wfn_nuclear_coordinates_bohr"
        elif suffix == ".wfx":
            coordinates = _parse_wfx_nuclear_coordinates(source)
            parser = "wfx_nuclear_cartesian_coordinates_bohr"
        elif suffix == ".mwfn":
            coordinates = _parse_mwfn_nuclear_coordinates(source)
            parser = "mwfn_centers_bohr"
        elif suffix == ".molden":
            coordinates, molden_unit = _parse_molden_nuclear_coordinates(source)
            parser = f"molden_atoms_{molden_unit}"
        elif suffix == ".47":
            coordinates = _parse_nbo47_nuclear_coordinates(source)
            parser = "nbo47_coord_angstrom"
        else:
            raise ValueError(f"unsupported wavefunction suffix {suffix!r}")
    except (OSError, UnicodeError) as exc:
        raise ValueError(
            f"Could not read {suffix or 'wavefunction'} nuclear coordinates for bounded "
            "NCI grid preflight"
        ) from exc

    minimum = [min(coordinate[axis] for coordinate in coordinates) for axis in range(3)]
    maximum = [max(coordinate[axis] for coordinate in coordinates) for axis in range(3)]
    padded_span = [
        maximum[axis] - minimum[axis] + 2.0 * grid_extension_bohr
        for axis in range(3)
    ]
    # Multiwfn covers both box endpoints and may slightly reduce the requested spacing
    # to fit an integral interval count. ceil(intervals) + 1 is therefore a safe bound.
    estimated_shape = [
        math.ceil(span / grid_spacing_bohr) + 1 for span in padded_span
    ]
    estimated_point_count = math.prod(estimated_shape)
    preflight = {
        "coordinate_parser": parser,
        "atom_count": len(coordinates),
        "atom_bounding_box_bohr": {
            "minimum_xyz": minimum,
            "maximum_xyz": maximum,
        },
        "padded_span_bohr_xyz": padded_span,
        "estimated_shape_upper_bound_xyz": estimated_shape,
        "estimated_point_count_upper_bound": estimated_point_count,
        "max_grid_points": max_grid_points,
    }
    if estimated_point_count > max_grid_points:
        raise ValueError(
            "NCI grid preflight rejected the request before Multiwfn execution: estimated "
            f"upper bound {estimated_point_count} points exceeds max_grid_points="
            f"{max_grid_points}. Increase grid spacing, reduce extension, or explicitly "
            f"choose a higher limit no greater than {_MULTIWFN_NCI_MAX_GRID_POINTS}."
        )
    return preflight


def _parse_multiwfn_rdg_scatter(
    path: Path,
    *,
    rdg_cutoff: float,
    signed_density_abs_cutoff: float,
) -> dict[str, Any]:
    count = 0
    low_rdg = 0
    attractive = 0
    near_zero = 0
    repulsive = 0
    signed_min = math.inf
    signed_max = -math.inf
    rdg_min = math.inf
    rdg_max = -math.inf
    for line_number, line in enumerate(path.open(encoding="utf-8"), start=1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise RuntimeError(
                f"Multiwfn RDG scatter row {line_number} has {len(fields)} columns, expected 5"
            )
        try:
            values = [float(value.replace("D", "E").replace("d", "e")) for value in fields]
        except ValueError as exc:
            raise RuntimeError(
                f"Multiwfn RDG scatter row {line_number} is not numeric"
            ) from exc
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError(
                f"Multiwfn RDG scatter row {line_number} contains a non-finite value"
            )
        signed_density = values[3]
        rdg = values[4]
        count += 1
        signed_min = min(signed_min, signed_density)
        signed_max = max(signed_max, signed_density)
        rdg_min = min(rdg_min, rdg)
        rdg_max = max(rdg_max, rdg)
        if rdg <= rdg_cutoff and abs(signed_density) <= signed_density_abs_cutoff:
            low_rdg += 1
            if signed_density < 0:
                attractive += 1
            elif signed_density > 0:
                repulsive += 1
            else:
                near_zero += 1
    if count == 0:
        raise RuntimeError("Multiwfn returned an empty RDG scatter table")
    return {
        "point_count": count,
        "signed_density_min_au": signed_min,
        "signed_density_max_au": signed_max,
        "rdg_min_au": rdg_min,
        "rdg_max_au": rdg_max,
        "descriptive_window": {
            "rdg_max_au": rdg_cutoff,
            "signed_density_abs_max_au": signed_density_abs_cutoff,
            "point_count": low_rdg,
            "negative_signed_density_point_count": attractive,
            "zero_signed_density_point_count": near_zero,
            "positive_signed_density_point_count": repulsive,
        },
    }


def _parse_multiwfn_rdg_critical_points(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    blocks = re.split(r"(?=^\s*-+\s+CP\s+\d+)", text, flags=re.M)
    points: list[dict[str, Any]] = []
    for block in blocks:
        header = re.search(r"CP\s+(\d+),\s+Type\s+(\(3,[+-]\d\))", block)
        if header is None:
            continue
        position = re.search(
            r"Position \(Angstrom\):\s*([-+0-9.eEdD]+)\s+"
            r"([-+0-9.eEdD]+)\s+([-+0-9.eEdD]+)",
            block,
        )
        density = re.search(r"Density of all electrons:\s*([-+0-9.eEdD]+)", block)
        rdg = re.search(r"Reduced density gradient \(RDG\):\s*([-+0-9.eEdD]+)", block)
        signed = re.search(r"Sign\(lambda2\)\*rho:\s*([-+0-9.eEdD]+)", block)
        if None in {position, density, rdg, signed}:
            raise RuntimeError(
                f"Multiwfn critical-point record {header.group(1)} is incomplete"
            )
        number = lambda value: float(value.replace("D", "E").replace("d", "e"))
        points.append(
            {
                "critical_point_index": int(header.group(1)),
                "type": header.group(2),
                "position_angstrom": [number(position.group(index)) for index in (1, 2, 3)],
                "electron_density_au": number(density.group(1)),
                "rdg_au": number(rdg.group(1)),
                "signed_density_au": number(signed.group(1)),
            }
        )
    return points


def _multiwfn_nci_rdg(request: dict[str, Any]) -> dict[str, Any]:
    inputs, _method, settings = request_parts(request)
    source = resolve_input_file(inputs["wavefunction_file"])
    if not source.is_file():
        raise ValueError("Multiwfn NCI/RDG requires one regular molecular wavefunction file")
    suffix = source.suffix.casefold()
    if suffix == ".chk":
        raise ValueError(
            "Gaussian binary .chk is not accepted; run the configured formchk utility and "
            "supply its .fchk output so the analyzed wavefunction is explicit and auditable"
        )
    if suffix in {".cub", ".cube"}:
        raise ValueError(
            "A density-only cube is insufficient for wavefunction-based NCI/RDG because "
            "Multiwfn requires wavefunction derivatives to evaluate sign(lambda2)rho; supply "
            "FCH/FCHK/WFN/WFX/MWFN/Molden/47"
        )
    if suffix not in _MULTIWFN_WAVEFUNCTION_SUFFIXES:
        raise ValueError(
            "analyze_non_covalent_interactions accepts only FCH/FCHK/WFN/WFX/MWFN/"
            "Molden/47 molecular wavefunctions containing basis and orbital information"
        )

    spacing = float(settings["grid_spacing_bohr"])
    extension = float(settings["grid_extension_bohr"])
    rdg_cutoff = float(settings["rdg_cutoff"])
    signed_density_cutoff = float(settings["signed_density_abs_cutoff_au"])
    max_grid_points = settings["max_grid_points"]
    if not math.isfinite(spacing) or not 0.05 <= spacing <= 1.0:
        raise ValueError("grid_spacing_bohr must be finite and between 0.05 and 1.0")
    if not math.isfinite(extension) or not 0.5 <= extension <= 10.0:
        raise ValueError("grid_extension_bohr must be finite and between 0.5 and 10.0")
    if not math.isfinite(rdg_cutoff) or not 0.0 < rdg_cutoff <= 5.0:
        raise ValueError("rdg_cutoff must be finite, positive, and no greater than 5.0")
    if not math.isfinite(signed_density_cutoff) or not 1.0e-6 <= signed_density_cutoff <= 1.0:
        raise ValueError(
            "signed_density_abs_cutoff_au must be finite and between 1e-6 and 1.0"
        )
    if (
        isinstance(max_grid_points, bool)
        or not isinstance(max_grid_points, int)
        or not 1 <= max_grid_points <= _MULTIWFN_NCI_MAX_GRID_POINTS
    ):
        raise ValueError(
            "max_grid_points must be an integer between 1 and "
            f"{_MULTIWFN_NCI_MAX_GRID_POINTS}"
        )

    grid_preflight = _multiwfn_nci_grid_preflight(
        source,
        grid_spacing_bohr=spacing,
        grid_extension_bohr=extension,
        max_grid_points=max_grid_points,
    )

    directory = output_directory("analyze_non_covalent_interactions", "multiwfn")
    staged = directory / source.name
    shutil.copy2(source, staged)
    cores = int(request.get("resource_limits", {}).get("cpu_cores") or 1)
    timeout_seconds = int(
        request.get("resource_limits", {}).get("walltime_seconds", 1800)
    )
    environment = {
        "OMP_NUM_THREADS": str(cores),
        "OMP_STACKSIZE": os.environ.get("OMP_STACKSIZE", "2G"),
        "KMP_STACKSIZE": os.environ.get("KMP_STACKSIZE", "4G"),
    }

    grid_menu = (
        f"20\n1\n-10\n{extension:.10g}\n4\n{spacing:.10g}\n"
        "2\n3\n0\n0\nq\n"
    )
    grid_run = run_external(
        executable="Multiwfn_noGUI",
        environment_variable="CHEMGRAPH_MULTIWFN_COMMAND",
        arguments=[staged.name],
        directory=directory,
        stdin_text=grid_menu,
        timeout_seconds=timeout_seconds,
        environment_overrides=environment,
    )
    (directory / "nci_grid_stdout.log").write_text(grid_run["stdout"], encoding="utf-8")
    (directory / "nci_grid_stderr.log").write_text(grid_run["stderr"], encoding="utf-8")
    if not grid_run["available"]:
        return unavailable(grid_run["stderr"], install="Configure Multiwfn 2026.7.15 noGUI")
    if grid_run["returncode"] != 0:
        raise RuntimeError(
            "Multiwfn NCI/RDG grid calculation failed: "
            + (grid_run["stderr"] or grid_run["stdout"])[-3000:]
        )
    grid_output = re.sub(r"\x1b\[[0-9;]*m", "", grid_run["stdout"])
    grid_version = _validated_multiwfn_nci_version(grid_output)
    if not _multiwfn_loaded(grid_output, staged):
        raise RuntimeError("Multiwfn did not confirm loading the supplied NCI wavefunction")

    scatter_path = directory / "output.txt"
    signed_density_grid = directory / "func1.cub"
    rdg_grid = directory / "func2.cub"
    expected_markers = (
        "Column 4/5: sign(lambda2)rho and RDG in a.u.",
        "The cube file of sign(lambda2)rho has been exported to func1.cub",
        "The cube file of RDG has been exported to func2.cub",
    )
    if not all(marker in grid_output for marker in expected_markers) or not all(
        path.is_file() and path.stat().st_size > 0
        for path in (scatter_path, signed_density_grid, rdg_grid)
    ):
        raise RuntimeError("Multiwfn did not produce the complete requested NCI/RDG artifacts")

    grid_match = re.search(
        r"Grid spacing in X,Y,Z is\s*([-+0-9.eEdD]+)\s+"
        r"([-+0-9.eEdD]+)\s+([-+0-9.eEdD]+)\s+Bohr.*?"
        r"Number of points in X,Y,Z is\s*(\d+)\s+(\d+)\s+(\d+)\s+"
        r"Total:\s*(\d+)",
        grid_output,
        flags=re.S,
    )
    if grid_match is None:
        raise RuntimeError("Could not parse the actual Multiwfn NCI grid dimensions")
    actual_shape = [int(grid_match.group(index)) for index in (4, 5, 6)]
    actual_point_count = int(grid_match.group(7))
    if math.prod(actual_shape) != actual_point_count:
        raise RuntimeError(
            "Multiwfn NCI grid shape does not match its reported total point count"
        )
    if actual_point_count > max_grid_points:
        raise RuntimeError(
            "Multiwfn generated more NCI grid points than the caller's max_grid_points "
            f"limit: actual={actual_point_count}, limit={max_grid_points}"
        )
    estimated_shape = grid_preflight["estimated_shape_upper_bound_xyz"]
    if any(actual > estimated for actual, estimated in zip(actual_shape, estimated_shape)):
        raise RuntimeError(
            "Multiwfn actual NCI grid dimensions exceeded the coordinate-based preflight "
            f"upper bound: actual={actual_shape}, estimated={estimated_shape}"
        )

    topology_menu = "2\n-11\n13\n3\n7\n-1\n-10\nq\n"
    topology_run = run_external(
        executable="Multiwfn_noGUI",
        environment_variable="CHEMGRAPH_MULTIWFN_COMMAND",
        arguments=[staged.name],
        directory=directory,
        stdin_text=topology_menu,
        timeout_seconds=timeout_seconds,
        environment_overrides=environment,
    )
    (directory / "rdg_critical_points_stdout.log").write_text(
        topology_run["stdout"], encoding="utf-8"
    )
    (directory / "rdg_critical_points_stderr.log").write_text(
        topology_run["stderr"], encoding="utf-8"
    )
    if not topology_run["available"]:
        return unavailable(
            topology_run["stderr"], install="Configure Multiwfn 2026.7.15 noGUI"
        )
    if topology_run["returncode"] != 0:
        raise RuntimeError(
            "Multiwfn RDG critical-point calculation failed: "
            + (topology_run["stderr"] or topology_run["stdout"])[-3000:]
        )
    topology_output = re.sub(r"\x1b\[[0-9;]*m", "", topology_run["stdout"])
    topology_version = _validated_multiwfn_nci_version(topology_output)
    if topology_version != grid_version:
        raise RuntimeError("Multiwfn version changed between the NCI grid and topology runs")
    if not _multiwfn_loaded(topology_output, staged):
        raise RuntimeError("Multiwfn did not confirm loading the NCI topology wavefunction")
    number = lambda value: float(value.replace("D", "E").replace("d", "e"))
    scatter_summary = _parse_multiwfn_rdg_scatter(
        scatter_path,
        rdg_cutoff=rdg_cutoff,
        signed_density_abs_cutoff=signed_density_cutoff,
    )
    if scatter_summary["point_count"] != int(grid_match.group(7)):
        raise RuntimeError(
            "Multiwfn scatter-table row count does not match its reported grid point count"
        )
    found_match = re.search(
        r"Totally find\s+(\d+)\s+new critical points", topology_output
    )
    export_complete = re.search(
        r"Done!\s+The results have been outputted to CPprop\.txt", topology_output
    )
    if found_match is None or export_complete is None:
        raise RuntimeError(
            "Multiwfn did not confirm a complete RDG critical-point search and export"
        )
    reported_count = int(found_match.group(1))
    critical_points_path = directory / "CPprop.txt"
    if reported_count == 0:
        critical_points = (
            _parse_multiwfn_rdg_critical_points(critical_points_path)
            if critical_points_path.is_file()
            else []
        )
    else:
        if (
            not critical_points_path.is_file()
            or critical_points_path.stat().st_size == 0
        ):
            raise RuntimeError("Multiwfn did not produce RDG critical-point property data")
        critical_points = _parse_multiwfn_rdg_critical_points(critical_points_path)
    indices = [point["critical_point_index"] for point in critical_points]
    if reported_count != len(critical_points) or indices != list(
        range(1, reported_count + 1)
    ):
        raise RuntimeError(
            "Multiwfn RDG critical-point export is incomplete or has non-contiguous indices: "
            f"reported={reported_count}, parsed={len(critical_points)}"
        )
    result = {
        "source_wavefunction": relative_workspace_path(source),
        "analysis": "wavefunction_based_nci_rdg",
        "grid": {
            "requested_spacing_bohr": spacing,
            "actual_spacing_bohr_xyz": [number(grid_match.group(index)) for index in (1, 2, 3)],
            "extension_bohr": extension,
            "shape_xyz": actual_shape,
            "point_count": actual_point_count,
            "resource_preflight": grid_preflight,
        },
        "scatter_summary": scatter_summary,
        "critical_points": critical_points,
        "interpretation_boundary": (
            "NCI/RDG is descriptive real-space evidence for interaction regions. It does not "
            "by itself establish a transition state's connectivity, relative free energy, "
            "enantioselectivity, or causal origin of selectivity. Compare candidates only with "
            "matched wavefunction methods, grid settings, and separately validated TS/IRC and "
            "thermochemical evidence."
        ),
    }
    result_path = write_json(directory, "result.json", result)
    artifacts = command_artifacts(directory)
    semantic_types = {
        relative_workspace_path(result_path): ("NonCovalentInteractionAnalysisResult", "application/json"),
        relative_workspace_path(scatter_path): ("NCIRDGScatterTable", "text/plain"),
        relative_workspace_path(signed_density_grid): ("SignedDensityGrid", "application/vnd.gaussian.cube"),
        relative_workspace_path(rdg_grid): ("ReducedDensityGradientGrid", "application/vnd.gaussian.cube"),
        relative_workspace_path(critical_points_path): ("RDGCriticalPointData", "text/plain"),
    }
    for artifact in artifacts:
        if artifact["path"] in semantic_types:
            artifact["semantic_type"], artifact["media_type"] = semantic_types[artifact["path"]]
    return success(
        result,
        artifact_files=artifacts,
        backend_version=grid_version,
        warnings=[result["interpretation_boundary"]],
        provenance={
            "commands": [grid_run["command"], topology_run["command"]],
            "menu_sequences": [grid_menu.splitlines(), topology_menu.splitlines()],
            "source_wavefunction": relative_workspace_path(source),
            "required_citations": [
                "T. Lu and F. Chen, J. Comput. Chem. 33, 580 (2012), DOI 10.1002/jcc.22885",
                "T. Lu, J. Chem. Phys. 161, 082503 (2024), DOI 10.1063/5.0216272",
                "E. R. Johnson et al., J. Am. Chem. Soc. 132, 6498 (2010), DOI 10.1021/ja100936w",
            ],
            "arbitrary_menu_input_allowed": False,
            "critical_point_seeding": "all_atom_pair_midpoints",
        },
    )


def _multiwfn_isodensity_surface(request: dict[str, Any]) -> dict[str, Any]:
    inputs, _method, settings = request_parts(request)
    source = resolve_input_file(inputs["density_file"])
    if not source.is_file():
        raise ValueError("Multiwfn requires one regular wavefunction or cube file")
    suffix = source.suffix.casefold()
    wavefunction_suffixes = {
        ".fch", ".fchk", ".wfn", ".wfx", ".mwfn", ".molden", ".47",
    }
    if suffix == ".cube":
        surface_definition = 11
        source_kind = "external_density_grid"
    elif suffix in wavefunction_suffixes:
        surface_definition = 1
        source_kind = "wavefunction_electron_density"
    else:
        raise ValueError(
            "calculate_electron_isodensity_surface accepts WFN/WFX/FCHK/MWFN/"
            "Molden/47 wavefunctions or a cube density grid"
        )

    raw_cutoffs = settings["cutoffs_au"]
    normalized_cutoff_string = isinstance(raw_cutoffs, str)
    if normalized_cutoff_string:
        raw_cutoffs = [
            item for item in re.split(r"[,\s]+", raw_cutoffs.strip()) if item
        ]
    if not isinstance(raw_cutoffs, (list, tuple)) or not raw_cutoffs:
        raise ValueError(
            "cutoffs_au must be a non-empty JSON array of numbers; a comma-separated "
            "numeric string is accepted only as a compatibility input"
        )
    if len(raw_cutoffs) > 100:
        raise ValueError("cutoffs_au accepts at most 100 explicit values per Action")
    cutoffs: list[float] = []
    for value in raw_cutoffs:
        cutoff = float(value)
        if not math.isfinite(cutoff) or cutoff <= 0 or cutoff >= 0.1:
            raise ValueError("Every density cutoff must be finite and between 0 and 0.1 a.u.")
        if cutoff in cutoffs:
            raise ValueError("cutoffs_au must not contain duplicate values")
        cutoffs.append(cutoff)
    canonical_spacing_supplied = "grid_spacing_bohr" in settings
    legacy_spacing_supplied = "grid_spacing_angstrom" in settings
    if canonical_spacing_supplied and legacy_spacing_supplied:
        raise ValueError(
            "Supply exactly one of grid_spacing_bohr or the deprecated "
            "grid_spacing_angstrom compatibility alias"
        )
    if canonical_spacing_supplied:
        spacing_bohr = float(settings["grid_spacing_bohr"])
        spacing_input_field = "grid_spacing_bohr"
    elif legacy_spacing_supplied:
        # Compatibility promise: old requests already used this numeric value as
        # bohr.  Preserve that behavior while reporting the physical unit honestly.
        spacing_bohr = float(settings["grid_spacing_angstrom"])
        spacing_input_field = "grid_spacing_angstrom"
    else:
        raise ValueError(
            "calculate_electron_isodensity_surface requires grid_spacing_bohr; "
            "legacy calls may instead supply deprecated grid_spacing_angstrom"
        )
    if not math.isfinite(spacing_bohr) or spacing_bohr < 0.02 or spacing_bohr > 1.0:
        raise ValueError("grid_spacing_bohr must be between 0.02 and 1.0")

    directory = output_directory("calculate_electron_isodensity_surface", "multiwfn")
    staged = directory / source.name
    shutil.copy2(source, staged)
    cores = int(request.get("resource_limits", {}).get("cpu_cores") or 1)
    timeout_seconds = int(
        request.get("resource_limits", {}).get("walltime_seconds", 1800)
    )
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    commands: list[list[str]] = []
    for index, cutoff in enumerate(cutoffs):
        stdin_text = (
            f"12\n1\n{surface_definition}\n{cutoff:.10g}\n"
            f"3\n{spacing_bohr:.10g}\n6\n-1\n-1\nq\n"
        )
        completed = run_external(
            executable="Multiwfn_noGUI",
            environment_variable="CHEMGRAPH_MULTIWFN_COMMAND",
            arguments=[staged.name],
            directory=directory,
            stdin_text=stdin_text,
            timeout_seconds=timeout_seconds,
            environment_overrides={
                "OMP_NUM_THREADS": str(cores),
                "OMP_STACKSIZE": os.environ.get("OMP_STACKSIZE", "2G"),
            },
        )
        commands.append(completed["command"])
        stdout_path = directory / f"cutoff_{index:03d}_{cutoff:.7f}.out"
        stderr_path = directory / f"cutoff_{index:03d}_{cutoff:.7f}.err"
        stdout_path.write_text(completed["stdout"], encoding="utf-8")
        stderr_path.write_text(completed["stderr"], encoding="utf-8")
        if not completed["available"]:
            return unavailable(
                completed["stderr"], install="Configure Multiwfn 2026.7.15 noGUI"
            )
        output = re.sub(r"\x1b\[[0-9;]*m", "", completed["stdout"])
        loaded = (
            f"Loaded {staged.name} successfully!" in output
            or f"Loaded {staged} successfully!" in output
        )
        area_match = re.search(
            r"Isosurface area:\s*[-+0-9.eEdD]+\s+Bohr\^2\s*"
            r"\(\s*([-+0-9.eEdD]+)\s+Angstrom\^2\)",
            output,
        )
        volume_match = re.search(
            r"Volume enclosed by the isosurface:\s*[-+0-9.eEdD]+\s+Bohr\^3\s*"
            r"\(\s*([-+0-9.eEdD]+)\s+Angstrom\^3\)",
            output,
        )
        if completed["returncode"] != 0 or not loaded or not area_match or not volume_match:
            failures.append(
                {
                    "cutoff_au": cutoff,
                    "returncode": completed["returncode"],
                    "loaded": loaded,
                    "message": (completed["stderr"] or output)[-2000:],
                }
            )
            continue
        rows.append(
            {
                "cutoff_au": cutoff,
                "surface_area_angstrom2": float(
                    area_match.group(1).replace("D", "E").replace("d", "e")
                ),
                "enclosed_volume_angstrom3": float(
                    volume_match.group(1).replace("D", "E").replace("d", "e")
                ),
            }
        )

    result = {
        "source_file": relative_workspace_path(source),
        "source_kind": source_kind,
        "density_unit": "electrons/bohr^3",
        "surface_area_unit": "angstrom^2",
        "volume_unit": "angstrom^3",
        "grid_spacing_bohr": spacing_bohr,
        "grid_spacing_angstrom": spacing_bohr * BOHR_TO_ANGSTROM,
        "grid_spacing_input_field": spacing_input_field,
        "surfaces": rows,
        "failures": failures,
    }
    result_path = write_json(directory, "result.json", result)
    artifacts = command_artifacts(directory)
    result_relative = relative_workspace_path(result_path)
    for item in artifacts:
        if item["path"] == result_relative:
            item["semantic_type"] = "ElectronIsodensitySurfaceResult"
            item["media_type"] = "application/json"
    common = {
        "artifact_files": artifacts,
        "backend_version": "2026.7.15",
        "warnings": (
            (
                [f"{len(failures)} of {len(cutoffs)} requested cutoffs failed"]
                if failures
                else []
            )
            + (
                [
                    "action_settings.cutoffs_au was normalized from a numeric string; "
                    "send a JSON array of numbers in new requests"
                ]
                if normalized_cutoff_string
                else []
            )
            + (
                [
                    "action_settings.grid_spacing_angstrom is deprecated and its numeric "
                    "value was interpreted in bohr for compatibility; use grid_spacing_bohr"
                ]
                if legacy_spacing_supplied
                else []
            )
        ),
        "provenance": {
            "commands": commands,
            "menu_template": [
                12, 1, surface_definition, "<cutoff>", 3, spacing_bohr,
                6, -1, -1, "q",
            ],
            "grid_spacing_bohr": spacing_bohr,
            "grid_spacing_angstrom": spacing_bohr * BOHR_TO_ANGSTROM,
            "grid_spacing_input_field": spacing_input_field,
            "cutoffs_input_form": (
                "normalized_numeric_string"
                if normalized_cutoff_string
                else "json_number_array"
            ),
            "parallel_threads": cores,
            "source_density_file": relative_workspace_path(source),
            "required_citations": [
                "T. Lu and F. Chen, J. Comput. Chem. 33, 580 (2012), DOI 10.1002/jcc.22885",
                "T. Lu, J. Chem. Phys. 161, 082503 (2024), DOI 10.1063/5.0216272",
                "T. Lu and F. Chen, J. Mol. Graph. Model. 38, 314-323 (2012)",
            ],
            "arbitrary_menu_input_allowed": False,
        },
    }
    if failures:
        return partial_success(result, **common)
    return success(result, **common)


def _critic2_path(path: Path) -> str:
    text = str(path)
    if any(character.isspace() for character in text) or '"' in text:
        raise ValueError("Staged Critic2 input paths cannot contain whitespace or quotes")
    return text


def _critic2_vector(value: Any, *, name: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must be a three-number vector")
    vector = [float(item) for item in value]
    if not all(math.isfinite(item) for item in vector):
        raise ValueError(f"{name} must contain finite numbers")
    return vector


def _critic2_seed_tokens(value: Any, *, system_type: str) -> list[str]:
    if isinstance(value, str):
        if value.strip().lower() != "default":
            raise ValueError("seed_strategy string must be 'default'; custom seeds use typed objects")
        return []
    seeds = [value] if isinstance(value, dict) else value
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("seed_strategy must be 'default', one typed seed object, or a non-empty list")
    tokens: list[str] = []
    for seed in seeds:
        if not isinstance(seed, dict):
            raise ValueError("Every custom Critic2 seed must be an object")
        kind = str(seed.get("type") or "").strip().lower()
        if kind == "pair":
            distance = float(seed["distance"])
            points = int(seed["points"])
            if distance <= 0 or points < 1:
                raise ValueError("PAIR seed distance and points must be positive")
            tokens.extend(["SEED", "PAIR", "DIST", f"{distance:.16g}", "NPTS", str(points)])
        elif kind == "triplet":
            distance = float(seed["distance"])
            if distance <= 0:
                raise ValueError("TRIPLET seed distance must be positive")
            tokens.extend(["SEED", "TRIPLET", "DIST", f"{distance:.16g}"])
        elif kind == "ws":
            if system_type != "crystal":
                raise ValueError("WS seeding is only valid for system_type=crystal")
            depth = int(seed["depth"])
            if depth < 0:
                raise ValueError("WS seed depth must be nonnegative")
            tokens.extend(["SEED", "WS", "DEPTH", str(depth)])
            if "origin" in seed:
                tokens.extend(["X0", *[f"{item:.16g}" for item in _critic2_vector(seed["origin"], name="WS origin")]])
            if "radius" in seed:
                radius = float(seed["radius"])
                if radius <= 0:
                    raise ValueError("WS seed radius must be positive")
                tokens.extend(["RADIUS", f"{radius:.16g}"])
        elif kind == "mesh":
            if system_type != "molecule":
                raise ValueError("MESH seeding is only valid for system_type=molecule")
            tokens.extend(["SEED", "MESH"])
        elif kind == "point":
            point = _critic2_vector(seed["position"], name="POINT position")
            tokens.extend(["SEED", "POINT", "X0", *[f"{item:.16g}" for item in point]])
        elif kind == "line":
            start = _critic2_vector(seed["start"], name="LINE start")
            end = _critic2_vector(seed["end"], name="LINE end")
            points = int(seed["points"])
            if points < 1:
                raise ValueError("LINE seed points must be positive")
            tokens.extend(
                [
                    "SEED", "LINE", "X0", *[f"{item:.16g}" for item in start],
                    "X1", *[f"{item:.16g}" for item in end], "NPTS", str(points),
                ]
            )
        elif kind in {"sphere", "oh"}:
            center = _critic2_vector(seed["center"], name=f"{kind.upper()} center")
            radius = float(seed["radius"])
            radial_points = int(seed["radial_points"])
            if radius <= 0 or radial_points < 1:
                raise ValueError(f"{kind.upper()} radius and radial_points must be positive")
            tokens.extend(
                [
                    "SEED", kind.upper(), "X0", *[f"{item:.16g}" for item in center],
                    "RADIUS", f"{radius:.16g}",
                ]
            )
            if kind == "sphere":
                polar_points = int(seed["polar_points"])
                azimuthal_points = int(seed["azimuthal_points"])
                if polar_points < 1 or azimuthal_points < 1:
                    raise ValueError("SPHERE angular point counts must be positive")
                tokens.extend(
                    [
                        "NTHETA", str(polar_points), "NPHI", str(azimuthal_points),
                        "NR", str(radial_points),
                    ]
                )
            else:
                depth = int(seed["depth"])
                if depth < 0:
                    raise ValueError("OH seed depth must be nonnegative")
                tokens.extend(["DEPTH", str(depth), "NR", str(radial_points)])
        else:
            raise ValueError(
                "Custom Critic2 seed type must be pair, triplet, ws, mesh, point, line, sphere, or oh"
            )
    return tokens


def _critic2_cp_type(item: dict[str, Any]) -> str:
    if bool(item.get("is_nucleus")):
        return "nucleus"
    signature = int(item.get("signature", 99))
    return {
        -3: "non_nuclear_attractor",
        -1: "bond",
        1: "ring",
        3: "cage",
    }.get(signature, "unclassified")


def _critic2_stage_inputs(
    action_id: str,
    inputs: dict[str, Any],
) -> tuple[Path, Path, Path, Path, Path]:
    density_path = resolve_input_file(inputs["density_file"])
    structure_path = resolve_input_file(inputs.get("structure_file", inputs["density_file"]))
    directory = output_directory(action_id, "critic2")

    def stage_input(source: Path, role: str) -> Path:
        name = source.name
        if not name or any(character.isspace() or character == '"' for character in name):
            name = f"{role}_input{source.suffix}"
        target = directory / name
        if target.exists() and target.resolve() != source.resolve():
            target = directory / f"{role}_input{source.suffix}"
        try:
            target.symlink_to(source.resolve())
        except OSError:
            shutil.copy2(source, target)
        return target

    density_local = stage_input(density_path, "density")
    structure_local = (
        density_local
        if structure_path.resolve() == density_path.resolve()
        else stage_input(structure_path, "structure")
    )
    return directory, density_path, structure_path, density_local, structure_local


def _critic2(action_id: str, request: dict[str, Any]) -> dict[str, Any]:
    if action_id in {"calculate_atomic_basin_properties", "calculate_bader_charges"}:
        return _critic2_basins(action_id, request)
    if action_id != "analyze_electron_density_topology":
        return unsupported(f"Critic2 does not implement {action_id}")
    inputs, _method, settings = request_parts(request)
    (
        directory,
        density_path,
        structure_path,
        density_local,
        structure_local,
    ) = _critic2_stage_inputs(action_id, inputs)
    system_type = str(settings["system_type"]).strip().lower()
    if system_type not in {"molecule", "crystal"}:
        raise ValueError("system_type must be molecule or crystal")

    density_format = str(settings["density_format"]).strip().lower()
    format_keywords = {
        "auto": "", "cube": "CUBE", "bincube": "BINCUBE", "abinit": "ABINIT",
        "vasp": "VASP", "vaspnov": "VASPNOV", "qub": "QUB", "xsf": "XSF",
        "fmt": "FMT", "txt": "TXT", "dat": "DAT", "elkgrid": "ELKGRID",
        "siesta": "SIESTA", "fplo": "FPLO", "dftb": "DFTB", "wfn": "WFN",
        "wfx": "WFX", "molden": "MOLDEN", "molden_orca": "MOLDEN_ORCA",
        "molden_psi4": "MOLDEN_PSI4", "fchk": "FCHK", "pwc": "PWC",
    }
    if density_format not in format_keywords:
        raise ValueError(f"Unsupported Critic2 density_format: {density_format}")
    interpolation = str(settings["interpolation"]).strip().lower()
    allowed_interpolation = {
        "native", "nearest", "trilinear", "trispline", "tricubic", "smoothrho",
    }
    if interpolation not in allowed_interpolation:
        raise ValueError(
            "interpolation must be native, nearest, trilinear, trispline, tricubic, or smoothrho"
        )

    raw_types = settings["critical_point_types"]
    type_aliases = {
        "n": "n", "nucleus": "n", "nuclear": "n",
        "b": "b", "bond": "b",
        "r": "r", "ring": "r",
        "c": "c", "cage": "c",
    }
    if isinstance(raw_types, str):
        compact = "".join(raw_types.lower().replace(",", " ").split())
        if not compact or any(character not in "nbrc" for character in compact):
            raise ValueError("critical_point_types string must contain only n, b, r, and c")
        cp_types = "".join(character for character in "nbrc" if character in compact)
    elif isinstance(raw_types, (list, tuple)) and raw_types:
        try:
            selected = {type_aliases[str(item).strip().lower()] for item in raw_types}
        except KeyError as exc:
            raise ValueError(f"Unknown critical-point type: {exc.args[0]}") from exc
        cp_types = "".join(character for character in "nbrc" if character in selected)
    else:
        raise ValueError("critical_point_types must be a non-empty string or list")

    gradient_tolerance = float(settings["gradient_tolerance"])
    if not math.isfinite(gradient_tolerance) or gradient_tolerance <= 0:
        raise ValueError("gradient_tolerance must be a positive finite number")
    auto_tokens = ["AUTO", "TYPES", cp_types, "GRADEPS", f"{gradient_tolerance:.16g}"]
    auto_tokens.extend(_critic2_seed_tokens(settings["seed_strategy"], system_type=system_type))
    for setting_name, keyword in (
        ("critical_point_equivalence_distance", "CPEPS"),
        ("nuclear_equivalence_distance", "NUCEPS"),
        ("hydrogen_nuclear_equivalence_distance", "NUCEPSH"),
        ("degeneracy_tolerance", "EPSDEGEN"),
    ):
        if setting_name in settings:
            value = float(settings[setting_name])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{setting_name} must be a positive finite number")
            auto_tokens.extend([keyword, f"{value:.16g}"])
    region = settings.get("search_region")
    if region is not None:
        if not isinstance(region, dict):
            raise ValueError("search_region must be a typed object")
        shape = str(region.get("shape") or "").strip().lower()
        if shape == "cube":
            minimum = _critic2_vector(region["minimum"], name="search-region minimum")
            maximum = _critic2_vector(region["maximum"], name="search-region maximum")
            auto_tokens.extend(
                ["CLIP", "CUBE", *[f"{item:.16g}" for item in [*minimum, *maximum]]]
            )
        elif shape == "sphere":
            center = _critic2_vector(region["center"], name="search-region center")
            radius = float(region["radius"])
            if radius <= 0:
                raise ValueError("search-region sphere radius must be positive")
            auto_tokens.extend(
                ["CLIP", "SPHERE", *[f"{item:.16g}" for item in center], f"{radius:.16g}"]
            )
        else:
            raise ValueError("search_region.shape must be cube or sphere")
    if "discard_density_below" in settings:
        threshold = float(settings["discard_density_below"])
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError("discard_density_below must be a finite nonnegative number")
        auto_tokens.extend(["DISCARD", f'"$1 < {threshold:.16g}"'])
    if bool(settings.get("checkpoint_critical_points", False)):
        auto_tokens.append("CHK")

    load_tokens = ["LOAD"]
    if format_keywords[density_format]:
        load_tokens.append(format_keywords[density_format])
    load_tokens.append(_critic2_path(Path(density_local.name)))
    if interpolation != "native":
        load_tokens.append(interpolation.upper())
    if interpolation == "smoothrho":
        if "smoothrho_environment_nodes" in settings:
            nodes = int(settings["smoothrho_environment_nodes"])
            if nodes < 1:
                raise ValueError("smoothrho_environment_nodes must be positive")
            load_tokens.extend(["NENV", str(nodes)])
        if "smoothrho_distance_factor" in settings:
            factor = float(settings["smoothrho_distance_factor"])
            if not math.isfinite(factor) or factor <= 0:
                raise ValueError("smoothrho_distance_factor must be positive and finite")
            load_tokens.extend(["FDMAX", f"{factor:.16g}"])

    report_detail = str(settings["report_detail"]).strip().lower()
    if report_detail not in {"short", "long", "verylong"}:
        raise ValueError("report_detail must be short, long, or verylong")
    max_reported = int(settings["max_reported_critical_points"])
    if max_reported < 1 or max_reported > 100000:
        raise ValueError("max_reported_critical_points must be between 1 and 100000")

    topology_name = "critic2_topology.json"
    script_lines = [
        f"{system_type.upper()} {_critic2_path(Path(structure_local.name))}",
        " ".join(load_tokens),
        "REFERENCE 1",
        " ".join(auto_tokens),
        f"CPREPORT {report_detail.upper()}",
        f"CPREPORT {topology_name}",
        "END",
    ]
    script_text = "\n".join(script_lines) + "\n"
    input_path = directory / "critic2_input.cri"
    input_path.write_text(script_text, encoding="utf-8")
    completed = run_external(
        executable="critic2",
        environment_variable="CHEMGRAPH_CRITIC2_COMMAND",
        arguments=[],
        directory=directory,
        stdin_text=script_text,
        timeout_seconds=int(request.get("resource_limits", {}).get("walltime_seconds", 1800)),
    )
    stdout_path = directory / "stdout.log"
    stderr_path = directory / "stderr.log"
    stdout_path.write_text(completed["stdout"], encoding="utf-8")
    stderr_path.write_text(completed["stderr"], encoding="utf-8")
    if not completed["available"]:
        return unavailable(
            completed["stderr"],
            install="Build Critic2 and set CHEMGRAPH_CRITIC2_COMMAND to its executable",
        )
    critic2_errors = re.findall(r"^ERROR(?:\s|\s*:)?.*$", completed["stdout"], flags=re.M)
    if (
        completed["returncode"] != 0
        or "CRITIC2 ended successfully" not in completed["stdout"]
        or critic2_errors
    ):
        raise RuntimeError(
            "Critic2 topology analysis failed: "
            + ("; ".join(critic2_errors[-10:]) or completed["stderr"] or completed["stdout"])[-4000:]
        )
    topology_path = directory / topology_name
    if not topology_path.is_file():
        raise RuntimeError("Critic2 completed without writing its topology JSON report")
    topology = json.loads(topology_path.read_text(encoding="utf-8"))
    cp_block = dict(topology.get("critical_points") or {})
    all_points = list(cp_block.get("nonequivalent_cps") or [])
    centering = [0.0, 0.0, 0.0]
    if system_type == "molecule":
        centering = [
            float(item)
            for item in (topology.get("structure") or {}).get(
                "molecule_centering_vector", [0.0, 0.0, 0.0]
            )
        ]
    bohr_to_angstrom = 0.529177210903
    points = []
    counts = {
        "nucleus": 0,
        "non_nuclear_attractor": 0,
        "bond": 0,
        "ring": 0,
        "cage": 0,
        "unclassified": 0,
    }
    for item in all_points:
        cp_type = _critic2_cp_type(item)
        counts[cp_type] = counts.get(cp_type, 0) + 1
        if len(points) >= max_reported:
            continue
        cartesian_bohr = [
            float(value) + centering[index]
            for index, value in enumerate(item.get("cartesian_coordinates") or [0.0, 0.0, 0.0])
        ]
        record = {
            "id": int(item["id"]),
            "type": cp_type,
            "rank": int(item.get("rank", 3)),
            "signature": int(item.get("signature", 0)),
            "name": item.get("name"),
            "multiplicity": int(item.get("multiplicity", 1)),
            "cartesian_coordinates_angstrom": [
                float(value * bohr_to_angstrom) for value in cartesian_bohr
            ],
            "field_value_atomic_unit": float(item.get("field", 0.0)),
            "gradient_norm_atomic_unit": float(item.get("gradient_norm", 0.0)),
            "laplacian_atomic_unit": float(item.get("laplacian", 0.0)),
            "hessian_eigenvalues_atomic_unit": [
                float(value) for value in item.get("hessian_eigenvalues", [])
            ],
        }
        if system_type == "crystal":
            record["fractional_coordinates"] = [
                float(value) for value in item.get("fractional_coordinates", [])
            ]
        points.append(record)

    attractors = counts["nucleus"] + counts["non_nuclear_attractor"]
    topological_sum = attractors - counts["bond"] + counts["ring"] - counts["cage"]
    expected_sum = 1 if system_type == "molecule" else 0
    result = {
        "system_type": system_type,
        "density_format": density_format,
        "interpolation": interpolation,
        "critical_point_types_requested": cp_types,
        "nonequivalent_critical_point_count": int(
            cp_block.get("number_of_nonequivalent_cps", len(all_points))
        ),
        "unit_cell_critical_point_count": int(
            cp_block.get("number_of_cell_cps", len(cp_block.get("cell_cps") or all_points))
        ),
        "counts_by_type": counts,
        "topological_sum": topological_sum,
        "expected_topological_sum": expected_sum,
        "matches_expected_topological_sum": topological_sum == expected_sum,
        "critical_points": points,
        "reported_critical_point_count": len(points),
        "critical_points_truncated": len(all_points) > len(points),
        "source_density_path": relative_workspace_path(density_path),
        "source_structure_path": relative_workspace_path(structure_path),
        "search_configuration": {
            "gradient_tolerance": gradient_tolerance,
            "seed_strategy": settings["seed_strategy"],
            "search_region": region,
            "discard_density_below": settings.get("discard_density_below"),
            "checkpoint_critical_points": bool(settings.get("checkpoint_critical_points", False)),
        },
    }
    result_path = write_json(directory, "result.json", result)
    version_match = re.search(r"version\s+([0-9]+(?:\.[0-9]+)+)", completed["stdout"], re.I)
    return success(
        result,
        artifact_files=command_artifacts(directory),
        backend_version=version_match.group(1) if version_match else None,
        warnings=(
            []
            if topological_sum == expected_sum
            else [
                "The critical-point alternating sum does not match the ideal complete-topology value; "
                "the Agent should review grid quality, interpolation, density cutoff, and seeding."
            ]
        ),
        provenance={
            "command": completed["command"],
            "critic2_input": relative_workspace_path(input_path),
            "critic2_json": relative_workspace_path(topology_path),
            "normalized_result": relative_workspace_path(result_path),
        },
    )


def _critic2_basins(action_id: str, request: dict[str, Any]) -> dict[str, Any]:
    inputs, _method, settings = request_parts(request)
    (
        directory,
        density_path,
        structure_path,
        density_local,
        structure_local,
    ) = _critic2_stage_inputs(action_id, inputs)
    system_type = str(settings["system_type"]).strip().lower()
    if system_type not in {"molecule", "crystal"}:
        raise ValueError("system_type must be molecule or crystal")

    density_format = str(settings["density_format"]).strip().lower()
    format_keywords = {
        "auto": "", "cube": "CUBE", "bincube": "BINCUBE", "abinit": "ABINIT",
        "vasp": "VASP", "vaspnov": "VASPNOV", "qub": "QUB", "xsf": "XSF",
        "fmt": "FMT", "txt": "TXT", "dat": "DAT", "elkgrid": "ELKGRID",
        "siesta": "SIESTA", "fplo": "FPLO", "dftb": "DFTB", "wfn": "WFN",
        "wfx": "WFX", "molden": "MOLDEN", "molden_orca": "MOLDEN_ORCA",
        "molden_psi4": "MOLDEN_PSI4", "fchk": "FCHK", "pwc": "PWC",
    }
    if density_format not in format_keywords:
        raise ValueError(f"Unsupported Critic2 density_format: {density_format}")
    interpolation = str(settings["interpolation"]).strip().lower()
    if interpolation not in {
        "native", "nearest", "trilinear", "trispline", "tricubic", "smoothrho",
    }:
        raise ValueError(
            "interpolation must be native, nearest, trilinear, trispline, tricubic, or smoothrho"
        )
    load_tokens = ["LOAD"]
    if format_keywords[density_format]:
        load_tokens.append(format_keywords[density_format])
    load_tokens.append(_critic2_path(Path(density_local.name)))
    if interpolation != "native":
        load_tokens.append(interpolation.upper())
    if interpolation == "smoothrho":
        if "smoothrho_environment_nodes" in settings:
            nodes = int(settings["smoothrho_environment_nodes"])
            if nodes < 1:
                raise ValueError("smoothrho_environment_nodes must be positive")
            load_tokens.extend(["NENV", str(nodes)])
        if "smoothrho_distance_factor" in settings:
            factor = float(settings["smoothrho_distance_factor"])
            if not math.isfinite(factor) or factor <= 0:
                raise ValueError("smoothrho_distance_factor must be positive and finite")
            load_tokens.extend(["FDMAX", f"{factor:.16g}"])

    partition = str(settings["partition_method"]).strip().lower()
    partition_keywords = {
        "yu_trinkle": "YT", "yt": "YT",
        "henkelman_bader": "BADER", "bader": "BADER",
        "hirshfeld": "HIRSHFELD", "voronoi": "VORONOI",
    }
    if partition not in partition_keywords:
        raise ValueError(
            "partition_method must be yu_trinkle, henkelman_bader, hirshfeld, or voronoi"
        )
    partition_keyword = partition_keywords[partition]
    if action_id == "calculate_bader_charges" and partition_keyword not in {"YT", "BADER"}:
        raise ValueError(
            "calculate_bader_charges requires partition_method=yu_trinkle or henkelman_bader"
        )

    for boolean_name in (
        "non_nuclear_maxima", "all_maxima_non_atomic", "write_weight_cubes",
    ):
        if not isinstance(settings[boolean_name], bool):
            raise ValueError(f"{boolean_name} must be an explicit boolean")
    non_nuclear = settings["non_nuclear_maxima"]
    all_non_atomic = settings["all_maxima_non_atomic"]
    write_weights = settings["write_weight_cubes"]
    if non_nuclear and all_non_atomic:
        raise ValueError("non_nuclear_maxima and all_maxima_non_atomic cannot both be true")
    if partition_keyword not in {"YT", "BADER"} and (non_nuclear or all_non_atomic):
        raise ValueError("NNM/NOATOMS settings apply only to YT and BADER partitions")
    if partition_keyword == "VORONOI" and write_weights:
        raise ValueError("Critic2 VORONOI does not support write_weight_cubes")

    integration_tokens = [partition_keyword]
    if non_nuclear:
        integration_tokens.append("NNM")
    if all_non_atomic:
        integration_tokens.append("NOATOMS")
    if write_weights:
        integration_tokens.append("WCUBE")
    if "attractor_assignment_radius" in settings:
        if partition_keyword not in {"YT", "BADER"}:
            raise ValueError("attractor_assignment_radius applies only to YT and BADER")
        radius = float(settings["attractor_assignment_radius"])
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("attractor_assignment_radius must be positive and finite")
        integration_tokens.extend(["RATOM", f"{radius:.16g}"])
    if "discard_density_below" in settings:
        if partition_keyword not in {"YT", "BADER"}:
            raise ValueError("discard_density_below applies only to YT and BADER")
        threshold = float(settings["discard_density_below"])
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError("discard_density_below must be finite and nonnegative")
        integration_tokens.extend(["DISCARD", f'"$1 < {threshold:.16g}"'])
    selected = settings.get("selected_attractors")
    if selected is not None:
        if not isinstance(selected, (list, tuple)) or not selected:
            raise ValueError("selected_attractors must be a non-empty positive-integer list")
        identifiers = [int(value) for value in selected]
        if any(value < 1 for value in identifiers):
            raise ValueError("selected_attractors identifiers must be positive")
        integration_tokens.extend(["ONLY", *[str(value) for value in identifiers]])
    selected_range = settings.get("selected_attractor_range")
    if selected_range is not None:
        if not isinstance(selected_range, (list, tuple)) or len(selected_range) != 2:
            raise ValueError("selected_attractor_range must contain two positive identifiers")
        start, stop = [int(value) for value in selected_range]
        if start < 1 or stop < start:
            raise ValueError("selected_attractor_range must satisfy 1 <= start <= stop")
        integration_tokens.extend(["ONLY_RANGE", str(start), str(stop)])

    max_reported = int(settings["max_reported_basins"])
    if max_reported < 1 or max_reported > 100000:
        raise ValueError("max_reported_basins must be between 1 and 100000")
    laplacian_tolerance = float(settings["laplacian_sum_tolerance"])
    if not math.isfinite(laplacian_tolerance) or laplacian_tolerance <= 0:
        raise ValueError("laplacian_sum_tolerance must be positive and finite")

    integration_name = "critic2_basin_integration.json"
    integration_tokens.extend(["JSON", integration_name])
    script_lines = [
        f"{system_type.upper()} {_critic2_path(Path(structure_local.name))}",
        " ".join(load_tokens),
        "REFERENCE 1",
        " ".join(integration_tokens),
        "END",
    ]
    script_text = "\n".join(script_lines) + "\n"
    input_path = directory / "critic2_input.cri"
    input_path.write_text(script_text, encoding="utf-8")
    completed = run_external(
        executable="critic2",
        environment_variable="CHEMGRAPH_CRITIC2_COMMAND",
        arguments=[],
        directory=directory,
        stdin_text=script_text,
        timeout_seconds=int(request.get("resource_limits", {}).get("walltime_seconds", 1800)),
    )
    (directory / "stdout.log").write_text(completed["stdout"], encoding="utf-8")
    (directory / "stderr.log").write_text(completed["stderr"], encoding="utf-8")
    if not completed["available"]:
        return unavailable(
            completed["stderr"],
            install="Build Critic2 and set CHEMGRAPH_CRITIC2_COMMAND to its executable",
        )
    critic2_errors = re.findall(r"^ERROR(?:\s|\s*:)?.*$", completed["stdout"], flags=re.M)
    if (
        completed["returncode"] != 0
        or "CRITIC2 ended successfully" not in completed["stdout"]
        or critic2_errors
    ):
        raise RuntimeError(
            "Critic2 basin integration failed: "
            + ("; ".join(critic2_errors[-10:]) or completed["stderr"] or completed["stdout"])[-4000:]
        )
    integration_path = directory / integration_name
    if not integration_path.is_file():
        raise RuntimeError("Critic2 completed without writing its basin-integration JSON report")
    payload = json.loads(integration_path.read_text(encoding="utf-8"))
    integration = dict(payload.get("integration") or {})
    properties = list(integration.get("properties") or [])
    property_labels = [str(item.get("label") or f"property_{index + 1}") for index, item in enumerate(properties)]
    all_attractors = list(integration.get("attractors") or [])
    structure = dict(payload.get("structure") or {})
    matrix = list(structure.get("crys_to_cart_matrix") or [])
    if matrix and isinstance(matrix[0], list):
        matrix = [value for row in matrix for value in row]
    centering = [0.0, 0.0, 0.0]
    if system_type == "molecule":
        centering = [
            float(value)
            for value in structure.get("molecule_centering_vector", [0.0, 0.0, 0.0])
        ]
    bohr_to_angstrom = 0.529177210903
    basin_records = []
    charge_records = []
    population_sum = 0.0
    laplacian_sum = 0.0
    charge_sum = 0.0
    charge_count = 0
    for attractor in all_attractors:
        integrals = [float(value) for value in attractor.get("integrals") or []]
        values = {
            label: integrals[index]
            for index, label in enumerate(property_labels)
            if index < len(integrals)
        }
        normalized_values = {
            re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_"): value
            for label, value in values.items()
        }
        population = next(
            (
                value for key, value in normalized_values.items()
                if key in {"pop", "population", "charge"} or "population" in key
            ),
            None,
        )
        laplacian = next(
            (value for key, value in normalized_values.items() if key == "lap" or "laplac" in key),
            None,
        )
        if population is not None:
            population_sum += population
        if laplacian is not None:
            laplacian_sum += laplacian
        fractional = [float(value) for value in attractor.get("fractional_coordinates") or []]
        coordinates_angstrom = []
        if len(fractional) == 3 and len(matrix) == 9:
            cartesian_bohr = [
                sum(float(matrix[3 * row + column]) * fractional[column] for column in range(3))
                + centering[row]
                for row in range(3)
            ]
            coordinates_angstrom = [value * bohr_to_angstrom for value in cartesian_bohr]
        atomic_number_value = attractor.get("atomic_number")
        try:
            atomic_number = int(atomic_number_value)
        except (TypeError, ValueError):
            atomic_number = None
        record = {
            "attractor_id": int(attractor["id"]),
            "name": attractor.get("name"),
            "atomic_number": atomic_number,
            "critical_point_id": attractor.get("cell_cp"),
            "nonequivalent_critical_point_id": attractor.get("nonequivalent_cp"),
            "fractional_coordinates": fractional,
            "cartesian_coordinates_angstrom": coordinates_angstrom,
            "integrated_properties": values,
        }
        if len(basin_records) < max_reported:
            basin_records.append(record)
        if atomic_number is not None and population is not None:
            charge = float(atomic_number - population)
            charge_sum += charge
            charge_count += 1
            if len(charge_records) < max_reported:
                charge_records.append(
                    {
                        "attractor_id": int(attractor["id"]),
                        "name": attractor.get("name"),
                        "atomic_number": atomic_number,
                        "electron_population": population,
                        "charge": charge,
                        "unit": "elementary_charge",
                        "cartesian_coordinates_angstrom": coordinates_angstrom,
                    }
                )

    warnings = []
    if abs(laplacian_sum) > laplacian_tolerance:
        warnings.append(
            "The summed basin Laplacian exceeds laplacian_sum_tolerance; review grid resolution, "
            "interpolation, and partition settings."
        )
    expected_charge = settings.get("expected_total_charge")
    if expected_charge is not None:
        expected_charge = float(expected_charge)
        charge_tolerance = float(settings.get("total_charge_tolerance", 0.05))
        if not math.isfinite(charge_tolerance) or charge_tolerance <= 0:
            raise ValueError("total_charge_tolerance must be positive and finite")
        if abs(charge_sum - expected_charge) > charge_tolerance:
            warnings.append(
                "The summed atomic basin charges do not match expected_total_charge within "
                "total_charge_tolerance."
            )

    common = {
        "system_type": system_type,
        "density_format": density_format,
        "interpolation": interpolation,
        "partition_method": partition,
        "critic2_partition_keyword": partition_keyword,
        "basin_count": int(integration.get("number_of_attractors", len(all_attractors))),
        "property_definitions": properties,
        "electron_population_sum": population_sum,
        "integrated_laplacian_sum": laplacian_sum,
        "laplacian_sum_tolerance": laplacian_tolerance,
        "source_density_path": relative_workspace_path(density_path),
        "source_structure_path": relative_workspace_path(structure_path),
        "reported_basin_count": min(len(all_attractors), max_reported),
        "basins_truncated": len(all_attractors) > max_reported,
    }
    if action_id == "calculate_bader_charges":
        if charge_count == 0:
            raise RuntimeError("Critic2 basin report contained no atomic populations for charge derivation")
        result = {
            **common,
            "charges": charge_records,
            "charge_count": charge_count,
            "total_charge": charge_sum,
            "expected_total_charge": expected_charge,
            "unit": "elementary_charge",
        }
    else:
        result = {**common, "basins": basin_records}
    result_path = write_json(directory, "result.json", result)
    version_match = re.search(r"version\s+([0-9]+(?:\.[0-9]+)+)", completed["stdout"], re.I)
    return success(
        result,
        artifact_files=command_artifacts(directory),
        backend_version=version_match.group(1) if version_match else None,
        warnings=warnings,
        provenance={
            "command": completed["command"],
            "critic2_input": relative_workspace_path(input_path),
            "critic2_json": relative_workspace_path(integration_path),
            "normalized_result": relative_workspace_path(result_path),
        },
    )


def _psi4_geometry(structure_value: Any) -> str:
    structure = structure_dict(structure_value)
    symbols = []
    coordinates = []
    if "atoms" in structure:
        for atom in structure["atoms"]:
            symbols.append(atom["element"])
            coordinates.append(atom["position_angstrom"])
    else:
        symbols = structure["symbols"]
        coordinates = structure.get("coordinates_angstrom") or structure["positions"]
    charge = int(structure.get("charge", 0))
    multiplicity = int(structure.get("multiplicity", 1))
    lines = [f"{charge} {multiplicity}"]
    lines.extend(f"{s} {r[0]} {r[1]} {r[2]}" for s, r in zip(symbols, coordinates))
    lines.append("units angstrom")
    lines.append("no_reorient")
    lines.append("no_com")
    return "\n".join(lines)


def _psi4_irrep_blocks(vector: Any) -> list[list[float]]:
    """Convert a Psi4 Vector into explicit per-irrep numeric blocks."""

    import numpy as np

    value = vector.to_array() if hasattr(vector, "to_array") else vector
    raw_blocks = value if isinstance(value, (tuple, list)) else (value,)
    return [
        np.asarray(block, dtype=float).reshape(-1).tolist()
        for block in raw_blocks
    ]


def _psi4_irrep_occupations(
    dimension: Any,
    block_sizes: list[int],
    occupied_value: float,
) -> list[float]:
    """Build occupations in the same flattened irrep order as orbital energies."""

    counts_value = dimension.to_tuple() if hasattr(dimension, "to_tuple") else dimension
    counts = [int(value) for value in counts_value]
    if len(counts) != len(block_sizes):
        raise RuntimeError("Psi4 occupation and orbital irrep dimensions do not match")
    occupations: list[float] = []
    for occupied, size in zip(counts, block_sizes):
        if occupied < 0 or occupied > size:
            raise RuntimeError("Psi4 occupied-orbital count is outside its irrep dimension")
        occupations.extend([occupied_value] * occupied)
        occupations.extend([0.0] * (size - occupied))
    return occupations


def _psi4_restricted_occupations(
    alpha_dimension: Any,
    beta_dimension: Any,
    block_sizes: list[int],
) -> list[float]:
    alpha_value = alpha_dimension.to_tuple() if hasattr(alpha_dimension, "to_tuple") else alpha_dimension
    beta_value = beta_dimension.to_tuple() if hasattr(beta_dimension, "to_tuple") else beta_dimension
    alpha_counts = [int(value) for value in alpha_value]
    beta_counts = [int(value) for value in beta_value]
    if len(alpha_counts) != len(block_sizes) or len(beta_counts) != len(block_sizes):
        raise RuntimeError("Psi4 restricted occupations do not match orbital irrep dimensions")
    occupations: list[float] = []
    for alpha, beta, size in zip(alpha_counts, beta_counts, block_sizes):
        if beta < 0 or alpha < beta or alpha > size:
            raise RuntimeError("Psi4 restricted occupied-orbital counts are inconsistent")
        occupations.extend([2.0] * beta)
        occupations.extend([1.0] * (alpha - beta))
        occupations.extend([0.0] * (size - alpha))
    return occupations


def _psi4(action_id: str, request: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import psi4

    inputs, method, settings = request_parts(request)
    directory = output_directory(action_id, "psi4")
    output_path = directory / "psi4.out"
    psi4.core.clean()
    psi4.set_memory(f"{int(request.get('resource_limits', {}).get('memory_mb') or 1000)} MB")
    psi4.set_num_threads(int(request.get("resource_limits", {}).get("cpu_cores") or 1))
    psi4.core.set_output_file(str(output_path), False)
    molecule = psi4.geometry(_psi4_geometry(inputs["structure"]))
    model = f"{method['method']}/{method['basis']}"
    psi4.set_options(dict(method.get("options") or {}))
    if action_id == "calculate_hessian":
        matrix = np.asarray(psi4.hessian(model, molecule=molecule), dtype=float)
        result = {"matrix": matrix.tolist(), "unit": "hartree/bohr^2", "model": model}
    else:
        energy, wavefunction = psi4.energy(model, molecule=molecule, return_wfn=True)
        common = {"energy_hartree": float(energy), "model": model}
        if action_id == "calculate_energy":
            result = {**common, "energy": float(energy), "unit": "hartree"}
        elif action_id == "calculate_dipole_moment":
            psi4.oeprop(wavefunction, "DIPOLE")
            dipole_au = np.asarray(psi4.core.variable("SCF DIPOLE"), dtype=float).reshape(-1)
            if dipole_au.size != 3:
                raise RuntimeError("Psi4 SCF DIPOLE did not contain three Cartesian components")
            dipole = (dipole_au * 2.541746473).tolist()
            result = {**common, "dipole": dipole, "unit": "debye"}
        elif action_id == "calculate_atomic_charges":
            psi4.oeprop(wavefunction, "MULLIKEN_CHARGES")
            charges = np.asarray(wavefunction.atomic_point_charges(), dtype=float).tolist()
            result = {**common, "charges": charges, "analysis": "mulliken", "unit": "elementary_charge"}
        elif action_id == "calculate_orbitals":
            alpha_blocks = _psi4_irrep_blocks(wavefunction.epsilon_a())
            alpha_sizes = [len(block) for block in alpha_blocks]
            restricted = bool(wavefunction.same_a_b_orbs())
            result = {
                **common,
                "alpha_energies_hartree": [value for block in alpha_blocks for value in block],
                "alpha_occupations": (
                    _psi4_restricted_occupations(
                        wavefunction.nalphapi(), wavefunction.nbetapi(), alpha_sizes
                    )
                    if restricted
                    else _psi4_irrep_occupations(wavefunction.nalphapi(), alpha_sizes, 1.0)
                ),
                "alpha_irrep_dimensions": alpha_sizes,
            }
            if not restricted:
                beta_blocks = _psi4_irrep_blocks(wavefunction.epsilon_b())
                beta_sizes = [len(block) for block in beta_blocks]
                result["beta_energies_hartree"] = [value for block in beta_blocks for value in block]
                result["beta_occupations"] = _psi4_irrep_occupations(
                    wavefunction.nbetapi(), beta_sizes, 1.0
                )
                result["beta_irrep_dimensions"] = beta_sizes
        else:
            return unsupported(f"Psi4 does not implement {action_id}")
    result_path = write_json(directory, "result.json", result)
    return success(
        result,
        artifact_files=[
            {"path": relative_workspace_path(output_path), "semantic_type": "BackendOutput", "media_type": "text/plain"},
            {"path": relative_workspace_path(result_path), "semantic_type": "ElectronicResult", "media_type": "application/json"},
        ],
        backend_version=module_version("psi4"),
    )


def _render_orca(
    action_id: str,
    structure_value: Any,
    method: dict[str, Any],
    settings: dict[str, Any],
    resource_limits: dict[str, Any] | None = None,
) -> str:
    structure = structure_dict(structure_value)
    symbols, coordinates = atoms_and_coordinates(structure)
    keyword = {
        "calculate_energy": "SP",
        "calculate_forces": "EnGrad",
        "calculate_hessian": "Freq",
        "optimize_geometry": "Opt",
        "calculate_dipole_moment": "SP",
        "calculate_atomic_charges": "SP",
        "calculate_orbitals": "SP",
        "calculate_bond_orders": "SP",
        "calculate_excited_states": "SP",
    }[action_id]
    method_name, dispersion = _orca_method_and_dispersion_tokens(method)
    method_name, basis = normalize_orca_method_basis(
        method_name, method.get("basis")
    )
    header = "! " + " ".join(
        token for token in (method_name, basis, keyword) if token is not None
    )
    if dispersion:
        header += f" {dispersion}"
    solvation_lines: list[str] = []
    if method.get("solvation_model") is not None:
        model = str(method["solvation_model"]).strip().casefold()
        if model not in {"cpcm", "smd"}:
            raise ValueError("ORCA solvation_model must be cpcm or smd")
        if not method.get("solvent"):
            raise ValueError("ORCA solvation_model requires method_spec.solvent")
        solvent = str(method["solvent"]).strip()
        if not re.fullmatch(r"[A-Za-z0-9_.+-]+", solvent):
            raise ValueError("ORCA solvent contains unsupported characters")
        if model == "cpcm":
            header += f" CPCM({solvent})"
        else:
            header += " CPCM"
            solvation_lines = [
                "%cpcm",
                "  smd true",
                f'  SMDsolvent "{solvent}"',
                "end",
            ]
    charge = int(method.get("charge", structure.get("charge", 0)))
    multiplicity = int(method.get("multiplicity", structure.get("multiplicity", 1)))
    lines = [header, *solvation_lines]
    parallel_processes = int((resource_limits or {}).get("cpu_cores") or 1)
    if parallel_processes > 1:
        lines.extend(["%pal", f"  nprocs {parallel_processes}", "end"])
    if action_id == "optimize_geometry":
        convergence = str(settings["optimization_convergence"]).strip()
        convergence_choices = {
            "loose": "Loose",
            "normal": "Normal",
            "tight": "Tight",
            "verytight": "VeryTight",
        }
        try:
            convergence = convergence_choices[convergence.casefold()]
        except KeyError as exc:
            raise ValueError(
                "ORCA optimization_convergence must be Loose, Normal, Tight, or VeryTight"
            ) from exc
        lines.extend(
            [
                "%geom",
                f"  Convergence {convergence}",
                f"  MaxIter {int(settings['max_steps'])}",
                "end",
            ]
        )
    if action_id == "calculate_orbitals":
        lines.extend(["%output", "  Print[P_OrbEn] 2", "end"])
    if action_id == "calculate_bond_orders":
        minimum = float(settings["minimum_bond_order"])
        if minimum < 0:
            raise ValueError("minimum_bond_order must be nonnegative")
        lines.extend(["%method", f"  Mayer_BondOrderThresh {minimum}", "end"])
    if action_id == "calculate_excited_states":
        excited_method = str(method["excited_state_method"]).strip().lower()
        if excited_method not in {"tda", "tddft"}:
            raise ValueError("ORCA excited_state_method must be tda or tddft")
        spin_symmetry = str(settings["spin_symmetry"]).strip().lower()
        if spin_symmetry != "singlet":
            raise ValueError("The validated ORCA excited-state adapter currently exposes singlet roots only")
        lines.extend(
            [
                "%tddft",
                f"  nroots {int(settings['number_of_states'])}",
                f"  tda {'true' if excited_method == 'tda' else 'false'}",
                "  triplets false",
                f"  etol {float(settings['excited_energy_tolerance_hartree'])}",
                f"  rtol {float(settings['residual_tolerance'])}",
                "end",
            ]
        )
    lines.append(f"* xyz {charge} {multiplicity}")
    lines.extend(f"{s} {r[0]} {r[1]} {r[2]}" for s, r in zip(symbols, coordinates))
    lines.append("*")
    return "\n".join(lines) + "\n"


def _parse_orca_hessian(path: Path) -> list[list[float]] | None:
    """Parse ORCA's block-column $hessian section into a dense matrix."""

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    try:
        section = next(index for index, line in enumerate(lines) if line.strip().lower() == "$hessian")
        dimension = int(lines[section + 1].strip())
    except (StopIteration, ValueError, IndexError):
        return None
    matrix = [[0.0 for _ in range(dimension)] for _ in range(dimension)]
    populated: set[tuple[int, int]] = set()
    columns: list[int] = []
    for line in lines[section + 2 :]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("$"):
            break
        parts = stripped.split()
        if parts and all(part.isdigit() for part in parts):
            columns = [int(part) for part in parts]
            continue
        if not columns or not parts[0].isdigit():
            continue
        row = int(parts[0])
        if row >= dimension:
            continue
        values = parts[1:]
        for column, value in zip(columns, values):
            if column >= dimension:
                continue
            try:
                matrix[row][column] = float(value.replace("D", "E").replace("d", "e"))
            except ValueError:
                return None
            populated.add((row, column))
    return matrix if len(populated) == dimension * dimension else None


def _orca_density_token(value: Any, *, field_name: str) -> str:
    token = str(value).strip()
    if not token or not re.fullmatch(r"[A-Za-z0-9_+().,/=-]+", token):
        raise ValueError(f"ORCA {field_name} contains unsupported characters")
    return token


def _orca_method_and_dispersion_tokens(method: dict[str, Any]) -> tuple[str, str | None]:
    """Normalize literature-style method labels to valid ORCA tokens.

    Scientific papers often join a functional and dispersion correction into
    one hyphenated label, while ORCA's simple input requires two tokens.  This
    compatibility layer accepts those conventional labels without overriding
    a conflicting correction explicitly selected by the agent.
    """

    method_name = _orca_density_token(method["method"], field_name="method")
    explicit_dispersion = (
        _orca_density_token(method["dispersion"], field_name="dispersion")
        if method.get("dispersion")
        else None
    )
    aliases = {
        "dsd-pbep86-d3bj": ("DSD-PBEP86", "D3BJ"),
    }
    normalized = aliases.get(method_name.casefold())
    if normalized is None:
        return method_name, explicit_dispersion
    normalized_method, implied_dispersion = normalized
    if (
        explicit_dispersion is not None
        and explicit_dispersion.casefold() != implied_dispersion.casefold()
    ):
        raise ValueError(
            f"ORCA method alias {method_name} implies dispersion={implied_dispersion}, "
            f"but dispersion={explicit_dispersion} was requested"
        )
    return normalized_method, explicit_dispersion or implied_dispersion


def _orca_density_resource_allocation(
    resource_limits: dict[str, Any] | None,
) -> dict[str, Any]:
    limits = resource_limits or {}
    cores = max(1, int(limits.get("cpu_cores") or 1))
    explicit_memory = limits.get("memory_mb") is not None
    total_memory_mb = int(
        limits.get("memory_mb")
        if explicit_memory
        else ORCA_DENSITY_DEFAULT_MAXCORE_MB * cores
    )
    return {
        "cpu_cores": cores,
        "requested_total_memory_mb": total_memory_mb,
        "orca_maxcore_mb_per_process": max(128, total_memory_mb // cores),
        "memory_default_applied": not explicit_memory,
    }


def _render_orca_density(
    structure_value: Any,
    method: dict[str, Any],
    settings: dict[str, Any],
    resource_limits: dict[str, Any] | None = None,
) -> str:
    structure = structure_dict(structure_value)
    symbols, coordinates = atoms_and_coordinates(structure)
    method_name, dispersion = _orca_method_and_dispersion_tokens(method)
    method_name, basis = normalize_orca_method_basis(
        method_name, method.get("basis")
    )
    if basis is not None:
        basis = _orca_density_token(basis, field_name="basis")
    density_type = str(method["density_type"]).strip().casefold()
    normalized_method = method_name.casefold().replace("-", "")
    if density_type == "unrelaxed_ccsd" and normalized_method != "ccsd":
        if "ccsd(t)" in method_name.casefold():
            raise ValueError(
                "ORCA does not provide an unrelaxed CCSD(T) one-particle density; "
                "request method=CCSD and label any separate CCSD(T) energy explicitly"
            )
        raise ValueError("unrelaxed_ccsd density_type requires method=CCSD")
    if density_type == "relaxed_mp2" and not (
        "mp2" in method_name.casefold() or "dsd" in method_name.casefold()
    ):
        raise ValueError(
            "relaxed_mp2 density_type requires an MP2 or double-hybrid DSD method"
        )
    if density_type == "scf" and normalized_method == "ccsd":
        raise ValueError("CCSD density must use density_type=unrelaxed_ccsd")

    convergence_map = {
        "loosescf": "LooseSCF",
        "tightscf": "TightSCF",
        "verytightscf": "VeryTightSCF",
    }
    try:
        convergence = convergence_map[str(settings["scf_convergence"]).casefold()]
    except KeyError as exc:
        raise ValueError(
            "scf_convergence must be LooseSCF, TightSCF, or VeryTightSCF"
        ) from exc
    max_cycles = int(settings["max_scf_cycles"])
    if max_cycles < 1 or max_cycles > 5000:
        raise ValueError("max_scf_cycles must be between 1 and 5000")
    stability = settings["stability_analysis"]
    if not isinstance(stability, bool):
        raise ValueError("stability_analysis must be a boolean")

    header = [method_name, *([basis] if basis is not None else [])]
    if method.get("auxiliary_basis"):
        header.append(
            _orca_density_token(method["auxiliary_basis"], field_name="auxiliary_basis")
        )
    if dispersion:
        header.append(dispersion)
    if method.get("frozen_core") is False:
        header.append("NoFrozenCore")
    elif method.get("frozen_core") is True:
        header.append("FrozenCore")
    if method.get("pmodel") is True:
        header.append("PModel")
    header.extend([convergence, "SP"])

    allocation = _orca_density_resource_allocation(resource_limits)
    cores = int(allocation["cpu_cores"])
    maxcore_mb = int(allocation["orca_maxcore_mb_per_process"])
    lines = ["! " + " ".join(header), f"%maxcore {maxcore_mb}"]
    if cores > 1:
        lines.extend(["%pal", f"  nprocs {cores}", "end"])
    lines.extend(["%scf", f"  MaxIter {max_cycles}"])
    if stability:
        lines.extend(
            [
                "  GuessMode CMatrix",
                "  STABPerform true",
                "  STABRestartUHFifUnstable true",
            ]
        )
    lines.append("end")
    if density_type == "relaxed_mp2":
        lines.extend(["%mp2", "  Density relaxed", "  NatOrbs true", "end"])
    elif density_type == "unrelaxed_ccsd":
        lines.extend(["%mdci", "  Density unrelaxed", "end"])

    charge = int(method.get("charge", structure.get("charge", 0)))
    multiplicity = int(
        method.get("multiplicity", structure.get("multiplicity", 1))
    )
    lines.append(f"* xyz {charge} {multiplicity}")
    lines.extend(
        f"{symbol} {row[0]:.12f} {row[1]:.12f} {row[2]:.12f}"
        for symbol, row in zip(symbols, coordinates)
    )
    lines.append("*")
    return "\n".join(lines) + "\n"


def _orca_density_artifacts(directory: Path) -> list[dict[str, str]]:
    semantic_by_suffix = {
        ".inp": ("BackendInput", "text/plain"),
        ".out": ("BackendOutput", "text/plain"),
        ".err": ("BackendDiagnostic", "text/plain"),
        ".gbw": ("ORCAWavefunction", "application/octet-stream"),
        ".mp2nat": ("ElectronDensityNaturalOrbitals", "application/octet-stream"),
        ".densities": ("ElectronDensityContainer", "application/octet-stream"),
        ".wfn": ("ElectronDensityWavefunction", "chemical/x-wfn"),
        ".wfx": ("ElectronDensityWavefunction", "chemical/x-wfx"),
        ".cube": ("ElectronDensityGrid", "chemical/x-gaussian-cube"),
        ".json": ("ElectronicResult", "application/json"),
    }
    artifacts: list[dict[str, str]] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        semantic, media = semantic_by_suffix.get(
            path.suffix.casefold(), ("BackendFile", "application/octet-stream")
        )
        artifacts.append(
            {
                "path": relative_workspace_path(path),
                "semantic_type": semantic,
                "media_type": media,
            }
        )
    return artifacts


def _orca_correlated_electron_density(request: dict[str, Any]) -> dict[str, Any]:
    inputs, method, settings = request_parts(request)
    normalized_method, normalized_basis = normalize_orca_method_basis(
        method["method"], method.get("basis")
    )
    allocation = _orca_density_resource_allocation(
        dict(request.get("resource_limits") or {})
    )
    directory = output_directory("calculate_correlated_electron_density", "orca")
    input_path = directory / "job.inp"
    input_path.write_text(
        _render_orca_density(
            inputs["structure"],
            method,
            settings,
            dict(request.get("resource_limits") or {}),
        ),
        encoding="utf-8",
    )
    completed = run_external(
        executable="orca",
        environment_variable="CHEMGRAPH_ORCA_COMMAND",
        # ORCA propagates the supplied input path to module-specific scratch
        # basenames.  The MDCI/CCSD modules in ORCA 6.1.1 can crash when that
        # basename is a full, deeply nested output path.  A
        # bare filename is not accepted by ORCA, so use an explicit short
        # relative path while keeping the calculation cwd at ``directory``.
        arguments=[f"./{input_path.name}"],
        directory=directory,
        timeout_seconds=int(
            request.get("resource_limits", {}).get("walltime_seconds", 1800)
        ),
    )
    output_path = directory / "job.out"
    error_path = directory / "job.err"
    output_path.write_text(completed["stdout"], encoding="utf-8")
    error_path.write_text(completed["stderr"], encoding="utf-8")
    if not completed["available"]:
        return unavailable(
            completed["stderr"],
            install="Configure the licensed ORCA 6.1.1 bundle and OpenMPI 4.1.8 runtime",
        )
    if completed["returncode"] != 0:
        detail = (completed["stderr"] or completed["stdout"])[-4000:]
        raise RuntimeError(f"ORCA electron-density calculation failed: {detail}")
    completion_error = _orca_completion_error(completed["stdout"], completed["stderr"])
    if completion_error is not None:
        raise RuntimeError(
            f"ORCA electron-density calculation did not terminate normally: {completion_error}"
        )

    density_type = str(method["density_type"]).casefold()
    files = {
        "input": relative_workspace_path(input_path),
        "output": relative_workspace_path(output_path),
    }
    candidates = {
        "gbw": directory / "job.gbw",
        "density_container": directory / "job.densities",
        "density_info": directory / "job.densitiesinfo",
        "natural_orbitals": directory / "job.mp2nat",
    }
    for key, path in candidates.items():
        if path.is_file():
            files[key] = relative_workspace_path(path)
    required_key = {
        "scf": "gbw",
        "relaxed_mp2": "natural_orbitals",
        "unrelaxed_ccsd": "density_container",
    }[density_type]
    if required_key not in files:
        raise RuntimeError(
            f"ORCA completed but did not produce the required {required_key} density artifact"
        )
    if density_type == "unrelaxed_ccsd" and "density_info" not in files:
        raise RuntimeError(
            "ORCA completed but did not produce the density metadata required by orca_plot"
        )
    if density_type == "unrelaxed_ccsd" and not re.search(
        r"(?:unrelaxed density|mdcip)", completed["stdout"], re.I
    ):
        raise RuntimeError("ORCA output did not confirm an unrelaxed CCSD/MDCI density")
    if density_type == "relaxed_mp2" and "mp2nat" not in completed["stdout"].casefold():
        # The physical file is authoritative, but make the version-dependent omission visible.
        warning = "ORCA produced .mp2nat but did not mention that filename in the main output"
    else:
        warning = None

    energy_matches = re.findall(
        r"FINAL SINGLE POINT ENERGY\s+(-?\d+(?:\.\d+)?)", completed["stdout"]
    )
    version_match = re.search(r"Program Version\s+(\d+\.\d+\.\d+)", completed["stdout"])
    charge = int(
        method.get("charge", structure_dict(inputs["structure"]).get("charge", 0))
    )
    expected_electron_count = int(
        sum(ase_atoms(inputs["structure"]).get_atomic_numbers()) - charge
    )
    result = {
        "method": normalized_method,
        "basis": normalized_basis or "method_default",
        "density_type": density_type,
        "energy_hartree": float(energy_matches[-1]) if energy_matches else None,
        "charge": charge,
        "multiplicity": int(
            method.get(
                "multiplicity", structure_dict(inputs["structure"]).get("multiplicity", 1)
            )
        ),
        "electron_count": expected_electron_count,
        "files": files,
    }
    result_path = write_json(directory, "result.json", result)
    artifacts = _orca_density_artifacts(directory)
    result_relative = relative_workspace_path(result_path)
    for item in artifacts:
        if item["path"] == result_relative:
            item["semantic_type"] = "ElectronDensityResult"
            item["media_type"] = "application/json"
    return success(
        result,
        artifact_files=artifacts,
        backend_version=version_match.group(1) if version_match else None,
        warnings=[warning] if warning else [],
        provenance={
            "command": completed["command"],
            "parallel_processes": allocation["cpu_cores"],
            "requested_total_memory_mb": allocation["requested_total_memory_mb"],
            "orca_maxcore_mb_per_process": allocation[
                "orca_maxcore_mb_per_process"
            ],
            "memory_default_applied": allocation["memory_default_applied"],
            "density_source": density_type,
            "silent_density_fallback_allowed": False,
        },
    )


def _electron_density_result(value: Any) -> dict[str, Any]:
    item = unwrap_artifact(value)
    if isinstance(item, dict) and isinstance(item.get("result"), dict):
        item = item["result"]
    if not isinstance(item, dict) or not isinstance(item.get("files"), dict):
        raise ValueError(
            "electron_density must resolve to the ElectronDensityResult returned by "
            "calculate_correlated_electron_density"
        )
    return item


def _copy_density_file(result: dict[str, Any], key: str, target: Path) -> Path:
    raw = result["files"].get(key)
    if not raw:
        raise ValueError(f"ElectronDensityResult does not contain required file {key!r}")
    source = resolve_workspace_path(raw, must_exist=True)
    shutil.copy2(source, target)
    return target


def _cube_electron_integral(path: Path) -> float | None:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 7:
        return None
    try:
        atom_count = abs(int(lines[2].split()[0]))
        vectors = []
        grid_counts = []
        for line in lines[3:6]:
            fields = line.split()
            grid_counts.append(abs(int(fields[0])))
            vectors.append([float(fields[1]), float(fields[2]), float(fields[3])])
        a, b, c = vectors
        voxel = abs(
            a[0] * (b[1] * c[2] - b[2] * c[1])
            - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0])
        )
        values = [
            float(token.replace("D", "E").replace("d", "e"))
            for line in lines[6 + atom_count :]
            for token in line.split()
        ]
        expected = grid_counts[0] * grid_counts[1] * grid_counts[2]
        if len(values) < expected:
            return None
        return sum(values[:expected]) * voxel
    except (ValueError, IndexError):
        return None


def _orca_export_electron_density(request: dict[str, Any]) -> dict[str, Any]:
    inputs, _method, settings = request_parts(request)
    density = _electron_density_result(inputs["electron_density"])
    requested_source = str(settings["density_source"]).casefold()
    output_format = str(settings["output_format"]).casefold()
    calculated_type = str(density.get("density_type", "")).casefold()
    compatible_source = {
        "scf": "scf",
        "relaxed_mp2": "relaxed_mp2",
        "unrelaxed_ccsd": "mdci",
    }.get(calculated_type)
    if requested_source != compatible_source:
        raise ValueError(
            f"Requested density_source={requested_source!r} does not match the calculated "
            f"density_type={calculated_type!r}; silent density substitution is forbidden"
        )
    if requested_source == "mdci" and output_format != "cube":
        raise ValueError(
            "ORCA MDCI density export requires output_format=cube; orca_2aim on the ordinary "
            "GBW would export reference orbitals instead of the MDCI density"
        )
    if requested_source != "mdci" and output_format == "cube":
        raise ValueError(
            f"ORCA density export combination density_source={requested_source!r}, "
            "output_format='cube' is unsupported by the validated adapter. SCF and "
            "relaxed_mp2 densities support output_format='wfn' or 'wfx'; only "
            "density_source='mdci' supports output_format='cube'. Change "
            "action_settings.output_format without changing the calculated density."
        )

    directory = output_directory("export_electron_density_grid", "orca")
    commands: list[list[str]] = []
    warnings: list[str] = []
    if requested_source == "mdci":
        # Keep the original ORCA basename.  The GBW/density metadata records
        # sibling files as ``job.densities`` and ``job.densitiesinfo``;
        # renaming only the copied bundle makes orca_plot look for missing
        # original-name files even when all three payloads are present.
        _copy_density_file(density, "gbw", directory / "job.gbw")
        _copy_density_file(density, "density_container", directory / "job.densities")
        _copy_density_file(density, "density_info", directory / "job.densitiesinfo")
        grid_points = int(settings.get("grid_points_per_axis", 300))
        if grid_points < 20 or grid_points > 400:
            raise ValueError("grid_points_per_axis must be between 20 and 400")
        electron_tolerance = float(
            settings.get("electron_count_tolerance_percent", 0.2)
        )
        if (
            not math.isfinite(electron_tolerance)
            or electron_tolerance < 0.001
            or electron_tolerance > 10
        ):
            raise ValueError(
                "electron_count_tolerance_percent must be between 0.001 and 10"
            )
        strict_electron_validation = settings.get(
            "strict_electron_count_validation", False
        )
        if not isinstance(strict_electron_validation, bool):
            raise ValueError("strict_electron_count_validation must be a boolean")
        walltime_seconds = int(
            request.get("resource_limits", {}).get("walltime_seconds", 1800)
        )
        if grid_points >= 300 and walltime_seconds < 1800:
            raise ValueError(
                "A 300^3-or-larger MDCI cube export requires "
                "resource_limits.walltime_seconds>=1800. orca_plot is a "
                "single-process exporter, so additional cpu_cores do not compensate "
                "for a shorter walltime."
            )
        menu = f"1\n7\ny\n4\n{grid_points} {grid_points} {grid_points}\n11\n12\n"
        completed = run_external(
            executable="orca_plot",
            arguments=["job.gbw", "-i"],
            directory=directory,
            stdin_text=menu,
            timeout_seconds=walltime_seconds,
        )
        commands.append(completed["command"])
        (directory / "orca_plot.out").write_text(completed["stdout"], encoding="utf-8")
        (directory / "orca_plot.err").write_text(completed["stderr"], encoding="utf-8")
        if not completed["available"]:
            return unavailable(completed["stderr"], install="Configure ORCA orca_plot")
        if completed["returncode"] != 0:
            raise RuntimeError(
                "orca_plot MDCI density export failed: "
                + (completed["stderr"] or completed["stdout"])[-3000:]
            )
        candidates = sorted(directory.glob("*.cube"))
        if not candidates:
            raise RuntimeError("orca_plot completed without producing an MDCI density cube")
        primary = candidates[-1]
        integrated_electrons = _cube_electron_integral(primary)
        expected_electrons = density.get("electron_count")
        electron_count_error_percent = None
        electron_count_validation_passed = None
        if integrated_electrons is None:
            warnings.append("Could not integrate the generated cube for an electron-count check")
        elif expected_electrons is None or float(expected_electrons) <= 0:
            warnings.append(
                "The source density artifact does not declare an expected electron count"
            )
        else:
            electron_count_error_percent = (
                abs(float(integrated_electrons) - float(expected_electrons))
                / float(expected_electrons)
                * 100.0
            )
            electron_count_validation_passed = (
                electron_count_error_percent <= electron_tolerance
            )
            if not electron_count_validation_passed:
                message = (
                    f"Generated cube integrates to {integrated_electrons:.8g} electrons "
                    f"instead of {float(expected_electrons):.8g} "
                    f"({electron_count_error_percent:.4g}% error; tolerance "
                    f"{electron_tolerance:.4g}%). Increase grid_points_per_axis or "
                    "review the density/ECP electron-count convention."
                )
                if strict_electron_validation:
                    raise RuntimeError(message)
                warnings.append(message)
    else:
        _copy_density_file(density, "gbw", directory / "density.gbw")
        if requested_source == "relaxed_mp2":
            _copy_density_file(
                density, "natural_orbitals", directory / "density.mp2nat"
            )
        completed = run_external(
            executable="orca_2aim",
            arguments=["density"],
            directory=directory,
            timeout_seconds=int(
                request.get("resource_limits", {}).get("walltime_seconds", 1800)
            ),
        )
        commands.append(completed["command"])
        (directory / "orca_2aim.out").write_text(completed["stdout"], encoding="utf-8")
        (directory / "orca_2aim.err").write_text(completed["stderr"], encoding="utf-8")
        if not completed["available"]:
            return unavailable(completed["stderr"], install="Configure ORCA orca_2aim")
        if completed["returncode"] != 0:
            raise RuntimeError(
                "orca_2aim density export failed: "
                + (completed["stderr"] or completed["stdout"])[-3000:]
            )
        if requested_source == "relaxed_mp2" and "mp2nat" not in completed["stdout"].casefold():
            raise RuntimeError(
                "orca_2aim did not confirm reading the requested relaxed-MP2 .mp2nat density"
            )
        primary = directory / f"density.{output_format}"
        if not primary.is_file():
            raise RuntimeError(
                f"orca_2aim completed without producing the requested {output_format.upper()} file"
            )
        integrated_electrons = None
        expected_electrons = None
        electron_count_error_percent = None
        electron_count_validation_passed = None
        electron_tolerance = None
        strict_electron_validation = None

    result = {
        "density_source": requested_source,
        "output_format": output_format,
        "output_file": relative_workspace_path(primary),
        "grid_points_per_axis": grid_points if requested_source == "mdci" else None,
        "integrated_electrons": integrated_electrons,
        "expected_electrons": expected_electrons,
        "electron_count_error_percent": electron_count_error_percent,
        "electron_count_tolerance_percent": electron_tolerance,
        "electron_count_validation_passed": electron_count_validation_passed,
        "strict_electron_count_validation": strict_electron_validation,
        "source_method": density.get("method"),
        "source_basis": density.get("basis"),
    }
    result_path = write_json(directory, "result.json", result)
    artifacts = _orca_density_artifacts(directory)
    result_relative = relative_workspace_path(result_path)
    for item in artifacts:
        if item["path"] == result_relative:
            item["semantic_type"] = "ElectronDensityExportResult"
            item["media_type"] = "application/json"
    return success(
        result,
        artifact_files=artifacts,
        backend_version="6.1.1",
        warnings=warnings,
        provenance={
            "commands": commands,
            "density_source": requested_source,
            "grid_points_per_axis": grid_points if requested_source == "mdci" else None,
            "electron_count_tolerance_percent": electron_tolerance,
            "strict_electron_count_validation": strict_electron_validation,
            "orca_plot_parallel_processes": 1 if requested_source == "mdci" else None,
            "silent_density_fallback_allowed": False,
        },
    )


def _orca_completion_error(stdout: str, stderr: str) -> str | None:
    """Return a concise diagnostic when ORCA did not finish normally.

    ORCA can return process exit code zero even when one of its MPI helper
    programs aborts.  The program's own termination banner is therefore the
    authoritative completion signal.
    """

    error_markers = (
        "ORCA finished by error termination",
        "aborting the run",
    )
    combined = f"{stdout}\n{stderr}"
    if not any(marker.casefold() in combined.casefold() for marker in error_markers):
        if "ORCA TERMINATED NORMALLY" in stdout:
            return None

    diagnostic_lines = [
        line.strip()
        for line in (*stderr.splitlines(), *stdout.splitlines()[-40:])
        if line.strip()
        and (
            "error" in line.casefold()
            or "abort" in line.casefold()
            or "pmix" in line.casefold()
            or "mpi_" in line.casefold()
            or "terminated" in line.casefold()
        )
    ]
    if diagnostic_lines:
        return " | ".join(dict.fromkeys(diagnostic_lines))[-4000:]
    return "ORCA output did not contain the 'ORCA TERMINATED NORMALLY' completion marker"


def _orca(action_id: str, request: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    if action_id == "calculate_correlated_electron_density":
        return _orca_correlated_electron_density(request)
    if action_id == "export_electron_density_grid":
        return _orca_export_electron_density(request)

    inputs, method, settings = request_parts(request)
    directory = output_directory(action_id, "orca")
    input_path = directory / "job.inp"
    input_path.write_text(
        _render_orca(
            action_id,
            inputs["structure"],
            method,
            settings,
            dict(request.get("resource_limits") or {}),
        ),
        encoding="utf-8",
    )
    completed = run_external(
        executable="orca", environment_variable="CHEMGRAPH_ORCA_COMMAND",
        # ORCA reuses the supplied input pathname as the basename for several
        # module-specific files.  Passing the deeply nested absolute workspace
        # path can make property, frequency, and correlated-density modules
        # terminate without a useful diagnostic.  Keep the calculation cwd at
        # the output directory and give ORCA a short explicit relative path.
        # (A bare ``job.inp`` is rejected by some ORCA builds.)
        arguments=[f"./{input_path.name}"], directory=directory,
        timeout_seconds=int(request.get("resource_limits", {}).get("walltime_seconds", 1800)),
    )
    output_path = directory / "job.out"
    output_path.write_text(completed["stdout"], encoding="utf-8")
    (directory / "job.err").write_text(completed["stderr"], encoding="utf-8")
    if not completed["available"]:
        return unavailable(
            completed["stderr"],
            install=(
                "Place the licensed ORCA bundle in .software_cache/orca/6.1.1 "
                "and configure CHEMGRAPH_ORCA_COMMAND."
            ),
        )
    if completed["returncode"] != 0:
        detail = (completed["stderr"] or completed["stdout"])[-4000:]
        raise RuntimeError(f"ORCA failed: {detail}")
    completion_error = _orca_completion_error(completed["stdout"], completed["stderr"])
    if completion_error is not None:
        raise RuntimeError(f"ORCA did not terminate normally: {completion_error}")
    version_match = re.search(r"Program Version\s+(\d+\.\d+\.\d+)", completed["stdout"])
    backend_version = version_match.group(1) if version_match else None
    energy_matches = re.findall(r"FINAL SINGLE POINT ENERGY\s+(-?\d+(?:\.\d+)?)", completed["stdout"])
    energy = float(energy_matches[-1]) if energy_matches else None
    if action_id == "calculate_energy":
        if energy is None:
            raise RuntimeError("Could not parse ORCA final energy")
        result = {"energy": energy, "unit": "hartree"}
    elif action_id == "calculate_dipole_moment":
        match = re.search(
            r"Total Dipole Moment\s*:\s*(-?\S+)\s+(-?\S+)\s+(-?\S+)", completed["stdout"]
        )
        if not match:
            raise RuntimeError("Could not parse ORCA dipole moment")
        result = {"dipole": [float(match.group(i)) for i in (1, 2, 3)], "unit": "atomic_unit", "energy_hartree": energy}
    elif action_id == "calculate_forces":
        engrad_files = sorted(directory.glob("*.engrad"))
        if not engrad_files:
            raise RuntimeError("ORCA gradient job produced no .engrad file")
        numeric_lines = []
        for line in engrad_files[-1].read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = line.strip().replace("D", "E").replace("d", "e")
            if not stripped or stripped.startswith("#") or len(stripped.split()) != 1:
                continue
            try:
                numeric_lines.append(float(stripped))
            except ValueError:
                continue
        atom_count = len(structure_dict(inputs["structure"])["atoms"])
        if len(numeric_lines) < 2 + 3 * atom_count:
            raise RuntimeError("Could not parse the ORCA .engrad gradient block")
        gradient = np.asarray(numeric_lines[2 : 2 + 3 * atom_count], dtype=float).reshape(atom_count, 3)
        conversion = 27.211386245988 / 0.529177210903
        result = {
            "forces": (-gradient * conversion).tolist(),
            "unit": "eV/angstrom",
            "atom_count": atom_count,
            "energy_hartree": energy,
            "raw_gradient_path": relative_workspace_path(engrad_files[-1]),
        }
    elif action_id == "calculate_atomic_charges":
        analysis = str(method["population_analysis"]).strip().lower()
        title = {
            "mulliken": "MULLIKEN ATOMIC CHARGES",
            "loewdin": "LOEWDIN ATOMIC CHARGES",
        }.get(analysis)
        if title is None:
            raise ValueError("ORCA population_analysis must be mulliken or loewdin")
        block = re.search(
            rf"{title}\s*-+\s*(.*?)(?:\n\s*\n|\n\s*Sum of)",
            completed["stdout"],
            flags=re.S,
        )
        if not block:
            raise RuntimeError(f"Could not parse ORCA {analysis} atomic charges")
        charges = [
            float(match.group(1))
            for match in re.finditer(
                r"^\s*\d+\s+[A-Za-z][A-Za-z]?\s*:\s*(-?\d+(?:\.\d+)?)",
                block.group(1),
                flags=re.M,
            )
        ]
        if not charges:
            raise RuntimeError(f"ORCA {analysis} charge block contained no atom rows")
        result = {
            "charges": charges,
            "analysis": analysis,
            "unit": "elementary_charge",
            "energy_hartree": energy,
        }
    elif action_id == "calculate_orbitals":
        block = re.search(
            r"ORBITAL ENERGIES\s*-+\s*(.*?)(?:\n\s*\n|\*Only)",
            completed["stdout"],
            flags=re.S,
        )
        if not block:
            raise RuntimeError("Could not parse ORCA orbital energies")
        orbitals = []
        for line in block.group(1).splitlines():
            match = re.match(
                r"^\s*(\d+)\s+([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)",
                line,
            )
            if match:
                orbitals.append(
                    {
                        "orbital_index": int(match.group(1)),
                        "occupation": float(match.group(2)),
                        "energy_hartree": float(match.group(3)),
                        "energy_ev": float(match.group(4)),
                    }
                )
        if not orbitals:
            raise RuntimeError("ORCA orbital table contained no orbital rows")
        result = {"orbitals": orbitals, "energy_hartree": energy}
    elif action_id == "calculate_bond_orders":
        minimum = float(settings["minimum_bond_order"])
        bonds = [
            {
                "atom_index_a": int(match.group(1)),
                "element_a": match.group(2),
                "atom_index_b": int(match.group(3)),
                "element_b": match.group(4),
                "bond_order": float(match.group(5)),
            }
            for match in re.finditer(
                r"B\(\s*(\d+)-([A-Za-z]+)\s*,\s*(\d+)-([A-Za-z]+)\s*\)\s*:\s*([-+]?\d+(?:\.\d+)?)",
                completed["stdout"],
            )
        ]
        result = {
            "bonds": bonds,
            "analysis": "mayer",
            "minimum_bond_order": minimum,
            "energy_hartree": energy,
        }
    elif action_id == "calculate_excited_states":
        oscillator_by_state = {}
        absorption = re.search(
            r"ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS(.*?)(?:ABSORPTION SPECTRUM VIA TRANSITION VELOCITY|CD SPECTRUM)",
            completed["stdout"],
            flags=re.S,
        )
        if absorption:
            for match in re.finditer(
                r"^\s*0-\S+\s+->\s+(\d+)-\S+\s+([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)\s+[-+]?\d+(?:\.\d+)?\s+([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)",
                absorption.group(1),
                flags=re.M,
            ):
                oscillator_by_state[int(match.group(1))] = {
                    "energy_ev": float(match.group(2)),
                    "wavenumber_cm1": float(match.group(3)),
                    "wavelength_nm": float(match.group(4)),
                    "oscillator_strength": float(match.group(5)),
                    "transition_dipole_atomic_unit": [
                        float(match.group(6)),
                        float(match.group(7)),
                        float(match.group(8)),
                    ],
                }
        states = []
        for match in re.finditer(
            r"^STATE\s+(\d+):\s+E=\s*([-+]?\d+(?:\.\d+)?)\s+au\s+([-+]?\d+(?:\.\d+)?)\s+eV\s+([-+]?\d+(?:\.\d+)?)\s+cm\*\*-1\s+<S\*\*2>\s*=\s*([-+]?\d+(?:\.\d+)?)\s+Mult\s+(\d+)",
            completed["stdout"],
            flags=re.M,
        ):
            index = int(match.group(1))
            states.append(
                {
                    "state_index": index,
                    "energy_hartree": float(match.group(2)),
                    "energy_ev": float(match.group(3)),
                    "wavenumber_cm1": float(match.group(4)),
                    "spin_squared": float(match.group(5)),
                    "multiplicity": int(match.group(6)),
                    "spin_symmetry": "singlet",
                    **oscillator_by_state.get(index, {}),
                }
            )
        if not states:
            raise RuntimeError("Could not parse ORCA excited states")
        result = {
            "states": states,
            "excited_state_method": str(method["excited_state_method"]).lower(),
            "spin_symmetry": "singlet",
            "ground_state_energy_hartree": energy,
        }
    elif action_id == "optimize_geometry":
        final_xyz = directory / f"{input_path.stem}.xyz"
        if not final_xyz.is_file():
            candidates = sorted(
                path for path in directory.glob("*.xyz") if not path.name.endswith("_trj.xyz")
            )
            final_xyz = candidates[-1] if candidates else final_xyz
        if not final_xyz.is_file():
            raise RuntimeError("ORCA optimization produced no XYZ structure")
        result = {
            "structure": structure_dict(relative_workspace_path(final_xyz)),
            "converged": "THE OPTIMIZATION HAS CONVERGED" in completed["stdout"],
            "energy": energy,
            "energy_unit": "hartree",
            "optimization_convergence": str(settings["optimization_convergence"]),
        }
    elif action_id == "calculate_hessian":
        hess_files = list(directory.glob("*.hess"))
        if not hess_files:
            raise RuntimeError("ORCA frequency job produced no .hess file")
        matrix = _parse_orca_hessian(hess_files[-1])
        result = {
            "matrix": matrix,
            "unit": "hartree/bohr^2",
            "raw_hessian_path": relative_workspace_path(hess_files[-1]),
        }
    else:
        return unsupported(f"ORCA does not implement {action_id}")
    artifacts = command_artifacts(directory)
    provenance = {
        "command": completed["command"],
        "parallel_processes": int(
            (request.get("resource_limits") or {}).get("cpu_cores") or 1
        ),
    }
    incomplete = (
        (action_id == "calculate_hessian" and result.get("matrix") is None)
        or (action_id == "optimize_geometry" and not result.get("converged"))
    )
    if incomplete:
        return partial_success(
            result,
            artifact_files=artifacts,
            backend_version=backend_version,
            provenance=provenance,
            warnings=[
                "ORCA completed, but the primary result was not fully parsed or the optimization did not converge."
            ],
        )
    return success(
        result,
        artifact_files=artifacts,
        backend_version=backend_version,
        provenance=provenance,
    )


def _vibrations(request: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    from ase import units

    inputs, _method, settings = request_parts(request)
    hessian_value = unwrap_artifact(inputs["hessian"])
    if isinstance(hessian_value, dict) and "matrix" not in hessian_value and "result" in hessian_value:
        hessian_value = hessian_value["result"]
    if not isinstance(hessian_value, dict) or hessian_value.get("matrix") is None:
        raise ValueError("hessian must resolve to a Hessian result containing a dense matrix")
    matrix = np.asarray(hessian_value["matrix"] if isinstance(hessian_value, dict) else hessian_value, dtype=float)
    unit = str(hessian_value.get("unit", "eV/angstrom^2")) if isinstance(hessian_value, dict) else "eV/angstrom^2"
    if unit == "hartree/bohr^2":
        matrix = matrix * 27.211386245988 / (0.529177210903**2)
    elif unit != "eV/angstrom^2":
        raise ValueError(f"Unsupported Hessian unit: {unit}")
    atoms = ase_atoms(inputs["structure"])
    expected = 3 * len(atoms)
    if matrix.shape != (expected, expected):
        raise ValueError(f"Hessian shape must be {(expected, expected)}, received {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Hessian matrix must contain only finite values")

    masses = np.asarray(atoms.get_masses(), dtype=float)
    if not np.all(np.isfinite(masses)) or np.any(masses <= 0.0):
        raise ValueError("Every atom requires a finite positive mass")
    positions = np.asarray(atoms.get_positions(), dtype=float)
    center_of_mass = np.average(positions, axis=0, weights=masses)
    centered_positions = positions - center_of_mass

    inertia_tensor = np.zeros((3, 3), dtype=float)
    for mass, position in zip(masses, centered_positions):
        inertia_tensor += mass * (
            np.dot(position, position) * np.eye(3) - np.outer(position, position)
        )
    principal_moments, principal_axes = np.linalg.eigh(inertia_tensor)
    principal_moments = np.maximum(principal_moments, 0.0)
    maximum_moment = float(principal_moments[-1])
    if len(atoms) > 1 and maximum_moment <= np.finfo(float).eps:
        raise ValueError(
            "Molecular linearity cannot be determined because all atoms are coincident"
        )

    linearity_tolerance = 1.0e-8
    if len(atoms) == 1:
        inferred_linearity = "monatomic"
        moment_ratio = 0.0
    else:
        moment_ratio = float(principal_moments[0] / maximum_moment)
        inferred_linearity = (
            "linear" if moment_ratio <= linearity_tolerance else "nonlinear"
        )
    requested_linearity = str(settings["linearity"]).strip().casefold()
    if requested_linearity not in {"linear", "nonlinear", "auto"}:
        raise ValueError("linearity must be linear, nonlinear, or auto")
    if len(atoms) == 1:
        if requested_linearity == "nonlinear":
            raise ValueError("linearity='nonlinear' is incompatible with a monatomic structure")
        resolved_linearity = "monatomic"
    elif requested_linearity == "auto":
        resolved_linearity = inferred_linearity
    else:
        if requested_linearity != inferred_linearity:
            raise ValueError(
                f"linearity={requested_linearity!r} conflicts with the structure's "
                f"inertia-tensor classification {inferred_linearity!r} "
                f"(minimum/maximum principal-moment ratio={moment_ratio:.6g}, "
                f"linear threshold={linearity_tolerance:.6g})"
            )
        resolved_linearity = requested_linearity

    square_root_masses = np.sqrt(masses)
    translations = np.zeros((expected, 3), dtype=float)
    for axis in range(3):
        translations[axis::3, axis] = square_root_masses
    translation_basis, _ = np.linalg.qr(translations, mode="reduced")

    rotations = []
    for axis in principal_axes.T:
        displacements = np.cross(axis[np.newaxis, :], centered_positions)
        rotations.append((square_root_masses[:, np.newaxis] * displacements).reshape(-1))
    rotation_candidates = np.stack(rotations, axis=1)
    rotation_candidates -= translation_basis @ (
        translation_basis.T @ rotation_candidates
    )
    rotation_vectors, rotation_singular_values, _ = np.linalg.svd(
        rotation_candidates, full_matrices=False
    )
    rotational_degrees = {
        "monatomic": 0,
        "linear": 2,
        "nonlinear": 3,
    }[resolved_linearity]
    rotation_basis = rotation_vectors[:, :rotational_degrees]
    external_candidates = np.column_stack((translation_basis, rotation_basis))
    external_basis, _ = np.linalg.qr(external_candidates, mode="reduced")
    external_degrees = 3 + rotational_degrees
    complete_basis = np.linalg.qr(external_basis, mode="complete")[0]
    internal_basis = complete_basis[:, external_degrees:]

    inverse_square_root_masses = np.repeat(masses**-0.5, 3)
    symmetric_matrix = 0.5 * (matrix + matrix.T)
    mass_weighted_hessian = (
        inverse_square_root_masses[:, np.newaxis]
        * symmetric_matrix
        * inverse_square_root_masses[np.newaxis, :]
    )
    internal_hessian = internal_basis.T @ mass_weighted_hessian @ internal_basis
    eigenvalues, internal_modes = np.linalg.eigh(
        0.5 * (internal_hessian + internal_hessian.T)
    )
    mass_weighted_modes = internal_basis @ internal_modes
    modes = (
        inverse_square_root_masses[:, np.newaxis] * mass_weighted_modes
    ).T.reshape(len(eigenvalues), len(atoms), 3)

    unit_conversion = units._hbar * units.m / math.sqrt(units._e * units._amu)
    energies = unit_conversion * eigenvalues.astype(complex) ** 0.5
    frequencies = energies / units.invcm
    imaginary_frequency_count = int(np.count_nonzero(eigenvalues < 0.0))
    return success(
        {
            "frequencies_cm1": [
                {"real": float(value.real), "imaginary": float(value.imag)} for value in frequencies
            ],
            "vibrational_energies_ev": [
                {"real": float(value.real), "imaginary": float(value.imag)} for value in energies
            ],
            "modes": np.asarray(modes).real.tolist(),
            "linearity": resolved_linearity,
            "linearity_requested": requested_linearity,
            "linearity_resolution": {
                "method": "center_of_mass_inertia_tensor_principal_moment_ratio",
                "inferred_from_structure": inferred_linearity,
                "principal_moments_amu_angstrom2": [
                    float(value) for value in principal_moments
                ],
                "minimum_to_maximum_principal_moment_ratio": moment_ratio,
                "linear_threshold": linearity_tolerance,
                "rotational_subspace_singular_values_sqrt_amu_angstrom": [
                    float(value) for value in rotation_singular_values
                ],
            },
            "cartesian_degree_count": expected,
            "external_degree_count": external_degrees,
            "vibrational_mode_count": len(eigenvalues),
            "imaginary_frequency_count": imaginary_frequency_count,
            "external_mode_treatment": (
                "Translations and rotations were projected in mass-weighted Cartesian "
                "coordinates before diagonalization; modes were not removed by a frequency cutoff."
            ),
            "structure": structure_from_atoms(atoms),
        },
        backend_version=module_version("ase"),
        provenance={
            "external_mode_projection": "mass_weighted_translation_rotation_subspace",
            "requested_linearity": requested_linearity,
            "resolved_linearity": resolved_linearity,
            "automatic_scientific_defaults": False,
            "automatic_fallback_count": 0,
        },
    )


def _ir_spectrum(request: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    inputs, _method, settings = request_parts(request)
    vibrations = unwrap_artifact(inputs["vibrations"])
    if isinstance(vibrations, dict) and "result" in vibrations:
        vibrations = vibrations["result"]
    frequencies = vibrations.get("frequencies_cm1")
    intensities = vibrations.get("intensities") or vibrations.get("intensities_km_mol")
    if not frequencies or intensities is None:
        raise ValueError("vibrations must contain frequencies_cm1 and intensities")
    values = [float(item.get("real", item) if isinstance(item, dict) else item) for item in frequencies]
    intensities = [float(value) for value in intensities]
    if len(values) != len(intensities):
        raise ValueError("frequencies and intensities must be aligned")
    fwhm = float(settings["fwhm_cm1"])
    if fwhm <= 0:
        raise ValueError("fwhm_cm1 must be positive")
    start = float(settings.get("min_wavenumber_cm1", max(0.0, min(values) - 5 * fwhm)))
    end = float(settings.get("max_wavenumber_cm1", max(values) + 5 * fwhm))
    points = int(settings.get("points", 4000))
    grid = np.linspace(start, end, points)
    broadening = str(settings["broadening"]).lower()
    spectrum = np.zeros_like(grid)
    if broadening == "gaussian":
        sigma = fwhm / (2 * math.sqrt(2 * math.log(2)))
        for frequency, intensity in zip(values, intensities):
            spectrum += intensity * np.exp(-0.5 * ((grid - frequency) / sigma) ** 2)
    elif broadening == "lorentzian":
        gamma = fwhm / 2
        for frequency, intensity in zip(values, intensities):
            spectrum += intensity * gamma**2 / ((grid - frequency) ** 2 + gamma**2)
    else:
        raise ValueError("broadening must be gaussian or lorentzian")
    return success(
        {
            "wavenumber_cm1": grid.tolist(),
            "intensity": spectrum.tolist(),
            "line_frequencies_cm1": values,
            "line_intensities": intensities,
            "broadening": broadening,
            "fwhm_cm1": fwhm,
        },
        backend_version=module_version("numpy"),
    )


def _uv_vis_spectrum(request: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    inputs, _method, settings = request_parts(request)
    supplied = unwrap_artifact(inputs["excited_states"])
    if isinstance(supplied, dict) and "result" in supplied:
        supplied = supplied["result"]
    states = supplied.get("states") if isinstance(supplied, dict) else supplied
    if not isinstance(states, list) or not states:
        raise ValueError("excited_states must contain a non-empty states list")
    energies = np.asarray([float(state["energy_ev"]) for state in states], dtype=float)
    strengths = np.asarray(
        [float(state.get("oscillator_strength", 0.0)) for state in states], dtype=float
    )
    minimum = float(settings["minimum_energy_ev"])
    maximum = float(settings["maximum_energy_ev"])
    points = int(settings["grid_points"])
    fwhm = float(settings["fwhm_ev"])
    if not minimum < maximum or points < 2 or fwhm <= 0:
        raise ValueError("UV/visible grid bounds, grid_points, and fwhm_ev are invalid")
    grid = np.linspace(minimum, maximum, points)
    broadening = str(settings["broadening"]).strip().lower()
    intensity = np.zeros_like(grid)
    if broadening == "gaussian":
        sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        for energy, strength in zip(energies, strengths):
            intensity += strength * np.exp(-0.5 * ((grid - energy) / sigma) ** 2)
    elif broadening == "lorentzian":
        gamma = fwhm / 2.0
        for energy, strength in zip(energies, strengths):
            intensity += strength * gamma**2 / ((grid - energy) ** 2 + gamma**2)
    else:
        raise ValueError("broadening must be gaussian or lorentzian")
    return success(
        {
            "energy_ev": grid.tolist(),
            "intensity": intensity.tolist(),
            "line_energy_ev": energies.tolist(),
            "line_oscillator_strength": strengths.tolist(),
            "broadening": broadening,
            "fwhm_ev": fwhm,
        },
        backend_version=module_version("numpy"),
    )


def _internal_thermochemistry(request: dict[str, Any]) -> dict[str, Any]:
    from ase.thermochemistry import IdealGasThermo

    inputs, _method, settings = request_parts(request)
    energy_value = unwrap_artifact(inputs["energy"])
    frequencies = unwrap_artifact(inputs["frequencies"])
    if isinstance(energy_value, dict) and "result" in energy_value:
        energy_value = energy_value["result"]
    if isinstance(frequencies, dict) and "result" in frequencies:
        frequencies = frequencies["result"]
    if not isinstance(energy_value, dict):
        raise ValueError("energy must be an EnergyResult containing both a value and explicit unit")
    if "energy" in energy_value:
        energy = float(energy_value["energy"])
        unit_value = energy_value.get("unit") or energy_value.get("energy_unit")
        if unit_value is None:
            raise ValueError("EnergyResult.energy requires unit or energy_unit")
        unit = str(unit_value)
    elif "energy_hartree" in energy_value:
        energy = float(energy_value["energy_hartree"])
        unit = "hartree"
    elif "energy_ev" in energy_value:
        energy = float(energy_value["energy_ev"])
        unit = "eV"
    else:
        raise ValueError("EnergyResult must contain energy, energy_hartree, or energy_ev")
    normalized_unit = unit.strip().casefold()
    if normalized_unit in {"hartree", "eh"}:
        energy *= 27.211386245988
    elif normalized_unit != "ev":
        raise ValueError(f"Unsupported energy unit: {unit}")
    if not isinstance(frequencies, dict):
        raise ValueError("frequencies must resolve to a FrequencyResult object")

    def mode_value(item: Any, conversion: float = 1.0) -> complex:
        if isinstance(item, dict):
            if "real" not in item:
                raise ValueError("Frequency entries must contain real and optional imaginary values")
            value = complex(float(item["real"]), float(item.get("imaginary", 0.0)))
        else:
            value = complex(float(item), 0.0)
        return value * conversion

    vib_energies = frequencies.get("vibrational_energies_ev")
    if vib_energies is None:
        cm1 = frequencies["frequencies_cm1"]
        vib_energies = [mode_value(item, 1.2398419843320026e-4) for item in cm1]
    else:
        vib_energies = [mode_value(item) for item in vib_energies]
    atoms_value = inputs["structure"] if "structure" in inputs else frequencies.get("structure")
    if atoms_value is None:
        return unsupported("internal_thermochemistry requires structure in inputs or FrequencyResult")
    atoms = ase_atoms(atoms_value)
    geometry = str(settings["geometry"]).strip().casefold()
    symmetry_number = int(settings["symmetry_number"])
    spin = float(settings["spin"])
    temperature = float(settings["temperature_kelvin"])
    pressure = float(settings["pressure_pa"])
    if symmetry_number <= 0:
        raise ValueError("symmetry_number must be a positive integer")
    if spin < 0 or not math.isfinite(spin):
        raise ValueError("spin must be a finite nonnegative number")
    if temperature <= 0 or pressure <= 0:
        raise ValueError("temperature_kelvin and pressure_pa must be positive")
    thermo = IdealGasThermo(
        vib_energies=vib_energies,
        potentialenergy=energy,
        atoms=atoms,
        geometry=geometry,
        symmetrynumber=symmetry_number,
        spin=spin,
        ignore_imag_modes=bool(settings["ignore_imaginary_modes"]),
    )
    enthalpy = float(thermo.get_enthalpy(temperature, verbose=False))
    entropy = float(thermo.get_entropy(temperature, pressure, verbose=False))
    gibbs = float(thermo.get_gibbs_energy(temperature, pressure, verbose=False))
    if not all(math.isfinite(value) for value in (enthalpy, entropy, gibbs)):
        raise ValueError(
            "Thermochemistry produced a non-finite value; verify the explicitly supplied "
            "geometry, structure, symmetry number, spin, and vibrational modes"
        )
    return success(
        {
            "temperature_kelvin": temperature,
            "pressure_pa": pressure,
            "enthalpy_ev": enthalpy,
            "entropy_ev_per_kelvin": entropy,
            "gibbs_free_energy_ev": gibbs,
        },
        backend_version=module_version("ase"),
    )


_FREE_ENERGY_TO_KCAL_MOL = {
    "hartree": 627.5094740631,
    "ev": 23.060547830619,
    "kcal_mol": 1.0,
    "kj_mol": 1.0 / 4.184,
}
_GAS_CONSTANT_KCAL_MOL_K = 0.00198720425864083


def _canonical_free_energy_unit(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be an explicit non-empty string")
    unit = value.strip().casefold().replace("-", "_").replace("/", "_")
    unit = {
        "eh": "hartree",
        "au": "hartree",
        "a.u.": "hartree",
        "electron_volt": "ev",
        "electron_volts": "ev",
        "kcalmol": "kcal_mol",
        "kilocalorie_mol": "kcal_mol",
        "kilocalories_mol": "kcal_mol",
        "kjmol": "kj_mol",
        "kilojoule_mol": "kj_mol",
        "kilojoules_mol": "kj_mol",
    }.get(unit, unit)
    if unit not in _FREE_ENERGY_TO_KCAL_MOL:
        raise ValueError(
            f"{path} must be one of hartree, eV, kcal_mol, or kJ_mol"
        )
    return unit


def _finite_real(value: Any, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be finite")
    return result


def _free_energy_selectivity(request: dict[str, Any]) -> dict[str, Any]:
    inputs, _method, settings = request_parts(request)
    records = unwrap_artifact(inputs["free_energies"])
    if isinstance(records, dict) and "result" in records:
        records = records["result"]
    if not isinstance(records, list) or len(records) < 2:
        raise ValueError("free_energies must be a list containing at least two records")

    unit = _canonical_free_energy_unit(inputs["energy_unit"], path="energy_unit")
    temperature = _finite_real(
        settings["temperature_kelvin"], path="temperature_kelvin"
    )
    if temperature <= 0.0:
        raise ValueError("temperature_kelvin must be a finite positive number")
    rt = _GAS_CONSTANT_KCAL_MOL_K * temperature
    if not math.isfinite(rt) or rt <= 0.0:
        raise ValueError(
            "temperature_kelvin is too small to produce a finite positive RT in kcal/mol"
        )
    comparison_type = str(settings["comparison_type"]).strip().casefold()
    if comparison_type not in {"population", "enantiomeric"}:
        raise ValueError("comparison_type must be population or enantiomeric")

    labels: list[str] = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    conformer_id_locations: dict[str, tuple[int, str]] = {}
    factor = _FREE_ENERGY_TO_KCAL_MOL[unit]
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"free_energies[{index}] must be a mapping")
        raw_label = record.get("label")
        if not isinstance(raw_label, str) or not raw_label.strip():
            raise ValueError(f"free_energies[{index}].label must be a non-empty string")
        label = raw_label.strip()
        if label not in grouped:
            labels.append(label)
            grouped[label] = []
        if "gibbs_free_energy" not in record:
            raise ValueError(
                f"free_energies[{index}] requires gibbs_free_energy; no energy field is inferred"
            )
        for local_unit_field in ("energy_unit", "unit"):
            if local_unit_field in record:
                local_unit = _canonical_free_energy_unit(
                    record[local_unit_field],
                    path=f"free_energies[{index}].{local_unit_field}",
                )
                if local_unit != unit:
                    raise ValueError(
                        f"free_energies[{index}].{local_unit_field} conflicts with "
                        f"inputs.energy_unit={unit!r}"
                    )
        value = _finite_real(
            record["gibbs_free_energy"],
            path=f"free_energies[{index}].gibbs_free_energy",
        )
        value_kcal_mol = value * factor
        if not math.isfinite(value_kcal_mol):
            raise ValueError(
                f"free_energies[{index}].gibbs_free_energy overflows after unit conversion"
            )
        conformer_fields: dict[str, Any] = {}
        if "conformer_id" in record:
            conformer_id = record["conformer_id"]
            if not isinstance(conformer_id, str) or not conformer_id.strip():
                raise ValueError(
                    f"free_energies[{index}].conformer_id must be a non-empty string"
                )
            conformer_id = conformer_id.strip()
            if conformer_id in conformer_id_locations:
                previous_index, previous_label = conformer_id_locations[conformer_id]
                raise ValueError(
                    f"free_energies[{index}].conformer_id={conformer_id!r} duplicates "
                    f"free_energies[{previous_index}] in label {previous_label!r}; explicit "
                    "conformer_id values must be globally unique"
                )
            conformer_id_locations[conformer_id] = (index, label)
            conformer_fields["conformer_id"] = conformer_id
        grouped[label].append(
            {
                "record_index": index,
                **conformer_fields,
                "gibbs_free_energy": value,
                "gibbs_free_energy_kcal_mol": value_kcal_mol,
            }
        )
    if len(labels) < 2:
        raise ValueError("free_energies must contain at least two distinct labels")
    if comparison_type == "enantiomeric" and len(labels) != 2:
        raise ValueError(
            "comparison_type='enantiomeric' requires exactly two exhaustive enantiomer labels"
        )

    label_rows: list[dict[str, Any]] = []
    for label in labels:
        members = grouped[label]
        group_minimum = min(
            item["gibbs_free_energy_kcal_mol"] for item in members
        )
        relative_energies = [
            item["gibbs_free_energy_kcal_mol"] - group_minimum for item in members
        ]
        if not all(math.isfinite(value) for value in relative_energies):
            raise ValueError(
                f"free-energy range for label {label!r} is too large to represent"
            )
        member_weights = [
            math.exp(-relative_energy / rt)
            for relative_energy in relative_energies
        ]
        relative_partition_function = math.fsum(member_weights)
        group_free_energy_kcal_mol = (
            group_minimum - rt * math.log(relative_partition_function)
        )
        if not math.isfinite(group_free_energy_kcal_mol):
            raise ValueError(
                f"group free energy for label {label!r} is not finite"
            )
        conformers = []
        for member, relative_energy, member_weight in zip(
            members, relative_energies, member_weights
        ):
            conformers.append(
                {
                    **member,
                    "relative_free_energy_within_label_kcal_mol": relative_energy,
                    "population_within_label": (
                        member_weight / relative_partition_function
                    ),
                }
            )
        label_rows.append(
            {
                "label": label,
                "conformer_count": len(members),
                "relative_partition_function": relative_partition_function,
                "group_free_energy_kcal_mol": group_free_energy_kcal_mol,
                "conformers": conformers,
            }
        )

    minimum_group_free_energy = min(
        row["group_free_energy_kcal_mol"] for row in label_rows
    )
    group_relative_energies = [
        row["group_free_energy_kcal_mol"] - minimum_group_free_energy
        for row in label_rows
    ]
    if not all(math.isfinite(value) for value in group_relative_energies):
        raise ValueError("label-group free-energy range is too large to represent")
    group_weights = [
        math.exp(-relative_energy / rt)
        for relative_energy in group_relative_energies
    ]
    total_weight = math.fsum(group_weights)
    for row, relative_energy, group_weight in zip(
        label_rows, group_relative_energies, group_weights
    ):
        row["relative_group_free_energy_kcal_mol"] = relative_energy
        row["population"] = group_weight / total_weight
        row["population_percent"] = 100.0 * row["population"]

    result: dict[str, Any] = {
        "temperature_kelvin": temperature,
        "input_energy_unit": unit,
        "label_order": labels,
        "labels": label_rows,
        "populations": [
            {
                "label": row["label"],
                "population": row["population"],
                "population_percent": row["population_percent"],
            }
            for row in label_rows
        ],
        "population_sum": math.fsum(row["population"] for row in label_rows),
        "weighting": "boltzmann_partition_function",
        "gas_constant_kcal_mol_kelvin": _GAS_CONSTANT_KCAL_MOL_K,
        "comparison_type": comparison_type,
        "label_semantics_inferred": False,
    }
    if len(label_rows) == 2:
        first, second = label_rows
        delta_delta_g = (
            second["group_free_energy_kcal_mol"]
            - first["group_free_energy_kcal_mol"]
        )
        if not math.isfinite(delta_delta_g):
            raise ValueError("two-label delta-delta-G is not finite")
        excess = 100.0 * (first["population"] - second["population"])
        log_population_ratio = delta_delta_g / rt
        if not math.isfinite(log_population_ratio):
            raise ValueError("two-label log population ratio is not finite")
        maximum_log_float = math.log(float.fromhex("0x1.fffffffffffffp+1023"))
        if abs(log_population_ratio) > maximum_log_float:
            raise ValueError(
                "two-label population ratio cannot be represented as a finite nonzero number "
                "at this temperature"
            )
        population_ratio = math.exp(log_population_ratio)
        absolute_population_excess = abs(excess)
        comparison = {
            "first_label": first["label"],
            "second_label": second["label"],
            "comparison_type": comparison_type,
            "delta_delta_g_kcal_mol": delta_delta_g,
            "delta_delta_g_definition": "second_label_group_free_energy - first_label_group_free_energy",
            "signed_population_excess_percent": excess,
            "signed_population_excess_definition": (
                "100 * (first_label_population - second_label_population)"
            ),
            "absolute_population_excess_percent": absolute_population_excess,
            "population_ratio_first_over_second": population_ratio,
            "log_population_ratio_first_over_second": log_population_ratio,
        }
        two_label_result = {
            "delta_delta_g_kcal_mol": delta_delta_g,
            "signed_population_excess_percent": excess,
            "absolute_population_excess_percent": absolute_population_excess,
            "population_ratio_first_over_second": population_ratio,
            "two_label_comparison": comparison,
        }
        if comparison_type == "enantiomeric":
            comparison["ee_percent"] = absolute_population_excess
            comparison["ee_definition"] = (
                "absolute population excess for two caller-asserted exhaustive enantiomer labels"
            )
            two_label_result["ee_percent"] = absolute_population_excess
        result.update(two_label_result)
    return success(
        result,
        backend_version="internal-1",
        provenance={
            "automatic_scientific_defaults": False,
            "automatic_fallback_count": 0,
            "label_order_basis": "first_appearance_in_free_energies",
            "comparison_type": comparison_type,
        },
    )


_BOLTZMANN_CONSTANT_JOULE_K = 1.380649e-23
_PLANCK_CONSTANT_JOULE_SECOND = 6.62607015e-34


def _energetic_span(request: dict[str, Any]) -> dict[str, Any]:
    inputs, _method, settings = request_parts(request)
    records = unwrap_artifact(inputs["cycle_states"])
    if isinstance(records, dict) and "result" in records:
        records = records["result"]
    if not isinstance(records, list) or len(records) < 2:
        raise ValueError("cycle_states must be a list containing at least two states")
    if len(records) % 2 != 0:
        raise ValueError(
            "cycle_states must contain equal numbers of alternating intermediates and "
            "transition states"
        )

    unit = _canonical_free_energy_unit(inputs["energy_unit"], path="energy_unit")
    factor = _FREE_ENERGY_TO_KCAL_MOL[unit]
    temperature = _finite_real(
        settings["temperature_kelvin"], path="temperature_kelvin"
    )
    if temperature <= 0.0:
        raise ValueError("temperature_kelvin must be a finite positive number")
    rt = _GAS_CONSTANT_KCAL_MOL_K * temperature
    if not math.isfinite(rt) or rt <= 0.0:
        raise ValueError(
            "temperature_kelvin is too small to produce a finite positive RT in kcal/mol"
        )

    overall = _finite_real(
        inputs["overall_reaction_free_energy"],
        path="overall_reaction_free_energy",
    )
    overall_kcal_mol = overall * factor
    if not math.isfinite(overall_kcal_mol):
        raise ValueError(
            "overall_reaction_free_energy overflows after unit conversion"
        )
    if overall_kcal_mol >= 0.0:
        raise ValueError(
            "overall_reaction_free_energy must be negative for the explicitly ordered "
            "forward cycle; reverse an endergonic cycle before requesting a positive net TOF"
        )

    states: list[dict[str, Any]] = []
    seen_ids: dict[str, int] = {}
    expected_types = ("intermediate", "transition_state")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"cycle_states[{index}] must be a mapping")
        raw_id = record.get("id")
        if not isinstance(raw_id, str) or not raw_id.strip():
            raise ValueError(f"cycle_states[{index}].id must be a non-empty string")
        state_id = raw_id.strip()
        if state_id in seen_ids:
            raise ValueError(
                f"cycle_states[{index}].id={state_id!r} duplicates "
                f"cycle_states[{seen_ids[state_id]}].id; state ids must be globally unique"
            )
        seen_ids[state_id] = index
        raw_type = record.get("state_type")
        if not isinstance(raw_type, str):
            raise ValueError(
                f"cycle_states[{index}].state_type must be intermediate or transition_state"
            )
        state_type = raw_type.strip().casefold()
        expected_type = expected_types[index % 2]
        if state_type != expected_type:
            raise ValueError(
                f"cycle_states[{index}].state_type must be {expected_type!r}; the ordered "
                "cycle must alternate intermediate and transition_state, beginning with an "
                "intermediate"
            )
        if "gibbs_free_energy" not in record:
            raise ValueError(
                f"cycle_states[{index}] requires gibbs_free_energy; no energy is inferred"
            )
        for local_unit_field in ("energy_unit", "unit"):
            if local_unit_field in record:
                local_unit = _canonical_free_energy_unit(
                    record[local_unit_field],
                    path=f"cycle_states[{index}].{local_unit_field}",
                )
                if local_unit != unit:
                    raise ValueError(
                        f"cycle_states[{index}].{local_unit_field} conflicts with "
                        f"inputs.energy_unit={unit!r}"
                    )
        value = _finite_real(
            record["gibbs_free_energy"],
            path=f"cycle_states[{index}].gibbs_free_energy",
        )
        value_kcal_mol = value * factor
        if not math.isfinite(value_kcal_mol):
            raise ValueError(
                f"cycle_states[{index}].gibbs_free_energy overflows after unit conversion"
            )
        states.append(
            {
                "id": state_id,
                "state_type": state_type,
                "cycle_position": index,
                "gibbs_free_energy": value,
                "gibbs_free_energy_kcal_mol": value_kcal_mol,
            }
        )

    intermediates = [
        state for state in states if state["state_type"] == "intermediate"
    ]
    transition_states = [
        state for state in states if state["state_type"] == "transition_state"
    ]
    pair_spans: list[dict[str, Any]] = []
    for transition_state in transition_states:
        for intermediate in intermediates:
            transition_state_after_intermediate = (
                transition_state["cycle_position"] > intermediate["cycle_position"]
            )
            correction = 0.0 if transition_state_after_intermediate else overall_kcal_mol
            span = (
                transition_state["gibbs_free_energy_kcal_mol"]
                - intermediate["gibbs_free_energy_kcal_mol"]
                + correction
            )
            if not math.isfinite(span):
                raise ValueError(
                    "an intermediate/transition-state span is not finite after unit conversion"
                )
            pair_spans.append(
                {
                    "intermediate_id": intermediate["id"],
                    "intermediate_cycle_position": intermediate["cycle_position"],
                    "transition_state_id": transition_state["id"],
                    "transition_state_cycle_position": transition_state["cycle_position"],
                    "transition_state_after_intermediate": transition_state_after_intermediate,
                    "overall_reaction_free_energy_correction_applied": (
                        not transition_state_after_intermediate
                    ),
                    "overall_reaction_free_energy_correction_kcal_mol": correction,
                    "span_kcal_mol": span,
                }
            )

    energetic_span = max(pair["span_kcal_mol"] for pair in pair_spans)
    determining_pairs = [
        pair for pair in pair_spans if pair["span_kcal_mol"] == energetic_span
    ]
    primary_pair = determining_pairs[0]

    dimensionless_spans = [pair["span_kcal_mol"] / rt for pair in pair_spans]
    maximum_dimensionless_span = max(dimensionless_spans)
    scaled_denominator = math.fsum(
        math.exp(value - maximum_dimensionless_span)
        for value in dimensionless_spans
    )
    log_denominator = maximum_dimensionless_span + math.log(scaled_denominator)
    overall_over_rt = overall_kcal_mol / rt
    driving_force_factor = -math.expm1(overall_over_rt)
    if not math.isfinite(driving_force_factor) or driving_force_factor <= 0.0:
        raise ValueError(
            "overall_reaction_free_energy and temperature do not produce a finite positive "
            "forward driving-force factor"
        )
    log_prefactor = (
        math.log(_BOLTZMANN_CONSTANT_JOULE_K)
        + math.log(temperature)
        - math.log(_PLANCK_CONSTANT_JOULE_SECOND)
    )
    log_tof = log_prefactor + math.log(driving_force_factor) - log_denominator
    minimum_log_float = math.log(float.fromhex("0x0.0000000000001p-1022"))
    maximum_log_float = math.log(float.fromhex("0x1.fffffffffffffp+1023"))
    if not math.isfinite(log_tof) or not minimum_log_float <= log_tof <= maximum_log_float:
        raise ValueError(
            "turnover frequency cannot be represented as a finite nonzero number at this "
            "temperature; inspect the energy scale and units"
        )
    turnover_frequency = math.exp(log_tof)
    if not math.isfinite(turnover_frequency) or turnover_frequency <= 0.0:
        raise ValueError(
            "turnover frequency cannot be represented as a finite nonzero number at this "
            "temperature"
        )
    full_denominator = (
        math.exp(log_denominator)
        if log_denominator <= maximum_log_float
        else None
    )
    transition_state_theory_prefactor = (
        math.exp(log_prefactor)
        if log_prefactor <= maximum_log_float
        else None
    )

    return success(
        {
            "temperature_kelvin": temperature,
            "input_energy_unit": unit,
            "overall_reaction_free_energy": overall,
            "overall_reaction_free_energy_kcal_mol": overall_kcal_mol,
            "cycle_state_order": [state["id"] for state in states],
            "states": states,
            "intermediate_count": len(intermediates),
            "transition_state_count": len(transition_states),
            "pair_count": len(pair_spans),
            "pair_spans": pair_spans,
            "energetic_span_kcal_mol": energetic_span,
            "turnover_determining_intermediate_id": primary_pair["intermediate_id"],
            "turnover_determining_transition_state_id": primary_pair["transition_state_id"],
            "determining_pair_count": len(determining_pairs),
            "determining_pairs": determining_pairs,
            "transition_state_theory_prefactor_per_second": transition_state_theory_prefactor,
            "log_transition_state_theory_prefactor_per_second": log_prefactor,
            "forward_driving_force_factor": driving_force_factor,
            "full_denominator_dimensionless": full_denominator,
            "full_denominator_log": log_denominator,
            "full_denominator_scaled_sum": scaled_denominator,
            "full_denominator_log_scale_offset": maximum_dimensionless_span,
            "turnover_frequency_per_second": turnover_frequency,
            "log_turnover_frequency_per_second": log_tof,
            "formula": (
                "TOF=(k_B*T/h)*(1-exp(delta_G_r/(R*T)))/"
                "sum_i,j(exp(delta_E_i,j/(R*T)))"
            ),
            "span_definition": (
                "G_TS-G_I when TS follows I in the supplied cycle; "
                "G_TS-G_I+delta_G_r otherwise"
            ),
            "full_denominator_used": True,
            "state_types_or_order_inferred": False,
        },
        backend_version="internal-1",
        provenance={
            "automatic_scientific_defaults": False,
            "automatic_fallback_count": 0,
            "cycle_order_basis": "caller_supplied_cycle_states_order",
            "state_semantics_inferred": False,
            "numerical_evaluation": "log_sum_exp",
            "model": "Kozuch-Shaik energetic span",
        },
    )


def execute(action_id: str, backend_id: str, request: dict[str, Any]) -> dict[str, Any]:
    if backend_id == "gaussian_codegen" and action_id == "generate_gaussian_code":
        from .gaussian_codegen import execute as execute_gaussian_codegen

        return execute_gaussian_codegen(action_id, backend_id, request)
    if backend_id == "geometric" and action_id == "optimize_geometry":
        return _geometric_optimize(request)
    if backend_id == "rdkit_mmff" and action_id == "optimize_geometry":
        from .rdkit_mmff import execute as execute_rdkit_mmff

        return execute_rdkit_mmff(action_id, backend_id, request)
    if backend_id == "fairchem_uma" and action_id == "optimize_geometry":
        from .fairchem import execute as execute_fairchem

        return execute_fairchem(action_id, backend_id, request)
    if backend_id == "sella" and action_id == "optimize_geometry":
        return execute_sella(action_id, request)
    if backend_id == "xtb":
        return _xtb(action_id, request)
    if backend_id == "multiwfn":
        if action_id == "analyze_non_covalent_interactions":
            return _multiwfn_nci_rdg(request)
        if action_id == "calculate_electron_isodensity_surface":
            return _multiwfn_isodensity_surface(request)
        return _multiwfn_wavefunction_analysis(action_id, request)
    if backend_id == "gaussian":
        return _gaussian(action_id, request)
    if backend_id == "internal_vibrations" and action_id == "derive_vibrational_modes":
        return _vibrations(request)
    if backend_id == "internal_spectroscopy" and action_id in {
        "derive_ir_spectrum",
        "derive_uv_vis_spectrum",
        "derive_raman_spectrum",
        "derive_roa_spectrum",
        "derive_vcd_spectrum",
        "derive_ecd_spectrum",
        "derive_nmr_spectrum",
    }:
        from .spectroscopy import execute as execute_spectroscopy

        return execute_spectroscopy(action_id, backend_id, request)
    if backend_id == "internal_thermochemistry" and action_id == "derive_thermochemistry":
        return _internal_thermochemistry(request)
    if backend_id == "internal_thermochemistry" and action_id == "analyze_free_energy_selectivity":
        return _free_energy_selectivity(request)
    if backend_id == "internal_thermochemistry" and action_id == "analyze_energetic_span":
        return _energetic_span(request)
    if backend_id == "goodvibes":
        return execute_goodvibes(action_id, request)
    return unsupported(f"Unsupported electronic action/backend combination: {action_id}/{backend_id}")
