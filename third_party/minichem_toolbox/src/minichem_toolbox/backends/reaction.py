"""Reaction-path, equilibrium, and kinetics actions."""

from __future__ import annotations

import math
import pprint
import re
import shutil
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .composite import execute_sella
from .goodvibes import execute as execute_goodvibes
from .gaussian_irc import parse_gaussian_irc_output as _parse_gaussian_irc_text
from .common import (
    atoms_and_coordinates,
    command_artifacts,
    module_version,
    output_directory,
    partial_success,
    relative_workspace_path,
    request_parts,
    resolve_input_file,
    run_external,
    structure_dict,
    success,
    unavailable,
    unwrap_artifact,
    unsupported,
    write_json,
    write_xyz,
)


ACTIONS = {
    "map_reaction_atoms", "analyze_reaction_changes",
    "materialize_mapped_hydrogen_transfers",
    "apply_reaction_changes_to_structure",
    "locate_transition_state", "search_reaction_path", "scan_reaction_coordinates",
    "validate_reaction_path", "analyze_reaction_coordinate",
    "extract_reaction_path_image",
    "parse_gaussian_irc_output",
    "trace_intrinsic_reaction_coordinate",
    "calculate_chemical_equilibrium", "integrate_reaction_network",
    "calculate_rate_constants", "calculate_tunneling_correction",
    "search_transition_state_path",
    "solve_microkinetic_model", "solve_master_equation",
    "analyze_thermochemical_selectivity", "analyze_free_energy_selectivity",
    "analyze_energetic_span",
    "analyze_reaction_free_energy_profile",
}


def _pysisyphus_xtb_gfn(method_name: str) -> int | str:
    """Map conventional xTB method spellings to pysisyphus' native GFN value."""

    normalized = re.sub(r"[^a-z0-9]+", "", method_name.casefold())
    values: dict[str, int | str] = {
        "gfn0": 0,
        "gfn0xtb": 0,
        "xtbgfn0": 0,
        "gfn1": 1,
        "gfn1xtb": 1,
        "xtbgfn1": 1,
        "gfn2": 2,
        "gfn2xtb": 2,
        "xtbgfn2": 2,
        "gfnff": "ff",
        "gfnffxtb": "ff",
        "xtbgfnff": "ff",
        "0": 0,
        "1": 1,
        "2": 2,
        "ff": "ff",
    }
    if normalized not in values:
        raise ValueError(
            "pysisyphus/XTB method must identify GFN0-xTB, GFN1-xTB, GFN2-xTB, "
            "or GFN-FF (compact forms gfn0, gfn1, gfn2, and gfnff are also accepted)"
        )
    return values[normalized]


_XTB_ALPB_SOLVENTS = {
    "acetone", "acetonitrile", "aniline", "benzaldehyde", "benzene",
    "ch2cl2", "chcl3", "cs2", "dioxane", "dmf", "dmso", "ether",
    "ethylacetate", "furane", "hexandecane", "hexane", "methanol",
    "nitromethane", "octanol", "woctanol", "phenol", "toluene", "thf",
    "water",
}
_XTB_GBSA_COMMON_SOLVENTS = {
    "acetone", "acetonitrile", "ch2cl2", "chcl3", "cs2", "dmso", "ether",
    "h2o", "methanol", "thf", "toluene",
}


def _validated_xtb_solvent(model: str, solvent: str, gfn: int | str) -> str:
    """Validate xTB 6.7's model- and GFN-specific built-in solvents."""

    normalized = solvent.strip().casefold()
    normalized = {
        "dichloromethane": "ch2cl2",
        "chloroform": "chcl3",
        "carbon disulfide": "cs2",
        "dimethylformamide": "dmf",
        "dimethyl sulfoxide": "dmso",
        "tetrahydrofuran": "thf",
    }.get(normalized, normalized)
    if model == "alpb":
        normalized = "water" if normalized == "h2o" else normalized
        choices = _XTB_ALPB_SOLVENTS
    else:
        normalized = "h2o" if normalized == "water" else normalized
        normalized = "n-hexane" if normalized == "hexane" else normalized
        choices = set(_XTB_GBSA_COMMON_SOLVENTS)
        if gfn == 1:
            choices.add("benzene")
        if gfn == 2:
            choices.update({"dmf", "n-hexane"})
    if normalized not in choices:
        raise ValueError(
            f"xTB {model.upper()} solvent {solvent!r} is not parametrized for GFN{gfn}; "
            f"choose one of {sorted(choices)} or omit implicit solvation"
        )
    return normalized


def _pysisyphus_failure_detail(directory: Path, stderr: str) -> str:
    """Summarize native calculator failures without hiding scientific nonconvergence."""

    scc_markers = sorted(directory.rglob(".sccnotconverged"))
    if scc_markers:
        marker = scc_markers[0].relative_to(directory)
        return (
            "the Agent-selected xTB calculator did not converge its SCC for the supplied "
            f"geometry (native marker: {marker}). This is a numerical calculation failure, "
            "not an automatic backend-selection event; inspect the saved geometry/logs and "
            "explicitly decide the next scientific step."
        )
    crashed_outputs = sorted(directory.glob("crashed_calculator_*/xtb.out"))
    if crashed_outputs:
        diagnostic_phrases = (
            "error",
            "failed",
            "abnormal",
            "converg",
            "atoms in the start geometry",
            "very close",
            "too close",
            "short distance",
            "refuses",
        )
        lines = [
            line.strip()
            for line in crashed_outputs[-1].read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()[-80:]
            if line.strip()
            and any(token in line.casefold() for token in diagnostic_phrases)
        ]
        if lines:
            return "xTB calculator failure: " + " | ".join(dict.fromkeys(lines))[-1600:]
    return stderr[-2000:] or "native pysisyphus process exited without a diagnostic message"


def _pysisyphus_final_path_energies(
    directory: Path, expected_image_count: int
) -> list[float] | None:
    """Recover the final path energy vector from the optimizer archive.

    pysisyphus 1.0.0 can omit internal-image energies from the comments in
    ``final_geometries.trj`` while retaining the complete vector in
    ``optimization.h5``. Read only the final optimizer cycle and fail closed
    on a malformed archive, unavailable optional dependency, wrong image
    count, or non-finite value.
    """

    archive_path = directory / "optimization.h5"
    if expected_image_count < 1 or not archive_path.is_file():
        return None
    try:
        import h5py

        with h5py.File(archive_path, "r") as archive:
            dataset = archive.get("opt/energies")
            if dataset is None or dataset.ndim != 2 or dataset.shape[0] < 1:
                return None
            values = [float(value) for value in dataset[-1]]
    except (ImportError, OSError, TypeError, ValueError):
        return None
    if len(values) != expected_image_count or not all(
        math.isfinite(value) for value in values
    ):
        return None
    return values


def _pysisyphus_calculator(
    method: dict[str, Any],
    input_structure: dict[str, Any],
    resource_limits: dict[str, Any],
) -> tuple[dict[str, Any], int, int]:
    calculator_backend = str(method["calculator_backend"]).lower()
    if calculator_backend not in {"xtb", "pyscf", "orca"}:
        raise ValueError("pysisyphus calculator_backend must be xtb, pyscf, or orca")
    method_name = str(method["method"]).strip()
    charge = int(method.get("charge", input_structure.get("charge", 0)))
    multiplicity = int(
        method.get("multiplicity", input_structure.get("multiplicity", 1))
    )
    calculator: dict[str, Any] = {
        "type": calculator_backend,
        "charge": charge,
        "mult": multiplicity,
        "pal": int(resource_limits.get("cpu_cores") or 1),
    }
    if calculator_backend == "xtb":
        calculator["gfn"] = _pysisyphus_xtb_gfn(method_name)
    elif calculator_backend == "pyscf":
        if not method.get("basis"):
            raise ValueError("pysisyphus/PySCF requires method_spec.basis")
        normalized = method_name.lower().replace("-", "").replace("_", "")
        if normalized in {"scf", "hf", "rhf", "uhf"}:
            calculator["method"] = "scf"
        elif normalized in {"mp2", "ump2"}:
            calculator["method"] = "mp2"
        else:
            calculator["method"] = "dft"
            calculator["xc"] = str(method.get("functional") or method_name)
        calculator["basis"] = str(method["basis"])
        if normalized in {"uhf", "uks", "ump2"}:
            calculator["unrestricted"] = True
    else:
        keywords = [method_name]
        if method.get("basis"):
            keywords.append(str(method["basis"]))
        calculator["keywords"] = " ".join(keywords)

    solvation_model = str(method.get("solvation_model") or "").strip().lower()
    if solvation_model:
        solvent = str(method.get("solvent") or "").strip()
        if not solvent:
            raise ValueError("pysisyphus solvation_model requires method_spec.solvent")
        if calculator_backend == "xtb" and solvation_model in {"alpb", "gbsa"}:
            calculator[solvation_model] = _validated_xtb_solvent(
                solvation_model, solvent, calculator["gfn"]
            )
        elif calculator_backend == "orca" and solvation_model == "cpcm":
            calculator["keywords"] += f" CPCM({solvent})"
        elif calculator_backend == "orca" and solvation_model == "smd":
            calculator["keywords"] += " CPCM"
            calculator["blocks"] = (
                f'%cpcm\n  smd true\n  SMDsolvent "{solvent}"\nend'
            )
        else:
            raise ValueError(
                f"solvation_model={solvation_model!r} is not supported with "
                f"calculator_backend={calculator_backend!r}"
            )
    return calculator, charge, multiplicity


def _assert_matching_endpoint_atoms(
    reactant: dict[str, Any], product: dict[str, Any]
) -> None:
    reactant_elements = [atom["element"] for atom in reactant.get("atoms") or []]
    product_elements = [atom["element"] for atom in product.get("atoms") or []]
    if reactant_elements != product_elements:
        raise ValueError(
            "Double-ended path endpoints must have identical atom counts, elements, and ordering"
        )
    if int(reactant.get("charge", 0)) != int(product.get("charge", 0)):
        raise ValueError("Double-ended path endpoints must have the same charge")
    if int(reactant.get("multiplicity", 1)) != int(product.get("multiplicity", 1)):
        raise ValueError("Double-ended path endpoints must have the same multiplicity")


def _copy_atom_identity(
    reference: dict[str, Any], structure: dict[str, Any]
) -> dict[str, Any]:
    """Copy validated atom-map identity through formats such as XYZ that cannot store it."""

    reference_atoms = reference.get("atoms") or []
    atoms = structure.get("atoms") or []
    if [atom.get("element") for atom in atoms] != [
        atom.get("element") for atom in reference_atoms
    ]:
        raise ValueError("Output structure changed atom count, elements, or ordering")
    map_sequence = _structure_atom_map_sequence(reference)
    if map_sequence is not None:
        for atom, map_number in zip(atoms, map_sequence):
            atom["atom_map_number"] = map_number
        structure["atom_map_numbers"] = list(map_sequence)
    return structure


def _parse_multixyz(
    path: Path,
    *,
    charge: int,
    multiplicity: int,
    reference_structure: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    frames: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(lines):
        if not lines[cursor].strip():
            cursor += 1
            continue
        try:
            atom_count = int(lines[cursor].strip())
        except ValueError as exc:
            raise ValueError(f"Invalid multi-XYZ atom count in {path}: {lines[cursor]!r}") from exc
        if cursor + atom_count + 1 >= len(lines):
            raise ValueError(f"Truncated multi-XYZ frame in {path}")
        comment = lines[cursor + 1]
        atoms = []
        for line in lines[cursor + 2 : cursor + 2 + atom_count]:
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"Invalid multi-XYZ atom line in {path}: {line!r}")
            atoms.append(
                {
                    "element": parts[0],
                    "position_angstrom": [float(parts[1]), float(parts[2]), float(parts[3])],
                }
            )
        frame: dict[str, Any] = {
            "atoms": atoms,
            "charge": charge,
            "multiplicity": multiplicity,
            "comment": comment,
        }
        if reference_structure is not None:
            _copy_atom_identity(reference_structure, frame)
        energy_match = re.search(
            r"\b(?:energy|E)\b\s*[=:]\s*(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)",
            comment,
            re.IGNORECASE,
        )
        if energy_match is None:
            energy_match = re.match(
                r"\s*(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s*(?:,|$)", comment
            )
        if energy_match:
            frame["energy_hartree"] = float(energy_match.group(1))
        frames.append(frame)
        cursor += atom_count + 2
    return frames


def _pysisyphus_freezing_string(
    *,
    directory: Path,
    endpoint_paths: list[Path],
    calculator: dict[str, Any],
    charge: int,
    multiplicity: int,
    settings: dict[str, Any],
    configuration: dict[str, Any],
    reference_structure: dict[str, Any],
) -> dict[str, Any]:
    """Drive pysisyphus FreezingString directly around upstream CLI incompatibilities."""

    import numpy as np
    from pysisyphus.cos.FreezingString import FreezingString
    from pysisyphus.run import get_calc_closure
    from pysisyphus.trj import get_geoms

    calculator_kwargs = dict(calculator)
    calculator_key = str(calculator_kwargs.pop("type"))
    calculator_kwargs["out_dir"] = directory / "qm_calcs"
    calculator_getter = get_calc_closure(
        "freezing_string_image", calculator_key, calculator_kwargs
    )
    endpoints = get_geoms([str(path) for path in endpoint_paths], quiet=True)
    for endpoint in endpoints:
        endpoint.set_calculator(calculator_getter())
    internal_images = int(settings["images"]) - 2
    string = FreezingString(
        endpoints,
        calculator_getter,
        max_nodes=internal_images,
        opt_steps=3,
    )
    max_cycles = int(settings["max_cycles"])
    if max_cycles < internal_images:
        raise ValueError(
            "freezing_string max_cycles is too small to grow the requested number of images"
        )
    force_thresholds = {
        "nwchem_loose": 5.0e-3,
        "gau_loose": 2.5e-3,
        "gau": 4.5e-4,
        "gau_tight": 1.5e-5,
        "gau_vtight": 2.0e-6,
        "baker": 3.0e-4,
        "never": 0.0,
    }
    threshold = force_thresholds[str(settings["convergence"])]
    cycles_after_growth = 0
    history = []
    final_max_force = math.inf
    for cycle in range(max_cycles):
        forces = np.asarray(string.forces, dtype=float)
        final_max_force = float(np.max(np.abs(forces)))
        step = 0.20 * forces
        max_component = float(np.max(np.abs(step)))
        if max_component > 0.08:
            step *= 0.08 / max_component
        string.coords = np.asarray(string.coords, dtype=float) + step
        was_fully_grown = string.fully_grown
        string.reparametrize(string.energy, forces)
        if string.fully_grown:
            cycles_after_growth = cycles_after_growth + 1 if was_fully_grown else 0
        history.append(
            {
                "cycle": cycle,
                "image_count": len(string.left_string) + len(string.right_string),
                "maximum_frontier_force_hartree_per_bohr": final_max_force,
                "fully_grown": bool(string.fully_grown),
            }
        )
        if string.fully_grown and cycles_after_growth >= 3:
            break

    images = string.left_string + string.right_string
    energies = [float(image.energy) for image in images]
    trajectory_path = directory / "freezing_string_final.trj"
    trajectory_path.write_text(
        "\n".join(image.as_xyz() for image in images), encoding="utf-8"
    )
    frames = _parse_multixyz(
        trajectory_path,
        charge=charge,
        multiplicity=multiplicity,
        reference_structure=reference_structure,
    )
    image_files = []
    for index, frame in enumerate(frames):
        image_files.append(
            relative_workspace_path(
                write_xyz(frame, directory / f"path_image_{index:03d}.xyz")
            )
        )
    highest = max(range(len(energies)), key=energies.__getitem__)
    highest_interior = max(range(1, len(energies) - 1), key=energies.__getitem__)
    fully_grown = bool(string.fully_grown)
    force_converged = threshold > 0 and final_max_force <= threshold
    result = {
        "path_file": relative_workspace_path(trajectory_path),
        "image_files": image_files,
        "image_count": len(image_files),
        "energies_hartree": energies,
        "highest_energy_image_index": highest,
        "highest_interior_energy_image_index": highest_interior,
        "highest_energy_image": frames[highest],
        "highest_interior_energy_image": frames[highest_interior],
        "converged": fully_grown,
        "fully_grown": fully_grown,
        "frontier_force_converged": force_converged,
        "final_maximum_frontier_force_hartree_per_bohr": final_max_force,
        "endpoints_consumed": True,
        "transition_state_validated": False,
    }
    history_path = write_json(directory, "freezing_string_history.json", history)
    warnings = []
    if not fully_grown:
        warnings.append("Freezing String did not grow all requested images before max_cycles.")
    if not force_converged:
        warnings.append(
            "Freezing String grew the requested path, but the last frontier force did not "
            "meet the selected threshold; refine the highest image before TS validation."
        )
    provenance = {
        "command": ["python:pysisyphus.FreezingString"],
        "generated_config": configuration,
        "upstream_cli_workaround": (
            "pysisyphus 1.0.0 CLI passes ChainOfStates-only keywords to FreezingString; "
            "the adapter invokes the installed class directly without changing its calculator"
        ),
        "resolved_molecular_state": {"charge": charge, "multiplicity": multiplicity},
    }
    artifacts = command_artifacts(directory)
    complete = fully_grown and len(image_files) == int(settings["images"])
    if complete:
        return success(
            result,
            artifact_files=artifacts,
            backend_version=module_version("pysisyphus"),
            provenance=provenance,
            warnings=warnings,
        )
    return partial_success(
        result,
        artifact_files=artifacts,
        backend_version=module_version("pysisyphus"),
        provenance=provenance,
        warnings=warnings or [f"Incomplete Freezing String history: {history_path}"],
    )


def _pysisyphus(action_id: str, request: dict[str, Any]) -> dict[str, Any]:
    inputs, method, settings = request_parts(request)
    directory = output_directory(action_id, "pysisyphus")
    resource_limits = dict(request.get("resource_limits") or {})
    staged_inputs: list[Path] = []
    if action_id == "search_reaction_path":
        reactant = structure_dict(inputs["reactant"])
        product = structure_dict(inputs["product"])
        _assert_matching_endpoint_atoms(reactant, product)
        reactant_xyz = write_xyz(reactant, directory / "reactant.xyz")
        product_xyz = write_xyz(product, directory / "product.xyz")
        staged_inputs = [reactant_xyz, product_xyz]
        input_structure = reactant
        geom = {"type": "cart", "fn": [str(reactant_xyz), str(product_xyz)]}
    elif action_id == "scan_reaction_coordinates":
        input_structure = structure_dict(inputs["structure"])
        xyz = write_xyz(input_structure, directory / "input.xyz")
        staged_inputs = [xyz]
        geom = {"type": "redund", "fn": str(xyz)}
    else:
        structure_key = (
            "initial_guess" if action_id == "locate_transition_state" else "transition_state"
        )
        input_structure = structure_dict(inputs[structure_key])
        xyz = write_xyz(input_structure, directory / "input.xyz")
        staged_inputs = [xyz]
        geom = {"type": "cart", "fn": str(xyz)}
    calculator, charge, multiplicity = _pysisyphus_calculator(
        method, input_structure, resource_limits
    )
    configuration: dict[str, Any] = {
        "geom": geom,
        "calc": calculator,
    }
    if action_id == "locate_transition_state":
        configuration["tsopt"] = {
            "type": str(settings.get("optimizer", "rsirfo")),
            "thresh": str(settings.get("convergence", "gau")),
            "max_cycles": int(settings.get("max_cycles", 200)),
            "hessian_init": str(settings["hessian_init"]),
        }
    elif action_id == "trace_intrinsic_reaction_coordinate":
        configuration["irc"] = {
            "type": str(settings.get("integrator", "eulerpc")),
            "step_length": float(settings.get("step_length", 0.1)),
            "max_cycles": int(settings.get("max_cycles", 150)),
            "forward": bool(settings.get("forward", True)),
            "backward": bool(settings.get("backward", True)),
            "hessian_init": str(settings["hessian_init"]),
        }
    elif action_id == "search_reaction_path":
        images = int(settings["images"])
        if images < 3:
            raise ValueError("search_reaction_path requires at least three total images")
        path_method = str(settings["path_method"]).strip().lower()
        pysis_path_method = {
            "neb": "neb",
            "growing_string": "gs",
            "freezing_string": "fs",
        }[path_method]
        optimizer = str(settings["optimizer"]).lower()
        if pysis_path_method == "gs" and optimizer != "string":
            raise ValueError(
                "growing_string requires optimizer=string because generic optimizers cannot "
                "safely resize their history when pysisyphus adds nodes"
            )
        if pysis_path_method == "fs" and optimizer != "sd":
            raise ValueError(
                "freezing_string requires optimizer=sd in the direct installed-class driver"
            )
        cos: dict[str, Any] = {"type": pysis_path_method}
        if pysis_path_method != "fs":
            cos.update({"fix_first": True, "fix_last": True})
        if pysis_path_method == "neb":
            configuration["interpol"] = {
                "type": str(settings["interpolation"]),
                "between": images - 2,
                "align": True,
            }
            cos["climb"] = bool(settings["climb"])
        else:
            if pysis_path_method == "fs" and (images - 2) % 2:
                raise ValueError(
                    "freezing_string requires an even number of internal images "
                    "(images - 2 must be even)"
                )
            cos["max_nodes"] = images - 2
            if pysis_path_method == "gs":
                cos["climb"] = bool(settings["climb"])
        configuration["cos"] = cos
        configuration["opt"] = {
            "type": optimizer,
            "thresh": str(settings["convergence"]),
            "max_cycles": int(settings["max_cycles"]),
            "dump": True,
        }
    elif action_id == "scan_reaction_coordinates":
        coordinate_type = str(settings["coordinate_type"]).strip().lower()
        primitive, expected_indices = {
            "bond": ("BOND", 2),
            "angle": ("BEND", 3),
            "dihedral": ("PROPER_DIHEDRAL", 4),
        }[coordinate_type]
        indices = [int(value) for value in settings["atom_indices"]]
        if len(indices) != expected_indices or len(indices) != len(set(indices)):
            raise ValueError(
                f"coordinate_type={coordinate_type} requires {expected_indices} unique atom_indices"
            )
        atom_count = len(input_structure.get("atoms") or [])
        if any(index < 0 or index >= atom_count for index in indices):
            raise ValueError("atom_indices contains an index outside the supplied structure")
        # A reaction-coordinate bond scan commonly starts from two atoms that
        # are not bonded yet. pysisyphus only auto-defines geometrically
        # perceived primitives; when start_value is explicit its scan driver
        # does not add a missing primitive itself. Register the requested
        # primitive up front so forming-bond scans are valid too.
        configuration["geom"]["coord_kwargs"] = {
            "define_prims": [[primitive, *indices]]
        }
        value_unit = str(settings["value_unit"]).strip().lower()
        start = float(settings["start_value"])
        end = float(settings["end_value"])
        if coordinate_type == "bond":
            if value_unit != "angstrom":
                raise ValueError("bond scans require value_unit=angstrom")
            start *= 1.8897261254578281
            end *= 1.8897261254578281
        else:
            if value_unit == "degree":
                start = math.radians(start)
                end = math.radians(end)
            elif value_unit != "radian":
                raise ValueError("angle/dihedral scans require value_unit=degree or radian")
        steps = int(settings["steps"])
        if steps < 1:
            raise ValueError("steps must be positive")
        opt: dict[str, Any] = {
            "type": str(settings["optimizer"]),
            "thresh": str(settings["convergence"]),
            "max_cycles": int(settings["max_cycles"]),
            "dump": True,
        }
        if str(settings["optimizer"]).lower() == "rfo":
            opt["hessian_init"] = str(settings["hessian_init"])
            # pysisyphus 1.0.0 保留内坐标重建前后的力历史；扫描中的冗余内坐标
            # 数量变化后，RFO 的 GDIIS 会把不同形状的历史合并为数组并崩溃。
            # 关闭这条仅用于加速的插值路径，仍保留 Hessian 优化和绝对扫描端点。
            opt["gdiis"] = False
        configuration["scan"] = {
            "type": primitive,
            "indices": indices,
            "start": start,
            "end": end,
            "steps": steps,
            "symmetric": False,
            "opt": opt,
        }
    input_path = directory / "pysis.yaml"
    input_path.write_text(yaml.safe_dump(configuration, sort_keys=False), encoding="utf-8")
    if (
        action_id == "search_reaction_path"
        and str(settings["path_method"]).strip().lower() == "freezing_string"
    ):
        return _pysisyphus_freezing_string(
            directory=directory,
            endpoint_paths=staged_inputs,
            calculator=calculator,
            charge=charge,
            multiplicity=multiplicity,
            settings=settings,
            configuration=configuration,
            reference_structure=input_structure,
        )
    completed = run_external(
        executable="pysis",
        environment_variable="CHEMGRAPH_PYSIS_COMMAND",
        arguments=[str(input_path)],
        directory=directory,
        timeout_seconds=int(request.get("resource_limits", {}).get("walltime_seconds", 1800)),
    )
    (directory / "stdout.log").write_text(completed["stdout"], encoding="utf-8")
    (directory / "stderr.log").write_text(completed["stderr"], encoding="utf-8")
    if not completed["available"]:
        return unavailable(completed["stderr"], install="pip install pysisyphus==1.0.0")
    if completed["returncode"] != 0:
        raise RuntimeError(
            f"pysisyphus failed: {_pysisyphus_failure_detail(directory, completed['stderr'])}"
        )
    xyz_outputs = [path for path in directory.rglob("*.xyz") if path not in staged_inputs]
    native_converged = bool(
        re.search(r"(?m)^\s*Converged!\s*$", completed["stdout"])
    ) and "Number of cycles exceeded!" not in completed["stdout"]
    partial_warnings: list[str] = []
    if action_id == "locate_transition_state":
        candidate = xyz_outputs[-1] if xyz_outputs else None
        result_structure = (
            structure_dict(relative_workspace_path(candidate)) if candidate else None
        )
        if result_structure is not None:
            _copy_atom_identity(input_structure, result_structure)
            # Native XYZ writers rarely retain molecular electronic-state metadata.
            # These values are the exact Agent-selected state used by the calculator.
            result_structure["charge"] = charge
            result_structure["multiplicity"] = multiplicity
        result = {
            "structure": result_structure,
            "converged": bool(candidate) and native_converged,
            "validation_required": "Call calculate_hessian and derive_vibrational_modes explicitly.",
        }
        if candidate is None:
            partial_warnings.append(
                "pysisyphus completed without a parseable transition-state geometry."
            )
        elif not native_converged:
            partial_warnings.append(
                "pysisyphus wrote a last geometry but did not satisfy its optimizer "
                "convergence criteria; the structure is retained only as an unvalidated "
                "search endpoint and must not be described as a converged transition state."
            )
    elif action_id == "trace_intrinsic_reaction_coordinate":
        structures = []
        for path in xyz_outputs:
            structure = structure_dict(relative_workspace_path(path))
            structures.append(_copy_atom_identity(input_structure, structure))
        result = {
            "path_files": [relative_workspace_path(path) for path in xyz_outputs],
            "structures": structures,
            "forward": bool(settings.get("forward", True)),
            "backward": bool(settings.get("backward", True)),
        }
    elif action_id == "search_reaction_path":
        path_candidates = [
            directory / "final_geometries.trj",
            directory / "current_geometries.trj",
            directory / "interpolated.trj",
        ]
        path_file = next((path for path in path_candidates if path.is_file()), None)
        frames = (
            _parse_multixyz(
                path_file,
                charge=charge,
                multiplicity=multiplicity,
                reference_structure=input_structure,
            )
            if path_file
            else []
        )
        image_files = []
        for index, frame in enumerate(frames):
            image_path = write_xyz(frame, directory / f"path_image_{index:03d}.xyz")
            image_files.append(relative_workspace_path(image_path))
        energies = [frame.get("energy_hartree") for frame in frames]
        numeric_energies = [value for value in energies if value is not None]
        energy_source = "trajectory_comments"
        if frames and len(numeric_energies) != len(frames):
            archived_energies = _pysisyphus_final_path_energies(
                directory, len(frames)
            )
            if archived_energies is not None:
                energies = archived_energies
                numeric_energies = archived_energies
                energy_source = "optimization_h5_final_cycle"
        highest_index = (
            max(range(len(energies)), key=lambda index: float(energies[index]))
            if energies and len(numeric_energies) == len(energies)
            else None
        )
        highest_interior_index = (
            max(range(1, len(energies) - 1), key=lambda index: float(energies[index]))
            if len(energies) >= 3 and len(numeric_energies) == len(energies)
            else None
        )
        result = {
            "path_file": relative_workspace_path(path_file) if path_file else None,
            "image_files": image_files,
            "image_count": len(image_files),
            "energies_hartree": energies if numeric_energies else None,
            "energy_source": energy_source if numeric_energies else None,
            "highest_energy_image_index": highest_index,
            "highest_interior_energy_image_index": highest_interior_index,
            "highest_energy_image": (
                frames[highest_index] if highest_index is not None else None
            ),
            "highest_interior_energy_image": (
                frames[highest_interior_index]
                if highest_interior_index is not None
                else None
            ),
            "converged": native_converged,
            "endpoints_consumed": True,
            "transition_state_validated": False,
        }
        if not image_files:
            partial_warnings.append(
                "pysisyphus completed without a parseable final/current chain-of-states trajectory."
            )
        elif not native_converged:
            partial_warnings.append(
                "The saved path did not satisfy the selected chain-of-states convergence criteria."
            )
    else:
        data_path = directory / "relaxed_scan.dat"
        trajectory_path = directory / "relaxed_scan.trj"
        values: list[float] = []
        energies: list[float] = []
        if data_path.is_file():
            for line in data_path.read_text(encoding="utf-8", errors="replace").splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    values.append(float(parts[0]))
                    energies.append(float(parts[1]))
        frames = (
            _parse_multixyz(
                trajectory_path,
                charge=charge,
                multiplicity=multiplicity,
                reference_structure=input_structure,
            )
            if trajectory_path.is_file()
            else []
        )
        if str(settings["coordinate_type"]).lower() == "bond":
            reported_values = [value / 1.8897261254578281 for value in values]
            reported_unit = "angstrom"
        elif str(settings["value_unit"]).lower() == "degree":
            reported_values = [math.degrees(value) for value in values]
            reported_unit = "degree"
        else:
            reported_values = values
            reported_unit = "radian"
        result = {
            "coordinate_type": str(settings["coordinate_type"]).lower(),
            "atom_indices": [int(value) for value in settings["atom_indices"]],
            "coordinate_values": reported_values,
            "coordinate_unit": reported_unit,
            "energies_hartree": energies,
            "structures": frames,
            "trajectory_file": (
                relative_workspace_path(trajectory_path) if trajectory_path.is_file() else None
            ),
            "data_file": relative_workspace_path(data_path) if data_path.is_file() else None,
            "completed_points": len(values),
            "requested_points": int(settings["steps"]) + 1,
        }
        native_converged = (
            len(values) == int(settings["steps"]) + 1
            and "did not converge. Breaking!" not in completed["stdout"]
        )
        if not native_converged:
            partial_warnings.append(
                "The relaxed scan did not complete every requested point with native convergence."
            )
    artifacts = command_artifacts(directory)
    provenance = {
        "command": completed["command"],
        "generated_config": configuration,
        "resolved_molecular_state": {
            "charge": charge,
            "multiplicity": multiplicity,
            "charge_source": "method_spec" if "charge" in method else "input_structure",
            "multiplicity_source": (
                "method_spec" if "multiplicity" in method else "input_structure"
            ),
        },
    }
    if action_id in {"locate_transition_state", "search_reaction_path", "scan_reaction_coordinates"}:
        provenance["native_optimizer_convergence"] = {
            "converged": native_converged,
            "positive_marker": "Converged!" if native_converged else None,
            "cycle_limit_exceeded": "Number of cycles exceeded!" in completed["stdout"],
        }
    complete = (
        action_id == "locate_transition_state"
        and result.get("structure") is not None
        and result.get("converged") is True
    ) or (
        action_id == "trace_intrinsic_reaction_coordinate" and bool(result.get("path_files"))
    ) or (
        action_id == "search_reaction_path"
        and bool(result.get("image_files"))
        and result.get("converged") is True
    ) or (
        action_id == "scan_reaction_coordinates" and native_converged
    )
    if not complete:
        return partial_success(
            result,
            artifact_files=artifacts,
            backend_version=module_version("pysisyphus"),
            provenance=provenance,
            warnings=partial_warnings
            or ["pysisyphus completed without a parseable primary structure/path result."],
        )
    return success(
        result,
        artifact_files=artifacts,
        backend_version=module_version("pysisyphus"),
        provenance=provenance,
    )


def _path_frames(
    value: Any, *, reference_structure: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    item = unwrap_artifact(value)
    if isinstance(item, dict) and isinstance(item.get("result"), dict):
        item = item["result"]
    candidates: Any = item
    if isinstance(item, dict):
        for key in ("structures", "images", "image_files", "path_files"):
            if isinstance(item.get(key), list) and item[key]:
                candidates = item[key]
                break
        else:
            for key in ("path_file", "trajectory_file"):
                if item.get(key):
                    path = resolve_input_file(item[key])
                    return _parse_multixyz(
                        path,
                        charge=int(item.get("charge", 0)),
                        multiplicity=int(item.get("multiplicity", 1)),
                        reference_structure=reference_structure,
                    )
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(
            "path must contain a non-empty structures/images/image_files/path_files list "
            "or one path_file/trajectory_file"
        )
    frames = []
    for value in candidates:
        try:
            frames.append(structure_dict(value))
        except (TypeError, ValueError):
            path = resolve_input_file(value)
            if path.suffix.lower() == ".trj":
                frames.extend(
                    _parse_multixyz(
                        path,
                        charge=0,
                        multiplicity=1,
                        reference_structure=reference_structure,
                    )
                )
            else:
                structure = structure_dict(relative_workspace_path(path))
                frames.append(
                    _copy_atom_identity(reference_structure, structure)
                    if reference_structure is not None
                    else structure
                )
    if not frames:
        raise ValueError("path contains no parseable structures")
    return frames


_GAUSSIAN_IRC_PROVENANCE_FIELDS = {
    "pair_id",
    "candidate_id",
    "case1_candidate_id",
    "re_si",
    "re/si",
    "stereochemical_face",
    "transition_state_label",
    "ts_label",
    "source_path",
}


def _strip_gaussian_irc_provenance_fields(
    structure: dict[str, Any],
) -> dict[str, Any]:
    """移除候选身份和工作区路径，只保留可用于路径校验的结构事实。"""

    result = deepcopy(structure)
    for field_name in _GAUSSIAN_IRC_PROVENANCE_FIELDS:
        result.pop(field_name, None)
    for atom in result.get("atoms") or []:
        if isinstance(atom, dict):
            for field_name in _GAUSSIAN_IRC_PROVENANCE_FIELDS:
                atom.pop(field_name, None)
    return result


def _parse_gaussian_irc_output(request: dict[str, Any]) -> dict[str, Any]:
    """解析已完成 Gaussian IRC stdout，并返回可交给路径校验的 ReactionPath。

    输入必须提供 GaussianIRCLog 文件或 ArtifactRef、显式 forward/backward 方向以及
    transition_state 结构。函数只读取和解析文件，不提交 Gaussian；异常终止、方向不匹配、
    无路径点和未到达 PES minimum 都保留机械事实，但不会返回成功的已验证路径。
    """

    inputs, _method, settings = request_parts(request)
    direction = str(settings.get("direction") or "").strip().casefold()
    if direction not in {"forward", "backward"}:
        return {
            "status": "invalid_request",
            "error": {
                "code": "invalid_direction",
                "message": "Gaussian IRC direction must be explicitly forward or backward",
            },
            "retryable": False,
        }
    try:
        log_path = resolve_input_file(inputs["gaussian_irc_log"])
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        transition_state = _strip_gaussian_irc_provenance_fields(
            structure_dict(inputs["transition_state"])
        )
        atoms_and_coordinates(transition_state)
        from . import quantum_legacy

        parsed = _parse_gaussian_irc_text(
            log_text,
            transition_state,
            direction,
            parse_geometries=quantum_legacy._parse_gaussian_geometries,
            parse_energy=quantum_legacy._parse_gaussian_energy,
            preserve_structure_metadata=quantum_legacy._preserve_gaussian_structure_metadata,
        )
    except (FileNotFoundError, OSError, UnicodeError, KeyError, TypeError, ValueError) as exc:
        return {
            "status": "failed",
            "result": {
                "direction": direction,
                "normal_termination": False,
                "branch_completed": False,
                "pes_minimum_detected": False,
                "completed_points": 0,
                "termination_reason": "input_parse_error",
                "transition_state_validated": False,
            },
            "error": {
                "code": "gaussian_irc_input_parse_failed",
                "message": f"Gaussian IRC input could not be parsed: {type(exc).__name__}",
            },
            "retryable": False,
        }

    points = list(parsed["points"])
    transition_state_energy = parsed.get("transition_state_energy_hartree")
    if transition_state_energy is None:
        transition_state_energy = transition_state.get("energy_hartree")
    structures = [transition_state, *[point["structure"] for point in points]]
    energies = [transition_state_energy, *[point["energy_hartree"] for point in points]]
    energy_values = (
        [float(value) for value in energies]
        if all(value is not None for value in energies)
        else None
    )
    result = {
        "direction": direction,
        "transition_state": transition_state,
        "structures": structures,
        "points": points,
        "summary": parsed["summary"],
        "energy_unit": "hartree",
        "energies_hartree": energy_values,
        "completed_points": len(points),
        "branch_completed": bool(parsed["branch_completed"]),
        "branch_completion": bool(parsed["branch_completed"]),
        "gaussian_branch_completed": bool(parsed["branch_completed"]),
        "pes_minimum_detected": bool(parsed["pes_minimum_detected"]),
        "pes_minimum": bool(parsed["pes_minimum_detected"]),
        "termination_reason": parsed["termination_reason"],
        "normal_termination": bool(parsed["normal_termination"]),
        "error_termination": bool(parsed["error_termination"]),
        "direction_marker_present": bool(parsed["direction_marker_present"]),
        "observed_native_directions": parsed["observed_native_directions"],
        "transition_state_validated": False,
        "path_validated": False,
        "validation_status": "not_validated",
    }
    failure_reason = str(result["termination_reason"])
    if failure_reason in {
        "abnormal_termination",
        "direction_mismatch",
        "no_parseable_points",
        "completion_marker_missing",
    }:
        messages = {
            "abnormal_termination": "Gaussian IRC log has abnormal or error termination.",
            "direction_mismatch": "Gaussian IRC log contains the opposite direction branch, not the requested direction.",
            "no_parseable_points": "Gaussian IRC log contains no parseable points for the requested direction.",
            "completion_marker_missing": "Gaussian IRC log has points but no branch completion marker.",
        }
        return {
            "status": "failed",
            "result": result,
            "error": {"code": failure_reason, "message": messages[failure_reason]},
            "retryable": False,
        }
    if not result["branch_completed"] or not result["pes_minimum_detected"]:
        return {
            "status": "partial_success",
            "result": result,
            "warnings": [
                "The Gaussian IRC branch is retained as a partial ReactionPath and is not a validated path.",
            ],
            "provenance": {"direction_is_atomic": True},
            "retryable": False,
        }
    return {
        "status": "success",
        "result": result,
        "provenance": {"direction_is_atomic": True},
        "retryable": False,
    }


def _extract_reaction_path_image(request: dict[str, Any]) -> dict[str, Any]:
    """按请求给出的索引提取一张路径图像并写出可交接的 XYZ 文件。

    输入必须包含 ``path_image_candidate``，其中的 ``path`` 是已有 ReactionPath
    或其 ArtifactRef，``image_index`` 是非负整数。函数只读取该索引对应的图像，
    不按能量排序、不推断过渡态，也不会修改原始路径。成功时返回选中的
    AtomicStructure、pair/candidate 身份和一个 XYZFile 文件 Artifact 候选；索引
    越界、路径为空或图像无法解析时直接失败。
    """

    inputs, _method, _settings = request_parts(request)
    candidate = inputs.get("path_image_candidate")
    if not isinstance(candidate, dict):
        raise ValueError("inputs.path_image_candidate must be an object")

    image_index = candidate.get("image_index")
    if isinstance(image_index, bool) or not isinstance(image_index, int):
        raise ValueError("path_image_candidate.image_index must be a non-negative integer")
    if image_index < 0:
        raise ValueError("path_image_candidate.image_index must be a non-negative integer")
    if "path" not in candidate:
        raise ValueError("path_image_candidate.path is required")

    frames = _path_frames(candidate["path"])
    if image_index >= len(frames):
        raise ValueError(
            "path_image_candidate.image_index is outside the supplied reaction path "
            f"(image_count={len(frames)})"
        )
    structure = deepcopy(frames[image_index])

    candidate_id = candidate.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        candidate_id = f"path_image_{image_index:04d}"
    pair_id = candidate.get("pair_id")
    if not isinstance(pair_id, str) or not pair_id.strip():
        pair_id = "unassigned_pair"

    directory = output_directory(
        "extract_reaction_path_image", "internal_reaction_analysis"
    )
    xyz_path = write_xyz(structure, directory / "selected_path_image.xyz")
    result = {
        "structure": structure,
        "candidate_id": candidate_id,
        "pair_id": pair_id,
        "image_index": image_index,
        "image_count": len(frames),
        "transition_state_validated": False,
    }
    return success(
        result,
        artifact_files=[
            {
                "path": relative_workspace_path(xyz_path),
                "semantic_type": "XYZFile",
                "media_type": "chemical/x-xyz",
            }
        ],
        provenance={
            "selected_image_index": image_index,
            "automatic_image_selection": False,
            "transition_state_validated": False,
            "candidate_id": candidate_id,
            "pair_id": pair_id,
        },
    )


def _coordinates(structure: dict[str, Any]):
    import numpy as np

    return np.asarray(
        [atom["position_angstrom"] for atom in structure.get("atoms") or []],
        dtype=float,
    )


def _aligned_rmsd(first: dict[str, Any], second: dict[str, Any]) -> float:
    import numpy as np

    first_elements = [atom["element"] for atom in first.get("atoms") or []]
    second_elements = [atom["element"] for atom in second.get("atoms") or []]
    if first_elements != second_elements or not first_elements:
        raise ValueError("RMSD comparison requires identical non-empty atom ordering")
    a = _coordinates(first)
    b = _coordinates(second)
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    u, _singular, vt = np.linalg.svd(a.T @ b)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u @ vt))
    rotated = a @ u @ correction @ vt
    return float(np.sqrt(np.mean(np.sum((rotated - b) ** 2, axis=1))))


def _structure_atom_map_indices(
    structure: dict[str, Any],
) -> dict[int, int] | None:
    atoms = structure.get("atoms")
    if not isinstance(atoms, list) or not atoms:
        raise ValueError("Mapped bond changes require a non-empty AtomicStructure.atoms")
    aligned = structure.get("atom_map_numbers")
    if aligned is not None and (
        not isinstance(aligned, list) or len(aligned) != len(atoms)
    ):
        raise ValueError("AtomicStructure.atom_map_numbers must align with atoms")
    values: list[int | None] = []
    for index, atom in enumerate(atoms):
        if not isinstance(atom, dict):
            raise ValueError(f"AtomicStructure.atoms[{index}] must be a mapping")
        raw_values = [
            atom[key]
            for key in ("atom_map_number", "map_number", "atom_map")
            if key in atom and atom[key] is not None
        ]
        if aligned is not None:
            raw_values.append(aligned[index])
        if not raw_values:
            values.append(None)
            continue
        parsed = []
        for raw in raw_values:
            if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
                raise ValueError(
                    f"AtomicStructure atom map at index {index} must be a positive integer"
                )
            parsed.append(raw)
        if len(set(parsed)) != 1:
            raise ValueError(
                f"AtomicStructure atom index {index} has conflicting atom-map fields"
            )
        values.append(parsed[0])
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError("AtomicStructure has partial atom mapping")
    mapped = [int(value) for value in values if value is not None]
    if len(set(mapped)) != len(mapped):
        raise ValueError("AtomicStructure atom-map numbers must be unique")
    return {map_number: index for index, map_number in enumerate(mapped)}


def _structure_atom_map_sequence(
    structure: dict[str, Any],
) -> tuple[int, ...] | None:
    map_to_index = _structure_atom_map_indices(structure)
    if map_to_index is None:
        return None
    sequence = [0] * len(map_to_index)
    for map_number, atom_index in map_to_index.items():
        sequence[atom_index] = map_number
    return tuple(sequence)


def _path_atom_identity_status(
    structures: list[dict[str, Any]],
) -> tuple[bool, bool]:
    sequences = [_structure_atom_map_sequence(structure) for structure in structures]
    mapped_sequences = [sequence for sequence in sequences if sequence is not None]
    if not mapped_sequences:
        return False, True
    if len(mapped_sequences) != len(sequences):
        return True, False
    reference = mapped_sequences[0]
    return True, all(sequence == reference for sequence in mapped_sequences[1:])


def _mapped_change_pair(
    record: Any, map_to_index: dict[int, int], *, context: str
) -> tuple[list[int], list[int]]:
    if not isinstance(record, dict):
        raise ValueError(f"{context} must be a mapped bond record")
    raw_pair = (record.get("atom_map_a"), record.get("atom_map_b"))
    maps = []
    for raw in raw_pair:
        if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
            raise ValueError(f"{context} requires positive integer atom-map numbers")
        maps.append(raw)
    if maps[0] == maps[1]:
        raise ValueError(f"{context} cannot reference the same atom twice")
    try:
        indices = [map_to_index[maps[0]], map_to_index[maps[1]]]
    except KeyError as exc:
        raise ValueError(
            f"{context} references atom map {exc.args[0]} absent from the path endpoint"
        ) from exc
    return indices, maps


def _bond_change_records(
    value: Any, *, reference_structure: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    item = unwrap_artifact(value) if value is not None else None
    if item is None:
        return []
    records: list[dict[str, Any]] = []
    if isinstance(item, dict):
        while isinstance(item.get("result"), dict):
            item = item["result"]
        if isinstance(item.get("reaction_changes"), dict):
            item = item["reaction_changes"]
        if any(key in item for key in ("formed_bonds", "broken_bonds")):
            if item.get("mapping_complete") is not True:
                raise ValueError(
                    "ReactionChangeSet must have complete explicit atom mapping before "
                    "bond progress can be validated"
                )
            if reference_structure is None:
                raise ValueError(
                    "Mapped ReactionChangeSet requires a reference path structure"
                )
            map_to_index = _structure_atom_map_indices(reference_structure)
            if map_to_index is None:
                raise ValueError(
                    "Mapped ReactionChangeSet requires complete atom maps on the path endpoint"
                )
            for key, change_type in (
                ("formed_bonds", "form"),
                ("broken_bonds", "break"),
            ):
                for index, mapped_record in enumerate(item.get(key) or []):
                    indices, maps = _mapped_change_pair(
                        mapped_record,
                        map_to_index,
                        context=f"ReactionChangeSet.{key}[{index}]",
                    )
                    records.append(
                        {
                            "type": change_type,
                            "atom_indices": indices,
                            "atom_maps": maps,
                        }
                    )
            return records
        for key, change_type in (
            ("forming_bonds", "form"),
            ("breaking_bonds", "break"),
            ("retained_bonds_to_monitor", "monitor"),
        ):
            for pair in item.get(key) or []:
                records.append({"type": change_type, "atom_indices": pair})
        if records:
            return records
        item = item.get("bond_changes") or item.get("changes") or []
    if not isinstance(item, list):
        raise ValueError("bond_changes must be a list or a forming_bonds/breaking_bonds mapping")
    for record in item:
        if not isinstance(record, dict):
            raise ValueError("each bond change must be a mapping")
        pair = record.get("atom_indices") or record.get("atoms")
        if pair is None and "atom1" in record and "atom2" in record:
            pair = [record["atom1"], record["atom2"]]
        change_type = str(record.get("type") or record.get("change") or "monitor").lower()
        if change_type in {"forming", "formed"}:
            change_type = "form"
        if change_type in {"breaking", "broken"}:
            change_type = "break"
        records.append({"type": change_type, "atom_indices": pair})
    return records


def _bond_distance(structure: dict[str, Any], pair: Any) -> float:
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        raise ValueError("bond atom_indices must contain exactly two indices")
    first, second = (int(pair[0]), int(pair[1]))
    atoms = structure.get("atoms") or []
    if first == second or min(first, second) < 0 or max(first, second) >= len(atoms):
        raise ValueError("bond atom_indices are outside the supplied path structures")
    return math.dist(
        atoms[first]["position_angstrom"], atoms[second]["position_angstrom"]
    )


def _validate_reaction_path(request: dict[str, Any]) -> dict[str, Any]:
    inputs, _method, settings = request_parts(request)
    reactant = structure_dict(inputs["reactant"])
    product = structure_dict(inputs["product"])
    frames = _path_frames(inputs["path"], reference_structure=reactant)
    for structure in (reactant, product, *frames):
        atoms_and_coordinates(structure)
    _assert_matching_endpoint_atoms(reactant, product)
    endpoint_tolerance = float(settings["endpoint_rmsd_tolerance_angstrom"])
    step_tolerance = float(settings["maximum_image_step_rmsd_angstrom"])
    bond_tolerance = float(settings["bond_distance_tolerance_angstrom"])
    if min(endpoint_tolerance, step_tolerance, bond_tolerance) <= 0:
        raise ValueError("reaction-path validation tolerances must be positive")

    expected_elements = [atom["element"] for atom in reactant["atoms"]]
    atom_order_consistent = all(
        [atom["element"] for atom in frame.get("atoms") or []] == expected_elements
        for frame in frames
    )
    atom_maps_checked, atom_identity_consistent = _path_atom_identity_status(
        [reactant, product, *frames]
    )
    atom_identity_consistent = atom_order_consistent and atom_identity_consistent
    state_consistent = all(
        int(frame.get("charge", reactant.get("charge", 0))) == int(reactant.get("charge", 0))
        and int(frame.get("multiplicity", reactant.get("multiplicity", 1)))
        == int(reactant.get("multiplicity", 1))
        for frame in frames
    )
    endpoint_rmsd = {
        "reactant_angstrom": _aligned_rmsd(frames[0], reactant),
        "product_angstrom": _aligned_rmsd(frames[-1], product),
    }
    step_rmsd = [
        _aligned_rmsd(first, second) for first, second in zip(frames, frames[1:])
    ]
    bond_changes = _bond_change_records(
        inputs.get("bond_changes"), reference_structure=reactant
    )
    bond_progress = []
    bond_changes_checked = bool(bond_changes) and atom_identity_consistent
    if bond_changes_checked:
        for change in bond_changes:
            distances = [_bond_distance(frame, change["atom_indices"]) for frame in frames]
            if change["type"] == "form":
                progress_ok = distances[-1] <= distances[0] - bond_tolerance
            elif change["type"] == "break":
                progress_ok = distances[-1] >= distances[0] + bond_tolerance
            else:
                progress_ok = True
            bond_progress.append(
                {
                    **change,
                    "distances_angstrom": distances,
                    "endpoint_progress_ok": progress_ok,
                }
            )
    checks = {
        "at_least_three_images": len(frames) >= 3,
        "atom_order_consistent": atom_order_consistent,
        "atom_identity_consistent": atom_identity_consistent,
        "charge_and_multiplicity_consistent": state_consistent,
        "reactant_endpoint_matches": endpoint_rmsd["reactant_angstrom"] <= endpoint_tolerance,
        "product_endpoint_matches": endpoint_rmsd["product_angstrom"] <= endpoint_tolerance,
        "image_steps_continuous": bool(step_rmsd) and max(step_rmsd) <= step_tolerance,
        "bond_changes_progress": (
            all(item["endpoint_progress_ok"] for item in bond_progress)
            if bond_changes_checked
            else None
        ),
    }
    required_checks = [value for value in checks.values() if value is not None]
    return success(
        {
            "valid": all(required_checks),
            "checks": checks,
            "atom_maps_checked": atom_maps_checked,
            "atom_identity_basis": (
                "atom_map_sequence" if atom_maps_checked else "element_order_only"
            ),
            "image_count": len(frames),
            "endpoint_rmsd_angstrom": endpoint_rmsd,
            "consecutive_image_rmsd_angstrom": step_rmsd,
            "maximum_consecutive_image_rmsd_angstrom": max(step_rmsd) if step_rmsd else None,
            "bond_progress": bond_progress,
            "bond_changes_requested": bool(bond_changes),
            "bond_changes_checked": bond_changes_checked,
            "tolerances": {
                "endpoint_rmsd_angstrom": endpoint_tolerance,
                "maximum_image_step_rmsd_angstrom": step_tolerance,
                "bond_distance_progress_angstrom": bond_tolerance,
            },
        }
    )


def _aligned_energies(value: Any) -> list[float]:
    item = unwrap_artifact(value)
    if isinstance(item, dict):
        for key in ("energies", "energies_hartree", "values"):
            if isinstance(item.get(key), list):
                item = item[key]
                break
    if not isinstance(item, list):
        raise ValueError("energies must be an aligned list or a mapping containing one")
    values = [
        float(record.get("value") if isinstance(record, dict) else record)
        for record in item
    ]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("energies must contain only finite values")
    return values


def _analyze_reaction_coordinate(request: dict[str, Any]) -> dict[str, Any]:
    inputs, _method, settings = request_parts(request)
    frames = _path_frames(inputs["path"])
    for frame in frames:
        atoms_and_coordinates(frame)
    expected_elements = [atom["element"] for atom in frames[0].get("atoms") or []]
    if any(
        [atom["element"] for atom in frame.get("atoms") or []] != expected_elements
        for frame in frames[1:]
    ):
        raise ValueError(
            "Reaction-coordinate analysis requires identical atom counts, elements, "
            "and ordering in every path image"
        )
    atom_maps_checked, atom_identity_consistent = _path_atom_identity_status(frames)
    if not atom_identity_consistent:
        raise ValueError(
            "Reaction-coordinate analysis requires complete, identical atom-map "
            "sequences in every path image when any image contains atom maps"
        )
    energies = _aligned_energies(inputs["energies"])
    if len(frames) != len(energies):
        raise ValueError("energies must contain exactly one value per path image")
    unit = str(settings["energy_unit"]).strip().lower()
    kcal_factor = {
        "hartree": 627.5094740631,
        "kj/mol": 1.0 / 4.184,
        "kcal/mol": 1.0,
        "ev": 23.0605478306,
    }[unit]
    minimum = min(energies)
    relative_input_unit = [value - minimum for value in energies]
    relative_kcal = [value * kcal_factor for value in relative_input_unit]
    step_rmsd = [
        _aligned_rmsd(first, second) for first, second in zip(frames, frames[1:])
    ]
    cumulative = [0.0]
    for value in step_rmsd:
        cumulative.append(cumulative[-1] + value)
    total = cumulative[-1]
    normalized = [value / total for value in cumulative] if total > 0 else [0.0] * len(frames)
    maxima = [
        index
        for index in range(1, len(energies) - 1)
        if energies[index] >= energies[index - 1] and energies[index] >= energies[index + 1]
    ]
    minima = [
        index
        for index in range(1, len(energies) - 1)
        if energies[index] <= energies[index - 1] and energies[index] <= energies[index + 1]
    ]
    highest = max(range(len(energies)), key=energies.__getitem__)
    bond_profiles = []
    for change in _bond_change_records(
        inputs.get("bond_changes"), reference_structure=frames[0]
    ):
        bond_profiles.append(
            {
                **change,
                "distances_angstrom": [
                    _bond_distance(frame, change["atom_indices"]) for frame in frames
                ],
            }
        )
    return success(
        {
            "image_count": len(frames),
            "reaction_coordinate": normalized,
            "cumulative_aligned_rmsd_angstrom": cumulative,
            "energies": energies,
            "energy_unit": unit,
            "relative_energies": relative_input_unit,
            "relative_energies_kcal_mol": relative_kcal,
            "highest_energy_image_index": highest,
            "highest_energy_relative_kcal_mol": relative_kcal[highest],
            "interior_local_maxima_indices": maxima,
            "interior_local_minima_indices": minima,
            "bond_distance_profiles": bond_profiles,
            "atom_maps_checked": atom_maps_checked,
            "atom_identity_consistent": atom_identity_consistent,
            "atom_identity_basis": (
                "atom_map_sequence" if atom_maps_checked else "element_order_only"
            ),
            "transition_state_validated": False,
            "validation_note": (
                "A highest path image is only a candidate; validate with a TS optimization, "
                "one-imaginary-frequency Hessian, and endpoint-connected IRC."
            ),
        }
    )


def _cantera_equilibrium(request: dict[str, Any]) -> dict[str, Any]:
    import cantera as ct

    inputs, _method, settings = request_parts(request)
    mechanism = inputs.get("mechanism") or settings.get("mechanism")
    if not isinstance(mechanism, str) or not mechanism:
        raise ValueError("Cantera requires an explicit mechanism name/path")
    gas = ct.Solution(mechanism)
    composition = inputs["composition"]
    gas.TPX = (
        float(settings["temperature_kelvin"]),
        float(settings["pressure_pa"]),
        composition,
    )
    gas.equilibrate(str(settings["equilibrium_mode"]))
    species = sorted(
        (
            {"name": gas.species_names[index], "mole_fraction": float(value)}
            for index, value in enumerate(gas.X)
            if value > float(settings.get("report_threshold", 1e-12))
        ),
        key=lambda item: item["mole_fraction"],
        reverse=True,
    )
    return success(
        {
            "mechanism": mechanism,
            "equilibrium_mode": str(settings["equilibrium_mode"]),
            "temperature_kelvin": float(gas.T),
            "pressure_pa": float(gas.P),
            "enthalpy_mole_j_mol": float(gas.enthalpy_mole),
            "gibbs_mole_j_mol": float(gas.gibbs_mole),
            "species": species,
        },
        backend_version=getattr(ct, "__version__", None),
    )


def _network_parts(request: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    inputs, method, settings = request_parts(request)
    network = inputs["network"]
    initial = inputs["initial_state"]
    if not isinstance(network, dict) or not isinstance(initial, dict):
        raise ValueError("network and initial_state must be mappings")
    return network, initial, settings


def _scipy_network(request: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    from scipy.integrate import solve_ivp

    network, initial, settings = _network_parts(request)
    species = [str(value) for value in network["species"]]
    reactions = list(network["reactions"])
    if not species or not reactions:
        raise ValueError("ReactionNetwork requires non-empty species and reactions")
    initial_vector = np.asarray([float(initial.get(name, 0.0)) for name in species], dtype=float)
    if np.any(initial_vector < 0):
        raise ValueError("Initial concentrations cannot be negative")
    reactants = np.zeros((len(reactions), len(species)), dtype=float)
    products = np.zeros_like(reactants)
    constants = np.zeros(len(reactions), dtype=float)
    index = {name: position for position, name in enumerate(species)}
    for reaction_index, reaction in enumerate(reactions):
        for name, coefficient in dict(reaction["reactants"]).items():
            reactants[reaction_index, index[name]] = float(coefficient)
        for name, coefficient in dict(reaction["products"]).items():
            products[reaction_index, index[name]] = float(coefficient)
        constants[reaction_index] = float(reaction["forward_rate_constant"])
    net = products - reactants

    def derivative(_time, concentrations):
        clipped = np.maximum(concentrations, 0.0)
        rates = constants * np.prod(np.power(clipped[None, :], reactants), axis=1)
        return net.T @ rates

    end = float(settings["time_end_seconds"])
    points = int(settings["num_points"])
    times = np.linspace(0.0, end, points)
    solution = solve_ivp(
        derivative,
        (0.0, end),
        initial_vector,
        t_eval=times,
        method=str(settings.get("integrator", "LSODA")),
        rtol=float(settings.get("relative_tolerance", 1e-8)),
        atol=float(settings.get("absolute_tolerance", 1e-12)),
    )
    if not solution.success or not np.all(np.isfinite(solution.y)):
        raise RuntimeError(f"Kinetics integration failed: {solution.message}")
    return success(
        {
            "species": species,
            "time_seconds": solution.t.tolist(),
            "concentrations": {
                name: solution.y[position].tolist() for position, name in enumerate(species)
            },
            "final_state": {
                name: float(solution.y[position, -1]) for position, name in enumerate(species)
            },
            "integrator": str(settings.get("integrator", "LSODA")),
        },
        backend_version=module_version("scipy"),
    )


def _cantera_network(request: dict[str, Any]) -> dict[str, Any]:
    import cantera as ct
    import numpy as np

    network, initial, settings = _network_parts(request)
    mechanism = network.get("mechanism")
    composition = initial.get("composition")
    if not mechanism or not composition:
        return unsupported(
            "Cantera network integration requires network.mechanism and initial_state.composition"
        )
    gas = ct.Solution(str(mechanism))
    gas.TPX = (
        float(initial["temperature_kelvin"]),
        float(initial["pressure_pa"]),
        composition,
    )
    reactor = ct.IdealGasConstPressureReactor(gas)
    simulation = ct.ReactorNet([reactor])
    times = np.linspace(0.0, float(settings["time_end_seconds"]), int(settings["num_points"]))
    temperatures = []
    compositions = []
    for time_value in times:
        simulation.advance(float(time_value))
        temperatures.append(float(reactor.T))
        compositions.append(reactor.thermo.X.tolist())
    return success(
        {
            "time_seconds": times.tolist(),
            "temperature_kelvin": temperatures,
            "species": list(gas.species_names),
            "mole_fractions": compositions,
        },
        backend_version=getattr(ct, "__version__", None),
    )


def _catmap(request: dict[str, Any]) -> dict[str, Any]:
    inputs, _method, settings = request_parts(request)
    model_value = inputs["model"]
    if not isinstance(model_value, dict):
        raise ValueError("CatMAP model must be a typed mapping")
    allowed = {
        "rxn_expressions", "species_definitions", "descriptor_names", "descriptor_ranges",
        "resolution", "surface_names", "input_file", "gas_thermo_mode",
        "adsorbate_thermo_mode", "scaler", "solver", "mapper", "output_variables",
        "scaling_constraint_dict", "numerical_representation",
        "adsorbate_interaction_model", "decimal_precision", "tolerance",
        "max_rootfinding_iterations", "max_bisections", "max_damping_iterations",
        "use_numbers_solver", "prefactor_list", "descriptor_values",
    }
    unknown = sorted(set(model_value) - allowed)
    if unknown:
        raise ValueError(f"Unsupported CatMAP model fields: {unknown}")
    from catmap import ReactionModel

    directory = output_directory("solve_microkinetic_model", "catmap")
    configuration = dict(model_value)
    if "input_file" in configuration:
        configuration["input_file"] = str(resolve_input_file(configuration["input_file"]))
    configuration["temperature"] = float(settings["temperature_kelvin"])
    configuration["pressure"] = float(settings["pressure_bar"])
    # CatMAP's setup-file loader initializes the parser/scaler/solver/mapper
    # defaults.  Constructing ReactionModel() and setting attributes directly
    # leaves those objects uninitialized in CatMAP 0.3.x.  The generated file is
    # restricted to allow-listed literal assignments; arbitrary user code is
    # never accepted or executed.
    configuration["data_file"] = str(directory / "catmap_data.pkl")
    setup_path = directory / "model.mkm"
    setup_path.write_text(
        "\n".join(
            f"{name} = {pprint.pformat(value, sort_dicts=True, width=100)}"
            for name, value in configuration.items()
        )
        + "\n",
        encoding="utf-8",
    )
    model = ReactionModel(setup_file=str(setup_path))
    model.run()

    def json_value(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(key): json_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_value(item) for item in value]
        if hasattr(value, "tolist"):
            return json_value(value.tolist())
        try:
            return float(value)
        except (TypeError, ValueError, OverflowError):
            return str(value)

    result = {
        "temperature_kelvin": model.temperature,
        "pressure_bar": model.pressure,
        "output_variables": list(getattr(model, "output_variables", [])),
        "output_labels": json_value(getattr(model, "output_labels", {})),
    }
    for name in result["output_variables"]:
        value = getattr(model, f"{name}_map", None)
        if value is None:
            value = getattr(model, name, None)
        if value is not None:
            result[name] = json_value(value)
    path = write_json(directory, "catmap_result.json", result)
    return success(
        result,
        artifact_files=[
            {
                "path": relative_workspace_path(path),
                "semantic_type": "MicrokineticResult",
                "media_type": "application/json",
            },
            *[
                artifact
                for artifact in command_artifacts(directory)
                if artifact["path"] != relative_workspace_path(path)
            ],
        ],
        backend_version=module_version("catmap"),
        provenance={
            "generated_setup": relative_workspace_path(setup_path),
            "accepted_model_fields": sorted(model_value),
        },
    )


def _rmg_temperatures(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("temperatures_kelvin must be a non-empty list")
    temperatures = [float(item) for item in value]
    if any(not math.isfinite(item) or item <= 0 for item in temperatures):
        raise ValueError("temperatures_kelvin must contain positive finite values")
    return temperatures


def _rmg_arrhenius(specification: dict[str, Any]):
    from rmgpy.kinetics import Arrhenius

    required = {
        "pre_exponential_factor", "pre_exponential_unit", "temperature_exponent",
        "activation_energy_kj_mol", "reference_temperature_kelvin",
    }
    missing = sorted(required - set(specification))
    if missing:
        raise ValueError(f"Arrhenius term is missing fields: {missing}")
    keywords: dict[str, Any] = {
        "A": (
            float(specification["pre_exponential_factor"]),
            str(specification["pre_exponential_unit"]),
        ),
        "n": float(specification["temperature_exponent"]),
        "Ea": (float(specification["activation_energy_kj_mol"]), "kJ/mol"),
        "T0": (float(specification["reference_temperature_kelvin"]), "K"),
    }
    if specification.get("minimum_temperature_kelvin") is not None:
        keywords["Tmin"] = (float(specification["minimum_temperature_kelvin"]), "K")
    if specification.get("maximum_temperature_kelvin") is not None:
        keywords["Tmax"] = (float(specification["maximum_temperature_kelvin"]), "K")
    return Arrhenius(**keywords)


def _rmg_rate_constants(request: dict[str, Any]) -> dict[str, Any]:
    import rmgpy
    from rmgpy.kinetics import Chebyshev, MultiArrhenius, PDepArrhenius

    inputs, _method, settings = request_parts(request)
    model_spec = inputs["kinetics_model"]
    if not isinstance(model_spec, dict):
        raise ValueError("kinetics_model must be a typed object")
    model_type = str(model_spec.get("type") or "").strip().lower()
    temperatures = _rmg_temperatures(inputs["temperatures_kelvin"])
    if not isinstance(settings["allow_extrapolation"], bool):
        raise ValueError("allow_extrapolation must be an explicit boolean")
    allow_extrapolation = settings["allow_extrapolation"]
    reaction_order = int(model_spec.get("reaction_order", 0))
    rate_units = {
        1: "s^-1",
        2: "m^3/(mol*s)",
        3: "m^6/(mol^2*s)",
        4: "m^9/(mol^3*s)",
    }
    if reaction_order not in rate_units:
        raise ValueError("kinetics_model.reaction_order must be 1, 2, 3, or 4")

    minimum_temperature = model_spec.get("minimum_temperature_kelvin")
    maximum_temperature = model_spec.get("maximum_temperature_kelvin")
    if not allow_extrapolation:
        if minimum_temperature is not None and min(temperatures) < float(minimum_temperature):
            raise ValueError("Requested temperature is below the kinetics model validity range")
        if maximum_temperature is not None and max(temperatures) > float(maximum_temperature):
            raise ValueError("Requested temperature is above the kinetics model validity range")

    pressure_dependent = False
    if model_type == "arrhenius":
        model = _rmg_arrhenius(model_spec)
    elif model_type == "multi_arrhenius":
        terms = model_spec.get("terms")
        if not isinstance(terms, list) or not terms:
            raise ValueError("multi_arrhenius requires a non-empty terms list")
        model = MultiArrhenius(
            arrhenius=[_rmg_arrhenius(dict(term)) for term in terms],
            Tmin=(float(minimum_temperature), "K") if minimum_temperature is not None else None,
            Tmax=(float(maximum_temperature), "K") if maximum_temperature is not None else None,
        )
    elif model_type == "pressure_dependent_arrhenius":
        pressure_dependent = True
        tabulated_pressures = model_spec.get("pressures_bar")
        terms = model_spec.get("terms")
        if (
            not isinstance(tabulated_pressures, list)
            or not isinstance(terms, list)
            or not tabulated_pressures
            or len(tabulated_pressures) != len(terms)
        ):
            raise ValueError(
                "pressure_dependent_arrhenius requires aligned non-empty pressures_bar and terms"
            )
        tabulated_pressures = [float(value) for value in tabulated_pressures]
        if any(value <= 0 for value in tabulated_pressures):
            raise ValueError("pressures_bar must be positive")
        model = PDepArrhenius(
            pressures=(tabulated_pressures, "bar"),
            arrhenius=[_rmg_arrhenius(dict(term)) for term in terms],
            Tmin=(float(minimum_temperature), "K") if minimum_temperature is not None else None,
            Tmax=(float(maximum_temperature), "K") if maximum_temperature is not None else None,
            Pmin=(min(tabulated_pressures), "bar"),
            Pmax=(max(tabulated_pressures), "bar"),
        )
    elif model_type == "chebyshev":
        pressure_dependent = True
        coefficients = model_spec.get("coefficients")
        if not isinstance(coefficients, list) or not coefficients or not all(
            isinstance(row, list) and row for row in coefficients
        ):
            raise ValueError("chebyshev requires a non-empty rectangular coefficients matrix")
        width = len(coefficients[0])
        if any(len(row) != width for row in coefficients):
            raise ValueError("Chebyshev coefficients must form a rectangular matrix")
        required = {
            "rate_coefficient_unit", "minimum_temperature_kelvin",
            "maximum_temperature_kelvin", "minimum_pressure_bar", "maximum_pressure_bar",
        }
        missing = sorted(required - set(model_spec))
        if missing:
            raise ValueError(f"Chebyshev model is missing fields: {missing}")
        model = Chebyshev(
            coeffs=coefficients,
            kunits=str(model_spec["rate_coefficient_unit"]),
            Tmin=(float(model_spec["minimum_temperature_kelvin"]), "K"),
            Tmax=(float(model_spec["maximum_temperature_kelvin"]), "K"),
            Pmin=(float(model_spec["minimum_pressure_bar"]), "bar"),
            Pmax=(float(model_spec["maximum_pressure_bar"]), "bar"),
        )
    else:
        raise ValueError(
            "kinetics_model.type must be arrhenius, multi_arrhenius, "
            "pressure_dependent_arrhenius, or chebyshev"
        )

    if pressure_dependent:
        raw_pressures = inputs.get("pressures_pa")
        if not isinstance(raw_pressures, (list, tuple)) or not raw_pressures:
            raise ValueError("Pressure-dependent kinetics require a non-empty pressures_pa list")
        pressures = [float(value) for value in raw_pressures]
        if any(not math.isfinite(value) or value <= 0 for value in pressures):
            raise ValueError("pressures_pa must contain positive finite values")
        if not allow_extrapolation:
            minimum_pressure = float(model_spec.get("minimum_pressure_bar", min(model_spec.get("pressures_bar", [0.0])))) * 1e5
            maximum_pressure = float(model_spec.get("maximum_pressure_bar", max(model_spec.get("pressures_bar", [0.0])))) * 1e5
            if min(pressures) < minimum_pressure or max(pressures) > maximum_pressure:
                raise ValueError("Requested pressure is outside the kinetics model validity range")
        values = [
            [float(model.get_rate_coefficient(temperature, pressure)) for pressure in pressures]
            for temperature in temperatures
        ]
    else:
        pressures = None
        values = [float(model.get_rate_coefficient(temperature)) for temperature in temperatures]

    result = {
        "model_type": model_type,
        "reaction_order": reaction_order,
        "temperatures_kelvin": temperatures,
        "pressures_pa": pressures,
        "rate_coefficients": values,
        "rate_coefficient_unit": rate_units[reaction_order],
        "allow_extrapolation": allow_extrapolation,
    }
    directory = output_directory("calculate_rate_constants", "rmg")
    path = write_json(directory, "rate_constants.json", result)
    return success(
        result,
        artifact_files=[
            {
                "path": relative_workspace_path(path),
                "semantic_type": "RateConstantResult",
                "media_type": "application/json",
            }
        ],
        backend_version=getattr(rmgpy, "__version__", None),
    )


def _rmg_tunneling(request: dict[str, Any]) -> dict[str, Any]:
    import rmgpy
    from rmgpy.kinetics.tunneling import Eckart, Wigner

    inputs, method, _settings = request_parts(request)
    temperatures = _rmg_temperatures(inputs["temperatures_kelvin"])
    frequency = float(inputs["imaginary_frequency_cm1"])
    if not math.isfinite(frequency) or frequency >= 0:
        raise ValueError("imaginary_frequency_cm1 must be a finite negative frequency")
    model_name = str(method["tunneling_model"]).strip().lower()
    if model_name == "wigner":
        model = Wigner(frequency=(frequency, "cm^-1"))
        energies = None
    elif model_name == "eckart":
        required = ("reactant_energy_kj_mol", "transition_state_energy_kj_mol")
        missing = [name for name in required if name not in inputs]
        if missing:
            raise ValueError(f"Eckart tunneling requires inputs: {missing}")
        reactant = float(inputs["reactant_energy_kj_mol"])
        transition = float(inputs["transition_state_energy_kj_mol"])
        product = inputs.get("product_energy_kj_mol")
        if transition <= reactant or (product is not None and transition <= float(product)):
            raise ValueError("Eckart transition-state energy must exceed reactant and product energies")
        model = Eckart(
            frequency=(frequency, "cm^-1"),
            E0_reac=(reactant, "kJ/mol"),
            E0_TS=(transition, "kJ/mol"),
            E0_prod=(float(product), "kJ/mol") if product is not None else None,
        )
        energies = {
            "reactant_energy_kj_mol": reactant,
            "transition_state_energy_kj_mol": transition,
            "product_energy_kj_mol": float(product) if product is not None else None,
        }
    else:
        raise ValueError("tunneling_model must be wigner or eckart")
    factors = [float(model.calculate_tunneling_factor(value)) for value in temperatures]
    result = {
        "tunneling_model": model_name,
        "imaginary_frequency_cm1": frequency,
        "temperatures_kelvin": temperatures,
        "tunneling_factors": factors,
        "energies": energies,
    }
    directory = output_directory("calculate_tunneling_correction", "rmg")
    path = write_json(directory, "tunneling_correction.json", result)
    return success(
        result,
        artifact_files=[
            {
                "path": relative_workspace_path(path),
                "semantic_type": "TunnelingCorrectionResult",
                "media_type": "application/json",
            }
        ],
        backend_version=getattr(rmgpy, "__version__", None),
    )


def _safe_stage_relative_path(value: Any, *, field: str) -> Path:
    text = str(value).strip()
    path = Path(text)
    if not text or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{field} must be a normalized relative path inside the staged model")
    return path


def _stage_master_equation_model(
    action_id: str,
    backend_id: str,
    inputs: dict[str, Any],
) -> tuple[Path, Path, list[dict[str, str]]]:
    """Stage only the native model and explicitly supplied companion files.

    ``model_relative_path`` and companion ``relative_path`` values let an Agent
    preserve a backend's relative-file layout without granting arbitrary writes
    outside the isolated action directory.
    """

    directory = output_directory(action_id, backend_id)
    stage_root = directory / "staged_model"
    stage_root.mkdir()
    model_source = resolve_input_file(inputs["model_file"])
    if not model_source.is_file():
        raise ValueError("model_file must resolve to one regular file")
    default_relative = model_source.name
    model_relative = _safe_stage_relative_path(
        inputs.get("model_relative_path", default_relative),
        field="model_relative_path",
    )
    model_target = stage_root / model_relative
    model_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_source, model_target)
    staged = [
        {
            "source": relative_workspace_path(model_source),
            "relative_path": model_relative.as_posix(),
        }
    ]

    companions = inputs.get("companion_files", [])
    if not isinstance(companions, list):
        raise ValueError("companion_files must be a list")
    if len(companions) > 128:
        raise ValueError("At most 128 explicit companion files may be staged")
    occupied = {model_relative.as_posix()}
    for index, item in enumerate(companions):
        if isinstance(item, dict) and "source" in item:
            source_value = item["source"]
            source = resolve_input_file(source_value)
            relative = _safe_stage_relative_path(
                item.get("relative_path", source.name),
                field=f"companion_files[{index}].relative_path",
            )
        else:
            source = resolve_input_file(item)
            relative = _safe_stage_relative_path(
                source.name,
                field=f"companion_files[{index}]",
            )
        if not source.is_file():
            raise ValueError(f"companion_files[{index}] must resolve to one regular file")
        relative_text = relative.as_posix()
        if relative_text in occupied:
            raise ValueError(f"Duplicate staged path: {relative_text}")
        occupied.add(relative_text)
        target = stage_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        staged.append(
            {
                "source": relative_workspace_path(source),
                "relative_path": relative_text,
            }
        )
    return directory, model_target, staged


def _float_token(value: str) -> float | None:
    token = value.strip().replace("D", "E").replace("d", "e")
    if token in {"", "***", "-", "nan", "NaN"}:
        return None
    try:
        result = float(token)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def _bounded_warning_lines(*texts: str, limit: int = 40) -> list[str]:
    warnings: list[str] = []
    for text in texts:
        for line in text.splitlines():
            stripped = line.strip()
            lowered = stripped.lower()
            if stripped and (
                "warning" in lowered
                or "needs to be checked" in lowered
                or lowered.startswith("error")
            ):
                if stripped not in warnings:
                    warnings.append(stripped[:1000])
                if len(warnings) >= limit:
                    return warnings
    return warnings


def _mess_species_names(lines: list[str], heading: str) -> set[str]:
    names: set[str] = set()
    for index, line in enumerate(lines):
        if not line.strip().startswith(heading):
            continue
        cursor = index + 1
        while cursor < len(lines) and "Name" not in lines[cursor]:
            cursor += 1
        cursor += 1
        while cursor < len(lines) and lines[cursor].strip():
            token = lines[cursor].split()[0] if lines[cursor].split() else ""
            if token:
                names.add(token)
            cursor += 1
        break
    return names


def _parse_mess_rates(text: str, maximum_records: int) -> dict[str, Any]:
    lines = text.splitlines()
    wells = _mess_species_names(lines, "Wells (")
    bimolecular = _mess_species_names(lines, "Bimolecular Products (")
    start = next(
        (index for index, line in enumerate(lines) if line.strip() == "Species-Species Rate Tables:"),
        None,
    )
    if start is None:
        return {
            "rate_records": [],
            "total_rate_record_count": 0,
            "truncated": False,
            "wells": sorted(wells),
            "bimolecular_species": sorted(bimolecular),
        }
    stop = next(
        (
            index
            for index in range(start + 1, len(lines))
            if lines[index].strip().startswith(
                "High Pressure Rate Coefficients (Temperature-Species Rate Tables)"
            )
        ),
        len(lines),
    )
    temperature: float | None = None
    pressure: float | None = None
    pressure_unit: str | None = None
    records: list[dict[str, Any]] = []
    total = 0
    cursor = start + 1
    condition_pattern = re.compile(
        r"^Temperature\s*=\s*([-+0-9.eEdD]+)\s*K"
        r"(?:\s+Pressure\s*=\s*([-+0-9.eEdD]+)\s*(\S+))?$"
    )
    while cursor < stop:
        stripped = lines[cursor].strip()
        condition = condition_pattern.match(stripped)
        if condition:
            temperature = _float_token(condition.group(1))
            pressure = _float_token(condition.group(2) or "")
            pressure_unit = condition.group(3) if condition.group(2) else None
            cursor += 1
            continue
        if stripped.startswith("From\\To") and temperature is not None:
            destinations = stripped.split()[1:]
            cursor += 1
            while cursor < stop and lines[cursor].strip():
                tokens = lines[cursor].split()
                if len(tokens) < 2:
                    break
                source = tokens[0]
                for destination, token in zip(destinations, tokens[1:]):
                    value = _float_token(token)
                    if value is None or source == destination:
                        continue
                    total += 1
                    if len(records) >= maximum_records:
                        continue
                    if source in wells:
                        unit = "s^-1"
                    elif source in bimolecular:
                        unit = "cm^3/s"
                    else:
                        unit = "backend_native"
                    records.append(
                        {
                            "from_species": source,
                            "to_species": destination,
                            "temperature_kelvin": temperature,
                            "pressure_value": pressure,
                            "pressure_unit": pressure_unit,
                            "pressure_limit": "finite" if pressure is not None else "high",
                            "rate_coefficient": value,
                            "rate_coefficient_unit": unit,
                        }
                    )
                cursor += 1
            continue
        cursor += 1
    return {
        "rate_records": records,
        "total_rate_record_count": total,
        "truncated": total > len(records),
        "wells": sorted(wells),
        "bimolecular_species": sorted(bimolecular),
    }


def _mess_master_equation(request: dict[str, Any]) -> dict[str, Any]:
    inputs, _method, settings = request_parts(request)
    maximum_records = int(settings.get("maximum_rate_records", 10000))
    if not 1 <= maximum_records <= 100000:
        raise ValueError("maximum_rate_records must be between 1 and 100000")
    directory, model, staged = _stage_master_equation_model(
        "solve_master_equation", "mess", inputs
    )
    if model.suffix.lower() not in {".inp", ".in", ".mess"}:
        raise ValueError("MESS model_file must use .inp, .in, or .mess")
    completed = run_external(
        executable="mess",
        environment_variable="CHEMGRAPH_MESS_COMMAND",
        arguments=[model.name],
        directory=model.parent,
        timeout_seconds=int(request.get("resource_limits", {}).get("walltime_seconds", 1800)),
    )
    (directory / "stdout.log").write_text(completed["stdout"], encoding="utf-8")
    (directory / "stderr.log").write_text(completed["stderr"], encoding="utf-8")
    if not completed["available"]:
        return unavailable(completed["stderr"], install="Configure MESS 2020.1.24")
    if completed["returncode"] != 0:
        raise RuntimeError(f"MESS failed: {completed['stderr'][-2000:]}")
    output = model.with_suffix(".out")
    log = model.with_suffix(".log")
    output_text = output.read_text(encoding="utf-8", errors="replace") if output.is_file() else ""
    log_text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
    parsed = _parse_mess_rates(output_text, maximum_records)
    result = {
        "model_format": "mess_native",
        "model_file": relative_workspace_path(model),
        "staged_files": staged,
        **parsed,
        "completed": "rate calculation done" in log_text.lower(),
    }
    summary = write_json(directory, "master_equation_result.json", result)
    summary_path = relative_workspace_path(summary)
    artifacts = [
        item for item in command_artifacts(directory) if item["path"] != summary_path
    ]
    artifacts.append(
        {
            "path": summary_path,
            "semantic_type": "MasterEquationResult",
            "media_type": "application/json",
        }
    )
    warnings = _bounded_warning_lines(output_text, log_text, completed["stdout"], completed["stderr"])
    provenance = {
        "command": completed["command"],
        "native_model_preserved": True,
        "automatic_model_construction": False,
    }
    if not result["completed"] or not parsed["rate_records"]:
        return partial_success(
            result,
            artifact_files=artifacts,
            backend_version="2020.1.24",
            provenance=provenance,
            warnings=[*warnings, "MESS completed without a fully parseable rate table."],
        )
    return success(
        result,
        artifact_files=artifacts,
        backend_version="2020.1.24",
        provenance=provenance,
        warnings=warnings,
    )


def _xml_local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _parse_mesmer_rates(path: Path, maximum_records: int) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    conditions: list[dict[str, Any]] = []
    for element in root.iter():
        if _xml_local_name(element.tag) != "PTpair":
            continue
        conditions.append(
            {
                "temperature_kelvin": _float_token(element.attrib.get("T", "")),
                "pressure_value": _float_token(element.attrib.get("P", "")),
                "pressure_unit": element.attrib.get("units"),
            }
        )
    used_conditions: set[int] = set()
    records: list[dict[str, Any]] = []
    total = 0
    rate_lists = [
        element for element in root.iter() if _xml_local_name(element.tag) == "rateList"
    ]
    for rate_index, rate_list in enumerate(rate_lists):
        temperature = _float_token(rate_list.attrib.get("T", ""))
        condition_index = next(
            (
                index
                for index, condition in enumerate(conditions)
                if index not in used_conditions
                and condition["temperature_kelvin"] == temperature
            ),
            None,
        )
        if condition_index is None and rate_index < len(conditions):
            condition_index = rate_index
        condition = conditions[condition_index] if condition_index is not None else {}
        if condition_index is not None:
            used_conditions.add(condition_index)
        for element in rate_list.iter():
            rate_type = _xml_local_name(element.tag)
            if rate_type not in {"firstOrderRate", "secondOrderRate"}:
                continue
            value = _float_token(element.text or "")
            if value is None:
                continue
            total += 1
            if len(records) >= maximum_records:
                continue
            records.append(
                {
                    "from_species": element.attrib.get("fromRef"),
                    "to_species": element.attrib.get("toRef"),
                    "reaction_type": element.attrib.get("reactionType"),
                    "rate_type": rate_type,
                    "temperature_kelvin": temperature,
                    "pressure_value": condition.get("pressure_value"),
                    "pressure_unit": condition.get("pressure_unit"),
                    "bath_gas": rate_list.attrib.get("bathGas"),
                    "excess_reactant_concentration": _float_token(
                        rate_list.attrib.get("conc", "")
                    ),
                    "rate_coefficient": value,
                    "rate_coefficient_unit": (
                        "s^-1"
                        if rate_type == "firstOrderRate"
                        else "cm^3 molecule^-1 s^-1"
                    ),
                }
            )
    return {
        "conditions": conditions,
        "rate_records": records,
        "total_rate_record_count": total,
        "truncated": total > len(records),
    }


def _mesmer_master_equation(request: dict[str, Any]) -> dict[str, Any]:
    inputs, _method, settings = request_parts(request)
    maximum_records = int(settings.get("maximum_rate_records", 10000))
    if not 1 <= maximum_records <= 100000:
        raise ValueError("maximum_rate_records must be between 1 and 100000")
    directory, model, staged = _stage_master_equation_model(
        "solve_master_equation", "mesmer", inputs
    )
    if model.suffix.lower() != ".xml":
        raise ValueError("MESMER model_file must use .xml")
    audit = model.parent / "mesmer.audit.xml"
    completed = run_external(
        executable="mesmer",
        environment_variable="CHEMGRAPH_MESMER_COMMAND",
        arguments=[model.name, f"-o{audit.name}"],
        directory=model.parent,
        timeout_seconds=int(request.get("resource_limits", {}).get("walltime_seconds", 1800)),
    )
    (directory / "stdout.log").write_text(completed["stdout"], encoding="utf-8")
    (directory / "stderr.log").write_text(completed["stderr"], encoding="utf-8")
    if not completed["available"]:
        return unavailable(completed["stderr"], install="Configure MESMER 7.1")
    if completed["returncode"] != 0:
        raise RuntimeError(f"MESMER failed: {completed['stderr'][-2000:]}")
    parsed = _parse_mesmer_rates(audit, maximum_records) if audit.is_file() else {
        "conditions": [], "rate_records": [], "total_rate_record_count": 0, "truncated": False
    }
    log = model.parent / "mesmer.log"
    log_text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
    result = {
        "model_format": "mesmer_xml",
        "model_file": relative_workspace_path(model),
        "staged_files": staged,
        **parsed,
        "completed": audit.is_file() and bool(parsed["rate_records"]),
    }
    summary = write_json(directory, "master_equation_result.json", result)
    summary_path = relative_workspace_path(summary)
    artifacts = [
        item for item in command_artifacts(directory) if item["path"] != summary_path
    ]
    artifacts.append(
        {
            "path": summary_path,
            "semantic_type": "MasterEquationResult",
            "media_type": "application/json",
        }
    )
    warnings = _bounded_warning_lines(
        completed["stdout"], completed["stderr"], log_text,
        audit.read_text(encoding="utf-8", errors="replace") if audit.is_file() else "",
    )
    provenance = {
        "command": completed["command"],
        "native_model_preserved": True,
        "automatic_model_construction": False,
    }
    if not result["completed"]:
        return partial_success(
            result,
            artifact_files=artifacts,
            backend_version="7.1",
            provenance=provenance,
            warnings=[*warnings, "MESMER completed without a parseable phenomenological rate list."],
        )
    return success(
        result,
        artifact_files=artifacts,
        backend_version="7.1",
        provenance=provenance,
        warnings=warnings,
    )


def execute(action_id: str, backend_id: str, request: dict[str, Any]) -> dict[str, Any]:
    if action_id == "search_transition_state_path" and backend_id == "transition_state_path_search":
        from .arche_ts_pipeline import execute as execute_arche_ts_pipeline

        return execute_arche_ts_pipeline(action_id, backend_id, request)
    if action_id in {
        "map_reaction_atoms",
        "materialize_mapped_hydrogen_transfers",
        "analyze_reaction_changes",
        "apply_reaction_changes_to_structure",
    }:
        from .reaction_mapping import execute as execute_reaction_mapping

        return execute_reaction_mapping(action_id, backend_id, request)
    if backend_id == "ash_neb_uma" and action_id == "search_reaction_path":
        from .ash_neb import execute as execute_ash_neb

        return execute_ash_neb(action_id, backend_id, request)
    if backend_id == "goodvibes":
        return execute_goodvibes(action_id, request)
    if (
        action_id in {"analyze_free_energy_selectivity", "analyze_energetic_span"}
        and backend_id == "internal_thermochemistry"
    ):
        from .electronic import execute as execute_electronic

        return execute_electronic(action_id, backend_id, request)
    if backend_id == "sella" and action_id == "locate_transition_state":
        return execute_sella(action_id, request)
    if backend_id == "pysisyphus":
        return _pysisyphus(action_id, request)
    if backend_id == "gaussian" and action_id in {
        "locate_transition_state", "scan_reaction_coordinates",
        "trace_intrinsic_reaction_coordinate",
    }:
        from .quantum_legacy import gaussian

        return gaussian(action_id, request)
    if backend_id == "internal_reaction_analysis":
        if action_id == "parse_gaussian_irc_output":
            return _parse_gaussian_irc_output(request)
        if action_id == "validate_reaction_path":
            return _validate_reaction_path(request)
        if action_id == "analyze_reaction_coordinate":
            return _analyze_reaction_coordinate(request)
        if action_id == "extract_reaction_path_image":
            return _extract_reaction_path_image(request)
    if action_id == "calculate_chemical_equilibrium" and backend_id == "cantera":
        return _cantera_equilibrium(request)
    if action_id == "integrate_reaction_network":
        return _scipy_network(request) if backend_id == "scipy" else _cantera_network(request)
    if action_id == "solve_microkinetic_model" and backend_id == "catmap":
        return _catmap(request)
    if action_id == "calculate_rate_constants" and backend_id == "rmg":
        return _rmg_rate_constants(request)
    if action_id == "calculate_tunneling_correction" and backend_id == "rmg":
        return _rmg_tunneling(request)
    if action_id == "solve_master_equation":
        if backend_id == "mess":
            return _mess_master_equation(request)
        if backend_id == "mesmer":
            return _mesmer_master_equation(request)
    return unsupported(f"Unsupported reaction action/backend combination: {action_id}/{backend_id}")
