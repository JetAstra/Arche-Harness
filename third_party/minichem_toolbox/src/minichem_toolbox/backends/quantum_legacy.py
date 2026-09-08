"""Typed atomic adapters for operator-provided Gaussian and GAMESS runtimes."""

from __future__ import annotations

import math
import re
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

from .common import (
    atoms_and_coordinates,
    command_artifacts,
    output_directory,
    partial_success,
    relative_workspace_path,
    request_parts,
    run_external,
    structure_dict,
    success,
    unavailable,
    unsupported,
    write_json,
)
from .gaussian_irc import (
    parse_gaussian_irc_points as _shared_parse_gaussian_irc_points,
    parse_gaussian_irc_summary as _shared_parse_gaussian_irc_summary,
    parse_gaussian_irc_termination as _shared_parse_gaussian_irc_termination,
)
from minichem_toolbox.gaussian_termination import (
    GAUSSIAN_ARCHIVE_ENTRY_FAILURE_MARKER as _GAUSSIAN_ARCHIVE_ENTRY_FAILURE_MARKER,
    GAUSSIAN_ARCHIVE_PUNCHED_MARKER as _GAUSSIAN_ARCHIVE_PUNCHED_MARKER,
    gaussian_archive_only_completion as _gaussian_archive_only_completion,
    gaussian_contains_fatal_marker as _gaussian_contains_fatal_marker,
    gaussian_output_ends_with_normal_termination as _gaussian_output_ends_with_normal_termination,
)


_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[DdEe][-+]?\d+)?"
_GAUSSIAN_OPTIMIZATION_COMPLETION = re.compile(
    r"Optimization completed(?: on the basis of negligible forces)?\."
)
_PERIODIC_SYMBOLS = (
    "",
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
)
_ATOMIC_NUMBERS = {symbol.lower(): number for number, symbol in enumerate(_PERIODIC_SYMBOLS) if symbol}
# Gaussian formatted checkpoints store Cartesian coordinates in bohr.  The
# public AtomicStructure contract (and ASE) uses Angstrom, so conversion is
# explicit at this backend boundary.  Keeping this constant here also avoids
# silently mixing the Hessian's atomic-unit coordinates with caller geometry.
BOHR_TO_ANGSTROM = 0.529177210903

class GaussianExecutionError(RuntimeError):
    """保存 Gaussian 外部进程状态，供 Action worker 写入安全诊断。"""

    def __init__(
        self,
        message: str,
        *,
        returncode: int | None,
        termination: str,
        archive_warning: bool = False,
        timeout: bool = False,
    ) -> None:
        self.returncode = returncode
        self.inner_returncode = returncode
        self.termination = termination
        self.archive_warning = archive_warning
        self.timeout = timeout
        detail = (
            f" (Gaussian inner returncode={returncode!r}; termination={termination}; "
            f"archive_warning={archive_warning}; timeout={timeout})"
        )
        super().__init__(message + detail)


def _is_finite_gaussian_value(value: Any) -> bool:
    """检查扫描记录中的标量能否安全转换为有限浮点数。"""

    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _gaussian_scan_has_complete_finite_points(
    points: list[dict[str, Any]], requested_points: int, stdout: str
) -> bool:
    """核对扫描点数、索引、有限数值、坐标约束和原生 summary。"""

    if len(points) != requested_points or [
        point.get("point_index") for point in points
    ] != list(range(requested_points)):
        return False
    if "Summary of Optimized Potential Surface Scan" not in stdout:
        return False
    scalar_names = (
        "target_value",
        "coordinate_value",
        "target_deviation",
        "target_tolerance",
        "energy_hartree",
        "relative_energy_kcal_mol",
    )
    for point in points:
        if not isinstance(point, dict):
            return False
        if point.get("coordinate_constraint_satisfied") is not True:
            return False
        if any(
            not _is_finite_gaussian_value(point.get(name)) for name in scalar_names
        ):
            return False
        structure = point.get("structure")
        atoms = structure.get("atoms") if isinstance(structure, dict) else None
        if not isinstance(atoms, list) or not atoms:
            return False
        for atom in atoms:
            coordinates = (
                atom.get("position_angstrom") if isinstance(atom, dict) else None
            )
            if (
                not isinstance(coordinates, (list, tuple))
                or len(coordinates) != 3
                or any(not _is_finite_gaussian_value(value) for value in coordinates)
            ):
                return False
    return True


def _finite_float(value: Any, name: str, *, positive: bool = False) -> float:
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        qualifier = "positive and " if positive else ""
        raise ValueError(f"{name} must be {qualifier}finite")
    return result


def _safe_keyword(value: Any, name: str, pattern: str = r"[A-Za-z0-9+*(),./_=-]+") -> str:
    text = str(value).strip()
    if not text or not re.fullmatch(pattern, text):
        raise ValueError(f"{name} contains unsupported characters")
    return text


def _gaussian_scf_or_dft_method(value: Any) -> str:
    method = _safe_keyword(value, "Gaussian method")
    unsupported_fragments = (
        "MP2", "MP3", "MP4", "CCSD", "QCISD", "CASSCF", "CASPT",
        "CISD", "EOM", "B2PLYP", "DSD", "CBS-", "G4", "W1",
    )
    if any(fragment in method.upper() for fragment in unsupported_fragments):
        raise ValueError(
            "The Gaussian adapter currently reports SCF/DFT total energies only; "
            "correlated/composite methods are not accepted until their energies have a typed parser"
        )
    return method


def _gaussian_optimization_convergence(value: Any) -> str:
    """Return a canonical Gaussian Opt convergence keyword.

    pysisyphus uses labels such as ``gau`` and ``gau_loose``. Passing those
    labels through to Gaussian produces an invalid route and can make the
    licensed executable abort with a core dump instead of a useful input
    error, so keep this backend boundary deliberately strict.
    """

    text = _safe_keyword(
        value,
        "Gaussian optimization convergence",
        r"[A-Za-z0-9_-]+",
    )
    canonical = {
        "loose": "Loose",
        "tight": "Tight",
        "verytight": "VeryTight",
    }
    try:
        return canonical[text.casefold()]
    except KeyError as exc:
        raise ValueError(
            "Gaussian optimization_convergence must be Loose, Tight, or VeryTight"
        ) from exc


def _gaussian_boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _gaussian_bounded_integer(
    value: Any, name: str, *, minimum: int = 1, maximum: int = 1000
) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if result != value or result < minimum or result > maximum:
        raise ValueError(f"{name} must be an integer from {minimum} to {maximum}")
    return result


def _gaussian_choice(value: Any, name: str, choices: dict[str, str]) -> str:
    normalized = str(value).strip().casefold()
    try:
        return choices[normalized]
    except KeyError as exc:
        raise ValueError(f"{name} must be one of {sorted(choices)}") from exc


def _gaussian_solvent_keyword(value: Any) -> str:
    text = str(value).strip()
    plain = r"[A-Za-z0-9_.+-]+"
    if re.fullmatch(plain, text):
        if text.casefold() == "ethylacetate":
            raise ValueError(
                "Gaussian ethyl acetate solvent alias must be EthylEthanoate"
            )
        return text
    parts = text.split(",")
    locants = parts[:-1]
    terminal = parts[-1] if parts else ""
    if (
        len(parts) >= 2
        and all(re.fullmatch(r"(?:\d+[A-Za-z]?|[NOSP])", item) for item in locants)
        and re.fullmatch(r"[A-Za-z0-9_.+]*-[A-Za-z0-9_.+-]+", terminal)
    ):
        return text
    raise ValueError(
        "Gaussian solvent must be a plain alias or a comma-separated numeric/N/O/S/P "
        "locant chemical name"
    )


def _gaussian_electronic_state(
    structure: dict[str, Any], method: dict[str, Any]
) -> tuple[int, int]:
    charge = int(method.get("charge", structure.get("charge", 0)))
    multiplicity = int(method.get("multiplicity", structure.get("multiplicity", 1)))
    if multiplicity < 1:
        raise ValueError("Gaussian multiplicity must be at least one")
    return charge, multiplicity


def _gaussian_structure_with_effective_state(
    structure: dict[str, Any], method: dict[str, Any]
) -> dict[str, Any]:
    result = deepcopy(structure)
    result["charge"], result["multiplicity"] = _gaussian_electronic_state(
        structure, method
    )
    return result


def _gaussian_scan_definition(
    structure: dict[str, Any], settings: dict[str, Any]
) -> dict[str, Any]:
    coordinate_type = _gaussian_choice(
        settings["coordinate_type"],
        "Gaussian scan coordinate_type",
        {"bond": "bond", "angle": "angle", "dihedral": "dihedral"},
    )
    expected_indices = {"bond": 2, "angle": 3, "dihedral": 4}[coordinate_type]
    raw_indices = settings["atom_indices"]
    if not isinstance(raw_indices, (list, tuple)):
        raise ValueError("Gaussian scan atom_indices must be an array of zero-based integers")
    indices: list[int] = []
    for value in raw_indices:
        if isinstance(value, bool):
            raise ValueError("Gaussian scan atom_indices must contain integers")
        try:
            index = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("Gaussian scan atom_indices must contain integers") from exc
        if index != value:
            raise ValueError("Gaussian scan atom_indices must contain integers")
        indices.append(index)
    if len(indices) != expected_indices or len(set(indices)) != len(indices):
        raise ValueError(
            f"Gaussian {coordinate_type} scan requires {expected_indices} unique atom_indices"
        )
    atom_count = len(structure.get("atoms") or [])
    if any(index < 0 or index >= atom_count for index in indices):
        raise ValueError("Gaussian scan atom_indices contains an index outside the structure")

    value_unit = _gaussian_choice(
        settings["value_unit"],
        "Gaussian scan value_unit",
        {"angstrom": "angstrom", "degree": "degree", "radian": "radian"},
    )
    start = _finite_float(settings["start_value"], "Gaussian scan start_value")
    end = _finite_float(settings["end_value"], "Gaussian scan end_value")
    if start == end:
        raise ValueError("Gaussian scan start_value and end_value must differ")
    native_start = start
    native_end = end
    if coordinate_type == "bond":
        if value_unit != "angstrom":
            raise ValueError("Gaussian bond scans require value_unit=angstrom")
        if start <= 0 or end <= 0:
            raise ValueError("Gaussian bond scan distances must be positive")
    else:
        if value_unit == "radian":
            native_start = math.degrees(start)
            native_end = math.degrees(end)
        elif value_unit != "degree":
            raise ValueError("Gaussian angle/dihedral scans require degree or radian values")
        if coordinate_type == "angle" and not (
            0.0 < native_start < 180.0 and 0.0 < native_end < 180.0
        ):
            raise ValueError("Gaussian angle scan values must lie strictly between 0 and 180 degrees")
    steps = _gaussian_bounded_integer(
        settings["steps"], "Gaussian scan steps", maximum=1000
    )
    increment = (native_end - native_start) / steps
    return {
        "coordinate_type": coordinate_type,
        "atom_indices": indices,
        "native_coordinate_code": {"bond": "B", "angle": "A", "dihedral": "D"}[
            coordinate_type
        ],
        "start_value": start,
        "end_value": end,
        "value_unit": value_unit,
        "native_start_value": native_start,
        "native_end_value": native_end,
        "native_increment": increment,
        "steps": steps,
    }


def _gaussian_modredundant_number(value: float) -> str:
    """Keep a decimal point so Gaussian does not parse a value as another atom index."""

    rendered = f"{value:.12f}".rstrip("0")
    return rendered if rendered.endswith(".") is False else rendered + "0"


def _molecular_structure(value: Any) -> tuple[dict[str, Any], list[str], list[list[float]]]:
    structure = structure_dict(value)
    if any(bool(flag) for flag in structure.get("pbc", [])):
        raise ValueError("Gaussian/GAMESS molecular adapters require a non-periodic structure")
    symbols, coordinates = atoms_and_coordinates(structure)
    return structure, symbols, coordinates


def _render_gaussian(
    action_id: str,
    structure_value: Any,
    method: dict[str, Any],
    settings: dict[str, Any],
    resource_limits: dict[str, Any] | None = None,
) -> str:
    """Render one bounded Gaussian job without accepting an arbitrary route deck."""

    structure, symbols, coordinates = _molecular_structure(structure_value)
    method_name = _gaussian_scf_or_dft_method(method["method"])
    basis = _safe_keyword(method["basis"], "Gaussian basis")
    if action_id == "optimize_geometry":
        max_steps = int(settings["max_steps"])
        if max_steps < 1:
            raise ValueError("Gaussian max_steps must be positive")
        task_keyword = (
            "Opt=("
            + _gaussian_optimization_convergence(
                settings["optimization_convergence"]
            )
            + f",MaxCycles={max_steps})"
        )
    elif action_id == "locate_transition_state":
        max_step = _gaussian_bounded_integer(
            settings["max_step"], "Gaussian TS max_step", maximum=100
        )
        options = ["TS"]
        if _gaussian_boolean(settings["no_eigen_test"], "Gaussian TS no_eigen_test"):
            options.append("NoEigenTest")
        if _gaussian_boolean(
            settings["calculate_all_force_constants"],
            "Gaussian TS calculate_all_force_constants",
        ):
            options.append("CalcAll")
        elif settings.get("initial_force_constants") is not None:
            options.append(
                _gaussian_choice(
                    settings["initial_force_constants"],
                    "Gaussian TS initial_force_constants",
                    {"calcfc": "CalcFC"},
                )
            )
        options.append(f"MaxStep={max_step}")
        if settings.get("max_cycles") is not None:
            max_cycles = _gaussian_bounded_integer(
                settings["max_cycles"], "Gaussian TS max_cycles"
            )
            options.append(f"MaxCycles={max_cycles}")
        task_keyword = "Opt=(" + ",".join(options) + ")"
    elif action_id == "scan_reaction_coordinates":
        scan = _gaussian_scan_definition(structure, settings)
        options = ["ModRedundant"]
        optimization_convergence = _gaussian_choice(
            settings["optimization_convergence"],
            "Gaussian scan optimization_convergence",
            {
                "default": "",
                "loose": "Loose",
                "tight": "Tight",
                "verytight": "VeryTight",
            },
        )
        if optimization_convergence:
            options.append(optimization_convergence)
        options.append(
            f"MaxCycles={_gaussian_bounded_integer(settings['max_cycles'], 'Gaussian scan max_cycles')}"
        )
        force_constants = _gaussian_choice(
            settings["force_constant_strategy"],
            "Gaussian scan force_constant_strategy",
            {"default": "", "calcfc": "CalcFC", "calcall": "CalcAll"},
        )
        if force_constants:
            options.append(force_constants)
        task_keyword = "Opt=(" + ",".join(options) + ")"
    elif action_id == "trace_intrinsic_reaction_coordinate":
        direction = _gaussian_choice(
            settings["direction"],
            "Gaussian IRC direction",
            {"forward": "Forward", "backward": "Reverse"},
        )
        force_constants = _gaussian_choice(
            settings["initial_force_constants"],
            "Gaussian IRC initial_force_constants",
            {"calcfc": "CalcFC"},
        )
        integrator = _gaussian_choice(
            settings["integrator"],
            "Gaussian IRC integrator",
            {"lqa": "LQA"},
        )
        max_points = _gaussian_bounded_integer(
            settings["max_points"], "Gaussian IRC max_points"
        )
        recalc = _gaussian_bounded_integer(
            settings["recalc"], "Gaussian IRC recalc"
        )
        task_keyword = (
            "IRC=("
            f"{force_constants},{integrator},MaxPoints={max_points},"
            f"Recalc={recalc},{direction})"
        )
    elif action_id == "calculate_excited_states":
        _charge, multiplicity = _gaussian_electronic_state(structure, method)
        if multiplicity != 1 or re.match(r"^(?:U|RO)", method_name, re.IGNORECASE):
            raise ValueError(
                "The Gaussian vertical excited-state adapter currently requires a "
                "restricted closed-shell singlet reference (multiplicity=1 and no U/RO method prefix)"
            )
        excited_state_method = _gaussian_choice(
            method["excited_state_method"],
            "Gaussian excited_state_method",
            {"tddft": "tddft", "tda": "tda"},
        )
        number_of_states = _gaussian_bounded_integer(
            settings["number_of_states"],
            "Gaussian number_of_states",
            maximum=100,
        )
        spin_symmetry = _gaussian_choice(
            settings["spin_symmetry"],
            "Gaussian spin_symmetry",
            {"singlet": "Singlets", "triplet": "Triplets"},
        )
        options = [f"NStates={number_of_states}", spin_symmetry]
        excited_state_keyword = "TDA" if excited_state_method == "tda" else "TD"
        task_keyword = excited_state_keyword + "=(" + ",".join(options) + ")"
    elif action_id == "calculate_bond_orders":
        _gaussian_choice(
            method["bond_order_definition"],
            "Gaussian bond_order_definition",
            {"wiberg_nao": "wiberg_nao"},
        )
        minimum_bond_order = _finite_float(
            settings["minimum_bond_order"], "Gaussian minimum_bond_order"
        )
        if minimum_bond_order < 0 or minimum_bond_order > 10:
            raise ValueError("Gaussian minimum_bond_order must be from 0 to 10")
        task_keyword = "SP Pop=NBORead"
    else:
        task_keyword = {
            "calculate_energy": "SP",
            "calculate_hessian": "Freq",
            "calculate_dipole_moment": "SP",
        }[action_id]
    scf_keyword = _gaussian_choice(
        settings["scf_convergence"],
        "Gaussian SCF convergence",
        {
            "default": "",
            "loose": "Loose",
            "tight": "Tight",
            "verytight": "VeryTight",
        },
    )
    route = [f"{method_name}/{basis}", task_keyword]
    if scf_keyword:
        route.append(f"SCF={scf_keyword}")
    if method.get("dispersion"):
        dispersion = _gaussian_choice(
            method["dispersion"],
            "Gaussian dispersion",
            {"gd2": "GD2", "gd3": "GD3", "gd3bj": "GD3BJ"},
        )
        route.append(f"EmpiricalDispersion={dispersion}")
    if method.get("integration_grid"):
        integration_grid = _gaussian_choice(
            method["integration_grid"],
            "Gaussian integration_grid",
            {
                "finegrid": "FineGrid",
                "ultrafine": "UltraFine",
                "superfinegrid": "SuperFineGrid",
            },
        )
        route.append(f"Int={integration_grid}")
    if method.get("solvation_model") is not None:
        model = str(method["solvation_model"]).strip().casefold()
        model_keyword = {
            "pcm": "PCM", "iefpcm": "IEFPCM", "cpcm": "CPCM", "smd": "SMD"
        }.get(model)
        if model_keyword is None:
            raise ValueError("Gaussian solvation_model must be pcm, iefpcm, cpcm, or smd")
        if not method.get("solvent"):
            raise ValueError("Gaussian solvation_model requires method_spec.solvent")
        solvent = _gaussian_solvent_keyword(method["solvent"])
        route.append(f"SCRF=({model_keyword},Solvent={solvent})")
    if action_id == "calculate_hessian" and settings.get("temperature_kelvin") is not None:
        temperature = _finite_float(
            settings["temperature_kelvin"],
            "Gaussian temperature_kelvin",
            positive=True,
        )
        route.append(f"Temperature={temperature:.12g}")
    if action_id == "scan_reaction_coordinates" and _gaussian_boolean(
        settings["disable_symmetry"], "Gaussian scan disable_symmetry"
    ):
        route.append("NoSymm")
    limits = resource_limits or {}
    cores = max(1, int(limits.get("cpu_cores") or 1))
    memory_mb = max(128, int(limits.get("memory_mb") or 1000))
    charge, multiplicity = _gaussian_electronic_state(structure, method)
    lines = [
        "%Chk=job.chk",
        f"%NProcShared={cores}",
        f"%Mem={memory_mb}MB",
        "#P " + " ".join(route),
        "",
        "ResearchChemBench typed atomic action",
        "",
        f"{charge} {multiplicity}",
    ]
    lines.extend(
        f"{symbol} {row[0]:.12f} {row[1]:.12f} {row[2]:.12f}"
        for symbol, row in zip(symbols, coordinates)
    )
    lines.append("")
    if action_id == "scan_reaction_coordinates":
        scan = _gaussian_scan_definition(structure, settings)
        native_indices = " ".join(str(index + 1) for index in scan["atom_indices"])
        lines.extend(
            [
                f"{scan['native_coordinate_code']} {native_indices} "
                f"{_gaussian_modredundant_number(scan['native_start_value'])}",
                f"{scan['native_coordinate_code']} {native_indices} S {scan['steps']} "
                f"{_gaussian_modredundant_number(scan['native_increment'])}",
                "",
            ]
        )
    elif action_id == "calculate_bond_orders":
        lines.extend(["$NBO BNDIDX $END", ""])
    else:
        lines.append("")
    return "\n".join(lines)


def _parse_gaussian_energy(text: str) -> float | None:
    matches = re.findall(rf"SCF Done:\s+E\([^\n=]+\)\s*=\s*({_FLOAT})", text)
    return float(matches[-1].replace("D", "E")) if matches else None


def _parse_gaussian_dipole(text: str) -> list[float] | None:
    matches = re.findall(
        rf"Dipole moment \(field-independent basis, Debye\):\s*\n\s*"
        rf"X=\s*({_FLOAT})\s+Y=\s*({_FLOAT})\s+Z=\s*({_FLOAT})",
        text,
    )
    if not matches:
        return None
    return [float(value.replace("D", "E")) for value in matches[-1]]


def _parse_gaussian_excited_states(text: str) -> list[dict[str, Any]]:
    header = re.compile(
        rf"Excited State\s+(?P<index>\d+):\s+(?P<label>\S+)\s+"
        rf"(?P<energy>{_FLOAT})\s+eV\s+(?P<wavelength>{_FLOAT})\s+nm\s+"
        rf"f=(?P<oscillator>{_FLOAT})(?:\s+<S\*\*2>=(?P<spin_squared>{_FLOAT}))?",
        flags=re.IGNORECASE,
    )
    matches = list(header.finditer(text))
    states: list[dict[str, Any]] = []
    for position, match in enumerate(matches):
        stop = matches[position + 1].start() if position + 1 < len(matches) else len(text)
        segment = text[match.end() : stop]
        label = match.group("label")
        spin_label, separator, symmetry = label.partition("-")
        energy_ev = float(match.group("energy").replace("D", "E"))
        transitions: list[dict[str, Any]] = []
        for transition in re.finditer(
            rf"^\s*(?P<source>\d+)(?P<source_spin>[AB])?\s*->\s*"
            rf"(?P<target>\d+)(?P<target_spin>[AB])?\s+"
            rf"(?P<coefficient>{_FLOAT})\s*$",
            segment,
            flags=re.MULTILINE | re.IGNORECASE,
        ):
            source_spin = transition.group("source_spin")
            target_spin = transition.group("target_spin")
            record: dict[str, Any] = {
                "source_orbital_index": int(transition.group("source")) - 1,
                "target_orbital_index": int(transition.group("target")) - 1,
                "coefficient": float(
                    transition.group("coefficient").replace("D", "E")
                ),
            }
            if source_spin:
                record["source_spin_channel"] = source_spin.lower()
            if target_spin:
                record["target_spin_channel"] = target_spin.lower()
            transitions.append(record)
        state: dict[str, Any] = {
            "state_index": int(match.group("index")),
            "state_label": label,
            "spin_symmetry": spin_label.casefold(),
            "spatial_symmetry": symmetry if separator else None,
            "energy_hartree": energy_ev / 27.211386245988,
            "energy_ev": energy_ev,
            "wavenumber_cm1": energy_ev * 8065.544005,
            "wavelength_nm": float(match.group("wavelength").replace("D", "E")),
            "oscillator_strength": float(
                match.group("oscillator").replace("D", "E")
            ),
            "transitions": transitions,
        }
        if match.group("spin_squared") is not None:
            state["spin_squared"] = float(
                match.group("spin_squared").replace("D", "E")
            )
        states.append(state)
    return states


def _parse_gaussian_wiberg_matrix(
    text: str, symbols: list[str]
) -> tuple[list[list[float]], str | None]:
    block = re.search(
        r"Wiberg bond index matrix in the NAO basis:\s*(?P<body>.*?)"
        r"Wiberg bond index, Totals by atom:",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if block is None:
        raise RuntimeError("Could not find the Gaussian NBO Wiberg/NAO bond-index matrix")
    atom_count = len(symbols)
    values: list[list[float | None]] = [
        [None for _ in range(atom_count)] for _ in range(atom_count)
    ]
    columns: list[int] = []
    parsed_symbols: dict[int, str] = {}
    for line in block.group("body").splitlines():
        fields = line.split()
        if fields and fields[0].casefold() == "atom" and all(
            field.isdigit() for field in fields[1:]
        ):
            columns = [int(field) - 1 for field in fields[1:]]
            if any(column < 0 or column >= atom_count for column in columns):
                raise RuntimeError("Gaussian Wiberg matrix contains an out-of-range column")
            continue
        row_match = re.match(r"^\s*(\d+)\.\s+([A-Za-z]+)\s+(.*)$", line)
        if row_match is None or not columns:
            continue
        row = int(row_match.group(1)) - 1
        if row < 0 or row >= atom_count:
            raise RuntimeError("Gaussian Wiberg matrix contains an out-of-range row")
        row_values = row_match.group(3).split()
        if len(row_values) != len(columns):
            raise RuntimeError("Gaussian Wiberg matrix row is truncated")
        parsed_symbols[row] = row_match.group(2)
        for column, token in zip(columns, row_values):
            values[row][column] = float(token.replace("D", "E"))
    if set(parsed_symbols) != set(range(atom_count)) or any(
        value is None for row in values for value in row
    ):
        raise RuntimeError("Gaussian Wiberg matrix is incomplete")
    for index, symbol in enumerate(symbols):
        if parsed_symbols[index].casefold() != symbol.casefold():
            raise RuntimeError("Gaussian Wiberg atom order does not match the input structure")
    matrix = [[float(value) for value in row] for row in values]
    for row in range(atom_count):
        for column in range(row):
            if abs(matrix[row][column] - matrix[column][row]) > 5.0e-4:
                raise RuntimeError("Gaussian Wiberg matrix is not symmetric within print precision")
    version_match = re.search(
        r"Gaussian NBO Version\s+([^*\s]+)", text, flags=re.IGNORECASE
    )
    return matrix, version_match.group(1) if version_match else None


def _coordinate_value(
    coordinates: list[list[float]], coordinate_type: str, indices: list[int]
) -> float:
    def subtract(left: list[float], right: list[float]) -> list[float]:
        return [left[axis] - right[axis] for axis in range(3)]

    def dot(left: list[float], right: list[float]) -> float:
        return sum(left[axis] * right[axis] for axis in range(3))

    def norm(vector: list[float]) -> float:
        return math.sqrt(dot(vector, vector))

    def cross(left: list[float], right: list[float]) -> list[float]:
        return [
            left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0],
        ]

    if coordinate_type == "bond":
        return norm(subtract(coordinates[indices[0]], coordinates[indices[1]]))
    if coordinate_type == "angle":
        left = subtract(coordinates[indices[0]], coordinates[indices[1]])
        right = subtract(coordinates[indices[2]], coordinates[indices[1]])
        denominator = norm(left) * norm(right)
        if denominator <= 1.0e-14:
            raise RuntimeError("Cannot evaluate a Gaussian scan angle with a zero-length arm")
        cosine = max(-1.0, min(1.0, dot(left, right) / denominator))
        return math.degrees(math.acos(cosine))

    first = subtract(coordinates[indices[1]], coordinates[indices[0]])
    second = subtract(coordinates[indices[2]], coordinates[indices[1]])
    third = subtract(coordinates[indices[3]], coordinates[indices[2]])
    first_normal = cross(first, second)
    second_normal = cross(second, third)
    second_norm = norm(second)
    if norm(first_normal) <= 1.0e-14 or norm(second_normal) <= 1.0e-14 or second_norm <= 1.0e-14:
        raise RuntimeError("Cannot evaluate a Gaussian scan dihedral for collinear atoms")
    second_unit = [value / second_norm for value in second]
    auxiliary = cross(first_normal, second_unit)
    # Gaussian's D(i,j,k,l) convention has the opposite sign from the direct
    # atan2 expression above.
    return -math.degrees(
        math.atan2(dot(auxiliary, second_normal), dot(first_normal, second_normal))
    )


def _scan_coordinate_near_target(
    value: float, target: float, coordinate_type: str, value_unit: str
) -> tuple[float, float, float]:
    if coordinate_type == "dihedral":
        period = 2.0 * math.pi if value_unit == "radian" else 360.0
        value += period * round((target - value) / period)
    deviation = abs(value - target)
    tolerance = (
        0.005
        if coordinate_type == "bond"
        else (math.radians(0.05) if value_unit == "radian" else 0.05)
    )
    return value, deviation, tolerance


def _parse_gaussian_relaxed_scan(
    text: str,
    symbols: list[str],
    original: dict[str, Any],
    scan: dict[str, Any],
) -> list[dict[str, Any]]:
    markers = list(_GAUSSIAN_OPTIMIZATION_COMPLETION.finditer(text))
    records: list[dict[str, Any]] = []
    previous_stop = 0
    for point_index, marker in enumerate(markers):
        segment = text[previous_stop : marker.start()]
        previous_stop = marker.end()
        geometry = _parse_gaussian_geometry(segment, symbols)
        energy = _parse_gaussian_energy(segment)
        if geometry is None or energy is None:
            continue
        geometry = _preserve_gaussian_structure_metadata(geometry, original)
        coordinates = [
            [float(value) for value in atom["position_angstrom"]]
            for atom in geometry["atoms"]
        ]
        target = scan["start_value"] + (
            scan["end_value"] - scan["start_value"]
        ) * point_index / scan["steps"]
        value = _coordinate_value(
            coordinates, scan["coordinate_type"], scan["atom_indices"]
        )
        if scan["coordinate_type"] != "bond" and scan["value_unit"] == "radian":
            value = math.radians(value)
        value, deviation, tolerance = _scan_coordinate_near_target(
            value, target, scan["coordinate_type"], scan["value_unit"]
        )
        records.append(
            {
                "point_index": point_index,
                "target_value": target,
                "coordinate_value": value,
                "target_deviation": deviation,
                "target_tolerance": tolerance,
                "coordinate_constraint_satisfied": deviation <= tolerance,
                "energy_hartree": energy,
                "structure": geometry,
                "converged": True,
            }
        )
    if records:
        minimum = min(record["energy_hartree"] for record in records)
        for record in records:
            record["relative_energy_kcal_mol"] = (
                record["energy_hartree"] - minimum
            ) * 627.5094740631
    return records


def _parse_gaussian_geometries(text: str, symbols: list[str]) -> list[dict[str, Any]]:
    lines = text.splitlines()
    blocks: list[list[list[float]]] = []
    for index, line in enumerate(lines):
        if "Standard orientation:" not in line and "Input orientation:" not in line:
            continue
        cursor = index + 1
        separators = 0
        while cursor < len(lines):
            if set(lines[cursor].strip()) == {"-"}:
                separators += 1
                cursor += 1
                if separators == 2:
                    break
            else:
                cursor += 1
        coordinates: list[list[float]] = []
        while cursor < len(lines):
            stripped = lines[cursor].strip()
            if set(stripped) == {"-"}:
                break
            parts = stripped.split()
            if len(parts) >= 6 and parts[0].isdigit():
                try:
                    coordinates.append([float(parts[3]), float(parts[4]), float(parts[5])])
                except ValueError:
                    coordinates = []
                    break
            cursor += 1
        if len(coordinates) == len(symbols):
            blocks.append(coordinates)
    return [
        {
            "atoms": [
                {"element": symbol, "position_angstrom": row}
                for symbol, row in zip(symbols, coordinates)
            ],
            "pbc": [False, False, False],
        }
        for coordinates in blocks
    ]


def _parse_gaussian_geometry(text: str, symbols: list[str]) -> dict[str, Any] | None:
    geometries = _parse_gaussian_geometries(text, symbols)
    return geometries[-1] if geometries else None


def _preserve_gaussian_structure_metadata(
    optimized: dict[str, Any], original: dict[str, Any]
) -> dict[str, Any]:
    """Carry molecular state and graph identity through a coordinate-only optimization."""

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
                atom[name] = deepcopy(value)
    return result


def _parse_gaussian_irc_summary(text: str) -> list[dict[str, float | int]]:
    return _shared_parse_gaussian_irc_summary(text)


def _parse_gaussian_irc_points(
    text: str,
    symbols: list[str],
    original: dict[str, Any],
    requested_direction: str,
) -> list[dict[str, Any]]:
    return _shared_parse_gaussian_irc_points(
        text,
        symbols,
        original,
        requested_direction,
        parse_geometries=_parse_gaussian_geometries,
        parse_energy=_parse_gaussian_energy,
        preserve_structure_metadata=_preserve_gaussian_structure_metadata,
    )


def _parse_gaussian_irc_termination(
    text: str, requested_direction: str
) -> tuple[bool, str]:
    return _shared_parse_gaussian_irc_termination(text, requested_direction)


def _parse_gaussian_fchk_array(
    lines: list[str], label: str, value_type: str
) -> list[int] | list[float]:
    """Read one bounded array from a Gaussian formatted checkpoint.

    FCHK ``R`` values, including ``Current cartesian coordinates`` and
    ``Cartesian Force Constants``, are written in Fortran ``D`` notation in
    some Gaussian revisions.  Parse only the declared number of values and
    reject a truncated/overfull field instead of accidentally consuming the
    next named field.  The coordinate unit is documented by Gaussian as bohr;
    conversion is deliberately done by the structure helper below.
    """

    header = re.compile(
        rf"^\s*{re.escape(label)}\s+{re.escape(value_type)}\s+N=\s*(\d+)\s*$"
    )
    for index, line in enumerate(lines):
        match = header.match(line)
        if match is None:
            continue
        count = int(match.group(1))
        values: list[int] | list[float] = []
        for following in lines[index + 1 :]:
            tokens = following.split()
            if len(values) + len(tokens) > count:
                raise ValueError(
                    f"Gaussian FCHK {label} array contains more values than declared"
                )
            for token in tokens:
                if value_type == "I":
                    try:
                        values.append(int(token))
                    except ValueError as exc:
                        raise ValueError(
                            f"Gaussian FCHK {label} array contains a non-integer value"
                        ) from exc
                else:
                    try:
                        value = float(token.replace("D", "E").replace("d", "e"))
                    except ValueError as exc:
                        raise ValueError(
                            f"Gaussian FCHK {label} array contains a non-numeric value"
                        ) from exc
                    if not math.isfinite(value):
                        raise ValueError(
                            f"Gaussian FCHK {label} array contains a non-finite value"
                        )
                    values.append(value)
            if len(values) == count:
                return values
        raise ValueError(f"Gaussian FCHK {label} array is truncated")
    raise ValueError(f"Gaussian FCHK is missing the {label} array")


def _parse_gaussian_fchk_scalar(lines: list[str], label: str) -> int:
    header = re.compile(rf"^\s*{re.escape(label)}\s+I\s+(-?\d+)\s*$")
    for line in lines:
        match = header.match(line)
        if match is not None:
            return int(match.group(1))
    raise ValueError(f"Gaussian FCHK is missing the {label} scalar")


def _parse_gaussian_fchk_structure(
    path: Path, original: dict[str, Any]
) -> dict[str, Any]:
    """Return the molecular structure in the exact FCHK/Hessian Cartesian frame.

    Gaussian may rotate/translate a molecule into its standard orientation
    before writing ``job.fchk``.  The Hessian is expressed in that frame.  We
    therefore use the fchk ``Atomic numbers`` ordering and
    ``Current cartesian coordinates`` (bohr), convert coordinates to the
    public Angstrom unit, and then copy graph/mapping/electronic metadata from
    the caller's structure.  No inverse rotation is attempted because the
    Hessian itself is intentionally left in the Gaussian frame.
    """

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    atom_count = _parse_gaussian_fchk_scalar(lines, "Number of atoms")
    atomic_numbers_raw = _parse_gaussian_fchk_array(lines, "Atomic numbers", "I")
    coordinate_values_raw = _parse_gaussian_fchk_array(
        lines, "Current cartesian coordinates", "R"
    )
    atomic_numbers = [int(value) for value in atomic_numbers_raw]
    coordinate_values = [float(value) for value in coordinate_values_raw]
    if atom_count < 1:
        raise ValueError("Gaussian FCHK Number of atoms must be positive")
    if len(atomic_numbers) != atom_count:
        raise ValueError(
            "Gaussian FCHK Atomic numbers count does not match Number of atoms"
        )
    if len(coordinate_values) != 3 * atom_count:
        raise ValueError(
            "Gaussian FCHK Current cartesian coordinates count does not match Number of atoms"
        )
    if any(number < 1 or number >= len(_PERIODIC_SYMBOLS) for number in atomic_numbers):
        raise ValueError("Gaussian FCHK contains an unsupported atomic number")

    original_symbols, _original_coordinates = atoms_and_coordinates(original)
    if len(original_symbols) != atom_count:
        raise ValueError(
            "Gaussian FCHK atom count does not match the requested structure"
        )
    expected_numbers = [_ATOMIC_NUMBERS.get(symbol.casefold()) for symbol in original_symbols]
    if expected_numbers != atomic_numbers:
        raise ValueError(
            "Gaussian FCHK Atomic numbers/order do not match the requested structure"
        )

    coordinates_angstrom = [
        [
            coordinate_values[offset] * BOHR_TO_ANGSTROM,
            coordinate_values[offset + 1] * BOHR_TO_ANGSTROM,
            coordinate_values[offset + 2] * BOHR_TO_ANGSTROM,
        ]
        for offset in range(0, len(coordinate_values), 3)
    ]
    parsed = {
        "atoms": [
            {
                "element": _PERIODIC_SYMBOLS[number],
                "position_angstrom": coordinates,
            }
            for number, coordinates in zip(atomic_numbers, coordinates_angstrom)
        ],
        "pbc": [False, False, False],
    }
    result = _preserve_gaussian_structure_metadata(parsed, original)
    result["coordinate_frame"] = "gaussian_fchk_current_cartesian"
    result["coordinate_unit"] = "angstrom"
    return result


def _parse_gaussian_fchk_hessian(path: Path, expected_dimension: int) -> list[list[float]] | None:
    try:
        values_raw = _parse_gaussian_fchk_array(
            path.read_text(encoding="utf-8", errors="replace").splitlines(),
            "Cartesian Force Constants",
            "R",
        )
    except (OSError, ValueError):
        return None
    values = [float(value) for value in values_raw]
    count = len(values)
    expected_count = expected_dimension * (expected_dimension + 1) // 2
    if count != expected_count or len(values) != expected_count:
        return None
    matrix = [[0.0 for _ in range(expected_dimension)] for _ in range(expected_dimension)]
    cursor = 0
    for row in range(expected_dimension):
        for column in range(row + 1):
            matrix[row][column] = values[cursor]
            matrix[column][row] = values[cursor]
            cursor += 1
    return matrix


def _remove_generated_gaussian_core_dumps(directory: Path) -> list[str]:
    """Remove only core dumps produced inside this isolated Gaussian job directory."""

    removed = []
    for path in directory.glob("core.*"):
        if not path.is_file():
            continue
        try:
            path.unlink()
        except OSError:
            continue
        removed.append(path.name)
    return sorted(removed)


def _gaussian_qperr_message(
    action_id: str,
    method: dict[str, Any],
    evidence: str,
) -> str:
    """根据 Gaussian 的 QPErr 输出给出不武断归因的字段级修订提示。"""

    route_contains_solvent = bool(
        re.search(r"\b(?:SCRF|SOLVENT)\b", evidence, flags=re.IGNORECASE)
    )
    route_contains_scan = action_id == "scan_reaction_coordinates" and bool(
        re.search(r"\b(?:OPT|MODREDUNDANT)\b", evidence, flags=re.IGNORECASE)
    )
    if route_contains_solvent and route_contains_scan:
        return (
            "Gaussian route 语法无效（QPErr）；输出中的 route 同时含 SCRF/Solvent 和 "
            "Opt/ModRedundant，无法仅凭该信息定位单一字段。请同时核对 "
            f"method_spec.solvation_model={method.get('solvation_model')!r}、"
            f"method_spec.solvent={method.get('solvent')!r}，以及 "
            "action_settings.coordinate_type、action_settings.atom_indices、"
            "action_settings.start_value、action_settings.end_value、"
            "action_settings.value_unit 和 action_settings.steps；所有值必须使用 Gaussian "
            "原生关键字和扫描语法。"
        )
    if route_contains_solvent:
        return (
            "Gaussian route 语法无效（QPErr）；输出中的 route 含 SCRF/Solvent，"
            "这不能单独证明 method/basis 有误，请核对 "
            f"method_spec.solvation_model={method.get('solvation_model')!r}、"
            f"method_spec.solvent={method.get('solvent')!r}，并使用 Gaussian 原生关键字；"
            "不要把其他软件的溶剂展示名直接写入 Gaussian。"
        )
    if route_contains_scan:
        return (
            "Gaussian route 语法无效（QPErr）；输出中的 route 含 Opt/ModRedundant，"
            "请核对 "
            "action_settings.coordinate_type、atom_indices、start_value、end_value、"
            "value_unit 和 steps；相关值必须符合 Gaussian 原生关键字和相对扫描语法。"
        )
    return (
        "Gaussian route 语法无效（QPErr），但输出未能确定具体字段：请逐项核对 "
        f"method_spec.method={method.get('method')!r}、"
        f"method_spec.basis={method.get('basis')!r}、"
        "method_spec.solvation_model、method_spec.solvent 及相关 action_settings；"
        "这些字段必须使用 Gaussian 原生关键字，不要使用其他程序或论文中的展示写法。"
    )


def gaussian(action_id: str, request: dict[str, Any]) -> dict[str, Any]:
    inputs, method, settings = request_parts(request)
    structure_input_name = {
        "locate_transition_state": "initial_guess",
        "trace_intrinsic_reaction_coordinate": "transition_state",
    }.get(action_id, "structure")
    structure, symbols, _coordinates = _molecular_structure(
        inputs[structure_input_name]
    )
    structure = _gaussian_structure_with_effective_state(structure, method)
    directory = output_directory(action_id, "gaussian")
    input_path = directory / "job.gjf"
    input_path.write_text(
        _render_gaussian(
            action_id,
            structure,
            method,
            settings,
            dict(request.get("resource_limits") or {}),
        ),
        encoding="utf-8",
    )
    gaussian_scratch = Path(
        tempfile.mkdtemp(prefix=".gaussian_scratch_", dir=directory)
    )
    try:
        completed = run_external(
            executable="g16",
            environment_variable="CHEMGRAPH_GAUSSIAN_COMMAND",
            arguments=[],
            directory=directory,
            stdin_text=input_path.read_text(encoding="utf-8"),
            timeout_seconds=int(request.get("resource_limits", {}).get("walltime_seconds", 1800)),
            environment_overrides={"GAUSS_SCRDIR": str(gaussian_scratch)},
        )
    finally:
        shutil.rmtree(gaussian_scratch, ignore_errors=True)
    output_path = directory / "job.log"
    output_path.write_text(completed["stdout"], encoding="utf-8")
    (directory / "job.err").write_text(completed["stderr"], encoding="utf-8")
    if not completed["available"]:
        return unavailable(
            completed["stderr"],
            install="Place the licensed Gaussian 16 distribution in .mini_software_cache/gaussian/g16.",
        )
    normal_termination = _gaussian_output_ends_with_normal_termination(
        completed["stdout"]
    )
    fatal_marker = _gaussian_contains_fatal_marker(
        completed["stdout"], completed["stderr"]
    )
    archive_only_completion = _gaussian_archive_only_completion(completed)
    if (
        (completed["returncode"] != 0 and not archive_only_completion)
        or not normal_termination
        or fatal_marker
    ):
        detail = (completed["stderr"] or completed["stdout"])[-4000:]
        removed_core_dumps = _remove_generated_gaussian_core_dumps(directory)
        if "QPErr --- A syntax error was detected in the input line." in (
            completed["stdout"] + "\n" + completed["stderr"]
        ):
            raise ValueError(
                _gaussian_qperr_message(
                    action_id,
                    method,
                    completed["stdout"]
                    + "\n"
                    + completed["stderr"],
                )
            )
        cleanup_note = (
            f" Removed generated core dumps: {removed_core_dumps}."
            if removed_core_dumps
            else ""
        )
        raise GaussianExecutionError(
            f"Gaussian failed or did not terminate normally: {detail}{cleanup_note}",
            returncode=completed.get("returncode"),
            termination="normal" if normal_termination else "abnormal",
            archive_warning=(
                _GAUSSIAN_ARCHIVE_ENTRY_FAILURE_MARKER in completed["stdout"]
                or _GAUSSIAN_ARCHIVE_PUNCHED_MARKER in completed["stdout"]
            ),
            timeout=bool(completed.get("timeout")),
        )
    completion_provenance: dict[str, Any] = {}
    completion_warnings: list[str] = []
    if archive_only_completion:
        completion_provenance = {
            "gaussian_external_returncode": completed["returncode"],
            "gaussian_termination": "normal",
            "gaussian_archive_warning": {
                "entry_marker": _GAUSSIAN_ARCHIVE_ENTRY_FAILURE_MARKER,
                "punched_marker": _GAUSSIAN_ARCHIVE_PUNCHED_MARKER,
            },
        }
        completion_warnings.append(
            "Gaussian returned inner exit code 1 after normal termination; the complete "
            "scientific output was accepted because only the archive entry warning was present."
        )
    version_match = re.search(r"Gaussian 16, Revision\s+([^,\s]+)", completed["stdout"])
    backend_version = f"16 {version_match.group(1)}" if version_match else "16"
    energy = _parse_gaussian_energy(completed["stdout"])
    if energy is None:
        raise RuntimeError("Could not parse Gaussian SCF/DFT energy")
    output_file = relative_workspace_path(output_path)
    if action_id == "calculate_energy":
        result = {"energy": energy, "unit": "hartree", "output_file": output_file}
    elif action_id == "calculate_dipole_moment":
        dipole = _parse_gaussian_dipole(completed["stdout"])
        if dipole is None:
            raise RuntimeError("Could not parse Gaussian dipole moment")
        result = {
            "dipole": dipole,
            "unit": "debye",
            "energy_hartree": energy,
            "output_file": output_file,
        }
    elif action_id == "calculate_excited_states":
        states = _parse_gaussian_excited_states(completed["stdout"])
        requested_states = int(settings["number_of_states"])
        requested_spin = str(settings["spin_symmetry"]).strip().casefold()
        state_indices_complete = [state["state_index"] for state in states] == list(
            range(1, requested_states + 1)
        )
        spin_manifold_complete = all(
            state["spin_symmetry"] == requested_spin for state in states
        )
        result = {
            "states": states,
            "excited_state_method": str(method["excited_state_method"]).strip().casefold(),
            "spin_symmetry": requested_spin,
            "ground_state_energy_hartree": energy,
            "requested_states": requested_states,
            "completed_states": len(states),
            "state_indices_complete": state_indices_complete,
            "spin_manifold_complete": spin_manifold_complete,
            "output_file": output_file,
        }
        if (
            len(states) != requested_states
            or not state_indices_complete
            or not spin_manifold_complete
        ):
            return partial_success(
                result,
                artifact_files=command_artifacts(directory),
                backend_version=backend_version,
                provenance={
                    "command": completed["command"],
                    **completion_provenance,
                },
                warnings=[
                    *completion_warnings,
                    "Gaussian terminated normally, but the parsed excited states did not "
                    f"form the requested contiguous {requested_spin} roots 1..{requested_states} "
                    f"(parsed count: {len(states)})."
                ],
            )
    elif action_id == "calculate_bond_orders":
        matrix, nbo_version = _parse_gaussian_wiberg_matrix(
            completed["stdout"], symbols
        )
        minimum = float(settings["minimum_bond_order"])
        bonds = []
        for atom_a in range(len(symbols)):
            for atom_b in range(atom_a + 1, len(symbols)):
                bond_order = matrix[atom_a][atom_b]
                if bond_order < minimum:
                    continue
                bonds.append(
                    {
                        "atom_index_a": atom_a,
                        "element_a": symbols[atom_a],
                        "atom_index_b": atom_b,
                        "element_b": symbols[atom_b],
                        "bond_order": bond_order,
                    }
                )
        result = {
            "bonds": bonds,
            "bond_order_matrix": matrix,
            "analysis": "wiberg_nao",
            "population_engine": "Gaussian NBO",
            "population_engine_version": nbo_version,
            "minimum_bond_order": minimum,
            "energy_hartree": energy,
            "output_file": output_file,
        }
    elif action_id == "scan_reaction_coordinates":
        scan = _gaussian_scan_definition(structure, settings)
        points = _parse_gaussian_relaxed_scan(
            completed["stdout"], symbols, structure, scan
        )
        requested_points = int(scan["steps"]) + 1
        complete = _gaussian_scan_has_complete_finite_points(
            points, requested_points, completed["stdout"]
        )
        scan_file = write_json(
            directory,
            "relaxed_scan.json",
            {
                "coordinate_type": scan["coordinate_type"],
                "atom_indices": scan["atom_indices"],
                "coordinate_unit": scan["value_unit"],
                "points": points,
            },
        )
        result = {
            "coordinate_type": scan["coordinate_type"],
            "atom_indices": scan["atom_indices"],
            "coordinate_values": [point["coordinate_value"] for point in points],
            "coordinate_unit": scan["value_unit"],
            "energies_hartree": [point["energy_hartree"] for point in points],
            "relative_energies_kcal_mol": [
                point["relative_energy_kcal_mol"] for point in points
            ],
            "structures": [point["structure"] for point in points],
            "points": points,
            "completed_points": len(points),
            "requested_points": requested_points,
            "converged": complete,
            "data_file": relative_workspace_path(scan_file),
            "output_file": output_file,
        }
        if archive_only_completion and not complete:
            raise GaussianExecutionError(
                "Gaussian normal termination contained the archive warning, but the "
                "relaxed scan did not satisfy the complete scan gate.",
                returncode=completed.get("returncode"),
                termination="normal",
                archive_warning=True,
                timeout=bool(completed.get("timeout")),
            )
        if not complete:
            return partial_success(
                result,
                artifact_files=command_artifacts(directory),
                backend_version=backend_version,
                provenance={
                    "command": completed["command"],
                    **completion_provenance,
                    "native_optimizer_convergence": {
                        "converged": False,
                        "completed_points": len(points),
                        "requested_points": requested_points,
                    },
                },
                warnings=[
                    *completion_warnings,
                    "Gaussian ended normally, but the relaxed scan did not produce every "
                    "requested optimized point and its completed scan summary."
                ],
            )
    elif action_id in {"optimize_geometry", "locate_transition_state"}:
        geometry = _parse_gaussian_geometry(completed["stdout"], symbols)
        if geometry is None:
            raise RuntimeError("Could not parse Gaussian optimized/TS geometry")
        geometry = _preserve_gaussian_structure_metadata(geometry, structure)
        converged = bool(
            _GAUSSIAN_OPTIMIZATION_COMPLETION.search(completed["stdout"])
        )
        result = {
            "structure": geometry,
            "converged": converged,
            "energy": energy,
            "energy_unit": "hartree",
            "output_file": output_file,
        }
        if action_id == "locate_transition_state":
            result["validation_required"] = (
                "Call calculate_hessian and derive_vibrational_modes explicitly, then call "
                "trace_intrinsic_reaction_coordinate separately in each required direction."
            )
        if not converged:
            return partial_success(
                result,
                artifact_files=command_artifacts(directory),
                backend_version=backend_version,
                provenance={
                    "command": completed["command"],
                    **completion_provenance,
                },
                warnings=[
                    *completion_warnings,
                    "Gaussian ended normally, but the geometry/transition-state optimization "
                    "did not converge."
                ],
            )
    elif action_id == "calculate_hessian":
        checkpoint = directory / "job.chk"
        formatted = directory / "job.fchk"
        if not checkpoint.is_file():
            raise RuntimeError("Gaussian frequency calculation produced no checkpoint")
        formchk = run_external(
            executable="formchk",
            environment_variable="CHEMGRAPH_GAUSSIAN_FORMCHK_COMMAND",
            arguments=[checkpoint.name, formatted.name],
            directory=directory,
            timeout_seconds=int(request.get("resource_limits", {}).get("walltime_seconds", 7200)),
        )
        (directory / "formchk.stdout.log").write_text(formchk["stdout"], encoding="utf-8")
        (directory / "formchk.stderr.log").write_text(formchk["stderr"], encoding="utf-8")
        matrix = (
            _parse_gaussian_fchk_hessian(formatted, 3 * len(symbols))
            if formchk["available"] and formchk["returncode"] == 0 and formatted.is_file()
            else None
        )
        hessian_structure = None
        structure_parse_error = None
        if formchk["available"] and formchk["returncode"] == 0 and formatted.is_file():
            try:
                hessian_structure = _parse_gaussian_fchk_structure(
                    formatted, structure
                )
            except (OSError, ValueError) as exc:
                structure_parse_error = str(exc)
        result = {
            "matrix": matrix,
            "unit": "hartree/bohr^2",
            "structure": hessian_structure,
            "coordinate_frame": {
                "source": "gaussian_fchk_current_cartesian_coordinates",
                "fchk_coordinate_unit": "bohr",
                "structure_coordinate_unit": "angstrom",
                "hessian_and_structure_share_cartesian_frame": (
                    matrix is not None and hessian_structure is not None
                ),
                "hessian_was_rotated_to_input_frame": False,
            },
            "energy_hartree": energy,
            "raw_hessian_path": relative_workspace_path(formatted) if formatted.is_file() else None,
            "output_file": output_file,
        }
        if matrix is None or hessian_structure is None:
            warnings = []
            if matrix is None:
                warnings.append(
                    "Gaussian completed, but the formatted checkpoint Hessian could not be parsed."
                )
            if hessian_structure is None:
                detail = (
                    f" ({structure_parse_error})" if structure_parse_error else ""
                )
                warnings.append(
                    "Gaussian completed, but the formatted checkpoint structure in the "
                    f"Hessian Cartesian frame could not be parsed{detail}."
                )
            return partial_success(
                result,
                artifact_files=command_artifacts(directory),
                backend_version=backend_version,
                provenance={
                    "command": completed["command"],
                    **completion_provenance,
                    "formchk_command": formchk["command"],
                    "hessian_structure_contract": (
                        "matrix and result.structure must come from the same formatted "
                        "checkpoint Cartesian frame"
                    ),
                },
                warnings=[*completion_warnings, *warnings],
            )
    elif action_id == "trace_intrinsic_reaction_coordinate":
        direction = str(settings["direction"]).strip().casefold()
        points = _parse_gaussian_irc_points(
            completed["stdout"], symbols, structure, direction
        )
        summary = _parse_gaussian_irc_summary(completed["stdout"])
        transition_state = deepcopy(structure)
        path_file = write_json(
            directory,
            "irc_path.json",
            {
                "direction": direction,
                "transition_state": transition_state,
                "points": points,
            },
        )
        summary_file = write_json(directory, "irc_summary.json", summary)
        complete, termination_reason = _parse_gaussian_irc_termination(
            completed["stdout"], direction
        )
        result = {
            "direction": direction,
            "transition_state": transition_state,
            "structures": [transition_state, *[record["structure"] for record in points]],
            "points": points,
            "path_file": relative_workspace_path(path_file),
            "summary_file": relative_workspace_path(summary_file),
            "summary": summary,
            "completed_points": len(points),
            "path_structure_count": len(points) + 1,
            "requested_max_points": int(settings["max_points"]),
            "gaussian_branch_completed": termination_reason in {
                "pes_minimum_detected",
                "branch_completed_without_pes_minimum",
                "maximum_points_reached",
            },
            "pes_minimum_detected": termination_reason == "pes_minimum_detected",
            "converged": complete,
            "termination_reason": termination_reason,
            "output_file": output_file,
        }
        if not complete or not points:
            warnings = []
            if not complete:
                if termination_reason == "maximum_points_reached":
                    warnings.append(
                        f"Gaussian reached the configured maximum for the {direction} IRC "
                        "branch before detecting a PES minimum."
                    )
                elif termination_reason == "branch_completed_without_pes_minimum":
                    warnings.append(
                        f"Gaussian printed a completion marker for the {direction} IRC branch, "
                        "but did not report that a PES minimum was detected. The path is retained "
                        "for audit but is not marked converged."
                    )
                else:
                    warnings.append(
                        f"Gaussian ended normally without a completion marker for the "
                        f"{direction} IRC branch."
                    )
            if not points:
                warnings.append(
                    "Gaussian IRC output contained no parseable direction-tagged path structures; "
                    "the complete native output remains available for audit."
                )
            return partial_success(
                result,
                artifact_files=command_artifacts(directory),
                backend_version=backend_version,
                provenance={
                    "command": completed["command"],
                    **completion_provenance,
                    "direction_is_atomic": True,
                },
                warnings=[*completion_warnings, *warnings],
            )
    else:
        return unsupported(f"Gaussian does not implement {action_id}")
    return success(
        result,
        artifact_files=command_artifacts(directory),
        backend_version=backend_version,
        warnings=completion_warnings,
        provenance={
            "command": completed["command"],
            **completion_provenance,
            **(
                {"direction_is_atomic": True}
                if action_id == "trace_intrinsic_reaction_coordinate"
                else {}
            ),
            **(
                {
                    "formchk_command": formchk["command"],
                    "hessian_structure_contract": (
                        "matrix and result.structure come from the same formatted "
                        "checkpoint Cartesian frame"
                    ),
                }
                if action_id == "calculate_hessian"
                else {}
            ),
            **(
                {
                    "native_optimizer_convergence": {
                        "converged": True,
                        "completed_points": result["completed_points"],
                        "requested_points": result["requested_points"],
                    }
                }
                if action_id == "scan_reaction_coordinates"
                else {}
            ),
        },
    )


def _gamess_token(value: Any, name: str) -> str:
    return _safe_keyword(value, name, r"[A-Za-z0-9+-]+").upper()


def _gamess_convergence(value: Any) -> int:
    tolerance = _finite_float(value, "GAMESS scf_convergence", positive=True)
    exponent = int(round(-math.log10(tolerance)))
    if exponent < 4 or exponent > 14 or not math.isclose(tolerance, 10.0 ** (-exponent), rel_tol=1e-6):
        raise ValueError("GAMESS scf_convergence must be an exact power of ten from 1e-4 to 1e-14")
    return exponent


def _render_gamess(
    action_id: str,
    structure_value: Any,
    method: dict[str, Any],
    settings: dict[str, Any],
    resource_limits: dict[str, Any] | None = None,
) -> str:
    """Render a molecular GAMESS input from explicit scientific fields."""

    structure, symbols, coordinates = _molecular_structure(structure_value)
    scftyp = _gamess_token(method["scftyp"], "GAMESS SCFTYP")
    gbasis = _gamess_token(method["gbasis"], "GAMESS GBASIS")
    ngauss = int(method["ngauss"])
    if ngauss < 1 or ngauss > 9:
        raise ValueError("GAMESS NGAUSS must be between 1 and 9")
    runtyp = "OPTIMIZE" if action_id == "optimize_geometry" else "ENERGY"
    charge = int(method.get("charge", structure.get("charge", 0)))
    multiplicity = int(method.get("multiplicity", structure.get("multiplicity", 1)))
    if multiplicity < 1:
        raise ValueError("GAMESS multiplicity must be at least one")
    control = [
        f"SCFTYP={scftyp}",
        f"RUNTYP={runtyp}",
        "COORD=UNIQUE",
        "UNITS=ANGS",
        f"ICHARG={charge}",
        f"MULT={multiplicity}",
    ]
    if method.get("dfttyp"):
        control.append(f"DFTTYP={_gamess_token(method['dfttyp'], 'GAMESS DFTTYP')}")
    limits = resource_limits or {}
    memory_mb = max(128, int(limits.get("memory_mb") or 1000))
    mwords = max(1, memory_mb // 8)
    basis = [f"GBASIS={gbasis}", f"NGAUSS={ngauss}"]
    for key in ("ndfunc", "npfunc", "nffunc"):
        if key in method:
            value = int(method[key])
            if value < 0 or value > 9:
                raise ValueError(f"GAMESS {key.upper()} must be between 0 and 9")
            basis.append(f"{key.upper()}={value}")
    for key in ("diffsp", "diffs"):
        if key in method:
            basis.append(f"{key.upper()}={'.TRUE.' if bool(method[key]) else '.FALSE.'}")
    lines = [
        " $CONTRL " + " ".join(control) + " $END",
        f" $SYSTEM MWORDS={mwords} $END",
        f" $SCF DIRSCF=.TRUE. CONV={_gamess_convergence(settings['scf_convergence'])} $END",
        " $BASIS " + " ".join(basis) + " $END",
        f" $GUESS GUESS={_gamess_token(method.get('guess', 'HUCKEL'), 'GAMESS GUESS')} $END",
    ]
    if action_id == "optimize_geometry":
        tolerance = _finite_float(settings["gradient_tolerance_hartree_per_bohr"], "GAMESS gradient tolerance", positive=True)
        max_steps = int(settings["max_steps"])
        if max_steps < 1:
            raise ValueError("GAMESS max_steps must be positive")
        lines.append(f" $STATPT OPTTOL={tolerance:.12g} NSTEP={max_steps} $END")
    lines.extend([" $DATA", "ResearchChemBench typed atomic action", "C1"])
    for symbol, row in zip(symbols, coordinates):
        atomic_number = _ATOMIC_NUMBERS.get(symbol.lower())
        if atomic_number is None:
            raise ValueError(f"Unknown element for GAMESS: {symbol}")
        lines.append(
            f"{symbol} {float(atomic_number):.1f} {row[0]:.12f} {row[1]:.12f} {row[2]:.12f}"
        )
    lines.extend([" $END", ""])
    return "\n".join(lines)


def _parse_gamess_energy(text: str) -> float | None:
    matches = re.findall(rf"FINAL\s+[A-Z0-9()+-]+\s+ENERGY IS\s+({_FLOAT})", text)
    return float(matches[-1].replace("D", "E")) if matches else None


def _parse_gamess_dipole(text: str) -> list[float] | None:
    matches = re.findall(
        rf"DX\s+DY\s+DZ\s+/D/\s+\(DEBYE\)\s*\n\s*({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})",
        text,
    )
    if not matches:
        return None
    return [float(value.replace("D", "E")) for value in matches[-1]]


def _parse_gamess_geometry(text: str) -> dict[str, Any] | None:
    lines = text.splitlines()
    blocks: list[list[dict[str, Any]]] = []
    row_pattern = re.compile(
        rf"^\s*([A-Za-z][A-Za-z]?)\s+{_FLOAT}\s+({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})\s*$"
    )
    for index, line in enumerate(lines):
        if "COORDINATES OF ALL ATOMS ARE (ANGS)" not in line:
            continue
        cursor = index + 1
        while cursor < len(lines) and set(lines[cursor].strip()) != {"-"}:
            cursor += 1
        cursor += 1
        atoms: list[dict[str, Any]] = []
        while cursor < len(lines):
            match = row_pattern.match(lines[cursor])
            if not match:
                break
            atoms.append(
                {
                    "element": match.group(1).title(),
                    "position_angstrom": [
                        float(match.group(2).replace("D", "E")),
                        float(match.group(3).replace("D", "E")),
                        float(match.group(4).replace("D", "E")),
                    ],
                }
            )
            cursor += 1
        if atoms:
            blocks.append(atoms)
    if not blocks:
        return None
    return {"atoms": blocks[-1], "pbc": [False, False, False]}


def gamess(action_id: str, request: dict[str, Any]) -> dict[str, Any]:
    inputs, method, settings = request_parts(request)
    directory = output_directory(action_id, "gamess")
    job_name = f"rchem_{directory.name}"
    input_path = directory / f"{job_name}.inp"
    input_path.write_text(
        _render_gamess(
            action_id,
            inputs["structure"],
            method,
            settings,
            dict(request.get("resource_limits") or {}),
        ),
        encoding="utf-8",
    )
    cores = max(1, int(request.get("resource_limits", {}).get("cpu_cores") or 1))
    completed = run_external(
        executable="rungms",
        environment_variable="CHEMGRAPH_GAMESS_COMMAND",
        arguments=[job_name, "00", str(cores), str(cores)],
        directory=directory,
        timeout_seconds=int(request.get("resource_limits", {}).get("walltime_seconds", 1800)),
    )
    output_path = directory / f"{job_name}.log"
    output_path.write_text(completed["stdout"], encoding="utf-8")
    (directory / f"{job_name}.err").write_text(completed["stderr"], encoding="utf-8")
    if not completed["available"]:
        return unavailable(
            completed["stderr"],
            install="Build the registered GAMESS distribution under .software_cache/gamess.",
        )
    if completed["returncode"] != 0 or "EXECUTION OF GAMESS TERMINATED NORMALLY" not in completed["stdout"]:
        detail = (completed["stderr"] or completed["stdout"])[-4000:]
        raise RuntimeError(f"GAMESS failed or did not terminate normally: {detail}")
    version_match = re.search(r"GAMESS VERSION\s*=\s*([^*\n]+)", completed["stdout"])
    backend_version = version_match.group(1).strip() if version_match else None
    energy = _parse_gamess_energy(completed["stdout"])
    if energy is None:
        raise RuntimeError("Could not parse GAMESS final energy")
    output_file = relative_workspace_path(output_path)
    if action_id == "calculate_energy":
        result = {"energy": energy, "unit": "hartree", "output_file": output_file}
    elif action_id == "calculate_dipole_moment":
        dipole = _parse_gamess_dipole(completed["stdout"])
        if dipole is None:
            raise RuntimeError("Could not parse GAMESS dipole moment")
        result = {
            "dipole": dipole,
            "unit": "debye",
            "energy_hartree": energy,
            "output_file": output_file,
        }
    elif action_id == "optimize_geometry":
        geometry = _parse_gamess_geometry(completed["stdout"])
        if geometry is None:
            raise RuntimeError("Could not parse GAMESS final geometry")
        converged = "EQUILIBRIUM GEOMETRY LOCATED" in completed["stdout"]
        result = {
            "structure": geometry,
            "converged": converged,
            "energy": energy,
            "energy_unit": "hartree",
            "output_file": output_file,
        }
        if not converged:
            return partial_success(
                result,
                artifact_files=command_artifacts(directory),
                backend_version=backend_version,
                provenance={"command": completed["command"], "parallel_processes": cores},
                warnings=["GAMESS ended normally, but the geometry optimization did not converge."],
            )
    else:
        return unsupported(f"GAMESS does not implement {action_id}")
    return success(
        result,
        artifact_files=command_artifacts(directory),
        backend_version=backend_version,
        provenance={"command": completed["command"], "parallel_processes": cores},
    )
