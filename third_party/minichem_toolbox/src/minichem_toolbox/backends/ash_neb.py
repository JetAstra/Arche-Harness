"""ASH/KNARR NEB backend using an explicitly selected FairChem UMA model."""

from __future__ import annotations

import io
import math
import os
import re
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from copy import deepcopy
from pathlib import Path
from typing import Any

from .common import (
    atoms_and_coordinates,
    command_artifacts,
    module_version,
    output_directory,
    partial_success,
    relative_workspace_path,
    request_parts,
    structure_dict,
    success,
    unavailable,
    unsupported,
    write_xyz,
)


ACTIONS = {"search_reaction_path"}

HARD_MAX_TOTAL_IMAGES = 128
HARD_MAX_NEB_STEPS = 10_000
HARD_MAX_IDPP_ITERATIONS = 10_000
HARD_MAX_FORCE_THRESHOLD_EV_PER_ANGSTROM = 10.0
_TRANSITION_STATE_GUESS_SELECTION_RULE_VERSION = "ts-guess-selection.v1"

_ATOM_MAP_KEYS = ("atom_map_number", "map_number", "atom_map")
_COORDINATE_ATOM_FIELDS = {
    "position",
    "position_angstrom",
}
_IDENTITY_ATOM_FIELDS = {
    *_ATOM_MAP_KEYS,
    "isotope",
    "label",
    "site_label",
    "atom_id",
    "name",
}


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@contextmanager
def _model_download_policy(allow_download: bool) -> Iterator[None]:
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


def _strict_integer(
    value: Any,
    *,
    field_name: str,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(f"{field_name} must be between {minimum} and {maximum}")
    return value


def _finite_positive_float(
    value: Any,
    *,
    field_name: str,
    maximum: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0 or converted > maximum:
        raise ValueError(
            f"{field_name} must be finite, positive, and no greater than {maximum}"
        )
    return converted


def _parse_positive_integer_atom_count(raw_value: str, *, file_name: str) -> int:
    """解析 XYZ 原子数头，允许表示整数的有限数值文本并拒绝非法计数。

    输入是单行 XYZ 原子数；函数只接受有限、正且没有小数部分的数值，返回 Python
    整数。空值、非数值、NaN、无穷、非正数和小数原子数都会抛出可操作的 ValueError，
    供调用方把该文件标记为无效而不是继续读取错误帧。
    """

    text = raw_value.strip()
    try:
        numeric = float(text)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"Invalid KNARR saddle atom count in {file_name}; expected a finite positive integer"
        ) from exc
    if not math.isfinite(numeric) or numeric <= 0 or not numeric.is_integer():
        raise ValueError(
            f"Invalid KNARR saddle atom count in {file_name}; expected a finite positive integer"
        )
    return int(numeric)


def _validated_neb_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    total_images = _strict_integer(
        settings.get("total_images"),
        field_name="action_settings.total_images",
        minimum=3,
        maximum=HARD_MAX_TOTAL_IMAGES,
    )
    max_steps = _strict_integer(
        settings.get("max_steps"),
        field_name="action_settings.max_steps",
        minimum=1,
        maximum=HARD_MAX_NEB_STEPS,
    )
    idpp_max_iterations = _strict_integer(
        settings.get("idpp_max_iterations"),
        field_name="action_settings.idpp_max_iterations",
        minimum=1,
        maximum=HARD_MAX_IDPP_ITERATIONS,
    )
    maximum_force = _finite_positive_float(
        settings.get("maximum_force_ev_per_angstrom"),
        field_name="action_settings.maximum_force_ev_per_angstrom",
        maximum=HARD_MAX_FORCE_THRESHOLD_EV_PER_ANGSTROM,
    )
    rms_force = _finite_positive_float(
        settings.get("rms_force_ev_per_angstrom"),
        field_name="action_settings.rms_force_ev_per_angstrom",
        maximum=HARD_MAX_FORCE_THRESHOLD_EV_PER_ANGSTROM,
    )
    if rms_force > maximum_force:
        raise ValueError(
            "action_settings.rms_force_ev_per_angstrom must not exceed "
            "action_settings.maximum_force_ev_per_angstrom"
        )
    climb = settings.get("climb")
    if not isinstance(climb, bool):
        raise ValueError("action_settings.climb must be an explicit boolean")
    interpolation_value = settings.get("interpolation")
    if not isinstance(interpolation_value, str):
        raise ValueError("action_settings.interpolation must be a string")
    interpolation = interpolation_value.strip().upper()
    if interpolation not in {"IDPP", "GEODESIC"}:
        raise ValueError("action_settings.interpolation must be idpp or geodesic")
    return {
        "total_images": total_images,
        "max_steps": max_steps,
        "idpp_max_iterations": idpp_max_iterations,
        "maximum_force_ev_per_angstrom": maximum_force,
        "rms_force_ev_per_angstrom": rms_force,
        "climb": climb,
        "interpolation": interpolation,
    }


def _atom_map_sequence(
    structure: Mapping[str, Any], *, endpoint: str
) -> list[int] | None:
    atoms = structure.get("atoms")
    symbols, _coordinates = atoms_and_coordinates(dict(structure))
    aligned = structure.get("atom_map_numbers")
    has_aligned = aligned is not None
    has_per_atom = isinstance(atoms, list) and any(
        isinstance(atom, Mapping)
        and any(key in atom and atom[key] is not None for key in _ATOM_MAP_KEYS)
        for atom in atoms
    )
    if not has_aligned and not has_per_atom:
        return None
    if has_per_atom and not isinstance(atoms, list):
        raise ValueError(f"{endpoint}.atoms must be a list")
    if has_aligned and (not isinstance(aligned, list) or len(aligned) != len(symbols)):
        raise ValueError(
            f"{endpoint}.atom_map_numbers must align one-to-one with endpoint atoms"
        )

    sequence: list[int] = []
    for index in range(len(symbols)):
        raw_values = []
        if isinstance(atoms, list):
            atom = atoms[index]
            if not isinstance(atom, Mapping):
                raise ValueError(f"{endpoint}.atoms[{index}] must be a mapping")
            raw_values.extend(
                atom[key]
                for key in _ATOM_MAP_KEYS
                if key in atom and atom[key] is not None
            )
        if has_aligned:
            raw_values.append(aligned[index])
        if not raw_values:
            raise ValueError(
                f"{endpoint} endpoint has partial atom mapping; atom index {index} is unmapped"
            )
        parsed = []
        for raw in raw_values:
            if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
                raise ValueError(
                    f"{endpoint} atom map at index {index} must be a positive integer"
                )
            parsed.append(raw)
        if len(set(parsed)) != 1:
            raise ValueError(
                f"{endpoint} atom index {index} contains conflicting atom-map fields"
            )
        sequence.append(parsed[0])
    if len(set(sequence)) != len(sequence):
        raise ValueError(f"{endpoint} endpoint atom-map numbers must be unique")
    return sequence


def _validate_endpoints(
    reactant: dict[str, Any], product: dict[str, Any]
) -> tuple[list[str], int, int]:
    reactant_symbols, _ = atoms_and_coordinates(reactant)
    product_symbols, _ = atoms_and_coordinates(product)
    if reactant_symbols != product_symbols:
        raise ValueError(
            "Double-ended NEB endpoints must have identical atom counts, elements, and order"
        )
    for endpoint_name, endpoint in (("reactant", reactant), ("product", product)):
        if "charge" not in endpoint or "multiplicity" not in endpoint:
            raise ValueError(
                f"{endpoint_name} endpoint must explicitly declare charge and multiplicity"
            )
        if any(bool(value) for value in endpoint.get("pbc", [False, False, False])):
            raise ValueError("ASH/UMA molecular NEB does not accept periodic endpoints")
    if isinstance(reactant["charge"], bool) or not isinstance(reactant["charge"], int):
        raise ValueError("reactant endpoint charge must be an integer")
    if isinstance(product["charge"], bool) or not isinstance(product["charge"], int):
        raise ValueError("product endpoint charge must be an integer")
    charge = reactant["charge"]
    if charge != product["charge"]:
        raise ValueError("Double-ended NEB endpoints must have the same charge")
    for endpoint_name, endpoint in (("reactant", reactant), ("product", product)):
        multiplicity_value = endpoint["multiplicity"]
        if (
            isinstance(multiplicity_value, bool)
            or not isinstance(multiplicity_value, int)
            or multiplicity_value < 1
        ):
            raise ValueError(
                f"{endpoint_name} endpoint multiplicity must be a positive integer"
            )
    multiplicity = reactant["multiplicity"]
    if multiplicity != product["multiplicity"]:
        raise ValueError("Double-ended NEB endpoints must have the same multiplicity")
    reactant_maps = _atom_map_sequence(reactant, endpoint="reactant")
    product_maps = _atom_map_sequence(product, endpoint="product")
    if (reactant_maps is None) != (product_maps is None):
        raise ValueError(
            "If either NEB endpoint has atom maps, both endpoints must provide complete "
            "atom maps"
        )
    if reactant_maps is not None and reactant_maps != product_maps:
        raise ValueError(
            "Double-ended NEB endpoint atom maps must be identical at every atom index"
        )
    return reactant_symbols, charge, multiplicity


def _restore_reference_metadata(
    structure: dict[str, Any], reference: Mapping[str, Any], *, endpoint_state: bool
) -> dict[str, Any]:
    output = deepcopy(structure)
    output_atoms = output.get("atoms")
    reference_atoms = reference.get("atoms")
    if not isinstance(output_atoms, list):
        raise ValueError("ASH path structures must contain an atoms list")

    reference_symbols, _coordinates = atoms_and_coordinates(dict(reference))
    if len(output_atoms) != len(reference_symbols):
        raise ValueError(
            "ASH path structure atom count does not match the reactant endpoint"
        )

    for index, (atom, expected_symbol) in enumerate(
        zip(output_atoms, reference_symbols)
    ):
        if not isinstance(atom, dict):
            raise ValueError(f"Invalid atom record at path atom index {index}")
        actual_symbol = str(atom.get("element") or atom.get("symbol") or "")
        if actual_symbol.casefold() != expected_symbol.casefold():
            raise ValueError(
                "ASH path atom order does not match the reactant endpoint at index "
                f"{index}: expected {expected_symbol}, recovered {actual_symbol or '<missing>'}"
            )
        if isinstance(reference_atoms, list):
            reference_atom = reference_atoms[index]
            if not isinstance(reference_atom, Mapping):
                raise ValueError(f"Invalid reactant atom record at index {index}")
            for field, value in reference_atom.items():
                if field in _IDENTITY_ATOM_FIELDS or (
                    endpoint_state and field not in _COORDINATE_ATOM_FIELDS
                ):
                    atom[field] = deepcopy(value)

    if endpoint_state and "bonds" in reference:
        output["bonds"] = deepcopy(reference["bonds"])
    if "atom_map_numbers" in reference:
        output["atom_map_numbers"] = deepcopy(reference["atom_map_numbers"])
    return output


def _parse_multixyz(
    path: Path,
    *,
    charge: int,
    multiplicity: int,
    reactant_structure: Mapping[str, Any],
    product_structure: Mapping[str, Any],
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
            raise ValueError(f"Invalid multi-XYZ atom count in {path.name}") from exc
        stop = cursor + atom_count + 2
        if stop > len(lines):
            raise ValueError(f"Truncated multi-XYZ frame in {path.name}")
        comment = lines[cursor + 1]
        atoms = []
        for line in lines[cursor + 2 : stop]:
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"Invalid multi-XYZ atom row in {path.name}: {line!r}")
            atoms.append(
                {
                    "element": parts[0],
                    "position_angstrom": [
                        float(parts[1]),
                        float(parts[2]),
                        float(parts[3]),
                    ],
                }
            )
        frame: dict[str, Any] = {
            "atoms": atoms,
            "charge": charge,
            "multiplicity": multiplicity,
            "pbc": [False, False, False],
            "comment": comment,
        }
        try:
            atoms_and_coordinates(frame)
        except ValueError as exc:
            raise ValueError(
                f"Invalid three-dimensional finite coordinates in {path.name}"
            ) from exc
        match = re.search(
            r"(?:energy|E)\s*[.:=]?\s*(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)",
            comment,
            re.IGNORECASE,
        )
        if match:
            frame["energy"] = float(match.group(1))
        frames.append(
            _restore_reference_metadata(frame, reactant_structure, endpoint_state=False)
        )
        cursor = stop
    if frames:
        frames[0] = _restore_reference_metadata(
            frames[0], reactant_structure, endpoint_state=True
        )
        frames[-1] = _restore_reference_metadata(
            frames[-1], product_structure, endpoint_state=True
        )
    return frames


def _final_path_file(directory: Path) -> Path | None:
    candidates = (
        directory / "knarr_MEP_FULL.xyz",
        directory / "knarr_MEP.xyz",
    )
    return next((path for path in candidates if path.is_file()), None)


def _saddle_structure(
    result: Any,
    symbols: list[str],
    *,
    charge: int,
    multiplicity: int,
    reference_structure: Mapping[str, Any],
) -> dict[str, Any] | None:
    coordinates = getattr(result, "geometry", None)
    if coordinates is None:
        fragment = getattr(result, "saddlepoint_fragment", None)
        coordinates = getattr(fragment, "coords", None)
    if coordinates is None:
        return None
    rows = coordinates.tolist() if hasattr(coordinates, "tolist") else coordinates
    if len(rows) != len(symbols):
        return None
    structure = {
        "atoms": [
            {
                "element": symbol,
                "position_angstrom": [float(value) for value in row],
            }
            for symbol, row in zip(symbols, rows)
        ],
        "charge": charge,
        "multiplicity": multiplicity,
        "pbc": [False, False, False],
    }
    try:
        atoms_and_coordinates(structure)
    except ValueError as exc:
        raise ValueError(
            "ASH saddle geometry must contain exactly three finite coordinates per atom"
        ) from exc
    return _restore_reference_metadata(
        structure, reference_structure, endpoint_state=False
    )


def _parse_knarr_saddle_xyz(
    path: Path,
    *,
    symbols: list[str],
    charge: int,
    multiplicity: int,
    reference_structure: Mapping[str, Any],
) -> dict[str, Any]:
    """解析作者 KNARR 的单帧 saddle XYZ，并恢复端点提供的原子身份元数据。

    输入必须是恰好一帧、原子数和元素顺序都匹配的 XYZ 文件；坐标或结构不完整时
    抛出 ValueError，由上层把它记录为不可交接的候选，而不是猜造一个 TS guess。
    """

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if len(lines) < 2:
        raise ValueError(f"KNARR saddle XYZ {path.name} is incomplete")
    atom_count = _parse_positive_integer_atom_count(
        lines[0], file_name=path.name
    )
    if atom_count != len(symbols):
        raise ValueError(
            f"KNARR saddle atom count in {path.name} does not match the endpoints"
        )
    expected_line_count = atom_count + 2
    if len(lines) != expected_line_count:
        raise ValueError(
            f"KNARR saddle {path.name} must contain exactly one complete XYZ frame"
        )
    atoms: list[dict[str, Any]] = []
    for line in lines[2:]:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Invalid KNARR saddle atom row in {path.name}")
        try:
            coordinates = [float(parts[1]), float(parts[2]), float(parts[3])]
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"KNARR saddle {path.name} has invalid three-dimensional coordinates"
            ) from exc
        atoms.append(
            {"element": parts[0], "position_angstrom": coordinates}
        )
    structure = {
        "atoms": atoms,
        "charge": charge,
        "multiplicity": multiplicity,
        "pbc": [False, False, False],
    }
    try:
        atoms_and_coordinates(structure)
    except ValueError as exc:
        raise ValueError(
            f"KNARR saddle {path.name} must contain finite three-dimensional coordinates"
        ) from exc
    return _restore_reference_metadata(
        structure, reference_structure, endpoint_state=False
    )


def _recover_unvalidated_transition_state_guess(
    directory: Path,
    *,
    images: list[dict[str, Any]],
    symbols: list[str],
    charge: int,
    multiplicity: int,
    reference_structure: Mapping[str, Any],
    ash_saddle: dict[str, Any] | None,
    highest_interior_index: int | None,
    energy_index_complete: bool,
    final_path_complete: bool,
    interior_peak_candidate_valid: bool,
    ash_saddle_warnings: list[str] | None = None,
) -> tuple[dict[str, Any] | None, str | None, int | None, list[str]]:
    """从完整路径选择未验证 guess，并返回来源、图像索引和可操作警告。

    端点最高、能量索引不完整或路径帧数不足时均返回空值。完整路径里的候选严格按
    KNARR saddle 文件、ASH saddle 几何、最高内部路径图像的顺序选择；无效 saddle
    文件会保留具体警告并继续尝试后续候选。返回的来源标识和图像索引让下游知道该
    几何来自作者文件、ASH 几何还是路径图像；选择规则版本由结果层固定登记。
    """

    if not (
        energy_index_complete
        and final_path_complete
        and interior_peak_candidate_valid
        and highest_interior_index is not None
        and 0 <= highest_interior_index < len(images)
    ):
        return None, None, None, []

    saddle_path = directory / "knarr_saddle.xyz"
    retained_warnings = list(ash_saddle_warnings or [])
    if saddle_path.is_file():
        try:
            return (
                _parse_knarr_saddle_xyz(
                    saddle_path,
                    symbols=symbols,
                    charge=charge,
                    multiplicity=multiplicity,
                    reference_structure=reference_structure,
                ),
                "knarr_saddle_xyz",
                None,
                retained_warnings,
            )
        except (OSError, ValueError) as exc:
            warning = (
                "ASH produced an invalid knarr_saddle.xyz; the file was not used: "
                f"{exc}."
            )
            if ash_saddle is not None:
                return (
                    ash_saddle,
                    "ash_saddle_geometry",
                    None,
                    retained_warnings
                    + [
                        warning
                        + " Using ASH saddle geometry as the next transition-state guess."
                    ],
                )
            return (
                images[highest_interior_index],
                "highest_interior_path_image",
                highest_interior_index,
                retained_warnings
                + [
                    warning
                    + " falling back to the highest interior path image at image index "
                    f"{highest_interior_index}."
                ],
            )

    if ash_saddle is not None:
        return ash_saddle, "ash_saddle_geometry", None, retained_warnings
    return (
        images[highest_interior_index],
        "highest_interior_path_image",
        highest_interior_index,
        retained_warnings,
    )


def _validated_energy_pairs(
    energy_mapping: Any, *, total_images: int
) -> tuple[list[tuple[int, float]], bool, list[str]]:
    if not isinstance(energy_mapping, Mapping):
        return [], False, ["ASH did not return an indexed MEP energy mapping."]

    pairs: list[tuple[int, float]] = []
    seen_indices: set[int] = set()
    warnings: list[str] = []
    invalid = False
    for raw_index, raw_value in energy_mapping.items():
        if isinstance(raw_index, bool):
            invalid = True
            warnings.append(
                f"Discarded invalid boolean MEP energy index {raw_index!r}."
            )
            continue
        if isinstance(raw_index, str):
            try:
                index = int(raw_index)
            except (ValueError, OverflowError):
                invalid = True
                warnings.append(f"Discarded invalid MEP energy index {raw_index!r}.")
                continue
            if raw_index.strip() != str(index):
                invalid = True
                warnings.append(
                    f"Discarded non-canonical MEP energy index {raw_index!r}."
                )
                continue
        else:
            try:
                index = raw_index.__index__()
            except (AttributeError, TypeError, ValueError, OverflowError):
                invalid = True
                warnings.append(f"Discarded invalid MEP energy index {raw_index!r}.")
                continue
        if index in seen_indices:
            invalid = True
            warnings.append(f"Discarded duplicate MEP energy index {index}.")
            continue
        if isinstance(raw_value, bool):
            invalid = True
            warnings.append(f"Discarded non-numeric MEP energy at image {index}.")
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError, OverflowError):
            invalid = True
            warnings.append(f"Discarded non-numeric MEP energy at image {index}.")
            continue
        if not math.isfinite(value):
            invalid = True
            warnings.append(f"Discarded non-finite MEP energy at image {index}.")
            continue
        seen_indices.add(index)
        pairs.append((index, value))

    pairs.sort(key=lambda item: item[0])
    expected_indices = list(range(total_images))
    received_indices = [index for index, _value in pairs]
    complete = not invalid and received_indices == expected_indices
    if received_indices != expected_indices:
        warnings.append(
            "ASH MEP energy indices are incomplete: expected "
            f"{expected_indices}, recovered {received_indices}."
        )
    return pairs, complete, warnings


def search_reaction_path(request: dict[str, Any]) -> dict[str, Any]:
    try:
        from ash import FairchemTheory, Fragment, NEB
    except (ImportError, ModuleNotFoundError) as exc:
        return unavailable(
            f"ASH/FairChem NEB dependencies are unavailable: {exc}",
            install=(
                "Install a compatible ASH release plus fairchem-core, torch, ase, and numpy "
                "in the ash_neb runtime."
            ),
        )

    inputs, method, settings = request_parts(request)
    reactant = structure_dict(inputs["reactant"])
    product = structure_dict(inputs["product"])
    symbols, charge, multiplicity = _validate_endpoints(reactant, product)

    allow_download = method.get("allow_model_download")
    if not isinstance(allow_download, bool):
        raise ValueError("method_spec.allow_model_download must be an explicit boolean")
    device = str(method["device"]).strip().lower()
    requested_gpus = int((request.get("resource_limits") or {}).get("gpu_count") or 0)
    if device.startswith("cuda") and requested_gpus < 1:
        raise ValueError(
            "A CUDA ASH/UMA path search requires resource_limits.gpu_count >= 1"
        )

    validated_settings = _validated_neb_settings(settings)
    total_images = validated_settings["total_images"]
    max_steps = validated_settings["max_steps"]
    maximum_force = validated_settings["maximum_force_ev_per_angstrom"]
    rms_force = validated_settings["rms_force_ev_per_angstrom"]
    climb = validated_settings["climb"]
    interpolation = validated_settings["interpolation"]
    idpp_max_iterations = validated_settings["idpp_max_iterations"]

    directory = output_directory("search_reaction_path", "ash_neb_uma")
    reactant_path = write_xyz(reactant, directory / "reactant.xyz")
    product_path = write_xyz(product, directory / "product.xyz")
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    seed = int(method["random_seed"])
    cpu_cores = int((request.get("resource_limits") or {}).get("cpu_cores") or 1)
    try:
        with _working_directory(directory), _model_download_policy(allow_download):
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                theory = FairchemTheory(
                    model_name=str(method["model"]),
                    task_name=str(method["task_name"]),
                    device=device,
                    seed=seed,
                    numcores=cpu_cores,
                )
                ash_result = NEB(
                    reactant=Fragment(
                        xyzfile=str(reactant_path),
                        charge=charge,
                        mult=multiplicity,
                        printlevel=0,
                    ),
                    product=Fragment(
                        xyzfile=str(product_path),
                        charge=charge,
                        mult=multiplicity,
                        printlevel=0,
                    ),
                    theory=theory,
                    images=total_images - 2,
                    CI=climb,
                    maxiter=max_steps,
                    tol_max_fci=maximum_force,
                    tol_rms_fci=rms_force,
                    interpolation=interpolation,
                    idpp_maxiter=idpp_max_iterations,
                    charge=charge,
                    mult=multiplicity,
                    runmode="serial",
                    numcores=1,
                    printlevel=1,
                )
    except (ImportError, ModuleNotFoundError) as exc:
        return unavailable(
            f"ASH/FairChem NEB dependencies are unavailable: {exc}",
            install="Install ASH and fairchem-core in the ash_neb runtime.",
        )
    except SystemExit as exc:
        raise RuntimeError(f"ASH terminated the NEB calculation: {exc}") from exc
    finally:
        (directory / "stdout.log").write_text(
            stdout_buffer.getvalue(), encoding="utf-8"
        )
        (directory / "stderr.log").write_text(
            stderr_buffer.getvalue(), encoding="utf-8"
        )

    energy_mapping = getattr(ash_result, "MEP_energies_dict", None)
    energy_pairs, energy_index_complete, energy_warnings = _validated_energy_pairs(
        energy_mapping, total_images=total_images
    )
    path_file = _final_path_file(directory)
    initial_guess_path = directory / "initial_guess_path.xyz"
    images = (
        _parse_multixyz(
            path_file,
            charge=charge,
            multiplicity=multiplicity,
            reactant_structure=reactant,
            product_structure=product,
        )
        if path_file is not None
        else []
    )
    saddle_warnings: list[str] = []
    try:
        saddle = _saddle_structure(
            ash_result,
            symbols,
            charge=charge,
            multiplicity=multiplicity,
            reference_structure=reactant,
        )
    except ValueError:
        # ASH 的诊断几何损坏时，完整路径仍可提供最高内部图像；不要因诊断字段失败而
        # 丢弃可交接的路径，同时不把原始异常文本提升到模型可见结果。
        saddle = None
        saddle_warnings.append(
            "ASH returned invalid saddle geometry; the geometry was not used and "
            "the next available path candidate will be considered."
        )
    converged = (
        energy_index_complete
        and "fail" not in str(getattr(ash_result, "label", "")).casefold()
    )
    final_path_complete = path_file is not None and len(images) == total_images
    highest_index = (
        max(energy_pairs, key=lambda item: item[1])[0] if energy_pairs else None
    )
    interior_energy_pairs = [
        item for item in energy_pairs if 0 < item[0] < total_images - 1
    ]
    highest_interior_index = (
        max(interior_energy_pairs, key=lambda item: item[1])[0]
        if interior_energy_pairs
        else None
    )
    interior_peak_candidate_valid = (
        final_path_complete
        and energy_index_complete
        and highest_index is not None
        and highest_index == highest_interior_index
    )
    (
        transition_state_guess,
        transition_state_guess_source,
        transition_state_guess_source_image_index,
        guess_warnings,
    ) = (
        _recover_unvalidated_transition_state_guess(
            directory,
            images=images,
            symbols=symbols,
            charge=charge,
            multiplicity=multiplicity,
            reference_structure=reactant,
            ash_saddle=saddle,
            highest_interior_index=highest_interior_index,
            energy_index_complete=energy_index_complete,
            final_path_complete=final_path_complete,
            interior_peak_candidate_valid=interior_peak_candidate_valid,
            ash_saddle_warnings=saddle_warnings,
        )
    )
    result = {
        "images": images,
        "energies_hartree": [value for _index, value in energy_pairs],
        "energy_image_indices": [index for index, _value in energy_pairs],
        "energy_unit": "hartree",
        "highest_energy_image_index": highest_index,
        "highest_interior_image_index": highest_interior_index,
        "transition_state_guess": transition_state_guess,
        "transition_state_guess_source": transition_state_guess_source,
        "transition_state_guess_source_image_index": (
            transition_state_guess_source_image_index
        ),
        "transition_state_guess_selection_rule_version": (
            _TRANSITION_STATE_GUESS_SELECTION_RULE_VERSION
            if transition_state_guess_source is not None
            else None
        ),
        "diagnostic_ash_saddle_geometry": saddle,
        "interior_peak_candidate_valid": interior_peak_candidate_valid,
        "converged": converged,
        "energy_index_complete": energy_index_complete,
        "final_path_complete": final_path_complete,
        "total_images": total_images,
        "intermediate_images_passed_to_ash": total_images - 2,
        "climbing_image": climb,
        "maximum_force_ev_per_angstrom": maximum_force,
        "rms_force_ev_per_angstrom": rms_force,
        "maximum_steps": max_steps,
        "interpolation": interpolation.lower(),
        "model": str(method["model"]),
        "task_name": str(method["task_name"]),
        "device": device,
        "allow_model_download": allow_download,
        "path_file": relative_workspace_path(path_file) if path_file else None,
        "initial_guess_path": (
            relative_workspace_path(initial_guess_path)
            if initial_guess_path.is_file()
            else None
        ),
    }
    warnings = [
        "The highest interior NEB image and ASH saddle geometry are only transition-state "
        "guesses. Run an independent TS "
        "optimization, Hessian/mode check, and bidirectional IRC before assigning a TS."
    ]
    warnings.extend(guess_warnings)
    warnings.extend(energy_warnings)
    if (
        energy_index_complete
        and final_path_complete
        and not interior_peak_candidate_valid
    ):
        warnings.append(
            "The global maximum of the recovered band is an endpoint, so ASH/KNARR did "
            "not provide a defensible interior climbing-image candidate. The reported "
            "ASH saddle geometry is retained only as a diagnostic and must not be used as "
            "a transition-state guess."
        )
    if path_file is None:
        warning = (
            "ASH did not produce knarr_MEP_FULL.xyz or knarr_MEP.xyz; no final NEB path "
            "was recovered."
        )
        if initial_guess_path.is_file():
            warning += " initial_guess_path.xyz is diagnostic input and cannot satisfy success."
        warnings.append(warning)
    elif len(images) != total_images:
        warnings.append(
            f"Expected {total_images} path images but recovered {len(images)} from ASH output."
        )
    kwargs = {
        "artifact_files": command_artifacts(directory),
        "backend_version": module_version("ash"),
        "warnings": warnings,
        "provenance": {
            "engine": "ASH/KNARR NEB",
            "calculator": "ASH FairchemTheory",
            "model_download_permitted": allow_download,
            "endpoint_atom_order_checked": True,
            "endpoint_atom_maps_checked": _atom_map_sequence(
                reactant, endpoint="reactant"
            )
            is not None,
            "path_identity_metadata_restored": True,
            "endpoint_state_metadata_restored_from": ["reactant", "product"],
            "intermediate_topology_or_formal_charge_inferred": False,
            "path_is_validated_transition_state": False,
        },
    }
    if (
        converged
        and final_path_complete
        and interior_peak_candidate_valid
        and transition_state_guess is not None
        and not guess_warnings
    ):
        return success(result, **kwargs)
    return partial_success(result, **kwargs)


def execute(action_id: str, backend_id: str, request: dict[str, Any]) -> dict[str, Any]:
    if action_id == "search_reaction_path" and backend_id == "ash_neb_uma":
        return search_reaction_path(request)
    return unsupported(
        f"Unsupported ASH action/backend combination: {action_id}/{backend_id}"
    )
