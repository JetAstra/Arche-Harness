"""Scientific schema normalization and existing-output parsing actions."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from threading import Lock
from typing import Any

from .common import (
    atoms_and_coordinates,
    module_version,
    output_directory,
    partial_success,
    relative_workspace_path,
    request_parts,
    resolve_input_file,
    structure_dict,
    success,
    unwrap_artifact,
    unsupported,
    write_json,
    write_xyz,
)


ACTIONS = {
    "normalize_qcschema_molecule",
    "validate_qcschema_record",
    "parse_quantum_chemistry_output",
    "extract_final_structure",
    "render_spectrum_plot",
    "smiles2sdf",
    "sdf_to_gjf",
    "xyz_to_gjf",
}


def _model_json(model: Any) -> dict[str, Any]:
    return json.loads(model.json())


def _record_value(value: Any) -> Any:
    item = unwrap_artifact(value)
    if isinstance(item, dict) and "result" in item and len(item) == 1:
        item = item["result"]
    if isinstance(item, dict) and isinstance(item.get("path"), str):
        item = item["path"]
    if isinstance(item, (str, Path)):
        path = resolve_input_file(item)
        return json.loads(path.read_text(encoding="utf-8"))
    return item


def _normalize_qcschema_molecule(request: dict[str, Any]) -> dict[str, Any]:
    import qcelemental as qcel

    inputs, _method, settings = request_parts(request)
    value = _record_value(inputs["structure"])
    if isinstance(value, dict) and value.get("schema_name") == "qcschema_molecule":
        molecule = qcel.models.Molecule(**value)
    else:
        structure = structure_dict(value)
        symbols, coordinates = atoms_and_coordinates(structure)
        bohr_per_angstrom = 1.0 / 0.529177210903
        geometry = [
            [float(component * bohr_per_angstrom) for component in position]
            for position in coordinates
        ]
        molecule = qcel.models.Molecule(
            symbols=symbols,
            geometry=geometry,
            molecular_charge=int(structure.get("charge", 0)),
            molecular_multiplicity=int(structure.get("multiplicity", 1)),
            fix_com=bool(settings["fix_center_of_mass"]),
            fix_orientation=bool(settings["fix_orientation"]),
        )
    normalized = _model_json(molecule)
    directory = output_directory("normalize_qcschema_molecule", "qcelemental")
    path = write_json(directory, "qcschema_molecule.json", normalized)
    return success(
        {
            "molecule": normalized,
            "schema_name": normalized.get("schema_name"),
            "schema_version": normalized.get("schema_version"),
            "geometry_unit": "bohr",
            "atom_count": len(normalized.get("symbols") or []),
        },
        artifact_files=[
            {
                "path": relative_workspace_path(path),
                "semantic_type": "QCSchemaMolecule",
                "media_type": "application/json",
            }
        ],
        backend_version=module_version("qcelemental"),
    )


def _validate_qcschema_record(request: dict[str, Any]) -> dict[str, Any]:
    import qcelemental as qcel

    inputs, _method, settings = request_parts(request)
    record = _record_value(inputs["record"])
    if not isinstance(record, dict):
        raise ValueError("record must resolve to a JSON object")
    record_type = str(settings["record_type"]).strip().lower()
    models = {
        "molecule": qcel.models.Molecule,
        "atomic_input": qcel.models.AtomicInput,
        "atomic_result": qcel.models.AtomicResult,
        "optimization_input": qcel.models.OptimizationInput,
        "optimization_result": qcel.models.OptimizationResult,
    }
    if record_type not in models:
        raise ValueError(
            "record_type must be molecule, atomic_input, atomic_result, "
            "optimization_input, or optimization_result"
        )
    directory = output_directory("validate_qcschema_record", "qcelemental")
    try:
        normalized = _model_json(models[record_type](**record))
    except Exception as exc:
        errors = exc.errors() if hasattr(exc, "errors") else [{"message": str(exc)}]
        result = {
            "valid": False,
            "record_type": record_type,
            "errors": errors,
            "normalized_record": None,
        }
        path = write_json(directory, "validation_result.json", result)
        return success(
            result,
            artifact_files=[
                {
                    "path": relative_workspace_path(path),
                    "semantic_type": "QCSchemaValidationResult",
                    "media_type": "application/json",
                }
            ],
            backend_version=module_version("qcelemental"),
        )
    result = {
        "valid": True,
        "record_type": record_type,
        "errors": [],
        "normalized_record": normalized,
        "schema_name": normalized.get("schema_name"),
        "schema_version": normalized.get("schema_version"),
    }
    path = write_json(directory, "validated_record.json", result)
    return success(
        result,
        artifact_files=[
            {
                "path": relative_workspace_path(path),
                "semantic_type": "QCSchemaValidationResult",
                "media_type": "application/json",
            }
        ],
        backend_version=module_version("qcelemental"),
    )


_CCLIB_GROUPS: dict[str, tuple[str, ...]] = {
    "metadata": (
        "metadata", "charge", "mult", "natom", "nbasis", "nmo", "coreelectrons",
    ),
    "atom_coordinates": ("atomcoords", "atomnos", "atommasses"),
    "energies": (
        "scfenergies", "mpenergies", "ccenergies", "dispersionenergies",
        "freeenergy", "enthalpy", "entropy", "temperature", "zpve",
    ),
    "gradients": ("grads",),
    "hessian": ("hessian",),
    "vibrational_frequencies": ("vibfreqs", "vibdisps"),
    "vibrational_intensities": ("vibirs", "vibramans", "vibfconsts", "vibrmasses"),
    "molecular_orbitals": ("moenergies", "homos", "mosyms", "mocoeffs"),
    "charges": ("atomcharges", "atomspins"),
    "multipoles": ("moments", "polarizabilities"),
    "excited_states": ("etenergies", "etoscs", "etsyms", "etsecs", "etrotats"),
    "nmr": ("atomnos", "nmrtensors", "nmrcouplingtensors"),
}


_CCLIB_UNITS = {
    "atomcoords": "angstrom",
    "scfenergies": "eV",
    "mpenergies": "eV",
    "ccenergies": "eV",
    "dispersionenergies": "eV",
    "freeenergy": "hartree_per_particle",
    "enthalpy": "hartree_per_particle",
    "entropy": "hartree_per_particle_kelvin",
    "temperature": "kelvin",
    "zpve": "hartree_per_particle",
    "grads": "hartree_per_bohr",
    "hessian": "hartree_per_bohr_squared",
    "vibfreqs": "inverse_centimeter",
    "vibirs": "kilometer_per_mole",
    "vibramans": "angstrom_fourth_per_amu",
    "moenergies": "eV",
    "etenergies": "inverse_centimeter",
    "nmrtensors": "parts_per_million",
    "nmrcouplingtensors": "hertz",
}


_CCLIB_ORCA_PARSE_LOCK = Lock()


def _array_elements(value: Any) -> int:
    if isinstance(value, dict):
        return sum(_array_elements(item) for item in value.values())
    if isinstance(value, list):
        return sum(_array_elements(item) for item in value)
    return 1


def _convert_cclib_result_to_json(value: Any) -> Any:
    """把 cclib 属性转成与落盘 Artifact 完全一致的 JSON 值。

    输入是 ``getattributes(tolists=True)`` 产生的属性树；函数不修改原对象，也不写文件，
    而是按 Backend Artifact 既有的 ``default=str`` 规则转换 ``timedelta`` 等残留 Python
    对象并返回新的 JSON 值。若编码结果无法再解析为 JSON，则原样抛出序列化异常。
    """

    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _append_orca_scf_targets_compat(parser: Any, inputfile: Any, line: str) -> bool:
    """Handle ORCA blocks whose first convergence report omits RMS-density."""

    while "Last Energy change" not in line:
        line = next(inputfile)
    delta_energy_value = float(line.split()[4])
    delta_energy_target = float(line.split()[7])
    line = next(inputfile)
    used_workaround = False
    if "Last MAX-Density change" in line:
        maximum_density_value = float(line.split()[4])
        maximum_density_target = float(line.split()[7])
        line = next(inputfile)
        if "Last RMS-Density change" in line:
            rms_density_value = float(line.split()[4])
            rms_density_target = float(line.split()[7])
        else:
            previous_values = parser.scfvalues[-1][-1]
            rms_density_value = (
                float(previous_values[2]) if len(previous_values) > 2 else math.nan
            )
            if parser.scftargets:
                rms_density_target = float(parser.scftargets[-1][2])
                if delta_energy_target != parser.scftargets[-1][0]:
                    raise ValueError("ORCA SCF energy target changed unexpectedly")
                if maximum_density_target != parser.scftargets[-1][1]:
                    raise ValueError("ORCA SCF maximum-density target changed unexpectedly")
            else:
                # ORCA 4 may omit the RMS target in the first convergence
                # summary. cclib 1.8.1 indexes a nonexistent previous target.
                # Preserve the missing value as NaN; it is parser metadata and
                # does not alter any electronic energy or requested property.
                rms_density_target = math.nan
                used_workaround = True
        parser.scfvalues[-1].append(
            [delta_energy_value, maximum_density_value, rms_density_value]
        )
        parser.scftargets.append(
            [delta_energy_target, maximum_density_target, rms_density_target]
        )
    return used_workaround


def _ccread_with_orca_compatibility(source: Path) -> tuple[Any, list[str]]:
    """Run cclib with a narrow ORCA 4/cclib 1.8.1 convergence-block fix."""

    from cclib.io import ccread
    from cclib.parser.orcaparser import ORCA

    workaround_used = False
    with _CCLIB_ORCA_PARSE_LOCK:
        original = ORCA._append_scfvalues_scftargets

        def patched(parser: Any, inputfile: Any, line: str) -> None:
            nonlocal workaround_used
            workaround_used = (
                _append_orca_scf_targets_compat(parser, inputfile, line)
                or workaround_used
            )

        ORCA._append_scfvalues_scftargets = patched
        try:
            parsed = ccread(str(source), loglevel=40)
        except Exception as exc:
            raise RuntimeError(
                f"cclib could not parse {source.name}: {type(exc).__name__}: {exc}"
            ) from exc
        finally:
            ORCA._append_scfvalues_scftargets = original
    warnings = []
    if workaround_used:
        warnings.append(
            "Applied the cclib 1.8.1 compatibility fix for an ORCA convergence block "
            "whose first summary omitted the RMS-density target; electronic energies "
            "and requested scientific properties were not changed."
        )
    return parsed, warnings


def _parse_quantum_output(request: dict[str, Any]) -> dict[str, Any]:
    import cclib

    inputs, _method, settings = request_parts(request)
    source = resolve_input_file(inputs["output_file"])
    requested = settings["properties"]
    if not isinstance(requested, (list, tuple)) or not requested:
        raise ValueError("properties must be a non-empty list")
    groups = []
    for value in requested:
        group = str(value).strip().lower()
        if group not in _CCLIB_GROUPS:
            raise ValueError(
                f"Unknown cclib property group {group!r}; choose from {sorted(_CCLIB_GROUPS)}"
            )
        if group not in groups:
            groups.append(group)
    coordinate_frames = str(settings["coordinate_frames"]).strip().lower()
    if coordinate_frames not in {"last", "all"}:
        raise ValueError("coordinate_frames must be last or all")
    for name in ("include_orbital_coefficients", "include_excited_state_configurations"):
        if not isinstance(settings[name], bool):
            raise ValueError(f"{name} must be an explicit boolean")
    maximum = int(settings["max_array_elements"])
    if maximum < 1 or maximum > 100000000:
        raise ValueError("max_array_elements must be between 1 and 100000000")

    parsed, parser_warnings = _ccread_with_orca_compatibility(source)
    if parsed is None:
        raise RuntimeError("cclib could not recognize or parse the supplied output file")
    attributes = parsed.getattributes(tolists=True)
    selected: dict[str, Any] = {}
    missing_groups = []
    for group in groups:
        present = False
        for attribute in _CCLIB_GROUPS[group]:
            if attribute == "mocoeffs" and not settings["include_orbital_coefficients"]:
                continue
            if attribute == "etsecs" and not settings["include_excited_state_configurations"]:
                continue
            if attribute not in attributes:
                continue
            value = attributes[attribute]
            if attribute == "atomcoords" and coordinate_frames == "last" and value:
                value = value[-1]
            selected[attribute] = value
            present = True
        if not present:
            missing_groups.append(group)
    selected = _convert_cclib_result_to_json(selected)
    element_count = _array_elements(selected)
    if element_count > maximum:
        raise ValueError(
            f"Selected cclib data contains {element_count} scalar elements, exceeding "
            f"max_array_elements={maximum}; request fewer groups or raise the explicit limit"
        )
    metadata = dict(attributes.get("metadata") or {})
    result = {
        "parser": metadata.get("package") or parsed.__class__.__module__.split(".")[-1],
        "parser_class": parsed.__class__.__name__,
        "source_path": relative_workspace_path(source),
        "requested_property_groups": groups,
        "missing_property_groups": missing_groups,
        "coordinate_frames": coordinate_frames,
        "properties": selected,
        "attribute_units": {
            name: unit for name, unit in _CCLIB_UNITS.items() if name in selected
        },
        "scalar_element_count": element_count,
    }
    directory = output_directory("parse_quantum_chemistry_output", "cclib")
    path = write_json(directory, "parsed_quantum_output.json", result)
    version = getattr(cclib, "__version__", None) or module_version("cclib")
    kwargs = {
        "artifact_files": [
            {
                "path": relative_workspace_path(path),
                "semantic_type": "ParsedQuantumChemistryResult",
                "media_type": "application/json",
            }
        ],
        "backend_version": version,
        "warnings": [
            *parser_warnings,
            *(
                [f"Requested property groups not present in this output: {', '.join(missing_groups)}"]
                if missing_groups else []
            ),
        ],
    }
    return partial_success(result, **kwargs) if missing_groups else success(result, **kwargs)


def _lookup_periodic_table_element(periodic_table: Any, atomic_number: int) -> str:
    """按原子序数安全读取 cclib 元素表，并拒绝空值或越界值。

    输入是 cclib ``PeriodicTable`` 实例和已经转换为整数的原子序数；函数兼容旧式
    映射表与 cclib 1.8.1 使用的序列表，只读取表项，不产生文件或其他副作用。未知、
    越界、空元素或不支持的表类型都会以 ``ValueError`` 报告，避免把错误元素写入结果。
    """

    table = getattr(periodic_table, "element", None)
    element: Any
    if isinstance(table, Mapping):
        element = table.get(atomic_number)
    elif isinstance(table, Sequence) and not isinstance(table, (str, bytes, bytearray)):
        if atomic_number < 0 or atomic_number >= len(table):
            element = None
        else:
            element = table[atomic_number]
    else:
        raise ValueError("cclib PeriodicTable.element must be a mapping or sequence")
    if not isinstance(element, str) or not element.strip():
        raise ValueError(f"cclib returned an unknown atomic number: {atomic_number}")
    return element.strip()


def _coerce_structure_integer(value: Any, field_name: str) -> int:
    """把参考结构的电荷、重数或 map 值转为整数并拒绝布尔和小数。"""

    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{field_name} must be an integer")
    return parsed


def _reference_atom_map_numbers(
    reference: dict[str, Any], atom_count: int
) -> list[int] | None:
    """读取参考结构的完整原子 map 序列，并拒绝部分或互相矛盾的身份字段。"""

    atoms = reference.get("atoms")
    if not isinstance(atoms, list):
        raise ValueError("reference_structure.atoms must be a list")
    atom_map_present = [
        isinstance(atom, Mapping) and "atom_map_number" in atom for atom in atoms
    ]
    if any(atom_map_present) and not all(atom_map_present):
        raise ValueError(
            "reference_structure.atom_map_number must be present for every atom or none"
        )
    atom_maps: list[int] | None = None
    if all(atom_map_present):
        atom_maps = [
            _coerce_structure_integer(
                atom["atom_map_number"],
                f"reference_structure.atoms[{index}].atom_map_number",
            )
            for index, atom in enumerate(atoms)
        ]

    list_maps: list[int] | None = None
    if "atom_map_numbers" in reference:
        raw_maps = reference["atom_map_numbers"]
        if not isinstance(raw_maps, (list, tuple)) or len(raw_maps) != atom_count:
            raise ValueError(
                "reference_structure.atom_map_numbers must align one-to-one with atoms"
            )
        list_maps = [
            _coerce_structure_integer(value, f"reference_structure.atom_map_numbers[{index}]")
            for index, value in enumerate(raw_maps)
        ]

    if atom_maps is not None and list_maps is not None and atom_maps != list_maps:
        raise ValueError(
            "reference_structure atom_map_number and atom_map_numbers disagree"
        )
    selected_maps = list_maps if list_maps is not None else atom_maps
    if selected_maps is None:
        return None
    if any(map_number <= 0 for map_number in selected_maps) or len(
        set(selected_maps)
    ) != len(selected_maps):
        raise ValueError(
            "reference_structure atom map numbers must be positive and unique"
        )
    return selected_maps


def _validate_reference_structure_identity(
    reference_value: Any,
    atoms: list[dict[str, Any]],
    charge: int,
    multiplicity: int,
) -> list[int] | None:
    """按原子索引核对参考结构并返回可复制的 map 序列。

    输入可以是内联 AtomicStructure 或 ArtifactRef；函数只比较原子数、元素顺序、电荷
    和自旋多重度，任何差异立即抛出 ``ValueError``。它不尝试图同构、坐标匹配、重排或
    其他猜测；参考结构没有 map 字段时返回 ``None``。
    """

    reference = structure_dict(reference_value)
    reference_atoms, _coordinates = atoms_and_coordinates(reference)
    if len(reference_atoms) != len(atoms):
        raise ValueError(
            "reference_structure atom count does not match the parsed final frame"
        )
    for index, (reference_element, parsed_atom) in enumerate(
        zip(reference_atoms, atoms)
    ):
        if str(reference_element).casefold() != str(parsed_atom["element"]).casefold():
            raise ValueError(
                "reference_structure element order does not match parsed final frame "
                f"at atom index {index}"
            )
    if "charge" not in reference or "multiplicity" not in reference:
        raise ValueError(
            "reference_structure must explicitly declare charge and multiplicity"
        )
    reference_charge = _coerce_structure_integer(
        reference["charge"], "reference_structure.charge"
    )
    reference_multiplicity = _coerce_structure_integer(
        reference["multiplicity"], "reference_structure.multiplicity"
    )
    if reference_charge != charge:
        raise ValueError(
            "reference_structure charge does not match parsed molecular charge"
        )
    if reference_multiplicity != multiplicity:
        raise ValueError(
            "reference_structure multiplicity does not match parsed spin multiplicity"
        )
    return _reference_atom_map_numbers(reference, len(atoms))


def _extract_final_structure(request: dict[str, Any]) -> dict[str, Any]:
    """提取 cclib 最后一帧并保留明确的电子态与可验证的原子身份。"""

    import cclib
    from cclib.parser.utils import PeriodicTable

    inputs, _method, settings = request_parts(request)
    require_normal = settings["require_normal_termination"]
    if not isinstance(require_normal, bool):
        raise ValueError(
            "action_settings.require_normal_termination must be an explicit boolean"
        )
    source = resolve_input_file(inputs["output_file"])
    parsed, parser_warnings = _ccread_with_orca_compatibility(source)
    if parsed is None:
        raise RuntimeError("cclib could not recognize or parse the supplied output file")
    attributes = parsed.getattributes(tolists=True)
    coordinates = attributes.get("atomcoords") or []
    atomic_numbers = attributes.get("atomnos") or []
    if not coordinates:
        raise ValueError("The supplied output contains no coordinate frames")
    if not atomic_numbers:
        raise ValueError("The supplied output contains no atomic-number array")
    final_coordinates = coordinates[-1]
    if len(final_coordinates) != len(atomic_numbers):
        raise ValueError(
            "The final coordinate frame is not aligned with the parsed atomic-number array"
        )

    periodic_table = PeriodicTable()
    atoms = []
    for atomic_number, position in zip(atomic_numbers, final_coordinates):
        number = int(atomic_number)
        element = _lookup_periodic_table_element(periodic_table, number)
        if not isinstance(position, (list, tuple)) or len(position) != 3:
            raise ValueError(
                "Every coordinate in the final cclib frame must contain exactly "
                "three values"
            )
        coordinates_angstrom = [float(value) for value in position]
        if not all(math.isfinite(value) for value in coordinates_angstrom):
            raise ValueError("The final cclib coordinate frame contains non-finite values")
        atoms.append(
            {
                "element": str(element),
                "atomic_number": number,
                "position_angstrom": coordinates_angstrom,
            }
        )

    metadata = dict(attributes.get("metadata") or {})
    normal_termination = metadata.get("success") is True
    charge_value = attributes.get("charge")
    multiplicity_value = attributes.get("mult")
    if charge_value is None or multiplicity_value is None:
        raise ValueError(
            "cclib did not report both molecular charge and spin multiplicity; "
            "extract_final_structure refuses to invent the missing electronic state"
        )
    charge = int(charge_value)
    multiplicity = int(multiplicity_value)
    if multiplicity < 1:
        raise ValueError("cclib returned an invalid spin multiplicity")
    reference_provided = "reference_structure" in inputs
    reference_maps: list[int] | None = None
    if reference_provided:
        if inputs["reference_structure"] is None:
            raise ValueError("inputs.reference_structure cannot be null")
        reference_maps = _validate_reference_structure_identity(
            inputs["reference_structure"], atoms, charge, multiplicity
        )
        if reference_maps is not None:
            for atom, map_number in zip(atoms, reference_maps):
                atom["atom_map_number"] = map_number
    structure = {
        "atoms": atoms,
        "charge": charge,
        "multiplicity": multiplicity,
        "pbc": [False, False, False],
        "metadata": {
            "source_path": relative_workspace_path(source),
            "coordinate_frame_index": len(coordinates) - 1,
            "coordinate_frame_count": len(coordinates),
            "normal_termination": normal_termination,
            "parser": metadata.get("package")
            or parsed.__class__.__module__.split(".")[-1],
            "parser_class": parsed.__class__.__name__,
        },
    }
    if reference_maps is not None:
        structure["atom_map_numbers"] = list(reference_maps)
    if reference_provided:
        structure["metadata"]["identity_source"] = "reference_structure"
    result = {
        "structure": structure,
        "coordinate_frame_index": len(coordinates) - 1,
        "coordinate_frame_count": len(coordinates),
        "normal_termination": normal_termination,
        "source_path": relative_workspace_path(source),
        "require_normal_termination": require_normal,
    }
    if reference_provided:
        result["identity_source"] = "reference_structure"
    directory = output_directory("extract_final_structure", "cclib")
    xyz_path = write_xyz(structure, directory / "final_structure.xyz")
    kwargs = {
        "artifact_files": [
            {
                "path": relative_workspace_path(xyz_path),
                "semantic_type": "AtomicStructureFile",
                "media_type": "chemical/x-xyz",
            }
        ],
        "backend_version": getattr(cclib, "__version__", None)
        or module_version("cclib"),
        "provenance": {
            "coordinate_selection": "last_parsed_frame",
            "geometry_convergence_inferred": False,
            **(
                {"identity_source": "reference_structure"}
                if reference_provided
                else {}
            ),
        },
    }
    warnings = list(parser_warnings)
    if not normal_termination:
        warnings.append(
            "The source output does not report normal termination; the returned last frame "
            "must not be treated as a converged structure."
        )
    if require_normal and not normal_termination:
        return partial_success(result, warnings=warnings, **kwargs)
    return success(result, warnings=warnings, **kwargs)


def execute(action_id: str, backend_id: str, request: dict[str, Any]) -> dict[str, Any]:
    if backend_id == "format_conversion" and action_id in {
        "smiles2sdf",
        "sdf_to_gjf",
        "xyz_to_gjf",
    }:
        from .format_conversion import execute as execute_author_format_conversion

        return execute_author_format_conversion(action_id, backend_id, request)
    if backend_id == "qcelemental" and action_id == "normalize_qcschema_molecule":
        return _normalize_qcschema_molecule(request)
    if backend_id == "qcelemental" and action_id == "validate_qcschema_record":
        return _validate_qcschema_record(request)
    if backend_id == "cclib" and action_id == "parse_quantum_chemistry_output":
        return _parse_quantum_output(request)
    if backend_id == "cclib" and action_id == "extract_final_structure":
        return _extract_final_structure(request)
    if backend_id == "internal_plotting" and action_id == "render_spectrum_plot":
        from .plotting import execute as execute_plotting

        return execute_plotting(action_id, backend_id, request)
    return unsupported(f"Unsupported interchange action/backend combination: {action_id}/{backend_id}")
