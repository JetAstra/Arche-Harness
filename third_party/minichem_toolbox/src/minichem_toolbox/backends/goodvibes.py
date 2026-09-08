"""GoodVibes 4.3 structured thermochemistry and reaction-analysis adapter."""

from __future__ import annotations

import glob
import hashlib
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

import yaml

from .common import (
    command_artifacts,
    module_version,
    output_directory,
    relative_workspace_path,
    request_parts,
    resolve_input_file,
    run_external,
    success,
    unavailable,
)


GOODVIBES_ACTIONS = {
    "derive_thermochemistry",
    "scan_thermochemistry_temperature",
    "analyze_thermochemical_ensemble",
    "validate_thermochemistry_inputs",
    "analyze_thermochemical_selectivity",
    "analyze_reaction_free_energy_profile",
}

_SAFE_SUFFIX = re.compile(r"^[A-Za-z0-9_.-]+$")
_SAFE_EXTENSION = re.compile(r"^\.[A-Za-z0-9]+$")


def _finite_float(value: Any, *, field: str, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number, not a boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    if minimum is not None and number < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return number


def _positive_float(value: Any, *, field: str) -> float:
    number = _finite_float(value, field=field)
    if number <= 0:
        raise ValueError(f"{field} must be positive")
    return number


def _explicit_bool(settings: dict[str, Any], field: str) -> bool:
    value = settings[field]
    if not isinstance(value, bool):
        raise ValueError(f"action_settings.{field} must be an explicit boolean")
    return value


def _resolve_output_files(value: Any, *, field: str = "output_files") -> list[Path]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"inputs.{field} must be a non-empty list of file references")
    if len(value) > 10000:
        raise ValueError(f"inputs.{field} may contain at most 10000 files")
    paths = [resolve_input_file(item) for item in value]
    if any(not path.is_file() for path in paths):
        raise ValueError(f"every inputs.{field} member must resolve to a regular file")
    resolved = [path.resolve() for path in paths]
    if len(resolved) != len(set(resolved)):
        raise ValueError(f"inputs.{field} contains duplicate files")
    return resolved


def _temperature_list(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("inputs.temperatures_kelvin must be a non-empty list")
    if len(value) > 500:
        raise ValueError("inputs.temperatures_kelvin may contain at most 500 points")
    temperatures = [
        _positive_float(item, field=f"inputs.temperatures_kelvin[{index}]")
        for index, item in enumerate(value)
    ]
    if len(temperatures) != len(set(temperatures)):
        raise ValueError("inputs.temperatures_kelvin contains duplicate points")
    return temperatures


def _scale_argument(value: Any, *, field: str) -> float | str:
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"auto", "same_as_frequency"}:
            return normalized
        raise ValueError(
            f"action_settings.{field} must be a positive number, 'auto', or "
            "'same_as_frequency' where documented"
        )
    return _positive_float(value, field=f"action_settings.{field}")


def _common_arguments(
    settings: dict[str, Any],
    method: dict[str, Any],
    *,
    temperature_kelvin: float,
) -> list[str]:
    temperature = _positive_float(
        temperature_kelvin, field="action_settings.temperature_kelvin"
    )
    arguments = ["--temp", f"{temperature:.12g}"]

    standard_state = str(settings["standard_state"]).strip().casefold()
    if standard_state == "gas_1atm":
        pass
    elif standard_state == "solution_1mol_l":
        arguments.extend(["--conc", "1.0"])
    elif standard_state == "custom_concentration":
        if "concentration_mol_l" not in settings:
            raise ValueError(
                "standard_state=custom_concentration requires "
                "action_settings.concentration_mol_l"
            )
        concentration = _positive_float(
            settings["concentration_mol_l"],
            field="action_settings.concentration_mol_l",
        )
        arguments.extend(["--conc", f"{concentration:.12g}"])
    else:
        raise ValueError(
            "action_settings.standard_state must be gas_1atm, solution_1mol_l, "
            "or custom_concentration"
        )

    entropy_model = str(settings["entropy_model"]).strip().casefold()
    if entropy_model not in {"rrho", "grimme", "truhlar"}:
        raise ValueError("action_settings.entropy_model must be rrho, grimme, or truhlar")
    if entropy_model in {"grimme", "truhlar"}:
        if "entropy_frequency_cutoff_cm1" not in settings:
            raise ValueError(
                f"entropy_model={entropy_model} requires "
                "action_settings.entropy_frequency_cutoff_cm1"
            )
        cutoff = _finite_float(
            settings["entropy_frequency_cutoff_cm1"],
            field="action_settings.entropy_frequency_cutoff_cm1",
            minimum=0.0,
        )
        arguments.extend(["--qs", entropy_model, "--fs", f"{cutoff:.12g}"])
        if entropy_model == "grimme":
            inertia = str(settings.get("free_rotor_inertia_model", "")).strip().casefold()
            inertia_value = {"global": "global", "per_conformer": "conf"}.get(inertia)
            if inertia_value is None:
                raise ValueError(
                    "entropy_model=grimme requires action_settings.free_rotor_inertia_model "
                    "equal to global or per_conformer"
                )
            arguments.extend(["--bav", inertia_value])

    enthalpy_model = str(settings["enthalpy_model"]).strip().casefold()
    if enthalpy_model not in {"rrho", "head_gordon"}:
        raise ValueError("action_settings.enthalpy_model must be rrho or head_gordon")
    if enthalpy_model == "head_gordon":
        if "enthalpy_frequency_cutoff_cm1" not in settings:
            raise ValueError(
                "enthalpy_model=head_gordon requires "
                "action_settings.enthalpy_frequency_cutoff_cm1"
            )
        cutoff = _finite_float(
            settings["enthalpy_frequency_cutoff_cm1"],
            field="action_settings.enthalpy_frequency_cutoff_cm1",
            minimum=0.0,
        )
        arguments.extend(["--qh", "--fh", f"{cutoff:.12g}"])

    frequency_scale = _scale_argument(
        settings["frequency_scale_factor"], field="frequency_scale_factor"
    )
    zpe_scale = _scale_argument(settings["zpe_scale_factor"], field="zpe_scale_factor")
    if frequency_scale != "auto":
        if frequency_scale == "same_as_frequency":
            raise ValueError("frequency_scale_factor cannot be same_as_frequency")
        arguments.extend(["--vscal", f"{frequency_scale:.12g}"])
    if zpe_scale == "auto":
        if frequency_scale != "auto":
            raise ValueError(
                "zpe_scale_factor='auto' cannot be represented faithfully when an explicit "
                "frequency_scale_factor is supplied; choose a numeric ZPE scale or "
                "same_as_frequency"
            )
    elif zpe_scale == "same_as_frequency":
        if frequency_scale == "auto":
            raise ValueError(
                "zpe_scale_factor='same_as_frequency' requires a numeric "
                "frequency_scale_factor; choose zpe_scale_factor='auto' for GoodVibes' "
                "independent level-of-theory ZPE lookup"
            )
    else:
        arguments.extend(["--zpe-vscal", f"{zpe_scale:.12g}"])

    if _explicit_bool(settings, "symmetry_correction"):
        arguments.append("--symm")

    imaginary_policy = str(settings["imaginary_frequency_policy"]).strip().casefold()
    if imaginary_policy == "retain":
        pass
    elif imaginary_policy == "invert_below_threshold":
        if "imaginary_frequency_threshold_cm1" not in settings:
            raise ValueError(
                "imaginary_frequency_policy=invert_below_threshold requires "
                "action_settings.imaginary_frequency_threshold_cm1"
            )
        threshold = _positive_float(
            settings["imaginary_frequency_threshold_cm1"],
            field="action_settings.imaginary_frequency_threshold_cm1",
        )
        arguments.extend(["--invert", f"{threshold:.12g}"])
    else:
        raise ValueError(
            "action_settings.imaginary_frequency_policy must be retain or "
            "invert_below_threshold"
        )

    suffix = method.get("single_point_correction_suffix")
    if suffix is not None:
        suffix_text = str(suffix).strip()
        if not suffix_text or not _SAFE_SUFFIX.fullmatch(suffix_text):
            raise ValueError(
                "method_spec.single_point_correction_suffix must contain only letters, "
                "digits, dot, underscore, or hyphen"
            )
        arguments.extend(["--spc", suffix_text])

    extensions = method.get("custom_file_extensions")
    if extensions is not None:
        if not isinstance(extensions, (list, tuple)) or not extensions:
            raise ValueError("method_spec.custom_file_extensions must be a non-empty list")
        normalized_extensions = []
        for extension in extensions:
            value = str(extension).strip()
            if not _SAFE_EXTENSION.fullmatch(value):
                raise ValueError(
                    "every custom_file_extensions entry must start with a dot and contain "
                    "only letters or digits"
                )
            normalized_extensions.append(value)
        arguments.extend(["--custom_ext", ",".join(normalized_extensions)])

    exclude_pattern = method.get("exclude_pattern")
    if exclude_pattern is not None:
        pattern = str(exclude_pattern).strip()
        if not pattern:
            raise ValueError("method_spec.exclude_pattern cannot be empty")
        arguments.extend(["--exclude", pattern])

    free_space_solvent = method.get("free_space_solvent")
    if free_space_solvent is not None:
        solvent = str(free_space_solvent).strip()
        if not solvent:
            raise ValueError("method_spec.free_space_solvent cannot be empty")
        arguments.extend(["--freespace", solvent])

    media_solvent = method.get("media_solvent")
    if media_solvent is not None:
        solvent = str(media_solvent).strip()
        if not solvent:
            raise ValueError("method_spec.media_solvent cannot be empty")
        arguments.extend(["--media", solvent])

    return arguments


def _deduplication_arguments(settings: dict[str, Any]) -> list[str]:
    if not _explicit_bool(settings, "deduplicate_structures"):
        return []
    required = (
        "duplicate_energy_cutoff_kcal_mol",
        "duplicate_rotational_cutoff_fraction",
        "duplicate_rmsd_cutoff_angstrom",
    )
    missing = [field for field in required if field not in settings]
    if missing:
        raise ValueError(
            "deduplicate_structures=true requires explicit duplicate cutoffs: "
            + ", ".join(missing)
        )
    energy = _finite_float(
        settings["duplicate_energy_cutoff_kcal_mol"],
        field="action_settings.duplicate_energy_cutoff_kcal_mol",
        minimum=0.0,
    )
    rotational = _finite_float(
        settings["duplicate_rotational_cutoff_fraction"],
        field="action_settings.duplicate_rotational_cutoff_fraction",
        minimum=0.0,
    )
    arguments = [
        "--dedup",
        "--e_cutoff",
        f"{energy:.12g}",
        "--ro_cutoff",
        f"{rotational:.12g}",
    ]
    rmsd = settings["duplicate_rmsd_cutoff_angstrom"]
    if rmsd is not None:
        value = _positive_float(
            rmsd, field="action_settings.duplicate_rmsd_cutoff_angstrom"
        )
        arguments.extend(["--rmsd_cutoff", f"{value:.12g}"])
    return arguments


def _validation_cutoff_arguments(settings: dict[str, Any]) -> list[str]:
    energy = _finite_float(
        settings["duplicate_energy_cutoff_kcal_mol"],
        field="action_settings.duplicate_energy_cutoff_kcal_mol",
        minimum=0.0,
    )
    rotational = _finite_float(
        settings["duplicate_rotational_cutoff_fraction"],
        field="action_settings.duplicate_rotational_cutoff_fraction",
        minimum=0.0,
    )
    arguments = [
        "--e_cutoff",
        f"{energy:.12g}",
        "--ro_cutoff",
        f"{rotational:.12g}",
    ]
    rmsd = settings["duplicate_rmsd_cutoff_angstrom"]
    if rmsd is not None:
        value = _positive_float(
            rmsd, field="action_settings.duplicate_rmsd_cutoff_angstrom"
        )
        arguments.extend(["--rmsd_cutoff", f"{value:.12g}"])
    return arguments


def _population_arguments(settings: dict[str, Any]) -> list[str]:
    basis = str(settings["population_basis"]).strip().casefold()
    if basis == "electronic_energy":
        return ["--boltz", "energy"]
    if basis == "quasi_harmonic_gibbs":
        if str(settings["entropy_model"]).strip().casefold() == "rrho":
            raise ValueError(
                "population_basis=quasi_harmonic_gibbs requires entropy_model=grimme "
                "or truhlar"
            )
        return ["--boltz", "gibbs"]
    raise ValueError(
        "action_settings.population_basis must be electronic_energy or "
        "quasi_harmonic_gibbs"
    )


def _normalize_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_paths(item) for item in value]
    if isinstance(value, str):
        path = Path(value)
        if path.is_absolute():
            try:
                return relative_workspace_path(path)
            except (OSError, ValueError):
                pass
    return value


def _selected_thermochemistry(
    entry: dict[str, Any], settings: dict[str, Any], temperature_kelvin: float
) -> dict[str, Any]:
    thermo = dict(entry.get("thermo") or {})
    enthalpy_field = (
        "qh_enthalpy"
        if str(settings["enthalpy_model"]).strip().casefold() == "head_gordon"
        else "enthalpy"
    )
    entropy_field = (
        "qh_entropy"
        if str(settings["entropy_model"]).strip().casefold() in {"grimme", "truhlar"}
        else "entropy"
    )
    enthalpy = thermo.get(enthalpy_field)
    entropy = thermo.get(entropy_field)
    selected_gibbs = None
    if enthalpy is not None and entropy is not None:
        selected_gibbs = float(enthalpy) - float(temperature_kelvin) * float(entropy)
    result = dict(entry)
    result["selected_thermochemistry"] = {
        "temperature_kelvin": float(temperature_kelvin),
        "enthalpy_model": settings["enthalpy_model"],
        "entropy_model": settings["entropy_model"],
        "enthalpy_field": enthalpy_field,
        "entropy_field": entropy_field,
        "enthalpy_hartree": enthalpy,
        "entropy_hartree_per_kelvin": entropy,
        "gibbs_free_energy_hartree": selected_gibbs,
    }
    return _normalize_paths(result)


def _run_once(
    *,
    directory: Path,
    output_files: list[Path],
    arguments: list[str],
    output_stem: str,
    timeout_seconds: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    json_name = f"{output_stem}.json"
    completed = run_external(
        executable="goodvibes",
        environment_variable="CHEMGRAPH_GOODVIBES_COMMAND",
        arguments=[
            *(glob.escape(str(path)) for path in output_files),
            *arguments,
            "--output",
            output_stem,
            "--json",
            json_name,
        ],
        directory=directory,
        timeout_seconds=timeout_seconds,
    )
    (directory / f"{output_stem}.stdout.log").write_text(
        completed["stdout"], encoding="utf-8"
    )
    (directory / f"{output_stem}.stderr.log").write_text(
        completed["stderr"], encoding="utf-8"
    )
    if not completed["available"]:
        return None, completed
    if completed.get("timeout"):
        raise TimeoutError(f"GoodVibes exceeded {timeout_seconds} seconds")
    if completed["returncode"] != 0:
        detail = (completed["stderr"] or completed["stdout"])[-4000:]
        raise RuntimeError(f"GoodVibes failed with exit code {completed['returncode']}: {detail}")
    json_path = directory / json_name
    if not json_path.is_file():
        raise RuntimeError("GoodVibes completed without writing the requested structured JSON")
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"GoodVibes wrote invalid structured JSON: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise RuntimeError("GoodVibes structured JSON is missing its results list")
    return payload, completed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_or_link(source: Path, target: Path) -> None:
    try:
        target.hardlink_to(source)
    except OSError:
        shutil.copy2(source, target)


def _matching_single_point_file(output_file: Path, suffix: str) -> Path | None:
    candidates = [
        output_file.with_name(f"{output_file.stem}_{suffix}{extension}")
        for extension in dict.fromkeys((output_file.suffix, ".log", ".out"))
    ]
    return next((path.resolve() for path in candidates if path.is_file()), None)


def _stage_derive_inputs(
    *,
    directory: Path,
    output_file: Path,
    single_point_output_file: Path | None,
    single_point_suffix: str | None,
) -> tuple[list[Path], list[dict[str, Any]]]:
    frequency_extension = output_file.suffix.casefold()
    if frequency_extension not in {".log", ".out"}:
        frequency_extension = ".log"
    staged_frequency = directory / f"frequency{frequency_extension}"
    _copy_or_link(output_file, staged_frequency)
    records = [
        {
            "role": "frequency_output",
            "source_path": relative_workspace_path(output_file),
            "staged_path": staged_frequency.name,
            "size_bytes": staged_frequency.stat().st_size,
            "sha256": _sha256(staged_frequency),
        }
    ]
    if single_point_suffix is not None:
        if single_point_output_file is None:
            single_point_output_file = _matching_single_point_file(
                output_file, single_point_suffix
            )
        if single_point_output_file is None:
            raise ValueError(
                "method_spec.single_point_correction_suffix requires "
                "inputs.single_point_output_file, or a matching '<frequency_stem>_"
                f"{single_point_suffix}.log/.out' file beside inputs.output_file"
            )
        single_point_extension = single_point_output_file.suffix.casefold()
        if single_point_extension not in {".log", ".out"}:
            single_point_extension = ".out"
        staged_single_point = directory / (
            f"frequency_{single_point_suffix}{single_point_extension}"
        )
        _copy_or_link(single_point_output_file, staged_single_point)
        records.append(
            {
                "role": "single_point_output",
                "source_path": relative_workspace_path(single_point_output_file),
                "staged_path": staged_single_point.name,
                "size_bytes": staged_single_point.stat().st_size,
                "sha256": _sha256(staged_single_point),
                "single_point_suffix": single_point_suffix,
            }
        )
    (directory / "staged_input_manifest.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )
    return [staged_frequency], records


def _artifact_files(directory: Path) -> list[dict[str, str]]:
    artifacts = command_artifacts(directory)
    for artifact in artifacts:
        suffix = Path(artifact["path"]).suffix.casefold()
        if suffix == ".json":
            artifact["media_type"] = "application/json"
            artifact["semantic_type"] = "GoodVibesStructuredResult"
        elif suffix in {".yaml", ".yml"}:
            artifact["media_type"] = "application/yaml"
            artifact["semantic_type"] = "GoodVibesAnalysisSpecification"
    return artifacts


def _warning_lines(*texts: str) -> list[str]:
    warnings: list[str] = []
    for text in texts:
        for line in text.splitlines():
            stripped = line.strip()
            lowered = stripped.casefold()
            if stripped and ("caution" in lowered or "warning" in lowered):
                if stripped not in warnings:
                    warnings.append(stripped[:1000])
                if len(warnings) >= 50:
                    return warnings
    return warnings


def _validation_issue_lines(text: str) -> list[str]:
    issues: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        lowered = stripped.casefold()
        if (
            not stripped
            or "point group symmetry parsed directly" in lowered
            or "implicit solvation (smd/cpcm) detected" in lowered
        ):
            continue
        if (
            "caution!" in lowered
            or "caution:" in lowered
            or lowered.startswith("! error termination")
            or "may not have terminated normally" in lowered
        ):
            if stripped not in issues:
                issues.append(stripped[:1000])
            if len(issues) >= 100:
                break
    return issues


def _single_point_consistency_issues(payload: dict[str, Any]) -> list[str]:
    """Validate SPC metadata without GoodVibes 4.3's ``--check --spc`` crash."""

    entries = [dict(item.get("qcdata") or {}) for item in payload.get("results", [])]
    if not entries:
        return ["No single-point-corrected structures were parsed."]
    issues: list[str] = []
    fields = {
        "sp_version_program": "single-point program/version",
        "sp_solvation_model": "single-point solvation model",
        "sp_charge": "single-point charge",
        "sp_multiplicity": "single-point multiplicity",
        "sp_suffix": "single-point suffix",
    }
    for field, label in fields.items():
        values = [json.dumps(entry.get(field), sort_keys=True) for entry in entries]
        if any(entry.get(field) is None for entry in entries):
            issues.append(f"At least one structure is missing its {label} metadata.")
        elif len(set(values)) != 1:
            issues.append(f"Inconsistent {label} values were detected.")
    for entry in entries:
        if entry.get("sp_energy") is None:
            issues.append("At least one structure is missing a single-point energy.")
            break
    for entry in entries:
        if (
            entry.get("charge") is not None
            and entry.get("sp_charge") is not None
            and entry["charge"] != entry["sp_charge"]
        ):
            issues.append("A frequency/single-point charge mismatch was detected.")
            break
    for entry in entries:
        if (
            entry.get("multiplicity") is not None
            and entry.get("sp_multiplicity") is not None
            and entry["multiplicity"] != entry["sp_multiplicity"]
        ):
            issues.append("A frequency/single-point multiplicity mismatch was detected.")
            break
    return issues


def _base_result(
    payload: dict[str, Any], settings: dict[str, Any], temperature_kelvin: float
) -> dict[str, Any]:
    structures = [
        _selected_thermochemistry(dict(entry), settings, temperature_kelvin)
        for entry in payload.get("results", [])
    ]
    return {
        "goodvibes_schema_version": payload.get("schema_version"),
        "goodvibes_version": payload.get("goodvibes_version"),
        "temperature_kelvin": float(temperature_kelvin),
        "standard_state": settings["standard_state"],
        "entropy_model": settings["entropy_model"],
        "enthalpy_model": settings["enthalpy_model"],
        "structures": structures,
        "goodvibes_options": _normalize_paths(payload.get("options") or {}),
    }


def _label_groups(value: Any, output_files: list[Path]) -> dict[str, list[str]]:
    if not isinstance(value, dict) or len(value) < 2:
        raise ValueError("inputs.label_groups must map at least two labels to file lists")
    allowed = set(output_files)
    groups: dict[str, list[str]] = {}
    assigned: set[Path] = set()
    for raw_label, members in value.items():
        label = str(raw_label).strip()
        if not label:
            raise ValueError("inputs.label_groups labels cannot be empty")
        if not isinstance(members, (list, tuple)) or not members:
            raise ValueError(f"inputs.label_groups[{label!r}] must be a non-empty file list")
        paths = [resolve_input_file(member).resolve() for member in members]
        unknown = [path for path in paths if path not in allowed]
        if unknown:
            raise ValueError(
                f"inputs.label_groups[{label!r}] contains files absent from output_files: "
                + ", ".join(str(path) for path in unknown)
            )
        overlap = [path for path in paths if path in assigned]
        if overlap:
            raise ValueError(
                "each selectivity file must belong to exactly one label; repeated: "
                + ", ".join(str(path) for path in overlap)
            )
        assigned.update(paths)
        groups[label] = [str(path) for path in paths]
    unassigned = sorted(allowed - assigned)
    if unassigned:
        raise ValueError(
            "every output_files member must belong to exactly one selectivity label; "
            "unassigned: " + ", ".join(str(path) for path in unassigned)
        )
    return groups


def _provenance(
    *, commands: list[list[str]], payloads: list[dict[str, Any]], settings: dict[str, Any]
) -> dict[str, Any]:
    return {
        "commands": commands,
        "goodvibes_schema_versions": sorted(
            {str(payload.get("schema_version")) for payload in payloads}
        ),
        "structured_json_parser": True,
        "selected_thermochemistry_models": {
            "entropy": settings["entropy_model"],
            "enthalpy": settings["enthalpy_model"],
        },
        "automatic_scientific_defaults": False,
        "automatic_fallback_count": 0,
    }


def execute(action_id: str, request: dict[str, Any]) -> dict[str, Any]:
    if action_id not in GOODVIBES_ACTIONS:
        raise ValueError(f"Unsupported GoodVibes Action: {action_id}")
    inputs, method, settings = request_parts(request)
    directory = output_directory(action_id, "goodvibes")
    timeout = int(request.get("resource_limits", {}).get("walltime_seconds", 1800))
    cpu_cores = request.get("resource_limits", {}).get("cpu_cores")
    jobs_arguments = ["--jobs", str(int(cpu_cores))] if cpu_cores is not None else ["--jobs", "1"]
    commands: list[list[str]] = []
    payloads: list[dict[str, Any]] = []
    captured_texts: list[str] = []

    if action_id == "derive_thermochemistry":
        output_file = resolve_input_file(inputs["output_file"]).resolve()
        if not output_file.is_file():
            raise ValueError("inputs.output_file must resolve to a regular file")
        single_point_output_file = None
        if inputs.get("single_point_output_file") is not None:
            single_point_output_file = resolve_input_file(
                inputs["single_point_output_file"]
            ).resolve()
            if not single_point_output_file.is_file():
                raise ValueError(
                    "inputs.single_point_output_file must resolve to a regular file"
                )
        suffix_value = method.get("single_point_correction_suffix")
        suffix = str(suffix_value).strip() if suffix_value is not None else None
        if suffix is not None and (
            not suffix or not _SAFE_SUFFIX.fullmatch(suffix)
        ):
            raise ValueError(
                "method_spec.single_point_correction_suffix must contain only letters, "
                "digits, dot, underscore, or hyphen"
            )
        if single_point_output_file is not None and suffix is None:
            raise ValueError(
                "inputs.single_point_output_file requires "
                "method_spec.single_point_correction_suffix"
            )
        output_files, staged_input_records = _stage_derive_inputs(
            directory=directory,
            output_file=output_file,
            single_point_output_file=single_point_output_file,
            single_point_suffix=suffix,
        )
        temperature = _positive_float(
            settings["temperature_kelvin"], field="action_settings.temperature_kelvin"
        )
        arguments = [
            *_common_arguments(settings, method, temperature_kelvin=temperature),
            *jobs_arguments,
        ]
        payload, completed = _run_once(
            directory=directory,
            output_files=output_files,
            arguments=arguments,
            output_stem="thermochemistry",
            timeout_seconds=timeout,
        )
        if payload is None:
            return unavailable(
                completed["stderr"], install="pip install 'goodvibes[full]==4.3.0'"
            )
        if len(payload["results"]) != 1:
            raise RuntimeError("derive_thermochemistry expected exactly one GoodVibes result")
        commands.append(completed["command"])
        payloads.append(payload)
        captured_texts.extend([completed["stdout"], completed["stderr"]])
        base = _base_result(payload, settings, temperature)
        result = {key: value for key, value in base.items() if key != "structures"}
        result["thermochemistry"] = base["structures"][0]
        result["staged_inputs"] = staged_input_records

    elif action_id == "scan_thermochemistry_temperature":
        output_files = _resolve_output_files(inputs["output_files"])
        temperatures = _temperature_list(inputs["temperatures_kelvin"])
        series = []
        schema_versions: set[str] = set()
        goodvibes_versions: set[str] = set()
        for index, temperature in enumerate(temperatures):
            arguments = [
                *_common_arguments(settings, method, temperature_kelvin=temperature),
                *jobs_arguments,
            ]
            payload, completed = _run_once(
                directory=directory,
                output_files=output_files,
                arguments=arguments,
                output_stem=f"temperature_{index:03d}",
                timeout_seconds=timeout,
            )
            if payload is None:
                return unavailable(
                    completed["stderr"], install="pip install 'goodvibes[full]==4.3.0'"
                )
            commands.append(completed["command"])
            payloads.append(payload)
            captured_texts.extend([completed["stdout"], completed["stderr"]])
            base = _base_result(payload, settings, temperature)
            schema_versions.add(str(base["goodvibes_schema_version"]))
            goodvibes_versions.add(str(base["goodvibes_version"]))
            series.append(
                {
                    "temperature_kelvin": temperature,
                    "structures": base["structures"],
                    "goodvibes_options": base["goodvibes_options"],
                }
            )
        result = {
            "temperatures_kelvin": temperatures,
            "standard_state": settings["standard_state"],
            "entropy_model": settings["entropy_model"],
            "enthalpy_model": settings["enthalpy_model"],
            "goodvibes_schema_versions": sorted(schema_versions),
            "goodvibes_versions": sorted(goodvibes_versions),
            "series": series,
        }

    else:
        output_files = _resolve_output_files(inputs["output_files"])
        temperature = _positive_float(
            settings["temperature_kelvin"], field="action_settings.temperature_kelvin"
        )
        arguments = _common_arguments(settings, method, temperature_kelvin=temperature)

        if action_id == "analyze_thermochemical_ensemble":
            if len(output_files) < 2:
                raise ValueError("analyze_thermochemical_ensemble requires at least two files")
            arguments.extend(_population_arguments(settings))
            arguments.extend(_deduplication_arguments(settings))
            output_stem = "ensemble"
        elif action_id == "analyze_thermochemical_selectivity":
            groups = _label_groups(inputs["label_groups"], output_files)
            specification_path = directory / "selectivity.yaml"
            specification_path.write_text(
                yaml.safe_dump({"files": groups}, sort_keys=False), encoding="utf-8"
            )
            arguments.extend(["--selectivity", str(specification_path)])
            arguments.extend(_population_arguments(settings))
            arguments.extend(_deduplication_arguments(settings))
            output_stem = "selectivity"
        elif action_id == "analyze_reaction_free_energy_profile":
            profile = resolve_input_file(inputs["profile_definition_file"]).resolve()
            if profile.suffix.casefold() not in {".yaml", ".yml"}:
                raise ValueError("profile_definition_file must be a GoodVibes YAML file")
            arguments.extend(["--pes", str(profile)])
            mode = str(settings["profile_ensemble_mode"]).strip().casefold()
            if mode == "gconf":
                pass
            elif mode == "lowest_conformer":
                arguments.append("--lowest-only")
            elif mode == "boltzmann_without_gconf":
                arguments.append("--nogconf")
            else:
                raise ValueError(
                    "profile_ensemble_mode must be gconf, lowest_conformer, or "
                    "boltzmann_without_gconf"
                )
            output_stem = "reaction_profile"
        elif action_id == "validate_thermochemistry_inputs":
            # GoodVibes 4.3.0 shadows its level_of_theory parser with a list
            # inside check_files(), so the native combination --check --spc
            # raises TypeError after parsing otherwise valid data. With SPC we
            # first parse the corrected records, then run the native check on
            # the frequency files alone and validate SPC metadata here.
            validation_with_spc = method.get("single_point_correction_suffix") is not None
            if not validation_with_spc:
                arguments.append("--check")
            arguments.extend(_validation_cutoff_arguments(settings))
            output_stem = "validation"
        else:  # pragma: no cover - guarded by GOODVIBES_ACTIONS
            raise ValueError(f"Unhandled GoodVibes Action: {action_id}")

        arguments.extend(jobs_arguments)
        payload, completed = _run_once(
            directory=directory,
            output_files=output_files,
            arguments=arguments,
            output_stem=output_stem,
            timeout_seconds=timeout,
        )
        if payload is None:
            return unavailable(
                completed["stderr"], install="pip install 'goodvibes[full]==4.3.0'"
            )
        commands.append(completed["command"])
        payloads.append(payload)
        captured_texts.extend([completed["stdout"], completed["stderr"]])
        if action_id == "validate_thermochemistry_inputs" and validation_with_spc:
            frequency_method = dict(method)
            frequency_method.pop("single_point_correction_suffix", None)
            check_arguments = [
                *_common_arguments(
                    settings, frequency_method, temperature_kelvin=temperature
                ),
                "--check",
                *_validation_cutoff_arguments(settings),
                *jobs_arguments,
            ]
            check_payload, check_completed = _run_once(
                directory=directory,
                output_files=output_files,
                arguments=check_arguments,
                output_stem="validation_frequency_files",
                timeout_seconds=timeout,
            )
            if check_payload is None:
                return unavailable(
                    check_completed["stderr"],
                    install="pip install 'goodvibes[full]==4.3.0'",
                )
            commands.append(check_completed["command"])
            payloads.append(check_payload)
            captured_texts.extend(
                [check_completed["stdout"], check_completed["stderr"]]
            )
        base = _base_result(payload, settings, temperature)

        if action_id == "analyze_thermochemical_ensemble":
            populations = [
                {
                    "file": item.get("file"),
                    "name": item.get("name"),
                    "boltzmann_factor": item.get("boltzmann_factor"),
                }
                for item in base["structures"]
                if item.get("boltzmann_factor") is not None
            ]
            if not populations:
                raise RuntimeError(
                    "GoodVibes did not produce Boltzmann populations for the selected basis"
                )
            result = {
                **base,
                "population_basis": settings["population_basis"],
                "deduplicate_structures": settings["deduplicate_structures"],
                "populations": populations,
            }
        elif action_id == "analyze_thermochemical_selectivity":
            if not payload.get("selectivity"):
                raise RuntimeError("GoodVibes did not produce the requested selectivity block")
            result = {
                **base,
                "population_basis": settings["population_basis"],
                "deduplicate_structures": settings["deduplicate_structures"],
                "selectivity": _normalize_paths(payload["selectivity"]),
                "selectivity_lowest_conformer": _normalize_paths(
                    payload.get("selectivity_lowest")
                ),
            }
        elif action_id == "analyze_reaction_free_energy_profile":
            if not payload.get("pes"):
                raise RuntimeError("GoodVibes did not produce the requested PES block")
            result = {
                **base,
                "profile_ensemble_mode": settings["profile_ensemble_mode"],
                "profile": _normalize_paths(payload["pes"]),
                "profile_definition_file": relative_workspace_path(profile),
            }
        else:
            report_text = "\n".join(captured_texts)
            issues = _validation_issue_lines(report_text)
            if validation_with_spc:
                issues.extend(_single_point_consistency_issues(payload))
            result = {
                **base,
                "consistent": not issues,
                "issues": issues,
                "goodvibes_check_report": report_text[-30000:],
                "single_point_check_workaround": bool(validation_with_spc),
            }

    warnings = _warning_lines(*captured_texts)
    return success(
        result,
        artifact_files=_artifact_files(directory),
        backend_version=module_version("goodvibes"),
        warnings=warnings,
        provenance=_provenance(commands=commands, payloads=payloads, settings=settings),
    )
