"""Deterministic conversion of spectral sticks into broadened spectra.

This backend deliberately stops at numerical spectrum construction.  Parsing a
quantum-chemistry output and rendering a figure remain separate Actions.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .common import request_parts, success, unsupported, unwrap_artifact

BACKEND_ID = "internal_spectroscopy"
BACKEND_VERSION = "1.0.0"
_EV_TO_INVERSE_CENTIMETER = 8065.544005


@dataclass(frozen=True)
class _SpectrumDefinition:
    spectrum_type: str
    axis_name: str
    default_axis_unit: str
    default_intensity_unit: str
    position_aliases: tuple[str, ...]
    intensity_aliases: tuple[str, ...]
    signed: bool = False


_DEFINITIONS = {
    "IR": _SpectrumDefinition(
        "IR",
        "wavenumber",
        "inverse_centimeter",
        "kilometer_per_mole",
        (
            "positions",
            "wavenumbers",
            "frequencies",
            "frequencies_cm1",
            "vibfreqs",
        ),
        (
            "intensities",
            "strengths",
            "ir_intensities",
            "intensities_km_mol",
            "ir_intensities_km_mol",
            "vibirs",
        ),
    ),
    "Raman": _SpectrumDefinition(
        "Raman",
        "wavenumber",
        "inverse_centimeter",
        "angstrom_fourth_per_amu",
        ("positions", "wavenumbers", "frequencies", "vibfreqs"),
        (
            "intensities",
            "strengths",
            "raman_activity",
            "raman_activities",
            "vibramans",
        ),
    ),
    "ROA": _SpectrumDefinition(
        "ROA",
        "wavenumber",
        "inverse_centimeter",
        "arbitrary_unit",
        ("positions", "wavenumbers", "frequencies", "vibfreqs"),
        (
            "intensities",
            "strengths",
            "roa_activity",
            "roa_intensities",
            "roa_activities",
            "vibroa",
        ),
        signed=True,
    ),
    "VCD": _SpectrumDefinition(
        "VCD",
        "wavenumber",
        "inverse_centimeter",
        "arbitrary_unit",
        ("positions", "wavenumbers", "frequencies", "vibfreqs"),
        (
            "intensities",
            "strengths",
            "rotatory_strength",
            "vcd_rotatory_strengths",
            "vibvcd",
            "rotatory_strengths",
        ),
        signed=True,
    ),
    "ECD": _SpectrumDefinition(
        "ECD",
        "excitation_energy",
        "inverse_centimeter",
        "arbitrary_unit",
        (
            "positions",
            "energies",
            "energy_ev",
            "excitation_energies",
            "etenergies",
        ),
        (
            "intensities",
            "strengths",
            "rotatory_strength",
            "ecd_rotatory_strengths",
            "etrotats",
            "etrotatorystrengths",
            "rotatory_strengths",
        ),
        signed=True,
    ),
    "UV-Vis": _SpectrumDefinition(
        "UV-Vis",
        "excitation_energy",
        "inverse_centimeter",
        "dimensionless_oscillator_strength",
        (
            "positions",
            "energies",
            "energy_ev",
            "excitation_energies",
            "etenergies",
        ),
        (
            "intensities",
            "strengths",
            "oscillator_strength",
            "oscillator_strengths",
            "etoscs",
        ),
    ),
    "NMR": _SpectrumDefinition(
        "NMR",
        "chemical_shift",
        "parts_per_million",
        "relative_intensity",
        ("positions", "chemical_shifts", "shifts"),
        ("intensities", "strengths", "degeneracies", "relative_intensities"),
    ),
}

_ACTION_TYPES = {
    "derive_ir_spectrum": "IR",
    "derive_raman_spectrum": "Raman",
    "derive_roa_spectrum": "ROA",
    "derive_vcd_spectrum": "VCD",
    "derive_ecd_spectrum": "ECD",
    "derive_uv_vis_spectrum": "UV-Vis",
    "derive_nmr_spectrum": "NMR",
}

_INPUT_KEYS = (
    "vibrations",
    "excited_states",
    "spectrum_data",
    "spectral_data",
    "parsed_result",
    "parsed_quantum_chemistry_result",
    "quantum_chemistry_result",
)


def _finite_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number, not a boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _number_list(value: Any, *, name: str) -> list[float]:
    if isinstance(value, Mapping) and "values" in value:
        value = value["values"]
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be an array of finite numbers")
    numbers = []
    for index, item in enumerate(value):
        if isinstance(item, Mapping):
            if "value" in item:
                item = item["value"]
            elif "real" in item:
                imaginary = _finite_number(
                    item.get("imaginary", 0.0), name=f"{name}[{index}].imaginary"
                )
                if imaginary != 0.0:
                    raise ValueError(
                        f"{name}[{index}] is imaginary; select real spectral modes "
                        "explicitly before broadening"
                    )
                item = item["real"]
            else:
                raise ValueError(
                    f"{name}[{index}] must contain value or real/imaginary components"
                )
        numbers.append(_finite_number(item, name=f"{name}[{index}]"))
    if not numbers:
        raise ValueError(f"{name} must not be empty")
    return numbers


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    item = unwrap_artifact(value)
    while (
        isinstance(item, Mapping)
        and "result" in item
        and ("status" in item or set(item) == {"result"})
    ):
        item = item["result"]
    if not isinstance(item, Mapping):
        raise ValueError(f"{name} must resolve to a JSON object")
    return dict(item)


def _source_payload(inputs: dict[str, Any]) -> dict[str, Any]:
    supplied = [key for key in _INPUT_KEYS if key in inputs]
    if len(supplied) > 1:
        raise ValueError(
            "Supply exactly one spectral source; received aliases "
            + ", ".join(supplied)
        )
    return _mapping(inputs[supplied[0]], name=supplied[0]) if supplied else dict(inputs)


def _property_maps(
    payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    properties = payload.get("properties")
    maps = [dict(properties), payload] if isinstance(properties, Mapping) else [payload]
    raw_units = payload.get("attribute_units")
    units = (
        {str(key): str(value) for key, value in raw_units.items()}
        if isinstance(raw_units, Mapping)
        else {}
    )
    return maps, units


def _comparison_value(value: Any) -> Any:
    if hasattr(value, "tolist") and callable(value.tolist):
        value = value.tolist()
    if isinstance(value, Mapping):
        return {
            str(key): _comparison_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_comparison_value(item) for item in value]
    return value


def _find_field(
    maps: list[dict[str, Any]], aliases: tuple[str, ...]
) -> tuple[Any, str, dict[str, Any]]:
    matches: list[tuple[Any, str, dict[str, Any]]] = []
    for values in maps:
        for alias in aliases:
            if alias in values and values[alias] is not None:
                matches.append((values[alias], alias, values))
    if matches:
        first_value, first_alias, first_map = matches[0]
        first_comparison = _comparison_value(first_value)
        conflicts = sorted(
            {
                alias
                for value, alias, _source in matches[1:]
                if _comparison_value(value) != first_comparison
            }
        )
        if conflicts:
            raise ValueError(
                "Conflicting spectral aliases were supplied: "
                + ", ".join([first_alias, *conflicts])
            )
        return first_value, first_alias, first_map
    raise KeyError


def _unit_for(
    raw_value: Any,
    source_field: str,
    source_map: dict[str, Any],
    attribute_units: dict[str, str],
    *,
    generic_names: tuple[str, ...],
    default: str | None,
) -> str:
    candidates: list[Any] = []
    if isinstance(raw_value, Mapping):
        candidates.append(raw_value.get("unit"))
    candidates.extend(source_map.get(name) for name in generic_names)
    candidates.append(source_map.get(f"{source_field}_unit"))
    candidates.append(attribute_units.get(source_field))
    for value in candidates:
        if value is not None:
            unit = str(value).strip()
            if not unit:
                raise ValueError(f"Unit for {source_field} must not be empty")
            return unit
    if default is None:
        raise ValueError(
            f"Unit for ambiguous spectral field {source_field!r} must be supplied "
            "explicitly with position_unit/axis_unit or intensity_unit"
        )
    return default


def _default_axis_unit(
    definition: _SpectrumDefinition, source_field: str
) -> str | None:
    normalized = source_field.strip().casefold()
    if normalized == "energy_ev" or normalized.endswith("_energy_ev"):
        return "electron_volt"
    if (
        "cm1" in normalized
        or "wavenumber" in normalized
        or normalized in {"frequency", "frequencies", "vibfreqs", "etenergies"}
    ):
        return "inverse_centimeter"
    if normalized in {"chemical_shift", "chemical_shifts", "shifts"}:
        return "parts_per_million"
    if normalized in {
        "position",
        "positions",
        "energy",
        "energies",
        "excitation_energies",
    }:
        return None
    return definition.default_axis_unit


def _default_intensity_unit(
    definition: _SpectrumDefinition, source_field: str
) -> str | None:
    normalized = source_field.strip().casefold()
    if normalized in {"intensity", "intensities", "strength", "strengths"}:
        return None
    return definition.default_intensity_unit


def _labels(payload: dict[str, Any], count: int) -> list[str | None]:
    values = payload.get("labels")
    if values is None:
        return [None] * count
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("labels must be an array")
    if len(values) != count:
        raise ValueError("labels must have the same length as spectral positions")
    return [None if value is None else str(value) for value in values]


def _line_records(
    payload: dict[str, Any], definition: _SpectrumDefinition
) -> tuple[list[float], list[float], list[str | None], str, str, dict[str, str]] | None:
    records: Any = None
    record_field = ""
    for candidate in (
        "spectral_lines",
        "lines",
        "transitions",
        "modes",
        "states",
        "resonances",
    ):
        if candidate in payload:
            records = payload[candidate]
            record_field = candidate
            break
    if records is None:
        return None
    if (
        isinstance(records, (str, bytes))
        or not isinstance(records, Sequence)
        or not records
    ):
        raise ValueError(f"{record_field} must be a non-empty array of objects")

    positions: list[float] = []
    intensities: list[float] = []
    labels: list[str | None] = []
    position_fields: set[str] = set()
    intensity_fields: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"{record_field}[{index}] must be an object")
        try:
            raw_position, position_field, _ = _find_field(
                [dict(record)],
                (
                    "position",
                    "frequency",
                    "wavenumber",
                    "energy",
                    *definition.position_aliases,
                ),
            )
        except KeyError as exc:
            raise ValueError(
                f"{record_field}[{index}] is missing a spectral position"
            ) from exc
        try:
            raw_intensity, intensity_field, _ = _find_field(
                [dict(record)], ("intensity", "strength", *definition.intensity_aliases)
            )
        except KeyError as exc:
            if definition.spectrum_type == "ECD" and any(
                alias in record
                for alias in (
                    "etoscs",
                    "oscillator_strength",
                    "oscillator_strengths",
                )
            ):
                raise ValueError(
                    "ECD requires signed rotatory strengths; oscillator strengths "
                    "describe UV/Vis absorption and are not a valid substitute"
                ) from exc
            raise ValueError(
                f"{record_field}[{index}] is missing a spectral intensity"
            ) from exc
        positions.append(
            _finite_number(raw_position, name=f"{record_field}[{index}].position")
        )
        intensities.append(
            _finite_number(raw_intensity, name=f"{record_field}[{index}].intensity")
        )
        labels.append(None if record.get("label") is None else str(record["label"]))
        position_fields.add(position_field)
        intensity_fields.add(intensity_field)

    if len(position_fields) != 1 or len(intensity_fields) != 1:
        raise ValueError(
            f"{record_field} must use one consistent position field and one "
            "consistent intensity field"
        )
    position_field = next(iter(position_fields))
    intensity_field = next(iter(intensity_fields))

    axis_unit = _unit_for(
        None,
        record_field,
        payload,
        {},
        generic_names=("position_unit", "axis_unit"),
        default=_default_axis_unit(definition, position_field),
    )
    intensity_unit = _unit_for(
        None,
        record_field,
        payload,
        {},
        generic_names=("intensity_unit",),
        default=_default_intensity_unit(definition, intensity_field),
    )
    return (
        positions,
        intensities,
        labels,
        axis_unit,
        intensity_unit,
        {"positions": position_field, "intensities": intensity_field},
    )


def _isotropic_shieldings(
    value: Any,
) -> tuple[list[float], list[str | None], list[int]]:
    if isinstance(value, Mapping):

        def sort_key(item: tuple[Any, Any]) -> tuple[int, int | str]:
            text = str(item[0])
            try:
                return 0, int(text)
            except ValueError:
                return 1, text

        items = sorted(value.items(), key=sort_key)
        tensors = [item[1] for item in items]
        labels = [str(item[0]) for item in items]
        try:
            atom_indices = [int(item[0]) for item in items]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Atom-indexed nmrtensors keys must be integer atom indices"
            ) from exc
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        tensors = list(value)
        labels = [str(index) for index in range(len(tensors))]
        atom_indices = list(range(len(tensors)))
    else:
        raise ValueError("nmrtensors must be an array or atom-indexed object")
    if not tensors:
        raise ValueError("nmrtensors must not be empty")

    shieldings: list[float] = []
    for index, tensor in enumerate(tensors):
        if isinstance(tensor, Mapping):
            if tensor.get("isotropic") is not None:
                shieldings.append(
                    _finite_number(
                        tensor["isotropic"],
                        name=f"nmrtensors[{atom_indices[index]}].isotropic",
                    )
                )
                continue
            if tensor.get("total") is not None:
                tensor = tensor["total"]
            else:
                raise ValueError(
                    f"nmrtensors[{atom_indices[index]}] requires an isotropic value "
                    "or total 3x3 shielding tensor"
                )
        if (
            isinstance(tensor, Sequence)
            and not isinstance(tensor, (str, bytes))
            and len(tensor) == 3
            and all(
                isinstance(row, Sequence)
                and not isinstance(row, (str, bytes))
                and len(row) == 3
                for row in tensor
            )
        ):
            diagonal = [tensor[i][i] for i in range(3)]
        elif (
            isinstance(tensor, Sequence)
            and not isinstance(tensor, (str, bytes))
            and len(tensor) == 9
        ):
            diagonal = [tensor[0], tensor[4], tensor[8]]
        else:
            raise ValueError(
                f"nmrtensors[{atom_indices[index]}] must be a cclib tensor mapping, "
                "3x3 tensor, or flat nine-value tensor"
            )
        shieldings.append(
            sum(
                _finite_number(
                    item,
                    name=f"nmrtensors[{atom_indices[index]}].diagonal",
                )
                for item in diagonal
            )
            / 3.0
        )
    return shieldings, labels, atom_indices


def _setting(
    name: str,
    method: dict[str, Any],
    settings: dict[str, Any],
    *,
    default: Any = None,
    required: bool = False,
) -> Any:
    if name in method and name in settings and method[name] != settings[name]:
        raise ValueError(
            f"Conflicting {name} values in method_spec and action_settings"
        )
    if name in settings:
        return settings[name]
    if name in method:
        return method[name]
    if required:
        raise ValueError(f"{name} must be supplied explicitly")
    return default


def _energy_ev_to_axis(value: Any, axis_unit: str, *, name: str) -> float:
    number = _finite_number(value, name=name)
    normalized = axis_unit.strip().casefold().replace(" ", "_")
    if normalized in {"ev", "electron_volt", "electronvolt"}:
        return number
    if normalized in {
        "inverse_centimeter",
        "cm^-1",
        "cm-1",
        "cm**-1",
        "1/cm",
    }:
        return number * _EV_TO_INVERSE_CENTIMETER
    raise ValueError(
        f"Cannot convert {name} from electron_volt to unsupported axis unit "
        f"{axis_unit!r}; supply fwhm/grid bounds directly in the axis unit"
    )


def _cm1_to_axis(value: Any, axis_unit: str, *, name: str) -> float:
    number = _finite_number(value, name=name)
    normalized = axis_unit.strip().casefold().replace(" ", "_")
    if normalized in {
        "inverse_centimeter",
        "cm^-1",
        "cm-1",
        "cm**-1",
        "1/cm",
    }:
        return number
    raise ValueError(
        f"Cannot use {name} expressed in inverse_centimeter with axis unit "
        f"{axis_unit!r}; supply fwhm/grid bounds directly in the axis unit"
    )


def _canonical_settings(
    action_id: str, axis_unit: str, settings: dict[str, Any]
) -> dict[str, Any]:
    normalized = dict(settings)

    def alias(canonical: str, source: str, transform=lambda value: value) -> None:
        if source not in settings:
            return
        value = transform(settings[source])
        if canonical in normalized and normalized[canonical] != value:
            raise ValueError(f"Conflicting {canonical} and {source} values")
        normalized[canonical] = value

    if action_id in {"derive_ir_spectrum", "derive_uv_vis_spectrum"}:
        alias("line_shape", "broadening", lambda value: str(value).strip().lower())
    if action_id == "derive_ir_spectrum":
        alias(
            "fwhm",
            "fwhm_cm1",
            lambda value: _cm1_to_axis(value, axis_unit, name="fwhm_cm1"),
        )
        alias(
            "grid_min",
            "min_wavenumber_cm1",
            lambda value: _cm1_to_axis(value, axis_unit, name="min_wavenumber_cm1"),
        )
        alias(
            "grid_max",
            "max_wavenumber_cm1",
            lambda value: _cm1_to_axis(value, axis_unit, name="max_wavenumber_cm1"),
        )
        alias("grid_points", "points")
    elif action_id == "derive_uv_vis_spectrum":
        alias(
            "fwhm",
            "fwhm_ev",
            lambda value: _energy_ev_to_axis(value, axis_unit, name="fwhm_ev"),
        )
        alias(
            "grid_min",
            "minimum_energy_ev",
            lambda value: _energy_ev_to_axis(
                value, axis_unit, name="minimum_energy_ev"
            ),
        )
        alias(
            "grid_max",
            "maximum_energy_ev",
            lambda value: _energy_ev_to_axis(
                value, axis_unit, name="maximum_energy_ev"
            ),
        )
    return normalized


def _nmr_sticks(
    payload: dict[str, Any],
    method: dict[str, Any],
    settings: dict[str, Any],
    definition: _SpectrumDefinition,
) -> tuple[
    list[float], list[float], list[str | None], str, str, dict[str, str], list[str]
]:
    maps, units = _property_maps(payload)
    warnings: list[str] = []
    labels: list[str | None]
    source_field: str
    source_map: dict[str, Any]
    raw_positions: Any
    tensor_atom_indices: list[int] | None = None
    try:
        raw_positions, source_field, source_map = _find_field(
            maps, definition.position_aliases
        )
        positions = _number_list(raw_positions, name=source_field)
        labels = _labels(payload, len(positions))
    except KeyError:
        try:
            raw_shieldings, source_field, source_map = _find_field(
                maps, ("isotropic_shieldings", "nmr_shieldings", "shieldings")
            )
            shieldings = _number_list(raw_shieldings, name=source_field)
            labels = _labels(payload, len(shieldings))
        except KeyError:
            try:
                raw_tensors, source_field, source_map = _find_field(
                    maps, ("nmrtensors",)
                )
            except KeyError as exc:
                raise ValueError(
                    "NMR input requires chemical_shifts, isotropic shieldings, "
                    "or nmrtensors"
                ) from exc
            shieldings, labels, tensor_atom_indices = _isotropic_shieldings(
                raw_tensors
            )
        reference = _finite_number(
            _setting("reference_shielding", method, settings, required=True),
            name="reference_shielding",
        )
        positions = [reference - value for value in shieldings]

    try:
        raw_intensities, intensity_field, intensity_map = _find_field(
            maps, definition.intensity_aliases
        )
        intensities = _number_list(raw_intensities, name=intensity_field)
    except KeyError:
        intensities = [1.0] * len(positions)
        intensity_field = "unit_weights"
        intensity_map = payload
        warnings.append(
            "No NMR degeneracies/intensities were supplied; each resonance was "
            "assigned unit weight."
        )
    if len(positions) != len(intensities):
        raise ValueError("NMR positions and intensities must have the same length")

    raw_target = _setting(
        "nucleus_atomic_number", method, settings, required=True
    )
    if isinstance(raw_target, bool):
        raise ValueError("nucleus_atomic_number must be a positive integer")
    try:
        target_atomic_number = int(raw_target)
    except (TypeError, ValueError) as exc:
        raise ValueError("nucleus_atomic_number must be a positive integer") from exc
    if (
        target_atomic_number < 1
        or isinstance(raw_target, float)
        and not raw_target.is_integer()
    ):
        raise ValueError("nucleus_atomic_number must be a positive integer")
    try:
        raw_atom_numbers, atom_number_field, _atom_number_map = _find_field(
            maps,
            ("atomnos", "atomic_numbers", "nucleus_atomic_numbers"),
        )
    except KeyError as exc:
        raise ValueError(
            "NMR input requires atomnos/atomic_numbers so the selected nucleus can "
            "be filtered without mixing chemical-shift scales"
        ) from exc
    atom_numbers = _number_list(raw_atom_numbers, name=atom_number_field)
    normalized_atom_numbers: list[int] = []
    for index, value in enumerate(atom_numbers):
        converted = int(value)
        if converted < 1 or converted != value:
            raise ValueError(
                f"{atom_number_field}[{index}] must be a positive atomic number"
            )
        normalized_atom_numbers.append(converted)
    if tensor_atom_indices is not None:
        try:
            line_atom_numbers = [
                normalized_atom_numbers[index] for index in tensor_atom_indices
            ]
        except IndexError as exc:
            raise ValueError(
                "nmrtensors contains an atom index outside the atomnos array"
            ) from exc
    else:
        if len(normalized_atom_numbers) != len(positions):
            raise ValueError(
                f"{atom_number_field} must align with the supplied NMR lines"
            )
        line_atom_numbers = normalized_atom_numbers
    selected_indices = [
        index
        for index, atomic_number in enumerate(line_atom_numbers)
        if atomic_number == target_atomic_number
    ]
    if not selected_indices:
        raise ValueError(
            f"No NMR lines match nucleus_atomic_number={target_atomic_number}"
        )
    positions = [positions[index] for index in selected_indices]
    intensities = [intensities[index] for index in selected_indices]
    labels = [labels[index] for index in selected_indices]

    axis_unit = _unit_for(
        raw_positions if "raw_positions" in locals() else None,
        source_field,
        source_map,
        units,
        generic_names=("position_unit", "axis_unit"),
        default=_default_axis_unit(definition, source_field),
    )
    intensity_unit = _unit_for(
        raw_intensities if "raw_intensities" in locals() else None,
        intensity_field,
        intensity_map,
        units,
        generic_names=("intensity_unit",),
        default=_default_intensity_unit(definition, intensity_field),
    )
    return (
        positions,
        intensities,
        labels,
        axis_unit,
        intensity_unit,
        {"positions": source_field, "intensities": intensity_field},
        warnings,
    )


def _ordinary_sticks(
    payload: dict[str, Any], definition: _SpectrumDefinition
) -> tuple[list[float], list[float], list[str | None], str, str, dict[str, str]]:
    records = _line_records(payload, definition)
    if records is not None:
        return records
    maps, units = _property_maps(payload)
    try:
        raw_positions, position_field, position_map = _find_field(
            maps, definition.position_aliases
        )
    except KeyError as exc:
        raise ValueError(
            f"{definition.spectrum_type} input is missing positions; accepted fields: "
            + ", ".join(definition.position_aliases)
        ) from exc
    try:
        raw_intensities, intensity_field, intensity_map = _find_field(
            maps, definition.intensity_aliases
        )
    except KeyError as exc:
        if definition.spectrum_type == "ECD" and any(
            any(
                alias in values
                for alias in ("etoscs", "oscillator_strength", "oscillator_strengths")
            )
            for values in maps
        ):
            raise ValueError(
                "ECD requires signed rotatory strengths; oscillator strengths describe "
                "UV/Vis absorption and are not a valid substitute"
            ) from exc
        raise ValueError(
            f"{definition.spectrum_type} input is missing intensities; accepted "
            "fields: " + ", ".join(definition.intensity_aliases)
        ) from exc

    positions = _number_list(raw_positions, name=position_field)
    intensities = _number_list(raw_intensities, name=intensity_field)
    if len(positions) != len(intensities):
        raise ValueError(
            f"{position_field} and {intensity_field} must have the same length"
        )
    labels = _labels(payload, len(positions))
    axis_unit = _unit_for(
        raw_positions,
        position_field,
        position_map,
        units,
        generic_names=("position_unit", "axis_unit"),
        default=_default_axis_unit(definition, position_field),
    )
    intensity_unit = _unit_for(
        raw_intensities,
        intensity_field,
        intensity_map,
        units,
        generic_names=("intensity_unit",),
        default=_default_intensity_unit(definition, intensity_field),
    )
    return (
        positions,
        intensities,
        labels,
        axis_unit,
        intensity_unit,
        {"positions": position_field, "intensities": intensity_field},
    )


def _grid(
    positions: list[float],
    fwhm: float,
    method: dict[str, Any],
    settings: dict[str, Any],
) -> list[float]:
    lower_value = _setting("grid_min", method, settings)
    upper_value = _setting("grid_max", method, settings)
    lower = (
        min(positions) - 5.0 * fwhm
        if lower_value is None
        else _finite_number(lower_value, name="grid_min")
    )
    upper = (
        max(positions) + 5.0 * fwhm
        if upper_value is None
        else _finite_number(upper_value, name="grid_max")
    )
    if lower >= upper:
        raise ValueError("grid_min must be smaller than grid_max")
    raw_count = _setting("grid_points", method, settings, default=2001)
    if isinstance(raw_count, bool):
        raise ValueError("grid_points must be an integer")
    try:
        count = int(raw_count)
    except (TypeError, ValueError) as exc:
        raise ValueError("grid_points must be an integer") from exc
    if count != raw_count or count < 2 or count > 200000:
        raise ValueError("grid_points must be an integer between 2 and 200000")
    step = (upper - lower) / (count - 1)
    return [lower + index * step for index in range(count)]


def _broaden(
    grid: list[float],
    positions: list[float],
    intensities: list[float],
    *,
    fwhm: float,
    line_shape: str,
) -> list[float]:
    if line_shape == "gaussian":
        coefficient = -4.0 * math.log(2.0) / (fwhm * fwhm)
        return [
            sum(
                intensity * math.exp(coefficient * (point - position) ** 2)
                for position, intensity in zip(positions, intensities, strict=True)
            )
            for point in grid
        ]
    if line_shape == "lorentzian":
        coefficient = 4.0 / (fwhm * fwhm)
        return [
            sum(
                intensity / (1.0 + coefficient * (point - position) ** 2)
                for position, intensity in zip(positions, intensities, strict=True)
            )
            for point in grid
        ]
    raise ValueError("line_shape must be gaussian or lorentzian")


def _derive(action_id: str, request: dict[str, Any]) -> dict[str, Any]:
    inputs, method, settings = request_parts(request)
    spectrum_type = _ACTION_TYPES[action_id]
    definition = _DEFINITIONS[spectrum_type]
    payload = _source_payload(inputs)

    warnings: list[str] = []
    if spectrum_type == "NMR":
        (
            positions,
            intensities,
            labels,
            axis_unit,
            intensity_unit,
            source_fields,
            warnings,
        ) = _nmr_sticks(payload, method, settings, definition)
    else:
        (
            positions,
            intensities,
            labels,
            axis_unit,
            intensity_unit,
            source_fields,
        ) = _ordinary_sticks(payload, definition)

    settings = _canonical_settings(action_id, axis_unit, settings)
    if (
        action_id == "derive_ir_spectrum"
        and "grid_points" not in settings
        and "grid_points" not in method
    ):
        # Preserve the original MiniChem IR default while returning the richer v1 schema.
        settings["grid_points"] = 4000
    scale_factor = _finite_number(
        _setting("position_scale_factor", method, settings, default=1.0),
        name="position_scale_factor",
    )
    if scale_factor <= 0.0:
        raise ValueError("position_scale_factor must be greater than zero")
    positions = [value * scale_factor for value in positions]
    fwhm = _finite_number(
        _setting("fwhm", method, settings, required=True), name="fwhm"
    )
    if fwhm <= 0.0:
        raise ValueError("fwhm must be greater than zero")
    if (
        action_id == "derive_ir_spectrum"
        and "grid_min" not in settings
        and "grid_min" not in method
    ):
        settings["grid_min"] = max(0.0, min(positions) - 5.0 * fwhm)
    line_shape = str(
        _setting("line_shape", method, settings, default="gaussian")
    ).lower()
    grid = _grid(positions, fwhm, method, settings)
    if len(grid) * len(positions) > 20_000_000:
        raise ValueError(
            "Requested spectrum exceeds 20000000 stick-grid evaluations; reduce "
            "grid_points or split the explicitly selected transitions"
        )
    curve = _broaden(
        grid,
        positions,
        intensities,
        fwhm=fwhm,
        line_shape=line_shape,
    )

    sticks = []
    for position, intensity, label in zip(positions, intensities, labels, strict=True):
        stick = {"position": position, "intensity": intensity}
        if label is not None:
            stick["label"] = label
        sticks.append(stick)
    signed = definition.signed or any(value < 0.0 for value in intensities)
    result = {
        "schema_name": "SpectrumResult",
        "schema_version": "1.0",
        "spectrum_type": spectrum_type,
        "axis": {
            "name": definition.axis_name,
            "unit": axis_unit,
            "values": grid,
        },
        "intensity": {
            "name": "spectral_intensity",
            "unit": intensity_unit,
            "values": curve,
            "signed": signed,
        },
        "sticks": sticks,
        "line_shape": {
            "kind": line_shape,
            "fwhm": fwhm,
            "unit": axis_unit,
            "normalization": "peak_height",
        },
        "point_count": len(grid),
        "stick_count": len(sticks),
        "position_scale_factor": scale_factor,
        "source_fields": source_fields,
    }
    if spectrum_type == "NMR":
        result["nucleus_atomic_number"] = int(
            _setting(
                "nucleus_atomic_number", method, settings, required=True
            )
        )
    if spectrum_type == "IR":
        result.update(
            {
                "wavenumber_cm1": grid,
                "intensity_values": curve,
                "line_frequencies_cm1": positions,
                "line_intensities": intensities,
                "broadening": line_shape,
                "fwhm_cm1": fwhm,
            }
        )
    return success(
        result,
        backend_version=BACKEND_VERSION,
        warnings=warnings,
        provenance={
            "implementation": "minichem_toolbox.backends.spectroscopy",
            "algorithm": f"deterministic_{line_shape}_stick_broadening",
        },
    )


def execute(action_id: str, backend_id: str, request: dict[str, Any]) -> dict[str, Any]:
    if backend_id != BACKEND_ID:
        return unsupported(f"Unsupported spectroscopy backend: {backend_id}")
    if action_id not in _ACTION_TYPES:
        return unsupported(f"Unsupported spectroscopy action: {action_id}")
    return _derive(action_id, request)
