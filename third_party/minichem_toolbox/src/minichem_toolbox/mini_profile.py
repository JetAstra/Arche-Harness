"""Focused Action/backend profile for ARCHE molecular chemistry tasks."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from .models import ActionSpec, BackendSpec


MINI_ACTION_IDS = frozenset(
    {
        "normalize_qcschema_molecule",
        "validate_qcschema_record",
        "parse_quantum_chemistry_output",
        "extract_final_structure",
        "render_spectrum_plot",
        "smiles2sdf",
        "sdf_to_gjf",
        "xyz_to_gjf",
        "standardize_structure",
        "generate_3d_structure",
        "generate_conformer_ensemble",
        "enumerate_conformer_combinations",
        "build_reactive_complex_ensemble",
        "resolve_mapped_reactive_contacts",
        "build_mapped_distance_constraints",
        "select_reactive_complex_candidate",
        "cluster_conformers",
        "align_molecular_structures",
        "rank_conformers_from_results",
        "enumerate_coordination_isomers",
        "assign_protonation_states",
        "assign_partial_charges",
        "search_local_substructures",
        "enumerate_tautomers",
        "enumerate_stereoisomers",
        "calculate_energy",
        "generate_gaussian_code",
        "calculate_forces",
        "calculate_hessian",
        "optimize_geometry",
        "calculate_dipole_moment",
        "calculate_atomic_charges",
        "calculate_electron_isodensity_surface",
        "analyze_non_covalent_interactions",
        "calculate_bond_orders",
        "calculate_excited_states",
        "derive_vibrational_modes",
        "derive_ir_spectrum",
        "derive_uv_vis_spectrum",
        "derive_raman_spectrum",
        "derive_roa_spectrum",
        "derive_vcd_spectrum",
        "derive_ecd_spectrum",
        "derive_nmr_spectrum",
        "derive_thermochemistry",
        "scan_thermochemistry_temperature",
        "analyze_thermochemical_ensemble",
        "validate_thermochemistry_inputs",
        "locate_transition_state",
        "map_reaction_atoms",
        "materialize_mapped_hydrogen_transfers",
        "analyze_reaction_changes",
        "apply_reaction_changes_to_structure",
        "search_reaction_path",
        "search_transition_state_path",
        "scan_reaction_coordinates",
        "validate_reaction_path",
        "analyze_reaction_coordinate",
        "extract_reaction_path_image",
        "trace_intrinsic_reaction_coordinate",
        "parse_gaussian_irc_output",
        "analyze_thermochemical_selectivity",
        "analyze_free_energy_selectivity",
        "analyze_energetic_span",
        "analyze_reaction_free_energy_profile",
    }
)

MINI_BACKEND_IDS = frozenset(
    {
        "qcelemental",
        "format_conversion",
        "cclib",
        "internal_plotting",
        "rdkit",
        "openbabel",
        "rdkit_etkdg",
        "rdkit_mmff",
        "crest",
        "internal_statistics",
        "internal_reaction_analysis",
        "internal_reactive_complex",
        "rxnmapper",
        "rdkit_gasteiger",
        "xtb",
        "multiwfn",
        "gaussian",
        "gaussian_codegen",
        "internal_vibrations",
        "internal_spectroscopy",
        "fairchem_uma",
        "geometric",
        "internal_thermochemistry",
        "goodvibes",
        "sella",
        "pysisyphus",
        "ash_neb_uma",
        "transition_state_path_search",
    }
)

MINI_NATIVE_SOFTWARE_IDS = frozenset(
    {
        "gaussian",
        "crest",
        "xtb",
        "openbabel",
        "pysisyphus",
        "goodvibes",
        "multiwfn",
    }
)

MINI_RUNTIME_IDS = frozenset(
    {
        "core",
        "quantum",
        "reaction",
        "gaussian",
        "goodvibes",
        "multiwfn",
        "workflows",
        "sella",
        "rxnmapper",
        "fairchem",
        "ash_neb",
    }
)


def _for_actions(mapping: Mapping[str, Any], action_ids: set[str]) -> dict[str, Any]:
    return {key: value for key, value in mapping.items() if key in action_ids}


def _component_options(
    mapping: Mapping[str, Mapping[str, tuple[str, ...]]],
    action_ids: set[str],
    backend_ids: set[str],
) -> dict[str, dict[str, tuple[str, ...]]]:
    result: dict[str, dict[str, tuple[str, ...]]] = {}
    for action_id, roles in mapping.items():
        if action_id not in action_ids:
            continue
        filtered = {
            role: tuple(option for option in options if option in backend_ids)
            for role, options in roles.items()
        }
        result[action_id] = {role: options for role, options in filtered.items() if options}
    return result


def _allowed_methods(
    mapping: Mapping[str, Mapping[str, tuple[str, ...]]],
    action_ids: set[str],
    backend_ids: set[str],
) -> dict[str, dict[str, tuple[str, ...]]]:
    result: dict[str, dict[str, tuple[str, ...]]] = {}
    for action_id, fields in mapping.items():
        if action_id not in action_ids:
            continue
        values = dict(fields)
        if "calculator_backend" in values:
            values["calculator_backend"] = tuple(
                item for item in values["calculator_backend"] if item in backend_ids
            )
        result[action_id] = values
    return result


def build_mini_specs(
    full_actions: tuple[ActionSpec, ...],
    full_backends: tuple[BackendSpec, ...],
) -> tuple[tuple[ActionSpec, ...], tuple[BackendSpec, ...]]:
    """Return an internally consistent subset of the comprehensive toolbox."""

    backend_ids = {item.id for item in full_backends if item.id in MINI_BACKEND_IDS}
    actions = tuple(
        replace(
            item,
            backend_ids=tuple(value for value in item.backend_ids if value in backend_ids),
        )
        for item in full_actions
        if item.id in MINI_ACTION_IDS
        and any(value in backend_ids for value in item.backend_ids)
    )
    action_ids = {item.id for item in actions}

    backends: list[BackendSpec] = []
    for item in full_backends:
        if item.id not in backend_ids:
            continue
        capabilities = tuple(value for value in item.capabilities if value in action_ids)
        if not capabilities:
            continue
        capability_ids = set(capabilities)
        backends.append(
            replace(
                item,
                capabilities=capabilities,
                required_input_fields=_for_actions(item.required_input_fields, capability_ids),
                required_method_fields=_for_actions(item.required_method_fields, capability_ids),
                required_setting_fields=_for_actions(item.required_setting_fields, capability_ids),
                allowed_method_values=_allowed_methods(
                    item.allowed_method_values, capability_ids, backend_ids
                ),
                allowed_setting_values=_for_actions(
                    item.allowed_setting_values, capability_ids
                ),
                required_component_roles=_for_actions(
                    item.required_component_roles, capability_ids
                ),
                component_backend_options=_component_options(
                    item.component_backend_options, capability_ids, backend_ids
                ),
                supported_system_types=_for_actions(
                    item.supported_system_types, capability_ids
                ),
                validation_levels=_for_actions(item.validation_levels, capability_ids),
                parameter_specs=_for_actions(item.parameter_specs, capability_ids),
                fixed_parameter_specs=_for_actions(
                    item.fixed_parameter_specs, capability_ids
                ),
            )
        )

    final_backend_ids = {item.id for item in backends}
    actions = tuple(
        replace(
            item,
            backend_ids=tuple(value for value in item.backend_ids if value in final_backend_ids),
        )
        for item in actions
    )
    return actions, tuple(backends)


__all__ = [
    "MINI_ACTION_IDS",
    "MINI_BACKEND_IDS",
    "MINI_NATIVE_SOFTWARE_IDS",
    "MINI_RUNTIME_IDS",
    "build_mini_specs",
]
