"""Deterministic rigid-body assembly of explicit reactive-complex components.

This backend deliberately does not generate conformers, edit molecular graphs, or
optimize geometries.  It only places supplied coordinate-bearing components using
explicit inter-component contact constraints and reports geometric diagnostics.
"""

from __future__ import annotations

import copy
import itertools
import math
import random
from typing import Any

from .common import (
    atoms_and_coordinates,
    failed,
    partial_success,
    request_parts,
    structure_dict,
    success,
    unsupported,
    unwrap_artifact,
)


ACTION_ID = "build_reactive_complex_ensemble"
SELECT_ACTION_ID = "select_reactive_complex_candidate"
ENUMERATE_ACTION_ID = "enumerate_conformer_combinations"
RESOLVE_CONTACTS_ACTION_ID = "resolve_mapped_reactive_contacts"
BUILD_CONSTRAINTS_ACTION_ID = "build_mapped_distance_constraints"
BACKEND_ID = "internal_reactive_complex"

HARD_MAX_CANDIDATES = 128
HARD_MAX_COMPONENTS = 16
HARD_MAX_CONTACTS = 128
HARD_MAX_ORIENTATION_TRIALS = 256
HARD_MAX_PAIRWISE_EVALUATIONS = 20_000_000
HARD_MAX_REPORTED_COLLISIONS = 256
HARD_MAX_TOTAL_ATOMS = 4_096
HARD_MAX_CANDIDATE_OUTPUT_ATOM_RECORDS = 262_144
HARD_MAX_CONFORMER_COMBINATIONS = 4_096

_ARCHE_SINGLE_BOND_LENGTHS_ANGSTROM = {
    ("C", "C"): 1.54,
    ("C", "H"): 1.09,
    ("C", "N"): 1.47,
    ("C", "O"): 1.43,
    ("H", "H"): 0.74,
    ("H", "N"): 1.01,
    ("H", "O"): 0.96,
    ("N", "N"): 1.45,
    ("N", "O"): 1.41,
    ("O", "O"): 1.48,
}


def _distance(first: list[float], second: list[float]) -> float:
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(first, second)))


def _random_unit_vector(generator: random.Random) -> list[float]:
    z = generator.uniform(-1.0, 1.0)
    angle = generator.uniform(0.0, 2.0 * math.pi)
    radius = math.sqrt(max(0.0, 1.0 - z * z))
    return [radius * math.cos(angle), radius * math.sin(angle), z]


def _random_rotation(generator: random.Random) -> list[list[float]]:
    """Return a uniformly sampled 3D rotation matrix from a unit quaternion."""

    first = generator.random()
    second = generator.random()
    third = generator.random()
    qx = math.sqrt(1.0 - first) * math.sin(2.0 * math.pi * second)
    qy = math.sqrt(1.0 - first) * math.cos(2.0 * math.pi * second)
    qz = math.sqrt(first) * math.sin(2.0 * math.pi * third)
    qw = math.sqrt(first) * math.cos(2.0 * math.pi * third)
    return [
        [
            1.0 - 2.0 * (qy * qy + qz * qz),
            2.0 * (qx * qy - qz * qw),
            2.0 * (qx * qz + qy * qw),
        ],
        [
            2.0 * (qx * qy + qz * qw),
            1.0 - 2.0 * (qx * qx + qz * qz),
            2.0 * (qy * qz - qx * qw),
        ],
        [
            2.0 * (qx * qz - qy * qw),
            2.0 * (qy * qz + qx * qw),
            1.0 - 2.0 * (qx * qx + qy * qy),
        ],
    ]


def _author_euler_rotation(generator: random.Random) -> list[list[float]]:
    """按作者后续尝试的三个独立 Euler 角生成旋转矩阵。"""

    angles = [math.radians(generator.uniform(0.0, 360.0)) for _ in range(3)]
    rx, ry, rz = angles
    sin_x, cos_x = math.sin(rx), math.cos(rx)
    sin_y, cos_y = math.sin(ry), math.cos(ry)
    sin_z, cos_z = math.sin(rz), math.cos(rz)
    rx_matrix = [[1.0, 0.0, 0.0], [0.0, cos_x, -sin_x], [0.0, sin_x, cos_x]]
    ry_matrix = [[cos_y, 0.0, sin_y], [0.0, 1.0, 0.0], [-sin_y, 0.0, cos_y]]
    rz_matrix = [[cos_z, -sin_z, 0.0], [sin_z, cos_z, 0.0], [0.0, 0.0, 1.0]]

    def multiply(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
        return [
            [sum(left[row][index] * right[index][column] for index in range(3)) for column in range(3)]
            for row in range(3)
        ]

    return multiply(multiply(rz_matrix, ry_matrix), rx_matrix)


def _matrix_vector(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [sum(row[index] * vector[index] for index in range(3)) for row in matrix]


def _rigid_transform(
    coordinates: list[list[float]],
    rotation: list[list[float]],
    moving_contact: list[float],
    target_contact: list[float],
) -> tuple[list[list[float]], list[float]]:
    rotated_contact = _matrix_vector(rotation, moving_contact)
    translation = [
        target_contact[index] - rotated_contact[index] for index in range(3)
    ]
    transformed = []
    for coordinate in coordinates:
        rotated = _matrix_vector(rotation, coordinate)
        transformed.append(
            [rotated[index] + translation[index] for index in range(3)]
        )
    return transformed, translation


def _integer(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        converted = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{label} must be an integer")
    if minimum is not None and converted < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return converted


def _positive_float(value: Any, label: str, *, allow_zero: bool = False) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(converted) or (converted < 0.0 if allow_zero else converted <= 0.0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{label} must be a finite {qualifier} number")
    return converted


def _atom_map_number(atom: dict[str, Any], label: str) -> int | None:
    values = []
    for key in ("atom_map_number", "atom_map", "map_number"):
        if key in atom and atom[key] is not None:
            values.append(_integer(atom[key], f"{label}.{key}", minimum=1))
    if not values:
        return None
    if len(set(values)) != 1:
        raise ValueError(f"{label} contains conflicting atom-map fields")
    return values[0]


def _bond_index_keys(bond: dict[str, Any]) -> tuple[str, str] | None:
    for keys in (
        ("atom_index_a", "atom_index_b"),
        ("atom_a", "atom_b"),
        ("begin_atom_index", "end_atom_index"),
        ("i", "j"),
    ):
        if keys[0] in bond and keys[1] in bond:
            return keys
    return None


def _validate_bonds(
    component: dict[str, Any], component_index: int, atom_count: int
) -> None:
    bonds = component.get("bonds")
    if bonds is None:
        return
    if not isinstance(bonds, list):
        raise ValueError(f"components[{component_index}].bonds must be an array")
    for bond_index, bond in enumerate(bonds):
        label = f"components[{component_index}].bonds[{bond_index}]"
        if isinstance(bond, dict):
            keys = _bond_index_keys(bond)
            if keys is None:
                raise ValueError(
                    f"{label} must identify endpoints with atom_index_a/atom_index_b"
                )
            endpoints = [bond[keys[0]], bond[keys[1]]]
        elif isinstance(bond, (list, tuple)) and len(bond) >= 2:
            endpoints = [bond[0], bond[1]]
        else:
            raise ValueError(f"{label} must be a bond mapping or [atom_a, atom_b, ...]")
        for endpoint_index, endpoint in enumerate(endpoints):
            value = _integer(endpoint, f"{label}.endpoint[{endpoint_index}]", minimum=0)
            if value >= atom_count:
                raise ValueError(f"{label} references atom index {value} outside the component")


def _resolve_components(value: Any) -> tuple[list[dict[str, Any]], list[list[list[float]]]]:
    resolved = unwrap_artifact(value)
    if isinstance(resolved, dict) and isinstance(resolved.get("result"), dict):
        resolved = resolved["result"]
    if isinstance(resolved, dict) and "components" in resolved:
        resolved = resolved["components"]
    if not isinstance(resolved, list):
        raise ValueError("inputs.components must be an array of explicit AtomicStructure values")
    if not 2 <= len(resolved) <= HARD_MAX_COMPONENTS:
        raise ValueError(
            f"inputs.components must contain between 2 and {HARD_MAX_COMPONENTS} structures"
        )

    components: list[dict[str, Any]] = []
    coordinate_sets: list[list[list[float]]] = []
    seen_maps: dict[int, tuple[int, int]] = {}
    total_atoms = 0
    for component_index, value in enumerate(resolved):
        item = structure_dict(unwrap_artifact(value))
        atoms = item.get("atoms")
        if not isinstance(atoms, list) or not atoms:
            raise ValueError(
                f"components[{component_index}] must use AtomicStructure.atoms so atom "
                "metadata can be preserved"
            )
        _symbols, coordinates = atoms_and_coordinates(item)
        if any(bool(flag) for flag in item.get("pbc", [False, False, False])):
            raise ValueError("Reactive-complex rigid placement currently requires non-periodic components")
        _validate_bonds(item, component_index, len(atoms))
        total_atoms += len(atoms)
        if total_atoms > HARD_MAX_TOTAL_ATOMS:
            raise ValueError(
                "inputs.components exceeds the hard total-atom limit "
                f"{HARD_MAX_TOTAL_ATOMS}"
            )
        for state_field in ("charge", "multiplicity"):
            if state_field not in item or item[state_field] is None:
                raise ValueError(
                    f"components[{component_index}] requires an explicit {state_field}; "
                    "reactive-complex assembly never infers electronic state"
                )
        charge = _integer(item["charge"], f"components[{component_index}].charge")
        multiplicity = _integer(
            item["multiplicity"],
            f"components[{component_index}].multiplicity",
            minimum=1,
        )
        component = copy.deepcopy(item)
        component["charge"] = charge
        component["multiplicity"] = multiplicity
        component["pbc"] = [False, False, False]
        for atom_index, atom in enumerate(component["atoms"]):
            map_number = _atom_map_number(
                atom, f"components[{component_index}].atoms[{atom_index}]"
            )
            if map_number is not None:
                if map_number in seen_maps:
                    first_component, first_atom = seen_maps[map_number]
                    raise ValueError(
                        f"atom-map number {map_number} is duplicated by components[{first_component}]"
                        f".atoms[{first_atom}] and components[{component_index}].atoms[{atom_index}]"
                    )
                seen_maps[map_number] = (component_index, atom_index)
        components.append(component)
        coordinate_sets.append(coordinates)
    return components, coordinate_sets


def _resolve_contacts(
    value: Any,
    components: list[dict[str, Any]],
    default_tolerance: float,
) -> list[dict[str, Any]]:
    resolved = unwrap_artifact(value)
    if isinstance(resolved, dict) and isinstance(resolved.get("result"), dict):
        resolved = resolved["result"]
    if isinstance(resolved, dict) and "contact_pairs" in resolved:
        resolved = resolved["contact_pairs"]
    if not isinstance(resolved, list) or not resolved:
        raise ValueError("inputs.contact_pairs must be a non-empty array")
    if len(resolved) > HARD_MAX_CONTACTS:
        raise ValueError(f"inputs.contact_pairs cannot exceed {HARD_MAX_CONTACTS} contacts")

    contacts = []
    seen = set()
    for contact_index, raw in enumerate(resolved):
        if not isinstance(raw, dict):
            raise ValueError(f"contact_pairs[{contact_index}] must be a mapping")
        prefix = f"contact_pairs[{contact_index}]"
        component_a = _integer(raw.get("component_a"), f"{prefix}.component_a", minimum=0)
        component_b = _integer(raw.get("component_b"), f"{prefix}.component_b", minimum=0)
        if component_a >= len(components) or component_b >= len(components):
            raise ValueError(f"{prefix} references a component outside inputs.components")
        if component_a == component_b:
            raise ValueError(f"{prefix} must connect two different components")
        atom_a = _integer(raw.get("atom_a"), f"{prefix}.atom_a", minimum=0)
        atom_b = _integer(raw.get("atom_b"), f"{prefix}.atom_b", minimum=0)
        if atom_a >= len(components[component_a]["atoms"]):
            raise ValueError(f"{prefix}.atom_a is outside components[{component_a}]")
        if atom_b >= len(components[component_b]["atoms"]):
            raise ValueError(f"{prefix}.atom_b is outside components[{component_b}]")
        target = _positive_float(
            raw.get("target_distance_angstrom"),
            f"{prefix}.target_distance_angstrom",
        )
        tolerance = _positive_float(
            raw.get("tolerance_angstrom", default_tolerance),
            f"{prefix}.tolerance_angstrom",
        )
        normalized_pair = tuple(
            sorted(((component_a, atom_a), (component_b, atom_b)))
        )
        if normalized_pair in seen:
            raise ValueError(f"{prefix} duplicates an earlier atom-pair contact")
        seen.add(normalized_pair)
        contacts.append(
            {
                "contact_index": contact_index,
                "component_a": component_a,
                "atom_a": atom_a,
                "component_b": component_b,
                "atom_b": atom_b,
                "target_distance_angstrom": target,
                "tolerance_angstrom": tolerance,
            }
        )
    return contacts


def _placement_order(
    component_count: int, contacts: list[dict[str, Any]]
) -> list[tuple[int, dict[str, Any]]]:
    placed = {0}
    order: list[tuple[int, dict[str, Any]]] = []
    while len(placed) < component_count:
        selected = None
        for component_index in range(component_count):
            if component_index in placed:
                continue
            for contact in contacts:
                endpoints = {contact["component_a"], contact["component_b"]}
                if component_index in endpoints and len(endpoints & placed) == 1:
                    selected = (component_index, contact)
                    break
            if selected is not None:
                break
        if selected is None:
            missing = sorted(set(range(component_count)) - placed)
            raise ValueError(
                "contact_pairs must form one connected component graph; "
                f"components {missing} are disconnected from component 0"
            )
        order.append(selected)
        placed.add(selected[0])
    return order


def _contact_coordinates(
    contact: dict[str, Any], coordinate_sets: list[list[list[float]]]
) -> tuple[list[float], list[float]]:
    return (
        coordinate_sets[contact["component_a"]][contact["atom_a"]],
        coordinate_sets[contact["component_b"]][contact["atom_b"]],
    )


def _contact_diagnostics(
    contacts: list[dict[str, Any]], coordinate_sets: list[list[list[float]]]
) -> list[dict[str, Any]]:
    diagnostics = []
    for contact in contacts:
        first, second = _contact_coordinates(contact, coordinate_sets)
        actual = _distance(first, second)
        deviation = actual - contact["target_distance_angstrom"]
        diagnostics.append(
            {
                **contact,
                "actual_distance_angstrom": actual,
                "deviation_angstrom": deviation,
                "absolute_deviation_angstrom": abs(deviation),
                "within_tolerance": abs(deviation) <= contact["tolerance_angstrom"],
            }
        )
    return diagnostics


def _collision_diagnostics(
    components: list[dict[str, Any]],
    coordinate_sets: list[list[list[float]]],
    threshold: float,
    max_reported: int,
    *,
    selected_components: set[int] | None = None,
) -> dict[str, Any]:
    selected = selected_components or set(range(len(components)))
    collision_count = 0
    minimum_distance = math.inf
    pairs = []
    for component_a in sorted(selected):
        for component_b in sorted(selected):
            if component_b <= component_a:
                continue
            for atom_a, first in enumerate(coordinate_sets[component_a]):
                for atom_b, second in enumerate(coordinate_sets[component_b]):
                    distance = _distance(first, second)
                    minimum_distance = min(minimum_distance, distance)
                    if distance >= threshold:
                        continue
                    collision_count += 1
                    if len(pairs) < max_reported:
                        pairs.append(
                            {
                                "component_a": component_a,
                                "atom_a": atom_a,
                                "element_a": components[component_a]["atoms"][atom_a].get("element"),
                                "component_b": component_b,
                                "atom_b": atom_b,
                                "element_b": components[component_b]["atoms"][atom_b].get("element"),
                                "distance_angstrom": distance,
                                "threshold_angstrom": threshold,
                            }
                        )
    return {
        "threshold_angstrom": threshold,
        "collision_count": collision_count,
        "minimum_intercomponent_distance_angstrom": (
            None if math.isinf(minimum_distance) else minimum_distance
        ),
        "reported_pairs": pairs,
        "reported_pair_count": len(pairs),
        "reports_truncated": collision_count > len(pairs),
    }


def _moving_collision_count(
    moving_component: int,
    placed: set[int],
    coordinate_sets: list[list[list[float]]],
    threshold: float,
) -> int:
    count = 0
    moving_coordinates = coordinate_sets[moving_component]
    for fixed_component in placed:
        for first in moving_coordinates:
            for second in coordinate_sets[fixed_component]:
                if _distance(first, second) < threshold:
                    count += 1
    return count


def _trial_score(
    moving_component: int,
    placed: set[int],
    contacts: list[dict[str, Any]],
    coordinate_sets: list[list[list[float]]],
    collision_threshold: float,
) -> tuple[int, int, float]:
    applicable = [
        contact
        for contact in contacts
        if moving_component in {contact["component_a"], contact["component_b"]}
        and ({contact["component_a"], contact["component_b"]} - {moving_component}) <= placed
    ]
    diagnostics = _contact_diagnostics(applicable, coordinate_sets)
    violations = sum(not item["within_tolerance"] for item in diagnostics)
    residual = sum(
        (item["absolute_deviation_angstrom"] / item["tolerance_angstrom"]) ** 2
        for item in diagnostics
    )
    collision_count = _moving_collision_count(
        moving_component,
        placed,
        coordinate_sets,
        collision_threshold,
    )
    return violations, collision_count, residual


def _allowed_multiplicities(component_multiplicities: list[int]) -> list[int]:
    doubled_spins = {0}
    for multiplicity in component_multiplicities:
        component_spin = multiplicity - 1
        combined = set()
        for current in doubled_spins:
            combined.update(range(abs(current - component_spin), current + component_spin + 1, 2))
        doubled_spins = combined
    return sorted(spin + 1 for spin in doubled_spins)


def _complex_state(components: list[dict[str, Any]], settings: dict[str, Any]) -> dict[str, Any]:
    component_charges = [int(component["charge"]) for component in components]
    component_multiplicities = [int(component["multiplicity"]) for component in components]
    charge = sum(component_charges)
    if "complex_charge" in settings:
        supplied_charge = _integer(settings["complex_charge"], "complex_charge")
        if supplied_charge != charge:
            raise ValueError(
                f"complex_charge={supplied_charge} does not equal the sum of unchanged "
                f"component charges ({charge})"
            )
    allowed = _allowed_multiplicities(component_multiplicities)
    if "complex_multiplicity" in settings:
        multiplicity = _integer(
            settings["complex_multiplicity"], "complex_multiplicity", minimum=1
        )
        if multiplicity not in allowed:
            raise ValueError(
                f"complex_multiplicity={multiplicity} is incompatible with component multiplicities; "
                f"allowed values are {allowed}"
            )
        multiplicity_source = "explicit"
    elif len(allowed) == 1:
        multiplicity = allowed[0]
        multiplicity_source = "uniquely_derived"
    else:
        raise ValueError(
            "Multiple total-spin couplings are possible; supply action_settings.complex_multiplicity "
            f"from the allowed values {allowed}"
        )
    return {
        "charge": charge,
        "multiplicity": multiplicity,
        "component_charges": component_charges,
        "component_multiplicities": component_multiplicities,
        "allowed_multiplicities": allowed,
        "charge_source": "sum_of_unchanged_components",
        "multiplicity_source": multiplicity_source,
    }


def _transformed_component(
    component: dict[str, Any], coordinates: list[list[float]]
) -> dict[str, Any]:
    transformed = copy.deepcopy(component)
    for atom, position in zip(transformed["atoms"], coordinates):
        atom["position_angstrom"] = [float(value) for value in position]
        if "position" in atom:
            atom["position"] = [float(value) for value in position]
    return transformed


def _offset_bond(bond: Any, offset: int) -> Any:
    shifted = copy.deepcopy(bond)
    if isinstance(shifted, dict):
        keys = _bond_index_keys(shifted)
        assert keys is not None
        shifted[keys[0]] = int(shifted[keys[0]]) + offset
        shifted[keys[1]] = int(shifted[keys[1]]) + offset
        return shifted
    shifted = list(shifted)
    shifted[0] = int(shifted[0]) + offset
    shifted[1] = int(shifted[1]) + offset
    return shifted


def _combined_structure(
    components: list[dict[str, Any]],
    coordinate_sets: list[list[list[float]]],
    state: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, int]]]:
    atoms = []
    bonds = []
    ranges = []
    offset = 0
    for component_index, (component, coordinates) in enumerate(zip(components, coordinate_sets)):
        start = offset
        for atom_index, (atom, position) in enumerate(zip(component["atoms"], coordinates)):
            combined_atom = copy.deepcopy(atom)
            combined_atom["position_angstrom"] = [float(value) for value in position]
            if "position" in combined_atom:
                combined_atom["position"] = [float(value) for value in position]
            combined_atom["component_index"] = component_index
            combined_atom["component_atom_index"] = atom_index
            atoms.append(combined_atom)
        for bond in component.get("bonds") or []:
            bonds.append(_offset_bond(bond, offset))
        offset += len(component["atoms"])
        ranges.append(
            {
                "component_index": component_index,
                "start_atom_index": start,
                "stop_atom_index_exclusive": offset,
            }
        )
    structure: dict[str, Any] = {
        "atoms": atoms,
        "charge": state["charge"],
        "multiplicity": state["multiplicity"],
        "pbc": [False, False, False],
    }
    if bonds:
        structure["bonds"] = bonds
    return structure, ranges


def _centroid(coordinates: list[list[float]]) -> list[float]:
    if not coordinates:
        return [0.0, 0.0, 0.0]
    return [
        sum(position[index] for position in coordinates) / len(coordinates)
        for index in range(3)
    ]


def _translate_coordinates(
    coordinates: list[list[float]], translation: list[float]
) -> list[list[float]]:
    return [
        [position[index] + translation[index] for index in range(3)]
        for position in coordinates
    ]


def _rotate_coordinates_about_centroid(
    coordinates: list[list[float]], rotation: list[list[float]]
) -> list[list[float]]:
    center = _centroid(coordinates)
    rotated = []
    for position in coordinates:
        relative = [position[index] - center[index] for index in range(3)]
        transformed = _matrix_vector(rotation, relative)
        rotated.append([transformed[index] + center[index] for index in range(3)])
    return rotated


def _build_author_case1_candidate(
    candidate_index: int,
    generator: random.Random,
    random_draw_start: int,
    components: list[dict[str, Any]],
    original_coordinates: list[list[list[float]]],
    contacts: list[dict[str, Any]],
    state: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    """复现作者 20 Å 质心合并和十次尝试语义。"""

    coordinate_sets = copy.deepcopy(original_coordinates)
    transforms = [
        {
            "component_index": 0,
            "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "translation_angstrom": [0.0, 0.0, 0.0],
            "anchor_component": True,
        }
    ]
    placed_coordinates = copy.deepcopy(coordinate_sets[0])
    distance = float(settings["initial_centroid_distance_angstrom"])
    for component_index in range(1, len(components)):
        current = copy.deepcopy(original_coordinates[component_index])
        if candidate_index > 0:
            rotation = _author_euler_rotation(generator)
            current = _rotate_coordinates_about_centroid(current, rotation)
            random_translation = [generator.uniform(-5.0, 5.0) for _ in range(3)]
            current = _translate_coordinates(current, random_translation)
        else:
            rotation = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            random_translation = [0.0, 0.0, 0.0]
        fixed_centroid = _centroid(placed_coordinates)
        moving_centroid = _centroid(current)
        direction = [fixed_centroid[index] - moving_centroid[index] for index in range(3)]
        norm = math.sqrt(sum(value * value for value in direction))
        if norm == 0.0:
            direction = [1.0, 0.0, 0.0]
            norm = 1.0
        translation = [
            direction[index] / norm * (distance - norm)
            for index in range(3)
        ]
        current = _translate_coordinates(current, translation)
        coordinate_sets[component_index] = current
        placed_coordinates.extend(current)
        transforms.append(
            {
                "component_index": component_index,
                "rotation_matrix": rotation,
                "translation_angstrom": translation,
                "random_translation_angstrom": random_translation,
                "anchor_component": False,
                "initial_centroid_distance_angstrom": distance,
            }
        )

    collision_diagnostics = _collision_diagnostics(
        components,
        coordinate_sets,
        settings["minimum_interatomic_distance_angstrom"],
        settings["maximum_reported_collisions"],
    )
    contact_diagnostics = _contact_diagnostics(contacts, coordinate_sets)
    # 作者流程没有在刚体合并后用碰撞距离预筛选；所有十次尝试都先进入
    # 约束 MMFF，随后由成键距离检查决定是否保留。碰撞信息仍保留为内部诊断。
    accepted = True
    transformed_components = [
        _transformed_component(component, coordinates)
        for component, coordinates in zip(components, coordinate_sets)
    ]
    structure, ranges = _combined_structure(components, coordinate_sets, state)
    return {
        "candidate_id": f"candidate_{candidate_index + 1:04d}",
        "candidate_seed": None,
        "random_draw_start": int(random_draw_start),
        "accepted": accepted,
        "structure": structure,
        "components": transformed_components,
        "component_atom_ranges": ranges,
        "component_transforms": transforms,
        "contact_diagnostics": contact_diagnostics,
        "collision_diagnostics": collision_diagnostics,
        "acceptance_reason": "deferred_to_mmff_distance_gate",
    }


def _build_candidate(
    candidate_index: int,
    candidate_seed: int,
    components: list[dict[str, Any]],
    original_coordinates: list[list[list[float]]],
    contacts: list[dict[str, Any]],
    placement_order: list[tuple[int, dict[str, Any]]],
    state: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    generator = random.Random(candidate_seed)
    coordinate_sets = copy.deepcopy(original_coordinates)
    transforms = [
        {
            "component_index": 0,
            "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "translation_angstrom": [0.0, 0.0, 0.0],
            "anchor_component": True,
        }
    ]
    placed = {0}
    collision_threshold = settings["minimum_interatomic_distance_angstrom"]
    for moving_component, primary_contact in placement_order:
        if primary_contact["component_a"] == moving_component:
            moving_atom = primary_contact["atom_a"]
            fixed_component = primary_contact["component_b"]
            fixed_atom = primary_contact["atom_b"]
        else:
            moving_atom = primary_contact["atom_b"]
            fixed_component = primary_contact["component_a"]
            fixed_atom = primary_contact["atom_a"]
        fixed_position = coordinate_sets[fixed_component][fixed_atom]
        best = None
        for trial_index in range(settings["orientation_trials"]):
            rotation = _random_rotation(generator)
            direction = _random_unit_vector(generator)
            target_position = [
                fixed_position[index]
                + primary_contact["target_distance_angstrom"] * direction[index]
                for index in range(3)
            ]
            trial_coordinates, translation = _rigid_transform(
                original_coordinates[moving_component],
                rotation,
                original_coordinates[moving_component][moving_atom],
                target_position,
            )
            coordinate_sets[moving_component] = trial_coordinates
            score = _trial_score(
                moving_component,
                placed,
                contacts,
                coordinate_sets,
                collision_threshold,
            )
            key = (*score, trial_index)
            if best is None or key < best[0]:
                best = (key, copy.deepcopy(trial_coordinates), rotation, translation, trial_index)
        assert best is not None
        coordinate_sets[moving_component] = best[1]
        transforms.append(
            {
                "component_index": moving_component,
                "rotation_matrix": best[2],
                "translation_angstrom": best[3],
                "anchor_component": False,
                "placement_contact_index": primary_contact["contact_index"],
                "selected_orientation_trial": best[4],
            }
        )
        placed.add(moving_component)

    contact_diagnostics = _contact_diagnostics(contacts, coordinate_sets)
    collision_diagnostics = _collision_diagnostics(
        components,
        coordinate_sets,
        collision_threshold,
        settings["maximum_reported_collisions"],
    )
    accepted = all(item["within_tolerance"] for item in contact_diagnostics)
    accepted = accepted and collision_diagnostics["collision_count"] == 0
    transformed_components = [
        _transformed_component(component, coordinates)
        for component, coordinates in zip(components, coordinate_sets)
    ]
    structure, ranges = _combined_structure(components, coordinate_sets, state)
    return {
        "candidate_id": f"candidate_{candidate_index + 1:04d}",
        "candidate_seed": candidate_seed,
        "accepted": accepted,
        "structure": structure,
        "components": transformed_components,
        "component_atom_ranges": ranges,
        "component_transforms": sorted(transforms, key=lambda item: item["component_index"]),
        "contact_diagnostics": contact_diagnostics,
        "collision_diagnostics": collision_diagnostics,
    }


def _settings(raw: dict[str, Any], atom_counts: list[int]) -> dict[str, Any]:
    if "contact_validation" in raw:
        raise ValueError(
            "action_settings.contact_validation is not supported; contact and collision "
            "gates are always evaluated by the backend"
        )
    max_candidates = _integer(raw.get("max_candidates", 32), "max_candidates", minimum=1)
    if max_candidates > HARD_MAX_CANDIDATES:
        raise ValueError(f"max_candidates cannot exceed the hard limit {HARD_MAX_CANDIDATES}")
    candidate_count = _integer(raw.get("candidate_count", 1), "candidate_count", minimum=1)
    if candidate_count > max_candidates:
        raise ValueError("candidate_count cannot exceed max_candidates")
    orientation_trials = _integer(
        raw.get("orientation_trials", 32), "orientation_trials", minimum=1
    )
    if orientation_trials > HARD_MAX_ORIENTATION_TRIALS:
        raise ValueError(
            f"orientation_trials cannot exceed the hard limit {HARD_MAX_ORIENTATION_TRIALS}"
        )
    maximum_reported = _integer(
        raw.get("maximum_reported_collisions", 32),
        "maximum_reported_collisions",
        minimum=0,
    )
    if maximum_reported > HARD_MAX_REPORTED_COLLISIONS:
        raise ValueError(
            "maximum_reported_collisions cannot exceed the hard limit "
            f"{HARD_MAX_REPORTED_COLLISIONS}"
        )
    pairwise_per_trial = sum(
        atom_counts[left] * atom_counts[right]
        for left in range(len(atom_counts))
        for right in range(left + 1, len(atom_counts))
    )
    estimated = candidate_count * orientation_trials * pairwise_per_trial
    if estimated > HARD_MAX_PAIRWISE_EVALUATIONS:
        raise ValueError(
            "Requested candidate/orientation search exceeds the hard pairwise-evaluation "
            f"budget ({estimated} > {HARD_MAX_PAIRWISE_EVALUATIONS}); reduce candidate_count "
            "or orientation_trials"
        )
    total_atoms = sum(atom_counts)
    estimated_output_atom_records = 2 * candidate_count * total_atoms
    if estimated_output_atom_records > HARD_MAX_CANDIDATE_OUTPUT_ATOM_RECORDS:
        raise ValueError(
            "Requested candidate ensemble exceeds the hard candidate-output atom-record "
            f"budget ({estimated_output_atom_records} > "
            f"{HARD_MAX_CANDIDATE_OUTPUT_ATOM_RECORDS}); reduce candidate_count"
        )
    return {
        "random_seed": _integer(raw.get("random_seed", 0), "random_seed"),
        # 固定 Case 1 的服务端调用把全局 random.seed(42) 的已消耗 uniform
        # 数量传入这里。这样每次原子 Action 虽然独立创建生成器，仍复现一个
        # 连续随机流，而不是为每个构象组合重新开始或派生子 seed。
        "random_stream_offset": _integer(
            raw.get("random_stream_offset", 0),
            "random_stream_offset",
            minimum=0,
        ),
        "candidate_count": candidate_count,
        "max_candidates": max_candidates,
        "orientation_trials": orientation_trials,
        "contact_tolerance_angstrom": _positive_float(
            raw.get("contact_tolerance_angstrom", 0.25),
            "contact_tolerance_angstrom",
        ),
        "minimum_interatomic_distance_angstrom": _positive_float(
            raw.get("minimum_interatomic_distance_angstrom", 0.75),
            "minimum_interatomic_distance_angstrom",
            allow_zero=True,
        ),
        "maximum_reported_collisions": maximum_reported,
        "estimated_pairwise_evaluations": estimated,
        "total_atom_count": total_atoms,
        "estimated_output_atom_records": estimated_output_atom_records,
        "initial_centroid_distance_angstrom": (
            None
            if raw.get("initial_centroid_distance_angstrom") is None
            else _positive_float(
                raw["initial_centroid_distance_angstrom"],
                "initial_centroid_distance_angstrom",
            )
        ),
    }


def _attempt_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    """Keep rejection auditing separate without duplicating coordinate payloads."""

    return {
        "candidate_id": candidate["candidate_id"],
        "candidate_seed": candidate["candidate_seed"],
        "accepted": candidate["accepted"],
        "contact_diagnostics": copy.deepcopy(candidate["contact_diagnostics"]),
        "collision_diagnostics": copy.deepcopy(candidate["collision_diagnostics"]),
    }


def _summarize_reactive_complex_rejections(
    attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    """汇总候选拒绝原因，供公开错误和 Revision 调参使用。

    输入是已经移除坐标正文的候选尝试；动作累计接触约束违反数、发生碰撞的
    候选数和最大碰撞数，并读取最小组分间距离。函数不修改候选、不写文件，
    返回只含有界数值与固定修复建议的字典；字段缺失时按零计数并省略距离。
    """

    contact_violation_count = 0
    collision_candidate_count = 0
    maximum_collision_count = 0
    minimum_intercomponent_distance: float | None = None
    for attempt in attempts:
        contacts = attempt.get("contact_diagnostics")
        if isinstance(contacts, list):
            contact_violation_count += sum(
                1
                for item in contacts
                if isinstance(item, dict) and item.get("within_tolerance") is not True
            )
        collisions = attempt.get("collision_diagnostics")
        if not isinstance(collisions, dict):
            continue
        collision_count = collisions.get("collision_count")
        if isinstance(collision_count, int) and not isinstance(collision_count, bool):
            maximum_collision_count = max(maximum_collision_count, collision_count)
            if collision_count > 0:
                collision_candidate_count += 1
        distance = collisions.get("minimum_intercomponent_distance_angstrom")
        if isinstance(distance, (int, float)) and not isinstance(distance, bool):
            finite_distance = float(distance)
            if math.isfinite(finite_distance):
                minimum_intercomponent_distance = (
                    finite_distance
                    if minimum_intercomponent_distance is None
                    else min(minimum_intercomponent_distance, finite_distance)
                )

    summary: dict[str, Any] = {
        "failure_stage": "candidate_acceptance",
        "attempted_count": len(attempts),
        "contact_violation_count": contact_violation_count,
        "collision_candidate_count": collision_candidate_count,
        "maximum_collision_count": maximum_collision_count,
        "repair_guidance": (
            "若接触约束未满足，检查 contact_pairs 的原子索引、目标距离和 tolerance；"
            "若候选发生碰撞，在保持物理合理的前提下调整最小原子距离或接触几何，"
            "并可扩大 candidate_count 或 orientation_trials。"
        ),
    }
    if minimum_intercomponent_distance is not None:
        summary["minimum_intercomponent_distance_angstrom"] = round(
            minimum_intercomponent_distance, 8
        )
    return summary


def _nested_result(value: Any) -> Any:
    resolved = unwrap_artifact(value)
    while isinstance(resolved, dict):
        if isinstance(resolved.get("raw_output"), dict):
            resolved = resolved["raw_output"]
            continue
        if isinstance(resolved.get("result"), dict):
            resolved = resolved["result"]
            continue
        break
    return resolved


def _enumerate_conformer_combinations(request: dict[str, Any]) -> dict[str, Any]:
    inputs, _method, settings = request_parts(request)
    ensembles = inputs.get("ensembles")
    if not isinstance(ensembles, list) or not ensembles:
        raise ValueError("inputs.ensembles must be a non-empty array")
    if len(ensembles) > HARD_MAX_COMPONENTS:
        raise ValueError(f"inputs.ensembles cannot exceed {HARD_MAX_COMPONENTS} components")
    maximum = _integer(
        settings.get("max_combinations"), "max_combinations", minimum=1
    )
    if maximum > HARD_MAX_CONFORMER_COMBINATIONS:
        raise ValueError(
            f"max_combinations exceeds hard limit {HARD_MAX_CONFORMER_COMBINATIONS}"
        )

    normalized: list[list[dict[str, Any]]] = []
    for ensemble_index, value in enumerate(ensembles):
        payload = _nested_result(value)
        records = payload.get("ensemble") if isinstance(payload, dict) else None
        if not isinstance(records, list) or not records:
            raise ValueError(f"ensembles[{ensemble_index}] contains no conformers")
        component_records = []
        for conformer_index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(
                    f"ensembles[{ensemble_index}].ensemble[{conformer_index}] must be a mapping"
                )
            structure = structure_dict(record.get("structure", record))
            atoms_and_coordinates(structure)
            component_records.append(
                {
                    "conformer_id": str(record.get("conformer_id", conformer_index)),
                    "structure": structure,
                }
            )
        normalized.append(component_records)

    total = math.prod(len(records) for records in normalized)
    combinations = []
    for index, records in enumerate(itertools.islice(itertools.product(*normalized), maximum)):
        combinations.append(
            {
                "combination_id": f"combination_{index + 1:04d}",
                "conformer_ids": [record["conformer_id"] for record in records],
                "components": [copy.deepcopy(record["structure"]) for record in records],
            }
        )
    return success(
        {
            "combinations": combinations,
            "combination_count": len(combinations),
            "total_possible_combinations": total,
            "component_count": len(normalized),
            "truncated": total > maximum,
            "max_combinations": maximum,
        },
        provenance={
            "operation": "bounded_cartesian_product",
            "ordering": "input_ensemble_order_then_conformer_order",
            "structures_generated": False,
            "ranking_performed": False,
        },
    )


def _resolve_mapped_reactive_contacts(request: dict[str, Any]) -> dict[str, Any]:
    inputs, _method, settings = request_parts(request)
    raw_components = inputs.get("components")
    if not isinstance(raw_components, list) or not raw_components:
        raise ValueError("inputs.components must be a non-empty array")
    components, _coordinates = _resolve_components(raw_components)
    changes = _nested_result(inputs.get("reaction_changes"))
    if not isinstance(changes, dict):
        raise ValueError("inputs.reaction_changes must resolve to a ReactionChangeSet")
    direction = str(settings.get("direction") or "").strip().lower()
    if direction not in {"forward", "reverse"}:
        raise ValueError("direction must be forward or reverse")
    scale = _positive_float(
        settings.get("equilibrium_distance_scale"), "equilibrium_distance_scale"
    )
    tolerance = _positive_float(
        settings.get("contact_tolerance_angstrom"), "contact_tolerance_angstrom"
    )

    locations: dict[int, tuple[int, int, str]] = {}
    for component_index, component in enumerate(components):
        for atom_index, atom in enumerate(component["atoms"]):
            map_number = _atom_map_number(
                atom, f"components[{component_index}].atoms[{atom_index}]"
            )
            if map_number is None:
                continue
            locations[map_number] = (
                component_index,
                atom_index,
                str(atom.get("element") or atom.get("symbol")),
            )

    explicit = inputs.get("forming_bonds")
    if explicit is not None:
        raw_bonds = _nested_result(explicit)
    else:
        raw_bonds = changes.get("formed_bonds" if direction == "forward" else "broken_bonds")
    if not isinstance(raw_bonds, list) or not raw_bonds:
        raise ValueError(
            "No forming bonds are available; supply inputs.forming_bonds explicitly"
        )

    contact_pairs = []
    intracomponent_bonds = []
    for index, record in enumerate(raw_bonds):
        if not isinstance(record, dict):
            raise ValueError(f"forming_bonds[{index}] must be a mapping")
        first_map = _integer(record.get("atom_map_a"), f"forming_bonds[{index}].atom_map_a", minimum=1)
        second_map = _integer(record.get("atom_map_b"), f"forming_bonds[{index}].atom_map_b", minimum=1)
        if first_map not in locations or second_map not in locations:
            raise ValueError(
                f"forming_bonds[{index}] references atom maps absent from components"
            )
        first_component, first_atom, first_element = locations[first_map]
        second_component, second_atom, second_element = locations[second_map]
        if first_component == second_component:
            intracomponent_bonds.append(copy.deepcopy(record))
            continue
        key = tuple(sorted((first_element, second_element)))
        base_distance = _ARCHE_SINGLE_BOND_LENGTHS_ANGSTROM.get(key)
        if base_distance is None:
            raise ValueError(
                f"No ARCHE single-bond reference distance for {first_element}-{second_element}"
            )
        contact_pairs.append(
            {
                "component_a": first_component,
                "atom_a": first_atom,
                "component_b": second_component,
                "atom_b": second_atom,
                "atom_map_a": first_map,
                "atom_map_b": second_map,
                "target_distance_angstrom": base_distance * scale,
                "tolerance_angstrom": tolerance,
            }
        )
    if not contact_pairs:
        raise ValueError("No inter-component forming bond was resolved")
    return success(
        {
            "components": components,
            "contact_pairs": contact_pairs,
            "intracomponent_forming_bonds": intracomponent_bonds,
            "direction": direction,
            "equilibrium_distance_scale": scale,
            "distance_reference": "transition_state_path_single_bond_table_v1",
        },
        provenance={
            "operation": "mapped_atom_to_component_contact_resolution",
            "mechanism_inferred": False,
            "coordinates_modified": False,
        },
    )


def _build_mapped_distance_constraints(request: dict[str, Any]) -> dict[str, Any]:
    inputs, _method, settings = request_parts(request)
    structure = structure_dict(inputs.get("structure"))
    atoms = structure.get("atoms")
    if not isinstance(atoms, list) or not atoms:
        raise ValueError("inputs.structure must contain a non-empty atoms array")
    atoms_and_coordinates(structure)
    raw_bonds = _nested_result(inputs.get("mapped_bonds"))
    if not isinstance(raw_bonds, list) or not raw_bonds:
        raise ValueError("inputs.mapped_bonds must be a non-empty array")
    if len(raw_bonds) > HARD_MAX_CONTACTS:
        raise ValueError(f"inputs.mapped_bonds cannot exceed {HARD_MAX_CONTACTS} pairs")

    scale = _positive_float(
        settings.get("equilibrium_distance_scale"), "equilibrium_distance_scale"
    )
    minimum_ratio = _positive_float(
        settings.get("minimum_distance_ratio"), "minimum_distance_ratio"
    )
    maximum_ratio = _positive_float(
        settings.get("maximum_distance_ratio"), "maximum_distance_ratio"
    )
    if maximum_ratio < minimum_ratio:
        raise ValueError("maximum_distance_ratio must be at least minimum_distance_ratio")
    force_constant = _positive_float(
        settings.get("force_constant_kcal_mol_angstrom2"),
        "force_constant_kcal_mol_angstrom2",
    )

    locations: dict[int, tuple[int, str]] = {}
    for atom_index, atom in enumerate(atoms):
        if not isinstance(atom, dict):
            raise ValueError(f"structure.atoms[{atom_index}] must be a mapping")
        map_number = _atom_map_number(atom, f"structure.atoms[{atom_index}]")
        if map_number is None:
            continue
        if map_number in locations:
            raise ValueError(f"atom map {map_number} is duplicated in structure.atoms")
        element = str(atom.get("element") or atom.get("symbol") or "").strip()
        if not element:
            raise ValueError(f"structure.atoms[{atom_index}] has no element")
        locations[map_number] = (atom_index, element)

    constraints = []
    seen_pairs: set[tuple[int, int]] = set()
    for pair_index, record in enumerate(raw_bonds):
        if not isinstance(record, dict):
            raise ValueError(f"mapped_bonds[{pair_index}] must be a mapping")
        first_map = _integer(
            record.get("atom_map_a"), f"mapped_bonds[{pair_index}].atom_map_a", minimum=1
        )
        second_map = _integer(
            record.get("atom_map_b"), f"mapped_bonds[{pair_index}].atom_map_b", minimum=1
        )
        pair = tuple(sorted((first_map, second_map)))
        if pair[0] == pair[1] or pair in seen_pairs:
            raise ValueError(f"mapped_bonds[{pair_index}] is self-referential or duplicated")
        seen_pairs.add(pair)
        if first_map not in locations or second_map not in locations:
            raise ValueError(f"mapped_bonds[{pair_index}] references an absent atom map")
        first_index, first_element = locations[first_map]
        second_index, second_element = locations[second_map]
        element_pair = tuple(sorted((first_element, second_element)))
        reference = _ARCHE_SINGLE_BOND_LENGTHS_ANGSTROM.get(element_pair)
        if reference is None:
            raise ValueError(
                f"No ARCHE single-bond reference distance for {first_element}-{second_element}"
            )
        equilibrium = reference * scale
        constraints.append(
            {
                "atom_indices": [first_index, second_index],
                "atom_maps": [first_map, second_map],
                "reference_bond_length_angstrom": reference,
                "equilibrium_distance_angstrom": equilibrium,
                "minimum_distance_angstrom": equilibrium * minimum_ratio,
                "maximum_distance_angstrom": equilibrium * maximum_ratio,
                "force_constant_kcal_mol_angstrom2": force_constant,
            }
        )
    return success(
        {
            "constraints": constraints,
            "constraint_count": len(constraints),
            "distance_reference": "transition_state_path_single_bond_table_v1",
            "equilibrium_distance_scale": scale,
            "minimum_distance_ratio": minimum_ratio,
            "maximum_distance_ratio": maximum_ratio,
        },
        provenance={
            "operation": "mapped_pair_to_distance_constraint",
            "mechanism_inferred": False,
            "coordinates_modified": False,
        },
    )


def _build(request: dict[str, Any]) -> dict[str, Any]:
    inputs, _method, raw_settings = request_parts(request)
    if "components" not in inputs:
        raise ValueError("build_reactive_complex_ensemble requires inputs.components")
    if "contact_pairs" not in inputs:
        raise ValueError("build_reactive_complex_ensemble requires inputs.contact_pairs")
    components, original_coordinates = _resolve_components(inputs["components"])
    normalized_settings = _settings(
        raw_settings, [len(component["atoms"]) for component in components]
    )
    contacts = _resolve_contacts(
        inputs["contact_pairs"],
        components,
        normalized_settings["contact_tolerance_angstrom"],
    )
    state = _complex_state(components, raw_settings)

    if normalized_settings["initial_centroid_distance_angstrom"] is not None:
        seed_generator = random.Random(normalized_settings["random_seed"])
        for _ in range(normalized_settings["random_stream_offset"]):
            seed_generator.random()
        random_draws_per_candidate = 6 * (len(components) - 1)
        candidates = [
            _build_author_case1_candidate(
                candidate_index,
                seed_generator,
                normalized_settings["random_stream_offset"]
                + max(candidate_index - 1, 0) * random_draws_per_candidate,
                components,
                original_coordinates,
                contacts,
                state,
                normalized_settings,
            )
            for candidate_index in range(normalized_settings["candidate_count"])
        ]
    else:
        seed_generator = random.Random(normalized_settings["random_seed"])
        placement_order = _placement_order(len(components), contacts)
        candidates = [
            _build_candidate(
                candidate_index,
                seed_generator.randrange(0, 2**63),
                components,
                original_coordinates,
                contacts,
                placement_order,
                state,
                normalized_settings,
            )
            for candidate_index in range(normalized_settings["candidate_count"])
        ]
    ensemble = [candidate for candidate in candidates if candidate["accepted"]]
    rejected_candidates = [candidate for candidate in candidates if not candidate["accepted"]]
    attempts = [_attempt_summary(candidate) for candidate in candidates]
    accepted_count = len(ensemble)
    rejected_count = len(rejected_candidates)
    result = {
        "ensemble": ensemble,
        "rejected_candidates": rejected_candidates,
        "attempts": attempts,
        "count": accepted_count,
        "attempted_count": len(candidates),
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "component_count": len(components),
        "contact_pair_count": len(contacts),
        "state_diagnostics": state,
        "assembly_settings": normalized_settings,
        "graph_edits_applied": False,
        "conformers_generated": False,
        "geometry_optimization_performed": False,
    }
    provenance = {
        "algorithm": "seeded_rigid_body_contact_placement",
        "random_seed": normalized_settings["random_seed"],
        "random_stream_offset": normalized_settings["random_stream_offset"],
        "random_stream": (
            "python_random_uniform_continuation"
            if normalized_settings["initial_centroid_distance_angstrom"] is not None
            else "per_candidate_seeded"
        ),
        "anchor_component_index": 0,
        "preserves_component_internal_geometry": True,
    }
    if accepted_count == 0:
        rejection_summary = _summarize_reactive_complex_rejections(attempts)
        response = failed(
            "No reactive-complex candidate passed all contact-tolerance and "
            "inter-component collision gates "
            f"(attempted_count={rejection_summary['attempted_count']}, "
            f"contact_violation_count={rejection_summary['contact_violation_count']}, "
            f"collision_candidate_count={rejection_summary['collision_candidate_count']}, "
            f"maximum_collision_count={rejection_summary['maximum_collision_count']})",
            code="no_accepted_reactive_complex_candidates",
        )
        response["error"].update(rejection_summary)
        response["diagnostics"] = {
            "attempted_count": len(candidates),
            "accepted_count": 0,
            "rejected_count": rejected_count,
            "rejected_candidates": rejected_candidates,
            "attempts": attempts,
            "state_diagnostics": state,
            "assembly_settings": normalized_settings,
        }
        response["warnings"] = [
            "All generated candidates were rejected; no consumable reactive-complex "
            "ensemble was produced"
        ]
        response["provenance"] = provenance
        return response
    if rejected_count:
        warnings = [
            f"{rejected_count} of {len(candidates)} candidates were filtered by the "
            "reactive-complex acceptance gate; inspect candidate diagnostics"
        ]
        return partial_success(result, warnings=warnings, provenance=provenance)
    return success(result, provenance=provenance)


def _select_candidate(request: dict[str, Any]) -> dict[str, Any]:
    inputs, _method, settings = request_parts(request)
    if "ensemble" not in inputs:
        raise ValueError("select_reactive_complex_candidate requires inputs.ensemble")
    value = unwrap_artifact(inputs["ensemble"])
    while isinstance(value, dict) and isinstance(value.get("result"), dict):
        value = value["result"]
    if not isinstance(value, dict):
        raise ValueError("inputs.ensemble must resolve to a ReactiveComplexEnsemble mapping")
    candidates = value.get("ensemble")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(
            "inputs.ensemble contains no accepted reactive-complex candidates"
        )
    candidate_id = str(settings.get("candidate_id") or "").strip()
    if not candidate_id:
        raise ValueError("action_settings.candidate_id must be explicitly set")
    matches = [
        candidate
        for candidate in candidates
        if isinstance(candidate, dict) and candidate.get("candidate_id") == candidate_id
    ]
    if len(matches) != 1:
        available = sorted(
            str(candidate.get("candidate_id"))
            for candidate in candidates
            if isinstance(candidate, dict) and candidate.get("candidate_id")
        )
        raise ValueError(
            f"candidate_id={candidate_id!r} does not identify exactly one accepted "
            f"candidate; available accepted IDs are {available}"
        )
    candidate = matches[0]
    if candidate.get("accepted") is not True:
        raise ValueError("Only an explicitly accepted reactive-complex candidate can be selected")
    structure = candidate.get("structure")
    if not isinstance(structure, dict):
        raise ValueError(f"Accepted candidate {candidate_id!r} contains no AtomicStructure")
    atoms_and_coordinates(structure)
    return success(
        {
            "structure": copy.deepcopy(structure),
            "candidate_id": candidate_id,
            "selection_policy": "explicit_candidate_id",
            "ranking_performed": False,
        },
        provenance={
            "operation": "deterministic accepted-candidate extraction",
            "candidate_id": candidate_id,
            "scientific_ranking_performed": False,
        },
    )


def execute(action_id: str, backend_id: str, request: dict[str, Any]) -> dict[str, Any]:
    if action_id == ENUMERATE_ACTION_ID and backend_id == BACKEND_ID:
        return _enumerate_conformer_combinations(request)
    if action_id == RESOLVE_CONTACTS_ACTION_ID and backend_id == BACKEND_ID:
        return _resolve_mapped_reactive_contacts(request)
    if action_id == BUILD_CONSTRAINTS_ACTION_ID and backend_id == BACKEND_ID:
        return _build_mapped_distance_constraints(request)
    if action_id == ACTION_ID and backend_id == BACKEND_ID:
        return _build(request)
    if action_id == SELECT_ACTION_ID and backend_id == BACKEND_ID:
        return _select_candidate(request)
    return unsupported(
        f"Unsupported reactive-complex action/backend combination: {action_id}/{backend_id}"
    )
