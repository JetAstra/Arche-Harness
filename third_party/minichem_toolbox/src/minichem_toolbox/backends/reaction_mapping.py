"""Atom mapping and deterministic reaction-connectivity transformations."""

from __future__ import annotations

import math
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from importlib import resources as package_resources
from pathlib import Path
from typing import Any

from .common import (
    atoms_and_coordinates,
    module_version,
    partial_success,
    request_parts,
    structure_dict,
    success,
    unavailable,
    unsupported,
    unwrap_artifact,
)

ACTIONS = {
    "map_reaction_atoms",
    "analyze_reaction_changes",
    "apply_reaction_changes_to_structure",
}

_INTERNAL_BACKENDS = {"internal_reaction_analysis"}
_MAP_NUMBER_KEYS = ("map_number", "atom_map_number", "atom_map")
_ALLOWED_BOND_ORDERS = (0.0, 1.0, 1.5, 2.0, 3.0)
_STALE_GRAPH_IDENTIFIER_KEYS = (
    "smiles",
    "canonical_smiles",
    "isomeric_smiles",
    "mapped_smiles",
    "inchi",
    "inchikey",
    "molblock",
    "graph_hash",
    "connectivity_hash",
)

_RXNMAPPER_MODEL_ENV = "MINICHEM_RXNMAPPER_MODEL_PATH"
_RXNMAPPER_RESOURCE_ID = "rxnmapper_albert_uspto_all_1310k"
_RXNMAPPER_PACKAGE_SUBPATH = (
    "models",
    "transformers",
    "albert_heads_8_uspto_all_1310k",
)
_RXNMAPPER_REQUIRED_FILES = (
    "config.json",
    "pytorch_model.bin",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "training_args.bin",
    "vocab.txt",
)


def _load_rdkit_chem():
    try:
        from rdkit import Chem
    except (ImportError, ModuleNotFoundError):
        return None
    return Chem


def _load_rxnmapper_class():
    try:
        from rxnmapper import RXNMapper
    except (ImportError, ModuleNotFoundError):
        return None
    return RXNMapper


def _rxnmapper_resource_status() -> dict[str, Any]:
    """检查 RXNMapper 显式路径或随包 checkpoint，结果不包含本机路径。"""

    configured_path = os.environ.get(_RXNMAPPER_MODEL_ENV, "").strip()
    source = "configured" if configured_path else "package"
    try:
        if configured_path:
            root: Any = Path(configured_path).expanduser()
        else:
            root = package_resources.files("rxnmapper")
            for part in _RXNMAPPER_PACKAGE_SUBPATH:
                root = root.joinpath(part)
    except Exception:
        return {
            "available": False,
            "resource_id": _RXNMAPPER_RESOURCE_ID,
            "reason_code": "resource_probe_failed",
            "source": source,
        }
    try:
        if not root.is_dir():
            return {
                "available": False,
                "resource_id": _RXNMAPPER_RESOURCE_ID,
                "reason_code": "resource_missing",
                "source": source,
            }

        def has_content(candidate: Any) -> bool:
            if not candidate.is_file():
                return False
            with candidate.open("rb") as stream:
                return bool(stream.read(1))

        missing_files = [
            filename
            for filename in _RXNMAPPER_REQUIRED_FILES
            if not has_content(root.joinpath(filename))
        ]
    except (AttributeError, OSError):
        return {
            "available": False,
            "resource_id": _RXNMAPPER_RESOURCE_ID,
            "reason_code": "resource_probe_failed",
            "source": source,
        }
    if missing_files:
        return {
            "available": False,
            "resource_id": _RXNMAPPER_RESOURCE_ID,
            "reason_code": "resource_incomplete",
            "source": source,
            "missing_files": missing_files,
        }
    return {
        "available": True,
        "resource_id": _RXNMAPPER_RESOURCE_ID,
        "reason_code": "ok",
        "source": source,
    }


def _rxnmapper_resource_unavailable(status: Mapping[str, Any]) -> dict[str, Any]:
    """构造不泄漏路径且不可重试的模型资源不可用结果。"""

    result = unavailable(
        "RXNMapper model resource is unavailable; install the package with its bundled checkpoint",
        install="pip install rxnmapper",
    )
    result["error"].update(
        {
            "reason_code": str(status.get("reason_code") or "resource_missing"),
            "resource_id": str(status.get("resource_id") or _RXNMAPPER_RESOURCE_ID),
        }
    )
    if status.get("missing_files"):
        result["error"]["missing_files"] = list(status["missing_files"])
    return result


def _construct_rxnmapper(mapper_class: Any) -> tuple[Any | None, dict[str, Any] | None]:
    """构造 RXNMapper；资源竞态或加载失败统一返回 unavailable。"""

    status = _rxnmapper_resource_status()
    if not status["available"]:
        return None, _rxnmapper_resource_unavailable(status)
    model_path = os.environ.get(_RXNMAPPER_MODEL_ENV, "").strip()
    try:
        mapper = mapper_class({"model_path": model_path}) if model_path else mapper_class()
    except Exception:
        failed_status = dict(status)
        failed_status["available"] = False
        failed_status["reason_code"] = "resource_load_failed"
        return None, _rxnmapper_resource_unavailable(failed_status)
    return mapper, None


def _unwrap_value(value: Any) -> Any:
    item = unwrap_artifact(value)
    while isinstance(item, Mapping) and isinstance(item.get("result"), Mapping):
        item = item["result"]
    return item


def _component_smiles(value: Any, *, field_name: str) -> str:
    item = _unwrap_value(value)
    if isinstance(item, str):
        text = item.strip()
    elif isinstance(item, Mapping):
        text = str(
            item.get("smiles")
            or item.get("mapped_smiles")
            or item.get("canonical_smiles")
            or ""
        ).strip()
    else:
        text = ""
    if not text:
        raise ValueError(f"{field_name} contains an empty or unsupported molecule")
    if ">" in text:
        raise ValueError(
            f"{field_name} components must be molecule SMILES, not reaction SMILES"
        )
    return text


def _reaction_side(value: Any, *, field_name: str) -> str:
    item = _unwrap_value(value)
    if isinstance(item, str):
        text = item.strip()
        if not text:
            raise ValueError(f"{field_name} must not be empty")
        if ">" in text:
            raise ValueError(
                f"{field_name} must contain molecule SMILES, not a reaction"
            )
        return text
    if not isinstance(item, Sequence) or isinstance(item, (bytes, bytearray)):
        raise ValueError(
            f"{field_name} must be a SMILES string or a non-empty molecule list"
        )
    components = [
        _component_smiles(component, field_name=f"{field_name}[{index}]")
        for index, component in enumerate(item)
    ]
    if not components:
        raise ValueError(f"{field_name} must contain at least one molecule")
    return ".".join(components)


def _reaction_value_to_smiles(value: Any) -> str:
    item = _unwrap_value(value)
    if isinstance(item, str):
        return item.strip()
    if not isinstance(item, Mapping):
        raise ValueError(
            "reaction must be a reaction SMILES string or a reaction mapping"
        )
    for key in (
        "mapped_reaction_smiles",
        "mapped_rxn",
        "reaction_smiles",
    ):
        if item.get(key):
            return str(item[key]).strip()
    if item.get("reaction") is not None:
        return _reaction_value_to_smiles(item["reaction"])
    if "reactants" in item and "products" in item:
        return (
            _reaction_side(item["reactants"], field_name="reactants")
            + ">>"
            + _reaction_side(item["products"], field_name="products")
        )
    raise ValueError(
        "reaction mapping requires mapped_reaction_smiles/mapped_rxn/reaction_smiles "
        "or reactants and products"
    )


def _reaction_from_inputs(inputs: Mapping[str, Any], *, mapped: bool) -> str:
    keys = (
        ("mapped_reaction", "atom_mapped_reaction", "reaction")
        if mapped
        else ("reaction", "reaction_smiles")
    )
    for key in keys:
        if key in inputs:
            return _reaction_value_to_smiles(inputs[key])
    if "reactants" in inputs and "products" in inputs:
        return (
            _reaction_side(inputs["reactants"], field_name="reactants")
            + ">>"
            + _reaction_side(inputs["products"], field_name="products")
        )
    expected = "mapped_reaction" if mapped else "reaction"
    raise ValueError(
        f"inputs.{expected}, or both inputs.reactants and inputs.products, are required"
    )


def _split_reaction(reaction_smiles: str) -> tuple[str, str]:
    text = reaction_smiles.strip()
    if text.count(">>") != 1:
        raise ValueError("Reaction SMILES must contain exactly one '>>' separator")
    reactants, products = (part.strip() for part in text.split(">>", 1))
    if not reactants or not products:
        raise ValueError(
            "Reaction SMILES requires non-empty reactant and product sides"
        )
    for side_name, side in (("reactant", reactants), ("product", products)):
        if side.startswith(".") or side.endswith(".") or ".." in side:
            raise ValueError(f"Reaction SMILES contains an empty {side_name} component")
    return reactants, products


def _parse_side(chem: Any, side: str, *, side_name: str) -> list[Any]:
    parameters = chem.SmilesParserParams()
    parameters.removeHs = False
    molecules = []
    for component_index, component in enumerate(side.split(".")):
        molecule = chem.MolFromSmiles(component, parameters)
        if molecule is None:
            raise ValueError(
                f"RDKit could not parse {side_name} component "
                f"{component_index}: {component!r}"
            )
        molecules.append(molecule)
    return molecules


def _parse_reaction(chem: Any, reaction_smiles: str) -> tuple[list[Any], list[Any]]:
    reactant_text, product_text = _split_reaction(reaction_smiles)
    return (
        _parse_side(chem, reactant_text, side_name="reactant"),
        _parse_side(chem, product_text, side_name="product"),
    )


def _element_composition(molecules: Sequence[Any]) -> Counter[str]:
    composition: Counter[str] = Counter()
    for molecule in molecules:
        for atom in molecule.GetAtoms():
            symbol = str(atom.GetSymbol())
            composition[symbol] += 1
            if symbol != "H":
                composition["H"] += int(atom.GetNumImplicitHs()) + int(
                    atom.GetNumExplicitHs()
                )
    return +composition


def _format_composition(composition: Counter[str]) -> dict[str, int]:
    return {symbol: int(composition[symbol]) for symbol in sorted(composition)}


def _total_formal_charge(molecules: Sequence[Any]) -> int:
    return sum(
        int(atom.GetFormalCharge())
        for molecule in molecules
        for atom in molecule.GetAtoms()
    )


def _canonical_graph_multiset(chem: Any, molecules: Sequence[Any]) -> Counter[str]:
    canonical: Counter[str] = Counter()
    for molecule in molecules:
        graph = chem.Mol(molecule)
        for atom in graph.GetAtoms():
            atom.SetAtomMapNum(0)
        canonical[
            chem.MolToSmiles(graph, canonical=True, isomericSmiles=True)
        ] += 1
    return canonical


def _require_mapper_graph_identity(
    chem: Any,
    input_reactants: Sequence[Any],
    input_products: Sequence[Any],
    mapped_reactants: Sequence[Any],
    mapped_products: Sequence[Any],
) -> None:
    for side_name, original, mapped in (
        ("reactant", input_reactants, mapped_reactants),
        ("product", input_products, mapped_products),
    ):
        original_graphs = _canonical_graph_multiset(chem, original)
        mapped_graphs = _canonical_graph_multiset(chem, mapped)
        if original_graphs != mapped_graphs:
            raise ValueError(
                "RXNMapper changed the molecular graph or encoded stereochemistry on "
                f"the {side_name} side: input={dict(original_graphs)}, "
                f"mapped={dict(mapped_graphs)}"
            )


def _require_element_conservation(
    reactants: Sequence[Any], products: Sequence[Any]
) -> Counter[str]:
    reactant_composition = _element_composition(reactants)
    product_composition = _element_composition(products)
    if reactant_composition != product_composition:
        raise ValueError(
            "Reaction is not element-conserving: "
            f"reactants={_format_composition(reactant_composition)}, "
            f"products={_format_composition(product_composition)}. "
            "Include all atoms and "
            "counter-species explicitly before atom mapping."
        )
    return reactant_composition


def _mapped_atom_records(
    molecules: Sequence[Any], *, side_name: str
) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    side_atom_index = 0
    for component_index, molecule in enumerate(molecules):
        for component_atom_index, atom in enumerate(molecule.GetAtoms()):
            map_number = int(atom.GetAtomMapNum())
            if map_number <= 0:
                raise ValueError(
                    f"Every explicit atom must have a positive map number; {side_name} "
                    f"component {component_index} atom {component_atom_index} "
                    "is unmapped"
                )
            if map_number in records:
                previous = records[map_number]
                raise ValueError(
                    f"Atom map number {map_number} is duplicated on the "
                    f"{side_name} side at side atom indices "
                    f"{previous['side_atom_index']} and {side_atom_index}"
                )
            records[map_number] = {
                "component_index": int(component_index),
                "component_atom_index": int(component_atom_index),
                "side_atom_index": int(side_atom_index),
                "element": str(atom.GetSymbol()),
                "formal_charge": int(atom.GetFormalCharge()),
                "attached_hydrogen_count": int(
                    atom.GetTotalNumHs(includeNeighbors=True)
                ),
            }
            side_atom_index += 1
    return records


def _bond_table(molecules: Sequence[Any]) -> dict[tuple[int, int], float]:
    bonds: dict[tuple[int, int], float] = {}
    for molecule in molecules:
        for bond in molecule.GetBonds():
            first = int(bond.GetBeginAtom().GetAtomMapNum())
            second = int(bond.GetEndAtom().GetAtomMapNum())
            pair = tuple(sorted((first, second)))
            order = float(bond.GetBondTypeAsDouble())
            if pair in bonds:
                raise ValueError(f"Mapped bond {pair} is duplicated")
            bonds[pair] = order
    return bonds


def _bond_record(pair: tuple[int, int], order: float) -> dict[str, Any]:
    return {
        "atom_map_a": int(pair[0]),
        "atom_map_b": int(pair[1]),
        "bond_order": float(order),
    }


def _validate_mapped_reaction(chem: Any, reaction_smiles: str) -> dict[str, Any]:
    reactants, products = _parse_reaction(chem, reaction_smiles)
    composition = _require_element_conservation(reactants, products)
    reactant_atoms = _mapped_atom_records(reactants, side_name="reactant")
    product_atoms = _mapped_atom_records(products, side_name="product")
    reactant_maps = set(reactant_atoms)
    product_maps = set(product_atoms)
    if reactant_maps != product_maps:
        raise ValueError(
            "Mapped reaction must use the exact same atom-map set on both sides; "
            f"reactant_only={sorted(reactant_maps - product_maps)}, "
            f"product_only={sorted(product_maps - reactant_maps)}"
        )
    element_mismatches = [
        map_number
        for map_number in sorted(reactant_maps)
        if reactant_atoms[map_number]["element"] != product_atoms[map_number]["element"]
    ]
    if element_mismatches:
        details = {
            map_number: (
                reactant_atoms[map_number]["element"],
                product_atoms[map_number]["element"],
            )
            for map_number in element_mismatches
        }
        raise ValueError(
            f"Atom-map element identity changed across the reaction: {details}"
        )
    reactant_charge = _total_formal_charge(reactants)
    product_charge = _total_formal_charge(products)
    if reactant_charge != product_charge:
        raise ValueError(
            "Mapped reaction does not conserve total formal charge: "
            f"reactants={reactant_charge}, products={product_charge}. Include electrons "
            "or counter-species explicitly before applying reaction changes."
        )
    atom_map = [
        {
            "map_number": map_number,
            "element": reactant_atoms[map_number]["element"],
            "reactant": dict(reactant_atoms[map_number]),
            "product": dict(product_atoms[map_number]),
        }
        for map_number in sorted(reactant_maps)
    ]
    explicit_hydrogen_count = sum(
        1 for record in atom_map if record["element"] == "H"
    )
    total_atom_count = sum(composition.values())
    unmapped_hydrogen_count = int(composition.get("H", 0)) - explicit_hydrogen_count
    return {
        "reactants": reactants,
        "products": products,
        "reactant_atoms": reactant_atoms,
        "product_atoms": product_atoms,
        "reactant_bonds": _bond_table(reactants),
        "product_bonds": _bond_table(products),
        "atom_map": atom_map,
        "atom_count": len(atom_map),
        "explicit_atom_count": len(atom_map),
        "total_atom_count": total_atom_count,
        "explicit_hydrogen_count": explicit_hydrogen_count,
        "unmapped_hydrogen_count": unmapped_hydrogen_count,
        "mapping_complete": unmapped_hydrogen_count == 0,
        "total_formal_charge": reactant_charge,
        "element_counts": _format_composition(composition),
    }


def map_reaction_atoms(request: dict[str, Any]) -> dict[str, Any]:
    mapper_class = _load_rxnmapper_class()
    if mapper_class is None:
        return unavailable(
            "RXNMapper is not installed in the selected runtime",
            install="pip install rxnmapper",
        )
    chem = _load_rdkit_chem()
    if chem is None:
        return unavailable(
            "RDKit is required to validate RXNMapper output",
            install="conda install -c conda-forge rdkit",
        )

    inputs, _method, settings = request_parts(request)
    reaction_smiles = _reaction_from_inputs(inputs, mapped=False)
    reactants, products = _parse_reaction(chem, reaction_smiles)
    _require_element_conservation(reactants, products)

    mapper, unavailable_result = _construct_rxnmapper(mapper_class)
    if unavailable_result is not None:
        return unavailable_result
    try:
        mapped_values = mapper.get_attention_guided_atom_maps([reaction_smiles])
    except Exception:
        status = _rxnmapper_resource_status()
        if not status["available"]:
            return _rxnmapper_resource_unavailable(status)
        raise
    if not isinstance(mapped_values, Sequence) or isinstance(
        mapped_values, (str, bytes)
    ):
        raise RuntimeError("RXNMapper returned an invalid result container")
    if len(mapped_values) != 1 or not isinstance(mapped_values[0], Mapping):
        raise RuntimeError(
            "RXNMapper must return exactly one mapping record per reaction"
        )
    mapped_record = mapped_values[0]
    mapped_smiles = str(
        mapped_record.get("mapped_rxn")
        or mapped_record.get("mapped_reaction_smiles")
        or ""
    ).strip()
    if not mapped_smiles:
        raise RuntimeError("RXNMapper returned no mapped reaction SMILES")
    try:
        confidence = float(mapped_record["confidence"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("RXNMapper returned no numeric confidence") from exc
    if not math.isfinite(confidence) or confidence < 0.0 or confidence > 1.0:
        raise RuntimeError(
            "RXNMapper confidence must be finite and between zero and one"
        )
    minimum_confidence = float(settings.get("minimum_confidence", 0.0))
    if not math.isfinite(minimum_confidence) or not 0.0 <= minimum_confidence <= 1.0:
        raise ValueError("minimum_confidence must be between zero and one")
    if confidence < minimum_confidence:
        raise ValueError(
            f"RXNMapper confidence {confidence:.6f} is below the explicit "
            f"minimum_confidence {minimum_confidence:.6f}"
        )

    validation = _validate_mapped_reaction(chem, mapped_smiles)
    _require_mapper_graph_identity(
        chem,
        reactants,
        products,
        validation["reactants"],
        validation["products"],
    )
    reactant_text, product_text = _split_reaction(mapped_smiles)
    result = {
        "mapped_reaction_smiles": mapped_smiles,
        "mapped_rxn": mapped_smiles,
        "reactants": reactant_text.split("."),
        "products": product_text.split("."),
        "confidence": confidence,
        "mapping_complete": validation["mapping_complete"],
        "atom_conserved": True,
        "atom_count": validation["atom_count"],
        "explicit_atom_count": validation["explicit_atom_count"],
        "total_atom_count": validation["total_atom_count"],
        "explicit_hydrogen_count": validation["explicit_hydrogen_count"],
        "unmapped_hydrogen_count": validation["unmapped_hydrogen_count"],
        "total_formal_charge": validation["total_formal_charge"],
        "element_counts": validation["element_counts"],
        "atom_map": validation["atom_map"],
    }
    warnings = []
    if not validation["mapping_complete"]:
        warnings.append(
            f"The mapped reaction contains {validation['unmapped_hydrogen_count']} "
            "implicit or atom-attached hydrogens without independent atom-map numbers. "
            "Materialize and map every hydrogen before constructing atom-order-matched "
            "3D reaction endpoints."
        )
    responder = success if validation["mapping_complete"] else partial_success
    return responder(
        result,
        backend_version=module_version("rxnmapper"),
        warnings=warnings,
        provenance={
            "mapping_engine": "RXNMapper",
            "input_reaction_smiles": reaction_smiles,
            "strict_map_uniqueness_checked": True,
            "element_conservation_checked": True,
            "formal_charge_conservation_checked": True,
            "input_output_graph_identity_checked": True,
            "explicit_hydrogen_mapping_checked": True,
        },
    )


def materialize_mapped_hydrogen_transfers(request: dict[str, Any]) -> dict[str, Any]:
    """Map every hydrogen while requiring explicit pairing for transferred H atoms."""

    chem = _load_rdkit_chem()
    if chem is None:
        return unavailable(
            "RDKit is required to materialize mapped hydrogen transfers",
            install="conda install -c conda-forge rdkit",
        )
    inputs, _method, _settings = request_parts(request)
    reaction_smiles = _reaction_from_inputs(inputs, mapped=True)
    reactants, products = _parse_reaction(chem, reaction_smiles)
    transfers = inputs.get("hydrogen_transfers")
    if not isinstance(transfers, list) or not transfers:
        raise ValueError("inputs.hydrogen_transfers must be a non-empty explicit array")

    reactants = [chem.AddHs(molecule) for molecule in reactants]
    products = [chem.AddHs(molecule) for molecule in products]

    def heavy_atoms(molecules: Sequence[Any], side: str) -> dict[int, Any]:
        result: dict[int, Any] = {}
        for molecule in molecules:
            for atom in molecule.GetAtoms():
                if atom.GetAtomicNum() == 1:
                    continue
                map_number = int(atom.GetAtomMapNum())
                if map_number <= 0:
                    raise ValueError(f"Every {side} heavy atom must already be mapped")
                if map_number in result:
                    raise ValueError(f"Heavy-atom map {map_number} is duplicated on {side}")
                result[map_number] = atom
        return result

    reactant_heavy = heavy_atoms(reactants, "reactant")
    product_heavy = heavy_atoms(products, "product")
    if set(reactant_heavy) != set(product_heavy):
        raise ValueError("Heavy-atom map sets differ across the mapped reaction")

    def hydrogen_neighbors(atom: Any) -> list[Any]:
        return sorted(
            (neighbor for neighbor in atom.GetNeighbors() if neighbor.GetAtomicNum() == 1),
            key=lambda neighbor: neighbor.GetIdx(),
        )

    normalized_transfers: list[tuple[int, int]] = []
    for index, record in enumerate(transfers):
        if not isinstance(record, Mapping):
            raise ValueError(f"hydrogen_transfers[{index}] must be a mapping")
        donor = record.get("donor_atom_map")
        acceptor = record.get("acceptor_atom_map")
        if (
            isinstance(donor, bool)
            or not isinstance(donor, int)
            or donor <= 0
            or isinstance(acceptor, bool)
            or not isinstance(acceptor, int)
            or acceptor <= 0
        ):
            raise ValueError(
                f"hydrogen_transfers[{index}] donor_atom_map and acceptor_atom_map "
                "must be positive integers"
            )
        if donor not in reactant_heavy or acceptor not in product_heavy:
            raise ValueError(
                f"hydrogen_transfers[{index}] references an unknown heavy-atom map"
            )
        normalized_transfers.append((donor, acceptor))
    if len(normalized_transfers) != len(set(normalized_transfers)):
        raise ValueError("hydrogen_transfers contains duplicate donor/acceptor records")

    reactant_available = {
        map_number: hydrogen_neighbors(atom)
        for map_number, atom in reactant_heavy.items()
    }
    product_available = {
        map_number: hydrogen_neighbors(atom)
        for map_number, atom in product_heavy.items()
    }
    next_map = max(reactant_heavy) + 1
    transfer_records = []
    for donor, acceptor in normalized_transfers:
        if not reactant_available[donor]:
            raise ValueError(f"Hydrogen donor atom map {donor} has no available hydrogen")
        if not product_available[acceptor]:
            raise ValueError(f"Hydrogen acceptor atom map {acceptor} has no product hydrogen")
        reactant_hydrogen = reactant_available[donor].pop(0)
        product_hydrogen = product_available[acceptor].pop(0)
        reactant_hydrogen.SetAtomMapNum(next_map)
        product_hydrogen.SetAtomMapNum(next_map)
        transfer_records.append(
            {
                "hydrogen_atom_map": next_map,
                "donor_atom_map": donor,
                "acceptor_atom_map": acceptor,
            }
        )
        next_map += 1

    # Hydrogens that remain attached to the same mapped heavy atom are paired
    # deterministically by local RDKit atom order. Any residual imbalance means
    # the caller omitted or misstated a transfer and is rejected.
    retained_records = []
    for heavy_map in sorted(reactant_heavy):
        reactant_hydrogens = reactant_available[heavy_map]
        product_hydrogens = product_available[heavy_map]
        if len(reactant_hydrogens) != len(product_hydrogens):
            raise ValueError(
                "Explicit hydrogen-transfer records do not account for attached-hydrogen "
                f"change at heavy atom map {heavy_map}: reactant_remaining="
                f"{len(reactant_hydrogens)}, product_remaining={len(product_hydrogens)}"
            )
        for reactant_hydrogen, product_hydrogen in zip(
            reactant_hydrogens, product_hydrogens
        ):
            reactant_hydrogen.SetAtomMapNum(next_map)
            product_hydrogen.SetAtomMapNum(next_map)
            retained_records.append(
                {"hydrogen_atom_map": next_map, "attached_heavy_atom_map": heavy_map}
            )
            next_map += 1

    reactant_text = ".".join(
        chem.MolToSmiles(
            molecule, canonical=False, isomericSmiles=True, allHsExplicit=True
        )
        for molecule in reactants
    )
    product_text = ".".join(
        chem.MolToSmiles(
            molecule, canonical=False, isomericSmiles=True, allHsExplicit=True
        )
        for molecule in products
    )
    mapped_smiles = f"{reactant_text}>>{product_text}"
    validation = _validate_mapped_reaction(chem, mapped_smiles)
    if not validation["mapping_complete"]:
        raise RuntimeError("Hydrogen materialization completed without a full atom mapping")
    result = {
        "mapped_reaction_smiles": mapped_smiles,
        "mapped_rxn": mapped_smiles,
        "reactants": reactant_text.split("."),
        "products": product_text.split("."),
        "confidence": (
            unwrap_artifact(inputs["mapped_reaction"]).get("confidence")
            if isinstance(unwrap_artifact(inputs["mapped_reaction"]), Mapping)
            else None
        ),
        "mapping_complete": True,
        "atom_conserved": True,
        "atom_count": validation["atom_count"],
        "explicit_atom_count": validation["explicit_atom_count"],
        "total_atom_count": validation["total_atom_count"],
        "explicit_hydrogen_count": validation["explicit_hydrogen_count"],
        "unmapped_hydrogen_count": 0,
        "total_formal_charge": validation["total_formal_charge"],
        "element_counts": validation["element_counts"],
        "atom_map": validation["atom_map"],
        "hydrogen_transfer_maps": transfer_records,
        "retained_hydrogen_maps": retained_records,
    }
    return success(
        result,
        backend_version=module_version("rdkit"),
        provenance={
            "pairing_source": "caller_explicit",
            "transferred_hydrogen_count": len(transfer_records),
            "retained_hydrogen_count": len(retained_records),
            "all_hydrogens_materialized_and_mapped": True,
            "mechanism_inferred": False,
        },
    )


def _directional_changes(
    reactant_bonds: Mapping[tuple[int, int], float],
    product_bonds: Mapping[tuple[int, int], float],
    reactant_atoms: Mapping[int, Mapping[str, Any]],
    product_atoms: Mapping[int, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    bond_changes = []
    for pair in sorted(set(reactant_bonds) | set(product_bonds)):
        before = float(reactant_bonds.get(pair, 0.0))
        after = float(product_bonds.get(pair, 0.0))
        if not math.isclose(before, after, rel_tol=0.0, abs_tol=1.0e-8):
            bond_changes.append(
                {
                    "atom_map_a": pair[0],
                    "atom_map_b": pair[1],
                    "from_order": before,
                    "to_order": after,
                }
            )
    charge_changes = []
    for map_number in sorted(reactant_atoms):
        before = int(reactant_atoms[map_number]["formal_charge"])
        after = int(product_atoms[map_number]["formal_charge"])
        if before != after:
            charge_changes.append(
                {
                    "atom_map": map_number,
                    "from_formal_charge": before,
                    "to_formal_charge": after,
                }
            )
    forward = {
        "source_side": "reactant",
        "target_side": "product",
        "bond_changes": bond_changes,
        "formal_charge_changes": charge_changes,
    }
    reverse = {
        "source_side": "product",
        "target_side": "reactant",
        "bond_changes": [
            {
                **change,
                "from_order": change["to_order"],
                "to_order": change["from_order"],
            }
            for change in bond_changes
        ],
        "formal_charge_changes": [
            {
                **change,
                "from_formal_charge": change["to_formal_charge"],
                "to_formal_charge": change["from_formal_charge"],
            }
            for change in charge_changes
        ],
    }
    return forward, reverse


def analyze_reaction_changes(request: dict[str, Any]) -> dict[str, Any]:
    chem = _load_rdkit_chem()
    if chem is None:
        return unavailable(
            "RDKit is required for deterministic reaction-change analysis",
            install="conda install -c conda-forge rdkit",
        )
    inputs, _method, _settings = request_parts(request)
    mapped_smiles = _reaction_from_inputs(inputs, mapped=True)
    validation = _validate_mapped_reaction(chem, mapped_smiles)
    reactant_bonds = validation["reactant_bonds"]
    product_bonds = validation["product_bonds"]
    reactant_atoms = validation["reactant_atoms"]
    product_atoms = validation["product_atoms"]

    formed_pairs = sorted(set(product_bonds) - set(reactant_bonds))
    broken_pairs = sorted(set(reactant_bonds) - set(product_bonds))
    common_pairs = sorted(set(reactant_bonds) & set(product_bonds))
    changed_pairs = [
        pair
        for pair in common_pairs
        if not math.isclose(
            reactant_bonds[pair], product_bonds[pair], rel_tol=0.0, abs_tol=1.0e-8
        )
    ]
    formal_charge_changes = [
        {
            "atom_map": map_number,
            "element": reactant_atoms[map_number]["element"],
            "reactant_formal_charge": int(reactant_atoms[map_number]["formal_charge"]),
            "product_formal_charge": int(product_atoms[map_number]["formal_charge"]),
        }
        for map_number in sorted(reactant_atoms)
        if reactant_atoms[map_number]["formal_charge"]
        != product_atoms[map_number]["formal_charge"]
    ]
    hydrogen_count_changes = [
        {
            "atom_map": map_number,
            "element": reactant_atoms[map_number]["element"],
            "reactant_attached_hydrogen_count": int(
                reactant_atoms[map_number]["attached_hydrogen_count"]
            ),
            "product_attached_hydrogen_count": int(
                product_atoms[map_number]["attached_hydrogen_count"]
            ),
            "delta": int(
                product_atoms[map_number]["attached_hydrogen_count"]
                - reactant_atoms[map_number]["attached_hydrogen_count"]
            ),
        }
        for map_number in sorted(reactant_atoms)
        if reactant_atoms[map_number]["attached_hydrogen_count"]
        != product_atoms[map_number]["attached_hydrogen_count"]
    ]
    donors = [
        {"atom_map": change["atom_map"], "hydrogens_lost": -change["delta"]}
        for change in hydrogen_count_changes
        if change["delta"] < 0
    ]
    acceptors = [
        {"atom_map": change["atom_map"], "hydrogens_gained": change["delta"]}
        for change in hydrogen_count_changes
        if change["delta"] > 0
    ]
    forward, reverse = _directional_changes(
        reactant_bonds, product_bonds, reactant_atoms, product_atoms
    )

    result = {
        "mapped_reaction_smiles": mapped_smiles,
        "mapping_complete": validation["mapping_complete"],
        "atom_conserved": True,
        "atom_count": validation["atom_count"],
        "explicit_atom_count": validation["explicit_atom_count"],
        "total_atom_count": validation["total_atom_count"],
        "explicit_hydrogen_count": validation["explicit_hydrogen_count"],
        "unmapped_hydrogen_count": validation["unmapped_hydrogen_count"],
        "total_formal_charge": validation["total_formal_charge"],
        "element_counts": validation["element_counts"],
        "atom_map": validation["atom_map"],
        "reactant_bonds": [
            _bond_record(pair, reactant_bonds[pair]) for pair in sorted(reactant_bonds)
        ],
        "product_bonds": [
            _bond_record(pair, product_bonds[pair]) for pair in sorted(product_bonds)
        ],
        "formed_bonds": [
            _bond_record(pair, product_bonds[pair]) for pair in formed_pairs
        ],
        "broken_bonds": [
            _bond_record(pair, reactant_bonds[pair]) for pair in broken_pairs
        ],
        "bond_order_changes": [
            {
                "atom_map_a": pair[0],
                "atom_map_b": pair[1],
                "reactant_bond_order": float(reactant_bonds[pair]),
                "product_bond_order": float(product_bonds[pair]),
            }
            for pair in changed_pairs
        ],
        "formal_charge_changes": formal_charge_changes,
        "hydrogen_count_changes": hydrogen_count_changes,
        "hydrogen_transfer_summary": {
            "donors": donors,
            "acceptors": acceptors,
            "balanced": sum(item["hydrogens_lost"] for item in donors)
            == sum(item["hydrogens_gained"] for item in acceptors),
            "donor_acceptor_pairing_inferred": False,
        },
        "forward_changes": forward,
        "reverse_changes": reverse,
    }
    warnings = [
        "Reaction-change analysis covers mapped connectivity, bond order, formal charge, "
        "and attached-hydrogen counts; it does not encode or validate atom/bond "
        "stereochemistry."
    ]
    if not validation["mapping_complete"]:
        warnings.append(
            f"The reaction contains {validation['unmapped_hydrogen_count']} hydrogens "
            "without independent map numbers. Hydrogen-count changes are diagnostic only; "
            "donor/acceptor atom pairing was not inferred."
        )
    responder = success if validation["mapping_complete"] else partial_success
    return responder(
        result,
        backend_version=module_version("rdkit"),
        warnings=warnings,
        provenance={
            "analysis_engine": "RDKit deterministic graph comparison",
            "strict_map_uniqueness_checked": True,
            "element_conservation_checked": True,
            "formal_charge_conservation_checked": True,
            "hydrogen_transfer_pairing_inferred": False,
            "stereochemistry_analyzed": False,
        },
    )


def _positive_map_number(value: Any, *, context: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{context} must be a positive integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be a positive integer") from exc
    if number <= 0 or isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{context} must be a positive integer")
    return number


def _structure_atom_maps(
    structure: Mapping[str, Any], inputs: Mapping[str, Any]
) -> tuple[list[int], dict[int, int]]:
    atoms = structure.get("atoms")
    if not isinstance(atoms, list) or not atoms:
        raise ValueError("AtomicStructure.atoms must be a non-empty list")
    raw_aligned = (
        inputs.get("atom_map_numbers")
        or structure.get("atom_map_numbers")
        or inputs.get("atom_map")
    )
    maps: list[int] = []
    for index, atom in enumerate(atoms):
        if not isinstance(atom, Mapping):
            raise ValueError(f"AtomicStructure.atoms[{index}] must be a mapping")
        raw = next((atom[key] for key in _MAP_NUMBER_KEYS if key in atom), None)
        if (
            raw is None
            and isinstance(raw_aligned, Sequence)
            and not isinstance(raw_aligned, (str, bytes, bytearray))
        ):
            if len(raw_aligned) != len(atoms):
                raise ValueError(
                    "atom_map_numbers must align one-to-one with structure atoms"
                )
            raw = raw_aligned[index]
        if raw is None:
            raise ValueError(
                f"AtomicStructure.atoms[{index}] requires a positive map_number, "
                "or supply "
                "an aligned inputs.atom_map_numbers list"
            )
        maps.append(
            _positive_map_number(raw, context=f"atom map at structure index {index}")
        )
    if len(set(maps)) != len(maps):
        duplicates = sorted(
            number for number, count in Counter(maps).items() if count > 1
        )
        raise ValueError(
            f"Structure atom map numbers must be unique; duplicates={duplicates}"
        )
    return maps, {map_number: index for index, map_number in enumerate(maps)}


def _coerce_bond_order(value: Any, *, context: str) -> float:
    try:
        order = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} requires a numeric bond order") from exc
    if not math.isfinite(order) or not any(
        math.isclose(order, allowed, rel_tol=0.0, abs_tol=1.0e-8)
        for allowed in _ALLOWED_BOND_ORDERS
    ):
        raise ValueError(
            f"{context} bond order must be one of {list(_ALLOWED_BOND_ORDERS)}"
        )
    return next(
        allowed
        for allowed in _ALLOWED_BOND_ORDERS
        if math.isclose(order, allowed, rel_tol=0.0, abs_tol=1.0e-8)
    )


def _map_pair(record: Any, *, context: str) -> tuple[int, int]:
    if isinstance(record, Mapping):
        first = record.get("atom_map_a", record.get("map_number_a", record.get("map1")))
        second = record.get(
            "atom_map_b", record.get("map_number_b", record.get("map2"))
        )
    elif (
        isinstance(record, Sequence)
        and not isinstance(record, (str, bytes))
        and len(record) >= 2
    ):
        first, second = record[0], record[1]
    else:
        raise ValueError(f"{context} requires two atom map numbers")
    pair = tuple(
        sorted(
            (
                _positive_map_number(first, context=f"{context}.atom_map_a"),
                _positive_map_number(second, context=f"{context}.atom_map_b"),
            )
        )
    )
    if pair[0] == pair[1]:
        raise ValueError(f"{context} cannot connect an atom map to itself")
    return pair


def _full_bond_table(value: Any, *, context: str) -> dict[tuple[int, int], float]:
    item = _unwrap_value(value)
    if not isinstance(item, Sequence) or isinstance(item, (str, bytes, bytearray)):
        raise ValueError(f"{context} must be a list of mapped bond records")
    table: dict[tuple[int, int], float] = {}
    for index, record in enumerate(item):
        pair = _map_pair(record, context=f"{context}[{index}]")
        if isinstance(record, Mapping):
            raw_order = record.get("bond_order", record.get("order"))
        else:
            raw_order = record[2] if len(record) >= 3 else None
        order = _coerce_bond_order(raw_order, context=f"{context}[{index}]")
        if order == 0.0:
            raise ValueError(
                f"{context}[{index}] is a full bond and cannot have order zero"
            )
        if pair in table:
            raise ValueError(f"{context} duplicates mapped bond {pair}")
        table[pair] = order
    return table


def _structure_bond_table(
    structure: Mapping[str, Any], index_by_map: Mapping[int, int]
) -> dict[tuple[int, int], float] | None:
    if "bonds" not in structure:
        return None
    raw_bonds = structure["bonds"]
    if not isinstance(raw_bonds, Sequence) or isinstance(raw_bonds, (str, bytes)):
        raise ValueError("AtomicStructure.bonds must be a list")
    map_by_index = {index: map_number for map_number, index in index_by_map.items()}
    table: dict[tuple[int, int], float] = {}
    for record_index, record in enumerate(raw_bonds):
        context = f"AtomicStructure.bonds[{record_index}]"
        if isinstance(record, Mapping) and any(
            key in record for key in ("atom_map_a", "map_number_a", "map1")
        ):
            pair = _map_pair(record, context=context)
            raw_order = record.get("bond_order", record.get("order"))
        elif isinstance(record, Mapping):
            first = record.get("atom_index_a", record.get("begin_atom_index"))
            second = record.get("atom_index_b", record.get("end_atom_index"))
            try:
                first_index, second_index = int(first), int(second)
                pair = tuple(
                    sorted((map_by_index[first_index], map_by_index[second_index]))
                )
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError(
                    f"{context} contains an invalid structure atom index"
                ) from exc
            raw_order = record.get("bond_order", record.get("order"))
        elif (
            isinstance(record, Sequence)
            and not isinstance(record, (str, bytes))
            and len(record) >= 3
        ):
            try:
                first_index, second_index = int(record[0]), int(record[1])
                pair = tuple(
                    sorted((map_by_index[first_index], map_by_index[second_index]))
                )
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError(
                    f"{context} contains an invalid structure atom index"
                ) from exc
            raw_order = record[2]
        else:
            raise ValueError(f"{context} must be an indexed or mapped bond record")
        order = _coerce_bond_order(raw_order, context=context)
        if order == 0.0:
            raise ValueError(f"{context} cannot have bond order zero")
        if pair in table:
            raise ValueError(f"AtomicStructure.bonds duplicates mapped bond {pair}")
        table[pair] = order
    return table


def _atom_map_metadata(value: Any) -> dict[int, Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("reaction_changes.atom_map must be a list")
    result: dict[int, Mapping[str, Any]] = {}
    for index, record in enumerate(value):
        if not isinstance(record, Mapping):
            raise ValueError(f"reaction_changes.atom_map[{index}] must be a mapping")
        map_number = _positive_map_number(
            record.get("map_number"),
            context=f"reaction_changes.atom_map[{index}].map_number",
        )
        if map_number in result:
            raise ValueError(
                f"reaction_changes.atom_map duplicates map number {map_number}"
            )
        result[map_number] = record
    return result


def _direction_payload(changes: Mapping[str, Any], direction: str) -> Mapping[str, Any]:
    key = f"{direction}_changes"
    value = changes.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"reaction_changes.{key} must be a mapping")
    return value


def _directional_bond_changes(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("directional bond_changes must be a list")
    records = []
    seen: set[tuple[int, int]] = set()
    for index, record in enumerate(value):
        if not isinstance(record, Mapping):
            raise ValueError(f"bond_changes[{index}] must be a mapping")
        pair = _map_pair(record, context=f"bond_changes[{index}]")
        if pair in seen:
            raise ValueError(f"bond_changes duplicates mapped pair {pair}")
        seen.add(pair)
        records.append(
            {
                "atom_map_a": pair[0],
                "atom_map_b": pair[1],
                "from_order": _coerce_bond_order(
                    record.get("from_order"),
                    context=f"bond_changes[{index}].from_order",
                ),
                "to_order": _coerce_bond_order(
                    record.get("to_order"), context=f"bond_changes[{index}].to_order"
                ),
            }
        )
    return records


def _directional_charge_changes(value: Any) -> list[dict[str, int]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("directional formal_charge_changes must be a list")
    records = []
    seen: set[int] = set()
    for index, record in enumerate(value):
        if not isinstance(record, Mapping):
            raise ValueError(f"formal_charge_changes[{index}] must be a mapping")
        map_number = _positive_map_number(
            record.get("atom_map"), context=f"formal_charge_changes[{index}].atom_map"
        )
        if map_number in seen:
            raise ValueError(f"formal_charge_changes duplicates atom map {map_number}")
        seen.add(map_number)
        try:
            before = int(record["from_formal_charge"])
            after = int(record["to_formal_charge"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"formal_charge_changes[{index}] requires integer from/to_formal_charge"
            ) from exc
        records.append(
            {
                "atom_map": map_number,
                "from_formal_charge": before,
                "to_formal_charge": after,
            }
        )
    return records


def _metadata_side(record: Mapping[str, Any], side: str) -> Mapping[str, Any]:
    value = record.get(side)
    if not isinstance(value, Mapping):
        raise ValueError(f"reaction_changes.atom_map record requires {side} metadata")
    return value


def _integer_state(value: Any, *, context: str, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{context} must be an integer")
    try:
        converted = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{context} must be an integer")
    if minimum is not None and converted < minimum:
        raise ValueError(f"{context} must be at least {minimum}")
    return converted


def _remove_stale_graph_identifiers(structure: dict[str, Any]) -> list[str]:
    removed: list[str] = []
    for key in _STALE_GRAPH_IDENTIFIER_KEYS:
        if key in structure:
            structure.pop(key)
            removed.append(key)
    metadata = structure.get("metadata")
    if isinstance(metadata, Mapping):
        normalized_metadata = dict(metadata)
        for key in _STALE_GRAPH_IDENTIFIER_KEYS:
            if key in normalized_metadata:
                normalized_metadata.pop(key)
                removed.append(f"metadata.{key}")
        structure["metadata"] = normalized_metadata
    return removed


def apply_reaction_changes_to_structure(request: dict[str, Any]) -> dict[str, Any]:
    inputs, method, settings = request_parts(request)
    if "structure" not in inputs or "reaction_changes" not in inputs:
        raise ValueError("inputs.structure and inputs.reaction_changes are required")
    direction = (
        str(
            settings.get("direction")
            or method.get("direction")
            or inputs.get("direction")
            or ""
        )
        .strip()
        .lower()
    )
    if direction not in {"forward", "reverse"}:
        raise ValueError("direction must be explicitly set to forward or reverse")
    if "target_charge" not in settings or "target_multiplicity" not in settings:
        raise ValueError(
            "action_settings.target_charge and target_multiplicity must be explicitly "
            "set because a reaction graph does not determine the electronic state"
        )
    requested_target_charge = _integer_state(
        settings["target_charge"], context="action_settings.target_charge"
    )
    requested_target_multiplicity = _integer_state(
        settings["target_multiplicity"],
        context="action_settings.target_multiplicity",
        minimum=1,
    )
    source_side, target_side = (
        ("reactant", "product") if direction == "forward" else ("product", "reactant")
    )

    structure = structure_dict(inputs["structure"])
    atoms_and_coordinates(structure)
    changes = _unwrap_value(inputs["reaction_changes"])
    if isinstance(changes, Mapping) and isinstance(
        changes.get("reaction_changes"), Mapping
    ):
        changes = changes["reaction_changes"]
    if not isinstance(changes, Mapping):
        raise ValueError("reaction_changes must resolve to a ReactionChangeSet mapping")
    if not bool(changes.get("mapping_complete")) or not bool(
        changes.get("atom_conserved")
    ):
        raise ValueError(
            "reaction_changes must attest complete atom mapping and "
            "element conservation"
        )
    metadata = _atom_map_metadata(changes.get("atom_map"))
    maps, index_by_map = _structure_atom_maps(structure, inputs)
    if set(maps) != set(metadata):
        raise ValueError(
            "Structure and ReactionChangeSet must contain the exact same atom-map set; "
            f"structure_only={sorted(set(maps) - set(metadata))}, "
            f"changes_only={sorted(set(metadata) - set(maps))}"
        )
    atoms = structure["atoms"]
    if "charge" not in structure or "multiplicity" not in structure:
        raise ValueError(
            "AtomicStructure must explicitly declare charge and multiplicity before "
            "reaction changes are applied"
        )
    source_structure_charge = _integer_state(
        structure["charge"], context="AtomicStructure.charge"
    )
    source_structure_multiplicity = _integer_state(
        structure["multiplicity"],
        context="AtomicStructure.multiplicity",
        minimum=1,
    )
    for map_number, atom_index in index_by_map.items():
        expected = str(metadata[map_number].get("element") or "")
        actual = str(
            atoms[atom_index].get("element") or atoms[atom_index].get("symbol") or ""
        )
        if not expected or actual.casefold() != expected.casefold():
            raise ValueError(
                f"Structure atom map {map_number} has element {actual!r}, "
                f"expected {expected!r}"
            )

    directional = _direction_payload(changes, direction)
    bond_changes = _directional_bond_changes(directional.get("bond_changes"))
    charge_changes = _directional_charge_changes(
        directional.get("formal_charge_changes")
    )
    source_key = f"{source_side}_bonds"
    target_key = f"{target_side}_bonds"
    source_bonds = _full_bond_table(
        changes.get(source_key), context=f"reaction_changes.{source_key}"
    )
    target_bonds = _full_bond_table(
        changes.get(target_key), context=f"reaction_changes.{target_key}"
    )
    structure_bonds = _structure_bond_table(structure, index_by_map)
    if structure_bonds is not None and structure_bonds != source_bonds:
        raise ValueError(
            "AtomicStructure.bonds does not match the selected "
            "ReactionChangeSet source side"
        )
    working_bonds = dict(source_bonds if structure_bonds is None else structure_bonds)
    for change in bond_changes:
        pair = (change["atom_map_a"], change["atom_map_b"])
        if pair[0] not in index_by_map or pair[1] not in index_by_map:
            raise ValueError(
                f"Bond change references atom map absent from structure: {pair}"
            )
        current_order = float(working_bonds.get(pair, 0.0))
        if not math.isclose(
            current_order, change["from_order"], rel_tol=0.0, abs_tol=1.0e-8
        ):
            raise ValueError(
                f"Bond change {pair} expected source order {change['from_order']} but "
                f"structure has {current_order}"
            )
        if change["to_order"] == 0.0:
            working_bonds.pop(pair, None)
        else:
            working_bonds[pair] = change["to_order"]
    if working_bonds != target_bonds:
        raise ValueError(
            "Applying directional bond changes did not reproduce the "
            "target-side topology"
        )

    output = deepcopy(structure)
    output_atoms = output["atoms"]
    charge_changes_by_map = {record["atom_map"]: record for record in charge_changes}
    for map_number, atom_index in index_by_map.items():
        source_metadata = _metadata_side(metadata[map_number], source_side)
        target_metadata = _metadata_side(metadata[map_number], target_side)
        source_charge = int(source_metadata["formal_charge"])
        target_charge = int(target_metadata["formal_charge"])
        current_charge = output_atoms[atom_index].get("formal_charge")
        if current_charge is not None and int(current_charge) != source_charge:
            raise ValueError(
                f"Structure atom map {map_number} formal charge "
                f"{current_charge} does not "
                f"match source-side charge {source_charge}"
            )
        declared = charge_changes_by_map.get(map_number)
        if declared is not None and (
            declared["from_formal_charge"] != source_charge
            or declared["to_formal_charge"] != target_charge
        ):
            raise ValueError(
                f"Directional charge change for atom map {map_number} "
                "conflicts with atom metadata"
            )
        if declared is None and source_charge != target_charge:
            raise ValueError(
                f"Missing directional charge change for atom map {map_number}"
            )
        output_atoms[atom_index]["map_number"] = map_number
        output_atoms[atom_index]["formal_charge"] = target_charge

    source_total_charge = sum(
        int(_metadata_side(metadata[map_number], source_side)["formal_charge"])
        for map_number in sorted(metadata)
    )
    target_total_charge = sum(
        int(_metadata_side(metadata[map_number], target_side)["formal_charge"])
        for map_number in sorted(metadata)
    )
    if source_structure_charge != source_total_charge:
        raise ValueError(
            f"AtomicStructure.charge={source_structure_charge} does not match the "
            f"source-side mapped formal charge {source_total_charge}"
        )
    if requested_target_charge != target_total_charge:
        raise ValueError(
            f"action_settings.target_charge={requested_target_charge} does not match "
            f"the target-side mapped formal charge {target_total_charge}"
        )

    output["bonds"] = [
        {
            "atom_index_a": int(index_by_map[pair[0]]),
            "atom_index_b": int(index_by_map[pair[1]]),
            "atom_map_a": int(pair[0]),
            "atom_map_b": int(pair[1]),
            "bond_order": float(working_bonds[pair]),
        }
        for pair in sorted(working_bonds)
    ]
    output["charge"] = requested_target_charge
    output["multiplicity"] = requested_target_multiplicity
    output["atom_map_numbers"] = list(maps)
    removed_graph_identifiers = _remove_stale_graph_identifiers(output)

    return success(
        {
            "structure": output,
            "direction": direction,
            "source_side": source_side,
            "target_side": target_side,
            "applied_bond_changes": bond_changes,
            "applied_formal_charge_changes": charge_changes,
            "atom_count": len(output_atoms),
            "coordinates_preserved": True,
            "source_charge": source_structure_charge,
            "source_multiplicity": source_structure_multiplicity,
            "target_charge": requested_target_charge,
            "target_multiplicity": requested_target_multiplicity,
            "removed_stale_graph_identifiers": removed_graph_identifiers,
        },
        backend_version=module_version("rdkit"),
        provenance={
            "operation": "deterministic mapped-topology edit",
            "geometry_optimization_performed": False,
            "conformer_generation_performed": False,
            "strict_map_uniqueness_checked": True,
            "element_conservation_checked": True,
            "electronic_state_explicitly_selected": True,
            "stereochemistry_transformed_or_validated": False,
        },
    )


def execute(action_id: str, backend_id: str, request: dict[str, Any]) -> dict[str, Any]:
    if action_id == "map_reaction_atoms":
        if backend_id != "rxnmapper":
            return unsupported(f"Unsupported atom-mapping backend: {backend_id}")
        return map_reaction_atoms(request)
    if action_id == "analyze_reaction_changes":
        if backend_id not in _INTERNAL_BACKENDS:
            return unsupported(f"Unsupported reaction-analysis backend: {backend_id}")
        return analyze_reaction_changes(request)
    if action_id == "materialize_mapped_hydrogen_transfers":
        if backend_id not in _INTERNAL_BACKENDS:
            return unsupported(f"Unsupported hydrogen-mapping backend: {backend_id}")
        return materialize_mapped_hydrogen_transfers(request)
    if action_id == "apply_reaction_changes_to_structure":
        if backend_id not in _INTERNAL_BACKENDS:
            return unsupported(f"Unsupported reaction-edit backend: {backend_id}")
        return apply_reaction_changes_to_structure(request)
    return unsupported(f"Unsupported reaction-mapping action: {action_id}")
