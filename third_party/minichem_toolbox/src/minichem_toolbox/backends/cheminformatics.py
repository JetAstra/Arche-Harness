"""Atomic local cheminformatics actions."""

from __future__ import annotations

from typing import Any

from .common import module_version, request_parts, structure_dict, success, unsupported


ACTIONS = {
    "calculate_molecular_descriptors",
    "calculate_molecular_fingerprint",
    "calculate_molecular_similarity",
    "search_local_substructures",
    "enumerate_tautomers",
    "enumerate_stereoisomers",
}


def _molecule(value: Any):
    from rdkit import Chem

    item = structure_dict(value)
    smiles = item.get("smiles")
    molecule = Chem.MolFromSmiles(str(smiles)) if smiles else None
    if molecule is None and item.get("source_path"):
        path = str(item["source_path"])
        if path.lower().endswith((".sdf", ".mol")):
            molecule = Chem.MolFromMolFile(path, removeHs=False)
        elif path.lower().endswith(".pdb"):
            molecule = Chem.MolFromPDBFile(path, removeHs=False)
    if molecule is None:
        raise ValueError("RDKit cheminformatics actions require valid SMILES, SDF/MOL, or PDB input")
    return molecule


def _fingerprint(molecule, method: dict[str, Any], settings: dict[str, Any]):
    from rdkit import Chem
    from rdkit.Chem import AllChem, MACCSkeys

    fingerprint_type = str(method["fingerprint_type"]).lower().replace("-", "_")
    if fingerprint_type in {"morgan", "ecfp"}:
        radius = int(settings.get("radius", 2))
        n_bits = int(settings.get("n_bits", 2048))
        if radius < 0 or n_bits < 64 or n_bits > 65536:
            raise ValueError("Morgan fingerprint requires radius >= 0 and 64 <= n_bits <= 65536")
        return AllChem.GetMorganFingerprintAsBitVect(
            molecule,
            radius,
            nBits=n_bits,
            useChirality=bool(settings.get("use_chirality", True)),
        ), {"fingerprint_type": "morgan", "radius": radius, "n_bits": n_bits}
    if fingerprint_type in {"rdkit", "topological"}:
        n_bits = int(settings.get("n_bits", 2048))
        if n_bits < 64 or n_bits > 65536:
            raise ValueError("RDKit fingerprint requires 64 <= n_bits <= 65536")
        return Chem.RDKFingerprint(
            molecule,
            fpSize=n_bits,
            useHs=bool(settings.get("include_hydrogens", True)),
        ), {"fingerprint_type": "rdkit", "n_bits": n_bits}
    if fingerprint_type in {"maccs", "maccs_keys"}:
        value = MACCSkeys.GenMACCSKeys(molecule)
        return value, {"fingerprint_type": "maccs", "n_bits": int(value.GetNumBits())}
    raise ValueError("fingerprint_type must be morgan, rdkit, or maccs")


def _descriptors(request: dict[str, Any]) -> dict[str, Any]:
    from rdkit.Chem import Descriptors

    inputs, _method, settings = request_parts(request)
    molecule = _molecule(inputs["molecule"])
    available = {name: function for name, function in Descriptors._descList}
    default_names = [
        "MolWt",
        "ExactMolWt",
        "MolLogP",
        "TPSA",
        "NumHDonors",
        "NumHAcceptors",
        "NumRotatableBonds",
        "RingCount",
        "FractionCSP3",
        "HeavyAtomCount",
    ]
    names = list(settings.get("descriptor_names") or default_names)
    if not names or len(names) > 256:
        raise ValueError("descriptor_names must contain between 1 and 256 descriptor names")
    unknown = sorted(set(names) - set(available))
    if unknown:
        raise ValueError(f"Unknown RDKit descriptors: {unknown}")
    values = {name: float(available[name](molecule)) for name in names}
    return success(
        {"descriptors": values, "descriptor_count": len(values)},
        backend_version=module_version("rdkit"),
    )


def _fingerprint_action(request: dict[str, Any]) -> dict[str, Any]:
    inputs, method, settings = request_parts(request)
    fingerprint, metadata = _fingerprint(_molecule(inputs["molecule"]), method, settings)
    on_bits = [int(value) for value in fingerprint.GetOnBits()]
    return success(
        {
            **metadata,
            "on_bits": on_bits,
            "bit_count": len(on_bits),
        },
        backend_version=module_version("rdkit"),
    )


def _similarity(request: dict[str, Any]) -> dict[str, Any]:
    from rdkit import DataStructs

    inputs, method, settings = request_parts(request)
    left, metadata_left = _fingerprint(_molecule(inputs["molecule_a"]), method, settings)
    right, metadata_right = _fingerprint(_molecule(inputs["molecule_b"]), method, settings)
    if metadata_left != metadata_right:
        raise RuntimeError("Fingerprint metadata mismatch")
    metric = str(method["similarity_metric"]).lower()
    functions = {
        "tanimoto": DataStructs.TanimotoSimilarity,
        "dice": DataStructs.DiceSimilarity,
        "cosine": DataStructs.CosineSimilarity,
    }
    if metric not in functions:
        raise ValueError("similarity_metric must be tanimoto, dice, or cosine")
    value = float(functions[metric](left, right))
    return success(
        {"similarity": value, "metric": metric, **metadata_left},
        backend_version=module_version("rdkit"),
    )


def _substructures(request: dict[str, Any]) -> dict[str, Any]:
    from rdkit import Chem

    inputs, method, settings = request_parts(request)
    molecule = _molecule(inputs["molecule"])
    query_text = str(inputs["query"])
    query_format = str(method["query_format"]).lower()
    if query_format == "smarts":
        query = Chem.MolFromSmarts(query_text)
    elif query_format == "smiles":
        query = Chem.MolFromSmiles(query_text)
    else:
        raise ValueError("query_format must be smarts or smiles")
    if query is None:
        raise ValueError(f"Could not parse {query_format} query")
    max_matches = int(settings.get("max_matches", 1000))
    if max_matches < 1 or max_matches > 100000:
        raise ValueError("max_matches must be between 1 and 100000")
    matches = molecule.GetSubstructMatches(
        query,
        uniquify=bool(settings.get("unique", True)),
        useChirality=bool(settings.get("use_chirality", True)),
        maxMatches=max_matches,
    )
    return success(
        {
            "query": query_text,
            "query_format": query_format,
            "matches": [list(map(int, match)) for match in matches],
            "match_count": len(matches),
        },
        backend_version=module_version("rdkit"),
    )


def _tautomers(request: dict[str, Any]) -> dict[str, Any]:
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize

    inputs, _method, settings = request_parts(request)
    maximum = int(settings["max_tautomers"])
    if maximum < 1 or maximum > 10000:
        raise ValueError("max_tautomers must be between 1 and 10000")
    enumerator = rdMolStandardize.TautomerEnumerator()
    enumerator.SetMaxTautomers(maximum)
    values = enumerator.Enumerate(_molecule(inputs["molecule"]))
    smiles = sorted({Chem.MolToSmiles(value, canonical=True, isomericSmiles=True) for value in values})
    return success(
        {"molecules": [{"smiles": value} for value in smiles], "count": len(smiles), "max_tautomers": maximum},
        backend_version=module_version("rdkit"),
    )


def _stereoisomers(request: dict[str, Any]) -> dict[str, Any]:
    from rdkit import Chem
    from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions

    inputs, _method, settings = request_parts(request)
    maximum = int(settings["max_isomers"])
    if maximum < 1 or maximum > 10000:
        raise ValueError("max_isomers must be between 1 and 10000")
    options = StereoEnumerationOptions(
        onlyUnassigned=bool(settings["only_unassigned"]),
        unique=bool(settings["unique"]),
        maxIsomers=maximum,
        tryEmbedding=bool(settings.get("try_embedding", False)),
    )
    values = EnumerateStereoisomers(_molecule(inputs["molecule"]), options=options)
    smiles = sorted({Chem.MolToSmiles(value, canonical=True, isomericSmiles=True) for value in values})
    return success(
        {"molecules": [{"smiles": value} for value in smiles], "count": len(smiles), "settings": settings},
        backend_version=module_version("rdkit"),
    )


def execute(action_id: str, backend_id: str, request: dict[str, Any]) -> dict[str, Any]:
    if backend_id != "rdkit":
        return unsupported(f"Unsupported cheminformatics backend: {backend_id}")
    if action_id == "calculate_molecular_descriptors":
        return _descriptors(request)
    if action_id == "calculate_molecular_fingerprint":
        return _fingerprint_action(request)
    if action_id == "calculate_molecular_similarity":
        return _similarity(request)
    if action_id == "search_local_substructures":
        return _substructures(request)
    if action_id == "enumerate_tautomers":
        return _tautomers(request)
    if action_id == "enumerate_stereoisomers":
        return _stereoisomers(request)
    return unsupported(f"Unsupported cheminformatics action: {action_id}")
