"""ActionSpec declarations for cheminformatics."""

from __future__ import annotations

from .._spec_builders import action as _action


ACTION_SPECS = (
    _action(
            "calculate_molecular_descriptors",
            "cheminformatics",
            "Calculate an explicitly selected set of graph-based molecular descriptors without generating coordinates or running electronic structure.",
            "MolecularDescriptorResult",
            ("rdkit",),
            ("molecule",),
            input_description="molecule: SMILES, molecular structure, or compatible Artifact",
        ),
    _action(
            "calculate_molecular_fingerprint",
            "cheminformatics",
            "Calculate one explicitly selected molecular fingerprint representation.",
            "MolecularFingerprint",
            ("rdkit",),
            ("molecule",),
            input_description="molecule: SMILES or molecular structure",
        ),
    _action(
            "calculate_molecular_similarity",
            "cheminformatics",
            "Calculate one similarity value between two molecules using an explicit fingerprint and metric.",
            "MolecularSimilarityResult",
            ("rdkit",),
            ("molecule_a", "molecule_b"),
            input_description="two molecular representations plus explicit fingerprint and similarity metric",
        ),
    _action(
            "search_local_substructures",
            "cheminformatics",
            "Find atom-index matches for an explicit SMARTS or SMILES query in one supplied molecule.",
            "SubstructureMatchResult",
            ("rdkit",),
            ("molecule", "query"),
            input_description="target molecule and explicit SMARTS/SMILES query",
        ),
    _action(
            "enumerate_tautomers",
            "cheminformatics",
            "Enumerate bounded tautomeric forms without selecting a preferred tautomer for the agent.",
            "MoleculeCollection",
            ("rdkit",),
            ("molecule",),
            input_description="one molecular representation and an explicit enumeration bound",
        ),
    _action(
            "enumerate_stereoisomers",
            "cheminformatics",
            "Enumerate bounded stereoisomers under explicit uniqueness and assignment rules.",
            "MoleculeCollection",
            ("rdkit",),
            ("molecule",),
            input_description="one molecular representation and explicit stereoisomer enumeration settings",
        ),
)
