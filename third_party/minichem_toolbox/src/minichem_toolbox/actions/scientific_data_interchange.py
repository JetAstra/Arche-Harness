"""ActionSpec declarations for scientific_data_interchange."""

from __future__ import annotations

from .._spec_builders import action as _action


ACTION_SPECS = (
    _action(
            "normalize_qcschema_molecule",
            "scientific_data_interchange",
            "Validate and normalize one supplied molecular structure into a QCSchema Molecule record without launching a calculation.",
            "QCSchemaMolecule",
            ("qcelemental",),
            ("structure",),
            input_description="AtomicStructure or existing QCSchema Molecule mapping",
            selection_policy="internal_deterministic",
        ),
    _action(
            "validate_qcschema_record",
            "scientific_data_interchange",
            "Validate one explicit QCSchema/QCArchive record type and return a normalized record or structured validation errors without executing it.",
            "QCSchemaValidationResult",
            ("qcelemental",),
            ("record",),
            input_description="JSON-compatible record mapping or workspace JSON Artifact",
            selection_policy="internal_deterministic",
        ),
    _action(
            "parse_quantum_chemistry_output",
            "scientific_data_interchange",
            "Parse explicitly selected properties from one existing quantum-chemistry output file without rerunning the calculation.",
            "ParsedQuantumChemistryResult",
            ("cclib",),
            ("output_file",),
            input_description="workspace Gaussian, ORCA, NWChem, GAMESS, Q-Chem, Molpro, or other cclib-supported output Artifact",
            selection_policy="internal_deterministic",
        ),
    _action(
            "extract_final_structure",
            "scientific_data_interchange",
            "Extract the last coordinate frame and reported electronic state from one existing quantum-chemistry output without claiming that an abnormal or unconverged calculation succeeded.",
            "AtomicStructure",
            ("cclib",),
            ("output_file",),
            ("reference_structure",),
            input_description=(
                "工作区中的 Gaussian、ORCA 或其他 cclib 支持的输出 Artifact；可选参考 "
                "AtomicStructure 或 AtomicStructureFile ArtifactRef 必须具有相同的原子数、元素顺序、"
                "电荷和多重度，通过后按原子索引复制身份 map"
            ),
            selection_policy="internal_deterministic",
        ),
    _action(
            "render_spectrum_plot",
            "scientific_data_interchange",
            "Render one already constructed SpectrumResult to an isolated image file without reparsing quantum output or changing numerical spectrum values.",
            "PlotArtifact",
            ("internal_plotting",),
            ("spectrum",),
            input_description="SpectrumResult returned by a derive_*_spectrum Action",
            selection_policy="internal_deterministic",
        ),
)
