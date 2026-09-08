"""ActionSpec declarations for reaction_and_kinetics."""

from __future__ import annotations

from .._spec_builders import action as _action


ACTION_SPECS = (
    _action(
            "map_reaction_atoms",
            "reaction_and_kinetics",
            "Assign and validate atom-map identities for one explicitly supplied, atom-balanced reaction without inferring reaction changes or a mechanism.",
            "AtomMappedReaction",
            ("rxnmapper",),
            ("reaction",),
            input_description="reaction SMILES or {reactants, products}; every participating atom and counter-species must be explicit",
        ),
    _action(
            "materialize_mapped_hydrogen_transfers",
            "reaction_and_kinetics",
            "Materialize and map explicitly supplied donor-to-acceptor hydrogen transfers in an already heavy-atom-mapped reaction without inferring transfer pairing or reaction mechanism.",
            "AtomMappedReaction",
            ("internal_reaction_analysis",),
            ("mapped_reaction", "hydrogen_transfers"),
            input_description="AtomMappedReaction plus explicit donor_atom_map/acceptor_atom_map records validated against attached-hydrogen changes",
            selection_policy="internal_deterministic",
        ),
    _action(
            "analyze_reaction_changes",
            "reaction_and_kinetics",
            "Deterministically compare both sides of an atom-mapped reaction and report bond, formal-charge, and attached-hydrogen changes without pairing proton donors and acceptors by guesswork.",
            "ReactionChangeSet",
            ("internal_reaction_analysis",),
            ("mapped_reaction",),
            input_description="AtomMappedReaction returned by map_reaction_atoms",
            selection_policy="internal_deterministic",
        ),
    _action(
            "apply_reaction_changes_to_structure",
            "reaction_and_kinetics",
            "Apply one explicit forward or reverse mapped topology/charge edit to a coordinate-bearing structure while preserving atom order and coordinates and performing no optimization.",
            "AtomicStructure",
            ("internal_reaction_analysis",),
            ("structure", "reaction_changes"),
            input_description="mapped AtomicStructure and ReactionChangeSet with an explicit direction setting",
            selection_policy="internal_deterministic",
        ),
        _action(
            "search_transition_state_path",
            "reaction_and_kinetics",
            "Search for transition-state path candidates for caller-provided reactant and product SMILES through a service-managed path-search procedure. The bounded profile applies only when reactants contain at least two components and the reaction has exactly one balanced hydrogen transfer from one donor to one acceptor. That constraint counts hydrogen-transfer events and still permits other mapped bond formations, bond cleavages, or bond-order changes in the same reaction. It runs forward only with service-fixed total charge +1 and spin multiplicity 1 and requires a GPU. It preserves candidate paths and may expose transition_state_guess_artifact as an unvalidated XYZFile ArtifactRef for xyz_to_gjf. The initial guess cannot be used as scientific evidence for transition-state identity, reaction connectivity, or mechanism. Before any guess is used to support a transition-state or reaction-channel claim, independent downstream transition-state optimization must converge; Hessian/frequency analysis must confirm exactly one imaginary frequency whose normal mode matches the intended reaction coordinate, with all remaining frequencies real; and forward and reverse (bidirectional) IRC calculations must converge to and be checked against the expected reactant and product sides. "
            "execute_action returns one synchronous ActionResult and does not submit an asynchronous job; walltime follows configured policy and is not a scientific input. Action-level invalid_request, unavailable, failed, timeout, or cancelled statuses are reported through safe error/retryable fields. A timeout or cancellation of the MCP call while waiting for a result is a transport failure, not an ActionResult. Success or partial_success may retain completed endpoint pairs. Neither case triggers automatic backend fallback.",
            "TransitionStateGuessPipelineResult",
            ("transition_state_path_search",),
            ("reactants", "products"),
            ("main_force_field", "target_force_field"),
            input_description=(
                "reactants and products are non-empty, atom-balanced arrays of SMILES for one "
                "elementary reaction. Do not supply three-dimensional endpoints or a TS guess; "
                "the Action prepares both from the supplied reaction. main_force_field and "
                "target_force_field are optional selectors from the supported force-field set; "
                "reactants must contain at least two components, and the mapped reaction must "
                "have exactly one balanced hydrogen transfer from one donor to one acceptor. "
                "That constraint counts hydrogen-transfer events and still permits other mapped "
                "bond formations, bond cleavages, or bond-order changes in the same reaction. "
                "The service runs forward only with fixed total charge +1 and spin multiplicity 1; "
                "resource_limits.gpu_count must be at least 1. direction, charge, multiplicity, "
                "conformer, path-search, device, and resource-policy controls are service-managed "
                "and must not be included in inputs, method_spec, or action_settings. The stages "
                "map and analyze the supplied reaction and search candidate paths. A complete endpoint pair may expose an unvalidated "
                "transition_state_guess_artifact XYZFile ArtifactRef that can be bound to "
                "xyz_to_gjf.inputs.input_xyz_path. The scientific output is only an unvalidated "
                "TS guess/path candidate, not scientific evidence for a transition-state or "
                "reaction-channel claim. Before using a guess as such evidence, independently "
                "complete transition-state optimization, confirm exactly one imaginary frequency "
                "whose normal mode matches the intended reaction coordinate, and verify with "
                "forward and reverse (bidirectional) IRC calculations that it connects the expected reactant and "
                "product sides."
            ),
        ),
    _action(
            "locate_transition_state",
            "reaction_and_kinetics",
            "Locate one candidate transition-state structure without automatically running frequencies or IRC.",
            "AtomicStructure",
            ("pysisyphus", "sella", "gaussian"),
            ("initial_guess",),
            input_description="candidate structure; use search_reaction_path when two endpoints should drive the search",
        ),
    _action(
            "search_reaction_path",
            "reaction_and_kinetics",
            "Run an explicitly selected double-ended chain-of-states search between supplied reactant and product structures without asserting that the highest image is a validated transition state.",
            "ReactionPath",
            ("pysisyphus", "ash_neb_uma"),
            ("reactant", "product"),
            input_description="atom-order-matched reactant and product endpoint structures",
        ),
    _action(
            "scan_reaction_coordinates",
            "reaction_and_kinetics",
            "Run one relaxed one-dimensional internal-coordinate scan from a supplied structure using explicit coordinate, range, optimizer, and calculator settings.",
            "ReactionCoordinateScan",
            ("pysisyphus", "gaussian"),
            ("structure",),
            input_description="one starting structure; the scanned atom indices and range are explicit action settings",
        ),
    _action(
            "validate_reaction_path",
            "reaction_and_kinetics",
            "Check atom/state consistency, endpoint agreement, image continuity, and optional bond-change progress for an already calculated reaction path.",
            "ReactionPathValidationResult",
            ("internal_reaction_analysis",),
            ("path", "reactant", "product"),
            ("bond_changes",),
            input_description="calculated path plus the intended endpoint structures and optional forming/breaking bond definitions",
            selection_policy="internal_deterministic",
        ),
    _action(
            "analyze_reaction_coordinate",
            "reaction_and_kinetics",
            "Convert supplied path images and aligned electronic energies into a normalized reaction-coordinate profile with relative energies and highest-image candidates.",
            "ReactionCoordinateAnalysisResult",
            ("internal_reaction_analysis",),
            ("path", "energies"),
            ("bond_changes",),
            input_description="calculated path and one explicitly aligned energy value per image",
            selection_policy="internal_deterministic",
        ),
    _action(
        "extract_reaction_path_image",
        "reaction_and_kinetics",
        "Extract exactly one explicitly indexed image from a supplied reaction path without ranking it or asserting that it is a transition state.",
        "AtomicStructure",
        ("internal_reaction_analysis",),
        ("path_image_candidate",),
        input_description=(
            "path_image_candidate: object containing a ReactionPath or ReactionPath ArtifactRef, "
            "an explicit non-negative image_index, and optional pair/candidate identities"
        ),
        selection_policy="internal_deterministic",
    ),
    _action(
            "trace_intrinsic_reaction_coordinate",
            "reaction_and_kinetics",
            "Trace one explicitly selected forward or backward IRC branch from an already supplied transition-state structure.",
            "ReactionPath",
            ("pysisyphus", "gaussian"),
            ("transition_state",),
            input_description="validated candidate transition-state structure",
        ),
    _action(
            "parse_gaussian_irc_output",
            "reaction_and_kinetics",
            "Parse one completed native Gaussian IRC stdout for an explicitly selected forward or backward branch without submitting or running Gaussian. The output is a typed ReactionPath with structures, energies, units, branch completion, PES-minimum status, termination reason, and normal-termination status; abnormal logs, direction mismatches, missing points, and incomplete branches are failed or partial and are never described as a validated path.",
            "ReactionPath",
            ("internal_reaction_analysis",),
            ("gaussian_irc_log", "transition_state"),
            input_description=(
                "GaussianIRCLog ArtifactRef or workspace file, explicit transition_state/reference "
                "structure, and action_settings.direction=forward or backward; direction is never "
                "inferred from a filename, and pair/candidate/stereochemical provenance is not "
                "created by this Action."
            ),
            selection_policy="internal_deterministic",
        ),
    _action(
            "calculate_chemical_equilibrium",
            "reaction_and_kinetics",
            "Calculate an equilibrium composition/state for an explicitly supplied mechanism and thermodynamic condition.",
            "EquilibriumResult",
            ("cantera",),
            ("composition",),
            ("mechanism",),
            input_description="composition and optional mechanism Artifact",
        ),
    _action(
            "integrate_reaction_network",
            "reaction_and_kinetics",
            "Integrate one explicitly specified reaction network over time.",
            "KineticsTrajectory",
            ("scipy", "cantera"),
            ("network", "initial_state"),
            input_description="ReactionNetwork and initial concentrations/state",
        ),
    _action(
            "calculate_rate_constants",
            "reaction_and_kinetics",
            "Evaluate an explicitly supplied Arrhenius, multi-Arrhenius, pressure-dependent Arrhenius, or Chebyshev kinetics model on explicit temperature/pressure points.",
            "RateConstantResult",
            ("rmg",),
            ("kinetics_model", "temperatures_kelvin"),
            ("pressures_pa",),
            input_description="typed kinetics model plus temperature points and, for pressure-dependent models, pressure points",
        ),
    _action(
            "calculate_tunneling_correction",
            "reaction_and_kinetics",
            "Calculate Wigner or Eckart transition-state tunneling correction factors at explicit temperatures.",
            "TunnelingCorrectionResult",
            ("rmg",),
            ("temperatures_kelvin", "imaginary_frequency_cm1"),
            ("reactant_energy_kj_mol", "transition_state_energy_kj_mol", "product_energy_kj_mol"),
            input_description="temperature grid, imaginary transition-state frequency, and explicit Eckart energies when selected",
        ),
    _action(
            "solve_master_equation",
            "reaction_and_kinetics",
            "Solve an explicitly supplied gas-phase chemical master-equation model and extract pressure/temperature-dependent phenomenological rate coefficients without constructing or modifying the reaction model.",
            "MasterEquationResult",
            ("mess", "mesmer"),
            ("model_file",),
            ("companion_files", "model_relative_path"),
            input_description=(
                "native MESS input or MESMER XML model Artifact; optional explicitly listed "
                "companion files and safe staged relative paths preserve model-local references"
            ),
        ),
    _action(
            "solve_microkinetic_model",
            "reaction_and_kinetics",
            "Solve one explicitly supplied microkinetic model without constructing the reaction model for the caller.",
            "MicrokineticResult",
            ("catmap",),
            ("model",),
            input_description=(
                "typed CatMAP model fields, including explicit reaction expressions, species/site "
                "definitions, descriptor space, energetics input Artifact, thermochemistry modes, "
                "solver controls, and requested output variables"
            ),
        ),
    _action(
            "analyze_thermochemical_selectivity",
            "reaction_and_kinetics",
            "Calculate N-way thermodynamic selectivity from explicitly labeled structure ensembles, including two-label excess and delta-delta-G when applicable.",
            "ThermochemicalSelectivityResult",
            ("goodvibes",),
            ("output_files", "label_groups"),
            input_description=(
                "compatible completed quantum outputs plus a mapping from at least two caller-defined "
                "labels to exact members of output_files"
            ),
        ),
    _action(
            "analyze_free_energy_selectivity",
            "reaction_and_kinetics",
            "Calculate deterministic N-way populations from explicitly supplied labeled Gibbs free energies, including label-group partition functions and explicitly typed two-label delta-delta-G and population or enantiomeric excess.",
            "FreeEnergySelectivityResult",
            ("internal_thermochemistry",),
            ("free_energies", "energy_unit"),
            input_description=(
                "at least two records containing an explicit label and finite Gibbs free-energy "
                "value in one declared unit; repeated labels represent conformers of that group, "
                "explicit conformer ids are globally unique, and comparison_type declares semantics"
            ),
            selection_policy="internal_deterministic",
        ),
    _action(
            "analyze_energetic_span",
            "reaction_and_kinetics",
            "Calculate the Kozuch-Shaik energetic span and full-denominator forward turnover frequency from one explicitly ordered, complete catalytic-cycle Gibbs free-energy profile without constructing or inferring a mechanism.",
            "EnergeticSpanResult",
            ("internal_thermochemistry",),
            ("cycle_states", "energy_unit", "overall_reaction_free_energy"),
            input_description=(
                "an ordered alternating list [intermediate, transition_state, ...] starting "
                "with an intermediate and ending with a transition state; every state has a "
                "globally unique id and finite Gibbs free energy in one declared unit, and the "
                "explicit overall reaction free energy defines the forward cycle"
            ),
            selection_policy="internal_deterministic",
        ),
    _action(
            "analyze_reaction_free_energy_profile",
            "reaction_and_kinetics",
            "Calculate relative electronic and thermochemical energies along explicitly defined reaction pathways, including stoichiometric sums and conformer ensembles.",
            "ReactionFreeEnergyProfileResult",
            ("goodvibes",),
            ("output_files", "profile_definition_file"),
            input_description=(
                "compatible completed quantum outputs plus a caller-supplied GoodVibes PES YAML "
                "defining pathways, species membership, zero references, and units"
            ),
        ),
)
