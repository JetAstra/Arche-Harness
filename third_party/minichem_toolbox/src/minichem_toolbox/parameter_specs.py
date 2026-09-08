"""Agent-visible parameter metadata for Action/backend contracts.

The execution adapters intentionally accept extensible ``method_spec`` and
``action_settings`` mappings.  Every scientifically or numerically meaningful
optional value implemented by a backend must be registered here so it is
discoverable instead of surviving as a hidden implementation default.
"""

from __future__ import annotations

from typing import Any, Mapping


ParameterMetadata = Mapping[str, Any]


ORCA_DENSITY_DEFAULT_MAXCORE_MB = 2000


RESOURCE_LIMIT_PARAMETER_SPECS: dict[str, ParameterMetadata] = {
    "memory_mb": {
        "description": "Requested memory budget in megabytes when the backend supports it.",
        "default": 4096,
        "type": "integer",
        "minimum": 128,
        "impact": (
            "More memory may enable larger calculations, while excessive per-process memory "
                "combined with many CPU processes can exceed the task budget. The configured policy rejects "
            "single or concurrent requests above the declared per-task maximum."
        ),
    },
    "cpu_cores": {
        "description": "Requested CPU process/thread count for the selected backend.",
        "default": 1,
        "type": "integer",
        "minimum": 1,
        "impact": (
            "More cores may reduce elapsed time, but scaling is backend- and system-dependent "
            "and may increase total memory use. The configured policy rejects single or concurrent "
            "requests above the declared per-task maximum."
        ),
    },
    "gpu_count": {
        "description": "Requested GPU count for a backend that supports GPU execution.",
        "default": 0,
        "type": "integer",
        "minimum": 0,
        "impact": (
            "A positive value requests accelerator resources but does not change the selected "
            "scientific method. Zero disables GPU visibility for managed execution."
        ),
    },
}


BACKEND_PARAMETER_SPECS: dict[
    str, dict[str, dict[str, ParameterMetadata]]
] = {
    "orca": {
        "export_electron_density_grid": {
            "action_settings.grid_points_per_axis": {
                "description": (
                    "Number of ORCA cube grid points along each Cartesian axis when "
                    "exporting an MDCI/CCSD density."
                ),
                "default": 300,
                "type": "integer",
                "minimum": 20,
                "maximum": 400,
                "impact": (
                    "Larger values reduce density discretization and isosurface-area error "
                    "but increase time, memory, and cube size approximately with the cube "
                    "of this value. Fixed point counts also give coarser voxels for larger "
                    "molecular boxes, so quantitative work should verify convergence."
                ),
            },
            "action_settings.electron_count_tolerance_percent": {
                "description": (
                    "Maximum allowed relative difference between the integrated cube "
                    "density and the source density's expected electron count."
                ),
                "default": 0.2,
                "type": "number",
                "minimum": 0.001,
                "maximum": 10.0,
                "impact": (
                    "A smaller tolerance rejects or warns about more discretization loss; "
                    "a larger tolerance accepts coarser cubes but may allow isosurface-area "
                    "bias large enough to change close method rankings."
                ),
            },
            "action_settings.strict_electron_count_validation": {
                "description": (
                    "Whether an electron-count error above the selected tolerance makes "
                    "the cube export fail instead of returning it with a warning."
                ),
                "default": False,
                "type": "boolean",
                "impact": (
                    "Enabling strict validation prevents downstream use of a demonstrably "
                    "under-resolved cube. Disabling it preserves exploratory workflows but "
                    "requires the caller to treat the warning as numerical uncertainty."
                ),
            },
        },
    },
    "multiwfn": {
        "calculate_electron_isodensity_surface": {
            "action_settings.grid_spacing_bohr": {
                "description": (
                    "Real-space spacing, in bohr, used by Multiwfn when constructing the "
                    "electron-isodensity surface grid."
                ),
                "type": "number",
                "minimum": 0.02,
                "maximum": 1.0,
                "impact": (
                    "A smaller spacing produces a finer surface grid and usually reduces "
                    "discretization error, but increases runtime and memory use. Quantitative "
                    "surface comparisons should verify convergence with at least one finer value."
                ),
            },
            "action_settings.grid_spacing_angstrom": {
                "description": (
                    "Deprecated compatibility alias for grid_spacing_bohr. Despite its historical "
                    "name, its numeric value is interpreted in bohr so existing calls preserve "
                    "their original numerical behavior. Omit it in new requests."
                ),
                "default": None,
                "type": "number | null",
                "minimum": 0.02,
                "maximum": 1.0,
                "deprecated": True,
                "replacement": "action_settings.grid_spacing_bohr",
                "impact": (
                    "When supplied alone, it selects exactly the same bohr spacing as the canonical "
                    "field and emits a deprecation warning. Supplying both fields is rejected."
                ),
            },
        },
        "analyze_non_covalent_interactions": {
            "action_settings.grid_spacing_bohr": {
                "description": "Uniform X/Y/Z spacing in bohr for the wavefunction-based NCI/RDG grid.",
                "type": "number",
                "minimum": 0.05,
                "maximum": 1.0,
                "impact": (
                    "Smaller spacing resolves interaction regions more finely but increases "
                    "runtime, memory, and artifact size approximately cubically."
                ),
            },
            "action_settings.grid_extension_bohr": {
                "description": "Padding in bohr beyond the molecular coordinate bounds for the NCI/RDG grid.",
                "type": "number",
                "minimum": 0.5,
                "maximum": 10.0,
                "impact": (
                    "Larger padding captures more diffuse density but adds mostly empty grid "
                    "points and can materially increase cost."
                ),
            },
            "action_settings.rdg_cutoff": {
                "description": (
                    "RDG threshold used only for structured descriptive counts over the "
                    "complete exported scatter table; it does not alter the raw grids."
                ),
                "type": "number",
                "minimum": 0.0,
                "maximum": 5.0,
                "impact": (
                    "Changing this threshold changes which sampled points are summarized as "
                    "low-RDG; candidate comparisons require the same value."
                ),
            },
            "action_settings.signed_density_abs_cutoff_au": {
                "description": (
                    "Absolute sign(lambda2)rho cutoff in a.u. used only for structured "
                    "low-RDG descriptive counts."
                ),
                "type": "number",
                "minimum": 0.000001,
                "maximum": 1.0,
                "impact": (
                    "Changing the window changes attractive/near-zero/repulsive point counts; "
                    "candidate comparisons require identical cutoffs and converged grids."
                ),
            },
            "action_settings.max_grid_points": {
                "description": (
                    "Caller-selected hard upper limit for the estimated NCI/RDG grid point "
                    "count; no scientific default is inferred."
                ),
                "type": "integer",
                "minimum": 1,
                "maximum": 10000000,
                "impact": (
                    "The Action rejects the request before Multiwfn grid generation when "
                    "the coordinate-box estimate exceeds this limit. The catalog ceiling "
                    "prevents an unbounded opt-in."
                ),
            },
        },
    },
}


BACKEND_FIXED_PARAMETER_SPECS: dict[
    str, dict[str, dict[str, ParameterMetadata]]
] = {
    "orca": {
        "export_electron_density_grid": {
            "backend_runtime.cube_boundary_selection": {
                "description": (
                    "ORCA orca_plot determines the density-cube bounding box from the "
                    "calculated molecular density and its internal padding policy."
                ),
                "reason": (
                    "The installed interactive orca_plot adapter currently exposes grid "
                    "resolution but not an independently typed bounding-box override."
                ),
            },
            "backend_runtime.cube_export_parallelism": {
                "description": (
                    "The installed ORCA orca_plot MDCI cube exporter runs as one process; "
                    "resource_limits.cpu_cores does not parallelize this export stage."
                ),
                "reason": (
                    "orca_plot does not expose a validated parallel cube-export control. "
                    "Use sufficient walltime, and parallelize independent molecule exports "
                    "as separate jobs when server capacity permits."
                ),
            },
        },
    },
}


_MISSING = object()


def _register_parameter(
    backend_id: str,
    action_ids: str | tuple[str, ...],
    field_path: str,
    *,
    description: str,
    impact: str,
    default: Any = _MISSING,
    **constraints: Any,
) -> None:
    actions = (action_ids,) if isinstance(action_ids, str) else action_ids
    for action_id in actions:
        metadata: dict[str, Any] = {
            "description": description,
            "impact": impact,
            **constraints,
        }
        if default is not _MISSING:
            metadata["default"] = default
        BACKEND_PARAMETER_SPECS.setdefault(backend_id, {}).setdefault(
            action_id, {}
        )[field_path] = metadata


def _register_fixed(
    backend_id: str,
    action_ids: str | tuple[str, ...],
    field_path: str,
    *,
    description: str,
    reason: str,
) -> None:
    actions = (action_ids,) if isinstance(action_ids, str) else action_ids
    for action_id in actions:
        BACKEND_FIXED_PARAMETER_SPECS.setdefault(backend_id, {}).setdefault(
            action_id, {}
        )[field_path] = {"description": description, "reason": reason}


# RDKit cheminformatics optional controls.
for _action_id in ("calculate_molecular_fingerprint", "calculate_molecular_similarity"):
    _register_parameter(
        "rdkit", _action_id, "action_settings.radius",
        description="Morgan/ECFP neighborhood radius in bonds.", default=2,
        type="integer", minimum=0,
        impact="A larger radius encodes a wider chemical environment and can change similarity while increasing fingerprint construction cost.",
    )
    _register_parameter(
        "rdkit", _action_id, "action_settings.n_bits",
        description="Bit-vector length for Morgan or RDKit fingerprints.", default=2048,
        type="integer", minimum=64, maximum=65536,
        impact="More bits reduce hash collisions and increase memory/output size; MACCS uses its fixed key length instead.",
    )
    _register_parameter(
        "rdkit", _action_id, "action_settings.use_chirality",
        description="Whether fingerprint construction distinguishes stereochemical environments.", default=True,
        type="boolean",
        impact="Enabling chirality can distinguish stereoisomers; disabling it makes fingerprints invariant to encoded stereochemistry.",
    )
    _register_parameter(
        "rdkit", _action_id, "action_settings.include_hydrogens",
        description="Whether the RDKit topological fingerprint includes explicit hydrogen paths.", default=True,
        type="boolean",
        impact="This only affects the RDKit/topological fingerprint branch and can change the set of activated bits.",
    )

_register_parameter(
    "rdkit", "calculate_molecular_descriptors", "action_settings.descriptor_names",
    description="Explicit RDKit descriptor names to calculate.",
    default=["MolWt", "ExactMolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds", "RingCount", "FractionCSP3", "HeavyAtomCount"],
    type="array[string]", maximum_items=256,
    impact="Selecting more descriptors increases output breadth and modestly increases calculation time; values not requested are not computed.",
)
for _field_path, _default, _description, _impact in (
    ("action_settings.max_matches", 1000, "Maximum number of local substructure matches returned.", "A larger bound can reveal more matches but increases matching time and output size."),
    ("action_settings.unique", True, "Whether symmetry-equivalent local substructure matches are deduplicated.", "Disabling uniqueness can return repeated symmetry-equivalent atom mappings."),
    ("action_settings.use_chirality", True, "Whether local substructure matching enforces query stereochemistry.", "Enabling chirality makes stereochemical labels part of the match criterion."),
):
    _register_parameter(
        "rdkit", "search_local_substructures", _field_path,
        description=_description, default=_default, impact=_impact,
        type="boolean" if isinstance(_default, bool) else "integer",
    )
_register_parameter(
    "rdkit", "enumerate_stereoisomers", "action_settings.try_embedding",
    description="Whether RDKit attempts a 3D embedding while enumerating stereoisomers.",
    default=False, type="boolean",
    impact="Enabling embedding can filter or validate stereoisomers geometrically but adds stochastic geometry-generation cost.",
)


# 网络数据源的重试参数仍由调用方选择；超时由服务政策固定，因此不进入公开目录。
_PUBCHEM_ACTIONS = (
    "search_compounds", "resolve_chemical_identity", "retrieve_compound_properties",
    "retrieve_compound_structure", "search_similar_compounds", "search_substructures",
)
for _field_path, _default, _description, _impact, _extra in (
    ("action_settings.max_retries", 1, "Maximum retry count for a transient PubChem request failure.", "More retries improve resilience but increase worst-case latency and duplicate remote requests.", {"type": "integer", "minimum": 0}),
    ("action_settings.retry_backoff_seconds", 1.0, "Initial delay between PubChem retry attempts.", "Longer backoff reduces pressure on a failing service but increases latency.", {"type": "number", "minimum": 0}),
    ("action_settings.minimum_request_interval_seconds", 0.25, "Cross-worker minimum interval between PubChem HTTP requests.", "A longer interval reduces rate-limit risk but lowers throughput.", {"type": "number", "minimum": 0}),
    ("action_settings.max_poll_attempts", 15, "Maximum PubChem asynchronous-search polling attempts.", "More attempts allow longer remote searches but increase worst-case elapsed time.", {"type": "integer", "minimum": 1}),
    ("action_settings.poll_interval_seconds", 2.0, "Delay between PubChem asynchronous-search polls.", "Longer intervals reduce polling traffic but delay result collection.", {"type": "number", "minimum": 0}),
):
    _register_parameter("pubchem", _PUBCHEM_ACTIONS, _field_path, description=_description, default=_default, impact=_impact, **_extra)
_register_parameter(
    "pubchem", "search_compounds", "action_settings.namespace",
    description="PubChem identifier namespace used to interpret the query.", default="name",
    type="string", allowed_values=["name", "cid", "smiles", "inchi", "inchikey"],
    impact="Changing the namespace changes how the same query text is resolved.",
)
_register_parameter(
    "pubchem", "search_compounds", "action_settings.max_records",
    description="Maximum PubChem compound records returned.", default=10,
    type="integer", minimum=1,
    impact="A larger limit returns more candidates and increases network and output cost.",
)
_register_parameter(
    "pubchem", "resolve_chemical_identity", "action_settings.max_records",
    description="Maximum candidate identities considered before uniqueness checks.", default=10,
    type="integer", minimum=1,
    impact="A larger limit can expose ambiguous matches but increases response processing.",
)
for _backend_id, _action_id, _default_records in (
    ("rcsb_pdb", "search_protein_structures", 10),
    ("materials_project", "search_materials", 20),
    ("catalysis_hub", "search_catalysis_records", 20),
    ("nist_webbook", "lookup_nist_webbook_species", 10),
):
    _register_parameter(
        _backend_id, _action_id, "action_settings.max_records",
        description="Maximum number of remote records returned.", default=_default_records,
        type="integer", minimum=1,
        impact="A larger limit broadens the returned result set and increases network and output size.",
    )
_register_parameter(
    "materials_project", "search_materials", "action_settings.fields",
    description="Explicit Materials Project summary fields requested from the API.",
    default=["material_id", "formula_pretty", "energy_above_hull", "band_gap", "is_stable", "symmetry", "volume", "density"],
    type="array[string]",
    impact="Requesting more fields increases response size and may require permissions for additional API fields.",
)


# Docking backend-specific optional controls.
for _field_path, _description, _impact in (
    ("method_spec.cnn_scoring", "Optional GNINA CNN scoring mode.", "Changing the mode changes whether and when CNN rescoring/refinement is applied."),
    ("method_spec.scoring_function", "Optional explicit GNINA empirical scoring function.", "Changing the scoring function changes pose ranking and reported affinities."),
    ("action_settings.gpu_device", "GPU device identifier used when GNINA use_gpu is true.", "Selecting a different device changes resource placement but not the requested scientific model."),
    ("action_settings.cpu", "GNINA CPU worker count passed to the executable.", "More CPU workers may reduce runtime but increase host contention."),
    ("action_settings.seed", "GNINA stochastic search seed.", "Changing the seed changes stochastic docking trajectories; a fixed seed improves reproducibility."),
):
    _register_parameter("gnina", "dock_ligand", _field_path, description=_description, default=None, impact=_impact)


# Structure and system-construction hidden defaults.
_register_parameter(
    "pymatgen", "enumerate_surface_slabs", "action_settings.symmetrize",
    description="Whether pymatgen attempts to symmetrize generated slab terminations.", default=False,
    type="boolean", impact="Enabling symmetrization can reduce asymmetric artifacts but may alter terminations or reject slabs.",
)
_register_parameter(
    "rdkit", "generate_3d_structure", "action_settings.max_attempts",
    description="Maximum RDKit embedding attempts.", default=1000, type="integer", minimum=1,
    impact="More attempts improve the chance of embedding difficult molecules but increase runtime.",
)
_register_parameter(
    "rdkit_etkdg", "generate_conformer_ensemble", "action_settings.prune_rms_threshold_angstrom",
    description="RMSD threshold used to prune newly generated conformers; a negative value disables pruning.", default=-1.0,
    type="number", impact="Larger positive thresholds retain fewer, more geometrically distinct conformers and reduce downstream cost.",
)
for _field_path, _default, _description, _impact in (
    ("action_settings.ionic_strength_molar", 0.0, "Target ionic strength for OpenMM solvation.", "Increasing ionic strength adds counterions/salt and changes the simulated chemical environment."),
    ("action_settings.positive_ion", "Na+", "Positive ion species used by OpenMM solvation.", "Changing the ion changes system composition and force-field requirements."),
    ("action_settings.negative_ion", "Cl-", "Negative ion species used by OpenMM solvation.", "Changing the ion changes system composition and force-field requirements."),
):
    _register_parameter("openmm_builder", "solvate_molecular_system", _field_path, description=_description, default=_default, impact=_impact)
_register_parameter(
    "packmol", "solvate_molecular_system", "action_settings.tolerance_angstrom",
    description="Minimum Packmol interatomic placement tolerance.", default=2.0, type="number", minimum=0,
    impact="A larger tolerance reduces close contacts but makes packing harder and may require a larger box.",
)


# Molecular-dynamics execution and trajectory-analysis optional controls.
_OPENMM_ACTIONS = (
    "minimize_system_energy", "propagate_dynamics", "calculate_force_field_energy",
    "calculate_force_field_forces", "decompose_force_field_energy",
)
for _field_path, _description, _impact in (
    ("method_spec.platform", "Optional explicit OpenMM compute platform name.", "Selecting CPU, CUDA, OpenCL, or another installed platform changes resource placement and may change floating-point reproducibility."),
    ("method_spec.platform_properties", "Optional OpenMM platform-property mapping.", "Properties such as precision or device index change accelerator behavior, performance, and sometimes numerical reproducibility."),
):
    _register_parameter("openmm", _OPENMM_ACTIONS, _field_path, description=_description, default=None, impact=_impact)
for _field_path, _default, _description, _impact, _extra in (
    ("action_settings.friction_per_ps", 1.0, "Langevin collision/friction coefficient for OpenMM NVT/NPT propagation.", "Larger friction produces stronger thermostat coupling and can alter dynamical properties.", {"type": "number", "minimum": 0}),
    ("action_settings.initialize_velocities", True, "Whether OpenMM initializes velocities when no saved state is supplied.", "Disabling initialization requires usable velocities from the supplied state; enabling it samples a Maxwell distribution.", {"type": "boolean"}),
    ("action_settings.random_seed", 0, "Seed used for OpenMM velocity initialization.", "Changing the seed changes initial velocities and the resulting trajectory.", {"type": "integer"}),
    ("action_settings.pressure_bar", None, "Target pressure for OpenMM NPT propagation.", "Changing pressure changes the sampled NPT ensemble; it is ignored outside NPT.", {"type": "number"}),
):
    _register_parameter("openmm", "propagate_dynamics", _field_path, description=_description, default=_default, impact=_impact, **_extra)

_GROMACS_ACTIONS = ("minimize_system_energy", "propagate_dynamics")
_register_parameter(
    "gromacs", _GROMACS_ACTIONS, "method_spec.cutoff_scheme",
    description="GROMACS neighbor-search and nonbonded cutoff scheme.", default="Verlet",
    type="string", impact="Changing the cutoff scheme changes neighbor-list behavior and compatibility with other nonbonded settings.",
)
for _field_path, _default, _description, _impact in (
    ("action_settings.continuation", False, "Whether GROMACS dynamics continues from an existing state.", "Continuation changes initialization and thermostat/barostat handling and should match the supplied checkpoint."),
    ("action_settings.temperature_coupling_ps", 1.0, "Thermostat coupling time constant in ps.", "Smaller values couple temperature more strongly and can distort physical dynamics."),
    ("action_settings.pressure_coupling_ps", 5.0, "Barostat coupling time constant in ps.", "Smaller values couple pressure more strongly and can cause unstable volume fluctuations."),
    ("action_settings.compressibility_bar_inverse", 4.5e-5, "Isothermal compressibility used by the GROMACS barostat.", "Changing compressibility changes the volume response to pressure coupling."),
    ("action_settings.pressure_bar", None, "Target pressure required for GROMACS NPT propagation.", "Changing it changes the sampled NPT thermodynamic state."),
    ("action_settings.random_seed", None, "Explicit GROMACS velocity-generation seed required when generate_velocities=true.", "Changing it changes initial velocities and the resulting trajectory."),
    ("action_settings.temperature_coupling_groups", None, "Explicit GROMACS index groups coupled to the thermostat.", "Changing group definitions changes which atoms share thermostat baths and can materially alter dynamics."),
):
    _register_parameter("gromacs", "propagate_dynamics", _field_path, description=_description, default=_default, type="boolean" if isinstance(_default, bool) else "number", impact=_impact)

_LAMMPS_ACTIONS = ("minimize_system_energy", "propagate_dynamics")
for _field_path, _default, _description, _impact in (
    ("method_spec.units", "real", "LAMMPS unit system.", "Changing units changes the interpretation of every numeric force-field and dynamics parameter."),
    ("method_spec.atom_style", "full", "LAMMPS atom style used to read the data file.", "Changing atom style changes required per-atom fields and must match the supplied data file."),
    ("method_spec.neighbor_skin", 2.0, "LAMMPS neighbor-list skin distance in the selected unit system.", "A larger skin reduces rebuild frequency but increases neighbor-list size and memory."),
):
    _register_parameter("lammps", _LAMMPS_ACTIONS, _field_path, description=_description, default=_default, impact=_impact)
_register_parameter(
    "lammps", "minimize_system_energy", "action_settings.max_evaluations",
    description="Maximum LAMMPS force/energy evaluations during minimization.",
    default="10 * max_iterations", type="integer",
    impact="A larger bound may allow difficult minimizations to converge but increases worst-case cost.",
)
for _field_path, _default, _description, _impact in (
    ("action_settings.temperature_damping_fs", 100.0, "LAMMPS thermostat damping time in fs.", "Smaller damping couples temperature more strongly and can perturb dynamics."),
    ("action_settings.pressure_damping_fs", 1000.0, "LAMMPS barostat damping time in fs.", "Smaller damping couples pressure more strongly and can destabilize cell fluctuations."),
):
    _register_parameter("lammps", "propagate_dynamics", _field_path, description=_description, default=_default, type="number", minimum=0, impact=_impact)

_HOOMD_ACTIONS = (
    "minimize_system_energy", "propagate_dynamics", "calculate_force_field_energy",
    "calculate_force_field_forces",
)
for _field_path, _default, _description, _impact in (
    ("method_spec.bond_potential", "harmonic", "Optional HOOMD bond-potential family when bonds are present.", "Changing the potential changes bonded energetics and requires compatible parameters."),
    ("method_spec.bond_parameters", None, "Typed HOOMD bond-parameter mapping.", "Changing parameters changes bonded forces and energies; it is required when bonded interactions are present."),
):
    _register_parameter("hoomd", _HOOMD_ACTIONS, _field_path, description=_description, default=_default, impact=_impact)
for _action_id in ("minimize_system_energy", "calculate_force_field_energy", "calculate_force_field_forces"):
    _register_parameter(
        "hoomd", _action_id, "action_settings.random_seed",
        description="HOOMD simulation seed used by stochastic components.", default=0,
        type="integer", impact="Changing the seed changes stochastic state where applicable; fixed seeds improve reproducibility.",
    )
_register_parameter(
    "hoomd", "minimize_system_energy", "action_settings.angular_momentum_tolerance",
    description="FIRE angular-momentum convergence tolerance; defaults to force_tolerance.",
    default="force_tolerance", type="number",
    impact="A smaller tolerance imposes stricter rotational convergence and can increase iterations.",
)
_register_parameter(
    "hoomd", "minimize_system_energy", "action_settings.steps_per_check",
    description="Number of HOOMD FIRE integration steps between convergence checks.", default=10,
    type="integer", minimum=1,
    impact="Larger values reduce checking overhead but can overshoot the earliest converged step.",
)
_register_parameter(
    "hoomd", "propagate_dynamics", "action_settings.thermostat_tau",
    description="HOOMD thermostat coupling time in reduced time units.", default=1.0,
    type="number", minimum=0,
    impact="Smaller values produce stronger temperature coupling and may alter dynamics.",
)

_MDANALYSIS_FRAME_ACTIONS = (
    "calculate_trajectory_rmsd", "calculate_radius_of_gyration",
    "calculate_radial_distribution", "calculate_mean_squared_displacement",
    "calculate_dihedral_distribution",
)
for _field_path, _default, _description, _impact in (
    ("action_settings.start_frame", 0, "First zero-based trajectory frame included.", "Increasing it discards more initial trajectory frames."),
    ("action_settings.stop_frame", -1, "Exclusive final trajectory frame; -1 selects the trajectory end.", "Changing it changes the analyzed time window."),
    ("action_settings.frame_stride", 1, "Trajectory frame stride.", "A larger stride reduces cost and temporal resolution."),
):
    _register_parameter("mdanalysis", _MDANALYSIS_FRAME_ACTIONS, _field_path, description=_description, default=_default, type="integer", impact=_impact)
_register_parameter(
    "mdanalysis", "calculate_trajectory_rmsd", "inputs.reference",
    description="Optional separate reference structure or trajectory for RMSD alignment.", default=None,
    impact="Supplying a different reference changes the absolute RMSD values and alignment target.",
)
_register_parameter(
    "mdanalysis", "calculate_mean_squared_displacement", "action_settings.fft",
    description="Whether MDAnalysis uses its FFT-accelerated MSD implementation.", default=True,
    type="boolean", impact="FFT mode is faster for long trajectories but imposes the backend's FFT algorithm assumptions.",
)
_register_parameter(
    "mdanalysis", "calculate_hydrogen_bonds", "inputs.between_selections",
    description="Optional selection-pair restrictions for hydrogen-bond analysis.", default=None,
    impact="Supplying pairs restricts reported hydrogen bonds to the named groups.",
)
_register_parameter(
    "mdtraj", "calculate_trajectory_rmsd", "inputs.reference",
    description="Optional separate MDTraj reference trajectory.", default=None,
    impact="Changing the reference changes RMSD alignment and values.",
)
_register_parameter(
    "mdtraj", "calculate_contacts", "action_settings.soft_min_beta",
    description="Inverse softness parameter for MDTraj soft-min contact distances.", default=20.0,
    type="number", minimum=0,
    impact="Larger beta approaches a hard minimum; smaller beta averages more atom-pair distances. It is used only when soft_min is enabled.",
)

for _field_path, _default, _description, _impact in (
    ("action_settings.stride", 1, "PLUMED PRINT stride in evaluated driver frames.", "A larger stride writes fewer collective-variable samples."),
    ("action_settings.box_angstrom", None, "Optional periodic box supplied when absent from the trajectory.", "Changing the box changes periodic-image handling of collective variables."),
    ("action_settings.timestep_ps", None, "Optional trajectory timestep metadata in ps.", "Changing it changes the reported time axis but not coordinates."),
    ("action_settings.trajectory_stride", None, "Optional stride of stored frames relative to the original simulation.", "Changing it changes the reconstructed time mapping."),
):
    _register_parameter("plumed", "evaluate_collective_variables", _field_path, description=_description, default=_default, impact=_impact)

_PYMBAR_ACTIONS = (
    "estimate_free_energy_difference", "estimate_thermodynamic_expectations",
    "calculate_potential_of_mean_force", "analyze_free_energy_convergence",
)
for _field_path, _default, _description, _impact in (
    ("action_settings.initialize", "zeros", "PyMBAR initial free-energy estimate strategy.", "Changing initialization can affect convergence speed but should not change a converged solution."),
    ("action_settings.random_seed", 0, "Seed used by PyMBAR bootstrap sampling.", "Changing it changes bootstrap realizations and uncertainty estimates."),
):
    _register_parameter("pymbar", _PYMBAR_ACTIONS, _field_path, description=_description, default=_default, impact=_impact)
for _action_id in ("estimate_free_energy_difference", "analyze_free_energy_convergence"):
    _register_parameter(
        "pymbar", _action_id, "action_settings.bootstrap_samples",
        description="Number of PyMBAR bootstrap replicates when bootstrap uncertainty is selected.", default=0,
        type="integer", minimum=0,
        impact="More replicates improve Monte Carlo stability of uncertainty estimates and increase cost approximately linearly.",
    )
for _action_id in ("estimate_free_energy_difference", "calculate_potential_of_mean_force", "analyze_free_energy_convergence"):
    _register_parameter(
        "pymbar", _action_id, "action_settings.temperature_kelvin",
        description="Optional temperature used to convert reduced free energies to kJ/mol.", default=None,
        type="number", minimum=0,
        impact="It changes only dimensional energy conversion when reduced potentials are already supplied.",
    )
_register_parameter(
    "pymbar", "calculate_potential_of_mean_force", "action_settings.reference_coordinate",
    description="Explicit PMF reference coordinate when reference='specified'.", default=None,
    type="number", impact="Changing the reference adds a constant offset anchored to a different occupied bin.",
)
_register_parameter(
    "pymbar", "calculate_potential_of_mean_force", "action_settings.coordinate_unit",
    description="Label describing the collective-variable coordinate unit.", default="unspecified",
    type="string", impact="This labels output coordinates and does not rescale supplied values.",
)
_register_parameter(
    "pymbar", "estimate_free_energy_difference", "action_settings.include_theta",
    description="Whether PyMBAR returns the asymptotic covariance matrix Theta.", default=False,
    type="boolean", impact="Enabling it increases output size and exposes covariance diagnostics without changing the estimate.",
)


# Molecular electronic-structure and spectrum optional controls.
_XTB_ACTIONS = (
    "calculate_energy", "calculate_forces", "calculate_hessian", "optimize_geometry",
    "calculate_dipole_moment", "calculate_atomic_charges", "calculate_bond_orders",
)
for _field_path, _description, _impact in (
    ("method_spec.charge", "Optional xTB molecular charge override; otherwise structure.charge is used.", "Changing charge changes electron count and the electronic state."),
    ("method_spec.unpaired_electrons", "Optional xTB unpaired-electron count override; otherwise multiplicity-1 is used.", "Changing it changes spin occupation and can produce a different electronic state."),
):
    _register_parameter("xtb", _XTB_ACTIONS, _field_path, description=_description, default=None, impact=_impact)

_PYSCF_ACTIONS = (
    "calculate_energy", "calculate_forces", "calculate_hessian", "calculate_dipole_moment",
    "calculate_atomic_charges", "calculate_orbitals", "calculate_excited_states",
)
for _field_path, _description, _impact in (
    ("method_spec.charge", "Optional PySCF molecular charge override; otherwise structure.charge is used.", "Changing charge changes electron count and electronic state."),
    ("method_spec.spin", "Optional PySCF 2S spin value; otherwise multiplicity-1 is used.", "Changing spin changes alpha/beta occupations and electronic state."),
    ("method_spec.functional", "PySCF DFT functional alias used for RKS/UKS methods.", "Changing it changes the density-functional approximation."),
    ("method_spec.xc", "Alternative PySCF exchange-correlation functional field.", "Changing it changes the density-functional approximation."),
):
    _register_parameter("pyscf", _PYSCF_ACTIONS, _field_path, description=_description, default=None, impact=_impact)
_register_parameter(
    "pyscf", _PYSCF_ACTIONS, "action_settings.scf_convergence",
    description="PySCF SCF energy convergence tolerance.", default=1e-9,
    type="number", minimum=0,
    impact="A smaller value requests tighter SCF convergence and usually increases iterations and numerical reliability.",
)
_register_parameter(
    "pyscf", "calculate_atomic_charges", "method_spec.population_analysis",
    description="PySCF population-analysis definition.", default="mulliken",
    type="string", impact="Changing the population definition changes reported atomic charges without changing the wavefunction.",
)
_register_parameter(
    "pyscf", "calculate_orbitals", "action_settings.include_coefficients",
    description="Whether PySCF includes the full molecular-orbital coefficient matrix.", default=False,
    type="boolean", impact="Enabling it substantially increases output size but does not change the calculation.",
)

_NWCHEM_ACTIONS = (
    "calculate_energy", "calculate_forces", "calculate_hessian",
    "calculate_dipole_moment", "calculate_atomic_charges",
)
for _field_path, _description, _impact in (
    ("method_spec.charge", "Optional NWChem charge override; otherwise structure metadata is used.", "Changing charge changes electron count and electronic state."),
    ("method_spec.multiplicity", "Optional NWChem spin multiplicity override.", "Changing multiplicity changes spin occupation and electronic state."),
    ("method_spec.functional", "NWChem XC functional required when method=dft.", "Changing it changes the density-functional approximation."),
    ("method_spec.reference", "Optional NWChem RHF/UHF/ROHF reference.", "Changing the reference changes spin treatment and may affect convergence and results."),
):
    _register_parameter("nwchem", _NWCHEM_ACTIONS, _field_path, description=_description, default=None, impact=_impact)

_OPENMOLCAS_ACTIONS = (
    "calculate_energy", "calculate_dipole_moment", "calculate_atomic_charges",
    "calculate_orbitals",
)
for _field_path, _description, _impact in (
    ("method_spec.charge", "Optional OpenMolcas molecular charge override.", "Changing charge changes electron count and electronic state."),
    ("method_spec.multiplicity", "Optional OpenMolcas spin multiplicity override.", "Changing multiplicity changes electronic state."),
    ("method_spec.functional", "OpenMolcas/Libxc functional required for DFT.", "Changing it changes the density-functional approximation."),
):
    _register_parameter("openmolcas", _OPENMOLCAS_ACTIONS, _field_path, description=_description, default=None, impact=_impact)

_ASE_HESSIAN_BACKENDS = ("mace", "chgnet", "deepmd", "ase_emt")
for _backend_id in _ASE_HESSIAN_BACKENDS:
    # The field is required in the current public contract, but explicit metadata records
    # the implementation default used by direct backend calls and explains its effect.
    _register_parameter(
        _backend_id, "calculate_hessian", "action_settings.displacement_angstrom",
        description="Central finite-difference displacement used to construct the Hessian.", default=0.01,
        type="number", minimum=0,
        impact="Too large a displacement introduces anharmonic error; too small amplifies force noise and roundoff. Convergence should be checked for noisy potentials.",
    )
_register_parameter(
    "mace", ("calculate_energy", "calculate_forces", "calculate_hessian", "optimize_geometry"),
    "method_spec.default_dtype",
    description="MACE floating-point dtype.", default="float64", type="string",
    allowed_values=["float32", "float64"],
    impact="float64 improves numerical precision at higher memory/runtime cost; float32 is faster on many accelerators.",
)
for _backend_id in ("nequip", "allegro"):
    _actions = (
        "calculate_periodic_energy", "calculate_periodic_forces",
        "calculate_periodic_stress", "relax_periodic_structure",
    )
    _register_parameter(
        _backend_id, _actions, "method_spec.neighborlist_backend",
        description="Neighbor-list implementation used by compatible MLIP inference.", default="matscipy",
        type="string", impact="Changing the implementation primarily affects compatibility and performance; equivalent implementations should preserve the model definition.",
    )
_register_parameter(
    "deepmd", (
        "calculate_energy", "calculate_forces", "calculate_hessian", "optimize_geometry",
        "calculate_periodic_energy", "calculate_periodic_forces", "calculate_periodic_stress",
        "relax_periodic_structure",
    ), "method_spec.frame_parameters",
    description="Optional additional DeePMD per-frame model parameters.", default=None,
    impact="Supplying values changes model conditioning and must match the checkpoint's expected parameter schema.",
)

_PSI4_ACTIONS = (
    "calculate_energy", "calculate_hessian", "calculate_dipole_moment",
    "calculate_atomic_charges", "calculate_orbitals",
)
_register_parameter(
    "psi4", _PSI4_ACTIONS, "method_spec.options",
    description="Optional explicit Psi4 option mapping forwarded to the selected calculation.", default=None,
    type="object", impact="Options can change convergence algorithms, integral approximations, reference states, and numerical results; they must be chosen consistently with method and basis.",
)

_ORCA_STANDARD_ACTIONS = (
    "calculate_energy", "calculate_forces", "calculate_hessian", "optimize_geometry",
    "calculate_dipole_moment", "calculate_atomic_charges", "calculate_orbitals",
    "calculate_bond_orders", "calculate_excited_states",
)
for _field_path, _description, _impact in (
    (
        "method_spec.basis",
        "Explicit ORCA orbital basis for ordinary methods; omit it or use method_default/auto for built-in 3c composite methods.",
        "Changing the orbital basis changes basis-set completeness and cost. Built-in 3c methods define their own basis and reject a concrete override.",
    ),
    ("method_spec.dispersion", "Optional ORCA dispersion-correction keyword.", "Adding or changing dispersion alters energies and gradients and can change optimized structures."),
    ("method_spec.solvation_model", "Optional ORCA implicit-solvation model.", "Enabling solvation changes the Hamiltonian/environment and resulting energies, densities, and structures."),
    ("method_spec.solvent", "Solvent name used when an ORCA solvation model is selected.", "Changing solvent changes dielectric/solvation parameters and computed properties."),
    ("method_spec.charge", "Optional ORCA molecular charge override.", "Changing charge changes electron count and electronic state."),
    ("method_spec.multiplicity", "Optional ORCA spin multiplicity override.", "Changing multiplicity changes electronic state and reference occupation."),
):
    _register_parameter("orca", _ORCA_STANDARD_ACTIONS, _field_path, description=_description, default=None, impact=_impact)
_ORCA_DENSITY_ACTION = "calculate_correlated_electron_density"
for _field_path, _description, _impact, _extra in (
    (
        "method_spec.basis",
        "Explicit ORCA orbital basis for ordinary methods; omit it or use method_default/auto for built-in 3c composite methods.",
        "Changing the orbital basis changes density accuracy and cost. Built-in 3c methods define their own basis and reject a concrete override.",
        {},
    ),
    ("method_spec.auxiliary_basis", "Optional ORCA auxiliary/C basis for MP2 or double-hybrid density calculations.", "Changing it changes density-fitting accuracy and cost.", {}),
    ("method_spec.dispersion", "Optional ORCA dispersion-correction keyword.", "Dispersion typically changes energy and gradients; exact density influence depends on the selected ORCA method implementation.", {}),
    ("method_spec.frozen_core", "Whether correlated density calculations use the frozen-core approximation.", "Disabling frozen core correlates more electrons and increases cost; it can change correlated density.", {"type": "boolean"}),
    ("method_spec.pmodel", "Whether ORCA PModel is enabled for the density calculation.", "Changing PModel changes the double-hybrid/MP2 model details used by ORCA.", {"type": "boolean"}),
    ("method_spec.charge", "Optional molecular charge override.", "Changing charge changes electron count and density.", {"type": "integer"}),
    ("method_spec.multiplicity", "Optional spin multiplicity override.", "Changing multiplicity changes electronic state and density.", {"type": "integer", "minimum": 1}),
):
    _register_parameter(
        "orca", _ORCA_DENSITY_ACTION, _field_path,
        description=_description, default=None, impact=_impact, **_extra,
    )
_register_parameter(
    "orca", _ORCA_DENSITY_ACTION, "action_settings.stability_analysis",
    description="Whether ORCA performs an SCF wavefunction-stability analysis before the density calculation.",
    type="boolean",
    impact=(
        "Enabling it can detect and restart an unstable SCF solution at additional cost; "
        "disabling it skips that validation."
    ),
)
_register_parameter(
    "orca", _ORCA_DENSITY_ACTION, "resource_limits.memory_mb",
    description=(
        "Total memory budget in MB across all requested ORCA MPI processes. The adapter "
        "sets %maxcore to floor(memory_mb / cpu_cores), with a 128 MB floor."
    ),
    default=None,
    type="integer | null",
    minimum=128,
    default_behavior=(
        f"When null, the adapter allocates {ORCA_DENSITY_DEFAULT_MAXCORE_MB} MB per "
        "process, so total memory is that value multiplied by the effective cpu_cores."
    ),
    derived_backend_parameter={
        "name": "orca_maxcore_mb_per_process",
        "formula": "max(128, floor(memory_mb / effective_cpu_cores))",
        "null_memory_formula": (
            f"{ORCA_DENSITY_DEFAULT_MAXCORE_MB} MB per process"
        ),
    },
    impact=(
        "Increasing the total budget raises ORCA %maxcore at fixed cpu_cores and can "
        "prevent correlated-method out-of-memory failures. Increasing cpu_cores without "
        "also increasing memory_mb lowers %maxcore per process."
    ),
)

_GAUSSIAN_ACTIONS = (
    "calculate_energy", "calculate_hessian", "optimize_geometry",
    "calculate_dipole_moment", "calculate_bond_orders",
    "calculate_excited_states", "locate_transition_state",
    "scan_reaction_coordinates", "trace_intrinsic_reaction_coordinate",
)
for _field_path, _description, _impact in (
    ("method_spec.dispersion", "Optional bounded Gaussian empirical-dispersion model (GD2, GD3, or GD3BJ).", "Changing dispersion changes energy and gradients."),
    ("method_spec.solvation_model", "Optional Gaussian implicit-solvation model.", "Enabling or changing solvation changes the computed environment and properties."),
    ("method_spec.solvent", "Gaussian solvent name used by the solvation model.", "Changing solvent changes solvation parameters and results."),
    ("method_spec.integration_grid", "Optional bounded Gaussian DFT numerical integration grid.", "A finer integration grid can improve numerical stability and reproducibility while increasing cost."),
    ("method_spec.charge", "Optional molecular charge override.", "Changing charge changes electron count and electronic state."),
    ("method_spec.multiplicity", "Optional spin multiplicity override.", "Changing multiplicity changes electronic state."),
):
    _register_parameter("gaussian", _GAUSSIAN_ACTIONS, _field_path, description=_description, default=None, impact=_impact)

_register_parameter(
    "format_conversion",
    "smiles2sdf",
    "inputs.smiles_list",
    description="格式转换 Action 接受的非空 SMILES 字符串列表。",
    impact="列表中的每个 SMILES 独立生成一个三维分子记录；无法解析或嵌入的项目会跳过并报告消息。",
    type="array[string]",
    minimum_items=1,
    maximum_items=32,
)

_register_parameter(
    "format_conversion",
    "sdf_to_gjf",
    "inputs.sdf_path",
    description="单记录 SDFFile 或 SDFRecordFile ArtifactRef，或工作区内的 SDF 路径。",
    impact="单记录 SDFRecordFile 可直接交给 sdf_to_gjf；合并 SDFFile 只有在恰好一条有效记录时可用。",
    type="SDFFile | SDFRecordFile ArtifactRef | workspace-relative .sdf path",
    accepted_artifact_semantic_types=["SDFFile", "SDFRecordFile"],
)

_register_parameter(
    "format_conversion",
    "sdf_to_gjf",
    "inputs.route_parameters",
    description="sdf_to_gjf 使用的显式 Gaussian route 参数。",
    impact="修改 route 会改变待提交 Gaussian 输入的计算设置；单行内容仍须通过 native deck 预检。",
    type="string | object",
    default=(
        "p UB3LYP/6-31+G(d,p) empiricaldispersion=gd3bj opt=(calcfc) freq "
        "stable=opt integral=ultrafine scrf=(smd,solvent=EthylEthanoate)"
    ),
)
for _field_path, _description, _type, _default, _minimum, _maximum in (
    (
        "inputs.title",
        "写入 SDF 转换结果的单行 Gaussian 标题。",
        "string",
        "Generated from SDF",
        None,
        None,
    ),
):
    _constraints = {"type": _type}
    if _minimum is not None:
        _constraints["minimum"] = _minimum
    if _maximum is not None:
        _constraints["maximum"] = _maximum
    _register_parameter(
        "format_conversion",
        "sdf_to_gjf",
        _field_path,
        description=_description,
        impact="修改该字段只影响生成文件的受限元数据或资源指令，不会运行 Gaussian。",
        default=_default,
        **_constraints,
    )

_register_parameter(
    "format_conversion",
    "xyz_to_gjf",
    "inputs.input_xyz_path",
    description="单帧 XYZFile 或 AtomicStructureFile ArtifactRef，或工作区内的 .xyz 路径；转换前会拒绝空、截断和多帧输入。",
    impact="输入必须恰好包含一个完整 XYZ frame；规范 comment 中的电荷和多重度可被转换器安全继承。",
    type="XYZFile | AtomicStructureFile ArtifactRef | workspace-relative .xyz path",
    accepted_artifact_semantic_types=["XYZFile", "AtomicStructureFile"],
)

_XYZ_INPUT_PARAMETERS = (
    ("inputs.route_section", "xyz_to_gjf 使用的单行 Gaussian route 内容。", "string | object", "#p opt freq B3LYP/6-31G(d) scrf=(iefpcm,solvent=acetone) Temperature=303.15", None, None),
    ("inputs.charge", "XYZ 转换写入的分子总电荷。", "integer | null", 0, -100, 100),
    ("inputs.multiplicity", "XYZ 转换写入的自旋多重度。", "integer | null", 1, 1, 100),
    ("inputs.title", "写入 XYZ 转换结果的单行 Gaussian 标题。", "string", "Converted from XYZ file", None, None),
    ("inputs.chk_file", "XYZ 转换使用的 checkpoint 文件名，不允许路径；省略时服务按输出文件名生成。", "string | null", None, None, None),
    ("inputs.fix_atom_count_line", "是否按服务默认值修复 XYZ 首行的小数原子数；不会回写输入文件。", "boolean", True, None, None),
    ("inputs.write_back_fixed_xyz", "兼容字段；MCP 禁止回写输入，因此只能为 false。", "boolean", False, None, None),
)
for _field_path, _description, _type, _default, _minimum, _maximum in _XYZ_INPUT_PARAMETERS:
    _constraints = {"type": _type}
    if _minimum is not None:
        _constraints["minimum"] = _minimum
    if _maximum is not None:
        _constraints["maximum"] = _maximum
    _register_parameter(
        "format_conversion",
        "xyz_to_gjf",
        _field_path,
        description=_description,
        impact="修改该字段只影响 XYZ 到 Gaussian 输入文件的受限内容，不会运行 Gaussian；输入文件不会被回写。",
        default=_default,
        **_constraints,
    )

_register_parameter(
    "gaussian", _GAUSSIAN_ACTIONS, "action_settings.scf_convergence",
    description=(
        "Gaussian SCF convergence selection; use 'default' to omit the SCF route keyword "
        "and retain the program's native method-dependent default."
    ),
    type="string", allowed_values=["default", "Loose", "Tight", "VeryTight"],
    impact=(
        "Tighter convergence can improve numerical reliability at additional cost; selecting "
        "default preserves a published route that did not override Gaussian's SCF behavior."
    ),
)

_register_parameter(
    "gaussian", "calculate_excited_states", "method_spec.excited_state_method",
    description="Vertical excited-state approximation applied to the selected SCF/DFT reference.",
    type="string", allowed_values=["tddft", "tda"],
    impact=(
        "TDA omits the de-excitation block and can change excitation energies and stability; "
        "TDDFT retains the full response equations."
    ),
)
for _field_path, _description, _impact, _constraints in (
    (
        "action_settings.number_of_states",
        "Number of lowest vertical excited states requested from Gaussian.",
        "More states extend spectral coverage and increase response-equation cost.",
        {"type": "integer", "minimum": 1, "maximum": 100},
    ),
    (
        "action_settings.spin_symmetry",
        "Spin manifold requested for a restricted-reference vertical excitation calculation.",
        "Singlet and triplet response roots represent different electronic states and cannot be interchanged.",
        {"type": "string", "allowed_values": ["singlet", "triplet"]},
    ),
):
    _register_parameter(
        "gaussian", "calculate_excited_states", _field_path,
        description=_description, impact=_impact, **_constraints,
    )

_register_parameter(
    "gaussian", "calculate_bond_orders", "method_spec.bond_order_definition",
    description="Population-analysis definition used for reported atom-pair bond indices.",
    type="string", allowed_values=["wiberg_nao"],
    impact=(
        "Wiberg indices in the natural atomic orbital basis are basis- and wavefunction-dependent "
        "population descriptors rather than observables."
    ),
)
_register_parameter(
    "gaussian", "calculate_bond_orders", "action_settings.minimum_bond_order",
    description="Inclusive lower threshold for atom-pair Wiberg indices retained in the bonds list.",
    type="number", minimum=0.0, maximum=10.0,
    impact=(
        "Raising the threshold shortens the reported pair list but does not alter the full matrix "
        "or the underlying Gaussian calculation."
    ),
)

for _field_path, _description, _impact, _constraints in (
    (
        "action_settings.coordinate_type",
        "Single internal coordinate to scan.",
        "Bond, angle, and dihedral coordinates impose different geometric constraints.",
        {"type": "string", "allowed_values": ["bond", "angle", "dihedral"]},
    ),
    (
        "action_settings.atom_indices",
        "Zero-based unique atom indices defining the selected internal coordinate.",
        "Changing atom order or identity changes the physical coordinate being scanned.",
        {"type": "array[integer]"},
    ),
    (
        "action_settings.start_value",
        "Explicit value imposed at the first relaxed scan point.",
        "The start value sets the first constrained geometry and may affect which local branch is followed.",
        {"type": "number"},
    ),
    (
        "action_settings.end_value",
        "Explicit value imposed at the final relaxed scan point.",
        "The end value sets scan coverage and, together with steps, the increment.",
        {"type": "number"},
    ),
    (
        "action_settings.value_unit",
        "Unit of start_value, end_value, and returned coordinate values.",
        "Bond scans require angstrom; angular scans accept degree or radian.",
        {"type": "string", "allowed_values": ["angstrom", "degree", "radian"]},
    ),
    (
        "action_settings.steps",
        "Number of equal increments between the two endpoints; the requested point count is steps plus one.",
        "More steps improve profile resolution and require more constrained optimizations.",
        {"type": "integer", "minimum": 1, "maximum": 1000},
    ),
    (
        "action_settings.optimization_convergence",
        "Gaussian convergence keyword used for each relaxed scan optimization, or default to omit it.",
        "Tighter convergence increases cost and can reduce profile noise.",
        {"type": "string", "allowed_values": ["default", "Loose", "Tight", "VeryTight"]},
    ),
    (
        "action_settings.max_cycles",
        "Maximum optimization cycles allowed at each scan point.",
        "A larger limit permits difficult points to converge and increases worst-case runtime.",
        {"type": "integer", "minimum": 1, "maximum": 1000},
    ),
    (
        "action_settings.force_constant_strategy",
        "Optional Gaussian force-constant strategy for relaxed scan optimization.",
        "CalcFC or CalcAll can stabilize difficult scans at increasing Hessian cost.",
        {"type": "string", "allowed_values": ["default", "calcfc", "calcall"]},
    ),
    (
        "action_settings.disable_symmetry",
        "Whether Gaussian molecular symmetry is disabled during the scan.",
        "Disabling symmetry prevents reorientation/symmetry constraints from changing the requested coordinate branch.",
        {"type": "boolean"},
    ),
):
    _register_parameter(
        "gaussian", "scan_reaction_coordinates", _field_path,
        description=_description, impact=_impact, **_constraints,
    )

_register_parameter(
    "gaussian", "calculate_hessian", "action_settings.temperature_kelvin",
    description="Temperature printed and used by Gaussian's frequency thermochemistry analysis.",
    default=None, type="number", minimum=0.01,
    impact=(
        "Temperature does not change the electronic Hessian, but it changes the thermal and "
        "free-energy corrections available in the auditable Gaussian frequency output."
    ),
)
for _field_path, _description, _impact, _constraints in (
    (
        "action_settings.no_eigen_test",
        "Whether Gaussian skips its initial curvature-sign test during TS optimization.",
        "Skipping the test permits refinement of a supplied TS guess but increases reliance on later frequency validation.",
        {"type": "boolean"},
    ),
    (
        "action_settings.calculate_all_force_constants",
        "Whether Gaussian recalculates exact force constants at every TS optimization point (CalcAll).",
        "CalcAll is substantially more expensive but can stabilize difficult TS refinement.",
        {"type": "boolean"},
    ),
    (
        "action_settings.initial_force_constants",
        "Optional bounded initial force-constant strategy for Gaussian TS optimization.",
        "CalcFC computes one exact Hessian at the starting geometry and then updates it during optimization; it is less expensive than CalcAll.",
        {"default": None, "type": "string | null", "allowed_values": ["calcfc"]},
    ),
    (
        "action_settings.max_step",
        "Gaussian Opt MaxStep value in its native hundredths-of-a-bohr trust-radius scale.",
        "A smaller value limits each geometry step and can stabilize TS refinement at additional cost.",
        {"type": "integer", "minimum": 1, "maximum": 100},
    ),
    (
        "action_settings.max_cycles",
        "Maximum Gaussian TS optimization cycles.",
        "More cycles permit a longer convergence attempt and increase worst-case runtime.",
        {"default": None, "type": "integer", "minimum": 1, "maximum": 1000},
    ),
):
    _register_parameter(
        "gaussian", "locate_transition_state", _field_path,
        description=_description, impact=_impact, **_constraints,
    )

for _field_path, _description, _impact, _constraints in (
    (
        "action_settings.direction",
        "Single IRC branch to trace from the supplied transition state.",
        "Forward and backward are separate atomic calculations and may terminate at different minima.",
        {"type": "string", "allowed_values": ["forward", "backward"]},
    ),
    (
        "action_settings.initial_force_constants",
        "Bounded Gaussian initial force-constant strategy; this adapter supports CalcFC.",
        "CalcFC computes the starting Hessian required to leave the transition state along the reaction path.",
        {"type": "string", "allowed_values": ["calcfc"]},
    ),
    (
        "action_settings.integrator",
        "Bounded Gaussian IRC integration algorithm; this adapter supports LQA.",
        "The integrator changes numerical path following and may affect convergence and point placement.",
        {"type": "string", "allowed_values": ["lqa"]},
    ),
    (
        "action_settings.max_points",
        "Maximum number of IRC points on the selected branch.",
        "More points trace farther from the TS and increase gradient/Hessian work.",
        {"type": "integer", "minimum": 1, "maximum": 1000},
    ),
    (
        "action_settings.recalc",
        "Interval in IRC points between exact Hessian recalculations.",
        "A smaller interval can stabilize the LQA path at substantial additional cost.",
        {"type": "integer", "minimum": 1, "maximum": 1000},
    ),
):
    _register_parameter(
        "gaussian", "trace_intrinsic_reaction_coordinate", _field_path,
        description=_description, impact=_impact, **_constraints,
    )

_GAMESS_ACTIONS = ("calculate_energy", "optimize_geometry", "calculate_dipole_moment")
for _field_path, _default, _description, _impact in (
    ("method_spec.charge", None, "Optional GAMESS molecular charge override.", "Changing charge changes electron count and electronic state."),
    ("method_spec.multiplicity", None, "Optional GAMESS spin multiplicity override.", "Changing multiplicity changes electronic state."),
    ("method_spec.dfttyp", None, "Optional GAMESS DFT functional keyword.", "Supplying a DFT type changes the electronic method from pure SCF to the selected density functional."),
    ("method_spec.guess", "HUCKEL", "GAMESS initial orbital guess method.", "Changing the guess can alter SCF convergence behavior and, for difficult systems, the converged solution."),
):
    _register_parameter("gamess", _GAMESS_ACTIONS, _field_path, description=_description, default=_default, impact=_impact)

for _field_path, _default, _description, _impact in (
    ("action_settings.min_wavenumber_cm1", "min(frequencies)-5*fwhm, clipped at 0", "Lower IR spectrum grid boundary.", "Extending the range increases output coverage and point spacing at fixed point count."),
    ("action_settings.max_wavenumber_cm1", "max(frequencies)+5*fwhm", "Upper IR spectrum grid boundary.", "Extending the range increases output coverage and point spacing at fixed point count."),
    ("action_settings.points", 4000, "Number of IR spectrum grid points.", "More points improve sampled line-shape resolution and increase output size."),
):
    _register_parameter("internal_spectroscopy", "derive_ir_spectrum", _field_path, description=_description, default=_default, impact=_impact)


# ARCHE atomic reaction, UMA/NEB, structure extraction, and extended spectra.
_register_parameter(
    "cclib", "extract_final_structure", "action_settings.require_normal_termination",
    description=(
        "Whether a source lacking cclib's normal-termination marker must be returned as "
        "partial_success rather than success."
    ),
    type="boolean",
    impact="This changes result status, never which coordinate frame is extracted.",
)
_register_parameter(
    "cclib", "extract_final_structure", "inputs.reference_structure",
    description=(
        "可选参考 AtomicStructure、AtomicStructureFile ArtifactRef 或工作区结构路径；必须与 cclib 最后一帧保持原子数、元素顺序、"
        "总电荷和自旋多重度一致，匹配后按原子索引复制 atom_map_number 身份。"
    ),
    type="AtomicStructure | AtomicStructureFile ArtifactRef | workspace-relative structure path",
    accepted_artifact_semantic_types=["AtomicStructure", "AtomicStructureFile"],
    impact=(
        "提供参考结构可保留上游原子 map 身份；任一身份核对字段不一致都会拒绝结果，"
        "不会进行图同构猜测或原子重排。"
    ),
)
_register_parameter(
    "rxnmapper", "map_reaction_atoms", "action_settings.minimum_confidence",
    description="Minimum accepted RXNMapper confidence after strict graph validation.",
    type="number", minimum=0.0, maximum=1.0,
    impact="A higher threshold rejects more mappings; it does not create an alternative mapping.",
)
_register_parameter(
    "internal_reaction_analysis", "apply_reaction_changes_to_structure",
    "action_settings.direction",
    description="Explicit direction in which the mapped topology and formal-charge edits are applied.",
    type="string", allowed_values=["forward", "reverse"],
    impact="Reversing direction swaps the required source and target graph without moving atoms.",
)
_register_parameter(
    "internal_reaction_analysis", "apply_reaction_changes_to_structure",
    "action_settings.target_charge",
    description="Explicit total molecular charge of the generated endpoint structure.",
    type="integer",
    impact=(
        "The value is checked against target-side mapped formal charges; the backend "
        "never infers or silently changes the requested electronic state."
    ),
)
_register_parameter(
    "internal_reaction_analysis", "extract_reaction_path_image",
    "inputs.path_image_candidate",
    description=(
        "包含完整 ReactionPath 或 ReactionPath ArtifactRef、显式 image_index 以及可选 "
        "pair_id/candidate_id 的单个路径图像选择记录。"
    ),
    type="object",
    impact=(
        "只提取所给索引的路径图像；不会按能量排序、自动选择最高图像或断言其为过渡态。"
    ),
)
_register_parameter(
    "internal_reaction_analysis",
    "parse_gaussian_irc_output",
    "inputs.gaussian_irc_log",
    description=(
        "已完成 native Gaussian IRC stdout 的 GaussianIRCLog ArtifactRef 或工作区文件；"
        "不根据文件名推断方向，也不提交或执行 Gaussian。"
    ),
    type="GaussianIRCLog ArtifactRef | workspace-relative Gaussian stdout path",
    accepted_artifact_semantic_types=["GaussianIRCLog"],
    accepted_forms=[
        "完整 GaussianIRCLog ArtifactRef",
        "工作区内已完成 Gaussian stdout 文件",
    ],
    impact=(
        "更换日志会更换被解析的路径点、能量和终止事实；日志内容不会触发新的计算。"
    ),
)
_register_parameter(
    "internal_reaction_analysis",
    "parse_gaussian_irc_output",
    "inputs.transition_state",
    description=(
        "显式 transition_state/reference AtomicStructure；用于原子顺序、身份和路径首点，"
        "不会从 Gaussian 日志猜测结构标签或候选身份。"
    ),
    type="AtomicStructure | AtomicStructureFile | XYZFile ArtifactRef | workspace-relative structure path",
    accepted_artifact_semantic_types=["AtomicStructure", "AtomicStructureFile", "XYZFile"],
    accepted_forms=[
        "完整 AtomicStructure mapping",
        "单结构 ArtifactRef",
        "工作区内结构文件",
    ],
    impact=(
        "更换参考结构会改变路径点的原子身份复制和下游端点校验；不会移动原子或优化结构。"
    ),
)
_register_parameter(
    "internal_reaction_analysis",
    "parse_gaussian_irc_output",
    "action_settings.direction",
    description="显式选择要解析的 Gaussian IRC forward 或 backward 分支。",
    type="string",
    allowed_values=["forward", "backward"],
    impact=(
        "方向必须由调用方提供；选择错误方向会失败，不能根据文件名或其他标签自动改选。"
    ),
)
_register_parameter(
    "internal_reaction_analysis", "apply_reaction_changes_to_structure",
    "action_settings.target_multiplicity",
    description="Explicit spin multiplicity of the generated endpoint structure.",
    type="integer", minimum=1,
    impact=(
        "Reaction connectivity alone does not determine spin, so selecting a different "
        "multiplicity changes the electronic state used by downstream calculations."
    ),
)

for _field_path, _description, _impact, _constraints in (
    (
        "action_settings.random_seed",
        "Seed for deterministic rigid-body orientation sampling.",
        "Changing it changes sampled orientations while preserving the explicit contacts.",
        {"type": "integer"},
    ),
    (
        "action_settings.candidate_count",
        "Number of reactive-complex candidates requested.",
        "More candidates broaden orientation sampling and increase pairwise collision checks.",
        {"type": "integer", "minimum": 1, "maximum": 128},
    ),
    (
        "action_settings.max_candidates",
        "Per-request hard candidate ceiling, itself bounded by the backend hard limit.",
        "A smaller ceiling prevents accidental combinatorial expansion.",
        {"type": "integer", "minimum": 1, "maximum": 128},
    ),
    (
        "action_settings.orientation_trials",
        "Rigid-body orientations evaluated for each moving component and candidate.",
        "More trials can reduce clashes at increased deterministic CPU cost.",
        {"type": "integer", "minimum": 1, "maximum": 256},
    ),
    (
        "action_settings.contact_tolerance_angstrom",
        "Default allowed absolute deviation from each requested contact distance.",
        "A smaller tolerance makes candidate acceptance stricter.",
        {"type": "number", "exclusive_minimum": 0.0},
    ),
    (
        "action_settings.minimum_interatomic_distance_angstrom",
        "Minimum accepted distance between atoms belonging to different components.",
        "A larger threshold rejects more close-contact placements.",
        {"type": "number", "minimum": 0.0},
    ),
):
    _register_parameter(
        "internal_reactive_complex", "build_reactive_complex_ensemble", _field_path,
        description=_description, impact=_impact, **_constraints,
    )
for _field_path, _default, _description, _impact, _constraints in (
    (
        "action_settings.maximum_reported_collisions", 32,
        "Maximum collision records retained per candidate.",
        "A larger value increases diagnostic output but does not change acceptance.",
        {"type": "integer", "minimum": 0, "maximum": 256},
    ),
    (
        "action_settings.complex_charge", None,
        "Optional explicit check against the sum of unchanged component charges.",
        "A mismatched value rejects the request; it never overrides component charge.",
        {"type": "integer | null"},
    ),
    (
        "action_settings.complex_multiplicity", None,
        "Explicit total multiplicity when component spin coupling is not unique.",
        "Selecting it fixes the electronic state of the assembled complex.",
        {"type": "integer | null", "minimum": 1},
    ),
):
    _register_parameter(
        "internal_reactive_complex", "build_reactive_complex_ensemble", _field_path,
        description=_description, default=_default, impact=_impact, **_constraints,
    )

_register_parameter(
    "internal_reactive_complex", "select_reactive_complex_candidate",
    "action_settings.candidate_id",
    description="Exact ID of one accepted candidate in the supplied ReactiveComplexEnsemble.",
    type="string",
    impact=(
        "This performs no ranking or chemistry inference; changing the ID selects a "
        "different already generated geometry for downstream calculation."
    ),
)

for _field_path, _description, _impact, _constraints in (
    ("action_settings.equilibrium_distance_scale", "Multiplier applied to the registered reference single-bond length.", "Changing it moves the center of the allowed mapped-pair distance interval.", {"type": "number", "exclusive_minimum": 0.0}),
    ("action_settings.minimum_distance_ratio", "Lower bound as a fraction of the scaled equilibrium distance.", "Increasing it raises the minimum allowed distance.", {"type": "number", "exclusive_minimum": 0.0}),
    ("action_settings.maximum_distance_ratio", "Upper bound as a fraction of the scaled equilibrium distance.", "Increasing it widens the allowed distance interval.", {"type": "number", "exclusive_minimum": 0.0}),
    ("action_settings.force_constant_kcal_mol_angstrom2", "MMFF distance-restraint force constant.", "Increasing it penalizes deviations more strongly.", {"type": "number", "exclusive_minimum": 0.0}),
):
    _register_parameter(
        "internal_reactive_complex", "build_mapped_distance_constraints", _field_path,
        description=_description, impact=_impact, **_constraints,
    )

_TRANSITION_STATE_PATH_ACTION = "search_transition_state_path"
for _field_path, _description, _impact, _constraints in (
    (
        "inputs.reactants",
        "Non-empty reactant-side SMILES array for the reaction whose candidate path is requested.",
        "Changing the reactants changes the mapped reaction, endpoints, and candidate path.",
        {"type": "array", "min_items": 1, "items": {"type": "string"}},
    ),
    (
        "inputs.products",
        "Non-empty product-side SMILES array for the reaction whose candidate path is requested.",
        "Changing the products changes the mapped reaction, endpoints, and candidate path.",
        {"type": "array", "min_items": 1, "items": {"type": "string"}},
    ),
    (
        "inputs.main_force_field",
        "Force field used for conformer and endpoint preparation.",
        "Changing this method changes the prepared endpoint geometries.",
        {"type": "string", "allowed_values": ["MMFF"], "default": "MMFF"},
    ),
    (
        "inputs.target_force_field",
        "Potential used to refine endpoint pairs and evaluate the NEB path.",
        "Changing this method changes endpoint refinement and path energies and gradients.",
        {"type": "string", "allowed_values": ["UMA"], "default": "UMA"},
    ),
):
    _register_parameter(
        "transition_state_path_search",
        _TRANSITION_STATE_PATH_ACTION,
        _field_path,
        description=_description,
        impact=_impact,
        **_constraints,
    )

for _field_path, _description in (
    ("backend_runtime.direction", "Service-fixed reaction-path direction: forward only; reverse is unsupported."),
    ("backend_runtime.charge", "Service-fixed total electronic charge: +1 for the UMA/ASH stages."),
    ("backend_runtime.multiplicity", "Service-fixed spin multiplicity: 1 for the UMA/ASH stages."),
    ("backend_runtime.conformer_controls", "Conformer generation, pruning, and endpoint bounds."),
    ("backend_runtime.uma_neb_controls", "UMA model, device policy, and ASH/KNARR NEB controls."),
):
    _register_fixed(
        "transition_state_path_search",
        _TRANSITION_STATE_PATH_ACTION,
        _field_path,
        description=_description,
        reason="This control is service-managed and is not part of the model-visible Action request.",
    )
for _field_path, _description, _impact, _constraints in (
    (
        "method_spec.model",
        "Exact FairChem UMA checkpoint name.",
        "Changing it changes the learned potential.",
        {"type": "string", "allowed_values": ["uma-s-1p1"]},
    ),
    ("method_spec.task_name", "FairChem prediction task; molecular calculations use omol.", "Changing task changes model heads and supported chemistry.", {"type": "string", "allowed_values": ["omol"]}),
    ("method_spec.device", "Explicit inference device such as cpu, cuda, or cuda:0.", "Changing device changes placement and numerical hardware.", {"type": "string"}),
    ("method_spec.allow_model_download", "Whether the selected checkpoint may be downloaded when absent from cache.", "False makes execution offline/cache-only; true permits network and cache mutation.", {"type": "boolean"}),
    ("action_settings.fmax_ev_per_angstrom", "Maximum force threshold for geometry convergence.", "A smaller threshold tightens optimization at increased cost.", {"type": "number", "exclusive_minimum": 0.0}),
    ("action_settings.optimizer", "ASE geometry optimizer.", "Different optimizers can follow different numerical trajectories.", {"type": "string", "allowed_values": ["bfgs", "lbfgs", "fire"]}),
    ("action_settings.max_steps", "Maximum geometry-optimization steps.", "More steps allow longer convergence attempts.", {"type": "integer", "minimum": 1}),
):
    _register_parameter(
        "fairchem_uma", "optimize_geometry", _field_path,
        description=_description, impact=_impact, **_constraints,
    )

for _field_path, _description, _impact, _constraints in (
    ("method_spec.model", "Exact FairChem UMA checkpoint name.", "Changing it changes every NEB energy and gradient.", {"type": "string"}),
    ("method_spec.task_name", "FairChem prediction task; molecular paths use omol.", "Changing task changes model heads and supported chemistry.", {"type": "string", "allowed_values": ["omol"]}),
    ("method_spec.device", "Explicit inference device such as cpu, cuda, or cuda:0.", "Changing it changes resource placement.", {"type": "string"}),
    ("method_spec.allow_model_download", "Whether an absent UMA checkpoint may be downloaded.", "False enforces offline/cache-only loading.", {"type": "boolean"}),
    ("method_spec.random_seed", "Seed forwarded to ASH FairchemTheory.", "Fixing it makes stochastic calculator components reproducible.", {"type": "integer"}),
    ("action_settings.total_images", "Total NEB image count including both endpoints.", "More images resolve the path more finely and increase cost.", {"type": "integer", "minimum": 3}),
    ("action_settings.climb", "Whether ASH enables the climbing-image NEB refinement.", "When true, the highest image is driven toward a saddle guess but remains unvalidated.", {"type": "boolean"}),
    ("action_settings.maximum_force_ev_per_angstrom", "ASH/KNARR maximum climbing-image force threshold.", "A smaller value tightens convergence.", {"type": "number", "exclusive_minimum": 0.0}),
    ("action_settings.rms_force_ev_per_angstrom", "ASH/KNARR RMS climbing-image force threshold.", "A smaller value tightens convergence and must not exceed the maximum threshold.", {"type": "number", "exclusive_minimum": 0.0}),
    ("action_settings.max_steps", "Maximum primary NEB iterations.", "More iterations allow longer convergence attempts.", {"type": "integer", "minimum": 1}),
    ("action_settings.interpolation", "Initial ASH path interpolation algorithm.", "Changing interpolation changes the starting band.", {"type": "string", "allowed_values": ["idpp", "geodesic"]}),
    ("action_settings.idpp_max_iterations", "Maximum IDPP interpolation iterations.", "More iterations can improve the initial IDPP band at additional setup cost.", {"type": "integer", "minimum": 1}),
):
    _register_parameter(
        "ash_neb_uma", "search_reaction_path", _field_path,
        description=_description, impact=_impact, **_constraints,
    )

for _action_id in (
    "derive_raman_spectrum", "derive_roa_spectrum", "derive_vcd_spectrum",
    "derive_ecd_spectrum", "derive_nmr_spectrum",
):
    _register_parameter(
        "internal_spectroscopy", _action_id, "action_settings.line_shape",
        description="Analytic line shape used to broaden the supplied sticks.",
        type="string", allowed_values=["gaussian", "lorentzian"],
        impact="Changing line shape changes curve tails without changing stick positions or strengths.",
    )
    _register_parameter(
        "internal_spectroscopy", _action_id, "action_settings.fwhm",
        description="Full width at half maximum in the same unit as the spectral axis.",
        type="number", exclusive_minimum=0.0,
        impact="A larger FWHM broadens and overlaps neighboring peaks.",
    )

_register_parameter(
    "internal_spectroscopy", "derive_nmr_spectrum",
    "action_settings.nucleus_atomic_number",
    description="Atomic number of the NMR-active nucleus to retain, for example 1 for hydrogen or 6 for carbon.",
    type="integer", minimum=1,
    impact=(
        "NMR references and chemical-shift ranges are nucleus-specific; explicit filtering "
        "prevents lines from different elements being combined on one axis."
    ),
)
for _action_id in (
    "derive_ir_spectrum", "derive_uv_vis_spectrum", "derive_raman_spectrum",
    "derive_roa_spectrum", "derive_vcd_spectrum", "derive_ecd_spectrum",
    "derive_nmr_spectrum",
):
    for _field_path, _default, _description, _impact, _constraints in (
        ("action_settings.grid_min", None, "Optional lower broadened-grid boundary.", "Changing it changes plotted/numerical coverage.", {"type": "number | null"}),
        ("action_settings.grid_max", None, "Optional upper broadened-grid boundary.", "Changing it changes plotted/numerical coverage.", {"type": "number | null"}),
        ("action_settings.grid_points", 2001, "Number of broadened-grid samples.", "More points increase resolution and output size.", {"type": "integer", "minimum": 2, "maximum": 200000}),
        ("action_settings.position_scale_factor", 1.0, "Multiplicative correction applied to supplied line positions.", "Changing it shifts every stick and must be scientifically justified.", {"type": "number", "exclusive_minimum": 0.0}),
    ):
        _register_parameter(
            "internal_spectroscopy", _action_id, _field_path,
            description=_description,
            default=(
                4000
                if _action_id == "derive_ir_spectrum"
                and _field_path == "action_settings.grid_points"
                else _default
            ),
            impact=_impact,
            **_constraints,
        )
_register_parameter(
    "internal_spectroscopy", "derive_nmr_spectrum", "action_settings.reference_shielding",
    description="Reference isotropic shielding used to convert shieldings into chemical shifts.",
    default=None, type="number | null",
    impact="Changing the reference translates all derived chemical shifts; it is required when shifts are not supplied directly.",
)

for _field_path, _description, _impact, _constraints in (
    ("action_settings.output_format", "Plot file format; the dependency-free renderer emits SVG only.", "The fixed SVG format is portable and preserves vector geometry.", {"type": "string", "allowed_values": ["svg"]}),
    ("action_settings.width_inches", "Requested plot width before conversion to the SVG pixel viewport.", "Larger width increases the generated viewport.", {"type": "number", "minimum": 1.0, "maximum": 40.0}),
    ("action_settings.height_inches", "Requested plot height before conversion to the SVG pixel viewport.", "Larger height increases the generated viewport.", {"type": "number", "minimum": 1.0, "maximum": 40.0}),
    ("action_settings.dpi", "Pixel conversion density used to determine the SVG viewport dimensions.", "Higher DPI increases viewport dimensions and file size.", {"type": "integer", "minimum": 36, "maximum": 1200}),
    ("action_settings.reverse_x_axis", "Whether the displayed spectral x axis runs from high to low values.", "This changes presentation only and leaves SpectrumResult values unchanged.", {"type": "boolean"}),
):
    _register_parameter(
        "internal_plotting", "render_spectrum_plot", _field_path,
        description=_description, impact=_impact, **_constraints,
    )
for _field_path, _default, _description, _impact, _constraints in (
    ("action_settings.reverse_y_axis", False, "Whether to reverse the displayed intensity axis.", "This changes presentation only.", {"type": "boolean"}),
    ("action_settings.show_grid", True, "Whether to draw reference grid lines.", "This changes presentation only.", {"type": "boolean"}),
    ("action_settings.show_sticks", True, "Whether to overlay the supplied discrete spectral sticks.", "This changes presentation only.", {"type": "boolean"}),
    ("action_settings.x_min", None, "Optional displayed x-axis minimum.", "This can crop displayed data but does not alter numerical spectrum artifacts.", {"type": "number | null"}),
    ("action_settings.x_max", None, "Optional displayed x-axis maximum.", "This can crop displayed data but does not alter numerical spectrum artifacts.", {"type": "number | null"}),
    ("action_settings.y_min", None, "Optional displayed y-axis minimum.", "This can crop displayed data but does not alter numerical spectrum artifacts.", {"type": "number | null"}),
    ("action_settings.y_max", None, "Optional displayed y-axis maximum.", "This can crop displayed data but does not alter numerical spectrum artifacts.", {"type": "number | null"}),
    ("action_settings.title", None, "Optional plot title.", "This changes presentation metadata only.", {"type": "string | null"}),
    ("action_settings.x_label", None, "Optional x-axis label override.", "This changes presentation metadata only.", {"type": "string | null"}),
    ("action_settings.y_label", None, "Optional y-axis label override.", "This changes presentation metadata only.", {"type": "string | null"}),
):
    _register_parameter(
        "internal_plotting", "render_spectrum_plot", _field_path,
        description=_description, default=_default, impact=_impact, **_constraints,
    )


# Critic2 optional search, filtering, and validation controls.
for _field_path, _default, _description, _impact in (
    ("action_settings.search_region", None, "Optional typed spatial region restricting critical-point search.", "Restricting the region reduces cost but can omit critical points outside it."),
    ("action_settings.discard_density_below", None, "Optional density threshold below which candidate critical points are discarded.", "Increasing it filters low-density features and may remove physically relevant weak-interaction critical points."),
    ("action_settings.checkpoint_critical_points", False, "Whether Critic2 checkpoints critical points during AUTO search.", "Enabling it adds restart/diagnostic output without changing intended search criteria."),
    ("action_settings.smoothrho_environment_nodes", None, "Optional Smoothrho interpolation environment-node count.", "Increasing it uses a wider interpolation environment and can improve smoothness at higher cost."),
    ("action_settings.smoothrho_distance_factor", None, "Optional Smoothrho maximum-distance factor.", "Changing it changes the interpolation neighborhood and may affect critical-point topology."),
):
    _register_parameter("critic2", "analyze_electron_density_topology", _field_path, description=_description, default=_default, impact=_impact)
_CRITIC2_BASIN_ACTIONS = ("calculate_atomic_basin_properties", "calculate_bader_charges")
for _field_path, _default, _description, _impact in (
    ("action_settings.discard_density_below", None, "Optional density threshold below which integration candidates are discarded.", "Increasing it can reduce noise and cost but may remove low-density basin contributions."),
    ("action_settings.selected_attractors", None, "Optional explicit attractor identifiers to integrate.", "Restricting attractors reduces output and omits all unselected basins."),
    ("action_settings.selected_attractor_range", None, "Optional inclusive attractor identifier range.", "Restricting the range reduces output and excludes basins outside it."),
    ("action_settings.expected_total_charge", None, "Optional expected total molecular/system charge used for a validation warning.", "It does not change integration; it enables a physical consistency check."),
    ("action_settings.total_charge_tolerance", 0.05, "Allowed absolute deviation from expected total charge.", "A smaller tolerance makes validation stricter and may flag grid or partition errors more often."),
    ("action_settings.smoothrho_environment_nodes", None, "Optional Smoothrho interpolation environment-node count.", "Increasing it uses a wider interpolation environment and can improve smoothness at higher cost."),
    ("action_settings.smoothrho_distance_factor", None, "Optional Smoothrho maximum-distance factor.", "Changing it changes the interpolation neighborhood and may affect basin boundaries."),
    ("action_settings.attractor_assignment_radius", None, "Optional YT/Bader attractor-assignment radius.", "Increasing it expands the association radius and can change basin-to-attractor assignments."),
):
    _register_parameter("critic2", _CRITIC2_BASIN_ACTIONS, _field_path, description=_description, default=_default, impact=_impact)


# Remaining data-source, structure, and system-construction controls.
for _field_name in ("match_charges", "match_isotopes"):
    _register_parameter(
        "pubchem", "search_substructures", f"action_settings.{_field_name}",
        description=f"Whether PubChem substructure matching requires exact {_field_name.removeprefix('match_')} annotations.",
        default=False, type="boolean",
        impact="Enabling this restriction narrows matches and can exclude records lacking the requested annotation.",
    )
for _field_path, _description, _impact in (
    ("method_spec.method", "CREST Hamiltonian: GFN1-xTB, GFN2-xTB, or GFN-FF; common compact and -xTB aliases are accepted.", "Changing the Hamiltonian changes conformer energies, geometries, sampling cost, and potentially basin coverage."),
    ("method_spec.charge", "Optional CREST molecular charge override; otherwise the input structure charge is used.", "Changing charge changes electron count and the conformational potential-energy surface."),
    ("method_spec.multiplicity", "Optional CREST spin multiplicity override; otherwise the input structure multiplicity is used.", "Changing multiplicity changes the electronic state used by xTB/CREST."),
    ("method_spec.solvation_model", "Optional CREST implicit-solvation model: ALPB or GBSA.", "Adding or changing implicit solvation changes conformer stabilization and may change the sampled/ranked ensemble."),
    ("method_spec.solvent", "Solvent name passed to CREST when an implicit-solvation model is selected.", "Changing solvent changes implicit-solvation parameters and relative conformer energies."),
):
    _register_parameter("crest", "generate_conformer_ensemble", _field_path, description=_description, default=None, impact=_impact)
_register_parameter(
    "openmm_builder", "solvate_molecular_system", "method_spec.force_field",
    description="Optional OpenMM force-field XML file or list; otherwise the ParameterizedSystem force-field metadata is reused.",
    default=None, type="string | array[string]",
    impact="Changing the force field changes solvent parameters and the generated OpenMM System; it must cover every topology residue.",
)
_register_parameter(
    "crest", "generate_conformer_ensemble", "action_settings.energy_window_kcal_mol",
    description="Optional CREST conformer energy window in kcal/mol.", default=None,
    type="number", minimum=0,
    impact="Increasing it retains higher-energy conformers and increases ensemble size and downstream cost.",
)


# Licensed molecular-dynamics adapters retain task-independent advanced controls.
_register_parameter(
    "namd", ("minimize_system_energy", "propagate_dynamics"), "method_spec.margin_angstrom",
    description="NAMD patch-grid margin in angstrom.", default=1.0, type="number", minimum=0,
    impact="A larger margin can avoid patch-grid failures for moving atoms but increases spatial bookkeeping and memory.",
)
for _field_path, _default, _description, _impact in (
    ("action_settings.langevin_damping_per_ps", 1.0, "NAMD Langevin damping coefficient for NVT/NPT.", "Larger damping couples more strongly to the thermostat and can distort dynamics."),
    ("action_settings.langevin_hydrogen", False, "Whether NAMD applies Langevin forces to hydrogen atoms.", "Enabling it changes thermostat coupling of high-frequency hydrogen motion."),
    ("action_settings.piston_period_fs", 200.0, "NAMD Langevin-piston oscillation period for NPT.", "Changing it alters barostat response; overly small values can destabilize volume dynamics."),
    ("action_settings.piston_decay_fs", 100.0, "NAMD Langevin-piston decay time for NPT.", "Changing it alters damping of pressure oscillations."),
    ("action_settings.pressure_bar", None, "Target NAMD pressure required for NPT propagation.", "Changing it changes the sampled thermodynamic state; it is unused for NVE/NVT."),
):
    _register_parameter("namd", "propagate_dynamics", _field_path, description=_description, default=_default, impact=_impact)
_register_parameter(
    "amber_pmemd", ("minimize_system_energy", "propagate_dynamics"), "method_spec.saltcon_molar",
    description="Amber implicit-solvent salt concentration in mol/L.", default=0.0, type="number", minimum=0,
    impact="Increasing it changes electrostatic screening in compatible generalized-Born models.",
)
_register_parameter(
    "amber_pmemd", ("minimize_system_energy", "propagate_dynamics"), "method_spec.igb",
    description="Amber generalized-Born model identifier required when boundary=implicit.",
    default=None, type="integer", allowed_values=[1, 2, 5, 6, 7, 8, 10],
    impact="Changing IGB selects a different implicit-solvent model and can materially change energies and dynamics.",
)
for _field_path, _default, _description, _impact in (
    ("action_settings.langevin_collision_per_ps", 1.0, "Amber Langevin collision frequency for NVT/NPT.", "A larger value thermostats more strongly and changes dynamical correlations."),
    ("action_settings.pressure_relaxation_ps", 2.0, "Amber barostat pressure-relaxation time for NPT.", "A smaller value couples pressure more strongly and may destabilize volume fluctuations."),
    ("action_settings.pressure_bar", None, "Target Amber pressure required for NPT.", "Changing it changes the NPT thermodynamic state; it is unused outside NPT."),
):
    _register_parameter("amber_pmemd", "propagate_dynamics", _field_path, description=_description, default=_default, impact=_impact)
_register_parameter(
    "charmm", ("minimize_system_energy", "propagate_dynamics"), "method_spec.constraint_tolerance",
    description="CHARMM SHAKE bond-constraint tolerance.", default=1e-8, type="number", minimum=0,
    impact="A smaller tolerance enforces constraints more tightly and may require more iterations.",
)
_register_parameter(
    "charmm", "propagate_dynamics", "action_settings.temperature_coupling_ps",
    description="CHARMM NVT temperature-coupling time constant.", default=5.0, type="number", minimum=0,
    impact="A smaller value couples temperature more strongly and can perturb physical dynamics.",
)


# GoodVibes options and conditional controls are available on every GoodVibes Action.
_GOODVIBES_ACTIONS = (
    "derive_thermochemistry", "scan_thermochemistry_temperature",
    "analyze_thermochemical_ensemble", "validate_thermochemistry_inputs",
    "analyze_thermochemical_selectivity", "analyze_reaction_free_energy_profile",
)
for _field_path, _description, _impact in (
    ("method_spec.single_point_correction_suffix", "Optional GoodVibes --spc suffix for already-existing single-point files.", "Selecting a suffix replaces frequency-job electronic energies with the matching single-point energies."),
    ("method_spec.custom_file_extensions", "Optional additional quantum-output filename extensions accepted by GoodVibes.", "Adding extensions broadens input discovery but does not change parsed scientific data."),
    ("method_spec.exclude_pattern", "Optional filename glob excluded from GoodVibes analysis.", "A broader pattern removes more structures and can change ensemble/profile conclusions."),
    ("method_spec.free_space_solvent", "Optional GoodVibes free-space solvent correction name.", "Changing the solvent changes the free-volume entropy correction."),
    ("method_spec.media_solvent", "Optional GoodVibes solution-media solvent used for concentration correction.", "Changing the solvent changes the solution-phase standard-state correction applied by GoodVibes."),
):
    _register_parameter("goodvibes", _GOODVIBES_ACTIONS, _field_path, description=_description, default=None, impact=_impact)
for _field_path, _default, _description, _impact in (
    ("action_settings.concentration_mol_l", None, "Custom standard-state concentration used when standard_state=custom_concentration.", "Changing concentration changes the translational standard-state correction."),
    ("action_settings.entropy_frequency_cutoff_cm1", None, "Frequency cutoff for Grimme or Truhlar quasi-harmonic entropy.", "Raising it applies the quasi-harmonic replacement to more low-frequency modes and changes entropy."),
    ("action_settings.free_rotor_inertia_model", None, "Free-rotor inertia model required for Grimme entropy.", "global and per_conformer use different inertia treatments and can change conformer free energies."),
    ("action_settings.enthalpy_frequency_cutoff_cm1", None, "Frequency cutoff for Head-Gordon quasi-harmonic enthalpy.", "Raising it applies the correction to more modes and changes enthalpy."),
    ("action_settings.imaginary_frequency_threshold_cm1", None, "Magnitude threshold used when inverting small imaginary frequencies.", "A larger threshold inverts a wider set of imaginary modes and can mask genuine instabilities if chosen poorly."),
):
    _register_parameter("goodvibes", _GOODVIBES_ACTIONS, _field_path, description=_description, default=_default, impact=_impact)
for _action_id in ("analyze_thermochemical_ensemble", "analyze_thermochemical_selectivity"):
    for _field_path, _description, _impact in (
        ("action_settings.duplicate_energy_cutoff_kcal_mol", "Energy cutoff used when deduplicate_structures=true.", "Increasing it makes more structures eligible to be classified as duplicates."),
        ("action_settings.duplicate_rotational_cutoff_fraction", "Rotational-constant fractional cutoff used for duplicate detection.", "Increasing it makes duplicate classification less strict in rotational constants."),
        ("action_settings.duplicate_rmsd_cutoff_angstrom", "Optional RMSD cutoff used for duplicate detection.", "Increasing it makes geometrically less similar structures eligible as duplicates; null disables this criterion."),
    ):
        _register_parameter("goodvibes", _action_id, _field_path, description=_description, default=None, impact=_impact)


# GPAW and legacy quantum-chemistry advanced controls.
_GPAW_CALCULATION_ACTIONS = (
    "calculate_energy", "calculate_forces", "optimize_geometry",
    "calculate_periodic_energy", "calculate_periodic_forces",
    "calculate_periodic_stress", "relax_periodic_structure",
)
for _field_path, _description, _impact in (
    ("method_spec.charge", "Optional GPAW charge override; otherwise structure metadata is used.", "Changing charge changes electron count and electronic state."),
    ("method_spec.ecut_ev", "Plane-wave cutoff required when GPAW mode=pw.", "Increasing it improves plane-wave completeness at rapidly increasing FFT and memory cost; convergence testing is required."),
    ("method_spec.grid_spacing_angstrom", "Real-space grid spacing required when GPAW mode=fd.", "A smaller spacing gives a finer grid and usually better accuracy at sharply increased memory and runtime cost."),
    ("method_spec.basis", "LCAO basis required when GPAW mode=lcao.", "Changing it changes basis completeness, cost, and basis-set error."),
    ("method_spec.occupations_width_ev", "Optional Fermi-Dirac occupation smearing width in eV.", "A larger width can improve metallic SCF convergence but changes fractional occupations and free-energy interpretation."),
    ("method_spec.setups", "Optional GPAW PAW setup selection.", "Changing PAW setups changes the frozen-core/valence representation and numerical results."),
    ("method_spec.initial_magnetic_moments", "Per-atom initial magnetic moments required for spin-polarized GPAW calculations.", "Changing the initial moments can lead SCF to a different magnetic solution."),
):
    _register_parameter("gpaw", _GPAW_CALCULATION_ACTIONS, _field_path, description=_description, default=None, impact=_impact)
for _field_name, _description, _impact, _type in (
    ("ndfunc", "Number of d polarization functions in the GAMESS basis.", "Increasing it enlarges the basis and cost and can improve polarization flexibility.", "integer"),
    ("npfunc", "Number of p polarization functions in the GAMESS basis.", "Increasing it enlarges the basis and cost and can improve polarization flexibility.", "integer"),
    ("nffunc", "Number of f polarization functions in the GAMESS basis.", "Increasing it enlarges the basis and cost and can improve polarization flexibility.", "integer"),
    ("diffsp", "Whether GAMESS adds diffuse s/p functions.", "Enabling diffuse functions is important for anions and diffuse states but increases cost and linear-dependence risk.", "boolean"),
    ("diffs", "Whether GAMESS adds diffuse s functions on hydrogens.", "Enabling it improves diffuse hydrogen basis flexibility at added cost.", "boolean"),
):
    _register_parameter("gamess", _GAMESS_ACTIONS, f"method_spec.{_field_name}", description=_description, default=None, type=_type, impact=_impact)


# Periodic electronic-structure controls.
_PERIODIC_ACTIONS = (
    "calculate_periodic_energy", "calculate_periodic_forces",
    "calculate_periodic_stress", "relax_periodic_structure",
)
for _field_path, _default, _description, _impact in (
    ("method_spec.ecutrho_ry", None, "Quantum ESPRESSO charge-density cutoff in rydberg.", "Increasing it improves density representation for demanding pseudopotentials at substantial FFT/memory cost."),
    ("method_spec.occupations", None, "Quantum ESPRESSO occupation scheme.", "Changing it changes how electronic occupations are assigned, especially for metals."),
    ("method_spec.smearing", None, "Quantum ESPRESSO smearing function.", "Changing the smearing function changes metallic occupation broadening and convergence."),
    ("method_spec.degauss_ry", 0.01, "Quantum ESPRESSO smearing width in rydberg.", "A larger width improves metallic convergence but broadens occupations and changes finite-smearing energies."),
    ("method_spec.atomic_masses", {}, "Optional element-to-mass mapping written to Quantum ESPRESSO.", "Changing masses affects ionic dynamics/phonons but not a static Born-Oppenheimer electronic energy."),
    ("action_settings.scf_convergence_ry", 1e-8, "Quantum ESPRESSO SCF energy threshold in rydberg.", "A smaller threshold tightens SCF convergence and increases iterations."),
):
    _register_parameter("quantum_espresso", _PERIODIC_ACTIONS, _field_path, description=_description, default=_default, impact=_impact)
for _field_path, _default, _description, _impact in (
    ("action_settings.ion_dynamics", "bfgs", "Quantum ESPRESSO ionic optimizer for relaxation.", "Changing it changes the geometry-search algorithm and convergence path."),
    ("action_settings.pressure_threshold_kbar", 0.5, "Cell-pressure convergence threshold for vc-relax in kbar.", "A smaller value requires tighter cell convergence and usually more steps."),
):
    _register_parameter("quantum_espresso", "relax_periodic_structure", _field_path, description=_description, default=_default, impact=_impact)

for _field_path, _default, _description, _impact in (
    ("action_settings.scf_convergence", 1e-6, "CP2K inner-SCF convergence threshold.", "A smaller threshold tightens electronic convergence and increases iterations."),
    ("action_settings.max_scf_cycles", 100, "Maximum CP2K inner-SCF cycles.", "A larger value allows harder SCF problems to converge but increases worst-case time."),
    ("method_spec.basis_set_file", "BASIS_MOLOPT", "CP2K basis-set database filename.", "Changing it changes which named basis definitions are available and must match the installation."),
    ("method_spec.potential_file", "GTH_POTENTIALS", "CP2K pseudopotential database filename.", "Changing it changes available pseudopotential definitions and must match selected potentials."),
):
    _register_parameter("cp2k", _PERIODIC_ACTIONS, _field_path, description=_description, default=_default, impact=_impact)
for _field_path, _default, _description, _impact in (
    ("method_spec.ot_minimizer", "DIIS", "CP2K orbital-transformation minimizer.", "Changing it alters OT convergence behavior."),
    ("method_spec.ot_preconditioner", "FULL_SINGLE_INVERSE", "CP2K OT preconditioner.", "Changing it affects convergence speed, memory, and robustness."),
    ("action_settings.outer_scf_convergence", "same as scf_convergence", "CP2K outer-SCF convergence threshold.", "A smaller threshold tightens outer-loop convergence and increases work."),
    ("action_settings.max_outer_scf_cycles", 20, "Maximum CP2K outer-SCF cycles.", "A larger value permits more recovery iterations at greater worst-case cost."),
    ("method_spec.added_mos", 20, "Additional unoccupied orbitals for CP2K diagonalization.", "More orbitals support smearing/excited occupations but increase diagonalization cost."),
    ("method_spec.electronic_temperature_kelvin", 300.0, "CP2K Fermi-Dirac electronic temperature.", "Increasing it broadens occupations and can improve metallic convergence while changing electronic free energy."),
    ("method_spec.mixing_method", "BROYDEN_MIXING", "CP2K density-mixing algorithm.", "Changing it alters SCF stability and convergence rate."),
    ("method_spec.mixing_alpha", 0.2, "CP2K density-mixing amplitude.", "Larger values update density more aggressively and may accelerate or destabilize SCF."),
):
    _register_parameter("cp2k", _PERIODIC_ACTIONS, _field_path, description=_description, default=_default, impact=_impact)

_DFTB_ACTIONS = ("calculate_periodic_energy", "calculate_periodic_forces", "relax_periodic_structure")
for _field_path, _description, _impact in (
    ("method_spec.charge", "Optional DFTB+ total charge.", "Changing charge changes electron count and electronic state."),
    ("method_spec.shell_resolved_scc", "Optional DFTB+ shell-resolved SCC switch.", "Toggling it changes the charge self-consistency model when supported by the parameter set."),
    ("method_spec.third_order_full", "Optional full third-order DFTB correction.", "Enabling it changes the Hamiltonian and requires compatible Hubbard derivatives."),
    ("method_spec.hubbard_derivatives", "Optional element-to-Hubbard-derivative mapping.", "Changing values changes third-order charge corrections and must match the parameter set."),
    ("method_spec.damp_xh_exponent", "Optional X-H damping exponent.", "Changing it alters short-range hydrogen damping in compatible DFTB parameterizations."),
    ("method_spec.fermi_temperature_kelvin", "Optional electronic Fermi temperature.", "Increasing it broadens occupations and can improve metallic SCC convergence."),
):
    _register_parameter("dftbplus", _DFTB_ACTIONS, _field_path, description=_description, default=None, impact=_impact)
_register_parameter(
    "abinit", _PERIODIC_ACTIONS, "action_settings.scf_convergence_hartree",
    description="ABINIT SCF density-residual threshold in hartree units used by this adapter.",
    default=1e-10, type="number", minimum=0,
    impact="A smaller threshold tightens SCF convergence and typically increases iterations.",
)
for _field_path, _description, _impact in (
    ("method_spec.initial_magnetic_moments", "Optional per-atom VASP MAGMOM initialization.", "Changing initial moments can lead to different converged magnetic states."),
    ("method_spec.electron_count", "Optional VASP NELECT override.", "Changing it changes total electron count and charged-state treatment."),
    ("method_spec.additional_incar", "Optional validated additional INCAR mapping excluding Action-semantic keys.", "Additional INCAR tags can materially alter algorithms and results; the caller is responsible for consistency."),
):
    _register_parameter("vasp", _PERIODIC_ACTIONS, _field_path, description=_description, default=None, impact=_impact)


# Phonon, thermal-transport, and periodic post-processing controls.
_PHONON_HARMONIC_ACTIONS = (
    "generate_displaced_supercells", "assemble_force_constants",
    "calculate_phonon_dispersion", "calculate_phonon_density_of_states",
    "calculate_harmonic_thermodynamics", "calculate_phonon_group_velocities",
)
for _backend_id in ("phonopy", "phono3py"):
    _register_parameter(
        _backend_id, "generate_displaced_supercells", "action_settings.primitive_matrix",
        description="Primitive-cell transformation matrix or 'auto' for phonon construction.",
        default="auto", impact="Changing it changes primitive-cell mapping, displacement symmetry, and reciprocal-space folding.",
    )
_register_parameter(
    "phono3py", "generate_displaced_supercells", "action_settings.phonon_supercell_matrix",
    description="Optional separate second-order phonon supercell matrix.", default="same as supercell_matrix",
    impact="Changing it changes the fc2 displacement cell independently of the fc3 cell and changes cost/finite-size error.",
)
for _field_path, _default, _description, _impact in (
    ("action_settings.symmetrize_fc3q", False, "Whether phono3py symmetrizes reciprocal-space third-order force constants.", "Enabling it enforces symmetry relations and can reduce numerical noise."),
    ("action_settings.mass_variances", None, "Optional isotope mass-variance parameters.", "Supplying them changes isotope-scattering rates and thermal conductivity."),
    ("action_settings.use_kappa_star", True, "Whether phono3py uses kappa-star symmetry reduction.", "Disabling it can increase cost and serves as a symmetry/convergence check."),
    ("action_settings.full_phonon_phonon_interaction", False, "Whether phono3py retains the full phonon-phonon interaction representation.", "Enabling it increases memory and cost and can change approximation details."),
    ("action_settings.include_mode_data", False, "Whether per-mode linewidth, velocity, and heat-capacity arrays are returned.", "Enabling it substantially increases output size but not the conductivity solution."),
):
    _register_parameter("phono3py", "calculate_lattice_thermal_conductivity", _field_path, description=_description, default=_default, impact=_impact)
for _backend_id in ("phonopy", "phono3py"):
    for _action_id in ("calculate_phonon_dispersion", "calculate_phonon_group_velocities"):
        _register_parameter(
            _backend_id, _action_id, "action_settings.with_eigenvectors",
            description="Whether phonon eigenvectors are calculated with the band path.", default=False, type="boolean",
            impact="Enabling it increases time/output and provides mode polarization without changing frequencies.",
        )
        _register_parameter(
            _backend_id, _action_id, "action_settings.labels",
            description="Optional labels aligned to q-path segment endpoints.", default=None,
            impact="Labels affect presentation only and do not change the phonon calculation.",
        )
    for _field_path, _default, _description, _impact in (
        ("action_settings.cutoff_frequency_thz", None, "Optional minimum frequency included in harmonic thermodynamics.", "Raising it excludes more low-frequency modes and changes thermodynamic sums."),
        ("action_settings.pretend_real", False, "Whether imaginary phonon frequencies are treated as real magnitudes.", "Enabling it can produce finite thermodynamics for unstable structures but changes physical interpretation."),
        ("action_settings.classical", False, "Whether classical rather than quantum harmonic statistics are used.", "Enabling it changes heat capacities, entropy, and free energy, especially at low temperature."),
    ):
        _register_parameter(_backend_id, "calculate_harmonic_thermodynamics", _field_path, description=_description, default=_default, impact=_impact)

for _backend_id in ("nequip", "allegro", "deepmd"):
    for _field_path, _default, _description, _impact in (
        ("action_settings.hydrostatic_strain", False, "Whether ASE cell relaxation is constrained to hydrostatic strain.", "Enabling it removes shear/deviatoric cell degrees of freedom."),
        ("action_settings.scalar_pressure_ev_per_angstrom3", 0.0, "External scalar pressure supplied to the ASE cell filter.", "Changing it changes the target enthalpy and relaxed cell volume."),
    ):
        _register_parameter(_backend_id, "relax_periodic_structure", _field_path, description=_description, default=_default, impact=_impact)
for _field_path, _description, _impact in (
    ("inputs.born_file", "Optional ShengBTE BORN file staged under its native filename.", "Supplying it enables non-analytical/Born-charge information when required by the model."),
    ("inputs.companion_files", "Optional bounded list of additional native ShengBTE input files.", "Adding files can enable model features but leaves their scientific consistency to the caller."),
):
    _register_parameter("shengbte", "calculate_lattice_thermal_conductivity", _field_path, description=_description, default=None, impact=_impact)


# Reaction-path, kinetics, and master-equation controls.
_PYSISYPHUS_ACTIONS = (
    "locate_transition_state", "search_reaction_path",
    "scan_reaction_coordinates", "trace_intrinsic_reaction_coordinate",
)
for _field_path, _description, _impact in (
    ("method_spec.charge", "Optional reaction-path molecular charge override.", "Changing charge changes the potential-energy surface."),
    ("method_spec.multiplicity", "Optional reaction-path spin multiplicity override.", "Changing multiplicity changes the electronic-state surface."),
    ("method_spec.basis", "Basis set used by PySCF and optionally ORCA calculators.", "Changing it changes accuracy, cost, and the potential-energy surface."),
    ("method_spec.functional", "DFT functional used by the PySCF calculator.", "Changing it changes the density-functional approximation and reaction path."),
    ("method_spec.solvation_model", "Optional xTB or ORCA implicit-solvation model.", "Enabling it changes the reaction environment and energy surface."),
    ("method_spec.solvent", "Solvent name required when solvation_model is selected.", "Changing solvent changes implicit-solvation parameters and barriers."),
):
    _register_parameter("pysisyphus", _PYSISYPHUS_ACTIONS, _field_path, description=_description, default=None, impact=_impact)
_register_parameter(
    "pysisyphus",
    "scan_reaction_coordinates",
    "action_settings.steps",
    description=(
        "Number of equal scan intervals between start_value and end_value. "
        "Pysisyphus evaluates both endpoints, so the requested point count is steps + 1."
    ),
    type="integer",
    minimum=1,
    impact=(
        "More intervals increase coordinate resolution and create one additional optimized "
        "point per interval; runtime and output generally scale with steps + 1."
    ),
)
for _field_path, _description, _impact in (
    ("inputs.mechanism", "Optional Cantera mechanism file/name for equilibrium calculation.", "Changing the mechanism changes available species, thermochemistry, and reactions."),
    ("action_settings.mechanism", "Legacy alternative location for the Cantera mechanism.", "Changing the mechanism changes available species, thermochemistry, and reactions."),
):
    _register_parameter("cantera", "calculate_chemical_equilibrium", _field_path, description=_description, default=None, impact=_impact)
_register_parameter(
    "cantera", "calculate_chemical_equilibrium", "action_settings.report_threshold",
    description="Minimum equilibrium mole fraction included in the result.", default=1e-12, type="number", minimum=0,
    impact="Increasing it shortens output but hides more trace species; it does not change equilibrium itself.",
)
for _field_path, _default, _description, _impact in (
    ("action_settings.integrator", "LSODA", "SciPy solve_ivp integration method.", "Changing it alters stiffness handling, stability, and cost; converged solutions should agree within tolerances."),
    ("action_settings.relative_tolerance", 1e-8, "Relative ODE integration tolerance.", "A smaller value tightens integration accuracy and increases solver work."),
    ("action_settings.absolute_tolerance", 1e-12, "Absolute ODE integration tolerance.", "A smaller value resolves lower concentrations more accurately and increases solver work."),
):
    _register_parameter("scipy", "integrate_reaction_network", _field_path, description=_description, default=_default, impact=_impact)
_register_parameter(
    "rmg", "calculate_rate_constants", "inputs.kinetics_model",
    description=("Typed RMG kinetics model. Caller-controlled nested fields include type, reaction_order, "
                 "Arrhenius terms, temperature bounds, pressure grids/bounds, Chebyshev coefficients, and units."),
    impact="Changing any nested model coefficient, validity range, pressure, or unit changes calculated rate coefficients.",
    nested_fields={
        "type": "arrhenius | multi_arrhenius | pressure_dependent_arrhenius | chebyshev",
        "reaction_order": "1..4; determines output rate unit",
        "minimum_temperature_kelvin": "optional validity lower bound",
        "maximum_temperature_kelvin": "optional validity upper bound",
        "terms": "Arrhenius term or term list",
        "pressures_bar": "tabulated pressures for pressure-dependent Arrhenius",
        "coefficients": "Chebyshev coefficient matrix",
        "minimum_pressure_bar": "Chebyshev validity lower bound",
        "maximum_pressure_bar": "Chebyshev validity upper bound",
    },
)


# Multiwfn 原生单位属于实现事实，与调用方设置的规范参数 grid_spacing_bohr 分开保存。
_register_parameter(
    "internal_thermochemistry",
    "analyze_free_energy_selectivity",
    "inputs.free_energies",
    description=(
        "Ordered list of at least two explicit records containing label and finite "
        "gibbs_free_energy; optional conformer_id is preserved, and repeated labels "
        "are combined as one conformer group. Every supplied conformer_id must be "
        "globally unique across all labels."
    ),
    type="array[object] | LabeledFreeEnergyCollection ArtifactRef",
    minimum_items=2,
    impact=(
        "Adding, removing, relabeling, or changing a conformer changes that label's "
        "partition function and therefore all reported populations and selectivity."
    ),
)
_register_parameter(
    "internal_thermochemistry",
    "analyze_free_energy_selectivity",
    "inputs.energy_unit",
    description=(
        "One explicit common unit for all gibbs_free_energy values: hartree, eV, "
        "kcal_mol, or kJ_mol."
    ),
    type="string",
    allowed_values=["hartree", "eV", "kcal_mol", "kJ_mol"],
    impact=(
        "The unit determines conversion into kcal/mol before Boltzmann weighting; "
        "a wrong unit changes the selectivity exponentially."
    ),
)
_register_parameter(
    "internal_thermochemistry",
    "analyze_free_energy_selectivity",
    "action_settings.temperature_kelvin",
    description="Positive finite temperature used for every Boltzmann partition function.",
    type="number",
    exclusive_minimum=0.0,
    impact="Changing temperature changes conformer and label populations and the resulting excess.",
)
_register_parameter(
    "internal_thermochemistry",
    "analyze_free_energy_selectivity",
    "action_settings.comparison_type",
    description=(
        "Explicit interpretation of the supplied labels. Use population for a generic "
        "two- or N-way population comparison; use enantiomeric only for exactly two "
        "exhaustive opposite-enantiomer labels."
    ),
    type="string",
    allowed_values=["population", "enantiomeric"],
    impact=(
        "ee_percent is emitted only for an explicit two-label enantiomeric comparison; "
        "the Action never infers stereochemical meaning from label text."
    ),
)

_register_parameter(
    "internal_thermochemistry",
    "analyze_energetic_span",
    "inputs.cycle_states",
    description=(
        "Complete ordered catalytic cycle as strict alternating intermediate and "
        "transition_state records, beginning with an intermediate. Each record requires a "
        "globally unique id and finite gibbs_free_energy."
    ),
    type="array[object] | CatalyticCycleFreeEnergyProfile ArtifactRef",
    minimum_items=2,
    nested_fields={
        "id": "globally unique non-empty state identifier",
        "state_type": "intermediate | transition_state",
        "gibbs_free_energy": "finite numeric value in inputs.energy_unit",
    },
    impact=(
        "Adding, removing, reordering, retyping, or changing any state changes the "
        "intermediate/transition-state pair spans, TDI/TDTS, and full TOF denominator."
    ),
)
_register_parameter(
    "internal_thermochemistry",
    "analyze_energetic_span",
    "inputs.energy_unit",
    description=(
        "One explicit common unit for all state Gibbs free energies and the overall reaction "
        "free energy: hartree, eV, kcal_mol, or kJ_mol."
    ),
    type="string",
    allowed_values=["hartree", "eV", "kcal_mol", "kJ_mol"],
    impact=(
        "The unit determines conversion into kcal/mol before energetic-span and exponential "
        "TOF evaluation; a wrong unit changes the result exponentially."
    ),
)
_register_parameter(
    "internal_thermochemistry",
    "analyze_energetic_span",
    "inputs.overall_reaction_free_energy",
    description=(
        "Finite Gibbs free energy of one turnover in the explicitly supplied forward cycle, "
        "in inputs.energy_unit. It must be negative for a positive forward net TOF."
    ),
    type="number",
    exclusive_maximum=0.0,
    impact=(
        "This driving force enters wrap-around intermediate/transition-state pairs and the "
        "full TOF numerator; changing its sign changes the forward-cycle semantics."
    ),
)
_register_parameter(
    "internal_thermochemistry",
    "analyze_energetic_span",
    "action_settings.temperature_kelvin",
    description=(
        "Positive finite temperature used in the transition-state-theory prefactor and every "
        "energetic-span exponential."
    ),
    type="number",
    exclusive_minimum=0.0,
    impact="Changing temperature changes the full-denominator turnover frequency.",
)


_register_fixed(
    "multiwfn", "calculate_electron_isodensity_surface", "backend_runtime.native_grid_spacing_unit",
    description="The installed Multiwfn surface-grid menu accepts grid spacing in bohr.",
    reason="The software-native unit is fixed by Multiwfn; the public Action exposes it explicitly as action_settings.grid_spacing_bohr.",
)


def parameter_specs_for_backend(
    backend_id: str,
) -> Mapping[str, Mapping[str, ParameterMetadata]]:
    return BACKEND_PARAMETER_SPECS.get(backend_id, {})


def fixed_parameter_specs_for_backend(
    backend_id: str,
) -> Mapping[str, Mapping[str, ParameterMetadata]]:
    return BACKEND_FIXED_PARAMETER_SPECS.get(backend_id, {})


def common_fixed_parameter_specs(
    *,
    runtime: str,
    executables: tuple[str, ...],
    python_modules: tuple[str, ...],
    resource_constraints: Mapping[str, Any],
    validation_level: str | None,
) -> dict[str, dict[str, Any]]:
    """Describe implementation constraints that remain fixed after provider selection."""

    implementation = ", ".join(executables or python_modules) or "internal implementation"
    values: dict[str, dict[str, Any]] = {
        "backend_runtime.runtime_family": {
            "description": f"Selected provider executes through the {runtime!r} runtime family.",
            "reason": "The caller can select another advertised backend, but cannot replace the implementation behind a selected backend id inside a predefined Action call.",
        },
        "backend_runtime.implementation": {
            "description": f"Executable/module implementation: {implementation}.",
            "reason": "Executable and module allowlists are operator-installed security and reproducibility boundaries.",
        },
        "backend_runtime.automatic_fallback": {
            "description": "Automatic fallback to another scientific backend is disabled.",
            "reason": "Execution provenance requires the declared provider to be executed or to fail explicitly.",
        },
    }
    if validation_level:
        values["backend_runtime.validation_level"] = {
            "description": f"Declared adapter validation level: {validation_level}.",
            "reason": "Validation status records tested implementation coverage; it is not a per-call scientific parameter.",
        }
    for field_name in ("maximum_cpu_cores", "maximum_walltime_seconds"):
        if field_name in resource_constraints:
            reason_key = (
                "walltime_reason"
                if field_name == "maximum_walltime_seconds"
                else "reason"
            )
            values[f"resource_limits.{field_name}"] = {
                "description": f"Backend-enforced {field_name.replace('_', ' ')}: {resource_constraints[field_name]}.",
                "reason": str(resource_constraints.get(reason_key) or "Operator-defined backend resource safety cap."),
            }
    return values


_FIELD_DESCRIPTIONS = {
    "method": "Electronic-structure, force-field, kinetic, or statistical method selected by the caller.",
    "basis": "Orbital basis set used by the selected quantum-chemistry method.",
    "basis_set": "Basis-set family or exact basis resource used by the backend.",
    "functional": "Density-functional approximation selected for the calculation.",
    "xc": "Exchange-correlation functional selected for the calculation.",
    "charge": "Net system charge in units of the elementary charge.",
    "multiplicity": "Spin multiplicity, equal to two times total spin plus one.",
    "spin": "Backend-specific spin or number of unpaired electrons.",
    "temperature_kelvin": "Thermodynamic or simulation temperature in kelvin.",
    "pressure_bar": "Pressure in bar.",
    "random_seed": "Random-number seed controlling reproducible stochastic sampling.",
    "max_steps": "Maximum number of optimization, dynamics, or iteration steps.",
    "max_iterations": "Maximum number of numerical iterations.",
    "max_cycles": "Maximum number of solver or optimizer cycles.",
    "grid_points_per_axis": "Number of density-grid samples along every Cartesian axis.",
    "grid_spacing_bohr": "Requested real-space grid spacing in bohr.",
    "grid_spacing_angstrom": "Requested real-space grid spacing as currently named by the backend contract.",
    "cutoffs_au": "Explicit electron-density isovalue list in atomic units.",
    "output_format": "Caller-selected representation or serialization format for the result.",
    "density_source": "Named electronic density stored by the preceding calculation.",
    "properties": "Explicit result properties requested from the parser or data source.",
}


def inferred_parameter_metadata(
    section: str,
    field_name: str,
    *,
    required: bool,
) -> dict[str, Any]:
    """Provide a complete fallback explanation for every public field.

    Backend-specific registrations override these neutral descriptions.  The
    fallback prevents a required field from appearing in discovery or audit
    output without any statement of purpose or directional effect.
    """

    description = _FIELD_DESCRIPTIONS.get(field_name)
    if description is None:
        readable = field_name.replace("_", " ")
        if section == "inputs":
            description = f"Input value for {readable}."
        elif section == "method_spec":
            description = f"Caller-selected scientific method parameter controlling {readable}."
        elif section == "action_settings":
            description = f"Caller-selected execution setting controlling {readable}."
        else:
            description = f"Caller-controlled parameter for {readable}."

    lower = field_name.casefold()
    if any(token in lower for token in ("tolerance", "convergence", "threshold")):
        impact = (
            "A stricter numerical threshold usually improves convergence quality or filtering "
            "selectivity but may increase cost or reject more results; interpret the direction "
            "according to the selected backend definition."
        )
    elif any(token in lower for token in ("max_steps", "max_iterations", "max_cycles", "max_records", "max_")):
        impact = (
            "A larger bound allows more search, iteration, or returned records and can increase "
            "runtime and output size; it does not guarantee convergence or higher scientific quality."
        )
    elif any(token in lower for token in ("grid", "points", "bins")):
        impact = (
            "Finer or more numerous samples can reduce discretization error or increase resolution "
            "at greater runtime, memory, and output cost; convergence should be checked."
        )
    elif "cutoff" in lower or "range" in lower:
        impact = (
            "Changing this boundary changes which interactions, structures, records, or numerical "
            "region are included and may also change cost."
        )
    elif "temperature" in lower or "pressure" in lower:
        impact = "Changing this thermodynamic state variable changes the physical ensemble or state evaluated."
    elif "seed" in lower:
        impact = "Changing the seed changes a stochastic realization; fixed seeds improve reproducibility."
    elif lower.startswith(("include_", "use_", "allow_", "fix_", "keep_", "require_", "relax_", "generate_")):
        impact = "Toggling this flag enables or disables the named behavior and can change both results and cost."
    elif section == "method_spec":
        impact = "Changing this value changes the scientific model or representation used by the backend."
    elif section == "inputs":
        impact = "Changing this value changes the scientific data supplied to the Action."
    else:
        impact = "Changing this value changes the named Action behavior or numerical result."

    value: dict[str, Any] = {"description": description, "impact": impact}
    if not required:
        value["default"] = None
    return value
