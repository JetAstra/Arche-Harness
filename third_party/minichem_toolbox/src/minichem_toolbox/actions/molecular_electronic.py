"""ActionSpec declarations for molecular_electronic."""

from __future__ import annotations

from .._spec_builders import action as _action


ACTION_SPECS = (
    _action(
        "generate_gaussian_code",
        "molecular_electronic",
        (
            "Optional, non-default route drafter: draft one Gaussian route line from the "
            "natural-language requirement in "
            "inputs.user_input. The wrapper only checks route text and safe shape; it does not "
            "perform scientific validation, generate a Gaussian input deck, run Gaussian, or "
            "produce scientific evidence."
        ),
        "GaussianCodeResult",
        ("gaussian_codegen",),
        ("user_input",),
        input_description=(
            "Optional, non-default route-drafting helper. Natural-language Gaussian calculation "
            "requirement in inputs.user_input; the Action returns one route line after route-text "
            "and safe-shape checks, does not accept "
            "explicit method or settings, generate a Gaussian input deck, run Gaussian, or "
            "produce scientific evidence"
        ),
        requires_network=True,
    ),
    _action(
            "calculate_energy",
            "molecular_electronic",
            "Calculate one molecular or non-periodic scalar energy with the exact software and method selected by the agent.",
            "EnergyResult",
            ("xtb", "pyscf", "psi4", "tblite", "gpaw", "nwchem", "openmolcas", "mace", "chgnet", "deepmd", "orca", "gaussian", "gamess", "ase_emt"),
            ("structure",),
            input_description="non-periodic AtomicStructure",
            batch_safe=True,
        ),
    _action(
            "calculate_forces",
            "molecular_electronic",
            "Calculate atomic forces for one non-periodic structure.",
            "ForceResult",
            ("xtb", "pyscf", "tblite", "gpaw", "nwchem", "orca", "mace", "chgnet", "deepmd", "ase_emt"),
            ("structure",),
            input_description="one non-periodic AtomicStructure",
        ),
    _action(
            "calculate_hessian",
            "molecular_electronic",
            "Calculate one molecular Hessian without deriving modes, spectra, or thermochemistry.",
            "Hessian",
            (
                "xtb", "pyscf", "psi4", "tblite", "nwchem", "orca", "gaussian",
                "mace", "chgnet", "deepmd", "ase_emt",
            ),
            ("structure",),
            input_description="AtomicStructure",
            batch_safe=True,
        ),
    _action(
            "optimize_geometry",
            "molecular_electronic",
            "Optimize one non-periodic geometry and return the optimized structure only as the primary result.",
            "AtomicStructure",
            ("xtb", "tblite", "gpaw", "mace", "chgnet", "deepmd", "rdkit_mmff", "fairchem_uma", "orca", "gaussian", "gamess", "ase_emt", "geometric", "sella"),
            ("structure",),
            ("constraints",),
            input_description="AtomicStructure plus explicit convergence and optional constraints",
            batch_safe=True,
        ),
    _action(
            "calculate_dipole_moment",
            "molecular_electronic",
            "Calculate one molecular dipole moment with an explicitly chosen electronic method.",
            "DipoleResult",
            ("xtb", "tblite", "pyscf", "psi4", "nwchem", "openmolcas", "orca", "gaussian", "gamess"),
            ("structure",),
            input_description="AtomicStructure",
        ),
    _action(
            "calculate_atomic_charges",
            "molecular_electronic",
            "Calculate electronic-structure population-analysis charges without attaching force-field parameters.",
            "AtomicChargeResult",
            ("xtb", "pyscf", "psi4", "nwchem", "openmolcas", "multiwfn", "orca"),
            ("structure",),
            input_description="AtomicStructure for calculation backends or a compatible wavefunction/ElectronicState Artifact for analysis backends",
        ),
    _action(
            "calculate_orbitals",
            "molecular_electronic",
            "Calculate orbital energies, occupations, and optional coefficient artifacts.",
            "OrbitalResult",
            ("pyscf", "psi4", "openmolcas", "orca"),
            ("structure",),
            input_description="AtomicStructure or compatible ElectronicState Artifact",
        ),
    _action(
            "calculate_correlated_electron_density",
            "molecular_electronic",
            (
                "Calculate and retain one explicitly selected molecular electron density, "
                "including SCF/DFT, relaxed MP2 or double-hybrid, and unrelaxed CCSD density "
                "sources, without silently substituting an unavailable density model."
            ),
            "ElectronDensityResult",
            ("orca",),
            ("structure",),
            input_description=(
                "one molecular AtomicStructure plus an explicit ORCA method, basis, density "
                "source, SCF controls, charge/spin, and resource limits"
            ),
        ),
    _action(
            "export_electron_density_grid",
            "molecular_electronic",
            (
                "Export a previously calculated ORCA electron density to an explicitly "
                "selected WFN, WFX, or cube representation while preserving the named "
                "density source in provenance."
            ),
            "ElectronDensityExportResult",
            ("orca",),
            ("electron_density",),
            input_description=(
                "ElectronDensityResult from calculate_correlated_electron_density plus an "
                "explicit density source and output format"
            ),
        ),
    _action(
            "calculate_electron_isodensity_surface",
            "molecular_electronic",
            (
                "Calculate molecular electron-isodensity surface area and enclosed volume "
                "for an explicit list of density cutoffs using one supplied wavefunction or "
                "electron-density grid."
            ),
            "ElectronIsodensitySurfaceResult",
            ("multiwfn",),
            ("density_file",),
            input_description=(
                "one WFN/WFX/FCHK/MWFN/Molden wavefunction or cube density grid plus explicit "
                "cutoff values and surface-grid spacing in bohr"
            ),
        ),
    _action(
            "analyze_non_covalent_interactions",
            "molecular_electronic",
            (
                "Calculate a wavefunction-based NCI/RDG grid, sign(lambda2)rho grid, "
                "scatter table, and atom-pair-seeded RDG critical-point data for one "
                "explicit molecular wavefunction without inferring selectivity."
            ),
            "NonCovalentInteractionAnalysisResult",
            ("multiwfn",),
            ("wavefunction_file",),
            input_description=(
                "one molecular FCH/FCHK/WFN/WFX/MWFN/Molden/47 wavefunction containing "
                "basis/orbital information; a Gaussian binary CHK or density-only cube "
                "is not compatible"
            ),
        ),
    _action(
            "calculate_bond_orders",
            "molecular_electronic",
            "Calculate atom-pair electronic bond-order indices using one explicitly selected population-analysis backend.",
            "BondOrderResult",
            ("xtb", "multiwfn", "orca", "gaussian"),
            ("structure",),
            input_description="non-periodic AtomicStructure or compatible wavefunction Artifact and explicit population definition/minimum reported bond order",
        ),
    _action(
            "calculate_excited_states",
            "molecular_electronic",
            "Calculate a bounded set of vertical electronic excited states without constructing a broadened spectrum or propagating dynamics.",
            "ExcitedStateResult",
            ("pyscf", "orca", "gaussian"),
            ("structure",),
            input_description="non-periodic AtomicStructure plus explicit ground-state and excited-state methods",
        ),
    _action(
            "analyze_electron_density_topology",
            "molecular_electronic",
            "Locate and characterize critical points in one supplied molecular or periodic electron-density field without generating that field or integrating atomic basins.",
            "ElectronDensityTopologyResult",
            ("critic2",),
            ("density_file",),
            ("structure_file",),
            input_description=(
                "electron-density grid or wavefunction file plus an optional separate structure file; "
                "the caller explicitly selects molecular/periodic interpretation, field format, "
                "interpolation, critical-point classes, tolerances, and seeding strategy"
            ),
        ),
    _action(
            "calculate_atomic_basin_properties",
            "molecular_electronic",
            "Integrate population, Laplacian, and available volume properties over atomic or attractor basins in one supplied scalar-field grid using an explicitly selected partition algorithm.",
            "AtomicBasinPropertyResult",
            ("critic2",),
            ("density_file",),
            ("structure_file",),
            input_description=(
                "electron-density or scalar-field grid plus optional separate structure; the caller "
                "selects the Critic2 YT, Henkelman BADER, Hirshfeld, or Voronoi partition"
            ),
        ),
    _action(
            "calculate_bader_charges",
            "molecular_electronic",
            "Calculate atomic Bader charges from one supplied electron-density grid using an explicitly selected Yu-Trinkle or Henkelman grid partition.",
            "AtomicChargeResult",
            ("critic2",),
            ("density_file",),
            ("structure_file",),
            input_description=(
                "electron-density grid plus optional separate structure and explicit grid partition, "
                "attractor-assignment, filtering, and reporting settings"
            ),
        ),
    _action(
            "derive_vibrational_modes",
            "molecular_electronic",
            "Derive frequencies and normal modes from an existing Hessian and structure.",
            "FrequencyResult",
            ("internal_vibrations",),
            ("hessian", "structure"),
            input_description="Hessian Artifact and matching AtomicStructure",
            selection_policy="internal_deterministic",
        ),
    _action(
            "derive_ir_spectrum",
            "molecular_electronic",
            "Construct an IR spectrum from vibration results that already contain intensities.",
            "SpectrumResult",
            ("internal_spectroscopy",),
            ("vibrations",),
            input_description="FrequencyResult with intensities",
            selection_policy="internal_deterministic",
        ),
    _action(
            "derive_uv_vis_spectrum",
            "molecular_electronic",
            "Construct a deterministic broadened UV/visible spectrum from supplied transition energies and oscillator strengths.",
            "SpectrumResult",
            ("internal_spectroscopy",),
            ("excited_states",),
            input_description="ExcitedStateResult containing energy_ev and oscillator_strength for each state",
            selection_policy="internal_deterministic",
        ),
    _action(
            "derive_raman_spectrum",
            "molecular_electronic",
            "Construct a deterministic Raman spectrum from explicitly supplied vibrational frequencies and Raman activities without running a quantum calculation or rendering a plot.",
            "SpectrumResult",
            ("internal_spectroscopy",),
            ("spectral_data",),
            input_description="structured Raman line data or ParsedQuantumChemistryResult containing frequencies and Raman activities",
            selection_policy="internal_deterministic",
        ),
    _action(
            "derive_roa_spectrum",
            "molecular_electronic",
            "Construct a deterministic signed Raman optical activity spectrum from supplied line positions and ROA activities without inferring missing chiroptical data.",
            "SpectrumResult",
            ("internal_spectroscopy",),
            ("spectral_data",),
            input_description="structured ROA line data or parsed result containing signed ROA activities",
            selection_policy="internal_deterministic",
        ),
    _action(
            "derive_vcd_spectrum",
            "molecular_electronic",
            "Construct a deterministic signed vibrational circular dichroism spectrum from supplied frequencies and rotatory strengths.",
            "SpectrumResult",
            ("internal_spectroscopy",),
            ("spectral_data",),
            input_description="structured VCD line data or parsed result containing signed VCD rotatory strengths",
            selection_policy="internal_deterministic",
        ),
    _action(
            "derive_ecd_spectrum",
            "molecular_electronic",
            "Construct a deterministic signed electronic circular dichroism spectrum from supplied transition energies and rotatory strengths without substituting oscillator strengths.",
            "SpectrumResult",
            ("internal_spectroscopy",),
            ("excited_states",),
            input_description="electronic transitions containing energies and signed rotatory strengths",
            selection_policy="internal_deterministic",
        ),
    _action(
            "derive_nmr_spectrum",
            "molecular_electronic",
            "Construct a deterministic NMR stick and broadened spectrum from explicit shifts or shieldings and an explicit reference when shieldings require conversion.",
            "SpectrumResult",
            ("internal_spectroscopy",),
            ("spectral_data",),
            input_description="chemical shifts, isotropic shieldings, or shielding tensors with explicit reference when needed",
            selection_policy="internal_deterministic",
        ),
    _action(
            "derive_thermochemistry",
            "molecular_electronic",
            "Derive thermochemical quantities from supplied electronic energy and frequencies; no optimization or Hessian is hidden.",
            "ThermochemistryResult",
            ("internal_thermochemistry", "goodvibes"),
            (),
            (
                "energy",
                "frequencies",
                "structure",
                "output_file",
                "single_point_output_file",
            ),
            input_description=(
                "Backend-specific contract: internal_thermochemistry requires EnergyResult plus "
                "FrequencyResult (with its matching structure embedded, or structure supplied "
                "separately); goodvibes requires one compatible quantum output_file Artifact "
                "and accepts an explicit single_point_output_file when --spc correction is used"
            ),
        ),
    _action(
            "scan_thermochemistry_temperature",
            "molecular_electronic",
            "Evaluate thermochemical quantities for supplied quantum outputs at each explicitly listed temperature without rerunning electronic-structure calculations.",
            "ThermochemistryTemperatureSeries",
            ("goodvibes",),
            ("output_files", "temperatures_kelvin"),
            input_description=(
                "one or more completed Gaussian, ORCA, NWChem, Q-Chem, xTB, or ASE-extxyz "
                "outputs plus an explicit non-empty temperature list"
            ),
        ),
    _action(
            "analyze_thermochemical_ensemble",
            "molecular_electronic",
            "Calculate per-structure thermochemistry and Boltzmann populations for an explicitly supplied conformer or structure ensemble.",
            "ThermochemicalEnsembleResult",
            ("goodvibes",),
            ("output_files",),
            input_description=(
                "two or more compatible completed quantum outputs representing the exact ensemble "
                "chosen by the caller"
            ),
        ),
    _action(
            "validate_thermochemistry_inputs",
            "molecular_electronic",
            "Check supplied quantum outputs for thermochemistry compatibility, calculation consistency, frequency issues, and possible duplicate structures.",
            "ThermochemistryValidationReport",
            ("goodvibes",),
            ("output_files",),
            input_description="one or more compatible completed quantum-chemistry output files",
        ),
)
