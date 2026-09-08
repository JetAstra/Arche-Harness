---
software_id: multiwfn
versions: ["2026.7.15"]
topics: ["common-tasks", "inputs", "outputs", "convergence"]
aliases: ["Multiwfn", "multiwfn"]
inputs: ["wavefunction file", "commands.txt"]
outputs: ["stdout.log", "exported grids", "tables", "images or structure files"]
last_smoke_tested: "2026-07-28"
generated_from: minichem_toolbox/config/native_software_guides.yaml + minichem_toolbox/config/native_software_example_contracts.yaml
---
# Multiwfn Common Tasks

## Appropriate calculation families
- **Density Grids**: author the method-specific input, stage every dependency, and declare the outputs needed for interpretation.
- **Surface Properties**: author the method-specific input, stage every dependency, and declare the outputs needed for interpretation.
- **Orbital Analysis**: author the method-specific input, stage every dependency, and declare the outputs needed for interpretation.
- **Population Analysis**: author the method-specific input, stage every dependency, and declare the outputs needed for interpretation.
- **Topology**: author the method-specific input, stage every dependency, and declare the outputs needed for interpretation.
- **Spectra**: author the method-specific input, stage every dependency, and declare the outputs needed for interpretation.

## Minimum input responsibilities
- `wavefunction file`
- `commands.txt`

A minimum runnable input must still specify every scientifically material quantity: molecular or periodic structure, charge and spin where applicable, model or Hamiltonian, numerical controls, boundary conditions, task type, and requested outputs. Workflow programs additionally require their database, model, or upstream-calculation references.

## Expected output families
- `stdout.log`
- `exported grids`
- `tables`
- `images or structure files`

Only collect outputs produced by the same job or by explicitly linked parent jobs. Do not combine checkpoints, force constants, trajectories, pseudopotentials, wavefunctions, or databases from unrelated calculations.

## State-specific validation
| State | Required evidence | Not sufficient |
|---|---|---|
| Process completed | Exit code zero and Supervisor did not cancel or time out | A submitted job ID |
| Software normal | Program-specific normal marker and absence of fatal diagnostics | Exit code zero alone |
| Electronic convergence | Requested SCF or electronic tolerance reached | A printed energy from an unconverged cycle |
| Geometry convergence | Optimization stopping criteria reached for the requested degrees of freedom | SCF convergence at the last geometry |
| Frequency completion | Hessian/frequencies finished and modes were parsed | Optimization completion alone |
| Transition-state candidate | Optimization converged and exactly the intended imaginary mode was verified | One negative number without mode inspection |
| Dynamics completion | Requested steps completed with acceptable stability diagnostics | Creation of a partial trajectory |
| Artifact validity | Required files exist, are non-empty, and can be parsed | Files with expected names only |

## Software-specific end markers
- `Multiwfn finished`
- `Thank you for using Multiwfn`

## Scientific convergence notes
Multiwfn is post-processing; require the requested exported artifact and validate upstream wavefunction level and completeness.

The atomic `analyze_non_covalent_interactions` Action accepts one explicit
`FCH/FCHK/WFN/WFX/MWFN/Molden/47` molecular wavefunction and emits the raw
`sign(lambda2)rho` and RDG cube grids, the complete five-column scatter table,
atom-pair-seeded RDG critical-point properties, and a structured audit summary.
A Gaussian binary `.chk` must first be converted with `formchk`; a density-only
cube is rejected because it does not contain the wavefunction information needed
for this NCI/RDG calculation. Use matched electronic methods and identical grid
settings when comparing candidates. NCI/RDG evidence describes interaction
regions and does not, by itself, prove transition-state connectivity, relative
free energy, stereoselectivity, or the causal origin of selectivity.

## Version-specific caution
These mechanics target the installed `2026.7.15` environment. Verify keywords and file formats against the official references before reusing an input written for another release.

## Minimal-example policy
The tested file under `examples/interface_smoke/` verifies the configured command route. It does not choose a paper-specific method. For scientific work, start from the smallest official example for the intended calculation family, replace all structures and methods explicitly, run the toolbox validator, and retain the complete inputs and outputs as provenance.

## Command: `Multiwfn_noGUI`
- Synopsis: `Multiwfn_noGUI wavefunction_file < commands.txt`.
- Input mode: `arguments_and_stdin_file`.
- Declared example inputs: `commands.txt`, `wavefunction.fchk`.
- Declared example outputs: stdout/stderr or task-dependent outputs.
- Example resources: `{'cpu_cores': 1, 'memory_mb': 2048, 'gpu_count': 0}`.
- Output behavior: Writes interactive-menu output to stdout and selected analysis files to the job directory.
- Caution: Set stdin_file to commands.txt; the Agent must author every menu selection and numerical parameter.
