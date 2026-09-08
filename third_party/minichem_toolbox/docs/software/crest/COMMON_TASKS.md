---
software_id: crest
versions: ["3.0.2"]
topics: ["common-tasks", "inputs", "outputs", "convergence"]
aliases: ["CREST", "crest"]
inputs: ["input.xyz"]
outputs: ["crest_conformers.xyz", "crest.energies", "protonated.xyz", "deprotonated.xyz", "tautomers.xyz"]
last_smoke_tested: "2026-07-28"
generated_from: minichem_toolbox/config/native_software_guides.yaml + minichem_toolbox/config/native_software_example_contracts.yaml
---
# CREST Common Tasks

## Appropriate calculation families
- **Conformer Search**: author the method-specific input, stage every dependency, and declare the outputs needed for interpretation.
- **Protonation**: author the method-specific input, stage every dependency, and declare the outputs needed for interpretation.
- **Deprotonation**: author the method-specific input, stage every dependency, and declare the outputs needed for interpretation.
- **Tautomerization**: author the method-specific input, stage every dependency, and declare the outputs needed for interpretation.
- **Ensemble Screening**: author the method-specific input, stage every dependency, and declare the outputs needed for interpretation.

## Minimum input responsibilities
- `input.xyz`

A minimum runnable input must still specify every scientifically material quantity: molecular or periodic structure, charge and spin where applicable, model or Hamiltonian, numerical controls, boundary conditions, task type, and requested outputs. Workflow programs additionally require their database, model, or upstream-calculation references.

## Expected output families
- `crest_conformers.xyz`
- `crest.energies`
- `protonated.xyz`
- `deprotonated.xyz`
- `tautomers.xyz`

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
- `CREST terminated normally`

## Scientific convergence notes
Require the mode-specific final ensemble file in addition to normal termination.

## Version-specific caution
These mechanics target the installed `3.0.2` environment. Verify keywords and file formats against the official references before reusing an input written for another release.

## Minimal-example policy
The tested file under `examples/interface_smoke/` verifies the configured command route. It does not choose a paper-specific method. For scientific work, start from the smallest official example for the intended calculation family, replace all structures and methods explicitly, run the toolbox validator, and retain the complete inputs and outputs as provenance.

## Command: `crest`
- Synopsis: `crest input.xyz --gfn2 --T <threads> [sampling options]`.
- Input mode: `arguments`.
- Declared example inputs: `input.xyz`.
- Declared example outputs: stdout/stderr or task-dependent outputs.
- Example resources: `{'cpu_cores': 4, 'memory_mb': 2048, 'gpu_count': 0}`.
- Output behavior: Writes CREST result files such as crest_conformers.xyz in the job directory.
- Caution: Charge, spin, solvent, energy method, and sampling controls must be supplied explicitly when scientifically required.
