---
software_id: xtb
versions: ["6.7.1"]
topics: ["common-tasks", "inputs", "outputs", "convergence"]
aliases: ["xTB", "xtb"]
inputs: ["structure.xyz", "optional xcontrol file"]
outputs: ["stdout.log", "xtbopt.xyz", "hessian", "charges", "wbo", "trajectory files"]
last_smoke_tested: "2026-07-28"
generated_from: minichem_toolbox/config/native_software_guides.yaml + minichem_toolbox/config/native_software_example_contracts.yaml
---
# xTB Common Tasks

## Appropriate calculation families
- **Single Point**: author the method-specific input, stage every dependency, and declare the outputs needed for interpretation.
- **Geometry Optimization**: author the method-specific input, stage every dependency, and declare the outputs needed for interpretation.
- **Frequency**: author the method-specific input, stage every dependency, and declare the outputs needed for interpretation.
- **Molecular Dynamics**: author the method-specific input, stage every dependency, and declare the outputs needed for interpretation.
- **Solvation**: author the method-specific input, stage every dependency, and declare the outputs needed for interpretation.
- **Properties**: author the method-specific input, stage every dependency, and declare the outputs needed for interpretation.

## Minimum input responsibilities
- `structure.xyz`
- `optional xcontrol file`

A minimum runnable input must still specify every scientifically material quantity: molecular or periodic structure, charge and spin where applicable, model or Hamiltonian, numerical controls, boundary conditions, task type, and requested outputs. Workflow programs additionally require their database, model, or upstream-calculation references.

## Expected output families
- `stdout.log`
- `xtbopt.xyz`
- `hessian`
- `charges`
- `wbo`
- `trajectory files`

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
- `normal termination of xtb`

## Scientific convergence notes
Require SCC convergence and task-specific geometry, Hessian, or dynamics completion. Verify charge, UHF count, and solvent model.

## Version-specific caution
These mechanics target the installed `6.7.1` environment. Verify keywords and file formats against the official references before reusing an input written for another release.

## Minimal-example policy
The tested file under `examples/interface_smoke/` verifies the configured command route. It does not choose a paper-specific method. For scientific work, start from the smallest official example for the intended calculation family, replace all structures and methods explicitly, run the toolbox validator, and retain the complete inputs and outputs as provenance.

## Command: `xtb`
- Synopsis: `xtb structure.xyz --gfn <level> [--sp|--opt|--hess|--md] [explicit options]`.
- Input mode: `arguments`.
- Declared example inputs: `structure.xyz`.
- Declared example outputs: stdout/stderr or task-dependent outputs.
- Example resources: `{'cpu_cores': 1, 'memory_mb': 2048, 'gpu_count': 0}`.
- Output behavior: Writes primary text to stdout and mode-specific files in the job directory.
- Caution: Choose the xTB level, charge, unpaired electrons, solvent, and requested calculation mode explicitly.
