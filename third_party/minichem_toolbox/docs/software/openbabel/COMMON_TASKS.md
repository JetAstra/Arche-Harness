---
software_id: openbabel
versions: ["3.1.0"]
topics: ["common-tasks", "inputs", "outputs", "convergence"]
aliases: ["Open Babel", "openbabel"]
inputs: ["molecular input file"]
outputs: ["converted molecular file", "console summary"]
last_smoke_tested: "2026-07-28"
generated_from: minichem_toolbox/config/native_software_guides.yaml + minichem_toolbox/config/native_software_example_contracts.yaml
---
# Open Babel Common Tasks

## Appropriate calculation families
- **Format Conversion**: author the method-specific input, stage every dependency, and declare the outputs needed for interpretation.
- **Hydrogen Addition**: author the method-specific input, stage every dependency, and declare the outputs needed for interpretation.
- **2D To 3D Conversion**: author the method-specific input, stage every dependency, and declare the outputs needed for interpretation.
- **Filtering**: author the method-specific input, stage every dependency, and declare the outputs needed for interpretation.
- **Descriptor Calculation**: author the method-specific input, stage every dependency, and declare the outputs needed for interpretation.

## Minimum input responsibilities
- `molecular input file`

A minimum runnable input must still specify every scientifically material quantity: molecular or periodic structure, charge and spin where applicable, model or Hamiltonian, numerical controls, boundary conditions, task type, and requested outputs. Workflow programs additionally require their database, model, or upstream-calculation references.

## Expected output families
- `converted molecular file`
- `console summary`

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
- `molecule converted`

## Scientific convergence notes
Validate molecule count, atom count, bond orders, charge, stereochemistry, and coordinates in the output; conversion exit code alone is insufficient.

## Version-specific caution
These mechanics target the installed `3.1.0` environment. Verify keywords and file formats against the official references before reusing an input written for another release.

## Minimal-example policy
The tested file under `examples/interface_smoke/` verifies the configured command route. It does not choose a paper-specific method. For scientific work, start from the smallest official example for the intended calculation family, replace all structures and methods explicitly, run the toolbox validator, and retain the complete inputs and outputs as provenance.

## Command: `obabel`
- Synopsis: `obabel -i<input_format> input -o<output_format> -O output [options]`.
- Input mode: `arguments`.
- Declared example inputs: `input.xyz`.
- Declared example outputs: `output.sdf`.
- Example resources: `{'cpu_cores': 1, 'memory_mb': 2048, 'gpu_count': 0}`.
- Output behavior: Writes the path supplied after -O; diagnostics are written to stderr.
- Caution: Select input and output formats explicitly; do not rely on filename inference in benchmark calls.
