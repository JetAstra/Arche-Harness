---
software_id: pysisyphus
versions: ["1.0.0"]
topics: ["common-tasks", "inputs", "outputs", "convergence"]
aliases: ["pysisyphus", "pysisyphus"]
inputs: ["config.yaml", "referenced geometry and calculator files"]
outputs: ["optimization log", "trajectory", "final geometry", "Hessian or path files"]
last_smoke_tested: "2026-07-28"
generated_from: minichem_toolbox/config/native_software_guides.yaml + minichem_toolbox/config/native_software_example_contracts.yaml
---
# pysisyphus Common Tasks

## Appropriate calculation families
- **Minimum Optimization**: author the method-specific input, stage every dependency, and declare the outputs needed for interpretation.
- **Transition State**: author the method-specific input, stage every dependency, and declare the outputs needed for interpretation.
- **Irc**: author the method-specific input, stage every dependency, and declare the outputs needed for interpretation.
- **Neb**: author the method-specific input, stage every dependency, and declare the outputs needed for interpretation.
- **Growing String**: author the method-specific input, stage every dependency, and declare the outputs needed for interpretation.

## Minimum input responsibilities
- `config.yaml`
- `referenced geometry and calculator files`

A minimum runnable input must still specify every scientifically material quantity: molecular or periodic structure, charge and spin where applicable, model or Hamiltonian, numerical controls, boundary conditions, task type, and requested outputs. Workflow programs additionally require their database, model, or upstream-calculation references.

## Expected output families
- `optimization log`
- `trajectory`
- `final geometry`
- `Hessian or path files`

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
- `Converged!`
- `All done`

## Scientific convergence notes
Require optimizer or path convergence and verify stationary-point imaginary modes and endpoint connectivity where applicable.

## Version-specific caution
These mechanics target the installed `1.0.0` environment. Verify keywords and file formats against the official references before reusing an input written for another release.

## Minimal-example policy
The tested file under `examples/interface_smoke/` verifies the configured command route. It does not choose a paper-specific method. For scientific work, start from the smallest official example for the intended calculation family, replace all structures and methods explicitly, run the toolbox validator, and retain the complete inputs and outputs as provenance.

## Command: `pysis`
- Synopsis: `pysis config.yaml`.
- Input mode: `arguments`.
- Declared example inputs: `config.yaml`.
- Declared example outputs: stdout/stderr or task-dependent outputs.
- Example resources: `{'cpu_cores': 1, 'memory_mb': 2048, 'gpu_count': 0}`.
- Output behavior: Writes optimization, pathway, or dynamics outputs selected by config.yaml.
- Caution: The templates document input mechanics, not recommended chemistry. Replace charge, multiplicity, calculator, method, cores, image count, optimizer, thresholds, and cycle limits deliberately for the system at hand.
- Caution: For a two-endpoint path, stage reactant.xyz and product.xyz and reference those exact target names from geom.fn. Do not invent endpoints or cos.images keys.
- Caution: A converged chain-of-states path is not automatically a validated transition state; explicitly inspect its highest-energy image and run the chosen TS, Hessian/mode, and connectivity checks when the scientific claim requires them.
