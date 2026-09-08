---
software_id: goodvibes
versions: ["4.3.0"]
topics: ["common-tasks", "inputs", "outputs", "convergence"]
aliases: ["GoodVibes", "goodvibes"]
inputs: ["Gaussian", "ORCA", "NWChem", "Q-Chem", "xTB", "or ASE frequency output"]
outputs: ["console table", "JSON or CSV result", "optional PES and plots"]
last_smoke_tested: "2026-07-28"
generated_from: minichem_toolbox/config/native_software_guides.yaml + minichem_toolbox/config/native_software_example_contracts.yaml
---
# GoodVibes Common Tasks

## Appropriate calculation families
- **Single-Temperature Thermochemistry**: author the method-specific input, stage every dependency, and declare the outputs needed for interpretation.
- **Temperature Scans**: author the method-specific input, stage every dependency, and declare the outputs needed for interpretation.
- **Concentration Corrections**: author the method-specific input, stage every dependency, and declare the outputs needed for interpretation.
- **Selectivity**: author the method-specific input, stage every dependency, and declare the outputs needed for interpretation.
- **Reaction Profiles**: author the method-specific input, stage every dependency, and declare the outputs needed for interpretation.

## Minimum input responsibilities
- `Gaussian`
- `ORCA`
- `NWChem`
- `Q-Chem`
- `xTB`
- `or ASE frequency output`

A minimum runnable input must still specify every scientifically material quantity: molecular or periodic structure, charge and spin where applicable, model or Hamiltonian, numerical controls, boundary conditions, task type, and requested outputs. Workflow programs additionally require their database, model, or upstream-calculation references.

## Expected output families
- `console table`
- `JSON or CSV result`
- `optional PES and plots`

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
- `GoodVibes`
- `Structure`

## Scientific convergence notes
Verify that every input has a completed frequency calculation, correct stationary-point imaginary modes, consistent temperature, concentration, solvent convention, and energy level.

## Version-specific caution
These mechanics target the installed `4.3.0` environment. Verify keywords and file formats against the official references before reusing an input written for another release.

## Minimal-example policy
The tested file under `examples/interface_smoke/` verifies the configured command route. It does not choose a paper-specific method. For scientific work, start from the smallest official example for the intended calculation family, replace all structures and methods explicitly, run the toolbox validator, and retain the complete inputs and outputs as provenance.

## Command: `goodvibes`
- Synopsis: `goodvibes OUTPUT... --temp K [state/scaling/qh options] [analysis option] --json result.json`.
- Input mode: `arguments`.
- Declared example inputs: `output.log`.
- Declared example outputs: `result.json`.
- Example resources: `{'cpu_cores': 1, 'memory_mb': 2048, 'gpu_count': 0}`.
- Output behavior: Writes a GoodVibes_NAME.dat report and, when requested, structured JSON/CSV/Parquet plus plots or XYZ files in the isolated job directory.
- Caution: Temperature, concentration/standard state, frequency scaling, quasi-harmonic treatment, and solvation corrections are scientific choices.
- Caution: Use -v/--vscal for vibrational scaling. --fs is the quasi-harmonic entropy cutoff in cm-1; it is not a scale factor.
- Caution: Omit -v only when deliberately choosing GoodVibes level-of-theory auto lookup; record that choice in the task trace.
- Caution: Use --boltz energy or --boltz gibbs for ensemble populations, --dedup with explicit cutoffs for duplicate filtering, and --check for consistency diagnostics.
- Caution: Use repeated --label NAME=PATTERN or --selectivity labels.yaml for N-way selectivity; prefer the non-deprecated interfaces over --ee.
- Caution: Use --pes profile.yaml for a reaction profile. The YAML defines pathways, species file membership, zero references, and output units; --nogconf and --lowest-only select the conformer treatment.
- Caution: Use --ti START,END,STEP for the native integer-grid temperature report. The predefined temperature-scan Action instead accepts an explicit temperature list and returns structured results at every point.
- Caution: --json/--export writes schema 1.0 in GoodVibes 4.3.0, but upstream describes the schema as preview before v5; consumers should retain schema_version and goodvibes_version.
- Caution: result.json is an output and must not be listed in staged_inputs. For --spc SUFFIX, stage matching safe names such as frequency.log and frequency_SUFFIX.out, and pass SUFFIX without the separator underscore (for frequency_DLPNO.out use --spc DLPNO, not --spc _DLPNO); the derive_thermochemistry Action performs this pairing when single_point_output_file is supplied.
