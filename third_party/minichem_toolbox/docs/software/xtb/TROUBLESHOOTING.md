---
software_id: xtb
versions: ["6.7.1"]
topics: ["troubleshooting", "errors", "preflight"]
aliases: ["xTB", "xtb"]
inputs: ["structure.xyz", "optional xcontrol file"]
outputs: ["stdout.log", "xtbopt.xyz", "hessian", "charges", "wbo", "trajectory files"]
last_smoke_tested: "2026-07-28"
generated_from: minichem_toolbox/config/native_software_guides.yaml + minichem_toolbox/config/native_software_example_contracts.yaml
---
# xTB Troubleshooting

## Diagnose in this order
1. Confirm that `inspect_software` resolves the expected executable and version.
2. Read the first fatal message in `stderr.log` or the primary software output; later messages are often consequences.
3. Verify staged target names, input-relative paths, file encodings, line endings, and required blank sections.
4. Verify task syntax against the installed version, then check method and data compatibility.
5. Compare internal MPI, thread, memory, scratch, and GPU settings with the declared job resources.
6. Only after syntax and staging pass, investigate numerical convergence or increase resources.

## Known failures and repairs
| Symptom | Likely cause | Corrective action |
|---|---|---|
| SCC did not converge | electronic state, geometry, charge, spin, or model is unsuitable | validate charge and UHF, then inspect SCC settings |
| coordinate file could not be read | XYZ formatting or path is wrong | stage one valid XYZ with matching atom count |
| geometry optimization failed | structure or optimizer did not converge | inspect xtbopt.log and do not use an unconverged geometry |

## Path and staging failures
A source file existing in the benchmark workspace does not make it visible to the native process. Every dependency must be declared in `staged_inputs`. The content of an input deck must reference the staged `target_path`, not its original workspace path. Fixed-name programs are case-sensitive. Never assume the process starts in the task workspace.

## Resource failures
Set OMP_NUM_THREADS or --parallel consistent with cpu_cores; most small-molecule jobs are modest-memory.
If the Supervisor reports `memory_limit_exceeded`, reduce software parallelism or request a justified larger total allocation. If it reports timeout, inspect whether the software was progressing and whether the requested task can finish within the remaining evaluation lifetime. Resource increases do not repair malformed input.

## False-success prevention
Do not accept a zero exit code when the main output contains `ERROR`, `FATAL`, an abort marker, non-convergence, or missing-result diagnostics. Likewise, do not promote an electronic convergence marker to geometry, frequency, transition-state, dynamics, or projection success. Preserve the independent status axes in the final report.

## Pre-submission checklist
- Installed executable and version inspected.
- Official syntax checked for the intended calculation family.
- Input is complete and uses English/ASCII-safe filenames where possible.
- Every referenced file is staged to the exact target name.
- stdin and argument modes match the Catalog contract.
- Charge, multiplicity, periodicity, units, atom ordering, and upstream provenance are consistent.
- CPU, total memory, per-rank/per-core memory, GPU, scratch, and walltime agree.
- `validate_native_job` returns no errors.
- Required outputs and task-specific success criteria are declared before execution.
