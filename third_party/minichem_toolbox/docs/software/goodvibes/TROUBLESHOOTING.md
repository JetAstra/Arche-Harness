---
software_id: goodvibes
versions: ["4.3.0"]
topics: ["troubleshooting", "errors", "preflight"]
aliases: ["GoodVibes", "goodvibes"]
inputs: ["Gaussian", "ORCA", "NWChem", "Q-Chem", "xTB", "or ASE frequency output"]
outputs: ["console table", "JSON or CSV result", "optional PES and plots"]
last_smoke_tested: "2026-07-28"
generated_from: minichem_toolbox/config/native_software_guides.yaml + minichem_toolbox/config/native_software_example_contracts.yaml
---
# GoodVibes Troubleshooting

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
| no frequencies or thermochemistry found | the upstream output is incomplete or unsupported | stage the complete final log and verify frequency termination |
| inconsistent level of theory | files mix methods or scaling factors | separate groups or explicitly justify the correction protocol |
| None value or parse failure | log formatting or missing energy terms broke extraction | inspect the exact file and do not silently drop it |
| Specify at least one output file although a path was supplied | shell-style glob characters such as brackets changed GoodVibes filename matching | use ASCII-safe staged target names; the preset Action stages safe names automatically |
| ModuleNotFoundError for goodvibes.gaussian or goodvibes.orca | those modules are not public GoodVibes 4.3.0 APIs | use the CLI, preset Action, or documented goodvibes.api functions instead of guessing parser modules |
| NumPy binary or API mismatch after importing GoodVibes | site-packages from another Python runtime were inserted manually | select the configured GoodVibes runtime and never splice sys.path or PYTHONPATH across environments |

## Path and staging failures
A source file existing in the benchmark workspace does not make it visible to the native process. Every dependency must be declared in `staged_inputs`. The content of an input deck must reference the staged `target_path`, not its original workspace path. Fixed-name programs are case-sensitive. Never assume the process starts in the task workspace.

## Resource failures
Parsing is usually single-core and low-memory; input count and plotting dominate runtime rather than CPU parallelism.
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
