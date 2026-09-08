---
software_id: gaussian
versions: ["16 C.01"]
topics: ["quickstart", "staging", "submission", "resources"]
aliases: ["Gaussian", "gaussian"]
inputs: ["input.gjf or input.com"]
outputs: ["stdout.log", "checkpoint file", "optional formatted checkpoint"]
last_smoke_tested: "2026-07-28"
generated_from: minichem_toolbox/config/native_software_guides.yaml + minichem_toolbox/config/native_software_example_contracts.yaml
---
# Gaussian Quickstart

## Complete invocation flow
1. Call `inspect_software` and confirm the executable, installed version, runtime, and documentation topics.
2. Author the smallest scientifically meaningful input for the selected task and installed version.
3. Put every source file under the evaluation workspace and map it to the exact job-local target expected by the input deck.
4. Set explicit CPU, total memory, GPU, and walltime limits. Keep all software-internal parallel settings within those limits.
5. Call `validate_native_job`; repair every error before submitting.
6. Call `submit_native_job`, poll `get_execution_job`, and finally call `collect_execution_job`.
7. Check process, software, convergence, artifact, and scientific-validation axes independently.

## Working directory contract
The runner creates an isolated job directory and executes the resolved binary there without a shell. Relative paths in arguments and input files resolve from that directory, not from the benchmark workspace root. `source_path` is workspace-relative; `target_path` is job-relative. Stage nested dependencies explicitly. The runner captures `request.json`, `stdout.log`, `stderr.log`, status, hashes, and collected artifacts.

## Input mode
The primary executable is `g16` and its input mode is `stdin_file`.
Required inputs: `input.com`.
Expected outputs: stdout/stderr or task-dependent outputs only.
Example classification: `scientific_template`.
Output behavior: Writes Gaussian output to stdout and checkpoint/scratch files named in the route input.

## Native command template
```bash
g16
```
Run that command only inside a directory containing the exact referenced files. The toolbox resolves the executable itself; do not embed shell redirection, pipes, `cd`, or environment activation in `arguments`.

## Toolbox submission template
```json
{
  "software_id": "gaussian",
  "executable": "g16",
  "arguments": [],
  "staged_inputs": [
    {
      "source_path": "workspace_inputs/input.com",
      "target_path": "input.com"
    }
  ],
  "resource_limits": {
    "cpu_cores": 1,
    "memory_mb": 2048,
    "gpu_count": 0
  },
  "stdin_target": "input.com"
}
```
The paths under `workspace_inputs/` are illustrative workspace-relative sources. Replace them with real files and keep the target names synchronized with the input deck.

## stdin, arguments, and fixed files
- For `arguments`, put each token in `arguments`; never pass one shell command string.
- For `stdin_file`, stage the input and set `stdin_target`; do not put `< input` in `arguments`.
- For `arguments_and_stdin_file`, provide both the positional file arguments and `stdin_target`.
- For `fixed_files`, stage every required filename exactly and normally leave `arguments` empty.

## Resource mapping
%NProcShared must not exceed cpu_cores; %Mem is total job memory and must remain below resource_limits.memory_mb.
`resource_limits.memory_mb` is total memory for the entire process group, not memory per MPI rank. `cpu_cores` is the allocation ceiling. Software thread or rank controls must not exceed it. Walltime is enforced by the Supervisor; a timeout is distinct from software non-convergence.

## Collection checklist
- `request_status=accepted` confirms only that the interface accepted the request.
- `process_status=completed` confirms only process exit code zero.
- `software_status=normal` requires a recognized normal end and no fatal message when a parser exists.
- `convergence_status` must match the requested task type; an SCF marker alone cannot validate an optimization or frequency job.
- `artifact_status=valid` requires all declared results to exist and parse.
- The Judger, not the execution layer, evaluates whether the results support the requested scientific conclusion.

## Required sections
Place Link 0 commands first, followed by one route section beginning with `#`. A blank line must separate the route from a non-empty title, another blank line must separate the title from charge and multiplicity, coordinates follow, and the input must end with a final blank line. Each Link1 segment repeats the required structure unless it explicitly and validly reads the checkpoint.
