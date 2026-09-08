---
software_id: gaussian
versions: ["16 C.01"]
topics: ["index", "navigation", "capabilities"]
aliases: ["Gaussian", "gaussian"]
inputs: ["input.gjf or input.com"]
outputs: ["stdout.log", "checkpoint file", "optional formatted checkpoint"]
last_smoke_tested: "2026-07-28"
generated_from: minichem_toolbox/config/native_software_guides.yaml + minichem_toolbox/config/native_software_example_contracts.yaml
---
# Gaussian Native Software Guide

## Installed software
- Installed version: `16 C.01`.
- Operational status: `runnable`.
- Configured runtime: `gaussian`.
- Primary use: molecular electronic-structure calculation from a Gaussian input stream.

## When to use this interface
Execute Gaussian 16 input and convert checkpoint files with formchk. The native layer is appropriate when the Agent must author the software input or select version-specific options that are not represented by a preset Action.

## Documentation map
- `QUICKSTART.md`: complete staging, command, resource, submission, and collection flow.
- `COMMON_TASKS.md`: supported calculation families, input responsibilities, outputs, and validation states.
- `TROUBLESHOOTING.md`: high-frequency failure signatures, causes, fixes, and a pre-submission checklist.
- `examples/interface_smoke/`: the exact native command, toolbox request, and latest interface-smoke result.
- Additional topic files in this directory contain software-specific scientific mechanics where available.

## Supported command entries
| Executable | Input mode | Native invocation | Required staged inputs |
|---|---|---|---|
| `g16` | `stdin_file` | `g16` | `input.com` |
| `formchk` | `arguments` | `formchk input.chk output.fchk` | `input.chk`, `output.fchk` |

## Supported task families
- single point.
- geometry optimization.
- frequency.
- transition state.
- IRC.
- Link1 workflow.

## Required knowledge before submission
The toolbox does not select a scientific method, force field, pseudopotential, basis, database, training set, convergence threshold, or workflow ordering. The Agent must obtain those choices from the task, a paper, or an authoritative source and then author a complete input.

## Official references
- https://gaussian.com/g16main/
- https://gaussian.com/techsupport/

## Test interpretation
An interface smoke proves that the configured executable can be resolved and started through `submit_native_job`. A scientific smoke additionally requires a valid input, normal software termination, task-specific convergence, and parseable expected artifacts. The two levels are recorded separately and must not be conflated.
