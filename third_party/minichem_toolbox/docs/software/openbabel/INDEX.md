---
software_id: openbabel
versions: ["3.1.0"]
topics: ["index", "navigation", "capabilities"]
aliases: ["Open Babel", "openbabel"]
inputs: ["molecular input file"]
outputs: ["converted molecular file", "console summary"]
last_smoke_tested: "2026-07-28"
generated_from: minichem_toolbox/config/native_software_guides.yaml + minichem_toolbox/config/native_software_example_contracts.yaml
---
# Open Babel Native Software Guide

## Installed software
- Installed version: `3.1.0`.
- Operational status: `runnable`.
- Configured runtime: `quantum`.
- Primary use: molecular file conversion and deterministic cheminformatics manipulation.

## When to use this interface
Convert, filter, and manipulate molecular file formats with Open Babel. The native layer is appropriate when the Agent must author the software input or select version-specific options that are not represented by a preset Action.

## Documentation map
- `QUICKSTART.md`: complete staging, command, resource, submission, and collection flow.
- `COMMON_TASKS.md`: supported calculation families, input responsibilities, outputs, and validation states.
- `TROUBLESHOOTING.md`: high-frequency failure signatures, causes, fixes, and a pre-submission checklist.
- `examples/interface_smoke/`: the exact native command, toolbox request, and latest interface-smoke result.
- Additional topic files in this directory contain software-specific scientific mechanics where available.

## Supported command entries
| Executable | Input mode | Native invocation | Required staged inputs |
|---|---|---|---|
| `obabel` | `arguments` | `obabel -ixyz input.xyz -osdf -O output.sdf` | `input.xyz` |

## Supported task families
- format conversion.
- hydrogen addition.
- 2D to 3D conversion.
- filtering.
- descriptor calculation.

## Required knowledge before submission
The toolbox does not select a scientific method, force field, pseudopotential, basis, database, training set, convergence threshold, or workflow ordering. The Agent must obtain those choices from the task, a paper, or an authoritative source and then author a complete input.

## Official references
- https://openbabel.org/docs/Command-line_tools/babel.html
- https://openbabel.org/docs/

## Test interpretation
An interface smoke proves that the configured executable can be resolved and started through `submit_native_job`. A scientific smoke additionally requires a valid input, normal software termination, task-specific convergence, and parseable expected artifacts. The two levels are recorded separately and must not be conflated.
