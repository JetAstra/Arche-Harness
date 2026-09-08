---
software_id: pysisyphus
versions: ["1.0.0"]
topics: ["index", "navigation", "capabilities"]
aliases: ["pysisyphus", "pysisyphus"]
inputs: ["config.yaml", "referenced geometry and calculator files"]
outputs: ["optimization log", "trajectory", "final geometry", "Hessian or path files"]
last_smoke_tested: "2026-07-28"
generated_from: minichem_toolbox/config/native_software_guides.yaml + minichem_toolbox/config/native_software_example_contracts.yaml
---
# pysisyphus Native Software Guide

## Installed software
- Installed version: `1.0.0`.
- Operational status: `runnable_with_calculator`.
- Configured runtime: `reaction`.
- Primary use: geometry, transition-state, IRC, or chain-of-states optimization from a YAML configuration.

## When to use this interface
Execute a complete pysisyphus workflow configuration authored by the Agent. The native layer is appropriate when the Agent must author the software input or select version-specific options that are not represented by a preset Action.

## Documentation map
- `QUICKSTART.md`: complete staging, command, resource, submission, and collection flow.
- `COMMON_TASKS.md`: supported calculation families, input responsibilities, outputs, and validation states.
- `TROUBLESHOOTING.md`: high-frequency failure signatures, causes, fixes, and a pre-submission checklist.
- `examples/interface_smoke/`: the exact native command, toolbox request, and latest interface-smoke result.
- Additional topic files in this directory contain software-specific scientific mechanics where available.

## Supported command entries
| Executable | Input mode | Native invocation | Required staged inputs |
|---|---|---|---|
| `pysis` | `arguments` | `pysis config.yaml` | `config.yaml` |

## Supported task families
- minimum optimization.
- transition state.
- IRC.
- NEB.
- growing string.

## Required knowledge before submission
The toolbox does not select a scientific method, force field, pseudopotential, basis, database, training set, convergence threshold, or workflow ordering. The Agent must obtain those choices from the task, a paper, or an authoritative source and then author a complete input.

## Official references
- https://pysisyphus.readthedocs.io/en/stable/overview.html
- https://pysisyphus.readthedocs.io/en/stable/quickstart.html

## Test interpretation
An interface smoke proves that the configured executable can be resolved and started through `submit_native_job`. A scientific smoke additionally requires a valid input, normal software termination, task-specific convergence, and parseable expected artifacts. The two levels are recorded separately and must not be conflated.
