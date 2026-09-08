---
software_id: xtb
versions: ["6.7.1"]
topics: ["index", "navigation", "capabilities"]
aliases: ["xTB", "xtb"]
inputs: ["structure.xyz", "optional xcontrol file"]
outputs: ["stdout.log", "xtbopt.xyz", "hessian", "charges", "wbo", "trajectory files"]
last_smoke_tested: "2026-07-28"
generated_from: minichem_toolbox/config/native_software_guides.yaml + minichem_toolbox/config/native_software_example_contracts.yaml
---
# xTB Native Software Guide

## Installed software
- Installed version: `6.7.1`.
- Operational status: `runnable`.
- Configured runtime: `quantum`.
- Primary use: semiempirical molecular energy, optimization, frequency, or dynamics calculation.

## When to use this interface
Execute native xTB single points, properties, optimization, frequencies, dynamics, and related modes selected by the Agent. The native layer is appropriate when the Agent must author the software input or select version-specific options that are not represented by a preset Action.

## Documentation map
- `QUICKSTART.md`: complete staging, command, resource, submission, and collection flow.
- `COMMON_TASKS.md`: supported calculation families, input responsibilities, outputs, and validation states.
- `TROUBLESHOOTING.md`: high-frequency failure signatures, causes, fixes, and a pre-submission checklist.
- `examples/interface_smoke/`: the exact native command, toolbox request, and latest interface-smoke result.
- Additional topic files in this directory contain software-specific scientific mechanics where available.

## Supported command entries
| Executable | Input mode | Native invocation | Required staged inputs |
|---|---|---|---|
| `xtb` | `arguments` | `xtb structure.xyz --gfn 2 --sp --chrg 0 --uhf 0` | `structure.xyz` |

## Supported task families
- single point.
- geometry optimization.
- frequency.
- molecular dynamics.
- solvation.
- properties.

## Required knowledge before submission
The toolbox does not select a scientific method, force field, pseudopotential, basis, database, training set, convergence threshold, or workflow ordering. The Agent must obtain those choices from the task, a paper, or an authoritative source and then author a complete input.

## Official references
- https://xtb-docs.readthedocs.io/en/latest/
- https://github.com/grimme-lab/xtb

## Test interpretation
An interface smoke proves that the configured executable can be resolved and started through `submit_native_job`. A scientific smoke additionally requires a valid input, normal software termination, task-specific convergence, and parseable expected artifacts. The two levels are recorded separately and must not be conflated.
