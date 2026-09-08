---
software_id: crest
versions: ["3.0.2"]
topics: ["index", "navigation", "capabilities"]
aliases: ["CREST", "crest"]
inputs: ["input.xyz"]
outputs: ["crest_conformers.xyz", "crest.energies", "protonated.xyz", "deprotonated.xyz", "tautomers.xyz"]
last_smoke_tested: "2026-07-28"
generated_from: minichem_toolbox/config/native_software_guides.yaml + minichem_toolbox/config/native_software_example_contracts.yaml
---
# CREST Native Software Guide

## Installed software
- Installed version: `3.0.2`.
- Operational status: `runnable`.
- Configured runtime: `reaction`.
- Primary use: xTB-driven conformer, protomer, deprotomer, or tautomer enumeration.

## When to use this interface
Perform CREST conformer searches and related xTB-driven sampling from an Agent-authored structure and options. The native layer is appropriate when the Agent must author the software input or select version-specific options that are not represented by a preset Action.

## Documentation map
- `QUICKSTART.md`: complete staging, command, resource, submission, and collection flow.
- `COMMON_TASKS.md`: supported calculation families, input responsibilities, outputs, and validation states.
- `TROUBLESHOOTING.md`: high-frequency failure signatures, causes, fixes, and a pre-submission checklist.
- `examples/interface_smoke/`: the exact native command, toolbox request, and latest interface-smoke result.
- Additional topic files in this directory contain software-specific scientific mechanics where available.

## Supported command entries
| Executable | Input mode | Native invocation | Required staged inputs |
|---|---|---|---|
| `crest` | `arguments` | `crest input.xyz --gfn2 --T 4` | `input.xyz` |

## Supported task families
- conformer search.
- protonation.
- deprotonation.
- tautomerization.
- ensemble screening.

## Required knowledge before submission
The toolbox does not select a scientific method, force field, pseudopotential, basis, database, training set, convergence threshold, or workflow ordering. The Agent must obtain those choices from the task, a paper, or an authoritative source and then author a complete input.

## Official references
- https://crest-lab.github.io/crest-docs/
- https://github.com/crest-lab/crest

## Test interpretation
An interface smoke proves that the configured executable can be resolved and started through `submit_native_job`. A scientific smoke additionally requires a valid input, normal software termination, task-specific convergence, and parseable expected artifacts. The two levels are recorded separately and must not be conflated.
