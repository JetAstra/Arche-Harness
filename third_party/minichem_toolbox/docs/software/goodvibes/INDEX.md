---
software_id: goodvibes
versions: ["4.3.0"]
topics: ["index", "navigation", "capabilities"]
aliases: ["GoodVibes", "goodvibes"]
inputs: ["Gaussian", "ORCA", "NWChem", "Q-Chem", "xTB", "or ASE frequency output"]
outputs: ["console table", "JSON or CSV result", "optional PES and plots"]
last_smoke_tested: "2026-07-28"
generated_from: minichem_toolbox/config/native_software_guides.yaml + minichem_toolbox/config/native_software_example_contracts.yaml
---
# GoodVibes Native Software Guide

## Installed software
- Installed version: `4.3.0`.
- Operational status: `runnable_with_qm_outputs`.
- Configured runtime: `goodvibes`.
- Primary use: quasi-harmonic thermochemistry from completed quantum-chemistry frequency outputs.

## When to use this interface
Apply GoodVibes 4.3.0 thermochemistry, ensemble, selectivity, consistency, and reaction-profile analysis to explicit completed quantum-chemistry outputs. The native layer is appropriate when the Agent must author the software input or select version-specific options that are not represented by a preset Action.

## Documentation map
- `QUICKSTART.md`: complete staging, command, resource, submission, and collection flow.
- `COMMON_TASKS.md`: supported calculation families, input responsibilities, outputs, and validation states.
- `TROUBLESHOOTING.md`: high-frequency failure signatures, causes, fixes, and a pre-submission checklist.
- `examples/interface_smoke/`: the exact native command, toolbox request, and latest interface-smoke result.
- Additional topic files in this directory contain software-specific scientific mechanics where available.

## Supported command entries
| Executable | Input mode | Native invocation | Required staged inputs |
|---|---|---|---|
| `goodvibes` | `arguments` | `goodvibes output.log --temp 298.15 --conc 1.0 --qs grimme --qh --fs 100 --fh 100 -v 0.99 --zpe-vscal 0.98 --json result.json` | `output.log` |

## Supported task families
- single-temperature thermochemistry.
- temperature scans.
- concentration corrections.
- selectivity.
- reaction profiles.

## Required knowledge before submission
The toolbox does not select a scientific method, force field, pseudopotential, basis, database, training set, convergence threshold, or workflow ordering. The Agent must obtain those choices from the task, a paper, or an authoritative source and then author a complete input.

## Official references
- https://goodvibespy.readthedocs.io/en/latest/
- https://github.com/patonlab/GoodVibes

## Test interpretation
An interface smoke proves that the configured executable can be resolved and started through `submit_native_job`. A scientific smoke additionally requires a valid input, normal software termination, task-specific convergence, and parseable expected artifacts. The two levels are recorded separately and must not be conflated.
