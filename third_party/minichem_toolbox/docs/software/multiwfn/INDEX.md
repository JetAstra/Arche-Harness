---
software_id: multiwfn
versions: ["2026.7.15"]
topics: ["index", "navigation", "capabilities"]
aliases: ["Multiwfn", "multiwfn"]
inputs: ["wavefunction file", "commands.txt"]
outputs: ["stdout.log", "exported grids", "tables", "images or structure files"]
last_smoke_tested: "2026-07-28"
generated_from: minichem_toolbox/config/native_software_guides.yaml + minichem_toolbox/config/native_software_example_contracts.yaml
---
# Multiwfn Native Software Guide

## Installed software
- Installed version: `2026.7.15`.
- Operational status: `runnable_with_wavefunction`.
- Configured runtime: `multiwfn`.
- Primary use: scripted wavefunction and real-space analysis through the no-GUI executable.

## When to use this interface
Run Multiwfn analyses using an explicit wavefunction file and menu-command stream. The native layer is appropriate when the Agent must author the software input or select version-specific options that are not represented by a preset Action.

## Documentation map
- `QUICKSTART.md`: complete staging, command, resource, submission, and collection flow.
- `COMMON_TASKS.md`: supported calculation families, input responsibilities, outputs, and validation states.
- `TROUBLESHOOTING.md`: high-frequency failure signatures, causes, fixes, and a pre-submission checklist.
- `examples/interface_smoke/`: the exact native command, toolbox request, and latest interface-smoke result.
- Additional topic files in this directory contain software-specific scientific mechanics where available.

## Supported command entries
| Executable | Input mode | Native invocation | Required staged inputs |
|---|---|---|---|
| `Multiwfn_noGUI` | `arguments_and_stdin_file` | `Multiwfn_noGUI wavefunction.fchk` | `commands.txt`, `wavefunction.fchk` |

## Supported task families
- density grids.
- surface properties.
- orbital analysis.
- population analysis.
- topology.
- spectra.

## Required knowledge before submission
The toolbox does not select a scientific method, force field, pseudopotential, basis, database, training set, convergence threshold, or workflow ordering. The Agent must obtain those choices from the task, a paper, or an authoritative source and then author a complete input.

## Official references
- http://sobereva.com/multiwfn/
- http://sobereva.com/multiwfn/res/Manual_3.8.pdf

## Test interpretation
An interface smoke proves that the configured executable can be resolved and started through `submit_native_job`. A scientific smoke additionally requires a valid input, normal software termination, task-specific convergence, and parseable expected artifacts. The two levels are recorded separately and must not be conflated.
