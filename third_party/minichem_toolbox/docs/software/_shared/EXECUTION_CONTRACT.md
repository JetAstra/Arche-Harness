---
software_id: _shared
versions: []
topics: [execution, contract]
aliases: [native job lifecycle, execution rules]
inputs: []
outputs: []
last_smoke_tested: null
---
# Native Execution Contract

## Request
Use an executable returned by `inspect_software`. Stage every input explicitly and make every command argument refer to the staged target, not the workspace source path.

## Working directory
The process starts in an isolated job directory. Relative paths resolve inside that directory. The task workspace is not the process working directory.

## Result levels
A zero exit code proves only that the process ended normally. Separately verify the software termination marker, scientific convergence marker, required files, parseability, and whether the artifacts support the intended conclusion.

## Before submission
Run `validate_native_job`, read the relevant software topic, compare the generated deck with the tested example, and request only the CPU and memory supported by that software input.
