---
software_id: _shared
versions: []
topics: [staging, paths, working-directory]
aliases: [file paths, staged target, cwd]
inputs: []
outputs: []
last_smoke_tested: null
---
# Staging and Paths

## Source and target
`source_path` identifies an existing workspace file. `target_path` is the relative filename created inside the job directory. Software input must reference `target_path`.

## Rules
Targets cannot be absolute, contain `..`, or overwrite job control files. Stage companion files under the exact names expected by the software. For stdin programs, `stdin_target` must be one of the staged targets.

## Outputs
Do not write to the task workspace by absolute path. Let the program create outputs in the job directory, then collect them through the job manifest.
