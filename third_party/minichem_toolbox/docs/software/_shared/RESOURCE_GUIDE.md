---
software_id: _shared
versions: []
topics: [resources, cpu, memory, parallelism]
aliases: [resource limits, threads, MPI, memory allocation]
inputs: []
outputs: []
last_smoke_tested: null
---
# Resource Guide

## CPU
The scheduler CPU request and software-level thread or process count must agree. Do not request more threads in the input than `resource_limits.cpu_cores`.

## Memory
`resource_limits.memory_mb` is the whole job allocation. Software memory keywords may be per process, per core, or total; follow the software topic and leave headroom for runtime overhead.

## Concurrency
Prefer one validated small job before launching a batch. Large concurrent Gaussian, ORCA, or VASP jobs can exhaust the task-wide budget even when each request is individually valid.
