---
software_id: gaussian
versions: ["16"]
topics: [optimization-frequency, optimization, frequency]
aliases: [Gaussian Opt Freq, geometry optimization, vibrational frequencies]
inputs: ["Gaussian input deck", "optional checkpoint files"]
outputs: ["stdout.log", "checkpoint and requested property files"]
last_smoke_tested: 2026-07-28
---
# Gaussian Optimization and Frequency

## Route
Use explicit method and basis with `Opt`, `Freq`, or both. Do not infer a minimum or transition state only from normal termination.

## Validation
An optimization must report completion. A frequency calculation must return the expected modes. A minimum normally has no chemically meaningful imaginary mode; a transition-state candidate normally has one intended imaginary mode.
