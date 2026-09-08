---
software_id: _shared
versions: []
topics: [success, convergence, validation]
aliases: [normal termination, scientific convergence, artifact validation]
inputs: []
outputs: []
last_smoke_tested: null
---
# Success and Convergence

## Process success
The supervisor observed exit code zero and no forced cancellation or timeout.

## Software success
The main output contains the software-specific normal termination marker and no fatal error marker.

## Scientific convergence
The requested SCF, geometry, frequency, molecular dynamics, or electronic minimization criterion was reached. Normal software termination does not imply convergence.

## Artifact validity
Every required output exists, is non-empty, is parseable in its declared format, and belongs to the same calculation. Final scientific correctness remains the responsibility of the task Judger.
