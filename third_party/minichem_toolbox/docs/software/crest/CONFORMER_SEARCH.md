---
software_id: crest
versions: ["3"]
topics: [conformer-search, conformers]
aliases: [CREST conformer sampling, GFN conformer search]
inputs: ["XYZ structure"]
outputs: ["stdout.log", "CREST ensemble or protonation files"]
last_smoke_tested: 2026-07-28
---
# CREST Conformer Search

## Invocation
Stage one valid XYZ file and pass its target as the positional input. Declare charge, unpaired electrons, solvent, energy window, and thread count only when required by the task and supported by the installed version.

## Validation
Require normal CREST termination and a non-empty conformer ensemble output. Record the number of conformers and rotamers and do not treat the lowest CREST structure as a quantum-refined final answer.
