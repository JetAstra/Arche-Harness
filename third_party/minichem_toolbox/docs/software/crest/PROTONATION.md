---
software_id: crest
versions: ["3"]
topics: [protonation, deprotonation, tautomer]
aliases: [CREST protonate, CREST deprotonate, CREST tautomerize]
inputs: ["XYZ structure"]
outputs: ["stdout.log", "protonated.xyz", "deprotonated.xyz", "tautomers.xyz"]
last_smoke_tested: null
---
# CREST Protonation Modes

## Mode selection
Choose exactly one of `-protonate`, `-deprotonate`, `-tautomerize`, or standard conformer-search mode. CREST 3.0.2 documents the single-dash spellings; the validator also accepts the historical double-dash spellings. Explicitly set the starting molecular charge and spin state.

## Expected outputs

- `-protonate` writes the final sorted ensemble to `protonated.xyz`.
- `-deprotonate` writes the final sorted ensemble to `deprotonated.xyz`.
- `-tautomerize` writes the final sorted ensemble to `tautomers.xyz`.

Normal termination alone is insufficient. The result file for the selected mode must exist and be non-empty before the job is mechanically valid.

## Interpretation
CREST enumerates and ranks candidates under its selected model. The final scientific task may still require higher-level geometry optimization, frequency validation, and free-energy ranking.
