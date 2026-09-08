---
software_id: gaussian
versions: ["16"]
topics: [link1, multi-step]
aliases: [Gaussian Link1, checkpoint workflow]
inputs: ["Gaussian input deck", "optional checkpoint files"]
outputs: ["stdout.log", "checkpoint and requested property files"]
last_smoke_tested: null
---
# Gaussian Link1

## Segmentation
Separate jobs with `--Link1--`. Each segment needs its own route section, title separation, and valid molecule source. Use `%Chk` consistently when a later segment reads the checkpoint.

## Geometry reuse
When using `Geom=AllCheck` or `Geom=Check`, ensure the referenced checkpoint is created by the previous segment and use route keywords compatible with the intended reuse. Do not add conflicting coordinate sections.
