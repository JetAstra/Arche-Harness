# MiniChem Model Cache

This directory retains the optional `sentence-transformers/all-MiniLM-L6-v2` ONNX files used for
semantic Action and software-documentation search. The model was originally downloaded from
Hugging Face and is placed at `all-MiniLM-L6-v2/`.

FairChem UMA checkpoints also belong here. `prepare_uma_checkpoint.py` directs FairChem,
Hugging Face, Xet, and Torch payloads to `.mini_model_cache/fairchem/`,
`.mini_model_cache/huggingface/`, and `.mini_model_cache/torch/`. The
`facebook/UMA` repository is gated: a local Hugging Face login or `HF_TOKEN` must have upstream
access before `uma-s-1p1` can be cached. A Python environment containing FairChem is not evidence
that its checkpoint has been downloaded.

The authorized `uma-s-1p1` checkpoint is currently cached under `fairchem/`. Run
`scripts/verify_uma_runtime.py` to distinguish a complete local checkpoint from an import-only
runtime. Cache payloads are intentionally excluded from Git; `manifest.json` records their hashes.

The RXNMapper ALBERT checkpoint is copied to
`rxnmapper/albert_heads_8_uspto_all_1310k/`. The `rxnmapper` runtime passes this exact cache path to
the library; the copy bundled inside the installed Python wheel is retained only for package
integrity and is not the configured runtime source.

The Agent harness model (`deepseek-v4-flash` in the OpenCode test) is accessed through its configured
API and is not stored in this directory.
