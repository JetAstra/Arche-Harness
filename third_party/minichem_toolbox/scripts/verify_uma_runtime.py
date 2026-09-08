#!/usr/bin/env python3
"""Verify the isolated FairChem/ASH runtime without downloading a model."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
from pathlib import Path


MODULES = ("ase", "ash", "fairchem.core", "numpy", "rdkit", "rxnmapper", "torch")
UMA_MODEL = "uma-s-1p1"


def _distribution_version(name: str) -> str | None:
    for candidate in (name, name.split(".")[0]):
        try:
            return importlib.metadata.version(candidate)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def main() -> int:
    modules: dict[str, dict[str, object]] = {}
    failures: list[str] = []
    for name in MODULES:
        try:
            importlib.import_module(name)
            modules[name] = {"available": True, "version": _distribution_version(name)}
        except Exception as exc:
            modules[name] = {
                "available": False,
                "error": f"{type(exc).__name__}: {exc}",
                "version": _distribution_version(name),
            }
            failures.append(name)

    torch = importlib.import_module("torch") if modules["torch"]["available"] else None
    model_cache = Path(
        os.environ.get("MINICHEM_MODEL_CACHE")
        or Path(os.environ.get("MINICHEM_HOME", ".")) / ".mini_model_cache"
    ).resolve()
    fairchem_cache = Path(
        os.environ.get("FAIRCHEM_CACHE_DIR") or model_cache / "fairchem"
    ).resolve()
    checkpoints = sorted(
        path.resolve()
        for path in fairchem_cache.glob(
            f"models--facebook--UMA/snapshots/*/checkpoints/{UMA_MODEL}.pt"
        )
        if path.is_file()
    )
    checkpoint = checkpoints[-1] if checkpoints else None
    checkpoint_ready = checkpoint is not None and checkpoint.stat().st_size > 0
    display_path = lambda path: os.path.relpath(path, start=Path.cwd().resolve())
    report = {
        "status": "ready" if not failures and checkpoint_ready else "incomplete",
        "modules": modules,
        "torch_cuda_build": getattr(getattr(torch, "version", None), "cuda", None),
        "cuda_available": bool(torch is not None and torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch is not None else 0,
        "model_cache": display_path(model_cache),
        "fairchem_cache": display_path(fairchem_cache),
        "checkpoint": {
            "model": UMA_MODEL,
            "ready": checkpoint_ready,
            "path": display_path(checkpoint) if checkpoint else None,
            "size_bytes": checkpoint.stat().st_size if checkpoint else 0,
        },
        "checkpoint_download_attempted": False,
    }
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 1 if failures or not checkpoint_ready else 0


if __name__ == "__main__":
    raise SystemExit(main())
