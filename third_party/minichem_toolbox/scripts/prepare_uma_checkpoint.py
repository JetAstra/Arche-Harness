#!/usr/bin/env python3
"""Load one UMA checkpoint into the project cache with an explicit download policy."""

from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def _offline_mode(enabled: bool) -> Iterator[None]:
    names = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    previous = {name: os.environ.get(name) for name in names}
    if enabled:
        os.environ.update({name: "1" for name in names})
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _existing_hugging_face_token() -> str:
    """Read an existing login before redirecting HF_HOME to the project cache."""

    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    candidates: list[Path] = []
    configured_token_path = os.environ.get("HF_TOKEN_PATH", "").strip()
    if configured_token_path:
        candidates.append(Path(configured_token_path).expanduser())
    configured_home = os.environ.get("HF_HOME", "").strip()
    if configured_home:
        candidates.append(Path(configured_home).expanduser() / "token")
    xdg_cache = os.environ.get("XDG_CACHE_HOME", "").strip()
    candidates.append(
        (Path(xdg_cache).expanduser() if xdg_cache else Path.home() / ".cache")
        / "huggingface"
        / "token"
    )
    for path in candidates:
        try:
            token = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if token:
            return token
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="uma-s-1p1")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Permit network access; gated models require an authorized HF_TOKEN.",
    )
    arguments = parser.parse_args()

    token = _existing_hugging_face_token() if arguments.allow_download else ""
    if arguments.allow_download and not token:
        parser.error(
            "--allow-download requires an authorized Hugging Face login or HF_TOKEN"
        )

    cache = Path(
        os.environ.get("MINICHEM_MODEL_CACHE")
        or Path(os.environ.get("MINICHEM_HOME", ".")) / ".mini_model_cache"
    ).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache / "huggingface")
    os.environ["HF_HUB_CACHE"] = str(cache / "huggingface" / "hub")
    os.environ["HF_XET_CACHE"] = str(cache / "huggingface" / "xet")
    os.environ["TORCH_HOME"] = str(cache / "torch")
    os.environ["FAIRCHEM_CACHE_DIR"] = str(cache / "fairchem")
    if token:
        os.environ["HF_TOKEN"] = token

    import torch

    device = arguments.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda was requested, but torch reports no visible CUDA device")

    with _offline_mode(not arguments.allow_download):
        from fairchem.core import pretrained_mlip

        predictor = pretrained_mlip.get_predict_unit(arguments.model, device=device)
    del predictor
    print(
        json.dumps(
            {
                "status": "ready",
                "model": arguments.model,
                "device": device,
                "cuda_available": torch.cuda.is_available(),
                "torch_cuda_build": torch.version.cuda,
                "model_cache": str(cache),
                "fairchem_cache": os.environ["FAIRCHEM_CACHE_DIR"],
                "download_permitted": arguments.allow_download,
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
