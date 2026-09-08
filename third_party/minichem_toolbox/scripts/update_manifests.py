#!/usr/bin/env python3
"""Generate portable metadata for MiniChem software and model payloads."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOFTWARE_CACHE = ROOT / ".mini_software_cache"
MODEL_CACHE = ROOT / ".mini_model_cache"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, base: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(base).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def directory_size(path: Path) -> int:
    completed = subprocess.run(
        ["du", "-sb", str(path)], check=True, text=True, capture_output=True
    )
    return int(completed.stdout.split()[0])


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    gaussian = SOFTWARE_CACHE / "gaussian" / "g16" / "install" / "g16"
    multiwfn = (
        SOFTWARE_CACHE
        / "multiwfn"
        / "2026.7.15"
        / "Multiwfn_2026.7.15_bin_Linux_noGUI"
    )
    multiwfn_archive = multiwfn.parent / "Multiwfn_2026.7.15_bin_Linux_noGUI.zip"
    gaussian_files = [gaussian / "g16", gaussian / "formchk"]
    multiwfn_files = [
        multiwfn_archive,
        multiwfn / "Multiwfn_noGUI",
        multiwfn / "settings.ini",
        multiwfn / "LICENSE.txt",
    ]
    software_files = [*gaussian_files, *multiwfn_files]
    missing = [path for path in software_files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing software cache files: " + ", ".join(map(str, missing)))
    components: dict[str, object] = {
        "gaussian_16_c01": {
            "license": "commercial; redistribution is not permitted by MiniChem",
            "size_bytes": directory_size(SOFTWARE_CACHE / "gaussian"),
            "files": [file_record(path, SOFTWARE_CACHE) for path in gaussian_files],
        },
        "multiwfn_2026_7_15": {
            "license": "upstream Multiwfn terms and citation requirements apply",
            "size_bytes": directory_size(SOFTWARE_CACHE / "multiwfn"),
            "files": [file_record(path, SOFTWARE_CACHE) for path in multiwfn_files],
        },
    }
    write(
        SOFTWARE_CACHE / "manifest.json",
        {
            "schema_version": 1,
            "portable_root": ".mini_software_cache",
            "components": components,
        },
    )

    model_files = sorted(
        path
        for path in MODEL_CACHE.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and path.name not in {"README.md", "manifest.json"}
        and not path.name.endswith(".lock")
        and not path.name.endswith(".incomplete")
        and "logs" not in path.parts
    )
    write(
        MODEL_CACHE / "manifest.json",
        {
            "schema_version": 1,
            "portable_root": ".mini_model_cache",
            "components": {
                "sentence-transformers/all-MiniLM-L6-v2": (
                    "optional offline semantic recall for Actions and software documentation"
                ),
                "rxnmapper/albert_heads_8_uspto_all_1310k": (
                    "offline attention-guided reaction atom mapping"
                ),
                "facebook/UMA": (
                    "gated uma-s-1p1 checkpoint and elemental reference energies"
                ),
            },
            "files": [file_record(path, MODEL_CACHE) for path in model_files],
        },
    )


if __name__ == "__main__":
    main()
