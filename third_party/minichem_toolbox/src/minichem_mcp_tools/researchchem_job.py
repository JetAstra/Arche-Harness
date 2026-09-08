"""Path helper for Agent-authored programmable analysis jobs.

This SDK validates declared logical names and resolves paths inside the job
layout. It cannot prevent ordinary Python code from using ``open()``, absolute
paths, subprocesses, or other libraries directly; it is a reliability and
audit helper, not an OS security boundary.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class JobContext:
    root: Path
    contract: dict[str, Any]

    @classmethod
    def load(cls) -> "JobContext":
        root = Path(os.environ.get("MINICHEM_EXECUTION_JOB_DIRECTORY", Path.cwd()))
        contract_path = root / "analysis_contract.json"
        if not contract_path.is_file():
            raise RuntimeError(f"Analysis contract does not exist: {contract_path}")
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        return cls(root=root.resolve(), contract=contract)

    def _declared_path(self, group: str, name: str) -> Path:
        records = {item["name"]: item for item in self.contract.get(group, [])}
        if name not in records:
            raise KeyError(f"Undeclared {group[:-1]} name: {name!r}")
        relative = records[name].get("target_path") or records[name].get("path")
        path = (self.root / relative).resolve()
        path.relative_to(self.root)
        return path

    def input(self, name: str) -> Path:
        path = self._declared_path("inputs", name)
        if not path.is_file():
            raise FileNotFoundError(f"Declared input does not exist: {path}")
        return path

    def output(self, name: str) -> Path:
        path = self._declared_path("outputs", name)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def write_json(self, name: str, payload: Any) -> Path:
        """Write finite, standards-compliant JSON to one declared output."""

        self._reject_nonfinite(payload)
        path = self.output(name)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        self.register_output(name)
        return path

    def register_output(self, name: str, path: str | Path | None = None) -> dict[str, Any]:
        """Record one declared output after the program has created it."""

        declared = self._declared_path("outputs", name)
        candidate = declared if path is None else Path(path)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        candidate = candidate.resolve()
        candidate.relative_to(self.root)
        if candidate != declared:
            raise ValueError(
                f"Registered path for {name!r} must match the declared output path {declared}"
            )
        if candidate.is_symlink() or not candidate.is_file():
            raise FileNotFoundError(f"Declared output does not exist: {candidate}")
        declaration = {
            item["name"]: item for item in self.contract.get("outputs", [])
        }[name]
        digest = hashlib.sha256()
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        record = {
            "path": str(candidate.relative_to(self.root)),
            "size_bytes": candidate.stat().st_size,
            "sha256": digest.hexdigest(),
            "semantic_type": declaration.get("semantic_type"),
            "media_type": declaration.get("media_type"),
            "parent_artifacts": list(declaration.get("parent_artifact_ids") or []),
            "registered_at": datetime.now(timezone.utc).isoformat(),
        }
        registration_path = self.root / "runtime_output_registrations.json"
        registrations = {"schema_version": 1, "outputs": {}}
        if registration_path.is_file():
            registrations = json.loads(registration_path.read_text(encoding="utf-8"))
        registrations.setdefault("outputs", {})[name] = record
        temporary = registration_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(registrations, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, registration_path)
        return record

    @classmethod
    def _reject_nonfinite(cls, value: Any, *, path: str = "$") -> None:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"non-finite numeric value at {path}")
        if isinstance(value, list):
            for index, item in enumerate(value):
                cls._reject_nonfinite(item, path=f"{path}[{index}]")
        elif isinstance(value, dict):
            for key, item in value.items():
                cls._reject_nonfinite(item, path=f"{path}.{key}")

    @property
    def inputs_dir(self) -> Path:
        return self.root / "inputs"

    @property
    def outputs_dir(self) -> Path:
        return self.root / "outputs"

    def input_dir(self) -> Path:
        """Compatibility helper returning the isolated inputs directory."""

        return self.inputs_dir

    def output_dir(self) -> Path:
        """Compatibility helper returning the isolated outputs directory."""

        return self.outputs_dir

    @property
    def report_dir(self) -> Path:
        return self.root / "report"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"


__all__ = ["JobContext"]
