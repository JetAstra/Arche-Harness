"""Inspect, validate, document, and probe the immutable chemistry catalog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from minichem_toolbox.catalog import (
    action_specs,
    backend_specs,
    catalog_snapshot,
    markdown_catalog,
    validate_catalog,
)
from minichem_toolbox.paths import PROJECT_ROOT

from .registry import configuration_errors
from .open_tools import OPEN_EXECUTION_TOOL_NAMES
from .discovery_tools import PROGRESSIVE_DISCOVERY_TOOL_NAMES
from .software_catalog import load_native_guides, validate_native_guides


load_dotenv(PROJECT_ROOT / "config.local.env", override=False)
load_dotenv(PROJECT_ROOT.parent.parent / "config.local.env", override=False)


def installation_report() -> dict[str, Any]:
    snapshot = catalog_snapshot(include_health=True)
    unavailable = [
        backend for backend in snapshot["backends"]
        if not (backend.get("health") or {}).get("available", False)
    ]
    conda_packages = sorted(
        {
            package
            for backend in unavailable
            if (backend.get("health") or {}).get("missing_python_modules")
            or (backend.get("health") or {}).get("missing_executables")
            for package in backend.get("conda_packages", [])
        }
    )
    pip_packages = sorted(
        {
            package
            for backend in unavailable
            if (backend.get("health") or {}).get("missing_python_modules")
            or (backend.get("health") or {}).get("missing_executables")
            for package in backend.get("pip_packages", [])
        }
    )
    manual = [
        {
            "backend_id": backend["id"],
            "display_name": backend["display_name"],
            "license_class": backend["license_class"],
            "install_notes": backend.get("install_notes", ""),
            "missing_executables": (backend.get("health") or {}).get("missing_executables", []),
            "missing_environment": (backend.get("health") or {}).get("missing_environment", []),
        }
        for backend in unavailable
        if backend.get("license_class") != "open_source" or backend.get("install_notes")
    ]
    credentials = [
        {
            "backend_id": backend["id"],
            "environment_variables": [
                name
                for name in (backend.get("health") or {}).get("missing_environment", [])
                if name.endswith("_KEY")
            ],
        }
        for backend in unavailable
        if any(
            name.endswith("_KEY")
            for name in (backend.get("health") or {}).get("missing_environment", [])
        )
    ]
    return {
        "catalog_hash": snapshot["catalog_hash"],
        "unavailable_backend_count": len(unavailable),
        "unavailable_backends": unavailable,
        "conda_packages": conda_packages,
        "pip_packages": pip_packages,
        "required_data_resources": [
            {
                "backend_id": backend["id"],
                "resources": backend.get("required_data_resources", []),
            }
            for backend in snapshot["backends"]
            if backend.get("required_data_resources")
        ],
        "registered_scientific_resources": [
            {
                "resource_id": resource["id"],
                "kind": resource.get("kind"),
                "compatible_backends": resource.get("compatible_backends", []),
                "available": bool(resource.get("available")),
                "selection_syntax": resource.get("selection_syntax", "runtime-managed"),
            }
            for resource in snapshot.get("resources", [])
        ],
        "missing_registered_scientific_resources": [
            resource["id"]
            for resource in snapshot.get("resources", [])
            if not resource.get("available")
        ],
        "missing_credentials": credentials,
        "manual_installations": manual,
    }


def write_catalog(path: Path, *, include_health: bool = True) -> Path:
    path.write_text(markdown_catalog(include_health=include_health), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list")
    subparsers.add_parser("validate")
    catalog_parser = subparsers.add_parser("catalog")
    catalog_parser.add_argument("--output", type=Path, required=True)
    catalog_parser.add_argument("--no-health", action="store_true")
    missing_parser = subparsers.add_parser("missing")
    missing_parser.add_argument("--json", action="store_true")
    snapshot_parser = subparsers.add_parser("snapshot")
    snapshot_parser.add_argument("--no-health", action="store_true")
    args = parser.parse_args()

    if args.command == "list":
        for specification in action_specs().values():
            kind = "data" if specification.data_action else "scientific"
            print(
                f"{specification.id:42s} {kind:10s} {specification.category:24s} "
                f"{','.join(specification.backend_ids)}"
            )
        return 0
    if args.command == "validate":
        validate_catalog()
        validate_native_guides()
        errors = configuration_errors()
        if errors:
            raise SystemExit("\n".join(errors))
        print(
            f"ok: {len(action_specs())} actions, {len(backend_specs())} backends, "
            f"{len(OPEN_EXECUTION_TOOL_NAMES)} open-execution tools, "
            f"{len(PROGRESSIVE_DISCOVERY_TOOL_NAMES)} progressive discovery/dispatch tools, "
            f"{len(load_native_guides()['software'])} native software guides, "
            "three peer layers, agent-required choices, no fallback"
        )
        return 0
    if args.command == "catalog":
        print(write_catalog(args.output, include_health=not args.no_health))
        return 0
    if args.command == "missing":
        report = installation_report()
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            print(f"Unavailable backends: {report['unavailable_backend_count']}")
            print("Conda packages: " + (", ".join(report["conda_packages"]) or "none"))
            print("Pip packages: " + (", ".join(report["pip_packages"]) or "none"))
            for item in report["required_data_resources"]:
                print(f"Data: {item['backend_id']} — {'; '.join(item['resources'])}")
            for item in report["missing_credentials"]:
                print(f"Credential: {item['backend_id']} — {', '.join(item['environment_variables'])}")
            for item in report["manual_installations"]:
                print(f"Manual: {item['backend_id']} — {item['install_notes']}")
        return 0
    print(json.dumps(catalog_snapshot(include_health=not args.no_health), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
