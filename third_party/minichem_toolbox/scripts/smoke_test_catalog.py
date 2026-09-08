#!/usr/bin/env python3
"""Smoke-check every MiniChem Action/backend contract without heavy calculations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from minichem_toolbox.catalog import action_specs, backend_specs, validate_catalog
from minichem_toolbox.discovery import inspect_action
from minichem_toolbox.runtime import probe_all_backends
from minichem_mcp_tools.execution_models import (
    AnalysisRuntimeListRequest,
    SoftwareListRequest,
)
from minichem_mcp_tools.software_catalog import (
    list_analysis_runtimes,
    list_software,
    validate_native_guides,
)


def build_report() -> dict[str, Any]:
    validate_catalog()
    validate_native_guides()
    actions = action_specs()
    backends = backend_specs()
    health = probe_all_backends(backends.values())
    action_rows: list[dict[str, Any]] = []
    contract_failures: list[dict[str, str]] = []
    pair_count = 0

    for action_id, action in sorted(actions.items()):
        providers = []
        for backend_id in action.backend_ids:
            pair_count += 1
            contract_status = "passed"
            error = None
            try:
                inspected = inspect_action(
                    action_id,
                    backend_id=backend_id,
                    detail_level="contract",
                )
                if (
                    inspected.get("status") != "success"
                    or not inspected.get("selected_request_contract")
                ):
                    raise ValueError("inspect_action returned no selected request contract")
            except Exception as exc:
                contract_status = "failed"
                error = f"{type(exc).__name__}: {exc}"
                contract_failures.append(
                    {"action_id": action_id, "backend_id": backend_id, "error": error}
                )
            providers.append(
                {
                    "backend_id": backend_id,
                    "health_status": health[backend_id].get("status", "unknown"),
                    "available": bool(health[backend_id].get("available")),
                    "contract_status": contract_status,
                    "error": error,
                }
            )
        action_rows.append(
            {
                "action_id": action_id,
                "category": action.category,
                "provider_count": len(providers),
                "at_least_one_available_provider": any(
                    item["available"] for item in providers
                ),
                "all_contracts_passed": all(
                    item["contract_status"] == "passed" for item in providers
                ),
                "providers": providers,
            }
        )

    unavailable = sorted(
        backend_id
        for backend_id, value in health.items()
        if not value.get("available")
    )
    actions_without_provider = sorted(
        row["action_id"]
        for row in action_rows
        if not row["at_least_one_available_provider"]
    )
    status = (
        "passed_with_unavailable_optional_backends"
        if not contract_failures and unavailable
        else "passed"
        if not contract_failures
        else "failed"
    )
    native_inventory = list_software(
        SoftwareListRequest(native_only=True, available_only=False, limit=200)
    )
    runtime_inventory = list_analysis_runtimes(
        AnalysisRuntimeListRequest(available_only=False, limit=500)
    )
    native_software = native_inventory["software"]
    analysis_runtimes = runtime_inventory["runtimes"]
    return {
        "status": status,
        "test_scope": "catalog_and_request_contract_smoke_without_heavy_execution",
        "action_count": len(actions),
        "backend_count": len(backends),
        "action_backend_pair_count": pair_count,
        "contract_pass_count": pair_count - len(contract_failures),
        "contract_failure_count": len(contract_failures),
        "available_backend_count": len(backends) - len(unavailable),
        "unavailable_backends": unavailable,
        "actions_without_available_provider": actions_without_provider,
        "native_software_count": len(native_software),
        "available_native_software_count": sum(
            bool(item["available"]) for item in native_software
        ),
        "unavailable_native_software": sorted(
            item["software_id"] for item in native_software if not item["available"]
        ),
        "analysis_runtime_count": len(analysis_runtimes),
        "available_analysis_runtime_count": sum(
            bool(item["available"]) for item in analysis_runtimes
        ),
        "unavailable_analysis_runtimes": sorted(
            item["runtime"] for item in analysis_runtimes if not item["available"]
        ),
        "contract_failures": contract_failures,
        "native_software": native_software,
        "analysis_runtimes": analysis_runtimes,
        "actions": action_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report()
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
