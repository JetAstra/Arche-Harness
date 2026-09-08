#!/usr/bin/env python3
"""Run ARCHE Case1 through atomic Actions or the composite MiniChem Action."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from minichem_toolbox.backends import execute_local
from minichem_toolbox.backends.arche_ts_pipeline import run_atomic_workflow


def main() -> int:
    parser = argparse.ArgumentParser()
    default_request = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "arche_23_ts_pipeline_case1.json"
    )
    parser.add_argument("--request", type=Path, default=default_request)
    parser.add_argument("--entry", choices=("atomic", "composite"), default="atomic")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    request = json.loads(arguments.request.read_text(encoding="utf-8"))
    if request.get("inputs") or request.get("method_spec") or request.get("action_settings"):
        raise ValueError("Case 1 request must keep inputs, method_spec, and action_settings empty")
    if int((request.get("resource_limits") or {}).get("gpu_count", 0)) < 1:
        raise ValueError("Case 1 request requires resource_limits.gpu_count >= 1")

    if arguments.entry == "atomic":
        response = run_atomic_workflow(request)
    else:
        response = execute_local(
            request["action_id"], request["backend_id"], request
        )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(response, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    result = response.get("result") or {}
    print(
        json.dumps(
            {
                "entry": arguments.entry,
                "status": response.get("status"),
                "endpoint_pair_count": result.get("endpoint_pair_count"),
                "stage_statuses": result.get("stage_statuses"),
                "output": os.path.relpath(
                    arguments.output.resolve(), start=Path.cwd().resolve()
                ),
            },
            ensure_ascii=True,
        )
    )
    return 0 if response.get("status") in {"success", "partial_success"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
