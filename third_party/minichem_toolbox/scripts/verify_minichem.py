#!/usr/bin/env python3
"""Fast installation and catalog verification for a copied MiniChem toolbox."""

from __future__ import annotations

import json

from minichem_toolbox.catalog import backend_specs, catalog_snapshot, validate_catalog
from minichem_toolbox.runtime import probe_all_backends


def main() -> None:
    validate_catalog()
    backends = backend_specs()
    health = probe_all_backends(backends.values())
    snapshot = catalog_snapshot(include_health=False)
    print(
        json.dumps(
            {
                "status": "success",
                "action_count": len(snapshot["actions"]),
                "backend_count": len(backends),
                "available_backends": sorted(
                    key for key, value in health.items() if value.get("available")
                ),
                "unavailable_backends": sorted(
                    key for key, value in health.items() if not value.get("available")
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
