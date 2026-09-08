"""Backend-local action execution entry point."""

from __future__ import annotations

import importlib
from typing import Any

from ..catalog import action_specs


_CATEGORY_MODULES = {
    "scientific_data_interchange": "interchange",
    "structure_and_system": "structure",
    "cheminformatics": "cheminformatics",
    "molecular_electronic": "electronic",
    "reaction_and_kinetics": "reaction",
}


def execute_local(
    action_id: str,
    backend_id: str,
    request: dict[str, Any],
) -> dict[str, Any]:
    specification = action_specs().get(action_id)
    if specification is not None:
        module_name = _CATEGORY_MODULES.get(specification.category)
        if module_name is not None:
            module = importlib.import_module(f"{__name__}.{module_name}")
            if action_id in module.ACTIONS:
                return module.execute(action_id, backend_id, request)
    return {
        "status": "unsupported",
        "error": {
            "code": "handler_missing",
            "message": f"No local handler is registered for action {action_id}",
        },
        "retryable": False,
    }
