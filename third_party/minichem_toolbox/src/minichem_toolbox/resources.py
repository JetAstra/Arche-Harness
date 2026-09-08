"""Empty scientific-resource registry for the focused MiniChem profile."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


def resource_specs() -> dict[str, dict[str, Any]]:
    return {}


def is_resource_reference(value: Any) -> bool:
    return isinstance(value, dict) and bool(value.get("resource_id"))


def parse_resource_reference(value: Any) -> tuple[str, str | None]:
    if not is_resource_reference(value):
        raise ValueError("Expected a resource reference with resource_id")
    return str(value["resource_id"]), value.get("variant")


def resolve_resource_reference(value: Any) -> Path:
    resource_id, _variant = parse_resource_reference(value)
    raise KeyError(f"MiniChem has no registered scientific resource {resource_id!r}")


def resource_reference_metadata(value: Any) -> dict[str, Any]:
    resource_id, variant = parse_resource_reference(value)
    return {"resource_id": resource_id, "variant": variant, "available": False}


def collect_resource_references(value: Any) -> list[dict[str, Any]]:
    if is_resource_reference(value):
        return [resource_reference_metadata(value)]
    if isinstance(value, dict):
        return [
            item
            for child in value.values()
            for item in collect_resource_references(child)
        ]
    if isinstance(value, (list, tuple)):
        return [item for child in value for item in collect_resource_references(child)]
    return []


def resource_snapshot() -> list[dict[str, Any]]:
    return []


def resources_for_backends(backend_ids: Iterable[str]) -> list[dict[str, Any]]:
    del backend_ids
    return []
