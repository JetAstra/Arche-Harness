"""Shared ORCA model-chemistry validation for discovery and execution."""

from __future__ import annotations

from typing import Any


ORCA_COMPOSITE_3C_METHODS = {
    "hf-3c": "HF-3c",
    "b97-3c": "B97-3c",
    "r2scan-3c": "r2SCAN-3c",
    "pbeh-3c": "PBEh-3c",
    "b3lyp-3c": "B3LYP-3c",
    "wb97x-3c": "wB97X-3c",
}
ORCA_METHOD_DEFAULT_BASIS_SENTINELS = {
    "auto",
    "builtin",
    "method_default",
    "method-default",
}


def normalize_orca_method_basis(
    method_value: Any,
    basis_value: Any | None,
) -> tuple[str, str | None]:
    """Return the ORCA method token and optional explicit orbital-basis token."""

    method = str(method_value).strip()
    if not method:
        raise ValueError("ORCA method_spec.method must not be empty")
    composite = ORCA_COMPOSITE_3C_METHODS.get(method.casefold())
    basis = "" if basis_value is None else str(basis_value).strip()
    basis_is_default = not basis or basis.casefold() in ORCA_METHOD_DEFAULT_BASIS_SENTINELS
    if composite is not None:
        if not basis_is_default:
            raise ValueError(
                f"ORCA composite method {composite} includes its own orbital basis; "
                "omit method_spec.basis or use method_default/auto, not "
                f"basis={basis!r}"
            )
        return composite, None
    if basis_is_default:
        raise ValueError(
            f"ORCA method {method!r} requires a concrete method_spec.basis; "
            "auto/method_default is valid only for built-in 3c composite methods"
        )
    return method, basis


__all__ = [
    "ORCA_COMPOSITE_3C_METHODS",
    "ORCA_METHOD_DEFAULT_BASIS_SENTINELS",
    "normalize_orca_method_basis",
]
