"""Compatibility exports for the split ActionSpec and BackendSpec catalog."""

from __future__ import annotations

from .actions import ACTION_SPECS as _FULL_ACTION_SPECS
from .backend_specs import BACKEND_SPECS as _FULL_BACKEND_SPECS
from .mini_profile import build_mini_specs


ACTION_SPECS, BACKEND_SPECS = build_mini_specs(
    _FULL_ACTION_SPECS,
    _FULL_BACKEND_SPECS,
)

__all__ = ["ACTION_SPECS", "BACKEND_SPECS"]
