"""Make MiniChem contract tests independent of operator resource overrides."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _evaluation_resource_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the checked-in ARCHE evaluation policy unless a test overrides it."""

    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_CPU_CORES", "60")
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_MEMORY_MB", "320000")
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_GPU_COUNT", "1")
