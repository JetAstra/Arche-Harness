from __future__ import annotations

from pathlib import Path

import pytest

from minichem_toolbox import environment_layout


TOOLBOX_ROOT = Path(__file__).resolve().parents[1]
STANDARD_PYTHON = TOOLBOX_ROOT.parent.parent / ".venv" / "bin" / "python"


def test_runtime_path_uses_arche_standard_environment(monkeypatch):
    monkeypatch.delenv(environment_layout.ARCHE_PYTHON_ENV, raising=False)
    monkeypatch.delenv(
        environment_layout.MINICHEM_RUNTIME_PYTHON_ENV, raising=False
    )

    resolved = environment_layout.environment_path("fairchem")

    assert resolved == STANDARD_PYTHON.parent.parent.resolve()


def test_runtime_path_uses_explicit_arche_python_override(monkeypatch):
    monkeypatch.setenv(environment_layout.ARCHE_PYTHON_ENV, str(STANDARD_PYTHON))

    resolved = environment_layout.environment_path("ash_neb")

    assert resolved == STANDARD_PYTHON.parent.parent.resolve()


def test_nonstandard_arche_python_override_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv(
        environment_layout.ARCHE_PYTHON_ENV,
        str(tmp_path / "override" / "bin" / "python"),
    )

    with pytest.raises(ValueError, match="标准 Python"):
        environment_layout.environment_path("fairchem")


def test_standard_absolute_runtime_path_is_preserved(monkeypatch):
    monkeypatch.delenv(environment_layout.ARCHE_PYTHON_ENV, raising=False)
    monkeypatch.delenv(
        environment_layout.MINICHEM_RUNTIME_PYTHON_ENV, raising=False
    )

    configured = STANDARD_PYTHON.parent.parent

    assert (
        environment_layout.resolve_runtime_path("fairchem", configured)
        == configured.resolve()
    )


def test_nonstandard_absolute_runtime_path_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="标准环境"):
        environment_layout.resolve_runtime_path(
            "fairchem", tmp_path / "external-runtime"
        )
