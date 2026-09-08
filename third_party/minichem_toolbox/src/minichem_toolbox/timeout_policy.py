"""Evaluator-controlled timeout policy for managed toolbox execution."""

from __future__ import annotations

import os
from typing import Literal


ExecutionClass = Literal["compute", "fast"]

COMPUTE_ACTION_TIMEOUT_ENV = "RESEARCHCHEMBENCH_COMPUTE_ACTION_TIMEOUT_SECONDS"
FAST_ACTION_TIMEOUT_ENV = "RESEARCHCHEMBENCH_FAST_ACTION_TIMEOUT_SECONDS"

DEFAULT_COMPUTE_ACTION_TIMEOUT_SECONDS = 7200
DEFAULT_FAST_ACTION_TIMEOUT_SECONDS = 60


def _positive_environment_integer(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def compute_action_timeout_seconds() -> int:
    return _positive_environment_integer(
        COMPUTE_ACTION_TIMEOUT_ENV,
        DEFAULT_COMPUTE_ACTION_TIMEOUT_SECONDS,
    )


def fast_action_timeout_seconds() -> int:
    value = _positive_environment_integer(
        FAST_ACTION_TIMEOUT_ENV,
        DEFAULT_FAST_ACTION_TIMEOUT_SECONDS,
    )
    if value > compute_action_timeout_seconds():
        raise ValueError(
            f"{FAST_ACTION_TIMEOUT_ENV} cannot exceed {COMPUTE_ACTION_TIMEOUT_ENV}"
        )
    return value


def timeout_seconds_for(execution_class: ExecutionClass) -> int:
    if execution_class == "compute":
        return compute_action_timeout_seconds()
    if execution_class == "fast":
        return fast_action_timeout_seconds()
    raise ValueError(f"Unknown execution class: {execution_class!r}")


def timeout_policy_record(execution_class: ExecutionClass) -> dict[str, object]:
    return {
        "execution_class": execution_class,
        "timeout_seconds": timeout_seconds_for(execution_class),
        "source": "evaluation_policy",
        "agent_controllable": False,
    }


__all__ = [
    "COMPUTE_ACTION_TIMEOUT_ENV",
    "DEFAULT_COMPUTE_ACTION_TIMEOUT_SECONDS",
    "DEFAULT_FAST_ACTION_TIMEOUT_SECONDS",
    "ExecutionClass",
    "FAST_ACTION_TIMEOUT_ENV",
    "compute_action_timeout_seconds",
    "fast_action_timeout_seconds",
    "timeout_policy_record",
    "timeout_seconds_for",
]
