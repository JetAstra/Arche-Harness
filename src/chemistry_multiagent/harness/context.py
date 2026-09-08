"""Out-of-band context for stable Harness task/workspace identity.

Compiled model requests are content-addressed ARCHE contracts.  Harness routing
metadata therefore travels through a context variable rather than being appended
to the request and changing its schema/hash.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator, Mapping


@dataclass(frozen=True)
class HarnessCallContext:
    task_scope_id: str
    operation: str
    operation_instance_id: str
    display_context: Mapping[str, Any]

    @property
    def combination_scope_id(self) -> str | None:
        batch = self.display_context.get("batch_index")
        combination = self.display_context.get("combination_index")
        if isinstance(batch, int) and not isinstance(batch, bool) and isinstance(combination, int) and not isinstance(combination, bool):
            return f"batch-{batch:03d}-combination-{combination:03d}"
        return None


_CURRENT: ContextVar[HarnessCallContext | None] = ContextVar(
    "arche_harness_call_context", default=None
)


def current_harness_call_context() -> HarnessCallContext | None:
    return _CURRENT.get()


@contextmanager
def bind_harness_call_context(
    *,
    task_scope_id: str,
    operation: str,
    operation_instance_id: str,
    display_context: Mapping[str, Any] | None = None,
) -> Iterator[HarnessCallContext]:
    context = HarnessCallContext(
        task_scope_id=str(task_scope_id),
        operation=str(operation),
        operation_instance_id=str(operation_instance_id),
        display_context=dict(display_context or {}),
    )
    token = _CURRENT.set(context)
    try:
        yield context
    finally:
        _CURRENT.reset(token)


__all__ = [
    "HarnessCallContext",
    "bind_harness_call_context",
    "current_harness_call_context",
]
