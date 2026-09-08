"""Shared constructors for static ActionSpec declarations."""

from __future__ import annotations

from .models import ActionSpec


def action(
    action_id: str,
    category: str,
    description: str,
    primary_output: str,
    backends: tuple[str, ...],
    required: tuple[str, ...],
    optional: tuple[str, ...] = (),
    *,
    input_description: str = "",
    data_action: bool = False,
    requires_network: bool = False,
    selection_policy: str | None = None,
    batch_safe: bool = False,
) -> ActionSpec:
    return ActionSpec(
        id=action_id,
        category=category,
        description=description,
        primary_output=primary_output,
        backend_ids=backends,
        required_inputs=required,
        optional_inputs=optional,
        input_description=input_description,
        data_action=data_action,
        requires_network=requires_network,
        selection_policy=(
            selection_policy
            or ("fixed_source" if data_action else "agent_backend_required")
        ),
        batch_safe=batch_safe,
    )
