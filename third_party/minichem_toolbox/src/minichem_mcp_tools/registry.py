"""Register the complete task-independent atomic Action catalog with FastMCP."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from minichem_toolbox.catalog import (
    action_specs,
    catalog_snapshot,
    mcp_action_description,
    public_catalog_snapshot,
    resolve_tool_discovery_mode,
    validate_catalog,
)
from minichem_toolbox.models import ActionRequest, ActionSpec
from minichem_toolbox.service import execute_action
from minichem_toolbox.idempotency import (
    scoped_submission_request_error,
    validate_submission_key,
)
from minichem_toolbox.scope import current_scope

from .tracing import execute_traced
from .discovery_tools import register_progressive_discovery_tools
from .open_tools import register_open_execution_tools
from .software_catalog import open_execution_prompt, software_resource_snapshot


PACKAGE_ROOT = Path(__file__).resolve().parent
TOOL_CONFIG_PATH = PACKAGE_ROOT / "tool_config.json"
class ToolRegistryError(RuntimeError):
    """Raised when the public atomic catalog cannot be registered safely."""


class GaussianActionRequest(ActionRequest):
    """Gaussian 专用公开请求 Schema；资源由服务统一注入。"""


class GaussianCapableActionRequest(ActionRequest):
    """共享 Action 的公开 Schema；资源由服务统一注入。"""


@dataclass(frozen=True)
class ToolRecord:
    """Compatibility view over an ActionSpec; there are no per-task enabled flags."""

    module_stem: str
    module_name: str
    path: Path
    enabled: bool
    spec: ActionSpec
    module: None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "module_stem": self.module_stem,
            "module_name": self.module_name,
            "path": str(self.path),
            "enabled": True,
            "spec": self.spec.as_dict(),
            "error": self.error,
        }


def load_tool_config() -> dict[str, Any]:
    if not TOOL_CONFIG_PATH.is_file():
        return {}
    value = json.loads(TOOL_CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Tool configuration must be a JSON object: {TOOL_CONFIG_PATH}")
    return value


def discovered_module_stems() -> list[str]:
    """Return every public action id; task/profile filtering is intentionally absent."""

    return sorted(action_specs())


def tool_is_enabled(module_stem: str, config: dict[str, Any] | None = None) -> bool:
    return module_stem in action_specs()


def configuration_errors(config: dict[str, Any] | None = None) -> list[str]:
    value = config or load_tool_config()
    errors: list[str] = []
    if value.get("exposure_policy", "progressive_discovery") not in {
        "progressive_discovery",
        "atomic_all",
    }:
        errors.append(
            "tool_config.json exposure_policy must be progressive_discovery or atomic_all"
        )
    if value.get("backend_selection_policy", "per_action_explicit") != "per_action_explicit":
        errors.append("tool_config.json backend_selection_policy must be per_action_explicit")
    if value.get("automatic_fallback", False) is not False:
        errors.append("tool_config.json automatic_fallback must be false")
    if value.get("open_execution_policy") != "agent_explicit":
        errors.append("tool_config.json open_execution_policy must be agent_explicit")
    if value.get("execution_layers") != [
        "predefined_actions",
        "native_software",
        "programmable_analysis",
    ]:
        errors.append("tool_config.json must declare the three execution layers in order")
    if value.get("native_shell") is not False or value.get("programmable_shell") is not False:
        errors.append("tool_config.json native_shell and programmable_shell must be false")
    if "enabled_tools" in value or "disabled_tools" in value:
        errors.append(
            "tool_config.json cannot contain enabled_tools/disabled_tools because the "
            "benchmark exposes the complete catalog for every task"
        )
    return errors


def discover_tools(*, include_disabled: bool = True, strict: bool = False) -> list[ToolRecord]:
    del include_disabled
    errors = configuration_errors()
    try:
        validate_catalog()
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    if strict and errors:
        raise ToolRegistryError("; ".join(errors))
    return [
        ToolRecord(
            module_stem=specification.id,
            module_name=f"minichem_toolbox.actions.{specification.id}",
            path=PACKAGE_ROOT / "<generated-from-action-catalog>",
            enabled=True,
            spec=specification,
            error="; ".join(errors) if errors else None,
        )
        for specification in action_specs().values()
    ]


def _execute_public_registered_action(
    specification: ActionSpec,
    request: ActionRequest,
) -> dict[str, Any]:
    """执行一个兼容 Action，并返回服务端的完整 ActionResult。"""

    return execute_action(
        specification.id,
        request,
        operation_id=(None if current_scope() is not None else request.operation_id),
        submission_key=request.submission_key,
    )


def _restore_internal_request_identity(
    request: ActionRequest,
) -> tuple[ActionRequest, str | None, str | None, dict[str, Any] | None]:
    """在公开请求类型转换前保存并核对内部幂等身份。

    ``operation_id`` 为内部字段，Pydantic 的公开 ``model_dump`` 会有意排除它；
    这里先读取该字段，再把它放回内部 ``ActionRequest``，以便 legacy 调用继续
    走持久幂等路径。认证 scope 只接受合法 ``submission_key``，伪造的
    ``operation_id`` 交给服务边界返回稳定的 invalid_request，不执行副作用。
    """

    operation_id = request.operation_id
    submission_key = request.submission_key
    identity_error = scoped_submission_request_error(request)
    if submission_key is not None:
        submission_key = validate_submission_key(submission_key)
    payload = request.model_dump(mode="python")
    if operation_id is not None:
        payload["operation_id"] = operation_id
    if submission_key is not None:
        payload["submission_key"] = submission_key
    restored = ActionRequest.model_validate(payload)
    return restored, operation_id, submission_key, identity_error


def _make_action_callable(specification: ActionSpec):
    if (
        specification.backend_ids == ("gaussian",)
        or specification.primary_output == "GaussianInputDeck"
        or specification.id == "generate_gaussian_code"
    ):
        public_request_type = GaussianActionRequest
    elif "gaussian" in specification.backend_ids:
        public_request_type = GaussianCapableActionRequest
    else:
        public_request_type = ActionRequest

    def invoke(request: ActionRequest) -> dict[str, Any]:
        request, _operation_id, _submission_key, _identity_error = (
            _restore_internal_request_identity(request)
        )
        arguments = {"request": request.model_dump(mode="json")}
        return execute_traced(
            specification.id,
            arguments,
            partial(_execute_public_registered_action, specification, request),
        )

    invoke.__name__ = specification.id
    invoke.__qualname__ = specification.id
    invoke.__annotations__["request"] = public_request_type
    invoke.__doc__ = mcp_action_description(specification)
    return invoke


def register_all_tools(mcp) -> list[str]:
    """Register the historical one-MCP-tool-per-Action compatibility surface."""

    discover_tools(strict=True)
    registered = []
    for specification in action_specs().values():
        function = _make_action_callable(specification)
        mcp.tool(
            name=specification.id,
            description=mcp_action_description(specification),
        )(function)
        registered.append(specification.id)
    registered.extend(register_open_execution_tools(mcp))
    return registered


def register_progressive_tools(mcp) -> list[str]:
    """Register compact discovery/dispatch plus native/program primitives."""

    discover_tools(strict=True)
    registered = register_progressive_discovery_tools(mcp)
    registered.extend(register_open_execution_tools(mcp))
    return registered


def register_public_tools(mcp, discovery_mode: str | None = None) -> list[str]:
    """Register the selected transport surface over the same complete catalog."""

    mode = resolve_tool_discovery_mode(discovery_mode)
    return register_all_tools(mcp) if mode == "full" else register_progressive_tools(mcp)


def register_catalog_resources(
    mcp,
    discovery_mode: str | None = None,
    snapshot: dict[str, Any] | None = None,
) -> None:
    """注册同一冻结目录上的完整只读 Action、软件和执行策略资源。"""

    mode = resolve_tool_discovery_mode(discovery_mode)
    frozen_snapshot = deepcopy(snapshot) if snapshot is not None else None

    @mcp.resource(
        "minichem://catalog",
        description="按需读取公开的 Action、Backend 和资源元数据。",
    )
    def complete_catalog() -> str:
        return json.dumps(
            public_catalog_snapshot(frozen_snapshot),
            ensure_ascii=False,
            indent=2,
        )

    @mcp.resource(
        "minichem://software",
        description="按需读取 native 与 analysis 执行层的软件能力清单。",
    )
    def complete_software_inventory() -> str:
        return json.dumps(software_resource_snapshot(), ensure_ascii=False, indent=2)

    @mcp.resource(
        "minichem://execution-policy",
        description="按需读取执行层与提交策略说明。",
    )
    def execution_policy() -> str:
        return open_execution_prompt(include_command_index=mode == "full")
