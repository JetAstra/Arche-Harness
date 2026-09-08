"""ARCHE 对官方 Codex Python SDK 0.149.0 的固定适配层。

本模块只使用仓库安装的 ``openai-codex`` 包，并把每个 Planner 会话连接到
仓库内的 0.149.0 CLI 包装器。它负责 app-server 的线程启动、恢复、turn 事件
落盘和超时处理；上层 Runner 继续负责 ARCHE 的请求、工作区和会话清单语义。
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import inspect
import json
import math
import os
import queue
import re
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, Callable, Mapping, NoReturn, Sequence
from urllib.parse import urlsplit

from openai_codex import ApprovalMode, Codex, CodexConfig, GoalCheckpoint, Sandbox
from openai_codex.errors import GoalNotFoundError, TransportClosedError, TransportDiagnostic

from .base import HarnessError, HarnessRequest, HarnessResponse
from .codex_auth import (
    CODEX_AUTH_MODE_LOCAL_LOGIN,
    CODEX_INVESTIGATION_PERMISSION_PROFILE,
    CODEX_LOCAL_LOGIN_ENV_KEY,
    CODEX_AUTH_SOURCE_ENV,
    normalize_codex_auth_mode,
)
from .codex_process_sandbox import (
    CodexProcessSandboxSpec,
    build_codex_process_sandbox_argv,
    build_codex_process_sandbox_environment,
    prepare_codex_process_sandbox,
)
from .session_store import (
    CodexSessionOwner,
    CodexSessionStore,
    SchemaV1CodexSessionOwnerReader,
    _atomic_write_json_file,
    _remove_file_and_fsync_directory,
)


CODEX_PROFILE_LEGACY_PLANNER = "legacy_planner"
CODEX_PROFILE_INVESTIGATION = "investigation"
CODEX_PROFILE_TASK_REVIEW = "task_review"
# ``review`` 仍是外部旧请求的兼容输入；内部能力策略统一使用 task_review。
CODEX_PROFILE_REVIEW = CODEX_PROFILE_TASK_REVIEW
DEFAULT_CODEX_PROFILE = CODEX_PROFILE_TASK_REVIEW
_INVESTIGATION_GOAL_REQUEST_KINDS = frozenset(
    {
        "initial_investigation",
        "deepseek_repair",
        "investigation_job_wake",
        "investigation_job_continuation",
    }
)
# 普通 provider 事件流的有限空闲检测；它不是 Investigation 的总体 walltime。
DEFAULT_CODEX_IDLE_TIMEOUT_SECONDS = 300.0
# Investigation 模型可能在一次工具回执之后继续进行较长的科学推理；该值仍是
# 有限的 SDK transport watchdog，不是 turn 的总时限，也不适用于 Task Review。
DEFAULT_INVESTIGATION_IDLE_TIMEOUT_SECONDS = 1800.0
# MiniChem 与 evidence MCP 的单次工具调用及已确认外部作业等待局部时限。它不是
# Investigation turn 的总体时限；工具调用期间仍由事件或 progress 通知维持活动状态。
DEFAULT_MCP_TOOL_IDLE_TIMEOUT_SECONDS = 21600.0
# 这些工具会返回一个仍在外部运行的作业。状态回执结束后，Codex 可能在下一次
# 查询前暂时不产生 SDK 事件；适配器只在已核对同一 job_id 时使用工具级有限预算。
_ASYNC_JOB_SUBMIT_TOOLS = frozenset(
    {
        "submit_action_job",
        "submit_native_job",
        "submit_analysis_program",
        "submit_transition_state_path_job",
    }
)
_ASYNC_JOB_STATUS_TOOLS = frozenset(
    {
        "get_execution_job",
        "get_transition_state_path_job",
    }
)
_ASYNC_JOB_COLLECT_TOOLS = frozenset(
    {
        "collect_execution_job",
        "collect_transition_state_path_job",
    }
)
_ASYNC_JOB_NON_TERMINAL_STATUSES = frozenset(
    {"queued", "pending", "running", "active", "submitted"}
)
_ASYNC_JOB_TERMINAL_STATUSES = frozenset(
    {"succeeded", "failed", "cancelled", "expired", "unknown", "completed"}
)
_ASYNC_JOB_STATUS_ALIASES = {
    "done": "completed",
}
_ASYNC_JOB_PAYLOAD_NODE_LIMIT = 256

# Codex shell 只继承 core 环境，并再次排除常见 provider 凭据。这里仅保存变量名，
# 不读取或记录变量值；即使父进程为某个 provider 设置了密钥，也不会把它传给 shell。
_SHELL_ENV_EXCLUDES = (
    "ARCHE_CODEX_API_KEY",
    "ARCHE_CHEM_API_KEY",
    "ARCHE_REVIEW_CODEX_API_KEY",
    "DEEPSEEK_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "HF_TOKEN",
)
_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_POLICY_DIGEST = re.compile(r"^[0-9a-f]{64}$")
CODEX_TERMINAL_PROVIDER_ERROR_CODES = frozenset({"invalid_encrypted_content"})

# 这些是 Codex 0.149 中已经存在、但当前化学 Harness 不需要的动态能力。
# 它们通过 thread config 明确关闭，避免依赖空 CODEX_HOME 的默认值；新增版本
# 的能力不会因此自动获得准入。shell/update_plan 不在此集合中，保持可用。
_DISABLED_CODEX_FEATURES = frozenset(
    {
        "apply_patch_freeform",
        "apply_patch_preserve_line_endings",
        "apply_patch_streaming_events",
        "apps",
        "browser_use",
        "browser_use_external",
        "browser_use_full_cdp_access",
        "code_mode",
        "code_mode_interrupt",
        "code_mode_only",
        "collaboration_modes",
        "computer_use",
        "default_mode_request_user_input",
        "enable_fanout",
        "enable_mcp_apps",
        "external_agent_memory_import",
        "goals",
        "image_generation",
        "in_app_browser",
        "multi_agent",
        "multi_agent_v2",
        "plugin_hooks",
        "plugin_sharing",
        "plugins",
        "recommended_plugins",
        "remote_control",
        "remote_plugin",
        "request_permissions_tool",
        "realtime_conversation",
        "search_tool",
        "send_async_message",
        "skill_env_var_dependency_prompt",
        "skill_mcp_dependency_install",
        "skill_search",
        "standalone_web_search",
        "tool_search",
        "tool_suggest",
        "view_image",
        "web_search_cached",
        "web_search_request",
    }
)
_KNOWN_CODEX_FEATURES = frozenset(
    {
        *_DISABLED_CODEX_FEATURES,
        "shell_tool",
        "unified_exec",
        "update_plan",
    }
)
_KNOWN_THREAD_CONFIG_KEYS = frozenset(
    {
        "compact_prompt",
        "model_context_window",
        "model_provider",
        "model_providers",
        "model_reasoning_summary",
        "model_supports_reasoning_summaries",
        "model_reasoning_effort",
        "mcp_servers",
        "features",
        "tools",
        "agents",
        "web_search",
        "shell_environment_policy",
        "sandbox_workspace_write",
    }
)
_KNOWN_CODEX_TOOL_CONFIG_KEYS = frozenset(
    {
        "experimental_request_user_input",
        "update_plan",
    }
)
_PROFILE_MCP_SERVERS = {
    CODEX_PROFILE_LEGACY_PLANNER: frozenset({"minichem_toolbox", "arche_planner_control"}),
    CODEX_PROFILE_INVESTIGATION: frozenset({"minichem_toolbox"}),
    CODEX_PROFILE_TASK_REVIEW: frozenset({"evidence_reader"}),
}
_HTTP_MCP_DESCRIPTOR_FIELDS = frozenset(
    {
        "url",
        "env_http_headers",
        "required",
        "startup_timeout_sec",
        "tool_timeout_sec",
        "default_tools_approval_mode",
    }
)
_TASK_REVIEW_MCP_DESCRIPTOR_FIELDS = frozenset(
    {*_HTTP_MCP_DESCRIPTOR_FIELDS, "read_only"}
)
_LEGACY_PLANNER_MCP_DESCRIPTOR_FIELDS = frozenset(
    {
        "command",
        "args",
        "env",
        "required",
        "startup_timeout_sec",
        "tool_timeout_sec",
        "default_tools_approval_mode",
    }
)
_LEGACY_PLANNER_MCP_ENV_NAMES = frozenset(
    {
        # 这些是 vendored Codex/MiniChem 第三方进程协议字段；ARCHE 生产配置
        # 在 Runner adapter 边界之外不再公开或读取它们。
        "MINICHEM_MCP_WORKSPACE",
        "RESEARCHCHEMBENCH_WORKSPACE",
        "ARCHE_PLANNER_CONTROL_CONTEXT",
    }
)
_LOCAL_MCP_HOSTNAMES = frozenset({"127.0.0.1", "localhost", "::1"})

# ``investigation_and_experiment`` 是当前迁移阶段的业务名称。保留短名称便于
# 独立 Harness 调用方显式传入 profile，而旧 workflow 阶段仍自动落到 Review。
_INVESTIGATION_STAGES = frozenset(
    {"investigation", "investigation_and_experiment", "experiment"}
)
_PERSISTENT_CODEX_ROLES = frozenset(
    {
        "planner",
        "investigation",
        "review",
        "task_review",
        "investigation_and_experiment",
        "experiment",
    }
)


def normalize_codex_profile(profile: str | None) -> str:
    """规范化三类 Codex profile，并把旧 review 名称映射到 task_review。"""

    normalized = str(profile or "").strip().lower().replace("-", "_")
    if normalized in {"investigation", "investigation_and_experiment", "experiment"}:
        return CODEX_PROFILE_INVESTIGATION
    if normalized in {"task_review", "review"}:
        return CODEX_PROFILE_TASK_REVIEW
    if normalized in {"legacy_planner", "planner", "legacy"}:
        return CODEX_PROFILE_LEGACY_PLANNER
    if not normalized:
        return DEFAULT_CODEX_PROFILE
    raise HarnessError(
        f"unsupported Codex Harness profile: {profile}",
        kind="provider_envelope",
    )


def resolve_codex_profile(request: HarnessRequest) -> str:
    """根据请求显式 profile 或阶段名称选择能力策略。

    旧 Planner 请求没有 profile 字段时保持 Review/deny_all；只有明确的
    Investigation 阶段才自动选择可执行的调查策略。
    """

    role = str(request.role).strip().lower().replace("-", "_")
    if role == CODEX_PROFILE_INVESTIGATION or role in _INVESTIGATION_STAGES:
        return CODEX_PROFILE_INVESTIGATION
    if role == "planner":
        return CODEX_PROFILE_LEGACY_PLANNER
    if role in {"review", "task_review"}:
        return CODEX_PROFILE_TASK_REVIEW
    explicit = request.profile
    if explicit is None:
        for key in ("codex_profile", "profile"):
            value = request.metadata.get(key)
            if isinstance(value, str) and value.strip():
                explicit = value
                break
    if explicit is not None:
        return normalize_codex_profile(explicit)
    if str(request.stage).strip().lower().replace("-", "_") in _INVESTIGATION_STAGES:
        return CODEX_PROFILE_INVESTIGATION
    return DEFAULT_CODEX_PROFILE


def _validate_environment_names(values: Sequence[str]) -> list[str]:
    """校验只用于进程白名单和 shell 排除清单的环境变量名。"""

    normalized: list[str] = []
    for raw_name in values:
        name = str(raw_name).strip()
        if _ENVIRONMENT_NAME.fullmatch(name) is None:
            raise HarnessError(
                "Codex sensitive environment name is malformed",
                kind="provider_envelope",
            )
        if name not in normalized:
            normalized.append(name)
    return normalized


def _validate_secret_environment(
    values: Mapping[str, str] | None,
) -> dict[str, str]:
    """复制 app-server 私有环境，并拒绝空名、空值和非文本秘密。"""

    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise HarnessError(
            "Codex secret environment must be an object",
            kind="provider_envelope",
        )
    normalized: dict[str, str] = {}
    for raw_name, raw_value in values.items():
        if not isinstance(raw_name, str) or _ENVIRONMENT_NAME.fullmatch(raw_name) is None:
            raise HarnessError(
                "Codex secret environment name is malformed",
                kind="provider_envelope",
            )
        if not isinstance(raw_value, str) or not raw_value:
            raise HarnessError(
                "Codex secret environment value is missing",
                kind="provider_envelope",
            )
        normalized[raw_name] = raw_value
    return normalized


def _validate_mcp_descriptor_fields(
    descriptor: Mapping[str, Any],
    *,
    server_name: str,
    allow_read_only: bool,
) -> None:
    """拒绝 HTTP MCP 描述符中的本地进程启动字段和未知字段。"""

    allowed_fields = (
        _TASK_REVIEW_MCP_DESCRIPTOR_FIELDS
        if allow_read_only
        else _HTTP_MCP_DESCRIPTOR_FIELDS
    )
    unknown_fields = set(descriptor).difference(allowed_fields)
    if unknown_fields:
        raise HarnessError(
            f"Codex MCP server {server_name} has unsupported descriptor fields: "
            f"{sorted(str(field) for field in unknown_fields)}",
            kind="provider_envelope",
        )
    approval_mode = descriptor.get("default_tools_approval_mode")
    if approval_mode is not None and approval_mode != "approve":
        raise HarnessError(
            f"Codex MCP server {server_name} has unsupported tool approval mode",
            kind="provider_envelope",
        )


def _validate_mcp_local_http_endpoint(endpoint: Any, *, server_name: str) -> None:
    """核对 MCP 只能访问本地无凭据的固定 HTTP ``/mcp`` 端点。"""

    if not isinstance(endpoint, str) or not endpoint.strip():
        raise HarnessError(
            f"Codex MCP server {server_name} requires an HTTP endpoint",
            kind="provider_envelope",
        )
    parsed_endpoint = None
    hostname = None
    try:
        parsed_endpoint = urlsplit(endpoint.strip())
        hostname = parsed_endpoint.hostname
        parsed_endpoint.port
    except ValueError:
        parsed_endpoint = None
    if (
        parsed_endpoint is None
        or parsed_endpoint.scheme.lower() != "http"
        or hostname is None
        or hostname.lower() not in _LOCAL_MCP_HOSTNAMES
        or parsed_endpoint.username is not None
        or parsed_endpoint.password is not None
        or parsed_endpoint.query
        or parsed_endpoint.fragment
        or parsed_endpoint.path.rstrip("/") != "/mcp"
    ):
        raise HarnessError(
            f"Codex MCP server {server_name} endpoint must be a local HTTP /mcp endpoint",
            kind="provider_envelope",
        )


def _validate_mcp_required_http_headers(
    headers: Any,
    *,
    server_name: str,
) -> None:
    """校验 HTTP header 只能引用环境变量名，绝不接受 header 秘密正文。"""

    if not isinstance(headers, Mapping) or not headers:
        raise HarnessError(
            f"Codex MCP server {server_name} requires environment-backed HTTP headers",
            kind="provider_envelope",
        )
    for header_name, environment_name in headers.items():
        if (
            not isinstance(header_name, str)
            or not header_name.strip()
            or not isinstance(environment_name, str)
            or _ENVIRONMENT_NAME.fullmatch(environment_name.strip()) is None
        ):
            raise HarnessError(
                f"Codex MCP server {server_name} has malformed HTTP header binding",
                kind="provider_envelope",
            )


def _validate_mcp_timeout_values(
    descriptor: Mapping[str, Any],
    *,
    server_name: str,
) -> None:
    """核对 MCP 启动和工具超时均为有限的正数。"""

    for timeout_name in ("startup_timeout_sec", "tool_timeout_sec"):
        timeout = descriptor.get(timeout_name)
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or timeout <= 0
        ):
            raise HarnessError(
                f"Codex MCP server {server_name} has invalid {timeout_name}",
                kind="provider_envelope",
            )


def _validate_http_mcp_descriptor(
    descriptor: Mapping[str, Any],
    *,
    server_name: str,
    allow_read_only: bool,
) -> None:
    """校验 HTTP MCP 描述符，只接受固定本地端点和非敏感连接参数。

    输入是即将传给 Codex 的一个 MCP 描述符；动作是拒绝 command、args、env、cwd
    以及其它未列入白名单的字段，并核对端点、必需标志、超时和环境变量名；没有
    本地进程启动能力或 header 秘密正文时返回 ``None``，否则抛出配置错误。
    """

    _validate_mcp_descriptor_fields(
        descriptor,
        server_name=server_name,
        allow_read_only=allow_read_only,
    )
    _validate_mcp_local_http_endpoint(descriptor.get("url"), server_name=server_name)
    if descriptor.get("required") is not True:
        raise HarnessError(
            f"Codex MCP server {server_name} must be required",
            kind="provider_envelope",
        )
    _validate_mcp_required_http_headers(
        descriptor.get("env_http_headers"),
        server_name=server_name,
    )
    timeout_values = dict(descriptor)
    # Planner control server 只有启动超时；缺少工具超时时沿用同一有限值，
    # 仍然拒绝显式的空值、负值或无限值。
    timeout_values.setdefault(
        "tool_timeout_sec",
        timeout_values.get("startup_timeout_sec"),
    )
    _validate_mcp_timeout_values(timeout_values, server_name=server_name)
    if allow_read_only and descriptor.get("read_only", True) is not True:
        raise HarnessError(
            "task_review Evidence MCP must be explicitly read-only",
            kind="provider_envelope",
        )


def _validate_codex_feature_configuration(features: Any) -> None:
    """校验 thread features 对象只包含 0.149 白名单能力。"""

    if features is None:
        return
    if not isinstance(features, Mapping):
        raise HarnessError("Codex features must be an object", kind="provider_envelope")
    unknown_features = set(features).difference(_KNOWN_CODEX_FEATURES)
    if unknown_features:
        raise HarnessError(
            f"Codex features have unsupported tools: {sorted(unknown_features)}",
            kind="provider_envelope",
        )
    invalid_values = [
        name for name, value in features.items() if not isinstance(value, bool)
    ]
    if invalid_values:
        raise HarnessError(
            "Codex feature values must be boolean: "
            + ", ".join(sorted(str(name) for name in invalid_values)),
            kind="provider_envelope",
        )


def _validate_codex_tool_configuration(tools: Any) -> None:
    """校验 thread tools 对象及每个工具的 enabled 字段。"""

    if tools is None:
        return
    if not isinstance(tools, Mapping):
        raise HarnessError("Codex tools must be an object", kind="provider_envelope")
    unknown_tools = set(tools).difference(_KNOWN_CODEX_TOOL_CONFIG_KEYS)
    if unknown_tools:
        raise HarnessError(
            f"Codex tools have unsupported keys: {sorted(unknown_tools)}",
            kind="provider_envelope",
        )
    for tool_name, tool_config in tools.items():
        if not isinstance(tool_config, Mapping):
            raise HarnessError(
                f"Codex tools.{tool_name} must be an object",
                kind="provider_envelope",
            )
        unknown_fields = set(tool_config).difference({"enabled"})
        if unknown_fields:
            raise HarnessError(
                f"Codex tools.{tool_name} has unsupported fields: "
                f"{sorted(unknown_fields)}",
                kind="provider_envelope",
            )
        enabled = tool_config.get("enabled")
        if enabled is not None and not isinstance(enabled, bool):
            raise HarnessError(
                f"Codex tools.{tool_name}.enabled must be boolean",
                kind="provider_envelope",
            )


def _validate_profile_mcp_servers(
    servers: Any,
    *,
    profile: str,
) -> None:
    """校验 profile 的 MCP server 名称、描述符和本地 HTTP 连接边界。"""

    if servers is None:
        return
    if not isinstance(servers, Mapping):
        raise HarnessError("Codex mcp_servers must be an object", kind="provider_envelope")
    allowed_servers = _PROFILE_MCP_SERVERS[profile]
    unknown_servers = set(servers).difference(allowed_servers)
    if unknown_servers:
        raise HarnessError(
            f"Codex {profile} profile has unsupported MCP servers: {sorted(unknown_servers)}",
            kind="provider_envelope",
        )
    for name, descriptor in servers.items():
        if not isinstance(name, str) or not name.strip():
            raise HarnessError("Codex MCP server name is malformed", kind="provider_envelope")
        if not isinstance(descriptor, Mapping):
            raise HarnessError(
                f"Codex MCP server {name} configuration must be an object",
                kind="provider_envelope",
            )
        if profile == CODEX_PROFILE_TASK_REVIEW and name == "evidence_reader":
            # Review 只能连接固定的本地只读 Evidence HTTP 服务；描述符不能借由
            # command、args、env 或 cwd 启动另一个本地程序。
            _validate_http_mcp_descriptor(
                descriptor,
                server_name=name,
                allow_read_only=True,
            )
        if profile == CODEX_PROFILE_LEGACY_PLANNER:
            _validate_legacy_planner_mcp_descriptor(
                descriptor,
                server_name=name,
            )
        if profile == CODEX_PROFILE_INVESTIGATION and name == "minichem_toolbox":
            # Investigation 的 MiniChem 也只能走预期本地 HTTP 服务，不能通过
            # 未知 descriptor 字段扩展为任意 stdio 进程或携带 literal secret。
            _validate_http_mcp_descriptor(
                descriptor,
                server_name=name,
                allow_read_only=False,
            )


def _validate_legacy_planner_mcp_fields(
    descriptor: Mapping[str, Any],
    *,
    server_name: str,
) -> None:
    """拒绝 Planner stdio 描述符中的未知字段。"""

    unknown_fields = set(descriptor).difference(_LEGACY_PLANNER_MCP_DESCRIPTOR_FIELDS)
    if unknown_fields:
        raise HarnessError(
            f"Codex MCP server {server_name} has unsupported descriptor fields: "
            f"{sorted(str(field) for field in unknown_fields)}",
            kind="provider_envelope",
        )
    approval_mode = descriptor.get("default_tools_approval_mode")
    if approval_mode is not None and approval_mode != "approve":
        raise HarnessError(
            f"Codex MCP server {server_name} has unsupported tool approval mode",
            kind="provider_envelope",
        )


def _validate_legacy_planner_mcp_command(
    descriptor: Mapping[str, Any],
    *,
    server_name: str,
) -> None:
    """校验 Planner MCP command、args 均为无控制字符的固定文本。"""

    command = descriptor.get("command")
    if (
        not isinstance(command, str)
        or not command.strip()
        or not Path(command.strip()).is_absolute()
        or "\x00" in command
    ):
        raise HarnessError(
            f"Codex MCP server {server_name} command must be an absolute path",
            kind="provider_envelope",
        )
    args = descriptor.get("args", [])
    if not isinstance(args, list) or any(
        not isinstance(item, str) or "\x00" in item for item in args
    ):
        raise HarnessError(
            f"Codex MCP server {server_name} args are malformed",
            kind="provider_envelope",
        )


def _validate_legacy_planner_mcp_environment(
    descriptor: Mapping[str, Any],
    *,
    server_name: str,
) -> None:
    """校验 Planner MCP 环境名和值，不允许空值或控制字符。"""

    environment = descriptor.get("env", {})
    if not isinstance(environment, Mapping):
        raise HarnessError(
            f"Codex MCP server {server_name} env is malformed",
            kind="provider_envelope",
        )
    for name, value in environment.items():
        if (
            not isinstance(name, str)
            or name not in _LEGACY_PLANNER_MCP_ENV_NAMES
            or _ENVIRONMENT_NAME.fullmatch(name) is None
            or not isinstance(value, str)
            or not value
            or "\x00" in value
            or not Path(value).is_absolute()
        ):
            raise HarnessError(
                f"Codex MCP server {server_name} env is malformed",
                kind="provider_envelope",
            )


def _validate_legacy_planner_mcp_timeouts(
    descriptor: Mapping[str, Any],
    *,
    server_name: str,
) -> None:
    """校验 Planner MCP 必须 required 且超时有限为正数。"""

    if descriptor.get("required") is not True:
        raise HarnessError(
            f"Codex MCP server {server_name} must be required",
            kind="provider_envelope",
        )
    timeout_values = dict(descriptor)
    # Planner control server 只有启动超时；缺少工具超时时沿用同一有限值，
    # 仍然拒绝显式的空值、负值或无限值。
    timeout_values.setdefault(
        "tool_timeout_sec",
        timeout_values.get("startup_timeout_sec"),
    )
    _validate_mcp_timeout_values(timeout_values, server_name=server_name)


def _validate_legacy_planner_mcp_descriptor(
    descriptor: Mapping[str, Any],
    *,
    server_name: str,
) -> None:
    """校验 Planner 内置 stdio MCP 描述符，不接受任意进程扩展字段。

    Planner 的两个 MCP server 由 Runner 生成并在会话间复用；这里保留官方
    0.149 的 command/args/env 连接形式，但拒绝 cwd、URL、header 正文和未知
    工具字段。命令必须是绝对路径，环境只允许非空的普通文本，超时必须有限且
    为正数；不启动、不允许的 server 名称由上层 profile 白名单另行拒绝。
    """

    _validate_legacy_planner_mcp_fields(descriptor, server_name=server_name)
    _validate_legacy_planner_mcp_command(descriptor, server_name=server_name)
    _validate_legacy_planner_mcp_environment(descriptor, server_name=server_name)
    _validate_legacy_planner_mcp_timeouts(descriptor, server_name=server_name)


def _validate_profile_thread_configuration(
    thread_config: Mapping[str, Any],
    *,
    profile: str,
) -> None:
    """校验 profile 可见的 MCP 与工具配置，未知能力一律拒绝。"""

    unknown_keys = set(thread_config).difference(_KNOWN_THREAD_CONFIG_KEYS)
    if unknown_keys:
        raise HarnessError(
            f"Codex thread config has unsupported keys: {sorted(unknown_keys)}",
            kind="provider_envelope",
        )
    if "compact_prompt" in thread_config:
        if profile != CODEX_PROFILE_INVESTIGATION:
            raise HarnessError(
                "Codex compact prompt is only available for Investigation",
                kind="provider_envelope",
            )
        compact_prompt = thread_config["compact_prompt"]
        if not isinstance(compact_prompt, str) or not compact_prompt.strip():
            raise HarnessError(
                "Codex compact prompt must be a non-empty string",
                kind="provider_envelope",
            )
        if any(
            (ord(character) < 0x20 and character not in "\t\n\r")
            or ord(character) == 0x7F
            for character in compact_prompt
        ):
            raise HarnessError(
                "Codex compact prompt contains control characters",
                kind="provider_envelope",
            )
    if "model_context_window" in thread_config:
        if profile != CODEX_PROFILE_INVESTIGATION:
            raise HarnessError(
                "Codex model context window is only available for Investigation",
                kind="provider_envelope",
            )
        model_context_window = thread_config["model_context_window"]
        if (
            isinstance(model_context_window, bool)
            or not isinstance(model_context_window, int)
            or model_context_window <= 0
        ):
            raise HarnessError(
                "Codex model context window must be a positive integer",
                kind="provider_envelope",
            )
    _validate_codex_feature_configuration(thread_config.get("features"))
    _validate_codex_tool_configuration(thread_config.get("tools"))
    _validate_profile_mcp_servers(thread_config.get("mcp_servers"), profile=profile)


def _collect_mcp_environment_names(value: Any) -> tuple[str, ...]:
    """从 MCP ``env_http_headers`` 映射提取环境变量名，不读取其值。"""

    names: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) == "env_http_headers" and isinstance(item, Mapping):
                for raw_name in item.values():
                    if isinstance(raw_name, str):
                        names.append(raw_name.strip())
                    else:
                        names.append("\x00")
            names.extend(_collect_mcp_environment_names(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            names.extend(_collect_mcp_environment_names(item))
    return tuple(name for name in names if name)


def build_codex_capability_config(
    *,
    thread_config: Mapping[str, Any],
    workspace: Path,
    profile: str = DEFAULT_CODEX_PROFILE,
    env_key: str | None = None,
    environment_keys: Sequence[str] = (),
    auth_mode: str = "api",
    enable_investigation_goals: bool | None = None,
) -> dict[str, Any]:
    """叠加 Harness 能力策略并返回可传给 SDK 的 thread config。

    ``thread_config`` 中已有的 provider、MCP server 和其它明确配置会被保留；
    若给出 ``env_key`` 或 ``environment_keys``，它们会加入 shell 排除清单但只
    保存变量名，不读取变量值。MCP 的 ``env_http_headers`` 中引用的环境变量
    名也会自动加入排除清单。
    能力策略使用 0.149 的真实 ``features``、``web_search`` 和 ``tools`` 键构造
    完整内建工具面；Investigation 只保留 shell、update_plan 与 profile 允许的
    MCP，Review 只保留只读 Evidence MCP。shell 只继承 core 环境并排除凭据；
    工作区写入策略只允许当前 workspace，关闭网络，同时不挂载 slash-tmp 和
    TMPDIR。``enable_investigation_goals`` 只用于恢复已经冻结为 Goal 的
    Investigation session 或运行显式实验；省略时所有 profile 都关闭 Goal。
    该配置不是宿主机读取隔离承诺。
    """

    normalized_profile = normalize_codex_profile(profile)
    normalized_auth_mode = normalize_codex_auth_mode(auth_mode)
    if enable_investigation_goals is not None and not isinstance(
        enable_investigation_goals,
        bool,
    ):
        raise TypeError("enable_investigation_goals must be bool or None")
    if (
        enable_investigation_goals is True
        and normalized_profile != CODEX_PROFILE_INVESTIGATION
    ):
        raise HarnessError(
            "Goal capability is only available for Investigation",
            kind="provider_envelope",
        )
    goals_enabled = bool(enable_investigation_goals)
    if not isinstance(thread_config, Mapping):
        raise HarnessError("Codex thread config must be an object", kind="provider_envelope")
    _validate_profile_thread_configuration(thread_config, profile=normalized_profile)
    result = copy.deepcopy(dict(thread_config))
    mcp_servers = result.get("mcp_servers")
    if isinstance(mcp_servers, Mapping):
        # 线程审批固定为 never 后，Codex 不会弹出交互确认；这里仅对当前
        # profile 已通过名称、端点和连接字段白名单校验的 MCP 服务预授权。
        # capability token 仍由 HTTP header 校验，shell 与文件权限不会扩大。
        result["mcp_servers"] = {
            str(server_name): {
                **dict(descriptor),
                "default_tools_approval_mode": "approve",
            }
            for server_name, descriptor in mcp_servers.items()
        }
    features = result.get("features")
    feature_values = dict(features) if isinstance(features, Mapping) else {}
    feature_values.update({name: False for name in _DISABLED_CODEX_FEATURES})
    # 这些开关必须显式关闭，避免 0.149 默认启用目标、协作、图片和浏览能力；
    # apply_patch 也保持关闭，当前 Investigation 只开放 shell/update_plan。
    feature_values.update(
        {
            "apply_patch_freeform": False,
            "apply_patch_preserve_line_endings": False,
            "apply_patch_streaming_events": False,
            "apps": False,
            "enable_mcp_apps": False,
            # Goal 只给显式选择的 Investigation 使用；生产默认使用普通 turn。
            "goals": goals_enabled,
            "multi_agent": False,
            "multi_agent_v2": False,
            "view_image": False,
        }
    )
    if normalized_profile == CODEX_PROFILE_TASK_REVIEW:
        # Review 只读 Evidence MCP，不得通过 shell、unified_exec 或计划工具派生
        # 宿主机副作用；update_plan 的 0.149 原生对象开关在下方再次固定关闭。
        feature_values["shell_tool"] = False
        feature_values["unified_exec"] = False
        feature_values["update_plan"] = False
    else:
        feature_values["shell_tool"] = True
        feature_values["unified_exec"] = True
        # 计划工具是否出现由 0.149 ``tools.update_plan`` 对象控制；保留 feature
        # 关闭值可以防止其它动态入口绕过 profile，而不会影响原生计划注册。
        feature_values["update_plan"] = False
    result["features"] = feature_values
    result["tools"] = {
        "experimental_request_user_input": {"enabled": False},
        "update_plan": {
            "enabled": normalized_profile != CODEX_PROFILE_TASK_REVIEW,
        },
    }
    # ``agents.enabled`` 是 0.149 的稳定总开关；与 feature 开关同时关闭，
    # 防止未来某个多智能体入口不再复用旧 feature 名称。
    result["agents"] = {"enabled": False}
    result["web_search"] = "disabled"
    excluded_environment_names = list(_SHELL_ENV_EXCLUDES)
    if normalized_auth_mode == CODEX_AUTH_MODE_LOCAL_LOGIN:
        # local_login 的控制变量不应成为模型 shell 的读取入口；真正的登录
        # clone 也由权限 profile 拒绝读取。
        excluded_environment_names.extend(
            name
            for name in (CODEX_LOCAL_LOGIN_ENV_KEY, CODEX_AUTH_SOURCE_ENV)
            if name not in excluded_environment_names
        )
    candidate_environment_names: list[str] = []
    if isinstance(env_key, str) and env_key.strip():
        candidate_environment_names.append(env_key.strip())
    candidate_environment_names.extend(str(name).strip() for name in environment_keys)
    candidate_environment_names.extend(_collect_mcp_environment_names(result))
    for name in candidate_environment_names:
        if not _ENVIRONMENT_NAME.fullmatch(name):
            raise HarnessError(
                "Codex sensitive environment name is malformed",
                kind="provider_envelope",
            )
        if name not in excluded_environment_names:
            excluded_environment_names.append(name)
    result["shell_environment_policy"] = {
        "inherit": "core",
        "ignore_default_excludes": False,
        "exclude": excluded_environment_names,
    }
    result["sandbox_workspace_write"] = {
        "network_access": False,
        "exclude_slash_tmp": True,
        "exclude_tmpdir_env_var": True,
        "writable_roots": [str(workspace.resolve())],
    }
    # profile 仅由调用方用于选择 approval_mode，随响应 metadata 记录；不把
    # ARCHE 私有审计字段发送给只认识 Codex 配置的 app-server。
    return result


_POLICY_SECRET_KEY_MARKERS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "credential",
        "headers",
        "password",
        "secret",
        "token",
    }
)


def _policy_key_contains_secret_marker(key: str) -> bool:
    """判断配置字段名是否可能承载 header、token 或其它秘密正文。"""

    normalized = key.strip().lower().replace("-", "_")
    return any(marker in normalized for marker in _POLICY_SECRET_KEY_MARKERS)


def _canonicalize_policy_key_value(value: Any, key: str) -> tuple[bool, Any]:
    """把一个可能含凭据的策略字段转换为安全值。"""

    normalized_key = key.strip().lower().replace("-", "_")
    if normalized_key == "env_http_headers":
        if not isinstance(value, Mapping):
            return True, "[REDACTED]"
        return True, {
            str(header_name): (
                item
                if isinstance(item, str) and _ENVIRONMENT_NAME.fullmatch(item.strip())
                else "[REDACTED]"
            )
            for header_name, item in sorted(
                value.items(), key=lambda pair: str(pair[0])
            )
        }
    if normalized_key.endswith("_env") or normalized_key.endswith("_env_name"):
        if isinstance(value, str) and _ENVIRONMENT_NAME.fullmatch(value.strip()):
            return True, value.strip()
        return True, "[REDACTED]"
    if _policy_key_contains_secret_marker(normalized_key):
        return True, "[REDACTED]"
    return False, value


def _canonicalize_thread_policy_value(value: Any, *, key: str | None = None) -> Any:
    """生成只含非 secret 值的 thread policy 规范 JSON 投影。"""

    if key is not None:
        handled, safe_value = _canonicalize_policy_key_value(value, key)
        if handled:
            return safe_value
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise HarnessError(
                "Codex thread policy contains a non-finite number",
                kind="provider_envelope",
            )
        return value
    if isinstance(value, Mapping):
        return {
            str(item_key): _canonicalize_thread_policy_value(item, key=str(item_key))
            for item_key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize_thread_policy_value(item) for item in value]
    raise HarnessError(
        "Codex thread policy contains an unsupported value",
        kind="provider_envelope",
    )


def calculate_codex_thread_policy_digest(
    *,
    thread_config: Mapping[str, Any],
    workspace: Path,
    provider_id: str,
    env_key: str,
    model: str,
    role: str,
    stage: str,
    profile: str,
    auth_mode: str = "api",
    permission_profile: str | None = None,
) -> str:
    """计算不含 secret 的 effective Codex thread policy 摘要。

    输入是已经组装的 provider、MCP 和能力策略配置；动作是保留 endpoint、模型、
    provider、工作区、profile 和其它非敏感策略，主动把 literal header、token、API
    key 等字段替换为固定标记后计算 SHA-256。返回值只含十六进制摘要，绝不把 secret
    正文写入 request-intent、receipt、recovery、trace 或异常。``stage`` 参数仅为
    旧调用方兼容保留，不属于线程策略；同一 thread 的 generation、optimization
    和 repair 阶段因此共享摘要，而阶段差异仍由 request digest 绑定。
    """

    if not isinstance(thread_config, Mapping):
        raise HarnessError("Codex thread policy must be an object", kind="provider_envelope")
    policy = {
        "schema_version": 1,
        "auth_mode": str(auth_mode).strip(),
        "provider_id": str(provider_id).strip(),
        "env_key": str(env_key).strip(),
        "model": str(model).strip(),
        "role": str(role).strip(),
        "profile": normalize_codex_profile(profile),
        "permission_profile": (
            str(permission_profile).strip() if permission_profile is not None else None
        ),
        "workspace": str(Path(workspace).expanduser().resolve()),
        "thread_config": _canonicalize_thread_policy_value(thread_config),
    }
    if any(not isinstance(policy[field], str) or not policy[field] for field in (
        "provider_id",
        "env_key",
        "model",
        "role",
        "profile",
        "auth_mode",
        "workspace",
    )):
        raise HarnessError("Codex thread policy identity is incomplete", kind="provider_envelope")
    encoded = json.dumps(
        policy,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def approval_mode_for_codex_profile(profile: str) -> ApprovalMode:
    """根据 profile 校验输入并返回 SDK thread 的审批方式。

    输入是调用方选择的 Codex profile；动作是先规范化 profile 以拒绝未知策略，
    再固定返回 ``deny_all``，使所有工具调用都不会触发不可用的自动审批服务。
    返回值是 SDK 使用的 ``ApprovalMode``，不会修改外部状态；profile 不受支持时
    由规范化函数抛出 ``HarnessError``。
    """

    normalize_codex_profile(profile)
    return ApprovalMode.deny_all


def sandbox_for_codex_profile(
    profile: str,
    *,
    permission_profile: str | None = None,
) -> Sandbox | None:
    """根据 profile 校验输入并返回 SDK thread 的文件系统沙箱。

    输入是调用方选择的 Codex profile；动作是规范化 profile 并仅为 Investigation
    选择 ``full_access``，其它受支持 profile 继续使用 ``workspace_write``。
    Investigation 在显式启用自定义权限 profile 时返回 ``None``，让 0.149
    app-server 从 CODEX_HOME 的 ``default_permissions`` 读取读隔离配置；这不
    会关闭 shell/unified_exec，权限 profile 只限制其文件和网络边界。
    返回值是 SDK 使用的 ``Sandbox``，不会修改外部状态；profile 不受支持时由
    规范化函数抛出 ``HarnessError``。
    """

    normalized_profile = normalize_codex_profile(profile)
    if (
        normalized_profile == CODEX_PROFILE_INVESTIGATION
        and permission_profile == CODEX_INVESTIGATION_PERMISSION_PROFILE
    ):
        return None
    if normalized_profile == CODEX_PROFILE_INVESTIGATION:
        return Sandbox.full_access
    return Sandbox.workspace_write


@dataclass(frozen=True)
class CodexSdkRunResult:
    """保存一次 SDK turn 的响应、thread 标识和真实工具完成事件。

    ``receipt_handle`` 只在调用方需要把 completed receipt 延迟到外层事实落盘
    后再确认时出现。adapter 不会凭这个字段自行删除 receipt；删除动作必须由
    调用方在对应的持久化门槛之后显式执行。
    """

    response: HarnessResponse
    thread_id: str
    turn_id: str = ""
    replayed: bool = False
    receipt_turn_count_before: int | None = None
    request_digest: str = ""
    thread_policy_digest: str = ""
    # 这些 item 来自 app-server 的 item/completed 事件，不来自模型最终 JSON。
    # 仅供上层校验 submit 工具回执；不会写入 prompt 或科学结果。
    completed_tool_items: tuple[Any, ...] = ()
    receipt_handle: "CodexCompletedTurnReceiptHandle | CodexGoalReceiptHandle | None" = None
    # Investigation Goal 的逻辑身份与全部连续物理 turn；普通 turn 保持默认空值。
    logical_goal_id: str = ""
    goal_id: str = ""
    goal_status: str = ""
    physical_turn_ids: tuple[str, ...] = ()
    physical_turn_start: str = ""
    physical_turn_end: str = ""


_RECEIPT_HANDOFF_KINDS = frozenset({"result", "waiting"})
# receipt handle 只能由下方的 SessionStore 绑定工厂创建。把令牌保留在模块内，
# 可以继续提供公开的受控工厂调用，同时阻止调用方仅凭一个目录手工拼出句柄。
_RECEIPT_HANDLE_FACTORY_TOKEN = object()


def _normalize_receipt_owner(value: Any) -> dict[str, str]:
    """规范化 receipt handle 的调用方归属身份。"""

    if isinstance(value, CodexSessionOwner):
        value = value.identity
    elif isinstance(value, Mapping) and (
        "task_id" in value or "batch_id" in value or "combination_id" in value
    ):
        value = SchemaV1CodexSessionOwnerReader.read(value).identity
    if not isinstance(value, Mapping) or not value:
        raise HarnessError("completed turn receipt owner is invalid", kind="session_state")
    owner: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key or key != key.strip():
            raise HarnessError("completed turn receipt owner field is invalid", kind="session_state")
        if not isinstance(item, str) or _RECEIPT_IDENTIFIER.fullmatch(item) is None:
            raise HarnessError(
                "completed turn receipt owner value is malformed",
                kind="session_state",
            )
        owner[key] = item
    kind = owner.get("kind")
    required_fields = {
        "investigation": {
            "kind",
            "research_task_id",
            "investigation_context_id",
            "hypothesis_execution_input_record_id",
        },
        "task_review": {"kind", "research_task_id"},
    }.get(kind)
    if required_fields is None or set(owner) != required_fields:
        raise HarnessError(
            "completed turn receipt owner fields are invalid",
            kind="session_state",
        )
    return owner


@dataclass(frozen=True, slots=True, init=False)
class CodexCompletedTurnReceiptHandle:
    """把一次 completed receipt 绑定到调用方归属和外层持久化门槛。

    ``handoff_kind`` 为 ``result`` 时，调用方必须先完成 Evidence/Working Memory
    Result（及必要的 repair 完成）再调用 ``acknowledge_after_result_persisted``；
    为 ``waiting`` 时，调用方必须先原子保存等待作业 sidecar 并写入 resumable
    waiting 事实，再调用 ``acknowledge_after_waiting_persisted``。句柄自身只核对
    固定身份并执行一次原子 receipt acknowledgement，不读取或猜测外层事实。
    同一个句柄重复确认是幂等的；新的句柄在 receipt 缺失或身份不匹配时仍失败关闭。
    """

    receipt_root: Path | str
    owner: Mapping[str, Any] | CodexSessionOwner
    thread_id: str
    turn_id: str
    request_digest: str
    thread_policy_digest: str
    turn_count_before: int
    handoff_kind: str
    _acknowledged: bool = dataclasses.field(default=False, init=False, repr=False)
    _already_acknowledged: bool = dataclasses.field(
        default=False,
        init=False,
        repr=False,
    )
    _lock: threading.RLock = dataclasses.field(
        default_factory=threading.RLock,
        init=False,
        repr=False,
        compare=False,
    )

    def __init__(
        self,
        receipt_root: Path | str,
        owner: Mapping[str, Any] | CodexSessionOwner,
        thread_id: str,
        turn_id: str,
        request_digest: str,
        thread_policy_digest: str,
        turn_count_before: int,
        handoff_kind: str,
        *,
        _factory_token: object | None = None,
    ) -> None:
        """只接受本模块受控工厂传入的完整句柄身份。

        公开兼容入口是 ``build_codex_completed_turn_receipt_handle`` 或
        ``resolve_codex_completed_turn_receipt_handle``；它们会先绑定 canonical
        SessionStore、owner 和 receipt 根目录。直接调用构造函数即使传入一个现存
        目录也不能绕过这些核对。
        """

        if _factory_token is not _RECEIPT_HANDLE_FACTORY_TOKEN:
            raise HarnessError(
                "completed turn receipt handle must be created by its SessionStore factory",
                kind="session_state",
            )
        object.__setattr__(self, "receipt_root", receipt_root)
        object.__setattr__(self, "owner", owner)
        object.__setattr__(self, "thread_id", thread_id)
        object.__setattr__(self, "turn_id", turn_id)
        object.__setattr__(self, "request_digest", request_digest)
        object.__setattr__(self, "thread_policy_digest", thread_policy_digest)
        object.__setattr__(self, "turn_count_before", turn_count_before)
        object.__setattr__(self, "handoff_kind", handoff_kind)
        object.__setattr__(self, "_acknowledged", False)
        object.__setattr__(self, "_already_acknowledged", False)
        object.__setattr__(self, "_lock", threading.RLock())
        self.__post_init__()

    def __post_init__(self) -> None:
        """校验并冻结句柄身份；不删除或确认 receipt。"""

        try:
            raw_root = Path(self.receipt_root).expanduser()
        except (TypeError, ValueError) as exc:
            raise HarnessError(
                "completed turn receipt handle root is invalid",
                kind="session_state",
            ) from exc
        if raw_root.is_symlink() or not raw_root.is_dir():
            raise HarnessError(
                "completed turn receipt handle root is not a regular directory",
                kind="session_state",
            )
        object.__setattr__(self, "receipt_root", raw_root.resolve())
        object.__setattr__(
            self,
            "owner",
            MappingProxyType(_normalize_receipt_owner(self.owner)),
        )
        object.__setattr__(
            self,
            "thread_id",
            _validate_receipt_identifier(self.thread_id, "thread_id"),
        )
        object.__setattr__(
            self,
            "turn_id",
            _validate_receipt_identifier(self.turn_id, "turn_id"),
        )
        object.__setattr__(
            self,
            "request_digest",
            _validate_request_digest(self.request_digest),
        )
        object.__setattr__(
            self,
            "thread_policy_digest",
            _validate_thread_policy_digest(self.thread_policy_digest),
        )
        if (
            isinstance(self.turn_count_before, bool)
            or not isinstance(self.turn_count_before, int)
            or self.turn_count_before < 0
        ):
            raise HarnessError(
                "completed turn receipt handle turn_count_before is invalid",
                kind="session_state",
            )
        if self.handoff_kind not in _RECEIPT_HANDOFF_KINDS:
            raise HarnessError(
                "completed turn receipt handle handoff kind is invalid",
                kind="session_state",
            )

    @classmethod
    def from_already_acknowledged(
        cls,
        receipt_root: Path | str,
        *,
        owner: Mapping[str, Any] | CodexSessionOwner,
        thread_id: str,
        turn_id: str,
        request_digest: str,
        turn_count_before: int,
        thread_policy_digest: str,
        handoff_kind: str,
        _factory_token: object | None = None,
    ) -> "CodexCompletedTurnReceiptHandle":
        """构造跨进程已确认的 no-op 句柄，不读取或创建 receipt。

        调用方必须已经在外层核对同一条持久化 Result 和 sidecar digest。manifest
        没有保存每个 turn 的 turn_id，因此本方法不声称自己验证了 Result 的
        turn_id；它只把已由 resolver 核对的固定身份标记为跨进程已确认。
        """

        if _factory_token is not _RECEIPT_HANDLE_FACTORY_TOKEN:
            raise HarnessError(
                "already-acknowledged receipt handle must use its SessionStore factory",
                kind="session_state",
            )
        handle = cls(
            receipt_root=receipt_root,
            owner=owner,
            thread_id=thread_id,
            turn_id=turn_id,
            request_digest=request_digest,
            turn_count_before=turn_count_before,
            thread_policy_digest=thread_policy_digest,
            handoff_kind=handoff_kind,
            _factory_token=_RECEIPT_HANDLE_FACTORY_TOKEN,
        )
        object.__setattr__(handle, "_acknowledged", True)
        object.__setattr__(handle, "_already_acknowledged", True)
        return handle

    @property
    def turn_request_digest(self) -> str:
        """返回绑定本次 turn 的请求摘要。"""

        return self.request_digest

    @property
    def turn_count(self) -> int:
        """返回 receipt 记录的 turn 开始前成功计数。"""

        return self.turn_count_before

    @property
    def acknowledged(self) -> bool:
        """返回当前进程中的句柄是否已经完成确认。"""

        with self._lock:
            return self._acknowledged

    @property
    def already_acknowledged(self) -> bool:
        """返回当前句柄是否代表前一进程已经完成 receipt 确认。"""

        with self._lock:
            return self._already_acknowledged

    def as_mapping(self) -> dict[str, Any]:
        """返回不含 prompt/响应正文的固定句柄身份。"""

        return {
            "owner": dict(self.owner),
            "thread_id": self.thread_id,
            "turn_id": self.turn_id,
            "turn_request_digest": self.turn_request_digest,
            "thread_policy_digest": self.thread_policy_digest,
            "turn_count_before": self.turn_count_before,
            "handoff_kind": self.handoff_kind,
        }

    def _acknowledge(self, expected_kind: str | None) -> None:
        """在调用方声明对应外层事实已落盘后确认 receipt。"""

        if expected_kind is not None and self.handoff_kind != expected_kind:
            raise HarnessError(
                "completed turn receipt handle handoff kind does not match acknowledgement",
                kind="session_state",
            )
        with self._lock:
            if self._acknowledged:
                return
            if self.receipt_root.is_symlink() or not self.receipt_root.is_dir():
                raise HarnessError(
                    "completed turn receipt handle root is missing",
                    kind="session_state",
                )
            acknowledge_codex_completed_turn_receipt(
                self.receipt_root,
                thread_id=self.thread_id,
                turn_id=self.turn_id,
                request_digest=self.request_digest,
                turn_count_before=self.turn_count_before,
                thread_policy_digest=self.thread_policy_digest,
            )
            object.__setattr__(self, "_acknowledged", True)

    def acknowledge_after_result_persisted(self) -> None:
        """在 Evidence/Working Memory Result 已成功落盘后确认 receipt。"""

        self._acknowledge("result")

    def acknowledge_after_waiting_persisted(self) -> None:
        """在等待 sidecar 和 resumable waiting 事实已成功落盘后确认 receipt。"""

        self._acknowledge("waiting")

    def acknowledge(self) -> None:
        """兼容旧调用方，按句柄固定的 handoff 类型确认 receipt。"""

        self._acknowledge(self.handoff_kind)


@dataclass(frozen=True, slots=True, init=False)
class CodexGoalReceiptHandle:
    """绑定 Investigation Goal 完成收据，供外层事实落盘后确认。"""

    receipt_root: Path | str
    owner: Mapping[str, Any] | CodexSessionOwner
    thread_id: str
    turn_id: str
    logical_goal_id: str
    goal_id: str
    physical_turn_ids: tuple[str, ...]
    objective_digest: str
    physical_turn_count_before: int
    physical_turn_start: str
    physical_turn_end: str
    goal_status: str
    final_turn_status: str
    request_digest: str
    thread_policy_digest: str
    turn_count_before: int
    handoff_kind: str
    _acknowledged: bool = dataclasses.field(default=False, init=False, repr=False)
    _lock: threading.RLock = dataclasses.field(
        default_factory=threading.RLock,
        init=False,
        repr=False,
        compare=False,
    )

    def __init__(
        self,
        receipt_root: Path | str,
        owner: Mapping[str, Any] | CodexSessionOwner,
        thread_id: str,
        turn_id: str,
        logical_goal_id: str,
        goal_id: str,
        physical_turn_ids: Sequence[str],
        objective_digest: str,
        physical_turn_count_before: int,
        physical_turn_start: str,
        physical_turn_end: str,
        goal_status: str,
        final_turn_status: str,
        request_digest: str,
        thread_policy_digest: str,
        turn_count_before: int,
        handoff_kind: str,
        *,
        _factory_token: object | None = None,
    ) -> None:
        """只接受受控工厂生成的 Goal 收据句柄。"""

        if _factory_token is not _RECEIPT_HANDLE_FACTORY_TOKEN:
            raise HarnessError(
                "Goal receipt handle must be created by its SessionStore factory",
                kind="session_state",
            )
        object.__setattr__(self, "receipt_root", Path(receipt_root).resolve())
        object.__setattr__(self, "owner", MappingProxyType(_normalize_receipt_owner(owner)))
        object.__setattr__(self, "thread_id", _validate_receipt_identifier(thread_id, "thread_id"))
        object.__setattr__(self, "turn_id", _validate_receipt_identifier(turn_id, "turn_id"))
        object.__setattr__(self, "logical_goal_id", _validate_receipt_identifier(logical_goal_id, "logical_goal_id"))
        object.__setattr__(self, "goal_id", _validate_receipt_identifier(goal_id, "goal_id"))
        normalized_ids = tuple(_validate_receipt_identifier(item, "physical_turn_id") for item in physical_turn_ids)
        if not normalized_ids or len(set(normalized_ids)) != len(normalized_ids) or normalized_ids[0] != self.turn_id:
            raise HarnessError("Goal receipt handle physical range is invalid", kind="session_state")
        object.__setattr__(self, "physical_turn_ids", normalized_ids)
        object.__setattr__(self, "objective_digest", _validate_request_digest(objective_digest))
        if (
            isinstance(physical_turn_count_before, bool)
            or not isinstance(physical_turn_count_before, int)
            or physical_turn_count_before < 0
        ):
            raise HarnessError("Goal receipt handle physical baseline is invalid", kind="session_state")
        object.__setattr__(self, "physical_turn_count_before", physical_turn_count_before)
        if (
            _validate_receipt_identifier(physical_turn_start, "physical_turn_start") != normalized_ids[0]
            or _validate_receipt_identifier(physical_turn_end, "physical_turn_end") != normalized_ids[-1]
        ):
            raise HarnessError("Goal receipt handle physical endpoints are invalid", kind="session_state")
        object.__setattr__(self, "physical_turn_start", normalized_ids[0])
        object.__setattr__(self, "physical_turn_end", normalized_ids[-1])
        object.__setattr__(self, "goal_status", _validate_goal_status(goal_status))
        object.__setattr__(self, "final_turn_status", _validate_goal_turn_status(final_turn_status))
        if self.goal_status != "complete" or self.final_turn_status != "completed":
            raise HarnessError("Goal receipt handle requires complete successful final turn", kind="session_protocol")
        object.__setattr__(self, "request_digest", _validate_request_digest(request_digest))
        object.__setattr__(self, "thread_policy_digest", _validate_thread_policy_digest(thread_policy_digest))
        if isinstance(turn_count_before, bool) or not isinstance(turn_count_before, int) or turn_count_before < 0:
            raise HarnessError("Goal receipt handle turn count is invalid", kind="session_state")
        object.__setattr__(self, "turn_count_before", turn_count_before)
        if handoff_kind not in _RECEIPT_HANDOFF_KINDS:
            raise HarnessError("Goal receipt handle handoff kind is invalid", kind="session_state")
        object.__setattr__(self, "handoff_kind", handoff_kind)
        object.__setattr__(self, "_acknowledged", False)
        object.__setattr__(self, "_lock", threading.RLock())

    @property
    def turn_request_digest(self) -> str:
        """返回当前 Goal 请求摘要，兼容 Working Memory receipt 字段。"""

        return self.request_digest

    @property
    def turn_count(self) -> int:
        """返回 Goal 开始前的逻辑成功计数。"""

        return self.turn_count_before

    @property
    def acknowledged(self) -> bool:
        """返回当前进程是否已经确认 Goal 收据。"""

        with self._lock:
            return self._acknowledged

    @property
    def terminal_turn_id(self) -> str:
        """返回 Goal 范围内最后一个物理 turn 身份。"""

        return self.physical_turn_ids[-1]

    def as_mapping(self) -> dict[str, Any]:
        """返回固定 Goal 身份，不包含模型正文。"""

        return {
            "owner": dict(self.owner),
            "thread_id": self.thread_id,
            "turn_id": self.turn_id,
            "turn_request_digest": self.turn_request_digest,
            "request_digest": self.request_digest,
            "thread_policy_digest": self.thread_policy_digest,
            "turn_count_before": self.turn_count_before,
            "handoff_kind": self.handoff_kind,
            "logical_goal_id": self.logical_goal_id,
            "goal_id": self.goal_id,
            "physical_turn_ids": list(self.physical_turn_ids),
            "objective_digest": self.objective_digest,
            "physical_turn_count_before": self.physical_turn_count_before,
            "physical_turn_start": self.physical_turn_start,
            "physical_turn_end": self.physical_turn_end,
            "goal_status": self.goal_status,
            "final_turn_status": self.final_turn_status,
            "terminal_turn_id": self.terminal_turn_id,
        }

    def _acknowledge(self, expected_kind: str | None) -> None:
        """在外层结果或 waiting 事实落盘后删除 Goal 附加记录。"""

        if expected_kind is not None and self.handoff_kind != expected_kind:
            raise HarnessError("Goal receipt handoff kind does not match acknowledgement", kind="session_state")
        with self._lock:
            if self._acknowledged:
                return
            # 确认阶段只能读取既有收据；收据缺失时不得创建目录、修正权限或猜测
            # 已经发生的外层事实。
            receipt = _read_goal_receipt_read_only(self.receipt_root)
            if receipt is None:
                raise HarnessError("Goal receipt is missing for acknowledgement", kind="session_state")
            expected = {
                "thread_id": self.thread_id,
                "logical_goal_id": self.logical_goal_id,
                "goal_id": self.goal_id,
                "objective_digest": self.objective_digest,
                "request_digest": self.request_digest,
                "thread_policy_digest": self.thread_policy_digest,
                "turn_count_before": self.turn_count_before,
                "physical_turn_count_before": self.physical_turn_count_before,
                "physical_turn_ids": list(self.physical_turn_ids),
                "physical_turn_start": self.physical_turn_start,
                "physical_turn_end": self.physical_turn_end,
                "goal_status": self.goal_status,
                "final_turn_status": self.final_turn_status,
            }
            if any(receipt.get(field) != value for field, value in expected.items()):
                raise HarnessError("Goal receipt identity changed before acknowledgement", kind="session_state")
            _remove_goal_turn_journal(self.receipt_root)
            _remove_goal_intent(self.receipt_root)
            _remove_goal_receipt(self.receipt_root)
            object.__setattr__(self, "_acknowledged", True)

    def acknowledge_after_result_persisted(self) -> None:
        """在科学结果事实成功落盘后确认 Goal 收据。"""

        self._acknowledge("result")

    def acknowledge_after_waiting_persisted(self) -> None:
        """在 waiting 事实成功落盘后确认 Goal 收据。"""

        self._acknowledge("waiting")

    def acknowledge(self) -> None:
        """按句柄固定的 handoff 类型确认 Goal 收据。"""

        self._acknowledge(self.handoff_kind)


def _validate_goal_receipt_store_binding(
    root: Path | str,
    *,
    owner: Mapping[str, Any] | CodexSessionOwner,
    session_store: CodexSessionStore,
) -> dict[str, str]:
    """核对 Goal 收据根目录属于给定 SessionStore 和归属身份。"""

    normalized_owner = _normalize_receipt_owner(owner)
    if not isinstance(session_store, CodexSessionStore):
        raise HarnessError(
            "Goal receipt session store is invalid",
            kind="session_state",
        )
    if normalized_owner != session_store.owner.identity:
        raise HarnessError(
            "Goal receipt owner does not match the session store",
            kind="session_state",
        )
    try:
        raw_root = Path(root).expanduser()
        if raw_root.is_symlink():
            raise HarnessError(
                "Goal receipt root must not be a symlink",
                kind="session_state",
            )
        actual_root = raw_root.resolve()
    except (TypeError, ValueError) as exc:
        raise HarnessError("Goal receipt root is invalid", kind="session_state") from exc
    canonical_root = (
        session_store.manifest_path.parent / "completed_turn_receipts"
    ).resolve()
    if actual_root != canonical_root:
        raise HarnessError(
            "Goal receipt root is not canonical for the session",
            kind="session_state",
        )
    return normalized_owner


def build_codex_goal_receipt_handle(
    root: Path | str,
    *,
    owner: Mapping[str, Any] | CodexSessionOwner,
    session_store: CodexSessionStore,
    thread_id: str,
    turn_id: str,
    request_digest: str,
    objective_digest: str | None = None,
    thread_policy_digest: str,
    turn_count_before: int,
    handoff_kind: str,
) -> CodexGoalReceiptHandle:
    """从当前 SessionStore 的规范目录读取 Goal 收据并构造延迟确认句柄。"""

    normalized_owner = _validate_goal_receipt_store_binding(
        root,
        owner=owner,
        session_store=session_store,
    )
    receipt = _read_goal_receipt(root)
    if receipt is None:
        raise HarnessError("Goal receipt is missing", kind="session_state")
    return _build_codex_goal_receipt_handle_from_record(
        root,
        owner=normalized_owner,
        thread_id=thread_id,
        turn_id=turn_id,
        request_digest=request_digest,
        objective_digest=objective_digest,
        thread_policy_digest=thread_policy_digest,
        turn_count_before=turn_count_before,
        handoff_kind=handoff_kind,
        receipt=receipt,
    )


def _build_codex_goal_receipt_handle_from_record(
    root: Path | str,
    *,
    owner: Mapping[str, Any] | CodexSessionOwner,
    thread_id: str,
    turn_id: str,
    request_digest: str,
    objective_digest: str | None,
    thread_policy_digest: str,
    turn_count_before: int,
    handoff_kind: str,
    receipt: Mapping[str, Any],
) -> CodexGoalReceiptHandle:
    """用已读且已验证的 Goal 收据构造句柄，不再次访问文件系统。"""

    if (
        receipt["thread_id"] != thread_id
        or receipt["physical_turn_ids"][0] != turn_id
        or receipt["request_digest"] != request_digest
        or receipt["thread_policy_digest"] != thread_policy_digest
        or receipt["turn_count_before"] != turn_count_before
        or (
            objective_digest is not None
            and receipt["objective_digest"] != _validate_request_digest(objective_digest)
        )
    ):
        raise HarnessError("Goal receipt does not match returned operation", kind="session_state")
    return CodexGoalReceiptHandle(
        root,
        owner,
        thread_id,
        turn_id,
        receipt["logical_goal_id"],
        receipt["goal_id"],
        receipt["physical_turn_ids"],
        receipt["objective_digest"],
        receipt["physical_turn_count_before"],
        receipt["physical_turn_start"],
        receipt["physical_turn_end"],
        receipt["goal_status"],
        receipt["final_turn_status"],
        receipt["request_digest"],
        receipt["thread_policy_digest"],
        receipt["turn_count_before"],
        handoff_kind,
        _factory_token=_RECEIPT_HANDLE_FACTORY_TOKEN,
    )


def resolve_codex_goal_receipt_handle(
    root: Path | str,
    *,
    owner: Mapping[str, Any] | CodexSessionOwner,
    session_store: CodexSessionStore,
    thread_id: str,
    turn_id: str,
    request_digest: str | None,
    objective_digest: str | None = None,
    turn_count_before: int | None,
    thread_policy_digest: str | None,
    handoff_kind: str,
) -> CodexGoalReceiptHandle | None:
    """只读解析当前 SessionStore 的 Goal 收据，不创建目录或启动 provider。"""

    normalized_owner = _validate_goal_receipt_store_binding(
        root,
        owner=owner,
        session_store=session_store,
    )
    receipt = _read_goal_receipt_read_only(root)
    if receipt is None:
        return None
    if request_digest is not None and receipt["request_digest"] != request_digest:
        raise HarnessError("Goal receipt request digest does not match", kind="session_state")
    if objective_digest is not None and receipt["objective_digest"] != _validate_request_digest(objective_digest):
        raise HarnessError("Goal receipt objective digest does not match", kind="session_state")
    if thread_policy_digest is not None and receipt["thread_policy_digest"] != thread_policy_digest:
        raise HarnessError("Goal receipt thread policy digest does not match", kind="session_state")
    if turn_count_before is not None and receipt["turn_count_before"] != turn_count_before:
        raise HarnessError("Goal receipt turn count does not match", kind="session_state")
    return _build_codex_goal_receipt_handle_from_record(
        root,
        owner=normalized_owner,
        thread_id=thread_id,
        turn_id=turn_id,
        request_digest=receipt["request_digest"],
        objective_digest=receipt["objective_digest"],
        thread_policy_digest=receipt["thread_policy_digest"],
        turn_count_before=receipt["turn_count_before"],
        handoff_kind=handoff_kind,
        receipt=receipt,
    )


@dataclass(frozen=True)
class _CodexThreadStartup:
    """保存已核对的 thread、profile 和非秘密策略摘要。"""

    thread: Any
    profile: str
    thread_policy_digest: str
    trace_values: tuple[str, ...]
    approval_mode: ApprovalMode


@dataclass(frozen=True)
class _CodexReplayState:
    """保存 request-intent、pending journal 和可安全重放的 turn。"""

    request_intent: dict[str, Any] | None
    pending_journal: dict[str, Any] | None
    replay_receipt: dict[str, Any] | None
    replay_content: str | None
    replay_completed_tool_items: tuple[Any, ...] = ()


class CodexRecoveryPending(HarnessError):
    """需要先完成持久恢复动作，禁止同会话盲目再次提交模型 turn。"""

    def __init__(self, message: str, *, turn_id: str | None, reason: str) -> None:
        super().__init__(message, kind="recovery_pending")
        self.turn_id = turn_id
        self.reason = reason


class CodexWorkerRecoveryPending(CodexRecoveryPending):
    """SDK worker 在中断后的有限等待内仍未停止，禁止同会话继续发 turn。"""


class CodexCompletionRecoveryPending(CodexRecoveryPending):
    """completed turn 的 receipt 未确认落盘，必须严格恢复后才能继续。"""


class CodexRequestIntentRecoveryPending(CodexRecoveryPending):
    """worker 请求意图没有对应的已完成 turn，必须显式确认后才能继续。

    ``request_digest`` 是磁盘中原 request-intent 的摘要，而不是本次恢复请求的
    摘要。runtime 必须用它保存恢复身份；字段缺失时只能失败关闭，不能拿当前
    摘要代替。``terminal_turn_completed`` 只在 provider 已明确结束 turn、但
    intent 删除失败时为真，供 worker 清理跳过无意义的 interrupt。
    """

    def __init__(
        self,
        message: str,
        *,
        turn_id: str | None,
        reason: str,
        request_digest: str | None = None,
        terminal_turn_completed: bool = False,
    ) -> None:
        super().__init__(message, turn_id=turn_id, reason=reason)
        self.request_digest = request_digest
        self.terminal_turn_completed = terminal_turn_completed


class CodexGoalRecoveryPending(CodexRecoveryPending):
    """Goal 进入可恢复非完成终态或身份不完整，禁止再次启动目标。"""

    def __init__(
        self,
        message: str,
        *,
        goal_status: str | None,
        logical_goal_id: str | None,
        physical_turn_ids: Sequence[str] = (),
        request_digest: str | None = None,
    ) -> None:
        super().__init__(
            message,
            turn_id=(physical_turn_ids[-1] if physical_turn_ids else None),
            reason="goal",
        )
        self.goal_status = goal_status
        self.logical_goal_id = logical_goal_id
        self.physical_turn_ids = tuple(physical_turn_ids)
        self.request_digest = request_digest


class CodexTerminalTurnRecoveryPending(CodexRecoveryPending):
    """明确终态尚未得到上层确认，禁止同会话提交新 turn。"""

    def __init__(
        self,
        message: str,
        *,
        turn_id: str,
        status: str,
        request_digest: str,
    ) -> None:
        if status not in {"failed", "interrupted"}:
            raise ValueError("Codex terminal turn status must be failed or interrupted")
        super().__init__(message, turn_id=turn_id, reason="terminal_turn")
        self.status = status
        self.request_digest = request_digest
        self.terminal_turn_completed = True


class CodexTerminalTurnError(HarnessError):
    """provider 已明确结束 failed/interrupted turn，并携带有限错误代码。"""

    def __init__(
        self,
        *,
        turn_id: str | None,
        status: str,
        error_code: str | None = None,
    ) -> None:
        if status not in {"failed", "interrupted"}:
            raise ValueError("Codex terminal turn status must be failed or interrupted")
        if error_code is not None and error_code not in CODEX_TERMINAL_PROVIDER_ERROR_CODES:
            raise HarnessError(
                "Codex terminal provider error code is not approved",
                kind="transport",
            )
        super().__init__(
            f"Codex turn ended with status={status}",
            kind="transport",
        )
        self.turn_id = turn_id
        self.status = status
        self.error_code = error_code
        self.provider_error_code = error_code
        self.terminal_turn_completed = True


def write_codex_prompt_artifact(request: HarnessRequest, trace_path: Path) -> Path:
    """把不含密钥的请求摘要写入 trace 目录并返回路径。"""

    request.workspace.mkdir(parents=True, exist_ok=True)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(request.workspace, 0o700)
    os.chmod(trace_path.parent, 0o700)
    prompt_path = trace_path.with_suffix(".prompt.json")
    prompt_path.write_text(
        json.dumps(
            {
                "role": request.role,
                "stage": request.stage,
                "messages": list(request.messages),
                "response_schema": request.response_schema,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    os.chmod(prompt_path, 0o600)
    return prompt_path


def build_codex_config(
    *,
    codex_bin: str,
    workspace: Path,
    codex_home: Path,
    sandbox_spec: CodexProcessSandboxSpec | None = None,
    spec: CodexProcessSandboxSpec | None = None,
    environment_keys: Sequence[str] = (),
    secret_environment: Mapping[str, str] | None = None,
) -> CodexConfig:
    """构造 SDK 配置；显式 spec 时把 app-server 放进 bwrap 外层进程沙箱。

    未提供 ``sandbox_spec`` 或兼容别名 ``spec`` 时，返回值保持旧行为：
    SDK 直接启动 vendored wrapper，只有 ``CODEX_HOME`` 进入每进程环境。
    提供 spec 后会严格核对它绑定的 workspace、CODEX_HOME 和 wrapper 与本次
    参数一致，再使用 ``launch_args_override`` 启动 ``bwrap`` 内的实际 native
    ``codex app-server --listen stdio://``。沙箱准备或能力核验失败时直接
    抛错，不会偷偷降级到旧启动方式。``secret_environment`` 只注入 app-server
    进程环境；它的变量名会加入 bwrap 白名单，但值不会出现在 argv、thread
    config、trace 或 CODEX_HOME 文件中。
    """

    if sandbox_spec is not None and spec is not None:
        raise HarnessError(
            "Codex process sandbox spec was provided twice",
            kind="provider_envelope",
        )
    effective_sandbox_spec = sandbox_spec if sandbox_spec is not None else spec

    normalized_secret_environment = _validate_secret_environment(secret_environment)
    normalized_environment_keys = _validate_environment_names(environment_keys)
    for name in normalized_secret_environment:
        if name not in normalized_environment_keys:
            normalized_environment_keys.append(name)

    binary = Path(codex_bin).expanduser().resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise HarnessError(
            f"Codex vendored runtime is not executable: {binary}",
            kind="transport",
        )
    workspace = workspace.resolve()
    codex_home = codex_home.expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    codex_home.mkdir(parents=True, exist_ok=True)
    os.chmod(workspace, 0o700)
    os.chmod(codex_home, 0o700)
    if effective_sandbox_spec is not None:
        prepared_spec = prepare_codex_process_sandbox(effective_sandbox_spec)
        if prepared_spec.workspace != workspace or prepared_spec.codex_home != codex_home:
            raise HarnessError(
                "Codex process sandbox identity does not match SDK workspace",
                kind="provider_envelope",
            )
        if prepared_spec.codex_bin != binary:
            raise HarnessError(
                "Codex process sandbox wrapper does not match SDK codex_bin",
                kind="provider_envelope",
            )
        return CodexConfig(
            codex_bin=str(binary),
            launch_args_override=build_codex_process_sandbox_argv(
                prepared_spec,
                environment_names=normalized_environment_keys,
            ),
            cwd=str(workspace),
            # SDK 只合并这一份 mapping；父进程的 os.environ 不被修改，
            # bwrap 负责为 app-server 提供绑定后的路径和标准工具环境。
            env={
                **build_codex_process_sandbox_environment(prepared_spec),
                **normalized_secret_environment,
            },
        )
    return CodexConfig(
        codex_bin=str(binary),
        cwd=str(workspace),
        # SDK 会把这个映射合并到父进程环境；API key 仍只从环境变量读取。
        env={"CODEX_HOME": str(codex_home), **normalized_secret_environment},
    )


def config_values_to_object(values: Sequence[str]) -> dict[str, Any]:
    """把旧 CLI 的 ``--config key=value`` 列表转成 thread config 对象。

    Runner 仍集中生成既有 provider/MCP 配置键，适配层只负责把同一组键转为
    SDK 的 ``thread_start(config=...)`` 和 ``thread_resume(config=...)`` 参数。
    """

    if len(values) % 2:
        raise HarnessError("Codex config override list is malformed", kind="provider_envelope")
    result: dict[str, Any] = {}
    for index in range(0, len(values), 2):
        if values[index] != "--config":
            raise HarnessError("Codex config override list is malformed", kind="provider_envelope")
        assignment = values[index + 1]
        if "=" not in assignment:
            raise HarnessError("Codex config override is missing '='", kind="provider_envelope")
        key, raw_value = assignment.split("=", 1)
        key = key.strip()
        if not key or any(part == "" for part in key.split(".")):
            raise HarnessError("Codex config override has an invalid key", kind="provider_envelope")
        try:
            value: Any = json.loads(raw_value)
        except json.JSONDecodeError:
            # Codex 的 CLI 会把无法解析为 TOML 的值当作字符串；保持相同语义。
            value = raw_value
        cursor = result
        parts = key.split(".")
        for part in parts[:-1]:
            existing = cursor.get(part)
            if existing is None:
                nested: dict[str, Any] = {}
                cursor[part] = nested
                cursor = nested
            elif isinstance(existing, dict):
                cursor = existing
            else:
                raise HarnessError(
                    f"Codex config override conflicts at {part}",
                    kind="provider_envelope",
                )
        cursor[parts[-1]] = value
    return result


def _json_safe(value: Any) -> Any:
    """把 SDK/Pydantic 事件转换成可写入 JSONL 的值。"""

    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "model_dump"):
        return value.model_dump(by_alias=True, mode="json", exclude_none=False)
    if dataclasses.is_dataclass(value):
        return _json_safe(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _collect_trace_sensitive_values(value: Any, *, in_headers: bool = False) -> tuple[str, ...]:
    """收集 thread config 中 HTTP header 的值，供 trace 写入前统一脱敏。"""

    values: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_is_headers = str(key).lower() in {"http_headers", "headers"}
            values.extend(
                _collect_trace_sensitive_values(
                    item,
                    in_headers=in_headers or key_is_headers,
                )
            )
        return tuple(value for value in values if value)
    if isinstance(value, (list, tuple)):
        for item in value:
            values.extend(_collect_trace_sensitive_values(item, in_headers=in_headers))
        return tuple(value for value in values if value)
    if in_headers and isinstance(value, str) and value:
        values.append(value)
    return tuple(values)


def _redact_trace_value(value: Any, sensitive_values: Sequence[str]) -> Any:
    """递归替换 trace 事件中的静态 header 值，不改变内存中的 SDK 事件。"""

    if isinstance(value, str):
        redacted = value
        for sensitive in sensitive_values:
            if sensitive:
                redacted = redacted.replace(sensitive, "[REDACTED]")
        return redacted
    if isinstance(value, Mapping):
        return {
            str(key): _redact_trace_value(item, sensitive_values)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_trace_value(item, sensitive_values) for item in value]
    if isinstance(value, tuple):
        return [_redact_trace_value(item, sensitive_values) for item in value]
    return value


def _write_notification(
    trace_handle: Any,
    event: Any,
    *,
    sensitive_values: Sequence[str] = (),
) -> None:
    """把脱敏后的 SDK notification 写入 0600 JSONL trace。"""

    params = _redact_trace_value(
        _json_safe(getattr(event, "payload", None)),
        sensitive_values,
    )
    trace_handle.write(
        json.dumps(
            {
                "type": "codex_notification",
                "method": str(getattr(event, "method", "")),
                "params": params,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
    )
    trace_handle.flush()


def _write_error_notification(trace_handle: Any, payload: Any) -> None:
    """把 provider error 通知缩减为固定字段后写入 trace。

    输入可以是 SDK 模型或协议字典；动作只读取严格布尔型 ``will_retry``，不读取
    error 正文、thread/turn/request 标识或未知扩展字段。返回值为空，成功时刷新
    trace；写入失败原样抛错，由 turn worker 失败关闭。
    """

    params: dict[str, bool] = {}
    will_retry = _read_protocol_field(payload, "will_retry")
    if isinstance(will_retry, bool):
        params["will_retry"] = will_retry
    trace_handle.write(
        json.dumps(
            {
                "type": "codex_notification",
                "method": "error",
                "params": params,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
    )
    trace_handle.flush()


def _write_terminal_turn_notification(
    trace_handle: Any,
    *,
    thread_id: str,
    turn_id: str,
    status: str,
    error_code: str | None = None,
) -> None:
    """把明确终态按固定白名单写入 trace 并刷盘。

    输入只接受已知 thread/turn 标识和 failed/interrupted 状态；动作写入既有
    notification JSONL 外壳，但不读取 provider error、message、request id 或其它
    payload。成功后同时刷盘文件；字段非法或写盘失败时抛错，调用方必须保留终态
    journal 与 request-intent，不能执行 acknowledgement。
    """

    normalized_thread_id = _validate_receipt_identifier(thread_id, "thread_id")
    normalized_turn_id = _validate_receipt_identifier(turn_id, "turn_id")
    if status not in {"failed", "interrupted"}:
        raise HarnessError("Codex terminal turn status is invalid", kind="session_protocol")
    if error_code is not None and error_code not in CODEX_TERMINAL_PROVIDER_ERROR_CODES:
        raise HarnessError(
            "Codex terminal provider error code is not approved",
            kind="session_protocol",
        )
    params: dict[str, str] = {
        "thread_id": normalized_thread_id,
        "turn_id": normalized_turn_id,
        "status": status,
    }
    if error_code is not None:
        params["error_code"] = error_code
    trace_handle.write(
        json.dumps(
            {
                "type": "codex_notification",
                "method": "turn/completed",
                "params": params,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
    )
    trace_handle.flush()
    os.fsync(trace_handle.fileno())


def _item_text(item: Any) -> tuple[str | None, str | None]:
    root = getattr(item, "root", item)
    text = getattr(root, "text", None)
    phase = getattr(root, "phase", None)
    if not isinstance(text, str):
        return None, None
    phase_value = getattr(phase, "value", phase)
    return text, phase_value if isinstance(phase_value, str) else None


def _extract_final_content(items: Sequence[Any]) -> str:
    """从当前 turn 的协议 item 中提取最终 assistant 正文。"""

    fallback: str | None = None
    for item in reversed(items):
        text, phase = _item_text(item)
        if not text:
            continue
        if phase == "final_answer":
            return text
        if phase is None and fallback is None:
            fallback = text
    if fallback:
        return fallback
    raise HarnessError("Codex SDK returned no final assistant content", kind="empty_content")


def _usage_mapping(token_usage: Any) -> dict[str, int]:
    """将 Codex token breakdown 映射为 ARCHE 旧响应需要的 usage 字段。"""

    if token_usage is None:
        return {}
    last = getattr(token_usage, "last", None)
    if last is None:
        return {}
    values: dict[str, int] = {}
    for source, aliases in (
        ("input_tokens", ("input_tokens", "prompt_tokens")),
        ("output_tokens", ("output_tokens", "completion_tokens")),
        ("total_tokens", ("total_tokens",)),
        ("cached_input_tokens", ("cached_input_tokens",)),
        ("reasoning_output_tokens", ("reasoning_output_tokens",)),
        ("cache_write_input_tokens", ("cache_write_input_tokens",)),
    ):
        value = getattr(last, source, None)
        if isinstance(value, int) and not isinstance(value, bool):
            for alias in aliases:
                values[alias] = value
    return values


def _read_terminal_turn_failure_status(payload: Any) -> str | None:
    """读取 failed/interrupted 终态；其它 payload 返回 ``None``。"""

    turn = _read_protocol_field(payload, "turn")
    status = _protocol_enum_value(_read_protocol_field(turn, "status"))
    if status not in {"failed", "interrupted"}:
        return None
    return str(status)


def _read_safe_terminal_provider_error_code(payload: Any) -> str | None:
    """从终态错误的固定 JSON 结构读取唯一允许的 provider code。"""

    turn = _read_protocol_field(payload, "turn")
    error = _read_protocol_field(turn, "error")
    direct_code = _protocol_enum_value(_read_protocol_field(error, "code"))
    if direct_code in CODEX_TERMINAL_PROVIDER_ERROR_CODES:
        return str(direct_code)
    message = _read_protocol_field(error, "message")
    if not isinstance(message, str):
        return None
    try:
        decoded = json.loads(message)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(decoded, Mapping):
        return None
    nested_error = decoded.get("error")
    if not isinstance(nested_error, Mapping):
        return None
    code = nested_error.get("code")
    if code in CODEX_TERMINAL_PROVIDER_ERROR_CODES and isinstance(code, str):
        return code
    return None


def _read_terminal_turn_failure_details(payload: Any) -> tuple[str, str | None] | None:
    """返回终态状态和已过滤的 provider code，不保留 provider 原文。"""

    status = _read_terminal_turn_failure_status(payload)
    if status is None:
        return None
    return status, _read_safe_terminal_provider_error_code(payload)


def _value_contains_sensitive_text(value: Any, sensitive_values: Sequence[str]) -> bool:
    """递归检查异常对象或其参数，避免密钥藏在异常链中。"""

    if isinstance(value, str):
        return any(secret and secret in value for secret in sensitive_values)
    if isinstance(value, Mapping):
        return any(
            _value_contains_sensitive_text(key, sensitive_values)
            or _value_contains_sensitive_text(item, sensitive_values)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_value_contains_sensitive_text(item, sensitive_values) for item in value)
    return False


def _exception_chain_contains_sensitive_text(
    error: BaseException,
    sensitive_values: Sequence[str],
) -> bool:
    """递归扫描 cause/context 和 ExceptionGroup，不能让错误链回显 token。"""

    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if _value_contains_sensitive_text(current.args, sensitive_values):
            return True
        nested = getattr(current, "exceptions", None)
        if isinstance(nested, (list, tuple)) and any(
            isinstance(item, BaseException)
            and _exception_chain_contains_sensitive_text(item, sensitive_values)
            for item in nested
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def _find_safe_transport_diagnostic(
    error: BaseException,
) -> dict[str, object] | None:
    """从异常链提取 SDK 已验证的固定 transport 诊断字段。

    只接受 vendored SDK 的 ``TransportDiagnostic`` 类型；异常正文、stderr、
    argv、环境和 provider 响应都不会进入返回值。这样即使启动异常被包装过，
    上层也只能看到阶段枚举、布尔 EOF、退出码/信号和白名单 indicator。找到时
    返回新的安全字段字典，未找到时返回 ``None``；异常链循环或未知异常类型不会
    导致失败，本函数不修改异常对象。
    """

    pending: list[BaseException] = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        candidate = getattr(current, "diagnostic", None)
        if isinstance(current, TransportClosedError) and isinstance(
            candidate, TransportDiagnostic
        ):
            return candidate.as_dict()
        nested = getattr(current, "exceptions", None)
        if isinstance(nested, (list, tuple)):
            pending.extend(
                item for item in nested if isinstance(item, BaseException)
            )
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        elif current.__context__ is not None:
            pending.append(current.__context__)
    return None


def _format_safe_transport_diagnostic(diagnostic: Mapping[str, object]) -> str:
    """把固定 transport 字段格式化为不含自由文本的错误后缀。

    输入必须包含 SDK 已校验的五个标量字段；动作只执行固定模板替换，无外部
    副作用。成功返回单行字符串，字段缺失时由 ``str.format`` 抛出 ``KeyError``，
    调用方不得把任意 provider 数据传入本函数。
    """

    return (
        "phase={phase} stdout_eof={stdout_eof} returncode={returncode} "
        "termination_signal={termination_signal} stderr_indicator={stderr_indicator}"
    ).format(**diagnostic)


def _sanitize_sdk_error(
    prefix: str,
    error: BaseException,
    sensitive_values: Sequence[str],
    *,
    kind: str,
) -> BaseException:
    """把 SDK 异常转换成不携带敏感正文的 HarnessError。"""

    diagnostic = _find_safe_transport_diagnostic(error)
    if _exception_chain_contains_sensitive_text(error, sensitive_values):
        sanitized = HarnessError(f"{prefix}: redacted sensitive error", kind=kind)
    else:
        # SDK 异常可能携带 provider 原始响应、header 或 traceback。即使当前 args
        # 没有命中已知 secret，也不把原始 cause/context 交给上层或持久化日志。
        suffix = (
            f" ({_format_safe_transport_diagnostic(diagnostic)})"
            if diagnostic is not None
            else ""
        )
        sanitized = HarnessError(
            f"{prefix}: {type(error).__name__}{suffix}",
            kind=kind,
        )
    if diagnostic is not None:
        # HarnessError 目前保持最小兼容接口；动态属性只包含上述固定标量，
        # 供恢复诊断或测试读取，不会把 SDK 异常对象本身暴露给调用方。
        sanitized.transport_diagnostic = diagnostic  # type: ignore[attr-defined]
    return sanitized


# 请求摘要已升级为完整 HarnessRequest 规范向量；旧摘要不能被当作同一请求重放。
# receipt/pending 还绑定 effective thread policy；旧 schema 没有该绑定，必须拒绝恢复。
_RECEIPT_SCHEMA_VERSION = 3
_RECEIPT_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_RECEIPT_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.:-]{1,256}$")
_COMPLETION_PENDING_NAME = ".arche_codex_completion_pending.json"
_REQUEST_INTENT_SCHEMA_VERSION = 3
_LEGACY_REQUEST_INTENT_SCHEMA_VERSION = 2
_REQUEST_INTENT_NAME = ".arche_codex_request_intent.json"
_TERMINAL_TURN_SCHEMA_VERSION = 1
_TERMINAL_TURN_NAME = ".arche_codex_terminal_turn.json"
_REQUEST_INTENT_FIELDS = frozenset(
    {
        "schema_version",
        "thread_id",
        "request_digest",
        "turn_count_before",
        "physical_turn_count_before",
        "provider_id",
        "env_key",
        "model",
        "role",
        "stage",
        "profile",
        "workspace",
        "thread_policy_digest",
    }
)
_LEGACY_REQUEST_INTENT_FIELDS = frozenset(
    field for field in _REQUEST_INTENT_FIELDS if field != "physical_turn_count_before"
)
_COMPLETION_PENDING_FIELDS = frozenset(
    {
        "schema_version",
        "thread_id",
        "turn_id",
        "request_digest",
        "turn_count_before",
        "thread_policy_digest",
    }
)
_TERMINAL_TURN_FIELDS = frozenset(
    {
        "schema_version",
        "thread_id",
        "turn_id",
        "status",
        "request_digest",
        "turn_count_before",
        "thread_policy_digest",
    }
)
_TERMINAL_TURN_OPTIONAL_FIELDS = frozenset({"error_code"})
_TRAILING_ROLLBACK_SCHEMA_VERSION = 1
_TRAILING_ROLLBACK_INTENT_NAME = ".arche_codex_trailing_rollback_intent.json"
_TRAILING_ROLLBACK_RECEIPT_NAME = ".arche_codex_trailing_rollback_receipt.json"
_TRAILING_ROLLBACK_FIELDS = frozenset(
    {
        "schema_version",
        "state",
        "thread_id",
        "receipt_turn_id",
        "request_digest",
        "turn_count_before",
        "physical_turn_count_before",
        "trailing_turn_ids",
        "preserved_turn_ids",
        "thread_policy_digest",
    }
)
_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "thread_id",
        "turn_id",
        "request_digest",
        "turn_count_before",
        "thread_policy_digest",
    }
)
# Goal 使用独立的严格附加记录，避免改变 schema v3 普通 turn receipt。
_GOAL_JOURNAL_SCHEMA_VERSION = 1
_GOAL_INTENT_NAME = ".arche_codex_goal_intent.json"
_GOAL_TURN_JOURNAL_NAME = ".arche_codex_goal_turns.jsonl"
_GOAL_RECEIPT_NAME = ".arche_codex_goal_receipt.json"
_GOAL_INTENT_FIELDS = frozenset(
    {
        "schema_version",
        "thread_id",
        "logical_goal_id",
        "goal_id",
        "objective_digest",
        "request_digest",
        "turn_count_before",
        "physical_turn_count_before",
        "profile",
        "stage",
        "thread_policy_digest",
        "status",
    }
)
_GOAL_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "thread_id",
        "logical_goal_id",
        "goal_id",
        "objective_digest",
        "request_digest",
        "turn_count_before",
        "physical_turn_count_before",
        "physical_turn_ids",
        "physical_turn_start",
        "physical_turn_end",
        "goal_status",
        "final_turn_status",
        "thread_policy_digest",
    }
)
_GOAL_TURN_FIELDS = frozenset(
    {
        "schema_version",
        "thread_id",
        "logical_goal_id",
        "goal_id",
        "ordinal",
        "turn_id",
        "status",
    }
)


def _read_protocol_field(value: Any, field: str) -> Any:
    """同时读取 SDK 模型属性、字典字段和协议的 camelCase 别名。"""

    aliases = {
        "thread_id": "threadId",
        "turn_id": "turnId",
        "turn_count_before": "turnCountBefore",
        "will_retry": "willRetry",
        "items_view": "itemsView",
        "item_id": "itemId",
        "process_id": "processId",
        "process_handle": "processHandle",
    }
    alias = aliases.get(field, field)
    if isinstance(value, Mapping):
        if field in value:
            return value[field]
        if alias in value:
            return value[alias]
        nested_params = value.get("params")
        if isinstance(nested_params, Mapping):
            if field in nested_params:
                return nested_params[field]
            return nested_params.get(alias)
        return None
    direct = getattr(value, field, None)
    if direct is not None:
        return direct
    # 未知通知由 SDK 包装为 UnknownNotification(params=...)；当前 process
    # 事件属于该路径，仍需读取其协议字段才能清理 commandExecution 映射。
    params = getattr(value, "params", None)
    if isinstance(params, Mapping):
        if field in params:
            return params[field]
        return params.get(alias)
    return None


def _protocol_enum_value(value: Any) -> Any:
    """把 SDK Enum 或测试字典值统一成协议字符串。"""

    return getattr(value, "value", value)


def _validate_receipt_identifier(value: Any, field: str) -> str:
    """校验 receipt 中用于身份核对和文件名的标识。"""

    if not isinstance(value, str) or _RECEIPT_IDENTIFIER.fullmatch(value) is None:
        raise HarnessError(f"completed turn receipt {field} is malformed", kind="session_state")
    return value


def _validate_request_digest(value: Any) -> str:
    """校验当前 prompt 的不可逆摘要，不把 prompt 原文写入 receipt。"""

    if not isinstance(value, str) or _RECEIPT_DIGEST.fullmatch(value) is None:
        raise HarnessError("completed turn request digest is malformed", kind="session_state")
    return value


def _validate_thread_policy_digest(value: Any) -> str:
    """校验 effective thread policy 的不可逆摘要。"""

    if not isinstance(value, str) or _POLICY_DIGEST.fullmatch(value) is None:
        raise HarnessError("Codex thread policy digest is malformed", kind="session_state")
    return value


def _validate_intent_text(value: Any, field: str) -> str:
    """校验 request-intent 中的公开配置身份文本，不允许空值或控制字符。"""

    if not isinstance(value, str) or not value.strip() or any(
        ord(char) < 32 or ord(char) == 127 for char in value
    ):
        raise HarnessError(f"request intent {field} is malformed", kind="session_state")
    return value.strip()


def _canonical_request_value(value: Any, field: str) -> Any:
    """把 HarnessRequest 转成稳定 JSON 值，不把不可预测 repr 当摘要输入。"""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise HarnessError(
                f"Codex request {field} contains a non-finite number",
                kind="provider_envelope",
            )
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise HarnessError(
                    f"Codex request {field} contains a non-string key",
                    kind="provider_envelope",
                )
            result[key] = _canonical_request_value(item, f"{field}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [
            _canonical_request_value(item, f"{field}[{index}]")
            for index, item in enumerate(value)
        ]
    raise HarnessError(
        f"Codex request {field} contains an unsupported value",
        kind="provider_envelope",
    )


def calculate_codex_request_digest(request: HarnessRequest) -> str:
    """计算模型请求摘要，供 request-intent 和 receipt 防重放。

    输入是已经完成内容校验的 ``HarnessRequest``。动作覆盖模型实际可见的消息、
    工具、schema、采样参数和业务 metadata，但删除只用于把请求路由回既有
    session 的 ``codex_thread_id``；thread 身份仍由顶层 ``thread_id`` 与 manifest
    独立核对。返回 SHA-256 摘要，不写文件；请求含非 JSON 值时抛出合同错误。
    """

    digest_metadata = {
        key: value
        for key, value in request.metadata.items()
        if key != "codex_thread_id"
    }

    vector = {
        "messages": request.messages,
        "model": request.model,
        "role": request.role,
        "stage": request.stage,
        "profile": request.profile,
        "research_task_id": request.research_task_id,
        "invocation_id": request.invocation_id,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "timeout_seconds": request.timeout_seconds,
        "response_schema": request.response_schema,
        "tools": request.tools,
        "metadata": digest_metadata,
        "prompt_text": request.prompt_text(),
    }
    encoded = json.dumps(
        _canonical_request_value(vector, "request"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _request_intent_path(root: Path | str) -> Path:
    """返回 request-intent 文件路径；文件位于 receipt 目录旁。"""

    directory = _prepare_receipt_directory(root)
    return directory.parent / _REQUEST_INTENT_NAME


def _validate_request_intent(value: Any, path: Path) -> dict[str, Any]:
    """严格校验当前或待显式迁移的旧版 request-intent。"""

    if not isinstance(value, Mapping):
        raise HarnessError(
            "request intent fields are invalid; effective thread policy digest is required",
            kind="session_state",
        )
    schema_version = value.get("schema_version")
    expected_fields = {
        _REQUEST_INTENT_SCHEMA_VERSION: _REQUEST_INTENT_FIELDS,
        _LEGACY_REQUEST_INTENT_SCHEMA_VERSION: _LEGACY_REQUEST_INTENT_FIELDS,
    }.get(schema_version)
    if expected_fields is None:
        raise HarnessError("request intent schema is invalid", kind="session_state")
    if set(value) != expected_fields:
        raise HarnessError(
            "request intent fields are invalid; effective thread policy digest is required",
            kind="session_state",
        )
    if path.name != _REQUEST_INTENT_NAME:
        raise HarnessError("request intent filename is invalid", kind="session_state")
    thread_id = _validate_receipt_identifier(value.get("thread_id"), "thread_id")
    request_digest = _validate_request_digest(value.get("request_digest"))
    thread_policy_digest = _validate_thread_policy_digest(value.get("thread_policy_digest"))
    turn_count_before = value.get("turn_count_before")
    if (
        isinstance(turn_count_before, bool)
        or not isinstance(turn_count_before, int)
        or turn_count_before < 0
    ):
        raise HarnessError("request intent turn_count_before is invalid", kind="session_state")
    validated = {
        "schema_version": schema_version,
        "thread_id": thread_id,
        "request_digest": request_digest,
        "turn_count_before": turn_count_before,
        "provider_id": _validate_intent_text(value.get("provider_id"), "provider_id"),
        "env_key": _validate_intent_text(value.get("env_key"), "env_key"),
        "model": _validate_intent_text(value.get("model"), "model"),
        "role": _validate_intent_text(value.get("role"), "role"),
        "stage": _validate_intent_text(value.get("stage"), "stage"),
        "profile": _validate_intent_text(value.get("profile"), "profile"),
        "workspace": _validate_intent_text(value.get("workspace"), "workspace"),
        "thread_policy_digest": thread_policy_digest,
    }
    if schema_version == _REQUEST_INTENT_SCHEMA_VERSION:
        physical_turn_count_before = value.get("physical_turn_count_before")
        if (
            isinstance(physical_turn_count_before, bool)
            or not isinstance(physical_turn_count_before, int)
            or physical_turn_count_before < 0
        ):
            raise HarnessError(
                "request intent physical_turn_count_before is invalid",
                kind="session_state",
            )
        validated["physical_turn_count_before"] = physical_turn_count_before
    return validated


def _read_request_intent(root: Path | str) -> dict[str, Any] | None:
    """读取 request-intent；损坏、符号链接或临时残留时 fail-closed。"""

    path = _request_intent_path(root)
    if path.is_symlink():
        raise HarnessError("request intent must not be a symlink", kind="session_state")
    try:
        incomplete = [
            child
            for child in path.parent.iterdir()
            if child.name.startswith(f"{_REQUEST_INTENT_NAME}.")
        ]
    except OSError as exc:
        raise HarnessError("request intent directory cannot be listed", kind="session_state") from exc
    if incomplete:
        raise HarnessError("request intent has an incomplete entry", kind="session_state")
    if not path.exists():
        return None
    if not path.is_file():
        raise HarnessError("request intent is not a regular file", kind="session_state")
    try:
        os.chmod(path, 0o600)
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HarnessError("request intent cannot be read", kind="session_state") from exc
    return _validate_request_intent(value, path)


def _write_request_intent(
    root: Path | str,
    *,
    thread_id: str,
    request_digest: str,
    turn_count_before: int,
    physical_turn_count_before: int,
    provider_id: str,
    env_key: str,
    model: str,
    role: str,
    stage: str,
    profile: str,
    workspace: Path | str,
    thread_policy_digest: str,
) -> Path:
    """在 worker 启动前原子保存完整请求意图，供崩溃恢复阻止重复 turn。"""

    path = _request_intent_path(root)
    payload = {
        "schema_version": _REQUEST_INTENT_SCHEMA_VERSION,
        "thread_id": _validate_receipt_identifier(thread_id, "thread_id"),
        "request_digest": _validate_request_digest(request_digest),
        "turn_count_before": turn_count_before,
        "physical_turn_count_before": physical_turn_count_before,
        "provider_id": _validate_intent_text(provider_id, "provider_id"),
        "env_key": _validate_intent_text(env_key, "env_key"),
        "model": _validate_intent_text(model, "model"),
        "role": _validate_intent_text(role, "role"),
        "stage": _validate_intent_text(stage, "stage"),
        "profile": _validate_intent_text(profile, "profile"),
        "workspace": _validate_intent_text(str(Path(workspace).resolve()), "workspace"),
        "thread_policy_digest": _validate_thread_policy_digest(thread_policy_digest),
    }
    if (
        isinstance(turn_count_before, bool)
        or not isinstance(turn_count_before, int)
        or turn_count_before < 0
    ):
        raise HarnessError("request intent turn_count_before is invalid", kind="session_state")
    if (
        isinstance(physical_turn_count_before, bool)
        or not isinstance(physical_turn_count_before, int)
        or physical_turn_count_before < 0
    ):
        raise HarnessError(
            "request intent physical_turn_count_before is invalid",
            kind="session_state",
        )
    existing = _read_request_intent(root)
    if existing is not None:
        if existing != payload:
            raise HarnessError("request intent conflicts with an existing turn", kind="session_state")
        return path
    _atomic_write_json_file(path, payload, description="request intent")
    return path


def _remove_request_intent(root: Path | str) -> None:
    """删除已被 receipt 或显式无 turn 确认替代的 request-intent。"""

    path = _request_intent_path(root)
    if path.is_symlink():
        raise HarnessError("request intent must not be a symlink", kind="session_state")
    _remove_file_and_fsync_directory(path, description="request intent")


def _terminal_turn_journal_path(root: Path | str) -> Path:
    """返回 terminal-turn journal 路径；文件位于 receipt 目录旁。"""

    directory = _prepare_receipt_directory(root)
    return directory.parent / _TERMINAL_TURN_NAME


def _validate_terminal_turn_journal(value: Any, path: Path) -> dict[str, Any]:
    """严格校验只含终态身份的 journal，不接受任意 provider 字段。"""

    if not isinstance(value, Mapping) or set(value) not in {
        _TERMINAL_TURN_FIELDS,
        _TERMINAL_TURN_FIELDS | _TERMINAL_TURN_OPTIONAL_FIELDS,
    }:
        raise HarnessError(
            "terminal turn journal fields are invalid",
            kind="session_state",
        )
    if value.get("schema_version") != _TERMINAL_TURN_SCHEMA_VERSION:
        raise HarnessError("terminal turn journal schema is invalid", kind="session_state")
    if path.name != _TERMINAL_TURN_NAME:
        raise HarnessError("terminal turn journal filename is invalid", kind="session_state")
    status = value.get("status")
    if status not in {"failed", "interrupted"}:
        raise HarnessError("terminal turn journal status is invalid", kind="session_state")
    turn_count_before = value.get("turn_count_before")
    if (
        isinstance(turn_count_before, bool)
        or not isinstance(turn_count_before, int)
        or turn_count_before < 0
    ):
        raise HarnessError(
            "terminal turn journal turn_count_before is invalid",
            kind="session_state",
        )
    normalized = {
        "schema_version": _TERMINAL_TURN_SCHEMA_VERSION,
        "thread_id": _validate_receipt_identifier(value.get("thread_id"), "thread_id"),
        "turn_id": _validate_receipt_identifier(value.get("turn_id"), "turn_id"),
        "status": status,
        "request_digest": _validate_request_digest(value.get("request_digest")),
        "turn_count_before": turn_count_before,
        "thread_policy_digest": _validate_thread_policy_digest(
            value.get("thread_policy_digest")
        ),
    }
    if "error_code" in value:
        error_code = value.get("error_code")
        if error_code not in CODEX_TERMINAL_PROVIDER_ERROR_CODES:
            raise HarnessError(
                "terminal turn journal provider error code is not approved",
                kind="session_state",
            )
        normalized["error_code"] = error_code
    return normalized


def _read_terminal_turn_journal(root: Path | str) -> dict[str, Any] | None:
    """读取 terminal-turn journal；损坏、链接或临时残留时失败关闭。"""

    path = _terminal_turn_journal_path(root)
    if path.is_symlink():
        raise HarnessError("terminal turn journal must not be a symlink", kind="session_state")
    try:
        incomplete = [
            child
            for child in path.parent.iterdir()
            if child.name.startswith(f"{_TERMINAL_TURN_NAME}.")
        ]
    except OSError as exc:
        raise HarnessError(
            "terminal turn journal directory cannot be listed",
            kind="session_state",
        ) from exc
    if incomplete:
        raise HarnessError("terminal turn journal has an incomplete entry", kind="session_state")
    if not path.exists():
        return None
    if not path.is_file():
        raise HarnessError("terminal turn journal is not a regular file", kind="session_state")
    try:
        os.chmod(path, 0o600)
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HarnessError("terminal turn journal cannot be read", kind="session_state") from exc
    return _validate_terminal_turn_journal(parsed, path)


def _write_terminal_turn_journal(
    root: Path | str,
    *,
    thread_id: str,
    turn_id: str,
    status: str,
    request_digest: str,
    turn_count_before: int,
    thread_policy_digest: str,
    error_code: str | None = None,
) -> Path:
    """以 0600 原子文件保存终态身份，文件与目录都完成 fsync。

    输入只含 thread、turn、状态、两个摘要和成功 turn 旧计数；动作先严格构造
    白名单 payload，再复用原子写入和目录刷盘 helper。相同身份幂等返回，冲突、
    非法字段或写盘失败时抛错，既有 request-intent 不会在本函数内改变。
    """

    path = _terminal_turn_journal_path(root)
    candidate = {
        "schema_version": _TERMINAL_TURN_SCHEMA_VERSION,
        "thread_id": thread_id,
        "turn_id": turn_id,
        "status": status,
        "request_digest": request_digest,
        "turn_count_before": turn_count_before,
        "thread_policy_digest": thread_policy_digest,
    }
    if error_code is not None:
        candidate["error_code"] = error_code
    payload = _validate_terminal_turn_journal(
        candidate,
        path,
    )
    existing = _read_terminal_turn_journal(root)
    if existing is not None:
        if existing != payload:
            raise HarnessError(
                "terminal turn journal conflicts with an existing turn",
                kind="session_state",
            )
        return path
    _atomic_write_json_file(path, payload, description="terminal turn journal", mode=0o600)
    return path


def _remove_terminal_turn_journal(root: Path | str) -> None:
    """在同步 acknowledgement 成功后幂等删除终态 journal 并刷盘目录。"""

    path = _terminal_turn_journal_path(root)
    if path.is_symlink():
        raise HarnessError("terminal turn journal must not be a symlink", kind="session_state")
    _remove_file_and_fsync_directory(path, description="terminal turn journal")


def _completion_pending_path(root: Path | str) -> Path:
    """返回 completed receipt 目录旁的 completion-pending journal 路径。"""

    directory = _prepare_receipt_directory(root)
    return directory.parent / _COMPLETION_PENDING_NAME


def _validate_completion_pending(value: Any, path: Path) -> dict[str, Any]:
    """严格校验记录已完成但 receipt 尚未确认落盘的 journal。"""

    if not isinstance(value, Mapping) or set(value) != _COMPLETION_PENDING_FIELDS:
        raise HarnessError(
            "completion pending journal fields are invalid; effective thread policy digest is required",
            kind="session_state",
        )
    if value.get("schema_version") != _RECEIPT_SCHEMA_VERSION:
        raise HarnessError("completion pending journal schema is invalid", kind="session_state")
    thread_id = _validate_receipt_identifier(value.get("thread_id"), "thread_id")
    turn_id = _validate_receipt_identifier(value.get("turn_id"), "turn_id")
    request_digest = _validate_request_digest(value.get("request_digest"))
    thread_policy_digest = _validate_thread_policy_digest(value.get("thread_policy_digest"))
    turn_count_before = value.get("turn_count_before")
    if isinstance(turn_count_before, bool) or not isinstance(turn_count_before, int) or turn_count_before < 0:
        raise HarnessError(
            "completion pending journal turn_count_before is invalid",
            kind="session_state",
        )
    if path.name != _COMPLETION_PENDING_NAME:
        raise HarnessError("completion pending journal filename is invalid", kind="session_state")
    return {
        "schema_version": _RECEIPT_SCHEMA_VERSION,
        "thread_id": thread_id,
        "turn_id": turn_id,
        "request_digest": request_digest,
        "turn_count_before": turn_count_before,
        "thread_policy_digest": thread_policy_digest,
    }


def _read_completion_pending(root: Path | str) -> dict[str, Any] | None:
    """读取 completion-pending journal；损坏或临时残留时 fail-closed。"""

    path = _completion_pending_path(root)
    if path.is_symlink():
        raise HarnessError("completion pending journal must not be a symlink", kind="session_state")
    try:
        incomplete = [
            child
            for child in path.parent.iterdir()
            if child.name.startswith(f"{_COMPLETION_PENDING_NAME}.")
        ]
    except OSError as exc:
        raise HarnessError("completion pending journal directory cannot be listed", kind="session_state") from exc
    if incomplete:
        raise HarnessError("completion pending journal has an incomplete entry", kind="session_state")
    if not path.exists():
        return None
    if not path.is_file():
        raise HarnessError("completion pending journal is not a regular file", kind="session_state")
    try:
        os.chmod(path, 0o600)
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HarnessError("completion pending journal cannot be read", kind="session_state") from exc
    return _validate_completion_pending(parsed, path)


def _write_completion_pending(
    root: Path | str,
    *,
    thread_id: str,
    turn_id: str,
    request_digest: str,
    turn_count_before: int,
    thread_policy_digest: str,
) -> Path:
    """在 receipt 前原子保存完成 turn 的身份，供崩溃恢复而不重发模型。"""

    path = _completion_pending_path(root)
    payload = {
        "schema_version": _RECEIPT_SCHEMA_VERSION,
        "thread_id": _validate_receipt_identifier(thread_id, "thread_id"),
        "turn_id": _validate_receipt_identifier(turn_id, "turn_id"),
        "request_digest": _validate_request_digest(request_digest),
        "turn_count_before": turn_count_before,
        "thread_policy_digest": _validate_thread_policy_digest(thread_policy_digest),
    }
    if isinstance(turn_count_before, bool) or not isinstance(turn_count_before, int) or turn_count_before < 0:
        raise HarnessError("completion pending journal turn_count_before is invalid", kind="session_state")
    existing = _read_completion_pending(root)
    if existing is not None:
        if existing != payload:
            raise HarnessError("completion pending journal conflicts with an existing turn", kind="session_state")
        return path
    _atomic_write_json_file(path, payload, description="completion pending journal")
    return path


def _remove_completion_pending(root: Path | str) -> None:
    """删除已由 receipt 或严格 history 恢复替代的 completion journal。"""

    path = _completion_pending_path(root)
    if path.is_symlink():
        raise HarnessError("completion pending journal must not be a symlink", kind="session_state")
    _remove_file_and_fsync_directory(path, description="completion pending journal")


def _prepare_receipt_directory(path: Path | str) -> Path:
    """创建并收紧 completed-turn receipt 目录，拒绝符号链接。"""

    raw_path = Path(path).expanduser()
    if raw_path.is_symlink():
        raise HarnessError("completed turn receipt directory must not be a symlink", kind="session_state")
    try:
        resolved = raw_path.resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        if resolved.is_symlink() or not resolved.is_dir():
            raise HarnessError(
                "completed turn receipt directory is not a regular directory",
                kind="session_state",
            )
        os.chmod(resolved, 0o700)
    except HarnessError:
        raise
    except OSError as exc:
        raise HarnessError(
            "completed turn receipt directory cannot be prepared",
            kind="session_state",
        ) from exc
    return resolved


def _resolve_goal_record_parent(
    root: Path | str,
    *,
    create: bool,
) -> Path | None:
    """解析 Goal 记录所在目录；只读查询不得创建目录或修改权限。"""

    try:
        raw_path = Path(root).expanduser()
    except (TypeError, ValueError) as exc:
        raise HarnessError(
            "completed turn receipt directory is invalid",
            kind="session_state",
        ) from exc
    if raw_path.is_symlink():
        raise HarnessError(
            "completed turn receipt directory must not be a symlink",
            kind="session_state",
        )
    if create:
        return _prepare_receipt_directory(raw_path).parent
    if not raw_path.exists():
        return None
    if not raw_path.is_dir():
        raise HarnessError(
            "completed turn receipt directory is missing or not a regular directory",
            kind="session_state",
        )
    try:
        resolved = raw_path.resolve(strict=True)
    except OSError as exc:
        raise HarnessError(
            "completed turn receipt directory cannot be resolved",
            kind="session_state",
        ) from exc
    if resolved.is_symlink() or not resolved.is_dir():
        raise HarnessError(
            "completed turn receipt directory is not a regular directory",
            kind="session_state",
        )
    return resolved.parent


def _goal_intent_path(root: Path | str) -> Path:
    """返回 Goal 意图文件路径；Goal 记录与普通收据文件分开放置。"""

    parent = _resolve_goal_record_parent(root, create=True)
    assert parent is not None
    return parent / _GOAL_INTENT_NAME


def _goal_turn_journal_path(root: Path | str) -> Path:
    """返回 Goal 物理 turn 顺序事实文件路径。"""

    parent = _resolve_goal_record_parent(root, create=True)
    assert parent is not None
    return parent / _GOAL_TURN_JOURNAL_NAME


def _goal_receipt_path(root: Path | str) -> Path:
    """返回 Goal 逻辑完成收据文件路径。"""

    parent = _resolve_goal_record_parent(root, create=True)
    assert parent is not None
    return parent / _GOAL_RECEIPT_NAME


def _validate_goal_identifier(value: Any, label: str, *, allow_empty: bool = False) -> str:
    """校验 Goal 相关固定身份，拒绝正文、路径和控制字符。"""

    if allow_empty and value == "":
        return ""
    return _validate_receipt_identifier(value, label)


def _validate_goal_status(value: Any, label: str = "goal_status") -> str:
    """把 0.149 Goal 状态规范化为内部固定枚举。"""

    if hasattr(value, "value"):
        value = value.value
    if not isinstance(value, str):
        raise HarnessError(f"{label} is invalid", kind="session_protocol")
    normalized = re.sub(r"(?<!^)(?=[A-Z])", "_", value.strip()).lower()
    normalized = normalized.replace("-", "_")
    aliases = {
        "usagelimited": "usage_limited",
        "budgetlimited": "budget_limited",
        "in_progress": "active",
        "running": "active",
        "cancel": "cancelled",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {
        "active",
        "paused",
        "blocked",
        "usage_limited",
        "budget_limited",
        "complete",
        "cancelled",
        "interrupted",
    }:
        raise HarnessError(f"{label} has unsupported status", kind="session_protocol")
    return normalized


def _validate_goal_turn_status(value: Any, label: str = "turn status") -> str:
    """校验物理 turn 只能记录固定协议状态。"""

    if hasattr(value, "value"):
        value = value.value
    if not isinstance(value, str):
        raise HarnessError(f"{label} is invalid", kind="session_protocol")
    normalized = re.sub(r"(?<!^)(?=[A-Z])", "_", value.strip()).lower()
    normalized = normalized.replace("-", "_")
    if normalized in {"in_progress", "running", "started"}:
        normalized = "active"
    if normalized in {"success", "succeeded", "done"}:
        normalized = "completed"
    if normalized not in {"active", "completed", "failed", "interrupted", "aborted"}:
        raise HarnessError(f"{label} has unsupported status", kind="session_protocol")
    return normalized


def _validate_goal_intent(value: Any, path: Path) -> dict[str, Any]:
    """严格校验 Goal 意图，只接受固定身份和摘要字段。"""

    if not isinstance(value, Mapping) or set(value) != _GOAL_INTENT_FIELDS:
        raise HarnessError("Goal intent fields are invalid", kind="session_state")
    if value.get("schema_version") != _GOAL_JOURNAL_SCHEMA_VERSION:
        raise HarnessError("Goal intent schema is invalid", kind="session_state")
    if path.name != _GOAL_INTENT_NAME:
        raise HarnessError("Goal intent filename is invalid", kind="session_state")
    status = _validate_goal_status(value.get("status"), "Goal intent status")
    normalized = {
        "schema_version": _GOAL_JOURNAL_SCHEMA_VERSION,
        "thread_id": _validate_goal_identifier(value.get("thread_id"), "thread_id"),
        "logical_goal_id": _validate_goal_identifier(
            value.get("logical_goal_id"), "logical_goal_id"
        ),
        "goal_id": _validate_goal_identifier(value.get("goal_id"), "goal_id", allow_empty=True),
        "objective_digest": _validate_request_digest(value.get("objective_digest")),
        "request_digest": _validate_request_digest(value.get("request_digest")),
        "turn_count_before": value.get("turn_count_before"),
        "physical_turn_count_before": value.get("physical_turn_count_before"),
        "profile": value.get("profile"),
        "stage": value.get("stage"),
        "thread_policy_digest": _validate_thread_policy_digest(
            value.get("thread_policy_digest")
        ),
        "status": status,
    }
    for field in ("turn_count_before", "physical_turn_count_before"):
        number = normalized[field]
        if isinstance(number, bool) or not isinstance(number, int) or number < 0:
            raise HarnessError(f"Goal intent {field} is invalid", kind="session_state")
    for field in ("profile", "stage"):
        if not isinstance(normalized[field], str) or not normalized[field].strip():
            raise HarnessError(f"Goal intent {field} is invalid", kind="session_state")
    return normalized


def _read_goal_intent(root: Path | str) -> dict[str, Any] | None:
    """读取 Goal 意图；损坏或临时文件会阻止恢复。"""

    path = _goal_intent_path(root)
    if path.is_symlink():
        raise HarnessError("Goal intent must not be a symlink", kind="session_state")
    incomplete = list(path.parent.glob(f"{_GOAL_INTENT_NAME}.*"))
    if incomplete:
        raise HarnessError("Goal intent has an incomplete entry", kind="session_state")
    if not path.exists():
        return None
    if not path.is_file():
        raise HarnessError("Goal intent is not a regular file", kind="session_state")
    try:
        os.chmod(path, 0o600)
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise HarnessError("Goal intent cannot be read", kind="session_state") from None
    return _validate_goal_intent(value, path)


def _write_goal_intent(
    root: Path | str,
    *,
    thread_id: str,
    logical_goal_id: str,
    goal_id: str,
    objective_digest: str,
    request_digest: str,
    turn_count_before: int,
    physical_turn_count_before: int,
    profile: str,
    stage: str,
    thread_policy_digest: str,
    status: str,
) -> Path:
    """原子保存 Goal 意图；已有不同身份时拒绝覆盖。"""

    path = _goal_intent_path(root)
    payload = _validate_goal_intent(
        {
            "schema_version": _GOAL_JOURNAL_SCHEMA_VERSION,
            "thread_id": thread_id,
            "logical_goal_id": logical_goal_id,
            "goal_id": goal_id,
            "objective_digest": objective_digest,
            "request_digest": request_digest,
            "turn_count_before": turn_count_before,
            "physical_turn_count_before": physical_turn_count_before,
            "profile": profile,
            "stage": stage,
            "thread_policy_digest": thread_policy_digest,
            "status": status,
        },
        path,
    )
    existing = _read_goal_intent(root)
    if existing is not None:
        if existing != payload:
            raise HarnessError("Goal intent conflicts with an existing operation", kind="session_state")
        return path
    _atomic_write_json_file(path, payload, description="Goal intent", mode=0o600)
    return path


def _update_goal_intent(root: Path | str, **updates: Any) -> dict[str, Any]:
    """只更新已有 Goal 意图的固定状态或 provider goal 身份。"""

    current = _read_goal_intent(root)
    if current is None:
        raise HarnessError("Goal intent is missing", kind="session_state")
    candidate = {**current, **updates}
    if current["goal_id"] and candidate["goal_id"] != current["goal_id"]:
        raise HarnessError(
            "Goal provider identity cannot be replaced or cleared",
            kind="session_state",
        )
    path = _goal_intent_path(root)
    normalized = _validate_goal_intent(candidate, path)
    _atomic_write_json_file(path, normalized, description="Goal intent update", mode=0o600)
    return normalized


def _validate_goal_turn_record(value: Any, path: Path) -> dict[str, Any]:
    """严格校验一个物理 turn 顺序事实。"""

    if not isinstance(value, Mapping) or set(value) != _GOAL_TURN_FIELDS:
        raise HarnessError("Goal physical turn fields are invalid", kind="session_state")
    if value.get("schema_version") != _GOAL_JOURNAL_SCHEMA_VERSION:
        raise HarnessError("Goal physical turn schema is invalid", kind="session_state")
    ordinal = value.get("ordinal")
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
        raise HarnessError("Goal physical turn ordinal is invalid", kind="session_state")
    return {
        "schema_version": _GOAL_JOURNAL_SCHEMA_VERSION,
        "thread_id": _validate_goal_identifier(value.get("thread_id"), "thread_id"),
        "logical_goal_id": _validate_goal_identifier(
            value.get("logical_goal_id"), "logical_goal_id"
        ),
        "goal_id": _validate_goal_identifier(value.get("goal_id"), "goal_id", allow_empty=True),
        "ordinal": ordinal,
        "turn_id": _validate_goal_identifier(value.get("turn_id"), "turn_id"),
        "status": _validate_goal_turn_status(value.get("status")),
    }


def _read_goal_turn_journal(root: Path | str) -> list[dict[str, Any]]:
    """读取并核对连续物理 turn 顺序，拒绝重复、缺口和 provider 正文。"""

    path = _goal_turn_journal_path(root)
    if path.is_symlink():
        raise HarnessError("Goal physical turn journal must not be a symlink", kind="session_state")
    incomplete = list(path.parent.glob(f"{_GOAL_TURN_JOURNAL_NAME}.*"))
    if incomplete:
        raise HarnessError("Goal physical turn journal has an incomplete entry", kind="session_state")
    if not path.exists():
        return []
    if not path.is_file():
        raise HarnessError("Goal physical turn journal is not a regular file", kind="session_state")
    try:
        os.chmod(path, 0o600)
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        raise HarnessError("Goal physical turn journal cannot be read", kind="session_state") from None
    records: list[dict[str, Any]] = []
    for line in lines:
        try:
            records.append(_validate_goal_turn_record(json.loads(line), path))
        except (TypeError, json.JSONDecodeError):
            raise HarnessError("Goal physical turn journal contains invalid JSON", kind="session_state") from None
    for expected, record in enumerate(records):
        if record["ordinal"] != expected:
            raise HarnessError("Goal physical turn journal order is not continuous", kind="session_state")
    if len({record["turn_id"] for record in records}) != len(records):
        raise HarnessError("Goal physical turn journal has duplicate identity", kind="session_protocol")
    if records:
        first_thread_id = records[0]["thread_id"]
        first_logical_goal_id = records[0]["logical_goal_id"]
        goal_ids = {record["goal_id"] for record in records}
        if any(
            record["thread_id"] != first_thread_id
            or record["logical_goal_id"] != first_logical_goal_id
            for record in records
        ):
            raise HarnessError(
                "Goal physical turn journal identity is inconsistent",
                kind="session_state",
            )
        if len(goal_ids) != 1:
            raise HarnessError(
                "Goal physical turn journal provider identity is inconsistent",
                kind="session_state",
            )
    return records


def _write_goal_turn_journal(
    root: Path | str,
    *,
    thread_id: str,
    logical_goal_id: str,
    goal_id: str,
    turn_records: Sequence[Mapping[str, Any]],
) -> Path:
    """原子保存所有物理 turn 的顺序和固定状态，不写入事件正文。"""

    path = _goal_turn_journal_path(root)
    normalized: list[dict[str, Any]] = []
    for ordinal, record in enumerate(turn_records):
        normalized.append(
            _validate_goal_turn_record(
                {
                    "schema_version": _GOAL_JOURNAL_SCHEMA_VERSION,
                    "thread_id": thread_id,
                    "logical_goal_id": logical_goal_id,
                    "goal_id": goal_id,
                    "ordinal": ordinal,
                    "turn_id": record.get("turn_id"),
                    "status": record.get("status"),
                },
                path,
            )
        )
    existing = _read_goal_turn_journal(root)
    if normalized and len({record["goal_id"] for record in normalized}) != 1:
        raise HarnessError(
            "Goal physical turn journal provider identity is inconsistent",
            kind="session_state",
        )
    if existing:
        if len(normalized) < len(existing):
            raise HarnessError("Goal physical turn journal was shortened", kind="session_state")
        for old, new in zip(existing, normalized):
            if (
                old["thread_id"] != new["thread_id"]
                or old["logical_goal_id"] != new["logical_goal_id"]
                or old["turn_id"] != new["turn_id"]
                or (
                    old["goal_id"] != new["goal_id"]
                    and not (old["goal_id"] == "" and new["goal_id"] != "")
                )
            ):
                raise HarnessError("Goal physical turn journal changed identity", kind="session_state")
            if old["status"] != new["status"] and old["status"] != "active":
                raise HarnessError("Goal physical turn journal changed terminal status", kind="session_state")
        if len(normalized) == len(existing) and all(
            old["status"] == new["status"] for old, new in zip(existing, normalized)
        ):
            return path
    if not normalized:
        raise HarnessError("Goal has no physical turn identity", kind="session_protocol")
    encoded = "".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in normalized)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise
    return path


def _validate_goal_receipt(value: Any, path: Path) -> dict[str, Any]:
    """严格校验 Goal 逻辑完成收据和完整物理范围。"""

    if not isinstance(value, Mapping) or set(value) != _GOAL_RECEIPT_FIELDS:
        raise HarnessError("Goal receipt fields are invalid", kind="session_state")
    if value.get("schema_version") != _GOAL_JOURNAL_SCHEMA_VERSION:
        raise HarnessError("Goal receipt schema is invalid", kind="session_state")
    if path.name != _GOAL_RECEIPT_NAME:
        raise HarnessError("Goal receipt filename is invalid", kind="session_state")
    physical_ids = value.get("physical_turn_ids")
    if not isinstance(physical_ids, list) or not physical_ids or any(
        not isinstance(item, str) for item in physical_ids
    ):
        raise HarnessError("Goal receipt physical turn range is invalid", kind="session_state")
    normalized_ids = [_validate_goal_identifier(item, "physical_turn_id") for item in physical_ids]
    if len(set(normalized_ids)) != len(normalized_ids):
        raise HarnessError("Goal receipt physical turn range is ambiguous", kind="session_protocol")
    if value.get("physical_turn_start") != normalized_ids[0] or value.get("physical_turn_end") != normalized_ids[-1]:
        raise HarnessError("Goal receipt physical turn endpoints are invalid", kind="session_state")
    normalized = {
        "schema_version": _GOAL_JOURNAL_SCHEMA_VERSION,
        "thread_id": _validate_goal_identifier(value.get("thread_id"), "thread_id"),
        "logical_goal_id": _validate_goal_identifier(value.get("logical_goal_id"), "logical_goal_id"),
        "goal_id": _validate_goal_identifier(value.get("goal_id"), "goal_id"),
        "objective_digest": _validate_request_digest(value.get("objective_digest")),
        "request_digest": _validate_request_digest(value.get("request_digest")),
        "turn_count_before": value.get("turn_count_before"),
        "physical_turn_count_before": value.get("physical_turn_count_before"),
        "physical_turn_ids": normalized_ids,
        "physical_turn_start": normalized_ids[0],
        "physical_turn_end": normalized_ids[-1],
        "goal_status": _validate_goal_status(value.get("goal_status")),
        "final_turn_status": _validate_goal_turn_status(value.get("final_turn_status")),
        "thread_policy_digest": _validate_thread_policy_digest(value.get("thread_policy_digest")),
    }
    if normalized["goal_status"] != "complete" or normalized["final_turn_status"] != "completed":
        raise HarnessError("Goal receipt requires complete successful final turn", kind="session_protocol")
    for field in ("turn_count_before", "physical_turn_count_before"):
        number = normalized[field]
        if isinstance(number, bool) or not isinstance(number, int) or number < 0:
            raise HarnessError(f"Goal receipt {field} is invalid", kind="session_state")
    return normalized


def _read_goal_receipt(root: Path | str) -> dict[str, Any] | None:
    """读取 Goal 收据；缺失返回空，损坏或半写入则失败关闭。"""

    path = _goal_receipt_path(root)
    if path.is_symlink():
        raise HarnessError("Goal receipt must not be a symlink", kind="session_state")
    incomplete = list(path.parent.glob(f"{_GOAL_RECEIPT_NAME}.*"))
    if incomplete:
        raise HarnessError("Goal receipt has an incomplete entry", kind="session_state")
    if not path.exists():
        return None
    if not path.is_file():
        raise HarnessError("Goal receipt is not a regular file", kind="session_state")
    try:
        os.chmod(path, 0o600)
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise HarnessError("Goal receipt cannot be read", kind="session_state") from None
    return _validate_goal_receipt(value, path)


def _read_goal_receipt_read_only(root: Path | str) -> dict[str, Any] | None:
    """只读读取已经存在的 Goal 收据，不创建目录或修正权限。"""

    parent = _resolve_goal_record_parent(root, create=False)
    if parent is None:
        return None
    path = parent / _GOAL_RECEIPT_NAME
    if path.is_symlink():
        raise HarnessError("Goal receipt must not be a symlink", kind="session_state")
    try:
        incomplete = tuple(parent.glob(f"{_GOAL_RECEIPT_NAME}.*"))
    except OSError as exc:
        raise HarnessError("Goal receipt directory cannot be listed", kind="session_state") from exc
    if incomplete:
        raise HarnessError("Goal receipt has an incomplete entry", kind="session_state")
    if not path.exists():
        return None
    if not path.is_file():
        raise HarnessError("Goal receipt is not a regular file", kind="session_state")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise HarnessError("Goal receipt cannot be read", kind="session_state") from None
    return _validate_goal_receipt(value, path)


def _remove_goal_file(root: Path | str, path_builder: Callable[[Path | str], Path], description: str) -> None:
    """安全删除一个 Goal 附加文件并刷盘目录。"""

    path = path_builder(root)
    if path.is_symlink():
        raise HarnessError(f"{description} must not be a symlink", kind="session_state")
    _remove_file_and_fsync_directory(path, description=description)


def _remove_goal_receipt(root: Path | str) -> None:
    """删除已由外层事实确认的 Goal 完成收据。"""

    _remove_goal_file(root, _goal_receipt_path, "Goal receipt")


def _remove_goal_intent(root: Path | str) -> None:
    """删除已确认完成的 Goal 意图。"""

    _remove_goal_file(root, _goal_intent_path, "Goal intent")


def _remove_goal_turn_journal(root: Path | str) -> None:
    """删除已确认完成的物理 turn 账本。"""

    _remove_goal_file(root, _goal_turn_journal_path, "Goal physical turn journal")


def _write_goal_receipt(root: Path | str, payload: Mapping[str, Any]) -> Path:
    """原子写入唯一 Goal 完成收据，冲突时拒绝覆盖。"""

    path = _goal_receipt_path(root)
    normalized = _validate_goal_receipt(payload, path)
    existing = _read_goal_receipt(root)
    if existing is not None:
        if existing != normalized:
            raise HarnessError("Goal receipt conflicts with existing logical completion", kind="session_state")
        return path
    _atomic_write_json_file(path, normalized, description="Goal completion receipt", mode=0o600)
    return path


def read_codex_goal_receipt(root: Path | str) -> dict[str, Any] | None:
    """读取严格 Goal 附加收据，供恢复审计和上层确认使用。"""

    return _read_goal_receipt(root)


def _validate_completed_turn_receipt(value: Any, path: Path) -> dict[str, Any]:
    """严格校验不含 prompt/响应正文的 completed-turn receipt。"""

    if not isinstance(value, Mapping):
        raise HarnessError("completed turn receipt must be an object", kind="session_state")
    if set(value) != _RECEIPT_FIELDS:
        raise HarnessError(
            "completed turn receipt fields are invalid; effective thread policy digest is required",
            kind="session_state",
        )
    schema_version = value.get("schema_version")
    if isinstance(schema_version, bool) or schema_version != _RECEIPT_SCHEMA_VERSION:
        raise HarnessError("completed turn receipt schema is invalid", kind="session_state")
    thread_id = _validate_receipt_identifier(value.get("thread_id"), "thread_id")
    turn_id = _validate_receipt_identifier(value.get("turn_id"), "turn_id")
    if path.name != f"{turn_id}.json":
        raise HarnessError("completed turn receipt filename does not match turn_id", kind="session_state")
    request_digest = _validate_request_digest(value.get("request_digest"))
    thread_policy_digest = _validate_thread_policy_digest(value.get("thread_policy_digest"))
    turn_count_before = value.get("turn_count_before")
    if isinstance(turn_count_before, bool) or not isinstance(turn_count_before, int) or turn_count_before < 0:
        raise HarnessError(
            "completed turn receipt turn_count_before is invalid",
            kind="session_state",
        )
    return {
        "schema_version": _RECEIPT_SCHEMA_VERSION,
        "thread_id": thread_id,
        "turn_id": turn_id,
        "request_digest": request_digest,
        "turn_count_before": turn_count_before,
        "thread_policy_digest": thread_policy_digest,
    }


def _read_completed_turn_receipt(
    path: Path,
    *,
    ensure_private_mode: bool = True,
) -> dict[str, Any]:
    """读取一个 receipt，并拒绝损坏文件或符号链接。"""

    if path.is_symlink() or not path.is_file():
        raise HarnessError("completed turn receipt is not a regular file", kind="session_state")
    try:
        if ensure_private_mode:
            os.chmod(path, 0o600)
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HarnessError("completed turn receipt cannot be read", kind="session_state") from exc
    return _validate_completed_turn_receipt(value, path)


def _load_completed_turn_receipts(root: Path | str) -> tuple[dict[str, Any], ...]:
    """加载一个 session 的全部 receipt；发现残留临时文件即 fail-closed。"""

    directory = _prepare_receipt_directory(root)
    return _load_completed_turn_receipts_from_directory(directory)


def _load_completed_turn_receipts_read_only(root: Path | str) -> tuple[dict[str, Any], ...]:
    """只读加载已有 receipt 目录，不为句柄构造创建目录或修正权限。"""

    return _load_completed_turn_receipts_from_directory(
        _resolve_existing_completed_turn_receipt_directory(root),
        read_only=True,
    )


def _resolve_existing_completed_turn_receipt_directory(root: Path | str) -> Path:
    """只解析已经存在的 receipt 目录，不创建目录或修改权限。"""

    try:
        raw_path = Path(root).expanduser()
    except (TypeError, ValueError) as exc:
        raise HarnessError(
            "completed turn receipt directory is invalid",
            kind="session_state",
        ) from exc
    if raw_path.is_symlink() or not raw_path.is_dir():
        raise HarnessError(
            "completed turn receipt directory is missing or not a regular directory",
            kind="session_state",
        )
    try:
        resolved = raw_path.resolve(strict=True)
    except OSError as exc:
        raise HarnessError(
            "completed turn receipt directory cannot be resolved",
            kind="session_state",
        ) from exc
    if resolved.is_symlink() or not resolved.is_dir():
        raise HarnessError(
            "completed turn receipt directory is not a regular directory",
            kind="session_state",
        )
    return resolved


def _load_completed_turn_receipts_from_directory(
    directory: Path,
    *,
    read_only: bool = False,
) -> tuple[dict[str, Any], ...]:
    """从已经核对的 receipt 目录读取全部 JSON 文件。"""

    try:
        children = sorted(directory.iterdir(), key=lambda item: item.name)
    except OSError as exc:
        raise HarnessError("completed turn receipts cannot be listed", kind="session_state") from exc
    receipts: list[dict[str, Any]] = []
    for child in children:
        if child.is_symlink():
            raise HarnessError("completed turn receipt entry must not be a symlink", kind="session_state")
        if not child.is_file() or child.suffix != ".json":
            raise HarnessError("completed turn receipt directory contains an incomplete entry", kind="session_state")
        receipts.append(
            _read_completed_turn_receipt(child, ensure_private_mode=not read_only)
        )
    return tuple(receipts)


def _assert_completed_receipt_recovery_journals_absent(root: Path | str) -> None:
    """确认 receipt 旁的完成、请求、终态和尾部回滚 journal 均已清理。"""

    directory = _resolve_existing_completed_turn_receipt_directory(root)
    journal_names = (
        _COMPLETION_PENDING_NAME,
        _REQUEST_INTENT_NAME,
        _TERMINAL_TURN_NAME,
        _TRAILING_ROLLBACK_INTENT_NAME,
        _TRAILING_ROLLBACK_RECEIPT_NAME,
        _GOAL_INTENT_NAME,
        _GOAL_TURN_JOURNAL_NAME,
        _GOAL_RECEIPT_NAME,
    )
    try:
        children = tuple(directory.parent.iterdir())
    except OSError as exc:
        raise HarnessError(
            "completed turn recovery journal directory cannot be listed",
            kind="session_state",
        ) from exc
    for name in journal_names:
        path = directory.parent / name
        if path.is_symlink() or path.exists():
            raise HarnessError(
                f"completed turn recovery journal remains: {name}",
                kind="recovery_pending",
            )
        if any(child.name.startswith(f"{name}.") for child in children):
            raise HarnessError(
                f"completed turn recovery journal has an incomplete entry: {name}",
                kind="recovery_pending",
            )


def _trailing_rollback_path(root: Path | str, name: str) -> Path:
    """返回 completed receipt 目录旁的尾部回滚状态路径。"""

    directory = _prepare_receipt_directory(root)
    return directory.parent / name


def _validate_trailing_rollback_record(
    value: Any,
    path: Path,
    *,
    expected_state: str,
) -> dict[str, Any]:
    """严格校验只描述已确认失败尾部的回滚 intent 或 receipt。"""

    if not isinstance(value, Mapping) or set(value) != _TRAILING_ROLLBACK_FIELDS:
        raise HarnessError(
            "Codex trailing rollback record fields are invalid",
            kind="session_state",
        )
    if value.get("schema_version") != _TRAILING_ROLLBACK_SCHEMA_VERSION:
        raise HarnessError(
            "Codex trailing rollback record schema is invalid",
            kind="session_state",
        )
    if path.name not in {
        _TRAILING_ROLLBACK_INTENT_NAME,
        _TRAILING_ROLLBACK_RECEIPT_NAME,
    }:
        raise HarnessError(
            "Codex trailing rollback record filename is invalid",
            kind="session_state",
        )
    if value.get("state") != expected_state:
        raise HarnessError(
            "Codex trailing rollback record state is invalid",
            kind="session_state",
        )
    thread_id = _validate_receipt_identifier(value.get("thread_id"), "thread_id")
    receipt_turn_id = _validate_receipt_identifier(
        value.get("receipt_turn_id"),
        "receipt_turn_id",
    )
    request_digest = _validate_request_digest(value.get("request_digest"))
    thread_policy_digest = _validate_thread_policy_digest(
        value.get("thread_policy_digest")
    )
    turn_count_before = value.get("turn_count_before")
    if (
        isinstance(turn_count_before, bool)
        or not isinstance(turn_count_before, int)
        or turn_count_before < 0
    ):
        raise HarnessError(
            "Codex trailing rollback successful count is invalid",
            kind="session_state",
        )
    physical_turn_count_before = value.get("physical_turn_count_before")
    if (
        isinstance(physical_turn_count_before, bool)
        or not isinstance(physical_turn_count_before, int)
        or physical_turn_count_before < 1
    ):
        raise HarnessError(
            "Codex trailing rollback physical count is invalid",
            kind="session_state",
        )
    trailing_turn_ids = value.get("trailing_turn_ids")
    preserved_turn_ids = value.get("preserved_turn_ids")
    if not isinstance(trailing_turn_ids, list) or not isinstance(
        preserved_turn_ids, list
    ):
        raise HarnessError(
            "Codex trailing rollback turn lists are invalid",
            kind="session_state",
        )
    normalized_trailing = [
        _validate_receipt_identifier(turn_id, "trailing_turn_id")
        for turn_id in trailing_turn_ids
    ]
    normalized_preserved = [
        _validate_receipt_identifier(turn_id, "preserved_turn_id")
        for turn_id in preserved_turn_ids
    ]
    if not normalized_trailing or len(set(normalized_trailing)) != len(normalized_trailing):
        raise HarnessError(
            "Codex trailing rollback must identify unique failed turns",
            kind="session_state",
        )
    if len(set(normalized_preserved)) != len(normalized_preserved):
        raise HarnessError(
            "Codex trailing rollback preserved turn identity is ambiguous",
            kind="session_state",
        )
    if physical_turn_count_before != len(normalized_preserved) + len(normalized_trailing):
        raise HarnessError(
            "Codex trailing rollback physical count does not match turn lists",
            kind="session_state",
        )
    if not normalized_preserved or normalized_preserved[-1] != receipt_turn_id:
        raise HarnessError(
            "Codex trailing rollback receipt turn is not the preserved history tail",
            kind="session_state",
        )
    return {
        "schema_version": _TRAILING_ROLLBACK_SCHEMA_VERSION,
        "state": expected_state,
        "thread_id": thread_id,
        "receipt_turn_id": receipt_turn_id,
        "request_digest": request_digest,
        "turn_count_before": turn_count_before,
        "physical_turn_count_before": physical_turn_count_before,
        "trailing_turn_ids": normalized_trailing,
        "preserved_turn_ids": normalized_preserved,
        "thread_policy_digest": thread_policy_digest,
    }


def _read_trailing_rollback_record(
    root: Path | str,
    *,
    receipt: bool,
) -> dict[str, Any] | None:
    """读取尾部回滚状态；缺失返回 None，损坏或半写入直接失败。"""

    name = _TRAILING_ROLLBACK_RECEIPT_NAME if receipt else _TRAILING_ROLLBACK_INTENT_NAME
    expected_state = "rolled_back" if receipt else "intent"
    path = _trailing_rollback_path(root, name)
    if path.is_symlink():
        raise HarnessError(
            "Codex trailing rollback record must not be a symlink",
            kind="session_state",
        )
    try:
        incomplete = [
            child
            for child in path.parent.iterdir()
            if child.name.startswith(f"{name}.")
        ]
    except OSError as exc:
        raise HarnessError(
            "Codex trailing rollback record directory cannot be listed",
            kind="session_state",
        ) from exc
    if incomplete:
        raise HarnessError(
            "Codex trailing rollback record has an incomplete entry",
            kind="session_state",
        )
    if not path.exists():
        return None
    if not path.is_file():
        raise HarnessError(
            "Codex trailing rollback record is not a regular file",
            kind="session_state",
        )
    try:
        os.chmod(path, 0o600)
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HarnessError(
            "Codex trailing rollback record cannot be read",
            kind="session_state",
        ) from exc
    return _validate_trailing_rollback_record(
        value,
        path,
        expected_state=expected_state,
    )


def _write_trailing_rollback_record(
    root: Path | str,
    record: Mapping[str, Any],
    *,
    receipt: bool,
) -> dict[str, Any]:
    """原子保存尾部回滚 intent 或 after receipt，并支持同身份幂等重放。"""

    name = _TRAILING_ROLLBACK_RECEIPT_NAME if receipt else _TRAILING_ROLLBACK_INTENT_NAME
    expected_state = "rolled_back" if receipt else "intent"
    path = _trailing_rollback_path(root, name)
    payload = _validate_trailing_rollback_record(
        dict(record),
        path,
        expected_state=expected_state,
    )
    existing = _read_trailing_rollback_record(root, receipt=receipt)
    if existing is not None:
        if existing != payload:
            raise HarnessError(
                "Codex trailing rollback record conflicts with current identity",
                kind="session_state",
            )
        return existing
    _atomic_write_json_file(
        path,
        payload,
        description="Codex trailing rollback receipt" if receipt else "Codex trailing rollback intent",
        mode=0o600,
    )
    return payload


def _validate_trailing_rollback_identity(
    record: Mapping[str, Any],
    *,
    thread_id: str,
    receipt: Mapping[str, Any],
    thread_policy_digest: str,
) -> None:
    """核对尾部回滚状态与当前 completed receipt、thread 和策略完全一致。"""

    expected = {
        "thread_id": thread_id,
        "receipt_turn_id": receipt["turn_id"],
        "request_digest": receipt["request_digest"],
        "turn_count_before": receipt["turn_count_before"],
        "thread_policy_digest": thread_policy_digest,
    }
    if any(record.get(field) != value for field, value in expected.items()):
        raise HarnessError(
            "Codex trailing rollback identity drifted",
            kind="session_state",
        )


def _read_trailing_turn_plan(
    thread: Any,
    *,
    expected_thread_id: str,
    receipt: Mapping[str, Any],
    thread_policy_digest: str,
    current_turn_count: int,
    request_intent: Mapping[str, Any] | None = None,
) -> tuple[list[Any], dict[str, Any] | None]:
    """核对 waiting receipt 所在 history，并返回需要回滚的失败尾部。"""

    if (
        isinstance(current_turn_count, bool)
        or not isinstance(current_turn_count, int)
        or current_turn_count != int(receipt["turn_count_before"]) + 1
    ):
        raise HarnessError(
            "Codex completed receipt turn count does not match session manifest",
            kind="session_state",
        )
    turns = _read_thread_turns(thread, expected_thread_id=expected_thread_id)
    turn_ids = _validate_unique_codex_history_turn_ids(turns)
    target_indexes = [
        index
        for index, turn_id in enumerate(turn_ids)
        if turn_id == receipt["turn_id"]
    ]
    if len(target_indexes) != 1:
        raise HarnessError(
            "Codex completed receipt turn is missing or ambiguous in thread history",
            kind="session_protocol",
        )
    target_index = target_indexes[0]
    if request_intent is not None:
        if (
            request_intent.get("thread_id") != expected_thread_id
            or request_intent.get("thread_policy_digest") != thread_policy_digest
            or request_intent.get("turn_count_before") != current_turn_count
            or request_intent.get("physical_turn_count_before") != target_index + 1
        ):
            raise HarnessError(
                "Codex waiting replay request-intent physical baseline drifted",
                kind="session_state",
            )
    target = turns[target_index]
    if _protocol_enum_value(_read_protocol_field(target, "status")) != "completed":
        raise HarnessError(
            "Codex completed receipt turn is not completed in thread history",
            kind="session_protocol",
        )
    completed_turns = _select_completed_thread_history_turns(turns)
    if len(completed_turns) != int(receipt["turn_count_before"]) + 1:
        raise HarnessError(
            "Codex completed receipt successful count does not match thread history",
            kind="session_state",
        )
    trailing = turns[target_index + 1 :]
    if not trailing:
        return turns, None
    trailing_ids = turn_ids[target_index + 1 :]
    trailing_statuses = [
        _protocol_enum_value(_read_protocol_field(turn, "status"))
        for turn in trailing
    ]
    if any(
        status not in {"failed", "interrupted", "aborted"}
        for status in trailing_statuses
    ):
        raise HarnessError(
            "Codex completed receipt has a non-terminal or unknown trailing turn",
            kind="recovery_pending",
        )
    candidate = {
        "schema_version": _TRAILING_ROLLBACK_SCHEMA_VERSION,
        "state": "intent",
        "thread_id": expected_thread_id,
        "receipt_turn_id": receipt["turn_id"],
        "request_digest": receipt["request_digest"],
        "turn_count_before": receipt["turn_count_before"],
        "physical_turn_count_before": len(turns),
        "trailing_turn_ids": trailing_ids,
        "preserved_turn_ids": turn_ids[: target_index + 1],
        "thread_policy_digest": thread_policy_digest,
    }
    return turns, candidate


def _rollback_trailing_codex_turns_after_receipt(
    root: Path,
    *,
    thread: Any,
    thread_id: str,
    receipt: Mapping[str, Any],
    thread_policy_digest: str,
    current_turn_count: int,
    request_intent: Mapping[str, Any] | None = None,
) -> None:
    """在重放 waiting receipt 前只回滚其后的失败尾部，并保存两阶段凭证。"""

    turns, candidate = _read_trailing_turn_plan(
        thread,
        expected_thread_id=thread_id,
        receipt=receipt,
        thread_policy_digest=thread_policy_digest,
        current_turn_count=current_turn_count,
        request_intent=request_intent,
    )
    existing_receipt = _read_trailing_rollback_record(root, receipt=True)
    if existing_receipt is not None:
        _validate_trailing_rollback_identity(
            existing_receipt,
            thread_id=thread_id,
            receipt=receipt,
            thread_policy_digest=thread_policy_digest,
        )
        current_ids = _validate_unique_codex_history_turn_ids(
            _read_thread_turns(thread, expected_thread_id=thread_id)
        )
        if current_ids != existing_receipt["preserved_turn_ids"]:
            raise HarnessError(
                "Codex trailing rollback receipt history does not match",
                kind="session_state",
            )
        return
    if candidate is None:
        existing_intent = _read_trailing_rollback_record(root, receipt=False)
        if existing_intent is not None:
            _validate_trailing_rollback_identity(
                existing_intent,
                thread_id=thread_id,
                receipt=receipt,
                thread_policy_digest=thread_policy_digest,
            )
            current_ids = _validate_unique_codex_history_turn_ids(turns)
            if current_ids != existing_intent["preserved_turn_ids"]:
                raise HarnessError(
                    "Codex trailing rollback intent history does not match",
                    kind="session_state",
                )
            _write_trailing_rollback_record(
                root,
                {**existing_intent, "state": "rolled_back"},
                receipt=True,
            )
        return
    existing_intent = _read_trailing_rollback_record(root, receipt=False)
    if existing_intent is not None:
        _validate_trailing_rollback_identity(
            existing_intent,
            thread_id=thread_id,
            receipt=receipt,
            thread_policy_digest=thread_policy_digest,
        )
        if existing_intent != candidate:
            raise HarnessError(
                "Codex trailing rollback intent history drifted",
                kind="session_state",
            )
        plan = existing_intent
    else:
        # 这是官方 thread/rollback 之前的唯一持久写入点；崩溃后下一次
        # resume 会沿同一 turn 列表继续，而不会猜测新的删除范围。
        plan = _write_trailing_rollback_record(root, candidate, receipt=False)
    client = getattr(thread, "_client", None)
    request_method = getattr(client, "request", None)
    if not callable(request_method):
        raise HarnessError(
            "Codex trailing rollback requires the official app-server client",
            kind="session_protocol",
        )
    try:
        from openai_codex.generated.v2_all import ThreadRollbackResponse

        response = request_method(
            "thread/rollback",
            {
                "threadId": thread_id,
                "numTurns": len(plan["trailing_turn_ids"]),
            },
            response_model=ThreadRollbackResponse,
        )
    except HarnessError:
        raise
    except BaseException:
        raise HarnessError(
            "Codex trailing thread rollback request failed",
            kind="recovery_pending",
        ) from None
    rolled_thread = _read_protocol_field(response, "thread")
    rolled_thread_id = _read_protocol_field(rolled_thread, "id")
    if rolled_thread_id != thread_id:
        raise HarnessError(
            "Codex trailing rollback returned a different thread",
            kind="session_protocol",
        )
    rolled_turns = _read_protocol_field(rolled_thread, "turns")
    if not isinstance(rolled_turns, list):
        raise HarnessError(
            "Codex trailing rollback response history is incomplete",
            kind="session_protocol",
        )
    rolled_ids = _validate_unique_codex_history_turn_ids(rolled_turns)
    if rolled_ids != plan["preserved_turn_ids"]:
        raise HarnessError(
            "Codex trailing rollback deleted a preserved turn",
            kind="session_protocol",
        )
    current_turns = _read_thread_turns(thread, expected_thread_id=thread_id)
    current_ids = _validate_unique_codex_history_turn_ids(current_turns)
    if current_ids != plan["preserved_turn_ids"]:
        raise HarnessError(
            "Codex trailing rollback did not leave the preserved history",
            kind="session_protocol",
        )
    _write_trailing_rollback_record(
        root,
        {**plan, "state": "rolled_back"},
        receipt=True,
    )


def _find_matching_completed_turn_receipt(
    root: Path | str,
    *,
    thread_id: str,
    request_digest: str,
    thread_policy_digest: str,
) -> dict[str, Any] | None:
    """按 thread 与完整 prompt 摘要寻找唯一可重放 receipt。"""

    validated_thread_id = _validate_receipt_identifier(thread_id, "thread_id")
    validated_digest = _validate_request_digest(request_digest)
    validated_policy_digest = _validate_thread_policy_digest(thread_policy_digest)
    matches: list[dict[str, Any]] = []
    for receipt in _load_completed_turn_receipts(root):
        if receipt["thread_id"] != validated_thread_id:
            raise HarnessError(
                "completed turn receipt thread_id drifted",
                kind="session_protocol",
            )
        if receipt["request_digest"] == validated_digest:
            if receipt["thread_policy_digest"] != validated_policy_digest:
                raise HarnessError(
                    "completed turn receipt thread policy drifted",
                    kind="session_state",
                )
            matches.append(receipt)
    if len(matches) > 1:
        raise HarnessError(
            "multiple completed turn receipts match the current request",
            kind="session_state",
        )
    return matches[0] if matches else None


def find_codex_completed_turn_receipt(
    root: Path | str,
    *,
    thread_id: str,
    request_digest: str,
    thread_policy_digest: str,
) -> dict[str, Any] | None:
    """读取与完整请求身份匹配的 completed-turn receipt，不启动模型。"""

    return _find_matching_completed_turn_receipt(
        root,
        thread_id=thread_id,
        request_digest=request_digest,
        thread_policy_digest=thread_policy_digest,
    )


def read_codex_completed_turn_receipt(
    root: Path | str,
    *,
    thread_id: str,
    turn_id: str,
    thread_policy_digest: str,
    read_only: bool = False,
) -> dict[str, Any] | None:
    """按 thread、turn 和策略摘要读取一个已完成 turn 回执。

    输入是当前会话的回执目录和等待 sidecar 提供的 thread/turn 身份；动作只读
    校验该目录中的全部回执，并返回唯一匹配项，不访问 provider。``read_only`` 为
    True 时还不创建目录或修正权限，供延迟句柄构造使用；默认值保持旧调用方的
    目录准备语义。回执缺失时返回 ``None``；发现损坏、跨 thread 或策略漂移时抛出
    ``HarnessError``，防止恢复逻辑把不属于当前会话的成功记录当成可重放依据。
    """

    validated_thread_id = _validate_receipt_identifier(thread_id, "thread_id")
    validated_turn_id = _validate_receipt_identifier(turn_id, "turn_id")
    validated_policy_digest = _validate_thread_policy_digest(thread_policy_digest)
    receipts = (
        _load_completed_turn_receipts_read_only(root)
        if read_only
        else _load_completed_turn_receipts(root)
    )
    for receipt in receipts:
        if receipt["thread_id"] != validated_thread_id:
            raise HarnessError(
                "completed turn receipt thread_id drifted",
                kind="session_protocol",
            )
    matches = tuple(
        receipt
        for receipt in receipts
        if receipt["turn_id"] == validated_turn_id
    )
    if not matches:
        return None
    if len(matches) != 1:
        raise HarnessError(
            "multiple completed turn receipts match the current turn",
            kind="session_state",
        )
    receipt = matches[0]
    if receipt["thread_policy_digest"] != validated_policy_digest:
        raise HarnessError(
            "completed turn receipt thread policy drifted",
            kind="session_state",
        )
    return receipt


def _write_completed_turn_receipt(
    root: Path | str,
    *,
    thread_id: str,
    turn_id: str,
    request_digest: str,
    turn_count_before: int,
    thread_policy_digest: str,
) -> Path:
    """在 turn/completed 到达时原子保存 0600 receipt，正文留在 Codex history。"""

    directory = _prepare_receipt_directory(root)
    payload = {
        "schema_version": _RECEIPT_SCHEMA_VERSION,
        "thread_id": _validate_receipt_identifier(thread_id, "thread_id"),
        "turn_id": _validate_receipt_identifier(turn_id, "turn_id"),
        "request_digest": _validate_request_digest(request_digest),
        "turn_count_before": turn_count_before,
        "thread_policy_digest": _validate_thread_policy_digest(thread_policy_digest),
    }
    if isinstance(turn_count_before, bool) or not isinstance(turn_count_before, int) or turn_count_before < 0:
        raise HarnessError(
            "completed turn receipt turn_count_before is invalid",
            kind="session_state",
        )
    target = directory / f"{payload['turn_id']}.json"
    if target.exists() or target.is_symlink():
        if target.is_symlink():
            raise HarnessError("completed turn receipt must not be a symlink", kind="session_state")
        existing = _read_completed_turn_receipt(target)
        if existing != payload:
            raise HarnessError("completed turn receipt conflicts with an existing turn", kind="session_state")
        return target

    _atomic_write_json_file(target, payload, description="completed turn receipt")
    return target


def acknowledge_codex_completed_turn_receipt(
    root: Path | str,
    *,
    thread_id: str,
    turn_id: str,
    request_digest: str,
    turn_count_before: int,
    thread_policy_digest: str,
) -> None:
    """在上层已保存 manifest 后确认并删除同一 completed-turn receipt。"""

    validate_codex_completed_turn_receipt(
        root,
        thread_id=thread_id,
        turn_id=turn_id,
        request_digest=request_digest,
        turn_count_before=turn_count_before,
        thread_policy_digest=thread_policy_digest,
    )
    directory = _prepare_receipt_directory(root)
    target = directory / f"{_validate_receipt_identifier(turn_id, 'turn_id')}.json"

    _remove_file_and_fsync_directory(target, description="completed turn receipt")
    _remove_completion_pending(root)
    _remove_request_intent(root)


def validate_codex_completed_turn_receipt(
    root: Path | str,
    *,
    thread_id: str,
    turn_id: str,
    request_digest: str,
    turn_count_before: int,
    thread_policy_digest: str,
) -> None:
    """在更新会话清单前验证 completed-turn receipt 存在且身份完全匹配。

    输入是 receipt 目录和当前 thread、turn、请求摘要、旧计数及策略摘要；动作是
    只读校验 receipt，成功返回 ``None``，失败抛出状态错误。函数不删除 receipt，
    让更新清单前后的崩溃仍可由下一次调用重放。
    """

    directory = _prepare_receipt_directory(root)
    validated_turn_id = _validate_receipt_identifier(turn_id, "turn_id")
    validated_thread_id = _validate_receipt_identifier(thread_id, "thread_id")
    validated_request_digest = _validate_request_digest(request_digest)
    validated_policy_digest = _validate_thread_policy_digest(thread_policy_digest)
    if (
        isinstance(turn_count_before, bool)
        or not isinstance(turn_count_before, int)
        or turn_count_before < 0
    ):
        raise HarnessError(
            "completed turn receipt turn_count_before is invalid",
            kind="session_state",
        )
    target = directory / f"{validated_turn_id}.json"
    if not target.exists() and not target.is_symlink():
        raise HarnessError(
            "completed turn receipt is missing for acknowledgement",
            kind="session_state",
        )
    actual = _read_completed_turn_receipt(target)
    expected = {
        "schema_version": _RECEIPT_SCHEMA_VERSION,
        "thread_id": validated_thread_id,
        "turn_id": validated_turn_id,
        "request_digest": validated_request_digest,
        "turn_count_before": turn_count_before,
        "thread_policy_digest": validated_policy_digest,
    }
    if actual != expected:
        raise HarnessError(
            "completed turn receipt identity does not match acknowledgement",
            kind="session_state",
        )


def build_codex_completed_turn_receipt_handle(
    root: Path | str,
    *,
    owner: Mapping[str, Any] | CodexSessionOwner,
    session_store: CodexSessionStore,
    thread_id: str,
    turn_id: str,
    request_digest: str | None,
    turn_count_before: int | None,
    thread_policy_digest: str | None,
    handoff_kind: str,
) -> CodexCompletedTurnReceiptHandle | None:
    """读取并绑定一个现存 receipt，返回延迟确认句柄。

    输入必须包含调用方归属、thread/turn、请求摘要、策略摘要、计数以及外层事实
    类型；动作只读核对 receipt，不删除文件或修改 recovery。必须提供真实的
    ``CodexSessionStore``，由它推导规范 receipt 目录并核对 owner；句柄会拒绝任意
    路径替换。receipt 缺失、重复、策略漂移或任一字段不一致都会抛出状态错误，
    调用方不能只凭一个现存目录伪造可确认句柄。
    """

    normalized_owner = _normalize_receipt_owner(owner)
    if not isinstance(session_store, CodexSessionStore):
        raise HarnessError(
            "completed turn receipt session store is invalid",
            kind="session_state",
        )
    if normalized_owner != session_store.owner.identity:
        raise HarnessError(
            "completed turn receipt owner does not match the session store",
            kind="session_state",
        )
    canonical_receipt_root = (
        session_store.manifest_path.parent / "completed_turn_receipts"
    ).resolve()
    try:
        actual_root = Path(root).expanduser().resolve()
    except (TypeError, ValueError) as exc:
        raise HarnessError(
            "completed turn receipt root is invalid",
            kind="session_state",
        ) from exc
    if actual_root != canonical_receipt_root:
        raise HarnessError(
            "completed turn receipt root is not canonical for the session",
            kind="session_state",
        )

    validated_thread_id = _validate_receipt_identifier(thread_id, "thread_id")
    validated_turn_id = _validate_receipt_identifier(turn_id, "turn_id")
    validated_request_digest = _validate_request_digest(request_digest)
    validated_policy_digest = _validate_thread_policy_digest(thread_policy_digest)
    receipts = _load_completed_turn_receipts_read_only(root)
    for candidate in receipts:
        if candidate["thread_id"] != validated_thread_id:
            raise HarnessError(
                "completed turn receipt thread_id drifted",
                kind="session_protocol",
            )
    matches = tuple(
        candidate for candidate in receipts if candidate["turn_id"] == validated_turn_id
    )
    if len(matches) > 1:
        raise HarnessError(
            "multiple completed turn receipts match the current turn",
            kind="session_state",
        )
    receipt = matches[0] if matches else None
    if receipt is None:
        raise HarnessError(
            "completed turn receipt is missing for deferred acknowledgement",
            kind="session_state",
        )
    if (
        receipt["request_digest"] != validated_request_digest
        or receipt["turn_count_before"] != turn_count_before
        or receipt["thread_policy_digest"] != validated_policy_digest
    ):
        raise HarnessError(
            "completed turn receipt identity does not match deferred acknowledgement",
            kind="session_state",
        )
    return CodexCompletedTurnReceiptHandle(
        receipt_root=root,
        owner=normalized_owner,
        thread_id=receipt["thread_id"],
        turn_id=receipt["turn_id"],
        request_digest=receipt["request_digest"],
        turn_count_before=receipt["turn_count_before"],
        thread_policy_digest=receipt["thread_policy_digest"],
        handoff_kind=handoff_kind,
        _factory_token=_RECEIPT_HANDLE_FACTORY_TOKEN,
    )


def _assert_codex_session_manifest_matches_completed_receipt(
    session_store: CodexSessionStore,
    *,
    owner: Mapping[str, str],
    thread_id: str,
    thread_policy_digest: str,
    turn_count_before: int,
) -> None:
    """核对已完成 manifest；manifest 没有 turn_id，因此不验证具体 turn。"""

    manifest = session_store.load()
    if manifest is None:
        raise HarnessError(
            "completed receipt acknowledgement requires an existing session manifest",
            kind="session_state",
        )
    expected_count = turn_count_before + 1
    if (
        manifest.get("owner") != dict(owner)
        or manifest.get("thread_id") != thread_id
        or manifest.get("thread_policy_digest") != thread_policy_digest
        or manifest.get("turn_count") != expected_count
        or manifest.get("status") != "completed"
    ):
        raise HarnessError(
            "completed receipt acknowledgement manifest identity does not match",
            kind="session_state",
        )


def _assert_codex_session_recovery_journals_absent(
    session_store: CodexSessionStore,
) -> None:
    """确认 SessionStore 的恢复、回滚和验证凭证均已清理。"""

    paths = (
        session_store.recovery_path,
        session_store.recovery_poison_path,
        session_store.validated_turn_path,
        session_store.thread_rollback_intent_path,
        session_store.thread_rollback_receipt_path,
    )
    parent = session_store.manifest_path.parent
    try:
        children = tuple(parent.iterdir())
    except OSError as exc:
        raise HarnessError(
            "Codex session recovery journal directory cannot be listed",
            kind="session_state",
        ) from exc
    for path in paths:
        if path.is_symlink() or path.exists():
            raise HarnessError(
                f"Codex session recovery journal remains: {path.name}",
                kind="recovery_pending",
            )
        if any(child.name.startswith(f"{path.name}.") for child in children):
            raise HarnessError(
                f"Codex session recovery journal has an incomplete entry: {path.name}",
                kind="recovery_pending",
            )


def resolve_codex_completed_turn_receipt_handle(
    root: Path | str,
    *,
    owner: Mapping[str, Any] | CodexSessionOwner,
    session_store: CodexSessionStore,
    thread_id: str,
    turn_id: str,
    request_digest: str,
    turn_count_before: int,
    thread_policy_digest: str,
    handoff_kind: str,
) -> CodexCompletedTurnReceiptHandle:
    """恢复一个已完成 Result 的 receipt 句柄并支持跨进程幂等。

    receipt 仍存在时只读返回严格句柄；receipt 已被前一进程删除时，仅在同一
    canonical SessionStore manifest 已精确完成、且 completion-pending、request-intent、
    terminal-turn、尾部回滚和 SessionStore 恢复 journal 全部不存在时返回
    ``already_acknowledged`` no-op 句柄。``turn_id`` 不在 manifest 中保存，所以
    缺少 receipt 时，本函数不声称验证了该 turn；调用方必须先用已持久化 Result 和
    sidecar record digest 完成外层核对。submit callback 早于 turn 完成时，sidecar
    还没有 provider checkpoint；这时三个 checkpoint 字段必须全部传为 ``None``，
    本函数只允许 Result handoff，并从精确 thread/turn receipt 读取完整字段。若收据
    已确认删除，则完成相同恢复检查后返回 ``None``，表示无需再次确认。三个字段只缺
    一部分时停止恢复。该接口只读，不创建目录，不启动 provider；waiting handoff
    不允许省略 checkpoint。
    """

    normalized_owner = _normalize_receipt_owner(owner)
    if not isinstance(session_store, CodexSessionStore):
        raise HarnessError(
            "completed turn receipt session store is invalid",
            kind="session_state",
        )
    if normalized_owner != session_store.owner.identity:
        raise HarnessError(
            "completed turn receipt owner does not match the session store",
            kind="session_state",
        )
    canonical_receipt_root = (
        session_store.manifest_path.parent / "completed_turn_receipts"
    ).resolve()
    try:
        actual_root = Path(root).expanduser().resolve()
    except (TypeError, ValueError) as exc:
        raise HarnessError(
            "completed turn receipt root is invalid",
            kind="session_state",
        ) from exc
    if actual_root != canonical_receipt_root:
        raise HarnessError(
            "completed turn receipt root is not canonical for the session",
            kind="session_state",
        )

    validated_thread_id = _validate_receipt_identifier(thread_id, "thread_id")
    validated_turn_id = _validate_receipt_identifier(turn_id, "turn_id")
    checkpoint_fields = (
        request_digest,
        turn_count_before,
        thread_policy_digest,
    )
    has_any_checkpoint = any(value is not None for value in checkpoint_fields)
    has_complete_checkpoint = all(value is not None for value in checkpoint_fields)
    if has_any_checkpoint != has_complete_checkpoint:
        raise HarnessError(
            "completed turn receipt recovery checkpoint is incomplete",
            kind="session_state",
        )
    if not has_complete_checkpoint and handoff_kind != "result":
        raise HarnessError(
            "completed turn receipt checkpoint is required for non-result handoff",
            kind="session_state",
        )
    validated_request_digest = (
        _validate_request_digest(request_digest)
        if has_complete_checkpoint
        else None
    )
    validated_policy_digest = (
        _validate_thread_policy_digest(thread_policy_digest)
        if has_complete_checkpoint
        else None
    )
    if has_complete_checkpoint and (
        isinstance(turn_count_before, bool)
        or not isinstance(turn_count_before, int)
        or turn_count_before < 0
    ):
        raise HarnessError(
            "completed turn receipt turn_count_before is invalid",
            kind="session_state",
        )

    receipts = _load_completed_turn_receipts_read_only(root)
    for receipt in receipts:
        if receipt["thread_id"] != validated_thread_id:
            raise HarnessError(
                "completed turn receipt thread_id drifted",
                kind="session_protocol",
            )
    matches = tuple(
        receipt for receipt in receipts if receipt["turn_id"] == validated_turn_id
    )
    if len(matches) > 1:
        raise HarnessError(
            "multiple completed turn receipts match the current turn",
            kind="session_state",
        )
    if matches:
        receipt = matches[0]
        if not has_complete_checkpoint:
            validated_request_digest = receipt["request_digest"]
            validated_policy_digest = receipt["thread_policy_digest"]
            turn_count_before = receipt["turn_count_before"]
        assert validated_request_digest is not None
        assert validated_policy_digest is not None
        assert turn_count_before is not None
        return build_codex_completed_turn_receipt_handle(
            root,
            owner=normalized_owner,
            session_store=session_store,
            thread_id=validated_thread_id,
            turn_id=validated_turn_id,
            request_digest=validated_request_digest,
            turn_count_before=turn_count_before,
            thread_policy_digest=validated_policy_digest,
            handoff_kind=handoff_kind,
        )
    if receipts:
        raise HarnessError(
            "completed turn receipt for another turn remains",
            kind="recovery_pending",
        )
    if handoff_kind != "result":
        raise HarnessError(
            "completed turn receipt is missing for non-result handoff",
            kind="session_state",
        )

    _assert_completed_receipt_recovery_journals_absent(root)
    if not has_complete_checkpoint:
        manifest = session_store.load()
        if manifest is None:
            raise HarnessError(
                "completed receipt acknowledgement requires an existing session manifest",
                kind="session_state",
            )
        manifest_policy_digest = manifest.get("thread_policy_digest")
        manifest_turn_count = manifest.get("turn_count")
        if (
            manifest.get("owner") != normalized_owner
            or manifest.get("thread_id") != validated_thread_id
            or manifest.get("status") != "completed"
            or not isinstance(manifest_policy_digest, str)
            or _RECEIPT_DIGEST.fullmatch(manifest_policy_digest) is None
            or isinstance(manifest_turn_count, bool)
            or not isinstance(manifest_turn_count, int)
            or manifest_turn_count < 1
        ):
            raise HarnessError(
                "completed receipt acknowledgement manifest identity does not match",
                kind="session_state",
            )
        _assert_codex_session_recovery_journals_absent(session_store)
        return None
    assert validated_request_digest is not None
    assert validated_policy_digest is not None
    assert turn_count_before is not None
    _assert_codex_session_manifest_matches_completed_receipt(
        session_store,
        owner=normalized_owner,
        thread_id=validated_thread_id,
        thread_policy_digest=validated_policy_digest,
        turn_count_before=turn_count_before,
    )
    _assert_codex_session_recovery_journals_absent(session_store)
    return CodexCompletedTurnReceiptHandle.from_already_acknowledged(
        root,
        owner=normalized_owner,
        thread_id=validated_thread_id,
        turn_id=validated_turn_id,
        request_digest=validated_request_digest,
        turn_count_before=turn_count_before,
        thread_policy_digest=validated_policy_digest,
        handoff_kind="result",
        _factory_token=_RECEIPT_HANDLE_FACTORY_TOKEN,
    )


def _unwrap_protocol_root(value: Any) -> Any:
    """读取 RootModel 的 root，同时兼容字典对象。"""

    root = _read_protocol_field(value, "root")
    return root if root is not None else value


def _protocol_item_type(value: Any) -> Any:
    """读取 thread history item 的 type 字段。"""

    return _protocol_enum_value(_read_protocol_field(_unwrap_protocol_root(value), "type"))


def _protocol_item_id(value: Any) -> str | None:
    """读取工具 item 的稳定标识；无法读取时返回 None。"""

    candidate = _read_protocol_field(_unwrap_protocol_root(value), "id")
    if isinstance(candidate, str) and candidate.strip():
        return candidate.strip()
    return None


def _protocol_item_process_id(value: Any) -> str | None:
    """读取 commandExecution 关联的进程标识或句柄；没有时返回 None。"""

    item_root = _unwrap_protocol_root(value)
    for field in ("process_id", "process_handle"):
        candidate = _read_protocol_field(item_root, field)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _protocol_process_identity(value: Any) -> str | None:
    """读取 process 通知中的 id 或 handle，统一成可追踪字符串。"""

    for field in ("process_id", "process_handle"):
        candidate = _read_protocol_field(value, field)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _extract_persisted_user_prompt(item: Any) -> str | None:
    """从 persisted userMessage 读取唯一的 text 输入，不能还原时返回 None。"""

    item_root = _unwrap_protocol_root(item)
    if _protocol_item_type(item_root) != "userMessage":
        return None
    content = _read_protocol_field(item_root, "content")
    if not isinstance(content, list) or len(content) != 1:
        return None
    input_root = _unwrap_protocol_root(content[0])
    if _protocol_item_type(input_root) != "text":
        return None
    text = _read_protocol_field(input_root, "text")
    return text if isinstance(text, str) else None


def _extract_persisted_final_answer(item: Any) -> str | None:
    """从 persisted agentMessage 读取 final_answer 文本。"""

    item_root = _unwrap_protocol_root(item)
    if _protocol_item_type(item_root) != "agentMessage":
        return None
    phase = _protocol_enum_value(_read_protocol_field(item_root, "phase"))
    text = _read_protocol_field(item_root, "text")
    if phase != "final_answer" or not isinstance(text, str) or not text:
        return None
    return text


def _select_replayed_completed_turn(
    history_turns: Sequence[Any],
    *,
    receipt: Mapping[str, Any],
) -> Any:
    """从完整 history 中选择 receipt 指向的唯一已完成 turn。"""

    if not history_turns:
        raise HarnessError(
            "Codex thread history has no complete turns for replay",
            kind="session_protocol",
        )
    last_turn = history_turns[-1]
    if _read_protocol_field(last_turn, "id") != receipt["turn_id"]:
        raise HarnessError(
            "Codex thread history last turn does not match completed receipt",
            kind="session_protocol",
        )
    matching_turns = [
        turn for turn in history_turns if _read_protocol_field(turn, "id") == receipt["turn_id"]
    ]
    if len(matching_turns) != 1:
        raise HarnessError("Codex thread history has an ambiguous completed turn", kind="session_protocol")
    turn = matching_turns[0]
    if _protocol_enum_value(_read_protocol_field(turn, "status")) != "completed":
        raise HarnessError("Codex receipt turn is not completed", kind="session_protocol")
    items_view = _protocol_enum_value(_read_protocol_field(turn, "items_view"))
    if items_view is not None and items_view != "full":
        raise HarnessError(
            "Codex receipt turn items are incomplete and cannot be replayed",
            kind="session_protocol",
        )
    return turn


def _extract_replayed_turn_messages(items: Any, *, prompt_text: str | None) -> str:
    """校验 replay turn 的 persisted user prompt 和唯一 final_answer。

    ``prompt_text`` 为 None 只表示上层已用 waiting turn、receipt、manifest 和
    thread history 绑定了目标回合；此时仍要求 history 保存唯一 user prompt，但
    不把恢复时重建的当前 prompt 当作旧回合正文的身份依据。
    """

    if not isinstance(items, list) or not items:
        raise HarnessError("Codex receipt turn has no persisted items", kind="session_protocol")
    user_prompts = [
        prompt
        for item in items
        for prompt in (_extract_persisted_user_prompt(item),)
        if prompt is not None
    ]
    if not user_prompts:
        raise HarnessError(
            "Codex receipt turn has no persisted user prompt",
            kind="session_protocol",
        )
    if prompt_text is not None and user_prompts != [prompt_text]:
        raise HarnessError(
            "Codex receipt user prompt does not match the current request",
            kind="session_protocol",
        )
    final_answers = [
        answer
        for item in items
        for answer in (_extract_persisted_final_answer(item),)
        if answer is not None
    ]
    if len(final_answers) != 1:
        raise HarnessError(
            "Codex receipt does not contain one complete final_answer message",
            kind="session_protocol",
        )
    return final_answers[0]


def _extract_replayed_turn_content(
    thread: Any,
    *,
    expected_thread_id: str,
    receipt: Mapping[str, Any],
    prompt_text: str | None,
    turns: Sequence[Any] | None = None,
) -> str:
    """读取并严格核对最后一个 completed turn，失败时绝不猜测正文。"""

    content, _ = _extract_replayed_turn_payload(
        thread,
        expected_thread_id=expected_thread_id,
        receipt=receipt,
        prompt_text=prompt_text,
        turns=turns,
    )
    return content


def _extract_replayed_turn_payload(
    thread: Any,
    *,
    expected_thread_id: str,
    receipt: Mapping[str, Any],
    prompt_text: str | None,
    turns: Sequence[Any] | None = None,
) -> tuple[str, tuple[Any, ...]]:
    """读取 replay turn 的正文及完整 MCP 工具回执，供恢复 tracker 使用。"""

    history_turns = list(turns) if turns is not None else _read_thread_turns(
        thread,
        expected_thread_id=expected_thread_id,
    )
    turn = _select_replayed_completed_turn(history_turns, receipt=receipt)
    items = _read_protocol_field(turn, "items")
    content = _extract_replayed_turn_messages(items, prompt_text=prompt_text)
    completed_tool_items = tuple(
        item for item in items if _protocol_item_type(item) == "mcpToolCall"
    )
    return content, completed_tool_items


def _read_thread_turns(thread: Any, *, expected_thread_id: str) -> list[Any]:
    """读取完整 thread history，校验 thread 身份后返回 turns。"""

    read_method = getattr(thread, "read", None)
    if not callable(read_method):
        raise HarnessError(
            "Codex thread history cannot safely replay a completed turn",
            kind="session_protocol",
        )
    history_error: HarnessError | None = None
    try:
        response = read_method(include_turns=True)
    except BaseException:
        # SDK 异常可能包含 provider secret；在 except 外重新抛出全新的错误，
        # 同时清除 cause/context，避免 traceback、异常链或 repr 泄露原文。
        history_error = HarnessError(
            "Codex thread history read failed before replay",
            kind="transport",
        )
    if history_error is not None:
        raise history_error from None
    read_thread = _read_protocol_field(response, "thread")
    read_thread_id = _read_protocol_field(read_thread, "id")
    if read_thread_id != expected_thread_id:
        raise HarnessError("Codex thread history returned a different thread_id", kind="session_protocol")
    turns = _read_protocol_field(read_thread, "turns")
    if not isinstance(turns, list):
        raise HarnessError(
            "Codex thread history turns are not complete",
            kind="session_protocol",
        )
    return turns


def _validate_unique_codex_history_turn_ids(turns: Sequence[Any]) -> list[str]:
    """校验完整 history 的 turn 身份并返回规范化后的唯一 ID 列表。

    输入是同一 thread 的物理 turn 顺序；动作拒绝缺失、非法、重复或仅因空白等
    规范化造成歧义的 ID，返回与 history 顺序一致的合法 ID。调用方在写入 receipt
    或 terminal journal 前必须先完成这项检查，失败时抛出协议错误且不猜测 turn。
    """

    validated_ids: list[str] = []
    for turn in turns:
        try:
            validated_ids.append(
                _validate_receipt_identifier(
                    _read_protocol_field(turn, "id"),
                    "turn_id",
                )
            )
        except HarnessError:
            raise HarnessError(
                "Codex thread history contains a turn with invalid identity",
                kind="session_protocol",
            ) from None
    if len(set(validated_ids)) != len(validated_ids):
        raise HarnessError(
            "Codex thread history contains duplicate turn identity",
            kind="session_protocol",
        )
    return validated_ids


def _select_completed_thread_history_turns(turns: Sequence[Any]) -> list[Any]:
    """返回 history 中的 completed turn，并拒绝未终结或未知状态。

    输入是同一 thread 的完整物理 turn 顺序；动作允许 failed/interrupted/aborted
    终态留在历史中，只选择 ``completed`` 项。返回的新列表不修改 history；出现
    in-progress、缺失或未知状态时抛出 recovery_pending，调用方不得继续确认或
    重放。
    """

    completed_turns: list[Any] = []
    for turn in turns:
        status = _protocol_enum_value(_read_protocol_field(turn, "status"))
        if status == "completed":
            completed_turns.append(turn)
            continue
        if status not in {"failed", "interrupted", "aborted"}:
            raise HarnessError(
                "Codex thread history has an unfinished turn before a new request",
                kind="recovery_pending",
            )
    return completed_turns


def rollback_codex_thread_for_investigation_turn(
    *,
    thread: Any,
    thread_id: str,
    successful_turn_count: int,
    expected_physical_turn_count: int | None = None,
    before_rollback: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """用官方 ``thread/rollback`` 清空仅含失败终态的 Investigation turn 历史。"""

    normalized_thread_id = _validate_receipt_identifier(thread_id, "thread_id")
    if isinstance(successful_turn_count, bool) or not isinstance(successful_turn_count, int):
        raise HarnessError(
            "Investigation turn rollback successful turn count is invalid",
            kind="session_protocol",
        )
    if successful_turn_count != 0:
        raise HarnessError(
            "Investigation turn rollback requires zero successful turns",
            kind="session_state",
        )
    actual_thread_id = _validate_receipt_identifier(
        getattr(thread, "id", None),
        "thread_id",
    )
    if actual_thread_id != normalized_thread_id:
        raise HarnessError(
            "Investigation turn rollback thread identity does not match",
            kind="session_protocol",
        )
    turns = _read_thread_turns(thread, expected_thread_id=normalized_thread_id)
    if not turns:
        if (
            isinstance(expected_physical_turn_count, bool)
            or not isinstance(expected_physical_turn_count, int)
            or expected_physical_turn_count < 1
        ):
            raise HarnessError(
                "Investigation turn rollback requires physical turns",
                kind="session_state",
            )
        return {
            "thread_id": normalized_thread_id,
            "num_turns": expected_physical_turn_count,
            "already_rolled_back": True,
        }
    _validate_unique_codex_history_turn_ids(turns)
    for turn in turns:
        status = _protocol_enum_value(_read_protocol_field(turn, "status"))
        if status not in {"failed", "interrupted"}:
            raise HarnessError(
                "Investigation turn rollback refuses completed or unfinished history",
                kind="session_state",
            )
    num_turns = len(turns)
    if (
        expected_physical_turn_count is not None
        and expected_physical_turn_count != num_turns
    ):
        raise HarnessError(
            "Investigation turn rollback physical turn count drifted",
            kind="session_state",
        )
    if before_rollback is not None:
        try:
            before_rollback(num_turns)
        except BaseException:
            raise HarnessError(
                "Investigation turn rollback intent cannot be saved",
                kind="recovery_pending",
            ) from None
    client = getattr(thread, "_client", None)
    request_method = getattr(client, "request", None)
    if not callable(request_method):
        raise HarnessError(
            "Codex thread rollback requires the official app-server client",
            kind="session_protocol",
        )
    try:
        from openai_codex.generated.v2_all import ThreadRollbackResponse

        response = request_method(
            "thread/rollback",
            {"threadId": normalized_thread_id, "numTurns": num_turns},
            response_model=ThreadRollbackResponse,
        )
    except HarnessError:
        raise
    except BaseException:
        raise HarnessError(
            "Investigation turn thread rollback request failed",
            kind="transport",
        ) from None
    rolled_thread = _read_protocol_field(response, "thread")
    rolled_thread_id = _read_protocol_field(rolled_thread, "id")
    if rolled_thread_id != normalized_thread_id:
        raise HarnessError(
            "Investigation turn rollback returned a different thread",
            kind="session_protocol",
        )
    rolled_turns = _read_protocol_field(rolled_thread, "turns")
    if not isinstance(rolled_turns, list) or rolled_turns:
        raise HarnessError(
            "Investigation turn rollback did not clear physical history",
            kind="session_protocol",
        )
    return {
        "thread_id": normalized_thread_id,
        "num_turns": num_turns,
        "already_rolled_back": False,
    }


# 旧名称仍接受已经持久化的 Investigation wake 调用方；新代码按 turn 语义调用。
rollback_codex_thread_for_investigation_wake = rollback_codex_thread_for_investigation_turn


def confirm_codex_request_intent_without_new_turn(
    completed_turn_receipt_root: Path,
    *,
    thread: Any,
    thread_id: str,
    request_digest: str,
    turn_count_before: int,
    provider_id: str,
    env_key: str,
    model: str,
    role: str,
    stage: str,
    profile: str,
    workspace: Path | str,
    thread_policy_digest: str,
) -> None:
    """确认原 thread 没有新增 turn 后删除 request-intent。

    输入是 receipt 目录、可读取 history 的原 thread 以及完整请求和配置身份；动作是
    逐字段核对 intent，再确认完整 history 的物理 turn 数仍精确等于 intent 基线，
    且其中成功数仍等于 ``turn_count_before``，最后原子删除 intent。更早的
    failed/interrupted 可包含在基线内；任意新增终态、未终结或未知状态都失败关闭。
    副作用只有删除该阻断文件，不提交模型 turn；旧 v2 intent 保留原摘要并要求
    显式迁移，不能通过此入口猜测物理基线。
    """

    intent = _read_request_intent(completed_turn_receipt_root)
    if intent is None:
        raise HarnessError("request intent is missing", kind="session_state")
    if intent["schema_version"] != _REQUEST_INTENT_SCHEMA_VERSION:
        raise CodexRequestIntentRecoveryPending(
            "request intent requires explicit schema migration",
            turn_id=None,
            reason="request_intent",
            request_digest=intent["request_digest"],
        )
    physical_turn_count_before = intent["physical_turn_count_before"]
    expected = {
        "schema_version": _REQUEST_INTENT_SCHEMA_VERSION,
        "thread_id": _validate_receipt_identifier(thread_id, "thread_id"),
        "request_digest": _validate_request_digest(request_digest),
        "turn_count_before": turn_count_before,
        "physical_turn_count_before": physical_turn_count_before,
        "provider_id": _validate_intent_text(provider_id, "provider_id"),
        "env_key": _validate_intent_text(env_key, "env_key"),
        "model": _validate_intent_text(model, "model"),
        "role": _validate_intent_text(role, "role"),
        "stage": _validate_intent_text(stage, "stage"),
        "profile": _validate_intent_text(profile, "profile"),
        "workspace": _validate_intent_text(str(Path(workspace).resolve()), "workspace"),
        "thread_policy_digest": _validate_thread_policy_digest(thread_policy_digest),
    }
    if (
        isinstance(turn_count_before, bool)
        or not isinstance(turn_count_before, int)
        or turn_count_before < 0
    ):
        raise HarnessError("request intent turn_count_before is invalid", kind="session_state")
    if intent != expected:
        raise HarnessError("request intent identity does not match confirmation", kind="session_state")
    turns = _read_thread_turns(thread, expected_thread_id=thread_id)
    completed_turns = _select_completed_thread_history_turns(turns)
    if len(turns) != physical_turn_count_before:
        raise HarnessError(
            "request intent cannot be confirmed because physical history has a new turn",
            kind="session_state",
        )
    if len(completed_turns) != turn_count_before:
        raise HarnessError(
            "request intent successful history does not match confirmation",
            kind="session_state",
        )
    _remove_request_intent(completed_turn_receipt_root)


def _select_unique_codex_request_intent_migration_terminal_turn(
    turns: Sequence[Any],
    *,
    baseline: int,
    expected_completed_turn_count: int,
    expected_terminal_status: str | None,
) -> tuple[str, str]:
    """选择并核对 request-intent 物理基线后的唯一终态 turn。

    输入是同一 thread 的完整 history、独立证明的物理基线、旧 intent 的成功
    turn 计数和可选终态。动作先确认基线后恰好只有一个新增 turn，再核对全部
    history 的身份和状态；返回唯一新增 turn 的 id/status。此前的
    ``failed``/``interrupted`` 可以作为显式基线中的既有事实，但不能被忽略，
    未完成或未知状态都会 fail-closed。
    """

    if len(turns) <= baseline:
        raise HarnessError(
            "request intent migration found no new terminal turn",
            kind="recovery_pending",
        )
    baseline_turns = turns[:baseline]
    new_turns = turns[baseline:]
    if len(new_turns) != 1:
        raise HarnessError(
            "request intent migration requires one physical turn after the baseline",
            kind="recovery_pending",
        )
    completed_baseline = _select_completed_thread_history_turns(baseline_turns)
    if len(completed_baseline) != expected_completed_turn_count:
        raise HarnessError(
            "request intent migration successful baseline does not match intent",
            kind="recovery_pending",
        )
    normalized_turn_ids = _validate_unique_codex_history_turn_ids(turns)
    candidate = new_turns[0]
    candidate_id = normalized_turn_ids[baseline]
    candidate_status = _protocol_enum_value(_read_protocol_field(candidate, "status"))
    if candidate_status not in {"failed", "interrupted"} or (
        expected_terminal_status is not None
        and candidate_status != expected_terminal_status
    ):
        raise HarnessError(
            "request intent migration history terminal status does not match",
            kind="recovery_pending",
        )
    # 对显式基线也检查全部 history 的状态，避免把未知/未完成 turn 当成
    # 已经提交的旧边界；failed/interrupted 只能作为明确的 terminal 事实。
    _select_completed_thread_history_turns(turns)
    return candidate_id, str(candidate_status)


def _assemble_codex_request_intent_migration_result(
    completed_turn_receipt_root: Path,
    *,
    migrated_intent: Mapping[str, Any],
    terminal_journal: Mapping[str, Any],
) -> dict[str, Any]:
    """返回迁移后的固定身份，避免把 provider 正文带出恢复边界。"""

    return {
        "request_intent_path": str(_request_intent_path(completed_turn_receipt_root)),
        "terminal_journal_path": str(_terminal_turn_journal_path(completed_turn_receipt_root)),
        "request_digest": migrated_intent["request_digest"],
        "thread_id": migrated_intent["thread_id"],
        "turn_id": terminal_journal["turn_id"],
        "terminal_status": terminal_journal["status"],
        "turn_count_before": migrated_intent["turn_count_before"],
        "physical_turn_count_before": migrated_intent["physical_turn_count_before"],
        "thread_policy_digest": migrated_intent["thread_policy_digest"],
    }


def _repair_v3_codex_request_intent_migration(
    completed_turn_receipt_root: Path,
    *,
    thread: Any,
    validated_thread_id: str,
    validated_request_digest: str,
    intent: Mapping[str, Any],
    physical_turn_count_before: int | None,
    terminal_status: str | None,
) -> dict[str, Any]:
    """修复已有 v3 intent 的 terminal journal 并核对同一物理终态。

    输入是已经核对请求摘要和 thread 身份的 v3 intent、同一 thread 及可选的
    物理基线/终态。动作在 journal 缺失时按唯一 history 证据补写 journal，随后
    核对 journal 与 history 的 turn 身份和状态；不会创建新 turn 或修改 v3 intent。
    任何基线、journal、history 不一致都 fail-closed。
    """

    baseline = intent["physical_turn_count_before"]
    if (
        physical_turn_count_before is not None
        and physical_turn_count_before != baseline
    ):
        raise HarnessError(
            "v3 request intent migration physical baseline conflicts with intent",
            kind="session_state",
        )
    journal = _read_terminal_turn_journal(completed_turn_receipt_root)
    if journal is None:
        # 旧草稿可能已经完成 v3 intent 替换但尚未写 journal；只在同一
        # history 能唯一证明终态时补写，不能把普通 completed turn 猜作失败。
        turns = _read_thread_turns(thread, expected_thread_id=validated_thread_id)
        candidate_id, candidate_status = _select_unique_codex_request_intent_migration_terminal_turn(
            turns,
            baseline=baseline,
            expected_completed_turn_count=intent["turn_count_before"],
            expected_terminal_status=terminal_status,
        )
        _write_terminal_turn_journal(
            completed_turn_receipt_root,
            thread_id=validated_thread_id,
            turn_id=candidate_id,
            status=candidate_status,
            request_digest=validated_request_digest,
            turn_count_before=intent["turn_count_before"],
            thread_policy_digest=intent["thread_policy_digest"],
        )
        journal = _read_terminal_turn_journal(completed_turn_receipt_root)
        if journal is None:  # pragma: no cover - 原子写入成功后不可达
            raise HarnessError(
                "v3 request intent terminal journal is missing after migration",
                kind="session_state",
            )
    if (
        journal["thread_id"] != validated_thread_id
        or journal["request_digest"] != validated_request_digest
        or journal["turn_count_before"] != intent["turn_count_before"]
        or journal["thread_policy_digest"] != intent["thread_policy_digest"]
        or (terminal_status is not None and journal["status"] != terminal_status)
        or (
            physical_turn_count_before is not None
            and intent["physical_turn_count_before"] != physical_turn_count_before
        )
    ):
        raise HarnessError(
            "v3 request intent migration identity conflicts with terminal journal",
            kind="session_state",
        )
    turns = _read_thread_turns(thread, expected_thread_id=validated_thread_id)
    candidate_id, candidate_status = _select_unique_codex_request_intent_migration_terminal_turn(
        turns,
        baseline=baseline,
        expected_completed_turn_count=intent["turn_count_before"],
        expected_terminal_status=terminal_status,
    )
    if candidate_id != journal["turn_id"]:
        raise HarnessError(
            "v3 request intent migration terminal turn identity is not in history",
            kind="session_protocol",
        )
    if candidate_status != journal["status"]:
        raise HarnessError(
            "v3 request intent migration terminal status does not match history",
            kind="session_protocol",
        )
    return _assemble_codex_request_intent_migration_result(
        completed_turn_receipt_root,
        migrated_intent=intent,
        terminal_journal=journal,
    )


def migrate_codex_v2_request_intent_to_terminal_turn(
    completed_turn_receipt_root: Path,
    *,
    thread: Any,
    thread_id: str,
    request_digest: str,
    physical_turn_count_before: int | None = None,
    terminal_status: str | None = None,
) -> dict[str, Any]:
    """把缺少物理基线的 v2 intent 安全升级为可恢复的 v3 终态事实。

    输入是旧 session 的 request-intent 目录、可读取完整 history 的同一 Codex
    thread、旧请求摘要和可选的物理基线。动作先完整核对 v2 intent、history 的基线
    和唯一新增 failed/interrupted turn；没有显式基线时，仅接受“新增 turn 之前的
    history 全部是已完成 turn”这一可唯一证明的边界。持久化按 journal-first 顺序
    执行：先原子写白名单 terminal journal，再原子替换 v2 intent 为 v3。第二步失败
    时保留 v2 intent 和匹配 journal，下一次调用会幂等补齐 v3；因此不会留下“只有
    v3 intent、没有 terminal journal”的半迁移状态。函数不读取 provider 正文、不删除
    intent，任一身份、history 或写盘错误都会 fail-closed，调用方随后应保存 Working
    Memory 终止事实并显式 acknowledgement。

    ``physical_turn_count_before`` 为 ``None`` 时由 history 推导唯一安全边界；调用方
    只有在持有同一 thread 的独立物理计数证据时才应显式传入其它值。v2 intent 本身
    没有该字段，不能凭成功 turn 计数猜测此前是否有 failed/interrupted turn。
    """

    if terminal_status is not None and terminal_status not in {"failed", "interrupted"}:
        raise HarnessError(
            "request intent migration terminal status is invalid",
            kind="session_state",
        )
    if (
        physical_turn_count_before is not None
        and (
            isinstance(physical_turn_count_before, bool)
            or not isinstance(physical_turn_count_before, int)
            or physical_turn_count_before < 0
        )
    ):
        raise HarnessError(
            "request intent migration physical baseline is invalid",
            kind="session_state",
        )

    intent = _read_request_intent(completed_turn_receipt_root)
    if intent is None:
        raise HarnessError("request intent is missing", kind="session_state")
    validated_request_digest = _validate_request_digest(request_digest)
    validated_thread_id = _validate_receipt_identifier(thread_id, "thread_id")
    if intent["request_digest"] != validated_request_digest:
        raise HarnessError(
            "request intent migration request digest does not match the original intent",
            kind="session_state",
        )
    if intent["thread_id"] != validated_thread_id:
        raise HarnessError(
            "request intent migration thread identity does not match",
            kind="session_state",
        )

    if intent["schema_version"] == _REQUEST_INTENT_SCHEMA_VERSION:
        return _repair_v3_codex_request_intent_migration(
            completed_turn_receipt_root,
            thread=thread,
            validated_thread_id=validated_thread_id,
            validated_request_digest=validated_request_digest,
            intent=intent,
            physical_turn_count_before=physical_turn_count_before,
            terminal_status=terminal_status,
        )
    if intent["schema_version"] != _LEGACY_REQUEST_INTENT_SCHEMA_VERSION:
        raise HarnessError(
            "request intent migration requires schema v2",
            kind="session_state",
        )

    turns = _read_thread_turns(thread, expected_thread_id=validated_thread_id)
    if physical_turn_count_before is None:
        # v2 没有物理基线。只有“最新一项是唯一终态、此前每项都 completed，
        # 且 completed 数等于旧成功计数”时，边界才不依赖猜测。
        physical_turn_count_before = len(turns) - 1
        if physical_turn_count_before != intent["turn_count_before"]:
            raise CodexRequestIntentRecoveryPending(
                "request intent migration physical boundary is ambiguous",
                turn_id=None,
                reason="request_intent",
                request_digest=intent["request_digest"],
            )
    candidate_id, candidate_status = _select_unique_codex_request_intent_migration_terminal_turn(
        turns,
        baseline=physical_turn_count_before,
        expected_completed_turn_count=intent["turn_count_before"],
        expected_terminal_status=terminal_status,
    )

    target_intent = {
        "schema_version": _REQUEST_INTENT_SCHEMA_VERSION,
        "thread_id": validated_thread_id,
        "request_digest": intent["request_digest"],
        "turn_count_before": intent["turn_count_before"],
        "physical_turn_count_before": physical_turn_count_before,
        "provider_id": intent["provider_id"],
        "env_key": intent["env_key"],
        "model": intent["model"],
        "role": intent["role"],
        "stage": intent["stage"],
        "profile": intent["profile"],
        "workspace": intent["workspace"],
        "thread_policy_digest": intent["thread_policy_digest"],
    }
    target_intent = _validate_request_intent(
        target_intent,
        _request_intent_path(completed_turn_receipt_root),
    )
    target_journal = _validate_terminal_turn_journal(
        {
            "schema_version": _TERMINAL_TURN_SCHEMA_VERSION,
            "thread_id": validated_thread_id,
            "turn_id": candidate_id,
            "status": candidate_status,
            "request_digest": target_intent["request_digest"],
            "turn_count_before": target_intent["turn_count_before"],
            "thread_policy_digest": target_intent["thread_policy_digest"],
        },
        _terminal_turn_journal_path(completed_turn_receipt_root),
    )
    intent_path = _request_intent_path(completed_turn_receipt_root)
    existing_journal = _read_terminal_turn_journal(completed_turn_receipt_root)
    if existing_journal is not None and existing_journal != target_journal:
        raise HarnessError(
            "terminal turn journal conflicts with the v2 request intent migration",
            kind="session_state",
        )
    if existing_journal is None:
        # journal 是已证明的物理终态事实，必须先于 v2→v3 替换落盘。
        _write_terminal_turn_journal(
            completed_turn_receipt_root,
            thread_id=target_journal["thread_id"],
            turn_id=target_journal["turn_id"],
            status=target_journal["status"],
            request_digest=target_journal["request_digest"],
            turn_count_before=target_journal["turn_count_before"],
            thread_policy_digest=target_journal["thread_policy_digest"],
        )
    current_intent = _read_request_intent(completed_turn_receipt_root)
    if current_intent is None:
        raise HarnessError(
            "request intent disappeared during migration",
            kind="session_state",
        )
    if current_intent == target_intent:
        return _assemble_codex_request_intent_migration_result(
            completed_turn_receipt_root,
            migrated_intent=target_intent,
            terminal_journal=target_journal,
        )
    if current_intent != intent:
        raise HarnessError(
            "request intent changed during migration",
            kind="session_state",
        )
    try:
        _atomic_write_json_file(
            intent_path,
            target_intent,
            description="migrated Codex request intent",
            mode=0o600,
        )
    except BaseException:
        # 不尝试删除已经证明的 journal：v2+匹配 journal 是可重启补偿状态，
        # 删除失败反而会丢失唯一的终态物理证据。
        raise
    return _assemble_codex_request_intent_migration_result(
        completed_turn_receipt_root,
        migrated_intent=target_intent,
        terminal_journal=target_journal,
    )


def read_codex_request_intent(root: Path | str) -> dict[str, Any] | None:
    """读取并严格校验一个 Codex request-intent，供恢复审计使用。"""

    return _read_request_intent(root)


def _find_unreceipted_completed_turn(
    thread: Any,
    *,
    expected_thread_id: str,
    request_digest: str,
    prompt_text: str,
    turn_count_before: int,
    physical_turn_count_before: int,
    thread_policy_digest: str,
) -> tuple[dict[str, Any], str] | None:
    """按成功计数与物理基线找出尚未生成 receipt 的唯一成功 turn。

    输入的 ``turn_count_before`` 与 session manifest 一样只统计 ``completed``
    turn；``physical_turn_count_before`` 来自本次 request-intent。动作要求候选恰好
    是物理基线后的唯一新增 turn，并核对基线内成功数。返回临时 receipt 与正文；
    没有新增 turn 时返回 ``None``，history 不完整或有歧义时抛错。
    """

    turns = _read_thread_turns(thread, expected_thread_id=expected_thread_id)
    if physical_turn_count_before > len(turns):
        raise HarnessError(
            "Codex thread history is shorter than request intent baseline",
            kind="session_state",
        )
    baseline_completed = _select_completed_thread_history_turns(
        turns[:physical_turn_count_before]
    )
    if len(baseline_completed) != turn_count_before:
        raise HarnessError(
            "Codex thread history successful baseline does not match request intent",
            kind="session_state",
        )
    new_turns = turns[physical_turn_count_before:]
    if not new_turns:
        return None
    if len(new_turns) != 1:
        raise HarnessError(
            "Codex thread history has unaccounted physical turns",
            kind="session_state",
        )
    candidate = new_turns[0]
    completed_turns = _select_completed_thread_history_turns(turns)
    if (
        _protocol_enum_value(_read_protocol_field(candidate, "status")) != "completed"
        or len(completed_turns) != turn_count_before + 1
    ):
        raise HarnessError(
            "Codex thread history pending turn is not one completed request",
            kind="session_state",
        )
    candidate_id = _read_protocol_field(candidate, "id")
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise HarnessError(
            "Codex thread history pending turn has no id",
            kind="session_protocol",
        )
    candidate_receipt = {
        "schema_version": _RECEIPT_SCHEMA_VERSION,
        "thread_id": expected_thread_id,
        "turn_id": candidate_id.strip(),
        "request_digest": _validate_request_digest(request_digest),
        "turn_count_before": turn_count_before,
        "thread_policy_digest": _validate_thread_policy_digest(thread_policy_digest),
    }
    # 先核对本次 prompt，再生成内存中的临时 receipt；调用方随后会原子补写
    # 真实请求摘要，不能把只有部分字段的历史猜作当前 turn。
    history_prompt = [
        prompt
        for item in (_read_protocol_field(candidate, "items") or [])
        for prompt in (_extract_persisted_user_prompt(item),)
        if prompt is not None
    ]
    if history_prompt != [prompt_text]:
        raise HarnessError(
            "Codex thread history contains an unrecorded different request",
            kind="session_state",
        )
    content = _extract_replayed_turn_content(
        thread,
        expected_thread_id=expected_thread_id,
        receipt=candidate_receipt,
        prompt_text=prompt_text,
        turns=turns,
    )
    return candidate_receipt, content


def _validate_thread_provider_binding(
    thread_config: Mapping[str, Any],
    *,
    provider_id: str,
    env_key: str,
) -> None:
    """核对调用方传入的 provider 配置与 SDK thread 参数一致。"""

    configured_provider = thread_config.get("model_provider")
    if configured_provider is not None and configured_provider != provider_id:
        raise HarnessError(
            "Codex thread provider_id does not match adapter provider_id",
            kind="provider_envelope",
        )
    providers = thread_config.get("model_providers")
    if not isinstance(providers, Mapping):
        return
    provider_config = providers.get(provider_id)
    if provider_config is None:
        return
    if not isinstance(provider_config, Mapping):
        raise HarnessError("Codex thread provider config is malformed", kind="provider_envelope")
    configured_env_key = provider_config.get("env_key")
    if configured_env_key is not None and configured_env_key != env_key:
        raise HarnessError(
            "Codex thread env_key does not match adapter env_key",
            kind="provider_envelope",
        )


def _validate_idle_timeout(value: Any) -> float:
    """校验独立 transport idle watchdog 的有限等待时长。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HarnessError(
            "idle_timeout_seconds must be a positive finite number",
            kind="provider_envelope",
        )
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0:
        raise HarnessError(
            "idle_timeout_seconds must be a positive finite number",
            kind="provider_envelope",
        )
    return normalized


def _default_idle_timeout_for_role(role: Any) -> float:
    """按公开角色选择普通模型事件流的有限默认空闲预算。"""

    if role == CODEX_PROFILE_INVESTIGATION:
        return DEFAULT_INVESTIGATION_IDLE_TIMEOUT_SECONDS
    return DEFAULT_CODEX_IDLE_TIMEOUT_SECONDS


def _turn_handle_id(turn_handle: Any) -> str | None:
    """读取 SDK turn 标识；不存在时返回 None，不猜测标识。"""

    value = getattr(turn_handle, "id", None)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _join_codex_worker_with_timeout(
    worker: threading.Thread | None,
    *,
    timeout_seconds: float,
) -> bool:
    """有限等待 Codex worker，返回它是否仍在运行。

    输入是可选 worker 与正数等待时长；动作只调用 ``join``，不会发送 interrupt。
    worker 已停止或不存在时返回 ``False``；join 抛错或超时后仍存活时返回
    ``True``，供调用方持久化 recovery_pending。
    """

    if worker is None:
        return False
    try:
        worker.join(timeout=timeout_seconds)
    except BaseException:
        return True
    return worker.is_alive()


def _request_codex_turn_interrupt(
    interrupt: Callable[[], Any],
) -> None:
    """调用 interrupt，并在后台 wrapper 内收束 RPC 异常。

    输入是 SDK interrupt 方法；动作最多调用一次 RPC。包括 ``no active turn``
    在内的任意异常都会被吞并，不保留异常正文或打印 traceback。函数无返回值，
    调用方随后仍以 interrupt 线程和 worker 的存活状态决定是否阻断恢复。
    """

    try:
        interrupt()
    except BaseException:
        return


def _interrupt_and_join_worker(
    turn_handle: Any,
    worker: threading.Thread | None,
    *,
    interrupt_timeout_seconds: float = 1.0,
    join_timeout_seconds: float = 1.0,
) -> bool:
    """请求 SDK 中断并有限等待 worker；返回是否仍有无法确认的清理。

    interrupt 在独立 daemon 线程中运行。wrapper 捕获已结束 turn 的
    ``no active turn`` 竞态和其它 RPC 异常，避免线程 traceback 污染 stderr；
    最终只根据 interrupt 线程或 worker 是否仍存活决定是否需要恢复阻断。
    """

    interrupt_thread: threading.Thread | None = None
    if turn_handle is not None:
        interrupt = getattr(turn_handle, "interrupt", None)
        if not callable(interrupt):
            # 没有 interrupt 方法不等于 worker 仍在运行；下面仍会检查线程
            # 是否已经自行结束，已结束时无需把一个完成 turn 标成清理阻断。
            interrupt_thread = None
        else:
            interrupt_thread = threading.Thread(
                target=_request_codex_turn_interrupt,
                args=(interrupt,),
                name="arche-codex-turn-interrupt",
                daemon=True,
            )
            try:
                interrupt_thread.start()
                interrupt_thread.join(timeout=interrupt_timeout_seconds)
            except BaseException:
                return True
    worker_alive = _join_codex_worker_with_timeout(
        worker,
        timeout_seconds=join_timeout_seconds,
    )
    return bool(
        (interrupt_thread is not None and interrupt_thread.is_alive())
        or worker_alive
    )


def _run_turn_stream(
    *,
    thread: Any,
    prompt: str,
    request: HarnessRequest,
    output_schema: Mapping[str, Any] | None,
    result_queue: queue.Queue[tuple[str, Any]],
) -> None:
    """在后台消费 SDK turn 流，避免同步 SDK 阻塞事件处理主线程。"""

    try:
        # approval/cwd/model/sandbox 等细粒度设置已经在 thread_start/resume
        # 固定；turn 只提交本轮输入和可选输出 schema，避免覆盖 thread policy。
        turn_kwargs: dict[str, Any] = {}
        if output_schema is not None:
            turn_kwargs["output_schema"] = dict(output_schema)
        turn = thread.turn(prompt, **turn_kwargs)
        result_queue.put(("started", turn))
        for event in turn.stream():
            result_queue.put(("event", event))
        result_queue.put(("done", None))
    except BaseException as exc:  # 交给主线程统一转换为不含密钥的错误。
        result_queue.put(("error", exc))


def _run_goal_stream(
    *,
    thread: Any,
    prompt_text: str,
    token_budget: int | None,
    checkpoint: GoalCheckpoint | None,
    result_queue: queue.Queue[tuple[str, Any]],
) -> None:
    """在后台启动或恢复 Goal，并把完整通知流交给主线程消费。"""

    try:
        if checkpoint is None:
            handle = _goal_call_initial(thread, prompt_text, token_budget)
        else:
            handle = _goal_call_restore(thread, checkpoint)
        result_queue.put(("started", handle))
        stream_factory = getattr(handle, "stream", None)
        if not callable(stream_factory):
            raise HarnessError(
                "Goal handle has no stream for event tracking",
                kind="session_protocol",
            )
        stream = stream_factory()
        if hasattr(stream, "__aiter__"):
            raise HarnessError(
                "synchronous adapter received an asynchronous Goal stream",
                kind="session_protocol",
            )
        try:
            for event in stream:
                result_queue.put(("event", event))
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()
        result_queue.put(("done", handle))
    except BaseException as exc:
        result_queue.put(("error", exc))


def _track_started_codex_tool(
    payload: Any,
    *,
    active_tool_ids: set[str],
    process_to_item: dict[str, str],
    anonymous_active_tool_count: int,
) -> int:
    """记录 item/started 中的 MCP 或 commandExecution 工具。"""

    item = _read_protocol_field(payload, "item")
    item_type = _protocol_item_type(item)
    if item_type not in {"mcpToolCall", "commandExecution"}:
        return anonymous_active_tool_count
    item_id = _protocol_item_id(item)
    if item_id is None:
        return anonymous_active_tool_count + 1
    item_key = f"item:{item_type}:{item_id}"
    active_tool_ids.add(item_key)
    if item_type == "commandExecution":
        process_id = _protocol_item_process_id(item)
        if process_id is not None:
            process_key = f"process:{process_id}"
            process_to_item[process_key] = item_key
            active_tool_ids.discard(process_key)
    return anonymous_active_tool_count


def _track_item_progress(
    payload: Any,
    *,
    item_type: str,
    active_tool_ids: set[str],
    anonymous_active_tool_count: int,
) -> int:
    """记录带 item_id 的 MCP/commandExecution 进度事件。"""

    item_id = _read_protocol_field(payload, "item_id")
    if isinstance(item_id, str) and item_id.strip():
        active_tool_ids.add(f"item:{item_type}:{item_id.strip()}")
    elif not active_tool_ids:
        anonymous_active_tool_count = max(1, anonymous_active_tool_count)
    return anonymous_active_tool_count


def _track_process_progress(
    payload: Any,
    *,
    active_tool_ids: set[str],
    process_to_item: Mapping[str, str],
    anonymous_active_tool_count: int,
) -> int:
    """记录 process/outputDelta，并把进程映射回 command item。"""

    process_handle = _protocol_process_identity(payload)
    if process_handle is not None:
        process_key = f"process:{process_handle}"
        active_tool_ids.add(process_to_item.get(process_key, process_key))
    elif not active_tool_ids:
        anonymous_active_tool_count = max(1, anonymous_active_tool_count)
    return anonymous_active_tool_count


def _finish_exited_process(
    payload: Any,
    *,
    active_tool_ids: set[str],
    process_to_item: dict[str, str],
) -> None:
    """清理 process/exited 的进程和关联 command item 标记。"""

    process_handle = _protocol_process_identity(payload)
    if process_handle is None:
        return
    process_key = f"process:{process_handle}"
    item_key = process_to_item.pop(process_key, None)
    active_tool_ids.discard(process_key)
    if item_key is not None:
        active_tool_ids.discard(item_key)


def _track_codex_tool_activity(
    *,
    method: Any,
    payload: Any,
    active_tool_ids: set[str],
    process_to_item: dict[str, str],
    anonymous_active_tool_count: int,
) -> int:
    """根据工具进度事件更新活动集合，返回匿名工具计数。"""

    if method == "item/started":
        return _track_started_codex_tool(
            payload,
            active_tool_ids=active_tool_ids,
            process_to_item=process_to_item,
            anonymous_active_tool_count=anonymous_active_tool_count,
        )
    if method == "item/mcpToolCall/progress":
        return _track_item_progress(
            payload,
            item_type="mcpToolCall",
            active_tool_ids=active_tool_ids,
            anonymous_active_tool_count=anonymous_active_tool_count,
        )
    if method == "item/commandExecution/outputDelta":
        return _track_item_progress(
            payload,
            item_type="commandExecution",
            active_tool_ids=active_tool_ids,
            anonymous_active_tool_count=anonymous_active_tool_count,
        )
    if method in {"command/exec/outputDelta", "process/outputDelta"}:
        return _track_process_progress(
            payload,
            active_tool_ids=active_tool_ids,
            process_to_item=process_to_item,
            anonymous_active_tool_count=anonymous_active_tool_count,
        )
    if method == "process/exited":
        _finish_exited_process(
            payload,
            active_tool_ids=active_tool_ids,
            process_to_item=process_to_item,
        )
    return anonymous_active_tool_count


def _finish_codex_tool_item(
    *,
    item: Any,
    active_tool_ids: set[str],
    process_to_item: dict[str, str],
    anonymous_active_tool_count: int,
) -> int:
    """收到 item/completed 后移除工具及其关联进程的活动标记。"""

    item_type = _protocol_item_type(item)
    if item_type not in {"mcpToolCall", "commandExecution"}:
        return anonymous_active_tool_count
    item_id = _protocol_item_id(item)
    if item_id is None:
        return max(0, anonymous_active_tool_count - 1)
    item_key = f"item:{item_type}:{item_id}"
    active_tool_ids.discard(item_key)
    if item_type == "commandExecution":
        process_id = _protocol_item_process_id(item)
        if process_id is not None:
            process_key = f"process:{process_id}"
            process_to_item.pop(process_key, None)
            active_tool_ids.discard(process_key)
        else:
            for process_key, mapped_item_key in tuple(process_to_item.items()):
                if mapped_item_key == item_key:
                    process_to_item.pop(process_key, None)
                    active_tool_ids.discard(process_key)
    return anonymous_active_tool_count


def _read_async_job_field(value: Any, field: str) -> Any:
    """读取异步作业回执字段，兼容 SDK 属性、snake_case 和 camelCase。"""

    aliases = {
        "structured_content": "structuredContent",
        "job_id": "jobId",
        "job_status": "jobStatus",
    }
    alias = aliases.get(field, field)
    if isinstance(value, Mapping):
        if field in value:
            return value[field]
        return value.get(alias)
    direct = getattr(value, field, None)
    if direct is not None:
        return direct
    return getattr(value, alias, None)


def _decode_async_job_json_text(value: Any) -> Mapping[str, Any] | list[Any] | None:
    """解码工具结果中的有限 JSON 文本，失败时返回 None 而不猜测作业状态。"""

    if not isinstance(value, str) or not value.strip():
        return None
    try:
        if len(value.encode("utf-8")) > 1_048_576:
            return None
        decoded = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError, RecursionError):
        return None
    return decoded if isinstance(decoded, (Mapping, list)) else None


def _iter_async_job_payloads(value: Any) -> tuple[Mapping[str, Any], ...]:
    """展开 MCP result、structured content 和文本 envelope 中的 object。"""

    pending: list[Any] = [value]
    result: list[Mapping[str, Any]] = []
    seen: set[int] = set()
    visited = 0
    while pending and visited < _ASYNC_JOB_PAYLOAD_NODE_LIMIT:
        candidate = pending.pop()
        visited += 1
        if isinstance(candidate, (Mapping, list, tuple)):
            candidate_identity = id(candidate)
            if candidate_identity in seen:
                continue
            seen.add(candidate_identity)
        if isinstance(candidate, Mapping):
            result.append(candidate)
            for field in (
                "structured_content",
                "structuredContent",
                "job",
                "content",
            ):
                child = candidate.get(field)
                if child is not None:
                    pending.append(child)
            decoded = _decode_async_job_json_text(candidate.get("text"))
            if decoded is not None:
                pending.append(decoded)
        elif isinstance(candidate, (list, tuple)):
            pending.extend(reversed(candidate))
        else:
            structured = _read_async_job_field(candidate, "structured_content")
            content = _read_async_job_field(candidate, "content")
            if structured is not None:
                pending.append(structured)
            if content is not None:
                pending.append(content)
            decoded = _decode_async_job_json_text(
                _read_async_job_field(candidate, "text")
            )
            if decoded is not None:
                pending.append(decoded)
    return tuple(result)


def _extract_async_job_ids(item: Any) -> set[str]:
    """从同一工具 item 的参数和结构化结果提取明确 job_id。"""

    root = _unwrap_protocol_root(item)
    pending: list[Any] = [_read_async_job_field(root, "arguments")]
    pending.extend(
        _iter_async_job_payloads(_read_async_job_field(root, "result"))
    )
    ids: set[str] = set()
    seen: set[int] = set()
    visited = 0
    while pending and visited < _ASYNC_JOB_PAYLOAD_NODE_LIMIT:
        candidate = pending.pop()
        visited += 1
        if isinstance(candidate, (Mapping, list, tuple)):
            candidate_identity = id(candidate)
            if candidate_identity in seen:
                continue
            seen.add(candidate_identity)
        if isinstance(candidate, Mapping):
            for field in ("job_id", "jobId"):
                value = candidate.get(field)
                if isinstance(value, str) and value.strip():
                    ids.add(value.strip())
            pending.extend(candidate.values())
        elif isinstance(candidate, (list, tuple)):
            pending.extend(candidate)
        else:
            for field in ("job_id", "jobId", "job", "structured_content", "content"):
                child = _read_async_job_field(candidate, field)
                if child is not None:
                    pending.append(child)
    return ids


def _extract_async_job_states(item: Any) -> tuple[set[str], set[str]]:
    """读取明确的非终态和终态字段，忽略 MCP transport 的 success。"""

    root = _unwrap_protocol_root(item)
    states: set[str] = set()
    terminal_flags: set[bool] = set()
    for payload in _iter_async_job_payloads(_read_async_job_field(root, "result")):
        for field in ("job_status", "state"):
            value = _read_async_job_field(payload, field)
            if isinstance(value, str) and value.strip():
                normalized = value.strip().casefold().replace("-", "_")
                normalized = _ASYNC_JOB_STATUS_ALIASES.get(normalized, normalized)
                if normalized in _ASYNC_JOB_NON_TERMINAL_STATUSES | _ASYNC_JOB_TERMINAL_STATUSES:
                    states.add(normalized)
        raw_status = _read_async_job_field(payload, "status")
        if isinstance(raw_status, str) and raw_status.strip():
            normalized = raw_status.strip().casefold().replace("-", "_")
            normalized = _ASYNC_JOB_STATUS_ALIASES.get(normalized, normalized)
            if normalized in _ASYNC_JOB_NON_TERMINAL_STATUSES | _ASYNC_JOB_TERMINAL_STATUSES:
                states.add(normalized)
        terminal = _read_async_job_field(payload, "terminal")
        if isinstance(terminal, bool):
            terminal_flags.add(terminal)
    if len(terminal_flags) > 1:
        return set(), set()
    if terminal_flags == {True}:
        if states & _ASYNC_JOB_NON_TERMINAL_STATUSES:
            return set(), set()
        return set(), {"completed"}
    if len(states) != 1:
        return set(), set()
    state = next(iter(states))
    if state in _ASYNC_JOB_NON_TERMINAL_STATUSES:
        return {state}, set()
    return set(), {state}


def _update_async_job_wait_state_from_completed_item(
    item: Any,
    *,
    known_job_ids: set[str],
    pending_job_ids: set[str],
) -> None:
    """按已完成 MCP item 更新外部作业等待状态。

    输入是 SDK 的 ``item/completed`` item，以及本 turn 已确认的作业身份集合；动作
    只接受 submit/status/collect 工具返回的明确 job_id 和结构化状态，副作用是更新
    两个内存集合。返回 None；字段缺失、未知工具或状态冲突时保持普通 watchdog，
    不把不可靠的模型正文当作 heartbeat。
    """

    root = _unwrap_protocol_root(item)
    if _protocol_item_type(root) != "mcpToolCall":
        return
    tool_name = _read_async_job_field(root, "tool")
    if not isinstance(tool_name, str):
        return
    tool_name = tool_name.strip()
    if tool_name not in (
        _ASYNC_JOB_SUBMIT_TOOLS
        | _ASYNC_JOB_STATUS_TOOLS
        | _ASYNC_JOB_COLLECT_TOOLS
    ):
        return
    item_job_ids = _extract_async_job_ids(root)
    if tool_name in _ASYNC_JOB_SUBMIT_TOOLS:
        known_job_ids.update(item_job_ids)
        non_terminal_states, terminal_states = _extract_async_job_states(root)
        # submit 的结构化回执已经是同一外部作业的第一条状态事实。只有它
        # 同时给出唯一作业身份和明确非终态，才把后续模型长计算切换到作业
        # watchdog；缺失、多个身份或状态冲突都继续使用有限的普通 watchdog。
        if (
            len(item_job_ids) == 1
            and non_terminal_states
            and not terminal_states
        ):
            pending_job_ids.update(item_job_ids)
        elif terminal_states and not non_terminal_states:
            pending_job_ids.difference_update(item_job_ids)
        return
    matched_job_ids = item_job_ids & known_job_ids
    if not matched_job_ids:
        return
    non_terminal_states, terminal_states = _extract_async_job_states(root)
    if tool_name in _ASYNC_JOB_STATUS_TOOLS:
        if non_terminal_states and not terminal_states:
            pending_job_ids.update(matched_job_ids)
        elif terminal_states and not non_terminal_states:
            pending_job_ids.difference_update(matched_job_ids)
        return
    pending_job_ids.difference_update(matched_job_ids)


def _raise_terminal_turn_recovery(journal: Mapping[str, Any], message: str) -> NoReturn:
    """用 journal 中的固定身份抛出可恢复错误，不附带 provider 正文。"""

    raise CodexTerminalTurnRecoveryPending(
        message,
        turn_id=journal["turn_id"],
        status=journal["status"],
        request_digest=journal["request_digest"],
    ) from None


def _acknowledge_terminal_codex_turn(
    journal: Mapping[str, Any],
    *,
    thread: Any | None,
    trace_handle: Any,
    completed_turn_receipt_root: Path,
    on_turn_started: Callable[[str], None] | None,
    on_terminal_turn: Callable[[str, str, str], None] | None,
    on_investigation_rollback: Callable[[str, Any, str], None] | None,
) -> NoReturn:
    """写入白名单 trace，先补记 Started，再让上层同步确认明确终态。

    输入是已经严格校验且刷盘的 journal、trace 和同步回调。动作先把固定四字段
    终态证据写入 trace 并 fsync，再用同一真实 turn_id 幂等重放或补记 Started，
    最后调用 terminal 回调；只有两类回调都成功后才依次删除 intent 和 journal。
    缺少回调、回调失败或任一清理失败时保留尚未删除的恢复事实并抛
    ``terminal_turn`` recovery；全部完成后抛出稳定的终态错误，本函数不返回。
    """

    try:
        _write_terminal_turn_notification(
            trace_handle,
            thread_id=journal["thread_id"],
            turn_id=journal["turn_id"],
            status=journal["status"],
            error_code=journal.get("error_code"),
        )
    except BaseException:
        _raise_terminal_turn_recovery(
            journal,
            "terminal Codex turn trace cannot be durably saved",
        )
    if on_terminal_turn is None:
        _raise_terminal_turn_recovery(
            journal,
            "terminal Codex turn requires synchronous acknowledgement",
        )
    if on_turn_started is not None:
        try:
            on_turn_started(journal["turn_id"])
        except BaseException:
            _raise_terminal_turn_recovery(
                journal,
                "terminal Codex turn Started replay failed",
            )
    try:
        on_terminal_turn(
            journal["thread_id"],
            journal["turn_id"],
            journal["status"],
        )
    except BaseException:
        _raise_terminal_turn_recovery(
            journal,
            "terminal Codex turn acknowledgement failed",
        )
    error_code = journal.get("error_code")
    if error_code is not None and on_investigation_rollback is not None:
        if thread is None:
            _raise_terminal_turn_recovery(
                journal,
                "Investigation turn rollback has no original thread",
            )
        try:
            on_investigation_rollback(journal["thread_id"], thread, error_code)
        except BaseException:
            _raise_terminal_turn_recovery(
                journal,
                "Investigation turn rollback failed",
            )
    try:
        _remove_request_intent(completed_turn_receipt_root)
        _remove_terminal_turn_journal(completed_turn_receipt_root)
    except BaseException:
        _raise_terminal_turn_recovery(
            journal,
            "terminal Codex turn acknowledgement cleanup is incomplete",
        )
    raise CodexTerminalTurnError(
        turn_id=journal["turn_id"],
        status=journal["status"],
        error_code=error_code,
    )


def _raise_terminal_codex_turn_error(
    *,
    payload: Any,
    status: str,
    error_code: str | None,
    thread: Any | None,
    trace_handle: Any,
    thread_id: str,
    completed_turn_receipt_root: Path | None,
    request_digest: str,
    turn_count_before: int,
    thread_policy_digest: str,
    on_turn_started: Callable[[str], None] | None,
    on_terminal_turn: Callable[[str, str, str], None] | None,
    on_investigation_rollback: Callable[[str, Any, str], None] | None,
) -> NoReturn:
    """先保存白名单终态 journal，再请求上层同步确认。

    provider payload 只用于读取 turn id 和枚举状态，error、message、request id
    等自由字段绝不传给 trace、journal 或异常。启用 receipt 时先原子写 journal；
    写入失败会保留 request-intent 并失败关闭。未启用 receipt 时只能写白名单 trace
    后返回终态错误，因为该调用没有可恢复的持久化根。
    """

    turn_value = _read_protocol_field(payload, "turn")
    turn_id = _validate_receipt_identifier(
        _read_protocol_field(turn_value, "id"),
        "turn_id",
    )
    if completed_turn_receipt_root is None:
        _write_terminal_turn_notification(
            trace_handle,
            thread_id=thread_id,
            turn_id=turn_id,
            status=status,
            error_code=error_code,
        )
        raise CodexTerminalTurnError(
            turn_id=turn_id,
            status=status,
            error_code=error_code,
        )
    journal: dict[str, Any] = {
        "schema_version": _TERMINAL_TURN_SCHEMA_VERSION,
        "thread_id": thread_id,
        "turn_id": turn_id,
        "status": status,
        "request_digest": request_digest,
        "turn_count_before": turn_count_before,
        "thread_policy_digest": thread_policy_digest,
    }
    if error_code is not None:
        journal["error_code"] = error_code
    try:
        _write_terminal_turn_journal(
            completed_turn_receipt_root,
            thread_id=thread_id,
            turn_id=turn_id,
            status=status,
            request_digest=request_digest,
            turn_count_before=turn_count_before,
            thread_policy_digest=thread_policy_digest,
            error_code=error_code,
        )
    except BaseException:
        _raise_terminal_turn_recovery(
            journal,
            "terminal Codex turn journal cannot be durably saved",
        )
    _acknowledge_terminal_codex_turn(
        journal,
        thread=thread,
        trace_handle=trace_handle,
        completed_turn_receipt_root=completed_turn_receipt_root,
        on_turn_started=on_turn_started,
        on_terminal_turn=on_terminal_turn,
        on_investigation_rollback=on_investigation_rollback,
    )


def _persist_completed_codex_turn_receipt(
    payload: Any,
    *,
    completed_turn_receipt_root: Path | None,
    thread_id: str,
    request_digest: str,
    turn_count_before: int,
    thread_policy_digest: str,
) -> None:
    """校验成功终态并持久保存 receipt，随后删除 request-intent。

    输入是 ``turn/completed`` payload、可选 receipt 目录和会话身份；动作只接受
    ``completed`` 状态，并在启用持久化时依次写 pending、receipt、删除 pending
    与 intent。成功无返回值；状态异常时抛 session_protocol，任一写盘步骤失败时
    抛 completion recovery_pending，调用方不得把该 turn 当作新请求再次提交。
    """

    turn_value = _read_protocol_field(payload, "turn")
    turn_id = str(_read_protocol_field(turn_value, "id") or "").strip()
    status = _protocol_enum_value(_read_protocol_field(turn_value, "status"))
    if status != "completed":
        raise HarnessError(
            "Codex turn/completed notification has no completed status",
            kind="session_protocol",
        )
    if completed_turn_receipt_root is None:
        return
    try:
        _write_completion_pending(
            completed_turn_receipt_root,
            thread_id=thread_id,
            turn_id=turn_id,
            request_digest=request_digest,
            turn_count_before=turn_count_before,
            thread_policy_digest=thread_policy_digest,
        )
        _write_completed_turn_receipt(
            completed_turn_receipt_root,
            thread_id=thread_id,
            turn_id=turn_id,
            request_digest=request_digest,
            turn_count_before=turn_count_before,
            thread_policy_digest=thread_policy_digest,
        )
        _remove_completion_pending(completed_turn_receipt_root)
        _remove_request_intent(completed_turn_receipt_root)
    except BaseException:
        raise CodexCompletionRecoveryPending(
            "completed turn receipt cannot be durably saved",
            turn_id=turn_id,
            reason="completion_receipt",
        ) from None


def _handle_codex_turn_event(
    event: Any,
    *,
    thread: Any | None,
    trace_handle: Any,
    sensitive_values: Sequence[str],
    items: list[Any],
    thread_id: str,
    request_digest: str,
    turn_count_before: int,
    thread_policy_digest: str,
    completed_turn_receipt_root: Path | None,
    active_tool_ids: set[str],
    process_to_item: dict[str, str],
    anonymous_active_tool_count: int,
    known_async_job_ids: set[str],
    pending_async_job_ids: set[str],
    token_usage: Any,
    on_turn_started: Callable[[str], None] | None,
    on_terminal_turn: Callable[[str, str, str], None] | None,
    on_investigation_rollback: Callable[[str, Any, str], None] | None,
    on_tool_completed: Callable[[Any], None] | None,
) -> tuple[Any, Any, int]:
    """消费一个 Codex 通知，更新工具活动、receipt 和响应事件状态。"""

    payload = getattr(event, "payload", None)
    method = getattr(event, "method", None)
    failure = None
    anonymous_active_tool_count = _track_codex_tool_activity(
        method=method,
        payload=payload,
        active_tool_ids=active_tool_ids,
        process_to_item=process_to_item,
        anonymous_active_tool_count=anonymous_active_tool_count,
    )

    if method == "turn/completed":
        failure_details = _read_terminal_turn_failure_details(payload)
        if failure_details is not None:
            failure, error_code = failure_details
            _raise_terminal_codex_turn_error(
                payload=payload,
                status=failure,
                error_code=error_code,
                thread=thread,
                trace_handle=trace_handle,
                thread_id=thread_id,
                completed_turn_receipt_root=completed_turn_receipt_root,
                request_digest=request_digest,
                turn_count_before=turn_count_before,
                thread_policy_digest=thread_policy_digest,
                on_turn_started=on_turn_started,
                on_terminal_turn=on_terminal_turn,
                on_investigation_rollback=on_investigation_rollback,
            )
        _persist_completed_codex_turn_receipt(
            payload,
            completed_turn_receipt_root=completed_turn_receipt_root,
            thread_id=thread_id,
            request_digest=request_digest,
            turn_count_before=turn_count_before,
            thread_policy_digest=thread_policy_digest,
        )

    if method == "error":
        _write_error_notification(trace_handle, payload)
    else:
        _write_notification(trace_handle, event, sensitive_values=sensitive_values)
    if method == "item/completed":
        item = _read_protocol_field(payload, "item")
        if item is not None:
            items.append(item)
            _update_async_job_wait_state_from_completed_item(
                item,
                known_job_ids=known_async_job_ids,
                pending_job_ids=pending_async_job_ids,
            )
            if (
                on_tool_completed is not None
                and _protocol_item_type(item) == "mcpToolCall"
            ):
                # 只转发 app-server 已完成的真实 MCP item；不读取模型正文，
                # 上层随后按工具名和服务端回执核对是否是可等待作业。
                on_tool_completed(item)
        anonymous_active_tool_count = _finish_codex_tool_item(
            item=item,
            active_tool_ids=active_tool_ids,
            process_to_item=process_to_item,
            anonymous_active_tool_count=anonymous_active_tool_count,
        )
    elif method == "thread/tokenUsage/updated":
        token_usage = getattr(payload, "token_usage", None)
    else:
        failure = _read_terminal_turn_failure_status(payload)
    if failure is not None:
        raise HarnessError("Codex turn failed", kind="transport")
    if method == "turn/completed":
        active_tool_ids.clear()
        process_to_item.clear()
        anonymous_active_tool_count = 0
    completed_payload = payload if method == "turn/completed" else None
    return token_usage, completed_payload, anonymous_active_tool_count


def _consume_codex_turn_events(
    *,
    result_queue: queue.Queue[tuple[str, Any]],
    worker: threading.Thread,
    thread: Any | None,
    request: HarnessRequest,
    trace_handle: Any,
    trace_path: Path,
    thread_id: str,
    profile: str,
    approval_mode: ApprovalMode,
    sensitive_values: Sequence[str],
    request_digest: str,
    turn_count_before: int,
    thread_policy_digest: str,
    completed_turn_receipt_root: Path | None,
    idle_timeout_seconds: float,
    tool_idle_timeout_seconds: float,
    turn_handle_state: dict[str, Any],
    on_turn_started: Callable[[str], None] | None,
    on_terminal_turn: Callable[[str, str, str], None] | None,
    on_investigation_rollback: Callable[[str, Any, str], None] | None,
    on_tool_completed: Callable[[Any], None] | None,
) -> tuple[CodexSdkRunResult, Any]:
    """消费 worker 事件直到完成，并返回响应及 SDK turn 句柄。

    普通模型静默使用 ``idle_timeout_seconds``；已由同一 turn 的结构化 submit/get
    回执确认仍在运行的外部作业，使用独立的 ``tool_idle_timeout_seconds``。作业进入
    明确终态或完成 collect 后立即恢复普通 watchdog，未知或冲突回执不会延长等待。
    """

    items: list[Any] = []
    token_usage: Any = None
    completed_payload: Any = None
    deadline = (
        None
        if request.timeout_seconds is None
        else time.monotonic() + request.timeout_seconds
    )
    last_activity = time.monotonic()
    active_tool_ids: set[str] = set()
    process_to_item: dict[str, str] = {}
    anonymous_active_tool_count = 0
    known_async_job_ids: set[str] = set()
    pending_async_job_ids: set[str] = set()
    request_job_id = _read_async_job_field(request.metadata, "job_id")
    if isinstance(request_job_id, str) and request_job_id.strip():
        known_async_job_ids.add(request_job_id.strip())
    turn_handle: Any = None
    done = False
    while not done:
        now = time.monotonic()
        watchdog_timeout = (
            tool_idle_timeout_seconds
            if active_tool_ids
            or anonymous_active_tool_count
            or pending_async_job_ids
            else idle_timeout_seconds
        )
        idle_remaining = watchdog_timeout - (now - last_activity)
        if idle_remaining <= 0:
            raise HarnessError("Codex SDK event stream idle timeout", kind="transport")
        remaining = min(0.25, idle_remaining)
        if deadline is not None:
            total_remaining = deadline - now
            if total_remaining <= 0:
                raise HarnessError("Codex SDK turn timed out", kind="transport")
            remaining = min(remaining, total_remaining)
        try:
            kind, value = result_queue.get(timeout=remaining)
        except queue.Empty:
            continue
        last_activity = time.monotonic()
        if kind == "started":
            turn_handle = value
            turn_handle_state["value"] = value
            _notify_codex_turn_started(
                turn_handle,
                on_turn_started=on_turn_started,
                sensitive_values=sensitive_values,
            )
            continue
        if kind == "event":
            token_usage, completed_payload, anonymous_active_tool_count = _handle_codex_turn_event(
                value,
                thread=thread,
                trace_handle=trace_handle,
                sensitive_values=sensitive_values,
                items=items,
                thread_id=thread_id,
                request_digest=request_digest,
                turn_count_before=turn_count_before,
                thread_policy_digest=thread_policy_digest,
                completed_turn_receipt_root=completed_turn_receipt_root,
                active_tool_ids=active_tool_ids,
                process_to_item=process_to_item,
                anonymous_active_tool_count=anonymous_active_tool_count,
                known_async_job_ids=known_async_job_ids,
                pending_async_job_ids=pending_async_job_ids,
                token_usage=token_usage,
                on_turn_started=on_turn_started,
                on_terminal_turn=on_terminal_turn,
                on_investigation_rollback=on_investigation_rollback,
                on_tool_completed=on_tool_completed,
            )
        elif kind == "error":
            turn_error = _sanitize_sdk_error(
                "Codex SDK turn failed",
                value,
                sensitive_values,
                kind="transport",
            )
            raise turn_error from None
        elif kind == "done":
            done = True
    worker.join(timeout=1.0)
    if worker.is_alive():
        raise HarnessError("Codex SDK turn did not stop after completion", kind="transport")
    if completed_payload is None:
        raise HarnessError("Codex SDK turn completed event is missing", kind="session_protocol")
    content = _extract_final_content(items)
    turn_id = str(getattr(getattr(completed_payload, "turn", None), "id", "")).strip()
    return CodexSdkRunResult(
        response=HarnessResponse(
            content=content,
            finish_reason="stop",
            model=request.model,
            usage=_usage_mapping(token_usage),
            raw_trace_path=str(trace_path),
            metadata={
                "harness": request.metadata.get("harness"),
                "role": request.role,
                "stage": request.stage,
                "profile": profile,
                "approval_mode": approval_mode.value,
                "thread_id": thread_id,
                "turn_id": turn_id,
                "sdk": "openai-codex-0.149.0+arche.1",
            },
        ),
        thread_id=thread_id,
        turn_id=turn_id,
        request_digest=request_digest,
        thread_policy_digest=thread_policy_digest,
        completed_tool_items=tuple(
            item for item in items if _protocol_item_type(item) == "mcpToolCall"
        ),
    ), turn_handle


def _wait_for_codex_worker_cleanup(
    error: BaseException,
    *,
    turn_handle: Any,
    worker: threading.Thread,
) -> tuple[bool, bool]:
    """按 turn 终态清理 worker，并返回终态标记与是否仍无法确认停止。

    输入是原异常、turn 句柄和 worker；明确终态只等待 worker 自行退出，其它失败
    才请求 interrupt。返回 ``(terminal_turn_completed, cleanup_pending)``，不写
    recovery；清理函数自身异常按无法确认停止处理，不让原 transport 异常泄露。
    """

    terminal_turn_completed = bool(getattr(error, "terminal_turn_completed", False))
    if terminal_turn_completed:
        return True, _join_codex_worker_with_timeout(worker, timeout_seconds=1.0)
    try:
        return False, _interrupt_and_join_worker(turn_handle, worker)
    except BaseException:
        return False, True


def _select_codex_worker_recovery(
    error: BaseException,
    *,
    terminal_turn_completed: bool,
    turn_handle: Any,
) -> CodexRecoveryPending:
    """把未停止 worker 映射成应持久化的恢复异常。

    输入是原异常、终态标记和 turn 句柄；明确终态、receipt 写盘失败及普通异常会
    生成 ``worker_cleanup``，已有 worker/recovery 异常则尽量保留原身份。返回
    recovery 异常对象，不写状态；缺少 turn 身份时保持 ``None``，由上层失败关闭。
    """

    if isinstance(error, CodexTerminalTurnRecoveryPending):
        return error
    if terminal_turn_completed:
        return CodexWorkerRecoveryPending(
            "Codex SDK worker did not stop after terminal turn completion",
            turn_id=getattr(error, "turn_id", None) or _turn_handle_id(turn_handle),
            reason="worker_cleanup",
        )
    if isinstance(error, CodexWorkerRecoveryPending):
        return error
    if isinstance(error, CodexCompletionRecoveryPending):
        return CodexWorkerRecoveryPending(
            "Codex SDK turn worker could not be stopped while saving receipt",
            turn_id=error.turn_id,
            reason="worker_cleanup",
        )
    if isinstance(error, CodexRecoveryPending):
        return error
    return CodexWorkerRecoveryPending(
        "Codex SDK turn worker could not be stopped",
        turn_id=_turn_handle_id(turn_handle),
        reason="worker_cleanup",
    )


def _raise_worker_recovery_if_needed(
    error: BaseException,
    *,
    turn_handle: Any,
    worker: threading.Thread,
    on_recovery_pending: Callable[[str | None], None] | None,
) -> None:
    """按 turn 终态等待或中断 worker；无法确认停止时只抛 recovery_pending。"""

    terminal_turn_completed, cleanup_pending = _wait_for_codex_worker_cleanup(
        error,
        turn_handle=turn_handle,
        worker=worker,
    )
    if not cleanup_pending:
        return
    pending = _select_codex_worker_recovery(
        error,
        terminal_turn_completed=terminal_turn_completed,
        turn_handle=turn_handle,
    )
    if on_recovery_pending is not None and pending.reason == "worker_cleanup":
        try:
            on_recovery_pending(pending.turn_id)
        except BaseException:
            raise pending from None
    if pending is not error:
        raise pending from None


def _notify_codex_turn_started(
    turn_handle: Any,
    *,
    on_turn_started: Callable[[str], None] | None,
    sensitive_values: Sequence[str],
) -> None:
    """用 SDK 已接受的真实 turn 身份通知上层持久化启动事实。

    输入是 ``thread.turn`` 返回的句柄、可选回调和敏感值；动作先读取非空
    ``turn_id``，再调用一次回调。回调为空时无副作用；句柄缺少身份或回调失败时
    抛出已脱敏的 HarnessError，外层会中断当前 turn，不能在完成后补造启动事实。
    """

    if on_turn_started is None:
        return
    turn_id = _turn_handle_id(turn_handle)
    if turn_id is None:
        raise HarnessError(
            "Codex SDK accepted turn has no turn id",
            kind="session_protocol",
        )
    try:
        on_turn_started(turn_id)
    except BaseException as exc:
        if isinstance(exc, HarnessError) and not _exception_chain_contains_sensitive_text(
            exc, sensitive_values
        ):
            raise
        raise _sanitize_sdk_error(
            "Codex turn-start callback failed",
            exc,
            sensitive_values,
            kind="session_state",
        ) from None


def _run_codex_turn_worker(
    *,
    thread: Any,
    prompt_text: str,
    request: HarnessRequest,
    trace_handle: Any,
    trace_path: Path,
    thread_id: str,
    profile: str,
    approval_mode: ApprovalMode,
    sensitive_values: Sequence[str],
    request_digest: str,
    turn_count_before: int,
    thread_policy_digest: str,
    completed_turn_receipt_root: Path | None,
    idle_timeout_seconds: float,
    tool_idle_timeout_seconds: float,
    on_recovery_pending: Callable[[str | None], None] | None,
    on_turn_started: Callable[[str], None] | None,
    on_terminal_turn: Callable[[str, str, str], None] | None,
    on_investigation_rollback: Callable[[str, Any, str], None] | None,
    on_tool_completed: Callable[[Any], None] | None,
) -> CodexSdkRunResult:
    """启动并监管一个 Codex worker，统一处理事件、超时和回收。"""

    result_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
    worker = threading.Thread(
        target=_run_turn_stream,
        kwargs={
            "thread": thread,
            "prompt": prompt_text,
            "request": request,
            # Investigation 的工具结果由运行时先处理，最终科学 JSON 仍由运行时
            # 校验；SDK 只负责保证当前 turn 返回非空 assistant 正文。
            "output_schema": (
                None if request.role == "investigation" else request.response_schema
            ),
            "result_queue": result_queue,
        },
        name="arche-codex-turn",
        daemon=True,
    )
    worker.start()
    turn_handle_state: dict[str, Any] = {"value": None}
    turn_handle: Any = None
    try:
        result, turn_handle = _consume_codex_turn_events(
            result_queue=result_queue,
            worker=worker,
            thread=thread,
            request=request,
            trace_handle=trace_handle,
            trace_path=trace_path,
            thread_id=thread_id,
            profile=profile,
            approval_mode=approval_mode,
            sensitive_values=sensitive_values,
            request_digest=request_digest,
            turn_count_before=turn_count_before,
            thread_policy_digest=thread_policy_digest,
            completed_turn_receipt_root=completed_turn_receipt_root,
            idle_timeout_seconds=idle_timeout_seconds,
            tool_idle_timeout_seconds=tool_idle_timeout_seconds,
            turn_handle_state=turn_handle_state,
            on_turn_started=on_turn_started,
            on_terminal_turn=on_terminal_turn,
            on_investigation_rollback=on_investigation_rollback,
            on_tool_completed=on_tool_completed,
        )
        return result
    except BaseException as exc:
        turn_handle = turn_handle_state.get("value")
        _raise_worker_recovery_if_needed(
            exc,
            turn_handle=turn_handle,
            worker=worker,
            on_recovery_pending=on_recovery_pending,
        )
        raise


def _reject_new_thread_with_recovery_records(
    completed_turn_receipt_root: Path,
    *,
    completion_recovery: Mapping[str, Any] | None,
) -> None:
    """有遗留恢复记录时拒绝用新 thread 猜测性重提交 turn。"""

    completed_receipts = _load_completed_turn_receipts(completed_turn_receipt_root)
    pending_journal = _read_completion_pending(completed_turn_receipt_root)
    terminal_journal = _read_terminal_turn_journal(completed_turn_receipt_root)
    request_intent = _read_request_intent(completed_turn_receipt_root)
    goal_intent = _read_goal_intent(completed_turn_receipt_root)
    goal_turn_journal = _read_goal_turn_journal(completed_turn_receipt_root)
    goal_receipt = _read_goal_receipt(completed_turn_receipt_root)
    if goal_intent is not None or goal_turn_journal or goal_receipt is not None:
        raise CodexGoalRecoveryPending(
            "Goal recovery requires the original Codex session",
            goal_status=(goal_intent or {}).get("status")
            or (goal_receipt or {}).get("goal_status"),
            logical_goal_id=(goal_intent or {}).get("logical_goal_id")
            or (goal_receipt or {}).get("logical_goal_id"),
            physical_turn_ids=[
                record["turn_id"] for record in goal_turn_journal
            ]
            or (goal_receipt or {}).get("physical_turn_ids", ()),
        )
    if completed_receipts:
        raise HarnessError(
            "completed turn receipt requires the original Codex session; "
            "refusing to create a new thread",
            kind="session_state",
        )
    if pending_journal is not None or (
        isinstance(completion_recovery, Mapping)
        and completion_recovery.get("reason") == "completion_receipt"
    ):
        raise HarnessError(
            "completion recovery requires the original Codex session",
            kind="session_state",
        )
    if terminal_journal is not None:
        raise CodexTerminalTurnRecoveryPending(
            "terminal turn acknowledgement requires the original Codex session",
            turn_id=terminal_journal["turn_id"],
            status=terminal_journal["status"],
            request_digest=terminal_journal["request_digest"],
        )
    if (
        isinstance(completion_recovery, Mapping)
        and completion_recovery.get("reason") == "terminal_turn"
    ):
        raise HarnessError(
            "terminal recovery requires the original Codex session",
            kind="recovery_pending",
        )
    if request_intent is not None:
        raise CodexRequestIntentRecoveryPending(
            "request intent requires the original Codex session",
            turn_id=None,
            reason="request_intent",
            request_digest=request_intent["request_digest"],
        )


def _validate_codex_turn_parameters(
    *,
    provider_id: str,
    env_key: str,
    trace_sensitive_values: Sequence[str],
    secret_environment: Mapping[str, str] | None,
    completed_turn_receipt_root: Path | None,
    request_digest: str | None,
    turn_count_before: int | None,
    session_id: str | None,
    completion_recovery: Mapping[str, Any] | None,
    idle_timeout_seconds: float,
    tool_idle_timeout_seconds: float,
) -> tuple[str, str, float, float, dict[str, str], tuple[str, ...]]:
    """校验 provider、秘密环境、watchdog 和 receipt 参数并返回规范值。"""

    provider_id, env_key = _validate_codex_provider_identity(provider_id, env_key)
    normalized_idle_timeout = _validate_idle_timeout(idle_timeout_seconds)
    normalized_tool_idle_timeout = _validate_idle_timeout(tool_idle_timeout_seconds)
    normalized_secret_environment = _validate_secret_environment(secret_environment)
    sensitive_values = _merge_sensitive_values(
        trace_sensitive_values,
        normalized_secret_environment.values(),
    )
    _validate_codex_receipt_parameters(
        completed_turn_receipt_root=completed_turn_receipt_root,
        request_digest=request_digest,
        turn_count_before=turn_count_before,
        session_id=session_id,
        completion_recovery=completion_recovery,
    )
    return (
        provider_id,
        env_key,
        normalized_idle_timeout,
        normalized_tool_idle_timeout,
        normalized_secret_environment,
        sensitive_values,
    )


def _validate_codex_provider_identity(
    provider_id: str,
    env_key: str,
) -> tuple[str, str]:
    """校验 provider 标识和环境变量名，拒绝路径或控制字符。"""

    if not isinstance(provider_id, str) or not provider_id.strip():
        raise HarnessError("Codex provider_id is missing", kind="provider_envelope")
    if not isinstance(env_key, str) or not env_key.strip():
        raise HarnessError("Codex provider env_key is missing", kind="provider_envelope")
    provider_id = provider_id.strip()
    env_key = env_key.strip()
    if any(ord(char) < 32 or ord(char) == 127 for char in provider_id) or any(
        char in provider_id for char in "/\\"
    ):
        raise HarnessError("Codex provider_id is malformed", kind="provider_envelope")
    if _ENVIRONMENT_NAME.fullmatch(env_key) is None:
        raise HarnessError("Codex provider env_key is malformed", kind="provider_envelope")
    return provider_id, env_key


def _merge_sensitive_values(
    trace_sensitive_values: Sequence[str],
    secret_values: Any,
) -> tuple[str, ...]:
    """合并 trace 和 app-server 私有环境中的敏感正文并去重。"""

    return tuple(
        dict.fromkeys(
            value
            for value in (*trace_sensitive_values, *secret_values)
            if isinstance(value, str) and value
        )
    )


def _validate_codex_receipt_parameters(
    *,
    completed_turn_receipt_root: Path | None,
    request_digest: str | None,
    turn_count_before: int | None,
    session_id: str | None,
    completion_recovery: Mapping[str, Any] | None,
) -> None:
    """校验 receipt 参数配对，并拒绝新 thread 猜测性重放旧 turn。"""

    if completed_turn_receipt_root is not None:
        _validate_present_receipt_parameters(request_digest, turn_count_before)
    elif request_digest is not None or turn_count_before is not None:
        raise HarnessError(
            "completed turn receipt root is missing",
            kind="session_state",
        )
    if completed_turn_receipt_root is not None and session_id is None:
        _reject_new_thread_with_recovery_records(
            completed_turn_receipt_root,
            completion_recovery=completion_recovery,
        )


def _validate_present_receipt_parameters(
    request_digest: str | None,
    turn_count_before: int | None,
) -> None:
    """校验已启用 receipt 时必需的请求摘要和旧 turn 计数。"""

    if request_digest is None or turn_count_before is None:
        raise HarnessError(
            "completed turn receipt configuration is incomplete",
            kind="session_state",
        )
    _validate_request_digest(request_digest)
    if (
        isinstance(turn_count_before, bool)
        or not isinstance(turn_count_before, int)
        or turn_count_before < 0
    ):
        raise HarnessError(
            "completed turn receipt turn_count_before is invalid",
            kind="session_state",
        )


def _start_codex_thread(
    *,
    codex: Codex,
    request: HarnessRequest,
    thread_config: Mapping[str, Any],
    session_id: str | None,
    provider_id: str,
    env_key: str,
    normalized_secret_environment: Mapping[str, str],
    sensitive_values: Sequence[str],
    expected_thread_policy_digest: str | None,
    auth_mode: str = "api",
    permission_profile: str | None = None,
    enable_investigation_goals: bool | None = None,
) -> _CodexThreadStartup:
    """构造 effective thread policy、启动或恢复 thread，并隔离 SDK 异常。"""

    try:
        profile = resolve_codex_profile(request)
        auth_mode = normalize_codex_auth_mode(auth_mode)
        effective_thread_config = build_codex_capability_config(
            thread_config=thread_config,
            workspace=request.workspace,
            profile=profile,
            env_key=env_key,
            environment_keys=tuple(normalized_secret_environment),
            auth_mode=auth_mode,
            enable_investigation_goals=enable_investigation_goals,
        )
        thread_policy_digest = calculate_codex_thread_policy_digest(
            thread_config=effective_thread_config,
            workspace=request.workspace,
            provider_id=provider_id,
            env_key=env_key,
            model=request.model,
            role=request.role,
            stage=request.stage,
            profile=profile,
            auth_mode=auth_mode,
            permission_profile=permission_profile,
        )
        if expected_thread_policy_digest is not None:
            _validate_thread_policy_digest(expected_thread_policy_digest)
            if expected_thread_policy_digest != thread_policy_digest:
                raise HarnessError(
                    "Codex effective thread policy digest does not match current config",
                    kind="session_state",
                )
        configured_trace_values = _collect_trace_sensitive_values(effective_thread_config)
        trace_values = tuple(
            dict.fromkeys(
                value
                for value in (*sensitive_values, *configured_trace_values)
                if isinstance(value, str) and value
            )
        )
        _validate_thread_provider_binding(
            effective_thread_config,
            provider_id=provider_id,
            env_key=env_key,
        )
        approval_mode = approval_mode_for_codex_profile(profile)
        sandbox = sandbox_for_codex_profile(
            profile,
            permission_profile=permission_profile,
        )
        if session_id:
            thread = codex.thread_resume(
                session_id,
                approval_mode=approval_mode,
                config=effective_thread_config,
                cwd=str(request.workspace.resolve()),
                model=request.model,
                model_provider=provider_id,
                sandbox=sandbox,
            )
        else:
            thread = codex.thread_start(
                approval_mode=approval_mode,
                config=effective_thread_config,
                cwd=str(request.workspace.resolve()),
                ephemeral=str(request.role).strip().lower().replace("-", "_")
                not in _PERSISTENT_CODEX_ROLES,
                model=request.model,
                model_provider=provider_id,
                sandbox=sandbox,
            )
    except HarnessError:
        raise
    return _CodexThreadStartup(
        thread=thread,
        profile=profile,
        thread_policy_digest=thread_policy_digest,
        trace_values=trace_values,
        approval_mode=approval_mode,
    )


def _expected_codex_request_intent(
    *,
    thread_id: str,
    request: HarnessRequest,
    request_digest: str,
    turn_count_before: int,
    physical_turn_count_before: int,
    provider_id: str,
    env_key: str,
    profile: str,
    thread_policy_digest: str,
) -> dict[str, Any]:
    """生成不含 secret 的本轮 request-intent 身份。"""

    return {
        "schema_version": _REQUEST_INTENT_SCHEMA_VERSION,
        "thread_id": thread_id,
        "request_digest": request_digest,
        "turn_count_before": turn_count_before,
        "physical_turn_count_before": physical_turn_count_before,
        "provider_id": provider_id,
        "env_key": env_key,
        "model": request.model,
        "role": request.role,
        "stage": request.stage,
        "profile": profile,
        "workspace": str(request.workspace.resolve()),
        "thread_policy_digest": thread_policy_digest,
    }


def _validate_current_codex_request_intent(
    request_intent: Mapping[str, Any],
    *,
    thread_id: str,
    request: HarnessRequest,
    request_digest: str,
    turn_count_before: int,
    provider_id: str,
    env_key: str,
    profile: str,
    thread_policy_digest: str,
) -> int:
    """核对当前请求身份并返回磁盘记录的物理 history 基线。

    输入是已严格解析的 intent 与当前调用身份；动作拒绝没有物理基线的 v2，且
    异常始终携带磁盘中的原 request digest。成功返回非负物理 turn 数，不写状态；
    旧 schema 或任一身份漂移时抛 request-intent recovery_pending。
    """

    if request_intent["schema_version"] != _REQUEST_INTENT_SCHEMA_VERSION:
        raise CodexRequestIntentRecoveryPending(
            "request intent requires explicit schema migration",
            turn_id=None,
            reason="request_intent",
            request_digest=request_intent["request_digest"],
        )
    physical_turn_count_before = request_intent["physical_turn_count_before"]
    expected_intent = _expected_codex_request_intent(
        thread_id=thread_id,
        request=request,
        request_digest=request_digest,
        turn_count_before=turn_count_before,
        physical_turn_count_before=physical_turn_count_before,
        provider_id=provider_id,
        env_key=env_key,
        profile=profile,
        thread_policy_digest=thread_policy_digest,
    )
    if request_intent != expected_intent:
        raise CodexRequestIntentRecoveryPending(
            "request intent does not match the current request",
            turn_id=None,
            reason="request_intent",
            request_digest=request_intent["request_digest"],
        )
    return physical_turn_count_before


def _validate_legacy_codex_request_intent(
    request_intent: Mapping[str, Any],
    *,
    thread_id: str,
    request: HarnessRequest,
    request_digest: str,
    turn_count_before: int,
    provider_id: str,
    env_key: str,
    profile: str,
    thread_policy_digest: str,
) -> None:
    """在迁移写盘前核对旧 v2 intent 的完整当前请求身份。

    v2 只缺少物理 history 基线，其余字段仍必须与本次恢复请求完全一致。动作只
    读取并比较固定字段；匹配返回空值，任何漂移都以原 request digest 抛出恢复
    阻断，调用方不得先写 v3 intent 或 terminal journal。
    """

    if request_intent.get("schema_version") != _LEGACY_REQUEST_INTENT_SCHEMA_VERSION:
        raise CodexRequestIntentRecoveryPending(
            "request intent is not a legacy schema",
            turn_id=None,
            reason="request_intent",
            request_digest=request_intent.get("request_digest"),
        )
    expected_intent = _expected_codex_request_intent(
        thread_id=thread_id,
        request=request,
        request_digest=request_digest,
        turn_count_before=turn_count_before,
        physical_turn_count_before=0,
        provider_id=provider_id,
        env_key=env_key,
        profile=profile,
        thread_policy_digest=thread_policy_digest,
    )
    expected_intent.pop("physical_turn_count_before")
    expected_intent["schema_version"] = _LEGACY_REQUEST_INTENT_SCHEMA_VERSION
    if dict(request_intent) != expected_intent:
        raise CodexRequestIntentRecoveryPending(
            "legacy request intent does not match the current request",
            turn_id=None,
            reason="request_intent",
            request_digest=request_intent.get("request_digest"),
        )


def _validate_pending_journal_identity(
    pending_journal: Mapping[str, Any],
    *,
    thread_id: str,
    request_digest: str,
    thread_policy_digest: str,
    context: str = "completion pending journal",
) -> None:
    """核对 pending journal 的 thread、请求摘要和策略摘要。"""

    if pending_journal["thread_id"] != thread_id:
        raise HarnessError(
            f"{context} thread_id does not match session",
            kind="session_protocol",
        )
    if pending_journal["request_digest"] != request_digest:
        raise HarnessError(
            f"{context} request does not match current request",
            kind="session_state",
        )
    if pending_journal["thread_policy_digest"] != thread_policy_digest:
        raise HarnessError(
            f"{context} thread policy drifted",
            kind="session_state",
        )


def _restore_completion_pending_journal(
    completion_recovery: Mapping[str, Any],
    *,
    thread_id: str,
    request_digest: str,
    thread_policy_digest: str,
) -> dict[str, Any]:
    """从已持久化 recovery 状态重建并核对 completion-pending journal。"""

    pending_journal = _validate_completion_pending(
        {
            "schema_version": _RECEIPT_SCHEMA_VERSION,
            "thread_id": completion_recovery.get("thread_id", thread_id),
            "turn_id": completion_recovery.get("turn_id"),
            "request_digest": completion_recovery.get("request_digest"),
            "turn_count_before": completion_recovery.get("turn_count"),
            "thread_policy_digest": completion_recovery.get("thread_policy_digest"),
        },
        Path(_COMPLETION_PENDING_NAME),
    )
    _validate_pending_journal_identity(
        pending_journal,
        thread_id=thread_id,
        request_digest=request_digest,
        thread_policy_digest=thread_policy_digest,
        context="completion recovery",
    )
    return pending_journal


def _validate_terminal_turn_history(
    thread: Any,
    journal: Mapping[str, Any],
    *,
    physical_turn_count_before: int | None = None,
) -> None:
    """核对 journal 指向 history 中最后一个唯一明确终态。

    manifest 的 ``turn_count_before`` 只统计成功 completed turn；历史中的更早
    failed/interrupted 不增加该计数。当前终态必须是最后一个物理 turn，且状态、
    thread 和 turn 身份完全匹配；未终结、未知或多余成功 turn 均失败关闭。
    """

    turns = _read_thread_turns(
        thread,
        expected_thread_id=journal["thread_id"],
    )
    validated_turn_ids = _validate_unique_codex_history_turn_ids(turns)
    if (
        physical_turn_count_before is not None
        and len(turns) != physical_turn_count_before + 1
    ):
        raise HarnessError(
            "terminal turn journal physical history does not match request intent",
            kind="session_state",
        )
    completed_turns = _select_completed_thread_history_turns(turns)
    if len(completed_turns) != journal["turn_count_before"]:
        raise HarnessError(
            "terminal turn journal completed count does not match session",
            kind="session_state",
        )
    matching = [
        turn
        for index, turn in enumerate(turns)
        if validated_turn_ids[index] == journal["turn_id"]
    ]
    if len(matching) != 1 or not turns or matching[0] is not turns[-1]:
        raise HarnessError(
            "terminal turn journal does not match the latest history turn",
            kind="session_protocol",
        )
    actual_status = _protocol_enum_value(
        _read_protocol_field(matching[0], "status")
    )
    if actual_status != journal["status"]:
        raise HarnessError(
            "terminal turn journal status does not match history",
            kind="session_protocol",
        )


def _terminal_turn_status_from_recovery_history(
    thread: Any,
    *,
    thread_id: str,
    turn_id: str,
    turn_count_before: int,
) -> str:
    """从完整 history 恢复 terminal recovery 缺失的有限状态枚举。"""

    turns = _read_thread_turns(thread, expected_thread_id=thread_id)
    validated_turn_ids = _validate_unique_codex_history_turn_ids(turns)
    completed_turns = _select_completed_thread_history_turns(turns)
    if len(completed_turns) != turn_count_before:
        raise HarnessError(
            "terminal recovery completed count does not match session",
            kind="session_state",
        )
    matching = [
        turn
        for index, turn in enumerate(turns)
        if validated_turn_ids[index] == turn_id
    ]
    if len(matching) != 1 or not turns or matching[0] is not turns[-1]:
        raise HarnessError(
            "terminal recovery does not match the latest history turn",
            kind="session_protocol",
        )
    status = _protocol_enum_value(_read_protocol_field(matching[0], "status"))
    if status not in {"failed", "interrupted"}:
        raise HarnessError(
            "terminal recovery history status is invalid",
            kind="session_protocol",
        )
    return str(status)


def _restore_terminal_turn_journal(
    thread: Any,
    recovery: Mapping[str, Any],
    *,
    root: Path,
    thread_id: str,
    request_digest: str,
    turn_count_before: int,
    thread_policy_digest: str,
) -> dict[str, Any]:
    """从持久化 recovery 和原 thread history 重建缺失的终态 journal。"""

    recovery_turn_id = _validate_receipt_identifier(
        recovery.get("turn_id"),
        "turn_id",
    )
    if recovery.get("request_digest") != request_digest:
        raise HarnessError(
            "terminal recovery request does not match current request",
            kind="session_state",
        )
    if recovery.get("thread_policy_digest") != thread_policy_digest:
        raise HarnessError("terminal recovery thread policy drifted", kind="session_state")
    if recovery.get("turn_count") != turn_count_before:
        raise HarnessError(
            "terminal recovery turn count does not match session",
            kind="session_state",
        )
    status = _terminal_turn_status_from_recovery_history(
        thread,
        thread_id=thread_id,
        turn_id=recovery_turn_id,
        turn_count_before=turn_count_before,
    )
    journal = {
        "schema_version": _TERMINAL_TURN_SCHEMA_VERSION,
        "thread_id": thread_id,
        "turn_id": recovery_turn_id,
        "status": status,
        "request_digest": request_digest,
        "turn_count_before": turn_count_before,
        "thread_policy_digest": thread_policy_digest,
    }
    try:
        _write_terminal_turn_journal(
            root,
            thread_id=journal["thread_id"],
            turn_id=journal["turn_id"],
            status=journal["status"],
            request_digest=journal["request_digest"],
            turn_count_before=journal["turn_count_before"],
            thread_policy_digest=journal["thread_policy_digest"],
        )
    except BaseException:
        _raise_terminal_turn_recovery(
            journal,
            "terminal Codex turn journal cannot be restored",
        )
    return journal


def _validate_terminal_turn_recovery_consistency(
    journal: Mapping[str, Any],
    recovery: Mapping[str, Any] | None,
    *,
    turn_count_before: int,
    stage: str,
    recovery_only: bool = False,
) -> None:
    """核对 journal、当前成功计数与可选 store recovery 完全一致。

    旧版本可能只持久化了 ``request_intent`` recovery；它只有在 action、阶段、
    成功计数、请求摘要、策略摘要和空 turn 身份都能与已经写入的 terminal journal
    对上时，才允许作为一次性过渡输入。其它原因或任一字段冲突都继续 fail-closed。
    """

    if journal["turn_count_before"] != turn_count_before:
        raise HarnessError(
            "terminal turn journal turn count does not match session",
            kind="session_state",
        )
    if not isinstance(recovery, Mapping):
        return
    if recovery.get("reason") == "request_intent":
        expected_legacy = {
            "state": "recovery_pending",
            "reason": "request_intent",
            "action": "request_intent_confirmation_required",
            "turn_id": "",
            "turn_count": journal["turn_count_before"],
            "request_digest": journal["request_digest"],
            "thread_policy_digest": journal["thread_policy_digest"],
        }
        actual_legacy = {
            field: recovery.get(field) for field in expected_legacy
        }
        recovery_stage = recovery.get("stage")
        stage_matches = recovery_stage == stage
        if recovery_only:
            # recovery-only 由上层先核对恢复记录允许的观察阶段；adapter 在这里
            # 仍独立核对真正的 request-intent、journal、thread history、摘要、
            # 策略和计数，但不把“发现旧 intent 的入口”误作原请求 stage。
            stage_matches = isinstance(recovery_stage, str) and bool(
                recovery_stage.strip()
            )
        if actual_legacy != expected_legacy or not stage_matches:
            raise HarnessError(
                "terminal turn journal conflicts with legacy request-intent recovery",
                kind="session_state",
            )
        return
    expected = {
        "reason": "terminal_turn",
        "turn_id": journal["turn_id"],
        "turn_count": journal["turn_count_before"],
        "request_digest": journal["request_digest"],
        "thread_policy_digest": journal["thread_policy_digest"],
    }
    actual = {field: recovery.get(field) for field in expected}
    if actual != expected:
        raise HarnessError(
            "terminal turn journal conflicts with current recovery",
            kind="session_state",
        )


def _validate_terminal_turn_request_intent(
    request_intent: Mapping[str, Any] | None,
    journal: Mapping[str, Any],
    *,
    thread_id: str,
    request: HarnessRequest,
    request_digest: str,
    turn_count_before: int,
    provider_id: str,
    env_key: str,
    profile: str,
    thread_policy_digest: str,
) -> int | None:
    """允许已清理 intent；存在时返回其已核对的物理 history 基线。"""

    if request_intent is None:
        return None
    physical_turn_count_before = _validate_current_codex_request_intent(
        request_intent,
        thread_id=thread_id,
        request=request,
        request_digest=request_digest,
        turn_count_before=turn_count_before,
        provider_id=provider_id,
        env_key=env_key,
        profile=profile,
        thread_policy_digest=thread_policy_digest,
    )
    if request_intent["request_digest"] != journal["request_digest"]:
        raise CodexTerminalTurnRecoveryPending(
            "terminal turn request intent does not match current request",
            turn_id=journal["turn_id"],
            status=journal["status"],
            request_digest=journal["request_digest"],
        )
    return physical_turn_count_before


def _raise_request_intent_history_pending(
    request_intent: Mapping[str, Any],
    message: str,
) -> NoReturn:
    """以磁盘中的原请求摘要报告 history 与 intent 无法唯一对应。"""

    raise CodexRequestIntentRecoveryPending(
        message,
        turn_id=None,
        reason="request_intent",
        request_digest=request_intent["request_digest"],
    )


def _find_terminal_turn_after_request_intent(
    thread: Any,
    request_intent: Mapping[str, Any],
) -> dict[str, Any] | None:
    """只识别物理基线后唯一新增的 failed/interrupted turn。

    输入是已核对的 v3 intent 和原 thread；动作读取完整 history，先核对基线阶段
    的成功计数，再检查恰好一个新增物理 turn。新增成功 turn 返回 ``None`` 交给
    receipt replay；新增失败/中断返回白名单 journal 内容。没有新增也返回
    ``None``；数量、状态或身份有歧义时保留 intent 并抛 recovery_pending。
    """

    turns = _read_thread_turns(
        thread,
        expected_thread_id=request_intent["thread_id"],
    )
    try:
        validated_turn_ids = _validate_unique_codex_history_turn_ids(turns)
    except HarnessError:
        _raise_request_intent_history_pending(
            request_intent,
            "request intent terminal history turn identity is invalid or ambiguous",
        )
    physical_turn_count_before = request_intent["physical_turn_count_before"]
    if len(turns) < physical_turn_count_before:
        _raise_request_intent_history_pending(
            request_intent,
            "request intent physical history is shorter than its baseline",
        )
    completed_turns = _select_completed_thread_history_turns(turns)
    baseline_completed = _select_completed_thread_history_turns(
        turns[:physical_turn_count_before]
    )
    if len(baseline_completed) != request_intent["turn_count_before"]:
        _raise_request_intent_history_pending(
            request_intent,
            "request intent successful history does not match its baseline",
        )
    new_turn_count = len(turns) - physical_turn_count_before
    if new_turn_count == 0:
        return None
    if new_turn_count != 1:
        _raise_request_intent_history_pending(
            request_intent,
            "request intent history has multiple unaccounted physical turns",
        )
    latest = turns[-1]
    status = _protocol_enum_value(_read_protocol_field(latest, "status"))
    if status == "completed":
        if len(completed_turns) != request_intent["turn_count_before"] + 1:
            _raise_request_intent_history_pending(
                request_intent,
                "request intent completed history is ambiguous",
            )
        return None
    if status not in {"failed", "interrupted"}:
        _raise_request_intent_history_pending(
            request_intent,
            "request intent latest history turn is unfinished or unknown",
        )
    if len(completed_turns) != request_intent["turn_count_before"]:
        _raise_request_intent_history_pending(
            request_intent,
            "request intent terminal history has an unexpected successful turn",
        )
    turn_id = validated_turn_ids[-1]
    return {
        "schema_version": _TERMINAL_TURN_SCHEMA_VERSION,
        "thread_id": request_intent["thread_id"],
        "turn_id": turn_id,
        "status": status,
        "request_digest": request_intent["request_digest"],
        "turn_count_before": request_intent["turn_count_before"],
        "thread_policy_digest": request_intent["thread_policy_digest"],
    }


def _restore_terminal_turn_journal_from_request_intent(
    thread: Any,
    request_intent: Mapping[str, Any],
    *,
    root: Path,
) -> dict[str, Any] | None:
    """从 intent 与完整 history 补写崩溃窗口中缺失的终态 journal。"""

    journal = _find_terminal_turn_after_request_intent(thread, request_intent)
    if journal is None:
        return None
    try:
        _write_terminal_turn_journal(
            root,
            thread_id=journal["thread_id"],
            turn_id=journal["turn_id"],
            status=journal["status"],
            request_digest=journal["request_digest"],
            turn_count_before=journal["turn_count_before"],
            thread_policy_digest=journal["thread_policy_digest"],
        )
    except BaseException:
        _raise_terminal_turn_recovery(
            journal,
            "terminal Codex turn journal cannot be restored from request intent",
        )
    return journal


def _prepare_terminal_turn_replay(
    *,
    thread: Any,
    thread_id: str,
    request: HarnessRequest,
    completed_turn_receipt_root: Path | None,
    request_digest: str,
    turn_count_before: int,
    provider_id: str,
    env_key: str,
    profile: str,
    thread_policy_digest: str,
    completion_recovery: Mapping[str, Any] | None,
    recovery_only: bool = False,
) -> dict[str, Any] | None:
    """恢复并严格核对待确认终态；存在时禁止进入普通 replay 或新 turn。"""

    if completed_turn_receipt_root is None:
        return None
    journal = _read_terminal_turn_journal(completed_turn_receipt_root)
    recovery_is_terminal = (
        isinstance(completion_recovery, Mapping)
        and completion_recovery.get("reason") == "terminal_turn"
    )
    if journal is None and recovery_is_terminal:
        journal = _restore_terminal_turn_journal(
            thread,
            completion_recovery,
            root=completed_turn_receipt_root,
            thread_id=thread_id,
            request_digest=request_digest,
            turn_count_before=turn_count_before,
            thread_policy_digest=thread_policy_digest,
        )
    if journal is None:
        request_intent = _read_request_intent(completed_turn_receipt_root)
        if request_intent is not None:
            if request_intent["schema_version"] == _LEGACY_REQUEST_INTENT_SCHEMA_VERSION:
                _validate_legacy_codex_request_intent(
                    request_intent,
                    thread_id=thread_id,
                    request=request,
                    request_digest=request_digest,
                    turn_count_before=turn_count_before,
                    provider_id=provider_id,
                    env_key=env_key,
                    profile=profile,
                    thread_policy_digest=thread_policy_digest,
                )
                # v2 没有物理 history 基线；迁移函数只在同一 thread 能唯一证明
                # 最新失败/中断 turn 时补齐 v3 和 terminal journal。迁移失败时
                # 保留原 intent，不能进入普通 replay 或启动 replacement。
                try:
                    migrate_codex_v2_request_intent_to_terminal_turn(
                        completed_turn_receipt_root,
                        thread=thread,
                        thread_id=thread_id,
                        request_digest=request_digest,
                    )
                except CodexRequestIntentRecoveryPending:
                    raise
                except BaseException as exc:
                    # 迁移写盘或 history 读取失败仍属于旧 intent 的恢复阻断；
                    # 用原摘要保存身份，禁止上层把当前请求摘要覆盖旧事实。
                    raise CodexRequestIntentRecoveryPending(
                        "legacy request intent migration is pending",
                        turn_id=getattr(exc, "turn_id", None),
                        reason="request_intent",
                        request_digest=request_intent["request_digest"],
                    ) from None
                request_intent = _read_request_intent(completed_turn_receipt_root)
                if request_intent is None:
                    raise HarnessError(
                        "request intent disappeared after schema migration",
                        kind="session_state",
                    )
            _validate_current_codex_request_intent(
                request_intent,
                thread_id=thread_id,
                request=request,
                request_digest=request_digest,
                turn_count_before=turn_count_before,
                provider_id=provider_id,
                env_key=env_key,
                profile=profile,
                thread_policy_digest=thread_policy_digest,
            )
            journal = _restore_terminal_turn_journal_from_request_intent(
                thread,
                request_intent,
                root=completed_turn_receipt_root,
            )
    if journal is None:
        return None
    _validate_pending_journal_identity(
        journal,
        thread_id=thread_id,
        request_digest=request_digest,
        thread_policy_digest=thread_policy_digest,
        context="terminal turn journal",
    )
    _validate_terminal_turn_recovery_consistency(
        journal,
        completion_recovery,
        turn_count_before=turn_count_before,
        stage=request.stage,
        recovery_only=recovery_only,
    )
    request_intent = _read_request_intent(completed_turn_receipt_root)
    physical_turn_count_before = _validate_terminal_turn_request_intent(
        request_intent,
        journal,
        thread_id=thread_id,
        request=request,
        request_digest=request_digest,
        turn_count_before=turn_count_before,
        provider_id=provider_id,
        env_key=env_key,
        profile=profile,
        thread_policy_digest=thread_policy_digest,
    )
    _validate_terminal_turn_history(
        thread,
        journal,
        physical_turn_count_before=physical_turn_count_before,
    )
    return journal


def _load_codex_replay_journals(
    *,
    completed_turn_receipt_root: Path | None,
    completion_recovery: Mapping[str, Any] | None,
    thread_id: str,
    request: HarnessRequest,
    request_digest: str,
    turn_count_before: int,
    provider_id: str,
    env_key: str,
    profile: str,
    thread_policy_digest: str,
    replay_turn_id: str | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """读取并核对 request-intent 与 completion-pending journal。"""

    if completed_turn_receipt_root is None:
        return None, None
    request_intent = _read_request_intent(completed_turn_receipt_root)
    if request_intent is not None:
        if replay_turn_id is None:
            _validate_current_codex_request_intent(
                request_intent,
                thread_id=thread_id,
                request=request,
                request_digest=request_digest,
                turn_count_before=turn_count_before,
                provider_id=provider_id,
                env_key=env_key,
                profile=profile,
                thread_policy_digest=thread_policy_digest,
            )
        else:
            # waiting sidecar 已经把目标 turn 固定下来；残留 request-intent
            # 可能保存的是旧 job prompt，因而只核对 session/thread/policy 和
            # completed 计数，不再要求当前 job 的 request digest。
            if (
                request_intent["thread_id"] != thread_id
                or request_intent["thread_policy_digest"] != thread_policy_digest
                or request_intent["turn_count_before"] != turn_count_before
            ):
                raise HarnessError(
                    "Codex waiting replay request-intent identity drifted",
                    kind="session_state",
                )
    pending_journal = _read_completion_pending(completed_turn_receipt_root)
    if pending_journal is not None:
        _validate_pending_journal_identity(
            pending_journal,
            thread_id=thread_id,
            request_digest=request_digest,
            thread_policy_digest=thread_policy_digest,
        )
    if (
        pending_journal is None
        and isinstance(completion_recovery, Mapping)
        and completion_recovery.get("reason") == "completion_receipt"
    ):
        pending_journal = _restore_completion_pending_journal(
            completion_recovery,
            thread_id=thread_id,
            request_digest=request_digest,
            thread_policy_digest=thread_policy_digest,
        )
    return request_intent, pending_journal


def _resolve_completed_codex_replay(
    *,
    thread: Any,
    thread_id: str,
    prompt_text: str,
    completed_turn_receipt_root: Path | None,
    session_id: str | None,
    pending_journal: Mapping[str, Any] | None,
    request_digest: str,
    thread_policy_digest: str,
    replay_turn_id: str | None = None,
    current_turn_count: int = 0,
    request_intent: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, str | None, tuple[Any, ...]]:
    """查找 receipt，并从原 thread history 提取正文与工具回执。

    有 waiting sidecar 的 ``replay_turn_id`` 时，receipt 的原始 request digest
    优先于本次重建的 prompt；这允许 job replacement 造成 prompt 漂移，同时仍以
    同一 thread/turn、manifest 计数、策略和官方 history 作为安全边界。
    """

    if completed_turn_receipt_root is None or not (session_id or pending_journal is not None):
        return None, None, ()
    if replay_turn_id is not None:
        replay_turn_id = _validate_receipt_identifier(replay_turn_id, "replay_turn_id")
        replay_receipt = read_codex_completed_turn_receipt(
            completed_turn_receipt_root,
            thread_id=thread_id,
            turn_id=replay_turn_id,
            thread_policy_digest=thread_policy_digest,
        )
        if replay_receipt is None and _load_completed_turn_receipts(
            completed_turn_receipt_root
        ):
            raise HarnessError(
                "waiting replay turn does not match a completed receipt",
                kind="session_state",
            )
    else:
        replay_receipt = _find_matching_completed_turn_receipt(
            completed_turn_receipt_root,
            thread_id=thread_id,
            request_digest=request_digest,
            thread_policy_digest=thread_policy_digest,
        )
    if pending_journal is not None:
        if replay_receipt is not None and replay_receipt != pending_journal:
            raise HarnessError(
                "completed receipt conflicts with completion pending journal",
                kind="session_state",
            )
        replay_receipt = dict(pending_journal)
    if replay_receipt is None:
        return None, None, ()
    if replay_turn_id is not None:
        _rollback_trailing_codex_turns_after_receipt(
            completed_turn_receipt_root,
            thread=thread,
            thread_id=thread_id,
            receipt=replay_receipt,
            thread_policy_digest=thread_policy_digest,
            current_turn_count=current_turn_count,
            request_intent=request_intent,
        )
    replay_content, replay_completed_tool_items = _extract_replayed_turn_payload(
        thread,
        expected_thread_id=thread_id,
        receipt=replay_receipt,
        prompt_text=None if replay_turn_id is not None else prompt_text,
    )
    return replay_receipt, replay_content, replay_completed_tool_items


def _resolve_unreceipted_codex_replay(
    *,
    thread: Any,
    thread_id: str,
    request_intent: Mapping[str, Any] | None,
    prompt_text: str,
    completed_turn_receipt_root: Path | None,
    replay_receipt: Mapping[str, Any] | None,
    pending_journal: Mapping[str, Any] | None,
    session_id: str | None,
) -> tuple[dict[str, Any] | None, str | None, tuple[Any, ...]]:
    """从 request-intent 对应的 history turn 恢复正文与工具回执。"""

    if not (
        completed_turn_receipt_root is not None
        and replay_receipt is None
        and pending_journal is None
        and session_id is not None
        and request_intent is not None
    ):
        return replay_receipt, None, ()
    history_replay = _find_unreceipted_completed_turn(
        thread,
        expected_thread_id=thread_id,
        request_digest=request_intent["request_digest"],
        prompt_text=prompt_text,
        turn_count_before=request_intent["turn_count_before"],
        physical_turn_count_before=request_intent["physical_turn_count_before"],
        thread_policy_digest=request_intent["thread_policy_digest"],
    )
    if history_replay is None:
        raise CodexRequestIntentRecoveryPending(
            "request intent has no corresponding completed turn",
            turn_id=None,
            reason="request_intent",
            request_digest=request_intent["request_digest"],
        )
    replay_receipt, replay_content = history_replay
    replay_content, replay_completed_tool_items = _extract_replayed_turn_payload(
        thread,
        expected_thread_id=thread_id,
        receipt=replay_receipt,
        prompt_text=prompt_text,
    )
    if replay_content != history_replay[1]:
        raise HarnessError(
            "Codex thread history changed during replay",
            kind="session_state",
        )
    return replay_receipt, replay_content, replay_completed_tool_items


def _prepare_codex_replay_state(
    *,
    thread: Any,
    thread_id: str,
    request: HarnessRequest,
    prompt_text: str,
    session_id: str | None,
    completed_turn_receipt_root: Path | None,
    request_digest: str,
    turn_count_before: int,
    provider_id: str,
    env_key: str,
    profile: str,
    thread_policy_digest: str,
    completion_recovery: Mapping[str, Any] | None,
    replay_turn_id: str | None = None,
) -> _CodexReplayState:
    """读取 request-intent、pending 和 history，只返回可严格重放的 turn。"""

    request_intent, pending_journal = _load_codex_replay_journals(
        completed_turn_receipt_root=completed_turn_receipt_root,
        completion_recovery=completion_recovery,
        thread_id=thread_id,
        request=request,
        request_digest=request_digest,
        turn_count_before=turn_count_before,
        provider_id=provider_id,
        env_key=env_key,
        profile=profile,
        thread_policy_digest=thread_policy_digest,
        replay_turn_id=replay_turn_id,
    )
    replay_receipt, replay_content, replay_completed_tool_items = _resolve_completed_codex_replay(
        thread=thread,
        thread_id=thread_id,
        prompt_text=prompt_text,
        completed_turn_receipt_root=completed_turn_receipt_root,
        session_id=session_id,
        pending_journal=pending_journal,
        request_digest=request_digest,
        thread_policy_digest=thread_policy_digest,
        replay_turn_id=replay_turn_id,
        current_turn_count=turn_count_before,
        request_intent=request_intent,
    )
    if replay_content is None:
        (
            replay_receipt,
            replay_content,
            replay_completed_tool_items,
        ) = _resolve_unreceipted_codex_replay(
            thread=thread,
            thread_id=thread_id,
            request_intent=request_intent,
            prompt_text=prompt_text,
            completed_turn_receipt_root=completed_turn_receipt_root,
            replay_receipt=replay_receipt,
            pending_journal=pending_journal,
            session_id=session_id,
        )
    return _CodexReplayState(
        request_intent=request_intent,
        pending_journal=pending_journal,
        replay_receipt=replay_receipt,
        replay_content=replay_content,
        replay_completed_tool_items=replay_completed_tool_items,
    )


def _register_codex_thread(
    *,
    thread_id: str,
    on_thread_ready: Callable[[str], None] | None,
    sensitive_values: Sequence[str],
) -> None:
    """调用 thread 登记回调并把回调异常转换为不泄密的状态错误。"""

    if on_thread_ready is None:
        return
    try:
        on_thread_ready(thread_id)
    except BaseException as exc:
        if isinstance(exc, HarnessError) and not _exception_chain_contains_sensitive_text(
            exc, sensitive_values
        ):
            raise
        raise _sanitize_sdk_error(
            "Codex thread registration failed",
            exc,
            sensitive_values,
            kind="session_state",
        ) from None


def _persist_replayed_receipt(
    completed_turn_receipt_root: Path,
    receipt: Mapping[str, Any],
) -> None:
    """确认重放结果已经可恢复地写入 receipt，并清理旧 journal。"""

    try:
        _write_completed_turn_receipt(
            completed_turn_receipt_root,
            thread_id=receipt["thread_id"],
            turn_id=receipt["turn_id"],
            request_digest=receipt["request_digest"],
            turn_count_before=receipt["turn_count_before"],
            thread_policy_digest=receipt["thread_policy_digest"],
        )
        _remove_completion_pending(completed_turn_receipt_root)
        _remove_request_intent(completed_turn_receipt_root)
    except BaseException:
        raise CodexCompletionRecoveryPending(
            "completed turn receipt recovery is not durable",
            turn_id=receipt["turn_id"],
            reason="completion_receipt",
        ) from None


def _build_replayed_codex_result(
    *,
    request: HarnessRequest,
    trace_path: Path,
    profile: str,
    approval_mode: ApprovalMode,
    thread_id: str,
    receipt: Mapping[str, Any],
    content: str,
    request_digest: str,
    thread_policy_digest: str,
    completed_tool_items: tuple[Any, ...],
) -> CodexSdkRunResult:
    """把已核对的 history 正文和工具回执包装为不再调用模型的 SDK 结果。"""

    turn_id = receipt["turn_id"]
    return CodexSdkRunResult(
        response=HarnessResponse(
            content=content,
            finish_reason="stop",
            model=request.model,
            usage={},
            raw_trace_path=str(trace_path),
            metadata={
                "harness": request.metadata.get("harness"),
                "role": request.role,
                "stage": request.stage,
                "profile": profile,
                "approval_mode": approval_mode.value,
                "thread_id": thread_id,
                "turn_id": turn_id,
                "replayed": True,
                "sdk": "openai-codex-0.149.0+arche.1",
            },
        ),
        thread_id=thread_id,
        turn_id=turn_id,
        replayed=True,
        receipt_turn_count_before=receipt["turn_count_before"],
        request_digest=request_digest,
        thread_policy_digest=thread_policy_digest,
        completed_tool_items=completed_tool_items,
    )


def _run_codex_pre_turn_fence(
    before_new_turn: Callable[[], None] | None,
    *,
    before_new_turn_with_thread: Callable[[Any, str], None] | None = None,
    thread: Any,
    thread_id: str,
    sensitive_values: Sequence[str],
) -> None:
    """在写入 request-intent 和启动 worker 之前执行调用方的新 turn 副作用栅栏。

    ``before_new_turn`` 保留原有的无参数 scope 栅栏；需要在已有 Codex thread
    上执行一次恢复动作的调用方使用 ``before_new_turn_with_thread``。两类回调都
    发生在 request-intent 和真实模型 turn 之前，任一失败都会阻止新 turn。
    """

    if before_new_turn is None and before_new_turn_with_thread is None:
        return
    if before_new_turn is not None:
        try:
            before_new_turn()
        except BaseException as exc:
            if isinstance(exc, HarnessError) and not _exception_chain_contains_sensitive_text(
                exc, sensitive_values
            ):
                raise
            raise _sanitize_sdk_error(
                "Codex pre-turn fence failed",
                exc,
                sensitive_values,
                kind="session_state",
            ) from None
    if before_new_turn_with_thread is not None:
        try:
            before_new_turn_with_thread(thread, thread_id)
        except BaseException as exc:
            if isinstance(exc, HarnessError) and not _exception_chain_contains_sensitive_text(
                exc, sensitive_values
            ):
                raise
            raise _sanitize_sdk_error(
                "Codex pre-turn thread recovery failed",
                exc,
                sensitive_values,
                kind="session_state",
            ) from None


def _read_request_intent_history_baseline(
    thread: Any,
    *,
    thread_id: str,
    session_id: str | None,
    turn_count_before: int,
) -> int:
    """返回本次 request-intent 前的物理 turn 数。

    新建 thread 由 SDK 合同确定没有历史，直接返回零；恢复 thread 时读取完整
    history，拒绝未终结状态或与 manifest 成功计数不符的记录。函数不写状态，
    history 不可读或有漂移时抛错，调用方不得启动 worker。
    """

    if session_id is None:
        return 0
    turns = _read_thread_turns(thread, expected_thread_id=thread_id)
    completed_turns = _select_completed_thread_history_turns(turns)
    if len(completed_turns) != turn_count_before:
        raise HarnessError(
            "Codex thread successful history does not match session before request intent",
            kind="session_state",
        )
    return len(turns)


def _persist_codex_request_intent(
    completed_turn_receipt_root: Path | None,
    *,
    thread: Any,
    thread_id: str,
    session_id: str | None,
    request_digest: str,
    turn_count_before: int,
    provider_id: str,
    env_key: str,
    request: HarnessRequest,
    profile: str,
    thread_policy_digest: str,
) -> None:
    """在真实 worker 前持久化完整 request-intent，失败时保持恢复阻断。"""

    if completed_turn_receipt_root is None:
        return
    try:
        physical_turn_count_before = _read_request_intent_history_baseline(
            thread,
            thread_id=thread_id,
            session_id=session_id,
            turn_count_before=turn_count_before,
        )
        _write_request_intent(
            completed_turn_receipt_root,
            thread_id=thread_id,
            request_digest=request_digest,
            turn_count_before=turn_count_before,
            physical_turn_count_before=physical_turn_count_before,
            provider_id=provider_id,
            env_key=env_key,
            model=request.model,
            role=request.role,
            stage=request.stage,
            profile=profile,
            workspace=request.workspace,
            thread_policy_digest=thread_policy_digest,
        )
    except BaseException:
        raise CodexRequestIntentRecoveryPending(
            "request intent cannot be durably saved",
            turn_id=None,
            reason="request_intent",
            request_digest=request_digest,
        ) from None


def _goal_field(value: Any, *names: str) -> Any:
    """从 Goal 结果或通知读取第一个存在的固定字段。"""

    for name in names:
        candidate = _read_protocol_field(value, name)
        if candidate is not None:
            return candidate
        camel = name.split("_")
        alias = camel[0] + "".join(part.title() for part in camel[1:])
        candidate = _read_protocol_field(value, alias)
        if candidate is not None:
            return candidate
    return None


def _goal_event_turn(payload: Any) -> Any:
    """提取通知中的物理 turn 对象。"""

    turn = _goal_field(payload, "turn")
    if turn is not None:
        return turn
    if _goal_field(payload, "turn_id", "id") is not None:
        return payload
    return None


def _goal_event_item(payload: Any) -> Any:
    """提取通知中的 item 对象，不读取其自由文本。"""

    return _goal_field(payload, "item")


def _goal_event_goal(payload: Any) -> Any:
    """提取通知中的 Goal 状态对象。"""

    return _goal_field(payload, "goal")


def _goal_event_method(event: Any) -> str:
    """返回通知方法名，缺失时使用固定未知标记。"""

    method = _goal_field(event, "method")
    return method.strip() if isinstance(method, str) and method.strip() else "unknown"


def _goal_event_payload(event: Any) -> Any:
    """返回通知 payload；不把原始 provider 对象写入 trace。"""

    payload = _goal_field(event, "payload", "params")
    return payload if payload is not None else event


def _goal_handle_identifier(handle: Any, result: Any = None) -> str | None:
    """读取 Goal handle 的首个物理 turn 身份。"""

    for source in (handle, result):
        if source is None:
            continue
        for field in ("id", "turn_id", "first_turn_id", "logical_turn_id"):
            try:
                value = _goal_field(source, field)
            except BaseException:
                # 0.149 恢复句柄在首个物理 turn 尚未出现时，读取 id
                # 可能主动抛错；这只表示暂时没有身份，不能把它当成流失败。
                continue
            if isinstance(value, str) and value.strip():
                try:
                    return _validate_receipt_identifier(value.strip(), "turn_id")
                except HarnessError:
                    continue
    return None


def _goal_provider_identifier(handle: Any, result: Any = None) -> str:
    """读取 provider Goal 身份；缺失时用本地逻辑身份代替内部占位。"""

    for source in (result, handle):
        if source is None:
            continue
        for field in ("goal_id", "provider_goal_id", "operation_id", "logical_goal_id"):
            value = _goal_field(source, field)
            if isinstance(value, str) and value.strip():
                return _validate_receipt_identifier(value.strip(), "goal_id")
    return ""


def _goal_physical_ids(source: Any) -> list[str]:
    """读取结果中提供的物理 turn 列表，不接受模糊区间。"""

    for field in ("physical_turn_ids", "turn_ids", "physical_turn_range"):
        value = _goal_field(source, field)
        if isinstance(value, Mapping):
            value = _goal_field(value, "turn_ids", "ids")
        if isinstance(value, (list, tuple)):
            identifiers = [
                _validate_receipt_identifier(item, "physical_turn_id")
                for item in value
                if isinstance(item, str) and item.strip()
            ]
            if len(identifiers) != len(value) or len(set(identifiers)) != len(identifiers):
                raise HarnessError("Goal physical turn range is ambiguous", kind="session_protocol")
            if identifiers:
                return identifiers
    return []


def _goal_result_field(result: Any, handle: Any, *names: str) -> Any:
    """从最终结果和 handle 读取固定 Goal 字段。"""

    for source in (result, handle):
        if source is None:
            continue
        value = _goal_field(source, *names)
        if value is not None:
            return value
    return None


def _write_goal_trace_event(
    trace_handle: Any,
    *,
    method: str,
    thread_id: str,
    logical_goal_id: str,
    turn_id: str | None = None,
    turn_status: str | None = None,
    goal_status: str | None = None,
    item_type: str | None = None,
    item_id: str | None = None,
) -> None:
    """只把 Goal 固定身份和状态写入 trace，禁止 provider 自由文本落盘。"""

    record = {
        "method": method,
        "thread_id": thread_id,
        "logical_goal_id": logical_goal_id,
        "turn_id": turn_id,
        "turn_status": turn_status,
        "goal_status": goal_status,
        "item_type": item_type,
        "item_id": item_id,
    }
    trace_handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    trace_handle.flush()


def _goal_request_logical_id(thread_id: str, request_digest: str) -> str:
    """根据 thread 与原始请求摘要生成跨进程稳定的本地逻辑 Goal 身份。"""

    digest = hashlib.sha256(f"{thread_id}:{request_digest}".encode("utf-8")).hexdigest()
    return f"logical_goal_{digest[:32]}"


def _goal_call_initial(thread: Any, prompt_text: str, token_budget: int | None) -> Any:
    """调用固定公开 Goal 入口，并始终传递可选预算字段。"""

    method = getattr(thread, "goal", None)
    if not callable(method):
        raise CodexGoalRecoveryPending(
            "Investigation thread has no Goal API",
            goal_status=None,
            logical_goal_id=None,
        )
    # 固定 vendored SDK 的公开签名包含 token_budget；缺少该参数时让调用
    # 立即失败并进入统一 recovery_pending，绝不改成无预算重试。
    return method(prompt_text, token_budget=token_budget)


def _build_goal_checkpoint(
    intent: Mapping[str, Any],
    turn_records: Sequence[Mapping[str, Any]],
    *,
    prompt_text: str,
    token_budget: int | None,
    provider_goal_id: str | None = None,
) -> GoalCheckpoint:
    """从本地 Goal 意图和物理 turn 账本构造严格恢复检查点。"""

    physical_turn_ids = tuple(record["turn_id"] for record in turn_records)
    values: dict[str, Any] = {
        "logical_turn_id": physical_turn_ids[0] if physical_turn_ids else None,
        "logical_goal_id": provider_goal_id or intent["goal_id"] or None,
        "thread_id": intent["thread_id"],
        "physical_turn_ids": physical_turn_ids,
        "objective": prompt_text,
        "token_budget": token_budget,
    }
    try:
        checkpoint_parameters = inspect.signature(GoalCheckpoint).parameters
    except (TypeError, ValueError):
        raise CodexGoalRecoveryPending(
            "Goal checkpoint API signature cannot be verified",
            goal_status="active",
            logical_goal_id=intent.get("logical_goal_id"),
            physical_turn_ids=physical_turn_ids,
        ) from None
    required_parameters = {"physical_turn_count_before", "token_budget"}
    if not required_parameters.issubset(checkpoint_parameters):
        raise CodexGoalRecoveryPending(
            "Goal checkpoint API lacks required physical baseline or token budget",
            goal_status="active",
            logical_goal_id=intent.get("logical_goal_id"),
            physical_turn_ids=physical_turn_ids,
        )
    values["physical_turn_count_before"] = intent["physical_turn_count_before"]
    return GoalCheckpoint(
        **values,
    )


def _goal_call_restore(
    thread: Any,
    checkpoint: GoalCheckpoint,
    *,
    sensitive_values: Sequence[str] = (),
) -> Any:
    """调用带检查点的只读 Goal 恢复入口，禁止 clear/set 或重新提交。"""

    method = getattr(thread, "resume_goal", None)
    if not callable(method):
        raise CodexGoalRecoveryPending(
            "existing Goal intent cannot be restored by this SDK surface",
            goal_status=None,
            logical_goal_id=None,
        )
    try:
        return method(checkpoint=checkpoint)
    except GoalNotFoundError:
        raise
    except BaseException as exc:
        raise _sanitize_sdk_error(
            "Goal checkpoint restore failed",
            exc,
            sensitive_values,
            kind="recovery_pending",
        ) from None


def _has_legacy_codex_recovery_records(root: Path | str) -> bool:
    """判断目录是否仍有 schema v3 普通 turn 的恢复记录。"""

    return bool(
        _load_completed_turn_receipts(root)
        or _read_completion_pending(root) is not None
        or _read_request_intent(root) is not None
        or _read_terminal_turn_journal(root) is not None
    )


def _goal_history_slice(
    thread: Any,
    *,
    thread_id: str,
    receipt: Mapping[str, Any],
    prompt_text: str,
) -> tuple[str, tuple[Any, ...]]:
    """按 receipt 的完整物理范围恢复首 prompt、末答案和全部 MCP 项。"""

    selected = _select_goal_history_range(
        thread,
        thread_id=thread_id,
        physical_turn_count_before=int(receipt["physical_turn_count_before"]),
        physical_turn_ids=tuple(receipt["physical_turn_ids"]),
        require_completed=True,
    )
    first_items = _read_protocol_field(selected[0], "items")
    first_prompts = [
        prompt
        for item in (first_items or [])
        for prompt in (_extract_persisted_user_prompt(item),)
        if prompt is not None
    ]
    if first_prompts != [prompt_text]:
        raise HarnessError("Goal receipt first physical turn prompt does not match", kind="session_protocol")
    last_items = _read_protocol_field(selected[-1], "items")
    final_answers = [
        answer
        for item in (last_items or [])
        for answer in (_extract_persisted_final_answer(item),)
        if answer is not None
    ]
    if len(final_answers) != 1:
        raise HarnessError("Goal receipt last physical turn has no unique final answer", kind="session_protocol")
    final_content = final_answers[0]
    completed_items = tuple(
        item
        for turn in selected
        for item in (_read_protocol_field(turn, "items") or [])
        if _protocol_item_type(item) == "mcpToolCall"
    )
    return final_content, completed_items


def _select_goal_history_range(
    thread: Any,
    *,
    thread_id: str,
    physical_turn_count_before: int,
    physical_turn_ids: Sequence[str],
    require_completed: bool,
) -> list[Any]:
    """从完整 thread history 选择 Goal 的连续物理 turn 范围。"""

    if (
        isinstance(physical_turn_count_before, bool)
        or not isinstance(physical_turn_count_before, int)
        or physical_turn_count_before < 0
    ):
        raise HarnessError("Goal physical turn baseline is invalid", kind="session_state")
    turns = _read_thread_turns(thread, expected_thread_id=thread_id)
    ids = _validate_unique_codex_history_turn_ids(turns)
    normalized_ids = [
        _validate_receipt_identifier(turn_id, "physical_turn_id")
        for turn_id in physical_turn_ids
    ]
    if not normalized_ids:
        raise HarnessError("Goal has no physical turn range", kind="session_protocol")
    if len(set(normalized_ids)) != len(normalized_ids):
        raise HarnessError("Goal physical turn range is ambiguous", kind="session_protocol")
    end = physical_turn_count_before + len(normalized_ids)
    if end != len(turns) or ids[physical_turn_count_before:end] != normalized_ids:
        raise HarnessError(
            "Goal physical range does not match contiguous thread history",
            kind="session_protocol",
        )
    selected = turns[physical_turn_count_before:end]
    if require_completed and any(
        _protocol_enum_value(_read_protocol_field(turn, "status")) != "completed"
        for turn in selected
    ):
        raise HarnessError(
            "Goal range contains a non-completed physical turn",
            kind="session_protocol",
        )
    return selected


def _read_goal_history_records(
    thread: Any,
    *,
    thread_id: str,
    physical_turn_count_before: int,
    physical_turn_ids: Sequence[str],
) -> tuple[list[Any], list[dict[str, Any]]]:
    """读取 Goal 连续物理范围及其有限状态事实。"""

    selected = _select_goal_history_range(
        thread,
        thread_id=thread_id,
        physical_turn_count_before=physical_turn_count_before,
        physical_turn_ids=physical_turn_ids,
        require_completed=False,
    )
    records: list[dict[str, Any]] = []
    for turn_id, turn in zip(physical_turn_ids, selected):
        raw_status = _read_protocol_field(turn, "status")
        status = _validate_goal_turn_status(raw_status)
        records.append({"turn_id": turn_id, "status": status})
    return selected, records


def _record_codex_goal_event(
    event: Any,
    *,
    trace_handle: Any,
    thread_id: str,
    logical_goal_id: str,
    sensitive_values: Sequence[str],
    on_tool_completed: Callable[[Any], None] | None,
    goal_status: str | None,
    physical_ids: list[str],
    turn_statuses: dict[str, str],
    completed_items: list[Any],
) -> tuple[str | None, str | None]:
    """处理一条 Goal 通知，只保存固定身份并转发真实工具完成事件。"""

    method = _goal_event_method(event)
    payload = _goal_event_payload(event)
    turn = _goal_event_turn(payload)
    turn_id: str | None = None
    turn_status: str | None = None
    if turn is not None:
        raw_turn_id = _goal_field(turn, "id", "turn_id")
        if isinstance(raw_turn_id, str) and raw_turn_id.strip():
            turn_id = _validate_receipt_identifier(raw_turn_id.strip(), "turn_id")
            if turn_id not in physical_ids:
                physical_ids.append(turn_id)
        raw_status = _goal_field(turn, "status")
        if raw_status is not None:
            turn_status = _validate_goal_turn_status(raw_status)
            if turn_id is not None:
                turn_statuses[turn_id] = turn_status
    item = _goal_event_item(payload)
    item_type = _protocol_item_type(item) if item is not None else None
    raw_item_id = _read_protocol_field(item, "id") if item is not None else None
    item_id = None
    if raw_item_id is not None:
        item_id = _validate_receipt_identifier(raw_item_id, "item_id")
    if item is not None and item_type == "mcpToolCall" and method.endswith("completed"):
        completed_items.append(item)
        if on_tool_completed is not None:
            try:
                on_tool_completed(item)
            except BaseException as exc:
                raise _sanitize_sdk_error(
                    "Goal tool completion callback failed",
                    exc,
                    sensitive_values,
                    kind="session_state",
                ) from None
    current_goal_status = goal_status
    goal = _goal_event_goal(payload)
    if goal is not None:
        raw_goal_status = _goal_field(goal, "status")
        if raw_goal_status is not None:
            current_goal_status = _validate_goal_status(raw_goal_status)
    _write_goal_trace_event(
        trace_handle,
        method=method,
        thread_id=thread_id,
        logical_goal_id=logical_goal_id,
        turn_id=turn_id,
        turn_status=turn_status,
        goal_status=current_goal_status,
        item_type=item_type if isinstance(item_type, str) else None,
        item_id=item_id,
    )
    final_content = None
    if item is not None and item_type == "agentMessage":
        phase = _protocol_enum_value(_read_protocol_field(item, "phase"))
        text = _read_protocol_field(item, "text")
        if phase == "final_answer" and isinstance(text, str) and text:
            if _value_contains_sensitive_text(text, sensitive_values):
                raise HarnessError(
                    "Goal final answer contains sensitive text",
                    kind="session_state",
                )
            final_content = text
    return current_goal_status, final_content


def _notify_goal_recovery_pending(
    callback: Callable[..., None] | None,
    turn_id: str | None,
) -> None:
    """保存 Goal 恢复阻断事实，并兼容旧的一参数回调。"""

    if callback is None:
        return
    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError):
        signature = None
    accepts_reason = False
    if signature is not None:
        accepts_reason = "reason" in signature.parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
    if accepts_reason:
        callback(turn_id, reason="goal")
    else:
        callback(turn_id)


def _consume_codex_goal_worker_events(
    *,
    result_queue: queue.Queue[tuple[str, Any]],
    worker: threading.Thread,
    request: HarnessRequest,
    trace_handle: Any,
    thread_id: str,
    logical_goal_id: str,
    sensitive_values: Sequence[str],
    idle_timeout_seconds: float,
    tool_idle_timeout_seconds: float,
    turn_handle_state: dict[str, Any],
    on_turn_started: Callable[[str], None] | None,
    on_tool_completed: Callable[[Any], None] | None,
    persist_turn_facts: Callable[[Any, Sequence[str], Mapping[str, str]], None] | None = None,
) -> tuple[Any, str | None, str | None, list[str], dict[str, str], tuple[Any, ...]]:
    """在后台 worker 中消费 Goal 全部通知，并执行普通 turn 的活动看门狗。"""

    deadline = (
        None
        if request.timeout_seconds is None
        else time.monotonic() + request.timeout_seconds
    )
    last_activity = time.monotonic()
    active_tool_ids: set[str] = set()
    process_to_item: dict[str, str] = {}
    anonymous_active_tool_count = 0
    known_async_job_ids: set[str] = set()
    pending_async_job_ids: set[str] = set()
    request_job_id = _read_async_job_field(request.metadata, "job_id")
    if isinstance(request_job_id, str) and request_job_id.strip():
        known_async_job_ids.add(request_job_id.strip())
    goal_status: str | None = None
    final_content: str | None = None
    physical_ids: list[str] = []
    turn_statuses: dict[str, str] = {}
    completed_items: list[Any] = []
    handle: Any = None
    # SDK 恢复窗口可能先返回没有 id 的句柄；等逻辑 stream 的首个 turn/started
    # 通知给出首个物理 turn 身份后，再只通知一次 Working Memory。
    turn_started_observed = False
    done = False
    while not done:
        now = time.monotonic()
        watchdog_timeout = (
            tool_idle_timeout_seconds
            if active_tool_ids or anonymous_active_tool_count or pending_async_job_ids
            else idle_timeout_seconds
        )
        remaining = watchdog_timeout - (now - last_activity)
        if remaining <= 0:
            raise HarnessError("Codex Goal event stream idle timeout", kind="transport")
        if deadline is not None:
            total_remaining = deadline - now
            if total_remaining <= 0:
                raise HarnessError("Codex Goal timed out", kind="transport")
            remaining = min(remaining, total_remaining)
        try:
            kind, value = result_queue.get(timeout=min(0.25, remaining))
        except queue.Empty:
            continue
        last_activity = time.monotonic()
        if kind == "started":
            handle = value
            turn_handle_state["value"] = value
            if persist_turn_facts is not None:
                # Goal 已被 provider 接受但尚无 TurnStarted 时，先保存可见的
                # provider Goal 身份；物理 turn 序列仍等真实通知逐步追加。
                persist_turn_facts(value, (), {})
            handle_id = _goal_handle_identifier(value)
            if not turn_started_observed and handle_id is not None:
                _notify_codex_turn_started(
                    value,
                    on_turn_started=on_turn_started,
                    sensitive_values=sensitive_values,
                )
                turn_started_observed = True
            continue
        if kind == "event":
            method = _goal_event_method(value)
            payload = _goal_event_payload(value)
            anonymous_active_tool_count = _track_codex_tool_activity(
                method=method,
                payload=payload,
                active_tool_ids=active_tool_ids,
                process_to_item=process_to_item,
                anonymous_active_tool_count=anonymous_active_tool_count,
            )
            goal_status, event_content = _record_codex_goal_event(
                value,
                trace_handle=trace_handle,
                thread_id=thread_id,
                logical_goal_id=logical_goal_id,
                sensitive_values=sensitive_values,
                on_tool_completed=on_tool_completed,
                goal_status=goal_status,
                physical_ids=physical_ids,
                turn_statuses=turn_statuses,
                completed_items=completed_items,
            )
            if not turn_started_observed and physical_ids:
                _notify_codex_turn_started(
                    SimpleNamespace(id=physical_ids[0]),
                    on_turn_started=on_turn_started,
                    sensitive_values=sensitive_values,
                )
                turn_started_observed = True
            if persist_turn_facts is not None:
                handle_ids = _goal_physical_ids(handle)
                persist_turn_facts(
                    handle,
                    handle_ids or physical_ids,
                    turn_statuses,
                )
            if event_content is not None:
                final_content = event_content
            if method.endswith("completed"):
                item = _goal_event_item(payload)
                if item is not None:
                    _update_async_job_wait_state_from_completed_item(
                        item,
                        known_job_ids=known_async_job_ids,
                        pending_job_ids=pending_async_job_ids,
                    )
                    anonymous_active_tool_count = _finish_codex_tool_item(
                        item=item,
                        active_tool_ids=active_tool_ids,
                        process_to_item=process_to_item,
                        anonymous_active_tool_count=anonymous_active_tool_count,
                    )
            continue
        if kind == "error":
            if isinstance(value, GoalNotFoundError):
                raise value
            raise _sanitize_sdk_error(
                "Codex Goal stream failed",
                value,
                sensitive_values,
                kind="transport",
            ) from None
        if kind == "done":
            handle = value or handle
            done = True
            continue
        raise HarnessError("Codex Goal worker returned an unknown event", kind="session_protocol")
    worker.join(timeout=1.0)
    if worker.is_alive():
        raise HarnessError("Codex Goal worker did not stop after completion", kind="transport")
    if handle is None:
        raise HarnessError("Codex Goal worker returned no handle", kind="session_protocol")
    if not turn_started_observed:
        raise HarnessError(
            "Codex Goal stream has no accepted physical turn identity",
            kind="session_protocol",
        )
    return handle, goal_status, final_content, physical_ids, turn_statuses, tuple(completed_items)


def _build_goal_result(
    *,
    request: HarnessRequest,
    trace_path: Path,
    profile: str,
    approval_mode: ApprovalMode,
    thread_id: str,
    logical_goal_id: str,
    goal_id: str,
    goal_status: str,
    physical_turn_ids: Sequence[str],
    content: str,
    request_digest: str,
    thread_policy_digest: str,
    completed_tool_items: Sequence[Any],
    replayed: bool,
    receipt_turn_count_before: int | None = None,
) -> CodexSdkRunResult:
    """构造 Investigation Goal 结果；turn_id 固定为首个物理 turn。"""

    first_turn_id = physical_turn_ids[0]
    return CodexSdkRunResult(
        response=HarnessResponse(
            content=content,
            finish_reason="stop",
            model=request.model,
            usage={},
            raw_trace_path=str(trace_path),
            metadata={
                "harness": request.metadata.get("harness"),
                "role": request.role,
                "stage": request.stage,
                "profile": profile,
                "approval_mode": approval_mode.value,
                "thread_id": thread_id,
                "turn_id": first_turn_id,
                "logical_goal_id": logical_goal_id,
                "goal_id": goal_id,
                "goal_status": goal_status,
                "physical_turn_start": physical_turn_ids[0],
                "physical_turn_end": physical_turn_ids[-1],
                "physical_turn_ids": list(physical_turn_ids),
                "replayed": replayed,
                "sdk": "openai-codex-0.149.0+arche.1",
            },
        ),
        thread_id=thread_id,
        turn_id=first_turn_id,
        replayed=replayed,
        receipt_turn_count_before=receipt_turn_count_before,
        request_digest=request_digest,
        thread_policy_digest=thread_policy_digest,
        completed_tool_items=tuple(completed_tool_items),
        logical_goal_id=logical_goal_id,
        goal_id=goal_id,
        goal_status=goal_status,
        physical_turn_ids=tuple(physical_turn_ids),
        physical_turn_start=physical_turn_ids[0],
        physical_turn_end=physical_turn_ids[-1],
    )


def _run_codex_goal_operation(
    *,
    thread: Any,
    request: HarnessRequest,
    trace_handle: Any,
    trace_path: Path,
    thread_id: str,
    profile: str,
    approval_mode: ApprovalMode,
    sensitive_values: Sequence[str],
    request_digest: str,
    turn_count_before: int,
    thread_policy_digest: str,
    completed_turn_receipt_root: Path | None,
    on_thread_ready: Callable[[str], None] | None,
    on_turn_started: Callable[[str], None] | None,
    on_tool_completed: Callable[[Any], None] | None,
    before_new_turn: Callable[[], None] | None = None,
    before_new_turn_with_thread: Callable[[Any, str], None] | None = None,
    on_recovery_pending: Callable[..., None] | None = None,
    on_goal_terminal_turn: Callable[[str, str, str], None] | None = None,
    idle_timeout_seconds: float = DEFAULT_INVESTIGATION_IDLE_TIMEOUT_SECONDS,
    tool_idle_timeout_seconds: float = DEFAULT_MCP_TOOL_IDLE_TIMEOUT_SECONDS,
    session_id: str | None = None,
) -> CodexSdkRunResult:
    """执行或恢复一个 Investigation Goal，并持久化完整物理 turn 范围。"""

    if profile != CODEX_PROFILE_INVESTIGATION:
        raise HarnessError("Goal operation is only available for Investigation", kind="provider_envelope")
    if completed_turn_receipt_root is None:
        raise HarnessError("Investigation Goal requires durable Goal 附加记录存储", kind="session_state")
    prompt_text = request.prompt_text()
    objective_digest = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
    root = _prepare_receipt_directory(completed_turn_receipt_root)
    intent: dict[str, Any] | None = None
    goal_turn_records: list[dict[str, Any]] = []

    def raise_goal_recovery(
        message: str,
        *,
        goal_status: str | None,
        physical_turn_ids: Sequence[str] = (),
        request_digest_value: str | None = request_digest,
        logical_goal_id_value: str | None = None,
    ) -> NoReturn:
        """记录 Goal 恢复阻断并抛出固定身份异常。"""

        pending = CodexGoalRecoveryPending(
            message,
            goal_status=goal_status,
            logical_goal_id=(
                logical_goal_id_value
                if logical_goal_id_value is not None
                else (intent or {}).get("logical_goal_id")
            ),
            physical_turn_ids=physical_turn_ids,
            request_digest=request_digest_value,
        )
        try:
            _notify_goal_recovery_pending(
                on_recovery_pending,
                pending.turn_id,
            )
        except BaseException:
            raise HarnessError(
                "Goal recovery fact cannot be saved",
                kind="recovery_pending",
            ) from None
        raise pending from None

    intent = _read_goal_intent(root)
    receipt = _read_goal_receipt(root)
    goal_turn_records = _read_goal_turn_journal(root)
    if intent is None and receipt is not None:
        if (
            receipt["thread_id"] != thread_id
            or receipt["request_digest"] != request_digest
            or receipt["objective_digest"] != objective_digest
            or receipt["thread_policy_digest"] != thread_policy_digest
        ):
            raise_goal_recovery(
                "Goal receipt identity does not match current request",
                goal_status=receipt.get("goal_status"),
                logical_goal_id_value=receipt.get("logical_goal_id"),
                physical_turn_ids=receipt.get("physical_turn_ids", ()),
                request_digest_value=receipt.get("request_digest"),
            )
        if goal_turn_records:
            journal_ids = [record["turn_id"] for record in goal_turn_records]
            if (
                journal_ids != receipt["physical_turn_ids"]
                or any(
                    record["thread_id"] != receipt["thread_id"]
                    or record["logical_goal_id"] != receipt["logical_goal_id"]
                    or record["goal_id"] != receipt["goal_id"]
                    for record in goal_turn_records
                )
            ):
                raise_goal_recovery(
                    "Goal receipt and physical journal do not identify one range",
                    goal_status=receipt.get("goal_status"),
                    logical_goal_id_value=receipt.get("logical_goal_id"),
                    physical_turn_ids=journal_ids,
                    request_digest_value=receipt.get("request_digest"),
                )
        content, completed_items = _goal_history_slice(
            thread,
            thread_id=thread_id,
            receipt=receipt,
            prompt_text=prompt_text,
        )
        _register_codex_thread(
            thread_id=thread_id,
            on_thread_ready=on_thread_ready,
            sensitive_values=sensitive_values,
        )
        return _build_goal_result(
            request=request,
            trace_path=trace_path,
            profile=profile,
            approval_mode=approval_mode,
            thread_id=thread_id,
            logical_goal_id=receipt["logical_goal_id"],
            goal_id=receipt["goal_id"],
            goal_status="complete",
            physical_turn_ids=receipt["physical_turn_ids"],
            content=content,
            request_digest=request_digest,
            thread_policy_digest=thread_policy_digest,
            completed_tool_items=completed_items,
            replayed=True,
            receipt_turn_count_before=receipt["turn_count_before"],
        )

    if intent is None and goal_turn_records:
        raise_goal_recovery(
            "Goal physical journal exists without its logical intent",
            goal_status=None,
            logical_goal_id_value=goal_turn_records[0]["logical_goal_id"],
            physical_turn_ids=[record["turn_id"] for record in goal_turn_records],
            request_digest_value=None,
        )
    journal_provider_goal_id = ""
    if intent is not None:
        journal_provider_goal_id = goal_turn_records[0]["goal_id"] if goal_turn_records else ""
        if any(
            record["thread_id"] != intent["thread_id"]
            or record["logical_goal_id"] != intent["logical_goal_id"]
            for record in goal_turn_records
        ):
            raise_goal_recovery(
                "Goal physical journal identity does not match its intent",
                goal_status=intent.get("status"),
                physical_turn_ids=[record["turn_id"] for record in goal_turn_records],
            )
        if intent["goal_id"] and journal_provider_goal_id not in {"", intent["goal_id"]}:
            raise_goal_recovery(
                "Goal physical journal provider identity does not match its intent",
                goal_status=intent.get("status"),
                physical_turn_ids=[record["turn_id"] for record in goal_turn_records],
            )
        expected = {
            "thread_id": thread_id,
            "objective_digest": objective_digest,
            "request_digest": request_digest,
            "turn_count_before": turn_count_before,
            "profile": profile,
            "stage": request.stage,
            "thread_policy_digest": thread_policy_digest,
        }
        if any(intent[field] != value for field, value in expected.items()):
            raise_goal_recovery(
                "Goal intent identity does not match current request",
                goal_status=intent.get("status"),
                logical_goal_id_value=intent.get("logical_goal_id"),
                request_digest_value=intent.get("request_digest"),
            )
        logical_goal_id = intent["logical_goal_id"]
        if receipt is not None:
            if (
                receipt["logical_goal_id"] != logical_goal_id
                or receipt["thread_id"] != thread_id
                or receipt["objective_digest"] != intent["objective_digest"]
                or receipt["request_digest"] != intent["request_digest"]
                or receipt["turn_count_before"] != intent["turn_count_before"]
                or receipt["physical_turn_count_before"]
                != intent["physical_turn_count_before"]
                or receipt["thread_policy_digest"] != intent["thread_policy_digest"]
            ):
                raise_goal_recovery(
                    "Goal receipt identity conflicts with intent",
                    goal_status=intent.get("status"),
                    logical_goal_id_value=logical_goal_id,
                    physical_turn_ids=receipt.get("physical_turn_ids", ()),
                    request_digest_value=receipt.get("request_digest"),
                )
            if intent["goal_id"] not in {"", receipt["goal_id"]}:
                raise_goal_recovery(
                    "Goal receipt provider identity conflicts with intent",
                    goal_status=intent.get("status"),
                    logical_goal_id_value=logical_goal_id,
                    physical_turn_ids=receipt.get("physical_turn_ids", ()),
                )
            if journal_provider_goal_id not in {"", receipt["goal_id"]}:
                raise_goal_recovery(
                    "Goal receipt provider identity conflicts with journal",
                    goal_status=intent.get("status"),
                    logical_goal_id_value=logical_goal_id,
                    physical_turn_ids=[record["turn_id"] for record in goal_turn_records],
                )
            if goal_turn_records and (
                [record["turn_id"] for record in goal_turn_records]
                != receipt["physical_turn_ids"]
                or any(record["status"] != "completed" for record in goal_turn_records)
            ):
                raise_goal_recovery(
                    "Goal receipt and physical journal range or status conflict",
                    goal_status=intent.get("status"),
                    logical_goal_id_value=logical_goal_id,
                    physical_turn_ids=[record["turn_id"] for record in goal_turn_records],
                )
            if intent["goal_id"] == "":
                intent = _update_goal_intent(root, goal_id=receipt["goal_id"])
                logical_goal_id = intent["logical_goal_id"]
            if goal_turn_records and journal_provider_goal_id == "":
                try:
                    _write_goal_turn_journal(
                        root,
                        thread_id=thread_id,
                        logical_goal_id=logical_goal_id,
                        goal_id=receipt["goal_id"],
                        turn_records=goal_turn_records,
                    )
                except BaseException:
                    raise_goal_recovery(
                        "Goal physical journal cannot bind its provider identity",
                        goal_status=intent.get("status"),
                        logical_goal_id_value=logical_goal_id,
                        physical_turn_ids=[record["turn_id"] for record in goal_turn_records],
                    )
            content, completed_items = _goal_history_slice(
                thread,
                thread_id=thread_id,
                receipt=receipt,
                prompt_text=prompt_text,
            )
            _register_codex_thread(
                thread_id=thread_id,
                on_thread_ready=on_thread_ready,
                sensitive_values=sensitive_values,
            )
            return _build_goal_result(
                request=request,
                trace_path=trace_path,
                profile=profile,
                approval_mode=approval_mode,
                thread_id=thread_id,
                logical_goal_id=logical_goal_id,
                goal_id=receipt["goal_id"],
                goal_status="complete",
                physical_turn_ids=receipt["physical_turn_ids"],
                content=content,
                request_digest=request_digest,
                thread_policy_digest=thread_policy_digest,
                completed_tool_items=completed_items,
                replayed=True,
                receipt_turn_count_before=receipt["turn_count_before"],
            )
        goal_turn_records = _read_goal_turn_journal(root)
        if intent["status"] == "complete":
            if not goal_turn_records:
                raise_goal_recovery(
                    "completed Goal intent has no physical turn journal",
                    goal_status="complete",
                )
            physical_ids = [record["turn_id"] for record in goal_turn_records]
            try:
                _read_goal_history_records(
                    thread,
                    thread_id=thread_id,
                    physical_turn_count_before=intent["physical_turn_count_before"],
                    physical_turn_ids=physical_ids,
                )
                content, completed_items = _goal_history_slice(
                    thread,
                    thread_id=thread_id,
                    receipt={
                        "physical_turn_count_before": intent["physical_turn_count_before"],
                        "physical_turn_ids": physical_ids,
                    },
                    prompt_text=prompt_text,
                )
            except BaseException:
                raise_goal_recovery(
                    "completed Goal history cannot be replayed",
                    goal_status="complete",
                    physical_turn_ids=physical_ids,
                )
            try:
                _write_goal_receipt(
                    root,
                    {
                        "schema_version": _GOAL_JOURNAL_SCHEMA_VERSION,
                        "thread_id": thread_id,
                        "logical_goal_id": logical_goal_id,
                        "goal_id": intent["goal_id"],
                        "objective_digest": intent["objective_digest"],
                        "request_digest": intent["request_digest"],
                        "turn_count_before": intent["turn_count_before"],
                        "physical_turn_count_before": intent["physical_turn_count_before"],
                        "physical_turn_ids": physical_ids,
                        "physical_turn_start": physical_ids[0],
                        "physical_turn_end": physical_ids[-1],
                        "goal_status": "complete",
                        "final_turn_status": "completed",
                        "thread_policy_digest": intent["thread_policy_digest"],
                    },
                )
            except BaseException:
                raise_goal_recovery(
                    "completed Goal receipt recovery is not durable",
                    goal_status="complete",
                    physical_turn_ids=physical_ids,
                )
            return _build_goal_result(
                request=request,
                trace_path=trace_path,
                profile=profile,
                approval_mode=approval_mode,
                thread_id=thread_id,
                logical_goal_id=logical_goal_id,
                goal_id=intent["goal_id"],
                goal_status="complete",
                physical_turn_ids=physical_ids,
                content=content,
                request_digest=request_digest,
                thread_policy_digest=thread_policy_digest,
                completed_tool_items=completed_items,
                replayed=True,
                receipt_turn_count_before=intent["turn_count_before"],
            )
        if not goal_turn_records:
            try:
                history = _read_thread_turns(thread, expected_thread_id=thread_id)
                baseline = intent["physical_turn_count_before"]
                history_ids = _validate_unique_codex_history_turn_ids(history)
                goal_turn_records = [
                    {
                        "turn_id": turn_id,
                        "status": _validate_goal_turn_status(
                            _read_protocol_field(turn, "status")
                        ),
                    }
                    for turn_id, turn in zip(history_ids[baseline:], history[baseline:])
                ]
            except BaseException:
                raise_goal_recovery(
                    "existing Goal physical history cannot be identified",
                    goal_status=intent.get("status"),
                )
        try:
            checkpoint = _build_goal_checkpoint(
                intent,
                goal_turn_records,
                prompt_text=prompt_text,
                token_budget=request.max_tokens,
                provider_goal_id=journal_provider_goal_id or None,
            )
        except CodexGoalRecoveryPending as exc:
            try:
                _notify_goal_recovery_pending(on_recovery_pending, exc.turn_id)
            except BaseException:
                raise HarnessError(
                    "Goal recovery fact cannot be saved",
                    kind="recovery_pending",
                ) from None
            raise
    else:
        logical_goal_id = _goal_request_logical_id(thread_id, request_digest)
        if session_id is None:
            physical_turn_count_before = 0
        else:
            history = _read_thread_turns(thread, expected_thread_id=thread_id)
            _select_completed_thread_history_turns(history)
            if len(_select_completed_thread_history_turns(history)) != turn_count_before:
                raise HarnessError(
                    "Goal thread successful history does not match session",
                    kind="session_state",
                )
            physical_turn_count_before = len(history)
        _run_codex_pre_turn_fence(
            before_new_turn,
            before_new_turn_with_thread=before_new_turn_with_thread,
            thread=thread,
            thread_id=thread_id,
            sensitive_values=sensitive_values,
        )
        _write_goal_intent(
            root,
            thread_id=thread_id,
            logical_goal_id=logical_goal_id,
            goal_id="",
            objective_digest=objective_digest,
            request_digest=request_digest,
            turn_count_before=turn_count_before,
            physical_turn_count_before=physical_turn_count_before,
            profile=profile,
            stage=request.stage,
            thread_policy_digest=thread_policy_digest,
            status="active",
        )
        intent = _read_goal_intent(root)
        if intent is None:
            raise_goal_recovery(
                "Goal intent disappeared after saving",
                goal_status="active",
                logical_goal_id_value=logical_goal_id,
                request_digest_value=request_digest,
            )
        checkpoint = None

    def persist_goal_turn_facts(
        handle: Any,
        observed_ids: Sequence[str],
        observed_statuses: Mapping[str, str],
    ) -> None:
        """每次通知后保存当前已知的物理 turn 前缀。"""

        provider_id = _goal_provider_identifier(handle)
        current_intent = _read_goal_intent(root)
        if current_intent is None:
            raise HarnessError("Goal intent disappeared while recording physical turn", kind="session_state")
        if provider_id:
            if current_intent["goal_id"] not in {"", provider_id}:
                raise HarnessError("Goal provider identity changed while recording physical turn", kind="session_state")
            if current_intent["goal_id"] != provider_id:
                _update_goal_intent(root, goal_id=provider_id)
        if not observed_ids:
            return
        normalized_ids = [
            _validate_receipt_identifier(turn_id, "physical_turn_id")
            for turn_id in observed_ids
        ]
        if len(set(normalized_ids)) != len(normalized_ids):
            raise HarnessError("Goal physical turn range is ambiguous", kind="session_protocol")
        records = [
            {
                "turn_id": turn_id,
                "status": observed_statuses.get(turn_id, "active"),
            }
            for turn_id in normalized_ids
        ]
        _write_goal_turn_journal(
            root,
            thread_id=thread_id,
            logical_goal_id=logical_goal_id,
            goal_id=provider_id,
            turn_records=records,
        )

    def consume_goal_worker(
        active_checkpoint: GoalCheckpoint | None,
    ) -> tuple[Any, str | None, str | None, list[str], dict[str, str], tuple[Any, ...]]:
        """启动一个 Goal worker 并在异常时有限中断它。"""

        result_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        worker = threading.Thread(
            target=_run_goal_stream,
            kwargs={
                "thread": thread,
                "prompt_text": prompt_text,
                "token_budget": request.max_tokens,
                "checkpoint": active_checkpoint,
                "result_queue": result_queue,
            },
            name="arche-codex-goal",
            daemon=True,
        )
        turn_handle_state: dict[str, Any] = {"value": None}
        worker.start()
        try:
            return _consume_codex_goal_worker_events(
                result_queue=result_queue,
                worker=worker,
                request=request,
                trace_handle=trace_handle,
                thread_id=thread_id,
                logical_goal_id=logical_goal_id,
                sensitive_values=sensitive_values,
                idle_timeout_seconds=idle_timeout_seconds,
                tool_idle_timeout_seconds=tool_idle_timeout_seconds,
                turn_handle_state=turn_handle_state,
                on_turn_started=on_turn_started,
                on_tool_completed=on_tool_completed,
                persist_turn_facts=persist_goal_turn_facts,
            )
        except BaseException as exc:
            handle_value = turn_handle_state.get("value")
            if _interrupt_and_join_worker(handle_value, worker):
                raise_goal_recovery(
                    "Goal worker could not be stopped",
                    goal_status=(intent or {}).get("status"),
                    logical_goal_id_value=logical_goal_id,
                )
            raise

    try:
        (
            handle,
            stream_goal_status,
            stream_content,
            stream_physical_ids,
            stream_turn_statuses,
            completed_items,
        ) = consume_goal_worker(checkpoint)
    except GoalNotFoundError:
        # 仅允许“本地意图已保存、provider 尚未接受 Goal、没有物理 turn”这一
        # 窗口重新执行首次 Goal；其它恢复错误不能猜测服务端状态。
        if not (
            intent is not None
            and intent["goal_id"] == ""
            and not goal_turn_records
        ):
            raise_goal_recovery(
                "provider Goal is missing for an existing operation",
                goal_status=(intent or {}).get("status"),
            )
        history = _read_thread_turns(thread, expected_thread_id=thread_id)
        if len(history) != intent["physical_turn_count_before"]:
            raise_goal_recovery(
                "provider Goal is missing after a physical turn was recorded",
                goal_status=intent.get("status"),
            )
        try:
            (
                handle,
                stream_goal_status,
                stream_content,
                stream_physical_ids,
                stream_turn_statuses,
                completed_items,
            ) = consume_goal_worker(None)
        except BaseException as exc:
            if isinstance(exc, CodexGoalRecoveryPending):
                try:
                    _notify_goal_recovery_pending(on_recovery_pending, exc.turn_id)
                except BaseException:
                    raise HarnessError(
                        "Goal recovery fact cannot be saved",
                        kind="recovery_pending",
                    ) from None
                raise
            raise_goal_recovery(
                "initial Goal retry failed before completion",
                goal_status=(intent or {}).get("status"),
            )
    except BaseException as exc:
        if isinstance(exc, CodexGoalRecoveryPending):
            try:
                _notify_goal_recovery_pending(on_recovery_pending, exc.turn_id)
            except BaseException:
                raise HarnessError(
                    "Goal recovery fact cannot be saved",
                    kind="recovery_pending",
                ) from None
            raise
        raise_goal_recovery(
            "Goal stream failed before completion",
            goal_status=(intent or {}).get("status"),
        )
    current_intent = _read_goal_intent(root)
    if current_intent is None:
        raise_goal_recovery(
            "Goal intent disappeared before physical turn start",
            goal_status=None,
            logical_goal_id_value=logical_goal_id,
            request_digest_value=request_digest,
        )
    provider_goal_id = _goal_provider_identifier(handle)
    if not provider_goal_id:
        raise_goal_recovery(
            "Goal provider identity is missing",
            goal_status=stream_goal_status,
            physical_turn_ids=stream_physical_ids,
        )
    if current_intent["goal_id"] not in {"", provider_goal_id}:
        raise_goal_recovery(
            "Goal provider identity changed during recovery",
            goal_status=stream_goal_status,
            physical_turn_ids=stream_physical_ids,
        )
    if current_intent["goal_id"] != provider_goal_id:
        current_intent = _update_goal_intent(root, goal_id=provider_goal_id)
    first_turn_id = _goal_handle_identifier(handle)
    goal_status_raw = _goal_result_field(handle, None, "goal_status", "status", "provider_status")
    goal_status = (
        _validate_goal_status(goal_status_raw)
        if goal_status_raw is not None
        else stream_goal_status
    )
    if goal_status is None:
        raise_goal_recovery(
            "Goal stream has no terminal status",
            goal_status=None,
            physical_turn_ids=stream_physical_ids,
        )
    history = _read_thread_turns(thread, expected_thread_id=thread_id)
    history_ids = _validate_unique_codex_history_turn_ids(history)
    baseline = current_intent["physical_turn_count_before"]
    if baseline > len(history):
        raise_goal_recovery(
            "Goal thread history is shorter than its saved physical baseline",
            goal_status=goal_status,
        )
    physical_ids = history_ids[baseline:]
    advertised_ids = _goal_physical_ids(handle)
    if advertised_ids and advertised_ids != physical_ids:
        raise_goal_recovery(
            "Goal handle physical range does not match thread history",
            goal_status=goal_status,
            physical_turn_ids=physical_ids,
        )
    if stream_physical_ids and stream_physical_ids != physical_ids[: len(stream_physical_ids)]:
        raise_goal_recovery(
            "Goal logical stream identity is not an ordered history prefix",
            goal_status=goal_status,
            physical_turn_ids=physical_ids,
        )
    if first_turn_id is not None and (not physical_ids or first_turn_id != physical_ids[0]):
        raise_goal_recovery(
            "Goal handle id is not the first physical turn",
            goal_status=goal_status,
            physical_turn_ids=physical_ids,
        )
    if not physical_ids:
        raise_goal_recovery(
            "Goal has no physical turn range",
            goal_status=goal_status,
        )
    try:
        _selected_history, turn_records = _read_goal_history_records(
            thread,
            thread_id=thread_id,
            physical_turn_count_before=baseline,
            physical_turn_ids=physical_ids,
        )
        _write_goal_turn_journal(
            root,
            thread_id=thread_id,
            logical_goal_id=logical_goal_id,
            goal_id=provider_goal_id,
            turn_records=turn_records,
        )
    except BaseException:
        raise_goal_recovery(
            "Goal physical history cannot be durably recorded",
            goal_status=goal_status,
            physical_turn_ids=physical_ids,
        )
    final_turn_status_raw = _goal_result_field(handle, None, "final_turn_status", "last_turn_status")
    final_turn_status = (
        _validate_goal_turn_status(final_turn_status_raw)
        if final_turn_status_raw is not None
        else turn_records[-1]["status"]
    )
    _update_goal_intent(root, goal_id=provider_goal_id, status=goal_status)
    if goal_status != "complete" or final_turn_status != "completed":
        if final_turn_status in {"failed", "interrupted"} and on_goal_terminal_turn is not None:
            try:
                # Goal 的 Working Memory attempt 使用首个物理 turn 身份；末个
                # 物理 turn 只保留在 Goal journal 中，不能拿来关闭首个 Started。
                on_goal_terminal_turn(thread_id, physical_ids[0], final_turn_status)
            except BaseException:
                raise_goal_recovery(
                    "Goal terminal turn acknowledgement failed",
                    goal_status=goal_status,
                    physical_turn_ids=physical_ids,
                )
        raise_goal_recovery(
            "Goal did not complete with a successful final physical turn",
            goal_status=goal_status,
            physical_turn_ids=physical_ids,
        )
    try:
        content, history_completed_items = _goal_history_slice(
            thread,
            thread_id=thread_id,
            receipt={
                "physical_turn_count_before": baseline,
                "physical_turn_ids": physical_ids,
            },
            prompt_text=prompt_text,
        )
        if _value_contains_sensitive_text(content, sensitive_values):
            raise HarnessError(
                "Goal final answer contains sensitive text",
                kind="session_state",
            )
        _write_goal_receipt(
            root,
            {
                "schema_version": _GOAL_JOURNAL_SCHEMA_VERSION,
                "thread_id": thread_id,
                "logical_goal_id": logical_goal_id,
                "goal_id": provider_goal_id,
                "objective_digest": objective_digest,
                "request_digest": request_digest,
                "turn_count_before": turn_count_before,
                "physical_turn_count_before": baseline,
                "physical_turn_ids": physical_ids,
                "physical_turn_start": physical_ids[0],
                "physical_turn_end": physical_ids[-1],
                "goal_status": goal_status,
                "final_turn_status": final_turn_status,
                "thread_policy_digest": thread_policy_digest,
            },
        )
    except BaseException:
        raise_goal_recovery(
            "completed Goal receipt cannot be durably saved",
            goal_status=goal_status,
            physical_turn_ids=physical_ids,
        )
    return _build_goal_result(
        request=request,
        trace_path=trace_path,
        profile=profile,
        approval_mode=approval_mode,
        thread_id=thread_id,
        logical_goal_id=logical_goal_id,
        goal_id=provider_goal_id,
        goal_status=goal_status,
        physical_turn_ids=physical_ids,
        content=content,
        request_digest=request_digest,
        thread_policy_digest=thread_policy_digest,
        completed_tool_items=history_completed_items or completed_items,
        replayed=False,
    )


def run_codex_sdk_turn(
    *,
    config: CodexConfig,
    request: HarnessRequest,
    thread_config: Mapping[str, Any],
    session_id: str | None,
    trace_path: Path,
    provider_id: str = "arche_codex",
    env_key: str = "ARCHE_CODEX_API_KEY",
    on_thread_ready: Callable[[str], None] | None = None,
    trace_sensitive_values: Sequence[str] = (),
    secret_environment: Mapping[str, str] | None = None,
    completed_turn_receipt_root: Path | None = None,
    request_digest: str | None = None,
    turn_count_before: int | None = None,
    thread_policy_digest: str | None = None,
    before_new_turn: Callable[[], None] | None = None,
    before_new_turn_with_thread: Callable[[Any, str], None] | None = None,
    idle_timeout_seconds: float | None = None,
    tool_idle_timeout_seconds: float = DEFAULT_MCP_TOOL_IDLE_TIMEOUT_SECONDS,
    on_recovery_pending: Callable[[str | None], None] | None = None,
    on_turn_started: Callable[[str], None] | None = None,
    on_terminal_turn: Callable[[str, str, str], None] | None = None,
    on_goal_terminal_turn: Callable[[str, str, str], None] | None = None,
    on_investigation_rollback: Callable[[str, Any, str], None] | None = None,
    # 兼容旧的 wake-only 调用方；内部统一按 Investigation turn 处理。
    on_wake_rollback: Callable[[str, Any, str], None] | None = None,
    on_tool_completed: Callable[[Any], None] | None = None,
    completion_recovery: Mapping[str, Any] | None = None,
    auth_mode: str = "api",
    permission_profile: str | None = None,
    recovery_only: bool = False,
    recovery_turn_id: str | None = None,
    replay_turn_id: str | None = None,
    enable_investigation_goals: bool | None = None,
) -> CodexSdkRunResult:
    """通过 app-server SDK 完成一次新建或恢复的 Codex turn。

    ``provider_id`` 和 ``env_key`` 由调用方显式绑定到 thread provider；适配器不
    读取或记录环境变量值。``on_thread_ready`` 在 thread 已由 app-server 返回、但
    尚未提交真实 turn 时调用，供调用方原子保存可恢复的 thread。回调失败会直接
    终止本次调用，因此不会启动 turn。``trace_sensitive_values`` 是调用方额外
    提供的敏感值集合；它们只用于写 trace 前脱敏，不会被写入事件或错误。
    ``secret_environment`` 是仅供 app-server 使用的环境变量值；变量名会加入
    shell 排除清单，值不会进入 thread config、trace、CODEX_HOME 或异常正文。
    ``completed_turn_receipt_root``、``request_digest``、``turn_count_before`` 和
    ``thread_policy_digest`` 一起启用 completed-turn receipt：turn/completed 到达时只原子保存 thread、turn
    和 prompt 摘要；恢复时先读取同一 thread 的完整 history，匹配成功则重放而不
    再启动模型 turn。``before_new_turn`` 只在没有匹配 receipt、即将启动真正
    worker 前调用；回调失败会在任何模型 turn 启动前直接阻断本次调用。
    ``idle_timeout_seconds`` 只监视普通 SDK 事件流两次事件之间的空闲时间（Investigation
    默认 1800 秒，Task Review 默认 300 秒），不限制 Investigation 的科学作业总时长。
    调用方显式传入的值始终优先。收到 ``mcpToolCall`` 或
    ``commandExecution`` 的 ``item/started``，对应的 progress/output delta，或
    同一 turn 中结构化回执确认仍在运行的同一异步作业后，该次工具/作业等待使用
    ``tool_idle_timeout_seconds``（默认与 MCP ``tool_timeout_sec=21600`` 一致）。
    明确终态或 collect 会恢复普通阈值；因此合法的长时 MiniChem 不会被普通传输
    阈值误杀，失联工具或真实静默模型仍有有限上限。超时或其它异常都会请求 turn
    interrupt 并有限等待 worker；若仍无法停止则抛出
    ``recovery_pending``，由上层持久阻断同一会话的新 turn。
    ``on_recovery_pending`` 在抛出该错误前保存恢复记录，这样进程随后崩溃时也
    不会丢失“worker 尚未确认停止”的状态；回调失败则继续 fail-closed，不把本次
    调用当作可重试。``completion_recovery`` 只由上层传入已经持久化的
    ``completion_receipt`` 或 ``terminal_turn`` 状态，用于严格读取 history、
    补写对应 journal，不会因此重新启动模型。``on_turn_started`` 只在
    ``thread.turn`` 已返回带真实身份的
    SDK turn 句柄后调用；回调失败会中断该 turn 并 fail-closed，completed receipt
    重放不会重复调用它。failed/interrupted 终态只写固定身份字段；``on_terminal_turn``
    必须同步、幂等地保存上层终态事实。terminal journal 恢复会先用同一 turn_id
    重放 ``on_turn_started``，再调用 ``on_terminal_turn``；仅在两者都成功后，adapter
    才删除 intent 与 terminal journal，恢复过程不启动新 turn。
    ``recovery_only`` 是显式的两阶段恢复入口：它只允许在已有 session 上
    ``thread_resume``，从旧 request-intent 和同一 thread history 恢复唯一的
    failed/interrupted turn，完成 terminal ack 后以 ``CodexTerminalTurnError``
    结束；该分支绝不调用 ``thread.turn``，也不接受新的 wake 请求摘要。调用方
    必须在下一次独立调用中传入已经持久化的 wake intent。
    ``replay_turn_id`` 由 waiting sidecar 提供；存在时优先按同一 thread/turn
    读取 completed receipt，允许当前 job replacement 造成 request digest 漂移。
    ``enable_investigation_goals`` 必须与上层冻结的 thread policy 一致；只有显式
    为 ``True`` 时才进入实验性 Goal 路径。省略或为 ``False`` 的新 Investigation
    使用普通 ``thread.turn``；已经冻结为 Goal 的 session 恢复时仍显式传入
    ``True``，不得在恢复中途切换运行方式。
    """

    effective_idle_timeout = (
        _default_idle_timeout_for_role(request.role)
        if idle_timeout_seconds is None
        else idle_timeout_seconds
    )
    try:
        (
            provider_id,
            env_key,
            normalized_idle_timeout,
            normalized_tool_idle_timeout,
            normalized_secret_environment,
            sensitive_values,
        ) = _validate_codex_turn_parameters(
            provider_id=provider_id,
            env_key=env_key,
            trace_sensitive_values=trace_sensitive_values,
            secret_environment=secret_environment,
            completed_turn_receipt_root=completed_turn_receipt_root,
            request_digest=request_digest,
            turn_count_before=turn_count_before,
            session_id=session_id,
            completion_recovery=completion_recovery,
            idle_timeout_seconds=effective_idle_timeout,
            tool_idle_timeout_seconds=tool_idle_timeout_seconds,
        )
    except CodexGoalRecoveryPending as exc:
        try:
            _notify_goal_recovery_pending(on_recovery_pending, exc.turn_id)
        except BaseException:
            raise HarnessError(
                "Goal recovery fact cannot be saved",
                kind="recovery_pending",
            ) from None
        raise
    if on_terminal_turn is not None and not callable(on_terminal_turn):
        raise TypeError("on_terminal_turn must be callable or None")
    if on_goal_terminal_turn is not None and not callable(on_goal_terminal_turn):
        raise TypeError("on_goal_terminal_turn must be callable or None")
    if not isinstance(recovery_only, bool):
        raise TypeError("recovery_only must be bool")
    if enable_investigation_goals is not None and not isinstance(
        enable_investigation_goals,
        bool,
    ):
        raise TypeError("enable_investigation_goals must be bool or None")
    if recovery_only and session_id is None:
        raise HarnessError(
            "Codex recovery-only turn requires an existing session",
            kind="session_state",
        )
    if recovery_only and before_new_turn is not None:
        raise HarnessError(
            "Codex recovery-only turn cannot run a new-turn fence",
            kind="session_state",
        )
    if recovery_turn_id is not None:
        recovery_turn_id = _validate_receipt_identifier(
            recovery_turn_id,
            "recovery_turn_id",
        )
    if recovery_turn_id is not None and not recovery_only:
        raise HarnessError(
            "recovery_turn_id requires recovery-only mode",
            kind="session_state",
        )
    if replay_turn_id is not None:
        replay_turn_id = _validate_receipt_identifier(replay_turn_id, "replay_turn_id")
    if replay_turn_id is not None and recovery_only:
        raise HarnessError(
            "replay_turn_id cannot be combined with recovery-only mode",
            kind="session_state",
        )
    if before_new_turn_with_thread is not None and not callable(before_new_turn_with_thread):
        raise TypeError("before_new_turn_with_thread must be callable or None")
    if on_investigation_rollback is not None and on_wake_rollback is not None:
        raise TypeError(
            "on_investigation_rollback and on_wake_rollback cannot both be provided"
        )
    if on_wake_rollback is not None:
        on_investigation_rollback = on_wake_rollback
    if on_investigation_rollback is not None and not callable(on_investigation_rollback):
        raise TypeError("on_investigation_rollback must be callable or None")
    if on_tool_completed is not None and not callable(on_tool_completed):
        raise TypeError("on_tool_completed must be callable or None")
    request.workspace.mkdir(parents=True, exist_ok=True)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(request.workspace, 0o700)
    os.chmod(trace_path.parent, 0o700)
    trace_handle = trace_path.open("w", encoding="utf-8")
    os.chmod(trace_path, 0o600)
    codex: Codex | None = None
    prompt_text = request.prompt_text()
    try:
        startup_error: BaseException | None = None
        try:
            codex = Codex(config)
            startup = _start_codex_thread(
                codex=codex,
                request=request,
                thread_config=thread_config,
                session_id=session_id,
                provider_id=provider_id,
                env_key=env_key,
                normalized_secret_environment=normalized_secret_environment,
                sensitive_values=sensitive_values,
                expected_thread_policy_digest=thread_policy_digest,
                auth_mode=auth_mode,
                permission_profile=permission_profile,
                enable_investigation_goals=enable_investigation_goals,
            )
        except BaseException as exc:
            startup_error = _sanitize_sdk_error(
                "Codex SDK startup failed",
                exc,
                sensitive_values,
                kind="transport",
            )
        if startup_error is not None:
            raise startup_error from None

        thread = startup.thread
        profile = startup.profile
        thread_policy_digest = startup.thread_policy_digest
        trace_values = startup.trace_values
        approval_mode = startup.approval_mode
        thread_id = str(getattr(thread, "id", "")).strip()
        if not thread_id:
            raise HarnessError("Codex SDK returned no thread id", kind="session_protocol")
        if session_id is not None and thread_id != session_id:
            raise HarnessError(
                "Codex resume returned a different thread_id",
                kind="session_protocol",
            )
        goal_request = (
            enable_investigation_goals is True
            and profile == CODEX_PROFILE_INVESTIGATION
            and request.metadata.get("request_kind") in _INVESTIGATION_GOAL_REQUEST_KINDS
        )
        goal_intent = (
            _read_goal_intent(completed_turn_receipt_root)
            if goal_request and completed_turn_receipt_root is not None
            else None
        )
        goal_receipt = (
            _read_goal_receipt(completed_turn_receipt_root)
            if goal_request and completed_turn_receipt_root is not None
            else None
        )
        legacy_recovery_records = (
            _has_legacy_codex_recovery_records(completed_turn_receipt_root)
            if goal_request and completed_turn_receipt_root is not None
            else False
        )
        if (
            goal_request
            and not legacy_recovery_records
            and request.metadata.get("request_kind")
            in {"investigation_job_wake", "investigation_job_continuation"}
            and session_id is not None
            and goal_intent is None
            and goal_receipt is None
        ):
            # 已完成旧 schema v3 的 interrupted/failed 收束后，历史仍保留旧
            # 普通 turn 事实；该 wake 必须沿用普通 turn 恢复语义。
            legacy_history = _read_thread_turns(thread, expected_thread_id=thread_id)
            if legacy_history:
                last_status = _protocol_enum_value(
                    _read_protocol_field(legacy_history[-1], "status")
                )
                legacy_recovery_records = last_status in {"failed", "interrupted"}
        if goal_request and (goal_intent is not None or goal_receipt is not None) and legacy_recovery_records:
            raise HarnessError(
                "Goal records conflict with schema v3 ordinary turn recovery records",
                kind="session_state",
            )
        goal_request = goal_request and not legacy_recovery_records
        if goal_request:
            # Investigation v2 的四类调用统一走 Goal；旧的普通 turn 请求不带
            # request_kind，因此继续走下方 schema v3 普通 turn 恢复路径。
            _register_codex_thread(
                thread_id=thread_id,
                on_thread_ready=on_thread_ready,
                sensitive_values=sensitive_values,
            )
            return _run_codex_goal_operation(
                thread=thread,
                request=request,
                trace_handle=trace_handle,
                trace_path=trace_path,
                thread_id=thread_id,
                profile=profile,
                approval_mode=approval_mode,
                sensitive_values=trace_values,
                request_digest=request_digest or "",
                turn_count_before=turn_count_before if turn_count_before is not None else 0,
                thread_policy_digest=thread_policy_digest or "",
                completed_turn_receipt_root=completed_turn_receipt_root,
                on_thread_ready=None,
                on_turn_started=on_turn_started,
                on_tool_completed=on_tool_completed,
                before_new_turn=before_new_turn,
                before_new_turn_with_thread=before_new_turn_with_thread,
                on_recovery_pending=on_recovery_pending,
                on_goal_terminal_turn=on_goal_terminal_turn or on_terminal_turn,
                idle_timeout_seconds=normalized_idle_timeout,
                tool_idle_timeout_seconds=normalized_tool_idle_timeout,
                session_id=session_id,
            )
        if replay_turn_id is None:
            terminal_journal = _prepare_terminal_turn_replay(
                thread=thread,
                thread_id=thread_id,
                request=request,
                completed_turn_receipt_root=completed_turn_receipt_root,
                request_digest=request_digest or "",
                turn_count_before=turn_count_before if turn_count_before is not None else 0,
                provider_id=provider_id,
                env_key=env_key,
                profile=profile,
                thread_policy_digest=thread_policy_digest,
                completion_recovery=completion_recovery,
                recovery_only=recovery_only,
            )
        else:
            # waiting receipt 已经固定本次要重放的 completed turn；旧
            # request-intent 可能来自更早的物理基线，不能把漂移的当前
            # prompt 误送入 terminal-turn migration。其它 terminal journal
            # 仍需先走原 turn 的确认流程，不能与 waiting replay 混用。
            if completed_turn_receipt_root is not None and _read_terminal_turn_journal(
                completed_turn_receipt_root
            ) is not None:
                raise HarnessError(
                    "waiting replay conflicts with a pending terminal turn",
                    kind="session_state",
                )
            terminal_journal = None
        if recovery_only:
            if completed_turn_receipt_root is None:
                raise HarnessError(
                    "Codex recovery-only turn requires durable recovery storage",
                    kind="session_state",
                )
            if terminal_journal is None:
                raise CodexRequestIntentRecoveryPending(
                    "Codex recovery-only turn has no unique terminal history",
                    turn_id=None,
                    reason="request_intent",
                    request_digest=request_digest,
                )
            if (
                recovery_turn_id is not None
                and terminal_journal["turn_id"] != recovery_turn_id
            ):
                raise CodexRequestIntentRecoveryPending(
                    "Codex recovery-only turn does not match waiting turn",
                    turn_id=None,
                    reason="request_intent",
                    request_digest=request_digest,
                )
            _register_codex_thread(
                thread_id=thread_id,
                on_thread_ready=on_thread_ready,
                sensitive_values=sensitive_values,
            )
            _acknowledge_terminal_codex_turn(
                terminal_journal,
                thread=thread,
                trace_handle=trace_handle,
                completed_turn_receipt_root=completed_turn_receipt_root,
                on_turn_started=on_turn_started,
                on_terminal_turn=on_terminal_turn,
                on_investigation_rollback=on_investigation_rollback,
            )
        if terminal_journal is not None:
            _register_codex_thread(
                thread_id=thread_id,
                on_thread_ready=on_thread_ready,
                sensitive_values=sensitive_values,
            )
            _acknowledge_terminal_codex_turn(
                terminal_journal,
                thread=thread,
                trace_handle=trace_handle,
                completed_turn_receipt_root=completed_turn_receipt_root,
                on_turn_started=on_turn_started,
                on_terminal_turn=on_terminal_turn,
                on_investigation_rollback=on_investigation_rollback,
            )
        replay_state = _prepare_codex_replay_state(
            thread=thread,
            thread_id=thread_id,
            request=request,
            prompt_text=prompt_text,
            session_id=session_id,
            completed_turn_receipt_root=completed_turn_receipt_root,
            request_digest=request_digest or "",
            turn_count_before=turn_count_before if turn_count_before is not None else 0,
            provider_id=provider_id,
            env_key=env_key,
            profile=profile,
            thread_policy_digest=thread_policy_digest,
            completion_recovery=completion_recovery,
            replay_turn_id=replay_turn_id,
        )
        _register_codex_thread(
            thread_id=thread_id,
            on_thread_ready=on_thread_ready,
            sensitive_values=sensitive_values,
        )
        if replay_state.replay_receipt is not None and replay_state.replay_content is not None:
            if completed_turn_receipt_root is None:
                raise HarnessError(
                    "replayed Codex receipt has no persistence root",
                    kind="session_state",
                )
            _persist_replayed_receipt(
                completed_turn_receipt_root,
                replay_state.replay_receipt,
            )
            return _build_replayed_codex_result(
                request=request,
                trace_path=trace_path,
                profile=profile,
                approval_mode=approval_mode,
                thread_id=thread_id,
                receipt=replay_state.replay_receipt,
                content=replay_state.replay_content,
                request_digest=request_digest or "",
                thread_policy_digest=thread_policy_digest,
                completed_tool_items=replay_state.replay_completed_tool_items,
            )
        _run_codex_pre_turn_fence(
            before_new_turn,
            before_new_turn_with_thread=before_new_turn_with_thread,
            thread=thread,
            thread_id=thread_id,
            sensitive_values=sensitive_values,
        )
        _persist_codex_request_intent(
            completed_turn_receipt_root,
            thread=thread,
            thread_id=thread_id,
            session_id=session_id,
            request_digest=request_digest or "",
            turn_count_before=turn_count_before if turn_count_before is not None else 0,
            provider_id=provider_id,
            env_key=env_key,
            request=request,
            profile=profile,
            thread_policy_digest=thread_policy_digest,
        )
        return _run_codex_turn_worker(
            thread=thread,
            prompt_text=prompt_text,
            request=request,
            trace_handle=trace_handle,
            trace_path=trace_path,
            thread_id=thread_id,
            profile=profile,
            approval_mode=approval_mode,
            sensitive_values=trace_values,
            request_digest=request_digest or "",
            turn_count_before=turn_count_before if turn_count_before is not None else 0,
            thread_policy_digest=thread_policy_digest,
            completed_turn_receipt_root=completed_turn_receipt_root,
            idle_timeout_seconds=normalized_idle_timeout,
            tool_idle_timeout_seconds=normalized_tool_idle_timeout,
            on_recovery_pending=on_recovery_pending,
            on_turn_started=on_turn_started,
            on_terminal_turn=on_terminal_turn,
            on_investigation_rollback=on_investigation_rollback,
            on_tool_completed=on_tool_completed,
        )
    finally:
        trace_handle.close()
        if codex is not None:
            codex.close()


__all__ = [
    "CODEX_PROFILE_LEGACY_PLANNER",
    "CODEX_PROFILE_INVESTIGATION",
    "CODEX_PROFILE_REVIEW",
    "CODEX_PROFILE_TASK_REVIEW",
    "CODEX_TERMINAL_PROVIDER_ERROR_CODES",
    "DEFAULT_CODEX_IDLE_TIMEOUT_SECONDS",
    "DEFAULT_INVESTIGATION_IDLE_TIMEOUT_SECONDS",
    "DEFAULT_MCP_TOOL_IDLE_TIMEOUT_SECONDS",
    "CodexCompletionRecoveryPending",
    "CodexCompletedTurnReceiptHandle",
    "CodexGoalReceiptHandle",
    "CodexGoalRecoveryPending",
    "CodexRequestIntentRecoveryPending",
    "CodexRecoveryPending",
    "CodexSdkRunResult",
    "CodexTerminalTurnError",
    "CodexTerminalTurnRecoveryPending",
    "CodexWorkerRecoveryPending",
    "CodexProcessSandboxSpec",
    "approval_mode_for_codex_profile",
    "sandbox_for_codex_profile",
    "build_codex_config",
    "build_codex_capability_config",
    "calculate_codex_request_digest",
    "calculate_codex_thread_policy_digest",
    "acknowledge_codex_completed_turn_receipt",
    "build_codex_completed_turn_receipt_handle",
    "build_codex_goal_receipt_handle",
    "resolve_codex_completed_turn_receipt_handle",
    "resolve_codex_goal_receipt_handle",
    "find_codex_completed_turn_receipt",
    "read_codex_completed_turn_receipt",
    "validate_codex_completed_turn_receipt",
    "migrate_codex_v2_request_intent_to_terminal_turn",
    "read_codex_request_intent",
    "read_codex_goal_receipt",
    "config_values_to_object",
    "confirm_codex_request_intent_without_new_turn",
    "normalize_codex_profile",
    "resolve_codex_profile",
    "rollback_codex_thread_for_investigation_turn",
    "rollback_codex_thread_for_investigation_wake",
    "run_codex_sdk_turn",
    "write_codex_prompt_artifact",
]
