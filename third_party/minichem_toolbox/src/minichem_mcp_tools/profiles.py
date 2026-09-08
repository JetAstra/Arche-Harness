"""MiniChem MCP 工具 profile 的显式实例级注册清单。"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
from copy import deepcopy
from typing import Any, Callable

from minichem_toolbox.catalog import CatalogFreezeError, public_catalog_snapshot


CATALOG_BOOTSTRAP_TOOL_NAMES = (
    "inspect_action",
    "list_software",
    "inspect_software",
    "list_analysis_runtimes",
)
CATALOG_BOOTSTRAP_RESOURCE_URIS = ("minichem://catalog",)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def resolve_server_profile(value: str | None) -> str:
    """解析服务 profile；非法名称直接失败，不回退到完整服务。"""

    normalized = str(value or "full").strip().lower().replace("-", "_")
    if normalized in {"full", "legacy", "progressive"}:
        return "legacy"
    if normalized == "investigation":
        return "investigation"
    if normalized == "catalog_bootstrap":
        return "catalog_bootstrap"
    raise ValueError(
        "Unsupported MiniChem server profile "
        f"{value!r}; choose full, investigation, or catalog_bootstrap"
    )


def normalize_loopback_host(value: str | None) -> str:
    """将服务监听地址规范化为 loopback IP；空值、通配地址和远程地址直接拒绝。

    输入是 CLI 或启动脚本传入的主机名，不执行 DNS 查询，也不接受带端口的地址。
    返回值是可直接交给 Uvicorn 的 ``127.0.0.1`` 或规范化 IPv6 loopback 地址；
    ``localhost`` 映射为 IPv4 loopback。解析失败时抛出 ``ValueError``，避免 profile
    因主机名解析结果变化而意外监听远程接口。
    """

    candidate = str(value or "").strip().lower()
    if candidate.startswith("[") and candidate.endswith("]"):
        candidate = candidate[1:-1].strip()
    if candidate == "localhost":
        return "127.0.0.1"
    if not candidate or "%" in candidate:
        raise ValueError("loopback host must be 127.0.0.1, localhost, or ::1")
    try:
        address = ipaddress.ip_address(candidate)
    except ValueError as exc:
        raise ValueError(
            "MiniChem profiles only accept loopback host "
            "127.0.0.1, localhost, or ::1"
        ) from exc
    if not address.is_loopback:
        raise ValueError(
            "MiniChem profiles reject non-loopback host"
        )
    return address.compressed


def validate_catalog_bootstrap_profile_options(
    *,
    host: str | None = None,
    transport: str | None = None,
    scope_auth: bool = False,
    discovery_mode: str | None = None,
    port: int | None = None,
) -> str:
    """校验 catalog bootstrap 的启动边界并返回规范化监听地址。

    输入是 profile 启动参数；动作拒绝非 HTTP、scope-auth、full discovery 和旧端口配置，
    只允许 loopback 主机。返回规范化主机地址；任何不匹配都抛出 ``ValueError``，调用方
    不得继续构造部分开放的服务。直接构造可省略端口，实际 CLI 绑定时必须使用 9011。
    """

    normalized_host = normalize_loopback_host(
        "127.0.0.1" if host is None else host
    )
    if transport is not None and transport != "streamable_http":
        raise ValueError(
            "catalog_bootstrap profile requires streamable_http transport"
        )
    if scope_auth:
        raise ValueError("catalog_bootstrap profile does not support scope authentication")
    if discovery_mode is not None:
        normalized_mode = str(discovery_mode).strip().lower().replace("-", "_")
        if normalized_mode != "progressive":
            raise ValueError(
                "catalog_bootstrap profile requires progressive discovery mode"
            )
    if port is not None and port != 9011:
        raise ValueError("catalog_bootstrap profile requires port 9011")
    return normalized_host


def validate_investigation_profile_options(
    *,
    host: str | None = None,
    transport: str | None = None,
    scope_auth: bool = False,
    discovery_mode: str | None = None,
    port: int | None = None,
) -> str:
    """校验 Investigation 服务的 loopback、HTTP 和 scope 边界。

    输入是 profile 构造或 CLI 传入的监听选项；动作拒绝远程监听、stdio、未启用
    capability scope 和 full 目录。CLI 传入 ``port`` 时还必须使用固定的 9012；
    直接构造服务可省略端口，因为它尚未绑定网络。返回规范化 loopback 地址，失败
    直接抛出 ``ValueError``，调用方不能继续创建服务。
    """

    normalized_host = normalize_loopback_host(
        "127.0.0.1" if host is None else host
    )
    if transport is not None and transport != "streamable_http":
        raise ValueError(
            "investigation profile requires streamable_http transport"
        )
    if not scope_auth:
        raise ValueError("investigation profile requires scope authentication")
    if discovery_mode is not None:
        normalized_mode = str(discovery_mode).strip().lower().replace("-", "_")
        if normalized_mode != "progressive":
            raise ValueError(
                "investigation profile requires progressive discovery mode"
            )
    if port is not None and port != 9012:
        raise ValueError("investigation profile requires port 9012")
    return normalized_host


def _calculate_catalog_payload_hash(
    payload: dict[str, Any],
    failure_message: str,
) -> str:
    """按目录协议计算 JSON 摘要；不可规范编码时抛出启动错误。"""

    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CatalogFreezeError(failure_message) from exc
    return hashlib.sha256(encoded).hexdigest()


def _is_object_collection(value: object) -> bool:
    """判断目录集合是否为 JSON object 列表。"""

    return isinstance(value, list) and all(isinstance(item, dict) for item in value)


def _validate_catalog_identity_hashes(
    snapshot: dict[str, Any],
    public: dict[str, Any],
) -> tuple[str, str]:
    """核对 canonical 快照 hash 与公开投影 hash，返回两个稳定摘要。"""

    canonical_hash = snapshot.get("catalog_hash")
    projection_hash = public.get("catalog_projection_hash")
    if (
        not isinstance(canonical_hash, str)
        or _SHA256.fullmatch(canonical_hash) is None
        or public.get("catalog_hash") != canonical_hash
        or not isinstance(projection_hash, str)
        or _SHA256.fullmatch(projection_hash) is None
    ):
        raise CatalogFreezeError("MiniChem 目录 canonical/public catalog_hash 不一致")
    canonical_payload = deepcopy(snapshot)
    canonical_payload.pop("catalog_hash", None)
    if canonical_hash != _calculate_catalog_payload_hash(
        canonical_payload,
        "MiniChem 目录 canonical 快照不是可规范编码的 JSON",
    ):
        raise CatalogFreezeError("MiniChem 目录 canonical catalog_hash 不一致")
    projection_payload = deepcopy(public)
    projection_payload.pop("catalog_projection_hash", None)
    if projection_hash != _calculate_catalog_payload_hash(
        projection_payload,
        "MiniChem 目录公开投影不是可规范编码的 JSON",
    ):
        raise CatalogFreezeError("MiniChem 目录公开 projection hash 不一致")
    return canonical_hash, projection_hash


def _validate_catalog_identity_shape(
    snapshot: dict[str, Any],
    public: dict[str, Any],
) -> tuple[int, str]:
    """核对目录版本、模式和公开集合的结构，拒绝部分快照。"""

    schema_version = snapshot.get("schema_version")
    discovery_mode = public.get("discovery_mode")
    raw_collection_fields = ("actions", "backends", "resources")
    public_collection_fields = (
        "execution_layers",
        "actions",
        "backends",
        "resources",
    )
    if (
        schema_version != 7
        or public.get("schema_version") != schema_version
        or discovery_mode not in {"progressive", "full"}
        or public.get("catalog_projection_schema_version") != 1
        or public.get("catalog_visibility_policy") != "public_contract_projection"
        or any(
            not _is_object_collection(snapshot.get(field))
            for field in raw_collection_fields
        )
        or any(
            not isinstance(public.get(field), list)
            for field in public_collection_fields
        )
    ):
        raise CatalogFreezeError("MiniChem 目录公开身份字段非法")
    return int(schema_version), str(discovery_mode)


def build_public_catalog_identity(snapshot: dict[str, Any]) -> dict[str, Any]:
    """从冻结目录生成只含公开摘要的服务身份。

    输入是已经由 catalog 冻结流程返回的 canonical snapshot；动作重新构造公开投影，
    核对其 ``catalog_hash`` 仍与 canonical 身份一致，并只返回摘要、版本和 discovery
    mode。任何类型、摘要或公开投影不一致都抛出 ``CatalogFreezeError``，调用方不能
    用部分目录继续构造 profile。返回值不含 workspace、runtime 路径、健康探针细节或
    capability/token 字段，供服务实例 readiness 和进程内测试读取。
    """

    if not isinstance(snapshot, dict):
        raise CatalogFreezeError("MiniChem 目录快照必须是 JSON object")
    try:
        public = public_catalog_snapshot(snapshot)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise CatalogFreezeError("MiniChem 目录公开投影无法构造") from exc
    canonical_hash, projection_hash = _validate_catalog_identity_hashes(snapshot, public)
    schema_version, discovery_mode = _validate_catalog_identity_shape(snapshot, public)
    identity = {
        "catalog_hash": canonical_hash,
        "catalog_projection_hash": projection_hash,
        "catalog_version": str(schema_version),
        "discovery_mode": discovery_mode,
    }
    return deepcopy(identity)


INVESTIGATION_DISCOVERY_TOOL_NAMES = (
    "list_action_domains",
    "browse_action_category",
    "search_actions",
    "inspect_action",
    "inspect_backend",
    "search_resources",
    "inspect_resource",
)
INVESTIGATION_ACTION_SUBMISSION_TOOL_NAMES = (
    "submit_action_job",
)
INVESTIGATION_SOFTWARE_TOOL_NAMES = (
    "list_software",
    "inspect_software",
    "read_software_documentation",
    "search_software_documentation",
)
INVESTIGATION_WORKSPACE_TOOL_NAMES = (
    "write_workspace_text",
    "read_workspace_text",
)
INVESTIGATION_NATIVE_TOOL_NAMES = (
    "validate_native_job",
    "submit_native_job",
    "get_execution_job",
    "collect_execution_job",
    "cancel_execution_job",
)
INVESTIGATION_ANALYSIS_TOOL_NAMES = (
    "list_analysis_runtimes",
    "inspect_analysis_inputs",
    "validate_analysis_program",
    "submit_analysis_program",
)
INVESTIGATION_ARTIFACT_TOOL_NAMES = (
    "declare_scientific_artifact",
    "read_scientific_artifact_text",
)
INVESTIGATION_READ_ONLY_TOOL_NAMES = (
    "get_execution_resources",
    "list_execution_jobs",
    "verify_execution_references",
)

# 该顺序同时是工具目录的稳定顺序；新增工具必须在这里明确分类后才能进入
# Investigation profile，不能依赖注册后删除或名称前缀推断权限。
INVESTIGATION_TOOL_NAMES = (
    *INVESTIGATION_DISCOVERY_TOOL_NAMES,
    *INVESTIGATION_ACTION_SUBMISSION_TOOL_NAMES,
    *INVESTIGATION_SOFTWARE_TOOL_NAMES,
    *INVESTIGATION_WORKSPACE_TOOL_NAMES,
    *INVESTIGATION_NATIVE_TOOL_NAMES,
    *INVESTIGATION_ANALYSIS_TOOL_NAMES,
    *INVESTIGATION_ARTIFACT_TOOL_NAMES,
    *INVESTIGATION_READ_ONLY_TOOL_NAMES,
)


def register_investigation_tools(mcp: Any) -> list[str]:
    """在一个 FastMCP 实例中只注册 Investigation 的显式 allowlist。

    输入是尚未注册业务工具的 FastMCP 实例；动作按 discovery、统一 Action 提交、
    软件文档、工作区 staging、native/analysis 提交和只读核验逐项绑定。返回值是
    实际注册顺序；若清单与绑定实现不一致立即失败，不会返回部分开放的服务。
    """

    from . import (
        action_jobs,
        discovery_tools,
        investigation_submission_gate,
        open_tools,
        transition_state_jobs,
    )

    functions: dict[str, Callable[..., Any]] = {
        "list_action_domains": discovery_tools.list_action_domains,
        "browse_action_category": discovery_tools.browse_action_category,
        "search_actions": discovery_tools.search_actions,
        "inspect_action": action_jobs.investigation_inspect_action,
        "inspect_backend": discovery_tools.inspect_backend,
        "search_resources": discovery_tools.search_resources,
        "inspect_resource": discovery_tools.inspect_resource,
        "submit_action_job": investigation_submission_gate.submit_investigation_action_job,
        "list_software": open_tools.list_software,
        "inspect_software": open_tools.inspect_software,
        "read_software_documentation": open_tools.read_software_documentation,
        "search_software_documentation": open_tools.search_software_documentation,
        "write_workspace_text": open_tools.write_workspace_text,
        "read_workspace_text": open_tools.read_workspace_text,
        "validate_native_job": open_tools.validate_native_job,
        "submit_native_job": investigation_submission_gate.submit_investigation_native_job,
        "get_execution_job": open_tools.get_execution_job,
        "collect_execution_job": open_tools.collect_execution_job,
        "cancel_execution_job": open_tools.cancel_execution_job,
        "list_analysis_runtimes": open_tools.list_analysis_runtimes,
        "inspect_analysis_inputs": open_tools.inspect_analysis_inputs,
        "validate_analysis_program": open_tools.validate_analysis_program,
        "submit_analysis_program": (
            investigation_submission_gate.submit_investigation_analysis_program
        ),
        "declare_scientific_artifact": open_tools.declare_scientific_artifact,
        "read_scientific_artifact_text": open_tools.read_scientific_artifact_text,
        "get_execution_resources": open_tools.get_execution_resources,
        "list_execution_jobs": open_tools.list_execution_jobs,
        "verify_execution_references": open_tools.verify_execution_references,
    }
    if tuple(functions) != INVESTIGATION_TOOL_NAMES:
        raise RuntimeError("Investigation tool binding does not match its explicit allowlist")
    descriptions = {
        **discovery_tools.TOOL_DESCRIPTIONS,
        **open_tools.TOOL_DESCRIPTIONS,
        **action_jobs.TOOL_DESCRIPTIONS,
        **transition_state_jobs.TOOL_DESCRIPTIONS,
    }
    if any(name not in descriptions for name in functions):
        raise RuntimeError("Investigation tool descriptions do not cover its allowlist")
    for name, function in functions.items():
        mcp.tool(name=name, description=descriptions[name])(function)
    actual = tuple(getattr(mcp._tool_manager, "_tools", {}).keys())
    if actual != INVESTIGATION_TOOL_NAMES:
        raise RuntimeError("FastMCP registered an unexpected Investigation tool set")
    return list(actual)


def register_catalog_bootstrap_tools(mcp: Any) -> list[str]:
    """在 FastMCP 实例上注册初始目录冻结所需的四个只读工具。

    输入是尚未注册业务工具的 FastMCP 实例；动作只绑定精确 Action 合同、软件清单、
    软件合同和分析运行时清单，不包含任何执行、提交、工作区或 Artifact 写入能力。
    返回实际注册顺序；清单或描述不一致时抛出异常，不返回半开放服务。
    """

    from . import discovery_tools, open_tools

    functions: dict[str, Callable[..., Any]] = {
        "inspect_action": discovery_tools.inspect_action,
        "list_software": open_tools.list_software,
        "inspect_software": open_tools.inspect_software,
        "list_analysis_runtimes": open_tools.list_analysis_runtimes,
    }
    if tuple(functions) != CATALOG_BOOTSTRAP_TOOL_NAMES:
        raise RuntimeError(
            "catalog_bootstrap tool binding does not match its explicit allowlist"
        )
    descriptions = {
        **discovery_tools.TOOL_DESCRIPTIONS,
        **open_tools.TOOL_DESCRIPTIONS,
    }
    if any(name not in descriptions for name in functions):
        raise RuntimeError(
            "catalog_bootstrap tool descriptions do not cover its allowlist"
        )
    for name, function in functions.items():
        mcp.tool(name=name, description=descriptions[name])(function)
    actual = tuple(getattr(mcp._tool_manager, "_tools", {}).keys())
    if actual != CATALOG_BOOTSTRAP_TOOL_NAMES:
        raise RuntimeError(
            "FastMCP registered an unexpected catalog_bootstrap tool set"
        )
    return list(actual)


def register_catalog_bootstrap_resources(
    mcp: Any,
    snapshot: dict[str, Any] | None = None,
) -> list[str]:
    """只注册脱敏 ``minichem://catalog`` 资源并绑定启动时冻结快照。

    输入是 FastMCP 实例和可选的已冻结目录；动作不创建工作区文件、不暴露 token 或
    scope 认证资料，只返回目录公开投影。返回资源 URI 顺序；资源表出现额外项目时
    抛出异常，避免把完整服务资源误带入 bootstrap endpoint。
    """

    frozen_snapshot = deepcopy(snapshot) if snapshot is not None else None
    if frozen_snapshot is not None:
        identity = build_public_catalog_identity(frozen_snapshot)
        if identity["discovery_mode"] != "progressive":
            raise CatalogFreezeError(
                "catalog_bootstrap 只接受 progressive 目录快照"
            )

    @mcp.resource(
        "minichem://catalog",
        description="读取启动时冻结的公开 Action、Backend 和资源元数据。",
    )
    def catalog_bootstrap_resource() -> str:
        return json.dumps(
            public_catalog_snapshot(frozen_snapshot),
            ensure_ascii=False,
            indent=2,
        )

    actual = tuple(
        str(resource.uri)
        for resource in getattr(mcp._resource_manager, "_resources", {}).values()
    )
    if actual != CATALOG_BOOTSTRAP_RESOURCE_URIS:
        raise RuntimeError(
            "FastMCP registered an unexpected catalog_bootstrap resource set"
        )
    return list(actual)


def catalog_bootstrap_profile_instructions() -> str:
    """返回 bootstrap endpoint 的最小公开说明，不携带宿主或认证资料。"""

    return (
        "Catalog bootstrap profile. Loopback-only, read-only catalog inspection "
        "for initial catalog freezing."
    )


def investigation_profile_instructions() -> str:
    """返回不包含完整目录、宿主路径或未批准工具名的 profile 说明。"""

    return (
        "Investigation execution profile. Capability authentication is required for every "
        "Streamable HTTP message. Use only the explicitly registered discovery, Action "
        "submission, software documentation, workspace staging, native/analysis execution, "
        "resource, and execution-reference verification tools."
    )


__all__ = [
    "CATALOG_BOOTSTRAP_RESOURCE_URIS",
    "CATALOG_BOOTSTRAP_TOOL_NAMES",
    "INVESTIGATION_ACTION_SUBMISSION_TOOL_NAMES",
    "INVESTIGATION_ANALYSIS_TOOL_NAMES",
    "INVESTIGATION_ARTIFACT_TOOL_NAMES",
    "INVESTIGATION_DISCOVERY_TOOL_NAMES",
    "INVESTIGATION_NATIVE_TOOL_NAMES",
    "INVESTIGATION_READ_ONLY_TOOL_NAMES",
    "INVESTIGATION_SOFTWARE_TOOL_NAMES",
    "INVESTIGATION_TOOL_NAMES",
    "INVESTIGATION_WORKSPACE_TOOL_NAMES",
    "catalog_bootstrap_profile_instructions",
    "build_public_catalog_identity",
    "investigation_profile_instructions",
    "normalize_loopback_host",
    "register_catalog_bootstrap_resources",
    "register_catalog_bootstrap_tools",
    "register_investigation_tools",
    "resolve_server_profile",
    "validate_catalog_bootstrap_profile_options",
    "validate_investigation_profile_options",
]
