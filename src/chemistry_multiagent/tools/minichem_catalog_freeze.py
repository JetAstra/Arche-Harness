"""MiniChem Action Catalog 的任务级冻结与复验。

目录事实来自已打开的 MiniChem MCP adapter。此模块只把服务返回的快照和服务给出
的 catalog hash 放进任务目录，并在恢复时核对路径、哈希和严格 JSON；不会复制 Action
定义到 ARCHE 的 YAML，也不会从旧 Tool Catalog 生成兼容快照。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from copy import deepcopy
from dataclasses import dataclass
from collections.abc import Sequence
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Mapping

from chemistry_multiagent.tools.minichem_adapter import (
    _load_configured_evaluation_resource_budget,
    validate_public_catalog_projection,
    MiniChemCatalogSnapshot,
    MiniChemMcpAdapter,
    MiniChemMcpError,
    MiniChemMcpProtocolError,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CATALOG_FIELDS = {
    "schema_version",
    "execution_layers",
    "discovery_mode",
    "exposure_policy",
    "catalog_visibility_policy",
    "task_specific_tool_filtering",
    "backend_selection_policy",
    "provider_selection_policies",
    "scientific_resource_selection_policy",
    "automatic_fallback",
    "evaluation_resource_budget",
    "actions",
    "backends",
    "resources",
    "catalog_hash",
}
_LAYER_ROLES = {
    "predefined_actions": "validated common scientific operations",
    "native_software": "Agent-authored native inputs and exact allowlisted commands",
    "programmable_analysis": "Agent-authored programs in explicitly selected chemistry runtimes",
}
_PROVIDER_POLICIES = {
    "agent_backend_required",
    "agent_components_required",
    "agent_source_required",
    "fixed_source",
    "internal_deterministic",
}
_OPEN_EXECUTION_FIELDS = {
    "schema_version",
    "native_software",
    "analysis_runtimes",
    "authoring_contract",
}
_OPEN_EXECUTION_OPTIONAL_FIELDS = {
    "managed_program_contract",
    "submit_analysis_program_request_template",
    "execution_timeout_policy",
    "selection_note",
    "execution_note",
    "evaluation_resource_budget",
}
_OPEN_SOFTWARE_FIELDS = {"software_id", "display_name", "commands"}
_OPEN_COMMAND_FIELDS = {
    "executable",
    "synopsis",
    "input_mode",
    "required_files",
    "example_request",
}
_OPEN_COMMAND_OPTIONAL_FIELDS = {"output_behavior"}
_OPEN_RUNTIME_FIELDS = {"runtime", "description", "modules"}
_OPEN_AUTHORING_FIELDS = {"path_prefix", "max_files", "max_chars_per_file"}
_OPEN_EXAMPLE_FIELDS = {
    "software_id",
    "executable",
    "arguments",
    "staged_inputs",
    "stdin_target",
    "resource_limits",
    "resource_policy",
    "calculation_intent",
    "label",
    "parent_job_id",
}
_OPEN_GAUSSIAN_RESOURCE_POLICY = {
    "mode": "service_managed_full_capacity",
    "concurrency": "serial",
    "agent_controllable": False,
}
_OPEN_RESOURCE_FIELDS = {"cpu_cores", "memory_mb", "gpu_count"}
_OPEN_EVALUATION_BUDGET_FIELDS = {
    "source",
    "agent_controllable",
    "scope",
}
_OPEN_LEGACY_EVALUATION_BUDGET_FIELDS = _OPEN_EVALUATION_BUDGET_FIELDS | {
    "cpu_cores",
    "memory_mb",
    "gpu_count",
}
_OPEN_TIMEOUT_FIELDS = {
    "execution_class",
    "source",
    "agent_controllable",
}
_OPEN_LEGACY_TIMEOUT_FIELDS = _OPEN_TIMEOUT_FIELDS | {"timeout_seconds"}
_CALCULATION_INTENTS = {
    "single_point",
    "geometry_optimization",
    "frequency",
    "optimization_frequency",
    "transition_state",
    "ionic_relaxation",
    "reaction_path",
    "conformer_search",
    "protonation",
    "deprotonation",
    "tautomerization",
    "projection",
    "other",
}
_JOB_ID = re.compile(r"^job_[0-9a-f]{32}$")
_MODULE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")
_OPEN_EXECUTABLE = re.compile(r"^[A-Za-z0-9_.+-]+$")
_NON_EMPTY_TEXT = re.compile(r"[^\x00\r\n]+")
_CONTRACT_ID = re.compile(r"^[a-z][a-z0-9_]*$")
# discovery 返回的 Analysis 模板并不是一次真实提交；它用这个标记提醒
# Planner 在提交时必须选择具体 runtime。真正提交的 runtime 仍只能是安全标识符。
_ANALYSIS_RUNTIME_PLACEHOLDER = "<exact runtime id>"
_MAX_MANAGED_PROGRAM_TEMPLATE_CHARS = 100_000

_ACTION_CONTRACT_SNAPSHOT_FIELDS = {
    "schema_version",
    "catalog_hash",
    "contracts",
}
_ACTION_CONTRACT_ENTRY_FIELDS = {
    "action_id",
    "backend_id",
    "request_contract",
    "output_contract",
}
_CONTRACT_SECTIONS = ("inputs", "method_spec", "action_settings", "resource_limits")
_ACTION_REQUEST_CONTRACT_FIELDS = {
    "input_description",
    "required_contract",
    "optional_field_names",
    "execute_action_request_template",
    "enumerated_options",
    "conditional_requirements",
    "output_contract",
    "template_kind",
    "important_optional_inputs",
    "component_request_contracts",
    "usage_notes",
    "retry_policy",
    "sections",
    "request_fields",
    "output_fields",
    "description",
    "selection_notes",
    "evaluation_resource_budget",
    "resource_policy",
}
_CONTRACTS_FILE = "action_request_contracts.json"
_CONTRACTS_MANIFEST_FILE = "action_request_contracts_manifest.json"


class MiniChemCatalogFreezeError(ValueError):
    """表示 MiniChem 目录无法冻结或恢复。"""


class _DetachedSnapshot(Mapping[str, Any]):
    """保存目录副本，并为每次读取返回新的、可修改的副本。

    冻结目录会被控制器写入 checkpoint，也会被执行层读取。如果直接把内部
    ``dict`` 暴露出去，调用方修改任意嵌套列表或字典后，后续读取就可能看到
    与目录哈希不一致的内容。这里保留 Mapping 的读取接口，但每次读取都做
    深复制；调用方即使修改返回值，也不会改变 FrozenMiniChemCatalog 的内容。
    """

    __slots__ = ("_value",)

    def __init__(self, value: Mapping[str, Any]) -> None:
        self._value = deepcopy(dict(value))

    def __getitem__(self, key: str) -> Any:
        return deepcopy(self._value[key])

    def __iter__(self):
        return iter(self._value)

    def __len__(self) -> int:
        return len(self._value)


@dataclass(frozen=True)
class FrozenMiniChemCatalog:
    """保存任务目录、服务哈希和冻结快照的只读句柄。"""

    catalog_hash: str
    snapshot: Mapping[str, Any]
    package_root: Path

    def __post_init__(self) -> None:
        """复制快照并隔离嵌套可变值，避免调用方改写冻结目录。"""

        if not isinstance(self.snapshot, Mapping):
            raise TypeError("MiniChem 冻结快照必须是 object")
        object.__setattr__(self, "snapshot", _DetachedSnapshot(self.snapshot))


def _canonical_bytes(value: Any) -> bytes:
    """把目录快照编码成无歧义 JSON 字节。"""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise MiniChemCatalogFreezeError("MiniChem catalog 不是严格 JSON") from exc


def _collect_action_request_contracts(
    adapter: MiniChemMcpAdapter, snapshot: Mapping[str, Any]
) -> dict[str, Any]:
    contracts: dict[str, dict[str, Any]] = {}
    for action in snapshot.get("actions", []):
        if not isinstance(action, Mapping):
            raise MiniChemCatalogFreezeError("MiniChem Action 记录非法")
        action_id = action.get("id")
        backend_ids = action.get("backend_ids")
        if not isinstance(action_id, str) or not isinstance(backend_ids, list):
            raise MiniChemCatalogFreezeError("MiniChem Action 身份或 Backend 列表非法")
        selected: dict[str, Any] = {}
        for backend_id in backend_ids:
            if not isinstance(backend_id, str):
                raise MiniChemCatalogFreezeError("MiniChem backend_id 非法")
            try:
                inspected = adapter.inspect_action(action_id, backend_id)
            except (MiniChemMcpError, MiniChemMcpProtocolError) as exc:
                raise MiniChemCatalogFreezeError(
                    f"冻结 Action 请求合同失败：{action_id}/{backend_id}"
                ) from exc
            contract = inspected.get("selected_request_contract")
            if not isinstance(contract, Mapping):
                raise MiniChemCatalogFreezeError(
                    f"inspect_action 缺少请求合同：{action_id}/{backend_id}"
                )
            selected[backend_id] = deepcopy(dict(contract))
        contracts[action_id] = selected
    software_listing = adapter.list_software(native_only=True, available_only=True)
    native_software: list[dict[str, Any]] = []
    for item in software_listing["software"]:
        if not isinstance(item, Mapping) or not isinstance(item.get("software_id"), str):
            raise MiniChemCatalogFreezeError("MiniChem native software 清单非法")
        inspected = adapter.inspect_software(item["software_id"])
        guides = inspected.get("native_invocation_guides")
        if not isinstance(guides, list):
            raise MiniChemCatalogFreezeError(
                f"inspect_software 缺少调用合同：{item['software_id']}"
            )
        compact_guides = []
        for guide in guides:
            if not isinstance(guide, Mapping):
                raise MiniChemCatalogFreezeError("MiniChem native command guide 非法")
            compact_guides.append(
                {
                    key: deepcopy(guide[key])
                    for key in (
                        "executable",
                        "enabled",
                        "available",
                        "synopsis",
                        "input_mode",
                        "required_files",
                        "output_behavior",
                        "example_arguments",
                        "notes",
                        "native_job_request_template",
                    )
                    if key in guide
                }
            )
        native_software.append(
            {
                "software_id": item["software_id"],
                "display_name": item.get("display_name", item["software_id"]),
                "available": item.get("available") is True,
                "purpose": inspected.get("purpose"),
                "native_invocation_guides": compact_guides,
            }
        )
    runtime_listing = adapter.list_analysis_runtimes()
    runtimes = runtime_listing.get("runtimes")
    if not isinstance(runtimes, list):
        raise MiniChemCatalogFreezeError("MiniChem analysis runtime 清单非法")
    result = {
        "schema_version": 2,
        "catalog_hash": snapshot["catalog_hash"],
        "contracts": contracts,
        "native_software": native_software,
        "analysis_runtimes": deepcopy(runtimes),
        "managed_program_contract": deepcopy(
            runtime_listing.get("managed_program_contract")
        ),
    }
    if "execution_timeout_policy" in runtime_listing:
        result["execution_timeout_policy"] = deepcopy(
            runtime_listing["execution_timeout_policy"]
        )
    return result


def _enrich_snapshot_with_contracts(
    snapshot: Mapping[str, Any], bundle: Mapping[str, Any]
) -> dict[str, Any]:
    if bundle.get("catalog_hash") != snapshot.get("catalog_hash"):
        raise MiniChemCatalogFreezeError("Action 请求合同与 catalog_hash 不一致")
    contracts = bundle.get("contracts")
    if not isinstance(contracts, Mapping):
        raise MiniChemCatalogFreezeError("Action 请求合同包缺少 contracts")
    enriched = deepcopy(dict(snapshot))
    for action in enriched["actions"]:
        action_id = action["id"]
        action_contracts = contracts.get(action_id)
        if not isinstance(action_contracts, Mapping) or set(action_contracts) != set(
            action["backend_ids"]
        ):
            raise MiniChemCatalogFreezeError(f"Action 请求合同覆盖不完整：{action_id}")
        action["request_contracts"] = deepcopy(dict(action_contracts))
    if bundle.get("schema_version") == 2:
        for field in ("native_software", "analysis_runtimes"):
            if not isinstance(bundle.get(field), list):
                raise MiniChemCatalogFreezeError(f"执行层合同缺少 {field}")
            enriched[field] = deepcopy(bundle[field])
        enriched["managed_program_contract"] = deepcopy(
            bundle.get("managed_program_contract")
        )
    return enriched


def _write_contract_bundle(package_root: Path, bundle: Mapping[str, Any]) -> None:
    payload = _canonical_bytes(bundle)
    manifest = {
        "schema_version": 1,
        "catalog_hash": bundle["catalog_hash"],
        "contracts_relative_path": _CONTRACTS_FILE,
        "contracts_sha256": hashlib.sha256(payload).hexdigest(),
        "contracts_size": len(payload),
    }
    _atomic_write(package_root / _CONTRACTS_FILE, payload)
    _atomic_write(package_root / _CONTRACTS_MANIFEST_FILE, _canonical_bytes(manifest))


def _load_contract_bundle(
    package_root: Path, catalog_hash: str
) -> dict[str, Any] | None:
    path = package_root / _CONTRACTS_FILE
    manifest_path = package_root / _CONTRACTS_MANIFEST_FILE
    if not path.exists() and not manifest_path.exists():
        return None
    if (
        path.is_symlink()
        or manifest_path.is_symlink()
        or not path.is_file()
        or not manifest_path.is_file()
    ):
        raise MiniChemCatalogFreezeError("Action 请求合同冻结文件不完整")
    try:
        payload = path.read_bytes()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        bundle = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MiniChemCatalogFreezeError("Action 请求合同冻结文件不可读") from exc
    expected_manifest = {
        "schema_version": 1,
        "catalog_hash": catalog_hash,
        "contracts_relative_path": _CONTRACTS_FILE,
        "contracts_sha256": hashlib.sha256(payload).hexdigest(),
        "contracts_size": len(payload),
    }
    if manifest != expected_manifest or _canonical_bytes(bundle) != payload:
        raise MiniChemCatalogFreezeError("Action 请求合同 manifest 或内容校验失败")
    return bundle


def _safe_hash(value: Any, label: str = "catalog_hash") -> str:
    """验证目录身份只包含小写 SHA-256。"""

    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise MiniChemCatalogFreezeError(f"{label} 不是 64 位小写 SHA-256")
    return value


def _validate_catalog_v7_snapshot(snapshot: Mapping[str, Any], expected_hash: str) -> dict[str, Any]:
    """完整核对冻结快照的 v7 字段、固定策略、引用关系和 canonical hash。"""

    if not isinstance(snapshot, Mapping):
        raise MiniChemCatalogFreezeError("MiniChem catalog 快照必须是 object")
    value = deepcopy(dict(snapshot))
    if value.get("catalog_visibility_policy") == "public_contract_projection":
        try:
            projected = validate_public_catalog_projection(value)
        except MiniChemMcpProtocolError as exc:
            raise MiniChemCatalogFreezeError(
                "MiniChem public catalog projection 校验失败"
            ) from exc
        if projected.get("catalog_hash") != expected_hash:
            raise MiniChemCatalogFreezeError(
                "MiniChem public catalog projection canonical hash 与冻结目标不一致"
            )
        return projected
    missing = _CATALOG_FIELDS - set(value)
    unknown = set(value) - _CATALOG_FIELDS
    if missing or unknown:
        raise MiniChemCatalogFreezeError(
            f"MiniChem catalog v7 顶层字段不匹配：missing={sorted(missing)} unknown={sorted(unknown)}"
        )
    if value.get("schema_version") != 7:
        raise MiniChemCatalogFreezeError("MiniChem catalog schema_version 必须为 7")
    if value.get("catalog_hash") != expected_hash:
        raise MiniChemCatalogFreezeError("MiniChem catalog_hash 与冻结目标不一致")
    mode = value.get("discovery_mode")
    if mode not in {"progressive", "full"}:
        raise MiniChemCatalogFreezeError("MiniChem catalog.discovery_mode 非法")
    if value.get("exposure_policy") != {
        "progressive": "progressive_discovery",
        "full": "atomic_all",
    }[mode]:
        raise MiniChemCatalogFreezeError("MiniChem catalog.exposure_policy 与 mode 不一致")
    if value.get("catalog_visibility_policy") != "focused_task_independent":
        raise MiniChemCatalogFreezeError("MiniChem catalog.catalog_visibility_policy 非法")
    if value.get("task_specific_tool_filtering") is not False:
        raise MiniChemCatalogFreezeError("MiniChem catalog.task_specific_tool_filtering 必须为 false")
    if value.get("backend_selection_policy") != "per_action_explicit":
        raise MiniChemCatalogFreezeError("MiniChem catalog.backend_selection_policy 非法")
    if value.get("scientific_resource_selection_policy") != "agent_explicit_no_default":
        raise MiniChemCatalogFreezeError(
            "MiniChem catalog.scientific_resource_selection_policy 非法"
        )
    if value.get("automatic_fallback") is not False:
        raise MiniChemCatalogFreezeError("MiniChem catalog.automatic_fallback 必须为 false")
    policies = value.get("provider_selection_policies")
    if (
        not isinstance(policies, list)
        or any(not isinstance(item, str) or item not in _PROVIDER_POLICIES for item in policies)
        or policies != sorted(set(policies))
    ):
        raise MiniChemCatalogFreezeError("MiniChem provider_selection_policies 不符合 v7 policy")
    budget = value.get("evaluation_resource_budget")
    if not isinstance(budget, Mapping) or set(budget) != {
        "cpu_cores",
        "memory_mb",
        "gpu_count",
        "source",
        "agent_controllable",
        "scope",
    }:
        raise MiniChemCatalogFreezeError("MiniChem evaluation_resource_budget 不符合 v7 schema")
    if (
        type(budget.get("cpu_cores")) is not int
        or budget["cpu_cores"] < 1
        or type(budget.get("memory_mb")) is not int
        or budget["memory_mb"] < 128
        or type(budget.get("gpu_count")) is not int
        or budget["gpu_count"] < 0
        or budget.get("source") != "evaluation_policy"
        or budget.get("agent_controllable") is not False
        or budget.get("scope") != "per_task"
    ):
        raise MiniChemCatalogFreezeError("MiniChem evaluation_resource_budget 不符合 v7 policy")
    try:
        configured_budget = _load_configured_evaluation_resource_budget()
    except MiniChemMcpError as exc:
        raise MiniChemCatalogFreezeError("MiniChem ARCHE 资源政策不可读") from exc
    if dict(budget) != configured_budget:
        raise MiniChemCatalogFreezeError(
            "MiniChem evaluation_resource_budget 与 ARCHE 资源政策不一致："
            f"expected={configured_budget!r} actual={dict(budget)!r}"
        )
    layers = value.get("execution_layers")
    if not isinstance(layers, list) or len(layers) != len(_LAYER_ROLES):
        raise MiniChemCatalogFreezeError("MiniChem execution_layers 不完整")
    for field in ("actions", "backends", "resources"):
        if not isinstance(value.get(field), list):
            raise MiniChemCatalogFreezeError(f"MiniChem {field} 必须是数组")
    layer_ids: set[str] = set()
    for item in layers:
        if not isinstance(item, Mapping) or set(item) != {"id", "role", "mandatory"}:
            raise MiniChemCatalogFreezeError("MiniChem execution_layers 项字段不符合 v7 schema")
        layer_id = item.get("id")
        if layer_id not in _LAYER_ROLES or layer_id in layer_ids:
            raise MiniChemCatalogFreezeError("MiniChem execution_layers id 非法或重复")
        if item.get("role") != _LAYER_ROLES[layer_id] or item.get("mandatory") is not False:
            raise MiniChemCatalogFreezeError("MiniChem execution_layers role/mandatory 非法")
        layer_ids.add(layer_id)
    action_ids: set[str] = set()
    for item in value["actions"]:
        if not isinstance(item, Mapping):
            raise MiniChemCatalogFreezeError("MiniChem actions 项必须是 object")
        legacy_fields = {"action_id", "provider_ids", "backend_id"} & set(item)
        if legacy_fields:
            raise MiniChemCatalogFreezeError(
                "MiniChem action 使用了旧字段，v7 必须使用 id/backend_ids："
                f"{sorted(legacy_fields)}"
            )
        action_id = item.get("id")
        if not isinstance(action_id, str) or not re.fullmatch(r"^[a-z][a-z0-9_]*$", action_id):
            raise MiniChemCatalogFreezeError("MiniChem action id 非法")
        if action_id in action_ids:
            raise MiniChemCatalogFreezeError(f"MiniChem action id 重复：{action_id}")
        action_ids.add(action_id)
    backend_ids: set[str] = set()
    for item in value["backends"]:
        if not isinstance(item, Mapping):
            raise MiniChemCatalogFreezeError("MiniChem backends 项必须是 object")
        legacy_fields = {"action_id", "provider_ids", "backend_id"} & set(item)
        if legacy_fields:
            raise MiniChemCatalogFreezeError(
                "MiniChem backend 使用了旧字段，v7 必须使用 id："
                f"{sorted(legacy_fields)}"
            )
        backend_id = item.get("id")
        if not isinstance(backend_id, str) or not re.fullmatch(r"^[a-z][a-z0-9_]*$", backend_id):
            raise MiniChemCatalogFreezeError("MiniChem backend id 非法")
        if backend_id in backend_ids:
            raise MiniChemCatalogFreezeError(f"MiniChem backend id 重复：{backend_id}")
        backend_ids.add(backend_id)
    for item in value["actions"]:
        backend_choices = item.get("backend_ids")
        if not isinstance(backend_choices, list) or any(
            not isinstance(backend_id, str)
            or re.fullmatch(r"^[a-z][a-z0-9_]*$", backend_id) is None
            for backend_id in backend_choices
        ):
            raise MiniChemCatalogFreezeError("MiniChem action backend_ids 必须是 lower_snake_case 数组")
        for backend_id in backend_choices:
            if backend_id not in backend_ids:
                raise MiniChemCatalogFreezeError(
                    f"MiniChem action 引用了未声明 backend：{backend_id}"
                )
    if not all(
        isinstance(item, Mapping)
        and isinstance(item.get("id"), str)
        and re.fullmatch(r"^[a-z][a-z0-9_]*$", item["id"]) is not None
        for item in value["resources"]
    ):
        raise MiniChemCatalogFreezeError("MiniChem resources 项必须包含 id")
    payload = dict(value)
    payload.pop("catalog_hash", None)
    calculated = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    if calculated != expected_hash:
        raise MiniChemCatalogFreezeError(
            f"MiniChem catalog canonical hash 不一致：expected={calculated} actual={expected_hash}"
        )
    return value


def _validate_open_text(value: Any, label: str) -> str:
    """校验 Open Execution 中用于说明能力的非空文本。"""

    if not isinstance(value, str) or not value.strip() or _NON_EMPTY_TEXT.fullmatch(value) is None:
        raise MiniChemCatalogFreezeError(f"MiniChem Open Execution {label} 必须是单行非空文本")
    return value


def _validate_managed_program_template(value: Any, label: str) -> str:
    """校验可放入 JSON 快照的多行托管程序模板。

    MiniChem 返回的最小 Python 模板本身是给 Planner 参考的代码片段，真实值可能
    含有多行换行符；其他说明字段仍使用 ``_validate_open_text`` 的单行规则。这里
    只允许有限长度、非空且能以严格 JSON 字符串编码并往返解析的内容，避免把坏的
    Unicode、NUL 或异常大的模板写入任务冻结目录。
    """

    if not isinstance(value, str) or not value.strip():
        raise MiniChemCatalogFreezeError(
            f"MiniChem Open Execution {label} 必须是非空多行文本"
        )
    if len(value) > _MAX_MANAGED_PROGRAM_TEMPLATE_CHARS:
        raise MiniChemCatalogFreezeError(
            f"MiniChem Open Execution {label} 超过 {_MAX_MANAGED_PROGRAM_TEMPLATE_CHARS} 字符上限"
        )
    if "\x00" in value:
        raise MiniChemCatalogFreezeError(
            f"MiniChem Open Execution {label} 不得包含 NUL 字符"
        )
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        if json.loads(encoded) != value:
            raise ValueError("JSON 字符串往返结果不一致")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise MiniChemCatalogFreezeError(
            f"MiniChem Open Execution {label} 必须是严格 JSON 文本"
        ) from exc
    return value


def _validate_open_id(value: Any, label: str) -> str:
    """校验 Open Execution 的软件、命令和运行时标识。"""

    if not isinstance(value, str) or re.fullmatch(r"^[a-z][a-z0-9_]*$", value) is None:
        raise MiniChemCatalogFreezeError(
            f"MiniChem Open Execution {label} 必须使用 lower_snake_case"
        )
    return value


def _validate_analysis_template_runtime(value: Any, label: str) -> str:
    """校验 Analysis discovery 模板中的 runtime 占位符或具体运行时标识。

    ``list_analysis_runtimes`` 返回的是给 Planner 看的请求模板，其中
    ``<exact runtime id>`` 明确表示“提交时再选择运行时”，不是可直接提交给
    MiniChem 的 runtime。除这个协议规定的占位符外，快照只接受 lower_snake_case
    标识，避免把任意说明文本误当成可执行运行时。
    """

    if value == _ANALYSIS_RUNTIME_PLACEHOLDER:
        return value
    return _validate_open_id(value, label)


def _validate_relative_open_path(value: Any, label: str) -> str:
    """校验快照中的路径只能引用任务工作区内的相对文件。"""

    if not isinstance(value, str) or not value or "\x00" in value:
        raise MiniChemCatalogFreezeError(f"MiniChem Open Execution {label} 路径非法")
    if "\\" in value or re.match(r"^[A-Za-z]:[/\\]", value):
        raise MiniChemCatalogFreezeError(f"MiniChem Open Execution {label} 路径非法")
    normalized = value
    path = PurePosixPath(normalized)
    if (
        path.is_absolute()
        or normalized.startswith("~")
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise MiniChemCatalogFreezeError(
            f"MiniChem Open Execution {label} 不得使用绝对路径或路径回退"
        )
    return normalized


def _validate_open_command_argument(value: Any, label: str) -> str:
    """拒绝命令模板或参数中可能绕过工作区边界的路径。"""

    if not isinstance(value, str) or "\x00" in value:
        raise MiniChemCatalogFreezeError(f"MiniChem Open Execution {label} 参数非法")
    for token in re.split(r"\s+", value.strip()):
        candidate = token.split("=", 1)[1] if "=" in token else token
        normalized = candidate.replace("\\", "/").strip()
        path = PurePosixPath(normalized)
        if (
            path.is_absolute()
            or normalized.startswith("~")
            or re.match(r"^[A-Za-z]:/", normalized)
            or ".." in path.parts
        ):
            raise MiniChemCatalogFreezeError(
                f"MiniChem Open Execution {label} 不得包含绝对路径或路径回退"
            )
    return value


def _validate_open_resource_limits(value: Any, label: str) -> dict[str, Any]:
    """校验示例请求中的资源限制只包含三个可声明的整数。"""

    if not isinstance(value, Mapping) or not set(value) <= _OPEN_RESOURCE_FIELDS:
        raise MiniChemCatalogFreezeError(
            f"MiniChem Open Execution {label} resource_limits 字段非法"
        )
    limits = {
        "cpu_cores": (1, "正整数"),
        "memory_mb": (128, "不小于 128 的整数"),
        "gpu_count": (0, "非负整数"),
    }
    for field, (minimum, description) in limits.items():
        if field not in value:
            continue
        if type(value[field]) is not int or value[field] < minimum:
            raise MiniChemCatalogFreezeError(
                f"MiniChem Open Execution {label}.{field} 必须是{description}"
            )
    return deepcopy(dict(value))


def _validate_open_evaluation_budget(value: Any, label: str) -> dict[str, Any]:
    """校验服务声明的任务级资源政策记录。"""

    if not isinstance(value, Mapping) or set(value) not in {
        frozenset(_OPEN_EVALUATION_BUDGET_FIELDS),
        frozenset(_OPEN_LEGACY_EVALUATION_BUDGET_FIELDS),
    }:
        raise MiniChemCatalogFreezeError(f"{label} 字段不完整")
    fields = frozenset(value)
    if fields == _OPEN_LEGACY_EVALUATION_BUDGET_FIELDS:
        if (
            type(value["cpu_cores"]) is not int
            or value["cpu_cores"] < 1
            or type(value["memory_mb"]) is not int
            or value["memory_mb"] < 128
            or type(value["gpu_count"]) is not int
            or value["gpu_count"] < 0
            or value["source"] != "evaluation_policy"
        ):
            raise MiniChemCatalogFreezeError(f"{label} 不符合任务资源政策")
    elif value["source"] != "service_policy":
        raise MiniChemCatalogFreezeError(f"{label} 不符合任务资源政策")
    if value["agent_controllable"] is not False or value["scope"] != "per_task":
        raise MiniChemCatalogFreezeError(f"{label} 不符合任务资源政策")
    return deepcopy(dict(value))


def _validate_open_timeout_policy(value: Any, label: str) -> dict[str, Any]:
    """校验服务声明的执行超时政策。"""

    if not isinstance(value, Mapping) or set(value) not in {
        frozenset(_OPEN_TIMEOUT_FIELDS),
        frozenset(_OPEN_LEGACY_TIMEOUT_FIELDS),
    }:
        raise MiniChemCatalogFreezeError(f"{label} 字段不完整")
    if (
        value["execution_class"] not in {"compute", "fast"}
        or value["agent_controllable"] is not False
    ):
        raise MiniChemCatalogFreezeError(f"{label} 不符合执行超时政策")
    if frozenset(value) == frozenset(_OPEN_LEGACY_TIMEOUT_FIELDS):
        if (
            type(value["timeout_seconds"]) is not int
            or value["timeout_seconds"] < 1
            or value["source"] != "evaluation_policy"
        ):
            raise MiniChemCatalogFreezeError(f"{label} 不符合执行超时政策")
    elif value["source"] != "service_policy":
        raise MiniChemCatalogFreezeError(f"{label} 不符合执行超时政策")
    return deepcopy(dict(value))


def _validate_open_output_path(value: Any, label: str) -> str:
    """校验开放执行声明的输出只能写到任务公开目录。"""

    normalized = _validate_relative_open_path(value, label)
    if not normalized.startswith(("outputs/", "report/", "logs/")):
        raise MiniChemCatalogFreezeError(
            f"{label} 必须位于 outputs/、report/ 或 logs/ 目录"
        )
    return normalized


def _validate_open_example_request(
    value: Any,
    *,
    software_id: str,
    executable: str,
) -> dict[str, Any]:
    """校验 native 命令的示例请求，不允许借示例保存主机路径或未知字段。"""

    if not isinstance(value, Mapping):
        raise MiniChemCatalogFreezeError("MiniChem Open Execution example_request 必须是 object")
    unknown = set(value) - _OPEN_EXAMPLE_FIELDS
    if unknown:
        raise MiniChemCatalogFreezeError(
            f"MiniChem Open Execution example_request 含未知字段：{sorted(unknown)}"
        )
    result = deepcopy(dict(value))
    if "software_id" in result and result["software_id"] != software_id:
        raise MiniChemCatalogFreezeError(
            "MiniChem Open Execution example_request.software_id 与软件不一致"
        )
    if "executable" in result and result["executable"] != executable:
        raise MiniChemCatalogFreezeError(
            "MiniChem Open Execution example_request.executable 与命令不一致"
        )
    for field in ("software_id",):
        if field in result:
            _validate_open_id(result[field], f"example_request.{field}")
    if "executable" in result and (
        not isinstance(result["executable"], str)
        or _OPEN_EXECUTABLE.fullmatch(result["executable"]) is None
    ):
        raise MiniChemCatalogFreezeError(
            "MiniChem Open Execution example_request.executable 必须是命令名"
        )
    arguments = result.get("arguments", [])
    if not isinstance(arguments, list):
        raise MiniChemCatalogFreezeError("MiniChem Open Execution example_request.arguments 必须是数组")
    for index, argument in enumerate(arguments):
        _validate_open_command_argument(argument, f"example_request.arguments[{index}]")
    staged = result.get("staged_inputs", [])
    if not isinstance(staged, list):
        raise MiniChemCatalogFreezeError(
            "MiniChem Open Execution example_request.staged_inputs 必须是数组"
        )
    targets: set[str] = set()
    for index, item in enumerate(staged):
        if not isinstance(item, Mapping) or set(item) != {"source_path", "target_path"}:
            raise MiniChemCatalogFreezeError(
                "MiniChem Open Execution staged_inputs 项字段不符合 NativeJobRequest"
            )
        source = _validate_relative_open_path(item["source_path"], f"staged_inputs[{index}].source_path")
        target = _validate_relative_open_path(item["target_path"], f"staged_inputs[{index}].target_path")
        if target in targets:
            raise MiniChemCatalogFreezeError("MiniChem Open Execution staged_inputs target_path 不能重复")
        targets.add(target)
        item["source_path"] = source
        item["target_path"] = target
    if result.get("stdin_target") is not None:
        stdin_target = _validate_relative_open_path(
            result["stdin_target"], "example_request.stdin_target"
        )
        if stdin_target not in targets:
            raise MiniChemCatalogFreezeError(
                "MiniChem Open Execution stdin_target 必须引用 staged_inputs"
            )
        result["stdin_target"] = stdin_target
    if "resource_limits" in result:
        result["resource_limits"] = _validate_open_resource_limits(
            result["resource_limits"], "example_request"
        )
    if "resource_policy" in result:
        if software_id != "gaussian" or result["resource_policy"] != (
            _OPEN_GAUSSIAN_RESOURCE_POLICY
        ):
            raise MiniChemCatalogFreezeError(
                "MiniChem Open Execution example_request.resource_policy 非法"
            )
        if "resource_limits" in result:
            raise MiniChemCatalogFreezeError(
                "MiniChem Open Execution Gaussian 示例不得同时公开资源数值和资源政策"
            )
        result["resource_policy"] = deepcopy(_OPEN_GAUSSIAN_RESOURCE_POLICY)
    intent = result.get("calculation_intent")
    if intent is not None and intent not in _CALCULATION_INTENTS:
        raise MiniChemCatalogFreezeError(
            "MiniChem Open Execution example_request.calculation_intent 非法"
        )
    if result.get("label") is not None:
        _validate_open_text(result["label"], "example_request.label")
    parent_job_id = result.get("parent_job_id")
    if parent_job_id is not None and (
        not isinstance(parent_job_id, str) or _JOB_ID.fullmatch(parent_job_id) is None
    ):
        raise MiniChemCatalogFreezeError(
            "MiniChem Open Execution example_request.parent_job_id 非法"
        )
    return result


def _validate_open_execution_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """完整校验独立 Open Execution schema v1，返回可安全放入任务状态的副本。"""

    if not isinstance(snapshot, Mapping):
        raise MiniChemCatalogFreezeError("MiniChem Open Execution 快照必须是 object")
    value = deepcopy(dict(snapshot))
    missing = _OPEN_EXECUTION_FIELDS - set(value)
    unknown = set(value) - (_OPEN_EXECUTION_FIELDS | _OPEN_EXECUTION_OPTIONAL_FIELDS)
    if missing or unknown:
        raise MiniChemCatalogFreezeError(
            "MiniChem Open Execution 顶层字段不匹配："
            f"missing={sorted(missing)} unknown={sorted(unknown)}"
        )
    if value["schema_version"] != 1:
        raise MiniChemCatalogFreezeError("MiniChem Open Execution schema_version 必须为 1")
    if "managed_program_contract" in value:
        value["managed_program_contract"] = _validate_managed_program_contract(
            value["managed_program_contract"]
        )
    if "submit_analysis_program_request_template" in value:
        value["submit_analysis_program_request_template"] = _validate_open_analysis_template(
            value["submit_analysis_program_request_template"]
        )
    if "execution_timeout_policy" in value:
        value["execution_timeout_policy"] = _validate_open_timeout_policy(
            value["execution_timeout_policy"],
            "open_execution.execution_timeout_policy",
        )
    for note_field in ("selection_note", "execution_note"):
        if note_field in value:
            value[note_field] = _validate_open_text(value[note_field], note_field)
    if "evaluation_resource_budget" in value:
        value["evaluation_resource_budget"] = _validate_open_evaluation_budget(
            value["evaluation_resource_budget"], "open_execution.evaluation_resource_budget"
        )
    software_items = value["native_software"]
    if not isinstance(software_items, list):
        raise MiniChemCatalogFreezeError("MiniChem Open Execution native_software 必须是数组")
    software_ids: set[str] = set()
    for index, item in enumerate(software_items):
        if not isinstance(item, Mapping) or set(item) != _OPEN_SOFTWARE_FIELDS:
            raise MiniChemCatalogFreezeError(
                f"MiniChem Open Execution native_software[{index}] 字段不完整"
            )
        software_id = _validate_open_id(item["software_id"], f"native_software[{index}].software_id")
        if software_id in software_ids:
            raise MiniChemCatalogFreezeError("MiniChem Open Execution software_id 不能重复")
        software_ids.add(software_id)
        _validate_open_text(item["display_name"], f"native_software[{index}].display_name")
        commands = item["commands"]
        if not isinstance(commands, list):
            raise MiniChemCatalogFreezeError(
                f"MiniChem Open Execution native_software[{index}].commands 必须是数组"
            )
        command_ids: set[str] = set()
        for command_index, command in enumerate(commands):
            if not isinstance(command, Mapping) or not (
                set(command) <= _OPEN_COMMAND_FIELDS | _OPEN_COMMAND_OPTIONAL_FIELDS
                and _OPEN_COMMAND_FIELDS <= set(command)
            ):
                raise MiniChemCatalogFreezeError(
                    f"MiniChem Open Execution command[{index}:{command_index}] 字段不完整"
                )
            executable = command["executable"]
            if not isinstance(executable, str) or _OPEN_EXECUTABLE.fullmatch(executable) is None:
                raise MiniChemCatalogFreezeError(
                    f"MiniChem Open Execution command[{index}:{command_index}].executable 必须是命令名"
                )
            if executable in command_ids:
                raise MiniChemCatalogFreezeError("MiniChem Open Execution executable 不能重复")
            command_ids.add(executable)
            _validate_open_text(command["synopsis"], f"command[{index}:{command_index}].synopsis")
            _validate_open_command_argument(
                command["synopsis"], f"command[{index}:{command_index}].synopsis"
            )
            _validate_open_text(command["input_mode"], f"command[{index}:{command_index}].input_mode")
            required_files = command["required_files"]
            if not isinstance(required_files, list):
                raise MiniChemCatalogFreezeError("MiniChem Open Execution required_files 必须是数组")
            seen_files: set[str] = set()
            for file_index, path in enumerate(required_files):
                normalized = _validate_relative_open_path(
                    path, f"command[{index}:{command_index}].required_files[{file_index}]"
                )
                if normalized in seen_files:
                    raise MiniChemCatalogFreezeError("MiniChem Open Execution required_files 不能重复")
                seen_files.add(normalized)
                required_files[file_index] = normalized
            command["example_request"] = _validate_open_example_request(
                command["example_request"], software_id=software_id, executable=executable
            )
            if "output_behavior" in command:
                _validate_open_text(
                    command["output_behavior"],
                    f"command[{index}:{command_index}].output_behavior",
                )
    runtimes = value["analysis_runtimes"]
    if not isinstance(runtimes, list):
        raise MiniChemCatalogFreezeError("MiniChem Open Execution analysis_runtimes 必须是数组")
    runtime_ids: set[str] = set()
    for index, item in enumerate(runtimes):
        if not isinstance(item, Mapping) or set(item) != _OPEN_RUNTIME_FIELDS:
            raise MiniChemCatalogFreezeError(
                f"MiniChem Open Execution analysis_runtimes[{index}] 字段不完整"
            )
        runtime = _validate_open_id(item["runtime"], f"analysis_runtimes[{index}].runtime")
        if runtime in runtime_ids:
            raise MiniChemCatalogFreezeError("MiniChem Open Execution runtime 不能重复")
        runtime_ids.add(runtime)
        _validate_open_text(item["description"], f"analysis_runtimes[{index}].description")
        modules = item["modules"]
        if not isinstance(modules, list):
            raise MiniChemCatalogFreezeError("MiniChem Open Execution modules 必须是数组")
        seen_modules: set[str] = set()
        for module_index, module in enumerate(modules):
            if not isinstance(module, str) or _MODULE_NAME.fullmatch(module) is None:
                raise MiniChemCatalogFreezeError("MiniChem Open Execution module 名称非法")
            if module in seen_modules:
                raise MiniChemCatalogFreezeError("MiniChem Open Execution modules 不能重复")
            seen_modules.add(module)
    authoring = value["authoring_contract"]
    if not isinstance(authoring, Mapping) or set(authoring) != _OPEN_AUTHORING_FIELDS:
        raise MiniChemCatalogFreezeError("MiniChem Open Execution authoring_contract 字段不完整")
    if (
        authoring["path_prefix"] != "code/"
        or type(authoring["max_files"]) is not int
        or authoring["max_files"] != 4
        or type(authoring["max_chars_per_file"]) is not int
        or authoring["max_chars_per_file"] != 100000
    ):
        raise MiniChemCatalogFreezeError("MiniChem Open Execution authoring_contract 不符合固定合同")
    try:
        _canonical_bytes(value)
    except MiniChemCatalogFreezeError:
        raise
    return value


def _empty_open_execution_snapshot() -> dict[str, Any]:
    """为 schema1 历史任务提供不声明任何软件能力的安全合同。"""

    return {
        "schema_version": 1,
        "native_software": [],
        "analysis_runtimes": [],
        "authoring_contract": {
            "path_prefix": "code/",
            "max_files": 4,
            "max_chars_per_file": 100000,
        },
    }


def _validate_managed_program_contract(value: Any) -> dict[str, Any]:
    """校验分析程序运行时给 Planner 的固定编程边界。"""

    fields = {
        "job_context_required_for_compliance",
        "execution_note",
        "minimal_python_template",
        "helper_name_rule",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise MiniChemCatalogFreezeError(
            "MiniChem Open Execution managed_program_contract 字段不完整"
        )
    if type(value["job_context_required_for_compliance"]) is not bool:
        raise MiniChemCatalogFreezeError(
            "MiniChem managed_program_contract.job_context_required_for_compliance 必须是 bool"
        )
    result = deepcopy(dict(value))
    _validate_open_text(
        result["execution_note"], "managed_program_contract.execution_note"
    )
    _validate_managed_program_template(
        result["minimal_python_template"],
        "managed_program_contract.minimal_python_template",
    )
    _validate_open_text(
        result["helper_name_rule"], "managed_program_contract.helper_name_rule"
    )
    return result


def _validate_open_analysis_template(value: Any) -> dict[str, Any]:
    """校验 submit_analysis_program 的最小请求模板和路径边界。"""

    required_fields = {
        "runtime",
        "script_path",
        "script_target",
        "arguments",
        "staged_inputs",
        "inputs",
        "outputs",
        "label",
    }
    if not isinstance(value, Mapping) or not required_fields <= set(value):
        raise MiniChemCatalogFreezeError(
            "MiniChem Open Execution submit_analysis_program_request_template 字段不完整"
        )
    allowed_fields = required_fields | {
        "required_modules",
        "required_module_versions",
        "required_symbols",
        # 只用于恢复旧冻结包；新公开模板不再携带资源默认值。
        "resource_limits",
    }
    unknown = set(value) - allowed_fields
    if unknown:
        raise MiniChemCatalogFreezeError(
            "MiniChem Open Execution submit_analysis_program_request_template 含未知字段："
            f"{sorted(unknown)}"
        )
    result = deepcopy(dict(value))
    result["runtime"] = _validate_analysis_template_runtime(
        result["runtime"],
        "submit_analysis_program_request_template.runtime",
    )
    for field in ("script_path", "script_target"):
        result[field] = _validate_relative_open_path(
            result[field],
            f"submit_analysis_program_request_template.{field}",
        )
    if not isinstance(result["arguments"], list) or any(
        not isinstance(item, str)
        for item in result["arguments"]
    ):
        raise MiniChemCatalogFreezeError(
            "MiniChem Open Execution analysis template.arguments 必须是字符串数组"
        )
    staged = result["staged_inputs"]
    if not isinstance(staged, list):
        raise MiniChemCatalogFreezeError(
            "MiniChem Open Execution analysis template.staged_inputs 必须是数组"
        )
    for index, item in enumerate(staged):
        if not isinstance(item, Mapping) or set(item) != {"source_path", "target_path"}:
            raise MiniChemCatalogFreezeError(
                "MiniChem Open Execution analysis template.staged_inputs 项字段不完整"
            )
        item["source_path"] = _validate_relative_open_path(
            item["source_path"],
            f"analysis template.staged_inputs[{index}].source_path",
        )
        item["target_path"] = _validate_relative_open_path(
            item["target_path"],
            f"analysis template.staged_inputs[{index}].target_path",
        )
    if not isinstance(result["inputs"], list) or not isinstance(result["outputs"], list):
        raise MiniChemCatalogFreezeError(
            "MiniChem Open Execution analysis template.inputs/outputs 必须是数组"
        )
    for index, item in enumerate(result["inputs"]):
        if not isinstance(item, Mapping) or not {"name", "source_path", "target_path"} <= set(item):
            raise MiniChemCatalogFreezeError(
                "MiniChem Open Execution analysis template.inputs 项字段不完整"
            )
        for field in ("name", "source_path"):
            _validate_open_text(item[field], f"analysis template.inputs[{index}].{field}")
        if item.get("target_path") is not None:
            item["target_path"] = _validate_relative_open_path(
                item["target_path"],
                f"analysis template.inputs[{index}].target_path",
            )
    for index, item in enumerate(result["outputs"]):
        if not isinstance(item, Mapping) or not {"name", "path", "semantic_type", "media_type", "required"} <= set(item):
            raise MiniChemCatalogFreezeError(
                "MiniChem Open Execution analysis template.outputs 项字段不完整"
            )
        for field in ("name", "semantic_type", "media_type"):
            _validate_open_text(item[field], f"analysis template.outputs[{index}].{field}")
        item["path"] = _validate_open_output_path(
            item["path"],
            f"analysis template.outputs[{index}].path",
        )
        if type(item["required"]) is not bool:
            raise MiniChemCatalogFreezeError(
                f"analysis template.outputs[{index}].required 必须是 bool"
            )
    if "resource_limits" in result:
        result["resource_limits"] = _validate_open_resource_limits(
            result["resource_limits"],
            "submit_analysis_program_request_template.resource_limits",
        )
    if result["label"] is not None:
        _validate_open_text(result["label"], "analysis template.label")
    return result


def _normalize_required_section(value: Any, label: str) -> dict[str, Any]:
    """把 inspect_action 的 section 合同收敛为 required/optional/allowed 三项。"""

    if isinstance(value, list):
        value = {"required": value}
    if not isinstance(value, Mapping):
        raise MiniChemCatalogFreezeError(f"{label} 必须是 object 或字段数组")
    allowed = {"required", "optional", "allowed", "properties", "description"}
    unknown = set(value) - allowed
    if unknown:
        raise MiniChemCatalogFreezeError(f"{label} 含未知字段：{sorted(unknown)}")
    normalized: dict[str, Any] = {}
    for field in ("required", "optional", "allowed"):
        raw = value.get(field, [])
        if not isinstance(raw, list) or any(
            not isinstance(item, (str, Mapping)) for item in raw
        ):
            raise MiniChemCatalogFreezeError(f"{label}.{field} 必须是字符串或 object 数组")
        names: list[Any] = []
        for item in raw:
            name = item.get("name") if isinstance(item, Mapping) else item
            if not isinstance(name, str) or not name.strip() or name in names:
                raise MiniChemCatalogFreezeError(f"{label}.{field} 含重复或非法字段")
            names.append(name)
        normalized[field] = names
    overlap = set(normalized["required"]) & set(normalized["optional"])
    if overlap:
        raise MiniChemCatalogFreezeError(f"{label} required/optional 重叠：{sorted(overlap)}")
    if "properties" in value:
        if not isinstance(value["properties"], Mapping):
            raise MiniChemCatalogFreezeError(f"{label}.properties 必须是 object")
        normalized["properties"] = deepcopy(dict(value["properties"]))
    if "description" in value:
        normalized["description"] = _validate_open_text(
            value["description"], f"{label}.description"
        )
    if not normalized["allowed"]:
        normalized["allowed"] = list(
            dict.fromkeys(normalized["required"] + normalized["optional"])
        )
    return normalized


def _normalize_action_request_contract(value: Any, label: str) -> dict[str, Any]:
    """校验并标准化单个 Action/Backend 的精确请求合同。"""

    if not isinstance(value, Mapping):
        raise MiniChemCatalogFreezeError(f"{label} 必须是 object")
    source = deepcopy(dict(value))
    unknown = set(source) - _ACTION_REQUEST_CONTRACT_FIELDS
    if unknown:
        raise MiniChemCatalogFreezeError(
            f"{label} 含未知字段：{sorted(unknown)}"
        )
    sections = source.get("sections")
    required_contract = source.get("required_contract")
    if sections is None and isinstance(required_contract, Mapping):
        sections = deepcopy(dict(required_contract))
    if sections is None:
        sections = {}
        aliases = {
            "required_inputs": "inputs",
            "required_method_fields": "method_spec",
            "required_action_settings": "action_settings",
            "required_resource_fields": "resource_limits",
        }
        for source_key, section in aliases.items():
            if source_key in source:
                sections[section] = {"required": source[source_key]}
        for section in _CONTRACT_SECTIONS:
            if section in source and section not in sections:
                sections[section] = source[section]
    if not isinstance(sections, Mapping):
        raise MiniChemCatalogFreezeError(f"{label}.sections 必须是 object")
    normalized_sections: dict[str, Any] = {}
    for section in _CONTRACT_SECTIONS:
        if section in sections:
            normalized_sections[section] = _normalize_required_section(
                sections[section], f"{label}.sections.{section}"
            )
        else:
            normalized_sections[section] = {
                "required": [],
                "optional": [],
                "allowed": [],
            }
    result: dict[str, Any] = {"sections": normalized_sections}
    if isinstance(required_contract, Mapping):
        result["required_contract"] = _assert_contract_json_copy(
            required_contract,
            f"{label}.required_contract",
        )
    if "optional_field_names" in source:
        optional_names = source["optional_field_names"]
        if isinstance(optional_names, Mapping):
            normalized_optional: dict[str, list[str]] = {}
            for section, fields in optional_names.items():
                if section not in _CONTRACT_SECTIONS or not isinstance(fields, list):
                    raise MiniChemCatalogFreezeError(
                        f"{label}.optional_field_names 不是合法 section map"
                    )
                checked = []
                for field_name in fields:
                    if not isinstance(field_name, str) or not field_name.strip():
                        raise MiniChemCatalogFreezeError(
                            f"{label}.optional_field_names.{section} 含非法字段"
                        )
                    checked.append(field_name)
                    if field_name not in normalized_sections[section]["optional"]:
                        normalized_sections[section]["optional"].append(field_name)
                        normalized_sections[section]["allowed"].append(field_name)
                normalized_optional[section] = list(dict.fromkeys(checked))
            result["optional_field_names"] = normalized_optional
        elif isinstance(optional_names, list) and all(
            isinstance(item, str) and item.strip() for item in optional_names
        ):
            result["optional_field_names"] = list(dict.fromkeys(optional_names))
            for field_name in optional_names:
                if "." in field_name:
                    prefix, name = field_name.split(".", 1)
                    if prefix in normalized_sections and name not in normalized_sections[prefix]["optional"]:
                        normalized_sections[prefix]["optional"].append(name)
                        normalized_sections[prefix]["allowed"].append(name)
                elif field_name not in normalized_sections["inputs"]["optional"]:
                    normalized_sections["inputs"]["optional"].append(field_name)
                    normalized_sections["inputs"]["allowed"].append(field_name)
        else:
            raise MiniChemCatalogFreezeError(
                f"{label}.optional_field_names 必须是 section map 或字符串数组"
            )
    for field in (
        "execute_action_request_template",
        "conditional_requirements",
        "output_contract",
    ):
        if field in source:
            result[field] = _assert_contract_json_copy(source[field], f"{label}.{field}")
    if "evaluation_resource_budget" in source:
        result["evaluation_resource_budget"] = _validate_open_evaluation_budget(
            source["evaluation_resource_budget"],
            f"{label}.evaluation_resource_budget",
        )
    if "resource_policy" in source:
        if source["resource_policy"] != _OPEN_GAUSSIAN_RESOURCE_POLICY:
            raise MiniChemCatalogFreezeError(
                f"{label}.resource_policy 不符合服务管理策略"
            )
        result["resource_policy"] = deepcopy(_OPEN_GAUSSIAN_RESOURCE_POLICY)
    if "evaluation_resource_budget" in source and "resource_policy" in source:
        raise MiniChemCatalogFreezeError(
            f"{label} 不得同时声明通用资源政策和 Gaussian 资源政策"
        )
    if "enumerated_options" in source:
        options = source["enumerated_options"]
        if not isinstance(options, list) or any(
            not isinstance(item, Mapping)
            or set(item) - {"path", "required", "allowed_values"}
            or not isinstance(item.get("path"), str)
            or not isinstance(item.get("required"), bool)
            or not isinstance(item.get("allowed_values"), list)
            for item in options
        ):
            raise MiniChemCatalogFreezeError(
                f"{label}.enumerated_options 必须是 path/required/allowed_values 数组"
            )
        result["enumerated_options"] = _assert_contract_json_copy(
            options, f"{label}.enumerated_options"
        )
    for field in (
        "template_kind",
        "important_optional_inputs",
        "component_request_contracts",
        "usage_notes",
        "retry_policy",
    ):
        if field in source:
            result[field] = _assert_contract_json_copy(source[field], f"{label}.{field}")
    for field in ("request_fields", "output_fields"):
        if field in source:
            if not isinstance(source[field], list) or any(
                not isinstance(item, str) or not item.strip() for item in source[field]
            ):
                raise MiniChemCatalogFreezeError(f"{label}.{field} 必须是字符串数组")
            result[field] = list(dict.fromkeys(source[field]))
    for field in ("input_description", "description", "selection_notes"):
        if field in source:
            result[field] = _validate_open_text(source[field], f"{label}.{field}")
    return result


def _assert_contract_json_copy(value: Any, label: str) -> Any:
    """复制 inspect_action 合同中的 JSON 值并拒绝主机对象。"""

    _assert_strict_json(value, label)
    return deepcopy(value)


def _validate_action_contract_snapshot(
    snapshot: Mapping[str, Any],
    expected_hash: str,
    catalog_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """校验与 catalog_hash 绑定的 pair-specific Action 合同快照。"""

    if not isinstance(snapshot, Mapping):
        raise MiniChemCatalogFreezeError("MiniChem Action 合同快照必须是 object")
    value = deepcopy(dict(snapshot))
    if set(value) != _ACTION_CONTRACT_SNAPSHOT_FIELDS:
        raise MiniChemCatalogFreezeError("MiniChem Action 合同快照顶层字段不完整")
    if value.get("schema_version") != 1 or value.get("catalog_hash") != expected_hash:
        raise MiniChemCatalogFreezeError("MiniChem Action 合同快照身份与 catalog_hash 不一致")
    records = minichem_ids = {
        str(item.get("id"))
        for item in catalog_snapshot.get("actions", [])
        if isinstance(item, Mapping)
    }
    pairs: set[tuple[str, str]] = set()
    contracts = value.get("contracts")
    if not isinstance(contracts, list):
        raise MiniChemCatalogFreezeError("MiniChem Action 合同 contracts 必须是数组")
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(contracts):
        label = f"contracts[{index}]"
        if not isinstance(item, Mapping) or set(item) != _ACTION_CONTRACT_ENTRY_FIELDS:
            raise MiniChemCatalogFreezeError(f"MiniChem Action 合同 {label} 字段不完整")
        action_id = item.get("action_id")
        backend_id = item.get("backend_id")
        if (
            not isinstance(action_id, str)
            or _CONTRACT_ID.fullmatch(action_id) is None
            or action_id not in minichem_ids
            or not isinstance(backend_id, str)
            or _CONTRACT_ID.fullmatch(backend_id) is None
        ):
            raise MiniChemCatalogFreezeError(f"MiniChem Action 合同 {label} 身份非法")
        action = next(
            candidate
            for candidate in catalog_snapshot["actions"]
            if isinstance(candidate, Mapping) and candidate.get("id") == action_id
        )
        if backend_id not in action.get("backend_ids", []):
            raise MiniChemCatalogFreezeError(
                f"MiniChem Action 合同 {label} 引用了不属于该 Action 的 backend"
            )
        pair = (action_id, backend_id)
        if pair in pairs:
            raise MiniChemCatalogFreezeError(f"MiniChem Action 合同重复 pair：{pair}")
        pairs.add(pair)
        request_contract = _normalize_action_request_contract(
            item["request_contract"], f"{label}.request_contract"
        )
        output_contract = item["output_contract"]
        if output_contract is None:
            output_contract = request_contract.get("output_contract")
        if output_contract is not None:
            if not isinstance(output_contract, Mapping):
                raise MiniChemCatalogFreezeError(f"{label}.output_contract 必须是 object 或 null")
            output_contract = deepcopy(dict(output_contract))
            _assert_strict_json(output_contract, f"{label}.output_contract")
        normalized.append(
            {
                "action_id": action_id,
                "backend_id": backend_id,
                "request_contract": request_contract,
                "output_contract": output_contract,
            }
        )
    expected_pairs = {
        (str(action.get("id")), str(backend_id))
        for action in catalog_snapshot.get("actions", [])
        if isinstance(action, Mapping)
        for backend_id in action.get("backend_ids", [])
    }
    if pairs != expected_pairs:
        missing = sorted(expected_pairs - pairs)
        extra = sorted(pairs - expected_pairs)
        raise MiniChemCatalogFreezeError(
            f"MiniChem Action 合同必须覆盖全部 Action/Backend pair：missing={missing} extra={extra}"
        )
    return {
        "schema_version": 1,
        "catalog_hash": expected_hash,
        "contracts": normalized,
    }


def _assert_strict_json(value: Any, label: str) -> None:
    """递归确认 companion 快照只含严格 JSON 值。"""

    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and (value != value or value in {float("inf"), float("-inf")}):
            raise MiniChemCatalogFreezeError(f"{label} 含非有限数字")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_strict_json(item, f"{label}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise MiniChemCatalogFreezeError(f"{label} 的键必须是字符串")
            _assert_strict_json(item, f"{label}.{key}")
        return
    raise MiniChemCatalogFreezeError(f"{label} 含不可序列化类型")


def _safe_task_root(value: str | os.PathLike[str]) -> Path:
    """解析任务根并拒绝符号链接任务目录。"""

    raw_root = Path(value)
    raw_absolute = raw_root if raw_root.is_absolute() else Path.cwd() / raw_root
    current = Path(raw_absolute.anchor)
    for component in raw_absolute.parts[1:]:
        current = current / component
        if current.is_symlink():
            raise MiniChemCatalogFreezeError("MiniChem task_root 路径不得经过符号链接")
    root = raw_root.resolve(strict=False)
    if root.exists() and not root.is_dir():
        raise MiniChemCatalogFreezeError("MiniChem task_root 必须是普通目录")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _atomic_write(path: Path, payload: bytes) -> None:
    """原子发布目录 JSON，避免恢复读到半份快照。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise MiniChemCatalogFreezeError("MiniChem 冻结文件不得是符号链接")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise MiniChemCatalogFreezeError("MiniChem 冻结临时文件已存在")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _discover_action_contract_snapshot(
    adapter: MiniChemMcpAdapter,
    catalog_snapshot: Mapping[str, Any],
    catalog_hash: str,
) -> dict[str, Any]:
    """通过 adapter 的精确 discovery 读取全部 Action/Backend 合同。

    真实适配器必须实现该 discovery；缺失任意 pair 或合同不完整都会让冻结失败，
    而不是把不确定的请求交给 Planner。历史 schema1/2 包只允许在恢复时读取，
    新冻结不再降级为旧 manifest。
    """

    discover = getattr(adapter, "discover_action_contracts", None)
    if not callable(discover):
        raise MiniChemCatalogFreezeError(
            "adapter 缺少 discover_action_contracts；新任务不能降级为 schema2"
        )
    try:
        discovered = discover()
    except (MiniChemMcpError, MiniChemMcpProtocolError, OSError, ValueError) as exc:
        raise MiniChemCatalogFreezeError("MiniChem Action 合同 discovery 失败") from exc
    if isinstance(discovered, Mapping) and "contracts" in discovered:
        candidate = deepcopy(dict(discovered))
    elif isinstance(discovered, list):
        candidate = {
            "schema_version": 1,
            "catalog_hash": catalog_hash,
            "contracts": deepcopy(discovered),
        }
    else:
        raise MiniChemCatalogFreezeError("MiniChem Action 合同 discovery 必须返回 object")
    candidate.setdefault("schema_version", 1)
    candidate.setdefault("catalog_hash", catalog_hash)
    try:
        return _validate_action_contract_snapshot(
            candidate,
            catalog_hash,
            catalog_snapshot,
        )
    except (MiniChemCatalogFreezeError, TypeError, KeyError) as exc:
        if isinstance(exc, MiniChemCatalogFreezeError):
            raise
        raise MiniChemCatalogFreezeError("MiniChem Action 合同快照结构非法") from exc


def freeze_minichem_catalog(
    *,
    adapter: MiniChemMcpAdapter,
    task_root: str | os.PathLike[str],
) -> FrozenMiniChemCatalog:
    """读取一次服务目录并写入任务级快照。"""

    if not isinstance(adapter, MiniChemMcpAdapter):
        raise TypeError("adapter 必须是 MiniChemMcpAdapter")
    root = _safe_task_root(task_root)
    try:
        catalog = adapter.discover_catalog()
    except (MiniChemMcpError, MiniChemMcpProtocolError, OSError, ValueError) as exc:
        raise MiniChemCatalogFreezeError("MiniChem catalog discovery 失败") from exc
    if not isinstance(catalog, MiniChemCatalogSnapshot):
        raise MiniChemCatalogFreezeError("adapter 没有返回 MiniChemCatalogSnapshot")
    catalog_hash = _safe_hash(catalog.catalog_hash)
    snapshot = _validate_catalog_v7_snapshot(catalog.snapshot, catalog_hash)
    if str(catalog.catalog_version) != str(snapshot["schema_version"]):
        raise MiniChemCatalogFreezeError("MiniChem catalog_version 与 schema_version 不一致")
    discover_open_execution_catalog = getattr(adapter, "discover_open_execution_catalog", None)
    if not callable(discover_open_execution_catalog):
        raise MiniChemCatalogFreezeError("adapter 缺少 discover_open_execution_catalog")
    try:
        discovered_open_execution = discover_open_execution_catalog()
    except (MiniChemMcpError, MiniChemMcpProtocolError, OSError, ValueError) as exc:
        raise MiniChemCatalogFreezeError("MiniChem Open Execution discovery 失败") from exc
    open_execution = _validate_open_execution_snapshot(discovered_open_execution)
    action_contracts = _discover_action_contract_snapshot(
        adapter,
        snapshot,
        catalog_hash,
    )
    payload = _canonical_bytes(snapshot)
    open_execution_payload = _canonical_bytes(open_execution)
    package_root = root / "internal" / "catalog" / catalog_hash
    current = root
    for component in package_root.relative_to(root).parts:
        current = current / component
        if current.is_symlink():
            raise MiniChemCatalogFreezeError("MiniChem 冻结包路径不得经过符号链接")
    snapshot_path = package_root / "catalog_snapshot.json"
    open_execution_path = package_root / "open_execution_snapshot.json"
    action_contract_path = package_root / "action_contract_snapshot.json"
    manifest_path = package_root / "catalog_manifest.json"
    manifest = {
        "schema_version": 3 if action_contracts is not None else 2,
        "catalog_hash": catalog_hash,
        "catalog_version": catalog.catalog_version,
        "snapshot_relative_path": "catalog_snapshot.json",
        "snapshot_sha256": hashlib.sha256(payload).hexdigest(),
        "snapshot_size": len(payload),
        "open_execution_relative_path": "open_execution_snapshot.json",
        "open_execution_sha256": hashlib.sha256(open_execution_payload).hexdigest(),
        "open_execution_size": len(open_execution_payload),
    }
    if action_contracts is not None:
        contract_payload = _canonical_bytes(action_contracts)
        manifest.update(
            {
                "action_contract_relative_path": "action_contract_snapshot.json",
                "action_contract_sha256": hashlib.sha256(contract_payload).hexdigest(),
                "action_contract_size": len(contract_payload),
            }
        )
        _atomic_write(action_contract_path, contract_payload)
    _atomic_write(snapshot_path, payload)
    _atomic_write(open_execution_path, open_execution_payload)
    _atomic_write(manifest_path, _canonical_bytes(manifest))
    memory_snapshot = deepcopy(dict(snapshot))
    memory_snapshot["open_execution"] = open_execution
    if action_contracts is not None:
        memory_snapshot["action_contracts"] = action_contracts
    return FrozenMiniChemCatalog(
        catalog_hash=catalog_hash,
        snapshot=memory_snapshot,
        package_root=package_root,
    )


def load_frozen_minichem_catalog(
    *,
    task_root: str | os.PathLike[str],
    catalog_hash: str,
    verify_hashes: Sequence[str] | None = None,
) -> FrozenMiniChemCatalog:
    """按任务保存的 hash 读取并复验 MiniChem 目录。

    默认对冻结包执行完整复验，保持历史恢复和迁移入口的严格语义。启动阶段若
    只需要确认关键身份，可显式传入 ``verify_hashes``；允许的关键项为
    ``snapshot``、``action_contract`` 和 ``open_execution``。未选中的 companion
    文件仍会解析并做 schema/canonical 校验，但不会重复计算其文件摘要。
    """

    if verify_hashes is None:
        selected_hashes = {"snapshot", "open_execution", "action_contract"}
    else:
        selected_hashes = set(verify_hashes)
        unknown_hashes = selected_hashes - {
            "snapshot",
            "open_execution",
            "action_contract",
        }
        if unknown_hashes:
            raise MiniChemCatalogFreezeError(
                "verify_hashes 含未注册项：" + ", ".join(sorted(unknown_hashes))
            )
        if "snapshot" not in selected_hashes:
            raise MiniChemCatalogFreezeError("启动复验必须包含 snapshot 关键摘要")

    root = _safe_task_root(task_root)
    expected_hash = _safe_hash(catalog_hash)
    package_root = root / "internal" / "catalog" / expected_hash
    snapshot_path = package_root / "catalog_snapshot.json"
    open_execution_path = package_root / "open_execution_snapshot.json"
    action_contract_path = package_root / "action_contract_snapshot.json"
    manifest_path = package_root / "catalog_manifest.json"
    current = root
    for component in package_root.relative_to(root).parts:
        current = current / component
        if current.is_symlink():
            raise MiniChemCatalogFreezeError("MiniChem 冻结包不得经过符号链接")
    if any(
        path.is_symlink()
        for path in (snapshot_path, open_execution_path, action_contract_path, manifest_path)
    ):
        raise MiniChemCatalogFreezeError("MiniChem 冻结包不得经过符号链接")
    try:
        manifest_payload = manifest_path.read_bytes()
        manifest = json.loads(manifest_payload)
        payload = snapshot_path.read_bytes()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MiniChemCatalogFreezeError("MiniChem 冻结包不可读") from exc
    if not isinstance(manifest, Mapping) or _canonical_bytes(manifest) != manifest_payload:
        raise MiniChemCatalogFreezeError("MiniChem 冻结包 manifest 不是 canonical JSON")
    if not isinstance(manifest, Mapping) or set(manifest) not in ({
        "schema_version",
        "catalog_hash",
        "catalog_version",
        "snapshot_relative_path",
        "snapshot_sha256",
        "snapshot_size",
    }, {
        "schema_version",
        "catalog_hash",
        "catalog_version",
        "snapshot_relative_path",
        "snapshot_sha256",
        "snapshot_size",
        "open_execution_relative_path",
        "open_execution_sha256",
        "open_execution_size",
    }, {
        "schema_version",
        "catalog_hash",
        "catalog_version",
        "snapshot_relative_path",
        "snapshot_sha256",
        "snapshot_size",
        "open_execution_relative_path",
        "open_execution_sha256",
        "open_execution_size",
        "action_contract_relative_path",
        "action_contract_sha256",
        "action_contract_size",
    }) or manifest.get("catalog_hash") != expected_hash:
        raise MiniChemCatalogFreezeError("MiniChem 冻结包 manifest hash 不一致")
    if manifest.get("snapshot_relative_path") != "catalog_snapshot.json":
        raise MiniChemCatalogFreezeError("MiniChem 冻结包 manifest 路径不一致")
    if "snapshot" in selected_hashes:
        if manifest.get("snapshot_sha256") != hashlib.sha256(payload).hexdigest():
            raise MiniChemCatalogFreezeError("MiniChem 冻结快照 sha256 不一致")
        if manifest.get("snapshot_size") != len(payload):
            raise MiniChemCatalogFreezeError("MiniChem 冻结快照大小不一致")
    try:
        snapshot = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MiniChemCatalogFreezeError("MiniChem 冻结快照不是 JSON") from exc
    if not isinstance(snapshot, Mapping):
        raise MiniChemCatalogFreezeError("MiniChem 冻结快照必须是 object")
    if _canonical_bytes(snapshot) != payload:
        raise MiniChemCatalogFreezeError("MiniChem 冻结快照不是 canonical JSON")
    snapshot = _validate_catalog_v7_snapshot(snapshot, expected_hash)
    if manifest.get("catalog_version") != str(snapshot["schema_version"]):
        raise MiniChemCatalogFreezeError("MiniChem 冻结包 manifest catalog_version 不一致")
    manifest_schema = manifest.get("schema_version")
    action_contracts = None
    if manifest_schema == 1:
        open_execution = _empty_open_execution_snapshot()
    elif manifest_schema in {2, 3}:
        if manifest.get("snapshot_relative_path") != "catalog_snapshot.json":
            raise MiniChemCatalogFreezeError("MiniChem 冻结包 manifest 路径不一致")
        if manifest.get("open_execution_relative_path") != "open_execution_snapshot.json":
            raise MiniChemCatalogFreezeError("MiniChem Open Execution manifest 路径不一致")
        try:
            open_payload = open_execution_path.read_bytes()
        except OSError as exc:
            raise MiniChemCatalogFreezeError("MiniChem Open Execution 冻结快照不可读") from exc
        if "open_execution" in selected_hashes:
            if manifest.get("open_execution_sha256") != hashlib.sha256(open_payload).hexdigest():
                raise MiniChemCatalogFreezeError("MiniChem Open Execution sha256 不一致")
            if manifest.get("open_execution_size") != len(open_payload):
                raise MiniChemCatalogFreezeError("MiniChem Open Execution 大小不一致")
        try:
            open_execution = json.loads(open_payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MiniChemCatalogFreezeError(
                "MiniChem Open Execution 冻结快照不是 JSON"
            ) from exc
        if not isinstance(open_execution, Mapping) or _canonical_bytes(open_execution) != open_payload:
            raise MiniChemCatalogFreezeError(
                "MiniChem Open Execution 冻结快照不是 canonical JSON"
            )
        open_execution = _validate_open_execution_snapshot(open_execution)
        action_contracts = None
        if manifest_schema == 3:
            if manifest.get("action_contract_relative_path") != "action_contract_snapshot.json":
                raise MiniChemCatalogFreezeError("MiniChem Action 合同 manifest 路径不一致")
            try:
                contract_payload = action_contract_path.read_bytes()
            except OSError as exc:
                raise MiniChemCatalogFreezeError(
                    "MiniChem Action 合同冻结快照不可读"
                ) from exc
            if "action_contract" in selected_hashes:
                if manifest.get("action_contract_sha256") != hashlib.sha256(contract_payload).hexdigest():
                    raise MiniChemCatalogFreezeError("MiniChem Action 合同 sha256 不一致")
                if manifest.get("action_contract_size") != len(contract_payload):
                    raise MiniChemCatalogFreezeError("MiniChem Action 合同大小不一致")
            try:
                action_contracts = json.loads(contract_payload)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise MiniChemCatalogFreezeError(
                    "MiniChem Action 合同冻结快照不是 JSON"
                ) from exc
            if not isinstance(action_contracts, Mapping) or _canonical_bytes(action_contracts) != contract_payload:
                raise MiniChemCatalogFreezeError(
                    "MiniChem Action 合同冻结快照不是 canonical JSON"
                )
            action_contracts = _validate_action_contract_snapshot(
                action_contracts,
                expected_hash,
                snapshot,
            )
    else:
        raise MiniChemCatalogFreezeError("MiniChem 冻结包 manifest schema_version 只能是 1、2 或 3")
    memory_snapshot = deepcopy(dict(snapshot))
    memory_snapshot["open_execution"] = open_execution
    if action_contracts is not None:
        memory_snapshot["action_contracts"] = action_contracts
    return FrozenMiniChemCatalog(
        catalog_hash=expected_hash,
        snapshot=memory_snapshot,
        package_root=package_root,
    )


def load_frozen_minichem_catalog_startup(
    *,
    task_root: str | os.PathLike[str],
    catalog_hash: str,
) -> FrozenMiniChemCatalog:
    """启动/恢复时只核对 Catalog 身份和 Action 合同两个关键摘要。

    Open Execution companion 仍会做严格 JSON 与 schema 校验，但其文件摘要留给
    显式完整复验入口，避免每次启动都扫描所有冻结文件。
    """

    return load_frozen_minichem_catalog(
        task_root=task_root,
        catalog_hash=catalog_hash,
        verify_hashes=("snapshot", "action_contract"),
    )


__all__ = [
    "FrozenMiniChemCatalog",
    "MiniChemCatalogFreezeError",
    "freeze_minichem_catalog",
    "load_frozen_minichem_catalog",
    "load_frozen_minichem_catalog_startup",
]
