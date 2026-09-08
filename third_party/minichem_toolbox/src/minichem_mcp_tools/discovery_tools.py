"""FastMCP bindings for progressive, task-independent catalog discovery."""

from __future__ import annotations

import hashlib
import threading
import time
from contextvars import copy_context
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, TypeVar

from pydantic import BaseModel

from minichem_toolbox.discovery import (
    browse_action_category as _browse_action_category,
    inspect_action as _inspect_action,
    inspect_backend as _inspect_backend,
    inspect_resource as _inspect_resource,
    list_action_domains as _list_action_domains,
    search_actions as _search_actions,
    search_resources as _search_resources,
)
from minichem_toolbox.catalog import action_specs
from minichem_toolbox import service
from minichem_toolbox.idempotency import (
    OperationIdempotencyStore,
    calculate_request_fingerprint,
    scoped_submission_operation_id,
    scoped_submission_request_error,
    validate_submission_key,
)
from minichem_toolbox.resource_budget import (
    active_resource_jobs,
    active_resource_usage,
    evaluation_resource_budget,
    fixed_job_resource_limits,
    ResourceStateUnavailable,
)
from minichem_toolbox.service import execute_action as _execute_action
from minichem_toolbox.scope import current_scope

from .discovery_models import (
    ActionCategoryBrowseRequest,
    ActionBatchRequest,
    ActionDomainListRequest,
    ActionInspectRequest,
    ActionSearchRequest,
    BackendInspectRequest,
    ProgressiveActionRequest,
    ResourceInspectRequest,
    ResourceSearchRequest,
)
from .tracing import execute_traced


RequestT = TypeVar("RequestT", bound=BaseModel)


@dataclass
class _BatchExecutionState:
    """记录批量 Action 的并发计数；锁只保护计数，不包住科学调用。"""

    lock: threading.Lock
    active_children: int = 0
    peak_concurrency: int = 0

    def enter(self) -> None:
        """登记一个子 Action 开始执行。"""

        with self.lock:
            self.active_children += 1
            self.peak_concurrency = max(self.peak_concurrency, self.active_children)

    def leave(self) -> None:
        """登记一个子 Action 完成执行。"""

        with self.lock:
            self.active_children -= 1


PROGRESSIVE_DISCOVERY_TOOL_NAMES = (
    "list_action_domains",
    "browse_action_category",
    "search_actions",
    "inspect_action",
    "inspect_backend",
    "search_resources",
    "inspect_resource",
    "execute_action",
    "submit_action_batch",
)


TOOL_DESCRIPTIONS = {
    "list_action_domains": (
        "Return the compact complete domain index, counts, and by default every exact action_id "
        "grouped by domain. This does not select a domain or infer anything from the task."
    ),
    "browse_action_category": (
        "Return all compact Action summaries in one exact category selected from "
        "list_action_domains. Use this when the category is known and the caller needs to compare "
        "the complete local choice set before inspecting one Action."
    ),
    "search_actions": (
        "Search the complete Action catalog using action ids, names, aliases, and scientific terms. "
        "Results are ordered by lexical relevance with optional semantic support; the catalog and "
        "registered ids are unchanged."
    ),
    "inspect_action": (
        "Inspect one exact Action before calling it. Pass backend_id and use the default contract "
        "level to receive a compact executable request template with every required field. Use "
        "full when the complete input, output, backend, and service constraint contract is needed."
    ),
    "inspect_backend": (
        "Inspect one exact Backend, its available capabilities and service constraints, and every "
        "Action it provides. Optionally request the exact contract for one Action."
    ),
    "search_resources": (
        "Search all registered scientific data/model/resource entries with explicit neutral filters. "
        "Returns compact exact ids and selection syntax without choosing a family or variant."
    ),
    "inspect_resource": (
        "Resolve one exact resource_id to its public selection contract, supported coverage, and "
        "compatible backends."
    ),
    "execute_action": (
        "Execute one exact predefined Scientific/Data Action selected by action_id. Start from the "
        "compact template returned by inspect_action for this exact backend and replace every "
        "placeholder. If validation returns repair_guidance, correct that same request before "
        "resubmitting it. The dispatcher performs no scientific defaults or fallback. The ActionResult "
        "contains the registered result fields and artifact references."
    ),
    "submit_action_batch": (
        "Run up to 32 independent requests concurrently when they share one Action and backend. "
        "Use this instead of repeated execute_action calls for two or more independent structures. "
        "Only batch_safe Actions are accepted; the service manages concurrency and queues excess "
        "items, while every child keeps an independent result."
    ),
}


def _invoke_discovery(
    name: str,
    request: RequestT,
    function: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    arguments = {"request": request.model_dump(mode="json")}

    def run() -> dict[str, Any]:
        try:
            return function()
        except (KeyError, ValueError) as exc:
            return {
                "status": "invalid_request",
                "error": {
                    "code": "invalid_catalog_discovery_request",
                    "message": str(exc),
                    "exception_type": type(exc).__name__,
                },
                "automatic_fallback": False,
            }
        except OSError as exc:
            return {
                "status": "failed",
                "error": {
                    "code": "catalog_discovery_os_error",
                    "message": str(exc),
                    "exception_type": type(exc).__name__,
                },
                "automatic_fallback": False,
            }

    return execute_traced(name, arguments, run, capture_artifacts=False)


def list_action_domains(request: ActionDomainListRequest) -> dict[str, Any]:
    return _invoke_discovery(
        "list_action_domains",
        request,
        lambda: _list_action_domains(**request.model_dump(mode="python")),
    )


def browse_action_category(request: ActionCategoryBrowseRequest) -> dict[str, Any]:
    return _invoke_discovery(
        "browse_action_category",
        request,
        lambda: _browse_action_category(**request.model_dump(mode="python")),
    )


def search_actions(request: ActionSearchRequest) -> dict[str, Any]:
    return _invoke_discovery(
        "search_actions",
        request,
        lambda: _search_actions(**request.model_dump(mode="python")),
    )


def inspect_action(request: ActionInspectRequest) -> dict[str, Any]:
    return _invoke_discovery(
        "inspect_action",
        request,
        lambda: _inspect_action(**request.model_dump(mode="python")),
    )


def inspect_backend(request: BackendInspectRequest) -> dict[str, Any]:
    return _invoke_discovery(
        "inspect_backend",
        request,
        lambda: _inspect_backend(**request.model_dump(mode="python")),
    )


def search_resources(request: ResourceSearchRequest) -> dict[str, Any]:
    return _invoke_discovery(
        "search_resources",
        request,
        lambda: _search_resources(**request.model_dump(mode="python")),
    )


def inspect_resource(request: ResourceInspectRequest) -> dict[str, Any]:
    return _invoke_discovery(
        "inspect_resource",
        request,
        lambda: _inspect_resource(**request.model_dump(mode="python")),
    )


def execute_action(request: ProgressiveActionRequest) -> dict[str, Any]:
    identity_error = scoped_submission_request_error(request)
    if identity_error is not None:
        return {
            "status": "invalid_request",
            "error": {
                "code": identity_error["code"],
                "message": identity_error["message"],
            },
            "automatic_fallback": False,
        }
    value = request.model_dump(mode="json")
    action_id = str(value.pop("action_id"))
    arguments = {
        "entrypoint": "progressive_execute_action",
        "request": value,
    }
    execute_options = {}
    if request.submission_key is not None:
        execute_options["submission_key"] = request.submission_key
    # Trace under the real scientific Action id so scoring and provenance stay
    # identical to the historical one-tool-per-Action surface.
    return execute_traced(
        action_id,
        arguments,
        lambda: _execute_action(
            action_id,
            value,
            **execute_options,
        ),
    )


def _batch_operation_identity(
    request: ActionBatchRequest,
) -> tuple[str | None, str | None, str]:
    """返回批请求的规范指纹和父 operation 身份。

    输入是已由 Pydantic 校验且带有内部执行身份的批量请求；动作只序列化不含内部
    身份的规范请求并计算 SHA-256。返回 ``(父身份, submission_key, fingerprint)``；缺少身份时
    立即拒绝，不从请求内容派生长期身份。JSON 序列化失败时向 MCP 边界传播。
    """

    canonical_request = request.model_dump(mode="json")
    canonical_request.pop("submission_key", None)
    fingerprint = calculate_request_fingerprint(
        "submit_action_batch", canonical_request
    )
    if current_scope() is not None:
        identity_error = scoped_submission_request_error(request)
        if identity_error is not None:
            raise ValueError(str(identity_error["message"]))
        key = validate_submission_key(str(request.submission_key))
        return scoped_submission_operation_id(key), key, fingerprint
    if request.operation_id is None:
        return None, None, fingerprint
    return request.operation_id, None, fingerprint


def _batch_child_operation_id(
    parent_operation_id: str,
    item_id: str,
    item_index: int,
) -> str:
    """由父身份、稳定 item_id 和数组序号派生安全的子 operation 身份。"""

    seed = f"{parent_operation_id}\0{item_id}\0{item_index}".encode("utf-8")
    digest = hashlib.sha256(seed).hexdigest()
    return f"batch-item-{digest}"


def _batch_child_submission_key(
    parent_submission_key: str,
    item_id: str,
    item_index: int,
) -> str:
    """为 batch 子 Action 生成 scope 内唯一的服务端提交键。"""

    digest = hashlib.sha256(
        f"{parent_submission_key}\0{item_id}\0{item_index}".encode("utf-8")
    ).hexdigest()
    return f"batch-child-{digest}"


def _validate_action_batch_request(request: ActionBatchRequest) -> dict[str, Any] | None:
    """检查批量 Action、后端和 item 身份，失败时返回公开错误。"""

    specification = action_specs().get(request.action_id)
    if specification is None:
        return {
            "status": "invalid_request",
            "error": {"code": "unknown_action", "message": request.action_id},
        }
    if not specification.batch_safe:
        return {
            "status": "invalid_request",
            "error": {
                "code": "action_not_batch_safe",
                "message": f"Action {request.action_id!r} does not declare batch_safe=true.",
            },
        }
    if request.backend_id not in specification.backend_ids:
        return {
            "status": "invalid_request",
            "error": {
                "code": "backend_not_supported",
                "message": f"Backend {request.backend_id!r} does not provide {request.action_id!r}.",
            },
        }
    if len({item.item_id for item in request.items}) != len(request.items):
        return {
            "status": "invalid_request",
            "error": {
                "code": "duplicate_batch_item_id",
                "message": "Every batch item_id must be unique.",
            },
        }
    return None


def _normalize_batch_item_resources(request: ActionBatchRequest) -> list[dict[str, int]]:
    """为每个批量 item 注入同一份服务固定资源。"""

    budget = evaluation_resource_budget()
    return [fixed_job_resource_limits(budget=budget) for _item in request.items]


def _calculate_batch_capacity(
    request: ActionBatchRequest,
    item_resources: list[dict[str, int]],
) -> tuple[int, dict[str, Any]] | dict[str, Any]:
    """计算当前本地预算允许的批量并发度，并拒绝暂时无法容纳的批次。"""

    maximum_item = {
        name: max(resources[name] for resources in item_resources)
        for name in ("cpu_cores", "memory_mb", "gpu_count")
    }
    budget = evaluation_resource_budget()
    try:
        reserved = active_resource_usage()
    except ResourceStateUnavailable as exc:
        return {
            "status": "unavailable",
            "error": exc.as_error(),
            "automatic_fallback": False,
        }
    available = {
        name: max(0, int(getattr(budget, name)) - int(reserved[name]))
        for name in ("cpu_cores", "memory_mb", "gpu_count")
    }
    capacity_limits = [
        available["cpu_cores"] // maximum_item["cpu_cores"],
        available["memory_mb"] // maximum_item["memory_mb"],
    ]
    if maximum_item["gpu_count"] > 0:
        capacity_limits.append(available["gpu_count"] // maximum_item["gpu_count"])
    parallelism = (
        1
        if request.backend_id == "gaussian"
        else min(len(request.items), *capacity_limits)
    )
    capacity = {
        "budget": budget.as_dict(),
        "reserved_at_submission": reserved,
        "available_at_submission": available,
        "maximum_item_resources": maximum_item,
        "active_jobs": active_resource_jobs(),
        "scope": "local_evaluation_budget",
    }
    if request.max_concurrency is not None:
        parallelism = min(parallelism, request.max_concurrency)
    if parallelism < 1:
        return {
            "status": "invalid_request",
            "error": {
                "code": "batch_resource_capacity_unavailable",
                "message": "No batch item currently fits the remaining configured resource budget.",
                "retryable": True,
                "retry_when": "An active job releases enough CPU, memory, and GPU capacity.",
                "resource_capacity": capacity,
            },
        }
    return parallelism, capacity


def _build_batch_action_request(
    request: ActionBatchRequest,
    item: Any,
    item_resources: dict[str, int],
) -> dict[str, Any]:
    """把批量 item 映射为单个 service Action 请求。"""

    return {
        "backend_id": request.backend_id,
        "component_backends": request.component_backends,
        "source_id": None,
        "inputs": item.inputs,
        "method_spec": request.method_spec,
        "action_settings": request.action_settings,
        "resource_limits": item_resources,
    }


def _execute_action_batch_item(
    request: ActionBatchRequest,
    parent_operation_id: str,
    parent_submission_key: str | None,
    index: int,
    item: Any,
    item_resources: dict[str, int],
    state: _BatchExecutionState,
) -> dict[str, Any]:
    """执行一个批量 item，并保存稳定的子 operation 与计时信息。"""

    action_request = _build_batch_action_request(request, item, item_resources)
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    state.enter()
    try:
        child_operation_id = (
            _batch_child_operation_id(parent_operation_id, str(item.item_id), index)
            if parent_submission_key is None
            else None
        )
        child_submission_key = (
            _batch_child_submission_key(parent_submission_key, str(item.item_id), index)
            if parent_submission_key is not None
            else None
        )
        child_kwargs: dict[str, Any] = {"operation_id": child_operation_id}
        if child_submission_key is not None:
            child_kwargs["submission_key"] = child_submission_key
        traced = execute_traced(
            request.action_id,
            {
                "entrypoint": "submit_action_batch",
                "batch_item_id": item.item_id,
                "batch_index": index,
                "request": action_request,
            },
            lambda: service.execute_action(request.action_id, action_request, **child_kwargs),
            capture_artifacts=False,
        )
    finally:
        state.leave()
    return {
        "item_id": item.item_id,
        "batch_index": index,
        "resource_limits": action_request["resource_limits"],
        "started_at": started_at,
        "duration_seconds": round(time.monotonic() - started, 6),
        "result": traced,
    }


def _run_action_batch_items(
    request: ActionBatchRequest,
    parent_operation_id: str,
    parent_submission_key: str | None,
    item_resources: list[dict[str, int]],
    parallelism: int,
) -> tuple[list[dict[str, Any]], int, float]:
    """按有效并发度提交全部 item，并按原数组顺序收集结果。"""

    state = _BatchExecutionState(threading.Lock())
    started = time.monotonic()
    results: list[dict[str, Any] | None] = [None] * len(request.items)
    with ThreadPoolExecutor(
        max_workers=parallelism,
        thread_name_prefix="researchchem-action-batch",
    ) as executor:
        futures = {
            executor.submit(
                copy_context().run,
                _execute_action_batch_item,
                request,
                parent_operation_id,
                parent_submission_key,
                index,
                item,
                item_resources[index],
                state,
            ): index
            for index, item in enumerate(request.items)
        }
        for future, index in futures.items():
            results[index] = future.result()
    return [item for item in results if item is not None], state.peak_concurrency, time.monotonic() - started


def _summarize_action_batch(
    request: ActionBatchRequest,
    completed_results: list[dict[str, Any]],
    parallelism: int,
    peak_concurrency: int,
    capacity: dict[str, Any],
    started: float,
) -> dict[str, Any]:
    """生成批量执行的状态、资源和逐项结果摘要。"""

    successful = sum(
        item["result"].get("status") in {"success", "partial_success"}
        for item in completed_results
    )
    return {
        "status": (
            "success"
            if successful == len(completed_results)
            else "partial_success" if successful else "failed"
        ),
        "action_id": request.action_id,
        "backend_id": request.backend_id,
        "batch_safe": True,
        "execution_mode": (
            "service_managed_serial"
            if request.backend_id == "gaussian"
            else "resource_aware_parallel"
        ),
        "configured_max_concurrency": request.max_concurrency,
        "effective_max_concurrency": parallelism,
        "peak_concurrency": peak_concurrency,
        "wall_duration_seconds": round(time.monotonic() - started, 6),
        "resource_capacity": capacity,
        "item_count": len(completed_results),
        "successful_item_count": successful,
        "failed_item_count": len(completed_results) - successful,
        "items": completed_results,
        "automatic_fallback": False,
    }


def _execute_action_batch_once(
    request: ActionBatchRequest,
    parent_operation_id: str,
    parent_submission_key: str | None = None,
) -> dict[str, Any]:
    """执行一次已由父幂等收据保护的批量 Action。"""

    invalid_request = _validate_action_batch_request(request)
    if invalid_request is not None:
        return invalid_request
    item_resources = _normalize_batch_item_resources(request)
    capacity_result = _calculate_batch_capacity(request, item_resources)
    if isinstance(capacity_result, dict):
        return capacity_result
    parallelism, capacity = capacity_result
    started = time.monotonic()
    completed_results, peak_concurrency, _elapsed = _run_action_batch_items(
        request,
        parent_operation_id,
        parent_submission_key,
        item_resources,
        parallelism,
    )
    return _summarize_action_batch(
        request,
        completed_results,
        parallelism,
        peak_concurrency,
        capacity,
        started,
    )

def _submit_action_batch_in_scope(request: ActionBatchRequest) -> dict[str, Any]:
    """在已认证 scope 中按公开提交键执行批量 Action。"""

    try:
        parent_operation_id, parent_submission_key, fingerprint = (
            _batch_operation_identity(request)
        )
    except ValueError as exc:
        code = "invalid_submission_key"
        if "必须携带" in str(exc):
            code = "missing_submission_key"
        elif "operation_id" in str(exc):
            code = "operation_id_not_assignable"
        return {
            "status": "invalid_request",
            "error": {"code": code, "message": str(exc)},
            "automatic_fallback": False,
        }
    if parent_operation_id is None:
        return {
            "status": "invalid_request",
            "error": {
                "code": "missing_operation_id",
                "message": "legacy 批量执行需要内部 operation_id",
            },
            "automatic_fallback": False,
        }
    store = OperationIdempotencyStore()
    if parent_submission_key is not None:
        result = store.execute_scoped_submission(
            parent_submission_key,
            fingerprint,
            lambda: _execute_action_batch_once(
                request, parent_operation_id, parent_submission_key
            ),
        )
    else:
        result = store.execute_idempotent_operation(
            parent_operation_id,
            fingerprint,
            lambda: _execute_action_batch_once(request, parent_operation_id),
        )
    return dict(result)


def submit_action_batch(request: ActionBatchRequest) -> dict[str, Any]:
    """认证当前消息后，提交并持久化可安全回放的批量 Action。

    输入是由 MCP 内部注入执行身份的批量请求；每条 HTTP 消息先重新认证 capability，
    再在对应工作区读写父 operation 收据并执行子 Action。相同身份与请求指纹直接返回
    已完成响应；身份冲突、running 或损坏收据 fail-closed。没有内部身份时返回
    ``invalid_request``，不从请求内容派生永久幂等身份。
    """

    arguments = {"request": request.model_dump(mode="json")}
    return execute_traced(
        "submit_action_batch",
        arguments,
        lambda: _submit_action_batch_in_scope(request),
        capture_artifacts=False,
    )


def register_progressive_discovery_tools(mcp) -> list[str]:
    functions = {
        name: globals()[name]
        for name in PROGRESSIVE_DISCOVERY_TOOL_NAMES
    }
    for name, function in functions.items():
        mcp.tool(name=name, description=TOOL_DESCRIPTIONS[name])(function)
    return list(functions)


__all__ = [
    *PROGRESSIVE_DISCOVERY_TOOL_NAMES,
    "PROGRESSIVE_DISCOVERY_TOOL_NAMES",
    "register_progressive_discovery_tools",
]
