"""为现有模型操作调用方延迟选择 Harness。"""

from __future__ import annotations

import os
import threading
import uuid
from typing import Any, Mapping, Sequence

from .base import HarnessError, HarnessRequest
from .config import HarnessConfig
from .context import current_harness_call_context
from .runners import (
    CodexRunner,
    ReactRunner,
)


_OPERATION_ROLES = {
    "workflow_generation": "planner",
    "workflow_optimization": "planner",
    "workflow_revision": "planner",
    "hypothesis_generation": "hypothesis",
    "retrieval": "retrieval",
    "router": "reflection",
    "gaussian_expert": "expert_model",
}


def _request_identity(request: Mapping[str, Any]) -> tuple[str, str]:
    """从一个已编译的 catalog key 解析稳定的 ARCHE role/stage 名称。"""

    catalog_key = str(request.get("catalog_key") or "model")
    operation, separator, catalog_stage = catalog_key.partition(".")
    role = str(
        request.get("role")
        or os.environ.get("ARCHE_HARNESS_ROLE")
        or _OPERATION_ROLES.get(operation, operation)
        or "generic"
    )
    stage = str(
        request.get("stage")
        or os.environ.get("ARCHE_HARNESS_STAGE")
        or (catalog_stage if separator else operation)
        or "model"
    )
    return role, stage


def _resolve_research_task_id(request: Mapping[str, Any]) -> str:
    """在不改变已编译请求结构的前提下解析稳定 research_task scope。

    ModelOperation 请求保持不可变，不携带 Harness bookkeeping 字段。controller
    通过正式的 ``research_task_id`` 请求字段或绑定上下文提供任务身份。没有任务
    scope 的独立调用才使用 UUID；不会从旧 run/task 环境变量或历史 Bundle 路径推断
    活动任务。
    """

    explicit = request.get("research_task_id")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    bound = current_harness_call_context()
    if bound is not None and bound.task_scope_id.strip():
        return bound.task_scope_id.strip()
    return uuid.uuid4().hex


class HarnessDispatcher:
    """在不改变 ARCHE 业务合同的前提下路由已编译请求。"""

    def __init__(self, config: HarnessConfig, *, react_fallback: Any) -> None:
        self.config = config
        self._react_fallback = react_fallback
        self._lock = threading.Lock()
        self._runner = None

    @property
    def kind(self) -> str:
        return self.config.kind

    def _get_runner(self):
        with self._lock:
            if self._runner is None:
                if self.config.kind == "react":
                    self._runner = ReactRunner(self._react_fallback)
                elif self.config.kind == "codex":
                    self._runner = CodexRunner(self.config)
                else:  # pragma: no cover - config validates this
                    raise HarnessError(f"unsupported harness: {self.config.kind}")
            return self._runner

    def run_compiled(self, request: Mapping[str, Any], *, timeout: float | None = 900.0) -> Any:
        if "run_id" in request:
            raise HarnessError(
                "compiled request must use research_task_id; run_id is legacy-only",
                kind="provider_envelope",
            )
        role, stage = _request_identity(request)
        # 已编译模型操作属于 Planner；其它 ARCHE role 固定使用原有 React/API provider。
        if role != "planner" or self.config.driver_for(role) == "react":
            return self._react_fallback(request, timeout=timeout)
        research_task_id = _resolve_research_task_id(request)
        invocation_id = uuid.uuid4().hex
        workspace = self.config.workspace_for(
            role=role,
            research_task_id=research_task_id,
        )
        response_schema = request.get("response_schema")
        if not isinstance(response_schema, Mapping):
            response_schema = None
        raw_tools = request.get("tools", ())
        tools = (
            tuple(dict(item) for item in raw_tools if isinstance(item, Mapping))
            if isinstance(raw_tools, Sequence) and not isinstance(raw_tools, (str, bytes))
            else ()
        )
        if self.config.uses_unbounded_turn_timeout(role=role, stage=stage):
            # Investigation 阶段由配置显式授权无总体 turn 时限；其余阶段仍
            # 采用调用方和 Harness 配置中较短的有限时限。
            request_timeout: float | None = None
        else:
            if timeout is None:
                raise ValueError(
                    "只有显式配置的 Investigation stage 才能使用无限 turn timeout"
                )
            request_timeout = min(float(timeout), self.config.timeout_seconds)
        harness_request = HarnessRequest.from_compiled(
            request,
            role=role,
            stage=stage,
            workspace=workspace,
            research_task_id=research_task_id,
            invocation_id=invocation_id,
            response_schema=response_schema,
            tools=tools,
            # 非 Investigation 请求仍保留调用方的较短 timeout（例如 smoke test
            # 或有限重试），并不会静默替换为更长的 Harness 默认值。显式配置的
            # Investigation 请求则使用上面确定的 None。
            timeout_seconds=request_timeout,
            profile=(
                request.get("profile")
                if isinstance(request.get("profile"), str)
                else None
            ),
        )
        bound = current_harness_call_context()
        if bound is not None:
            from dataclasses import replace

            # 一个可恢复的 Planner 对话属于顶层任务，而不是某个 stage 或组合。
            # 组合身份保留在 prompt/control snapshot 中；若用它作为 session key，
            # 下一个候选组合会因 manifest scope 校验失败。
            session_scope = bound.task_scope_id
            harness_request = replace(
                harness_request,
                metadata={
                    "task_scope_id": bound.task_scope_id,
                    "operation": bound.operation,
                    "operation_instance_id": bound.operation_instance_id,
                    "harness_session_id": f"{role}-{session_scope}",
                },
            )
        response = self._get_runner().run(harness_request)
        return response.to_provider_envelope()

    def run_messages(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        role: str,
        stage: str,
        research_task_id: str | None = None,
    ) -> Any:
        """让未编译的专家请求经过同一个 Harness 边界。"""

        return self._react_fallback(
            {
                "model": os.environ.get("ARCHE_CHEM_MODEL_NAME") or "deepseek-v4-pro",
                "messages": [dict(item) for item in messages],
                "temperature": 0.2,
                "top_p": 0.95,
                "max_tokens": None,
                "stream": False,
                "stop": None,
            }
        )


_DISPATCHERS: dict[tuple[HarnessConfig, int], HarnessDispatcher] = {}
_DISPATCHER_LOCK = threading.Lock()


def get_harness_dispatcher(*, react_fallback: Any) -> HarnessDispatcher:
    config = HarnessConfig.from_env()
    # 缓存键包含完整冻结配置和 fallback 身份。测试或长进程可能在两次调用之间
    # 改变模型、endpoint 或 binary；只按 kind/root 缓存会悄悄保留过期凭据或传输层。
    key = (config, id(react_fallback))
    with _DISPATCHER_LOCK:
        dispatcher = _DISPATCHERS.get(key)
        if dispatcher is None:
            dispatcher = HarnessDispatcher(config, react_fallback=react_fallback)
            _DISPATCHERS[key] = dispatcher
        return dispatcher
