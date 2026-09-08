"""ARCHE 各可选模型 Harness 的运行器实现。"""

from __future__ import annotations

import json
import os
import uuid
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping

from chemistry_multiagent.utils.api_retry import MODEL_CALL_RETRY_COUNT

from .base import HarnessError, HarnessRequest, HarnessResponse, HarnessRunner
from .codex_sdk_adapter import (
    CodexRecoveryPending,
    acknowledge_codex_completed_turn_receipt,
    build_codex_config,
    build_codex_capability_config,
    calculate_codex_request_digest,
    calculate_codex_thread_policy_digest,
    config_values_to_object,
    resolve_codex_profile,
    run_codex_sdk_turn,
    validate_codex_completed_turn_receipt,
    write_codex_prompt_artifact,
)
from .codex_process_sandbox import (
    CodexProcessSandboxSpec,
    validate_vendored_codex_runtime,
)
from .config import CODEX_API_KEY_ENV, DEFAULT_CODEX_BIN, HarnessConfig, safe_component
from .session_store import PlannerSessionStore


def _prepare_request(
    config: HarnessConfig,
    request: HarnessRequest,
    *,
    harness_name: str,
) -> HarnessRequest:
    """为每个运行器应用一致的 Harness、research_task 和 role 隔离。"""

    return replace(
        request,
        workspace=config.workspace_for(
            role=request.role,
            research_task_id=request.research_task_id,
        ),
        metadata={**request.metadata, "harness": harness_name},
    )


def _trace_stem(request: HarnessRequest) -> str:
    stage = safe_component(request.stage, "stage")
    if not request.invocation_id:
        return stage
    invocation = safe_component(request.invocation_id, "invocation_id")
    return f"{stage}.{invocation}"


def _trace_path(config: HarnessConfig, request: HarnessRequest, suffix: str) -> Path:
    return config.trace_for(
        role=request.role,
        research_task_id=request.research_task_id,
    ) / (_trace_stem(request) + suffix)


def _write_response_schema_artifact(config: HarnessConfig, request: HarnessRequest) -> None:
    """把结构化响应 schema 以 0600 文件写入当前 trace 目录。"""

    if request.response_schema is None:
        return
    schema_path = _trace_path(config, request, ".schema.json")
    schema_path.write_text(
        json.dumps(dict(request.response_schema), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.chmod(schema_path, 0o600)


class ReactRunner(HarnessRunner):
    name = "react"

    def __init__(self, fallback: Callable[[Mapping[str, Any]], Any]) -> None:
        self._fallback = fallback

    def run(self, request: HarnessRequest) -> HarnessResponse:
        # React 实现只在显式选择时使用；原有 provider 调用仍是事实来源，
        # 并由调用方已有的 model-operation 校验器规范化。
        value = self._fallback(
            {
                "model": request.model,
                "messages": list(request.messages),
                "temperature": request.temperature,
                "top_p": request.top_p,
                "max_tokens": request.max_tokens,
                "stream": False,
                "stop": None,
            }
        )
        if isinstance(value, Mapping) and isinstance(value.get("choices"), list):
            return HarnessResponse(
                content=_content_from_envelope(value),
                finish_reason=_finish_from_envelope(value),
                model=str(value.get("model") or request.model),
                usage=value.get("usage") if isinstance(value.get("usage"), Mapping) else {},
                metadata={"harness": self.name, "role": request.role, "stage": request.stage},
            )
        return HarnessResponse(content=str(value), model=request.model, metadata={"harness": self.name})


@dataclass
class _CodexSessionCoordinator:
    """协调一个 Codex 会话的身份登记、恢复记录和成功计数。

    该对象只接收已经脱敏的摘要和会话清单；它在 SDK worker 前登记 thread，
    在 worker 不确定时保存恢复状态，并在成功返回时先核对 receipt 再更新清单。
    任一持久化失败都会保留阻断状态，不会让调用方继续提交新的模型 turn。
    """

    store: PlannerSessionStore
    request: HarnessRequest
    request_digest: str
    thread_policy_digest: str
    initial_session_id: str | None
    manifest: dict[str, Any] | None
    recovery: dict[str, Any] | None
    registered_session_id: str | None
    recovery_persisted_identity: tuple[str | None, str] | None = None

    def register_thread(self, thread_id: str) -> None:
        """在首轮 turn 前登记 thread，或核对恢复 thread 与清单身份一致。"""

        normalized_thread_id = str(thread_id).strip()
        if not normalized_thread_id:
            raise HarnessError(
                "Codex thread registration returned no session_id",
                kind="session_protocol",
            )
        if self.registered_session_id is not None:
            if normalized_thread_id != self.registered_session_id:
                raise HarnessError(
                    "Codex resume returned a different session_id",
                    kind="session_state",
                )
            return
        self.manifest = self.store.create_thread_manifest(
            session_id=normalized_thread_id,
            model=self.request.model,
            stage=self.request.stage,
        )
        self.registered_session_id = normalized_thread_id

    def before_new_turn(self) -> None:
        """在模型 worker 启动前阻断尚未完成确认的恢复状态。"""

        if self.recovery is not None and self.recovery.get("reason") not in {
            "completion_receipt",
            "request_intent",
        }:
            raise HarnessError(
                "Planner session recovery requires explicit recovery confirmation",
                kind="recovery_pending",
            )

    def _record_recovery_pending_or_poison(
        self,
        *,
        turn_id: str | None,
        reason: str,
        turn_count_before: int,
        request_digest: str | None = None,
    ) -> bool:
        """保存恢复状态；失败时写 poison，返回是否完成普通 pending 写入。"""

        recovery_request_digest = request_digest or self.request_digest
        try:
            self.store.record_recovery_pending(
                stage=self.request.stage,
                request_digest=recovery_request_digest,
                thread_policy_digest=self.thread_policy_digest,
                turn_id=turn_id,
                reason=reason,
                turn_count_before=turn_count_before,
            )
            self.recovery_persisted_identity = (turn_id, reason)
            return True
        except BaseException:
            try:
                self.store.record_recovery_poison(
                    stage=self.request.stage,
                    request_digest=recovery_request_digest,
                    thread_policy_digest=self.thread_policy_digest,
                    turn_id=turn_id,
                    reason=reason,
                    turn_count_before=turn_count_before,
                )
            except BaseException:
                raise HarnessError(
                    "Planner recovery persistence failed; manual recovery is required",
                    kind="recovery_pending",
                ) from None
            self.recovery_persisted_identity = (turn_id, reason)
            return False

    def persist_worker_recovery(self, turn_id: str | None) -> None:
        """保存 worker 清理不确定状态，二次失败时以 poison 阻断后续 turn。"""

        if self.recovery_persisted_identity == (turn_id, "worker_cleanup"):
            return
        turn_count_before = self.manifest["turn_count"] if self.manifest is not None else 0
        if not self._record_recovery_pending_or_poison(
            turn_id=turn_id,
            reason="worker_cleanup",
            turn_count_before=turn_count_before,
        ):
            raise HarnessError(
                "Planner recovery persistence failed; manual recovery is required",
                kind="recovery_pending",
            ) from None

    def persist_codex_recovery(self, recovery: CodexRecoveryPending) -> None:
        """为 SDK 已分类的恢复异常落盘；成功写 poison 后交还原异常分类。"""

        if self.recovery_persisted_identity == (recovery.turn_id, recovery.reason):
            return
        current_count = self.manifest["turn_count"] if self.manifest is not None else 0
        if self.recovery is not None and self.recovery.get("reason") == recovery.reason:
            current_count = int(self.recovery.get("turn_count", current_count))
        recovery_request_digest: str | None = None
        if recovery.reason in {"request_intent", "terminal_turn"}:
            candidate = getattr(recovery, "request_digest", None)
            if not isinstance(candidate, str) or not candidate:
                raise HarnessError(
                    "Codex recovery has no original request digest",
                    kind="recovery_pending",
                )
            recovery_request_digest = candidate
        self._record_recovery_pending_or_poison(
            turn_id=recovery.turn_id,
            reason=recovery.reason,
            turn_count_before=current_count,
            request_digest=recovery_request_digest,
        )

    def _resolve_success_session_identity(self, sdk_result: Any) -> str:
        """校验 SDK 返回的 thread 身份与已登记会话完全一致。"""

        returned_session_id = str(sdk_result.thread_id).strip()
        effective_session_id = (
            self.registered_session_id or returned_session_id or self.initial_session_id
        )
        if not effective_session_id:
            raise HarnessError(
                "Codex initial run returned no resumable session_id",
                kind="session_protocol",
            )
        if returned_session_id != effective_session_id:
            raise HarnessError(
                "Codex resume returned a different session_id",
                kind="session_protocol",
            )
        return effective_session_id

    def _read_success_turn_identity(
        self,
        sdk_result: Any,
        *,
        effective_session_id: str,
    ) -> tuple[dict[str, Any], str, int]:
        """读取 manifest、turn 标识和 receipt 所需的旧计数。"""

        manifest = self.store.load()
        if manifest is None or manifest.get("session_id") != effective_session_id:
            raise HarnessError(
                "Codex thread manifest was not registered before turn",
                kind="session_state",
            )
        turn_id = sdk_result.turn_id
        if not isinstance(turn_id, str) or not turn_id.strip():
            raise HarnessError(
                "Codex completed turn returned no turn_id; session progress is blocked",
                kind="session_protocol",
            )
        turn_id = turn_id.strip()
        receipt_turn_count_before = sdk_result.receipt_turn_count_before
        if sdk_result.replayed and receipt_turn_count_before is None:
            raise HarnessError(
                "Replayed Codex turn has no receipt turn count",
                kind="session_state",
            )
        expected_turn_count = (
            receipt_turn_count_before
            if receipt_turn_count_before is not None
            else manifest["turn_count"]
        )
        return manifest, turn_id, expected_turn_count

    def _record_success_manifest(
        self,
        sdk_result: Any,
        *,
        manifest: dict[str, Any],
        effective_session_id: str,
    ) -> dict[str, Any]:
        """按新 turn 或已确认 receipt 推进 manifest 一次。"""

        receipt_turn_count_before = sdk_result.receipt_turn_count_before
        if sdk_result.replayed and receipt_turn_count_before is not None:
            if manifest["turn_count"] == receipt_turn_count_before:
                return self.store.record_turn(
                    manifest,
                    stage=self.request.stage,
                    session_id=effective_session_id,
                )
            if manifest["turn_count"] != receipt_turn_count_before + 1:
                raise HarnessError(
                    "Planner completed receipt turn count conflicts with manifest",
                    kind="session_state",
                )
            return manifest
        return self.store.record_turn(
            manifest,
            stage=self.request.stage,
            session_id=effective_session_id,
        )

    def _record_validated_success_evidence(self, *, turn_id: str) -> None:
        """恢复路径下保存合同通过凭证，供后续清理恢复记录。"""

        if self.recovery is None:
            return
        self.store.record_validated_turn(
            stage=self.request.stage,
            request_digest=self.request_digest,
            thread_policy_digest=self.thread_policy_digest,
            turn_id=turn_id,
        )

    def finalize_success(
        self,
        sdk_result: Any,
        *,
        receipt_root: Path,
    ) -> HarnessResponse:
        """先验证 receipt，再记录 manifest、validated 凭证并确认恢复完成。"""

        effective_session_id = self._resolve_success_session_identity(sdk_result)
        manifest, turn_id, expected_turn_count = self._read_success_turn_identity(
            sdk_result,
            effective_session_id=effective_session_id,
        )
        validate_codex_completed_turn_receipt(
            receipt_root,
            thread_id=effective_session_id,
            turn_id=turn_id,
            request_digest=self.request_digest,
            turn_count_before=expected_turn_count,
            thread_policy_digest=self.thread_policy_digest,
        )
        manifest = self._record_success_manifest(
            sdk_result,
            manifest=manifest,
            effective_session_id=effective_session_id,
        )
        self._record_validated_success_evidence(turn_id=turn_id)
        acknowledge_codex_completed_turn_receipt(
            receipt_root,
            thread_id=effective_session_id,
            turn_id=turn_id,
            request_digest=self.request_digest,
            turn_count_before=expected_turn_count,
            thread_policy_digest=self.thread_policy_digest,
        )
        if self.recovery is not None:
            self.store.clear_recovery_after_validated_turn(
                stage=self.request.stage,
                thread_policy_digest=self.thread_policy_digest,
            )
        return replace(
            sdk_result.response,
            metadata={
                **sdk_result.response.metadata,
                "session_id": effective_session_id,
                "session_resumed": self.initial_session_id is not None,
                "session_manifest_path": str(self.store.manifest_path),
            },
        )


def _create_codex_session_coordinator(
    *,
    session_aware: bool,
    store: PlannerSessionStore,
    request: HarnessRequest,
    request_digest: str,
    thread_policy_digest: str,
    session_id: str | None,
    manifest: dict[str, Any] | None,
    recovery: dict[str, Any] | None,
) -> _CodexSessionCoordinator | None:
    """按请求角色创建会话协调器；一次性 Codex 调用保持旧的无状态语义。"""

    if not session_aware:
        return None
    return _CodexSessionCoordinator(
        store=store,
        request=request,
        request_digest=request_digest,
        thread_policy_digest=thread_policy_digest,
        initial_session_id=session_id,
        manifest=manifest,
        recovery=recovery,
        registered_session_id=session_id,
    )


@dataclass(frozen=True)
class _CodexTurnContext:
    """保存一次 Codex turn 的策略、恢复和 SDK 配置绑定。"""

    request: HarnessRequest
    session_aware: bool
    session_id: str | None
    thread_config: dict[str, Any]
    thread_policy_digest: str
    receipt_root: Path
    manifest: dict[str, Any] | None
    coordinator: _CodexSessionCoordinator | None
    sdk_config: Any


def _resolve_codex_provider_inputs(config: HarnessConfig) -> tuple[str, str, str]:
    """解析 Codex 地址、密钥和模型，缺少任一项时停止本次运行。"""

    base_url = (config.base_url or os.environ.get("ARCHE_CODEX_BASE_URL", "")).strip()
    if not base_url:
        raise HarnessError("Codex Runner 缺少 ARCHE_CODEX_BASE_URL", kind="transport")
    provider_key = os.environ.get(CODEX_API_KEY_ENV, "").strip()
    if not provider_key:
        raise HarnessError(f"Codex Runner 缺少 {CODEX_API_KEY_ENV}", kind="transport")
    model = (config.model or os.environ.get("ARCHE_CODEX_MODEL_NAME", "")).strip()
    if not model:
        raise HarnessError("Codex Runner 缺少 ARCHE_CODEX_MODEL_NAME", kind="transport")
    return base_url, provider_key, model


def _codex_request_session_identity(
    request: HarnessRequest,
) -> tuple[bool, str, str]:
    """返回 session-aware 标志、任务 scope 和 Planner scope。"""

    request_role = str(request.role).strip().lower()
    session_aware = request_role in {"planner", "investigation", "review", "task_review"}
    task_scope_id = str(
        request.metadata.get("task_scope_id")
        or request.research_task_id
        or "planner-research-task"
    )
    planner_scope_id = str(
        request.metadata.get("harness_session_id")
        or f"planner-{request.research_task_id or task_scope_id}"
    )
    return session_aware, task_scope_id, planner_scope_id


def _build_codex_process_sandbox_spec(
    runner: "CodexRunner",
    *,
    request: HarnessRequest,
    codex_home: Path,
) -> CodexProcessSandboxSpec | None:
    """为固定 0.149 wrapper 绑定每会话进程沙箱，测试替身保留旧 seam。

    输入是已经解析的 Runner、角色工作区和 CODEX_HOME；动作是只对仓库固定的
    0.149.0 wrapper 做完整 SHA-256 校验，准备 native Codex/bwrap，并把运行
    MCP 所需的标准 Python、当前源码和 MiniChem 源码作为只读根绑定。非生产
    测试替身返回 ``None``，继续由测试注入 SDK，不会把任意二进制误当作生产运行时。
    运行时校验、目录准备或只读根异常时直接抛出，调用方不会回退到裸 app-server。
    """

    configured_binary = Path(runner.config.codex_bin).expanduser()
    if not configured_binary.is_absolute():
        configured_binary = Path.cwd() / configured_binary
    if configured_binary.resolve(strict=False) != DEFAULT_CODEX_BIN:
        return None
    runtime = validate_vendored_codex_runtime(codex_bin=configured_binary)
    repository_root = Path(__file__).resolve().parents[3]
    read_only_roots = tuple(
        path
        for path in (
            repository_root / ".venv",
            repository_root / "src",
            repository_root / "third_party" / "minichem_toolbox",
        )
        if path.is_dir()
    )
    if len(read_only_roots) != 3:
        raise HarnessError(
            "Codex 进程沙箱缺少 MCP 运行时只读目录",
            kind="transport",
        )
    return CodexProcessSandboxSpec(
        workspace=request.workspace,
        codex_home=codex_home,
        codex_bin=runtime.wrapper,
        runtime_root=runtime.runtime_root,
        native_codex_bin=runtime.native_codex_bin,
        bwrap_bin=runtime.bwrap_bin,
        read_only_roots=read_only_roots,
    )


def _prepare_codex_turn_context(
    runner: "CodexRunner",
    *,
    request: HarnessRequest,
    base_url: str,
    provider_key: str,
    store: PlannerSessionStore,
    session_aware: bool,
) -> _CodexTurnContext:
    """组装并核对本次 turn 的 MCP、能力策略、恢复清单和 SDK 配置。"""

    provider_values = runner._provider_config(base_url=base_url)
    mcp_values = runner._minichem_mcp_config(request)
    thread_config = config_values_to_object([*provider_values, *mcp_values])
    profile = resolve_codex_profile(request)
    effective_thread_config = build_codex_capability_config(
        thread_config=thread_config,
        workspace=request.workspace,
        profile=profile,
        env_key=CODEX_API_KEY_ENV,
        environment_keys=(CODEX_API_KEY_ENV,),
    )
    request_digest = calculate_codex_request_digest(request)
    thread_policy_digest = calculate_codex_thread_policy_digest(
        thread_config=effective_thread_config,
        workspace=request.workspace,
        provider_id=runner._provider_id,
        env_key=CODEX_API_KEY_ENV,
        model=request.model,
        role=request.role,
        stage=request.stage,
        profile=profile,
    )
    receipt_root = request.workspace / "completed_turn_receipts"
    manifest, recovery = (
        store.reconcile_recovery_state(thread_policy_digest=thread_policy_digest)
        if session_aware
        else (None, None)
    )
    session_id = manifest["session_id"] if manifest is not None else None
    if manifest is not None and manifest.get("model") not in {None, request.model}:
        raise HarnessError(
            "Planner session model does not match the current request",
            kind="session_state",
        )
    coordinator = _create_codex_session_coordinator(
        session_aware=session_aware,
        store=store,
        request=request,
        request_digest=request_digest,
        thread_policy_digest=thread_policy_digest,
        session_id=session_id,
        manifest=manifest,
        recovery=recovery,
    )
    codex_home = runner.config.codex_home
    if codex_home is None:
        # 空的、会话稳定的 CODEX_HOME 取代旧 CLI 的 ignore-user-config；它与
        # workspace 保持兄弟目录，避免沙箱把一个可写根嵌套到另一个可写根。
        codex_home = request.workspace.with_name(
            request.workspace.name + ".codex_home"
        )
    elif not codex_home.is_absolute():
        codex_home = Path.cwd() / codex_home
    sandbox_spec = _build_codex_process_sandbox_spec(
        runner,
        request=request,
        codex_home=codex_home,
    )
    sdk_config = build_codex_config(
        codex_bin=runner.config.codex_bin,
        workspace=request.workspace,
        codex_home=codex_home,
        sandbox_spec=sandbox_spec,
        environment_keys=(CODEX_API_KEY_ENV,),
        secret_environment={CODEX_API_KEY_ENV: provider_key},
    )
    return _CodexTurnContext(
        request=request,
        session_aware=session_aware,
        session_id=session_id,
        thread_config=thread_config,
        thread_policy_digest=thread_policy_digest,
        receipt_root=receipt_root,
        manifest=manifest,
        coordinator=coordinator,
        sdk_config=sdk_config,
    )


def _execute_codex_turn_context(
    context: _CodexTurnContext,
    *,
    provider_key: str,
    trace: Path,
) -> HarnessResponse:
    """执行已核对的 SDK turn，持久化输出并完成恢复凭证收束。"""

    coordinator = context.coordinator
    try:
        sdk_result = run_codex_sdk_turn(
            config=context.sdk_config,
            request=context.request,
            thread_config=context.thread_config,
            session_id=context.session_id,
            trace_path=trace,
            provider_id="arche_codex",
            env_key=CODEX_API_KEY_ENV,
            on_thread_ready=coordinator.register_thread if coordinator else None,
            trace_sensitive_values=(provider_key,),
            secret_environment={CODEX_API_KEY_ENV: provider_key},
            completed_turn_receipt_root=context.receipt_root if context.session_aware else None,
            request_digest=(
                coordinator.request_digest if coordinator is not None else None
            ),
            turn_count_before=(context.manifest["turn_count"] if context.manifest is not None else 0)
            if context.session_aware
            else None,
            thread_policy_digest=(
                context.thread_policy_digest if context.session_aware else None
            ),
            before_new_turn=coordinator.before_new_turn if coordinator else None,
            on_recovery_pending=(
                coordinator.persist_worker_recovery if coordinator else None
            ),
            completion_recovery=coordinator.recovery if coordinator else None,
        )
    except CodexRecoveryPending as exc:
        if coordinator is not None:
            coordinator.persist_codex_recovery(exc)
        raise
    response = sdk_result.response
    final_message = trace.with_suffix(".final.md")
    final_message.write_text(response.content + "\n", encoding="utf-8")
    os.chmod(final_message, 0o600)
    if coordinator is not None:
        return coordinator.finalize_success(sdk_result, receipt_root=context.receipt_root)
    return response


class CodexRunner(HarnessRunner):
    name = "codex"
    _provider_id = "arche_codex"

    def __init__(self, config: HarnessConfig) -> None:
        self.config = config

    def _provider_config(self, *, base_url: str) -> list[str]:
        values = [
            "--config",
            f'model_provider={json.dumps(self._provider_id)}',
            "--config",
            f'model_providers.{self._provider_id}.name={json.dumps("ARCHE Codex endpoint")}',
            "--config",
            f'model_providers.{self._provider_id}.base_url={json.dumps(base_url)}',
            "--config",
            f'model_providers.{self._provider_id}.env_key={json.dumps(CODEX_API_KEY_ENV)}',
            "--config",
            f'model_providers.{self._provider_id}.wire_api={json.dumps("responses")}',
            "--config",
            f"model_providers.{self._provider_id}.request_max_retries={MODEL_CALL_RETRY_COUNT}",
            "--config",
            f"model_providers.{self._provider_id}.stream_max_retries={MODEL_CALL_RETRY_COUNT}",
            # ARCHE 的自定义兼容端点可能没有 Codex model metadata。明确关闭
            # reasoning summary 续传，保留模型自身推理，同时避免 resume 把
            # fallback reasoning item 作为下一轮 provider 输入。
            "--config",
            'model_reasoning_summary="none"',
            "--config",
            "model_supports_reasoning_summaries=false",
        ]
        if self.config.reasoning_effort:
            values.extend(
                [
                    "--config",
                    f'model_reasoning_effort={json.dumps(self.config.reasoning_effort)}',
                ]
            )
        return values

    def _minichem_mcp_config(self, request: HarnessRequest) -> list[str]:
        """构造需要 MiniChem 目录的模型 turn 配置。

        MiniChem 是独立工具入口；旧的 Planner 控制 MCP 已从 V2 Harness 中移除。
        当前函数只负责工具服务本身，不读取控制快照，也不启动旧控制 server。
        """

        if request.role != "planner" or os.environ.get("ARCHE_PLANNER_MCP_ENABLED", "1") == "0":
            return []
        repository_root = Path(__file__).resolve().parents[3]
        minichem_startup = (
            repository_root
            / "third_party"
            / "minichem_toolbox"
            / "scripts"
            / "start_mcp.sh"
        ).resolve()
        if not minichem_startup.is_file():
            raise HarnessError(
                f"MiniChem MCP startup script not found: {minichem_startup}",
                kind="transport",
            )
        values = [
            "--config",
            f"mcp_servers.minichem_toolbox.command={json.dumps(str(minichem_startup))}",
            "--config",
            f"mcp_servers.minichem_toolbox.args={json.dumps(['--transport', 'stdio', '--discovery-mode', 'full'])}",
            "--config",
            "mcp_servers.minichem_toolbox.required=true",
            "--config",
            "mcp_servers.minichem_toolbox.startup_timeout_sec=60",
            "--config",
            "mcp_servers.minichem_toolbox.tool_timeout_sec=21600",
            "--config",
            # MiniChem 第三方启动协议仍要求旧变量；仅在此 adapter 边界映射
            # 已解析的正式 Harness workspace，不向 ARCHE 上层公开旧名称。
            f"mcp_servers.minichem_toolbox.env.MINICHEM_MCP_WORKSPACE={json.dumps(str(request.workspace))}",
            "--config",
            f"mcp_servers.minichem_toolbox.env.RESEARCHCHEMBENCH_WORKSPACE={json.dumps(str(request.workspace))}",
        ]
        return values

    def run(self, request: HarnessRequest) -> HarnessResponse:
        """通过固定的 Codex SDK/app-server 运行一个新建或恢复的 Planner turn。"""

        base_url, provider_key, model = _resolve_codex_provider_inputs(self.config)
        # Codex Planner 只使用独立的 Codex 模型；其他角色仍由 Dispatcher 路由到 React。
        request = replace(request, model=model)
        request = _prepare_request(self.config, request, harness_name=self.name)
        trace_dir = self.config.trace_for(
            role=request.role,
            research_task_id=request.research_task_id,
        )
        trace_stem = _trace_stem(request)
        trace = trace_dir / f"{trace_stem}.jsonl"
        final_message = trace_dir / f"{trace_stem}.final.md"
        write_codex_prompt_artifact(request, trace)
        _write_response_schema_artifact(self.config, request)
        # Planner、Investigation 和 Review 都需要跨 turn/进程恢复同一 thread；
        # 其它直接调用仍保留旧的一次性 ephemeral 语义。
        session_aware, task_scope_id, planner_scope_id = _codex_request_session_identity(request)
        store = PlannerSessionStore(
            request.workspace,
            task_scope_id=task_scope_id,
            planner_scope_id=planner_scope_id,
        )

        # 其它直接 Codex 调用仍保留一次性语义；Dispatcher 的普通非 Planner
        # 角色继续走 React，这里同时支持 Investigation/Review 独立调用方。
        lock = store.locked() if session_aware else nullcontext()
        with lock:
            context = _prepare_codex_turn_context(
                self,
                request=request,
                base_url=base_url,
                provider_key=provider_key,
                store=store,
                session_aware=session_aware,
            )
            return _execute_codex_turn_context(
                context,
                provider_key=provider_key,
                trace=trace,
            )


def _content_from_envelope(value: Mapping[str, Any]) -> str:
    try:
        message = value["choices"][0]["message"]
        content = message.get("content") if isinstance(message, Mapping) else None
    except (KeyError, IndexError, TypeError):
        content = None
    return str(content or "")


def _finish_from_envelope(value: Mapping[str, Any]) -> str | None:
    try:
        return value["choices"][0].get("finish_reason")
    except (KeyError, IndexError, TypeError):
        return None
