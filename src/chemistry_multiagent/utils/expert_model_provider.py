"""构造 Gaussian 专家与通用裁判使用的 OpenAI-compatible provider。

本模块只保存进程内的认证信息并在调用时把消息转发到兼容
``chat.completions`` 的服务。对外暴露的运行元数据只包含模型、地址、密钥环境
变量名称和“是否已配置”标记，不包含密钥正文；该元数据可安全写入 TaskCreated
和模型展示文件，用于恢复时校验同一公开配置。
"""

from __future__ import annotations

import base64
import hashlib
import inspect
import json
import math
import os
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import openai

from chemistry_multiagent.utils.model_operation import normalize_provider_envelope
from chemistry_multiagent.utils.llm_api import DEEPSEEK_BASE_URL
from chemistry_multiagent.utils.model_defaults import (
    DEFAULT_EXPERT_FALLBACK_MODEL,
    is_deepseek_v4_pro_model,
    resolve_deepseek_v4_pro_reasoning_effort,
    resolve_general_model,
)
from chemistry_multiagent.utils.api_retry import (
    MODEL_CALL_RETRY_COUNT,
    MODEL_CALL_RETRY_DELAY_SECONDS,
    is_retryable_http_status,
    response_http_status,
    retry_api_call,
)
from chemistry_multiagent.utils.model_identity import model_identity_matches


DEFAULT_EXPERT_API_KEY_ENV = "ARCHE_CHEM_API_KEY"
DEFAULT_CLUSTER_MODEL = "interns2-preview-sft"
DEFAULT_CLUSTER_BASE_URL_ENV = "ARCHE_EXPERT_CLUSTER_BASE_URL"
DEFAULT_CLUSTER_MODEL_ENV = "ARCHE_EXPERT_CLUSTER_MODEL"
DEFAULT_CLUSTER_INGRESS_AK_ENV = "ARCHE_EXPERT_CLUSTER_INGRESS_AK"
DEFAULT_CLUSTER_INGRESS_SK_ENV = "ARCHE_EXPERT_CLUSTER_INGRESS_SK"
DEFAULT_CLUSTER_PUBLIC_API_KEY_ENV = "ARCHE_EXPERT_CLUSTER_PUBLIC_API_KEY"
DEFAULT_GENERAL_API_KEY_ENV = "ARCHE_CHEM_API_KEY"
DEFAULT_FALLBACK_API_KEY_ENV = DEFAULT_GENERAL_API_KEY_ENV
DEFAULT_EXPERT_PROVIDER_OUTPUT_TOKENS = 4096
DEFAULT_GENERAL_PROVIDER_OUTPUT_TOKENS: int | None = None
DEFAULT_EXPERT_STARTUP_PROBE_TIMEOUT_SECONDS = 30.0
# Pro 类模型可能把内部推理计入生成预算；512 仍是轻量探针，但不会让正文在
# 固定 JSON 出现前因 128 token 上限被截断。
DEFAULT_EXPERT_STARTUP_PROBE_MAX_TOKENS = 512
# 严格 JSON 启动探测最多补发一次；认证、连接和模型身份失败不进入该重试。
DEFAULT_EXPERT_STARTUP_PROBE_RETRY_COUNT = 1
EXPERT_RUNTIME_SCHEMA_VERSION = "expert-runtime-selection.v2"
EXPERT_RUNTIME_7B = "arche_chem_7b"
EXPERT_RUNTIME_CLUSTER = "cluster_expert"
EXPERT_RUNTIME_FALLBACK = "fallback"
EXPERT_RUNTIME_STARTUP_FAILED = "startup_failed"


def _canonical_fingerprint(value: Mapping[str, Any]) -> str:
    """根据不含密钥的公开配置计算稳定 SHA-256 指纹。"""

    encoded = json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalise_base_url(value: str | None) -> str | None:
    """清理地址尾部斜杠；空地址表示角色尚未配置。"""

    if value is None:
        return None
    text = value.strip()
    return text.rstrip("/") or None


def _normalise_model(value: str | None) -> str | None:
    """清理模型公开标识；空模型表示角色尚未配置。"""

    if value is None:
        return None
    text = value.strip()
    return text or None


@dataclass(frozen=True)
class ExpertProviderRole:
    """一个专家角色的公开配置和进程内 provider。"""

    role: str
    model: str | None
    base_url: str | None
    api_key_env: str
    api_key_configured: bool
    provider: "OpenAICompatibleExpertProvider | None"
    config_fingerprint: str

    @property
    def configured(self) -> bool:
        """返回模型、地址和密钥是否都已配置。"""

        return self.provider is not None

    def public_metadata(self) -> dict[str, Any]:
        """返回可持久化的角色元数据，不包含认证正文。"""

        return {
            "model": self.model,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "api_key_configured": self.api_key_configured,
            "config_fingerprint": self.config_fingerprint,
        }


@dataclass(frozen=True)
class ExpertProviderBundle:
    """保存专家开关、本地专家、可选集群专家、Flash 回退和 Pro 通用角色的公开元数据。"""

    enabled: bool
    expert: ExpertProviderRole
    general: ExpertProviderRole
    fallback: ExpertProviderRole
    # 集群专家是可选候选。旧调用方没有该字段时省略 cluster，保持原有公开字段格式，
    # 避免旧的测试替身和历史关闭任务被迫升级持久化协议。
    cluster: ExpertProviderRole | None = None
    # 数据库只保存 ``ExpertRuntimeSelection.to_dict()``；provider 对象只留在当前进程。
    runtime_selection: "ExpertRuntimeSelection | None" = None

    @property
    def expert_provider(self):
        """返回 Gaussian 单步审查使用的 provider。"""

        return self.expert.provider

    @property
    def general_provider(self):
        """返回完整结果通用意见和最终裁判使用的 provider。"""

        return self.general.provider

    @property
    def fallback_provider(self):
        """返回本地和集群专家均不可用时使用的 DS Flash provider。"""

        return self.fallback.provider

    @property
    def status(self) -> str:
        """返回 disabled、configured 或 not_configured。"""

        if not self.enabled:
            return "disabled"
        if self.general.configured and (
            self.expert.configured or self.cluster_configured or self.fallback.configured
        ):
            return "configured"
        return "not_configured"

    @property
    def cluster_configured(self) -> bool:
        """返回集群专家候选是否完整配置。"""

        return self.cluster is not None and self.cluster.configured

    def public_metadata(self) -> dict[str, Any]:
        """返回任务创建和模型展示可保存的专家运行元数据。"""

        metadata = {
            "enabled": self.enabled,
            "status": self.status,
            "expert": self.expert.public_metadata(),
            "fallback": self.fallback.public_metadata(),
            "general": self.general.public_metadata(),
        }
        if self.cluster is not None:
            metadata["cluster"] = self.cluster.public_metadata()
        return metadata

    @property
    def selected_provider(self):
        """返回任务冻结后供 Gaussian 与专家观点共同使用的 provider。"""

        if self.runtime_selection is None:
            return None
        return self.runtime_selection.provider


@dataclass(frozen=True)
class ExpertRuntimeSelection:
    """保存一次任务级专家后端选择及探测结论。

    ``provider`` 只在当前进程使用；持久化内容只包含脱敏的 provider 身份和探测
    结果。请求长度由远端 provider 处理，本地不保存 tokenizer、模板或上下文预算。
    """

    status: str
    backend: str
    provider_role: str | None
    provider: Any | None
    provider_metadata: Mapping[str, Any]
    served_model: str | None
    probe: Mapping[str, Any]
    fallback_reason: str | None = None
    failure_role: str | None = None

    def __post_init__(self) -> None:
        """校验启动选择的公开状态和 provider。

        输入是 dataclass 已接收的状态、后端、角色和 provider；动作检查状态枚举、
        注册后端及可调用 provider，不访问认证正文或修改字段。返回值为 ``None``；
        状态、后端或 provider 不符合合同要求时抛出 ``ValueError``，防止冻结无效选择。
        """

        if self.status not in {"selected", EXPERT_RUNTIME_STARTUP_FAILED, "disabled"}:
            raise ValueError("专家运行选择 status 非法")
        if self.provider_role not in {None, "expert", "cluster", "fallback"}:
            raise ValueError("专家运行选择 provider_role 非法")
        if self.failure_role not in {None, "fallback", "general"}:
            raise ValueError("专家运行选择 failure_role 非法")
        if self.status == "selected":
            if self.backend not in {
                EXPERT_RUNTIME_7B,
                EXPERT_RUNTIME_CLUSTER,
                EXPERT_RUNTIME_FALLBACK,
            }:
                raise ValueError("专家运行选择 backend 未注册")
            if not callable(self.provider):
                raise ValueError("专家运行选择缺少可调用 provider")
            expected_role = {
                EXPERT_RUNTIME_7B: "expert",
                EXPERT_RUNTIME_CLUSTER: "cluster",
                EXPERT_RUNTIME_FALLBACK: "fallback",
            }[self.backend]
            if self.provider_role != expected_role:
                raise ValueError("专家运行选择 provider_role 与 backend 不一致")
            if self.failure_role is not None:
                raise ValueError("selected 专家运行选择不能携带 failure_role")
            if self.backend in {EXPERT_RUNTIME_7B, EXPERT_RUNTIME_CLUSTER} and (
                self.fallback_reason is not None
            ):
                raise ValueError("已选专家运行选择不能携带 fallback_reason")
            if self.backend == EXPERT_RUNTIME_FALLBACK and not (
                isinstance(self.fallback_reason, str) and self.fallback_reason.strip()
            ):
                raise ValueError("Flash 回退选择必须携带 7B 失败原因")
        elif self.status == EXPERT_RUNTIME_STARTUP_FAILED:
            if self.backend != EXPERT_RUNTIME_STARTUP_FAILED:
                raise ValueError("startup_failed 专家运行选择 backend 非法")
            if self.provider_role is not None or self.provider is not None:
                raise ValueError("startup_failed 专家运行选择不得携带可调用 provider")
            if self.failure_role is None:
                raise ValueError("startup_failed 专家运行选择必须携带 failure_role")
            if not (
                isinstance(self.fallback_reason, str) and self.fallback_reason.strip()
            ):
                raise ValueError("startup_failed 专家运行选择必须携带失败原因")
        else:
            if self.backend != "disabled":
                raise ValueError("disabled 专家运行选择 backend 非法")
            if self.provider_role is not None or self.provider is not None:
                raise ValueError("disabled 专家运行选择不得携带可调用 provider")
            if self.fallback_reason is not None:
                raise ValueError("disabled 专家运行选择不得携带失败原因")
        if self.status != EXPERT_RUNTIME_STARTUP_FAILED and self.failure_role is not None:
            raise ValueError("非 startup_failed 专家运行选择不能携带 failure_role")

    @property
    def ready(self) -> bool:
        """返回是否可以进入科学阶段。"""

        return self.status == "selected"

    def to_dict(self) -> dict[str, Any]:
        """返回无密钥、可写入 TaskCreated/checkpoint 的冻结事实。"""

        return {
            "schema_version": EXPERT_RUNTIME_SCHEMA_VERSION,
            "status": self.status,
            "backend": self.backend,
            "provider_role": self.provider_role,
            "provider": dict(self.provider_metadata),
            "served_model": self.served_model,
            "probe": dict(self.probe),
            "fallback_reason": self.fallback_reason,
            "failure_role": self.failure_role,
        }


def _extract_probe_json(response: Any) -> tuple[bool, str | None, str]:
    """从 provider 响应提取严格 JSON 探测结果和 served model。"""

    if not isinstance(response, Mapping):
        return False, None, "provider_probe_not_object"
    served_model = response.get("model")
    if not isinstance(served_model, str) or not served_model.strip():
        served_model = None
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        return False, served_model, "provider_probe_invalid_choices"
    choice = choices[0]
    if not isinstance(choice, Mapping) or choice.get("finish_reason") != "stop":
        return False, served_model, "provider_probe_truncated"
    message = choice.get("message") if isinstance(choice, Mapping) else None
    content = message.get("content") if isinstance(message, Mapping) else None
    if not isinstance(content, str) or not content.strip():
        return False, served_model, "provider_probe_empty_content"

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        """拒绝重复键，避免探测 JSON 依赖解析器的最后一次赋值。"""

        parsed_pairs: dict[str, Any] = {}
        for key, value in pairs:
            if key in parsed_pairs:
                raise ValueError("provider_probe_duplicate_json_key")
            parsed_pairs[key] = value
        return parsed_pairs

    try:
        parsed = json.loads(content, object_pairs_hook=reject_duplicate_keys)
    except (TypeError, ValueError):
        return False, served_model, "provider_probe_invalid_json"
    if not isinstance(parsed, dict) or set(parsed) != {"ok"} or parsed["ok"] is not True:
        return False, served_model, "provider_probe_schema_failure"
    return True, served_model, "ok"


_TRANSIENT_EXPERT_STARTUP_PROBE_REASONS = frozenset(
    {
        "provider_probe_invalid_json",
        "provider_probe_schema_failure",
        "provider_probe_truncated",
    }
)


def _run_expert_startup_probe(
    provider: Any,
    *,
    configured_model: str | None,
    timeout_seconds: float,
) -> tuple[bool, dict[str, Any], str | None]:
    """执行最小严格 JSON provider 探测，协议输出波动时最多补发一次。

    输入是可调用 provider、配置模型名和超时秒数；动作发送固定短请求并检查连接、
    认证、服务模型和严格 JSON 响应，不测量长度或依赖本地 tokenizer。返回值是
    ``(是否成功, 脱敏事实, 原因码)``；只有已通过连接、认证和模型身份检查的
    非法 JSON、结构不符合要求或输出截断时允许一次重试，其他 provider 错误保持
    失败，不把异常正文写入任务记录。
    """

    facts: dict[str, Any] = {
        "connectivity": False,
        "authentication": False,
        "served_model": None,
        "strict_json": False,
        "reason_code": None,
    }

    def finish(ok: bool, reason: str | None) -> tuple[bool, dict[str, Any], str | None]:
        """写入安全失败类别，避免把异常正文放进 TaskCreated。"""

        facts["reason_code"] = "ok" if ok else reason
        return ok, facts, reason

    if not callable(provider):
        return finish(False, "provider_unconfigured")
    messages = [
        {"role": "system", "content": "Return exactly {\"ok\":true} as JSON."},
        {"role": "user", "content": "{\"probe\":\"ok\"}"},
    ]
    for attempt_index in range(DEFAULT_EXPERT_STARTUP_PROBE_RETRY_COUNT + 1):
        # 每次尝试只保留该次调用的脱敏事实，避免重试后把旧的成功标记带入失败结果。
        facts.update(
            {
                "connectivity": False,
                "authentication": False,
                "served_model": None,
                "strict_json": False,
            }
        )
        try:
            try:
                signature = inspect.signature(provider)
                accepts_timeout = "timeout" in signature.parameters or any(
                    parameter.kind == inspect.Parameter.VAR_KEYWORD
                    for parameter in signature.parameters.values()
                )
            except (TypeError, ValueError):
                accepts_timeout = False
            probe_arguments: dict[str, Any] = {}
            if accepts_timeout:
                probe_arguments["timeout"] = timeout_seconds
            # Gaussian 专家和回退模型只需返回很短的 JSON，因此探针保留轻量输出预算。
            # 通用模型的正式调用不设输出上限，探针也沿用这一合同，避免推理模型在
            # 输出 JSON 前耗尽预算。测试替身不强制接受该扩展参数，因此只对本模块
            # provider 生效。
            if (
                isinstance(provider, OpenAICompatibleExpertProvider)
                and provider.role != "general"
            ):
                probe_arguments["max_tokens"] = DEFAULT_EXPERT_STARTUP_PROBE_MAX_TOKENS
            response = provider(messages, **probe_arguments)
        except Exception as exc:
            name = type(exc).__name__.casefold()
            if "auth" in name or "permission" in name:
                return finish(False, "provider_authentication_error")
            if "refus" in name or "content" in name:
                return finish(False, "provider_model_rejected")
            return finish(False, "provider_transport_error")
        facts["connectivity"] = True
        facts["authentication"] = True
        ok, served_model, reason = _extract_probe_json(response)
        facts["served_model"] = served_model
        if served_model is None:
            return finish(False, "provider_probe_missing_served_model")
        if not model_identity_matches(configured_model, served_model):
            return finish(False, "provider_probe_model_identity_mismatch")
        facts["strict_json"] = ok
        if ok:
            return finish(True, None)

        if (
            reason in _TRANSIENT_EXPERT_STARTUP_PROBE_REASONS
            and attempt_index < DEFAULT_EXPERT_STARTUP_PROBE_RETRY_COUNT
        ):
            # 只有服务已连通、身份已核对且输出属于三类瞬时协议波动时才补发一次。
            continue
        if reason == "provider_probe_invalid_json":
            return finish(False, "provider_invalid_json")
        if reason in {"provider_probe_truncated", "provider_probe_schema_failure"}:
            return finish(False, "provider_model_rejected")
        return finish(False, "provider_protocol_error")
    raise AssertionError("专家启动探针重试循环必须返回")


def select_expert_runtime(
    bundle: ExpertProviderBundle,
    *,
    probe_timeout_seconds: float = DEFAULT_EXPERT_STARTUP_PROBE_TIMEOUT_SECONDS,
) -> ExpertProviderBundle:
    """按本地、集群、DeepSeek Flash 顺序探测并冻结新任务的专家后端。

    本地专家成功时不触碰集群和 Flash；本地失败后才探测可选集群专家，集群也失败
    才探测 Flash。任一专家后端被选中后，还必须确认已配置的通用模型可以承担
    General opinion/Judge。探测只发送最小严格 JSON，不读取 Working Memory，
    也不测量长度或依赖本地 tokenizer。
    """

    if not isinstance(bundle, ExpertProviderBundle):
        raise TypeError("bundle 必须是 ExpertProviderBundle")
    if (
        isinstance(probe_timeout_seconds, bool)
        or not isinstance(probe_timeout_seconds, (int, float))
        or probe_timeout_seconds <= 0
    ):
        raise ValueError("probe_timeout_seconds 必须是正数")
    probe_timeout_seconds = float(probe_timeout_seconds)

    has_cluster_candidate = bundle.cluster is not None

    def build_probe_facts(
        *,
        expert_probe: Mapping[str, Any] | None,
        cluster_probe: Mapping[str, Any] | None,
        fallback_probe: Mapping[str, Any] | None,
        general_probe: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """构造探测事实；没有集群候选时省略可选集群字段，保留本地、Flash 和 Pro 三项。"""

        facts: dict[str, Any] = {
            "expert": dict(expert_probe) if expert_probe is not None else None,
            "fallback": dict(fallback_probe) if fallback_probe is not None else None,
            "general": dict(general_probe) if general_probe is not None else None,
        }
        if has_cluster_candidate:
            facts["cluster"] = (
                dict(cluster_probe) if cluster_probe is not None else None
            )
        return facts

    if not bundle.enabled:
        disabled_provider_metadata = bundle.expert.public_metadata()
        selection = ExpertRuntimeSelection(
            status="disabled",
            backend="disabled",
            provider_role=None,
            provider=None,
            provider_metadata=disabled_provider_metadata,
            served_model=None,
            probe=build_probe_facts(
                expert_probe=None,
                cluster_probe=None,
                fallback_probe=None,
                general_probe=None,
            ),
        )
        return replace(bundle, runtime_selection=selection)

    def freeze_startup_failure(
        *,
        expert_probe: Mapping[str, Any],
        cluster_probe: Mapping[str, Any] | None,
        fallback_probe: Mapping[str, Any] | None,
        general_probe: Mapping[str, Any],
        failure_reason: str,
        failure_role: str,
    ) -> ExpertProviderBundle:
        """冻结必需后端无法组成完整专家流程的启动失败事实。"""

        failure_provider = (
            bundle.fallback
            if failure_role == "fallback"
            else bundle.general
        )
        selection = ExpertRuntimeSelection(
            status=EXPERT_RUNTIME_STARTUP_FAILED,
            backend=EXPERT_RUNTIME_STARTUP_FAILED,
            provider_role=None,
            provider=None,
            provider_metadata=failure_provider.public_metadata(),
            served_model=(
                fallback_probe.get("served_model")
                if failure_role == "fallback"
                and fallback_probe is not None
                else general_probe.get("served_model")
            ),
            probe=build_probe_facts(
                expert_probe=expert_probe,
                cluster_probe=cluster_probe,
                fallback_probe=fallback_probe,
                general_probe=general_probe,
            ),
            fallback_reason=failure_reason,
            failure_role=failure_role,
        )
        return replace(bundle, runtime_selection=selection)

    expert_ok, expert_probe, expert_reason = _run_expert_startup_probe(
        bundle.expert_provider,
        configured_model=bundle.expert.model,
        timeout_seconds=probe_timeout_seconds,
    )
    if expert_ok:
        # 7B 承担单步专家职责，但进入科学阶段前仍须确认已配置的通用 provider
        # 可以完成 General 意见和 Judge；两项探测都成功后才冻结 7B 选择。
        general_ok, general_probe, general_reason = _run_expert_startup_probe(
            bundle.general_provider,
            configured_model=bundle.general.model,
            timeout_seconds=probe_timeout_seconds,
        )
        if not general_ok:
            return freeze_startup_failure(
                expert_probe=expert_probe,
                cluster_probe=None,
                fallback_probe=None,
                general_probe=general_probe,
                failure_reason=general_reason or "provider_connectivity_or_auth_failed",
                failure_role="general",
            )
        selection = ExpertRuntimeSelection(
            status="selected",
            backend=EXPERT_RUNTIME_7B,
            provider_role="expert",
            provider=bundle.expert_provider,
            provider_metadata=bundle.expert.public_metadata(),
            served_model=expert_probe.get("served_model"),
            probe=build_probe_facts(
                expert_probe=expert_probe,
                cluster_probe=None,
                fallback_probe=None,
                general_probe=general_probe,
            ),
        )
        return replace(
            bundle,
            expert=replace(bundle.expert, provider=bundle.expert_provider),
            runtime_selection=selection,
        )

    cluster_ok = False
    cluster_probe: dict[str, Any] | None = None
    cluster_reason: str | None = None
    if has_cluster_candidate:
        cluster_role = bundle.cluster
        cluster_ok, cluster_probe, cluster_reason = _run_expert_startup_probe(
            cluster_role.provider if cluster_role is not None else None,
            configured_model=cluster_role.model if cluster_role is not None else None,
            timeout_seconds=probe_timeout_seconds,
        )
        if cluster_ok:
            general_ok, general_probe, general_reason = _run_expert_startup_probe(
                bundle.general_provider,
                configured_model=bundle.general.model,
                timeout_seconds=probe_timeout_seconds,
            )
            if not general_ok:
                return freeze_startup_failure(
                    expert_probe=expert_probe,
                    cluster_probe=cluster_probe,
                    fallback_probe=None,
                    general_probe=general_probe,
                    failure_reason=general_reason or "provider_connectivity_or_auth_failed",
                    failure_role="general",
                )
            selection = ExpertRuntimeSelection(
                status="selected",
                backend=EXPERT_RUNTIME_CLUSTER,
                provider_role="cluster",
                provider=cluster_role.provider if cluster_role is not None else None,
                provider_metadata=(
                    cluster_role.public_metadata() if cluster_role is not None else {}
                ),
                served_model=cluster_probe.get("served_model"),
                probe=build_probe_facts(
                    expert_probe=expert_probe,
                    cluster_probe=cluster_probe,
                    fallback_probe=None,
                    general_probe=general_probe,
                ),
            )
            return replace(bundle, runtime_selection=selection)

    fallback_ok, fallback_probe, fallback_reason = _run_expert_startup_probe(
        bundle.fallback_provider,
        configured_model=bundle.fallback.model,
        timeout_seconds=probe_timeout_seconds,
    )
    general_ok, general_probe, general_reason = _run_expert_startup_probe(
        bundle.general_provider,
        configured_model=bundle.general.model,
        timeout_seconds=probe_timeout_seconds,
    )
    if not fallback_ok:
        return freeze_startup_failure(
            expert_probe=expert_probe,
            cluster_probe=cluster_probe,
            fallback_probe=fallback_probe,
            general_probe=general_probe,
            failure_reason=fallback_reason or "provider_connectivity_or_auth_failed",
            failure_role="fallback",
        )
    if not general_ok:
        return freeze_startup_failure(
            expert_probe=expert_probe,
            cluster_probe=cluster_probe,
            fallback_probe=fallback_probe,
            general_probe=general_probe,
            failure_reason=general_reason or "provider_connectivity_or_auth_failed",
            failure_role="general",
        )
    selection = ExpertRuntimeSelection(
        status="selected",
        backend=EXPERT_RUNTIME_FALLBACK,
        provider_role="fallback",
        provider=bundle.fallback_provider,
        provider_metadata=bundle.fallback.public_metadata(),
        served_model=fallback_probe.get("served_model"),
        probe=build_probe_facts(
            expert_probe=expert_probe,
            cluster_probe=cluster_probe,
            fallback_probe=fallback_probe,
            general_probe=general_probe,
        ),
        fallback_reason=(cluster_reason if has_cluster_candidate else expert_reason),
    )
    return replace(
        bundle,
        runtime_selection=selection,
    )


class OpenAICompatibleExpertProvider:
    """把专家消息调用到一个 OpenAI-compatible chat completions 服务。

    默认使用 OpenAI 风格 Bearer API key。集群候选可额外提供 Basic 认证的账号和
    密钥以及 ``x-api-key``；认证头只存在于当前 OpenAI client，不进入公开元数据。
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        role: str,
        basic_auth_username: str | None = None,
        basic_auth_password: str | None = None,
        public_api_key: str | None = None,
        supports_json_mode: bool = True,
        timeout: int = 900,
        default_max_tokens: int | None = None,
        retry_count: int = MODEL_CALL_RETRY_COUNT,
        retry_delay: float = MODEL_CALL_RETRY_DELAY_SECONDS,
    ) -> None:
        """保存 provider 调用配置。

        输入是模型、服务地址、认证信息、角色、超时和可选默认输出上限；动作校验
        并保存运行时字段，认证信息只留在当前进程，不参与元数据序列化。General
        provider 默认不设置本地上限，Gaussian 专家和 Flash fallback 默认仍使用
        4096；显式正整数可以覆盖这两个角色的默认值。返回值为 ``None``；参数
        缺失、类型错误或数值不合法时抛出 ``ValueError``。
        """

        if not model or not model.strip():
            raise ValueError("expert provider model 必须是非空字符串")
        if not base_url or not base_url.strip():
            raise ValueError("expert provider base_url 必须是非空字符串")
        if not api_key or not api_key.strip():
            raise ValueError("expert provider api_key 必须是非空字符串")
        if not role or not role.strip():
            raise ValueError("expert provider role 必须是非空字符串")
        normalized_role = role.strip()
        if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
            raise ValueError("expert provider timeout 必须是正整数")
        if (
            default_max_tokens is not None
            and (
                isinstance(default_max_tokens, bool)
                or not isinstance(default_max_tokens, int)
                or default_max_tokens <= 0
            )
        ):
            raise ValueError("expert provider default_max_tokens 必须是正整数或 None")
        if (
            isinstance(retry_count, bool)
            or not isinstance(retry_count, int)
            or retry_count < 0
        ):
            raise ValueError("expert provider retry_count 必须是非负整数")
        if (
            isinstance(retry_delay, bool)
            or not isinstance(retry_delay, (int, float))
            or not math.isfinite(float(retry_delay))
            or retry_delay < 0
        ):
            raise ValueError("expert provider retry_delay 必须是非负数字")
        if default_max_tokens is None:
            resolved_default_max_tokens = (
                DEFAULT_GENERAL_PROVIDER_OUTPUT_TOKENS
                if normalized_role == "general"
                else DEFAULT_EXPERT_PROVIDER_OUTPUT_TOKENS
            )
        else:
            resolved_default_max_tokens = default_max_tokens
        self.model = model.strip()
        self.base_url = base_url.strip().rstrip("/")
        self.role = normalized_role
        self.timeout = timeout
        self.default_max_tokens = resolved_default_max_tokens
        self.retry_count = retry_count
        self.retry_delay = float(retry_delay)
        self.reasoning_effort = (
            resolve_deepseek_v4_pro_reasoning_effort()
            if is_deepseek_v4_pro_model(self.model)
            else None
        )
        if (basic_auth_username is None) != (basic_auth_password is None):
            raise ValueError("Basic 认证必须同时提供用户名和密码")
        if basic_auth_username is not None and (
            not basic_auth_username.strip() or not basic_auth_password.strip()
        ):
            raise ValueError("Basic 认证用户名和密码必须是非空字符串")
        if public_api_key is not None and not public_api_key.strip():
            raise ValueError("x-api-key 必须是非空字符串")
        if not isinstance(supports_json_mode, bool):
            raise TypeError("expert provider supports_json_mode 必须是 boolean")
        self.supports_json_mode = supports_json_mode
        default_headers: dict[str, str] = {}
        if basic_auth_username is not None and basic_auth_password is not None:
            basic_token = base64.b64encode(
                f"{basic_auth_username}:{basic_auth_password}".encode("utf-8")
            ).decode("ascii")
            default_headers["Authorization"] = f"Basic {basic_token}"
        if public_api_key is not None:
            default_headers["x-api-key"] = public_api_key
        self._client = openai.OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            max_retries=0,
            default_headers=default_headers or None,
        )

    def __call__(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_tokens: int | None = None,
        timeout: int | float | None = None,
        retry_count: int | None = None,
        retry_delay: float | None = None,
    ) -> Any:
        """发送一次非流式对话并返回标准 provider envelope。

        ``max_tokens`` 只控制 provider 允许生成的最大输出；General provider 默认
        不设置该参数，而 Gaussian 专家和 Flash fallback 仍使用各自的 4096 默认值。
        DeepSeek v4 Pro 发送环境配置的 ``reasoning_effort``，并仅依赖本地
        JSON/schema 校验，不发送远端 ``response_format``。输入长度和截断策略由
        远端服务处理，ARCHE 不做本地 token 计数。每次调用都会重新读取该配置，
        因此非法值在发请求前失败，审计和实际发送保持一致。
        """

        if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
            raise TypeError("expert provider messages 必须是消息数组")
        if max_tokens is None:
            resolved_max_tokens = self.default_max_tokens
        elif (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or max_tokens <= 0
        ):
            raise ValueError("expert provider max_tokens 必须是正整数")
        else:
            resolved_max_tokens = max_tokens
        if timeout is None:
            resolved_timeout: int | float = self.timeout
        elif (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or timeout <= 0
        ):
            raise ValueError("expert provider timeout 必须是正数")
        else:
            resolved_timeout = timeout
        resolved_retry_count = self.retry_count if retry_count is None else retry_count
        resolved_retry_delay = self.retry_delay if retry_delay is None else retry_delay
        if (
            isinstance(resolved_retry_count, bool)
            or not isinstance(resolved_retry_count, int)
            or resolved_retry_count < 0
        ):
            raise ValueError("expert provider retry_count 必须是非负整数")
        if (
            isinstance(resolved_retry_delay, bool)
            or not isinstance(resolved_retry_delay, (int, float))
            or not math.isfinite(float(resolved_retry_delay))
            or resolved_retry_delay < 0
        ):
            raise ValueError("expert provider retry_delay 必须是非负数字")
        copied = [dict(message) for message in messages]
        request_kwargs = {
            "model": self.model,
            "messages": copied,
            "temperature": 0.0,
            "timeout": resolved_timeout,
        }
        if self.reasoning_effort is not None:
            self.reasoning_effort = resolve_deepseek_v4_pro_reasoning_effort()
            request_kwargs.update(
                {
                    "top_p": 1.0,
                    "stream": False,
                    "reasoning_effort": self.reasoning_effort,
                }
            )
        elif self.supports_json_mode:
            request_kwargs.update(
                {
                    "top_p": 1.0,
                    "stream": False,
                    "response_format": {"type": "json_object"},
                }
            )
        if resolved_max_tokens is not None:
            request_kwargs["max_tokens"] = resolved_max_tokens
        response = retry_api_call(
            lambda: self._client.chat.completions.create(**request_kwargs),
            retry_count=resolved_retry_count,
            retry_delay=float(resolved_retry_delay),
            retryable_result=lambda value: is_retryable_http_status(
                response_http_status(value)
            ),
        )
        return normalize_provider_envelope(response)


def _build_role(
    *,
    role: str,
    model: str | None,
    base_url: str | None,
    api_key: str | None,
    api_key_env: str,
    timeout: int,
) -> ExpertProviderRole:
    """按角色配置构造 provider，并计算不含密钥正文的指纹。"""

    normalized_model = _normalise_model(model)
    normalized_base_url = _normalise_base_url(base_url)
    normalized_key_env = api_key_env.strip() if isinstance(api_key_env, str) else ""
    if not normalized_key_env:
        raise ValueError(f"{role} api_key_env 必须是非空字符串")
    key_present = isinstance(api_key, str) and bool(api_key.strip())
    fingerprint = _canonical_fingerprint(
        {
            "role": role,
            "model": normalized_model,
            "base_url": normalized_base_url,
            "api_key_env": normalized_key_env,
            "api_key_configured": key_present,
        }
    )
    provider = None
    if normalized_model and normalized_base_url and key_present:
        provider = OpenAICompatibleExpertProvider(
            model=normalized_model,
            base_url=normalized_base_url,
            api_key=api_key.strip(),
            role=role,
            timeout=timeout,
        )
    return ExpertProviderRole(
        role=role,
        model=normalized_model,
        base_url=normalized_base_url,
        api_key_env=normalized_key_env,
        api_key_configured=key_present,
        provider=provider,
        config_fingerprint=fingerprint,
    )


def _build_cluster_role(
    *,
    model: str | None,
    base_url: str | None,
    ingress_ak: str | None,
    ingress_sk: str | None,
    public_api_key: str | None,
    timeout: int,
) -> ExpertProviderRole:
    """构造集群专家角色，并把 Basic/x-api-key 凭据限制在进程内。

    输入是集群公开模型/地址和三段认证正文；动作仅在三段认证齐全时创建带请求头
    的 OpenAI client，公开角色元数据只记录固定环境变量名与齐全标记。返回值是可
    持久化的角色对象；缺任一字段时 provider 为 ``None``，后续启动选择会继续
    尝试 DeepSeek Flash，而不会把异常正文或认证值写入任务事实。
    """

    normalized_model = _normalise_model(model)
    normalized_base_url = _normalise_base_url(base_url)
    api_key_env = DEFAULT_CLUSTER_PUBLIC_API_KEY_ENV
    credentials_configured = all(
        isinstance(value, str) and bool(value.strip())
        for value in (ingress_ak, ingress_sk, public_api_key)
    )
    fingerprint = _canonical_fingerprint(
        {
            "role": "cluster",
            "model": normalized_model,
            "base_url": normalized_base_url,
            "api_key_env": api_key_env,
            "api_key_configured": credentials_configured,
        }
    )
    provider = None
    if normalized_model and normalized_base_url and credentials_configured:
        assert ingress_ak is not None
        assert ingress_sk is not None
        assert public_api_key is not None
        provider = OpenAICompatibleExpertProvider(
            model=normalized_model,
            base_url=normalized_base_url,
            # OpenAI client 仍需要一个非空 API key；Authorization 会由 default_headers
            # 覆盖为 Basic，x-api-key 则由同一处额外头发送。
            api_key=public_api_key.strip(),
            role="cluster",
            basic_auth_username=ingress_ak.strip(),
            basic_auth_password=ingress_sk.strip(),
            public_api_key=public_api_key.strip(),
            supports_json_mode=False,
            timeout=timeout,
        )
    return ExpertProviderRole(
        role="cluster",
        model=normalized_model,
        base_url=normalized_base_url,
        api_key_env=api_key_env,
        api_key_configured=credentials_configured,
        provider=provider,
        config_fingerprint=fingerprint,
    )


def build_expert_provider_bundle(
    *,
    enabled: bool = True,
    expert_model: str | None = None,
    expert_base_url: str | None = None,
    expert_api_key: str | None = None,
    expert_api_key_env: str = DEFAULT_EXPERT_API_KEY_ENV,
    cluster_model: str | None = None,
    cluster_base_url: str | None = None,
    cluster_ingress_ak: str | None = None,
    cluster_ingress_sk: str | None = None,
    cluster_public_api_key: str | None = None,
    fallback_model: str | None = None,
    fallback_base_url: str | None = None,
    fallback_api_key: str | None = None,
    fallback_api_key_env: str = DEFAULT_FALLBACK_API_KEY_ENV,
    general_model: str | None = None,
    general_base_url: str | None = None,
    general_api_key: str | None = None,
    general_api_key_env: str = DEFAULT_GENERAL_API_KEY_ENV,
    timeout: int = 900,
) -> ExpertProviderBundle:
    """构造专家、可选集群专家、回退和通用 provider。

    未显式传入角色参数时，专家、回退和通用角色统一读取
    ``ARCHE_CHEM_API_KEY/ARCHE_CHEM_BASE_URL/ARCHE_CHEM_MODEL_NAME``；集群候选只有
    在显式提供集群凭据时才加入。这样 React 与各 Harness 的默认模型路由保持一致，
    同时保留远端新增的可选集群专家能力。
    """

    if not isinstance(enabled, bool):
        raise TypeError("expert provider enabled 必须是 boolean")
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
        raise ValueError("expert provider timeout 必须是正整数")
    if not enabled:
        # 显式关闭或恢复历史关闭任务时不读取任何环境变量，也不初始化 OpenAI
        # client。元数据恒定为中性值，不会因本机后来配置了 DEEPSEEK 或专家密钥
        # 而被误判为“换了模型”。
        disabled_role = ExpertProviderRole(
            role="disabled",
            model=None,
            base_url=None,
            api_key_env="",
            api_key_configured=False,
            provider=None,
            config_fingerprint="0" * 64,
        )
        return ExpertProviderBundle(
            enabled=False,
            expert=disabled_role,
            general=disabled_role,
            fallback=disabled_role,
        )
    configured_model = os.environ.get("ARCHE_CHEM_MODEL_NAME", "").strip() or None
    configured_base = os.environ.get("ARCHE_CHEM_BASE_URL", "").strip() or None
    configured_key = os.environ.get("ARCHE_CHEM_API_KEY", "").strip() or None
    resolved_expert_model = expert_model or configured_model
    resolved_expert_base = expert_base_url or configured_base
    resolved_expert_key_env = expert_api_key_env or DEFAULT_EXPERT_API_KEY_ENV
    resolved_expert_key = expert_api_key or configured_key

    resolved_cluster_model = cluster_model
    if resolved_cluster_model is None:
        resolved_cluster_model = os.environ.get(
            DEFAULT_CLUSTER_MODEL_ENV, DEFAULT_CLUSTER_MODEL
        )
    resolved_cluster_base = cluster_base_url
    if resolved_cluster_base is None:
        resolved_cluster_base = os.environ.get(DEFAULT_CLUSTER_BASE_URL_ENV)
    if resolved_cluster_base is None:
        # 兼容集群服务提供的调用示例；正式配置仍建议使用 ARCHE_EXPERT_CLUSTER_*。
        resolved_cluster_base = os.environ.get("MODEL_BASE_URL")
    resolved_cluster_ak = cluster_ingress_ak
    if resolved_cluster_ak is None:
        resolved_cluster_ak = os.environ.get(DEFAULT_CLUSTER_INGRESS_AK_ENV)
    if resolved_cluster_ak is None:
        resolved_cluster_ak = os.environ.get("INGRESS_AK")
    resolved_cluster_sk = cluster_ingress_sk
    if resolved_cluster_sk is None:
        resolved_cluster_sk = os.environ.get(DEFAULT_CLUSTER_INGRESS_SK_ENV)
    if resolved_cluster_sk is None:
        resolved_cluster_sk = os.environ.get("INGRESS_SK")
    resolved_cluster_public_key = cluster_public_api_key
    if resolved_cluster_public_key is None:
        resolved_cluster_public_key = os.environ.get(
            DEFAULT_CLUSTER_PUBLIC_API_KEY_ENV
        )
    if resolved_cluster_public_key is None:
        resolved_cluster_public_key = os.environ.get("PUBLIC_API_KEY")

    resolved_fallback_model = fallback_model or configured_model or DEFAULT_EXPERT_FALLBACK_MODEL
    resolved_fallback_base = fallback_base_url or configured_base
    resolved_fallback_key_env = fallback_api_key_env or DEFAULT_FALLBACK_API_KEY_ENV
    resolved_fallback_key = fallback_api_key or general_api_key or configured_key

    resolved_general_model = resolve_general_model(general_model or configured_model)
    resolved_general_base = general_base_url or configured_base
    resolved_general_key_env = general_api_key_env or DEFAULT_GENERAL_API_KEY_ENV
    resolved_general_key = general_api_key or configured_key

    expert = _build_role(
        role="expert",
        model=resolved_expert_model,
        base_url=resolved_expert_base,
        api_key=resolved_expert_key,
        api_key_env=resolved_expert_key_env,
        timeout=timeout,
    )
    cluster_requested = any(
        isinstance(value, str) and bool(value.strip())
        for value in (
            cluster_model,
            cluster_base_url,
            cluster_ingress_ak,
            cluster_ingress_sk,
            cluster_public_api_key,
            os.environ.get(DEFAULT_CLUSTER_MODEL_ENV),
            os.environ.get(DEFAULT_CLUSTER_BASE_URL_ENV),
            os.environ.get("MODEL_BASE_URL"),
            os.environ.get(DEFAULT_CLUSTER_INGRESS_AK_ENV),
            os.environ.get(DEFAULT_CLUSTER_INGRESS_SK_ENV),
            os.environ.get(DEFAULT_CLUSTER_PUBLIC_API_KEY_ENV),
            os.environ.get("INGRESS_AK"),
            os.environ.get("INGRESS_SK"),
            os.environ.get("PUBLIC_API_KEY"),
        )
    )
    cluster = (
        _build_cluster_role(
            model=resolved_cluster_model,
            base_url=resolved_cluster_base,
            ingress_ak=resolved_cluster_ak,
            ingress_sk=resolved_cluster_sk,
            public_api_key=resolved_cluster_public_key,
            timeout=timeout,
        )
        if cluster_requested
        else None
    )
    general = _build_role(
        role="general",
        model=resolved_general_model,
        base_url=resolved_general_base,
        api_key=resolved_general_key,
        api_key_env=resolved_general_key_env,
        timeout=timeout,
    )
    fallback = _build_role(
        role="fallback",
        model=resolved_fallback_model,
        base_url=resolved_fallback_base,
        api_key=resolved_fallback_key,
        api_key_env=resolved_fallback_key_env,
        timeout=timeout,
    )
    return ExpertProviderBundle(
        enabled=enabled,
        expert=expert,
        general=general,
        fallback=fallback,
        cluster=cluster,
    )


# 兼容更短的调用名，方便 CLI 和测试读取而不暴露实现类细节。
build_expert_model_providers = build_expert_provider_bundle


__all__ = [
    "DEFAULT_CLUSTER_BASE_URL_ENV",
    "DEFAULT_CLUSTER_INGRESS_AK_ENV",
    "DEFAULT_CLUSTER_INGRESS_SK_ENV",
    "DEFAULT_CLUSTER_MODEL",
    "DEFAULT_CLUSTER_MODEL_ENV",
    "DEFAULT_CLUSTER_PUBLIC_API_KEY_ENV",
    "DEFAULT_EXPERT_API_KEY_ENV",
    "DEFAULT_FALLBACK_API_KEY_ENV",
    "DEFAULT_GENERAL_API_KEY_ENV",
    "DEFAULT_EXPERT_PROVIDER_OUTPUT_TOKENS",
    "DEFAULT_GENERAL_PROVIDER_OUTPUT_TOKENS",
    "DEFAULT_EXPERT_STARTUP_PROBE_TIMEOUT_SECONDS",
    "DEFAULT_EXPERT_STARTUP_PROBE_MAX_TOKENS",
    "DEFAULT_EXPERT_STARTUP_PROBE_RETRY_COUNT",
    "EXPERT_RUNTIME_7B",
    "EXPERT_RUNTIME_CLUSTER",
    "EXPERT_RUNTIME_FALLBACK",
    "EXPERT_RUNTIME_SCHEMA_VERSION",
    "EXPERT_RUNTIME_STARTUP_FAILED",
    "ExpertProviderBundle",
    "ExpertProviderRole",
    "ExpertRuntimeSelection",
    "OpenAICompatibleExpertProvider",
    "model_identity_matches",
    "build_expert_model_providers",
    "build_expert_provider_bundle",
    "select_expert_runtime",
]
