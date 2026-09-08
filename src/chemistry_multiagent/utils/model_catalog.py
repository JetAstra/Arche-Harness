"""集中定义各模型操作的调用参数和可审计请求。"""

from __future__ import annotations

import hashlib
import json
import math
import os
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from chemistry_multiagent.utils.model_defaults import (
    AGENT_A1_CHAT_COMPLETIONS_PROVIDER,
    ARCHE_CHEM_CHAT_COMPLETIONS_PROVIDER,
    CODEX_MODEL_NAME_ENV,
    CODEX_REASONING_EFFORT,
    CODEX_RESPONSES_PROVIDER,
    DEFAULT_CODEX_MODEL,
    DEEPSEEK_V4_PRO_REASONING_EFFORT_VALUES,
    HYPOTHESIS_FORMATION_PROVIDER_AGENT_A1,
    is_deepseek_v4_pro_model,
    resolve_agent_a1_provider_settings,
    resolve_codex_provider_settings,
    resolve_deepseek_v4_pro_reasoning_effort,
    resolve_general_model,
    resolve_hypothesis_formation_provider,
)


PROMPT_CATALOG_VERSION = "model_prompt_catalog.v33"
PROJECTOR_VERSION = "model_input_projection.v23"
SERIALIZER_VERSION = "canonical_model_request.v3"
# v13 固化既有 Codex 与 ARCHE_CHEM profile 身份；A1 使用下方独立版本。
MODEL_PROFILE_VERSION = "deepseek_v4_model_profiles.v13"
# A1 使用独立 profile 身份，避免把 Chat Completions 请求审计成 Codex。
AGENT_A1_MODEL_PROFILE_VERSION = "agent_a1_hypothesis_profiles.v1"
# Gaussian 专家 provider 有自己的输出策略；该值只保留给兼容的专家目录项，
# 通用模型目录项不会把它写入 compiled request。
MODEL_MAX_TOKENS = 16384


MODEL_OPERATION_KEYS = (
    ("workflow_generation", None),
    ("workflow_optimization", None),
    ("router", None),
    ("workflow_revision", None),
    ("gaussian_expert", "technical_failure"),
    ("gaussian_expert", "failed_check"),
    ("retrieval", "chemistry_context"),
    ("retrieval", "keyword"),
    ("retrieval", "document_qa"),
    ("retrieval", "literature_review"),
    ("hypothesis_generation", "query"),
    ("hypothesis_generation", "hypothesis"),
    ("hypothesis_generation", "relationship"),
    ("hypothesis_generation", "merge"),
    ("hypothesis_generation", "complement"),
    ("hypothesis_generation", "conflict"),
    ("hypothesis_generation", "rank"),
    ("hypothesis_generation", "regeneration"),
)

# 假设形成的六个阶段固定走独立 Sol Responses provider；关系比较和排名以及
# Retrieval 使用 ARCHE_CHEM Chat Completions。provider 是传输能力选择，模型名
# 由对应 provider 的环境配置决定，不在这里写死某一个供应商模型。
CODEX_MODEL_OPERATION_KEYS = frozenset(
    {
        ("hypothesis_generation", "query"),
        ("hypothesis_generation", "hypothesis"),
        ("hypothesis_generation", "merge"),
        ("hypothesis_generation", "complement"),
        ("hypothesis_generation", "conflict"),
        ("hypothesis_generation", "regeneration"),
    }
)

# Relationship/Rank 是假设比较阶段，单独列出以避免后续新增通用路由时意外
# 把比较模型切到 Sol；它不再限制具体模型名称。
DEEPSEEK_COMPARISON_OPERATION_KEYS = frozenset(
    {
        ("hypothesis_generation", "relationship"),
        ("hypothesis_generation", "rank"),
    }
)


@dataclass(frozen=True)
class ModelProfile:
    """保存一次模型调用真正会传给服务端的生成参数。

    通用模型的 ``max_tokens`` 为 ``None`` 时表示不人为设置输出上限；编译请求可以
    保存这个 ``null`` 事实，但发送适配器不会向 OpenAI-compatible 服务发送该字段。
    ``stream`` 记录并约束真实传输方式，避免审计中的请求形状与发送参数不一致。
    ``provider_output_mode`` 表示本地 JSON/schema 校验模式；实际发送的
    ``provider_response_format`` 单独记录，DeepSeek v4 Pro 会明确保存为 ``None``。
    Gaussian 专家 provider 不使用这里的通用发送路径，仍由自己的 provider 合同决定
    输出策略。
    """

    profile_id: str
    temperature: float
    top_p: float
    max_tokens: int | None
    provider_output_mode: str
    stop: None = None
    stream: bool = False
    thinking: None = None
    reasoning_effort: str | None = None
    provider_response_format: Mapping[str, str] | None = None
    provider: str = ARCHE_CHEM_CHAT_COMPLETIONS_PROVIDER

    def __post_init__(self) -> None:
        """校验可选输出预算，避免目录把无效值传播到请求和审计。"""

        if self.max_tokens is not None and (
            isinstance(self.max_tokens, bool)
            or not isinstance(self.max_tokens, int)
            or self.max_tokens <= 0
        ):
            raise ValueError("模型 profile 的 max_tokens 必须是正整数或 None")
        if self.reasoning_effort not in {
            None,
            *DEEPSEEK_V4_PRO_REASONING_EFFORT_VALUES,
        }:
            raise ValueError("模型 profile 的 reasoning_effort 未注册")
        if self.provider_response_format not in (None, {"type": "json_object"}):
            raise ValueError("模型 profile 的 provider_response_format 未注册")
        if self.provider not in {
            ARCHE_CHEM_CHAT_COMPLETIONS_PROVIDER,
            CODEX_RESPONSES_PROVIDER,
            AGENT_A1_CHAT_COMPLETIONS_PROVIDER,
        }:
            raise ValueError("模型 profile 的 provider 未注册")


def _profile(
    name: str,
    *,
    temperature: float,
    max_tokens: int | None,
    output_mode: str,
    stream: bool = False,
) -> ModelProfile:
    return ModelProfile(
        profile_id=f"model_operation.{name}.v4",
        temperature=temperature,
        top_p=1.0,
        max_tokens=max_tokens,
        provider_output_mode=output_mode,
        stream=stream,
        provider_response_format=(
            {"type": "json_object"} if output_mode == "json_object" else None
        ),
        provider=ARCHE_CHEM_CHAT_COMPLETIONS_PROVIDER,
    )


_PROFILES = {
    ("workflow_generation", None): _profile(
        "workflow_generation",
        temperature=0.1,
        max_tokens=None,
        output_mode="json_object",
        # 工作流生成可能先进行数分钟隐藏推理再输出 JSON；流式传输可在
        # provider 继续生成时保持网关连接，由传输适配器重组为完整响应。
        stream=True,
    ),
    ("workflow_optimization", None): _profile(
        "workflow_optimization",
        temperature=0.1,
        max_tokens=None,
        output_mode="json_object",
    ),
    ("router", None): _profile(
        "router",
        temperature=0.0,
        # 推理模型的正文长度交给服务端决定，不在通用请求中人为设置上限。
        max_tokens=None,
        output_mode="json_object",
    ),
    ("workflow_revision", None): _profile(
        "workflow_revision",
        temperature=0.1,
        max_tokens=None,
        output_mode="json_object",
    ),
    ("gaussian_expert", "technical_failure"): _profile(
        "gaussian_expert.technical_failure",
        temperature=0.2,
        max_tokens=MODEL_MAX_TOKENS,
        output_mode="text",
    ),
    ("gaussian_expert", "failed_check"): _profile(
        "gaussian_expert.failed_check",
        temperature=0.2,
        max_tokens=MODEL_MAX_TOKENS,
        output_mode="text",
    ),
    ("retrieval", "chemistry_context"): _profile(
        "retrieval.chemistry_context",
        temperature=0.0,
        max_tokens=None,
        output_mode="json_object",
    ),
    ("retrieval", "keyword"): _profile(
        "retrieval.keyword",
        temperature=0.2,
        max_tokens=None,
        output_mode="text",
    ),
    ("retrieval", "document_qa"): _profile(
        "retrieval.document_qa",
        temperature=0.2,
        max_tokens=None,
        output_mode="text",
    ),
    ("retrieval", "literature_review"): _profile(
        "retrieval.literature_review",
        temperature=0.2,
        max_tokens=None,
        output_mode="text",
    ),
    ("hypothesis_generation", "query"): _profile(
        "hypothesis_generation.query",
        temperature=0.7,
        max_tokens=None,
        output_mode="text",
    ),
    ("hypothesis_generation", "hypothesis"): _profile(
        "hypothesis_generation.hypothesis",
        temperature=0.7,
        max_tokens=None,
        output_mode="text",
    ),
    ("hypothesis_generation", "relationship"): _profile(
        "hypothesis_generation.relationship",
        temperature=0.0,
        # 即使正文只有一个枚举值，也不截断服务端尚未完成的推理过程。
        max_tokens=None,
        output_mode="json_object",
    ),
    ("hypothesis_generation", "merge"): _profile(
        "hypothesis_generation.merge",
        temperature=0.0,
        max_tokens=None,
        output_mode="json_object",
    ),
    ("hypothesis_generation", "complement"): _profile(
        "hypothesis_generation.complement",
        temperature=0.0,
        max_tokens=None,
        output_mode="json_object",
    ),
    ("hypothesis_generation", "conflict"): _profile(
        "hypothesis_generation.conflict",
        temperature=0.0,
        max_tokens=None,
        output_mode="json_object",
    ),
    ("hypothesis_generation", "rank"): _profile(
        "hypothesis_generation.rank",
        temperature=0.0,
        # 排名正文很短，但推理过程仍不应被本地固定上限截断。
        max_tokens=None,
        output_mode="json_object",
    ),
    ("hypothesis_generation", "regeneration"): _profile(
        "hypothesis_generation.regeneration",
        temperature=0.7,
        max_tokens=None,
        output_mode="text",
    ),
}


if tuple(_PROFILES) != MODEL_OPERATION_KEYS:
    raise RuntimeError("模型目录与提示目录的操作顺序不一致")


def resolve_provider_for_operation_stage(
    operation: str,
    stage: str | None,
    *,
    require_provider_config: bool = True,
) -> str:
    """返回操作阶段实际使用的公开 provider 身份。

    输入是模型目录中的操作和阶段；动作只对六个假设形成阶段读取形成 provider
    开关，关系/排名及其它阶段始终返回 ARCHE_CHEM Chat Completions。返回值只含
    provider 类型，不含地址或认证正文。正式请求默认校验独立 provider 配置；profile
    兼容读取可关闭该项，但 A1 选择仍会由模型解析阶段严格校验配置。
    """

    key = (operation, stage)
    if key in CODEX_MODEL_OPERATION_KEYS:
        selected = resolve_hypothesis_formation_provider()
        if selected == HYPOTHESIS_FORMATION_PROVIDER_AGENT_A1:
            if require_provider_config:
                resolve_agent_a1_provider_settings()
            return AGENT_A1_CHAT_COMPLETIONS_PROVIDER
        if require_provider_config:
            resolve_codex_provider_settings()
        return CODEX_RESPONSES_PROVIDER
    return ARCHE_CHEM_CHAT_COMPLETIONS_PROVIDER


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TypeError("模型请求必须能稳定编码为有限 JSON") from exc


def get_model_profile(
    operation: str,
    stage: str | None,
    *,
    model: str | None = None,
) -> dict[str, Any]:
    """返回按实际模型解析的独立 profile 副本，避免调用方改写全局目录。

    输入是操作、阶段和可选模型标识；动作读取固定目录并按 provider 能力写入
    reasoning_effort 与远端 JSON mode。返回值不共享目录对象；只有明确属于
    DeepSeek v4 Pro 的模型才读取其专用推理强度配置，其他模型不附带该参数。
    """

    key = (operation, stage)
    try:
        profile = _PROFILES[key]
    except KeyError as exc:
        raise ValueError(f"未注册的模型操作：{key!r}") from exc
    values = deepcopy(asdict(profile))
    effective_model = resolve_model_for_operation_stage(
        operation,
        stage,
        model=model,
        require_codex_config=False,
    )
    provider = resolve_provider_for_operation_stage(
        operation,
        stage,
        require_provider_config=False,
    )
    values["provider"] = provider
    if provider == CODEX_RESPONSES_PROVIDER:
        values["reasoning_effort"] = CODEX_REASONING_EFFORT
        # Sol Responses 长调用必须通过流式接口消费；该事实写入 compiled request。
        values["stream"] = True
        # Responses 不接受 Chat Completions 的 response_format；schema 只在本地校验。
        values["provider_response_format"] = None
    elif provider == AGENT_A1_CHAT_COMPLETIONS_PROVIDER:
        # A1 是独立 Chat Completions，不使用 Codex 推理档位或流式 Responses。
        values["profile_id"] = f"{values['profile_id']}.agent_a1"
        values["reasoning_effort"] = None
        values["stream"] = False
        values["provider_response_format"] = None
    elif is_deepseek_v4_pro_model(effective_model):
        values["reasoning_effort"] = resolve_deepseek_v4_pro_reasoning_effort()
        values["provider_response_format"] = None
    else:
        values["reasoning_effort"] = None
    return values


def resolve_model_for_operation_stage(
    operation: str,
    stage: str | None,
    *,
    model: str | None = None,
    require_codex_config: bool = True,
) -> str:
    """按阶段解析最终模型，Codex 形成阶段不受通用模型覆盖影响。

    输入是操作、阶段和兼容旧调用方的显式模型；动作先查询阶段路由，命中六个
    假设形成阶段时读取当前选择的独立 Codex 或 Agent A1 配置，否则沿用
    ARCHE_CHEM 通用模型解析。正式编译请求会校验对应 provider 的地址、模型和
    密钥均存在；配置缺失时在网络请求前失败关闭，不会借用 ARCHE_CHEM 的值。
    """

    key = (operation, stage)
    if key in CODEX_MODEL_OPERATION_KEYS:
        selected_provider = resolve_hypothesis_formation_provider()
        if selected_provider == HYPOTHESIS_FORMATION_PROVIDER_AGENT_A1:
            configured_model, _base_url, _api_key = (
                resolve_agent_a1_provider_settings()
            )
            return configured_model
        if not require_codex_config:
            return (
                os.environ.get(CODEX_MODEL_NAME_ENV, "").strip()
                or DEFAULT_CODEX_MODEL
            )
        configured_model, _base_url, _api_key = resolve_codex_provider_settings()
        return configured_model
    return resolve_general_model(model)


def _validate_messages(messages: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
        raise TypeError("messages 必须是非空消息数组")
    normalized = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping) or set(message) != {"role", "content"}:
            raise TypeError(f"messages[{index}] 字段不符合固定协议")
        role = message["role"]
        content = message["content"]
        if not isinstance(role, str) or not role:
            raise TypeError(f"messages[{index}].role 必须是非空字符串")
        if not isinstance(content, str) or not content:
            raise TypeError(f"messages[{index}].content 必须是非空字符串")
        normalized.append({"role": role, "content": content})
    if not normalized:
        raise TypeError("messages 必须是非空消息数组")
    return normalized


def _validate_response_contract(response_contract: Mapping[str, Any]) -> dict[str, Any]:
    expected = {"validation_kind", "schema", "domain_validator_id"}
    if not isinstance(response_contract, Mapping) or set(response_contract) != expected:
        raise TypeError("response_contract 字段不符合固定协议")
    validation_kind = response_contract["validation_kind"]
    schema = response_contract["schema"]
    validator_id = response_contract["domain_validator_id"]
    if not isinstance(validation_kind, str) or not validation_kind:
        raise TypeError("validation_kind 必须是非空字符串")
    if schema is not None and not isinstance(schema, Mapping):
        raise TypeError("schema 必须是 object 或 null")
    if validator_id is not None and (
        not isinstance(validator_id, str) or not validator_id.strip()
    ):
        raise TypeError("domain_validator_id 必须是非空字符串或 null")
    normalized = {
        "validation_kind": validation_kind,
        "schema": deepcopy(dict(schema)) if schema is not None else None,
        "domain_validator_id": validator_id,
    }
    _canonical_json_bytes(normalized)
    return normalized


def compile_model_request(
    *,
    operation: str,
    stage: str | None,
    messages: Sequence[Mapping[str, str]],
    response_contract: Mapping[str, Any],
    model: str | None = None,
) -> dict[str, Any]:
    """把提示、生成参数和响应校验身份编译成一份稳定请求。"""

    configured_model = resolve_model_for_operation_stage(operation, stage, model=model)
    profile = get_model_profile(operation, stage, model=configured_model)
    normalized_messages = _validate_messages(messages)
    normalized_contract = _validate_response_contract(response_contract)
    # 历史调用仍可能传入通用别名。清单必须保存服务端实际模型，而不是等发送时
    # 才改名，否则模型审计和真实请求会出现逐值漂移。
    if not math.isfinite(profile["temperature"]) or not math.isfinite(profile["top_p"]):
        raise ValueError("模型采样参数必须是有限数值")
    schema_hash = hashlib.sha256(
        _canonical_json_bytes(normalized_contract)
    ).hexdigest()
    catalog_key = operation if stage is None else f"{operation}.{stage}"
    request = {
        "catalog_key": catalog_key,
        "prompt_catalog_version": PROMPT_CATALOG_VERSION,
        "projector_version": PROJECTOR_VERSION,
        "serializer_version": SERIALIZER_VERSION,
        "model_profile_version": (
            AGENT_A1_MODEL_PROFILE_VERSION
            if profile["provider"] == AGENT_A1_CHAT_COMPLETIONS_PROVIDER
            else MODEL_PROFILE_VERSION
        ),
        "profile_id": profile["profile_id"],
        "model": configured_model.strip(),
        "thinking": profile["thinking"],
        "reasoning_effort": profile["reasoning_effort"],
        "messages": normalized_messages,
        "temperature": profile["temperature"],
        "top_p": profile["top_p"],
        # 用 null 在可审计请求中明确记录“未设置预算”；实际发送前由 llm_api
        # 删除该字段，避免 OpenAI-compatible 服务收到 max_tokens=null。
        "max_tokens": profile["max_tokens"],
        "stop": profile["stop"],
        "stream": profile["stream"],
        "provider_output_mode": profile["provider_output_mode"],
        "provider_response_format": profile["provider_response_format"],
        # 只保存传输类型，禁止把 URL 或认证正文混入请求审计。
        "provider": profile["provider"],
        "response_schema_hash": schema_hash,
    }
    _canonical_json_bytes(request)
    return request


__all__ = [
    "AGENT_A1_MODEL_PROFILE_VERSION",
    "MODEL_PROFILE_VERSION",
    "MODEL_MAX_TOKENS",
    "MODEL_OPERATION_KEYS",
    "CODEX_MODEL_OPERATION_KEYS",
    "DEEPSEEK_COMPARISON_OPERATION_KEYS",
    "resolve_provider_for_operation_stage",
    "PROJECTOR_VERSION",
    "PROMPT_CATALOG_VERSION",
    "SERIALIZER_VERSION",
    "compile_model_request",
    "get_model_profile",
    "resolve_model_for_operation_stage",
]
