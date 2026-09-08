import os
import json
import threading
import openai
from typing import List, Dict, Any, Mapping

from chemistry_multiagent.utils.model_operation import normalize_provider_envelope
from chemistry_multiagent.utils.model_defaults import (
    AGENT_A1_CHAT_COMPLETIONS_PROVIDER,
    ARCHE_CHEM_CHAT_COMPLETIONS_PROVIDER,
    CODEX_REASONING_EFFORT,
    CODEX_RESPONSES_PROVIDER,
    DEEPSEEK_V4_PRO_REASONING_EFFORT_VALUES,
    DEFAULT_GENERAL_MODEL,
    is_codex_model,
    is_deepseek_v4_pro_model,
    resolve_agent_a1_provider_settings,
    resolve_codex_provider_settings,
    resolve_deepseek_v4_pro_reasoning_effort,
    resolve_general_model,
)
from chemistry_multiagent.utils.api_retry import (
    MODEL_CALL_RETRY_COUNT,
    MODEL_CALL_RETRY_DELAY_SECONDS,
    is_retryable_http_status,
    is_retryable_api_exception,
    response_http_status,
    retry_api_call,
)

try:
    from chemistry_multiagent.utils.llm_trace import trace_llm_call
except ImportError:
    try:
        from .llm_trace import trace_llm_call
    except ImportError:
        trace_llm_call = None

# 统一模型 API。所有 ARCHE 角色共用这组三元组配置。
ARCHE_CHEM_API_KEY = os.environ.get("ARCHE_CHEM_API_KEY") or ""
ARCHE_CHEM_BASE_URL = os.environ.get("ARCHE_CHEM_BASE_URL") or "https://api.deepseek.com/v1"
# Python 导入兼容名；它们只是别名，不再读取 DEEPSEEK_* 环境变量。
DEEPSEEK_API_KEY = ARCHE_CHEM_API_KEY
DEEPSEEK_BASE_URL = ARCHE_CHEM_BASE_URL
# 保留旧的公开名称供 Controller 和外部调用方导入；真正解析环境覆盖由
# ``resolve_general_model`` 在每次请求编译时完成。
DEFAULT_MODEL = DEFAULT_GENERAL_MODEL
# 通用模型不设置本地输出上限；保留公开常量和显式参数只是为了兼容旧调用方。
DEFAULT_MAX_TOKENS: int | None = None

_COMPILED_REQUEST_KEYS = {
    "catalog_key",
    "prompt_catalog_version",
    "projector_version",
    "serializer_version",
    "model_profile_version",
    "profile_id",
    "model",
    "thinking",
    "reasoning_effort",
    "messages",
    "temperature",
    "top_p",
    "max_tokens",
    "stop",
    "stream",
    "provider_output_mode",
    "provider_response_format",
    "provider",
    "response_schema_hash",
}
_COMPILED_REQUEST_KEYS_WITHOUT_OPTIONAL_BUDGET = (
    _COMPILED_REQUEST_KEYS - {"max_tokens"}
)
_LEGACY_COMPILED_REQUEST_KEYS = _COMPILED_REQUEST_KEYS - {"provider"}
_LEGACY_COMPILED_REQUEST_KEYS_WITHOUT_OPTIONAL_BUDGET = (
    _LEGACY_COMPILED_REQUEST_KEYS - {"max_tokens"}
)

# 初始化兼容 OpenAI 的客户端。离线测试也需要能够导入本模块，但不会发起真实请求。
client = openai.OpenAI(
    # 新版客户端不接受空 key。占位文本仅用于让纯规则和离线测试顺利导入；真实调用
    # 仍由服务端校验本地配置中的有效凭据，代码不会把占位值当作可用密钥。
    api_key=DEEPSEEK_API_KEY or "EMPTY",
    base_url=DEEPSEEK_BASE_URL,
    # 外层 ModelOperation 会逐次记录传输失败；关闭 SDK 隐藏重试后，每次记录
    # 才能与一次真实网络请求准确对应，也不会在服务变慢时成倍拉长等待时间。
    max_retries=0,
)

_CODEX_CLIENT_LOCK = threading.RLock()
_CODEX_CLIENT: Any | None = None
_CODEX_CLIENT_SIGNATURE: tuple[str, str, str] | None = None

_AGENT_A1_CLIENT_LOCK = threading.RLock()
_AGENT_A1_CLIENT: Any | None = None
_AGENT_A1_CLIENT_SIGNATURE: tuple[str, str, str] | None = None


def _build_codex_responses_client(api_key: str, base_url: str):
    """按独立 Sol 配置创建 Responses 客户端，不接入 ARCHE_CHEM 槽位。"""

    return openai.OpenAI(
        api_key=api_key or "EMPTY",
        base_url=base_url,
        # 外层 ModelOperation 负责正式传输重试；SDK 不得隐藏重复请求。
        max_retries=0,
    )


def _get_codex_responses_client(requested_model: str):
    """校验 Sol 三元组并返回按地址和模型缓存的 Responses client。

    输入是 compiled request 的模型名；动作读取独立 Codex 配置并按配置创建
    客户端。返回值只供当前进程的 Sol 请求使用；配置缺失或模型身份不一致时在
    网络请求前失败关闭，且不会触碰 ARCHE_CHEM client。
    """

    configured_model, base_url, api_key = resolve_codex_provider_settings()
    if configured_model != requested_model:
        raise ValueError(
            "Codex compiled request 的模型与 ARCHE_CODEX_MODEL_NAME 不一致"
        )
    global _CODEX_CLIENT, _CODEX_CLIENT_SIGNATURE
    # 把 key 只作为进程内缓存身份的一部分，避免环境切换后复用旧认证；它不进入
    # 请求 receipt、trace 或任何异常文本。
    signature = (base_url, configured_model, api_key)
    with _CODEX_CLIENT_LOCK:
        if _CODEX_CLIENT is None or _CODEX_CLIENT_SIGNATURE != signature:
            _CODEX_CLIENT = _build_codex_responses_client(api_key, base_url)
            _CODEX_CLIENT_SIGNATURE = signature
        return _CODEX_CLIENT


def _build_agent_a1_chat_client(
    *,
    api_key: str,
    base_url: str,
):
    """按独立 Bearer 配置创建 Agent A1 Chat Completions 客户端。

    输入是已通过配置校验的地址和 API key；动作交给 OpenAI-compatible SDK 生成
    Bearer Authorization 头。返回值只供 A1 传输使用，认证不会进入 compiled
    request、trace 或模型输出审计。
    """

    return openai.OpenAI(
        api_key=api_key,
        base_url=base_url,
        max_retries=0,
    )


def _get_agent_a1_chat_client(requested_model: str):
    """校验 A1 三项配置并返回缓存的 Chat Completions client。

    输入是 compiled request 中的模型名；动作读取独立 A1 配置，按完整配置身份
    缓存客户端。返回值只包含当前进程的网络客户端；缺项或模型身份漂移时在发起
    网络请求前失败关闭，且不会回退到 ARCHE_CHEM 或 Codex。
    """

    configured_model, base_url, api_key = (
        resolve_agent_a1_provider_settings()
    )
    if configured_model != requested_model:
        raise ValueError(
            "Agent A1 compiled request 的模型与 ARCHE_AGENT_A1_MODEL_NAME 不一致"
        )
    global _AGENT_A1_CLIENT, _AGENT_A1_CLIENT_SIGNATURE
    # 认证正文只作为进程内缓存分片，绝不复制到运行审计或错误信息。
    signature = (base_url, configured_model, api_key)
    with _AGENT_A1_CLIENT_LOCK:
        if (
            _AGENT_A1_CLIENT is None
            or _AGENT_A1_CLIENT_SIGNATURE != signature
        ):
            _AGENT_A1_CLIENT = _build_agent_a1_chat_client(
                api_key=api_key,
                base_url=base_url,
            )
            _AGENT_A1_CLIENT_SIGNATURE = signature
        return _AGENT_A1_CLIENT


class _AgentA1ChatFailure(RuntimeError):
    """保存 A1 请求的可重试事实，不携带原始异常正文。"""

    __slots__ = ("_retryable",)

    def __init__(self, *, retryable: bool) -> None:
        super().__init__("Agent A1 Chat Completions request failed")
        self._retryable = bool(retryable)

    @property
    def retryable(self) -> bool:
        """返回原始传输失败是否允许再次请求。"""

        return self._retryable


def _is_retryable_agent_a1_exception(error: BaseException) -> bool:
    """只按脱敏 marker 或瞬时传输类型判断 A1 是否可重试。"""

    if isinstance(error, _AgentA1ChatFailure):
        return error.retryable
    return is_retryable_api_exception(error)


def _extract_codex_response_text(response: Any) -> str:
    """从 Responses response 的 output message 中提取完整正文。"""

    normalized = normalize_provider_envelope(response)
    if not isinstance(normalized, Mapping):
        return ""
    output_text = normalized.get("output_text")
    if isinstance(output_text, str):
        return output_text
    output = normalized.get("output")
    fragments: list[str] = []
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, Mapping):
                continue
            content = item.get("content")
            if isinstance(content, str):
                fragments.append(content)
                continue
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, Mapping) and isinstance(part.get("text"), str):
                    fragments.append(part["text"])
    return "".join(fragments)


def _normalize_codex_responses_envelope(response: Any, model: str) -> dict[str, Any]:
    """把 Responses response 规整为现有 ModelOperation 的 choices envelope。

    只复制普通响应身份、状态、usage 和 assistant 正文；provider 地址、请求头、
    凭据和原始异常正文不会进入 envelope 或 trace。
    """

    normalized = normalize_provider_envelope(response)
    if not isinstance(normalized, Mapping):
        normalized = {}
    status = normalized.get("status")
    finish_reason = "length" if status in {"incomplete", "failed"} else "stop"
    incomplete_details = normalized.get("incomplete_details")
    if isinstance(incomplete_details, Mapping) and incomplete_details.get("reason"):
        finish_reason = "length"
    identity = normalized.get("model")
    envelope: dict[str, Any] = {
        "id": normalized.get("id"),
        "object": normalized.get("object", "response"),
        "model": identity if isinstance(identity, str) and identity else model,
        "status": status,
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": {
                    "role": "assistant",
                    "content": _extract_codex_response_text(response),
                },
            }
        ],
    }
    for field in ("created", "created_at", "usage", "incomplete_details"):
        if field in normalized:
            envelope[field] = normalized[field]
    return envelope


def _responses_input_from_messages(
    messages: list[dict[str, str]],
) -> tuple[str | None, str | list[dict[str, str]]]:
    """把既有 chat 消息拆成 Responses 的 instructions 和 input。"""

    instruction_parts: list[str] = []
    input_messages: list[dict[str, str]] = []
    for message in messages:
        if message["role"] in {"system", "developer"}:
            instruction_parts.append(message["content"])
        else:
            input_messages.append(dict(message))
    instructions = "\n\n".join(instruction_parts) or None
    if len(input_messages) == 1 and input_messages[0]["role"] == "user":
        return instructions, input_messages[0]["content"]
    if input_messages:
        return instructions, input_messages
    return instructions, ""


class _CodexResponsesStreamFailure(RuntimeError):
    """保存脱敏的 Responses 流失败事实，避免依赖原始异常可扩展。"""

    __slots__ = ("_received_event", "_retryable")

    def __init__(self, *, received_event: bool, retryable: bool) -> None:
        # 不保存原始异常、URL、响应正文或认证信息；上层只需要两个重试事实。
        super().__init__("Codex Responses stream failed")
        self._received_event = bool(received_event)
        self._retryable = bool(retryable)

    @property
    def received_event(self) -> bool:
        """返回流失败前是否已经观察到服务端事件。"""

        return self._received_event

    @property
    def retryable(self) -> bool:
        """返回失败是否可在未收到事件前重新请求。"""

        return self._retryable


def _responses_stream_has_received_event(
    stream: Any,
    *,
    observed_event: bool = False,
) -> bool:
    """判断 Responses 流是否已收到服务端事件，抑制上层重复提交。

    正常路径由公开的事件迭代设置 ``observed_event``；公开的兼容标记只用于不支持
    迭代的测试替身或旧适配器。SDK 私有状态仅作为最后的兼容回退，不作为首选事实源。
    """

    if observed_event:
        return True
    if stream is None:
        return False
    for name in ("received_event", "received_body", "_received_body"):
        value = getattr(stream, name, None)
        if isinstance(value, bool):
            if value:
                return True
    events = getattr(stream, "events", None)
    if isinstance(events, (list, tuple)) and events:
        return True
    state = getattr(stream, "_state", None)
    if state is None:
        return False
    # SDK 以私有快照保存首个 response.created；只读取布尔事实，不保存 SDK 对象。
    return any(
        getattr(state, name, None) is not None
        for name in ("_ResponseStreamState__current_snapshot", "_current_snapshot")
    )


def _call_codex_responses_request(
    request: Mapping[str, Any],
    *,
    timeout: int,
) -> dict[str, Any]:
    """通过 Responses 流消费完整 Sol 响应，不在适配器内重复提交。

    输入是已验证的 compiled request；动作把 system/developer 消息作为 instructions，
    以 Responses 流读取最终响应并规整为统一 envelope。若流已收到事件后失败，抛出
    只保留重试事实的脱敏 wrapper 供上层停止重试；原始正文不会进入异常或 trace。
    """

    model = str(request["model"])
    if request["stream"] is not True:
        raise ValueError("Codex Responses compiled request 必须使用 stream=true")
    client_for_request = _get_codex_responses_client(model)
    instructions, input_value = _responses_input_from_messages(request["messages"])
    kwargs: dict[str, Any] = {
        "model": model,
        "input": input_value,
        "reasoning": {"effort": CODEX_REASONING_EFFORT},
        "store": False,
        "timeout": timeout,
    }
    if instructions is not None:
        kwargs["instructions"] = instructions
    active_stream: Any | None = None
    observed_event = False
    stream_failure: _CodexResponsesStreamFailure | None = None
    try:
        with client_for_request.responses.stream(**kwargs) as active_stream:
            # OpenAI ResponseStream 的公开迭代器会逐个交付服务端事件。先消费迭代器
            # 再取最终响应，才能在 get_final_response 失败时保留真实的收流事实。
            try:
                stream_iterator = iter(active_stream)
            except TypeError:
                # 旧兼容替身只提供 get_final_response；此时使用下方公开标记或
                # SDK 私有状态回退，不凭异常对象猜测是否已收流。
                stream_iterator = None
            if stream_iterator is not None:
                for _event in stream_iterator:
                    observed_event = True
            response = active_stream.get_final_response()
    except Exception as exc:
        received_event = _responses_stream_has_received_event(
            active_stream,
            observed_event=observed_event,
        )
        stream_failure = _CodexResponsesStreamFailure(
            received_event=received_event,
            retryable=(
                not received_event and is_retryable_api_exception(exc)
            ),
        )
        # 清掉异常上下文；在 except 块外抛出 wrapper，避免 Python 自动重新挂回
        # provider 原始异常对象。
        stream_failure.__context__ = None
    if stream_failure is not None:
        raise stream_failure
    return _normalize_codex_responses_envelope(response, model)


def _is_retryable_codex_responses_exception(error: BaseException) -> bool:
    """只允许未收事件的瞬时 Sol 流失败进入统一重试器。"""

    if isinstance(error, _CodexResponsesStreamFailure):
        return error.retryable
    # 兼容旧适配器传出的内部标记；新路径不再修改原始异常。
    if getattr(error, "_stream_response_body_received", False):
        return False
    return is_retryable_api_exception(error)


def _call_agent_a1_chat_request(
    request: Mapping[str, Any],
    *,
    timeout: int,
) -> Any:
    """通过独立 Agent A1 的 Chat Completions 发送一次非流式请求。

    输入是已验证的 A1 compiled request；动作沿用目录中的 messages、temperature
    和 top_p，不设置本地 max_tokens，也不发送 reasoning 或 response_format。返回
    值是统一 provider envelope；网络异常只以脱敏重试 marker 继续向上层传播。
    """

    model = str(request["model"])
    if request["stream"] is not False:
        raise ValueError("Agent A1 compiled request 必须使用 stream=false")
    client_for_request = _get_agent_a1_chat_client(model)
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": request["messages"],
        "temperature": request["temperature"],
        "top_p": request["top_p"],
        "stop": None,
        "stream": False,
        "timeout": timeout,
    }
    try:
        response = client_for_request.chat.completions.create(**kwargs)
    except Exception as exc:
        failure = _AgentA1ChatFailure(
            retryable=is_retryable_api_exception(exc),
        )
        # 先离开 except 块再抛出 wrapper，避免 Python 自动恢复原始异常上下文。
        failure.__context__ = None
    else:
        return normalize_provider_envelope(response)
    raise failure


def normalize_configured_model(model: str | None = None) -> str:
    """把显式模型或环境配置解析为通用 API 的实际模型名称。"""

    return resolve_general_model(model)


def _validate_optional_max_tokens(value: Any) -> int | None:
    """校验可选输出预算；``None`` 表示发送时完全省略该字段。"""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("max_tokens 必须是正整数或 None")
    return value


def _resolve_model_api_retry_count(
    retry_count: int,
    max_retries: int | None,
) -> int:
    """解析新重试参数，并保留旧 ``max_retries`` 的总尝试次数语义。"""

    if max_retries is not None:
        if (
            isinstance(max_retries, bool)
            or not isinstance(max_retries, int)
            or max_retries <= 0
        ):
            raise ValueError("max_retries 必须是正整数，表示总尝试次数")
        return max_retries - 1
    if isinstance(retry_count, bool) or not isinstance(retry_count, int) or retry_count < 0:
        raise ValueError("retry_count 必须是非负整数")
    return retry_count


def _stream_text_fragments(value: Any) -> list[str]:
    """从流式 delta 的正文或文本片段中提取可拼接字符串。"""

    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        return []
    fragments: list[str] = []
    for item in value:
        if isinstance(item, str):
            fragments.append(item)
            continue
        if isinstance(item, Mapping) and isinstance(item.get("text"), str):
            fragments.append(item["text"])
    return fragments


def _assemble_streaming_response(stream: Any) -> dict[str, Any]:
    """把 OpenAI-compatible 流式块重组成 ModelOperation 的完整响应 envelope。

    输入是 provider 返回的可迭代 chunk；动作是按 choice/index 拼接正文和
    reasoning_content，并保留 provider 在任一 chunk 提供的身份、结束原因和 usage。
    返回值只含普通 JSON 值，失败会把 provider 的迭代异常原样抛出，使外层传输
    重试和审计能够把它记录为一次失败请求。
    """

    if isinstance(stream, Mapping) or (
        not hasattr(stream, "__iter__") and hasattr(stream, "choices")
    ):
        normalized_stream = normalize_provider_envelope(stream)
        if isinstance(normalized_stream, Mapping) and "choices" in normalized_stream:
            # 某些 OpenAI-compatible 测试替身在 stream=true 时仍返回完整响应对象；
            # 它已经满足既有 envelope 契约，不必再次按字符或属性迭代。
            return normalized_stream
    try:
        iterator = iter(stream)
    except TypeError as exc:
        raise TypeError("stream provider response 必须是可迭代对象") from exc

    envelope: dict[str, Any] = {}
    choice_states: dict[int, dict[str, Any]] = {}
    next_choice_index = 0

    for raw_chunk in iterator:
        chunk = normalize_provider_envelope(raw_chunk)
        if not isinstance(chunk, Mapping):
            continue

        for field in (
            "id",
            "object",
            "created",
            "model",
            "system_fingerprint",
            "service_tier",
        ):
            value = chunk.get(field)
            if value is not None and field not in envelope:
                envelope[field] = value
        usage = chunk.get("usage")
        if usage is not None:
            envelope["usage"] = usage

        choices = chunk.get("choices")
        if not isinstance(choices, list):
            continue
        for raw_choice in choices:
            if not isinstance(raw_choice, Mapping):
                continue
            raw_index = raw_choice.get("index", next_choice_index)
            index = (
                raw_index
                if isinstance(raw_index, int) and not isinstance(raw_index, bool)
                else next_choice_index
            )
            next_choice_index = max(next_choice_index, index + 1)
            state = choice_states.setdefault(
                index,
                {
                    "role": None,
                    "content": [],
                    "reasoning_content": [],
                    "finish_reason": None,
                    "logprobs": None,
                },
            )
            finish_reason = raw_choice.get("finish_reason")
            if finish_reason is not None:
                state["finish_reason"] = finish_reason
            if raw_choice.get("logprobs") is not None:
                state["logprobs"] = raw_choice["logprobs"]

            delta = raw_choice.get("delta")
            if not isinstance(delta, Mapping):
                delta = raw_choice.get("message")
            if not isinstance(delta, Mapping):
                continue
            role = delta.get("role")
            if isinstance(role, str) and role and state["role"] is None:
                state["role"] = role
            state["content"].extend(_stream_text_fragments(delta.get("content")))
            reasoning = delta.get("reasoning_content")
            if reasoning is None:
                reasoning = delta.get("reasoning")
            if reasoning is None:
                reasoning = delta.get("reasoningContent")
            state["reasoning_content"].extend(_stream_text_fragments(reasoning))

    assembled_choices: list[dict[str, Any]] = []
    for index in sorted(choice_states):
        state = choice_states[index]
        message = {
            "role": state["role"] or "assistant",
            "content": "".join(state["content"]),
            "reasoning_content": "".join(state["reasoning_content"]),
        }
        choice: dict[str, Any] = {
            "index": index,
            "finish_reason": state["finish_reason"],
            "message": message,
        }
        if state["logprobs"] is not None:
            choice["logprobs"] = state["logprobs"]
        assembled_choices.append(choice)
    envelope["choices"] = assembled_choices
    return envelope


def _validate_compiled_model_request_contract(
    request: Mapping[str, Any],
) -> tuple[int | None, list[Any], bool, str, Any, str, Any, str]:
    """Validate the provider-neutral compiled-request contract once."""

    if not isinstance(request, Mapping) or set(request) not in (
        _COMPILED_REQUEST_KEYS,
        _COMPILED_REQUEST_KEYS_WITHOUT_OPTIONAL_BUDGET,
        _LEGACY_COMPILED_REQUEST_KEYS,
        _LEGACY_COMPILED_REQUEST_KEYS_WITHOUT_OPTIONAL_BUDGET,
    ):
        raise TypeError("compiled model request 字段不符合固定协议")
    # 新目录会在审计中保存 max_tokens=null；兼容调用方也可以完全省略这个可选键。
    max_tokens = _validate_optional_max_tokens(request.get("max_tokens"))
    messages = request["messages"]
    if not isinstance(messages, list) or not messages:
        raise TypeError("compiled model request 缺少消息")
    stream = request["stream"]
    if not isinstance(stream, bool) or request["stop"] is not None:
        raise ValueError("compiled model request 的 stream 必须是布尔值且 stop 必须为空")
    requested_provider = request.get("provider")
    if requested_provider is not None and not isinstance(requested_provider, str):
        raise ValueError("compiled model request 的 provider 必须是字符串")
    if requested_provider is not None and requested_provider not in {
        ARCHE_CHEM_CHAT_COMPLETIONS_PROVIDER,
        CODEX_RESPONSES_PROVIDER,
        AGENT_A1_CHAT_COMPLETIONS_PROVIDER,
    }:
        raise ValueError("compiled model request 的 provider 未注册")
    if requested_provider == AGENT_A1_CHAT_COMPLETIONS_PROVIDER:
        raw_model = request["model"]
        if not isinstance(raw_model, str) or not raw_model.strip():
            raise ValueError("Agent A1 compiled request 的模型必须是非空字符串")
        effective_model = raw_model.strip()
        provider = AGENT_A1_CHAT_COMPLETIONS_PROVIDER
    else:
        effective_model = normalize_configured_model(request["model"])
        provider = requested_provider
    # 显式 provider 是请求的路由事实；模型名称可以由该 provider 自行注册，不能
    # 通过名称前缀把可用模型误判为另一后端。只有未携带 provider 的历史请求才
    # 使用 is_codex_model 推断路由。
    is_codex = is_codex_model(effective_model)
    is_pro_model = is_deepseek_v4_pro_model(effective_model)
    reasoning_effort = request["reasoning_effort"]
    if request["thinking"] is not None:
        raise ValueError("当前服务尚未验证 thinking 参数，不能静默发送")

    if provider is None:
        provider = CODEX_RESPONSES_PROVIDER if is_codex else ARCHE_CHEM_CHAT_COMPLETIONS_PROVIDER

    if requested_provider is not None:
        # 新 compiled request 的 provider 必须仍与当前阶段开关一致；这样环境在
        # 编译和发送之间变化时会停止，而不会把 A1 请求偷偷送到其它后端。
        from chemistry_multiagent.utils.model_catalog import (
            resolve_provider_for_operation_stage,
        )

        catalog_key = request.get("catalog_key")
        if not isinstance(catalog_key, str) or not catalog_key:
            raise ValueError("compiled model request 缺少可校验的 catalog key")
        if "." in catalog_key:
            operation, stage = catalog_key.split(".", 1)
        else:
            operation, stage = catalog_key, None
        expected_provider = resolve_provider_for_operation_stage(
            operation,
            stage,
            require_provider_config=True,
        )
        if expected_provider != provider:
            raise ValueError("compiled model request 的 provider 与当前阶段路由不一致")

    if provider == AGENT_A1_CHAT_COMPLETIONS_PROVIDER:
        # 只有目录注册的六个形成阶段可以使用 A1；关系/排名和 Retrieval 不能误路由。
        from chemistry_multiagent.utils.model_catalog import CODEX_MODEL_OPERATION_KEYS

        catalog_key = request.get("catalog_key")
        if not isinstance(catalog_key, str) or "." not in catalog_key:
            raise ValueError("Agent A1 provider 只能用于假设形成阶段")
        operation, stage = catalog_key.split(".", 1)
        if (operation, stage) not in CODEX_MODEL_OPERATION_KEYS:
            raise ValueError("Agent A1 provider 只能用于假设形成阶段")
        configured_model, _base_url, _api_key = (
            resolve_agent_a1_provider_settings()
        )
        if configured_model != effective_model:
            raise ValueError(
                "Agent A1 compiled request 的模型与 ARCHE_AGENT_A1_MODEL_NAME 不一致"
            )
        if stream or max_tokens is not None or reasoning_effort is not None:
            raise ValueError(
                "Agent A1 Chat Completions 必须使用 stream=false、无 max_tokens 和无 reasoning_effort"
            )
    elif provider == CODEX_RESPONSES_PROVIDER:
        if reasoning_effort != CODEX_REASONING_EFFORT:
            raise ValueError(
                "Codex compiled request 必须发送 "
                f"reasoning_effort={CODEX_REASONING_EFFORT}"
            )
    else:
        # 未携带 provider 的历史请求按模型身份推断；显式 ARCHE_CHEM provider
        # 以自身配置为准，允许同名模型在不同兼容端点注册。
        if reasoning_effort not in {None, *DEEPSEEK_V4_PRO_REASONING_EFFORT_VALUES}:
            raise ValueError("reasoning_effort 未注册")

    if provider != AGENT_A1_CHAT_COMPLETIONS_PROVIDER and is_pro_model:
        configured_reasoning_effort = resolve_deepseek_v4_pro_reasoning_effort()
        if reasoning_effort != configured_reasoning_effort:
            raise ValueError(
                "DeepSeek v4 Pro compiled request 必须发送 "
                f"reasoning_effort={configured_reasoning_effort}"
            )
    if (
        provider != AGENT_A1_CHAT_COMPLETIONS_PROVIDER
        and reasoning_effort is not None
        and not is_pro_model
        and not is_codex
    ):
        raise ValueError("只有 DeepSeek v4 Pro 或 Codex Sol 可以发送 reasoning_effort")
    # provider_output_mode 只描述本地 JSON/schema 校验；远端请求格式由独立字段
    # provider_response_format 决定，Pro 的该字段必须为 None。
    output_mode = request["provider_output_mode"]
    if output_mode not in {"json_object", "text"}:
        raise ValueError("provider_output_mode 未注册")
    provider_response_format = request["provider_response_format"]
    if provider_response_format not in (None, {"type": "json_object"}):
        raise ValueError("provider_response_format 未注册")
    if provider in {
        AGENT_A1_CHAT_COMPLETIONS_PROVIDER,
        CODEX_RESPONSES_PROVIDER,
    } and provider_response_format is not None:
        raise ValueError(f"{provider} 不支持 response_format 参数")
    if (
        provider != AGENT_A1_CHAT_COMPLETIONS_PROVIDER
        and is_pro_model
        and provider_response_format is not None
    ):
        raise ValueError("DeepSeek v4 Pro 不支持 response_format 参数")
    return (
        max_tokens,
        messages,
        stream,
        effective_model,
        reasoning_effort,
        output_mode,
        provider_response_format,
        provider,
    )


def _call_compiled_model_request_react(
    request: Mapping[str, Any],
    timeout: int = 900,
    *,
    retry_count: int | None = None,
    retry_delay: float = MODEL_CALL_RETRY_DELAY_SECONDS,
) -> Any:
    """按编译请求调用服务端，并只重试传输层瞬时失败。

    本地请求合同、JSON 校验和模型内容问题不会在这里重答；模型阶段的上层运行器
    会把它们记录为内容失败并生成新的完整请求。这里的重试只覆盖 OpenAI-compatible
    客户端抛出的连接、超时、429 或 5xx 异常。流式 compiled request 默认只尝试
    一次，避免部分响应后的长请求被重复提交；直接调用方可以显式传入 retry_count。
    """

    (
        max_tokens,
        messages,
        stream,
        effective_model,
        reasoning_effort,
        output_mode,
        provider_response_format,
        provider,
    ) = _validate_compiled_model_request_contract(request)
    kwargs = {
        "model": effective_model,
        "messages": messages,
        "temperature": request["temperature"],
        "top_p": request["top_p"],
        "stop": None,
        "stream": stream,
        "timeout": timeout,
    }
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort
    # None 只作为审计事实保留在 compiled request；OpenAI-compatible 请求不能携带
    # max_tokens=null，否则某些服务会把它误判为不支持的参数。
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if provider_response_format is not None:
        kwargs["response_format"] = dict(provider_response_format)
    trace_parameters = {
        "catalog_key": request["catalog_key"],
        "profile_id": request["profile_id"],
        "model_profile_version": request["model_profile_version"],
        "temperature": request["temperature"],
        "top_p": request["top_p"],
        "max_tokens": max_tokens,
        "reasoning_effort": reasoning_effort,
        "provider_output_mode": output_mode,
        "provider_response_format": (
            dict(provider_response_format)
            if provider_response_format is not None
            else None
        ),
        "timeout": timeout,
        "response_schema_hash": request["response_schema_hash"],
        "stream": stream,
    }
    last_attempt_index = 0

    # 流式响应一旦已经收到部分正文，重新发送同一份长请求会制造重复生成，
    # 也无法修复已发生的网关空闲超时。默认流式调用只尝试一次；需要重试时
    # 由直接调用方显式传入 retry_count，ModelOperation 的嵌套重试仍会被抑制。
    if retry_count is None:
        effective_retry_count = 0 if stream else MODEL_CALL_RETRY_COUNT
    else:
        if (
            isinstance(retry_count, bool)
            or not isinstance(retry_count, int)
            or retry_count < 0
        ):
            raise ValueError("retry_count 必须是非负整数")
        effective_retry_count = retry_count

    if provider == AGENT_A1_CHAT_COMPLETIONS_PROVIDER:
        a1_attempt_index = 0

        def request_agent_a1_response() -> Any:
            """发送一次 A1 Chat Completions 请求。"""

            return _call_agent_a1_chat_request(request, timeout=timeout)

        def record_agent_a1_attempt(
            attempt_index: int,
            _response: Any,
            _error: BaseException | None,
        ) -> None:
            """只保存 A1 本地尝试序号，不复制地址、认证或异常正文。"""

            nonlocal a1_attempt_index
            a1_attempt_index = attempt_index

        a1_trace_parameters = {
            **trace_parameters,
            "provider": AGENT_A1_CHAT_COMPLETIONS_PROVIDER,
        }
        try:
            envelope = retry_api_call(
                request_agent_a1_response,
                retry_count=effective_retry_count,
                retry_delay=retry_delay,
                retryable_result=lambda value: is_retryable_http_status(
                    response_http_status(value)
                ),
                is_retryable_exception=_is_retryable_agent_a1_exception,
                on_attempt=record_agent_a1_attempt,
            )
        except Exception as exc:
            if trace_llm_call is not None:
                trace_llm_call(
                    provider="agent_a1",
                    backend="openai_chat_completions",
                    model=effective_model,
                    messages=messages,
                    error=f"agent_a1_chat_error:{type(exc).__name__}",
                    parameters={
                        **a1_trace_parameters,
                        "attempt": a1_attempt_index,
                    },
                )
            raise
        if trace_llm_call is not None:
            trace_llm_call(
                provider="agent_a1",
                backend="openai_chat_completions",
                model=effective_model,
                messages=messages,
                raw_response=json.dumps(
                    envelope,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                parameters={
                    **a1_trace_parameters,
                    "attempt": a1_attempt_index,
                },
            )
        return envelope

    if provider == CODEX_RESPONSES_PROVIDER:
        codex_attempt_index = 0

        def request_codex_response() -> dict[str, Any]:
            """发送一次 Responses 请求；统一重试器决定是否再次连接。"""

            return _call_codex_responses_request(
                request,
                timeout=timeout,
            )

        def record_codex_attempt(
            attempt_index: int,
            _response: dict[str, Any] | None,
            _error: BaseException | None,
        ) -> None:
            """只保存本地尝试序号，不把 provider 异常正文写入 trace。"""

            nonlocal codex_attempt_index
            codex_attempt_index = attempt_index

        try:
            # 直接调用方传入的重试预算在这里生效；ModelOperation 外层通过
            # suppress_nested_api_retries 把同一预算收束到唯一的业务边界。
            envelope = retry_api_call(
                request_codex_response,
                retry_count=effective_retry_count,
                retry_delay=retry_delay,
                retryable_result=lambda value: is_retryable_http_status(
                    response_http_status(value)
                ),
                is_retryable_exception=_is_retryable_codex_responses_exception,
                on_attempt=record_codex_attempt,
            )
        except Exception as exc:
            # SDK 异常可能包含完整 URL 或 provider 正文；trace 只保存稳定类型，
            # 不把原始异常正文、地址或认证内容写入运行记录。
            if trace_llm_call is not None:
                trace_llm_call(
                    provider="arche_codex",
                    backend="openai_responses",
                    model=effective_model,
                    messages=messages,
                    error=f"codex_responses_error:{type(exc).__name__}",
                    parameters={
                        **trace_parameters,
                        "attempt": codex_attempt_index,
                    },
                )
            raise
        if trace_llm_call is not None:
            trace_llm_call(
                provider="arche_codex",
                backend="openai_responses",
                model=effective_model,
                messages=messages,
                raw_response=json.dumps(
                    envelope,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                parameters={
                    **trace_parameters,
                    "attempt": codex_attempt_index,
                },
            )
        return envelope

    def record_failed_attempt(
        attempt_index: int,
        response: Any,
        error: BaseException | None,
    ) -> None:
        nonlocal last_attempt_index
        last_attempt_index = attempt_index
        if error is None:
            status = response_http_status(response)
            if not is_retryable_http_status(status):
                return
            error = RuntimeError(f"OpenAI-compatible HTTP {status} 可重试失败")
        if error is None or trace_llm_call is None:
            return
        trace_llm_call(
            provider="deepseek",
            backend="openai_compatible",
            model=effective_model,
            messages=messages,
            error=str(error),
            parameters={**trace_parameters, "attempt": attempt_index},
        )

    def request_provider_response() -> Any:
        """发送一次真实请求，并在流式模式下先完成响应重组。"""

        response = client.chat.completions.create(**kwargs)
        if stream:
            return _assemble_streaming_response(response)
        return response

    response = retry_api_call(
        request_provider_response,
        retry_count=effective_retry_count,
        retry_delay=retry_delay,
        retryable_result=lambda value: is_retryable_http_status(
            response_http_status(value)
        ),
        on_attempt=record_failed_attempt,
    )
    envelope = normalize_provider_envelope(response)
    if trace_llm_call is not None:
        trace_llm_call(
            provider="deepseek",
            backend="openai_compatible",
            model=effective_model,
            messages=messages,
            raw_response=json.dumps(
                envelope,
                ensure_ascii=False,
                sort_keys=True,
            ),
            parameters={**trace_parameters, "attempt": last_attempt_index},
        )
    return envelope


def call_compiled_model_request(
    request: Mapping[str, Any],
    timeout: int = 900,
    *,
    retry_count: int | None = None,
    retry_delay: float = MODEL_CALL_RETRY_DELAY_SECONDS,
) -> Any:
    """Dispatch one compiled model request through the selected harness.

    ``ARCHE_PLANNER_DRIVER=react`` preserves the OpenAI-compatible transport above,
    including its streaming and retry semantics.  Other harnesses replace only
    the model-driving boundary; callers retain the existing schema, domain,
    receipt, and audit validation around the returned provider envelope.
    """

    from chemistry_multiagent.harness.dispatcher import get_harness_dispatcher

    dispatcher = get_harness_dispatcher(
        react_fallback=_call_compiled_model_request_react,
    )
    if dispatcher.kind == "react":
        return _call_compiled_model_request_react(
            request,
            timeout=timeout,
            retry_count=retry_count,
            retry_delay=retry_delay,
        )
    # All alternative Harnesses retain the same immutable request contract as
    # the original React transport. Harness selection changes execution only.
    _validate_compiled_model_request_contract(request)
    return dispatcher.run_compiled(request, timeout=float(timeout))


def call_deepseek_api_envelope(
    messages: List[Dict[str, str]],
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int | None = DEFAULT_MAX_TOKENS,
    timeout: int = 300,
    max_retries: int | None = None,
    retry_delay: float = MODEL_CALL_RETRY_DELAY_SECONDS,
    *,
    retry_count: int = MODEL_CALL_RETRY_COUNT,
) -> Any:
    """仅在网络传输失败时重试，并保留原始响应结构交给上层严格校验。

    空正文、JSON 错误或字段缺失属于模型内容问题，不在这里重试或修补，避免同一
    逻辑操作产生多份无法区分的模型答案。
    """

    retry_count = _resolve_model_api_retry_count(retry_count, max_retries)
    resolved_max_tokens = _validate_optional_max_tokens(max_tokens)
    effective_model = normalize_configured_model(model)
    reasoning_effort = (
        resolve_deepseek_v4_pro_reasoning_effort()
        if is_deepseek_v4_pro_model(effective_model)
        else None
    )
    request_kwargs = {
        "model": effective_model,
        "messages": messages,
        "temperature": temperature,
        "timeout": timeout,
    }
    if reasoning_effort is not None:
        request_kwargs["reasoning_effort"] = reasoning_effort
    if resolved_max_tokens is not None:
        request_kwargs["max_tokens"] = resolved_max_tokens

    trace_parameters = {
        "temperature": temperature,
        "max_tokens": resolved_max_tokens,
        "reasoning_effort": reasoning_effort,
        "provider_response_format": None,
        "timeout": timeout,
        "requested_model": model,
    }
    last_attempt_index = 0

    def record_failed_attempt(
        attempt_index: int,
        response: Any,
        error: BaseException | None,
    ) -> None:
        nonlocal last_attempt_index
        last_attempt_index = attempt_index
        if error is None:
            status = response_http_status(response)
            if not is_retryable_http_status(status):
                return
            error = RuntimeError(f"OpenAI-compatible HTTP {status} 可重试失败")
        if error is None or trace_llm_call is None:
            return
        trace_llm_call(
            provider="deepseek",
            backend="openai_compatible",
            model=effective_model,
            messages=messages,
            error=str(error),
            parameters={**trace_parameters, "attempt": attempt_index},
        )

    response = retry_api_call(
        lambda: client.chat.completions.create(**request_kwargs),
        retry_count=retry_count,
        retry_delay=retry_delay,
        retryable_result=lambda value: is_retryable_http_status(
            response_http_status(value)
        ),
        on_attempt=record_failed_attempt,
    )
    envelope = normalize_provider_envelope(response)
    choices = envelope.get("choices") if isinstance(envelope, dict) else None
    content = None
    finish_reason = None
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        finish_reason = choices[0].get("finish_reason")
        message = choices[0].get("message")
        if isinstance(message, dict):
            content = message.get("content")
    if trace_llm_call is not None:
        trace_llm_call(
            provider="deepseek",
            backend="openai_compatible",
            model=effective_model,
            messages=messages,
            raw_response=(
                content
                if isinstance(content, str)
                else json.dumps(envelope, ensure_ascii=False, sort_keys=True)
            ),
            parameters={
                **trace_parameters,
                "attempt": last_attempt_index,
                "response_finish_reason": finish_reason,
                "response_content_length": (
                    len(content) if isinstance(content, str) else 0
                ),
            },
        )
    return envelope


def call_deepseek_api(
    messages: List[Dict[str, str]],
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int | None = DEFAULT_MAX_TOKENS,
    timeout: int = 300,
    max_retries: int | None = None,
    retry_delay: float = MODEL_CALL_RETRY_DELAY_SECONDS,
    *,
    retry_count: int = MODEL_CALL_RETRY_COUNT,
) -> str:
    """调用 Deepseek API，并在传输失败后按固定间隔重试。

    空正文、非法内容和其他模型输出问题只返回一次给调用方；JSON/领域校验的
    完整重答由 ModelOperation 负责，避免把内容错误误当成网络错误重复提交。
    """

    # 旧代码中许多调用显式传入 deepseek-chat；统一配置会把这些历史别名
    # 映射到 ARCHE_CHEM_MODEL_NAME 指定的实际部署模型。
    retry_count = _resolve_model_api_retry_count(retry_count, max_retries)
    resolved_max_tokens = _validate_optional_max_tokens(max_tokens)
    effective_model = normalize_configured_model(model)
    reasoning_effort = (
        resolve_deepseek_v4_pro_reasoning_effort()
        if is_deepseek_v4_pro_model(effective_model)
        else None
    )
    request_kwargs = {
        "model": effective_model,
        "messages": messages,
        "temperature": temperature,
        "timeout": timeout,
    }
    if reasoning_effort is not None:
        request_kwargs["reasoning_effort"] = reasoning_effort
    if resolved_max_tokens is not None:
        request_kwargs["max_tokens"] = resolved_max_tokens
    trace_parameters = {
        "temperature": temperature,
        "max_tokens": resolved_max_tokens,
        "reasoning_effort": reasoning_effort,
        "provider_response_format": None,
        "timeout": timeout,
        "requested_model": model,
    }
    last_attempt_index = 0

    def record_failed_attempt(
        attempt_index: int,
        response: Any,
        error: BaseException | None,
    ) -> None:
        nonlocal last_attempt_index
        last_attempt_index = attempt_index
        if error is None:
            status = response_http_status(response)
            if not is_retryable_http_status(status):
                return
            error = RuntimeError(f"OpenAI-compatible HTTP {status} 可重试失败")
        if error is None or trace_llm_call is None:
            return
        trace_llm_call(
            provider="deepseek",
            backend="openai_compatible",
            model=effective_model,
            messages=messages,
            error=str(error),
            parameters={**trace_parameters, "attempt": attempt_index},
        )

    response = retry_api_call(
        lambda: client.chat.completions.create(**request_kwargs),
        retry_count=retry_count,
        retry_delay=retry_delay,
        retryable_result=lambda value: is_retryable_http_status(
            response_http_status(value)
        ),
        on_attempt=record_failed_attempt,
    )
    try:
        if not response or not getattr(response, "choices", None):
            raise RuntimeError("模型响应中没有候选结果")
        choice = response.choices[0]
        message = choice.message
        content = (getattr(message, "content", None) or "").strip()
        reasoning_content = getattr(message, "reasoning_content", None) or ""
        trace_parameters.update(
            {
                "attempt": last_attempt_index,
                "response_finish_reason": getattr(choice, "finish_reason", None),
                "response_content_length": len(content),
                "response_reasoning_length": len(reasoning_content),
            }
        )
        if not content:
            raise RuntimeError(
                "模型返回空内容"
                f"（finish_reason={trace_parameters['response_finish_reason']}, "
                f"reasoning_length={len(reasoning_content)}）"
            )
    except Exception as exc:
        if trace_llm_call is not None:
            trace_llm_call(
                provider="deepseek",
                backend="openai_compatible",
                model=effective_model,
                messages=messages,
                error=str(exc),
                parameters={**trace_parameters, "attempt": last_attempt_index},
            )
        raise
    if trace_llm_call is not None:
        trace_llm_call(
            provider="deepseek",
            backend="openai_compatible",
            model=effective_model,
            messages=messages,
            raw_response=content,
            parameters=trace_parameters,
        )
    return content
