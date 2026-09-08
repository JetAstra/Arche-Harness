import copy
import inspect

import pytest

from chemistry_multiagent.utils import llm_api
from chemistry_multiagent.utils.model_defaults import (
    DEFAULT_EXPERT_FALLBACK_MODEL,
    DEFAULT_GENERAL_MODEL,
    DEEPSEEK_V4_PRO_REASONING_EFFORT,
)
from chemistry_multiagent.utils.model_catalog import (
    MODEL_OPERATION_KEYS,
    compile_model_request,
    get_model_profile,
)


def _router_contract(validator_id="router.output.v2"):
    return {
        "validation_kind": "json_schema",
        "schema": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "additionalProperties": False,
            "properties": {"decision": {"type": "string"}},
            "required": ["decision"],
        },
        "domain_validator_id": validator_id,
    }


def test_all_prompt_catalog_keys_have_an_explicit_model_profile():
    # Planner/Optimizer 与 Hypothesis 的 complement/conflict 都是独立操作注册。
    assert len(MODEL_OPERATION_KEYS) == 18
    for operation, stage in MODEL_OPERATION_KEYS:
        profile = get_model_profile(operation, stage)
        assert profile["profile_id"]
        if operation == "gaussian_expert":
            assert profile["max_tokens"] == 16384
        else:
            assert profile["max_tokens"] is None
        assert profile["provider_output_mode"] in {"json_object", "text"}


def test_general_model_profiles_leave_output_budget_to_the_provider():
    """通用模型不设置本地输出上限，Gaussian 专家目录项保持独立预算。"""

    for operation, stage in MODEL_OPERATION_KEYS:
        profile = get_model_profile(operation, stage)
        expected = 16384 if operation == "gaussian_expert" else None
        assert profile["max_tokens"] == expected

    planner = get_model_profile("workflow_generation", None)
    router = get_model_profile("router", None)
    expert = get_model_profile("gaussian_expert", "technical_failure")

    assert planner["temperature"] == 0.1
    assert planner["provider_output_mode"] == "json_object"
    assert router["temperature"] == 0.0
    assert expert["temperature"] == 0.2


def test_enum_profiles_leave_room_for_deepseek_reasoning():
    """单词答案也不能因通用模型的本地预算而在正文出现前被截断。"""

    for stage in ("relationship", "rank"):
        profile = get_model_profile("hypothesis_generation", stage)
        assert profile["max_tokens"] is None
        assert profile["profile_id"].endswith(".v4")


def test_compiled_request_is_stable_and_does_not_alias_inputs(monkeypatch):
    monkeypatch.setenv("ARCHE_CHEM_MODEL_NAME", "configured-model")
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]
    contract = _router_contract()

    first = compile_model_request(
        operation="router",
        stage=None,
        messages=messages,
        response_contract=contract,
    )
    second = compile_model_request(
        operation="router",
        stage=None,
        messages=copy.deepcopy(messages),
        response_contract=copy.deepcopy(contract),
    )

    assert first == second
    assert first["model"] == "configured-model"
    assert first["prompt_catalog_version"] == "model_prompt_catalog.v33"
    assert first["projector_version"] == "model_input_projection.v23"
    assert first["serializer_version"] == "canonical_model_request.v3"
    assert first["model_profile_version"] == "deepseek_v4_model_profiles.v13"
    assert first["max_tokens"] is None
    assert first["reasoning_effort"] is None
    assert first["provider_response_format"] == {"type": "json_object"}
    assert len(first["response_schema_hash"]) == 64
    messages[0]["content"] = "changed"
    contract["schema"]["properties"] = {}
    assert first["messages"][0]["content"] == "system"
    assert first == second


def test_validator_identity_participates_in_response_schema_hash(monkeypatch):
    monkeypatch.setenv("ARCHE_CHEM_MODEL_NAME", "configured-model")
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]

    first = compile_model_request(
        operation="router",
        stage=None,
        messages=messages,
        response_contract=_router_contract("router.output.v2"),
    )
    changed = compile_model_request(
        operation="router",
        stage=None,
        messages=messages,
        response_contract=_router_contract("router.output.v3"),
    )

    assert first["response_schema_hash"] != changed["response_schema_hash"]


def test_legacy_model_alias_is_resolved_before_the_request_is_saved(monkeypatch):
    monkeypatch.setenv("ARCHE_CHEM_MODEL_NAME", "deployed-model")

    request = compile_model_request(
        operation="router",
        stage=None,
        messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ],
        response_contract=_router_contract(),
        model="deepseek-chat",
    )

    assert request["model"] == "deployed-model"


def test_model_request_defaults_to_official_pro_without_environment(monkeypatch):
    """未配置环境覆盖时，通用请求使用官方 Pro 名称。"""

    monkeypatch.delenv("ARCHE_CHEM_MODEL_NAME", raising=False)

    request = compile_model_request(
        operation="router",
        stage=None,
        messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ],
        response_contract=_router_contract(),
    )

    assert request["model"] == DEFAULT_GENERAL_MODEL


def test_model_defaults_keep_general_and_expert_fallback_unified():
    assert DEFAULT_GENERAL_MODEL == "deepseek-v4-pro"
    assert DEFAULT_EXPERT_FALLBACK_MODEL == DEFAULT_GENERAL_MODEL


@pytest.mark.parametrize(
    ("configured", "expected"),
    [(None, "max"), ("max", "max"), (" high ", "high")],
)
def test_pro_reasoning_effort_resolves_from_environment(monkeypatch, configured, expected):
    """Pro 推理强度默认使用 max，并允许显式降到 high。"""

    if configured is None:
        monkeypatch.delenv("DEEPSEEK_V4_PRO_REASONING_EFFORT", raising=False)
    else:
        monkeypatch.setenv("DEEPSEEK_V4_PRO_REASONING_EFFORT", configured)

    request = compile_model_request(
        operation="router",
        stage=None,
        messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ],
        response_contract=_router_contract(),
        model="deepseek-v4-pro",
    )

    assert request["reasoning_effort"] == expected
    assert DEEPSEEK_V4_PRO_REASONING_EFFORT == "max"


@pytest.mark.parametrize("configured", ["low", "", "MAX", "highly", "max high"])
def test_invalid_pro_reasoning_effort_fails_before_compiling_request(
    monkeypatch, configured
):
    """未登记的 Pro 推理强度必须在请求编译前拒绝。"""

    monkeypatch.setenv("DEEPSEEK_V4_PRO_REASONING_EFFORT", configured)

    with pytest.raises(ValueError, match="DEEPSEEK_V4_PRO_REASONING_EFFORT"):
        compile_model_request(
            operation="router",
            stage=None,
            messages=[
                {"role": "system", "content": "system"},
                {"role": "user", "content": "user"},
            ],
            response_contract=_router_contract(),
            model="deepseek-v4-pro",
        )


@pytest.mark.parametrize(
    ("model", "expected_reasoning_effort", "expected_response_format"),
    [
        ("deepseek-v4-pro", "max", None),
        ("tenant/deepseek-v4-pro-0813", "max", None),
        ("deepseek-v4-flash", None, {"type": "json_object"}),
        ("local-7b", None, {"type": "json_object"}),
    ],
)
def test_compiled_request_records_pro_only_reasoning_and_response_policy(
    monkeypatch,
    model,
    expected_reasoning_effort,
    expected_response_format,
):
    """compiled request 明确记录 Pro 推理参数和实际 response_format 策略。"""

    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    request = compile_model_request(
        operation="router",
        stage=None,
        messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ],
        response_contract=_router_contract(),
        model=model,
    )

    assert request["reasoning_effort"] == expected_reasoning_effort
    assert request["provider_response_format"] == expected_response_format


def test_compiled_pro_request_records_configured_high_reasoning_effort(monkeypatch):
    """compiled request 与实际 Pro 配置一致记录 high。"""

    monkeypatch.setenv("DEEPSEEK_V4_PRO_REASONING_EFFORT", "high")
    request = compile_model_request(
        operation="router",
        stage=None,
        messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ],
        response_contract=_router_contract(),
        model="deepseek-v4-pro",
    )

    assert request["reasoning_effort"] == "high"


@pytest.mark.parametrize("legacy_name", ["deepseek-chat", "deepseek-reasoner"])
def test_legacy_general_aliases_resolve_to_pro_without_environment(
    monkeypatch, legacy_name
):
    monkeypatch.delenv("ARCHE_CHEM_MODEL_NAME", raising=False)

    request = compile_model_request(
        operation="router",
        stage=None,
        messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ],
        response_contract=_router_contract(),
        model=legacy_name,
    )

    assert request["model"] == DEFAULT_GENERAL_MODEL


def test_explicit_blank_model_is_rejected(monkeypatch):
    monkeypatch.delenv("ARCHE_CHEM_MODEL_NAME", raising=False)

    with pytest.raises(TypeError, match="非空"):
        compile_model_request(
            operation="router",
            stage=None,
            messages=[
                {"role": "system", "content": "system"},
                {"role": "user", "content": "user"},
            ],
            response_contract=_router_contract(),
            model=" ",
        )


class _FakeCompletions:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        if self.error is not None:
            raise self.error
        return copy.deepcopy(self.response)


class _FakeClient:
    def __init__(self, completions):
        self.chat = type("Chat", (), {"completions": completions})()


def test_compiled_provider_uses_json_mode_and_exact_profile(monkeypatch):
    completions = _FakeCompletions(
        response={
            "id": "response-1",
            "choices": [
                {"finish_reason": "stop", "message": {"content": "{}"}}
            ],
        }
    )
    monkeypatch.setattr(llm_api, "client", _FakeClient(completions))
    request = compile_model_request(
        operation="router",
        stage=None,
        messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ],
        response_contract=_router_contract(),
        model="configured-model",
    )

    response = llm_api.call_compiled_model_request(request, timeout=17)

    assert response["id"] == "response-1"
    assert len(completions.calls) == 1
    assert completions.calls[0] == {
        "model": "configured-model",
        "messages": request["messages"],
        "temperature": 0.0,
        "top_p": 1.0,
        "stop": None,
        "stream": False,
        "timeout": 17,
        "response_format": {"type": "json_object"},
    }


def test_compiled_pro_provider_uses_max_without_remote_json_mode(monkeypatch):
    """DeepSeek v4 Pro 使用 max，JSON/schema 校验留在本地。"""

    completions = _FakeCompletions(
        response={
            "id": "pro-response-1",
            "choices": [
                {"finish_reason": "stop", "message": {"content": "{}"}}
            ],
        }
    )
    traces = []
    monkeypatch.setattr(llm_api, "client", _FakeClient(completions))
    monkeypatch.setattr(
        llm_api,
        "trace_llm_call",
        lambda **kwargs: traces.append(kwargs),
    )
    request = compile_model_request(
        operation="router",
        stage=None,
        messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ],
        response_contract=_router_contract(),
        model="deepseek-v4-pro",
    )

    llm_api.call_compiled_model_request(request, timeout=17)

    sent = completions.calls[0]
    assert sent["reasoning_effort"] == "max"
    assert "response_format" not in sent
    assert traces[-1]["parameters"]["reasoning_effort"] == "max"
    assert traces[-1]["parameters"]["provider_response_format"] is None


def test_compiled_pro_provider_uses_configured_high_without_remote_json_mode(
    monkeypatch,
):
    """compiled Pro 请求发送 high，并在传输 trace 中记录 high。"""

    monkeypatch.setenv("DEEPSEEK_V4_PRO_REASONING_EFFORT", "high")
    completions = _FakeCompletions(
        response={
            "id": "pro-response-high",
            "choices": [
                {"finish_reason": "stop", "message": {"content": "{}"}}
            ],
        }
    )
    traces = []
    monkeypatch.setattr(llm_api, "client", _FakeClient(completions))
    monkeypatch.setattr(
        llm_api,
        "trace_llm_call",
        lambda **kwargs: traces.append(kwargs),
    )
    request = compile_model_request(
        operation="router",
        stage=None,
        messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ],
        response_contract=_router_contract(),
        model="deepseek-v4-pro",
    )

    llm_api.call_compiled_model_request(request, timeout=17)

    assert completions.calls[0]["reasoning_effort"] == "high"
    assert traces[-1]["parameters"]["reasoning_effort"] == "high"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "reasoning_effort",
            None,
            "必须发送 reasoning_effort=max",
        ),
        (
            "provider_response_format",
            {"type": "json_object"},
            "不支持 response_format",
        ),
    ],
)
def test_compiled_pro_rejects_inaccurate_request_policy(
    monkeypatch,
    field,
    value,
    message,
):
    """Pro compiled request 的审计策略与实际发送策略不一致时直接拒绝。"""

    completions = _FakeCompletions(response={"choices": []})
    monkeypatch.setattr(llm_api, "client", _FakeClient(completions))
    request = compile_model_request(
        operation="router",
        stage=None,
        messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ],
        response_contract=_router_contract(),
        model="deepseek-v4-pro",
    )
    request[field] = value

    with pytest.raises(ValueError, match=message):
        llm_api.call_compiled_model_request(request)
    assert completions.calls == []


def test_compiled_provider_accepts_a_request_that_omits_optional_budget(monkeypatch):
    """兼容没有预算字段的旧 compiled request，发送时仍不补回上限。"""

    completions = _FakeCompletions(
        response={
            "choices": [
                {"finish_reason": "stop", "message": {"content": "{}"}}
            ],
        }
    )
    monkeypatch.setattr(llm_api, "client", _FakeClient(completions))
    request = compile_model_request(
        operation="router",
        stage=None,
        messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ],
        response_contract=_router_contract(),
        model="configured-model",
    )
    request.pop("max_tokens")

    llm_api.call_compiled_model_request(request)

    assert "max_tokens" not in completions.calls[0]


@pytest.mark.parametrize("invalid_budget", [0, -1, True, "512"])
def test_compiled_provider_rejects_invalid_optional_budget(monkeypatch, invalid_budget):
    """compiled request 的可选预算仍必须保持严格的整数类型。"""

    completions = _FakeCompletions(response={"choices": []})
    monkeypatch.setattr(llm_api, "client", _FakeClient(completions))
    request = compile_model_request(
        operation="router",
        stage=None,
        messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ],
        response_contract=_router_contract(),
        model="configured-model",
    )
    request["max_tokens"] = invalid_budget

    with pytest.raises(ValueError, match="max_tokens"):
        llm_api.call_compiled_model_request(request)

    assert completions.calls == []


def test_compiled_provider_does_not_retry_transport_errors(monkeypatch):
    completions = _FakeCompletions(error=RuntimeError("network failed"))
    monkeypatch.setattr(llm_api, "client", _FakeClient(completions))
    request = compile_model_request(
        operation="hypothesis_generation",
        stage="rank",
        messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ],
        response_contract={
            "validation_kind": "text_enum_rank",
            "schema": None,
            "domain_validator_id": None,
        },
        model="deepseek-v4-pro",
    )

    with pytest.raises(RuntimeError, match="network failed"):
        llm_api.call_compiled_model_request(request)

    assert len(completions.calls) == 1
    # rank 的本地输出合同仍是 JSON object；DeepSeek Pro 请求不携带未验证的
    # response_format 参数。
    assert "response_format" not in completions.calls[0]


def test_compiled_provider_disables_sdk_hidden_retries():
    assert llm_api.client.max_retries == 0


def test_compiled_provider_default_timeout_allows_slow_generation():
    timeout = inspect.signature(
        llm_api.call_compiled_model_request
    ).parameters["timeout"].default

    assert timeout == 900
