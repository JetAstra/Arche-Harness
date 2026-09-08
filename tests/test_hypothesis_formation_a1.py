"""Agent A1 假设形成 provider 的模型目录、请求和传输边界测试。"""

from __future__ import annotations

import copy
import json

import pytest

from chemistry_multiagent.utils import llm_api
from chemistry_multiagent.utils.model_catalog import compile_model_request
from chemistry_multiagent.utils.model_defaults import (
    AGENT_A1_API_KEY_ENV,
    AGENT_A1_BASE_URL_ENV,
    AGENT_A1_CHAT_COMPLETIONS_PROVIDER,
    AGENT_A1_MODEL_NAME,
    AGENT_A1_MODEL_NAME_ENV,
    ARCHE_CHEM_CHAT_COMPLETIONS_PROVIDER,
    HYPOTHESIS_FORMATION_PROVIDER_ENV,
)


def _contract(kind: str = "text") -> dict[str, object]:
    """构造不含地址和认证正文的最小响应合同。"""

    return {
        "validation_kind": kind,
        "schema": None,
        "domain_validator_id": None,
    }


@pytest.fixture
def agent_a1_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """安装脱敏 A1 配置标记，并固定 ARCHE_CHEM 比较模型。"""

    monkeypatch.setenv(HYPOTHESIS_FORMATION_PROVIDER_ENV, "agent_a1")
    monkeypatch.setenv(AGENT_A1_MODEL_NAME_ENV, AGENT_A1_MODEL_NAME)
    monkeypatch.setenv(AGENT_A1_BASE_URL_ENV, "https://a1.example/v1")
    monkeypatch.setenv(AGENT_A1_API_KEY_ENV, "a1-api-key-test")
    monkeypatch.setenv("ARCHE_CHEM_MODEL_NAME", "deepseek-v4-pro")
    monkeypatch.delenv("DEEPSEEK_V4_PRO_REASONING_EFFORT", raising=False)


@pytest.mark.parametrize(
    ("stage", "output_mode", "temperature"),
    [
        ("query", "text", 0.7),
        ("hypothesis", "text", 0.7),
        ("merge", "json_object", 0.0),
        ("complement", "json_object", 0.0),
        ("conflict", "json_object", 0.0),
        ("regeneration", "text", 0.7),
    ],
)
def test_agent_a1_compiled_formation_request_records_real_chat_policy(
    agent_a1_environment: None,
    stage: str,
    output_mode: str,
    temperature: float,
) -> None:
    """六个形成阶段记录 A1 模型、profile 和非流式 Chat Completions 策略。"""

    request = compile_model_request(
        operation="hypothesis_generation",
        stage=stage,
        messages=[
            {"role": "system", "content": "system-marker"},
            {"role": "user", "content": "user-marker"},
        ],
        response_contract=_contract(
            "json_schema" if output_mode == "json_object" else "text"
        ),
        # 形成 provider 选择后，显式旧模型不能覆盖真实 A1 模型。
        model="deepseek-v4-pro",
    )

    assert request["provider"] == AGENT_A1_CHAT_COMPLETIONS_PROVIDER
    assert request["model"] == AGENT_A1_MODEL_NAME
    assert request["serializer_version"] == "canonical_model_request.v3"
    assert request["model_profile_version"] == "agent_a1_hypothesis_profiles.v1"
    assert request["profile_id"].endswith(".agent_a1")
    assert request["temperature"] == temperature
    assert request["top_p"] == 1.0
    assert request["stream"] is False
    assert request["reasoning_effort"] is None
    assert request["max_tokens"] is None
    assert request["provider_output_mode"] == output_mode
    assert request["provider_response_format"] is None
    request_text = json.dumps(request, ensure_ascii=False, sort_keys=True)
    assert "a1.example" not in request_text
    assert "a1-api-key-test" not in request_text


def test_cluster_alias_selects_the_same_a1_provider(
    agent_a1_environment: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """显式 cluster 兼容别名只归一为 Agent A1，不改变其它模型。"""

    monkeypatch.setenv(HYPOTHESIS_FORMATION_PROVIDER_ENV, "cluster")
    request = compile_model_request(
        operation="hypothesis_generation",
        stage="query",
        messages=[{"role": "user", "content": "user-marker"}],
        response_contract=_contract(),
    )

    assert request["provider"] == AGENT_A1_CHAT_COMPLETIONS_PROVIDER
    assert request["model"] == AGENT_A1_MODEL_NAME


def test_unset_formation_provider_keeps_codex_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """未设置新开关时，六个形成阶段仍编译为既有 Codex Responses。"""

    monkeypatch.delenv(HYPOTHESIS_FORMATION_PROVIDER_ENV, raising=False)
    monkeypatch.setenv("ARCHE_CODEX_MODEL_NAME", "sol-model-marker")
    monkeypatch.setenv("ARCHE_CODEX_BASE_URL", "https://codex.example/v1")
    monkeypatch.setenv("ARCHE_CODEX_API_KEY", "codex-key-marker")
    request = compile_model_request(
        operation="hypothesis_generation",
        stage="query",
        messages=[{"role": "user", "content": "user-marker"}],
        response_contract=_contract(),
    )

    assert request["provider"] == "codex_responses"
    assert request["model"] == "sol-model-marker"
    assert request["serializer_version"] == "canonical_model_request.v3"
    assert request["model_profile_version"] == "deepseek_v4_model_profiles.v13"
    assert request["profile_id"].endswith(".v4")
    assert request["stream"] is True
    assert request["reasoning_effort"] == "xhigh"


@pytest.mark.parametrize("stage", ["relationship", "rank"])
def test_a1_switch_does_not_move_relationship_or_rank(
    agent_a1_environment: None,
    stage: str,
) -> None:
    """A1 开关不影响比较阶段仍使用 ARCHE_CHEM provider。"""

    request = compile_model_request(
        operation="hypothesis_generation",
        stage=stage,
        messages=[{"role": "user", "content": "user-marker"}],
        response_contract=_contract("json_schema"),
    )

    assert request["provider"] == ARCHE_CHEM_CHAT_COMPLETIONS_PROVIDER
    assert request["model"] == "deepseek-v4-pro"
    assert request["reasoning_effort"] == "max"
    assert request["stream"] is False


def test_a1_provider_identity_rejects_tampered_nonformation_catalog(
    agent_a1_environment: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """即使请求被篡改，A1 provider 也不能落到关系比较阶段。"""

    request = compile_model_request(
        operation="hypothesis_generation",
        stage="query",
        messages=[{"role": "user", "content": "user-marker"}],
        response_contract=_contract(),
    )
    request["catalog_key"] = "hypothesis_generation.relationship"
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        llm_api,
        "_call_agent_a1_chat_request",
        lambda *_args, **_kwargs: calls.append({}) or {},
    )

    with pytest.raises(ValueError, match="provider 与当前阶段路由不一致"):
        llm_api.call_compiled_model_request(request, retry_count=0)
    assert calls == []


def test_invalid_formation_provider_fails_before_model_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """非法开关值不能静默回退到 Codex 或集群。"""

    monkeypatch.setenv(HYPOTHESIS_FORMATION_PROVIDER_ENV, "gpt")
    with pytest.raises(ValueError, match=HYPOTHESIS_FORMATION_PROVIDER_ENV):
        compile_model_request(
            operation="hypothesis_generation",
            stage="query",
            messages=[{"role": "user", "content": "user-marker"}],
            response_contract=_contract(),
        )


@pytest.mark.parametrize(
    "missing_env",
    [AGENT_A1_BASE_URL_ENV, AGENT_A1_API_KEY_ENV, AGENT_A1_MODEL_NAME_ENV],
)
def test_missing_a1_setting_fails_before_network_request(
    agent_a1_environment: None,
    monkeypatch: pytest.MonkeyPatch,
    missing_env: str,
) -> None:
    """A1 三项配置缺失时只报告变量名，并在编译阶段停止。"""

    monkeypatch.delenv(missing_env)
    with pytest.raises(ValueError, match=missing_env) as raised:
        compile_model_request(
            operation="hypothesis_generation",
            stage="query",
            messages=[{"role": "user", "content": "user-marker"}],
            response_contract=_contract(),
        )
    assert "a1.example" not in str(raised.value)
    assert "a1-api-key-test" not in str(raised.value)


def test_a1_model_name_can_be_changed_before_network_request(
    agent_a1_environment: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A1 provider 的模型名来自配置，不绑定历史 Agents-A1 字符串。"""

    monkeypatch.setenv(AGENT_A1_MODEL_NAME_ENV, "a1-model-marker")
    request = compile_model_request(
        operation="hypothesis_generation",
        stage="query",
        messages=[{"role": "user", "content": "user-marker"}],
        response_contract=_contract(),
    )
    assert request["model"] == "a1-model-marker"


class _FakeA1Completions:
    """记录 Chat Completions 请求的离线替身。"""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(copy.deepcopy(kwargs))
        return {
            "id": "a1-response-marker",
            "model": AGENT_A1_MODEL_NAME,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "answer-marker"},
                }
            ],
        }


class _FakeA1Client:
    """保存 client 初始化参数并暴露兼容的 chat.completions 资源。"""

    instances: list["_FakeA1Client"] = []

    def __init__(self, **kwargs: object) -> None:
        self.init_kwargs = copy.deepcopy(kwargs)
        self.completions = _FakeA1Completions()
        self.chat = type("Chat", (), {"completions": self.completions})()
        self.instances.append(self)


def test_a1_transport_uses_bearer_api_key_and_stage_generation_parameters(
    agent_a1_environment: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A1 使用 Bearer API key 和阶段参数，不发送 Codex 或本地预算字段。"""

    _FakeA1Client.instances.clear()
    monkeypatch.setattr(llm_api.openai, "OpenAI", _FakeA1Client)
    monkeypatch.setattr(llm_api, "_AGENT_A1_CLIENT", None)
    monkeypatch.setattr(llm_api, "_AGENT_A1_CLIENT_SIGNATURE", None)
    traces: list[dict[str, object]] = []
    monkeypatch.setattr(llm_api, "trace_llm_call", lambda **kwargs: traces.append(kwargs))
    request = compile_model_request(
        operation="hypothesis_generation",
        stage="hypothesis",
        messages=[
            {"role": "system", "content": "system-marker"},
            {"role": "user", "content": "user-marker"},
        ],
        response_contract=_contract(),
    )

    response = llm_api.call_compiled_model_request(request, timeout=17, retry_count=0)

    assert response["model"] == AGENT_A1_MODEL_NAME
    assert len(_FakeA1Client.instances) == 1
    client = _FakeA1Client.instances[0]
    assert client.init_kwargs["base_url"] == "https://a1.example/v1"
    assert client.init_kwargs["api_key"] == "a1-api-key-test"
    assert "default_headers" not in client.init_kwargs
    sent = client.completions.calls[0]
    assert sent == {
        "model": AGENT_A1_MODEL_NAME,
        "messages": request["messages"],
        "temperature": 0.7,
        "top_p": 1.0,
        "stop": None,
        "stream": False,
        "timeout": 17,
    }
    trace_text = json.dumps(traces, ensure_ascii=False, sort_keys=True)
    assert "a1-api-key-test" not in trace_text
    assert "a1.example" not in trace_text
    assert traces[-1]["provider"] == "agent_a1"
    assert traces[-1]["backend"] == "openai_chat_completions"


def test_a1_transport_sanitizes_provider_exception_before_audit(
    agent_a1_environment: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A1 传输失败只保存脱敏错误类型，不保留原始地址或认证正文。"""

    class _FailingCompletions(_FakeA1Completions):
        def create(self, **_kwargs: object) -> dict[str, object]:
            raise RuntimeError("secret-key https://a1.invalid/raw-response")

    class _FailingClient(_FakeA1Client):
        def __init__(self, **kwargs: object) -> None:
            super().__init__(**kwargs)
            self.completions = _FailingCompletions()
            self.chat = type("Chat", (), {"completions": self.completions})()

    monkeypatch.setattr(llm_api.openai, "OpenAI", _FailingClient)
    monkeypatch.setattr(llm_api, "_AGENT_A1_CLIENT", None)
    monkeypatch.setattr(llm_api, "_AGENT_A1_CLIENT_SIGNATURE", None)
    traces: list[dict[str, object]] = []
    monkeypatch.setattr(llm_api, "trace_llm_call", lambda **kwargs: traces.append(kwargs))
    request = compile_model_request(
        operation="hypothesis_generation",
        stage="query",
        messages=[{"role": "user", "content": "user-marker"}],
        response_contract=_contract(),
    )

    with pytest.raises(RuntimeError, match="Agent A1 Chat Completions request failed") as raised:
        llm_api.call_compiled_model_request(request, retry_count=0)

    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    trace_text = json.dumps(traces, ensure_ascii=False, sort_keys=True)
    assert "secret-key" not in trace_text
    assert "a1.invalid" not in trace_text
    assert "raw-response" not in trace_text
