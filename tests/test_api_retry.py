"""统一 API 重试策略的确定性测试。"""

from __future__ import annotations

import socket
from types import SimpleNamespace

import httpx
import pytest
from requests import exceptions as requests_exceptions

from chemistry_multiagent.agents.retrieval_agent import RetrievalAgent
from chemistry_multiagent.utils import expert_model_provider, llm_api
from chemistry_multiagent.utils import api_retry as api_retry_module
from chemistry_multiagent.utils.api_retry import (
    DEFAULT_API_RETRY_COUNT,
    DEFAULT_API_RETRY_DELAY_SECONDS,
    MODEL_CALL_RETRY_COUNT,
    MODEL_CALL_RETRY_DELAY_SECONDS,
    is_retryable_api_exception,
    retry_api_call,
)


def test_retry_count_means_extra_attempts_and_uses_fixed_delay() -> None:
    calls: list[int] = []
    sleeps: list[float] = []

    def operation() -> str:
        calls.append(len(calls) + 1)
        if len(calls) <= 5:
            raise TimeoutError("temporary timeout")
        return "ok"

    result = retry_api_call(operation, sleeper=sleeps.append)

    assert result == "ok"
    assert len(calls) == DEFAULT_API_RETRY_COUNT + 1
    assert sleeps == [DEFAULT_API_RETRY_DELAY_SECONDS] * DEFAULT_API_RETRY_COUNT


def test_retryable_http_response_is_retried_but_final_response_is_returned() -> None:
    responses = [{"status_code": 503}, {"status_code": 429}, {"status_code": 200}]
    sleeps: list[float] = []

    result = retry_api_call(
        lambda: responses.pop(0),
        sleeper=sleeps.append,
        retryable_result=lambda response: response["status_code"] in {429, 503},
    )

    assert result == {"status_code": 200}
    assert sleeps == [6.0, 6.0]


def test_non_retryable_parameter_and_authentication_failures_are_not_retried() -> None:
    calls: list[int] = []

    def operation() -> None:
        calls.append(1)
        raise ValueError("invalid request")

    with pytest.raises(ValueError, match="invalid request"):
        retry_api_call(operation, sleeper=lambda _delay: pytest.fail("must not sleep"))
    assert len(calls) == 1


def test_http_status_classifier_does_not_retry_authentication_error() -> None:
    class UnauthorizedError(Exception):
        status_code = 401

    assert not is_retryable_api_exception(UnauthorizedError())
    assert is_retryable_api_exception(socket.timeout("temporary"))


def test_requests_parameter_errors_are_not_mistaken_for_transport_failures() -> None:
    assert not is_retryable_api_exception(requests_exceptions.InvalidURL("bad url"))
    assert is_retryable_api_exception(requests_exceptions.ConnectionError("down"))


@pytest.mark.parametrize(
    "error",
    [
        httpx.ReadError("read failed"),
        httpx.WriteError("write failed"),
        httpx.RemoteProtocolError("server disconnected"),
    ],
)
def test_httpx_transport_errors_are_retryable(error: Exception) -> None:
    """HTTPX 已连接后的读写或远端协议中断仍属于瞬时传输失败。"""

    assert is_retryable_api_exception(error)


def _openai_response(content: str = "ok") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason="stop",
            )
        ],
        usage=None,
    )


def test_legacy_llm_api_retries_six_times_with_ten_second_interval(monkeypatch) -> None:
    calls: list[int] = []
    sleeps: list[float] = []

    def fake_create(**_kwargs):
        calls.append(1)
        if len(calls) <= MODEL_CALL_RETRY_COUNT:
            raise TimeoutError("temporary provider timeout")
        return _openai_response("ok")

    monkeypatch.setattr(llm_api.client.chat.completions, "create", fake_create)
    monkeypatch.setattr(llm_api, "trace_llm_call", None)
    monkeypatch.setattr(api_retry_module.time, "sleep", sleeps.append)

    assert llm_api.call_deepseek_api([{"role": "user", "content": "test"}]) == "ok"
    assert len(calls) == MODEL_CALL_RETRY_COUNT + 1
    assert sleeps == [MODEL_CALL_RETRY_DELAY_SECONDS] * MODEL_CALL_RETRY_COUNT


def test_expert_provider_retries_six_times_with_ten_second_interval(monkeypatch) -> None:
    calls: list[int] = []
    sleeps: list[float] = []

    class FakeCompletions:
        def create(self, **_kwargs):
            calls.append(1)
            if len(calls) <= MODEL_CALL_RETRY_COUNT:
                raise TimeoutError("temporary expert timeout")
            return _openai_response('{"status":"approved"}')

    class FakeClient:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(expert_model_provider.openai, "OpenAI", FakeClient)
    monkeypatch.setattr(api_retry_module.time, "sleep", sleeps.append)
    provider = expert_model_provider.OpenAICompatibleExpertProvider(
        model="expert",
        base_url="https://expert.invalid/v1",
        api_key="test-key",
        role="expert",
    )

    result = provider([{"role": "user", "content": "test"}])
    assert result["choices"][0]["message"]["content"] == '{"status":"approved"}'
    assert len(calls) == MODEL_CALL_RETRY_COUNT + 1
    assert sleeps == [MODEL_CALL_RETRY_DELAY_SECONDS] * MODEL_CALL_RETRY_COUNT


def test_retrieval_outer_retry_suppresses_injected_provider_retry() -> None:
    """Retrieval 注入 provider 使用公共重试器时也不能形成两层重试乘积。"""

    transport_calls: list[int] = []
    inner_sleeps: list[float] = []
    outer_sleeps: list[float] = []

    def call_transport() -> None:
        transport_calls.append(1)
        raise TimeoutError("temporary retrieval timeout")

    def provider(_messages):
        return retry_api_call(call_transport, sleeper=inner_sleeps.append)

    agent = object.__new__(RetrievalAgent)
    agent.retrieval_provider = provider
    agent._paper_search_sleeper = outer_sleeps.append

    with pytest.raises(TimeoutError, match="temporary retrieval timeout"):
        agent._call_retrieval_provider(
            {"messages": [{"role": "user", "content": "test"}]},
            stage="retrieval_test",
        )

    assert len(transport_calls) == MODEL_CALL_RETRY_COUNT + 1
    assert inner_sleeps == []
    assert outer_sleeps == [MODEL_CALL_RETRY_DELAY_SECONDS] * MODEL_CALL_RETRY_COUNT
