"""假设形成阶段的 Sol 路由、Responses 请求合同和流式失败边界测试。"""

from __future__ import annotations

import copy

import pytest

from chemistry_multiagent.utils import llm_api, model_operation
from chemistry_multiagent.utils.model_catalog import compile_model_request
from chemistry_multiagent.utils.model_defaults import (
    CODEX_API_KEY_ENV,
    CODEX_BASE_URL_ENV,
    CODEX_MODEL_NAME_ENV,
    CODEX_REASONING_EFFORT,
)


def _contract(validation_kind: str = "text") -> dict[str, object]:
    """构造最小响应合同，避免测试正文携带配置内容。"""

    return {
        "validation_kind": validation_kind,
        "schema": None,
        "domain_validator_id": None,
    }


@pytest.fixture(autouse=True)
def _marker_provider_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """为路由测试安装脱敏配置标记，不读取或打印本机配置正文。"""

    monkeypatch.setenv(CODEX_API_KEY_ENV, "codex-key-marker")
    monkeypatch.setenv(CODEX_BASE_URL_ENV, "https://codex.example/v1")
    monkeypatch.setenv(CODEX_MODEL_NAME_ENV, "sol-model-marker")
    monkeypatch.delenv("ARCHE_CHEM_MODEL_NAME", raising=False)
    monkeypatch.delenv("DEEPSEEK_V4_PRO_REASONING_EFFORT", raising=False)


@pytest.mark.parametrize(
    ("stage", "output_mode"),
    [
        ("query", "text"),
        ("hypothesis", "text"),
        ("merge", "json_object"),
        ("complement", "json_object"),
        ("conflict", "json_object"),
        ("regeneration", "text"),
    ],
)
def test_hypothesis_formation_stages_compile_to_sol_xhigh_stream(
    stage: str,
    output_mode: str,
) -> None:
    """六个假设形成阶段统一使用独立 Sol、xhigh 和 Responses 流。"""

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
        # 显式传入 DeepSeek 也不能覆盖阶段路由。
        model="deepseek-v4-pro",
    )

    assert request["model"] == "sol-model-marker"
    assert request["reasoning_effort"] == CODEX_REASONING_EFFORT == "xhigh"
    assert request["stream"] is True
    assert request["provider_output_mode"] == output_mode
    assert request["provider_response_format"] is None


@pytest.mark.parametrize("stage", ["relationship", "rank"])
def test_relationship_and_rank_use_configured_arche_chem_model(stage: str) -> None:
    """关系比较和排名使用 ARCHE_CHEM 配置模型，不绑定具体模型名称。"""

    request = compile_model_request(
        operation="hypothesis_generation",
        stage=stage,
        messages=[
            {"role": "system", "content": "system-marker"},
            {"role": "user", "content": "user-marker"},
        ],
        response_contract=_contract("json_schema"),
    )

    assert request["model"] == "deepseek-v4-pro"
    assert request["reasoning_effort"] == "max"
    assert request["stream"] is False
    assert request["provider_output_mode"] == "json_object"
    assert request["provider_response_format"] is None


@pytest.mark.parametrize("stage", ["relationship", "rank"])
def test_comparison_accepts_explicit_model_override(stage: str) -> None:
    """比较阶段可以显式选择 ARCHE_CHEM 端点提供的其他模型。"""

    request = compile_model_request(
        operation="hypothesis_generation",
        stage=stage,
        messages=[
            {"role": "system", "content": "system-marker"},
            {"role": "user", "content": "user-marker"},
        ],
        response_contract=_contract("json_schema"),
        model="bailian/deepseek-v4-flash",
    )

    assert request["model"] == "bailian/deepseek-v4-flash"
    assert request["provider"] == "arche_chem_chat_completions"
    assert request["reasoning_effort"] is None
    assert request["provider_response_format"] == {"type": "json_object"}


@pytest.mark.parametrize("stage", ["relationship", "rank"])
def test_comparison_accepts_non_pro_arche_chem_environment(
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    """比较阶段的 ARCHE_CHEM 环境可以使用非 Pro 模型。"""

    monkeypatch.setenv("ARCHE_CHEM_MODEL_NAME", "bailian/deepseek-v4-flash")
    request = compile_model_request(
        operation="hypothesis_generation",
        stage=stage,
        messages=[
            {"role": "system", "content": "system-marker"},
            {"role": "user", "content": "user-marker"},
        ],
        response_contract=_contract("json_schema"),
    )
    assert request["model"] == "bailian/deepseek-v4-flash"
    assert request["reasoning_effort"] is None


def test_comparison_accepts_registered_deepseek_pro_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """比较阶段接受已登记的 DeepSeek Pro 数字版本名并保持 max。"""

    monkeypatch.setenv("ARCHE_CHEM_MODEL_NAME", "deepseek-v4-pro-0813")
    request = compile_model_request(
        operation="hypothesis_generation",
        stage="relationship",
        messages=[
            {"role": "system", "content": "system-marker"},
            {"role": "user", "content": "user-marker"},
        ],
        response_contract=_contract("json_schema"),
    )

    assert request["model"] == "deepseek-v4-pro-0813"
    assert request["reasoning_effort"] == "max"


@pytest.mark.parametrize("stage", ["chemistry_context", "keyword", "document_qa", "literature_review"])
def test_retrieval_remains_arche_chem(stage: str) -> None:
    """Retrieval 不随假设形成阶段迁移到 Sol。"""

    request = compile_model_request(
        operation="retrieval",
        stage=stage,
        messages=[
            {"role": "system", "content": "system-marker"},
            {"role": "user", "content": "user-marker"},
        ],
        response_contract=_contract(),
    )

    assert request["model"] == "deepseek-v4-pro"
    assert request["reasoning_effort"] == "max"
    assert request["stream"] is False


@pytest.mark.parametrize(
    "missing_name",
    [CODEX_API_KEY_ENV, CODEX_BASE_URL_ENV, CODEX_MODEL_NAME_ENV],
)
def test_missing_sol_setting_fails_closed_before_compiling(
    monkeypatch: pytest.MonkeyPatch,
    missing_name: str,
) -> None:
    """Sol 阶段缺少任一独立配置时不回退 ARCHE_CHEM。"""

    monkeypatch.delenv(missing_name, raising=False)
    with pytest.raises(ValueError, match=missing_name):
        compile_model_request(
            operation="hypothesis_generation",
            stage="query",
            messages=[
                {"role": "system", "content": "system-marker"},
                {"role": "user", "content": "user-marker"},
            ],
            response_contract=_contract(),
        )


class _FakeResponseStream:
    """模拟 Responses 流上下文和最终响应读取。"""

    def __init__(self, response: dict[str, object]) -> None:
        self.response = response

    def __enter__(self) -> "_FakeResponseStream":
        return self

    def __exit__(self, *_args: object) -> bool:
        return False

    def get_final_response(self) -> dict[str, object]:
        return self.response


class _FakeResponses:
    """记录 Responses 流请求并返回最小成功 response。"""

    def __init__(self) -> None:
        self.stream_calls: list[dict[str, object]] = []

    def stream(self, **kwargs: object) -> _FakeResponseStream:
        self.stream_calls.append(copy.deepcopy(kwargs))
        return _FakeResponseStream(
            {
                "id": "sol-response-marker",
                "model": "sol-model-marker",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "response-marker"}
                        ],
                    }
                ],
            }
        )


class _FakeCodexClient:
    """提供只包含 Responses 资源的测试客户端。"""

    def __init__(self, responses: _FakeResponses) -> None:
        self.responses = responses


def _reset_codex_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """清除进程内测试客户端缓存，不触碰真实配置。"""

    monkeypatch.setattr(llm_api, "_CODEX_CLIENT", None)
    monkeypatch.setattr(llm_api, "_CODEX_CLIENT_SIGNATURE", None)


def test_sol_compiled_request_uses_responses_stream_and_safe_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sol 只发送 Responses 参数，不发送 Chat Completions 参数。"""

    responses = _FakeResponses()
    created: list[tuple[str, str]] = []

    def build_client(api_key: str, base_url: str) -> _FakeCodexClient:
        created.append((api_key, base_url))
        return _FakeCodexClient(responses)

    monkeypatch.setattr(llm_api, "_build_codex_responses_client", build_client)
    _reset_codex_client(monkeypatch)
    request = compile_model_request(
        operation="hypothesis_generation",
        stage="query",
        messages=[
            {"role": "system", "content": "system-marker"},
            {"role": "user", "content": "user-marker"},
        ],
        response_contract=_contract(),
    )

    envelope = llm_api.call_compiled_model_request(request, timeout=17)

    assert created == [("codex-key-marker", "https://codex.example/v1")]
    assert request["stream"] is True
    assert responses.stream_calls == [
        {
            "model": "sol-model-marker",
            "input": "user-marker",
            "reasoning": {"effort": "xhigh"},
            "store": False,
            "timeout": 17,
            "instructions": "system-marker",
        }
    ]
    assert envelope["model"] == "sol-model-marker"
    assert envelope["choices"][0]["message"]["content"] == "response-marker"
    assert "messages" not in responses.stream_calls[0]
    assert "reasoning_effort" not in responses.stream_calls[0]
    assert "response_format" not in responses.stream_calls[0]
    assert "codex-key-marker" not in repr(responses.stream_calls[0])
    assert "https://codex.example/v1" not in repr(responses.stream_calls[0])


def test_direct_sol_retry_count_retries_before_any_stream_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """直接调用传入 retry_count 时，未收事件的瞬时失败会统一重试。"""

    class _RetryingResponseStream:
        def __init__(self, attempt_index: int) -> None:
            self.attempt_index = attempt_index

        def __enter__(self) -> "_RetryingResponseStream":
            return self

        def __exit__(self, *_args: object) -> bool:
            return False

        def __iter__(self):
            # 保持真实 ResponseStream 的公开迭代形状，但本次连接尚未收到事件。
            return iter(())

        def get_final_response(self) -> dict[str, object]:
            if self.attempt_index < 3:
                raise ConnectionError("transient provider failure")
            return {
                "id": "sol-response-marker",
                "model": "sol-model-marker",
                "status": "completed",
                "output_text": "response-marker",
            }

    class _RetryingResponses:
        def __init__(self) -> None:
            self.calls = 0

        def stream(self, **_kwargs: object) -> _RetryingResponseStream:
            self.calls += 1
            return _RetryingResponseStream(self.calls)

    responses = _RetryingResponses()
    monkeypatch.setattr(
        llm_api,
        "_build_codex_responses_client",
        lambda _api_key, _base_url: _FakeCodexClient(responses),
    )
    _reset_codex_client(monkeypatch)
    traces: list[dict[str, object]] = []
    monkeypatch.setattr(llm_api, "trace_llm_call", lambda **kwargs: traces.append(kwargs))
    request = compile_model_request(
        operation="hypothesis_generation",
        stage="query",
        messages=[
            {"role": "system", "content": "system-marker"},
            {"role": "user", "content": "user-marker"},
        ],
        response_contract=_contract(),
    )

    envelope = llm_api.call_compiled_model_request(
        request,
        retry_count=2,
        retry_delay=0,
    )

    assert responses.calls == 3
    assert envelope["choices"][0]["message"]["content"] == "response-marker"
    assert traces
    assert traces[-1]["parameters"]["attempt"] == 3


def test_immutable_event_stream_failure_uses_safe_marker_and_never_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """真实迭代器在不可扩展异常后仍以脱敏 marker 禁止上层重发。"""

    class _ImmutableConnectionError(ConnectionError):
        def __setattr__(self, _name: str, _value: object) -> None:
            raise TypeError("immutable provider exception")

    class _EventThenFailingResponseStream:
        def __enter__(self) -> "_EventThenFailingResponseStream":
            return self

        def __exit__(self, *_args: object) -> bool:
            return False

        def __iter__(self):
            yield {"type": "response.created"}
            raise _ImmutableConnectionError(
                "secret-key https://provider.invalid raw-body"
            )

        def get_final_response(self) -> dict[str, object]:
            raise AssertionError("事件失败后不应读取最终响应")

    class _EventThenFailingResponses:
        def __init__(self) -> None:
            self.calls = 0

        def stream(self, **_kwargs: object) -> _EventThenFailingResponseStream:
            self.calls += 1
            return _EventThenFailingResponseStream()

    responses = _EventThenFailingResponses()
    monkeypatch.setattr(
        llm_api,
        "_build_codex_responses_client",
        lambda _api_key, _base_url: _FakeCodexClient(responses),
    )
    _reset_codex_client(monkeypatch)
    traces: list[dict[str, object]] = []
    monkeypatch.setattr(llm_api, "trace_llm_call", lambda **kwargs: traces.append(kwargs))
    request = compile_model_request(
        operation="hypothesis_generation",
        stage="query",
        messages=[
            {"role": "system", "content": "system-marker"},
            {"role": "user", "content": "user-marker"},
        ],
        response_contract=_contract(),
    )

    with pytest.raises(RuntimeError, match="Codex Responses stream failed") as raised:
        llm_api.call_compiled_model_request(
            request,
            retry_count=4,
            retry_delay=0,
        )

    assert responses.calls == 1
    assert getattr(raised.value, "received_event") is True
    assert getattr(raised.value, "retryable") is False
    assert raised.value.__context__ is None
    assert "secret-key" not in repr(raised.value)
    assert "provider.invalid" not in repr(raised.value)
    assert "raw-body" not in repr(raised.value)
    assert traces
    assert traces[-1]["parameters"]["attempt"] == 1
    assert all("secret-key" not in repr(trace) for trace in traces)
    assert all("provider.invalid" not in repr(trace) for trace in traces)
    assert all("raw-body" not in repr(trace) for trace in traces)


def test_sol_provider_never_uses_arche_chem_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sol 请求使用独立 client，即使 ARCHE_CHEM client 被替换也不会触发。"""

    responses = _FakeResponses()
    monkeypatch.setattr(
        llm_api,
        "_build_codex_responses_client",
        lambda _api_key, _base_url: _FakeCodexClient(responses),
    )
    _reset_codex_client(monkeypatch)

    class _ForbiddenChemClient:
        class _Chat:
            class _Completions:
                def create(self, **_kwargs: object) -> object:
                    raise AssertionError("Sol 不应使用 ARCHE_CHEM chat client")

            completions = _Completions()

        chat = _Chat()

    monkeypatch.setattr(llm_api, "client", _ForbiddenChemClient())
    request = compile_model_request(
        operation="hypothesis_generation",
        stage="hypothesis",
        messages=[
            {"role": "system", "content": "system-marker"},
            {"role": "user", "content": "user-marker"},
        ],
        response_contract=_contract(),
    )

    envelope = llm_api.call_compiled_model_request(request, retry_count=0)
    assert envelope["choices"][0]["message"]["content"] == "response-marker"


def test_unreceived_sol_stream_allows_outer_operation_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """未收事件的瞬时流失败由 ModelOperation 外层重试。"""

    class _TransientResponseStream:
        def __init__(self, attempt_index: int) -> None:
            self.attempt_index = attempt_index

        def __enter__(self) -> "_TransientResponseStream":
            return self

        def __exit__(self, *_args: object) -> bool:
            return False

        def __iter__(self):
            return iter(())

        def get_final_response(self) -> dict[str, object]:
            if self.attempt_index < 3:
                raise ConnectionError("transient provider failure")
            return {
                "id": "sol-response-marker",
                "model": "sol-model-marker",
                "status": "completed",
                "output_text": "response-marker",
            }

    class _TransientResponses:
        def __init__(self) -> None:
            self.calls = 0

        def stream(self, **_kwargs: object) -> _TransientResponseStream:
            self.calls += 1
            return _TransientResponseStream(self.calls)

    responses = _TransientResponses()
    monkeypatch.setattr(
        llm_api,
        "_build_codex_responses_client",
        lambda _api_key, _base_url: _FakeCodexClient(responses),
    )
    _reset_codex_client(monkeypatch)
    request = compile_model_request(
        operation="hypothesis_generation",
        stage="query",
        messages=[
            {"role": "system", "content": "system-marker"},
            {"role": "user", "content": "user-marker"},
        ],
        response_contract=_contract(),
    )
    runner = model_operation.ModelOperationRunner(
        task_root=tmp_path,
        operation_instance_id="sol-stream-outer-retry",
        operation="hypothesis_generation",
        base_input={"scientific_question": "验证未收流重试边界。"},
        stages=[
            model_operation.ModelStage(
                stage="query",
                stage_kind="model",
                validation_kind="query_delimited_text",
                query_index=1,
                schema=None,
                domain_validator_id=None,
                expected_count=1,
            )
        ],
        provider=lambda _request: llm_api.call_compiled_model_request(
            request,
            retry_count=5,
            retry_delay=0,
        ),
        retry_count=2,
        retry_delay=0,
        message_builder=lambda *_args, **_kwargs: request["messages"],
    )

    transported, response = runner._call_provider(request)

    assert transported is True
    assert response["choices"][0]["message"]["content"] == "response-marker"
    assert responses.calls == 3


def test_received_sol_stream_event_suppresses_operation_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Responses 已收到服务端事件后，ModelOperation 不重复提交同一请求。"""

    class _FailingResponseStream:
        received_body = True

        def __enter__(self) -> "_FailingResponseStream":
            return self

        def __exit__(self, *_args: object) -> bool:
            return False

        def get_final_response(self) -> dict[str, object]:
            raise ConnectionError("secret-key https://provider.invalid raw-body")

    class _FailingResponses:
        def __init__(self) -> None:
            self.calls = 0

        def stream(self, **_kwargs: object) -> _FailingResponseStream:
            self.calls += 1
            return _FailingResponseStream()

    responses = _FailingResponses()
    monkeypatch.setattr(
        llm_api,
        "_build_codex_responses_client",
        lambda _api_key, _base_url: _FakeCodexClient(responses),
    )
    _reset_codex_client(monkeypatch)
    traces: list[dict[str, object]] = []
    monkeypatch.setattr(llm_api, "trace_llm_call", lambda **kwargs: traces.append(kwargs))
    request = compile_model_request(
        operation="hypothesis_generation",
        stage="query",
        messages=[
            {"role": "system", "content": "system-marker"},
            {"role": "user", "content": "user-marker"},
        ],
        response_contract=_contract(),
    )
    runner = model_operation.ModelOperationRunner(
        task_root=tmp_path,
        operation_instance_id="sol-stream-retry",
        operation="hypothesis_generation",
        base_input={"scientific_question": "验证流式失败边界。"},
        stages=[
            model_operation.ModelStage(
                stage="query",
                stage_kind="model",
                validation_kind="query_delimited_text",
                query_index=1,
                schema=None,
                domain_validator_id=None,
                expected_count=1,
            )
        ],
        provider=lambda _request: llm_api.call_compiled_model_request(
            request,
            retry_count=5,
            retry_delay=0,
        ),
        retry_count=5,
        retry_delay=0,
        message_builder=lambda *_args, **_kwargs: request["messages"],
    )

    transported, response = runner._call_provider(request)

    assert transported is False
    assert response is None
    assert responses.calls == 1
    assert traces
    assert all("secret-key" not in repr(trace) for trace in traces)
    assert all("provider.invalid" not in repr(trace) for trace in traces)
    assert all("raw-body" not in repr(trace) for trace in traces)
