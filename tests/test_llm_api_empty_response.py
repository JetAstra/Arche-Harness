import inspect
from types import SimpleNamespace

import pytest

from chemistry_multiagent.utils import llm_api


def test_general_model_does_not_have_a_default_output_budget():
    """通用模型入口默认不设置本地输出上限。"""

    parameter = inspect.signature(llm_api.call_deepseek_api).parameters[
        "max_tokens"
    ]

    assert parameter.default is None


def test_model_timeouts_default_to_300_seconds():
    """通用模型保持此前确认过的完整生成等待时间。"""

    assert (
        inspect.signature(llm_api.call_deepseek_api)
        .parameters["timeout"]
        .default
        == 300
    )


@pytest.mark.parametrize(
    "entrypoint",
    [llm_api.call_deepseek_api, llm_api.call_deepseek_api_envelope],
)
def test_model_api_rejects_boolean_legacy_attempt_count(entrypoint):
    """布尔值不能因继承 int 而绕过旧总尝试次数参数校验。"""

    with pytest.raises(ValueError, match="max_retries"):
        entrypoint(
            [{"role": "user", "content": "测试"}],
            max_retries=True,
        )


@pytest.mark.parametrize(
    "entrypoint",
    [llm_api.call_deepseek_api, llm_api.call_deepseek_api_envelope],
)
def test_legacy_max_retries_remains_total_attempt_count(entrypoint, monkeypatch):
    """旧参数 max_retries=1 仍只允许一次真实模型请求。"""

    calls = []

    def fail_once(**kwargs):
        calls.append(kwargs)
        raise TimeoutError("temporary timeout")

    monkeypatch.setattr(llm_api.client.chat.completions, "create", fail_once)
    monkeypatch.setattr(llm_api, "trace_llm_call", None)

    with pytest.raises(TimeoutError, match="temporary timeout"):
        entrypoint(
            [{"role": "user", "content": "测试"}],
            max_retries=1,
            retry_delay=0,
        )

    assert len(calls) == 1


@pytest.mark.parametrize(
    "entrypoint",
    [llm_api.call_deepseek_api, llm_api.call_deepseek_api_envelope],
)
def test_legacy_max_retries_rejects_non_positive_values(entrypoint):
    """旧总尝试次数不能为零或负数。"""

    with pytest.raises(ValueError, match="max_retries"):
        entrypoint(
            [{"role": "user", "content": "测试"}],
            max_retries=0,
        )


@pytest.mark.parametrize(
    "entrypoint",
    [llm_api.call_deepseek_api, llm_api.call_deepseek_api_envelope],
)
def test_legacy_retry_arguments_keep_their_positional_slots(entrypoint, monkeypatch):
    """旧调用方的位置参数仍按 max_retries、retry_delay 解释。"""

    calls = []
    monkeypatch.setattr(
        llm_api.client.chat.completions,
        "create",
        lambda **kwargs: calls.append(kwargs) or _response("OK"),
    )
    monkeypatch.setattr(llm_api, "trace_llm_call", None)

    entrypoint(
        [{"role": "user", "content": "测试"}],
        None,
        0.7,
        None,
        300,
        1,
        0,
    )

    assert len(calls) == 1


def _response(content, reasoning_content="", finish_reason="stop"):
    message = SimpleNamespace(
        content=content,
        reasoning_content=reasoning_content,
    )
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    usage = SimpleNamespace(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
    )
    return SimpleNamespace(choices=[choice], usage=usage)


@pytest.mark.parametrize(
    "entrypoint",
    [llm_api.call_deepseek_api, llm_api.call_deepseek_api_envelope],
)
def test_direct_deepseek_entries_use_pro_reasoning_only(entrypoint, monkeypatch):
    """两条兼容 direct API 入口都只给明确的 Pro 模型发送 max。"""

    calls = []
    monkeypatch.setattr(
        llm_api.client.chat.completions,
        "create",
        lambda **kwargs: calls.append(kwargs) or _response("OK"),
    )
    monkeypatch.setattr(llm_api, "trace_llm_call", None)

    entrypoint(
        [{"role": "user", "content": "测试"}],
        model="deepseek-v4-pro-0813",
        max_retries=1,
        retry_delay=0,
    )

    assert calls[0]["reasoning_effort"] == "max"
    assert "response_format" not in calls[0]


def test_general_api_defaults_to_pro_without_environment(monkeypatch):
    monkeypatch.delenv("ARCHE_CHEM_MODEL_NAME", raising=False)
    response = _response("OK")
    calls = []

    def fake_create(**kwargs):
        calls.append(kwargs)
        return response

    monkeypatch.setattr(llm_api.client.chat.completions, "create", fake_create)
    monkeypatch.setattr(llm_api, "trace_llm_call", None)

    assert llm_api.call_deepseek_api(
        [{"role": "user", "content": "测试"}],
        max_retries=1,
        retry_delay=0,
    ) == "OK"
    assert calls[0]["model"] == "deepseek-v4-pro"
    assert calls[0]["reasoning_effort"] == "max"
    assert "max_tokens" not in calls[0]


def test_general_api_uses_configured_pro_reasoning_effort(monkeypatch):
    """direct Pro API 入口发送环境中配置的 high，并在 trace 中记录同值。"""

    monkeypatch.setenv("DEEPSEEK_V4_PRO_REASONING_EFFORT", "high")
    response = _response("OK")
    calls = []
    traces = []

    def fake_create(**kwargs):
        calls.append(kwargs)
        return response

    monkeypatch.setattr(llm_api.client.chat.completions, "create", fake_create)
    monkeypatch.setattr(llm_api, "trace_llm_call", lambda **kwargs: traces.append(kwargs))

    assert llm_api.call_deepseek_api(
        [{"role": "user", "content": "测试"}],
        max_retries=1,
        retry_delay=0,
    ) == "OK"
    assert calls[0]["reasoning_effort"] == "high"
    assert traces[-1]["parameters"]["reasoning_effort"] == "high"


def test_general_api_envelope_uses_configured_pro_reasoning_effort(monkeypatch):
    """envelope direct API 入口也发送同一份 Pro 推理强度。"""

    monkeypatch.setenv("DEEPSEEK_V4_PRO_REASONING_EFFORT", "high")
    calls = []
    monkeypatch.setattr(
        llm_api.client.chat.completions,
        "create",
        lambda **kwargs: calls.append(kwargs) or _response("OK"),
    )
    monkeypatch.setattr(llm_api, "trace_llm_call", None)

    llm_api.call_deepseek_api_envelope(
        [{"role": "user", "content": "测试"}],
        max_retries=1,
        retry_delay=0,
    )

    assert calls[0]["reasoning_effort"] == "high"


def test_general_api_rejects_invalid_pro_reasoning_effort_before_request(monkeypatch):
    """direct Pro API 遇到非法推理强度时不能触发网络请求。"""

    monkeypatch.setenv("DEEPSEEK_V4_PRO_REASONING_EFFORT", "low")
    calls = []
    monkeypatch.setattr(
        llm_api.client.chat.completions,
        "create",
        lambda **kwargs: calls.append(kwargs) or _response("OK"),
    )
    monkeypatch.setattr(llm_api, "trace_llm_call", None)

    with pytest.raises(ValueError, match="DEEPSEEK_V4_PRO_REASONING_EFFORT"):
        llm_api.call_deepseek_api(
            [{"role": "user", "content": "测试"}],
            max_retries=1,
            retry_delay=0,
        )
    assert calls == []


def test_general_api_honors_explicit_environment_override(monkeypatch):
    monkeypatch.setenv("ARCHE_CHEM_MODEL_NAME", "configured-general")
    response = _response("OK")
    calls = []

    def fake_create(**kwargs):
        calls.append(kwargs)
        return response

    monkeypatch.setattr(llm_api.client.chat.completions, "create", fake_create)
    monkeypatch.setattr(llm_api, "trace_llm_call", None)

    llm_api.call_deepseek_api(
        [{"role": "user", "content": "测试"}],
        max_retries=1,
        retry_delay=0,
    )

    assert calls[0]["model"] == "configured-general"
    assert "reasoning_effort" not in calls[0]
    assert "max_tokens" not in calls[0]


def test_general_api_keeps_explicit_output_budget_compatible(monkeypatch):
    """旧调用方显式给出正整数时仍可请求有界输出。"""

    response = _response("OK")
    calls = []

    def fake_create(**kwargs):
        calls.append(kwargs)
        return response

    monkeypatch.setattr(llm_api.client.chat.completions, "create", fake_create)
    monkeypatch.setattr(llm_api, "trace_llm_call", None)

    llm_api.call_deepseek_api(
        [{"role": "user", "content": "测试"}],
        max_tokens=321,
        max_retries=1,
        retry_delay=0,
    )

    assert calls[0]["max_tokens"] == 321


def test_empty_content_is_not_retried(monkeypatch):
    """空正文属于内容问题，不能被传输层重试重复提交。"""

    calls = []

    def fake_create(**kwargs):
        calls.append(kwargs)
        return _response("", reasoning_content="模型仍在推理")

    monkeypatch.setattr(llm_api.client.chat.completions, "create", fake_create)
    monkeypatch.setattr(llm_api, "trace_llm_call", None)

    with pytest.raises(RuntimeError, match="模型返回空内容"):
        llm_api.call_deepseek_api(
            [{"role": "user", "content": "测试"}],
            max_retries=2,
            retry_delay=0,
        )

    assert len(calls) == 1


def test_empty_content_is_traced_as_content_failure(monkeypatch):
    """provider 已回答但正文为空时，审计必须记录失败而非成功响应。"""

    traces = []
    monkeypatch.setattr(
        llm_api.client.chat.completions,
        "create",
        lambda **_kwargs: _response("", reasoning_content="模型仍在推理"),
    )
    monkeypatch.setattr(llm_api, "trace_llm_call", lambda **value: traces.append(value))

    with pytest.raises(RuntimeError, match="模型返回空内容"):
        llm_api.call_deepseek_api(
            [{"role": "user", "content": "测试"}],
            max_retries=2,
            retry_delay=0,
        )

    assert len(traces) == 1
    assert "模型返回空内容" in traces[0]["error"]
    assert "raw_response" not in traces[0]


def test_envelope_entrypoint_omits_default_output_budget(monkeypatch):
    """通用 envelope 入口未显式设预算时也不向 provider 传 max_tokens。"""

    calls = []

    def fake_create(**kwargs):
        calls.append(kwargs)
        return _response("OK")

    monkeypatch.setattr(llm_api.client.chat.completions, "create", fake_create)
    monkeypatch.setattr(llm_api, "trace_llm_call", None)

    llm_api.call_deepseek_api_envelope(
        [{"role": "user", "content": "测试"}],
        max_retries=1,
        retry_delay=0,
    )

    assert "max_tokens" not in calls[0]


def test_repeated_empty_content_raises_clear_error(monkeypatch):
    monkeypatch.setattr(
        llm_api.client.chat.completions,
        "create",
        lambda **kwargs: _response("", reasoning_content="只有推理内容"),
    )
    monkeypatch.setattr(llm_api, "trace_llm_call", None)

    with pytest.raises(RuntimeError, match="模型返回空内容"):
        llm_api.call_deepseek_api(
            [{"role": "user", "content": "测试"}],
            max_retries=2,
            retry_delay=0,
        )


def test_envelope_entrypoint_preserves_empty_choices_for_protocol_classifier(
    monkeypatch,
):
    response = SimpleNamespace(id="trace-empty", choices=[], usage=None)
    monkeypatch.setattr(
        llm_api.client.chat.completions,
        "create",
        lambda **kwargs: response,
    )
    monkeypatch.setattr(llm_api, "trace_llm_call", None)

    envelope = llm_api.call_deepseek_api_envelope(
        [{"role": "user", "content": "测试"}],
        max_retries=1,
        retry_delay=0,
    )

    assert envelope == {"id": "trace-empty", "choices": [], "usage": None}


def test_envelope_entrypoint_preserves_truncated_body_without_parsing(monkeypatch):
    response = _response(
        '说明文字 {"value":1}',
        reasoning_content="推理",
        finish_reason="length",
    )
    response.id = "trace-length"
    monkeypatch.setattr(
        llm_api.client.chat.completions,
        "create",
        lambda **kwargs: response,
    )
    monkeypatch.setattr(llm_api, "trace_llm_call", None)

    envelope = llm_api.call_deepseek_api_envelope(
        [{"role": "user", "content": "测试"}],
        max_retries=1,
        retry_delay=0,
    )

    assert envelope["choices"][0]["finish_reason"] == "length"
    assert envelope["choices"][0]["message"]["content"] == (
        '说明文字 {"value":1}'
    )


def test_envelope_entrypoint_returns_none_response_as_classifiable_evidence(
    monkeypatch,
):
    monkeypatch.setattr(
        llm_api.client.chat.completions,
        "create",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(llm_api, "trace_llm_call", None)

    envelope = llm_api.call_deepseek_api_envelope(
        [{"role": "user", "content": "测试"}],
        max_retries=1,
        retry_delay=0,
    )

    assert envelope is None
