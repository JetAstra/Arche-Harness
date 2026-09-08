from types import SimpleNamespace

import pytest

from minichem_toolbox.backends import gaussian_codegen


def _response(content="ok", status_code=200, finish_reason="stop"):
    return SimpleNamespace(
        status_code=status_code,
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason=finish_reason,
            )
        ],
    )


class _Completions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class _Client:
    def __init__(self, completions):
        self.chat = SimpleNamespace(completions=completions)


def test_gaussian_codegen_retries_only_transient_failures_with_fixed_delay(monkeypatch):
    completions = _Completions(
        [TimeoutError("temporary network timeout") for _ in range(5)]
        + [_response("#P B3LYP/6-31G(d) SP")]
    )
    client_options = {}
    client = _Client(completions)

    def fake_client(**kwargs):
        client_options.update(kwargs)
        return client

    monkeypatch.setattr(gaussian_codegen.openai, "OpenAI", fake_client)
    sleeps = []

    result = gaussian_codegen._call_author_model_api(
        [{"role": "user", "content": "generate a route"}],
        api_key="test-key",
        base_url="https://expert.invalid/v1",
        model="arche-chem-test",
        retry_sleeper=sleeps.append,
    )

    assert result.startswith("#P")
    assert len(completions.calls) == 6
    assert sleeps == [6.0] * 5
    assert client_options["max_retries"] == 0


def test_gaussian_codegen_reserves_output_budget_for_reasoning_models(monkeypatch):
    completions = _Completions([_response("done")])
    monkeypatch.setattr(
        gaussian_codegen.openai,
        "OpenAI",
        lambda **_kwargs: _Client(completions),
    )

    result = gaussian_codegen._call_author_model_api(
        [{"role": "user", "content": "repair a route"}],
        api_key="test-key",
        base_url="https://expert.invalid/v1",
        model="arche-chem-test",
    )

    assert result == "done"
    assert completions.calls[0]["max_tokens"] == 4096


@pytest.mark.parametrize("finish_reason", ["length", "max_tokens"])
def test_gaussian_codegen_retries_truncated_response_with_bounded_budget(
    monkeypatch, finish_reason
):
    completions = _Completions(
        [
            _response("partial response", finish_reason=finish_reason),
            _response("Corrected Gaussian keyword line: #P B3LYP/6-31G(d) SP"),
        ]
    )
    monkeypatch.setattr(
        gaussian_codegen.openai,
        "OpenAI",
        lambda **_kwargs: _Client(completions),
    )

    result = gaussian_codegen._call_author_model_api(
        [{"role": "user", "content": "repair a route"}],
        api_key="test-key",
        base_url="https://expert.invalid/v1",
        model="arche-chem-test",
    )

    assert result.startswith("Corrected Gaussian keyword line")
    assert [call["max_tokens"] for call in completions.calls] == [4096, 8192]


def test_gaussian_codegen_rejects_truncation_after_one_budget_upgrade(monkeypatch):
    completions = _Completions(
        [
            _response("partial response", finish_reason="length"),
            _response("still partial", finish_reason="max_tokens"),
        ]
    )
    monkeypatch.setattr(
        gaussian_codegen.openai,
        "OpenAI",
        lambda **_kwargs: _Client(completions),
    )

    with pytest.raises(
        gaussian_codegen._AuthorModelTruncatedError,
        match="预算升级后仍被截断",
    ):
        gaussian_codegen._call_author_model_api(
            [{"role": "user", "content": "repair a route"}],
            api_key="test-key",
            base_url="https://expert.invalid/v1",
            model="arche-chem-test",
        )

    assert [call["max_tokens"] for call in completions.calls] == [4096, 8192]


def test_gaussian_codegen_does_not_promote_raw_route_after_truncation(monkeypatch):
    route = "#P B3LYP/6-31G(d) SP"
    completions = _Completions(
        [
            _response(
                "Analysis: use a single point calculation.\n"
                "Gaussian computational task description: Calculate a single point energy."
            ),
            _response(route),
            _response(f"Gaussian_code: {route}"),
            _response("Analysis: incomplete fix", finish_reason="length"),
            _response("Analysis: still incomplete fix", finish_reason="length"),
        ]
    )
    monkeypatch.setattr(
        gaussian_codegen.openai,
        "OpenAI",
        lambda **_kwargs: _Client(completions),
    )
    monkeypatch.setenv("ARCHE_CHEM_API_KEY", "test-key")
    monkeypatch.setenv("ARCHE_CHEM_BASE_URL", "https://expert.invalid/v1")
    monkeypatch.setenv("ARCHE_CHEM_MODEL_NAME", "arche-chem-test")
    monkeypatch.setenv("ARCHE_CHEM_BACKEND", "openai_compatible")

    result = gaussian_codegen.execute(
        "generate_gaussian_code",
        "gaussian_codegen",
        {"inputs": {"user_input": "Calculate a single point energy."}},
    )

    assert result["status"] == "failed"
    assert result["error"]["code"] == "author_gaussian_codegen_failed"
    assert result["retryable"] is False
    assert result["result"]["raw_gaussian_code"] == route
    assert result["result"]["gaussian_code"] == ""
    assert "route_selection" not in result["result"]["metadata"]
    assert [call["max_tokens"] for call in completions.calls] == [
        4096,
        4096,
        4096,
        4096,
        8192,
    ]


def test_gaussian_codegen_retries_http_429_and_5xx(monkeypatch):
    completions = _Completions(
        [_response(status_code=429), _response(status_code=503), _response("done")]
    )
    monkeypatch.setattr(
        gaussian_codegen.openai,
        "OpenAI",
        lambda **_kwargs: _Client(completions),
    )
    sleeps = []

    result = gaussian_codegen._call_author_model_api(
        [{"role": "user", "content": "generate a route"}],
        api_key="test-key",
        base_url="https://expert.invalid/v1",
        model="arche-chem-test",
        retry_sleeper=sleeps.append,
    )

    assert result == "done"
    assert len(completions.calls) == 3
    assert sleeps == [6.0, 6.0]


def test_gaussian_codegen_does_not_retry_authentication_or_response_errors(monkeypatch):
    class UnauthorizedError(Exception):
        status_code = 401

    completions = _Completions([UnauthorizedError("invalid key")])
    monkeypatch.setattr(
        gaussian_codegen.openai,
        "OpenAI",
        lambda **_kwargs: _Client(completions),
    )

    with pytest.raises(UnauthorizedError, match="invalid key"):
        gaussian_codegen._call_author_model_api(
            [{"role": "user", "content": "generate a route"}],
            api_key="test-key",
            base_url="https://expert.invalid/v1",
            model="arche-chem-test",
            retry_sleeper=lambda _delay: pytest.fail("must not sleep"),
        )

    assert len(completions.calls) == 1


def test_gaussian_codegen_does_not_retry_empty_content(monkeypatch):
    completions = _Completions([_response("")])
    monkeypatch.setattr(
        gaussian_codegen.openai,
        "OpenAI",
        lambda **_kwargs: _Client(completions),
    )

    with pytest.raises(ValueError, match="内容为空"):
        gaussian_codegen._call_author_model_api(
            [{"role": "user", "content": "generate a route"}],
            api_key="test-key",
            base_url="https://expert.invalid/v1",
            model="arche-chem-test",
            retry_sleeper=lambda _delay: pytest.fail("must not sleep"),
        )

    assert len(completions.calls) == 1
