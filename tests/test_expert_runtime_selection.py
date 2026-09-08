"""专家后端启动选择、独立 Flash 回退和恢复复用测试。"""

from __future__ import annotations

import json
from typing import Any

import pytest

from chemistry_multiagent.agents.expert_model_agent import (
    ExpertModelAgent,
    ExpertModelConfig,
)
from chemistry_multiagent.controllers.chemistry_multiagent_controller import (
    ResearchTaskApplication,
    _expert_role_backend_fingerprints,
    _build_cli_parser,
)
from chemistry_multiagent.memory.contracts import (
    ContractValidationError,
    validate_expert_backend_selection,
)
from chemistry_multiagent.utils.expert_model_provider import (
    EXPERT_RUNTIME_7B,
    EXPERT_RUNTIME_CLUSTER,
    EXPERT_RUNTIME_FALLBACK,
    EXPERT_RUNTIME_STARTUP_FAILED,
    ExpertProviderBundle,
    ExpertProviderRole,
    build_expert_provider_bundle,
    select_expert_runtime,
)


def _probe(model: str) -> dict[str, Any]:
    """构造启动探测所需的严格 JSON provider 响应。"""

    return {
        "model": model,
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": json.dumps({"ok": True})},
            }
        ],
    }


def _probe_response(
    model: str,
    content: str,
    *,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    """构造可控的启动探测响应，用于覆盖协议输出失败类别。"""

    return {
        "model": model,
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {"content": content},
            }
        ],
    }


def _role(
    role: str,
    provider: Any,
    *,
    configured: bool = True,
    model: str | None = None,
) -> ExpertProviderRole:
    """构造测试用公开角色，不包含任何认证正文。"""

    return ExpertProviderRole(
        role=role,
        model=(model or f"{role}-model") if configured else None,
        base_url=f"https://{role}.example/v1" if configured else None,
        api_key_env=f"{role.upper()}_API_KEY",
        api_key_configured=configured,
        provider=provider if configured else None,
        config_fingerprint={
            "expert": "a",
            "cluster": "d",
            "fallback": "c",
            "general": "b",
        }[role]
        * 64,
    )


def _bundle(
    expert: Any,
    general: Any,
    fallback: Any | None = None,
    cluster: Any | None = None,
) -> ExpertProviderBundle:
    """构造一个启用且只由启动阶段选择路由的 provider bundle。"""

    if fallback is None and general is not None:

        def fallback_provider(messages: Any, **kwargs: Any) -> Any:
            value = general(messages, **kwargs)
            if isinstance(value, dict) and isinstance(value.get("model"), str):
                value = dict(value)
                value["model"] = "fallback-model"
            return value

        fallback = fallback_provider

    return ExpertProviderBundle(
        enabled=True,
        expert=_role("expert", expert, configured=expert is not None),
        general=_role("general", general, configured=general is not None),
        fallback=_role("fallback", fallback, configured=fallback is not None),
        cluster=(
            _role("cluster", cluster, configured=cluster is not None)
            if cluster is not None
            else None
        ),
    )


def test_cli_defaults_to_enabled_expert_with_explicit_disable_switch() -> None:
    """默认启用专家，关闭必须使用显式开关。"""

    args = _build_cli_parser().parse_args(
        ["create-standard-research-task", "--question", "Q"]
    )
    assert args.disable_expert_model is False
    disabled = _build_cli_parser().parse_args(
        [
            "create-standard-research-task",
            "--question",
            "Q",
            "--disable-expert-model",
        ]
    )
    assert disabled.disable_expert_model is True


def test_cli_accepts_all_fixed_reproduction_profiles() -> None:
    """CLI 允许三个固定复现 profile，并拒绝未登记标识。"""

    for profile in (
        "case1_fixed_reproduction",
        "case2_fixed_reproduction",
        "case3_fixed_reproduction",
    ):
        args = _build_cli_parser().parse_args(
            [
                "create-standard-research-task",
                "--question",
                "Q",
                "--case-profile",
                profile,
            ]
        )
        assert args.case_profile == profile

    with pytest.raises(SystemExit):
        _build_cli_parser().parse_args(
            ["--question", "Q", "--case-profile", "case4_fixed_reproduction"]
        )


def test_disabled_selection_uses_stable_empty_provider_schema() -> None:
    """关闭专家时仍保存固定 provider 字段，并能通过持久化合同。"""

    runtime = select_expert_runtime(build_expert_provider_bundle(enabled=False))
    selection = runtime.runtime_selection
    assert selection is not None
    frozen = validate_expert_backend_selection(selection.to_dict())
    assert frozen["status"] == "disabled"
    assert frozen["provider"] == runtime.expert.public_metadata()


def test_missing_expert_freezes_expert_role_to_flash_fallback() -> None:
    """专家未配置时探测 Flash 回退和 Pro 通用后端，并冻结两者。"""

    calls: list[str] = []
    probe_kwargs: list[dict[str, Any]] = []

    def general(messages: Any, **kwargs: Any) -> Any:
        calls.append(messages[0]["content"])
        probe_kwargs.append(kwargs)
        return _probe("general-model")

    runtime = select_expert_runtime(_bundle(None, general))
    selection = runtime.runtime_selection
    assert selection is not None
    assert selection.backend == EXPERT_RUNTIME_FALLBACK
    assert selection.provider_role == "fallback"
    assert selection.probe["expert"]["reason_code"] == "provider_unconfigured"
    assert selection.probe["fallback"]["reason_code"] == "ok"
    assert selection.probe["general"]["reason_code"] == "ok"
    frozen = validate_expert_backend_selection(selection.to_dict())
    assert frozen["backend"] == EXPERT_RUNTIME_FALLBACK
    assert len(calls) == 2
    assert probe_kwargs and all("max_tokens" not in item for item in probe_kwargs)
    assert "tokenizer" not in json.dumps(frozen)
    assert "profile" not in json.dumps(frozen)


def test_cluster_expert_is_selected_after_local_probe_failure() -> None:
    """本地探测失败后才尝试集群，并在集群成功时冻结集群后端。"""

    calls: list[str] = []

    def local(_: Any, **__: Any) -> Any:
        calls.append("local")
        raise ConnectionError("local unavailable")

    def cluster(_: Any, **__: Any) -> Any:
        calls.append("cluster")
        return _probe("cluster-model")

    def general(_: Any, **__: Any) -> Any:
        calls.append("general")
        return _probe("general-model")

    def fallback(_: Any, **__: Any) -> Any:
        raise AssertionError("集群成功后不应探测 Flash")

    runtime = select_expert_runtime(_bundle(local, general, fallback, cluster))
    selection = runtime.runtime_selection
    assert selection is not None
    assert selection.backend == EXPERT_RUNTIME_CLUSTER
    assert selection.provider_role == "cluster"
    assert calls == ["local", "cluster", "general"]
    assert selection.probe["expert"]["reason_code"] == "provider_transport_error"
    assert selection.probe["cluster"]["reason_code"] == "ok"
    frozen = validate_expert_backend_selection(selection.to_dict())
    assert frozen["provider"] == runtime.cluster.public_metadata()


def test_cluster_failure_is_recorded_before_flash_fallback() -> None:
    """集群协议失败后才探测 Flash，且冻结原因保留集群失败类别。"""

    calls: list[str] = []

    def cluster(_: Any, **__: Any) -> Any:
        calls.append("cluster")
        return {"model": "cluster-model", "choices": []}

    def fallback(_: Any, **__: Any) -> Any:
        calls.append("fallback")
        return _probe("fallback-model")

    def general(_: Any, **__: Any) -> Any:
        calls.append("general")
        return _probe("general-model")

    runtime = select_expert_runtime(
        _bundle(None, general, fallback, cluster)
    )
    selection = runtime.runtime_selection
    assert selection is not None
    assert selection.backend == EXPERT_RUNTIME_FALLBACK
    assert calls == ["cluster", "fallback", "general"]
    assert selection.probe["cluster"]["reason_code"] == "provider_protocol_error"
    assert selection.fallback_reason == "provider_protocol_error"
    validate_expert_backend_selection(selection.to_dict())


def test_selected_7b_probes_general_provider_before_freezing() -> None:
    """7B 成功时仍先确认通用 provider，随后才冻结选择。"""

    calls: list[str] = []

    def expert(messages: Any, **_: Any) -> Any:
        calls.append("expert")
        return _probe("expert-model")

    def general(_: Any, **__: Any) -> Any:
        calls.append("general")
        return _probe("general-model")

    runtime = select_expert_runtime(_bundle(expert, general))
    assert runtime.runtime_selection is not None
    assert runtime.runtime_selection.backend == EXPERT_RUNTIME_7B
    assert calls == ["expert", "general"]
    assert runtime.runtime_selection.probe["general"]["reason_code"] == "ok"
    assert _expert_role_backend_fingerprints(runtime) == {
        "expert": "a" * 64,
        "general": "b" * 64,
    }


@pytest.mark.parametrize(
    "first_response",
    [
        _probe_response("expert-model", "{invalid-json"),
        _probe_response("expert-model", json.dumps({"ok": False})),
        _probe_response(
            "expert-model",
            json.dumps({"ok": True}),
            finish_reason="length",
        ),
    ],
    ids=["invalid-json", "schema-failure", "truncated"],
)
def test_startup_probe_retries_transient_protocol_output_once(
    first_response: dict[str, Any],
) -> None:
    """一次协议输出失败后成功时，探针有限重试并冻结成功事实。"""

    responses = [first_response, _probe("expert-model")]
    calls: list[dict[str, Any]] = []

    def expert(messages: Any, **kwargs: Any) -> Any:
        calls.append({"messages": messages, "kwargs": kwargs})
        return responses.pop(0)

    runtime = select_expert_runtime(
        _bundle(expert, lambda *_args, **_kwargs: _probe("general-model"))
    )
    selection = runtime.runtime_selection
    assert selection is not None
    assert selection.backend == EXPERT_RUNTIME_7B
    assert len(calls) == 2
    assert selection.probe["expert"]["reason_code"] == "ok"
    assert selection.probe["expert"]["strict_json"] is True


@pytest.mark.parametrize(
    "failed_response",
    [
        _probe_response("general-model", "{invalid-json"),
        _probe_response("general-model", json.dumps({"ok": False})),
        _probe_response(
            "general-model",
            json.dumps({"ok": True}),
            finish_reason="length",
        ),
    ],
    ids=["invalid-json", "schema-failure", "truncated"],
)
def test_continuous_protocol_output_failure_remains_startup_failed(
    failed_response: dict[str, Any],
) -> None:
    """连续协议输出失败达到上限时仍以失败关闭，不冻结可用后端。"""

    general_calls: list[dict[str, Any]] = []

    def general(messages: Any, **kwargs: Any) -> Any:
        general_calls.append({"messages": messages, "kwargs": kwargs})
        return failed_response

    runtime = select_expert_runtime(
        _bundle(
            lambda *_args, **_kwargs: _probe("expert-model"),
            general,
        )
    )
    selection = runtime.runtime_selection
    assert selection is not None
    assert selection.status == EXPERT_RUNTIME_STARTUP_FAILED
    assert selection.failure_role == "general"
    assert len(general_calls) == 2
    assert selection.probe["general"]["connectivity"] is True
    assert selection.probe["general"]["authentication"] is True
    assert selection.probe["general"]["strict_json"] is False
    assert selection.probe["general"]["reason_code"] in {
        "provider_invalid_json",
        "provider_model_rejected",
    }


@pytest.mark.parametrize(
    ("failure_kind", "expected_reason"),
    [
        ("authentication", "provider_authentication_error"),
        ("connection", "provider_transport_error"),
        ("model_identity", "provider_probe_model_identity_mismatch"),
    ],
    ids=["authentication", "connection", "model-identity"],
)
def test_non_transient_probe_failures_are_not_retried(
    failure_kind: str,
    expected_reason: str,
) -> None:
    """认证、连接和模型身份失败不补发探针，也不记录异常正文。"""

    calls: list[dict[str, Any]] = []

    class ProbeAuthenticationError(Exception):
        """测试用认证异常。"""

    def expert(messages: Any, **kwargs: Any) -> Any:
        calls.append({"messages": messages, "kwargs": kwargs})
        if failure_kind == "authentication":
            raise ProbeAuthenticationError("sensitive-auth-detail")
        if failure_kind == "connection":
            raise ConnectionError("sensitive-connection-detail")
        return _probe("unrelated-model")

    runtime = select_expert_runtime(_bundle(expert, None))
    selection = runtime.runtime_selection
    assert selection is not None
    assert selection.status == EXPERT_RUNTIME_STARTUP_FAILED
    assert len(calls) == 1
    assert selection.probe["expert"]["reason_code"] == expected_reason
    frozen = json.dumps(selection.to_dict(), ensure_ascii=False)
    assert "sensitive-auth-detail" not in frozen
    assert "sensitive-connection-detail" not in frozen


def test_selected_7b_is_rejected_when_general_probe_fails() -> None:
    """7B 可用但通用后端不可用时不得进入科学阶段。"""

    def expert(_: Any, **__: Any) -> Any:
        return _probe("expert-model")

    def general(_: Any, **__: Any) -> Any:
        return {"model": "general-model", "choices": []}

    runtime = select_expert_runtime(_bundle(expert, general))
    selection = runtime.runtime_selection
    assert selection is not None
    assert selection.status == EXPERT_RUNTIME_STARTUP_FAILED
    assert selection.probe["expert"]["reason_code"] == "ok"
    assert selection.probe["general"]["reason_code"] == "provider_protocol_error"


def test_fallback_role_fingerprints_follow_flash_and_providers() -> None:
    """专家回退收据使用 Flash 指纹，General opinion/Judge 使用 Pro 指纹。"""

    def general(_: Any, **__: Any) -> Any:
        return _probe("general-model")

    runtime = select_expert_runtime(_bundle(None, general))
    assert _expert_role_backend_fingerprints(runtime) == {
        "expert": "c" * 64,
        "general": "b" * 64,
    }


def test_namespaced_served_model_matches_configured_basename() -> None:
    """代理返回命名模型的 basename 时仍保留两种模型事实并通过校验。"""

    configured_model = "tenant/DeepSeek-V3"

    def expert(_: Any, **__: Any) -> Any:
        return _probe("deepseek-v3")

    runtime = select_expert_runtime(
        ExpertProviderBundle(
            enabled=True,
            expert=_role("expert", expert, model=configured_model),
            general=_role("general", lambda *_args, **_kwargs: _probe("general-model")),
            fallback=_role("fallback", lambda *_args, **_kwargs: _probe("fallback-model")),
        )
    )
    selection = runtime.runtime_selection
    assert selection is not None
    assert selection.backend == EXPERT_RUNTIME_7B
    assert selection.served_model == "deepseek-v3"
    assert selection.provider_metadata["model"] == configured_model
    validate_expert_backend_selection(selection.to_dict())


def test_unrelated_served_model_is_rejected() -> None:
    """无关模型标识不能被 basename 规则误判为成功。"""

    configured_model = "tenant/DeepSeek-V3"

    def expert(_: Any, **__: Any) -> Any:
        return _probe("other-model")

    runtime = select_expert_runtime(
        ExpertProviderBundle(
            enabled=True,
            expert=_role("expert", expert, model=configured_model),
            general=_role("general", None, configured=False),
            fallback=_role("fallback", None, configured=False),
        )
    )
    selection = runtime.runtime_selection
    assert selection is not None
    assert selection.status == EXPERT_RUNTIME_STARTUP_FAILED
    assert selection.probe["expert"]["reason_code"] == "provider_probe_model_identity_mismatch"
    validate_expert_backend_selection(selection.to_dict())


def test_general_probe_failure_freezes_startup_failed() -> None:
    """专家候选失败、Flash 可用而 Pro 失败时，启动事实标记 general。"""

    def fallback(_: Any, **__: Any) -> Any:
        return _probe("fallback-model")

    def general(_: Any, **__: Any) -> Any:
        return {"model": "general-model", "choices": []}

    runtime = select_expert_runtime(_bundle(None, general, fallback))
    selection = runtime.runtime_selection
    assert selection is not None
    assert selection.status == EXPERT_RUNTIME_STARTUP_FAILED
    assert selection.backend == EXPERT_RUNTIME_STARTUP_FAILED
    assert selection.provider is None
    assert selection.probe["expert"]["reason_code"] == "provider_unconfigured"
    assert selection.probe["general"]["reason_code"] == "provider_protocol_error"
    assert selection.failure_role == "general"
    assert selection.fallback_reason == "provider_protocol_error"
    validate_expert_backend_selection(selection.to_dict())


def test_general_probe_failure_is_valid_when_expert_probe_succeeds() -> None:
    """7B 可用但通用意见/Judge 不可用时，启动失败事实仍可持久化。"""

    def expert(_: Any, **__: Any) -> Any:
        return _probe("expert-model")

    def general(_: Any, **__: Any) -> Any:
        return {"model": "general-model", "choices": []}

    runtime = select_expert_runtime(_bundle(expert, general))
    selection = runtime.runtime_selection
    assert selection is not None
    assert selection.status == EXPERT_RUNTIME_STARTUP_FAILED
    assert selection.probe["expert"]["reason_code"] == "ok"
    assert selection.probe["general"]["reason_code"] == "provider_protocol_error"
    assert selection.fallback_reason == "provider_protocol_error"
    validate_expert_backend_selection(selection.to_dict())


def test_fallback_agent_does_not_retry_unselected_expert() -> None:
    """Flash 回退选择后，专家职责不重新调用未选中的本地或集群专家。"""

    calls: list[str] = []

    def expert(_: Any, **__: Any) -> Any:
        calls.append("expert")
        raise AssertionError("运行中不应重新调用未选中的 expert")

    def fallback(messages: Any, **__: Any) -> Any:
        calls.append("fallback")
        if messages[0]["content"].startswith("Return exactly"):
            return _probe("fallback-model")
        return {"status": "approved", "reason": "ok"}

    def general(_: Any, **__: Any) -> Any:
        calls.append("general")
        return _probe("general-model")

    selected = select_expert_runtime(_bundle(expert, general, fallback))
    assert selected.runtime_selection is not None
    assert selected.runtime_selection.backend == EXPERT_RUNTIME_FALLBACK
    agent = ExpertModelAgent(
        ExpertModelConfig(
            enabled=True,
            expert_provider=selected.runtime_selection.provider,
            general_provider=general,
            runtime_selection=selected.runtime_selection,
        )
    )
    result = agent._call("Return JSON", "gaussian_step_review", {"value": 1}, role="expert")
    assert result == {"status": "approved", "reason": "ok"}
    assert calls == ["expert", "fallback", "general", "fallback"]


def test_cluster_selection_rejects_failed_general_probe() -> None:
    """冻结的集群选择必须同时证明通用 Pro provider 可用。"""

    runtime = select_expert_runtime(
        _bundle(
            None,
            lambda *_args, **_kwargs: _probe("general-model"),
            lambda *_args, **_kwargs: _probe("fallback-model"),
            lambda *_args, **_kwargs: _probe("cluster-model"),
        )
    )
    selection = runtime.runtime_selection
    assert selection is not None
    frozen = selection.to_dict()
    frozen["probe"]["general"]["reason_code"] = "provider_transport_error"
    frozen["probe"]["general"]["connectivity"] = False
    frozen["probe"]["general"]["authentication"] = False
    frozen["probe"]["general"]["strict_json"] = False
    with pytest.raises(ContractValidationError, match="cluster selected"):
        validate_expert_backend_selection(frozen)


def test_contradictory_selection_is_rejected() -> None:
    """失败的通用后端探测不能同时声明成功 reason_code。"""

    runtime = select_expert_runtime(_bundle(lambda *_args, **_kwargs: _probe("expert-model"), None))
    selection = runtime.runtime_selection
    assert selection is not None
    assert selection.status == EXPERT_RUNTIME_STARTUP_FAILED
    frozen = selection.to_dict()
    frozen["probe"]["general"]["reason_code"] = "ok"
    with pytest.raises(ContractValidationError):
        validate_expert_backend_selection(frozen)


def test_selected_runtime_rejects_mismatched_provider_role() -> None:
    """运行时对象不能把 Flash 后端错误标记为 expert 角色。"""

    runtime = select_expert_runtime(
        _bundle(None, lambda *_args, **_kwargs: _probe("general-model"))
    )
    selection = runtime.runtime_selection
    assert selection is not None
    values = dict(selection.__dict__)
    values["provider_role"] = "expert"
    with pytest.raises(ValueError, match="provider_role 与 backend 不一致"):
        type(selection)(**values)
