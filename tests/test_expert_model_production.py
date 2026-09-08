"""专家模型生产 provider、配置冻结和命令行接线测试。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from chemistry_multiagent.controllers import chemistry_multiagent_controller as controller_module
from chemistry_multiagent.controllers.chemistry_multiagent_controller import (
    ResearchTaskApplication,
)
from chemistry_multiagent.memory.contracts import (
    ContractValidationError,
    validate_model_configuration,
)
from chemistry_multiagent.memory import ResearchTaskWorkingMemory
from chemistry_multiagent.utils import expert_model_provider as provider_module


def test_bundle_uses_unified_model_for_expert_and_general(monkeypatch) -> None:
    """统一 ARCHE_CHEM 配置同时驱动专家和通用角色。"""

    # 测试的是“只配置通用 Pro”的角色选择；仓库本地 config.local.env 可能同时
    # 提供 7B、集群和专用 Flash 候选，必须在测试边界显式清除这些环境覆盖。
    for name in (
        "ARCHE_EXPERT_MODEL",
        "ARCHE_EXPERT_BASE_URL",
        "ARCHE_EXPERT_API_KEY",
        "ARCHE_EXPERT_API_KEY_ENV",
        "ARCHE_EXPERT_CLUSTER_MODEL",
        "ARCHE_EXPERT_CLUSTER_BASE_URL",
        "ARCHE_EXPERT_CLUSTER_INGRESS_AK",
        "ARCHE_EXPERT_CLUSTER_INGRESS_SK",
        "ARCHE_EXPERT_CLUSTER_PUBLIC_API_KEY",
        "ARCHE_EXPERT_GENERAL_MODEL",
        "ARCHE_EXPERT_GENERAL_BASE_URL",
        "ARCHE_EXPERT_GENERAL_API_KEY",
        "ARCHE_EXPERT_GENERAL_API_KEY_ENV",
        "ARCHE_EXPERT_FALLBACK_MODEL",
        "ARCHE_EXPERT_FALLBACK_BASE_URL",
        "ARCHE_EXPERT_FALLBACK_API_KEY",
        "ARCHE_EXPERT_FALLBACK_API_KEY_ENV",
        "MODEL_BASE_URL",
        "INGRESS_AK",
        "INGRESS_SK",
        "PUBLIC_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ARCHE_CHEM_MODEL_NAME", "general-model")
    monkeypatch.setenv("ARCHE_CHEM_BASE_URL", "https://general.example/v1")
    monkeypatch.setenv("ARCHE_CHEM_API_KEY", "general-secret")
    bundle = provider_module.build_expert_provider_bundle(enabled=True)

    assert bundle.status == "configured"
    assert bundle.expert_provider is not None
    assert bundle.general_provider is not None
    metadata = bundle.public_metadata()
    assert metadata["enabled"] is True
    assert metadata["expert"]["model"] == "general-model"
    assert metadata["expert"]["api_key_configured"] is True
    assert "secret" not in json.dumps(metadata, ensure_ascii=False)


def test_bundle_uses_one_unified_model_for_fallback_and_general(monkeypatch) -> None:
    """默认回退与通用角色共用统一 ARCHE_CHEM 配置。"""

    for name in (
        "ARCHE_CHEM_MODEL_NAME",
        "ARCHE_CHEM_BASE_URL",
        "ARCHE_CHEM_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    bundle = provider_module.build_expert_provider_bundle(
        enabled=True,
        fallback_api_key="shared-secret",
        general_api_key="shared-secret",
    )

    assert bundle.fallback.model == "deepseek-v4-pro"
    assert bundle.general.model == "deepseek-v4-pro"
    assert bundle.fallback.api_key_env == "ARCHE_CHEM_API_KEY"
    assert bundle.general.api_key_env == "ARCHE_CHEM_API_KEY"


def test_expert_general_role_normalizes_legacy_general_alias(monkeypatch) -> None:
    """专家 debate 的历史通用别名也应解析到统一 Pro 默认值。"""

    monkeypatch.setenv("ARCHE_CHEM_MODEL_NAME", "deepseek-chat")
    monkeypatch.delenv("ARCHE_EXPERT_GENERAL_MODEL", raising=False)
    bundle = provider_module.build_expert_provider_bundle(
        enabled=True,
        fallback_api_key="shared-secret",
        general_api_key="shared-secret",
    )

    assert bundle.general.model == "deepseek-v4-pro"


def test_fallback_model_uses_explicit_python_override(monkeypatch) -> None:
    """fallback_model 可显式覆盖统一模型，但角色环境变量不会改路由。"""

    monkeypatch.setenv("ARCHE_EXPERT_FALLBACK_MODEL", "env-flash")
    monkeypatch.setenv("ARCHE_EXPERT_FALLBACK_API_KEY", "env-secret")
    env_bundle = provider_module.build_expert_provider_bundle(enabled=True)
    assert env_bundle.fallback.model == "deepseek-v4-pro"
    assert env_bundle.fallback.api_key_env == "ARCHE_CHEM_API_KEY"

    monkeypatch.delenv("ARCHE_EXPERT_FALLBACK_API_KEY", raising=False)
    explicit_bundle = provider_module.build_expert_provider_bundle(
        enabled=True,
        fallback_model="explicit-flash",
        fallback_api_key="explicit-secret",
    )
    assert explicit_bundle.fallback.model == "explicit-flash"
    assert explicit_bundle.fallback.api_key_env == "ARCHE_CHEM_API_KEY"


def test_disabled_bundle_is_stable_across_environment_changes(monkeypatch) -> None:
    """关闭专家时冻结的 disabled bundle 不读取当前专家环境。"""

    monkeypatch.setenv("ARCHE_CHEM_MODEL_NAME", "changed-general")
    monkeypatch.setenv("ARCHE_CHEM_API_KEY", "changed-secret")
    monkeypatch.setenv("ARCHE_EXPERT_MODEL", "changed-expert")
    bundle = provider_module.build_expert_provider_bundle(enabled=False)
    assert bundle.public_metadata() == {
        "enabled": False,
        "status": "disabled",
        "expert": {
            "model": None,
            "base_url": None,
            "api_key_env": "",
            "api_key_configured": False,
            "config_fingerprint": "0" * 64,
        },
        "fallback": {
            "model": None,
            "base_url": None,
            "api_key_env": "",
            "api_key_configured": False,
            "config_fingerprint": "0" * 64,
        },
        "general": {
            "model": None,
            "base_url": None,
            "api_key_env": "",
            "api_key_configured": False,
            "config_fingerprint": "0" * 64,
        },
    }


def test_openai_compatible_provider_returns_envelope_without_network(monkeypatch) -> None:
    """生产 provider 使用固定 chat 参数并返回可解析的 OpenAI envelope。"""

    calls: list[dict] = []

    class _Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return {
                "id": "fake-response",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": '{"status":"approved"}',
                        },
                    }
                ],
            }

    class _Client:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.chat = SimpleNamespace(completions=_Completions())

    monkeypatch.setattr(provider_module.openai, "OpenAI", _Client)
    provider = provider_module.OpenAICompatibleExpertProvider(
        model="small-expert",
        base_url="https://expert.example/v1/",
        api_key="secret-value",
        role="expert",
    )
    envelope = provider([{"role": "user", "content": "{}"}])

    assert envelope["choices"][0]["message"]["content"] == '{"status":"approved"}'
    assert calls[0]["model"] == "small-expert"
    assert "reasoning_effort" not in calls[0]
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert calls[0]["stream"] is False


def test_deepseek_pro_provider_uses_max_without_response_format(monkeypatch) -> None:
    """General/Judge 的 DeepSeek v4 Pro provider 发送 max 且不发送 JSON mode。"""

    calls: list[dict] = []

    class _Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return {
                "model": "deepseek-v4-pro",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": '{"status":"approved"}'},
                    }
                ],
            }

    class _Client:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=_Completions())

    monkeypatch.setattr(provider_module.openai, "OpenAI", _Client)
    provider = provider_module.OpenAICompatibleExpertProvider(
        model="deepseek-v4-pro",
        base_url="https://general.example/v1",
        api_key="secret-value",
        role="general",
    )
    provider([{"role": "user", "content": "{}"}])

    assert calls[0]["reasoning_effort"] == "max"
    assert calls[0]["temperature"] == 0.0
    assert calls[0]["top_p"] == 1.0
    assert calls[0]["stream"] is False
    assert "response_format" not in calls[0]


def test_deepseek_pro_provider_uses_configured_high_without_response_format(
    monkeypatch,
) -> None:
    """General/Judge provider 使用配置的 high，并保留 Pro 传输策略。"""

    monkeypatch.setenv("DEEPSEEK_V4_PRO_REASONING_EFFORT", "high")
    calls: list[dict] = []

    class _Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return {
                "model": "deepseek-v4-pro",
                "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}],
            }

    class _Client:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=_Completions())

    monkeypatch.setattr(provider_module.openai, "OpenAI", _Client)
    provider = provider_module.OpenAICompatibleExpertProvider(
        model="deepseek-v4-pro",
        base_url="https://general.example/v1",
        api_key="secret-value",
        role="general",
    )
    provider([{"role": "user", "content": "{}"}])

    assert calls[0]["reasoning_effort"] == "high"
    assert "response_format" not in calls[0]


def test_deepseek_pro_provider_rejects_invalid_reasoning_before_request(
    monkeypatch,
) -> None:
    """provider 调用时遇到非法推理强度不能触发网络请求。"""

    monkeypatch.delenv("DEEPSEEK_V4_PRO_REASONING_EFFORT", raising=False)
    calls: list[dict] = []

    class _Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return {"choices": [{"message": {"content": "{}"}}]}

    class _Client:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=_Completions())

    monkeypatch.setattr(provider_module.openai, "OpenAI", _Client)
    provider = provider_module.OpenAICompatibleExpertProvider(
        model="deepseek-v4-pro",
        base_url="https://general.example/v1",
        api_key="secret-value",
        role="general",
    )
    monkeypatch.setenv("DEEPSEEK_V4_PRO_REASONING_EFFORT", "low")

    with pytest.raises(ValueError, match="DEEPSEEK_V4_PRO_REASONING_EFFORT"):
        provider([{"role": "user", "content": "{}"}])
    assert calls == []


def test_cluster_provider_builds_basic_and_public_api_headers_without_network(
    monkeypatch,
) -> None:
    """集群 provider 使用 Basic 与 x-api-key，且公开配置不含认证正文。"""

    clients: list[dict] = []

    class _Completions:
        def create(self, **kwargs):
            return {
                "model": "interns2-preview-sft",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": '{"ok":true}'},
                    }
                ],
                "request": kwargs,
            }

    class _Client:
        def __init__(self, **kwargs):
            clients.append(kwargs)
            self.chat = SimpleNamespace(completions=_Completions())

    monkeypatch.setattr(provider_module.openai, "OpenAI", _Client)
    bundle = provider_module.build_expert_provider_bundle(
        enabled=True,
        cluster_model="interns2-preview-sft",
        cluster_base_url="https://cluster.example/v1",
        cluster_ingress_ak="ak-test",
        cluster_ingress_sk="sk-test",
        cluster_public_api_key="public-test",
        fallback_api_key="fallback-test",
        general_api_key="general-test",
    )
    assert bundle.cluster is not None
    assert bundle.cluster.provider is not None
    assert "ak-test" not in json.dumps(bundle.public_metadata())
    assert "sk-test" not in json.dumps(bundle.public_metadata())
    assert "public-test" not in json.dumps(bundle.public_metadata())
    cluster_client = next(
        client
        for client in clients
        if isinstance(client.get("default_headers"), dict)
        and client["default_headers"].get("x-api-key") == "public-test"
    )
    headers = cluster_client["default_headers"]
    assert headers["Authorization"] == "Basic YWstdGVzdDpzay10ZXN0"
    assert headers["x-api-key"] == "public-test"
    response = bundle.cluster.provider(
        [{"role": "user", "content": "{}"}],
    )
    assert response["choices"][0]["message"]["content"] == '{"ok":true}'


def test_openai_compatible_provider_accepts_operation_output_budget(monkeypatch) -> None:
    """调用点可以把 operation profile 的输出预留量传给 provider。"""

    calls: list[dict] = []

    class _Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return {
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"content": '{"status":"approved"}'},
                    }
                ]
            }

    class _Client:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=_Completions())

    monkeypatch.setattr(provider_module.openai, "OpenAI", _Client)
    provider = provider_module.OpenAICompatibleExpertProvider(
        model="small-expert",
        base_url="https://expert.example/v1",
        api_key="secret-value",
        role="expert",
    )
    provider([{"role": "user", "content": "{}"}], max_tokens=1024)

    assert calls[0]["max_tokens"] == 1024


@pytest.mark.parametrize(
    ("role", "expects_probe_budget"),
    [("general", False), ("expert", True), ("fallback", True)],
)
def test_startup_probe_uses_role_specific_output_budget(
    monkeypatch,
    role,
    expects_probe_budget,
) -> None:
    """通用探针保持无上限；Gaussian 专家和回退探针使用轻量预算。"""

    calls: list[dict] = []

    class _Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return {
                "model": "general-model",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": '{"ok":true}'},
                    }
                ],
            }

    class _Client:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=_Completions())

    monkeypatch.setattr(provider_module.openai, "OpenAI", _Client)
    provider = provider_module.OpenAICompatibleExpertProvider(
        model="general-model",
        base_url="https://provider.example/v1",
        api_key="secret-value",
        role=role,
    )

    ok, facts, reason = provider_module._run_expert_startup_probe(
        provider,
        configured_model="general-model",
        timeout_seconds=30.0,
    )

    assert ok is True
    assert reason is None
    assert facts["strict_json"] is True
    if expects_probe_budget:
        assert (
            calls[0]["max_tokens"]
            == provider_module.DEFAULT_EXPERT_STARTUP_PROBE_MAX_TOKENS
        )
    else:
        assert "max_tokens" not in calls[0]


@pytest.mark.parametrize(
    ("role", "expected_default"),
    [
        ("general", None),
        ("expert", provider_module.DEFAULT_EXPERT_PROVIDER_OUTPUT_TOKENS),
        ("fallback", provider_module.DEFAULT_EXPERT_PROVIDER_OUTPUT_TOKENS),
    ],
)
def test_openai_compatible_provider_keeps_role_specific_default_budget(
    monkeypatch,
    role,
    expected_default,
) -> None:
    """General opinion/Judge 不设上限，Gaussian 与 Flash 专家仍保留 4096。"""

    calls: list[dict] = []

    class _Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return {"choices": [{"message": {"content": "{}"}}]}

    class _Client:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=_Completions())

    monkeypatch.setattr(provider_module.openai, "OpenAI", _Client)
    provider = provider_module.OpenAICompatibleExpertProvider(
        model=f"{role}-model",
        base_url="https://provider.example/v1",
        api_key="secret-value",
        role=role,
    )
    provider([{"role": "user", "content": "{}"}])

    if expected_default is None:
        assert "max_tokens" not in calls[0]
    else:
        assert calls[0]["max_tokens"] == expected_default


def test_research_task_freezes_model_configuration_without_credentials(
    tmp_path: Path,
) -> None:
    """V2 TaskCreated 保存专家公开配置，不保存密钥正文。"""

    bundle = provider_module.build_expert_provider_bundle(
        enabled=True,
        expert_model="small-expert",
        expert_base_url="https://expert.example/v1",
        expert_api_key="expert-secret",
        general_model="general-model",
        general_base_url="https://general.example/v1",
        fallback_api_key="general-secret",
        general_api_key="general-secret",
    )
    application = object.__new__(ResearchTaskApplication)
    application.expert_model_runtime = bundle
    configuration = application._expert_model_configuration()
    memory = ResearchTaskWorkingMemory(tmp_path / "research_task.sqlite3")
    memory.create_research_task(
        research_task_id="research-task-production",
        contract_version="investigation_v2.task.v2",
        input={
            "scientific_question": "测试问题",
            "task_config": {"max_hypothesis_batches": 1},
        },
        policy={
            "stage_architecture_version": "investigation_v2",
            "entry_strategy": "standard_hypothesis_generation",
            "model_configuration": configuration,
        },
        dependencies={},
    )

    facts = memory.read_research_task_facts("research-task-production")
    frozen = facts.research_task.policy["model_configuration"]
    assert frozen == configuration
    assert "expert-secret" not in json.dumps(frozen)
    assert "general-secret" not in json.dumps(frozen)

    changed = json.loads(json.dumps(configuration))
    changed["expert"]["model"] = "different-model"
    assert validate_model_configuration(changed)["expert"]["model"] != frozen["expert"]["model"]
    malformed = json.loads(json.dumps(configuration))
    malformed["expert"].pop("config_fingerprint")
    with pytest.raises(ContractValidationError):
        validate_model_configuration(malformed)


def test_cli_application_passes_runtime_bundle(monkeypatch, tmp_path: Path) -> None:
    """CLI 开关和公开模型参数真实传给 ResearchTaskApplication。"""

    captured: dict[str, object] = {}

    class _Application:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def create_standard_research_task(self, **_kwargs):
            return {"status": "succeeded", "research_task_id": "research-task-test"}

    monkeypatch.setattr(controller_module, "ResearchTaskApplication", _Application)
    monkeypatch.setenv("ARCHE_CHEM_API_KEY", "general-secret")
    monkeypatch.setenv("ARCHE_CHEM_BASE_URL", "https://general.example/v1")
    monkeypatch.setenv("ARCHE_CHEM_MODEL_NAME", "general-model")
    result = controller_module.run_from_args(
        [
            "create-standard-research-task",
            "--question",
            "测试问题",
            "--work-dir",
            str(tmp_path),
            "--expert-model",
            "small-expert",
            "--expert-base-url",
            "https://expert.example/v1",
            "--expert-fallback-model",
            "cli-flash",
            "--no-search",
            "--num-queries",
            "1",
            "--num-hypotheses-per-query",
            "1",
            "--top-n-hypotheses",
            "5",
        ]
    )

    assert result == {
        "status": "succeeded",
        "research_task_id": "research-task-test",
    }
    runtime = captured["expert_model_runtime"]
    assert runtime.enabled is True
    assert runtime.expert.model == "small-expert"
    assert runtime.general.model
    assert runtime.expert_provider is not None
    assert runtime.fallback.model == "cli-flash"
    assert runtime.fallback_provider is not None


def test_cli_role_key_environments_do_not_override_shared_api_key(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """旧角色密钥参数不会覆盖统一的 ARCHE_CHEM_API_KEY。"""

    captured: dict[str, object] = {}

    class _Client:
        def __init__(self, *, api_key: str, **_kwargs: object) -> None:
            self.api_key = api_key

    class _Application:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(provider_module.openai, "OpenAI", _Client)
    monkeypatch.setattr(controller_module, "ResearchTaskApplication", _Application)
    monkeypatch.setenv("ARCHE_CHEM_API_KEY", "shared-secret")
    monkeypatch.setenv("ARCHE_CHEM_BASE_URL", "https://general.example/v1")
    monkeypatch.setenv("ARCHE_CHEM_MODEL_NAME", "general-model")
    args = controller_module._build_cli_parser().parse_args(
        [
            "create-standard-research-task",
            "--question",
            "测试问题",
            "--work-dir",
            str(tmp_path),
            "--api-key",
            "shared-secret",
            "--expert-fallback-api-key-env",
            "TEST_FLASH_KEY",
            "--expert-general-api-key-env",
            "TEST_PRO_KEY",
            "--no-search",
        ]
    )

    controller_module._controller_from_cli_args(args)

    runtime = captured["expert_model_runtime"]
    assert runtime.fallback.api_key_env == "TEST_FLASH_KEY"
    assert runtime.general.api_key_env == "TEST_PRO_KEY"
    assert runtime.fallback_provider._client.api_key == "shared-secret"
    assert runtime.general_provider._client.api_key == "shared-secret"


def test_python_controller_role_key_environments_do_not_override_shared_api_key(
    monkeypatch,
) -> None:
    """直接 Python 入口统一使用同一个显式密钥。"""

    captured: dict[str, object] = {}
    real_builder = provider_module.build_expert_provider_bundle

    def capture_builder(**kwargs: object):
        captured.update(kwargs)
        return real_builder(enabled=False)

    monkeypatch.setenv(
        "ARCHE_EXPERT_FALLBACK_API_KEY_ENV",
        "TEST_FLASH_KEY",
    )
    monkeypatch.setenv(
        "ARCHE_EXPERT_GENERAL_API_KEY_ENV",
        "TEST_PRO_KEY",
    )
    monkeypatch.setattr(
        controller_module,
        "build_expert_provider_bundle",
        capture_builder,
    )
    monkeypatch.setattr(
        ResearchTaskApplication,
        "_initialize_agents",
        lambda _self: None,
    )

    ResearchTaskApplication(
        deepseek_api_key="shared-secret",
        memory_enabled=False,
    )

    assert captured["fallback_api_key"] == "shared-secret"
    assert captured["general_api_key"] == "shared-secret"


def test_disabled_cli_does_not_read_expert_environment_or_initialize_provider(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """显式关闭专家时不读取 ARCHE_EXPERT_*，也不初始化专家客户端。"""

    original_environment = controller_module.os.environ

    class _EnvironmentProxy:
        def _reject_expert_key(self, key: object) -> None:
            if isinstance(key, str) and key.startswith("ARCHE_EXPERT_"):
                raise AssertionError(
                    "专家关闭时不应读取 ARCHE_EXPERT_* 环境变量"
                )

        def get(self, key: object, default: object = None) -> object:
            self._reject_expert_key(key)
            return original_environment.get(key, default)

        def __getitem__(self, key: object) -> str:
            self._reject_expert_key(key)
            return original_environment[key]

        def __contains__(self, key: object) -> bool:
            self._reject_expert_key(key)
            return key in original_environment

        def __setitem__(self, key: object, value: str) -> None:
            self._reject_expert_key(key)
            original_environment[key] = value

        def __delitem__(self, key: object) -> None:
            self._reject_expert_key(key)
            del original_environment[key]

        def __iter__(self):
            return iter(original_environment)

        def __len__(self) -> int:
            return len(original_environment)

    class _ForbiddenOpenAI:
        def __init__(self, **_kwargs):
            raise AssertionError("专家关闭时不应初始化 OpenAI client")

    captured: dict[str, object] = {}

    class _Application:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(controller_module.os, "environ", _EnvironmentProxy())
    monkeypatch.setattr(provider_module.openai, "OpenAI", _ForbiddenOpenAI)
    monkeypatch.setattr(controller_module, "ResearchTaskApplication", _Application)

    args = controller_module._build_cli_parser().parse_args(
        [
            "create-standard-research-task",
            "--question",
            "测试问题",
            "--work-dir",
            str(tmp_path),
            "--disable-expert-model",
            "--no-search",
        ]
    )
    controller_module._controller_from_cli_args(args)

    runtime = captured["expert_model_runtime"]
    assert runtime.enabled is False
    assert runtime.expert_provider is None
    assert runtime.general_provider is None
