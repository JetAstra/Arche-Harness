"""Working Memory contracts 的最终 CRAP 分支覆盖。"""

from __future__ import annotations

import copy

import pytest

from chemistry_multiagent.memory.contracts import (
    ContractValidationError,
    INVESTIGATION_STAGE_ARCHITECTURE_VERSION,
    validate_model_configuration,
    validate_task_created_payload,
)


def _role(
    *,
    model: str | None = "model",
    base_url: str | None = "https://provider.example/v1",
    api_key_env: str = "PROVIDER_KEY",
    api_key_configured: bool = True,
    fingerprint: str = "a" * 64,
) -> dict[str, object]:
    """构造不含认证正文的 provider 公开元数据。"""

    return {
        "model": model,
        "base_url": base_url,
        "api_key_env": api_key_env,
        "api_key_configured": api_key_configured,
        "config_fingerprint": fingerprint,
    }


def _disabled_configuration() -> dict[str, object]:
    role = _role(
        model=None,
        base_url=None,
        api_key_env="",
        api_key_configured=False,
        fingerprint="0" * 64,
    )
    return {
        "enabled": False,
        "status": "disabled",
        "expert": copy.deepcopy(role),
        "fallback": copy.deepcopy(role),
        "general": copy.deepcopy(role),
    }


def _enabled_configuration(
    *,
    expert_ready: bool = True,
    cluster_ready: bool = False,
) -> dict[str, object]:
    empty = _role(
        model=None,
        base_url=None,
        api_key_env="",
        api_key_configured=False,
        fingerprint="0" * 64,
    )
    configuration: dict[str, object] = {
        "enabled": True,
        "status": "configured",
        "expert": _role() if expert_ready else copy.deepcopy(empty),
        "fallback": copy.deepcopy(empty),
        "general": _role(model="general-model"),
    }
    if cluster_ready:
        configuration["cluster"] = _role(model="cluster-model")
    if not expert_ready and not cluster_ready:
        configuration["status"] = "not_configured"
    return configuration


def _disabled_selection() -> dict[str, object]:
    return {
        "schema_version": "expert-runtime-selection.v2",
        "status": "disabled",
        "backend": "disabled",
        "provider_role": None,
        "provider": _role(
            model=None,
            base_url=None,
            api_key_env="",
            api_key_configured=False,
            fingerprint="0" * 64,
        ),
        "served_model": None,
        "probe": {"expert": None, "fallback": None, "general": None},
        "fallback_reason": None,
        "failure_role": None,
    }


def _legacy_payload() -> dict[str, object]:
    return {
        "scientific_question": "验证反应路径",
        "chemistry_context": {"reaction_type": "test"},
        "task_config": {
            "num_queries": 1,
            "num_hypotheses_per_query": 5,
            "top_n_workflows": 5,
            "max_parallel_combinations": 5,
            "max_workflow_revisions_per_combination": 3,
            "max_hypothesis_batches": 3,
        },
        "task_output_relative_dir": "outputs/runs/task-1",
        "tool_catalog_hash": "b" * 64,
    }


def test_model_configuration_accepts_disabled_and_enabled_variants() -> None:
    """关闭、专家可用和集群专家可用三种配置都保持原协议语义。"""

    assert validate_model_configuration(_disabled_configuration())["status"] == "disabled"
    assert validate_model_configuration(_enabled_configuration())["status"] == "configured"
    cluster = _enabled_configuration(expert_ready=False, cluster_ready=True)
    assert validate_model_configuration(cluster)["status"] == "configured"
    incomplete = _enabled_configuration(expert_ready=False)
    assert validate_model_configuration(incomplete)["status"] == "not_configured"


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda value: value.update(enabled="yes"), "enabled"),
        (lambda value: value.update(status="unknown"), "status"),
        (lambda value: value.update(status="configured"), "关闭专家"),
        (lambda value: value["expert"].update(model=""), "非空字符串"),
        (lambda value: value["expert"].update(base_url=3), "非空字符串"),
        (lambda value: value["expert"].update(api_key_env=None), "字符串"),
        (lambda value: value["expert"].update(api_key_configured="yes"), "boolean"),
        (lambda value: value["expert"].update(config_fingerprint="Z" * 64), "SHA-256"),
        (lambda value: value["expert"].pop("model"), "字段不匹配"),
        (lambda value: value.update(extra=True), "字段不匹配"),
    ],
)
def test_model_configuration_rejects_malformed_roles_and_header(mutator, message: str) -> None:
    """顶层和角色字段的每种 fail-closed 分支都必须抛出合同错误。"""

    configuration = (
        _disabled_configuration()
        if message == "关闭专家"
        else _enabled_configuration()
    )
    mutator(configuration)
    with pytest.raises(ContractValidationError, match=message):
        validate_model_configuration(configuration)


def test_model_configuration_rejects_non_object_and_disabled_role_residue() -> None:
    """非对象输入及关闭专家后残留的角色配置不能被静默接受。"""

    with pytest.raises(ContractValidationError, match="JSON object"):
        validate_model_configuration([])
    configuration = _disabled_configuration()
    configuration["general"] = _role()
    with pytest.raises(ContractValidationError, match="固定空配置"):
        validate_model_configuration(configuration)


def test_task_created_accepts_legacy_v2_and_disabled_selection() -> None:
    """TaskCreated 同时保留 legacy/v2 配置选择和禁用专家选择语义。"""

    legacy = validate_task_created_payload(_legacy_payload())
    assert legacy["stage_architecture_version"] == "legacy_workflow_v1"
    assert legacy["model_configuration"]["status"] == "disabled"

    v2 = _legacy_payload()
    v2["stage_architecture_version"] = INVESTIGATION_STAGE_ARCHITECTURE_VERSION
    v2["task_config"] = {
        "num_queries": 1,
        "num_hypotheses_per_query": 5,
        "top_n_hypotheses": 5,
        "max_parallel_investigations": 5,
        "max_investigation_repairs_per_combination": 3,
        "max_hypothesis_batches": 3,
    }
    v2["expert_backend_selection"] = _disabled_selection()
    result = validate_task_created_payload(v2)
    assert (
        result["stage_architecture_version"]
        == INVESTIGATION_STAGE_ARCHITECTURE_VERSION
    )
    assert result["expert_backend_selection"]["backend"] == "disabled"


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda value: value.pop("scientific_question"), "missing"),
        (lambda value: value.update(unregistered=True), "extra"),
        (lambda value: value.update(scientific_question=""), "scientific_question"),
        (lambda value: value.update(chemistry_context=[]), "chemistry_context"),
        (lambda value: value.update(task_output_relative_dir="../outside"), "task_output"),
        (lambda value: value.update(tool_catalog_hash="bad"), "tool_catalog_hash"),
        (
            lambda value: value.update(model_configuration={"enabled": True}),
            "model_configuration",
        ),
    ],
)
def test_task_created_rejects_invalid_static_fields(mutator, message: str) -> None:
    """任务静态身份、输出引用和模型配置的错误都在入口被拒绝。"""

    payload = _legacy_payload()
    mutator(payload)
    with pytest.raises(ContractValidationError, match=message):
        validate_task_created_payload(payload)
