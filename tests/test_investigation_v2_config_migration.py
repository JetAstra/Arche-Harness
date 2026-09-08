"""Investigation v2 配置合同与 legacy workflow 配置的边界测试。"""

from __future__ import annotations

import pytest

from chemistry_multiagent.controllers.chemistry_multiagent_controller import (
    _normalize_hypothesis_comparison_workers,
    _validate_expert_runtime_inputs,
    _validate_investigation_service_endpoints,
)
from chemistry_multiagent.memory.contracts import (
    ContractValidationError,
    validate_investigation_task_config,
    validate_task_config,
)
from chemistry_multiagent.utils.expert_model_provider import (
    ExpertProviderBundle,
    ExpertProviderRole,
    ExpertRuntimeSelection,
)


def _investigation_config(**overrides: object) -> dict[str, object]:
    """返回独立 v2 配置样例。"""

    config: dict[str, object] = {
        "num_queries": 4,
        "num_hypotheses_per_query": 3,
        "top_n_hypotheses": 5,
        "max_parallel_investigations": 5,
        "max_investigation_repairs_per_combination": 3,
        "max_hypothesis_batches": 3,
    }
    config.update(overrides)
    return config


def test_investigation_config_uses_only_v2_fields() -> None:
    """v2 配置返回新字段，不能把旧 workflow 字段当作事实。"""

    result = validate_investigation_task_config(_investigation_config())

    assert result == _investigation_config()
    assert "top_n_workflows" not in result
    assert "max_parallel_combinations" not in result
    assert "max_workflow_revisions_per_combination" not in result


@pytest.mark.parametrize("repair_budget", [1, 2, 4, 7])
def test_investigation_config_accepts_configurable_repair_budget(
    repair_budget: int,
) -> None:
    """Investigation repair 预算接受任意正整数，不再由固定上限卡住。"""

    result = validate_investigation_task_config(
        _investigation_config(
            max_investigation_repairs_per_combination=repair_budget,
        )
    )

    assert result["max_investigation_repairs_per_combination"] == repair_budget


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("top_n_hypotheses", 4),
        ("max_parallel_investigations", 0),
        ("max_parallel_investigations", 6),
    ],
)
def test_investigation_config_rejects_invalid_fixed_limits(
    field: str,
    value: int,
) -> None:
    """v2 Top5 和并发固定合同继续 fail closed。"""

    with pytest.raises(ContractValidationError):
        validate_investigation_task_config(_investigation_config(**{field: value}))


@pytest.mark.parametrize("repair_budget", [True, False, 0, -1, 1.5, "1"])
def test_investigation_config_rejects_invalid_repair_budget_types_and_ranges(
    repair_budget: object,
) -> None:
    """repair 预算必须是大于等于 1 的 JSON integer，不能接受布尔或其他类型。"""

    with pytest.raises(ContractValidationError):
        validate_investigation_task_config(
            _investigation_config(
                max_investigation_repairs_per_combination=repair_budget,
            )
        )


def test_legacy_config_validator_keeps_old_schema() -> None:
    """legacy validator 仍接受原有工作流字段，不被 v2 改名污染。"""

    result = validate_task_config(
        {
            "num_queries": 1,
            "num_hypotheses_per_query": 5,
            "top_n_workflows": 5,
            "max_parallel_combinations": 5,
            "max_workflow_revisions_per_combination": 3,
            "max_hypothesis_batches": 1,
        }
    )

    assert result["top_n_workflows"] == 5


def test_investigation_service_defaults_remain_separate() -> None:
    """active v2 默认把 catalog bootstrap 和正式 Investigation 分到 9011/9012。"""

    import os

    old_catalog = os.environ.pop("MINICHEM_CATALOG_ENDPOINT", None)
    old_investigation = os.environ.pop("MINICHEM_INVESTIGATION_ENDPOINT", None)
    try:
        catalog, investigation = _validate_investigation_service_endpoints()
        assert catalog == "http://127.0.0.1:9011/mcp"
        assert investigation == "http://127.0.0.1:9012/mcp"
    finally:
        if old_catalog is None:
            os.environ.pop("MINICHEM_CATALOG_ENDPOINT", None)
        else:
            os.environ["MINICHEM_CATALOG_ENDPOINT"] = old_catalog
        if old_investigation is None:
            os.environ.pop("MINICHEM_INVESTIGATION_ENDPOINT", None)
        else:
            os.environ["MINICHEM_INVESTIGATION_ENDPOINT"] = old_investigation


def test_investigation_service_allows_explicit_distinct_loopback_ports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """显式部署端口可改变，但两个 loopback /mcp profile 仍必须分离。"""

    monkeypatch.setenv("MINICHEM_CATALOG_ENDPOINT", "http://localhost:19011/mcp")
    monkeypatch.setenv(
        "MINICHEM_INVESTIGATION_ENDPOINT",
        "http://127.0.0.1:19012/mcp",
    )
    assert _validate_investigation_service_endpoints() == (
        "http://localhost:19011/mcp",
        "http://127.0.0.1:19012/mcp",
    )


@pytest.mark.parametrize(
    "invalid_endpoint",
    [
        "https://127.0.0.1:19011/mcp",
        "http://user:password@127.0.0.1:19011/mcp",
        "http://127.0.0.1:9010/mcp",
    ],
)
def test_investigation_service_rejects_legacy_or_credential_endpoint_profiles(
    invalid_endpoint: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """active v2 只接受无凭据的 loopback HTTP，不能复用 legacy 9010。"""

    monkeypatch.setenv("MINICHEM_CATALOG_ENDPOINT", invalid_endpoint)
    monkeypatch.setenv(
        "MINICHEM_INVESTIGATION_ENDPOINT",
        "http://127.0.0.1:19012/mcp",
    )
    with pytest.raises(ValueError):
        _validate_investigation_service_endpoints()


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0, 0), ("auto", 0), ("AUTO", 0), (3, 3)],
)
def test_controller_comparison_worker_normalization_accepts_auto_and_positive(
    value: int | str,
    expected: int,
) -> None:
    """Controller 比较并发把 0/auto 归一为全并发标记。"""

    assert _normalize_hypothesis_comparison_workers(value) == expected


@pytest.mark.parametrize("value", [True, -1, "invalid", 1.5])
def test_controller_comparison_worker_normalization_fails_closed(
    value: object,
) -> None:
    """布尔、负数、非法字符串和浮点并发配置必须拒绝。"""

    with pytest.raises(ValueError):
        _normalize_hypothesis_comparison_workers(value)  # type: ignore[arg-type]


def _empty_expert_role(role: str) -> ExpertProviderRole:
    """构造不含 provider 或认证内容的专家角色替身。"""

    return ExpertProviderRole(
        role=role,
        model=None,
        base_url=None,
        api_key_env=f"{role.upper()}_API_KEY",
        api_key_configured=False,
        provider=None,
        config_fingerprint="0" * 64,
    )


def _unselected_expert_bundle() -> ExpertProviderBundle:
    """构造尚未冻结后端的最小候选 bundle。"""

    return ExpertProviderBundle(
        enabled=False,
        expert=_empty_expert_role("expert"),
        general=_empty_expert_role("general"),
        fallback=_empty_expert_role("fallback"),
    )


def test_controller_expert_runtime_validation_accepts_empty_candidate_bundle() -> None:
    """候选 bundle 在启动选择前可通过窄校验。"""

    _validate_expert_runtime_inputs(_unselected_expert_bundle(), None, None, None)
    _validate_expert_runtime_inputs(None, None, None, None)


@pytest.mark.parametrize(
    "case",
    [
        "invalid_runtime",
        "selected_bundle",
        "runtime_and_provider",
        "bare_provider",
    ],
)
def test_controller_expert_runtime_validation_rejects_ambiguous_injection(
    case: str,
) -> None:
    """非法 runtime、已选择 bundle 和裸 provider 不能混入 Controller。"""

    if case == "invalid_runtime":
        with pytest.raises(TypeError):
            _validate_expert_runtime_inputs(object(), None, None, None)
        return

    if case == "selected_bundle":
        selection = ExpertRuntimeSelection(
            status="disabled",
            backend="disabled",
            provider_role=None,
            provider=None,
            provider_metadata={},
            served_model=None,
            probe={},
        )
        selected = ExpertProviderBundle(
            enabled=False,
            expert=_empty_expert_role("expert"),
            general=_empty_expert_role("general"),
            fallback=_empty_expert_role("fallback"),
            runtime_selection=selection,
        )
        with pytest.raises(ValueError, match="尚未选择后端"):
            _validate_expert_runtime_inputs(selected, None, None, None)
        return

    if case == "runtime_and_provider":
        with pytest.raises(ValueError, match="不能与裸专家 provider"):
            _validate_expert_runtime_inputs(
                _unselected_expert_bundle(),
                object(),
                None,
                None,
            )
        return

    with pytest.raises(ValueError, match="必须通过 ExpertProviderBundle"):
        _validate_expert_runtime_inputs(None, object(), None, None)
