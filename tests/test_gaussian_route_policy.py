"""scientific question 到 Gaussian 路由策略的公开接口测试。"""

import pytest

from chemistry_multiagent.utils.gaussian_routing import (
    CATALOG_DECLARED_POLICY_ID,
    derive_gaussian_route_policy,
    is_allowed_typed_gaussian_step,
    verify_frozen_gaussian_route_policy,
)
from chemistry_multiagent.utils.workflow_contract import (
    project_minichem_stage_tool_projection,
)
from tests.test_gaussian_workflow_contract import _gaussian_catalog


def test_question_does_not_select_gaussian_route_policy() -> None:
    """任意题面和兼容 profile 都只得到同一目录选择记录。"""

    first = derive_gaussian_route_policy(
        "一个未登记的科学问题。",
        case_profile="case1_fixed_reproduction",
    )
    second = derive_gaussian_route_policy(
        "另一个完全不同的科学问题。",
        case_profile="arbitrary_compatibility_profile",
    )

    assert first.policy_id == second.policy_id == CATALOG_DECLARED_POLICY_ID
    assert first.allows_typed_gaussian_action("arbitrary_action", "arbitrary_backend")
    assert second.allows_typed_gaussian_action("arbitrary_action", "arbitrary_backend")
    assert first.case_profile == "case1_fixed_reproduction"
    assert second.case_profile == "arbitrary_compatibility_profile"


def test_question_normalization_uses_nfkc_and_all_whitespace() -> None:
    """NFKC 与空白折叠等价，但科学文字、标点和大小写仍参与指纹。"""

    first = derive_gaussian_route_policy("ＡＢＣ\t  1．０")
    equivalent = derive_gaussian_route_policy("ABC\n1.0")
    changed_case = derive_gaussian_route_policy("abc\n1.0")
    changed_punctuation = derive_gaussian_route_policy("ABC\n1,0")
    changed_science = derive_gaussian_route_policy("ABC\n1.1")

    assert first.question_fingerprint == equivalent.question_fingerprint
    assert first.policy_id == equivalent.policy_id == CATALOG_DECLARED_POLICY_ID
    assert changed_case.question_fingerprint != first.question_fingerprint
    assert changed_punctuation.question_fingerprint != first.question_fingerprint
    assert changed_science.question_fingerprint != first.question_fingerprint


@pytest.mark.parametrize("value", [None, 1, "", " \t\n"])
def test_non_string_or_empty_question_is_rejected(value: object) -> None:
    """空值和非字符串不能进入路由接口。"""

    with pytest.raises(ValueError):
        derive_gaussian_route_policy(value)


def test_frozen_route_record_only_verifies_same_question() -> None:
    """恢复时核对同一题面指纹，但不依据指纹改变目录行为。"""

    policy = derive_gaussian_route_policy("需要核对的科学问题。")
    restored = verify_frozen_gaussian_route_policy(
        "需要核对的科学问题。",
        policy.to_persisted_record(),
    )
    assert restored.policy_id == CATALOG_DECLARED_POLICY_ID
    with pytest.raises(ValueError, match="fingerprint"):
        verify_frozen_gaussian_route_policy(
            "恢复时被替换的科学问题。",
            policy.to_persisted_record(),
        )


def test_catalog_pairs_are_visible_for_different_questions() -> None:
    """不同题面得到相同的完整 Action/Backend 投影，且不含隐藏 pair。"""

    first_projection = project_minichem_stage_tool_projection(
        _gaussian_catalog(),
        gaussian_route_policy=derive_gaussian_route_policy("问题 A"),
    )
    second_projection = project_minichem_stage_tool_projection(
        _gaussian_catalog(),
        gaussian_route_policy=derive_gaussian_route_policy("问题 B"),
    )
    assert first_projection == second_projection
    assert all("planner_constraints" not in item for item in first_projection)


def test_gaussian_pair_detection_prefers_catalog_metadata_over_ids() -> None:
    """Gaussian pair 识别优先读取目录 metadata，不把相似 ID 当成科学路线。"""

    non_gaussian_catalog = {
        "actions": [
            {
                "id": "legacy_gaussian_route_builder",
                "backend_ids": ["legacy_gaussian_backend"],
                "domain": "format_conversion",
            }
        ],
        "backends": [
            {
                "id": "legacy_gaussian_backend",
                "runtime": "format_conversion",
                "capabilities": ["route_text"],
            }
        ],
    }
    step = {
        "execution_layer": "predefined_actions",
        "action_id": "legacy_gaussian_route_builder",
        "backend_id": "legacy_gaussian_backend",
    }

    assert not is_allowed_typed_gaussian_step(step, non_gaussian_catalog)


def test_gaussian_pair_detection_accepts_explicit_metadata_and_backend_identity() -> None:
    """显式 Gaussian metadata 与精确 Backend 身份均可识别。"""

    metadata_catalog = {
        "actions": [
            {
                "id": "calculate_property",
                "backend_ids": ["backend_17"],
                "software_family": "Gaussian",
            }
        ],
        "backends": [{"id": "backend_17", "runtime": "native"}],
    }
    metadata_step = {
        "execution_layer": "predefined_actions",
        "action_id": "calculate_property",
        "backend_id": "backend_17",
    }
    legacy_catalog = {
        "actions": [{"id": "legacy_action", "backend_ids": ["gaussian"]}],
        "backends": [{"id": "gaussian"}],
    }
    legacy_step = {
        "execution_layer": "predefined_actions",
        "action_id": "legacy_action",
        "backend_id": "gaussian",
    }

    assert is_allowed_typed_gaussian_step(metadata_step, metadata_catalog)
    assert is_allowed_typed_gaussian_step(legacy_step, legacy_catalog)


def test_gaussian_pair_detection_rejects_text_and_nested_boolean_lookalikes() -> None:
    """描述文本、相似 ID 和无关布尔值不能冒充软件身份。"""

    catalog = {
        "actions": [
            {
                "id": "gaussian_report_builder",
                "backend_ids": ["gaussian_wrapper"],
                "description": "Summarize Gaussian results.",
                "capabilities": {"supports_gpu": True},
            }
        ],
        "backends": [
            {
                "id": "gaussian_wrapper",
                "display_name": "Gaussian-compatible report wrapper",
                "runtime": "python",
            }
        ],
    }
    step = {
        "execution_layer": "predefined_actions",
        "action_id": "gaussian_report_builder",
        "backend_id": "gaussian_wrapper",
    }

    assert not is_allowed_typed_gaussian_step(step, catalog)
