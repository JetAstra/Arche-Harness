"""V2 Program Gate 检查身份与 Router 证据投影回归。"""

from __future__ import annotations

from copy import deepcopy

import pytest

from chemistry_multiagent.utils.router_evidence import (
    build_router_evidence_index,
    validate_program_gate_for_evidence_ready,
)
from tests.test_router_evidence import _router_input


def test_program_gate_check_identity_projects_to_continuous_router_reference() -> None:
    """持久检查的顺序身份必须投影为当前连续的 c1/evidence_ref。"""

    router_input = _router_input(["结构产物已完成。"])
    check = router_input["program_validation"]["registered_check_results"][0]
    assert check["check_ref"] == "c1"
    assert check["evidence_ref"] == (
        "/execution_result/steps/0/raw_output/output_artifacts/0"
    )
    assert router_input["evidence_index"][0]["evidence_ref"] == (
        "/program_validation/registered_check_results/0/actual"
    )


def test_router_rejects_missing_or_forged_program_check_identity() -> None:
    """删除或改写检查身份不能伪装成当前执行证据。"""

    router_input = _router_input(["结构产物已完成。"])
    missing = deepcopy(router_input["program_validation"])
    del missing["registered_check_results"][0]["check_ref"]
    with pytest.raises(ValueError, match="检查|字段|一致"):
        build_router_evidence_index(
            router_input["execution_result"],
            missing,
            router_input["workflow"],
            router_input["tool_catalog_projection"],
        )

    forged = deepcopy(router_input["program_validation"])
    forged["registered_check_results"][0]["actual"]["artifact_id"] = "forged"
    with pytest.raises(ValueError, match="派生|一致|证据"):
        build_router_evidence_index(
            router_input["execution_result"],
            forged,
            router_input["workflow"],
            router_input["tool_catalog_projection"],
        )


def test_final_evidence_ready_recheck_uses_saved_gate_constraints() -> None:
    """最终发布只读取已保存 Gate 的可靠性，不把失败检查当成功。"""

    router_input = _router_input(["结构产物已完成。"])
    validation = deepcopy(router_input["program_validation"])
    validation["program_gate_result"]["constraint_reasons"] = [
        "scientific_validation_status_invalid"
    ]
    with pytest.raises(ValueError, match="evidence_ready"):
        validate_program_gate_for_evidence_ready(validation)
