"""通用 ActionResult 传输与 wrapper 原样返回合同测试。"""

from __future__ import annotations

from typing import Any

from minichem_toolbox.catalog import action_specs
from minichem_toolbox.models import ActionRequest
from minichem_toolbox.scope import ScopeContext, activate_scope
from minichem_toolbox import service
from minichem_mcp_tools import discovery_tools, registry
from minichem_mcp_tools.discovery_models import ProgressiveActionRequest
from minichem_mcp_tools.registry import GaussianActionRequest
from minichem_mcp_tools.result_transport import compact_action_result


def _artifact(
    artifact_id: str,
    semantic_type: str,
    *,
    action: str = "calculate_energy",
    backend: str = "internal_energy",
) -> dict[str, Any]:
    """构造带稳定身份的通用 ArtifactRef。"""

    return {
        "artifact_id": artifact_id,
        "semantic_type": semantic_type,
        "media_type": "application/json",
        "sha256": "a" * 64,
        "path": f"_tool_artifacts/objects/{artifact_id}.json",
        "producer_action": action,
        "producer_backend": backend,
        "parent_artifact_ids": [],
    }


def _result(
    action: str,
    result: Any,
    *,
    backend: str = "internal_energy",
    artifacts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """构造最小规范 ActionResult。"""

    return {
        "status": "success",
        "action": action,
        "action_version": "1.0.0",
        "requested_backend": backend,
        "backend": backend,
        "result": result,
        "output_artifacts": artifacts
        or [_artifact("art_" + "1" * 32, "EnergyResult", action=action, backend=backend)],
    }


def test_compact_view_is_explicit_and_action_independent() -> None:
    """显式 compact 视图只按通用大小规则压缩，不按 Action ID 改写字段。"""

    payload = {
        "workflow_id": "caller-defined-workflow",
        "unknown_result_field": {"keep": True},
        "path_image_candidates": [
            {"candidate_id": f"candidate-{index}", "extra": index}
            for index in range(161)
        ],
    }
    first = compact_action_result(
        _result("search_transition_state_path", payload, backend="transition_state_path_search")
    )
    second = compact_action_result(_result("unrelated_action", payload))

    assert first["result"]["workflow_id"] == payload["workflow_id"]
    assert first["result"]["unknown_result_field"] == payload["unknown_result_field"]
    assert first["result"]["path_image_candidates"]["inline"] is False
    assert first["result"]["path_image_candidates"]["item_count"] == 161
    assert second["result"] == first["result"]


def test_compact_view_keeps_all_artifact_refs_and_unknown_error_fields() -> None:
    """通用视图保留所有 ArtifactRef 身份，不根据科学 Action 丢弃未知字段。"""

    artifacts = [
        _artifact("art_" + "1" * 32, "PrimaryResult"),
        _artifact("art_" + "2" * 32, "AuxiliaryResult"),
        _artifact("art_" + "3" * 32, "BackendDiagnostic"),
    ]
    value = _result(
        "arbitrary_action",
        {"new_field": "new-value", "candidates": list(range(180))},
        artifacts=artifacts,
    )
    value["error"] = {"code": "new_error_code", "retryable": True}

    compacted = compact_action_result(value)

    assert compacted["result"]["new_field"] == "new-value"
    assert compacted["output_artifacts"] == artifacts
    assert compacted["error"] == value["error"]
    assert compacted["artifact_handoff"]["primary_output"] == artifacts[0]


def test_compact_view_bounds_large_scalar_mappings_without_field_routes() -> None:
    """超大结果只保留通用标量预算，完整内容仍可从完整结果 Artifact 获取。"""

    payload = {
        "opaque_scalar": "x" * 17_000,
        "another_scalar": 42,
        "unknown_nested": {"value": "y" * 17_000},
    }
    compacted = compact_action_result(
        _result(
            "arbitrary_action",
            payload,
            artifacts=[
                _artifact("art_" + "4" * 32, "ActionResult"),
                _artifact("art_" + "5" * 32, "AuxiliaryResult"),
            ],
        )
    )

    view = compacted["result"]
    assert view["inline"] is False
    assert view["another_scalar"] == 42
    assert len(view["inline_scalars"]["opaque_scalar"]) <= 4_000
    assert compacted["transport"]["complete_primary_result"]["semantic_type"] == "ActionResult"


def test_progressive_execute_returns_complete_service_result(monkeypatch, tmp_path) -> None:
    """progressive execute_action 不压缩、分页或重写服务端结果。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    artifacts = [
        _artifact("art_" + "6" * 32, "PrimaryResult"),
        _artifact("art_" + "7" * 32, "CandidateFile"),
    ]
    service_result = _result(
        "search_transition_state_path",
        {
            "unknown_result_field": {"sentinel": "preserved"},
            "path_image_candidates": [
                {"candidate_id": f"candidate-{index}"} for index in range(161)
            ],
        },
        backend="transition_state_path_search",
        artifacts=artifacts,
    )
    monkeypatch.setattr(discovery_tools, "_execute_action", lambda *_args, **_kwargs: service_result)

    returned = discovery_tools.execute_action(
        ProgressiveActionRequest(
            action_id="search_transition_state_path",
            backend_id="transition_state_path_search",
            inputs={"reactants": ["CC"], "products": ["CO"]},
            method_spec={},
            action_settings={},
        )
    )

    assert returned is service_result
    assert returned["result"] == service_result["result"]
    assert len(returned["result"]["path_image_candidates"]) == 161
    assert returned["output_artifacts"] == artifacts


def test_full_registered_action_returns_complete_service_result(monkeypatch, tmp_path) -> None:
    """full 一 Action 一 tool 兼容入口也只返回 service ActionResult。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    service_result = _result(
        "calculate_energy",
        {"unknown": {"candidate": [1, 2, 3]}, "all_candidates": list(range(161))},
    )
    monkeypatch.setattr(registry, "execute_action", lambda *_args, **_kwargs: service_result)
    specification = action_specs()["calculate_energy"]

    returned = registry._execute_public_registered_action(
        specification,
        ActionRequest.model_validate(
            {
            "backend_id": "internal_energy",
            "inputs": {"structure": {"atoms": []}},
            "method_spec": {},
            "action_settings": {},
            "resource_limits": {},
            }
        ),
    )

    assert returned is service_result
    assert len(returned["result"]["all_candidates"]) == 161


def test_full_registered_gaussian_action_preserves_legacy_operation_identity(
    monkeypatch, tmp_path
) -> None:
    """专用公开请求转换后仍保留 legacy operation_id，重复副作用只发生一次。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    monkeypatch.setattr(registry, "execute_traced", lambda _name, _args, callback: callback())
    calls: list[int] = []

    def execute_once(_action_id, _request, *, resources_bound=False):
        calls.append(1)
        return {"status": "success", "result": {"count": len(calls)}}

    monkeypatch.setattr(service, "_execute_action_once", execute_once)
    invoke = registry._make_action_callable(action_specs()["generate_gaussian_code"])
    request = GaussianActionRequest.model_validate(
        {
            "backend_id": "gaussian_codegen",
            "inputs": {"user_input": "Single point energy."},
            "method_spec": {},
            "action_settings": {},
            "resource_limits": {},
            "operation_id": "legacy-public-operation",
        }
    )

    first = invoke(request)
    second = invoke(request)

    assert first == second
    assert calls == [1]


def test_full_registered_action_rejects_forged_scope_operation_identity(
    monkeypatch, tmp_path
) -> None:
    """scope 公开入口拒绝伪造 operation_id，且转换前后都不执行副作用。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    monkeypatch.setattr(registry, "execute_traced", lambda _name, _args, callback: callback())
    calls: list[int] = []
    monkeypatch.setattr(
        service,
        "_execute_action_once",
        lambda *_args, **_kwargs: calls.append(1) or {"status": "success"},
    )
    invoke = registry._make_action_callable(action_specs()["generate_gaussian_code"])
    request = GaussianActionRequest.model_validate(
        {
            "backend_id": "gaussian_codegen",
            "inputs": {"user_input": "Single point energy."},
            "method_spec": {},
            "action_settings": {},
            "resource_limits": {},
            "operation_id": "forged-public-operation",
            "submission_key": "public-submission-key",
        }
    )
    scope = ScopeContext(
        task_id="task-test",
        batch_id="batch-test",
        combination_id="combination-test",
        scope_id="scope-test",
        workspace=tmp_path,
        capability_hash="",
    )

    with activate_scope(scope):
        result = invoke(request)

    assert result["status"] == "invalid_request"
    assert result["error"]["code"] == "operation_id_not_assignable"
    assert calls == []
    assert "operation_id" not in GaussianActionRequest.model_json_schema()["properties"]


def test_result_transport_does_not_define_action_specific_routes() -> None:
    """传输模块不包含 Action ID 分支或旧流程专用投影名称。"""

    from pathlib import Path

    source = Path(compact_action_result.__code__.co_filename).read_text(encoding="utf-8")
    for marker in ("_FIXED_CASE1", "_fixed_case1", "_TRANSITION_STATE_PATH_ACTION_ID"):
        assert marker not in source
    assert "action_id ==" not in source
