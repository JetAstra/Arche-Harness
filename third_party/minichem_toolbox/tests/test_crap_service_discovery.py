"""覆盖 service/discovery 深模块拆分后的选择与合同分支。"""

from __future__ import annotations

from types import SimpleNamespace

from minichem_mcp_tools import discovery_tools
from minichem_toolbox import service
from minichem_toolbox.models import ActionRequest


def _spec(policy: str, *backend_ids: str) -> SimpleNamespace:
    """构造选择器所需的最小 Action 规格。"""

    return SimpleNamespace(selection_policy=policy, backend_ids=backend_ids or ("fixed",))


def test_service_backend_selector_covers_each_policy_and_rejections(monkeypatch) -> None:
    """所有 provider selection policy 都保持显式选择和拒绝语义。"""

    monkeypatch.setattr(
        service,
        "action_specs",
        lambda: {
            name: SimpleNamespace(version="test")
            for name in ("data", "det", "search", "compute")
        },
    )
    monkeypatch.setattr(service, "active_catalog_hash", lambda: "hash")
    fixed = _spec("fixed_source", "source")
    assert service._select_action_backend(
        "data", ActionRequest(backend_id="source", source_id="source"), fixed
    ) == ("source", "fixed_data_source")
    assert service._select_action_backend(
        "data", ActionRequest(backend_id="other"), fixed
    )["error"]["code"] == "invalid_request"
    assert service._select_action_backend(
        "data", ActionRequest(source_id="other"), fixed
    )["error"]["code"] == "invalid_request"
    assert service._select_action_backend(
        "data", ActionRequest(component_backends={"role": "source"}), fixed
    )["error"]["code"] == "invalid_request"

    deterministic = _spec("internal_deterministic", "internal")
    assert service._select_action_backend("det", ActionRequest(), deterministic) == (
        "internal",
        "internal_deterministic",
    )
    assert service._select_action_backend(
        "det", ActionRequest(source_id="source"), deterministic
    )["error"]["code"] == "invalid_request"

    source = _spec("agent_source_required", "source")
    assert service._select_action_backend(
        "search", ActionRequest(source_id="source"), source
    ) == ("source", "agent_data_source")
    assert service._select_action_backend("search", ActionRequest(), source)["status"] == "invalid_request"
    assert service._select_action_backend(
        "search", ActionRequest(source_id="source", backend_id="backend"), source
    )["status"] == "invalid_request"

    agent = _spec("agent_backend_required", "backend")
    assert service._select_action_backend(
        "compute", ActionRequest(backend_id="backend"), agent
    ) == ("backend", "agent")
    components = _spec("agent_components_required", "backend")
    assert service._select_action_backend(
        "compute", ActionRequest(backend_id="backend"), components
    ) == ("backend", "agent_components")
    assert service._select_action_backend("compute", ActionRequest(), agent)["status"] == "invalid_request"
    unknown = _spec("unknown_policy", "backend")
    assert service._select_action_backend("compute", ActionRequest(), unknown)["status"] == "invalid_request"


def test_service_component_contract_covers_required_and_allowed_branches(monkeypatch) -> None:
    """组件后端角色缺失、未知、不允许和成功路径都保持可诊断。"""

    monkeypatch.setattr(
        service,
        "action_specs",
        lambda: {
            "composite": SimpleNamespace(
                version="test", selection_policy="agent_backend_required"
            )
        },
    )
    monkeypatch.setattr(service, "active_catalog_hash", lambda: "hash")
    backend = SimpleNamespace(
        required_component_roles={"composite": ("energy",)},
        component_backend_options={"composite": {"energy": ("gaussian",)}},
    )
    backends = {"gaussian": object()}
    missing = service._resolve_action_component_backends(
        "composite", "composite_backend", ActionRequest(), backend, backends
    )
    assert isinstance(missing, dict)
    assert missing["error"]["code"] == "invalid_request"
    unknown = service._resolve_action_component_backends(
        "composite",
        "composite_backend",
        ActionRequest(component_backends={"energy": "unknown"}),
        backend,
        backends,
    )
    assert isinstance(unknown, dict)
    disallowed = service._resolve_action_component_backends(
        "composite",
        "composite_backend",
        ActionRequest(component_backends={"energy": "other"}),
        backend,
        {"other": object()},
    )
    assert isinstance(disallowed, dict)
    unsupported_role = service._resolve_action_component_backends(
        "composite",
        "composite_backend",
        ActionRequest(component_backends={"gradient": "gaussian"}),
        backend,
        backends,
    )
    assert isinstance(unsupported_role, dict)
    valid = service._resolve_action_component_backends(
        "composite",
        "composite_backend",
        ActionRequest(component_backends={"energy": "gaussian"}),
        backend,
        backends,
    )
    assert valid[1] is True
    assert valid[0] == (backends["gaussian"],)


def test_service_required_fields_and_discovery_batch_validation(monkeypatch) -> None:
    """必填科学字段和批请求四类拒绝都走稳定错误 envelope。"""

    monkeypatch.setattr(
        service,
        "action_specs",
        lambda: {
            "action": SimpleNamespace(
                version="test", selection_policy="agent_backend_required"
            )
        },
    )
    monkeypatch.setattr(service, "active_catalog_hash", lambda: "hash")
    backend = SimpleNamespace(
        required_method_fields={"action": ("method",)},
        required_setting_fields={"action": ("setting",)},
        allowed_method_values={},
        allowed_setting_values={},
    )
    request = ActionRequest()
    missing = service._validate_required_scientific_fields("action", "backend", request, backend)
    assert missing is not None
    complete = ActionRequest(method_spec={"method": "value"}, action_settings={"setting": "value"})
    assert service._validate_required_scientific_fields("action", "backend", complete, backend) is None

    valid = discovery_tools.ActionBatchRequest(
        action_id="action",
        backend_id="backend",
        items=[{"item_id": "one", "inputs": {}}],
    )
    monkeypatch.setattr(
        discovery_tools,
        "action_specs",
        lambda: {"action": SimpleNamespace(batch_safe=True, backend_ids=("backend",))},
    )
    assert discovery_tools._validate_action_batch_request(valid) is None


def test_service_scope_submission_key_attachment_covers_request_shapes() -> None:
    """scope 提交键在无键、有键和不适用请求上的传递保持稳定。"""

    request = ActionRequest()
    attached_request = service._attach_selected_submission_key(request, "scope-key")
    assert isinstance(attached_request, ActionRequest)
    assert attached_request.submission_key == "scope-key"
    existing_request = ActionRequest(submission_key="existing")
    assert service._attach_selected_submission_key(existing_request, "scope-key") is existing_request
    assert service._attach_selected_submission_key(request, None) is request

    attached_mapping = service._attach_selected_submission_key({}, "scope-key")
    assert attached_mapping == {"submission_key": "scope-key"}
    existing_mapping = {"submission_key": "existing"}
    assert service._attach_selected_submission_key(existing_mapping, "scope-key") is existing_mapping
    scalar = object()
    assert service._attach_selected_submission_key(scalar, "scope-key") is scalar
