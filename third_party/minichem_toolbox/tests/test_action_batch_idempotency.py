"""MCP Action 批量副作用的 operation 幂等与安全回放测试。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from types import SimpleNamespace

import pytest

from minichem_mcp_tools import discovery_tools
from minichem_mcp_tools.discovery_models import ActionBatchRequest
from minichem_toolbox.idempotency import (
    OperationIdempotencyStore,
    calculate_request_fingerprint,
)
from minichem_toolbox import service


@dataclass(frozen=True)
class _Budget:
    """为批量测试提供足够且确定的本地资源预算。"""

    cpu_cores: int = 8
    memory_mb: int = 8192
    gpu_count: int = 0

    def as_dict(self) -> dict[str, int]:
        """返回 discovery_tools 所需的 JSON 资源投影。"""

        return {
            "cpu_cores": self.cpu_cores,
            "memory_mb": self.memory_mb,
            "gpu_count": self.gpu_count,
        }


def _request(*, operation_id: str | None = "batch-operation") -> ActionBatchRequest:
    """构造两个独立输入的 batch 请求；可显式或省略内部身份。"""

    values: dict[str, object] = {
        "action_id": "test_batch_action",
        "backend_id": "test_backend",
        "items": [
            {"item_id": "item-a", "inputs": {"value": 1}},
            {"item_id": "item-b", "inputs": {"value": 2}},
        ],
    }
    if operation_id is not None:
        values["operation_id"] = operation_id
    return ActionBatchRequest.model_validate(values)


@pytest.fixture
def batch_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """隔离收据目录、目录能力和可观察的 service 子调用。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    specification = SimpleNamespace(batch_safe=True, backend_ids=("test_backend",))
    monkeypatch.setattr(
        discovery_tools,
        "action_specs",
        lambda: {"test_batch_action": specification},
    )
    monkeypatch.setattr(discovery_tools, "evaluation_resource_budget", lambda: _Budget())
    monkeypatch.setattr(
        discovery_tools,
        "active_resource_usage",
        lambda: {"cpu_cores": 0, "memory_mb": 0, "gpu_count": 0},
    )
    monkeypatch.setattr(discovery_tools, "active_resource_jobs", lambda: [])

    calls: list[tuple[str, str, dict[str, object]]] = []
    calls_lock = Lock()

    def execute_action(
        action_id: str,
        request: dict[str, object],
        *,
        operation_id: str | None = None,
    ) -> dict[str, object]:
        """记录子 operation，并返回可压缩的成功 ActionResult。"""

        with calls_lock:
            calls.append((action_id, str(operation_id), dict(request)))
        return {
            "status": "success",
            "action": action_id,
            "result": {"value": request["inputs"]},
            "output_artifacts": [],
        }

    monkeypatch.setattr(service, "execute_action", execute_action)
    return calls, tmp_path


def test_batch_schema_hides_internal_operation_id() -> None:
    """批量请求可接受内部身份，但公开 JSON Schema 不泄露该字段。"""

    schema = ActionBatchRequest.model_json_schema()
    assert "operation_id" not in json.dumps(schema, ensure_ascii=False)
    assert "operation_id" not in schema.get("properties", {})
    request = _request(operation_id="hidden-batch-operation")
    assert request.operation_id == "hidden-batch-operation"
    assert "operation_id" not in request.model_dump(mode="json")


def test_explicit_batch_replays_parent_and_uses_stable_child_operations(
    batch_environment,
) -> None:
    """相同父身份只执行一次，并按 item_id/index 固定子身份。"""

    calls, _tmp_path = batch_environment
    request = _request(operation_id="explicit-batch-operation")

    first = discovery_tools.submit_action_batch(request)
    second = discovery_tools.submit_action_batch(request)

    assert first == second
    assert len(calls) == 2
    assert {item[0] for item in calls} == {"test_batch_action"}
    child_operations = [item[1] for item in calls]
    assert len(set(child_operations)) == 2
    assert all(value.startswith("batch-item-") for value in child_operations)
    assert all(item[2]["inputs"] in ({"value": 1}, {"value": 2}) for item in calls)


def test_batch_children_keep_complete_action_results(batch_environment, monkeypatch) -> None:
    """批量 wrapper 保留每个 child 的未知字段、完整数组和全部 ArtifactRef。"""

    calls, _tmp_path = batch_environment
    artifacts = [
        {
            "artifact_id": "art_" + "1" * 32,
            "semantic_type": "PrimaryResult",
            "media_type": "application/json",
            "sha256": "a" * 64,
            "path": "_tool_artifacts/objects/primary.json",
            "producer_action": "test_batch_action",
            "producer_backend": "test_backend",
            "parent_artifact_ids": [],
        },
        {
            "artifact_id": "art_" + "2" * 32,
            "semantic_type": "AuxiliaryResult",
            "media_type": "application/json",
            "sha256": "b" * 64,
            "path": "_tool_artifacts/objects/auxiliary.json",
            "producer_action": "test_batch_action",
            "producer_backend": "test_backend",
            "parent_artifact_ids": [],
        },
    ]

    def execute_action(
        action_id: str,
        request: dict[str, object],
        *,
        operation_id: str | None = None,
    ) -> dict[str, object]:
        """返回包含大数组和未登记字段的规范结果。"""

        calls.append((action_id, str(operation_id), dict(request)))
        return {
            "status": "success",
            "action": action_id,
            "result": {
                "unknown_result_field": {"item": request["inputs"]},
                "all_candidates": list(range(161)),
            },
            "output_artifacts": artifacts,
            "unknown_top_level_field": "preserved",
        }

    monkeypatch.setattr(service, "execute_action", execute_action)
    result = discovery_tools.submit_action_batch(_request(operation_id="complete-batch"))

    assert result["status"] == "success"
    assert len(result["items"]) == 2
    for child in result["items"]:
        child_result = child["result"]
        assert child_result["result"]["all_candidates"] == list(range(161))
        assert child_result["result"]["unknown_result_field"]
        assert child_result["output_artifacts"] == artifacts
        assert child_result["unknown_top_level_field"] == "preserved"


def test_missing_identity_fails_closed_without_request_deduplication(
    batch_environment,
) -> None:
    """缺少内部身份时拒绝批量执行，不按请求内容创建永久收据。"""

    calls, tmp_path = batch_environment
    request = _request(operation_id=None)

    first = discovery_tools.submit_action_batch(request)
    second = discovery_tools.submit_action_batch(request)

    assert first["status"] == "invalid_request"
    assert first["error"]["code"] == "missing_operation_id"
    assert second == first
    assert calls == []
    assert not list((tmp_path / "_tool_artifacts" / "operations").glob("*.json"))


def test_structured_action_failure_completes_batch_without_transport_error(
    batch_environment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """结构化 ActionResult 失败可作为批结果，并正常完成父收据。"""

    calls, tmp_path = batch_environment

    def execute_action(
        action_id: str,
        request: dict[str, object],
        *,
        operation_id: str | None = None,
    ) -> dict[str, object]:
        calls.append((action_id, str(operation_id), dict(request)))
        if request["inputs"] == {"value": 1}:
            return {
                "status": "failed",
                "action": action_id,
                "error": {"code": "backend_failed", "message": "计算失败"},
                "output_artifacts": [],
            }
        return {
            "status": "success",
            "action": action_id,
            "result": {"value": request["inputs"]},
            "output_artifacts": [],
        }

    # 该替身返回规范 ActionResult 失败；只有异常才应穿透到父收据边界。
    # 这里不使用 fixture 中带锁的记录器，以便保留失败结果的完整断言。
    monkeypatch.setattr(service, "execute_action", execute_action)
    request = _request(operation_id="structured-failure-operation")
    first = discovery_tools.submit_action_batch(request)
    second = discovery_tools.submit_action_batch(request)

    assert first["status"] == "partial_success"
    assert first["successful_item_count"] == 1
    assert first["failed_item_count"] == 1
    assert any(
        item["result"]["status"] == "failed"
        for item in first["items"]
    )
    assert second == first
    assert len(calls) == 2
    receipt_files = list((tmp_path / "_tool_artifacts" / "operations").glob("*.json"))
    assert len(receipt_files) == 1
    receipt = json.loads(receipt_files[0].read_text(encoding="utf-8"))
    assert receipt["status"] == "completed"


def test_transport_exception_keeps_parent_running_and_replay_is_blocked(
    batch_environment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """子项传输异常穿透父收据，重放只暂停且不产生新副作用。"""

    calls, tmp_path = batch_environment

    def execute_action(
        action_id: str,
        request: dict[str, object],
        *,
        operation_id: str | None = None,
    ) -> dict[str, object]:
        calls.append((action_id, str(operation_id), dict(request)))
        if request["inputs"] == {"value": 1}:
            raise RuntimeError("backend transport disconnected")
        return {
            "status": "success",
            "action": action_id,
            "result": {"value": request["inputs"]},
            "output_artifacts": [],
        }

    monkeypatch.setattr(service, "execute_action", execute_action)
    request = _request(operation_id="transport-failure-operation")
    with pytest.raises(RuntimeError, match="transport disconnected"):
        discovery_tools.submit_action_batch(request)

    assert len(calls) == 2
    receipt_files = list(
        (tmp_path / "_tool_artifacts" / "operations").glob("*.json")
    )
    assert len(receipt_files) == 1
    receipt = json.loads(receipt_files[0].read_text(encoding="utf-8"))
    assert receipt["status"] == "running"

    replay = discovery_tools.submit_action_batch(request)

    assert replay["status"] == "technical_pause"
    assert replay["error"]["code"] == "operation_in_progress"
    assert len(calls) == 2


def test_same_batch_identity_with_changed_request_fails_closed(
    batch_environment,
) -> None:
    """已完成父收据遇到改请求时只返回 identity conflict，不执行子项。"""

    calls, _tmp_path = batch_environment
    request = _request(operation_id="conflicting-batch-operation")
    discovery_tools.submit_action_batch(request)

    changed = _request(operation_id="conflicting-batch-operation")
    changed.items[0].inputs["value"] = 999
    result = discovery_tools.submit_action_batch(changed)

    assert result["status"] == "technical_pause"
    assert result["minichem_status"] == "operation_id_conflict"
    assert len(calls) == 2


@pytest.mark.parametrize("receipt_state", ["running", "corrupt"])
def test_running_or_corrupt_parent_receipt_never_reexecutes(
    batch_environment,
    receipt_state: str,
) -> None:
    """父收据遗留或损坏时沿用幂等存储的 fail-closed 行为。"""

    calls, tmp_path = batch_environment
    request = _request(operation_id="unsafe-parent-operation")
    canonical_request = request.model_dump(mode="json")
    canonical_request.pop("submission_key", None)
    fingerprint = calculate_request_fingerprint(
        "submit_action_batch", canonical_request
    )
    store = OperationIdempotencyStore()
    store.write_running_operation_receipt("unsafe-parent-operation", fingerprint)
    if receipt_state == "corrupt":
        [receipt_path] = (tmp_path / "_tool_artifacts" / "operations").glob("*.json")
        receipt_path.write_text("{not-json", encoding="utf-8")

    result = discovery_tools.submit_action_batch(request)

    assert result["status"] == "technical_pause"
    expected_code = (
        "operation_in_progress" if receipt_state == "running" else "operation_record_corrupt"
    )
    assert result["error"]["code"] == expected_code
    assert calls == []
