"""MiniChem 同步 Action、工作区写入和 Artifact 声明的幂等边界测试。"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
import threading
from pathlib import Path

import pytest

from minichem_mcp_tools import open_execution
from minichem_mcp_tools import open_tools
from minichem_mcp_tools.execution_models import (
    ArtifactDeclarationRequest,
    VerifyExecutionReferencesRequest,
    WorkspaceTextWriteRequest,
)
from minichem_toolbox.idempotency import (
    OperationIdempotencyStore,
    calculate_request_fingerprint,
    scoped_submission_operation_id,
)
import minichem_toolbox.idempotency as idempotency
from minichem_toolbox import service
from minichem_toolbox.models import ActionRequest
from minichem_toolbox.scope import ScopeContext, activate_scope


def _action_request() -> dict[str, object]:
    return {
        "backend_id": "openbabel",
        "inputs": {"structure": {"smiles": "C"}},
        "method_spec": {},
        "action_settings": {},
        "resource_limits": {"cpu_cores": 1, "memory_mb": 128, "gpu_count": 0},
    }


def _pysisyphus_scan_request() -> dict[str, object]:
    """构造使用服务端 CPU 上限的 pysisyphus 同步 Action 请求。"""

    return {
        "backend_id": "pysisyphus",
        "inputs": {
            "structure": {
                "atoms": [
                    {"element": "C", "position_angstrom": [0.0, 0.0, 0.0]},
                    {"element": "C", "position_angstrom": [1.4, 0.0, 0.0]},
                    {"element": "N", "position_angstrom": [2.1, 1.1, 0.0]},
                    {"element": "H", "position_angstrom": [2.7, 1.2, 0.9]},
                ],
                "bonds": [
                    {"atom_index_a": 0, "atom_index_b": 1, "bond_order": 1.0},
                    {"atom_index_a": 1, "atom_index_b": 2, "bond_order": 1.0},
                    {"atom_index_a": 2, "atom_index_b": 3, "bond_order": 1.0},
                ],
                "charge": 0,
                "multiplicity": 1,
                "pbc": [False, False, False],
            }
        },
        "method_spec": {"calculator_backend": "xtb", "method": "GFN2-xTB"},
        "action_settings": {
            "coordinate_type": "dihedral",
            "atom_indices": [0, 1, 2, 3],
            "start_value": 0.0,
            "end_value": 60.0,
            "value_unit": "degree",
            "steps": 12,
            "optimizer": "rfo",
            "convergence": "gau",
            "max_cycles": 200,
            "hessian_init": "xtb",
        },
        "resource_limits": {"cpu_cores": 1, "memory_mb": 128, "gpu_count": 0},
    }


@pytest.mark.parametrize("identity_kind", ["operation", "scope"])
def test_sync_idempotency_binds_selected_backend_cpu_cap_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    identity_kind: str,
) -> None:
    """operation/scope 同步 Action 在执行前都必须使用服务端 backend 上限。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    observed_cpu_cores: list[int] = []

    monkeypatch.setattr(service, "_validate_action_backend_health", lambda _plan: None)

    def invoke_worker(plan):
        observed_cpu_cores.append(plan.request.resource_limits.cpu_cores)
        return service._WorkerExecution(
            {
                "status": "failed",
                "error": {
                    "code": "test_worker_failure",
                    "message": "测试 worker 未执行科学计算",
                },
            },
            {},
        )

    monkeypatch.setattr(service, "_invoke_action_worker", invoke_worker)
    request = _pysisyphus_scan_request()

    if identity_kind == "operation":
        response = service.execute_action(
            "scan_reaction_coordinates",
            request,
            operation_id="pysisyphus-operation",
        )
    else:
        scope = ScopeContext(
            task_id="task",
            batch_id="batch",
            combination_id="combination",
            scope_id="scope",
            workspace=tmp_path,
            capability_hash="capability",
        )
        with activate_scope(scope):
            response = service.execute_action(
                "scan_reaction_coordinates",
                request,
                submission_key="pysisyphus-scope",
            )

    assert response["status"] == "failed"
    assert response["error"]["code"] == "test_worker_failure"
    assert observed_cpu_cores == [48]


def test_action_response_loss_is_replayed_without_second_side_effect(
    tmp_path, monkeypatch
) -> None:
    """完成 receipt 已落盘后，第二次调用只返回原响应。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    calls: list[tuple[str, dict[str, object]]] = []

    def execute_once(action_id, request, *, resources_bound=False):
        calls.append((action_id, dict(request)))
        return {
            "status": "success",
            "action": action_id,
            "result": {"side_effect_count": len(calls)},
        }

    monkeypatch.setattr(service, "_execute_action_once", execute_once)
    first = service.execute_action(
        "convert_structure",
        _action_request(),
        operation_id="execute-operation-1",
    )
    second = service.execute_action(
        "convert_structure",
        _action_request(),
        operation_id="execute-operation-1",
    )

    assert first == second
    assert len(calls) == 1
    receipts = list((tmp_path / "_tool_artifacts").rglob("*.json"))
    assert any(
        json.loads(path.read_text(encoding="utf-8")).get("status") == "completed"
        for path in receipts
    )


def test_unknown_worker_state_keeps_idempotent_receipt_running(
    tmp_path,
    monkeypatch,
) -> None:
    """worker 终止未知时保留 running receipt，后续调用只能技术暂停。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    calls: list[int] = []

    def execute_once(_action_id, _request, *, resources_bound=False):
        calls.append(1)
        return {
            "status": "failed",
            "action": "convert_structure",
            "provenance": {"resource_state": "unknown"},
            "retryable": True,
        }

    monkeypatch.setattr(service, "_execute_action_once", execute_once)
    request = _action_request()
    first = service.execute_action(
        "convert_structure",
        request,
        operation_id="unknown-worker-operation",
    )

    assert first["status"] == "failed"
    receipts = list((tmp_path / "_tool_artifacts").rglob("*.json"))
    assert len(receipts) == 1
    assert json.loads(receipts[0].read_text(encoding="utf-8"))["status"] == "running"

    replay = service.execute_action(
        "convert_structure",
        request,
        operation_id="unknown-worker-operation",
    )
    assert replay["status"] == "technical_pause"
    assert replay["error"]["code"] == "operation_in_progress"
    assert calls == [1]


def test_action_identity_conflict_and_orphan_running_never_execute_again(
    tmp_path, monkeypatch
) -> None:
    """同身份改请求或遗留 running 都只能形成技术暂停。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    calls: list[int] = []

    def execute_once(_action_id, _request, *, resources_bound=False):
        calls.append(1)
        return {"status": "success", "result": {"count": len(calls)}}

    monkeypatch.setattr(service, "_execute_action_once", execute_once)
    service.execute_action(
        "convert_structure",
        _action_request(),
        operation_id="conflicting-operation",
    )
    conflict_request = _action_request()
    conflict_request["inputs"] = {"structure": {"smiles": "N"}}
    conflict = service.execute_action(
        "convert_structure",
        conflict_request,
        operation_id="conflicting-operation",
    )
    assert conflict["status"] == "technical_pause"
    assert conflict["error"]["code"] == "operation_id_conflict"
    assert len(calls) == 1

    running_operation = "orphan-running-operation"
    bound_request = service._bind_fixed_action_resources(
        ActionRequest.model_validate(_action_request())
    )
    OperationIdempotencyStore().write_running_operation_receipt(
        running_operation,
        calculate_request_fingerprint(
            "convert_structure",
            bound_request.model_dump(mode="json"),
        ),
    )
    running = service.execute_action(
        "convert_structure",
        _action_request(),
        operation_id=running_operation,
    )
    assert running["status"] == "technical_pause"
    assert running["error"]["code"] == "operation_in_progress"
    assert len(calls) == 1


def test_explicit_and_embedded_operation_identity_mismatch_is_rejected(
    tmp_path, monkeypatch
) -> None:
    """两处内部身份不一致时必须在执行副作用前返回 invalid_request。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    calls: list[int] = []
    monkeypatch.setattr(
        service,
        "_execute_action_once",
        lambda _action_id, _request: calls.append(1)
        or {"status": "success", "result": {}},
    )
    request = {**_action_request(), "operation_id": "embedded-operation"}

    result = service.execute_action(
        "convert_structure",
        request,
        operation_id="explicit-operation",
    )

    assert result["status"] == "invalid_request"
    assert result["error"]["code"] == "operation_id_conflict"
    assert calls == []


@pytest.mark.parametrize(
    ("operation_name", "request_arg"),
    [
        (
            "write_workspace_text",
            WorkspaceTextWriteRequest(
                path="code/input.inp",
                content="#p sp\n",
                overwrite=False,
                operation_id="write-operation-1",
            ),
        ),
        (
            "declare_scientific_artifact",
            ArtifactDeclarationRequest(
                path="outputs/result.xyz",
                semantic_type="AtomicStructure",
                media_type="chemical/x-xyz",
                producer_layer="native_software",
                producer_id="xtb",
                parent_artifact_ids=[],
                operation_id="artifact-operation-1",
            ),
        ),
    ],
)
def test_open_execution_side_effect_receipts_replay_identical_response(
    operation_name, request_arg, tmp_path, monkeypatch
) -> None:
    """写入和声明的相同请求重放同一响应，不重复执行后端副作用。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    calls: list[int] = []
    if operation_name == "write_workspace_text":
        monkeypatch.setattr(open_execution, "_write_workspace_text_once", lambda value: calls.append(1) or {"status": "success", "path": value.path, "sha256": "a" * 64})
    else:
        artifact_path = tmp_path / "outputs" / "result.xyz"
        artifact_path.parent.mkdir(mode=0o700)
        artifact_path.write_text("0 0 0\n", encoding="utf-8")
        monkeypatch.setattr(open_execution, "_declare_scientific_artifact_once", lambda value, **_kwargs: calls.append(1) or {"status": "success", "artifact": {"artifact_id": "art_test", "path": value.path}})
    function = getattr(open_execution, operation_name)
    first = function(request_arg)
    if operation_name == "declare_scientific_artifact":
        artifact_path.unlink()
    second = function(request_arg)
    assert first == second
    assert calls == [1]


def test_invalid_artifact_declaration_does_not_block_scoped_retry_or_inventory(
    tmp_path: Path,
) -> None:
    """确定性的声明校验失败不得留下 running 收据阻塞同提交键重试。"""

    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    scope = ScopeContext(
        task_id="task",
        batch_id="batch",
        combination_id="combination",
        scope_id="scope",
        workspace=workspace,
        capability_hash="capability",
    )
    submission_key = "artifact-retry-after-invalid"
    invalid_request = ArtifactDeclarationRequest(
        path="_tool_results/invalid-artifact.json",
        semantic_type="GaussianLog",
        media_type="text/plain",
        producer_layer="native_software",
        producer_id="gaussian",
        submission_key=submission_key,
    )
    valid_path = workspace / "code" / "gaussian.log"
    valid_path.parent.mkdir(mode=0o700)
    valid_path.write_text("SCF converged\n", encoding="utf-8")
    valid_request = invalid_request.model_copy(update={"path": "code/gaussian.log"})
    with activate_scope(scope):
        operation_id = scoped_submission_operation_id(submission_key)
        first = open_tools.declare_scientific_artifact(invalid_request)
        assert first["status"] == "invalid_request"
        assert first["error"]["code"] == "invalid_open_execution_request"
        assert OperationIdempotencyStore().read_operation_receipt(operation_id) is None

        second = open_tools.declare_scientific_artifact(valid_request)
        assert second["status"] == "success"
        artifact_id = second["artifact"]["artifact_id"]

        verification = open_execution.verify_execution_references(
            VerifyExecutionReferencesRequest(
                operations=[operation_id],
                artifacts=[artifact_id],
            )
        )

    # 直接声明的 Artifact 没有 Action job 的结果来源链；它可以保存在
    # ArtifactStore 中，但不能冒充 Investigation 结果的已验证执行产物。
    assert verification["all_valid"] is False
    assert verification["has_active_operations"] is False
    assert verification["operation_inventory_complete"] is True
    assert verification["scope_operation_inventory"] == [
        {"operation_id": operation_id, "status": "completed"}
    ]
    assert verification["items"][-1] == {
        "kind": "artifact",
        "status": "invalid",
        "reason_code": "producer_binding_unavailable",
    }


def test_running_receipt_cannot_overwrite_existing_operation(tmp_path: Path) -> None:
    """恢复探针不能覆盖既有 running/completed 状态。"""

    store = OperationIdempotencyStore(tmp_path / "operations")
    fingerprint = calculate_request_fingerprint("action", {"value": 1})
    store.write_running_operation_receipt("existing-operation", fingerprint)

    with pytest.raises(FileExistsError, match="收据已存在"):
        store.write_running_operation_receipt("existing-operation", fingerprint)


def test_corrupt_receipt_stops_without_being_misreported_as_identity_conflict(
    tmp_path: Path,
) -> None:
    """不可解析的收据必须报告损坏，且不得执行副作用。"""

    store = OperationIdempotencyStore(tmp_path / "operations")
    fingerprint = calculate_request_fingerprint("action", {"value": 1})
    store.write_running_operation_receipt("corrupt-operation", fingerprint)
    [receipt_path] = list((tmp_path / "operations").glob("*.json"))
    receipt_path.write_text("{not-json", encoding="utf-8")
    calls: list[int] = []

    result = store.execute_idempotent_operation(
        "corrupt-operation",
        fingerprint,
        lambda: calls.append(1) or {"status": "success"},
    )

    assert result["minichem_status"] == "operation_record_corrupt"
    assert result["error"]["code"] == "operation_record_corrupt"
    assert calls == []


def test_fifo_receipt_returns_corrupt_without_blocking_or_side_effect(
    tmp_path: Path,
) -> None:
    """FIFO 收据必须快速按损坏处理，且不得执行副作用。"""

    store = OperationIdempotencyStore(tmp_path / "operations")
    fingerprint = calculate_request_fingerprint("action", {"value": 1})
    receipt_path, _lock_path = store._paths("fifo-operation")
    os.mkfifo(receipt_path)
    calls: list[int] = []

    started = time.monotonic()
    result = store.execute_idempotent_operation(
        "fifo-operation",
        fingerprint,
        lambda: calls.append(1) or {"status": "success"},
    )
    elapsed = time.monotonic() - started

    assert elapsed < 1.0
    assert result["minichem_status"] == "operation_record_corrupt"
    assert result["error"]["code"] == "operation_record_corrupt"
    assert calls == []


@pytest.mark.parametrize("fingerprint", ["", "A" * 64, "f" * 63, True])
def test_invalid_fingerprint_is_rejected_before_receipt_write(
    tmp_path: Path,
    fingerprint,
) -> None:
    """只有规范的小写 SHA-256 可作为请求身份。"""

    store = OperationIdempotencyStore(tmp_path / "operations")
    with pytest.raises(ValueError, match="fingerprint"):
        store.execute_idempotent_operation(
            "invalid-fingerprint-operation",
            fingerprint,
            lambda: {"status": "success"},
        )


def test_operation_store_rejects_symlinked_root_component(tmp_path: Path) -> None:
    """收据根路径经过符号链接时不得写入链接目标。"""

    outside = tmp_path / "outside"
    outside.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeError, match="符号链接"):
        OperationIdempotencyStore(linked_parent / "operations")

    assert list(outside.iterdir()) == []


def test_operation_store_rejects_symlinked_lock_file(tmp_path: Path) -> None:
    """既有锁文件若是符号链接，调用不得打开其目标。"""

    root = tmp_path / "operations"
    store = OperationIdempotencyStore(root)
    _receipt_path, lock_path = store._paths("linked-lock-operation")
    target = tmp_path / "outside.lock"
    target.write_bytes(b"")
    lock_path.symlink_to(target)

    with pytest.raises(RuntimeError, match="锁文件不得是符号链接"):
        store.execute_idempotent_operation(
            "linked-lock-operation",
            calculate_request_fingerprint("action", {"value": 1}),
            lambda: {"status": "success"},
        )


def test_operation_store_rejects_symlinked_receipt_without_reading_target(
    tmp_path: Path,
) -> None:
    """收据文件若是符号链接，读取不得触及工作区外的目标。"""

    root = tmp_path / "operations"
    store = OperationIdempotencyStore(root)
    fingerprint = calculate_request_fingerprint("action", {"value": 1})
    receipt_path, _lock_path = store._paths("linked-receipt-operation")
    outside = tmp_path / "outside.json"
    outside_value = {
        "schema_version": 1,
        "operation_id": "linked-receipt-operation",
        "fingerprint": fingerprint,
        "status": "completed",
        "response": {"status": "success"},
    }
    outside.write_text(json.dumps(outside_value), encoding="utf-8")
    receipt_path.symlink_to(outside)
    calls: list[int] = []

    with pytest.raises(RuntimeError, match="收据文件不得是符号链接"):
        store.execute_idempotent_operation(
            "linked-receipt-operation",
            fingerprint,
            lambda: calls.append(1) or {"status": "success"},
        )

    assert calls == []
    assert json.loads(outside.read_text(encoding="utf-8")) == outside_value


def test_operation_store_rejects_symlinked_temporary_collision_without_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """临时文件名碰撞为符号链接时不得触碰链接目标。"""

    root = tmp_path / "operations"
    store = OperationIdempotencyStore(root)
    receipt_path, _lock_path = store._paths("temporary-link-operation")
    outside = tmp_path / "outside.json"
    outside.write_text("unchanged", encoding="utf-8")
    token = "a" * 24
    monkeypatch.setattr(idempotency.secrets, "token_hex", lambda _size: token)
    temporary = root / f".{receipt_path.name}.{os.getpid()}.{token}.tmp"
    temporary.symlink_to(outside)

    with pytest.raises(RuntimeError, match="临时文件不得是符号链接"):
        store.execute_idempotent_operation(
            "temporary-link-operation",
            calculate_request_fingerprint("action", {"value": 1}),
            lambda: {"status": "success"},
        )

    assert outside.read_text(encoding="utf-8") == "unchanged"


def test_concurrent_legacy_operation_executes_side_effect_exactly_once(
    tmp_path: Path,
) -> None:
    """同一 legacy operation 并发提交时只有一个调用进入副作用函数。"""

    store = OperationIdempotencyStore(tmp_path / "operations")
    fingerprint = calculate_request_fingerprint("action", {"value": 1})
    calls: list[int] = []
    calls_lock = threading.Lock()
    entered = threading.Event()

    def side_effect() -> dict[str, object]:
        with calls_lock:
            calls.append(1)
            count = len(calls)
        entered.set()
        time.sleep(0.05)
        return {"status": "success", "count": count}

    def submit() -> dict[str, object]:
        return store.execute_idempotent_operation(
            "concurrent-legacy-operation",
            fingerprint,
            side_effect,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(submit)
        assert entered.wait(timeout=2)
        second = executor.submit(submit)
        results = [first.result(timeout=2), second.result(timeout=2)]

    assert results[0] == results[1] == {"status": "success", "count": 1}
    assert calls == [1]
