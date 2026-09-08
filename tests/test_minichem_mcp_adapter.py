"""MiniChem MCP adapter 的确定性协议合同测试。

这些测试只使用 ``tests.fakes.minichem_mcp.FakeMcpSession``，不启动真实 MCP、
Gaussian 或任何外部化学程序。它们验证 ARCHE adapter 的边界：目录冻结、显式
Action/Backend 请求、统一终态、作业身份恢复、ArtifactRef 传递和可诊断错误。
Program Gate 与 Router 属于上层工作流，本文件只确认 adapter 不会主动绕过它们。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import threading
from copy import deepcopy
from pathlib import Path

import pytest

import chemistry_multiagent.tools.minichem_adapter as minichem_adapter_module
from chemistry_multiagent.tools.minichem_adapter import (
    _AsyncSessionBridge,
    _CombinedMcpContext,
    MiniChemActionRequest,
    MiniChemCatalogSnapshot,
    MiniChemMcpAdapter,
    MiniChemMcpError,
    MiniChemMcpProtocolError,
)
from chemistry_multiagent.utils.api_retry import (
    is_retryable_api_exception,
)
from tests.fakes.minichem_mcp import (
    FakeMcpSession,
    FakeToolCall,
    build_artifact_ref,
    build_catalog_snapshot,
)
from minichem_mcp_tools.result_transport import compact_action_result


def _action_request() -> MiniChemActionRequest:
    """构造所有科学选择均显式给出的最小 Action 请求。"""

    return MiniChemActionRequest(
        action_id="convert_structure",
        backend_id="openbabel",
        inputs={"structure": {"smiles": "O=Cc1ccccc1"}},
        method_spec={},
        action_settings={},
        resource_limits={"cpu_cores": 1, "memory_mb": 128, "gpu_count": 0},
    )


@pytest.fixture
def adapter_factory(tmp_path: Path):
    """返回隔离的 adapter 与 fake session，测试结束后关闭会话线程。"""

    sessions: list[FakeMcpSession] = []

    def create(
        session: FakeMcpSession | None = None,
        *,
        operation_root: Path | None = None,
        retry_sleeper=None,
    ) -> tuple[MiniChemMcpAdapter, FakeMcpSession]:
        value = session or FakeMcpSession()
        sessions.append(value)
        adapter = MiniChemMcpAdapter(
            session_factory=lambda value=value: value,
            operation_root=operation_root or (tmp_path / "operations"),
            poll_interval=0.0,
            # API 重试测试单独注入 sleep；本协议回归测试不应真实等待固定退避。
            retry_sleeper=retry_sleeper or (lambda _delay: None),
        )
        adapter.open()
        return adapter, value

    yield create


def _close(adapter: MiniChemMcpAdapter) -> None:
    """显式关闭 adapter，避免测试留下会话线程。"""

    adapter.close()


def test_catalog_snapshot_hash_and_projection_are_frozen(adapter_factory) -> None:
    """目录首次读取后冻结版本/哈希，Planner 只能获得服务投影。"""

    session = FakeMcpSession()
    adapter, _ = adapter_factory(session)
    try:
        first = adapter.discover_catalog()
        assert isinstance(first, MiniChemCatalogSnapshot)
        assert first.catalog_hash == session.catalog["catalog_hash"]
        assert len(first.catalog_hash) == 64
        assert first.catalog_version == "7"
        assert first.snapshot["exposure_policy"] == "progressive_discovery"
        assert "exposure" not in first.snapshot
        assert "policies" not in first.snapshot
        assert first.snapshot["evaluation_resource_budget"]["scope"] == "per_task"
        assert first.snapshot["execution_layers"][0]["id"] == "predefined_actions"
        assert first.planner_projection()["catalog_hash"] == first.catalog_hash
        assert first.planner_projection()["actions"][0]["action_id"] == "convert_structure"

        # 修改服务端对象后，默认读取仍返回首次冻结的内容；refresh 才观察新版本。
        session.catalog = build_catalog_snapshot(
            actions=[
                {
                    "id": "convert_structure",
                    "version": "1.0.1",
                    "category": "structure_and_system",
                    "description": "Changed deterministic action.",
                    "backend_ids": ["openbabel"],
                    "required_inputs": ["structure"],
                    "optional_inputs": [],
                    "selection_policy": "agent_backend_required",
                    "primary_output": "AtomicStructure",
                    "data_action": False,
                    "batch_safe": False,
                }
            ]
        )
        frozen = adapter.discover_catalog()
        assert frozen.catalog_hash == first.catalog_hash
        assert frozen.catalog_version == first.catalog_version
        refreshed = adapter.discover_catalog(refresh=True)
        assert refreshed.catalog_hash == session.catalog["catalog_hash"]
        assert refreshed.catalog_hash != first.catalog_hash
        assert [call.name for call in session.calls] == []
    finally:
        _close(adapter)


def test_catalog_snapshot_reads_are_detached_from_frozen_content(adapter_factory) -> None:
    """调用方修改 snapshot 读出的嵌套值后，冻结 hash 和 Planner 投影仍保持一致。"""

    adapter, _ = adapter_factory()
    try:
        catalog = adapter.discover_catalog()
        original_hash = catalog.catalog_hash
        original_projection = catalog.planner_projection()

        with pytest.raises(TypeError):
            catalog.snapshot["actions"] = []
        detached_actions = catalog.snapshot["actions"]
        detached_actions[0]["id"] = "tampered_action"
        detached_budget = catalog.snapshot["evaluation_resource_budget"]
        detached_budget["memory_mb"] = 1

        assert catalog.catalog_hash == original_hash
        assert catalog.planner_projection() == original_projection
        assert catalog.actions[0]["id"] == "convert_structure"
        assert catalog.snapshot["evaluation_resource_budget"]["memory_mb"] == 320000
    finally:
        _close(adapter)


def test_discover_action_contracts_reads_each_action_backend_pair(adapter_factory) -> None:
    """Action Catalog 只声明可选 Backend，精确请求合同必须按 pair 读取。"""

    adapter, session = adapter_factory()
    try:
        contracts = adapter.discover_action_contracts()
        assert contracts["schema_version"] == 1
        assert contracts["contracts"] == [
            {
                "action_id": "convert_structure",
                "backend_id": "openbabel",
                "request_contract": {
                    "required_contract": {
                        "inputs": [
                            {
                                "name": "structure",
                                "required": True,
                                "type": "AtomicStructure",
                            }
                        ],
                        "method_spec": [],
                        "action_settings": [],
                        "component_backends": [],
                    }
                },
                "output_contract": None,
            }
        ]
        assert [call.name for call in session.calls] == [
            "inspect_action",
        ]
        assert session.calls[0].arguments["request"] == {
            "action_id": "convert_structure",
            "backend_id": "openbabel",
            "detail_level": "contract",
        }
    finally:
        _close(adapter)


def test_discover_action_contracts_keeps_output_contract_from_selected_request(
    adapter_factory,
) -> None:
    """MiniChem contract 的 output_contract 位于 selected_request_contract 内。"""

    session = FakeMcpSession(
        tool_results={
            "inspect_action": {
                "status": "success",
                "catalog_hash": build_catalog_snapshot()["catalog_hash"],
                "catalog_version": "7",
                "action": {
                    "id": "convert_structure",
                    "backend_ids": ["openbabel"],
                },
                "selected_request_contract": {
                    "sections": {"inputs": {"required": ["structure"]}},
                    "output_contract": {
                        "primary_output": "AtomicStructure",
                        "result_envelope": "ActionResult",
                    },
                },
            }
        }
    )
    adapter, _ = adapter_factory(session)
    try:
        contract = adapter.discover_action_contracts()["contracts"][0]
        assert contract["output_contract"] == {
            "primary_output": "AtomicStructure",
            "result_envelope": "ActionResult",
        }
    finally:
        _close(adapter)


def test_inspect_action_and_backend_forward_exact_requests(adapter_factory) -> None:
    """inspect_action/backend 只转发显式 ID，不读取旧 Tool Card 或自动选 provider。"""

    adapter, session = adapter_factory()
    try:
        action = adapter.inspect_action("convert_structure", "openbabel")
        backend = adapter.inspect_backend("openbabel", "convert_structure")

        assert action["action"]["id"] == "convert_structure"
        assert action["action"]["category"] == "structure_and_system"
        assert backend["backend"]["backend_id"] == "openbabel"
        assert backend["capabilities"][0]["action_id"] == "convert_structure"
        assert session.calls[0].name == "inspect_action"
        assert session.calls[0].arguments == {
            "request": {
                "action_id": "convert_structure",
                "backend_id": "openbabel",
                "detail_level": "contract",
            },
        }
        assert session.calls[1].name == "inspect_backend"
        assert session.calls[1].arguments == {
            "request": {
                "backend_id": "openbabel",
                "action_id": "convert_structure",
            },
        }
    finally:
        _close(adapter)


@pytest.mark.parametrize(
    ("minichem_status", "expected_error_code"),
    [
        ("success", None),
        ("partial_success", None),
        ("failed", "backend_unavailable"),
    ],
)
def test_action_statuses_map_to_arche_terminal_contract(
    adapter_factory,
    minichem_status: str,
    expected_error_code: str | None,
) -> None:
    """同步 Action 的 success/partial_success 与失败状态归一为 ARCHE 终态。"""

    result = {
        "status": minichem_status,
        "action": "convert_structure",
        "action_version": "1.0.0",
        "requested_backend": "openbabel",
        "backend": "openbabel",
        "selection_source": "agent",
        "result": {"converted": True}
        if minichem_status in {"success", "partial_success"}
        else None,
        "provenance": {"trace_id": f"trace-{minichem_status}"},
    }
    if expected_error_code:
        result["error"] = {"code": expected_error_code, "message": "deterministic test"}
    session = FakeMcpSession(action_results={"convert_structure": result})
    adapter, _ = adapter_factory(session)
    try:
        normalized = adapter.execute_action(_action_request())
        assert normalized["status"] == (
            "completed"
            if minichem_status in {"success", "partial_success"}
            else "technical_pause"
        )
        assert normalized["minichem_status"] == minichem_status
        assert normalized["provenance"]["trace_id"] == f"trace-{minichem_status}"
        assert session.calls[0].name == "execute_action"
        assert "operation_id" not in session.calls[0].arguments["request"]
        assert session.calls[0].arguments["request"]["action_id"] == "convert_structure"
        assert session.calls[0].arguments["request"]["backend_id"] == "openbabel"
        if expected_error_code:
            assert normalized["error"]["code"] == expected_error_code
    finally:
        _close(adapter)


def test_adapter_accepts_compacted_generic_result_with_full_artifact_refs(
    adapter_factory,
) -> None:
    """真实 Adapter.execute_action 接口接受通用 Action 的有界公开响应。"""

    catalog = build_catalog_snapshot()
    action_id = "convert_structure"
    backend_id = "openbabel"
    artifact = build_artifact_ref(
        artifact_id="art_" + "1" * 32,
        semantic_type="AtomicStructure",
        media_type="chemical/x-xyz",
        path="_tool_artifacts/objects/art_11111111111111111111111111111111.xyz",
        producer_action=action_id,
        producer_backend=backend_id,
    )
    public_result = compact_action_result(
        {
            "status": "success",
            "action": action_id,
            "action_version": "1.0.0",
            "requested_backend": backend_id,
            "backend": backend_id,
            "selection_source": "agent",
            "result": {
                "converted": True,
                "output_format": "xyz",
            },
            "output_artifacts": [artifact],
            "provenance": {"catalog_hash": catalog["catalog_hash"]},
            "retryable": False,
        }
    )
    session = FakeMcpSession(catalog=catalog, action_results={action_id: public_result})
    adapter, _ = adapter_factory(session)
    try:
        adapter.discover_catalog()
        request = MiniChemActionRequest(
            action_id=action_id,
            backend_id=backend_id,
            inputs={"structure": {"smiles": "CO"}},
            method_spec={},
            action_settings={},
            resource_limits={"cpu_cores": 1},
        )
        normalized = adapter.execute_action(request)
        assert normalized["status"] == "completed"
        assert normalized["output_artifacts"] == [artifact]
        assert normalized["provenance"] == {"catalog_hash": catalog["catalog_hash"]}
    finally:
        _close(adapter)


def test_adapter_reads_generic_large_result_page_and_arbitrary_item(
    adapter_factory,
) -> None:
    """Adapter 通过完整结果 Artifact 读取通用大数组的字符页和任意单项。"""

    action_id = "convert_structure"
    backend_id = "openbabel"
    items = []
    for index in range(65):
        item_artifact = build_artifact_ref(
            artifact_id=f"art_{index:032x}",
            semantic_type="XYZFile",
            media_type="chemical/x-xyz",
            path=(
                "_tool_artifacts/objects/"
                f"art_{index:032x}.json"
            ),
            producer_action=action_id,
            producer_backend=backend_id,
        )
        items.append(
            {
                "item_id": f"structure_{index:04d}",
                "item_index": index,
                "artifact": item_artifact,
            }
        )
    stored = json.dumps(
        {"items": items},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    artifact_path = "_tool_artifacts/objects/" + "art_" + "1" * 32 + ".json"
    artifact = build_artifact_ref(
        content=stored.encode("utf-8"),
        artifact_id="art_" + "1" * 32,
        semantic_type="ActionResult",
        media_type="application/json",
        path=artifact_path,
        producer_action=action_id,
        producer_backend=backend_id,
    )
    public_result = compact_action_result(
        {
            "status": "success",
            "action": action_id,
            "action_version": "1.0.0",
            "requested_backend": backend_id,
            "backend": backend_id,
            "selection_source": "agent",
            "result": {"items": items},
            "output_artifacts": [artifact],
            "provenance": {"catalog_hash": "a" * 64},
            "retryable": False,
        }
    )
    session = FakeMcpSession(
        workspace_files={artifact_path: stored},
    )
    adapter, _ = adapter_factory(session)
    try:
        pointer = public_result["result"]
        assert pointer["inline"] is False
        assert pointer["artifact_id"] == artifact["artifact_id"]
        page = adapter.read_workspace_text(
            artifact_path,
            json_pointer="/items",
            offset=0,
            max_chars=1_000,
            expected_sha256=artifact["sha256"],
        )
        assert page["offset"] == 0
        assert page["json_pointer"] == "/items"
        assert page["truncated"] is True
        assert page["next_offset"] == 1_000

        item = adapter.read_workspace_text(
            artifact_path,
            json_pointer="/items/64",
            max_chars=2_000,
            expected_sha256=artifact["sha256"],
        )
        assert json.loads(item["content"])["item_id"] == "structure_0064"
    finally:
        _close(adapter)


def test_failed_action_with_empty_artifacts_stays_diagnostic_and_has_no_identity(
    adapter_factory,
) -> None:
    """失败 Action 可以明确返回空 ArtifactRef，不得被伪造成成功或作业。"""

    session = FakeMcpSession(
        action_results={
            "convert_structure": {
                "status": "failed",
                "action": "convert_structure",
                "action_version": "1.0.0",
                "requested_backend": "openbabel",
                "backend": "openbabel",
                "selection_source": "agent",
                "result": None,
                "output_artifacts": [],
                "error": {
                    "code": "validation_failed",
                    "message": "deterministic input validation failure",
                },
            }
        }
    )
    adapter, _ = adapter_factory(session)
    try:
        result = adapter.execute_action(_action_request())
        assert result["status"] == "technical_pause"
        assert result["minichem_status"] == "failed"
        assert result["output_artifacts"] == []
        assert result["error"]["code"] == "validation_failed"
        assert "operation_id" not in result
        assert "job_id" not in result
    finally:
        _close(adapter)


def test_submit_wait_get_collect_use_the_same_job_identity(adapter_factory) -> None:
    """native submit→queued→get→collect 始终围绕同一个 job_id。"""

    job_id = "job_00000000000000000000000000000001"
    session = FakeMcpSession(
        job_states={
            job_id: [
                {
                    "status": "success",
                    "job": {"job_id": job_id, "status": "queued"},
                    "terminal": False,
                },
                {
                    "status": "success",
                    "job": {"job_id": job_id, "status": "success"},
                    "terminal": True,
                },
            ]
        }
    )
    adapter, _ = adapter_factory(session)
    try:
        submitted = adapter.submit_native_job(
            {"software_id": "xtb", "executable": "xtb", "arguments": ["input.xyz"]},
            "native-operation-1",
        )
        assert submitted["job_id"] == job_id
        assert submitted["status"] == "waiting"
        assert submitted["minichem_status"] == "queued"
        assert submitted["operation_id"] == "native-operation-1"

        waiting = adapter.get_execution_job(job_id)
        completed = adapter.get_execution_job(job_id)
        collected = adapter.collect_execution_job(job_id)

        assert waiting["status"] == "waiting"
        assert completed["status"] == "completed"
        assert collected["status"] == "completed"
        assert waiting["job_id"] == completed["job_id"] == collected["job_id"] == job_id
        assert session.native_submit_count == 1
        assert session.get_count[job_id] == 2
        assert session.collect_count[job_id] == 1
        names = [call.name for call in session.calls]
        assert names == [
            "submit_native_job",
            "get_execution_job",
            "get_execution_job",
            "collect_execution_job",
        ]
    finally:
        _close(adapter)


def test_resume_from_persisted_operation_does_not_submit_again(
    adapter_factory,
    tmp_path: Path,
) -> None:
    """新 adapter 从 operation 收据恢复时只 get/collect，不重复触发 submit 副作用。"""

    operation_root = tmp_path / "persisted-operations"
    job_id = "job_00000000000000000000000000000001"
    session = FakeMcpSession(
        job_states={
            job_id: [
                {
                    "status": "success",
                    "job": {"job_id": job_id, "status": "queued"},
                    "terminal": False,
                },
                {
                    "status": "success",
                    "job": {"job_id": job_id, "status": "success"},
                    "terminal": True,
                },
            ]
        }
    )
    first, _ = adapter_factory(session, operation_root=operation_root)
    try:
        submitted = first.submit_native_job(
            {"software_id": "xtb", "executable": "xtb", "arguments": ["input.xyz"]},
            "resume-operation-1",
        )
        assert submitted["job_id"] == job_id
    finally:
        _close(first)

    second, _ = adapter_factory(session, operation_root=operation_root)
    try:
        waiting = second.resume_operation("resume-operation-1", collect=True)
        completed = second.resume_operation("resume-operation-1", collect=True)
        assert waiting["status"] == "waiting"
        assert completed["status"] == "completed"
        assert completed["job_id"] == job_id
        assert session.native_submit_count == 1
        assert session.get_count[job_id] == 2
        assert session.collect_count[job_id] == 1
        record = json.loads((operation_root / "resume-operation-1.json").read_text())
        assert record["operation_id"] == "resume-operation-1"
        assert record["job_id"] == job_id
    finally:
        _close(second)


def test_load_operation_rejects_existing_symlink_without_reading_external_target(
    adapter_factory,
    tmp_path: Path,
) -> None:
    """读取既有符号链接时立即失败，不能把外部文件当作 operation 收据。"""

    operation_root = tmp_path / "existing-receipt-symlink"
    operation_root.mkdir()
    external = tmp_path / "external-receipt.json"
    external_content = '{"operation_id":"external-operation"}\n'
    external.write_text(external_content, encoding="utf-8")
    (operation_root / "symlink-operation.json").symlink_to(external)
    adapter, session = adapter_factory(operation_root=operation_root)
    try:
        with pytest.raises(MiniChemMcpError, match="operation 收据文件不得是符号链接"):
            adapter._load_operation("symlink-operation")
        assert external.read_text(encoding="utf-8") == external_content
        assert session.calls == []
    finally:
        _close(adapter)


def test_load_operation_rejects_symlink_replacement_before_safe_open(
    adapter_factory,
    monkeypatch,
    tmp_path: Path,
) -> None:
    """收据在检查后被替换为符号链接时，安全相对打开必须拒绝读取。"""

    operation_root = tmp_path / "receipt-race"
    operation_root.mkdir()
    operation_name = "raced-operation.json"
    local_receipt = operation_root / operation_name
    local_receipt.write_text(
        '{"operation_id":"local-operation"}\n',
        encoding="utf-8",
    )
    external = tmp_path / "external-race-receipt.json"
    external_content = '{"operation_id":"external-operation"}\n'
    external.write_text(external_content, encoding="utf-8")
    original_open = minichem_adapter_module.os.open
    replaced = False

    def replace_before_relative_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal replaced
        if dir_fd is not None and path == operation_name and not replaced:
            local_receipt.unlink()
            local_receipt.symlink_to(external)
            replaced = True
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(minichem_adapter_module.os, "open", replace_before_relative_open)
    adapter, session = adapter_factory(operation_root=operation_root)
    try:
        with pytest.raises(MiniChemMcpError, match="operation 收据文件不得是符号链接"):
            adapter._load_operation("raced-operation")
        assert replaced is True
        assert external.read_text(encoding="utf-8") == external_content
        assert session.calls == []
    finally:
        _close(adapter)


def test_operation_submission_lock_rejects_existing_symlink_without_locking_external_target(
    adapter_factory,
    tmp_path: Path,
) -> None:
    """既有 operation 锁符号链接必须被拒绝，不能锁定外部文件。"""

    operation_root = tmp_path / "existing-lock-symlink"
    operation_root.mkdir()
    external = tmp_path / "external-lock"
    external_content = "external-lock-content\n"
    external.write_text(external_content, encoding="utf-8")
    (operation_root / ".symlink-operation.lock").symlink_to(external)
    adapter, session = adapter_factory(operation_root=operation_root)
    try:
        with pytest.raises(MiniChemMcpError, match="operation 锁文件不得是符号链接"):
            with adapter._operation_submission_lock("symlink-operation"):
                raise AssertionError("符号链接锁不应进入临界区")
        assert external.read_text(encoding="utf-8") == external_content
        assert session.calls == []
    finally:
        _close(adapter)


def test_operation_submission_lock_rejects_symlink_replacement_before_safe_open(
    adapter_factory,
    monkeypatch,
    tmp_path: Path,
) -> None:
    """锁文件在检查后被替换为符号链接时，安全相对打开必须 fail closed。"""

    operation_root = tmp_path / "lock-race"
    operation_root.mkdir()
    lock_name = ".raced-operation.lock"
    local_lock = operation_root / lock_name
    local_lock.touch()
    external = tmp_path / "external-race-lock"
    external_content = "external-lock-content\n"
    external.write_text(external_content, encoding="utf-8")
    original_open = minichem_adapter_module.os.open
    replaced = False

    def replace_before_relative_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal replaced
        if dir_fd is not None and path == lock_name and not replaced:
            local_lock.unlink()
            local_lock.symlink_to(external)
            replaced = True
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(minichem_adapter_module.os, "open", replace_before_relative_open)
    adapter, session = adapter_factory(operation_root=operation_root)
    try:
        with pytest.raises(MiniChemMcpError, match="operation 锁文件不得是符号链接"):
            with adapter._operation_submission_lock("raced-operation"):
                raise AssertionError("竞态符号链接锁不应进入临界区")
        assert replaced is True
        assert external.read_text(encoding="utf-8") == external_content
        assert session.calls == []
    finally:
        _close(adapter)


def test_request_fingerprint_conflict_survives_resume_without_resubmission(
    adapter_factory,
    tmp_path: Path,
) -> None:
    """恢复只更新同一作业状态，原请求指纹必须继续阻断冲突提交。"""

    operation_root = tmp_path / "fingerprint-operations"
    operation_id = "fingerprint-resume-operation"
    request = {
        "software_id": "xtb",
        "executable": "xtb",
        "arguments": ["input.xyz"],
    }
    changed = {**request, "arguments": ["different.xyz"]}
    session = FakeMcpSession()
    adapter, _ = adapter_factory(session, operation_root=operation_root)
    try:
        submitted = adapter.submit_native_job(request, operation_id)
        fingerprint = submitted["request_fingerprint"]
        resumed = adapter.resume_operation(operation_id, collect=False)
        assert resumed["status"] == "waiting"
        assert resumed["request_fingerprint"] == fingerprint

        conflict = adapter.submit_native_job(changed, operation_id)
        assert conflict["status"] == "technical_pause"
        assert conflict["minichem_status"] == "operation_id_conflict"
        assert conflict["job_id"] == submitted["job_id"]
        assert session.native_submit_count == 1
        saved = json.loads((operation_root / f"{operation_id}.json").read_text())
        assert saved["request_fingerprint"] == fingerprint
    finally:
        _close(adapter)


def test_submit_analysis_uses_same_job_status_and_identity_contract(adapter_factory) -> None:
    """programmable analysis 提交同样返回 queued/job_id 合同。"""

    adapter, session = adapter_factory()
    try:
        submitted = adapter.submit_analysis_program(
            {
                "runtime": "python",
                "script_path": "scripts/analysis.py",
                "inputs": [],
                "resource_limits": {"cpu_cores": 1},
            },
            "analysis-operation-1",
        )
        assert submitted["status"] == "waiting"
        assert submitted["minichem_status"] == "queued"
        assert re.fullmatch(r"job_[0-9a-f]{32}", submitted["job_id"])
        assert submitted["operation_id"] == "analysis-operation-1"
        assert [call.name for call in session.calls] == ["submit_analysis_program"]
    finally:
        _close(adapter)


def test_submit_error_envelope_with_job_id_does_not_write_operation_receipt(
    adapter_factory,
    tmp_path: Path,
) -> None:
    """错误 envelope 即使携带 job_id，也不能被记录成可恢复提交。"""

    operation_root = tmp_path / "invalid-submit"
    job_id = "job_00000000000000000000000000000001"
    session = FakeMcpSession(
        tool_results={
            "submit_native_job": {
                "status": "invalid_request",
                "job_id": job_id,
                "job_status": "queued",
                "error": {"code": "invalid_input", "job_id": job_id},
            }
        }
    )
    adapter, _ = adapter_factory(session, operation_root=operation_root)
    try:
        result = adapter.submit_native_job(
            {"software_id": "xtb", "executable": "xtb", "arguments": []},
            "invalid-submit-operation",
        )
        assert result["status"] == "technical_pause"
        assert result["minichem_envelope_status"] == "invalid_request"
        assert not (operation_root / "invalid-submit-operation.json").exists()
    finally:
        _close(adapter)


def test_submit_success_with_unknown_job_status_does_not_write_receipt(
    adapter_factory,
    tmp_path: Path,
) -> None:
    """success envelope 的未知 job_status 不是明确接受，不能写恢复收据。"""

    operation_root = tmp_path / "unknown-submit"
    session = FakeMcpSession(
        tool_results={
            "submit_native_job": {
                "status": "success",
                "job_id": "job_00000000000000000000000000000001",
                "job_status": "unknown",
            }
        }
    )
    adapter, _ = adapter_factory(session, operation_root=operation_root)
    try:
        result = adapter.submit_native_job(
            {"software_id": "xtb", "executable": "xtb", "arguments": []},
            "unknown-submit-operation",
        )
        assert result["status"] == "technical_pause"
        assert not (operation_root / "unknown-submit-operation.json").exists()
    finally:
        _close(adapter)


@pytest.mark.parametrize(
    ("method_name", "response"),
    [
        (
            "get_execution_job",
            {
                "status": "success",
                "job_id": "job_00000000000000000000000000000002",
                "job": {
                    "job_id": "job_00000000000000000000000000000001",
                    "status": "queued",
                },
            },
        ),
        (
            "collect_execution_job",
            {
                "status": "success",
                "job_id": "job_00000000000000000000000000000001",
                "job": {
                    "job_id": "job_00000000000000000000000000000002",
                    "status": "success",
                },
            },
        ),
        (
            "cancel_execution_job",
            {
                "status": "success",
                "job_id": "job_00000000000000000000000000000002",
                "job_status": "cancelled",
                "cancellation_sent": True,
            },
        ),
    ],
)
def test_job_queries_reject_every_mismatched_returned_job_id(
    adapter_factory,
    method_name: str,
    response: dict,
) -> None:
    """get/collect/cancel 不能只信一个位置，任何错 job_id 都必须拒绝。"""

    requested_job_id = "job_00000000000000000000000000000001"
    session = FakeMcpSession(tool_results={method_name: response})
    adapter, _ = adapter_factory(session)
    try:
        with pytest.raises(MiniChemMcpProtocolError, match="job_id"):
            if method_name == "get_execution_job":
                adapter.get_execution_job(requested_job_id)
            elif method_name == "collect_execution_job":
                adapter.collect_execution_job(requested_job_id)
            else:
                adapter.cancel_execution_job(requested_job_id)
    finally:
        _close(adapter)


def test_get_job_allows_other_active_jobs_in_resource_availability(
    adapter_factory,
) -> None:
    """资源余量可以列出其他作业，但响应主体必须仍是被查询的作业。"""

    requested_job_id = "job_00000000000000000000000000000001"
    other_job_id = "job_00000000000000000000000000000002"
    session = FakeMcpSession(
        tool_results={
            "get_execution_job": {
                "status": "success",
                "job": {"job_id": requested_job_id, "status": "running"},
                "terminal": False,
                "resource_availability": {
                    "active_jobs": [
                        {"job_id": requested_job_id, "status": "running"},
                        {"job_id": other_job_id, "status": "running"},
                    ]
                },
            }
        }
    )
    adapter, _ = adapter_factory(session)
    try:
        result = adapter.get_execution_job(requested_job_id)
        assert result["status"] == "waiting"
        assert result["job_id"] == requested_job_id
        assert result["resource_availability"]["active_jobs"][1]["job_id"] == other_job_id
    finally:
        _close(adapter)


def test_collect_declarations_are_not_treated_as_artifact_refs(adapter_factory) -> None:
    """collect 的声明清单只描述文件验证，不强行套用 ArtifactRef 字段。"""

    job_id = "job_00000000000000000000000000000001"
    output = {
        "path": "outputs/result.xyz",
        "job_relative_path": "outputs/result.xyz",
        "size_bytes": 6,
        "sha256": "a" * 64,
        "kind": "output",
    }
    session = FakeMcpSession(
        collect_results={
            job_id: {
                "status": "success",
                "job_id": job_id,
                "job_status": "success",
                "job": {"job_id": job_id, "status": "success"},
                "ready": True,
                "outputs": [output],
                "declared_artifacts": [
                    {
                        "name": "result",
                        "path": "outputs/result.xyz",
                        "workspace_path": "outputs/result.xyz",
                        "semantic_type": "AtomicStructure",
                        "media_type": "chemical/x-xyz",
                        "producer": f"programmable_analysis:{job_id}",
                        "parent_artifacts": [],
                        "validation_status": "valid",
                    }
                ],
                "artifact_manifest": "outputs/artifact_manifest.json",
            }
        }
    )
    adapter, _ = adapter_factory(session)
    try:
        collected = adapter.collect_execution_job(job_id)
        assert collected["status"] == "completed"
        assert collected["declared_artifacts"][0]["name"] == "result"
        assert "artifact_id" not in collected["declared_artifacts"][0]
        assert collected["artifact_manifest"] == "outputs/artifact_manifest.json"
    finally:
        _close(adapter)


def test_collect_failed_job_is_technical_pause_and_keeps_job_identity(adapter_factory) -> None:
    """collect 终态失败必须保留失败状态，不能因 ready=true 伪造 completed。"""

    job_id = "job_00000000000000000000000000000001"
    session = FakeMcpSession(
        collect_results={
            job_id: {
                "status": "success",
                "job_id": job_id,
                "job_status": "failed",
                "job": {"job_id": job_id, "status": "failed"},
                "ready": True,
                "outputs": [],
                "files": [],
                "declared_artifacts": [],
                "artifact_manifest": None,
                "error": {
                    "code": "execution_failed",
                    "message": "deterministic child failure",
                },
            }
        }
    )
    adapter, _ = adapter_factory(session)
    try:
        result = adapter.collect_execution_job(job_id)
        assert result["status"] == "technical_pause"
        assert result["minichem_status"] == "failed"
        assert result["job_id"] == job_id
        assert result["error"]["code"] == "execution_failed"
    finally:
        _close(adapter)


def test_failed_job_promotes_bounded_failure_diagnostic_to_error(adapter_factory) -> None:
    """作业查询成功但 Gaussian 失败时，应把有限诊断交给 Router。"""

    job_id = "job_00000000000000000000000000000001"
    session = FakeMcpSession(
        tool_results={
            "get_execution_job": {
                "status": "success",
                "job": {"job_id": job_id, "status": "failed"},
                "terminal": True,
                "failure_diagnostic": {
                    "classification": "gaussian_route_syntax_error",
                    "message": "Gaussian 拒绝了 route section 的输入语法。",
                    "details": {
                        "rejected_route_field": "SCRF.solvent",
                        "recommended_value": "EthylEthanoate",
                    },
                    "candidate_fixes": [
                        "乙酸乙酯使用 Gaussian 原生别名 EthylEthanoate。"
                    ],
                    "internal_path": "/hidden/job/stdout.log",
                },
            }
        }
    )
    adapter, _ = adapter_factory(session)
    try:
        result = adapter.get_execution_job(job_id)
        assert result["status"] == "technical_pause"
        assert result["error"] == {
            "code": "gaussian_route_syntax_error",
            "message": "Gaussian 拒绝了 route section 的输入语法。",
            "details": {
                "rejected_route_field": "SCRF.solvent",
                "recommended_value": "EthylEthanoate",
            },
            "candidate_fixes": [
                "乙酸乙酯使用 Gaussian 原生别名 EthylEthanoate。"
            ],
        }
        assert "internal_path" not in result["error"]
    finally:
        _close(adapter)


def test_repeated_action_calls_do_not_create_operation_identity(adapter_factory) -> None:
    """同步 Action 重复调用不伪造 operation/job 幂等身份。"""

    session = FakeMcpSession(
        action_results={
            "convert_structure": {
                "status": "success",
                "action": "convert_structure",
                "action_version": "1.0.0",
                "requested_backend": "openbabel",
                "backend": "openbabel",
                "selection_source": "agent",
            }
        }
    )
    adapter, _ = adapter_factory(session)
    try:
        first = adapter.execute_action(_action_request())
        second = adapter.execute_action(_action_request())
        assert first == second
        assert session.submit_count == 2
        assert [call.name for call in session.calls] == [
            "execute_action",
            "execute_action",
        ]
        assert all("operation_id" not in call.arguments["request"] for call in session.calls)
    finally:
        _close(adapter)


def test_sync_action_never_enters_waiting_or_job_protocol(adapter_factory) -> None:
    """同步 Action 即使服务错误返回 waiting，也不携带 job/operation 身份。"""

    session = FakeMcpSession(
        action_results={
            "convert_structure": {
                "status": "waiting",
            }
        },
    )
    adapter, _ = adapter_factory(session)
    try:
        result = adapter.execute_action(_action_request())
        assert result["status"] == "technical_pause"
        assert result["minichem_status"] == "waiting"
        assert "job_id" not in result
        assert "operation_id" not in result
        assert session.submit_count == 1
        assert [call.name for call in session.calls] == ["execute_action"]
    finally:
        _close(adapter)


def test_cancel_maps_to_technical_pause_and_keeps_job_identity(adapter_factory) -> None:
    """取消只作用于原 job；原始 cancelled 状态和取消收据均可诊断。"""

    session = FakeMcpSession()
    adapter, _ = adapter_factory(session)
    try:
        submitted = adapter.submit_native_job(
            {"software_id": "xtb", "executable": "xtb", "arguments": []},
            "cancel-operation-1",
        )
        job_id = submitted["job_id"]
        cancelled = adapter.cancel_execution_job(job_id)
        assert cancelled["status"] == "technical_pause"
        assert cancelled["minichem_status"] == "cancellation_sent"
        assert cancelled["job_id"] == job_id
        assert cancelled["error"]["code"] == "cancellation_sent"
        assert session.native_submit_count == 1
        assert session.cancel_count[job_id] == 1
    finally:
        _close(adapter)


def test_cancel_response_loss_is_not_retried_without_operation_identity(
    adapter_factory,
) -> None:
    """取消信号已发送但响应丢失时，不得再次发送取消请求。"""

    job_id = "job_00000000000000000000000000000001"

    class CancelResponseLossSession(FakeMcpSession):
        def call_tool(self, name, arguments=None):
            response = super().call_tool(name, arguments)
            if name == "cancel_execution_job":
                raise ConnectionError("response lost after cancellation")
            return response

    sleeps: list[float] = []
    session = CancelResponseLossSession()
    adapter, _ = adapter_factory(session, retry_sleeper=sleeps.append)
    try:
        with pytest.raises(MiniChemMcpError, match="阻止不安全重放"):
            adapter.cancel_execution_job(job_id)
        assert session.cancel_count[job_id] == 1
        assert [call.name for call in session.calls] == ["cancel_execution_job"]
        assert sleeps == []
    finally:
        _close(adapter)


def test_artifact_ref_hash_and_provenance_are_preserved(adapter_factory) -> None:
    """ArtifactRef 的 ID、内容哈希、类型、来源和父链不被 adapter 改写。"""

    artifact = build_artifact_ref(
        content=b"benzaldehyde\n",
        artifact_id="art_structure_0001",
        producer_action="native_software:xtb",
        producer_backend="xtb",
        parent_artifact_ids=["art_input_0001"],
    )
    session = FakeMcpSession(
        artifact_results={
            "default": {
                "status": "success",
                "artifact": artifact,
                "provenance": {
                    "catalog_hash": build_catalog_snapshot()["catalog_hash"],
                    "request_id": "req-artifact-1",
                    "source": "fake-mcp",
                },
            }
        }
    )
    adapter, _ = adapter_factory(session)
    try:
        result = adapter.declare_scientific_artifact(
            {
                "path": artifact["path"],
                "semantic_type": artifact["semantic_type"],
                "media_type": artifact["media_type"],
                "producer_layer": "native_software",
                "producer_id": "xtb",
                "parent_artifact_ids": artifact["parent_artifact_ids"],
            }
        )
        assert session.calls[-1].arguments == {
            "request": {
                "path": artifact["path"],
                "semantic_type": artifact["semantic_type"],
                "media_type": artifact["media_type"],
                "producer_layer": "native_software",
                "producer_id": "xtb",
                "parent_artifact_ids": artifact["parent_artifact_ids"],
            }
        }
        assert result["status"] == "completed"
        declared = result["output_artifacts"][0]
        assert declared == artifact
        assert declared["sha256"] == hashlib.sha256(b"benzaldehyde\n").hexdigest()
        assert result["provenance"]["request_id"] == "req-artifact-1"
        assert result["provenance"]["source"] == "fake-mcp"
    finally:
        _close(adapter)


def test_successful_artifact_declaration_requires_an_artifact_ref(adapter_factory) -> None:
    """declare_scientific_artifact 成功但没有 ArtifactRef 时必须报告协议错误。"""

    session = FakeMcpSession(
        artifact_results={"default": {"status": "success", "provenance": {}}}
    )
    adapter, _ = adapter_factory(session)
    try:
        with pytest.raises(MiniChemMcpProtocolError, match="ArtifactRef"):
            adapter.declare_scientific_artifact(
                {
                    "path": "outputs/result.xyz",
                    "semantic_type": "AtomicStructure",
                    "producer_layer": "native_software",
                    "producer_id": "xtb",
                    "parent_artifact_ids": [],
                }
            )
    finally:
        _close(adapter)


@pytest.mark.parametrize("extra_field", ["operation_id", "job_id", "artifact_id"])
def test_declare_artifact_request_rejects_non_wire_fields(
    adapter_factory,
    extra_field: str,
) -> None:
    """ArtifactDeclarationRequest 只接受 archive 的六个公开字段。"""

    request = {
        "path": "outputs/result.xyz",
        "semantic_type": "AtomicStructure",
        "media_type": "chemical/x-xyz",
        "producer_layer": "native_software",
        "producer_id": "xtb",
        "parent_artifact_ids": [],
        extra_field: "unexpected-identity",
    }
    adapter, session = adapter_factory()
    try:
        with pytest.raises(MiniChemMcpProtocolError, match=extra_field):
            adapter.declare_scientific_artifact(request)
        assert session.calls == []
    finally:
        _close(adapter)


@pytest.mark.parametrize("missing_field", ["artifact_id", "semantic_type", "sha256", "path"])
def test_action_artifact_ref_requires_complete_wire_fields(adapter_factory, missing_field: str) -> None:
    """Action 输出 ArtifactRef 缺少任一身份/内容字段时拒绝响应。"""

    artifact = build_artifact_ref(producer_action="convert_structure", producer_backend="openbabel")
    artifact.pop(missing_field)
    session = FakeMcpSession(
        action_results={
            "convert_structure": {
                "status": "success",
                "action": "convert_structure",
                "action_version": "1.0.0",
                "requested_backend": "openbabel",
                "backend": "openbabel",
                "selection_source": "agent",
                "result": {"converted": True},
                "output_artifacts": [artifact],
            }
        }
    )
    adapter, _ = adapter_factory(session)
    try:
        with pytest.raises(MiniChemMcpProtocolError, match="output_artifacts"):
            adapter.execute_action(_action_request())
    finally:
        _close(adapter)


def test_action_artifact_refs_reject_duplicate_artifact_ids(adapter_factory) -> None:
    """同一 Action 响应不得重复声明同一个 ArtifactRef 身份。"""

    artifact = build_artifact_ref(producer_action="convert_structure", producer_backend="openbabel")
    session = FakeMcpSession(
        action_results={
            "convert_structure": {
                "status": "success",
                "action": "convert_structure",
                "action_version": "1.0.0",
                "requested_backend": "openbabel",
                "backend": "openbabel",
                "selection_source": "agent",
                "result": {"converted": True},
                "output_artifacts": [artifact, deepcopy(artifact)],
            }
        }
    )
    adapter, _ = adapter_factory(session)
    try:
        with pytest.raises(MiniChemMcpProtocolError, match="duplicate|重复"):
            adapter.execute_action(_action_request())
    finally:
        _close(adapter)


def test_remote_error_is_wrapped_with_mcp_method_context(adapter_factory) -> None:
    """远端瞬时断连只重试一次成功调用，并保留 MCP 方法边界。"""

    session = FakeMcpSession(
        tool_errors={"inspect_action": [ConnectionError("remote down")]}
    )
    adapter, _ = adapter_factory(session)
    try:
        result = adapter.inspect_action("convert_structure", "openbabel")
        assert result["status"] == "success"
        assert [call.name for call in session.calls] == [
            "inspect_action",
            "inspect_action",
        ]
    finally:
        _close(adapter)


def test_execute_action_blocks_transient_replay_without_operation_identity(
    adapter_factory,
) -> None:
    """同步 Action 没有幂等身份时，断连不能静默重复产生副作用。"""

    session = FakeMcpSession(
        tool_errors={"execute_action": ConnectionError("remote down")}
    )
    adapter, _ = adapter_factory(session)
    try:
        with pytest.raises(MiniChemMcpError, match="阻止不安全重放"):
            adapter.execute_action(_action_request())
        assert [call.name for call in session.calls] == ["execute_action"]
    finally:
        _close(adapter)


def test_execute_action_retries_response_loss_with_one_server_side_effect(
    adapter_factory,
) -> None:
    """有 operation 身份时，响应丢失后重试应读取服务端缓存而不重做 Action。"""

    class ResponseLossSession(FakeMcpSession):
        def __init__(self) -> None:
            super().__init__()
            self.side_effect_count = 0
            self.cached_response = None

        def call_tool(self, name, arguments=None):
            if name != "execute_action":
                return super().call_tool(name, arguments)
            if self.cached_response is None:
                self.cached_response = super().call_tool(name, arguments)
                self.side_effect_count += 1
                raise ConnectionError("response lost after completed receipt")
            self.calls.append(
                FakeToolCall(name=name, arguments=deepcopy(dict(arguments or {})))
            )
            return deepcopy(self.cached_response)

    sleeps: list[float] = []
    session = ResponseLossSession()
    adapter, _ = adapter_factory(session, retry_sleeper=sleeps.append)
    try:
        result = adapter.execute_action(
            _action_request(), operation_id="response-loss-operation"
        )
        assert result["status"] == "completed"
        assert result["operation_id"] == "response-loss-operation"
        assert session.side_effect_count == 1
        assert [call.name for call in session.calls] == [
            "execute_action",
            "execute_action",
        ]
        assert sleeps == [6.0]
    finally:
        _close(adapter)


def test_workspace_write_blocks_transient_replay_without_operation_identity(
    adapter_factory,
) -> None:
    """工作区写入同样没有 operation 身份，传输失败时必须显式阻塞。"""

    session = FakeMcpSession(
        tool_errors={"write_workspace_text": ConnectionError("remote down")}
    )
    adapter, _ = adapter_factory(session)
    try:
        with pytest.raises(MiniChemMcpError, match="阻止不安全重放"):
            adapter.write_workspace_text("inputs/job.gjf", "#P SP")
        assert [call.name for call in session.calls] == ["write_workspace_text"]
    finally:
        _close(adapter)


def test_artifact_declaration_blocks_transient_replay_without_operation_identity(
    adapter_factory,
) -> None:
    """Artifact 声明没有 operation 身份时，断连不能重复登记远端引用。"""

    session = FakeMcpSession(
        tool_errors={"declare_scientific_artifact": ConnectionError("remote down")}
    )
    adapter, _ = adapter_factory(session)
    try:
        with pytest.raises(MiniChemMcpError, match="阻止不安全重放"):
            adapter.declare_scientific_artifact(
                {
                    "path": "outputs/result.xyz",
                    "semantic_type": "AtomicStructure",
                    "media_type": "chemical/x-xyz",
                    "producer_layer": "native_software",
                    "producer_id": "xtb",
                    "parent_artifact_ids": [],
                }
            )
        assert [call.name for call in session.calls] == [
            "declare_scientific_artifact"
        ]
    finally:
        _close(adapter)


def test_call_tool_error_plain_text_is_preserved_as_diagnostic(adapter_factory) -> None:
    """CallToolResult.isError/plain text 必须形成可诊断技术暂停。"""

    session = FakeMcpSession(
        tool_results={
            "execute_action": {
                "isError": True,
                "content": [{"type": "text", "text": "backend unavailable"}],
            }
        }
    )
    adapter, _ = adapter_factory(session)
    try:
        result = adapter.execute_action(_action_request())
        assert result["status"] == "technical_pause"
        assert result["minichem_status"] == "mcp_error"
        assert result["error"]["message"] == "backend unavailable"
    finally:
        _close(adapter)


def test_catalog_freeze_drift_is_classified_for_recovery(adapter_factory) -> None:
    """服务冻结目录漂移必须使用稳定错误码，供恢复层开放重启工作流。"""

    session = FakeMcpSession(
        tool_results={
            "execute_action": {
                "isError": True,
                "content": [
                    {
                        "type": "text",
                        "text": "Error executing tool execute_action: 服务冻结目录文件已被删除或替换",
                    }
                ],
            }
        }
    )
    adapter, _ = adapter_factory(session)
    try:
        result = adapter.execute_action(_action_request())
        assert result["status"] == "technical_pause"
        assert result["minichem_status"] == "mcp_error"
        assert result["error"]["code"] == "catalog_freeze_drift"
    finally:
        _close(adapter)


def test_call_tool_is_error_json_is_preserved_as_mcp_diagnostic(adapter_factory) -> None:
    """MCP isError 的 JSON 文本不能被当作 ActionResult 成功解析。"""

    session = FakeMcpSession(
        tool_results={
            "execute_action": {
                "isError": True,
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "code": "backend_unavailable",
                                "message": "JSON diagnostic from MiniChem",
                            }
                        ),
                    }
                ],
            }
        }
    )
    adapter, _ = adapter_factory(session)
    try:
        result = adapter.execute_action(_action_request())
        assert result["status"] == "technical_pause"
        assert result["minichem_status"] == "mcp_error"
        assert result["error"]["code"] == "mcp_error"
        assert "backend_unavailable" in json.dumps(result["error"], ensure_ascii=False)
        assert "JSON diagnostic from MiniChem" in json.dumps(result["error"], ensure_ascii=False)
    finally:
        _close(adapter)


def test_protocol_error_rejects_bad_catalog_hash(adapter_factory) -> None:
    """目录哈希格式错误时立即抛出可识别的协议异常。"""

    catalog = build_catalog_snapshot()
    catalog["catalog_hash"] = "not-a-sha256"
    session = FakeMcpSession(catalog=catalog)
    adapter, _ = adapter_factory(session)
    try:
        with pytest.raises(MiniChemMcpProtocolError, match="catalog_hash"):
            adapter.discover_catalog()
    finally:
        _close(adapter)


def test_job_id_must_follow_public_regex(adapter_factory) -> None:
    """get/collect/cancel 不接受旧的伪造 job_id 格式。"""

    adapter, _ = adapter_factory()
    try:
        with pytest.raises(ValueError, match="job_id"):
            adapter.get_execution_job("job_fake_0001")
    finally:
        _close(adapter)


def test_action_waiting_is_not_a_job_protocol(adapter_factory) -> None:
    """同步 Action 的 waiting 响应只能形成技术暂停，不能伪造作业身份。"""

    session = FakeMcpSession(
        action_results={"convert_structure": {"status": "waiting", "result": {}}}
    )
    adapter, _ = adapter_factory(session)
    try:
        result = adapter.execute_action(_action_request())
        assert result["status"] == "technical_pause"
        assert result["minichem_status"] == "waiting"
        assert "operation_id" not in result
        assert "job_id" not in result
    finally:
        _close(adapter)


def test_adapter_does_not_bypass_program_gate_or_router(adapter_factory) -> None:
    """adapter 只调用 MCP discovery/action，不主动创建 Program Gate/Router 事实。

    Program Gate/Router 的事实投影仍需在生产工作流集成测试中验证；这里先冻结
    转换边界，确保 adapter 本身没有 ``program_gate``、``router`` 或旧工具调用。
    """

    adapter, session = adapter_factory()
    try:
        adapter.discover_catalog()
        adapter.execute_action(_action_request())
        names = [call.name for call in session.calls]
        assert "program_gate" not in names
        assert "router" not in names
        assert all(
            name
            in {
                "execute_action",
                "list_action_domains",
                "search_actions",
                "inspect_action",
                "inspect_backend",
            }
            for name in names
        )
    finally:
        _close(adapter)


def test_catalog_uses_approved_task_budget_and_forwards_explicit_action_limits(
    adapter_factory,
) -> None:
    """任务级预算固定为 60 核/320000 MB/1 GPU，Action 仍只转发显式值。"""

    adapter, session = adapter_factory()
    try:
        catalog = adapter.discover_catalog()
        assert catalog.snapshot["evaluation_resource_budget"] == {
            "cpu_cores": 60,
            "memory_mb": 320000,
            "gpu_count": 1,
            "source": "evaluation_policy",
            "agent_controllable": False,
            "scope": "per_task",
        }
        request = _action_request().to_payload()
        request["resource_limits"] = {
            "cpu_cores": 60,
            "memory_mb": 320000,
            "gpu_count": 0,
        }
        result = adapter.execute_action(request)
        assert result["status"] == "completed"
        assert session.calls[-1].arguments["request"]["resource_limits"] == request[
            "resource_limits"
        ]
    finally:
        _close(adapter)


@pytest.mark.parametrize(
    "resource_limits",
    [
        {"cpu_cores": True},
        {"memory_mb": 127},
        {"gpu_count": -1},
        {"unknown": 1},
    ],
)
def test_action_resource_limits_reject_malformed_values_before_transport(
    adapter_factory,
    resource_limits: dict,
) -> None:
    """单次 Action 的资源字段非法时不应产生 MCP 副作用。"""

    adapter, session = adapter_factory()
    try:
        request = _action_request().to_payload()
        request["resource_limits"] = resource_limits
        with pytest.raises(MiniChemMcpProtocolError, match="resource_limits"):
            adapter.execute_action(request)
        assert session.calls == []
    finally:
        _close(adapter)


def test_async_fake_transport_lifecycle_initializes_and_closes_once() -> None:
    """异步 transport 路径在同一 task 中完成调用，并且只进出一次。"""

    class FakeAsyncSession:
        def __init__(self) -> None:
            self.initialize_count = 0
            self.close_count = 0
            self.task_ids: list[int] = []

        async def initialize(self) -> dict:
            self.initialize_count += 1
            self.task_ids.append(id(asyncio.current_task()))
            return {"protocolVersion": "fake"}

        async def close(self) -> None:
            self.close_count += 1

        async def list_tools(self) -> dict:
            self.task_ids.append(id(asyncio.current_task()))
            return {"tools": []}

        async def call_tool(self, _name: str, _arguments: dict) -> dict:
            self.task_ids.append(id(asyncio.current_task()))
            return {"content": [{"type": "text", "text": "{}"}]}

        async def read_resource(self, _uri: str) -> dict:
            self.task_ids.append(id(asyncio.current_task()))
            return {"contents": []}

    class FakeAsyncTransport:
        def __init__(self, session: FakeAsyncSession) -> None:
            self.session = session
            self.enter_count = 0
            self.exit_count = 0
            self.enter_task_ids: list[int] = []
            self.exit_task_ids: list[int] = []

        async def __aenter__(self) -> FakeAsyncSession:
            self.enter_count += 1
            self.enter_task_ids.append(id(asyncio.current_task()))
            return self.session

        async def __aexit__(self, _exc_type, _exc, _traceback) -> None:
            self.exit_count += 1
            self.exit_task_ids.append(id(asyncio.current_task()))

    session = FakeAsyncSession()
    transport = FakeAsyncTransport(session)
    bridge = _AsyncSessionBridge("http://127.0.0.1:9010/mcp", lambda: transport)
    try:
        bridge.start()
        assert bridge.call("list_tools") == {"tools": []}
        assert session.initialize_count == 1
        assert transport.enter_count == 1
        assert len(set(session.task_ids + transport.enter_task_ids)) == 1
    finally:
        bridge.close()
    assert transport.exit_count == 1
    assert session.close_count == 0
    assert transport.exit_task_ids == transport.enter_task_ids


def test_combined_context_closes_owned_client_on_enter_and_exit_failures() -> None:
    """组合上下文在 transport 进入/退出失败时都必须释放自有 HTTP client。"""

    class TrackingClient:
        def __init__(self) -> None:
            self.enter_count = 0
            self.exit_count = 0

        async def __aenter__(self):
            self.enter_count += 1
            return self

        async def __aexit__(self, _exc_type, _exc, _traceback):
            self.exit_count += 1
            return None

    class FailingEnterTransport:
        async def __aenter__(self):
            raise RuntimeError("transport enter failed")

        async def __aexit__(self, _exc_type, _exc, _traceback):
            raise AssertionError("未进入的 transport 不应退出")

    class FailingExitTransport:
        async def __aenter__(self):
            return (None, None, None)

        async def __aexit__(self, _exc_type, _exc, _traceback):
            raise RuntimeError("transport exit failed")

    class Session:
        def __init__(self, _receive, _send) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, _exc_type, _exc, _traceback):
            return None

    async def exercise() -> None:
        enter_client = TrackingClient()
        enter_context = _CombinedMcpContext(
            FailingEnterTransport(),
            Session,
            owned_client=enter_client,
        )
        with pytest.raises(RuntimeError, match="transport enter failed"):
            await enter_context.__aenter__()
        assert enter_client.enter_count == 1
        assert enter_client.exit_count == 1

        exit_client = TrackingClient()
        exit_context = _CombinedMcpContext(
            FailingExitTransport(),
            Session,
            owned_client=exit_client,
        )
        await exit_context.__aenter__()
        with pytest.raises(RuntimeError, match="transport exit failed"):
            await exit_context.__aexit__(None, None, None)
        assert exit_client.enter_count == 1
        assert exit_client.exit_count == 1

    asyncio.run(exercise())


def test_async_bridge_rejects_active_call_when_session_runner_stops() -> None:
    """会话线程在调用中退出时，活动调用也必须尽快收到传输错误。"""

    class RunnerStoppingSession:
        def __init__(self) -> None:
            self.bridge: _AsyncSessionBridge | None = None
            self.started = threading.Event()
            self.exit_count = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, _exc_type, _exc, _traceback):
            self.exit_count += 1
            return None

        async def initialize(self):
            return {"protocolVersion": "fake"}

        async def list_tools(self):
            self.started.set()
            assert self.bridge is not None
            with self.bridge._state_lock:
                loop = self.bridge._loop
            assert loop is not None
            # 模拟 Streamable HTTP transport 的生命周期线程在断连后停止。
            loop.call_soon(loop.stop)
            await asyncio.sleep(3600)

    session = RunnerStoppingSession()
    bridge = _AsyncSessionBridge(
        "http://127.0.0.1:9010/mcp",
        lambda: session,
    )
    session.bridge = bridge
    result: dict[str, BaseException] = {}

    def call_from_worker() -> None:
        try:
            bridge.call("list_tools")
        except BaseException as exc:
            result["error"] = exc

    worker = threading.Thread(target=call_from_worker, daemon=True)
    try:
        bridge.start()
        worker.start()
        assert session.started.wait(timeout=1)
        worker.join(timeout=1)
        assert not worker.is_alive()
        assert isinstance(result.get("error"), MiniChemMcpError)
    finally:
        bridge.close()
    assert session.exit_count == 1


def test_async_bridge_converts_cancelled_command_to_retryable_error() -> None:
    """会话命令取消时，同步调用方收到可识别的瞬时传输错误。"""

    class CancelledSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, _exc_type, _exc, _traceback):
            return None

        async def initialize(self):
            return {"protocolVersion": "fake"}

        async def list_tools(self):
            raise asyncio.CancelledError()

    bridge = _AsyncSessionBridge(
        "http://127.0.0.1:9010/mcp",
        CancelledSession,
    )
    try:
        bridge.start()
        with pytest.raises(MiniChemMcpError) as raised:
            bridge.call("list_tools")
        assert isinstance(raised.value.__cause__, ConnectionError)
        assert is_retryable_api_exception(raised.value)
    finally:
        bridge.close()


def test_async_bridge_rebuilds_session_before_retry_after_disconnect(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """活动调用断连后，retry 必须等待旧线程退出并建立新会话。"""

    monkeypatch.setattr(
        minichem_adapter_module,
        "_bridge_result_timeout_seconds",
        lambda: 0.05,
    )

    class BrokenContext:
        def __init__(self) -> None:
            self.bridge: _AsyncSessionBridge | None = None
            self.exit_started = threading.Event()
            self.release_exit = threading.Event()
            self.started = threading.Event()

        async def __aenter__(self):
            return self

        async def __aexit__(self, _exc_type, _exc, _traceback):
            self.exit_started.set()
            # 模拟断连后的 transport 清理仍在进行；旧线程在此期间
            # 不能被新的 retry 复用，测试由定时器释放它。
            self.release_exit.wait(timeout=1)
            return None

        async def initialize(self):
            return {"protocolVersion": "fake"}

        async def list_tools(self):
            self.started.set()
            assert self.bridge is not None
            with self.bridge._state_lock:
                loop = self.bridge._loop
            assert loop is not None
            loop.call_soon(loop.stop)
            await asyncio.sleep(3600)

    class HealthyContext:
        async def __aenter__(self):
            return self

        async def __aexit__(self, _exc_type, _exc, _traceback):
            return None

        async def initialize(self):
            return {"protocolVersion": "fake"}

        async def list_tools(self):
            return {"content": [{"type": "text", "text": '{"tools": []}'}]}

    broken = BrokenContext()
    sessions = [broken, HealthyContext()]
    factory_calls = 0

    def session_factory():
        nonlocal factory_calls
        factory_calls += 1
        candidate = sessions.pop(0)
        if candidate is broken:
            broken.bridge = bridge
        return candidate

    adapter = MiniChemMcpAdapter(
        endpoint="http://127.0.0.1:9010/mcp",
        session_factory=session_factory,
        operation_root=tmp_path / "operations",
        retry_count=1,
        retry_sleeper=lambda _delay: None,
    )
    bridge = adapter._bridge
    release_timer = threading.Timer(0.2, broken.release_exit.set)
    try:
        adapter.open()
        release_timer.start()
        assert adapter.list_tools() == {"tools": []}
        assert factory_calls == 2
    finally:
        broken.release_exit.set()
        release_timer.cancel()
        adapter.close()


def test_execute_action_replays_same_operation_after_async_session_disconnect(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Action 传输中断后必须建新会话，并以同一 operation_id 重放。"""

    monkeypatch.setattr(
        minichem_adapter_module,
        "_bridge_result_timeout_seconds",
        lambda: 0.05,
    )

    class BrokenContext:
        def __init__(self) -> None:
            self.bridge: _AsyncSessionBridge | None = None
            self.release_exit = threading.Event()
            self.calls: list[dict] = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, _exc_type, _exc, _traceback):
            self.release_exit.wait(timeout=1)
            return None

        async def initialize(self):
            return {"protocolVersion": "fake"}

        async def call_tool(self, name, arguments=None):
            self.calls.append(
                {"name": name, "arguments": deepcopy(dict(arguments or {}))}
            )
            assert name == "execute_action"
            assert self.bridge is not None
            with self.bridge._state_lock:
                loop = self.bridge._loop
            assert loop is not None
            # 模拟当前 Action 的 Streamable HTTP transport 断连。
            loop.call_soon(loop.stop)
            await asyncio.sleep(3600)

    class HealthyContext:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, _exc_type, _exc, _traceback):
            return None

        async def initialize(self):
            return {"protocolVersion": "fake"}

        async def call_tool(self, name, arguments=None):
            self.calls.append(
                {"name": name, "arguments": deepcopy(dict(arguments or {}))}
            )
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "status": "success",
                                "action": "convert_structure",
                                "requested_backend": "openbabel",
                                "backend": "openbabel",
                                "result": {"converted": True},
                            }
                        ),
                    }
                ]
            }

    broken = BrokenContext()
    healthy = HealthyContext()
    sessions = [broken, healthy]
    factory_calls = 0

    def session_factory():
        nonlocal factory_calls
        factory_calls += 1
        candidate = sessions.pop(0)
        if candidate is broken:
            broken.bridge = bridge
        return candidate

    operation_id = "async-action-replay-operation"
    adapter = MiniChemMcpAdapter(
        endpoint="http://127.0.0.1:9010/mcp",
        session_factory=session_factory,
        operation_root=tmp_path / "operations",
        retry_count=1,
        retry_sleeper=lambda _delay: None,
    )
    bridge = adapter._bridge
    release_timer = threading.Timer(0.2, broken.release_exit.set)
    try:
        adapter.open()
        release_timer.start()
        result = adapter.execute_action(_action_request(), operation_id=operation_id)
        assert result["status"] == "completed"
        assert result["operation_id"] == operation_id
        assert factory_calls == 2
        assert len(broken.calls) == 1
        assert len(healthy.calls) == 1
        assert broken.calls[0]["name"] == healthy.calls[0]["name"] == "execute_action"
        assert (
            broken.calls[0]["arguments"]["request"]
            == healthy.calls[0]["arguments"]["request"]
        )
        assert (
            healthy.calls[0]["arguments"]["request"]["operation_id"]
            == operation_id
        )
    finally:
        broken.release_exit.set()
        release_timer.cancel()
        adapter.close()


def test_execute_action_recovers_when_http_reconnect_409_leaves_lifecycle_alive(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """HTTP 重连最终 409 时，即使 lifecycle 未退出也必须恢复 Action。"""

    monkeypatch.setattr(
        minichem_adapter_module,
        "_bridge_result_timeout_seconds",
        lambda: 0.2,
    )

    class FakeClientSession:
        def __init__(self, _receive, _send) -> None:
            self.healthy = current_context.healthy

        async def __aenter__(self):
            return self

        async def __aexit__(self, _exc_type, _exc, _traceback):
            return None

        async def initialize(self):
            return {"protocolVersion": "fake"}

        async def call_tool(self, name, arguments=None):
            assert name == "execute_action"
            if not self.healthy:
                # 409 没有被通知到 bridge 时，模拟底层 call_tool 短时间内仍挂起；
                # 时间上限只为让红测可回收，不代表生产 Action timeout。
                await asyncio.sleep(0.5)
            current_context.calls.append(deepcopy(dict(arguments or {})))
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "status": "success",
                                "action": "convert_structure",
                                "requested_backend": "openbabel",
                                "backend": "openbabel",
                                "result": {"converted": True},
                            }
                        ),
                    }
                ]
            }

    class FakeTransportContext:
        def __init__(self, http_client, healthy: bool) -> None:
            self.http_client = http_client
            self.healthy = healthy
            self.trigger_task: asyncio.Task | None = None
            self.calls: list[dict] = []

        async def __aenter__(self):
            if self.http_client is not None:
                async def report_reconnect_409():
                    await asyncio.sleep(0.02)
                    response = type(
                        "FakeResponse",
                        (),
                        {
                            "status_code": 409,
                            "request": type("FakeRequest", (), {"method": "GET"})(),
                        },
                    )()
                    for hook in self.http_client.event_hooks["response"]:
                        await hook(response)

                self.trigger_task = asyncio.create_task(report_reconnect_409())
            return (None, None, None)

        async def __aexit__(self, _exc_type, _exc, _traceback):
            if self.trigger_task is not None:
                self.trigger_task.cancel()
                try:
                    await self.trigger_task
                except asyncio.CancelledError:
                    pass
            return None

    class ContextState:
        def __init__(self, healthy: bool) -> None:
            self.healthy = healthy
            self.calls: list[dict] = []

    contexts = [ContextState(False), ContextState(True)]
    current_context = contexts[0]
    factory_calls = 0
    captured_client = None

    def fake_streamable_http_client(_endpoint, *, http_client=None):
        nonlocal captured_client, factory_calls, current_context
        current_context = contexts.pop(0)
        captured_client = http_client
        factory_calls += 1
        return FakeTransportContext(http_client, current_context.healthy)

    monkeypatch.setattr("mcp.ClientSession", FakeClientSession)
    monkeypatch.setattr(
        "mcp.client.streamable_http.streamable_http_client",
        fake_streamable_http_client,
    )
    adapter = MiniChemMcpAdapter(
        endpoint="http://127.0.0.1:9010/mcp",
        operation_root=tmp_path / "operations",
        retry_count=1,
        retry_sleeper=lambda _delay: None,
    )
    operation_id = "http-reconnect-409-operation"
    try:
        result = adapter.execute_action(_action_request(), operation_id=operation_id)
        assert result["status"] == "completed"
        assert result["operation_id"] == operation_id
        assert factory_calls == 2
        assert contexts == []
        assert current_context.calls[0]["request"]["operation_id"] == operation_id
    finally:
        adapter.close()
    assert captured_client is not None
    assert captured_client.is_closed


def test_async_context_close_exception_is_visible_but_exit_remains_idempotent() -> None:
    """关闭异常必须传给同步调用方，重复 close 不得重复退出上下文。"""

    class FailingCloseContext:
        def __init__(self) -> None:
            self.enter_count = 0
            self.exit_count = 0

        async def __aenter__(self):
            self.enter_count += 1
            return self

        async def __aexit__(self, _exc_type, _exc, _traceback):
            self.exit_count += 1
            raise RuntimeError("deterministic close failure")

        async def initialize(self):
            return {"protocolVersion": "fake"}

        async def list_tools(self):
            return {"tools": []}

        async def call_tool(self, _name, _arguments):
            return {"content": [{"type": "text", "text": "{}"}]}

        async def read_resource(self, _uri):
            return {"contents": []}

    context = FailingCloseContext()
    bridge = _AsyncSessionBridge("http://127.0.0.1:9010/mcp", lambda: context)
    bridge.start()
    with pytest.raises(MiniChemMcpError, match="关闭 MiniChem MCP 会话失败"):
        bridge.close()
    bridge.close()
    assert context.enter_count == 1
    assert context.exit_count == 1


def test_async_context_open_failure_releases_entered_context_once() -> None:
    """初始化失败仍须在同一生命周期 task 中退出已进入的异步上下文。"""

    class FailingInitializeContext:
        def __init__(self) -> None:
            self.enter_count = 0
            self.exit_count = 0

        async def __aenter__(self):
            self.enter_count += 1
            return self

        async def __aexit__(self, _exc_type, _exc, _traceback):
            self.exit_count += 1

        async def initialize(self):
            raise RuntimeError("deterministic initialize failure")

        async def list_tools(self):
            return {"tools": []}

        async def call_tool(self, _name, _arguments):
            return {"content": [{"type": "text", "text": "{}"}]}

        async def read_resource(self, _uri):
            return {"contents": []}

    context = FailingInitializeContext()
    bridge = _AsyncSessionBridge("http://127.0.0.1:9010/mcp", lambda: context)
    with pytest.raises(MiniChemMcpError, match="deterministic initialize failure"):
        bridge.start()
    bridge.close()
    assert context.enter_count == 1
    assert context.exit_count == 1


def test_wait_for_minichem_operation_completion_polls_without_resubmitting(
    adapter_factory,
) -> None:
    """显式等待接口只轮询/收集原 job，不能再次 submit。"""

    adapter, session = adapter_factory()
    try:
        submitted = adapter.submit_native_job(
            {"software_id": "xtb", "executable": "xtb", "arguments": []},
            "wait-operation-1",
        )
        completed = adapter.wait_for_minichem_operation_completion(
            "wait-operation-1",
            timeout_seconds=1.0,
            collect=True,
        )
        assert completed["status"] == "completed"
        assert completed["job_id"] == submitted["job_id"]
        assert session.native_submit_count == 1
        assert session.get_count[submitted["job_id"]] == 2
        assert session.collect_count[submitted["job_id"]] == 1
    finally:
        _close(adapter)


def _open_execution_inspection(software_id: str, *, command: str = "g16") -> dict:
    """构造带有路径和文档字段的 inspect 响应，验证适配器会做脱敏投影。"""

    return {
        "status": "success",
        "software_id": software_id,
        "display_name": "Gaussian 16",
        "native_invocation_guides": [
            {
                "executable": command,
                "enabled": True,
                "available": True,
                "resolved_path": "/secret/absolute/g16",
                "synopsis": "g16 < input.com",
                "input_mode": "stdin_file",
                "required_files": ["input.com"],
                "native_job_request_template": {
                    "software_id": software_id,
                    "executable": command,
                    "arguments": [],
                    "staged_inputs": [
                        {"source_path": "code/input.com", "target_path": "input.com"}
                    ],
                    "stdin_target": "input.com",
                    "resource_limits": {"cpu_cores": 1, "memory_mb": 4096, "gpu_count": 0},
                    "calculation_intent": "frequency",
                    "label": "benzaldehyde smoke test",
                    "parent_job_id": None,
                    "operation_id": "must-not-leak",
                    "python": "/secret/python",
                    "external_commands": ["/secret/g16"],
                    "documentation_body": "must not be copied",
                },
            },
            {
                "executable": "disabled",
                "enabled": False,
                "available": True,
                "synopsis": "not exposed",
                "input_mode": "arguments",
                "required_files": [],
            },
        ],
    }


def test_discover_open_execution_catalog_uses_exact_sources_and_sanitizes_examples(
    adapter_factory,
) -> None:
    """Open Execution 目录只读取三类 MCP 来源，并返回稳定、脱敏的公开投影。"""

    session = FakeMcpSession(
        tool_results={
            "list_software": {
                "status": "success",
                "software": [
                    {
                        "software_id": "gaussian",
                        "display_name": "Gaussian 16",
                        "available": True,
                        "native_executables": [
                            {"executable": "g16", "enabled": True, "available": True}
                        ],
                    }
                ],
            },
            "inspect_software": _open_execution_inspection("gaussian"),
            "list_analysis_runtimes": {
                "status": "success",
                "runtimes": [
                    {
                        "runtime": "core",
                        "description": "RDKit and deterministic chemistry analysis.",
                        "available": True,
                        "modules": {"rdkit": {"available": None, "version": None}},
                        "python": "/secret/python",
                        "health_checks": {"external_commands": ["/secret/tool"]},
                    }
                ],
            },
        }
    )
    adapter, _ = adapter_factory(session)
    try:
        catalog = adapter.discover_open_execution_catalog()
        assert catalog == {
            "schema_version": 1,
            "native_software": [
                {
                    "software_id": "gaussian",
                    "display_name": "Gaussian 16",
                    "commands": [
                        {
                            "executable": "g16",
                            "synopsis": "g16 < input.com",
                            "input_mode": "stdin_file",
                            "required_files": ["input.com"],
                            "example_request": {
                                "software_id": "gaussian",
                                "executable": "g16",
                                "arguments": [],
                                "staged_inputs": [
                                    {
                                        "source_path": "code/input.com",
                                        "target_path": "input.com",
                                    }
                                ],
                                "stdin_target": "input.com",
                                "resource_limits": {
                                    "cpu_cores": 1,
                                    "memory_mb": 4096,
                                    "gpu_count": 0,
                                },
                                "calculation_intent": "frequency",
                                "label": "benzaldehyde smoke test",
                                "parent_job_id": None,
                            },
                        }
                    ],
                }
            ],
            "analysis_runtimes": [
                {
                    "runtime": "core",
                    "description": "RDKit and deterministic chemistry analysis.",
                    "modules": ["rdkit"],
                }
            ],
            "authoring_contract": {
                "path_prefix": "code/",
                "max_files": 4,
                "max_chars_per_file": 100000,
            },
        }
        serialized = json.dumps(catalog, ensure_ascii=False)
        assert "/secret/" not in serialized
        assert "operation_id" not in serialized
        assert "documentation_body" not in serialized
        assert [call.name for call in session.calls] == [
            "list_software",
            "inspect_software",
            "list_analysis_runtimes",
        ]
        assert session.calls[0].arguments["request"] == {
            "native_only": True,
            "available_only": True,
            "limit": 200,
        }
        assert session.calls[1].arguments["request"] == {"software_id": "gaussian"}
        assert session.calls[2].arguments["request"] == {
            "available_only": True,
            "limit": 200,
            "include_details": True,
        }
    finally:
        _close(adapter)


def test_discover_open_execution_catalog_rejects_duplicate_ids(adapter_factory) -> None:
    """Open Execution 服务返回重复软件或运行时身份时必须拒绝整个目录。"""

    session = FakeMcpSession(
        tool_results={
            "list_software": {
                "status": "success",
                "software": [
                    {"software_id": "gaussian", "display_name": "Gaussian", "available": True},
                    {"software_id": "gaussian", "display_name": "Gaussian duplicate", "available": True},
                ],
            }
        }
    )
    adapter, _ = adapter_factory(session)
    try:
        with pytest.raises(MiniChemMcpProtocolError, match="重复"):
            adapter.discover_open_execution_catalog()
    finally:
        _close(adapter)


def test_discover_open_execution_catalog_keeps_only_commands_enabled_and_available(
    adapter_factory,
) -> None:
    """紧凑清单或 inspect 任一层标记不可用时，命令都不进入公开目录。"""

    inspection = _open_execution_inspection("gaussian")
    inspection["native_invocation_guides"].append(
        {
            "executable": "formchk",
            "enabled": True,
            "available": True,
            "synopsis": "formchk input.chk output.fchk",
            "input_mode": "arguments",
            "required_files": ["input.chk"],
            "native_job_request_template": {
                "software_id": "gaussian",
                "executable": "formchk",
                "arguments": ["input.chk", "output.fchk"],
                "staged_inputs": [],
                "resource_limits": {"cpu_cores": 1, "memory_mb": 128, "gpu_count": 0},
            },
        }
    )
    session = FakeMcpSession(
        tool_results={
            "list_software": {
                "status": "success",
                "software": [
                    {
                        "software_id": "gaussian",
                        "display_name": "Gaussian 16",
                        "available": True,
                        "native_executables": [
                            {"executable": "g16", "enabled": True, "available": True},
                            {"executable": "formchk", "enabled": False, "available": True},
                        ],
                    }
                ],
            },
            "inspect_software": inspection,
            "list_analysis_runtimes": {"status": "success", "runtimes": []},
        }
    )
    adapter, _ = adapter_factory(session)
    try:
        catalog = adapter.discover_open_execution_catalog()
        assert [item["executable"] for item in catalog["native_software"][0]["commands"]] == [
            "g16"
        ]
    finally:
        _close(adapter)


def test_discover_open_execution_catalog_preserves_managed_contract_and_output_behavior(
    adapter_factory,
) -> None:
    """Open Execution 目录必须保留 JobContext、请求模板和 native 输出行为。"""

    inspection = _open_execution_inspection("gaussian")
    guide = inspection["native_invocation_guides"][0]
    guide["output_behavior"] = "Writes Gaussian output to stdout."
    runtime_contract = {
        "job_context_required_for_compliance": True,
        "execution_note": "Use JobContext for declared inputs and outputs.",
        "minimal_python_template": "from researchchem_job import JobContext\nctx = JobContext.load()\n",
        "helper_name_rule": "Helper names must match declaration names.",
    }
    analysis_template = {
        "runtime": "<exact runtime id>",
        "script_path": "code/<agent-authored-program>.py",
        "script_target": "code/agent_program.py",
        "arguments": [],
        "staged_inputs": [],
        "inputs": [
            {
                "name": "<logical_input_name>",
                "source_path": "data/<required-input>",
                "target_path": "inputs/<required-input>",
                "semantic_type": "<input semantic type>",
            }
        ],
        "outputs": [
            {
                "name": "<logical_output_name>",
                "path": "outputs/<result-file>",
                "semantic_type": "<output semantic type>",
                "media_type": "application/json",
                "required": True,
            }
        ],
        "label": "<descriptive scientific operation>",
    }
    session = FakeMcpSession(
        tool_results={
            "list_software": {
                "status": "success",
                "software": [
                    {
                        "software_id": "gaussian",
                        "display_name": "Gaussian 16",
                        "available": True,
                        "native_executables": [
                            {"executable": "g16", "enabled": True, "available": True}
                        ],
                    }
                ],
            },
            "inspect_software": inspection,
            "list_analysis_runtimes": {
                "status": "success",
                "managed_program_contract": runtime_contract,
                "submit_analysis_program_request_template": analysis_template,
                "selection_note": "Select one exact runtime; no runtime is chosen automatically.",
                "execution_note": "Submit authored programs through the managed job runner.",
                "evaluation_resource_budget": {
                    "source": "service_policy",
                    "agent_controllable": False,
                    "scope": "per_task",
                },
                "execution_timeout_policy": {
                    "execution_class": "compute",
                    "source": "service_policy",
                    "agent_controllable": False,
                },
                "runtimes": [
                    {
                        "runtime": "core",
                        "description": "RDKit runtime",
                        "available": True,
                        "modules": {"rdkit": {"available": None, "version": None}},
                    }
                ],
            },
        }
    )
    adapter, _ = adapter_factory(session)
    try:
        catalog = adapter.discover_open_execution_catalog()
        command = catalog["native_software"][0]["commands"][0]
        assert command["output_behavior"] == "Writes Gaussian output to stdout."
        assert "declared_outputs" not in command["example_request"]
        assert catalog["managed_program_contract"] == runtime_contract
        assert catalog["submit_analysis_program_request_template"] == analysis_template
        assert "resource_limits" not in catalog["submit_analysis_program_request_template"]
        assert catalog["selection_note"].startswith("Select one exact runtime")
        assert catalog["execution_note"].startswith("Submit authored programs")
        assert catalog["evaluation_resource_budget"] == {
            "source": "service_policy",
            "agent_controllable": False,
            "scope": "per_task",
        }
        assert catalog["execution_timeout_policy"] == {
            "execution_class": "compute",
            "source": "service_policy",
            "agent_controllable": False,
        }
        public_analysis_policy = json.dumps(
            {
                "template": catalog["submit_analysis_program_request_template"],
                "budget": catalog["evaluation_resource_budget"],
                "timeout": catalog["execution_timeout_policy"],
            },
            ensure_ascii=False,
        )
        assert "cpu_cores" not in public_analysis_policy
        assert "memory_mb" not in public_analysis_policy
        assert "timeout_seconds" not in public_analysis_policy
        serialized = json.dumps(catalog, ensure_ascii=False)
        assert "/secret/" not in serialized
    finally:
        _close(adapter)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "inputs",
            [{"name": "input", "source_path": None, "target_path": "inputs/input"}],
            "source_path",
        ),
        (
            "outputs",
            [
                {
                    "name": "result",
                    "path": "code/result.json",
                    "semantic_type": "Table",
                }
            ],
            "outputs",
        ),
    ],
)
def test_validate_analysis_program_rejects_unresolved_inputs_and_unsafe_outputs(
    adapter_factory,
    field: str,
    value: list[dict],
    message: str,
) -> None:
    """Analysis 请求不能把 null 输入或 code/ 下的输出交给 MiniChem。"""

    session = FakeMcpSession()
    adapter, _ = adapter_factory(session)
    try:
        request = {"runtime": "core", "script_path": "code/analysis.py", field: value}
        with pytest.raises(MiniChemMcpProtocolError, match=message):
            adapter.validate_analysis_program(request, "analysis-contract-1")
        assert session.calls == []
    finally:
        _close(adapter)


def test_ensure_workspace_text_reuses_identical_file_and_rejects_conflict(adapter_factory) -> None:
    """工作区写入只允许 code/，同内容复用，不同内容拒绝且不覆盖。"""

    content = "%chk=benzaldehyde.chk\n"
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    session = FakeMcpSession(
        tool_results={
            "read_workspace_text": {
                "status": "success",
                "path": "code/input.com",
                "size_bytes": len(content.encode("utf-8")),
                "sha256": digest,
                "encoding": "utf-8",
                "truncated": False,
                "content": content,
            }
        }
    )
    adapter, _ = adapter_factory(session)
    try:
        reused = adapter.ensure_workspace_text("code/input.com", content)
        assert reused["reused"] is True
        assert reused["sha256"] == digest
        assert [call.name for call in session.calls] == ["read_workspace_text"]

        with pytest.raises(MiniChemMcpProtocolError, match="内容不一致"):
            adapter.ensure_workspace_text("code/input.com", "different")
        with pytest.raises(MiniChemMcpProtocolError, match="code/"):
            adapter.ensure_workspace_text("outputs/input.com", content)
        with pytest.raises(MiniChemMcpProtocolError, match="安全"):
            adapter.ensure_workspace_text("code/../secret", content)
    finally:
        _close(adapter)


def test_ensure_workspace_text_writes_missing_file_then_reads_and_verifies(adapter_factory) -> None:
    """缺失文件只使用 overwrite=false 写入，并在写入后再次读取核对哈希。"""

    content = "input\n"
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    session = FakeMcpSession(
        tool_results={
            "read_workspace_text": [
                {
                    "status": "invalid_request",
                    "error": {"exception_type": "FileNotFoundError", "code": "missing"},
                },
                {
                    "status": "success",
                    "path": "code/input.com",
                    "size_bytes": len(content.encode("utf-8")),
                    "sha256": digest,
                    "encoding": "utf-8",
                    "truncated": False,
                    "content": content,
                },
            ],
            "write_workspace_text": {
                "status": "success",
                "path": "code/input.com",
                "size_bytes": len(content.encode("utf-8")),
                "sha256": digest,
                "encoding": "utf-8",
            },
        }
    )
    adapter, _ = adapter_factory(session)
    try:
        result = adapter.ensure_workspace_text("code/input.com", content)
        assert result["reused"] is False
        assert result["sha256"] == digest
        assert [call.name for call in session.calls] == [
            "read_workspace_text",
            "write_workspace_text",
            "read_workspace_text",
        ]
        assert session.calls[1].arguments["request"] == {
            "path": "code/input.com",
            "content": content,
            "overwrite": False,
        }
    finally:
        _close(adapter)


def test_read_workspace_text_paginates_large_content_and_reassembles_at_eof(adapter_factory) -> None:
    """adapter 传递字符 offset，并核对服务端返回的分页元数据。"""

    content = "甲乙丙丁戊"
    session = FakeMcpSession(workspace_files={"outputs/large.txt": content})
    adapter, _ = adapter_factory(session)
    try:
        pages: list[str] = []
        offset = 0
        while True:
            page = adapter.read_workspace_text(
                "outputs/large.txt", max_chars=2, offset=offset
            )
            pages.append(page["content"])
            if page["next_offset"] is None:
                break
            offset = page["next_offset"]

        assert "".join(pages) == content
        assert page["offset"] == len(content) - 1
        assert page["returned_chars"] == 1
        assert page["total_chars"] == len(content)
        assert page["truncated"] is False
        eof = adapter.read_workspace_text(
            "outputs/large.txt", max_chars=2, offset=len(content)
        )
        assert eof["content"] == ""
        assert eof["next_offset"] is None
        far_eof = adapter.read_workspace_text(
            "outputs/large.txt", max_chars=2, offset=10_000_001
        )
        assert far_eof["offset"] == 10_000_001
        assert far_eof["content"] == ""
        assert far_eof["next_offset"] is None
    finally:
        _close(adapter)


def test_read_workspace_text_passes_and_checks_json_pointer(adapter_factory) -> None:
    """adapter 传递 RFC 6901 pointer，并保留服务端选择身份。"""

    content = '{"a/b":{"~key":["zero",{"value":2}]}}'
    session = FakeMcpSession(workspace_files={"outputs/result.json": content})
    adapter, _ = adapter_factory(session)
    try:
        result = adapter.read_workspace_text(
            "outputs/result.json", json_pointer="/a~1b/~0key/1", max_chars=10
        )
        assert result["json_pointer"] == "/a~1b/~0key/1"
        assert result["content"] == '{"value":2}'[:10]
        assert session.calls[-1].arguments["request"] == {
            "path": "outputs/result.json",
            "max_chars": 10,
            "offset": 0,
            "json_pointer": "/a~1b/~0key/1",
        }
    finally:
        _close(adapter)


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (
            {
                "status": "success",
                "path": "outputs/data.txt",
                "size_bytes": 3,
                "sha256": "a" * 64,
                "encoding": "utf-8",
                "offset": 1,
                "returned_chars": 1,
                "next_offset": 1,
                "total_chars": 3,
                "truncated": True,
                "content": "x",
            },
            "offset",
        ),
        (
            {
                "status": "success",
                "path": "outputs/data.txt",
                "size_bytes": 3,
                "sha256": "not-a-sha256",
                "encoding": "utf-8",
                "offset": 0,
                "returned_chars": 1,
                "next_offset": None,
                "total_chars": 1,
                "truncated": False,
                "content": "x",
            },
            "sha256",
        ),
    ],
)
def test_read_workspace_text_rejects_server_offset_or_hash_drift(
    adapter_factory, response: dict, message: str
) -> None:
    """adapter 不能接受串页的服务端 offset 或变化的文件哈希。"""

    session = FakeMcpSession(tool_results={"read_workspace_text": response})
    adapter, _ = adapter_factory(session)
    try:
        with pytest.raises(MiniChemMcpProtocolError, match=message):
            adapter.read_workspace_text("outputs/data.txt", max_chars=1)
    finally:
        _close(adapter)


def test_read_workspace_text_path_mismatch_does_not_echo_server_path(
    adapter_factory,
) -> None:
    """协议路径不一致时不得把服务端返回的本机路径带入异常。"""

    leaked_path = "/private/runtime/secret.txt"
    session = FakeMcpSession(
        tool_results={
            "read_workspace_text": {
                "status": "success",
                "path": leaked_path,
                "size_bytes": 1,
                "sha256": "a" * 64,
                "encoding": "utf-8",
                "offset": 0,
                "returned_chars": 1,
                "next_offset": None,
                "total_chars": 1,
                "truncated": False,
                "json_pointer": None,
                "content": "x",
            }
        }
    )
    adapter, _ = adapter_factory(session)
    try:
        with pytest.raises(MiniChemMcpProtocolError) as error:
            adapter.read_workspace_text("outputs/data.txt", max_chars=1)
        assert leaked_path not in str(error.value)
    finally:
        _close(adapter)


def test_read_workspace_text_rejects_hash_drift_between_pages(adapter_factory) -> None:
    """同一文件的连续页不能在服务端切换到另一份内容。"""

    session = FakeMcpSession(
        tool_results={
            "read_workspace_text": [
                {
                    "status": "success",
                    "path": "outputs/data.txt",
                    "size_bytes": 2,
                    "sha256": "a" * 64,
                    "encoding": "utf-8",
                    "offset": 0,
                    "returned_chars": 1,
                    "next_offset": 1,
                    "total_chars": 2,
                    "truncated": True,
                    "content": "x",
                },
                {
                    "status": "success",
                    "path": "outputs/data.txt",
                    "size_bytes": 2,
                    "sha256": "b" * 64,
                    "encoding": "utf-8",
                    "offset": 1,
                    "returned_chars": 1,
                    "next_offset": None,
                    "total_chars": 2,
                    "truncated": False,
                    "content": "y",
                },
            ]
        }
    )
    adapter, _ = adapter_factory(session)
    try:
        first = adapter.read_workspace_text("outputs/data.txt", max_chars=1)
        with pytest.raises(MiniChemMcpProtocolError, match="sha256"):
            adapter.read_workspace_text(
                "outputs/data.txt", max_chars=1, offset=first["next_offset"]
            )
    finally:
        _close(adapter)


def test_read_workspace_text_checks_caller_expected_hash(adapter_factory) -> None:
    """ArtifactRef 携带的期望哈希不匹配时，adapter 在首个分页就拒绝响应。"""

    content = "stable"
    session = FakeMcpSession(workspace_files={"outputs/data.txt": content})
    adapter, _ = adapter_factory(session)
    try:
        with pytest.raises(MiniChemMcpProtocolError, match="expected_sha256"):
            adapter.read_workspace_text(
                "outputs/data.txt",
                max_chars=2,
                expected_sha256="a" * 64,
            )
    finally:
        _close(adapter)


def test_validate_open_execution_injects_operation_id_without_mutating_request(adapter_factory) -> None:
    """native/analysis 校验统一注入安全 operation_id，并保留服务结构化结果。"""

    native = {"software_id": "gaussian", "executable": "g16", "arguments": []}
    analysis = {"runtime": "core", "script_path": "code/analysis.py"}
    session = FakeMcpSession(
        tool_results={
            "validate_native_job": lambda payload: {
                "status": "success",
                "valid": True,
                "operation_id": payload["operation_id"],
            },
            "validate_analysis_program": lambda payload: {
                "status": "success",
                "valid": True,
                "operation_id": payload["operation_id"],
            },
        }
    )
    adapter, _ = adapter_factory(session)
    try:
        assert adapter.validate_native_job(native, "native-op-1")["valid"] is True
        assert adapter.validate_analysis_program(analysis, "analysis-op-1")["valid"] is True
        assert "operation_id" not in native
        assert "operation_id" not in analysis
        assert session.calls[-2].arguments["request"]["operation_id"] == "native-op-1"
        assert session.calls[-1].arguments["request"]["operation_id"] == "analysis-op-1"
        with pytest.raises(ValueError, match="operation_id"):
            adapter.validate_native_job(native, "bad/op")
        with pytest.raises(MiniChemMcpProtocolError, match="不一致"):
            adapter.validate_analysis_program(
                {**analysis, "operation_id": "other"},
                "analysis-op-1",
            )
    finally:
        _close(adapter)
