"""MiniChem native operation crash-window 恢复接口的确定性合同测试。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from chemistry_multiagent.tools.minichem_adapter import (
    MiniChemMcpAdapter,
    MiniChemMcpError,
    MiniChemMcpProtocolError,
)
from tests.fakes.minichem_mcp import FakeMcpSession


def _native_request() -> dict[str, Any]:
    """构造不会触发真实软件的最小 native 请求。"""

    return {
        "software_id": "xtb",
        "executable": "xtb",
        "arguments": [],
        "staged_inputs": [],
        "stdin_target": None,
        "resource_limits": {"cpu_cores": 1, "memory_mb": 128, "gpu_count": 0},
        "calculation_intent": "single_point",
        "label": "native-recovery-contract",
        "parent_job_id": None,
    }


def _adapter(session: FakeMcpSession, operation_root: Path) -> MiniChemMcpAdapter:
    """打开注入 fake session 的 adapter。"""

    adapter = MiniChemMcpAdapter(
        session_factory=lambda session=session: session,
        operation_root=operation_root,
        poll_interval=0.0,
    )
    adapter.open()
    return adapter


def _completed_state(job_id: str, **extra: Any) -> dict[str, Any]:
    """构造带指定作业身份的终态 get 响应。"""

    job = {"job_id": job_id, "status": "success", **extra}
    return {"status": "success", "job": job, "terminal": True}


def test_recover_native_operation_with_receipt_only_gets_and_collects(
    tmp_path: Path,
) -> None:
    """已有合法 receipt 时恢复只 get/collect，恢复会话不产生 submit。"""

    operation_id = "native-recovery-valid-receipt"
    operation_root = tmp_path / "operations"
    submit_session = FakeMcpSession()
    submit_adapter = _adapter(submit_session, operation_root)
    try:
        submitted = submit_adapter.submit_native_job(_native_request(), operation_id)
        request_fingerprint = submitted["request_fingerprint"]
        job_id = submitted["job_id"]
    finally:
        submit_adapter.close()

    recovery_session = FakeMcpSession(
        job_states={job_id: [_completed_state(job_id)]},
    )
    recovery_adapter = _adapter(recovery_session, operation_root)
    try:
        recovered = recovery_adapter.recover_native_operation(
            operation_id,
            request_fingerprint=request_fingerprint,
            collect=True,
        )
        assert recovered["status"] == "completed"
        assert recovered["operation_id"] == operation_id
        assert recovered["job_id"] == job_id
        assert recovered["tool"] == "submit_native_job"
        submit_call = submit_session.calls[0]
        assert submit_call.arguments["request"]["caller_request_fingerprint"] == (
            request_fingerprint
        )
        persisted = json.loads(
            (operation_root / f"{operation_id}.json").read_text(encoding="utf-8")
        )
        assert "caller_request_fingerprint" not in persisted["request"]
        assert recovery_session.native_submit_count == 0
        assert [call.name for call in recovery_session.calls] == [
            "get_execution_job",
            "collect_execution_job",
        ]
    finally:
        recovery_adapter.close()


@pytest.mark.parametrize(
    ("case", "job_extra", "expected_message"),
    [
        (
            "metadata_missing",
            {"operation_id": "native-recovery-missing-receipt", "operation_fingerprint": "b" * 64},
            "tool=submit_native_job",
        ),
        (
            "operation_mismatch",
            {
                "operation_id": "another-operation",
                "operation_fingerprint": "a" * 64,
                "tool": "submit_native_job",
                "request_fingerprint": "a" * 64,
            },
            "operation_id",
        ),
        (
            "fingerprint_mismatch",
            {
                "operation_id": "native-recovery-missing-receipt",
                "operation_fingerprint": "b" * 64,
                "tool": "submit_native_job",
                "request_fingerprint": "b" * 64,
            },
            "request_fingerprint",
        ),
    ],
)
def test_recover_native_operation_without_receipt_fails_closed_without_submit(
    tmp_path: Path,
    case: str,
    job_extra: dict[str, Any],
    expected_message: str,
) -> None:
    """缺 receipt 时只有完整远端身份证据才能恢复，任何不足都不重提。"""

    del case
    operation_id = "native-recovery-missing-receipt"
    # 该值是依据 MiniChem 公开 operation→job 算法预先计算的协议样例，
    # 测试不在被测函数旁重算算法，避免把同一错误复制到断言中。
    job_id = "job_ac01953bc26a854c248f13b140fab385"
    session = FakeMcpSession(
        job_states={job_id: [_completed_state(job_id, **job_extra)]},
    )
    adapter = _adapter(session, tmp_path / "operations")
    try:
        with pytest.raises(MiniChemMcpProtocolError, match=expected_message):
            adapter.recover_native_operation(
                operation_id,
                request_fingerprint="a" * 64,
                collect=True,
            )
        assert session.native_submit_count == 0
        assert [call.name for call in session.calls] == ["get_execution_job"]
    finally:
        adapter.close()


def test_recover_native_operation_unknown_remote_job_fails_closed_without_submit(
    tmp_path: Path,
) -> None:
    """确定性 job 不存在时只报告未知作业，不能创建 receipt 或重提。"""

    operation_id = "native-recovery-missing-receipt"
    session = FakeMcpSession()
    operation_root = tmp_path / "operations"
    adapter = _adapter(session, operation_root)
    try:
        with pytest.raises(MiniChemMcpProtocolError, match="unknown_job"):
            adapter.recover_native_operation(
                operation_id,
                request_fingerprint="a" * 64,
                collect=True,
            )
        assert session.native_submit_count == 0
        assert [call.name for call in session.calls] == ["get_execution_job"]
        assert not (operation_root / f"{operation_id}.json").exists()
    finally:
        adapter.close()


def test_recover_native_operation_persists_minimal_receipt_only_after_remote_proof(
    tmp_path: Path,
) -> None:
    """未来协议返回 native/caller fingerprint 时才允许补写最小 receipt。"""

    operation_id = "native-recovery-missing-receipt"
    job_id = "job_ac01953bc26a854c248f13b140fab385"
    fingerprint = "a" * 64
    session = FakeMcpSession(
        job_states={
            job_id: [
                _completed_state(
                    job_id,
                    operation_id=operation_id,
                    job_type="native_software",
                    caller_request_fingerprint=fingerprint,
                )
            ]
        }
    )
    operation_root = tmp_path / "operations"
    adapter = _adapter(session, operation_root)
    try:
        recovered = adapter.recover_native_operation(
            operation_id,
            request_fingerprint=fingerprint,
            collect=False,
        )
        assert recovered["status"] == "completed"
        receipt = json.loads(
            (operation_root / f"{operation_id}.json").read_text(encoding="utf-8")
        )
        assert receipt == {
            "schema_version": 1,
            "receipt_kind": "minimal_native_recovery",
            "operation_id": operation_id,
            "job_id": job_id,
            "tool": "submit_native_job",
            "request_fingerprint": fingerprint,
            "operation_fingerprint": fingerprint,
        }
        assert session.native_submit_count == 0
        assert [call.name for call in session.calls] == ["get_execution_job"]
    finally:
        adapter.close()


def test_submit_native_job_reused_operation_recovers_without_second_submit(
    tmp_path: Path,
) -> None:
    """远端已占用 operation 时，submission_in_progress 只转为只读恢复。"""

    operation_id = "native-recovery-reused-operation"
    job_id = "job_f6b4cf44d114823817877afeacba254e"
    caller_fingerprint: dict[str, str] = {}

    def reused_submit(payload: dict[str, Any]) -> dict[str, Any]:
        caller_fingerprint["value"] = str(payload["caller_request_fingerprint"])
        return {
            "status": "submission_in_progress",
            "operation_id": operation_id,
            "operation_fingerprint": "b" * 64,
            "job_id": job_id,
            "job_status": "queued",
            "idempotent_reused": True,
            "error": {"code": "operation_record_incomplete"},
        }

    def reused_get(_payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": "success",
            "job": {
                "job_id": job_id,
                "status": "queued",
                "operation_id": operation_id,
                "job_type": "native_software",
                "caller_request_fingerprint": caller_fingerprint["value"],
            },
            "terminal": False,
        }

    session = FakeMcpSession(
        tool_results={
            "submit_native_job": reused_submit,
            "get_execution_job": reused_get,
        }
    )
    # 这次 fake 调用模拟的是“历史 submit 已成功”的服务端事实；本次 adapter
    # submit endpoint 响应由 tool_results 直接返回，不能把它计为第二个 supervisor。
    session.native_submit_count = 1
    adapter = _adapter(session, tmp_path / "operations")
    try:
        result = adapter.submit_native_job(_native_request(), operation_id)
        assert result["status"] == "waiting"
        assert result["operation_id"] == operation_id
        assert result["job_id"] == job_id
        assert result["tool"] == "submit_native_job"
        assert session.native_submit_count == 1
        assert [call.name for call in session.calls] == [
            "submit_native_job",
            "get_execution_job",
        ]
        receipt = json.loads(
            (tmp_path / "operations" / f"{operation_id}.json").read_text(
                encoding="utf-8"
            )
        )
        assert receipt["request"]["operation_id"] == operation_id
        assert "caller_request_fingerprint" not in receipt["request"]
    finally:
        adapter.close()


def test_submit_native_job_transport_loss_recovers_without_retrying_submit(
    tmp_path: Path,
) -> None:
    """submit 响应丢失时只读查询已占用 job，不把传输重试当成新提交。"""

    operation_id = "native-recovery-submit-transport"
    job_id = "job_5c54c12e7970eec73260a605983e4d93"
    caller_fingerprint: dict[str, str] = {}

    def lost_submit(payload: dict[str, Any]) -> dict[str, Any]:
        caller_fingerprint["value"] = str(payload["caller_request_fingerprint"])
        raise ConnectionError("submit response lost after remote reservation")

    def recovered_get(_payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": "success",
            "job": {
                "job_id": job_id,
                "status": "queued",
                "operation_id": operation_id,
                "job_type": "native_software",
                "caller_request_fingerprint": caller_fingerprint["value"],
            },
            "terminal": False,
        }

    session = FakeMcpSession(
        tool_results={
            "submit_native_job": lost_submit,
            "get_execution_job": recovered_get,
        }
    )
    # fake 不会在抛出异常的 submit 回调中自动计数，这里保留远端历史 supervisor。
    session.native_submit_count = 1
    adapter = _adapter(session, tmp_path / "operations")
    try:
        result = adapter.submit_native_job(_native_request(), operation_id)
        assert result["status"] == "waiting"
        assert result["operation_id"] == operation_id
        assert result["job_id"] == job_id
        assert session.native_submit_count == 1
        assert [call.name for call in session.calls] == [
            "submit_native_job",
            "get_execution_job",
        ]
        receipt = json.loads(
            (tmp_path / "operations" / f"{operation_id}.json").read_text(
                encoding="utf-8"
            )
        )
        assert receipt["request"]["operation_id"] == operation_id
        assert "caller_request_fingerprint" not in receipt["request"]
    finally:
        adapter.close()


def test_submission_in_progress_without_remote_proof_persists_pending_receipt(
    tmp_path: Path,
) -> None:
    """远端身份证据暂缺时，后续同 operation 只能继续只读 recovery。"""

    operation_id = "native-recovery-pending-no-proof"
    job_id = "job_2892e1187a42fccc15175bda96eb69af"
    submit_calls: list[dict[str, Any]] = []

    def pending_submit(payload: dict[str, Any]) -> dict[str, Any]:
        submit_calls.append(payload)
        return {
            "status": "submission_in_progress",
            "operation_id": operation_id,
            "operation_fingerprint": "b" * 64,
            "job_id": job_id,
            "job_status": "queued",
            "idempotent_reused": True,
            "error": {"code": "operation_record_incomplete"},
        }

    def unproven_get(_payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": "success",
            "job": {"job_id": job_id, "status": "queued"},
            "terminal": False,
        }

    session = FakeMcpSession(
        tool_results={
            "submit_native_job": pending_submit,
            "get_execution_job": unproven_get,
        }
    )
    operation_root = tmp_path / "operations"
    adapter = _adapter(session, operation_root)
    try:
        first = adapter.submit_native_job(_native_request(), operation_id)
        second = adapter.submit_native_job(_native_request(), operation_id)

        assert first["status"] == "technical_pause"
        assert second["status"] == "technical_pause"
        assert len(submit_calls) == 1
        assert [call.name for call in session.calls] == [
            "submit_native_job",
            "get_execution_job",
            "get_execution_job",
        ]
        receipt = json.loads(
            (operation_root / f"{operation_id}.json").read_text(encoding="utf-8")
        )
        assert receipt["receipt_kind"] == "ambiguous_native_submission"
        assert receipt["operation_id"] == operation_id
        assert receipt["job_id"] == job_id
        assert receipt["request_fingerprint"] == first["request_fingerprint"]
        assert "caller_request_fingerprint" not in receipt["request"]
    finally:
        adapter.close()


def test_submit_transport_loss_without_remote_proof_persists_pending_receipt(
    tmp_path: Path,
) -> None:
    """submit 传输异常且无法证明远端身份时，重试同 operation 不得再次 submit。"""

    operation_id = "native-recovery-pending-transport"
    job_id = "job_66bcec84f31920551d4ccc0977f55e2a"
    submit_calls: list[dict[str, Any]] = []

    def lost_submit(payload: dict[str, Any]) -> dict[str, Any]:
        submit_calls.append(payload)
        raise ConnectionError("submit response lost after remote reservation")

    def unproven_get(_payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": "success",
            "job": {"job_id": job_id, "status": "queued"},
            "terminal": False,
        }

    session = FakeMcpSession(
        tool_results={
            "submit_native_job": lost_submit,
            "get_execution_job": unproven_get,
        }
    )
    operation_root = tmp_path / "operations"
    adapter = _adapter(session, operation_root)
    try:
        with pytest.raises(MiniChemMcpError, match="已阻止重复提交"):
            adapter.submit_native_job(_native_request(), operation_id)
        second = adapter.submit_native_job(_native_request(), operation_id)
        assert second["status"] == "technical_pause"

        assert len(submit_calls) == 1
        assert [call.name for call in session.calls] == [
            "submit_native_job",
            "get_execution_job",
            "get_execution_job",
        ]
        receipt = json.loads(
            (operation_root / f"{operation_id}.json").read_text(encoding="utf-8")
        )
        assert receipt["receipt_kind"] == "ambiguous_native_submission"
        assert receipt["operation_id"] == operation_id
        assert receipt["job_id"] == job_id
    finally:
        adapter.close()


@pytest.mark.parametrize("receipt_content", ["[]", "null", '"not-an-object"', "1"])
def test_non_object_operation_receipt_is_corrupt_and_never_probed(
    tmp_path: Path,
    receipt_content: str,
) -> None:
    """合法 JSON 但不是 object 的 operation receipt 必须 fail closed。"""

    operation_id = "native-recovery-corrupt-json-array"
    operation_root = tmp_path / "operations"
    operation_root.mkdir()
    receipt_path = operation_root / f"{operation_id}.json"
    receipt_path.write_text(receipt_content, encoding="utf-8")
    session = FakeMcpSession()
    adapter = _adapter(session, operation_root)
    try:
        with pytest.raises(MiniChemMcpError, match="operation 记录不是 object"):
            adapter.recover_native_operation(
                operation_id,
                request_fingerprint="a" * 64,
                collect=True,
            )
        assert session.calls == []
        assert receipt_path.read_text(encoding="utf-8") == receipt_content
    finally:
        adapter.close()


def test_recover_native_operation_corrupt_receipt_fails_before_remote_probe(
    tmp_path: Path,
) -> None:
    """本地 receipt JSON 损坏时不读取远端，也不触发第二次 submit。"""

    operation_id = "native-recovery-corrupt-receipt"
    operation_root = tmp_path / "operations"
    operation_root.mkdir()
    (operation_root / f"{operation_id}.json").write_text(
        "{not-json",
        encoding="utf-8",
    )
    session = FakeMcpSession()
    adapter = _adapter(session, operation_root)
    try:
        with pytest.raises(MiniChemMcpError, match="operation 记录不可读"):
            adapter.recover_native_operation(
                operation_id,
                request_fingerprint="a" * 64,
                collect=True,
            )
        assert session.native_submit_count == 0
        assert session.calls == []
    finally:
        adapter.close()
