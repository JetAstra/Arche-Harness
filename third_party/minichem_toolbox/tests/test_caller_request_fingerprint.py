"""MiniChem native operation 的调用方请求指纹协议测试。"""

from __future__ import annotations

import json

import pytest

from minichem_mcp_tools import open_execution
from minichem_mcp_tools.execution_models import JobStatusRequest
from minichem_mcp_tools.tracing import execute_traced

from test_minichem_operation_idempotency import _native_request, submission_sandbox


def test_caller_request_fingerprint_is_hidden_but_is_persisted(
    submission_sandbox,
) -> None:
    """调用方指纹不进入公开 Schema，却随新 native operation 落盘。"""

    fingerprint = "a" * 64
    request = _native_request("caller-fingerprint-normal").model_copy(
        update={"caller_request_fingerprint": fingerprint}
    )
    result = open_execution.submit_native_job(request)

    schema_text = json.dumps(
        type(request).model_json_schema(), ensure_ascii=False, sort_keys=True
    )
    assert "caller_request_fingerprint" not in schema_text
    assert "caller_request_fingerprint" not in request.model_dump(mode="json")
    assert result["caller_request_fingerprint"] == fingerprint

    job_directory = submission_sandbox.root / result["job_directory"]
    receipt = json.loads(
        (job_directory / "operation_receipt.json").read_text(encoding="utf-8")
    )
    request_record = json.loads(
        (job_directory / "request.json").read_text(encoding="utf-8")
    )
    status = json.loads(
        (job_directory / "status.json").read_text(encoding="utf-8")
    )
    for record in (receipt, request_record, status):
        assert record["job_type"] == "native_software"
        assert record["operation_id"] == request.operation_id
        assert record["job_id"] == result["job_id"]
        assert record["caller_request_fingerprint"] == fingerprint

    observed = open_execution.get_execution_job(
        JobStatusRequest(job_id=result["job_id"])
    )
    assert observed["job_id"] == result["job_id"]
    assert "caller_request_fingerprint" not in observed
    assert observed["job"] == {"job_id": result["job_id"], "status": "queued"}


def test_existing_operation_conflicting_caller_fingerprint_fails_closed(
    submission_sandbox,
) -> None:
    """同一 operation 的调用方指纹变化时不启动第二个 supervisor。"""

    operation_id = "caller-fingerprint-conflict"
    first = open_execution.submit_native_job(
        _native_request(operation_id).model_copy(
            update={"caller_request_fingerprint": "a" * 64}
        )
    )
    conflict = open_execution.submit_native_job(
        _native_request(operation_id).model_copy(
            update={"caller_request_fingerprint": "b" * 64}
        )
    )

    assert first["status"] == "success"
    assert conflict["status"] == "invalid_request"
    assert conflict["error"]["code"] == "operation_id_conflict"
    assert conflict["error"]["job_id"] == first["job_id"]
    assert conflict["error"]["existing_caller_request_fingerprint"] == "a" * 64
    assert len(submission_sandbox.calls) == 1


def test_old_operation_without_caller_fingerprint_rejects_new_proof(
    submission_sandbox,
) -> None:
    """旧 receipt 缺少调用方证据时，带新指纹的提交只能安全失败。"""

    operation_id = "caller-fingerprint-missing"
    first = open_execution.submit_native_job(_native_request(operation_id))
    job_directory = submission_sandbox.root / first["job_directory"]
    receipt_path = job_directory / "operation_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt.pop("caller_request_fingerprint", None)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    retry = open_execution.submit_native_job(
        _native_request(operation_id).model_copy(
            update={"caller_request_fingerprint": "a" * 64}
        )
    )

    assert retry["status"] == "invalid_request"
    assert retry["error"]["code"] == "operation_caller_fingerprint_missing"
    assert retry["job_id"] == first["job_id"]
    assert len(submission_sandbox.calls) == 1


def test_legacy_request_without_caller_fingerprint_remains_compatible(
    submission_sandbox,
) -> None:
    """未提供新字段的旧请求仍可提交和复用已有 operation。"""

    request = _native_request("caller-fingerprint-legacy")
    first = open_execution.submit_native_job(request)
    second = open_execution.submit_native_job(request)

    assert first["status"] == "success"
    assert second["status"] == "submission_in_progress"
    assert second["job_id"] == first["job_id"]
    assert second["caller_request_fingerprint"] == first["caller_request_fingerprint"]
    assert len(submission_sandbox.calls) == 1


def test_get_execution_job_rejects_conflicting_status_and_receipt_identity(
    submission_sandbox,
) -> None:
    """status 与 receipt 的调用方指纹漂移时不输出可恢复身份。"""

    operation_id = "caller-fingerprint-get-conflict"
    first = open_execution.submit_native_job(
        _native_request(operation_id).model_copy(
            update={"caller_request_fingerprint": "a" * 64}
        )
    )
    job_directory = submission_sandbox.root / first["job_directory"]
    receipt_path = job_directory / "operation_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["caller_request_fingerprint"] = "b" * 64
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    observed = open_execution.get_execution_job(
        JobStatusRequest(job_id=first["job_id"])
    )

    assert observed["status"] == "failed"
    assert observed["valid"] is False
    assert observed["error"]["code"] == "operation_identity_conflict"
    assert "caller_request_fingerprint" not in observed
    assert "job" not in observed


@pytest.mark.parametrize("location", ["receipt", "status"])
def test_get_execution_job_rejects_conflicting_canonical_fingerprint_aliases(
    submission_sandbox,
    location: str,
) -> None:
    """receipt/status 的 canonical fingerprint 别名漂移时不输出恢复身份。"""

    first = open_execution.submit_native_job(
        _native_request(f"canonical-fingerprint-{location}").model_copy(
            update={"caller_request_fingerprint": "c" * 64}
        )
    )
    job_directory = submission_sandbox.root / first["job_directory"]
    if location == "receipt":
        path = job_directory / "operation_receipt.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record["operation_fingerprint"] = "d" * 64
    else:
        path = job_directory / "status.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record["fingerprint"] = "d" * 64
    path.write_text(json.dumps(record), encoding="utf-8")

    observed = open_execution.get_execution_job(
        JobStatusRequest(job_id=first["job_id"])
    )

    assert observed["status"] == "failed"
    assert observed["valid"] is False
    assert observed["error"]["code"] == "operation_identity_conflict"
    assert "operation_fingerprint" not in observed
    assert "caller_request_fingerprint" not in observed


def test_tracing_redacts_caller_fingerprint_only_in_persisted_copies(
    submission_sandbox,
) -> None:
    """落盘结果和轨迹隐藏调用方指纹，但 transport 返回值保持完整。"""

    caller_fingerprint = "e" * 64
    canonical_fingerprint = "f" * 64
    returned = execute_traced(
        "identity-redaction-probe",
        {
            "request": {
                "caller_request_fingerprint": caller_fingerprint,
                "operation_fingerprint": canonical_fingerprint,
            }
        },
        lambda: {
            "caller_request_fingerprint": caller_fingerprint,
            "operation_fingerprint": canonical_fingerprint,
            "nested": {"existing_caller_request_fingerprint": caller_fingerprint},
        },
        capture_artifacts=False,
    )

    assert returned["caller_request_fingerprint"] == caller_fingerprint
    assert returned["operation_fingerprint"] == canonical_fingerprint
    result_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (submission_sandbox.root / "_tool_results").glob(
            "*_identity-redaction-probe.json"
        )
    )
    trace_text = (submission_sandbox.root / "_tool_trace.jsonl").read_text(
        encoding="utf-8"
    )
    persisted_text = result_text + trace_text
    assert caller_fingerprint not in persisted_text
    assert "caller_request_fingerprint" not in persisted_text
    assert "existing_caller_request_fingerprint" not in persisted_text
    assert canonical_fingerprint in persisted_text
