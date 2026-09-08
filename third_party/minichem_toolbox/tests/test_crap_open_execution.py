"""覆盖 open_execution 的身份、队列、环境和结果传输分支。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from minichem_mcp_tools import open_execution
from minichem_mcp_tools.execution_models import (
    ExecutionJobReference,
    ExecutionOperationReference,
    JobCancelRequest,
    JobCollectRequest,
    JobStatusRequest,
    ScientificArtifactTextReadRequest,
    VerifyExecutionReferencesRequest,
    WorkspaceTextWriteRequest,
)
from minichem_toolbox.process_identity import ProcessIdentity


def _scope(tmp_path: Path) -> SimpleNamespace:
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    workspace.chmod(0o700)
    return SimpleNamespace(scope_id="scope-1", workspace=workspace)


def _status(job_id: str = "job_" + "a" * 32, state: str = "running") -> dict:
    return {
        "job_id": job_id,
        "job_type": "native_software",
        "status": state,
        "metadata": {},
        "supervisor_pid": 123,
    }


VALID_JOB_ID = "job_" + "a" * 32


def test_operation_receipt_identity_paths_are_fail_closed(monkeypatch, tmp_path: Path) -> None:
    job_directory = tmp_path / "job"
    job_directory.mkdir()
    status_path = job_directory / "status.json"
    status_path.write_text(json.dumps({"status": "queued"}), encoding="utf-8")
    receipt_path = job_directory / "operation_receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "fingerprint": "f",
                "submission_key": "submission-1",
                "caller_request_fingerprint": "caller",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(open_execution, "_job_directory", lambda *_args, **_kwargs: job_directory)
    monkeypatch.setattr(open_execution, "relative_workspace_path", lambda path: str(path))

    reused = open_execution._existing_operation_result(
        "operation", "job", "f", submission_key="submission-1", caller_request_fingerprint="caller"
    )
    assert reused["status"] == "submission_in_progress"
    assert reused["job_status"] == "queued"
    assert open_execution._existing_operation_result(
        "operation", "job", "other", submission_key="submission-1"
    )["error"]["code"] == "submission_key_conflict"
    assert open_execution._existing_operation_result(
        "operation", "job", "f", submission_key="other"
    )["error"]["code"] == "operation_record_incomplete"
    receipt_path.write_text(
        json.dumps({"fingerprint": "f", "caller_request_fingerprint": "caller"}),
        encoding="utf-8",
    )
    assert open_execution._existing_operation_result(
        "operation", "job", "f", caller_request_fingerprint="other"
    )["error"]["code"] == "operation_id_conflict"
    receipt_path.write_text(json.dumps({"fingerprint": "f"}), encoding="utf-8")
    assert open_execution._existing_operation_result(
        "operation", "job", "f", caller_request_fingerprint="caller"
    )["error"]["code"] == "operation_caller_fingerprint_missing"
    receipt_path.write_text("not-json", encoding="utf-8")
    assert open_execution._read_existing_operation_receipt(receipt_path) is None


def test_scope_identity_loaders_reject_bad_records(monkeypatch, tmp_path: Path) -> None:
    scope = _scope(tmp_path)
    monkeypatch.setattr(open_execution, "current_scope", lambda: scope)
    directory = tmp_path / "job"
    directory.mkdir()
    good = {"scope_id": scope.scope_id, "scope_workspace": str(scope.workspace)}
    (directory / "status.json").write_text(json.dumps(good), encoding="utf-8")
    assert open_execution._read_existing_scope_records(directory)
    open_execution._validate_existing_job_scope(directory)
    assert open_execution._scope_records_match(scope, [good])
    assert not open_execution._scope_records_match(scope, [{"scope_id": "other"}])
    (directory / "request.json").write_text("[]", encoding="utf-8")
    with pytest.raises(open_execution._ExecutionJobUnavailable):
        open_execution._read_existing_scope_records(directory)
    (directory / "request.json").write_text("{bad", encoding="utf-8")
    with pytest.raises(open_execution._ExecutionJobUnavailable):
        open_execution._read_existing_scope_records(directory)

    identity = {**good, "operation_id": "scope-submission-op", "submission_key": "valid-key"}
    assert open_execution._scope_submission_identity_is_complete(
        [identity, {**identity, "submission_key": "other-key"}]
    ) is False
    assert open_execution._scope_persisted_identity_is_complete(directory, good) is False


def test_scope_submission_identity_and_reference_helpers_cover_conflicts(
    monkeypatch, tmp_path: Path
) -> None:
    scope = _scope(tmp_path)
    monkeypatch.setattr(open_execution, "current_scope", lambda: scope)
    job_id = "job_" + "a" * 32
    reference = ExecutionJobReference(job_id=job_id)
    status = {
        "job_id": job_id,
        "job_type": "native_software",
        "status": "success",
        "operation_id": "operation-1",
        "operation_fingerprint": "f",
    }
    receipt = {
        "schema_version": 1,
        "job_id": job_id,
        "job_type": "native_software",
        "operation_id": "operation-1",
        "operation_fingerprint": "f",
        "fingerprint": "f",
        "scope_id": scope.scope_id,
        "scope_workspace": str(scope.workspace),
        "phase": "success",
    }
    monkeypatch.setattr(open_execution, "_operation_job_id", lambda _value: job_id)
    assert open_execution._canonical_job_reference_identity(reference, status, receipt) == (
        "operation-1",
        "f",
    )
    assert open_execution._canonical_reference_values(status, receipt) == ("operation-1", "f")
    assert open_execution._receipt_fingerprint_aliases_conflict({"fingerprint": "a", "operation_fingerprint": "b"})
    assert open_execution._canonical_job_reference_identity(
        reference, status, {**receipt, "fingerprint": "different"}
    ) is None
    assert open_execution._canonical_job_reference_identity(
        reference, status, {**receipt, "phase": "running"}
    ) is None
    assert not open_execution._job_submission_identity_is_consistent(
        status, {**receipt, "submission_key": "bad key"}, "scope-submission-op"
    )
    assert open_execution._job_submission_identity_is_consistent(status, receipt, "operation-1")
    assert open_execution._reference_operation_identity_matches(
        reference, status, receipt, "operation-1", "f"
    )


def test_operation_identity_evidence_validates_aliases_and_scoped_keys(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "job"
    directory.mkdir()
    status = {
        "job_type": "native_software",
        "operation_id": "scope-submission-op",
        "operation_fingerprint": "f",
        "job_id": "job_1",
        "metadata": {"operation_id": "scope-submission-op"},
    }
    (directory / "operation_receipt.json").write_text(
        json.dumps(
            {
                "fingerprint": "f",
                "operation_fingerprint": "f",
                "operation_id": "scope-submission-op",
                "submission_key": "submission-1",
            }
        ),
        encoding="utf-8",
    )
    evidence = open_execution._operation_identity_evidence(directory, status)
    assert evidence["operation_fingerprint"] == "f"
    assert evidence["submission_key"] == "submission-1"
    with pytest.raises(open_execution._OperationIdentityConflict):
        open_execution._collect_consistent_identity_field(
            {"a": {"value": 1}, "b": {"value": 2}}, {}, "value"
        )
    with pytest.raises(open_execution._OperationIdentityConflict):
        open_execution._validate_scope_operation_identity(
            {"operation_id": "scope-submission-op"}
        )


def test_job_environment_applies_cpu_gpu_private_and_analysis_rules(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(open_execution, "runtime_environment", lambda _runtime: {"PYTHONPATH": "/x/site-packages:/framework", "GAUSS_SCRDIR": "/old"})
    monkeypatch.setattr(open_execution, "SAFE_INHERITED_ENVIRONMENT", ("LANG",))
    monkeypatch.setenv("LANG", "C")
    directory = tmp_path / "job"
    directory.mkdir(mode=0o700)
    directory.chmod(0o700)
    env = open_execution._job_environment(
        "runtime", "job-1", directory, {"cpu_cores": 2, "gpu_count": 0}, {"cpu_ids": [3]}, job_type="native_software"
    )
    assert env["OPENBLAS_NUM_THREADS"] == "1"
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert env["GAUSS_SCRDIR"].startswith(str(directory))
    env = open_execution._job_environment(
        "runtime", "job-2", directory / "analysis", {"cpu_cores": 2, "gpu_count": 2}, {"gpu_ids": [4]}, job_type="programmable_analysis"
    )
    assert env["CUDA_VISIBLE_DEVICES"] == "4"
    assert env["PYTHONPATH"].split(os.pathsep)[0].endswith("analysis")
    assert env["PYTHONNOUSERSITE"] == "1"


def test_pidfd_signal_helpers_fail_closed_and_verify_identity(monkeypatch) -> None:
    assert open_execution._open_pidfd_for_signal(-1) is None
    monkeypatch.setattr(open_execution.os, "pidfd_open", lambda pid, flags: 9, raising=False)
    assert open_execution._open_pidfd_for_signal(12) == 9
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(open_execution.signal, "pidfd_send_signal", lambda fd, sig, *_args: sent.append((fd, sig)), raising=False)
    assert open_execution._send_pidfd_signal(9, 15)
    assert sent == [(9, 15)]
    identity = ProcessIdentity(pid=12, start_time_ticks=2, boot_id="boot")
    monkeypatch.setattr(open_execution, "capture_process_identity", lambda _pid: identity)
    monkeypatch.setattr(open_execution, "_open_pidfd_for_signal", lambda _pid: 9)
    monkeypatch.setattr(open_execution, "_send_pidfd_signal", lambda *_args: True)
    monkeypatch.setattr(open_execution.os, "close", lambda _fd: None)
    assert open_execution._send_verified_supervisor_signal(identity, 15)
    monkeypatch.setattr(open_execution, "capture_process_identity", lambda _pid: ProcessIdentity(pid=12, start_time_ticks=3, boot_id="boot"))
    assert not open_execution._send_verified_supervisor_signal(identity, 15)


def test_cancel_helpers_record_fifo_and_reject_unverified_supervisors(monkeypatch, tmp_path: Path) -> None:
    directory = tmp_path / "job"
    directory.mkdir()
    request = JobCancelRequest(job_id=VALID_JOB_ID)
    status = {"status": "queued", "supervisor_pid": None}
    (directory / "supervisor_spec.json").write_text(
        json.dumps({"gaussian_queue_ticket": {"position": 1}}), encoding="utf-8"
    )
    markers: list[Path] = []
    monkeypatch.setattr(open_execution, "_ensure_private_marker", markers.append)
    result = open_execution._record_queued_gaussian_cancellation(request, directory, status)
    assert result["cancellation_sent"] is True
    assert markers
    result = open_execution._cancel_without_published_supervisor(request, directory, status, None)
    assert result["cancellation_sent"] is True
    (directory / "supervisor_spec.json").write_text(json.dumps({}), encoding="utf-8")
    assert open_execution._cancel_without_published_supervisor(request, directory, status, None)["error"]["code"] == "supervisor_not_started"
    assert open_execution._cancel_without_published_supervisor(request, status, status, True)["error"]["code"] == "supervisor_identity_unavailable"
    monkeypatch.setattr(open_execution, "_read_status", lambda _job: (directory, {"status": "success"}))
    assert open_execution._cancel_execution_job_once(request)["cancellation_sent"] is False


def test_reference_and_artifact_text_helpers_cover_safe_rejections(monkeypatch, tmp_path: Path) -> None:
    request = VerifyExecutionReferencesRequest(
        jobs=[VALID_JOB_ID],
        operations=[ExecutionOperationReference(operation_id="op-1")],
    )
    monkeypatch.setattr(open_execution, "current_scope", lambda: None)
    result = open_execution.verify_execution_references(request)
    assert result["all_valid"] is False
    monkeypatch.setattr(open_execution, "current_scope", lambda: SimpleNamespace(scope_id="s", workspace=tmp_path))
    monkeypatch.setattr(open_execution, "workspace_root", lambda: tmp_path)
    monkeypatch.setattr(open_execution, "ArtifactStore", lambda **_kwargs: (_ for _ in ()).throw(FileNotFoundError()))
    page = open_execution.read_scientific_artifact_text(
        ScientificArtifactTextReadRequest(artifact_id="art_" + "a" * 32)
    )
    assert page["status"] == "unavailable"


def test_collection_and_workspace_helpers_preserve_hidden_files(monkeypatch, tmp_path: Path) -> None:
    directory = tmp_path / "job"
    directory.mkdir()
    (directory / "stdout.log").write_text("ok", encoding="utf-8")
    (directory / "internal.json").write_text("internal", encoding="utf-8")
    monkeypatch.setattr(open_execution, "relative_workspace_path", lambda path: str(path))
    request = JobCollectRequest(job_id=VALID_JOB_ID, max_files=10)
    files = open_execution._collect_execution_output_files(
        request, directory, {"status": "success"}, {"staged_inputs": []}
    )
    assert {item["job_relative_path"] for item in files} == {"stdout.log", "internal.json"}
    assert open_execution._output_file_is_hidden(
        directory / "stdout.log", "stdout.log", {"stdout.log"}, request, set()
    )
    manifest = open_execution._collection_manifest(
        request,
        {"status": "success"},
        {"submission_key": "submission-1"},
        files,
        {"details": {}},
        None,
        None,
    )
    assert manifest["job_submission_key"] == "submission-1"
    monkeypatch.setattr(open_execution, "resolve_workspace_public_path", lambda *_args, **_kwargs: tmp_path / "target.txt")
    monkeypatch.setattr(open_execution, "resolve_workspace_output_path", lambda *_args, **_kwargs: tmp_path / "target.txt")
    monkeypatch.setattr(open_execution, "_write_private_text_atomically", lambda path, text: path.write_text(text, encoding="utf-8"))
    result = open_execution._write_workspace_text_once(WorkspaceTextWriteRequest(path="target.txt", content="hello"))
    assert result["status"] == "success"


def test_private_copy_rejects_non_regular_source(tmp_path: Path) -> None:
    """复制源不是普通文件时必须失败，并关闭已经打开的描述符。"""

    source = tmp_path / "source-directory"
    source.mkdir()
    target = tmp_path / "target" / "copy.txt"

    with pytest.raises(ValueError, match="regular file"):
        open_execution._copy_private_file_atomically(source, target)

    assert not target.exists()
    assert not list(target.parent.glob(f".{target.name}.tmp-*"))


@pytest.mark.parametrize("race", ["symlink", "regular", "missing_temp"])
def test_private_copy_rejects_post_copy_target_races(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    race: str,
) -> None:
    """复制完成后的目标竞争或临时文件丢失都必须失败并清理临时文件。"""

    source = tmp_path / "source.txt"
    source.write_text("payload", encoding="utf-8")
    target = tmp_path / "nested" / "copy.txt"
    original_verify = open_execution._verify_private_directory_identity
    verification_calls = 0

    def create_race(path: Path, descriptor: int) -> None:
        nonlocal verification_calls
        original_verify(path, descriptor)
        verification_calls += 1
        # 第四次核对发生在复制完成、目标二次 stat 之前。
        if verification_calls != 4:
            return
        if race == "symlink":
            target.symlink_to(source)
        elif race == "regular":
            target.write_text("unexpected", encoding="utf-8")
        else:
            temporary = next(target.parent.glob(f".{target.name}.tmp-*"))
            temporary.unlink()

    monkeypatch.setattr(
        open_execution,
        "_verify_private_directory_identity",
        create_race,
    )
    expected = FileNotFoundError if race == "missing_temp" else RuntimeError
    with pytest.raises(expected):
        open_execution._copy_private_file_atomically(source, target)

    assert not list(target.parent.glob(f".{target.name}.tmp-*"))


def test_get_execution_job_preserves_non_mapping_metadata_branch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """作业状态中的非映射 metadata 不应被错误当作 scope 字段过滤。"""

    status = {
        "job_id": VALID_JOB_ID,
        "status": "running",
        "metadata": "opaque metadata",
        "scope_id": "scope-1",
        "scope_workspace": str(tmp_path),
    }
    monkeypatch.setattr(open_execution, "_read_status", lambda _job_id: (tmp_path, status))
    monkeypatch.setattr(open_execution, "_execution_status_axes", lambda *_args: {"details": {}})
    monkeypatch.setattr(open_execution, "_execution_failure_diagnostic", lambda *_args: {})
    monkeypatch.setattr(
        open_execution,
        "_operation_identity_evidence",
        lambda *_args: {"job_id": VALID_JOB_ID},
    )
    monkeypatch.setattr(open_execution, "_tail", lambda *_args: "")
    monkeypatch.setattr(open_execution, "_resource_availability", lambda: {})

    response = open_execution.get_execution_job(JobStatusRequest(job_id=VALID_JOB_ID))

    assert response["status"] == "success"
    assert response["job"] == {"job_id": VALID_JOB_ID, "status": "running"}


def test_scientific_artifact_byte_validation_covers_text_integrity_paths(
    monkeypatch, tmp_path: Path
) -> None:
    path = tmp_path / "artifact.txt"
    path.write_text("abcdef", encoding="utf-8")
    digest = open_execution._sha256(path)
    canonical = SimpleNamespace(
        media_type="text/plain; charset=utf-8",
        sha256=digest,
        artifact_id="art_" + "a" * 32,
        semantic_type="Result",
        path="artifact.txt",
    )
    request = ScientificArtifactTextReadRequest(
        artifact_id=canonical.artifact_id,
        expected_sha256=digest,
        offset=1,
        limit=3,
    )
    monkeypatch.setattr(open_execution, "read_file_snapshot", lambda _path: (b"abcdef", digest))
    assert open_execution._read_scientific_artifact_bytes(canonical, path, request) == ("abcdef", digest)
    assert open_execution._scientific_artifact_page(request, canonical, "abcdef", digest)["content"] == "bcd"
    assert open_execution._read_scientific_artifact_bytes(
        SimpleNamespace(**{**canonical.__dict__, "media_type": "application/octet-stream"}), path, request
    )["reason_code"] == "artifact_not_text"
    monkeypatch.setattr(open_execution, "read_file_snapshot", lambda _path: (b"x", "bad"))
    assert open_execution._read_scientific_artifact_bytes(canonical, path, request)["reason_code"] == "artifact_hash_mismatch"
    monkeypatch.setattr(open_execution, "read_file_snapshot", lambda _path: (b"\xff", "bad"))
    assert open_execution._read_scientific_artifact_bytes(canonical, path, request)["reason_code"] == "artifact_not_utf8"
    monkeypatch.setattr(open_execution, "read_file_snapshot", lambda _path: (b"x", digest))
    bad_expected = ScientificArtifactTextReadRequest(
        artifact_id=canonical.artifact_id, expected_sha256="0" * 64
    )
    assert open_execution._read_scientific_artifact_bytes(canonical, path, bad_expected)["reason_code"] == "expected_sha256_mismatch"
    monkeypatch.setattr(open_execution, "MAX_SCIENTIFIC_ARTIFACT_TEXT_BYTES", 1)
    monkeypatch.setattr(open_execution, "read_file_snapshot", lambda _path: (b"xx", digest))
    assert open_execution._read_scientific_artifact_bytes(canonical, path, request)["reason_code"] == "artifact_too_large"
    assert open_execution._read_scientific_artifact_bytes(canonical, tmp_path, request)["reason_code"] == "artifact_missing"


def test_operation_reference_receipt_validation_covers_all_identity_states(
    monkeypatch, tmp_path: Path
) -> None:
    scope = _scope(tmp_path)
    monkeypatch.setattr(open_execution, "current_scope", lambda: scope)
    reference = ExecutionOperationReference(operation_id="op-1", fingerprint="f" * 64)
    base = {
        "scope_id": scope.scope_id,
        "scope_workspace": str(scope.workspace),
        "schema_version": 1,
        "operation_id": "op-1",
        "fingerprint": "f" * 64,
        "status": "completed",
        "response": {},
    }
    assert open_execution._validate_operation_reference_receipt(reference, base) is None
    assert open_execution._validate_operation_reference_receipt(reference, {**base, "scope_id": "other"})["status"] == "unavailable"
    assert open_execution._validate_operation_reference_receipt(reference, {**base, "fingerprint": "g"})["reason_code"] == "operation_identity_mismatch"
    assert open_execution._validate_operation_reference_receipt(reference, {**base, "status": "running"})["reason_code"] == "operation_not_completed"
    assert open_execution._validate_operation_reference_receipt(reference, {**base, "response": None})["status"] == "unavailable"
    assert open_execution._validate_operation_reference_receipt(reference, {**base, "schema_version": 2})["status"] == "unavailable"
    monkeypatch.setattr(open_execution, "OperationIdempotencyStore", lambda **_kwargs: SimpleNamespace(read_operation_receipt=lambda _op: base))
    assert open_execution._read_operation_reference_receipt(reference) == base
    monkeypatch.setattr(open_execution, "OperationIdempotencyStore", lambda **_kwargs: (_ for _ in ()).throw(FileNotFoundError()))
    assert open_execution._read_operation_reference_receipt(reference) is None


def test_artifact_reference_integrity_helper_reports_missing_unreadable_and_hash(
    monkeypatch, tmp_path: Path
) -> None:
    path = tmp_path / "result.json"
    path.write_text("{}", encoding="utf-8")
    canonical = SimpleNamespace(path="result.json", sha256="f" * 64)
    monkeypatch.setattr(open_execution, "resolve_workspace_path", lambda *_args, **_kwargs: path)
    monkeypatch.setattr(open_execution, "read_file_snapshot", lambda _path: (b"{}", "f" * 64))
    assert open_execution._artifact_reference_integrity_error(canonical) is None
    monkeypatch.setattr(open_execution, "resolve_workspace_path", lambda *_args, **_kwargs: tmp_path / "missing")
    assert open_execution._artifact_reference_integrity_error(canonical) == "artifact_missing"
    monkeypatch.setattr(open_execution, "resolve_workspace_path", lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError()))
    assert open_execution._artifact_reference_integrity_error(canonical) == "artifact_missing"
    monkeypatch.setattr(open_execution, "resolve_workspace_path", lambda *_args, **_kwargs: path)
    monkeypatch.setattr(open_execution, "read_file_snapshot", lambda _path: (_ for _ in ()).throw(OSError()))
    assert open_execution._artifact_reference_integrity_error(canonical) == "artifact_unreadable"
    monkeypatch.setattr(open_execution, "read_file_snapshot", lambda _path: (b"{}", "a" * 64))
    assert open_execution._artifact_reference_integrity_error(canonical) == "artifact_hash_mismatch"


def test_process_identity_and_scope_inventory_helpers_cover_recovery_edges(
    monkeypatch, tmp_path: Path
) -> None:
    assert open_execution._persisted_process_identity({"x_pid": True}, "x") is None
    assert open_execution._persisted_process_identity({"x_pid": 1, "x_start_ticks": 2, "x_boot_id": "b"}, "x") is not None
    assert open_execution._persisted_process_identity({"x_pid": 0, "x_start_time_ticks": 2, "x_boot_id": "b"}, "x") is None
    scope = _scope(tmp_path)
    monkeypatch.setattr(open_execution, "current_scope", lambda: scope)
    job = tmp_path / ("job_" + "a" * 32)
    job.mkdir()
    monkeypatch.setattr(open_execution, "_read_status", lambda _name: (job, {"status": "running"}))
    monkeypatch.setattr(open_execution, "_scope_persisted_identity_is_complete", lambda *_args: True)
    item, active, valid = open_execution._inspect_scope_inventory_child(job)
    assert item["status"] == "running" and active and valid
    lock = tmp_path / ("." + "a" * 64 + ".lock")
    lock.write_text("", encoding="utf-8")
    assert open_execution._inspect_scope_inventory_child(lock) == (None, False, True)
    assert open_execution._inspect_scope_inventory_child(tmp_path / "random") == (None, False, False)
    (tmp_path / "jobs").mkdir()
    monkeypatch.setattr(open_execution, "workspace_root", lambda: tmp_path)
    monkeypatch.setattr(open_execution, "JOB_ROOT", Path("jobs"))
    assert open_execution._scope_job_inventory() == ([], False, True)


def test_reserved_input_and_gaussian_queue_helpers_keep_release_semantics(
    monkeypatch, tmp_path: Path
) -> None:
    context = SimpleNamespace(
        job_directory=tmp_path,
        receipt=None,
        receipt_path=tmp_path / "receipt.json",
        staged_records=[],
        metadata={"software_id": "other"},
        job_id=VALID_JOB_ID,
    )
    monkeypatch.setattr(open_execution, "_stage_reserved_input_files", lambda *_args: [])
    monkeypatch.setattr(open_execution, "_verify_expected_snapshots", lambda _items: [])
    assert open_execution._stage_reserved_job_inputs(
        context,
        staged_inputs=[],
        stdin_target=None,
        resource_limits={},
        expected_input_snapshots=[],
        operation_id=None,
    ) is None
    context.metadata = {"software_id": "gaussian"}
    context.staged_records = [{"target_path": "input.com"}]
    (tmp_path / "input.com").write_text("deck", encoding="utf-8")
    monkeypatch.setattr(open_execution, "_normalize_gaussian_staged_input", lambda *_args: {"mode": "serial"})
    monkeypatch.setattr(open_execution, "_sha256", lambda _path: "a" * 64)
    assert open_execution._stage_reserved_job_inputs(
        context,
        staged_inputs=[],
        stdin_target="input.com",
        resource_limits={"cpu_cores": 1, "memory_mb": 1},
        expected_input_snapshots=[],
        operation_id=None,
    ) is None
    class Ticket:
        def __init__(self):
            self.released = False
        def release(self):
            self.released = True
        def as_dict(self):
            return {"position": 1}
    ticket = Ticket()
    monkeypatch.setattr(open_execution, "enqueue_gaussian_job", lambda *_args, **_kwargs: ticket)
    monkeypatch.setattr(open_execution, "_start_reserved_job", lambda **_kwargs: {"status": "success"})
    monkeypatch.setattr(open_execution, "resource_budget_record", lambda: {})
    result = open_execution._reserve_and_start_gaussian_job(
        job_type="native_software", runtime="gaussian", command=["g16"], stdin_target=None,
        staged_inputs=[], resource_limits={"walltime_seconds": 1}, metadata={"software_id": "gaussian"},
        operation_id=None, operation_fingerprint=None, submission_key=None,
        caller_request_fingerprint=None, expected_input_snapshots=None, launch_file_snapshots=None,
    )
    assert result["status"] == "success" and not ticket.released
