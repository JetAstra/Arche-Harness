"""ExecutionJobRegistry 的短作业身份、碰撞和旧记录恢复合同测试。"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from minichem_mcp_tools.execution_job_registry import (
    ExecutionJobRegistry,
    ExecutionJobRegistryConflict,
)


def _registry(
    workspace: Path,
    *,
    scope_id: str = "scope-1",
    candidates: list[str] | None = None,
) -> ExecutionJobRegistry:
    values = iter(candidates or ["job_ab12cd34"])
    return ExecutionJobRegistry(
        workspace=workspace,
        scope_id=scope_id,
        job_id_factory=lambda: next(values),
    )


def test_reserve_allocates_short_lowercase_id_and_replays_same_operation(
    tmp_path: Path,
) -> None:
    """同一 operation/fingerprint 只占用一个短 ID，registry 记录内部路由身份。"""

    registry = _registry(tmp_path)
    first = registry.reserve(
        job_type="native_software",
        operation_id="operation-1",
        operation_fingerprint="a" * 64,
        submission_key="submission-1",
    )
    replay = registry.reserve(
        job_type="native_software",
        operation_id="operation-1",
        operation_fingerprint="a" * 64,
        submission_key="submission-1",
    )

    assert first.job_id == "job_ab12cd34"
    assert len(first.job_id) == len("job_") + 8
    assert first.reused is False
    assert replay.job_id == first.job_id
    assert replay.reused is True
    assert registry.get(first.job_id)["job_type"] == "native_software"


def test_collected_job_keeps_route_identity_for_get_and_operation_replay(
    tmp_path: Path,
) -> None:
    """collect 后仍可按可信路由查询，并重放同一 operation 的原 job_id。"""

    registry = _registry(tmp_path)
    first = registry.reserve(
        job_type="transition_state_path",
        operation_id="operation-collected",
        operation_fingerprint="a" * 64,
        submission_key="submission-collected",
    )
    collected = registry.mark(first.job_id, phase="collected")

    assert collected["job_type"] == "transition_state_path"
    assert collected["scope_id"] == "scope-1"
    assert collected["operation_id"] == "operation-collected"
    reopened = _registry(tmp_path, candidates=["job_01020304"])
    assert reopened.get(first.job_id)["phase"] == "collected"
    replay = reopened.reserve(
        job_type="transition_state_path",
        operation_id="operation-collected",
        operation_fingerprint="a" * 64,
        submission_key="submission-collected",
    )
    assert replay.job_id == first.job_id
    assert replay.reused is True
    assert reopened.get(first.job_id)["phase"] == "collected"


def test_unrelated_corrupt_legacy_directory_does_not_block_operation_recovery(
    tmp_path: Path,
) -> None:
    """恢复目标 operation 时跳过无关损坏目录，但仍保留目标身份的 fail-closed。"""

    unrelated = tmp_path / "outputs" / "execution_jobs" / ("job_" + "b" * 32)
    unrelated.mkdir(parents=True)
    (unrelated / "status.json").write_text("{not-json", encoding="utf-8")
    (unrelated / "operation_receipt.json").write_text(
        json.dumps({"operation_id": "other-operation"}), encoding="utf-8"
    )

    old_job_id = "job_" + "c" * 32
    directory = tmp_path / "outputs" / "execution_jobs" / old_job_id
    directory.mkdir(parents=True)
    identity = {
        "schema_version": 1,
        "job_id": old_job_id,
        "job_type": "native_software",
        "operation_id": "recoverable-operation",
        "operation_fingerprint": "d" * 64,
        "submission_key": "recoverable-submission",
        "scope_id": "scope-1",
        "scope_workspace": str(tmp_path),
    }
    (directory / "status.json").write_text(
        json.dumps({**identity, "status": "running"}), encoding="utf-8"
    )
    (directory / "operation_receipt.json").write_text(
        json.dumps({**identity, "fingerprint": identity["operation_fingerprint"]}),
        encoding="utf-8",
    )

    recovered = _registry(tmp_path).reserve(
        job_type="native_software",
        operation_id="recoverable-operation",
        operation_fingerprint="d" * 64,
        submission_key="recoverable-submission",
    )

    assert recovered.job_id == old_job_id
    assert recovered.reused is True


def test_reserve_rejects_same_operation_with_different_fingerprint(
    tmp_path: Path,
) -> None:
    """同一 operation 绑定不同请求时 fail-closed，不分配第二个作业。"""

    registry = _registry(tmp_path, candidates=["job_ab12cd34", "job_01020304"])
    registry.reserve(
        job_type="analysis",
        operation_id="operation-1",
        operation_fingerprint="a" * 64,
        submission_key="submission-1",
    )

    with pytest.raises(ExecutionJobRegistryConflict):
        registry.reserve(
            job_type="analysis",
            operation_id="operation-1",
            operation_fingerprint="b" * 64,
            submission_key="submission-1",
        )


def test_mark_rejects_mutation_of_immutable_execution_identity(tmp_path: Path) -> None:
    """状态更新不能改写已预留作业的 operation 身份。"""

    registry = _registry(tmp_path)
    reservation = registry.reserve(
        job_type="native_software",
        operation_id="operation-immutable",
        operation_fingerprint="a" * 64,
        submission_key="submission-immutable",
    )

    with pytest.raises(ExecutionJobRegistryConflict):
        registry.mark(
            reservation.job_id,
            phase="running",
            operation_id="operation-other",
        )

    assert registry.get(reservation.job_id)["operation_id"] == "operation-immutable"


def test_legacy_recovery_rejects_conflicting_caller_fingerprints(tmp_path: Path) -> None:
    """旧目录多来源 caller fingerprint 冲突时不能猜测恢复身份。"""

    old_job_id = "job_" + "e" * 32
    directory = tmp_path / "outputs" / "execution_jobs" / old_job_id
    directory.mkdir(parents=True)
    common = {
        "schema_version": 1,
        "job_id": old_job_id,
        "job_type": "native_software",
        "operation_id": "legacy-caller-conflict",
        "operation_fingerprint": "f" * 64,
        "submission_key": "legacy-caller-submission",
        "scope_id": "scope-1",
        "scope_workspace": str(tmp_path),
    }
    (directory / "status.json").write_text(
        json.dumps({
            **common,
            "status": "running",
            "caller_request_fingerprint": "a" * 64,
        }),
        encoding="utf-8",
    )
    (directory / "operation_receipt.json").write_text(
        json.dumps({
            **common,
            "fingerprint": common["operation_fingerprint"],
            "caller_request_fingerprint": "b" * 64,
        }),
        encoding="utf-8",
    )

    with pytest.raises(ExecutionJobRegistryConflict):
        _registry(tmp_path).reserve(
            job_type="native_software",
            operation_id="legacy-caller-conflict",
            operation_fingerprint="f" * 64,
            submission_key="legacy-caller-submission",
        )


def test_concurrent_registry_reserve_starts_one_identity(
    tmp_path: Path,
) -> None:
    """并发 registry 实例只能让同一 operation 占用一个短 ID。"""

    def reserve_once() -> tuple[str, bool]:
        registry = _registry(tmp_path, candidates=["job_ab12cd34"])
        result = registry.reserve(
            job_type="native_software",
            operation_id="concurrent-operation",
            operation_fingerprint="a" * 64,
            submission_key="concurrent-submission",
        )
        return result.job_id, result.reused

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _index: reserve_once(), range(2)))

    assert {job_id for job_id, _reused in results} == {"job_ab12cd34"}
    assert sorted(reused for _job_id, reused in results) == [False, True]


def test_reserve_retries_candidate_that_collides_with_directory_or_collected_record(
    tmp_path: Path,
) -> None:
    """短 ID 与旧目录或 collected registry 记录冲突时重试，不覆盖已有身份。"""

    old_directory = tmp_path / "outputs" / "execution_jobs" / "job_ab12cd34"
    old_directory.mkdir(parents=True)
    registry = _registry(
        tmp_path,
        candidates=["job_ab12cd34", "job_01020304", "job_05060708"],
    )
    first = registry.reserve(
        job_type="analysis",
        operation_id="operation-2",
        operation_fingerprint="a" * 64,
        submission_key="submission-2",
    )
    registry.mark(first.job_id, phase="collected")
    second = registry.reserve(
        job_type="analysis",
        operation_id="operation-3",
        operation_fingerprint="b" * 64,
        submission_key="submission-3",
    )

    assert first.job_id == "job_01020304"
    assert second.job_id == "job_05060708"
    assert old_directory.is_dir()


def test_workspace_global_allocation_prevents_cross_scope_short_id_collision(
    tmp_path: Path,
) -> None:
    """同一物理 workspace 的不同 scope 也不能并发占用同一个短 ID。"""

    first = _registry(tmp_path, scope_id="scope-a", candidates=["job_ab12cd34"])
    second = _registry(
        tmp_path,
        scope_id="scope-b",
        candidates=["job_ab12cd34", "job_01020304"],
    )
    first_reservation = first.reserve(
        job_type="native_software",
        operation_id="scope-a-operation",
        operation_fingerprint="a" * 64,
        submission_key="scope-a-submission",
    )
    second_reservation = second.reserve(
        job_type="analysis",
        operation_id="scope-b-operation",
        operation_fingerprint="b" * 64,
        submission_key="scope-b-submission",
    )

    assert first_reservation.job_id == "job_ab12cd34"
    assert second_reservation.job_id == "job_01020304"


def test_reserve_imports_legacy_32_hex_job_for_same_operation(
    tmp_path: Path,
) -> None:
    """registry 缺失时可从旧 32 hex execution receipt 恢复原 job_id。"""

    old_job_id = "job_" + "a" * 32
    directory = tmp_path / "outputs" / "execution_jobs" / old_job_id
    directory.mkdir(parents=True)
    identity = {
        "schema_version": 1,
        "job_id": old_job_id,
        "job_type": "programmable_analysis",
        "operation_id": "legacy-operation",
        "operation_fingerprint": "c" * 64,
        "submission_key": "legacy-submission",
        "scope_id": "scope-1",
        "scope_workspace": str(tmp_path),
    }
    (directory / "status.json").write_text(
        json.dumps({**identity, "status": "running"}), encoding="utf-8"
    )
    (directory / "request.json").write_text(json.dumps(identity), encoding="utf-8")
    (directory / "operation_receipt.json").write_text(
        json.dumps({**identity, "fingerprint": identity["operation_fingerprint"]}),
        encoding="utf-8",
    )

    registry = _registry(tmp_path, candidates=["job_01020304"])
    recovered = registry.reserve(
        job_type="programmable_analysis",
        operation_id="legacy-operation",
        operation_fingerprint="c" * 64,
        submission_key="legacy-submission",
    )

    assert recovered.job_id == old_job_id
    assert recovered.reused is True
    assert registry.get(old_job_id)["phase"] == "running"
