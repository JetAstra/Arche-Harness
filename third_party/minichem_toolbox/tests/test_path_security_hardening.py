import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import minichem_toolbox.catalog as catalog_state
from minichem_mcp_tools import open_execution
from minichem_toolbox import resource_budget
from minichem_toolbox.backends import common


@pytest.mark.parametrize(
    ("writer", "payload"),
    [
        (open_execution._write_private_text_atomically, "text"),
        (resource_budget._write_private_text_atomically, "text"),
        (common._atomic_write_bytes, b"bytes"),
    ],
)
def test_atomic_writers_reject_ancestor_symlink(
    tmp_path: Path, writer, payload
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "link"
    link.symlink_to(outside, target_is_directory=True)

    with pytest.raises((OSError, RuntimeError)):
        writer(link / "state.json", payload)

    assert not (outside / "state.json").exists()


def test_atomic_writers_create_private_regular_files(tmp_path: Path) -> None:
    open_path = tmp_path / "open.json"
    resource_path = tmp_path / "resource.json"
    backend_path = tmp_path / "backend.json"
    open_execution._write_private_text_atomically(open_path, "open")
    resource_budget._write_private_text_atomically(resource_path, "resource")
    common._atomic_write_bytes(backend_path, b"backend")

    assert open_path.read_text(encoding="utf-8") == "open"
    assert resource_path.read_text(encoding="utf-8") == "resource"
    assert backend_path.read_bytes() == b"backend"
    for path in (open_path, resource_path, backend_path):
        assert os.stat(path).st_mode & 0o777 == 0o600
        assert not list(path.parent.glob(f".{path.name}.*"))

    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(OSError):
        common._atomic_write_bytes(directory, b"no")


@pytest.mark.parametrize(
    ("module", "writer", "payload"),
    [
        (open_execution, open_execution._write_private_text_atomically, "text"),
        (resource_budget, resource_budget._write_private_text_atomically, "text"),
        (common, common._atomic_write_bytes, b"bytes"),
    ],
)
def test_atomic_writer_keeps_write_inside_open_directory_on_ancestor_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, module, writer, payload
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    original_verify = module._verify_private_directory_identity
    calls = 0

    def swap_after_initial_identity(path: Path, descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            moved = tmp_path / "moved"
            parent.rename(moved)
            parent.symlink_to(outside, target_is_directory=True)
        original_verify(path, descriptor)

    monkeypatch.setattr(
        module,
        "_verify_private_directory_identity",
        swap_after_initial_identity,
    )
    with pytest.raises((OSError, RuntimeError)):
        writer(parent / "state.json", payload)

    assert not (outside / "state.json").exists()
    assert not (tmp_path / "moved" / "state.json").exists()


def test_private_lock_rejects_symlink_leaf(tmp_path: Path) -> None:
    parent = tmp_path / "private"
    parent.mkdir()
    outside = tmp_path / "outside.lock"
    outside.write_text("outside", encoding="utf-8")
    link = parent / "quota.lock"
    link.symlink_to(outside)

    with pytest.raises((OSError, RuntimeError)):
        resource_budget._open_private_lock(link)


def test_private_copy_uses_private_target_and_rejects_existing_target(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.txt"
    source.write_text("payload", encoding="utf-8")
    target = tmp_path / "nested" / "copy.txt"
    open_execution._copy_private_file_atomically(source, target)

    assert target.read_text(encoding="utf-8") == "payload"
    assert os.stat(target).st_mode & 0o777 == 0o600
    with pytest.raises(ValueError):
        open_execution._copy_private_file_atomically(source, target)


def test_private_marker_creation_and_existing_marker_validation(tmp_path: Path) -> None:
    marker = tmp_path / "private" / "cancel_requested"
    open_execution._ensure_private_marker(marker)
    open_execution._ensure_private_marker(marker)
    assert marker.is_file()
    assert os.stat(marker).st_mode & 0o777 == 0o600


def test_workspace_and_scheduler_symlinks_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_workspace = tmp_path / "workspace"
    real_workspace.mkdir()
    workspace_link = tmp_path / "workspace-link"
    workspace_link.symlink_to(real_workspace, target_is_directory=True)
    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(workspace_link))
    monkeypatch.delenv("RESEARCHCHEMBENCH_WORKSPACE", raising=False)
    monkeypatch.setattr(resource_budget, "current_scope", lambda: None)

    with pytest.raises(resource_budget.ResourceStateUnavailable):
        resource_budget._workspace_root()

    real_scheduler = tmp_path / "scheduler"
    real_scheduler.mkdir()
    scheduler_link = tmp_path / "scheduler-link"
    scheduler_link.symlink_to(real_scheduler, target_is_directory=True)
    monkeypatch.setenv(resource_budget.SHARED_SCHEDULER_ROOT_ENV, str(scheduler_link))
    with pytest.raises(resource_budget.ResourceStateUnavailable):
        resource_budget._configured_scheduler_root()


def test_scope_workspace_registry_reads_and_validates_stable_identity(
    tmp_path: Path,
) -> None:
    scheduler = tmp_path / "scheduler"
    workspace = tmp_path / "workspace"
    scheduler.mkdir()
    workspace.mkdir()
    scheduler.chmod(0o700)
    workspace.chmod(0o700)
    scope = SimpleNamespace(scope_id="scope-1", workspace=workspace)

    resource_budget._register_scope_workspace(scheduler, scope)
    indexed = list((scheduler / "scope_workspaces").glob("workspace_*.json"))
    assert len(indexed) == 1
    assert resource_budget._read_registered_scope_workspace(indexed[0]) == workspace
    assert resource_budget._registered_scope_workspaces(scheduler) == [workspace]
    resource_budget._validate_registered_scope_workspace(
        scheduler,
        "scope-1",
        str(workspace),
    )


def test_queue_release_rechecks_record_identity_and_syncs_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    monkeypatch.delenv("RESEARCHCHEMBENCH_WORKSPACE", raising=False)
    ticket = resource_budget.enqueue_gaussian_job("owner")
    record = json.loads(ticket_path(ticket).read_text(encoding="utf-8"))
    record["owner_id"] = "other-owner"
    ticket_path(ticket).write_text(json.dumps(record), encoding="utf-8")
    ticket.release()
    assert ticket_path(ticket).exists()

    record["owner_id"] = ticket.owner_id
    ticket_path(ticket).write_text(json.dumps(record), encoding="utf-8")
    sync_calls: list[int] = []
    original_fsync = os.fsync
    monkeypatch.setattr(
        resource_budget.os,
        "fsync",
        lambda descriptor: (sync_calls.append(descriptor), original_fsync(descriptor))[1],
    )
    ticket.release()
    assert sync_calls
    assert not ticket_path(ticket).exists()


def ticket_path(ticket: resource_budget.GaussianQueueTicket) -> Path:
    directory = resource_budget._gaussian_queue_directory()
    return directory / f"ticket_{ticket.ticket_id:020d}.json"


def test_reservation_release_rechecks_locked_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    monkeypatch.delenv("RESEARCHCHEMBENCH_WORKSPACE", raising=False)
    monkeypatch.setenv(resource_budget.AVAILABLE_CPU_CORES_ENV, "2")
    monkeypatch.setenv(resource_budget.AVAILABLE_MEMORY_MB_ENV, "128")
    monkeypatch.setenv(resource_budget.AVAILABLE_GPU_COUNT_ENV, "0")
    reservation = resource_budget.reserve_resources(
        {"cpu_cores": 1, "memory_mb": 128, "gpu_count": 0},
        kind="test",
        label="original",
    )
    record = json.loads(reservation.path.read_text(encoding="utf-8"))
    record["label"] = "replacement"
    reservation.path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(resource_budget.ResourceStateUnavailable):
        reservation.release()
    assert reservation.path.exists()


def test_reservation_validation_covers_scope_and_worker_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    monkeypatch.delenv("RESEARCHCHEMBENCH_WORKSPACE", raising=False)
    monkeypatch.setenv(resource_budget.AVAILABLE_CPU_CORES_ENV, "2")
    monkeypatch.setenv(resource_budget.AVAILABLE_MEMORY_MB_ENV, "128")
    monkeypatch.setenv(resource_budget.AVAILABLE_GPU_COUNT_ENV, "0")
    reservation = resource_budget.reserve_resources(
        {"cpu_cores": 1, "memory_mb": 128, "gpu_count": 0},
        kind="test",
        label="validation",
    )
    record = json.loads(reservation.path.read_text(encoding="utf-8"))
    assert resource_budget._validate_persisted_reservation(
        reservation.path, record
    ) == record

    typed_record = dict(record)
    typed_record.update(
        {
            "worker_tracking": "gaussian_typed",
            "worker_pid": os.getpid(),
            "worker_start_time_ticks": 1,
            "worker_boot_id": "boot-id",
        }
    )
    assert resource_budget._validate_persisted_reservation(
        reservation.path, typed_record
    ) == typed_record

    scheduler = tmp_path / "scheduler"
    workspace = tmp_path / "scope-workspace"
    scheduler.mkdir()
    workspace.mkdir()
    scheduler.chmod(0o700)
    workspace.chmod(0o700)
    resource_budget._register_scope_workspace(
        scheduler,
        SimpleNamespace(scope_id="scope-1", workspace=workspace),
    )
    scoped_record = dict(record)
    scoped_record.update(
        {
            "scope_id": "scope-1",
            "workspace": str(workspace),
            "job_id": "job_" + "1" * 32,
        }
    )
    monkeypatch.setattr(resource_budget, "current_scope", lambda: None)
    monkeypatch.setenv(resource_budget.SHARED_SCHEDULER_ROOT_ENV, str(scheduler))
    assert resource_budget._validate_persisted_reservation(
        reservation.path, scoped_record
    ) == scoped_record
    sync_calls: list[int] = []
    original_fsync = os.fsync
    monkeypatch.setattr(
        resource_budget.os,
        "fsync",
        lambda descriptor: (sync_calls.append(descriptor), original_fsync(descriptor))[1],
    )
    reservation.release()
    assert sync_calls


def test_reservation_binding_rechecks_locked_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    monkeypatch.delenv("RESEARCHCHEMBENCH_WORKSPACE", raising=False)
    monkeypatch.setenv(resource_budget.AVAILABLE_CPU_CORES_ENV, "2")
    monkeypatch.setenv(resource_budget.AVAILABLE_MEMORY_MB_ENV, "128")
    monkeypatch.setenv(resource_budget.AVAILABLE_GPU_COUNT_ENV, "0")
    reservation = resource_budget.reserve_resources(
        {"cpu_cores": 1, "memory_mb": 128, "gpu_count": 0},
        kind="test",
        label="binding",
        worker_tracking="gaussian_typed",
    )
    assert reservation.bind_worker(
        {"pid": os.getpid(), "start_time_ticks": 1, "boot_id": "boot-id"}
    )
    bound = json.loads(reservation.path.read_text(encoding="utf-8"))
    assert bound["worker_status"] == "running"
    reservation.release()


def test_frozen_catalog_identity_is_startup_snapshot_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = {"catalog_hash": "a" * 64, "schema_version": 7}
    monkeypatch.setattr(catalog_state, "_ACTIVE_CATALOG_SNAPSHOT", snapshot)
    monkeypatch.setattr(catalog_state, "_ACTIVE_CATALOG_WORKSPACE", tmp_path)
    monkeypatch.setattr(catalog_state, "_catalog_workspace_path", lambda: tmp_path)
    monkeypatch.setattr(catalog_state, "active_catalog_snapshot", lambda: dict(snapshot))

    assert open_execution._read_frozen_catalog_identity() == {
        "catalog_hash": "a" * 64,
        "catalog_version": "7",
    }

    monkeypatch.setattr(
        catalog_state,
        "active_catalog_snapshot",
        lambda: {"catalog_hash": "b" * 64, "schema_version": 7},
    )
    with pytest.raises(resource_budget.ResourceStateUnavailable):
        open_execution._read_frozen_catalog_identity()


def test_frozen_catalog_identity_missing_snapshot_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(catalog_state, "_ACTIVE_CATALOG_SNAPSHOT", None)
    monkeypatch.setattr(catalog_state, "_ACTIVE_CATALOG_WORKSPACE", None)
    with pytest.raises(resource_budget.ResourceStateUnavailable):
        open_execution._read_frozen_catalog_identity()
