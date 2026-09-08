"""Gaussian 服务资源策略、输入规范化和公开合同的最小回归测试。"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from minichem_mcp_tools import job_supervisor, open_execution
from minichem_mcp_tools.execution_models import NativeJobRequest
from minichem_toolbox.catalog import action_specs, mcp_action_description
from minichem_toolbox.discovery import inspect_action
from minichem_toolbox.models import ActionRequest, ResourceLimits
from minichem_toolbox.resource_budget import (
    GAUSSIAN_CPU_CORES_ENV,
    GAUSSIAN_MEMORY_MB_ENV,
    ResourceBudget,
    apply_gaussian_resource_policy,
    gaussian_resource_policy,
    gaussian_resource_policy_record,
)
from minichem_toolbox.service import _bind_gaussian_action_resources


def _budget() -> ResourceBudget:
    """返回测试用的服务总预算。"""

    return ResourceBudget(cpu_cores=60, memory_mb=320_000, gpu_count=1)


def test_gaussian_policy_defaults_to_total_budget_and_accepts_explicit_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """未配置时使用总预算，配置时接受不超过预算的正整数。"""

    monkeypatch.delenv(GAUSSIAN_CPU_CORES_ENV, raising=False)
    monkeypatch.delenv(GAUSSIAN_MEMORY_MB_ENV, raising=False)
    assert gaussian_resource_policy(budget=_budget()).as_dict() == {
        "cpu_cores": 60,
        "memory_mb": 320_000,
    }

    monkeypatch.setenv(GAUSSIAN_CPU_CORES_ENV, "12")
    monkeypatch.setenv(GAUSSIAN_MEMORY_MB_ENV, "64000")
    assert gaussian_resource_policy(budget=_budget()).as_dict() == {
        "cpu_cores": 12,
        "memory_mb": 64_000,
    }


@pytest.mark.parametrize(
    ("name", "value"),
    [
        (GAUSSIAN_CPU_CORES_ENV, "0"),
        (GAUSSIAN_CPU_CORES_ENV, "-1"),
        (GAUSSIAN_CPU_CORES_ENV, "1.5"),
        (GAUSSIAN_CPU_CORES_ENV, " 12"),
        (GAUSSIAN_MEMORY_MB_ENV, "127"),
        (GAUSSIAN_MEMORY_MB_ENV, "not-an-integer"),
    ],
)
def test_gaussian_policy_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
) -> None:
    """非正整数、带空白值和低于内存下限的配置必须失败。"""

    monkeypatch.delenv(GAUSSIAN_CPU_CORES_ENV, raising=False)
    monkeypatch.delenv(GAUSSIAN_MEMORY_MB_ENV, raising=False)
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        gaussian_resource_policy(budget=_budget())


@pytest.mark.parametrize(
    ("name", "value"),
    [
        (GAUSSIAN_CPU_CORES_ENV, "61"),
        (GAUSSIAN_MEMORY_MB_ENV, "320001"),
    ],
)
def test_gaussian_policy_rejects_values_above_service_budget(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
) -> None:
    """超过服务总预算的显式值不能启动或冻结目录。"""

    monkeypatch.delenv(GAUSSIAN_CPU_CORES_ENV, raising=False)
    monkeypatch.delenv(GAUSSIAN_MEMORY_MB_ENV, raising=False)
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        gaussian_resource_policy(budget=_budget())


def test_gaussian_override_keeps_gpu_and_non_gaussian_resources_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gaussian 只覆盖 CPU/内存，GPU 等字段和非 Gaussian 请求保持原样。"""

    monkeypatch.setenv(GAUSSIAN_CPU_CORES_ENV, "12")
    monkeypatch.setenv(GAUSSIAN_MEMORY_MB_ENV, "64000")
    assert apply_gaussian_resource_policy(
        {"cpu_cores": 1, "memory_mb": 128, "gpu_count": 1, "walltime_seconds": 8},
        budget=_budget(),
    ) == {
        "cpu_cores": 12,
        "memory_mb": 64_000,
        "gpu_count": 1,
        "walltime_seconds": 8,
    }

    request = ActionRequest(
        backend_id="gaussian",
        resource_limits=ResourceLimits(cpu_cores=1, memory_mb=128, gpu_count=1),
    )
    bound = _bind_gaussian_action_resources(request)
    assert bound.resource_limits.cpu_cores == 12
    assert bound.resource_limits.memory_mb == 64_000
    assert bound.resource_limits.gpu_count == 1

    other = ActionRequest(
        backend_id="xtb",
        resource_limits=ResourceLimits(cpu_cores=2, memory_mb=512, gpu_count=0),
    )
    assert _bind_gaussian_action_resources(other).resource_limits == other.resource_limits


def test_gaussian_staged_deck_normalization_covers_every_link1_segment(
    tmp_path: Path,
) -> None:
    """规范化只修改 staged 副本，并为每个 Link1 段保留一组统一指令。"""

    job_directory = tmp_path / "job"
    input_directory = job_directory / "inputs"
    input_directory.mkdir(parents=True)
    source = input_directory / "source.gjf"
    original = (
        "%NProcShared=1\n"
        "%Mem=128MB\n"
        "%mem=999MB\n"
        "#p hf/3-21g opt\n\n"
        "first\n\n"
        "0 1\n"
        "H 0 0 0\n\n"
        "--Link1--\n"
        "%NProcShared=88\n"
        "#p hf/3-21g freq\n\n"
        "second\n\n"
        "0 1\n"
        "H 0 0 0\n\n"
    )
    source.write_text(original, encoding="utf-8")

    record = open_execution._normalize_gaussian_staged_input(
        job_directory,
        "inputs/source.gjf",
        {"cpu_cores": 12, "memory_mb": 64_000},
    )
    normalized = source.read_text(encoding="utf-8")
    assert record == {
        "mode": "service_managed_full_capacity",
        "concurrency": "serial",
        "segment_count": 2,
        "cpu_cores": 12,
        "memory_mb": 63_360,
        "reserved_memory_mb": 64_000,
        "memory_headroom_mb": 640,
        "input_target": "inputs/source.gjf",
    }
    assert normalized.count("%NProcShared=12") == 2
    assert normalized.count("%Mem=63360MB") == 2
    assert "%NProcShared=1" not in normalized.splitlines()
    assert "%mem=999MB" not in normalized.splitlines()
    assert normalized.count("%NProcShared=") == 2
    assert normalized.count("%Mem=") == 2
    assert source.read_text(encoding="utf-8") != original


def test_native_gaussian_effective_resources_ignore_caller_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """native Gaussian 的请求值和旧专用覆盖均不改变统一固定资源。"""

    monkeypatch.setenv(GAUSSIAN_CPU_CORES_ENV, "12")
    monkeypatch.setenv(GAUSSIAN_MEMORY_MB_ENV, "64000")
    request = NativeJobRequest(
        software_id="gaussian",
        executable="g16",
        resource_limits=ResourceLimits(cpu_cores=1, memory_mb=128, gpu_count=0),
    )
    effective = open_execution._effective_native_resource_limits(request)
    assert effective == {"cpu_cores": 60, "memory_mb": 320_000, "gpu_count": 1}

    other = NativeJobRequest(
        software_id="orca",
        executable="orca",
        resource_limits=ResourceLimits(cpu_cores=2, memory_mb=512, gpu_count=0),
    )
    assert open_execution._effective_native_resource_limits(other) == {
        "cpu_cores": 60,
        "memory_mb": 320_000,
        "gpu_count": 1,
    }


def _write_supervisor_spec(
    root: Path,
    *,
    script: Path,
    events: Path,
    metadata: dict[str, Any],
    gate_path: Path,
    job_name: str,
) -> Path:
    """写入可由真实 supervisor 子进程执行的最小作业 spec。"""

    job_directory = root / job_name
    job_directory.mkdir()
    status_path = job_directory / "status.json"
    stdout_path = job_directory / "stdout.log"
    stderr_path = job_directory / "stderr.log"
    status_path.write_text(json.dumps({"status": "queued"}), encoding="utf-8")
    specification = {
        "schema_version": 1,
        "job_id": f"job_{job_name:0<32}",
        "job_type": "native_software",
        "runtime": "core",
        "command": [sys.executable, str(script), str(events)],
        "job_directory": str(job_directory),
        "relative_job_directory": job_name,
        "status_path": str(status_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "relative_stdout_path": f"{job_name}/stdout.log",
        "relative_stderr_path": f"{job_name}/stderr.log",
        "stdin_path": None,
        "resource_limits": {
            "cpu_cores": 1,
            "memory_mb": 256,
            "gpu_count": 0,
            "walltime_seconds": 10,
        },
        "resource_allocation": {},
        "evaluation_resource_budget": {},
        "metadata": metadata,
        "gaussian_serial_gate_path": str(gate_path),
        "submitted_at": "2026-01-01T00:00:00+00:00",
    }
    spec_path = job_directory / "supervisor_spec.json"
    spec_path.write_text(json.dumps(specification), encoding="utf-8")
    return spec_path


def _run_supervisors(specifications: list[Path]) -> None:
    """并发启动真实 supervisor，并等待全部作业结束。"""

    processes = [
        subprocess.Popen(
            [sys.executable, str(job_supervisor.__file__), str(specification)],
            cwd=specification.parent,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for specification in specifications
    ]
    try:
        for process in processes:
            assert process.wait(timeout=15) == 0
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()


def test_supervisor_holds_gaussian_gate_until_child_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """两个 Gaussian supervisor 的实际子进程不能重叠，非 Gaussian 不受该锁影响。"""

    monkeypatch.setenv(job_supervisor.COMPUTE_HARD_TIMEOUT_ENV, "60")
    script = tmp_path / "record_events.py"
    script.write_text(
        "import sys, time\n"
        "from pathlib import Path\n"
        "events = Path(sys.argv[1])\n"
        "with events.open('a', encoding='utf-8') as handle:\n"
        "    handle.write('start\\n')\n"
        "    handle.flush()\n"
        "print('started', flush=True)\n"
        "time.sleep(0.35)\n"
        "with events.open('a', encoding='utf-8') as handle:\n"
        "    handle.write('finish\\n')\n"
        "    handle.flush()\n"
        "print('finished', flush=True)\n",
        encoding="utf-8",
    )
    gaussian_events = tmp_path / "gaussian.events"
    gate_path = tmp_path / "gaussian.lock"
    gaussian_specs = [
        _write_supervisor_spec(
            tmp_path,
            script=script,
            events=gaussian_events,
            metadata={"software_id": "gaussian"},
            gate_path=gate_path,
            job_name=f"gaussian_{index}",
        )
        for index in (1, 2)
    ]
    _run_supervisors(gaussian_specs)
    assert gaussian_events.read_text(encoding="utf-8").splitlines() == [
        "start",
        "finish",
        "start",
        "finish",
    ]

    non_gaussian_events = tmp_path / "non_gaussian.events"
    barrier_script = tmp_path / "record_barrier_events.py"
    barrier_script.write_text(
        "import sys, time\n"
        "from pathlib import Path\n"
        "events = Path(sys.argv[1])\n"
        "with events.open('a', encoding='utf-8') as handle:\n"
        "    handle.write('start\\n')\n"
        "    handle.flush()\n"
        "print('started', flush=True)\n"
        "deadline = time.monotonic() + 5\n"
        "while len(events.read_text(encoding='utf-8').splitlines()) < 2 and time.monotonic() < deadline:\n"
        "    time.sleep(0.01)\n"
        "with events.open('a', encoding='utf-8') as handle:\n"
        "    handle.write('finish\\n')\n"
        "    handle.flush()\n"
        "print('finished', flush=True)\n",
        encoding="utf-8",
    )
    non_gaussian_specs = [
        _write_supervisor_spec(
            tmp_path,
            script=barrier_script,
            events=non_gaussian_events,
            metadata={"software_id": "orca"},
            gate_path=gate_path,
            job_name=f"orca_{index}",
        )
        for index in (1, 2)
    ]
    _run_supervisors(non_gaussian_specs)
    assert non_gaussian_events.read_text(encoding="utf-8").splitlines() == [
        "start",
        "start",
        "finish",
        "finish",
    ]


def test_public_gaussian_contract_is_neutral_and_hides_resource_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """公开合同只说明服务管理策略，不返回 CPU/内存、环境名或运行路径。"""

    monkeypatch.setenv(GAUSSIAN_CPU_CORES_ENV, "12")
    monkeypatch.setenv(GAUSSIAN_MEMORY_MB_ENV, "64000")
    snapshot = {"schema_version": 7, "catalog_hash": "test-catalog"}
    for action_id, backend_id in (
        ("calculate_hessian", "gaussian"),
        ("sdf_to_gjf", "format_conversion"),
        ("xyz_to_gjf", "format_conversion"),
    ):
        inspected = inspect_action(
            action_id,
            backend_id=backend_id,
            detail_level="contract",
            snapshot=snapshot,
        )
        serialized = json.dumps(inspected, ensure_ascii=False)
        assert "resource_limits" not in inspected["action_request_fields"]
        assert inspected["resource_policy"] == gaussian_resource_policy_record()
        assert GAUSSIAN_CPU_CORES_ENV not in serialized
        assert GAUSSIAN_MEMORY_MB_ENV not in serialized
        assert "/mnt/" not in serialized
        assert "nprocshared" not in serialized.casefold()
        assert "inputs.mem" not in serialized.casefold()

    description = mcp_action_description(action_specs()["sdf_to_gjf"])
    assert "service_managed_full_capacity" in description
    assert "caller cannot choose its CPU or memory" in description
    assert GAUSSIAN_CPU_CORES_ENV not in description
    assert GAUSSIAN_MEMORY_MB_ENV not in description
