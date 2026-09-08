"""覆盖 runtime 和公共 Backend 辅助函数的安全边界与失败分支。"""

from __future__ import annotations

import json
import os
import signal
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from minichem_toolbox import runtime
from minichem_toolbox.backends import common
from minichem_toolbox.process_identity import ProcessIdentity
from minichem_toolbox.worker_protocol import (
    WORKER_RESPONSE_PREFIX,
    format_worker_response,
)


def _identity(
    pid: int = 123,
    *,
    start: int | None = 11,
    boot: str | None = "boot",
    group: int | None = 123,
    session: int | None = 123,
) -> ProcessIdentity:
    """构造测试用的可核对进程身份。"""

    return ProcessIdentity(pid, start, boot, group, session)


def _backend_spec(**overrides):
    """构造只包含健康投影所需字段的 Backend 描述。"""

    values = {
        "id": "test_backend",
        "runtime": "core",
        "python_modules": (),
        "executables": (),
        "environment_variables": (),
        "conda_packages": (),
        "pip_packages": (),
        "required_data_resources": (),
        "license_class": "open",
        "install_notes": "",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("PATH", True),
        ("MINICHEM_SAFE_SETTING", True),
        ("MINICHEM_MODEL_CACHE", False),
        ("HF_HOME", False),
        ("ARCHE_API_KEY", False),
        ("UNRELATED_SETTING", False),
    ],
)
def test_external_environment_allowlist_has_cache_and_secret_boundaries(name, expected):
    """环境白名单允许科学运行参数，但拒绝缓存覆盖和认证材料。"""

    assert common._external_environment_name_is_allowed(name) is expected


@pytest.mark.parametrize("attribute", ["pidfd_open", "pidfd_send_signal"])
def test_pidfd_helpers_return_unknown_when_native_capability_is_missing(
    monkeypatch, attribute
):
    """系统没有 pidfd 能力时辅助函数保持未知，不退回裸 PID 信号。"""

    if attribute == "pidfd_open":
        monkeypatch.setattr(common.os, "pidfd_open", None, raising=False)
        assert common._open_pidfd_for_signal(123) is None
    else:
        monkeypatch.setattr(common.signal, "pidfd_send_signal", None, raising=False)
        assert common._send_pidfd_signal(9, signal.SIGTERM) is False


def test_pidfd_open_supports_native_and_linux_syscall_paths(monkeypatch):
    """pidfd 打开同时覆盖标准库、Linux syscall 和失败结果。"""

    monkeypatch.setattr(
        common.os,
        "pidfd_open",
        lambda pid, flags: pid + flags + 1,
        raising=False,
    )
    assert common._open_pidfd_for_signal(12) == 13

    class FakeLibc:
        def syscall(self, *_args):
            return 17

    monkeypatch.setattr(common.os, "pidfd_open", 1, raising=False)
    monkeypatch.setattr(common.sys, "platform", "linux")
    monkeypatch.setattr(common.ctypes, "CDLL", lambda *_args, **_kwargs: FakeLibc())
    assert common._open_pidfd_for_signal(12) == 17

    class FailingLibc:
        def syscall(self, *_args):
            return -1

    monkeypatch.setattr(common.ctypes, "CDLL", lambda *_args, **_kwargs: FailingLibc())
    monkeypatch.setattr(common.ctypes, "get_errno", lambda: 1)
    assert common._open_pidfd_for_signal(12) is None

    monkeypatch.setattr(common.sys, "platform", "darwin")
    assert common._open_pidfd_for_signal(12) is None


def test_pidfd_send_supports_native_and_linux_syscall_paths(monkeypatch):
    """pidfd 发信号覆盖标准库成功、异常和 Linux syscall 分支。"""

    sent = []
    monkeypatch.setattr(
        common.signal,
        "pidfd_send_signal",
        lambda descriptor, received, *_args: sent.append((descriptor, received)),
        raising=False,
    )
    assert common._send_pidfd_signal(4, signal.SIGTERM) is True
    assert sent == [(4, signal.SIGTERM)]

    def fail_native(*_args):
        raise OSError("pidfd unavailable")

    monkeypatch.setattr(common.signal, "pidfd_send_signal", fail_native)
    assert common._send_pidfd_signal(4, signal.SIGTERM) is False

    class FakeLibc:
        def syscall(self, *_args):
            return 0

    monkeypatch.setattr(common.signal, "pidfd_send_signal", 1, raising=False)
    monkeypatch.setattr(common.sys, "platform", "linux")
    monkeypatch.setattr(common.ctypes, "CDLL", lambda *_args, **_kwargs: FakeLibc())
    assert common._send_pidfd_signal(4, signal.SIGKILL) is True

    class FailingLibc:
        def syscall(self, *_args):
            return -1

    monkeypatch.setattr(common.ctypes, "CDLL", lambda *_args, **_kwargs: FailingLibc())
    monkeypatch.setattr(common.ctypes, "get_errno", lambda: 1)
    assert common._send_pidfd_signal(4, signal.SIGKILL) is False
    monkeypatch.setattr(common.sys, "platform", "darwin")
    assert common._send_pidfd_signal(4, signal.SIGKILL) is False


def test_runtime_cache_profile_validation_accepts_only_profile_paths(
    monkeypatch, tmp_path: Path
):
    """缓存变量必须来自可信 profile 且位于声明的缓存根目录。"""

    root = tmp_path / "cache"
    root.mkdir()
    profile = {
        "MINICHEM_MODEL_CACHE": str(root),
        "HF_HOME": str(root / "hf"),
        "TORCH_HOME": str(root / "torch"),
    }
    monkeypatch.setattr(runtime, "runtime_environment", lambda _name: profile)

    values = common._validated_runtime_cache_environment(
        {"MINICHEM_BACKEND_RUNTIME": "core"},
        {"HF_HOME": str(root / "hf")},
    )
    assert values == {
        "MINICHEM_MODEL_CACHE": str(root.resolve()),
        "HF_HOME": str((root / "hf").resolve()),
        "TORCH_HOME": str((root / "torch").resolve()),
    }

    monkeypatch.setattr(runtime, "runtime_environment", lambda _name: {})
    assert common._validated_runtime_cache_environment(
        {"MINICHEM_BACKEND_RUNTIME": "core"}, {}
    ) == {}
    monkeypatch.setattr(
        runtime,
        "runtime_environment",
        lambda _name: (_ for _ in ()).throw(KeyError("profile")),
    )
    with pytest.raises(OSError, match="trusted runtime"):
        common._validated_runtime_cache_environment(
            {"MINICHEM_BACKEND_RUNTIME": "core"}, {"HF_HOME": "x"}
        )

    monkeypatch.setattr(
        runtime,
        "runtime_environment",
        lambda _name: (_ for _ in ()).throw(KeyError("profile")),
    )
    assert common._validated_runtime_cache_environment(
        {"MINICHEM_BACKEND_RUNTIME": "core"}, {}
    ) == {}


@pytest.mark.parametrize("runtime_name", ["fairchem", "ash_neb"])
def test_gpu_runtime_cache_paths_stay_under_declared_model_cache(runtime_name: str):
    """GPU runtime 的所有缓存目录都必须位于声明的真实缓存根内。"""

    environment = runtime.runtime_environment(runtime_name)
    cache_root = Path(environment["MINICHEM_MODEL_CACHE"]).resolve()
    for name in (
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_XET_CACHE",
        "TORCH_HOME",
        "FAIRCHEM_CACHE_DIR",
    ):
        Path(environment[name]).resolve().relative_to(cache_root)


def test_runtime_cache_profile_rejects_escape_and_override(
    monkeypatch, tmp_path: Path
):
    """profile 缓存路径越界或被显式覆盖时必须失败关闭。"""

    root = tmp_path / "cache"
    root.mkdir()
    monkeypatch.setattr(
        runtime,
        "runtime_environment",
        lambda _name: {
            "MINICHEM_MODEL_CACHE": str(root),
            "HF_HOME": str(tmp_path / "outside"),
        },
    )
    with pytest.raises(OSError, match="escapes"):
        common._validated_runtime_cache_environment(
            {"MINICHEM_BACKEND_RUNTIME": "core"}, {}
        )

    monkeypatch.setattr(
        runtime,
        "runtime_environment",
        lambda _name: {
            "MINICHEM_MODEL_CACHE": str(root),
            "HF_HOME": str(root / "hf"),
        },
    )
    with pytest.raises(OSError, match="overridden"):
        common._validated_runtime_cache_environment(
            {"MINICHEM_BACKEND_RUNTIME": "core"}, {"HF_HOME": str(root / "other")}
        )


def test_external_environment_adds_private_dirs_and_trusted_cache(
    monkeypatch, tmp_path: Path
):
    """外部环境固定 scratch、HOME 和 profile 缓存，不继承调用方缓存。"""

    directory = tmp_path / "job"
    directory.mkdir()
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("PATH", "/safe/bin")
    monkeypatch.setenv("HF_HOME", "/caller/cache")
    monkeypatch.setenv("MINICHEM_BACKEND_RUNTIME", "core")
    monkeypatch.setattr(
        runtime,
        "runtime_environment",
        lambda _name: {
            "MINICHEM_MODEL_CACHE": str(cache),
            "HF_HOME": str(cache / "hf"),
        },
    )
    values = common._build_external_environment(
        directory,
        {"MINICHEM_MODEL_CACHE": str(cache), "MINICHEM_SAFE_SETTING": "yes"},
    )
    assert values["PATH"] == "/safe/bin"
    assert values["HF_HOME"] == str((cache / "hf").resolve())
    assert values["MINICHEM_SAFE_SETTING"] == "yes"
    assert values["TMPDIR"] == str(directory / ".tmp")
    assert values["HOME"] == str(directory / ".home")
    assert values["MPLCONFIGDIR"] == str(directory / ".tmp" / "matplotlib")
    for name in ("TMPDIR", "HOME", "MPLCONFIGDIR"):
        assert Path(values[name]).is_dir()

    monkeypatch.delenv("MINICHEM_BACKEND_RUNTIME")
    with pytest.raises(OSError, match="profile is required"):
        common._build_external_environment(
            directory,
            {"HF_HOME": str(cache / "hf")},
        )

    monkeypatch.delenv("PATH", raising=False)
    fallback = common._build_external_environment(directory, None)
    assert fallback["PATH"] == os.defpath


def test_external_environment_handles_explicit_and_inherited_gaussian_scratch(
    monkeypatch, tmp_path: Path
):
    """Gaussian scratch 显式目录必须核验，继承目录则重定向到 job 临时目录。"""

    directory = tmp_path / "job"
    directory.mkdir()
    explicit = tmp_path / "gaussian"
    monkeypatch.setenv("MINICHEM_BACKEND_RUNTIME", "core")
    monkeypatch.setattr(runtime, "runtime_environment", lambda _name: {})
    values = common._build_external_environment(
        directory,
        {"GAUSS_SCRDIR": str(explicit)},
    )
    assert values["GAUSS_SCRDIR"] == str(explicit)
    assert explicit.is_dir()

    monkeypatch.setenv("GAUSS_SCRDIR", "/caller/scratch")
    inherited = common._build_external_environment(directory, None)
    assert inherited["GAUSS_SCRDIR"] == str(directory / ".tmp" / "gaussian")

    invalid = tmp_path / "not-directory"
    invalid.write_text("x", encoding="utf-8")
    with pytest.raises(OSError):
        common._build_external_environment(invalid, None)


def test_process_identity_helpers_require_complete_matching_identity(monkeypatch):
    """进程身份核验缺字段、活动异常或二次读取变化时均失败关闭。"""

    expected = _identity()
    assert common._process_identity_matches(expected, expected) is True
    assert common._process_identity_matches(expected, _identity(start=12)) is False
    assert common._process_identity_matches(_identity(start=None), expected) is False

    process = SimpleNamespace(pid=123)
    missing = _identity(start=None)
    monkeypatch.setattr(common, "capture_process_identity", lambda _pid: missing)
    assert common._capture_live_process_identity(process) is None

    calls = iter([expected, _identity(start=12)])
    monkeypatch.setattr(common, "capture_process_identity", lambda _pid: next(calls))
    monkeypatch.setattr(common, "process_identity_activity", lambda _value: "alive")
    assert common._capture_live_process_identity(process) is None

    calls = iter([expected, expected])
    monkeypatch.setattr(common, "capture_process_identity", lambda _pid: next(calls))
    monkeypatch.setattr(common, "process_identity_activity", lambda _value: "alive")
    assert common._capture_live_process_identity(process) == expected

    monkeypatch.setattr(common, "process_identity_activity", lambda _value: "unknown")
    calls = iter([expected])
    monkeypatch.setattr(common, "capture_process_identity", lambda _pid: next(calls))
    assert common._capture_live_process_identity(process) is None


def test_verified_process_signal_requires_pidfd_and_rechecks_identity(monkeypatch):
    """外部进程发信号前必须重新核对身份，并正确关闭 pidfd。"""

    process = SimpleNamespace(pid=123, terminate=lambda: None, kill=lambda: None)
    expected = _identity()
    monkeypatch.setattr(common, "_capture_live_process_identity", lambda _p: expected)
    monkeypatch.setattr(common, "_open_pidfd_for_signal", lambda _pid: None)
    assert common._signal_verified_process_group(process, signal.SIGTERM) is False

    descriptors = []
    monkeypatch.setattr(common, "_open_pidfd_for_signal", lambda _pid: 8)
    monkeypatch.setattr(common, "capture_process_identity", lambda _pid: _identity(start=12))
    monkeypatch.setattr(common.os, "close", lambda descriptor: descriptors.append(descriptor))
    assert common._signal_verified_process_group(process, signal.SIGTERM) is False
    assert descriptors == [8]

    sent = []
    monkeypatch.setattr(common, "capture_process_identity", lambda _pid: expected)
    monkeypatch.setattr(
        common,
        "_send_pidfd_signal",
        lambda descriptor, received: sent.append((descriptor, received)) or True,
    )
    assert common._signal_verified_process_group(process, signal.SIGKILL) is True
    assert sent == [(8, signal.SIGKILL)]


def test_communicate_after_verified_termination_reports_cleanup_state(monkeypatch):
    """终止后收集输出分别覆盖正常、二次超时和身份未知。"""

    identity = _identity()

    class Process:
        def __init__(self, responses):
            self.responses = iter(responses)
            self.returncode = 0

        def poll(self):
            return self.returncode

        def communicate(self, timeout=None):
            return next(self.responses)

    monkeypatch.setattr(common, "inspect_process_group_membership", lambda *_: "empty")
    process = Process([("out", "err")])
    assert common._communicate_after_verified_termination(process, identity) == (
        "out",
        "err",
        True,
    )

    class TimeoutProcess(Process):
        def __init__(self):
            self.calls = 0
            self.returncode = None

        def poll(self):
            return self.returncode

        def communicate(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired("worker", timeout)
            self.returncode = 0
            return "out", "err"

    process = TimeoutProcess()
    monkeypatch.setattr(common, "_signal_verified_process_group", lambda *_: True)
    assert common._communicate_after_verified_termination(process, identity)[2] is True

    class StuckProcess(TimeoutProcess):
        def communicate(self, timeout=None):
            self.calls += 1
            raise subprocess.TimeoutExpired("worker", timeout)

    process = StuckProcess()
    monkeypatch.setattr(common, "_signal_verified_process_group", lambda *_: False)
    stdout, stderr, verified = common._communicate_after_verified_termination(
        process, identity
    )
    assert stdout == ""
    assert "cleanup remained pending" in stderr
    assert verified is False


def test_runtime_worker_environment_helpers_cover_cpu_and_gpu_allocations(monkeypatch):
    """worker 环境覆盖零 GPU、已有 GPU、分配 CPU 和 CPU 数量回退。"""

    environment = {"CUDA_VISIBLE_DEVICES": "2, 3", "ROCR_VISIBLE_DEVICES": "4"}
    runtime._apply_worker_thread_limits(environment, None)
    runtime._apply_worker_gpu_visibility(environment, 0, [])
    assert environment["CUDA_VISIBLE_DEVICES"] == ""
    assert environment["ROCR_VISIBLE_DEVICES"] == ""
    runtime._apply_worker_thread_limits(environment, 2)
    runtime._apply_worker_gpu_visibility(environment, 1, [])
    assert environment["OMP_NUM_THREADS"] == "2"
    assert environment["CUDA_VISIBLE_DEVICES"] == "0"
    runtime._apply_worker_gpu_visibility(environment, 2, ["7", "8"])
    assert environment["CUDA_VISIBLE_DEVICES"] == "7,8"
    runtime._apply_worker_cpu_allocation(environment, [1, 3], 2)
    assert environment["MINICHEM_WORKER_CPU_IDS"] == "1,3"
    runtime._apply_worker_cpu_allocation(environment, [], 2)
    assert environment["MINICHEM_WORKER_CPU_COUNT"] == "2"

    monkeypatch.setattr(runtime, "runtime_environment", lambda _name: {"BASE": "1"})
    prepared = runtime._prepare_worker_environment(
        "core",
        {"request": {"resource_limits": {"cpu_cores": 2, "gpu_count": 1}}},
        {"cpu_ids": [0], "gpu_ids": ["5"]},
    )
    assert prepared["BASE"] == "1"
    assert prepared["CUDA_VISIBLE_DEVICES"] == "5"


def test_runtime_worker_response_and_identity_failure_paths(monkeypatch):
    """worker 响应解析、stderr 附加和身份回调失败保持结构化。"""

    process = SimpleNamespace(returncode=0, pid=123)
    parsed = runtime._parse_worker_exchange_result(
        '{"status": "success"}\n', "diagnostic\n", process
    )
    assert parsed == {
        "status": "success",
        "worker_stderr": "diagnostic\n",
        "worker_returncode": 0,
    }
    invalid = runtime._parse_worker_exchange_result("", "bad", process)
    assert invalid["error"]["code"] == "invalid_worker_response"

    terminated = []
    monkeypatch.setattr(
        runtime,
        "_terminate_worker_process",
        lambda worker: terminated.append(worker) or ("", "stopped"),
    )
    failure = runtime._record_worker_identity_or_failure(
        process,
        lambda _identity: (_ for _ in ()).throw(RuntimeError("receipt failed")),
    )
    assert failure["error"]["code"] == "worker_identity_record_failed"
    assert terminated == [process]
    assert runtime._record_worker_identity_or_failure(process, None) is None


def test_worker_response_formatter_emits_reserved_marker():
    """worker formatter 只把规范 object 放在保留 marker 后。"""

    line = format_worker_response({"status": "success"})

    assert line.startswith(WORKER_RESPONSE_PREFIX)
    assert json.loads(line[len(WORKER_RESPONSE_PREFIX) :]) == {"status": "success"}


def test_runtime_worker_response_accepts_result_before_ash_exit_footer():
    """worker 正文后追加 ASH 退出 footer 时仍应解析结构化结果。"""

    process = SimpleNamespace(returncode=0, pid=123)
    parsed = runtime._parse_worker_exchange_result(
        f'{WORKER_RESPONSE_PREFIX}{{"status": "success"}}\n'
        "ASH timing footer: wall=0.1s\n",
        "",
        process,
    )

    assert parsed["status"] == "success"
    assert parsed["worker_returncode"] == 0


def test_runtime_worker_response_rejects_unframed_multi_line_json():
    """没有 marker 的多行 JSON 不能把 backend 任意输出当作结果。"""

    process = SimpleNamespace(returncode=0, pid=123)
    invalid = runtime._parse_worker_exchange_result(
        '{"backend": "raw"}\n{"status": "success"}\n',
        "",
        process,
    )

    assert invalid["error"]["code"] == "invalid_worker_response"


def test_runtime_worker_exchange_oserror_after_start_preserves_unknown_state(
    monkeypatch,
):
    """worker 已启动后 exchange OSError 且无法确认退出时必须保留 unknown。"""

    class StartedProcess:
        pid = 123
        returncode = None

        def communicate(self, *_args, **_kwargs):
            raise OSError(516, "Unknown error 516")

        def poll(self):
            return None

    process = StartedProcess()
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(
        runtime,
        "_terminate_worker_process_safely",
        lambda _process: ("", "cleanup pending", False),
    )

    result = runtime._exchange_worker_payload(
        python=Path("worker-python"),
        environment={},
        payload={},
        timeout_seconds=1,
        on_worker_started=None,
    )

    assert result["error"]["code"] == "worker_exchange_state_unknown"
    assert result["resource_state"] == "unknown"
    assert result["error"]["code"] != "worker_start_failed"


def test_runtime_worker_exchange_oserror_after_confirmed_exit_is_exchange_failure(
    monkeypatch,
):
    """worker 已启动且退出已确认时，OSError 只能标为 exchange failure。"""

    class FinishedProcess:
        pid = 123
        returncode = 1

        def communicate(self, *_args, **_kwargs):
            raise OSError(516, "Unknown error 516")

        def poll(self):
            return self.returncode

    process = FinishedProcess()
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(
        runtime,
        "_terminate_worker_process_safely",
        lambda _process: ("", "cleanup complete", True),
    )

    result = runtime._exchange_worker_payload(
        python=Path("worker-python"),
        environment={},
        payload={},
        timeout_seconds=1,
        on_worker_started=None,
    )

    assert result["error"]["code"] == "worker_exchange_failed"
    assert "resource_state" not in result
    assert result["error"]["code"] != "worker_start_failed"


def test_runtime_worker_exchange_oserror_before_start_is_start_failure(monkeypatch):
    """Popen 尚未返回 worker 时的 OSError 才能标为 start failure。"""

    monkeypatch.setattr(
        runtime.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError(2, "not found")),
    )

    result = runtime._exchange_worker_payload(
        python=Path("worker-python"),
        environment={},
        payload={},
        timeout_seconds=1,
        on_worker_started=None,
    )

    assert result["error"]["code"] == "worker_start_failed"


def test_runtime_worker_signal_tree_uses_verified_pidfd(monkeypatch):
    """worker 终止只向身份二次核对后的 leader pidfd 发信号。"""

    expected = _identity(pid=321, group=321, session=321)
    sent = []
    closed = []
    monkeypatch.setattr(runtime, "_worker_descendant_process_groups", lambda _pid: (9,))
    monkeypatch.setattr(runtime, "capture_process_identity", lambda _pid: expected)
    monkeypatch.setattr(runtime.os, "pidfd_open", lambda _pid, _flags: 4, raising=False)
    monkeypatch.setattr(
        runtime.signal,
        "pidfd_send_signal",
        lambda descriptor, received, *_args: sent.append((descriptor, received)),
        raising=False,
    )
    monkeypatch.setattr(runtime.os, "close", lambda descriptor: closed.append(descriptor))
    assert runtime._signal_worker_tree(321, signal.SIGTERM) == (9,)
    assert sent == [(4, signal.SIGTERM)]
    assert closed == [4]

    monkeypatch.setattr(runtime.os, "pidfd_open", None, raising=False)
    assert runtime._signal_worker_tree(321, signal.SIGKILL) == (9,)


def test_runtime_worker_termination_bounds_nonposix_cleanup(monkeypatch):
    """非 POSIX worker 终止覆盖正常回收、强制回收和二次超时。"""

    class Process:
        pid = 1

        def __init__(self, responses):
            self.responses = iter(responses)
            self.terminated = 0
            self.killed = 0

        def terminate(self):
            self.terminated += 1

        def kill(self):
            self.killed += 1

        def communicate(self, timeout=None):
            response = next(self.responses)
            if isinstance(response, BaseException):
                raise response
            return response

    monkeypatch.setattr(runtime.os, "name", "nt")
    process = Process([("out", "err")])
    assert runtime._terminate_worker_process(process) == ("out", "err")
    assert process.terminated == 1

    process = Process([
        subprocess.TimeoutExpired("worker", 1),
        ("out", "forced"),
    ])
    assert runtime._terminate_worker_process(process) == ("out", "forced")
    assert process.terminated == 1
    assert process.killed == 1

    process = Process([
        subprocess.TimeoutExpired("worker", 1),
        subprocess.TimeoutExpired("worker", 2, output="partial", stderr="still"),
    ])
    assert runtime._terminate_worker_process(process) == ("partial", "still")
    assert process.killed == 1


def test_runtime_probe_helpers_preserve_health_reason_priority(monkeypatch, tmp_path):
    """拆分后的健康投影辅助函数保留缺失原因优先级。"""

    specification = _backend_spec(
        python_modules=("missing",),
        executables=("missing-executable",),
        environment_variables=("API_KEY",),
    )
    missing = runtime._collect_backend_missing_dependencies(
        {"missing": {"available": False}},
        {"missing-executable": None},
        {"API_KEY": False},
        {"resource": {"available": False, "reason_code": "resource_missing"}},
    )
    assert missing == (["missing"], ["missing-executable"], ["API_KEY"], ["resource"])
    assert runtime._resolve_backend_probe_reason(
        tmp_path / "missing-python",
        None,
        *missing,
        {"resource": {"reason_code": "resource_missing", "available": False}},
        False,
    ) == "runtime_python_missing"

    available_spec = _backend_spec()
    assert runtime._backend_implementation_is_available(
        available_spec,
        {},
        {},
        {},
        {},
        None,
    ) is True
    assert runtime._resolve_backend_probe_reason(
        Path("python"), None, [], [], [], [], {}, True
    ) == "ok"


def test_run_external_reports_missing_success_and_timeout(monkeypatch, tmp_path):
    """外部程序入口覆盖找不到命令、成功返回和超时保守状态。"""

    directory = tmp_path / "job"
    directory.mkdir()
    monkeypatch.setattr(common, "resolve_command", lambda *_: None)
    missing = common.run_external(
        executable="not-found",
        arguments=[],
        directory=directory,
    )
    assert missing["available"] is False

    class Process:
        pid = 12
        returncode = 0

        def communicate(self, *_args, **_kwargs):
            return '{"status": "success"}\n', ""

    monkeypatch.setattr(common, "resolve_command", lambda *_: ["tool"])
    monkeypatch.setattr(common.subprocess, "Popen", lambda *_args, **_kwargs: Process())
    monkeypatch.setattr(common, "_build_external_environment", lambda *_: {"PATH": "/bin"})
    success = common.run_external(
        executable="tool",
        arguments=["--check"],
        directory=directory,
    )
    assert success["returncode"] == 0

    class TimeoutProcess(Process):
        returncode = None

        def communicate(self, *_args, **_kwargs):
            raise subprocess.TimeoutExpired("tool", 1)

    monkeypatch.setattr(
        common.subprocess, "Popen", lambda *_args, **_kwargs: TimeoutProcess()
    )
    monkeypatch.setattr(common, "_capture_live_process_identity", lambda _p: None)
    monkeypatch.setattr(common, "_signal_verified_process_group", lambda *_: False)
    monkeypatch.setattr(
        common,
        "_communicate_after_verified_termination",
        lambda *_args: ("", "pending", False),
    )
    timed = common.run_external(
        executable="tool",
        arguments=[],
        directory=directory,
        timeout_seconds=1,
    )
    assert timed["cleanup_pending"] is True
    assert timed["resource_state"] == "unknown"


def test_run_external_decodes_timeout_bytes_before_backend_persistence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """外部程序超时后的 bytes 输出必须先转成文本，避免掩盖真实超时。"""

    class TimeoutProcess:
        pid = 12
        returncode = None

        def communicate(self, *_args, **_kwargs):
            raise subprocess.TimeoutExpired("tool", 1)

    monkeypatch.setattr(common, "resolve_command", lambda *_: ["tool"])
    monkeypatch.setattr(
        common.subprocess, "Popen", lambda *_args, **_kwargs: TimeoutProcess()
    )
    monkeypatch.setattr(common, "_capture_live_process_identity", lambda _p: None)
    monkeypatch.setattr(common, "_signal_verified_process_group", lambda *_: False)
    monkeypatch.setattr(
        common,
        "_communicate_after_verified_termination",
        lambda *_args: (b"partial stdout", b"partial stderr", True),
    )

    result = common.run_external(
        executable="tool",
        arguments=[],
        directory=tmp_path,
        timeout_seconds=1,
    )

    assert result["timeout"] is True
    assert result["returncode"] == 124
    assert result["stdout"] == "partial stdout"
    assert result["stderr"] == "partial stderr"
    assert isinstance(result["stdout"], str)
    assert isinstance(result["stderr"], str)
