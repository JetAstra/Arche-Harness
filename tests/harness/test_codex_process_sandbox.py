"""Codex 外层进程沙箱的公开合同测试。"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest
from openai_codex import Codex
from openai_codex.generated.v2_all import CommandExecResponse

from chemistry_multiagent.harness.base import HarnessError
from chemistry_multiagent.harness import codex_process_sandbox as sandbox_module
from chemistry_multiagent.harness.codex_process_sandbox import (
    CodexProcessSandboxSpec,
    build_codex_process_sandbox_argv,
    validate_vendored_codex_runtime,
)
from chemistry_multiagent.harness.codex_sdk_adapter import build_codex_config


def _executable(path: Path, content: str = "#!/bin/sh\nexit 0\n") -> Path:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _spec(tmp_path: Path, *, read_only_roots: tuple[Path, ...] = ()) -> CodexProcessSandboxSpec:
    runtime = tmp_path / "runtime"
    native = runtime / "vendor" / "target" / "bin" / "codex"
    bwrap = runtime / "vendor" / "target" / "codex-resources" / "bwrap"
    native.parent.mkdir(parents=True)
    bwrap.parent.mkdir(parents=True)
    _executable(native)
    _executable(bwrap)
    wrapper = _executable(tmp_path / "codex-wrapper")
    return CodexProcessSandboxSpec(
        workspace=tmp_path / "workspace",
        codex_home=tmp_path / "codex-home",
        codex_bin=wrapper,
        runtime_root=runtime,
        native_codex_bin=native,
        bwrap_bin=bwrap,
        read_only_roots=read_only_roots,
    )


def test_build_bwrap_argv_binds_only_session_and_approved_roots(tmp_path: Path) -> None:
    """参数数组包含生命周期、网络保留和精确绑定，不经过 shell。"""

    approved = tmp_path / "approved"
    approved.mkdir()
    spec = _spec(tmp_path, read_only_roots=(approved,))

    argv = build_codex_process_sandbox_argv(spec)

    assert argv[0] == str(spec.bwrap_bin)
    assert "--die-with-parent" in argv
    assert "--unshare-all" not in argv
    assert "--unshare-user" not in argv
    assert "--unshare-pid" in argv
    assert "--unshare-ipc" in argv
    assert "--unshare-uts" in argv
    assert "--unshare-cgroup" in argv
    assert "--share-net" in argv
    assert ("--cap-drop", "ALL") in tuple(zip(argv, argv[1:]))
    for capability in ("CAP_SETUID", "CAP_SETGID", "CAP_SETFCAP"):
        assert ("--cap-add", capability) in tuple(zip(argv, argv[1:]))
    assert ("--proc", "/proc") in tuple(zip(argv, argv[1:]))
    assert ("--tmpfs", "/") in tuple(zip(argv, argv[1:]))
    assert "--clearenv" not in argv
    assert "app-server" in argv
    assert argv[-4:] == (
        str(spec.native_codex_bin),
        "app-server",
        "--listen",
        "stdio://",
    )
    assert "stdio://" in argv
    assert all("sh -c" not in item for item in argv)
    assert str(approved) in argv
    assert str(spec.workspace) in argv
    assert str(spec.codex_home) in argv
    assert str(spec.runtime_root) in argv
    argv_triples = tuple(zip(argv, argv[1:], argv[2:]))
    assert ("--setenv", "CODEX_HOME", "/.arche/codex-home") in argv_triples
    assert ("--setenv", "HOME", "/.arche/codex-home") in argv_triples
    assert ("--setenv", "XDG_CONFIG_HOME", "/.arche/codex-home/xdg-config") in argv_triples


def test_build_bwrap_argv_keeps_two_specs_disjoint(tmp_path: Path) -> None:
    """两个会话的可写绑定不能互相复用。"""

    first = _spec(tmp_path / "first")
    second = _spec(tmp_path / "second")

    first_argv = build_codex_process_sandbox_argv(first)
    second_argv = build_codex_process_sandbox_argv(second)

    assert str(first.workspace) in first_argv
    assert str(first.codex_home) in first_argv
    assert str(second.workspace) not in first_argv
    assert str(second.codex_home) not in first_argv
    assert str(second.workspace) in second_argv
    assert str(second.codex_home) in second_argv
    assert str(first.workspace) not in second_argv
    assert str(first.codex_home) not in second_argv


@pytest.mark.parametrize(
    "mutator",
    [
        lambda spec: CodexProcessSandboxSpec(
            workspace=spec.workspace,
            codex_home=spec.workspace,
            codex_bin=spec.codex_bin,
            runtime_root=spec.runtime_root,
            native_codex_bin=spec.native_codex_bin,
            bwrap_bin=spec.bwrap_bin,
        ),
        lambda spec: CodexProcessSandboxSpec(
            workspace=spec.workspace,
            codex_home=spec.workspace / "nested-home",
            codex_bin=spec.codex_bin,
            runtime_root=spec.runtime_root,
            native_codex_bin=spec.native_codex_bin,
            bwrap_bin=spec.bwrap_bin,
        ),
    ],
)
def test_sandbox_rejects_workspace_home_conflicts(tmp_path: Path, mutator) -> None:
    """workspace 与 CODEX_HOME 相同或嵌套时必须 fail-closed。"""

    spec = mutator(_spec(tmp_path))

    with pytest.raises(HarnessError):
        build_codex_process_sandbox_argv(spec)


def test_sandbox_rejects_symlink_and_non_directory_roots(tmp_path: Path) -> None:
    """只读根的符号链接和普通文件都不能成为绑定源。"""

    target = tmp_path / "target"
    target.mkdir()
    symlink = tmp_path / "symlink"
    symlink.symlink_to(target, target_is_directory=True)
    with pytest.raises(HarnessError):
        build_codex_process_sandbox_argv(_spec(tmp_path / "symlink-case", read_only_roots=(symlink,)))

    regular_file = tmp_path / "not-a-directory"
    regular_file.write_text("x", encoding="utf-8")
    with pytest.raises(HarnessError):
        build_codex_process_sandbox_argv(_spec(tmp_path / "file-case", read_only_roots=(regular_file,)))


def test_sandbox_rejects_relative_paths_and_overlapping_read_only_root(tmp_path: Path) -> None:
    """路径不能依赖当前目录，批准只读根也不能覆盖会话或 runtime。"""

    absolute = _spec(tmp_path)
    with pytest.raises(HarnessError):
        build_codex_process_sandbox_argv(
            CodexProcessSandboxSpec(
                workspace=Path("relative-workspace"),
                codex_home=absolute.codex_home,
                codex_bin=absolute.codex_bin,
                runtime_root=absolute.runtime_root,
                native_codex_bin=absolute.native_codex_bin,
                bwrap_bin=absolute.bwrap_bin,
            )
        )

    with pytest.raises(HarnessError, match="protected"):
        build_codex_process_sandbox_argv(
            CodexProcessSandboxSpec(
                workspace=absolute.workspace,
                codex_home=absolute.codex_home,
                codex_bin=absolute.codex_bin,
                runtime_root=absolute.runtime_root,
                native_codex_bin=absolute.native_codex_bin,
                bwrap_bin=absolute.bwrap_bin,
                read_only_roots=(Path("/etc"),),
            )
        )

    with pytest.raises(HarnessError):
        build_codex_process_sandbox_argv(
            CodexProcessSandboxSpec(
                workspace=absolute.workspace,
                codex_home=absolute.codex_home,
                codex_bin=absolute.codex_bin,
                runtime_root=absolute.runtime_root,
                native_codex_bin=absolute.native_codex_bin,
                bwrap_bin=absolute.bwrap_bin,
                read_only_roots=(absolute.runtime_root,),
            )
        )


@pytest.mark.parametrize(
    "conflict_kind",
    ["workspace", "codex_home", "runtime_root", "read_only_root"],
)
def test_sandbox_rejects_reserved_virtual_codex_home_destination_conflicts(
    tmp_path: Path,
    conflict_kind: str,
) -> None:
    """保留的虚拟 CODEX_HOME 不能被其它绑定再次占用。"""

    runtime = tmp_path / "runtime"
    roots = {
        "workspace": tmp_path / "workspace",
        "codex_home": tmp_path / "codex-home",
        "runtime_root": runtime,
        "read_only_root": tmp_path / "approved",
    }
    roots[conflict_kind] = Path("/.arche")
    native = roots["runtime_root"] / "native"
    bwrap = roots["runtime_root"] / "bwrap"
    with pytest.raises(HarnessError, match="reserved virtual CODEX_HOME"):
        sandbox_module._validate_sandbox_binding_paths(
            workspace=roots["workspace"],
            codex_home=roots["codex_home"],
            wrapper=tmp_path / "wrapper",
            runtime_root=roots["runtime_root"],
            native=native,
            bwrap=bwrap,
            read_only_roots=(roots["read_only_root"],),
        )


def test_sandbox_config_uses_launch_override_and_default_config_is_unchanged(tmp_path: Path) -> None:
    """提供 spec 时只改启动参数；不提供 spec 时保留旧 SDK 配置。"""

    spec = _spec(tmp_path)
    sandboxed = build_codex_config(
        codex_bin=str(spec.codex_bin),
        workspace=spec.workspace,
        codex_home=spec.codex_home,
        sandbox_spec=spec,
    )
    legacy = build_codex_config(
        codex_bin=str(spec.codex_bin),
        workspace=spec.workspace,
        codex_home=spec.codex_home,
    )

    assert sandboxed.launch_args_override
    assert sandboxed.launch_args_override[-4:] == (
        str(spec.native_codex_bin),
        "app-server",
        "--listen",
        "stdio://",
    )
    assert sandboxed.env is not None
    assert sandboxed.env["CODEX_HOME"] == "/.arche/codex-home"
    assert sandboxed.env["HOME"] == "/.arche/codex-home"
    assert sandboxed.env["XDG_CONFIG_HOME"] == "/.arche/codex-home/xdg-config"
    assert sandboxed.env["XDG_CACHE_HOME"] == "/.arche/codex-home/xdg-cache"
    assert sandboxed.env["XDG_DATA_HOME"] == "/.arche/codex-home/xdg-data"
    assert ("--bind", str(spec.codex_home.resolve()), "/.arche/codex-home") in tuple(
        zip(
            sandboxed.launch_args_override,
            sandboxed.launch_args_override[1:],
            sandboxed.launch_args_override[2:],
        )
    )
    assert legacy.launch_args_override is None
    assert legacy.codex_bin == str(spec.codex_bin.resolve())
    assert legacy.env == {"CODEX_HOME": str(spec.codex_home.resolve())}
    with pytest.raises(HarnessError, match="provided twice"):
        build_codex_config(
            codex_bin=str(spec.codex_bin),
            workspace=spec.workspace,
            codex_home=spec.codex_home,
            sandbox_spec=spec,
            spec=spec,
        )


def test_sandbox_argv_only_keeps_explicit_provider_environment_name(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """SDK 复制父环境时只保留调用方批准的 provider 变量名。"""

    spec = _spec(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "parent-secret")
    monkeypatch.setenv("CUSTOM_PROVIDER_KEY", "provider-secret")

    argv = build_codex_process_sandbox_argv(spec, environment_names=("CUSTOM_PROVIDER_KEY",))

    assert ("--unsetenv", "OPENAI_API_KEY") in tuple(zip(argv, argv[1:]))
    assert ("--unsetenv", "CUSTOM_PROVIDER_KEY") not in tuple(zip(argv, argv[1:]))
    assert "parent-secret" not in argv
    assert "provider-secret" not in argv


def test_sandbox_fails_closed_when_namespace_probe_fails(tmp_path: Path) -> None:
    """启动者不具备挂载或 namespace 能力时不能回退到裸 app-server。"""

    spec = _spec(tmp_path)
    assert spec.bwrap_bin is not None
    spec.bwrap_bin.write_text("#!/bin/sh\nexit 23\n", encoding="utf-8")
    spec.bwrap_bin.chmod(spec.bwrap_bin.stat().st_mode | stat.S_IXUSR)

    with pytest.raises(HarnessError):
        build_codex_process_sandbox_argv(spec)


def test_wrapper_runtime_probe_uses_private_runtime_home_and_xdg_directories(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """wrapper 首次解包不能把父进程的 Codex 家目录当作临时目录。"""

    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    parent_home = tmp_path / "parent-home"
    parent_home.mkdir()
    (parent_home / ".codex").mkdir()
    (parent_home / ".codex" / "sentinel").write_text("private", encoding="utf-8")
    monkeypatch.setenv("HOME", str(parent_home))
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-cross-boundary")
    captured_environment: dict[str, str] = {}

    def fake_run(command, **kwargs):
        environment = kwargs["env"]
        captured_environment.update(environment)
        for name in (
            "HOME",
            "CODEX_HOME",
            "TMPDIR",
            "XDG_CONFIG_HOME",
            "XDG_CACHE_HOME",
            "XDG_DATA_HOME",
        ):
            path = Path(environment[name])
            assert path != parent_home
            assert path.is_dir()
            assert not path.is_symlink()
            assert stat.S_IMODE(path.stat().st_mode) == 0o700
        assert Path(environment["CODEX_HOME"]) != Path(environment["HOME"])
        assert environment.get("OPENAI_API_KEY") is None
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(sandbox_module.subprocess, "run", fake_run)

    sandbox_module._materialize_wrapper_runtime(
        tmp_path / "codex-wrapper",
        runtime_root,
        expected_native=runtime_root / "native",
        expected_bwrap=runtime_root / "bwrap",
    )

    assert captured_environment["HOME"] != str(parent_home)
    assert captured_environment["CODEX_HOME"] != str(parent_home / ".codex")
    for name in (
        "HOME",
        "CODEX_HOME",
        "TMPDIR",
        "XDG_CONFIG_HOME",
        "XDG_CACHE_HOME",
        "XDG_DATA_HOME",
    ):
        assert not Path(captured_environment[name]).exists()
    assert (parent_home / ".codex" / "sentinel").read_text(encoding="utf-8") == "private"


def _hardlink_runtime_copy(source: Path, target: Path) -> None:
    """用硬链接复制 runtime 目录，测试替换文件时不修改仓库产物。"""

    for source_path in source.rglob("*"):
        relative = source_path.relative_to(source)
        target_path = target / relative
        if source_path.is_dir():
            target_path.mkdir(parents=True, exist_ok=True)
        else:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            os.link(source_path, target_path)


def test_vendored_runtime_integrity_accepts_only_fixed_0149_layout() -> None:
    """固定 wrapper、清单、native 与 bwrap 通过同一份完整性核对。"""

    runtime = validate_vendored_codex_runtime()

    assert runtime.version == "0.149.0"
    assert runtime.target == "x86_64-unknown-linux-musl"
    assert runtime.wrapper.name == "codex"
    assert runtime.native_codex_bin.name == "codex"
    assert runtime.bwrap_bin.name == "bwrap"


def test_vendored_runtime_integrity_rejects_forged_wrapper(tmp_path: Path) -> None:
    """不在仓库固定入口的 wrapper 即使可执行也不能被接纳。"""

    forged = _executable(tmp_path / "codex", "#!/bin/sh\nexit 0\n")

    with pytest.raises(HarnessError, match="fixed vendored"):
        validate_vendored_codex_runtime(forged)


def test_derive_runtime_root_uses_wrapper_layout_and_rejects_missing_manifest(
    tmp_path: Path,
) -> None:
    """runtime 默认目录必须从固定 wrapper 层级推导，缺清单时拒绝猜测。"""

    wrapper = (
        tmp_path
        / "repo"
        / "third_party"
        / "codex_cli"
        / "0.149.0"
        / "linux-x64"
        / "codex"
    )
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text("#!/bin/sh\n", encoding="utf-8")
    expected = (
        tmp_path
        / "repo"
        / "local_data"
        / "runtime"
        / "codex_cli"
        / "0.149.0"
        / "linux-x64"
    )

    assert sandbox_module._derive_runtime_root(wrapper, {"version": "0.149.0"}) == expected
    with pytest.raises(HarnessError, match="package manifest"):
        sandbox_module._derive_runtime_root(wrapper, None)


def test_vendored_runtime_integrity_rejects_forged_layout_manifest(tmp_path: Path) -> None:
    """布局清单被改写后必须在读取 runtime 前失败。"""

    repository_root = Path(__file__).resolve().parents[2]
    source = repository_root / "third_party" / "codex_cli" / "0.149.0" / "manifest.json"
    forged = tmp_path / "manifest.json"
    shutil.copyfile(source, forged)
    forged.write_bytes(forged.read_bytes().replace(b'"0.149.0"', b'"0.149.1"', 1))

    with pytest.raises(HarnessError, match="SHA-256"):
        validate_vendored_codex_runtime(layout_manifest=forged)


@pytest.mark.parametrize("component", ["native", "bwrap"])
def test_vendored_runtime_integrity_rejects_forged_native_or_bwrap(
    tmp_path: Path,
    component: str,
) -> None:
    """runtime 中伪造 native 或 bwrap 时必须拒绝启动。"""

    repository_root = Path(__file__).resolve().parents[2]
    source = (
        repository_root
        / "local_data"
        / "runtime"
        / "codex_cli"
        / "0.149.0"
        / "linux-x64"
    )
    if not source.is_dir():
        pytest.skip("vendored Codex runtime is not materialized")
    runtime = source.parent / f"_integrity_probe_{tmp_path.name}"
    _hardlink_runtime_copy(source, runtime)
    try:
        relative = (
            "vendor/x86_64-unknown-linux-musl/bin/codex"
            if component == "native"
            else "vendor/x86_64-unknown-linux-musl/codex-resources/bwrap"
        )
        forged = runtime / relative
        forged.unlink()
        forged.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        forged.chmod(forged.stat().st_mode | stat.S_IXUSR)

        with pytest.raises(HarnessError, match="size|SHA-256"):
            validate_vendored_codex_runtime(runtime_root=runtime)
    finally:
        shutil.rmtree(runtime)


def test_real_vendored_bwrap_hides_sibling_and_initializes_app_server(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """真实 vendored bwrap 允许会话写入且隐藏 sibling sentinel。"""

    repository_root = Path(__file__).resolve().parents[2]
    wrapper = repository_root / "third_party" / "codex_cli" / "0.149.0" / "linux-x64" / "codex"
    runtime_root = repository_root / "local_data" / "runtime" / "codex_cli" / "0.149.0" / "linux-x64"
    bwrap = runtime_root / "vendor" / "x86_64-unknown-linux-musl" / "codex-resources" / "bwrap"
    native = runtime_root / "vendor" / "x86_64-unknown-linux-musl" / "bin" / "codex"
    if not wrapper.is_file() or not os.access(wrapper, os.X_OK):
        pytest.skip("vendored Codex wrapper unavailable")
    if not bwrap.is_file() or not os.access(bwrap, os.X_OK) or not native.is_file():
        pytest.skip("vendored Codex runtime is not materialized")
    if subprocess.run([str(bwrap), "--version"], capture_output=True, timeout=10).returncode != 0:
        pytest.skip("vendored bwrap cannot run on this host")

    workspace = tmp_path / "workspace"
    # 模拟真实运行时的长持久目录；Codex sqlite 初始化必须只看到沙箱内的
    # 固定短路径，而 manifest 仍然绑定这个宿主目录。
    home = tmp_path
    for index in range(6):
        home /= f"host-codex-home-{index}-" + ("h" * 90)
    sibling = tmp_path / "private"
    workspace.mkdir()
    home.mkdir(parents=True)
    sibling.mkdir()
    assert len(str(home.resolve())) >= 503
    (sibling / "sentinel").write_text("private", encoding="utf-8")
    host_sentinel_name = "ARCHE_CODEX_PROCESS_SANDBOX_HOST_SENTINEL"
    monkeypatch.setenv(host_sentinel_name, "must-not-cross-boundary")
    host_pid = str(os.getpid())
    host_namespace_inodes = tuple(
        str(os.stat(f"/proc/self/ns/{name}").st_ino)
        for name in ("user", "mnt", "pid", "ipc", "uts", "cgroup")
    )
    spec = CodexProcessSandboxSpec(
        workspace=workspace,
        codex_home=home,
        codex_bin=wrapper,
        runtime_root=runtime_root,
        native_codex_bin=native,
        bwrap_bin=bwrap,
    )
    argv = build_codex_process_sandbox_argv(spec)
    probe = subprocess.run(
        [
            *argv[:-4],
            "/usr/bin/sh",
            "-c",
            'printf visible > "$PWD/probe"; printf home > "$CODEX_HOME/home-probe"; '
            'test ! -e "$1"; test ! -e /root; test ! -e "/proc/$2"; '
            'test ! -e "/proc/$2/environ"; '
            'test "$(stat -c %i /proc/self/ns/user)" = "$5"; '
            'test "$(stat -c %i /proc/self/ns/mnt)" != "$6"; '
            'test "$(stat -c %i /proc/self/ns/pid)" != "$7"; '
            'test "$(stat -c %i /proc/self/ns/ipc)" != "$8"; '
            'test "$(stat -c %i /proc/self/ns/uts)" != "$9"; '
            'test "$(stat -c %i /proc/self/ns/cgroup)" != "${10}"; '
            'if touch "$3/.codex-runtime-write-probe" 2>/dev/null; then '
            'rm -f "$3/.codex-runtime-write-probe"; exit 1; fi; '
            'if touch /usr/.codex-system-write-probe 2>/dev/null; then '
            'rm -f /usr/.codex-system-write-probe; exit 1; fi; '
            'hits=0; for proc_dir in /proc/[0-9]*; do '
            f'if test -r "$proc_dir/environ" && tr "\\0" "\\n" < "$proc_dir/environ" '
            f'| grep -q "^{host_sentinel_name}="; then hits=$((hits + 1)); fi; '
            'done; test "$hits" -eq 0; test -z "${'
            f'{host_sentinel_name}:-}}"',
            "probe",
            str(sibling / "sentinel"),
            host_pid,
            str(runtime_root),
            str(home),
            *host_namespace_inodes,
        ],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert probe.returncode == 0, probe.stderr
    assert (workspace / "probe").read_text(encoding="utf-8") == "visible"
    assert (home / "home-probe").read_text(encoding="utf-8") == "home"

    host_home_before_initialize = set(home.iterdir())
    config = build_codex_config(
        codex_bin=str(wrapper),
        workspace=workspace,
        codex_home=home,
        sandbox_spec=spec,
    )
    client = Codex(config)
    try:
        assert client.metadata.serverInfo.version.startswith("0.149.0")
        assert set(home.iterdir()) != host_home_before_initialize
        response = client._client.request(
            "command/exec",
            {
                "command": [
                    "/bin/sh",
                    "-c",
                    'line=$(grep "^CapEff:" /proc/self/status); '
                    'test "${line#CapEff: }" = 0000000000000000; '
                    'printf inner > "$PWD/inner-probe"; test ! -e "$1"; '
                    'test ! -e /root; test ! -e "/proc/$2"; '
                    'test ! -e "/proc/$2/environ"; test -z "${'
                    f'{host_sentinel_name}:-}}"; printf PASS',
                    "inner-probe",
                    str(sibling / "sentinel"),
                    host_pid,
                ],
                "cwd": str(workspace),
                "sandboxPolicy": {
                    "type": "workspaceWrite",
                    "networkAccess": False,
                    "writableRoots": [str(workspace)],
                },
                "timeoutMs": 10000,
            },
            response_model=CommandExecResponse,
        )
        assert response.exit_code == 0, response.stderr
        assert response.stdout == "PASS"
        assert (workspace / "inner-probe").read_text(encoding="utf-8") == "inner"
    finally:
        client.close()
