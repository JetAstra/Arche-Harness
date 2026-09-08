"""为每个 Codex app-server 进程构造独立的 bubblewrap 外层沙箱。

这个模块只负责宿主进程的启动边界。它不替代 Codex thread 的
``workspace-write`` 和 ``network_access=false`` 策略，也不承诺隐藏内核、
``/proc`` 元数据或抵御宿主 root。调用方必须显式提供一个不可变 spec；
未提供 spec 时，适配层继续使用原来的 SDK 启动方式。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .base import HarnessError


_RUNTIME_PROBE_TIMEOUT_SECONDS = 60.0
_RUNTIME_VERSION_ENV = "ARCHE_CODEX_RUNTIME_ROOT"
_OUTER_CAPABILITIES_REQUIRED_FOR_INNER_BWRAP = (
    "CAP_SETUID",
    "CAP_SETGID",
    "CAP_SETFCAP",
)
_PROBE_ENV_EXCLUDES = frozenset(
    {
        "ARCHE_CODEX_API_KEY",
        "ARCHE_CHEM_API_KEY",
        "ARCHE_REVIEW_CODEX_API_KEY",
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "HF_TOKEN",
    }
)
_RUNTIME_TARGET = "x86_64-unknown-linux-musl"
_CODEX_VERSION = "0.149.0"
_CODEX_PLATFORM = "linux-x64"
_CODEX_WRAPPER_SHA256 = "5be09b89d6cbad86496423a6119423abf1ad1de4a69dfaf4245766433a7675f6"
_CODEX_LAYOUT_MANIFEST_SHA256 = "729fa32efdf15d3cd9a02f05f87e0ad576dcd23b3d0aa3e7b2aed8957e6fbce7"
_CODEX_PACKAGE_MANIFEST_SHA256 = "0b3f869f9fbd009f4d0c68c68ab50d53778855899b19106a607c6cd1861637a3"
_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_BASE_ENV_NAMES = frozenset(
    {
        "CODEX_HOME",
        "HOME",
        "CODEX_MANAGED_PACKAGE_ROOT",
        "ARCHE_CODEX_RUNTIME_ROOT",
        "PATH",
        "TMPDIR",
        "XDG_CONFIG_HOME",
        "XDG_CACHE_HOME",
        "XDG_DATA_HOME",
    }
)
_NETWORK_ENV_NAMES = frozenset(
    {
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "CODEX_CA_CERTIFICATES",
        "LANG",
        "LC_ALL",
        "TERM",
    }
)
_PROTECTED_BIND_DESTINATIONS = frozenset(
    {
        "/",
        "/tmp",
        "/run",
        "/proc",
        "/dev",
        "/etc",
        "/usr",
        "/bin",
        "/sbin",
        "/lib",
        "/lib64",
        "/home",
        "/root",
        "/mnt",
    }
)
# bwrap 内部使用固定的短路径承载宿主上的持久 CODEX_HOME。宿主路径仍由
# CodexProcessSandboxSpec 和会话 manifest 保存；该常量只作为沙箱内的目的地。
_VIRTUAL_CODEX_HOME = Path("/.arche/codex-home")
_CODEX_COMPONENT_INTEGRITY = {
    "codex": {
        "runtime_path": "vendor/x86_64-unknown-linux-musl/bin/codex",
        "sha256": "bbc3341e44c9ead340ed9570c17be936e37870f570751a941699ffd04d672827",
        "size": 258322048,
        "executable": True,
    },
    "codex-code-mode-host": {
        "runtime_path": "vendor/x86_64-unknown-linux-musl/bin/codex-code-mode-host",
        "sha256": "ba0f620a1d242a7555750e96c30c4c03b8b89ec0d4cff987e21bcd79fa18a363",
        "size": 57882552,
        "executable": True,
    },
    "bwrap": {
        "runtime_path": "vendor/x86_64-unknown-linux-musl/codex-resources/bwrap",
        "sha256": "7df960565a0dece99240ea4b9d0e011307817f9f3b73176c7b71fda44fe84765",
        "size": 529776,
        "executable": True,
    },
    "zsh": {
        "runtime_path": "vendor/x86_64-unknown-linux-musl/codex-resources/zsh/bin/zsh",
        "sha256": "67faaaa89242c4a332e16e508a1977cffc24bf7fca31d4411cdfd101f3831ef3",
        "size": 898480,
        "executable": True,
    },
    "rg": {
        "runtime_path": "vendor/x86_64-unknown-linux-musl/codex-path/rg",
        "sha256": "e62198eb19b136b88c330af83647b5a962cb99b6b1f066758568f12de1974849",
        "size": 5408904,
        "executable": True,
    },
}
_CODEX_COMPONENT_ARCHIVES = {
    "codex": "linux-x64/codex.bin.xz",
    "codex-code-mode-host": "linux-x64/codex-code-mode-host.bin.xz",
    "bwrap": "linux-x64/bwrap.bin.xz",
    "zsh": "linux-x64/zsh.bin.xz",
    "rg": "linux-x64/rg.bin.xz",
}
_CODEX_ARCHIVE_INTEGRITY = {
    "codex": "aa107e694a655b646642d8519e4d4783fb01bd4a5ed166f4238bf0157b9a4a68",
    "codex-code-mode-host": "1e53f1b2759e3863eb4aa482ce2578a63fa7a536ddba6338d8d7b5575c36cf28",
    "bwrap": "86018e9bd1de9095984b913f4b3a973e91c7c2e542048058e1aa0d166da9b912",
    "zsh": "a3a3777b4402cf6f12051ed4686246f49a06fa969fc43ba59c88e06fd9e7f395",
    "rg": "acbbf1c51e2320edacd7ff8c17cf9bbc90a456268f3974084ac77a03b0325b5a",
}


class CodexProcessSandboxError(HarnessError):
    """表示外层 Codex 进程沙箱无法安全准备或启动。"""

    def __init__(self, message: str) -> None:
        super().__init__(message, kind="transport")


@dataclass(frozen=True, slots=True)
class CodexVendoredRuntime:
    """保存一次已通过固定版本和 SHA-256 核对的 Codex 运行时路径。"""

    wrapper: Path
    layout_manifest: Path
    package_manifest: Path
    runtime_root: Path
    native_codex_bin: Path
    bwrap_bin: Path
    version: str
    target: str


@dataclass(frozen=True, slots=True)
class CodexProcessSandboxSpec:
    """冻结一次 Codex 进程沙箱的身份和允许绑定的目录。

    ``workspace`` 和宿主上的 ``codex_home`` 是本次 Harness 唯一的可写目录；
    两者不能相同或互相包含。启动时宿主 ``codex_home`` 会绑定到沙箱内固定的
    短路径 ``/.arche/codex-home``，因此其长的持久路径不会进入 Codex 的
    sqlite 状态路径解析；spec 和会话 manifest 仍保存宿主路径。``codex_bin``
    是 vendored wrapper，wrapper 解包后的 native binary 和 bwrap 必须落在同一
    个 ``runtime_root`` 内。除了 ``runtime_root`` 和系统必需只读目录，只有
    ``read_only_roots`` 中的调用方批准目录会被绑定到沙箱；每个根必须是已有的
    真实目录。

    构造 spec 不执行外部程序。首次生成 SDK 配置时，缺少 wrapper 解包产物
    才会以参数数组、局部环境和固定超时调用 ``codex --version``；失败会
    抛出异常，不会退回未隔离启动。
    """

    workspace: Path
    codex_home: Path
    codex_bin: Path
    runtime_root: Path | None = None
    read_only_roots: tuple[Path, ...] = ()
    bwrap_bin: Path | None = None
    native_codex_bin: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "workspace", Path(self.workspace))
        object.__setattr__(self, "codex_home", Path(self.codex_home))
        object.__setattr__(self, "codex_bin", Path(self.codex_bin))
        if self.runtime_root is not None:
            object.__setattr__(self, "runtime_root", Path(self.runtime_root))
        if self.bwrap_bin is not None:
            object.__setattr__(self, "bwrap_bin", Path(self.bwrap_bin))
        if self.native_codex_bin is not None:
            object.__setattr__(self, "native_codex_bin", Path(self.native_codex_bin))
        if isinstance(self.read_only_roots, (str, bytes)):
            raise TypeError("read_only_roots must be a sequence of paths")
        object.__setattr__(
            self,
            "read_only_roots",
            tuple(Path(root) for root in self.read_only_roots),
        )


def _sandbox_error(message: str) -> CodexProcessSandboxError:
    """创建不带路径内容和环境值的沙箱错误。"""

    return CodexProcessSandboxError(message)


def _repository_root() -> Path:
    """返回仓库根目录，路径只由当前源码位置推导。"""

    return Path(__file__).resolve().parents[3]


def _canonical_vendored_paths() -> tuple[Path, Path, Path]:
    """返回固定 0.149.0 wrapper、布局清单和 package 清单路径。"""

    codex_root = _repository_root() / "third_party" / "codex_cli" / _CODEX_VERSION
    wrapper = codex_root / _CODEX_PLATFORM / "codex"
    layout_manifest = codex_root / "manifest.json"
    package_manifest = wrapper.with_name("codex-package.json")
    return wrapper, layout_manifest, package_manifest


def _default_runtime_root() -> Path:
    """返回固定 0.149.0 解包目录，不接受 PATH 或其它运行时替代品。"""

    return (
        _repository_root()
        / "local_data"
        / "runtime"
        / "codex_cli"
        / _CODEX_VERSION
        / _CODEX_PLATFORM
    )


def _sha256_file(path: Path, *, field_name: str) -> str:
    """分块计算普通文件摘要；文件不可读时以沙箱错误终止。"""

    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise _sandbox_error(f"{field_name} cannot be hashed") from exc
    return digest.hexdigest()


def _validate_hashed_file(
    path: Path,
    *,
    field_name: str,
    expected_sha256: str,
    expected_size: int | None = None,
    executable: bool = False,
) -> Path:
    """核对普通文件、大小、摘要和可执行位，任一差异都拒绝启动。"""

    absolute = _absolute_path(path, field_name)
    try:
        if absolute.is_symlink() or not absolute.is_file():
            raise _sandbox_error(f"{field_name} is not a regular file")
        if expected_size is not None and absolute.stat().st_size != expected_size:
            raise _sandbox_error(f"{field_name} size does not match the vendored manifest")
        if executable and not os.access(absolute, os.X_OK):
            raise _sandbox_error(f"{field_name} is not executable")
    except CodexProcessSandboxError:
        raise
    except OSError as exc:
        raise _sandbox_error(f"{field_name} cannot be inspected") from exc
    if _sha256_file(absolute, field_name=field_name) != expected_sha256:
        raise _sandbox_error(f"{field_name} SHA-256 does not match the vendored manifest")
    return absolute.resolve()


def _absolute_path(path: Path, field_name: str) -> Path:
    """核对调用方给出的绝对路径，并在解析前拒绝符号链接组件。

    这里不执行相对路径、``~``、环境变量或 glob 展开；bwrap 的每个源和
    目标都来自已核对的字面路径，不依赖全局当前工作目录。
    """

    try:
        expanded = Path(path)
        if not expanded.is_absolute():
            raise _sandbox_error(f"{field_name} must be absolute")
        _reject_symlink_components(expanded, field_name)
        return expanded.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise _sandbox_error(f"{field_name} path cannot be resolved") from exc


def _reject_symlink_components(path: Path, field_name: str) -> None:
    """拒绝路径中的任意符号链接，避免绑定目标在校验后改变。"""

    current = Path(path.anchor or "/")
    parts = path.parts[1:] if path.is_absolute() else path.parts
    for part in parts:
        current /= part
        try:
            if current.is_symlink():
                raise _sandbox_error(f"{field_name} must not contain a symlink")
        except OSError as exc:
            raise _sandbox_error(f"{field_name} path cannot be inspected") from exc


def _prepare_writable_directory(path: Path, field_name: str) -> Path:
    """创建会话可写目录并设为 0700；目标异常时 fail-closed。"""

    absolute = _absolute_path(path, field_name)
    try:
        if absolute.exists() and not absolute.is_dir():
            raise _sandbox_error(f"{field_name} must be a directory")
        absolute.mkdir(parents=True, exist_ok=True)
        _reject_symlink_components(absolute, field_name)
        if not absolute.is_dir():
            raise _sandbox_error(f"{field_name} must be a directory")
        os.chmod(absolute, 0o700)
    except CodexProcessSandboxError:
        raise
    except OSError as exc:
        raise _sandbox_error(f"{field_name} cannot be prepared") from exc
    return absolute.resolve()


def _require_directory(path: Path, field_name: str) -> Path:
    """核对只读绑定根是已有的真实目录。"""

    absolute = _absolute_path(path, field_name)
    try:
        if absolute.is_symlink() or not absolute.is_dir():
            raise _sandbox_error(f"{field_name} must be a regular directory")
    except CodexProcessSandboxError:
        raise
    except OSError as exc:
        raise _sandbox_error(f"{field_name} cannot be inspected") from exc
    return absolute.resolve()


def _require_executable(path: Path, field_name: str) -> Path:
    """核对 vendored wrapper、native binary 或 bwrap 是真实可执行文件。"""

    absolute = _absolute_path(path, field_name)
    try:
        if absolute.is_symlink() or not absolute.is_file() or not os.access(absolute, os.X_OK):
            raise _sandbox_error(f"{field_name} is not executable")
    except CodexProcessSandboxError:
        raise
    except OSError as exc:
        raise _sandbox_error(f"{field_name} cannot be inspected") from exc
    return absolute.resolve()


def _path_is_within(path: Path, parent: Path) -> bool:
    """判断一个已解析路径是否等于或位于另一个路径内。"""

    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _paths_overlap(first: Path, second: Path) -> bool:
    """判断两个绑定路径是否相同或互相包含。"""

    return _path_is_within(first, second) or _path_is_within(second, first)


def _ensure_distinct_paths(paths: Sequence[tuple[str, Path]]) -> None:
    """拒绝 writable、runtime 和批准 roots 的冲突或嵌套。"""

    for index, (first_name, first) in enumerate(paths):
        for second_name, second in paths[index + 1 :]:
            if _paths_overlap(first, second):
                raise _sandbox_error(f"{first_name} overlaps {second_name}")


def _read_json_file(path: Path) -> Mapping[str, Any] | None:
    """读取本地 runtime 清单；清单损坏时返回 None 让后续校验失败。"""

    try:
        if path.is_symlink() or not path.is_file():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, Mapping) else None


def _load_wrapper_manifest(wrapper: Path) -> Mapping[str, Any] | None:
    """读取 wrapper 旁边的官方 package 清单，不读取任何环境值。"""

    return _read_json_file(wrapper.parent / "codex-package.json")


def _read_vendored_manifest_documents(
    layout: Path,
    package: Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """读取布局和 package 清单，并确认二者都是 JSON 对象。"""

    try:
        layout_value = json.loads(layout.read_text(encoding="utf-8"))
        package_value = json.loads(package.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _sandbox_error("Codex vendored manifest cannot be parsed") from exc
    if not isinstance(layout_value, Mapping) or not isinstance(package_value, Mapping):
        raise _sandbox_error("Codex vendored manifest must contain objects")
    return layout_value, package_value


def _validate_vendored_layout_identity(layout_value: Mapping[str, Any]) -> None:
    """核对固定 Codex 版本、平台、目标和布局版本。"""

    expected = {
        "version": _CODEX_VERSION,
        "platform": _CODEX_PLATFORM,
        "target": _RUNTIME_TARGET,
        "runtime_layout_version": 1,
    }
    if any(layout_value.get(key) != value for key, value in expected.items()):
        raise _sandbox_error("Codex vendored layout manifest identity is invalid")


def _validate_vendored_package_identity(package_value: Mapping[str, Any]) -> None:
    """核对 package 清单的入口、资源目录和版本绑定。"""

    expected = {
        "layoutVersion": 1,
        "version": _CODEX_VERSION,
        "target": _RUNTIME_TARGET,
        "variant": "codex",
        "entrypoint": "bin/codex",
        "resourcesDir": "codex-resources",
        "pathDir": "codex-path",
    }
    if any(package_value.get(key) != value for key, value in expected.items()):
        raise _sandbox_error("Codex package manifest identity is invalid")


def _validate_vendored_package_binding(layout_value: Mapping[str, Any]) -> None:
    """核对布局清单记录的 package 路径和固定摘要。"""

    package_metadata = layout_value.get("package_manifest")
    if not isinstance(package_metadata, Mapping):
        raise _sandbox_error("Codex layout manifest package binding is missing")
    if package_metadata.get("path") != f"{_CODEX_PLATFORM}/codex-package.json":
        raise _sandbox_error("Codex layout manifest package binding is invalid")
    if package_metadata.get("sha256") != _CODEX_PACKAGE_MANIFEST_SHA256:
        raise _sandbox_error("Codex layout manifest package hash is invalid")


def _index_vendored_components(
    layout_value: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    """把组件清单转成唯一名称映射，并核对组件集合完整。"""

    components = layout_value.get("components")
    if not isinstance(components, list) or len(components) != len(_CODEX_COMPONENT_INTEGRITY):
        raise _sandbox_error("Codex vendored component manifest is incomplete")
    component_by_name: dict[str, Mapping[str, Any]] = {}
    for component in components:
        if not isinstance(component, Mapping):
            raise _sandbox_error("Codex vendored component manifest is malformed")
        name = component.get("name")
        if not isinstance(name, str) or name in component_by_name:
            raise _sandbox_error("Codex vendored component manifest has duplicate names")
        component_by_name[name] = component
    if set(component_by_name) != set(_CODEX_COMPONENT_INTEGRITY):
        raise _sandbox_error("Codex vendored component manifest identity is invalid")
    return component_by_name


def _load_vendored_manifest_values(
    layout: Path,
    package: Path,
) -> tuple[Mapping[str, Any], dict[str, Mapping[str, Any]]]:
    """读取并核对布局、package 和组件身份字段。"""

    layout_value, package_value = _read_vendored_manifest_documents(layout, package)
    _validate_vendored_layout_identity(layout_value)
    _validate_vendored_package_identity(package_value)
    _validate_vendored_package_binding(layout_value)
    return layout_value, _index_vendored_components(layout_value)


def _validate_vendored_archives(
    component_by_name: Mapping[str, Mapping[str, Any]],
    codex_root: Path,
) -> None:
    """核对组件的归档绑定和仓库内归档 SHA-256。"""

    for name, expected in _CODEX_COMPONENT_INTEGRITY.items():
        component = component_by_name[name]
        archive = component.get("archive")
        if archive != _CODEX_COMPONENT_ARCHIVES[name]:
            raise _sandbox_error(f"Codex component {name} archive binding is invalid")
        if (
            component.get("runtime_path") != expected["runtime_path"]
            or component.get("binary_sha256") != expected["sha256"]
            or component.get("binary_size") != expected["size"]
            or component.get("executable") is not True
        ):
            raise _sandbox_error(f"Codex component {name} manifest identity is invalid")
        _validate_hashed_file(
            codex_root / str(archive),
            field_name=f"Codex component {name} archive",
            expected_sha256=_CODEX_ARCHIVE_INTEGRITY[name],
        )


def _prepare_and_validate_vendored_runtime(
    wrapper: Path,
    runtime_root: Path | str | None,
) -> tuple[Path, dict[str, Path]]:
    """物化并核对 runtime 内全部组件，返回目录和组件路径映射。"""

    runtime = _default_runtime_root() if runtime_root is None else _absolute_path(
        Path(runtime_root), "runtime_root"
    )
    try:
        runtime.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise _sandbox_error("runtime_root cannot be prepared") from exc
    _reject_symlink_components(runtime, "runtime_root")
    runtime = _require_directory(runtime, "runtime_root")
    components = {
        name: runtime / str(expected["runtime_path"])
        for name, expected in _CODEX_COMPONENT_INTEGRITY.items()
    }
    if any(not path.is_file() for path in components.values()):
        _materialize_wrapper_runtime(
            wrapper,
            runtime,
            expected_native=components["codex"],
            expected_bwrap=components["bwrap"],
        )
    for name, expected in _CODEX_COMPONENT_INTEGRITY.items():
        _validate_hashed_file(
            components[name],
            field_name=f"Codex runtime component {name}",
            expected_sha256=str(expected["sha256"]),
            expected_size=int(expected["size"]),
            executable=bool(expected["executable"]),
        )
    _validate_hashed_file(
        runtime / "vendor" / _RUNTIME_TARGET / "codex-package.json",
        field_name="Codex runtime package manifest",
        expected_sha256=_CODEX_PACKAGE_MANIFEST_SHA256,
    )
    return runtime, components


def validate_vendored_codex_runtime(
    codex_bin: Path | str | None = None,
    *,
    runtime_root: Path | str | None = None,
    layout_manifest: Path | str | None = None,
) -> CodexVendoredRuntime:
    """核对仓库固定的 Codex 0.149.0 wrapper、清单、native 和 bwrap。

    输入是可选的固定 wrapper、解包目录和布局清单路径；动作是拒绝非仓库
    wrapper、清单 SHA-256 漂移、归档摘要漂移，以及 runtime 内任一 sidecar、
    native 或 bwrap 的大小、可执行位和 SHA-256 漂移。缺失的 runtime 组件会
    通过固定 wrapper 的 ``--version`` 在指定目录原子解包；网络、PATH 和其它
    Codex 安装不会参与选择。成功返回只含已核对路径的运行时身份；任一步失败
    都抛出 ``CodexProcessSandboxError``，调用方不得降级到未核对的二进制。

    ``layout_manifest`` 仅用于把一份副本拿来做离线完整性测试，内容仍必须
    与仓库固定 SHA-256 完全一致；wrapper 路径始终必须是仓库内固定入口。
    """

    canonical_wrapper, canonical_layout, canonical_package = _canonical_vendored_paths()
    wrapper = canonical_wrapper if codex_bin is None else _absolute_path(Path(codex_bin), "Codex wrapper")
    if wrapper != canonical_wrapper:
        raise _sandbox_error("Codex wrapper is not the fixed vendored 0.149.0 entrypoint")
    wrapper = _validate_hashed_file(
        wrapper,
        field_name="Codex wrapper",
        expected_sha256=_CODEX_WRAPPER_SHA256,
        executable=True,
    )
    layout = canonical_layout if layout_manifest is None else _absolute_path(
        Path(layout_manifest), "Codex layout manifest"
    )
    layout = _validate_hashed_file(
        layout,
        field_name="Codex layout manifest",
        expected_sha256=_CODEX_LAYOUT_MANIFEST_SHA256,
    )
    package = _validate_hashed_file(
        canonical_package,
        field_name="Codex package manifest",
        expected_sha256=_CODEX_PACKAGE_MANIFEST_SHA256,
    )
    layout_value, component_by_name = _load_vendored_manifest_values(layout, package)
    _validate_vendored_archives(component_by_name, canonical_wrapper.parent.parent)
    runtime, components = _prepare_and_validate_vendored_runtime(wrapper, runtime_root)
    return CodexVendoredRuntime(
        wrapper=wrapper,
        layout_manifest=layout,
        package_manifest=package,
        runtime_root=runtime,
        native_codex_bin=components["codex"],
        bwrap_bin=components["bwrap"],
        version=str(layout_value["version"]),
        target=str(layout_value["target"]),
    )


def _derive_runtime_root(wrapper: Path, manifest: Mapping[str, Any] | None) -> Path:
    """从 wrapper 的 vendored 相对布局推导默认 runtime 目录。"""

    version = manifest.get("version") if manifest is not None else None
    if not isinstance(version, str) or not version.strip():
        raise _sandbox_error("Codex wrapper package manifest is unavailable")
    # 官方 wrapper 的脚本目录为 third_party/codex_cli/<version>/linux-x64；
    # 只从 wrapper 所在位置向上推导仓库，不依赖当前工作目录或固定绝对路径。
    repository_root = wrapper.parent
    for _ in range(4):
        repository_root = repository_root.parent
    return repository_root / "local_data" / "runtime" / "codex_cli" / version / "linux-x64"


def _runtime_probe_environment(
    runtime_root: Path,
    *,
    probe_root: Path | None = None,
) -> dict[str, str]:
    """构造 wrapper 解包或外层能力探针的局部环境，并排除凭据。

    输入是 runtime 根和可选的临时 probe 根；提供后者时，HOME、CODEX_HOME、
    TMPDIR 和 XDG 目录都落在这个 runtime 专属目录中，避免 wrapper 把父进程的
    ``~/.codex`` 当作共享写入位置。副作用是创建并校验这些 0700 目录；返回排除
    provider 凭据后的子进程环境。目录不安全或无法创建时抛出 sandbox 错误。
    外层 namespace 探针不提供 ``probe_root``，因此仍沿用原来的启动行为。
    """

    environment: dict[str, str] = {}
    for name in ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL"):
        value = os.environ.get(name)
        if isinstance(value, str) and value:
            environment[name] = value
    environment[_RUNTIME_VERSION_ENV] = str(runtime_root)
    if probe_root is not None:
        isolated_directories = {
            "HOME": "home",
            "CODEX_HOME": "codex-home",
            "TMPDIR": "tmp",
            "XDG_CONFIG_HOME": "xdg-config",
            "XDG_CACHE_HOME": "xdg-cache",
            "XDG_DATA_HOME": "xdg-data",
        }
        for environment_name, directory_name in isolated_directories.items():
            directory = _prepare_writable_directory(
                probe_root / directory_name,
                f"runtime probe {environment_name}",
            )
            environment[environment_name] = str(directory)
    for name in _PROBE_ENV_EXCLUDES:
        environment.pop(name, None)
    return environment


def _materialize_wrapper_runtime(
    wrapper: Path,
    runtime_root: Path,
    *,
    expected_native: Path,
    expected_bwrap: Path,
) -> None:
    """在缺少组件时以参数数组触发 wrapper 解包，失败不降级。

    输入是固定 wrapper、runtime 根和两个必需组件路径；组件已存在时直接返回。
    缺失时以参数数组执行 ``--version`` 触发解包，并为该进程创建独立且 0700 的
    HOME、CODEX_HOME、TMPDIR 和 XDG 目录。副作用仅限 runtime 物化及临时目录
    创建，探针完成后临时目录会被删除。成功返回 ``None``；进程、超时、目录安全
    或非零退出都会抛出不含子进程正文的 sandbox 错误。
    """

    if expected_native.is_file() and expected_bwrap.is_file():
        return
    try:
        with tempfile.TemporaryDirectory(
            prefix=".codex-runtime-probe-",
            dir=str(runtime_root),
        ) as raw_probe_root:
            probe_root = _prepare_writable_directory(
                Path(raw_probe_root),
                "runtime probe root",
            )
            completed = subprocess.run(
                [str(wrapper), "--version"],
                cwd=str(wrapper.parent),
                env=_runtime_probe_environment(runtime_root, probe_root=probe_root),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,
                check=False,
                timeout=_RUNTIME_PROBE_TIMEOUT_SECONDS,
            )
    except (OSError, subprocess.SubprocessError, CodexProcessSandboxError) as exc:
        raise _sandbox_error("Codex wrapper runtime probe failed") from exc
    if completed.returncode != 0:
        raise _sandbox_error("Codex wrapper runtime probe failed")


def _probe_outer_namespace_capability(bwrap: Path) -> None:
    """证明当前启动者能挂载私有 proc 并保留 inner bwrap 所需的最小能力。

    本节点在 outer user namespace 内禁止挂载 proc，因此显式 opt-in 模式不
    创建 outer user namespace，而由 bwrap 在启动设置 mount/PID/IPC/UTS/cgroup
    namespace 后挂载私有 proc。probe 使用 ``--tmpfs /``，只绑定基本系统目录，
    并在命令进程上 ``--cap-drop ALL`` 后恢复 inner bwrap 写 uid/gid map 所需的
    三个位；任何权限、内核或 bwrap 能力失败都会抛错，不会回退伪隔离启动。
    """

    system_binds = _system_read_only_binds()
    destination_paths = [
        "/",
        "/proc",
        "/dev",
        *(destination for _, destination in system_binds),
    ]
    arguments: list[str] = [
        str(bwrap),
        "--die-with-parent",
        "--new-session",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--unshare-cgroup",
        "--share-net",
        "--cap-drop",
        "ALL",
    ]
    for capability in _OUTER_CAPABILITIES_REQUIRED_FOR_INNER_BWRAP:
        arguments.extend(("--cap-add", capability))
    arguments.extend(("--tmpfs", "/"))
    for directory in _destination_parent_directories(destination_paths):
        arguments.extend(("--dir", directory))
    arguments.extend(("--proc", "/proc", "--dev", "/dev"))
    for source, destination in system_binds:
        arguments.extend(("--ro-bind", str(source), destination))
    arguments.extend(("--", "/usr/bin/true"))
    try:
        completed = subprocess.run(
            arguments,
            cwd="/",
            env=_runtime_probe_environment(bwrap.parent),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            check=False,
            timeout=_RUNTIME_PROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise _sandbox_error("outer namespace capability probe failed") from exc
    if completed.returncode != 0:
        raise _sandbox_error("outer namespace capability probe failed")


def _runtime_layout_paths(
    wrapper: Path,
    runtime_root: Path,
    manifest: Mapping[str, Any] | None,
) -> tuple[Path, Path]:
    """返回 runtime 内 native Codex 与 bwrap 的预期位置。"""

    target = manifest.get("target") if manifest is not None else None
    target = target if isinstance(target, str) and target.strip() else _RUNTIME_TARGET
    entrypoint = manifest.get("entrypoint") if manifest is not None else None
    entrypoint = entrypoint if isinstance(entrypoint, str) and entrypoint.strip() else "bin/codex"
    native = runtime_root / "vendor" / target / entrypoint
    bwrap = runtime_root / "vendor" / target / "codex-resources" / "bwrap"
    return native, bwrap


def _prepare_sandbox_runtime_root(
    spec: CodexProcessSandboxSpec,
    *,
    wrapper: Path,
    manifest: Mapping[str, Any] | None,
) -> tuple[Path, Path, Path]:
    """准备 runtime_root 并物化 native Codex 与 bwrap 预期文件。"""

    runtime_root_raw = spec.runtime_root
    runtime_root = (
        _absolute_path(runtime_root_raw, "runtime_root")
        if runtime_root_raw is not None
        else _derive_runtime_root(wrapper, manifest)
    )
    try:
        runtime_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise _sandbox_error("runtime_root cannot be prepared") from exc
    _reject_symlink_components(runtime_root, "runtime_root")
    runtime_root = _require_directory(runtime_root, "runtime_root")
    expected_native, expected_bwrap = _runtime_layout_paths(wrapper, runtime_root, manifest)
    _materialize_wrapper_runtime(
        wrapper,
        runtime_root,
        expected_native=expected_native,
        expected_bwrap=expected_bwrap,
    )
    return runtime_root, expected_native, expected_bwrap


def _validate_canonical_sandbox_runtime_bindings(
    spec: CodexProcessSandboxSpec,
    *,
    wrapper: Path,
    runtime_root: Path,
    expected_native: Path,
    expected_bwrap: Path,
) -> tuple[Path, Path]:
    """对固定 wrapper 做完整性核对，并核对 spec 没有替换组件路径。

    非仓库测试替身沿用轻量文件校验；固定 0.149.0 wrapper 则必须通过完整
    wrapper、layout、package、归档和 runtime 组件 SHA-256 核对，避免公共
    ``build_codex_process_sandbox_argv`` 被调用时绕过 Runner 的生产校验。
    """

    canonical_wrapper, _, _ = _canonical_vendored_paths()
    if wrapper != canonical_wrapper:
        return expected_native, expected_bwrap
    verified = validate_vendored_codex_runtime(
        codex_bin=wrapper,
        runtime_root=runtime_root,
    )
    if verified.runtime_root != runtime_root:
        raise _sandbox_error("Codex runtime_root changed during integrity validation")
    if spec.native_codex_bin is not None and Path(spec.native_codex_bin).resolve(
        strict=False
    ) != verified.native_codex_bin:
        raise _sandbox_error("Codex native binary does not match the vendored runtime")
    if spec.bwrap_bin is not None and Path(spec.bwrap_bin).resolve(
        strict=False
    ) != verified.bwrap_bin:
        raise _sandbox_error("vendored bwrap does not match the vendored runtime")
    return verified.native_codex_bin, verified.bwrap_bin


def _validate_sandbox_runtime_components(
    spec: CodexProcessSandboxSpec,
    *,
    runtime_root: Path,
    expected_native: Path,
    expected_bwrap: Path,
) -> tuple[Path, Path]:
    """核对 native Codex、bwrap 和外层 namespace 能力。"""

    native = _require_executable(
        spec.native_codex_bin if spec.native_codex_bin is not None else expected_native,
        "Codex native binary",
    )
    bwrap = _require_executable(
        spec.bwrap_bin if spec.bwrap_bin is not None else expected_bwrap,
        "vendored bwrap",
    )
    _probe_outer_namespace_capability(bwrap)
    _require_directory(runtime_root, "runtime_root")
    return native, bwrap


def _validate_sandbox_binding_paths(
    *,
    workspace: Path,
    codex_home: Path,
    wrapper: Path,
    runtime_root: Path,
    native: Path,
    bwrap: Path,
    read_only_roots: tuple[Path, ...],
) -> None:
    """拒绝 runtime、wrapper、会话目录和只读绑定之间的重叠。

    虚拟 CODEX_HOME 是保留的 bwrap 目的地，不能被 workspace、runtime、宿主
    CODEX_HOME 或只读根再次作为同路径或嵌套目的地；否则后续 bind 顺序可能
    覆盖持久状态或把其它宿主目录暴露到保留位置。路径冲突直接 fail-closed。
    """

    if len(set(read_only_roots)) != len(read_only_roots):
        raise _sandbox_error("read_only_roots contain duplicates")
    for index, root in enumerate(read_only_roots):
        if str(root) in _PROTECTED_BIND_DESTINATIONS:
            raise _sandbox_error(
                f"read_only_root[{index}] is a protected sandbox destination"
            )
    if not _path_is_within(native, runtime_root):
        raise _sandbox_error("Codex native binary is outside runtime_root")
    if not _path_is_within(bwrap, runtime_root):
        raise _sandbox_error("vendored bwrap is outside runtime_root")
    _ensure_distinct_paths(
        [
            ("workspace", workspace),
            ("codex_home", codex_home),
            ("runtime_root", runtime_root),
            *[(f"read_only_root[{index}]", root) for index, root in enumerate(read_only_roots)],
        ]
    )
    for name, path in (
        ("workspace", workspace),
        ("codex_home", codex_home),
        ("runtime_root", runtime_root),
        *[(f"read_only_root[{index}]", root) for index, root in enumerate(read_only_roots)],
    ):
        if _paths_overlap(path, _VIRTUAL_CODEX_HOME):
            raise _sandbox_error(f"{name} overlaps the reserved virtual CODEX_HOME")
    if _paths_overlap(wrapper, workspace) or _paths_overlap(wrapper, codex_home):
        raise _sandbox_error("Codex wrapper overlaps a writable session directory")


def prepare_codex_process_sandbox(spec: CodexProcessSandboxSpec) -> CodexProcessSandboxSpec:
    """校验并物化 spec，返回所有路径已解析的可启动 spec。

    输入是调用方显式提供的冻结 spec；实际动作包括创建两个 0700 会话目录、
    必要时运行 vendored wrapper 解包、核对 native/bwrap 和拒绝所有路径重叠。
    成功返回可直接用于构造 argv 的新 spec；任一步失败都抛出
    ``CodexProcessSandboxError``，不会返回未隔离配置。
    """

    if not isinstance(spec, CodexProcessSandboxSpec):
        raise _sandbox_error("Codex process sandbox spec is invalid")
    workspace = _prepare_writable_directory(spec.workspace, "workspace")
    codex_home = _prepare_writable_directory(spec.codex_home, "codex_home")
    wrapper = _require_executable(spec.codex_bin, "Codex wrapper")
    manifest = _load_wrapper_manifest(wrapper)
    runtime_root, expected_native, expected_bwrap = _prepare_sandbox_runtime_root(
        spec,
        wrapper=wrapper,
        manifest=manifest,
    )
    expected_native, expected_bwrap = _validate_canonical_sandbox_runtime_bindings(
        spec,
        wrapper=wrapper,
        runtime_root=runtime_root,
        expected_native=expected_native,
        expected_bwrap=expected_bwrap,
    )
    native, bwrap = _validate_sandbox_runtime_components(
        spec,
        runtime_root=runtime_root,
        expected_native=expected_native,
        expected_bwrap=expected_bwrap,
    )
    read_only_roots = tuple(
        _require_directory(root, f"read_only_root[{index}]")
        for index, root in enumerate(spec.read_only_roots)
    )
    _validate_sandbox_binding_paths(
        workspace=workspace,
        codex_home=codex_home,
        wrapper=wrapper,
        runtime_root=runtime_root,
        native=native,
        bwrap=bwrap,
        read_only_roots=read_only_roots,
    )
    return replace(
        spec,
        workspace=workspace,
        codex_home=codex_home,
        codex_bin=wrapper,
        runtime_root=runtime_root,
        read_only_roots=read_only_roots,
        bwrap_bin=bwrap,
        native_codex_bin=native,
    )


def _existing_system_directory(path: str) -> tuple[Path, str] | None:
    """返回可绑定的基本系统目录，并把源符号链接解析为真实目录。"""

    source = Path(path)
    try:
        if not source.exists() or not source.is_dir():
            return None
        return source.resolve(), path
    except OSError:
        return None


def _existing_system_file(path: str) -> tuple[Path, str] | None:
    """返回可绑定的 CA、DNS 或用户数据库文件。"""

    source = Path(path)
    try:
        if not source.exists() or not source.is_file():
            return None
        return source.resolve(), path
    except OSError:
        return None


def _system_read_only_binds() -> tuple[tuple[Path, str], ...]:
    """列出运行静态 Codex、shell 和 HTTPS 所需的最小系统目录与文件。"""

    bindings: list[tuple[Path, str]] = []
    for path in ("/usr", "/bin", "/sbin", "/lib", "/lib64"):
        binding = _existing_system_directory(path)
        if binding is not None:
            bindings.append(binding)
    for path in (
        "/etc/passwd",
        "/etc/group",
        "/etc/nsswitch.conf",
        "/etc/hosts",
        "/etc/resolv.conf",
        "/etc/localtime",
        "/etc/ssl/certs",
        "/etc/ca-certificates",
        "/etc/pki",
    ):
        if path.endswith(("/certs", "/ca-certificates", "/pki")):
            binding = _existing_system_directory(path)
        else:
            binding = _existing_system_file(path)
        if binding is not None:
            bindings.append(binding)
    return tuple(bindings)


def _destination_parent_directories(paths: Iterable[str]) -> tuple[str, ...]:
    """返回所有 bind 目标的父目录，按深度排序供 bwrap 先创建。"""

    parents: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path)
        current = path.parent
        while str(current) not in {"", "/", "."}:
            parents.add(str(current))
            current = current.parent
    return tuple(sorted(parents, key=lambda value: (len(Path(value).parts), value)))


def build_codex_process_sandbox_argv(
    spec: CodexProcessSandboxSpec,
    *,
    environment_names: Sequence[str] = (),
) -> tuple[str, ...]:
    """生成无 shell 的 bwrap 参数数组，最后启动实际 native app-server。

    返回值可直接传给 ``subprocess.Popen`` 或 SDK ``launch_args_override``。
    外层不创建 user namespace，而使用独立的 PID、IPC、UTS 和 cgroup namespace；
    ``--proc /proc`` 因 PID namespace 只显示沙箱内进程。通过 ``--share-net``
    保留网络，以便 app-server 访问 Responses 与 localhost MCP；启动前 probe
    已证明挂载权限和 inner bwrap 所需的最小能力。``--cap-drop ALL`` 后只恢复
    inner bwrap 写 uid/gid map 所需的三个能力，inner command 自身仍由 Codex
    bwrap 清零 CapEff。本函数不宣称隐藏所有内核元数据或抵御宿主 root。
    """

    prepared = prepare_codex_process_sandbox(spec)
    assert prepared.runtime_root is not None
    assert prepared.bwrap_bin is not None
    assert prepared.native_codex_bin is not None
    system_binds = _system_read_only_binds()
    fixed_destination_paths = [
        str(prepared.workspace),
        str(_VIRTUAL_CODEX_HOME),
        str(prepared.runtime_root),
        *(str(root) for root in prepared.read_only_roots),
        *(destination for _, destination in system_binds),
        "/etc",
        "/tmp",
        "/run",
        "/dev",
    ]
    arguments: list[str] = [
        str(prepared.bwrap_bin),
        "--die-with-parent",
        "--new-session",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--unshare-cgroup",
        # 不创建 outer user namespace；probe 已证明当前启动者可挂载私有 proc。
        "--share-net",
        "--cap-drop",
        "ALL",
    ]
    for capability in _OUTER_CAPABILITIES_REQUIRED_FOR_INNER_BWRAP:
        arguments.extend(("--cap-add", capability))
    allowed_environment_names = {
        *_BASE_ENV_NAMES,
        *_NETWORK_ENV_NAMES,
    }
    for name in environment_names:
        normalized_name = str(name).strip()
        if not _ENVIRONMENT_NAME.fullmatch(normalized_name):
            raise _sandbox_error("environment_names contain an invalid variable name")
        allowed_environment_names.add(normalized_name)
    # SDK 0.149 会把父进程环境先复制给 bwrap；逐名删除未批准变量，
    # provider key 只有在调用方通过 environment_names 明确批准时才保留。
    # 这样不需要把 API key 或 proxy URL 放进 argv 的 --setenv 值中。
    for name in sorted(set(os.environ).difference(allowed_environment_names)):
        arguments.extend(("--unsetenv", name))
    virtual_home = str(_VIRTUAL_CODEX_HOME)
    # 这些值不含凭据，直接固定在 argv 中，使调用方即使只使用 argv 也不会
    # 把宿主长 CODEX_HOME、XDG 或临时目录带入沙箱；provider 凭据仍只按上面的
    # environment_names 白名单从继承环境进入。
    for name, value in (
        ("CODEX_HOME", virtual_home),
        ("HOME", virtual_home),
        ("CODEX_MANAGED_PACKAGE_ROOT", str(prepared.runtime_root)),
        ("ARCHE_CODEX_RUNTIME_ROOT", str(prepared.runtime_root)),
        ("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"),
        ("TMPDIR", "/tmp"),
        ("XDG_CONFIG_HOME", f"{virtual_home}/xdg-config"),
        ("XDG_CACHE_HOME", f"{virtual_home}/xdg-cache"),
        ("XDG_DATA_HOME", f"{virtual_home}/xdg-data"),
    ):
        arguments.extend(("--setenv", name, value))
    for directory in _destination_parent_directories(fixed_destination_paths):
        arguments.extend(("--dir", directory))
    arguments.extend(
        (
            "--tmpfs",
            "/",
            "--tmpfs",
            "/tmp",
            "--tmpfs",
            "/run",
            "--proc",
            "/proc",
        )
    )
    arguments.extend(("--dev", "/dev"))
    arguments.extend(("--bind", str(prepared.workspace), str(prepared.workspace)))
    # 宿主 CODEX_HOME 只作为 source 出现；短的固定 destination 同时供 Codex
    # 环境变量和 bwrap 使用，避免长宿主路径参与 app-server sqlite 初始化。
    arguments.extend(("--bind", str(prepared.codex_home), str(_VIRTUAL_CODEX_HOME)))
    arguments.extend(
        ("--ro-bind", str(prepared.runtime_root), str(prepared.runtime_root))
    )
    for root in prepared.read_only_roots:
        arguments.extend(("--ro-bind", str(root), str(root)))
    for source, destination in system_binds:
        arguments.extend(("--ro-bind", str(source), destination))
    arguments.extend(
        (
            "--chdir",
            str(prepared.workspace),
            "--",
            str(prepared.native_codex_bin),
            "app-server",
            "--listen",
            "stdio://",
        )
    )
    return tuple(arguments)


def build_codex_process_sandbox_environment(
    spec: CodexProcessSandboxSpec,
) -> dict[str, str]:
    """返回传给 SDK 的进程级环境覆盖，不修改父进程环境。

    宿主 ``codex_home`` 是持久身份目录，环境中的 HOME、CODEX_HOME 和 XDG
    目录统一使用 bwrap 内的固定短路径；普通未启用沙箱的 SDK 配置仍继续使用
    宿主路径。调用方只获得环境 mapping，无法借此修改父进程环境。
    """

    prepared = prepare_codex_process_sandbox(spec)
    assert prepared.runtime_root is not None
    virtual_home = str(_VIRTUAL_CODEX_HOME)
    return {
        "CODEX_HOME": virtual_home,
        "HOME": virtual_home,
        "CODEX_MANAGED_PACKAGE_ROOT": str(prepared.runtime_root),
        "ARCHE_CODEX_RUNTIME_ROOT": str(prepared.runtime_root),
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "TMPDIR": "/tmp",
        "XDG_CONFIG_HOME": f"{virtual_home}/xdg-config",
        "XDG_CACHE_HOME": f"{virtual_home}/xdg-cache",
        "XDG_DATA_HOME": f"{virtual_home}/xdg-data",
    }


__all__ = [
    "CodexProcessSandboxError",
    "CodexProcessSandboxSpec",
    "CodexVendoredRuntime",
    "build_codex_process_sandbox_argv",
    "build_codex_process_sandbox_environment",
    "prepare_codex_process_sandbox",
    "validate_vendored_codex_runtime",
]
