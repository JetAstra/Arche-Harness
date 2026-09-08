"""从环境变量读取 Planner Harness 配置。"""

from __future__ import annotations

import os
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


SUPPORTED_HARNESSES = frozenset({"react", "codex"})
SUPPORTED_REASONING_EFFORTS = frozenset({"minimal", "low", "medium", "high", "xhigh"})
CODEX_API_KEY_ENV = "ARCHE_CODEX_API_KEY"
CODEX_UNBOUNDED_TURN_STAGES_ENV = "ARCHE_CODEX_UNBOUNDED_TURN_STAGES"
CODEX_HARNESS_ROLES = frozenset({"planner", "investigation", "review", "task_review"})
DEFAULT_CODEX_BIN = (
    Path(__file__).resolve().parents[3]
    / "third_party"
    / "codex_cli"
    / "0.149.0"
    / "linux-x64"
    / "codex"
).resolve()
CODEX_PACKAGE_MANIFEST = DEFAULT_CODEX_BIN.with_name("codex-package.json")
CODEX_LAYOUT_MANIFEST = DEFAULT_CODEX_BIN.parents[1] / "manifest.json"
_CODEX_VERSION = "0.149.0"


def _read_repository_codex_manifests() -> tuple[dict[str, object], dict[str, object]]:
    """读取并确认固定 0.149.0 的两个 JSON 清单都是对象。"""

    try:
        layout = json.loads(CODEX_LAYOUT_MANIFEST.read_text(encoding="utf-8"))
        package = json.loads(CODEX_PACKAGE_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Codex Planner 0.149.0 wrapper manifest 缺失或损坏") from exc
    if not isinstance(layout, dict) or not isinstance(package, dict):
        raise ValueError("Codex Planner wrapper manifest 必须是 JSON 对象")
    return layout, package


def _validate_repository_codex_manifest_identity(
    layout: dict[str, object],
    package: dict[str, object],
) -> None:
    """核对清单版本、相对路径和 package 哈希绑定。"""

    if layout.get("version") != _CODEX_VERSION:
        raise ValueError(f"Codex Planner wrapper manifest version 不是 {_CODEX_VERSION}")
    package_metadata = layout.get("package_manifest")
    if not isinstance(package_metadata, dict) or package_metadata.get("path") != str(
        CODEX_PACKAGE_MANIFEST.relative_to(DEFAULT_CODEX_BIN.parents[1])
    ):
        raise ValueError("Codex Planner wrapper manifest 未绑定 package manifest")
    if package.get("version") != _CODEX_VERSION:
        raise ValueError(f"Codex Planner package manifest version 不是 {_CODEX_VERSION}")
    expected_hash = package_metadata.get("sha256")
    if not isinstance(expected_hash, str) or hashlib.sha256(
        CODEX_PACKAGE_MANIFEST.read_bytes()
    ).hexdigest() != expected_hash:
        raise ValueError("Codex Planner package manifest hash 校验失败")


def _validate_repository_codex_wrapper(path: Path) -> None:
    """校验生产入口、0.149.0 包清单和仓库布局清单的绑定关系。"""

    if path != DEFAULT_CODEX_BIN:
        raise ValueError(
            "Codex Planner 生产入口必须是仓库内固定的 third_party/codex_cli/0.149.0 wrapper"
        )
    if path.is_symlink() or not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"Codex Planner wrapper 不可执行：{path}")
    if CODEX_LAYOUT_MANIFEST.is_symlink() or CODEX_PACKAGE_MANIFEST.is_symlink():
        raise ValueError("Codex Planner wrapper manifest 不得是符号链接")
    layout, package = _read_repository_codex_manifests()
    _validate_repository_codex_manifest_identity(layout, package)


def _positive_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    value = float(raw)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _stage_names_from_env(name: str) -> frozenset[str]:
    """读取明确允许取消 Codex turn 总时限的阶段名称集合。"""

    raw = os.environ.get(name, "").strip()
    if not raw:
        return frozenset()
    return frozenset(
        item.strip().lower()
        for item in raw.split(",")
        if item.strip()
    )


def _resolve_executable_path(value: str) -> str:
    """在 Runner 切换工作目录前解析显式给出的可执行文件路径。"""

    raw = value.strip() or "codex"
    expanded = os.path.expanduser(raw)
    has_path_separator = os.sep in expanded or (
        os.altsep is not None and os.altsep in expanded
    )
    if not has_path_separator:
        return expanded
    path = Path(expanded)
    if not path.is_absolute():
        path = Path.cwd() / path
    return str(path.resolve())


def _read_codex_provider_settings(kind: str) -> tuple[str, str, str]:
    """读取 Codex provider 的地址、模型和密钥存在性，不返回密钥正文。"""

    codex_base_url = os.environ.get("ARCHE_CODEX_BASE_URL", "").strip()
    codex_model = os.environ.get("ARCHE_CODEX_MODEL_NAME", "").strip()
    codex_api_key = os.environ.get(CODEX_API_KEY_ENV, "").strip()
    if kind == "codex":
        missing = [
            name
            for name, value in (
                ("ARCHE_CODEX_BASE_URL", codex_base_url),
                (CODEX_API_KEY_ENV, codex_api_key),
                ("ARCHE_CODEX_MODEL_NAME", codex_model),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                "Codex Planner 缺少独立 provider 配置；缺少：" + ", ".join(missing)
            )
    return codex_base_url, codex_model, codex_api_key


def _resolve_harness_output_paths() -> tuple[Path, Path, Path | None]:
    """解析 workspace、trace 和 CODEX_HOME，避免 Runner 切换目录后漂移。"""

    workspace_root = Path(
        os.environ.get(
            "ARCHE_STAGE_HARNESS_WORKSPACE_ROOT",
            "local_data/runtime/harness_workspaces",
        )
    ).expanduser()
    trace_root = Path(
        os.environ.get(
            "ARCHE_STAGE_HARNESS_TRACE_ROOT",
            "local_data/runtime/harness_traces",
        )
    ).expanduser()
    if not workspace_root.is_absolute():
        workspace_root = Path.cwd() / workspace_root
    if not trace_root.is_absolute():
        trace_root = Path.cwd() / trace_root
    codex_home_raw = os.environ.get("ARCHE_CODEX_HOME", "").strip()
    codex_home = Path(codex_home_raw).expanduser() if codex_home_raw else None
    if codex_home is not None and not codex_home.is_absolute():
        codex_home = Path.cwd() / codex_home
    return workspace_root.resolve(), trace_root.resolve(), (
        codex_home.resolve() if codex_home is not None else None
    )


def _validate_config_codex_binary(
    kind: str,
    *,
    codex_bin: str,
    allow_test_codex_binary: bool,
) -> None:
    """校验生产 wrapper 固定性，或显式允许的绝对可执行测试替身。"""

    if kind != "codex":
        return
    codex_path = Path(codex_bin)
    if allow_test_codex_binary and codex_path != DEFAULT_CODEX_BIN:
        if not codex_path.is_absolute() or not codex_path.is_file() or not os.access(
            codex_path, os.X_OK
        ):
            raise ValueError("Codex 测试替身必须是绝对路径且可执行：" + str(codex_path))
        return
    _validate_repository_codex_wrapper(codex_path)


@dataclass(frozen=True)
class HarnessConfig:
    # ``kind`` 是唯一可选的 Harness 驱动，只作用于 Planner。
    kind: str = "react"
    workspace_root: Path = Path("local_data/runtime/harness_workspaces")
    trace_root: Path = Path("local_data/runtime/harness_traces")
    timeout_seconds: float = 900.0
    # 只有 Planner 使用 Codex 且 stage 命中此集合时，Dispatcher 才会把
    # HarnessRequest.timeout_seconds 设为 None。默认集合为空，避免误把旧阶段
    # 变成无限等待；Investigation 阶段落地后可通过环境变量显式启用。
    unbounded_turn_stages: frozenset[str] = frozenset()
    codex_bin: str = str(DEFAULT_CODEX_BIN)
    codex_home: Path | None = None
    model: str | None = None
    base_url: str | None = None
    reasoning_effort: str | None = None

    @classmethod
    def from_env(cls, *, allow_test_codex_binary: bool = False) -> "HarnessConfig":
        """读取环境配置；非生产 Codex 替身只能由调用方显式开启测试 seam。"""
        kind = os.environ.get("ARCHE_PLANNER_DRIVER", "react").strip().lower() or "react"
        if kind not in SUPPORTED_HARNESSES:
            raise ValueError(
                "ARCHE_PLANNER_DRIVER must be one of: "
                + ", ".join(sorted(SUPPORTED_HARNESSES))
            )
        reasoning_effort = os.environ.get("ARCHE_CHEM_REASONING_EFFORT", "").strip().lower()
        if reasoning_effort and reasoning_effort not in SUPPORTED_REASONING_EFFORTS:
            raise ValueError(
                "ARCHE_CHEM_REASONING_EFFORT must be one of: "
                + ", ".join(sorted(SUPPORTED_REASONING_EFFORTS))
            )
        codex_base_url, codex_model, _ = _read_codex_provider_settings(kind)
        configured_model = (
            codex_model
            if kind == "codex"
            else os.environ.get("ARCHE_CHEM_MODEL_NAME", "").strip()
        )
        configured_base_url = (
            codex_base_url
            if kind == "codex"
            else os.environ.get("ARCHE_CHEM_BASE_URL", "").strip()
        )
        workspace_root, trace_root, codex_home = _resolve_harness_output_paths()
        codex_bin = _resolve_executable_path(
            os.environ.get("ARCHE_CODEX_BIN", str(DEFAULT_CODEX_BIN))
        )
        _validate_config_codex_binary(
            kind,
            codex_bin=codex_bin,
            allow_test_codex_binary=allow_test_codex_binary,
        )
        return cls(
            kind=kind,
            workspace_root=workspace_root.resolve(),
            trace_root=trace_root.resolve(),
            timeout_seconds=_positive_float(
                "ARCHE_PLANNER_HARNESS_TIMEOUT_SECONDS", 900.0
            ),
            unbounded_turn_stages=_stage_names_from_env(
                CODEX_UNBOUNDED_TURN_STAGES_ENV
            ),
            codex_bin=codex_bin,
            codex_home=codex_home.resolve() if codex_home is not None else None,
            model=configured_model or None,
            base_url=configured_base_url or None,
            reasoning_effort=reasoning_effort or None,
        )

    def workspace_for(self, *, role: str, research_task_id: str) -> Path:
        """返回按 Harness、research_task 和角色隔离的工作区。"""

        normalized_role = str(role).strip().lower()
        safe_role = safe_component(normalized_role, "role")
        safe_task = safe_component(research_task_id or "research-task", "research_task_id")
        driver = self.kind if normalized_role in CODEX_HARNESS_ROLES else "react"
        path = self.workspace_root / driver / safe_task / safe_role
        path.mkdir(parents=True, exist_ok=True)
        return path

    def trace_for(self, *, role: str, research_task_id: str) -> Path:
        """返回按 Harness、research_task 和角色隔离的 trace 目录。"""

        normalized_role = str(role).strip().lower()
        safe_role = safe_component(normalized_role, "role")
        safe_task = safe_component(research_task_id or "research-task", "research_task_id")
        driver = self.kind if normalized_role in CODEX_HARNESS_ROLES else "react"
        path = self.trace_root / driver / safe_task / safe_role
        path.mkdir(parents=True, exist_ok=True)
        return path

    def role_enabled(self, role: str) -> bool:
        """返回角色是否启用了非 React 驱动器。"""

        normalized = str(role).strip().lower()
        return normalized in CODEX_HARNESS_ROLES and self.kind != "react"

    def driver_for(self, role: str) -> str:
        """返回角色对应的驱动器；未列出的角色一律使用 React。"""

        normalized = str(role).strip().lower()
        if normalized in CODEX_HARNESS_ROLES:
            return self.kind
        return "react"

    def uses_unbounded_turn_timeout(self, *, role: str, stage: str) -> bool:
        """判断一个请求是否属于显式配置的 Investigation turn。

        只有 Codex Planner 且阶段名命中 ``unbounded_turn_stages`` 时才返回 True。
        该判断只影响 SDK turn 的总体 wall-clock deadline，不会改变任何传输、MCP
        工具或计算作业的局部时限。阶段集合默认为空，避免旧 workflow 阶段被误放宽。
        """

        return (
            self.kind == "codex"
            and str(role).strip().lower() == "planner"
            and str(stage).strip().lower() in {
                str(item).strip().lower() for item in self.unbounded_turn_stages
            }
        )


def safe_component(value: str, label: str) -> str:
    value = str(value).strip()
    if not value or value in {".", ".."} or any(char in value for char in "/\\\x00"):
        raise ValueError(f"{label} contains an unsafe path component")
    return value[:120]


# 保留早期 Harness 原型使用的私有名称，兼容仍导入该辅助函数的下游代码。
_safe_component = safe_component
