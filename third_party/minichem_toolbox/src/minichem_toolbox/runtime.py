"""Backend health probes and exact runtime dispatch without scientific fallback."""

from __future__ import annotations

import json
import os
import signal
import shutil
import subprocess
from functools import lru_cache
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import yaml

from .backend_specs import BACKEND_HEALTH_POLICIES
from .models import BackendSpec
from .environment_layout import resolve_configured_path, resolve_runtime_path
from .paths import CONFIG_ROOT, PROJECT_ROOT, SOURCE_ROOT
from .process_identity import ProcessIdentity, capture_process_identity
from .scope import current_scope
from .worker_protocol import WORKER_RESPONSE_PREFIX


PROFILE_CONFIG_PATH = CONFIG_ROOT / "mcp_profiles.yaml"
WORKER_TERMINATION_GRACE_SECONDS = 10

GAUSSIAN_CODEGEN_MODEL_ENVIRONMENT = "ARCHE_GAUSSIAN_CODEGEN_MODEL_NAME"
GAUSSIAN_CODEGEN_REQUIRED_MODEL_NAME = "deepseek-v4-pro"

GAUSSIAN_CODEGEN_CONFIGURATION_ENVIRONMENT = (
    "ARCHE_CHEM_API_KEY",
    "ARCHE_CHEM_BASE_URL",
    "ARCHE_CHEM_BACKEND",
    GAUSSIAN_CODEGEN_MODEL_ENVIRONMENT,
)
GAUSSIAN_CODEGEN_SUPPORTED_HTTP_BACKENDS = frozenset(
    {
        "openai_compatible",
        "openai",
        "arche_chem",
    }
)


def inspect_gaussian_codegen_configuration(
    environment: Mapping[str, str],
) -> dict[str, Any]:
    """检查 Gaussian codegen 所需的专家配置，不发请求也不暴露配置值。

    输入是待检查的环境变量映射。函数检查三项 ARCHE-Chem 连接变量是否非空，
    并校验 Backend 名称、HTTP/HTTPS Base URL 以及独立 Gaussian codegen 模型
    是否符合合同。独立模型变量省略时使用固定的 ``deepseek-v4-pro`` 默认值；
    显式填写其他模型会被拒绝，因此通用 ``ARCHE_CHEM_MODEL_NAME`` 的变更不会
    静默改变这四个阶段使用的模型。检查只在内存中完成，不读取网络、不写文件、
    不打印或记录 API key、地址和模型名称。

    返回值包含 ``available``、``reason_code``、``missing_environment``、
    ``invalid_environment`` 和可能使用默认值的 ``defaulted_environment``，只列出
    安全的状态和变量名。配置缺失或格式非法不抛异常，而是返回
    ``available=False``；不兼容的输入映射仍会按 Python 的类型错误失败。
    """

    missing = [
        name
        for name in GAUSSIAN_CODEGEN_CONFIGURATION_ENVIRONMENT
        if name != GAUSSIAN_CODEGEN_MODEL_ENVIRONMENT
        and not str(environment.get(name, "")).strip()
    ]
    configured_model = str(
        environment.get(GAUSSIAN_CODEGEN_MODEL_ENVIRONMENT, "")
    ).strip()
    defaulted_environment = []
    if not configured_model:
        configured_model = GAUSSIAN_CODEGEN_REQUIRED_MODEL_NAME
        defaulted_environment.append(GAUSSIAN_CODEGEN_MODEL_ENVIRONMENT)
    if missing:
        return {
            "available": False,
            "reason_code": "environment_missing",
            "missing_environment": missing,
            "invalid_environment": [],
            "defaulted_environment": defaulted_environment,
        }

    backend = str(environment["ARCHE_CHEM_BACKEND"]).strip().casefold()
    if backend not in GAUSSIAN_CODEGEN_SUPPORTED_HTTP_BACKENDS:
        return {
            "available": False,
            "reason_code": "environment_invalid",
            "missing_environment": [],
            "invalid_environment": ["ARCHE_CHEM_BACKEND"],
            "defaulted_environment": defaulted_environment,
        }

    base_url = str(environment["ARCHE_CHEM_BASE_URL"]).strip()
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return {
            "available": False,
            "reason_code": "environment_invalid",
            "missing_environment": [],
            "invalid_environment": ["ARCHE_CHEM_BASE_URL"],
            "defaulted_environment": defaulted_environment,
        }
    if configured_model != GAUSSIAN_CODEGEN_REQUIRED_MODEL_NAME:
        return {
            "available": False,
            "reason_code": "environment_invalid",
            "missing_environment": [],
            "invalid_environment": [GAUSSIAN_CODEGEN_MODEL_ENVIRONMENT],
            "defaulted_environment": defaulted_environment,
        }
    return {
        "available": True,
        "reason_code": "ok",
        "missing_environment": [],
        "invalid_environment": [],
        "defaulted_environment": defaulted_environment,
    }


def _worker_descendant_process_groups(worker_pid: int) -> tuple[int, ...]:
    """Return Linux process groups below a worker without crossing its own group."""

    if os.name != "posix" or not Path("/proc").is_dir():
        return ()
    pending = [worker_pid]
    seen = {worker_pid}
    groups: set[int] = set()
    while pending:
        parent = pending.pop()
        children_path = Path(f"/proc/{parent}/task/{parent}/children")
        try:
            children = [int(item) for item in children_path.read_text().split()]
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
        for child in children:
            if child in seen:
                continue
            seen.add(child)
            pending.append(child)
            try:
                group = os.getpgid(child)
            except ProcessLookupError:
                continue
            if group != worker_pid:
                groups.add(group)
    return tuple(sorted(groups))


def _signal_worker_tree(
    worker_pid: int,
    received_signal: signal.Signals,
    descendant_groups: Iterable[int] = (),
) -> tuple[int, ...]:
    """仅通过稳定 pidfd 终止 worker；无法取得句柄时不发送 PID/PGID 信号。"""

    groups = set(descendant_groups)
    groups.update(_worker_descendant_process_groups(worker_pid))
    pidfd_open = getattr(os, "pidfd_open", None)
    pidfd_send_signal = getattr(signal, "pidfd_send_signal", None)
    if not callable(pidfd_open) or not callable(pidfd_send_signal):
        return tuple(sorted(groups))
    identity = capture_process_identity(worker_pid)
    if identity.start_time_ticks is None or not identity.boot_id:
        return tuple(sorted(groups))
    try:
        descriptor = pidfd_open(worker_pid, 0)
    except (OSError, ValueError):
        return tuple(sorted(groups))
    try:
        current = capture_process_identity(worker_pid)
        if (
            current.start_time_ticks != identity.start_time_ticks
            or current.boot_id != identity.boot_id
            or current.process_group_id != identity.process_group_id
            or current.session_id != identity.session_id
        ):
            return tuple(sorted(groups))
        pidfd_send_signal(descriptor, int(received_signal), None, 0)
    except (OSError, ValueError, ProcessLookupError, PermissionError):
        pass
    finally:
        os.close(descriptor)
    # pidfd 只能稳定指向 worker leader；descendant PGID 没有同等句柄时不能冒险
    # 批量发信号。调用方应继续按未知状态处理这些组。
    return tuple(sorted(groups))


def _terminate_worker_process(process: subprocess.Popen[str]) -> tuple[str, str]:
    """请求停止 worker；无稳定句柄时保留未知状态，不宣称已清理 detached 进程。"""

    if os.name == "posix":
        descendant_groups = _signal_worker_tree(process.pid, signal.SIGTERM)
    else:
        descendant_groups = ()
        process.terminate()
    try:
        stdout, stderr = process.communicate(
            timeout=WORKER_TERMINATION_GRACE_SECONDS
        )
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            _signal_worker_tree(process.pid, signal.SIGKILL, descendant_groups)
        else:
            process.kill()
        try:
            stdout, stderr = process.communicate(
                timeout=WORKER_TERMINATION_GRACE_SECONDS
            )
        except subprocess.TimeoutExpired as exc:
            # 无 pidfd 时不能继续冒险发信号或无限等待；调用方应把未结束
            # worker 视为 unknown，并保留其资源占用供恢复处理。
            stdout = exc.output or ""
            stderr = exc.stderr or ""
    return stdout or "", stderr or ""


@lru_cache(maxsize=1)
def load_runtime_config() -> dict[str, Any]:
    value = yaml.safe_load(PROFILE_CONFIG_PATH.read_text(encoding="utf-8")) or {}
    if not isinstance(value.get("profiles"), dict):
        raise ValueError(f"Invalid runtime profile configuration: {PROFILE_CONFIG_PATH}")
    return value


def runtime_spec(name: str) -> dict[str, Any]:
    config = load_runtime_config()
    values = config["profiles"]
    if name in values:
        result = dict(values[name])
        result["name"] = name
        return result
    raise KeyError(f"Unknown backend runtime {name!r}")


def runtime_path(name: str) -> Path:
    specification = runtime_spec(name)
    return resolve_runtime_path(name, str(specification["environment"]))


def runtime_python(name: str) -> Path:
    configured = str(runtime_spec(name).get("python") or "").strip()
    if configured:
        return resolve_configured_path(configured)
    return runtime_path(name) / "bin" / "python"


def runtime_names() -> tuple[str, ...]:
    """Return every configured runtime id without treating runtimes as tool filters."""

    return tuple(sorted(map(str, load_runtime_config()["profiles"])))


def _runtime_entries(specification: dict[str, Any], key: str) -> list[str]:
    entries = []
    for value in specification.get(key) or []:
        entries.append(str(resolve_configured_path(str(value))))
    return entries


def _runtime_environment_value(value: Any) -> str:
    """Resolve configured path values while leaving scalar environment values intact."""

    text = str(value)
    if text.startswith(".") or "/" in text:
        preserve_trailing_slash = text.endswith("/")
        resolved = str(resolve_configured_path(text))
        return resolved + "/" if preserve_trailing_slash else resolved
    return text


def runtime_environment(name: str) -> dict[str, str]:
    specification = runtime_spec(name)
    environment = runtime_path(name)
    path_entries = [
        *_runtime_entries(specification, "prepend_path_entries"),
        str(environment / "bin"),
        *_runtime_entries(specification, "path_entries"),
    ]
    library_entries = [
        *_runtime_entries(specification, "prepend_library_path_entries"),
        str(environment / "lib"),
        *_runtime_entries(specification, "library_path_entries"),
    ]
    inherited_pythonpath = []
    for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if not entry:
            continue
        path = Path(entry).expanduser()
        inherited_pythonpath.append(
            str(path if path.is_absolute() else (Path.cwd() / path).resolve())
        )
    values = {
        "PATH": os.pathsep.join([*path_entries, os.environ.get("PATH", "")]),
        "LD_LIBRARY_PATH": os.pathsep.join(
            [*library_entries, os.environ.get("LD_LIBRARY_PATH", "")]
        ),
        "PYTHONPATH": os.pathsep.join(
            [str(SOURCE_ROOT), *inherited_pythonpath]
        ),
        "MINICHEM_BACKEND_RUNTIME": name,
    }
    values.update(
        {
            str(variable): _runtime_environment_value(value)
            for variable, value in dict(
                specification.get("environment_variables") or {}
            ).items()
        }
    )
    model_cache = load_runtime_config().get("model_cache_root", ".mini_model_cache")
    if specification.get("use_project_model_cache", False):
        cache = Path(str(model_cache)).expanduser()
        if not cache.is_absolute():
            cache = PROJECT_ROOT / cache
        cache.mkdir(parents=True, exist_ok=True)
        values["MINICHEM_MODEL_CACHE"] = str(cache.resolve())
        values["XDG_CACHE_HOME"] = str(cache.resolve())
    for variable, executable in dict(specification.get("command_variables") or {}).items():
        configured = Path(str(executable)).expanduser()
        if configured.is_absolute():
            candidate = configured
        elif configured.parent != Path("."):
            candidate = resolve_configured_path(configured)
        else:
            candidate = environment / "bin" / configured
        if candidate.exists():
            values[str(variable)] = str(candidate.resolve())
    scoped = current_scope()
    if scoped is not None:
        # worker/supervisor 只接收当前请求的显式环境快照，不依赖父进程 cwd。
        values.update(scoped.environment())
    return values


def _module_probe(runtime: str, modules: Iterable[str]) -> dict[str, dict[str, Any]]:
    names = sorted(set(modules))
    if not names:
        return {}
    python = runtime_python(runtime)
    if not python.is_file():
        return {name: {"available": False, "version": None} for name in names}
    script = """
import importlib.metadata
from importlib.machinery import PathFinder
import json
import sys

out = {}

def find_module_spec_without_import(name):
    # 只按模块搜索路径探测，不执行包的 __init__ 或科学后端导入。

    parent_path = None
    qualified = []
    spec = None
    for part in name.split('.'):
        qualified.append(part)
        spec = PathFinder.find_spec('.'.join(qualified), parent_path)
        if spec is None:
            return None
        parent_path = spec.submodule_search_locations
    return spec

for name in json.loads(sys.argv[1]):
    try:
        available = find_module_spec_without_import(name) is not None
    except Exception:
        available = False
    version = None
    if available:
        for distribution in (name, name.split('.')[0]):
            try:
                version = importlib.metadata.version(distribution)
                break
            except Exception:
                pass
    out[name] = {'available': available, 'version': version}
print(json.dumps(out))
"""
    try:
        completed = subprocess.run(
            [str(python), "-c", script, json.dumps(names)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
            env={**os.environ, **runtime_environment(runtime)},
        )
        if completed.returncode == 0:
            return json.loads(completed.stdout.strip().splitlines()[-1])
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError, IndexError):
        pass
    return {name: {"available": False, "version": None} for name in names}


def _configured_package_resources(specification: Mapping[str, Any]) -> list[dict[str, Any]]:
    """读取运行时声明的包内资源检查，而不把本机路径写入健康结果。"""

    health_checks = specification.get("health_checks") or {}
    raw_resources = health_checks.get("package_resources") or health_checks.get(
        "resources"
    ) or []
    if isinstance(raw_resources, Mapping):
        raw_resources = [
            {"resource_id": resource_id, **dict(resource_value or {})}
            for resource_id, resource_value in raw_resources.items()
        ]
    resources: list[dict[str, Any]] = []
    for raw in raw_resources:
        if not isinstance(raw, Mapping):
            continue
        resource_id = str(raw.get("resource_id") or "").strip()
        module = str(raw.get("module") or "").strip()
        relative_path = str(raw.get("relative_path") or "").strip()
        if not resource_id or not module or not relative_path:
            continue
        required_files = raw.get("required_files") or []
        if isinstance(required_files, str):
            required_files = [required_files]
        resources.append(
            {
                "resource_id": resource_id,
                "module": module,
                "relative_path": relative_path,
                "path_environment_variable": str(
                    raw.get("path_environment_variable") or ""
                ).strip(),
                "required_files": [str(item) for item in required_files],
            }
        )
    return resources


def _package_resource_probe(
    runtime: str, resources: Iterable[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    """在声明的运行时检查随包或显式配置的模型资源是否完整。"""

    specifications = [dict(item) for item in resources]
    if not specifications:
        return {}
    resource_ids = [str(item["resource_id"]) for item in specifications]
    python = runtime_python(runtime)
    if not python.is_file():
        return {
            resource_id: {
                "available": False,
                "resource_id": resource_id,
                "reason_code": "runtime_python_missing",
            }
            for resource_id in resource_ids
        }
    script = r'''
import importlib.resources
import json
import os
from pathlib import Path
import sys


def _join_path(base, relative):
    current = base
    for part in str(relative).replace("\\", "/").split("/"):
        if not part or part == ".":
            continue
        if part == "..":
            return None
        current = current.joinpath(part)
    return current


def _has_content(candidate):
    try:
        if candidate is None or not candidate.is_file():
            return False
        with candidate.open("rb") as stream:
            return bool(stream.read(1))
    except (OSError, AttributeError):
        return False


out = {}
for specification in json.loads(sys.argv[1]):
    resource_id = str(specification["resource_id"])
    configured_variable = str(specification.get("path_environment_variable") or "")
    configured_path = os.environ.get(configured_variable, "").strip() if configured_variable else ""
    source = "configured" if configured_path else "package"
    try:
        if configured_path:
            target = _join_path(Path(configured_path).expanduser(), "")
        else:
            package_root = importlib.resources.files(str(specification["module"]))
            target = _join_path(package_root, specification["relative_path"])
    except Exception:
        out[resource_id] = {
            "available": False,
            "resource_id": resource_id,
            "reason_code": "resource_probe_failed",
        }
        continue
    if target is None or not target.is_dir():
        out[resource_id] = {
            "available": False,
            "resource_id": resource_id,
            "reason_code": "resource_missing",
            "source": source,
        }
        continue
    missing = []
    for filename in specification.get("required_files") or []:
        candidate = _join_path(target, filename)
        present = _has_content(candidate)
        if not present:
            missing.append(str(filename))
    if missing:
        out[resource_id] = {
            "available": False,
            "resource_id": resource_id,
            "reason_code": "resource_incomplete",
            "source": source,
            "missing_files": missing,
        }
        continue
    out[resource_id] = {
        "available": True,
        "resource_id": resource_id,
        "reason_code": "ok",
        "source": source,
    }
print(json.dumps(out))
'''
    try:
        completed = subprocess.run(
            [str(python), "-c", script, json.dumps(specifications)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
            cwd=PROJECT_ROOT,
            env={**os.environ, **runtime_environment(runtime)},
        )
        if completed.returncode == 0:
            value = json.loads(completed.stdout.strip().splitlines()[-1])
            if isinstance(value, dict):
                return {
                    resource_id: dict(value.get(resource_id) or {
                        "available": False,
                        "resource_id": resource_id,
                        "reason_code": "resource_probe_failed",
                    })
                    for resource_id in resource_ids
                }
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError, IndexError):
        pass
    return {
        resource_id: {
            "available": False,
            "resource_id": resource_id,
            "reason_code": "resource_probe_failed",
        }
        for resource_id in resource_ids
    }


def _resolve_executable(runtime: str, executable: str) -> str | None:
    specification = runtime_spec(runtime)
    bin_directory = runtime_path(runtime) / "bin"
    candidates = [bin_directory / executable]
    if executable == "cp2k":
        candidates.extend(bin_directory / name for name in ("cp2k.psmp", "cp2k.popt", "cp2k.ssmp"))
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())
    external_commands = [
        *(specification.get("external_commands") or []),
        *((specification.get("health_checks") or {}).get("external_commands") or []),
    ]
    for value in external_commands:
        candidate = Path(str(value)).expanduser()
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        if candidate.name == executable and candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())
    configured = shutil.which(executable, path=runtime_environment(runtime).get("PATH"))
    if configured:
        return str(Path(configured).resolve())
    return None


def resolve_executable(runtime: str, executable: str) -> str | None:
    """Resolve one exact configured command name in one declared runtime."""

    if "/" in executable or "\\" in executable or not executable.strip():
        raise ValueError("executable must be a configured command name, not a path")
    return _resolve_executable(runtime, executable)


def _inspect_backend_configuration(
    specification: BackendSpec,
    environment: Mapping[str, str],
) -> dict[str, Any] | None:
    """按 Backend 声明检查配置，不执行外部操作或输出配置值。

    输入是 Backend 描述和已合并运行时环境变量。函数根据 Backend 声明选择
    专用检查；未声明策略时返回 ``None``，Gaussian codegen 则返回配置检查的
    状态字典。检查只调用无网络格式校验，不发送模型请求、不写文件，也不把
    配置值放入返回结果。

    返回值为 ``None`` 或包含可用状态、原因码和缺失/非法变量名的字典。声明
    了未知健康策略时抛出 ``ValueError``，用于阻止未实现的策略被静默放行。
    """

    policy = BACKEND_HEALTH_POLICIES.get(specification.id)
    if policy is None:
        return None
    if policy == "all_required_environment" and specification.id == "gaussian_codegen":
        return inspect_gaussian_codegen_configuration(environment)
    raise ValueError(
        f"Unsupported health policy {policy!r} for backend {specification.id!r}"
    )


def _collect_backend_probe_inputs(
    specifications: list[BackendSpec],
) -> tuple[dict[str, set[str]], dict[str, list[dict[str, Any]]]]:
    """收集每个 runtime 需要探测的模块和资源，保持声明顺序及资源去重。"""

    modules_by_runtime: dict[str, set[str]] = {}
    resources_by_runtime: dict[str, list[dict[str, Any]]] = {}
    for specification in specifications:
        modules_by_runtime.setdefault(specification.runtime, set()).update(
            specification.python_modules
        )
        profile = runtime_spec(specification.runtime)
        resources = _configured_package_resources(profile)
        if resources:
            current = resources_by_runtime.setdefault(specification.runtime, [])
            known = {str(item["resource_id"]) for item in current}
            current.extend(
                item for item in resources if str(item["resource_id"]) not in known
            )
    return modules_by_runtime, resources_by_runtime


def _probe_backend_runtime_dependencies(
    modules_by_runtime: Mapping[str, set[str]],
    resources_by_runtime: Mapping[str, list[dict[str, Any]]],
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, dict[str, dict[str, Any]]]]:
    """批量执行模块和包资源探测，失败状态由各探测器统一生成。"""

    module_status = {
        runtime: _module_probe(runtime, modules)
        for runtime, modules in modules_by_runtime.items()
    }
    resource_status = {
        runtime: _package_resource_probe(runtime, resources)
        for runtime, resources in resources_by_runtime.items()
    }
    return module_status, resource_status


def _resolve_backend_probe_availability(
    specification: BackendSpec,
    python: Path,
    modules: Mapping[str, Mapping[str, Any]],
    executables: Mapping[str, str | None],
    environment: Mapping[str, bool],
    environment_values: Mapping[str, str],
    configuration_health: Mapping[str, Any] | None,
    resources: Mapping[str, Mapping[str, Any]],
) -> tuple[bool, list[str], list[str], list[str], str]:
    """根据已探测的依赖计算可用性、缺失字段和稳定原因码。"""

    missing_modules, missing_executables, missing_environment, missing_resources = (
        _collect_backend_missing_dependencies(
            modules,
            executables,
            environment,
            resources,
        )
    )
    implementation_available = _backend_implementation_is_available(
        specification,
        modules,
        executables,
        environment,
        environment_values,
        configuration_health,
    )
    resources_ok = all(item.get("available") for item in resources.values())
    available = python.is_file() and implementation_available and resources_ok
    reason_code = _resolve_backend_probe_reason(
        python,
        configuration_health,
        missing_modules,
        missing_executables,
        missing_environment,
        missing_resources,
        resources,
        available,
    )
    return (
        available,
        missing_modules,
        missing_executables,
        missing_environment,
        reason_code,
    )


def _collect_backend_missing_dependencies(
    modules: Mapping[str, Mapping[str, Any]],
    executables: Mapping[str, str | None],
    environment: Mapping[str, bool],
    resources: Mapping[str, Mapping[str, Any]],
) -> tuple[list[str], list[str], list[str], list[str]]:
    """列出模块、可执行文件、环境变量和资源的缺失项。"""

    missing_modules = [name for name, item in modules.items() if not item["available"]]
    missing_executables = [name for name, path in executables.items() if not path]
    missing_environment = [name for name, present in environment.items() if not present]
    missing_resources = [
        resource_id
        for resource_id, item in resources.items()
        if not item.get("available")
    ]
    return missing_modules, missing_executables, missing_environment, missing_resources


def _backend_implementation_is_available(
    specification: BackendSpec,
    modules: Mapping[str, Mapping[str, Any]],
    executables: Mapping[str, str | None],
    environment: Mapping[str, bool],
    environment_values: Mapping[str, str],
    configuration_health: Mapping[str, Any] | None,
) -> bool:
    """判断 Backend 的实现、凭据和专用配置是否都满足。"""

    modules_ok = bool(modules) and all(item["available"] for item in modules.values())
    executables_ok = bool(executables) and any(executables.values())
    configured_command_ok = any(
        bool(environment_values.get(name))
        for name in specification.environment_variables
        if not name.endswith("_KEY")
    )
    implementation_available = (
        modules_ok or executables_ok or configured_command_ok
        if (modules or executables or specification.environment_variables)
        else True
    )
    credential_names = [
        name for name in specification.environment_variables if name.endswith("_KEY")
    ]
    if credential_names:
        implementation_available = implementation_available and all(
            environment[name] for name in credential_names
        )
    if configuration_health is not None:
        implementation_available = (
            implementation_available and configuration_health["available"]
        )
    return implementation_available


def _resolve_backend_probe_reason(
    python: Path,
    configuration_health: Mapping[str, Any] | None,
    missing_modules: list[str],
    missing_executables: list[str],
    missing_environment: list[str],
    missing_resources: list[str],
    resources: Mapping[str, Mapping[str, Any]],
    available: bool,
) -> str:
    """按既有优先级选取稳定的 Backend 健康原因码。"""

    if available:
        return "ok"
    if not python.is_file():
        return "runtime_python_missing"
    if configuration_health is not None and not configuration_health["available"]:
        return str(configuration_health["reason_code"])
    if missing_modules:
        return "python_module_missing"
    if missing_executables:
        return "executable_missing"
    if missing_environment:
        return "environment_missing"
    if missing_resources:
        return str(
            resources[missing_resources[0]].get("reason_code") or "resource_missing"
        )
    return "implementation_missing"


def _assemble_backend_probe_result(
    specification: BackendSpec,
    module_status: Mapping[str, Mapping[str, Mapping[str, Any]]],
    resource_status: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    """组装单个 Backend 的公开健康快照，不暴露本机路径或认证值。"""

    python = runtime_python(specification.runtime)
    modules = {
        name: module_status.get(specification.runtime, {}).get(
            name, {"available": False, "version": None}
        )
        for name in specification.python_modules
    }
    executables = {
        name: _resolve_executable(specification.runtime, name)
        for name in specification.executables
    }
    runtime_env = runtime_environment(specification.runtime)
    environment_values = {
        name: str(os.environ.get(name) or runtime_env.get(name) or "")
        for name in specification.environment_variables
    }
    strict_environment = (
        BACKEND_HEALTH_POLICIES.get(specification.id)
        == "all_required_environment"
    )
    environment = {
        name: bool(value.strip()) if strict_environment else bool(value)
        for name, value in environment_values.items()
    }
    configuration_health = _inspect_backend_configuration(
        specification, environment_values
    )
    if configuration_health is not None:
        for name in configuration_health.get("defaulted_environment", ()):
            environment[name] = True
    configured_resources = _configured_package_resources(runtime_spec(specification.runtime))
    resources = {
        str(item["resource_id"]): dict(
            resource_status.get(specification.runtime, {}).get(
                str(item["resource_id"]),
                {
                    "available": False,
                    "resource_id": str(item["resource_id"]),
                    "reason_code": "resource_probe_failed",
                },
            )
        )
        for item in configured_resources
    }
    (
        available,
        missing_modules,
        missing_executables,
        missing_environment,
        reason_code,
    ) = _resolve_backend_probe_availability(
        specification,
        python,
        modules,
        executables,
        environment,
        environment_values,
        configuration_health,
        resources,
    )
    missing_resources = [
        resource_id
        for resource_id, item in resources.items()
        if not item.get("available")
    ]
    return {
        "status": "available" if available else "unavailable",
        "available": available,
        "reason_code": reason_code,
        "runtime": specification.runtime,
        "runtime_python": str(python),
        "runtime_python_exists": python.is_file(),
        "python_modules": modules,
        "executables": executables,
        "environment": environment,
        "missing_python_modules": missing_modules,
        "missing_executables": missing_executables,
        "missing_environment": missing_environment,
        "invalid_environment": list(
            (configuration_health or {}).get("invalid_environment") or []
        ),
        "resources": resources,
        "resource_id": missing_resources[0] if len(missing_resources) == 1 else None,
        "missing_resources": missing_resources,
        "conda_packages": list(specification.conda_packages),
        "pip_packages": list(specification.pip_packages),
        "required_data_resources": list(specification.required_data_resources),
        "license_class": specification.license_class,
        "install_notes": specification.install_notes,
    }


def probe_all_backends(specs: Iterable[BackendSpec]) -> dict[str, dict[str, Any]]:
    """探测声明的 Backend，并返回稳定的可用性快照。"""

    specifications = list(specs)
    modules_by_runtime, resources_by_runtime = _collect_backend_probe_inputs(
        specifications
    )
    module_status, resource_status = _probe_backend_runtime_dependencies(
        modules_by_runtime,
        resources_by_runtime,
    )
    return {
        specification.id: _assemble_backend_probe_result(
            specification,
            module_status,
            resource_status,
        )
        for specification in specifications
    }


def _apply_worker_thread_limits(
    environment: dict[str, str], requested_cores: Any,
) -> None:
    """把请求的 CPU 数量映射到各科学库线程环境变量。"""

    if requested_cores is None:
        return
    threads = str(max(1, int(requested_cores)))
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        environment[variable] = threads


def _apply_worker_gpu_visibility(
    environment: dict[str, str],
    requested_gpus: int,
    allocated_gpu_ids: list[str],
) -> None:
    """设置 worker 可见 GPU；零 GPU 请求清空两种驱动变量。"""

    if requested_gpus == 0:
        environment["CUDA_VISIBLE_DEVICES"] = ""
        environment["ROCR_VISIBLE_DEVICES"] = ""
        return
    for variable in ("CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        visible = [
            item.strip()
            for item in environment.get(variable, "").split(",")
            if item.strip()
        ]
        selected = allocated_gpu_ids or (
            visible[:requested_gpus]
            if visible
            else list(map(str, range(requested_gpus)))
        )
        environment[variable] = ",".join(selected)


def _apply_worker_cpu_allocation(
    environment: dict[str, str],
    allocated_cpu_ids: list[int],
    requested_cores: Any,
) -> None:
    """设置 worker 的 CPU 亲和性或回退的 CPU 数量限制。"""

    # preexec_fn is not safe when submit_action_batch starts workers from
    # multiple threads.  The lightweight launcher applies these limits inside
    # the new interpreter before importing any scientific backend modules.
    if allocated_cpu_ids:
        allocated_cpu_list = ",".join(str(item) for item in allocated_cpu_ids)
        environment["MINICHEM_WORKER_CPU_IDS"] = allocated_cpu_list
        # MPI-backed Actions may launch a runtime that deliberately rebinds
        # ranks instead of inheriting the worker's Linux affinity. Restrict
        # both Open MPI 4 and PRRTE/Open MPI 5 to this evaluator allocation.
        environment["OMPI_MCA_hwloc_base_cpu_list"] = allocated_cpu_list
        environment["PRTE_MCA_hwloc_default_cpu_list"] = allocated_cpu_list
    elif requested_cores is not None:
        environment["MINICHEM_WORKER_CPU_COUNT"] = str(
            max(1, int(requested_cores))
        )


def _prepare_worker_environment(
    runtime: str,
    payload: Mapping[str, Any],
    resource_allocation: Mapping[str, Any] | None,
) -> dict[str, str]:
    """合并 runtime 环境和请求资源设置，返回 worker 的启动快照。"""

    environment = {**os.environ, **runtime_environment(runtime)}
    requested_resources = (
        (payload.get("request") or {}).get("resource_limits") or {}
    )
    requested_cores = requested_resources.get("cpu_cores")
    _apply_worker_thread_limits(environment, requested_cores)
    allocation = dict(resource_allocation or {})
    allocated_cpu_ids = [int(item) for item in allocation.get("cpu_ids") or []]
    allocated_gpu_ids = [str(item) for item in allocation.get("gpu_ids") or []]
    requested_gpus = int(requested_resources.get("gpu_count") or 0)
    _apply_worker_gpu_visibility(environment, requested_gpus, allocated_gpu_ids)
    _apply_worker_cpu_allocation(environment, allocated_cpu_ids, requested_cores)
    return environment


def _record_worker_identity_or_failure(
    process: subprocess.Popen[str],
    on_worker_started: Callable[[ProcessIdentity], None] | None,
) -> dict[str, Any] | None:
    """记录刚启动 worker 的身份；失败时终止 worker 并返回结构化错误。"""

    if on_worker_started is None:
        return None
    try:
        on_worker_started(capture_process_identity(process.pid))
    except Exception as exc:
        stdout, stderr, termination_confirmed = _terminate_worker_process_safely(process)
        failure = {
            "status": "failed",
            "error": {
                "code": "worker_identity_record_failed",
                "message": str(exc),
                "stderr": (stderr or "")[-2000:],
            },
            "retryable": False,
        }
        if not termination_confirmed:
            failure["resource_state"] = "unknown"
        return failure
    return None


def _worker_process_termination_confirmed(
    process: subprocess.Popen[str],
) -> bool:
    """只在操作系统确认 worker 已退出时返回 True。"""

    try:
        return process.poll() is not None
    except (AttributeError, OSError):
        return False


def _terminate_worker_process_safely(
    process: subprocess.Popen[str],
) -> tuple[str, str, bool]:
    """尽力终止 worker，并显式返回是否确认已退出。"""

    try:
        stdout, stderr = _terminate_worker_process(process)
    except OSError as exc:
        stdout, stderr = "", str(exc)
    return (
        stdout or "",
        stderr or "",
        _worker_process_termination_confirmed(process),
    )


def _invalid_worker_response(
    stdout: str,
    stderr: str,
    process: subprocess.Popen[str],
    message: str = "Backend worker did not return a valid framed JSON object",
) -> dict[str, Any]:
    """构造不泄露额外协议内容的 worker 响应失败。"""

    return {
        "status": "failed",
        "error": {
            "code": "invalid_worker_response",
            "message": message,
            "stdout": stdout[-2000:],
            "stderr": stderr[-2000:],
            "returncode": process.returncode,
        },
        "retryable": False,
    }


def _decode_worker_response_line(line: str) -> dict[str, Any]:
    """解码并核对一行 worker 响应必须具备的 object/status 形状。"""

    result = json.loads(line)
    if (
        not isinstance(result, dict)
        or not isinstance(result.get("status"), str)
        or not result["status"]
    ):
        raise ValueError("worker response must be an object with a non-empty status")
    return result


def _parse_worker_exchange_result(
    stdout: str,
    stderr: str,
    process: subprocess.Popen[str],
) -> dict[str, Any]:
    """解析带 marker 的 worker JSON，并兼容无 footer 的旧单行协议。"""

    try:
        lines = stdout.splitlines()
        framed_lines = [
            line[len(WORKER_RESPONSE_PREFIX) :].strip()
            for line in lines
            if line.startswith(WORKER_RESPONSE_PREFIX)
        ]
        if len(framed_lines) > 1:
            raise ValueError("worker response contains multiple framed result lines")
        if framed_lines:
            result = _decode_worker_response_line(framed_lines[0])
        else:
            legacy_lines = [line.strip() for line in lines if line.strip()]
            if len(legacy_lines) != 1:
                raise ValueError("worker response has no unique legacy result line")
            result = _decode_worker_response_line(legacy_lines[0])
    except (json.JSONDecodeError, ValueError):
        return _invalid_worker_response(stdout, stderr, process)
    if stderr.strip():
        result.setdefault("worker_stderr", stderr[-4000:])
    result.setdefault("worker_returncode", process.returncode)
    return result


def _exchange_worker_payload(
    *,
    python: Path,
    environment: Mapping[str, str],
    payload: Mapping[str, Any],
    timeout_seconds: int,
    on_worker_started: Callable[[ProcessIdentity], None] | None,
) -> dict[str, Any]:
    """启动 worker、交换一次请求并把超时或启动失败转换成协议结果。"""

    process = None
    try:
        process = subprocess.Popen(
            [str(python), "-m", "minichem_toolbox.worker_launcher"],
            stdin=subprocess.PIPE,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=PROJECT_ROOT,
            env=environment,
            start_new_session=os.name == "posix",
        )
        identity_failure = _record_worker_identity_or_failure(
            process,
            on_worker_started,
        )
        if identity_failure is not None:
            return identity_failure
        stdout, stderr = process.communicate(
            json.dumps(payload, ensure_ascii=False), timeout=timeout_seconds
        )
    except subprocess.TimeoutExpired:
        if process is None:
            raise
        stdout, stderr = _terminate_worker_process(process)
        return {
            "status": "timeout",
            "resource_state": "unknown",
            "error": {
                "code": "worker_timeout",
                "message": (
                    f"Backend runtime exceeded {timeout_seconds} seconds; "
                    "process-tree termination cannot be atomically proven, so "
                    "resource occupancy remains held for recovery"
                ),
                "stderr": (stderr or "")[-2000:],
            },
            "retryable": True,
        }
    except OSError as exc:
        if process is None:
            return {
                "status": "failed",
                "error": {"code": "worker_start_failed", "message": str(exc)},
                "retryable": False,
            }
        _stdout, termination_stderr, termination_confirmed = (
            _terminate_worker_process_safely(process)
        )
        if termination_confirmed:
            return {
                "status": "failed",
                "error": {
                    "code": "worker_exchange_failed",
                    "message": "Backend worker exchange failed after startup",
                    "stderr": (termination_stderr or "")[-2000:],
                    "returncode": process.returncode,
                },
                "retryable": False,
            }
        return {
            "status": "failed",
            "resource_state": "unknown",
            "error": {
                "code": "worker_exchange_state_unknown",
                "message": (
                    "Backend worker exchange failed after startup and worker termination "
                    "could not be confirmed"
                ),
                "stderr": (termination_stderr or "")[-2000:],
            },
            "retryable": True,
        }
    except BaseException:
        if process is not None:
            _terminate_worker_process(process)
        raise
    return _parse_worker_exchange_result(stdout, stderr, process)


def invoke_worker(
    *,
    runtime: str,
    payload: dict[str, Any],
    timeout_seconds: int,
    resource_allocation: dict[str, Any] | None = None,
    on_worker_started: Callable[[ProcessIdentity], None] | None = None,
) -> dict[str, Any]:
    """在声明 runtime 中执行一个后端，并在需要时持久交接 worker 身份。

    ``on_worker_started`` 在 ``Popen`` 成功后立即调用，供 Gaussian typed 调用方把
    实际 worker 的 PID、启动时钟和 boot id 写入队列及 reservation。回调失败时会尽力
    终止新 worker 并返回结构化失败；服务在回调前崩溃留下的记录没有 worker 身份，恢复
    逻辑会按未知状态保守保留资源。超时且无法原子证明进程树已结束时，返回
    ``resource_state=unknown``，调用方必须继续保留 reservation。
    """

    python = runtime_python(runtime)
    if not python.is_file():
        return {
            "status": "unavailable",
            "error": {
                "code": "runtime_missing",
                "message": f"Runtime Python does not exist: {python}",
            },
            "retryable": False,
        }
    environment = _prepare_worker_environment(
        runtime,
        payload,
        resource_allocation,
    )
    return _exchange_worker_payload(
        python=python,
        environment=environment,
        payload=payload,
        timeout_seconds=timeout_seconds,
        on_worker_started=on_worker_started,
    )
