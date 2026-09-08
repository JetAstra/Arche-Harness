"""Focused, immutable-per-run Action/backend catalog for MiniChem."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import threading
from copy import deepcopy
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

from .models import ActionSpec, BackendSpec
from .runtime import probe_all_backends
from .resources import resources_for_backends
from .specs import ACTION_SPECS, BACKEND_SPECS
from .parameter_specs import RESOURCE_LIMIT_PARAMETER_SPECS, common_fixed_parameter_specs
from .resource_budget import gaussian_resource_policy, resource_budget_record
from .timeout_policy import timeout_policy_record
from .scope import current_scope


CATEGORY_LABELS = {
    "scientific_data_interchange": "Scientific records, schemas, and output parsing",
    "structure_and_system": "Structure, conformers, charges, and system construction",
    "cheminformatics": "Molecular descriptors, fingerprints, identifiers, and graph operations",
    "molecular_electronic": "Molecular electronic structure and derived properties",
    "reaction_and_kinetics": "Reaction paths, equilibrium, and kinetics",
    "molecular_dynamics": "Molecular dynamics propagation and trajectory analysis",
    "periodic_and_phonons": "Periodic electronic structure and lattice dynamics",
    "docking": "Molecular docking",
    "data_sources": "External chemistry data sources",
}


TOOL_DISCOVERY_MODE_ENV = "MINICHEM_TOOL_DISCOVERY_MODE"
CATALOG_FILENAME = "_toolbox_catalog.json"
CATALOG_HASH_ENV = "MINICHEM_MCP_CATALOG_HASH"
CATALOG_WORKSPACE_ENV = "MINICHEM_MCP_CATALOG_WORKSPACE"
ToolDiscoveryMode = Literal["progressive", "full"]


class CatalogFreezeError(RuntimeError):
    """目录快照无法安全探测、写入或校验时抛出的启动错误。"""


_ACTIVE_CATALOG_LOCK = threading.RLock()
_CATALOG_FREEZE_LOCK = threading.Lock()
_ACTIVE_CATALOG_SNAPSHOT: dict[str, Any] | None = None
_ACTIVE_CATALOG_WORKSPACE: Path | None = None
_FAILED_CATALOG_WORKSPACES: dict[Path, dict[str, str]] = {}


def _agent_resource_contract(budget: dict[str, Any]) -> str:
    """返回不包含宿主容量数字的公开资源说明。"""

    del budget
    return (
        "Caller-selected resource limits are checked against the service policy; requests "
        "outside that policy are rejected without automatic reduction."
    )


def resolve_tool_discovery_mode(value: str | None = None) -> ToolDiscoveryMode:
    """Resolve the public MCP surface without changing catalog membership.

    ``progressive`` exposes neutral catalog-discovery tools plus one explicit
    Action dispatcher. ``full`` retains the historical one-MCP-tool-per-Action
    surface for regression comparisons. Both modes make the same immutable
    task-independent catalog available and neither performs task retrieval.
    """

    raw = value
    if raw is None:
        raw = os.environ.get(TOOL_DISCOVERY_MODE_ENV, "progressive")
    normalized = str(raw).strip().lower().replace("-", "_")
    aliases = {
        "progressive": "progressive",
        "progressive_discovery": "progressive",
        "full": "full",
        "atomic_all": "full",
    }
    try:
        return aliases[normalized]  # type: ignore[return-value]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported tool discovery mode {raw!r}; choose progressive or full"
        ) from exc


def _resource_coverage(resource: dict[str, Any]) -> str:
    """返回不含资源数量的科学资源类型说明。"""

    kind = resource.get("kind")
    if kind == "element_file_collection":
        return "element-indexed files"
    if kind == "variant_file_collection":
        return "variant-indexed files"
    if kind == "slater_koster_parameter_set":
        return "element-pair parameters"
    if kind == "model_checkpoint":
        return "model checkpoint"
    if kind == "single_file_resource":
        return "registered file"
    return "runtime-managed executable"


def action_specs() -> dict[str, ActionSpec]:
    return {spec.id: spec for spec in ACTION_SPECS}


def backend_specs() -> dict[str, BackendSpec]:
    return {spec.id: spec for spec in BACKEND_SPECS}


def _mini_resource_snapshot() -> list[dict[str, Any]]:
    return resources_for_backends(spec.id for spec in BACKEND_SPECS)


def validate_catalog() -> None:
    actions = action_specs()
    backends = backend_specs()
    if len(actions) != len(ACTION_SPECS):
        raise ValueError("Duplicate action ids in catalog")
    if len(backends) != len(BACKEND_SPECS):
        raise ValueError("Duplicate backend ids in catalog")
    for specification in ACTION_SPECS:
        specification.validate()
        unknown = sorted(set(specification.backend_ids) - set(backends))
        if unknown:
            raise ValueError(f"Action {specification.id} references unknown backends: {unknown}")
        for backend_id in specification.backend_ids:
            if specification.id not in backends[backend_id].capabilities:
                raise ValueError(
                    f"Action {specification.id} is missing from backend {backend_id} capabilities"
                )
    for specification in BACKEND_SPECS:
        specification.validate()
        unknown = sorted(set(specification.capabilities) - set(actions))
        if unknown:
            raise ValueError(f"Backend {specification.id} references unknown actions: {unknown}")
        for action_id in specification.capabilities:
            if specification.id not in actions[action_id].backend_ids:
                raise ValueError(
                    f"Backend {specification.id} is missing from action {action_id} choices"
                )
        for action_id, roles in specification.component_backend_options.items():
            for role, options in roles.items():
                unknown_components = sorted(set(options) - set(backends))
                if unknown_components:
                    raise ValueError(
                        f"Backend {specification.id}/{action_id} role {role} references "
                        f"unknown component backends: {unknown_components}"
                    )
    if not ACTION_SPECS:
        raise ValueError("MiniChem catalog requires at least one Scientific Action")


def catalog_snapshot(
    *,
    include_health: bool = True,
    discovery_mode: str | None = None,
    resource_budget: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validate_catalog()
    # 目录冻结前先解析 Gaussian 服务策略；配置不明确时禁止安装 active 目录。
    gaussian_resource_policy()
    mode = resolve_tool_discovery_mode(discovery_mode)
    health = probe_all_backends(BACKEND_SPECS) if include_health else {}
    payload: dict[str, Any] = {
        "schema_version": 7,
        "execution_layers": [
            {
                "id": "predefined_actions",
                "role": "validated common scientific operations",
                "mandatory": False,
            },
            {
                "id": "native_software",
                "role": "Agent-authored native inputs and exact allowlisted commands",
                "mandatory": False,
            },
            {
                "id": "programmable_analysis",
                "role": "Agent-authored programs in explicitly selected chemistry runtimes",
                "mandatory": False,
            },
        ],
        "discovery_mode": mode,
        "exposure_policy": (
            "progressive_discovery" if mode == "progressive" else "atomic_all"
        ),
        "catalog_visibility_policy": "focused_task_independent",
        "task_specific_tool_filtering": False,
        "backend_selection_policy": "per_action_explicit",
        "provider_selection_policies": sorted(
            {spec.selection_policy for spec in ACTION_SPECS}
        ),
        "scientific_resource_selection_policy": "agent_explicit_no_default",
        "automatic_fallback": False,
        "evaluation_resource_budget": resource_budget or resource_budget_record(),
        "actions": [spec.as_dict() for spec in ACTION_SPECS],
        "backends": [
            {**spec.as_dict(), "health": health.get(spec.id)}
            for spec in BACKEND_SPECS
        ],
        "resources": _mini_resource_snapshot(),
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    payload["catalog_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def _canonical_catalog_bytes(snapshot: dict[str, Any]) -> bytes:
    """按目录协议编码快照；非 JSON 值会让启动直接失败。"""

    try:
        return json.dumps(
            snapshot,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CatalogFreezeError("MiniChem 目录快照不是可规范编码的 JSON 对象") from exc


def _catalog_workspace_path(workspace: str | os.PathLike[str] | Path | None = None) -> Path:
    """解析目录快照所在 workspace，并拒绝不存在或非目录路径。"""

    scoped = current_scope()
    service_marker = os.environ.get(CATALOG_WORKSPACE_ENV, "").strip()
    configured = (
        os.environ.get("MINICHEM_MCP_WORKSPACE", "").strip()
        or os.environ.get("RESEARCHCHEMBENCH_WORKSPACE", "").strip()
        or service_marker
    )
    if workspace is not None:
        candidate = Path(workspace)
    elif scoped is not None and service_marker:
        # catalog 是服务启动时冻结的一份共享目录，不随每条 capability 请求切到
        # 任务工作区；marker 由启动服务发布，并不要求每个 scope 复制一份目录。
        candidate = Path(service_marker)
    elif scoped is not None and _ACTIVE_CATALOG_WORKSPACE is not None:
        candidate = _ACTIVE_CATALOG_WORKSPACE
    else:
        candidate = Path(
            scoped.workspace if scoped is not None else configured or Path.cwd()
        )
    try:
        root = candidate.expanduser().resolve(strict=False)
    except OSError as exc:
        raise CatalogFreezeError("无法解析 MiniChem workspace") from exc
    try:
        root_stat = root.stat()
    except OSError as exc:
        raise CatalogFreezeError("MiniChem workspace 不可访问") from exc
    if not stat.S_ISDIR(root_stat.st_mode):
        raise CatalogFreezeError("MiniChem workspace 必须是目录")
    return root


def _catalog_target_path(workspace: Path) -> Path:
    """检查固定目录文件目标，避免跟随符号链接或写入特殊文件。"""

    target = workspace / CATALOG_FILENAME
    try:
        target_stat = target.lstat()
    except FileNotFoundError:
        return target
    except OSError as exc:
        raise CatalogFreezeError("无法检查 MiniChem 目录快照文件") from exc
    if stat.S_ISLNK(target_stat.st_mode):
        raise CatalogFreezeError("MiniChem 目录快照目标不能是符号链接")
    if not stat.S_ISREG(target_stat.st_mode):
        raise CatalogFreezeError("MiniChem 目录快照目标必须是普通文件")
    return target


def _remove_stale_catalog_file(target: Path) -> None:
    """探测失败时删除可确认的旧普通文件，防止后续误用陈旧目录。"""

    try:
        target_stat = target.lstat()
    except FileNotFoundError:
        return
    except OSError:
        return
    if stat.S_ISREG(target_stat.st_mode):
        try:
            target.unlink()
        except OSError:
            # 启动原始错误优先返回；不触碰不安全的特殊文件或符号链接。
            pass


def _write_catalog_atomically(target: Path, payload: bytes) -> None:
    """在同一目录中以 0600 临时文件写入并原子替换目录快照。"""

    temporary_path: Path | None = None
    file_descriptor: int | None = None
    try:
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{CATALOG_FILENAME}.",
            suffix=".tmp",
            dir=target.parent,
        )
        temporary_path = Path(temporary_name)
        os.fchmod(file_descriptor, 0o600)
        with os.fdopen(file_descriptor, "wb") as stream:
            file_descriptor = None
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())

        # 临时文件准备好后再次检查目标，避免把并发创建的链接或特殊文件覆盖掉。
        _catalog_target_path(target.parent)
        os.replace(temporary_path, target)
        temporary_path = None

        no_follow = getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(target, os.O_RDONLY | no_follow)
        try:
            target_stat = os.fstat(descriptor)
            if not stat.S_ISREG(target_stat.st_mode):
                raise CatalogFreezeError("原子替换后的目录快照不是普通文件")
            os.fchmod(descriptor, 0o600)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

        # 同步目录项，保证崩溃恢复不会暴露半个替换结果。
        try:
            parent_descriptor = os.open(target.parent, os.O_RDONLY | no_follow)
        except OSError:
            parent_descriptor = None
        if parent_descriptor is not None:
            try:
                os.fsync(parent_descriptor)
            finally:
                os.close(parent_descriptor)
    except CatalogFreezeError:
        raise
    except (OSError, ValueError) as exc:
        raise CatalogFreezeError("无法原子写入 MiniChem 目录快照") from exc
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass


def _install_active_catalog_snapshot(
    snapshot: dict[str, Any], workspace: Path, *, publish_marker: bool = False
) -> None:
    """安装本服务生命周期内唯一的目录快照，并隔离调用方可变对象。"""

    global _ACTIVE_CATALOG_SNAPSHOT, _ACTIVE_CATALOG_WORKSPACE
    with _ACTIVE_CATALOG_LOCK:
        _ACTIVE_CATALOG_SNAPSHOT = deepcopy(snapshot)
        _ACTIVE_CATALOG_WORKSPACE = workspace
    if publish_marker:
        _publish_catalog_marker(snapshot, workspace)


def _publish_catalog_marker(snapshot: dict[str, Any], workspace: Path) -> None:
    """发布供 worker 继承的服务生命周期标记。"""

    os.environ[CATALOG_HASH_ENV] = str(snapshot["catalog_hash"])
    os.environ[CATALOG_WORKSPACE_ENV] = str(workspace)


def _clear_catalog_marker() -> None:
    """清理进程级 worker 标记，避免普通库调用污染后续 workspace。"""

    os.environ.pop(CATALOG_HASH_ENV, None)
    os.environ.pop(CATALOG_WORKSPACE_ENV, None)


def _clear_active_catalog_snapshot(workspace: Path) -> None:
    """清除指定 workspace 的旧进程缓存，避免启动失败后继续暴露旧目录。"""

    global _ACTIVE_CATALOG_SNAPSHOT, _ACTIVE_CATALOG_WORKSPACE
    with _ACTIVE_CATALOG_LOCK:
        if _ACTIVE_CATALOG_WORKSPACE == workspace:
            _ACTIVE_CATALOG_SNAPSHOT = None
            _ACTIVE_CATALOG_WORKSPACE = None
    if os.environ.get(CATALOG_WORKSPACE_ENV) == str(workspace):
        _clear_catalog_marker()


def _mark_catalog_freeze_failure(workspace: Path, error: BaseException) -> None:
    """记录 workspace 的启动冻结失败，阻止后续调用退回旧目录或重新探测。"""

    code = (
        "catalog_freeze_error"
        if isinstance(error, CatalogFreezeError)
        else "catalog_freeze_failed"
    )
    failure = {
        "code": code,
        "exception_type": type(error).__name__,
        "message": "MiniChem 目录冻结失败",
    }
    with _ACTIVE_CATALOG_LOCK:
        _FAILED_CATALOG_WORKSPACES[workspace] = failure


def _clear_catalog_freeze_failure(workspace: Path) -> None:
    """仅在新冻结完整成功后清除该 workspace 的失败状态。"""

    with _ACTIVE_CATALOG_LOCK:
        _FAILED_CATALOG_WORKSPACES.pop(workspace, None)


def _catalog_freeze_failure(workspace: Path) -> dict[str, str] | None:
    """读取 workspace 的启动冻结失败原因。"""

    with _ACTIVE_CATALOG_LOCK:
        return _FAILED_CATALOG_WORKSPACES.get(workspace)


def _catalog_hash_matches(snapshot: dict[str, Any]) -> bool:
    """校验快照 hash，损坏文件不能被当作冻结目录。"""

    supplied = snapshot.get("catalog_hash")
    if not isinstance(supplied, str) or not supplied:
        return False
    payload = deepcopy(snapshot)
    payload.pop("catalog_hash", None)
    try:
        expected = hashlib.sha256(_canonical_catalog_bytes(payload)).hexdigest()
    except CatalogFreezeError:
        return False
    return supplied == expected


def _read_catalog_file(path: Path) -> dict[str, Any] | None:
    """读取并校验冻结文件；链接、特殊文件和 hash 不一致都视为无效。"""

    try:
        target_stat = path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise CatalogFreezeError("无法读取 MiniChem 目录快照文件") from exc
    if stat.S_ISLNK(target_stat.st_mode):
        raise CatalogFreezeError("MiniChem 目录快照文件不能是符号链接")
    if not stat.S_ISREG(target_stat.st_mode):
        raise CatalogFreezeError("MiniChem 目录快照文件必须是普通文件")
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        opened_stat = os.fstat(descriptor)
        if not stat.S_ISREG(opened_stat.st_mode):
            raise CatalogFreezeError("MiniChem 目录快照文件必须是普通文件")
        with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
            descriptor = None
            value = json.load(stream)
    except CatalogFreezeError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CatalogFreezeError("MiniChem 目录快照文件缺失或损坏") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if not isinstance(value, dict) or not _catalog_hash_matches(value):
        raise CatalogFreezeError("MiniChem 目录快照文件的 catalog_hash 不一致")
    return value


def _freeze_catalog_snapshot(
    *,
    workspace: str | os.PathLike[str] | Path | None = None,
    discovery_mode: str | None = None,
    resource_budget: dict[str, Any] | None = None,
    publish_marker: bool = False,
) -> dict[str, Any]:
    """启动时探测并冻结一份目录，成功前绝不信任 workspace 中的旧文件。

    输入是可选 workspace、公开发现模式和资源预算。函数完整探测一次后，将
    规范 JSON 原子替换到 ``_toolbox_catalog.json``，并把同一内容安装到进程
    缓存供 Action、MCP resource 和 worker 调用。探测、目标安全检查或写入失败
    会保留原始异常（或抛出 ``CatalogFreezeError``）；失败状态会阻止后续调用把
    陈旧目录误认作本次服务的有效结果，并移除可确认的旧普通文件。
    """

    root = _catalog_workspace_path(workspace)
    _clear_active_catalog_snapshot(root)
    # 旧服务若已结束，新的启动必须先撤销旧 worker 标记；成功后再按参数发布新标记。
    _clear_catalog_marker()
    target = root / CATALOG_FILENAME
    try:
        _catalog_target_path(root)
        mode = resolve_tool_discovery_mode(discovery_mode)
        snapshot = catalog_snapshot(
            include_health=True,
            discovery_mode=mode,
            resource_budget=resource_budget,
        )
        if not isinstance(snapshot, dict) or not _catalog_hash_matches(snapshot):
            raise CatalogFreezeError(
                "MiniChem 目录探测没有返回有效且一致的 catalog_hash"
            )
        payload = _canonical_catalog_bytes(snapshot)
        # JSON 编码会把 ActionSpec 中的 tuple 统一为 list；缓存也使用该文件原文
        # 的对象形状，确保内存消费者和 MCP resource 看到完全一致的内容。
        normalized_snapshot = json.loads(payload.decode("utf-8"))
        if not isinstance(normalized_snapshot, dict):
            raise CatalogFreezeError("MiniChem 目录快照规范化后不是对象")
        _write_catalog_atomically(target, payload)
        _install_active_catalog_snapshot(
            normalized_snapshot,
            root,
            publish_marker=publish_marker,
        )
    except Exception as exc:
        # 失败状态优先于兼容读取；普通文件可以清理，链接和特殊文件由
        # _remove_stale_catalog_file 的 lstat 检查保护，不会被触碰。
        _mark_catalog_freeze_failure(root, exc)
        _remove_stale_catalog_file(target)
        raise
    _clear_catalog_freeze_failure(root)
    return deepcopy(normalized_snapshot)


def freeze_catalog_snapshot(
    *,
    workspace: str | os.PathLike[str] | Path | None = None,
    discovery_mode: str | None = None,
    resource_budget: dict[str, Any] | None = None,
    publish_marker: bool = False,
) -> dict[str, Any]:
    """串行执行一次服务启动目录探测，避免并发写入产生不同快照。"""

    with _CATALOG_FREEZE_LOCK:
        return _freeze_catalog_snapshot(
            workspace=workspace,
            discovery_mode=discovery_mode,
            resource_budget=resource_budget,
            publish_marker=publish_marker,
        )


def catalog_hash(*, include_health: bool = False) -> str:
    return str(catalog_snapshot(include_health=include_health)["catalog_hash"])


def active_catalog_snapshot() -> dict[str, Any]:
    """返回本服务冻结目录；未启动冻结时兼容读取旧文件或重新探测。"""

    current_workspace = _catalog_workspace_path()
    path = current_workspace / CATALOG_FILENAME
    failure = _catalog_freeze_failure(current_workspace)
    if failure is not None:
        raise CatalogFreezeError(
            "workspace 的 MiniChem 目录冻结已失败，拒绝继续读取："
            f"{failure['message']}（code={failure['code']}，"
            f"exception_type={failure['exception_type']}）"
        )
    with _ACTIVE_CATALOG_LOCK:
        if (
            _ACTIVE_CATALOG_SNAPSHOT is not None
            and _ACTIVE_CATALOG_WORKSPACE == current_workspace
        ):
            frozen = _read_catalog_file(path)
            if frozen is None or frozen != _ACTIVE_CATALOG_SNAPSHOT:
                raise CatalogFreezeError("服务冻结目录文件已被删除或替换")
            return deepcopy(_ACTIVE_CATALOG_SNAPSHOT)
    marker_workspace = os.environ.get(CATALOG_WORKSPACE_ENV, "")
    marker_hash = os.environ.get(CATALOG_HASH_ENV, "")
    if marker_workspace or marker_hash:
        if marker_workspace != str(current_workspace) or not marker_hash:
            raise CatalogFreezeError("服务冻结目录标记与当前 workspace 不一致")
        value = _read_catalog_file(path)
        if value is None or value.get("catalog_hash") != marker_hash:
            raise CatalogFreezeError("服务冻结目录文件缺失或 hash 已变化")
        return value
    try:
        value = _read_catalog_file(path)
    except CatalogFreezeError:
        # 未进入服务启动路径时保留旧调用的兼容行为：损坏文件触发一次新探测。
        value = None
    if value is not None:
        return value
    return catalog_snapshot(include_health=True)


_PUBLIC_RESOURCE_BUDGET = {
    "source": "service_policy",
    "agent_controllable": False,
    "scope": "per_task",
}


_PUBLIC_ACTION_TEXT = {
    ("sdf_to_gjf", "description"): (
        "Convert one supplied SDF record into a Gaussian input deck using explicit route and "
        "title settings. The operation writes the deck and does not run Gaussian."
    ),
    ("sdf_to_gjf", "input_description"): (
        "sdf_path is a workspace-relative SDF path or SDF ArtifactRef; route_parameters is an "
        "explicit Gaussian route mapping."
    ),
    ("xyz_to_gjf", "description"): (
        "Convert one supplied XYZ structure into a Gaussian input deck using explicit route, "
        "charge, multiplicity, and title settings. The operation writes the deck and does not "
        "run Gaussian."
    ),
    ("xyz_to_gjf", "input_description"): (
        "input_xyz_path is a workspace-relative XYZ path or XYZ/atomic-structure ArtifactRef; "
        "route_section, charge, and multiplicity are explicit request values."
    ),
    ("generate_gaussian_code", "description"): (
        "Optionally run the non-default human-authored four-stage route-composition micro-workflow "
        "for the natural-language "
        "requirement in inputs.user_input; its four model calls are fixed to DeepSeek Pro and the "
        "small workflow temporarily narrows Gaussian keyword exploration. The wrapper returns one "
        "route line after route-text and safe-shape checks; it does not perform scientific validation, "
        "generate a Gaussian input deck, submit or run Gaussian, or produce scientific evidence."
    ),
    ("generate_gaussian_code", "input_description"): (
        "inputs.user_input is the only input: provide a natural-language Gaussian calculation "
        "requirement. The Action uses its fixed DeepSeek Pro four-stage route-composition helper "
        "and returns one route line after route-text and safe-shape checks; it does not accept "
        "explicit method or settings, generate a Gaussian input deck, submit or run Gaussian, or "
        "produce scientific evidence."
    ),
    ("rank_conformers_from_results", "description"): (
        "Rank and weight conformers only from aligned energies or free energies supplied "
        "by the caller."
    ),
    ("assign_protonation_states", "description"): (
        "Assign explicit protonation states using the selected backend and pH/rule settings."
    ),
    ("enumerate_tautomers", "description"): (
        "Enumerate bounded tautomeric forms without selecting a preferred tautomer."
    ),
    ("calculate_energy", "description"): (
        "Calculate one molecular or non-periodic scalar energy with the exact software and "
        "method selected by the caller."
    ),
    ("search_transition_state_path", "description"): (
        "Generate transition-state path candidates from caller-provided, atom-balanced reactant "
        "and product SMILES. The Action prepares mapped three-dimensional endpoint pairs, "
        "searches candidate paths, and may return transition_state_guess_artifact as an "
        "unvalidated XYZFile; callers do not provide endpoint geometries or a TS guess. The "
        "bounded profile requires at least two reactant components and exactly one balanced "
        "hydrogen transfer from one donor to one acceptor. That condition limits only the number "
        "of hydrogen transfers: the same reaction may also form or break mapped heavy-atom bonds "
        "or undergo mapped bond-order changes. It runs forward only with service-fixed total "
        "charge +1 and spin multiplicity 1 and requires a GPU. An initial guess cannot be used as "
        "scientific evidence for transition-state identity, reaction connectivity, or mechanism. "
        "Before any guess is used to support a "
        "transition-state or reaction-channel claim, independent downstream Gaussian TS "
        "optimization must converge; Hessian/frequency analysis must confirm exactly one "
        "imaginary frequency whose normal mode matches the intended reaction coordinate, with all "
        "remaining frequencies real; and forward and reverse (bidirectional) IRC calculations "
        "must converge to "
        "and be checked against the expected reactant and product sides. The call returns one "
        "synchronous ActionResult "
        "and does not submit an asynchronous job. A success or partial_success result may retain "
        "completed endpoint pairs. invalid_request, unavailable, failed, "
        "timeout, and cancelled results use the public error and retryable fields. A client-side "
        "wait timeout or cancellation is a transport failure, not an ActionResult. No result "
        "triggers automatic backend fallback."
    ),
    ("search_transition_state_path", "input_description"): (
        "Provide atom-balanced reactant and product SMILES arrays for one elementary reaction; "
        "do not provide three-dimensional endpoints or a TS guess. Optional force-field choices "
        "control endpoint preparation and path search. The service uses a forward-only profile "
        "with fixed total charge and spin multiplicity; direction, charge, multiplicity, device, "
        "and resource policy are service-managed and are not request fields. A GPU is required "
        "and selected by the service."
    ),
}

_PUBLIC_BACKEND_TEXT = {
    "gaussian": (
        "Gaussian 16 C.01 SCF/DFT calculations, vertical excited states, NBO/NAO Wiberg "
        "indices, relaxed one-coordinate scans, transition-state searches, and one-direction "
        "IRC jobs rendered from typed molecular structures and explicit bounded settings."
    ),
    "transition_state_path_search": (
        "Service-managed transition-state path generation from caller-provided, atom-balanced "
        "reactant and product SMILES. It prepares mapped three-dimensional endpoints and an "
        "unvalidated TS guess without requiring either geometry from the caller. The bounded "
        "profile applies only to reactions with at least two reactant components and exactly one "
        "balanced hydrogen transfer from one donor to one acceptor. This condition does not "
        "exclude simultaneous mapped heavy-atom bond formation, bond cleavage, or bond-order "
        "changes; "
        "it runs forward only with service-fixed total charge +1 and spin multiplicity 1 and "
        "requires a GPU. The scientific output is an unvalidated transition-state guess/path "
        "candidate that cannot be used as scientific evidence for transition-state identity, "
        "reaction connectivity, or mechanism. Before any guess "
        "is used to support a transition-state or reaction-channel claim, independent downstream "
        "transition-state optimization must converge; Hessian/frequency analysis must confirm "
        "exactly one imaginary frequency whose normal mode matches the intended reaction "
        "coordinate, with all remaining frequencies real; and forward and reverse "
        "(bidirectional) IRC calculations must converge to and be checked against the expected "
        "reactant and product sides. The call is synchronous and does not "
        "choose another backend."
    ),
}

_PUBLIC_PARAMETER_TEXT = {
    ("sdf_to_gjf", "inputs.route_parameters"): (
        "Explicit Gaussian route parameters for the generated input deck.",
        "Changing this value changes the generated calculation settings.",
    ),
    ("xyz_to_gjf", "inputs.route_section"): (
        "Explicit Gaussian route section for the generated input deck.",
        "Changing this value changes the generated calculation settings.",
    ),
    ("build_mapped_distance_constraints", "action_settings.equilibrium_distance_scale"): (
        "Multiplier applied to the reference single-bond length.",
        "Changing it moves the center of the allowed mapped-pair distance interval.",
    ),
}


def _public_action_text(action_id: str, field_name: str, value: Any) -> Any:
    """按 Action 与字段的显式映射返回公开文案，不执行通用文本清洗。"""

    return _PUBLIC_ACTION_TEXT.get((action_id, field_name), value)


def _public_backend_text(backend_id: str, value: Any) -> Any:
    """按 Backend id 返回公开能力说明，不执行通用文本清洗。"""

    return _PUBLIC_BACKEND_TEXT.get(
        backend_id,
        "Registered backend for the scientific capabilities exposed by this entry.",
    )


def _public_parameter_text(
    action_id: str,
    field_path: str,
    description: Any,
    impact: Any = None,
) -> tuple[Any, Any]:
    """按 Action 字段显式提供科学参数文案，不清洗内部文本。"""

    override = _PUBLIC_PARAMETER_TEXT.get((action_id, field_path))
    if override is not None:
        return override
    field_name = field_path.rsplit(".", 1)[-1]
    if field_name == "allow_model_download":
        return (
            "Service resource availability option.",
            "Changes resource preparation without changing the scientific method.",
        )
    if field_name in {"model", "model_branch"}:
        return (
            "Scientific model or registered resource identifier required by the selected backend.",
            "Changing it can change the calculated scientific result.",
        )
    return description, impact


def _public_catalog_health_projection(value: Any) -> dict[str, Any]:
    """按健康结论白名单返回公开字段，不遍历或清洗内部探针对象。"""

    if not isinstance(value, dict):
        return {}
    projected: dict[str, Any] = {}
    for field_name in (
        "status",
        "available",
        "reason_code",
    ):
        item = value.get(field_name)
        if item is not None:
            projected[field_name] = item
    resources = value.get("resources")
    if isinstance(resources, dict):
        public_resources: dict[str, dict[str, Any]] = {}
        for resource_id, resource in resources.items():
            if not isinstance(resource, dict):
                continue
            item = {
                field_name: resource[field_name]
                for field_name in ("available", "reason_code", "resource_id", "status", "version")
                if field_name in resource
            }
            if item:
                public_resources[str(resource_id)] = item
        if public_resources:
            projected["resources"] = public_resources
    return projected


def _public_resource_projection(resource: Any) -> dict[str, Any]:
    """按科学资源选择所需字段公开资源，不公开覆盖数量或宿主配置。"""

    if not isinstance(resource, dict):
        return {}
    fields = (
        "id",
        "display_name",
        "kind",
        "version",
        "format",
        "available",
        "selectable",
        "compatible_backends",
        "selection_syntax",
    )
    return {
        field_name: resource.get(field_name)
        for field_name in fields
        if field_name in resource
    }


def _public_action_projection(action: Any) -> dict[str, Any]:
    """按 Action 科学契约字段公开目录项。"""

    if not isinstance(action, dict):
        return {}
    fields = (
        "id",
        "version",
        "category",
        "description",
        "input_description",
        "primary_output",
        "backend_ids",
        "required_inputs",
        "optional_inputs",
        "selection_policy",
        "input_semantic_types",
        "output_semantic_types",
        "execution_class",
        "requires_network",
        "capability_tags",
        "data_action",
        "batch_safe",
    )
    projected: dict[str, Any] = {}
    for field_name in fields:
        if field_name not in action:
            continue
        value = action[field_name]
        if field_name in {"description", "input_description"}:
            value = _public_action_text(str(action.get("id")), field_name, value)
        elif isinstance(value, (tuple, list)):
            value = list(value)
        projected[field_name] = value
    return projected


def _public_backend_description(backend: Any) -> str | None:
    """返回 Backend 的科学能力说明，剔除内部项目称谓。"""

    if not isinstance(backend, dict):
        return None
    return _public_backend_text(str(backend.get("id")), backend.get("description"))


def public_catalog_snapshot(snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
    """返回 MCP 资源可见的脱敏目录，不改变服务端冻结目录。

    公开目录只包含 Action/Backend 的结构化能力、选择约束和安全健康结论；后端运行时、模块、
    安装提示、凭据及健康中的本机路径只保留在内部快照。

    ``catalog_hash`` 始终是服务端 canonical 快照的身份摘要，不能对脱敏
    projection 重新计算，否则 discovery 响应和执行 provenance 就无法对账。
    ``catalog_projection_hash`` 单独覆盖本资源的完整公开内容，并绑定上述
    canonical 身份，供客户端验证传输内容没有被截断或篡改。
    """

    current = deepcopy(snapshot) if snapshot is not None else active_catalog_snapshot()
    public_actions = [
        _public_action_projection(action)
        for action in current.get("actions", [])
        if isinstance(action, dict)
    ]
    public_backends: list[dict[str, Any]] = []
    for backend in current.get("backends", []):
        if not isinstance(backend, dict):
            continue
        public_backends.append(
            {
                "id": backend.get("id"),
                "display_name": backend.get("display_name"),
                "runtime": backend.get("runtime"),
                "description": _public_backend_description(backend),
                "capabilities": list(backend.get("capabilities") or []),
                "health": _public_catalog_health_projection(backend.get("health")),
            }
        )
    public_layers = [
        {
            "id": "predefined_actions",
            "role": "validated common scientific operations",
            "mandatory": False,
        },
        {
            "id": "native_software",
            "role": "Agent-authored native inputs and exact allowlisted commands",
            "mandatory": False,
        },
        {
            "id": "programmable_analysis",
            "role": "Agent-authored programs in explicitly selected chemistry runtimes",
            "mandatory": False,
        },
    ]
    projection = {
        "schema_version": current.get("schema_version"),
        "catalog_hash": current.get("catalog_hash"),
        "catalog_projection_schema_version": 1,
        "discovery_mode": current.get("discovery_mode"),
        "exposure_policy": current.get("exposure_policy"),
        "catalog_visibility_policy": "public_contract_projection",
        "task_specific_tool_filtering": current.get("task_specific_tool_filtering"),
        "backend_selection_policy": current.get("backend_selection_policy"),
        "provider_selection_policies": deepcopy(
            current.get("provider_selection_policies", [])
        ),
        "scientific_resource_selection_policy": current.get(
            "scientific_resource_selection_policy"
        ),
        "automatic_fallback": current.get("automatic_fallback"),
        "evaluation_resource_budget": dict(_PUBLIC_RESOURCE_BUDGET),
        "execution_layers": public_layers,
        "actions": public_actions,
        "backends": public_backends,
        "resources": [
            _public_resource_projection(resource)
            for resource in current.get("resources", [])
            if isinstance(resource, dict)
        ],
    }
    projection_payload = deepcopy(projection)
    projection["catalog_projection_hash"] = hashlib.sha256(
        _canonical_catalog_bytes(projection_payload)
    ).hexdigest()
    return projection


def validate_active_catalog_snapshot(
    snapshot: dict[str, Any],
    *,
    workspace: str | os.PathLike[str] | Path | None = None,
) -> dict[str, Any]:
    """校验显式快照与当前 workspace 的 active/file 完全一致后返回副本。

    输入快照必须能按目录协议规范编码并通过 ``catalog_hash`` 校验；当前
    workspace 必须已经成功安装同一快照，固定文件也必须存在且内容一致。
    已有 worker marker 若存在还必须与 workspace 和 hash 一致。任何失败都抛出
    ``CatalogFreezeError``，不会清理或改写调用方的环境变量。
    """

    root = _catalog_workspace_path(workspace)
    failure = _catalog_freeze_failure(root)
    if failure is not None:
        raise CatalogFreezeError(
            "workspace 的 MiniChem 目录冻结已失败，拒绝使用显式快照："
            f"{failure['message']}（code={failure['code']}，"
            f"exception_type={failure['exception_type']}）"
        )
    if not isinstance(snapshot, dict) or not _catalog_hash_matches(snapshot):
        raise CatalogFreezeError("显式 MiniChem 目录快照的 catalog_hash 不一致")
    try:
        normalized = json.loads(_canonical_catalog_bytes(snapshot).decode("utf-8"))
    except (CatalogFreezeError, UnicodeError, json.JSONDecodeError) as exc:
        raise CatalogFreezeError("显式 MiniChem 目录快照规范化失败") from exc
    if not isinstance(normalized, dict):
        raise CatalogFreezeError("显式 MiniChem 目录快照规范化后不是对象")

    marker_workspace = os.environ.get(CATALOG_WORKSPACE_ENV, "")
    marker_hash = os.environ.get(CATALOG_HASH_ENV, "")
    if marker_workspace or marker_hash:
        if marker_workspace != str(root) or marker_hash != normalized.get("catalog_hash"):
            raise CatalogFreezeError("显式快照与现有服务冻结 marker 不一致")

    with _ACTIVE_CATALOG_LOCK:
        if _ACTIVE_CATALOG_WORKSPACE != root or _ACTIVE_CATALOG_SNAPSHOT is None:
            raise CatalogFreezeError("当前 workspace 尚未安装可验证的 active 目录")
        active = deepcopy(_ACTIVE_CATALOG_SNAPSHOT)
        if active != normalized:
            raise CatalogFreezeError("显式快照与进程内 active 目录不一致")
        frozen = _read_catalog_file(root / CATALOG_FILENAME)
        if frozen is None or frozen != active:
            raise CatalogFreezeError("显式快照与 workspace 目录文件不一致")
    return normalized


def active_catalog_hash() -> str:
    return str(active_catalog_snapshot()["catalog_hash"])


def mcp_action_description(specification: ActionSpec) -> str:
    """生成面向调用方的 Action 说明，只公开科学字段和真实调用约束。"""

    backend_text = ", ".join(specification.backend_ids)
    backends = backend_specs()
    requirement_parts = []
    backend_notes = []
    for backend_id in specification.backend_ids:
        backend = backends[backend_id]
        input_fields = backend.required_input_fields.get(specification.id, ())
        method_fields = backend.required_method_fields.get(specification.id, ())
        setting_fields = backend.required_setting_fields.get(specification.id, ())
        method_choices = backend.allowed_method_values.get(specification.id, {})
        setting_choices = backend.allowed_setting_values.get(specification.id, {})
        component_roles = backend.required_component_roles.get(specification.id, ())
        component_options = backend.component_backend_options.get(specification.id, {})
        details = []
        if input_fields:
            details.append("inputs=" + ",".join(input_fields))
        if method_fields:
            details.append("method_spec=" + ",".join(method_fields))
        if setting_fields:
            details.append("action_settings=" + ",".join(setting_fields))
        for field_name, choices in method_choices.items():
            details.append(
                f"method_spec.{field_name}=[{'|'.join(str(choice) for choice in choices)}]"
            )
        for field_name, choices in setting_choices.items():
            details.append(
                f"action_settings.{field_name}=[{'|'.join(str(choice) for choice in choices)}]"
            )
        if component_roles:
            details.append(
                "component_backends="
                + ",".join(
                    f"{role}:[{'|'.join(component_options.get(role, ()))}]"
                    for role in component_roles
                )
            )
        requirement_parts.append(
            backend_id + (" [" + "; ".join(details) + "]" if details else "")
        )
        note = f"{backend_id}: {_public_backend_text(backend_id, backend.description)}"
        if backend.method_schema:
            documented_fields = set(method_fields) | set(setting_fields) | set(method_choices)
            note += " Scientific parameter schema: " + "; ".join(
                f"{name}={description}"
                for name, description in backend.method_schema.items()
                if name in documented_fields
            )
        registered_resources = resources_for_backends((backend_id,))
        if registered_resources:
            resource_text = ", ".join(
                f"{item['id']} ({item.get('selection_syntax', 'runtime-managed')})"
                for item in registered_resources
            )
            note += " Registered resources: " + resource_text
        parameter_specs = backend.parameter_specs.get(specification.id, {})
        relevant_paths = {
            *(f"inputs.{name}" for name in (*specification.required_inputs, *input_fields)),
            *(f"method_spec.{name}" for name in method_fields),
            *(f"action_settings.{name}" for name in setting_fields),
        }
        public_parameters = []
        for field_path, metadata in parameter_specs.items():
            if field_path not in relevant_paths:
                continue
            description, impact = _public_parameter_text(
                specification.id,
                field_path,
                metadata.get("description"),
                metadata.get("impact"),
            )
            detail = f"{field_path}: {description or 'documented scientific parameter'}"
            if impact:
                detail += f" Impact: {impact}"
            public_parameters.append(detail)
        if public_parameters:
            required_parameter_paths = {
                *(f"inputs.{name}" for name in (*specification.required_inputs, *input_fields)),
                *(f"method_spec.{name}" for name in method_fields),
                *(f"action_settings.{name}" for name in setting_fields),
            }
            note += " Caller-supplied scientific parameters: " + "; ".join(
                detail + (" (required)" if detail.split(":", 1)[0] in required_parameter_paths else "")
                for detail in public_parameters
            )
        backend_notes.append(note)
    policy = specification.selection_policy
    policy_labels = {
        "agent_backend_required": "caller_backend_required",
        "agent_components_required": "caller_components_required",
        "agent_source_required": "caller_source_required",
        "fixed_source": "fixed_source",
        "internal_deterministic": "service_deterministic",
    }
    if policy == "fixed_source":
        requirement = f"The fixed data source is {backend_text}; backend_id/source_id are not required."
    elif policy == "internal_deterministic":
        requirement = f"The deterministic service provider is {backend_text}; backend_id is not required."
    elif policy == "agent_source_required":
        requirement = f"source_id is required; choose exactly one of: {backend_text}."
    elif policy == "agent_components_required":
        requirement = (
            f"backend_id is required; choose exactly one primary backend from: {backend_text}. "
            "Also supply every component_backends role required by that backend."
        )
    else:
        requirement = f"backend_id is required; choose exactly one of: {backend_text}."
    required = ", ".join(specification.required_inputs) or "none"
    optional = ", ".join(specification.optional_inputs) or "none"
    input_contract = _public_action_text(
        specification.id,
        "input_description",
        specification.input_description or "structured inputs described by the action",
    )
    budget = resource_budget_record()
    includes_gaussian = "gaussian" in specification.backend_ids
    produces_gaussian_input = specification.primary_output == "GaussianInputDeck"
    managed_gaussian_resources = includes_gaussian or produces_gaussian_input
    only_gaussian = specification.backend_ids == ("gaussian",)
    if produces_gaussian_input:
        resource_description = (
            "This Gaussian-input-producing Action uses service_managed_full_capacity for "
            "the generated deck, and the caller cannot choose its CPU or memory."
        )
    elif only_gaussian:
        resource_description = (
            "For backend_id=gaussian, CPU and memory use service_managed_full_capacity, "
            "execution is serial, and the caller cannot override those resources."
        )
    elif managed_gaussian_resources:
        resource_description = (
            "For backend_id=gaussian, CPU and memory use service_managed_full_capacity, "
            "execution is serial, and the caller cannot override those resources. For other "
            "backends, resource_limits remain optional caller-selected values within the "
            "configured service budget."
        )
    else:
        resource_description = "Optional resource_limits are caller-supplied fields checked by the service policy. " + _agent_resource_contract(budget)
    batch_note = (
        "For two or more independent inputs using this same Action and Backend, use "
        "submit_action_batch; repeated execute_action calls are synchronous and serial. "
        if specification.batch_safe
        else ""
    )
    return (
        f"{_public_action_text(specification.id, 'description', specification.description)} "
        f"Primary output: {specification.primary_output}. "
        f"Input contract: {input_contract}. "
        f"Required inputs keys: {required}. Optional inputs keys: {optional}. "
        f"{requirement} Backend-specific required fields: {' | '.join(requirement_parts)}. "
        f"Backend notes: {' | '.join(backend_notes)}. "
        f"{resource_description} "
        f"Execution class: {specification.execution_class}; timeout follows the service policy "
        "and cannot be supplied as a scientific field. "
        f"{batch_note}"
        f"Provider selection policy: {policy_labels.get(policy, policy)}. The system validates and executes the exact "
        "declared provider choices; it never falls back to another backend or source."
    )


def agent_toolbox_overview(
    *,
    include_health: bool = True,
    snapshot: dict[str, Any] | None = None,
) -> str:
    """Neutral system-prompt summary; it contains no workflow recipe or ranking."""

    return progressive_toolbox_overview(snapshot=snapshot)

    if snapshot is not None:
        health = {
            backend["id"]: backend.get("health") or {}
            for backend in snapshot.get("backends", [])
        }
    else:
        health = probe_all_backends(BACKEND_SPECS) if include_health else {}
    budget = (
        dict(snapshot.get("evaluation_resource_budget") or {})
        if snapshot is not None
        else resource_budget_record()
    )
    if not budget:
        budget = resource_budget_record()
    grouped: dict[str, list[ActionSpec]] = defaultdict(list)
    for specification in ACTION_SPECS:
        grouped[specification.category].append(specification)
    lines = [
        "Layer 1 exposes this same complete predefined Action catalog to every task. These "
        "Actions are validated conveniences for common operations, not a required workflow or "
        "the boundary of the toolbox. Some entries are explicitly service-fixed reproductions; "
        "for those entries the listed contract determines the call. You decide which tools "
        "to call, their order, and every scientifically meaningful backend, component, source, "
        "and method choice. There is no hidden workflow or task-specific tool retrieval; fixed "
        "entries are labeled in their descriptions. Automatic backend selection and fallback are disabled.",
        "",
        _agent_resource_contract(budget),
        "",
        "Every predefined Action accepts one ActionRequest object with: backend_id, component_backends, "
        "source_id, inputs, method_spec, action_settings, and optional resource_limits. Read each "
            "tool description before calling it: numerical computations require a caller-selected "
        "backend unless that Action is service-fixed; composite computations also require every "
        "component role. A service-fixed Action may require inputs, method_spec, or action_settings "
        "to be empty; inspect its contract.",
        "",
        "A minimal AtomicStructure is {'atoms': [{'element': 'H', "
        "'position_angstrom': [0, 0, 0]}], 'charge': 0, 'multiplicity': 1}. "
        "Periodic structures additionally contain a 3x3 cell_angstrom and pbc=[true,true,true]. "
        "Outputs can be passed onward as structured result objects or registered ArtifactRef objects.",
        "Scientific files use explicit registered ResourceRef values. Use either "
        "resource://<resource_id>/<Element> for element-file collections or "
        "resource://<resource_id>/<Variant> for exact variant collections such as VASP POTCARs, or "
        "resource://<resource_id> for a parameter set, model checkpoint, or other registered "
        "single file. You must choose the resource family/variant/checkpoint, element mapping, model "
        "branch, device, cutoffs, and compatible method; the dispatcher never chooses them.",
    ]
    for category in CATEGORY_LABELS:
        lines.extend(["", f"### {CATEGORY_LABELS[category]}"])
        for specification in grouped.get(category, []):
            backend_parts = []
            for backend_id in specification.backend_ids:
                state = health.get(backend_id, {}).get("status")
                backend_parts.append(f"{backend_id} ({state})" if state else backend_id)
            lines.append(
                f"- `{specification.id}` — {_public_action_text(specification.id, 'description', specification.description)} "
                f"Providers: {', '.join(backend_parts)}. Selection policy: "
                f"{specification.selection_policy}."
            )
    lines.extend(
        [
            "",
            "### Registered read-only scientific resources",
        ]
    )
    resource_values = (
        snapshot.get("resources", []) if snapshot is not None else _mini_resource_snapshot()
    )
    for resource in resource_values:
        backends_text = ", ".join(resource.get("compatible_backends") or [])
        state = "available" if resource.get("available") else "unavailable"
        if resource.get("selectable", True):
            coverage = _resource_coverage(resource)
            lines.append(
                f"- `{resource['id']}` — {resource.get('display_name', resource['id'])}; "
                f"backend: {backends_text}; format: {resource.get('format', '-')}; "
                f"coverage: {coverage}; status: {state}; syntax: `{resource.get('selection_syntax')}`."
            )
        else:
            lines.append(
                f"- `{resource['id']}` — runtime-managed resource for {backends_text}; status: {state}."
            )
    lines.extend(
        [
            "",
            "An unavailable backend remains visible so that the choice set is not hidden. "
            "If a call fails, inspect that exact result and independently decide whether to "
            "change parameters, call another Action, use the native-software layer, or write an "
            "analysis program. The system never makes that decision for you.",
        ]
    )
    return "\n".join(lines)


def progressive_toolbox_overview(
    *,
    snapshot: dict[str, Any] | None = None,
) -> str:
    """Compact first-layer index for on-demand, neutral catalog discovery."""

    actions = (
        list(snapshot.get("actions", []))
        if snapshot is not None
        else [specification.as_dict() for specification in ACTION_SPECS]
    )
    backends = (
        list(snapshot.get("backends", []))
        if snapshot is not None
        else [specification.as_dict() for specification in BACKEND_SPECS]
    )
    resources = (
        list(snapshot.get("resources", []))
        if snapshot is not None
        else _mini_resource_snapshot()
    )
    budget = (
        dict(snapshot.get("evaluation_resource_budget") or {})
        if snapshot is not None
        else resource_budget_record()
    )
    if not budget:
        budget = resource_budget_record()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for action in actions:
        grouped[str(action.get("category", ""))].append(action)

    lines = [
        "The complete Action catalog is available to every caller and is loaded progressively "
        "instead of placing every Action schema in the initial context. It provides predefined "
        "Scientific and Data Actions, Backend entries, and registered scientific resources. "
        "Some operations have service-defined request sections; inspect_action states which "
        "sections are empty. No recommendation, automatic backend selection, or fallback is "
        "performed.",
        "",
        _agent_resource_contract(budget),
        "",
        "If you do not know the exact Action id, call `list_action_domains` once to receive every "
        "action_id grouped under the compact domain index; use `search_actions` when you need to "
        "filter those catalog entries with your own scientific terms or an exact category/backend "
        "filter. Call `inspect_action` before a new "
        "Action to obtain its exact inputs, provider policy, provider-specific field types, "
        "conditional rules, output contract, and fill-in `execute_action` request template; use "
        "`inspect_backend` to see one Backend's capabilities. Scientific files and data resources "
        "are found with `search_resources` and resolved exactly with `inspect_resource`. "
        "These discovery operations only return catalog facts.",
        "",
        "Run a selected predefined Action through `execute_action`. Its request contains an explicit "
        "action_id plus backend_id, component_backends, source_id, inputs, method_spec, "
        "action_settings, and resource_limits. Numerical Actions require the provider choices "
        "declared by `inspect_action`; service-defined operations may require inputs, method_spec, "
        "and action_settings to be empty. The dispatcher validates exactly what you provide and never "
        "substitutes another choice. `execute_action` is synchronous: obey any provider-specific "
        "maximum walltime returned by `inspect_action`.",
        "For two or more independent inputs that use the same batch_safe Action and Backend, use "
        "`submit_action_batch` instead of repeated `execute_action` calls. The batch tool runs "
        "children concurrently within the service policy, queues excess items, and preserves "
        "independent results and provenance.",
        "",
        "For capabilities outside the predefined Action layer, the software-native and programmable "
        "layers remain peers. Discover exact registered programs with `list_software`, retrieve one "
        "invocation guide with `inspect_software`, and search its documentation with "
        "`search_software_documentation`. You may instead write a complete Python analysis program "
        "and explicitly select a listed runtime. For a calculation longer than a synchronous Action "
        "permits, write the complete native input, call `submit_native_job`, and explicitly poll or "
        "collect that job; submission does not change the selected software or parameters. You "
        "decide whether and how to interleave all three layers.",
        "",
        "Available Action domains:",
    ]
    for category, label in CATEGORY_LABELS.items():
        values = grouped.get(category, [])
        scientific = sum(not bool(value.get("data_action")) for value in values)
        data = len(values) - scientific
        kind_text = f"{scientific} scientific"
        if data:
            kind_text += f", {data} data"
        lines.append(f"- `{category}` — {label} ({kind_text} Actions).")
    lines.extend(
        [
            "",
            "An unavailable Backend remains discoverable. If an exact call fails, inspect its result "
            "and independently decide the next scientific step; the service does not add defaults "
            "or redirect the call.",
        ]
    )
    return "\n".join(lines)


def toolbox_overview(
    *,
    discovery_mode: str | None = None,
    include_health: bool = True,
    snapshot: dict[str, Any] | None = None,
) -> str:
    """Return the prompt index corresponding to the selected public surface."""

    mode = resolve_tool_discovery_mode(discovery_mode)
    if mode == "full":
        return agent_toolbox_overview(
            include_health=include_health,
            snapshot=snapshot,
        )
    return progressive_toolbox_overview(snapshot=snapshot)


def markdown_catalog(*, include_health: bool = True) -> str:
    snapshot = catalog_snapshot(include_health=include_health)
    backend_by_id = {backend["id"]: backend for backend in snapshot["backends"]}
    lines = [
        "# MiniChem Atomic Tool Catalog",
        "",
        f"Catalog hash: `{snapshot['catalog_hash']}`",
        "",
        "These predefined Actions are the validated common-operation layer. They are exposed to every task but are not mandatory; the MCP server also exposes native-software and programmable-analysis layers. Provider selection follows each Action's policy.",
        "",
        "| Action | Category | Primary output | Selection policy | Providers | Required inputs | Description |",
        "|---|---|---|---|---|---|---|",
    ]
    for action in snapshot["actions"]:
        required = ", ".join(action["required_inputs"])
        if not required:
            contracts = []
            for backend_id in action["backend_ids"]:
                fields = backend_by_id[backend_id]["required_input_fields"].get(
                    action["id"], []
                )
                if fields:
                    contracts.append(f"{backend_id}({','.join(fields)})")
            required = "backend-specific: " + "; ".join(contracts) if contracts else "none"
        lines.append(
            "| {id} | {category} | {primary_output} | {policy} | {backends} | {required} | {description} |".format(
                id=action["id"],
                category=action["category"],
                primary_output=action["primary_output"],
                policy=action["selection_policy"],
                backends=", ".join(action["backend_ids"]),
                required=required,
                description=action["description"].replace("|", "\\|"),
            )
        )
    lines.extend(
        [
            "",
            "## Registered scientific resources",
            "",
            "| Resource | Kind | Backends | Version | Format | Status | Explicit selection syntax | Coverage |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )
    for resource in snapshot["resources"]:
        coverage = _resource_coverage(resource)
        lines.append(
            "| {id} | {kind} | {backends} | {version} | {format} | {status} | {syntax} | {coverage} |".format(
                id=resource["id"],
                kind=resource.get("kind", "-"),
                backends=", ".join(resource.get("compatible_backends") or []),
                version=resource.get("version", "-"),
                format=resource.get("format", "-"),
                status="available" if resource.get("available") else "unavailable",
                syntax=resource.get("selection_syntax", "runtime-managed"),
                coverage=coverage,
            )
        )
    lines.extend(
        [
            "",
            "## Backend installation and health",
            "",
            "| Backend | Runtime | Status | Conda packages | Pip packages | Executables | External scientific data |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for backend in snapshot["backends"]:
        health_item = backend.get("health") or {}
        lines.append(
            "| {id} | {runtime} | {status} | {conda} | {pip} | {executables} | {data} |".format(
                id=backend["id"],
                runtime=backend["runtime"],
                status=health_item.get("status", "not_probed"),
                conda=", ".join(backend["conda_packages"]),
                pip=", ".join(backend["pip_packages"]),
                executables=", ".join(backend["executables"]),
                data="; ".join(backend["required_data_resources"]).replace("|", "\\|"),
            )
        )
    return "\n".join(lines) + "\n"
