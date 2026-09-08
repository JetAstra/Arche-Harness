"""ARCHE research_task 和历史运行 Bundle 的目录及摘要存储。

本模块只保存运行身份、状态、阶段摘要和文件引用，不解释科学结果，也不复制
MiniChem 的大文件。新 research_task 写入 ``outputs/research_tasks/<id>``；
历史 ``outputs/runs/<run-id>`` 只能由显式兼容定位器读取。索引文件使用同目录
临时文件加原子替换，进程中断时不会留下半份 JSON。
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping

try:
    import fcntl
except ImportError:  # pragma: no cover - ARCHE 标准环境为 Linux
    fcntl = None


RUN_BUNDLE_SCHEMA_VERSION = 1
_RESEARCH_TASK_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

_THREAD_LOCK_REGISTRY_GUARD = threading.Lock()
_BUNDLE_THREAD_LOCKS: dict[str, threading.RLock] = {}
_OUTPUT_THREAD_LOCKS: dict[str, threading.RLock] = {}
_LOCK_STATE = threading.local()


def _shared_thread_lock(
    registry: dict[str, threading.RLock], key: str
) -> threading.RLock:
    """返回同一路径在进程内共享的可重入锁。"""

    with _THREAD_LOCK_REGISTRY_GUARD:
        lock = registry.get(key)
        if lock is None:
            lock = threading.RLock()
            registry[key] = lock
        return lock


def _open_lock_file(path: Path) -> int:
    """以不跟随符号链接的方式打开锁文件。"""

    flags = os.O_RDWR | os.O_CREAT
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        if not path.is_file() or path.is_symlink():
            raise ValueError("锁文件必须是普通文件")
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


@contextmanager
def _file_lock(path: Path):
    """在 Linux 进程间锁定一个已校验的普通文件。"""

    descriptor = _open_lock_file(path)
    try:
        if fcntl is not None:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        if fcntl is not None:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)

LEGACY_STAGE_LAYOUT = "legacy"
INVESTIGATION_STAGE_LAYOUT = "investigation"
# 新 research_task 的正式布局。旧的 investigation 布局保留给历史 V2 Bundle
# 的只读定位和兼容测试，不能成为新任务的默认写入目标。
RESEARCH_TASK_STAGE_LAYOUT = "research_task"
RESEARCH_TASK_LAYOUT = RESEARCH_TASK_STAGE_LAYOUT
DEFAULT_STAGE_LAYOUT = RESEARCH_TASK_STAGE_LAYOUT

LEGACY_STAGE_NAMES = (
    "retrieval",
    "hypothesis",
    "admission_debate",
    "planning",
    "expert_pre_execution",
    "execution",
    "expert_post_execution",
    "revision",
    "final_result",
)

INVESTIGATION_STAGE_NAMES = (
    "retrieval",
    "hypothesis",
    "investigation_and_experiment",
    "final_result",
)
RESEARCH_TASK_STAGE_NAMES = (
    "literature_retrieval",
    "hypothesis_preparation",
    "investigation_and_experiment",
    "task_review_and_result",
)
STAGE_NAMES = RESEARCH_TASK_STAGE_NAMES
STAGE_LAYOUTS = MappingProxyType(
    {
        LEGACY_STAGE_LAYOUT: LEGACY_STAGE_NAMES,
        INVESTIGATION_STAGE_LAYOUT: INVESTIGATION_STAGE_NAMES,
        RESEARCH_TASK_STAGE_LAYOUT: RESEARCH_TASK_STAGE_NAMES,
    }
)

_TERMINAL_STAGE_STATUSES = frozenset(
    {"completed", "failed", "disabled", "not_applicable", "skipped"}
)
_RESEARCH_TASK_RUN_STATUSES = frozenset(
    {
        "open",
        "running",
        "retrieval_incomplete",
        "waiting_investigation_job",
        "paused",
        "succeeded",
        "failed",
        "stopped_no_supported_hypothesis",
    }
)
_LEGACY_RUN_STATUSES = frozenset(
    {
        "running",
        "retrieval_incomplete",
        "waiting_execution",
        "paused",
        "succeeded",
        "failed",
        "stopped_no_supported_hypothesis",
    }
)
_RESEARCH_TASK_STAGE_STATUSES = frozenset(
    {
        "not_started",
        "open",
        "running",
        "waiting_investigation_job",
        "paused",
        "completed",
        "failed",
        "disabled",
        "not_applicable",
        "skipped",
    }
)
_LEGACY_STAGE_STATUSES = frozenset(
    {
        "not_started",
        "open",
        "running",
        "waiting_execution",
        "paused",
        "completed",
        "failed",
        "disabled",
        "not_applicable",
        "skipped",
    }
)


def _empty_research_task_recovery() -> dict[str, Any]:
    """返回正式 research_task 的恢复字段基座。"""

    return {"available": False, "waiting_investigation_jobs": [], "pauses": []}


def _empty_legacy_recovery() -> dict[str, Any]:
    """返回历史 Bundle 的恢复字段基座；只供显式 legacy 读取。"""

    return {"available": False, "waiting_executions": [], "pauses": []}


def _normalize_research_task_recovery(value: Any) -> dict[str, Any]:
    """按正式字段整理等待 Investigation 作业和暂停事实。

    等待作业由 Working Memory 与 waiting-job sidecar 保存；Bundle 只接受已经
    脱敏的身份摘要，不从运行时检查点推断或创建恢复记录。
    """

    normalized = _empty_research_task_recovery()
    if not isinstance(value, Mapping):
        return normalized
    waiting = value.get("waiting_investigation_jobs")
    if waiting is not None and not isinstance(waiting, list):
        raise ValueError("recovery.waiting_investigation_jobs 必须是数组")
    if isinstance(waiting, list):
        for index, item in enumerate(waiting):
            if not isinstance(item, Mapping):
                raise ValueError(
                    f"waiting_investigation_jobs[{index}] 必须是对象"
                )
            # 公开 Bundle 只保留作业/会话身份摘要；路径、凭据和科学正文不进入
            # 展示恢复字段。允许 canonical sidecar 的平面身份字段及 digest。
            allowed_fields = (
                "task_id",
                "batch_id",
                "combination_id",
                "job_id",
                "operation_id",
                "submission_key",
                "job_family",
                "scope_id",
                "codex_thread_id",
                "turn_id",
                "request_digest",
                "submission_request_digest",
                "turn_request_digest",
                "status",
                "session_id",
                "stage",
                "logical_operation_instance_id",
                "attempt_operation_instance_id",
                "attempt_number",
                "repair_request_number",
                "expected_result_version",
                "workspace_ref_digest",
                "submit_turn_id",
                "submit_turn_started_event_ref",
                "turn_started_event_ref",
                "repair_request_event_ref",
                "scope_credential_generation_digest",
            )
            unknown = set(item) - set(allowed_fields)
            if unknown:
                raise ValueError(
                    "waiting_investigation_jobs 含未知字段: "
                    + ", ".join(sorted(map(str, unknown)))
                )
            projected = {
                field: item[field]
                for field in allowed_fields
                if field in item
            }
            if not projected:
                raise ValueError(
                    f"waiting_investigation_jobs[{index}] 缺少作业身份"
                )
            for field, field_value in projected.items():
                if field in {
                    "attempt_number",
                    "repair_request_number",
                }:
                    if (
                        isinstance(field_value, bool)
                        or not isinstance(field_value, int)
                        or field_value < 1
                    ):
                        raise ValueError(
                            f"waiting_investigation_jobs[{index}].{field} 必须是正整数"
                        )
                elif not isinstance(field_value, str) or not field_value.strip():
                    raise ValueError(
                        f"waiting_investigation_jobs[{index}].{field} 必须是非空字符串"
                    )
            normalized["waiting_investigation_jobs"].append(projected)
    pauses = value.get("pauses")
    if pauses is not None and not isinstance(pauses, list):
        raise ValueError("recovery.pauses 必须是数组")
    if isinstance(pauses, list):
        for index, item in enumerate(pauses):
            if not isinstance(item, Mapping):
                raise ValueError(f"pauses[{index}] 必须是对象")
            required = {
                field: item.get(field)
                for field in ("pause_id", "operation", "reason_code", "restart_allowed")
            }
            if any(
                not isinstance(required[field], str) or not required[field].strip()
                for field in ("pause_id", "operation", "reason_code")
            ) or not isinstance(required["restart_allowed"], bool):
                raise ValueError(
                    f"pauses[{index}] 缺少非空 pause_id/operation/reason_code 或 bool restart_allowed"
                )
            normalized["pauses"].append(required)
    normalized["available"] = bool(
        normalized["waiting_investigation_jobs"] or normalized["pauses"]
    )
    return normalized


def _normalize_legacy_recovery(value: Any) -> dict[str, Any]:
    """按历史字段整理等待执行和暂停事实；只供显式 legacy 读取。

    等待执行必须同时带有组合、operation 和 checkpoint 三个非空字符串；任一
    字段缺失都抛出错误，防止把不可恢复的等待状态写成 ``available=true``。
    """

    normalized = _empty_legacy_recovery()
    if not isinstance(value, Mapping):
        return normalized
    waiting = value.get("waiting_executions")
    if waiting is not None and not isinstance(waiting, list):
        raise ValueError("recovery.waiting_executions 必须是数组")
    if isinstance(waiting, list):
        for index, item in enumerate(waiting):
            if not isinstance(item, Mapping):
                raise ValueError(f"waiting_executions[{index}] 必须是对象")
            required = {
                field: item.get(field)
                for field in ("combination_id", "operation_id", "checkpoint_id")
            }
            if any(
                not isinstance(value, str) or not value.strip()
                for value in required.values()
            ):
                raise ValueError(
                    f"waiting_executions[{index}] 缺少非空 combination_id/operation_id/checkpoint_id"
                )
            normalized["waiting_executions"].append(required)
    pauses = value.get("pauses")
    if pauses is not None and not isinstance(pauses, list):
        raise ValueError("recovery.pauses 必须是数组")
    if isinstance(pauses, list):
        for index, item in enumerate(pauses):
            if not isinstance(item, Mapping):
                raise ValueError(f"pauses[{index}] 必须是对象")
            required = {
                field: item.get(field)
                for field in ("pause_id", "operation", "reason_code", "restart_allowed")
            }
            if any(
                not isinstance(required[field], str) or not required[field].strip()
                for field in ("pause_id", "operation", "reason_code")
            ) or not isinstance(required["restart_allowed"], bool):
                raise ValueError(
                    f"pauses[{index}] 缺少非空 pause_id/operation/reason_code 或 bool restart_allowed"
                )
            normalized["pauses"].append(
                required
            )
    normalized["available"] = bool(
        normalized["waiting_executions"] or normalized["pauses"]
    )
    return normalized

_README_TEXT = {
    "outputs": """# 运行输出

`outputs/` 是 ARCHE 运行结果的统一入口。新 research_task 位于
`research_tasks/<research-task-id>/`；历史 `runs/<run-id>/` 只供显式兼容定位器读取。
""",
    "runs": """# 运行 Bundle

`runs/` 下保存历史兼容 Bundle，目录名是旧运行 ID。只有显式迁移或兼容定位器才能
读取这里的内容；新 `research_task` 必须创建在
`outputs/research_tasks/<research_task_id>/`。进入具体目录后，先查看 `status.json`
和 `summary.json`，再按需读取阶段摘要、计算产物和内部记录。
""",
    "research_tasks": """# research_task Bundle

`research_tasks/` 下的每个子目录代表一个新的 research_task，目录名是
`research_task_id`。新任务只在这里创建；历史 `runs/` 不会作为新任务的回退目录。
先查看具体任务目录中的 `status.json` 和 `summary.json`，再按需读取阶段摘要、计算
产物和内部记录。
""",
    "run": """# 单次运行

`run.json` 保存运行身份和配置，`status.json` 保存当前状态与恢复信息，
`summary.json` 汇总各阶段状态。`stages/` 保存结构化阶段摘要，`artifacts/`
保存计算产物，`internal/` 保存恢复、审计和工具目录记录。
""",
    "stages": """# 阶段摘要

这里保存给研究员阅读的结构化阶段摘要。阶段文件集合由创建 Bundle 时冻结的
布局决定，每个文件使用固定字段，并通过 `entries` 保存该阶段的多次记录。新的
research_task 使用 `literature_retrieval`、`hypothesis_preparation`、
`investigation_and_experiment` 和 `task_review_and_result` 四个阶段。
""",
    "artifacts": """# 计算产物

这里保存结构、计算输入、日志、谱图等文件。阶段摘要和总摘要通过相对路径引用
这些产物。
""",
    "internal": """# 内部记录

`state/` 保存运行恢复所需的数据，`audit/` 保存请求和响应记录，`catalog/`
保存本次运行使用的工具目录。通常先查看 Run Bundle 根目录和阶段摘要，只有需要
检查恢复状态或调用细节时才读取这里。
""",
    "state": """# 恢复状态

这里保存 Working Memory 数据库、checkpoint 和恢复任务所需的状态。恢复运行时，
系统从这里读取原有 operation 和 checkpoint，继续查询已经提交的外部作业。
""",
    "audit": """# 审计记录

这里保存模型调用全文、请求收据、工具协议记录和详细结构化视图。需要核对实际请求、
响应或验证过程时，从对应的编号目录进入。
""",
    "model_calls": """# 模型调用记录

这里按运行阶段保存模型请求、响应、收据和校验记录。每个编号目录对应一个阶段，
其中的 JSON 文件对应具体调用或结构化视图。
""",
    "catalog": """# 工具目录

这里保存本次运行使用的 MiniChem 工具目录快照、版本和哈希。执行和恢复时，系统用
该哈希核对 MiniChem 服务提供的工具定义。
""",
}


def _stages_readme_text(stage_names: tuple[str, ...]) -> str:
    """根据冻结布局生成阶段目录说明，不改变共享输出目录说明。"""

    stage_files = "\n".join(f"- `{stage}.json`" for stage in stage_names)
    return (
        _README_TEXT["stages"].rstrip()
        + "\n\n本次 Bundle 的阶段文件：\n"
        + stage_files
        + "\n"
    )


def utc_timestamp() -> str:
    """返回带毫秒和 ``Z`` 后缀的 UTC 时间。"""

    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def validate_research_task_id(research_task_id: str) -> str:
    """校验 research_task 身份只能作为单层目录名使用。"""

    if not isinstance(research_task_id, str) or not _RESEARCH_TASK_ID_PATTERN.fullmatch(
        research_task_id
    ):
        raise ValueError(
            "research_task_id 只能包含字母、数字、点、下划线和连字符"
        )
    return research_task_id


def _copy_json(value: Any) -> Any:
    """复制可 JSON 化的运行元数据，不截断科学字段。"""

    if isinstance(value, Mapping):
        return {str(key): _copy_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_copy_json(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _validate_source_task_sequence(value: Any) -> int:
    """校验展示文件引用的 Working Memory 事件序号。"""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("source_task_sequence 必须是非负整数")
    return value


class RunBundle:
    """创建、恢复并更新一次运行 Bundle。

    输入是工作目录、research_task 身份和可选配置。新运行固定创建
    ``outputs/research_tasks/<research_task_id>/``；schema-v10 历史 fixture 只由测试
    helper 手工构造，生产入口不能选择旧布局。
    正式 Bundle 可以继续更新状态；历史定位器返回只读视图，任何写入方法都会拒绝。
    恢复只读取同一路径并追加状态事件；目录外路径、符号链接和损坏的固定 JSON 会
    抛出 ``ValueError``，调用方可以把该失败转换成任务级诊断。
    """

    def __init__(
        self,
        work_dir: str | Path,
        research_task_id: str,
        *,
        scientific_question: str | None = None,
        task_config: Mapping[str, Any] | None = None,
        model_configuration: Mapping[str, Any] | None = None,
        resume: bool = False,
    ) -> None:
        self._initialize_paths(
            work_dir,
            research_task_id,
            stage_layout=RESEARCH_TASK_STAGE_LAYOUT,
        )
        if resume:
            self._load_existing_bundle()
        else:
            self._create_new_bundle(
                scientific_question=scientific_question,
                task_config=task_config,
                model_configuration=model_configuration,
            )

    @classmethod
    def create_research_task(
        cls,
        work_dir: str | Path,
        research_task_id: str,
        *,
        scientific_question: str | None = None,
        task_config: Mapping[str, Any] | None = None,
        model_configuration: Mapping[str, Any] | None = None,
    ) -> "RunBundle":
        """创建新的 research_task Bundle。

        新任务固定使用 ``outputs/research_tasks/<research-task-id>/``；该方法不
        检查或复用历史 ``outputs/runs`` 目录。
        """

        return cls(
            work_dir,
            research_task_id=research_task_id,
            scientific_question=scientific_question,
            task_config=task_config,
            model_configuration=model_configuration,
        )

    @classmethod
    def _open_existing_at_layout(
        cls,
        work_dir: str | Path,
        *,
        research_task_id: str,
        stage_layout: str,
    ) -> "RunBundle":
        """按已明确选择的布局只读打开 Bundle。"""

        instance = cls.__new__(cls)
        instance._initialize_paths(
            work_dir,
            research_task_id,
            stage_layout=stage_layout,
        )
        instance._validate_parent_chain(require_existing=True)
        instance._validate_bundle_structure()
        with instance._bundle_guard():
            instance._freeze_stage_layout_from_run_document()
            instance._validate_fixed_documents()
        return instance

    def _initialize_paths(
        self,
        work_dir: str | Path,
        research_task_id: str,
        *,
        stage_layout: str,
    ) -> None:
        """初始化 Bundle 路径；历史布局只能由私有兼容入口传入。"""

        self.work_dir = Path(work_dir).expanduser().resolve(strict=False)
        self.research_task_id = validate_research_task_id(research_task_id)
        self._set_stage_layout(stage_layout)
        self._research_task_namespace = self.stage_layout == RESEARCH_TASK_STAGE_LAYOUT
        # 历史 Bundle 只允许显式 locator 读取，不能沿用当前实例的写入方法。
        self._legacy_read_only = not self._research_task_namespace
        self.outputs_root = self.work_dir / "outputs"
        self.bundle_namespace_root = self.outputs_root / (
            "research_tasks" if self._research_task_namespace else "runs"
        )
        self.root = self.bundle_namespace_root / self.research_task_id
        self.run_path = self.root / "run.json"
        self.status_path = self.root / "status.json"
        self.summary_path = self.root / "summary.json"
        # 新任务的索引属于 research_tasks 命名空间，历史 Bundle 继续使用
        # outputs 级索引以保持只读兼容。
        index_root = (
            self.bundle_namespace_root
            if self._research_task_namespace
            else self.outputs_root
        )
        self.index_path = index_root / "index.json"
        self.latest_path = index_root / "latest.json"
        self._bundle_lock_path = self.root / ".bundle.lock"
        self._index_lock_path = index_root / ".index.lock"
        self._bundle_thread_lock = _shared_thread_lock(
            _BUNDLE_THREAD_LOCKS,
            str(self._bundle_lock_path),
        )
        self._index_thread_lock = _shared_thread_lock(
            _OUTPUT_THREAD_LOCKS,
            str(self._index_lock_path),
        )

    @classmethod
    def open_existing_research_task(
        cls,
        work_dir: str | Path,
        research_task_id: str,
    ) -> "RunBundle":
        """只读打开 ``outputs/research_tasks`` 下的 research_task Bundle。"""

        return cls._open_existing_at_layout(
            work_dir,
            research_task_id=research_task_id,
            stage_layout=RESEARCH_TASK_STAGE_LAYOUT,
        )

    @classmethod
    def open_existing_legacy(
        cls,
        work_dir: str | Path,
        run_id: str,
    ) -> "RunBundle":
        """只读打开历史 ``outputs/runs`` Bundle。

        该方法只供 schema-v10 迁移和兼容恢复；新任务不得传入历史运行身份。
        """

        return cls._open_existing_at_layout(
            work_dir,
            research_task_id=run_id,
            stage_layout=LEGACY_STAGE_LAYOUT,
        )

    def _set_stage_layout(self, stage_layout: str | None) -> None:
        """校验布局名称并冻结本实例可读写的阶段顺序。"""

        selected_layout = (
            DEFAULT_STAGE_LAYOUT if stage_layout is None else stage_layout
        )
        if not isinstance(selected_layout, str) or selected_layout not in STAGE_LAYOUTS:
            raise ValueError(f"阶段布局非法: {stage_layout!r}")
        self._stage_layout = selected_layout
        self._stage_names = tuple(STAGE_LAYOUTS[selected_layout])

    @property
    def is_research_task(self) -> bool:
        """返回当前 Bundle 是否属于新的 research_task 命名空间。"""

        return self._research_task_namespace

    @property
    def identity_field(self) -> str:
        """返回固定 JSON 使用的顶层身份字段。"""

        return "research_task_id" if self.is_research_task else "run_id"

    @property
    def storage_namespace(self) -> str:
        """返回当前 Bundle 的相对存储命名空间。"""

        return "research_tasks" if self.is_research_task else "runs"

    @property
    def _run_statuses(self) -> frozenset[str]:
        """返回当前命名空间允许的运行状态。"""

        return (
            _RESEARCH_TASK_RUN_STATUSES
            if self.is_research_task
            else _LEGACY_RUN_STATUSES
        )

    @property
    def _stage_statuses(self) -> frozenset[str]:
        """返回当前命名空间允许的阶段状态。"""

        return (
            _RESEARCH_TASK_STAGE_STATUSES
            if self.is_research_task
            else _LEGACY_STAGE_STATUSES
        )

    def _empty_recovery(self) -> dict[str, Any]:
        """返回当前命名空间的恢复字段基座。"""

        return (
            _empty_research_task_recovery()
            if self.is_research_task
            else _empty_legacy_recovery()
        )

    def _normalize_recovery(self, value: Any) -> dict[str, Any]:
        """按当前命名空间校验恢复字段。"""

        return (
            _normalize_research_task_recovery(value)
            if self.is_research_task
            else _normalize_legacy_recovery(value)
        )

    def _identity_fields(self) -> dict[str, str]:
        """返回固定 JSON 使用的公开身份字段。"""

        return {self.identity_field: self.research_task_id}

    def _validate_document_identity(
        self,
        document: Mapping[str, Any],
        *,
        document_name: str,
    ) -> None:
        """核对更新后的固定 JSON 仍只携带当前命名空间身份。"""

        if document.get(self.identity_field) != self.research_task_id:
            raise ValueError(f"{document_name} {self.identity_field} 不一致")
        if self.is_research_task and "run_id" in document:
            raise ValueError(f"{document_name} 不得包含 run_id")

    @property
    def stage_layout(self) -> str:
        """返回当前 Bundle 创建时冻结的布局身份。"""

        return self._stage_layout

    @property
    def stage_names(self) -> tuple[str, ...]:
        """返回当前 Bundle 创建时冻结的阶段顺序。"""

        return self._stage_names

    @staticmethod
    def _validate_stage_order(
        value: Any,
        expected: tuple[str, ...],
        *,
        document_name: str,
    ) -> None:
        """校验 JSON 中的阶段顺序必须与冻结布局完全一致。"""

        if not isinstance(value, list) or tuple(value) != expected:
            raise ValueError(
                f"{document_name} stage_order 与冻结阶段布局不一致"
            )

    def _freeze_stage_layout_from_run_document(self) -> dict[str, Any]:
        """从已有 run.json 恢复布局，缺失元数据时只推断 legacy。"""

        if self.run_path.is_symlink() or not self.run_path.is_file():
            raise ValueError("运行 Bundle 固定 JSON 缺失")
        run_document = self._read_json(self.run_path)
        if not isinstance(run_document, Mapping):
            raise ValueError("run.json 不是对象")
        if "stage_layout" not in run_document:
            # schema v1 旧 Bundle 没有布局身份，只能接受九阶段 legacy；即使
            # 恰好出现 stage_order，也不能把其它顺序解释成新布局。
            if self.is_research_task:
                raise ValueError("research_task Bundle 缺少 stage_layout")
            if "stage_order" in run_document:
                self._validate_stage_order(
                    run_document.get("stage_order"),
                    LEGACY_STAGE_NAMES,
                    document_name="run.json",
                )
            self._set_stage_layout(LEGACY_STAGE_LAYOUT)
            return dict(run_document)
        requested_layout = run_document.get("stage_layout")
        self._set_stage_layout(requested_layout)
        if (requested_layout == RESEARCH_TASK_STAGE_LAYOUT) != self.is_research_task:
            raise ValueError("Bundle 存储命名空间与 stage_layout 不一致")
        if "stage_order" not in run_document:
            raise ValueError("run.json stage_order 缺失")
        self._validate_stage_order(
            run_document.get("stage_order"),
            self.stage_names,
            document_name="run.json",
        )
        return dict(run_document)

    def _validate_stage_file_set(self) -> None:
        """拒绝阶段目录中缺失、增加或改名的阶段文件。"""

        stages_root = self.root / "stages"
        try:
            actual_names = {entry.name for entry in stages_root.iterdir()}
        except OSError as error:
            raise ValueError("阶段目录无法读取") from error
        expected_names = {f"{stage}.json" for stage in self.stage_names}
        allowed_names = expected_names | {"README.md"}
        unexpected_names = sorted(actual_names - allowed_names)
        missing_names = sorted(expected_names - actual_names)
        if missing_names or unexpected_names:
            details: list[str] = []
            if missing_names:
                details.append("缺少=" + ",".join(missing_names))
            if unexpected_names:
                details.append("多出=" + ",".join(unexpected_names))
            raise ValueError("阶段文件集合与冻结布局不一致: " + "; ".join(details))

    # ==================== 创建与恢复 ====================

    @contextmanager
    def _bundle_guard(self):
        """锁定当前 Bundle 的状态、阶段和运行元数据读改写。"""

        key = str(self._bundle_lock_path)
        with self._bundle_thread_lock:
            held = getattr(_LOCK_STATE, "bundle_paths", None)
            if not isinstance(held, set):
                held = set()
                _LOCK_STATE.bundle_paths = held
            if key in held:
                yield
                return
            held.add(key)
            try:
                with _file_lock(self._bundle_lock_path):
                    yield
            finally:
                held.discard(key)

    @contextmanager
    def _index_guard(self):
        """锁定当前输出命名空间 index/latest 的跨运行读改写。"""

        key = str(self._index_lock_path)
        with self._index_thread_lock:
            held = getattr(_LOCK_STATE, "index_paths", None)
            if not isinstance(held, set):
                held = set()
                _LOCK_STATE.index_paths = held
            if key in held:
                yield
                return
            held.add(key)
            try:
                with _file_lock(self._index_lock_path):
                    yield
            finally:
                held.discard(key)

    def _validate_bundle_structure(self) -> None:
        """校验 Bundle 根目录、固定子目录和锁路径不含符号链接。"""

        if self.root.is_symlink() or not self.root.is_dir():
            raise ValueError("运行 Bundle 目录不存在")
        root_resolved = self.root.resolve(strict=True)
        try:
            root_resolved.relative_to(self.work_dir)
        except ValueError as error:
            raise ValueError("运行 Bundle 目录越出工作目录") from error
        fixed_directories = (
            ".locks",
            "stages",
            "artifacts",
            "internal",
            "internal/state",
            "internal/audit",
            "internal/audit/model_calls",
            "internal/catalog",
        )
        for relative in fixed_directories:
            target = self.root / relative
            current = self.root
            for part in PurePosixPath(relative).parts:
                current = current / part
                if current.is_symlink() or not current.is_dir():
                    raise ValueError(f"运行 Bundle 固定目录非法: {relative}")
                try:
                    current.resolve(strict=True).relative_to(root_resolved)
                except (OSError, ValueError) as error:
                    raise ValueError(f"运行 Bundle 固定目录越出根目录: {relative}") from error
            if target.is_symlink() or not target.is_dir():
                raise ValueError(f"运行 Bundle 固定目录非法: {relative}")
        lock_paths = (self._bundle_lock_path,)
        if not self._legacy_read_only:
            # 正式 Bundle 的状态写入会更新命名空间索引，因此必须有索引锁。
            # 历史 locator 只读固定 JSON，不应因旧数据没有索引锁而无法读取。
            lock_paths += (self._index_lock_path,)
        for lock_path in lock_paths:
            parent = lock_path.parent
            if (
                parent.is_symlink()
                or not parent.is_dir()
                or lock_path.is_symlink()
                or not lock_path.is_file()
            ):
                raise ValueError("运行 Bundle 锁路径非法")

    def _validate_fixed_documents(self) -> None:
        """校验固定 JSON、阶段顺序、阶段摘要和普通文件属性。"""

        run_document, status_document, summary_document = self._read_fixed_documents()
        self._validate_fixed_document_identities(
            run_document, status_document, summary_document
        )
        self._validate_run_layout_document(run_document)
        self._validate_summary_layout_document(summary_document)
        self._validate_stage_file_set()
        for stage in self.stage_names:
            self._read_valid_stage_document(stage)

    def _read_fixed_documents(
        self,
    ) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
        """读取三个固定 JSON，并确认它们都是对象。"""

        paths = (self.run_path, self.status_path, self.summary_path)
        if any(path.is_symlink() or not path.is_file() for path in paths):
            raise ValueError("运行 Bundle 固定 JSON 缺失")
        documents = tuple(self._read_json(path) for path in paths)
        if any(not isinstance(document, Mapping) for document in documents):
            raise ValueError("运行 Bundle 固定 JSON 必须是对象")
        return documents  # type: ignore[return-value]

    def _validate_fixed_document_identities(
        self,
        *documents: Mapping[str, Any],
    ) -> None:
        """校验固定 JSON 的 schema 版本和运行身份一致。"""

        for document in documents:
            if document.get("schema_version") != RUN_BUNDLE_SCHEMA_VERSION:
                raise ValueError("运行 Bundle schema_version 不一致")
            identity_field = self.identity_field
            if document.get(identity_field) != self.research_task_id:
                raise ValueError(
                    f"运行 Bundle {identity_field} 不一致"
                )
            # 新 Bundle 不应把旧 run_id 当作第二个公开身份；历史 Bundle 仍按
            # 原字段读取，避免把兼容定位误认为新任务写入。
            if self.is_research_task and "run_id" in document:
                raise ValueError("research_task Bundle 不得包含 run_id")
            if "source_task_sequence" in document:
                _validate_source_task_sequence(document["source_task_sequence"])

    def _validate_run_layout_document(self, run_document: Mapping[str, Any]) -> None:
        """校验 run.json 的阶段布局，兼容缺失布局字段的 legacy Bundle。"""

        if "stage_layout" in run_document:
            if run_document.get("stage_layout") != self.stage_layout:
                raise ValueError("run.json stage_layout 与实例冻结布局不一致")
            self._validate_stage_order(
                run_document.get("stage_order"),
                self.stage_names,
                document_name="run.json",
            )
            return
        if "stage_order" in run_document:
            self._validate_stage_order(
                run_document.get("stage_order"),
                LEGACY_STAGE_NAMES,
                document_name="run.json",
            )
        if self.stage_layout != LEGACY_STAGE_LAYOUT:
            raise ValueError("缺少布局身份的 Bundle 只能是 legacy")

    def _validate_summary_layout_document(
        self,
        summary_document: Mapping[str, Any],
    ) -> None:
        """校验 summary.json 的阶段顺序和阶段集合。"""

        self._validate_stage_order(
            summary_document.get("stage_order"),
            self.stage_names,
            document_name="summary.json",
        )
        summary_stages = summary_document.get("stages")
        if not isinstance(summary_stages, Mapping):
            raise ValueError("summary.json stages 必须是对象")
        if set(summary_stages) != set(self.stage_names):
            raise ValueError("summary.json stages 与冻结阶段布局不一致")
        for stage, stage_record in summary_stages.items():
            if not isinstance(stage_record, Mapping):
                raise ValueError(f"summary.json 阶段记录必须是对象: {stage}")
            if "source_task_sequence" in stage_record:
                _validate_source_task_sequence(stage_record["source_task_sequence"])

    def _read_valid_stage_document(self, stage: str) -> dict[str, Any]:
        """读取并校验一个固定阶段文件，拒绝错绑或软链接文件。"""

        target = self._locate_stage_document(stage)
        document = self._read_json(target)
        if not isinstance(document, Mapping):
            raise ValueError(f"阶段固定 JSON 必须是对象: {stage}")
        self._validate_stage_document_identity(document, stage)
        self._validate_stage_document_values(document, stage)
        return dict(document)

    def _locate_stage_document(self, stage: str) -> Path:
        """定位冻结布局中的阶段文件，并拒绝软链接或非普通文件。"""

        if stage not in self.stage_names:
            raise ValueError(f"未知运行阶段: {stage}")
        target = self.root / "stages" / f"{stage}.json"
        if target.is_symlink() or not target.is_file():
            raise ValueError(f"阶段固定 JSON 缺失或不是普通文件: {stage}")
        return target

    def _validate_stage_document_identity(
        self,
        document: Mapping[str, Any],
        stage: str,
    ) -> None:
        """校验阶段文件的 schema、运行身份、名称和状态。"""

        if document.get("schema_version") != RUN_BUNDLE_SCHEMA_VERSION:
            raise ValueError(f"阶段 schema_version 非法: {stage}")
        identity_field = self.identity_field
        if document.get(identity_field) != self.research_task_id:
            raise ValueError(f"阶段 {identity_field} 不一致: {stage}")
        if self.is_research_task and "run_id" in document:
            raise ValueError(f"阶段不得包含 run_id: {stage}")
        if document.get("stage") != stage:
            raise ValueError(f"阶段身份不一致: {stage}")
        if document.get("status") not in self._stage_statuses:
            raise ValueError(f"阶段 status 非法: {stage}")

    def _validate_stage_document_values(
        self,
        document: Mapping[str, Any],
        stage: str,
    ) -> None:
        """校验阶段数组、摘要对象和规范化恢复字段。"""

        if "source_task_sequence" in document:
            _validate_source_task_sequence(document["source_task_sequence"])

        for field in ("entries", "artifacts"):
            if not isinstance(document.get(field), list):
                raise ValueError(f"阶段 {field} 必须是数组: {stage}")
        if any(not isinstance(entry, Mapping) for entry in document["entries"]):
            raise ValueError(f"阶段 entries 必须由对象组成: {stage}")
        if not isinstance(document.get("summary"), Mapping):
            raise ValueError(f"阶段 summary 必须是对象: {stage}")
        if "source_task_sequence" in document["summary"]:
            _validate_source_task_sequence(document["summary"]["source_task_sequence"])
        recovery = document.get("recovery")
        if not isinstance(recovery, Mapping):
            raise ValueError(f"阶段 recovery 必须是对象: {stage}")
        normalized_recovery = self._normalize_recovery(recovery)
        if not isinstance(recovery.get("available"), bool):
            raise ValueError(f"阶段 recovery.available 必须是布尔值: {stage}")
        expected_recovery_fields = {
            "available",
            "waiting_investigation_jobs" if self.is_research_task else "waiting_executions",
            "pauses",
        }
        if set(recovery) != expected_recovery_fields:
            raise ValueError(f"阶段 recovery 字段必须固定为三项: {stage}")
        if dict(recovery) != normalized_recovery:
            raise ValueError(f"阶段 recovery 字段值不一致: {stage}")

    def _validate_parent_chain(self, *, require_existing: bool) -> None:
        """确认当前输出命名空间父链是工作目录内的普通目录。"""

        for directory in (self.outputs_root, self.bundle_namespace_root):
            if directory.is_symlink():
                raise ValueError("运行输出父目录不得是符号链接")
            if not directory.exists():
                if require_existing:
                    raise ValueError("恢复所需运行输出目录不存在")
                continue
            if not directory.is_dir():
                raise ValueError("运行输出父路径不是目录")
            try:
                directory.resolve(strict=True).relative_to(self.work_dir)
            except (OSError, ValueError) as error:
                raise ValueError("运行输出目录越出工作目录") from error

    def _create_new_bundle(
        self,
        *,
        scientific_question: str | None,
        task_config: Mapping[str, Any] | None,
        model_configuration: Mapping[str, Any] | None,
    ) -> None:
        """创建新 Bundle、固定说明和稳定 JSON 字段。"""

        if not self.is_research_task:
            raise RuntimeError(
                "历史 Bundle 只能由显式 legacy 定位器只读打开，不能在生产代码中创建"
            )
        self._validate_parent_chain(require_existing=False)
        if self.root.exists():
            raise ValueError("运行 Bundle 已存在，不能覆盖历史运行")
        self.outputs_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.bundle_namespace_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root.mkdir(parents=True, exist_ok=False, mode=0o700)
        for relative in (
            ".locks",
            "stages",
            "artifacts",
            "internal",
            "internal/state",
            "internal/audit",
            "internal/audit/model_calls",
            "internal/catalog",
        ):
            (self.root / relative).mkdir(
                parents=True, exist_ok=False, mode=0o700
            )
        for lock_path in (self._bundle_lock_path, self._index_lock_path):
            descriptor = _open_lock_file(lock_path)
            os.close(descriptor)
        self._validate_bundle_structure()
        self._write_fixed_readmes()
        for stage_name in self.stage_names:
            self._atomic_write_json(
                self.root / "stages" / f"{stage_name}.json",
                {
                    "schema_version": RUN_BUNDLE_SCHEMA_VERSION,
                    **self._identity_fields(),
                    "stage": stage_name,
                    "status": "not_started",
                    "started_at": None,
                    "finished_at": None,
                    "summary": {},
                    "entries": [],
                    "artifacts": [],
                    "blocking": None,
                    "recovery": self._empty_recovery(),
                },
            )
        created_at = utc_timestamp()
        run_document = {
            "schema_version": RUN_BUNDLE_SCHEMA_VERSION,
            **self._identity_fields(),
            "created_at": created_at,
            "updated_at": created_at,
            "stage_layout": self.stage_layout,
            "stage_order": list(self.stage_names),
            "scientific_question": scientific_question,
            "persistence_mode": None,
            "task_config": _copy_json(task_config or {}),
            "model_configuration": _copy_json(model_configuration or {}),
            "source_locations": None,
            "state_database": None,
            "paths": {
                "research_task_root" if self.is_research_task else "run_root": (
                    f"outputs/{self.storage_namespace}/{self.research_task_id}"
                ),
                "status": "status.json",
                "summary": "summary.json",
                "stages": "stages",
                "artifacts": "artifacts",
                "state": "internal/state",
                "audit": "internal/audit",
                "catalog": "internal/catalog",
            },
        }
        status_document = self._base_status(created_at)
        summary_document = self._base_summary(created_at, scientific_question)
        self._atomic_write_json(self.run_path, run_document)
        self._atomic_write_json(self.status_path, status_document)
        self._atomic_write_json(self.summary_path, summary_document)
        self._update_machine_indexes(
            status="running",
            created_at=created_at,
            updated_at=created_at,
            state_database=None,
        )

    def _load_existing_bundle(self) -> None:
        """只读取已有 Bundle，拒绝缺文件时偷偷创建新运行。"""

        self._validate_parent_chain(require_existing=True)
        self._validate_bundle_structure()
        with self._bundle_guard():
            self._freeze_stage_layout_from_run_document()
            self._validate_fixed_documents()
            status_document = self._read_json(self.status_path)
            if not isinstance(status_document, Mapping):  # 已由校验覆盖，保留失败边界
                raise ValueError("status.json 不是对象")
            now = utc_timestamp()
            status = dict(status_document)
            events = status.get("events")
            if not isinstance(events, list):
                events = []
                status["events"] = events
            events.append({"event": "resumed", "at": now, "reason_code": None})
            status["updated_at"] = now
            status["resume_count"] = self._nonnegative_int(status.get("resume_count")) + 1
            self._atomic_write_json(self.status_path, status)
            self._update_machine_indexes(
                status=status.get("status") if isinstance(status.get("status"), str) else None,
                updated_at=now,
            )

    @staticmethod
    def _nonnegative_int(value: Any) -> int:
        """读取稳定整数字段，损坏值按零处理后由当前写入修复。"""

        return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0

    def _base_status(self, timestamp: str) -> dict[str, Any]:
        """返回 status.json 的稳定字段基座。"""

        return {
            "schema_version": RUN_BUNDLE_SCHEMA_VERSION,
            **self._identity_fields(),
            "status": "running",
            "reason_code": None,
            "blocking": None,
            "recovery": self._empty_recovery(),
            "created_at": timestamp,
            "updated_at": timestamp,
            "finished_at": None,
            "resume_count": 0,
            "events": [{"event": "started", "at": timestamp, "reason_code": None}],
        }

    def _base_summary(
        self,
        timestamp: str,
        scientific_question: str | None = None,
    ) -> dict[str, Any]:
        """返回 summary.json 的固定阶段字段基座。"""

        return {
            "schema_version": RUN_BUNDLE_SCHEMA_VERSION,
            **self._identity_fields(),
            "status": "running",
            "updated_at": timestamp,
            "scientific_question": scientific_question,
            "stage_order": list(self.stage_names),
            "stages": {
                name: {
                    "status": "not_started",
                    "path": f"stages/{name}.json",
                    "entry_count": 0,
                    "status_counts": {},
                }
                for name in self.stage_names
            },
            "artifacts": [],
            "blocking": None,
            "recovery": self._empty_recovery(),
            "final_result": None,
        }

    def _write_fixed_readmes(self) -> None:
        """写入各关键层固定模板，不写入运行身份、状态或结果。"""

        templates = {
            self.outputs_root / "README.md": _README_TEXT["outputs"],
            self.bundle_namespace_root / "README.md": _README_TEXT[
                "research_tasks" if self.is_research_task else "runs"
            ],
            self.root / "README.md": _README_TEXT["run"],
            self.root / "stages" / "README.md": _stages_readme_text(self.stage_names),
            self.root / "artifacts" / "README.md": _README_TEXT["artifacts"],
            self.root / "internal" / "README.md": _README_TEXT["internal"],
            self.root / "internal" / "state" / "README.md": _README_TEXT["state"],
            self.root / "internal" / "audit" / "README.md": _README_TEXT["audit"],
            self.root / "internal" / "audit" / "model_calls" / "README.md": _README_TEXT["model_calls"],
            self.root / "internal" / "catalog" / "README.md": _README_TEXT["catalog"],
        }
        for path, text in templates.items():
            if path.is_symlink() or (path.exists() and not path.is_file()):
                raise ValueError("固定 README 必须是普通文件")
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text.rstrip() + "\n", encoding="utf-8")

    # ==================== Bundle 更新 ====================

    def _ensure_writable(self) -> None:
        """拒绝通过历史只读 locator 改写 schema-v10 Bundle。"""

        if self._legacy_read_only:
            raise RuntimeError(
                "历史 Bundle 只读；请使用显式迁移工具写入新的 research_task Bundle"
            )

    def update_status(
        self,
        status: str,
        *,
        reason_code: str | None = None,
        blocking: Mapping[str, Any] | None = None,
        recovery: Mapping[str, Any] | None = None,
        source_task_sequence: int | None = None,
        display: Mapping[str, Any] | None = None,
    ) -> bool:
        """原子更新运行状态并写入机器索引。

        ``source_task_sequence`` 和 ``display`` 是 investigation_v2 展示投影的
        可选字段。legacy 调用不传它们，因此保持原有 JSON 合同和状态词。
        """

        self._ensure_writable()
        if not isinstance(status, str) or status not in self._run_statuses:
            raise ValueError(f"运行状态非法: {status!r}")
        if source_task_sequence is not None:
            source_task_sequence = _validate_source_task_sequence(source_task_sequence)
        if display is not None and not isinstance(display, Mapping):
            raise TypeError("display 必须是对象")
        with self._bundle_guard():
            current = self._read_json(self.status_path)
            if not isinstance(current, Mapping):
                raise ValueError("status.json 不是对象")
            now = utc_timestamp()
            document = dict(current)
            document.update(
                {
                    "schema_version": RUN_BUNDLE_SCHEMA_VERSION,
                    **self._identity_fields(),
                    "status": status,
                    "reason_code": reason_code,
                    "blocking": _copy_json(blocking) if blocking is not None else None,
                    "recovery": (
                        self._empty_recovery()
                        if status
                        in {"succeeded", "failed", "stopped_no_supported_hypothesis"}
                        else self._normalize_recovery(recovery)
                    ),
                    "updated_at": now,
                    "finished_at": now if status in {"succeeded", "failed", "stopped_no_supported_hypothesis"} else None,
                }
            )
            if display is not None:
                document.update(_copy_json(display))
            if source_task_sequence is not None:
                document["source_task_sequence"] = source_task_sequence
            self._validate_document_identity(document, document_name="status.json")
            events = document.get("events")
            if not isinstance(events, list):
                events = []
                document["events"] = events
            event_name = "finished" if document["finished_at"] is not None else status
            events.append({"event": event_name, "at": now, "reason_code": reason_code})
            self._atomic_write_json(self.status_path, document)
            self._update_summary_unlocked(
                result={
                    "status": status,
                    "blocking": document["blocking"],
                    "recovery": document["recovery"],
                },
                source_task_sequence=source_task_sequence,
                display=display,
            )
            self._update_machine_indexes(status=status, updated_at=now)
            return True

    def update_stage_status(
        self,
        stage: str,
        status: str,
        *,
        summary: Mapping[str, Any] | None = None,
        blocking: Mapping[str, Any] | None = None,
        recovery: Mapping[str, Any] | None = None,
        source_task_sequence: int | None = None,
    ) -> bool:
        """更新固定阶段的即时状态，并保留已有 entries。

        V2 投影可通过 ``source_task_sequence`` 让阶段文件和根摘要引用同一
        Working Memory 事件位置；不传时保持 legacy 行为。
        """

        self._ensure_writable()
        if stage not in self.stage_names:
            raise ValueError(f"未知运行阶段: {stage}")
        if not isinstance(status, str) or status not in self._stage_statuses:
            raise ValueError(f"阶段状态非法: {status!r}")
        if source_task_sequence is not None:
            source_task_sequence = _validate_source_task_sequence(source_task_sequence)
        with self._bundle_guard():
            target = self.root / "stages" / f"{stage}.json"
            current = self._read_valid_stage_document(stage)
            now = utc_timestamp()
            document = self._build_stage_status_document(
                current=current,
                stage=stage,
                status=status,
                now=now,
                blocking=blocking,
                recovery=recovery,
            )
            if summary is not None:
                if not isinstance(summary, Mapping):
                    raise TypeError("阶段即时摘要必须是对象")
                copied_summary = _copy_json(summary)
                if source_task_sequence is not None:
                    copied_summary["source_task_sequence"] = source_task_sequence
                document["summary"] = copied_summary
            if source_task_sequence is not None:
                document["source_task_sequence"] = source_task_sequence
            entries = document.get("entries")
            document["entries"] = list(entries) if isinstance(entries, list) else []
            self._atomic_write_json(target, document)

            summary_document = self._read_json(self.summary_path)
            if not isinstance(summary_document, Mapping):
                raise ValueError("summary.json 不是对象")
            updated_summary = self._update_stage_summary_document(
                summary_document=summary_document,
                stage=stage,
                status=status,
                entries=document["entries"],
                summary=summary,
                now=now,
                source_task_sequence=source_task_sequence,
            )
            self._atomic_write_json(self.summary_path, updated_summary)
            return True

    def _build_stage_status_document(
        self,
        *,
        current: Mapping[str, Any],
        stage: str,
        status: str,
        now: str,
        blocking: Mapping[str, Any] | None,
        recovery: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """生成阶段状态文档，保留已有 entries 和摘要。"""

        document = dict(current)
        document.update(
            {
                "schema_version": RUN_BUNDLE_SCHEMA_VERSION,
                **self._identity_fields(),
                "stage": stage,
                "status": status,
                "started_at": self._stage_started_at(current, status, now),
                "finished_at": now if status in _TERMINAL_STAGE_STATUSES else None,
                "blocking": _copy_json(blocking) if blocking is not None else None,
                "recovery": (
                    self._empty_recovery()
                    if status in _TERMINAL_STAGE_STATUSES
                    else self._normalize_recovery(recovery)
                ),
            }
        )
        entries = document.get("entries")
        document["entries"] = list(entries) if isinstance(entries, list) else []
        return document

    @staticmethod
    def _stage_started_at(
        current: Mapping[str, Any],
        status: str,
        now: str,
    ) -> Any:
        """按阶段状态决定是否首次记录 started_at。"""

        if status in {"not_started", "disabled", "not_applicable"}:
            return current.get("started_at")
        return current.get("started_at") or now

    def _update_stage_summary_document(
        self,
        *,
        summary_document: Mapping[str, Any],
        stage: str,
        status: str,
        entries: list[Any],
        summary: Mapping[str, Any] | None,
        now: str,
        source_task_sequence: int | None = None,
    ) -> dict[str, Any]:
        """同步 summary.json 中一个阶段的计数和最终摘要。"""

        updated_summary = deepcopy(dict(summary_document))
        stages = updated_summary.setdefault("stages", {})
        stage_record = stages.get(stage)
        stage_record = dict(stage_record) if isinstance(stage_record, Mapping) else {}
        stage_record.update(
            {
                "status": status,
                "path": f"stages/{stage}.json",
                "entry_count": len(entries),
                "status_counts": self._count_stage_entry_statuses(entries),
            }
        )
        if source_task_sequence is not None:
            source_task_sequence = _validate_source_task_sequence(
                source_task_sequence
            )
            updated_summary["source_task_sequence"] = source_task_sequence
        if summary is not None:
            copied_summary = _copy_json(summary)
            if source_task_sequence is not None:
                copied_summary["source_task_sequence"] = source_task_sequence
            stage_record["summary"] = copied_summary
        elif source_task_sequence is not None:
            existing_summary = stage_record.get("summary")
            if isinstance(existing_summary, Mapping):
                copied_summary = _copy_json(existing_summary)
                copied_summary["source_task_sequence"] = source_task_sequence
                stage_record["summary"] = copied_summary
        if source_task_sequence is not None:
            stage_record["source_task_sequence"] = source_task_sequence
        stages[stage] = stage_record
        if stage in {"final_result", "task_review_and_result"} and summary is not None:
            updated_summary["final_result"] = _copy_json(summary)
        updated_summary["updated_at"] = now
        return updated_summary

    @staticmethod
    def _count_stage_entry_statuses(entries: list[Any]) -> dict[str, int]:
        """统计阶段 entries 中的字符串状态。"""

        counts: dict[str, int] = {}
        for entry in entries:
            if isinstance(entry, Mapping) and isinstance(entry.get("status"), str):
                status = entry["status"]
                counts[status] = counts.get(status, 0) + 1
        return counts

    def update_summary(
        self,
        *,
        result: Mapping[str, Any] | None = None,
        stage: str | None = None,
        stage_summary: Mapping[str, Any] | None = None,
        source_task_sequence: int | None = None,
        display: Mapping[str, Any] | None = None,
    ) -> bool:
        """在 Bundle 锁内更新总摘要或单个阶段摘要。

        V2 展示投影可以同时保存来源事件序号和生命周期观察字段；默认参数
        保持旧 Bundle 的写入格式。
        """

        self._ensure_writable()
        if source_task_sequence is not None:
            source_task_sequence = _validate_source_task_sequence(source_task_sequence)
        if display is not None and not isinstance(display, Mapping):
            raise TypeError("display 必须是对象")

        with self._bundle_guard():
            return self._update_summary_unlocked(
                result=result,
                stage=stage,
                stage_summary=stage_summary,
                source_task_sequence=source_task_sequence,
                display=display,
            )

    def _update_summary_unlocked(
        self,
        *,
        result: Mapping[str, Any] | None = None,
        stage: str | None = None,
        stage_summary: Mapping[str, Any] | None = None,
        source_task_sequence: int | None = None,
        display: Mapping[str, Any] | None = None,
    ) -> bool:
        """更新总摘要或单个阶段摘要，缺失字段保持固定基座。"""

        document = self._read_json(self.summary_path)
        if not isinstance(document, Mapping):
            raise ValueError("summary.json 不是对象")
        updated = deepcopy(dict(document))
        if result is not None:
            self._apply_result_to_summary(updated, result)
        if stage is not None:
            self._apply_stage_to_summary(
                updated,
                stage=stage,
                stage_summary=stage_summary,
                source_task_sequence=source_task_sequence,
            )
        if display is not None:
            if not isinstance(display, Mapping):
                raise TypeError("display 必须是对象")
            updated.update(_copy_json(display))
        if source_task_sequence is not None:
            updated["source_task_sequence"] = _validate_source_task_sequence(
                source_task_sequence
            )
        self._validate_document_identity(updated, document_name="summary.json")
        updated["updated_at"] = utc_timestamp()
        self._atomic_write_json(self.summary_path, updated)
        return True

    def _apply_result_to_summary(
        self,
        summary: dict[str, Any],
        result: Mapping[str, Any],
    ) -> None:
        """把运行结果字段合并到总摘要，并规范化 recovery。"""

        if not isinstance(result, Mapping):
            raise TypeError("运行结果必须是对象")
        status = result.get("status", summary.get("status"))
        if "status" in result:
            summary["status"] = result.get("status")
            summary["recovery"] = (
                self._empty_recovery()
                if status
                in {"succeeded", "failed", "stopped_no_supported_hypothesis"}
                else self._normalize_recovery(result.get("recovery"))
            )
        if "scientific_question" in result:
            summary["scientific_question"] = _copy_json(result.get("scientific_question"))
        for key in ("blocking", "recovery", "final_result", "artifacts"):
            if key not in result:
                continue
            summary[key] = (
                self._empty_recovery()
                if key == "recovery"
                and status
                in {"succeeded", "failed", "stopped_no_supported_hypothesis"}
                else self._normalize_recovery(result.get(key))
                if key == "recovery"
                else _copy_json(result.get(key))
            )

    def _apply_stage_to_summary(
        self,
        summary: dict[str, Any],
        *,
        stage: str,
        stage_summary: Mapping[str, Any] | None,
        source_task_sequence: int | None = None,
    ) -> None:
        """从固定阶段文件更新 summary 中的状态、计数和摘要。"""

        if stage not in self.stage_names:
            raise ValueError(f"未知运行阶段: {stage}")
        stages = summary.setdefault("stages", {})
        current = stages.get(stage)
        current = (
            dict(current)
            if isinstance(current, Mapping)
            else {
                "status": "not_started",
                "path": f"stages/{stage}.json",
                "entry_count": 0,
                "status_counts": {},
            }
        )
        stage_path = self.root / "stages" / f"{stage}.json"
        stage_document = self._read_json(stage_path) if stage_path.is_file() else {}
        entries = (
            stage_document.get("entries")
            if isinstance(stage_document, Mapping)
            else []
        )
        entries = entries if isinstance(entries, list) else []
        stage_status = (
            stage_document.get("status")
            if isinstance(stage_document, Mapping)
            else None
        )
        current["path"] = f"stages/{stage}.json"
        current["status"] = (
            stage_status
            if isinstance(stage_status, str) and stage_status in self._stage_statuses
            else current.get("status", "not_started")
        )
        if stage_summary is not None:
            if not isinstance(stage_summary, Mapping):
                raise TypeError("阶段总摘要必须是对象")
            copied_stage_summary = _copy_json(stage_summary)
            if source_task_sequence is not None:
                copied_stage_summary["source_task_sequence"] = (
                    _validate_source_task_sequence(source_task_sequence)
                )
            current["summary"] = copied_stage_summary
        current["entry_count"] = len(entries)
        current["status_counts"] = self._count_stage_entry_statuses(entries)
        if source_task_sequence is not None:
            source_task_sequence = _validate_source_task_sequence(
                source_task_sequence
            )
            current["source_task_sequence"] = source_task_sequence
            existing_summary = current.get("summary")
            if isinstance(existing_summary, Mapping):
                copied_summary = _copy_json(existing_summary)
                copied_summary["source_task_sequence"] = source_task_sequence
                current["summary"] = copied_summary
        stages[stage] = current

    def write_stage_summary(
        self,
        stage: str,
        summary: Mapping[str, Any],
        *,
        source_file: str | None = None,
        call_identity: Mapping[str, Any] | None = None,
        source_task_sequence: int | None = None,
    ) -> Path:
        """在 Bundle 锁内追加或更新固定阶段摘要。

        V2 调用方可把阶段摘要绑定到读取到的 WM 事件序号；阶段 entry、阶段
        根文档和 ``summary.json`` 会保存同一个序号。
        """

        self._ensure_writable()
        if source_task_sequence is not None:
            source_task_sequence = _validate_source_task_sequence(source_task_sequence)

        with self._bundle_guard():
            return self._write_stage_summary_unlocked(
                stage,
                summary,
                source_file=source_file,
                call_identity=call_identity,
                source_task_sequence=source_task_sequence,
            )

    def _write_stage_summary_unlocked(
        self,
        stage: str,
        summary: Mapping[str, Any],
        *,
        source_file: str | None = None,
        call_identity: Mapping[str, Any] | None = None,
        source_task_sequence: int | None = None,
    ) -> Path:
        """将一条精简阶段摘要追加或更新到固定阶段文件。"""

        if stage not in self.stage_names:
            raise ValueError(f"未知运行阶段: {stage}")
        if not isinstance(summary, Mapping):
            raise TypeError("阶段摘要必须是对象")
        target = self.root / "stages" / f"{stage}.json"
        document = self._read_valid_stage_document(stage)
        entries = document.get("entries")
        entries = list(entries) if isinstance(entries, list) else []
        copied_identity = _copy_json(call_identity) if isinstance(call_identity, Mapping) else None
        copied_summary = _copy_json(summary)
        if source_task_sequence is not None:
            copied_summary["source_task_sequence"] = source_task_sequence
        entry = {
            "status": str(summary.get("status") or "completed"),
            "source_file": source_file,
            "call_identity": copied_identity,
            "summary": copied_summary,
        }
        entry_key = json.dumps(
            {"source_file": source_file, "call_identity": copied_identity},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        replaced = False
        for index, existing_entry in enumerate(entries):
            if not isinstance(existing_entry, Mapping):
                continue
            existing_key = json.dumps(
                {
                    "source_file": existing_entry.get("source_file"),
                    "call_identity": existing_entry.get("call_identity"),
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            if existing_key == entry_key:
                entries[index] = entry
                replaced = True
                break
        if not replaced:
            entries.append(entry)
        entry_statuses = {
            item.get("status")
            for item in entries
            if isinstance(item, Mapping) and isinstance(item.get("status"), str)
        }
        # 并发模型阶段的完成事件按实际结束顺序写入。只看最后一条会让后续
        # validated 覆盖此前已经发生的 failed，造成“没有 Top5 但阶段完成”的
        # 假成功。失败是阶段不可忽略的事实，必须在同一阶段摘要中保持失败状态。
        if entry_statuses.intersection({"failed", "error"}):
            stage_status = "failed"
        else:
            entry_status = entry["status"]
            active_statuses = (
                {"open", "running", "waiting_investigation_job", "paused"}
                if self.is_research_task
                else {"open", "running", "waiting_execution", "paused"}
            )
            if entry_status in active_statuses:
                stage_status = entry_status
            elif entry_status in {"disabled", "not_applicable", "skipped"}:
                stage_status = entry_status
            else:
                stage_status = "completed"
        document.update(
            {
                "schema_version": RUN_BUNDLE_SCHEMA_VERSION,
                **self._identity_fields(),
                "stage": stage,
                "status": stage_status if entries else "not_started",
                "finished_at": (
                    utc_timestamp()
                    if stage_status in _TERMINAL_STAGE_STATUSES
                    else None
                ),
                "entries": entries,
                "artifacts": document.get("artifacts") if isinstance(document.get("artifacts"), list) else [],
                "blocking": document.get("blocking"),
                "recovery": self._normalize_recovery(document.get("recovery")),
            }
        )
        if source_task_sequence is not None:
            document["source_task_sequence"] = source_task_sequence
        self._atomic_write_json(target, document)
        self.update_summary(
            stage=stage,
            stage_summary=copied_summary,
            source_task_sequence=source_task_sequence,
        )
        return target

    def register_state_database(self, path: str | Path) -> str:
        """在 Bundle 锁内登记 Memory SQLite 的相对路径。"""

        self._ensure_writable()
        with self._bundle_guard():
            return self._register_state_database_unlocked(path)

    def _register_state_database_unlocked(self, path: str | Path) -> str:
        """登记 Memory SQLite 的 Bundle 相对路径并刷新机器索引。"""

        raw_target = Path(path).expanduser()
        if raw_target.is_symlink():
            raise ValueError("Memory 数据库路径不得是符号链接")
        target = raw_target.resolve(strict=False)
        if not target.is_file():
            raise ValueError("Memory 数据库必须是已存在的普通文件")
        try:
            relative = target.relative_to(self.root).as_posix()
        except ValueError as error:
            raise ValueError("Memory 数据库必须位于当前 Bundle") from error
        if relative.startswith("internal/state/") is False:
            raise ValueError("Memory 数据库必须位于 internal/state")
        current = self.root
        for part in PurePosixPath(relative).parts:
            current = current / part
            if current.is_symlink():
                raise ValueError("Memory 数据库路径不得经过符号链接")
        run_document = self._read_json(self.run_path)
        if not isinstance(run_document, Mapping):
            raise ValueError("run.json 不是对象")
        updated = dict(run_document)
        updated["state_database"] = relative
        updated["updated_at"] = utc_timestamp()
        self._atomic_write_json(self.run_path, updated)
        self._update_machine_indexes(state_database=relative, updated_at=updated["updated_at"])
        return relative

    def update_run_metadata(
        self,
        *,
        persistence_mode: str | None = None,
        task_config: Mapping[str, Any] | None = None,
        model_configuration: Mapping[str, Any] | None = None,
        source_locations: Mapping[str, Any] | None = None,
    ) -> bool:
        """在 Bundle 锁内补充运行配置。"""

        self._ensure_writable()
        with self._bundle_guard():
            return self._update_run_metadata_unlocked(
                persistence_mode=persistence_mode,
                task_config=task_config,
                model_configuration=model_configuration,
                source_locations=source_locations,
            )

    def _update_run_metadata_unlocked(
        self,
        *,
        persistence_mode: str | None = None,
        task_config: Mapping[str, Any] | None = None,
        model_configuration: Mapping[str, Any] | None = None,
        source_locations: Mapping[str, Any] | None = None,
    ) -> bool:
        """补充运行配置，不改变运行身份和创建时间。"""

        document = self._read_json(self.run_path)
        if not isinstance(document, Mapping):
            raise ValueError("run.json 不是对象")
        updated = dict(document)
        if persistence_mode is not None:
            updated["persistence_mode"] = persistence_mode
        if task_config is not None:
            updated["task_config"] = _copy_json(task_config)
        if model_configuration is not None:
            updated["model_configuration"] = _copy_json(model_configuration)
        if source_locations is not None:
            updated["source_locations"] = _copy_json(source_locations)
        updated["updated_at"] = utc_timestamp()
        self._atomic_write_json(self.run_path, updated)
        self._update_machine_indexes(updated_at=updated["updated_at"])
        return True

    def locate_research_task_state_database(self) -> Path | None:
        """读取正式 research_task Bundle 登记的 Memory 数据库路径。"""

        if not self.is_research_task:
            raise RuntimeError("历史 Bundle 请使用 locate_legacy_state_database")

        with self._bundle_guard():
            return self._locate_state_database_unlocked()

    def locate_legacy_state_database(self) -> Path | None:
        """读取历史 schema-v10 Bundle 登记的 Memory 数据库路径。"""

        if self.is_research_task:
            raise RuntimeError("research_task Bundle 请使用 locate_research_task_state_database")

        with self._bundle_guard():
            return self._locate_state_database_unlocked()

    def _locate_state_database_unlocked(self) -> Path | None:
        """在 Bundle 已加锁时读取登记的 Memory 数据库路径。"""

        document = self._read_json(self.run_path)
        if not isinstance(document, Mapping):
            raise ValueError("run.json 不是对象")
        relative = document.get("state_database")
        if relative is None:
            return None
        if (
            not isinstance(relative, str)
            or not relative.startswith("internal/state/")
            or "\\" in relative
        ):
            raise ValueError("run.json state_database 必须是 internal/state 下的相对路径")
        relative_path = PurePosixPath(relative)
        if relative_path.is_absolute() or any(part in {"", ".", ".."} for part in relative_path.parts):
            raise ValueError("run.json state_database 路径非法")
        current = self.root
        for part in relative_path.parts:
            current = current / part
            if current.is_symlink():
                raise ValueError("state_database 路径不得经过符号链接")
        target = current.resolve(strict=False)
        try:
            target.relative_to(self.root.resolve(strict=False))
        except ValueError as error:
            raise ValueError("state_database 越出 Bundle") from error
        state_root = (self.root / "internal" / "state").resolve(strict=False)
        try:
            target.relative_to(state_root)
        except ValueError as error:
            raise ValueError("state_database 不在 internal/state") from error
        if target.is_symlink() or not target.is_file():
            raise ValueError("state_database 必须是已存在的普通文件")
        return target

    # ==================== 机器索引与文件工具 ====================

    def _update_machine_indexes(
        self,
        *,
        status: str | None = None,
        created_at: str | None = None,
        updated_at: str | None = None,
        state_database: str | None = None,
    ) -> None:
        """在当前输出命名空间锁内原子更新 index.json 和 latest.json。"""

        self._ensure_writable()
        if any(
            path.is_symlink() or (path.exists() and not path.is_file())
            for path in (self.index_path, self.latest_path)
        ):
            raise ValueError("运行机器索引不得是符号链接")
        with self._index_guard():
            self._update_machine_indexes_unlocked(
                status=status,
                created_at=created_at,
                updated_at=updated_at,
                state_database=state_database,
            )

    def _update_machine_indexes_unlocked(
        self,
        *,
        status: str | None = None,
        created_at: str | None = None,
        updated_at: str | None = None,
        state_database: str | None = None,
    ) -> None:
        """原子更新当前输出命名空间的 index.json 和 latest.json。"""

        self.outputs_root.mkdir(parents=True, exist_ok=True)
        existing = self._read_json(self.index_path) if self.index_path.is_file() else None
        now = updated_at or utc_timestamp()
        entries = self._read_index_entries(existing)
        previous_record = self._find_index_entry_record(entries)
        record = self._build_index_entry_record(
            previous_record=previous_record,
            status=status,
            created_at=created_at,
            updated_at=now,
            state_database=state_database,
        )
        normalized_entries = self._replace_index_entries(entries, record)
        index_document = {
            "schema_version": RUN_BUNDLE_SCHEMA_VERSION,
            "updated_at": now,
            self._index_entries_field: normalized_entries,
        }
        latest_document = {
            "schema_version": RUN_BUNDLE_SCHEMA_VERSION,
            **self._identity_fields(),
            "path": f"{self.storage_namespace}/{self.research_task_id}",
            "status": record["status"],
            "updated_at": now,
        }
        self._atomic_write_json(self.index_path, index_document)
        self._atomic_write_json(self.latest_path, latest_document)

    @property
    def _index_entries_field(self) -> str:
        """返回当前命名空间的索引数组字段。"""

        return "research_tasks" if self.is_research_task else "runs"

    def _read_index_entries(self, existing: Any) -> list[Any]:
        """从当前命名空间 index.json 读取条目，损坏时回到空索引。"""

        if not isinstance(existing, Mapping):
            return []
        entries = existing.get(self._index_entries_field)
        return entries if isinstance(entries, list) else []

    def _find_index_entry_record(
        self,
        entries: list[Any],
    ) -> Mapping[str, Any] | None:
        """找到当前 Bundle 在旧 index 中的记录。"""

        return next(
            (
                item
                for item in entries
                if isinstance(item, Mapping)
                and item.get(self.identity_field) == self.research_task_id
            ),
            None,
        )

    def _build_index_entry_record(
        self,
        *,
        previous_record: Mapping[str, Any] | None,
        status: str | None,
        created_at: str | None,
        updated_at: str,
        state_database: str | None,
    ) -> dict[str, Any]:
        """按当前更新和旧记录构建一个机器索引记录。"""

        return {
            **self._identity_fields(),
            "path": f"{self.storage_namespace}/{self.research_task_id}",
            "status": (
                status
                if status is not None
                else (
                    previous_record.get("status")
                    if previous_record
                    else "running"
                )
            ),
            "created_at": (
                created_at
                if created_at is not None
                else (previous_record.get("created_at") if previous_record else None)
            ),
            "updated_at": updated_at,
            "state_database": (
                state_database
                if state_database is not None
                else (
                    previous_record.get("state_database")
                    if previous_record
                    else None
                )
            ),
        }

    def _replace_index_entries(
        self,
        entries: list[Any],
        record: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        """替换当前 Bundle 记录，丢弃非法重复项并保持创建时间排序。"""

        replaced = False
        normalized_entries: list[dict[str, Any]] = []
        for item in entries:
            if not isinstance(item, Mapping):
                continue
            if self.is_research_task:
                # 新索引不能把历史 run_id 或其它旧身份字段继续带入；无法
                # 证明是 research_task 条目的旧记录直接丢弃，等待本次写入重建。
                if "run_id" in item and "research_task_id" not in item:
                    continue
                normalized_item = dict(item)
                normalized_item.pop("run_id", None)
            else:
                normalized_item = dict(item)
            if normalized_item.get(self.identity_field) == self.research_task_id:
                if not replaced:
                    normalized_entries.append(dict(record))
                    replaced = True
                continue
            if not isinstance(normalized_item.get(self.identity_field), str):
                continue
            normalized_entries.append(normalized_item)
        if not replaced:
            normalized_entries.append(dict(record))
        normalized_entries.sort(
            key=lambda item: str(
                item.get("created_at")
                or item.get(self.identity_field)
                or ""
            )
        )
        return normalized_entries

    @staticmethod
    def _read_json(path: Path) -> Any:
        """读取 JSON，错误转换为固定 ValueError。"""

        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"无法读取运行 JSON: {path.name}") from error

    @staticmethod
    def _atomic_write_json(path: Path, payload: Any) -> None:
        """使用唯一同目录临时文件和两次 fsync 原子写入 JSON。

        目标和父目录都必须是当前 Bundle 内的普通路径；父目录 fsync 失败会直接
        暴露，避免恢复时把已经替换但目录项尚未落盘的状态误判为可靠。
        """

        RunBundle._validate_atomic_parent_chain(path.parent)
        if path.parent.exists() and not path.parent.is_dir():
            raise ValueError("JSON 父目录必须是普通目录")
        path.parent.mkdir(parents=True, exist_ok=True)
        RunBundle._validate_atomic_parent_chain(path.parent)
        if not path.parent.is_dir():
            raise ValueError("JSON 父目录必须是普通目录")
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError("JSON 目标必须是普通文件")
        temporary: Path | None = None
        try:
            descriptor, name = tempfile.mkstemp(
                prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
            )
            temporary = Path(name)
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            temporary = None
            directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            directory_flags |= getattr(os, "O_CLOEXEC", 0)
            directory_flags |= getattr(os, "O_NOFOLLOW", 0)
            directory_descriptor = os.open(path.parent, directory_flags)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @staticmethod
    def _validate_atomic_parent_chain(parent: Path) -> None:
        """确认原子写入的父链没有符号链接，避免 ancestor TOCTOU 越界。"""

        for ancestor in (parent, *parent.parents):
            if ancestor.is_symlink():
                raise ValueError("JSON 父路径不得是符号链接")


def locate_research_task_bundle(
    work_dir: str | Path,
    research_task_id: str,
) -> RunBundle:
    """只读定位新 ``outputs/research_tasks`` 下的 research_task。"""

    return RunBundle.open_existing_research_task(
        work_dir,
        validate_research_task_id(research_task_id),
    )


def locate_legacy_run_bundle(work_dir: str | Path, run_id: str) -> RunBundle:
    """只读定位历史 ``outputs/runs`` Bundle。

    这是迁移和兼容恢复使用的显式入口；它不会创建 Bundle 或把历史内容写入
    ``outputs/research_tasks``。
    """

    return RunBundle.open_existing_legacy(
        work_dir,
        validate_research_task_id(run_id),
    )


def locate_legacy_state_database(work_dir: str | Path, run_id: str) -> Path | None:
    """只读定位历史 ``outputs/runs`` Bundle 内的 Memory 数据库。"""

    bundle = locate_legacy_run_bundle(work_dir, run_id)
    return bundle.locate_legacy_state_database()


def locate_research_task_state_database(
    work_dir: str | Path,
    research_task_id: str,
) -> Path | None:
    """只读定位 research_task Bundle 内登记的 Memory 数据库。"""

    bundle = locate_research_task_bundle(work_dir, research_task_id)
    return bundle.locate_research_task_state_database()


def relative_research_task_path(research_task_id: str) -> str:
    """返回新 research_task Bundle 的相对路径。"""

    return PurePosixPath(
        "outputs",
        "research_tasks",
        validate_research_task_id(research_task_id),
    ).as_posix()


def relative_legacy_bundle_path(run_id: str) -> str:
    """返回历史兼容 Bundle 的相对路径。"""

    return PurePosixPath(
        "outputs", "runs", validate_research_task_id(run_id)
    ).as_posix()


__all__ = [
    "DEFAULT_STAGE_LAYOUT",
    "INVESTIGATION_STAGE_LAYOUT",
    "INVESTIGATION_STAGE_NAMES",
    "LEGACY_STAGE_LAYOUT",
    "LEGACY_STAGE_NAMES",
    "RESEARCH_TASK_LAYOUT",
    "RESEARCH_TASK_STAGE_LAYOUT",
    "RESEARCH_TASK_STAGE_NAMES",
    "RUN_BUNDLE_SCHEMA_VERSION",
    "RunBundle",
    "STAGE_LAYOUTS",
    "STAGE_NAMES",
    "locate_legacy_run_bundle",
    "locate_research_task_bundle",
    "locate_research_task_state_database",
    "locate_legacy_state_database",
    "relative_legacy_bundle_path",
    "relative_research_task_path",
    "validate_research_task_id",
]
