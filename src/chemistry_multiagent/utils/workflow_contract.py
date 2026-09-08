"""负责把模型生成的工作流变成可安全执行、可稳定比较的正式工作流。

模型只接触便于阅读的短引用，Controller 则持有不会随排序变化的正式身份。
本模块位于两者之间，统一处理结构校验、身份转换、文件边界和修订应用，避免
模型文本未经检查就直接进入工具调用。
"""

from __future__ import annotations

import json
import math
import re
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, MutableMapping, Sequence

from chemistry_multiagent.utils.gaussian_routing import GaussianRoutePolicy

class WorkflowContractError(ValueError):
    """表示工作流内容不完整、不一致，或可能越过任务执行边界。"""


_SHORT_REF_PATTERN = re.compile(r"^[sobn][1-9][0-9]*$")
_WORKFLOW_KEYS = {
    "workflow_name",
    "goal",
    "strategy_name",
    "steps",
}

# MiniChem MCP 的 Action 工作流使用 action/backend/request 三元组；旧的
# ToolSpec 工作流仍由历史测试和只读迁移代码识别，但不再作为生产目录来源。
_MINICHEM_STEP_KEYS = {
    "step_id",
    "step_name",
    "description",
    "action_id",
    "backend_id",
    "request",
    "expected_outputs",
    "depends_on",
}
_MINICHEM_GENERATION_STEP_KEYS = _MINICHEM_STEP_KEYS - {
    "step_id",
    "expected_outputs",
} | {"step_ref"}
_MINICHEM_COMMON_STEP_KEYS = {
    "step_id",
    "step_name",
    "description",
    "request",
    "expected_outputs",
    "depends_on",
}
_MINICHEM_COMMON_GENERATION_STEP_KEYS = {
    "step_ref",
    "step_name",
    "description",
    "request",
    "depends_on",
}
_MINICHEM_JOB_STEP_KEYS = _MINICHEM_COMMON_STEP_KEYS | {"execution_layer"}
_MINICHEM_JOB_GENERATION_STEP_KEYS = _MINICHEM_COMMON_GENERATION_STEP_KEYS | {
    "execution_layer"
}
_MINICHEM_EXECUTION_LAYER_VALUES = {
    "predefined_actions",
    "native_software",
    "programmable_analysis",
}
_MINICHEM_NATIVE_JOB_FIELDS = {
    "software_id",
    "executable",
    "arguments",
    "staged_inputs",
    "stdin_target",
    "resource_limits",
    "calculation_intent",
    "label",
    "parent_job_id",
    "bindings",
    # foreach 是 ARCHE 在 native Gaussian 步骤上的逐项展开修饰符；predefined
    # Action request 也有独立的 fan-out 分支，但两者不会把该字段发送给 MiniChem。
    "foreach",
    # authored_files 是 ARCHE 在提交前使用的内部字段，不会发送给 MiniChem。
    "authored_files",
}
_MINICHEM_ANALYSIS_JOB_FIELDS = {
    "runtime",
    "script_path",
    "script_target",
    "arguments",
    "staged_inputs",
    "required_modules",
    "required_module_versions",
    "required_symbols",
    "inputs",
    "outputs",
    "execution_policy",
    "external_execution",
    "resource_limits",
    "label",
    "parent_job_id",
    "bindings",
    # authored_files 是 ARCHE 在提交前使用的内部字段，不会发送给 MiniChem。
    "authored_files",
}


def _validate_minichem_foreach_execution_layer(
    request: Mapping[str, Any],
    execution_layer: str,
    label: str,
) -> None:
    """核对 foreach 所在执行层与实际 fan-out 执行分支一致。

    输入是已经确认字段形状的 MiniChem request、执行层和错误标签；动作只读取
    ``foreach`` 及 native 软件身份，不修改 request、不解析上游结果。predefined
    Action 由执行器逐项调用同一 Action，Gaussian native 由专用子作业分支处理；
    programmable analysis 和非 Gaussian native 没有对应的逐项提交语义，遇到
    foreach 时抛出工作流合同错误。
    """

    if request.get("foreach") is None:
        return
    if execution_layer == "predefined_actions":
        return
    if execution_layer == "native_software":
        compact_identity = {
            _compact_gaussian_text(request.get("software_id")),
            _compact_gaussian_text(request.get("executable")),
        }
        if any(
            "gaussian" in value or value in {"g03", "g09", "g16", "g98"}
            for value in compact_identity
        ):
            return
    _fail(
        f"{label}.foreach 在开放执行层只允许 native Gaussian 作业；"
        "predefined Action 使用独立 fan-out 分支"
    )


_MINICHEM_CALCULATION_INTENTS = {
    "single_point",
    "geometry_optimization",
    "frequency",
    "optimization_frequency",
    "transition_state",
    "ionic_relaxation",
    "reaction_path",
    "conformer_search",
    "protonation",
    "deprotonation",
    "tautomerization",
    "projection",
    "other",
}

# Agent 只能在任务工作区内写入少量 UTF-8 文本。这里的上限同时保护合同校验、
# 任务证据和 MCP 请求，执行器不会因为目录快照缺字段而放宽这些边界。
MAX_AUTHORED_FILES = 4
MAX_AUTHORED_FILE_CHARS = 100_000
MAX_AUTHORED_TOTAL_CHARS = 400_000
_AUTHORED_CODE_PREFIX = "code/"
_OPEN_OUTPUT_PREFIXES = ("outputs/", "report/", "logs/")
_ABSOLUTE_PATH_TEXT = re.compile(
    r"(?:[A-Za-z]:[\\/]|(?<![A-Za-z0-9_.:/])/(?!/)(?:[^\s,;:'\"(){}\[\]]+))"
)
_PUBLIC_JSON_POINTER_TEXT = re.compile(
    r"(?<![A-Za-z0-9_.:/])/(?:[A-Za-z0-9_.~-]+)(?:/[A-Za-z0-9_.~-]+)*"
)
_MINICHEM_SELECTOR_PATTERNS = {
    "artifact.media_type": re.compile(r"^artifact\.media_type=[^\s=]+$"),
    "artifact.semantic_type": re.compile(r"^artifact\.semantic_type=[^\s=]+$"),
    "declared_artifact.name": re.compile(r"^declared_artifact\.name=[^\s=]+$"),
}

# 输出合同可能使用的显式结果字段键。这里只接受语义明确的键，不把任意
# 自然语言说明或 ``primary_output`` 猜成 ``/result/...`` 公开字段。
_MINICHEM_RESULT_PATH_KEYS = frozenset(
    {
        "result_paths",
        "public_result_paths",
        "declared_result_paths",
        "result_pointers",
        "public_result_pointers",
        "result_fields",
        "public_result_fields",
        "declared_result_fields",
        "public_fields",
    }
)
_MINICHEM_RESULT_SCHEMA_KEYS = frozenset(
    {
        "result_schema",
        "public_result_schema",
        "output_schema",
        "public_output_schema",
    }
)
_MINICHEM_OUTPUT_SEMANTIC_KEYS = frozenset(
    {
        "primary_output",
        "output_semantic_type",
        "output_semantic_types",
        "artifact_semantic_type",
        "artifact_semantic_types",
        "semantic_type",
        "semantic_types",
    }
)
_MINICHEM_OUTPUT_MEDIA_KEYS = frozenset(
    {"media_type", "media_types", "artifact_media_type", "artifact_media_types"}
)
_MINICHEM_OUTPUT_ARTIFACT_NAME_KEYS = frozenset(
    {"artifact_name", "artifact_names", "declared_artifact_names"}
)


def _valid_minichem_step_keys(value: Any, *, generation: bool) -> bool:
    """接受 action 工作流基础字段及可选 execution_layer 字段。"""

    expected = _MINICHEM_GENERATION_STEP_KEYS if generation else _MINICHEM_STEP_KEYS
    job_expected = (
        _MINICHEM_JOB_GENERATION_STEP_KEYS
        if generation
        else _MINICHEM_JOB_STEP_KEYS
    )
    actual = set(value) if isinstance(value, Mapping) else set()
    return actual == expected or actual == expected | {"execution_layer"} or actual == job_expected


def _validate_authored_file_path(value: Any, label: str) -> str:
    """检查模型提供的代码文件路径，只允许任务工作区的 ``code/`` 相对路径。"""

    if not isinstance(value, str) or not value.strip():
        _fail(f"{label} 必须是非空字符串")
    if "\\" in value:
        _fail(f"{label} 必须使用 POSIX 路径")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or not value.startswith(_AUTHORED_CODE_PREFIX)
        or len(path.parts) < 2
    ):
        _fail(f"{label} 必须是 code/ 下的安全相对路径")
    return path.as_posix()


def _validate_authored_files(value: Any, label: str) -> list[dict[str, str]]:
    """检查 ARCHE 临时写入文件的数量、路径、正文大小和总大小。"""

    if value is None:
        return []
    if not isinstance(value, list):
        _fail(f"{label} 必须是数组")
    if len(value) > MAX_AUTHORED_FILES:
        _fail(f"{label} 最多允许 {MAX_AUTHORED_FILES} 个文件")
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    total_chars = 0
    for index, item in enumerate(value):
        item_label = f"{label}[{index}]"
        if not isinstance(item, Mapping) or set(item) != {"path", "content"}:
            _fail(f"{item_label} 必须只包含 path/content")
        path = _validate_authored_file_path(item.get("path"), f"{item_label}.path")
        if path in seen:
            _fail(f"{item_label}.path 不能重复")
        seen.add(path)
        content = item.get("content")
        if not isinstance(content, str):
            _fail(f"{item_label}.content 必须是字符串")
        size = len(content)
        if size > MAX_AUTHORED_FILE_CHARS:
            _fail(f"{item_label}.content 超过单文件大小上限")
        total_chars += size
        if total_chars > MAX_AUTHORED_TOTAL_CHARS:
            _fail(f"{label} 总大小超过上限")
        normalized.append({"path": path, "content": content})
    return normalized


def _authored_file_reference_count(
    request: Mapping[str, Any],
    authored_path: str,
    execution_layer: str,
) -> int:
    """统计可以在提交前被物理改写的 authored file 引用位置。"""

    count = 0
    if execution_layer == "native_software":
        staged_inputs = request.get("staged_inputs")
        if isinstance(staged_inputs, list):
            count += sum(
                isinstance(item, Mapping) and item.get("source_path") == authored_path
                for item in staged_inputs
            )
        return count
    if request.get("script_path") == authored_path:
        count += 1
    staged_inputs = request.get("staged_inputs")
    if isinstance(staged_inputs, list):
        count += sum(
            isinstance(item, Mapping) and item.get("source_path") == authored_path
            for item in staged_inputs
        )
    inputs = request.get("inputs")
    if isinstance(inputs, list):
        count += sum(
            isinstance(item, Mapping) and item.get("source_path") == authored_path
            for item in inputs
        )
    return count


def _open_execution_catalog_parts(
    catalog_snapshot: Mapping[str, Any],
) -> tuple[dict[str, set[str]], set[str], Mapping[str, Any] | None]:
    """读取冻结 open_execution 的精确 native、analysis 和 authoring 合同。"""

    raw = catalog_snapshot.get("open_execution")
    if not isinstance(raw, Mapping):
        return {}, set(), None
    required_top_fields = {
        "schema_version",
        "native_software",
        "analysis_runtimes",
        "authoring_contract",
    }
    optional_top_fields = {
        "managed_program_contract",
        "submit_analysis_program_request_template",
        "execution_timeout_policy",
        "selection_note",
        "execution_note",
        "evaluation_resource_budget",
    }
    if not required_top_fields <= set(raw) or set(raw) - required_top_fields - optional_top_fields:
        _fail("冻结 open_execution 顶层字段不符合固定合同")
    if raw.get("schema_version") != 1:
        _fail("冻结 open_execution.schema_version 必须为 1")
    native_raw = raw.get("native_software")
    runtime_raw = raw.get("analysis_runtimes")
    contract = raw.get("authoring_contract")
    if not isinstance(native_raw, list) or not isinstance(runtime_raw, list):
        _fail("冻结 open_execution native_software/analysis_runtimes 必须是数组")
    if not isinstance(contract, Mapping) or set(contract) != {
        "path_prefix",
        "max_files",
        "max_chars_per_file",
    }:
        _fail("冻结 open_execution.authoring_contract 字段不符合固定合同")
    if contract.get("path_prefix") != _AUTHORED_CODE_PREFIX:
        _fail("冻结 open_execution.authoring_contract.path_prefix 必须为 code/")
    if contract.get("max_files") != MAX_AUTHORED_FILES:
        _fail("冻结 open_execution.authoring_contract.max_files 不符合 ARCHE 限制")
    if contract.get("max_chars_per_file") != MAX_AUTHORED_FILE_CHARS:
        _fail("冻结 open_execution.authoring_contract.max_chars_per_file 不符合 ARCHE 限制")

    native: dict[str, set[str]] = {}
    for index, item in enumerate(native_raw):
        label = f"open_execution.native_software[{index}]"
        if not isinstance(item, Mapping) or not {
            "software_id",
            "display_name",
            "commands",
        } <= set(item) or set(item) - {
            "software_id",
            "display_name",
            "commands",
        }:
            _fail(f"{label} 字段不符合固定合同")
        software_id = item.get("software_id")
        if not isinstance(software_id, str) or not software_id or software_id in native:
            _fail(f"{label}.software_id 非法或重复")
        if not isinstance(item.get("display_name"), str) or not item["display_name"].strip():
            _fail(f"{label}.display_name 必须是非空字符串")
        commands = item.get("commands")
        if not isinstance(commands, list) or not commands:
            _fail(f"{label}.commands 必须是非空数组")
        executable_set: set[str] = set()
        for command_index, command in enumerate(commands):
            command_label = f"{label}.commands[{command_index}]"
            command_required = {
                "executable",
                "synopsis",
                "input_mode",
                "required_files",
                "example_request",
            }
            if not isinstance(command, Mapping) or not command_required <= set(command) or set(command) - command_required - {"output_behavior"}:
                _fail(f"{command_label} 字段不符合固定合同")
            executable = command.get("executable")
            if not isinstance(executable, str) or not executable or executable in executable_set:
                _fail(f"{command_label}.executable 非法或重复")
            if not isinstance(command.get("synopsis"), str) or not command["synopsis"].strip():
                _fail(f"{command_label}.synopsis 必须是非空字符串")
            if not isinstance(command.get("input_mode"), str) or not command["input_mode"].strip():
                _fail(f"{command_label}.input_mode 必须是非空字符串")
            if not isinstance(command.get("required_files"), list) or any(
                not isinstance(path, str) or not path.strip() for path in command["required_files"]
            ):
                _fail(f"{command_label}.required_files 必须是字符串数组")
            if not isinstance(command.get("example_request"), Mapping):
                _fail(f"{command_label}.example_request 必须是 object")
            executable_set.add(executable)
        native[software_id] = executable_set

    runtimes: set[str] = set()
    for index, item in enumerate(runtime_raw):
        label = f"open_execution.analysis_runtimes[{index}]"
        if not isinstance(item, Mapping) or set(item) != {"runtime", "description", "modules"}:
            _fail(f"{label} 字段不符合固定合同")
        runtime = item.get("runtime")
        if not isinstance(runtime, str) or not runtime or runtime in runtimes:
            _fail(f"{label}.runtime 非法或重复")
        if not isinstance(item.get("description"), str) or not item["description"].strip():
            _fail(f"{label}.description 必须是非空字符串")
        if not isinstance(item.get("modules"), list) or any(
            not isinstance(module, str) or not module.strip() for module in item["modules"]
        ):
            _fail(f"{label}.modules 必须是字符串数组")
        runtimes.add(runtime)
    return native, runtimes, contract


def _validate_open_execution_identity(
    request: Mapping[str, Any],
    execution_layer: str,
    catalog_snapshot: Mapping[str, Any],
    label: str,
) -> None:
    """按冻结目录逐值核对 native software/executable 或 analysis runtime。"""

    native, runtimes, contract = _open_execution_catalog_parts(catalog_snapshot)
    if contract is None and not native and not runtimes:
        _fail(f"{label} 需要冻结 open_execution 快照；旧任务不能新建 open step")
    if execution_layer == "native_software":
        software_id = request.get("software_id")
        executable = request.get("executable")
        if software_id not in native:
            _fail(f"{label}.software_id 未列在冻结 open_execution 中")
        if executable not in native[software_id]:
            _fail(f"{label}.executable 未列在冻结 open_execution 中")
    else:
        if request.get("runtime") not in runtimes:
            _fail(f"{label}.runtime 未列在冻结 open_execution 中")


def _minichem_execution_layer(
    step: Mapping[str, Any],
    action: Mapping[str, Any] | None = None,
) -> str:
    """读取步骤声明的执行层；未声明时按同步 predefined Action 处理。"""

    value = step.get("execution_layer")
    if value is None and isinstance(action, Mapping):
        value = action.get("execution_layer")
    if value is None:
        value = "predefined_actions"
    if value not in _MINICHEM_EXECUTION_LAYER_VALUES:
        _fail(f"MiniChem execution_layer 不受支持：{value}")
    return str(value)


def _validate_minichem_job_request(
    request: Any,
    execution_layer: str,
    label: str,
    *,
    allow_short_refs: bool,
    catalog_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """按真实 NativeJobRequest/AnalysisJobRequest 字段检查 open-execution 请求。

    输入是模型生成的结构化请求和执行层名称；函数只复制 JSON 值，不拼接 shell
    命令。它拒绝 Action 身份、未知字段、operation_id 以及不符合 MiniChem 作业
    身份格式的 parent_job_id，返回值仍保留 ARCHE 内部 bindings 供依赖解析。
    """

    if execution_layer not in {"native_software", "programmable_analysis"}:
        _fail(f"{label} 不是 open-execution 层")
    if not isinstance(request, Mapping):
        _fail(f"{label} 必须是 object")
    allowed = set(request)
    if "action_id" in allowed or "backend_id" in allowed:
        _fail(f"{label} 不得含 predefined Action 字段")
    if "operation_id" in allowed:
        _fail(f"{label} 不得由 Planner 提供 operation_id；执行器会注入本地 operation identity")
    expected_fields = (
        _MINICHEM_NATIVE_JOB_FIELDS
        if execution_layer == "native_software"
        else _MINICHEM_ANALYSIS_JOB_FIELDS
    )
    unknown = allowed - expected_fields
    if unknown:
        _fail(f"{label} 含有该执行层未声明字段：{sorted(unknown)}")
    required_fields = (
        {"software_id", "executable"}
        if execution_layer == "native_software"
        else {"runtime", "script_path"}
    )
    missing_required = required_fields - allowed
    if missing_required:
        _fail(f"{label} 缺少 MiniChem 必需字段：{sorted(missing_required)}")

    def require_text(field: str, *, nullable: bool = False) -> None:
        value = request.get(field)
        if nullable and value is None:
            return
        if not isinstance(value, str) or not value.strip():
            _fail(f"{label}.{field} 必须是非空字符串" + ("或 null" if nullable else ""))

    def validate_string_list(field: str) -> None:
        value = request.get(field)
        if not isinstance(value, list) or any(
            not isinstance(item, str) or not item.strip() for item in value
        ):
            _fail(f"{label}.{field} 必须是非空字符串数组")

    def validate_staged_inputs(field: str = "staged_inputs") -> None:
        value = request.get(field)
        if not isinstance(value, list):
            _fail(f"{label}.{field} 必须是数组")
        for index, item in enumerate(value):
            if not isinstance(item, Mapping) or set(item) != {"source_path", "target_path"}:
                _fail(f"{label}.{field}[{index}] 必须只包含 source_path/target_path")
            source_path = item.get("source_path")
            if source_path is not None and (
                not isinstance(source_path, str) or not source_path.strip()
            ):
                _fail(f"{label}.{field}[{index}].source_path 必须是字符串或 null")
            target_path = item.get("target_path")
            if not isinstance(target_path, str) or not target_path.strip():
                _fail(f"{label}.{field}[{index}].target_path 必须是非空字符串")

    def validate_authored_files() -> None:
        value = request.get("authored_files")
        if not isinstance(value, list) or len(value) > 100:
            _fail(f"{label}.authored_files 必须是最多 100 项的数组")
        paths: list[str] = []
        for index, item in enumerate(value):
            if not isinstance(item, Mapping):
                _fail(f"{label}.authored_files[{index}] 必须是 object")
            if set(item) - {"path", "content", "overwrite"} or {
                "path",
                "content",
            } - set(item):
                _fail(
                    f"{label}.authored_files[{index}] 必须包含 path/content，"
                    "可选 overwrite"
                )
            path = item.get("path")
            content = item.get("content")
            if not isinstance(path, str) or not path.strip():
                _fail(f"{label}.authored_files[{index}].path 必须是非空字符串")
            normalized = PurePosixPath(path.replace("\\", "/"))
            if (
                normalized.is_absolute()
                or ".." in normalized.parts
                or not normalized.parts
                or normalized.parts[0] not in {"code", "outputs"}
            ):
                _fail(
                    f"{label}.authored_files[{index}].path 必须位于 code/ 或 outputs/"
                )
            if not isinstance(content, str) or len(content) > 10_000_000:
                _fail(f"{label}.authored_files[{index}].content 必须是受限字符串")
            if "overwrite" in item and not isinstance(item["overwrite"], bool):
                _fail(f"{label}.authored_files[{index}].overwrite 必须是 bool")
            paths.append(str(normalized))
        if len(paths) != len(set(paths)):
            _fail(f"{label}.authored_files.path 不能重复")

    def validate_resource_limits() -> None:
        value = request.get("resource_limits")
        if not isinstance(value, Mapping):
            _fail(f"{label}.resource_limits 必须是 object")
        if set(value) - {"memory_mb", "cpu_cores", "gpu_count"}:
            _fail(f"{label}.resource_limits 含有未声明字段")
        bounds = {"memory_mb": 128, "cpu_cores": 1, "gpu_count": 0}
        for key, minimum in bounds.items():
            if key in value and (
                isinstance(value[key], bool)
                or not isinstance(value[key], int)
                or value[key] < minimum
            ):
                _fail(f"{label}.resource_limits.{key} 小于 MiniChem 允许下限")

    for field in required_fields:
        require_text(field)
    if "authored_files" in request:
        validate_authored_files()
    _validate_minichem_foreach_execution_layer(request, execution_layer, label)
    if execution_layer == "native_software":
        if "arguments" in request:
            validate_string_list("arguments")
        if "staged_inputs" in request:
            validate_staged_inputs()
        if "stdin_target" in request and request["stdin_target"] is not None:
            require_text("stdin_target")
        if "calculation_intent" in request and request["calculation_intent"] is not None:
            if request["calculation_intent"] not in _MINICHEM_CALCULATION_INTENTS:
                _fail(f"{label}.calculation_intent 不是 MiniChem 允许值")
    else:
        if "script_target" in request:
            require_text("script_target")
        for field in ("arguments", "required_modules"):
            if field in request:
                validate_string_list(field)
        if "staged_inputs" in request:
            validate_staged_inputs()
        if "required_module_versions" in request:
            value = request["required_module_versions"]
            if not isinstance(value, Mapping) or any(
                not isinstance(key, str) or not key.strip()
                or not isinstance(item, str) or not item.strip()
                for key, item in value.items()
            ):
                _fail(f"{label}.required_module_versions 必须是字符串 object")
        if "required_symbols" in request:
            value = request["required_symbols"]
            if not isinstance(value, Mapping):
                _fail(f"{label}.required_symbols 必须是 object")
            for key, symbols in value.items():
                if not isinstance(key, str) or not key.strip() or not isinstance(symbols, list):
                    _fail(f"{label}.required_symbols 必须映射到字符串数组")
                if any(not isinstance(symbol, str) or not symbol.strip() for symbol in symbols):
                    _fail(f"{label}.required_symbols 必须映射到字符串数组")
        if "inputs" in request:
            value = request["inputs"]
            if not isinstance(value, list):
                _fail(f"{label}.inputs 必须是数组")
            for index, item in enumerate(value):
                if not isinstance(item, Mapping):
                    _fail(f"{label}.inputs[{index}] 必须是 object")
                allowed_input = {"name", "source_path", "semantic_type", "target_path"}
                if set(item) - allowed_input or "name" not in item or "source_path" not in item:
                    _fail(f"{label}.inputs[{index}] 缺少 AnalysisInputDeclaration 字段")
                if not isinstance(item["name"], str) or not item["name"].strip():
                    _fail(f"{label}.inputs[{index}].name 必须是非空字符串")
                if item["source_path"] is not None and (
                    not isinstance(item["source_path"], str)
                    or not item["source_path"].strip()
                ):
                    _fail(
                        f"{label}.inputs[{index}].source_path 必须是字符串或 null"
                    )
                if "semantic_type" in item and (
                    not isinstance(item["semantic_type"], str) or not item["semantic_type"].strip()
                ):
                    _fail(f"{label}.inputs[{index}].semantic_type 必须是字符串")
                if "target_path" in item and item["target_path"] is not None and (
                    not isinstance(item["target_path"], str) or not item["target_path"].strip()
                ):
                    _fail(f"{label}.inputs[{index}].target_path 必须是字符串或 null")
        if "outputs" in request:
            value = request["outputs"]
            if not isinstance(value, list):
                _fail(f"{label}.outputs 必须是数组")
            for index, item in enumerate(value):
                if not isinstance(item, Mapping):
                    _fail(f"{label}.outputs[{index}] 必须是 object")
                allowed_output = {
                    "name", "path", "semantic_type", "media_type", "required",
                    "json_schema", "parent_artifact_ids",
                }
                required_output = {"name", "path", "semantic_type"}
                if set(item) - allowed_output or required_output - set(item):
                    _fail(f"{label}.outputs[{index}] 缺少 AnalysisOutputDeclaration 字段")
                for key in ("name", "path", "semantic_type"):
                    if not isinstance(item[key], str) or not item[key].strip():
                        _fail(f"{label}.outputs[{index}].{key} 必须是非空字符串")
                output_path = _validate_task_relative_path(
                    item["path"],
                    f"{label}.outputs[{index}].path",
                )
                if not output_path.startswith(_OPEN_OUTPUT_PREFIXES):
                    _fail(
                        f"{label}.outputs[{index}].path 必须位于 outputs/、report/ 或 logs/"
                    )
                if "media_type" in item and (
                    not isinstance(item["media_type"], str) or not item["media_type"].strip()
                ):
                    _fail(f"{label}.outputs[{index}].media_type 必须是字符串")
                if "required" in item and not isinstance(item["required"], bool):
                    _fail(f"{label}.outputs[{index}].required 必须是 bool")
                if "parent_artifact_ids" in item:
                    parent_ids = item["parent_artifact_ids"]
                    if not isinstance(parent_ids, list) or any(
                        not isinstance(parent_id, str) or not parent_id.strip()
                        for parent_id in parent_ids
                    ):
                        _fail(f"{label}.outputs[{index}].parent_artifact_ids 必须是字符串数组")
        if "execution_policy" in request and request["execution_policy"] != "in_process_only":
            _fail(f"{label}.execution_policy 只能是 in_process_only")
        if "external_execution" in request and request["external_execution"] != "native_job_runner_only":
            _fail(f"{label}.external_execution 只能是 native_job_runner_only")
    if "resource_limits" in request:
        validate_resource_limits()
    if catalog_snapshot is not None:
        _validate_open_execution_identity(
            request,
            execution_layer,
            catalog_snapshot,
            label,
        )
    if "label" in request and request["label"] is not None:
        require_text("label")
    if "parent_job_id" in request and request["parent_job_id"] is not None:
        parent_job_id = request["parent_job_id"]
        if not isinstance(parent_job_id, str) or _MINICHEM_JOB_ID_PATTERN.fullmatch(parent_job_id) is None:
            _fail(f"{label}.parent_job_id 必须匹配 job_ 加 32 位小写十六进制")
    bindings = request.get("bindings", [])
    if not isinstance(bindings, list):
        _fail(f"{label}.bindings 必须是数组")
    normalized = deepcopy(dict(request))
    normalized_bindings: list[dict[str, Any]] = []
    for index, binding in enumerate(bindings):
        normalized_bindings.append(
            _validate_minichem_binding(
                binding,
                f"{label}.bindings[{index}]",
                allow_short_refs=allow_short_refs,
            )
        )
    normalized["bindings"] = normalized_bindings
    foreach = request.get("foreach")
    if foreach is not None:
        if normalized_bindings:
            _fail(
                f"{label}.bindings 与 foreach 互斥；单 deck 使用 binding，批量 deck 使用 foreach"
            )
        normalized["foreach"] = _validate_minichem_foreach_modifier(
            foreach,
            f"{label}.foreach",
            allow_short_refs=allow_short_refs,
        )
        target_tokens = _minichem_target_tokens(foreach.get("target_field"))
        if not re.fullmatch(
            r"staged_inputs\.\d+\.source_path",
            ".".join(target_tokens),
        ):
            _fail(
                f"{label}.foreach.target_field 必须指向 staged_inputs.<index>.source_path"
            )
        if request.get("authored_files") not in (None, []):
            _fail(f"{label} 使用 foreach 时不能同时提供 authored_files")
    if "authored_files" in request:
        normalized_authored = _validate_authored_files(
            request.get("authored_files"), f"{label}.authored_files"
        )
        normalized["authored_files"] = normalized_authored
        for item in normalized_authored:
            if _authored_file_reference_count(
                normalized,
                item["path"],
                execution_layer,
            ) < 1:
                _fail(
                    f"{label}.authored_files[{item['path']}] 必须至少被请求正文引用一次"
                )
    _assert_json_value(normalized, label)
    return normalized
_MINICHEM_REQUEST_KEYS = {
    "inputs",
    "method_spec",
    "action_settings",
    "resource_limits",
    "source_id",
    "component_backends",
    "bindings",
    "foreach",
}


_MINICHEM_FOREACH_KEYS = {
    "source_step_ref",
    "source_field",
    "item_field",
    "item_id_field",
    "target_field",
    "max_items",
    "only_present_items",
}


def _validate_minichem_foreach_modifier(
    value: Any,
    label: str,
    *,
    allow_short_refs: bool,
) -> dict[str, Any]:
    """校验 ARCHE foreach 修饰符并返回稳定副本。

    输入是步骤 request 中的逐项来源、目标路径和上限；动作只检查 JSON Pointer、
    正式步骤引用、条数上限及布尔策略，不解析上游结果或调用工具。返回值保留
    可选字段的规范默认值；字段缺失、空值或越过 32 项上限时抛出合同错误。
    """

    if not isinstance(value, Mapping) or set(value) - _MINICHEM_FOREACH_KEYS or {
        "source_step_ref",
        "source_field",
        "target_field",
    } - set(value):
        _fail(f"{label} 字段不完整")
    source_step_ref = value.get("source_step_ref")
    if not isinstance(source_step_ref, str) or not source_step_ref.strip():
        _fail(f"{label}.source_step_ref 必须是非空字符串")
    if not allow_short_refs and _SHORT_REF_PATTERN.fullmatch(source_step_ref):
        _fail(f"{label}.source_step_ref 不能持久化短引用")
    source_field = value.get("source_field")
    _validate_minichem_source_selector(source_field, f"{label}.source_field")
    target_field = value.get("target_field")
    _minichem_target_tokens(target_field)
    normalized: dict[str, Any] = {
        "source_step_ref": source_step_ref,
        "source_field": source_field,
        "target_field": str(target_field),
        "max_items": value.get("max_items", 32),
        "only_present_items": value.get("only_present_items", False),
    }
    for key in ("item_field", "item_id_field"):
        item_value = value.get(key)
        if item_value is not None:
            if not isinstance(item_value, str) or not item_value.strip():
                _fail(f"{label}.{key} 必须是非空字符串或 null")
            if key == "item_field":
                _validate_minichem_item_field_selector(
                    item_value,
                    f"{label}.{key}",
                )
            else:
                _pointer_tokens(item_value, f"{label}.{key}")
            normalized[key] = item_value
        else:
            normalized[key] = None
    maximum = normalized["max_items"]
    if isinstance(maximum, bool) or not isinstance(maximum, int) or not 1 <= maximum <= 32:
        _fail(f"{label}.max_items 必须在 1 到 32 之间")
    if type(normalized["only_present_items"]) is not bool:
        _fail(f"{label}.only_present_items 必须是 bool")
    return normalized

# Revision 的公开 request 为每个 execution layer 提供固定字段。predefined
# Action 会携带一组属于其他层的空占位；这些字段只用于稳定消息，不得进入
# MiniChem Action 正式请求正文。
_REVISION_ACTION_PLACEHOLDER_DEFAULTS = {
    "software_id": None,
    "executable": None,
    "runtime": None,
    "arguments": [],
    "outputs": [],
    "calculation_intent": None,
    "label": None,
}
_REVISION_HIDDEN_VALUE_KEYS = frozenset(
    {
        "path",
        "source_path",
        "script_path",
        "script_target",
        "parent_job_id",
        "parent_artifact_ids",
        "content",
    }
)
_REVISION_RESOURCE_PLACEHOLDER_FIELDS = frozenset(
    {"cpu_cores", "gpu_count", "memory_mb"}
)
_REVISION_EMPTY_PLACEHOLDER_KEYS = frozenset(
    {
        "arguments",
        "authored_files",
        "bindings",
        "inputs",
        "outputs",
        "required_modules",
        "staged_inputs",
    }
)
_REVISION_EMPTY_MAPPING_PLACEHOLDER_KEYS = frozenset(
    {"required_module_versions", "required_symbols", "resource_limits"}
)


def _is_revision_hidden_value_key(key: str) -> bool:
    """判断公开 request 中需要从原工作流恢复的隐藏值字段。

    输入是字段名；动作按固定字段集合和 ``*_path`` 规则比较，不读取文件或修改
    request。返回 ``True`` 表示提交前应从原步骤恢复该值，否则返回 ``False``；
    字段名异常由调用方类型校验处理。
    """

    normalized = key.casefold()
    return normalized in _REVISION_HIDDEN_VALUE_KEYS or (
        normalized.endswith("_path") and normalized != "target_path"
    )


def _revision_original_value_was_hidden(value: Any) -> bool:
    """判断原值是否因路径或运行内部标记而被公开清空。

    输入是原 request 字段值；动作检查绝对路径、作业身份和敏感标记，不修改值。
    返回 ``True`` 表示公开 wire 中应使用占位，其他值返回 ``False``；非字符串值
    视为普通值，不会抛出异常。
    """

    return isinstance(value, str) and (
        bool(_ABSOLUTE_PATH_TEXT.search(value))
        or bool(re.search(r"(?:operation|execution|job|step|artifact|checkpoint)_?id\s*[:=/_-]", value, re.IGNORECASE))
        or bool(re.search(r"(?:raw[_ -]?output|traceback|secret|token|password)", value, re.IGNORECASE))
    )


def _restore_revision_request_values(
    public_request: Any,
    original_request: Any,
) -> Any:
    """把稳定公开 request 恢复为可执行 request，同时保留模型的公开修改。

    ``project_workflow_for_revision`` 会把缺失字段固定为 null/空容器，并把
    路径、作业身份和文件正文清空。这里只从同一步原始 request 恢复被隐藏的
    值；原来没有的普通字段仍会被省略，避免把 null 占位传给严格 Action 合同。
    显式写入的普通 null 不会被笼统删除，因此合法的可空字段仍交给领域校验。
    输入是模型公开 request 和同一步原始 request；动作恢复隐藏路径、身份与文件内容，
    保留模型对公开科学字段的修改。返回值是可提交 request；输入形状不匹配时按空
    映射处理，最终合法性由领域合同校验，恢复过程不会凭空创建字段。
    """

    def merge(value: Any, original: Any, *, key: str | None = None) -> Any:
        original_mapping = original if isinstance(original, Mapping) else {}
        if isinstance(value, Mapping):
            result: dict[str, Any] = {}
            for raw_key, item in value.items():
                if not isinstance(raw_key, str):
                    continue
                # predefined Action 的跨层空字段只存在于稳定公开 wire 中。
                if (
                    key is None
                    and raw_key in _REVISION_ACTION_PLACEHOLDER_DEFAULTS
                    and raw_key not in original_mapping
                ):
                    if item == _REVISION_ACTION_PLACEHOLDER_DEFAULTS[raw_key]:
                        continue
                was_present = raw_key in original_mapping
                original_item = original_mapping.get(raw_key)
                if item is None:
                    if was_present and (
                        _is_revision_hidden_value_key(raw_key)
                        or _revision_original_value_was_hidden(original_item)
                        or (
                            raw_key in _REVISION_RESOURCE_PLACEHOLDER_FIELDS
                            and original_item is not None
                        )
                    ):
                        result[raw_key] = deepcopy(original_item)
                    elif was_present:
                        # 普通字段的 null 由领域合同决定是否有效；不能擅自
                        # 当作“缺失占位”删除。
                        result[raw_key] = None
                    elif raw_key == "source_path":
                        # 新增 staged/input 声明可以先以 null 表示待绑定来源；
                        # 该字段在 open-execution 合同中本身就是必需可空字段。
                        result[raw_key] = None
                    # 原始 request 没有该字段时，null 只是稳定 wire 占位。
                    continue
                merged_item = merge(
                    item,
                    original_item if was_present else None,
                    key=raw_key,
                )
                if not was_present:
                    if raw_key in _REVISION_EMPTY_PLACEHOLDER_KEYS and merged_item == []:
                        continue
                    if (
                        raw_key in _REVISION_EMPTY_MAPPING_PLACEHOLDER_KEYS
                        and merged_item == {}
                    ):
                        continue
                result[raw_key] = merged_item
            return result
        if isinstance(value, list):
            original_items = original if isinstance(original, list) else []
            # parent_artifact_ids 只在公开视图中清空；原始身份不能被模型
            # 修改，因此原样恢复。其他数组仍按模型提交的内容处理。
            if (
                key == "parent_artifact_ids"
                and not value
                and original_items
            ):
                return deepcopy(original_items)
            return [
                merge(
                    item,
                    original_items[index] if index < len(original_items) else None,
                    key=key,
                )
                for index, item in enumerate(value)
            ]
        if value is None and isinstance(key, str) and _is_revision_hidden_value_key(key):
            return deepcopy(original) if original is not None else None
        return deepcopy(value)

    return merge(public_request, original_request)


def is_minichem_catalog(catalog_snapshot: Mapping[str, Any]) -> bool:
    """判断快照是否来自 MiniChem MCP，而不是旧 Tool Catalog。"""

    return isinstance(catalog_snapshot, Mapping) and isinstance(
        catalog_snapshot.get("actions"), list
    ) and "tools" not in catalog_snapshot


def minichem_action_records(
    catalog_snapshot: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    """按 action_id 建立冻结 Action 索引，拒绝重复或缺失身份。"""

    if not is_minichem_catalog(catalog_snapshot):
        _fail("当前目录不是 MiniChem Action Catalog")
    records: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(catalog_snapshot["actions"]):
        if not isinstance(raw, Mapping):
            _fail(f"MiniChem actions[{index}] 必须是 object")
        action_id = raw.get("action_id", raw.get("id"))
        if not isinstance(action_id, str) or not action_id.strip():
            _fail(f"MiniChem actions[{index}] 缺少 action_id")
        if action_id in records:
            _fail(f"MiniChem action_id 重复：{action_id}")
        records[action_id] = raw
    if not records:
        _fail("MiniChem Action Catalog 不能为空")
    return records


def _redact_catalog_paths(value: Any) -> Any:
    """递归复制模型投影并替换文本中的主机绝对路径。"""

    if isinstance(value, Mapping):
        return {str(key): _redact_catalog_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_catalog_paths(item) for item in value]
    if isinstance(value, str):
        return _ABSOLUTE_PATH_TEXT.sub("[路径已隐藏]", value)
    return deepcopy(value)


def _redact_unlisted_minichem_action_references(
    value: str,
    unlisted_action_ids: Sequence[str],
) -> str:
    """隐藏科学描述中的未公开 Action 和执行层内部术语。"""

    redacted = _redact_catalog_paths(value)
    for action_id in sorted(unlisted_action_ids, key=len, reverse=True):
        if not action_id:
            continue
        redacted = re.sub(
            rf"(?<![A-Za-z0-9_]){re.escape(action_id)}(?![A-Za-z0-9_])",
            "[未列出的能力]",
            redacted,
        )
    redacted = re.sub(
        r"(?:that )?(?:can be )?bound to\s+\[未列出的能力\][A-Za-z0-9_./-]*",
        "",
        redacted,
        flags=re.IGNORECASE,
    )
    redacted = re.sub(r"\bbackend-specific\b", "input-specific", redacted, flags=re.IGNORECASE)
    redacted = re.sub(r"\bbackends?\b", "implementations", redacted, flags=re.IGNORECASE)
    redacted = re.sub(r"\bservice-(?:fixed|managed)\b", "catalog-fixed", redacted, flags=re.IGNORECASE)
    redacted = re.sub(r"\bthe service\b", "the action", redacted, flags=re.IGNORECASE)
    redacted = redacted.replace("ArtifactRef", "typed scientific result")
    redacted = re.sub(r"\bThe Artifact\b", "The result", redacted)
    redacted = redacted.replace("resource_limits.gpu_count", "GPU count")
    redacted = redacted.replace("action_settings", "method settings")
    redacted = redacted.replace("method_spec", "method settings")
    redacted = redacted.replace("execute_action", "execution")
    redacted = redacted.replace("ActionResult", "result")
    redacted = redacted.replace("Agent input", "model input")
    redacted = redacted.replace("workspace file", "file")
    redacted = re.sub(r"\bworkflow\b", "procedure", redacted, flags=re.IGNORECASE)
    redacted = redacted.replace("implementations fallback", "fallback")
    redacted = redacted.replace("an model input", "model input")
    redacted = redacted.replace("an result", "a result")
    redacted = re.sub(
        r"Gaussian TS optimization, frequency checks, and any needed bidirectional IRC remain downstream and are not run here",
        "further validation is not included",
        redacted,
        flags=re.IGNORECASE,
    )
    redacted = re.sub(
        r"perform Gaussian TS optimization, frequency checks, and any needed bidirectional IRC downstream",
        "perform separate validation",
        redacted,
        flags=re.IGNORECASE,
    )
    return redacted


def _redact_catalog_output_contract(value: Any) -> Any:
    """脱敏输出合同，同时保留合同明确声明的公开 JSON Pointer。

    输入是冻结的 ``output_contract``；动作从结果字段、结果 Schema 和
    ``batch_handoff`` 读取公开 selector，只在合同文本中临时保护这些 selector，
    其余绝对路径仍按目录脱敏规则替换。返回深拷贝后的公开合同；输入不是映射时
    退回普通路径脱敏，不修改原值，也不会凭说明文字新增公开路径。
    """

    if not isinstance(value, Mapping):
        return _redact_catalog_paths(value)

    declared_paths: set[str] = set()
    declared_array_patterns: set[tuple[str | None, ...]] = set()
    for key in _MINICHEM_RESULT_PATH_KEYS:
        if key in value:
            declared_paths.update(
                _minichem_result_path_from_declaration(
                    value[key],
                    field_mode="field" in key,
                )
            )
    for key in _MINICHEM_RESULT_SCHEMA_KEYS:
        if key in value:
            declared_paths.update(_minichem_result_schema_paths(value[key]))
            declared_array_patterns.update(
                _minichem_result_schema_array_patterns(value[key])
            )

    declared_selectors = set(declared_paths)
    for schema_key in _MINICHEM_RESULT_SCHEMA_KEYS:
        if schema_key not in value:
            continue
        declared_selectors.update(
            _minichem_batch_handoff_public_selectors(
                value.get("batch_handoff"),
                value[schema_key],
                declared_paths,
            )
        )

    def redact(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): redact(child) for key, child in item.items()}
        if isinstance(item, list):
            return [redact(child) for child in item]
        if not isinstance(item, str):
            return deepcopy(item)

        protected: list[str] = []

        def protect(match: re.Match[str]) -> str:
            selector = match.group(0)
            is_declared = selector in declared_selectors
            if not is_declared and selector.startswith("/result"):
                is_declared = _minichem_result_selector_matches_declared_paths(
                    selector,
                    declared_paths,
                    declared_array_patterns,
                )
            if not is_declared:
                return selector
            placeholder = f"MINICHEM_PUBLIC_POINTER_{len(protected)}_PLACEHOLDER"
            protected.append(selector)
            return placeholder

        projected = _PUBLIC_JSON_POINTER_TEXT.sub(protect, item)
        projected = _ABSOLUTE_PATH_TEXT.sub("[路径已隐藏]", projected)
        for index, selector in enumerate(protected):
            projected = projected.replace(
                f"MINICHEM_PUBLIC_POINTER_{index}_PLACEHOLDER",
                selector,
            )
        return projected

    return redact(value)


def _minichem_action_contract_records(
    catalog_snapshot: Mapping[str, Any],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    """读取冻结 companion 中的 Action/Backend 合同，不猜测缺失契约。"""

    companion = catalog_snapshot.get("action_contracts")
    if companion is None:
        return {}
    if not isinstance(companion, Mapping) or companion.get("schema_version") != 1:
        _fail("MiniChem action_contracts 必须是 schema_version=1 的 object")
    if companion.get("catalog_hash") != catalog_snapshot.get("catalog_hash"):
        _fail("MiniChem action_contracts.catalog_hash 与目录不一致")
    values = companion.get("contracts")
    if not isinstance(values, list):
        _fail("MiniChem action_contracts.contracts 必须是数组")
    records: dict[tuple[str, str], Mapping[str, Any]] = {}
    for index, item in enumerate(values):
        if not isinstance(item, Mapping) or set(item) != {
            "action_id",
            "backend_id",
            "request_contract",
            "output_contract",
        }:
            _fail(f"MiniChem action_contracts.contracts[{index}] 字段不完整")
        pair = (item.get("action_id"), item.get("backend_id"))
        if not all(isinstance(value, str) and value for value in pair):
            _fail(f"MiniChem action_contracts.contracts[{index}] 身份非法")
        if pair in records:
            _fail(f"MiniChem action_contracts 重复 pair：{pair}")
        records[pair] = item
    return records


def project_minichem_request_contracts(
    catalog_snapshot: Mapping[str, Any],
    *,
    gaussian_route_policy: GaussianRoutePolicy | None = None,
) -> list[dict[str, Any]]:
    """生成 Planner 可见的紧凑 pair-specific 请求合同列表。

    ``gaussian_route_policy`` 仅为旧调用链保留；所有目录声明的 pair 对每个任务
    一致可见，具体路线由模型依据合同自主选择。
    """

    records = _minichem_action_contract_records(catalog_snapshot)
    projected: list[dict[str, Any]] = []
    for (action_id, backend_id), item in sorted(records.items()):
        action = next(
            (
                candidate
                for candidate in catalog_snapshot.get("actions", [])
                if isinstance(candidate, Mapping)
                and candidate.get("id", candidate.get("action_id")) == action_id
            ),
            None,
        )
        # 这里是完整的 pair 合同可见性投影，必须保留冻结目录声明的每一个
        # Action/Backend pair。Gaussian route policy 对模型是否可规划的约束由
        # Planner 的 ``planner_contract_projection`` 单独表达；若在此处删除
        # typed Gaussian pair，Planner user JSON 会丢失真实目录合同，无法区分
        # “目录没有该 pair”和“该 pair 受当前任务策略限制”。
        projected.append(
            {
                "action_id": action_id,
                "backend_id": backend_id,
                "request_contract": _redact_catalog_paths(item["request_contract"]),
                "output_contract": _redact_catalog_output_contract(
                    item["output_contract"]
                ),
            }
        )
    return projected


def _minichem_pair_request_contract(
    catalog_snapshot: Mapping[str, Any],
    action_id: str,
    backend_id: str,
) -> Mapping[str, Any] | None:
    """返回指定 Action/Backend 的 companion 合同；没有 companion 时返回 None。"""

    return _minichem_action_contract_records(catalog_snapshot).get((action_id, backend_id))


def project_minichem_stage_tool_projection(
    catalog_snapshot: Mapping[str, Any],
    *,
    gaussian_route_policy: GaussianRoutePolicy | None = None,
) -> list[dict[str, Any]]:
    """从冻结 MiniChem Action/Backend 快照生成 Hypothesis 可见投影。

    输入必须是 canonical catalog；函数只复制 Action/Backend 的公开能力事实，
    不读取旧 Tool Card、入口函数或本地 provider 注册表。返回值不含执行路径和
    密钥，供模型判断可验证性；真正的请求仍由 Planner/Execution 根据同一快照校验。
    """

    records = minichem_action_records(catalog_snapshot)
    backends = {
        item.get("backend_id", item.get("id")): item
        for item in catalog_snapshot.get("backends", [])
        if isinstance(item, Mapping)
    }
    projection: list[dict[str, Any]] = []
    for action_id, action in records.items():
        backend_ids = action.get("backend_ids")
        if not isinstance(backend_ids, list):
            _fail(f"MiniChem Action {action_id} 的 backend_ids 必须是数组")
        backend_projection = []
        for backend_id in backend_ids:
            if not isinstance(backend_id, str) or not backend_id:
                _fail(f"MiniChem Action {action_id} 的 backend_id 非法")
            backend = backends.get(backend_id)
            if backend is None:
                _fail(f"MiniChem Action {action_id} 引用了未注册 backend：{backend_id}")
            backend_projection.append(
                {
                    "backend_id": backend_id,
                    "display_name": backend.get("display_name", backend.get("name", backend_id)),
                    "runtime": backend.get("runtime", backend.get("execution_layer")),
                    "availability": _redact_catalog_paths(
                        backend.get("availability", backend.get("health", "unknown"))
                    ),
                    "capabilities": _redact_catalog_paths(
                        backend.get("capabilities", [])
                    ),
                }
            )
        action_projection = {
                "action_id": action_id,
                "action_version": action.get("action_version", action.get("version")),
                "description": _redact_catalog_paths(action.get("description", "")),
                "backend_ids": list(backend_ids),
                "backends": backend_projection,
                "required_inputs": deepcopy(action.get("required_inputs", [])),
                "optional_inputs": deepcopy(action.get("optional_inputs", [])),
                "method_spec": _redact_catalog_paths(
                    action.get("method_spec", action.get("method_contract", {}))
                ),
                "primary_output": action.get("primary_output"),
                "output_semantic_types": list(
                    action.get("output_semantic_types", [])
                    if isinstance(
                        action.get("output_semantic_types", []), (list, tuple, set)
                    )
                    else []
                ),
                "selection_policy": action.get("selection_policy"),
                "data_action": bool(action.get("data_action", False)),
            }
        pair_contracts = []
        for backend_id in backend_ids:
            contract = _minichem_pair_request_contract(
                catalog_snapshot,
                action_id,
                backend_id,
            )
            if contract is not None:
                pair_contracts.append(
                    {
                        "backend_id": backend_id,
                        "request_contract": _redact_catalog_paths(
                            contract["request_contract"]
                        ),
                        "output_contract": _redact_catalog_output_contract(
                            contract["output_contract"]
                        ),
                    }
                )
        if pair_contracts:
            action_projection["backend_request_contracts"] = pair_contracts
        projection.append(action_projection)
    return projection


_MINICHEM_NON_CHEMICAL_CAPABILITY_CATEGORIES = frozenset(
    {
        "scientific_data_interchange",
        "format_conversion",
        "data_processing",
        "programmable_analysis",
    }
)
_MINICHEM_FORMAT_CONVERSION_BACKENDS = frozenset({"format_conversion"})
_MINICHEM_PROGRAMMABLE_ANALYSIS_BACKENDS = frozenset(
    {"python_runtime", "programmable_analysis"}
)
_MINICHEM_EXECUTION_CLASSES = frozenset({"compute", "fast"})


def _is_minichem_chemical_capability_action(action: Mapping[str, Any]) -> bool:
    """判断 Catalog Action 是否属于模型可用的化学计算能力。

    ``data_action``、执行层和类别均来自冻结目录，不能根据 Action 名称猜测。
    目录中少数旧快照没有 ``execution_layer``，这时只按公开类别和 Backend
    事实排除数据交换/格式转换；不会把可编程分析层伪装成预定义化学能力。
    """

    if bool(action.get("data_action", False)):
        return False
    # 旧版冻结测试目录没有保存 execution_class；这类 Action 仍可依据其余
    # 公开类别和 data_action 事实判断，按默认的短计算能力处理。
    execution_class = action.get("execution_class", "compute")
    if execution_class not in _MINICHEM_EXECUTION_CLASSES:
        return False
    execution_layer = action.get("execution_layer")
    if (
        isinstance(execution_layer, str)
        and execution_layer.strip().casefold() == "programmable_analysis"
    ):
        return False
    category = action.get("category")
    if isinstance(category, str) and category.strip().casefold() in {
        item.casefold() for item in _MINICHEM_NON_CHEMICAL_CAPABILITY_CATEGORIES
    }:
        return False
    backend_ids = action.get("backend_ids")
    if isinstance(backend_ids, (list, tuple, set)) and backend_ids:
        if any(
            isinstance(item, str)
            and item.strip().casefold()
            in {value.casefold() for value in _MINICHEM_PROGRAMMABLE_ANALYSIS_BACKENDS}
            for item in backend_ids
        ):
            return False
        if all(
            isinstance(item, str)
            and item.strip().casefold()
            in {value.casefold() for value in _MINICHEM_FORMAT_CONVERSION_BACKENDS}
            for item in backend_ids
        ):
            return False
    return True


def _project_minichem_chemical_capability_availability(
    backends: Any,
) -> dict[str, str]:
    """把 Backend 状态汇总为不含 Backend 身份的 Action 可用性。"""

    if not isinstance(backends, list) or not backends:
        _fail("化学能力投影缺少 Backend 可用性")
    statuses: list[str] = []
    for backend in backends:
        if not isinstance(backend, Mapping):
            _fail("化学能力投影的 Backend 状态必须是 object")
        availability = backend.get("availability", "unknown")
        if isinstance(availability, Mapping):
            if (
                availability.get("available") is True
                or availability.get("status") == "available"
            ):
                statuses.append("available")
            elif (
                availability.get("available") is False
                or availability.get("status") in {"unavailable", "missing", "disabled"}
            ):
                statuses.append("unavailable")
            else:
                statuses.append("unknown")
        elif isinstance(availability, str):
            normalized = availability.casefold()
            statuses.append(
                "available"
                if normalized in {"available", "healthy", "ready"}
                else "unavailable"
                if normalized in {"unavailable", "missing", "disabled"}
                else "unknown"
            )
        else:
            statuses.append("unknown")
    if "available" in statuses:
        return {"status": "available"}
    if statuses and all(status == "unavailable" for status in statuses):
        return {"status": "unavailable"}
    return {"status": "unknown"}


def project_minichem_chemical_capability_projection(
    catalog_snapshot: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """生成 Hypothesis/rank 使用的真实 MiniChem 化学能力摘要。

    该接口复用冻结 Action/Backend 事实，但只公开非 ``data_action``、非数据交换
    或格式转换、且不属于 ``programmable_analysis`` 的预定义能力。模型只接收
    Action 的科学类别、输入/输出语义、必要约束和汇总可用性，不接收 Backend 身份、
    选择策略、目录哈希、路径、服务身份、路由或通用 Python 运行时信息。
    """

    if not is_minichem_catalog(catalog_snapshot):
        _fail("当前目录不是 MiniChem Action Catalog")
    records = minichem_action_records(catalog_snapshot)
    selected_actions = [
        action
        for action in records.values()
        if _is_minichem_chemical_capability_action(action)
    ]
    if not selected_actions:
        _fail("冻结 MiniChem Catalog 没有可公开的化学计算能力")

    selected_catalog = dict(catalog_snapshot)
    selected_catalog["actions"] = selected_actions
    selected_action_ids = {
        action.get("id", action.get("action_id")) for action in selected_actions
    }
    unlisted_action_ids = tuple(
        action_id for action_id in records if action_id not in selected_action_ids
    )
    stage_projection = project_minichem_stage_tool_projection(selected_catalog)
    stage_by_id = {item["action_id"]: item for item in stage_projection}
    projection: list[dict[str, Any]] = []
    for action in selected_actions:
        action_id = action.get("id", action.get("action_id"))
        if not isinstance(action_id, str) or action_id not in stage_by_id:
            _fail("化学能力投影的 Action 身份与冻结目录不一致")
        stage_item = stage_by_id[action_id]
        category = action.get("category")
        if not isinstance(category, str) or not category.strip():
            _fail(f"MiniChem Action {action_id} 缺少公开科学类别")
        input_description = action.get("input_description", "")
        if not isinstance(input_description, str):
            _fail(f"MiniChem Action {action_id} 的 input_description 非法")
        input_semantic_types = action.get("input_semantic_types", [])
        if not isinstance(input_semantic_types, (list, tuple, set)) or any(
            not isinstance(item, str) or not item.strip()
            for item in input_semantic_types
        ):
            _fail(f"MiniChem Action {action_id} 的 input_semantic_types 非法")
        capability_tags = action.get("capability_tags", [])
        if not isinstance(capability_tags, (list, tuple, set)) or any(
            not isinstance(item, str) or not item.strip() for item in capability_tags
        ):
            _fail(f"MiniChem Action {action_id} 的 capability_tags 非法")
        normalized_capability_tags = (
            sorted(capability_tags) if isinstance(capability_tags, set) else list(capability_tags)
        )
        execution_class = action.get("execution_class", "compute")
        if execution_class not in _MINICHEM_EXECUTION_CLASSES:
            _fail(f"MiniChem Action {action_id} 的 execution_class 非法")
        projection.append(
            {
                "action_id": action_id,
                "action_version": stage_item["action_version"],
                "category": category,
                "description": _redact_unlisted_minichem_action_references(
                    stage_item["description"], unlisted_action_ids
                ),
                "input_description": _redact_unlisted_minichem_action_references(
                    input_description, unlisted_action_ids
                ),
                "input_semantic_types": sorted(input_semantic_types)
                if isinstance(input_semantic_types, set)
                else list(input_semantic_types),
                "required_inputs": deepcopy(stage_item["required_inputs"]),
                "optional_inputs": deepcopy(stage_item["optional_inputs"]),
                "primary_output": stage_item["primary_output"],
                "output_semantic_types": list(stage_item["output_semantic_types"]),
                "capability_tags": list(dict.fromkeys(normalized_capability_tags)),
                "availability": _project_minichem_chemical_capability_availability(
                    stage_item["backends"]
                ),
                "data_action": bool(action.get("data_action", False)),
            }
        )
    return projection


def _minichem_backend_ids(action: Mapping[str, Any]) -> list[str]:
    values = action.get("backend_ids")
    if not isinstance(values, list) or any(not isinstance(item, str) or not item for item in values):
        _fail("MiniChem Action 缺少合法 backend_ids")
    return list(dict.fromkeys(values))


def _minichem_required_fields(
    action: Mapping[str, Any],
    section: str,
    contract: Mapping[str, Any] | None = None,
    *,
    backend_id: str | None = None,
) -> list[str]:
    """读取 inspect_action 合同中的 required 字段，不复制一份 Action 定义。"""

    selected = contract
    if selected is None:
        contracts = action.get("request_contracts")
        selected = (
            contracts.get(backend_id)
            if isinstance(contracts, Mapping) and backend_id is not None
            else action.get("request_contract", action.get("contract"))
        )
    if isinstance(selected, Mapping):
        sections = selected.get("sections")
        raw_section = sections.get(section) if isinstance(sections, Mapping) else None
        if raw_section is None:
            required_contract = selected.get("required_contract")
            raw_section = (
                required_contract.get(section)
                if isinstance(required_contract, Mapping)
                else None
            )
        if isinstance(raw_section, Mapping):
            required = raw_section.get("required")
        elif isinstance(raw_section, list):
            required = raw_section
        else:
            required = None
        if isinstance(required, list):
            values = []
            for item in required:
                name = item.get("name") if isinstance(item, Mapping) else item
                if isinstance(name, str) and name:
                    values.append(name)
            return values
    fallback = action.get(f"required_{section}")
    if isinstance(fallback, list):
        return [item for item in fallback if isinstance(item, str) and item]
    return []


def _minichem_allowed_fields(
    action: Mapping[str, Any],
    section: str,
    contract: Mapping[str, Any] | None = None,
) -> list[str]:
    """读取 pair-specific 合同声明的完整字段集合；空数组表示兼容旧摘要。"""

    selected = contract
    if selected is None:
        selected = action.get("request_contract", action.get("contract"))
    if isinstance(selected, Mapping):
        sections = selected.get("sections")
        raw_section = sections.get(section) if isinstance(sections, Mapping) else None
        if isinstance(raw_section, Mapping):
            allowed = raw_section.get("allowed")
            if isinstance(allowed, list):
                return [item for item in allowed if isinstance(item, str) and item]
        template = selected.get("execute_action_request_template")
        if isinstance(template, Mapping) and isinstance(template.get(section), Mapping):
            return [item for item in template[section] if isinstance(item, str)]
        required = _minichem_required_fields(action, section, selected)
        optional_names = selected.get("optional_field_names")
        if isinstance(optional_names, Mapping):
            fields = optional_names.get(section, [])
            if isinstance(fields, list):
                return list(
                    dict.fromkeys(
                        [*required, *[item for item in fields if isinstance(item, str)]]
                    )
                )
        if isinstance(optional_names, list):
            optional = []
            for item in optional_names:
                if not isinstance(item, str):
                    continue
                if "." in item:
                    prefix, name = item.split(".", 1)
                    if prefix == section:
                        optional.append(name)
                elif section == "inputs":
                    optional.append(item)
            return list(dict.fromkeys(required + optional))
        if section == "resource_limits":
            return ["memory_mb", "cpu_cores", "gpu_count"]
    return []


def _minichem_enumerated_options(
    contract: Mapping[str, Any] | None,
    section: str,
) -> Mapping[str, Sequence[Any]]:
    """读取 inspect_action enumerated_options 中当前 section 的离散取值。"""

    if not isinstance(contract, Mapping):
        return {}
    options = contract.get("enumerated_options")
    if isinstance(options, list):
        selected_list: dict[str, Sequence[Any]] = {}
        for item in options:
            if not isinstance(item, Mapping):
                continue
            path = item.get("path")
            values = item.get("allowed_values")
            if not isinstance(path, str) or not isinstance(values, list):
                continue
            if "." in path:
                prefix, key = path.split(".", 1)
                if prefix != section:
                    continue
            else:
                key = path
            selected_list[key] = values
        return selected_list
    if not isinstance(options, Mapping):
        return {}
    selected: dict[str, Sequence[Any]] = {}
    for raw_key, values in options.items():
        if not isinstance(raw_key, str) or not isinstance(values, list):
            continue
        key = raw_key
        if "." in raw_key:
            prefix, key = raw_key.split(".", 1)
            if prefix != section:
                continue
        elif section != "action_settings":
            continue
        selected[key] = values
    return selected


def _minichem_required_component_roles(
    contract: Mapping[str, Any] | None,
) -> list[str]:
    """读取 pair 合同中必须显式选择的 component backend 角色。"""

    if not isinstance(contract, Mapping):
        return []
    required_contract = contract.get("required_contract")
    if isinstance(required_contract, Mapping):
        raw_roles = required_contract.get("component_backends")
        if isinstance(raw_roles, list):
            roles: list[str] = []
            for item in raw_roles:
                role = item.get("name") if isinstance(item, Mapping) else item
                if isinstance(role, str) and role and role not in roles:
                    roles.append(role)
            if roles:
                return roles
    values = contract.get("component_request_contracts")
    if isinstance(values, Mapping):
        result = []
        for role, specification in values.items():
            if isinstance(specification, Mapping):
                required = specification.get("required", specification.get("required_for_request", False))
                if required is True and isinstance(role, str):
                    result.append(role)
            elif specification is True and isinstance(role, str):
                result.append(role)
        return result
    if isinstance(values, list):
        result = []
        for item in values:
            if not isinstance(item, Mapping):
                continue
            role = item.get("role", item.get("name"))
            if item.get("required") is True and isinstance(role, str):
                result.append(role)
        return result
    return []


def _minichem_component_roles(
    contract: Mapping[str, Any] | None,
) -> list[str]:
    """读取 pair 合同声明的全部 component backend 角色。"""

    if not isinstance(contract, Mapping):
        return []
    roles: list[str] = []
    required_contract = contract.get("required_contract")
    if isinstance(required_contract, Mapping):
        raw_roles = required_contract.get("component_backends")
        if isinstance(raw_roles, list):
            for item in raw_roles:
                role = item.get("name") if isinstance(item, Mapping) else item
                if isinstance(role, str) and role and role not in roles:
                    roles.append(role)
    values = contract.get("component_request_contracts")
    if isinstance(values, Mapping):
        for role in values:
            if isinstance(role, str) and role and role not in roles:
                roles.append(role)
    elif isinstance(values, list):
        for item in values:
            if not isinstance(item, Mapping):
                continue
            role = item.get("role", item.get("name"))
            if isinstance(role, str) and role and role not in roles:
                roles.append(role)
    return roles


def _minichem_contract_field_specs(
    contract: Mapping[str, Any] | None,
    section: str,
) -> dict[str, Mapping[str, Any]]:
    """读取 section 中保留的字段类型和静态元数据。"""

    if not isinstance(contract, Mapping):
        return {}
    raw_section: Any = None
    required_contract = contract.get("required_contract")
    if isinstance(required_contract, Mapping):
        raw_section = required_contract.get(section)
    if raw_section is None:
        sections = contract.get("sections")
        if isinstance(sections, Mapping):
            raw_section = sections.get(section)
    values: list[Any] = []
    if isinstance(raw_section, list):
        values = raw_section
    elif isinstance(raw_section, Mapping):
        for field_name in ("required", "optional", "optional_documented", "optional_with_enumerated_values"):
            raw_values = raw_section.get(field_name)
            if isinstance(raw_values, list):
                values.extend(raw_values)
    specs: dict[str, Mapping[str, Any]] = {}
    for item in values:
        if not isinstance(item, Mapping):
            continue
        name = item.get("name")
        if isinstance(name, str) and name and name not in specs:
            specs[name] = item
    return specs


def _validate_minichem_static_field_type(
    value: Any,
    specification: Mapping[str, Any],
    label: str,
) -> None:
    """校验合同明确给出的基本 JSON 类型，不猜测自然语言条件。"""

    type_hint = specification.get("type")
    if not isinstance(type_hint, str) or not type_hint.strip():
        return
    lowered = type_hint.casefold()
    flexible = any(
        marker in lowered
        for marker in (
            "artifactref",
            "atomicstructure",
            "workspace-relative",
            "documented",
            "structured",
            "path",
        )
    )
    has_union = "|" in lowered or re.search(r"\bor\b", lowered)
    if has_union:
        # 对可以明确识别的 JSON 形状执行联合类型校验；若某个分支只是
        # documented/structured/path 等开放描述，则保留旧的宽松兼容行为。
        alternatives = re.split(r"\s*(?:\||\bor\b)\s*", lowered)
        allowed: list[tuple[str, str]] = []
        unknown_branch = False
        for alternative in alternatives:
            branch = alternative.strip()
            if not branch:
                unknown_branch = True
                continue
            if any(marker in branch for marker in (
                "artifactref",
                "atomicstructure",
                "workspace-relative",
                "documented",
                "structured",
                "path",
            )):
                unknown_branch = True
                continue
            if "reactants" in branch and "products" in branch and (
                "{" in branch or "mapping" in branch or "object" in branch
            ):
                allowed.append(("object", "object"))
            elif "array" in branch:
                allowed.append(("array", "array"))
            elif "boolean" in branch:
                allowed.append(("boolean", "boolean"))
            elif "integer" in branch and "number" not in branch:
                allowed.append(("integer", "integer"))
            elif "number" in branch:
                allowed.append(("number", "number"))
            elif "string" in branch:
                allowed.append(("string", "string"))
            elif re.search(r"\bnull\b", branch):
                allowed.append(("null", "null"))
            elif "object" in branch or "mapping" in branch:
                allowed.append(("object", "object"))
            else:
                unknown_branch = True
        if unknown_branch:
            return
        if not allowed:
            return
        for kind, _display in allowed:
            if kind == "null" and value is None:
                return
            if kind == "boolean" and type(value) is bool:
                return
            if kind == "integer" and type(value) is int:
                return
            if kind == "number" and type(value) in (int, float):
                return
            if kind == "array" and isinstance(value, list):
                return
            if kind == "string" and isinstance(value, str):
                return
            if kind == "object" and isinstance(value, Mapping):
                return
        display_names = {
            "null": "null",
            "boolean": "boolean",
            "integer": "integer",
            "number": "number",
            "array": "array",
            "string": "string",
            "object": "object",
        }
        expected = " 或 ".join(
            dict.fromkeys(display_names[kind] for kind, _display in allowed)
        )
        _fail(f"{label} 必须是 {expected}")
    # `array[integer]`、`array[string]` 等提示同时包含元素类型词；必须先按
    # 容器形状判断，否则会把整个合法数组误判为单个标量。
    if "array" in lowered and not flexible:
        if not isinstance(value, list):
            _fail(f"{label} 必须是 array")
        return
    if "boolean" in lowered and not flexible:
        if type(value) is not bool:
            _fail(f"{label} 必须是 boolean")
        return
    if "integer" in lowered and not flexible and "number" not in lowered:
        if type(value) is not int:
            _fail(f"{label} 必须是 integer")
        return
    if "number" in lowered and not flexible:
        if type(value) not in (int, float):
            _fail(f"{label} 必须是 number")
        return
    if "object" in lowered and not flexible and "array" not in lowered:
        if not isinstance(value, Mapping):
            _fail(f"{label} 必须是 object")
        return
    if "string" in lowered and not flexible and not any(
        marker in lowered for marker in ("array", "object", "integer", "number", "boolean")
    ):
        if not isinstance(value, str):
            _fail(f"{label} 必须是 string")


def _collect_minichem_binding_target_paths(request: Mapping[str, Any]) -> set[str]:
    """收集当前请求中 binding/foreach 精确声明的可写目标路径。

    输入是尚未规范化的 MiniChem request。函数解析点路径或 JSON Pointer，不修改
    请求，也不访问运行时；返回统一的点路径集合。目标路径非法时直接抛出合同错误，
    其余 binding 字段仍由后续完整校验负责。
    """

    targets: set[str] = set()
    bindings = request.get("bindings", [])
    if isinstance(bindings, list):
        for binding in bindings:
            if not isinstance(binding, Mapping) or "target_field" not in binding:
                continue
            targets.add(".".join(_minichem_target_tokens(binding["target_field"])))
    foreach = request.get("foreach")
    if isinstance(foreach, Mapping) and "target_field" in foreach:
        targets.add(".".join(_minichem_target_tokens(foreach["target_field"])))
    return targets


def _minichem_field_explicitly_allows_null(
    specification: Mapping[str, Any] | None,
    enumerated_values: Sequence[Any],
) -> bool:
    """判断 Action 合同是否把字段本身声明为可空。"""

    if None in enumerated_values:
        return True
    if not isinstance(specification, Mapping):
        return False
    allowed_values = specification.get("allowed_values")
    if isinstance(allowed_values, list) and None in allowed_values:
        return True
    type_hint = specification.get("type")
    return isinstance(type_hint, str) and re.search(
        r"(?:^|[|\s])null(?:$|[|\s])",
        type_hint.casefold(),
    ) is not None


def _build_minichem_binding_placeholder_type_schema(
    specification: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """提取空占位校验需要的数组及元素类型，不复制完整 Planner Schema。"""

    if not isinstance(specification, Mapping):
        return {}
    type_hint = specification.get("type")
    if not isinstance(type_hint, str):
        return {}
    lowered = type_hint.casefold()
    if "array" not in lowered:
        return {}
    schema: dict[str, Any] = {"type": "array"}
    for item_type in ("boolean", "integer", "number", "string", "object"):
        if item_type in lowered:
            schema["items"] = {"type": item_type}
            break
    return schema


def _validate_minichem_binding_placeholders(
    value: Any,
    *,
    field_path: str,
    field_schema: Mapping[str, Any],
    binding_targets: set[str],
    explicitly_nullable: bool,
    label: str,
) -> None:
    """只允许 binding/foreach 精确目标使用合同外的空占位。

    输入是单个 Action 字段、其类型 Schema 和本请求声明的目标路径；函数不修改
    数据。字段或带类型数组元素为 ``null`` 且既非合同可空值、也没有精确数据边时，
    抛出 ``WorkflowContractError``。
    """

    if value is None:
        if explicitly_nullable or field_path in binding_targets:
            return
        _fail(f"{label} 为 null 时必须是同一步 binding/foreach 的精确 target_field")
    if (
        field_schema.get("type") == "array"
        and isinstance(field_schema.get("items"), Mapping)
        and isinstance(value, list)
    ):
        for index, item in enumerate(value):
            if item is None and f"{field_path}.{index}" not in binding_targets:
                _fail(
                    f"{label}[{index}] 为 null 时必须是同一步 binding/foreach 的"
                    "精确 target_field"
                )


def _validate_minichem_request(
    request: Any,
    action: Mapping[str, Any],
    backend_id: str,
    label: str,
    *,
    allow_short_refs: bool,
    catalog_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """按 inspect_action 返回的请求契约检查工作流请求。"""

    if not isinstance(request, Mapping):
        _fail(f"{label} 必须是 object")
    if set(request) - _MINICHEM_REQUEST_KEYS:
        _fail(f"{label} 含有未声明字段")
    # predefined Action 的执行器已有逐项 fan-out 分支；不能把它与 open job
    # 的 native Gaussian foreach 校验混为一谈。
    _validate_minichem_foreach_execution_layer(request, "predefined_actions", label)
    binding_targets = _collect_minichem_binding_target_paths(request)
    providers = _minichem_backend_ids(action)
    if backend_id not in providers:
        _fail(f"MiniChem Action 未声明 backend_id：{backend_id}")
    normalized: dict[str, Any] = {}
    pair_contract = (
        _minichem_pair_request_contract(catalog_snapshot, action.get("id", action.get("action_id")), backend_id)
        if catalog_snapshot is not None
        else None
    )
    pair_request_contract = (
        pair_contract.get("request_contract")
        if isinstance(pair_contract, Mapping)
        else None
    )
    for section in ("inputs", "method_spec", "action_settings", "resource_limits"):
        value = request.get(section, {})
        if not isinstance(value, Mapping):
            _fail(f"{label}.{section} 必须是 object")
        normalized[section] = deepcopy(dict(value))
        required_fields = _minichem_required_fields(
            action,
            section,
            pair_request_contract,
            backend_id=backend_id,
        )
        for field_name in required_fields:
            if field_name not in value:
                _fail(f"{label}.{section} 缺少 inspect_action 必需字段 {field_name}")
        allowed_fields = _minichem_allowed_fields(action, section, pair_request_contract)
        if allowed_fields and any(field_name not in allowed_fields for field_name in value):
            _fail(
                f"{label}.{section} 含有 pair-specific 合同未声明字段："
                f"{sorted(set(value) - set(allowed_fields))}"
            )
        enumerated_options = _minichem_enumerated_options(
            pair_request_contract, section
        )
        field_specs = _minichem_contract_field_specs(
            pair_request_contract, section
        )
        placeholder_checked_fields = set(required_fields) | set(field_specs) | set(
            enumerated_options
        )
        for field_name in placeholder_checked_fields:
            if field_name not in value:
                continue
            specification = field_specs.get(field_name)
            choices = enumerated_options.get(field_name, [])
            _validate_minichem_binding_placeholders(
                value[field_name],
                field_path=f"{section}.{field_name}",
                field_schema=_build_minichem_binding_placeholder_type_schema(
                    specification
                ),
                binding_targets=binding_targets,
                explicitly_nullable=_minichem_field_explicitly_allows_null(
                    specification,
                    choices,
                ),
                label=f"{label}.{section}.{field_name}",
            )
        for field_name, choices in enumerated_options.items():
            is_binding_placeholder = (
                field_name in value
                and value[field_name] is None
                and f"{section}.{field_name}" in binding_targets
            )
            if (
                field_name in value
                and not is_binding_placeholder
                and value[field_name] not in choices
            ):
                _fail(
                    f"{label}.{section}.{field_name} 不在 inspect_action 允许值中："
                    f"{choices!r}"
                )
        for field_name, specification in field_specs.items():
            if field_name not in value:
                continue
            is_binding_placeholder = (
                value[field_name] is None
                and f"{section}.{field_name}" in binding_targets
            )
            if is_binding_placeholder:
                continue
            _validate_minichem_static_field_type(
                value[field_name],
                specification,
                f"{label}.{section}.{field_name}",
            )
            allowed_values = specification.get("allowed_values")
            if isinstance(allowed_values, list) and value[field_name] not in allowed_values:
                _fail(
                    f"{label}.{section}.{field_name} 不在 inspect_action 允许值中："
                    f"{allowed_values!r}"
                )
    for section in ("source_id",):
        if section in request and request[section] is not None and not isinstance(request[section], str):
            _fail(f"{label}.{section} 必须是字符串或 null")
        if section in request:
            normalized[section] = request[section]
    component = request.get("component_backends", {})
    if not isinstance(component, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in component.items()
    ):
        _fail(f"{label}.component_backends 必须是字符串 object")
    normalized["component_backends"] = deepcopy(dict(component))
    required_roles = _minichem_required_component_roles(pair_request_contract)
    missing_roles = [role for role in required_roles if role not in component]
    if missing_roles:
        _fail(
            f"{label}.component_backends 缺少 pair 合同必需角色：{missing_roles}"
        )
    allowed_roles = _minichem_component_roles(pair_request_contract)
    if allowed_roles and any(role not in allowed_roles for role in component):
        _fail(
            f"{label}.component_backends 含有 pair 合同未声明角色："
            f"{sorted(set(component) - set(allowed_roles))}"
        )
    bindings = request.get("bindings", [])
    if not isinstance(bindings, list):
        _fail(f"{label}.bindings 必须是数组")
    normalized_bindings = []
    for index, binding in enumerate(bindings):
        normalized_bindings.append(
            _validate_minichem_binding(
                binding,
                f"{label}.bindings[{index}]",
                allow_short_refs=allow_short_refs,
            )
        )
    normalized["bindings"] = normalized_bindings
    foreach = request.get("foreach")
    if foreach is not None:
        normalized["foreach"] = _validate_minichem_foreach_modifier(
            foreach,
            f"{label}.foreach",
            allow_short_refs=allow_short_refs,
        )
    validation_request = deepcopy(normalized)
    validation_request.pop("bindings", None)
    validation_request.pop("foreach", None)
    for index, binding in enumerate(normalized_bindings):
        _set_minichem_target_value(
            validation_request,
            _minichem_target_tokens(binding["target_field"]),
            f"__binding_{index}__",
        )
    if foreach is not None:
        _set_minichem_target_value(
            validation_request,
            _minichem_target_tokens(foreach["target_field"]),
            "__foreach_item__",
        )
    _assert_json_value(normalized, label)
    return normalized


def _minichem_expected_outputs(
    action: Mapping[str, Any],
    *,
    pair_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """从 Action 公开输出元数据构造执行器可复核的最小声明。"""

    value = {
        "action_id": action.get("action_id", action.get("id")),
        "action_version": action.get("action_version", action.get("version")),
        "primary_output": action.get("primary_output"),
        "output_semantic_types": list(action.get("output_semantic_types", [])),
    }
    if isinstance(pair_contract, Mapping) and pair_contract.get("output_contract") is not None:
        value["output_contract"] = deepcopy(pair_contract["output_contract"])
    return value


def _minichem_result_path_from_declaration(
    value: Any,
    *,
    field_mode: bool,
) -> set[str]:
    """把合同明确列出的 result 字段或路径规范化为 JSON Pointer。

    输入是合同字段值及是否允许裸字段名的标记；动作解析 ``result`` 前缀和安全路径
    片段，不修改输入。返回值是合法 pointer 集合；类型不符、空值或非法字符返回
    空集合，不会抛出解析异常。
    """

    paths: set[str] = set()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return paths
        if text.startswith("/result"):
            if text == "/result" or text.startswith("/result/"):
                paths.add(text)
            return paths
        if text.startswith("result."):
            text = text[len("result.") :]
        elif text.startswith("result/"):
            text = text[len("result/") :]
        elif not field_mode:
            return paths
        if text and not text.startswith("/"):
            tokens = [token for token in re.split(r"[./]", text) if token]
            if tokens and all(re.fullmatch(r"[A-Za-z0-9_.-]+", token) for token in tokens):
                paths.add("/result/" + "/".join(tokens))
        return paths
    if isinstance(value, Mapping):
        before = len(paths)
        for key in ("path", "pointer", "result_path", "result_pointer"):
            if key in value:
                paths.update(
                    _minichem_result_path_from_declaration(
                        value[key], field_mode=False
                    )
                )
        for key in ("name", "field", "result_field"):
            if key in value:
                paths.update(
                    _minichem_result_path_from_declaration(
                        value[key], field_mode=True
                    )
                )
        if len(paths) == before and field_mode:
            for key in value:
                if isinstance(key, str) and key not in {
                    "type",
                    "required",
                    "description",
                    "title",
                }:
                    paths.update(
                        _minichem_result_path_from_declaration(
                            key, field_mode=True
                        )
                    )
        return paths
    if isinstance(value, (list, tuple, set)):
        for item in value:
            paths.update(
                _minichem_result_path_from_declaration(item, field_mode=field_mode)
            )
    return paths


def _minichem_result_schema_paths(
    value: Any,
    base_pointer: str = "/result",
) -> set[str]:
    """读取 JSON Schema 或字段表中合同明确公开的 result 路径。

    输入是 schema/字段表及当前 JSON Pointer；动作递归遍历 properties 等公开字段，
    不修改输入。返回值是合法 pointer 集合；非映射、空字段或循环结构返回已收集的
    集合，不会把未知元数据当成结果字段。
    """

    paths: set[str] = set()
    if isinstance(value, list):
        for item in value:
            paths.update(_minichem_result_schema_paths(item, base_pointer))
        return paths
    if not isinstance(value, Mapping):
        return paths
    schema_type = value.get("type")
    if schema_type == "array" or (
        isinstance(schema_type, list) and "array" in schema_type
    ):
        # 数组项没有固定的 JSON Pointer 下标；项及其子字段由数组路径模式
        # 单独核对，不能在这里伪造未带下标的 ``/items/name`` 路径。
        return paths
    properties = value.get("properties")
    if isinstance(properties, Mapping):
        for name, schema in properties.items():
            if not isinstance(name, str) or not name:
                continue
            escaped = name.replace("~", "~0").replace("/", "~1")
            pointer = f"{base_pointer}/{escaped}"
            paths.add(pointer)
            if isinstance(schema, Mapping):
                paths.update(_minichem_result_schema_paths(schema, pointer))
        return paths
    fields = value.get("fields")
    if isinstance(fields, (list, tuple, set, Mapping)):
        if isinstance(fields, Mapping):
            entries = (
                (name, schema, False) for name, schema in fields.items()
            )
        else:
            entries = (
                (
                    item.get("name", item.get("field", item.get("result_field"))),
                    item,
                    True,
                )
                for item in fields
                if isinstance(item, Mapping)
            )
        for name, schema, is_descriptor in entries:
            if not isinstance(name, str) or not name:
                continue
            escaped = name.replace("~", "~0").replace("/", "~1")
            pointer = f"{base_pointer}/{escaped}"
            paths.add(pointer)
            if isinstance(schema, Mapping):
                nested_schema = (
                    {
                        key: item
                        for key, item in schema.items()
                        if key not in {"name", "field", "result_field"}
                    }
                    if is_descriptor
                    else schema
                )
                paths.update(_minichem_result_schema_paths(nested_schema, pointer))
        if paths:
            return paths
    if isinstance(value, Mapping):
        metadata_keys = {
            "type",
            "required",
            "description",
            "title",
            "items",
            "additionalProperties",
            "nullable",
            "fields",
        }
        for name, schema in value.items():
            if name in metadata_keys or not isinstance(name, str) or not name:
                continue
            escaped = name.replace("~", "~0").replace("/", "~1")
            pointer = f"{base_pointer}/{escaped}"
            paths.add(pointer)
            if isinstance(schema, Mapping):
                paths.update(_minichem_result_schema_paths(schema, pointer))
    return paths


def _minichem_result_schema_array_patterns(
    value: Any,
    base_tokens: tuple[str | None, ...] = ("result",),
) -> set[tuple[str | None, ...]]:
    """读取结果 Schema 明确声明的数组项路径。

    输入是 Schema 及当前已解码路径片段；动作递归遍历 properties 和 items，使用
    ``None`` 表示一个数组下标，不修改输入。返回数组项路径模式集合；非映射或未
    声明数组类型时返回空集合，不把普通字段名当成数组通配符。
    """

    patterns: set[tuple[str | None, ...]] = set()
    if isinstance(value, list):
        for item in value:
            patterns.update(
                _minichem_result_schema_array_patterns(item, base_tokens)
            )
        return patterns
    if not isinstance(value, Mapping):
        return patterns
    schema_type = value.get("type")
    declares_array = schema_type == "array" or (
        isinstance(schema_type, list) and "array" in schema_type
    )
    item_schema = value.get("items")
    if declares_array and isinstance(item_schema, Mapping):
        item_pattern = (*base_tokens, None)
        patterns.add(item_pattern)
        patterns.update(
            _minichem_result_schema_array_patterns(item_schema, item_pattern)
        )
    properties = value.get("properties")
    if isinstance(properties, Mapping):
        for name, schema in properties.items():
            if isinstance(name, str) and name and isinstance(schema, Mapping):
                property_tokens = (*base_tokens, name)
                patterns.add(property_tokens)
                patterns.update(
                    _minichem_result_schema_array_patterns(
                        schema,
                        property_tokens,
                    )
                )
        return patterns
    fields = value.get("fields")
    if isinstance(fields, (list, tuple, set, Mapping)):
        if isinstance(fields, Mapping):
            entries = fields.items()
        else:
            entries = (
                (
                    item.get("name", item.get("field", item.get("result_field"))),
                    item,
                )
                for item in fields
                if isinstance(item, Mapping)
            )
        for name, schema in entries:
            if isinstance(name, str) and name and isinstance(schema, Mapping):
                property_tokens = (*base_tokens, name)
                patterns.add(property_tokens)
                patterns.update(
                    _minichem_result_schema_array_patterns(
                        schema,
                        property_tokens,
                    )
                )
    return patterns


def _minichem_result_schema_child(
    schema: Any,
    token: str,
) -> Mapping[str, Any] | None:
    """按一个已解码路径片段读取结果 Schema 的子节点。"""

    if not isinstance(schema, Mapping):
        return None
    properties = schema.get("properties")
    if isinstance(properties, Mapping) and isinstance(properties.get(token), Mapping):
        return properties[token]
    fields = schema.get("fields")
    if isinstance(fields, Mapping) and isinstance(fields.get(token), Mapping):
        return fields[token]
    if isinstance(fields, (list, tuple, set)):
        for item in fields:
            if not isinstance(item, Mapping):
                continue
            name = item.get("name", item.get("field", item.get("result_field")))
            if name == token:
                return {
                    key: value
                    for key, value in item.items()
                    if key not in {"name", "field", "result_field"}
                }
    schema_type = schema.get("type")
    if schema_type == "array" or (
        isinstance(schema_type, list) and "array" in schema_type
    ):
        items = schema.get("items")
        if (
            isinstance(items, Mapping)
            and re.fullmatch(r"0|[1-9][0-9]*", token) is not None
        ):
            return items
    return None


def _minichem_batch_handoff_public_selectors(
    value: Any,
    result_schema: Any,
    declared_paths: set[str],
) -> set[str]:
    """返回 batch_handoff 中经结果/请求字段语义核对的公开 selector。"""

    if not isinstance(value, Mapping):
        return set()
    selectors: set[str] = set()
    source_field = value.get("source_field")
    if not isinstance(source_field, str) or source_field not in declared_paths:
        source_field = None
    if source_field is not None and isinstance(result_schema, Mapping):
        try:
            tokens = _pointer_tokens(source_field, "MiniChem batch_handoff.source_field")
        except WorkflowContractError:
            tokens = []
        if tokens and tokens[0] == "result":
            schema_node: Mapping[str, Any] | None = result_schema
            for token in tokens[1:]:
                schema_node = _minichem_result_schema_child(schema_node, token)
                if schema_node is None:
                    break
            if isinstance(schema_node, Mapping):
                schema_type = schema_node.get("type")
                if schema_type == "array" or (
                    isinstance(schema_type, list) and "array" in schema_type
                ):
                    item_schema = schema_node.get("items")
                    if isinstance(item_schema, Mapping):
                        item_paths = _minichem_result_schema_paths(
                            item_schema,
                            base_pointer="",
                        )
                        for field_name in ("item_field", "item_id_field"):
                            selector = value.get(field_name)
                            if isinstance(selector, str) and selector in item_paths:
                                selectors.add(selector)
    target_field = value.get("target_field")
    if isinstance(target_field, str) and target_field.startswith("/"):
        try:
            _minichem_target_tokens(target_field)
        except WorkflowContractError:
            pass
        else:
            selectors.add(target_field)
    return selectors


def _minichem_result_selector_matches_declared_paths(
    selector: str,
    declared_paths: set[str],
    declared_array_patterns: set[tuple[str | None, ...]],
) -> bool:
    """核对结果 selector 是否匹配合同公开路径或数组项路径。

    输入是规范 JSON Pointer、公开字段路径和数组项模式；动作逐段比较，只有模式
    中的 ``None`` 接受规范的非负十进制数组下标，不修改输入。返回 ``True`` 表示
    selector 受合同支持；字段不同、下标非法或路径长度不同时返回 ``False``。
    """

    if selector in declared_paths:
        return True
    selector_tokens = tuple(
        _pointer_tokens(selector, "MiniChem result selector")
    )
    for pattern in declared_array_patterns:
        if len(selector_tokens) != len(pattern):
            continue
        if all(
            expected == actual
            or (
                expected is None
                and re.fullmatch(r"0|[1-9][0-9]*", actual) is not None
            )
            for expected, actual in zip(pattern, selector_tokens)
        ):
            return True
    return False


def _minichem_contract_value_strings(value: Any) -> set[str]:
    """从合同字段中提取非空字符串。

    输入是任意合同值；动作递归读取字符串、序列和映射值，不修改输入。返回值是去
    空白后的字符串集合；其他类型或空字符串被忽略，不会抛出类型异常。
    """

    values: set[str] = set()
    if isinstance(value, str) and value.strip():
        values.add(value.strip())
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            values.update(_minichem_contract_value_strings(item))
    elif isinstance(value, Mapping):
        for item in value.values():
            values.update(_minichem_contract_value_strings(item))
    return values


def _minichem_output_contract_facts(
    action: Mapping[str, Any],
    pair_contract: Mapping[str, Any] | None,
) -> tuple[
    set[str],
    set[str],
    set[str],
    set[str],
    set[tuple[str | None, ...]],
    bool,
]:
    """读取上游 Action 输出合同的语义类型、公开字段和声明产物名称。

    返回值依次为语义类型、媒体类型、声明产物名称、公开 ``/result`` 路径、数组
    项路径模式和是否存在足以进行 selector 校验的合同事实。缺少冻结 pair 合同且
    没有显式 output schema 时返回 ``has_facts=False``，调用方不得凭经验猜测字段。
    """

    contract_values: list[Mapping[str, Any]] = []
    if isinstance(pair_contract, Mapping):
        output_contract = pair_contract.get("output_contract")
        if isinstance(output_contract, Mapping):
            contract_values.append(output_contract)
        request_contract = pair_contract.get("request_contract")
        if isinstance(request_contract, Mapping):
            nested_output = request_contract.get("output_contract")
            if isinstance(nested_output, Mapping):
                contract_values.append(nested_output)
    for key in ("output_contract", "output_schema", "result_schema"):
        value = action.get(key)
        if isinstance(value, Mapping):
            contract_values.append(value)
    if any(key in action for key in _MINICHEM_RESULT_PATH_KEYS | _MINICHEM_RESULT_SCHEMA_KEYS):
        contract_values.append(action)
    metadata = action.get("metadata")
    if isinstance(metadata, Mapping):
        for key in ("output_contract", "output_schema", "result_schema"):
            value = metadata.get(key)
            if isinstance(value, Mapping):
                contract_values.append(value)
        if any(
            key in metadata
            for key in _MINICHEM_RESULT_PATH_KEYS | _MINICHEM_RESULT_SCHEMA_KEYS
        ):
            contract_values.append(metadata)

    semantic_types: set[str] = set()
    media_types: set[str] = set()
    artifact_names: set[str] = set()
    result_paths: set[str] = set()
    result_array_patterns: set[tuple[str | None, ...]] = set()
    pair_semantic_keys_declared = False
    for contract in contract_values:
        for key in _MINICHEM_OUTPUT_SEMANTIC_KEYS:
            if key in contract:
                pair_semantic_keys_declared = True
                semantic_types.update(
                    _minichem_contract_value_strings(contract[key])
                )
        for key in _MINICHEM_OUTPUT_MEDIA_KEYS:
            if key in contract:
                media_types.update(_minichem_contract_value_strings(contract[key]))
        for key in _MINICHEM_OUTPUT_ARTIFACT_NAME_KEYS:
            if key in contract:
                artifact_names.update(
                    _minichem_contract_value_strings(contract[key])
                )
        for key in _MINICHEM_RESULT_PATH_KEYS:
            if key in contract:
                result_paths.update(
                    _minichem_result_path_from_declaration(
                        contract[key], field_mode="field" in key
                    )
                )
        for key in _MINICHEM_RESULT_SCHEMA_KEYS:
            if key in contract:
                result_paths.update(_minichem_result_schema_paths(contract[key]))
                result_array_patterns.update(
                    _minichem_result_schema_array_patterns(contract[key])
                )

    # Action Catalog 的 primary_output/output_semantic_types 是冻结输出语义事实，
    # 但它们不是公开 result 字段；只能用于 typed artifact selector。即使当前
    # pair 没有 companion，也不能把已经冻结的 primary_output 丢掉。
    if not pair_semantic_keys_declared:
        for key in _MINICHEM_OUTPUT_SEMANTIC_KEYS:
            if key in action:
                semantic_types.update(_minichem_contract_value_strings(action[key]))
    has_facts = bool(
        pair_contract is not None
        or contract_values
        or semantic_types
        or media_types
        or artifact_names
        or result_paths
        or result_array_patterns
    )
    return (
        semantic_types,
        media_types,
        artifact_names,
        result_paths,
        result_array_patterns,
        has_facts,
    )


def _is_existing_public_result_selector(
    selector: str,
    action: Mapping[str, Any],
    semantic_types: set[str],
) -> bool:
    """判断 selector 是否属于目录明确约定的 Gaussian route 字段。

    输入是 selector、Action 描述和语义类型集合；动作只比较固定路径与
    ``GaussianCodeResult`` 语义，不修改输入。返回 ``True`` 表示可兼容保留，其他
    selector 返回 ``False``；缺少语义类型不会猜测结果字段。
    """

    if selector != "/result/gaussian_code":
        return False
    normalized = {value.casefold() for value in semantic_types}
    return "gaussiancoderesult" in {
        value.replace("_", "").replace("-", "") for value in normalized
    } or action.get("primary_output") == "GaussianCodeResult"


def _validate_minichem_binding_source_contract(
    binding: Mapping[str, Any],
    source_step_label: str,
    source_action: Mapping[str, Any],
    source_pair_contract: Mapping[str, Any] | None,
    label: str,
) -> None:
    """按上游 Action 输出合同核对一条 binding 的来源 selector。

    输入是 binding、上游步骤标签、Action、pair 合同和错误标签；动作读取冻结输出
    语义与公开路径并执行严格匹配，不修改 binding。返回值为 ``None``；selector 不受
    合同支持、使用内部路径或缺少必要合同事实时抛出合同校验异常。
    """

    selector = binding.get("source_field")
    if not isinstance(selector, str):
        return
    (
        semantic_types,
        media_types,
        artifact_names,
        result_paths,
        result_array_patterns,
        has_facts,
    ) = _minichem_output_contract_facts(source_action, source_pair_contract)
    source_action_id = source_action.get("action_id", source_action.get("id", "unknown"))
    source_label = (
        f"上游步骤 {source_step_label}（Action {source_action_id}）"
    )
    if selector == "job.stdout":
        # predefined Action 的结果只能通过冻结 output contract 声明的公开
        # result 或 typed Artifact 传递。``job.stdout`` 属于 native/programmable
        # open execution 的作业事实，不能把 Action 的内部日志路径伪装成绑定值。
        _fail(
            f"{label}.source_field selector {selector!r} 不受{source_label}输出合同支持；"
            "predefined Action 不允许使用 job.stdout，请改用公开 result 或 typed artifact selector"
        )
    if selector.startswith("/raw_output"):
        # raw_output 是执行证据的内部包装，不是 Planner 可见的稳定输出根。即使
        # 其中恰好存在 result，也必须改用合同公开的 /result/... selector。
        _fail(
            f"{label}.source_field selector {selector!r} 不受{source_label}输出合同支持；"
            "predefined Action 不允许直接读取 raw_output 内部路径"
        )
    if not selector.startswith("/") and not (
        selector.startswith("artifact.semantic_type=")
        or selector.startswith("artifact.media_type=")
        or selector.startswith("declared_artifact.name=")
    ):
        return
    # /result 与 typed artifact selector 都必须由冻结输出合同逐项支持。没有
    # 合同事实时不能把任意字段名当作结果；唯一例外是 Gaussian route 的
    # 历史公开字段，它由明确的 GaussianCodeResult 输出身份兼容支持。
    if not has_facts and not _is_existing_public_result_selector(
        selector,
        source_action,
        semantic_types,
    ):
        _fail(
            f"{label}.source_field selector {selector!r} 不受{source_label}输出合同支持；"
            "冻结合同没有可用输出事实"
        )
    if (
        selector.startswith("/")
        and _minichem_result_selector_matches_declared_paths(
            selector,
            result_paths,
            result_array_patterns,
        )
    ):
        return
    if _is_existing_public_result_selector(selector, source_action, semantic_types):
        return
    if selector.startswith("/"):
        _fail(
            f"{label}.source_field selector {selector!r} 不受{source_label}输出合同支持；"
            f"可用 artifact semantic type={sorted(semantic_types)!r}；"
            f"公开 result 字段={sorted(result_paths)!r}；"
            "请改用合同声明的 /result/... 或 typed artifact selector"
        )
    if selector.startswith("artifact.semantic_type="):
        expected = selector.split("=", 1)[1]
        if expected not in semantic_types:
            _fail(
                f"{label}.source_field selector {selector!r} 不受{source_label}输出合同支持；"
                f"可用 artifact semantic type={sorted(semantic_types)!r}；"
                f"公开 result 字段={sorted(result_paths)!r}"
            )
    elif selector.startswith("artifact.media_type="):
        expected = selector.split("=", 1)[1]
        if expected not in media_types:
            _fail(
                f"{label}.source_field selector {selector!r} 不受{source_label}输出合同支持；"
                f"可用 artifact media type={sorted(media_types)!r}；"
                f"公开 result 字段={sorted(result_paths)!r}"
            )
    elif selector.startswith("declared_artifact.name="):
        expected = selector.split("=", 1)[1]
        if expected not in artifact_names:
            _fail(
                f"{label}.source_field selector {selector!r} 不受{source_label}输出合同支持；"
                f"可用 declared artifact name={sorted(artifact_names)!r}；"
                f"公开 result 字段={sorted(result_paths)!r}"
            )


def _predefined_step_has_foreach_result(step: Mapping[str, Any]) -> bool:
    """判断同步 Action 是否由执行器逐项运行并发布聚合 items。"""

    request = step.get("request")
    return isinstance(request, Mapping) and isinstance(
        request.get("foreach"), Mapping
    )


def _validate_predefined_foreach_binding_source(
    binding: Mapping[str, Any],
    *,
    source_step_label: str,
    source_action: Mapping[str, Any],
    source_pair_contract: Mapping[str, Any] | None,
    label: str,
) -> None:
    """核对同步 Action foreach 聚合结果供普通 binding 使用的来源字段。"""

    selector = binding.get("source_field")
    aggregate_fields = {
        "/result/items",
        "/result/item_count",
        "/result/failed_item_count",
    }
    if selector in aggregate_fields:
        return
    if isinstance(selector, str) and selector.startswith("/result/"):
        _fail(
            f"{label}.source_field 必须读取 foreach 聚合字段 /result/items、"
            "/result/item_count 或 /result/failed_item_count，不能把单次 Action"
            " result 路径用于批量父结果"
        )
    _validate_minichem_binding_source_contract(
        binding,
        source_step_label,
        source_action,
        source_pair_contract,
        label,
    )


def _validate_predefined_foreach_mapping_source(
    foreach: Mapping[str, Any],
    *,
    source_step_label: str,
    source_action: Mapping[str, Any],
    source_pair_contract: Mapping[str, Any] | None,
    label: str,
) -> None:
    """核对下游 foreach 对同步 Action 聚合 child 的稳定选择方式。

    输入是下游 foreach、上游 Action 及其冻结输出合同；函数只允许读取执行器实际
    发布的 ``/result/items``，并要求用 ``/item_id`` 保留 child 身份。逐项数据要么
    使用上游声明的 typed Artifact，要么使用 ``/result/result/...`` 访问 child 的
    ActionResult.result。字段不受冻结合同时抛出 ``WorkflowContractError``。
    """

    if foreach.get("source_field") != "/result/items":
        _fail(f"{label}.source_field 必须读取 foreach 聚合结果 /result/items")
    if foreach.get("item_id_field") != "/item_id":
        _fail(f"{label}.item_id_field 必须使用 foreach child 的 /item_id")
    item_field = foreach.get("item_field")
    if isinstance(item_field, str) and item_field.startswith(
        ("artifact.semantic_type=", "artifact.media_type=")
    ):
        _validate_minichem_binding_source_contract(
            {"source_field": item_field},
            source_step_label,
            source_action,
            source_pair_contract,
            f"{label}.item_field",
        )
        return
    if isinstance(item_field, str) and item_field.startswith("/result/result/"):
        _validate_minichem_binding_source_contract(
            {"source_field": item_field[len("/result") :]},
            source_step_label,
            source_action,
            source_pair_contract,
            f"{label}.item_field",
        )
        return
    _fail(
        f"{label}.item_field 必须使用上游合同声明的 typed Artifact selector，"
        "或 /result/result/... child 结果路径"
    )


def _validate_minichem_batch_handoff_contract(
    *,
    foreach: Mapping[str, Any],
    source_action: Mapping[str, Any],
    source_pair_contract: Mapping[str, Any] | None,
    target_action_id: Any,
    label: str,
) -> None:
    """核对普通 Action 输出合同声明的 batch_handoff 与请求映射。

    输入是源 Action、pair-specific 输出合同、目标 Action 身份和当前 foreach 请求；
    动作只读取合同与请求并比较 source/item/id/target 字段及条数上限，不修改任何
    输入。返回值为 ``None``；源合同没有 batch_handoff 时保持旧的通用校验行为，
    声明缺失、类型非法、映射不一致或请求超过上限时抛出合同校验异常。
    """

    if not isinstance(source_pair_contract, Mapping):
        return
    output_contract = source_pair_contract.get("output_contract")
    if not isinstance(output_contract, Mapping):
        return
    declaration = output_contract.get("batch_handoff")
    if declaration is None:
        return
    if not isinstance(declaration, Mapping):
        _fail(f"{label}.batch_handoff 必须是 object")
    allowed_fields = {
        "source_field",
        "item_field",
        "item_id_field",
        "target_field",
        "downstream_action_id",
        "max_items",
        "only_successful_items",
        "only_present_items",
        "description",
    }
    unknown_fields = set(declaration) - allowed_fields
    if unknown_fields:
        _fail(
            f"{label}.batch_handoff 含未知字段：{sorted(unknown_fields)}"
        )
    required_fields = {
        "source_field",
        "item_field",
        "item_id_field",
        "target_field",
        "downstream_action_id",
        "max_items",
    }
    missing_fields = sorted(required_fields - set(declaration))
    if missing_fields:
        _fail(
            f"{label}.batch_handoff 缺少声明字段：{missing_fields}"
        )
    text_fields = (
        "source_field",
        "item_field",
        "item_id_field",
        "target_field",
        "downstream_action_id",
    )
    for field_name in text_fields:
        value = declaration.get(field_name)
        if not isinstance(value, str) or not value:
            _fail(
                f"{label}.batch_handoff.{field_name} 必须是非空字符串"
            )
    _pointer_tokens(
        declaration["source_field"],
        f"{label}.batch_handoff.source_field",
    )
    _validate_minichem_item_field_selector(
        declaration["item_field"],
        f"{label}.batch_handoff.item_field",
    )
    _pointer_tokens(
        declaration["item_id_field"],
        f"{label}.batch_handoff.item_id_field",
    )
    _minichem_target_tokens(declaration["target_field"])
    declared_max_items = declaration.get("max_items")
    if (
        isinstance(declared_max_items, bool)
        or not isinstance(declared_max_items, int)
        or not 1 <= declared_max_items <= 32
    ):
        _fail(
            f"{label}.batch_handoff.max_items 必须在 1 到 32 之间"
        )
    foreach_fields = (
        "source_field",
        "item_field",
        "item_id_field",
        "target_field",
    )
    for field_name in foreach_fields:
        if foreach.get(field_name) != declaration[field_name]:
            _fail(
                f"{label}.batch_handoff.{field_name} 与请求 foreach 不一致"
            )
    if target_action_id != declaration["downstream_action_id"]:
        _fail(
            f"{label}.batch_handoff.downstream_action_id "
            f"{declaration['downstream_action_id']!r} 与目标 Action "
            f"{target_action_id!r} 不一致"
        )
    requested_max_items = foreach.get("max_items", 32)
    if (
        isinstance(requested_max_items, bool)
        or not isinstance(requested_max_items, int)
        or not 1 <= requested_max_items <= declared_max_items
    ):
        _fail(
            f"{label}.batch_handoff.max_items 请求值 {requested_max_items!r} "
            f"超过合同上限 {declared_max_items}"
        )
    if "only_present_items" in declaration and type(
        declaration["only_present_items"]
    ) is not bool:
        _fail(f"{label}.batch_handoff.only_present_items 必须是 bool")
    if foreach.get("only_present_items", False) != declaration.get(
        "only_present_items", False
    ):
        _fail(f"{label}.batch_handoff.only_present_items 与请求 foreach 不一致")
    (
        _semantic_types,
        _media_types,
        _artifact_names,
        result_paths,
        _result_array_patterns,
        _has_facts,
    ) = _minichem_output_contract_facts(source_action, source_pair_contract)
    if declaration["source_field"] not in result_paths:
        _fail(
            f"{label}.batch_handoff.source_field "
            f"{declaration['source_field']!r} 不在源 Action 公开 result 合同中"
        )


def _fail(message: str) -> None:
    raise WorkflowContractError(message)


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], label: str
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        _fail(f"{label} 字段不符合固定契约；缺少={missing}，多余={extra}")


def _require_nonempty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{label} 必须是非空字符串")
    return value


def _assert_json_value(value: Any, label: str = "JSON") -> None:
    """确认值能够跨进程稳定保存，拒绝 Python 私有类型和非有限数。"""

    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int) and not isinstance(value, bool):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail(f"{label} 不能包含非有限数")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_json_value(item, f"{label}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                _fail(f"{label} 的对象键必须是字符串")
            _assert_json_value(item, f"{label}.{key}")
        return
    _fail(f"{label} 包含不可序列化类型 {type(value).__name__}")


def canonical_json_text(value: Any) -> str:
    """把同一份数据固定编码为完全相同的 JSON 文本，供哈希和差异比较使用。"""

    _assert_json_value(value)
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _pointer_tokens(pointer: Any, label: str) -> list[str]:
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        _fail(f"{label} 必须是以 / 开头的 JSON Pointer")
    if pointer == "/":
        return [""]
    tokens: list[str] = []
    for raw in pointer[1:].split("/"):
        index = 0
        while index < len(raw):
            if raw[index] == "~":
                if index + 1 >= len(raw) or raw[index + 1] not in {"0", "1"}:
                    _fail(f"{label} 包含非法 JSON Pointer 转义")
                index += 2
            else:
                index += 1
        tokens.append(raw.replace("~1", "/").replace("~0", "~"))
    return tokens


def _pointer_get(value: Any, pointer: str, label: str) -> Any:
    current = value
    for token in _pointer_tokens(pointer, label):
        if isinstance(current, dict):
            if token not in current:
                _fail(f"{label} 指向不存在字段 {pointer}")
            current = current[token]
        elif isinstance(current, list):
            if not token.isdigit() or (len(token) > 1 and token.startswith("0")):
                _fail(f"{label} 的数组下标不合法：{token}")
            index = int(token)
            if index >= len(current):
                _fail(
                    f"{label} 的数组下标越界：{token}；source artifact 未登记"
                )
            current = current[index]
        else:
            _fail(f"{label} 无法继续解析 {pointer}")
    return current


_MISSING_POINTER_VALUE = object()


def _pointer_get_if_present(value: Any, pointer: str, label: str) -> Any:
    """读取可选 JSON Pointer；仅把缺失字段或越界项视为不存在。"""

    current = value
    for token in _pointer_tokens(pointer, label):
        if isinstance(current, dict):
            if token not in current:
                return _MISSING_POINTER_VALUE
            current = current[token]
        elif isinstance(current, list):
            if not token.isdigit() or (len(token) > 1 and token.startswith("0")):
                _fail(f"{label} 的数组下标不合法：{token}")
            index = int(token)
            if index >= len(current):
                return _MISSING_POINTER_VALUE
            current = current[index]
        else:
            _fail(f"{label} 无法继续解析 {pointer}")
    return current


def _collect_minichem_artifact_candidates(value: Any) -> list[Mapping[str, Any]]:
    """从已完成步骤结果中收集带路径和类型的 ArtifactRef。"""

    found: list[Mapping[str, Any]] = []
    if isinstance(value, Mapping):
        if (
            isinstance(value.get("path"), str)
            and isinstance(value.get("media_type"), str)
            and isinstance(value.get("semantic_type"), str)
        ):
            found.append(value)
        for child in value.values():
            found.extend(_collect_minichem_artifact_candidates(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_collect_minichem_artifact_candidates(child))
    return found


def _validate_minichem_item_field_selector(value: Any, label: str) -> str:
    """校验 foreach 单项字段，允许指针或限定的 ArtifactRef 选择器。

    输入是一个 foreach 的 item_field。动作保留原有 JSON Pointer 行为；新增的
    selector 只允许按单项内的 artifact.semantic_type 或 artifact.media_type 选择
    typed ArtifactRef，不允许 stdout、声明名称或 output_artifacts 数组下标猜测。
    返回原字符串；空值、自然语言和不稳定数组选择均抛出合同错误。
    """

    if not isinstance(value, str) or not value:
        _fail(f"{label} 必须是非空字符串")
    if value.startswith("/"):
        tokens = _pointer_tokens(value, label)
        if any(
            token in {"output_artifacts", "declared_artifacts"}
            and position + 1 < len(tokens)
            and tokens[position + 1].isdigit()
            for position, token in enumerate(tokens)
        ):
            _fail(
                f"{label} 不得按 output_artifacts/declared_artifacts 数组下标选择产物；"
                "请使用 artifact.semantic_type=... 或 artifact.media_type=..."
            )
        return value
    if value.startswith("artifact.semantic_type=") or value.startswith(
        "artifact.media_type="
    ):
        for pattern in _MINICHEM_SELECTOR_PATTERNS.values():
            if pattern.fullmatch(value):
                return value
    _fail(
        f"{label} 必须是 JSON Pointer 或 artifact.semantic_type=.../"
        "artifact.media_type=..."
    )


def _select_minichem_item_artifact(
    item: Any,
    selector: str,
    label: str,
) -> dict[str, Any]:
    """在单个 foreach item 内唯一选择 typed ArtifactRef。

    输入是一个已经从上游公开 result 数组取出的 item 和经过合同校验的 selector；
    动作递归查看该 item 内的 ArtifactRef，按 artifact.semantic_type 或
    artifact.media_type 匹配并去重同一 artifact_id。零个、多个或同一身份字段冲突
    都失败关闭，绝不退回父级聚合的扁平 output_artifacts。
    """

    _validate_minichem_item_field_selector(selector, label)
    if selector.startswith("/"):
        _fail(f"{label} 不是 typed Artifact selector")
    key, expected = selector.split("=", 1)
    candidates_by_id: dict[str, Mapping[str, Any]] = {}
    for candidate in _collect_minichem_artifact_candidates(item):
        artifact_id = candidate.get("artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id:
            continue
        if not all(
            isinstance(candidate.get(field), str) and candidate.get(field)
            for field in ("path", "media_type", "semantic_type")
        ):
            continue
        prior = candidates_by_id.get(artifact_id)
        if prior is not None:
            identity_fields = ("path", "media_type", "semantic_type", "sha256")
            if any(prior.get(field) != candidate.get(field) for field in identity_fields):
                _fail(
                    f"{label} 内 artifact_id={artifact_id!r} 的 ArtifactRef 身份冲突"
                )
            continue
        candidates_by_id[artifact_id] = candidate
    if key == "artifact.semantic_type":
        matches = [
            candidate
            for candidate in candidates_by_id.values()
            if candidate.get("semantic_type") == expected
        ]
    elif key == "artifact.media_type":
        matches = [
            candidate
            for candidate in candidates_by_id.values()
            if candidate.get("media_type") == expected
        ]
    else:
        _fail(f"{label} 不支持的 typed Artifact selector：{selector}")
    if len(matches) != 1:
        _fail(
            f"{label} selector {selector!r} 在当前 item 内命中 {len(matches)} 个"
            " ArtifactRef，必须恰好命中一个"
        )
    return deepcopy(dict(matches[0]))


def _normalized_minichem_artifact_candidates(value: Any) -> list[Mapping[str, Any]]:
    """只读取 normalized output_artifacts/raw_output，避免同一产物重复计数。"""

    if not isinstance(value, Mapping):
        return []
    candidates: list[Mapping[str, Any]] = []
    for key in ("output_artifacts", "declared_artifacts"):
        raw = value.get(key)
        if isinstance(raw, list):
            candidates.extend(
                item
                for item in raw
                if isinstance(item, Mapping)
                and isinstance(item.get("artifact_id"), str)
            )
    raw_output = value.get("raw_output")
    if isinstance(raw_output, Mapping):
        nested = raw_output.get("output_artifacts")
        if isinstance(nested, list):
            candidates.extend(
                item
                for item in nested
                if isinstance(item, Mapping)
                and isinstance(item.get("artifact_id"), str)
            )
    unique: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for item in candidates:
        artifact_id = item.get("artifact_id")
        if artifact_id not in seen:
            seen.add(artifact_id)
            unique.append(item)
    return unique


def _normalized_minichem_declared_artifacts(value: Any) -> list[Mapping[str, Any]]:
    """读取 collect 返回的声明产物，允许其暂时没有 ArtifactRef 身份。"""

    if not isinstance(value, Mapping):
        return []
    values: list[Mapping[str, Any]] = []
    for source in (value, value.get("raw_output") if isinstance(value.get("raw_output"), Mapping) else None):
        if not isinstance(source, Mapping):
            continue
        declared = source.get("declared_artifacts")
        if isinstance(declared, list):
            values.extend(item for item in declared if isinstance(item, Mapping))
    return values


def _select_minichem_source_value(
    completed_step: Any,
    selector: str,
    label: str,
) -> Any:
    """按固定 selector 选择一个上游值，零个或多个候选都拒绝。

    预定义 Action 的公开响应字段位于 MCP 原始结果中，而执行证据会把该结果包在
    ``raw_output`` 下。模型和 Action 合同仍使用 ``/result/...`` 这类公开指针；
    解析时只对该公开根映射到已经核验的 ``raw_output``，不会把内部证据包装结构
    泄漏到 Planner Prompt。其他 JSON Pointer 继续从规范步骤根读取。
    """

    if selector.startswith("/"):
        if selector == "/result" or selector.startswith("/result/"):
            if not isinstance(completed_step, Mapping) or not isinstance(
                completed_step.get("raw_output"), Mapping
            ):
                _fail(f"{label} 缺少可读取公开 result 的 raw_output")
            return _pointer_get(completed_step["raw_output"], selector, label)
        return _pointer_get(completed_step, selector, label)
    if selector == "job.stdout":
        if not isinstance(completed_step, Mapping):
            _fail(f"{label} 无法读取 job.stdout")
        raw_output = completed_step.get("raw_output")
        job = raw_output.get("job") if isinstance(raw_output, Mapping) else None
        stdout = job.get("stdout_path") if isinstance(job, Mapping) else None
        if not isinstance(stdout, str) or not stdout:
            _fail(f"{label} 未找到唯一 job.stdout")
        return stdout
    candidates = _normalized_minichem_artifact_candidates(completed_step)
    if not candidates:
        candidates = _collect_minichem_artifact_candidates(completed_step)
    key, expected = selector.split("=", 1)
    if key == "artifact.media_type":
        matches = [item for item in candidates if item.get("media_type") == expected]
    elif key == "artifact.semantic_type":
        matches = [item for item in candidates if item.get("semantic_type") == expected]
    else:
        declarations = [
            item
            for item in _normalized_minichem_declared_artifacts(completed_step)
            if item.get("name") == expected or item.get("artifact_name") == expected
        ]
        if len(declarations) != 1:
            _fail(
                f"{label} 选择器 {selector} 命中 {len(declarations)} 个声明产物，必须恰好命中一个"
            )
        declaration_path = declarations[0].get("workspace_path", declarations[0].get("path"))
        matches = [item for item in candidates if item.get("path") == declaration_path]
    if len(matches) != 1:
        _fail(
            f"{label} 选择器 {selector} 命中 {len(matches)} 个 Artifact，必须恰好命中一个"
        )
    return deepcopy(dict(matches[0]))


def _validate_minichem_source_selector(value: Any, label: str) -> str:
    """校验上游结果选择器，避免模型用不稳定的自然语言或数组猜测。"""

    if not isinstance(value, str) or not value:
        _fail(f"{label} 必须是非空字符串")
    if value.startswith("/"):
        tokens = _pointer_tokens(value, label)
        if (
            len(tokens) >= 2
            and tokens[0] in {"output_artifacts", "declared_artifacts"}
            and tokens[1].isdigit()
        ):
            _fail(
                f"{label} 不得按 output_artifacts/declared_artifacts 数组下标选择产物；"
                "请使用 artifact.semantic_type/media_type 或 declared_artifact.name"
            )
        return value
    if value == "job.stdout":
        return value
    for pattern_name, pattern in _MINICHEM_SELECTOR_PATTERNS.items():
        if pattern.fullmatch(value):
            selected = value.split("=", 1)[1]
            if not selected:
                _fail(f"{label} 的 {pattern_name} 选择值不能为空")
            return value
    _fail(
        f"{label} 必须是 JSON Pointer、artifact.media_type=...、"
        "artifact.semantic_type=...、declared_artifact.name=... 或 job.stdout"
    )


def _validate_minichem_binding(
    binding: Any,
    label: str,
    *,
    allow_short_refs: bool,
) -> dict[str, str]:
    """校验一条 MiniChem 绑定的来源身份、选择器和目标路径语法。"""

    if not isinstance(binding, Mapping) or set(binding) != {
        "source_step_ref",
        "source_field",
        "target_field",
    }:
        _fail(f"{label} 字段不完整")
    source_step_ref = binding.get("source_step_ref")
    if not isinstance(source_step_ref, str) or not source_step_ref:
        _fail(f"{label}.source_step_ref 必须是非空字符串")
    if not allow_short_refs and _SHORT_REF_PATTERN.fullmatch(source_step_ref):
        _fail(f"{label}.source_step_ref 不能持久化短引用")
    source_field = _validate_minichem_source_selector(
        binding.get("source_field"), f"{label}.source_field"
    )
    target_field = binding.get("target_field")
    _minichem_target_tokens(target_field)
    return {
        "source_step_ref": source_step_ref,
        "source_field": source_field,
        "target_field": str(target_field),
    }


def _validate_task_relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{label} 路径不能为空")
    if "\\" in value:
        _fail(f"{label} 必须使用 POSIX 路径")
    path = PurePosixPath(value)
    if path.is_absolute():
        _fail(f"{label} 禁止绝对路径")
    if any(part in {"", ".", ".."} for part in path.parts):
        _fail(f"{label} 路径不能包含 . 或 ..")
    return path.as_posix()


def _validate_open_evaluation_budget_record(value: Any, label: str) -> dict[str, Any]:
    """核对 open job 请求携带的不可修改任务资源政策。"""

    expected = {
        "cpu_cores",
        "memory_mb",
        "gpu_count",
        "source",
        "agent_controllable",
        "scope",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        _fail(f"{label} 字段不完整")
    if (
        type(value["cpu_cores"]) is not int
        or value["cpu_cores"] < 1
        or type(value["memory_mb"]) is not int
        or value["memory_mb"] < 128
        or type(value["gpu_count"]) is not int
        or value["gpu_count"] < 0
        or value["source"] != "evaluation_policy"
        or value["agent_controllable"] is not False
        or value["scope"] != "per_task"
    ):
        _fail(f"{label} 不符合任务资源政策")
    return deepcopy(dict(value))


def _validate_open_timeout_policy_record(value: Any, label: str) -> dict[str, Any]:
    """核对 open job 请求携带的不可修改执行超时政策。"""

    expected = {"execution_class", "timeout_seconds", "source", "agent_controllable"}
    if not isinstance(value, Mapping) or set(value) != expected:
        _fail(f"{label} 字段不完整")
    if (
        value["execution_class"] not in {"compute", "fast"}
        or type(value["timeout_seconds"]) is not int
        or value["timeout_seconds"] < 1
        or value["source"] != "evaluation_policy"
        or value["agent_controllable"] is not False
    ):
        _fail(f"{label} 不符合执行超时政策")
    return deepcopy(dict(value))


def safe_join_task_path(task_root: str | Path, relative_path: Any) -> Path:
    """在任务目录内解析相对路径，并阻止 ``..`` 或符号链接绕出边界。"""

    relative = _validate_task_relative_path(relative_path, "任务文件")
    root = Path(task_root).resolve(strict=False)
    current = root
    for part in PurePosixPath(relative).parts:
        current = current / part
        if current.is_symlink():
            _fail(f"任务路径包含 symlink，可能越过任务边界：{relative}")
    candidate = (root / relative).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError:
        _fail(f"任务路径越过边界：{relative}")
    return candidate


def expected_outputs_for_tool(tool: Mapping[str, Any]) -> str:
    """拒绝旧 ToolSpec 输出声明，提示调用方改用 MiniChem Action 元数据。"""

    _ = tool
    _fail("旧 ToolSpec expected_outputs 已停用；MiniChem 输出由 Action Catalog 派生")



def _hypothesis_claim_statements(hypothesis: Mapping[str, Any]) -> list[str]:
    """读取当前唯一核心判断，并兼容旧快照的正文校验。"""

    # 延迟导入可避免 memory package 初始化服务时与本模块形成循环依赖。
    from chemistry_multiagent.memory.contracts import (
        validate_structured_hypothesis_claims,
    )

    if not isinstance(hypothesis, Mapping):
        _fail("hypothesis 必须是对象")
    reasoning = hypothesis.get("reasoning")
    if (
        "core_claim" not in hypothesis
        and isinstance(reasoning, str)
        and reasoning.strip()
    ):
        if len(reasoning.strip()) > 3200:
            _fail("hypothesis.reasoning 最长为 3200 字符")
        return [reasoning.strip()]
    try:
        compatibility = validate_structured_hypothesis_claims(hypothesis)
    except ValueError as exc:
        _fail(f"hypothesis 科学正文未通过固定契约：{exc}")
    # 旧快照的 claims/core_claim 只用于校验并转换成同一段完整 reasoning；新的
    # 工作流目标不再复制旧版 core_claim 字段。
    return [compatibility["reasoning"]]


def canonicalize_generation_output(
    wire: Mapping[str, Any],
    catalog_snapshot: Mapping[str, Any],
    id_allocator: Callable[[str], str],
    *,
    hypothesis: Mapping[str, Any],
    expected_strategy_name: str | None = None,
    task_root: str | Path | None = None,
    gaussian_route_policy: GaussianRoutePolicy | None = None,
) -> dict[str, Any]:
    """把模型的短引用工作流转换为可持久化的正式工作流。

    全部结构先通过校验，之后才分配正式身份。这样即使模型输出无效，也不会
    消耗身份序列或留下半份工作流。
    """

    if not is_minichem_catalog(catalog_snapshot):
        _fail("工作流生成只接受 MiniChem Action Catalog；旧 ToolSpec 路径已停用")
    return _canonicalize_minichem_generation_output(
        wire,
        catalog_snapshot,
        id_allocator,
        hypothesis=hypothesis,
        expected_strategy_name=expected_strategy_name,
        gaussian_route_policy=gaussian_route_policy,
    )


def _canonicalize_minichem_generation_output(
    wire: Mapping[str, Any],
    catalog_snapshot: Mapping[str, Any],
    id_allocator: Callable[[str], str],
    *,
    hypothesis: Mapping[str, Any],
    expected_strategy_name: str | None,
    gaussian_route_policy: GaussianRoutePolicy | None,
) -> dict[str, Any]:
    """把 MiniChem Action 生成结果转换为正式 action/backend/request 工作流。"""

    if not isinstance(wire, Mapping) or set(wire) != {"steps"}:
        _fail("MiniChem workflow generation 只允许 steps 字段")
    records = minichem_action_records(catalog_snapshot)
    source_steps = wire.get("steps")
    if not isinstance(source_steps, list) or not source_steps:
        _fail("MiniChem workflow.steps 必须是非空数组")
    formal_ids: dict[str, str] = {}
    for index, item in enumerate(source_steps):
        if not _valid_minichem_step_keys(item, generation=True):
            _fail(f"MiniChem generation steps[{index}] 字段不完整")
        ref = item.get("step_ref")
        if not isinstance(ref, str) or _SHORT_REF_PATTERN.fullmatch(ref) is None:
            _fail("MiniChem generation step_ref 必须是短引用")
        if ref in formal_ids:
            _fail(f"重复 step_ref：{ref}")
        formal_ids[ref] = _require_nonempty_text(id_allocator("step_id"), "正式 step_id")
    steps: list[dict[str, Any]] = []
    for index, item in enumerate(source_steps):
        action_id = item.get("action_id")
        backend_id = item.get("backend_id")
        execution_layer = _minichem_execution_layer(
            item,
            records.get(action_id) if isinstance(action_id, str) else None,
        )
        if execution_layer == "predefined_actions":
            if action_id not in records:
                _fail(f"MiniChem Action 未在冻结目录中：{action_id}")
            if not isinstance(backend_id, str) or not backend_id:
                _fail("MiniChem predefined_actions 步骤必须声明 backend_id")
            request = _validate_minichem_request(
                item.get("request"),
                records[action_id],
                backend_id,
                f"steps[{index}].request",
                allow_short_refs=True,
                catalog_snapshot=catalog_snapshot,
            )
        else:
            if action_id is not None or backend_id is not None:
                _fail("MiniChem open-execution 步骤不得声明 action_id/backend_id")
            request = _validate_minichem_job_request(
                item.get("request"),
                execution_layer,
                f"steps[{index}].request",
                allow_short_refs=True,
                catalog_snapshot=catalog_snapshot,
            )
        depends = item.get("depends_on")
        if not isinstance(depends, list) or len(depends) != len(set(depends)):
            _fail("MiniChem depends_on 必须是互异数组")
        dependency_sources = [
            str(binding["source_step_ref"]) for binding in request["bindings"]
        ]
        foreach = request.get("foreach")
        if isinstance(foreach, Mapping) and foreach["source_step_ref"] not in dependency_sources:
            dependency_sources.append(str(foreach["source_step_ref"]))
        if any(source not in depends for source in dependency_sources):
            _fail("MiniChem depends_on 必须覆盖 request.bindings/foreach 的来源")
        for dependency in depends:
            if dependency not in formal_ids or list(formal_ids).index(dependency) >= index:
                _fail("MiniChem depends_on 只能引用更早步骤")
        for binding in request["bindings"]:
            binding["source_step_ref"] = formal_ids[binding["source_step_ref"]]
        if isinstance(foreach, MutableMapping):
            foreach["source_step_ref"] = formal_ids[foreach["source_step_ref"]]
        canonical_step = {
                "step_id": formal_ids[item["step_ref"]],
                "step_name": _require_nonempty_text(item.get("step_name"), "step_name"),
                "description": _require_nonempty_text(item.get("description"), "description"),
                "request": request,
                "expected_outputs": (
                    _minichem_expected_outputs(
                        records[action_id],
                        pair_contract=_minichem_pair_request_contract(
                            catalog_snapshot,
                            action_id,
                            backend_id,
                        ),
                    )
                    if execution_layer == "predefined_actions"
                    else {"execution_layer": execution_layer}
                ),
                "depends_on": [formal_ids[ref] for ref in depends],
            }
        if execution_layer == "predefined_actions":
            canonical_step["action_id"] = action_id
            canonical_step["backend_id"] = backend_id
        else:
            canonical_step["execution_layer"] = execution_layer
        steps.append(canonical_step)
    strategy_name = _require_nonempty_text(
        expected_strategy_name or hypothesis.get("strategy_name"),
        "hypothesis.strategy_name",
    )
    claims = _hypothesis_claim_statements(hypothesis)
    workflow = {
        "workflow_name": f"{strategy_name}工作流",
        "goal": claims[0],
        "strategy_name": strategy_name,
        "steps": steps,
    }
    validate_canonical_workflow(
        workflow,
        catalog_snapshot,
        gaussian_route_policy=gaussian_route_policy,
    )
    return workflow


def validate_canonical_workflow(
    workflow: Mapping[str, Any],
    catalog_snapshot: Mapping[str, Any],
    *,
    gaussian_route_policy: GaussianRoutePolicy | None = None,
) -> None:
    """确认正式 MiniChem 工作流的步骤顺序、请求身份和 Gaussian 路由边界。"""

    if not is_minichem_catalog(catalog_snapshot):
        _fail("canonical workflow 只接受 MiniChem Action Catalog；旧 ToolSpec 路径已停用")
    _validate_minichem_workflow(
        workflow,
        catalog_snapshot,
        gaussian_route_policy=gaussian_route_policy,
    )



def _validate_minichem_workflow(
    workflow: Mapping[str, Any],
    catalog_snapshot: Mapping[str, Any],
    *,
    gaussian_route_policy: GaussianRoutePolicy | None = None,
) -> None:
    """核对正式 MiniChem 工作流的 Action、请求和依赖边界。"""

    if not isinstance(workflow, Mapping):
        _fail("canonical workflow 必须是 object")
    _require_exact_keys(workflow, _WORKFLOW_KEYS, "canonical workflow")
    for field in ("workflow_name", "goal", "strategy_name"):
        _require_nonempty_text(workflow.get(field), f"workflow.{field}")
    records = minichem_action_records(catalog_snapshot)
    steps = workflow.get("steps")
    if not isinstance(steps, list) or not steps:
        _fail("MiniChem workflow.steps 必须是非空数组")
    formal_ids: list[str] = []
    for index, step in enumerate(steps):
        if not _valid_minichem_step_keys(step, generation=False):
            _fail(f"workflow.steps[{index}] 字段不完整")
        step_id = _require_nonempty_text(step.get("step_id"), f"steps[{index}].step_id")
        if _SHORT_REF_PATTERN.fullmatch(step_id) or step_id in formal_ids:
            _fail("MiniChem canonical step_id 非法或重复")
        action_id = step.get("action_id")
        backend_id = step.get("backend_id")
        execution_layer = _minichem_execution_layer(
            step,
            records.get(action_id) if isinstance(action_id, str) else None,
        )
        if execution_layer == "predefined_actions":
            if action_id not in records or not isinstance(backend_id, str) or not backend_id:
                _fail(f"steps[{index}] 的 Action/Backend 身份非法")
            _minichem_execution_layer(step, records[action_id])
            request = _validate_minichem_request(
                step.get("request"),
                records[action_id],
                backend_id,
                f"steps[{index}].request",
                allow_short_refs=False,
                catalog_snapshot=catalog_snapshot,
            )
            expected_value = _minichem_expected_outputs(
                records[action_id],
                pair_contract=_minichem_pair_request_contract(
                    catalog_snapshot,
                    action_id,
                    backend_id,
                ),
            )
        else:
            if action_id is not None or backend_id is not None:
                _fail(f"steps[{index}] open-execution 不得含 Action/Backend 身份")
            request = _validate_minichem_job_request(
                step.get("request"),
                execution_layer,
                f"steps[{index}].request",
                allow_short_refs=False,
                catalog_snapshot=catalog_snapshot,
            )
            expected_value = {"execution_layer": execution_layer}
        _require_nonempty_text(step.get("step_name"), f"steps[{index}].step_name")
        _require_nonempty_text(step.get("description"), f"steps[{index}].description")
        expected = step.get("expected_outputs")
        if expected != expected_value:
            _fail(f"steps[{index}].expected_outputs 与 Action Catalog 不一致")
        depends = step.get("depends_on")
        if not isinstance(depends, list) or len(depends) != len(set(depends)):
            _fail(f"steps[{index}].depends_on 必须是互异数组")
        earlier = set(formal_ids)
        if any(not isinstance(item, str) or item not in earlier for item in depends):
            _fail("MiniChem depends_on 必须引用更早正式 step_id")
        bindings = request.get("bindings", [])
        sources = [item["source_step_ref"] for item in bindings]
        foreach = request.get("foreach")
        if isinstance(foreach, Mapping) and foreach["source_step_ref"] not in sources:
            sources.append(foreach["source_step_ref"])
        if any(source not in depends for source in sources):
            _fail("MiniChem request.bindings/foreach 来源必须出现在 depends_on 中")
        formal_ids.append(step_id)
    steps_by_id = {
        step.get("step_id"): step
        for step in steps
        if isinstance(step, Mapping) and isinstance(step.get("step_id"), str)
    }
    step_positions = {
        step.get("step_id"): position
        for position, step in enumerate(steps)
        if isinstance(step, Mapping) and isinstance(step.get("step_id"), str)
    }
    for index, step in enumerate(steps):
        request = step.get("request")
        bindings = request.get("bindings", []) if isinstance(request, Mapping) else []
        for binding_index, binding in enumerate(bindings):
            if not isinstance(binding, Mapping):
                continue
            source_step = steps_by_id.get(binding.get("source_step_ref"))
            if not isinstance(source_step, Mapping):
                continue
            source_action_id = source_step.get("action_id")
            if source_step.get("execution_layer", "predefined_actions") != "predefined_actions":
                # native job/analysis 的输出由运行时事实决定，不能套用 Action 合同。
                continue
            if not isinstance(source_action_id, str) or source_action_id not in records:
                continue
            source_backend_id = source_step.get("backend_id")
            source_pair_contract = (
                _minichem_pair_request_contract(
                    catalog_snapshot,
                    source_action_id,
                    source_backend_id,
                )
                if isinstance(source_backend_id, str)
                else None
            )
            source_label = f"steps[{step_positions[binding['source_step_ref']]}]"
            binding_label = f"steps[{index}].request.bindings[{binding_index}]"
            if _predefined_step_has_foreach_result(source_step):
                _validate_predefined_foreach_binding_source(
                    binding,
                    source_step_label=source_label,
                    source_action=records[source_action_id],
                    source_pair_contract=source_pair_contract,
                    label=binding_label,
                )
            else:
                _validate_minichem_binding_source_contract(
                    binding,
                    source_label,
                    records[source_action_id],
                    source_pair_contract,
                    binding_label,
                )
        foreach = request.get("foreach") if isinstance(request, Mapping) else None
        if not isinstance(foreach, Mapping):
            continue
        source_step_ref = foreach.get("source_step_ref")
        source_step = steps_by_id.get(source_step_ref)
        if not isinstance(source_step, Mapping):
            continue
        source_action_id = source_step.get("action_id")
        if source_step.get("execution_layer", "predefined_actions") != "predefined_actions":
            # open-execution 的输出由运行时 Artifact/Result 事实确定，不能凭
            # Action 输出合同提前猜测；执行前仍会核对其实际完成结果。
            continue
        if not isinstance(source_action_id, str) or source_action_id not in records:
            continue
        if _is_gaussian_native_step(step):
            # native software 的逐项输入由当前 NativeJobRequest 和运行时结果
            # 合同核对，不假定某个 converter、结果字段或上游 Action。
            continue
        source_backend_id = source_step.get("backend_id")
        source_pair_contract = (
            _minichem_pair_request_contract(
                catalog_snapshot,
                source_action_id,
                source_backend_id,
            )
            if isinstance(source_backend_id, str)
            else None
        )
        if _predefined_step_has_foreach_result(source_step):
            _validate_predefined_foreach_mapping_source(
                foreach,
                source_step_label=f"steps[{step_positions[source_step_ref]}]",
                source_action=records[source_action_id],
                source_pair_contract=source_pair_contract,
                label=f"steps[{index}].request.foreach",
            )
            continue
        target_action_id = step.get("action_id")
        _validate_minichem_batch_handoff_contract(
            foreach=foreach,
            source_action=records[source_action_id],
            source_pair_contract=source_pair_contract,
            target_action_id=target_action_id,
            label=f"steps[{index}].request.foreach",
        )
        _validate_minichem_binding_source_contract(
            {"source_field": foreach.get("source_field")},
            f"steps[{step_positions[source_step_ref]}]",
            records[source_action_id],
            source_pair_contract,
            f"steps[{index}].request.foreach",
        )
    _validate_gaussian_workflow_chain(
        steps,
        records,
        gaussian_route_policy=gaussian_route_policy,
    )


def _compact_gaussian_text(value: Any) -> str:
    """把公开身份压缩成小写字母数字，便于稳定识别 Gaussian 名称。"""

    if not isinstance(value, str):
        return ""
    return "".join(character for character in value.casefold() if character.isalnum())


def _is_gaussian_native_step(step: Mapping[str, Any]) -> bool:
    """判断 native 步骤是否实际请求 Gaussian，而不是笼统的 open job。"""

    if step.get("execution_layer") != "native_software":
        return False
    request = step.get("request")
    if not isinstance(request, Mapping):
        return False
    software_id = _compact_gaussian_text(request.get("software_id"))
    executable = _compact_gaussian_text(request.get("executable"))
    return (
        "gaussian" in software_id
        or "gaussian" in executable
        or executable in {"g03", "g09", "g16", "g98"}
    )


def _normalized_minichem_target(value: Any) -> str:
    """将点路径和 JSON Pointer 统一为同一种绑定目标表示。"""

    if not isinstance(value, str):
        return ""
    return value.lstrip("/").replace("/", ".")


def _is_typed_gaussian_calculation_action(
    action_id: str,
    action: Mapping[str, Any],
    backend_id: str | None = None,
    *,
    gaussian_route_policy: GaussianRoutePolicy | None = None,
) -> bool:
    """保留旧内部接口，但不再用名称或 route policy 拒绝目录声明的 pair。

    Action/Backend 是否能回答当前问题由冻结 catalog metadata、request/output contract
    和 workflow 连接共同决定；本模块不把任意 Gaussian Action 改写成固定准备链。
    """

    del action_id, action, backend_id, gaussian_route_policy
    return False


def _validate_gaussian_workflow_chain(
    steps: Sequence[Mapping[str, Any]],
    records: Mapping[str, Mapping[str, Any]],
    *,
    gaussian_route_policy: GaussianRoutePolicy | None = None,
) -> None:
    """保留旧校验入口；native staged input 安全性由通用请求合同负责。

    输入已由 MiniChem workflow validator 校验。动作不要求特定 codegen、converter
    或 native 三段链，也不按题面选择路线；通用的字段、路径、依赖和 Artifact 连接
    校验仍在上层合同中执行。
    """

    del steps, records, gaussian_route_policy



def _is_minichem_workflow(workflow: Mapping[str, Any]) -> bool:
    """识别采用 MiniChem request 的工作流，包括首步就是 open job 的情况。"""

    steps = workflow.get("steps") if isinstance(workflow, Mapping) else None
    if not isinstance(steps, list) or not steps:
        return False
    for step in steps:
        if not isinstance(step, Mapping) or not isinstance(step.get("request"), Mapping):
            return False
        layer = step.get("execution_layer", "predefined_actions")
        if layer not in _MINICHEM_EXECUTION_LAYER_VALUES:
            return False
        if layer == "predefined_actions":
            if not isinstance(step.get("action_id"), str) or not isinstance(
                step.get("backend_id"), str
            ):
                return False
        elif "action_id" in step or "backend_id" in step:
            return False
    return True


def _project_minichem_workflow_refs(workflow: Mapping[str, Any]) -> dict[str, Any]:
    """将 MiniChem 正式步骤投影为模型可读的短引用。"""

    source = deepcopy(workflow)
    step_formal_to_short = {
        step["step_id"]: f"s{index}"
        for index, step in enumerate(source["steps"], start=1)
    }
    projected_steps: list[dict[str, Any]] = []
    for step in source["steps"]:
        request = deepcopy(dict(step["request"]))
        bindings = request.get("bindings", [])
        if not isinstance(bindings, list):
            _fail("MiniChem request.bindings 必须是数组")
        for binding in bindings:
            source_id = binding.get("source_step_ref")
            if source_id not in step_formal_to_short:
                _fail(f"投影时发现未知 MiniChem source_step_ref：{source_id}")
            binding["source_step_ref"] = step_formal_to_short[source_id]
        foreach = request.get("foreach")
        if isinstance(foreach, MutableMapping):
            source_id = foreach.get("source_step_ref")
            if source_id not in step_formal_to_short:
                _fail(f"投影时发现未知 MiniChem foreach source_step_ref：{source_id}")
            foreach["source_step_ref"] = step_formal_to_short[source_id]
        depends_on = []
        for dependency in step.get("depends_on", []):
            if dependency not in step_formal_to_short:
                _fail(f"投影时发现未知 MiniChem depends_on：{dependency}")
            depends_on.append(step_formal_to_short[dependency])
        projected_step = {
                "step_ref": step_formal_to_short[step["step_id"]],
                "step_name": step["step_name"],
                "description": step["description"],
                "request": request,
                "depends_on": depends_on,
            }
        if step.get("execution_layer") in {"native_software", "programmable_analysis"}:
            projected_step["execution_layer"] = step["execution_layer"]
        else:
            projected_step["action_id"] = step["action_id"]
            projected_step["backend_id"] = step["backend_id"]
        projected_steps.append(projected_step)
    return {
        "workflow": {
            "workflow_name": source["workflow_name"],
            "goal": source["goal"],
            "strategy_name": source["strategy_name"],
            "steps": projected_steps,
        },
        "ref_map": {
            "step_refs": {
                short: formal for formal, short in step_formal_to_short.items()
            },
        },
    }


def project_workflow_refs(workflow: Mapping[str, Any]) -> dict[str, Any]:
    """为 MiniChem 正式工作流生成短引用副本和身份映射。

    输入必须是已验证或至少具备 MiniChem request 结构的工作流；旧 ToolSpec
    工作流不再投影，避免修订协议在两套字段之间静默切换。
    """

    if not _is_minichem_workflow(workflow):
        _fail("工作流投影只接受 MiniChem request 工作流；旧 ToolSpec 路径已停用")
    return _project_minichem_workflow_refs(workflow)



def _minichem_target_tokens(target_field: Any) -> list[str]:
    """把 MiniChem 绑定目标解析为 Action 或 open job 请求字段路径。"""

    if not isinstance(target_field, str) or not target_field:
        _fail("MiniChem target_field 必须是非空字符串")
    if target_field.startswith("/"):
        tokens = _pointer_tokens(target_field, "MiniChem target_field")
    else:
        tokens = target_field.split(".")
        if any(not token for token in tokens):
            _fail("MiniChem target_field 点路径不能包含空字段")
    allowed_roots = {
        "inputs",
        "method_spec",
        "action_settings",
        "resource_limits",
        "source_id",
        "component_backends",
        "software_id",
        "executable",
        "arguments",
        "staged_inputs",
        "stdin_target",
        "calculation_intent",
        "label",
        "parent_job_id",
        "runtime",
        "script_path",
        "script_target",
        "required_modules",
        "required_module_versions",
        "required_symbols",
        "outputs",
        "execution_policy",
        "external_execution",
    }
    if len(tokens) < 1 or tokens[0] not in allowed_roots:
        _fail(
            "MiniChem target_field 必须从合法 Action/native/analysis 请求字段开始"
        )
    return tokens


def _set_minichem_target_value(
    request: MutableMapping[str, Any],
    tokens: Sequence[str],
    value: Any,
) -> None:
    """将绑定值写入 MiniChem 请求的嵌套字段，并拒绝覆盖 section。"""

    current: Any = request
    for token in tokens[:-1]:
        if isinstance(current, MutableMapping):
            if token not in current:
                _fail(f"MiniChem target_field 中间路径不存在：{token}")
            current = current[token]
        elif isinstance(current, list) and token.isdigit():
            index = int(token)
            if index < 0 or index >= len(current):
                _fail(f"MiniChem target_field 数组下标越界：{token}")
            current = current[index]
        else:
            _fail(f"MiniChem target_field 中间路径不可遍历：{token}")
    leaf = tokens[-1]
    if isinstance(current, MutableMapping):
        # binding 是请求中显式声明的数据边；模型可以保留一个可读占位值，
        # 执行时由已完成上游的不可变结果覆盖它。
        current[leaf] = deepcopy(value)
        return
    if isinstance(current, list) and leaf.isdigit():
        index = int(leaf)
        if index < 0 or index >= len(current):
            _fail(f"MiniChem target_field 数组下标越界：{leaf}")
        current[index] = deepcopy(value)
        return
    _fail(f"MiniChem target_field 不能写入：{'.'.join(tokens)}")


def _minichem_foreach_target_value(
    value: Any,
    target_tokens: Sequence[str],
) -> Any:
    """把逐项 ArtifactRef 转成请求字段需要的稳定值。

    输入是 foreach 的 item_field 选出的值和已校验的目标路径；动作只在目标是
    ``source_path`` 且值完整声明为 ArtifactRef 时读取其 path。这样 Gaussian
    子请求保存的是实际 deck 路径，而不是把整个 ArtifactRef 或 output_artifacts
    数组猜成输入；其他目标仍保留原始 JSON 值。缺少 path 时返回原值，随后由
    请求合同或执行器给出明确的失败原因。
    """

    if target_tokens[-1] != "source_path" or not isinstance(value, Mapping):
        return value
    path = value.get("path")
    if not isinstance(path, str) or not path:
        return value
    required_ref_fields = (
        "artifact_id",
        "semantic_type",
        "media_type",
    )
    if not all(isinstance(value.get(field), str) and value.get(field) for field in required_ref_fields):
        return value
    return path


def _minichem_foreach_item_provenance(
    item: Any,
    item_id: str,
    index: int,
    source_step_ref: str,
) -> dict[str, Any]:
    """提取 foreach item 的稳定来源身份，供子作业和父结果复用。

    输入是上游公开结果数组中的单项；动作只复制 item_id、数组位置、来源步骤、
    pair/candidate/direction/reaction_orientation 等标量身份字段，以及实际发现的
    artifact_id 列表，不复制任意 lineage/provenance 对象、原始日志或路径集合。
    返回值始终包含规范化 item_id 和数组位置；JSON 值异常时由调用方统一拒绝。
    """

    provenance: dict[str, Any] = {
        "item_id": item_id,
        "item_index": index,
        "source_step_ref": source_step_ref,
    }
    if not isinstance(item, Mapping):
        return provenance
    nested_result = item.get("result")
    declared_provenance = item.get("provenance")
    # 只允许有限的标量身份字段进入 manifest；不要把任意 provenance/lineage
    # 对象（其中可能包含日志、路径或大数组）原样复制到 checkpoint。
    for field in (
        "pair_id",
        "candidate_id",
        "direction",
        "reaction_orientation",
    ):
        value = item.get(field)
        if value is None and isinstance(declared_provenance, Mapping):
            value = declared_provenance.get(field)
        if value is None and isinstance(nested_result, Mapping):
            value = nested_result.get(field)
        if isinstance(value, (str, int, float, bool)):
            provenance[field] = deepcopy(value)
    artifact_ids: list[str] = []
    gaussian_artifact_ids: list[str] = []
    for artifact in _collect_minichem_artifact_candidates(item):
        artifact_id = artifact.get("artifact_id")
        if isinstance(artifact_id, str) and artifact_id and artifact_id not in artifact_ids:
            artifact_ids.append(artifact_id)
            if artifact.get("semantic_type") == "GaussianInputDeck":
                gaussian_artifact_ids.append(artifact_id)
    if artifact_ids:
        provenance["source_artifact_ids"] = artifact_ids
    if gaussian_artifact_ids:
        provenance["gaussian_input_deck_artifact_ids"] = gaussian_artifact_ids
    return provenance


def resolve_minichem_step_request(
    step: Mapping[str, Any],
    completed_steps: Mapping[str, Any],
) -> dict[str, Any]:
    """用已完成步骤的数据解析一个 MiniChem 请求。

    输入包含当前工作流步骤和按步骤编号保存的已完成结果；函数校验普通 binding
    或 ``foreach`` 选择器，把选中的 JSON 值或 Artifact 路径写入请求副本，并在
    批量模式下生成带稳定 item 身份和来源记录的子请求。函数不修改输入对象、不
    写文件，也不提交工具作业。返回单个已解析请求，或包含 ``_foreach_items`` 的
    批量请求；来源未完成、选择器无效、值为空、身份重复或展开数量越界时抛出
    ``CanonicalExecutionError``。
    """

    request = step.get("request")
    if not isinstance(request, Mapping):
        _fail("MiniChem step.request 必须是 object")
    resolved = deepcopy(dict(request))
    bindings = resolved.get("bindings", [])
    if not isinstance(bindings, list):
        _fail("MiniChem request.bindings 必须是数组")
    for index, binding in enumerate(bindings):
        if not isinstance(binding, Mapping):
            _fail(f"MiniChem request.bindings[{index}] 必须是 object")
        source_id = binding.get("source_step_ref")
        if source_id not in completed_steps:
            _fail(f"依赖 source step 尚未完成：{source_id}")
        source_field = binding.get("source_field")
        _validate_minichem_source_selector(
            source_field,
            f"MiniChem source step {source_id}.source_field",
        )
        value = _select_minichem_source_value(
            completed_steps[source_id],
            source_field,
            f"MiniChem source step {source_id}",
        )
        target_tokens = _minichem_target_tokens(binding.get("target_field"))
        if (
            not source_field.startswith("/")
            and isinstance(value, Mapping)
            and target_tokens[-1] == "source_path"
        ):
            path_value = value.get("path")
            if isinstance(path_value, str) and path_value:
                value = path_value
        if value is None:
            _fail(f"MiniChem source step {source_id} 的绑定值不能为空")
        _assert_json_value(value, f"MiniChem source step {source_id} binding")
        _set_minichem_target_value(
            resolved,
            target_tokens,
            value,
        )
    # bindings 是 ARCHE 内部的数据来源证明，不属于 MiniChem Action 请求正文。
    resolved.pop("bindings", None)
    foreach = resolved.pop("foreach", None)
    if foreach is not None:
        if not isinstance(foreach, Mapping):
            _fail("MiniChem request.foreach 必须是 object")
        source_id = foreach.get("source_step_ref")
        if source_id not in completed_steps:
            _fail(f"foreach source step 尚未完成：{source_id}")
        source_field = foreach.get("source_field")
        _validate_minichem_source_selector(
            source_field,
            f"MiniChem foreach source step {source_id}.source_field",
        )
        values = _select_minichem_source_value(
            completed_steps[source_id],
            source_field,
            f"MiniChem foreach source step {source_id}",
        )
        # typed Artifact selector 按合同恰好命中一个不可变 ArtifactRef；在
        # foreach 中把这个单个产物视为一项序列。普通 /result/... selector
        # 必须实际返回数组，不能把 scalar 或 object 猜测成数组。
        if (
            isinstance(values, Mapping)
            and (
                source_field.startswith("artifact.")
                or source_field.startswith("declared_artifact.name=")
            )
        ):
            values = [values]
        maximum = int(foreach.get("max_items", 32))
        if not isinstance(values, list) or not values:
            _fail(
                f"MiniChem foreach 来源必须是 1 到 {maximum} 项的非空数组"
            )
        only_present_items = foreach.get("only_present_items", False)
        if not only_present_items and len(values) > maximum:
            _fail(
                f"MiniChem foreach 来源必须是 1 到 {maximum} 项的非空数组"
            )
        instances: list[dict[str, Any]] = []
        seen_item_ids: set[str] = set()
        target_tokens = _minichem_target_tokens(foreach.get("target_field"))
        item_field = foreach.get("item_field")
        if item_field is not None:
            _validate_minichem_item_field_selector(
                item_field,
                f"MiniChem foreach source step {source_id}.item_field",
            )
        for index, item in enumerate(values):
            if item_field:
                if not item_field.startswith("/"):
                    value = _select_minichem_item_artifact(
                        item,
                        item_field,
                        f"MiniChem foreach item {index}",
                    )
                elif only_present_items:
                    value = _pointer_get_if_present(
                        item,
                        item_field,
                        f"MiniChem foreach item {index}",
                    )
                    if value is _MISSING_POINTER_VALUE or value is None:
                        continue
                else:
                    value = _pointer_get(
                        item,
                        item_field,
                        f"MiniChem foreach item {index}",
                    )
            else:
                value = item
            if (
                isinstance(value, Mapping)
                and isinstance(value.get("artifact_id"), str)
                and not isinstance(value.get("path"), str)
            ):
                matching_artifacts = [
                    artifact
                    for artifact in _collect_minichem_artifact_candidates(item)
                    if artifact.get("artifact_id") == value["artifact_id"]
                    and isinstance(artifact.get("path"), str)
                ]
                if len(matching_artifacts) == 1:
                    value = matching_artifacts[0]
            item_id_value = (
                _pointer_get(
                    item,
                    foreach["item_id_field"],
                    f"MiniChem foreach item {index} id",
                )
                if foreach.get("item_id_field")
                else f"item_{index + 1:04d}"
            )
            if not isinstance(item_id_value, (str, int)) or isinstance(
                item_id_value, bool
            ):
                _fail("MiniChem foreach item_id 必须是字符串或整数")
            item_id = str(item_id_value)[:100]
            if not item_id:
                _fail(f"MiniChem foreach item {index} 的 item_id 不能为空")
            if item_id in seen_item_ids:
                _fail(f"MiniChem foreach item_id 重复：{item_id}")
            seen_item_ids.add(item_id)
            artifact_candidates = _collect_minichem_artifact_candidates(value)
            if isinstance(value, Mapping) and isinstance(value.get("artifact_id"), str):
                artifact_candidates = [value]
            source_artifact_ids: list[str] = []
            for artifact in artifact_candidates:
                artifact_id = artifact.get("artifact_id")
                if (
                    isinstance(artifact_id, str)
                    and artifact_id
                    and artifact_id not in source_artifact_ids
                ):
                    source_artifact_ids.append(artifact_id)
            value = _minichem_foreach_target_value(
                value,
                target_tokens,
            )
            if value is None:
                _fail(f"MiniChem foreach item {index} 的目标值不能为空")
            _assert_json_value(value, f"MiniChem foreach item {index}")
            instance_request = deepcopy(resolved)
            _set_minichem_target_value(
                instance_request,
                target_tokens,
                value,
            )
            provenance = _minichem_foreach_item_provenance(
                item,
                item_id,
                index,
                str(source_id),
            )
            # 只记录当前 item 实际选中的 ArtifactRef，避免把同一结果中的其他
            # 候选产物误当成 native child 的父输入。
            provenance["source_artifact_ids"] = source_artifact_ids
            upstream_lineage = (
                item.get("lineage")
                if isinstance(item, Mapping)
                and isinstance(item.get("lineage"), Mapping)
                else {}
            )
            identity: dict[str, Any] = {}
            for identity_field in (
                "pair_id",
                "candidate_id",
                "group_label",
                "stereochemical_group",
                "direction",
                "reaction_orientation",
            ):
                identity_value = None
                for identity_source in (item, upstream_lineage, provenance):
                    if isinstance(identity_source, Mapping):
                        candidate = identity_source.get(identity_field)
                        if isinstance(candidate, (str, int, float, bool)) and not isinstance(
                            candidate, bool
                        ):
                            identity_value = str(candidate)[:100]
                            break
                if identity_value:
                    identity[identity_field] = identity_value
                    provenance.setdefault(identity_field, identity_value)
            origin_source_step_ref = upstream_lineage.get(
                "origin_source_step_ref",
                upstream_lineage.get("source_step_ref", str(source_id)),
            )
            lineage = {
                "source_step_ref": str(source_id),
                "origin_source_step_ref": str(origin_source_step_ref),
                "source_index": index,
                "source_artifact_ids": source_artifact_ids,
                **identity,
            }
            instances.append(
                {
                    "item_id": item_id,
                    "request": instance_request,
                    "lineage": lineage,
                    "provenance": provenance,
                }
            )
        if not instances:
            _fail("MiniChem foreach 过滤后没有可提交项目")
        if len(instances) > maximum:
            _fail(
                f"MiniChem foreach 可展开项目不能超过 {maximum} 项"
            )
        return {"_foreach_items": instances}
    return resolved


def collect_bound_artifact_ids(
    step: Mapping[str, Any],
    completed_steps: Mapping[str, Any],
) -> list[str]:
    """只收集当前绑定实际选择出的 ArtifactRef 身份，供父产物 lineage 使用。"""

    request = step.get("request") if isinstance(step, Mapping) else None
    bindings = request.get("bindings", []) if isinstance(request, Mapping) else []
    if not isinstance(bindings, list):
        _fail("MiniChem step.request.bindings 必须是数组")
    artifact_ids: list[str] = []
    seen: set[str] = set()
    for index, binding in enumerate(bindings):
        checked = _validate_minichem_binding(
            binding,
            f"step.request.bindings[{index}]",
            allow_short_refs=True,
        )
        source_id = checked["source_step_ref"]
        if source_id not in completed_steps:
            _fail(f"依赖 source step 尚未完成：{source_id}")
        selected = _select_minichem_source_value(
            completed_steps[source_id],
            checked["source_field"],
            f"MiniChem source step {source_id}",
        )
        candidates = _collect_minichem_artifact_candidates(selected)
        if isinstance(selected, Mapping) and isinstance(selected.get("artifact_id"), str):
            candidates = [selected]
        if checked["source_field"] == "job.stdout" and isinstance(selected, str):
            candidates = [
                item
                for item in _normalized_minichem_artifact_candidates(completed_steps[source_id])
                if item.get("path") == selected
            ]
        if checked["source_field"].startswith("declared_artifact.name="):
            declaration_path = selected.get("path") if isinstance(selected, Mapping) else None
            candidates = [
                item
                for item in _normalized_minichem_artifact_candidates(completed_steps[source_id])
                if item.get("path") == declaration_path
            ]
        for artifact in candidates:
            artifact_id = artifact.get("artifact_id")
            if isinstance(artifact_id, str) and artifact_id and artifact_id not in seen:
                seen.add(artifact_id)
                artifact_ids.append(artifact_id)
    return artifact_ids


def _minichem_execution_projection(workflow: Mapping[str, Any]) -> dict[str, Any]:
    """把 MiniChem 工作流压缩为与步骤身份、文字无关的执行投影。

    输入是正式 MiniChem 工作流；动作只复制 Action/Backend（或 execution_layer）、
    规范 request 和依赖顺序。返回值用于判断修订是否真的改变了执行行为，失败时
    明确拒绝旧 ToolSpec 字段，而不是尝试猜测其含义。
    """

    if not _is_minichem_workflow(workflow):
        _fail("执行差异只支持 MiniChem request 工作流；旧 ToolSpec 路径已停用")
    steps = workflow.get("steps")
    if not isinstance(steps, list):
        _fail("MiniChem workflow.steps 必须是数组")
    step_map = {
        step["step_id"]: f"p{index}"
        for index, step in enumerate(steps, start=1)
    }
    projected_steps: list[dict[str, Any]] = []
    for index, step in enumerate(steps):
        request = deepcopy(dict(step["request"]))
        bindings = request.get("bindings", [])
        if not isinstance(bindings, list):
            _fail(f"MiniChem steps[{index}].request.bindings 必须是数组")
        for binding in bindings:
            source = binding.get("source_step_ref")
            if source not in step_map:
                _fail(f"effective diff 遇到未知 MiniChem source_step_ref：{source}")
            binding["source_step_ref"] = step_map[source]
        foreach = request.get("foreach")
        if isinstance(foreach, MutableMapping):
            source = foreach.get("source_step_ref")
            if source not in step_map:
                _fail(f"effective diff 遇到未知 MiniChem foreach source_step_ref：{source}")
            foreach["source_step_ref"] = step_map[source]
        depends_on = []
        for dependency in step.get("depends_on", []):
            if dependency not in step_map:
                _fail(f"effective diff 遇到未知 MiniChem depends_on：{dependency}")
            depends_on.append(step_map[dependency])
        layer = step.get("execution_layer", "predefined_actions")
        identity: dict[str, Any] = {"execution_layer": layer}
        if layer == "predefined_actions":
            identity.update(
                {
                    "action_id": step.get("action_id"),
                    "backend_id": step.get("backend_id"),
                }
            )
        projected_steps.append(
            {
                **identity,
                "request": canonical_json_text(request),
                "depends_on": depends_on,
            }
        )
    return {"steps": projected_steps}


def effective_revision_diff(
    before_workflow: Mapping[str, Any], after_workflow: Mapping[str, Any]
) -> bool:
    """忽略正式身份、步骤名称和说明，只判断 MiniChem 调用是否改变。"""

    return canonical_json_text(_minichem_execution_projection(before_workflow)) != canonical_json_text(
        _minichem_execution_projection(after_workflow)
    )


def canonical_workflow_diff(
    before_workflow: Mapping[str, Any], after_workflow: Mapping[str, Any]
) -> dict[str, Any]:
    """把两个 MiniChem 工作流整理为顺序稳定、可审计的正式差异。"""

    before_steps = before_workflow.get("steps") if isinstance(before_workflow, Mapping) else None
    after_steps = after_workflow.get("steps") if isinstance(after_workflow, Mapping) else None
    if not _is_minichem_workflow(before_workflow) or not _is_minichem_workflow(after_workflow):
        _fail("canonical diff 只支持 MiniChem request 工作流；旧 ToolSpec 路径已停用")
    if not isinstance(before_steps, list) or not isinstance(after_steps, list):
        _fail("canonical diff 的 workflow.steps 必须是数组")
    before_by_id = {item["step_id"]: item for item in before_steps}
    after_by_id = {item["step_id"]: item for item in after_steps}
    deleted_ids = [
        item["step_id"] for item in before_steps if item["step_id"] not in after_by_id
    ]
    inserted_ids = [
        item["step_id"] for item in after_steps if item["step_id"] not in before_by_id
    ]
    operations: list[dict[str, Any]] = [
        {"op": "delete_step", "target_step_id": step_id}
        for step_id in deleted_ids
    ]
    mutable_fields = sorted(
        {
            field
            for item in (*before_steps, *after_steps)
            for field in item
            if field != "step_id"
        }
    )
    for item in after_steps:
        step_id = item["step_id"]
        if step_id not in before_by_id:
            continue
        previous = before_by_id[step_id]
        changes = {
            field: deepcopy(item.get(field))
            for field in mutable_fields
            if previous.get(field) != item.get(field)
        }
        if changes:
            operations.append(
                {
                    "op": "update_step",
                    "target_step_id": step_id,
                    "changes": changes,
                }
            )
    for index, item in enumerate(after_steps):
        if item["step_id"] not in inserted_ids:
            continue
        previous_id = after_steps[index - 1]["step_id"] if index else None
        operations.append(
            {
                "op": "insert_step_after",
                "after_step_id": previous_id,
                "step": deepcopy(item),
            }
        )

    working_order = [
        item["step_id"] for item in before_steps if item["step_id"] not in deleted_ids
    ]
    for index, item in enumerate(after_steps):
        step_id = item["step_id"]
        if step_id not in inserted_ids:
            continue
        previous_id = after_steps[index - 1]["step_id"] if index else None
        if previous_id is None:
            insertion_index = 0
        elif previous_id in working_order:
            insertion_index = working_order.index(previous_id) + 1
        else:
            _fail("canonical diff 的插入位置引用了已删除步骤")
        working_order.insert(insertion_index, step_id)
    target_order = [item["step_id"] for item in after_steps]
    for index, target_id in enumerate(target_order):
        if working_order[index] == target_id:
            continue
        working_order.remove(target_id)
        working_order.insert(index, target_id)
        operations.append(
            {
                "op": "move_step_after",
                "target_step_id": target_id,
                "after_step_id": target_order[index - 1] if index else None,
            }
        )
    if working_order != target_order:
        _fail("canonical diff 无法确定性重建 after step 顺序")
    return {"operations": operations}



def _apply_minichem_workflow_revision(
    before_workflow: Mapping[str, Any],
    wire_patch: Mapping[str, Any],
    catalog_snapshot: Mapping[str, Any],
    id_allocator: Callable[[str], str],
    *,
    gaussian_route_policy: GaussianRoutePolicy | None = None,
) -> dict[str, Any]:
    """在 MiniChem 短引用投影上应用修订，再恢复正式 Action 工作流。"""

    projection = _project_minichem_workflow_refs(before_workflow)
    before_steps = projection["workflow"]["steps"]
    ref_map = projection["ref_map"]["step_refs"]
    if not isinstance(wire_patch, Mapping) or set(wire_patch) != {"operations"}:
        _fail("MiniChem revision patch 只允许 operations 字段")
    operations = wire_patch.get("operations")
    if not isinstance(operations, list) or not operations:
        _fail("MiniChem revision operations 必须是非空数组")
    current = deepcopy(before_steps)
    original_request_by_ref = {
        item["step_ref"]: deepcopy(item.get("request", {}))
        for item in before_steps
        if isinstance(item, Mapping) and isinstance(item.get("step_ref"), str)
    }
    refs = {item["step_ref"] for item in current}
    inserted_refs: list[str] = []
    changed = False
    step_fields = (_MINICHEM_GENERATION_STEP_KEYS - {"step_ref"}) | {"execution_layer"}
    for index, operation in enumerate(operations):
        label = f"operations[{index}]"
        if not isinstance(operation, Mapping):
            _fail(f"{label} 必须是 object")
        op = operation.get("op")
        if op == "update_step":
            _require_exact_keys(operation, {"op", "target_step_ref", "changes"}, label)
            target = operation["target_step_ref"]
            if target not in refs:
                _fail(f"{label}.target_step_ref 未知")
            changes = operation["changes"]
            if not isinstance(changes, Mapping) or not changes:
                _fail(f"{label}.changes 必须是非空 object")
            if set(changes) - step_fields:
                _fail(f"{label}.changes 包含未声明字段")
            target_index = next(i for i, item in enumerate(current) if item["step_ref"] == target)
            previous = deepcopy(current[target_index])
            normalized_changes = deepcopy(dict(changes))
            if "request" in normalized_changes:
                # 模型看到的是固定公开 wire；提交前按原步骤恢复隐藏路径、
                # authored file 正文和正式身份，并移除跨层空占位。
                normalized_changes["request"] = _restore_revision_request_values(
                    normalized_changes["request"],
                    original_request_by_ref.get(target, previous.get("request", {})),
                )
            current[target_index].update(normalized_changes)
            if current[target_index] != previous:
                changed = True
        elif op == "insert_step_after":
            _require_exact_keys(
                operation,
                {"op", "new_step_ref", "after_step_ref", "step"},
                label,
            )
            expected_ref = f"n{len(inserted_refs) + 1}"
            if operation["new_step_ref"] != expected_ref or expected_ref in refs:
                _fail(f"{label}.new_step_ref 必须连续声明；期望 {expected_ref}")
            after = operation["after_step_ref"]
            if after is not None and after not in refs:
                _fail(f"{label}.after_step_ref 未知")
            step = operation["step"]
            if not isinstance(step, Mapping) or not _valid_minichem_step_keys(
                {"step_ref": expected_ref, **dict(step)}, generation=True
            ):
                _fail(f"{label}.step 字段不完整")
            item = deepcopy(dict(step))
            item["step_ref"] = expected_ref
            # 插入步骤沿用公开 Revision wire 时，路径、文件正文和运行身份都
            # 是空占位。优先用 after_step 的正式请求作为恢复模板；没有 after
            # 时再找同一执行层的已有步骤，确保复制 open step 后仍能通过真实
            # Native/Analysis 请求合同。
            template_request: Any = None
            if isinstance(after, str):
                template_request = original_request_by_ref.get(after)
            if template_request is None:
                requested_layer = item.get("execution_layer", "predefined_actions")
                requested_action = item.get("action_id")
                requested_backend = item.get("backend_id")
                for existing in current:
                    if existing.get("execution_layer", "predefined_actions") != requested_layer:
                        continue
                    if requested_layer == "predefined_actions" and (
                        existing.get("action_id") != requested_action
                        or existing.get("backend_id") != requested_backend
                    ):
                        continue
                    template_request = original_request_by_ref.get(existing.get("step_ref"))
                    if template_request is not None:
                        break
            if template_request is not None:
                item["request"] = _restore_revision_request_values(
                    item.get("request", {}),
                    template_request,
                )
            insert_at = len(current) if after is None else next(
                i for i, value in enumerate(current) if value["step_ref"] == after
            ) + 1
            current.insert(insert_at, item)
            refs.add(expected_ref)
            original_request_by_ref[expected_ref] = deepcopy(item.get("request", {}))
            inserted_refs.append(expected_ref)
            changed = True
        elif op == "delete_step":
            _require_exact_keys(operation, {"op", "target_step_ref"}, label)
            target = operation["target_step_ref"]
            if target not in refs:
                _fail(f"{label}.target_step_ref 未知")
            if any(
                target in item.get("depends_on", [])
                or any(
                    isinstance(binding, Mapping)
                    and binding.get("source_step_ref") == target
                    for binding in item.get("request", {}).get("bindings", [])
                )
                or (
                    isinstance(item.get("request", {}).get("foreach"), Mapping)
                    and item["request"]["foreach"].get("source_step_ref") == target
                )
                for item in current
                if item.get("step_ref") != target
            ):
                _fail(f"{label} 删除步骤后会留下悬空 MiniChem 绑定")
            current = [item for item in current if item["step_ref"] != target]
            refs.remove(target)
            changed = True
        elif op == "move_step_after":
            _require_exact_keys(
                operation,
                {"op", "target_step_ref", "after_step_ref"},
                label,
            )
            target = operation["target_step_ref"]
            after = operation["after_step_ref"]
            if target not in refs or (after is not None and after not in refs):
                _fail(f"{label} 引用了未知步骤")
            if target == after:
                _fail("move_step_after 不能把步骤移动到自身之后")
            moving = next(item for item in current if item["step_ref"] == target)
            current = [item for item in current if item["step_ref"] != target]
            insert_at = len(current) if after is None else next(
                i for i, value in enumerate(current) if value["step_ref"] == after
            ) + 1
            current.insert(insert_at, moving)
            changed = True
        else:
            _fail(f"{label}.op 未被允许：{op}")

    if not changed or current == before_steps:
        _fail("revision 没有 effective 有效差异")
    if not current:
        _fail("MiniChem 修订后的 steps 必须是非空数组")

    records = minichem_action_records(catalog_snapshot)
    for index, item in enumerate(current):
        if not _valid_minichem_step_keys(item, generation=True):
            _fail(f"MiniChem 修订 steps[{index}] 字段不完整")
        action_id = item.get("action_id")
        backend_id = item.get("backend_id")
        layer = item.get("execution_layer", "predefined_actions")
        if layer == "predefined_actions":
            if action_id not in records:
                _fail(f"MiniChem Action 未在冻结目录中：{action_id}")
            request = _validate_minichem_request(
                item.get("request"),
                records[action_id],
                backend_id,
                f"steps[{index}].request",
                allow_short_refs=True,
                catalog_snapshot=catalog_snapshot,
            )
        else:
            if action_id is not None or backend_id is not None:
                _fail("MiniChem open-execution 不得含 Action/Backend 身份")
            request = _validate_minichem_job_request(
                item.get("request"),
                layer,
                f"steps[{index}].request",
                allow_short_refs=True,
                catalog_snapshot=catalog_snapshot,
            )
        refs_before = {value["step_ref"] for value in current[:index]}
        depends = item.get("depends_on")
        bindings = request.get("bindings", [])
        if not isinstance(depends, list) or any(ref not in refs_before for ref in depends):
            _fail("MiniChem 修订依赖必须引用更早步骤")
        sources = [binding["source_step_ref"] for binding in bindings]
        foreach = request.get("foreach")
        if isinstance(foreach, Mapping) and foreach["source_step_ref"] not in sources:
            sources.append(foreach["source_step_ref"])
        if any(source not in depends for source in sources):
            _fail("MiniChem 修订 depends_on 必须覆盖 bindings/foreach 来源")

    formal_by_ref = dict(ref_map)
    formal_by_ref.update(
        {
            ref: _require_nonempty_text(id_allocator("step_id"), "正式 step_id")
            for ref in inserted_refs
        }
    )
    canonical_steps: list[dict[str, Any]] = []
    for index, item in enumerate(current):
        layer = item.get("execution_layer", "predefined_actions")
        if layer == "predefined_actions":
            request = _validate_minichem_request(
                item["request"],
                records[item["action_id"]],
                item["backend_id"],
                f"steps[{index}].request",
                allow_short_refs=True,
                catalog_snapshot=catalog_snapshot,
            )
        else:
            request = _validate_minichem_job_request(
                item["request"],
                layer,
                f"steps[{index}].request",
                allow_short_refs=True,
                catalog_snapshot=catalog_snapshot,
            )
        for binding in request["bindings"]:
            binding["source_step_ref"] = formal_by_ref[binding["source_step_ref"]]
        foreach = request.get("foreach")
        if isinstance(foreach, MutableMapping):
            foreach["source_step_ref"] = formal_by_ref[foreach["source_step_ref"]]
        canonical_step = {
                "step_id": formal_by_ref[item["step_ref"]],
                "step_name": _require_nonempty_text(item["step_name"], "step_name"),
                "description": _require_nonempty_text(item["description"], "description"),
                "request": request,
                "expected_outputs": (
                    _minichem_expected_outputs(
                        records[item["action_id"]],
                        pair_contract=_minichem_pair_request_contract(
                            catalog_snapshot,
                            item["action_id"],
                            item["backend_id"],
                        ),
                    )
                    if layer == "predefined_actions"
                    else {"execution_layer": layer}
                ),
                "depends_on": [formal_by_ref[ref] for ref in item["depends_on"]],
            }
        if layer == "predefined_actions":
            canonical_step["action_id"] = item["action_id"]
            canonical_step["backend_id"] = item["backend_id"]
        else:
            canonical_step["execution_layer"] = layer
        canonical_steps.append(canonical_step)
    after = {
        "workflow_name": before_workflow["workflow_name"],
        "goal": before_workflow["goal"],
        "strategy_name": before_workflow["strategy_name"],
        "steps": canonical_steps,
    }
    validate_canonical_workflow(
        after,
        catalog_snapshot,
        gaussian_route_policy=gaussian_route_policy,
    )
    return after


def apply_workflow_revision(
    before_workflow: Mapping[str, Any],
    wire_patch: Mapping[str, Any],
    catalog_snapshot: Mapping[str, Any],
    id_allocator: Callable[[str], str],
    *,
    task_root: str | Path | None = None,
    gaussian_route_policy: GaussianRoutePolicy | None = None,
) -> dict[str, Any]:
    """原子应用 MiniChem 工作流修订并返回新版本及规范差异。

    输入必须是 MiniChem 正式工作流和短引用补丁；旧 ToolSpec 修订路径不再执行。
    task_root 仅为兼容旧调用签名保留，MiniChem 请求由 MCP 管理其运行文件。
    """

    _ = task_root
    if not is_minichem_catalog(catalog_snapshot):
        _fail("工作流修订只接受 MiniChem Action Catalog；旧 ToolSpec 路径已停用")
    validate_canonical_workflow(
        before_workflow,
        catalog_snapshot,
        gaussian_route_policy=gaussian_route_policy,
    )
    after = _apply_minichem_workflow_revision(
        before_workflow,
        wire_patch,
        catalog_snapshot,
        id_allocator,
        gaussian_route_policy=gaussian_route_policy,
    )
    if not effective_revision_diff(before_workflow, after):
        _fail("revision 没有 effective 有效差异")
    normalized_patch = canonical_workflow_diff(before_workflow, after)
    if not normalized_patch["operations"]:
        _fail("revision canonical diff 为空")
    return {
        "after_workflow": after,
        "normalized_patch": normalized_patch,
    }
