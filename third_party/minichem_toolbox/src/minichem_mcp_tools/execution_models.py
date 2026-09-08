"""Typed MCP contracts for software-native and programmable execution layers."""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Literal, Mapping

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from minichem_toolbox.models import (
    ResourceLimits,
    _InternalOperationRequest,
    _strip_public_resource_selection_schema,
)

from .execution_job_registry import EXECUTION_JOB_ID_PATTERN


_SOFTWARE_ID = re.compile(r"^[a-z][a-z0-9_]*$")
_JOB_ID = EXECUTION_JOB_ID_PATTERN
_OPERATION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_FINGERPRINT = re.compile(r"^[0-9a-f]{64}$")
_SUBMISSION_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$")
_ARTIFACT_ID = re.compile(r"^art_[0-9a-f]{32}$")
_RUNTIME_ID = re.compile(r"^[a-z][a-z0-9_]*$")
_JOB_CONTROL_FILES = {
    "analysis_contract.json",
    "artifact_manifest.json",
    "request.json",
    "status.json",
    "supervisor_spec.json",
    "collection.json",
    "stdout.log",
    "stderr.log",
    "researchchem_job.py",
    "runtime_output_registrations.json",
}


def _validate_relative_target(value: str) -> str:
    normalized = value.strip().replace("\\", "/")
    path = PurePosixPath(normalized)
    if not normalized or path.is_absolute() or ".." in path.parts:
        raise ValueError("staged target must be a non-empty relative path without '..'")
    if any(part in {"", "."} for part in path.parts):
        raise ValueError("staged target contains an invalid path component")
    if path.parts[0] in {".tmp", ".home"} or str(path) in _JOB_CONTROL_FILES:
        raise ValueError("staged target conflicts with an execution job control path")
    return str(path)


class SoftwareListRequest(BaseModel):
    """按条件筛选已注册或有文档的软件清单，不替调用方选择软件。"""

    model_config = ConfigDict(extra="forbid")

    query: str | None = Field(default=None, max_length=200)
    native_only: bool = False
    available_only: bool = False
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=50, ge=1, le=200)


class SoftwareInspectRequest(BaseModel):
    """请求一个 software id 的精确能力与调用指南。"""

    model_config = ConfigDict(extra="forbid")

    software_id: str

    @field_validator("software_id")
    @classmethod
    def validate_software_id(cls, value: str) -> str:
        normalized = value.strip().lower().replace("-", "_")
        if not _SOFTWARE_ID.fullmatch(normalized):
            raise ValueError("software_id must use lower_snake_case")
        return normalized


class DocumentationSearchRequest(SoftwareInspectRequest):
    """按主题或章节定位软件文档的有限内容。"""

    query: str | None = Field(default=None, min_length=2, max_length=200)
    topic: str | None = Field(default=None, min_length=2, max_length=100)
    section: str | None = Field(default=None, min_length=1, max_length=200)
    retrieval_mode: Literal["lexical", "hybrid"] = "hybrid"
    max_results: int = Field(default=10, ge=1, le=50)
    context_chars: int = Field(default=500, ge=100, le=4000)

    @model_validator(mode="after")
    def require_route_or_query(self) -> "DocumentationSearchRequest":
        if not any((self.query, self.topic, self.section)):
            raise ValueError("provide query, topic, or section")
        return self


class DocumentationReadRequest(SoftwareInspectRequest):
    """读取按主题和可选章节选择的精确软件文档。"""

    topic: str = Field(min_length=2, max_length=100)
    section: str | None = Field(default=None, min_length=1, max_length=200)
    max_chars: int = Field(default=12_000, ge=200, le=100_000)


class WorkspaceTextWriteRequest(_InternalOperationRequest):
    """Write an auditable text input deck or program under code/ or outputs/."""

    path: str = Field(min_length=1, max_length=1000)
    content: str = Field(max_length=10_000_000)
    overwrite: bool = False


class WorkspaceTextReadRequest(BaseModel):
    """读取 UTF-8 工作区文件，支持按字符偏移续读和 JSON Pointer 定位。"""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, max_length=1000)
    max_chars: int = Field(default=100_000, ge=1, le=2_000_000)
    offset: int = Field(default=0, ge=0)
    json_pointer: str | None = Field(default=None, max_length=100_000)


class StagedInput(BaseModel):
    """Explicit mapping from an existing workspace file to a native job filename."""

    model_config = ConfigDict(extra="forbid")

    source_path: str = Field(min_length=1, max_length=1000)
    target_path: str = Field(min_length=1, max_length=1000)

    @field_validator("target_path")
    @classmethod
    def validate_target_path(cls, value: str) -> str:
        return _validate_relative_target(value)


class AnalysisInputDeclaration(BaseModel):
    """One named programmable-job input staged under inputs/."""

    model_config = ConfigDict(extra="forbid")

    name: str
    source_path: str = Field(min_length=1, max_length=1000)
    semantic_type: str = Field(default="InputFile", min_length=1, max_length=200)
    target_path: str | None = Field(default=None, max_length=1000)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return _catalog_name(value, field_name="input name")

    @model_validator(mode="after")
    def set_target_path(self) -> "AnalysisInputDeclaration":
        suffix = PurePosixPath(self.source_path.replace("\\", "/")).suffix
        target = self.target_path or f"inputs/{self.name}{suffix}"
        normalized = _validate_relative_target(target)
        if PurePosixPath(normalized).parts[0] != "inputs":
            raise ValueError("analysis input target_path must be under inputs/")
        self.target_path = normalized
        return self


class AnalysisOutputDeclaration(BaseModel):
    """One named expected output and its mechanical validation contract."""

    model_config = ConfigDict(extra="forbid")

    name: str
    path: str = Field(min_length=1, max_length=1000)
    semantic_type: str = Field(min_length=1, max_length=200)
    media_type: str = Field(default="application/octet-stream", min_length=1, max_length=200)
    required: bool = True
    json_schema: dict | None = None
    parent_artifact_ids: list[str] = Field(default_factory=list, max_length=1000)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return _catalog_name(value, field_name="output name")

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        normalized = _validate_relative_target(value)
        if PurePosixPath(normalized).parts[0] not in {"outputs", "report", "logs"}:
            raise ValueError("analysis output path must be under outputs/, report/, or logs/")
        return normalized


class AnalysisInputInspectionRequest(BaseModel):
    """Inspect bounded structural metadata for declared programmable-job inputs."""

    model_config = ConfigDict(extra="forbid")

    inputs: list[AnalysisInputDeclaration] = Field(min_length=1, max_length=100)
    max_depth: int = Field(default=3, ge=1, le=6)
    max_fields: int = Field(default=40, ge=1, le=200)
    max_rows: int = Field(default=200, ge=1, le=2000)
    include_scalar_samples: bool = False


def _catalog_name(value: str, *, field_name: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if not _SOFTWARE_ID.fullmatch(normalized):
        raise ValueError(f"{field_name} must use lower_snake_case")
    return normalized


class NativeJobRequest(_InternalOperationRequest):
    """描述调用方提供的完整、精确的原生程序调用。"""

    software_id: str
    executable: str = Field(min_length=1, max_length=200)
    arguments: list[str] = Field(default_factory=list, max_length=500)
    staged_inputs: list[StagedInput] = Field(default_factory=list, max_length=1000)
    stdin_target: str | None = Field(default=None, max_length=1000)
    resource_limits: ResourceLimits = Field(default_factory=ResourceLimits)
    calculation_intent: Literal[
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
    ] | None = None
    label: str | None = Field(default=None, max_length=200)
    parent_job_id: str | None = None

    @field_validator("software_id")
    @classmethod
    def validate_software_id(cls, value: str) -> str:
        normalized = value.strip().lower().replace("-", "_")
        if not _SOFTWARE_ID.fullmatch(normalized):
            raise ValueError("software_id must use lower_snake_case")
        return normalized

    @field_validator("executable")
    @classmethod
    def validate_executable(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or "/" in normalized or "\\" in normalized or "\x00" in normalized:
            raise ValueError("executable must be the configured command name, not a path")
        return normalized

    @field_validator("arguments")
    @classmethod
    def validate_arguments(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        for value in values:
            if "\x00" in value or len(value) > 8192:
                raise ValueError("arguments cannot contain NUL bytes or exceed 8192 characters")
            normalized.append(value)
        return normalized

    @field_validator("stdin_target")
    @classmethod
    def validate_stdin_target(cls, value: str | None) -> str | None:
        return _validate_relative_target(value) if value is not None else None

    @field_validator("parent_job_id")
    @classmethod
    def validate_parent_job_id(cls, value: str | None) -> str | None:
        if value is not None and not _JOB_ID.fullmatch(value):
            raise ValueError("invalid parent_job_id")
        return value

    @model_validator(mode="after")
    def validate_staging(self) -> "NativeJobRequest":
        targets = [item.target_path for item in self.staged_inputs]
        if len(targets) != len(set(targets)):
            raise ValueError("staged_inputs contains duplicate target_path values")
        if self.stdin_target is not None and self.stdin_target not in set(targets):
            raise ValueError("stdin_target must name one of the explicitly staged target files")
        return self

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        """生成不允许原生作业选择资源的公开 Schema。"""

        schema = super().__get_pydantic_json_schema__(core_schema, handler)
        return _strip_public_resource_selection_schema(schema)


class AnalysisJobRequest(_InternalOperationRequest):
    """在一个明确配置的化学运行时中运行调用方提供的 Python 文件。"""

    runtime: str
    script_path: str = Field(min_length=1, max_length=1000)
    script_target: str = Field(default="code/agent_program.py", min_length=1, max_length=1000)
    arguments: list[str] = Field(default_factory=list, max_length=500)
    staged_inputs: list[StagedInput] = Field(default_factory=list, max_length=1000)
    required_modules: list[str] = Field(default_factory=list, max_length=500)
    required_module_versions: dict[str, str] = Field(default_factory=dict)
    required_symbols: dict[str, list[str]] = Field(default_factory=dict)
    inputs: list[AnalysisInputDeclaration] = Field(default_factory=list, max_length=1000)
    outputs: list[AnalysisOutputDeclaration] = Field(default_factory=list, max_length=1000)
    execution_policy: Literal["in_process_only"] = "in_process_only"
    external_execution: Literal["native_job_runner_only"] = "native_job_runner_only"
    resource_limits: ResourceLimits = Field(default_factory=ResourceLimits)
    label: str | None = Field(default=None, max_length=200)
    parent_job_id: str | None = None

    @field_validator("runtime")
    @classmethod
    def validate_runtime(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not _RUNTIME_ID.fullmatch(normalized):
            raise ValueError("runtime must use lower_snake_case")
        return normalized

    @field_validator("script_target")
    @classmethod
    def validate_script_target(cls, value: str) -> str:
        normalized = _validate_relative_target(value)
        if not normalized.endswith(".py"):
            raise ValueError("script_target must end in .py")
        return normalized

    @field_validator("arguments")
    @classmethod
    def validate_arguments(cls, values: list[str]) -> list[str]:
        for value in values:
            if "\x00" in value or len(value) > 8192:
                raise ValueError("arguments cannot contain NUL bytes or exceed 8192 characters")
        return values

    @field_validator("required_modules")
    @classmethod
    def validate_required_modules(cls, values: list[str]) -> list[str]:
        normalized = []
        for value in values:
            name = value.strip()
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", name):
                raise ValueError(f"invalid Python module name: {value!r}")
            normalized.append(name)
        if len(normalized) != len(set(normalized)):
            raise ValueError("required_modules contains duplicates")
        return normalized

    @field_validator("required_module_versions")
    @classmethod
    def validate_required_module_versions(cls, values: dict[str, str]) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for raw_name, raw_specifier in values.items():
            name = raw_name.strip()
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", name):
                raise ValueError(f"invalid Python module name: {raw_name!r}")
            specifier = raw_specifier.strip()
            try:
                SpecifierSet(specifier)
            except InvalidSpecifier as exc:
                raise ValueError(
                    f"invalid version specifier for {name!r}: {raw_specifier!r}"
                ) from exc
            normalized[name] = specifier
        return normalized

    @field_validator("required_symbols")
    @classmethod
    def validate_required_symbols(cls, values: dict[str, list[str]]) -> dict[str, list[str]]:
        normalized: dict[str, list[str]] = {}
        for raw_name, raw_symbols in values.items():
            name = raw_name.strip()
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", name):
                raise ValueError(f"invalid Python module name: {raw_name!r}")
            symbols = [item.strip() for item in raw_symbols]
            if any(
                not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", item)
                for item in symbols
            ):
                raise ValueError(f"invalid required symbol for module {name!r}")
            if len(symbols) != len(set(symbols)):
                raise ValueError(f"required_symbols for {name!r} contains duplicates")
            normalized[name] = symbols
        return normalized

    @field_validator("parent_job_id")
    @classmethod
    def validate_parent_job_id(cls, value: str | None) -> str | None:
        if value is not None and not _JOB_ID.fullmatch(value):
            raise ValueError("invalid parent_job_id")
        return value

    @model_validator(mode="after")
    def validate_staging(self) -> "AnalysisJobRequest":
        targets = [item.target_path for item in self.staged_inputs] + [
            str(item.target_path) for item in self.inputs
        ]
        context_module_target = str(
            PurePosixPath(self.script_target).parent / "researchchem_job.py"
        )
        if (
            self.script_target == context_module_target
            or context_module_target in targets
        ):
            raise ValueError(
                "script directory researchchem_job.py is reserved for job context"
            )
        if self.script_target in targets or len(targets) != len(set(targets)):
            raise ValueError("script_target and all staged input targets must be unique")
        input_names = [item.name for item in self.inputs]
        output_names = [item.name for item in self.outputs]
        output_paths = [item.path for item in self.outputs]
        if len(input_names) != len(set(input_names)):
            raise ValueError("analysis inputs contain duplicate names")
        if len(output_names) != len(set(output_names)):
            raise ValueError("analysis outputs contain duplicate names")
        if len(output_paths) != len(set(output_paths)):
            raise ValueError("analysis outputs contain duplicate paths")
        return self

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        """生成不允许 Python 分析作业选择资源的公开 Schema。"""

        schema = super().__get_pydantic_json_schema__(core_schema, handler)
        return _strip_public_resource_selection_schema(schema)


class AnalysisRuntimeListRequest(BaseModel):
    """列出可供可编程分析层选择的精确 Python 运行时。"""

    model_config = ConfigDict(extra="forbid")

    available_only: bool = True
    query: str | None = Field(default=None, max_length=200)
    runtime: str | None = None
    include_details: bool = False
    limit: int = Field(default=100, ge=1, le=500)

    @field_validator("runtime")
    @classmethod
    def validate_optional_runtime(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        if not _RUNTIME_ID.fullmatch(normalized):
            raise ValueError("runtime must use lower_snake_case")
        return normalized


class ExecutionResourceRequest(BaseModel):
    """读取服务资源约束和当前可用容量状态。"""

    model_config = ConfigDict(extra="forbid")


class ExecutionJobListRequest(BaseModel):
    """读取当前 capability scope 的全部 execution job；请求不接受筛选参数。"""

    model_config = ConfigDict(extra="forbid")


class JobStatusRequest(BaseModel):
    """Read persistent state and bounded log tails for one submitted job."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    tail_chars: int = Field(default=8000, ge=0, le=100_000)

    @field_validator("job_id")
    @classmethod
    def validate_job_id(cls, value: str) -> str:
        if not _JOB_ID.fullmatch(value):
            raise ValueError("invalid job_id")
        return value


class JobCollectRequest(JobStatusRequest):
    """Collect a completed job manifest and output hashes without interpreting science."""

    include_inputs: bool = False
    max_files: int = Field(default=1000, ge=1, le=10_000)
    submission_key: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
        description="Investigation scope 下本次 manifest 收集的稳定提交键。",
    )

    @field_validator("submission_key")
    @classmethod
    def validate_submission_key(cls, value: str | None) -> str | None:
        if value is not None and _SUBMISSION_KEY.fullmatch(value.strip()) is None:
            raise ValueError("invalid submission_key")
        return value.strip() if value is not None else None


class JobCancelRequest(BaseModel):
    """Request cancellation of one running native or programmable job."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    submission_key: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
        description="Investigation scope 下本次取消请求的稳定提交键。",
    )

    @field_validator("submission_key")
    @classmethod
    def validate_submission_key(cls, value: str | None) -> str | None:
        if value is not None and _SUBMISSION_KEY.fullmatch(value.strip()) is None:
            raise ValueError("invalid submission_key")
        return value.strip() if value is not None else None

    @field_validator("job_id")
    @classmethod
    def validate_job_id(cls, value: str) -> str:
        if not _JOB_ID.fullmatch(value):
            raise ValueError("invalid job_id")
        return value


class ExecutionJobReference(BaseModel):
    """供只读核对的 native/analysis job 身份，不接受路径或日志字段。

    调用方可以只提交 job_id；服务端会从当前 scope 的完整持久收据读取其余身份，
    但调用方显式提供的 operation_id 或指纹仍必须逐项匹配。
    """

    model_config = ConfigDict(extra="forbid")

    job_id: str
    operation_id: str | None = None
    operation_fingerprint: str | None = None

    @field_validator("job_id")
    @classmethod
    def validate_job_id(cls, value: str) -> str:
        if not _JOB_ID.fullmatch(value):
            raise ValueError("invalid job_id")
        return value

    @field_validator("operation_id")
    @classmethod
    def validate_operation_id(cls, value: str | None) -> str | None:
        if value is not None and not _OPERATION_ID.fullmatch(value):
            raise ValueError("invalid operation_id")
        return value

    @field_validator("operation_fingerprint")
    @classmethod
    def validate_operation_fingerprint(cls, value: str | None) -> str | None:
        if value is not None and not _FINGERPRINT.fullmatch(value):
            raise ValueError("invalid operation_fingerprint")
        return value


class ExecutionOperationReference(BaseModel):
    """供只读核对的 typed Action 幂等收据身份，可只提供 operation_id。"""

    model_config = ConfigDict(extra="forbid")

    operation_id: str
    fingerprint: str | None = None

    @field_validator("operation_id")
    @classmethod
    def validate_operation_id(cls, value: str) -> str:
        if not _OPERATION_ID.fullmatch(value):
            raise ValueError("invalid operation_id")
        return value

    @field_validator("fingerprint")
    @classmethod
    def validate_fingerprint(cls, value: str | None) -> str | None:
        if value is not None and not _FINGERPRINT.fullmatch(value):
            raise ValueError("invalid fingerprint")
        return value


class ExecutionArtifactReference(BaseModel):
    """供只读核对的 ArtifactRef；可只提供 id，完整字段会与 canonical manifest 比较。"""

    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    semantic_type: str | None = None
    media_type: str | None = None
    sha256: str | None = None
    path: str | None = None
    producer_action: str | None = None
    producer_backend: str | None = None
    parent_artifact_ids: list[str] | None = None

    @field_validator("artifact_id")
    @classmethod
    def validate_artifact_id(cls, value: str) -> str:
        if not _ARTIFACT_ID.fullmatch(value):
            raise ValueError("invalid artifact_id")
        return value

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str | None) -> str | None:
        if value is not None and not _FINGERPRINT.fullmatch(value):
            raise ValueError("invalid artifact sha256")
        return value


class ScientificArtifactTextReadRequest(BaseModel):
    """按 ArtifactRef 读取当前 scope 的有限 UTF-8 文本分页，不接受宿主路径。"""

    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=12000, ge=1, le=20000)
    expected_sha256: str | None = None

    @field_validator("artifact_id")
    @classmethod
    def validate_artifact_id(cls, value: str) -> str:
        if not _ARTIFACT_ID.fullmatch(value):
            raise ValueError("invalid artifact_id")
        return value

    @field_validator("expected_sha256")
    @classmethod
    def validate_expected_sha256(cls, value: str | None) -> str | None:
        if value is not None and not _FINGERPRINT.fullmatch(value):
            raise ValueError("invalid expected_sha256")
        return value


class VerifyExecutionReferencesRequest(BaseModel):
    """只读核对当前 capability scope 内的 job、operation 和 artifact 引用。

    三个列表也接受 InvestigationResult 直接保存的字符串引用；字符串只作为对应
    类型的 id，canonical operation/fingerprint 必须从当前 scope 的收据中读取。
    """

    model_config = ConfigDict(extra="forbid")

    jobs: list[ExecutionJobReference] = Field(default_factory=list, max_length=100)
    operations: list[ExecutionOperationReference] = Field(
        default_factory=list, max_length=100
    )
    artifacts: list[ExecutionArtifactReference] = Field(
        default_factory=list, max_length=100
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_string_references(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        normalized = dict(value)
        for field_name, id_field in (
            ("jobs", "job_id"),
            ("operations", "operation_id"),
            ("artifacts", "artifact_id"),
        ):
            entries = normalized.get(field_name)
            if isinstance(entries, list):
                normalized[field_name] = [
                    {id_field: item} if isinstance(item, str) else item
                    for item in entries
                ]
        return normalized


class ArtifactDeclarationRequest(_InternalOperationRequest):
    """Attach scientific semantics and lineage to an existing workspace file."""

    path: str = Field(min_length=1, max_length=1000)
    semantic_type: str = Field(min_length=1, max_length=200)
    media_type: str = Field(default="application/octet-stream", min_length=1, max_length=200)
    producer_layer: Literal["native_software", "programmable_analysis", "external_input"]
    producer_id: str = Field(min_length=1, max_length=200)
    parent_artifact_ids: list[str] = Field(default_factory=list, max_length=1000)


__all__ = [
    "AnalysisInputDeclaration",
    "AnalysisInputInspectionRequest",
    "AnalysisJobRequest",
    "AnalysisOutputDeclaration",
    "AnalysisRuntimeListRequest",
    "ExecutionResourceRequest",
    "ExecutionJobListRequest",
    "ExecutionArtifactReference",
    "ExecutionJobReference",
    "ExecutionOperationReference",
    "ArtifactDeclarationRequest",
    "DocumentationReadRequest",
    "DocumentationSearchRequest",
    "JobCancelRequest",
    "JobCollectRequest",
    "JobStatusRequest",
    "NativeJobRequest",
    "SoftwareInspectRequest",
    "SoftwareListRequest",
    "StagedInput",
    "WorkspaceTextReadRequest",
    "WorkspaceTextWriteRequest",
    "ScientificArtifactTextReadRequest",
    "VerifyExecutionReferencesRequest",
]
