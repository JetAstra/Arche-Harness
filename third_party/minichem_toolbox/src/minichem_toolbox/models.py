"""Typed contracts shared by actions, backends, transports, and traces."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_SUBMISSION_KEY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$")
_PUBLIC_RESOURCE_SELECTION_FIELDS = frozenset(
    {"resource_limits", "cpu_cores", "memory_mb", "gpu_count"}
)


def _strip_public_resource_selection_schema(value: Any) -> Any:
    """从模型可见 JSON Schema 删除服务管理的资源选择字段。"""

    if isinstance(value, dict):
        for key in list(value):
            if key in _PUBLIC_RESOURCE_SELECTION_FIELDS:
                del value[key]
                continue
            _strip_public_resource_selection_schema(value[key])
        definitions = value.get("$defs")
        if isinstance(definitions, dict):
            definitions.pop("ResourceLimits", None)
    elif isinstance(value, list):
        for item in value:
            _strip_public_resource_selection_schema(item)
    return value
ActionStatus = Literal[
    "success",
    "partial_success",
    "invalid_request",
    "unsupported",
    "unavailable",
    "failed",
    "timeout",
    "cancelled",
]
ProviderSelectionPolicy = Literal[
    "agent_backend_required",
    "agent_components_required",
    "agent_source_required",
    "fixed_source",
    "internal_deterministic",
]


class ResourceLimits(BaseModel):
    """描述请求中的资源字段；实际服务约束由服务端检查。"""

    model_config = ConfigDict(extra="forbid")

    memory_mb: int = Field(
        default=4096,
        ge=128,
        description="可选内存限制；服务端按当前约束检查请求。",
    )
    cpu_cores: int = Field(
        default=1,
        ge=1,
        description="可选 CPU 限制；服务端按当前约束检查请求。",
    )
    gpu_count: int = Field(
        default=0,
        ge=0,
        description="可选 GPU 数量限制；服务端按当前约束检查请求。",
    )

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        """生成不包含服务容量数字的公开资源字段 Schema，运行时默认值保持不变。"""

        schema = handler(core_schema)
        schema["description"] = "可选资源限制；服务端检查真实约束，不能据此推断服务容量。"
        properties = schema.get("properties")
        if isinstance(properties, dict):
            for field_name in ("memory_mb", "cpu_cores", "gpu_count"):
                field_schema = properties.get(field_name)
                if isinstance(field_schema, dict):
                    field_schema.pop("default", None)
        return schema

    @field_validator("memory_mb", "cpu_cores", "gpu_count", mode="before")
    @classmethod
    def replace_null_with_default(cls, value: Any, info) -> Any:
        """Treat model-emitted nulls as omitted optional resource fields."""

        if value is not None:
            return value
        return {
            "memory_mb": 4096,
            "cpu_cores": 1,
            "gpu_count": 0,
        }[info.field_name]


class _InternalOperationRequest(BaseModel):
    """保存公开提交键和 legacy 内部幂等身份。

    Investigation scope 的公开请求必须提供 ``submission_key``；服务端依据
    ``scope_id + submission_key`` 生成 operation 身份，调用方不能填写
    ``operation_id``。``operation_id`` 用于兼容旧的非 scope 调用，
    ``caller_request_fingerprint`` 用于在服务重启或调用方丢失本地收据时核对
    调用方看到的规范请求。这两个字段仍只属于 MCP 内部协议，不进入公开工具描述。
    """

    model_config = ConfigDict(extra="allow")

    operation_id: str | None = Field(default=None, exclude=True, repr=False)
    caller_request_fingerprint: str | None = Field(
        default=None,
        exclude=True,
        repr=False,
    )
    submission_key: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
        description=(
            "稳定的语义提交键。Investigation capability scope 下必填；同一 scope "
            "和提交键只允许绑定一个规范请求，重试必须复用该键。"
        ),
    )

    @field_validator("operation_id")
    @classmethod
    def validate_internal_operation_id(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(
            r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$", value
        ):
            raise ValueError("operation_id must be a path-safe ASCII identifier")
        return value

    @field_validator("caller_request_fingerprint")
    @classmethod
    def validate_caller_request_fingerprint(cls, value: str | None) -> str | None:
        """只接受小写十六进制 SHA-256，避免把模糊值写入恢复身份。"""

        if value is not None and not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError(
                "caller_request_fingerprint must be a lowercase SHA-256 digest"
            )
        return value

    @field_validator("submission_key")
    @classmethod
    def validate_submission_key(cls, value: str | None) -> str | None:
        """校验公开提交键，避免把路径片段或空白身份带入服务端收据。"""

        if value is None:
            return None
        normalized = value.strip()
        if _SUBMISSION_KEY_PATTERN.fullmatch(normalized) is None:
            raise ValueError(
                "submission_key must be a non-empty path-safe ASCII identity"
            )
        return normalized

    @model_validator(mode="after")
    def reject_unknown_internal_fields(self):
        extras = getattr(self, "__pydantic_extra__", None) or {}
        unknown = sorted(
            set(extras) - {"operation_id", "caller_request_fingerprint"}
        )
        if unknown:
            raise ValueError(f"unknown request field(s): {unknown}")
        return self

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        """生成不包含内部调用身份字段的公开请求 Schema。"""

        schema = handler(core_schema)
        description = str(schema.get("description") or "")
        if "内部" in description or "internal" in description.casefold():
            schema["description"] = "公共请求结构；服务端会校验调用身份与请求内容。"
        properties = schema.get("properties")
        if isinstance(properties, dict):
            properties.pop("operation_id", None)
            properties.pop("caller_request_fingerprint", None)
        required = schema.get("required")
        if isinstance(required, list):
            schema["required"] = [
                item
                for item in required
                if item not in {"operation_id", "caller_request_fingerprint"}
            ]
        schema["additionalProperties"] = False
        return schema


class ActionRequest(_InternalOperationRequest):
    """所有公开科学与数据 Action 共用的统一请求结构。"""

    backend_id: str | None = Field(
        default=None,
        description=(
            "Exact backend selected by the caller. Required for each Scientific "
            "Action; data actions use their fixed named data source."
        ),
    )
    component_backends: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Caller-selected component backends for composite actions, such as an "
            "optimizer plus an energy/gradient engine or a dynamics driver plus an "
            "electronic-structure engine."
        ),
    )
    source_id: str | None = Field(
        default=None,
        description="Exact caller-selected data source when the Action offers multiple sources.",
    )
    inputs: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Structured chemistry objects and values. Artifact inputs may be a full immutable "
            "ArtifactRef, compact {'artifact_id': 'art_...'}, or the exact artifact-id string."
        ),
    )
    method_spec: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Caller-selected theory, functional, basis, model, force field, charge "
            "model, or equivalent scientific method settings."
        ),
    )
    action_settings: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Parameters belonging only to this atomic action, such as constraints, "
            "convergence thresholds, thermodynamic state, or search space."
        ),
    )
    resource_limits: ResourceLimits = Field(
        default_factory=ResourceLimits,
        description="Optional resource limits checked by the service policy.",
    )

    @field_validator("resource_limits", mode="before")
    @classmethod
    def replace_null_resource_limits(cls, value: Any) -> Any:
        return {} if value is None else value

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        """生成只暴露科学选择字段的公开 Action 请求 Schema。"""

        schema = super().__get_pydantic_json_schema__(core_schema, handler)
        return _strip_public_resource_selection_schema(schema)

    @field_validator("backend_id")
    @classmethod
    def validate_backend_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("backend_id cannot be empty")
        if normalized == "auto":
            raise ValueError("backend_id='auto' is forbidden in the benchmark")
        if not _ID_PATTERN.fullmatch(normalized):
            raise ValueError("backend_id must use lower_snake_case")
        return normalized

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("source_id cannot be empty")
        if normalized == "auto":
            raise ValueError("source_id='auto' is forbidden in the benchmark")
        if not _ID_PATTERN.fullmatch(normalized):
            raise ValueError("source_id must use lower_snake_case")
        return normalized

    @field_validator("component_backends")
    @classmethod
    def validate_component_backends(cls, value: dict[str, str]) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for role, backend_id in value.items():
            role_value = role.strip()
            backend_value = backend_id.strip()
            if not _ID_PATTERN.fullmatch(role_value):
                raise ValueError(f"component backend role must use lower_snake_case: {role!r}")
            if backend_value == "auto":
                raise ValueError("component backend value 'auto' is forbidden in the benchmark")
            if not _ID_PATTERN.fullmatch(backend_value):
                raise ValueError(f"component backend id must use lower_snake_case: {backend_id!r}")
            normalized[role_value] = backend_value
        return normalized


class ArtifactRef(BaseModel):
    """Stable semantic reference passed between atomic actions."""

    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    semantic_type: str
    media_type: str
    sha256: str
    path: str
    producer_action: str
    producer_backend: str | None = None
    parent_artifact_ids: list[str] = Field(default_factory=list)


class ActionResult(BaseModel):
    """Canonical result envelope returned by every public action."""

    model_config = ConfigDict(extra="forbid")

    status: ActionStatus
    action: str
    action_version: str
    requested_backend: str | None
    backend: str | None
    backend_version: str | None = None
    selection_source: Literal[
        "agent",
        "agent_components",
        "agent_data_source",
        "task_constraint",
        "fixed_data_source",
        "internal_deterministic",
    ]
    result: Any = None
    input_artifacts: list[ArtifactRef] = Field(default_factory=list)
    output_artifacts: list[ArtifactRef] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)
    error: dict[str, Any] | None = None
    retryable: bool = False


@dataclass(frozen=True)
class ActionSpec:
    """One public action with one primary scientific result."""

    id: str
    category: str
    description: str
    primary_output: str
    backend_ids: tuple[str, ...]
    required_inputs: tuple[str, ...] = ()
    optional_inputs: tuple[str, ...] = ()
    input_description: str = ""
    version: str = "1.0.0"
    data_action: bool = False
    requires_network: bool = False
    selection_policy: ProviderSelectionPolicy = "agent_backend_required"
    aliases: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    capability_tags: tuple[str, ...] = ()
    scientific_entities: tuple[str, ...] = ()
    task_verbs: tuple[str, ...] = ()
    input_semantic_types: tuple[str, ...] = ()
    output_semantic_types: tuple[str, ...] = ()
    batch_safe: bool = False

    def __post_init__(self) -> None:
        id_terms = tuple(self.id.split("_"))
        if not self.keywords:
            object.__setattr__(self, "keywords", id_terms)
        if not self.capability_tags:
            object.__setattr__(self, "capability_tags", (self.category,))
        if not self.scientific_entities:
            object.__setattr__(self, "scientific_entities", (self.primary_output,))
        if not self.task_verbs:
            object.__setattr__(self, "task_verbs", id_terms[:1])
        if not self.input_semantic_types:
            object.__setattr__(self, "input_semantic_types", self.required_inputs)
        if not self.output_semantic_types:
            object.__setattr__(self, "output_semantic_types", (self.primary_output,))

    @property
    def execution_class(self) -> Literal["compute", "fast"]:
        return "fast" if self.data_action else "compute"

    def validate(self) -> None:
        if not _ID_PATTERN.fullmatch(self.id):
            raise ValueError(f"Invalid action id: {self.id!r}")
        if self.id.startswith("run_"):
            raise ValueError(f"Software runner cannot be a public action: {self.id}")
        if not self.category.strip() or not self.description.strip():
            raise ValueError(f"Action {self.id} requires category and description")
        if not self.primary_output.strip():
            raise ValueError(f"Action {self.id} requires one primary output")
        if not self.backend_ids:
            raise ValueError(f"Action {self.id} requires at least one execution provider")
        if self.data_action and self.selection_policy not in {"fixed_source", "agent_source_required"}:
            raise ValueError(
                f"Data Action {self.id} requires fixed_source or agent_source_required policy"
            )
        if not self.data_action and self.selection_policy in {"fixed_source", "agent_source_required"}:
            raise ValueError(f"Scientific Action {self.id} cannot use a data-source policy")
        if self.selection_policy in {"fixed_source", "internal_deterministic"} and len(self.backend_ids) != 1:
            raise ValueError(f"Action {self.id} with {self.selection_policy} requires one provider")
        if "auto" in self.backend_ids:
            raise ValueError(f"Action {self.id} cannot advertise auto backend")
        if len(self.backend_ids) != len(set(self.backend_ids)):
            raise ValueError(f"Action {self.id} contains duplicate backend ids")
        for name in (*self.required_inputs, *self.optional_inputs):
            if not _ID_PATTERN.fullmatch(name):
                raise ValueError(f"Action {self.id} has invalid input key {name!r}")
        if set(self.required_inputs) & set(self.optional_inputs):
            raise ValueError(f"Action {self.id} repeats required/optional inputs")

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["execution_class"] = self.execution_class
        for key in ("backend_ids", "required_inputs", "optional_inputs"):
            value[key] = list(value[key])
        return value


@dataclass(frozen=True)
class BackendSpec:
    """Objective capability/install metadata for one agent-selectable backend."""

    id: str
    display_name: str
    runtime: str
    capabilities: tuple[str, ...]
    description: str
    python_modules: tuple[str, ...] = ()
    executables: tuple[str, ...] = ()
    environment_variables: tuple[str, ...] = ()
    conda_packages: tuple[str, ...] = ()
    pip_packages: tuple[str, ...] = ()
    required_data_resources: tuple[str, ...] = ()
    license_class: str = "open_source"
    install_notes: str = ""
    resource_constraints: Mapping[str, Any] = field(default_factory=dict)
    method_schema: Mapping[str, str] = field(default_factory=dict)
    required_input_fields: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    required_method_fields: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    required_setting_fields: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    allowed_method_values: Mapping[str, Mapping[str, tuple[str, ...]]] = field(
        default_factory=dict
    )
    allowed_setting_values: Mapping[str, Mapping[str, tuple[str, ...]]] = field(
        default_factory=dict
    )
    required_component_roles: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    component_backend_options: Mapping[str, Mapping[str, tuple[str, ...]]] = field(default_factory=dict)
    supported_system_types: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    validation_levels: Mapping[str, str] = field(default_factory=dict)
    parameter_specs: Mapping[str, Mapping[str, Mapping[str, Any]]] = field(
        default_factory=dict
    )
    fixed_parameter_specs: Mapping[str, Mapping[str, Mapping[str, Any]]] = field(
        default_factory=dict
    )

    def validate(self) -> None:
        if not _ID_PATTERN.fullmatch(self.id) or self.id == "auto":
            raise ValueError(f"Invalid backend id: {self.id!r}")
        if not self.display_name.strip() or not self.runtime.strip():
            raise ValueError(f"Backend {self.id} requires display_name and runtime")
        if not self.capabilities:
            raise ValueError(f"Backend {self.id} must declare capabilities")
        if len(self.capabilities) != len(set(self.capabilities)):
            raise ValueError(f"Backend {self.id} repeats capabilities")
        maximum_cpu_cores = self.resource_constraints.get("maximum_cpu_cores")
        if maximum_cpu_cores is not None and (
            not isinstance(maximum_cpu_cores, int) or maximum_cpu_cores < 1
        ):
            raise ValueError(
                f"Backend {self.id} resource_constraints.maximum_cpu_cores must be "
                "a positive integer"
            )
        maximum_walltime_seconds = self.resource_constraints.get(
            "maximum_walltime_seconds"
        )
        if maximum_walltime_seconds is not None and (
            not isinstance(maximum_walltime_seconds, int)
            or maximum_walltime_seconds < 1
        ):
            raise ValueError(
                f"Backend {self.id} resource_constraints.maximum_walltime_seconds "
                "must be a positive integer"
            )
        minimum_gpu_count = self.resource_constraints.get("minimum_gpu_count")
        if minimum_gpu_count is not None and (
            type(minimum_gpu_count) is not int or minimum_gpu_count < 0
        ):
            raise ValueError(
                f"Backend {self.id} resource_constraints.minimum_gpu_count must be "
                "a non-negative integer"
            )
        for mapping in (
            self.required_input_fields,
            self.required_method_fields,
            self.required_setting_fields,
            self.required_component_roles,
            self.supported_system_types,
            self.validation_levels,
        ):
            for action_id, fields in mapping.items():
                if action_id not in self.capabilities:
                    raise ValueError(
                        f"Backend {self.id} declares fields for unsupported {action_id}"
                    )
                if isinstance(fields, tuple) and len(fields) != len(set(fields)):
                    raise ValueError(
                        f"Backend {self.id}/{action_id} repeats required fields"
                    )
        for mapping_name, mapping in (
            ("method_spec", self.allowed_method_values),
            ("action_settings", self.allowed_setting_values),
        ):
            for action_id, fields in mapping.items():
                if action_id not in self.capabilities:
                    raise ValueError(
                        f"Backend {self.id} declares {mapping_name} choices for unsupported "
                        f"{action_id}"
                    )
                for field_name, choices in fields.items():
                    if not _ID_PATTERN.fullmatch(field_name):
                        raise ValueError(
                            f"Backend {self.id}/{action_id} has invalid {mapping_name} field "
                            f"{field_name!r}"
                        )
                    normalized = [str(choice).casefold() for choice in choices]
                    if not choices or len(normalized) != len(set(normalized)):
                        raise ValueError(
                            f"Backend {self.id}/{action_id} requires unique non-empty choices "
                            f"for {mapping_name}.{field_name}"
                        )
        for action_id, roles in self.component_backend_options.items():
            if action_id not in self.capabilities:
                raise ValueError(
                    f"Backend {self.id} declares component options for unsupported {action_id}"
                )
            required_roles = set(self.required_component_roles.get(action_id, ()))
            if not required_roles.issubset(roles):
                raise ValueError(
                    f"Backend {self.id}/{action_id} is missing component options for "
                    f"roles {sorted(required_roles - set(roles))}"
                )
        for mapping_name, mapping in (
            ("parameter_specs", self.parameter_specs),
            ("fixed_parameter_specs", self.fixed_parameter_specs),
        ):
            for action_id, fields in mapping.items():
                if action_id not in self.capabilities:
                    raise ValueError(
                        f"Backend {self.id} declares {mapping_name} for unsupported "
                        f"{action_id}"
                    )
                for field_path, metadata in fields.items():
                    if "." not in field_path:
                        raise ValueError(
                            f"Backend {self.id}/{action_id} {mapping_name} key "
                            f"{field_path!r} must be a section-qualified field path"
                        )
                    section, field_name = field_path.split(".", 1)
                    if section not in {
                        "inputs", "method_spec", "action_settings", "resource_limits",
                        "backend_runtime",
                    } or not _ID_PATTERN.fullmatch(field_name):
                        raise ValueError(
                            f"Backend {self.id}/{action_id} has invalid parameter path "
                            f"{field_path!r}"
                        )
                    if not isinstance(metadata, Mapping) or not str(
                        metadata.get("description", "")
                    ).strip():
                        raise ValueError(
                            f"Backend {self.id}/{action_id} {field_path} requires a "
                            "non-empty description"
                        )

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in (
            "capabilities",
            "python_modules",
            "executables",
            "environment_variables",
            "conda_packages",
            "pip_packages",
            "required_data_resources",
        ):
            value[key] = list(value[key])
        value["method_schema"] = dict(self.method_schema)
        value["required_input_fields"] = {
            key: list(fields) for key, fields in self.required_input_fields.items()
        }
        value["required_method_fields"] = {
            key: list(fields) for key, fields in self.required_method_fields.items()
        }
        value["required_setting_fields"] = {
            key: list(fields) for key, fields in self.required_setting_fields.items()
        }
        value["allowed_method_values"] = {
            action_id: {field_name: list(choices) for field_name, choices in fields.items()}
            for action_id, fields in self.allowed_method_values.items()
        }
        value["allowed_setting_values"] = {
            action_id: {field_name: list(choices) for field_name, choices in fields.items()}
            for action_id, fields in self.allowed_setting_values.items()
        }
        value["required_component_roles"] = {
            key: list(fields) for key, fields in self.required_component_roles.items()
        }
        value["component_backend_options"] = {
            action_id: {role: list(options) for role, options in roles.items()}
            for action_id, roles in self.component_backend_options.items()
        }
        value["supported_system_types"] = {
            key: list(fields) for key, fields in self.supported_system_types.items()
        }
        value["validation_levels"] = dict(self.validation_levels)
        value["parameter_specs"] = {
            action_id: {
                field_path: dict(metadata)
                for field_path, metadata in fields.items()
            }
            for action_id, fields in self.parameter_specs.items()
        }
        value["fixed_parameter_specs"] = {
            action_id: {
                field_path: dict(metadata)
                for field_path, metadata in fields.items()
            }
            for action_id, fields in self.fixed_parameter_specs.items()
        }
        return value
