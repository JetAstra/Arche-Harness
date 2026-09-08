"""Typed MCP requests for progressive catalog discovery and Action execution."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from minichem_toolbox.models import (
    ActionRequest,
    ResourceLimits,
    _strip_public_resource_selection_schema,
    _InternalOperationRequest,
)


_CATALOG_ID = re.compile(r"^[a-z][a-z0-9_]*$")
_BATCH_ID = re.compile(r"^batch_[0-9a-f]{32}$")


def _catalog_id(value: str, *, field_name: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if not _CATALOG_ID.fullmatch(normalized):
        raise ValueError(f"{field_name} must use lower_snake_case")
    return normalized


class ActionDomainListRequest(BaseModel):
    """返回完整的紧凑 Action 域索引。"""

    model_config = ConfigDict(extra="forbid")

    include_action_ids: bool = Field(
        default=True,
        description=(
            "按域返回每个精确 action_id；仅在只需要数量时关闭。"
        ),
    )


class ActionSearchRequest(BaseModel):
    """搜索并按相关性排列完整 Action 目录。"""

    model_config = ConfigDict(extra="forbid")

    query: str | None = Field(default=None, max_length=300)
    category: str | None = None
    backend_id: str | None = None
    action_kind: Literal["all", "scientific", "data"] = "all"
    retrieval_mode: Literal["lexical", "hybrid"] = "hybrid"
    detail_level: Literal["summary", "full"] = Field(
        default="summary",
        description="使用 summary 进行发现；需要完整契约详情时使用 full。",
    )
    available_only: bool = False
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0, le=100_000)

    @field_validator("category", "backend_id")
    @classmethod
    def validate_optional_ids(cls, value: str | None, info) -> str | None:
        return _catalog_id(value, field_name=info.field_name) if value is not None else None


class ActionCategoryBrowseRequest(BaseModel):
    """返回指定 Action 分类中的全部紧凑摘要。"""

    model_config = ConfigDict(extra="forbid")

    category: str
    action_kind: Literal["all", "scientific", "data"] = "all"
    available_only: bool = False
    detail_level: Literal["summary", "full"] = "summary"

    @field_validator("category")
    @classmethod
    def validate_category(cls, value: str) -> str:
        return _catalog_id(value, field_name="category")


class ActionInspectRequest(BaseModel):
    """查看一个精确 Action，可选限定一个精确 Backend。"""

    model_config = ConfigDict(extra="forbid")

    action_id: str
    backend_id: str | None = None
    detail_level: Literal["summary", "contract", "full"] = Field(
        default="contract",
        description=(
            "summary 比较 Backend，contract 返回紧凑可执行请求契约，full 额外返回完整契约详情。"
        ),
    )

    @field_validator("action_id", "backend_id")
    @classmethod
    def validate_ids(cls, value: str | None, info) -> str | None:
        return _catalog_id(value, field_name=info.field_name) if value is not None else None


class BackendInspectRequest(BaseModel):
    """查看一个精确 Backend 及其完整 Action 能力列表。"""

    model_config = ConfigDict(extra="forbid")

    backend_id: str
    action_id: str | None = None

    @field_validator("backend_id", "action_id")
    @classmethod
    def validate_ids(cls, value: str | None, info) -> str | None:
        return _catalog_id(value, field_name=info.field_name) if value is not None else None


class ResourceSearchRequest(BaseModel):
    """将调用方提供的筛选条件应用于全部已注册科学资源。"""

    model_config = ConfigDict(extra="forbid")

    query: str | None = Field(default=None, max_length=300)
    backend_id: str | None = None
    available_only: bool = False
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0, le=100_000)

    @field_validator("backend_id")
    @classmethod
    def validate_backend_id(cls, value: str | None) -> str | None:
        return _catalog_id(value, field_name="backend_id") if value is not None else None


class ResourceInspectRequest(BaseModel):
    """解析一个精确注册资源及其选择契约。"""

    model_config = ConfigDict(extra="forbid")

    resource_id: str

    @field_validator("resource_id")
    @classmethod
    def validate_resource_id(cls, value: str) -> str:
        return _catalog_id(value, field_name="resource_id")


class ProgressiveActionRequest(ActionRequest):
    """提供精确 Action id 和经过校验的 ActionRequest 字段。"""

    action_id: str = Field(
        description="填写 search_actions 或 inspect_action 返回的精确 action_id。"
    )

    @field_validator("action_id")
    @classmethod
    def validate_action_id(cls, value: str) -> str:
        return _catalog_id(value, field_name="action_id")

class ActionBatchItem(BaseModel):
    """batch-safe Action 提交中的一组独立输入。"""

    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(min_length=1, max_length=100)
    inputs: dict = Field(default_factory=dict)
    resource_limits: ResourceLimits = Field(default_factory=ResourceLimits)

    @field_validator("resource_limits", mode="before")
    @classmethod
    def replace_null_resource_limits(cls, value):
        return {} if value is None else value

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        """生成不允许批量 item 选择资源的公开 Schema。"""

        return _strip_public_resource_selection_schema(handler(core_schema))


class ActionBatchRequest(_InternalOperationRequest):
    """执行共享 batch-safe Action 和 Backend 的独立请求。

    服务端会校验请求身份；调用方只需提供 Action、Backend 和科学输入字段。
    """

    action_id: str
    backend_id: str
    component_backends: dict[str, str] = Field(default_factory=dict)
    method_spec: dict = Field(default_factory=dict)
    action_settings: dict = Field(default_factory=dict)
    items: list[ActionBatchItem] = Field(min_length=1, max_length=32)
    max_concurrency: int | None = Field(
        default=None,
        ge=1,
        le=32,
        description="可选并发上限；省略时由服务根据当前可用资源确定。",
    )

    @field_validator("action_id", "backend_id")
    @classmethod
    def validate_ids(cls, value: str, info) -> str:
        return _catalog_id(value, field_name=info.field_name)

class ExecutionEventWaitRequest(BaseModel):
    """Wait for persistent asynchronous Action-batch events."""

    model_config = ConfigDict(extra="forbid")

    batch_ids: list[str] = Field(min_length=1, max_length=64)
    after_sequences: dict[str, int] = Field(default_factory=dict)
    timeout_seconds: float = Field(default=0.0, ge=0.0, le=600.0)

    @field_validator("batch_ids")
    @classmethod
    def validate_batch_ids(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("batch_ids must be unique")
        if any(not _BATCH_ID.fullmatch(value) for value in values):
            raise ValueError("batch_ids must use batch_<32 lowercase hex> format")
        return values

    @field_validator("after_sequences")
    @classmethod
    def validate_after_sequences(cls, values: dict[str, int]) -> dict[str, int]:
        for batch_id, sequence in values.items():
            if not _BATCH_ID.fullmatch(batch_id):
                raise ValueError("after_sequences keys must be valid batch ids")
            if sequence < 0:
                raise ValueError("after_sequences values must be non-negative")
        return values


__all__ = [
    "ActionCategoryBrowseRequest",
    "ActionBatchItem",
    "ActionBatchRequest",
    "ActionDomainListRequest",
    "ExecutionEventWaitRequest",
    "ActionInspectRequest",
    "ActionSearchRequest",
    "BackendInspectRequest",
    "ProgressiveActionRequest",
    "ResourceInspectRequest",
    "ResourceSearchRequest",
]
