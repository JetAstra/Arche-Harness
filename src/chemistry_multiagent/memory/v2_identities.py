"""Investigation V2 的显式 context binding 类型。

这些类型只描述 Module Invocation 选择的长期 owner，不保存实时运行状态。它们
替代含义不明确的 ``runtime_ref``；旧 RuntimeBinding 仍作为读取旧账本的兼容层。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias


def _id(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or "/" in value or "\\" in value:
        raise ValueError(f"{field} 必须是不透明标识")
    return value.strip()


@dataclass(frozen=True, slots=True)
class StatelessModule:
    """不持有跨调用上下文的模块（Retrieval/Query/Hypothesis/Selection）。"""

    kind: str = "StatelessModule"

    def to_dict(self) -> dict[str, str]:
        """返回公共 envelope 使用的无状态 owner。"""

        return {"kind": self.kind}


@dataclass(frozen=True, slots=True)
class CreateInvestigationContext:
    """首次 Investigation 创建一个跨返修复用的 Context。"""

    investigation_context_id: str
    hypothesis_batch_record_id: str
    kind: str = "CreateInvestigationContext"

    def __post_init__(self) -> None:
        object.__setattr__(self, "investigation_context_id", _id(self.investigation_context_id, "investigation_context_id"))
        object.__setattr__(self, "hypothesis_batch_record_id", _id(self.hypothesis_batch_record_id, "hypothesis_batch_record_id"))

    def to_dict(self) -> dict[str, str]:
        """返回首次 Investigation 的公共 owner 绑定。"""

        return {
            "kind": self.kind,
            "investigation_context_id": self.investigation_context_id,
            "hypothesis_batch_record_id": self.hypothesis_batch_record_id,
        }


@dataclass(frozen=True, slots=True)
class ReuseInvestigationContext:
    """返修复用已存在的 Investigation Context。"""

    investigation_context_id: str
    hypothesis_batch_record_id: str
    source_module_invocation_id: str
    kind: str = "ReuseInvestigationContext"

    def __post_init__(self) -> None:
        for name in ("investigation_context_id", "hypothesis_batch_record_id", "source_module_invocation_id"):
            object.__setattr__(self, name, _id(getattr(self, name), name))

    def to_dict(self) -> dict[str, str]:
        """返回返修 Investigation 复用的公共 owner 绑定。"""

        return {
            "kind": self.kind,
            "investigation_context_id": self.investigation_context_id,
            "hypothesis_batch_record_id": self.hypothesis_batch_record_id,
            "source_module_invocation_id": self.source_module_invocation_id,
        }


@dataclass(frozen=True, slots=True)
class CreateResearchTaskReviewSession:
    """首次 task review 创建一个跨返修复用的 Review Session。"""

    research_task_review_session_id: str
    kind: str = "CreateResearchTaskReviewSession"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "research_task_review_session_id",
            _id(
                self.research_task_review_session_id,
                "research_task_review_session_id",
            ),
        )

    def to_dict(self) -> dict[str, str]:
        """返回首次 task review 的 typed context_binding。"""

        return {
            "kind": self.kind,
            "research_task_review_session_id": self.research_task_review_session_id,
        }


@dataclass(frozen=True, slots=True)
class ReuseResearchTaskReviewSession:
    """后续 task review 复用同一 Review Session。"""

    research_task_review_session_id: str
    source_module_invocation_id: str
    kind: str = "ReuseResearchTaskReviewSession"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "research_task_review_session_id",
            _id(
                self.research_task_review_session_id,
                "research_task_review_session_id",
            ),
        )
        object.__setattr__(
            self,
            "source_module_invocation_id",
            _id(self.source_module_invocation_id, "source_module_invocation_id"),
        )

    def to_dict(self) -> dict[str, str]:
        """返回后续 task review 的 typed context_binding。"""

        return {
            "kind": self.kind,
            "research_task_review_session_id": self.research_task_review_session_id,
            "source_module_invocation_id": self.source_module_invocation_id,
        }


ContextBinding: TypeAlias = (
    StatelessModule
    | CreateInvestigationContext
    | ReuseInvestigationContext
    | CreateResearchTaskReviewSession
    | ReuseResearchTaskReviewSession
)


__all__ = [
    "ContextBinding",
    "CreateResearchTaskReviewSession",
    "CreateInvestigationContext",
    "ReuseResearchTaskReviewSession",
    "ReuseInvestigationContext",
    "StatelessModule",
]
