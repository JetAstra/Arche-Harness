"""research_task Working Memory 的显式活动入口。

schema-v10 兼容服务必须从 ``schema_v10_compat`` 显式导入；本模块只导出
research_task 的 Working Memory 与领域类型。
"""

from .v2_ledger import (
    ResearchTaskWorkingMemory,
    ModuleInvocationRequest,
    ModuleInvocationSnapshot,
    PreparedModuleInvocationOutput,
    ResearchTaskFacts,
    ResearchTaskSnapshot,
)

__all__ = [
    "ResearchTaskWorkingMemory",
    "ModuleInvocationRequest",
    "ModuleInvocationSnapshot",
    "PreparedModuleInvocationOutput",
    "ResearchTaskFacts",
    "ResearchTaskSnapshot",
]
