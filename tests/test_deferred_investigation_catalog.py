"""验证 research_task 候选阶段与 Investigation 目录准入之间的延迟边界。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from chemistry_multiagent.controllers import chemistry_multiagent_controller as controller_module
from chemistry_multiagent.controllers.chemistry_multiagent_controller import (
    ResearchTaskApplication,
)
from chemistry_multiagent.controllers.investigation_v2_adapters import (
    HypothesisModuleAdapter,
)
from chemistry_multiagent.controllers.investigation_v2_module_controller import (
    ModuleCompleted,
    ModuleInvocationRequest,
)
from chemistry_multiagent.controllers.investigation_v2_native_adapters import (
    build_native_investigation_module_adapter,
)
from chemistry_multiagent.investigation.context_backend import (
    InvestigationSessionTemporarilyUnavailable,
)
from chemistry_multiagent.investigation.context_store import InvestigationSessionIdentity
from chemistry_multiagent.memory import ResearchTaskWorkingMemory
from chemistry_multiagent.memory.v2_identities import StatelessModule
from chemistry_multiagent.tools.minichem_adapter import MiniChemMcpError
from chemistry_multiagent.tools.minichem_catalog_freeze import FrozenMiniChemCatalog


def _application() -> ResearchTaskApplication:
    """构造只测试公开路由的轻量 Application。"""

    application = object.__new__(ResearchTaskApplication)
    application.memory_enabled = True
    application._author_main_hypothesis_mode = False
    application.case_profile = None
    return application


def test_standard_entry_does_not_probe_endpoints_before_task_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """候选生成入口不能在任务前调用 9011/9012 endpoint 准入检查。"""

    application = _application()
    monkeypatch.setattr(
        application,
        "_create_research_task_from_entry_strategy",
        lambda **_kwargs: {
            "status": "open",
            "reason_code": "awaiting_researcher_confirmation",
        },
        raising=False,
    )
    monkeypatch.setattr(
        controller_module,
        "_validate_investigation_service_endpoints",
        lambda: pytest.fail("候选阶段不应检查 9011/9012 endpoint"),
    )

    result = application.create_standard_research_task("一个待确认的科学问题")

    assert result == {
        "status": "open",
        "reason_code": "awaiting_researcher_confirmation",
    }


def test_author_entry_does_not_probe_endpoints_before_task_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """作者假设直入在确认前也不能检查 9011/9012 endpoint。"""

    application = _application()
    monkeypatch.setattr(
        application,
        "_create_research_task_from_entry_strategy",
        lambda **_kwargs: {
            "status": "open",
            "reason_code": "awaiting_researcher_confirmation",
        },
        raising=False,
    )
    monkeypatch.setattr(
        controller_module,
        "_validate_investigation_service_endpoints",
        lambda: pytest.fail("作者候选阶段不应检查 9011/9012 endpoint"),
    )

    result = application.create_author_hypothesis_task(
        "一个作者假设待确认的科学问题",
        [
            {
                "hypothesis_ref": "author-h1",
                "source_ref": "author-source-1",
                "source_digest": "a" * 64,
                "hypothesis": {
                    "hypothesis_id": "author-h1",
                    "strategy_name": "stereochemical catalyst control",
                    "reasoning": "the catalyst controls the stereochemical step",
                },
                "provenance": {"kind": "researcher"},
            }
        ],
    )

    assert result["reason_code"] == "awaiting_researcher_confirmation"


def test_standard_hypothesis_generation_uses_local_catalog_without_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """标准候选生成可以使用安装包能力描述，但不能连接 9011。"""

    class _HypothesisAgent:
        def run_hypothesis_operation(self, **kwargs: Any) -> dict[str, Any]:
            self.arguments = kwargs
            return {
                "hypothesis_result": {
                    "queries": ["local capability angle"],
                    "hypotheses_by_query": [
                        {
                            "query": "local capability angle",
                            "hypotheses": [
                                {
                                    "hypothesis_ref": "h1",
                                    "claim": "local capability only",
                                }
                            ],
                        }
                    ],
                    "ranked_strategies": [
                        {"hypothesis_ref": "h1", "claim": "local capability only"}
                    ],
                }
            }

    class _Memory:
        def read_scientific_records(
            self, _research_task_id: str, _record_ids: tuple[str, ...]
        ) -> tuple[Any, ...]:
            return ()

    monkeypatch.setattr(
        "chemistry_multiagent.tools.minichem_adapter.MiniChemMcpAdapter",
        lambda **_kwargs: pytest.fail("标准候选生成不应创建 9011 adapter"),
    )
    agent = _HypothesisAgent()
    adapter = HypothesisModuleAdapter(
        controller=SimpleNamespace(hypothesis_agent=agent),
        task_root=tmp_path,
        working_memory=_Memory(),
        task_input={
            "scientific_question": "验证候选阶段的本地能力描述",
            "supplementary_scientific_input": {},
        },
        policy={"entry_strategy": "standard_hypothesis_generation"},
        dependencies={},
        task_config={
            "num_queries": 1,
            "num_hypotheses_per_query": 1,
            "top_n_hypotheses": 1,
        },
    )
    request = ModuleInvocationRequest(
        research_task_id="research-task-local-catalog",
        module_invocation_id="hypothesis-local-catalog",
        module_kind="hypothesis-generation",
        input_scientific_record_ids=(),
        module_input_digest="digest-local-catalog",
        context_binding=StatelessModule(),
    )

    result = adapter.start(request)

    assert isinstance(result, ModuleCompleted)
    assert agent.arguments["chemical_capability_projection"]
    assert not (tmp_path / "internal" / "catalog").exists()


def _session_identity(session_id: str, task_id: str, context_id: str, input_id: str) -> InvestigationSessionIdentity:
    """构造当前字段的 Investigation Session owner。"""

    workspace = (
        "internal/state/investigation_runtime/sessions/"
        f"{session_id}/codex_sessions/investigations/{task_id}/{context_id}/{input_id}/workspace"
    )
    return InvestigationSessionIdentity(
        investigation_session_id=session_id,
        investigation_context_id=context_id,
        research_task_id=task_id,
        hypothesis_execution_input_record_id=input_id,
        workspace_identity=workspace,
        minichem_scope_id=f"scope-{session_id}",
        scope_contract_digest="b" * 64,
    )


def test_native_investigation_freezes_catalog_before_scope_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """首次 Investigation 必须先持久化目录，再创建 MiniChem scope。"""

    events: list[str] = []

    class _ScopeManager:
        def ensure_session(self, _session: InvestigationSessionIdentity) -> None:
            events.append("scope")

    class _CatalogAdapter:
        def __init__(self, **_kwargs: Any) -> None:
            events.append("adapter")

        def open(self) -> None:
            events.append("open")

        def close(self) -> None:
            events.append("close")

    def freeze(*, adapter: Any, task_root: Path) -> FrozenMiniChemCatalog:
        del adapter
        events.append("freeze")
        return FrozenMiniChemCatalog(
            catalog_hash="a" * 64,
            snapshot={"schema_version": 7},
            package_root=task_root / "internal" / "catalog" / ("a" * 64),
        )

    monkeypatch.setattr(
        "chemistry_multiagent.controllers.investigation_v2_native_adapters.MiniChemMcpAdapter",
        _CatalogAdapter,
    )
    monkeypatch.setattr(
        "chemistry_multiagent.controllers.investigation_v2_native_adapters.freeze_minichem_catalog",
        freeze,
    )
    dependencies: dict[str, Any] = {"native_scope_manager": _ScopeManager()}
    adapter = build_native_investigation_module_adapter(
        task_root=tmp_path / "bundle",
        working_memory=ResearchTaskWorkingMemory(tmp_path / "memory.sqlite3"),
        task_input={"scientific_question": "验证目录冻结顺序"},
        policy={"entry_strategy": "standard_hypothesis_generation"},
        dependencies=dependencies,
    )
    resolver = adapter.harness._backend._session_owner
    resolver.ensure_session(
        _session_identity(
            "session-catalog-order",
            "research-task-catalog-order",
            "context-catalog-order",
            "hypothesis-catalog-order",
        )
    )

    assert events.index("freeze") < events.index("scope")


def test_catalog_unavailable_keeps_scope_uncreated_and_is_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """目录服务暂不可用时，准备接缝不创建 scope 并返回可重试错误。"""

    events: list[str] = []

    class _ScopeManager:
        def ensure_session(self, _session: InvestigationSessionIdentity) -> None:
            events.append("scope")

    class _CatalogAdapter:
        def __init__(self, **_kwargs: Any) -> None:
            events.append("adapter")

        def open(self) -> None:
            events.append("open")

        def close(self) -> None:
            events.append("close")

    monkeypatch.setattr(
        "chemistry_multiagent.controllers.investigation_v2_native_adapters.MiniChemMcpAdapter",
        _CatalogAdapter,
    )
    monkeypatch.setattr(
        "chemistry_multiagent.controllers.investigation_v2_native_adapters.freeze_minichem_catalog",
        lambda **_kwargs: (_ for _ in ()).throw(MiniChemMcpError("catalog unavailable")),
    )
    dependencies: dict[str, Any] = {"native_scope_manager": _ScopeManager()}
    adapter = build_native_investigation_module_adapter(
        task_root=tmp_path / "bundle",
        working_memory=ResearchTaskWorkingMemory(tmp_path / "memory.sqlite3"),
        task_input={"scientific_question": "验证目录暂不可用"},
        policy={"entry_strategy": "standard_hypothesis_generation"},
        dependencies=dependencies,
    )
    resolver = adapter.harness._backend._session_owner

    with pytest.raises(InvestigationSessionTemporarilyUnavailable):
        resolver.ensure_session(
            _session_identity(
                "session-catalog-unavailable",
                "research-task-catalog-unavailable",
                "context-catalog-unavailable",
                "hypothesis-catalog-unavailable",
            )
        )

    assert events == ["adapter", "open", "close"]
