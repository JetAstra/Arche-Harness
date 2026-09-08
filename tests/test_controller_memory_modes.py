"""ResearchTaskApplication 的 Working Memory 模式与公开入口测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from chemistry_multiagent.controllers import chemistry_multiagent_controller as module
from chemistry_multiagent.controllers.chemistry_multiagent_controller import (
    ResearchTaskApplication,
)


def _application(*, memory_enabled: bool) -> ResearchTaskApplication:
    """构造只测试公开入口的轻量 Application。"""

    application = object.__new__(ResearchTaskApplication)
    application.memory_enabled = memory_enabled
    application._author_main_hypothesis_mode = False
    application.case_profile = None
    return application


def test_public_research_task_entry_requires_working_memory_and_does_not_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Memory Off 的公开入口在任何科学阶段前拒绝，不回退旧架构。"""

    application = _application(memory_enabled=False)
    monkeypatch.setattr(
        application,
        "_create_research_task_from_entry_strategy",
        lambda **_kwargs: pytest.fail("Memory Off 不得进入替代工作流"),
        raising=False,
    )

    with pytest.raises(RuntimeError, match="Memory Off"):
        application.create_standard_research_task("固定科学问题")


def test_public_research_task_entry_delegates_without_legacy_run_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """普通入口只传递正式科学参数，不访问旧 Planner/Execution/Reflection 对象。"""

    application = _application(memory_enabled=True)
    captured: dict[str, object] = {}

    def fake_v2_entry(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"status": "open", "research_task_id": "research-task-v2"}

    monkeypatch.setattr(
        application,
        "_create_research_task_from_entry_strategy",
        fake_v2_entry,
        raising=False,
    )
    result = application.create_standard_research_task(
        "固定科学问题",
        num_queries=2,
        top_n_hypotheses=5,
    )

    assert result == {"status": "open", "research_task_id": "research-task-v2"}
    assert captured["scientific_question"] == "固定科学问题"
    assert captured["num_queries"] == 2
    assert captured["top_n_hypotheses"] == 5
    assert "run_mode" not in captured
    assert not hasattr(application, "planner_agent")
    assert not hasattr(application, "reflection_agent")


def test_application_initialization_has_only_current_agent_slots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Application 初始化只装配当前 Retrieval/Hypothesis 与专家模块。"""

    class FakeRetrievalAgent:
        def __init__(self, **_kwargs: object) -> None:
            pass

    class FakeHypothesisAgent:
        def __init__(self, **_kwargs: object) -> None:
            pass

    monkeypatch.setattr(module, "RetrievalAgent", FakeRetrievalAgent)
    monkeypatch.setattr(module, "HypothesisAgent", FakeHypothesisAgent)
    monkeypatch.setenv("ARCHE_RUN_STORAGE_ROOT", str(tmp_path))
    application = ResearchTaskApplication(
        work_dir=str(tmp_path / "work"),
        memory_enabled=False,
        expert_model_enabled=False,
    )

    assert isinstance(application.retrieval_agent, FakeRetrievalAgent)
    assert isinstance(application.hypothesis_agent, FakeHypothesisAgent)
    assert not hasattr(application, "planner_agent")
    assert not hasattr(application, "reflection_agent")
