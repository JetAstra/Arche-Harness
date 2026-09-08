"""research_task 公开拓扑与已停用 Legacy V1 边界测试。"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

from chemistry_multiagent.controllers.chemistry_multiagent_controller import (
    ResearchTaskApplication,
)
from chemistry_multiagent.controllers.investigation_v2_module_controller import (
    AUTHOR_MODULE_SEQUENCE,
    MODULE_SEQUENCE,
    ResearchWorkflowController,
)
from chemistry_multiagent.investigation.harness_runtime import InvestigationHarnessRequest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONTROLLERS = REPOSITORY_ROOT / "src/chemistry_multiagent/controllers"
AGENTS = REPOSITORY_ROOT / "src/chemistry_multiagent/agents"
HARNESS = REPOSITORY_ROOT / "src/chemistry_multiagent/harness"

_REMOVED_FILES = (
    AGENTS / "planner_agent.py",
    AGENTS / "execution_agent.py",
    AGENTS / "reflection_agent.py",
    CONTROLLERS / "closed_loop_workflow.py",
    CONTROLLERS / "workflow_state.py",
    HARNESS / "planner_control_context.py",
    HARNESS / "planner_mcp_server.py",
    HARNESS / "planner_tool_bridge.py",
)


def test_legacy_v1_implementation_files_are_not_production_modules() -> None:
    """停用的 Legacy V1 实现不再作为当前生产 import 入口。"""

    assert all(not path.exists() for path in _REMOVED_FILES)
    controller_source = (
        CONTROLLERS / "chemistry_multiagent_controller.py"
    ).read_text(encoding="utf-8")
    assert "from chemistry_multiagent.controllers.closed_loop_workflow" not in controller_source
    assert "from chemistry_multiagent.agents.planner_agent" not in controller_source
    assert "from chemistry_multiagent.agents.execution_agent" not in controller_source
    assert "_run_legacy_workflow_v1" not in controller_source
    assert "_resume_legacy_workflow_v1" not in controller_source


def test_v2_module_sequence_is_explicit_and_harness_is_one_module() -> None:
    """Controller 显式编排确认边界，Investigation 内部由 Harness 自己继续。"""

    assert MODULE_SEQUENCE == (
        "retrieval",
        "research-query-formulation",
        "hypothesis-generation",
        "hypothesis-selection",
        "hypothesis-confirmation",
        "hypothesis-investigation",
        "task-review",
    )
    assert AUTHOR_MODULE_SEQUENCE == (
        "hypothesis-import",
        "hypothesis-confirmation",
        "hypothesis-investigation",
        "task-review",
    )
    request_fields = {field.name for field in fields(InvestigationHarnessRequest)}
    assert request_fields == {
        "research_task_id",
        "module_invocation_id",
        "module_kind",
        "input_scientific_record_ids",
        "module_input_digest",
        "context_binding",
    }
    assert not request_fields & {
        "thread_id",
        "turn_id",
        "goal_id",
        "scope_id",
        "job_id",
        "submission_key",
        "task_id",
        "module_run_id",
        "runtime_ref",
        "input_digest",
    }


def test_public_application_exposes_research_task_actions_and_no_old_runner_slots() -> None:
    """当前应用入口使用收敛后的 research_task 动作，不暴露旧工作流动作。"""

    names = {
        name for name in dir(ResearchTaskApplication) if not name.startswith("_")
    }
    assert {
        "create_standard_research_task",
        "create_author_hypothesis_task",
        "create_top5_confirmation_task",
        "get_hypothesis_confirmation",
        "submit_hypothesis_confirmation_and_continue",
        "resume_research_task",
    } <= names
    assert not {
        "run_investigation_and_experiment_task",
        "run_end_to_end_investigation_task",
        "run_author_hypothesis_investigation_task",
        "run_hypothesis_top5_selection_task",
        "read_hypothesis_confirmation",
        "confirm_hypothesis_execution",
        "resume_investigation_task",
    } & names
    source = (CONTROLLERS / "chemistry_multiagent_controller.py").read_text(
        encoding="utf-8"
    )
    assert "self.planner_agent" not in source
    assert "self.reflection_agent" not in source
    assert "from chemistry_multiagent.agents.execution_agent" not in source


def test_v2_controller_source_does_not_route_module_progress_to_legacy_runner() -> None:
    """V2 模块 Controller 的生产源码只暴露当前 ModuleRunner seam。"""

    source = (
        CONTROLLERS / "investigation_v2_module_controller.py"
    ).read_text(encoding="utf-8")
    assert "ClosedLoopWorkflowRunner" not in source
    assert "WorkflowExecutor" not in source
    assert "waiting_execution" not in source
    assert "def run_research_task" in source
    assert "def resume_research_task" in source
    assert ResearchWorkflowController.__name__ == "ResearchWorkflowController"
