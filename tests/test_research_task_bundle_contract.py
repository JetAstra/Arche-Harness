"""验证 research_task 的 Bundle 路径、身份字段、阶段和服务配置合同。"""

from __future__ import annotations

import json
import inspect
from pathlib import Path

import pytest

from chemistry_multiagent.investigation.v2_run_bundle_projector import (
    InvestigationV2RunBundleProjector,
)
from chemistry_multiagent.memory import ResearchTaskWorkingMemory
from chemistry_multiagent.utils.run_bundle import (
    DEFAULT_STAGE_LAYOUT,
    INVESTIGATION_STAGE_LAYOUT,
    RESEARCH_TASK_STAGE_LAYOUT,
    RESEARCH_TASK_STAGE_NAMES,
    RunBundle,
    locate_legacy_run_bundle,
    locate_legacy_state_database,
    locate_research_task_bundle,
    locate_research_task_state_database,
)
from chemistry_multiagent.utils import run_bundle as run_bundle_module
from tests.fakes.run_bundle import _SchemaV10BundleFixtureWriter


def _read_json(path: Path) -> dict[str, object]:
    """读取 JSON 对象。"""

    return json.loads(path.read_text(encoding="utf-8"))


def test_new_research_task_uses_namespace_and_public_identity(tmp_path: Path) -> None:
    """新 research_task 只能创建 research_tasks，并使用 research_task_id。"""

    bundle = RunBundle.create_research_task(
        tmp_path,
        "research-1",
        scientific_question="统一路径",
    )

    assert bundle.root == tmp_path / "outputs" / "research_tasks" / "research-1"
    assert not (tmp_path / "outputs" / "runs").exists()
    assert bundle.stage_layout == RESEARCH_TASK_STAGE_LAYOUT
    assert bundle.stage_names == RESEARCH_TASK_STAGE_NAMES
    assert bundle.research_task_id == "research-1"

    for path in (bundle.run_path, bundle.status_path, bundle.summary_path):
        document = _read_json(path)
        assert document["research_task_id"] == "research-1"
        assert "run_id" not in document
    for stage in RESEARCH_TASK_STAGE_NAMES:
        document = _read_json(bundle.root / "stages" / f"{stage}.json")
        assert document["research_task_id"] == "research-1"
        assert "run_id" not in document
    index = _read_json(tmp_path / "outputs" / "research_tasks" / "index.json")
    assert "research_tasks" in index
    assert "runs" not in index
    assert index["research_tasks"][0]["research_task_id"] == "research-1"
    assert "run_id" not in index["research_tasks"][0]

    # 即使索引里残留一条带旧身份的记录，下一次正式写入也不能把 run_id
    # 继续传播到 research_tasks/index.json。
    index["research_tasks"].append(
        {"research_task_id": "old-entry", "run_id": "old-entry"}
    )
    (tmp_path / "outputs" / "research_tasks" / "index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )
    bundle.update_status("open")
    refreshed_index = _read_json(tmp_path / "outputs" / "research_tasks" / "index.json")
    assert "runs" not in refreshed_index
    assert all("run_id" not in entry for entry in refreshed_index["research_tasks"])
    with pytest.raises(ValueError, match="run_id"):
        bundle.update_status("open", display={"run_id": "legacy-id"})
    with pytest.raises(ValueError, match="research_task_id"):
        bundle.update_summary(display={"research_task_id": "other-task"})


def test_default_bundle_constructor_creates_research_task(tmp_path: Path) -> None:
    """未指定历史兼容布局时，公开构造器只能创建 research_task。"""

    bundle = RunBundle(tmp_path, research_task_id="default-research-task")

    assert bundle.root == (
        tmp_path / "outputs" / "research_tasks" / "default-research-task"
    )
    assert bundle.bundle_namespace_root == tmp_path / "outputs" / "research_tasks"
    assert not hasattr(bundle, "runs_root")
    assert not (tmp_path / "outputs" / "runs").exists()
    for stage in RESEARCH_TASK_STAGE_NAMES:
        document = _read_json(bundle.root / "stages" / f"{stage}.json")
        assert document["research_task_id"] == "default-research-task"
        assert "run_id" not in document


def test_public_bundle_contract_has_only_research_task_identity(tmp_path: Path) -> None:
    """公开构造器、实例和默认阶段常量只暴露 research_task 语义。"""

    bundle = RunBundle(tmp_path, research_task_id="public-contract")

    assert DEFAULT_STAGE_LAYOUT == RESEARCH_TASK_STAGE_LAYOUT
    assert tuple(run_bundle_module.STAGE_NAMES) == RESEARCH_TASK_STAGE_NAMES
    assert "run_id" not in inspect.signature(RunBundle).parameters
    assert "stage_layout" not in inspect.signature(RunBundle).parameters
    assert not hasattr(bundle, "run_id")
    assert not hasattr(bundle, "runs_root")
    assert not hasattr(bundle, "locate_state_database")
    assert not hasattr(RunBundle, "open_existing")
    assert not hasattr(run_bundle_module, "locate_run_bundle")
    assert not hasattr(run_bundle_module, "locate_state_database")
    assert not hasattr(run_bundle_module, "relative_bundle_path")
    with pytest.raises(TypeError, match="stage_layout"):
        RunBundle(  # type: ignore[call-arg]
            tmp_path,
            research_task_id="stage-layout-is-not-public",
            stage_layout=INVESTIGATION_STAGE_LAYOUT,
        )


def test_research_task_locator_is_explicit_and_legacy_locator_is_separate(
    tmp_path: Path,
) -> None:
    """新旧 Bundle 定位入口不能互相回退。"""

    research = RunBundle.create_research_task(tmp_path, "new-task")
    assert locate_research_task_bundle(tmp_path, "new-task").root == research.root

    legacy = _SchemaV10BundleFixtureWriter.create(
        tmp_path,
        "old-task",
        stage_layout=INVESTIGATION_STAGE_LAYOUT,
    )
    assert locate_legacy_run_bundle(tmp_path, "old-task").root == legacy.root

    # 旧 schema-v10 数据可能没有命名空间索引锁；显式只读 locator 仍应能够读取，
    # 但不会因此创建或补写 outputs/runs 的索引。
    (tmp_path / "outputs" / ".index.lock").unlink()
    assert locate_legacy_run_bundle(tmp_path, "old-task").root == legacy.root

    # 新定位器不能因为 research_tasks 缺失而读取历史 outputs/runs。
    assert not (tmp_path / "outputs" / "research_tasks" / "old-task").exists()
    with pytest.raises(ValueError, match="运行 Bundle 目录不存在"):
        locate_research_task_bundle(tmp_path, "old-task")
    assert not hasattr(RunBundle, "open_existing")

    legacy_state = legacy.root / "internal" / "state" / "memory.sqlite3"
    legacy_state.touch()
    legacy_document = _read_json(legacy.run_path)
    legacy_document["state_database"] = "internal/state/memory.sqlite3"
    legacy.run_path.write_text(
        json.dumps(legacy_document, ensure_ascii=False), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="历史 Bundle 只读"):
        legacy.register_state_database(legacy_state)
    assert locate_legacy_state_database(tmp_path, "old-task") == legacy_state.resolve()
    with pytest.raises(ValueError, match="运行 Bundle 目录不存在"):
        locate_research_task_state_database(tmp_path, "old-task")


def test_research_task_constructor_keyword_selects_formal_layout(tmp_path: Path) -> None:
    """显式 research_task_id 使调用方无需依赖旧 run_id 命名。"""

    bundle = RunBundle(tmp_path, research_task_id="keyword-task")
    assert bundle.root == tmp_path / "outputs" / "research_tasks" / "keyword-task"
    assert bundle.stage_names == RESEARCH_TASK_STAGE_NAMES


def test_research_task_projector_uses_formal_stage_names(tmp_path: Path) -> None:
    """新 Bundle 投影四个正式阶段，并不把旧阶段名称写入展示。"""

    service = ResearchTaskWorkingMemory(str(tmp_path / "memory.sqlite3"))
    facts = service.create_research_task(
        research_task_id="research-task-projector",
        contract_version="investigation_v2.task.v1",
        input={"scientific_question": "阶段名称"},
        policy={"mode": "test"},
        dependencies={"catalog": {"ref": "catalog-1"}},
    )
    bundle = RunBundle.create_research_task(tmp_path, "research-task-projector")

    projection = InvestigationV2RunBundleProjector(bundle, service).project(facts)

    assert projection.research_task_id == "research-task-projector"
    assert tuple(projection.stage_statuses) == RESEARCH_TASK_STAGE_NAMES
    summary = _read_json(bundle.summary_path)
    assert tuple(summary["stage_order"]) == RESEARCH_TASK_STAGE_NAMES
    assert "hypothesis" not in summary["stages"]
    assert "final_result" not in summary["stages"]


def test_public_shell_entrypoints_use_research_task_configuration_names() -> None:
    """公开启动脚本声明正式配置名，旧名只应留在服务适配边界。"""

    root = Path(__file__).resolve().parents[1]
    end_to_end = (root / "scripts" / "run_end_to_end.sh").read_text(encoding="utf-8")
    top5 = (root / "scripts" / "run_to_hypothesis_top5.sh").read_text(encoding="utf-8")
    assert "ARCHE_RUN_STORAGE_ROOT" in end_to_end
    assert "ARCHE_STAGE_HARNESS_WORKSPACE_ROOT" in end_to_end
    assert "ARCHE_STAGE_HARNESS_TRACE_ROOT" in end_to_end
    assert "MINICHEM_INVESTIGATION_ENDPOINT" in end_to_end
    assert "MINICHEM_SERVICE_WORKSPACE" in end_to_end
    assert "MINICHEM_SCOPE_REGISTRY_ROOT" in end_to_end
    assert "ARCHE_RUN_STORAGE_ROOT" in top5
    assert "ARCHE_STAGE_HARNESS_WORKSPACE_ROOT" in top5
    assert "ARCHE_STAGE_HARNESS_TRACE_ROOT" in top5


def test_public_runtime_docs_use_formal_storage_names() -> None:
    """当前运行文档不得把旧输出根或身份当作新入口。"""

    root = Path(__file__).resolve().parents[1]
    for relative_path in ("README.md", "docs/current/DELIVERY_AND_REPRODUCTION.md"):
        document = (root / relative_path).read_text(encoding="utf-8")
        assert "ARCHE_OUTPUT_ROOT" not in document
        assert "outputs/latest.json" not in document
        assert "outputs/research_tasks/<research_task_id>/" in document
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert "ARCHE_RUN_STORAGE_ROOT" in readme
    assert "outputs/research_tasks/<research_task_id>/" in readme
