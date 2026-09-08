"""Case 3 advisory 同一 runtime 恢复入口的确定性合同测试。"""

from __future__ import annotations

import json
import sqlite3
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESUME_SCRIPT = (
    ROOT / "scripts/tasks/case3_resume_author_main_hypothesis_advisory_local.sh"
)


def make_resume_runtime(tmp_path: Path, task_id: str = "task-resume-1") -> Path:
    """构造只用于 dry-run 的最小已有 runtime，不启动服务或模型。"""

    runtime = tmp_path / "existing-runtime"
    bundle = runtime / "outputs" / "research_tasks" / task_id
    (bundle / "internal" / "audit").mkdir(parents=True)
    (bundle / "internal" / "state").mkdir(parents=True)
    task_config = {
        "research_task_id": task_id,
        "stage_layout": "research_task",
        "investigation_mode": "author_case3_advisory_main_hypothesis",
    }
    run_document = {
        "schema_version": 1,
        "research_task_id": task_id,
        "task_config": task_config,
    }
    for name in ("run.json", "status.json", "summary.json"):
        (bundle / name).write_text(
            json.dumps({**run_document, "status": "running"}),
            encoding="utf-8",
        )
    (bundle / "internal" / "audit" / "task.json").write_text(
        json.dumps({"research_task_id": task_id}),
        encoding="utf-8",
    )
    database = bundle / "internal" / "state" / "arche_memory.sqlite3"
    connection = sqlite3.connect(database)
    try:
        connection.execute(
            """
            CREATE TABLE research_tasks (
                research_task_id TEXT PRIMARY KEY,
                stage_architecture_version TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO research_tasks(research_task_id, stage_architecture_version) VALUES (?, ?)",
            (task_id, "investigation_v2"),
        )
        connection.commit()
    finally:
        connection.close()
    return runtime


def run_dry_resume(runtime: Path, task_id: str, config: Path) -> subprocess.CompletedProcess[str]:
    """运行零副作用恢复预览，并使用临时配置文件避免读取真实配置。"""

    return subprocess.run(
        [
            "bash",
            str(RESUME_SCRIPT),
            "--runtime-root",
            str(runtime),
            "--research-task-id",
            task_id,
            "--dry-run",
        ],
        cwd=ROOT,
        env={"RESEARCHCHEMBENCH_CONFIG": str(config)},
        check=False,
        capture_output=True,
        text=True,
    )


def test_resume_dry_run_reuses_task_identity_and_has_no_side_effects(tmp_path: Path) -> None:
    """dry-run 只显示原 Bundle 和恢复命令，不创建恢复目录。"""

    runtime = make_resume_runtime(tmp_path)
    config = tmp_path / "config.local.env"
    config.write_text("placeholder=1\n", encoding="utf-8")

    completed = run_dry_resume(runtime, "task-resume-1", config)

    assert completed.returncode == 0, completed.stderr
    assert "research_task_id=task-resume-1" in completed.stdout
    assert "controller_command=" in completed.stdout
    assert "resume-research-task task-resume-1" in completed.stdout
    assert "--authorize-legacy-waiting-resume" not in completed.stdout
    assert "--question" not in completed.stdout
    assert "--case-profile" not in completed.stdout
    assert "--scientific-input-json" not in completed.stdout
    assert "--num-queries" not in completed.stdout
    assert "resources=cpu:60,memory_mb:320000,gpu:1" in completed.stdout
    assert "side_effects=none" in completed.stdout
    assert not (runtime / "resume_attempts").exists()


def test_resume_dry_run_allows_resource_overrides_without_controller_budget_flags(
    tmp_path: Path,
) -> None:
    """资源覆盖只进入 MiniChem 预算预览，不改变恢复控制器参数。"""

    runtime = make_resume_runtime(tmp_path)
    config = tmp_path / "config.local.env"
    config.write_text("placeholder=1\n", encoding="utf-8")
    completed = subprocess.run(
        [
            "bash",
            str(RESUME_SCRIPT),
            "--runtime-root",
            str(runtime),
            "--research-task-id",
            "task-resume-1",
            "--cpu",
            "4",
            "--memory",
            "99",
            "--gpu",
            "0",
            "--dry-run",
        ],
        cwd=ROOT,
        env={"RESEARCHCHEMBENCH_CONFIG": str(config)},
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "resources=cpu:4,memory_mb:99,gpu:0" in completed.stdout
    assert "resume-research-task task-resume-1" in completed.stdout


def test_resume_dry_run_can_explicitly_authorize_legacy_waiting(
    tmp_path: Path,
) -> None:
    """显式旧 waiting 授权只作为同一 task 恢复参数传给 Controller。"""

    runtime = make_resume_runtime(tmp_path)
    config = tmp_path / "config.local.env"
    config.write_text("placeholder=1\n", encoding="utf-8")
    completed = subprocess.run(
        [
            "bash",
            str(RESUME_SCRIPT),
            "--runtime-root",
            str(runtime),
            "--research-task-id",
            "task-resume-1",
            "--authorize-legacy-waiting-resume",
            "--dry-run",
        ],
        cwd=ROOT,
        env={"RESEARCHCHEMBENCH_CONFIG": str(config)},
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert (
        "resume-research-task task-resume-1 --authorize-legacy-waiting-resume"
        in completed.stdout
    )
    assert "side_effects=none" in completed.stdout
    assert not (runtime / "resume_attempts").exists()


def test_resume_rejects_task_not_matching_the_only_run_bundle(tmp_path: Path) -> None:
    """错误 task 身份在任何恢复目录创建前被拒绝。"""

    runtime = make_resume_runtime(tmp_path)
    config = tmp_path / "config.local.env"
    config.write_text("placeholder=1\n", encoding="utf-8")

    completed = run_dry_resume(runtime, "task-other", config)

    assert completed.returncode != 0
    assert "不匹配" in completed.stderr
    assert not (runtime / "resume_attempts").exists()


def test_resume_script_has_only_direct_controller_resume_and_parses() -> None:
    """入口必须通过 Bash 语法检查，并保持恢复命令不带新任务字段。"""

    syntax = subprocess.run(
        ["bash", "-n", str(RESUME_SCRIPT)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert syntax.returncode == 0, syntax.stderr
    source = RESUME_SCRIPT.read_text(encoding="utf-8")
    assert "scripts/services/start_catalog_bootstrap_mcp.sh" in source
    assert "scripts/services/start_investigation_minichem.sh" in source
    assert "scripts/tasks/case3_check_mcp_readiness.py" in source
    assert '--runtime-root "$RESUME_ATTEMPT_ROOT"' in source
    assert '--scope-registry "$MINICHEM_SCOPE_REGISTRY_ROOT"' in source
    assert '"$PYTHON" -m chemistry_multiagent.controllers.chemistry_multiagent_controller' in source
    assert 'CONTROLLER_ARGS=(resume-research-task "$RESEARCH_TASK_ID")' in source
    assert 'CONTROLLER_ARGS+=(--authorize-legacy-waiting-resume)' in source
    assert "run_end_to_end.sh" not in source
    assert "--question" not in source
    assert "--case-profile" not in source
    assert "--scientific-input-json" not in source
    assert "--num-queries" not in source


def test_resume_script_reads_only_formal_research_task_bundle() -> None:
    """恢复 wrapper 只读取 research_tasks 与 canonical 配置。"""

    source = RESUME_SCRIPT.read_text(encoding="utf-8")
    for setting in (
        "outputs/research_tasks",
        "research_task_id",
        "export ARCHE_RUN_STORAGE_ROOT",
        "export ARCHE_STAGE_HARNESS_WORKSPACE_ROOT",
        "export ARCHE_STAGE_HARNESS_TRACE_ROOT",
        "export MINICHEM_SCOPE_REGISTRY_ROOT",
        "export MINICHEM_SERVICE_WORKSPACE",
        "export MINICHEM_INVESTIGATION_ENDPOINT",
    ):
        assert setting in source
    for legacy_name in (
        "ARCHE_OUTPUT_ROOT",
        "ARCHE_HARNESS_WORKSPACE_ROOT",
        "ARCHE_HARNESS_TRACE_DIR",
        "MINICHEM_SCOPE_REGISTRY=",
        "MINICHEM_MCP_WORKSPACE",
        "MINICHEM_MCP_ENDPOINT",
        "outputs/runs",
        "runtime_id",
    ):
        assert legacy_name not in source
