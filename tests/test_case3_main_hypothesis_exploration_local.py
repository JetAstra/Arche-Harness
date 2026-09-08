"""Case 3 主假设探索本机入口的参数、输入边界和 dry-run 合同测试。"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/tasks/case3_run_main_hypothesis_exploration_local.sh"
EXPECTED_FILES = (
    "origin.gjf",
    "ortho.gjf",
    "meta.gjf",
    "para.gjf",
    "dihedral_atoms.json",
)


def make_prepared_input_dir(tmp_path: Path, *, missing: str | None = None) -> Path:
    """创建只含测试占位内容的五文件 prepared input 目录。"""

    directory = tmp_path / "prepared-input"
    directory.mkdir(parents=True)
    for filename in EXPECTED_FILES:
        if filename != missing:
            (directory / filename).write_text("test input\n", encoding="utf-8")
    return directory


def run_dry_run(input_dir: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    """执行 dry-run；调用方可检查失败原因和零副作用。"""

    environment = os.environ.copy()
    environment["CASE3_RUN_NAME"] = "test-case3-exploration-dry-run"
    return subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--case3-prepared-input-dir",
            str(input_dir),
            "--dry-run",
            *arguments,
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def test_exploration_script_has_no_old_input_or_cluster_entry() -> None:
    """新入口只能创建本机新任务，不能引用旧结构化输入或集群提交入口。"""

    source = SCRIPT.read_text(encoding="utf-8")
    lowered = source.casefold()
    assert "rjob" not in lowered
    assert "submit_" not in lowered
    assert "--scientific-input-json" not in source
    assert "scientific_input_json" not in source
    assert "case3_author_main_hypothesis_scientific_input" not in source
    assert "--use-case3-author-main-hypothesis-advisory" not in source
    assert "--use-case3-author-main-hypothesis\"" not in source
    assert "--resume-task" not in source
    assert "local_data/legacy/author_result" not in source
    assert "gaussian_si" not in source
    assert "--use-case3-author-main-hypothesis-exploration" in source
    assert "--case3-prepared-input-dir" in source
    assert "--no-search" in source
    assert '"$PYTHON" -m chemistry_multiagent.controllers.chemistry_multiagent_controller' in source


def test_exploration_dry_run_displays_prepared_input_and_fixed_ports_without_content(
    tmp_path: Path,
) -> None:
    """dry-run 显示来源和端口，但不输出题面正文、密钥或输入文件正文。"""

    input_dir = make_prepared_input_dir(tmp_path)
    (input_dir / "origin.gjf").write_text("PRIVATE_INPUT_BODY\n", encoding="utf-8")
    completed = run_dry_run(
        input_dir,
        "--cpu",
        "7",
        "--memory-mb",
        "1234",
        "--gpu-count",
        "2",
    )

    assert completed.returncode == 0, completed.stderr
    output = completed.stdout
    assert "dry_run=1" in output
    assert f"case3_prepared_input_dir={input_dir}" in output
    assert "case3_prepared_input_files=origin.gjf ortho.gjf meta.gjf para.gjf dihedral_atoms.json" in output
    assert "catalog_endpoint=http://127.0.0.1:9011/mcp" in output
    assert "investigation_endpoint=http://127.0.0.1:9012/mcp" in output
    assert "services=catalog_9011,investigation_9012" in output
    assert "resources=cpu:7,memory_mb:1234,gpu:2" in output
    assert "ARCHE_PLANNER_DRIVER=codex" in output
    assert "ARCHE_CODEX_AUTH_MODE=api" in output
    assert "ARCHE_CODEX_MODEL_NAME=gpt-5.6-sol" in output
    assert "ARCHE_CODEX_REASONING_EFFORT=xhigh" in output
    assert "--use-case3-author-main-hypothesis-exploration" in output
    assert "--case3-prepared-input-dir" in output
    assert "--no-search" in output
    assert "PRIVATE_INPUT_BODY" not in output
    assert "In the nickel-catalyzed" not in output
    assert "ARCHE_CHEM_API_KEY" not in output
    assert "ARCHE_CODEX_API_KEY" not in output
    assert "side_effects=none" in output


def test_exploration_script_rejects_invalid_or_symlink_prepared_inputs(
    tmp_path: Path,
) -> None:
    """启动前必须看到五个普通文件；缺失或符号链接时不创建服务。"""

    missing_dir = tmp_path / "missing"
    completed = run_dry_run(missing_dir)
    assert completed.returncode != 0
    assert "prepared input" in completed.stderr

    incomplete = make_prepared_input_dir(tmp_path, missing="dihedral_atoms.json")
    completed = run_dry_run(incomplete)
    assert completed.returncode != 0
    assert "dihedral_atoms.json" in completed.stderr

    symlink_file_dir = tmp_path / "symlink-file"
    symlink_file_dir.mkdir()
    for filename in EXPECTED_FILES:
        target = tmp_path / f"target-{filename}"
        target.write_text("test input\n", encoding="utf-8")
        (symlink_file_dir / filename).symlink_to(target)
    completed = run_dry_run(symlink_file_dir)
    assert completed.returncode != 0
    assert "prepared input" in completed.stderr

    real_input = make_prepared_input_dir(tmp_path / "nested")
    symlink_dir = tmp_path / "prepared-link"
    symlink_dir.symlink_to(real_input, target_is_directory=True)
    completed = run_dry_run(symlink_dir)
    assert completed.returncode != 0
    assert "非符号链接目录" in completed.stderr


def test_exploration_script_rejects_port_overrides(tmp_path: Path) -> None:
    """入口固定使用 MiniChem profile 支持的 9011/9012，不接受端口改写。"""

    input_dir = make_prepared_input_dir(tmp_path)
    for arguments in (
        ("--catalog-port", "19011"),
        ("--investigation-port", "19012"),
    ):
        completed = run_dry_run(input_dir, *arguments)
        assert completed.returncode == 2
        assert "未知参数" in completed.stderr


def test_exploration_script_uses_services_readiness_and_new_task_settings() -> None:
    """入口应固定服务顺序、资源、端口传递、API Codex 和新任务参数。"""

    source = SCRIPT.read_text(encoding="utf-8")
    for setting in (
        "export ARCHE_PLANNER_DRIVER=codex",
        "export ARCHE_CODEX_AUTH_MODE=api",
        "export ARCHE_CODEX_MODEL_NAME=gpt-5.6-sol",
        "export ARCHE_CODEX_REASONING_EFFORT=xhigh",
        "export ARCHE_CODEX_ALLOW_PROVIDER_MIGRATION=0",
        'export ARCHE_CODEX_BIN="$VENDORED_CODEX_BIN"',
        "unset ARCHE_CODEX_AUTH_SOURCE ARCHE_CODEX_LOCAL_LOGIN",
        'export MINICHEM_CATALOG_ENDPOINT="$CATALOG_ENDPOINT"',
        'export MINICHEM_INVESTIGATION_ENDPOINT="$INVESTIGATION_ENDPOINT"',
        'export ARCHE_RUN_STORAGE_ROOT="$RUNTIME_ROOT"',
        'export ARCHE_STAGE_HARNESS_WORKSPACE_ROOT="$RUNTIME_ROOT/harness_workspaces"',
        'export ARCHE_STAGE_HARNESS_TRACE_ROOT="$RUNTIME_ROOT/harness_traces"',
        'export MINICHEM_SCOPE_REGISTRY_ROOT="$RUNTIME_ROOT/minichem_scope_registry"',
        'export MINICHEM_SERVICE_WORKSPACE="$RUNTIME_ROOT/minichem_service"',
        'MINICHEM_PROFILE_PORT="$CATALOG_PORT"',
        'MINICHEM_PROFILE_PORT="$INVESTIGATION_PORT"',
        "MINICHEM_CATALOG_BOOTSTRAP_MCP_PORT",
        "MINICHEM_INVESTIGATION_MCP_PORT",
        "RESEARCHCHEMBENCH_AVAILABLE_CPU_CORES",
        "RESEARCHCHEMBENCH_AVAILABLE_MEMORY_MB",
        "RESEARCHCHEMBENCH_AVAILABLE_GPU_COUNT",
        'bash "$ROOT/scripts/services/start_minichem.sh" --profile catalog_bootstrap',
        '--profile investigation',
        "--scope-auth",
        "--scope-registry",
        '--scope-registry "$MINICHEM_SCOPE_REGISTRY_ROOT"',
        "--scheduler-root",
        '"$PYTHON" "$ROOT/scripts/tasks/case3_check_mcp_readiness.py"',
        '--catalog-port "$CATALOG_PORT"',
        '--investigation-port "$INVESTIGATION_PORT"',
        "--timeout-seconds 120",
        "create-standard-research-task",
        "--use-case3-author-main-hypothesis-exploration",
        '--case3-prepared-input-dir "$PREPARED_INPUT_DIR"',
        "--paper-search-target-papers 50",
        "--top-n-hypotheses 1",
        "--max-parallel-investigations 1",
        "--max-hypothesis-batches 1",
        "--max-investigation-repairs-per-combination 3",
        'PIPE_STATUS=("${PIPESTATUS[@]}")',
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
    ):
        assert legacy_name not in source


def test_exploration_shell_script_parses() -> None:
    """本机入口必须通过 Bash 语法检查。"""

    completed = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
