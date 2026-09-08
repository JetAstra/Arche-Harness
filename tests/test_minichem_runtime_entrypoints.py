"""验证 MiniChem 启动入口使用统一的 arche 标准环境。"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TOOLBOX_ROOT = REPOSITORY_ROOT / "third_party" / "minichem_toolbox"
STANDARD_PYTHON = REPOSITORY_ROOT / ".venv" / "bin" / "python"
ARCHE_PYTHON = STANDARD_PYTHON
STANDARD_PYTHONPATH = "src:third_party/minichem_toolbox/src"
ROOT_STARTUP = REPOSITORY_ROOT / "scripts" / "services" / "start_minichem.sh"
INVESTIGATION_STARTUP = (
    REPOSITORY_ROOT / "scripts" / "services" / "start_investigation_minichem.sh"
)
CATALOG_STARTUP = (
    REPOSITORY_ROOT / "scripts" / "services" / "start_catalog_bootstrap_mcp.sh"
)
CONTROLLER_SOURCE = (
    REPOSITORY_ROOT
    / "src"
    / "chemistry_multiagent"
    / "controllers"
    / "chemistry_multiagent_controller.py"
)
CASE1_REPRODUCTION = REPOSITORY_ROOT / "scripts" / "tasks" / "case1_reproduce.py"
CASE23_REPRODUCTION = REPOSITORY_ROOT / "scripts" / "tasks" / "case23_reproduce.py"
PAPER_SEARCH_ONLY = REPOSITORY_ROOT / "scripts" / "run_paper_search_only.py"
CASE1_TOOL_PATH_VALIDATION = (
    REPOSITORY_ROOT / "scripts" / "tasks" / "case1_validate_tool_path.py"
)
HARNESS_BASE = (
    REPOSITORY_ROOT / "src" / "chemistry_multiagent" / "harness" / "base.py"
)
LLM_TRACE = (
    REPOSITORY_ROOT / "src" / "chemistry_multiagent" / "utils" / "llm_trace.py"
)
CASE23_CONTROLLER_REPRODUCTION = (
    REPOSITORY_ROOT
    / "src"
    / "chemistry_multiagent"
    / "controllers"
    / "case23_reproduction.py"
)
TOOLBOX_STARTUP = TOOLBOX_ROOT / "scripts" / "start_mcp.sh"
REPRODUCTION_BOOTSTRAP = REPOSITORY_ROOT / "scripts" / "bootstrap_reproduction.sh"
ACTIVE_RUNTIME_SCRIPTS = (
    ROOT_STARTUP,
    TOOLBOX_STARTUP,
    REPRODUCTION_BOOTSTRAP,
    REPOSITORY_ROOT / "scripts" / "run_end_to_end.sh",
    CATALOG_STARTUP,
    INVESTIGATION_STARTUP,
    TOOLBOX_ROOT / "scripts" / "export_environment_lock.sh",
    TOOLBOX_ROOT / "tests" / "harnesses" / "common.sh",
    TOOLBOX_ROOT / "tests" / "react" / "run_react.sh",
)
FORMAL_RUNTIME_DOCUMENTS = (
    REPOSITORY_ROOT / "README.md",
    REPOSITORY_ROOT / "docs" / "current" / "DELIVERY_AND_REPRODUCTION.md",
    TOOLBOX_ROOT / "README.md",
    TOOLBOX_ROOT / "TOOLBOX.md",
    TOOLBOX_ROOT / ".mini_software_cache" / "README.md",
)
def _help_environment(tmp_path: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("ARCHE_PYTHON", None)
    environment.pop("MINICHEM_RUNTIME_PYTHON", None)
    environment["RESEARCHCHEMBENCH_CONFIG"] = str(tmp_path / "empty-config.env")
    (tmp_path / "empty-config.env").write_text("", encoding="utf-8")
    runtime_root = tmp_path / "runtime"
    environment["ARCHE_RUN_STORAGE_ROOT"] = str(runtime_root)
    environment["MINICHEM_SERVICE_WORKSPACE"] = str(runtime_root / "workspace")
    environment["PYTHONPATH"] = ""
    return environment


def test_minichem_launchers_declare_standard_python() -> None:
    """两个启动脚本必须使用固定的 arche 标准解释器。"""

    for script in (ROOT_STARTUP, TOOLBOX_STARTUP):
        source = script.read_text(encoding="utf-8")
        assert "$ROOT/.venv/bin/python" in source or "$REPOSITORY_ROOT/.venv/bin/python" in source
        assert ".envs/minichem/bin/python" not in source


def test_reproduction_bootstrap_uses_one_standard_environment() -> None:
    """复现入口只检查并补充统一的 arche 标准环境。"""

    source = REPRODUCTION_BOOTSTRAP.read_text(encoding="utf-8")
    assert 'STANDARD_ARCHE_PYTHON="$ROOT/.venv/bin/python"' in source
    assert ".envs/minichem" not in source
    assert "TOOLBOX_ROOT/scripts/bootstrap.sh" in source


def test_active_entrypoints_and_formal_docs_use_standard_pythonpath() -> None:
    """活动入口和正式文档必须固定标准解释器与完整模块搜索路径。"""

    for path in (*ACTIVE_RUNTIME_SCRIPTS, *FORMAL_RUNTIME_DOCUMENTS):
        source = path.read_text(encoding="utf-8")
        if path not in (TOOLBOX_ROOT / ".mini_software_cache" / "README.md",):
            assert ".venv/bin/python" in source, path
        assert ".envs/minichem/bin/python" not in source, path
    for path in FORMAL_RUNTIME_DOCUMENTS:
        source = path.read_text(encoding="utf-8")
        if path != TOOLBOX_ROOT / ".mini_software_cache" / "README.md":
            assert STANDARD_PYTHONPATH in source, path


def test_formal_runtime_docs_name_standard_interpreter_and_pythonpath() -> None:
    """正式运行文档必须给出标准解释器和模块搜索路径。"""

    for path in FORMAL_RUNTIME_DOCUMENTS:
        source = path.read_text(encoding="utf-8")
        if path not in {
            REPOSITORY_ROOT / "docs" / "current" / "DELIVERY_AND_REPRODUCTION.md",
            TOOLBOX_ROOT / ".mini_software_cache" / "README.md",
        }:
            assert ".venv/bin/python" in source, path
        assert STANDARD_PYTHONPATH in source, path


def test_current_runtime_docs_keep_runtime_outside_the_repository() -> None:
    """当前 README 只要求仓库外私有目录，不暴露开发机内部路径。"""

    readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    delivery = (
        REPOSITORY_ROOT / "docs" / "current" / "DELIVERY_AND_REPRODUCTION.md"
    ).read_text(encoding="utf-8")
    config_example = (REPOSITORY_ROOT / "config.local.env.example").read_text(
        encoding="utf-8"
    )

    assert ".runtime/minichem" not in readme
    assert ".runtime/harness_workspaces" not in readme
    assert ".runtime/" not in delivery
    assert "Case 1 默认预算包含 1 张 GPU" not in readme
    assert "Case 1 TS Pipeline" not in config_example
    assert "local_data/runtime/" in readme
    assert "/absolute/path/to/arche-runtime" in readme


def test_active_python_defaults_use_ignored_local_data_locations() -> None:
    """活动 Python 入口不得把运行产物写入旧 `.runtime` 目录。"""

    for path in (
        CASE1_REPRODUCTION,
        CASE23_REPRODUCTION,
        PAPER_SEARCH_ONLY,
        CASE1_TOOL_PATH_VALIDATION,
        HARNESS_BASE,
        LLM_TRACE,
        CASE23_CONTROLLER_REPRODUCTION,
    ):
        source = path.read_text(encoding="utf-8")
        assert '".runtime/' not in source, path
        assert ' / ".runtime"' not in source, path
    assert "local_data/runtime/case1_gaussian_exact" in CASE1_REPRODUCTION.read_text(
        encoding="utf-8"
    )
    assert (
        "local_data/runtime/case1_harness/minichem"
        in CASE1_TOOL_PATH_VALIDATION.read_text(encoding="utf-8")
    )
    assert (
        "local_data/papers/case1-paper-search/papers"
        in PAPER_SEARCH_ONLY.read_text(encoding="utf-8")
    )


def test_root_minichem_startup_help_uses_local_python_and_stops_at_parser(
    tmp_path: Path,
) -> None:
    """根入口的帮助命令应由本地工具箱环境完成参数解析后立即退出。"""

    completed = subprocess.run(
        ["bash", str(ROOT_STARTUP), "--help"],
        cwd=REPOSITORY_ROOT,
        env=_help_environment(tmp_path),
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout
    assert "--transport" in completed.stdout
    assert not (tmp_path / "runtime" / "workspace" / "catalog_snapshot.json").exists()


def test_toolbox_minichem_startup_help_uses_local_python(
    tmp_path: Path,
) -> None:
    """工具箱入口的帮助命令应能在本地环境中完成参数解析。"""

    completed = subprocess.run(
        ["bash", str(TOOLBOX_STARTUP), "--help"],
        cwd=REPOSITORY_ROOT,
        env=_help_environment(tmp_path),
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout
    assert "--discovery-mode" in completed.stdout


def test_catalog_bootstrap_startup_help_isolated_to_private_runtime(
    tmp_path: Path,
) -> None:
    """catalog bootstrap 帮助入口也必须只创建私有 runtime 目录。"""

    repository_catalog = REPOSITORY_ROOT / "_toolbox_catalog.json"
    catalog_existed_before = repository_catalog.exists()
    catalog_content_before = (
        repository_catalog.read_bytes() if catalog_existed_before else None
    )
    environment = _help_environment(tmp_path)
    completed = subprocess.run(
        ["bash", str(CATALOG_STARTUP), "--help"],
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout
    workspace = tmp_path / "runtime" / "minichem_catalog_bootstrap"
    assert workspace.stat().st_mode & 0o777 == 0o700
    assert repository_catalog.exists() is catalog_existed_before
    if catalog_existed_before:
        assert repository_catalog.read_bytes() == catalog_content_before


def test_catalog_bootstrap_rejects_port_override_before_startup(
    tmp_path: Path,
) -> None:
    """旧 9010 环境不能把 catalog bootstrap 入口切走。"""

    environment = _help_environment(tmp_path)
    environment["MINICHEM_CATALOG_BOOTSTRAP_MCP_PORT"] = "9010"
    completed = subprocess.run(
        ["bash", str(CATALOG_STARTUP), "--help"],
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode != 0
    assert "9011" in completed.stderr


def test_investigation_minichem_startup_always_enables_scope_auth(
    tmp_path: Path,
) -> None:
    """新阶段专用入口必须固定 capability registry 和共享 scheduler。"""

    source = INVESTIGATION_STARTUP.read_text(encoding="utf-8")
    assert "--scope-auth" in source
    assert "--scope-registry" in source
    assert "--scheduler-root" in source

    environment = _help_environment(tmp_path)
    runtime_root = tmp_path / "runtime"
    environment["MINICHEM_SCOPE_REGISTRY_ROOT"] = str(runtime_root / "scope-registry")
    environment["MINICHEM_SHARED_SCHEDULER_ROOT"] = str(runtime_root / "scheduler")
    completed = subprocess.run(
        ["bash", str(INVESTIGATION_STARTUP), "--help"],
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--scope-auth" in completed.stdout
    assert (runtime_root / "scope-registry").stat().st_mode & 0o777 == 0o700
    assert (runtime_root / "scheduler").stat().st_mode & 0o777 == 0o700
    assert not (tmp_path / "runtime" / "workspace" / "catalog_snapshot.json").exists()


def test_investigation_minichem_startup_appends_enforced_options_after_overrides() -> None:
    """专用入口不能让调用者覆盖 Investigation profile 或 scope 参数。"""

    source = INVESTIGATION_STARTUP.read_text(encoding="utf-8")
    user_arguments = source.index('  "$@" \\\n')
    enforced_profile = source.index("  --profile investigation \\\n")
    enforced_scope_auth = source.index("  --scope-auth \\\n")
    enforced_registry = source.index("  --scope-registry \"$REGISTRY\" \\\n")
    enforced_scheduler = source.index("  --scheduler-root \"$SCHEDULER\" \\\n")
    enforced_transport = source.index("  --transport streamable_http \\\n")
    enforced_host = source.index("  --host \"$HOST\" \\\n")
    enforced_port = source.index("  --port \"$PORT\"\n")

    assert user_arguments < enforced_profile
    assert user_arguments < enforced_scope_auth
    assert user_arguments < enforced_registry
    assert user_arguments < enforced_scheduler
    assert user_arguments < enforced_transport
    assert user_arguments < enforced_host
    assert user_arguments < enforced_port


def test_startup_rejects_non_executable_arche_python_override(tmp_path: Path) -> None:
    """显式解释器覆盖路径不可执行时应在启动前失败。"""

    environment = _help_environment(tmp_path)
    environment["ARCHE_PYTHON"] = str(tmp_path / "missing-python")
    completed = subprocess.run(
        ["bash", str(TOOLBOX_STARTUP), "--help"],
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode != 0
    assert "MiniChem" in completed.stderr


def test_generic_entrypoint_declares_editable_defaults_and_private_runtime() -> None:
    """通用端到端入口的顶部参数区不携带特定论文实验，并隔离运行目录。"""

    source = (REPOSITORY_ROOT / "scripts" / "run_end_to_end.sh").read_text(
        encoding="utf-8"
    )

    assert "QUESTION=\"\"" in source
    assert "SCIENTIFIC_INPUT_JSON=\"\"" in source
    assert "CASE_PROFILE=\"\"" in source
    assert 'RUNTIME_ROOT="${ARCHE_RUN_STORAGE_ROOT:-$ROOT/local_data/runtime/case1_harness}"' in source
    assert 'export ARCHE_RUNTIME_ROOT' not in source
    assert "case1_fixed_reproduction" not in source
    assert "author_simplified_scientific_input.json" not in source
    assert "ARCHE_RUN_STORAGE_ROOT=" in source
    assert "ARCHE_STAGE_HARNESS_WORKSPACE_ROOT=" in source
    assert "ARCHE_STAGE_HARNESS_TRACE_ROOT=" in source
    assert "ARCHE_OUTPUT_ROOT=" not in source
    assert "ARCHE_HARNESS_WORKSPACE_ROOT=" not in source
    assert "ARCHE_HARNESS_TRACE_DIR=" not in source
    assert "RESEARCHCHEMBENCH_AVAILABLE_" not in source
    subprocess.run(["bash", "-n", str(REPOSITORY_ROOT / "scripts" / "run_end_to_end.sh")], check=True)


def test_generic_entrypoint_merges_top_defaults_and_cli_overrides_without_resume_config() -> None:
    """源码合同覆盖顶部参数映射、CLI 覆盖和恢复边界。"""

    source = (REPOSITORY_ROOT / "scripts" / "run_end_to_end.sh").read_text(
        encoding="utf-8"
    )

    for option in (
        "--question",
        "--scientific-input-json",
        "--case-profile",
        "--pdf-dir",
        "--index-dir",
        "--paper-search-manifest",
        "--no-search",
    ):
        assert option in source
    assert "argument_has_option" in source
    assert "CONTROLLER_ARGS+=(\"$@\")" in source
    assert "registry.revoke_scope_capability" not in source
    assert "streamable_http_client" not in source
    assert "curl" not in source
    assert "ARCHE_END_TO_END_" not in source
    assert "resume" in source.lower()
    assert "作者模式必须由脚本顶部 SCIENTIFIC_INPUT_JSON 或 CLI 提供" not in source


def test_generic_entrypoint_exports_fixed_endpoints_without_global_readiness() -> None:
    """通用入口只冻结双 MCP 地址，不在候选、确认或恢复前探测服务。"""

    source = (REPOSITORY_ROOT / "scripts" / "run_end_to_end.sh").read_text(
        encoding="utf-8"
    )

    assert 'EXPECTED_CATALOG_ENDPOINT="http://127.0.0.1:9011/mcp"' in source
    assert 'EXPECTED_INVESTIGATION_ENDPOINT="http://127.0.0.1:9012/mcp"' in source
    assert 'export MINICHEM_CATALOG_ENDPOINT="$EXPECTED_CATALOG_ENDPOINT"' in source
    assert 'export MINICHEM_INVESTIGATION_ENDPOINT="$EXPECTED_INVESTIGATION_ENDPOINT"' in source
    assert 'export MINICHEM_MCP_ENDPOINT="$EXPECTED_INVESTIGATION_ENDPOINT"' not in source
    assert "ensure_private_runtime_directory" in source
    assert "MCP readiness" not in source
    assert "provision_scope_capability" not in source
    assert "streamable_http_client" not in source
    assert "create-standard-research-task" in source
    assert "--run-mode" not in source
    assert "--resume-task" not in source


def test_current_config_example_only_lists_model_and_retrieval_fields() -> None:
    """共享配置模板只列当前端到端读取的模型、认证和检索字段。"""

    config_source = (REPOSITORY_ROOT / "config.local.env.example").read_text(
        encoding="utf-8"
    )
    active_lines = {
        line.strip()
        for line in config_source.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    active_names = {line.split("=", 1)[0] for line in active_lines}

    assert active_names == {
        "ARCHE_CHEM_API_KEY",
        "ARCHE_CHEM_BASE_URL",
        "ARCHE_CHEM_MODEL_NAME",
        "ARCHE_CHEM_BACKEND",
        "ARCHE_CODEX_API_KEY",
        "ARCHE_CODEX_BASE_URL",
        "ARCHE_CODEX_LOCAL_LOGIN_MODEL",
        "ARCHE_CODEX_LOCAL_LOGIN_REASONING_EFFORT",
        "ARCHE_CODEX_MODEL_NAME",
        "ARCHE_AGENT_A1_API_KEY",
        "ARCHE_AGENT_A1_BASE_URL",
        "ARCHE_AGENT_A1_MODEL_NAME",
        "ARCHE_REVIEW_CODEX_API_KEY",
        "ARCHE_REVIEW_CODEX_BASE_URL",
        "ARCHE_REVIEW_CODEX_MODEL_NAME",
        "ARCHE_RETRIEVAL_MODEL_PATH",
    }
    assert "ARCHE_REVIEW_CODEX_MODEL_NAME=deepseek-v4-pro" in active_lines

    startup_source = INVESTIGATION_STARTUP.read_text(encoding="utf-8")
    assert 'ENDPOINT="${MINICHEM_INVESTIGATION_ENDPOINT:-http://127.0.0.1:9012/mcp}"' in startup_source
    assert 'REGISTRY="${MINICHEM_SCOPE_REGISTRY_ROOT:-$RUNTIME_ROOT/minichem_scope_registry}"' in startup_source
    assert 'SCHEDULER="${MINICHEM_SHARED_SCHEDULER_ROOT:-$RUNTIME_ROOT/minichem_scheduler}"' in startup_source


def test_case1_minichem_launch_and_reproduction_use_six_hour_compute_timeout() -> None:
    """Case 1 的服务启动与独立复现入口都默认使用六小时计算墙钟。"""

    startup_source = ROOT_STARTUP.read_text(encoding="utf-8")
    reproduction_source = CASE1_REPRODUCTION.read_text(encoding="utf-8")

    assert (
        '${RESEARCHCHEMBENCH_COMPUTE_ACTION_TIMEOUT_SECONDS:-21600}'
        in startup_source
    )
    assert 'default=21600.0' in reproduction_source


def test_minichem_startup_uses_fixed_default_resource_budget() -> None:
    """Case 1 MiniChem 缺省资源预算必须固定包含一张 GPU。"""

    source = ROOT_STARTUP.read_text(encoding="utf-8")
    assert (
        'WORKSPACE="${MINICHEM_SERVICE_WORKSPACE:-$RUNTIME_ROOT/minichem}"' in source
    )
    assert "MINICHEM_DEFAULT_CPU_CORES=60" in source
    assert "MINICHEM_DEFAULT_MEMORY_MB=320000" in source
    assert "MINICHEM_DEFAULT_GPU_COUNT=1" in source
    assert (
        "${RESEARCHCHEMBENCH_AVAILABLE_CPU_CORES:-$MINICHEM_DEFAULT_CPU_CORES}"
        in source
    )
    assert (
        "${RESEARCHCHEMBENCH_AVAILABLE_MEMORY_MB:-$MINICHEM_DEFAULT_MEMORY_MB}"
        in source
    )
    assert (
        "${RESEARCHCHEMBENCH_AVAILABLE_GPU_COUNT:-$MINICHEM_DEFAULT_GPU_COUNT}"
        in source
    )
    assert "UMA/ASH TS Pipeline" not in source
    assert 'getconf _NPROCESSORS_ONLN' not in source
    assert 'nvidia-smi -L' not in source


def test_controller_defaults_minichem_artifacts_to_ignored_runtime() -> None:
    """绕过启动脚本时，Controller 也不得回退到旧 `.runtime` 工作区。"""

    source = CONTROLLER_SOURCE.read_text(encoding="utf-8")

    assert 'os.environ.get("ARCHE_RUN_STORAGE_ROOT", str(_REPOSITORY_ROOT))' in source
    assert '".runtime/minichem"' not in source
