"""脚本目录分类和头部说明的确定性测试。"""

from __future__ import annotations

import ast
from pathlib import Path
import re


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPOSITORY_ROOT / "scripts"
ROOT_SCRIPT_NAMES = {
    "bootstrap_reproduction.sh",
    "monitor_harness_run.sh",
    "run_end_to_end.sh",
    "run_paper_search_only.py",
    "run_to_hypothesis_top5.sh",
}
OBSOLETE_SCRIPT_NAMES = {
    "check_case1_mcp_readiness.py",
    "check_minichem_connection.py",
    "monitor_harness_exit.py",
    "run_case1_author_main_hypothesis_rjob_worker.sh",
    "submit_case1_author_main_hypothesis_rjob.sh",
}
CHINESE_CHARACTER = re.compile(r"[\u3400-\u9fff]")


def test_script_root_contains_only_generic_entrypoints() -> None:
    """根目录只保存通用入口，特定任务脚本统一放入扁平 tasks 目录。"""

    root_scripts = {
        path.name
        for path in SCRIPT_ROOT.iterdir()
        if path.is_file() and path.suffix in {".py", ".sh"}
    }
    assert root_scripts == ROOT_SCRIPT_NAMES

    task_root = SCRIPT_ROOT / "tasks"
    assert task_root.is_dir()
    assert all(
        path.is_file() for path in task_root.iterdir() if path.name != "__pycache__"
    )
    assert not any(
        path.name in OBSOLETE_SCRIPT_NAMES for path in SCRIPT_ROOT.rglob("*")
    )


def test_every_script_has_a_chinese_header_description() -> None:
    """每个 Python 或 Shell 脚本都在文件开头说明用途。"""

    scripts = sorted(
        path
        for path in SCRIPT_ROOT.rglob("*")
        if path.is_file() and path.suffix in {".py", ".sh"}
    )
    assert scripts
    for path in scripts:
        source = path.read_text(encoding="utf-8")
        if path.suffix == ".py":
            description = ast.get_docstring(ast.parse(source)) or ""
        else:
            header_lines = [
                line
                for line in source.splitlines()[:14]
                if line.startswith("#") and not line.startswith("#!")
            ]
            description = "\n".join(header_lines)
        assert description, path
        assert CHINESE_CHARACTER.search(description), path
