"""函数级 CRAP 测量脚本的确定性测试。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import coverage
import pytest

from scripts.quality.measure_crap import (
    CrapMeasurementError,
    CoverageMappingError,
    _parse_json_branch_arcs,
    _read_cohort_paths,
    main,
    measure_crap_for_sources,
)


def _write_source(tmp_path: Path, name: str, text: str) -> Path:
    """写入临时源码并返回绝对路径。"""

    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _capture(path: Path, data_path: Path, *, branch: bool = True) -> coverage.Coverage:
    """用 Coverage.py 真实执行临时源码并保存 SQLite 数据。"""

    runner = coverage.Coverage(
        data_file=str(data_path),
        branch=branch,
        source=[str(path.parent)],
    )
    runner.start()
    namespace: dict[str, object] = {}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
    runner.stop()
    runner.save()
    return runner


def _write_manifest(path: Path, source: Path, digest: str | None = None) -> Path:
    """写入覆盖率采集时的源码哈希清单。"""

    value = digest or hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = path / "hashes.json"
    manifest.write_text(json.dumps({str(source): value}), encoding="utf-8")
    return manifest


def test_branch_coverage_uses_real_arcs_and_formula(tmp_path: Path) -> None:
    """有分支函数必须使用 covered_branches/num_branches。"""

    source = _write_source(
        tmp_path,
        "branch.py",
        "def choose(value):\n"
        "    if value:\n"
        "        return 1\n"
        "    return 2\n\n"
        "choose(True)\n",
    )
    data_path = tmp_path / ".coverage"
    _capture(source, data_path)

    report = measure_crap_for_sources([source], data_path)
    result = report.functions[0]

    assert result.qualified_name == "choose"
    assert result.coverage_mode == "branches"
    assert (result.covered_branches, result.num_branches) == (1, 2)
    assert result.coverage == pytest.approx(0.5)
    assert result.crap == pytest.approx(2 + 2**2 * (1 - 0.5) ** 3)


def test_branchless_function_uses_line_fallback(tmp_path: Path) -> None:
    """没有分支的函数必须退回函数范围内可执行行覆盖率。"""

    source = _write_source(tmp_path, "plain.py", "def value():\n    return 3\n\nvalue()\n")
    data_path = tmp_path / ".coverage"
    _capture(source, data_path)

    result = measure_crap_for_sources([source], data_path).functions[0]

    assert result.coverage_mode == "line_fallback"
    assert result.covered_branches is None
    assert result.num_branches is None
    assert result.num_lines == 2
    assert result.covered_lines == 2
    assert result.coverage == 1.0


def test_sqlite_without_arcs_fails_only_for_static_branches(tmp_path: Path) -> None:
    """SQLite 无 arcs 时，分支源码失败而无分支源码仍可行覆盖率回退。"""

    plain = _write_source(tmp_path, "sqlite_plain.py", "def value():\n    return 3\n\nvalue()\n")
    plain_data = tmp_path / "plain.coverage"
    _capture(plain, plain_data, branch=False)
    plain_result = measure_crap_for_sources([plain], plain_data).functions[0]
    assert plain_result.coverage_mode == "line_fallback"

    branched = _write_source(
        tmp_path,
        "sqlite_branched.py",
        "def choose(value):\n"
        "    if value:\n"
        "        return 1\n"
        "    return 2\n\n"
        "choose(True)\n",
    )
    branched_data = tmp_path / "branched.coverage"
    _capture(branched, branched_data, branch=False)
    with pytest.raises(CoverageMappingError, match="没有 arcs 数据"):
        measure_crap_for_sources([branched], branched_data)


def test_methods_and_nested_functions_have_unique_qualified_keys(tmp_path: Path) -> None:
    """同名方法和嵌套函数必须按限定名与起始行区分。"""

    source = _write_source(
        tmp_path,
        "scopes.py",
        "class First:\n"
        "    def same(self, value):\n"
        "        return value\n\n"
        "class Second:\n"
        "    def same(self, value):\n"
        "        return value + 1\n\n"
        "def outer(value):\n"
        "    def same(inner):\n"
        "        if inner:\n"
        "            return 1\n"
        "        return 0\n"
        "    return same(value)\n\n"
        "First().same(1)\nSecond().same(1)\nouter(True)\n",
    )
    data_path = tmp_path / ".coverage"
    _capture(source, data_path)

    results = measure_crap_for_sources([source], data_path).functions
    names = [result.qualified_name for result in results]
    keys = [result.key for result in results]

    assert names == ["First.same", "Second.same", "outer", "outer.same"]
    assert len(keys) == len(set(keys)) == 4
    assert next(result for result in results if result.qualified_name == "outer").num_lines == 2
    assert next(result for result in results if result.qualified_name == "outer.same").coverage_mode == "branches"


def test_json_context_report_maps_function_ranges(tmp_path: Path) -> None:
    """Coverage.py JSON（含 contexts）与 SQLite 得出同一函数分支结果。"""

    source = _write_source(
        tmp_path,
        "json_case.py",
        "def choose(value):\n"
        "    if value:\n"
        "        return 1\n"
        "    return 2\n\n"
        "choose(True)\n",
    )
    data_path = tmp_path / ".coverage"
    runner = _capture(source, data_path)
    json_path = tmp_path / "coverage.json"
    runner.json_report(outfile=str(json_path), show_contexts=True)

    result = measure_crap_for_sources([source], json_path, coverage_format="json").functions[0]

    assert result.coverage_mode == "branches"
    assert (result.covered_branches, result.num_branches) == (1, 2)


def test_json_source_hash_is_read_from_file_entry(tmp_path: Path) -> None:
    """JSON 文件条目的有效源码哈希必须参与覆盖率快照校验。"""

    source = _write_source(
        tmp_path,
        "json_hash.py",
        "def choose(value):\n"
        "    if value:\n"
        "        return 1\n"
        "    return 2\n\n"
        "choose(True)\n",
    )
    data_path = tmp_path / ".coverage"
    runner = _capture(source, data_path)
    json_path = tmp_path / "coverage_hash.json"
    runner.json_report(outfile=str(json_path), show_contexts=True)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    payload["files"][str(source)]["source_hash"] = hashlib.sha256(
        source.read_bytes()
    ).hexdigest()
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    result = measure_crap_for_sources([source], json_path, coverage_format="json")

    assert result.functions[0].coverage_mode == "branches"


def test_json_branch_stats_rejects_unmapped_and_inconsistent_arcs(
    tmp_path: Path,
) -> None:
    """JSON 分支弧未知、数量不一致或缺失时必须 fail closed。"""

    source = _write_source(
        tmp_path,
        "json_invalid_branch.py",
        "def choose(value):\n"
        "    if value:\n"
        "        return 1\n"
        "    return 2\n\n"
        "choose(True)\n",
    )
    data_path = tmp_path / ".coverage"
    runner = _capture(source, data_path)
    valid_path = tmp_path / "valid_branch.json"
    runner.json_report(outfile=str(valid_path), show_contexts=True)
    valid_payload = json.loads(valid_path.read_text(encoding="utf-8"))
    valid_entry = valid_payload["files"][str(source)]

    mismatched_payload = {
        "meta": valid_payload["meta"],
        "files": {
            str(source): {
                **valid_entry,
                "summary": {
                    **valid_entry["summary"],
                    "num_branches": valid_entry["summary"]["num_branches"] + 1,
                },
            }
        },
    }
    mismatched_path = tmp_path / "mismatched_branch.json"
    mismatched_path.write_text(json.dumps(mismatched_payload), encoding="utf-8")
    with pytest.raises(CoverageMappingError, match="数量与弧数量不一致"):
        measure_crap_for_sources([source], mismatched_path, coverage_format="json")

    unknown_payload = {
        "meta": valid_payload["meta"],
        "files": {
            str(source): {
                "executed_lines": [1, 2],
                "summary": {"num_branches": 1},
                "executed_branches": [[999, 1000]],
                "missing_branches": [],
            }
        },
    }
    unknown_path = tmp_path / "unknown_branch.json"
    unknown_path.write_text(json.dumps(unknown_payload), encoding="utf-8")
    with pytest.raises(CoverageMappingError, match="无法映射"):
        measure_crap_for_sources([source], unknown_path, coverage_format="json")

    absent_payload = {
        "meta": valid_payload["meta"],
        "files": {
            str(source): {
                "executed_lines": [1, 2],
                "summary": {"num_branches": 0},
            }
        },
    }
    absent_path = tmp_path / "absent_branch.json"
    absent_path.write_text(json.dumps(absent_payload), encoding="utf-8")
    with pytest.raises(CoverageMappingError, match="没有分支覆盖率数据"):
        measure_crap_for_sources([source], absent_path, coverage_format="json")


def test_json_line_report_uses_line_fallback(tmp_path: Path) -> None:
    """没有分支的 Coverage.py JSON 必须使用函数语句行覆盖率。"""

    source = _write_source(tmp_path, "json_plain.py", "def value():\n    return 3\n\nvalue()\n")
    data_path = tmp_path / ".coverage"
    runner = _capture(source, data_path)
    json_path = tmp_path / "coverage_plain.json"
    runner.json_report(outfile=str(json_path), show_contexts=True)

    result = measure_crap_for_sources([source], json_path, coverage_format="json").functions[0]

    assert result.coverage_mode == "line_fallback"
    assert result.coverage == 1.0


@pytest.mark.parametrize("coverage_format", ["sqlite", "json"])
def test_no_cover_branch_is_not_counted_as_missing_static_data(
    tmp_path: Path,
    coverage_format: str,
) -> None:
    """Coverage 明确排除的分支不能被静态弧核对重新加入。"""

    source = _write_source(
        tmp_path,
        "excluded_branch.py",
        "def choose(value):\n"
        "    if value:  # pragma: no cover\n"
        "        return 1\n"
        "    return 2\n\n"
        "choose(False)\n",
    )
    data_path = tmp_path / ".coverage"
    runner = _capture(source, data_path)
    coverage_path = data_path
    if coverage_format == "json":
        coverage_path = tmp_path / "excluded_branch.json"
        runner.json_report(outfile=str(coverage_path), show_contexts=True)

    result = measure_crap_for_sources(
        [source],
        coverage_path,
        coverage_format=coverage_format,
    ).functions[0]

    assert result.coverage_mode == "line_fallback"
    assert result.coverage == 1.0


def test_json_no_branch_credits_only_explicitly_suppressed_arc(tmp_path: Path) -> None:
    """JSON 未列出的 no-branch 出口按 Coverage 摘要计为已覆盖。"""

    source = _write_source(
        tmp_path,
        "partial_branch.py",
        "def choose(value):\n"
        "    if value:  # pragma: no branch\n"
        "        return 1\n"
        "    return 2\n\n"
        "choose(True)\n",
    )
    data_path = tmp_path / ".coverage"
    runner = _capture(source, data_path)
    json_path = tmp_path / "partial_branch.json"
    runner.json_report(outfile=str(json_path), show_contexts=True)

    result = measure_crap_for_sources(
        [source], json_path, coverage_format="json"
    ).functions[0]

    assert result.coverage_mode == "branches"
    assert (result.covered_branches, result.num_branches) == (2, 2)
    assert result.coverage == 1.0


def test_json_branch_parser_rejects_incomplete_and_overlapping_arcs() -> None:
    """JSON 分支数组缺失或交叠时必须 fail closed。"""

    with pytest.raises(CoverageMappingError, match="summary 无效"):
        _parse_json_branch_arcs("x.py", {"summary": []})
    with pytest.raises(CoverageMappingError, match="缺少 executed_branches"):
        _parse_json_branch_arcs("x.py", {"summary": {"num_branches": 1}})
    with pytest.raises(CoverageMappingError, match="缺少 missing_branches"):
        _parse_json_branch_arcs(
            "x.py",
            {"summary": {"num_branches": 1}, "executed_branches": [[1, 2]]},
        )
    with pytest.raises(CoverageMappingError, match="同时出现在"):
        _parse_json_branch_arcs(
            "x.py",
            {
                "summary": {"num_branches": 1},
                "executed_branches": [[1, 2]],
                "missing_branches": [[1, 2]],
            },
        )
    assert _parse_json_branch_arcs("x.py", {"summary": {"num_branches": 0}})[0] == 0


def test_cohort_reader_supports_text_and_json_containers(tmp_path: Path) -> None:
    """cohort 支持注释行、路径列表和 paths 容器。"""

    text_path = tmp_path / "cohort.txt"
    text_path.write_text("# comment\nfirst.py\n\nsecond.py\n", encoding="utf-8")
    assert _read_cohort_paths(text_path) == ["first.py", "second.py"]

    list_path = tmp_path / "cohort-list.json"
    list_path.write_text(json.dumps(["first.py"]), encoding="utf-8")
    assert _read_cohort_paths(list_path) == ["first.py"]

    object_path = tmp_path / "cohort-object.json"
    object_path.write_text(json.dumps({"paths": ["second.py"]}), encoding="utf-8")
    assert _read_cohort_paths(object_path) == ["second.py"]


def test_cli_cohort_input_can_print_without_report_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """CLI 从 cohort 文件读取源码且不指定输出文件时仍返回通过。"""

    source = _write_source(tmp_path, "cohort_cli.py", "def value():\n    return 3\n\nvalue()\n")
    data_path = tmp_path / ".coverage"
    _capture(source, data_path)
    cohort_path = tmp_path / "cohort.txt"
    cohort_path.write_text(f"# comment\n{source}\n", encoding="utf-8")

    exit_code = main(
        [
            "--cohort",
            str(cohort_path),
            "--coverage-data",
            str(data_path),
        ]
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["status"] == "pass"


def test_source_hash_manifest_rejects_stale_coverage(tmp_path: Path) -> None:
    """覆盖率快照的源哈希过期时必须 fail closed。"""

    source = _write_source(tmp_path, "stale.py", "def value():\n    return 3\n\nvalue()\n")
    data_path = tmp_path / ".coverage"
    _capture(source, data_path)
    manifest = _write_manifest(tmp_path, source, "0" * 64)

    with pytest.raises(CrapMeasurementError, match="源哈希已过期"):
        measure_crap_for_sources([source], data_path, source_hash_manifest=manifest)


def test_unmapped_function_fails_closed(tmp_path: Path) -> None:
    """coverage 中没有源码文件记录时不能把文件覆盖率套给函数。"""

    source = _write_source(tmp_path, "unmapped.py", "def value():\n    return 3\n\nvalue()\n")
    other_dir = tmp_path / "measured"
    other_dir.mkdir()
    other = _write_source(other_dir, "other.py", "def other():\n    return 1\n\nother()\n")
    data_path = other_dir / ".coverage"
    _capture(other, data_path)

    with pytest.raises(CoverageMappingError, match="没有对应的覆盖率记录"):
        measure_crap_for_sources([source], data_path)


def test_ambiguous_coverage_paths_fail_closed(tmp_path: Path) -> None:
    """两个 coverage 路径同时命中同一源码时必须拒绝猜测。"""

    source = _write_source(tmp_path, "ambiguous.py", "def value():\n    return 3\n\nvalue()\n")
    payload = {
        "meta": {"format": 3, "version": "7.15.4", "branch_coverage": False},
        "files": {
            str(source): {"executed_lines": [1, 2], "summary": {"num_branches": 0}},
            "ambiguous.py": {"executed_lines": [1, 2], "summary": {"num_branches": 0}},
        },
    }
    json_path = tmp_path / "ambiguous.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CoverageMappingError, match="多个覆盖率记录"):
        measure_crap_for_sources([source], json_path, coverage_format="json")


def test_equal_threshold_is_a_strict_failure(tmp_path: Path) -> None:
    """CRAP 等于阈值时不能通过严格小于门禁。"""

    source = _write_source(tmp_path, "threshold.py", "def value():\n    return 3\n\nvalue()\n")
    data_path = tmp_path / ".coverage"
    _capture(source, data_path)

    report = measure_crap_for_sources([source], data_path, threshold=1.0)

    assert report.functions[0].crap == 1.0
    assert report.passes is False


def test_cli_writes_json_and_markdown_and_returns_gate_status(tmp_path: Path) -> None:
    """CLI 必须写入调用者指定路径，并以门禁状态返回退出码。"""

    source = _write_source(tmp_path, "cli.py", "def value():\n    return 3\n\nvalue()\n")
    data_path = tmp_path / ".coverage"
    _capture(source, data_path)
    json_path = tmp_path / "runtime" / "crap.json"
    markdown_path = tmp_path / "runtime" / "crap.md"

    exit_code = main(
        [
            "--source",
            str(source),
            "--coverage-data",
            str(data_path),
            "--threshold",
            "1",
            "--output-json",
            str(json_path),
            "--output-markdown",
            str(markdown_path),
        ]
    )

    assert exit_code == 1
    assert json.loads(json_path.read_text(encoding="utf-8"))["status"] == "fail"
    assert "line_fallback" in markdown_path.read_text(encoding="utf-8")
