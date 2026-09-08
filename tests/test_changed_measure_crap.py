"""基于 git diff 的新增/修改函数 CRAP cohort 确定性测试。"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import coverage
import pytest

from scripts.quality.measure_crap import (
    CrapMeasurementError,
    _write_text_output,
    generate_changed_function_cohort_manifest,
    main,
    measure_crap_for_changed_functions,
)


def _git(repository: Path, *arguments: str) -> str:
    """在临时仓库执行 git 命令并返回文本结果。"""

    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def _make_repository(tmp_path: Path, text: str, name: str = "module.py") -> tuple[Path, str]:
    """创建包含基线 commit 的临时仓库。"""

    repository = tmp_path / "repo"
    repository.mkdir()
    _git(repository, "init", "-b", "main")
    _git(repository, "config", "user.email", "test@example.invalid")
    _git(repository, "config", "user.name", "CRAP test")
    source = repository / name
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(text, encoding="utf-8")
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "baseline")
    baseline = _git(repository, "rev-parse", "HEAD")
    return repository, baseline


def _commit_current(repository: Path) -> None:
    """把工作树变更加入索引但不创建第二个 commit，模拟待审查 diff。"""

    _git(repository, "add", "-A")


def _capture(source: Path, data_path: Path) -> coverage.Coverage:
    """执行当前源码并保存 Coverage.py SQLite。"""

    runner = coverage.Coverage(
        data_file=str(data_path),
        branch=True,
        source=[str(source.parent.parent if source.parent.name == "pkg" else source.parent)],
    )
    runner.start()
    namespace: dict[str, object] = {}
    exec(compile(source.read_text(encoding="utf-8"), str(source), "exec"), namespace)
    runner.stop()
    runner.save()
    return runner


def test_unchanged_high_complexity_function_is_excluded(tmp_path: Path) -> None:
    """修改大文件中的一个函数时，未改的高复杂度历史函数不能混入。"""

    repository, baseline = _make_repository(
        tmp_path,
        "def ancient(value):\n"
        "    if value == 1:\n        return 1\n"
        "    elif value == 2:\n        return 2\n"
        "    elif value == 3:\n        return 3\n"
        "    else:\n        return 4\n\n"
        "def changed(value):\n    return value + 1\n\n"
        "ancient(1)\nchanged(1)\n",
    )
    source = repository / "module.py"
    source.write_text(
        source.read_text(encoding="utf-8").replace("return value + 1", "return value + 2"),
        encoding="utf-8",
    )
    _commit_current(repository)

    cohort = generate_changed_function_cohort_manifest(repository, baseline=baseline)

    assert [item.qualified_name for item in cohort.functions] == ["changed"]
    assert not cohort.deleted_functions
    assert cohort.functions[0].status == "modified"


def test_nested_functions_and_same_named_methods_are_located_by_scope(tmp_path: Path) -> None:
    """嵌套函数和同名方法必须各自按限定名纳入，未改方法不纳入。"""

    repository, baseline = _make_repository(
        tmp_path,
        "class First:\n"
        "    def same(self, value):\n        return value\n\n"
        "class Second:\n"
        "    def same(self, value):\n        return value + 1\n\n"
        "def outer(value):\n"
        "    def same(inner):\n        return inner\n"
        "    return same(value)\n\n"
        "First().same(1)\nSecond().same(1)\nouter(1)\n",
    )
    source = repository / "module.py"
    source.write_text(
        source.read_text(encoding="utf-8").replace("return value + 1", "return value + 2").replace(
            "return inner\n", "return inner + 1\n"
        ),
        encoding="utf-8",
    )
    _commit_current(repository)

    cohort = generate_changed_function_cohort_manifest(repository, baseline=baseline)
    names = [item.qualified_name for item in cohort.functions]

    assert names == ["Second.same", "outer", "outer.same"]
    assert "First.same" not in names
    assert len({item.current_key for item in cohort.functions}) == 3


def test_decorator_signature_and_body_changes_all_mark_same_function(tmp_path: Path) -> None:
    """装饰器、跨行签名和正文各自变化时不能漏掉同一个函数。"""

    repository, baseline = _make_repository(
        tmp_path,
        "def decorate(function):\n    return function\n\n"
        "@decorate\n"
        "def target(\n"
        "    value,\n"
        "):\n"
        "    return value + 1\n",
    )
    source = repository / "module.py"
    source.write_text(
        "def decorate(function):\n    return function\n\n"
        "@staticmethod\n"
        "def target(\n"
        "    value,\n"
        "    offset=1,\n"
        "):\n"
        "    return value + offset + 1\n",
        encoding="utf-8",
    )
    _commit_current(repository)

    cohort = generate_changed_function_cohort_manifest(repository, baseline=baseline)

    target = next(item for item in cohort.functions if item.qualified_name == "target")
    assert set(target.change_kinds) >= {"decorator", "signature", "body"}


def test_new_file_includes_every_function_and_method(tmp_path: Path) -> None:
    """全新 Python 文件的顶层、嵌套和方法函数都必须进入 cohort。"""

    repository, baseline = _make_repository(tmp_path, "def baseline():\n    return 1\n")
    source = repository / "new_module.py"
    source.write_text(
        "def first(value):\n"
        "    def nested(inner):\n        return inner + 1\n"
        "    return nested(value)\n\n"
        "class New:\n"
        "    def method(self):\n        return 2\n",
        encoding="utf-8",
    )
    _commit_current(repository)

    cohort = generate_changed_function_cohort_manifest(repository, baseline=baseline)

    assert [item.qualified_name for item in cohort.functions] == [
        "first",
        "first.nested",
        "New.method",
    ]
    assert all(item.change_kinds[0] == "file_added" for item in cohort.functions)


def test_explicit_untracked_source_is_treated_as_new_file(tmp_path: Path) -> None:
    """显式限定未跟踪 Python 文件时也必须纳入完整新增文件 cohort。"""

    repository, baseline = _make_repository(tmp_path, "def baseline():\n    return 1\n")
    source = repository / "untracked.py"
    source.write_text(
        "def first():\n    return 1\n\n"
        "def second():\n    return 2\n",
        encoding="utf-8",
    )

    cohort = generate_changed_function_cohort_manifest(
        repository,
        baseline=baseline,
    )

    assert [item.qualified_name for item in cohort.functions] == ["first", "second"]
    assert cohort.files[0].status == "A"


def test_staged_unstaged_and_untracked_python_files_are_collected_together(
    tmp_path: Path,
) -> None:
    """同一次扫描必须合并暂存、未暂存和未跟踪 Python 文件。"""

    repository, _ = _make_repository(tmp_path, "def baseline():\n    return 1\n")
    for name, function_name in (("staged.py", "staged"), ("unstaged.py", "unstaged")):
        (repository / name).write_text(
            f"def {function_name}():\n    return 1\n", encoding="utf-8"
        )
    _git(repository, "add", "staged.py", "unstaged.py")
    _git(repository, "commit", "-m", "tracked-files")
    baseline = _git(repository, "rev-parse", "HEAD")

    (repository / "staged.py").write_text("def staged():\n    return 2\n", encoding="utf-8")
    _git(repository, "add", "staged.py")
    (repository / "unstaged.py").write_text(
        "def unstaged():\n    return 2\n", encoding="utf-8"
    )
    (repository / "untracked.py").write_text(
        "def untracked():\n    return 3\n", encoding="utf-8"
    )

    cohort = generate_changed_function_cohort_manifest(repository, baseline=baseline)
    file_status = {item.path: item.status for item in cohort.files}
    function_paths = {item.path for item in cohort.functions}

    assert file_status == {
        "staged.py": "M",
        "untracked.py": "A",
        "unstaged.py": "M",
    }
    assert function_paths == {"staged.py", "unstaged.py", "untracked.py"}


def test_copy_file_is_detected_and_all_destination_functions_are_measured(
    tmp_path: Path,
) -> None:
    """复制文件必须保留旧路径并把目标文件函数纳入新增 cohort。"""

    repository, baseline = _make_repository(
        tmp_path,
        "def copied(value):\n    return value + 1\n\ncopied(1)\n",
        name="original.py",
    )
    (repository / "copy.py").write_text(
        (repository / "original.py").read_text(encoding="utf-8"), encoding="utf-8"
    )
    _git(repository, "add", "copy.py")

    cohort = generate_changed_function_cohort_manifest(repository, baseline=baseline)

    assert len(cohort.files) == 1
    assert cohort.files[0].status == "C"
    assert cohort.files[0].old_path == "original.py"
    assert [item.qualified_name for item in cohort.functions] == ["copied"]
    assert cohort.functions[0].status == "added"
    assert cohort.functions[0].old_path == "original.py"


def test_rename_and_line_displacement_use_current_path_and_lines(tmp_path: Path) -> None:
    """文件重命名并在前方插入行后，变更函数应定位到新路径的新行。"""

    repository, baseline = _make_repository(
        tmp_path,
        "def keep(value):\n    return value\n\n"
        "def changed(value):\n    return value + 1\n\n"
        "keep(1)\nchanged(1)\n",
        name="old_module.py",
    )
    _git(repository, "mv", "old_module.py", "new_module.py")
    source = repository / "new_module.py"
    source.write_text(
        "# moved\n" + source.read_text(encoding="utf-8").replace(
            "return value + 1", "return value + 2"
        ),
        encoding="utf-8",
    )
    _commit_current(repository)

    cohort = generate_changed_function_cohort_manifest(repository, baseline=baseline)

    assert [item.qualified_name for item in cohort.functions] == ["changed"]
    changed = cohort.functions[0]
    assert changed.path == "new_module.py"
    assert changed.old_path == "old_module.py"
    assert changed.current_start_line == 5
    assert changed.old_start_line == 4
    assert cohort.files[0].status == "R"


def test_deleted_function_is_reported_but_not_measured(tmp_path: Path) -> None:
    """删除函数进入 deleted_functions，不冒充当前 CRAP 结果。"""

    repository, baseline = _make_repository(
        tmp_path,
        "def removed(value):\n    return value\n\n"
        "def changed(value):\n    return value + 1\n",
    )
    source = repository / "module.py"
    source.write_text(
        "def changed(value):\n    return value + 2\n",
        encoding="utf-8",
    )
    _commit_current(repository)

    cohort = generate_changed_function_cohort_manifest(repository, baseline=baseline)

    assert [item.qualified_name for item in cohort.functions] == ["changed"]
    assert [item.qualified_name for item in cohort.deleted_functions] == ["removed"]
    assert all(item.current_start_line is None for item in cohort.deleted_functions)


def test_uncovered_changed_function_is_not_reported_as_full_coverage(tmp_path: Path) -> None:
    """Coverage 中未执行变更函数时，覆盖率必须低于 1 而不能假低。"""

    repository, baseline = _make_repository(
        tmp_path,
        "def target(value):\n"
        "    if value:\n        return 1\n"
        "    return 2\n",
    )
    source = repository / "module.py"
    source.write_text(
        source.read_text(encoding="utf-8").replace("return 2", "return 3"),
        encoding="utf-8",
    )
    _commit_current(repository)
    data_path = repository / ".coverage"
    _capture(source, data_path)

    report = measure_crap_for_changed_functions(
        data_path,
        repository=repository,
        baseline=baseline,
    )

    assert report.cohort_manifest is not None
    assert report.cohort_manifest["functions"][0]["qualified_name"] == "target"
    result = report.functions[0]
    assert result.coverage < 1.0
    assert result.crap > result.complexity


def test_default_baseline_requires_explicit_override_when_ref_is_missing(tmp_path: Path) -> None:
    """默认 origin/harness 不存在时必须要求显式基线，不能猜测 HEAD。"""

    repository, _ = _make_repository(tmp_path, "def target():\n    return 1\n")
    with pytest.raises(CrapMeasurementError, match="显式传入 --baseline"):
        generate_changed_function_cohort_manifest(repository)


def test_unresolvable_explicit_baseline_fails_closed(tmp_path: Path) -> None:
    """显式基线引用不存在时必须报错，不能退回当前 HEAD。"""

    repository, _ = _make_repository(tmp_path, "def target():\n    return 1\n")
    with pytest.raises(CrapMeasurementError, match="无法解析显式基线"):
        generate_changed_function_cohort_manifest(repository, baseline="missing-ref")


def test_manifest_is_machine_readable_and_contains_diff_evidence(tmp_path: Path) -> None:
    """manifest 必须可序列化并保留基线、函数和行级证据。"""

    repository, baseline = _make_repository(tmp_path, "def target():\n    return 1\n")
    source = repository / "module.py"
    source.write_text("def target():\n    return 2\n", encoding="utf-8")
    _commit_current(repository)

    cohort = generate_changed_function_cohort_manifest(repository, baseline=baseline)
    payload = json.loads(json.dumps(cohort.to_dict(), ensure_ascii=False))

    assert payload["baseline"]["resolved_commit"] == baseline
    assert payload["functions"][0]["evidence"]["added_lines"]
    assert payload["files"][0]["added_lines"]


def test_changed_cli_writes_report_and_cohort_manifest(tmp_path: Path) -> None:
    """CLI 以 baseline 自动切换 cohort 模式，并分别写出两个 JSON。"""

    repository, baseline = _make_repository(
        tmp_path,
        "def target():\n    return 1\n\n"
        "target()\n",
    )
    source = repository / "module.py"
    source.write_text("def target():\n    return 2\n\ntarget()\n", encoding="utf-8")
    _commit_current(repository)
    data_path = repository / ".coverage"
    _capture(source, data_path)
    report_path = tmp_path / "report.json"
    manifest_path = tmp_path / "cohort.json"

    exit_code = main(
        [
            "--baseline",
            baseline,
            "--repo-root",
            str(repository),
            "--coverage-data",
            str(data_path),
            "--output-json",
            str(report_path),
            "--cohort-manifest",
            str(manifest_path),
        ]
    )

    assert exit_code == 0
    assert json.loads(report_path.read_text(encoding="utf-8"))["functions"][0][
        "qualified_name"
    ] == "target"
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["counts"][
        "current_functions"
    ] == 1


def test_changed_measurement_rejects_source_hash_drift(tmp_path: Path) -> None:
    """变更 cohort 的覆盖率哈希过期时必须拒绝测量。"""

    repository, baseline = _make_repository(
        tmp_path,
        "def target():\n    return 1\n\ntarget()\n",
    )
    source = repository / "module.py"
    source.write_text("def target():\n    return 2\n\ntarget()\n", encoding="utf-8")
    _git(repository, "add", "module.py")
    data_path = repository / ".coverage"
    _capture(source, data_path)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest_path = tmp_path / "hashes.json"
    manifest_path.write_text(json.dumps({str(source): digest}), encoding="utf-8")
    source.write_text("def target():\n    return 3\n\ntarget()\n", encoding="utf-8")

    with pytest.raises(CrapMeasurementError, match="源哈希已过期"):
        measure_crap_for_changed_functions(
            data_path,
            repository=repository,
            baseline=baseline,
            source_hash_manifest=manifest_path,
        )


def test_atomic_report_write_keeps_previous_file_when_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """报告替换失败时旧内容必须保留且临时文件必须清理。"""

    import scripts.quality.measure_crap as measure_module

    output_path = tmp_path / "report.json"
    output_path.write_text("old\n", encoding="utf-8")

    def fail_replace(_temporary: object, _destination: object) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(measure_module.os, "replace", fail_replace)
    with pytest.raises(CrapMeasurementError, match="无法写入报告"):
        _write_text_output(output_path, "new\n")

    assert output_path.read_text(encoding="utf-8") == "old\n"
    assert not list(tmp_path.glob(f".{output_path.name}.*"))


def test_cli_invalid_baseline_returns_input_error_and_writes_error_payload(
    tmp_path: Path,
) -> None:
    """CLI 输入基线无效时返回 2，并原子写出结构化错误。"""

    repository, _ = _make_repository(tmp_path, "def target():\n    return 1\n")
    output_path = tmp_path / "error.json"

    exit_code = main(
        [
            "--baseline",
            "missing-ref",
            "--repo-root",
            str(repository),
            "--coverage-data",
            str(tmp_path / "missing.coverage"),
            "--output-json",
            str(output_path),
        ]
    )

    assert exit_code == 2
    assert json.loads(output_path.read_text(encoding="utf-8"))["status"] == "error"


def test_changed_cli_keeps_generated_cohort_in_error_payload_when_coverage_missing(
    tmp_path: Path,
) -> None:
    """变更清单已生成但覆盖率缺失时，错误报告仍须带完整 cohort。"""

    repository, baseline = _make_repository(
        tmp_path,
        "def target():\n    return 1\n\ntarget()\n",
    )
    source = repository / "module.py"
    source.write_text("def target():\n    return 2\n\ntarget()\n", encoding="utf-8")
    _git(repository, "add", "module.py")
    output_path = tmp_path / "missing-coverage-error.json"
    manifest_path = tmp_path / "generated-cohort.json"

    exit_code = main(
        [
            "--baseline",
            baseline,
            "--repo-root",
            str(repository),
            "--coverage-data",
            str(tmp_path / "missing.coverage"),
            "--output-json",
            str(output_path),
            "--cohort-manifest",
            str(manifest_path),
        ]
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 2
    assert payload["status"] == "error"
    assert payload["cohort_manifest"]["counts"]["current_functions"] == 1
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["counts"][
        "current_functions"
    ] == 1
