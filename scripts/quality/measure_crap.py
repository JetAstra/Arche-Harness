#!/usr/bin/env python3
"""测量函数级 CRAP 指数，并在覆盖率映射不可信时拒绝给出结论。

脚本把 Radon 的圈复杂度与 Coverage.py 的函数范围内覆盖率组合起来。覆盖率
优先使用分支弧；没有分支的函数才使用可执行语句行覆盖率。Coverage.py 的
SQLite 数据文件没有保存源文件哈希，因此可以通过 ``--source-hash-manifest``
提供采集覆盖率时的哈希清单，避免把旧覆盖率套在新源码上。

传入 ``--baseline`` 或 ``--changed-only`` 时，脚本先把该基线到工作树的
Python diff 映射为函数 cohort，再只对新增/修改函数执行同一套覆盖率与 Radon
测量；机器可读清单可通过 ``--cohort-manifest`` 输出。删除函数只在清单中报告。
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import coverage
from coverage.parser import PythonParser
from radon.complexity import cc_visit_ast


class CrapMeasurementError(RuntimeError):
    """覆盖率输入、源码或函数映射不可信时抛出的错误。"""


class CoverageMappingError(CrapMeasurementError):
    """覆盖率文件无法唯一映射到待测源码时抛出的错误。"""


Arc = tuple[int, int]

_GIT_HUNK_PATTERN = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@"
)


@dataclass(frozen=True)
class ChangedFunction:
    """保存一个由 git diff 证明发生变化的函数及其审计证据。"""

    status: str
    path: str
    old_path: str | None
    qualified_name: str
    current_start_line: int | None
    current_end_line: int | None
    old_start_line: int | None
    old_end_line: int | None
    change_kinds: tuple[str, ...]
    added_lines: tuple[int, ...]
    deleted_lines: tuple[int, ...]

    @property
    def current_key(self) -> str | None:
        """返回当前源码中的稳定定位键；删除函数没有当前键。"""

        if self.current_start_line is None:
            return None
        return f"{self.path}::{self.qualified_name}::{self.current_start_line}"

    def to_dict(self) -> dict[str, Any]:
        """返回不依赖本机绝对路径的机器可读函数记录。"""

        current = None
        if self.current_start_line is not None:
            current = {
                "qualified_name": self.qualified_name,
                "start_line": self.current_start_line,
                "end_line": self.current_end_line,
            }
        old = None
        if self.old_start_line is not None:
            old = {
                "qualified_name": self.qualified_name,
                "start_line": self.old_start_line,
                "end_line": self.old_end_line,
            }
        return {
            "status": self.status,
            "path": self.path,
            "old_path": self.old_path if self.old_path != self.path else None,
            "qualified_name": self.qualified_name,
            "current": current,
            "old": old,
            "change_kinds": list(self.change_kinds),
            "evidence": {
                "added_lines": list(self.added_lines),
                "deleted_lines": list(self.deleted_lines),
            },
        }


@dataclass(frozen=True)
class ChangedFileDiff:
    """保存一个 Python 文件的 git 状态、路径变更和逐行 diff 证据。"""

    status: str
    path: str
    old_path: str | None
    similarity: int | None
    added_lines: tuple[int, ...]
    deleted_lines: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        """返回文件级 diff 证据。"""

        return {
            "status": self.status,
            "path": self.path,
            "old_path": self.old_path if self.old_path != self.path else None,
            "similarity": self.similarity,
            "added_lines": list(self.added_lines),
            "deleted_lines": list(self.deleted_lines),
        }


@dataclass(frozen=True)
class ChangedFunctionCohort:
    """由基线和 git diff 自动得到的当前函数 cohort。"""

    baseline_spec: str
    baseline_commit: str
    repository: str
    files: tuple[ChangedFileDiff, ...]
    functions: tuple[ChangedFunction, ...]
    deleted_functions: tuple[ChangedFunction, ...]

    def to_dict(self) -> dict[str, Any]:
        """返回完整 cohort manifest；文件和函数均保留 diff 证据。"""

        return {
            "schema_version": 1,
            "baseline": {
                "requested": self.baseline_spec,
                "resolved_commit": self.baseline_commit,
            },
            "repository": self.repository,
            "files": [item.to_dict() for item in self.files],
            "functions": [item.to_dict() for item in self.functions],
            "deleted_functions": [item.to_dict() for item in self.deleted_functions],
            "counts": {
                "files": len(self.files),
                "current_functions": len(self.functions),
                "deleted_functions": len(self.deleted_functions),
            },
        }


@dataclass(frozen=True)
class CoverageFileRecord:
    """保存一个源码文件的覆盖率事实，不包含文件级覆盖率结论。"""

    raw_path: str
    statements: frozenset[int]
    executed_lines: frozenset[int]
    branch_totals: Mapping[int, int]
    branch_covered: Mapping[int, int]
    source_hash: str | None

    @property
    def has_branch_data(self) -> bool:
        """返回输入是否提供了可用于函数分支覆盖率的弧数据。"""

        return bool(self.branch_totals)


@dataclass(frozen=True)
class FunctionMeasurement:
    """一个函数的复杂度、覆盖率和 CRAP 结果。"""

    key: str
    path: str
    qualified_name: str
    start_line: int
    end_line: int
    complexity: int
    coverage_mode: str
    covered_branches: int | None
    num_branches: int | None
    covered_lines: int
    num_lines: int
    coverage: float
    crap: float

    def to_dict(self) -> dict[str, Any]:
        """返回稳定的机器可读函数记录。"""

        return {
            "key": self.key,
            "path": self.path,
            "qualified_name": self.qualified_name,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "complexity": self.complexity,
            "coverage_mode": self.coverage_mode,
            "covered_branches": self.covered_branches,
            "num_branches": self.num_branches,
            "covered_lines": self.covered_lines,
            "num_lines": self.num_lines,
            "coverage": self.coverage,
            "crap": self.crap,
        }


@dataclass(frozen=True)
class CrapReport:
    """一次 cohort 测量的完整结果。"""

    threshold: float
    coverage_data: str
    functions: tuple[FunctionMeasurement, ...]
    coverage_format: str
    cohort_manifest: Mapping[str, Any] | None = None

    @property
    def passes(self) -> bool:
        """严格判断所有函数是否满足 ``CRAP < threshold``。"""

        return bool(self.functions) and all(
            function.crap < self.threshold for function in self.functions
        )

    def to_dict(self) -> dict[str, Any]:
        """返回用于审计和 CI 门禁的 JSON 结构。"""

        payload = {
            "status": "pass" if self.passes else "fail",
            "threshold": self.threshold,
            "threshold_rule": "strictly_less_than",
            "coverage_data": self.coverage_data,
            "coverage_format": self.coverage_format,
            "formula": "CC + CC^2 * (1 - coverage)^3",
            "functions": [function.to_dict() for function in self.functions],
        }
        if self.cohort_manifest is not None:
            payload["cohort_manifest"] = dict(self.cohort_manifest)
        return payload

    def render_markdown(self) -> str:
        """把结果渲染为适合人工审阅的短表格。"""

        lines = [
            "# 函数级 CRAP 测量",
            "",
            f"- 状态：`{'pass' if self.passes else 'fail'}`",
            f"- 阈值：`CRAP < {self.threshold:g}`",
            f"- 覆盖率输入：`{self.coverage_data}`（{self.coverage_format}）",
            "- 公式：`CC + CC^2 * (1 - coverage)^3`",
            "",
            "| 函数唯一键 | CC | 覆盖率类型 | 覆盖率 | CRAP | 结果 |",
            "| --- | ---: | --- | ---: | ---: | --- |",
        ]
        if self.cohort_manifest is not None:
            baseline = self.cohort_manifest.get("baseline", {})
            lines.insert(
                5,
                f"- 变更基线：`{baseline.get('resolved_commit', 'unknown')}`；"
                f"当前函数：`{len(self.functions)}`；"
                f"删除函数：`{len(self.cohort_manifest.get('deleted_functions', []))}`",
            )
        for function in self.functions:
            result = "pass" if function.crap < self.threshold else "fail"
            lines.append(
                "| {key} | {cc} | {mode} | {coverage:.4f} | {crap:.4f} | {result} |".format(
                    key=function.key.replace("|", "\\|"),
                    cc=function.complexity,
                    mode=function.coverage_mode,
                    coverage=function.coverage,
                    crap=function.crap,
                    result=result,
                )
            )
        return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class _FunctionRecord:
    """AST 函数节点及其稳定限定名。"""

    node: ast.FunctionDef | ast.AsyncFunctionDef
    qualified_name: str
    start_line: int
    end_line: int


@dataclass
class _GitPatchParseState:
    """保存当前 git patch 文件头和 hunk 行号。"""

    current_file_key: tuple[str | None, str] | None = None
    old_line: int = 0
    new_line: int = 0
    in_hunk: bool = False
    pending_old_path: str | None = None
    pending_new_path: str | None = None


def _function_change_start_line(function: _FunctionRecord) -> int:
    """返回包含装饰器、签名和函数体的变更匹配起始行。"""

    decorator_lines = [decorator.lineno for decorator in function.node.decorator_list]
    return min([function.start_line, *decorator_lines])


def _function_body_start_line(function: _FunctionRecord) -> int:
    """返回函数体首行，用于区分装饰器、签名和正文变更。"""

    if not function.node.body:
        return function.start_line
    return min(statement.lineno for statement in function.node.body)


@dataclass(frozen=True)
class _CoverageSnapshot:
    """解析后的覆盖率文件集合。"""

    records: tuple[CoverageFileRecord, ...]
    format: str


def _normalise_path(path: str | os.PathLike[str], base: Path | None = None) -> str:
    """将覆盖率路径转换成绝对规范路径，失败时保留可诊断输入。"""

    candidate = Path(path)
    if not candidate.is_absolute() and base is not None:
        candidate = base / candidate
    return str(candidate.resolve())


def _read_source_text(path: Path) -> str:
    """读取 UTF-8 源码并在不能解析时抛出可定位错误。"""

    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CrapMeasurementError(f"无法读取源码：{path}: {exc}") from exc


def _source_hash(path: Path) -> str:
    """计算源码字节的 SHA-256，供覆盖率快照一致性校验使用。"""

    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise CrapMeasurementError(f"无法计算源码哈希：{path}: {exc}") from exc


def _parse_source(path: Path, text: str) -> ast.Module:
    """解析源码 AST；语法错误必须阻止本次门禁。"""

    try:
        return ast.parse(text, filename=str(path), type_comments=True)
    except SyntaxError as exc:
        raise CrapMeasurementError(
            f"源码无法解析，不能测量 CRAP：{path}:{exc.lineno}: {exc.msg}"
        ) from exc


def _collect_function_records(tree: ast.Module) -> tuple[_FunctionRecord, ...]:
    """收集函数、方法和嵌套函数，不生成 class 汇总记录。"""

    records: list[_FunctionRecord] = []

    class FunctionCollector(ast.NodeVisitor):
        """携带类和函数作用域收集 AST 函数。"""

        def __init__(self) -> None:
            self.scope: list[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._collect_function(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._collect_function(node)

        def _collect_function(
            self, node: ast.FunctionDef | ast.AsyncFunctionDef
        ) -> None:
            qualified_name = ".".join([*self.scope, node.name])
            if not qualified_name:
                raise CoverageMappingError("发现没有限定名的函数")
            records.append(
                _FunctionRecord(
                    node=node,
                    qualified_name=qualified_name,
                    start_line=node.lineno,
                    end_line=node.end_lineno or node.lineno,
                )
            )
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()

    FunctionCollector().visit(tree)
    records.sort(key=lambda item: (item.start_line, item.qualified_name))
    return tuple(records)


def _run_git_command(repository: Path, arguments: Sequence[str]) -> bytes:
    """在指定仓库执行只读 git 命令，失败时保留可诊断错误。"""

    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        command = "git " + " ".join(arguments)
        raise CrapMeasurementError(
            f"git 命令失败（{command}）：{detail or f'退出码 {completed.returncode}'}"
        )
    return completed.stdout


def _resolve_repository_root(repository: str | os.PathLike[str]) -> Path:
    """解析 git 仓库根目录，拒绝在普通目录中猜测基线。"""

    candidate = Path(repository).expanduser().resolve()
    if not candidate.exists():
        raise CrapMeasurementError(f"git 仓库目录不存在：{candidate}")
    output = _run_git_command(candidate, ["rev-parse", "--show-toplevel"])
    root = Path(output.decode("utf-8", errors="strict").strip()).resolve()
    if not root.is_dir():
        raise CrapMeasurementError(f"git 仓库根目录无效：{root}")
    return root


def _resolve_git_baseline(repository: Path, baseline: str | None) -> tuple[str, str]:
    """解析明确基线并返回请求值与不可变 commit；默认使用 origin/harness。"""

    requested = baseline or "origin/harness"
    try:
        resolved = _run_git_command(
            repository, ["rev-parse", "--verify", f"{requested}^{{commit}}"]
        ).decode("ascii", errors="strict").strip()
    except (CrapMeasurementError, UnicodeDecodeError) as exc:
        if baseline is None:
            raise CrapMeasurementError(
                "无法解析默认基线 origin/harness，请显式传入 --baseline <commit-or-ref>"
            ) from exc
        raise CrapMeasurementError(f"无法解析显式基线：{requested}") from exc
    if not re.fullmatch(r"[0-9a-fA-F]{40}", resolved):
        raise CrapMeasurementError(f"git 基线不是完整 commit：{requested} -> {resolved}")
    return requested, resolved.lower()


def _repository_relative_path(repository: Path, raw_path: str | os.PathLike[str]) -> str:
    """把路径转换为 git 使用的相对 POSIX 路径，并拒绝仓库外文件。"""

    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = repository / candidate
    resolved = candidate.resolve()
    try:
        relative = resolved.relative_to(repository)
    except ValueError as exc:
        raise CrapMeasurementError(f"路径不在 git 仓库内：{raw_path}") from exc
    return relative.as_posix()


def _normalise_git_path_filters(
    repository: Path, path_filters: Iterable[str | os.PathLike[str]]
) -> tuple[str, ...]:
    """规范化可选 diff 路径筛选；空筛选表示整个仓库。"""

    result = []
    for raw_path in path_filters:
        result.append(_repository_relative_path(repository, raw_path))
    return tuple(dict.fromkeys(result))


def _read_git_status_records(
    repository: Path,
    baseline_commit: str,
    path_filters: Sequence[str],
) -> list[ChangedFileDiff]:
    """读取 git diff 的文件状态，保留 rename 的旧路径和相似度。"""

    arguments = [
        "diff",
        "--no-ext-diff",
        "--find-renames=50%",
        "--find-copies=50%",
        "--find-copies-harder",
        "--name-status",
        "-z",
        baseline_commit,
        "--",
        *path_filters,
    ]
    raw_tokens = _run_git_command(repository, arguments).decode(
        "utf-8", errors="surrogateescape"
    ).split("\0")
    records: list[ChangedFileDiff] = []
    cursor = 0
    while cursor < len(raw_tokens) and raw_tokens[cursor]:
        status_token = raw_tokens[cursor]
        cursor += 1
        status = status_token[:1]
        if status in {"R", "C"}:
            if cursor + 1 >= len(raw_tokens):
                raise CrapMeasurementError("git rename/copy 状态缺少路径")
            old_path = raw_tokens[cursor]
            path = raw_tokens[cursor + 1]
            cursor += 2
        else:
            if cursor >= len(raw_tokens):
                raise CrapMeasurementError("git diff 状态缺少路径")
            path = raw_tokens[cursor]
            old_path = path if status != "A" else None
            cursor += 1
        similarity_text = status_token[1:]
        similarity = int(similarity_text) if similarity_text.isdigit() else None
        if not (path.endswith(".py") or (old_path and old_path.endswith(".py"))):
            continue
        records.append(
            ChangedFileDiff(
                status=status,
                path=path,
                old_path=old_path,
                similarity=similarity,
                added_lines=(),
                deleted_lines=(),
            )
        )
    return records


def _patch_path(raw_path: str, prefix: str) -> str | None:
    """从 git patch 的 ``---``/``+++`` 行取出相对路径。"""

    path = raw_path.split("\t", 1)[0]
    if path == "/dev/null":
        return None
    if path.startswith(prefix):
        return path[len(prefix) :]
    return path


def _create_git_patch_line_change_index(
    file_records: Sequence[ChangedFileDiff],
) -> dict[tuple[str | None, str], tuple[set[int], set[int]]]:
    """为每个 git 文件状态创建新增行和删除行集合。"""

    return {
        (record.old_path, record.path): (set(), set()) for record in file_records
    }


def _find_git_patch_file_key_matches(
    changes: Mapping[tuple[str | None, str], tuple[set[int], set[int]]],
    pending_old_path: str | None,
    pending_new_path: str | None,
) -> list[tuple[str | None, str]]:
    """查找与 patch 两侧路径相符的文件状态候选。"""

    return [
        key
        for key in changes
        if key[1] == pending_new_path
        or (pending_old_path is not None and key[0] == pending_old_path)
    ]


def _select_git_patch_file_key(
    changes: Mapping[tuple[str | None, str], tuple[set[int], set[int]]],
    pending_old_path: str | None,
    pending_new_path: str | None,
) -> tuple[str | None, str] | None:
    """根据 patch 两侧路径选择唯一文件状态，歧义时拒绝猜测。"""

    candidate = (pending_old_path, pending_new_path)
    if candidate in changes:
        return candidate
    matches = _find_git_patch_file_key_matches(
        changes, pending_old_path, pending_new_path
    )
    if len(matches) > 1:
        raise CrapMeasurementError(
            f"git patch 文件映射有歧义：{pending_old_path!r} -> {pending_new_path!r}"
        )
    return matches[0] if matches else None


def _parse_git_patch_hunk_header(raw_line: str) -> tuple[int, int] | None:
    """解析 hunk 头并返回旧文件、新文件的起始行号。"""

    hunk = _GIT_HUNK_PATTERN.match(raw_line)
    if hunk is None:
        return None
    return int(hunk.group(1)), int(hunk.group(3))


def _start_git_patch_hunk(
    state: _GitPatchParseState, old_line: int, new_line: int
) -> None:
    """记录当前文件 hunk 起始行号，并确认文件状态已经映射。"""

    if state.current_file_key is None:
        raise CrapMeasurementError("git patch 的 hunk 无法映射到文件状态")
    state.old_line = old_line
    state.new_line = new_line
    state.in_hunk = True


def _record_git_patch_content_line(
    raw_line: str,
    state: _GitPatchParseState,
    changes: Mapping[tuple[str | None, str], tuple[set[int], set[int]]],
) -> None:
    """把一行 hunk 内容计入当前文件并推进两侧行号。"""

    current_key = state.current_file_key
    if current_key is None:
        return
    added, deleted = changes[current_key]
    if raw_line.startswith("+"):
        added.add(state.new_line)
        state.new_line += 1
    elif raw_line.startswith("-"):
        deleted.add(state.old_line)
        state.old_line += 1
    elif raw_line.startswith("\\"):
        return
    else:
        state.old_line += 1
        state.new_line += 1


def _parse_git_patch_line_changes(
    patch: str, file_records: Sequence[ChangedFileDiff]
) -> dict[tuple[str | None, str], tuple[set[int], set[int]]]:
    """解析 zero-context git patch，得到每个文件的新增和删除行号。"""

    changes = _create_git_patch_line_change_index(file_records)
    state = _GitPatchParseState()
    for raw_line in patch.splitlines():
        if raw_line.startswith("diff --git "):
            state = _GitPatchParseState()
            continue
        if raw_line.startswith("--- "):
            state.pending_old_path = _patch_path(raw_line[4:], "a/")
            continue
        if raw_line.startswith("+++ "):
            state.pending_new_path = _patch_path(raw_line[4:], "b/")
            state.current_file_key = _select_git_patch_file_key(
                changes,
                state.pending_old_path,
                state.pending_new_path,
            )
            continue
        hunk_lines = _parse_git_patch_hunk_header(raw_line)
        if hunk_lines is not None:
            _start_git_patch_hunk(state, *hunk_lines)
            continue
        if not state.in_hunk or state.current_file_key is None or not raw_line:
            continue
        _record_git_patch_content_line(raw_line, state, changes)
    return changes


def _read_git_file_diffs(
    repository: Path,
    baseline_commit: str,
    path_filters: Sequence[str],
    *,
    include_untracked: bool = True,
) -> tuple[ChangedFileDiff, ...]:
    """读取 Python 文件状态及零上下文逐行变更证据。"""

    records = _read_git_status_records(repository, baseline_commit, path_filters)
    known_paths = {(record.old_path, record.path) for record in records}
    # 工作树中的 untracked Python 文件也是当前 diff 的新增文件；调用者可以用
    # path_filters 限定范围，避免把共享工作区的无关目录纳入门禁。
    if path_filters or include_untracked:
        untracked_arguments = [
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            *path_filters,
        ]
        untracked = _run_git_command(repository, untracked_arguments).decode(
            "utf-8", errors="surrogateescape"
        )
        for path in (item for item in untracked.split("\0") if item):
            if not path.endswith(".py") or (None, path) in known_paths:
                continue
            records.append(
                ChangedFileDiff(
                    status="A",
                    path=path,
                    old_path=None,
                    similarity=None,
                    added_lines=(),
                    deleted_lines=(),
                )
            )
    if not records:
        return ()
    patch_arguments = [
        "diff",
        "--no-ext-diff",
        "--find-renames=50%",
        "--find-copies=50%",
        "--find-copies-harder",
        "--unified=0",
        "--no-color",
        baseline_commit,
        "--",
        *path_filters,
    ]
    patch = _run_git_command(repository, patch_arguments).decode(
        "utf-8", errors="surrogateescape"
    )
    changes = _parse_git_patch_line_changes(patch, records)
    return tuple(
        ChangedFileDiff(
            status=record.status,
            path=record.path,
            old_path=record.old_path,
            similarity=record.similarity,
            added_lines=tuple(sorted(changes[(record.old_path, record.path)][0])),
            deleted_lines=tuple(sorted(changes[(record.old_path, record.path)][1])),
        )
        for record in records
    )


def _read_current_source(repository: Path, relative_path: str) -> tuple[Path, str]:
    """读取工作树中的当前源码，返回绝对路径和文本。"""

    path = (repository / relative_path).resolve()
    if not path.is_file():
        raise CrapMeasurementError(f"当前源码不存在：{relative_path}")
    return path, _read_source_text(path)


def _read_baseline_source(
    repository: Path, baseline_commit: str, relative_path: str
) -> str:
    """从基线 commit 读取旧源码；删除文件也可因此参与函数审计。"""

    try:
        raw = _run_git_command(
            repository, ["show", f"{baseline_commit}:{relative_path}"]
        )
    except CrapMeasurementError as exc:
        raise CrapMeasurementError(
            f"无法读取基线源码：{baseline_commit}:{relative_path}"
        ) from exc
    return raw.decode("utf-8", errors="strict")


def _find_containing_function_indexes(
    records: Sequence[_FunctionRecord], lines: Iterable[int]
) -> dict[int, set[str]]:
    """将变更行映射到所有包含它的函数，确保嵌套函数不漏算。"""

    result, _ = _map_change_lines_to_functions(records, lines)
    return result


def _map_change_lines_to_functions(
    records: Sequence[_FunctionRecord], lines: Iterable[int]
) -> tuple[dict[int, set[str]], dict[int, set[int]]]:
    """一次遍历把变更行映射为函数类别和函数内行号。"""

    kinds: dict[int, set[str]] = defaultdict(set)
    mapped_lines: dict[int, set[int]] = defaultdict(set)
    for line in lines:
        for index, record in enumerate(records):
            if _function_change_start_line(record) <= line <= record.end_line:
                kinds[index].add(_change_kind_for_line(record, line))
                mapped_lines[index].add(line)
    return kinds, mapped_lines


def _change_kind_for_line(record: _FunctionRecord, line: int) -> str:
    """按变更行位置标注装饰器、签名或函数正文。"""

    if line < record.start_line:
        return "decorator"
    if line < _function_body_start_line(record):
        return "signature"
    return "body"


def _pair_function_records(
    old_records: Sequence[_FunctionRecord],
    current_records: Sequence[_FunctionRecord],
) -> dict[int, int]:
    """按限定名和出现顺序配对旧新函数，支持同名方法和重复定义。"""

    old_by_name: dict[str, list[int]] = defaultdict(list)
    current_by_name: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(old_records):
        old_by_name[record.qualified_name].append(index)
    for index, record in enumerate(current_records):
        current_by_name[record.qualified_name].append(index)
    pairs: dict[int, int] = {}
    for name in sorted(set(old_by_name) & set(current_by_name)):
        old_indexes = sorted(old_by_name[name], key=lambda index: old_records[index].start_line)
        current_indexes = sorted(
            current_by_name[name], key=lambda index: current_records[index].start_line
        )
        pairs.update(zip(old_indexes, current_indexes))
    return pairs


def _changed_function_from_records(
    *,
    status: str,
    path: str,
    old_path: str | None,
    current: _FunctionRecord | None,
    old: _FunctionRecord | None,
    change_kinds: Iterable[str],
    added_lines: Iterable[int],
    deleted_lines: Iterable[int],
) -> ChangedFunction:
    """把 AST 函数和 diff 证据转换为 manifest 函数记录。"""

    qualified_name = _select_changed_function_qualified_name(current, old)
    ordered_kinds = _order_changed_function_kinds(change_kinds)
    return ChangedFunction(
        status=status,
        path=path,
        old_path=old_path,
        qualified_name=qualified_name,
        current_start_line=current.start_line if current else None,
        current_end_line=current.end_line if current else None,
        old_start_line=old.start_line if old else None,
        old_end_line=old.end_line if old else None,
        change_kinds=ordered_kinds,
        added_lines=tuple(sorted(set(added_lines))),
        deleted_lines=tuple(sorted(set(deleted_lines))),
    )


def _select_changed_function_qualified_name(
    current: _FunctionRecord | None,
    old: _FunctionRecord | None,
) -> str:
    """选择当前或旧函数限定名，并拒绝两侧都不存在的记录。"""

    if current is not None:
        return current.qualified_name
    if old is not None:
        return old.qualified_name
    raise CoverageMappingError("变更函数必须至少存在于一侧源码")


def _order_changed_function_kinds(change_kinds: Iterable[str]) -> tuple[str, ...]:
    """按协议顺序整理装饰器、签名、正文和文件新增标记。"""

    kinds = set(change_kinds)
    return tuple(
        kind
        for kind in ("file_added", "decorator", "signature", "body")
        if kind in kinds
    )


def _read_changed_file_function_records(
    repository: Path,
    baseline_commit: str,
    file_diff: ChangedFileDiff,
) -> tuple[tuple[_FunctionRecord, ...], tuple[_FunctionRecord, ...]]:
    """读取一个变更文件两侧源码并收集 AST 函数记录。"""

    current_records: tuple[_FunctionRecord, ...] = ()
    old_records: tuple[_FunctionRecord, ...] = ()
    if file_diff.status != "D" and file_diff.path.endswith(".py"):
        current_path, current_text = _read_current_source(repository, file_diff.path)
        current_records = _collect_function_records(_parse_source(current_path, current_text))
    if file_diff.old_path and file_diff.old_path.endswith(".py"):
        old_path = repository / file_diff.old_path
        old_text = _read_baseline_source(repository, baseline_commit, file_diff.old_path)
        old_records = _collect_function_records(_parse_source(old_path, old_text))
    return current_records, old_records


def _find_old_function_index_for_current(
    pairs: Mapping[int, int], current_index: int
) -> int | None:
    """在旧函数配对表中查找当前函数对应的旧索引。"""

    return next(
        (old_index for old_index, paired_index in pairs.items() if paired_index == current_index),
        None,
    )


def _build_added_file_changed_functions(
    file_diff: ChangedFileDiff,
    current_records: Sequence[_FunctionRecord],
    old_records: Sequence[_FunctionRecord],
) -> list[ChangedFunction]:
    """为新增或复制文件生成全部当前函数的 cohort 记录。"""

    mapped, mapped_lines = _map_change_lines_to_functions(
        current_records, file_diff.added_lines
    )
    pairs = _pair_function_records(old_records, current_records)
    changed_functions: list[ChangedFunction] = []
    for index, record in enumerate(current_records):
        old_index = _find_old_function_index_for_current(pairs, index)
        old_record = old_records[old_index] if old_index is not None else None
        changed_functions.append(
            _changed_function_from_records(
                status="added",
                path=file_diff.path,
                old_path=file_diff.old_path,
                current=record,
                old=old_record,
                change_kinds=mapped.get(index, set()) | {"file_added"},
                added_lines=mapped_lines.get(index, set()),
                deleted_lines=(),
            )
        )
    return changed_functions


def _build_deleted_file_changed_functions(
    file_diff: ChangedFileDiff,
    old_records: Sequence[_FunctionRecord],
) -> list[ChangedFunction]:
    """为删除文件生成旧函数记录，并保留删除行证据。"""

    old_mapping, old_mapped_lines = _map_change_lines_to_functions(
        old_records, file_diff.deleted_lines
    )
    deleted_functions: list[ChangedFunction] = []
    for index, record in enumerate(old_records):
        deleted_functions.append(
            _changed_function_from_records(
                status="deleted",
                path=file_diff.path,
                old_path=None,
                current=None,
                old=record,
                change_kinds=old_mapping.get(index, set()) or {"body"},
                added_lines=(),
                deleted_lines=old_mapped_lines.get(index, set()),
            )
        )
    return deleted_functions


def _build_modified_file_changed_functions(
    file_diff: ChangedFileDiff,
    old_records: Sequence[_FunctionRecord],
    current_records: Sequence[_FunctionRecord],
) -> tuple[list[ChangedFunction], list[ChangedFunction]]:
    """按新增、删除行和限定名配对生成修改文件的两侧函数记录。"""

    current_mapping, current_mapped_lines = _map_change_lines_to_functions(
        current_records, file_diff.added_lines
    )
    old_mapping, old_mapped_lines = _map_change_lines_to_functions(
        old_records, file_diff.deleted_lines
    )
    pairs = _pair_function_records(old_records, current_records)
    selected_current = set(current_mapping)
    selected_old = set(old_mapping)
    selected_current.update(pairs[index] for index in selected_old if index in pairs)

    changed_functions: list[ChangedFunction] = []
    for current_index in sorted(selected_current):
        record = current_records[current_index]
        old_index = _find_old_function_index_for_current(pairs, current_index)
        old_record = old_records[old_index] if old_index is not None else None
        kinds = set(current_mapping.get(current_index, set()))
        added_lines = current_mapped_lines.get(current_index, set())
        deleted_lines: Iterable[int] = ()
        if old_index is not None:
            kinds.update(old_mapping.get(old_index, set()))
            deleted_lines = old_mapped_lines.get(old_index, set())
        changed_functions.append(
            _changed_function_from_records(
                status="modified" if old_record else "added",
                path=file_diff.path,
                old_path=(
                    file_diff.old_path
                    if file_diff.old_path != file_diff.path
                    else None
                ),
                current=record,
                old=old_record,
                change_kinds=kinds,
                added_lines=added_lines,
                deleted_lines=deleted_lines,
            )
        )

    deleted_functions: list[ChangedFunction] = []
    for old_index in sorted(selected_old):
        if old_index in pairs:
            continue
        record = old_records[old_index]
        deleted_functions.append(
            _changed_function_from_records(
                status="deleted",
                path=file_diff.path,
                old_path=file_diff.old_path,
                current=None,
                old=record,
                change_kinds=old_mapping.get(old_index, {"body"}),
                added_lines=(),
                deleted_lines=old_mapped_lines.get(old_index, set()),
            )
        )
    return changed_functions, deleted_functions


def generate_changed_function_cohort_manifest(
    repository: str | os.PathLike[str] = ".",
    *,
    baseline: str | None = None,
    path_filters: Iterable[str | os.PathLike[str]] = (),
    include_untracked: bool = True,
) -> ChangedFunctionCohort:
    """从明确 git 基线生成当前新增/修改函数和删除函数清单。

    变更行通过 zero-context diff 映射到当前及基线 AST；新增文件的所有函数都
    纳入 cohort，删除函数只写入 ``deleted_functions``，不会被当前覆盖率测量。
    函数匹配包含装饰器、签名、正文、嵌套作用域、同名定义和文件重命名证据。
    """

    repository_root = _resolve_repository_root(repository)
    baseline_spec, baseline_commit = _resolve_git_baseline(repository_root, baseline)
    filters = _normalise_git_path_filters(repository_root, path_filters)
    file_diffs = _read_git_file_diffs(
        repository_root,
        baseline_commit,
        filters,
        include_untracked=include_untracked,
    )
    current_functions: list[ChangedFunction] = []
    deleted_functions: list[ChangedFunction] = []
    for file_diff in file_diffs:
        current_records, old_records = _read_changed_file_function_records(
            repository_root, baseline_commit, file_diff
        )
        if file_diff.status in {"A", "C"}:
            current_functions.extend(
                _build_added_file_changed_functions(
                    file_diff, current_records, old_records
                )
            )
            continue
        if file_diff.status == "D":
            deleted_functions.extend(
                _build_deleted_file_changed_functions(file_diff, old_records)
            )
            continue
        changed, deleted = _build_modified_file_changed_functions(
            file_diff, old_records, current_records
        )
        current_functions.extend(changed)
        deleted_functions.extend(deleted)
    current_functions.sort(key=lambda item: (item.path, item.current_start_line or 0, item.qualified_name))
    deleted_functions.sort(key=lambda item: (item.path, item.old_start_line or 0, item.qualified_name))
    return ChangedFunctionCohort(
        baseline_spec=baseline_spec,
        baseline_commit=baseline_commit,
        repository=str(repository_root),
        files=tuple(file_diffs),
        functions=tuple(current_functions),
        deleted_functions=tuple(deleted_functions),
    )


def generate_changed_function_cohort(
    repository: str | os.PathLike[str] = ".",
    *,
    baseline: str | None = None,
    path_filters: Iterable[str | os.PathLike[str]] = (),
    include_untracked: bool = True,
) -> ChangedFunctionCohort:
    """生成当前 git diff 函数 cohort；这是 manifest 生成函数的短入口。"""

    return generate_changed_function_cohort_manifest(
        repository,
        baseline=baseline,
        path_filters=path_filters,
        include_untracked=include_untracked,
    )


def _resolve_function_for_line(
    records: Sequence[_FunctionRecord], line: int
) -> _FunctionRecord | None:
    """把一行映射到最内层函数，重叠同层函数时拒绝猜测。"""

    containing = [
        record
        for record in records
        if record.start_line <= line <= record.end_line
    ]
    if not containing:
        return None
    deepest_line = max(record.start_line for record in containing)
    deepest = [record for record in containing if record.start_line == deepest_line]
    if len(deepest) != 1:
        names = ", ".join(record.qualified_name for record in deepest)
        raise CoverageMappingError(f"第 {line} 行的函数映射有歧义：{names}")
    return deepest[0]


def _build_parser(path: Path, text: str) -> PythonParser:
    """使用 Coverage.py 的排除规则获取可执行行和静态弧。

    Coverage 的报告阶段会排除 ``# pragma: no cover`` 等分支。如果这里使用
    没有排除规则的裸解析器，静态弧就会比 SQLite/JSON 报告多，进而把合法的
    排除分支误报为覆盖率损坏。
    """

    parser = PythonParser(
        text=text,
        filename=str(path),
        exclude=_coverage_option_regex("report:exclude_lines"),
    )
    try:
        parser.parse_source()
    except Exception as exc:  # Coverage.py 用多个异常类型表示语法错误。
        raise CrapMeasurementError(f"Coverage.py 无法解析源码：{path}: {exc}") from exc
    return parser


def _static_branch_lines(parser: PythonParser) -> set[int]:
    """从 Coverage.py 已应用排除规则的出口计数识别分支行。"""

    return {
        line
        for line, exit_count in parser.exit_counts().items()
        if line > 0 and exit_count > 1
    }


def _coverage_option_regex(option_name: str) -> str | None:
    """把 Coverage.py 当前配置的一组正则合并成解析器可用的表达式。"""

    values = coverage.Coverage().get_option(option_name)
    if not isinstance(values, list) or not values:
        return None
    if not all(isinstance(value, str) and value for value in values):
        raise CrapMeasurementError(f"Coverage.py {option_name} 配置无效")
    return "|".join(f"(?:{value})" for value in values)


def _partial_branch_lines(text: str) -> set[int]:
    """读取 Coverage.py 明确声明为无需另一出口的分支行。"""

    pattern = _coverage_option_regex("report:partial_branches")
    if pattern is None:
        return set()
    matcher = re.compile(pattern)
    return {
        line_number
        for line_number, line in enumerate(text.splitlines(), start=1)
        if matcher.search(line)
    }


def _parse_arc(value: Any, label: str) -> Arc:
    """校验 JSON 中的覆盖率弧。"""

    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise CoverageMappingError(f"{label} 不是 [from_line, to_line] 弧")
    if not all(isinstance(item, int) for item in value):
        raise CoverageMappingError(f"{label} 的行号必须是整数")
    return int(value[0]), int(value[1])


def _group_arcs(arcs: Iterable[Arc]) -> dict[int, set[Arc]]:
    """按来源行去重覆盖率弧。"""

    grouped: dict[int, set[Arc]] = defaultdict(set)
    for arc in arcs:
        grouped[arc[0]].add(arc)
    return grouped


def _json_source_hash(entry: Mapping[str, Any]) -> str | None:
    """读取 JSON 文件条目中的可选源码哈希。"""

    value = entry.get("source_hash")
    if value is None and isinstance(entry.get("meta"), Mapping):
        value = entry["meta"].get("source_hash")
    if value is None:
        return None
    if not isinstance(value, str) or len(value) != 64:
        raise CoverageMappingError("JSON source_hash 必须是 64 位十六进制字符串")
    try:
        int(value, 16)
    except ValueError as exc:
        raise CoverageMappingError("JSON source_hash 不是十六进制哈希") from exc
    return value.lower()


def _resolve_json_source_path(raw_path: str, coverage_dir: Path) -> Path:
    """按 JSON 所在目录再按当前工作目录解析源码路径。"""

    resolved_path = Path(_normalise_path(raw_path, coverage_dir))
    if resolved_path.exists():
        return resolved_path
    return Path(_normalise_path(raw_path))


def _parse_json_executed_lines(
    raw_path: str, entry: Mapping[str, Any]
) -> frozenset[int]:
    """读取并校验 JSON 中的已执行语句行。"""

    value = entry.get("executed_lines", [])
    if not isinstance(value, list) or not all(isinstance(line, int) for line in value):
        raise CoverageMappingError(f"JSON executed_lines 无效：{raw_path}")
    return frozenset(value)


def _parse_json_branch_arcs(
    raw_path: str, entry: Mapping[str, Any]
) -> tuple[int, set[Arc], set[Arc]]:
    """读取分支摘要和两类弧，并拒绝缺失的分支数组。"""

    summary = entry.get("summary", {})
    if not isinstance(summary, Mapping):
        raise CoverageMappingError(f"JSON summary 无效：{raw_path}")
    declared = int(summary.get("num_branches", 0) or 0)
    executed_raw = entry.get("executed_branches")
    missing_raw = entry.get("missing_branches")
    if declared and not isinstance(executed_raw, list):
        raise CoverageMappingError(f"JSON 缺少 executed_branches：{raw_path}")
    if declared and not isinstance(missing_raw, list):
        raise CoverageMappingError(f"JSON 缺少 missing_branches：{raw_path}")
    executed = {
        _parse_arc(value, f"{raw_path}.executed_branches")
        for value in (executed_raw or [])
    }
    missing = {
        _parse_arc(value, f"{raw_path}.missing_branches")
        for value in (missing_raw or [])
    }
    if executed & missing:
        raise CoverageMappingError(f"JSON 分支弧同时出现在已覆盖和缺失集合：{raw_path}")
    return declared, executed, missing


def _build_json_branch_stats(
    raw_path: str,
    parser: PythonParser,
    text: str,
    declared: int,
    executed: set[Arc],
    missing: set[Arc],
) -> tuple[dict[int, int], dict[int, int]]:
    """聚合 JSON 分支弧，并补回 ``no branch`` 被视为覆盖的出口。

    Coverage JSON 不会列出 ``# pragma: no branch`` 所豁免的缺失弧，但摘要仍把
    这些出口计入已覆盖分支。这里仅对源码中明确带该标记的分支补回静态弧；普通
    分支只要缺弧仍然 fail closed。
    """

    static_lines = _static_branch_lines(parser)
    partial_lines = _partial_branch_lines(text) & static_lines
    static_arcs = {
        arc
        for arc in parser.arcs()
        if arc[0] in static_lines
    }
    reported_arcs = executed | missing
    suppressed_arcs = {
        arc
        for arc in static_arcs - reported_arcs
        if arc[0] in partial_lines
    }
    executed = executed | suppressed_arcs
    all_arcs = executed | missing
    unknown_lines = {arc[0] for arc in all_arcs} - static_lines
    if unknown_lines:
        raise CoverageMappingError(
            f"JSON 分支弧无法映射到 Coverage.py 静态分支行：{raw_path}: "
            f"{sorted(unknown_lines)}"
        )
    all_grouped = _group_arcs(all_arcs)
    executed_grouped = _group_arcs(executed)
    totals = {line: len(arcs) for line, arcs in all_grouped.items()}
    covered = {
        line: len(executed_grouped.get(line, set())) for line in all_grouped
    }
    actual = sum(totals.values())
    if declared and declared != actual:
        raise CoverageMappingError(
            f"JSON 分支数量与弧数量不一致：{raw_path}: "
            f"summary={declared}, arcs={actual}"
        )
    if not declared and static_lines:
        raise CoverageMappingError(
            f"源码包含分支但 JSON 没有分支覆盖率数据：{raw_path}"
        )
    return totals, covered


def _parse_json_file_record(
    raw_path: str, entry: Mapping[str, Any], coverage_dir: Path
) -> CoverageFileRecord:
    """解析 Coverage.py JSON 的单文件记录并验证分支弧完整性。"""

    resolved_path = _resolve_json_source_path(raw_path, coverage_dir)
    text = _read_source_text(resolved_path)
    parser = _build_parser(resolved_path, text)
    statements = frozenset(int(line) for line in parser.statements)
    executed_lines = _parse_json_executed_lines(raw_path, entry)
    declared, executed, missing = _parse_json_branch_arcs(raw_path, entry)
    branch_totals, branch_covered = _build_json_branch_stats(
        raw_path, parser, text, declared, executed, missing
    )
    return CoverageFileRecord(
        raw_path=raw_path,
        statements=statements,
        executed_lines=executed_lines,
        branch_totals=branch_totals,
        branch_covered=branch_covered,
        source_hash=_json_source_hash(entry),
    )


def _load_json_snapshot(path: Path) -> _CoverageSnapshot:
    """加载 Coverage.py JSON，并拒绝缺少可映射弧的分支报告。"""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CrapMeasurementError(f"无法读取 Coverage.py JSON：{path}: {exc}") from exc
    files = payload.get("files") if isinstance(payload, Mapping) else None
    if not isinstance(files, Mapping) or not files:
        raise CoverageMappingError("Coverage.py JSON 缺少非空 files 对象")
    records = []
    for raw_path, entry in files.items():
        if not isinstance(raw_path, str) or not isinstance(entry, Mapping):
            raise CoverageMappingError("Coverage.py JSON files 条目无效")
        records.append(_parse_json_file_record(raw_path, entry, path.parent))
    return _CoverageSnapshot(tuple(records), "json")


def _load_sqlite_file_record(
    coverage_instance: coverage.Coverage,
    data: coverage.CoverageData,
    raw_path: str,
) -> CoverageFileRecord:
    """读取 SQLite 中一个文件，并把 Coverage.py Analysis 转成内部记录。"""

    source_path = Path(_normalise_path(raw_path))
    if not source_path.exists():
        raise CoverageMappingError(f"覆盖率引用的源码不存在：{raw_path}")
    text = _read_source_text(source_path)
    parser = _build_parser(source_path, text)
    try:
        analysis = coverage_instance._analyze(raw_path)
    except Exception:
        try:
            analysis = coverage_instance._analyze(str(source_path))
        except Exception as exc:
            raise CoverageMappingError(
                f"无法分析 Coverage.py 文件：{raw_path}: {exc}"
            ) from exc
    branch_totals, branch_covered = _sqlite_branch_stats(data, analysis, parser, raw_path)
    return CoverageFileRecord(
        raw_path=raw_path,
        statements=frozenset(analysis.statements),
        executed_lines=frozenset(analysis.executed),
        branch_totals=branch_totals,
        branch_covered=branch_covered,
        source_hash=None,
    )


def _sqlite_branch_stats(
    data: coverage.CoverageData,
    analysis: Any,
    parser: PythonParser,
    raw_path: str,
) -> tuple[dict[int, int], dict[int, int]]:
    """读取 SQLite 的真实分支统计，并拒绝无弧或不完整的输入。"""

    static_lines = _static_branch_lines(parser)
    if not data.has_arcs():
        if static_lines:
            raise CoverageMappingError(
                f"源码包含分支但 SQLite 没有 arcs 数据：{raw_path}"
            )
        return {}, {}
    stats = analysis.branch_stats()
    totals = {line: values[0] for line, values in stats.items()}
    covered = {line: values[1] for line, values in stats.items()}
    missing_lines = static_lines - set(totals)
    if missing_lines:
        raise CoverageMappingError(
            f"Coverage.py 分支统计未覆盖静态分支行：{raw_path}: "
            f"{sorted(missing_lines)}"
        )
    return totals, covered


def _load_sqlite_snapshot(path: Path) -> _CoverageSnapshot:
    """加载 Coverage.py SQLite，并使用其 Analysis 获取真实分支统计。"""

    coverage_instance = coverage.Coverage(data_file=str(path), branch=True)
    try:
        coverage_instance.load()
    except Exception as exc:
        raise CrapMeasurementError(f"无法读取 Coverage.py SQLite：{path}: {exc}") from exc
    data = coverage_instance.get_data()
    records = [
        _load_sqlite_file_record(coverage_instance, data, raw_path)
        for raw_path in sorted(data.measured_files())
    ]
    if not records:
        raise CoverageMappingError("Coverage.py SQLite 没有 measured files")
    return _CoverageSnapshot(tuple(records), "sqlite")


def _load_coverage_snapshot(path: Path, format_hint: str = "auto") -> _CoverageSnapshot:
    """根据显式或自动格式加载覆盖率快照。"""

    if format_hint not in {"auto", "sqlite", "json"}:
        raise CrapMeasurementError(f"不支持的 coverage 格式：{format_hint}")
    selected = format_hint
    if selected == "auto":
        selected = "json" if path.suffix.lower() == ".json" else "sqlite"
    if selected == "json":
        return _load_json_snapshot(path)
    return _load_sqlite_snapshot(path)


def _load_hash_manifest(path: Path | None) -> dict[str, str]:
    """读取采集覆盖率时保存的路径到 SHA-256 清单。"""

    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CrapMeasurementError(f"无法读取 source hash manifest：{path}: {exc}") from exc
    if isinstance(payload, Mapping) and isinstance(payload.get("files"), Mapping):
        payload = payload["files"]
    if not isinstance(payload, Mapping):
        raise CrapMeasurementError("source hash manifest 必须是路径到哈希的 JSON 对象")
    result: dict[str, str] = {}
    for raw_path, value in payload.items():
        value = _parse_manifest_hash(raw_path, value)
        result[_normalise_path(raw_path, path.parent)] = value
    return result


def _parse_manifest_hash(raw_path: Any, value: Any) -> str:
    """校验 source hash manifest 的单条路径和哈希。"""

    if isinstance(value, Mapping):
        value = value.get("sha256") or value.get("source_hash")
    if not isinstance(raw_path, str) or not isinstance(value, str):
        raise CrapMeasurementError("source hash manifest 条目无效")
    if len(value) != 64:
        raise CrapMeasurementError(f"source hash manifest 哈希无效：{raw_path}")
    try:
        int(value, 16)
    except ValueError as exc:
        raise CrapMeasurementError(f"source hash manifest 哈希无效：{raw_path}") from exc
    return value.lower()


def _match_coverage_record(
    source_path: Path, records: Sequence[CoverageFileRecord], coverage_path: Path
) -> CoverageFileRecord:
    """将源码路径唯一匹配到覆盖率文件，拒绝模糊后缀匹配。"""

    source_key = _normalise_path(source_path)
    candidates = []
    for record in records:
        raw_key = _normalise_path(record.raw_path, coverage_path.parent)
        cwd_key = _normalise_path(record.raw_path)
        if source_key in {raw_key, cwd_key}:
            candidates.append(record)
    if len(candidates) != 1:
        if not candidates:
            raise CoverageMappingError(f"源码没有对应的覆盖率记录：{source_path}")
        raise CoverageMappingError(
            f"源码对应多个覆盖率记录，不能猜测：{source_path}: "
            f"{[record.raw_path for record in candidates]}"
        )
    return candidates[0]


def _match_hash(source_path: Path, manifest: Mapping[str, str]) -> str | None:
    """按规范路径查找覆盖率采集时的源码哈希。"""

    return manifest.get(_normalise_path(source_path))


def _validate_source_hash_metadata(
    source_path: Path,
    coverage_hash: str | None,
    manifest_hash: str | None,
) -> None:
    """核对覆盖率记录和外部清单中的双重哈希元数据。"""

    if coverage_hash and manifest_hash and manifest_hash != coverage_hash:
        raise CrapMeasurementError(f"覆盖率源哈希元数据互相矛盾：{source_path}")


def _validate_source_hash_manifest_entry(
    source_path: Path,
    manifest: Mapping[str, str],
    manifest_hash: str | None,
    coverage_hash: str | None,
) -> None:
    """确认哈希清单没有漏掉缺少内嵌哈希的源码。"""

    if manifest and manifest_hash is None and coverage_hash is None:
        raise CrapMeasurementError(f"source hash manifest 缺少源码：{source_path}")


def _validate_current_source_hash(
    source_path: Path, expected_hash: str | None
) -> None:
    """将覆盖率采集时的哈希与当前源码字节核对。"""

    if expected_hash is not None and expected_hash != _source_hash(source_path):
        raise CrapMeasurementError(f"覆盖率源哈希已过期：{source_path}")


def _validate_source_hash(
    source_path: Path, record: CoverageFileRecord, manifest: Mapping[str, str]
) -> None:
    """校验可用的源哈希，任何明确不一致都拒绝继续。"""

    manifest_hash = _match_hash(source_path, manifest)
    _validate_source_hash_metadata(source_path, record.source_hash, manifest_hash)
    _validate_source_hash_manifest_entry(
        source_path, manifest, manifest_hash, record.source_hash
    )
    _validate_current_source_hash(source_path, record.source_hash or manifest_hash)


def _radon_complexity(record: _FunctionRecord, path: Path) -> int:
    """对单个 AST 函数调用 Radon，避免依赖文件级复杂度汇总。"""

    wrapper = ast.Module(body=[record.node], type_ignores=[])
    try:
        results = cc_visit_ast(wrapper)
    except Exception as exc:
        raise CrapMeasurementError(
            f"Radon 无法计算函数复杂度：{path}:{record.start_line}: "
            f"{record.qualified_name}: {exc}"
        ) from exc
    matches = [
        result
        for result in results
        if getattr(result, "name", None) == record.node.name
        and getattr(result, "lineno", None) == record.start_line
    ]
    if len(matches) != 1:
        raise CoverageMappingError(
            f"Radon 函数映射有歧义：{path}:{record.start_line}: "
            f"{record.qualified_name}"
        )
    return int(matches[0].complexity)


def _function_lines(
    function_records: Sequence[_FunctionRecord],
    function: _FunctionRecord,
    lines: Iterable[int],
) -> set[int]:
    """只保留属于最内层目标函数的覆盖率行。"""

    selected = set()
    for line in lines:
        resolved = _resolve_function_for_line(function_records, line)
        if resolved == function:
            selected.add(line)
    return selected


def _calculate_branch_coverage_metrics(
    source_path: Path,
    function: _FunctionRecord,
    branch_lines: Iterable[int],
    record: CoverageFileRecord,
) -> tuple[float, int, int]:
    """计算目标函数的分支覆盖率并校验分支计数。"""

    total = sum(record.branch_totals[line] for line in branch_lines)
    covered = sum(record.branch_covered.get(line, 0) for line in branch_lines)
    if total <= 0 or covered < 0 or covered > total:
        raise CoverageMappingError(
            f"函数分支覆盖率数据无效：{source_path}:{function.start_line}: "
            f"{function.qualified_name}"
        )
    return covered / total, covered, total


def _calculate_line_coverage_metrics(
    source_path: Path,
    function: _FunctionRecord,
    statements: set[int],
    executed: set[int],
) -> tuple[float, int]:
    """计算无分支函数的可执行语句覆盖率并拒绝空集合。"""

    total = len(statements)
    if total <= 0:
        raise CoverageMappingError(
            f"函数没有可执行语句：{source_path}:{function.start_line}: "
            f"{function.qualified_name}"
        )
    covered = len(executed & statements)
    return covered / total, covered


def _measure_function(
    source_path: Path,
    function_records: Sequence[_FunctionRecord],
    function: _FunctionRecord,
    record: CoverageFileRecord,
) -> FunctionMeasurement:
    """计算一个函数的覆盖率和 CRAP，分支与行覆盖率严格分流。"""

    statements = _function_lines(function_records, function, record.statements)
    executed = _function_lines(function_records, function, record.executed_lines)
    branch_lines = _function_lines(function_records, function, record.branch_totals)
    if not statements and not branch_lines:
        raise CoverageMappingError(
            f"函数没有可映射覆盖率行：{source_path}:{function.start_line}: "
            f"{function.qualified_name}"
        )
    if branch_lines:
        coverage_ratio, covered, total = _calculate_branch_coverage_metrics(
            source_path, function, branch_lines, record
        )
        mode = "branches"
        covered_branches: int | None = covered
        num_branches: int | None = total
    else:
        coverage_ratio, covered = _calculate_line_coverage_metrics(
            source_path, function, statements, executed
        )
        total = len(statements)
        mode = "line_fallback"
        covered_branches = None
        num_branches = None
    complexity = _radon_complexity(function, source_path)
    crap = complexity + complexity**2 * (1.0 - coverage_ratio) ** 3
    return FunctionMeasurement(
        key=f"{_normalise_path(source_path)}::{function.qualified_name}::{function.start_line}",
        path=_normalise_path(source_path),
        qualified_name=function.qualified_name,
        start_line=function.start_line,
        end_line=function.end_line,
        complexity=complexity,
        coverage_mode=mode,
        covered_branches=covered_branches,
        num_branches=num_branches,
        covered_lines=len(executed & statements),
        num_lines=len(statements),
        coverage=coverage_ratio,
        crap=crap,
    )


def _expand_source_paths(source_paths: Iterable[str | os.PathLike[str]]) -> tuple[Path, ...]:
    """展开显式文件和目录，去重后返回 Python 源码路径。"""

    found: dict[str, Path] = {}
    for raw_path in source_paths:
        path = Path(raw_path).expanduser()
        if not path.exists():
            raise CrapMeasurementError(f"源码路径不存在：{path}")
        candidates = [path] if path.is_file() else sorted(path.rglob("*.py"))
        for candidate in candidates:
            if candidate.is_file() and candidate.suffix == ".py":
                found[_normalise_path(candidate)] = candidate.resolve()
    if not found:
        raise CrapMeasurementError("source/cohort 没有找到 Python 源文件")
    return tuple(found[key] for key in sorted(found))


def _read_cohort_paths(path: Path) -> list[str]:
    """读取换行或 JSON 格式的 cohort 路径清单。"""

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CrapMeasurementError(f"无法读取 cohort：{path}: {exc}") from exc
    if path.suffix.lower() == ".json":
        return _parse_json_cohort_paths(path, text)
    return [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]


def _parse_json_cohort_paths(path: Path, text: str) -> list[str]:
    """解析 JSON cohort 的列表或 paths/sources/files 容器。"""

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CrapMeasurementError(f"cohort JSON 无效：{path}: {exc}") from exc
    if isinstance(payload, Mapping):
        payload = next(
            (
                payload[key]
                for key in ("paths", "sources", "files")
                if isinstance(payload.get(key), list)
            ),
            payload,
        )
    if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
        raise CrapMeasurementError("cohort JSON 必须是字符串路径列表")
    return list(payload)


def measure_crap_for_sources(
    source_paths: Iterable[str | os.PathLike[str]],
    coverage_data_path: str | os.PathLike[str],
    *,
    source_hash_manifest: str | os.PathLike[str] | None = None,
    threshold: float = 30.0,
    coverage_format: str = "auto",
) -> CrapReport:
    """测量 source/cohort 的所有函数并返回严格阈值报告。

    输入是源码文件或目录、Coverage.py SQLite/JSON 数据和可选源哈希清单。
    函数范围由 AST 的起止行和限定名确定；覆盖率无法唯一映射、源码哈希过期、
    分支数据缺失或源码函数没有覆盖率记录时会抛出异常，而不会生成猜测结果。
    """

    if threshold <= 0:
        raise CrapMeasurementError("CRAP 阈值必须大于 0")
    coverage_path = Path(coverage_data_path).expanduser().resolve()
    if not coverage_path.is_file():
        raise CrapMeasurementError(f"coverage 数据不存在：{coverage_path}")
    sources = _expand_source_paths(source_paths)
    snapshot = _load_coverage_snapshot(coverage_path, coverage_format)
    manifest = _load_hash_manifest(
        Path(source_hash_manifest).expanduser().resolve()
        if source_hash_manifest is not None
        else None
    )
    measurements = []
    for source_path in sources:
        text = _read_source_text(source_path)
        tree = _parse_source(source_path, text)
        function_records = _collect_function_records(tree)
        if not function_records:
            continue
        coverage_record = _match_coverage_record(source_path, snapshot.records, coverage_path)
        _validate_source_hash(source_path, coverage_record, manifest)
        for function in function_records:
            measurements.append(
                _measure_function(
                    source_path,
                    function_records,
                    function,
                    coverage_record,
                )
            )
    if not measurements:
        raise CoverageMappingError("cohort 中没有可测量函数")
    return CrapReport(
        threshold=float(threshold),
        coverage_data=str(coverage_path),
        functions=tuple(measurements),
        coverage_format=snapshot.format,
    )


def write_changed_function_cohort_manifest(
    cohort: ChangedFunctionCohort,
    output_path: str | os.PathLike[str],
) -> None:
    """把自动生成的 cohort manifest 写到调用者指定路径。"""

    payload = json.dumps(cohort.to_dict(), ensure_ascii=False, indent=2) + "\n"
    _write_text_output(Path(output_path).expanduser().resolve(), payload)


def _group_changed_functions_by_source_path(
    cohort: ChangedFunctionCohort,
) -> dict[str, list[ChangedFunction]]:
    """按当前源码相对路径分组变更函数。"""

    grouped: dict[str, list[ChangedFunction]] = defaultdict(list)
    for changed in cohort.functions:
        grouped[changed.path].append(changed)
    return grouped


def _resolve_changed_function_record(
    changed: ChangedFunction,
    current_records: Mapping[tuple[str, int], _FunctionRecord],
) -> _FunctionRecord:
    """将 manifest 函数定位到当前 AST，过期清单直接失败。"""

    if changed.current_start_line is None:
        raise CoverageMappingError(
            f"当前 cohort 函数缺少当前行：{changed.path}:{changed.qualified_name}"
        )
    key = (changed.qualified_name, changed.current_start_line)
    function = current_records.get(key)
    if function is None or function.end_line != changed.current_end_line:
        raise CoverageMappingError(
            "cohort manifest 与当前源码不一致，拒绝使用过期或手工缩减的清单："
            f"{changed.path}:{changed.qualified_name}:{changed.current_start_line}"
        )
    return function


def _measure_changed_source_path_functions(
    repository_root: Path,
    coverage_path: Path,
    snapshot: _CoverageSnapshot,
    manifest: Mapping[str, str],
    relative_path: str,
    changed_functions: Sequence[ChangedFunction],
) -> list[FunctionMeasurement]:
    """校验一个当前源码文件的 manifest 并测量其全部变更函数。"""

    source_path, source_text = _read_current_source(repository_root, relative_path)
    records = _collect_function_records(_parse_source(source_path, source_text))
    current_records = {
        (record.qualified_name, record.start_line): record for record in records
    }
    coverage_record = _match_coverage_record(source_path, snapshot.records, coverage_path)
    _validate_source_hash(source_path, coverage_record, manifest)
    measurements: list[FunctionMeasurement] = []
    for changed in sorted(
        changed_functions,
        key=lambda item: (item.current_start_line or 0, item.qualified_name),
    ):
        function = _resolve_changed_function_record(changed, current_records)
        measurements.append(
            _measure_function(source_path, records, function, coverage_record)
        )
    return measurements


def measure_crap_for_changed_functions(
    coverage_data_path: str | os.PathLike[str],
    *,
    repository: str | os.PathLike[str] = ".",
    baseline: str | None = None,
    path_filters: Iterable[str | os.PathLike[str]] = (),
    include_untracked: bool = True,
    source_hash_manifest: str | os.PathLike[str] | None = None,
    threshold: float = 30.0,
    coverage_format: str = "auto",
) -> CrapReport:
    """只测量 git diff 证明新增或修改的当前函数，并返回 cohort 证据。

    函数集合每次都由 ``baseline`` 与工作树重新生成，不接受手工函数白名单。
    覆盖率仍按现有严格函数范围映射；删除函数只保存在返回的 manifest 中，不
    参与 CRAP 或阈值判断。工作树没有可测量的当前函数时直接失败，避免空集合
    被错误解释为通过。
    """

    if threshold <= 0:
        raise CrapMeasurementError("CRAP 阈值必须大于 0")
    cohort = generate_changed_function_cohort_manifest(
        repository,
        baseline=baseline,
        path_filters=path_filters,
        include_untracked=include_untracked,
    )
    if not cohort.functions:
        raise CoverageMappingError(
            "git diff 没有可测量的新增/修改函数；删除函数已写入 cohort manifest"
        )
    coverage_path = Path(coverage_data_path).expanduser().resolve()
    if not coverage_path.is_file():
        raise CrapMeasurementError(f"coverage 数据不存在：{coverage_path}")
    snapshot = _load_coverage_snapshot(coverage_path, coverage_format)
    manifest = _load_hash_manifest(
        Path(source_hash_manifest).expanduser().resolve()
        if source_hash_manifest is not None
        else None
    )
    repository_root = Path(cohort.repository)
    grouped = _group_changed_functions_by_source_path(cohort)
    measurements: list[FunctionMeasurement] = []
    for relative_path in sorted(grouped):
        measurements.extend(
            _measure_changed_source_path_functions(
                repository_root,
                coverage_path,
                snapshot,
                manifest,
                relative_path,
                grouped[relative_path],
            )
        )
    report = CrapReport(
        threshold=float(threshold),
        coverage_data=str(coverage_path),
        functions=tuple(measurements),
        coverage_format=snapshot.format,
        cohort_manifest=cohort.to_dict(),
    )
    return report


def measure_crap_for_changed_cohort(
    coverage_data_path: str | os.PathLike[str],
    **kwargs: Any,
) -> CrapReport:
    """按自动生成的变更函数 cohort 测量 CRAP 的短入口。"""

    return measure_crap_for_changed_functions(coverage_data_path, **kwargs)


def _write_text_output(path: Path, text: str) -> None:
    """以同目录临时文件原子替换报告，失败时保留旧文件。"""

    temporary_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except OSError as exc:
        raise CrapMeasurementError(f"无法写入报告：{path}: {exc}") from exc
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def _build_argument_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""

    parser = argparse.ArgumentParser(description="测量函数级 CRAP 指数并执行严格门禁")
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="源码文件或目录，可重复传入",
    )
    parser.add_argument(
        "--cohort",
        action="append",
        default=[],
        help="换行或 JSON 格式的源码路径清单，可重复传入",
    )
    parser.add_argument(
        "--baseline",
        help="git 变更基线 commit/ref；传入后自动只测新增/修改函数，默认基线为 origin/harness",
    )
    parser.add_argument(
        "--changed-only",
        "--changed-cohort",
        action="store_true",
        help="按 git diff 自动生成当前函数 cohort；不传 --baseline 时使用 origin/harness",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="git 仓库目录，默认当前目录",
    )
    parser.add_argument(
        "--cohort-manifest",
        "--output-cohort-manifest",
        "--manifest",
        dest="cohort_manifest",
        help="写入自动生成的 git 函数 cohort manifest 的 JSON 路径",
    )
    parser.add_argument(
        "--include-untracked",
        action="store_true",
        default=True,
        help="把未跟踪 Python 文件作为新增文件纳入（默认开启；指定 --source 时限于该范围）",
    )
    parser.add_argument("--coverage-data", required=True, help="Coverage.py SQLite 或 JSON")
    parser.add_argument(
        "--coverage-format",
        choices=("auto", "sqlite", "json"),
        default="auto",
        help="覆盖率格式，默认按扩展名自动识别",
    )
    parser.add_argument("--source-hash-manifest", help="覆盖率采集时的源码 SHA-256 清单")
    parser.add_argument("--threshold", type=float, default=30.0, help="严格门禁阈值，默认 30")
    parser.add_argument("--output-json", help="JSON 报告路径；不传则只输出标准输出")
    parser.add_argument("--output-markdown", help="Markdown 报告路径")
    return parser


def _write_error_outputs(
    args: argparse.Namespace,
    message: str,
    cohort: ChangedFunctionCohort | None = None,
) -> None:
    """在失败闭环中尽力写出结构化错误，不吞掉原始失败。"""

    error_payload: dict[str, Any] = {"status": "error", "error": message}
    if cohort is not None:
        error_payload["cohort_manifest"] = cohort.to_dict()
    payload = json.dumps(error_payload, ensure_ascii=False, indent=2) + "\n"
    if args.output_json:
        _write_text_output(Path(args.output_json), payload)
    if args.output_markdown:
        _write_text_output(Path(args.output_markdown), f"# 函数级 CRAP 测量\n\n- 状态：`error`\n- 错误：{message}\n")


def main(argv: Sequence[str] | None = None) -> int:
    """执行 CLI；成功、阈值失败和输入错误分别返回 0、1、2。"""

    args = _build_argument_parser().parse_args(argv)
    source_paths = list(args.source)
    for cohort in args.cohort:
        source_paths.extend(_read_cohort_paths(Path(cohort).expanduser().resolve()))
    changed_mode = args.changed_only or args.baseline is not None
    if not source_paths and not changed_mode:
        _build_argument_parser().error("至少需要一个 --source 或 --cohort")
    generated_cohort: ChangedFunctionCohort | None = None
    try:
        if changed_mode:
            generated_cohort = generate_changed_function_cohort_manifest(
                args.repo_root,
                baseline=args.baseline,
                path_filters=source_paths,
                include_untracked=args.include_untracked,
            )
            if args.cohort_manifest:
                write_changed_function_cohort_manifest(
                    generated_cohort, args.cohort_manifest
                )
            report = measure_crap_for_changed_functions(
                args.coverage_data,
                repository=args.repo_root,
                baseline=args.baseline,
                path_filters=source_paths,
                include_untracked=args.include_untracked,
                source_hash_manifest=args.source_hash_manifest,
                threshold=args.threshold,
                coverage_format=args.coverage_format,
            )
        else:
            report = measure_crap_for_sources(
                source_paths,
                args.coverage_data,
                source_hash_manifest=args.source_hash_manifest,
                threshold=args.threshold,
                coverage_format=args.coverage_format,
            )
    except CrapMeasurementError as exc:
        _write_error_outputs(args, str(exc), generated_cohort)
        error_payload: dict[str, Any] = {"status": "error", "error": str(exc)}
        if generated_cohort is not None:
            error_payload["cohort_manifest"] = generated_cohort.to_dict()
        print(json.dumps(error_payload, ensure_ascii=False))
        return 2
    payload = json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n"
    if args.output_json:
        _write_text_output(Path(args.output_json), payload)
    if args.output_markdown:
        _write_text_output(Path(args.output_markdown), report.render_markdown())
    print(payload, end="")
    return 0 if report.passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
