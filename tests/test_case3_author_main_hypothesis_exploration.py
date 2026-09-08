"""Case 3 主假设探索模式的 prepared 输入、提示和恢复合同测试。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from chemistry_multiagent.controllers.case23_reproduction import (
    CASE23_AUTHOR_SCIENTIFIC_QUESTION_INPUTS,
)
from chemistry_multiagent.controllers.chemistry_multiagent_controller import (
    _build_cli_parser,
    _validate_cli_request,
)
from chemistry_multiagent.investigation.author_hypothesis_import import (
    CASE3_AUTHOR_MAIN_HYPOTHESIS_EXPLORATION_MODE,
    build_case3_workspace_input_paths,
    load_author_main_hypothesis,
    load_case3_prepared_input_bundle,
    materialize_case3_session_input_files,
    write_case3_prepared_input_files,
)
from chemistry_multiagent.investigation.prompts import (
    build_initial_investigation_request,
    build_investigation_job_continuation_request,
    build_investigation_job_wake_request,
)
from chemistry_multiagent.memory.contracts import ContractValidationError


QUESTION = CASE23_AUTHOR_SCIENTIFIC_QUESTION_INPUTS["case3"]
EXPECTED_FILES = {
    "origin.gjf",
    "ortho.gjf",
    "meta.gjf",
    "para.gjf",
    "dihedral_atoms.json",
}


def _write_prepared_input_directory(tmp_path: Path) -> Path:
    """在测试临时目录写入不含计算设置的五个最小输入文件。

    输入是 pytest 提供的临时目录；动作会新建 ``prepared-input`` 并写入四个结构
    文件和一个原子序号文件，返回该目录。目录已存在或写入失败时保留底层异常。
    """

    prepared_input_dir = tmp_path / "prepared-input"
    prepared_input_dir.mkdir()
    structure_contracts = {
        "origin": {
            "title": "Case 3 unsubstituted structure",
            "display_name": "unsubstituted",
            "atom_count": 23,
            "element_counts": {"C": 10, "H": 9, "N": 2, "Ni": 1, "Br": 1},
            "atom_indices": [21, 3, 8, 20],
        },
        "ortho": {
            "title": "Case 3 ortho-substituted structure",
            "display_name": "ortho-substituted",
            "atom_count": 29,
            "element_counts": {"C": 12, "H": 13, "N": 2, "Ni": 1, "Br": 1},
            "atom_indices": [19, 3, 8, 18],
        },
        "meta": {
            "title": "Case 3 meta-substituted structure",
            "display_name": "meta-substituted",
            "atom_count": 29,
            "element_counts": {"C": 12, "H": 13, "N": 2, "Ni": 1, "Br": 1},
            "atom_indices": [29, 4, 9, 28],
        },
        "para": {
            "title": "Case 3 para-substituted structure",
            "display_name": "para-substituted",
            "atom_count": 29,
            "element_counts": {"C": 12, "H": 13, "N": 2, "Ni": 1, "Br": 1},
            "atom_indices": [19, 3, 8, 18],
        },
    }
    for identity, structure_contract in structure_contracts.items():
        elements: list[str | None] = [None] * structure_contract["atom_count"]
        remaining_elements = [
            element
            for element, count in structure_contract["element_counts"].items()
            for _ in range(count)
        ]
        for index, element in zip(
            structure_contract["atom_indices"],
            ("Br", "N", "N", "H"),
            strict=True,
        ):
            elements[index - 1] = element
            remaining_elements.remove(element)
        for index, element in zip(
            (index for index, value in enumerate(elements) if value is None),
            remaining_elements,
            strict=True,
        ):
            elements[index] = element
        coordinate_lines = "\n".join(
            f"{element}  {index:.8f}  0.00000000  0.00000000"
            for index, element in enumerate(elements, start=1)
        )
        (prepared_input_dir / f"{identity}.gjf").write_text(
            f"{structure_contract['title']}\n\n0 1\n{coordinate_lines}\n\n",
            encoding="utf-8",
        )
    (prepared_input_dir / "dihedral_atoms.json").write_text(
        json.dumps(
            {
                "schema": "arche.case3.prepared_structure_inputs.v1",
                "index_base": 1,
                "coordinate_type": "dihedral",
                "atom_roles": ["Br", "N", "N", "H"],
                "structures": {
                    "origin": {
                        "display_name": structure_contracts["origin"]["display_name"],
                        "file": "origin.gjf",
                        "atom_indices": [21, 3, 8, 20],
                    },
                    "ortho": {
                        "display_name": structure_contracts["ortho"]["display_name"],
                        "file": "ortho.gjf",
                        "atom_indices": [19, 3, 8, 18],
                    },
                    "meta": {
                        "display_name": structure_contracts["meta"]["display_name"],
                        "file": "meta.gjf",
                        "atom_indices": [29, 4, 9, 28],
                    },
                    "para": {
                        "display_name": structure_contracts["para"]["display_name"],
                        "file": "para.gjf",
                        "atom_indices": [19, 3, 8, 18],
                    },
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return prepared_input_dir


def _fixture():
    """加载不携带结构化计算协议的作者主假设。"""

    return load_author_main_hypothesis(
        QUESTION,
        author_mode=CASE3_AUTHOR_MAIN_HYPOTHESIS_EXPLORATION_MODE,
    )


def test_prepared_bundle_is_exactly_five_files_and_has_expected_indices(
    tmp_path: Path,
) -> None:
    """prepared bundle 只接受五个普通文件，并冻结四组作者给出的序号。"""

    prepared_input_dir = _write_prepared_input_directory(tmp_path)
    assert not prepared_input_dir.is_symlink()
    assert {item.name for item in prepared_input_dir.iterdir()} == EXPECTED_FILES
    prepared = load_case3_prepared_input_bundle(str(prepared_input_dir.resolve()))
    assert prepared["schema"] == "arche.case3.prepared_input_bundle.v1"
    assert set(prepared["files"]) == EXPECTED_FILES
    for filename, record in prepared["files"].items():
        assert set(record) == {"sha256", "content"}
        assert record["sha256"] == hashlib.sha256(
            record["content"].encode("utf-8")
        ).hexdigest()
    mapping = json.loads(prepared["files"]["dihedral_atoms.json"]["content"])
    assert {
        identity: record["atom_indices"]
        for identity, record in mapping["structures"].items()
    } == {
        "origin": [21, 3, 8, 20],
        "ortho": [19, 3, 8, 18],
        "meta": [29, 4, 9, 28],
        "para": [19, 3, 8, 18],
    }


def test_prepared_bundle_rejects_symlink_extra_file_and_gaussian_settings(
    tmp_path: Path,
) -> None:
    """目录、文件和禁止设置任一越界时都 fail-closed。"""

    source_input_dir = _write_prepared_input_directory(tmp_path)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    for filename in EXPECTED_FILES:
        source = source_input_dir / filename
        (prepared / filename).write_bytes(source.read_bytes())
    (prepared / "extra.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(ContractValidationError, match="固定五个文件"):
        load_case3_prepared_input_bundle(str(prepared.resolve()))
    (prepared / "extra.txt").unlink()
    (prepared / "origin.gjf").write_text("%mem=1GB\n0 1\n", encoding="utf-8")
    with pytest.raises(ContractValidationError, match="Gaussian"):
        load_case3_prepared_input_bundle(str(prepared.resolve()))
    (prepared / "origin.gjf").unlink()
    (prepared / "origin.gjf").symlink_to(source_input_dir / "origin.gjf")
    with pytest.raises(ContractValidationError, match="普通文件"):
        load_case3_prepared_input_bundle(str(prepared.resolve()))
    linked_directory = tmp_path / "prepared-link"
    linked_directory.symlink_to(prepared, target_is_directory=True)
    with pytest.raises(ContractValidationError, match="非符号链接目录"):
        load_case3_prepared_input_bundle(str(linked_directory))


@pytest.mark.parametrize(
    "leaked_setting",
    (
        "B3LYP 6-31G DMA",
        "scan range 0 60 step 5",
        "cores 60 ram 320000MB",
    ),
)
def test_prepared_structure_format_rejects_unlisted_settings(
    tmp_path: Path,
    leaked_setting: str,
) -> None:
    """固定结构格式拒绝不在关键词黑名单中的方法、扫描或资源写法。"""

    prepared_input_dir = _write_prepared_input_directory(tmp_path)
    origin_path = prepared_input_dir / "origin.gjf"
    content = origin_path.read_text(encoding="utf-8")
    origin_path.write_text(
        content.replace(
            "Case 3 unsubstituted structure",
            f"Case 3 unsubstituted structure {leaked_setting}",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ContractValidationError, match="固定格式"):
        load_case3_prepared_input_bundle(str(prepared_input_dir.resolve()))


@pytest.mark.parametrize("invalid_json", ("duplicate_key", "nonfinite"))
def test_dihedral_mapping_requires_strict_json(
    tmp_path: Path,
    invalid_json: str,
) -> None:
    """原子序号文件拒绝重复字段和 JSON 非有限数字常量。"""

    prepared_input_dir = _write_prepared_input_directory(tmp_path)
    mapping_path = prepared_input_dir / "dihedral_atoms.json"
    content = mapping_path.read_text(encoding="utf-8")
    if invalid_json == "duplicate_key":
        content = content.replace(
            '"schema": "arche.case3.prepared_structure_inputs.v1",',
            '"schema": "arche.case3.prepared_structure_inputs.v1", '
            '"schema": "arche.case3.prepared_structure_inputs.v1",',
            1,
        )
    else:
        content = content.replace('"index_base": 1', '"index_base": NaN', 1)
    mapping_path.write_text(content, encoding="utf-8")
    with pytest.raises(ContractValidationError, match="重复字段|非法常量"):
        load_case3_prepared_input_bundle(str(prepared_input_dir.resolve()))


def test_session_copy_uses_frozen_task_content_after_external_directory_changes(
    tmp_path: Path,
) -> None:
    """workspace 复制只依赖任务冻结内容，外部 prepared 目录消失不影响恢复。"""

    prepared_input_dir = _write_prepared_input_directory(tmp_path)
    prepared = load_case3_prepared_input_bundle(str(prepared_input_dir.resolve()))
    task_root = tmp_path / "task"
    task_root.mkdir()
    write_case3_prepared_input_files(task_root, prepared)
    prepared_input_dir.rename(tmp_path / "external-input-no-longer-at-original-path")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    # 不把 prepared 目录路径传入复制函数，证明恢复只使用 task_root 的冻结副本。
    materialize_case3_session_input_files(task_root, workspace, prepared)
    copied = workspace / "inputs" / "case3"
    assert {item.name for item in copied.iterdir()} == EXPECTED_FILES
    for filename in EXPECTED_FILES:
        assert (copied / filename).read_bytes() == (
            task_root / "inputs" / "case3" / filename
        ).read_bytes()
    assert build_case3_workspace_input_paths() == {
        "structure_files": {
            "origin": "inputs/case3/origin.gjf",
            "ortho": "inputs/case3/ortho.gjf",
            "meta": "inputs/case3/meta.gjf",
            "para": "inputs/case3/para.gjf",
        },
        "dihedral_mapping_file": "inputs/case3/dihedral_atoms.json",
        "instructions": build_case3_workspace_input_paths()["instructions"],
    }


def test_exploration_prompts_only_expose_workspace_relative_paths() -> None:
    """initial/wake/continuation 消息只给相对路径，不带清理正文或方法设置。"""

    fixture = _fixture()
    paths = build_case3_workspace_input_paths()
    initial = build_initial_investigation_request(
        "task-case3-exploration",
        "batch-case3-exploration",
        "combination-case3-exploration",
        QUESTION,
        fixture.hypothesis,
        workspace_input_files=paths,
    )
    wake = build_investigation_job_wake_request(
        "task-case3-exploration",
        "batch-case3-exploration",
        "combination-case3-exploration",
        "thread-case3-exploration",
        job_id="job-case3-exploration",
        operation_id="operation-case3-exploration",
        submission_key="submission-case3-exploration",
        terminal_status="succeeded",
        scientific_question=QUESTION,
        hypothesis=fixture.hypothesis,
        workspace_input_files=paths,
    )
    continuation = build_investigation_job_continuation_request(
        "task-case3-exploration",
        "batch-case3-exploration",
        "combination-case3-exploration",
        "thread-case3-exploration",
        job_id="job-case3-exploration",
        operation_id="operation-case3-exploration",
        submission_key="submission-case3-exploration",
        scientific_question=QUESTION,
        hypothesis=fixture.hypothesis,
        workspace_input_files=paths,
    )
    for request in (initial, wake, continuation):
        serialized = json.dumps(request, ensure_ascii=False)
        assert "inputs/case3/" in serialized
        assert "wB97XD" not in serialized
        assert "def2SVP" not in serialized
        assert "SMD(DMA)" not in serialized
        assert "NoSymm" not in serialized
        assert "S 12 5" not in serialized
        assert "source_sha256" not in serialized
    payload = json.loads(initial["messages"][-1]["content"])
    assert payload["workspace_input_files"] == paths
    assert set(payload) == {
        "scientific_question",
        "explicit_conditions",
        "hypothesis",
        "frozen_context",
        "workspace_input_files",
    }


def test_exploration_cli_requires_prepared_input_and_new_flag(tmp_path: Path) -> None:
    """CLI 只注册新的主假设探索开关并要求 prepared 目录。"""

    prepared_input_dir = _write_prepared_input_directory(tmp_path)
    parser = _build_cli_parser()
    args = parser.parse_args(
        [
            "create-standard-research-task",
            "--question",
            QUESTION,
            "--use-case3-author-main-hypothesis-exploration",
            "--case3-prepared-input-dir",
            str(prepared_input_dir.resolve()),
        ]
    )
    _validate_cli_request(parser, args)
    assert args.use_case3_author_main_hypothesis_exploration is True
    with pytest.raises(SystemExit):
        _validate_cli_request(
            parser,
            parser.parse_args(
                [
                    "create-standard-research-task",
                    "--question",
                    QUESTION,
                    "--use-case3-author-main-hypothesis-exploration",
                ]
            ),
        )
    with pytest.raises(SystemExit):
        parser.parse_args(["--use-case3-author-main-hypothesis-only"])
