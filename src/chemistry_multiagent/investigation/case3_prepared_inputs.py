"""Case 3 清理输入 bundle 的校验、冻结和 workspace 文件复制。

父流程在进入 Controller 前负责从作者资料生成清理后的五文件目录。本模块只接收
这个目录或已经冻结的内容，不读取 raw GJF、计算日志，也不解析原子坐标。它只核对
固定文件名、普通文件、UTF-8、禁止的 Gaussian 协议词和四原子映射，并把同一内容
写入任务目录及每个 Investigation session 的相对 ``inputs/case3`` 目录。
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any


CASE3_PREPARED_INPUT_SCHEMA = "arche.case3.prepared_input_bundle.v1"
CASE3_PREPARED_INPUT_FILENAMES = (
    "origin.gjf",
    "ortho.gjf",
    "meta.gjf",
    "para.gjf",
    "dihedral_atoms.json",
)
_CASE3_STRUCTURE_CONTRACTS = {
    "origin": {
        "title": "Case 3 unsubstituted structure",
        "display_name": "unsubstituted",
        "atom_count": 23,
        "element_counts": {"C": 10, "H": 9, "N": 2, "Ni": 1, "Br": 1},
        "atom_indices": (21, 3, 8, 20),
    },
    "ortho": {
        "title": "Case 3 ortho-substituted structure",
        "display_name": "ortho-substituted",
        "atom_count": 29,
        "element_counts": {"C": 12, "H": 13, "N": 2, "Ni": 1, "Br": 1},
        "atom_indices": (19, 3, 8, 18),
    },
    "meta": {
        "title": "Case 3 meta-substituted structure",
        "display_name": "meta-substituted",
        "atom_count": 29,
        "element_counts": {"C": 12, "H": 13, "N": 2, "Ni": 1, "Br": 1},
        "atom_indices": (29, 4, 9, 28),
    },
    "para": {
        "title": "Case 3 para-substituted structure",
        "display_name": "para-substituted",
        "atom_count": 29,
        "element_counts": {"C": 12, "H": 13, "N": 2, "Ni": 1, "Br": 1},
        "atom_indices": (19, 3, 8, 18),
    },
}
_CASE3_PREPARED_FORBIDDEN_RE = re.compile(
    r"(?:"
    r"^\s*[%#]"
    r"|\b(?:route|gaussian|nproc(?:shared)?|mem(?:ory)?|wb97(?:xd)?|def2(?:-[a-z0-9]+)?|"
    r"smd|nosymm|modredundant|scrf|opt(?:imize)?|freq(?:uency)?)\b"
    r"|\bS\s+12\s+5(?:\.0+)?\s*$"
    r")",
    re.IGNORECASE | re.MULTILINE,
)
_CASE3_COORDINATE_LINE_RE = re.compile(
    r"^(Br|Ni|C|N|H)\s+(-?\d+\.\d{8})\s+(-?\d+\.\d{8})\s+(-?\d+\.\d{8})$"
)


def _require_prepared_directory(value: str | Path) -> Path:
    """校验 prepared bundle 的原始路径是绝对的非符号链接目录。"""

    path = Path(value)
    if not path.is_absolute():
        raise ValueError("Case 3 prepared input 目录必须是绝对路径")
    # 必须先检查原始路径；resolve 后再检查会把符号链接伪装成普通目录。
    if path.is_symlink() or not path.is_dir():
        raise ValueError("Case 3 prepared input 必须是非符号链接目录")
    return path.resolve(strict=True)


def _read_prepared_file(directory: Path, filename: str) -> tuple[str, str]:
    """读取固定 prepared 文件并返回 UTF-8 文本及清理内容 SHA-256。"""

    target = directory / filename
    if target.is_symlink() or not target.is_file():
        raise ValueError(f"Case 3 prepared input {filename} 缺失或不是普通文件")
    try:
        raw = target.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Case 3 prepared input {filename} 无法读取为 UTF-8") from exc
    if not raw or "\x00" in text:
        raise ValueError(f"Case 3 prepared input {filename} 为空或含 NUL 字符")
    if _CASE3_PREPARED_FORBIDDEN_RE.search(text):
        raise ValueError(f"Case 3 prepared input {filename} 含 Gaussian 设置或扫描协议")
    return text, hashlib.sha256(raw).hexdigest()


def _validate_dihedral_mapping(value: Any) -> dict[str, Any]:
    """校验四组 1-based 二面角映射，不解析 GJF 原子和坐标。"""

    if not isinstance(value, Mapping):
        raise ValueError("Case 3 dihedral_atoms.json 必须是 JSON object")
    if set(value) != {
        "schema",
        "index_base",
        "coordinate_type",
        "atom_roles",
        "structures",
    }:
        raise ValueError("Case 3 dihedral_atoms.json 字段不完整或含未知字段")
    if value["schema"] != "arche.case3.prepared_structure_inputs.v1":
        raise ValueError("Case 3 dihedral_atoms.json schema 非法")
    if (
        isinstance(value["index_base"], bool)
        or value["index_base"] != 1
        or value["coordinate_type"] != "dihedral"
    ):
        raise ValueError("Case 3 dihedral_atoms.json 索引协议非法")
    if value["atom_roles"] != ["Br", "N", "N", "H"]:
        raise ValueError("Case 3 二面角原子角色必须是 Br-N-N-H")
    structures = value["structures"]
    if not isinstance(structures, Mapping) or set(structures) != set(
        _CASE3_STRUCTURE_CONTRACTS
    ):
        raise ValueError("Case 3 二面角映射必须覆盖四种结构")
    normalized = copy.deepcopy(dict(value))
    for identity, structure_contract in _CASE3_STRUCTURE_CONTRACTS.items():
        record = structures[identity]
        if not isinstance(record, Mapping) or set(record) != {
            "display_name",
            "file",
            "atom_indices",
        }:
            raise ValueError(f"Case 3 {identity} 二面角映射字段非法")
        if record["file"] != f"{identity}.gjf":
            raise ValueError(f"Case 3 {identity} 二面角映射文件名非法")
        indices = record["atom_indices"]
        if (
            not isinstance(indices, list)
            or len(indices) != 4
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 1
                for item in indices
            )
            or tuple(indices) != structure_contract["atom_indices"]
        ):
            raise ValueError(
                f"Case 3 {identity} 二面角必须使用冻结的四个 1-based 原子序号"
            )
        if record["display_name"] != structure_contract["display_name"]:
            raise ValueError(f"Case 3 {identity} 二面角 display_name 非法")
    return normalized


def _validate_case3_structure_file(
    filename: str,
    content: str,
) -> tuple[str, ...]:
    """按固定文本格式校验一个清理结构文件。

    输入是已登记的 GJF 文件名和 UTF-8 正文；动作只接受固定标题、空行、``0 1``
    以及指定数量的元素/三维坐标行，返回按文件顺序排列的元素符号。标题、行数、
    电荷/多重度、坐标格式或元素集合不匹配时抛出 ValueError。
    """

    identity = filename.removesuffix(".gjf")
    structure_contract = _CASE3_STRUCTURE_CONTRACTS.get(identity)
    if structure_contract is None or filename != f"{identity}.gjf":
        raise ValueError("Case 3 清理结构文件名未登记")
    lines = content.splitlines()
    expected_atom_count = structure_contract["atom_count"]
    if (
        len(lines) != expected_atom_count + 4
        or lines[0] != structure_contract["title"]
        or lines[1] != ""
        or lines[2] != "0 1"
        or lines[-1] != ""
    ):
        raise ValueError(f"Case 3 {filename} 不符合清理结构固定格式")
    elements: list[str] = []
    for line in lines[3:-1]:
        match = _CASE3_COORDINATE_LINE_RE.fullmatch(line)
        if match is None:
            raise ValueError(f"Case 3 {filename} 含非坐标正文或非法坐标格式")
        elements.append(match.group(1))
    if Counter(elements) != structure_contract["element_counts"]:
        raise ValueError(f"Case 3 {filename} 元素组成与冻结结构不匹配")
    return tuple(elements)


def _load_strict_json_object(content: str) -> dict[str, Any]:
    """严格解析 JSON object，拒绝重复键和非有限数字常量。

    输入是 UTF-8 JSON 正文；动作使用成对键回调拒绝重复字段，并拒绝
    ``NaN``、``Infinity`` 和 ``-Infinity``，返回普通字典。语法、常量、重复键或
    顶层类型不合法时抛出 ValueError。
    """

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        parsed: dict[str, Any] = {}
        for key, value in pairs:
            if key in parsed:
                raise ValueError(f"Case 3 dihedral_atoms.json 含重复字段：{key}")
            parsed[key] = value
        return parsed

    def reject_nonfinite_constant(value: str) -> None:
        raise ValueError(f"Case 3 dihedral_atoms.json 含非法常量：{value}")

    try:
        parsed = json.loads(
            content,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError("Case 3 dihedral_atoms.json 不是严格 JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Case 3 dihedral_atoms.json 顶层必须是 JSON object")
    return parsed


def validate_case3_prepared_input_metadata(value: Any) -> dict[str, Any]:
    """校验 TaskCreated 冻结的清理内容和文件 SHA-256。

    输入是包含 schema、版本和五个文件正文/摘要的 JSON object；动作只在内存中
    检查固定文件名、摘要、禁止协议词和映射，不读取任何路径。返回可安全复制的
    规范普通字典；字段缺失、摘要漂移、映射错误或正文含协议时抛出 ValueError。
    """

    if not isinstance(value, Mapping) or set(value) != {
        "schema",
        "schema_version",
        "files",
    }:
        raise ValueError("Case 3 prepared input metadata 字段非法")
    if (
        value["schema"] != CASE3_PREPARED_INPUT_SCHEMA
        or isinstance(value["schema_version"], bool)
        or value["schema_version"] != 1
    ):
        raise ValueError("Case 3 prepared input metadata schema 非法")
    files = value["files"]
    if not isinstance(files, Mapping) or set(files) != set(CASE3_PREPARED_INPUT_FILENAMES):
        raise ValueError("Case 3 prepared input metadata 必须固定五个文件")
    normalized_files: dict[str, Any] = {}
    structure_elements: dict[str, tuple[str, ...]] = {}
    for filename in CASE3_PREPARED_INPUT_FILENAMES:
        record = files[filename]
        if not isinstance(record, Mapping) or set(record) != {"sha256", "content"}:
            raise ValueError(f"Case 3 prepared input metadata {filename} 字段非法")
        digest = record["sha256"]
        content = record["content"]
        if (
            not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or not isinstance(content, str)
            or not content
            or "\x00" in content
        ):
            raise ValueError(f"Case 3 prepared input metadata {filename} 摘要或内容非法")
        if hashlib.sha256(content.encode("utf-8")).hexdigest() != digest:
            raise ValueError(f"Case 3 prepared input metadata {filename} SHA-256 不匹配")
        if _CASE3_PREPARED_FORBIDDEN_RE.search(content):
            raise ValueError(
                f"Case 3 prepared input metadata {filename} 含 Gaussian 设置或扫描协议"
            )
        normalized_files[filename] = {"sha256": digest, "content": content}
        if filename.endswith(".gjf"):
            structure_elements[filename] = _validate_case3_structure_file(
                filename,
                content,
            )
    mapping = _load_strict_json_object(
        normalized_files["dihedral_atoms.json"]["content"]
    )
    _validate_dihedral_mapping(mapping)
    for identity, record in mapping["structures"].items():
        elements = structure_elements[record["file"]]
        selected_roles = tuple(
            elements[index - 1] for index in record["atom_indices"]
        )
        if selected_roles != ("Br", "N", "N", "H"):
            raise ValueError(
                f"Case 3 {identity} 二面角原子序号与结构元素不匹配"
            )
    return {
        "schema": CASE3_PREPARED_INPUT_SCHEMA,
        "schema_version": 1,
        "files": normalized_files,
    }


def load_case3_prepared_input_bundle(directory: str | Path) -> dict[str, Any]:
    """读取父流程准备的恰好五个清理文件。

    输入是绝对的非符号链接目录；动作只读取固定文件并计算清理内容摘要，不会
    解析 GJF、读取 raw 来源/日志或修改目录。返回可直接放入 TaskCreated 的冻结
    表示；目录、文件、UTF-8、映射或禁止词不符合要求时抛出 ValueError。
    """

    prepared_directory = _require_prepared_directory(directory)
    try:
        entries = {entry.name for entry in prepared_directory.iterdir()}
    except OSError as exc:
        raise ValueError("Case 3 prepared input 目录无法读取") from exc
    if entries != set(CASE3_PREPARED_INPUT_FILENAMES):
        raise ValueError("Case 3 prepared input 目录必须恰好包含固定五个文件")
    files: dict[str, Any] = {}
    for filename in CASE3_PREPARED_INPUT_FILENAMES:
        content, digest = _read_prepared_file(prepared_directory, filename)
        files[filename] = {"sha256": digest, "content": content}
    return validate_case3_prepared_input_metadata(
        {
            "schema": CASE3_PREPARED_INPUT_SCHEMA,
            "schema_version": 1,
            "files": files,
        }
    )


def _write_case3_file_atomically(root: Path, filename: str, content: str) -> None:
    """在任务或 workspace 的 inputs/case3 下原子写一个清理文件。"""

    if filename not in CASE3_PREPARED_INPUT_FILENAMES:
        raise ValueError("Case 3 prepared input 文件名未登记")
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Case 3 输入根目录必须是普通目录")
    target_directory = root / "inputs" / "case3"
    for directory in (root / "inputs", target_directory):
        if directory.exists() and (directory.is_symlink() or not directory.is_dir()):
            raise ValueError("Case 3 输入目录不能是符号链接或普通文件")
        directory.mkdir(parents=True, exist_ok=True)
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("Case 3 输入目录准备失败")
    target = target_directory / filename
    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise ValueError("Case 3 输入文件不能是符号链接或目录")
    temporary = ""
    descriptor = -1
    try:
        descriptor, temporary = tempfile.mkstemp(prefix=f".{filename}.", dir=str(target_directory))
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(content.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        if target.is_symlink():
            raise ValueError("Case 3 输入文件不能是符号链接")
        os.replace(temporary, target)
        temporary = ""
        os.chmod(target, 0o600)
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError("Case 3 输入文件无法持久化") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary:
            try:
                os.unlink(temporary)
            except OSError:
                pass


def write_case3_prepared_input_files(
    task_root: str | Path,
    structure_input: Mapping[str, Any],
) -> dict[str, Any]:
    """把冻结清理内容写入任务相对的 ``inputs/case3``。

    输入是任务根目录和已校验的五文件冻结表示；动作原子写入五个普通文件并收紧
    权限，不读取外部 prepared 目录，也不返回文件正文。返回规范化冻结表示；任务
    根、文件或元数据不安全时抛出 ValueError。
    """

    root = Path(task_root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Case 3 task_root 必须是非符号链接目录")
    metadata = validate_case3_prepared_input_metadata(structure_input)
    for filename in CASE3_PREPARED_INPUT_FILENAMES:
        _write_case3_file_atomically(root, filename, metadata["files"][filename]["content"])
    return metadata


def materialize_case3_session_input_files(
    task_root: str | Path,
    workspace: str | Path,
    structure_input: Mapping[str, Any],
) -> None:
    """从任务冻结内容机械复制五文件到一个 session workspace。

    输入是任务根、session workspace 和 TaskCreated 冻结表示；动作仅核对任务内副本
    的摘要并把同样的五个普通文件原子写入 workspace 相对的 ``inputs/case3``，不读
    外部作者目录、raw GJF 或日志。成功无返回值；路径、摘要或文件类型不安全时抛出
    ValueError。
    """

    root = Path(task_root)
    workspace_root = Path(workspace)
    if (
        root.is_symlink()
        or not root.is_dir()
        or workspace_root.is_symlink()
        or not workspace_root.is_dir()
    ):
        raise ValueError("Case 3 task_root/workspace 必须是普通目录")
    metadata = validate_case3_prepared_input_metadata(structure_input)
    task_input_directory = root / "inputs" / "case3"
    if task_input_directory.is_symlink() or not task_input_directory.is_dir():
        write_case3_prepared_input_files(root, metadata)
    for filename in CASE3_PREPARED_INPUT_FILENAMES:
        source = task_input_directory / filename
        expected = metadata["files"][filename]
        if source.is_symlink() or not source.is_file():
            write_case3_prepared_input_files(root, metadata)
            source = task_input_directory / filename
        try:
            source_bytes = source.read_bytes()
        except OSError as exc:
            raise ValueError("Case 3 任务输入文件无法读取") from exc
        if hashlib.sha256(source_bytes).hexdigest() != expected["sha256"]:
            raise ValueError("Case 3 任务输入文件 SHA-256 漂移")
        _write_case3_file_atomically(workspace_root, filename, expected["content"])


def build_case3_workspace_input_paths() -> dict[str, Any]:
    """生成只含 workspace 相对路径的 Case 3 模型可见输入说明。

    输入无；动作只构造固定的四个结构路径、一个映射路径和必要说明，不读取文件或
    写入状态。返回普通 JSON object；本函数没有外部失败条件。
    """

    return {
        "structure_files": {
            identity: f"inputs/case3/{identity}.gjf"
            for identity in ("origin", "ortho", "meta", "para")
        },
        "dihedral_mapping_file": "inputs/case3/dihedral_atoms.json",
        "instructions": (
            "读取 inputs/case3 下四个结构文件和二面角映射文件；请自行选择计算方法、"
            "工作流、扫描范围与步长，模型输入不预设软件设置。"
        ),
    }


__all__ = [
    "CASE3_PREPARED_INPUT_FILENAMES",
    "CASE3_PREPARED_INPUT_SCHEMA",
    "build_case3_workspace_input_paths",
    "load_case3_prepared_input_bundle",
    "materialize_case3_session_input_files",
    "validate_case3_prepared_input_metadata",
    "write_case3_prepared_input_files",
]
