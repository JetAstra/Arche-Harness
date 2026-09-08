"""Shared backend implementation helpers with workspace confinement."""

from __future__ import annotations

import ctypes
import importlib.metadata
import json
import math
import os
import re
import signal
import shlex
import stat
import shutil
import subprocess
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ..artifacts import ArtifactStore, relative_workspace_path, resolve_workspace_path, workspace_root
from ..paths import PROJECT_ROOT
from ..process_identity import (
    ProcessIdentity,
    capture_process_identity,
    inspect_process_group_membership,
    process_identity_activity,
)
from ..resources import is_resource_reference, resolve_resource_reference
from ..resource_budget import (
    _create_unique_private_temp,
    _ensure_private_directory as _ensure_stable_private_directory,
    _open_private_directory_descriptor,
    _verify_private_directory_identity,
)


PRIVATE_DIRECTORY_MODE = 0o700
PRIVATE_FILE_MODE = 0o600
EXTERNAL_ENVIRONMENT_NAMES = frozenset(
    {
        "PATH",
        "LD_LIBRARY_PATH",
        "PYTHONPATH",
        "PYTHONNOUSERSITE",
        "PYTHONDONTWRITEBYTECODE",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TZ",
        "TERM",
        "CUDA_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "SLURM_JOB_ID",
        "SLURM_JOB_NODELIST",
        "SLURM_CPUS_PER_TASK",
        "SLURM_GPUS",
        "LM_LICENSE_FILE",
        "MLM_LICENSE_FILE",
        "OMP_NUM_THREADS",
        "OMP_STACKSIZE",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "KMP_STACKSIZE",
        "OMPI_MCA_hwloc_base_cpu_list",
        "PRTE_MCA_hwloc_default_cpu_list",
        "MINICHEM_BACKEND_RUNTIME",
        "MINICHEM_MODEL_CACHE",
        "MINICHEM_EXECUTION_JOB_ID",
        "MINICHEM_EXECUTION_JOB_DIRECTORY",
        "MINICHEM_JOB_ROOT",
        "MINICHEM_JOB_INPUTS",
        "MINICHEM_JOB_OUTPUTS",
        "MINICHEM_JOB_REPORT",
        "MINICHEM_WORKER_CPU_IDS",
        "MINICHEM_WORKER_CPU_COUNT",
        "XDG_CACHE_HOME",
        "GAUSS_SCRDIR",
        "GAUSS_EXEDIR",
        "g16root",
    }
)
EXTERNAL_ENVIRONMENT_PREFIXES = (
    "MINICHEM_",
    "GAUSS_",
    "ORCA_",
    "OPENMOLCAS_",
    "MOLCAS_",
    "CREST_",
    "CHEMGRAPH_",
    "OMP_",
    "MKL_",
    "OPENBLAS_",
    "NUMEXPR_",
    "VECLIB_",
    "CUDA_",
    "ROCR_",
    "OMPI_",
    "PRTE_",
    "SLURM_",
    "LC_",
)
SENSITIVE_ENV_MARKERS = (
    "API_KEY",
    "ACCESS_KEY",
    "SECRET",
    "TOKEN",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
    "PRIVATE_KEY",
    "AUTHORIZATION",
    "CAPABILITY",
)

_PIDFD_OPEN_SYSCALL = 434
_PIDFD_SEND_SIGNAL_SYSCALL = 424
_MISSING_ATTRIBUTE = object()
_RUNTIME_CACHE_ENVIRONMENT_NAMES = frozenset(
    {
        "MINICHEM_MODEL_CACHE",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_XET_CACHE",
        "TORCH_HOME",
        "FAIRCHEM_CACHE_DIR",
    }
)


def success(
    result: Any,
    *,
    artifact_files: list[dict[str, str]] | None = None,
    backend_version: str | None = None,
    warnings: list[str] | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": "success",
        "result": result,
        "artifact_files": artifact_files or [],
        "backend_version": backend_version,
        "warnings": warnings or [],
        "provenance": provenance or {},
        "retryable": False,
    }


def partial_success(
    result: Any,
    *,
    artifact_files: list[dict[str, str]] | None = None,
    backend_version: str | None = None,
    warnings: list[str] | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return completed backend output whose primary scientific result is incomplete.

    This status is deliberately distinct from ``success``: raw logs or a partially
    parsed result must never masquerade as a complete Action result.
    """

    return {
        "status": "partial_success",
        "result": result,
        "artifact_files": artifact_files or [],
        "backend_version": backend_version,
        "warnings": warnings or [],
        "provenance": provenance or {},
        "retryable": False,
    }


def unavailable(message: str, *, install: str = "") -> dict[str, Any]:
    error = {"code": "backend_unavailable", "message": message}
    if install:
        error["install"] = install
    return {"status": "unavailable", "error": error, "retryable": False}


def unsupported(message: str) -> dict[str, Any]:
    return {
        "status": "unsupported",
        "error": {"code": "unsupported_request", "message": message},
        "retryable": False,
    }


def failed(message: str, *, code: str = "backend_failed", retryable: bool = False) -> dict[str, Any]:
    return {
        "status": "failed",
        "error": {"code": code, "message": message},
        "retryable": retryable,
    }


def module_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def request_parts(request: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    return (
        dict(request.get("inputs") or {}),
        dict(request.get("method_spec") or {}),
        dict(request.get("action_settings") or {}),
    )


def unwrap_artifact(value: Any) -> Any:
    if isinstance(value, dict) and "artifact_id" in value:
        return ArtifactStore().load(value)
    if isinstance(value, str) and value.startswith("art_"):
        return ArtifactStore().load(value)
    return value


def resolve_input_file(value: Any) -> Path:
    item = unwrap_artifact(value)
    if is_resource_reference(item):
        return resolve_resource_reference(item)
    if isinstance(item, dict) and isinstance(item.get("path"), str):
        return resolve_workspace_path(item["path"], must_exist=True)
    if isinstance(item, dict) and isinstance(item.get("file_path"), str):
        return resolve_workspace_path(item["file_path"], must_exist=True)
    if isinstance(item, str):
        return resolve_workspace_path(item, must_exist=True)
    raise ValueError(
        "Expected a workspace file path, file ArtifactRef, or explicit registered ResourceRef"
    )


def output_directory(action_id: str, backend_id: str) -> Path:
    directory = workspace_root() / "outputs" / action_id / backend_id / uuid.uuid4().hex[:12]
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def write_json(directory: Path, name: str, value: Any) -> Path:
    """原子写入 Backend JSON，并同步文件和父目录。"""

    path = directory / name
    _atomic_write_bytes(
        path,
        (json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n").encode(
            "utf-8"
        ),
    )
    return path


def _parse_xyz(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"Empty XYZ file: {path}")
    try:
        count = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError(f"Invalid XYZ atom count: {path}") from exc
    atoms = []
    for line in lines[2 : 2 + count]:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Invalid XYZ atom line: {line!r}")
        atoms.append(
            {
                "element": parts[0],
                "position_angstrom": [float(parts[1]), float(parts[2]), float(parts[3])],
            }
        )
    if len(atoms) != count:
        raise ValueError(
            f"XYZ atom count mismatch: {path} declares {count} atoms but only "
            f"{len(atoms)} coordinate rows were parsed. Supply one complete XYZ frame; "
            "split multi-frame trajectories and select the intended conformer explicitly."
        )
    comment = lines[1] if len(lines) > 1 else ""
    charge_match = re.search(r"(?:^|\s)charge=(-?\d+)(?:\s|$)", comment)
    multiplicity_match = re.search(
        r"(?:^|\s)multiplicity=(\d+)(?:\s|$)", comment
    )
    return {
        "atoms": atoms,
        "charge": int(charge_match.group(1)) if charge_match else 0,
        "multiplicity": int(multiplicity_match.group(1)) if multiplicity_match else 1,
        "pbc": [False, False, False],
        "source_path": relative_workspace_path(path),
    }


def _parse_pdb(path: Path) -> dict[str, Any]:
    atoms = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        element = line[76:78].strip()
        if not element:
            element = "".join(character for character in line[12:16].strip() if character.isalpha())[:2].title()
        atoms.append(
            {
                "element": element,
                "position_angstrom": [
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                ],
            }
        )
    if not atoms:
        raise ValueError(f"PDB contains no ATOM/HETATM coordinates: {path}")
    return {
        "atoms": atoms,
        "charge": 0,
        "multiplicity": 1,
        "pbc": [False, False, False],
        "source_path": relative_workspace_path(path),
    }


def _sdf_structure_payload(
    path: Path,
    *,
    atoms: list[dict[str, Any]],
    bonds: list[dict[str, Any]],
) -> dict[str, Any]:
    """根据已校验的原子和键记录组装 AtomicStructure。

    输入是任务工作区内的 SDF 路径以及解析出的原子、键列表；函数只在内存中
    组装结构，不写入或修改文件。返回带坐标、键、电荷、自旋和来源相对路径的
    AtomicStructure；原子列表为空时抛出 ValueError，避免把空结构交给下游 Action。
    """

    if not atoms:
        raise ValueError(f"SDF/MOL file contains no atoms: {path}")
    return {
        "atoms": atoms,
        "bonds": bonds,
        "charge": sum(int(atom.get("formal_charge", 0)) for atom in atoms),
        "multiplicity": 1,
        "pbc": [False, False, False],
        "source_path": relative_workspace_path(path),
    }


def _parse_sdf_fixed_integer(
    line: str,
    start: int,
    end: int,
    *,
    label: str,
    path: Path,
) -> int:
    """读取一行 V2000 固定列整数。

    输入是原始记录、列范围、字段名称和来源路径；函数只转换文本，不产生文件
    副作用。返回解析后的整数；列为空、格式错误或整数非法时抛出 ValueError，
    让调用方拒绝不完整的科学输入。
    """

    token = line[start:end].strip()
    try:
        return int(token)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {label} in SDF/MOL file: {path}") from exc


def _parse_sdf_fixed_float(
    line: str,
    start: int,
    end: int,
    *,
    label: str,
    path: Path,
) -> float:
    """读取一行 V2000 固定列坐标。

    输入是原始记录、列范围、字段名称和来源路径；函数只转换文本，不产生文件
    副作用。返回有限浮点数；格式错误或结果为 NaN/无穷大时抛出 ValueError，
    防止无效坐标进入 AtomicStructure。
    """

    token = line[start:end].strip()
    try:
        value = float(token)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {label} in SDF/MOL file: {path}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite in SDF/MOL file: {path}")
    return value


def _parse_v2000_sdf(lines: list[str], path: Path) -> dict[str, Any]:
    """解析一个严格的 V2000 单分子记录。

    输入是已从任务内文件读取的记录行和来源路径；函数解析坐标、键和电荷属性，
    只在内存中构造结果，不写回 SDF。返回完整 AtomicStructure；counts、记录、
    引用端点或数值格式不合法时抛出 ValueError，拒绝继续执行。
    """

    if len(lines) < 4:
        raise ValueError(f"Invalid V2000 SDF/MOL file: {path}")
    atom_count = _parse_sdf_fixed_integer(
        lines[3], 0, 3, label="V2000 atom count", path=path
    )
    bond_count = _parse_sdf_fixed_integer(
        lines[3], 3, 6, label="V2000 bond count", path=path
    )
    if atom_count <= 0:
        raise ValueError(f"V2000 SDF/MOL file contains no atoms: {path}")
    if bond_count < 0 or len(lines) < 4 + atom_count + bond_count:
        raise ValueError(f"V2000 SDF/MOL counts do not match records: {path}")

    atoms: list[dict[str, Any]] = []
    charge_codes = {1: 3, 2: 2, 3: 1, 5: -1, 6: -2, 7: -3}
    for index, line in enumerate(lines[4 : 4 + atom_count]):
        if len(line) < 34:
            raise ValueError(f"V2000 atom record {index + 1} is truncated: {path}")
        element = line[31:34].strip()
        if not element:
            raise ValueError(f"V2000 atom record {index + 1} has no element: {path}")
        atom: dict[str, Any] = {
            "element": element,
            "position_angstrom": [
                _parse_sdf_fixed_float(
                    line,
                    0,
                    10,
                    label=f"V2000 atom {index + 1} x coordinate",
                    path=path,
                ),
                _parse_sdf_fixed_float(
                    line,
                    10,
                    20,
                    label=f"V2000 atom {index + 1} y coordinate",
                    path=path,
                ),
                _parse_sdf_fixed_float(
                    line,
                    20,
                    30,
                    label=f"V2000 atom {index + 1} z coordinate",
                    path=path,
                ),
            ],
        }
        charge_code = _parse_sdf_fixed_integer(
            line,
            36,
            39,
            label=f"V2000 atom {index + 1} charge code",
            path=path,
        ) if len(line) >= 39 and line[36:39].strip() else 0
        if charge_code not in {0, *charge_codes}:
            raise ValueError(
                f"Unsupported V2000 atom charge code {charge_code} in {path}"
            )
        if charge_code:
            atom["formal_charge"] = charge_codes[charge_code]
        atoms.append(atom)

    bonds: list[dict[str, Any]] = []
    bond_orders = {1: 1.0, 2: 2.0, 3: 3.0, 4: 1.5}
    bond_start = 4 + atom_count
    for index, line in enumerate(lines[bond_start : bond_start + bond_count]):
        if len(line) < 9:
            raise ValueError(f"V2000 bond record {index + 1} is truncated: {path}")
        first = _parse_sdf_fixed_integer(
            line, 0, 3, label=f"V2000 bond {index + 1} first atom", path=path
        )
        second = _parse_sdf_fixed_integer(
            line, 3, 6, label=f"V2000 bond {index + 1} second atom", path=path
        )
        bond_type = _parse_sdf_fixed_integer(
            line, 6, 9, label=f"V2000 bond {index + 1} type", path=path
        )
        if not 1 <= first <= atom_count or not 1 <= second <= atom_count:
            raise ValueError(f"V2000 bond {index + 1} references an unknown atom: {path}")
        if bond_type not in bond_orders:
            raise ValueError(f"Unsupported V2000 bond type {bond_type} in {path}")
        bonds.append(
            {
                "atom_index_a": first - 1,
                "atom_index_b": second - 1,
                "bond_order": bond_orders[bond_type],
                **({"bond_type": "aromatic"} if bond_type == 4 else {}),
            }
        )

    for line in lines[bond_start + bond_count :]:
        fields = line.split()
        if len(fields) < 3 or fields[:2] != ["M", "CHG"]:
            continue
        try:
            pair_count = int(fields[2])
        except ValueError as exc:
            raise ValueError(f"Invalid V2000 M  CHG record: {path}") from exc
        if pair_count < 0 or len(fields) != 3 + 2 * pair_count:
            raise ValueError(f"Invalid V2000 M  CHG record: {path}")
        for offset in range(pair_count):
            atom_index = int(fields[3 + 2 * offset])
            charge = int(fields[4 + 2 * offset])
            if not 1 <= atom_index <= atom_count:
                raise ValueError(f"V2000 M  CHG references an unknown atom: {path}")
            atoms[atom_index - 1]["formal_charge"] = charge

    return _sdf_structure_payload(path, atoms=atoms, bonds=bonds)


def _v3000_section_records(lines: list[str], section: str, path: Path) -> list[str]:
    """提取一个 V3000 节的记录并拼接标准续行。

    输入是已读取的 V3000 行、节名称和来源路径；函数只筛选并拼接记录，不写入
    文件。返回去掉 ``M  V30`` 前缀的记录列表；节嵌套、缺失、续行未结束或包含
    非法行时抛出 ValueError，避免静默丢失结构字段。
    """

    begin = f"M  V30 BEGIN {section}"
    end = f"M  V30 END {section}"
    records: list[str] = []
    active = False
    continuation = ""
    for line in lines:
        stripped = line.strip()
        if stripped == begin:
            if active:
                raise ValueError(f"Nested V3000 {section} section: {path}")
            active = True
            continue
        if stripped == end:
            if continuation:
                raise ValueError(f"Unterminated V3000 continuation in {path}")
            if not active:
                raise ValueError(f"V3000 {section} section ended before it began: {path}")
            active = False
            continue
        if not active:
            continue
        prefix = "M  V30 "
        if not stripped.startswith(prefix):
            raise ValueError(f"Invalid V3000 {section} record: {path}")
        payload = stripped[len(prefix) :]
        if continuation:
            payload = continuation + " " + payload
            continuation = ""
        if payload.endswith("-"):
            continuation = payload[:-1].rstrip()
        else:
            records.append(payload)
    if active or continuation:
        raise ValueError(f"Unterminated V3000 {section} section: {path}")
    return records


def _parse_v3000_sdf(lines: list[str], path: Path) -> dict[str, Any]:
    """解析一个严格的 V3000 单分子记录。

    输入是已从任务内文件读取的记录行和来源路径；函数解析 ``COUNTS``、原子、
    坐标、键端点、芳香/配位键和电荷属性，且不写入或修改文件。返回完整
    AtomicStructure；counts、属性、原子身份、键身份或坐标不合法时抛出 ValueError。
    """

    counts = [line.strip() for line in lines if line.strip().startswith("M  V30 COUNTS ")]
    if len(counts) != 1:
        raise ValueError(f"V3000 SDF/MOL must contain exactly one COUNTS record: {path}")
    count_fields = counts[0][len("M  V30 COUNTS ") :].split()
    if len(count_fields) < 2:
        raise ValueError(f"Invalid V3000 COUNTS record: {path}")
    try:
        atom_count, bond_count = int(count_fields[0]), int(count_fields[1])
    except ValueError as exc:
        raise ValueError(f"Invalid V3000 COUNTS record: {path}") from exc
    if atom_count <= 0 or bond_count < 0:
        raise ValueError(f"Invalid V3000 atom/bond counts: {path}")

    atom_records = _v3000_section_records(lines, "ATOM", path)
    bond_records = _v3000_section_records(lines, "BOND", path)
    if len(atom_records) != atom_count or len(bond_records) != bond_count:
        raise ValueError(f"V3000 COUNTS do not match records: {path}")

    atoms: list[dict[str, Any]] = []
    atom_positions: dict[int, int] = {}
    for record in atom_records:
        fields = record.split()
        if len(fields) < 6:
            raise ValueError(f"Invalid V3000 atom record: {path}")
        try:
            atom_id = int(fields[0])
            coordinates = [float(value) for value in fields[2:5]]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid V3000 atom record: {path}") from exc
        if atom_id <= 0 or atom_id in atom_positions:
            raise ValueError(f"V3000 atom ids must be positive and unique: {path}")
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError(f"V3000 atom coordinates must be finite: {path}")
        element = fields[1]
        if not element:
            raise ValueError(f"V3000 atom has no element: {path}")
        atom: dict[str, Any] = {
            "element": element,
            "position_angstrom": coordinates,
        }
        try:
            atom_map_number = int(fields[5])
        except ValueError as exc:
            raise ValueError(f"Invalid V3000 atom-atom mapping number: {path}") from exc
        if atom_map_number:
            atom["atom_map_number"] = atom_map_number
        for attribute in fields[6:]:
            if "=" not in attribute:
                raise ValueError(f"Invalid V3000 atom attribute {attribute!r}: {path}")
            name, value = attribute.split("=", 1)
            if name == "CHG":
                atom["formal_charge"] = int(value)
            elif name == "MASS":
                atom["isotope"] = int(value)
            elif name == "RAD":
                atom["radical_electrons"] = int(value)
        atom_positions[atom_id] = len(atoms)
        atoms.append(atom)

    bond_orders = {1: 1.0, 2: 2.0, 3: 3.0, 4: 1.5, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0}
    bonds: list[dict[str, Any]] = []
    bond_ids: set[int] = set()
    for record in bond_records:
        fields = record.split()
        if len(fields) < 4:
            raise ValueError(f"Invalid V3000 bond record: {path}")
        try:
            bond_id, bond_type = int(fields[0]), int(fields[1])
            first, second = int(fields[2]), int(fields[3])
        except ValueError as exc:
            raise ValueError(f"Invalid V3000 bond record: {path}") from exc
        if bond_id <= 0 or bond_id in bond_ids:
            raise ValueError(f"V3000 bond ids must be positive and unique: {path}")
        if first not in atom_positions or second not in atom_positions:
            raise ValueError(f"V3000 bond references an unknown atom: {path}")
        if bond_type not in bond_orders:
            raise ValueError(f"Unsupported V3000 bond type {bond_type} in {path}")
        bond: dict[str, Any] = {
            "atom_index_a": atom_positions[first],
            "atom_index_b": atom_positions[second],
            "bond_order": bond_orders[bond_type],
        }
        if bond_type == 4:
            bond["bond_type"] = "aromatic"
        elif bond_type == 9:
            bond["bond_type"] = "dative"
        bonds.append(bond)
        bond_ids.add(bond_id)

    return _sdf_structure_payload(path, atoms=atoms, bonds=bonds)


def _parse_sdf(path: Path) -> dict[str, Any]:
    """严格解析一个任务内的单结构 SDF/MOL Artifact。

    输入是已由 ArtifactStore 路径校验得到的文件路径；函数读取文件并按 V2000 或
    V3000 解析为完整 AtomicStructure，不修改原文件。返回带坐标、键、电子状态和
    相对来源的结构；多记录、空结构、counts 不一致或格式错误时抛出 ValueError。
    """

    text = path.read_text(encoding="utf-8", errors="replace")
    records = [record for record in text.split("$$$$") if record.strip()]
    if len(records) != 1:
        raise ValueError(
            f"SDF/MOL Artifact must contain exactly one molecule record: {path}"
        )
    lines = records[0].splitlines()
    if len(lines) < 4:
        raise ValueError(f"Invalid SDF/MOL file: {path}")
    if "V3000" in lines[3] or any(
        line.strip().startswith("M  V30 COUNTS ") for line in lines
    ):
        return _parse_v3000_sdf(lines, path)
    return _parse_v2000_sdf(lines, path)


def _parse_vasp_structure(path: Path) -> dict[str, Any]:
    try:
        from pymatgen.io.vasp.inputs import Poscar
    except (ImportError, ModuleNotFoundError):
        from ase.io import read

        return structure_from_atoms(read(str(path), format="vasp")) | {
            "source_path": relative_workspace_path(path)
        }

    structure = Poscar.from_file(path, check_for_potcar=False).structure
    return {
        "atoms": [
            {
                "element": str(site.specie.symbol),
                "position_angstrom": [float(value) for value in site.coords],
            }
            for site in structure
        ],
        "cell_angstrom": [
            [float(value) for value in row] for row in structure.lattice.matrix
        ],
        "charge": 0,
        "multiplicity": 1,
        "pbc": [True, True, True],
        "source_path": relative_workspace_path(path),
    }


def structure_dict(value: Any) -> dict[str, Any]:
    reference = None
    if isinstance(value, str) and value.startswith("art_"):
        reference = ArtifactStore().find(value)
    elif isinstance(value, dict) and isinstance(value.get("artifact_id"), str):
        reference = ArtifactStore().find(str(value["artifact_id"]))
    if reference is not None and any(
        token in reference.semantic_type.casefold()
        for token in ("ensemble", "trajectory")
    ):
        raise ValueError(
            f"A single structure is required, but ArtifactRef {reference.artifact_id} has "
            f"semantic_type={reference.semantic_type!r}. Select one conformer/frame and pass "
            "its AtomicStructure ArtifactRef or export one complete structure file."
        )
    item = unwrap_artifact(value)
    if isinstance(item, dict) and "result" in item and isinstance(item["result"], dict):
        item = item["result"]
    if isinstance(item, dict) and "structure" in item and isinstance(item["structure"], dict):
        item = item["structure"]
    if isinstance(item, dict):
        if "atoms" in item or "symbols" in item or "smiles" in item:
            return dict(item)
        if isinstance(item.get("path"), str):
            return structure_dict(item["path"])
    if isinstance(item, str):
        candidate = resolve_workspace_path(item, must_exist=False)
        if candidate.is_file():
            if candidate.suffix.lower() == ".xyz":
                return _parse_xyz(candidate)
            if candidate.suffix.lower() == ".pdb":
                return _parse_pdb(candidate)
            if candidate.suffix.lower() in {".sdf", ".mol"}:
                return _parse_sdf(candidate)
            if candidate.suffix.lower() in {".vasp", ".poscar"} or candidate.name.upper() in {
                "POSCAR",
                "CONTCAR",
            }:
                try:
                    return _parse_vasp_structure(candidate)
                except Exception as exc:
                    raise ValueError(
                        f"Could not parse VASP structure file {candidate}: {exc}"
                    ) from exc
            try:
                from ase.io import read

                return structure_from_atoms(read(str(candidate)))
            except Exception as exc:
                raise ValueError(f"Could not parse structure file {candidate}: {exc}") from exc
        return {"smiles": item}
    raise ValueError(
        "Unsupported structure representation. Pass an AtomicStructure mapping, SMILES, a "
        "supported structure file path, or an ArtifactRef whose semantic type is a single "
        "structure; ensemble and trajectory artifacts require explicit frame selection."
    )


def atoms_and_coordinates(structure: dict[str, Any]) -> tuple[list[str], list[list[float]]]:
    if "atoms" in structure:
        atoms = structure["atoms"]
        if not isinstance(atoms, list) or not atoms:
            raise ValueError("AtomicStructure.atoms must be a non-empty array of atom mappings")
        symbols = []
        coordinates = []
        for index, atom in enumerate(atoms):
            if not isinstance(atom, dict):
                raise ValueError(f"AtomicStructure.atoms[{index}] must be a mapping")
            symbol = atom.get("element") or atom.get("symbol")
            if not symbol:
                raise ValueError(
                    f"AtomicStructure.atoms[{index}] requires an element field"
                )
            position = atom.get("position_angstrom")
            if position is None:
                position = atom.get("position")
            if position is None:
                suffix = "; the key 'xyz' is not part of AtomicStructure" if "xyz" in atom else ""
                raise ValueError(
                    f"AtomicStructure.atoms[{index}] requires position_angstrom=[x, y, z]"
                    f"{suffix}"
                )
            try:
                row = [float(value) for value in position]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"AtomicStructure.atoms[{index}].position_angstrom must contain three numbers"
                ) from exc
            symbols.append(str(symbol))
            coordinates.append(row)
    else:
        symbols = [str(value) for value in structure.get("symbols", [])]
        coordinates = [
            [float(value) for value in row]
            for row in (
                structure.get("coordinates_angstrom")
                or structure.get("positions_angstrom")
                or structure.get("positions")
                or []
            )
        ]
    if not symbols or len(symbols) != len(coordinates):
        raise ValueError("Structure must contain aligned symbols and coordinates")
    if any(len(row) != 3 or not all(math.isfinite(value) for value in row) for row in coordinates):
        raise ValueError("All coordinates must contain three finite Angstrom values")
    return symbols, coordinates


def ase_atoms(value: Any):
    from ase import Atoms

    structure = structure_dict(value)
    if "smiles" in structure and len(structure) == 1:
        raise ValueError("A 3D AtomicStructure is required; SMILES alone is insufficient")
    symbols, coordinates = atoms_and_coordinates(structure)
    atoms = Atoms(symbols=symbols, positions=coordinates)
    cell = structure.get("cell_angstrom") or structure.get("cell")
    if cell is not None:
        atoms.set_cell(cell)
    pbc = structure.get("pbc", [False, False, False])
    atoms.set_pbc(pbc)
    atoms.info["charge"] = int(structure.get("charge", 0))
    atoms.info["multiplicity"] = int(structure.get("multiplicity", 1))
    return atoms


def structure_from_atoms(atoms) -> dict[str, Any]:
    return {
        "atoms": [
            {
                "element": symbol,
                "position_angstrom": [float(value) for value in position],
            }
            for symbol, position in zip(atoms.get_chemical_symbols(), atoms.get_positions())
        ],
        "cell_angstrom": [[float(value) for value in row] for row in atoms.cell.array],
        "pbc": [bool(value) for value in atoms.pbc],
        "charge": int(atoms.info.get("charge", 0)),
        "multiplicity": int(atoms.info.get("multiplicity", 1)),
    }


def write_xyz(structure_value: Any, path: Path) -> Path:
    """原子写入 XYZ 结构文件，避免覆盖 symlink 指向的外部文件。"""

    structure = structure_dict(structure_value)
    symbols, coordinates = atoms_and_coordinates(structure)
    charge = int(structure.get("charge", 0))
    multiplicity = int(structure.get("multiplicity", 1))
    lines = [
        str(len(symbols)),
        (
            "generated by ResearchChemBench "
            f"charge={charge} multiplicity={multiplicity}"
        ),
    ]
    lines.extend(
        f"{symbol} {row[0]:.12f} {row[1]:.12f} {row[2]:.12f}"
        for symbol, row in zip(symbols, coordinates)
    )
    _atomic_write_bytes(path, ("\n".join(lines) + "\n").encode("utf-8"))
    return path


def atom_spec(structure_value: Any) -> str:
    structure = structure_dict(structure_value)
    symbols, coordinates = atoms_and_coordinates(structure)
    return "; ".join(
        f"{symbol} {row[0]:.12f} {row[1]:.12f} {row[2]:.12f}"
        for symbol, row in zip(symbols, coordinates)
    )


def resolve_command(executable: str, environment_variable: str | None = None) -> list[str] | None:
    configured = os.environ.get(environment_variable or "", "").strip()
    if configured:
        command = shlex.split(configured)
        if command:
            executable_path = Path(command[0]).expanduser()
            if executable_path.is_file():
                command[0] = str(executable_path.resolve())
                return command
            if not executable_path.is_absolute():
                # Native-software profiles store portable paths relative to the
                # copied MiniChem toolbox.  The service is often launched from
                # the repository root, so checking only cwd makes an installed
                # cache appear missing. Resolve against both locations while
                # keeping the configured path itself portable.
                for base in (PROJECT_ROOT, Path.cwd()):
                    candidate = (base / executable_path).resolve()
                    if candidate.is_file():
                        command[0] = str(candidate)
                        return command
            resolved = shutil.which(command[0])
            if resolved:
                command[0] = resolved
                return command
    path = shutil.which(executable)
    return [path] if path else None


def _is_sensitive_environment_name(name: str) -> bool:
    """判断环境变量名是否包含明显的认证或秘密材料。"""

    upper_name = name.upper()
    return any(marker in upper_name for marker in SENSITIVE_ENV_MARKERS)


def _external_environment_name_is_allowed(name: str) -> bool:
    """判断环境变量是否属于外部科学程序的最小继承白名单。"""

    # 模型缓存路径只允许由 runtime profile 解析后注入，不能沿用服务进程
    # 或调用方直接提供的同名环境变量。
    if name in _RUNTIME_CACHE_ENVIRONMENT_NAMES:
        return False
    return name in EXTERNAL_ENVIRONMENT_NAMES or (
        not _is_sensitive_environment_name(name)
        and name.startswith(EXTERNAL_ENVIRONMENT_PREFIXES)
    )


def _open_pidfd_for_signal(pid: int) -> int | None:
    """打开稳定的进程句柄；标准库缺失时通过 Linux syscall 补齐。"""

    native_open = getattr(os, "pidfd_open", _MISSING_ATTRIBUTE)
    if native_open is None:
        return None
    try:
        if callable(native_open):
            return int(native_open(pid, 0))
        if sys.platform != "linux":
            return None
        libc = ctypes.CDLL(None, use_errno=True)
        descriptor = int(
            libc.syscall(
                ctypes.c_long(_PIDFD_OPEN_SYSCALL),
                ctypes.c_int(pid),
                ctypes.c_uint(0),
            )
        )
        if descriptor < 0:
            error_number = ctypes.get_errno()
            raise OSError(error_number, os.strerror(error_number))
        return descriptor
    except (OSError, TypeError, ValueError):
        return None


def _send_pidfd_signal(descriptor: int, received_signal: signal.Signals) -> bool:
    """通过标准库或 Linux syscall 向已打开的 pidfd 发信号。"""

    native_send = getattr(signal, "pidfd_send_signal", _MISSING_ATTRIBUTE)
    if native_send is None:
        return False
    try:
        if callable(native_send):
            native_send(descriptor, int(received_signal), None, 0)
            return True
        if sys.platform != "linux":
            return False
        libc = ctypes.CDLL(None, use_errno=True)
        result = int(
            libc.syscall(
                ctypes.c_long(_PIDFD_SEND_SIGNAL_SYSCALL),
                ctypes.c_int(descriptor),
                ctypes.c_int(int(received_signal)),
                ctypes.c_void_p(0),
                ctypes.c_uint(0),
            )
        )
        if result < 0:
            error_number = ctypes.get_errno()
            raise OSError(error_number, os.strerror(error_number))
        return True
    except (OSError, TypeError, ValueError):
        return False


def _validated_runtime_cache_environment(
    environment: dict[str, str],
    explicit: dict[str, str],
) -> dict[str, str]:
    """仅保留 runtime profile 声明且位于其模型缓存根下的缓存路径。"""

    runtime_name = str(environment.get("MINICHEM_BACKEND_RUNTIME") or "").strip()
    if not runtime_name:
        return {}
    try:
        from ..runtime import runtime_environment

        profile = runtime_environment(runtime_name)
    except (KeyError, OSError, RuntimeError, ValueError):
        if any(name in explicit for name in _RUNTIME_CACHE_ENVIRONMENT_NAMES):
            raise OSError("trusted runtime profile is unavailable for cache paths")
        return {}
    root_value = profile.get("MINICHEM_MODEL_CACHE")
    if not isinstance(root_value, str) or not root_value.strip():
        return {}
    root = Path(root_value).expanduser().resolve()
    values: dict[str, str] = {}
    for name in _RUNTIME_CACHE_ENVIRONMENT_NAMES:
        raw_value = profile.get(name)
        if not isinstance(raw_value, str) or not raw_value.strip():
            if name in explicit:
                raise OSError(f"runtime profile does not declare cache path: {name}")
            continue
        candidate = Path(raw_value).expanduser().resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise OSError(
                f"runtime cache path escapes MINICHEM_MODEL_CACHE: {name}"
            ) from exc
        explicit_value = explicit.get(name)
        if explicit_value is not None and Path(explicit_value).expanduser().resolve() != candidate:
            raise OSError(f"runtime profile cache path is overridden: {name}")
        values[name] = str(candidate)
    return values


def _ensure_private_directory(path: Path) -> None:
    """逐级创建或核验外部程序私有 scratch 目录，拒绝祖先 symlink。"""

    try:
        _ensure_stable_private_directory(path)
    except (OSError, RuntimeError) as exc:
        raise OSError(f"external scratch path is not a real directory: {path}") from exc


def _fsync_parent_directory(path: Path) -> None:
    """用稳定目录描述符同步 Backend 输出目录元数据。"""

    descriptor, absolute = _open_private_directory_descriptor(path, create=False)
    try:
        _verify_private_directory_identity(absolute, descriptor)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    """用稳定 dirfd、唯一临时文件和双重 fsync 写入 Backend 输出。"""

    parent = path.parent
    _ensure_private_directory(parent)
    parent_descriptor, absolute_parent = _open_private_directory_descriptor(
        parent,
        create=False,
    )
    descriptor = -1
    temporary_name: str | None = None
    try:
        _verify_private_directory_identity(absolute_parent, parent_descriptor)
        try:
            target_metadata = os.stat(
                path.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            target_metadata = None
        if target_metadata is not None and stat.S_ISLNK(target_metadata.st_mode):
            raise OSError(f"Backend output cannot be a symlink: {path}")
        if target_metadata is not None and not stat.S_ISREG(target_metadata.st_mode):
            raise OSError(f"Backend output must be a regular file: {path}")
        descriptor, temporary_name = _create_unique_private_temp(
            parent_descriptor,
            f".{path.name}.",
        )
        os.fchmod(descriptor, PRIVATE_FILE_MODE)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _verify_private_directory_identity(absolute_parent, parent_descriptor)
        try:
            target_metadata = os.stat(
                path.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            target_metadata = None
        if target_metadata is not None and stat.S_ISLNK(target_metadata.st_mode):
            raise OSError(f"Backend output became a symlink: {path}")
        os.replace(
            temporary_name,
            path.name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        temporary_name = None
        os.fsync(parent_descriptor)
        _verify_private_directory_identity(absolute_parent, parent_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_name is not None:
            try:
                os.unlink(temporary_name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
        os.close(parent_descriptor)


def _build_external_environment(
    directory: Path, environment_overrides: dict[str, str] | None
) -> dict[str, str]:
    """构造最小外部程序环境并固定到当前 Action 的 scratch 目录。"""

    if directory.is_symlink() or not directory.is_dir():
        raise OSError(f"external working directory is invalid: {directory}")
    environment = {
        name: value
        for name, value in os.environ.items()
        if _external_environment_name_is_allowed(name)
    }
    explicit = {
        str(name): str(value)
        for name, value in (environment_overrides or {}).items()
        if str(name).strip()
    }
    explicit_cache_names = sorted(
        name for name in explicit if name in _RUNTIME_CACHE_ENVIRONMENT_NAMES
    )
    environment.update(
        {
            name: value
            for name, value in explicit.items()
            if name not in _RUNTIME_CACHE_ENVIRONMENT_NAMES
        }
    )
    if explicit_cache_names and not str(environment.get("MINICHEM_BACKEND_RUNTIME") or "").strip():
        raise OSError(
            "runtime profile is required for external cache environment: "
            + ", ".join(explicit_cache_names)
        )
    environment.update(_validated_runtime_cache_environment(environment, explicit))
    if "PATH" not in environment:
        environment["PATH"] = os.defpath
    temporary = directory / ".tmp"
    home = directory / ".home"
    matplotlib = temporary / "matplotlib"
    _ensure_private_directory(temporary)
    _ensure_private_directory(home)
    _ensure_private_directory(matplotlib)
    environment.update(
        {
            "TMPDIR": str(temporary),
            "HOME": str(home),
            "MPLCONFIGDIR": str(matplotlib),
        }
    )
    if "GAUSS_SCRDIR" in explicit:
        _ensure_private_directory(Path(environment["GAUSS_SCRDIR"]))
    elif environment.get("GAUSS_SCRDIR"):
        scratch = temporary / "gaussian"
        _ensure_private_directory(scratch)
        environment["GAUSS_SCRDIR"] = str(scratch)
    return environment


def _process_identity_matches(
    expected: ProcessIdentity, current: ProcessIdentity
) -> bool:
    """核对 PID、启动时钟、boot、PGID 和 session 是否仍为同一进程。"""

    if (
        expected.start_time_ticks is None
        or expected.boot_id is None
        or expected.process_group_id is None
        or expected.session_id is None
    ):
        return False
    return all(
        (
            expected.pid == current.pid,
            expected.start_time_ticks == current.start_time_ticks,
            expected.boot_id == current.boot_id,
            expected.process_group_id == current.process_group_id,
            expected.session_id == current.session_id,
        )
    )


def _capture_live_process_identity(process: subprocess.Popen[str]) -> ProcessIdentity | None:
    """读取并二次核对 child 身份；无法证明仍存活时返回 None。"""

    expected = capture_process_identity(process.pid)
    if (
        expected.start_time_ticks is None
        or not expected.boot_id
        or expected.process_group_id is None
        or expected.session_id is None
        or process_identity_activity(expected.as_dict()) != "alive"
    ):
        return None
    current = capture_process_identity(process.pid)
    return expected if _process_identity_matches(expected, current) else None


def _signal_verified_process_group(
    process: subprocess.Popen[str], received_signal: signal.Signals
) -> bool:
    """仅在身份二次核对后向外部程序进程组发信号，避免 PGID 复用误杀。"""

    if os.name != "posix":
        try:
            process.terminate() if received_signal == signal.SIGTERM else process.kill()
        except (ProcessLookupError, PermissionError, OSError):
            return False
        return True
    identity = _capture_live_process_identity(process)
    if identity is None or identity.process_group_id is None:
        return False
    descriptor = _open_pidfd_for_signal(identity.pid)
    if descriptor is None:
        return False
    try:
        # pidfd 指向已经核对过的 leader，避免身份检查和 killpg 之间的 PGID
        # 复用竞态；descendant 未被证明清理时上层必须保持未知状态。
        current = capture_process_identity(identity.pid)
        if not _process_identity_matches(identity, current):
            return False
        return _send_pidfd_signal(descriptor, received_signal)
    except (OSError, ValueError, ProcessLookupError, PermissionError):
        return False
    finally:
        os.close(descriptor)


def _communicate_after_verified_termination(
    process: subprocess.Popen[str],
    identity: ProcessIdentity | None = None,
) -> tuple[str, str, bool]:
    """有界收集终止后的输出，并返回进程组是否已确认清空。"""

    identity = identity or _capture_live_process_identity(process)

    def cleanup_verified() -> bool:
        if process.poll() is None or identity is None:
            return False
        return (
            inspect_process_group_membership(
                identity.process_group_id,
                identity.session_id,
                identity.boot_id,
            )
            == "empty"
        )

    try:
        stdout, stderr = process.communicate(timeout=5)
        return stdout, stderr, cleanup_verified()
    except subprocess.TimeoutExpired as first_timeout:
        kill_sent = _signal_verified_process_group(process, signal.SIGKILL)
        try:
            stdout, stderr = process.communicate(timeout=5 if kill_sent else 1)
            return stdout, stderr, cleanup_verified()
        except subprocess.TimeoutExpired as second_timeout:
            stdout = second_timeout.stdout or first_timeout.stdout or ""
            stderr = second_timeout.stderr or first_timeout.stderr or ""
            stderr = (
                f"{stderr}\nExternal process cleanup remained pending because its "
                "identity could not be verified."
            )
            return stdout, stderr, False


def _set_private_process_umask() -> None:
    """将外部科学程序新建文件默认限制为 owner 可读写。"""

    os.umask(0o077)


def _coerce_process_output_text(value: Any) -> str:
    """把 subprocess 输出统一为文本，保留超时分支的诊断内容。"""

    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    return str(value)


def run_external(
    *,
    executable: str,
    arguments: list[str],
    directory: Path,
    environment_variable: str | None = None,
    stdin_text: str | None = None,
    timeout_seconds: int = 1800,
    environment_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    command = resolve_command(executable, environment_variable)
    if command is None:
        return {
            "available": False,
            "returncode": None,
            "stdout": "",
            "stderr": f"Executable {executable!r} was not found",
            "command": [executable, *arguments],
        }
    environment = _build_external_environment(directory, environment_overrides)
    process = subprocess.Popen(
        [*command, *arguments],
        cwd=directory,
        stdin=subprocess.PIPE if stdin_text is not None else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        # Scientific launchers frequently spawn worker processes.  Give every
        # call its own process group so a timeout cannot leave an unobserved
        # ORCA/orca_plot child consuming resources and writing late artifacts.
        start_new_session=os.name == "posix",
        preexec_fn=_set_private_process_umask if os.name == "posix" else None,
    )

    try:
        stdout, stderr = process.communicate(stdin_text, timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        identity = _capture_live_process_identity(process)
        _signal_verified_process_group(process, signal.SIGTERM)
        stdout, stderr, cleanup_verified = _communicate_after_verified_termination(
            process,
            identity,
        )
        result = {
            "available": True,
            "returncode": 124 if cleanup_verified else None,
            "stdout": _coerce_process_output_text(stdout),
            "stderr": _coerce_process_output_text(stderr),
            "command": [Path(command[0]).name, *command[1:], *arguments],
            "timeout": True,
        }
        if not cleanup_verified:
            result.update({"cleanup_pending": True, "resource_state": "unknown"})
        return result
    except BaseException:
        # SIGINT/SIGTERM propagated by a cancelled evaluation must not leave a
        # detached Gaussian/ORCA/MPI process group running on the worker.
        identity = _capture_live_process_identity(process)
        _signal_verified_process_group(process, signal.SIGTERM)
        _communicate_after_verified_termination(process, identity)
        raise
    return {
        "available": True,
        "returncode": process.returncode,
        "stdout": _coerce_process_output_text(stdout),
        "stderr": _coerce_process_output_text(stderr),
        "command": [Path(command[0]).name, *command[1:], *arguments],
    }


def command_artifacts(directory: Path) -> list[dict[str, str]]:
    artifacts = []
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            artifacts.append(
                {
                    "path": relative_workspace_path(path),
                    "semantic_type": "BackendFile",
                    "media_type": "text/plain" if path.suffix.lower() in {".txt", ".log", ".out", ".xyz", ".pdb", ".inp"} else "application/octet-stream",
                }
            )
    return artifacts
