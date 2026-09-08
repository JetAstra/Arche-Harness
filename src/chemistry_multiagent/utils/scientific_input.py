"""校验并冻结调用方提供的计算单元输入。

控制层可以从严格 JSON 文件或 Python mapping 读取一份结构化计算单元。
归一化结果只写入 ``chemistry_context`` 中注册的计算单元、过渡态初始结构和普通计算
初始结构字段，
模型不会看到输入文件路径；作者坐标只会作为显式注册的起始输入进入，工具结果和运行身份
不属于这个输入合同。
"""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping
from typing import Any


_CALCULATION_UNIT_KEYS = frozenset(
    {
        "scope",
        "calculation_scope",
        "reaction_scope",
        "reactants",
        "products",
        "charge",
        "multiplicity",
        "solvent",
        "temperature",
        "temperature_k",
        "needs_ts",
        "needs_irc",
        "transition_state_initial_guess_source",
        "ts_initial_guess_source",
        "initial_guess_source",
        "method_requirements",
    }
)
_TEMPERATURE_PATTERN = re.compile(
    r"^[+-]?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?\s*(?:K|kelvin)$",
    re.IGNORECASE,
)
_SOURCE_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_METHOD_REQUIREMENT_KEYS = frozenset(
    {"endpoint", "endpoints", "transition_state", "irc", "scan"}
)
_METHOD_FIELDS = frozenset(
    {
        "method",
        "basis",
        "solvent",
        "temperature",
        "temperature_k",
        "jobs",
        "calcfc",
        "maxpoints",
        "recalc",
        "symmetry",
    }
)

# 调用方提供的过渡态初始结构不是计算结果，因而不注册能量、频率、标签答案或
# 其它 Gaussian 输出字段。公共合同只暴露原子序数和明确单位为 Å 的坐标；Case 1
# 的身份、数量和来源绑定在作者 owner 中校验。
_INITIAL_TRANSITION_STATE_RECORD_KEYS = frozenset(
    {"atomic_numbers", "coordinates_angstrom", "charge", "multiplicity"}
)
_SUPPLEMENTARY_INPUT_KEYS = frozenset(
    {
        "calculation_unit",
        "initial_transition_state_structures",
        "initial_calculation_structures",
    }
)
_INITIAL_CALCULATION_STRUCTURE_RECORD_KEYS = frozenset(
    {
        "atomic_numbers",
        "coordinates_angstrom",
        "charge",
        "multiplicity",
        "coordinate_definition",
        "coordinate_atom_indices",
        "tool_coordinate_atom_indices",
        "atom_roles",
        "scan",
    }
)
_INITIAL_CALCULATION_SCAN_KEYS = frozenset(
    {
        "coordinate_type",
        "start_degrees",
        "end_degrees",
        "step_degrees",
        "point_count",
        "tool_step_count",
        "tool_value_unit",
        "relaxed",
        "method",
        "basis",
        "solvent_model",
        "symmetry",
    }
)


def _require_text(value: Any, field: str, *, max_length: int = 400) -> str:
    """校验单个文本值并返回去除首尾空白的副本。

    函数不修改输入；值为空、类型错误或超过长度时抛出 ``ValueError``。
    """

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"calculation_unit.{field} 必须是非空字符串")
    normalized = value.strip()
    if len(normalized) > max_length:
        raise ValueError(f"calculation_unit.{field} 超过 {max_length} 个字符")
    return normalized


def _normalize_species(value: Any, field: str) -> list[str]:
    """校验字符串或字符串数组并按原顺序返回 SMILES 副本。

    函数不修改输入；数组为空、成员类型错误或单项过长时抛出 ``ValueError``。
    """

    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        raise ValueError(f"calculation_unit.{field} 必须是字符串或字符串数组")
    if not values or any(not isinstance(item, str) or not item.strip() for item in values):
        raise ValueError(f"calculation_unit.{field} 必须包含非空字符串")
    normalized = [item.strip() for item in values]
    if any(len(item) > 2000 for item in normalized):
        raise ValueError(f"calculation_unit.{field} 单项长度非法")
    return normalized


def _normalize_temperature(value: Any) -> str:
    """把带 K 单位的字符串规范为可读温度值。

    函数不修改输入；单位、数字格式或数值范围非法时抛出 ``ValueError``。
    """

    if not isinstance(value, str) or not value.strip():
        raise ValueError("calculation_unit.temperature 必须是带单位的字符串")
    normalized = value.strip()
    if not _TEMPERATURE_PATTERN.fullmatch(normalized):
        raise ValueError("calculation_unit.temperature 必须包含 K 或 kelvin 单位")
    number = re.match(
        r"^([+-]?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?)\s*(?:K|kelvin)$",
        normalized,
        re.IGNORECASE,
    ).group(1)
    numeric = float(number)
    if not math.isfinite(numeric) or numeric < 0 or numeric > 10000:
        raise ValueError("calculation_unit.temperature 超出允许范围")
    return f"{number} K"


def _normalize_method_requirements(value: Any) -> dict[str, Any]:
    """校验各计算阶段的方法要求并返回规范化副本。

    函数不修改输入；未知阶段、未知字段、冲突温度或非法控制值会抛出
    ``ValueError``。
    """

    if not isinstance(value, Mapping) or not value:
        raise ValueError("calculation_unit.method_requirements 必须是非空 object")
    unknown = set(value) - _METHOD_REQUIREMENT_KEYS
    if unknown:
        raise ValueError(
            f"calculation_unit.method_requirements 含未注册阶段：{sorted(unknown)}"
        )
    output: dict[str, Any] = {}
    for raw_role, raw_requirement in value.items():
        role = "endpoint" if raw_role == "endpoints" else raw_role
        if role in output:
            raise ValueError("calculation_unit.method_requirements 阶段别名冲突")
        if not isinstance(raw_requirement, Mapping) or not raw_requirement:
            raise ValueError(
                f"calculation_unit.method_requirements.{raw_role} 必须是非空 object"
            )
        extra = set(raw_requirement) - _METHOD_FIELDS
        if extra:
            raise ValueError(
                f"calculation_unit.method_requirements.{raw_role} 含未注册字段：{sorted(extra)}"
            )
        normalized: dict[str, Any] = {}
        for field in ("method", "basis", "solvent"):
            if field in raw_requirement:
                normalized[field] = _require_text(
                    raw_requirement[field],
                    f"method_requirements.{raw_role}.{field}",
                    max_length=160,
                )
        has_temperature = "temperature" in raw_requirement
        has_temperature_k = "temperature_k" in raw_requirement
        if has_temperature:
            normalized["temperature"] = _normalize_temperature(
                raw_requirement["temperature"]
            )
        if has_temperature_k:
            item = raw_requirement["temperature_k"]
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ValueError(
                    f"calculation_unit.method_requirements.{raw_role}.temperature_k 必须是有限数字"
                )
            numeric = float(item)
            if not math.isfinite(numeric) or numeric < 0 or numeric > 10000:
                raise ValueError(
                    f"calculation_unit.method_requirements.{raw_role}.temperature_k 超出范围"
                )
            if has_temperature:
                stated = float(normalized["temperature"].split()[0])
                if not math.isclose(stated, numeric, rel_tol=0.0, abs_tol=1e-9):
                    raise ValueError(
                        f"calculation_unit.method_requirements.{raw_role} 的温度字段冲突"
                    )
            normalized["temperature_k"] = item
            if not has_temperature:
                normalized["temperature"] = f"{item:g} K"
        if "jobs" in raw_requirement:
            jobs = raw_requirement["jobs"]
            if not isinstance(jobs, list) or not jobs or any(
                not isinstance(job, str) or not job.strip() for job in jobs
            ):
                raise ValueError(
                    f"calculation_unit.method_requirements.{raw_role}.jobs 必须是字符串数组"
                )
            normalized["jobs"] = list(dict.fromkeys(job.strip() for job in jobs))
        if "calcfc" in raw_requirement:
            if not isinstance(raw_requirement["calcfc"], bool):
                raise ValueError(
                    f"calculation_unit.method_requirements.{raw_role}.calcfc 必须是 boolean"
                )
            normalized["calcfc"] = raw_requirement["calcfc"]
        for field in ("maxpoints", "recalc"):
            if field not in raw_requirement:
                continue
            item = raw_requirement[field]
            if isinstance(item, bool) or not isinstance(item, int) or item < 1 or item > 10000:
                raise ValueError(
                    f"calculation_unit.method_requirements.{raw_role}.{field} 必须是正整数"
                )
            normalized[field] = item
        if "symmetry" in raw_requirement:
            normalized["symmetry"] = _require_text(
                raw_requirement["symmetry"],
                f"method_requirements.{raw_role}.symmetry",
                max_length=80,
            )
        if not normalized:
            raise ValueError(
                f"calculation_unit.method_requirements.{raw_role} 不能为空"
            )
        output[role] = normalized
    return output


def normalize_initial_transition_state_structures(value: Any) -> dict[str, Any]:
    """校验并复制通用的过渡态初始结构映射。

    输入必须是非空 JSON object；每个身份记录只允许 ``atomic_numbers``、
    ``coordinates_angstrom``、``charge`` 和 ``multiplicity``，原子序数与坐标数量
    必须一致，坐标必须是 Nx3 有限数字，电荷和多重度必须是合法整数。返回值是
    可重复归一化的模型可读副本，其中 ``coordinates_angstrom`` 明确以 Å 为单位；
    函数不修改输入、不读取文件，具体 Case 的身份数量、原子数和电子态绑定由
    所属 owner 继续校验。
    """

    if not isinstance(value, Mapping) or not value:
        raise ValueError("initial_transition_state_structures 必须是非空 JSON object")

    normalized: dict[str, Any] = {}
    for identity, record in value.items():
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError("initial_transition_state_structures 的身份必须是非空字符串")
        record = value[identity]
        if not isinstance(record, Mapping):
            raise ValueError(
                f"initial_transition_state_structures.{identity} 必须是 JSON object"
            )
        if set(record) != _INITIAL_TRANSITION_STATE_RECORD_KEYS:
            raise ValueError(
                f"initial_transition_state_structures.{identity} 字段必须严格为 "
                "atomic_numbers、coordinates_angstrom、charge、multiplicity"
            )

        atomnos = record["atomic_numbers"]
        if (
            not isinstance(atomnos, list)
            or not atomnos
            or any(
                isinstance(atomno, bool)
                or not isinstance(atomno, int)
                or not 1 <= atomno <= 118
                for atomno in atomnos
            )
        ):
            raise ValueError(
                f"initial_transition_state_structures.{identity}.atomic_numbers 必须是合法原子序数数组"
            )

        coordinates = record["coordinates_angstrom"]
        if (
            not isinstance(coordinates, list)
            or len(coordinates) != len(atomnos)
        ):
            raise ValueError(
                f"initial_transition_state_structures.{identity}.coordinates_angstrom 必须与原子数一致"
            )
        normalized_coordinates: list[list[float]] = []
        for atom_index, coordinate in enumerate(coordinates):
            if not isinstance(coordinate, list) or len(coordinate) != 3:
                raise ValueError(
                    f"initial_transition_state_structures.{identity}.coordinates_angstrom[{atom_index}] "
                    "必须包含 3 个坐标分量"
                )
            normalized_coordinate: list[float] = []
            for component in coordinate:
                if isinstance(component, bool) or not isinstance(component, (int, float)):
                    raise ValueError(
                        f"initial_transition_state_structures.{identity}.coordinates_angstrom 含非法坐标"
                    )
                numeric = float(component)
                if not math.isfinite(numeric):
                    raise ValueError(
                        f"initial_transition_state_structures.{identity}.coordinates_angstrom 必须是有限坐标"
                    )
                normalized_coordinate.append(numeric)
            normalized_coordinates.append(normalized_coordinate)

        for field, minimum, maximum in (
            ("charge", -20, 20),
            ("multiplicity", 1, 20),
        ):
            item = record[field]
            if (
                isinstance(item, bool)
                or not isinstance(item, int)
                or not minimum <= item <= maximum
            ):
                raise ValueError(
                    f"initial_transition_state_structures.{identity}.{field} 必须是范围内整数"
                )
        normalized[identity] = {
            "atomic_numbers": list(atomnos),
            "coordinates_angstrom": normalized_coordinates,
            "charge": record["charge"],
            "multiplicity": record["multiplicity"],
        }
    return normalized


def normalize_initial_calculation_structures(value: Any) -> dict[str, Any]:
    """校验普通计算的初始结构、坐标定义和 relaxed scan 合同。

    输入必须是非空 JSON object；每个身份记录包含一套首点原子序数/Å 坐标、电子态、
    四个来源 1-based 坐标原子索引、对应的 MiniChem 0-based 索引、原子角色、
    可读坐标定义以及完整 scan 参数。
    该合同只描述待提交计算的已知起点，不接受能量、后续扫描坐标、曲线、排序或结果
    标签。函数不修改输入、不读取文件；具体 Case 对身份、坐标含义和协议值的绑定由
    作者夹具 owner 继续核对。
    """

    if not isinstance(value, Mapping) or not value:
        raise ValueError("initial_calculation_structures 必须是非空 JSON object")

    normalized: dict[str, Any] = {}
    for identity, record in value.items():
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError(
                "initial_calculation_structures 的身份必须是非空字符串"
            )
        if not isinstance(record, Mapping):
            raise ValueError(
                f"initial_calculation_structures.{identity} 必须是 JSON object"
            )
        if set(record) != _INITIAL_CALCULATION_STRUCTURE_RECORD_KEYS:
            raise ValueError(
                f"initial_calculation_structures.{identity} 字段必须严格为 "
                "atomic_numbers、coordinates_angstrom、charge、multiplicity、"
                "coordinate_definition、coordinate_atom_indices、"
                "tool_coordinate_atom_indices、atom_roles、scan"
            )

        atomnos = record["atomic_numbers"]
        if (
            not isinstance(atomnos, list)
            or not atomnos
            or any(
                isinstance(atomno, bool)
                or not isinstance(atomno, int)
                or not 1 <= atomno <= 118
                for atomno in atomnos
            )
        ):
            raise ValueError(
                f"initial_calculation_structures.{identity}.atomic_numbers 必须是合法原子序数数组"
            )

        coordinates = record["coordinates_angstrom"]
        if not isinstance(coordinates, list) or len(coordinates) != len(atomnos):
            raise ValueError(
                f"initial_calculation_structures.{identity}.coordinates_angstrom 必须与原子数一致"
            )
        normalized_coordinates: list[list[float]] = []
        for atom_index, coordinate in enumerate(coordinates):
            if not isinstance(coordinate, list) or len(coordinate) != 3:
                raise ValueError(
                    f"initial_calculation_structures.{identity}.coordinates_angstrom[{atom_index}] "
                    "必须包含 3 个坐标分量"
                )
            normalized_coordinate: list[float] = []
            for component in coordinate:
                if isinstance(component, bool) or not isinstance(component, (int, float)):
                    raise ValueError(
                        f"initial_calculation_structures.{identity}.coordinates_angstrom 含非法坐标"
                    )
                numeric = float(component)
                if not math.isfinite(numeric):
                    raise ValueError(
                        f"initial_calculation_structures.{identity}.coordinates_angstrom 必须是有限坐标"
                    )
                normalized_coordinate.append(numeric)
            normalized_coordinates.append(normalized_coordinate)

        for field, minimum, maximum in (
            ("charge", -20, 20),
            ("multiplicity", 1, 20),
        ):
            item = record[field]
            if (
                isinstance(item, bool)
                or not isinstance(item, int)
                or not minimum <= item <= maximum
            ):
                raise ValueError(
                    f"initial_calculation_structures.{identity}.{field} 必须是范围内整数"
                )

        coordinate_definition = record["coordinate_definition"]
        if (
            not isinstance(coordinate_definition, str)
            or not coordinate_definition.strip()
            or len(coordinate_definition.strip()) > 300
        ):
            raise ValueError(
                f"initial_calculation_structures.{identity}.coordinate_definition 必须是非空短文本"
            )

        coordinate_atom_indices = record["coordinate_atom_indices"]
        if (
            not isinstance(coordinate_atom_indices, list)
            or len(coordinate_atom_indices) != 4
            or len(set(coordinate_atom_indices)) != 4
            or any(
                isinstance(item, bool)
                or not isinstance(item, int)
                or not 1 <= item <= len(atomnos)
                for item in coordinate_atom_indices
            )
        ):
            raise ValueError(
                f"initial_calculation_structures.{identity}.coordinate_atom_indices "
                "必须是四个互异的 1-based 原子索引"
            )

        tool_coordinate_atom_indices = record["tool_coordinate_atom_indices"]
        if (
            not isinstance(tool_coordinate_atom_indices, list)
            or len(tool_coordinate_atom_indices) != 4
            or len(set(tool_coordinate_atom_indices)) != 4
            or any(
                isinstance(item, bool)
                or not isinstance(item, int)
                or not 0 <= item < len(atomnos)
                for item in tool_coordinate_atom_indices
            )
            or tool_coordinate_atom_indices
            != [item - 1 for item in coordinate_atom_indices]
        ):
            raise ValueError(
                f"initial_calculation_structures.{identity}.tool_coordinate_atom_indices "
                "必须是来源 1-based 索引对应的 MiniChem 0-based 索引"
            )

        atom_roles = record["atom_roles"]
        if (
            not isinstance(atom_roles, list)
            or len(atom_roles) != 4
            or any(
                not isinstance(role, str)
                or not role.strip()
                or len(role.strip()) > 40
                for role in atom_roles
            )
        ):
            raise ValueError(
                f"initial_calculation_structures.{identity}.atom_roles 必须是四个非空短文本"
            )

        scan = record["scan"]
        if not isinstance(scan, Mapping) or set(scan) != _INITIAL_CALCULATION_SCAN_KEYS:
            raise ValueError(
                f"initial_calculation_structures.{identity}.scan 字段不完整或含未知字段"
            )
        coordinate_type = scan["coordinate_type"]
        if coordinate_type != "dihedral":
            raise ValueError(
                f"initial_calculation_structures.{identity}.scan.coordinate_type 必须是 dihedral"
            )
        scan_numbers: dict[str, float] = {}
        for field in ("start_degrees", "end_degrees", "step_degrees"):
            item = scan[field]
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ValueError(
                    f"initial_calculation_structures.{identity}.scan.{field} 必须是有限数字"
                )
            numeric = float(item)
            if not math.isfinite(numeric):
                raise ValueError(
                    f"initial_calculation_structures.{identity}.scan.{field} 必须是有限数字"
                )
            scan_numbers[field] = numeric
        if scan_numbers["step_degrees"] <= 0 or scan_numbers["end_degrees"] < scan_numbers["start_degrees"]:
            raise ValueError(
                f"initial_calculation_structures.{identity}.scan 的角度范围非法"
            )
        point_count = scan["point_count"]
        if isinstance(point_count, bool) or not isinstance(point_count, int) or point_count < 2:
            raise ValueError(
                f"initial_calculation_structures.{identity}.scan.point_count 必须是至少 2 的整数"
            )
        expected_point_count = (
            (scan_numbers["end_degrees"] - scan_numbers["start_degrees"])
            / scan_numbers["step_degrees"]
        ) + 1
        if not math.isclose(expected_point_count, point_count, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"initial_calculation_structures.{identity}.scan.point_count 与角度范围不一致"
            )
        tool_step_count = scan["tool_step_count"]
        if (
            isinstance(tool_step_count, bool)
            or not isinstance(tool_step_count, int)
            or tool_step_count != point_count - 1
        ):
            raise ValueError(
                f"initial_calculation_structures.{identity}.scan.tool_step_count "
                "必须等于 point_count - 1"
            )
        if scan["tool_value_unit"] != "degree":
            raise ValueError(
                f"initial_calculation_structures.{identity}.scan.tool_value_unit "
                "必须是 degree"
            )
        if not isinstance(scan["relaxed"], bool):
            raise ValueError(
                f"initial_calculation_structures.{identity}.scan.relaxed 必须是 boolean"
            )
        normalized_scan: dict[str, Any] = {
            "coordinate_type": coordinate_type,
            "start_degrees": scan["start_degrees"],
            "end_degrees": scan["end_degrees"],
            "step_degrees": scan["step_degrees"],
            "point_count": point_count,
            "tool_step_count": tool_step_count,
            "tool_value_unit": scan["tool_value_unit"],
            "relaxed": scan["relaxed"],
        }
        for field in ("method", "basis", "solvent_model", "symmetry"):
            item = scan[field]
            if not isinstance(item, str) or not item.strip() or len(item.strip()) > 120:
                raise ValueError(
                    f"initial_calculation_structures.{identity}.scan.{field} 必须是非空短文本"
                )
            normalized_scan[field] = item.strip()

        normalized[identity] = {
            "atomic_numbers": list(atomnos),
            "coordinates_angstrom": normalized_coordinates,
            "charge": record["charge"],
            "multiplicity": record["multiplicity"],
            "coordinate_definition": coordinate_definition.strip(),
            "coordinate_atom_indices": list(coordinate_atom_indices),
            "tool_coordinate_atom_indices": list(tool_coordinate_atom_indices),
            "atom_roles": [role.strip() for role in atom_roles],
            "scan": normalized_scan,
        }
    return normalized


def normalize_calculation_unit(value: Any, *, require_nonempty: bool = True) -> dict[str, Any]:
    """校验计算单元并返回稳定字段。

    输入可以是严格 JSON 文件中的 ``calculation_unit`` 对象，也可以是已经取出的
    对象。未知字段、文件路径、文献坐标或运行控制字段都会被拒绝；该函数不访问
    文件、不调用模型，也不把调用方路径写入返回值。
    """

    if not isinstance(value, Mapping):
        raise ValueError("calculation_unit 必须是 JSON object")
    unknown = set(value) - _CALCULATION_UNIT_KEYS
    if unknown:
        raise ValueError(f"calculation_unit 含未注册字段：{sorted(unknown)}")
    if require_nonempty and not value:
        raise ValueError("calculation_unit 不能为空")

    normalized: dict[str, Any] = {}
    aliases = {
        "calculation_scope": "scope",
        "reaction_scope": "scope",
        "ts_initial_guess_source": "transition_state_initial_guess_source",
        "initial_guess_source": "transition_state_initial_guess_source",
    }
    for key, item in value.items():
        canonical_key = aliases.get(key, key)
        if canonical_key in normalized and normalized[canonical_key] != item:
            raise ValueError(f"calculation_unit.{canonical_key} 的别名值冲突")
        normalized[canonical_key] = item

    output: dict[str, Any] = {}
    if "scope" in normalized:
        output["scope"] = _require_text(normalized["scope"], "scope")
    for field in ("reactants", "products"):
        if field in normalized:
            output[field] = _normalize_species(normalized[field], field)
    for field, minimum, maximum in (
        ("charge", -20, 20),
        ("multiplicity", 1, 20),
    ):
        if field not in normalized:
            continue
        item = normalized[field]
        if isinstance(item, bool) or not isinstance(item, int) or not minimum <= item <= maximum:
            raise ValueError(f"calculation_unit.{field} 必须是范围内整数")
        output[field] = item
    for field in ("solvent",):
        if field in normalized:
            output[field] = _require_text(normalized[field], field, max_length=120)

    has_temperature = "temperature" in normalized
    has_temperature_k = "temperature_k" in normalized
    if has_temperature:
        output["temperature"] = _normalize_temperature(normalized["temperature"])
    if has_temperature_k:
        item = normalized["temperature_k"]
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError("calculation_unit.temperature_k 必须是有限数字")
        numeric = float(item)
        if not math.isfinite(numeric) or numeric < 0 or numeric > 10000:
            raise ValueError("calculation_unit.temperature_k 超出允许范围")
        if has_temperature:
            stated = float(output["temperature"].split()[0])
            if not math.isclose(stated, numeric, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError("calculation_unit.temperature 与 temperature_k 数值冲突")
        output["temperature_k"] = item
        if not has_temperature:
            output["temperature"] = f"{item:g} K"

    for field in ("needs_ts", "needs_irc"):
        if field in normalized:
            if not isinstance(normalized[field], bool):
                raise ValueError(f"calculation_unit.{field} 必须是 boolean")
            output[field] = normalized[field]
    if "transition_state_initial_guess_source" in normalized:
        source = _require_text(
            normalized["transition_state_initial_guess_source"],
            "transition_state_initial_guess_source",
            max_length=128,
        )
        if _SOURCE_PATTERN.fullmatch(source) is None:
            raise ValueError(
                "calculation_unit.transition_state_initial_guess_source 必须是公开 producer 标识"
            )
        output["transition_state_initial_guess_source"] = source
    if "method_requirements" in normalized:
        output["method_requirements"] = _normalize_method_requirements(
            normalized["method_requirements"]
        )

    if require_nonempty and not output:
        raise ValueError("calculation_unit 不能为空")
    return output


def normalize_supplementary_scientific_input(value: Any) -> dict[str, Any]:
    """把可选调用方输入包装为固定科学上下文对象。

    ``None`` 返回空对象；其他输入会被深层校验且不会原地修改。非法根结构或
    计算单元、过渡态初始结构或普通计算初始结构字段会抛出 ``ValueError``。
    """

    if value is None or (isinstance(value, Mapping) and not value):
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("supplementary scientific input 必须是 JSON object")
    if set(value) & _SUPPLEMENTARY_INPUT_KEYS:
        unknown = set(value) - _SUPPLEMENTARY_INPUT_KEYS
        if unknown:
            raise ValueError(
                f"supplementary scientific input 含未注册字段：{sorted(unknown)}"
            )
        output: dict[str, Any] = {}
        if "calculation_unit" in value:
            output["calculation_unit"] = normalize_calculation_unit(
                value["calculation_unit"]
            )
        if "initial_transition_state_structures" in value:
            output["initial_transition_state_structures"] = (
                normalize_initial_transition_state_structures(
                    value["initial_transition_state_structures"]
                )
            )
        if "initial_calculation_structures" in value:
            output["initial_calculation_structures"] = (
                normalize_initial_calculation_structures(
                    value["initial_calculation_structures"]
                )
            )
        if not output:
            raise ValueError("supplementary scientific input 不能为空")
        return output
    return {"calculation_unit": normalize_calculation_unit(value)}


def project_calculation_unit_from_context(value: Any) -> dict[str, Any]:
    """从冻结科学上下文提取可恢复的计算单元副本。

    输入不是对象或计算单元不合法时返回空对象；函数不修改输入，也不抛出校验
    异常。
    """

    if not isinstance(value, Mapping):
        return {}
    unit = value.get("calculation_unit")
    if not isinstance(unit, Mapping):
        return {}
    try:
        return {"calculation_unit": normalize_calculation_unit(unit)}
    except ValueError:
        return {}


def project_supplementary_scientific_input_from_context(value: Any) -> dict[str, Any]:
    """从冻结 ``chemistry_context`` 提取计算单元和初始结构的安全副本。

    输入不是对象、任一注册字段不合法或坐标合同不完整时返回空对象；函数只读
    输入，不访问文件，也不会把历史运行字段投影回新任务。
    """

    if not isinstance(value, Mapping):
        return {}
    candidate: dict[str, Any] = {}
    if "calculation_unit" in value:
        candidate["calculation_unit"] = value["calculation_unit"]
    if "initial_transition_state_structures" in value:
        candidate["initial_transition_state_structures"] = value[
            "initial_transition_state_structures"
        ]
    if "initial_calculation_structures" in value:
        candidate["initial_calculation_structures"] = value[
            "initial_calculation_structures"
        ]
    if not candidate:
        return {}
    try:
        return normalize_supplementary_scientific_input(candidate)
    except ValueError:
        return {}


def merge_supplementary_chemistry_context(
    retrieved_context: Any,
    supplementary_context: Any,
) -> dict[str, Any]:
    """合并 Retrieval 结果与调用方科学输入并返回独立副本。

    Retrieval 字段保持原值，调用方注册字段覆盖同名字段；函数不修改任一输入，
    调用方科学输入不符合合同时抛出 ``ValueError``。
    """

    merged = copy.deepcopy(dict(retrieved_context)) if isinstance(retrieved_context, Mapping) else {}
    if (
        isinstance(supplementary_context, Mapping)
        and "calculation_unit" not in supplementary_context
        and "initial_transition_state_structures" not in supplementary_context
        and "initial_calculation_structures" not in supplementary_context
    ):
        # 历史 TaskCreated 的 chemistry_context 可能只有 Retrieval 兼容字段；
        # 它不是新的调用方输入，不能按 calculation_unit 重新解释。
        provided = {}
    else:
        provided = normalize_supplementary_scientific_input(supplementary_context)
    if provided:
        for field in (
            "calculation_unit",
            "initial_transition_state_structures",
            "initial_calculation_structures",
        ):
            if field in provided:
                merged[field] = copy.deepcopy(provided[field])
    return merged


def has_calculation_unit(value: Any) -> bool:
    """复验输入并返回其中是否含合法非空计算单元，不修改输入或抛出异常。"""

    return bool(project_calculation_unit_from_context(value))


__all__ = [
    "has_calculation_unit",
    "merge_supplementary_chemistry_context",
    "normalize_calculation_unit",
    "normalize_initial_calculation_structures",
    "normalize_initial_transition_state_structures",
    "normalize_supplementary_scientific_input",
    "project_calculation_unit_from_context",
    "project_supplementary_scientific_input_from_context",
]
