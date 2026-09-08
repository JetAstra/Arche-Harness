"""构造 Gaussian 专家角色使用的两个固定科学事实视图。

本模块集中构造 Expert/General 模型输入。调用方只提供当前问题、假设、规范
workflow 和执行结果；实现按固定字段重新构造对象，不复制正式步骤对象或
ActionResult。顶层和步骤字段始终稳定，未知结果类型只形成不可用状态，不会把
原始日志、路径或任意 JSON 直接交给模型。
"""

from __future__ import annotations

import json
import math
import re
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

EXPERT_VIEW_PROJECTOR_VERSION = "expert-views.v2"


PRE_VIEW_KEYS = (
    "scientific_question",
    "explicit_conditions",
    "hypothesis",
    "gaussian_unit",
)
POST_VIEW_KEYS = (
    "scientific_question",
    "explicit_conditions",
    "hypothesis",
    "workflow_evidence",
)
STEP_VIEW_KEYS = (
    "step_ref",
    "name",
    "scientific_purpose",
    "depends_on",
    "operation",
    "target",
    "native_preflight",
    "scientific_settings",
    "expected_observables",
    "status",
    "coverage",
    "observations",
    "program_checks",
    "failure",
    "not_executed",
)

_FAILURE_DETAIL_FIELDS = (
    "stage",
    "message",
    "repair_guidance",
    "input_requirements",
    "retryable",
    "retryable_after_input_correction",
    "retry_requires_changed_request",
    "status",
    "diagnostic_code",
    "reason_code",
)

_FORBIDDEN_FIELD_PARTS = frozenset(
    {
        "artifact",
        "backend",
        "command",
        "executable",
        "hash",
        "log",
        "path",
        "runtime",
        "step_id",
        "action_id",
        "arguments",
        "traceback",
        "request",
        "receipt",
        "operation_id",
        "job_id",
        "metadata",
        "catalog",
        "provenance",
        "source_step",
        "digest",
        "checksum",
        "sha",
        "version",
        "ref",
        "id",
    }
)
_FORBIDDEN_EXACT_FIELDS = frozenset(
    {
        "raw",
        "raw_output",
        "raw_log",
        "stdout",
        "stderr",
        "traceback",
        "transport",
        "provider",
        "credential",
        "secret",
        "token",
        "password",
    }
)
# 路径判断不能只列举常见工作目录：任务可能把文件放在任意 POSIX 根下。
# 负向前瞻要求斜线后紧跟路径段，负向后查避免把 Gaussian route 中的
# ``B3LYP/6-31G(d)`` 或普通 ``energy / volume`` 科学表达式当成路径。
_POSIX_ABSOLUTE_PATH_TOKEN = (
    r"(?<![A-Za-z0-9_.:/-])/(?![/*\s])"
    r"[^\\\s,;:)}\]>\"'，；：。！？、）】》]+"
)
_FILE_URI_TOKEN = (
    r"(?<![A-Za-z0-9_.-])file://"
    r"[^\\\s,;:)}\]>\"'，；：。！？、）】》]+"
)
# Windows 盘符路径和 UNC 共享路径必须作为整体匹配。若只匹配 POSIX 路径的
# ``/private/x`` 后缀，会把 ``D:/private/x`` 清理成残留的 ``D:``，反而继续
# 暴露盘符；先匹配下面两个完整形式，再处理 POSIX 根路径。
_WINDOWS_PATH_SEGMENT = r"[^\\/\s,;:)}\]>\"'，；：。！？、）】》]+"
_WINDOWS_DRIVE_PATH_TOKEN = (
    r"(?<![A-Za-z0-9_.-])[A-Za-z]:[\\/]"
    rf"{_WINDOWS_PATH_SEGMENT}(?:[\\/]{_WINDOWS_PATH_SEGMENT})*"
)
_UNC_PATH_TOKEN = (
    r"(?<![A-Za-z0-9_.-])\\\\"
    rf"{_WINDOWS_PATH_SEGMENT}(?:[\\/]{_WINDOWS_PATH_SEGMENT})+"
    r"|(?<![A-Za-z0-9_.:-])//"
    rf"{_WINDOWS_PATH_SEGMENT}(?:[\\/]{_WINDOWS_PATH_SEGMENT})+"
)
_WINDOWS_PATH_TOKEN = rf"(?:{_WINDOWS_DRIVE_PATH_TOKEN}|{_UNC_PATH_TOKEN})"
_WINDOWS_DRIVE_PREFIX_TOKEN = r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/]"
_ABSOLUTE_PATH_TOKEN = rf"(?:{_WINDOWS_PATH_TOKEN}|{_FILE_URI_TOKEN}|{_POSIX_ABSOLUTE_PATH_TOKEN})"
_ABSOLUTE_PATH_TEXT = re.compile(
    rf"(?:{_ABSOLUTE_PATH_TOKEN}|{_WINDOWS_DRIVE_PREFIX_TOKEN})",
    re.IGNORECASE,
)
_UNSAFE_DERIVED_TEXT = re.compile(
    rf"(?:{_ABSOLUTE_PATH_TOKEN}|{_WINDOWS_DRIVE_PREFIX_TOKEN}"
    r"|(?<![A-Za-z0-9_])(?:code|inputs|outputs|artifacts|working_memory|reports?|logs?)/[^ ,;)}\]]*"
    r"|\b(?:raw[_ -]?(?:output|log)|stdout|stderr|traceback|stack\s+trace)\b"
    r"|\b(?:job[_ -]?ids?|operation[_ -]?instance[_ -]?id|operation[_ -]?ids?|"
    r"artifact[_ -]?ids?|backend[_ -]?ids?|execution[_ -]?ids?|step[_ -]?ids?|"
    r"action[_ -]?ids?|artifact[_ -]?(?:hash|digest|checksum))\b"
    r"|\b(?:command|executable|working\s+directory|absolute\s+path|log(?:ging)?|runtime|"
    r"transport|provider|secret|token|password|credential|api[_ -]?key)\s*[:=]"
    r"|(?<![A-Za-z0-9_])[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}(?![A-Za-z0-9])"
    r"|(?<![A-Za-z0-9])[0-9a-f]{32,128}(?![A-Za-z0-9])"
    r")",
    re.IGNORECASE,
)
_FAILURE_PATH_TEXT = re.compile(
    rf"{_ABSOLUTE_PATH_TOKEN}|{_WINDOWS_DRIVE_PREFIX_TOKEN}"
    r"|(?<![A-Za-z0-9_])(?:code|inputs|outputs|artifacts|working_memory|reports?|logs?)/[^ ,;)}\]]*",
    re.IGNORECASE,
)
_FAILURE_INTERNAL_TEXT = re.compile(
    r"\b(?:artifact[_ -]?ids?|backend[_ -]?ids?|event[_ -]?ids?|"
    r"execution[_ -]?ids?|job[_ -]?ids?|operation[_ -]?instance[_ -]?id|"
    r"operation[_ -]?ids?|record[_ -]?ids?|step[_ -]?ids?|"
    r"raw[_ -]?(?:output|log)|stdout|stderr|traceback|runtime|command|"
    r"executable|action[_ -]?ids?|artifact[_ -]?(?:hash|digest|checksum))\b"
    r"(?:\s*[-:=]\s*[A-Za-z0-9][A-Za-z0-9_.-]*)?"
    r"|(?<![A-Za-z0-9])"
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    r"(?![A-Za-z0-9])"
    r"|(?<![A-Za-z0-9])[0-9a-f]{32,128}(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_MISSING = object()
_FORBIDDEN_OBSERVABLE_NAMES = frozenset({"gaussian_input_deck", "input_deck", "deck", "raw_output"})
_PROGRAMMABLE_ALLOWED_OBSERVABLES = frozenset(
    {
        "energy",
        "electronic_energy",
        "electronic_energies",
        "optimized_geometry",
        "final_atomic_structure",
        "vibrational_frequencies",
        "vibrational_intensities",
        "frequencies_cm1",
        "ir_intensities",
        "raman_intensities",
        "relative_gibbs_free_energies",
        "activation_free_energies",
        "rate_determining_step",
        "delta_delta_g",
        "enantiomeric_ratio",
        "enantiomeric_excess",
        "observed_support",
        "temperature",
        "temperature_c",
        "temperature_k",
        "temperature_kelvin",
        "pressure",
        "pressure_pa",
        "solvent",
        "method",
        "basis",
        "spin",
        "geometry",
    }
)

# Expert 只需要当前科学问题的明确条件。条件名必须先归一化到这个小集合；
# workflow、Action 或 Memory 中的任意内部字段不会因为恰好带有 name/value
# 就进入模型。别名用于兼容历史 chemistry_context 和已有条件记录。
_CONDITION_NAME_ALIASES = {
    "solvent": "solvent",
    "solvent_name": "solvent",
    "temperature": "temperature",
    "temperature_c": "temperature",
    "temperature_celsius": "temperature",
    "temperature_k": "temperature",
    "temperature_kelvin": "temperature",
    "kelvin": "temperature",
    "temp": "temperature",
    "pressure": "pressure",
    "pressure_pa": "pressure",
    "pressure_kpa": "pressure",
    "pressure_mpa": "pressure",
    "pressure_bar": "pressure",
    "pressure_atm": "pressure",
    "concentration": "concentration",
    "concentration_m": "concentration",
    "concentration_molar": "concentration",
    "molarity": "concentration",
    "ph": "pH",
    "p_h": "pH",
    "time": "time",
    "time_s": "time",
    "time_ms": "time",
    "duration": "time",
    "duration_s": "time",
    "duration_ms": "time",
    "reaction_time": "time",
    "atmosphere": "atmosphere",
    "atmospheric": "atmosphere",
    "atmosphere_gas": "atmosphere",
    "gas": "atmosphere",
    "environment": "atmosphere",
    # 这两个字段是现有 Gaussian 任务真实输入中的公开科学量。
    "charge": "charge",
    "multiplicity": "multiplicity",
    "spin_multiplicity": "multiplicity",
}
_CONDITION_DEFAULT_UNITS = {
    "temperature_c": "°C",
    "temperature_celsius": "°C",
    "temperature_k": "K",
    "temperature_kelvin": "K",
    "kelvin": "K",
    "pressure_pa": "Pa",
    "pressure_kpa": "kPa",
    "pressure_mpa": "MPa",
    "pressure_bar": "bar",
    "pressure_atm": "atm",
    "concentration_m": "M",
    "concentration_molar": "M",
    "molarity": "M",
    "time_s": "s",
    "time_ms": "ms",
    "duration_s": "s",
    "duration_ms": "ms",
}
_CONDITION_ALLOWED_UNITS = {
    "temperature": frozenset({"C", "°C", "degC", "K", "kelvin", "F", "°F", "degF"}),
    "pressure": frozenset({"Pa", "kPa", "MPa", "bar", "atm", "mbar"}),
    "concentration": frozenset({"M", "mM", "uM", "μM", "nM", "mol/L", "mol l-1"}),
    "time": frozenset({"s", "ms", "min", "h", "hr", "hour", "hours"}),
    "solvent": frozenset(),
    "pH": frozenset({"pH"}),
    "atmosphere": frozenset(),
    "charge": frozenset(),
    "multiplicity": frozenset(),
}
_CONDITION_UNSAFE_TEXT = re.compile(
    rf"(?:{_ABSOLUTE_PATH_TOKEN}|{_WINDOWS_DRIVE_PREFIX_TOKEN}"
    r"|\b(?:raw[_ -]?(?:output|log)|stdout|stderr|traceback|stack\s+trace|exception)\b"
    r"|\b(?:artifact|backend|event|execution|job|operation|record|step|combination|"
    r"hypothesis|request)[_ -]?id\b"
    r"|\b(?:transport|provider|secret|token|password|credential|api[_ -]?key)\s*[:=]"
    r"|\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"
    r"|(?<![A-Za-z0-9_])(?:inputs?|outputs?|artifacts?|working_memory|logs?)/"
    r")",
    re.IGNORECASE,
)
_CONDITION_REJECTED = object()
_VALID_ELEMENT_SYMBOLS = frozenset(
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
    "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce "
    "Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn "
    "Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl "
    "Mc Lv Ts Og"
    .split()
)
_SCIENTIFIC_OBSERVATION_NESTED_KEYS = frozenset(
    {
        "value",
        "unit",
        "conditions",
        "status",
        "ready",
        "calculation_intent",
        "link1_segment_count",
        "checks",
        "summary_status",
        "role",
        "input_format",
        "formula",
        "molecular_formula",
        "atom_count",
        "n_atoms",
        "element_counts",
        "atoms",
        "atom",
        "element",
        "atomic_number",
        "elements",
        "position_angstrom",
        "coordinates",
        "cell_angstrom",
        "pbc",
        "x",
        "y",
        "z",
        "shape",
        "element_count",
        "finite",
        "symmetric",
        "total_count",
        "real_count",
        "imaginary_count",
        "frequency_unit",
        "imaginary_frequencies_cm1",
        "real",
        "imaginary",
        "line_frequencies_cm1",
        "line_intensities",
        "intensity_unit",
        "broadening",
        "fwhm_cm1",
        "point_count",
        "stick_count",
        "charge",
        "multiplicity",
        "energy",
        "electronic_energy",
        "electronic_energies",
        "relative_gibbs_free_energies",
        "activation_free_energies",
        "rate_determining_step",
        "delta_delta_g",
        "enantiomeric_ratio",
        "enantiomeric_excess",
        "label",
        "population",
        "population_percent",
        "observed_support",
    }
) | _VALID_ELEMENT_SYMBOLS


def _as_text(value: Any, default: str = "") -> str:
    """把可见说明转换为完整字符串；非字符串值使用稳定空说明。"""

    return value if isinstance(value, str) else default


def _safe_derived_text(value: Any, default: str = "") -> str:
    """保留受控科学说明的完整正文，发现路径或运行日志标记时返回空说明。"""

    if not isinstance(value, str) or _UNSAFE_DERIVED_TEXT.search(value):
        return default
    return value


def _sanitize_failure_text(value: Any, default: str = "") -> str:
    """清理失败说明中的绝对路径和内部运行身份。

    输入是待展示的错误文本和缺省值；动作替换路径与运行身份占位符，不修改原文。
    返回值是可交给专家的诊断正文，输入不是字符串时返回 ``default``；清理不会
    抛出异常，也不会恢复被移除的内部内容。
    """

    if not isinstance(value, str):
        return default
    sanitized = _FAILURE_PATH_TEXT.sub("[removed-path]", value)
    return _FAILURE_INTERNAL_TEXT.sub("[removed-internal]", sanitized)


def _safe_program_value(value: Any) -> Any:
    """筛除程序检查/失败细节中的内部字段和值，不影响科学正文。"""

    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        projected: dict[str, Any] = {}
        for key, item in value.items():
            if (
                not isinstance(key, str)
                or _is_forbidden_field(key)
                or key not in _PROGRAM_CHECK_VALUE_KEYS
            ):
                continue
            safe = _safe_program_value(item)
            if safe is not None:
                projected[key] = safe
        return projected
    if isinstance(value, list):
        return [
            safe
            for item in value
            if (safe := _safe_program_value(item)) is not None
        ]
    if isinstance(value, tuple):
        return [
            safe
            for item in value
            if (safe := _safe_program_value(item)) is not None
        ]
    if isinstance(value, str):
        safe = _safe_derived_text(value, None)
        return safe if safe else None
    return _safe_scalar(value)


def _is_forbidden_field(name: str) -> bool:
    """判断字段名是否表示正式身份、路径或运行技术信息。"""

    lowered = name.casefold().replace("-", "_")
    if lowered in _FORBIDDEN_EXACT_FIELDS or re.match(
        r"^(?:raw(?:_|$)|transport(?:_|$)|provider(?:_|$)|credential(?:_|$)|"
        r"secret(?:_|$)|token(?:_|$)|password(?:_|$))",
        lowered,
    ):
        return True
    if any(part in lowered.split("_") for part in _FORBIDDEN_FIELD_PARTS):
        return True
    return bool(re.search(r"(?:^|_)(?:id|ref|path|hash|version)$", lowered))


def _safe_scalar(value: Any, *, allow_text: bool = True) -> Any:
    """复制请求可见的 JSON 标量。

    输入是任意值及是否允许文本的标记；动作保留布尔、整数、字符串和浮点标量，
    清除路径或内部运行文本，不修改输入。返回值是安全标量或 ``None``；非有限数
    暂留给严格 JSON 边界报告协议错误，调用方应据此停止发送无效请求。
    """

    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        # 请求参数中的非有限数保留到 JSON 编码边界，由调用方报告严格 JSON
        # 协议错误；执行结果等科学事实使用 _safe_scientific_value 先按缺失处理。
        return value
    if allow_text and isinstance(value, str):
        if _ABSOLUTE_PATH_TEXT.search(value) or _UNSAFE_DERIVED_TEXT.search(value):
            return None
        return value
    return None


def _safe_value(value: Any, *, _seen: set[int] | None = None) -> Any:
    """复制已注册科学值，遇到循环引用时只在该分支返回缺失。"""

    scalar = _safe_scalar(value)
    if scalar is not None or value is None:
        return scalar
    if _seen is None:
        _seen = set()
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in _seen:
            return None
        _seen.add(identity)
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or _is_forbidden_field(key):
                continue
            safe = _safe_value(item, _seen=_seen)
            if safe is not None or item is None:
                result[key] = safe
        _seen.remove(identity)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        identity = id(value)
        if identity in _seen:
            return None
        _seen.add(identity)
        result = []
        for item in value:
            safe = _safe_value(item, _seen=_seen)
            if safe is not None or item is None:
                result.append(safe)
        _seen.remove(identity)
        return result
    return None


def _safe_scientific_value(
    value: Any,
    *,
    _seen: set[int] | None = None,
    _allowed_keys: frozenset[str] | None = None,
) -> Any:
    """按科学结果字段白名单复制观测值。

    输入是待记录的结果值、递归访问集合和可选嵌套字段白名单；动作递归保留注册
    标量与键，丢弃未知成员及非有限数，不修改输入。返回值是安全 JSON 值或
    ``None``；循环引用、非法类型和不允许的文本按缺失处理，不会抛出原始异常。
    """

    if isinstance(value, float) and not math.isfinite(value):
        return None
    allowed_keys = _allowed_keys or _SCIENTIFIC_OBSERVATION_NESTED_KEYS
    scalar = _safe_scalar(value)
    if scalar is not None or value is None:
        return scalar
    if _seen is None:
        _seen = set()
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in _seen:
            return None
        _seen.add(identity)
        projected: dict[str, Any] = {}
        for key, item in value.items():
            if key not in allowed_keys:
                continue
            safe = _safe_scientific_value(
                item,
                _seen=_seen,
                _allowed_keys=allowed_keys,
            )
            if safe is not None:
                projected[key] = safe
        _seen.remove(identity)
        return projected
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        identity = id(value)
        if identity in _seen:
            return None
        _seen.add(identity)
        projected = [
            safe
            for item in value
            if (
                safe := _safe_scientific_value(
                    item,
                    _seen=_seen,
                    _allowed_keys=allowed_keys,
                )
            ) is not None
        ]
        _seen.remove(identity)
        return projected
    return None


def _safe_validation_requirement(value: Any, *, _seen: set[int] | None = None) -> Any:
    """筛选执行前验收要求的注册字段。

    输入是验收要求值及递归访问集合；动作递归保留注册键和安全标量，隐藏未知嵌套
    成员及路径文本，不修改输入。返回值是安全 JSON 值或 ``None``；循环引用、非
    有限数和不支持类型按缺失处理，不会把运行对象带入专家视图。
    """

    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, str):
        return _safe_derived_text(value, None)
    scalar = _safe_scalar(value)
    if scalar is not None or value is None:
        return scalar
    if _seen is None:
        _seen = set()
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in _seen:
            return None
        _seen.add(identity)
        projected: dict[str, Any] = {}
        for key, item in value.items():
            if key not in _VALIDATION_REQUIREMENT_KEYS:
                continue
            safe = _safe_validation_requirement(item, _seen=_seen)
            if safe is not None or item is None:
                projected[key] = safe
        _seen.remove(identity)
        return projected
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        identity = id(value)
        if identity in _seen:
            return None
        _seen.add(identity)
        projected = []
        for item in value:
            safe = _safe_validation_requirement(item, _seen=_seen)
            if safe is not None or item is None:
                projected.append(safe)
        _seen.remove(identity)
        return projected
    return None


def _normalise_condition_name(value: Any) -> tuple[str, str] | None:
    """把科学条件名归一化为固定公开名称。

    输入是条件名称；动作去除首尾空格、统一大小写和分隔符并查找别名，不修改输入。
    返回值是 ``(规范名, 原始别名)``，未知或非字符串输入返回 ``None``，不会抛出
    字段解析异常。
    """

    if not isinstance(value, str):
        return None
    alias = value.strip().casefold().replace("-", "_").replace(" ", "_")
    canonical = _CONDITION_NAME_ALIASES.get(alias)
    if canonical is None:
        return None
    return canonical, alias


def _safe_condition_scalar(value: Any) -> Any:
    """复制条件允许的严格 JSON 标量。

    输入是条件值；动作保留有限数值、布尔、字符串和 null，并拒绝路径、内部文本、
    对象及数组，不修改输入。返回值是安全标量或 ``_CONDITION_REJECTED`` 标记；
    非有限数和不支持类型都按拒绝处理，不会抛出异常。
    """

    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else _CONDITION_REJECTED
    if isinstance(value, str):
        if _CONDITION_UNSAFE_TEXT.search(value):
            return _CONDITION_REJECTED
        return value
    # Expert 条件不接受对象或数组；否则任意 raw/log/transport 子对象可能
    # 借由 value 字段绕过条件名 allowlist。
    return _CONDITION_REJECTED


def _safe_condition_unit(
    value: Any,
    canonical_name: str,
    alias: str,
) -> Any:
    """校验科学条件单位。

    输入是可选单位、规范条件名和原始别名；动作只保留该条件注册的单位，缺失时
    使用别名的固定默认值，不修改输入。返回值是单位字符串或拒绝标记；单位类型
    错误、包含内部文本或不在白名单时返回拒绝标记，不会抛出异常。
    """

    if value is None:
        return _CONDITION_DEFAULT_UNITS.get(alias)
    if not isinstance(value, str):
        return _CONDITION_REJECTED
    unit = value.strip()
    if not unit or _CONDITION_UNSAFE_TEXT.search(unit):
        return _CONDITION_REJECTED
    if unit not in _CONDITION_ALLOWED_UNITS.get(canonical_name, frozenset()):
        return _CONDITION_REJECTED
    return unit


def _iter_condition_items(candidate: Any) -> list[Any]:
    """把条件数组或简写对象展开为 ``name/value`` 项。

    输入是条件数组、名称到值的映射或其他值；动作将映射键转为固定项目，不修改
    输入。返回值是项目列表；其他形状或非字符串映射键返回空列表，不会抛出解析
    异常。
    """

    if isinstance(candidate, list):
        return candidate
    if isinstance(candidate, Mapping):
        return [
            {"name": name, "value": value}
            for name, value in candidate.items()
            if isinstance(name, str)
        ]
    return []


def _normalise_conditions(*sources: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """读取明确科学条件并固定为 name/value/unit 三字段。

    输入来源只读取 chemistry_context 和显式 conditions 数组/对象；名称经过
    固定 allowlist 归一化，值只允许严格 JSON 标量，未知名称、路径、运行身份、
    日志对象和未注册单位都会被忽略。返回数组即使为空也保持同一顶层字段形状。
    """

    candidates: list[list[Any]] = []
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        chemistry_context = source.get("chemistry_context")
        if isinstance(chemistry_context, Mapping):
            scalar_conditions: list[dict[str, Any]] = []
            for key in (
                "solvent",
                "temperature",
                "temperature_c",
                "temperature_k",
                "temperature_kelvin",
                "pressure",
                "pressure_pa",
                "pressure_kpa",
                "pressure_mpa",
                "pressure_bar",
                "pressure_atm",
                "concentration",
                "concentration_m",
                "concentration_molar",
                "molarity",
                "ph",
                "p_h",
                "time",
                "time_s",
                "time_ms",
                "duration",
                "duration_s",
                "duration_ms",
                "reaction_time",
                "atmosphere",
                "atmospheric",
                "atmosphere_gas",
                "gas",
                "environment",
                "charge",
                "multiplicity",
            ):
                if key not in chemistry_context:
                    continue
                value = chemistry_context[key]
                safe_value = _safe_condition_scalar(value)
                if safe_value is _CONDITION_REJECTED:
                    continue
                normalized_name = _normalise_condition_name(key)
                if normalized_name is None:
                    continue
                canonical_name, alias = normalized_name
                unit = _safe_condition_unit(None, canonical_name, alias)
                if unit is _CONDITION_REJECTED:
                    continue
                scalar_conditions.append(
                    {"name": canonical_name, "value": safe_value, "unit": unit}
                )
            if scalar_conditions:
                candidates.append(scalar_conditions)
        for key in ("explicit_conditions", "conditions", "scientific_conditions"):
            candidate = source.get(key)
            if isinstance(candidate, (list, Mapping)):
                # 空数组不应遮蔽后续来源；非空来源按传入顺序确定性合并。
                items = _iter_condition_items(candidate)
                if items:
                    candidates.append(items)
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in candidates:
        for item in raw:
            if isinstance(item, Mapping):
                name = item.get("name", item.get("label"))
                value = item.get("value", item.get("condition"))
                unit = item.get("unit")
            else:
                name, value, unit = None, item, None
            normalized_name = _normalise_condition_name(name)
            if normalized_name is None:
                continue
            canonical_name, alias = normalized_name
            safe_value = _safe_condition_scalar(value)
            if safe_value is _CONDITION_REJECTED:
                continue
            safe_unit = _safe_condition_unit(unit, canonical_name, alias)
            if safe_unit is _CONDITION_REJECTED:
                continue
            normalized = {
                "name": canonical_name,
                "value": safe_value,
                "unit": safe_unit,
            }
            identity = json.dumps(
                normalized,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            if identity in seen:
                continue
            seen.add(identity)
            result.append(normalized)
    return result


def _hypothesis_view(hypothesis: Mapping[str, Any]) -> dict[str, str]:
    """构造专家可见的假设文本视图。

    输入是假设映射；动作清理策略名称和推理正文中的路径、运行标记，不修改输入。
    返回值固定包含 ``strategy_name`` 与 ``reasoning`` 两个字符串；字段缺失或文本
    不安全时返回空字符串，不会泄露原始运行日志。
    """

    return {
        "strategy_name": _safe_derived_text(
            hypothesis.get("strategy_name"),
            "",
        ),
        "reasoning": _safe_derived_text(
            hypothesis.get("reasoning"),
            _safe_derived_text(hypothesis.get("detailed_reasoning"), ""),
        ),
    }


def _basename(value: Any) -> str | None:
    """把文件名字段限制为 basename，避免把目录放进模型消息。"""

    if not isinstance(value, str) or not value.strip():
        return None
    text = value.replace("\\", "/").split("/")[-1]
    return text if text not in {"", ".", ".."} else None


def _has_structure_scientific_summary(source: Mapping[str, Any]) -> bool:
    """判断结构对象是否至少含有一个可供专家阅读的科学摘要字段。"""

    return bool(
        source.get("summary_status") == "available"
        or isinstance(source.get("formula", source.get("molecular_formula")), str)
        or isinstance(source.get("atom_count", source.get("n_atoms")), int)
        or (
            isinstance(source.get("element_counts", source.get("elements")), Mapping)
            and bool(source.get("element_counts", source.get("elements")))
        )
    )


def _structure_source_step_ref(
    step: Mapping[str, Any],
    converter_kind: str,
) -> str | None:
    """读取结构输入绑定的上游步骤引用。

    输入是步骤映射和转换器类型；动作查找对应结构输入的 binding，只返回内部短引用
    供本地读取，不把引用写入专家视图。返回值是步骤引用字符串或 ``None``；请求
    形状、绑定列表或目标字段不匹配时返回 ``None``，不会抛出解析异常。
    """

    request = step.get("request")
    bindings = request.get("bindings") if isinstance(request, Mapping) else None
    if not isinstance(bindings, list):
        return None
    target_field = "inputs.sdf_path" if converter_kind == "sdf_to_gjf" else "inputs.input_xyz_path"
    for binding in bindings:
        if not isinstance(binding, Mapping):
            continue
        target = binding.get("target_field")
        normalized_target = str(target).lstrip("/").replace("/", ".")
        if normalized_target != target_field:
            continue
        source_ref = binding.get("source_step_ref")
        if isinstance(source_ref, str) and source_ref.strip():
            return source_ref
    return None


def _safe_structure_artifact_path(task_root: Path, value: Any) -> Path | None:
    """在任务目录内解析结构 ArtifactRef 路径。

    输入是任务根目录和相对 ArtifactRef；动作拒绝绝对路径、反斜杠、目录穿越、
    符号链接和非文件，不修改文件系统。返回值是位于任务根目录下的文件路径或
    ``None``；类型不符、路径不存在或解析失败时返回 ``None``，不会抛出路径异常。
    """

    if not isinstance(task_root, Path) or not isinstance(value, str) or not value.strip():
        return None
    if "\\" in value:
        return None
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        return None
    root = task_root.resolve(strict=False)
    path = task_root.joinpath(*relative.parts)
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError:
        return None
    if path.is_symlink() or not path.is_file():
        return None
    return path


def _structure_summary_from_completed_step(
    completed_step: Mapping[str, Any],
    converter_kind: str,
    task_root: Path | None,
) -> dict[str, Any] | None:
    """从已完成步骤的结构 ArtifactRef 读取有限分子摘要。

    这里只接受规范化步骤的 ``output_artifacts``，按真实化学媒体类型选择文件，
    不读取 raw_output，也不把 ArtifactRef 的身份、路径或摘要发送给模型。多个
    不同结构文件无法确定唯一输入时返回 None。
    """

    if not isinstance(completed_step, Mapping) or not isinstance(task_root, Path):
        return None
    artifacts = completed_step.get("output_artifacts")
    if not isinstance(artifacts, list):
        return None
    if converter_kind == "sdf_to_gjf":
        media_types = {
            "chemical/x-mdl-sd",
            "chemical/x-mdl-sdfile",
            "chemical/x-sdf",
            "chemical/sdf",
        }
        suffixes = {".sdf", ".sd"}
    else:
        media_types = {
            "chemical/xyz",
            "chemical/x-xyz",
            "chemical/x-xyzfile",
        }
        suffixes = {".xyz"}
    candidates: list[dict[str, Any]] = []
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            continue
        media_type = artifact.get("media_type")
        artifact_path = artifact.get("path")
        if media_type not in media_types and not (
            isinstance(artifact_path, str)
            and Path(artifact_path).suffix.casefold() in suffixes
        ):
            continue
        safe_path = _safe_structure_artifact_path(task_root, artifact_path)
        if safe_path is None:
            continue
        parsed = _read_structure_file_summary(str(safe_path), converter_kind)
        if parsed is not None:
            candidates.append(parsed)
    if not candidates:
        return None
    first = candidates[0]
    if any(candidate != first for candidate in candidates[1:]):
        return None
    return first


def _structure_summary(
    step: Mapping[str, Any],
    inputs: Mapping[str, Any],
    converter_kind: str,
    *,
    completed_steps: Mapping[str, Mapping[str, Any]] | None = None,
    task_root: Path | None = None,
) -> dict[str, Any]:
    """从当前步骤和已完成上游事实构造固定结构骨架。

    输入是当前步骤、执行输入、转换器类型及可选已完成步骤和任务根目录；动作读取
    有限分子摘要并隐藏 ArtifactRef、路径和运行字段，不修改输入。返回值是固定结构
    摘要；缺少可靠摘要时返回空字段，读取失败不会把原始异常传播给专家。
    """

    source = next(
        (
            candidate
            for candidate in (inputs.get("structure_summary"), step.get("structure_summary"))
            if isinstance(candidate, Mapping) and _has_structure_scientific_summary(candidate)
        ),
        {},
    )
    formula = source.get("formula", source.get("molecular_formula"))
    atom_count = source.get("atom_count", source.get("n_atoms"))
    element_counts = source.get("element_counts", source.get("elements"))
    if not isinstance(element_counts, Mapping):
        element_counts = {}
    normalized_element_counts: dict[str, int] = {}
    for key, value in element_counts.items():
        if not isinstance(key, str):
            continue
        normalized_key = key[:1].upper() + key[1:].lower()
        if (
            normalized_key in _VALID_ELEMENT_SYMBOLS
            and type(value) is int
            and value >= 0
        ):
            normalized_element_counts[normalized_key] = value
    element_counts = normalized_element_counts
    if not (isinstance(formula, str) or isinstance(atom_count, int) or element_counts):
        source_step_ref = _structure_source_step_ref(step, converter_kind)
        completed_source = (
            completed_steps.get(source_step_ref)
            if isinstance(completed_steps, Mapping) and source_step_ref
            else None
        )
        parsed = (
            _structure_summary_from_completed_step(
                completed_source,
                converter_kind,
                task_root,
            )
            if isinstance(completed_source, Mapping)
            else None
        )
        if parsed is not None:
            formula = parsed["formula"]
            atom_count = parsed["atom_count"]
            element_counts = parsed["element_counts"]
    if not (isinstance(formula, str) or isinstance(atom_count, int) or element_counts):
        structure_path = inputs.get(
            "sdf_path" if converter_kind == "sdf_to_gjf" else "input_xyz_path"
        )
        if converter_kind == "sdf_to_gjf" and structure_path is None:
            structure_path = inputs.get("input_sdf_path")
        parsed = _read_structure_file_summary(structure_path, converter_kind)
        if parsed is not None:
            formula = parsed["formula"]
            atom_count = parsed["atom_count"]
            element_counts = parsed["element_counts"]
    role = source.get("role", step.get("target_role", step.get("structure_role")))
    input_format = "sdf" if converter_kind == "sdf_to_gjf" else "xyz"
    safe_formula = _safe_derived_text(formula, None)
    safe_atom_count = (
        atom_count
        if type(atom_count) is int and atom_count >= 0
        else None
    )
    available = bool(
        source.get("summary_status") == "available"
        or safe_formula is not None
        or safe_atom_count is not None
        or bool(element_counts)
    )
    return {
        "summary_status": "available" if available else "unavailable",
        "role": _safe_derived_text(role, None),
        "input_format": input_format,
        "formula": safe_formula,
        "atom_count": safe_atom_count,
        "element_counts": element_counts,
    }


def _read_structure_file_summary(
    structure_path: Any,
    converter_kind: str,
) -> dict[str, Any] | None:
    """确定性读取本地 XYZ/SDF 摘要；失败返回 None，不把坐标带入模型。"""

    if not isinstance(structure_path, str) or not structure_path.strip():
        return None
    try:
        path = Path(structure_path)
        if not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
            return None
        lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    except (OSError, UnicodeError):
        return None
    elements: list[str] = []
    if converter_kind == "xyz_to_gjf":
        if not lines:
            return None
        try:
            count = int(lines[0].strip())
        except ValueError:
            return None
        atom_lines = lines[2 : 2 + count]
        if count < 0 or len(atom_lines) != count:
            return None
        for line in atom_lines:
            fields = line.split()
            if len(fields) < 4 or not re.fullmatch(r"[A-Za-z]{1,3}", fields[0]):
                return None
            elements.append(fields[0].capitalize())
        if len(elements) != count:
            return None
    else:
        if len(lines) < 4:
            return None
        counts_line = lines[3]
        count_match = re.match(r"\s*(\d+)\s+(\d+)", counts_line)
        if count_match:
            count = int(count_match.group(1))
            atom_lines = lines[4 : 4 + count]
            if len(atom_lines) != count:
                return None
            for line in atom_lines:
                fields = line.split()
                if len(fields) < 4 or not re.fullmatch(r"[A-Za-z]{1,3}", fields[3]):
                    return None
                elements.append(fields[3].capitalize())
        else:
            for line in lines:
                match = re.search(r"M\s+V30\s+\d+\s+([A-Za-z]{1,3})(?:\s|$)", line)
                if match:
                    elements.append(match.group(1).capitalize())
            count = len(elements)
            if count == 0:
                return None
    counts: dict[str, int] = {}
    for element in elements:
        counts[element] = counts.get(element, 0) + 1
    ordered = sorted(counts)
    if "C" in counts:
        ordered = ["C"] + (["H"] if "H" in counts else []) + [key for key in ordered if key not in {"C", "H"}]
    formula = "".join(element + (str(counts[element]) if counts[element] != 1 else "") for element in ordered)
    return {"formula": formula, "atom_count": len(elements), "element_counts": counts}


def _calculation_type(step: Mapping[str, Any], route: str) -> str:
    """从已注册步骤字段得到受控 Gaussian 计算类型。"""

    request = step.get("request")
    if isinstance(request, Mapping):
        for source in (request, request.get("method_spec"), request.get("action_settings")):
            if isinstance(source, Mapping):
                value = source.get("calculation_type", source.get("job_type"))
                if isinstance(value, str) and value.strip():
                    safe_value = _safe_derived_text(value, None)
                    if safe_value:
                        return safe_value.strip()
    lowered = route.casefold()
    kinds = [
        name
        for token, name in (
            (r"\bopt(?:imize|imization)?\b", "optimization"),
            (r"\bfreq(?:uency)?\b", "frequency"),
            (r"\birc\b", "irc"),
            (r"(?<![A-Za-z])sp(?![A-Za-z])", "single_point"),
        )
        if re.search(token, lowered)
    ]
    return "+".join(kinds) if kinds else "gaussian_calculation"


def build_gaussian_step_review_view(
    *,
    scientific_question: str,
    hypothesis: Mapping[str, Any],
    step: Mapping[str, Any],
    route: str,
    converter_kind: str,
    conditions_sources: Sequence[Mapping[str, Any] | None] = (),
    completed_steps: Mapping[str, Mapping[str, Any]] | None = None,
    task_root: Path | None = None,
) -> dict[str, Any]:
    """构造一次 Gaussian 执行前审查的固定视图。

    输入是科学问题、假设、步骤、route、转换器类型及可选条件和已完成步骤；动作
    读取注册科学字段与上游结构摘要，清除执行身份、路径和原始结果。返回值是固定
    Gaussian 审查视图；输入缺少可读字段时使用空或不可用值，读取失败不会把原始
    异常或 ArtifactRef 发送给模型。
    """

    request = step.get("request") if isinstance(step.get("request"), Mapping) else {}
    inputs = request.get("inputs") if isinstance(request.get("inputs"), Mapping) else {}
    method = request.get("method_spec") if isinstance(request.get("method_spec"), Mapping) else {}
    settings = request.get("action_settings") if isinstance(request.get("action_settings"), Mapping) else {}
    validation = hypothesis.get("validation_requirements", hypothesis.get("verification_requirements", []))
    if not isinstance(validation, list):
        validation = []
    mutable = [
        "route_override",
        "inputs.charge",
        "inputs.multiplicity",
        "inputs.title",
        "inputs.nprocshared",
        "inputs.mem",
        "inputs.chk_file",
    ]
    if converter_kind == "sdf_to_gjf":
        mutable = [item for item in mutable if item not in {"inputs.charge", "inputs.multiplicity"}]
    explicit = _normalise_conditions(hypothesis, step, *conditions_sources)
    return {
        "scientific_question": scientific_question,
        "explicit_conditions": explicit,
        "hypothesis": _hypothesis_view(hypothesis),
        "gaussian_unit": {
            "scientific_purpose": _safe_derived_text(step.get("description")),
            "calculation_type": _calculation_type(step, route),
            "structure": _structure_summary(
                step,
                inputs,
                converter_kind,
                completed_steps=completed_steps,
                task_root=task_root,
            ),
            "requested_route_intent": _safe_derived_text(
                inputs.get("user_input", request.get("user_input")),
                "",
            ),
            "generated_route": _safe_derived_text(route),
            "editable_parameters": {
                "charge": _safe_scalar(inputs.get("charge")),
                "multiplicity": _safe_scalar(inputs.get("multiplicity")),
                "title": _safe_derived_text(inputs.get("title"), None),
                "nprocshared": _safe_scalar(inputs.get("nprocshared")),
                "mem": _safe_derived_text(inputs.get("mem"), None),
                "chk_file": _safe_derived_text(
                    _basename(inputs.get("chk_file")),
                    None,
                ),
            },
            "validation_requirements": [
                safe
                for item in validation
                if (safe := _safe_validation_requirement(item)) is not None
            ],
            "mutable_fields": mutable,
        },
    }


_OPERATION_BY_ACTION = {
    "smiles2sdf": "structure_conversion",
    "generate_3d_structure": "structure_generation",
    "generate_gaussian_code": "gaussian_route_generation",
    "xyz_to_gjf": "gaussian_deck_preparation",
    "sdf_to_gjf": "gaussian_deck_preparation",
    "run_gaussian": "native_gaussian",
    "gaussian_calculation": "native_gaussian",
    "quantum_calculation": "native_gaussian",
    "optimize_geometry": "geometry_optimization",
    "calculate_hessian": "hessian_calculation",
    "derive_vibrational_modes": "vibrational_modes",
    "derive_ir_spectrum": "ir_spectrum",
    "derive_raman_spectrum": "raman_spectrum",
    "render_spectrum_plot": "spectrum_plot",
    "parse_quantum_chemistry_output": "quantum_chemistry_parser",
    "extract_final_structure": "quantum_chemistry_parser",
    "derive_thermochemistry": "thermochemistry",
    "scan_thermochemistry_temperature": "thermochemistry",
    "analyze_thermochemical_ensemble": "thermochemistry",
    "validate_thermochemistry_inputs": "thermochemistry",
    "analyze_reaction_free_energy_profile": "free_energy_profile",
    "calculate_potential_of_mean_force": "free_energy_profile",
    "analyze_thermochemical_selectivity": "thermochemical_selectivity",
    "analyze_free_energy_selectivity": "free_energy_selectivity",
}
_GAUSSIAN_SOFTWARE_IDS = frozenset({"gaussian", "gaussian16"})
_GAUSSIAN_BACKEND_IDS = frozenset(
    {"gaussian", "gaussian16", "gaussian_calculation"}
)
_SETTINGS_KEYS = {
    "structure_conversion": (),
    "structure_generation": ("random_seed",),
    "gaussian_route_generation": ("requested_route_intent", "route_kind"),
    "gaussian_deck_preparation": ("charge", "multiplicity", "input_format"),
    "geometry_optimization": (
        "method", "basis", "optimizer", "force_field", "optimization_level",
        "fmax_ev_per_angstrom", "max_steps", "max_iterations", "force_tolerance",
        "energy_tolerance", "constraints",
    ),
    "hessian_calculation": (
        "method", "basis", "displacement_angstrom", "coordinate_system",
    ),
    "vibrational_modes": ("linearity",),
    "ir_spectrum": (
        "broadening", "line_shape", "fwhm_cm1", "fwhm", "min_wavenumber_cm1",
        "max_wavenumber_cm1", "grid_min", "grid_max", "points", "grid_points",
    ),
    "raman_spectrum": (
        "line_shape", "fwhm", "grid_min", "grid_max", "grid_points",
        "position_scale_factor",
    ),
    "spectrum_plot": (),
    "native_gaussian": (
        "calculation_type", "method", "basis", "dispersion", "solvent_model",
        "solvent", "temperature", "integration_grid", "scf_convergence",
        "optimization_options", "frequency_options", "charge", "multiplicity",
    ),
    "quantum_chemistry_parser": (
        "coordinate_frames", "properties", "include_excited_state_configurations",
        "include_orbital_coefficients",
    ),
    "thermochemistry": (
        "temperature_kelvin", "pressure_pa", "geometry", "spin", "symmetry_number",
        "ignore_imaginary_modes", "standard_state", "entropy_model", "enthalpy_model",
        "frequency_scale_factor", "zpe_scale_factor", "imaginary_frequency_policy",
        "symmetry_correction",
    ),
    "free_energy_profile": (
        "temperature_kelvin", "standard_state", "entropy_model", "enthalpy_model",
        "frequency_scale_factor", "zpe_scale_factor", "imaginary_frequency_policy",
        "symmetry_correction", "population_basis", "deduplicate_structures",
    ),
    "thermochemical_selectivity": (
        "temperature_kelvin", "standard_state", "entropy_model", "enthalpy_model",
        "frequency_scale_factor", "zpe_scale_factor", "imaginary_frequency_policy",
        "symmetry_correction", "population_basis", "deduplicate_structures",
    ),
    "free_energy_selectivity": ("temperature_kelvin", "comparison_type"),
    "programmable_analysis": ("analysis_method",),
    "unknown": (),
}
_EXPECTED = {
    "structure_conversion": (("molecules_written", None),),
    "structure_generation": (("generated_structure", None),),
    "gaussian_route_generation": (("generated_gaussian_route", None),),
    "gaussian_deck_preparation": (("deck_preflight_status", None),),
    "geometry_optimization": (("optimized_geometry", None),),
    "hessian_calculation": (("hessian_summary", None),),
    "vibrational_modes": (("vibrational_mode_summary", None),),
    "ir_spectrum": (("ir_line_spectrum", None),),
    "raman_spectrum": (("raman_line_spectrum", None),),
    "native_gaussian": (
        ("optimized_geometry", None), ("electronic_energy", "hartree"),
        ("vibrational_frequencies", "cm-1"),
    ),
    "quantum_chemistry_parser": (
        ("final_atomic_structure", None), ("electronic_energies", "hartree"),
        ("vibrational_frequencies", "cm-1"), ("vibrational_intensities", None),
        ("charge", None), ("multiplicity", None),
    ),
    "thermochemistry": (
        ("zero_point_energy", "hartree"), ("enthalpy", "hartree"),
        ("entropy", "hartree/K"), ("gibbs_free_energy", "hartree"),
    ),
    "free_energy_profile": (
        ("relative_gibbs_free_energies", "hartree"),
        ("activation_free_energies", "hartree"), ("rate_determining_step", None),
    ),
    "thermochemical_selectivity": (
        ("delta_delta_g", "hartree"), ("enantiomeric_ratio", None),
        ("enantiomeric_excess", "%"),
    ),
    "free_energy_selectivity": (
        ("delta_delta_g_kcal_mol", "kcal/mol"),
        ("populations", None),
        ("population_sum", None),
        ("ee_percent", "%"),
    ),
    "spectrum_plot": (),
    "programmable_analysis": (),
    "unknown": (("unavailable_scientific_result", None),),
}

_SCIENTIFIC_SETTING_VALUE_KEYS = frozenset(
    set(_SCIENTIFIC_OBSERVATION_NESTED_KEYS)
    | {
        "requested_route_intent",
        "route_kind",
        "charge",
        "multiplicity",
        "input_format",
        "method",
        "basis",
        "optimizer",
        "force_field",
        "optimization_level",
        "fmax_ev_per_angstrom",
        "max_steps",
        "max_iterations",
        "force_tolerance",
        "energy_tolerance",
        "constraints",
        "displacement_angstrom",
        "coordinate_system",
        "linearity",
        "broadening",
        "line_shape",
        "fwhm_cm1",
        "fwhm",
        "min_wavenumber_cm1",
        "max_wavenumber_cm1",
        "grid_min",
        "grid_max",
        "points",
        "grid_points",
        "position_scale_factor",
        "calculation_type",
        "dispersion",
        "solvent_model",
        "solvent",
        "temperature",
        "integration_grid",
        "scf_convergence",
        "optimization_options",
        "frequency_options",
        "temperature_kelvin",
        "pressure_pa",
        "geometry",
        "spin",
        "symmetry_number",
        "ignore_imaginary_modes",
        "standard_state",
        "entropy_model",
        "enthalpy_model",
        "frequency_scale_factor",
        "zpe_scale_factor",
        "imaginary_frequency_policy",
        "symmetry_correction",
        "population_basis",
        "deduplicate_structures",
        "analysis_method",
        "random_seed",
        "value",
        "source",
        "name",
        "description",
        "type",
        "unit",
    }
)

_VALIDATION_REQUIREMENT_KEYS = frozenset(
    {
        "name",
        "observable",
        "expected",
        "expected_output",
        "output_format",
        "unit",
        "operator",
        "required",
        "value",
        "tolerance",
        "conditions",
        "description",
    }
)
_PROGRAM_CHECK_VALUE_KEYS = frozenset(
    set(_SCIENTIFIC_OBSERVATION_NESTED_KEYS)
    | {
        "semantic_type",
        "media_type",
        "normal_termination",
        "converged",
        "status",
        "validation_status",
        "validation_errors",
        "expected",
        "required",
        "value",
        "unit",
        "ready",
        "checks",
        "reason",
        "message",
    }
)


def _operation(step: Mapping[str, Any]) -> str:
    """把 Action 或开放执行层映射为受控科学操作名称。"""

    request = step.get("request") if isinstance(step.get("request"), Mapping) else {}
    has_software_id = "software_id" in request or "software_id" in step
    software = request.get("software_id", step.get("software_id")) if has_software_id else None
    backend = step.get("backend_id", request.get("backend_id"))
    has_backend_id = backend is not None
    if has_software_id:
        # 一旦 workflow 明确声明了软件身份，必须按冻结软件名称严格判断；
        # 不能仅凭自然语言中的 Gaussian 字样把其他程序当成 Gaussian。
        if not isinstance(software, str) or software.casefold() not in _GAUSSIAN_SOFTWARE_IDS:
            return "unknown"
        if step.get("execution_layer") == "native_software":
            return "native_gaussian"
    action = step.get("action_id")
    if isinstance(action, str) and action in _OPERATION_BY_ACTION:
        operation = _OPERATION_BY_ACTION[action]
        if (
            operation == "native_gaussian"
            and has_backend_id
            and (
                not isinstance(backend, str)
                or backend.casefold() not in _GAUSSIAN_BACKEND_IDS
            )
        ):
            return "unknown"
        return operation
    if step.get("execution_layer") == "native_software":
        # 兼容旧 fixture：没有 software_id 时才允许由受控动作或明确的
        # Gaussian 描述识别 native Gaussian；有未知软件身份时上面已 fail closed。
        description = " ".join(
            str(step.get(key, "")) for key in ("step_name", "description")
        ).casefold()
        if (
            "gaussian" in description
            and (
                not has_backend_id
                or (
                    isinstance(backend, str)
                    and backend.casefold() in _GAUSSIAN_BACKEND_IDS
                )
            )
        ):
            return "native_gaussian"
    if step.get("execution_layer") == "programmable_analysis":
        return "programmable_analysis"
    return "unknown"


def _step_status(source: Mapping[str, Any] | None) -> str:
    """归一化 completed、partial、failed、not_executed 四态。"""

    if not isinstance(source, Mapping):
        return "not_executed"
    value = source.get("status")
    if isinstance(value, str):
        lowered = value.casefold()
        if lowered in {"failed", "failure", "error", "timeout", "cancelled", "technical_pause"}:
            return "failed"
        if lowered in {"partial", "incomplete"}:
            return "partial"
        if lowered in {"not_executed", "not-executed", "skipped", "pending"}:
            return "not_executed"
        if lowered in {"completed", "success", "succeeded", "ready"}:
            return "completed"
    raw = source.get("raw_output")
    if isinstance(raw, Mapping) and isinstance(raw.get("status"), str):
        lowered = raw["status"].casefold()
        if lowered in {"failed", "failure", "error"}:
            return "failed"
        if lowered in {"completed", "success", "succeeded"}:
            return "completed"
    if any(key in source for key in ("error", "failure", "failure_kind")):
        return "failed"
    return "not_executed"


def _source_result(source: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """读取结构化 result；不读取 raw_output 之外的任意字符串。"""

    if not isinstance(source, Mapping):
        return {}
    result = source.get("result")
    if isinstance(result, Mapping):
        if isinstance(result.get("result"), Mapping):
            return result["result"]
        return result
    raw = source.get("raw_output")
    if isinstance(raw, Mapping) and isinstance(raw.get("result"), Mapping):
        return raw["result"]
    return {}


def _native_preflight_view(
    operation: str,
    result: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """整理 Gaussian deck/native 预检的公开事实。

    输入是操作名和可选执行结果；动作读取就绪状态、计算意图、Link1 数量及固定
    检查名，隐藏路径和运行字段，不修改结果。返回值是固定预检结构；非预检步骤、
    缺少事实或字段非法时返回 ``unavailable``，明确未就绪保留为 ``not_ready``。
    """

    unavailable = {
        "status": "unavailable",
        "ready": False,
        "calculation_intent": None,
        "link1_segment_count": None,
        "checks": [],
    }
    if operation not in {"gaussian_deck_preparation", "native_gaussian"}:
        return unavailable
    if not isinstance(result, Mapping):
        return unavailable
    native = result.get("native_preflight")
    if not isinstance(native, Mapping) and "native_preflight_ready" not in result:
        return unavailable
    ready = native.get("native_preflight_ready") if isinstance(native, Mapping) else None
    if not isinstance(ready, bool):
        ready = result.get("native_preflight_ready") is True
    has_ready_fact = (
        isinstance(native, Mapping) and "native_preflight_ready" in native
    ) or "native_preflight_ready" in result
    count = native.get("link1_segment_count") if isinstance(native, Mapping) else None
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        count = None
    checks = native.get("checks") if isinstance(native, Mapping) else None
    safe_checks = (
        [
            _safe_derived_text(item)
            for item in checks
            if isinstance(item, str) and _safe_derived_text(item)
        ]
        if isinstance(checks, list)
        else []
    )
    return {
        "status": "ready" if ready else ("not_ready" if has_ready_fact else "unavailable"),
        "ready": ready,
        "calculation_intent": _safe_derived_text(
            native.get("calculation_intent") if isinstance(native, Mapping) else None,
            None,
        ),
        "link1_segment_count": count,
        "checks": safe_checks,
    }


def _get_nested(source: Mapping[str, Any], keys: Sequence[str]) -> Any:
    """按固定别名读取一个科学结果字段。"""

    for key in keys:
        if key in source:
            return source[key]
    return _MISSING


def _setting_source(
    step: Mapping[str, Any],
    operation: str,
    source: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """按注册字段构造请求值与执行后确认状态。

    输入是规范 workflow 步骤和可选执行结果；函数只读取操作注册的设置字段，
    不复制请求或原始输出。只有执行结果显式放在 ``effective_settings``、
    ``settings``、``method_spec`` 或 ``action_settings`` 块中的字段才标记
    ``verified_effective``；顶层同名字段可能是科学结果，不能冒充已确认设置。
    只有请求中给出值时标记 ``requested_only``，其余字段统一返回
    ``{value: null, source: unavailable}``。
    """

    if operation == "unknown" or operation not in _SETTINGS_KEYS:
        return {}

    request = step.get("request") if isinstance(step.get("request"), Mapping) else {}
    method = request.get("method_spec") if isinstance(request.get("method_spec"), Mapping) else {}
    action = request.get("action_settings") if isinstance(request.get("action_settings"), Mapping) else {}
    inputs = request.get("inputs") if isinstance(request.get("inputs"), Mapping) else {}
    merged: dict[str, Any] = {}
    for field in (
        "calculation_type", "method", "basis", "dispersion", "solvent_model",
        "solvent", "temperature", "integration_grid", "scf_convergence",
        "optimization_options", "frequency_options", "charge", "multiplicity",
        "temperature_kelvin", "pressure_pa", "geometry", "spin", "symmetry_number",
        "ignore_imaginary_modes", "standard_state", "entropy_model", "enthalpy_model",
        "frequency_scale_factor", "zpe_scale_factor", "imaginary_frequency_policy",
        "symmetry_correction", "population_basis", "deduplicate_structures",
        "coordinate_frames", "properties", "include_excited_state_configurations",
        "include_orbital_coefficients", "analysis_method", "route_kind", "linearity",
        "optimizer", "force_field", "optimization_level", "fmax_ev_per_angstrom",
        "max_steps", "max_iterations", "force_tolerance", "energy_tolerance",
        "constraints", "input_format", "displacement_angstrom", "coordinate_system", "broadening",
        "line_shape", "fwhm_cm1", "fwhm", "min_wavenumber_cm1",
        "max_wavenumber_cm1", "grid_min", "grid_max", "points", "grid_points",
        "position_scale_factor", "random_seed",
    ):
        for candidate in (inputs, method, action, step):
            if isinstance(candidate, Mapping) and field in candidate:
                merged[field] = _safe_scientific_value(
                    candidate[field],
                    _allowed_keys=_SCIENTIFIC_SETTING_VALUE_KEYS,
                )
                break
    effective = _source_result(source)
    effective_lookup: dict[str, Any] = {}
    for block_name in (
        "effective_settings",
        "settings",
        "method_spec",
        "action_settings",
    ):
        block = effective.get(block_name)
        if isinstance(block, Mapping):
            for key, value in block.items():
                effective_lookup.setdefault(key, value)
    if operation == "gaussian_route_generation":
        intent = _MISSING
        for candidate in (inputs, request, method, action, step, effective_lookup):
            if not isinstance(candidate, Mapping):
                continue
            for key in ("user_input", "route_generation_request", "requested_route_intent"):
                value = candidate.get(key, _MISSING)
                if isinstance(value, str) and value.strip():
                    intent = value
                    break
            if intent is not _MISSING:
                break
        if intent is not _MISSING:
            merged["requested_route_intent"] = _safe_derived_text(intent)
    result: dict[str, dict[str, Any]] = {}
    for field in _SETTINGS_KEYS[operation]:
        effective_value = effective_lookup.get(field, _MISSING)
        if effective_value is not _MISSING and effective_value is not None:
            projected = _safe_scientific_value(
                effective_value,
                _allowed_keys=_SCIENTIFIC_SETTING_VALUE_KEYS,
            )
            if projected is not None:
                result[field] = {
                    "value": projected,
                    "source": "verified_effective",
                }
                continue
        requested_value = merged.get(field, _MISSING)
        if requested_value is not _MISSING and requested_value is not None:
            result[field] = {
                "value": requested_value,
                "source": "requested_only",
            }
        else:
            result[field] = {"value": None, "source": "unavailable"}
    return result


def _expected_observables(step: Mapping[str, Any], operation: str) -> list[dict[str, Any]]:
    """取得固定操作的具名科学目标；不复制 expected_outputs 合同。"""

    # 这些 MiniChem Action 的结果字段由固定 projector 决定；workflow 不能把
    # 一个密集数组或任意自定义名称注入模型视图。
    request = step.get("request") if isinstance(step.get("request"), Mapping) else {}
    if operation == "programmable_analysis":
        # AnalysisJobRequest.outputs 是唯一稳定的具名科学输出来源；不再读取
        # workflow 自定义的 scientific_observables 字段，避免模型伪造目标。
        declared = request.get("outputs")
    else:
        declared = (
            None
            if operation
            in {
                "structure_conversion",
                "structure_generation",
                "geometry_optimization",
                "hessian_calculation",
                "vibrational_modes",
                "ir_spectrum",
                "raman_spectrum",
                "spectrum_plot",
            }
            else step.get("scientific_observables")
        )
    if isinstance(declared, list):
        registered_names = (
            set(_PROGRAMMABLE_ALLOWED_OBSERVABLES)
            if operation == "programmable_analysis"
            else {name for name, _ in _EXPECTED[operation]}
        )
        values = []
        seen_names: set[str] = set()
        for item in declared:
            if isinstance(item, Mapping) and isinstance(item.get("name"), str):
                name = _safe_derived_text(item["name"])
                json_schema = item.get("json_schema")
                declared_unit = item.get("unit")
                if declared_unit is None and isinstance(json_schema, Mapping):
                    declared_unit = json_schema.get("unit")
                if (
                    name
                    and name in registered_names
                    and name not in _FORBIDDEN_OBSERVABLE_NAMES
                    and name not in seen_names
                ):
                    values.append({"name": name, "unit": _safe_scalar(declared_unit)})
                    seen_names.add(name)
            elif isinstance(item, str):
                name = _safe_derived_text(item)
                if (
                    name
                    and name in registered_names
                    and name not in _FORBIDDEN_OBSERVABLE_NAMES
                    and name not in seen_names
                ):
                    values.append({"name": name, "unit": None})
                    seen_names.add(name)
        if values:
            if operation == "native_gaussian":
                values = [
                    item
                    for item in values
                    if item["name"] not in {"normal_termination", "optimization_convergence"}
                ]
            return values
    values = [{"name": name, "unit": unit} for name, unit in _EXPECTED[operation]]
    if operation == "native_gaussian":
        values = [
            item
            for item in values
            if item["name"] not in {"normal_termination", "optimization_convergence"}
        ]
    return values


def _observation(evidence_ref: str, name: str, value: Any, unit: Any = None, conditions: Any = None) -> dict[str, Any]:
    """构造固定观测字段。"""

    return {
        "evidence_ref": evidence_ref,
        "name": name,
        "value": _safe_scientific_value(value),
        "unit": _safe_scalar(unit),
        "conditions": _safe_scientific_value(conditions) if isinstance(conditions, Mapping) else {},
    }


def _finite_numeric_list(value: Any) -> list[float] | None:
    """校验并复制一维有限数值数组；非法数组返回缺失。"""

    if not isinstance(value, list):
        return None
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return None
        converted = float(item)
        if not math.isfinite(converted):
            return None
        result.append(converted)
    return result


def _project_hessian_summary(result: Mapping[str, Any]) -> dict[str, Any] | None:
    """把 Hessian 密集矩阵归约为可复核的形状、有限性和对称性摘要。"""

    matrix = result.get("matrix", result.get("hessian_matrix"))
    if not isinstance(matrix, list) or not matrix or not all(isinstance(row, list) for row in matrix):
        return None
    row_count = len(matrix)
    column_count = len(matrix[0])
    if column_count == 0 or any(len(row) != column_count for row in matrix):
        return None
    values: list[list[float]] = []
    for row in matrix:
        numbers = _finite_numeric_list(row)
        if numbers is None:
            return None
        values.append(numbers)
    shape = [row_count, column_count]
    symmetric = row_count == column_count and all(
        math.isclose(values[row][column], values[column][row], rel_tol=1e-9, abs_tol=1e-12)
        for row in range(row_count)
        for column in range(row_count)
    )
    return {
        "shape": shape,
        "element_count": row_count * column_count,
        "unit": _safe_derived_text(result.get("unit"), None),
        "finite": True,
        "symmetric": symmetric,
    }


def _project_vibrational_mode_summary(result: Mapping[str, Any]) -> dict[str, Any] | None:
    """从完整频率数组提取总数、实/虚频计数和完整虚频条目。"""

    frequencies = result.get("frequencies_cm1")
    if not isinstance(frequencies, list):
        return None
    imaginary: list[Any] = []
    real_count = 0
    for item in frequencies:
        if isinstance(item, Mapping):
            real = item.get("real")
            imaginary_part = item.get("imaginary", 0.0)
            if (
                isinstance(real, bool)
                or not isinstance(real, (int, float))
                or isinstance(imaginary_part, bool)
                or not isinstance(imaginary_part, (int, float))
                or not math.isfinite(float(real))
                or not math.isfinite(float(imaginary_part))
            ):
                return None
            normalized = {
                "real": float(real),
                "imaginary": float(imaginary_part),
            }
            if float(imaginary_part) != 0.0:
                imaginary.append(normalized)
            else:
                real_count += 1
            continue
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return None
        converted = float(item)
        if not math.isfinite(converted):
            return None
        # 不把负实数改写成虚频；只有 ActionResult 明确给出的 imaginary
        # 分量才计入虚频，避免改变 ASE 复数频率的语义。
        real_count += 1
    return {
        "total_count": len(frequencies),
        "real_count": real_count,
        "imaginary_count": len(imaginary),
        "frequency_unit": "cm-1",
        "imaginary_frequencies_cm1": imaginary,
    }


def _project_line_spectrum_summary(
    result: Mapping[str, Any],
) -> dict[str, Any] | None:
    """规范化 SpectrumResult 的线谱数组并丢弃 dense 网格。"""

    axis = result.get("axis")
    axis_values = axis.get("values") if isinstance(axis, Mapping) else result.get("wavenumber_cm1")
    intensity = result.get("intensity")
    if isinstance(intensity, Mapping):
        intensity_values = intensity.get("values")
        intensity_unit = _safe_derived_text(intensity.get("unit"), None)
    else:
        intensity_values = result.get("intensity_values", intensity)
        intensity_unit = _safe_derived_text(result.get("intensity_unit"), None)
    dense_axis = _finite_numeric_list(axis_values) if axis_values is not None else None
    dense_intensity = _finite_numeric_list(intensity_values) if intensity_values is not None else None
    if dense_axis is not None and dense_intensity is not None and len(dense_axis) != len(dense_intensity):
        return None

    raw_frequencies = result.get("line_frequencies_cm1")
    raw_intensities = result.get("line_intensities")
    sticks = result.get("sticks")
    stick_frequencies: list[float] | None = None
    stick_intensities: list[float] | None = None
    if isinstance(sticks, list):
        stick_frequencies = []
        stick_intensities = []
        for item in sticks:
            if not isinstance(item, Mapping):
                return None
            position = item.get("position")
            strength = item.get("intensity")
            if (
                isinstance(position, bool)
                or not isinstance(position, (int, float))
                or isinstance(strength, bool)
                or not isinstance(strength, (int, float))
                or not math.isfinite(float(position))
                or not math.isfinite(float(strength))
            ):
                return None
            stick_frequencies.append(float(position))
            stick_intensities.append(float(strength))
    line_frequencies = _finite_numeric_list(raw_frequencies) if raw_frequencies is not None else stick_frequencies
    line_intensities = _finite_numeric_list(raw_intensities) if raw_intensities is not None else stick_intensities
    if line_frequencies is None or line_intensities is None or len(line_frequencies) != len(line_intensities):
        return None
    if stick_frequencies is not None and (
        stick_frequencies != line_frequencies or stick_intensities != line_intensities
    ):
        return None

    raw_point_count = result.get("point_count")
    point_count = raw_point_count if isinstance(raw_point_count, int) and not isinstance(raw_point_count, bool) else (
        len(dense_axis) if dense_axis is not None else None
    )
    if point_count is not None and point_count < 0:
        return None
    if dense_axis is not None and point_count is not None and point_count != len(dense_axis):
        return None
    raw_stick_count = result.get("stick_count")
    stick_count = raw_stick_count if isinstance(raw_stick_count, int) and not isinstance(raw_stick_count, bool) else len(line_frequencies)
    if stick_count < 0 or stick_count != len(line_frequencies):
        return None
    line_shape = result.get("line_shape")
    broadening = result.get("broadening")
    fwhm = result.get("fwhm_cm1", result.get("fwhm"))
    if isinstance(line_shape, Mapping):
        broadening = line_shape.get("kind", broadening)
        fwhm = line_shape.get("fwhm", fwhm)
    if fwhm is not None:
        if isinstance(fwhm, bool) or not isinstance(fwhm, (int, float)) or not math.isfinite(float(fwhm)):
            return None
        fwhm = float(fwhm)
    return {
        "line_frequencies_cm1": line_frequencies,
        "line_intensities": line_intensities,
        "frequency_unit": "cm-1",
        "intensity_unit": intensity_unit,
        "broadening": _safe_derived_text(broadening, None),
        "fwhm_cm1": fwhm,
        "point_count": point_count,
        "stick_count": stick_count,
    }


def _result_observations(
    operation: str,
    result: Mapping[str, Any],
    next_ref: callable,
    expected: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    """按操作登记科学结果，完整保留允许的数组而不取前 N 项。"""

    if operation == "unknown":
        return []

    aliases: dict[str, tuple[tuple[str, ...], Any]] = {
        "molecules_written": (("molecules_written",), None),
        "generated_gaussian_route": (("gaussian_code", "route", "generated_route"), None),
        "gaussian_input_deck": (("gaussian_input", "input_deck", "deck"), None),
        "deck_preflight_status": (("preflight_status", "validation_status", "status"), None),
        "generated_structure": (("generated_structure", "structure", "coordinates", "atomcoords"), None),
        "optimized_geometry": (("optimized_geometry", "final_structure", "structure", "coordinates", "atomcoords"), None),
        "hessian_matrix": (("hessian_matrix", "matrix"), None),
        "electronic_energy": (("electronic_energy", "energy", "energy_hartree"), "hartree"),
        "frequencies_cm1": (("frequencies_cm1",), "cm-1"),
        "wavenumber_cm1": (("wavenumber_cm1",), "cm-1"),
        "intensity": (("intensity", "intensity_values"), None),
        "line_frequencies_cm1": (("line_frequencies_cm1",), "cm-1"),
        "line_intensities": (("line_intensities",), None),
        "optimization_convergence": (("optimization_convergence", "converged", "convergence"), None),
        "normal_termination": (("normal_termination", "termination", "terminated_normally"), None),
        "final_atomic_structure": (("final_atomic_structure", "final_structure", "structure", "atomcoords"), None),
        "electronic_energies": (("electronic_energies", "energies", "energy", "scfenergies", "mpenergies", "ccenergies"), "hartree"),
        "vibrational_intensities": (("vibrational_intensities", "intensities", "vibirs", "vibramans"), None),
        "charge": (("charge",), None),
        "multiplicity": (("multiplicity", "spin_multiplicity", "mult"), None),
        "vibrational_frequencies": (("vibrational_frequencies", "frequencies", "frequency", "vibfreqs"), "cm-1"),
        "zero_point_energy": (("zero_point_energy", "zpe", "zpe_energy"), "hartree"),
        "enthalpy": (("enthalpy", "H"), "hartree"),
        "entropy": (("entropy", "S"), "hartree/K"),
        "gibbs_free_energy": (("gibbs_free_energy", "free_energy", "G"), "hartree"),
        "relative_gibbs_free_energies": (("relative_gibbs_free_energies", "relative_energies", "profile"), "hartree"),
        "activation_free_energies": (("activation_free_energies", "barriers", "activation_barriers"), "hartree"),
        "rate_determining_step": (("rate_determining_step", "highest_barrier", "rate_determining"), None),
        "delta_delta_g": (("delta_delta_g", "ddg", "delta_delta_G"), "hartree"),
        "enantiomeric_ratio": (("enantiomeric_ratio", "ratio"), None),
        "enantiomeric_excess": (("enantiomeric_excess", "ee"), "%"),
    }
    if expected:
        names = [
            item["name"]
            for item in expected
            if isinstance(item, Mapping) and isinstance(item.get("name"), str)
        ]
        declared_units = {
            item["name"]: item.get("unit")
            for item in expected
            if isinstance(item, Mapping) and isinstance(item.get("name"), str)
        }
    else:
        names = [name for name, _ in _EXPECTED[operation]]
        declared_units = {}
    if operation == "native_gaussian":
        # 终止和收敛属于程序检查，不再与科学 observations 重复。
        names = [name for name in names if name not in {"normal_termination", "optimization_convergence"}]
    # GoodVibes、parser 等结果会把科学块放在一层命名 object 下；这些块是
    # 注册结果类型的一部分，可以合并读取，但不会递归复制其余字段。
    lookup: dict[str, Any] = dict(result)
    for block_name in (
        "thermochemistry", "selected_thermochemistry", "selectivity", "profile",
        "properties", "metadata", "observables", "metrics", "analysis",
    ):
        block = result.get(block_name)
        if isinstance(block, Mapping):
            for key, value in block.items():
                lookup.setdefault(key, value)
    # Gaussian deck 预检结果包含输入目标和每个 Link1 片段的内部路径；专家只需
    # 知道预检是否就绪、计算意图、片段数量和固定检查名称，不能看到这些路径。
    native_preflight = result.get("native_preflight")
    if isinstance(native_preflight, Mapping):
        checks = native_preflight.get("checks")
        safe_checks = []
        if isinstance(checks, list):
            safe_checks = [
                item
                for item in checks
                if isinstance(item, str) and _safe_derived_text(item, None)
            ]
        ready = native_preflight.get("native_preflight_ready")
        has_ready_fact = isinstance(ready, bool)
        if not has_ready_fact and isinstance(result.get("native_preflight_ready"), bool):
            ready = result["native_preflight_ready"]
            has_ready_fact = True
        lookup["preflight_status"] = {
            "status": "ready" if ready else ("not_ready" if has_ready_fact else "unavailable"),
            "ready": ready,
            "calculation_intent": _safe_derived_text(
                native_preflight.get("calculation_intent"), None
            ),
            "link1_segment_count": (
                native_preflight.get("link1_segment_count")
                if isinstance(native_preflight.get("link1_segment_count"), int)
                and not isinstance(native_preflight.get("link1_segment_count"), bool)
                and native_preflight.get("link1_segment_count") >= 0
                else None
            ),
            "checks": safe_checks,
        }
    attribute_units = result.get("attribute_units")
    units_by_key = (
        {
            key: _safe_derived_text(value, None)
            for key, value in attribute_units.items()
            if isinstance(key, str) and isinstance(value, str) and _safe_derived_text(value, None)
        }
        if isinstance(attribute_units, Mapping)
        else {}
    )
    # SpectrumResult 的 Raman 输出以 axis/intensity/sticks 保存标准数组，而
    # IR 还会带一组兼容的顶层别名。只在缺少兼容字段时从这些固定结构展开，
    # 不复制 source_fields、路径或绘图元数据，也不截断任何数组。
    if operation in {"ir_spectrum", "raman_spectrum"}:
        axis = result.get("axis")
        if isinstance(axis, Mapping) and "wavenumber_cm1" not in lookup:
            if "values" in axis:
                lookup["wavenumber_cm1"] = axis["values"]
        intensity = result.get("intensity")
        if isinstance(intensity, Mapping) and "intensity" not in lookup:
            lookup["intensity"] = intensity
        if isinstance(intensity, Mapping) and "intensity_values" not in lookup:
            if "values" in intensity:
                lookup["intensity_values"] = intensity["values"]
        sticks = result.get("sticks")
        if isinstance(sticks, list):
            if "line_frequencies_cm1" not in lookup:
                positions = [
                    item.get("position")
                    for item in sticks
                    if isinstance(item, Mapping) and "position" in item
                ]
                if len(positions) == len(sticks):
                    lookup["line_frequencies_cm1"] = positions
            if "line_intensities" not in lookup:
                strengths = [
                    item.get("intensity")
                    for item in sticks
                    if isinstance(item, Mapping) and "intensity" in item
                ]
                if len(strengths) == len(sticks):
                    lookup["line_intensities"] = strengths
    result_items: list[dict[str, Any]] = []
    for name in names:
        if name == "hessian_summary":
            summary = _project_hessian_summary(result)
            if summary is None:
                continue
            ref = next_ref()
            result_items.append(_observation(ref, name, summary))
            continue
        if name == "vibrational_mode_summary":
            summary = _project_vibrational_mode_summary(result)
            if summary is None:
                continue
            ref = next_ref()
            result_items.append(_observation(ref, name, summary))
            continue
        if name in {"ir_line_spectrum", "raman_line_spectrum"}:
            summary = _project_line_spectrum_summary(result)
            if summary is None:
                continue
            ref = next_ref()
            result_items.append(_observation(ref, name, summary))
            continue
        if name == "molecules_written":
            count = result.get("molecules_written")
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                continue
            ref = next_ref()
            result_items.append(_observation(ref, name, count))
            continue
        entry = aliases.get(name)
        if entry is None:
            keys, unit = ((name,), None)
        else:
            keys, unit = entry
        declared_unit = declared_units.get(name)
        value_key = next((key for key in keys if key in lookup), None)
        value = _get_nested(lookup, keys)
        if value is _MISSING:
            continue
        # cclib 等互操作结果会逐字段声明单位；该声明优先于静态别名和请求中
        # 的预期单位，避免把 eV、Å 或 km/mol 错报成通用 hartree/cm-1。
        actual_unit = units_by_key.get(value_key) if value_key is not None else None
        if actual_unit is not None:
            unit = actual_unit
        elif declared_unit is not None:
            unit = declared_unit
        if name == "hessian_matrix":
            # Hessian 单位随 backend 合同可为 eV/angstrom^2 或 hartree/bohr^2；
            # 不在 expected schema 中猜测固定单位，直接保留 ActionResult.unit。
            unit = _safe_scalar(lookup.get("unit"))
        safe_value = _safe_scientific_value(value)
        if safe_value is None:
            continue
        ref = next_ref()
        result_items.append(_observation(ref, name, safe_value, unit))
    return result_items


def _program_checks(
    step_ref: str,
    formal_step_id: Any,
    program_validation: Mapping[str, Any] | None,
    result: Mapping[str, Any] | None,
    derive_gaussian_checks: bool,
    next_ref: callable,
) -> list[dict[str, Any]]:
    """把注册检查收缩为不含 artifact 身份的固定程序检查。"""

    registered = (
        program_validation.get("registered_check_results")
        if isinstance(program_validation, Mapping)
        else []
    )
    if not isinstance(registered, list):
        registered = []
    checks: list[dict[str, Any]] = []
    seen_registered_checks: set[tuple[str, str]] = set()
    for item in registered:
        if not isinstance(item, Mapping):
            raise ValueError("registered_check_results 每项必须是 object")
        identity = item.get("check_instance_id")
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError("registered_check_results 缺少合法 check_instance_id")
        identity = identity.strip()
        if isinstance(formal_step_id, str) and identity != formal_step_id and not identity.startswith(formal_step_id + ":"):
            continue
        actual = item.get("actual") if isinstance(item.get("actual"), Mapping) else {}
        safe_actual = {
            key: safe
            for key in (
                "semantic_type", "media_type", "normal_termination", "converged",
                "status", "validation_status", "validation_errors",
            )
            if key in actual
            for safe in (_safe_program_value(actual[key]),)
            if safe is not None
        }
        evaluation = item.get("evaluation")
        status = "passed" if evaluation == "matched" else "failed" if evaluation == "not_matched" else "not_run"
        requirement = item.get("requirement", item.get("expected", item.get("required", {})))
        safe_requirement = (
            _safe_program_value(requirement)
            if isinstance(requirement, (Mapping, list, tuple))
            else {}
        )
        # 同一稳定 check identity 可能在恢复/聚合时重复出现。只有 identity 与
        # 公开科学内容（状态、要求、实际值）都相同才合并；同一检查的新科学值
        # 仍保留为独立事实，避免把变化错误压成一条证据。
        science_content = {
            "status": status,
            "requirement": safe_requirement,
            "observed": safe_actual,
        }
        content_key = json.dumps(
            science_content,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        dedupe_key = (identity, content_key)
        if dedupe_key in seen_registered_checks:
            continue
        seen_registered_checks.add(dedupe_key)
        checks.append(
            {
                "evidence_ref": next_ref(),
                "name": "registered_program_check",
                "status": status,
                "requirement": safe_requirement,
                "observed": safe_actual,
            }
        )
    if derive_gaussian_checks and isinstance(result, Mapping):
        lookup = dict(result)
        for block_name in ("properties", "observables", "metrics", "analysis"):
            block = result.get(block_name)
            if isinstance(block, Mapping):
                for key, value in block.items():
                    lookup.setdefault(key, value)
        registered_observed_keys = {
            key
            for check in checks
            if isinstance(check.get("observed"), Mapping)
            for key in check["observed"]
        }
        for name, keys in (
            ("normal_termination", ("normal_termination", "termination", "terminated_normally")),
            ("optimization_convergence", ("optimization_convergence", "converged", "convergence")),
        ):
            if registered_observed_keys.intersection(keys):
                continue
            value = _get_nested(lookup, keys)
            if value is _MISSING:
                continue
            passed = value is True or (
                isinstance(value, str)
                and value.casefold() in {"true", "passed", "success", "completed", "normal"}
            )
            checks.append(
                {
                    "evidence_ref": next_ref(),
                    "name": name,
                    "status": "passed" if passed else "failed",
                    "requirement": {"expected": True},
                    "observed": {name: _safe_value(value)},
                }
            )
    return checks


def _project_failure_details(source: Mapping[str, Any], error: Mapping[str, Any]) -> dict[str, Any]:
    """提取当前步骤的固定失败诊断，隐藏运行身份和未知错误成员。

    只把失败阶段、可读消息、修复提示、输入要求和重试标记交给专家；不复制
    traceback、路径、Action/Backend、operation、artifact 或任意未注册字段。
    文本使用现有的内部标记清理规则且不截断；字段缺失时保留稳定空值。
    """

    raw_output = source.get("raw_output")
    raw_output = raw_output if isinstance(raw_output, Mapping) else {}

    def safe_text(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        projected = _sanitize_failure_text(value)
        return projected or None

    def safe_code(value: Any) -> str | None:
        projected = safe_text(value)
        if projected is None or "[removed-" in projected:
            return None
        return projected

    def safe_flag(value: Any) -> bool | None:
        return value if isinstance(value, bool) else None

    def first_value(*values: Any) -> Any:
        for value in values:
            if value is not None:
                return value
        return None

    requirements = error.get("input_requirements")
    if isinstance(requirements, (list, tuple)):
        safe_requirements = [
            projected
            for item in requirements
            if (projected := safe_text(item)) is not None
        ]
    else:
        safe_requirements = []

    details = {
        "stage": safe_code(first_value(error.get("failure_stage"), error.get("stage"))),
        "message": safe_text(error.get("message")),
        "repair_guidance": safe_text(error.get("repair_guidance")),
        "input_requirements": safe_requirements,
        "retryable": safe_flag(
            first_value(
                error.get("retryable"),
                source.get("retryable"),
                raw_output.get("retryable"),
            )
        ),
        "retryable_after_input_correction": safe_flag(
            first_value(
                error.get("retryable_after_input_correction"),
                source.get("retryable_after_input_correction"),
                raw_output.get("retryable_after_input_correction"),
            )
        ),
        "retry_requires_changed_request": safe_flag(
            first_value(
                error.get("retry_requires_changed_request"),
                source.get("retry_requires_changed_request"),
                raw_output.get("retry_requires_changed_request"),
            )
        ),
        "status": safe_code(error.get("status")),
        "diagnostic_code": safe_code(error.get("diagnostic_code")),
        "reason_code": safe_code(error.get("reason_code")),
    }
    return {key: details[key] for key in _FAILURE_DETAIL_FIELDS}


def _failure_record(source: Mapping[str, Any], next_ref: callable) -> dict[str, Any]:
    """把失败记录收缩为稳定分类、代码和少量注册结构化细节。"""

    error = source.get("error") if isinstance(source.get("error"), Mapping) else source.get("failure")
    if not isinstance(error, Mapping):
        error = {}
    code = error.get("code", source.get("failure_kind", source.get("status", "execution_failed")))
    category = error.get("category", "execution")
    safe_code = _safe_derived_text(code, "execution_failed")
    safe_category = _safe_derived_text(category, "execution")
    return {
        "evidence_ref": next_ref(),
        "category": _safe_scalar(safe_category) or "execution",
        "code": _safe_scalar(safe_code) or "execution_failed",
        "details": _project_failure_details(source, error),
    }


def build_workflow_scientific_evidence_view(
    *,
    scientific_question: str,
    hypothesis: Mapping[str, Any],
    workflow: Mapping[str, Any],
    execution_results: Mapping[str, Any],
    program_validation: Mapping[str, Any] | None = None,
    scientific_context: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """构造执行后固定 workflow_evidence，并返回 Debate 用扁平证据索引。"""

    workflow_steps = workflow.get("steps")
    if not isinstance(workflow_steps, list) or not workflow_steps:
        raise ValueError("Expert evidence workflow.steps 必须是非空 array")
    result_steps = execution_results.get("steps", [])
    if not isinstance(result_steps, list):
        raise ValueError("Expert evidence execution_results.steps 必须是 array")
    by_id: dict[str, Mapping[str, Any]] = {}
    workflow_ids: set[str] = set()
    for index, item in enumerate(workflow_steps):
        if not isinstance(item, Mapping):
            raise ValueError(f"workflow.steps[{index}] 必须是 object")
        step_id = item.get("step_id")
        if not isinstance(step_id, str) or not step_id.strip():
            raise ValueError(f"workflow.steps[{index}] 缺少非空 step_id")
        if step_id in workflow_ids:
            raise ValueError("Expert evidence workflow.steps 含重复 step_id")
        workflow_ids.add(step_id)
    for index, item in enumerate(workflow_steps):
        depends_on = item.get("depends_on", [])
        if not isinstance(depends_on, list):
            raise ValueError(f"workflow.steps[{index}].depends_on 必须是 array")
        for dependency in depends_on:
            if not isinstance(dependency, str) or dependency not in workflow_ids:
                raise ValueError(f"workflow.steps[{index}] 的 depends_on 无法连接到 workflow step")
    for item in result_steps:
        if not isinstance(item, Mapping):
            raise ValueError("Expert evidence execution_results.steps 每项必须是 object")
        step_id = item.get("step_id")
        if not isinstance(step_id, str) or not step_id.strip():
            raise ValueError("Expert evidence execution_results.steps 缺少非空 step_id")
        if step_id in by_id:
            raise ValueError("Expert evidence execution_results.steps 含重复 step_id")
        if step_id not in workflow_ids:
            raise ValueError("Expert evidence execution_results.steps 含 workflow 外 step_id")
        by_id[step_id] = item
    refs_by_id: dict[str, str] = {}
    for index, item in enumerate(workflow_steps, start=1):
        if not isinstance(item, Mapping) or not isinstance(item.get("step_id"), str):
            continue
        if item["step_id"] in refs_by_id:
            raise ValueError("Expert evidence workflow.steps 含重复 step_id")
        refs_by_id[item["step_id"]] = f"s{index}"
    evidence_index: list[dict[str, Any]] = []

    def allocate(step_ref: str, kind: str, fact: Mapping[str, Any]) -> str:
        reference = f"e{len(evidence_index) + 1}"
        evidence_index.append(
            {
                "evidence_ref": reference,
                "step_ref": step_ref,
                "evidence_kind": kind,
                "summary": json.dumps(fact, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            }
        )
        return reference

    projected_steps: list[dict[str, Any]] = []
    for index, formal in enumerate(workflow_steps, start=1):
        if not isinstance(formal, Mapping):
            raise ValueError(f"workflow.steps[{index - 1}] 必须是 object")
        formal_id = formal.get("step_id")
        source = by_id.get(formal_id) if isinstance(formal_id, str) else None
        step_ref = f"s{index}"
        operation = _operation(formal)
        status = _step_status(source)
        expected = _expected_observables(formal, operation)
        native_preflight = _native_preflight_view(
            operation,
            _source_result(source) if source else None,
        )
        missing = [
            item["name"]
            for item in expected
            if isinstance(item, Mapping) and isinstance(item.get("name"), str)
        ]
        local_facts: list[tuple[str, dict[str, Any]]] = []
        local_counter = 0

        def next_ref() -> str:
            nonlocal local_counter
            local_counter += 1
            return f"__local_{local_counter}"

        raw_observations = (
            _result_observations(operation, _source_result(source), next_ref, expected)
            if source
            else []
        )
        observations: list[dict[str, Any]] = []
        for item in raw_observations:
            reference = allocate(step_ref, "observation", {key: value for key, value in item.items() if key != "evidence_ref"})
            item["evidence_ref"] = reference
            observations.append(item)
            missing = [entry for entry in missing if entry != item["name"]]
        raw_checks = _program_checks(
            step_ref,
            formal_id,
            program_validation,
            _source_result(source) if source else None,
            operation in {"native_gaussian", "geometry_optimization"},
            next_ref,
        )
        checks: list[dict[str, Any]] = []
        for item in raw_checks:
            reference = allocate(step_ref, "program_check", {key: value for key, value in item.items() if key != "evidence_ref"})
            item["evidence_ref"] = reference
            checks.append(item)
        failure = None
        not_executed = None
        if status == "failed" and isinstance(source, Mapping):
            draft = _failure_record(source, next_ref)
            reference = allocate(step_ref, "failure", {key: value for key, value in draft.items() if key != "evidence_ref"})
            draft["evidence_ref"] = reference
            failure = draft
        elif status == "not_executed":
            blockers = [refs_by_id[item] for item in formal.get("depends_on", []) if item in refs_by_id]
            draft = {
                "evidence_ref": next_ref(),
                "reason_code": "blocked_by_prior_failure" if blockers else "not_started",
                "blocked_by": blockers,
            }
            reference = allocate(step_ref, "not_executed", {key: value for key, value in draft.items() if key != "evidence_ref"})
            draft["evidence_ref"] = reference
            not_executed = draft
        if operation == "unknown" or (
            operation == "programmable_analysis" and not expected
        ):
            coverage_status = "unavailable"
            missing = [
                item["name"]
                for item in expected
                if isinstance(item, Mapping) and isinstance(item.get("name"), str)
            ]
        elif status == "not_executed":
            coverage_status = "unavailable"
        elif not missing:
            coverage_status = "complete"
        elif observations or checks:
            coverage_status = "partial"
        else:
            coverage_status = "unavailable"
        scientific_settings = _setting_source(formal, operation, source)
        target = formal.get("target") if isinstance(formal.get("target"), Mapping) else {}
        request = formal.get("request") if isinstance(formal.get("request"), Mapping) else {}
        request_inputs = request.get("inputs") if isinstance(request.get("inputs"), Mapping) else {}
        target_label = target.get("label", formal.get("step_name"))
        target_role = target.get(
            "role",
            formal.get(
                "target_role",
                formal.get("structure_role", request_inputs.get("target_role")),
            ),
        )
        depends_on = [refs_by_id[item] for item in formal.get("depends_on", []) if item in refs_by_id] if isinstance(formal.get("depends_on"), list) else []
        step_record = {
            "step_ref": step_ref,
            "name": _safe_derived_text(formal.get("step_name")),
            "scientific_purpose": _safe_derived_text(formal.get("description")),
            "depends_on": depends_on,
            "operation": operation,
            "target": {
                "label": _safe_derived_text(target_label, None),
                "role": _safe_derived_text(target_role, None),
            },
            "native_preflight": native_preflight,
            "scientific_settings": scientific_settings,
            "expected_observables": expected,
            "status": status,
            "coverage": {"status": coverage_status, "missing_observables": missing},
            "observations": observations,
            "program_checks": checks,
            "failure": failure,
            "not_executed": not_executed,
        }
        projected_steps.append(step_record)
    view = {
        "scientific_question": scientific_question,
        "explicit_conditions": _normalise_conditions(
            scientific_context,
            hypothesis,
            workflow,
        ),
        "hypothesis": _hypothesis_view(hypothesis),
        "workflow_evidence": {"steps": projected_steps},
    }
    return view, evidence_index


def flatten_workflow_evidence_index(view: Mapping[str, Any]) -> list[dict[str, Any]]:
    """从已构造视图按出现顺序生成 Debate 专用 evidence_index。"""

    result: list[dict[str, Any]] = []
    steps = view.get("workflow_evidence", {}).get("steps", []) if isinstance(view.get("workflow_evidence"), Mapping) else []
    for step in steps:
        if not isinstance(step, Mapping):
            continue
        step_ref = step.get("step_ref")
        for field, kind in (("observations", "observation"), ("program_checks", "program_check")):
            for fact in step.get(field, []):
                if isinstance(fact, Mapping) and isinstance(fact.get("evidence_ref"), str):
                    result.append({"evidence_ref": fact["evidence_ref"], "step_ref": step_ref, "evidence_kind": kind, "summary": json.dumps({key: value for key, value in fact.items() if key != "evidence_ref"}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))})
        for field, kind in (("failure", "failure"), ("not_executed", "not_executed")):
            fact = step.get(field)
            if isinstance(fact, Mapping) and isinstance(fact.get("evidence_ref"), str):
                result.append({"evidence_ref": fact["evidence_ref"], "step_ref": step_ref, "evidence_kind": kind, "summary": json.dumps({key: value for key, value in fact.items() if key != "evidence_ref"}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))})
    return result


__all__ = [
    "EXPERT_VIEW_PROJECTOR_VERSION",
    "POST_VIEW_KEYS",
    "PRE_VIEW_KEYS",
    "STEP_VIEW_KEYS",
    "build_gaussian_step_review_view",
    "build_workflow_scientific_evidence_view",
    "flatten_workflow_evidence_index",
]
