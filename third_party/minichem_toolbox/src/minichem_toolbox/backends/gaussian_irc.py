"""共享的已完成 Gaussian IRC 文本解析接口。

本模块只读取已经存在的 Gaussian stdout，不创建作业、不提交进程，也不判断化学机理。
几何、能量和结构身份解析函数由现有 Gaussian 适配器注入，保证旧执行路径与新的离线
解析 Action 使用同一套底层格式解析实现。
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any


_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[DdEe][-+]?\d+)?"
_POINT_MARKER = re.compile(
    r"Point Number\s+(?P<number>\d+)\s+in\s+"
    r"(?P<direction>FORWARD|REVERSE)\s+path direction\.",
    flags=re.IGNORECASE,
)

GeometryParser = Callable[[str, list[str]], list[dict[str, Any]]]
EnergyParser = Callable[[str], float | None]
StructureMetadataPreserver = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


def _native_direction(direction: str) -> str:
    normalized = direction.strip().casefold()
    if normalized == "forward":
        return "forward"
    if normalized == "backward":
        return "reverse"
    raise ValueError("Gaussian IRC direction must be forward or backward")


def parse_gaussian_irc_summary(text: str) -> list[dict[str, float | int]]:
    """解析 Gaussian IRC 的汇总能量和反应坐标表。

    输入是完整 stdout 文本；函数返回能解析出的表格行，缺少表格或单行格式错误时返回
    空列表，不会把汇总表误当作路径结构。
    """

    match = re.search(
        r"Summary of reaction path following\s*\n\s*-+\s*\n"
        r"\s*Energy\s+Rx Coord\s*\n(?P<body>.*?)(?:\n\s*-{5,})",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        return []
    records: list[dict[str, float | int]] = []
    for line in match.group("body").splitlines():
        fields = line.split()
        if len(fields) != 3 or not fields[0].isdigit():
            continue
        try:
            records.append(
                {
                    "summary_index": int(fields[0]),
                    "relative_energy_hartree": float(fields[1].replace("D", "E")),
                    "reaction_coordinate": float(fields[2].replace("D", "E")),
                }
            )
        except ValueError:
            continue
    return records


def parse_gaussian_irc_points(
    text: str,
    symbols: list[str],
    original: dict[str, Any],
    requested_direction: str,
    *,
    parse_geometries: GeometryParser,
    parse_energy: EnergyParser,
    preserve_structure_metadata: StructureMetadataPreserver,
) -> list[dict[str, Any]]:
    """解析指定方向中同时包含结构、能量和接受标记的 IRC 点。

    Gaussian 会在一次点输出中打印多段几何和 SCF 能量；只有同一段中出现反应坐标变化
    与路径点数标记时才接受最后一帧，避免把中间试步当作正式路径点。
    """

    native_direction = _native_direction(requested_direction)
    matches = list(_POINT_MARKER.finditer(text))
    records: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        direction = match.group("direction").casefold()
        if direction != native_direction:
            continue
        stop = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        segment = text[match.end() : stop]
        accepted = re.search(
            r"CHANGE IN THE REACTION COORDINATE\s*=\s*"
            + _FLOAT
            + r".*?# OF POINTS ALONG THE PATH\s*=\s*\d+",
            segment,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if accepted is None:
            continue
        accepted_segment = segment[: accepted.end()]
        geometries = parse_geometries(accepted_segment, symbols)
        energy = parse_energy(accepted_segment)
        if not geometries or energy is None:
            continue
        structure = preserve_structure_metadata(geometries[-1], original)
        records.append(
            {
                "point_number": int(match.group("number")),
                "direction": (
                    "backward" if native_direction == "reverse" else "forward"
                ),
                "structure": structure,
                "energy_hartree": energy,
            }
        )
    return records


def parse_gaussian_irc_termination(
    text: str, requested_direction: str
) -> tuple[bool, str]:
    """判定指定 IRC 分支的完成、PES minimum 和最大点数终止原因。"""

    requested_native = _native_direction(requested_direction).upper()
    requested_start = re.search(
        rf"Point Number\s+\d+\s+in\s+{requested_native}\s+path direction\.",
        text,
        flags=re.IGNORECASE,
    )
    branch_text = text[requested_start.start() :] if requested_start else text
    opposite_native = "FORWARD" if requested_native == "REVERSE" else "REVERSE"
    opposite_start = re.search(
        rf"Point Number\s+\d+\s+in\s+{opposite_native}\s+path direction\.",
        branch_text,
        flags=re.IGNORECASE,
    )
    if opposite_start:
        branch_text = branch_text[: opposite_start.start()]

    completion = re.search(
        rf"Calculation of {requested_native} path complete",
        branch_text,
        flags=re.IGNORECASE,
    )
    before_completion = (
        branch_text[: completion.end()] if completion is not None else branch_text
    )
    if re.search(
        r"Maximum number of (?:steps|points) reached",
        before_completion,
        flags=re.IGNORECASE,
    ):
        return False, "maximum_points_reached"
    if re.search(
        r"PES minimum detected(?: on this side of the pathway)?\b",
        before_completion,
        flags=re.IGNORECASE,
    ):
        return True, "pes_minimum_detected"
    if completion is not None:
        return False, "branch_completed_without_pes_minimum"
    return False, "completion_marker_missing"


def parse_gaussian_irc_output(
    text: str,
    transition_state: dict[str, Any],
    requested_direction: str,
    *,
    parse_geometries: GeometryParser,
    parse_energy: EnergyParser,
    preserve_structure_metadata: StructureMetadataPreserver,
) -> dict[str, Any]:
    """解析一个方向的已完成 native Gaussian IRC stdout。

    返回值保留结构点和机械终止事实；调用方必须根据 ``normal_termination``、点数、
    ``branch_completed`` 和 ``pes_minimum_detected`` 决定 ActionResult 状态，不能把本
    函数的路径事实直接称作已验证路径。
    """

    direction = requested_direction.strip().casefold()
    native_direction = _native_direction(direction)
    marker_directions = {
        match.group("direction").casefold() for match in _POINT_MARKER.finditer(text)
    }
    expected_present = native_direction in marker_directions
    opposite_present = (
        ("reverse" if native_direction == "forward" else "forward")
        in marker_directions
    )
    normal_termination = bool(
        re.search(r"Normal termination of Gaussian", text, flags=re.IGNORECASE)
    )
    error_termination = bool(
        re.search(r"Error termination|Erroneous termination", text, flags=re.IGNORECASE)
    )
    if isinstance(transition_state.get("atoms"), list):
        symbols = [
            str(atom.get("element") or atom.get("symbol"))
            for atom in transition_state.get("atoms") or []
        ]
    else:
        symbols = [str(value) for value in transition_state.get("symbols") or []]
    points = parse_gaussian_irc_points(
        text,
        symbols,
        transition_state,
        direction,
        parse_geometries=parse_geometries,
        parse_energy=parse_energy,
        preserve_structure_metadata=preserve_structure_metadata,
    )
    summary = parse_gaussian_irc_summary(text)
    first_marker = next(_POINT_MARKER.finditer(text), None)
    transition_state_energy = parse_energy(text[: first_marker.start()]) if first_marker else parse_energy(text)

    branch_completed = False
    if error_termination or not normal_termination:
        termination_reason = "abnormal_termination"
    elif not expected_present and opposite_present:
        termination_reason = "direction_mismatch"
    elif not points:
        termination_reason = "no_parseable_points"
    else:
        branch_completed, termination_reason = parse_gaussian_irc_termination(
            text, direction
        )
    # ``maximum_points_reached`` 和缺少 PES minimum 都是可保留的机械事实，
    # 但不是已完成的科学分支；保留终止解析器的布尔值，避免下游把 partial
    # 路径误当作可验证路径。
    pes_minimum_detected = termination_reason == "pes_minimum_detected"
    return {
        "direction": direction,
        "points": points,
        "summary": summary,
        "transition_state_energy_hartree": transition_state_energy,
        "normal_termination": normal_termination and not error_termination,
        "error_termination": error_termination,
        "direction_marker_present": expected_present,
        "opposite_direction_marker_present": opposite_present,
        "observed_native_directions": sorted(marker_directions),
        "completed_points": len(points),
        "branch_completed": branch_completed,
        "pes_minimum_detected": pes_minimum_detected,
        "termination_reason": termination_reason,
    }


__all__ = [
    "parse_gaussian_irc_output",
    "parse_gaussian_irc_points",
    "parse_gaussian_irc_summary",
    "parse_gaussian_irc_termination",
]
