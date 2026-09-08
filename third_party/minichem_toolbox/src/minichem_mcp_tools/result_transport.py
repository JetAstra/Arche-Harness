"""为模型或展示层生成有界的通用 ActionResult 视图。

执行层先把完整 ActionResult 写入 canonical 文件，再由 generic collect 调用本模块生成
有界的模型传输视图。这里的函数不按任务、Action 或 Backend 选择结果，也不改变磁盘上的
科学数值、错误语义或 Artifact 身份。
"""

from __future__ import annotations

import json
from typing import Any


MAX_INLINE_LIST_ITEMS = 160
MAX_INLINE_MAPPING_CHARS = 16_000
MAX_STABLE_SCALAR_CHARS = 8_192
MAX_INLINE_SCALARS_CHARS = 8_000
MAX_INLINE_SCALAR_CHARS = 4_000
MAX_ERROR_CHARS = 4_000

_STRUCTURE_KEYS = {
    "structure",
    "optimized_structure",
    "transition_state",
    "transition_state_candidate",
    "reactant_structure",
    "product_structure",
}
_DENSE_ARRAY_KEYS = {
    "matrix",
    "hessian",
    "modes",
    "normal_modes",
    "trajectory_coordinates",
    "coordinates_by_frame",
    "eigenvectors",
    "density_grid",
    "potential_grid",
}
_POINTER_RESERVED_KEYS = {
    "inline",
    "content",
    "note",
    "artifact_id",
    "semantic_type",
    "path",
    "inline_scalars",
    "field_count",
}


def _compact_artifact(reference: dict[str, Any]) -> dict[str, Any]:
    """保留一个 ArtifactRef 的身份、类型和来源字段。

    输入是服务端返回的 ArtifactRef 映射；函数只创建副本，不写入存储。未知字段不会进入
    有界视图，完整引用仍由执行 wrapper 原样返回；缺少字段时对应键被省略。
    """

    return {
        key: reference[key]
        for key in (
            "artifact_id",
            "semantic_type",
            "media_type",
            "sha256",
            "path",
            "producer_action",
            "producer_backend",
            "parent_artifact_ids",
        )
        if key in reference
    }


def _array_shape(value: Any) -> list[int | str]:
    """返回嵌套列表的维度，并标记第一层发现的非规则数组。"""

    shape: list[int | str] = []
    current = value
    while isinstance(current, list):
        shape.append(len(current))
        if not current:
            break
        first = current[0]
        if any(type(item) is not type(first) for item in current[1:]):
            shape.append("ragged")
            break
        current = first
    return shape


def _artifact_pointer(
    complete_result_artifact: dict[str, Any] | None,
    *,
    content: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构造指向完整结果 Artifact 的有界视图对象。

    输入的 ArtifactRef 和附加摘要不会被修改，也不会触发写入。没有完整 Artifact 时仍返回
    内容摘要；附加值不可序列化的错误由调用方在传输边界处理。
    """

    value: dict[str, Any] = {
        "inline": False,
        "content": content,
        "note": "Complete value is stored in the complete result artifact.",
    }
    if complete_result_artifact:
        for key in ("artifact_id", "semantic_type", "path"):
            if key in complete_result_artifact:
                value[key] = complete_result_artifact[key]
    if extra:
        value.update(extra)
    return value


def _structure_summary(value: Any) -> dict[str, Any]:
    """提取结构值的原子数、元素计数和少量公开标量。"""

    if not isinstance(value, dict):
        return {}
    atoms = value.get("atoms")
    summary: dict[str, Any] = {}
    if isinstance(atoms, list):
        summary["atom_count"] = len(atoms)
        symbols = [
            str(atom.get("element") or atom.get("symbol") or "")
            for atom in atoms
            if isinstance(atom, dict)
        ]
        if symbols and all(symbols):
            counts: dict[str, int] = {}
            for symbol in symbols:
                counts[symbol] = counts.get(symbol, 0) + 1
            summary["element_counts"] = counts
    for key in ("charge", "multiplicity", "pbc", "source_path"):
        if key in value:
            summary[key] = value[key]
    return summary


def _compact_payload(
    value: Any,
    *,
    complete_result_artifact: dict[str, Any] | None,
    key: str | None = None,
) -> Any:
    """递归生成通用、有界的模型视图，不按 Action ID 解释字段。

    结构、稠密数组、超长列表和超大映射会用规模摘要或完整结果 Artifact 指针表示；普通小值
    按原形状复制。函数不修改输入、不写入 Artifact，循环对象或 JSON 编码错误继续向上传播。
    """

    if key in _STRUCTURE_KEYS and isinstance(value, (dict, list)):
        return _artifact_pointer(
            complete_result_artifact,
            content="AtomicStructure",
            extra=_structure_summary(value),
        )
    if key in _DENSE_ARRAY_KEYS and isinstance(value, list):
        return _artifact_pointer(
            complete_result_artifact,
            content=key,
            extra={"shape": _array_shape(value)},
        )
    if isinstance(value, list):
        if len(value) > MAX_INLINE_LIST_ITEMS:
            return _artifact_pointer(
                complete_result_artifact,
                content=key or "array",
                extra={
                    "shape": _array_shape(value),
                    "item_count": len(value),
                    "first_items": value[:3],
                    "last_items": value[-3:],
                },
            )
        return [
            _compact_payload(
                item,
                complete_result_artifact=complete_result_artifact,
                key=key,
            )
            for item in value
        ]
    if isinstance(value, dict):
        compacted = {
            item_key: _compact_payload(
                item,
                complete_result_artifact=complete_result_artifact,
                key=item_key,
            )
            for item_key, item in value.items()
        }
        if len(json.dumps(compacted, ensure_ascii=False, default=str)) > MAX_INLINE_MAPPING_CHARS:
            stable_scalars = {
                item_key: item
                for item_key, item in compacted.items()
                if item_key not in _POINTER_RESERVED_KEYS
                and _is_stable_public_scalar(item)
            }
            compatibility_values = {
                item_key: item
                for item_key, item in compacted.items()
                if item_key not in _POINTER_RESERVED_KEYS and _is_scalar(item)
            }
            return _build_large_mapping_pointer(
                complete_result_artifact=complete_result_artifact,
                content=key or "structured_result",
                field_count=len(value),
                stable_scalars=stable_scalars,
                compatibility_values=compatibility_values,
            )
        return compacted
    return value


def _bounded_text(value: Any, limit: int = MAX_ERROR_CHARS) -> Any:
    """将过长诊断字符串压成首尾摘要；其他值保持原值。"""

    if not isinstance(value, str) or len(value) <= limit:
        return value
    half = (limit - 80) // 2
    return value[:half] + "\n... diagnostic text compacted ...\n" + value[-half:]


def _is_scalar(value: Any) -> bool:
    """判断值是否为 JSON 标量，不转换或修改输入。"""

    return value is None or isinstance(value, (str, bool, int, float))


def _is_stable_public_scalar(value: Any) -> bool:
    """判断标量是否适合直接放在有界映射指针的根级。"""

    if isinstance(value, str):
        return len(value) <= MAX_STABLE_SCALAR_CHARS
    return _is_scalar(value)


def _bounded_inline_scalars(
    values: dict[str, Any],
    *,
    max_chars: int = MAX_INLINE_SCALARS_CHARS,
) -> dict[str, Any]:
    """按字符预算复制标量，长字符串只保留受限首尾摘要。"""

    bounded: dict[str, Any] = {}
    for item_key, item in values.items():
        compacted_item = (
            _bounded_text(item, MAX_INLINE_SCALAR_CHARS)
            if isinstance(item, str)
            else item
        )
        candidate = dict(bounded)
        candidate[item_key] = compacted_item
        if len(json.dumps(candidate, ensure_ascii=False, default=str)) <= max_chars:
            bounded = candidate
    return bounded


def _build_large_mapping_pointer(
    *,
    complete_result_artifact: dict[str, Any] | None,
    content: str,
    field_count: int,
    stable_scalars: dict[str, Any],
    compatibility_values: dict[str, Any],
) -> dict[str, Any]:
    """装配只依赖类型和字符预算的超大映射指针。"""

    def make_pointer(
        selected_scalars: dict[str, Any],
        inline_scalars: dict[str, Any],
    ) -> dict[str, Any]:
        return _artifact_pointer(
            complete_result_artifact,
            content=content,
            extra={
                "inline_scalars": inline_scalars,
                "field_count": field_count,
                **selected_scalars,
            },
        )

    selected_scalars: dict[str, Any] = {}
    for item_key, item in stable_scalars.items():
        candidate = dict(selected_scalars)
        candidate[item_key] = item
        if (
            len(json.dumps(make_pointer(candidate, {}), ensure_ascii=False, default=str))
            <= MAX_INLINE_MAPPING_CHARS
        ):
            selected_scalars = candidate

    inline_scalars: dict[str, Any] = {}
    for item_key, bounded_item in _bounded_inline_scalars(compatibility_values).items():
        candidate = dict(inline_scalars)
        candidate[item_key] = bounded_item
        if (
            len(json.dumps(candidate, ensure_ascii=False, default=str)) <= MAX_INLINE_SCALARS_CHARS
            and len(
                json.dumps(
                    make_pointer(selected_scalars, candidate),
                    ensure_ascii=False,
                    default=str,
                )
            )
            <= MAX_INLINE_MAPPING_CHARS
        ):
            inline_scalars = candidate
    return make_pointer(selected_scalars, inline_scalars)


def _compact_error(error: Any) -> Any:
    """限制错误诊断正文长度，同时隐藏 traceback 正文。"""

    if not isinstance(error, dict):
        return _bounded_text(error)
    return {
        key: (
            "Omitted from the model view; inspect the BackendDiagnostic artifact."
            if key == "traceback" and value
            else _bounded_text(value)
        )
        for key, value in error.items()
    }


def compact_action_result(result: dict[str, Any]) -> dict[str, Any]:
    """为明确需要的模型或展示调用生成通用 ActionResult 视图。

    输入是规范 ActionResult；函数只复制映射并有界压缩结果正文，保留所有 ArtifactRef 的
    类型和身份。它不执行服务端 Action，也不按 Action ID、Backend、任务或字段名选择路线。
    输入缺少 ``status`` 时原样返回，结构或 JSON 序列化错误向调用方传播。
    """

    if not isinstance(result, dict) or "status" not in result:
        return result
    output_artifacts = [
        _compact_artifact(item)
        for item in (result.get("output_artifacts") or [])
        if isinstance(item, dict)
    ]
    scientific_primary = output_artifacts[0] if output_artifacts else None
    complete_result_artifact = next(
        (
            item
            for item in output_artifacts
            if item.get("semantic_type") == "ActionResult"
        ),
        scientific_primary,
    )
    input_artifacts = [
        _compact_artifact(item)
        for item in (result.get("input_artifacts") or [])
        if isinstance(item, dict)
    ]

    value = dict(result)
    value["result"] = _compact_payload(
        result.get("result"),
        complete_result_artifact=complete_result_artifact,
        key="result",
    )
    value["input_artifacts"] = input_artifacts
    value["output_artifacts"] = output_artifacts
    value["error"] = _compact_error(result.get("error"))
    value["warnings"] = [
        _bounded_text(item, 1_500) for item in (result.get("warnings") or [])
    ]
    value["artifact_handoff"] = {
        "primary_output": scientific_primary,
        "matching_input_artifacts": input_artifacts,
        "rule": (
            "Pass an output ArtifactRef only to an input expecting its semantic_type. "
            "When a later Action needs another source value, use the separately listed matching "
            "input ArtifactRef rather than reusing one id for differently typed fields."
        ),
    }
    value["transport"] = {
        "mode": "compact_model_view",
        "scientific_values_changed": False,
        "complete_primary_result": complete_result_artifact,
        "supplementary_artifact_count": max(0, len(output_artifacts) - 1),
        "supplementary_artifact_manifest": "_tool_artifacts/index.jsonl",
    }
    return value


__all__ = ["compact_action_result"]
