"""根据不可修改的历史事实，计算任务当前所处的运行状态。

这些投影可以随时删除并重建，因此只用于查询和调度提速，不能反过来成为历史
事实的来源。
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import re
import sqlite3
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Mapping, Sequence

from .contracts import (
    ContractValidationError,
    canonical_json,
    empty_runtime_checkpoint,
    extract_gaussian_code,
    validate_planning_source_refs,
    validate_attempt_payload,
    validate_source_ref_shape,
    validate_tool_capability_gap_payload,
    validate_retrieval_completed_payload,
    validate_retrieval_pause_payload,
    validate_verification_requirements,
    validate_investigation_batch_review_payload,
    validate_investigation_repair_completed_payload,
    validate_investigation_repair_requested_payload,
    validate_investigation_result_payload,
    validate_investigation_route_marker_payload,
    validate_investigation_session_payload,
    validate_investigation_session_thread_bound_payload,
    validate_investigation_session_thread_migrated_payload,
    validate_investigation_turn_started_payload,
    validate_investigation_turn_waiting_payload,
    validate_investigation_turn_attempt_termination_payload,
    validate_investigation_legacy_waiting_resume_authorization_payload,
    validate_investigation_incomplete_v2_waiting_resume_authorization_payload,
    validate_investigation_task_terminal_payload,
    validate_investigation_task_correction_payload,
    validate_hypothesis_selection_completed_payload,
    validate_investigation_task_failure_payload,
    validate_investigation_evidence_payload,
    validate_task_created_payload,
    get_investigation_combination_count,
    validate_task_review_session_payload,
    validate_task_review_turn_started_payload,
    validate_investigation_repair_turn_started_payload,
)
from chemistry_multiagent.utils.router_evidence import _project_native_foreach_result


# ==================== 阶段记忆允许字段筛选 ====================

_STAGE_MEMORY_PROJECTOR_VERSION = "stage-memory.v1"
# 路径脱敏覆盖任意 POSIX 根、Windows 驱动器、UNC 和 file URI；扫描器只在路径
# 片段内部接受空格，并在自然标点、引号或结构边界处停止，避免吞掉后续句子。
_STAGE_MEMORY_PATH_START_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_.-])file://"
    r"|(?<![A-Za-z0-9_.-])[A-Za-z]:[\\/](?=[^\\/\s,;:)}\]>\"'，；：。！？、）】》]+)"
    r"|(?<![A-Za-z0-9_.:-])(?:\\\\|//)(?=[^\\/\s,;:)}\]>\"'，；：。！？、）】》]+)"
    r"|(?<![A-Za-z0-9_.:/-])/(?![/*\s])(?=[^\\/\s,;:)}\]>\"'，；：。！？、）】》]+)",
    re.IGNORECASE,
)
_STAGE_MEMORY_RELATIVE_PATH_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(?:code|inputs|outputs|artifacts|working_memory|reports?|logs?)/"
    r"[^ ,;)}\]]*",
    re.IGNORECASE,
)
_STAGE_MEMORY_PATH_BOUNDARIES = frozenset(
    "\t\r\n,;:)}]>\"'，；：。！？、）】》"
)
_STAGE_MEMORY_INTERNAL_TEXT_PATTERN = re.compile(
    r"\b(?:artifact[_ -]?ids?|backend[_ -]?ids?|event[_ -]?ids?|"
    r"execution[_ -]?ids?|job[_ -]?ids?|operation[_ -]?instance[_ -]?id|"
    r"operation[_ -]?ids?|record[_ -]?ids?|step[_ -]?ids?|action[_ -]?ids?|"
    r"artifact[_ -]?(?:hash|digest|checksum)|raw[_ -]?(?:output|log)|"
    r"stdout|stderr|traceback)\b"
    r"(?:\s*[-:=]\s*[A-Za-z0-9][A-Za-z0-9_-]*)?"
    r"|\b(?:runtime|command|executable)\s*[:=]\s*[^\s,;)}\]]+"
    r"|(?<![A-Za-z0-9])"
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    r"(?![A-Za-z0-9])"
    r"|(?<![A-Za-z0-9])[0-9a-f]{32,128}(?![A-Za-z0-9])",
    re.IGNORECASE,
)

# chemistry_context 的字段来自 Retrieval 合同以及历史上仍会被恢复的合法
# fixture。这里故意列出注册字段，而不是把未知键当作“可能有用的科学量”递归
# 保留；新增字段必须先进入注册表并补齐投影测试。
_CHEMISTRY_CONTEXT_KEYS = frozenset(
    {
        "reaction_type",
        "reaction_class",
        "mechanistic_goal",
        "suspected_job_types",
        "solvent",
        "temperature",
        "temperature_c",
        "temperature_k",
        "temperature_kelvin",
        "pressure",
        "pressure_pa",
        "charge",
        "multiplicity",
        "species_roles",
        "candidate_elements",
        "needs_ts",
        "needs_irc",
        "needs_excited_state",
        "evidence_gaps",
        "substrate",
        "product",
        "reactants",
        "catalyst",
        "catalysts",
        "target_species",
        "molecule",
        "condition",
        "reaction_conditions",
        "observables",
        "calculation_unit",
        "initial_transition_state_structures",
    }
)
_CHEMISTRY_CONTEXT_NESTED_KEYS = frozenset(
    {
        "name",
        "role",
        "type",
        "label",
        "description",
        "value",
        "unit",
        "conditions",
        "smiles",
        "formula",
        "element",
        "elements",
        "species",
        "reactants",
        "products",
        "substrate",
        "product",
        "catalyst",
        "catalysts",
        "solvent",
        "temperature",
        "temperature_c",
        "temperature_k",
        "pressure",
        "charge",
        "multiplicity",
        "geometry",
        "coordinates",
        "atoms",
        "position_angstrom",
        "needs_ts",
        "needs_irc",
        "needs_excited_state",
        "mechanistic_goal",
        "evidence_gaps",
        "observables",
        "calculation_unit",
        "initial_transition_state_structures",
        "atomic_numbers",
        "coordinates_angstrom",
        "scope",
        "products",
        "transition_state_initial_guess_source",
        "method_requirements",
        "endpoint",
        "transition_state",
        "irc",
        "method",
        "basis",
        "jobs",
        "calcfc",
        "maxpoints",
        "recalc",
    }
)
_INITIAL_TRANSITION_STATE_RECORD_KEYS = frozenset(
    {"atomic_numbers", "coordinates_angstrom", "charge", "multiplicity"}
)


def _project_registered_value(
    value: Any,
    allowed_keys: frozenset[str] | set[str],
    *,
    nested_keys: frozenset[str] | set[str] | None = None,
) -> Any:
    """递归复制已注册字段；未知对象键始终丢弃，字符串不截断。"""

    nested_allowed = nested_keys if nested_keys is not None else allowed_keys
    if isinstance(value, Mapping):
        projected: Dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or key not in allowed_keys:
                continue
            if key == "initial_transition_state_structures":
                child = _project_initial_transition_state_structures(item)
            else:
                child = _project_registered_value(
                    item,
                    nested_allowed,
                    nested_keys=nested_allowed,
                )
            if child is not None:
                projected[key] = child
        return projected
    if isinstance(value, (list, tuple)):
        return [
            _project_registered_value(
                item,
                nested_allowed,
                nested_keys=nested_allowed,
            )
            for item in value
        ]
    if isinstance(value, str):
        # 科学正文只有在显式字段（问题、Query、hypothesis reasoning）中放行；
        # 这些派生值仍需去除路径，防止错误消息把本机目录带入模型上下文。
        return _project_stage_derived_text(value)
    if value is None or isinstance(value, (bool, int)):
        return deepcopy(value)
    if isinstance(value, float):
        return deepcopy(value) if math.isfinite(value) else None
    return None


def _project_initial_transition_state_structures(value: Any) -> Dict[str, Any]:
    """投影带任意来源身份的 AtomicStructure 映射并保留其 canonical 字段。"""

    if not isinstance(value, Mapping):
        return {}
    projected: Dict[str, Any] = {}
    for identity, record in value.items():
        if not isinstance(identity, str) or re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", identity
        ) is None:
            continue
        if not isinstance(record, Mapping):
            continue
        projected_record: Dict[str, Any] = {}
        for field in _INITIAL_TRANSITION_STATE_RECORD_KEYS:
            if field not in record:
                continue
            projected_record[field] = _project_registered_value(
                record[field],
                _CHEMISTRY_CONTEXT_NESTED_KEYS,
                nested_keys=_CHEMISTRY_CONTEXT_NESTED_KEYS,
            )
        if projected_record:
            projected[identity] = projected_record
    return projected


def _project_chemistry_context(value: Any) -> Dict[str, Any]:
    """按 chemistry_context 注册表投影科学条件，并隐藏未知嵌套字段。"""

    if not isinstance(value, Mapping):
        return {}
    return _project_registered_value(
        value,
        _CHEMISTRY_CONTEXT_KEYS,
        nested_keys=_CHEMISTRY_CONTEXT_NESTED_KEYS,
    )


def _consume_stage_memory_path(value: str, start: int, prefix_end: int) -> int:
    """从路径起点读取到边界，允许路径段内部空格但不吞掉后文。"""

    length = len(value)
    index = prefix_end
    has_component = False
    while index < length:
        char = value[index]
        if char in _STAGE_MEMORY_PATH_BOUNDARIES:
            break
        if char in "/\\":
            index += 1
            continue
        if char in " \t":
            space_end = index
            while space_end < length and value[space_end] in " \t":
                space_end += 1
            if space_end >= length or value[space_end] in _STAGE_MEMORY_PATH_BOUNDARIES:
                break
            token_end = space_end
            while token_end < length:
                token_char = value[token_end]
                if token_char in _STAGE_MEMORY_PATH_BOUNDARIES or token_char in " /\\\t":
                    break
                token_end += 1
            next_nonspace = token_end
            while next_nonspace < length and value[next_nonspace] in " \t":
                next_nonspace += 1
            has_separator_continuation = (
                next_nonspace < length and value[next_nonspace] in "/\\"
            )
            token = value[space_end:token_end]
            has_file_suffix = (
                "." in token
                and not token.startswith(".")
                and token.rsplit(".", 1)[-1] != ""
            )
            if not (has_separator_continuation or has_file_suffix):
                break
            index = space_end
            continue
        has_component = True
        index += 1
    end = index
    while end > prefix_end and value[end - 1] in " \t":
        end -= 1
    return end if has_component else start


def _replace_stage_memory_paths(value: str) -> str:
    """替换绝对路径并保留自然边界之后的普通文本。"""

    pieces: list[str] = []
    cursor = 0
    for match in _STAGE_MEMORY_PATH_START_PATTERN.finditer(value):
        start, match_end = match.span()
        if start < cursor:
            continue
        if value.casefold().startswith("file://", start):
            prefix_end = match_end
        elif len(value) >= start + 3 and value[start + 1] == ":":
            prefix_end = start + 3
        elif value.startswith("\\\\", start) or value.startswith("//", start):
            prefix_end = start + 2
        else:
            prefix_end = start + 1
        end = _consume_stage_memory_path(value, start, prefix_end)
        if end <= match_end:
            continue
        pieces.append(value[cursor:start])
        pieces.append("[removed-path]")
        cursor = end
    pieces.append(value[cursor:])
    return "".join(pieces)


def _contains_stage_memory_path(value: str) -> bool:
    """判断文本是否含绝对或受控相对路径。"""

    return bool(
        _STAGE_MEMORY_PATH_START_PATTERN.search(value)
        or _STAGE_MEMORY_RELATIVE_PATH_PATTERN.search(value)
    )


def _project_stage_derived_text(value: Any, default: str = "") -> str:
    """清理裁判/失败等派生说明中的路径和内部运行标记。"""

    if not isinstance(value, str):
        return default
    sanitized = _replace_stage_memory_paths(value)
    sanitized = _STAGE_MEMORY_RELATIVE_PATH_PATTERN.sub("[removed-path]", sanitized)
    return _STAGE_MEMORY_INTERNAL_TEXT_PATTERN.sub("[removed-internal]", sanitized)


def _project_stage_token(value: Any, default: str) -> str:
    """读取失败分类/代码；含内部协议标记时回退到稳定占位。"""

    text = _project_stage_derived_text(value, default)
    if "[removed-internal]" in text or "[removed-path]" in text:
        return default
    return text or default


def project_hypothesis_regeneration_context(
    *,
    scientific_question: Any,
    initial_query_angles: Any,
    initial_retrieval_phase: Any,
    chemistry_context: Any,
) -> Dict[str, Any]:
    """构造 Hypothesis Regeneration 固定的冻结生成上下文。

    只读取问题、冻结 Query、检索结论和有效化学条件；检索清单身份、文件路径、
    原始响应及运行控制字段不会进入返回对象。每次调用都返回相同的字段骨架。
    """

    if not isinstance(scientific_question, str) or not scientific_question.strip():
        raise ContractValidationError("regeneration scientific_question 必须是非空文本")
    if not isinstance(initial_query_angles, list):
        raise ContractValidationError("regeneration initial_query_angles 必须是数组")
    if not all(isinstance(item, str) and item.strip() for item in initial_query_angles):
        raise ContractValidationError("regeneration initial_query_angles 内容非法")
    retrieval = initial_retrieval_phase if isinstance(initial_retrieval_phase, Mapping) else {}
    chemistry = chemistry_context if isinstance(chemistry_context, Mapping) else {}
    retrieval_view = {
        "keywords": _project_scientific_registered_value(retrieval.get("keywords", [])),
        "literature_review": _project_stage_derived_text(
            retrieval.get("literature_review", "")
        ),
        "mechanistic_clues": _project_scientific_registered_value(
            retrieval.get("mechanistic_clues", [])
        ),
        "limitations": _project_scientific_registered_value(retrieval.get("limitations", [])),
        "chemistry_context": _project_chemistry_context(
            retrieval.get("chemistry_context", chemistry)
        ),
    }
    return {
        "scientific_question": scientific_question,
        "initial_query_angles": list(initial_query_angles),
        "initial_retrieval_phase": retrieval_view,
        "chemistry_context": _project_chemistry_context(chemistry),
    }


def _stage_memory_short_evidence_refs(
    evidence_refs: Any,
    *,
    step_count: int = 0,
) -> tuple[list[str], Dict[str, str]]:
    """把 Router 证据指针改写成当前视图内的短引用。"""

    refs = evidence_refs if isinstance(evidence_refs, list) else []
    mapping: Dict[str, str] = {}
    output: list[str] = []
    next_index = max(1, step_count + 1)
    for ref in refs:
        if not isinstance(ref, str) or not ref.strip():
            continue
        # 只接受 Router 已注册的步骤/检查指针，或当前视图已经生成的合法 eN。
        # 任意字符串、e0、重复指针都不能被重新包装成看似有效的证据引用。
        if not (
            re.fullmatch(r"e[1-9][0-9]*", ref)
            or re.search(r"/steps/[0-9]+(?:/|$)", ref)
            or re.search(r"/program_validation/registered_check_results/[0-9]+(?:/|$)", ref)
        ):
            continue
        if ref in mapping:
            continue
        match = re.search(r"/steps/(\d+)", ref)
        if match is not None:
            short = f"e{int(match.group(1)) + 1}"
        else:
            short = f"e{next_index}"
            next_index += 1
        while short in mapping.values():
            short = f"e{next_index}"
            next_index += 1
        mapping[ref] = short
        output.append(short)
    return output, mapping


# 以下是正式阶段视图使用的显式 allowlist。递归兼容函数只服务于检索条件、缺口正文
# 等没有固定化学字段的旧记录；工作流和执行账本必须经过这里的固定字段。
_SCIENTIFIC_SETTING_KEYS = {
    "charge",
    "multiplicity",
    "method",
    "basis",
    "dispersion",
    "solvent_model",
    "solvent",
    "temperature",
    "temperature_kelvin",
    "pressure",
    "pressure_pa",
    "integration_grid",
    "grid",
    "scf_convergence",
    "optimization",
    "frequency",
    "route",
    "route_section",
    "route_parameters",
    "calculation_type",
    "analysis_method",
    "standard_state",
    "entropy_model",
    "enthalpy_model",
    "frequency_scale_factor",
    "zpe_scale_factor",
    "imaginary_frequency_policy",
    "symmetry_correction",
    "population_basis",
    "deduplicate_structures",
    "geometry",
    "spin",
    "fragment",
    "fragments",
    "label",
    "role",
}
_SCIENTIFIC_OBSERVATION_KEYS = {
    "molecules_written",
    "gaussian_code",
    "gaussian_route",
    "gaussian_input_deck",
    "input_deck",
    "route",
    "energy",
    "electronic_energy",
    "electronic_energies",
    "zero_point_energy",
    "enthalpy",
    "entropy",
    "gibbs_free_energy",
    "relative_gibbs_free_energies",
    "activation_free_energies",
    "delta_delta_g",
    "enantiomeric_ratio",
    "enantiomeric_excess",
    "rate_determining_step",
    "optimized_geometry",
    "hessian_summary",
    "vibrational_mode_summary",
    "ir_line_spectrum",
    "raman_line_spectrum",
    "frequencies_cm1",
    "wavenumber_cm1",
    "intensity",
    "line_frequencies_cm1",
    "line_intensities",
    "final_atomic_structure",
    "geometry",
    "coordinates",
    "vibrational_frequencies",
    "frequencies",
    "imaginary_frequencies",
    "imaginary_frequency_count",
    "vibrational_intensities",
    "ir_intensities",
    "raman_intensities",
    "spectrum",
    "peaks",
    "distortion_energy",
    "interaction_energy",
    "total_activation_energy",
    "pathway_energy_difference",
    "deck_preflight_status",
    "observed_support",
    "normal_termination",
    "optimization_convergence",
    "converged",
    "status",
    "value",
    "unit",
    "conditions",
}
_SCIENTIFIC_CHECK_KEYS = {
    "normal_termination",
    "optimization_convergence",
    "converged",
    "syntax_valid",
    "frequency_count",
    "imaginary_frequency_count",
    "required_observables",
}
_SCIENTIFIC_NESTED_KEYS = frozenset(
    _SCIENTIFIC_SETTING_KEYS
    | _SCIENTIFIC_OBSERVATION_KEYS
    | _SCIENTIFIC_CHECK_KEYS
    | {
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
        "atoms",
        "element",
        "position_angstrom",
        "coordinates",
        "formula",
        "count",
        "reason",
        "category",
        "code",
        "details",
        # 失败事实的固定、可行动字段；这些字段只在 failure.details
        # 中由显式读取逻辑填充，未知错误成员不会递归进入视图。
        "stage",
        "message",
        "repair_guidance",
        "input_requirements",
        "diagnostic_code",
        "reason_code",
        "retryable",
        "retryable_after_input_correction",
        "retry_requires_changed_request",
        "expected",
        "actual",
        "status",
        "evidence_ref",
        # native foreach 的科学来源标签；只描述观测属于哪个 pair/candidate，
        # 不包含 operation、job 或路径身份。
        "pair_id",
        "candidate_id",
        "direction",
        "role",
        "label",
        "pair_index",
        "candidate_index",
    }
)

_NATIVE_FOREACH_PROVENANCE_KEYS = frozenset(
    {
        "pair_id",
        "candidate_id",
        "direction",
        "role",
        "label",
        "pair_index",
        "candidate_index",
    }
)
_NATIVE_FOREACH_OBSERVATION_NAMES = tuple(
    sorted(
        _SCIENTIFIC_OBSERVATION_KEYS
        - {"status", "value", "unit", "conditions"}
    )
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
_SAFE_FAILURE_DIAGNOSTIC_DETAIL_FIELDS = (
    "missing_key",
    "rejected_route_field",
    "rejected_value",
    "recommended_value",
)
_ALLOWED_PROGRAM_DECISIONS = {
    "evidence_ready",
    "revise_workflow",
    "revise_hypothesis",
    "tool_capability_gap",
    "reject",
    "pause",
}
_OPERATION_NAMES = {
    "smiles2sdf": "structure_conversion",
    "generate_gaussian_code": "generate_gaussian_route",
    "xyz_to_gjf": "prepare_gaussian_deck",
    "sdf_to_gjf": "prepare_gaussian_deck",
    "parse_quantum_chemistry_output": "parse_quantum_chemistry",
    "derive_thermochemistry": "derive_thermochemistry",
    "analyze_reaction_free_energy_profile": "analyze_free_energy_profile",
    "analyze_thermochemical_selectivity": "analyze_selectivity",
    "distortion_interaction_analysis": "analyze_distortion_interaction",
    "optimize_geometry": "geometry_optimization",
    "calculate_hessian": "hessian_calculation",
    "derive_vibrational_modes": "vibrational_modes",
    "derive_ir_spectrum": "ir_spectrum",
    "derive_raman_spectrum": "raman_spectrum",
    "render_spectrum_plot": "spectrum_plot",
}
_OPERATION_OBSERVABLES = {
    "structure_conversion": ["molecules_written"],
    "generate_gaussian_route": ["gaussian_route"],
    "prepare_gaussian_deck": ["deck_preflight_status"],
    "parse_quantum_chemistry": [
        "final_atomic_structure",
        "electronic_energies",
        "vibrational_frequencies",
        "vibrational_intensities",
        "charge",
        "multiplicity",
    ],
    "derive_thermochemistry": [
        "zero_point_energy",
        "enthalpy",
        "entropy",
        "gibbs_free_energy",
    ],
    "analyze_free_energy_profile": [
        "relative_gibbs_free_energies",
        "activation_free_energies",
    ],
    "analyze_selectivity": [
        "delta_delta_g",
        "enantiomeric_ratio",
        "enantiomeric_excess",
    ],
    "analyze_distortion_interaction": [
        "distortion_energy",
        "interaction_energy",
        "total_activation_energy",
    ],
    "geometry_optimization": ["optimized_geometry"],
    "hessian_calculation": ["hessian_summary"],
    "vibrational_modes": ["vibrational_mode_summary"],
    "ir_spectrum": ["ir_line_spectrum"],
    "raman_spectrum": ["raman_line_spectrum"],
    "spectrum_plot": [],
}
_SCIENTIFIC_SETTINGS_BY_OPERATION = {
    "structure_conversion": (),
    "generate_gaussian_route": ("requested_route_intent", "route_kind"),
    "prepare_gaussian_deck": ("charge", "multiplicity", "input_format"),
    "geometry_optimization": (
        "method", "basis", "optimizer", "force_field", "optimization_level",
        "fmax_ev_per_angstrom", "max_steps", "max_iterations", "force_tolerance",
        "energy_tolerance", "constraints",
    ),
    "hessian_calculation": ("method", "basis", "displacement_angstrom", "coordinate_system"),
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
    "run_native_quantum_chemistry": (
        "calculation_type", "method", "basis", "dispersion", "solvent_model",
        "solvent", "temperature", "integration_grid", "scf_convergence",
        "optimization_options", "frequency_options", "charge", "multiplicity",
    ),
    "parse_quantum_chemistry": (
        "coordinate_frames", "properties", "include_excited_state_configurations",
        "include_orbital_coefficients",
    ),
    "derive_thermochemistry": (
        "temperature_kelvin", "pressure_pa", "geometry", "spin", "symmetry_number",
        "ignore_imaginary_modes", "standard_state", "entropy_model", "enthalpy_model",
        "frequency_scale_factor", "zpe_scale_factor", "imaginary_frequency_policy",
        "symmetry_correction",
    ),
    "analyze_free_energy_profile": (
        "temperature_kelvin", "standard_state", "entropy_model", "enthalpy_model",
        "frequency_scale_factor", "zpe_scale_factor", "imaginary_frequency_policy",
        "symmetry_correction", "population_basis", "deduplicate_structures",
    ),
    "analyze_selectivity": (
        "temperature_kelvin", "standard_state", "entropy_model", "enthalpy_model",
        "frequency_scale_factor", "zpe_scale_factor", "imaginary_frequency_policy",
        "symmetry_correction", "population_basis", "deduplicate_structures",
    ),
    "run_programmable_analysis": ("analysis_method",),
}


_SCIENTIFIC_PATCH_VALUE_KEYS = (
    _SCIENTIFIC_SETTING_KEYS
    | _SCIENTIFIC_OBSERVATION_KEYS
    | {
        "description",
        "step_name",
        "depends_on",
        "expected_observables",
        "scientific_purpose",
        "operation",
        "target",
    }
)


def _finite_numeric_list(value: Any) -> list[float] | None:
    """校验并复制一维有限数值数组；非法值不会进入记忆账本。"""

    if not isinstance(value, list):
        return None
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return None
        number = float(item)
        if not math.isfinite(number):
            return None
        result.append(number)
    return result


def _contains_nonfinite_numeric(value: Any) -> bool:
    """递归检查科学事实中是否含有非有限数值。

    输入是任意科学事实值；动作递归检查映射、列表和元组中的浮点数，不修改输入。
    返回 ``True`` 表示发现 ``NaN`` 或 ``Infinity``，否则返回 ``False``。其他标量
    类型不视为错误，读取过程中不会抛出异常。
    """

    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, Mapping):
        return any(_contains_nonfinite_numeric(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_nonfinite_numeric(item) for item in value)
    return False


def _project_hessian_summary(raw_result: Mapping[str, Any]) -> Dict[str, Any] | None:
    """把 Hessian 密集矩阵归约为形状、元素数、单位、有限性和对称性。"""

    matrix = raw_result.get("matrix", raw_result.get("hessian_matrix"))
    if not isinstance(matrix, list) or not matrix or not all(isinstance(row, list) for row in matrix):
        return None
    rows = len(matrix)
    columns = len(matrix[0])
    if columns == 0 or any(len(row) != columns for row in matrix):
        return None
    values: list[list[float]] = []
    for row in matrix:
        numbers = _finite_numeric_list(row)
        if numbers is None:
            return None
        values.append(numbers)
    symmetric = rows == columns and all(
        math.isclose(values[row][column], values[column][row], rel_tol=1e-9, abs_tol=1e-12)
        for row in range(rows)
        for column in range(rows)
    )
    return {
        "shape": [rows, columns],
        "element_count": rows * columns,
        "unit": _project_stage_token(raw_result.get("unit"), "") or None,
        "finite": True,
        "symmetric": symmetric,
    }


def _project_vibrational_mode_summary(raw_result: Mapping[str, Any]) -> Dict[str, Any] | None:
    """保留频率总数、实/虚频计数及完整原始虚频数值对。"""

    frequencies = raw_result.get("frequencies_cm1")
    if not isinstance(frequencies, list):
        return None
    imaginary: list[Dict[str, float]] = []
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
            normalized = {"real": float(real), "imaginary": float(imaginary_part)}
            if float(imaginary_part) != 0.0:
                imaginary.append(normalized)
            else:
                real_count += 1
            continue
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return None
        number = float(item)
        if not math.isfinite(number):
            return None
        real_count += 1
    return {
        "total_count": len(frequencies),
        "real_count": real_count,
        "imaginary_count": len(imaginary),
        "frequency_unit": "cm-1",
        "imaginary_frequencies_cm1": imaginary,
    }


def _project_line_spectrum_summary(raw_result: Mapping[str, Any]) -> Dict[str, Any] | None:
    """仅保留完整对齐线谱及展宽/计数信息，不复制 dense 网格。"""

    axis = raw_result.get("axis")
    axis_values = axis.get("values") if isinstance(axis, Mapping) else raw_result.get("wavenumber_cm1")
    intensity = raw_result.get("intensity")
    if isinstance(intensity, Mapping):
        intensity_values = intensity.get("values")
        intensity_unit = intensity.get("unit")
    else:
        intensity_values = raw_result.get("intensity_values", intensity)
        intensity_unit = raw_result.get("intensity_unit")
    dense_axis = _finite_numeric_list(axis_values) if axis_values is not None else None
    dense_intensity = _finite_numeric_list(intensity_values) if intensity_values is not None else None
    if dense_axis is not None and dense_intensity is not None and len(dense_axis) != len(dense_intensity):
        return None

    sticks = raw_result.get("sticks")
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
    line_frequencies = _finite_numeric_list(raw_result.get("line_frequencies_cm1"))
    if line_frequencies is None:
        line_frequencies = stick_frequencies
    line_intensities = _finite_numeric_list(raw_result.get("line_intensities"))
    if line_intensities is None:
        line_intensities = stick_intensities
    if line_frequencies is None or line_intensities is None or len(line_frequencies) != len(line_intensities):
        return None
    if stick_frequencies is not None and (
        stick_frequencies != line_frequencies or stick_intensities != line_intensities
    ):
        return None

    point_count = raw_result.get("point_count")
    if point_count is None:
        point_count = len(dense_axis) if dense_axis is not None else None
    if not isinstance(point_count, int) or isinstance(point_count, bool) or point_count < 0:
        return None
    if dense_axis is not None and point_count != len(dense_axis):
        return None
    stick_count = raw_result.get("stick_count")
    if stick_count is None:
        stick_count = len(line_frequencies)
    if not isinstance(stick_count, int) or isinstance(stick_count, bool) or stick_count < 0:
        return None
    if stick_count != len(line_frequencies):
        return None
    line_shape = raw_result.get("line_shape")
    broadening = raw_result.get("broadening")
    fwhm = raw_result.get("fwhm_cm1", raw_result.get("fwhm"))
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
        "intensity_unit": _project_stage_token(intensity_unit, "") or None,
        "broadening": _project_stage_token(broadening, "") or None,
        "fwhm_cm1": fwhm,
        "point_count": point_count,
        "stick_count": stick_count,
    }


def _project_scientific_patch_value(value: Any) -> Any:
    """只保留补丁中的科学参数、观测和短目标字段。"""

    if isinstance(value, Mapping):
        return {
            key: _project_scientific_patch_value(item)
            for key, item in value.items()
            if isinstance(key, str) and key in _SCIENTIFIC_PATCH_VALUE_KEYS
        }
    if isinstance(value, list):
        return [_project_scientific_patch_value(item) for item in value]
    if isinstance(value, str) and _contains_stage_memory_path(value.strip()):
        return None
    if value is None or isinstance(value, (str, int, bool)):
        return deepcopy(value)
    if isinstance(value, float):
        return deepcopy(value) if math.isfinite(value) else None
    return None


def _project_scientific_registered_value(value: Any) -> Any:
    """投影账本观测值，只允许注册科学字段及其固定摘要成员。"""

    if _contains_nonfinite_numeric(value):
        return None

    if isinstance(value, Mapping):
        return _project_registered_value(
            value,
            _SCIENTIFIC_NESTED_KEYS,
            nested_keys=_SCIENTIFIC_NESTED_KEYS,
        )
    if isinstance(value, (list, tuple)):
        projected = []
        for item in value:
            child = _project_scientific_registered_value(item)
            if child is None and item is not None:
                return None
            projected.append(child)
        return projected
    if isinstance(value, str):
        return _project_stage_derived_text(value)
    if value is None or isinstance(value, (bool, int)):
        return deepcopy(value)
    if isinstance(value, float):
        return deepcopy(value) if math.isfinite(value) else None
    return None


def _project_failure_details(
    error: Any,
    *,
    step: Mapping[str, Any] | None = None,
    raw_output: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """从当前步骤的 canonical error 提取固定的可行动失败字段。

    输入是错误对象、当前步骤和可选原始输出；动作只读取已注册的诊断、修复提示、
    有限诊断字段和候选修复，并把后三者合并为既有 ``repair_guidance``；traceback、
    路径、Action/Backend、operation、artifact 以及其他未知成员不会进入结果。字符串经过阶段视图已有
    的路径和内部标记清理，保留完整正文，不按字符数截断。即使来源缺少字段，
    返回对象仍保留完整字段骨架，便于 Revision 模型稳定解析；输入不是映射时按空
    错误处理，清理过程不会抛出原始异常。
    """

    error_value = error if isinstance(error, Mapping) else {}
    step_value = step if isinstance(step, Mapping) else {}
    output_value = raw_output if isinstance(raw_output, Mapping) else {}

    def first_value(*values: Any) -> Any:
        for value in values:
            if value is not None:
                return value
        return None

    def safe_text(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        projected = _project_stage_derived_text(value)
        return projected if projected else None

    def safe_code(value: Any) -> str | None:
        if value is None:
            return None
        projected = _project_stage_token(value, "")
        return projected or None

    def safe_flag(value: Any) -> bool | None:
        return value if isinstance(value, bool) else None

    def safe_diagnostic_value(value: Any) -> str | None:
        """把允许公开的诊断标量转为可读文本，拒绝非有限数和复杂对象。"""

        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            return str(value) if math.isfinite(value) else None
        return safe_text(value)

    guidance_parts: list[str] = []
    explicit_guidance = safe_text(error_value.get("repair_guidance"))
    if explicit_guidance is not None:
        guidance_parts.append(explicit_guidance)

    diagnostic = error_value.get("details")
    if isinstance(diagnostic, Mapping):
        diagnostic_fields = []
        for field in _SAFE_FAILURE_DIAGNOSTIC_DETAIL_FIELDS:
            projected = safe_diagnostic_value(diagnostic.get(field))
            if projected is not None:
                diagnostic_fields.append(f"{field}={projected}")
        if diagnostic_fields:
            guidance_parts.append("诊断字段：" + "；".join(diagnostic_fields) + "。")

    candidate_fixes = error_value.get("candidate_fixes")
    if isinstance(candidate_fixes, (list, tuple)):
        for item in candidate_fixes[:8]:
            projected = safe_text(item)
            if projected is not None and projected not in guidance_parts:
                guidance_parts.append(projected)
    repair_guidance = "\n".join(guidance_parts) or None

    requirements = error_value.get("input_requirements")
    if isinstance(requirements, (list, tuple)):
        safe_requirements = [
            projected
            for item in requirements
            if (projected := safe_text(item)) is not None
        ]
    else:
        safe_requirements = []

    details = {
        "stage": safe_code(
            first_value(error_value.get("failure_stage"), error_value.get("stage"))
        ),
        "message": safe_text(error_value.get("message")),
        "repair_guidance": repair_guidance,
        "input_requirements": safe_requirements,
        "retryable": safe_flag(
            first_value(
                error_value.get("retryable"),
                step_value.get("retryable"),
                output_value.get("retryable"),
            )
        ),
        "retryable_after_input_correction": safe_flag(
            first_value(
                error_value.get("retryable_after_input_correction"),
                step_value.get("retryable_after_input_correction"),
                output_value.get("retryable_after_input_correction"),
            )
        ),
        "retry_requires_changed_request": safe_flag(
            first_value(
                error_value.get("retry_requires_changed_request"),
                step_value.get("retry_requires_changed_request"),
                output_value.get("retry_requires_changed_request"),
            )
        ),
        "status": safe_code(error_value.get("status")),
        "diagnostic_code": safe_code(error_value.get("diagnostic_code")),
        "reason_code": safe_code(error_value.get("reason_code")),
    }
    return {key: details[key] for key in _FAILURE_DETAIL_FIELDS}


def _project_scientific_settings(
    step: Mapping[str, Any],
    operation: str,
    raw_result: Mapping[str, Any] | None,
) -> Dict[str, Dict[str, Any]]:
    """按操作注册字段记录执行确认、请求值或不可用状态。

    输入是步骤、操作名和可选原始结果；动作从请求与结果中读取注册科学字段，优先
    保存执行后确认值，并为缺失字段写入固定的 ``unavailable``。返回值是字段到
    ``value/source`` 的映射；未知操作返回空映射，非法或非有限值按缺失处理，不会
    修改输入或抛出原始解析异常。
    """

    fields = _SCIENTIFIC_SETTINGS_BY_OPERATION.get(operation, ())
    if not fields:
        return {}
    request = step.get("request") if isinstance(step.get("request"), Mapping) else {}
    requested: Dict[str, Any] = {}
    sections = [
        request.get(name) if isinstance(request.get(name), Mapping) else {}
        for name in ("inputs", "method_spec", "action_settings")
    ]
    sections.append(request)
    for value in sections:
        for field in fields:
            if field in value and field not in requested:
                projected = _project_scientific_registered_value(value[field])
                if projected is not None:
                    requested[field] = projected
    effective: Dict[str, Any] = {}
    source = raw_result if isinstance(raw_result, Mapping) else {}
    for block_name in ("effective_settings", "settings", "method_spec", "action_settings"):
        block = source.get(block_name)
        if isinstance(block, Mapping):
            for field, value in block.items():
                effective.setdefault(field, value)
    projected_settings: Dict[str, Dict[str, Any]] = {}
    for field in fields:
        effective_value = effective.get(field)
        projected_effective = _project_scientific_registered_value(effective_value)
        if field in effective and projected_effective is not None:
            projected_settings[field] = {
                "value": projected_effective,
                "source": "verified_effective",
            }
        elif field in requested:
            projected_settings[field] = {
                "value": requested[field],
                "source": "requested_only",
            }
        else:
            projected_settings[field] = {"value": None, "source": "unavailable"}
    return projected_settings


def _project_native_preflight(
    operation: str,
    raw_output: Mapping[str, Any],
    raw_result: Mapping[str, Any],
) -> Dict[str, Any]:
    """整理 native_preflight 的公开字段并保留明确的未就绪状态。

    输入是操作名、执行输出和原始结果；动作只读取预检是否就绪、计算意图、Link1
    数量和固定检查名，清除其他运行字段。返回值是固定字段结构，非预检操作或缺少
    预检事实时返回 ``unavailable``，明确 ``False`` 时返回 ``not_ready``；输入形状
    不符合预期不会把未知内容带入结果。
    """

    unavailable = {
        "status": "unavailable",
        "ready": False,
        "calculation_intent": None,
        "link1_segment_count": None,
        "checks": [],
    }
    if operation not in {"prepare_gaussian_deck", "run_native_quantum_chemistry"}:
        return unavailable
    native = raw_result.get("native_preflight")
    if not isinstance(native, Mapping):
        native = raw_output.get("native_preflight")
    has_fact = isinstance(native, Mapping) or isinstance(
        raw_result.get("native_preflight_ready", raw_output.get("native_preflight_ready")), bool
    )
    if not has_fact:
        return unavailable
    ready_value = native.get("native_preflight_ready") if isinstance(native, Mapping) else None
    if not isinstance(ready_value, bool):
        candidate = raw_result.get("native_preflight_ready", raw_output.get("native_preflight_ready"))
        ready_value = candidate if isinstance(candidate, bool) else False
    count = native.get("link1_segment_count") if isinstance(native, Mapping) else None
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        count = None
    checks = native.get("checks") if isinstance(native, Mapping) else []
    safe_checks = [
        _project_stage_derived_text(item)
        for item in checks
        if isinstance(item, str) and _project_stage_derived_text(item)
    ] if isinstance(checks, list) else []
    return {
        "status": "ready" if ready_value else "not_ready",
        "ready": ready_value,
        "calculation_intent": _project_stage_derived_text(
            native.get("calculation_intent") if isinstance(native, Mapping) else None,
            "",
        ) or None,
        "link1_segment_count": count,
        "checks": safe_checks,
    }


def _operation_name_for_step(step: Mapping[str, Any]) -> str:
    """把正式 Action 或执行层转换为受控科学操作名称。"""

    action = step.get("action_id")
    if isinstance(action, str):
        # 未注册的 Action 不能原样作为操作名，否则会把机器 Action 身份送入
        # 阶段记忆；仍保留一个稳定的科学操作占位，供修订模型判断步骤位置。
        return _OPERATION_NAMES.get(action, "predefined_scientific_action")
    layer = step.get("execution_layer")
    if layer == "native_software":
        return "run_native_quantum_chemistry"
    if layer == "programmable_analysis":
        return "run_programmable_analysis"
    return "scientific_computation"


_PUBLIC_REQUEST_KEYS = frozenset(
    {
        "inputs",
        "method_spec",
        "action_settings",
        "resource_limits",
        "source_id",
        "component_backends",
        "bindings",
        "foreach",
        "software_id",
        "executable",
        "runtime",
        "arguments",
        "outputs",
        "calculation_intent",
        "label",
    }
)
_PUBLIC_REQUEST_NESTED_KEYS = frozenset(
    {
        "structure",
        "structure_summary",
        "summary_status",
        "input_format",
        "output_format",
        "format",
        "data_type",
        "spectrum_type",
        "plot_type",
        "mode",
        "window",
        "grid",
        "user_input",
        "atom_count",
        "element_counts",
        "C",
        "H",
        "N",
        "O",
        "F",
        "P",
        "S",
        "Cl",
        "Br",
        "I",
        "structures",
        "smiles",
        "sdf",
        "xyz",
        "molecule",
        "molecules",
        "route_section",
        "route_parameters",
        "charge",
        "multiplicity",
        "title",
        "chk_file",
        "mem",
        "nprocshared",
        "method",
        "basis",
        "dispersion",
        "solvent",
        "solvent_model",
        "temperature",
        "temperature_kelvin",
        "pressure",
        "frequency",
        "optimization",
        "geometry",
        "coordinates",
        "atoms",
        "element",
        "position_angstrom",
        "source_step_ref",
        "source_field",
        "target_field",
        "max_items",
        "only_present_items",
        "item_field",
        "item_id_field",
        "cpu",
        "memory_mb",
        "walltime_seconds",
        "timeout_seconds",
        "gpu",
        "required_modules",
        "required_module_versions",
        "required_symbols",
        "name",
        "semantic_type",
        "media_type",
        "required",
        "value",
        "unit",
        "type",
        "description",
    }
)

# 严格 Pair 合同只声明 request section 的顶层字段；字段值如果是对象，仍须
# 经过这份注册表，不能因为顶层字段合法就递归接收任意嵌套键。集合覆盖当前
# MiniChem 科学输入的稳定结构，并保留 Case 1 反应映射/候选能量的公开成员。
_REVISION_STRICT_NESTED_KEYS = frozenset(
    set(_PUBLIC_REQUEST_NESTED_KEYS)
    | set(_SCIENTIFIC_NESTED_KEYS)
    | {
        "reactants",
        "products",
        "hydrogen_transfers",
        "from",
        "to",
        "conformer_id",
        "candidate_id",
        "molecule_id",
        "energy",
        "energies",
        "reaction",
        "mapped_reaction",
    }
)


# Revision 的 request section 必须来自冻结的精确 Action/Backend 合同。旧的固定
# ``_PUBLIC_REQUEST_NESTED_KEYS`` 只能作为没有 action_contracts 的历史兼容分支；一旦
# 目录带有 companion 合同，未知字段宁可丢弃，也不能凭字段名猜测 Action 能力。
_REVISION_REQUEST_SECTIONS = ("inputs", "method_spec", "action_settings", "resource_limits")
_REVISION_RESOURCE_FIELDS = frozenset({"cpu_cores", "gpu_count", "memory_mb"})
_REVISION_SENSITIVE_KEY_NAMES = frozenset(
    {
        "artifact",
        "artifacts",
        "artifact_id",
        "artifact_ids",
        "operation",
        "operation_id",
        "job",
        "job_id",
        "execution",
        "execution_id",
        "checkpoint",
        "checkpoint_id",
        "record",
        "record_id",
        "step",
        "step_id",
        "event",
        "event_id",
        "parent_artifact_ids",
        "raw_output",
        "raw_evidence",
        "trace",
        "traceback",
        "metadata",
        "server_info",
    }
)
_REVISION_PATH_FIELD_RE = re.compile(r"(?:^|_)(?:path|paths|file|files)$", re.IGNORECASE)
_REVISION_WORKSPACE_PATH_RE = re.compile(
    r"^(?:code|inputs|outputs|artifacts|working_memory|reports?|logs?)/",
    re.IGNORECASE,
)
_REVISION_SENSITIVE_CONTENT_RE = re.compile(
    r"(?:sk-[A-Za-z0-9_-]{8,}|(?:api[_-]?key|token|password|secret)\s*[:=]\s*[^\s]+|"
    r"(?:artifact|operation|job|execution|checkpoint|record|step)[_-]?id\s*[:=/_-]\s*[^\s]+|"
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",
    re.IGNORECASE,
)
_REVISION_SENSITIVE_CONTENT_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9])/(?:mnt|tmp|home|root|var|opt|usr|work)(?:/|$)"
    r"|(?<![A-Za-z0-9])[A-Za-z]:[\\/]",
    re.IGNORECASE,
)
_REVISION_UNSAFE_VALUE_RE = re.compile(
    r"\b(?:raw[_ -]?(?:output|log)|stdout|stderr|traceback|transport|secret|token|"
    r"password|credential|(?:artifact|operation|job|execution|checkpoint|record|step|"
    r"event|combination|hypothesis)[_-]?id)\b",
    re.IGNORECASE,
)

# 这两组字段与 workflow_contract 的 NativeJobRequest/AnalysisJobRequest 同步；策略
# 与执行资源政策字段不会进入模型，Revision 只接收可读、可修改的公开 request 外壳。
_REVISION_NATIVE_OPEN_FIELDS = (
    "software_id",
    "executable",
    "arguments",
    "staged_inputs",
    "stdin_target",
    "resource_limits",
    "calculation_intent",
    "label",
    "bindings",
    "authored_files",
    "parent_job_id",
)
_REVISION_ANALYSIS_OPEN_FIELDS = (
    "runtime",
    "script_path",
    "script_target",
    "arguments",
    "staged_inputs",
    "required_modules",
    "required_module_versions",
    "required_symbols",
    "inputs",
    "outputs",
    "execution_policy",
    "external_execution",
    "resource_limits",
    "label",
    "bindings",
    "authored_files",
    "parent_job_id",
)


def _revision_contract_entries(catalog_snapshot: Any) -> tuple[dict[tuple[str, str], Mapping[str, Any]], bool]:
    """读取 Revision 可见的 pair 合同并标记是否必须严格按合同收窄。

    输入可以是完整冻结目录，也可以是已经筛选给模型的 Action 数组。返回值第一项
    按 ``(action_id, backend_id)`` 索引 request_contract；第二项表示目录明确带有
    ``action_contracts``。严格目录中缺少某个 pair 时返回空合同，避免回退到静态
    字段表；没有 companion 合同的旧快照才允许走兼容字段表。
    """

    entries: dict[tuple[str, str], Mapping[str, Any]] = {}
    strict = False
    if isinstance(catalog_snapshot, Mapping):
        companion = catalog_snapshot.get("action_contracts")
        if isinstance(companion, Mapping):
            strict = True
            raw_contracts = companion.get("contracts")
            if isinstance(raw_contracts, list):
                for item in raw_contracts:
                    if not isinstance(item, Mapping):
                        continue
                    action_id = item.get("action_id")
                    backend_id = item.get("backend_id")
                    contract = item.get("request_contract")
                    if (
                        isinstance(action_id, str)
                        and isinstance(backend_id, str)
                        and isinstance(contract, Mapping)
                    ):
                        entries[(action_id, backend_id)] = contract
        actions = catalog_snapshot.get("actions")
    elif isinstance(catalog_snapshot, list):
        actions = catalog_snapshot
    else:
        actions = None
    if isinstance(actions, list):
        for action in actions:
            if not isinstance(action, Mapping):
                continue
            action_id = action.get("action_id", action.get("id"))
            pair_contracts = action.get("backend_request_contracts")
            if isinstance(pair_contracts, list):
                strict = True
                for item in pair_contracts:
                    if not isinstance(item, Mapping):
                        continue
                    backend_id = item.get("backend_id")
                    contract = item.get("request_contract")
                    if (
                        isinstance(action_id, str)
                        and isinstance(backend_id, str)
                        and isinstance(contract, Mapping)
                    ):
                        entries[(action_id, backend_id)] = contract
            request_contracts = action.get("request_contracts")
            if isinstance(request_contracts, Mapping) and isinstance(action_id, str):
                strict = True
                for backend_id, contract in request_contracts.items():
                    if isinstance(backend_id, str) and isinstance(contract, Mapping):
                        entries[(action_id, backend_id)] = contract
    return entries, strict


def _revision_pair_request_contract(
    catalog_snapshot: Any,
    action_id: Any,
    backend_id: Any,
) -> Mapping[str, Any] | None:
    """读取当前步骤的精确 Action/Backend request 合同。

    输入是目录快照及 Action、Backend 标识；动作查找对应合同，不修改目录。返回值
    是匹配的合同，严格目录缺少该 pair 时返回空映射，旧目录没有 companion 合同时
    返回 ``None``。标识类型不正确时同样返回 ``None``，不会猜测字段能力。
    """

    if not isinstance(action_id, str) or not isinstance(backend_id, str):
        return None
    entries, strict = _revision_contract_entries(catalog_snapshot)
    contract = entries.get((action_id, backend_id))
    if contract is not None:
        return contract
    return {} if strict else None


def _revision_section_allowed_fields(
    contract: Mapping[str, Any] | None,
    section: str,
) -> set[str] | None:
    """读取合同 ``sections.<section>.allowed`` 的顶层字段白名单。

    输入是单个 pair 合同和 section 名称；动作读取并筛选非空字符串字段，不修改
    合同。返回值是字段集合；合同缺失或没有明确 ``allowed`` 列表时返回 ``None``
    或空集合，调用方据此决定兼容路径，读取过程不会抛出原始数据异常。
    """

    if contract is None:
        return None
    sections = contract.get("sections")
    if not isinstance(sections, Mapping):
        # 早期快照只有 required_contract；保持旧视图兼容，不能把 required
        # 误称为新的精确 allowed 列表。
        return None
    raw_section = sections.get(section)
    if not isinstance(raw_section, Mapping):
        return set()
    allowed = raw_section.get("allowed")
    if not isinstance(allowed, list):
        return set()
    return {item for item in allowed if isinstance(item, str) and item}


def _revision_key_is_sensitive(key: str) -> bool:
    """判断动态 request 的键是否属于内部运行事实。

    输入是字段名字符串；动作按大小写不敏感的显式名单比较，不修改字段。返回
    ``True`` 表示字段必须从公开 request 中清除，否则返回 ``False``；非字符串由
    调用方的类型约束处理，不在此函数中抛出业务异常。
    """

    normalized = key.casefold().replace("-", "_")
    return normalized in _REVISION_SENSITIVE_KEY_NAMES


def _revision_key_is_path(key: str) -> bool:
    """判断字段是否保存路径或文件内容。

    输入是字段名字符串；动作按规范化名称和路径字段正则匹配，不读取文件。返回
    ``True`` 表示公开 wire 应将该字段置空，否则返回 ``False``；匹配只依赖字段名，
    不会因路径内容异常抛出错误。
    """

    normalized = key.casefold().replace("-", "_")
    return normalized in {"path", "paths", "file", "files"} or bool(
        _REVISION_PATH_FIELD_RE.search(normalized)
    )


def _project_revision_dynamic_value(value: Any, *, key: str | None = None) -> Any:
    """复制合同允许的科学标量/数组，并清空路径、文件和内部身份。

    顶层字段由 pair 合同控制；嵌套对象保留科学结构（例如 reactants、products 和
    hydrogen_transfers），同时递归移除产物身份、运行对象和路径内容。未知嵌套键
    不会自动升级为新的顶层 request 字段。
    """

    if key is not None:
        if _revision_key_is_sensitive(key):
            return None
        if _revision_key_is_path(key):
            return None
    if isinstance(value, Mapping):
        projected: dict[str, Any] = {}
        for raw_key, item in value.items():
            if not isinstance(raw_key, str) or _revision_key_is_sensitive(raw_key):
                continue
            if raw_key not in _REVISION_STRICT_NESTED_KEYS:
                continue
            child = _project_revision_dynamic_value(item, key=raw_key)
            # 路径字段保留稳定键但使用空值；其他字段的非法值直接关闭。
            if child is not None or _revision_key_is_path(raw_key):
                projected[raw_key] = child
        return projected
    if isinstance(value, (list, tuple)):
        return [
            _project_revision_dynamic_value(item)
            for item in value
        ]
    if isinstance(value, str):
        if (
            _contains_stage_memory_path(value)
            or _REVISION_WORKSPACE_PATH_RE.match(value)
            or _REVISION_SENSITIVE_CONTENT_PATH_RE.search(value)
            or _REVISION_SENSITIVE_CONTENT_RE.search(value)
            or _STAGE_MEMORY_INTERNAL_TEXT_PATTERN.search(value)
            or _REVISION_UNSAFE_VALUE_RE.search(value)
        ):
            return None
        return value
    if value is None or isinstance(value, (bool, int)):
        return deepcopy(value)
    if isinstance(value, float):
        return deepcopy(value) if math.isfinite(value) else None
    return None


def _project_revision_binding_selector(value: Any, *, target: bool = False) -> str:
    """保留安全短 selector，包括 ``job.stdout``，屏蔽路径和正式身份。"""

    if not isinstance(value, str):
        return ""
    text = value.strip()
    if target:
        if re.fullmatch(r"/?[A-Za-z][A-Za-z0-9_.]*(?:\.[0-9]+)?(?:/[A-Za-z0-9_.-]+)*", text):
            return text
        return ""
    if text in {"job.stdout", "job.stderr"}:
        return text
    if re.fullmatch(r"/result/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*", text):
        return text
    if re.fullmatch(
        r"(?:artifact\.media_type|artifact\.semantic_type|declared_artifact\.name)=[^\s=]+",
        text,
    ):
        return text
    return ""


def _project_open_relative_path(value: Any) -> str | None:
    """保留安全相对 task path；绝对路径、``..`` 和反斜杠统一置空。"""

    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if (
        "\\" in text
        or text.startswith("/")
        or re.match(r"^[A-Za-z]:[\\/]", text)
        or _REVISION_WORKSPACE_PATH_RE.match(text) is None
        and "/" in text
        and text.split("/", 1)[0].casefold()
        in {"mnt", "tmp", "home", "root", "var", "opt", "usr", "work"}
    ):
        return None
    path = PurePosixPath(text)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        return None
    return path.as_posix()


def _project_open_text(value: Any) -> str | None:
    """保留公开 open-execution 文本，清除绝对路径和内部运行标记。"""

    if not isinstance(value, str) or not value.strip():
        return None
    if (
        value.startswith("/")
        or re.match(r"^[A-Za-z]:[\\/]", value)
        or _contains_stage_memory_path(value)
    ):
        return None
    cleaned = _project_stage_derived_text(value)
    return cleaned if cleaned else None


def _project_open_path_list(value: Any) -> list[str]:
    """筛选 open-execution 的安全相对路径数组。

    输入是待公开的路径值；动作逐项调用相对路径校验并丢弃非法项，不修改原列表。
    返回值是安全相对路径列表；输入不是列表或没有合法项时返回空列表，校验失败
    不会抛出路径异常。
    """

    if not isinstance(value, list):
        return []
    return [
        projected
        for item in value
        if (projected := _project_open_relative_path(item)) is not None
    ]


def _project_revision_bindings(value: Any) -> list[dict[str, Any]]:
    """筛选 open/predefined 共用的绑定数组。

    输入是绑定对象数组；动作保留步骤短引用并校验来源、目标 selector，不修改输入。
    返回值是固定三字段的绑定列表；非映射项被丢弃，非法 selector 以空值表示，不
    会把内部路径或运行身份带入公开 request。
    """

    items = value if isinstance(value, list) else []
    return [
        {
            "source_step_ref": (
                item.get("source_step_ref")
                if isinstance(item, Mapping)
                and isinstance(item.get("source_step_ref"), str)
                else None
            ),
            "source_field": _project_revision_binding_selector(
                item.get("source_field") if isinstance(item, Mapping) else None
            ),
            "target_field": _project_revision_binding_selector(
                item.get("target_field") if isinstance(item, Mapping) else None,
                target=True,
            ),
        }
        for item in items
        if isinstance(item, Mapping)
    ]


def _project_open_staged_inputs(value: Any) -> list[dict[str, Any]]:
    """筛选 staged_inputs 的安全外壳。

    输入是 staged input 对象数组；动作隐藏 ``source_path``，仅校验并保留安全的
    ``target_path``，不修改输入。返回值是固定字段列表；输入项不是映射或目标路径
    非法时跳过该项，不会抛出路径解析异常。
    """

    items = value if isinstance(value, list) else []
    result: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        result.append(
            {
                "source_path": None,
                "target_path": _project_open_relative_path(item.get("target_path")),
            }
        )
    return result


def _project_open_authored_files(value: Any) -> list[dict[str, Any]]:
    """筛选 authored_files 的受限公开外壳。

    输入是 authored file 对象数组；动作限制数量和正文长度，校验相对路径并清除
    运行身份、绝对路径和敏感内容，不修改输入。返回值是可公开的文件外壳列表；
    非法项、越界正文或敏感内容被丢弃，不会因单项错误抛出异常。
    """

    items = value if isinstance(value, list) else []
    result: list[dict[str, Any]] = []
    total_chars = 0
    for item in items[:4]:
        if not isinstance(item, Mapping):
            continue
        path = _project_open_relative_path(item.get("path"))
        content = item.get("content")
        if path is None or not (
            path.startswith("code/") or path.startswith("outputs/")
        ):
            continue
        safe_content: str | None = None
        if isinstance(content, str):
            content_size = len(content)
            within_bounds = (
                content_size <= 100_000
                and total_chars + content_size <= 400_000
            )
            if within_bounds:
                total_chars += content_size
                contains_sensitive_text = bool(
                    _REVISION_SENSITIVE_CONTENT_PATH_RE.search(content)
                    or _REVISION_SENSITIVE_CONTENT_RE.search(content)
                    or _STAGE_MEMORY_INTERNAL_TEXT_PATTERN.search(content)
                )
                if not contains_sensitive_text:
                    safe_content = content
        overwrite = item.get("overwrite")
        result.append(
            {
                "path": path,
                "content": safe_content,
                "overwrite": overwrite if isinstance(overwrite, bool) else None,
            }
        )
    return result


_PUBLIC_JSON_SCHEMA_KEYS = frozenset(
    {
        "$schema",
        "$comment",
        "type",
        "title",
        "description",
        "default",
        "examples",
        "const",
        "enum",
        "format",
        "pattern",
        "properties",
        "required",
        "additionalProperties",
        "unevaluatedProperties",
        "items",
        "prefixItems",
        "contains",
        "minContains",
        "maxContains",
        "minProperties",
        "maxProperties",
        "minItems",
        "maxItems",
        "uniqueItems",
        "minLength",
        "maxLength",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "anyOf",
        "allOf",
        "oneOf",
        "not",
        "if",
        "then",
        "else",
        "dependentSchemas",
        "$defs",
        "definitions",
        "propertyNames",
        "contentEncoding",
        "contentMediaType",
        "readOnly",
        "writeOnly",
        "deprecated",
    }
)


def _project_public_json_schema(value: Any, *, property_map: bool = False) -> Any:
    """筛选公开 JSON Schema 的允许成员。

    输入是待发送的 schema 值及是否处于属性名映射的标记；动作递归保留 JSON Schema
    公开关键字，清理路径、内部扩展和不安全字符串，不修改输入。返回值是安全的
    schema 值；不支持的类型或非有限数字按缺失处理，不会把未知成员原样带入消息。
    """

    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        if (
            _REVISION_SENSITIVE_CONTENT_PATH_RE.search(value)
            or _REVISION_SENSITIVE_CONTENT_RE.search(value)
            or _STAGE_MEMORY_INTERNAL_TEXT_PATTERN.search(value)
            or (
                property_map
                and (
                    value.startswith("x-")
                    or _revision_key_is_sensitive(value)
                    or _revision_key_is_path(value)
                )
            )
        ):
            return None
        return value
    if isinstance(value, (list, tuple)):
        projected = []
        for item in value:
            safe = _project_public_json_schema(item, property_map=property_map)
            if safe is not None or item is None:
                projected.append(safe)
        return projected
    if not isinstance(value, Mapping):
        return None

    projected: dict[str, Any] = {}
    for raw_key, item in value.items():
        if not isinstance(raw_key, str):
            continue
        if property_map:
            # Schema property名本身是公开科学字段；内部 ID、x-扩展和路径
            # 名称不属于模型可见合同。
            if (
                raw_key.startswith("x-")
                or _revision_key_is_sensitive(raw_key)
                or _revision_key_is_path(raw_key)
                or _REVISION_SENSITIVE_CONTENT_RE.search(raw_key)
            ):
                continue
            safe = _project_public_json_schema(item)
            if safe is not None or item is None:
                projected[raw_key] = safe
            continue
        if raw_key not in _PUBLIC_JSON_SCHEMA_KEYS:
            continue
        child_property_map = raw_key in {
            "properties",
            "$defs",
            "definitions",
            "dependentSchemas",
            "required",
        }
        safe = _project_public_json_schema(item, property_map=child_property_map)
        if safe is not None or item is None:
            projected[raw_key] = safe
    return projected


def _project_open_outputs(value: Any) -> list[dict[str, Any]]:
    """筛选 Analysis outputs 的公开语义字段。

    输入是输出声明对象数组；动作保留名称、语义类型、媒体类型和安全 schema，清空
    路径及 parent artifact 身份，不修改输入。返回值是固定字段列表；缺少名称或语义
    类型的项被丢弃，非法字段不会抛出异常。
    """

    items = value if isinstance(value, list) else []
    result: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        name = _project_open_text(item.get("name"))
        semantic_type = _project_open_text(item.get("semantic_type"))
        if name is None or semantic_type is None:
            continue
        result.append(
            {
                "name": name,
                "path": None,
                "semantic_type": semantic_type,
                "media_type": _project_open_text(item.get("media_type")),
                "required": item.get("required")
                if isinstance(item.get("required"), bool)
                else None,
                "json_schema": _project_public_json_schema(item.get("json_schema"))
                if isinstance(item.get("json_schema"), Mapping)
                else {},
                "parent_artifact_ids": [],
            }
        )
    return result


def _project_open_execution_request(
    request: Any,
    execution_layer: str,
) -> Dict[str, Any]:
    """按 execution_layer 构造稳定 Native/Analysis request 公开外壳。

    evaluation_resource_budget 与 execution_timeout_policy 属于冻结执行政策，不在
    Revision 中作为可修改字段；parent_job_id 只保留稳定 null 占位。输入是原始 request
    和执行层，动作按层筛选公开字段并清空运行身份；返回值是模型可见 request。执行
    层未知时由调用方处理，字段类型异常按空值处理，不会泄露路径或资源事实。
    """

    source = request if isinstance(request, Mapping) else {}
    native = execution_layer == "native_software"
    fields = _REVISION_NATIVE_OPEN_FIELDS if native else _REVISION_ANALYSIS_OPEN_FIELDS
    result: Dict[str, Any] = {}
    for field in fields:
        if field in {"bindings", "arguments", "required_modules", "staged_inputs", "inputs", "outputs", "authored_files"}:
            result[field] = []
        elif field == "resource_limits":
            result[field] = {
                resource_field: None
                for resource_field in sorted(_REVISION_RESOURCE_FIELDS)
            }
        elif field in {"required_module_versions", "required_symbols"}:
            result[field] = {}
        else:
            result[field] = None

    if native:
        for field in ("software_id", "executable", "calculation_intent", "label"):
            result[field] = _project_open_text(source.get(field))
        result["arguments"] = [
            projected
            for item in source.get("arguments", [])
            if isinstance(item, str)
            if (projected := _project_open_text(item)) is not None
        ] if isinstance(source.get("arguments"), list) else []
        result["staged_inputs"] = _project_open_staged_inputs(source.get("staged_inputs"))
        result["stdin_target"] = _project_open_relative_path(source.get("stdin_target"))
    else:
        for field in ("runtime", "label"):
            result[field] = _project_open_text(source.get(field))
        result["script_path"] = _project_open_relative_path(source.get("script_path"))
        result["script_target"] = _project_open_relative_path(source.get("script_target"))
        result["arguments"] = [
            projected
            for item in source.get("arguments", [])
            if isinstance(item, str)
            if (projected := _project_open_text(item)) is not None
        ] if isinstance(source.get("arguments"), list) else []
        result["staged_inputs"] = _project_open_staged_inputs(source.get("staged_inputs"))
        result["required_modules"] = [
            projected
            for item in source.get("required_modules", [])
            if isinstance(item, str)
            if (projected := _project_open_text(item)) is not None
        ] if isinstance(source.get("required_modules"), list) else []
        versions = source.get("required_module_versions")
        if isinstance(versions, Mapping):
            result["required_module_versions"] = {
                key: _project_open_text(value)
                for key, value in versions.items()
                if isinstance(key, str) and _project_open_text(value) is not None
            }
        symbols = source.get("required_symbols")
        if isinstance(symbols, Mapping):
            result["required_symbols"] = {
                key: [
                    _project_open_text(item)
                    for item in values
                    if isinstance(item, str) and _project_open_text(item) is not None
                ]
                for key, values in symbols.items()
                if isinstance(key, str) and isinstance(values, list)
            }
        input_values = source.get("inputs")
        if isinstance(input_values, list):
            result["inputs"] = [
                {
                    "name": _project_open_text(item.get("name")),
                    "source_path": None,
                    "semantic_type": _project_open_text(item.get("semantic_type")),
                    "target_path": _project_open_relative_path(item.get("target_path")),
                }
                for item in input_values
                if isinstance(item, Mapping)
            ]
        result["outputs"] = _project_open_outputs(source.get("outputs"))
        result["execution_policy"] = (
            source.get("execution_policy")
            if source.get("execution_policy") == "in_process_only"
            else None
        )
        result["external_execution"] = (
            source.get("external_execution")
            if source.get("external_execution") == "native_job_runner_only"
            else None
        )

    limits = source.get("resource_limits")
    if isinstance(limits, Mapping):
        result["resource_limits"] = {
            field: limits.get(field)
            if isinstance(limits.get(field), (int, float))
            and not isinstance(limits.get(field), bool)
            and math.isfinite(float(limits.get(field)))
            else None
            for field in sorted(_REVISION_RESOURCE_FIELDS)
        }
    result["bindings"] = _project_revision_bindings(source.get("bindings"))
    result["authored_files"] = _project_open_authored_files(source.get("authored_files"))
    # parent_job_id 是运行身份，固定置空而不回显输入值。
    result["parent_job_id"] = None
    return result


def _project_public_request(
    request: Any,
    *,
    action_id: Any = None,
    backend_id: Any = None,
    execution_layer: str | None = None,
    catalog_snapshot: Any = None,
    pair_contract: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """保留 Planner 可修改的公开请求字段，按精确 pair 合同收窄内容。

    ``catalog_snapshot`` 可以是冻结目录或已筛选的 Action 数组；调用方也可直接传
    ``pair_contract``。带 ``sections.allowed`` 的合同严格控制四个 request section，
    同时保留 cpu_cores、gpu_count、memory_mb。没有 companion 合同的旧记录继续使用
    兼容白名单，避免历史状态无法恢复。
    """

    source = request if isinstance(request, Mapping) else {}
    if execution_layer in {"native_software", "programmable_analysis"}:
        return _project_open_execution_request(source, execution_layer)
    if pair_contract is None:
        pair_contract = _revision_pair_request_contract(
            catalog_snapshot,
            action_id,
            backend_id,
        )
    strict_sections = isinstance(pair_contract, Mapping) and (
        isinstance(pair_contract.get("sections"), Mapping)
        or pair_contract == {}
    )
    result: Dict[str, Any] = {}
    for key in sorted(_PUBLIC_REQUEST_KEYS):
        if key not in source:
            # 请求外壳保持稳定；模型可据此区分 Action 与 open execution。
            result[key] = [] if key in {"bindings", "arguments", "outputs"} else (
                {} if key in {"inputs", "method_spec", "action_settings", "resource_limits", "component_backends"}
                else None
            )
            continue
        value = source[key]
        if key == "component_backends":
            if isinstance(value, Mapping):
                result[key] = {
                    str(name): _project_stage_token(item, "") or None
                    for name, item in value.items()
                    if isinstance(name, str)
                }
            else:
                result[key] = {}
        elif key in {"bindings", "foreach"}:
            if key == "bindings":
                items = value if isinstance(value, list) else []
                result[key] = [
                    {
                        "source_step_ref": item.get("source_step_ref")
                        if isinstance(item, Mapping)
                        and isinstance(item.get("source_step_ref"), str)
                        else None,
                        "source_field": _project_revision_binding_selector(
                            item.get("source_field")
                        )
                        if isinstance(item, Mapping)
                        else "",
                        "target_field": _project_revision_binding_selector(
                            item.get("target_field"), target=True
                        )
                        if isinstance(item, Mapping)
                        else "",
                    }
                    for item in items
                    if isinstance(item, Mapping)
                ]
            elif isinstance(value, Mapping):
                maximum = value.get("max_items")
                result[key] = {
                    "source_step_ref": (
                        value.get("source_step_ref")
                        if isinstance(value.get("source_step_ref"), str)
                        else None
                    ),
                    "source_field": _project_revision_binding_selector(
                        value.get("source_field")
                    ),
                    "item_field": _project_revision_binding_selector(
                        value.get("item_field"), target=True
                    ),
                    "item_id_field": _project_revision_binding_selector(
                        value.get("item_id_field"), target=True
                    ),
                    "target_field": _project_revision_binding_selector(
                        value.get("target_field"), target=True
                    ),
                    "max_items": (
                        maximum
                        if isinstance(maximum, int)
                        and not isinstance(maximum, bool)
                        and maximum > 0
                        else None
                    ),
                    "only_present_items": value.get("only_present_items", False)
                    if type(value.get("only_present_items", False)) is bool
                    else False,
                }
            else:
                result[key] = None
        elif key in {"inputs", "method_spec", "action_settings", "resource_limits"}:
            allowed = _revision_section_allowed_fields(pair_contract, key)
            if strict_sections:
                allowed = allowed or set()
                if key == "resource_limits":
                    # MCP catalog 的 resource_limits.allowed 常为空；这三个值是
                    # 任务资源事实，不是可借由字段猜测的 Action 能力。
                    allowed.update(_REVISION_RESOURCE_FIELDS)
                # 合同声明的字段全部固定输出；缺失输入使用 null，避免同一 pair
                # 因某次请求是否携带 optional 字段而改变模型看到的 schema。
                projected = {
                    field: _project_revision_dynamic_value(value.get(field), key=field)
                    for field in sorted(allowed)
                }
            else:
                projected = _project_registered_value(
                    value,
                    _PUBLIC_REQUEST_NESTED_KEYS,
                    nested_keys=_PUBLIC_REQUEST_NESTED_KEYS,
                )
            result[key] = projected if isinstance(projected, (dict, list)) else {}
        elif key == "arguments":
            result[key] = [
                _project_stage_derived_text(item)
                for item in value
                if isinstance(item, str)
            ] if isinstance(value, list) else []
        elif key == "outputs":
            projected_outputs: list[dict[str, Any]] = []
            for item in value if isinstance(value, list) else []:
                if not isinstance(item, Mapping):
                    continue
                output: dict[str, Any] = {}
                output_fields = ("name", "semantic_type", "media_type")
                if not strict_sections:
                    # 旧快照兼容原有 named output 形状；带冻结合同的 Revision
                    # wire 不允许把输出路径或 parent artifact 身份交给模型。
                    output_fields = ("name", "path", "semantic_type", "media_type")
                for field in output_fields:
                    if isinstance(item.get(field), str) and item[field].strip():
                        output[field] = _project_stage_token(item[field], "")
                if isinstance(item.get("required"), bool):
                    output["required"] = item["required"]
                if isinstance(item.get("json_schema"), Mapping):
                    output["json_schema"] = _project_public_json_schema(item["json_schema"])
                parent_ids = item.get("parent_artifact_ids")
                if not strict_sections and isinstance(parent_ids, list):
                    output["parent_artifact_ids"] = [
                        _project_stage_token(parent_id, "")
                        for parent_id in parent_ids
                        if isinstance(parent_id, str) and parent_id.strip()
                    ]
                if strict_sections:
                    # 输出项保持固定外壳，但路径和 parent artifact 身份只能以
                    # 稳定空值表达，不能让模型看到工作区位置或正式产物 ID。
                    output["path"] = None
                    output["parent_artifact_ids"] = []
                    output.setdefault("required", None)
                    output.setdefault("json_schema", {})
                required_output_fields = {"name", "semantic_type"}
                if not strict_sections:
                    required_output_fields.add("path")
                if required_output_fields <= set(output):
                    projected_outputs.append(output)
            result[key] = projected_outputs
        else:
            result[key] = _project_stage_token(value, "") or None
    # source_id 是公开依赖入口，但不能接受正式 step_id；只允许短引用。
    if isinstance(source.get("source_id"), str) and re.fullmatch(r"[sn][1-9][0-9]*", source["source_id"]):
        result["source_id"] = source["source_id"]
    else:
        result["source_id"] = None
    return result


def project_workflow_for_revision(
    workflow: Any,
    catalog_snapshot: Any = None,
) -> Dict[str, Any]:
    """构造 Workflow Revision 可见的完整公开工作流 wire。

    步骤固定暴露 Planner 可以修改的公开字段；正式 ``step_id``、路径、运行时身份、
    expected output 和派生 operation/target 副本不会进入模型。提供冻结目录时，四个
    request section 逐 pair 使用 ``action_contracts.sections.allowed``，未知字段关闭。
    """

    source = workflow if isinstance(workflow, Mapping) else {}
    raw_steps = source.get("steps") if isinstance(source.get("steps"), list) else []
    formal_to_short: Dict[str, str] = {}
    for index, step in enumerate(raw_steps):
        if isinstance(step, Mapping) and isinstance(step.get("step_id"), str):
            formal_to_short[step["step_id"]] = f"s{index + 1}"
    steps: list[Dict[str, Any]] = []
    for index, raw_step in enumerate(raw_steps):
        step = raw_step if isinstance(raw_step, Mapping) else {}
        depends = [
            formal_to_short[item]
            for item in step.get("depends_on", [])
            if item in formal_to_short
        ]
        raw_request = step.get("request") if isinstance(step.get("request"), Mapping) else {}
        raw_source_id = raw_request.get("source_id") if isinstance(raw_request, Mapping) else None
        execution_layer = step.get("execution_layer", "predefined_actions")
        if execution_layer not in {
            "predefined_actions",
            "native_software",
            "programmable_analysis",
        }:
            execution_layer = "predefined_actions"
        public_request = _project_public_request(
            raw_request,
            action_id=step.get("action_id"),
            backend_id=step.get("backend_id"),
            execution_layer=execution_layer,
            catalog_snapshot=catalog_snapshot,
        )
        bindings = public_request.get("bindings")
        if isinstance(bindings, list):
            for binding in bindings:
                if isinstance(binding, dict):
                    binding["source_step_ref"] = formal_to_short.get(
                        binding.get("source_step_ref")
                    )
        foreach = public_request.get("foreach")
        if isinstance(foreach, dict):
            source_ref = formal_to_short.get(foreach.get("source_step_ref"))
            foreach["source_step_ref"] = source_ref
        if execution_layer == "predefined_actions":
            if raw_source_id in formal_to_short:
                public_request["source_id"] = formal_to_short[raw_source_id]
            elif public_request.get("source_id") not in set(formal_to_short.values()):
                public_request["source_id"] = None
        steps.append(
            {
                "step_ref": f"s{index + 1}",
                "step_name": _project_stage_derived_text(step.get("step_name", "")),
                "description": _project_stage_derived_text(step.get("description", "")),
                "execution_layer": execution_layer,
                "action_id": _project_stage_token(step.get("action_id"), "") or None,
                "backend_id": _project_stage_token(step.get("backend_id"), "") or None,
                "request": public_request,
                "depends_on": depends,
            }
        )
    return {
        # 这些字段属于从持久 Revision 记录派生的文本；统一清理路径和运行
        # 身份，避免 package 层在再次筛选消息前已经保存内部事实。正文不截断。
        "workflow_name": _project_stage_derived_text(source.get("workflow_name", "")),
        "goal": _project_stage_derived_text(source.get("goal", "")),
        "strategy_name": _project_stage_derived_text(source.get("strategy_name", "")),
        "steps": steps,
    }


def _project_native_foreach_observations(
    raw_result: Mapping[str, Any],
) -> list[Dict[str, Any]]:
    """从 native foreach 的子任务结果提取已注册科学观测。

    输入是 Router 已整理的 raw_result mapping。动作读取 items、子任务结果以及允许的
    pair/candidate 来源字段，只保留注册观测名、数值、单位和有限条件；副作用是无，不
    修改输入、不写文件、不访问数据库或工具。返回科学观测字典列表；items 不是列表、
    子任务结构不完整或单个观测值无法规范化时跳过无效项并返回其余结果，调用方若传入
    非 mapping 则由属性访问错误暴露输入错误。
    """

    items = raw_result.get("items")
    if not isinstance(items, list):
        return []
    observations: list[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        child = item.get("result")
        if not isinstance(child, Mapping):
            continue
        nested = child.get("result")
        source = nested if isinstance(nested, Mapping) else child
        provenance = item.get("provenance")
        provenance = provenance if isinstance(provenance, Mapping) else item
        conditions: dict[str, Any] = {}
        for key in sorted(_NATIVE_FOREACH_PROVENANCE_KEYS):
            if key not in provenance:
                continue
            projected = _project_scientific_registered_value(provenance[key])
            if projected is not None:
                conditions[key] = projected
        unit = _project_scientific_registered_value(source.get("unit"))
        for name in _NATIVE_FOREACH_OBSERVATION_NAMES:
            if name not in source:
                continue
            projected = _project_scientific_registered_value(source[name])
            if projected is None:
                continue
            observations.append(
                {
                    "name": name,
                    "value": projected,
                    "unit": unit,
                    "conditions": deepcopy(conditions),
                }
            )
    return observations


def _project_scientific_observations(
    raw_result: Any,
    operation: str | None = None,
    *,
    declared_names: Sequence[str] | None = None,
) -> list[Dict[str, Any]]:
    """按操作注册的 projector 选择具名科学观测。"""

    if not isinstance(raw_result, Mapping):
        return []
    if (
        operation == "run_native_quantum_chemistry"
        and isinstance(raw_result.get("items"), list)
    ):
        return _project_native_foreach_observations(raw_result)
    if operation == "parse_quantum_chemistry":
        lookup: Dict[str, Any] = dict(raw_result)
        for block_name in ("properties", "metadata"):
            block = raw_result.get(block_name)
            if isinstance(block, Mapping):
                for key, value in block.items():
                    lookup.setdefault(key, value)
        units = raw_result.get("attribute_units")
        units_by_key = (
            {
                key: value
                for key, value in units.items()
                if isinstance(key, str) and isinstance(value, str) and value.strip()
            }
            if isinstance(units, Mapping)
            else {}
        )
        aliases = {
            "final_atomic_structure": (("final_atomic_structure", "final_structure", "atomcoords"), None),
            "electronic_energies": (("electronic_energies", "energies", "scfenergies", "mpenergies", "ccenergies"), None),
            "vibrational_frequencies": (("vibrational_frequencies", "frequencies", "vibfreqs"), None),
            "vibrational_intensities": (("vibrational_intensities", "intensities", "vibirs", "vibramans"), None),
            "charge": (("charge",), None),
            "multiplicity": (("multiplicity", "mult"), None),
        }
        observations: list[Dict[str, Any]] = []
        for name in _OPERATION_OBSERVABLES["parse_quantum_chemistry"]:
            keys, default_unit = aliases[name]
            key = next((candidate for candidate in keys if candidate in lookup), None)
            if key is None:
                continue
            projected = _project_scientific_registered_value(lookup[key])
            if projected is None:
                continue
            observations.append(
                {
                    "name": name,
                    "value": projected,
                    "unit": units_by_key.get(key, default_unit),
                    "conditions": {},
                }
            )
        return observations
    if operation == "prepare_gaussian_deck":
        native = raw_result.get("native_preflight")
        if not isinstance(native, Mapping):
            native = raw_result
        ready = native.get("native_preflight_ready")
        if not isinstance(ready, bool) and isinstance(
            raw_result.get("native_preflight_ready"), bool
        ):
            ready = raw_result["native_preflight_ready"]
        if not isinstance(ready, bool):
            return []
        return [
            {
                "name": "deck_preflight_status",
                "value": {
                    "status": "ready" if ready else "not_ready",
                    "ready": ready,
                },
                "unit": None,
                "conditions": {},
            }
        ]
    if operation == "structure_conversion":
        count = raw_result.get("molecules_written")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            return []
        return [{"name": "molecules_written", "value": count, "unit": None, "conditions": {}}]
    if operation == "geometry_optimization":
        structure = raw_result.get("optimized_geometry", raw_result.get("structure"))
        if structure is None:
            return []
        projected = _project_scientific_registered_value(structure)
        return [{"name": "optimized_geometry", "value": projected, "unit": None, "conditions": {}}]
    if operation == "hessian_calculation":
        summary = _project_hessian_summary(raw_result)
        if summary is None:
            return []
        return [{"name": "hessian_summary", "value": summary, "unit": None, "conditions": {}}]
    if operation == "vibrational_modes":
        summary = _project_vibrational_mode_summary(raw_result)
        if summary is None:
            return []
        return [{"name": "vibrational_mode_summary", "value": summary, "unit": None, "conditions": {}}]
    if operation in {"ir_spectrum", "raman_spectrum"}:
        summary = _project_line_spectrum_summary(raw_result)
        if summary is None:
            return []
        name = "ir_line_spectrum" if operation == "ir_spectrum" else "raman_line_spectrum"
        return [{"name": name, "value": summary, "unit": None, "conditions": {}}]
    if operation == "spectrum_plot":
        return []
    shared_unit = _project_scientific_registered_value(raw_result.get("unit"))
    shared_conditions = _project_scientific_registered_value(
        raw_result.get("conditions", {})
    )
    if not isinstance(shared_conditions, Mapping):
        shared_conditions = {}
    observations: list[Dict[str, Any]] = []
    # unit、conditions 和 status 是观测的元数据，不是三个额外 observable。
    # 对 ``{value, unit, conditions}`` 包装只在其键集合明确匹配时拆包，避免把
    # 任意科学 object 中偶然名为 value 的成员误解为包装层。
    metadata_keys = {"unit", "conditions", "status"}
    wrapper_keys = {"value", "unit", "conditions"}
    if declared_names is not None:
        allowed_names = set(declared_names)
    elif operation in _OPERATION_OBSERVABLES:
        allowed_names = set(_OPERATION_OBSERVABLES[operation])
    else:
        # 未注册 operation 没有科学观测入口，避免字段名碰巧命中全局白名单时
        # 被误当成该 operation 的事实。
        allowed_names = set()
    for canonical_name in tuple(allowed_names):
        aliases = globals().get("_OBSERVATION_ALIASES", {}).get(canonical_name, ())
        if isinstance(aliases, (set, tuple, list, frozenset)):
            allowed_names.update(aliases)
    for key in sorted(_SCIENTIFIC_OBSERVATION_KEYS - metadata_keys):
        if key not in allowed_names:
            continue
        if key not in raw_result:
            continue
        raw_value = raw_result[key]
        if (
            isinstance(raw_value, Mapping)
            and "value" in raw_value
            and set(raw_value).issubset(wrapper_keys)
        ):
            projected = _project_scientific_registered_value(raw_value.get("value"))
            unit = _project_scientific_registered_value(
                raw_value.get("unit", shared_unit)
            )
            conditions = _project_scientific_registered_value(
                raw_value.get("conditions", shared_conditions)
            )
        else:
            projected = _project_scientific_registered_value(raw_value)
            unit = shared_unit
            conditions = shared_conditions
        if projected is None:
            continue
        if not isinstance(conditions, Mapping):
            conditions = {}
        observations.append(
            {
                "name": key,
                "value": projected,
                "unit": unit,
                "conditions": deepcopy(dict(conditions)),
            }
        )
    return observations


def _project_program_checks(raw_output: Any) -> list[Dict[str, Any]]:
    """选择程序侧科学检查，不暴露 semantic type、artifact 或内部身份。"""

    if not isinstance(raw_output, Mapping):
        return []
    checks = raw_output.get("checks")
    if isinstance(checks, Mapping):
        checks = [checks]
    if not isinstance(checks, list):
        checks = []
    result = []
    for check in checks:
        if not isinstance(check, Mapping):
            continue
        names = [key for key in _SCIENTIFIC_CHECK_KEYS if key in check]
        for name in sorted(names):
            result.append(
                {
                    "name": name,
                    "status": "passed" if check[name] in (True, "matched", "pass") else "failed",
                    "requirement": None,
                    "observed": _project_scientific_registered_value(check[name]),
                }
            )
    return result


_OBSERVATION_ALIASES = {
    "gaussian_route": {"gaussian_route", "gaussian_code", "route"},
    "gaussian_input_deck": {
        "gaussian_input_deck",
        "gaussian_code",
        "input_deck",
    },
}


def _canonicalize_observations(
    observations: list[Dict[str, Any]],
    expected: list[str],
) -> list[Dict[str, Any]]:
    """按当前步骤的预期名称合并工具输出中的科学字段别名。"""

    expected_by_alias: Dict[str, str] = {}
    for name in expected:
        for alias in _OBSERVATION_ALIASES.get(name, {name}):
            expected_by_alias.setdefault(alias, name)
    canonical: list[Dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in observations:
        name = item.get("name")
        if isinstance(name, str):
            name = expected_by_alias.get(name, name)
        value_key = canonical_json(item.get("value"))
        marker = (str(name), value_key)
        if marker in seen:
            continue
        seen.add(marker)
        canonical.append({**item, "name": name})
    return canonical


def _assign_ledger_evidence_refs(ledger: Dict[str, Any]) -> Dict[str, Any]:
    """给账本内每个可引用事实分配稳定的 eN 短引用。"""

    counter = 1
    for step in ledger.get("steps", []) if isinstance(ledger.get("steps"), list) else []:
        if not isinstance(step, Mapping):
            continue
        for collection_name in ("observations", "program_checks"):
            collection = step.get(collection_name)
            if not isinstance(collection, list):
                continue
            for fact in collection:
                if isinstance(fact, dict):
                    fact["evidence_ref"] = f"e{counter}"
                    counter += 1
        failure = step.get("failure")
        if isinstance(failure, dict):
            failure["evidence_ref"] = f"e{counter}"
            counter += 1
    return ledger


def _ledger_fact_count(ledger: Mapping[str, Any]) -> int:
    """统计账本内已分配的科学事实数量，供 Program Gate 继续编号。"""

    count = 0
    for step in ledger.get("steps", []) if isinstance(ledger.get("steps"), list) else []:
        if not isinstance(step, Mapping):
            continue
        for key in ("observations", "program_checks"):
            values = step.get(key)
            if isinstance(values, list):
                count += sum(1 for item in values if isinstance(item, Mapping))
        if isinstance(step.get("failure"), Mapping):
            count += 1
    return count


def _declared_programmable_observable_names(step: Mapping[str, Any]) -> list[str]:
    """读取 programmable analysis.outputs 中已注册的科学输出名称。"""

    request = step.get("request") if isinstance(step.get("request"), Mapping) else {}
    outputs = request.get("outputs") if isinstance(request, Mapping) else None
    if not isinstance(outputs, list):
        return []
    names: list[str] = []
    for item in outputs:
        if not isinstance(item, Mapping):
            continue
        name = item.get("name")
        if (
            isinstance(name, str)
            and name.strip()
            and name in _SCIENTIFIC_OBSERVATION_KEYS
            and name not in {"gaussian_input_deck", "input_deck", "deck"}
            and name not in names
        ):
            names.append(name)
    return names


def project_scientific_attempt_ledger(
    execution_result: Any,
    program_validation: Any = None,
    router_decision: Any = None,
    workflow: Any = None,
) -> Dict[str, Any]:
    """从完整工作流和执行结果构造固定科学账本。

    正式工作流决定步骤顺序和预期观测；执行结果按内部 ``step_id`` 联接，输出只
    保留短引用和显式科学字段。即使步骤尚未执行，也会生成一个 ``not_executed``
    条目并列出缺失观测，避免尾部步骤在阶段视图中消失。
    """

    result = execution_result if isinstance(execution_result, Mapping) else {}
    raw_steps = result.get("steps") if isinstance(result.get("steps"), list) else []
    raw_by_id = {
        item.get("step_id"): item
        for item in raw_steps
        if isinstance(item, Mapping) and isinstance(item.get("step_id"), str)
    }
    formal_steps = (
        workflow.get("steps")
        if isinstance(workflow, Mapping) and isinstance(workflow.get("steps"), list)
        else []
    )
    stage_steps = formal_steps if formal_steps else raw_steps
    ledger_steps: list[Dict[str, Any]] = []
    for index, raw_formal_step in enumerate(stage_steps):
        formal_step = raw_formal_step if isinstance(raw_formal_step, Mapping) else {}
        formal_id = formal_step.get("step_id")
        raw_step = raw_by_id.get(formal_id) if isinstance(formal_id, str) else None
        if raw_step is None and index < len(raw_steps):
            # 某些旧 Attempt 没有 step_id；在这种情况下按规范工作流顺序联接，
            # 但仍只把科学结果字段带入下方的固定副本。
            raw_step = raw_steps[index]
        step = raw_step if isinstance(raw_step, Mapping) else {}
        raw_output = step.get("raw_output")
        if not isinstance(raw_output, Mapping):
            raw_output = step
        status = step.get("status", raw_output.get("status", "not_executed"))
        if status not in {"completed", "partial", "failed", "not_executed"}:
            status = "partial"
        operation = "predefined_scientific_action"
        if isinstance(formal_step.get("action_id"), str):
            operation = _OPERATION_NAMES.get(
                formal_step["action_id"], "predefined_scientific_action"
            )
        elif formal_step.get("execution_layer") == "native_software":
            operation = "run_native_quantum_chemistry"
        elif formal_step.get("execution_layer") == "programmable_analysis":
            operation = "run_programmable_analysis"
        raw_result = raw_output.get("result")
        if raw_result is None:
            raw_result = raw_output.get(
                "observations",
                step.get("result", step.get("observations")),
            )
        if (
            operation == "run_native_quantum_chemistry"
            and status == "completed"
            and isinstance(raw_output.get("foreach"), Mapping)
        ):
            # 先复用 Router 的 canonical foreach 校验和 bounded 投影，再从公开
            # child result 提取科学观测；这样篡改的 manifest/身份不会悄悄变成账本事实。
            raw_result, _omissions, _artifacts = _project_native_foreach_result(
                raw_output,
                f"s{index + 1}",
            )
        result_mapping = raw_result if isinstance(raw_result, Mapping) else {}
        declared_names = (
            _declared_programmable_observable_names(formal_step)
            if operation == "run_programmable_analysis"
            else None
        )
        expected: list[str] = (
            list(declared_names)
            if declared_names is not None
            else list(_OPERATION_OBSERVABLES.get(operation, []))
        )
        observations = _canonicalize_observations(
            _project_scientific_observations(
                raw_result,
                operation,
                declared_names=declared_names,
            ),
            expected,
        )
        scientific_settings = _project_scientific_settings(
            formal_step,
            operation,
            result_mapping,
        )
        native_preflight = _project_native_preflight(
            operation,
            raw_output,
            result_mapping,
        )
        observed_names = {item["name"] for item in observations}
        missing = [name for name in expected if name not in observed_names]
        failure = None
        if status == "failed":
            error = step.get("error") if isinstance(step.get("error"), Mapping) else {}
            failure = {
                "category": _project_stage_token(
                    error.get("category", "execution_failure"), "execution_failure"
                ),
                "code": _project_stage_token(
                    error.get("code", error.get("reason_code", "unknown")), "unknown"
                ),
                "details": _project_failure_details(
                    error,
                    step=step,
                    raw_output=raw_output,
                ),
            }
        ledger_steps.append(
            {
                "step_ref": f"s{index + 1}",
                "status": status,
                "observations": observations,
                "program_checks": _project_program_checks(raw_output),
                "scientific_settings": scientific_settings,
                "native_preflight": native_preflight,
                "missing_observables": missing,
                "failure": failure,
            }
        )
    ledger = {
        "overall_status": _project_stage_token(
            result.get("overall_status", "unknown"), "unknown"
        ),
        "workflow_outcome": _project_stage_token(
            result.get("workflow_outcome", "unknown"), "unknown"
        ),
        "steps": ledger_steps,
    }
    return _assign_ledger_evidence_refs(ledger)


def project_program_gate_fact(
    program_validation: Any,
    *,
    evidence_offset: int = 0,
    workflow: Any = None,
    step_ref_map: Mapping[str, str] | None = None,
) -> Dict[str, Any]:
    """筛选当前 Attempt 的固定程序检查事实并改写步骤身份。

    ``step_ref_map`` 用于 Revision 视图：既有步骤保持 ``sN``，本轮新增步骤
    使用 ``nN``，使程序 Gate 的约束理由与同一视图中的 Attempt ledger 一致。
    """

    value = program_validation if isinstance(program_validation, Mapping) else {}
    gate = value.get("program_gate_result") if isinstance(value.get("program_gate_result"), Mapping) else {}
    checks = value.get("registered_check_results") if isinstance(value.get("registered_check_results"), list) else []
    workflow_steps = (
        workflow.get("steps")
        if isinstance(workflow, Mapping) and isinstance(workflow.get("steps"), list)
        else []
    )
    default_refs = {
        step.get("step_id"): f"s{index + 1}"
        for index, step in enumerate(workflow_steps)
        if isinstance(step, Mapping) and isinstance(step.get("step_id"), str)
    }
    formal_to_short = {
        step_id: ref
        for step_id, ref in (step_ref_map or default_refs).items()
        if isinstance(step_id, str)
        and isinstance(ref, str)
        and re.fullmatch(r"[sn][1-9][0-9]*", ref)
    }

    def project_constraint_reason(reason: Any) -> str | None:
        if not isinstance(reason, str):
            return None
        prefix, separator, suffix = reason.partition(":")
        if not separator:
            return _project_stage_derived_text(reason)
        cleaned_suffix = _project_stage_derived_text(suffix)
        if prefix in formal_to_short:
            return f"{formal_to_short[prefix]}:{cleaned_suffix}"
        if re.fullmatch(r"s[1-9][0-9]*", prefix):
            return f"{prefix}:{cleaned_suffix}"
        # Program Gate 的带冒号理由是步骤级机器代码。无法联接时保留失败类别，
        # 但不把旧 workflow 的正式步骤身份回显给模型。
        return f"unresolved_step:{cleaned_suffix}"

    projected_checks = []
    for index, check in enumerate(checks):
        if not isinstance(check, Mapping):
            continue
        projected_checks.append(
            {
                "evidence_ref": f"e{evidence_offset + index + 1}",
                "status": _project_stage_token(
                    check.get("evaluation", "not_run"), "not_run"
                ),
                "requirement": _project_scientific_registered_value(
                    check.get("requirement")
                ),
                "observed": _project_scientific_registered_value(
                    check.get("actual", check.get("observed"))
                ),
            }
        )
    return {
        "allowed_decisions": [
            item
            for item in gate.get("allowed_decisions", [])
            if item in _ALLOWED_PROGRAM_DECISIONS
        ]
        if isinstance(gate.get("allowed_decisions", []), list)
        else [],
        "constraint_reasons": [
            projected
            for projected in (
                project_constraint_reason(item)
                for item in gate.get("constraint_reasons", [])
            )
            if projected is not None
        ]
        if isinstance(gate.get("constraint_reasons"), list)
        else [],
        "checks": projected_checks,
    }


def project_judge_fact(
    router_decision: Any,
    *,
    step_count: int = 0,
    ledger: Mapping[str, Any] | None = None,
    program_validation: Any = None,
    workflow: Any = None,
) -> Dict[str, Any]:
    """构造 Router/裁判的短引用理由。"""

    value = router_decision if isinstance(router_decision, Mapping) else {}
    diagnosis = value.get("diagnosis") if isinstance(value.get("diagnosis"), Mapping) else {}
    if ledger is not None:
        cited = _project_cited_evidence(
            diagnosis.get("evidence_refs", []),
            ledger,
            program_validation,
            workflow,
        )
        refs = [
            item["evidence_ref"]
            for item in cited
            if item.get("evidence_kind") != "unavailable"
            and isinstance(item.get("evidence_ref"), str)
        ]
    else:
        refs, _ = _stage_memory_short_evidence_refs(
            diagnosis.get("evidence_refs", []),
            step_count=step_count,
        )
    decision = value.get("decision")
    if decision not in _ALLOWED_PROGRAM_DECISIONS:
        decision = None
    return {
        "decision": decision,
        "reason": _project_stage_derived_text(
            diagnosis.get("summary", diagnosis.get("reason", ""))
        ),
        "evidence_refs": refs,
    }


def _project_cited_evidence(
    evidence_refs: Any,
    ledger: Mapping[str, Any],
    program_validation: Any,
    workflow: Any,
) -> list[Dict[str, Any]]:
    """把 Router 的正式指针闭合为同一视图内的科学事实。

    只允许指向已注册观测、程序检查或失败分类；一个指针若对应多个事实，或
    指向未注册的 raw/path 字段，则明确写出 unavailable，而不复制指针正文冒充
    观测。返回项字段固定，便于再生成模型稳定解析值、单位和条件。
    """

    refs = evidence_refs if isinstance(evidence_refs, list) else []
    steps = ledger.get("steps", []) if isinstance(ledger, Mapping) else []
    steps = steps if isinstance(steps, list) else []
    facts_by_step: Dict[int, list[Dict[str, Any]]] = {}
    for index, step in enumerate(steps):
        if not isinstance(step, Mapping):
            continue
        facts: list[Dict[str, Any]] = []
        observations = step.get("observations")
        if isinstance(observations, list):
            for observation in observations:
                if not isinstance(observation, Mapping):
                    continue
                name = observation.get("name")
                if not isinstance(name, str) or not name:
                    continue
                facts.append(
                    {
                        "evidence_ref": observation.get("evidence_ref"),
                        "step_ref": step.get("step_ref", f"s{index + 1}"),
                        "evidence_kind": "observation",
                        "name": name,
                        "value": _project_scientific_registered_value(
                            observation.get("value")
                        ),
                        "unit": _project_scientific_registered_value(
                            observation.get("unit")
                        ),
                        "conditions": _project_scientific_registered_value(
                            observation.get("conditions", {})
                        ),
                    }
                )
        checks = step.get("program_checks")
        if isinstance(checks, list):
            for check in checks:
                if not isinstance(check, Mapping):
                    continue
                name = check.get("name")
                if not isinstance(name, str) or not name:
                    continue
                facts.append(
                    {
                        "evidence_ref": check.get("evidence_ref"),
                        "step_ref": step.get("step_ref", f"s{index + 1}"),
                        "evidence_kind": "program_check",
                        "name": name,
                        "value": _project_scientific_registered_value(
                            check.get("observed")
                        ),
                        "unit": None,
                        "conditions": _project_scientific_registered_value(
                            {"status": check.get("status"), "requirement": check.get("requirement")}
                        ),
                    }
                )
        failure = step.get("failure")
        if isinstance(failure, Mapping):
            facts.append(
                {
                    "evidence_ref": failure.get("evidence_ref"),
                    "step_ref": step.get("step_ref", f"s{index + 1}"),
                    "evidence_kind": "failure",
                    "name": "failure",
                    "value": _project_scientific_registered_value(failure),
                    "unit": None,
                    "conditions": {},
                }
            )
        facts_by_step[index] = facts

    facts_by_ref: Dict[str, Dict[str, Any]] = {}
    ambiguous_refs: set[str] = set()
    for facts in facts_by_step.values():
        for fact in facts:
            evidence_ref = fact.get("evidence_ref")
            if not isinstance(evidence_ref, str):
                continue
            if evidence_ref in facts_by_ref:
                facts_by_ref.pop(evidence_ref, None)
                ambiguous_refs.add(evidence_ref)
                continue
            if evidence_ref not in ambiguous_refs:
                facts_by_ref[evidence_ref] = fact

    formal_steps = (
        workflow.get("steps")
        if isinstance(workflow, Mapping) and isinstance(workflow.get("steps"), list)
        else []
    )
    formal_to_short = {
        item.get("step_id"): f"s{index + 1}"
        for index, item in enumerate(formal_steps)
        if isinstance(item, Mapping) and isinstance(item.get("step_id"), str)
    }

    validation = program_validation if isinstance(program_validation, Mapping) else {}
    registered_checks = validation.get("registered_check_results")
    registered_checks = registered_checks if isinstance(registered_checks, list) else []

    used_refs: set[str] = set()
    next_generated = 1

    def allocate_ref(preferred: Any = None) -> str:
        nonlocal next_generated
        if isinstance(preferred, str) and re.fullmatch(r"e[1-9][0-9]*", preferred) and preferred not in used_refs:
            used_refs.add(preferred)
            return preferred
        while f"e{next_generated}" in used_refs:
            next_generated += 1
        value = f"e{next_generated}"
        used_refs.add(value)
        next_generated += 1
        return value

    def unavailable(ref: Any, reason: str, step_ref: str | None = None) -> Dict[str, Any]:
        return {
            # 原始 pointer 只用于当前函数的解析，不回显到模型；eN 是本视图内
            # 稳定的新引用，即使事实不可用也不会伪造一个空科学值。
            "evidence_ref": allocate_ref(),
            "step_ref": step_ref,
            "evidence_kind": "unavailable",
            "name": "unavailable_evidence",
            "value": {"status": "unavailable", "reason": reason},
            "unit": None,
            "conditions": {},
        }

    def resolve_step_fact(index: int, tail: list[str]) -> Dict[str, Any] | None:
        facts = facts_by_step.get(index, [])
        if not tail:
            return facts[0] if len(facts) == 1 else None
        requested = tail[-1]
        alias_groups = {
            "energy": {"energy", "electronic_energy", "electronic_energies"},
            "route": {"route", "gaussian_route", "gaussian_code"},
            "frequencies": {"frequencies", "vibrational_frequencies", "vibrational_mode_summary"},
        }
        aliases = {requested}
        for group in alias_groups.values():
            if requested in group:
                aliases = group
                break
        matches = [fact for fact in facts if fact.get("name") in aliases]
        return matches[0] if len(matches) == 1 else None

    projected: list[Dict[str, Any]] = []
    seen_input_refs: set[str] = set()
    for ref in refs:
        if not isinstance(ref, str) or not ref.strip():
            continue
        if ref in seen_input_refs:
            continue
        seen_input_refs.add(ref)
        fact: Dict[str, Any] | None = None
        step_ref: str | None = None
        tokens = [token for token in ref.split("/") if token]
        direct_evidence_match = re.fullmatch(r"e[1-9][0-9]*", ref)
        step_match = re.search(r"(?:^|/)steps/(\d+)(?:/|$)", ref)
        if direct_evidence_match is not None:
            fact = facts_by_ref.get(ref)
            if isinstance(fact, Mapping):
                step_ref = fact.get("step_ref")
        elif step_match is not None:
            index = int(step_match.group(1))
            tail_start = ref.find("/steps/")
            tail = tokens[tokens.index("steps") + 2 :] if "steps" in tokens else []
            if index < len(steps) and isinstance(steps[index], Mapping):
                step_ref = steps[index].get("step_ref")
            fact = resolve_step_fact(index, tail)
        elif re.search(r"/program_validation/registered_check_results/\d+", ref):
            match = re.search(r"registered_check_results/(\d+)", ref)
            index = int(match.group(1)) if match else -1
            if 0 <= index < len(registered_checks):
                check = registered_checks[index]
                if isinstance(check, Mapping):
                    check_name = check.get("name", check.get("check_name", "program_check"))
                    check_step_id = check.get("step_id")
                    step_ref = formal_to_short.get(check_step_id)
                    if step_ref is None and isinstance(check.get("step_ref"), str):
                        step_ref = check["step_ref"]
                    fact = {
                        "evidence_ref": f"e{_ledger_fact_count(ledger) + index + 1}",
                        "step_ref": step_ref,
                        "evidence_kind": "program_check",
                        "name": check_name if isinstance(check_name, str) else "program_check",
                        "value": _project_scientific_registered_value(
                            check.get("actual", check.get("observed"))
                        ),
                        "unit": None,
                        "conditions": _project_scientific_registered_value(
                            {"status": check.get("evaluation", "not_run"), "requirement": check.get("requirement")}
                        ),
                    }
        if fact is None:
            projected.append(unavailable(ref, "引用未解析为唯一的注册科学事实", step_ref))
            continue
        if fact.get("value") is None:
            projected.append(unavailable(ref, "注册科学事实没有可供裁判引用的观测值", step_ref))
            continue
        projected.append(
            {
                "evidence_ref": allocate_ref(fact.get("evidence_ref")),
                "step_ref": fact.get("step_ref"),
                "evidence_kind": fact.get("evidence_kind"),
                "name": fact.get("name"),
                "value": fact.get("value"),
                "unit": fact.get("unit"),
                "conditions": fact.get("conditions", {}),
            }
        )
    return projected


def _project_tool_capability_gap(value: Any) -> Dict[str, Any] | None:
    """只保留能力缺口的三项科学说明，不复制校验包装和内部身份。"""

    if not isinstance(value, Mapping):
        return None
    missing = value.get("missing_capabilities")
    if not isinstance(missing, list):
        missing = []
    projected_missing = [
        _project_stage_derived_text(item)
        for item in missing
        if isinstance(item, str) and item.strip()
    ]
    reason = _project_stage_derived_text(value.get("reason", ""))
    return {
        "result_type": "tool_capability_gap",
        "missing_capabilities": projected_missing,
        "reason": reason if isinstance(reason, str) else "",
    }


def project_hypothesis_terminal_outcome(
    *,
    source_query: Any,
    hypothesis: Mapping[str, Any],
    terminal_classification: str,
    terminal_reason: Any,
    execution_result: Any = None,
    program_validation: Any = None,
    router_decision: Any = None,
    tool_capability_gap: Any = None,
    workflow: Any = None,
) -> Dict[str, Any]:
    """构造再生成阶段可见的一条具名终局科学经验。"""

    ledger = project_scientific_attempt_ledger(
        execution_result,
        program_validation,
        router_decision,
        workflow,
    )
    judge = project_judge_fact(router_decision, step_count=len(ledger["steps"]))
    diagnosis = (
        router_decision.get("diagnosis")
        if isinstance(router_decision, Mapping)
        and isinstance(router_decision.get("diagnosis"), Mapping)
        else {}
    )
    cited_evidence = _project_cited_evidence(
        diagnosis.get("evidence_refs", []),
        ledger,
        program_validation,
        workflow,
    )
    attempted: list[str] = []
    missing: list[str] = []
    attempted_seen: set[str] = set()
    missing_seen: set[str] = set()
    for step in ledger["steps"]:
        observed_names = [
            item["name"]
            for item in step["observations"]
            if isinstance(item.get("name"), str)
        ]
        missing_names = [
            name
            for name in step["missing_observables"]
            if isinstance(name, str)
        ]
        if step["status"] != "not_executed":
            for name in observed_names + missing_names:
                if name not in attempted_seen:
                    attempted_seen.add(name)
                    attempted.append(name)
        for name in missing_names:
            if name not in missing_seen:
                missing_seen.add(name)
                missing.append(name)
    classification = terminal_classification if terminal_classification in {
        "reliably_refuted", "insufficient_evidence", "tool_capability_gap", "supported"
    } else "insufficient_evidence"
    return {
        "source_query": source_query if isinstance(source_query, str) else "",
        "hypothesis": {
            "strategy_name": hypothesis.get("strategy_name", ""),
            "reasoning": hypothesis.get("reasoning", ""),
        },
        "terminal_classification": classification,
        "final_judge": {
            "reason": judge["reason"] or _project_stage_derived_text(terminal_reason),
        },
        "cited_evidence": cited_evidence,
        "attempted_observables": attempted,
        "missing_evidence": missing,
        "tool_capability_gap": _project_tool_capability_gap(tool_capability_gap),
    }


def _project_patch_and_affected_steps(
    normalized_patch: Any,
    before_workflow: Any,
    after_workflow: Any,
    catalog_snapshot: Any = None,
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]], list[Dict[str, Any]]]:
    """解析 canonical diff，并只返回确定受影响步骤的公开前后视图。"""

    def workflow_steps(value: Any) -> list[Mapping[str, Any]]:
        raw = value.get("steps") if isinstance(value, Mapping) else []
        return [item for item in raw if isinstance(item, Mapping)] if isinstance(raw, list) else []

    before_steps = workflow_steps(before_workflow)
    after_steps = workflow_steps(after_workflow)

    def formal_id(step: Mapping[str, Any], index: int) -> str:
        value = step.get("step_id")
        return value if isinstance(value, str) and value else f"__index_{index}"

    before_ids = [formal_id(step, index) for index, step in enumerate(before_steps)]
    after_ids = [formal_id(step, index) for index, step in enumerate(after_steps)]
    before_by_id = dict(zip(before_ids, before_steps))
    after_by_id = dict(zip(after_ids, after_steps))
    before_ref = {step_id: f"s{index + 1}" for index, step_id in enumerate(before_ids)}
    common_ids = set(before_ids) & set(after_ids)
    inserted_ids = [step_id for step_id in after_ids if step_id not in common_ids]
    new_ref = {step_id: f"n{index + 1}" for index, step_id in enumerate(inserted_ids)}
    after_ref = {
        step_id: before_ref.get(step_id, new_ref.get(step_id, f"n{index + 1}"))
        for index, step_id in enumerate(after_ids)
    }

    def project_workflow_with_refs(
        source: Any,
        ids: list[str],
        refs: Mapping[str, str],
    ) -> list[Dict[str, Any]]:
        # 先以完整 workflow 投影，确保每个 binding/foreach 都能在完整正式步骤
        # 集合中解析；不能逐步投影后暂存 formal ID，否则未知值会泄漏，合法的跨步
        # 依赖也会在 fail-closed 时被误删。
        base_steps = project_workflow_for_revision(
            source,
            catalog_snapshot=catalog_snapshot,
        )["steps"]
        default_to_requested = {
            f"s{index + 1}": refs[step_id]
            for index, step_id in enumerate(ids)
            if step_id in refs
        }
        projected: list[Dict[str, Any]] = []
        for step_id, item in zip(ids, base_steps):
            item = deepcopy(item)
            item["step_ref"] = refs.get(step_id, item["step_ref"])
            item["depends_on"] = [
                default_to_requested[dep]
                for dep in item.get("depends_on", [])
                if dep in default_to_requested
            ]
            request = item.get("request")
            if isinstance(request, dict):
                bindings = request.get("bindings")
                if isinstance(bindings, list):
                    valid_bindings = []
                    for binding in bindings:
                        if not isinstance(binding, dict):
                            continue
                        source_ref = default_to_requested.get(
                            binding.get("source_step_ref")
                        )
                        if source_ref is None:
                            continue
                        binding["source_step_ref"] = source_ref
                        valid_bindings.append(binding)
                    request["bindings"] = valid_bindings
                foreach = request.get("foreach")
                if isinstance(foreach, dict):
                    source_ref = default_to_requested.get(foreach.get("source_step_ref"))
                    if source_ref is None:
                        request["foreach"] = None
                    else:
                        foreach["source_step_ref"] = source_ref
                if "source_id" in request:
                    source_id = request.get("source_id")
                    request["source_id"] = default_to_requested.get(source_id)
            projected.append(item)
        return projected

    before_view_by_id = {
        step_id: item
        for step_id, item in zip(
            before_ids,
            project_workflow_with_refs(before_workflow, before_ids, before_ref),
        )
    }
    after_view_by_id = {
        step_id: item
        for step_id, item in zip(
            after_ids,
            project_workflow_with_refs(after_workflow, after_ids, after_ref),
        )
    }

    if isinstance(normalized_patch, list):
        raw_ops = normalized_patch
    elif isinstance(normalized_patch, Mapping) and isinstance(normalized_patch.get("operations"), list):
        raw_ops = normalized_patch["operations"]
    else:
        raw_ops = []
    patch_view: list[Dict[str, Any]] = []
    affected_ids: set[str] = set()
    allowed_fields = {
        "step_name",
        "description",
        "execution_layer",
        "action_id",
        "backend_id",
        "request",
        "depends_on",
    }

    def public_step_field(
        step_id: str,
        field: str,
        projected_by_id: Mapping[str, Mapping[str, Any]],
    ) -> Any:
        step = projected_by_id.get(step_id)
        if not isinstance(step, Mapping):
            return None
        if field in step:
            return deepcopy(step[field])
        return None

    for raw in raw_ops:
        if not isinstance(raw, Mapping):
            patch_view.append(
                {
                    "operation": "unavailable",
                    "step_ref": None,
                    "field": None,
                    "before": None,
                    "after": None,
                }
            )
            continue
        operation = raw.get("op")
        target = raw.get("target_step_id")
        if operation == "update_step":
            if not isinstance(target, str) or target not in before_by_id or target not in after_by_id:
                patch_view.append({"operation": "unavailable", "step_ref": None, "field": None, "before": None, "after": None})
                continue
            changes = raw.get("changes")
            if not isinstance(changes, Mapping) or not changes:
                patch_view.append({"operation": "unavailable", "step_ref": before_ref.get(target), "field": None, "before": None, "after": None})
                continue
            affected_ids.add(target)
            for field in changes:
                if field not in allowed_fields:
                    patch_view.append({"operation": "unavailable", "step_ref": before_ref.get(target), "field": None, "before": None, "after": None})
                    continue
                patch_view.append(
                    {
                        "operation": "update_step",
                        "step_ref": before_ref.get(target),
                        "field": field,
                        "before": public_step_field(target, field, before_view_by_id),
                        "after": public_step_field(target, field, after_view_by_id),
                    }
                )
            continue
        if operation == "delete_step":
            if not isinstance(target, str) or target not in before_by_id:
                patch_view.append({"operation": "unavailable", "step_ref": None, "field": None, "before": None, "after": None})
                continue
            affected_ids.add(target)
            patch_view.append(
                {
                    "operation": "delete_step",
                    "step_ref": before_ref[target],
                    "field": "step",
                    "before": deepcopy(before_view_by_id.get(target)),
                    "after": None,
                }
            )
            continue
        if operation == "insert_step_after":
            step = raw.get("step")
            inserted_id = formal_id(step, len(after_ids)) if isinstance(step, Mapping) else None
            if not isinstance(inserted_id, str) or inserted_id not in after_by_id:
                patch_view.append({"operation": "unavailable", "step_ref": None, "field": None, "before": None, "after": None})
                continue
            affected_ids.add(inserted_id)
            patch_view.append(
                {
                    "operation": "insert_step_after",
                    "step_ref": after_ref.get(inserted_id),
                    "field": "step",
                    "before": None,
                    "after": deepcopy(after_view_by_id.get(inserted_id)),
                }
            )
            continue
        if operation == "move_step_after":
            after_id = raw.get("after_step_id")
            if not isinstance(target, str) or target not in before_by_id or target not in after_by_id:
                patch_view.append({"operation": "unavailable", "step_ref": None, "field": None, "before": None, "after": None})
                continue
            affected_ids.add(target)
            before_index = before_ids.index(target)
            after_index = after_ids.index(target)
            patch_view.append(
                {
                    "operation": "move_step_after",
                    "step_ref": before_ref[target],
                    "field": "order",
                    "before": before_index,
                    "after": after_index,
                }
            )
            if isinstance(after_id, str) and after_id in before_by_id:
                affected_ids.add(after_id)
            continue
        patch_view.append({"operation": "unavailable", "step_ref": None, "field": None, "before": None, "after": None})

    # 解析失败时保持空受影响集合，绝不能为了填充模型上下文把完整 workflow 当作
    # “受影响”；这会让修订模型误判所有步骤都需要重做。
    before_view = [before_view_by_id[item] for item in before_ids if item in affected_ids]
    after_view = [after_view_by_id[item] for item in after_ids if item in affected_ids]
    return patch_view, before_view, after_view


def project_workflow_revision_repair(
    *,
    normalized_patch: Any,
    before_workflow: Any,
    after_workflow: Any,
    execution_result: Any,
    program_validation: Any,
    router_decision: Any,
    catalog_snapshot: Any = None,
) -> Dict[str, Any]:
    """把同一 combination 的一轮修复压缩为固定六字段模型副本。"""

    patch, before, after = _project_patch_and_affected_steps(
        normalized_patch,
        before_workflow,
        after_workflow,
        catalog_snapshot=catalog_snapshot,
    )
    ledger = project_scientific_attempt_ledger(execution_result, workflow=after_workflow)
    # 修订后新增步骤使用 nN；既有步骤沿用 before 的 sN。这样 Attempt ledger、
    # affected_workflow_after 与 patch 的 step_ref 可以在同一条历史中互相解析。
    after_steps = (
        after_workflow.get("steps")
        if isinstance(after_workflow, Mapping) and isinstance(after_workflow.get("steps"), list)
        else []
    )
    before_steps = (
        before_workflow.get("steps")
        if isinstance(before_workflow, Mapping) and isinstance(before_workflow.get("steps"), list)
        else []
    )
    before_ids = {
        item.get("step_id")
        for item in before_steps
        if isinstance(item, Mapping) and isinstance(item.get("step_id"), str)
    }
    new_ids = [
        item.get("step_id")
        for item in after_steps
        if isinstance(item, Mapping)
        and isinstance(item.get("step_id"), str)
        and item.get("step_id") not in before_ids
    ]
    old_ref_by_id = {
        item.get("step_id"): f"s{index + 1}"
        for index, item in enumerate(before_steps)
        if isinstance(item, Mapping) and isinstance(item.get("step_id"), str)
    }
    new_ref_by_id = {step_id: f"n{index + 1}" for index, step_id in enumerate(new_ids)}
    if isinstance(ledger.get("steps"), list):
        for index, item in enumerate(ledger["steps"]):
            if not isinstance(item, dict) or index >= len(after_steps):
                continue
            formal_id = after_steps[index].get("step_id") if isinstance(after_steps[index], Mapping) else None
            item["step_ref"] = old_ref_by_id.get(formal_id, new_ref_by_id.get(formal_id, f"s{index + 1}"))
    return {
        "patch": patch,
        "affected_workflow_before": before,
        "affected_workflow_after": after,
        "attempt_ledger": ledger,
        "program_gate": project_program_gate_fact(
            program_validation,
            evidence_offset=_ledger_fact_count(ledger),
            workflow=after_workflow,
            step_ref_map={**old_ref_by_id, **new_ref_by_id},
        ),
        "judge": project_judge_fact(
            router_decision,
            step_count=len(ledger["steps"]),
            ledger=ledger,
            program_validation=program_validation,
            workflow=after_workflow,
        ),
    }


# ==================== 数据库行转换 ====================

def _load_json(value: Any, default: Any) -> Any:
    """读取数据库中的JSON；损坏时停止重建，不生成看似正常的默认状态。"""

    if value is None:
        return default
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ContractValidationError("Working Memory 投影包含损坏的 JSON") from exc


def task_row(row: sqlite3.Row) -> Dict[str, Any]:
    """把任务数据库行转换为调用方可安全读取的普通字典。"""

    result = dict(row)
    for source, target, default in (
        ("chemistry_context_json", "chemistry_context", {}),
    ):
        result[target] = _load_json(result.pop(source, None), default)
    return result


def batch_row(row: sqlite3.Row) -> Dict[str, Any]:
    """把假设批次数据库行转换为普通字典。"""

    return dict(row)


def candidate_row(row: sqlite3.Row) -> Dict[str, Any]:
    """只返回候选的索引、排名和选择状态，不重复保存假设正文。"""

    return dict(row)


def combination_row(row: sqlite3.Row) -> Dict[str, Any]:
    """还原一个组合的当前工作流、进度游标和运行状态。"""

    result = dict(row)
    result["current_workflow"] = _load_json(result.pop("current_workflow_json", None), None)
    result["runtime_checkpoint"] = _load_json(result.pop("runtime_checkpoint_json", None), {})
    runtime = result["runtime_checkpoint"]
    if isinstance(runtime, Mapping):
        if isinstance(runtime.get("gaussian_code_results"), Mapping):
            result["gaussian_code_results"] = deepcopy(
                dict(runtime["gaussian_code_results"])
            )
            # 兼容尚未迁移的调用方；两者都指向同一份 manifest 投影，不复制原始
            # codegen 文件，也不创建第二份收据。
            result["gaussian_codegen_receipts"] = deepcopy(
                dict(runtime["gaussian_code_results"])
            )
        if isinstance(runtime.get("expert_review"), Mapping):
            result["expert_review"] = deepcopy(dict(runtime["expert_review"]))
            if result["expert_review"].get("status") == "reject":
                result["terminal_reason"] = "expert_gaussian_reject"
        if isinstance(runtime.get("expert_debate"), Mapping):
            result["debate_json"] = deepcopy(dict(runtime["expert_debate"]))
        if isinstance(runtime.get("expert_pause"), Mapping):
            result["expert_pause"] = deepcopy(dict(runtime["expert_pause"]))
            if runtime["expert_pause"].get("phase") == "gaussian_step_review":
                result["expert_review"] = {
                    "status": "not_configured",
                    "records": [],
                    "overrides": {},
                    "requested_transition": None,
                    "reason": runtime["expert_pause"].get("reason"),
                }
            result["terminal_reason"] = "expert_model_not_configured"
    return result


def event_row(row: sqlite3.Row) -> Dict[str, Any]:
    """把事件数据库行还原为包含正文和证据引用的完整事实。"""

    result = dict(row)
    result["payload"] = _load_json(result.pop("payload_json"), {})
    result["source_refs"] = _load_json(result.pop("source_refs_json"), [])
    return result


# ==================== Investigation 事实当前索引（由事件重建） ====================

def investigation_session_row(
    row: sqlite3.Row,
    *,
    include_internal: bool = False,
) -> Dict[str, Any]:
    """读取 Investigation session，并按开关隐藏恢复身份。

    输入是当前索引行和是否供恢复代码读取内部字段的开关；动作只复制组合阶段、
    版本和事件游标，默认删除 thread、workspace、scope；返回可读字典，不访问外部
    服务、不返回 transcript，也不修改数据库。
    """

    value = dict(row)
    if not include_internal:
        for field in ("codex_thread_id", "workspace_ref", "scope_id"):
            value.pop(field, None)
    return value


def investigation_result_row(
    row: sqlite3.Row,
    *,
    include_internal: bool = False,
) -> Dict[str, Any]:
    """读取一版 InvestigationResult，并按公开接口筛除内部恢复字段。

    输入是结果当前索引行和是否允许内部恢复字段的开关；动作解码科学观测、限制和
    安全引用数组，默认删除 thread、scope、operation 身份；返回完整科学结果字典。
    JSON 损坏时抛出合同错误，不访问 MiniChem、路径或 transcript，也不产生写副作用。
    """

    value = dict(row)
    for source, target in (
        ("job_refs_json", "job_refs"),
        ("operation_refs_json", "operation_refs"),
        ("artifact_refs_json", "artifact_refs"),
        ("observations_json", "observations"),
        ("limitations_json", "limitations"),
    ):
        value[target] = _load_json(value.pop(source, None), [])
    if not include_internal:
        for field in ("codex_thread_id", "scope_id", "operation_instance_id"):
            value.pop(field, None)
    return value


def investigation_review_row(
    row: sqlite3.Row,
    *,
    include_internal: bool = False,
) -> Dict[str, Any]:
    """读取批次 Review，并默认隐藏审查会话身份。

    输入是 Review 当前索引行和是否供恢复代码查看内部字段的开关；动作解码五项
    组合/版本向量与 findings，默认删除 thread、turn、operation；返回完整审查字典。
    JSON 损坏时抛出合同错误，不返回 transcript，也不修改数据库。
    """

    value = dict(row)
    for source, target in (
        ("combination_ids_json", "combination_ids"),
        ("result_versions_json", "result_versions"),
        ("result_version_vector_json", "result_version_vector"),
        ("findings_json", "findings"),
    ):
        value[target] = _load_json(value.pop(source, None), [])
    value["can_answer"] = bool(value["can_answer"])
    if not include_internal:
        for field in ("review_thread_id", "review_turn_id", "operation_instance_id"):
            value.pop(field, None)
    return value


def investigation_repair_row(
    row: sqlite3.Row,
    *,
    include_internal: bool = False,
) -> Dict[str, Any]:
    """读取返修当前索引，并保留请求/结果事件引用供恢复核对。

    输入是返修索引行和是否查看内部 operation 字段的开关；动作只复制请求编号、
    fail advice 与因果事件身份，默认隐藏 operation；返回返修字典，不读取 provider、
    不返回 transcript，也不修改数据库。
    """

    value = dict(row)
    if not include_internal:
        value.pop("operation_instance_id", None)
    return value


def investigation_route_marker_row(row: sqlite3.Row) -> Dict[str, Any]:
    """读取只含路由标识和真实来源引用的 marker 当前索引。

    输入是 marker 当前索引行；动作只复制组合身份、marker 和 Result/Review 事件引用；
    返回可供路由器核对的字典，不承载科学原因、不读取外部服务，也不修改数据库。
    """

    return dict(row)


def investigation_evidence_row(
    row: sqlite3.Row,
    *,
    include_internal: bool = False,
) -> Dict[str, Any]:
    """读取安全证据摘要索引，并按开关隐藏恢复身份。

    输入是已通过窄合同的当前索引行和是否查看内部字段的开关；动作解码终态
    jobs/operations/artifacts 摘要，默认删除 thread、scope、turn 等恢复身份；返回安全
    证据字典，不读取 MiniChem、路径、secret 或 transcript，也不产生写副作用。
    """

    value = dict(row)
    for source, target in (
        ("jobs_json", "jobs"),
        ("operations_json", "operations"),
        ("artifacts_json", "artifacts"),
    ):
        value[target] = _load_json(value.pop(source, None), [])
    if not include_internal:
        for field in ("codex_thread_id", "scope_id", "turn_id"):
            value.pop(field, None)
    return value


def task_review_session_row(
    row: sqlite3.Row,
    *,
    include_internal: bool = False,
) -> Dict[str, Any]:
    """读取每 task 唯一的 Review 会话，并按开关隐藏恢复身份。

    输入是当前索引行和是否供恢复代码读取内部字段的开关；动作只复制 provider、
    turn 游标和阶段状态，默认删除 sandbox 摘要、workspace、CODEX_HOME、scope、
    thread；返回可读字典，不访问 provider、不返回 transcript，也不修改数据库。
    """

    value = dict(row)
    if not include_internal:
        for field in (
            "sandbox_policy_digest",
            "workspace_ref",
            "codex_home_ref",
            "scope_id",
            "codex_thread_id",
        ):
            value.pop(field, None)
    return value


def task_review_turn_row(
    row: sqlite3.Row,
    *,
    include_internal: bool = False,
) -> Dict[str, Any]:
    """读取 Review turn 当前索引并按需要隐藏恢复身份。

    输入是数据库行和是否允许内部恢复代码查看身份的开关；动作只复制当前索引，
    默认删除 provider thread、请求摘要和 operation，避免把 transcript 入口带给
    模型；返回可读字典，行结构缺失时让数据库错误暴露，不修改数据库。
    """

    value = dict(row)
    if not include_internal:
        for field in (
            "review_thread_id",
            "request_digest",
            "operation_instance_id",
            "logical_operation_instance_id",
            "previous_attempt_closure_event_ref",
        ):
            value.pop(field, None)
    return value


def investigation_repair_turn_row(
    row: sqlite3.Row,
    *,
    include_internal: bool = False,
) -> Dict[str, Any]:
    """读取真正启动的返修 turn，并按需要隐藏恢复身份。

    输入是返修 turn 的当前索引行和内部读取开关；动作只复制事件身份、请求编号及
    安全摘要，默认删除 scope、thread、摘要和 operation，不返回 prompt/transcript；
    返回可读字典，不产生写副作用。
    """

    value = dict(row)
    if not include_internal:
        for field in (
            "codex_thread_id",
            "scope_id",
            "request_digest",
            "operation_instance_id",
            "logical_operation_instance_id",
            "previous_attempt_closure_event_ref",
        ):
            value.pop(field, None)
    return value


def investigation_turn_row(
    row: sqlite3.Row,
    *,
    include_internal: bool = False,
) -> Dict[str, Any]:
    """读取首轮 Investigation turn，并按开关隐藏 thread、scope 和 operation。

    输入是 turn 当前索引行和是否供恢复代码读取内部字段的开关；动作只复制 in-flight
    游标，默认删除恢复身份；返回可读字典，不返回 prompt/transcript，也不修改数据库。
    """

    value = dict(row)
    if not include_internal:
        for field in (
            "codex_thread_id",
            "scope_id",
            "operation_instance_id",
            "logical_operation_instance_id",
            "previous_attempt_closure_event_ref",
        ):
            value.pop(field, None)
    return value


def investigation_turn_wait_row(
    row: sqlite3.Row,
    *,
    include_internal: bool = False,
) -> Dict[str, Any]:
    """读取 waiting 历史索引，并按开关隐藏作业和恢复内部身份。

    输入是不可变 waiting 事件的索引行；动作只复制事件、attempt、作业三元身份和两
    个不可逆摘要，不读取 sidecar 正文或外部作业服务。默认查询删除 thread、workspace 摘要、
    scope、作业标识和摘要，恢复代码显式请求内部字段时才返回完整核对材料。
    """

    value = dict(row)
    if not include_internal:
        for field in (
            "session_id",
            "codex_thread_id",
            "workspace_ref_digest",
            "scope_id",
            "logical_operation_instance_id",
            "attempt_operation_instance_id",
            "job_id",
            "submission_operation_id",
            "submission_key",
            "submission_request_digest",
            "sidecar_record_digest",
        ):
            value.pop(field, None)
    return value


def investigation_task_terminal_row(row: sqlite3.Row) -> Dict[str, Any]:
    """把 TaskClosed 当前索引转换为安全的任务终态字典。

    输入是已由事件重建的 SQLite 行；动作解码 marker 来源并把整数布尔值转换为
    Python bool，同时删除 operation 内部身份。返回值只包含回答性字段和 Review/
    marker 事件引用；行缺字段或 JSON 损坏时抛出数据库/合同异常，不访问外部服务、
    不读取 transcript，也不修改数据库。
    """

    value = dict(row)
    value["accepted"] = bool(value["accepted"])
    value["can_answer"] = bool(value["can_answer"])
    value["marker_event_refs"] = _load_json(
        value.pop("marker_event_refs_json", None), []
    )
    value.pop("operation_instance_id", None)
    return value


def _investigation_combination_for_session(
    conn: sqlite3.Connection,
    task_id: str,
    batch_id: str,
    combination_id: str,
) -> sqlite3.Row:
    """读取已入选的组合身份，阻止 session 脱离批次事实自行伪造。"""

    row = conn.execute(
        "SELECT c.*, b.state AS batch_state FROM combinations c "
        "JOIN hypothesis_batches b ON b.batch_id=c.batch_id AND b.task_id=c.task_id "
        "WHERE c.task_id=? AND c.batch_id=? AND c.combination_id=?",
        (task_id, batch_id, combination_id),
    ).fetchone()
    if row is None:
        raise ContractValidationError(
            "Investigation session 必须引用当前 task 已入选的 combination"
        )
    if row["batch_state"] != "running":
        raise ContractValidationError("Investigation session 只能绑定当前运行批次")
    if row["investigation_phase"] not in {
        "pending",
        "investigating",
        "repair_pending",
        "validated",
        "marked_for_hypothesis_revision",
    }:
        raise ContractValidationError(
            "Investigation session 只能绑定 investigation_v2 入选组合"
        )
    return row


def insert_investigation_session_projection(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    event_id: str,
    payload: Mapping[str, Any],
    created_at: str,
) -> None:
    """从 SessionStarted 事件建立可由历史重建的当前会话索引。

    输入是同一事务中的事件正文、事件 ID、task 身份和时间；动作核对入选组合，
    确保 thread/workspace/scope 不跨组合复用，并写入会话当前视图。返回值为空；
    来源不存在、身份冲突、重复 session 或数据库约束失败时抛出合同错误并回滚整笔
    事件写入，不保存模型 transcript。
    """

    value = validate_investigation_session_payload(payload)
    combination = _investigation_combination_for_session(
        conn, task_id, value["batch_id"], value["combination_id"]
    )
    session_id = value.get("session_id") or (
        f"session:{task_id}:{value['batch_id']}:{value['combination_id']}"
    )
    for field in ("codex_thread_id", "workspace_ref", "scope_id"):
        reused = conn.execute(
            f"SELECT combination_id FROM investigation_sessions WHERE task_id=? "
            f"AND {field}=? AND NOT (batch_id=? AND combination_id=?) LIMIT 1",
            (
                task_id,
                value[field],
                value["batch_id"],
                value["combination_id"],
            ),
        ).fetchone()
        if reused is not None:
            raise ContractValidationError(
                f"同一 task 的 {field} 不能绑定多个 Investigation combination"
            )
    existing = conn.execute(
        "SELECT session_event_id FROM investigation_sessions "
        "WHERE task_id=? AND batch_id=? AND combination_id=?",
        (task_id, value["batch_id"], value["combination_id"]),
    ).fetchone()
    if existing is not None:
        raise ContractValidationError("同一 Investigation combination 存在重复 session")
    conn.execute(
        "INSERT INTO investigation_sessions("
        "task_id,session_id,batch_id,combination_id,codex_thread_id,workspace_ref,scope_id,"
        "repair_count,phase,session_event_id,version,created_at,updated_at) "
        "VALUES(?,?,?,?,?,?,?,0,'created',?,?,?,?)",
        (
            task_id,
            session_id,
            value["batch_id"],
            value["combination_id"],
            value["codex_thread_id"],
            value["workspace_ref"],
            value["scope_id"],
            event_id,
            0,
            created_at,
            created_at,
        ),
    )
    if combination["investigation_phase"] == "pending":
        conn.execute(
            "UPDATE combinations SET investigation_phase='investigating',"
            "investigation_ready_state='investigating',"
            "investigation_stage_state='investigating',"
            "version=version+1,updated_at=? WHERE task_id=? AND batch_id=? "
            "AND combination_id=? AND investigation_phase='pending'",
            (created_at, task_id, value["batch_id"], value["combination_id"]),
        )


def insert_investigation_session_thread_bound_projection(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    event_id: str,
    payload: Mapping[str, Any],
    created_at: str,
) -> None:
    """从 ThreadBound 事件把 SDK thread 绑定到既有 session。

    输入是同一事务中的绑定正文、事件 ID、task 和时间；动作核对稳定 session 身份、
    当前未绑定状态及 task 内 thread 唯一性，再更新会话当前索引。返回值为空；未知
    session、重复/漂移绑定、身份复用或数据库错误会回滚事件，不保存 transcript。
    """

    value = validate_investigation_session_thread_bound_payload(payload)
    if value["task_id"] != task_id:
        raise ContractValidationError("ThreadBound task 身份不一致")
    session = conn.execute(
        "SELECT * FROM investigation_sessions WHERE task_id=? AND batch_id=? "
        "AND combination_id=? AND session_id=?",
        (
            task_id,
            value["batch_id"],
            value["combination_id"],
            value["session_id"],
        ),
    ).fetchone()
    if session is None:
        raise ContractValidationError("ThreadBound 引用了未知 Investigation session")
    if session["codex_thread_id"] is not None:
        if session["codex_thread_id"] == value["codex_thread_id"]:
            raise ContractValidationError("Investigation session 已绑定 thread")
        raise ContractValidationError("Investigation session 不能漂移到第二个 thread")
    reused_identity = conn.execute(
        "SELECT combination_id FROM investigation_sessions WHERE task_id=? "
        "AND (workspace_ref=? OR scope_id=?) AND NOT (batch_id=? AND combination_id=?) LIMIT 1",
        (
            task_id,
            value["workspace_ref"],
            value["scope_id"],
            value["batch_id"],
            value["combination_id"],
        ),
    ).fetchone()
    if reused_identity is not None:
        raise ContractValidationError("同一 task 的 workspace_ref/scope_id 不能绑定多个组合")
    reused = conn.execute(
        "SELECT combination_id FROM investigation_sessions WHERE task_id=? "
        "AND codex_thread_id=? AND NOT (batch_id=? AND combination_id=?) LIMIT 1",
        (
            task_id,
            value["codex_thread_id"],
            value["batch_id"],
            value["combination_id"],
        ),
    ).fetchone()
    if reused is not None:
        raise ContractValidationError("同一 task 的 codex_thread_id 不能绑定多个组合")
    conn.execute(
        "UPDATE investigation_sessions SET codex_thread_id=?,workspace_ref=?,scope_id=?,"
        "phase='investigating',"
        "version=version+1,updated_at=? WHERE task_id=? AND batch_id=? AND combination_id=? "
        "AND session_id=? AND codex_thread_id IS NULL",
        (
            value["codex_thread_id"],
            value["workspace_ref"],
            value["scope_id"],
            created_at,
            task_id,
            value["batch_id"],
            value["combination_id"],
            value["session_id"],
        ),
    )


def insert_investigation_session_thread_migrated_projection(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    event_id: str,
    payload: Mapping[str, Any],
    created_at: str,
) -> None:
    """把显式 provider 迁移事实投影为 session 的当前 thread。

    输入是已经核验的旧 thread、新 thread、waiting 作业身份和 session 事实；动作只
    在当前 session 仍指向旧 thread、目标 thread 未被其它组合占用且 workspace/scope
    不变时更新当前索引。旧 thread 只存在于不可变事件和等待作业审计中，不会被删除或
    改写；重复事件在事实层幂等，重建时遇到漂移立即失败。
    """

    value = validate_investigation_session_thread_migrated_payload(payload)
    if value["task_id"] != task_id:
        raise ContractValidationError("ThreadMigrated task 身份不一致")
    session = conn.execute(
        "SELECT * FROM investigation_sessions WHERE task_id=? AND batch_id=? "
        "AND combination_id=? AND session_id=?",
        (
            task_id,
            value["batch_id"],
            value["combination_id"],
            value["session_id"],
        ),
    ).fetchone()
    if session is None:
        raise ContractValidationError("ThreadMigrated 引用了未知 Investigation session")
    if session["workspace_ref"] != value["workspace_ref"] or session["scope_id"] != value["scope_id"]:
        raise ContractValidationError("ThreadMigrated 与 session workspace/scope 不一致")
    if session["codex_thread_id"] != value["source_codex_thread_id"]:
        raise ContractValidationError("ThreadMigrated source thread 不是当前 session thread")
    reused = conn.execute(
        "SELECT combination_id FROM investigation_sessions WHERE task_id=? "
        "AND codex_thread_id=? AND NOT (batch_id=? AND combination_id=?) LIMIT 1",
        (
            task_id,
            value["target_codex_thread_id"],
            value["batch_id"],
            value["combination_id"],
        ),
    ).fetchone()
    if reused is not None:
        raise ContractValidationError("同一 task 的目标 codex_thread_id 不能绑定多个组合")
    conn.execute(
        "UPDATE investigation_sessions SET codex_thread_id=?,version=version+1,"
        "updated_at=? WHERE task_id=? AND batch_id=? AND combination_id=? "
        "AND session_id=? AND codex_thread_id=?",
        (
            value["target_codex_thread_id"],
            created_at,
            task_id,
            value["batch_id"],
            value["combination_id"],
            value["session_id"],
            value["source_codex_thread_id"],
        ),
    )
    changed = conn.execute("SELECT changes() AS count").fetchone()["count"]
    if changed != 1:
        raise ContractValidationError("ThreadMigrated 未能更新唯一 session")


def insert_investigation_turn_started_projection(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    event_id: str,
    payload: Mapping[str, Any],
    created_at: str,
) -> None:
    """从首轮 turn 事件建立组合 in-flight 游标。"""

    value = validate_investigation_turn_started_payload(payload)
    if value["task_id"] != task_id:
        raise ContractValidationError("InvestigationTurn task 身份不一致")
    session = conn.execute(
        "SELECT * FROM investigation_sessions WHERE task_id=? AND batch_id=? "
        "AND combination_id=? AND session_id=?",
        (
            task_id,
            value["batch_id"],
            value["combination_id"],
            value["session_id"],
        ),
    ).fetchone()
    if session is None:
        raise ContractValidationError("InvestigationTurn 引用了未知 session")
    if session["codex_thread_id"] != value["codex_thread_id"] or session["scope_id"] != value["scope_id"]:
        raise ContractValidationError("InvestigationTurn 与 session 身份不一致")
    if session["route_marker"] is not None or session["pending_repair_event_id"] is not None:
        raise ContractValidationError("终态或 pending repair 组合不能启动首轮 turn")
    if session["current_turn_event_id"] is not None:
        current = conn.execute(
            "SELECT attempt_status FROM investigation_turns WHERE task_id=? "
            "AND turn_started_event_id=?",
            (task_id, session["current_turn_event_id"]),
        ).fetchone()
        if current is None or current["attempt_status"] == "in_flight":
            raise ContractValidationError("当前 session 已有未消费的 Investigation initial turn")
    prior_rows = conn.execute(
        "SELECT * FROM investigation_turns WHERE task_id=? AND session_id=? "
        "ORDER BY attempt_number,created_at,turn_started_event_id",
        (task_id, value["session_id"]),
    ).fetchall()
    expected_attempt = (prior_rows[-1]["attempt_number"] + 1) if prior_rows else 1
    if value["attempt_number"] != expected_attempt:
        raise ContractValidationError("Investigation initial attempt_number 必须连续递增")
    if expected_attempt > 1:
        _validate_turn_attempt_replacement(
            conn,
            task_id=task_id,
            value=value,
            previous=prior_rows[-1],
            turn_field="turn_id",
            label="initial",
        )
    if conn.execute(
        "SELECT 1 FROM investigation_turns WHERE task_id=? AND session_id=? AND turn_id=?",
        (task_id, value["session_id"], value["turn_id"]),
    ).fetchone() is not None:
        raise ContractValidationError("同一 Investigation session 不能重复启动 turn")
    conn.execute(
        "INSERT INTO investigation_turns(turn_started_event_id,task_id,session_id,batch_id,"
        "combination_id,codex_thread_id,scope_id,turn_id,operation_instance_id,attempt_number,"
        "attempt_status,logical_operation_instance_id,previous_attempt_closure_event_ref,created_at) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            event_id,
            task_id,
            value["session_id"],
            value["batch_id"],
            value["combination_id"],
            value["codex_thread_id"],
            value["scope_id"],
            value["turn_id"],
            value["operation_instance_id"],
            value["attempt_number"],
            value["attempt_status"],
            value["logical_operation_instance_id"],
            value.get("previous_attempt_closure_event_ref"),
            created_at,
        ),
    )
    conn.execute(
        "UPDATE investigation_sessions SET current_turn_event_id=?,current_turn_id=?,"
        "current_turn_attempt_number=?,"
        "phase='investigating',version=version+1,updated_at=? WHERE task_id=? AND batch_id=? "
        "AND combination_id=? AND session_id=?",
        (
            event_id,
            value["turn_id"],
            value["attempt_number"],
            created_at,
            task_id,
            value["batch_id"],
            value["combination_id"],
            value["session_id"],
        ),
    )


def insert_investigation_turn_waiting_projection(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    event_id: str,
    payload: Mapping[str, Any],
    created_at: str,
) -> None:
    """从 waiting 事实建立 resumable attempt，并清除当前活动指针。

    输入是已追加的 initial/repair waiting 事件；动作重新读取并校验其 Started 事实、
    session 身份、当前 in-flight 投影和完整作业摘要，随后把 turn 标为 ``resumable``、
    清除对应 session active pointer，并把不可变等待记录写入历史索引。返回值为空；
    Review、非当前 attempt、身份/摘要漂移或重复 waiting 均抛出合同错误，调用事务回滚。
    """

    value = validate_investigation_turn_waiting_payload(payload)
    if value["task_id"] != task_id:
        raise ContractValidationError("InvestigationTurnWaiting task 身份不一致")
    started_event = conn.execute(
        "SELECT event_type,payload_json FROM memory_events WHERE task_id=? AND event_id=?",
        (task_id, value["turn_started_event_ref"]),
    ).fetchone()
    expected_type = {
        "initial": "investigation_turn_started",
        "repair": "investigation_repair_turn_started",
    }[value["kind"]]
    if started_event is None or started_event["event_type"] != expected_type:
        raise ContractValidationError(
            "InvestigationTurnWaiting 只能引用匹配的 initial/repair Started 事件"
        )
    started_validator = (
        validate_investigation_turn_started_payload
        if value["kind"] == "initial"
        else validate_investigation_repair_turn_started_payload
    )
    started = started_validator(_load_json(started_event["payload_json"], {}))
    identity_fields = [
        "task_id",
        "batch_id",
        "combination_id",
        "codex_thread_id",
        "scope_id",
        "turn_id",
        "attempt_number",
        "logical_operation_instance_id",
    ]
    if value["kind"] == "initial":
        identity_fields.append("session_id")
    for field in identity_fields:
        if started.get(field) != value.get(field):
            raise ContractValidationError(
                f"InvestigationTurnWaiting.{field} 与 Started 身份不一致"
            )
    if started["operation_instance_id"] != value["attempt_operation_instance_id"]:
        raise ContractValidationError(
            "InvestigationTurnWaiting.attempt_operation_instance_id 与 Started 不一致"
        )
    session = conn.execute(
        "SELECT * FROM investigation_sessions WHERE task_id=? AND batch_id=? "
        "AND combination_id=? AND session_id=?",
        (
            task_id,
            value["batch_id"],
            value["combination_id"],
            value["session_id"],
        ),
    ).fetchone()
    if session is None:
        raise ContractValidationError("InvestigationTurnWaiting 缺少对应 session")
    if (
        session["codex_thread_id"] != value["codex_thread_id"]
        or hashlib.sha256(
            canonical_json(session["workspace_ref"]).encode("utf-8")
        ).hexdigest()
        != value["workspace_ref_digest"]
        or session["scope_id"] != value["scope_id"]
        or session["session_id"] != value["session_id"]
    ):
        raise ContractValidationError(
            "InvestigationTurnWaiting 与 session thread/workspace/scope 身份不一致"
        )
    table = "investigation_turns" if value["kind"] == "initial" else "investigation_repair_turns"
    turn = conn.execute(
        f"SELECT * FROM {table} WHERE task_id=? AND turn_started_event_id=?",
        (task_id, value["turn_started_event_ref"]),
    ).fetchone()
    if turn is None:
        raise ContractValidationError("InvestigationTurnWaiting 缺少 Started 投影行")
    if (
        turn["attempt_status"] != "in_flight"
        or turn["attempt_number"] != value["attempt_number"]
        or turn["logical_operation_instance_id"]
        != value["logical_operation_instance_id"]
        or turn["operation_instance_id"] != value["attempt_operation_instance_id"]
        or turn["turn_id"] != value["turn_id"]
    ):
        raise ContractValidationError(
            "InvestigationTurnWaiting 与当前 in-flight attempt 身份不一致"
        )
    pointer = (
        session["current_turn_event_id"]
        if value["kind"] == "initial"
        else session["current_repair_turn_event_id"]
    )
    if pointer != value["turn_started_event_ref"]:
        raise ContractValidationError(
            "InvestigationTurnWaiting 不是当前 session active attempt"
        )
    conn.execute(
        "INSERT INTO investigation_turn_waits("
        "waiting_event_id,task_id,turn_started_event_ref,kind,batch_id,combination_id,"
        "session_id,codex_thread_id,workspace_ref_digest,scope_id,turn_id,attempt_number,"
        "logical_operation_instance_id,attempt_operation_instance_id,job_id,submission_operation_id,"
        "submission_key,submission_request_digest,sidecar_record_digest,created_at) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            event_id,
            task_id,
            value["turn_started_event_ref"],
            value["kind"],
            value["batch_id"],
            value["combination_id"],
            value["session_id"],
            value["codex_thread_id"],
            value["workspace_ref_digest"],
            value["scope_id"],
            value["turn_id"],
            value["attempt_number"],
            value["logical_operation_instance_id"],
            value["attempt_operation_instance_id"],
            value["job_id"],
            value["submission_operation_id"],
            value["submission_key"],
            value["submission_request_digest"],
            value["sidecar_record_digest"],
            created_at,
        ),
    )
    conn.execute(
        f"UPDATE {table} SET attempt_status='resumable' "
        "WHERE task_id=? AND turn_started_event_id=? AND attempt_status='in_flight'",
        (task_id, value["turn_started_event_ref"]),
    )
    if conn.execute("SELECT changes() AS count").fetchone()["count"] != 1:
        raise ContractValidationError("InvestigationTurnWaiting 未能标记唯一 resumable attempt")
    if value["kind"] == "initial":
        conn.execute(
            "UPDATE investigation_sessions SET current_turn_event_id=NULL,current_turn_id=NULL,"
            "current_turn_attempt_number=NULL,version=version+1,updated_at=? "
            "WHERE task_id=? AND batch_id=? AND combination_id=? AND current_turn_event_id=?",
            (
                created_at,
                task_id,
                value["batch_id"],
                value["combination_id"],
                value["turn_started_event_ref"],
            ),
        )
    else:
        conn.execute(
            "UPDATE investigation_sessions SET current_repair_turn_event_id=NULL,"
            "current_repair_turn_id=NULL,version=version+1,updated_at=? "
            "WHERE task_id=? AND batch_id=? AND combination_id=? "
            "AND current_repair_turn_event_id=?",
            (
                created_at,
                task_id,
                value["batch_id"],
                value["combination_id"],
                value["turn_started_event_ref"],
            ),
        )
    if conn.execute("SELECT changes() AS count").fetchone()["count"] != 1:
        raise ContractValidationError("InvestigationTurnWaiting 未能清除唯一 active pointer")


def _load_turn_termination_started_projection(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    value: Mapping[str, Any],
) -> tuple[Mapping[str, Any], sqlite3.Row]:
    """读取并校验 closure 引用的 Started 正文和投影行。

    输入是事务连接、task 和已通过 closure 合同的正文；动作按 kind 查找唯一 Started
    事件和对应 turn 行，逐项运行三类 Started 合同。返回冻结的 Started mapping 与
    当前投影行；事件类型、投影缺失或正文损坏时抛出合同错误，不修改数据库。
    """

    expected_type = {
        "initial": "investigation_turn_started",
        "repair": "investigation_repair_turn_started",
        "review": "task_review_turn_started",
    }[value["kind"]]
    started_event = conn.execute(
        "SELECT event_type,payload_json FROM memory_events WHERE task_id=? AND event_id=?",
        (task_id, value["turn_started_event_ref"]),
    ).fetchone()
    if started_event is None or started_event["event_type"] != expected_type:
        raise ContractValidationError("turn termination 引用的 Started 事件类型不正确")
    validators = {
        "initial": validate_investigation_turn_started_payload,
        "repair": validate_investigation_repair_turn_started_payload,
        "review": validate_task_review_turn_started_payload,
    }
    tables = {
        "initial": "investigation_turns",
        "repair": "investigation_repair_turns",
        "review": "task_review_turns",
    }
    started = validators[value["kind"]](_load_json(started_event["payload_json"], {}))
    row = conn.execute(
        f"SELECT * FROM {tables[value['kind']]} WHERE task_id=? AND turn_started_event_id=?",
        (task_id, value["turn_started_event_ref"]),
    ).fetchone()
    if row is None:
        raise ContractValidationError("turn termination 缺少 Started 投影行")
    return started, row


def _validate_turn_attempt_replacement(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    value: Mapping[str, Any],
    previous: sqlite3.Row,
    turn_field: str,
    label: str,
) -> None:
    """核对 replacement attempt 必须沿用逻辑身份并引用明确 closure。

    输入是当前 Started 正文、同一逻辑链上一条历史 turn 和事务连接；动作比较新的
    turn/op/logical/request_digest 身份，并核对 previous closure 事件及其状态。返回值
    为空；重复 turn、身份或摘要漂移、缺 closure 时抛出合同错误，不修改数据库。
    """

    if (
        value.get("batch_id") is not None
        and previous["batch_id"] != value["batch_id"]
    ):
        raise ContractValidationError(
            f"replacement {label} turn 不得漂移 batch_id"
        )
    if value[turn_field] == previous[turn_field]:
        raise ContractValidationError(
            f"replacement {label} turn 必须使用新的 {turn_field}"
        )
    if value["operation_instance_id"] == previous["operation_instance_id"]:
        raise ContractValidationError(
            f"replacement {label} turn 必须使用新的 attempt operation"
        )
    if value["logical_operation_instance_id"] != previous["logical_operation_instance_id"]:
        raise ContractValidationError(
            f"replacement {label} turn 不得漂移 logical operation"
        )
    if "request_digest" in value and value["request_digest"] != previous["request_digest"]:
        raise ContractValidationError(
            f"replacement {label} turn 不得漂移 request_digest"
        )
    closure_ref = value.get("previous_attempt_closure_event_ref")
    if not isinstance(closure_ref, str):
        raise ContractValidationError(
            f"replacement {label} turn 必须引用 previous closure"
        )
    previous_status = previous["attempt_status"]
    if previous_status in {"terminal_failed", "terminal_interrupted"}:
        closure = conn.execute(
            "SELECT termination_event_id FROM investigation_turn_attempt_terminations "
            "WHERE task_id=? AND turn_started_event_ref=?",
            (task_id, previous["turn_started_event_id"]),
        ).fetchone()
    elif previous_status == "resumable" and label in {"initial", "repair"}:
        # waiting 事件是 resumable attempt 的唯一前序收束。它不写入终止表，
        # 但 replacement 必须精确引用同一 waiting event，禁止只凭状态递增。
        closure = conn.execute(
            "SELECT event_id AS termination_event_id FROM memory_events "
            "WHERE task_id=? AND event_type='investigation_turn_waiting' "
            "AND json_extract(payload_json,'$.turn_started_event_ref')=?",
            (task_id, previous["turn_started_event_id"]),
        ).fetchone()
    else:
        closure = None
    if closure is None or closure["termination_event_id"] != closure_ref:
        raise ContractValidationError(
            f"previous {label} attempt closure 不存在、类型不符或与历史不一致"
        )


def _validate_turn_termination_attempt_identity(
    value: Mapping[str, Any],
    started: Mapping[str, Any],
    row: sqlite3.Row,
) -> None:
    """核对 closure、Started 和 attempt 投影的冻结身份。

    输入是 closure 正文、对应 Started mapping 和投影行；动作只比较 attempt 编号、
    logical/per-attempt operation、turn 与 previous closure 引用。身份不一致、已消费
    或 Started 链字段漂移时抛出合同错误；成功返回 None，不写数据库。
    """

    if row["attempt_status"] != "in_flight":
        raise ContractValidationError("只能终止当前 in_flight attempt")
    row_turn_id = row["review_turn_id"] if value["kind"] == "review" else row["turn_id"]
    if (
        row["attempt_number"] != value["attempt_number"]
        or row["logical_operation_instance_id"] != value["logical_operation_instance_id"]
        or row["operation_instance_id"] != value["attempt_operation_instance_id"]
        or row_turn_id != value["turn_id"]
        or started.get("operation_instance_id") != value["attempt_operation_instance_id"]
        or started.get("previous_attempt_closure_event_ref")
        != value.get("previous_attempt_closure_event_ref")
    ):
        raise ContractValidationError("turn termination 与 Started attempt 身份不一致")


def _validate_turn_termination_active_owner(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    value: Mapping[str, Any],
    started: Mapping[str, Any],
) -> None:
    """核对 closure 所属的当前 session 或 Review active 指针。

    输入是事务连接、task、closure 正文和 Started mapping；动作按 initial/repair/review
    检查 batch、组合、thread、scope、session、review thread 及唯一活动指针。非当前、
    task/thread 漂移或活动 session 缺失时抛出合同错误；成功返回 None，不写数据库。
    """

    if value["kind"] == "review":
        _validate_review_turn_termination_owner(
            conn, task_id=task_id, value=value, started=started
        )
        return
    _validate_investigation_turn_termination_owner(
        conn, task_id=task_id, value=value, started=started
    )


def _validate_review_turn_termination_owner(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    value: Mapping[str, Any],
    started: Mapping[str, Any],
) -> None:
    """核对 Review closure 的 batch、thread 和 task active 指针。"""

    if (
        started.get("batch_id") != value["batch_id"]
        or started.get("review_turn_id") != value.get("review_turn_id")
        or started.get("review_thread_id") != value.get("review_thread_id")
        or started.get("review_thread_id") != value.get("codex_thread_id")
    ):
        raise ContractValidationError("turn termination Review 身份漂移")
    review_session = conn.execute(
        "SELECT active_turn_started_event_id FROM task_review_sessions WHERE task_id=?",
        (task_id,),
    ).fetchone()
    if review_session is None or review_session["active_turn_started_event_id"] != value["turn_started_event_ref"]:
        raise ContractValidationError("turn termination 不是当前 active Review attempt")


def _validate_investigation_turn_termination_owner(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    value: Mapping[str, Any],
    started: Mapping[str, Any],
) -> None:
    """核对 initial/repair closure 的组合 provider 和当前 session 指针。"""

    if (
        started.get("batch_id") != value["batch_id"]
        or started.get("combination_id") != value.get("combination_id")
        or started.get("turn_id") != value["turn_id"]
        or value.get("session_id") != started.get("session_id")
        or value.get("codex_thread_id") != started.get("codex_thread_id")
        or value.get("scope_id") != started.get("scope_id")
    ):
        raise ContractValidationError("turn termination session/provider 身份漂移")
    session = _investigation_session_for_event(
        conn, task_id, value["batch_id"], value["combination_id"]
    )
    pointer = (
        session["current_turn_event_id"]
        if value["kind"] == "initial"
        else session["current_repair_turn_event_id"]
    )
    if pointer != value["turn_started_event_ref"]:
        raise ContractValidationError("turn termination 不是当前 active attempt")


def _write_turn_termination_projection_and_clear_active(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    event_id: str,
    value: Mapping[str, Any],
    created_at: str,
) -> None:
    """写入 closure 索引、标记 terminal 并清除对应活动指针。

    输入是事务连接、closure 事件身份、已通过全部身份检查的正文和时间；动作插入
    immutable closure 索引，将 Started 行改为 terminal_failed/interrupted，并清除
    initial/repair/Review 的 active 指针。返回值为空；数据库约束失败由外层事务回滚。
    """

    conn.execute(
        "INSERT INTO investigation_turn_attempt_terminations("
        "termination_event_id,task_id,turn_started_event_ref,kind,batch_id,combination_id,"
        "turn_id,attempt_number,terminal_status,logical_operation_instance_id,"
        "attempt_operation_instance_id,previous_attempt_closure_event_ref,created_at) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            event_id, task_id, value["turn_started_event_ref"], value["kind"],
            value["batch_id"], value.get("combination_id"), value["turn_id"],
            value["attempt_number"], value["terminal_status"],
            value["logical_operation_instance_id"], value["attempt_operation_instance_id"],
            value.get("previous_attempt_closure_event_ref"), created_at,
        ),
    )
    table = {
        "initial": "investigation_turns",
        "repair": "investigation_repair_turns",
        "review": "task_review_turns",
    }[value["kind"]]
    conn.execute(
        f"UPDATE {table} SET attempt_status=? WHERE turn_started_event_id=?",
        (f"terminal_{value['terminal_status']}", value["turn_started_event_ref"]),
    )
    if value["kind"] == "initial":
        conn.execute(
            "UPDATE investigation_sessions SET current_turn_event_id=NULL,current_turn_id=NULL,"
            "current_turn_attempt_number=NULL,version=version+1,updated_at=? "
            "WHERE task_id=? AND batch_id=? AND combination_id=?",
            (created_at, task_id, value["batch_id"], value["combination_id"]),
        )
    elif value["kind"] == "repair":
        conn.execute(
            "UPDATE investigation_sessions SET current_repair_turn_event_id=NULL,"
            "current_repair_turn_id=NULL,version=version+1,updated_at=? "
            "WHERE task_id=? AND batch_id=? AND combination_id=?",
            (created_at, task_id, value["batch_id"], value["combination_id"]),
        )
    else:
        conn.execute(
            "UPDATE task_review_sessions SET active_batch_id=NULL,active_turn_id=NULL,"
            "active_turn_started_event_id=NULL,phase='ready',version=version+1,updated_at=? "
            "WHERE task_id=?",
            (created_at, task_id),
        )


def insert_investigation_turn_attempt_termination_projection(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    event_id: str,
    payload: Mapping[str, Any],
    created_at: str,
) -> None:
    """从 immutable closure 事件关闭且仅关闭当前 in-flight attempt。

    输入是服务从 Started 事件逐值冻结的 closure 正文；动作回查 Started、当前活动
    指针和 attempt 行，确认 task/kind/turn/operation 全部一致，再写入 closure 表并
    清除对应活动指针。已消费、非当前、断链或身份漂移均 fail-closed；返回值为空，
    事务失败时不留下半条终止索引。
    """

    value = validate_investigation_turn_attempt_termination_payload(payload)
    if value["task_id"] != task_id:
        raise ContractValidationError("turn termination task 身份不一致")
    started, row = _load_turn_termination_started_projection(
        conn, task_id=task_id, value=value
    )
    _validate_turn_termination_attempt_identity(value, started, row)
    _validate_turn_termination_active_owner(
        conn, task_id=task_id, value=value, started=started
    )
    closure_event = conn.execute(
        "SELECT source_refs_json FROM memory_events WHERE task_id=? AND event_id=?",
        (task_id, event_id),
    ).fetchone()
    if closure_event is None:
        raise ContractValidationError("turn termination 事件缺少不可变来源")
    source_refs = _load_json(closure_event["source_refs_json"], [])
    if value["turn_started_event_ref"] not in source_refs:
        raise ContractValidationError("Started 事件来源已损坏")
    _write_turn_termination_projection_and_clear_active(
        conn,
        task_id=task_id,
        event_id=event_id,
        value=value,
        created_at=created_at,
    )


def _investigation_event_source_refs(
    conn: sqlite3.Connection,
    task_id: str,
    event_id: str,
) -> list[str]:
    """读取一条 Investigation 事件的来源列表，供 turn 因果核对使用。"""

    row = conn.execute(
        "SELECT source_refs_json FROM memory_events WHERE task_id=? AND event_id=?",
        (task_id, event_id),
    ).fetchone()
    if row is None:
        raise ContractValidationError("Investigation 事件缺少不可变来源")
    refs = _load_json(row["source_refs_json"], None)
    if not isinstance(refs, list) or any(not isinstance(item, str) for item in refs):
        raise ContractValidationError("Investigation 事件来源格式损坏")
    return refs


def _validate_investigation_evidence_turn_claim(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    event_id: str,
    value: Mapping[str, Any],
    session: sqlite3.Row,
) -> None:
    """核对证据确实由当前 initial 或 repair turn 产生。"""

    pending = session["pending_repair_event_id"]
    if pending is None:
        claim = None
        current_ref = session["current_turn_event_id"]
        if isinstance(current_ref, str):
            claim = conn.execute(
                "SELECT * FROM investigation_turns WHERE task_id=? "
                "AND turn_started_event_id=?",
                (task_id, current_ref),
            ).fetchone()
    else:
        claim = conn.execute(
            "SELECT * FROM investigation_repair_turns WHERE task_id=? "
            "AND request_event_ref=? ORDER BY created_at DESC,turn_started_event_id DESC LIMIT 1",
            (task_id, pending),
        ).fetchone()
    if claim is None:
        raise ContractValidationError("证据必须引用当前已成功启动的 initial 或 repair turn")
    if (
        claim["batch_id"] != value["batch_id"]
        or claim["combination_id"] != value["combination_id"]
        or claim["codex_thread_id"] != value["codex_thread_id"]
        or claim["scope_id"] != value["scope_id"]
        or claim["turn_id"] != value["turn_id"]
    ):
        raise ContractValidationError("证据的 turn/thread/scope 与当前 turn claim 不一致")
    if claim["turn_started_event_id"] not in _investigation_event_source_refs(
        conn, task_id, event_id
    ):
        raise ContractValidationError("证据 source_refs 必须引用当前 turn claim")


def _validate_investigation_result_turn_claim(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    event_id: str,
    value: Mapping[str, Any],
    session: sqlite3.Row,
    result_refs: Sequence[str],
) -> None:
    """核对 Result 的 turn 身份和当前 pending repair 因果。"""

    pending = session["pending_repair_event_id"]
    repair_ref = value.get("repair_turn_event_ref")
    if pending is None:
        if repair_ref is not None:
            raise ContractValidationError("首版 Result 不能引用 repair turn")
        current_ref = session["current_turn_event_id"]
        claim = (
            conn.execute(
                "SELECT * FROM investigation_turns WHERE task_id=? "
                "AND turn_started_event_id=?",
                (task_id, current_ref),
            ).fetchone()
            if isinstance(current_ref, str)
            else None
        )
    else:
        if not isinstance(repair_ref, str):
            raise ContractValidationError("返修 Result 必须引用当前 repair turn")
        claim = conn.execute(
            "SELECT * FROM investigation_repair_turns WHERE task_id=? "
            "AND turn_started_event_id=? AND request_event_ref=?",
            (task_id, repair_ref, pending),
        ).fetchone()
    if claim is None:
        raise ContractValidationError("Result 必须引用当前已成功启动的 initial 或 repair turn")
    if (
        claim["batch_id"] != value["batch_id"]
        or claim["combination_id"] != value["combination_id"]
        or claim["codex_thread_id"] != value["codex_thread_id"]
        or claim["scope_id"] != value["scope_id"]
        or claim["turn_id"] != value["turn_id"]
    ):
        raise ContractValidationError("Result 的 turn/thread/scope 与当前 turn claim 不一致")
    if claim["turn_started_event_id"] not in result_refs:
        raise ContractValidationError("Result source_refs 必须引用当前 turn claim")
def insert_investigation_evidence_verification_projection(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    event_id: str,
    payload: Mapping[str, Any],
    created_at: str,
) -> None:
    """从权威证据事件建立安全摘要索引，禁止普通字符串冒充验证对象。

    输入是已通过窄合同的证据事件；动作核对任务、批次、组合和 session 身份，并在
    同一事务写入 jobs/operations/artifacts 摘要。返回值为空；事件缺失、对象重复或
    thread/scope/turn 漂移时抛出合同错误，数据库不会留下半条索引。
    """

    value = validate_investigation_evidence_payload(payload)
    if value["task_id"] != task_id:
        raise ContractValidationError("证据事件 task 身份不一致")
    combination = _investigation_combination_for_session(
        conn, task_id, value["batch_id"], value["combination_id"]
    )
    session = conn.execute(
        "SELECT * FROM investigation_sessions "
        "WHERE task_id=? AND batch_id=? AND combination_id=?",
        (task_id, value["batch_id"], value["combination_id"]),
    ).fetchone()
    if session is not None and (
        session["codex_thread_id"] != value["codex_thread_id"]
        or session["scope_id"] != value["scope_id"]
    ):
        raise ContractValidationError("证据摘要与 Investigation session 身份不一致")
    if session is None:
        raise ContractValidationError("证据摘要缺少 Investigation session")
    _validate_investigation_evidence_turn_claim(
        conn,
        task_id=task_id,
        event_id=event_id,
        value=value,
        session=session,
    )
    existing = conn.execute(
        "SELECT verification_event_id FROM investigation_evidence_verifications "
        "WHERE task_id=? AND verification_object_id=?",
        (task_id, value["verification_object_id"]),
    ).fetchone()
    if existing is not None:
        raise ContractValidationError("同一 verification_object_id 不能绑定不同证据")
    conn.execute(
        "INSERT INTO investigation_evidence_verifications("
        "verification_event_id,task_id,batch_id,combination_id,verification_object_id,"
        "verifier_id,verifier_version,reference_binding_version,verification_digest,"
        "codex_thread_id,scope_id,"
        "turn_id,jobs_json,operations_json,artifacts_json,created_at) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            event_id,
            task_id,
            value["batch_id"],
            value["combination_id"],
            value["verification_object_id"],
            value["verifier_id"],
            value["verifier_version"],
            value.get("reference_binding_version"),
            value["verification_digest"],
            value["codex_thread_id"],
            value["scope_id"],
            value["turn_id"],
            canonical_json(value["jobs"]),
            canonical_json(value["operations"]),
            canonical_json(value["artifacts"]),
            created_at,
        ),
    )


def insert_task_review_session_projection(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    event_id: str,
    payload: Mapping[str, Any],
    created_at: str,
) -> None:
    """从 task Review 会话事件建立唯一 provider 冻结索引。

    输入是同事务中的 TaskReviewSessionStarted 正文、事件 ID、task 和时间；动作核对
    task 架构及唯一性，并冻结 provider、sandbox、workspace、CODEX_HOME、scope、thread
    和 turn 游标。返回值为空；legacy task、重复会话或合同/数据库错误会回滚事件与
    当前视图，不保存 transcript。
    """

    value = validate_task_review_session_payload(payload)
    if value["task_id"] != task_id:
        raise ContractValidationError("TaskReviewSession task 身份不一致")
    task_events = conn.execute(
        "SELECT event_type,payload_json FROM memory_events "
        "WHERE task_id=? AND event_type='task_created'",
        (task_id,),
    ).fetchall()
    if len(task_events) != 1:
        raise ContractValidationError("TaskReviewSession 缺少唯一 TaskCreated 事实")
    task_created = validate_task_created_payload(
        _load_json(task_events[0]["payload_json"], None)
    )
    if task_created["stage_architecture_version"] != "investigation_v2":
        raise ContractValidationError("只有 investigation_v2 task 可以创建 TaskReviewSession")
    if conn.execute(
        "SELECT 1 FROM task_review_sessions WHERE task_id=?", (task_id,)
    ).fetchone() is not None:
        raise ContractValidationError("每个 task 只能有一个 TaskReviewSession")
    conn.execute(
        "INSERT INTO task_review_sessions("
        "task_id,session_event_id,provider_id,model,sandbox_policy_digest,workspace_ref,"
        "codex_home_ref,scope_id,codex_thread_id,turn_count,phase,version,created_at,updated_at) "
        "VALUES(?,?,?,?,?,?,?,?,?,0,'created',0,?,?)",
        (
            task_id,
            event_id,
            value["provider_id"],
            value["model"],
            value["sandbox_policy_digest"],
            value["workspace_ref"],
            value["codex_home_ref"],
            value["scope_id"],
            value["codex_thread_id"],
            created_at,
            created_at,
        ),
    )


def insert_task_review_turn_projection(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    event_id: str,
    payload: Mapping[str, Any],
    created_at: str,
) -> None:
    """从 Review turn 启动事件更新 task 会话游标，并核对批次身份。

    输入是同一事务中的 turn 启动正文、事件 ID、task 和时间；动作核对冻结 thread、
    当前会话、已存在 batch 及连续 turn_index，再把会话置为 turn_in_flight。返回值为
    空；thread 漂移、编号跳跃、未知 batch、重复 turn 或数据库错误会回滚事件，不写
    prompt/transcript。
    """

    value = validate_task_review_turn_started_payload(payload)
    if value["task_id"] != task_id:
        raise ContractValidationError("TaskReviewTurn task 身份不一致")
    session = conn.execute(
        "SELECT * FROM task_review_sessions WHERE task_id=?", (task_id,)
    ).fetchone()
    if session is None or session["session_event_id"] != value["session_event_ref"]:
        raise ContractValidationError("Review turn 必须引用当前 task ReviewSession")
    if session["phase"] == "turn_in_flight" and session["active_turn_started_event_id"] is not None:
        active = conn.execute(
            "SELECT attempt_status FROM task_review_turns WHERE task_id=? "
            "AND turn_started_event_id=?",
            (task_id, session["active_turn_started_event_id"]),
        ).fetchone()
        if active is None or active["attempt_status"] == "in_flight":
            raise ContractValidationError("当前 Review turn 尚未完成，不能并行启动第二个 turn")
    if session["codex_thread_id"] != value["review_thread_id"]:
        raise ContractValidationError("Review turn 必须复用冻结的 review thread")
    batch = conn.execute(
        "SELECT batch_id FROM hypothesis_batches WHERE task_id=? AND batch_id=?",
        (task_id, value["batch_id"]),
    ).fetchone()
    if batch is None:
        raise ContractValidationError("Review turn 引用了未知 batch")
    expected_index = session["turn_count"] + 1
    if value["turn_index"] != expected_index:
        raise ContractValidationError("Review turn index 必须连续递增")
    prior_turns = conn.execute(
        "SELECT * FROM task_review_turns WHERE task_id=? AND turn_index=? "
        "ORDER BY attempt_number,created_at,turn_started_event_id",
        (task_id, value["turn_index"]),
    ).fetchall()
    expected_attempt = (prior_turns[-1]["attempt_number"] + 1) if prior_turns else 1
    if value["attempt_number"] != expected_attempt:
        raise ContractValidationError("Review attempt_number 必须连续递增")
    if expected_attempt > 1:
        _validate_turn_attempt_replacement(
            conn,
            task_id=task_id,
            value=value,
            previous=prior_turns[-1],
            turn_field="review_turn_id",
            label="Review",
        )
    if conn.execute(
        "SELECT 1 FROM task_review_turns WHERE task_id=? AND review_turn_id=?",
        (task_id, value["review_turn_id"]),
    ).fetchone() is not None:
        raise ContractValidationError("Review turn identity 已绑定不同内容")
    conn.execute(
        "INSERT INTO task_review_turns("
        "turn_started_event_id,task_id,session_event_ref,batch_id,review_thread_id,"
        "review_turn_id,turn_index,request_digest,operation_instance_id,attempt_number,"
        "attempt_status,logical_operation_instance_id,previous_attempt_closure_event_ref,created_at) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            event_id,
            task_id,
            value["session_event_ref"],
            value["batch_id"],
            value["review_thread_id"],
            value["review_turn_id"],
            value["turn_index"],
            value["request_digest"],
            value["operation_instance_id"],
            value["attempt_number"],
            value["attempt_status"],
            value["logical_operation_instance_id"],
            value.get("previous_attempt_closure_event_ref"),
            created_at,
        ),
    )
    conn.execute(
        "UPDATE task_review_sessions SET active_batch_id=?,active_turn_id=?,"
        "active_turn_started_event_id=?,phase='turn_in_flight',version=version+1,updated_at=? "
        "WHERE task_id=?",
        (
            value["batch_id"],
            value["review_turn_id"],
            event_id,
            created_at,
            task_id,
        ),
    )


def insert_investigation_repair_turn_started_projection(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    event_id: str,
    payload: Mapping[str, Any],
    created_at: str,
) -> None:
    """从成功启动的 deepseek_repair turn 建立启动索引和 pending 阶段。

    输入是 provider 已实际接受的 turn 启动正文、事件 ID、task 和时间；动作核对当前
    pending repair、请求编号、session 身份及唯一性，随后只登记启动事实；返修预算在
    新 Result 成功登记时递增。返回值为空；启动前失败、重复启动、编号越界或身份漂移
    会整体回滚，因此 provider 启动后的崩溃不会消耗预算，也不保存 prompt/transcript。
    """

    value = validate_investigation_repair_turn_started_payload(payload)
    if value["task_id"] != task_id:
        raise ContractValidationError("Repair turn task 身份不一致")
    session = _investigation_session_for_event(
        conn, task_id, value["batch_id"], value["combination_id"]
    )
    request = _validate_repair_turn_start_context(
        conn, task_id=task_id, value=value, session=session
    )
    prior_turns = conn.execute(
        "SELECT * FROM investigation_repair_turns WHERE task_id=? AND request_event_ref=? "
        "ORDER BY attempt_number,created_at,turn_started_event_id",
        (task_id, value["request_event_ref"]),
    ).fetchall()
    expected_attempt = (prior_turns[-1]["attempt_number"] + 1) if prior_turns else 1
    if value["attempt_number"] != expected_attempt:
        raise ContractValidationError("repair attempt_number 必须连续递增")
    if expected_attempt > 1:
        _validate_turn_attempt_replacement(
            conn,
            task_id=task_id,
            value=value,
            previous=prior_turns[-1],
            turn_field="turn_id",
            label="repair",
        )
    conn.execute(
        "INSERT INTO investigation_repair_turns("
        "turn_started_event_id,task_id,batch_id,combination_id,request_event_ref,"
        "repair_request_number,codex_thread_id,scope_id,turn_id,request_digest,"
        "operation_instance_id,attempt_number,attempt_status,logical_operation_instance_id,"
        "previous_attempt_closure_event_ref,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            event_id,
            task_id,
            value["batch_id"],
            value["combination_id"],
            value["request_event_ref"],
            value["repair_request_number"],
            value["codex_thread_id"],
            value["scope_id"],
            value["turn_id"],
            value["request_digest"],
            value["operation_instance_id"],
            value["attempt_number"],
            value["attempt_status"],
            value["logical_operation_instance_id"],
            value.get("previous_attempt_closure_event_ref"),
            created_at,
        ),
    )
    conn.execute(
        "UPDATE investigation_sessions SET phase='repair_pending',"
        "current_repair_turn_event_id=?,current_repair_turn_id=?,"
        "version=version+1,updated_at=? WHERE task_id=? AND batch_id=? AND combination_id=?",
        (
            event_id,
            value["turn_id"],
            created_at,
            task_id,
            value["batch_id"],
            value["combination_id"],
        ),
    )
    conn.execute(
        "UPDATE combinations SET investigation_ready_state='repair_pending',"
        "investigation_stage_state='repair_pending',version=version+1,updated_at=? "
        "WHERE task_id=? AND batch_id=? AND combination_id=?",
        (created_at, task_id, value["batch_id"], value["combination_id"]),
    )


def _validate_repair_turn_start_context(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    value: Mapping[str, Any],
    session: sqlite3.Row,
) -> sqlite3.Row:
    """核对 repair request、预算编号、session 身份和当前活动指针。

    输入是事务连接、task、已规范化的 RepairTurnStarted 正文和对应 session；动作读取
    request 并确认它仍是当前 pending、编号是下一次可消耗预算、provider 身份未漂移且没有
    未消费 turn。返回 request 行；未知/完成 request、预算越界、并发 claim 或身份漂移时
    抛出合同错误，不写数据库。
    """

    request = _validate_repair_turn_request_state(
        conn, task_id=task_id, value=value, session=session
    )
    _validate_repair_turn_session_state(
        conn, task_id=task_id, value=value, session=session
    )
    return request


def _validate_repair_turn_request_state(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    value: Mapping[str, Any],
    session: sqlite3.Row,
) -> sqlite3.Row:
    """核对 repair Started 引用的 pending request 及其编号归属。

    输入是事务连接、task、Started 正文和 session；动作读取 request，确认它仍是当前
    pending、未完成、批次/组合/请求编号逐值一致。返回 request 行；缺失、完成或身份
    漂移时抛出合同错误，不修改数据库。
    """

    if session["pending_repair_event_id"] != value["request_event_ref"]:
        raise ContractValidationError("Repair turn 必须引用当前 pending repair request")
    request = conn.execute(
        "SELECT batch_id,combination_id,repair_count,completion_event_id "
        "FROM investigation_repairs WHERE task_id=? AND repair_request_event_id=?",
        (task_id, value["request_event_ref"]),
    ).fetchone()
    if request is None or request["completion_event_id"] is not None:
        raise ContractValidationError("Repair turn request 不存在或已经完成")
    if (
        request["batch_id"] != value["batch_id"]
        or request["combination_id"] != value["combination_id"]
    ):
        raise ContractValidationError("Repair turn request 与当前组合不一致")
    if request["repair_count"] != value["repair_request_number"]:
        raise ContractValidationError("Repair turn 编号与 request 不一致")
    return request


def _validate_repair_turn_session_state(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    value: Mapping[str, Any],
    session: sqlite3.Row,
) -> None:
    """核对 repair Started 的预算顺序、provider 身份和活动 claim。

    输入是事务连接、task、Started 正文和 session；动作确认请求编号恰为下一个未消耗
    budget、thread/scope 与 session 一致且没有 in-flight repair。返回值为空；越界、身份
    漂移或并发 claim 时抛出合同错误，不修改数据库。
    """

    expected_count = session["repair_count"] + 1
    if value["repair_request_number"] != expected_count or expected_count > 3:
        raise ContractValidationError("只有下一个未消耗的 repair turn 可以启动")
    if (
        session["codex_thread_id"] != value["codex_thread_id"]
        or session["scope_id"] != value["scope_id"]
    ):
        raise ContractValidationError("Repair turn 必须复用 Investigation session 身份")
    if session["current_repair_turn_event_id"] is not None:
        active = conn.execute(
            "SELECT attempt_status FROM investigation_repair_turns WHERE task_id=? "
            "AND turn_started_event_id=?",
            (task_id, session["current_repair_turn_event_id"]),
        ).fetchone()
        if active is None or active["attempt_status"] == "in_flight":
            raise ContractValidationError("当前 repair request 已有未消费的 turn")


def _investigation_session_for_event(
    conn: sqlite3.Connection,
    task_id: str,
    batch_id: str,
    combination_id: str,
) -> sqlite3.Row:
    """读取并确认一个事件引用的当前 session。"""

    row = conn.execute(
        "SELECT * FROM investigation_sessions WHERE task_id=? AND batch_id=? "
        "AND combination_id=?",
        (task_id, batch_id, combination_id),
    ).fetchone()
    if row is None:
        raise ContractValidationError("Investigation 事件引用了不存在的 session")
    return row


def _validate_investigation_result_session_event_reference(
    value: Mapping[str, Any],
    session: sqlite3.Row,
) -> None:
    """核对 Result 可选 session_event_ref 与当前 session 事件身份。

    输入是已通过 Result 正文合同的 mapping 和当前 session 行；动作只在调用方显式
    提供引用时做严格绑定，返回值为空，身份漂移时抛出合同错误，不修改数据库。
    """

    session_event_ref = value.get("session_event_ref")
    if session_event_ref is not None and session_event_ref != session["session_event_id"]:
        raise ContractValidationError(
            "InvestigationResult 的 session_event_ref 与当前 session 不一致"
        )


def _event_payload_by_id(
    conn: sqlite3.Connection,
    task_id: str,
    event_id: str,
    expected_type: str,
) -> Dict[str, Any]:
    """读取同一任务中指定类型的不可变事件正文。"""

    row = conn.execute(
        "SELECT event_type,payload_json FROM memory_events WHERE task_id=? AND event_id=?",
        (task_id, event_id),
    ).fetchone()
    if row is None or row["event_type"] != expected_type:
        raise ContractValidationError(f"来源事件不是 {expected_type}")
    return _load_json(row["payload_json"], {})


def _investigation_expected_combination_count(
    conn: sqlite3.Connection,
    task_id: str,
) -> int:
    """读取 TaskCreated 冻结的 v2 组合数量，旧任务按 top_n 推导。"""

    row = conn.execute(
        "SELECT event_id FROM memory_events "
        "WHERE task_id=? AND event_type='task_created' ORDER BY created_at,event_id LIMIT 1",
        (task_id,),
    ).fetchone()
    if row is None:
        raise ContractValidationError("Investigation task 缺少 TaskCreated 事实")
    created = validate_task_created_payload(
        _event_payload_by_id(conn, task_id, row["event_id"], "task_created")
    )
    if created.get("stage_architecture_version") != "investigation_v2":
        raise ContractValidationError("当前 task 不是 investigation_v2")
    return get_investigation_combination_count(created["task_config"])


def _review_finding_for_combination(
    conn: sqlite3.Connection,
    task_id: str,
    review_event_id: str,
    combination_id: str,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """读取 Review 及指定组合的 finding。"""

    payload = _event_payload_by_id(
        conn,
        task_id,
        review_event_id,
        "investigation_batch_reviewed",
    )
    payload = validate_investigation_batch_review_payload(payload)
    matches = [
        item for item in payload["findings"]
        if item["combination_id"] == combination_id
    ]
    if len(matches) != 1:
        raise ContractValidationError("Review 缺少指定组合的唯一 finding")
    return payload, matches[0]


def _load_result_evidence_verification(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    evidence_ref: str,
) -> sqlite3.Row:
    """读取 Result 引用的不可变证据核验行。

    输入是事务连接、task 身份和证据事件身份；动作只读取同一任务的证据投影，返回
    数据库行。事件不存在时抛出合同错误，避免后续从缺失证据推断 job、operation 或
    artifact 来源。
    """

    evidence = conn.execute(
        "SELECT * FROM investigation_evidence_verifications "
        "WHERE task_id=? AND verification_event_id=?",
        (task_id, evidence_ref),
    ).fetchone()
    if evidence is None:
        raise ContractValidationError("Result 的验证证据事件与当前身份不一致")
    return evidence


def _validate_result_evidence_identity(
    value: Mapping[str, Any],
    evidence: sqlite3.Row,
) -> None:
    """核对证据摘要与 Result 的 batch、组合和会话身份。

    输入是已通过正文合同的 Result 和证据投影行；动作逐字段比较稳定身份，返回值为
    空。任一身份漂移都抛出合同错误，不允许仅凭 event_ref 复用别的计算组合的证据。
    """

    for field in ("batch_id", "combination_id", "codex_thread_id", "scope_id", "turn_id"):
        if evidence[field] != value[field]:
            raise ContractValidationError("Result 的验证证据事件与当前身份不一致")


def _validate_result_evidence_references(
    value: Mapping[str, Any],
    evidence: sqlite3.Row,
) -> None:
    """核对 Result 的 job、operation、artifact 引用完全来自证据摘要。

    输入是 Result 正文和已绑定身份的证据行；动作读取三类摘要并逐类比较排序后的
    引用集合，返回值为空。缺失、增加或替换任一来源时抛出合同错误，防止模型伪造
    计算产物引用。
    """

    evidence_jobs = _load_json(evidence["jobs_json"], [])
    evidence_operations = _load_json(evidence["operations_json"], [])
    evidence_artifacts = _load_json(evidence["artifacts_json"], [])
    expected_refs = {
        "job_refs": sorted(item["job_id"] for item in evidence_jobs),
        "operation_refs": sorted(item["operation_id"] for item in evidence_operations),
        "artifact_refs": sorted(item["artifact_id"] for item in evidence_artifacts),
    }
    for field, references in expected_refs.items():
        if sorted(value[field]) != references:
            raise ContractValidationError(
                "Result 的 job/operation/artifact refs 必须逐项沿用权威证据摘要"
            )


def _validate_v8_result_evidence_source(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    value: Mapping[str, Any],
) -> str:
    """校验 v8 Result 的证据事件身份并返回证据事件引用。

    输入是事务连接、task 身份和 v8 Result；动作要求证据引用存在、身份一致且三类
    计算来源逐项匹配，返回证据事件身份。任何缺失或漂移都抛出合同错误，不修改
    数据库。
    """

    evidence_ref = value.get("evidence_verification_event_ref")
    if not isinstance(evidence_ref, str):
        raise ContractValidationError("v8 Result 必须引用 VerifiedInvestigationEvidence 事件")
    evidence = _load_result_evidence_verification(
        conn,
        task_id=task_id,
        evidence_ref=evidence_ref,
    )
    _validate_result_evidence_identity(value, evidence)
    _validate_result_evidence_references(value, evidence)
    return evidence_ref


def _load_result_source_references(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    event_id: str,
) -> list[str]:
    """读取 Result 不可变事件的 source_refs。

    输入是事务连接、task 和 Result 事件身份；动作只确认事件类型并解析来源数组，
    返回来源事件身份列表。事件缺失或类型漂移时抛出合同错误，不回退到当前索引。
    """

    event = conn.execute(
        "SELECT event_type,source_refs_json FROM memory_events "
        "WHERE task_id=? AND event_id=?",
        (task_id, event_id),
    ).fetchone()
    if event is None or event["event_type"] != "investigation_result_recorded":
        raise ContractValidationError("Result 缺少对应不可变事件")
    return _load_json(event["source_refs_json"], [])


def _validate_result_evidence_reference(
    evidence_ref: str,
    result_refs: Sequence[str],
) -> None:
    """确认 v8 Result 的不可变来源数组包含证据核验事件。

    输入是已校验的证据事件身份和 Result source_refs；动作只做成员检查，返回值为
    空。缺少证据来源时抛出合同错误，阻止脱离证据链的 Result 入库。
    """

    if evidence_ref not in result_refs:
        raise ContractValidationError("Result source_refs 必须引用验证证据事件")


def _validate_result_evidence_source(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    event_id: str,
    value: Mapping[str, Any],
) -> tuple[bool, str | None, list[str]]:
    """核对 Result 事件的证据事件、事件类型和来源引用。"""

    strict_v8 = value.get("investigation_protocol_version") == 8
    evidence_ref = (
        _validate_v8_result_evidence_source(
            conn,
            task_id=task_id,
            value=value,
        )
        if strict_v8
        else None
    )
    result_refs = _load_result_source_references(
        conn,
        task_id=task_id,
        event_id=event_id,
    )
    if strict_v8:
        _validate_result_evidence_reference(evidence_ref, result_refs)
    return strict_v8, evidence_ref, result_refs


def _validate_investigation_initial_result_repair_state(
    *,
    session: sqlite3.Row,
    pending: str | None,
    value: Mapping[str, Any],
    strict_v8: bool,
) -> None:
    """校验首版 Result 没有越过当前返修状态。

    输入是当前 session、pending request、Result 正文和协议标记；动作只检查首版/非首版
    顺序及首版不得引用 repair turn，返回值为空，因果关系错误时抛出合同错误。
    """

    if session["latest_result_event_id"] is not None and pending is None:
        raise ContractValidationError("非首版 Result 必须先有当前 pending repair")
    if pending is None and strict_v8 and value.get("repair_turn_event_ref") is not None:
        raise ContractValidationError("首版 Result 不能引用 repair turn")


def _load_investigation_pending_repair_request(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    pending: str,
) -> sqlite3.Row:
    """读取当前 pending repair 的持久化请求版本。

    输入是事务连接、task 和 pending request 事件身份；动作只读取 request 行，返回
    request 数据库行。请求不存在时抛出合同错误，不修改数据库。
    """

    request = conn.execute(
        "SELECT result_version FROM investigation_repairs "
        "WHERE task_id=? AND repair_request_event_id=?",
        (task_id, pending),
    ).fetchone()
    if request is None:
        raise ContractValidationError("pending repair 缺少持久化 request")
    return request


def _validate_investigation_pending_result_version_and_sources(
    *,
    value: Mapping[str, Any],
    session: sqlite3.Row,
    pending: str,
    request: sqlite3.Row,
    result_refs: Sequence[str],
) -> str:
    """校验返修 Result 使用新版本并保留 request/前版因果来源。

    输入是 Result、当前 session、pending request、request 行和来源列表；动作只做版本
    与引用比较，返回前一版 Result 事件身份。版本重复或来源缺失时抛出合同错误。
    """

    prior_result_ref = session["latest_result_event_id"]
    if not isinstance(prior_result_ref, str):
        raise ContractValidationError("pending repair 缺少可引用的前一版 result")
    if value["result_version"] in {
        session["current_result_version"],
        request["result_version"],
    }:
        raise ContractValidationError("返修 Result 必须是 request 之后的新版本")
    if pending not in result_refs or prior_result_ref not in result_refs:
        raise ContractValidationError("返修 Result 必须同时引用 pending request 和前一版 result")
    return prior_result_ref


def _validate_investigation_repair_turn_for_result(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    value: Mapping[str, Any],
    pending: str,
    result_refs: Sequence[str],
) -> None:
    """校验 v8 返修 Result 引用与 pending request 对应的 repair turn。

    输入是事务连接、task、Result、pending request 和来源列表；动作读取 repair turn
    因果行并逐值核对 request，返回值为空，turn 缺失或漂移时抛出合同错误。
    """

    repair_turn_ref = value.get("repair_turn_event_ref")
    if not isinstance(repair_turn_ref, str) or repair_turn_ref not in result_refs:
        raise ContractValidationError("返修 Result 必须因果引用已启动的 repair turn")
    repair_turn = conn.execute(
        "SELECT request_event_ref FROM investigation_repair_turns "
        "WHERE task_id=? AND turn_started_event_id=?",
        (task_id, repair_turn_ref),
    ).fetchone()
    if repair_turn is None or repair_turn["request_event_ref"] != pending:
        raise ContractValidationError("repair turn 与当前 pending request 不一致")


def _validate_investigation_result_repair_chain(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    value: Mapping[str, Any],
    session: sqlite3.Row,
    strict_v8: bool,
    result_refs: Sequence[str],
) -> str | None:
    """完成首版或返修 Result 的完整因果链校验。

    输入是事务连接、task、Result、当前 session、协议标记和事件来源；动作按首版或
    pending repair 分派专门校验，返回 pending request 身份或 None。任何因果缺失、版本
    重复或 turn 漂移时抛出合同错误，不修改数据库。
    """

    pending = session["pending_repair_event_id"]
    _validate_investigation_initial_result_repair_state(
        session=session,
        pending=pending,
        value=value,
        strict_v8=strict_v8,
    )
    if pending is None:
        return None
    request = _load_investigation_pending_repair_request(
        conn, task_id=task_id, pending=pending
    )
    _validate_investigation_pending_result_version_and_sources(
        value=value,
        session=session,
        pending=pending,
        request=request,
        result_refs=result_refs,
    )
    if strict_v8:
        _validate_investigation_repair_turn_for_result(
            conn,
            task_id=task_id,
            value=value,
            pending=pending,
            result_refs=result_refs,
        )
    return pending


def _validate_result_repair_chain(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    value: Mapping[str, Any],
    session: sqlite3.Row,
    strict_v8: bool,
    result_refs: list[str],
) -> str | None:
    """核对首版/返修 Result 顺序及 pending request、turn 的因果链。"""

    return _validate_investigation_result_repair_chain(
        conn,
        task_id=task_id,
        value=value,
        session=session,
        strict_v8=strict_v8,
        result_refs=result_refs,
    )


def _advance_repair_budget_for_registered_result(
    conn: sqlite3.Connection,
    task_id: str,
    batch_id: str,
    combination_id: str,
    pending_request_ref: str | None,
    session: sqlite3.Row,
    strict_v8: bool,
) -> None:
    """在 v8 返修 Result 已写入后递增一次当前 repair_count。

    输入是 Result 已通过因果校验后的 pending request、session 和协议标记；动作核对
    request 编号仍是下一个预算值，再更新当前 session。返回值为空；request 缺失或
    编号不连续时抛出合同错误，事务调用方会回滚 Result 和预算更新，保证启动后崩溃
    不会提前消耗预算。
    """

    if not strict_v8 or pending_request_ref is None:
        return
    request = conn.execute(
        "SELECT repair_count FROM investigation_repairs "
        "WHERE task_id=? AND repair_request_event_id=?",
        (task_id, pending_request_ref),
    ).fetchone()
    if request is None:
        raise ContractValidationError("返修 Result 缺少当前 repair request")
    expected_count = session["repair_count"] + 1
    if request["repair_count"] != expected_count:
        raise ContractValidationError("返修 Result 的 repair_count 不是下一个预算值")
    conn.execute(
        "UPDATE investigation_sessions SET repair_count=? "
        "WHERE task_id=? AND batch_id=? AND combination_id=?",
        (request["repair_count"], task_id, batch_id, combination_id),
    )


def _mark_investigation_result_attempt_succeeded(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    value: Mapping[str, Any],
    session: sqlite3.Row,
    pending_request_ref: str | None,
    strict_v8: bool,
) -> None:
    """只把本次成功 Result 引用的 current in-flight attempt 标成 succeeded。

    输入是 Result 已通过证据/因果校验的事务资料；动作按首版或 repair 引用读取对应
    attempt，比较真实 turn_id 和 in-flight 状态后更新一行。返回值为空；缺失、漂移或
    已 terminal/succeeded 时抛出合同错误，事务回滚 Result 与索引，不消费其他 attempt。
    """

    if not strict_v8:
        return
    if pending_request_ref is None:
        table = "investigation_turns"
        turn_ref = session["current_turn_event_id"]
        missing_message = "成功 Result 缺少当前 initial attempt"
        terminal_message = "成功 Result 不能消费已终止的 initial attempt"
    else:
        table = "investigation_repair_turns"
        turn_ref = value.get("repair_turn_event_ref")
        missing_message = "成功 Result 缺少当前 repair attempt"
        terminal_message = "成功 Result 不能消费已终止的 repair attempt"
    turn = (
        conn.execute(
            f"SELECT * FROM {table} WHERE task_id=? AND turn_started_event_id=?",
            (task_id, turn_ref),
        ).fetchone()
        if isinstance(turn_ref, str)
        else None
    )
    if turn is None or turn["turn_id"] != value["turn_id"]:
        raise ContractValidationError(missing_message)
    if turn["attempt_status"] != "in_flight":
        raise ContractValidationError(terminal_message)
    conn.execute(
        f"UPDATE {table} SET attempt_status='succeeded' "
        "WHERE task_id=? AND turn_started_event_id=?",
        (task_id, turn_ref),
    )


def insert_investigation_result_projection(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    event_id: str,
    payload: Mapping[str, Any],
    created_at: str,
) -> None:
    """从 ResultRecorded 事件保存科学结果并推进当前 session 版本。

    输入是同一事务中的 Result 正文、事件 ID、task 和时间；动作核对安全证据事件、
    session 身份、首版/返修顺序、pending repair 与因果来源，再写结果历史并更新当前
    result 版本及组合就绪状态。返回值为空；旧版本重复、marker 后写入、证据缺失、
    返修因果不完整或数据库约束失败时抛出合同错误并回滚，不把 transcript/token/path
    写入当前视图。
    """

    value = validate_investigation_result_payload(payload)
    session = _investigation_session_for_event(
        conn, task_id, value["batch_id"], value["combination_id"]
    )
    _validate_investigation_result_session_event_reference(value, session)
    if session["route_marker"] is not None:
        raise ContractValidationError(
            "已写入 revise_hypothesis marker 的组合不能再登记 result"
        )
    if value["codex_thread_id"] != session["codex_thread_id"] or value["scope_id"] != session["scope_id"]:
        raise ContractValidationError("InvestigationResult 与 session 内部身份不一致")
    strict_v8, evidence_ref, result_refs = _validate_result_evidence_source(
        conn,
        task_id=task_id,
        event_id=event_id,
        value=value,
    )
    if strict_v8:
        _validate_investigation_result_turn_claim(
            conn,
            task_id=task_id,
            event_id=event_id,
            value=value,
            session=session,
            result_refs=result_refs,
        )
    duplicate = conn.execute(
        "SELECT result_event_id FROM investigation_results WHERE task_id=? AND batch_id=? "
        "AND combination_id=? AND result_version=?",
        (task_id, value["batch_id"], value["combination_id"], value["result_version"]),
    ).fetchone()
    if duplicate is not None:
        raise ContractValidationError("同一组合不能重复登记相同 result_version")
    pending = _validate_result_repair_chain(
        conn,
        task_id=task_id,
        value=value,
        session=session,
        strict_v8=strict_v8,
        result_refs=result_refs,
    )
    conn.execute(
        "INSERT INTO investigation_results("
        "result_event_id,task_id,batch_id,combination_id,result_version,"
        "codex_thread_id,scope_id,turn_id,self_judgment,narrative,job_refs_json,"
        "operation_refs_json,artifact_refs_json,observations_json,limitations_json,"
        "evidence_verification_event_ref,repair_turn_event_ref,operation_instance_id,created_at) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            event_id,
            task_id,
            value["batch_id"],
            value["combination_id"],
            value["result_version"],
            value["codex_thread_id"],
            value["scope_id"],
            value["turn_id"],
            value["self_judgment"],
            value["narrative"],
            canonical_json(value["job_refs"]),
            canonical_json(value["operation_refs"]),
            canonical_json(value["artifact_refs"]),
            canonical_json(value["observations"]),
            canonical_json(value["limitations"]),
            evidence_ref,
            value.get("repair_turn_event_ref"),
            value.get("operation_instance_id"),
            created_at,
        ),
    )
    _advance_repair_budget_for_registered_result(
        conn,
        task_id,
        value["batch_id"],
        value["combination_id"],
        pending,
        session,
        strict_v8,
    )
    _mark_investigation_result_attempt_succeeded(
        conn,
        task_id=task_id,
        value=value,
        session=session,
        pending_request_ref=pending,
        strict_v8=strict_v8,
    )
    phase = "repair_pending" if pending is not None else "ready_for_batch_validation"
    conn.execute(
        "UPDATE investigation_sessions SET current_result_version=?,"
        "latest_result_event_id=?,phase=?,current_turn_event_id=NULL,current_turn_id=NULL,"
        "current_turn_attempt_number=NULL,current_repair_turn_event_id=NULL,current_repair_turn_id=NULL,"
        "version=version+1,updated_at=? "
        "WHERE task_id=? AND batch_id=? AND combination_id=?",
        (
            value["result_version"],
            event_id,
            phase,
            created_at,
            task_id,
            value["batch_id"],
            value["combination_id"],
        ),
    )
    conn.execute(
        "UPDATE combinations SET investigation_ready_state=?,investigation_stage_state=?,"
        "version=version+1,updated_at=? WHERE task_id=? AND batch_id=? AND combination_id=?",
        (
            phase,
            phase,
            created_at,
            task_id,
            value["batch_id"],
            value["combination_id"],
        ),
    )


def _load_investigation_review_task_turn_rows(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    value: Mapping[str, Any],
) -> tuple[sqlite3.Row, sqlite3.Row, str]:
    """读取 v8 Review 引用的 task ReviewSession 和 Review turn。

    输入是事务连接、task 和 Review 正文；动作只读取正文声明的会话/turn 身份，返回
    两行数据库记录及 turn 引用。任一行不存在时抛出合同错误，不修改数据库。
    """

    session_ref = value.get("task_review_session_event_ref")
    turn_ref = value.get("task_review_turn_event_ref")
    if not isinstance(session_ref, str) or not isinstance(turn_ref, str):
        raise ContractValidationError("v8 Review 必须引用 TaskReviewSession 和 Review turn")
    task_review_session = conn.execute(
        "SELECT * FROM task_review_sessions WHERE task_id=? AND session_event_id=?",
        (task_id, session_ref),
    ).fetchone()
    task_review_turn = conn.execute(
        "SELECT * FROM task_review_turns WHERE task_id=? AND turn_started_event_id=?",
        (task_id, turn_ref),
    ).fetchone()
    if task_review_session is None or task_review_turn is None:
        raise ContractValidationError("Review 引用了不存在的 task Review session/turn")
    return task_review_session, task_review_turn, turn_ref


def _validate_investigation_review_task_turn_identity(
    *,
    value: Mapping[str, Any],
    task_review_session: sqlite3.Row,
    task_review_turn: sqlite3.Row,
    session_ref: str,
    turn_ref: str,
) -> None:
    """核对 Review turn 与冻结 task ReviewSession 的身份绑定。

    输入是 Review 正文、会话/turn 行和两项稳定引用；动作只比较 batch、thread、turn
    和 session 来源，返回值为空，身份漂移时抛出合同错误。
    """

    if (
        task_review_turn["session_event_ref"] != session_ref
        or task_review_turn["batch_id"] != value["batch_id"]
        or task_review_turn["review_thread_id"] != value["review_thread_id"]
        or task_review_turn["review_turn_id"] != value["review_turn_id"]
        or task_review_session["codex_thread_id"] != value["review_thread_id"]
        or task_review_session["active_turn_started_event_id"] != turn_ref
    ):
        raise ContractValidationError("Review thread/turn 与冻结 task Review session 不一致")


def _validate_investigation_review_task_turn_consumable(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    event_id: str,
    task_review_session: sqlite3.Row,
    task_review_turn: sqlite3.Row,
    turn_ref: str,
) -> None:
    """确认 Review turn 仍是当前 in-flight 且没有被其他 Review 消费。

    输入是事务连接、task、当前事件身份、会话/turn 行和 turn 引用；动作只读取阶段与
    历史消费记录，返回值为空，已完成或已消费时抛出合同错误。
    """

    if (
        task_review_session["phase"] != "turn_in_flight"
        or task_review_session["active_turn_started_event_id"] != turn_ref
    ):
        raise ContractValidationError("Review 必须消费当前 in-flight turn")
    consumed = conn.execute(
        "SELECT 1 FROM memory_events WHERE task_id=? "
        "AND event_type='investigation_batch_reviewed' "
        "AND event_id<>? "
        "AND json_extract(payload_json,'$.task_review_turn_event_ref')=? LIMIT 1",
        (task_id, event_id, turn_ref),
    ).fetchone()
    if consumed is not None:
        raise ContractValidationError("Review turn 已被上一条 Review 消费，不能重用")


def _validate_investigation_review_task_turn_projection(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    event_id: str,
    value: Mapping[str, Any],
) -> tuple[bool, sqlite3.Row | None]:
    """核对 v8 Review 对 task ReviewSession/turn 的冻结身份和消费状态。

    输入是事务连接、task、当前 Review 事件身份和正文；动作在 v8 下读取并核对会话、
    turn、thread、batch 及消费记录，返回协议标记和 session 行。缺少或重用 turn 时
    抛出合同错误；legacy 正文返回 False/None 供历史重建使用。
    """

    strict_v8 = value.get("investigation_protocol_version") == 8
    if not strict_v8:
        return False, None
    task_review_session, task_review_turn, turn_ref = _load_investigation_review_task_turn_rows(
        conn, task_id=task_id, value=value
    )
    session_ref = value["task_review_session_event_ref"]
    _validate_investigation_review_task_turn_identity(
        value=value,
        task_review_session=task_review_session,
        task_review_turn=task_review_turn,
        session_ref=session_ref,
        turn_ref=turn_ref,
    )
    _validate_investigation_review_task_turn_consumable(
        conn,
        task_id=task_id,
        event_id=event_id,
        task_review_session=task_review_session,
        task_review_turn=task_review_turn,
        turn_ref=turn_ref,
    )
    return True, task_review_session


def _validate_investigation_review_result_sessions(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    value: Mapping[str, Any],
) -> None:
    """核对 Review 的五个当前 Result、pending repair 和 marker 状态。

    输入是事务连接、task 和规范化 Review 正文；动作逐组合读取 session，返回值为空，
    版本过期、pending repair 或 marker 已存在时抛出合同错误。
    """

    for combination_id, result_version in zip(
        value["combination_ids"], value["result_versions"]
    ):
        session = _investigation_session_for_event(
            conn, task_id, value["batch_id"], combination_id
        )
        if session["current_result_version"] != result_version:
            raise ContractValidationError("Review 使用的 result_version 不是当前版本")
        if session["pending_repair_event_id"] is not None:
            raise ContractValidationError("存在未完成 repair 时不能完成批次 Review")
        if session["route_marker"] is not None:
            raise ContractValidationError("已写入 revise_hypothesis marker 的组合不能再次 Review")


def _validate_investigation_review_projection_history(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    value: Mapping[str, Any],
    idempotency_key: str,
) -> None:
    """核对 Review thread 复用规则和投影幂等键唯一性。

    输入是事务连接、task、Review 正文和幂等键；动作只读取当前 Review 索引，返回值为
    空，thread 漂移或幂等键冲突时抛出合同错误，不修改数据库。
    """

    prior_review = conn.execute(
        "SELECT review_thread_id FROM investigation_reviews WHERE task_id=? "
        "ORDER BY created_at,review_event_id LIMIT 1",
        (task_id,),
    ).fetchone()
    if prior_review is not None and prior_review["review_thread_id"] != value["review_thread_id"]:
        raise ContractValidationError(
            "同一 task 的所有 Investigation Review 必须复用同一 review thread"
        )
    if conn.execute(
        "SELECT 1 FROM investigation_reviews WHERE idempotency_key=?", (idempotency_key,)
    ).fetchone() is not None:
        raise ContractValidationError("Review 幂等键已存在但正文不同")


def _insert_investigation_review_row(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    event_id: str,
    value: Mapping[str, Any],
    created_at: str,
    idempotency_key: str,
) -> None:
    """写入一行 Review 当前索引。

    输入是事务连接、task、Review 事件身份、正文、时间和幂等键；动作保存五项向量、
    finding 与回答字段，返回值为空，数据库约束失败时抛出异常并由外层回滚。
    """

    conn.execute(
        "INSERT INTO investigation_reviews("
        "review_event_id,task_id,batch_id,combination_ids_json,result_versions_json,"
        "result_version_vector_json,findings_json,can_answer,final_answer,"
        "answerability_explanation,review_thread_id,review_turn_id,operation_instance_id,"
        "idempotency_key,created_at) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            event_id,
            task_id,
            value["batch_id"],
            canonical_json(value["combination_ids"]),
            canonical_json(value["result_versions"]),
            canonical_json(value["result_version_vector"]),
            canonical_json(value["findings"]),
            int(value["can_answer"]),
            value["final_answer"],
            value["answerability_explanation"],
            value["review_thread_id"],
            value["review_turn_id"],
            value.get("operation_instance_id"),
            idempotency_key,
            created_at,
        ),
    )


def _update_investigation_review_projection_rows(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    event_id: str,
    value: Mapping[str, Any],
    created_at: str,
    strict_v8: bool,
    task_review_session: Any,
) -> None:
    """推进 task ReviewSession 和五个组合的当前 Review 阶段。

    输入是事务连接、task、Review 事件身份、正文、时间和协议标记；动作更新 task 会话
    及每个组合的 phase/版本，返回值为空，数据库失败时由外层事务回滚。
    """

    if strict_v8:
        turn_ref = value.get("task_review_turn_event_ref")
        if task_review_session is None or not isinstance(turn_ref, str):
            raise ContractValidationError("Review 缺少当前 active turn")
        active_turn = conn.execute(
            "SELECT * FROM task_review_turns WHERE task_id=? AND turn_started_event_id=?",
            (task_id, turn_ref),
        ).fetchone()
        if active_turn is None or active_turn["attempt_status"] != "in_flight":
            raise ContractValidationError("Review 不能消费已终止或已消费的 attempt")
        conn.execute(
            "UPDATE task_review_turns SET attempt_status='succeeded' "
            "WHERE task_id=? AND turn_started_event_id=?",
            (task_id, turn_ref),
        )
        conn.execute(
            "UPDATE task_review_sessions SET turn_count=turn_count+1,last_batch_id=?,"
            "last_turn_id=?,last_turn_started_event_id=?,last_review_event_id=?,"
            "active_batch_id=NULL,active_turn_id=NULL,active_turn_started_event_id=NULL,"
            "phase='ready',version=version+1,updated_at=? WHERE task_id=?",
            (
                value["batch_id"],
                value["review_turn_id"],
                turn_ref,
                event_id,
                created_at,
                task_id,
            ),
        )
    finding_by_combination = {item["combination_id"]: item for item in value["findings"]}
    for combination_id in value["combination_ids"]:
        finding = finding_by_combination[combination_id]
        phase = (
            "validated"
            if finding["validity"] == "pass" and finding["hypothesis_judgment"] == "supported"
            else "ready_for_batch_validation"
        )
        investigation_phase = "validated" if phase == "validated" else "investigating"
        conn.execute(
            "UPDATE investigation_sessions SET latest_review_event_id=?,phase=?,"
            "version=version+1,updated_at=? WHERE task_id=? AND batch_id=? AND combination_id=?",
            (event_id, phase, created_at, task_id, value["batch_id"], combination_id),
        )
        conn.execute(
            "UPDATE combinations SET investigation_phase=?,investigation_ready_state=?,"
            "investigation_stage_state=?,version=version+1,"
            "updated_at=? WHERE task_id=? AND batch_id=? AND combination_id=?",
            (
                investigation_phase,
                phase,
                phase,
                created_at,
                task_id,
                value["batch_id"],
                combination_id,
            ),
        )


def insert_investigation_review_projection(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    event_id: str,
    payload: Mapping[str, Any],
    created_at: str,
    idempotency_key: str,
) -> None:
    """从整批 Review 事件建立可重放索引，并绑定当前五项结果。

    输入是同一事务中的完整五项 Review 正文、事件 ID、task、时间和幂等键；动作核对
    task Review session/turn、冻结 thread、五个当前 result、无 pending repair，并更新
    五个组合的当前阶段。返回值为空；向量过期、线程漂移、marker 后重审或数据库错误
    会回滚事件与当前视图，fail advice 只作为待返修建议保存，不保存 transcript。
    """

    value = validate_investigation_batch_review_payload(payload)
    strict_v8, _task_review_session = _validate_investigation_review_task_turn_projection(
        conn,
        task_id=task_id,
        event_id=event_id,
        value=value,
    )
    _validate_investigation_review_result_sessions(
        conn, task_id=task_id, value=value
    )
    _validate_investigation_review_projection_history(
        conn,
        task_id=task_id,
        value=value,
        idempotency_key=idempotency_key,
    )
    _insert_investigation_review_row(
        conn,
        task_id=task_id,
        event_id=event_id,
        value=value,
        created_at=created_at,
        idempotency_key=idempotency_key,
    )
    _update_investigation_review_projection_rows(
        conn,
        task_id=task_id,
        event_id=event_id,
        value=value,
        created_at=created_at,
        strict_v8=strict_v8,
        task_review_session=_task_review_session,
    )


def insert_investigation_repair_requested_projection(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    event_id: str,
    payload: Mapping[str, Any],
    created_at: str,
) -> None:
    """保存返修请求，并仅在旧协议下兼容旧的预算推进方式。

    v8 请求只登记 Review fail advice 和 pending request；新的返修 Result 成功登记后
    才会递增 session.repair_count。这样 provider 接受 turn 后、Result 写入前崩溃也
    不会消耗预算，旧 v7 历史仍可重建但不会被转换成新的 v8 事实。
    """

    value = validate_investigation_repair_requested_payload(payload)
    strict_v8 = value.get("investigation_protocol_version") == 8
    session = _investigation_session_for_event(conn, task_id, value["batch_id"], value["combination_id"])
    if session["route_marker"] is not None:
        raise ContractValidationError("已写入 revise_hypothesis marker 的组合不能再次 repair")
    review, finding = _review_finding_for_combination(
        conn, task_id, value["review_event_ref"], value["combination_id"]
    )
    if finding["validity"] != "fail" or finding["repair_advice"] != value["repair_advice"]:
        raise ContractValidationError("repair request 必须逐值来自 Review 的 fail advice")
    if value["result_version"] != session["current_result_version"]:
        raise ContractValidationError("repair request 不是当前 result_version")
    review_version = dict(
        zip(review["combination_ids"], review["result_versions"])
    ).get(value["combination_id"])
    if review_version != value["result_version"]:
        raise ContractValidationError(
            "repair request 的 Review 必须包含当前组合的 result 版本"
        )
    if session["latest_review_event_id"] != value["review_event_ref"]:
        raise ContractValidationError("repair request 必须引用组合当前 Review")
    if session["pending_repair_event_id"] is not None:
        raise ContractValidationError("同一组合已有未完成 repair")
    if value["repair_count"] != session["repair_count"] + 1:
        raise ContractValidationError("repair_count 必须连续递增")
    if value["repair_count"] > 3:
        raise ContractValidationError("同一组合最多允许三次 repair")
    conn.execute(
        "INSERT INTO investigation_repairs("
        "repair_request_event_id,task_id,batch_id,combination_id,result_version,"
        "repair_count,repair_advice,review_event_ref,operation_instance_id,created_at) "
        "VALUES(?,?,?,?,?,?,?,?,?,?)",
        (
            event_id,
            task_id,
            value["batch_id"],
            value["combination_id"],
            value["result_version"],
            value["repair_count"],
            value["repair_advice"],
            value["review_event_ref"],
            value["operation_instance_id"],
            created_at,
        ),
    )
    if strict_v8:
        conn.execute(
            "UPDATE investigation_sessions SET pending_repair_event_id=?,"
            "phase='repair_pending',version=version+1,updated_at=? "
            "WHERE task_id=? AND batch_id=? AND combination_id=?",
            (
                event_id,
                created_at,
                task_id,
                value["batch_id"],
                value["combination_id"],
            ),
        )
    else:
        conn.execute(
            "UPDATE investigation_sessions SET repair_count=?,pending_repair_event_id=?,"
            "phase='repair_pending',version=version+1,updated_at=? "
            "WHERE task_id=? AND batch_id=? AND combination_id=?",
            (
                value["repair_count"],
                event_id,
                created_at,
                task_id,
                value["batch_id"],
                value["combination_id"],
            ),
        )
    conn.execute(
        "UPDATE combinations SET investigation_phase='repair_pending',"
        "investigation_ready_state='repair_pending',investigation_stage_state='repair_pending',"
        "version=version+1,"
        "updated_at=? WHERE task_id=? AND batch_id=? AND combination_id=?",
        (created_at, task_id, value["batch_id"], value["combination_id"]),
    )


def _load_investigation_repair_completion_request(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    value: Mapping[str, Any],
) -> sqlite3.Row:
    """读取并核对返修完成正文引用的 request 行。

    输入是事务连接、task 和规范化完成正文；动作检查 request 存在、未完成且批次、
    组合、编号一致，返回 request 行。身份错误或重复完成时抛出合同错误。
    """

    request = conn.execute(
        "SELECT * FROM investigation_repairs WHERE repair_request_event_id=? AND task_id=?",
        (value["request_event_ref"], task_id),
    ).fetchone()
    if request is None:
        raise ContractValidationError("repair completed 找不到 request")
    if request["completion_event_id"] is not None:
        raise ContractValidationError("repair request 已经完成")
    if (
        request["batch_id"] != value["batch_id"]
        or request["combination_id"] != value["combination_id"]
        or request["repair_count"] != value["repair_count"]
    ):
        raise ContractValidationError("repair completed 身份不一致")
    return request


def _load_investigation_repair_completion_result(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    value: Mapping[str, Any],
) -> sqlite3.Row:
    """读取并核对返修完成正文引用的同组合 Result 行。

    输入是事务连接、task 和规范化完成正文；动作检查 Result 事件类型、批次、组合和
    可选版本，返回结果行。缺失、异批或版本漂移时抛出合同错误。
    """

    result = conn.execute(
        "SELECT r.*,e.event_type,e.source_refs_json "
        "FROM investigation_results r JOIN memory_events e ON e.event_id=r.result_event_id "
        "WHERE r.result_event_id=? AND r.task_id=?",
        (value["result_event_ref"], task_id),
    ).fetchone()
    if (
        result is None
        or result["batch_id"] != value["batch_id"]
        or result["combination_id"] != value["combination_id"]
    ):
        raise ContractValidationError("repair completed 必须引用同组合的新 result")
    if "result_version" in value and value["result_version"] != result["result_version"]:
        raise ContractValidationError("repair completed.result_version 与 result 漂移")
    return result


def _validate_investigation_repair_completion_request_causality(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    value: Mapping[str, Any],
    result: sqlite3.Row,
) -> None:
    """核对 request 事件类型、Result 身份和基础来源链。

    输入是事务连接、task、完成正文和 Result 行；动作只读取不可变 request 事件并核对
    Result 来源，返回值为空，来源缺失或身份漂移时抛出合同错误。
    """

    request_event = conn.execute(
        "SELECT event_type FROM memory_events "
        "WHERE task_id=? AND event_id=?",
        (task_id, value["request_event_ref"]),
    ).fetchone()
    if request_event is None or request_event["event_type"] not in {
        "investigation_repair_requested",
        "investigation_task_correction_recorded",
    }:
        raise ContractValidationError("repair completed 引用的 request 不是返修事件")
    if result["result_event_id"] == value["request_event_ref"]:
        raise ContractValidationError("repair completed 必须引用 request 之后的新 result")
    result_refs = _load_json(result["source_refs_json"], [])
    if value["request_event_ref"] not in result_refs:
        raise ContractValidationError("新 result 必须因果引用 request")


def _validate_investigation_repair_completion_turn_causality(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    value: Mapping[str, Any],
    result: sqlite3.Row,
) -> None:
    """核对 v8 repair completed 与结果对应 repair turn 的因果链。

    输入是事务连接、task、完成正文和 Result 行；动作读取 repair turn 并比较 request
    与结果中的 turn 身份，返回值为空，turn 缺失或漂移时抛出合同错误。
    """

    result_refs = _load_json(result["source_refs_json"], [])
    repair_turn_ref = value.get("repair_turn_event_ref")
    if not isinstance(repair_turn_ref, str) or repair_turn_ref not in result_refs:
        raise ContractValidationError("repair completed 必须引用结果对应的 repair turn")
    repair_turn = conn.execute(
        "SELECT request_event_ref FROM investigation_repair_turns "
        "WHERE task_id=? AND turn_started_event_id=?",
        (task_id, repair_turn_ref),
    ).fetchone()
    if repair_turn is None or repair_turn["request_event_ref"] != value["request_event_ref"]:
        raise ContractValidationError("repair completed 的 turn 与 request 不一致")
    if result["repair_turn_event_ref"] != repair_turn_ref:
        raise ContractValidationError("result 与 repair completed 的 turn 身份不一致")


def _validate_investigation_repair_completion_causality(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    value: Mapping[str, Any],
    result: sqlite3.Row,
    strict_v8: bool,
) -> None:
    """核对 request 事件类型、Result 来源和 v8 repair turn 因果链。

    输入是事务连接、task、完成正文、Result 行和协议标记；动作只读取不可变 request
    事件及 repair turn，返回值为空，来源缺失或身份漂移时抛出合同错误。
    """

    _validate_investigation_repair_completion_request_causality(
        conn, task_id=task_id, value=value, result=result
    )
    if strict_v8:
        _validate_investigation_repair_completion_turn_causality(
            conn, task_id=task_id, value=value, result=result
        )


def _validate_investigation_repair_completion_pending_session(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    value: Mapping[str, Any],
) -> None:
    """确认完成事件消费的 request 仍是组合当前 pending request。

    输入是事务连接、task 和规范化完成正文；动作读取当前 session 并比较 pending 事件，
    返回值为空，已被替换或清除时抛出合同错误。
    """

    session = _investigation_session_for_event(
        conn, task_id, value["batch_id"], value["combination_id"]
    )
    if session["pending_repair_event_id"] != value["request_event_ref"]:
        raise ContractValidationError("repair completed 不是当前 pending request")


def _write_investigation_repair_completion_projection(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    event_id: str,
    value: Mapping[str, Any],
    created_at: str,
) -> None:
    """写入返修完成索引并释放组合 pending 状态。

    输入是事务连接、task、完成事件身份、正文和时间；动作更新 repair、session 与组合
    当前阶段，返回值为空，数据库失败时由外层事务回滚。
    """

    conn.execute(
        "UPDATE investigation_repairs SET completion_event_id=?,result_event_ref=?,"
        "completed_at=? WHERE repair_request_event_id=?",
        (event_id, value["result_event_ref"], created_at, value["request_event_ref"]),
    )
    conn.execute(
        "UPDATE investigation_sessions SET pending_repair_event_id=NULL,phase='ready_for_batch_validation',"
        "version=version+1,updated_at=? WHERE task_id=? AND batch_id=? AND combination_id=?",
        (created_at, task_id, value["batch_id"], value["combination_id"]),
    )
    conn.execute(
        "UPDATE combinations SET investigation_phase='investigating',"
        "investigation_ready_state='ready_for_batch_validation',"
        "investigation_stage_state='ready_for_batch_validation',version=version+1,"
        "updated_at=? WHERE task_id=? AND batch_id=? AND combination_id=?",
        (created_at, task_id, value["batch_id"], value["combination_id"]),
    )


def insert_investigation_repair_completed_projection(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    event_id: str,
    payload: Mapping[str, Any],
    created_at: str,
) -> None:
    """完成一条返修请求，核对其结果事件和编号后释放 pending 状态。

    输入是同一事务中的完成正文、事件 ID、task 和时间；动作核对当前 request、同组合
    新 result、request/turn 因果引用和编号，再记录完成事件并清除 pending。返回值为空；
    旧 result、异批结果、时间顺序猜测或因果缺失会抛出合同错误并回滚，不改变预算和
    已有事实。
    """

    value = validate_investigation_repair_completed_payload(payload)
    strict_v8 = value.get("investigation_protocol_version") == 8
    _load_investigation_repair_completion_request(
        conn, task_id=task_id, value=value
    )
    result = _load_investigation_repair_completion_result(
        conn, task_id=task_id, value=value
    )
    _validate_investigation_repair_completion_causality(
        conn,
        task_id=task_id,
        value=value,
        result=result,
        strict_v8=strict_v8,
    )
    _validate_investigation_repair_completion_pending_session(
        conn, task_id=task_id, value=value
    )
    _write_investigation_repair_completion_projection(
        conn,
        task_id=task_id,
        event_id=event_id,
        value=value,
        created_at=created_at,
    )


def insert_investigation_route_marker_projection(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    event_id: str,
    payload: Mapping[str, Any],
    created_at: str,
) -> None:
    """保存 revise_hypothesis marker，并验证它确实由 Review/Result 导出。

    输入是同一事务中的只含 marker 与来源引用的正文、事件 ID、task 和时间；动作核对
    session 当前 result/Review、向量、finding、pending repair 和路由条件，再写入组合
    终态。返回值为空；过期来源、科学理由混入、未满足 gate 或重复 marker 会回滚，
    marker 不替代 Review，也不保存自由文本原因。
    """

    value = validate_investigation_route_marker_payload(payload)
    session = _investigation_session_for_event(
        conn, task_id, value["batch_id"], value["combination_id"]
    )
    result_ref, review_ref = _resolve_route_marker_source_references(value, session)
    gate = evaluate_investigation_batch_route_gate(
        conn,
        task_id=task_id,
        batch_id=value["batch_id"],
    )
    _validate_route_marker_gate_binding(
        gate,
        combination_id=value["combination_id"],
        result_ref=result_ref,
        review_ref=review_ref,
    )
    _reject_duplicate_route_marker(
        conn,
        task_id=task_id,
        batch_id=value["batch_id"],
        combination_id=value["combination_id"],
    )
    _write_route_marker_projection(
        conn,
        task_id=task_id,
        event_id=event_id,
        value=value,
        result_ref=result_ref,
        review_ref=review_ref,
        created_at=created_at,
    )


def _resolve_route_marker_source_references(
    value: Mapping[str, Any],
    session: Mapping[str, Any],
) -> tuple[str, str]:
    """把 marker 正文补齐为当前 session 的 result 和 Review 引用。

    输入是已通过正文合同的 marker 和当前 session；动作使用正文显式引用或 session
    当前引用，并拒绝缺失、非字符串和过期来源。返回值是稳定的 result/Review 引用；
    失败时抛出合同错误，不修改数据库。
    """

    result_ref = value.get("result_event_ref") or session["latest_result_event_id"]
    review_ref = value.get("review_event_ref") or session["latest_review_event_id"]
    if not isinstance(result_ref, str) or not isinstance(review_ref, str):
        raise ContractValidationError("route marker 必须同时解析到 result 和 review 事实")
    if (
        result_ref != session["latest_result_event_id"]
        or review_ref != session["latest_review_event_id"]
    ):
        raise ContractValidationError("route marker 必须在写入时绑定当前 result 和 Review")
    return result_ref, review_ref


def _is_supported_finding_in_accepted_mixed_gate(
    gate: Mapping[str, Any],
    finding: Mapping[str, Any],
) -> bool:
    """判断当前 finding 是否属于 accepted 混合 Review 中的 supported 项。

    输入是整批 route gate 和当前组合 finding；动作只判断 accepted 混合 Review 的
    路由边界，返回值为是否应该拒绝该组合 marker，不修改数据库；输入字段缺失时由
    调用方的正文合同负责拦截，本函数不抛出数据库异常。
    """

    findings = gate.get("findings")
    return bool(findings) and all(
        item["validity"] == "pass" for item in findings
    ) and finding["hypothesis_judgment"] == "supported"


def _validate_route_marker_gate_binding(
    gate: Mapping[str, Any],
    *,
    combination_id: str,
    result_ref: str,
    review_ref: str,
) -> Mapping[str, Any]:
    """核对单个 marker 是否绑定整批 gate 的当前来源和合法 finding。

    输入是批次 gate、组合身份及当前 result/Review 引用；动作检查 gate 状态、完整
    来源和 accepted 混合路由边界。返回值是当前 finding；任何不一致都会抛出合同
    错误，不写入数据库。
    """

    if not gate["allowed"]:
        raise ContractValidationError(
            gate.get("reason") or "当前 Review 尚未满足 revise_hypothesis marker gate"
        )
    if (
        gate["review_event_ref"] != review_ref
        or gate["result_event_refs"].get(combination_id) != result_ref
    ):
        raise ContractValidationError("route marker 必须引用当前完整 Review 和对应 result")
    finding = next(
        (item for item in gate["findings"] if item["combination_id"] == combination_id),
        None,
    )
    if finding is None:
        raise ContractValidationError("route marker 缺少当前组合 finding")
    # accepted 混合 Review 只给 refuted 组合保留 marker；耗尽 fail 的混合批次不走此
    # 分支，因此 supported finding 仍会保留，供下一批读取真实经验。
    if _is_supported_finding_in_accepted_mixed_gate(gate, finding):
        raise ContractValidationError("accepted 混合 Review 只能给 pass+refuted 组合写 marker")
    return finding


def _reject_duplicate_route_marker(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    batch_id: str,
    combination_id: str,
) -> None:
    """拒绝为同一 task/batch/combination 重复写入 route marker。

    输入是事务连接和组合身份；动作只查询 marker 索引，不修改数据库，返回值为空。
    已存在 marker 时抛出合同错误，查询失败则把数据库异常交给外层事务处理。
    """

    if conn.execute(
        "SELECT 1 FROM investigation_route_markers WHERE task_id=? AND batch_id=? AND combination_id=?",
        (task_id, batch_id, combination_id),
    ).fetchone() is not None:
        raise ContractValidationError("同一组合只能有一个 revise_hypothesis marker")


def _write_route_marker_projection(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    event_id: str,
    value: Mapping[str, Any],
    result_ref: str,
    review_ref: str,
    created_at: str,
) -> None:
    """写入 marker，并把组合 session 与当前索引同步到 hypothesis revision 阶段。

    输入是事务连接、事件身份、已校验正文、当前 result/Review 引用和时间；动作写入
    marker 并更新两张当前索引表，返回值为空，SQL 失败由外层事务回滚且不伪造事实。
    """

    conn.execute(
        "INSERT INTO investigation_route_markers("
        "marker_event_id,task_id,batch_id,combination_id,marker,result_event_ref,review_event_ref,created_at) "
        "VALUES(?,?,?,?,?,?,?,?)",
        (
            event_id,
            task_id,
            value["batch_id"],
            value["combination_id"],
            value["marker"],
            result_ref,
            review_ref,
            created_at,
        ),
    )
    conn.execute(
        "UPDATE investigation_sessions SET route_marker='revise_hypothesis',phase='marked_for_hypothesis_revision',"
        "version=version+1,updated_at=? WHERE task_id=? AND batch_id=? AND combination_id=?",
        (created_at, task_id, value["batch_id"], value["combination_id"]),
    )
    conn.execute(
        "UPDATE combinations SET investigation_phase='marked_for_hypothesis_revision',"
        "investigation_ready_state='marked_for_hypothesis_revision',"
        "investigation_stage_state='marked_for_hypothesis_revision',"
        "version=version+1,updated_at=? WHERE task_id=? AND batch_id=? AND combination_id=?",
        (created_at, task_id, value["batch_id"], value["combination_id"]),
    )


def _selected_investigation_route_combination_ids(
    conn: sqlite3.Connection,
    task_id: str,
    batch_id: str,
) -> List[str]:
    """读取当前批次按稳定顺序选中的组合身份。"""

    rows = conn.execute(
        "SELECT c.combination_id FROM combinations c "
        "JOIN hypothesis_candidates hc ON hc.batch_id=c.batch_id "
        "AND hc.combination_id=c.combination_id "
        "WHERE c.task_id=? AND c.batch_id=? AND hc.selection_status='selected' "
        "ORDER BY c.combination_index,c.combination_id",
        (task_id, batch_id),
    ).fetchall()
    return [row["combination_id"] for row in rows]


def _investigation_route_gate_response(
    task_id: str,
    batch_id: str,
    combination_ids: Sequence[str],
) -> Dict[str, Any]:
    """创建整批 route gate 的稳定返回骨架。"""

    return {
        "allowed": False,
        "task_id": task_id,
        "batch_id": batch_id,
        "combination_ids": list(combination_ids),
        "review_event_ref": None,
        "result_version_vector": [],
        "result_event_refs": {},
        "findings": [],
        "reason": None,
    }


def _read_investigation_route_gate_sessions(
    conn: sqlite3.Connection,
    task_id: str,
    batch_id: str,
) -> List[sqlite3.Row]:
    """读取整批 route gate 所需的当前 session 字段。"""

    return conn.execute(
        "SELECT combination_id,current_result_version,latest_result_event_id,"
        "latest_review_event_id,pending_repair_event_id,repair_count "
        "FROM investigation_sessions WHERE task_id=? AND batch_id=? "
        "ORDER BY combination_id",
        (task_id, batch_id),
    ).fetchall()


def _validate_investigation_route_gate_sessions(
    response: Dict[str, Any],
    combination_ids: Sequence[str],
    sessions: Sequence[sqlite3.Row],
) -> Dict[str, sqlite3.Row] | None:
    """核对整批 session 数量、身份、当前结果和 Review 引用。"""

    session_by_id = {row["combination_id"]: row for row in sessions}
    expected = set(combination_ids)
    if (
        not combination_ids
        or len(expected) != len(combination_ids)
        or len(sessions) != len(combination_ids)
        or set(session_by_id) != expected
    ):
        response["reason"] = "route marker 必须绑定全部完整 session"
        return None
    if any(
        not isinstance(row["current_result_version"], str)
        or not isinstance(row["latest_result_event_id"], str)
        or not isinstance(row["latest_review_event_id"], str)
        for row in sessions
    ):
        response["reason"] = "marker session 必须各有当前 result 和 Review"
        return None
    review_refs = {row["latest_review_event_id"] for row in sessions}
    if len(review_refs) != 1:
        response["reason"] = "五个 session 必须引用同一当前 Review"
        return None
    response["review_event_ref"] = next(iter(review_refs))
    response["result_version_vector"] = sorted(
        [[row["combination_id"], row["current_result_version"]] for row in sessions]
    )
    response["result_event_refs"] = {
        row["combination_id"]: row["latest_result_event_id"] for row in sessions
    }
    return session_by_id


def _validate_investigation_route_gate_results(
    conn: sqlite3.Connection,
    task_id: str,
    batch_id: str,
    result_event_refs: Mapping[str, str],
    result_version_vector: Sequence[Sequence[str]],
) -> bool:
    """核对五项当前结果事件与 session 版本向量逐项一致。"""

    for combination_id, result_event_ref in result_event_refs.items():
        result = conn.execute(
            "SELECT batch_id,combination_id,result_version FROM investigation_results "
            "WHERE task_id=? AND result_event_id=?",
            (task_id, result_event_ref),
        ).fetchone()
        expected_version = dict(result_version_vector).get(combination_id)
        if (
            result is None
            or result["batch_id"] != batch_id
            or result["combination_id"] != combination_id
            or result["result_version"] != expected_version
        ):
            return False
    return True


def _load_investigation_route_gate_review(
    conn: sqlite3.Connection,
    task_id: str,
    batch_id: str,
    review_event_ref: str,
) -> Dict[str, Any]:
    """读取并校验整批 route gate 绑定的不可变 Review。"""

    review = validate_investigation_batch_review_payload(
        _event_payload_by_id(
            conn,
            task_id,
            review_event_ref,
            "investigation_batch_reviewed",
        )
    )
    if review["batch_id"] != batch_id:
        raise ContractValidationError("route marker Review 不属于当前 batch")
    return review


def _reject_investigation_route_gate_terminal_review(
    response: Dict[str, Any],
    review: Mapping[str, Any],
) -> bool:
    """拒绝已表达 accept 或 answerability explanation 的 Review。"""

    if (
        bool(review["can_answer"])
        or review.get("final_answer") is not None
        or review.get("answerability_explanation") is not None
    ):
        response["reason"] = "已表达 accept 或 answerability explanation 的 Review 禁止重生"
        return True
    return False


def _validate_investigation_route_gate_failures(
    response: Dict[str, Any],
    fail_ids: Sequence[str],
    session_by_id: Mapping[str, sqlite3.Row],
) -> bool:
    """核对含 fail 的整批没有 pending 且每个 fail 都耗尽返修。"""

    pending = [
        combination_id
        for combination_id, session in session_by_id.items()
        if session["pending_repair_event_id"] is not None
    ]
    exhausted = [
        combination_id
        for combination_id in fail_ids
        if session_by_id[combination_id]["repair_count"] == 3
    ]
    if pending:
        response["reason"] = "存在 pending repair 时不能写整批 route marker"
        return False
    if set(exhausted) != set(fail_ids):
        response["reason"] = "每个 fail finding 都必须消耗三次 repair 后才能写 marker"
        return False
    return True


def _validate_investigation_route_gate_findings(
    response: Dict[str, Any],
    review: Mapping[str, Any],
    combination_ids: Sequence[str],
    session_by_id: Mapping[str, sqlite3.Row],
) -> bool:
    """按整批规则判断 finding、终态回答字段和返修预算。

    输入是 route gate 返回骨架、当前 Review、五个组合身份和 session 索引；动作按
    完整性、终态、pending 与返修预算依次校验，并复制合法 finding。返回值表示是否
    允许写 marker；失败只写入 reason，不修改数据库。
    """

    finding_by_id = _index_route_gate_findings(
        response, review, combination_ids
    )
    if finding_by_id is None:
        return False
    findings = list(finding_by_id.values())
    accepted_mixed_refutation = _is_accepted_mixed_route_gate(review, findings)
    if not _validate_route_gate_terminal_branch(
        response, review, accepted_mixed_refutation
    ):
        return False
    if not _validate_route_gate_pending_repairs(response, session_by_id):
        return False
    fail_ids = _route_gate_failure_ids(finding_by_id)
    if fail_ids and not _validate_investigation_route_gate_failures(
        response, fail_ids, session_by_id
    ):
        return False
    if not fail_ids and not _validate_route_gate_nonfailure_findings(
        response, findings, accepted_mixed_refutation
    ):
        return False
    response["findings"] = [dict(item) for item in findings]
    return True


def _index_route_gate_findings(
    response: Dict[str, Any],
    review: Mapping[str, Any],
    combination_ids: Sequence[str],
) -> dict[str, Mapping[str, Any]] | None:
    """核对 Review 是否恰好覆盖当前五个组合，并按组合建立 finding 索引。

    输入是 route gate 返回骨架、Review 和组合身份；动作只在 response 中记录失败原因，
    返回 finding 索引或 ``None``，不修改数据库；数量或集合漂移不会抛出原始异常。
    """

    findings = list(review["findings"])
    finding_by_id = {item["combination_id"]: item for item in findings}
    if len(findings) != len(combination_ids) or set(finding_by_id) != set(combination_ids):
        response["reason"] = "当前 Review 必须完整覆盖全部组合"
        return None
    return finding_by_id


def _is_accepted_mixed_route_gate(
    review: Mapping[str, Any],
    findings: Sequence[Mapping[str, Any]],
) -> bool:
    """判断 Review 是否为可回答但含 refuted finding 的混合终态。

    输入是 Review 和已按组合整理的 finding；动作只计算路由条件，返回布尔值，不写
    数据库；调用方合同缺字段时本函数可能传播键访问错误。
    """

    return bool(review["can_answer"]) and all(
        finding["validity"] == "pass" for finding in findings
    ) and any(
        finding["hypothesis_judgment"] == "refuted" for finding in findings
    )


def _validate_route_gate_terminal_branch(
    response: Dict[str, Any],
    review: Mapping[str, Any],
    accepted_mixed_refutation: bool,
) -> bool:
    """处理 accepted 或 answerability explanation 对 route gate 的终态阻断。

    输入是 response、Review 和 accepted mixed 判断；动作写入失败原因或清除混合终态的
    暂时原因，返回是否继续检查，不修改数据库；非法终态只返回 ``False``。
    """

    if not _reject_investigation_route_gate_terminal_review(response, review):
        return True
    if not accepted_mixed_refutation:
        return False
    # accepted mixed 仍需写 TaskClosed；这里清除终态拒绝原因，只允许其 refuted 项
    # 写 marker，不能把该原因泄漏为下一批 gate 的失败原因。
    response["reason"] = None
    return True


def _validate_route_gate_pending_repairs(
    response: Dict[str, Any],
    session_by_id: Mapping[str, sqlite3.Row],
) -> bool:
    """拒绝仍有 pending repair 的整批 marker 路由。

    输入是 response 和当前 session 索引；动作在发现 pending 时写入路由原因，返回是否
    可以继续，不修改数据库；索引字段缺失时传播调用方的合同错误。
    """

    if any(
        session["pending_repair_event_id"] is not None
        for session in session_by_id.values()
    ):
        response["reason"] = "存在 pending repair 时不能写整批 route marker"
        return False
    return True


def _route_gate_failure_ids(
    finding_by_id: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """提取当前 Review 中需要耗尽返修预算的 fail 组合。

    输入是按组合索引的 finding；动作只筛选 validity 为 ``fail`` 的身份，返回稳定列表，
    不修改输入或数据库；字段缺失时传播调用方合同错误。
    """

    return [
        combination_id
        for combination_id, finding in finding_by_id.items()
        if finding["validity"] == "fail"
    ]


def _is_pass_refuted_finding(finding: Mapping[str, Any]) -> bool:
    """判断 finding 是否是无 fail 路由允许的 pass+refuted 事实。

    输入是单个 finding；动作只读取两个判断字段，返回布尔值，不产生副作用；字段缺失
    时传播调用方的合同错误。
    """

    return (
        finding["validity"] == "pass"
        and finding["hypothesis_judgment"] == "refuted"
    )


def _validate_route_gate_nonfailure_findings(
    response: Dict[str, Any],
    findings: Sequence[Mapping[str, Any]],
    accepted_mixed_refutation: bool,
) -> bool:
    """核对没有 fail 时是否全部为 pass+refuted，或已被 accepted mixed 允许。

    输入是 response、finding 列表和混合终态判断；动作在失败时记录原因，返回是否通过，
    不修改数据库；finding 字段缺失时传播调用方合同错误。
    """

    if not accepted_mixed_refutation and any(
        not _is_pass_refuted_finding(finding) for finding in findings
    ):
        response["reason"] = "无 fail 时只能由五项 pass+refuted Review 触发 marker"
        return False
    return True


def evaluate_investigation_batch_route_gate(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    batch_id: str,
) -> Dict[str, Any]:
    """只读判断整批五项 revise_hypothesis marker 是否具备写入资格。

    输入是同一事务中的连接、任务和批次身份；动作核对恰好五个选中组合、五个当前
    session、同一 Review、完整且精确的当前 Result 版本向量，以及终态/返修条件；返回
    可供 marker 写入和下一批 gate 复用的结果，不修改数据库。支持与耗尽 fail 混合时
    marker 仍只是路由事实，不把 supported finding 改写成 refuted。
    """

    expected_count = _investigation_expected_combination_count(conn, task_id)
    combination_ids = _selected_investigation_route_combination_ids(
        conn, task_id, batch_id
    )
    response = _investigation_route_gate_response(task_id, batch_id, combination_ids)
    if len(combination_ids) != expected_count or len(set(combination_ids)) != expected_count:
        response["reason"] = "当前批次必须恰好覆盖 TaskCreated 冻结的全部组合"
        return response
    sessions = _read_investigation_route_gate_sessions(conn, task_id, batch_id)
    session_by_id = _validate_investigation_route_gate_sessions(
        response, combination_ids, sessions
    )
    if session_by_id is None:
        return response
    if not _validate_investigation_route_gate_results(
        conn,
        task_id,
        batch_id,
        response["result_event_refs"],
        response["result_version_vector"],
    ):
        response["reason"] = "marker 必须引用当前完整 Result 向量"
        return response
    review = _load_investigation_route_gate_review(
        conn, task_id, batch_id, response["review_event_ref"]
    )
    if sorted(review["combination_ids"]) != sorted(combination_ids):
        response["reason"] = "marker Review 必须覆盖当前全部入选组合"
        return response
    if review["result_version_vector"] != response["result_version_vector"]:
        response["reason"] = "marker Review 必须精确引用当前 result_version vector"
        return response
    _validate_investigation_route_gate_findings(
        response, review, combination_ids, session_by_id
    )
    if response["reason"] is not None:
        return response
    response["reason"] = None
    response["allowed"] = True
    return response


def _validate_terminal_task_identity_and_uniqueness(
    conn: sqlite3.Connection,
    task_id: str,
    value: Mapping[str, Any],
) -> None:
    """核对终态正文的 task 身份和唯一终态约束。"""

    if value["task_id"] != task_id:
        raise ContractValidationError("TaskClosed task 身份不一致")
    if conn.execute(
        "SELECT 1 FROM investigation_task_terminals WHERE task_id=?", (task_id,)
    ).fetchone() is not None:
        raise ContractValidationError("Investigation task 只能写一个终态")


def _load_and_validate_terminal_review(
    conn: sqlite3.Connection,
    task_id: str,
    value: Mapping[str, Any],
) -> Dict[str, Any]:
    """读取并核对终态引用的当前批次 Review 和回答字段。"""

    review = validate_investigation_batch_review_payload(
        _event_payload_by_id(
            conn,
            task_id,
            value["review_event_ref"],
            "investigation_batch_reviewed",
        )
    )
    if review["batch_id"] != value["batch_id"]:
        raise ContractValidationError("TaskClosed Review 不属于当前 batch")
    if bool(review["can_answer"]) != value["can_answer"]:
        raise ContractValidationError("TaskClosed answerability 与 Review 不一致")
    if value["can_answer"]:
        if review.get("final_answer") != value["final_answer"]:
            raise ContractValidationError("TaskClosed final_answer 必须逐值沿用当前 Review")
    elif (
        review.get("answerability_explanation") is not None
        and review.get("answerability_explanation")
        != value["answerability_explanation"]
    ):
        raise ContractValidationError(
            "TaskClosed answerability_explanation 不能漂移当前 Review"
        )
    return review


def _load_terminal_session_rows(
    conn: sqlite3.Connection,
    task_id: str,
    batch_id: str,
) -> List[sqlite3.Row]:
    """读取终态核对所需的当前批次组合 session。"""

    return conn.execute(
        "SELECT combination_id,latest_result_event_id,latest_review_event_id,"
        "current_result_version,pending_repair_event_id,route_marker,phase "
        "FROM investigation_sessions WHERE task_id=? AND batch_id=?",
        (task_id, batch_id),
    ).fetchall()


def _validate_terminal_session_shape(
    sessions: Sequence[sqlite3.Row],
) -> None:
    """核对终态必须有一个或五个已产生结果且没有 pending repair 的 session。

    输入是当前批次的 session 行；动作只检查结果存在和返修是否已收束，返回值为空。
    数量不足、额外组合、缺少结果或仍有 pending request 时抛出合同错误。
    """

    if len(sessions) not in {1, 5}:
        raise ContractValidationError("TaskClosed 必须绑定一个或五个无 pending 的当前 session")
    for session in sessions:
        if session["latest_result_event_id"] is None:
            raise ContractValidationError("TaskClosed 必须绑定当前批次无 pending 的 session")
        if session["pending_repair_event_id"] is not None:
            raise ContractValidationError("TaskClosed 必须绑定当前批次无 pending 的 session")


def _validate_terminal_session_combination_ids(
    review: Mapping[str, Any],
    sessions: Sequence[sqlite3.Row],
) -> set[str]:
    """核对 Review 的组合集合与当前 session 完全一致。

    输入是当前 Review 和 session 行；动作返回 Review 声明的组合身份集合。集合不一
    致时抛出合同错误，防止终态把别的批次或未调查组合混入结果。
    """

    expected_ids = set(review["combination_ids"])
    if {row["combination_id"] for row in sessions} != expected_ids:
        raise ContractValidationError("TaskClosed Review 未覆盖五个当前组合")
    return expected_ids


def _validate_terminal_session_review_references(
    value: Mapping[str, Any],
    sessions: Sequence[sqlite3.Row],
) -> None:
    """核对当前 session 都指向终态使用的 Review。

    输入是终态正文和 session 行；动作逐项比较 latest_review_event_id，返回值为空。
    任一组合仍指向旧 Review 时抛出合同错误。
    """

    for session in sessions:
        if session["latest_review_event_id"] != value["review_event_ref"]:
            raise ContractValidationError("TaskClosed Review 必须是五个组合的当前 Review")


def _validate_terminal_result_version_vector(
    review: Mapping[str, Any],
    sessions: Sequence[sqlite3.Row],
) -> None:
    """核对 Review 的 result_version vector 与五个 session 当前版本相同。

    输入是当前 Review 和 session 行；动作按组合排序构造实际版本向量，返回值为空。
    任一版本过期、错位或缺失时抛出合同错误，不允许终态复用旧结果。
    """

    current_vector = sorted(
        [row["combination_id"], row["current_result_version"]]
        for row in sessions
    )
    if review["result_version_vector"] != current_vector:
        raise ContractValidationError(
            "TaskClosed Review vector 必须逐项等于五个当前 result 版本"
        )


def _validate_terminal_session_consistency(
    value: Mapping[str, Any],
    review: Mapping[str, Any],
    sessions: Sequence[sqlite3.Row],
) -> set[str]:
    """核对五个 session 的当前 result、Review、pending 和版本向量。"""

    _validate_terminal_session_shape(sessions)
    expected_ids = _validate_terminal_session_combination_ids(review, sessions)
    _validate_terminal_session_review_references(value, sessions)
    _validate_terminal_result_version_vector(review, sessions)
    return expected_ids


def _load_terminal_marker_rows(
    conn: sqlite3.Connection,
    task_id: str,
    batch_id: str,
) -> List[sqlite3.Row]:
    """读取当前批次所有 revise_hypothesis marker 来源。"""

    return conn.execute(
        "SELECT combination_id,marker_event_id,result_event_ref,review_event_ref "
        "FROM investigation_route_markers WHERE task_id=? AND batch_id=?",
        (task_id, batch_id),
    ).fetchall()


def _validate_terminal_marker_references(
    value: Mapping[str, Any],
    markers: Sequence[sqlite3.Row],
) -> None:
    """核对终态正文声明的 marker 数量和来源集合。

    输入是已通过正文合同的终态值和当前批次 marker 行；动作按 accepted 或假设重生
    路径分派来源集合校验，返回值为空且不修改数据库；数量、声明或来源不一致时抛出
    合同错误，阻止错误路由事实进入终态索引。
    """

    if value["accepted"]:
        _validate_accepted_terminal_marker_references(value, markers)
        return
    _validate_nonaccepted_terminal_marker_references(value, markers)


def _validate_accepted_terminal_marker_references(
    value: Mapping[str, Any],
    markers: Sequence[sqlite3.Row],
) -> None:
    """核对 accepted 终态显式引用的部分 refuted marker 来源。

    输入是终态正文和当前 marker 行；动作比较来源集合，返回值为空，不修改数据库；缺少
    引用、集合漂移时抛出合同错误，查询数据的数据库异常由调用方处理。
    """

    marker_refs = value["marker_event_refs"]
    if not marker_refs and markers:
        raise ContractValidationError(
            "accepted TaskClosed 必须明确引用已写入的 refuted marker"
        )
    if marker_refs and {row["marker_event_id"] for row in markers} != set(marker_refs):
        raise ContractValidationError("accepted TaskClosed marker 来源与当前索引不一致")


def _validate_nonaccepted_terminal_marker_references(
    value: Mapping[str, Any],
    markers: Sequence[sqlite3.Row],
) -> None:
    """核对非 accepted 终态必须引用完整当前批次 marker，或完全没有 marker。

    输入是终态正文和当前 marker 行；动作按是否有引用分派完整性校验，返回值为空，不
    修改数据库；部分引用或残留 marker 未被正文声明时抛出合同错误。
    """

    marker_refs = value["marker_event_refs"]
    if marker_refs:
        _validate_complete_terminal_marker_references(marker_refs, markers)
        return
    if markers:
        raise ContractValidationError(
            "当前批次已有 marker 时，终态必须明确引用完整 marker"
        )


def _validate_complete_terminal_marker_references(
    marker_refs: Sequence[str],
    markers: Sequence[sqlite3.Row],
) -> None:
    """核对非 accepted 终态的 marker 引用集合恰好对应当前索引。

    输入是正文 marker 引用和索引行；动作检查数量及集合，返回值为空，不修改数据库；
    数量不足或来源漂移时抛出合同错误。
    """

    if len(marker_refs) not in {1, 5} or len(markers) != len(marker_refs):
        raise ContractValidationError(
            "stopped_no_supported 或 regeneration exhausted 必须完整引用当前 marker"
        )
    if {row["marker_event_id"] for row in markers} != set(marker_refs):
        raise ContractValidationError("TaskClosed marker 来源与当前索引不一致")


def _validate_terminal_marker_session_phases(
    sessions: Sequence[sqlite3.Row],
) -> None:
    """核对带 route marker 的 session 已进入 hypothesis revision 阶段。

    输入是当前批次 session 行；动作逐项检查 route_marker 与 phase 的绑定，返回值为
    空。阶段漂移时抛出合同错误，避免终态把未完成的调查当成可重生经验。
    """

    for session in sessions:
        if session["route_marker"] is None:
            continue
        if session["phase"] != "marked_for_hypothesis_revision":
            raise ContractValidationError("TaskClosed marker session 阶段不一致")


def _index_terminal_markers_by_combination(
    expected_ids: set[str],
    markers: Sequence[sqlite3.Row],
) -> dict[str, sqlite3.Row]:
    """建立并核对终态 marker 的组合索引。

    输入是 Review 的当前组合集合和 marker 行；动作返回按 combination_id 建立的索引。
    数量或集合不完整时抛出合同错误，拒绝只引用部分 revise_hypothesis 记录。
    """

    marker_by_combination = {row["combination_id"]: row for row in markers}
    if len(markers) not in {1, 5} or set(marker_by_combination) != expected_ids:
        raise ContractValidationError(
            "stopped_no_supported 或 regeneration exhausted 必须覆盖当前组合"
        )
    return marker_by_combination


def _validate_terminal_marker_result_bindings(
    value: Mapping[str, Any],
    sessions: Sequence[sqlite3.Row],
    marker_by_combination: Mapping[str, sqlite3.Row],
) -> None:
    """核对每个 marker 都绑定当前 Result 和当前 Review。

    输入是终态正文、session 行和按组合索引的 marker；动作逐项核对来源引用，返回值
    为空。缺失 marker、旧 Result 或旧 Review 时抛出合同错误，防止错误经验进入下一批。
    """

    for session in sessions:
        _validate_terminal_marker_binding_for_session(
            value,
            session,
            marker_by_combination.get(session["combination_id"]),
        )


def _validate_terminal_marker_binding_for_session(
    value: Mapping[str, Any],
    session: sqlite3.Row,
    marker: sqlite3.Row | None,
) -> None:
    """核对一条 marker 与组合当前 result、Review 的绑定。

    输入是终态正文、session 和按组合找到的 marker；动作比较当前来源，返回值为空，不
    修改数据库；缺失 marker、旧 result 或旧 Review 时抛出合同错误。
    """

    if marker is None:
        raise ContractValidationError("TaskClosed marker 必须绑定五个当前 result 和 Review")
    if (
        marker["result_event_ref"] != session["latest_result_event_id"]
        or marker["review_event_ref"] != value["review_event_ref"]
    ):
        raise ContractValidationError("TaskClosed marker 必须绑定五个当前 result 和 Review")


def _validate_terminal_marker_session_bindings(
    value: Mapping[str, Any],
    expected_ids: set[str],
    review: Mapping[str, Any],
    sessions: Sequence[sqlite3.Row],
    markers: Sequence[sqlite3.Row],
) -> None:
    """核对 marker 对应的 session 阶段、组合集合和当前来源。"""

    _validate_terminal_marker_session_phases(sessions)
    if value["accepted"]:
        _validate_accepted_terminal_marker_bindings(value, review, sessions, markers)
        return
    if not markers:
        return
    marker_by_combination = _index_terminal_markers_by_combination(expected_ids, markers)
    _validate_terminal_marker_result_bindings(value, sessions, marker_by_combination)


def _is_pass_refuted_terminal_finding(finding: Mapping[str, Any]) -> bool:
    """判断 accepted 终态中允许保存 marker 的 finding 类型。

    输入是单个 Review finding；动作只读取 validity 和 hypothesis judgment，返回布尔值，
    不修改输入或数据库；字段缺失时传播调用方合同错误。
    """

    return (
        finding["validity"] == "pass"
        and finding["hypothesis_judgment"] == "refuted"
    )


def _validate_accepted_terminal_marker_bindings(
    value: Mapping[str, Any],
    review: Mapping[str, Any],
    sessions: Sequence[sqlite3.Row],
    markers: Sequence[sqlite3.Row],
) -> None:
    """核对 accepted 终态只保存每个 pass+refuted 组合的当前 marker。

    输入是终态正文、Review、五个 session 和 marker 行；动作建立临时索引并逐条校验，
    返回值为空，不修改数据库；集合或来源不一致时抛出合同错误。
    """

    session_by_id = {row["combination_id"]: row for row in sessions}
    finding_by_id = {item["combination_id"]: item for item in review["findings"]}
    expected_refuted_ids = {
        item["combination_id"]
        for item in review["findings"]
        if _is_pass_refuted_terminal_finding(item)
    }
    marker_ids = {marker["combination_id"] for marker in markers}
    if marker_ids != expected_refuted_ids:
        raise ContractValidationError(
            "accepted TaskClosed marker 必须精确覆盖每个 pass+refuted 组合"
        )
    for marker in markers:
        _validate_accepted_terminal_marker_binding(
            value, marker, session_by_id, finding_by_id
        )


def _validate_accepted_terminal_marker_binding(
    value: Mapping[str, Any],
    marker: sqlite3.Row,
    session_by_id: Mapping[str, sqlite3.Row],
    finding_by_id: Mapping[str, Mapping[str, Any]],
) -> None:
    """核对 accepted 终态的一条 marker 属于当前 pass+refuted 组合。

    输入是终态正文、marker 及 session/finding 索引；动作检查组合类型和当前来源，返回
    值为空，不修改数据库；组合未知、finding 类型错误或引用漂移时抛出合同错误。
    """

    combination_id = marker["combination_id"]
    session = session_by_id.get(combination_id)
    finding = finding_by_id.get(combination_id)
    if session is None or finding is None:
        raise ContractValidationError("accepted TaskClosed marker 必须属于当前五个组合")
    if not _is_pass_refuted_terminal_finding(finding):
        raise ContractValidationError(
            "accepted TaskClosed marker 必须绑定 pass+refuted 的当前结果"
        )
    if marker["review_event_ref"] != value["review_event_ref"]:
        raise ContractValidationError(
            "accepted TaskClosed marker 必须绑定 pass+refuted 的当前结果"
        )
    if marker["result_event_ref"] != session["latest_result_event_id"]:
        raise ContractValidationError(
            "accepted TaskClosed marker 必须绑定 pass+refuted 的当前结果"
        )


def _validate_terminal_marker_consistency(
    value: Mapping[str, Any],
    expected_ids: set[str],
    review: Mapping[str, Any],
    sessions: Sequence[sqlite3.Row],
    markers: Sequence[sqlite3.Row],
) -> None:
    """核对终态 marker 数量、来源和组合阶段。"""

    _validate_terminal_marker_references(value, markers)
    _validate_terminal_marker_session_bindings(
        value, expected_ids, review, sessions, markers
    )


def _write_investigation_task_terminal_projection(
    conn: sqlite3.Connection,
    task_id: str,
    event_id: str,
    value: Mapping[str, Any],
    created_at: str,
) -> None:
    """写入终态当前索引并同步任务状态。"""

    conn.execute(
        "INSERT INTO investigation_task_terminals(terminal_event_id,task_id,batch_id,"
        "accepted,can_answer,final_answer,answerability_explanation,review_event_ref,"
        "marker_event_refs_json,operation_instance_id,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        (
            event_id,
            task_id,
            value["batch_id"],
            int(value["accepted"]),
            int(value["can_answer"]),
            value["final_answer"],
            value["answerability_explanation"],
            value["review_event_ref"],
            canonical_json(value["marker_event_refs"]),
            value["operation_instance_id"],
            created_at,
        ),
    )
    task_status = "succeeded" if value["accepted"] else "stopped_no_supported_hypothesis"
    conn.execute(
        "UPDATE tasks SET status=?,final_outcome=?,version=version+1,updated_at=? "
        "WHERE task_id=?",
        (
            task_status,
            value["final_answer"] or value["answerability_explanation"],
            created_at,
            task_id,
        ),
    )


def insert_investigation_task_correction_projection(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    event_id: str,
    payload: Mapping[str, Any],
    created_at: str,
) -> None:
    """从任务更正事实撤销旧终态并建立同组合的待返修当前视图。

    输入是维护者已明确绑定 TaskClosed、Review、Result、组合和建议的不可变事件；动作
    只删除可重建的当前终态/路由索引，保留所有历史事件，再把任务恢复为 running 并
    用更正事件作为新的待返修请求。重放时执行同样的身份核对；当前终态、来源或预算
    漂移时抛出合同错误，不自动判断科学内容，也不改写旧事件。
    """

    value = validate_investigation_task_correction_payload(payload)
    if value["task_id"] != task_id:
        raise ContractValidationError("InvestigationTaskCorrection task 身份不一致")
    required_runtime_fields = (
        "batch_id",
        "result_version",
        "operation_instance_id",
        "investigation_protocol_version",
    )
    if any(field not in value for field in required_runtime_fields):
        raise ContractValidationError("InvestigationTaskCorrection 缺少完整运行身份")

    task = conn.execute(
        "SELECT status FROM tasks WHERE task_id=?", (task_id,)
    ).fetchone()
    if task is None:
        raise ContractValidationError("更正事实引用了未知 task")
    if task["status"] != "succeeded":
        raise ContractValidationError("更正只能引用当前 succeeded task")

    terminal = conn.execute(
        "SELECT * FROM investigation_task_terminals WHERE task_id=?", (task_id,)
    ).fetchone()
    if terminal is None:
        raise ContractValidationError("更正缺少当前 TaskClosed 投影")
    if terminal["terminal_event_id"] != value["terminal_event_ref"]:
        raise ContractValidationError("更正引用的 TaskClosed 不是当前终态")
    if not bool(terminal["accepted"]):
        raise ContractValidationError("更正只能引用 accepted TaskClosed")
    terminal_payload = validate_investigation_task_terminal_payload(
        _event_payload_by_id(
            conn,
            task_id,
            value["terminal_event_ref"],
            "investigation_task_closed",
        )
    )
    if (
        terminal_payload["batch_id"] != value["batch_id"]
        or terminal_payload["review_event_ref"] != value["review_event_ref"]
        or not terminal_payload["accepted"]
    ):
        raise ContractValidationError("更正的 TaskClosed 身份或状态已漂移")

    session = _investigation_session_for_event(
        conn,
        task_id,
        value["batch_id"],
        value["combination_id"],
    )
    if session["pending_repair_event_id"] is not None:
        raise ContractValidationError("更正目标已有 pending repair")
    if session["latest_result_event_id"] != value["result_event_ref"]:
        raise ContractValidationError("更正引用的 Result 不是当前版本")
    if session["current_result_version"] != value["result_version"]:
        raise ContractValidationError("更正引用的 Result 版本已漂移")
    if session["latest_review_event_id"] != value["review_event_ref"]:
        raise ContractValidationError("更正引用的 Review 不是当前版本")
    if int(session["repair_count"]) >= 3:
        raise ContractValidationError("同一组合最多允许三次 repair")

    result = conn.execute(
        "SELECT r.batch_id,r.combination_id,r.result_version,e.event_type "
        "FROM investigation_results r JOIN memory_events e "
        "ON e.event_id=r.result_event_id AND e.task_id=r.task_id "
        "WHERE r.task_id=? AND r.result_event_id=?",
        (task_id, value["result_event_ref"]),
    ).fetchone()
    if (
        result is None
        or result["event_type"] != "investigation_result_recorded"
        or result["batch_id"] != value["batch_id"]
        or result["combination_id"] != value["combination_id"]
        or result["result_version"] != value["result_version"]
    ):
        raise ContractValidationError("更正引用的 Result 身份或版本已漂移")

    review = validate_investigation_batch_review_payload(
        _event_payload_by_id(
            conn,
            task_id,
            value["review_event_ref"],
            "investigation_batch_reviewed",
        )
    )
    review_vector = dict(
        zip(review["combination_ids"], review["result_versions"])
    )
    if (
        review["batch_id"] != value["batch_id"]
        or value["combination_id"] not in review_vector
        or review_vector[value["combination_id"]] != value["result_version"]
    ):
        raise ContractValidationError("更正引用的 Review 未覆盖当前 Result")

    repair_count = int(session["repair_count"]) + 1
    conn.execute(
        "INSERT INTO investigation_repairs("
        "repair_request_event_id,task_id,batch_id,combination_id,result_version,"
        "repair_count,repair_advice,review_event_ref,operation_instance_id,created_at) "
        "VALUES(?,?,?,?,?,?,?,?,?,?)",
        (
            event_id,
            task_id,
            value["batch_id"],
            value["combination_id"],
            value["result_version"],
            repair_count,
            value["repair_advice"],
            value["review_event_ref"],
            value["operation_instance_id"],
            created_at,
        ),
    )
    # 这些是旧 TaskClosed 的可重建当前索引；历史 marker 事件仍保留，新的 Review
    # 必须重新绑定新 Result 向量，不能继续消费旧的 route marker。
    conn.execute(
        "DELETE FROM investigation_task_terminals WHERE task_id=?", (task_id,)
    )
    conn.execute(
        "DELETE FROM investigation_route_markers WHERE task_id=? AND batch_id=?",
        (task_id, value["batch_id"]),
    )
    conn.execute(
        "UPDATE investigation_sessions SET route_marker=NULL,"
        "phase=CASE WHEN combination_id=? THEN 'repair_pending' "
        "ELSE 'ready_for_batch_validation' END,"
        "pending_repair_event_id=CASE WHEN combination_id=? THEN ? ELSE NULL END,"
        "version=version+1,updated_at=? WHERE task_id=? AND batch_id=?",
        (
            value["combination_id"],
            value["combination_id"],
            event_id,
            created_at,
            task_id,
            value["batch_id"],
        ),
    )
    conn.execute(
        "UPDATE combinations SET investigation_phase=CASE WHEN combination_id=? "
        "THEN 'repair_pending' ELSE 'investigating' END,"
        "investigation_ready_state=CASE WHEN combination_id=? "
        "THEN 'repair_pending' ELSE 'ready_for_batch_validation' END,"
        "investigation_stage_state=CASE WHEN combination_id=? "
        "THEN 'repair_pending' ELSE 'ready_for_batch_validation' END,"
        "version=version+1,updated_at=? WHERE task_id=? AND batch_id=?",
        (
            value["combination_id"],
            value["combination_id"],
            value["combination_id"],
            created_at,
            task_id,
            value["batch_id"],
        ),
    )
    conn.execute(
        "UPDATE tasks SET status='running',final_outcome=NULL,version=version+1,"
        "updated_at=? WHERE task_id=?",
        (created_at, task_id),
    )


def insert_investigation_task_terminal_projection(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    event_id: str,
    payload: Mapping[str, Any],
    created_at: str,
) -> None:
    """从 TaskClosed 事件建立任务终态索引并更新任务状态。

    输入是同一事务中的 terminal 正文、事件 ID、task 和时间；动作核对当前 Review、
    五个 session/result、回答字段与 marker 来源，再写终态索引和 tasks 状态。返回值
    为空；来源过期、批次不完整、重复终态或数据库错误会整体回滚，不伪造科学事实。
    """

    value = validate_investigation_task_terminal_payload(payload)
    _validate_terminal_task_identity_and_uniqueness(conn, task_id, value)
    review = _load_and_validate_terminal_review(conn, task_id, value)
    sessions = _load_terminal_session_rows(conn, task_id, value["batch_id"])
    expected_ids = _validate_terminal_session_consistency(value, review, sessions)
    markers = _load_terminal_marker_rows(conn, task_id, value["batch_id"])
    _validate_terminal_marker_consistency(
        value, expected_ids, review, sessions, markers
    )
    _write_investigation_task_terminal_projection(
        conn,
        task_id,
        event_id,
        value,
        created_at,
    )


def _apply_investigation_task_failure_projection(
    conn: sqlite3.Connection,
    task_id: str,
    event: Mapping[str, Any],
) -> None:
    """从 Investigation 执行失败事实恢复 tasks 的失败状态和安全摘要。

    输入是已经追加到历史的 InvestigationTaskFailed 事件；动作复验任务身份后更新
    可重建的 tasks 投影，只保存 reason_code/error_type 摘要，不读取异常正文。返回
    ``None``；事件身份或 payload 非法时抛出合同错误，事务由调用方回滚。
    """

    value = validate_investigation_task_failure_payload(event["payload"])
    if value["task_id"] != task_id:
        raise ContractValidationError(
            "InvestigationTaskFailed task_id 与事件所属 task 不一致"
        )
    conn.execute(
        "UPDATE tasks SET status='failed', final_outcome=?, version=version+1, "
        "updated_at=? WHERE task_id=?",
        (canonical_json(value), event["created_at"], task_id),
    )


def clear_investigation_projection_rows(conn: sqlite3.Connection, task_id: str) -> None:
    """删除指定任务的 Investigation 当前索引，保留不可变事件。"""

    for table in (
        "investigation_task_terminals",
        "investigation_turn_attempt_terminations",
        "investigation_turn_waits",
        "investigation_repair_turns",
        "investigation_turns",
        "task_review_turns",
        "task_review_sessions",
        "investigation_evidence_verifications",
        "investigation_route_markers",
        "investigation_repairs",
        "investigation_reviews",
        "investigation_results",
        "investigation_sessions",
    ):
        conn.execute(f"DELETE FROM {table} WHERE task_id=?", (task_id,))


# ==================== 批次投影写入 ====================

def insert_batch_projection(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    event_id: str,
    payload: Mapping[str, Any],
    created_at: str,
) -> None:
    """根据新批次事实建立批次、候选和组合三个运行视图。

    三类视图在同一事务中写入，任何候选失败都不会留下残缺批次。
    """

    selected = [
        item for item in payload["candidates"] if item["selection_status"] == "selected"
    ]
    task_event = conn.execute(
        "SELECT event_type,payload_json FROM memory_events "
        "WHERE task_id=? AND event_type='task_created'",
        (task_id,),
    ).fetchall()
    if len(task_event) != 1:
        raise ContractValidationError("批次当前视图缺少唯一 TaskCreated 事实")
    task_created = validate_task_created_payload(
        _load_json(task_event[0]["payload_json"], None)
    )
    investigation_phase = (
        "pending"
        if task_created["stage_architecture_version"] == "investigation_v2"
        else None
    )
    investigation_stage_state = (
        "investigation_pending"
        if task_created["stage_architecture_version"] == "investigation_v2"
        else None
    )
    conn.execute(
        """
        INSERT INTO hypothesis_batches(
            batch_id, task_id, batch_index, state, selected_count,
            created_event_id, version, created_at, updated_at
        ) VALUES(?, ?, ?, 'running', ?, ?, 0, ?, ?)
        """,
        (
            payload["batch_id"],
            task_id,
            payload["batch_index"],
            len(selected),
            event_id,
            created_at,
            created_at,
        ),
    )
    for item in selected:
        conn.execute(
            """
            INSERT INTO combinations(
                combination_id, task_id, batch_id, combination_index, hypothesis_id,
                workflow_version, attempt_index, state, runtime_checkpoint_json,
                causal_source_ref, investigation_phase, investigation_ready_state,
                investigation_stage_state,
                version, created_at, updated_at
            ) VALUES(?, ?, ?, ?, ?, 0, 0, 'generating_workflow', '{}', ?, ?, ?, ?, 0, ?, ?)
            """,
            (
                item["combination_id"],
                task_id,
                payload["batch_id"],
                item["combination_index"],
                item["hypothesis_id"],
                # 任务级人工恢复发生在组合创建之前，创建本批次就是它唯一授权的
                # 实际结果。新组合不能再继承这份授权，否则第一个完成规划的组合会
                # 被误判为又一次消费同一人工决定。
                None,
                investigation_phase,
                "pending",
                investigation_stage_state,
                created_at,
                created_at,
            ),
        )
    for item in payload["candidates"]:
        conn.execute(
            """
            INSERT INTO hypothesis_candidates(
                hypothesis_id, batch_id, candidate_index, rank, confidence,
                selection_status, combination_id
            ) VALUES(?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item["hypothesis_id"],
                payload["batch_id"],
                item["candidate_index"],
                item["rank"],
                item.get("confidence"),
                item["selection_status"],
                item["combination_id"],
            ),
        )


# ==================== 因果历史重建 ====================

def _event_manifest(
    event: Mapping[str, Any],
    *,
    task_root: Path | None,
    required_key: str,
) -> Dict[str, Any] | None:
    """从事件明确引用的正式清单中读取重建所需内容。

    服务层已经核对引用路径和文件摘要，本函数只在这些获准文件中寻找执行或检查点
    字段。它不会扫描任务目录，因此遗留临时文件和孤立结果无法混入恢复状态。
    """

    if task_root is None:
        return None
    for relative_path in event.get("source_refs", []):
        if "/" not in relative_path:
            continue
        try:
            value = json.loads((task_root / relative_path).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ContractValidationError("重建引用的 finalized manifest 无法读取") from exc
        if isinstance(value, dict) and required_key in value:
            return value
    return None


def _read_gaussian_normalized_artifact(
    task_root: Path | None,
    reference: Any,
) -> Mapping[str, Any] | None:
    """读取并核对 codegen 的 normalized artifact；无任务目录时返回空值。"""

    if task_root is None or not isinstance(reference, Mapping):
        return None
    relative_path = reference.get("relative_path")
    if (
        not isinstance(relative_path, str)
        or not relative_path
        or PurePosixPath(relative_path).is_absolute()
        or ".." in PurePosixPath(relative_path).parts
        or "\\" in relative_path
    ):
        raise ContractValidationError("Gaussian normalized artifact 路径非法")
    candidate = task_root / relative_path
    current = task_root
    for part in PurePosixPath(relative_path).parts:
        current = current / part
        if current.is_symlink():
            raise ContractValidationError("Gaussian normalized artifact 不能经过符号链接")
    path = candidate.resolve()
    if path.is_symlink() or not path.is_file():
        raise ContractValidationError("Gaussian normalized artifact 不是普通文件")
    try:
        path.relative_to(task_root.resolve())
    except ValueError as exc:
        raise ContractValidationError("Gaussian normalized artifact 越出任务目录") from exc
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractValidationError("Gaussian normalized artifact 无法读取") from exc
    if (
        reference.get("media_type") != "application/json"
        or reference.get("size") != len(raw)
        or reference.get("sha256") != hashlib.sha256(raw).hexdigest()
        or not isinstance(value, Mapping)
        or canonical_json(value).encode("utf-8") != raw
    ):
        raise ContractValidationError("Gaussian normalized artifact 摘要或格式不匹配")
    return value


def _gaussian_code_receipt_projection(
    *,
    manifest: Mapping[str, Any],
    manifest_ref: str,
    task_root: Path | None,
) -> Mapping[str, Any] | None:
    """把完成的 codegen manifest 投影为 route 收据，不复制原始文件。"""

    if manifest.get("action_id") != "generate_gaussian_code":
        return None
    normalized = _read_gaussian_normalized_artifact(
        task_root,
        manifest.get("normalized_step_ref"),
    )
    route = extract_gaussian_code(normalized)
    if route is None:
        raise ContractValidationError(
            "generate_gaussian_code 完成清单缺少 result.gaussian_code"
        )
    step_id = manifest.get("step_id")
    if not isinstance(step_id, str) or not step_id:
        raise ContractValidationError("Gaussian codegen manifest 缺少 step_id")
    return {
        "step_id": step_id,
        "step_index": manifest.get("step_index"),
        "step_manifest_ref": manifest_ref,
        "normalized_step_ref": deepcopy(dict(manifest["normalized_step_ref"])),
        "gaussian_code": route,
        "route_sha256": hashlib.sha256(route.encode("utf-8")).hexdigest(),
    }


def _merge_gaussian_review_projection(
    previous: Any,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """重建时按 step_id 合并多次 Gaussian unit 审查事实。"""

    previous = previous if isinstance(previous, Mapping) else {}
    merged = deepcopy(dict(payload))
    records_by_id: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for records in (previous.get("records", []), payload.get("records", [])):
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, Mapping) or not isinstance(record.get("step_id"), str):
                continue
            step_id = record["step_id"]
            if step_id not in records_by_id:
                order.append(step_id)
            records_by_id[step_id] = deepcopy(dict(record))
    merged["records"] = [records_by_id[item] for item in order]
    overrides: dict[str, Any] = {}
    for source in (previous.get("overrides", {}), payload.get("overrides", {})):
        if isinstance(source, Mapping):
            overrides.update(deepcopy(dict(source)))
    merged["overrides"] = overrides
    statuses = {record.get("status") for record in records_by_id.values()}
    if payload.get("status") == "not_configured":
        merged["status"] = "not_configured"
        merged["requested_transition"] = (
            "revise_workflow"
            if payload.get("review_phase") == "pre_execution"
            else None
        )
    elif "reject" in statuses or payload.get("status") == "reject":
        merged["status"] = "reject"
        merged["requested_transition"] = "revise_workflow"
    elif "revise" in statuses:
        merged["status"] = "revise"
        merged["requested_transition"] = (
            "revise_workflow"
            if payload.get("review_phase") == "pre_execution"
            else None
        )
    elif statuses:
        merged["status"] = "approved"
        merged["requested_transition"] = None
    return merged


def _restart_source_pause(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    event: Mapping[str, Any],
) -> Dict[str, Any]:
    """找到人工恢复所处理的那条暂停事实。

    恢复事件还可能引用人工工作流修订，所以不能假定第一个来源就是暂停。这里按
    事件类型识别唯一暂停记录，重建器随后才能知道应回到规划、执行还是终态。
    """

    pauses = []
    for source_ref in event.get("source_refs", []):
        if "/" in source_ref:
            continue
        row = conn.execute(
            "SELECT event_type, payload_json FROM memory_events "
            "WHERE task_id=? AND event_id=?",
            (task_id, source_ref),
        ).fetchone()
        if row is not None and row["event_type"] == "execution_paused":
            pauses.append(_load_json(row["payload_json"], {}))
    if len(pauses) != 1:
        raise ContractValidationError("Restart 必须精确引用一条来源暂停事实")
    return pauses[0]


def _load_projection_history_nodes(
    conn: sqlite3.Connection,
    task_id: str,
) -> Dict[str, Dict[str, Any]]:
    """读取任务的不可变事件和工作流修订。"""

    event_rows = conn.execute(
        "SELECT * FROM memory_events WHERE task_id=?",
        (task_id,),
    ).fetchall()
    if not event_rows:
        raise ContractValidationError("无法为没有不可变事实的 task 重建投影")
    revision_rows = conn.execute(
        "SELECT * FROM revisions WHERE task_id=?",
        (task_id,),
    ).fetchall()

    nodes: Dict[str, Dict[str, Any]] = {}
    for raw in event_rows:
        record = event_row(raw)
        record_id = record["event_id"]
        nodes[record_id] = {
            "record_id": record_id,
            "kind": "event",
            "created_at": record["created_at"],
            "record": record,
        }
    for raw in revision_rows:
        record = dict(raw)
        record["payload"] = _load_json(record.pop("revised_result_json"), {})
        record["source_refs"] = _load_json(record.pop("source_refs_json"), [])
        record_id = record["revision_id"]
        if record_id in nodes:
            raise ContractValidationError("事件与修订使用了相同记录 ID")
        nodes[record_id] = {
            "record_id": record_id,
            "kind": "revision",
            "created_at": record["created_at"],
            "record": record,
        }
    return nodes


def _add_history_dependency(
    nodes: Mapping[str, Mapping[str, Any]],
    dependencies: Dict[str, set[str]],
    record_id: str,
    source_id: str,
    reason: str,
) -> None:
    """登记一条历史先后关系，并拒绝缺失来源和自引用。"""

    if source_id not in nodes:
        raise ContractValidationError(f"{reason} 引用了不存在的历史记录")
    if source_id == record_id:
        raise ContractValidationError(f"{reason} 形成了自引用")
    dependencies[record_id].add(source_id)


def _task_created_record_id(nodes: Mapping[str, Mapping[str, Any]]) -> str:
    """返回任务唯一的创建事实 ID。"""

    task_created = [
        record_id
        for record_id, node in nodes.items()
        if node["kind"] == "event"
        and node["record"]["event_type"] == "task_created"
    ]
    if len(task_created) != 1:
        raise ContractValidationError("task 必须恰有一条 TaskCreated 事实")
    return task_created[0]


def _add_direct_history_dependencies(
    nodes: Mapping[str, Mapping[str, Any]],
    dependencies: Dict[str, set[str]],
    task_created_id: str,
) -> None:
    """登记任务创建、来源引用和因果来源给出的直接依赖。"""

    # artifact 路径由服务层校验内容摘要；这里只处理数据库记录 ID。
    for record_id, node in nodes.items():
        if record_id != task_created_id:
            dependencies[record_id].add(task_created_id)
        record = node["record"]
        for source_ref in record.get("source_refs", []):
            if "/" not in source_ref:
                _add_history_dependency(
                    nodes,
                    dependencies,
                    record_id,
                    source_ref,
                    "source_refs",
                )
        causal_source_ref = record.get("payload", {}).get("causal_source_ref")
        if causal_source_ref is not None:
            _add_history_dependency(
                nodes,
                dependencies,
                record_id,
                causal_source_ref,
                "causal_source_ref",
            )


def _record_batch_retrieval_history_event(
    history: Dict[str, Any],
    record_id: str,
    event: Mapping[str, Any],
) -> None:
    """登记一条任务、批次或 Retrieval 历史事件。"""

    event_type = event["event_type"]
    payload = event["payload"]
    if event_type == "task_created":
        task_created = validate_task_created_payload(payload)
        history["task_architecture_version"] = task_created["stage_architecture_version"]
    elif event_type == "hypothesis_batch_created":
        batch_index = payload["batch_index"]
        batch_created_by_index = history["batch_created_by_index"]
        if batch_index in batch_created_by_index:
            raise ContractValidationError("同一 batch_index 存在多条创建事实")
        batch_created_by_index[batch_index] = record_id
        for candidate in payload["candidates"]:
            combination_id = candidate.get("combination_id")
            if combination_id is not None:
                history["combination_batch_source"][combination_id] = record_id
    elif event_type == "retrieval_paused":
        validate_retrieval_pause_payload(payload)
    elif event_type == "retrieval_completed":
        validate_retrieval_completed_payload(payload)
    elif event_type == "hypothesis_batch_closed":
        history["batch_closed_by_index"][payload["batch_index"]] = record_id


def _record_planning_expert_history_event(
    history: Dict[str, Any],
    record_id: str,
    event: Mapping[str, Any],
) -> None:
    """登记一条组合规划、工具缺口或专家阶段历史事件。"""

    event_type = event["event_type"]
    payload = event["payload"]
    if event_type == "planning_completed":
        combination_id = payload["combination_id"]
        if combination_id in history["planning_by_combination"]:
            raise ContractValidationError("同一组合存在多条 PlanningCompleted")
        history["planning_by_combination"][combination_id] = record_id
    elif event_type == "tool_capability_gap":
        combination_id = payload["combination_id"]
        if combination_id in history["gap_by_combination"]:
            raise ContractValidationError("同一组合存在多条工具能力缺口事实")
        history["gap_by_combination"][combination_id] = record_id
    elif event_type == "expert_gaussian_review":
        combination_id = payload["combination_id"]
        history["expert_review_by_combination"].setdefault(combination_id, []).append(record_id)
    elif event_type == "expert_debate_completed":
        combination_id = payload["combination_id"]
        history["expert_debate_by_combination"].setdefault(combination_id, []).append(record_id)
    elif event_type == "expert_model_paused":
        combination_id = payload["combination_id"]
        history["expert_pause_by_combination"].setdefault(combination_id, []).append(record_id)


def _collect_batch_planning_history_events(
    nodes: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """收集批次、规划、能力缺口和专家阶段的历史来源。

    输入是已经解码的事件/修订节点；动作只验证并登记本阶段需要的事件身份，返回
    后续依赖 helper 使用的有序字典。缺少来源或同一组合重复事实时抛出合同错误，
    不修改节点和依赖图，也不读取外部文件。
    """

    history: Dict[str, Any] = {
        "batch_created_by_index": {},
        "batch_closed_by_index": {},
        "task_architecture_version": "legacy_workflow_v1",
        "combination_batch_source": {},
        "planning_by_combination": {},
        "gap_by_combination": {},
        "expert_review_by_combination": {},
        "expert_debate_by_combination": {},
        "expert_pause_by_combination": {},
    }
    for record_id, node in nodes.items():
        if node["kind"] != "event":
            continue
        event = node["record"]
        _record_batch_retrieval_history_event(history, record_id, event)
        _record_planning_expert_history_event(history, record_id, event)
    return history


def _resolve_investigation_v2_batch_marker_ids(
    nodes: Mapping[str, Mapping[str, Any]],
    source_record_ids: Sequence[str],
    prior_batch_id: Any,
) -> List[str]:
    """解析 v2 后续批次引用的 marker，并兼容 v7 来源缺失写法。"""

    marker_ids = []
    for source_id in source_record_ids:
        source_node = nodes.get(source_id)
        if source_node is None:
            raise ContractValidationError(
                "Investigation v2 后续批次引用了不存在的历史来源"
            )
        source_event = source_node["record"]
        if source_event["event_type"] == "investigation_route_marked":
            marker_ids.append(source_id)
    if marker_ids:
        return marker_ids
    # v7 事件可能只保存上一批 BatchCreated 来源；读取兼容时从同一批
    # 不可变 marker 事实补足拓扑依赖，不修改旧事件，也不把它伪造成新写入。
    return [
        source_id
        for source_id, source_node in nodes.items()
        if source_node["kind"] == "event"
        and source_node["record"]["event_type"] == "investigation_route_marked"
        and source_node["record"]["payload"].get("batch_id") == prior_batch_id
    ]


def _add_investigation_v2_batch_order_dependencies(
    nodes: Mapping[str, Mapping[str, Any]],
    dependencies: Dict[str, set[str]],
    batch_created_by_index: Mapping[int, str],
    record_id: str,
    batch_index: int,
) -> None:
    """为 Investigation v2 后续批次登记上一批和五个 marker 的依赖。"""

    batch_payload = nodes[record_id]["record"]["payload"]
    generation_source = batch_payload.get("generation_source")
    source_record_ids = (
        generation_source.get("source_record_ids")
        if isinstance(generation_source, Mapping)
        else None
    )
    if not isinstance(source_record_ids, list):
        raise ContractValidationError("Investigation v2 后续批次缺少可重建的 marker 来源")
    prior = batch_created_by_index.get(batch_index - 1)
    if prior is None or prior not in source_record_ids:
        raise ContractValidationError(
            "Investigation v2 后续批次必须因果引用直接上一批 BatchCreated"
        )
    marker_ids = _resolve_investigation_v2_batch_marker_ids(
        nodes,
        source_record_ids,
        nodes[prior]["record"]["payload"].get("batch_id"),
    )
    marker_combinations = {
        nodes[marker_id]["record"]["payload"].get("combination_id")
        for marker_id in marker_ids
    }
    if len(marker_ids) != 5 or len(marker_combinations) != 5:
        raise ContractValidationError(
            "Investigation v2 后续批次必须因果引用上一批五个 marker"
        )
    for marker_id in marker_ids:
        _add_history_dependency(
            nodes,
            dependencies,
            record_id,
            marker_id,
            "Investigation v2 批次 marker 顺序",
        )
    _add_history_dependency(
        nodes,
        dependencies,
        record_id,
        prior,
        "Investigation v2 批次顺序",
    )


def _add_batch_sequence_dependencies(
    nodes: Mapping[str, Mapping[str, Any]],
    dependencies: Dict[str, set[str]],
    history: Mapping[str, Any],
) -> None:
    """登记后续批次必须排在上一批之后的历史依赖。"""

    batch_created_by_index = history["batch_created_by_index"]
    for batch_index, record_id in batch_created_by_index.items():
        if batch_index <= 1:
            continue
        if history["task_architecture_version"] == "investigation_v2":
            _add_investigation_v2_batch_order_dependencies(
                nodes,
                dependencies,
                batch_created_by_index,
                record_id,
                batch_index,
            )
            continue
        prior = history["batch_closed_by_index"].get(batch_index - 1)
        if prior is None:
            raise ContractValidationError("后续批次缺少上一批关闭事实")
        _add_history_dependency(
            nodes,
            dependencies,
            record_id,
            prior,
            "批次顺序",
        )


def _add_planning_batch_source_dependencies(
    nodes: Mapping[str, Mapping[str, Any]],
    dependencies: Dict[str, set[str]],
    combination_batch_source: Mapping[str, str],
    planning_by_combination: Mapping[str, str],
) -> None:
    """登记每条组合规划必须排在对应批次创建之后。"""

    for combination_id, planning_id in planning_by_combination.items():
        batch_source = combination_batch_source.get(combination_id)
        if batch_source is None:
            raise ContractValidationError("PlanningCompleted 找不到组合创建事实")
        _add_history_dependency(
            nodes,
            dependencies,
            planning_id,
            batch_source,
            "组合规划",
        )


def _add_tool_gap_batch_source_dependencies(
    nodes: Mapping[str, Mapping[str, Any]],
    dependencies: Dict[str, set[str]],
    combination_batch_source: Mapping[str, str],
    planning_by_combination: Mapping[str, str],
    gap_by_combination: Mapping[str, str],
) -> None:
    """登记工具能力缺口的批次来源和可选规划来源。"""

    for combination_id, gap_id in gap_by_combination.items():
        batch_source = combination_batch_source.get(combination_id)
        if batch_source is None:
            raise ContractValidationError("ToolCapabilityGap 找不到组合创建事实")
        _add_history_dependency(
            nodes,
            dependencies,
            gap_id,
            batch_source,
            "组合工具能力缺口",
        )
        planning_source = planning_by_combination.get(combination_id)
        if planning_source is not None:
            # 修订阶段也可能确认能力缺口，必须排在原工作流规划之后。
            _add_history_dependency(
                nodes,
                dependencies,
                gap_id,
                planning_source,
                "修订阶段工具能力缺口",
            )


def _add_expert_stage_planning_dependencies(
    nodes: Mapping[str, Mapping[str, Any]],
    dependencies: Dict[str, set[str]],
    planning_by_combination: Mapping[str, str],
    history: Mapping[str, Any],
) -> None:
    """登记 Gaussian 审查、ExpertDebate 和暂停事实的规划来源。"""

    for label, records in (
        ("Gaussian 审查", history["expert_review_by_combination"]),
        ("ExpertDebate", history["expert_debate_by_combination"]),
        ("ExpertModelPaused", history["expert_pause_by_combination"]),
    ):
        for combination_id, event_ids in records.items():
            planning_source = planning_by_combination.get(combination_id)
            if planning_source is None:
                raise ContractValidationError(f"{label} 找不到组合规划事实")
            for event_id in event_ids:
                _add_history_dependency(nodes, dependencies, event_id, planning_source, label)


def _add_batch_and_planning_dependencies(
    nodes: Mapping[str, Mapping[str, Any]],
    dependencies: Dict[str, set[str]],
) -> Dict[str, str]:
    """登记批次顺序以及组合规划和能力缺口的来源批次。"""

    history = _collect_batch_planning_history_events(nodes)
    _add_batch_sequence_dependencies(nodes, dependencies, history)
    _add_planning_batch_source_dependencies(
        nodes,
        dependencies,
        history["combination_batch_source"],
        history["planning_by_combination"],
    )
    _add_tool_gap_batch_source_dependencies(
        nodes,
        dependencies,
        history["combination_batch_source"],
        history["planning_by_combination"],
        history["gap_by_combination"],
    )
    _add_expert_stage_planning_dependencies(
        nodes,
        dependencies,
        history["planning_by_combination"],
        history,
    )
    return history["planning_by_combination"]


def _add_workflow_revision_dependencies(
    nodes: Mapping[str, Mapping[str, Any]],
    dependencies: Dict[str, set[str]],
    planning_by_combination: Mapping[str, str],
) -> None:
    """登记组合历史和工作流版本的产生顺序。"""

    revision_by_version: Dict[tuple[str, int], str] = {}
    for record_id, node in nodes.items():
        if node["kind"] != "revision":
            continue
        revision = node["record"]
        payload = revision["payload"]
        if revision["revision_type"] not in {
            "model_workflow_revision",
            "human_workflow_patch",
        }:
            raise ContractValidationError("投影重建遇到未知 workflow revision 类型")
        key = (payload.get("combination_id"), payload.get("workflow_version_after"))
        if not isinstance(key[0], str) or not isinstance(key[1], int):
            raise ContractValidationError("workflow revision 缺少组合或版本身份")
        if key in revision_by_version:
            raise ContractValidationError("同一组合版本由多条修订产生")
        revision_by_version[key] = record_id
        before_version = payload.get("workflow_version_before")
        if isinstance(before_version, int) and before_version > 1:
            prior = revision_by_version.get((key[0], before_version))
            if prior is not None:
                _add_history_dependency(
                    nodes,
                    dependencies,
                    record_id,
                    prior,
                    "修订版本链",
                )

    for record_id, node in nodes.items():
        record = node["record"]
        payload = record.get("payload", {})
        combination_id = payload.get("combination_id")
        if isinstance(combination_id, str):
            planning_id = planning_by_combination.get(combination_id)
            if planning_id is not None and record_id != planning_id:
                _add_history_dependency(
                    nodes,
                    dependencies,
                    record_id,
                    planning_id,
                    "组合历史",
                )
        if node["kind"] != "event":
            continue
        workflow_version = payload.get("workflow_version")
        if isinstance(combination_id, str) and isinstance(workflow_version, int):
            if workflow_version <= 1:
                continue
            producer = revision_by_version.get((combination_id, workflow_version))
            if producer is None:
                raise ContractValidationError("事件使用了没有修订来源的工作流版本")
            _add_history_dependency(
                nodes,
                dependencies,
                record_id,
                producer,
                "工作流版本",
            )


def _add_execution_history_dependencies(
    nodes: Mapping[str, Mapping[str, Any]],
    dependencies: Dict[str, set[str]],
    *,
    task_root: Path | None,
) -> None:
    """根据步骤清单和检查点序号登记执行历史依赖。"""

    step_event_by_manifest: Dict[str, str] = {}
    request_rebound_by_ref: Dict[tuple[str, str], str] = {}
    for record_id, node in nodes.items():
        if node["kind"] != "event":
            continue
        event = node["record"]
        if event["event_type"] == "execution_request_rebound":
            payload = event["payload"]
            request_rebound_by_ref[(payload["combination_id"], payload["request_ref"])] = record_id
            continue
        if event["event_type"] not in {
            "execution_step_completed",
            "execution_step_failed",
        }:
            continue
        paths = [item for item in node["record"]["source_refs"] if "/" in item]
        if node["record"]["event_type"] == "execution_step_failed":
            failed_ref = node["record"]["payload"].get("failed_step_manifest_ref")
            paths = [failed_ref] if isinstance(failed_ref, str) else []
        if len(paths) != 1 or paths[0] in step_event_by_manifest:
            raise ContractValidationError("步骤终局事实必须唯一绑定失败或完成清单")
        step_event_by_manifest[paths[0]] = record_id

    checkpoint_groups: Dict[str, List[tuple[int, str]]] = {}
    for record_id, node in nodes.items():
        if node["kind"] != "event":
            continue
        event = node["record"]
        payload = event["payload"]
        if event["event_type"] == "execution_checkpointed":
            checkpoint_groups.setdefault(payload["execution_id"], []).append(
                (payload["checkpoint_index"], record_id)
            )
        manifest_key = None
        if event["event_type"] in {
            "execution_step_completed",
            "execution_step_failed",
            "execution_checkpointed",
            "attempt",
        }:
            manifest_key = "step_manifest_refs"
        elif (
            event["event_type"] == "execution_paused"
            and payload.get("operation") == "execute_step"
        ):
            manifest_key = "step_manifest_refs"
        if manifest_key is None:
            continue
        manifest = _event_manifest(
            event,
            task_root=task_root,
            required_key=manifest_key,
        )
        if manifest is None:
            raise ContractValidationError("执行历史缺少可重建的步骤清单")
        request_ref = manifest.get("request_ref")
        if isinstance(request_ref, Mapping):
            request_ref = request_ref.get("relative_path")
        combination_id = event["payload"].get("combination_id")
        rebound_id = request_rebound_by_ref.get((combination_id, request_ref))
        if rebound_id is not None and rebound_id != record_id:
            _add_history_dependency(
                nodes,
                dependencies,
                record_id,
                rebound_id,
                "物化 request 重绑定",
            )
        # 失败事实的 payload 保存完整、有序 workflow 清单；不要从按字典序排列的
        # 多份 source_refs 中任选一份 manifest 推断前缀，否则 not_executed 清单
        # 可能遮蔽失败点之前的 completed 依赖。
        prefix = (
            payload.get("step_manifest_refs", [])[: payload.get("step_index", 0)]
            if event["event_type"] == "execution_step_failed"
            else manifest.get("step_manifest_refs")
        )
        if not isinstance(prefix, list):
            raise ContractValidationError("步骤清单前缀必须是数组")
        for manifest_ref in prefix:
            source_id = step_event_by_manifest.get(manifest_ref)
            if source_id is not None:
                _add_history_dependency(
                    nodes,
                    dependencies,
                    record_id,
                    source_id,
                    "步骤前缀",
                )
        checkpoint_ids = manifest.get("checkpoint_record_ids", [])
        if not isinstance(checkpoint_ids, list):
            raise ContractValidationError("步骤清单的 checkpoint_record_ids 必须是数组")
        for checkpoint_id in checkpoint_ids:
            _add_history_dependency(
                nodes,
                dependencies,
                record_id,
                checkpoint_id,
                "步骤检查点",
            )

    for checkpoints in checkpoint_groups.values():
        checkpoints.sort()
        indexes = [item[0] for item in checkpoints]
        if len(indexes) != len(set(indexes)):
            raise ContractValidationError("同一 execution 存在重复 checkpoint_index")
        for (_, prior_id), (_, current_id) in zip(checkpoints, checkpoints[1:]):
            _add_history_dependency(
                nodes,
                dependencies,
                current_id,
                prior_id,
                "检查点顺序",
            )

    completed_by_identity: Dict[tuple[str, str], List[str]] = {}
    failed_by_identity: Dict[tuple[str, str], List[str]] = {}
    expert_debate_by_identity: Dict[tuple[str, str], List[str]] = {}
    for record_id, node in nodes.items():
        if node["kind"] != "event":
            continue
        event = node["record"]
        payload = event["payload"]
        if event["event_type"] == "execution_step_completed":
            key = (payload["combination_id"], payload["execution_id"])
            completed_by_identity.setdefault(key, []).append(record_id)
        elif event["event_type"] == "execution_step_failed":
            key = (payload["combination_id"], payload["execution_id"])
            failed_by_identity.setdefault(key, []).append(record_id)
        elif event["event_type"] == "expert_debate_completed":
            key = (payload["combination_id"], payload["execution_id"])
            expert_debate_by_identity.setdefault(key, []).append(record_id)
    for record_id, node in nodes.items():
        if node["kind"] != "event":
            continue
        event = node["record"]
        if event["event_type"] not in {
            "expert_debate_completed",
            "expert_model_paused",
        }:
            continue
        payload = event["payload"]
        if payload.get("phase") == "gaussian_step_review":
            continue
        execution_id = payload.get("execution_id")
        combination_id = payload.get("combination_id")
        if not isinstance(execution_id, str) or not isinstance(combination_id, str):
            continue
        sources = [
            *completed_by_identity.get((combination_id, execution_id), []),
            *failed_by_identity.get((combination_id, execution_id), []),
        ]
        if not sources:
            raise ContractValidationError("专家完整结果缺少执行步骤完成事实")
        for source_id in sources:
            _add_history_dependency(nodes, dependencies, record_id, source_id, "专家完整结果")

    # ExpertDebate 是完整执行结果的最终裁判输入；若后续已经提交 Attempt，重建时
    # 必须先应用 debate 再应用 Attempt，避免 Attempt 投影覆盖刚保存的 debate JSON。
    for record_id, node in nodes.items():
        if node["kind"] != "event" or node["record"]["event_type"] != "attempt":
            continue
        payload = node["record"]["payload"]
        key = (payload["combination_id"], payload["execution_id"])
        for debate_id in expert_debate_by_identity.get(key, []):
            _add_history_dependency(nodes, dependencies, record_id, debate_id, "专家 debate")


def _add_batch_terminal_dependencies(
    nodes: Mapping[str, Mapping[str, Any]],
    dependencies: Dict[str, set[str]],
) -> None:
    """让批次关闭事实等待全部组合终态。"""

    for record_id, node in nodes.items():
        if node["kind"] != "event":
            continue
        event = node["record"]
        if event["event_type"] != "hypothesis_batch_closed":
            continue
        for result in event["payload"]["combination_results"]:
            _add_history_dependency(
                nodes,
                dependencies,
                record_id,
                result["terminal_record_id"],
                "批次终态",
            )


def _validate_human_patch_consumers(
    nodes: Mapping[str, Mapping[str, Any]],
) -> None:
    """确认每条人工工作流补丁恰由一次 Restart 采用。"""

    for record_id, node in nodes.items():
        if node["kind"] != "revision" or node["record"]["revision_type"] != (
            "human_workflow_patch"
        ):
            continue
        consumers = [
            other_id
            for other_id, other in nodes.items()
            if other["kind"] == "event"
            and other["record"]["event_type"] == "restart_applied"
            and record_id in other["record"]["source_refs"]
        ]
        if len(consumers) != 1:
            raise ContractValidationError("人工工作流修订必须由唯一 Restart 采用")


def _topologically_order_history(
    nodes: Mapping[str, Dict[str, Any]],
    dependencies: Mapping[str, set[str]],
) -> List[Dict[str, Any]]:
    """按因果关系排序；创建时间仅排列彼此无关的记录。"""

    dependents: Dict[str, set[str]] = {record_id: set() for record_id in nodes}
    indegree = {record_id: len(items) for record_id, items in dependencies.items()}
    for record_id, source_ids in dependencies.items():
        for source_id in source_ids:
            dependents[source_id].add(record_id)

    ready: List[tuple[str, int, str]] = []
    for record_id, count in indegree.items():
        if count == 0:
            node = nodes[record_id]
            heapq.heappush(
                ready,
                (
                    node["created_at"],
                    0 if node["kind"] == "event" else 1,
                    record_id,
                ),
            )
    ordered: List[Dict[str, Any]] = []
    while ready:
        _, _, record_id = heapq.heappop(ready)
        ordered.append(nodes[record_id])
        for dependent_id in sorted(dependents[record_id]):
            indegree[dependent_id] -= 1
            if indegree[dependent_id] == 0:
                node = nodes[dependent_id]
                heapq.heappush(
                    ready,
                    (
                        node["created_at"],
                        0 if node["kind"] == "event" else 1,
                        dependent_id,
                    ),
                )
    if len(ordered) != len(nodes):
        raise ContractValidationError("事件与修订的因果关系形成了环")
    return ordered


def _projection_history_nodes(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    task_root: Path | None,
) -> List[Dict[str, Any]]:
    """把事件和修订按明确依赖排成一条可重放历史。

    每类因果证据依次加入同一张依赖图，最后才执行拓扑排序。公开重建入口因而能
    直接看出历史排序的完整阶段，而不必穿过一段三百行的混合循环。
    """

    nodes = _load_projection_history_nodes(conn, task_id)
    dependencies: Dict[str, set[str]] = {
        record_id: set() for record_id in nodes
    }
    task_created_id = _task_created_record_id(nodes)
    _add_direct_history_dependencies(nodes, dependencies, task_created_id)
    planning_by_combination = _add_batch_and_planning_dependencies(
        nodes,
        dependencies,
    )
    _add_workflow_revision_dependencies(
        nodes,
        dependencies,
        planning_by_combination,
    )
    _add_execution_history_dependencies(
        nodes,
        dependencies,
        task_root=task_root,
    )
    _add_batch_terminal_dependencies(nodes, dependencies)
    _validate_human_patch_consumers(nodes)
    return _topologically_order_history(nodes, dependencies)


# ==================== 事件投影应用 ====================

def _apply_revision_projection(
    conn: sqlite3.Connection,
    task_id: str,
    revision: Mapping[str, Any],
) -> None:
    """在修订真正发生的位置更新组合，而不是在全部事件结束后统一覆盖。"""

    payload = revision["payload"]
    combination_id = payload.get("combination_id")
    row = conn.execute(
        "SELECT * FROM combinations WHERE task_id=? AND combination_id=?",
        (task_id, combination_id),
    ).fetchone()
    if row is None:
        raise ContractValidationError("workflow revision 找不到来源组合")
    current_workflow = _load_json(row["current_workflow_json"], None)
    if (
        row["workflow_version"] != payload.get("workflow_version_before")
        or current_workflow != payload.get("before_workflow")
    ):
        raise ContractValidationError("workflow revision 的前置工作流与因果位置不一致")

    if revision["revision_type"] == "model_workflow_revision":
        execution_id_after = payload.get("execution_id_after")
        if not isinstance(execution_id_after, str) or not execution_id_after:
            raise ContractValidationError("模型修订缺少可重建的新 execution_id")
        previous_runtime = _load_json(row["runtime_checkpoint_json"], {})
        direct_failure = previous_runtime.get("execution_failure") if isinstance(previous_runtime, Mapping) else None
        if row["state"] != "revising_workflow" and not isinstance(direct_failure, Mapping):
            raise ContractValidationError("模型修订没有紧跟来源 Attempt 或 execution failure")
        runtime = empty_runtime_checkpoint()
        if isinstance(direct_failure, Mapping):
            failed_index = direct_failure.get("step_index")
            old_refs = previous_runtime.get("step_manifest_refs")
            before_steps = payload.get("before_workflow", {}).get("steps") if isinstance(payload.get("before_workflow"), Mapping) else None
            after_steps = payload.get("after_workflow", {}).get("steps") if isinstance(payload.get("after_workflow"), Mapping) else None
            if (
                isinstance(failed_index, bool)
                or not isinstance(failed_index, int)
                or failed_index < 0
                or not isinstance(old_refs, list)
                or not isinstance(before_steps, list)
                or not isinstance(after_steps, list)
                or len(before_steps) != len(old_refs)
                or len(after_steps) <= failed_index
                or before_steps[:failed_index] != after_steps[:failed_index]
            ):
                raise ContractValidationError("模型修订修改了已完成步骤前缀")
            runtime["next_step_index"] = failed_index
            runtime["step_manifest_refs"] = list(old_refs[:failed_index])
            runtime["step_revision_attempts"] = deepcopy(previous_runtime.get("step_revision_attempts", {}))
            prefix_ids = {
                step.get("step_id") for step in after_steps[:failed_index]
                if isinstance(step, Mapping)
            }
            old_code_results = previous_runtime.get("gaussian_code_results", {})
            runtime["gaussian_code_results"] = {
                str(step_id): deepcopy(value)
                for step_id, value in (old_code_results.items() if isinstance(old_code_results, Mapping) else [])
                if step_id in prefix_ids
            }
        cursor = conn.execute(
            """
            UPDATE combinations SET workflow_version=?, attempt_index=attempt_index+1,
                execution_id=?, state='running', current_workflow_json=?,
                runtime_checkpoint_json=?, causal_source_ref=NULL,
                version=version+1, updated_at=?
            WHERE task_id=? AND combination_id=?
            """,
            (
                payload["workflow_version_after"],
                execution_id_after,
                canonical_json(payload["after_workflow"]),
                canonical_json(runtime),
                revision["updated_at"],
                task_id,
                combination_id,
            ),
        )
    elif revision["revision_type"] == "human_workflow_patch":
        if row["state"] != "paused":
            raise ContractValidationError("人工补丁没有紧跟来源 Pause")
        # 人工补丁与随后 Restart 是同一个原子动作。实时路径只由 Restart 增加一次
        # projection version，因此这里先替换工作流，等 Restart 再完成状态推进。
        cursor = conn.execute(
            """
            UPDATE combinations SET workflow_version=?, current_workflow_json=?
            WHERE task_id=? AND combination_id=?
            """,
            (
                payload["workflow_version_after"],
                canonical_json(payload["after_workflow"]),
                task_id,
                combination_id,
            ),
        )
    else:
        raise ContractValidationError("投影重建遇到未知 workflow revision 类型")
    if cursor.rowcount != 1:
        raise ContractValidationError("workflow revision 无法更新来源组合")


def _clear_task_projection_rows(
    conn: sqlite3.Connection,
    task_id: str,
) -> None:
    """删除可由不可变历史重新计算的任务运行视图。"""

    conn.execute(
        "DELETE FROM hypothesis_candidates WHERE batch_id IN "
        "(SELECT batch_id FROM hypothesis_batches WHERE task_id=?)",
        (task_id,),
    )
    conn.execute("DELETE FROM combinations WHERE task_id=?", (task_id,))
    conn.execute("DELETE FROM hypothesis_batches WHERE task_id=?", (task_id,))


def _apply_planning_completed_projection(
    conn: sqlite3.Connection,
    task_id: str,
    event: Mapping[str, Any],
) -> None:
    """将规划完成事实投影为组合的首个可执行工作流。"""

    payload = event["payload"]
    # Planning 事件可能同时引用 Planner 和 Optimizer 的 finalized manifest；
    # 重建时复用同一份两阶段来源合同，避免投影悄悄丢掉首版 generation provenance。
    validate_planning_source_refs(event.get("source_refs"))
    runtime_checkpoint = empty_runtime_checkpoint()
    conn.execute(
        """
        UPDATE combinations SET
            workflow_id=?, workflow_version=1, attempt_index=1,
            execution_id=?, state='running', current_workflow_json=?,
            runtime_checkpoint_json=?, causal_source_ref=?,
            version=version+1, updated_at=?
        WHERE task_id=? AND combination_id=?
        """,
        (
            payload["workflow_id"],
            payload["execution_id"],
            canonical_json(payload["workflow"]),
            canonical_json(runtime_checkpoint),
            # 规划完成是本次人工重试的唯一结果，恢复授权到此结束。后续工具步骤
            # 不能再次把同一授权当作新操作依据。
            None,
            event["created_at"],
            task_id,
            payload["combination_id"],
        ),
    )


def _apply_expert_gaussian_review_projection(
    conn: sqlite3.Connection,
    task_id: str,
    event: Mapping[str, Any],
) -> None:
    """重建 Gaussian 审查、Executor 覆盖项和 reject 暂停状态。"""

    payload = event["payload"]
    row = conn.execute(
        "SELECT runtime_checkpoint_json, state FROM combinations "
        "WHERE task_id=? AND combination_id=?",
        (task_id, payload["combination_id"]),
    ).fetchone()
    if row is None or row["state"] != "running":
        raise ContractValidationError("ExpertGaussianReview 没有 running 来源组合")
    runtime = _load_json(row["runtime_checkpoint_json"], {})
    runtime["expert_review"] = _merge_gaussian_review_projection(
        runtime.get("expert_review"),
        payload,
    )
    approved = payload["status"] == "approved"
    pre_execution = payload.get("review_phase") == "pre_execution"
    runtime["workflow_frozen"] = approved
    if approved:
        runtime.pop("expert_revision_feedback", None)
    elif pre_execution:
        runtime["expert_revision_feedback"] = deepcopy(dict(payload))
        runtime["expert_revision_feedback"]["source_event_id"] = event["event_id"]
    next_state = "running" if approved else "revising_workflow" if pre_execution else (
        "paused" if payload["status"] in {"reject", "not_configured"} else "running"
    )
    terminal_reason = None if approved or pre_execution or payload["status"] == "revise" else (
        "expert_gaussian_reject" if payload["status"] == "reject" else "expert_model_not_configured"
    )
    conn.execute(
        "UPDATE combinations SET state=?, terminal_reason=?, runtime_checkpoint_json=?, version=version+1, updated_at=? "
        "WHERE task_id=? AND combination_id=? AND state='running'",
        (
            next_state,
            terminal_reason,
            canonical_json(runtime),
            event["created_at"],
            task_id,
            payload["combination_id"],
        ),
    )


def _apply_expert_debate_projection(
    conn: sqlite3.Connection,
    task_id: str,
    event: Mapping[str, Any],
) -> None:
    """重建完整执行后的 debate JSON。"""

    payload = event["payload"]
    row = conn.execute(
        "SELECT runtime_checkpoint_json, state FROM combinations "
        "WHERE task_id=? AND combination_id=?",
        (task_id, payload["combination_id"]),
    ).fetchone()
    if row is None or row["state"] not in {"running", "routing"}:
        raise ContractValidationError("ExpertDebate 没有可路由的来源组合")
    runtime = _load_json(row["runtime_checkpoint_json"], {})
    runtime["expert_debate"] = payload
    conn.execute(
        "UPDATE combinations SET state='routing', runtime_checkpoint_json=?, version=version+1, updated_at=? "
        "WHERE task_id=? AND combination_id=?",
        (
            canonical_json(runtime),
            event["created_at"],
            task_id,
            payload["combination_id"],
        ),
    )


def _apply_expert_pause_projection(
    conn: sqlite3.Connection,
    task_id: str,
    event: Mapping[str, Any],
) -> None:
    """重建 provider 未配置时的专家暂停。"""

    payload = event["payload"]
    row = conn.execute(
        "SELECT runtime_checkpoint_json FROM combinations "
        "WHERE task_id=? AND combination_id=?",
        (task_id, payload["combination_id"]),
    ).fetchone()
    if row is None:
        raise ContractValidationError("ExpertModelPaused 找不到来源组合")
    runtime = _load_json(row["runtime_checkpoint_json"], {})
    runtime["expert_pause"] = payload
    conn.execute(
        "UPDATE combinations SET state='paused', terminal_reason='expert_model_not_configured', runtime_checkpoint_json=?, version=version+1, updated_at=? "
        "WHERE task_id=? AND combination_id=?",
        (
            canonical_json(runtime),
            event["created_at"],
            task_id,
            payload["combination_id"],
        ),
    )


def _tool_gap_source_combination(
    conn: sqlite3.Connection,
    task_id: str,
    value: Mapping[str, Any],
    refs: List[str],
) -> sqlite3.Row:
    """核对能力缺口是否紧跟规划或工作流修订阶段。"""

    manifest_refs = [item for item in refs if "/" in item]
    record_refs = [item for item in refs if "/" not in item]
    row = conn.execute(
        "SELECT * FROM combinations WHERE task_id=? AND combination_id=?",
        (task_id, value["combination_id"]),
    ).fetchone()
    if (
        row is None
        or row["hypothesis_id"] != value["hypothesis_id"]
        or row["causal_source_ref"] != value["causal_source_ref"]
    ):
        raise ContractValidationError(
            "ToolCapabilityGap 没有紧跟未规划或待修订的来源组合"
        )

    if row["state"] == "generating_workflow":
        if (
            len(refs) != 1
            or len(manifest_refs) != 1
            or record_refs
            or row["workflow_id"] is not None
            or row["workflow_version"] != 0
            or row["attempt_index"] != 0
            or row["execution_id"] is not None
            or row["current_workflow_json"] is not None
        ):
            raise ContractValidationError(
                "Planner 能力缺口没有紧跟未规划的来源组合"
            )
        return row

    if row["state"] != "revising_workflow":
        raise ContractValidationError(
            "ToolCapabilityGap 没有紧跟规划或工作流修订阶段"
        )
    if (
        len(refs) != 3
        or len(manifest_refs) != 2
        or len(record_refs) != 1
        or not isinstance(row["workflow_id"], str)
        or row["workflow_version"] < 1
        or row["attempt_index"] < 1
        or not isinstance(row["execution_id"], str)
        or row["current_workflow_json"] is None
    ):
        raise ContractValidationError("修订能力缺口没有紧跟一轮完整执行")

    attempt_row = conn.execute(
        "SELECT event_type, payload_json FROM memory_events "
        "WHERE task_id=? AND event_id=?",
        (task_id, record_refs[0]),
    ).fetchone()
    if attempt_row is None or attempt_row["event_type"] != "attempt":
        raise ContractValidationError("修订能力缺口没有引用来源 Attempt")
    source_attempt = validate_attempt_payload(
        _load_json(attempt_row["payload_json"], {}),
        allow_legacy_router=True,
    )
    if (
        source_attempt["combination_id"] != value["combination_id"]
        or source_attempt["workflow_version"] != row["workflow_version"]
        or source_attempt["execution_id"] != row["execution_id"]
        or source_attempt["router_decision"].get("decision")
        != "revise_workflow"
    ):
        raise ContractValidationError(
            "修订能力缺口的来源 Attempt 与当前工作流身份不一致"
        )
    return row


def _frozen_tool_gap_hypothesis(
    conn: sqlite3.Connection,
    task_id: str,
    row: sqlite3.Row,
    hypothesis_id: str,
) -> Mapping[str, Any]:
    """从组合所属批次中读取能力缺口对应的冻结假设。"""

    batch_event = conn.execute(
        "SELECT e.payload_json FROM hypothesis_batches b "
        "JOIN memory_events e ON e.task_id=b.task_id "
        "AND e.event_id=b.created_event_id "
        "WHERE b.task_id=? AND b.batch_id=? "
        "AND e.event_type='hypothesis_batch_created'",
        (task_id, row["batch_id"]),
    ).fetchone()
    if batch_event is None:
        raise ContractValidationError("ToolCapabilityGap 找不到冻结批次创建事实")
    batch_payload = _load_json(batch_event["payload_json"], {})
    phase = batch_payload.get("hypothesis_phase")
    if isinstance(phase, Mapping):
        ranked = phase.get("ranked_strategies")
    else:
        ranked = None
    hypotheses = [
        item
        for item in ranked or []
        if isinstance(item, Mapping) and item.get("hypothesis_id") == hypothesis_id
    ]
    if len(hypotheses) != 1:
        raise ContractValidationError(
            "ToolCapabilityGap 没有唯一对应的冻结 hypothesis"
        )
    return hypotheses[0]


def _apply_tool_capability_gap_projection(
    conn: sqlite3.Connection,
    task_id: str,
    event: Mapping[str, Any],
) -> None:
    """验证能力缺口的完整来源，并将组合投影为终态。"""

    value = validate_tool_capability_gap_payload(event["payload"])
    refs = validate_source_ref_shape(event["source_refs"])
    row = _tool_gap_source_combination(conn, task_id, value, refs)
    hypothesis = _frozen_tool_gap_hypothesis(
        conn,
        task_id,
        row,
        value["hypothesis_id"],
    )
    if "unmet_verification_requirements" in value["gap"]:
        requirements = validate_verification_requirements(
            hypothesis.get("verification_requirements"),
            path="ToolCapabilityGap frozen verification_requirements",
        )
        if any(
            item not in requirements
            for item in value["gap"]["unmet_verification_requirements"]
        ):
            raise ContractValidationError(
                "旧版 ToolCapabilityGap 引用了当前 hypothesis 之外的验证需求"
            )
    cursor = conn.execute(
        """
        UPDATE combinations SET state='tool_capability_gap',
            terminal_record_id=?, terminal_reason='tool_capability_gap',
            causal_source_ref=NULL, version=version+1, updated_at=?
        WHERE task_id=? AND combination_id=?
        """,
        (
            event["event_id"],
            event["created_at"],
            task_id,
            value["combination_id"],
        ),
    )
    if cursor.rowcount != 1:
        raise ContractValidationError("ToolCapabilityGap 无法更新来源组合")


def _apply_execution_step_completed_projection(
    conn: sqlite3.Connection,
    task_id: str,
    event: Mapping[str, Any],
    *,
    task_root: Path | None,
) -> None:
    """根据步骤完成清单推进组合的执行游标。"""

    payload = event["payload"]
    manifest = _event_manifest(
        event,
        task_root=task_root,
        required_key="step_manifest_refs",
    )
    if manifest is None:
        raise ContractValidationError(
            "ExecutionStepCompleted 缺少可重建的 step manifest"
        )
    prefix = manifest.get("step_manifest_refs")
    if (
        manifest.get("status") != "completed"
        or not isinstance(prefix, list)
        or manifest.get("step_index") != len(prefix)
        or manifest.get("step_index") != payload["step_index"]
        or manifest.get("operation_instance_id")
        != payload["operation_instance_id"]
        or manifest.get("execution_id") != payload["execution_id"]
        or manifest.get("workflow_version") != payload["workflow_version"]
        or manifest.get("step_id") != payload["step_id"]
        or manifest.get("causal_source_ref") != payload["causal_source_ref"]
    ):
        raise ContractValidationError(
            "步骤完成事实与 step manifest 身份不一致"
        )
    row = conn.execute(
        "SELECT * FROM combinations WHERE task_id=? AND combination_id=?",
        (task_id, payload["combination_id"]),
    ).fetchone()
    if row is None:
        raise ContractValidationError("步骤完成事实找不到来源组合")
    runtime = _load_json(row["runtime_checkpoint_json"], {})
    if (
        row["state"] != "running"
        or row["execution_id"] != payload["execution_id"]
        or row["workflow_version"] != payload["workflow_version"]
        or row["causal_source_ref"] != payload["causal_source_ref"]
        or runtime.get("step_manifest_refs", []) != prefix
        or runtime.get("next_step_index") != payload["step_index"]
    ):
        raise ContractValidationError("步骤完成事实不在预期执行游标上")
    workflow = _load_json(row["current_workflow_json"], {})
    steps = workflow.get("steps") if isinstance(workflow, Mapping) else None
    current_step = steps[payload["step_index"]] if isinstance(steps, list) and payload["step_index"] < len(steps) else None
    if manifest.get("action_id") == "generate_gaussian_code" and (
        not isinstance(current_step, Mapping)
        or current_step.get("action_id") != "generate_gaussian_code"
    ):
        raise ContractValidationError("Gaussian codegen manifest 与当前 workflow step 不一致")

    code_receipt = _gaussian_code_receipt_projection(
        manifest=manifest,
        manifest_ref=event["source_refs"][0],
        task_root=task_root,
    )
    runtime["step_manifest_refs"] = [*prefix, event["source_refs"][0]]
    runtime["next_step_index"] = payload["step_index"] + 1
    runtime["current_operation_instance_id"] = None
    runtime["current_request_ref"] = None
    runtime["checkpoint_id"] = None
    runtime["checkpoint_manifest_ref"] = None
    if code_receipt is not None:
        code_results = runtime.get("gaussian_code_results")
        if not isinstance(code_results, dict):
            code_results = {}
        code_results[str(code_receipt["step_id"])] = dict(code_receipt)
        runtime["gaussian_code_results"] = code_results
    conn.execute(
        """
        UPDATE combinations SET state='running', runtime_checkpoint_json=?,
            version=version+1, updated_at=?
        WHERE task_id=? AND combination_id=?
        """,
        (
            canonical_json(runtime),
            event["created_at"],
            task_id,
            payload["combination_id"],
        ),
    )


def _apply_execution_request_rebound_projection(
    conn: sqlite3.Connection,
    task_id: str,
    event: Mapping[str, Any],
) -> None:
    """重建 authored files 物化后的当前 claim request 引用。"""

    payload = event["payload"]
    row = conn.execute(
        "SELECT * FROM combinations WHERE task_id=? AND combination_id=?",
        (task_id, payload["combination_id"]),
    ).fetchone()
    if row is None:
        raise ContractValidationError("ExecutionRequestRebound 找不到来源组合")
    runtime = _load_json(row["runtime_checkpoint_json"], {})
    workflow = _load_json(row["current_workflow_json"], {})
    steps = workflow.get("steps") if isinstance(workflow, Mapping) else None
    step = steps[payload["step_index"]] if isinstance(steps, list) and payload["step_index"] < len(steps) else None
    if (
        row["state"] != "running"
        or row["execution_id"] != payload["execution_id"]
        or row["workflow_version"] != payload["workflow_version"]
        or not isinstance(step, Mapping)
        or step.get("step_id") != payload["step_id"]
        or runtime.get("next_step_index") != payload["step_index"]
        or runtime.get("checkpoint_id") is not None
    ):
        raise ContractValidationError(
            "ExecutionRequestRebound 不在预期的执行 claim 游标上"
        )
    current_operation = runtime.get("current_operation_instance_id")
    current_request = runtime.get("current_request_ref")
    if current_operation is None and current_request is None:
        # claim 本身是运行时 CAS，不单独写历史；若进程在 rebind 后才留下
        # 不可变事件，重建器从该事件恢复 claim 的最小身份再应用新 request。
        runtime["current_operation_instance_id"] = payload["operation_instance_id"]
        runtime["current_request_ref"] = payload["previous_request_ref"]
    elif (
        current_operation != payload["operation_instance_id"]
        or current_request != payload["previous_request_ref"]
    ):
        raise ContractValidationError(
            "ExecutionRequestRebound 的现有 claim 身份不一致"
        )
    runtime["current_request_ref"] = payload["request_ref"]
    conn.execute(
        """
        UPDATE combinations SET runtime_checkpoint_json=?, version=version+1,
            updated_at=?
        WHERE task_id=? AND combination_id=?
        """,
        (
            canonical_json(runtime),
            event["created_at"],
            task_id,
            payload["combination_id"],
        ),
    )


def _apply_execution_step_failed_projection(
    conn: sqlite3.Connection,
    task_id: str,
    event: Mapping[str, Any],
) -> None:
    """把失败步骤和后续未执行清单投影为可进入 Program Gate 的完整游标。"""

    payload = event["payload"]
    row = conn.execute(
        "SELECT * FROM combinations WHERE task_id=? AND combination_id=?",
        (task_id, payload["combination_id"]),
    ).fetchone()
    if row is None:
        raise ContractValidationError("步骤失败事实找不到来源组合")
    runtime = _load_json(row["runtime_checkpoint_json"], {})
    refs = payload.get("step_manifest_refs")
    if (
        row["state"] != "running"
        or row["execution_id"] != payload["execution_id"]
        or row["workflow_version"] != payload["workflow_version"]
        or runtime.get("next_step_index") != payload["step_index"]
        or runtime.get("current_operation_instance_id")
        not in {None, payload["operation_instance_id"]}
        or not isinstance(refs, list)
        or not refs
    ):
        raise ContractValidationError("步骤失败事实不在预期执行游标上")
    runtime["step_manifest_refs"] = list(refs)
    runtime["next_step_index"] = len(refs)
    runtime["current_operation_instance_id"] = None
    runtime["current_request_ref"] = None
    runtime["checkpoint_id"] = None
    runtime["checkpoint_manifest_ref"] = None
    runtime["execution_failure"] = {
        "step_id": payload["step_id"],
        "step_index": payload["step_index"],
        "failed_step_manifest_ref": payload["failed_step_manifest_ref"],
        "event_id": event["event_id"],
    }
    diagnostic = payload.get("failure_diagnostic")
    if isinstance(diagnostic, Mapping):
        runtime["execution_failure"]["diagnostic"] = deepcopy(dict(diagnostic))
    attempts = runtime.get("step_revision_attempts")
    if not isinstance(attempts, Mapping):
        attempts = {}
    attempts = {
        str(key): int(value)
        for key, value in attempts.items()
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
    }
    attempts[payload["step_id"]] = attempts.get(payload["step_id"], 0) + 1
    runtime["step_revision_attempts"] = attempts
    exhausted = attempts[payload["step_id"]] >= 5
    if exhausted:
        diagnostic_code = diagnostic.get("code") if isinstance(diagnostic, Mapping) else None
        runtime["terminal_classification"] = (
            "tool_capability_gap_after_retry_limit"
            if diagnostic_code in {
                "unregistered_action",
                "backend_unavailable",
                "native_software_unavailable",
                "unsupported_execution_layer",
                "missing_required_software",
            }
            else "hypothesis_failed"
        )
    conn.execute(
        """
        UPDATE combinations SET state=?, terminal_record_id=?, terminal_reason=?,
            runtime_checkpoint_json=?,
            version=version+1, updated_at=?
        WHERE task_id=? AND combination_id=?
        """,
        (
            "hypothesis_failed" if exhausted else "running",
            event["event_id"] if exhausted else None,
            "max_revision_attempts_exceeded" if exhausted else None,
            canonical_json(runtime),
            event["created_at"],
            task_id,
            payload["combination_id"],
        ),
    )


def _apply_execution_checkpointed_projection(
    conn: sqlite3.Connection,
    task_id: str,
    event: Mapping[str, Any],
    *,
    task_root: Path | None,
) -> None:
    """根据冻结检查点清单恢复外部作业等待状态。"""

    payload = event["payload"]
    checkpoint = _event_manifest(
        event,
        task_root=task_root,
        required_key="checkpoint_index",
    )
    if checkpoint is None:
        raise ContractValidationError(
            "ExecutionCheckpointed 缺少可重建的 checkpoint manifest"
        )
    request_ref = checkpoint.get("request_ref")
    if isinstance(request_ref, dict):
        request_ref = request_ref.get("relative_path")
    row = conn.execute(
        "SELECT runtime_checkpoint_json FROM combinations "
        "WHERE task_id=? AND combination_id=?",
        (task_id, payload["combination_id"]),
    ).fetchone()
    if row is None:
        raise ContractValidationError("ExecutionCheckpointed 找不到来源组合")
    previous_runtime = _load_json(row["runtime_checkpoint_json"], {})
    # checkpoint 只替换执行游标；当前工作流的专家审查及其 overrides 仍是
    # 同一运行时事实，必须随等待状态一起保留。工作流修订会先用空游标重置
    # 这些字段，因此这里不会把已被新版本清除的审查状态带入新工作流。
    runtime = (
        deepcopy(dict(previous_runtime))
        if isinstance(previous_runtime, Mapping)
        else {}
    )
    runtime.update(
        {
            "checkpoint_id": event["event_id"],
            "checkpoint_index": payload["checkpoint_index"],
            "checkpoint_manifest_ref": event["source_refs"][0],
            "current_operation_instance_id": payload["operation_instance_id"],
            "current_request_ref": request_ref,
            "next_step_index": payload["step_index"],
            "step_manifest_refs": list(checkpoint.get("step_manifest_refs", [])),
        }
    )
    conn.execute(
        """
        UPDATE combinations SET state='waiting_execution',
            runtime_checkpoint_json=?, version=version+1, updated_at=?
        WHERE task_id=? AND combination_id=?
        """,
        (
            canonical_json(runtime),
            event["created_at"],
            task_id,
            payload["combination_id"],
        ),
    )


def _apply_execution_resumed_projection(
    conn: sqlite3.Connection,
    task_id: str,
    event: Mapping[str, Any],
) -> None:
    """将已恢复的外部作业重新投影为运行态。"""

    payload = event["payload"]
    conn.execute(
        """
        UPDATE combinations SET state='running', version=version+1, updated_at=?
        WHERE task_id=? AND combination_id=?
        """,
        (event["created_at"], task_id, payload["combination_id"]),
    )


def _apply_attempt_projection(
    conn: sqlite3.Connection,
    task_id: str,
    event: Mapping[str, Any],
    *,
    task_root: Path | None,
) -> None:
    """根据一次完整科学尝试更新组合终态或修订状态。"""

    payload = event["payload"]
    execution_manifest = _event_manifest(
        event,
        task_root=task_root,
        required_key="step_manifest_refs",
    )
    if execution_manifest is None:
        raise ContractValidationError("Attempt 缺少可重建的 execution manifest")
    step_refs = list(execution_manifest["step_manifest_refs"])
    row = conn.execute(
        "SELECT runtime_checkpoint_json FROM combinations "
        "WHERE task_id=? AND combination_id=?",
        (task_id, payload["combination_id"]),
    ).fetchone()
    if row is None:
        raise ContractValidationError("Attempt 找不到来源组合")
    previous_runtime = _load_json(row["runtime_checkpoint_json"], {})
    runtime = empty_runtime_checkpoint()
    # Attempt 会重置执行游标，但不得丢失同一执行中已经记录的专家审查、debate
    # 或暂停事实；这些字段是重建后仍需向调度器和研究员暴露的历史结果。
    for key in ("expert_review", "expert_debate", "expert_pause"):
        if isinstance(previous_runtime.get(key), Mapping):
            runtime[key] = deepcopy(dict(previous_runtime[key]))
    runtime["step_manifest_refs"] = step_refs
    runtime["next_step_index"] = len(step_refs)

    decision = payload["router_decision"]["decision"]
    if decision == "evidence_ready":
        state = "evidence_ready"
    elif decision == "revise_hypothesis" or payload["terminal_reason"] == (
        "workflow_revision_limit_exhausted"
    ):
        state = "hypothesis_failed"
    else:
        state = "revising_workflow"
    if state in {"evidence_ready", "hypothesis_failed"}:
        terminal_record_id = event["event_id"]
    else:
        terminal_record_id = None

    conn.execute(
        """
        UPDATE combinations SET state=?, runtime_checkpoint_json=?,
            terminal_record_id=?, terminal_reason=?, causal_source_ref=NULL,
            version=version+1, updated_at=?
        WHERE task_id=? AND combination_id=?
        """,
        (
            state,
            canonical_json(runtime),
            terminal_record_id,
            payload["terminal_reason"],
            event["created_at"],
            task_id,
            payload["combination_id"],
        ),
    )


def _apply_execution_paused_projection(
    conn: sqlite3.Connection,
    task_id: str,
    event: Mapping[str, Any],
    *,
    task_root: Path | None,
) -> None:
    """保留已完成执行前缀，并将组合投影为暂停态。"""

    payload = event["payload"]
    row = conn.execute(
        "SELECT runtime_checkpoint_json FROM combinations "
        "WHERE task_id=? AND combination_id=?",
        (task_id, payload["combination_id"]),
    ).fetchone()
    if row is None:
        raise ContractValidationError("Pause 找不到来源组合")
    runtime = _load_json(row["runtime_checkpoint_json"], {})
    if payload["operation"] == "execute_step":
        pause_manifest = _event_manifest(
            event,
            task_root=task_root,
            required_key="step_manifest_refs",
        )
        if pause_manifest is None:
            raise ContractValidationError("工具技术暂停缺少可重建的步骤前缀")
        prefix = pause_manifest["step_manifest_refs"]
        if (
            not isinstance(prefix, list)
            or pause_manifest.get("step_index") != len(prefix)
        ):
            raise ContractValidationError(
                "技术暂停清单中的步骤前缀与游标不一致"
            )
        runtime["step_manifest_refs"] = list(prefix)
        runtime["next_step_index"] = len(prefix)
        runtime["current_operation_instance_id"] = None
        runtime["current_request_ref"] = None
        runtime["checkpoint_id"] = None
        runtime["checkpoint_manifest_ref"] = None
    conn.execute(
        """
        UPDATE combinations SET state='paused', runtime_checkpoint_json=?,
            version=version+1, updated_at=?
        WHERE task_id=? AND combination_id=?
        """,
        (
            canonical_json(runtime),
            event["created_at"],
            task_id,
            payload["combination_id"],
        ),
    )


def _apply_restart_projection(
    conn: sqlite3.Connection,
    task_id: str,
    event: Mapping[str, Any],
) -> None:
    """按人工恢复动作重建任务或组合的继续执行位置。"""

    payload = event["payload"]
    pause = _restart_source_pause(
        conn,
        task_id=task_id,
        event=event,
    )
    combination_id = pause["combination_id"]
    if combination_id is None:
        # 任务级恢复用于重新生成首批假设；下一条批次创建事实将证明这份授权已使用。
        conn.execute(
            "UPDATE tasks SET status='running', version=version+1, updated_at=? "
            "WHERE task_id=?",
            (event["created_at"], task_id),
        )
        return

    row = conn.execute(
        "SELECT runtime_checkpoint_json FROM combinations "
        "WHERE task_id=? AND combination_id=?",
        (task_id, combination_id),
    ).fetchone()
    if row is None:
        raise ContractValidationError("Restart 找不到来源组合")
    runtime = _load_json(row["runtime_checkpoint_json"], {})
    for field in (
        "current_operation_instance_id",
        "current_request_ref",
        "checkpoint_id",
        "checkpoint_manifest_ref",
    ):
        runtime[field] = None

    action = payload["action"]
    if action == "revise_hypothesis":
        conn.execute(
            """
            UPDATE combinations SET state='hypothesis_failed',
                terminal_record_id=?, terminal_reason='human_revise_hypothesis',
                causal_source_ref=NULL, version=version+1, updated_at=?
            WHERE task_id=? AND combination_id=?
            """,
            (event["event_id"], event["created_at"], task_id, combination_id),
        )
        return

    if action == "restart_workflow":
        runtime = empty_runtime_checkpoint()
        execution_id = payload["execution_id"]
        state = "running"
    elif action == "retry_failed_operation":
        execution_id = None
        state_by_operation = {
            "workflow_generation": "generating_workflow",
            "router": "routing",
            "workflow_revision": "revising_workflow",
            "expert_gaussian_review": "running",
            "expert_debate": "routing",
        }
        state = state_by_operation[pause["operation"]]
    else:
        execution_id = None
        state = "running"

    # patch_and_continue 的新工作流稍后由正式修订记录恢复；这里先恢复共同的
    # 运行位置和人工授权。其他动作则完全沿用暂停前的工作流与执行身份。
    conn.execute(
        """
        UPDATE combinations SET state=?,
            execution_id=COALESCE(?, execution_id),
            runtime_checkpoint_json=?, terminal_record_id=NULL,
            terminal_reason=NULL, causal_source_ref=?,
            version=version+1, updated_at=?
        WHERE task_id=? AND combination_id=?
        """,
        (
            state,
            execution_id,
            canonical_json(runtime),
            event["event_id"],
            event["created_at"],
            task_id,
            combination_id,
        ),
    )


def _apply_batch_closed_projection(
    conn: sqlite3.Connection,
    task_id: str,
    event: Mapping[str, Any],
) -> None:
    """关闭批次并记录每个组合的最终结果路径。"""

    payload = event["payload"]
    batch_id_row = conn.execute(
        "SELECT batch_id FROM hypothesis_batches WHERE task_id=? AND batch_index=?",
        (task_id, payload["batch_index"]),
    ).fetchone()
    if batch_id_row is None:
        raise ContractValidationError("BatchClosed 找不到来源批次")
    conn.execute(
        "UPDATE hypothesis_batches SET state=?, version=version+1, updated_at=? "
        "WHERE task_id=? AND batch_id=?",
        (
            payload["batch_outcome"],
            event["created_at"],
            task_id,
            batch_id_row["batch_id"],
        ),
    )
    for result in payload["combination_results"]:
        conn.execute(
            "UPDATE combinations SET final_result_relative_path=?, updated_at=? "
            "WHERE task_id=? AND combination_id=?",
            (
                result["final_result_relative_path"],
                event["created_at"],
                task_id,
                result["combination_id"],
            ),
        )


def _apply_task_closed_projection(
    conn: sqlite3.Connection,
    task_id: str,
    event: Mapping[str, Any],
) -> None:
    """根据成功结果是否存在设置任务终态。"""

    if event["payload"]["successful_result_relative_paths"]:
        task_status = "succeeded"
    else:
        task_status = "stopped_no_supported_hypothesis"
    conn.execute(
        "UPDATE tasks SET status=?, updated_at=? WHERE task_id=?",
        (task_status, event["created_at"], task_id),
    )


def _apply_investigation_history_event_projection(
    conn: sqlite3.Connection,
    task_id: str,
    event: Mapping[str, Any],
) -> bool:
    """按历史事件类型重建 Investigation v8 当前索引。"""

    payload = event["payload"]
    event_type = event["event_type"]
    common = {
        "conn": conn,
        "task_id": task_id,
        "event_id": event["event_id"],
        "payload": payload,
        "created_at": event["created_at"],
    }
    if event_type == "investigation_evidence_verified":
        insert_investigation_evidence_verification_projection(**common)
    elif event_type == "task_review_session_started":
        insert_task_review_session_projection(**common)
    elif event_type == "task_review_turn_started":
        insert_task_review_turn_projection(**common)
    elif event_type == "investigation_repair_turn_started":
        insert_investigation_repair_turn_started_projection(**common)
    elif event_type == "investigation_session_started":
        insert_investigation_session_projection(**common)
    elif event_type == "investigation_session_thread_bound":
        insert_investigation_session_thread_bound_projection(**common)
    elif event_type == "investigation_session_thread_migrated":
        insert_investigation_session_thread_migrated_projection(**common)
    elif event_type == "investigation_turn_started":
        insert_investigation_turn_started_projection(**common)
    elif event_type == "investigation_turn_waiting":
        insert_investigation_turn_waiting_projection(**common)
    elif event_type == "investigation_turn_attempt_terminated":
        insert_investigation_turn_attempt_termination_projection(**common)
    elif event_type == "investigation_legacy_waiting_resume_authorized":
        # 该事件只保存一次性迁移授权，不拥有可变当前索引；重建时仍须重新校验正文。
        validate_investigation_legacy_waiting_resume_authorization_payload(payload)
    elif event_type == "investigation_incomplete_v2_waiting_resume_authorized":
        # 该事件只保存残缺 v2 waiting 的显式恢复授权，不更新可变当前索引；重建时仍须重新校验正文。
        validate_investigation_incomplete_v2_waiting_resume_authorization_payload(payload)
    elif event_type == "investigation_result_recorded":
        insert_investigation_result_projection(**common)
    elif event_type == "investigation_batch_reviewed":
        vector = validate_investigation_batch_review_payload(payload)[
            "result_version_vector"
        ]
        digest = hashlib.sha256(canonical_json(vector).encode("utf-8")).hexdigest()
        insert_investigation_review_projection(**common, idempotency_key=digest)
    elif event_type == "investigation_repair_requested":
        insert_investigation_repair_requested_projection(**common)
    elif event_type == "investigation_repair_completed":
        insert_investigation_repair_completed_projection(**common)
    elif event_type == "investigation_route_marked":
        insert_investigation_route_marker_projection(**common)
    elif event_type == "investigation_task_closed":
        insert_investigation_task_terminal_projection(**common)
    elif event_type == "investigation_task_correction_recorded":
        insert_investigation_task_correction_projection(**common)
    elif event_type == "hypothesis_selection_completed":
        value = validate_hypothesis_selection_completed_payload(payload)
        conn.execute(
            "UPDATE tasks SET status='succeeded', final_outcome=?, version=version+1, "
            "updated_at=? WHERE task_id=?",
            (canonical_json(value), event["created_at"], task_id),
        )
    elif event_type == "investigation_task_failed":
        _apply_investigation_task_failure_projection(conn, task_id, event)
    else:
        return False
    return True


def _apply_task_created_history_projection(
    conn: sqlite3.Connection,
    task_id: str,
    event: Mapping[str, Any],
) -> None:
    """按 TaskCreated 事实恢复任务科学上下文和专家启动失败状态。"""

    payload = event["payload"]
    task_created = validate_task_created_payload(payload)
    chemistry_context = task_created["chemistry_context"]
    reaction_class = chemistry_context.get("reaction_type") or chemistry_context.get(
        "reaction_class"
    )
    conn.execute(
        "UPDATE tasks SET scientific_question=?,reaction_class=?,"
        "chemistry_context_json=?,stage_architecture_version=? WHERE task_id=?",
        (
            task_created["scientific_question"],
            reaction_class,
            canonical_json(chemistry_context),
            task_created["stage_architecture_version"],
            task_id,
        ),
    )
    selection = payload.get("expert_backend_selection")
    if isinstance(selection, Mapping) and selection.get("status") == "startup_failed":
        conn.execute(
            "UPDATE tasks SET status='failed', updated_at=? WHERE task_id=?",
            (event["created_at"], task_id),
        )


def _apply_task_setup_history_event_projection(
    conn: sqlite3.Connection,
    task_id: str,
    event: Mapping[str, Any],
) -> bool:
    """按任务创建、批次创建和 Retrieval 事件恢复任务级索引。"""

    event_type = event["event_type"]
    payload = event["payload"]
    common = {
        "conn": conn,
        "task_id": task_id,
        "event_id": event["event_id"],
        "payload": payload,
        "created_at": event["created_at"],
    }
    if event_type == "task_created":
        _apply_task_created_history_projection(conn, task_id, event)
    elif event_type == "hypothesis_batch_created":
        insert_batch_projection(**common)
    elif event_type == "retrieval_paused":
        conn.execute(
            "UPDATE tasks SET status='paused', updated_at=? WHERE task_id=?",
            (event["created_at"], task_id),
        )
    elif event_type == "retrieval_completed":
        validate_retrieval_completed_payload(payload)
        if payload.get("causal_source_ref") is not None:
            conn.execute(
                "UPDATE tasks SET status='running', updated_at=? "
                "WHERE task_id=? AND status='paused'",
                (event["created_at"], task_id),
            )
    else:
        return False
    return True


def _apply_planning_expert_history_event_projection(
    conn: sqlite3.Connection,
    task_id: str,
    event: Mapping[str, Any],
) -> bool:
    """按 Planning、专家阶段和工具缺口事实恢复组合索引。"""

    event_type = event["event_type"]
    if event_type == "planning_completed":
        _apply_planning_completed_projection(conn, task_id, event)
    elif event_type == "expert_gaussian_review":
        _apply_expert_gaussian_review_projection(conn, task_id, event)
    elif event_type == "expert_debate_completed":
        _apply_expert_debate_projection(conn, task_id, event)
    elif event_type == "expert_model_paused":
        _apply_expert_pause_projection(conn, task_id, event)
    elif event_type == "tool_capability_gap":
        _apply_tool_capability_gap_projection(conn, task_id, event)
    else:
        return False
    return True


def _apply_execution_progress_history_event_projection(
    conn: sqlite3.Connection,
    task_id: str,
    event: Mapping[str, Any],
    *,
    task_root: Path | None,
) -> bool:
    """按执行进度事实恢复组合运行状态和 checkpoint。"""

    event_type = event["event_type"]
    if event_type == "execution_step_completed":
        _apply_execution_step_completed_projection(
            conn, task_id, event, task_root=task_root
        )
    elif event_type == "execution_request_rebound":
        _apply_execution_request_rebound_projection(conn, task_id, event)
    elif event_type == "execution_step_failed":
        _apply_execution_step_failed_projection(conn, task_id, event)
    elif event_type == "execution_checkpointed":
        _apply_execution_checkpointed_projection(
            conn, task_id, event, task_root=task_root
        )
    elif event_type == "execution_resumed":
        _apply_execution_resumed_projection(conn, task_id, event)
    elif event_type == "attempt":
        _apply_attempt_projection(conn, task_id, event, task_root=task_root)
    else:
        return False
    return True


def _apply_execution_pause_history_event_projection(
    conn: sqlite3.Connection,
    task_id: str,
    event: Mapping[str, Any],
    *,
    task_root: Path | None,
) -> bool:
    """按执行暂停和重启事实恢复组合运行位置。"""

    event_type = event["event_type"]
    if event_type == "execution_paused":
        if event["payload"]["combination_id"] is None:
            return False
        _apply_execution_paused_projection(
            conn,
            task_id,
            event,
            task_root=task_root,
        )
    elif event_type == "restart_applied":
        _apply_restart_projection(conn, task_id, event)
    else:
        return False
    return True


def _apply_batch_terminal_history_event_projection(
    conn: sqlite3.Connection,
    task_id: str,
    event: Mapping[str, Any],
) -> bool:
    """按批次关闭和旧版任务关闭事实恢复终态字段。"""

    event_type = event["event_type"]
    if event_type == "hypothesis_batch_closed":
        _apply_batch_closed_projection(conn, task_id, event)
    elif event_type == "task_closed":
        _apply_task_closed_projection(conn, task_id, event)
    else:
        return False
    return True


def _apply_rebuilt_event_projection(
    conn: sqlite3.Connection,
    task_id: str,
    event: Mapping[str, Any],
    *,
    task_root: Path | None,
) -> None:
    """按固定阶段顺序把一条历史事件应用到当前投影。"""

    if _apply_investigation_history_event_projection(conn, task_id, event):
        return
    if _apply_task_setup_history_event_projection(conn, task_id, event):
        return
    if _apply_planning_expert_history_event_projection(conn, task_id, event):
        return
    if _apply_execution_progress_history_event_projection(
        conn, task_id, event, task_root=task_root
    ):
        return
    _apply_execution_pause_history_event_projection(
        conn, task_id, event, task_root=task_root
    )
    _apply_batch_terminal_history_event_projection(conn, task_id, event)


# ==================== 公开重建与读取入口 ====================

def rebuild_task_projections(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    task_root: Path | None = None,
) -> None:
    """删除指定任务的当前视图，再按历史事件和修订逐条重算。

    重建不读取旧视图中的业务内容，因此能验证恢复所需信息确实存在于不可修改
    的事实记录中。
    """

    history = _projection_history_nodes(
        conn,
        task_id,
        task_root=task_root,
    )

    _clear_task_projection_rows(conn, task_id)
    clear_investigation_projection_rows(conn, task_id)

    for node in history:
        if node["kind"] == "revision":
            _apply_revision_projection(conn, task_id, node["record"])
            continue
        _apply_rebuilt_event_projection(
            conn,
            task_id,
            node["record"],
            task_root=task_root,
        )


def list_projection_rows(rows: List[sqlite3.Row], converter) -> List[Dict[str, Any]]:
    """批量复制数据库行，避免调用方持有与连接生命周期绑定的对象。"""

    return [converter(row) for row in rows]
