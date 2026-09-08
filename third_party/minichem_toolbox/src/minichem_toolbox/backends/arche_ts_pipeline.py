"""Bounded transition-state path search orchestration."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping
from importlib import resources as package_resources
from typing import Any, Callable

from .common import (
    atoms_and_coordinates,
    output_directory,
    partial_success,
    relative_workspace_path,
    request_parts,
    resolve_input_file,
    structure_dict,
    success,
    unavailable,
    unsupported,
    write_json,
    write_xyz,
)


_AUTHOR_SINGLE_BOND_LENGTHS_ANGSTROM = {
    ("C", "C"): 1.54,
    ("C", "H"): 1.09,
    ("C", "N"): 1.47,
    ("C", "O"): 1.43,
    ("H", "H"): 0.74,
    ("H", "N"): 1.01,
    ("H", "O"): 0.96,
    ("N", "N"): 1.45,
    ("N", "O"): 1.41,
    ("O", "O"): 1.48,
}


ACTION_ID = "search_transition_state_path"
BACKEND_ID = "transition_state_path_search"
_ACCEPTED = {"success", "partial_success"}
_DEFAULT_CONFIG_SHA256 = "4d163abebf48d42498efe1c093cee4b94ba999bf5b93bf1e0b516353809bbdc7"
_DEFAULT_CONFIG_RESOURCE = "data/transition_state_path_defaults.json"
_MAX_COMBINATIONS = 1924
_MAX_ENDPOINT_PAIRS = 19240
_INTERMEDIATE_IMAGES = 7
_TOTAL_PATH_IMAGES = _INTERMEDIATE_IMAGES + 2
_MAX_PATH_IMAGES = _MAX_ENDPOINT_PAIRS * _TOTAL_PATH_IMAGES
_TRANSITION_STATE_GUESS_SELECTION_RULE_VERSION = "ts-guess-selection.v1"
_TRANSITION_STATE_PIPELINE_PROTOCOL_VERSION = "ts-pipeline-protocol.v1"
_TRANSITION_STATE_PIPELINE_MODULE = "minichem_toolbox.backends.ash_neb"
_TRANSITION_STATE_PIPELINE_PROTOCOL = "search_reaction_path/ash_neb_uma"
_TRANSITION_STATE_GUESS_SOURCES = frozenset(
    {
        "knarr_saddle_xyz",
        "ash_saddle_geometry",
        "highest_interior_path_image",
    }
)
_ASH_MAX_FORCE = 0.026
_ASH_RMS_FORCE = 0.013
_ASH_IDPP_MAX_ITERATIONS = 700
_ASH_RANDOM_SEED = 41
_SAFE_STAGE_ERROR_CODES = frozenset(
    {
        "backend_input_error",
        "backend_unavailable",
        "component_backend_unavailable",
        "evaluator_controlled_timeout",
        "invalid_request",
        "invalid_resource_limits",
        "aggregate_resource_budget_exceeded",
        "batch_resource_capacity_unavailable",
        "resource_budget_exceeded",
        "unsupported_backend",
    }
)
_SAFE_PROVENANCE_VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+:-]{0,127}$")


Executor = Callable[[str, str, dict[str, Any]], dict[str, Any]]


class StageUnavailable(RuntimeError):
    def __init__(self, action_id: str, backend_id: str, response: dict[str, Any]):
        self.action_id = action_id
        self.backend_id = backend_id
        self.response = response
        # 上游错误可能带命令、路径或固定反应物；这些内容只保留在服务端对象中，
        # 异常文本统一使用不可识别的阶段原因，避免进入 MCP 响应。
        super().__init__("transition-state path stage unavailable")


def _stage_failure_code(response: Mapping[str, Any], *, fallback_code: str) -> str:
    """从嵌套阶段响应提取白名单错误码，不读取异常正文或本机路径。"""

    error = response.get("error")
    raw_code = error.get("code") if isinstance(error, Mapping) else None
    if isinstance(raw_code, str) and raw_code in _SAFE_STAGE_ERROR_CODES:
        return raw_code
    if response.get("status") == "invalid_request":
        return "backend_input_error"
    if response.get("status") == "unsupported":
        return "unsupported_backend"
    return fallback_code


def _safe_stage_response(
    *,
    status: str,
    code: str,
    retryable: bool,
    diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """构造带有受控诊断的 StageUnavailable 响应。

    只有 ASH/KNARR 阶段明确提供的 message/warnings 会由调用方传入；这些字段在
    传入前已经去除私有路径和异常原文，其他阶段仍只返回安全错误分类。
    """

    safe_status = (
        status
        if status in {"invalid_request", "unavailable", "failed", "unsupported"}
        else "unavailable"
    )
    response: dict[str, Any] = {
        "status": safe_status,
        "error": {"code": code},
        "retryable": bool(retryable),
    }
    if diagnostics:
        response.update(diagnostics)
    return response


def _safe_nested_stage_diagnostics(
    response: Mapping[str, Any], *, backend_id: str, status: str
) -> dict[str, Any]:
    """保留 ASH partial/failure 的可行动诊断并移除私有信息。

    输入是嵌套 Action 的公开响应；只读取 ASH/KNARR 在非成功状态下的 error.message
    和 warnings，借用服务层统一的路径、异常字段和长度清理规则。其他 Backend 或
    成功响应不会把原始诊断提升到复合流程。
    """

    if backend_id != "ash_neb_uma" or status not in {
        "partial_success",
        "failed",
        "unavailable",
    }:
        return {}
    try:
        from ..service import _sanitize_public_diagnostic_value
    except (ImportError, ModuleNotFoundError):
        return {}

    diagnostics: dict[str, Any] = {}
    error = response.get("error")
    message = error.get("message") if isinstance(error, Mapping) else None
    if isinstance(message, str) and message.strip():
        safe_message = _sanitize_public_diagnostic_value(message)
        if isinstance(safe_message, str) and safe_message.strip():
            diagnostics["message"] = safe_message
    raw_warnings = response.get("warnings")
    if raw_warnings:
        warnings = raw_warnings if isinstance(raw_warnings, (list, tuple)) else [raw_warnings]
        safe_warnings = _sanitize_public_diagnostic_value(warnings)
        if isinstance(safe_warnings, list):
            diagnostics["warnings"] = safe_warnings
    return diagnostics


def _safe_provenance_version(value: Any) -> str | None:
    """只接受有限字符集的版本标识，拒绝路径、异常文本和任意长字符串。"""

    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if _SAFE_PROVENANCE_VERSION_PATTERN.fullmatch(normalized) is None:
        return None
    return normalized


def _normalize_pipeline_provenance(
    value: Mapping[str, Any] | None,
    *,
    selection_rule_version: Any = None,
) -> dict[str, Any] | None:
    """把 ASH/KNARR 来源压缩成无路径、无密钥的固定来源记录。

    输入可以是嵌套 ASH Action 的 provenance 或已经写入 reaction_path 的记录；
    只保留固定模块/协议、版本、选择规则和少量布尔科学事实，拒绝任意上游字段。
    没有合法 backend_id 时返回 ``None``，使不完整或非 ASH 路径不会伪造来源。
    """

    if not isinstance(value, Mapping) or value.get("backend_id") != "ash_neb_uma":
        return None
    backend_version = _safe_provenance_version(value.get("backend_version"))
    module_version = _safe_provenance_version(
        value.get("module_version") or backend_version
    )
    selection = value.get("selection_rule_version", selection_rule_version)
    selection = _safe_provenance_version(selection)
    normalized: dict[str, Any] = {
        "schema_version": _TRANSITION_STATE_PIPELINE_PROTOCOL_VERSION,
        "backend_id": "ash_neb_uma",
        "backend_version": backend_version,
        "module": _TRANSITION_STATE_PIPELINE_MODULE,
        "module_version": module_version,
        "protocol": _TRANSITION_STATE_PIPELINE_PROTOCOL,
        "protocol_version": _TRANSITION_STATE_PIPELINE_PROTOCOL_VERSION,
        "selection_rule_version": selection,
    }
    for field in ("engine", "calculator"):
        declared = value.get(field)
        if isinstance(declared, str) and declared in {
            "ASH/KNARR NEB",
            "ASH FairchemTheory",
        }:
            normalized[field] = declared
    for field in (
        "path_identity_metadata_restored",
        "path_is_validated_transition_state",
        "endpoint_atom_order_checked",
        "endpoint_atom_maps_checked",
    ):
        declared = value.get(field)
        if isinstance(declared, bool):
            normalized[field] = declared
    return normalized


def _safe_nested_stage_provenance(
    response: Mapping[str, Any],
    *,
    action_id: str,
    backend_id: str,
    result: Mapping[str, Any],
) -> dict[str, Any] | None:
    """提取嵌套 ASH Action 的安全来源版本，供 trace、pair 和 reaction_path 复用。"""

    if backend_id != "ash_neb_uma" or action_id != "search_reaction_path":
        return None
    declared = response.get("provenance")
    nested = declared if isinstance(declared, Mapping) else {}
    payload: dict[str, Any] = {
        "backend_id": backend_id,
        "backend_version": response.get("backend_version"),
        "selection_rule_version": result.get(
            "transition_state_guess_selection_rule_version"
        ),
    }
    for field in (
        "engine",
        "calculator",
        "path_identity_metadata_restored",
        "path_is_validated_transition_state",
        "endpoint_atom_order_checked",
        "endpoint_atom_maps_checked",
    ):
        if field in nested:
            payload[field] = nested[field]
    return _normalize_pipeline_provenance(payload)


def _normalize_nested_exception(exc: Exception) -> dict[str, Any]:
    """把子 Action 异常归类为安全错误，不传播异常类型、文本或路径。"""

    if isinstance(exc, (ValueError, KeyError, TypeError)):
        return _safe_stage_response(
            status="invalid_request",
            code="backend_input_error",
            retryable=False,
        )
    return _safe_stage_response(
        status="unavailable",
        code="backend_unavailable",
        retryable=True,
    )


def _is_required_stage_unavailable(response: Mapping[str, Any]) -> bool:
    """判断早期准备阶段是否因服务能力而不可用，而非候选数据无效。"""

    return _stage_failure_code(
        response,
        fallback_code="backend_unavailable",
    ) in {
        "backend_unavailable",
        "component_backend_unavailable",
        "unsupported_backend",
        "invalid_resource_limits",
        "resource_budget_exceeded",
        "aggregate_resource_budget_exceeded",
    }


def _structured_required_stage_failure(exc: StageUnavailable) -> dict[str, Any]:
    """把早期必需阶段的安全异常收口为 Action 响应。"""

    reason = _safe_stage_failure_reason(
        exc.response,
        fallback_code="backend_unavailable",
    )
    result = unavailable("Required transition-state path stage unavailable")
    result["retryable"] = reason["retryable"]
    result["error"] = {"code": reason["reason_code"]}
    return result


def _load_server_configuration() -> dict[str, Any]:
    """加载并校验服务端默认配置；隐藏字段只在服务端内存中使用。"""

    try:
        resource = package_resources.files("minichem_toolbox").joinpath(
            _DEFAULT_CONFIG_RESOURCE
        )
        payload = resource.read_bytes()
        if hashlib.sha256(payload).hexdigest() != _DEFAULT_CONFIG_SHA256:
            raise ValueError("configuration integrity mismatch")
        configuration = json.loads(payload.decode("utf-8"))
        required_keys = {
            "direction",
            "main_force_field",
            "target_force_field",
            "seed",
            "conformer_rmsd_threshold",
            "conformer_max_attempts",
            "conformer_max_iter",
            "complex_max_attempts",
            "complex_initial_distance",
            "equilibrium_ratio",
            "min_equilibrium_ratio",
            "max_equilibrium_ratio",
            "max_allowed_ratio",
            "force_constant",
            "filter_rmsd_threshold",
            "reorder",
            "charge",
            "multiplicity",
            "opt_fmax",
            "opt_steps",
            "task_name",
            "num_images",
            "climb",
            "neb_steps",
            "rmsd_threshold",
            "neb_fmax",
        }
        if not isinstance(configuration, dict) or not required_keys <= configuration.keys():
            raise ValueError("configuration shape is invalid")
        return copy.deepcopy(configuration)
    except (OSError, UnicodeError, TypeError, ValueError, KeyError) as exc:
        raise RuntimeError("Transition-state path service configuration unavailable") from exc


def _server_settings() -> dict[str, Any]:
    """把默认配置转成内部 Action 参数，不向公开合同暴露隐藏控制。"""

    configuration = _load_server_configuration()
    return {
        "direction": configuration["direction"],
        "random_seed": int(configuration["seed"]),
        "conformer_prune_rms_threshold_angstrom": float(
            configuration["conformer_rmsd_threshold"]
        ),
        "conformer_max_attempts": int(configuration["conformer_max_attempts"]),
        "component_mmff_max_iterations": int(configuration["conformer_max_iter"]),
        "complex_max_attempts": int(configuration["complex_max_attempts"]),
        "complex_initial_distance": float(configuration["complex_initial_distance"]),
        "equilibrium_distance_scale": float(configuration["equilibrium_ratio"]),
        "minimum_distance_ratio": float(configuration["min_equilibrium_ratio"]),
        "maximum_distance_ratio": float(configuration["max_equilibrium_ratio"]),
        "maximum_allowed_ratio": float(configuration["max_allowed_ratio"]),
        "constraint_force_constant_kcal_mol_angstrom2": float(
            configuration["force_constant"]
        ),
        "filter_rmsd_threshold": float(configuration["filter_rmsd_threshold"]),
        "reorder": bool(configuration["reorder"]),
        "charge": int(configuration["charge"]),
        "multiplicity": int(configuration["multiplicity"]),
        "uma_fmax_ev_per_angstrom": float(configuration["opt_fmax"]),
        "uma_max_steps": int(configuration["opt_steps"]),
        "uma_model": "uma-s-1p1",
        "uma_task_name": configuration["task_name"],
        # ASH 的 images 表示中间图像，路径总帧数还包括两个端点。
        "intermediate_images": int(configuration["num_images"]),
        "total_images": int(configuration["num_images"]) + 2,
        "climb": bool(configuration["climb"]),
        "neb_max_steps": int(configuration["neb_steps"]),
        # 作者活动代码只把该字段传入函数但未传给 ASH；这里保留为服务端审计
        # 注释，实际 ASH 使用其默认阈值和 IDPP 设置。
        "unused_source_rmsd_threshold": float(configuration["rmsd_threshold"]),
        "unused_source_neb_fmax": float(configuration["neb_fmax"]),
        "main_force_field": str(configuration["main_force_field"]),
        "target_force_field": str(configuration["target_force_field"]),
    }


def _validate_public_inputs(inputs: Mapping[str, Any]) -> tuple[dict[str, list[str]], str, str]:
    """Validate the four model-visible fields and reject every hidden control."""

    allowed = {"reactants", "products", "main_force_field", "target_force_field"}
    unknown = sorted(set(inputs) - allowed)
    if unknown:
        raise ValueError(f"Unsupported inputs field(s): {unknown}")

    def smiles_list(name: str) -> list[str]:
        value = inputs.get(name)
        if (
            not isinstance(value, list)
            or not value
            or not all(isinstance(item, str) and item.strip() for item in value)
        ):
            raise ValueError(f"inputs.{name} must be a non-empty array of SMILES strings")
        return [item.strip() for item in value]

    defaults = _load_server_configuration()
    main_force_field = inputs.get("main_force_field", defaults["main_force_field"])
    target_force_field = inputs.get("target_force_field", defaults["target_force_field"])
    if main_force_field != "MMFF":
        raise ValueError("inputs.main_force_field currently supports only 'MMFF'")
    if target_force_field != "UMA":
        raise ValueError("inputs.target_force_field currently supports only 'UMA'")
    reactants = smiles_list("reactants")
    products = smiles_list("products")
    if len(reactants) < 2:
        raise ValueError(
            "Transition-state path search requires at least two reactant components"
        )
    return (
        {"reactants": reactants, "products": products},
        main_force_field,
        target_force_field,
    )


def _positive_integer(value: Any, label: str, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    if value > maximum:
        raise ValueError(f"{label} exceeds hard limit {maximum}")
    return value


def _finite_positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def _call(
    executor: Executor,
    trace: list[dict[str, Any]],
    action_id: str,
    backend_id: str,
    *,
    inputs: dict[str, Any],
    method_spec: dict[str, Any] | None = None,
    action_settings: dict[str, Any] | None = None,
    resource_limits: dict[str, Any] | None = None,
) -> dict[str, Any]:
    request = {
        "inputs": inputs,
        "method_spec": method_spec or {},
        "action_settings": action_settings or {},
        "resource_limits": resource_limits or {},
    }
    try:
        response = executor(action_id, backend_id, request)
    except Exception as exc:
        # Exception 不包含 KeyboardInterrupt/SystemExit；两者必须继续由上层控制。
        response = _normalize_nested_exception(exc)
    if not isinstance(response, dict):
        response = _safe_stage_response(
            status="failed",
            code="backend_unavailable",
            retryable=True,
        )
    status = str(response.get("status") or "unknown")
    safe_error_code = _stage_failure_code(response, fallback_code="backend_unavailable")
    diagnostics = _safe_nested_stage_diagnostics(
        response,
        backend_id=backend_id,
        status=status,
    )
    trace_entry = {
        "sequence": len(trace) + 1,
        "action_id": action_id,
        "backend_id": backend_id,
        "status": status,
        "error": {
            "code": safe_error_code,
            "retryable": response.get("retryable") is True,
        },
    }
    if diagnostics:
        trace_entry["diagnostics"] = diagnostics
    trace.append(trace_entry)
    if status not in _ACCEPTED:
        raise StageUnavailable(
            action_id,
            backend_id,
            _safe_stage_response(
                status=status,
                code=safe_error_code,
                retryable=response.get("retryable") is True,
                diagnostics=diagnostics,
            ),
        )
    result = response.get("result")
    if not isinstance(result, dict):
        raise StageUnavailable(
            action_id,
            backend_id,
            _safe_stage_response(
                status="failed",
                code="backend_unavailable",
                retryable=True,
            ),
        )
    nested_provenance = _safe_nested_stage_provenance(
        response,
        action_id=action_id,
        backend_id=backend_id,
        result=result,
    )
    if nested_provenance is not None:
        trace_entry["provenance"] = nested_provenance
    return result


def _safe_stage_failure_reason(
    response: dict[str, Any], *, fallback_code: str
) -> dict[str, Any]:
    """从嵌套 Action 响应提取错误分类、重试标志和安全诊断。

    输入是服务端收到的原子 Action 响应；错误码和布尔 ``retryable`` 仍受白名单
    约束，但 ASH/KNARR 已经清理过的 ``message``/``warnings`` 会继续保留，帮助
    调用方修正参数。未知错误统一降级为调用方提供的泛化分类，并视为不可重试。
    """

    error = response.get("error")
    raw_code = error.get("code") if isinstance(error, Mapping) else None
    safe_code = (
        isinstance(raw_code, str)
        and raw_code in _SAFE_STAGE_ERROR_CODES
    )
    code = raw_code if safe_code else fallback_code
    retryable = response.get("retryable") is True if safe_code else False
    reason: dict[str, Any] = {"reason_code": code, "retryable": retryable}
    message = response.get("message")
    if isinstance(message, str) and message.strip():
        reason["message"] = message
    warnings = response.get("warnings")
    if isinstance(warnings, list) and warnings:
        reason["warnings"] = list(warnings)
    return reason


def _merge_stage_failure_reasons(
    reasons: list[dict[str, Any]], *, fallback_code: str
) -> dict[str, Any]:
    """合并同一阶段多个端点失败，保留安全诊断和保守的重试语义。"""

    merged: dict[str, Any] = {}
    for reason in reasons:
        code = reason.get("reason_code")
        if code != fallback_code:
            merged["reason_code"] = code
            break
    if "reason_code" not in merged:
        merged["reason_code"] = fallback_code
    merged["retryable"] = (
        any(item.get("retryable") is True for item in reasons)
        if merged["reason_code"] != fallback_code
        else False
    )
    for reason in reasons:
        message = reason.get("message")
        if "message" not in merged and isinstance(message, str) and message.strip():
            merged["message"] = message
        warnings = reason.get("warnings")
        if isinstance(warnings, list) and warnings:
            existing = merged.setdefault("warnings", [])
            for warning in warnings:
                if warning not in existing:
                    existing.append(warning)
    return merged


def _validate_settings(settings: dict[str, Any]) -> dict[str, Any]:
    if settings:
        raise ValueError(
            "Transition-state path controls are server-managed; action_settings must be empty"
        )
    configuration = _server_settings()
    minimum_ratio = _finite_positive(
        configuration["minimum_distance_ratio"], "minimum_distance_ratio"
    )
    maximum_ratio = _finite_positive(
        configuration["maximum_distance_ratio"], "maximum_distance_ratio"
    )
    if maximum_ratio < minimum_ratio:
        raise RuntimeError("Server-managed distance constraint configuration is invalid")
    return {
        **configuration,
        "conformer_count": 1,
        "max_conformer_combinations": _MAX_COMBINATIONS,
        "complex_candidates_per_combination": configuration["complex_max_attempts"],
        "complex_orientation_trials": 1,
        "contact_tolerance_angstrom": 0.8,
        "minimum_interatomic_distance_angstrom": 0.75,
        "complex_mmff_max_iterations": 10000,
        "product_mmff_max_iterations": 300,
        # 组件构象阶段直接调用 RDKit MMFFOptimizeMolecule，未传容差，
        # 因而使用 RDKit 默认 forceTol=1e-4、energyTol=1e-6。
        "component_mmff_force_tolerance": 1.0e-4,
        "component_mmff_energy_tolerance": 1.0e-6,
        # 作者 optimize_with_constraints 明确使用更紧的复合物/产物容差。
        "constrained_mmff_force_tolerance": 1.0e-5,
        "constrained_mmff_energy_tolerance": 1.0e-7,
        "complex_rmsd_cutoff_angstrom": configuration["filter_rmsd_threshold"],
        "max_endpoint_pairs": _MAX_ENDPOINT_PAIRS,
        "run_uma": True,
        "run_neb": True,
        "uma_optimizer": "bfgs",
        "neb_maximum_force_ev_per_angstrom": _ASH_MAX_FORCE,
        "neb_rms_force_ev_per_angstrom": _ASH_RMS_FORCE,
        "neb_interpolation": "idpp",
        "idpp_max_iterations": _ASH_IDPP_MAX_ITERATIONS,
        "minimum_mapping_confidence": 0.0,
        "mmff_force_field": "mmff94",
    }


def _component_ensembles(
    source_smiles: list[str],
    settings: dict[str, Any],
    executor: Executor,
    trace: list[dict[str, Any]],
    mmff_method: dict[str, Any],
) -> list[dict[str, Any]]:
    def conformer_request_count(smiles: str) -> int:
        """按作者公式计算一个分子的完整构象请求数。"""

        try:
            from rdkit import Chem
            from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError("RDKit is required for transition-state path conformer counts") from exc
        parser = Chem.SmilesParserParams()
        parser.removeHs = False
        molecule = Chem.MolFromSmiles(smiles, parser)
        if molecule is None:
            raise ValueError("A reactant SMILES could not be parsed for conformer generation")
        rotatable_bonds = int(CalcNumRotatableBonds(Chem.AddHs(molecule), strict=True))
        return min(
            10 + 4**rotatable_bonds,
            int(settings["conformer_max_attempts"]),
        )

    ensembles = []
    for smiles in source_smiles:
        generated = _call(
            executor,
            trace,
            "generate_conformer_ensemble",
            "rdkit_etkdg",
            inputs={"molecule": {"smiles": smiles}},
            action_settings={
                "num_conformers": conformer_request_count(smiles),
                "random_seed": int(settings["random_seed"]),
                "prune_rms_threshold_angstrom": float(
                    settings["conformer_prune_rms_threshold_angstrom"]
                ),
                # 旧流程直接调用 EmbedMultipleConfs 的默认参数；这里只在
                # 服务端固定调用中选择该实现，不进入公开参数注册。
                "embedding_mode": "author_default",
            },
        )
        optimized_records = []
        scores = []
        for conformer in generated["ensemble"]:
            optimized = _call(
                executor,
                trace,
                "optimize_geometry",
                "rdkit_mmff",
                inputs={"structure": conformer["structure"], "constraints": []},
                method_spec=mmff_method,
                action_settings={
                    "max_iterations": int(settings["component_mmff_max_iterations"]),
                    "force_tolerance": float(settings["component_mmff_force_tolerance"]),
                    "energy_tolerance": float(settings["component_mmff_energy_tolerance"]),
                },
            )
            optimized_records.append(
                {
                    "conformer_id": str(conformer["conformer_id"]),
                    "structure": optimized["structure"],
                }
            )
            scores.append({"value": float(optimized["energy_kcal_mol"])})
        ranked = _call(
            executor,
            trace,
            "rank_conformers_from_results",
            "internal_statistics",
            inputs={"ensemble": optimized_records, "scores": scores},
            action_settings={"temperature_kelvin": 298.15, "score_unit": "kcal_mol"},
        )
        ensembles.append(ranked)
    return ensembles


def _constraint_settings(settings: dict[str, Any]) -> dict[str, Any]:
    return {
        "equilibrium_distance_scale": float(settings["equilibrium_distance_scale"]),
        "minimum_distance_ratio": float(settings["minimum_distance_ratio"]),
        "maximum_distance_ratio": float(settings["maximum_distance_ratio"]),
        "force_constant_kcal_mol_angstrom2": float(
            settings["constraint_force_constant_kcal_mol_angstrom2"]
        ),
    }


def _mmff_settings(settings: dict[str, Any], phase: str) -> dict[str, Any]:
    iteration_field = {
        "complex": "complex_mmff_max_iterations",
        "product": "product_mmff_max_iterations",
    }.get(phase)
    if iteration_field is None:
        raise ValueError(f"Unknown MMFF workflow phase: {phase}")
    return {
        "max_iterations": int(settings[iteration_field]),
        "force_tolerance": float(settings["constrained_mmff_force_tolerance"]),
        "energy_tolerance": float(settings["constrained_mmff_energy_tolerance"]),
    }


def _author_forming_bonds_pass_distance_gate(
    structure: dict[str, Any],
    forming_bonds: list[dict[str, Any]],
    settings: dict[str, Any],
) -> bool:
    """按旧流程在约束 MMFF 后检查所有成键距离。

    阈值是固定元素单键长度乘以 ``equilibrium_ratio`` 和
    ``max_allowed_ratio``，边界值本身不接受；这样刚体采样不会把作者原本
    在 MMFF 后执行的验收误写成装配阶段的碰撞筛选。
    """

    symbols, coordinates = atoms_and_coordinates(structure)
    positions_by_map = {}
    for index, atom in enumerate(structure.get("atoms") or []):
        map_number = atom.get(
            "atom_map_number", atom.get("atom_map", atom.get("map_number"))
        )
        if map_number is not None:
            positions_by_map[int(map_number)] = (symbols[index], coordinates[index])
    for bond in forming_bonds:
        first_map = int(bond["atom_map_a"])
        second_map = int(bond["atom_map_b"])
        try:
            first_symbol, first_position = positions_by_map[first_map]
            second_symbol, second_position = positions_by_map[second_map]
        except KeyError:
            return False
        reference = _AUTHOR_SINGLE_BOND_LENGTHS_ANGSTROM.get(
            tuple(sorted((first_symbol, second_symbol)))
        )
        if reference is None:
            return False
        threshold = (
            reference
            * float(settings["equilibrium_distance_scale"])
            * float(settings["maximum_allowed_ratio"])
        )
        distance = math.sqrt(
            sum(
                (first_position[index] - second_position[index]) ** 2
                for index in range(3)
            )
        )
        if not distance < threshold:
            return False
    return True


def _prepare_endpoint_pairs(
    combinations: dict[str, Any],
    changes: dict[str, Any],
    settings: dict[str, Any],
    executor: Executor,
    trace: list[dict[str, Any]],
    mmff_method: dict[str, Any],
) -> list[dict[str, Any]]:
    """为每个组合收集全部合格尝试；单组合失败只记录在聚合计数中。"""

    direction = settings["direction"]
    forming_bonds = (
        changes["formed_bonds"] if direction == "forward" else changes["broken_bonds"]
    )
    breaking_bonds = (
        changes["broken_bonds"] if direction == "forward" else changes["formed_bonds"]
    )
    pairs: list[dict[str, Any]] = []
    for combination_index, combination in enumerate(combinations["combinations"]):
        try:
            contacts = _call(
                executor,
                trace,
                "resolve_mapped_reactive_contacts",
                "internal_reactive_complex",
                inputs={
                    "components": combination["components"],
                    "reaction_changes": changes,
                    "forming_bonds": forming_bonds,
                },
                action_settings={
                    "direction": direction,
                    "equilibrium_distance_scale": float(
                        settings["equilibrium_distance_scale"]
                    ),
                    "contact_tolerance_angstrom": float(
                        settings["contact_tolerance_angstrom"]
                    ),
                },
            )
            assembled = _call(
                executor,
                trace,
                "build_reactive_complex_ensemble",
                "internal_reactive_complex",
                inputs={
                    "components": combination["components"],
                    "contact_pairs": contacts["contact_pairs"],
                },
                action_settings={
                    # 作者在整个进程只调用一次 random.seed(42)。两个分子时每个
                    # 组合的后 9 次尝试各消耗 3 个 Euler uniform 和 3 个平移
                    # uniform；offset 让独立原子 Action 继续同一连续流。
                    "random_seed": int(settings["random_seed"]),
                    "random_stream_offset": (
                        combination_index
                        * (int(settings["complex_max_attempts"]) - 1)
                        * 6
                        * max(0, len(combination["components"]) - 1)
                    ),
                    "candidate_count": settings["complex_candidates_per_combination"],
                    "max_candidates": settings["complex_candidates_per_combination"],
                    "orientation_trials": 1,
                    "initial_centroid_distance_angstrom": float(
                        settings["complex_initial_distance"]
                    ),
                    "contact_tolerance_angstrom": float(
                        settings["contact_tolerance_angstrom"]
                    ),
                    "minimum_interatomic_distance_angstrom": float(
                        settings["minimum_interatomic_distance_angstrom"]
                    ),
                    "maximum_reported_collisions": 32,
                    "charge": int(settings["charge"]),
                    "multiplicity": int(settings["multiplicity"]),
                },
            )
        except StageUnavailable as exc:
            if _is_required_stage_unavailable(exc.response):
                raise
            # 该组合没有可消费的候选时继续下一个组合；trace 已保存可审计的状态计数。
            continue
        except (RuntimeError, ValueError, KeyError, TypeError):
            # 该组合没有可消费的候选时继续下一个组合；trace 已保存可审计的状态计数。
            continue
        for candidate in assembled.get("ensemble") or []:
            try:
                source_constraints = _call(
                    executor,
                    trace,
                    "build_mapped_distance_constraints",
                    "internal_reactive_complex",
                    inputs={
                        "structure": candidate["structure"],
                        "mapped_bonds": forming_bonds,
                    },
                    action_settings=_constraint_settings(settings),
                )
                source_optimized = _call(
                    executor,
                    trace,
                    "optimize_geometry",
                    "rdkit_mmff",
                    inputs={
                        "structure": candidate["structure"],
                        "constraints": source_constraints["constraints"],
                    },
                    method_spec=mmff_method,
                    action_settings=_mmff_settings(settings, "complex"),
                )
                if not _author_forming_bonds_pass_distance_gate(
                    source_optimized["structure"], forming_bonds, settings
                ):
                    # 这是作者约束 MMFF 后的严格成键距离验收；等于阈值也拒绝。
                    continue
                target_edited = _call(
                    executor,
                    trace,
                    "apply_reaction_changes_to_structure",
                    "internal_reaction_analysis",
                    inputs={
                        "structure": source_optimized["structure"],
                        "reaction_changes": changes,
                    },
                    action_settings={
                        "direction": direction,
                        "target_charge": int(settings["charge"]),
                        "target_multiplicity": int(settings["multiplicity"]),
                    },
                )
                if breaking_bonds:
                    target_constraints = _call(
                        executor,
                        trace,
                        "build_mapped_distance_constraints",
                        "internal_reactive_complex",
                        inputs={
                            "structure": target_edited["structure"],
                            "mapped_bonds": breaking_bonds,
                        },
                        action_settings=_constraint_settings(settings),
                    )
                else:
                    target_constraints = {"constraints": []}
                target_optimized = _call(
                    executor,
                    trace,
                    "optimize_geometry",
                    "rdkit_mmff",
                    inputs={
                        "structure": target_edited["structure"],
                        "constraints": target_constraints["constraints"],
                    },
                    method_spec=mmff_method,
                    action_settings=_mmff_settings(settings, "product"),
                )
            except StageUnavailable as exc:
                if _is_required_stage_unavailable(exc.response):
                    raise
                # 单个候选失败不应中止组合或整个固定流程。
                continue
            except (RuntimeError, ValueError, KeyError, TypeError):
                # 单个候选失败不应中止组合或整个固定流程。
                continue
            pair_id = f"endpoint_pair_{len(pairs) + 1:04d}"
            source_converged = bool(source_optimized.get("converged"))
            target_converged = bool(target_optimized.get("converged"))
            pairs.append(
                {
                    "pair_id": pair_id,
                    "combination_id": combination["combination_id"],
                    "source_candidate_id": candidate["candidate_id"],
                    "reactant": (
                        source_optimized["structure"]
                        if direction == "forward"
                        else target_optimized["structure"]
                    ),
                    "product": (
                        target_optimized["structure"]
                        if direction == "forward"
                        else source_optimized["structure"]
                    ),
                    "source_mmff_converged": source_converged,
                    "target_mmff_converged": target_converged,
                    "stage_statuses": {
                        "mmff": "success"
                        if source_converged and target_converged
                        else "partial",
                        "uma": "pending",
                        "neb": "pending",
                    },
                }
            )
            if len(pairs) >= settings["max_endpoint_pairs"]:
                return pairs
    return pairs


def _cluster_endpoint_pairs(
    pairs: list[dict[str, Any]],
    settings: dict[str, Any],
    executor: Executor,
    trace: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(pairs) < 2:
        return pairs
    ensemble = [
        {"conformer_id": pair["pair_id"], "structure": pair["reactant"]}
        for pair in pairs
    ]
    clustered = _call(
        executor,
        trace,
        "cluster_conformers",
        "rdkit",
        inputs={"ensemble": {"ensemble": ensemble}},
        action_settings={
            "rmsd_cutoff_angstrom": float(settings["complex_rmsd_cutoff_angstrom"]),
            "atom_selection": "all",
            "prealign_conformers": False,
            "reorder_cluster_centers": bool(settings["reorder"]),
            "reorder_atoms": bool(settings["reorder"]),
            "selection_mode": "greedy_input_order",
        },
    )
    representatives = {
        cluster["representative_conformer_id"] for cluster in clustered["clusters"]
    }
    return [pair for pair in pairs if pair["pair_id"] in representatives]


def _reaction_path_has_complete_indexed_images(
    reaction_path: Any,
    *,
    expected_total_images: int | None = None,
) -> bool:
    """核对路径图像、能量和零起始索引是否形成完整一一对应结果。"""

    if not isinstance(reaction_path, Mapping):
        return False
    images = reaction_path.get("images")
    energies = reaction_path.get("energies_hartree")
    indices = reaction_path.get("energy_image_indices")
    if not isinstance(images, list) or len(images) < 3:
        return False
    total_images = reaction_path.get("total_images")
    if isinstance(total_images, bool) or not isinstance(total_images, int):
        total_images = len(images)
    if total_images != len(images) or (
        expected_total_images is not None and total_images != expected_total_images
    ):
        return False
    if reaction_path.get("final_path_complete") is False:
        return False
    if reaction_path.get("energy_index_complete") is False:
        return False
    if not isinstance(energies, list) or len(energies) != total_images:
        return False
    try:
        if any(
            isinstance(value, bool) or not math.isfinite(float(value))
            for value in energies
        ):
            return False
    except (TypeError, ValueError, OverflowError):
        return False
    if (
        not isinstance(indices, list)
        or len(indices) != total_images
        or any(isinstance(index, bool) or not isinstance(index, int) for index in indices)
        or indices != list(range(total_images))
    ):
        return False
    return True


def _reaction_path_has_valid_interior_peak(reaction_path: Any) -> bool:
    """判断完整路径是否有端点之外的最高能量图像可作未验证候选。"""

    if not isinstance(reaction_path, Mapping):
        return False
    explicit = reaction_path.get("interior_peak_candidate_valid")
    if explicit is not None:
        return explicit is True
    energies = reaction_path.get("energies_hartree")
    if not isinstance(energies, list) or len(energies) < 3:
        return False
    try:
        highest_index = max(range(len(energies)), key=lambda index: float(energies[index]))
    except (TypeError, ValueError, OverflowError):
        return False
    interior_index = reaction_path.get("highest_interior_image_index")
    if interior_index is None:
        interior_index = reaction_path.get("highest_interior_energy_image_index")
    return (
        isinstance(interior_index, int)
        and not isinstance(interior_index, bool)
        and 0 < interior_index < len(energies) - 1
        and highest_index == interior_index
    )


def _run_heavy_stages(
    pairs: list[dict[str, Any]],
    method: dict[str, Any],
    settings: dict[str, Any],
    resources: dict[str, Any],
    executor: Executor,
    trace: list[dict[str, Any]],
    *,
    failure_records: list[dict[str, Any]] | None = None,
    diagnostic_records: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    """逐 endpoint pair 执行 UMA 和 NEB，单 pair 失败不阻断其他 pair。

    ``failure_records`` 只接收安全的阶段失败摘要，供外层结果合同使用；它不会保存
    原子 Action 的错误正文或内部诊断字段。调用方不提供该列表时仍保持原有返回合同。
    """

    failures: list[str] = []
    uma_method = {
        "model": str(settings["uma_model"]),
        "task_name": str(settings["uma_task_name"]),
        # UMA and ASH/KNARR are fixed GPU stages for this Action.  Placement is
        # intentionally not exposed as an Agent or operator override.
        "device": "cuda",
        # 论文流程使用预置的 UMA 缓存；缓存缺失时让该 pair 明确失败，不联网下载或改写缓存。
        "allow_model_download": False,
    }
    uma_settings = {
        "fmax_ev_per_angstrom": float(settings["uma_fmax_ev_per_angstrom"]),
        "optimizer": str(settings["uma_optimizer"]),
        "max_steps": int(settings["uma_max_steps"]),
    }
    for pair in pairs:
        stage_statuses = pair.setdefault("stage_statuses", {})
        pair.setdefault("reactant_uma", None)
        pair.setdefault("product_uma", None)
        pair["reactant_uma_converged"] = False
        pair["product_uma_converged"] = False
        endpoint_failures: list[str] = []
        uma_failure_reasons: list[dict[str, Any]] = []
        for role in ("reactant", "product"):
            try:
                optimized = _call(
                    executor,
                    trace,
                    "optimize_geometry",
                    "fairchem_uma",
                    inputs={"structure": pair[role]},
                    method_spec=uma_method,
                    action_settings=uma_settings,
                    resource_limits=resources,
                )
            except StageUnavailable as exc:
                uma_failure_reasons.append(
                    _safe_stage_failure_reason(
                        exc.response,
                        fallback_code="pair_stage_failed",
                    )
                )
                # 原子 Action 即使返回 success 但结构不完整，也只影响当前
                # pair；公开结果只记录阶段状态，不携带原始异常或嵌套调试文本。
                endpoint_failures.append(f"{role}: unavailable")
                continue
            except (KeyError, TypeError, ValueError, RuntimeError, AttributeError):
                # 原子 Action 即使返回 success 但结构不完整，也只影响当前
                # pair；公开结果只记录阶段状态，不携带原始异常或嵌套调试文本。
                endpoint_failures.append(f"{role}: unavailable")
                continue
            optimized_structure = (
                optimized.get("structure")
                if isinstance(optimized, dict)
                else None
            )
            if not isinstance(optimized_structure, dict):
                endpoint_failures.append(f"{role}: malformed result")
                continue
            pair[f"{role}_uma"] = optimized_structure
            pair[f"{role}_uma_converged"] = optimized.get("converged") is True
            if not pair[f"{role}_uma_converged"]:
                endpoint_failures.append(f"{role}: converged=false")
        if endpoint_failures:
            stage_statuses["uma"] = "failed"
            stage_statuses["neb"] = "skipped"
            if failure_records is not None:
                reason = _merge_stage_failure_reasons(
                    uma_failure_reasons,
                    fallback_code="pair_stage_failed",
                )
                failure_records.append(
                    {
                        "pair_id": pair["pair_id"],
                        "stage": "uma",
                        "status": "failed",
                        **reason,
                    }
                )
                failure_records.append(
                    {
                        "pair_id": pair["pair_id"],
                        "stage": "neb",
                        "status": "skipped",
                        "reason_code": "pair_stage_skipped",
                        "retryable": False,
                    }
                )
            reason = (
                f"{pair['pair_id']} UMA endpoint stage unavailable or unconverged"
            )
            pair["neb_skipped_reason"] = reason
            failures.append(reason)
            continue
        stage_statuses["uma"] = "success"
        try:
            path = _call(
                executor,
                trace,
                "search_reaction_path",
                "ash_neb_uma",
                inputs={
                    "reactant": pair["reactant_uma"],
                    "product": pair["product_uma"],
                },
                method_spec={
                    **uma_method,
                    # ASH/FairchemTheory 未从作者 TSPipeline 接收 seed，使用
                    # 安装版本的 source-effective 默认值 41，而非构象 seed42。
                    "random_seed": _ASH_RANDOM_SEED,
                },
                action_settings={
                    "total_images": int(settings["total_images"]),
                    "climb": bool(settings["climb"]),
                    # 作者活动封装没有把旧配置中的 .05 字段传给 ASH，因此这里不使用；
                    # 保留 ASH 的实际默认阈值和 IDPP 设置。
                    "maximum_force_ev_per_angstrom": _ASH_MAX_FORCE,
                    "rms_force_ev_per_angstrom": _ASH_RMS_FORCE,
                    "max_steps": int(settings["neb_max_steps"]),
                    "interpolation": "idpp",
                    "idpp_max_iterations": _ASH_IDPP_MAX_ITERATIONS,
                },
                resource_limits=resources,
            )
        except StageUnavailable as exc:
            stage_statuses["neb"] = "failed"
            if failure_records is not None:
                failure_records.append(
                    {
                        "pair_id": pair["pair_id"],
                        "stage": "neb",
                        "status": "failed",
                        **_safe_stage_failure_reason(
                            exc.response,
                            fallback_code="pair_stage_failed",
                        ),
                    }
                )
            failures.append(f"{pair['pair_id']} NEB unavailable")
            continue
        except (KeyError, TypeError, ValueError, RuntimeError, AttributeError):
            stage_statuses["neb"] = "failed"
            if failure_records is not None:
                failure_records.append(
                    {
                        "pair_id": pair["pair_id"],
                        "stage": "neb",
                        "status": "failed",
                        "reason_code": "pair_stage_failed",
                        "retryable": False,
                    }
                )
            failures.append(f"{pair['pair_id']} NEB unavailable")
            continue
        nested_trace = trace[-1] if trace else None
        nested_diagnostics = (
            nested_trace.get("diagnostics")
            if isinstance(nested_trace, Mapping)
            and nested_trace.get("backend_id") == "ash_neb_uma"
            else None
        )
        if diagnostic_records is not None and isinstance(nested_diagnostics, Mapping):
            diagnostic_record: dict[str, Any] = {
                "pair_id": pair["pair_id"],
                "stage": "neb",
                "status": (
                    "partial"
                    if nested_trace.get("status") == "partial_success"
                    else "failed"
                ),
            }
            diagnostic_record.update(dict(nested_diagnostics))
            diagnostic_records.append(diagnostic_record)
        nested_provenance = (
            nested_trace.get("provenance")
            if isinstance(nested_trace, Mapping)
            else None
        )
        if isinstance(nested_provenance, Mapping):
            safe_provenance = _normalize_pipeline_provenance(nested_provenance)
            if safe_provenance is not None:
                path["provenance"] = copy.deepcopy(safe_provenance)
                pair["provenance"] = copy.deepcopy(safe_provenance)
        pair["reaction_path"] = path
        path_complete = _reaction_path_has_complete_indexed_images(
            path,
            expected_total_images=int(settings["total_images"]),
        ) and _reaction_path_has_valid_interior_peak(path)
        if not path_complete:
            stage_statuses["neb"] = "partial"
            incomplete_message = (
                f"{pair['pair_id']} ASH NEB returned an incomplete reaction path; "
                "the path must contain every image and every indexed energy before a "
                "transition-state guess can be handed off."
            )
            if failure_records is not None:
                failure_records.append(
                    {
                        "pair_id": pair["pair_id"],
                        "stage": "neb",
                        "status": "failed",
                        "reason_code": "reaction_path_incomplete",
                        "retryable": False,
                        "message": incomplete_message,
                    }
                )
            failures.append(incomplete_message)
        else:
            stage_statuses["neb"] = "success"
    return pairs, "; ".join(failures) if failures else None



def _reaction_path_image_sources(reaction_path: Any) -> list[tuple[int, Any, str]]:
    """读取已有路径中能明确对应到索引的图像来源。

    返回值按路径原有顺序保存 ``(image_index, source, source_kind)``。内嵌
    AtomicStructure 会被标记为 ``structure``；已经生成的单帧 XYZ 文件会被标记
    为 ``file``。函数不按能量选择图像，不读取 highest-energy 等诊断字段，也不把
    任意路径文件猜成过渡态。
    """

    if not isinstance(reaction_path, dict):
        return []
    sources: list[tuple[int, Any, str]] = []
    images = reaction_path.get("images")
    if isinstance(images, list):
        for image_index, image in enumerate(images):
            try:
                sources.append((image_index, structure_dict(image), "structure"))
            except (TypeError, ValueError, OSError):
                continue
        if sources:
            return sources

    image_files = reaction_path.get("image_files")
    if not isinstance(image_files, list):
        return []
    for image_index, image_file in enumerate(image_files):
        try:
            path = resolve_input_file(image_file)
            if path.suffix.lower() != ".xyz":
                continue
            structure_dict(relative_workspace_path(path))
        except (TypeError, ValueError, OSError):
            continue
        sources.append((image_index, path, "file"))
    return sources


def _materialize_transition_state_guess_xyz_artifact(
    reaction_path: dict[str, Any],
    *,
    directory,
    pair_index: int,
) -> dict[str, str] | None:
    """把 backend 已确认的未验证 TS guess 写成单独 XYZFile 交接条目。

    只有 reaction_path 明确提供的结构才会被写入；旧版结果若同时明确提供完整路径、
    完整能量索引和最高内部图像索引，则由 materialization 层按固定规则恢复该图像。
    其他缺少证据的结果不会从不完整路径猜造候选；结构无法通过统一 AtomicStructure
    校验时直接失败，避免静默制造一个下游可消费但内容不明的初猜。
    """

    transition_state_guess = reaction_path.get("transition_state_guess")
    if transition_state_guess is None:
        return None
    if not isinstance(transition_state_guess, dict):
        raise ValueError("TS Pipeline transition_state_guess must be an AtomicStructure")
    try:
        structure = structure_dict(transition_state_guess)
    except (TypeError, ValueError, OSError) as exc:
        raise ValueError("TS Pipeline transition_state_guess is not a valid structure") from exc
    guess_path = write_xyz(
        structure,
        directory / f"endpoint_pair_{pair_index:04d}_transition_state_guess.xyz",
    )
    return {
        "path": relative_workspace_path(guess_path),
        "semantic_type": "XYZFile",
        "media_type": "chemical/x-xyz",
        "result_pointer": f"/endpoint_pairs/{pair_index}/transition_state_guess_artifact",
    }


def _recover_legacy_complete_path_transition_state_guess(
    reaction_path: dict[str, Any],
) -> tuple[dict[str, Any] | None, int | None]:
    """从旧版完整路径结果恢复缺失的最高内部图像初猜。

    作者旧流程把 ``knarr_saddle.xyz`` 和路径文件分别交接，部分历史结果只留下
    完整 images、energy indices 和最高内部索引而把 ``transition_state_guess`` 写成
    null。只有路径、能量索引、内部峰值和端点排除条件都明确完整时才恢复；缺任何
    证据都保持 null，避免把不完整路径误当作 TS 初猜。
    """

    if reaction_path.get("transition_state_guess") is not None:
        return None, None
    images = reaction_path.get("images")
    if not isinstance(images, list) or len(images) < 3:
        return None, None
    total_images = reaction_path.get("total_images")
    if isinstance(total_images, bool) or not isinstance(total_images, int):
        total_images = len(images)
    if total_images != len(images):
        return None, None
    if not _reaction_path_has_complete_indexed_images(
        reaction_path,
        expected_total_images=total_images,
    ) or not _reaction_path_has_valid_interior_peak(reaction_path):
        return None, None

    energies = reaction_path.get("energies_hartree")
    if not isinstance(energies, list) or len(energies) != total_images:
        return None, None
    raw_indices = reaction_path.get("energy_image_indices")
    if not isinstance(raw_indices, list) or raw_indices != list(range(total_images)):
        return None, None
    raw_index = reaction_path.get("highest_interior_image_index")
    if raw_index is None:
        raw_index = reaction_path.get("highest_interior_energy_image_index")
    if (
        isinstance(raw_index, bool)
        or not isinstance(raw_index, int)
        or not 0 < raw_index < total_images - 1
    ):
        return None, None
    highest_energy_index = reaction_path.get("highest_energy_image_index")
    if highest_energy_index is None:
        try:
            highest_energy_index = max(
                range(total_images), key=lambda index: float(energies[index])
            )
        except (TypeError, ValueError, OverflowError):
            return None, None
    if isinstance(highest_energy_index, bool) or not isinstance(highest_energy_index, int):
        return None, None
    if highest_energy_index != raw_index:
        return None, None
    try:
        return structure_dict(images[raw_index]), raw_index
    except (TypeError, ValueError, OSError):
        return None, None


def _materialize_pipeline_artifacts(
    result: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """为每个 endpoint pair 和可用路径图像生成声明式文件交接条目。

    输入是 TS Pipeline 已完成的结构化结果；函数只把已有结构、路径和路径图像
    写入当前 Action 的输出目录，并在结果中预留由服务层注入的 ArtifactRef 槽位。
    不改变 pair 顺序、候选身份或路径内容，不判断图像是不是过渡态。返回结果和
    文件清单；服务层登记文件后，``result_pointer`` 指向的槽位才会变成 ArtifactRef。
    """

    directory = output_directory(ACTION_ID, BACKEND_ID)
    artifact_files: list[dict[str, str]] = []
    path_image_candidates: list[dict[str, Any]] = []
    endpoint_pairs = result.get("endpoint_pairs")
    if not isinstance(endpoint_pairs, list):
        raise ValueError("TS Pipeline result endpoint_pairs must be a list")
    if len(endpoint_pairs) > _MAX_ENDPOINT_PAIRS:
        raise ValueError("Endpoint-pair count exceeds the bounded complete-result limit")

    for pair_index, pair in enumerate(endpoint_pairs):
        if not isinstance(pair, dict):
            raise ValueError(f"TS Pipeline endpoint_pairs[{pair_index}] must be an object")
        pair_id = pair.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id.strip():
            raise ValueError(f"TS Pipeline endpoint_pairs[{pair_index}] is missing pair_id")
        combination_id = pair.get("combination_id")
        source_candidate_id = pair.get("source_candidate_id")
        if not isinstance(combination_id, str) or not combination_id.strip():
            raise ValueError(
                f"TS Pipeline endpoint_pairs[{pair_index}] is missing combination_id"
            )
        if not isinstance(source_candidate_id, str) or not source_candidate_id.strip():
            raise ValueError(
                f"TS Pipeline endpoint_pairs[{pair_index}] is missing source_candidate_id"
            )

        pair["reactant_artifact"] = None
        pair["product_artifact"] = None
        pair["endpoint_artifact_source"] = (
            "uma"
            if pair.get("stage_statuses", {}).get("uma") == "success"
            and pair.get("reactant_uma_converged") is True
            and pair.get("product_uma_converged") is True
            else "mmff_prepared"
        )
        pair["reaction_path_artifact"] = None
        pair["transition_state_guess_artifact"] = None
        pair["transition_state_guess_source"] = None
        pair["transition_state_guess_source_image_index"] = None
        pair["transition_state_guess_selection_rule_version"] = None
        pair["path_image_artifacts"] = []
        pair["provenance"] = None

        for role in ("reactant", "product"):
            use_uma = pair["endpoint_artifact_source"] == "uma"
            structure = pair.get(f"{role}_uma") if use_uma else pair.get(role)
            if not isinstance(structure, dict):
                raise ValueError(f"TS Pipeline pair {pair_id} has no {role} structure")
            suffix = "uma" if use_uma else "mmff_prepared"
            structure_path = write_json(
                directory,
                f"endpoint_pair_{pair_index:04d}_{role}_{suffix}.json",
                structure,
            )
            artifact_files.append(
                {
                    "path": relative_workspace_path(structure_path),
                    "semantic_type": "AtomicStructure",
                    "media_type": "application/json",
                    "result_pointer": f"/endpoint_pairs/{pair_index}/{role}_artifact",
                }
            )

        reaction_path = pair.get("reaction_path")
        # 稳定结果合同要求未运行或未得到 NEB 路径时也显式返回 null，不能让
        # 同一字段随重阶段可用性在“缺失”和“存在”之间漂移。
        pair["reaction_path"] = reaction_path if isinstance(reaction_path, dict) else None
        if not isinstance(reaction_path, dict):
            for sensitive_field in (
                "reactant",
                "product",
                "reactant_uma",
                "product_uma",
                "reaction_path",
                "neb_skipped_reason",
                "combination_id",
                "source_candidate_id",
                "source_mmff_converged",
                "target_mmff_converged",
                "reactant_uma_converged",
                "product_uma_converged",
            ):
                pair.pop(sensitive_field, None)
            continue
        recovered_guess, recovered_index = (
            _recover_legacy_complete_path_transition_state_guess(reaction_path)
        )
        if recovered_guess is not None and recovered_index is not None:
            reaction_path["transition_state_guess"] = recovered_guess
            reaction_path["transition_state_guess_source"] = (
                "highest_interior_path_image"
            )
            reaction_path["transition_state_guess_source_image_index"] = (
                recovered_index
            )
            reaction_path["transition_state_guess_selection_rule_version"] = (
                _TRANSITION_STATE_GUESS_SELECTION_RULE_VERSION
            )
        transition_state_guess_source = reaction_path.get(
            "transition_state_guess_source"
        )
        if transition_state_guess_source is not None and (
            not isinstance(transition_state_guess_source, str)
            or transition_state_guess_source not in _TRANSITION_STATE_GUESS_SOURCES
        ):
            raise ValueError(
                f"TS Pipeline pair {pair_id} has an invalid transition-state guess source"
            )
        transition_state_guess_source_image_index = reaction_path.get(
            "transition_state_guess_source_image_index"
        )
        if transition_state_guess_source_image_index is not None and (
            isinstance(transition_state_guess_source_image_index, bool)
            or not isinstance(transition_state_guess_source_image_index, int)
            or transition_state_guess_source_image_index < 0
        ):
            raise ValueError(
                f"TS Pipeline pair {pair_id} has an invalid transition-state guess image index"
            )
        transition_state_guess_selection_rule_version = reaction_path.get(
            "transition_state_guess_selection_rule_version"
        )
        if transition_state_guess_selection_rule_version is not None and (
            not isinstance(transition_state_guess_selection_rule_version, str)
            or not transition_state_guess_selection_rule_version.strip()
        ):
            raise ValueError(
                f"TS Pipeline pair {pair_id} has an invalid transition-state guess selection rule"
            )
        if transition_state_guess_source is None:
            transition_state_guess_source_image_index = None
            transition_state_guess_selection_rule_version = None
        elif reaction_path.get("transition_state_guess") is None:
            # 不完整路径允许没有 guess，但不得暴露与 null guess 不一致的来源元数据。
            transition_state_guess_source = None
            transition_state_guess_source_image_index = None
            transition_state_guess_selection_rule_version = None
        pair["transition_state_guess_source"] = transition_state_guess_source
        pair["transition_state_guess_source_image_index"] = (
            transition_state_guess_source_image_index
        )
        pair["transition_state_guess_selection_rule_version"] = (
            transition_state_guess_selection_rule_version
        )
        safe_path_provenance = _normalize_pipeline_provenance(
            reaction_path.get("provenance"),
            selection_rule_version=transition_state_guess_selection_rule_version,
        )
        if safe_path_provenance is None:
            reaction_path.pop("provenance", None)
        else:
            # 来源为 null 时选择规则也必须为 null，避免旧结果留下不一致元数据。
            safe_path_provenance["selection_rule_version"] = (
                transition_state_guess_selection_rule_version
            )
            reaction_path["provenance"] = safe_path_provenance
        pair["provenance"] = copy.deepcopy(safe_path_provenance)
        reaction_path_path = write_json(
            directory,
            f"endpoint_pair_{pair_index:04d}_reaction_path.json",
            reaction_path,
        )
        artifact_files.append(
            {
                "path": relative_workspace_path(reaction_path_path),
                "semantic_type": "ReactionPath",
                "media_type": "application/json",
                "result_pointer": f"/endpoint_pairs/{pair_index}/reaction_path_artifact",
            }
        )
        transition_state_artifact = _materialize_transition_state_guess_xyz_artifact(
            reaction_path,
            directory=directory,
            pair_index=pair_index,
        )
        if transition_state_artifact is not None:
            artifact_files.append(transition_state_artifact)

        for image_index, image_source, source_kind in _reaction_path_image_sources(
            reaction_path
        ):
            if source_kind == "structure":
                image_path = write_xyz(
                    image_source,
                    directory
                    / f"endpoint_pair_{pair_index:04d}_image_{image_index:04d}.xyz",
                )
            else:
                image_path = image_source
            candidate_id = f"{pair_id}_image_{image_index:04d}"
            pair["path_image_artifacts"].append(
                {
                    "candidate_id": candidate_id,
                    "image_index": image_index,
                }
            )
            candidate_index = len(path_image_candidates)
            path_image_candidates.append(
                {
                    "candidate_id": candidate_id,
                    "pair_id": pair_id,
                    "image_index": image_index,
                    "xyz_artifact": None,
                }
            )
            artifact_files.append(
                {
                    "path": relative_workspace_path(image_path),
                    "semantic_type": "XYZFile",
                    "media_type": "chemical/x-xyz",
                    "result_pointer": (
                        f"/path_image_candidates/{candidate_index}/xyz_artifact"
                    ),
                }
            )

        if _reaction_path_has_complete_indexed_images(
            reaction_path,
            expected_total_images=(
                reaction_path.get("total_images")
                if isinstance(reaction_path.get("total_images"), int)
                and not isinstance(reaction_path.get("total_images"), bool)
                else None
            ),
        ) and _reaction_path_has_valid_interior_peak(reaction_path):
            images = reaction_path.get("images")
            energies = reaction_path.get("energies_hartree")
            if (
                not isinstance(images, list)
                or not isinstance(energies, list)
                or len(images) != len(energies)
                or len(pair["path_image_artifacts"]) != len(images)
            ):
                raise ValueError(
                    f"TS Pipeline complete path {pair_id} lacks a one-to-one image/energy handoff"
                )

        # 公开结果只保留 ArtifactRef 槽位和身份/状态；完整坐标与路径已经写入
        # 当前 workspace 的受控 Artifact 文件，不再进入模型上下文。
        for sensitive_field in (
            "reactant",
            "product",
            "reactant_uma",
            "product_uma",
            "reaction_path",
            "neb_skipped_reason",
            "combination_id",
            "source_candidate_id",
            "source_mmff_converged",
            "target_mmff_converged",
            "reactant_uma_converged",
            "product_uma_converged",
        ):
            pair.pop(sensitive_field, None)

    result["path_image_candidates"] = path_image_candidates
    if len(path_image_candidates) > _MAX_PATH_IMAGES:
        raise ValueError("Path-image count exceeds the bounded complete-result limit")
    return result, artifact_files


def _pipeline_action_provenance(pairs: Any) -> dict[str, Any]:
    """汇总每个 pair 的安全嵌套 ASH 来源，放入外层 ActionResult provenance。"""

    entries: list[dict[str, Any]] = []
    if isinstance(pairs, list):
        for pair in pairs:
            if not isinstance(pair, Mapping):
                continue
            provenance = _normalize_pipeline_provenance(pair.get("provenance"))
            pair_id = pair.get("pair_id")
            if provenance is None or not isinstance(pair_id, str) or not pair_id.strip():
                continue
            entries.append({"pair_id": pair_id, **provenance})
    return {
        "pipeline_protocol_version": _TRANSITION_STATE_PIPELINE_PROTOCOL_VERSION,
        "nested_stage_provenance": entries,
    }


def run_atomic_workflow(
    request: dict[str, Any],
    *,
    executor: Executor | None = None,
) -> dict[str, Any]:
    """执行有界通用 TS path 工作流并返回脱敏的聚合结果。"""

    if executor is None:
        from . import execute_local

        executor = execute_local
    inputs, method, raw_settings = request_parts(request)
    reaction, main_force_field, target_force_field = _validate_public_inputs(inputs)
    if method:
        raise ValueError("Transition-state path method_spec is server-managed and must be empty")
    settings = _validate_settings(raw_settings)
    settings["main_force_field"] = main_force_field
    settings["target_force_field"] = target_force_field
    resources = dict(request.get("resource_limits") or {})
    gpu_count = resources.get("gpu_count", 0)
    if isinstance(gpu_count, bool) or not isinstance(gpu_count, int) or gpu_count < 1:
        raise ValueError("Transition-state path search requires resource_limits.gpu_count >= 1")

    trace: list[dict[str, Any]] = []
    stage_statuses: dict[str, str] = {}
    mapping_was_partial = False
    try:
        mapped = _call(
            executor,
            trace,
            "map_reaction_atoms",
            "rxnmapper",
            inputs={"reaction": reaction},
            action_settings={"minimum_confidence": 0.0},
        )
        mapping_was_partial = trace[-1]["status"] == "partial_success"
        heavy_changes = _call(
            executor,
            trace,
            "analyze_reaction_changes",
            "internal_reaction_analysis",
            inputs={"mapped_reaction": mapped},
        )
        heavy_analysis_partial = trace[-1]["status"] == "partial_success"
    except StageUnavailable as exc:
        return _structured_required_stage_failure(exc)
    transfer_summary = heavy_changes.get("hydrogen_transfer_summary") or {}
    donors = transfer_summary.get("donors")
    acceptors = transfer_summary.get("acceptors")
    if (
        not isinstance(donors, list)
        or len(donors) != 1
        or not isinstance(acceptors, list)
        or len(acceptors) != 1
        or donors[0].get("hydrogens_lost") != 1
        or acceptors[0].get("hydrogens_gained") != 1
        or transfer_summary.get("balanced") is not True
    ):
        raise ValueError(
            "The current path-search profile requires one balanced single-hydrogen transfer"
        )
    transfer_records = [
        {
            "donor_atom_map": int(donors[0]["atom_map"]),
            "acceptor_atom_map": int(acceptors[0]["atom_map"]),
        }
    ]
    try:
        fully_mapped = _call(
            executor,
            trace,
            "materialize_mapped_hydrogen_transfers",
            "internal_reaction_analysis",
            inputs={
                "mapped_reaction": mapped,
                "hydrogen_transfers": transfer_records,
            },
        )
        full_changes = _call(
            executor,
            trace,
            "analyze_reaction_changes",
            "internal_reaction_analysis",
            inputs={"mapped_reaction": fully_mapped},
        )
    except StageUnavailable as exc:
        return _structured_required_stage_failure(exc)
    if (
        full_changes.get("mapping_complete") is not True
        or full_changes.get("unmapped_hydrogen_count") not in (0, None)
    ):
        raise RuntimeError(
            "Hydrogen materialization did not produce a complete atom mapping"
        )
    stage_statuses["reaction_mapping"] = (
        "recovered" if mapping_was_partial or heavy_analysis_partial else "success"
    )

    source_smiles = fully_mapped.get("reactants")
    if settings["direction"] != "forward" or not isinstance(source_smiles, list):
        raise RuntimeError("Server-managed source direction is invalid")
    if len(source_smiles) < 2:
        raise RuntimeError("Transition-state path search requires at least two reactant components")
    mmff_method = {"force_field": settings["mmff_force_field"]}
    try:
        ensembles = _component_ensembles(
            source_smiles, settings, executor, trace, mmff_method
        )
        stage_statuses["conformer_generation"] = "success"
        combinations = _call(
            executor,
            trace,
            "enumerate_conformer_combinations",
            "internal_reactive_complex",
            inputs={"ensembles": ensembles},
            action_settings={"max_combinations": _MAX_COMBINATIONS},
        )
        if combinations.get("truncated") is True:
            raise RuntimeError(
                "Conformer Cartesian product exceeded the server-managed bound"
            )
        stage_statuses["combination_enumeration"] = "success"
        pairs = _prepare_endpoint_pairs(
            combinations,
            full_changes,
            settings,
            executor,
            trace,
            mmff_method,
        )
        if not pairs:
            raise RuntimeError("No eligible MMFF endpoint pair was generated")
        stage_statuses["complex_sampling"] = "success"
        stage_statuses["mmff_endpoint_preparation"] = (
            "success"
            if all(
                pair.get("source_mmff_converged") is True
                and pair.get("target_mmff_converged") is True
                for pair in pairs
            )
            else "partial"
        )
        before_filter = len(pairs)
        pairs = _cluster_endpoint_pairs(pairs, settings, executor, trace)
    except StageUnavailable as exc:
        return _structured_required_stage_failure(exc)
    stage_statuses["diversity_filter"] = (
        "success" if len(pairs) <= before_filter else "failed"
    )
    failure_records: list[dict[str, Any]] = []
    diagnostic_records: list[dict[str, Any]] = []
    pairs, unavailable_stage = _run_heavy_stages(
        pairs,
        {
            "uma_model": settings["uma_model"],
            "uma_task_name": settings["uma_task_name"],
        },
        settings,
        resources,
        executor,
        trace,
        failure_records=failure_records,
        diagnostic_records=diagnostic_records,
    )
    stage_statuses["uma_endpoint_refinement"] = (
        "partial" if any(pair.get("stage_statuses", {}).get("uma") != "success" for pair in pairs) else "success"
    )
    stage_statuses["ash_neb_candidate_path"] = (
        "partial" if any(pair.get("stage_statuses", {}).get("neb") != "success" for pair in pairs) else "success"
    )
    expected_partial_actions = {
        "map_reaction_atoms",
        "analyze_reaction_changes",
        "build_reactive_complex_ensemble",
    }
    unresolved_partial = any(
        record["status"] == "partial_success"
        and record["action_id"] not in expected_partial_actions
        for record in trace
    )
    failure_summary = list(failure_records)
    recorded_failures = {
        (item.get("pair_id"), item.get("stage"))
        for item in failure_summary
        if isinstance(item, dict)
    }
    for pair in pairs:
        for stage, status in pair.get("stage_statuses", {}).items():
            if status not in {"failed", "skipped"}:
                continue
            failure_key = (pair.get("pair_id"), stage)
            if failure_key in recorded_failures:
                continue
            failure_summary.append(
                {
                    "pair_id": pair["pair_id"],
                    "stage": stage,
                    "status": status,
                    "reason_code": (
                        "pair_stage_failed"
                        if status == "failed"
                        else "pair_stage_skipped"
                    ),
                    "retryable": False,
                }
            )
            recorded_failures.add(failure_key)
    retryable_partial = any(
        item.get("retryable") is True
        for item in failure_summary
        if isinstance(item, dict)
    )
    result = {
        "workflow_id": "transition_state_path_search",
        "main_force_field": settings["main_force_field"],
        "target_force_field": settings["target_force_field"],
        "endpoint_pairs": pairs,
        "endpoint_pair_count": len(pairs),
        "stage_statuses": stage_statuses,
        "failure_summary": failure_summary or None,
        "stage_diagnostics": diagnostic_records or None,
        "transition_states_validated": False,
    }
    result, artifact_files = _materialize_pipeline_artifacts(result)
    warnings = [
        "This Action returns only unvalidated transition-state initial guesses. They may be used "
        "as initial geometries for downstream calculations, but cannot be used as scientific "
        "evidence for transition-state identity, reaction connectivity, or mechanism. Before any "
        "guess is used to support a transition-state or "
        "reaction-channel claim, independently run transition-state optimization; use "
        "Hessian/frequency analysis to confirm exactly one imaginary frequency whose normal mode "
        "matches the intended reaction coordinate, with all remaining frequencies real; and run "
        "forward and reverse (bidirectional) IRC calculations that converge to and are checked "
        "against the "
        "expected reactant and product sides."
    ]
    if unavailable_stage:
        warnings.append(
            "One or more endpoint pairs did not complete a heavy stage; inspect pair stage_statuses."
        )
    if failure_summary:
        warnings.append("Some endpoint pairs were unavailable or unconverged.")
    if unresolved_partial:
        warnings.append("One or more non-recoverable atomic stages returned partial_success.")
    for diagnostic in [*diagnostic_records, *failure_summary]:
        if not isinstance(diagnostic, Mapping):
            continue
        message = diagnostic.get("message")
        if isinstance(message, str) and message.strip():
            warnings.append(f"Nested {diagnostic.get('stage', 'stage')} message: {message}")
        diagnostic_warnings = diagnostic.get("warnings")
        if isinstance(diagnostic_warnings, list):
            warnings.extend(
                warning
                for warning in diagnostic_warnings
                if isinstance(warning, str) and warning.strip()
            )
    if unavailable_stage or failure_summary or unresolved_partial:
        response = partial_success(
            result,
            artifact_files=artifact_files,
            warnings=warnings,
            provenance=_pipeline_action_provenance(pairs),
        )
        # 通用 partial_success 默认不可重试；复合 TS path Action 还需把嵌套阶段的
        # 安全重试语义传递到 ActionResult 顶层，供上层调度器判断是否重试。
        response["retryable"] = retryable_partial
        return response
    return success(
        result,
        artifact_files=artifact_files,
        warnings=warnings,
        provenance=_pipeline_action_provenance(pairs),
    )


def execute(action_id: str, backend_id: str, request: dict[str, Any]) -> dict[str, Any]:
    """执行公开 TS path Action；其他组合明确返回不支持。"""

    if action_id == ACTION_ID and backend_id == BACKEND_ID:
        return run_atomic_workflow(request)
    return unsupported(
        f"Unsupported transition-state path action/backend: {action_id}/{backend_id}"
    )
